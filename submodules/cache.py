"""Versioned frozen SigLIP features; a manifest is published only after completion."""
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from submodules.contracts import resolve_path
from .runtime import make_loader, autocast

logger = logging.getLogger(__name__)
CACHE_VERSION = 'siglip-mean-patch-v2'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def cache_identity(dataset, config, split):
    model_path = resolve_path(config['siglip_path'])
    weights = {p.name:sha256(p) for p in sorted(model_path.glob('*.safetensors'))}
    if not weights:
        raise FileNotFoundError(f'No safetensors in {model_path}')
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    sources = []
    for ds in base.datasets:
        files = [*ds.parquet_files]
        files.extend(ds.video_path(cam, ep) for ep in ds.episode_ids for cam in ds.cameras)
        sources.append({'dataset':ds.dataset_path.name, 'contract':ds.contract,
            'indices_sha256':hashlib.sha256(ds.index_mapping.tobytes()).hexdigest(),
            'targets_sha256':hashlib.sha256(b''.join(np.asarray(ds.returns_data[ep]['return'], dtype=np.float32).tobytes() for ep in ds.episode_ids)).hexdigest(),
            'transform':repr(ds.image_transform),
            'files':[(str(p.relative_to(ds.dataset_path)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]})
    identity = {'version':CACHE_VERSION, 'split':split, 'weights':weights,
                'processor':sha256(model_path/'preprocessor_config.json'),
                'model_config':sha256(model_path/'config.json'),
                'cameras':list(config['cameras']), 'precision':config['precision'],
                'torch':torch.__version__, 'count':len(dataset), 'sources':sources,
                'subset_indices':dataset.indices if isinstance(dataset, Subset) else None}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return digest, identity


class FeatureDataset(Dataset):
    def __init__(self, path, identity_digest=None):
        path = Path(path)
        self.manifest = json.loads((path/'manifest.json').read_text())
        if identity_digest is not None and self.manifest['digest'] != identity_digest:
            raise ValueError('Feature cache identity mismatch')
        for name in ('features.npy', 'targets.npy'):
            if sha256(path/name) != self.manifest['files_sha256'][name]:
                raise ValueError(f'Corrupted feature cache: {path/name}')
        # About 0.6 GiB for all training frames and one camera; fits this machine.
        self.features = torch.from_numpy(np.load(path/'features.npy'))
        self.targets = torch.from_numpy(np.load(path/'targets.npy'))
        if self.features.shape != (self.manifest['count'], self.manifest['feature_dim']) or self.targets.shape != (self.manifest['count'],):
            raise ValueError('Feature cache shape mismatch')
        if self.features.dtype != torch.float16 or self.targets.dtype != torch.float32:
            raise ValueError('Feature cache dtype mismatch')
        self.path = path

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {'features':self.features[index], 'target_values':self.targets[index]}


def cache_location(dataset, config, split):
    digest, identity = cache_identity(dataset, config, split)
    return resolve_path(config['cache_dir']) / f'{split}-{digest[:20]}', digest, identity


def prepare_features(dataset, model, config, split, device):
    if not config['freeze_vlm'] or not model.freeze_vlm:
        raise ValueError('Cannot cache a trainable encoder')
    path, digest, identity = cache_location(dataset, config, split)
    if (path/'manifest.json').exists():
        logger.info('Reusing %s', path)
        return FeatureDataset(path, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    loader = make_loader(dataset, config, split, batch_size=config['cache_batch_size'],
                         workers=config['cache_num_workers'], sequential=True)
    model.eval()
    started = last_log = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='.building-', dir=path.parent) as temporary:
        staging = Path(temporary)
        features = np.lib.format.open_memmap(staging/'features.npy', mode='w+', dtype=np.float16,
                                            shape=(len(dataset), model.feature_dim))
        targets = np.lib.format.open_memmap(staging/'targets.npy', mode='w+', dtype=np.float32, shape=(len(dataset),))
        offset = 0
        for batch in loader:
            images = {k:v.to(device, non_blocking=True) for k,v in batch['images'].items()}
            with torch.no_grad(), autocast(config, device):
                encoded = model.encode(images)
            values = encoded.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Non-finite cached features')
            n = len(values)
            features[offset:offset+n] = values
            targets[offset:offset+n] = batch['target_values'].numpy()
            offset += n
            now = time.perf_counter()
            if now-last_log >= 15 or offset == len(dataset):
                rate = offset/(now-started)
                logger.info('Cache %s: %d/%d, %.1f frames/s, ETA %.1f min', split, offset, len(dataset), rate, (len(dataset)-offset)/max(rate,1)/60)
                last_log = now
        if offset != len(dataset):
            raise ValueError('Incomplete feature extraction')
        features.flush(); targets.flush(); del features, targets
        manifest = {'digest':digest, 'identity':identity, 'count':offset, 'feature_dim':model.feature_dim,
                    'seconds':time.perf_counter()-started,
                    'files_sha256':{name:sha256(staging/name) for name in ('features.npy','targets.npy')}}
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        # No partial cache is visible to readers. Avoid overwriting concurrent work.
        if path.exists():
            raise FileExistsError(f'Cache already exists: {path}; retry to reuse it')
        staging.rename(path)
    del loader
    return FeatureDataset(path, digest)
