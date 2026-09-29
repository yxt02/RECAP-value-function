"""Deterministic local runtime, episode manifests and bounded DataLoader workers."""
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset, BatchSampler, RandomSampler, SequentialSampler

from submodules.contracts import resolve_path, read_split
from submodules.datasets import SimpleValueDataset


def configure_runtime(config):
    seed = int(config.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('torch_threads', 4)))
    cv2.setNumThreads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def init_worker(worker_id):
    torch.set_num_threads(1); cv2.setNumThreads(1)
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def load_config(path):
    config = OmegaConf.to_container(OmegaConf.load(resolve_path(path)), resolve=True)
    if config.get('precision') not in ('bf16', 'fp32'):
        raise ValueError('precision must be bf16 or fp32')
    if config.get('feature_cache') and not config['freeze_vlm']:
        raise ValueError('Feature cache requires freeze_vlm=true (no encoder updates/augmentation)')
    return config


def train_return_scale(config):
    """Use only training episodes to define the shared |return_min| scale."""
    source = OmegaConf.load(resolve_path(config['adaptation_config']))
    split_file = resolve_path(source.output_dir) / 'train.json'
    minimum = 0.0
    found = False
    for spec in source.datasets:
        selected = read_split(split_file, spec.name)
        if not selected:
            continue
        path = resolve_path(source.data_dir) / spec.name / 'meta' / f'returns_{source.tag}.parquet'
        table = pq.read_table(path, columns=['episode_index', 'return']).to_pandas()
        values = table.loc[table.episode_index.isin(selected), 'return'].to_numpy(dtype=np.float64)
        if not len(values) or not np.isfinite(values).all() or np.max(values) > 0:
            raise ValueError(f'Invalid training returns in {path}')
        minimum = min(minimum, float(np.min(values)))
        found = True
    if not found or minimum >= 0:
        raise ValueError('No negative training return available for return_min normalization')
    return -minimum


def raw_dataset(config, split, *, training=False):
    if split not in ('train', 'val', 'test'):
        raise ValueError(split)
    source = OmegaConf.load(resolve_path(config['adaptation_config']))
    return_mode = config.get('value_normalization', 'contract_scale')
    if return_mode not in ('contract_scale', 'train_return_min'):
        raise ValueError(f'Unknown value_normalization: {return_mode}')
    value_scale = (float(config.get('value_scale') or train_return_scale(config))
                   if return_mode == 'train_return_min' else None)
    split_file = resolve_path(source.output_dir) / f'{split}.json'
    datasets = []
    for spec in source.datasets:
        if not read_split(split_file, spec.name):
            continue
        ds = SimpleValueDataset(resolve_path(source.data_dir) / spec.name,
            tag=source.tag, split_file=split_file, cameras=config['cameras'],
            include_state=False, include_actions=False,
            return_scale=value_scale, clip_normalized_returns=return_mode == 'train_return_min',
            augment_images=training and split == 'train' and bool(config.get('train_image_augmentation', False)),
            image_processor_path=resolve_path(config['siglip_path']),
            cache_episodes=4, cache_videos=int(config.get('video_cache_handles', 4)))
        for key in ('tag', 'gamma', 'failure_reward', 'return_scale', 'max_episode_steps'):
            if ds.contract[key] != source[key]:
                raise ValueError(f'Stale return contract in {spec.name}: {key}')
        datasets.append(ds)
    if not datasets:
        raise ValueError(f'Empty {split} split')
    result = ConcatDataset(datasets)
    limit = config.get('max_samples')
    if limit is not None and int(limit) < len(result):
        if int(limit) < 1:
            raise ValueError('max_samples must be positive')
        # Uniformly cover the split rather than only the first episode.
        result = Subset(result, np.linspace(0, len(result)-1, int(limit), dtype=int).tolist())
    return result


class CappedBatchSampler:
    def __init__(self, batches, limit):
        self.batches, self.limit = batches, int(limit)
    def __len__(self):
        return min(len(self.batches), self.limit)
    def __iter__(self):
        from itertools import islice
        return islice(iter(self.batches), self.limit)


def make_loader(dataset, config, split, *, batch_size=None, workers=None, sequential=False,
                max_batches=None, sampler=None):
    if hasattr(dataset, 'features') and hasattr(dataset, 'targets'):
        from .feature_loader import resident_loader
        resident = resident_loader(dataset, config, split, int(batch_size or config['batch_size']), max_batches)
        if resident is not None:
            return resident
    if workers is None:
        workers = int(config['num_workers'] if split == 'train' else config['eval_num_workers'])
    generator = torch.Generator().manual_seed(int(config.get('seed', 42)))
    kwargs = dict(batch_size=int(batch_size or config['batch_size']),
                  shuffle=split=='train' and not sequential and sampler is None,
                  num_workers=workers, pin_memory=bool(config['pin_memory'] and torch.cuda.is_available()),
                  generator=generator)
    if sampler is not None:
        kwargs['sampler'] = sampler
    if max_batches is not None:
        if int(max_batches) < 1:
            raise ValueError('max_batches must be positive')
        source_sampler = kwargs.pop('sampler', None)
        if source_sampler is None:
            source_sampler = (RandomSampler(dataset, generator=generator)
                              if kwargs.pop('shuffle') else SequentialSampler(dataset))
        kwargs.pop('shuffle', None)
        batches = BatchSampler(source_sampler, kwargs.pop('batch_size'), drop_last=False)
        kwargs['batch_sampler'] = CappedBatchSampler(batches, max_batches)
    if workers:
        kwargs.update(worker_init_fn=init_worker, multiprocessing_context='spawn',
                      prefetch_factor=int(config['prefetch_factor']),
                      persistent_workers=bool(config['persistent_workers']))
    return DataLoader(dataset, **kwargs)


def autocast(config, device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type=='cuda' and config['precision']=='bf16')


def move_batch(batch, device):
    result = {'target_values': batch['target_values'].to(device, dtype=torch.float32, non_blocking=True).reshape(-1, 1)}
    if 'features' in batch:
        result['features'] = batch['features'].to(device, non_blocking=True)
    else:
        result['images'] = {k:v.to(device, non_blocking=True) for k,v in batch['images'].items()}
    return result
