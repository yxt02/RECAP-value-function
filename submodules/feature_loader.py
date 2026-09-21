"""Resident feature batches: avoid per-frame Python collation after encoding."""
import logging
import math
from types import SimpleNamespace
import torch

logger = logging.getLogger(__name__)


class DeviceFeatureLoader:
    """Small frozen features fit in GPU memory; shuffle only training examples.

    batch_sampler.limit is consumed before iteration, including a partial final epoch.
    Unlike an image DataLoader this has no worker processes or prefetch queue.
    """
    def __init__(self, dataset, batch_size, shuffle, seed, device, max_batches=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.features = dataset.features.to(self.device)
        self.targets = dataset.targets.to(self.device)
        self.batch_sampler = SimpleNamespace(limit=max_batches or math.ceil(len(dataset)/batch_size))

    def __len__(self):
        return min(math.ceil(len(self.dataset)/self.batch_size), self.batch_sampler.limit)

    def __iter__(self):
        indices = (torch.randperm(len(self.dataset), generator=self.generator, device=self.device)
                   if self.shuffle else None)
        for step in range(len(self)):
            start = step*self.batch_size; end = min(start+self.batch_size, len(self.dataset))
            selection = indices[start:end] if indices is not None else slice(start,end)
            yield {'features':self.features[selection], 'target_values':self.targets[selection]}


def resident_loader(dataset, config, split, batch_size, max_batches):
    if not config.get('cache_on_device',False) or not torch.cuda.is_available():
        return None
    needed = sum(t.numel()*t.element_size() for t in (dataset.features,dataset.targets))
    free, _ = torch.cuda.mem_get_info()
    # Leave substantial memory for the head/optimizer/desktop and other applications.
    if needed > free*.5 or free-needed < 2*1024**3:
        logger.info('Insufficient free GPU memory for resident %s features; using CPU batches', split)
        return None
    logger.info('Keeping %s features on GPU: %.1f MiB', split, needed/1024**2)
    return DeviceFeatureLoader(dataset,batch_size,split=='train',int(config.get('seed',42)),
                               'cuda',max_batches)
