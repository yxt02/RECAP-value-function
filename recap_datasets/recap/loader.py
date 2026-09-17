"""Actual local config-to-DataLoader wiring (no RLinf installation required)."""
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from .contracts import resolve_path, read_split
from .simple_dataset import SimpleValueDataset
from .common import ReCapMixtureDataset


def create_value_dataloader(config_path='config/recap_value_model_sft_z02.yaml',
                            split='train', batch_size=None, num_workers=None, epoch=0):
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'Unknown split: {split}')
    cfg = OmegaConf.load(resolve_path(config_path))
    data = cfg.data
    source = OmegaConf.load(resolve_path(data.adaptation_config))
    split_file = resolve_path(source.output_dir) / f'{split}.json'
    datasets, weights = [], []
    contracts = set()
    for spec in source.datasets:
        if not read_split(split_file, spec.name):
            continue
        ds = SimpleValueDataset(
            resolve_path(source.data_dir) / spec.name, tag=source.tag,
            cameras=list(source.cameras), split_file=split_file,
            include_state=data.include_state, include_actions=data.include_actions,
            action_dim=data.action_dim, action_horizon=data.action_horizon,
            image_processor_path=resolve_path(data.image_processor_path),
        )
        contracts.add(tuple(ds.contract[k] for k in ('tag', 'gamma', 'failure_reward', 'return_scale', 'max_episode_steps')))
        datasets.append(ds)
        weights.append(float(data.get('dataset_weights', {}).get(spec.name, len(ds))))
    expected = tuple(source[k] for k in ('tag', 'gamma', 'failure_reward', 'return_scale', 'max_episode_steps'))
    if contracts != {expected}:
        raise ValueError('Missing/stale return contracts; regenerate all datasets with the current adaptation config')
    mixture = ReCapMixtureDataset(datasets, weights=weights, seed=source.random_seed)
    sampler = mixture.make_sampler(epoch=epoch) if split == 'train' and data.balance_weights else None
    workers = num_workers if num_workers is not None else int(data.train_num_workers if split == 'train' else data.eval_num_workers)
    kwargs = dict(batch_size=batch_size or cfg.actor.micro_batch_size, num_workers=workers,
                  shuffle=split == 'train' and sampler is None, sampler=sampler,
                  pin_memory=bool(data.pin_memory))
    if workers > 0:
        kwargs.update(prefetch_factor=int(data.prefetch_factor), persistent_workers=bool(data.persistent_workers))
    return DataLoader(mixture, **kwargs)
