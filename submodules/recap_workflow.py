"""Local training and inference path for the patch-token RECAP value model."""
import json
import logging
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from submodules.cache import sha256
from submodules.contracts import resolve_path
from submodules.recap_model import ARCHITECTURE, RecapValueModel, two_hot_loss
from submodules.runtime import configure_runtime, make_loader, raw_dataset, train_return_scale

logger = logging.getLogger(__name__)


class ExactRankSampler(Sampler):
    """Partition validation without DistributedSampler's duplicate padding."""
    def __init__(self, dataset, rank, world_size):
        self.length, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.length, self.world_size))


def distributed_setup():
    """Use torchrun's process environment; ordinary python remains single-GPU."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world_size == 1:
        return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = int(os.environ['LOCAL_RANK'])
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError('Multi-GPU RECAP training requires one visible CUDA device per local rank')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world_size, torch.device('cuda', local_rank)


def optimizer_groups(model, config):
    """Give the vision tower, Gemma3, and expert/bridges independent rates."""
    groups = []
    choices = (
        ('vision', lambda name: name.startswith('vision_tower.'), 'vision_lr'),
        ('gemma3', lambda name: name.startswith('gemma3.'), 'gemma_lr'),
        ('expert', lambda name: not name.startswith(('vision_tower.', 'gemma3.')), 'expert_lr'),
    )
    named = list(model.named_parameters())
    for name, select, rate_key in choices:
        params = [parameter for key, parameter in named if parameter.requires_grad and select(key)]
        if params:
            rate = float(config[rate_key])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError(f'{rate_key} must be positive and finite')
            groups.append(dict(name=name, params=params, lr=rate))
    if not groups:
        raise ValueError('No trainable value-model parameters')
    return groups


def run_recap_epoch(model, loader, config, device, optimizer=None, scheduler=None,
                    remaining_steps=None):
    """Accumulate sample-weighted gradients; count optimizer steps, not batches."""
    training = optimizer is not None
    model.train(training)
    if training and scheduler is None:
        raise ValueError('Training requires a scheduler')
    accumulation = int(config.get('gradient_accumulation_steps', 1))
    if accumulation < 1:
        raise ValueError('gradient_accumulation_steps must be positive')
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = total_mse = 0.0
    samples = microbatches = steps = window_samples = window_batches = 0
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    value_model = model.module if isinstance(model, DistributedDataParallel) else model
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            images = {key: value.to(device, non_blocking=True) for key, value in batch['images'].items()}
            targets = batch['target_values'].to(device, dtype=torch.float32, non_blocking=True).reshape(-1)
            enabled = device.type == 'cuda' and config['precision'] == 'bf16'
            update_now = training and (window_batches + 1 == accumulation or batch_index + 1 == len(loader))
            # static_graph and no_sync can conflict on the first backward with this
            # model's gradient checkpointing. Synchronize each microbatch for DDP.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
                values, logits = model(images, batch['prompt'], return_distribution=True,
                                       image_masks=batch['image_masks'])
                loss = two_hot_loss(logits, targets, value_model.atoms)
                mse = torch.nn.functional.mse_loss(values.float().reshape(-1), targets)
            if not torch.isfinite(loss) or not torch.isfinite(mse):
                raise FloatingPointError('Non-finite RECAP value loss')
            if training:
                (loss * len(targets)).backward()
            count = len(targets)
            total_loss += float(loss.detach()) * count
            total_mse += float(mse.detach()) * count
            samples += count
            microbatches += 1
            if not training:
                continue
            window_samples += count
            window_batches += 1
            if update_now:
                for parameter in params:
                    if parameter.grad is not None:
                        parameter.grad.div_(window_samples)
                torch.nn.utils.clip_grad_norm_(params, float(config.get('clip_grad_norm', 1.0)),
                                               error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                window_samples = window_batches = 0
                if remaining_steps is not None and steps >= remaining_steps:
                    break
    if not samples or (training and not steps):
        raise ValueError('Empty RECAP training or validation epoch')
    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor([total_loss, total_mse, samples, microbatches],
                              dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_mse, samples, microbatches = totals.tolist()
        samples, microbatches = int(samples), int(microbatches)
    return dict(ce=total_loss / samples, mse=total_mse / samples,
                samples=samples, microbatches=microbatches, steps=steps)


def data_source_hashes(config):
    """Bind the checkpoint to the adapted returns and configured data contract."""
    from omegaconf import OmegaConf

    adaptation = resolve_path(config['adaptation_config'])
    spec = OmegaConf.load(adaptation)
    sources = {'adaptation_config': adaptation}
    for entry in spec.datasets:
        root = resolve_path(spec.data_dir) / entry.name / 'meta'
        label = f'returns_{spec.tag}'
        for suffix in ('.json', '.parquet'):
            sources[f'{entry.name}/{label}{suffix}'] = root / f'{label}{suffix}'
    return {name: sha256(path) for name, path in sources.items()}


def split_hashes(config):
    from omegaconf import OmegaConf

    spec = OmegaConf.load(resolve_path(config['adaptation_config']))
    root = resolve_path(spec.output_dir)
    return {name: sha256(root / f'{name}.json') for name in ('train', 'val', 'test')}


def train_recap(config, smoke_test=False):
    cfg = dict(config)
    if cfg.get('value_normalization') == 'train_return_min':
        cfg['value_scale'] = train_return_scale(cfg)
        cfg['return_min_train'] = -cfg['value_scale']
        logger.info('Train-only return_min %.1f; value scale %.1f',
                    cfg['return_min_train'], cfg['value_scale'])
    if cfg.get('feature_cache'):
        raise ValueError('A trainable RECAP backbone requires online images; disable feature_cache')
    if smoke_test:
        cfg.update(max_samples=8, num_epochs=1, max_steps=4, val_steps=1,
                   max_total_steps=2, batch_size=1, gradient_accumulation_steps=2,
                   num_workers=0,
                   eval_num_workers=0, warmup_steps=0,
                   save_dir='artifacts/performance/smoke-recap')
    for key in ('batch_size', 'num_epochs', 'gradient_accumulation_steps'):
        if int(cfg[key]) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('max_steps', 'val_steps', 'max_total_steps'):
        if cfg.get(key) is not None and int(cfg[key]) < 1:
            raise ValueError(f'{key} must be positive or null')
    rank, world_size, device = distributed_setup()
    configure_runtime({**cfg, 'seed': int(cfg['seed']) + rank})
    if device.type == 'cpu':
        cfg['precision'] = 'fp32'
    elif cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA device does not support BF16')
    train_data = val_data = None
    try:
        train_data, val_data = raw_dataset(cfg, 'train', training=True), raw_dataset(cfg, 'val')
        model = RecapValueModel(cfg).to(device)
        if cfg.get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        training_model = (DistributedDataParallel(model, device_ids=[device.index],
                                                  broadcast_buffers=False, static_graph=True)
                          if world_size > 1 else model)
        optimizer = torch.optim.AdamW(optimizer_groups(model, cfg),
                                      weight_decay=float(cfg.get('weight_decay', 0.0)),
                                      fused=device.type == 'cuda')
        warmup = int(cfg.get('warmup_steps', 0))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)
        train_sampler = (DistributedSampler(train_data, num_replicas=world_size, rank=rank,
                                            shuffle=True, seed=int(cfg['seed']))
                         if world_size > 1 else None)
        val_sampler = ExactRankSampler(val_data, rank, world_size) if world_size > 1 else None
        loader_cfg = {**cfg, 'seed': int(cfg['seed']) + rank}
        train_loader = make_loader(train_data, loader_cfg, 'train', batch_size=cfg['batch_size'],
                                   max_batches=cfg.get('max_steps'), sampler=train_sampler)
        val_loader = make_loader(val_data, loader_cfg, 'val', batch_size=cfg['batch_size'],
                                 max_batches=cfg.get('val_steps'), sampler=val_sampler)
        save_dir = resolve_path(cfg['save_dir'])
        if rank == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()
        sources = {
            'vision_weights': resolve_path(cfg['siglip_path']) / 'model.safetensors',
            'vision_processor': resolve_path(cfg['siglip_path']) / 'preprocessor_config.json',
            'gemma_weights': resolve_path(cfg['gemma3_path']) / 'model.safetensors',
            'tokenizer': resolve_path(cfg.get('tokenizer_path', cfg['gemma3_path'])) / 'tokenizer.json',
        }
        source_hashes = {name: sha256(path) for name, path in sources.items()} if rank == 0 else None
        data_hashes = data_source_hashes(cfg) if rank == 0 else None
        best, stale, global_step, history = math.inf, 0, 0, []
        max_total = int(cfg.get('max_total_steps') or 2**31-1)
        for epoch in range(int(cfg['num_epochs'])):
            if global_step >= max_total:
                break
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            training = run_recap_epoch(training_model, train_loader, cfg, device, optimizer,
                                       scheduler, remaining_steps=max_total - global_step)
            global_step += training['steps']
            validation = run_recap_epoch(model, val_loader, cfg, device)
            val_ce = validation['ce']
            history.append(dict(epoch=epoch + 1, global_step=global_step,
                                train=training, val=validation))
            if rank == 0:
                logger.info('RECAP epoch %d step %d train CE %.5f val CE %.5f',
                            epoch + 1, global_step, training['ce'], val_ce)
            if val_ce < best - float(cfg.get('early_stopping_min_delta', 0.0)):
                best, stale = val_ce, 0
                if rank == 0:
                    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                             if not (model.freeze_vision_encoder and k.startswith('vision_tower.'))
                             and not (model.freeze_vlm and k.startswith('gemma3.'))}
                    payload = dict(format_version=3, architecture=ARCHITECTURE,
                                   config=cfg, epoch=epoch + 1, global_step=global_step,
                                   val_loss=val_ce, model_state_dict=state,
                                   smoke_test=bool(smoke_test), train_samples=len(train_data),
                                   val_samples=len(val_data), world_size=world_size,
                                   model_sources_sha256=source_hashes,
                                   data_sources_sha256=data_hashes,
                                   split_sha256=split_hashes(cfg))
                    temp = save_dir / 'best_model.pt.tmp'
                    torch.save(payload, temp)
                    temp.replace(save_dir / 'best_model.pt')
            else:
                stale += 1
            if rank == 0:
                (save_dir / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')
            if world_size > 1:
                dist.barrier()
            if stale >= int(cfg.get('early_stopping_patience', 4)) or global_step >= max_total:
                break
        return history
    finally:
        for data in (train_data, val_data):
            if data is not None:
                for ds in (data.dataset if hasattr(data, 'dataset') else data).datasets:
                    ds.close()
        if world_size > 1:
            dist.destroy_process_group()


def load_recap_checkpoint(payload, device):
    if payload.get('architecture') != ARCHITECTURE or payload.get('format_version') != 3:
        raise ValueError('Not a RECAP patch checkpoint')
    cfg = payload['config']
    if cfg.get('value_normalization') == 'train_return_min' and not math.isclose(
            float(cfg['value_scale']), train_return_scale(cfg), rel_tol=0, abs_tol=1e-6):
        raise ValueError('RECAP checkpoint training return_min changed')
    sources = {
        'vision_weights': resolve_path(cfg['siglip_path']) / 'model.safetensors',
        'vision_processor': resolve_path(cfg['siglip_path']) / 'preprocessor_config.json',
        'gemma_weights': resolve_path(cfg['gemma3_path']) / 'model.safetensors',
        'tokenizer': resolve_path(cfg.get('tokenizer_path', cfg['gemma3_path'])) / 'tokenizer.json',
    }
    expected = payload.get('model_sources_sha256')
    if expected is not None and expected != {name: sha256(path) for name, path in sources.items()}:
        raise ValueError('RECAP checkpoint pretrained model or tokenizer identity changed')
    if payload.get('data_sources_sha256') is not None and payload['data_sources_sha256'] != data_source_hashes(cfg):
        raise ValueError('RECAP checkpoint return labels or data contract changed')
    if payload.get('split_sha256') is not None and payload['split_sha256'] != split_hashes(cfg):
        raise ValueError('RECAP checkpoint data splits changed')
    model = RecapValueModel(cfg).to(device)
    loaded = model.load_state_dict(payload['model_state_dict'], strict=False)
    allowed = set()
    if model.freeze_vision_encoder:
        allowed.add('vision_tower.')
    if model.freeze_vlm:
        allowed.add('gemma3.')
    if loaded.unexpected_keys or any(not any(k.startswith(prefix) for prefix in allowed)
                                     for k in loaded.missing_keys):
        raise ValueError(f'Checkpoint state mismatch: {loaded}')
    return model.eval()


def predict_recap_dataset(model, dataset, config, device, batch_size=None):
    # Keep one fixed numerical path for value differences across adjacent frames.
    # BF16 attention can shift predictions slightly when the inference batch changes.
    loader = make_loader(dataset, {**config, 'persistent_workers': False}, 'test',
                         batch_size=1 if batch_size is None else batch_size, workers=0,
                         sequential=True)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            images = {k: v.to(device) for k, v in batch['images'].items()}
            predictions.append(model(images, batch['prompt'],
                                     image_masks=batch['image_masks']).float().cpu().numpy().ravel())
    result = np.concatenate(predictions)
    if len(result) != len(dataset) or not np.isfinite(result).all():
        raise ValueError('Invalid RECAP predictions')
    return result
