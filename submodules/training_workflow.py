#!/usr/bin/env python3
"""Training workflows for the frozen-encoder scalar value baseline.

Subcommands:
  train       - fit the regression head; --prepare-cache only builds feature caches
  benchmark   - measure GPU, data loader or cached-head throughput
  check-cache - verify cache/online agreement and checkpoint reload

This is not the language-conditioned 201-bin RECAP critic. See docs/scripts.md.
"""
import argparse
import gc
import json
import logging
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Before numpy/torch: submodules sets single-threaded BLAS, which this environment
# needs for the heavy imports below not to crash. See docs/scripts.md.
import submodules  # noqa: F401

import torch
from torch import nn

from submodules.contracts import resolve_path
from submodules.cache import FeatureDataset, cache_location, prepare_features
from submodules.model import ValueModel
from submodules.runtime import autocast, configure_runtime, load_config, make_loader, move_batch, raw_dataset

logger = logging.getLogger(__name__)

# Tunable parameters. Command line flags override the YAML config; relative paths
# resolve against PROJECT_ROOT.
DEFAULT_CONFIG = 'config/train_value.yaml'
BENCHMARK_DIR = 'artifacts/performance'
CACHE_REPORT = 'artifacts/performance/cache-verification.json'

# Each entry adds a `--<name>` flag overriding the same key in the training config.
OVERRIDABLE = {'batch_size': int, 'num_epochs': int, 'lr': float, 'warmup_steps': int,
               'max_total_steps': int, 'early_stopping_patience': int, 'max_samples': int,
               'max_steps': int, 'val_steps': int, 'num_workers': int,
               'cache_batch_size': int, 'cached_batch_size': int, 'save_dir': str,
               'num_bins': int}

# --smoke_test: a few real batches, a bounded sample budget, throwaway output.
SMOKE_TEST = dict(num_epochs=1, max_steps=4, val_steps=2)
SMOKE_SAMPLES = 256


def targets_to_bin_indices(targets, num_bins, low=-1.0, high=0.0):
    """Convert continuous targets in [low, high] to bin indices in [0, num_bins-1].

    Args:
        targets: Tensor of shape [B, 1] or [B].
        num_bins: Number of bins.
        low: Lower bound of the value range.
        high: Upper bound of the value range.

    Returns:
        Long tensor of bin indices, shape [B].
    """
    targets = targets.clamp(low, high)
    # Map [low, high] -> [0, 1] -> [0, num_bins-1]
    normalized = (targets - low) / (high - low)
    indices = (normalized * (num_bins - 1)).round().long().squeeze(-1)
    return indices.clamp(0, num_bins - 1)


def epoch(model, loader, config, device, optimizer=None, scheduler=None, max_steps=None):
    """Run one training or validation epoch; returns loss/sample/step metrics."""
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    mse_sum = torch.zeros((), device=device)
    count = steps = 0
    start = time.perf_counter()
    num_bins = config.get('num_bins', 201)
    # The cap must come from make_loader; otherwise workers keep decoding prefetched batches.
    limit = min(len(loader), int(max_steps)) if max_steps is not None else len(loader)
    if limit != len(loader):
        raise ValueError('Set max_batches on make_loader so workers are not left decoding prefetched batches')
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(config, device):
                targets = batch['target_values']
                # Get distribution logits and compute expectation
                _, log_probs = model(images=batch.get('images'), features=batch.get('features'),
                                     return_distribution=True)
                # Cross-entropy loss: target bin index vs predicted distribution
                bin_indices = targets_to_bin_indices(targets, num_bins)
                loss = nn.functional.cross_entropy(log_probs, bin_indices)
                # Also compute MSE on expected value for monitoring
                expectation = (log_probs.exp() * model.bin_centers).sum(dim=-1, keepdim=True)
                mse = nn.functional.mse_loss(expectation.float(), targets.float())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         config['clip_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
            n = len(targets)
            count += n
            steps += 1
            loss_sum += loss.detach() * n
            mse_sum += mse.detach() * n
            if training and steps % int(config['log_interval']) == 0:
                logger.info('Step %d/%d, CE %.6f, MSE %.6f, LR %.3g',
                            steps, limit, (loss_sum / count).item(),
                            (mse_sum / count).item(), scheduler.get_last_lr()[0])
    if count == 0:
        raise ValueError('No samples processed')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {'ce': (loss_sum / count).item(), 'mse': (mse_sum / count).item(),
            'samples': count, 'steps': steps,
            'seconds': elapsed, 'samples_per_second': count / elapsed}


def build_scheduler(optimizer, total_steps, warmup_steps):
    """Linear warmup from 1% to 100%, then cosine decay back to 1%."""
    warmup = min(max(0, int(warmup_steps)), max(0, total_steps - 1))

    def multiplier(step):
        if warmup and step < warmup:
            return .01 + .99 * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return .01 + .99 * .5 * (1 + math.cos(math.pi * min(1., progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train(config, smoke_test=False, prepare_cache=False):
    """Train the value head, or build the feature caches and stop."""
    if smoke_test:
        config.update(**SMOKE_TEST,
                      cached_batch_size=min(64, config['cached_batch_size']),
                      max_samples=min(config.get('max_samples') or SMOKE_SAMPLES, SMOKE_SAMPLES),
                      save_dir=f'{BENCHMARK_DIR}/smoke-cached' if config['feature_cache']
                      else f'{BENCHMARK_DIR}/smoke-online')
    configure_runtime(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config['precision'] == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support BF16; select fp32')
    if device.type == 'cpu':
        config['precision'] = 'fp32'
    for key in ('batch_size', 'cached_batch_size', 'cache_batch_size', 'num_epochs', 'log_interval'):
        if int(config[key]) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('max_steps', 'val_steps', 'max_samples', 'max_total_steps'):
        if config[key] is not None and int(config[key]) < 1:
            raise ValueError(f'{key} must be positive or null')
    if config['early_stopping_patience'] < 0 or config['early_stopping_min_delta'] < 0:
        raise ValueError('Early stopping patience and min_delta must be nonnegative')
    if prepare_cache and not config['feature_cache']:
        raise ValueError('--prepare-cache requires feature_cache=true')
    logger.info('Device %s, configuration %s', device, json.dumps(config))

    train_data, val_data = raw_dataset(config, 'train'), raw_dataset(config, 'val')
    logger.info('Manifest-filtered samples: train=%d val=%d', len(train_data), len(val_data))

    # Skip loading the encoder when both caches are already published.
    cached = config['feature_cache'] and all(
        (cache_location(d, config, s)[0] / 'manifest.json').exists()
        for d, s in [(train_data, 'train'), (val_data, 'val')])
    num_bins = config.get('num_bins', 201)
    model = ValueModel(str(resolve_path(config['siglip_path'])), config['cameras'], config['freeze_vlm'],
                       config['precision'], config['projection_dim'], load_encoder=not cached,
                       num_bins=num_bins).to(device)

    if config['feature_cache']:
        train_data = prepare_features(train_data, model, config, 'train', device)
        val_data = prepare_features(val_data, model, config, 'val', device)
        if prepare_cache:
            logger.info('Feature caches ready; no training performed')
            return
        model.siglip = None  # The frozen encoder is unused by cached epochs and head checkpoints.
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Once at the phase boundary, never inside the training loop.
        train_loader = make_loader(
            train_data, config, 'train', batch_size=config['cached_batch_size'],
            workers=config['cached_num_workers'],
            max_batches=config['max_steps'] or config['max_total_steps'] or sys.maxsize)
        val_loader = make_loader(
            val_data, config, 'val', batch_size=config['cached_batch_size'],
            workers=config['cached_num_workers'], max_batches=config['val_steps'])
    else:
        train_loader = make_loader(
            train_data, config, 'train',
            max_batches=config['max_steps'] or config['max_total_steps'] or sys.maxsize)
        val_loader = make_loader(val_data, config, 'val', max_batches=config['val_steps'])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'],
                                  weight_decay=config['weight_decay'], fused=device.type == 'cuda')
    steps_per_epoch = min(len(train_loader), config['max_steps']) if config['max_steps'] else len(train_loader)
    total_steps = min(config['num_epochs'] * steps_per_epoch, config['max_total_steps'] or sys.maxsize)
    scheduler = build_scheduler(optimizer, total_steps, config['warmup_steps'])
    logger.info('Budget: at most %d epochs / %d optimizer steps; %d batches per full epoch',
                config['num_epochs'], total_steps, steps_per_epoch)

    save_dir = resolve_path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    best = early_best = float('inf')
    history = []
    global_step = stale_epochs = 0
    # Snapshot trainable parameters so --smoke_test can prove the optimizer moved them.
    initial = ({name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
               if smoke_test else None)

    for ep in range(config['num_epochs']):
        remaining = total_steps - global_step
        if remaining <= 0:
            break
        train_loader.batch_sampler.limit = min(steps_per_epoch, remaining)
        training = epoch(model, train_loader, config, device, optimizer, scheduler)
        global_step += training['steps']
        validation = epoch(model, val_loader, config, device, max_steps=config['val_steps'])
        history.append({'epoch': ep + 1, 'global_step': global_step,
                        'train': training, 'val': validation})
        logger.info('%s', json.dumps(history[-1]))

        # Use cross-entropy loss for model selection (primary metric)
        if validation['ce'] < best:
            best = validation['ce']
            # Save the head only; the frozen encoder is identified by config and cache manifest.
            state = {k: v for k, v in model.state_dict().items()
                     if not (config['freeze_vlm'] and k.startswith('siglip.'))}
            tmp = save_dir / 'best_model.pt.tmp'
            torch.save({'format_version': 2, 'architecture': 'siglip_mean_patch_scalar',
                        'epoch': ep + 1, 'global_step': global_step, 'config': config,
                        'model_state_dict': state, 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 'val_loss': best,
                        'feature_cache_manifest': train_data.manifest if config['feature_cache'] else None},
                       tmp)
            tmp.replace(save_dir / 'best_model.pt')
        (save_dir / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')

        if validation['ce'] < early_best - float(config['early_stopping_min_delta']):
            early_best = validation['ce']
            stale_epochs = 0
        else:
            stale_epochs += 1
        if config['early_stopping_patience'] and stale_epochs >= config['early_stopping_patience']:
            logger.info('Early stopping after %d validation checks without sufficient improvement', stale_epochs)
            break

    if smoke_test:
        changed = any(not torch.equal(initial[name], param)
                      for name, param in model.named_parameters() if param.requires_grad)
        if not changed:
            raise AssertionError('Optimizer did not update the head')
        logger.info('Smoke passed: real train/val batches, finite backward gradients and parameter update')
    (save_dir / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')


def main_train(argv=None):
    """CLI for `value train`: parse config overrides and start training."""
    p = argparse.ArgumentParser(description='训练视觉价值基线模型')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='训练配置文件路径')
    p.add_argument('--smoke_test', action='store_true', help='使用小样本冒烟测试配置')
    p.add_argument('--prepare-cache', action='store_true', help='只构建特征缓存，不执行训练')
    p.add_argument('--no-cache', action='store_true', help='从原始图像在线编码训练')
    for key, typ in OVERRIDABLE.items():
        p.add_argument('--' + key, type=typ)
    args = p.parse_args(argv)

    config = load_config(args.config)
    for key, value in vars(args).items():
        if key in OVERRIDABLE and value is not None:
            config[key] = value
    if args.no_cache:
        config['feature_cache'] = False
    train(config, smoke_test=args.smoke_test, prepare_cache=args.prepare_cache)


def timed_gpu(model, config, batch_size, steps=12):
    """Measure CUDA forward/backward throughput for one batch size."""
    steps = max(steps, 1024 // batch_size)
    images = {'observation.images.cam2': torch.rand(batch_size, 3, 224, 224, device='cuda') * 2 - 1}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    target = torch.full((batch_size, 1), -.5, device='cuda')

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast(config, torch.device('cuda')):
            loss = (model(images).float() - target).square().mean()
        loss.backward()
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {'batch_size': batch_size, 'precision': config['precision'], 'steps': steps,
            'samples_per_second': steps * batch_size / elapsed, 'seconds': elapsed,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}


def benchmark_gpu(cfg):
    """Compare the historical fp32 reference against the optimized bf16 head."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'historical_training', PROJECT_ROOT / 'submodules/reference/train_value.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ValueModel(str(PROJECT_ROOT / cfg['siglip_path']),
                              str(PROJECT_ROOT / 'models/gemma-3-270m'), True).cuda()
    results = [{**timed_gpu(model, {**cfg, 'precision': 'fp32'}, 2),
                'implementation': 'historical_fp32'}]
    print(json.dumps(results[-1]), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    num_bins = cfg.get('num_bins', 201)
    model = ValueModel(str(PROJECT_ROOT / cfg['siglip_path']), cfg['cameras'], True, 'bf16',
                       num_bins=num_bins).cuda()
    for size in [8, 16, 32, 64, 128]:
        try:
            results.append({**timed_gpu(model, cfg, size), 'implementation': 'optimized_bf16'})
            print(json.dumps(results[-1]), flush=True)
            if results[-1]['peak_reserved_gib'] > 10.5:
                break
        except torch.cuda.OutOfMemoryError:
            results.append({'batch_size': size, 'oom': True})
            torch.cuda.empty_cache()
            break
    return results


def benchmark_loader(cfg):
    """Sweep data loader worker counts on a bounded sample budget."""
    cfg['max_samples'] = 2048
    ds = raw_dataset(cfg, 'train')
    results = []
    for workers in [0, 2, 4, 8]:
        loader = make_loader(ds, cfg, 'train', batch_size=32, workers=workers, max_batches=28)
        start = time.perf_counter()
        iterator = iter(loader)
        for _ in range(4):  # warm up
            next(iterator)
        setup = time.perf_counter() - start

        start = time.perf_counter()
        count = 0
        for _ in range(24):
            count += len(next(iterator)['target_values'])
        elapsed = time.perf_counter() - start
        for _ in iterator:  # drain
            pass
        results.append({'workers': workers, 'batch_size': 32, 'samples_per_second': count / elapsed,
                        'measured_samples': count, 'startup_and_warmup_seconds': setup})
        print(json.dumps(results[-1]), flush=True)
        del iterator, loader
        gc.collect()
    return results


def benchmark_head(cfg):
    """Sweep batch size and GPU residency for cached-feature head training."""
    ds = raw_dataset(cfg, 'train')
    path, digest, _ = cache_location(ds, cfg, 'train')
    ds = FeatureDataset(path, digest)
    num_bins = cfg.get('num_bins', 201)
    results = []
    for resident in [False, True]:
        cfg['cache_on_device'] = resident
        for size in [256, 512, 1024, 2048]:
            model = ValueModel(str(PROJECT_ROOT / cfg['siglip_path']), cfg['cameras'], True,
                               cfg['precision'], cfg['projection_dim'], load_encoder=False,
                               num_bins=num_bins).cuda()
            loader = make_loader(ds, cfg, 'train', batch_size=size, workers=0)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], fused=True)
            sched = build_scheduler(opt, len(loader) * 3, 0)
            epoch(model, loader, cfg, torch.device('cuda'), opt, sched)  # warm up
            results.append({'batch_size': size, 'gpu_resident': resident, 'samples': len(ds),
                            'trials': [epoch(model, loader, cfg, torch.device('cuda'), opt, sched)
                                       for _ in range(2)]})
            print(json.dumps(results[-1]), flush=True)
            del model, loader, opt, sched
            gc.collect()
            torch.cuda.empty_cache()
    return results


def main_benchmark(argv=None):
    """CLI for `value benchmark`: measure throughput of one pipeline stage."""
    p = argparse.ArgumentParser(description='测量 GPU、数据加载器或缓存头的吞吐量')
    p.add_argument('--mode', choices=['gpu', 'loader', 'head'], required=True, help='测量模式')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='训练配置文件路径')
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    configure_runtime(cfg)
    output = PROJECT_ROOT / BENCHMARK_DIR
    output.mkdir(parents=True, exist_ok=True)
    results = {'gpu': benchmark_gpu, 'loader': benchmark_loader, 'head': benchmark_head}[args.mode](cfg)
    (output / f'{args.mode}_benchmark.json').write_text(json.dumps(results, indent=2) + '\n')


def main_check_cache(argv=None):
    """CLI for `value check-cache`: prove cached features match online encoding.

    Runs on a bounded set of real frames and never writes to the cache or checkpoint.
    """
    parser = argparse.ArgumentParser(description='验证缓存特征与在线编码的一致性')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='训练配置文件路径')
    parser.add_argument('--output', default=CACHE_REPORT, help='输出 JSON 文件路径')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    configure_runtime(cfg)
    ds = raw_dataset(cfg, 'train')
    path, digest, _ = cache_location(ds, cfg, 'train')
    cached = FeatureDataset(path, digest)
    num_bins = cfg.get('num_bins', 201)
    model = ValueModel(str(PROJECT_ROOT / cfg['siglip_path']), cfg['cameras'], True,
                       cfg['precision'], cfg['projection_dim'], num_bins=num_bins).cuda().eval()

    head = None
    checkpoint = PROJECT_ROOT / cfg['save_dir'] / 'best_model.pt'
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['format_version'] == 2
        assert not any(k.startswith('siglip.') for k in payload['model_state_dict'])
        result = model.load_state_dict(payload['model_state_dict'], strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith('siglip.') for k in result.missing_keys)
        head = ValueModel(str(PROJECT_ROOT / cfg['siglip_path']), cfg['cameras'], True,
                          cfg['precision'], cfg['projection_dim'], load_encoder=False,
                          num_bins=num_bins).cuda().eval()
        head.load_state_dict(payload['model_state_dict'], strict=True)

    # Cover the first batch, the last batch and every concatenated-dataset boundary.
    from torch.utils.data import Subset
    size = cfg['cache_batch_size']
    starts = sorted({0, (len(ds) // 2 // size) * size, ((len(ds) // size) - 1) * size,
                     *[(boundary // size) * size for boundary in ds.cumulative_sizes[:-1]]})

    maximum = prediction_error = 0.
    verified = 0
    output_dtypes = set()
    for offset in starts:
        loader = make_loader(Subset(ds, list(range(offset, offset + size))),
                             cfg, 'val', batch_size=size, workers=0, sequential=True)
        batch = next(iter(loader))
        n = len(batch['target_values'])
        images = {k: v.cuda() for k, v in batch['images'].items()}
        with torch.no_grad(), autocast(cfg, torch.device('cuda')):
            encoded = model.encode(images)
            online = model(images=images)
            replay = model(features=cached.features[offset:offset + n].cuda())

        saved = cached.features[offset:offset + n].cuda().float()
        maximum = max(maximum, (encoded.float() - saved).abs().max().item())
        output_dtypes.add(str(encoded.dtype))
        torch.testing.assert_close(encoded.half().float(), saved, atol=0, rtol=0)
        prediction_error = max(prediction_error, (online - replay).abs().max().item())
        torch.testing.assert_close(online, replay, atol=0, rtol=0)
        if head is not None:
            with torch.no_grad(), autocast(cfg, torch.device('cuda')):
                restored = head(features=cached.features[offset:offset + n].cuda())
            torch.testing.assert_close(online, restored, atol=0, rtol=0)
        torch.testing.assert_close(batch['target_values'].float(), cached.targets[offset:offset + n])
        verified += n

    report = {'verified_frames': verified, 'batch_start_indices': starts,
              'max_feature_abs_error': maximum,
              'max_online_vs_cached_prediction_abs_error': prediction_error,
              'encoder_output_dtypes': sorted(output_dtypes), 'cache': str(path),
              'checkpoint_reload': 'passed' if head is not None else 'not tested'}
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main(argv=None):
    """Route to the train, benchmark or check-cache subcommand."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('train', help='训练标量价值模型；--prepare-cache 只构建特征缓存', add_help=False)
    subparsers.add_parser('benchmark', help='测量 GPU、数据加载器或缓存头的吞吐量', add_help=False)
    subparsers.add_parser('check-cache', help='检查缓存/在线特征和 checkpoint 重载的一致性', add_help=False)
    args = parser.parse_args(argv[:1])

    if args.command == 'train':
        main_train(argv[1:])
    elif args.command == 'benchmark':
        main_benchmark(argv[1:])
    elif args.command == 'check-cache':
        main_check_cache(argv[1:])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
