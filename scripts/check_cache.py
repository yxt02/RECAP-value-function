#!/usr/bin/env python3
"""验证缓存特征与在线编码的一致性。

用法：
    python scripts/check_cache.py
    python scripts/check_cache.py --output artifacts/performance/my-check.json

详见 docs/scripts.md。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Before numpy/torch: submodules sets single-threaded BLAS
import submodules  # noqa: F401

import torch

from submodules.contracts import resolve_path
from submodules.cache import FeatureDataset, cache_location
from submodules.model import ValueModel
from submodules.runtime import autocast, configure_runtime, load_config, make_loader, raw_dataset

DEFAULT_CONFIG = 'config/train_value.yaml'
CACHE_REPORT = 'artifacts/performance/cache-verification.json'

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description='验证缓存特征与在线编码的一致性')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='训练配置文件路径')
    parser.add_argument('--output', default=CACHE_REPORT, help='输出 JSON 文件路径')
    args = parser.parse_args(argv)

    print('=' * 60)
    print('[缓存检查] 开始执行')
    print('=' * 60)
    print(f'[缓存检查] 配置文件: {args.config}')
    print(f'[缓存检查] 输出文件: {args.output}')
    print('-' * 60)

    cfg = load_config(args.config)
    configure_runtime(cfg)
    ds = raw_dataset(cfg, 'train')
    path, digest, _ = cache_location(ds, cfg, 'train')
    cached = FeatureDataset(path, digest)
    num_bins = cfg.get('num_bins', 201)
    model = ValueModel(str(PROJECT_ROOT / cfg['siglip_path']), cfg['cameras'], True,
                       cfg['precision'], cfg['projection_dim'], num_bins=num_bins).cuda().eval()

    print(f'[缓存检查] 缓存路径: {path}')
    print(f'[缓存检查] 缓存摘要: {digest[:16]}...')

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
        print(f'[缓存检查] Checkpoint: {checkpoint}')
    else:
        print('[缓存检查] Checkpoint: 未找到，跳过重载检查')

    # Cover the first batch, the last batch and every concatenated-dataset boundary.
    from torch.utils.data import Subset
    size = cfg['cache_batch_size']
    starts = sorted({0, (len(ds) // 2 // size) * size, ((len(ds) // size) - 1) * size,
                     *[(boundary // size) * size for boundary in ds.cumulative_sizes[:-1]]})

    print(f'[缓存检查] 检查 {len(starts)} 个批次位置...')

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

    print('-' * 60)
    print(f'[缓存检查] 验证帧数: {verified}')
    print(f'[缓存检查] 最大特征误差: {maximum:.2e}')
    print(f'[缓存检查] 最大预测误差: {prediction_error:.2e}')
    print(f'[缓存检查] Checkpoint重载: {report["checkpoint_reload"]}')
    print(f'[缓存检查] 结果已保存: {output}')
    print('[缓存检查] 执行完成')
    print('=' * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
