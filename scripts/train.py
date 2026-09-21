#!/usr/bin/env python3
"""训练视觉价值基线模型。

用法：
    python scripts/train.py                        # 默认配置训练
    python scripts/train.py --smoke_test           # 冒烟测试
    python scripts/train.py --prepare-cache        # 只构建特征缓存
    python scripts/train.py --no-cache             # 在线编码训练
    python scripts/train.py --num_epochs 5         # 覆盖轮数

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
from submodules.runtime import load_config, configure_runtime
from scripts.training_workflow import train, OVERRIDABLE, SMOKE_TEST, SMOKE_SAMPLES, BENCHMARK_DIR

DEFAULT_CONFIG = 'config/train_value.yaml'

logger = logging.getLogger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser(description='训练视觉价值基线模型')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='训练配置文件路径')
    p.add_argument('--smoke_test', action='store_true', help='使用小样本冒烟测试配置')
    p.add_argument('--prepare-cache', action='store_true', help='只构建特征缓存，不执行训练')
    p.add_argument('--no-cache', action='store_true', help='从原始图像在线编码训练')
    for key, typ in OVERRIDABLE.items():
        p.add_argument('--' + key, type=typ)
    args = p.parse_args(argv)

    print('=' * 60)
    print('[训练] 开始执行')
    print('=' * 60)

    config = load_config(args.config)
    for key, value in vars(args).items():
        if key in OVERRIDABLE and value is not None:
            config[key] = value
    if args.no_cache:
        config['feature_cache'] = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[训练] 设备: {device}')
    print(f'[训练] 配置文件: {args.config}')
    print(f'[训练] 保存目录: {config.get("save_dir", "checkpoints/optimized")}')

    if args.smoke_test:
        print('[训练] 模式: 冒烟测试（小样本快速验证）')
    elif args.prepare_cache:
        print('[训练] 模式: 只构建特征缓存')
    elif args.no_cache:
        print('[训练] 模式: 在线编码训练（无缓存）')
    else:
        print(f'[训练] 模式: 标准训练（{config.get("num_epochs", 24)}轮 / {config.get("max_total_steps", 5000)}步）')

    print(f'[训练] 特征缓存: {"启用" if config.get("feature_cache") else "禁用"}')
    print(f'[训练] 精度: {config.get("precision", "bf16")}')
    print(f'[训练] 批大小: {config.get("cached_batch_size" if config.get("feature_cache") else "batch_size", 16)}')
    print('-' * 60)

    train(config, smoke_test=args.smoke_test, prepare_cache=args.prepare_cache)

    print('-' * 60)
    print('[训练] 执行完成')
    print('=' * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
