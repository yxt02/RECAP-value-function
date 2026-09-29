#!/usr/bin/env python3
"""训练 RECAP 视觉与文字价值模型。

用法：
    python scripts/train.py                        # 默认配置训练
    python scripts/train.py --smoke_test           # 冒烟测试
    python scripts/train.py --num_epochs 5         # 覆盖轮数

详见 docs/usage.md。
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Before numpy/torch: submodules sets single-threaded BLAS
import submodules  # noqa: F401

import torch

from submodules.contracts import resolve_path
from submodules.runtime import load_config, configure_runtime
from submodules.training_workflow import train, OVERRIDABLE, SMOKE_TEST, SMOKE_SAMPLES, BENCHMARK_DIR

DEFAULT_CONFIG = 'config/train_value.yaml'

logger = logging.getLogger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser(description='训练 RECAP 视觉与文字价值模型')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='训练配置文件路径')
    p.add_argument('--smoke_test', action='store_true', help='使用小样本冒烟测试配置')
    p.add_argument('--prepare-cache', action='store_true', help='仅旧平均特征架构：构建特征缓存')
    p.add_argument('--no-cache', action='store_true', help='仅旧平均特征架构：从图像在线训练')
    for key, typ in OVERRIDABLE.items():
        p.add_argument('--' + key, type=typ)
    for key in ('freeze_vision_encoder', 'freeze_vlm', 'freeze_value_expert',
                'gradient_checkpointing'):
        p.add_argument('--' + key, action=argparse.BooleanOptionalAction, default=None)
    args = p.parse_args(argv)
    report = print if os.environ.get('RANK', '0') == '0' else lambda *args, **kwargs: None

    report('=' * 60)
    report('[训练] 开始执行')
    report('=' * 60)

    config = load_config(args.config)
    for key, value in vars(args).items():
        if (key in OVERRIDABLE or key in ('freeze_vision_encoder', 'freeze_vlm',
                                        'freeze_value_expert', 'gradient_checkpointing')) and value is not None:
            config[key] = value
    if args.no_cache:
        config['feature_cache'] = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report(f'[训练] 设备: {device}')
    report(f'[训练] 配置文件: {args.config}')
    report(f'[训练] 保存目录: {config.get("save_dir", "checkpoints/recap_patch")}')

    if args.smoke_test:
        report('[训练] 模式: 冒烟测试（小样本快速验证）')
    elif args.prepare_cache:
        report('[训练] 模式: 只构建特征缓存')
    elif args.no_cache:
        report('[训练] 模式: 在线编码训练（无缓存）')
    else:
        step_limit = config.get('max_total_steps')
        limit_text = '不设总步数上限' if step_limit is None else f'最多 {step_limit} 次参数更新'
        report(f'[训练] 模式: 标准训练（最多 {config["num_epochs"]} 轮，{limit_text}）')

    report(f'[训练] 特征缓存: {"启用" if config.get("feature_cache") else "禁用"}')
    report(f'[训练] 精度: {config.get("precision", "bf16")}')
    batch_size = (1 if args.smoke_test and config.get('architecture') == 'recap_patch_gemma_expert'
                  else config.get('cached_batch_size' if config.get('feature_cache') else 'batch_size', 16))
    report(f'[训练] 每卡批大小: {batch_size}; GPU 数: {os.environ.get("WORLD_SIZE", "1")}')
    report('-' * 60)

    train(config, smoke_test=args.smoke_test, prepare_cache=args.prepare_cache)

    report('-' * 60)
    report('[训练] 执行完成')
    report('=' * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
