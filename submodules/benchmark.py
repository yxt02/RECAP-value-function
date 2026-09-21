#!/usr/bin/env python3
"""测量 GPU、数据加载器或缓存头的吞吐量。

用法：
    python tests/benchmark.py --mode gpu      # GPU 吞吐量
    python tests/benchmark.py --mode loader   # 数据加载器
    python tests/benchmark.py --mode head     # 缓存头训练

详见 docs/scripts.md。
"""
import argparse
import gc
import json
import logging
import sys
import time
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
from submodules.training_workflow import benchmark_gpu, benchmark_loader, benchmark_head, build_scheduler, epoch, BENCHMARK_DIR

DEFAULT_CONFIG = 'config/train_value.yaml'

logger = logging.getLogger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser(description='测量 GPU、数据加载器或缓存头的吞吐量')
    p.add_argument('--mode', choices=['gpu', 'loader', 'head'], required=True, help='测量模式')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='训练配置文件路径')
    args = p.parse_args(argv)

    print('=' * 60)
    print(f'[性能测试] 模式: {args.mode}')
    print('=' * 60)
    print(f'[性能测试] 配置文件: {args.config}')
    print(f'[性能测试] 输出目录: {BENCHMARK_DIR}')
    print('-' * 60)

    cfg = load_config(args.config)
    configure_runtime(cfg)

    output = PROJECT_ROOT / BENCHMARK_DIR
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == 'gpu':
        print('[性能测试] 正在测试 GPU 吞吐量...')
        results = benchmark_gpu(cfg)
    elif args.mode == 'loader':
        print('[性能测试] 正在测试数据加载器...')
        results = benchmark_loader(cfg)
    elif args.mode == 'head':
        print('[性能测试] 正在测试缓存头训练...')
        results = benchmark_head(cfg)

    output_file = output / f'{args.mode}_benchmark.json'
    output_file.write_text(json.dumps(results, indent=2) + '\n')

    print('-' * 60)
    print(f'[性能测试] 结果已保存: {output_file}')
    print('[性能测试] 执行完成')
    print('=' * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
