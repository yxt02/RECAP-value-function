#!/usr/bin/env python3
"""使用固定 checkpoint 进行全测试集评估。

用法：
    python tests/evaluate.py
    python tests/evaluate.py --checkpoint checkpoints/optimized/best_model.pt
    python tests/evaluate.py --output artifacts/evaluation/my-run

详见 docs/scripts.md。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'tests'))

from submodules.evaluation_workflow import evaluate, DEFAULT_CHECKPOINT

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description='使用固定 checkpoint 进行全测试集评估')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='checkpoint 文件路径')
    parser.add_argument('--output', help='输出目录')
    args = parser.parse_args(argv)

    print('=' * 60)
    print('[评估] 开始执行')
    print('=' * 60)
    print(f'[评估] Checkpoint: {args.checkpoint}')
    if args.output:
        print(f'[评估] 输出目录: {args.output}')
    else:
        print('[评估] 输出目录: 自动生成')
    print('-' * 60)

    evaluate(args.checkpoint, args.output)

    print('-' * 60)
    print('[评估] 执行完成')
    print('=' * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
