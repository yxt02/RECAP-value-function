#!/usr/bin/env python3
"""从评估结果生成图表和 HTML 报告。

用法：
    python scripts/render.py artifacts/evaluation/<run>

详见 docs/scripts.md。
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation_workflow import render

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description='从评估结果生成图表和 HTML 报告')
    parser.add_argument('output', help='评估目录路径')
    args = parser.parse_args(argv)

    print('=' * 60)
    print('[报告渲染] 开始执行')
    print('=' * 60)
    print(f'[报告渲染] 评估目录: {args.output}')
    print('-' * 60)

    render(args.output)

    print('-' * 60)
    print('[报告渲染] 执行完成')
    print('=' * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
