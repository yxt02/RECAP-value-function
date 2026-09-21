#!/usr/bin/env python3
"""数据准备：审计、计算回报标签、生成轨迹划分。

用法：
    python scripts/prepare_data.py --analyze           # 只读检查
    python scripts/prepare_data.py --all               # 完整执行
    python scripts/prepare_data.py --all --verify-videos  # 含视频验证

详见 docs/scripts.md。
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Before numpy/cv2: submodules sets single-threaded BLAS
import submodules  # noqa: F401

from submodules.data_workflow import run

DEFAULT_CONFIG = 'config/z02_data.yaml'


def main(argv=None):
    p = argparse.ArgumentParser(description='审计和适配 z02 数据集')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='数据配置文件路径')
    p.add_argument('--all', action='store_true', help='执行所有操作（审计 + 回报 + 划分）')
    p.add_argument('--analyze', action='store_true', help='只读检查数据完整性')
    p.add_argument('--compute_returns', action='store_true', help='计算并写入回报标签')
    p.add_argument('--split_data', action='store_true', help='生成训练/验证/测试集划分')
    p.add_argument('--verify-videos', action='store_true', help='验证视频帧数、帧率和解码')
    args = p.parse_args(argv)

    if not any([args.all, args.analyze, args.compute_returns, args.split_data, args.verify_videos]):
        p.print_help()
        return

    print('=' * 60)
    print('[数据准备] 开始执行')
    print('=' * 60)

    if args.analyze:
        print('[数据准备] 模式: 只读分析')
    elif args.all:
        print('[数据准备] 模式: 完整执行（审计 + 回报 + 划分）')
    else:
        flags = []
        if args.compute_returns:
            flags.append('回报计算')
        if args.split_data:
            flags.append('数据划分')
        if args.verify_videos:
            flags.append('视频验证')
        print(f'[数据准备] 模式: {", ".join(flags)}')

    print(f'[数据准备] 配置文件: {args.config}')
    print('-' * 60)

    run(args.config, args.all or args.compute_returns, args.all or args.split_data, args.verify_videos)

    print('-' * 60)
    print('[数据准备] 执行完成')
    print('=' * 60)


if __name__ == '__main__':
    main()
