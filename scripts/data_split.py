#!/usr/bin/env python3
"""
数据划分脚本：统计并划分z02机器人数据集

功能：
1. 统计所有数据集的成功/失败/接管情况
2. 按episode级别划分训练集和测试集
3. 生成划分清单文件

使用方法：
    python scripts/data_split.py --data_dir data/raw --output_dir data/splits
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def analyze_episode(parquet_path: str, dataset_name: str) -> dict:
    """分析单个episode的数据
    
    Args:
        parquet_path: parquet文件路径
        dataset_name: 数据集名称
        
    Returns:
        episode信息字典
    """
    df = pq.read_table(parquet_path).to_pandas()
    
    info = {
        'episode_file': os.path.basename(parquet_path),
        'dataset': dataset_name,
        'total_frames': len(df),
        'episode_index': int(df['episode_index'].iloc[0]),
    }
    
    # 根据数据集类型处理不同的字段
    if dataset_name == '2026.09.08':
        # 08数据是完全的遥操作数据，都定为成功
        info['is_success'] = True
        info['reward'] = 0.0
        
        # 08数据没有intervention字段
        info['has_intervention'] = False
        info['intervention_frames'] = 0
        info['intervention_ratio'] = 0.0
        
    else:
        # 15和15_2数据使用 reward, intervention
        if 'reward' in df.columns:
            # 终帧reward=0表示成功，reward=-10000表示失败
            last_reward = float(df['reward'].iloc[-1])
            info['is_success'] = (last_reward == 0.0)
            info['reward'] = last_reward
        else:
            info['is_success'] = None
            info['reward'] = None
        
        # 统计intervention
        if 'intervention' in df.columns:
            info['has_intervention'] = True
            intervention_sum = df['intervention'].sum()
            info['intervention_frames'] = int(intervention_sum)
            info['intervention_ratio'] = float(intervention_sum / len(df))
        else:
            info['has_intervention'] = False
            info['intervention_frames'] = 0
            info['intervention_ratio'] = 0.0
    
    return info


def analyze_dataset(data_dir: str, dataset_name: str) -> list:
    """分析整个数据集
    
    Args:
        data_dir: 数据根目录
        dataset_name: 数据集名称
        
    Returns:
        episode信息列表
    """
    dataset_path = os.path.join(data_dir, dataset_name, 'data/chunk-000')
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path not found: {dataset_path}")
        return []
    
    parquet_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.parquet')])
    
    episodes = []
    for pf in parquet_files:
        parquet_path = os.path.join(dataset_path, pf)
        ep_info = analyze_episode(parquet_path, dataset_name)
        episodes.append(ep_info)
    
    return episodes


def print_statistics(all_episodes: list):
    """打印数据集统计信息
    
    Args:
        all_episodes: 所有episode信息列表
    """
    print("\n" + "="*80)
    print("数据集统计信息")
    print("="*80)
    
    # 按数据集分组
    by_dataset = defaultdict(list)
    for ep in all_episodes:
        by_dataset[ep['dataset']].append(ep)
    
    total_episodes = len(all_episodes)
    total_frames = sum(ep['total_frames'] for ep in all_episodes)
    
    print(f"\n总计: {total_episodes} episodes, {total_frames} frames")
    
    for dataset_name in sorted(by_dataset.keys()):
        episodes = by_dataset[dataset_name]
        print(f"\n--- {dataset_name} ---")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Frames: {sum(ep['total_frames'] for ep in episodes)}")
        
        # 统计成功/失败
        success_eps = [ep for ep in episodes if ep['is_success'] is True]
        failure_eps = [ep for ep in episodes if ep['is_success'] is False]
        unknown_eps = [ep for ep in episodes if ep['is_success'] is None]
        
        print(f"  成功: {len(success_eps)} episodes")
        print(f"  失败: {len(failure_eps)} episodes")
        if unknown_eps:
            print(f"  未知: {len(unknown_eps)} episodes")
        
        # 统计接管情况
        if any(ep['has_intervention'] for ep in episodes):
            no_intervention_eps = [ep for ep in episodes if ep['intervention_ratio'] == 0]
            partial_intervention_eps = [ep for ep in episodes if 0 < ep['intervention_ratio'] < 1]
            full_intervention_eps = [ep for ep in episodes if ep['intervention_ratio'] == 1]
            
            print(f"  无接管: {len(no_intervention_eps)} episodes")
            print(f"  部分接管: {len(partial_intervention_eps)} episodes")
            print(f"  全程接管: {len(full_intervention_eps)} episodes")
    
    # 总体统计
    print("\n--- 总体统计 ---")
    success_eps = [ep for ep in all_episodes if ep['is_success'] is True]
    failure_eps = [ep for ep in all_episodes if ep['is_success'] is False]
    
    print(f"成功episodes: {len(success_eps)} ({len(success_eps)/total_episodes*100:.1f}%)")
    print(f"失败episodes: {len(failure_eps)} ({len(failure_eps)/total_episodes*100:.1f}%)")
    
    success_frames = sum(ep['total_frames'] for ep in success_eps)
    failure_frames = sum(ep['total_frames'] for ep in failure_eps)
    print(f"成功frames: {success_frames} ({success_frames/total_frames*100:.1f}%)")
    print(f"失败frames: {failure_frames} ({failure_frames/total_frames*100:.1f}%)")


def split_dataset(all_episodes: list, 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> dict:
    """划分数据集
    
    Args:
        all_episodes: 所有episode信息列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        划分结果字典
    """
    import random
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    random.seed(seed)
    
    # 按数据集和成功/失败分组
    by_dataset = defaultdict(list)
    for ep in all_episodes:
        by_dataset[ep['dataset']].append(ep)
    
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for dataset_name in sorted(by_dataset.keys()):
        episodes = by_dataset[dataset_name]
        
        # 分离成功和失败episodes
        success_eps = [ep for ep in episodes if ep['is_success'] is True]
        failure_eps = [ep for ep in episodes if ep['is_success'] is False]
        unknown_eps = [ep for ep in episodes if ep['is_success'] is None]
        
        # 打乱顺序
        random.shuffle(success_eps)
        random.shuffle(failure_eps)
        random.shuffle(unknown_eps)
        
        # 划分成功episodes
        n_success = len(success_eps)
        n_train = int(n_success * train_ratio)
        n_val = int(n_success * val_ratio)
        
        splits['train'].extend(success_eps[:n_train])
        splits['val'].extend(success_eps[n_train:n_train+n_val])
        splits['test'].extend(success_eps[n_train+n_val:])
        
        # 划分失败episodes
        n_failure = len(failure_eps)
        n_train = int(n_failure * train_ratio)
        n_val = int(n_failure * val_ratio)
        
        splits['train'].extend(failure_eps[:n_train])
        splits['val'].extend(failure_eps[n_train:n_train+n_val])
        splits['test'].extend(failure_eps[n_train+n_val:])
        
        # 划分未知episodes（如果有）
        if unknown_eps:
            n_unknown = len(unknown_eps)
            n_train = int(n_unknown * train_ratio)
            n_val = int(n_unknown * val_ratio)
            
            splits['train'].extend(unknown_eps[:n_train])
            splits['val'].extend(unknown_eps[n_train:n_train+n_val])
            splits['test'].extend(unknown_eps[n_train+n_val:])
    
    # 打乱每个split的顺序
    for key in splits:
        random.shuffle(splits[key])
    
    return splits


def save_splits(splits: dict, output_dir: str):
    """保存划分结果
    
    Args:
        splits: 划分结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细信息
    for split_name, episodes in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}.json')
        
        # 统计信息
        stats = {
            'split_name': split_name,
            'total_episodes': len(episodes),
            'total_frames': sum(ep['total_frames'] for ep in episodes),
            'success_episodes': len([ep for ep in episodes if ep['is_success'] is True]),
            'failure_episodes': len([ep for ep in episodes if ep['is_success'] is False]),
            'datasets': {}
        }
        
        # 按数据集统计
        by_dataset = defaultdict(list)
        for ep in episodes:
            by_dataset[ep['dataset']].append(ep)
        
        for ds_name, ds_eps in by_dataset.items():
            stats['datasets'][ds_name] = {
                'episodes': len(ds_eps),
                'frames': sum(ep['total_frames'] for ep in ds_eps),
                'episode_indices': [ep['episode_index'] for ep in ds_eps]
            }
        
        # 保存episode列表
        stats['episodes'] = [
            {
                'dataset': ep['dataset'],
                'episode_index': ep['episode_index'],
                'episode_file': ep['episode_file'],
                'total_frames': ep['total_frames'],
                'is_success': ep['is_success'],
                'intervention_ratio': ep['intervention_ratio']
            }
            for ep in episodes
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {split_name}: {len(episodes)} episodes, {stats['total_frames']} frames -> {output_path}")
    
    # 保存汇总信息
    summary_path = os.path.join(output_dir, 'summary.json')
    summary = {
        'train_episodes': len(splits['train']),
        'val_episodes': len(splits['val']),
        'test_episodes': len(splits['test']),
        'train_frames': sum(ep['total_frames'] for ep in splits['train']),
        'val_frames': sum(ep['total_frames'] for ep in splits['val']),
        'test_frames': sum(ep['total_frames'] for ep in splits['test']),
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary -> {summary_path}")


def create_lerobot_splits(splits: dict, data_dir: str, output_dir: str):
    """创建LeRobot格式的划分文件
    
    Args:
        splits: 划分结果字典
        data_dir: 原始数据目录
        output_dir: 输出目录
    """
    lerobot_dir = os.path.join(output_dir, 'lerobot')
    os.makedirs(lerobot_dir, exist_ok=True)
    
    for split_name, episodes in splits.items():
        split_file = os.path.join(lerobot_dir, f'{split_name}_episodes.txt')
        
        with open(split_file, 'w', encoding='utf-8') as f:
            for ep in episodes:
                # 格式: dataset_name/episode_file
                line = f"{ep['dataset']}/{ep['episode_file']}"
                f.write(line + '\n')
        
        print(f"Saved LeRobot split -> {split_file}")


def main():
    parser = argparse.ArgumentParser(description='数据划分脚本')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--analyze_only', action='store_true',
                        help='只分析，不划分')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 分析所有数据集
    datasets = ['2026.09.08', '2026.09.15', '2026.09.15_2']
    all_episodes = []
    
    for ds_name in datasets:
        print(f"\n分析数据集: {ds_name}")
        episodes = analyze_dataset(data_dir, ds_name)
        all_episodes.extend(episodes)
        print(f"  找到 {len(episodes)} episodes")
    
    # 打印统计信息
    print_statistics(all_episodes)
    
    if args.analyze_only:
        print("\n分析完成（仅分析模式）")
        return
    
    # 划分数据集
    print("\n" + "="*80)
    print("划分数据集")
    print("="*80)
    
    splits = split_dataset(
        all_episodes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 打印划分结果
    print("\n划分结果:")
    for split_name, episodes in splits.items():
        success_count = len([ep for ep in episodes if ep['is_success'] is True])
        failure_count = len([ep for ep in episodes if ep['is_success'] is False])
        total_frames = sum(ep['total_frames'] for ep in episodes)
        print(f"  {split_name}: {len(episodes)} episodes ({success_count}成功, {failure_count}失败), {total_frames} frames")
    
    # 保存划分结果
    save_splits(splits, output_dir)
    create_lerobot_splits(splits, data_dir, output_dir)
    
    print("\n数据划分完成！")


if __name__ == '__main__':
    main()
