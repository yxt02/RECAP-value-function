#!/usr/bin/env python3
"""
z02 机器人数据适配脚本
集成数据准备、Returns 计算、数据划分等功能

使用方法:
    python scripts/adapt_z02.py --all           # 执行所有步骤
    python scripts/adapt_z02.py --compute_returns  # 只计算 returns
    python scripts/adapt_z02.py --split_data       # 只划分数据
    python scripts/adapt_z02.py --analyze          # 只分析数据
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 可配置参数（修改这里）
# ============================================================

CONFIG = {
    # 数据路径
    'data_dir': 'data/raw',                    # 原始数据目录
    'output_dir': 'data/splits',               # 输出目录
    
    # 数据集列表
    'datasets': [
        '2026.09.08',                          # 纯遥操作数据
        '2026.09.15',                          # 混合控制数据
        '2026.09.15_2',                        # 混合控制数据
    ],
    
    # 数据集属性
    'dataset_properties': {
        '2026.09.08': {
            'is_success': True,                # 全部标记为成功（纯遥操作）
            'has_intervention': False,         # 没有 intervention 字段
        },
        '2026.09.15': {
            'is_success': None,                # 从 reward 字段推导
            'has_intervention': True,
        },
        '2026.09.15_2': {
            'is_success': None,                # 从 reward 字段推导
            'has_intervention': True,
        },
    },
    
    # Returns 计算参数
    'gamma': 1.0,                              # 折扣因子（1.0 = 无折扣）
    'failure_reward': -300.0,                  # 失败终止奖励（官方推荐 -300）
    'tag': 'z0',                               # Returns 文件标签
    
    # 数据划分参数
    'train_ratio': 0.8,                        # 训练集比例
    'val_ratio': 0.1,                          # 验证集比例
    'test_ratio': 0.1,                         # 测试集比例
    'random_seed': 42,                         # 随机种子
    
    # 数据集信息
    'robot_type': 'z02',
    'fps': 30,
    'task_description': 'Pick up the red cup and hang it on the cup holder.',
}


# ============================================================
# 数据分析
# ============================================================

def analyze_dataset(data_dir: str, dataset_name: str, properties: dict) -> list:
    """分析单个数据集"""
    dataset_path = PROJECT_ROOT / data_dir / dataset_name / 'data' / 'chunk-000'
    
    if not dataset_path.exists():
        logger.warning(f"数据集路径不存在: {dataset_path}")
        return []
    
    parquet_files = sorted(dataset_path.glob('episode_*.parquet'))
    episodes = []
    
    for pf in parquet_files:
        df = pq.read_table(str(pf)).to_pandas()
        ep_idx = int(df['episode_index'].iloc[0])
        
        # 确定成功/失败
        if properties['is_success'] is not None:
            is_success = properties['is_success']
        else:
            # 从 reward 字段推导
            last_reward = float(df['reward'].iloc[-1])
            is_success = (last_reward == 0.0)
        
        # 统计 intervention
        if properties['has_intervention'] and 'intervention' in df.columns:
            intervention_counts = Counter(df['intervention'])
            intervention_ratio = intervention_counts[1] / len(df) if len(df) > 0 else 0
        else:
            intervention_ratio = 0.0
        
        episodes.append({
            'episode_file': pf.name,
            'dataset': dataset_name,
            'episode_index': ep_idx,
            'total_frames': len(df),
            'is_success': is_success,
            'intervention_ratio': intervention_ratio,
        })
    
    return episodes


def analyze_all_datasets(config: dict):
    """分析所有数据集"""
    logger.info("=" * 60)
    logger.info("数据分析")
    logger.info("=" * 60)
    
    all_episodes = []
    
    for ds_name in config['datasets']:
        properties = config['dataset_properties'][ds_name]
        episodes = analyze_dataset(config['data_dir'], ds_name, properties)
        all_episodes.extend(episodes)
        
        # 统计
        success = sum(1 for ep in episodes if ep['is_success'])
        failure = sum(1 for ep in episodes if not ep['is_success'])
        total_frames = sum(ep['total_frames'] for ep in episodes)
        
        logger.info(f"\n{ds_name}:")
        logger.info(f"  Episodes: {len(episodes)}")
        logger.info(f"  Frames: {total_frames}")
        logger.info(f"  成功: {success}, 失败: {failure}")
        
        if properties['has_intervention']:
            no_int = sum(1 for ep in episodes if ep['intervention_ratio'] == 0)
            low_int = sum(1 for ep in episodes if 0 < ep['intervention_ratio'] <= 0.3)
            mid_int = sum(1 for ep in episodes if 0.3 < ep['intervention_ratio'] <= 0.6)
            high_int = sum(1 for ep in episodes if ep['intervention_ratio'] > 0.6)
            logger.info(f"  Intervention: 无={no_int}, 低={low_int}, 中={mid_int}, 高={high_int}")
    
    # 总体统计
    total_success = sum(1 for ep in all_episodes if ep['is_success'])
    total_failure = sum(1 for ep in all_episodes if not ep['is_success'])
    total_frames = sum(ep['total_frames'] for ep in all_episodes)
    
    logger.info(f"\n总计:")
    logger.info(f"  Episodes: {len(all_episodes)}")
    logger.info(f"  Frames: {total_frames}")
    logger.info(f"  成功: {total_success} ({total_success/len(all_episodes)*100:.1f}%)")
    logger.info(f"  失败: {total_failure} ({total_failure/len(all_episodes)*100:.1f}%)")
    
    return all_episodes


# ============================================================
# Returns 计算
# ============================================================

def compute_returns_for_episode(
    episode_length: int,
    is_success: bool,
    gamma: float,
    failure_reward: float,
) -> tuple:
    """计算单个 episode 的 returns"""
    rewards = np.full(episode_length, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if is_success else failure_reward
    
    returns = np.zeros(episode_length, dtype=np.float32)
    returns[-1] = rewards[-1]
    for t in range(episode_length - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    
    return returns, rewards


def compute_returns(config: dict):
    """计算所有数据集的 returns"""
    logger.info("=" * 60)
    logger.info("计算 Returns")
    logger.info("=" * 60)
    logger.info(f"gamma: {config['gamma']}")
    logger.info(f"failure_reward: {config['failure_reward']}")
    logger.info(f"tag: {config['tag']}")
    
    for ds_name in config['datasets']:
        logger.info(f"\n处理数据集: {ds_name}")
        
        properties = config['dataset_properties'][ds_name]
        data_dir = PROJECT_ROOT / config['data_dir'] / ds_name
        parquet_dir = data_dir / 'data' / 'chunk-000'
        
        if not parquet_dir.exists():
            logger.warning(f"数据目录不存在: {parquet_dir}")
            continue
        
        parquet_files = sorted(parquet_dir.glob('episode_*.parquet'))
        
        # 加载 tasks
        tasks = {}
        tasks_path = data_dir / 'meta' / 'tasks.jsonl'
        if tasks_path.exists():
            with open(tasks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    tasks[entry['task_index']] = entry['task']
        
        # 处理每个 parquet 文件
        all_returns = []
        all_rewards = []
        all_episodes = []
        all_frames = []
        all_prompts = []
        
        for pf in parquet_files:
            df = pq.read_table(str(pf)).to_pandas()
            n = len(df)
            
            ep_idx = int(df['episode_index'].iloc[0])
            
            # 确定成功/失败
            if properties['is_success'] is not None:
                is_success = properties['is_success']
            else:
                last_reward = float(df['reward'].iloc[-1])
                is_success = (last_reward == 0.0)
            
            # 计算 returns
            returns, rewards = compute_returns_for_episode(
                episode_length=n,
                is_success=is_success,
                gamma=config['gamma'],
                failure_reward=config['failure_reward'],
            )
            
            # 收集结果
            all_returns.extend(returns.tolist())
            all_rewards.extend(rewards.tolist())
            all_episodes.extend([ep_idx] * n)
            all_frames.extend(df['frame_index'].tolist())
            
            # prompts
            task_idx = int(df['task_index'].iloc[0]) if 'task_index' in df.columns else 0
            prompt = tasks.get(task_idx, config['task_description'])
            all_prompts.extend([prompt] * n)
        
        # 保存到 parquet
        import pyarrow as pa
        
        result_table = pa.table({
            'episode_index': pa.array(all_episodes, type=pa.int64()),
            'frame_index': pa.array(all_frames, type=pa.int64()),
            'return': pa.array(all_returns, type=pa.float32()),
            'reward': pa.array(all_rewards, type=pa.float32()),
            'prompt': pa.array(all_prompts, type=pa.string()),
        })
        
        output_path = data_dir / 'meta' / f'returns_{config["tag"]}.parquet'
        pq.write_table(result_table, str(output_path))
        
        logger.info(f"  保存到: {output_path}")
        logger.info(f"  行数: {len(all_returns)}")
        logger.info(f"  Return 范围: [{min(all_returns):.2f}, {max(all_returns):.2f}]")
        
        # 更新 stats.json
        stats_path = data_dir / 'meta' / 'stats.json'
        stats = {}
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        
        stats['return'] = {
            'mean': float(np.mean(all_returns)),
            'std': float(np.std(all_returns)),
            'min': float(min(all_returns)),
            'max': float(max(all_returns)),
        }
        stats['reward'] = {
            'mean': float(np.mean(all_rewards)),
            'std': float(np.std(all_rewards)),
            'min': float(min(all_rewards)),
            'max': float(max(all_rewards)),
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"  更新 stats.json")


# ============================================================
# 数据划分
# ============================================================

def split_data(config: dict):
    """划分数据集"""
    import random
    
    logger.info("=" * 60)
    logger.info("数据划分")
    logger.info("=" * 60)
    
    # 分析所有数据集
    all_episodes = []
    for ds_name in config['datasets']:
        properties = config['dataset_properties'][ds_name]
        episodes = analyze_dataset(config['data_dir'], ds_name, properties)
        all_episodes.extend(episodes)
    
    random.seed(config['random_seed'])
    
    # 按数据集和成功/失败分组
    by_dataset = defaultdict(list)
    for ep in all_episodes:
        by_dataset[ep['dataset']].append(ep)
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for ds_name in sorted(by_dataset.keys()):
        episodes = by_dataset[ds_name]
        
        # 分离成功和失败
        success_eps = [ep for ep in episodes if ep['is_success']]
        failure_eps = [ep for ep in episodes if not ep['is_success']]
        
        random.shuffle(success_eps)
        random.shuffle(failure_eps)
        
        # 划分成功 episodes
        n = len(success_eps)
        n_train = int(n * config['train_ratio'])
        n_val = int(n * config['val_ratio'])
        
        splits['train'].extend(success_eps[:n_train])
        splits['val'].extend(success_eps[n_train:n_train+n_val])
        splits['test'].extend(success_eps[n_train+n_val:])
        
        # 划分失败 episodes
        n = len(failure_eps)
        n_train = int(n * config['train_ratio'])
        n_val = int(n * config['val_ratio'])
        
        splits['train'].extend(failure_eps[:n_train])
        splits['val'].extend(failure_eps[n_train:n_train+n_val])
        splits['test'].extend(failure_eps[n_train+n_val:])
    
    # 打乱顺序
    for key in splits:
        random.shuffle(splits[key])
    
    # 保存结果
    output_dir = PROJECT_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, episodes in splits.items():
        # 统计
        success = sum(1 for ep in episodes if ep['is_success'])
        failure = sum(1 for ep in episodes if not ep['is_success'])
        total_frames = sum(ep['total_frames'] for ep in episodes)
        
        output = {
            'split_name': split_name,
            'total_episodes': len(episodes),
            'total_frames': total_frames,
            'success_episodes': success,
            'failure_episodes': failure,
            'episodes': episodes,
        }
        
        output_path = output_dir / f'{split_name}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"{split_name}: {len(episodes)} episodes, {total_frames} frames "
                    f"(成功={success}, 失败={failure})")
    
    # 保存汇总
    summary = {k: len(v) for k, v in splits.items()}
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n保存到: {output_dir}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='z02 机器人数据适配脚本')
    parser.add_argument('--all', action='store_true', help='执行所有步骤')
    parser.add_argument('--analyze', action='store_true', help='分析数据')
    parser.add_argument('--compute_returns', action='store_true', help='计算 returns')
    parser.add_argument('--split_data', action='store_true', help='划分数据')
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    logger.info("=" * 60)
    logger.info("z02 机器人数据适配")
    logger.info("=" * 60)
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"数据目录: {CONFIG['data_dir']}")
    logger.info(f"Failure reward: {CONFIG['failure_reward']}")
    
    # 执行步骤
    if args.all or args.analyze:
        analyze_all_datasets(CONFIG)
    
    if args.all or args.compute_returns:
        compute_returns(CONFIG)
    
    if args.all or args.split_data:
        split_data(CONFIG)
    
    logger.info("\n" + "=" * 60)
    logger.info("完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()