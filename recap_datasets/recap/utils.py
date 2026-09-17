"""
简化版数据工具函数
绕过rlinf和openpi依赖
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def episode_boundaries(dataset) -> tuple[list[int], list[int]]:
    """获取每个episode的起止索引
    
    Args:
        dataset: LeRobotDataset对象
        
    Returns:
        (starts, ends) 两个列表，包含每个episode的起始和结束索引
    """
    # 从dataset中获取episode_index
    if hasattr(dataset, 'hf_dataset'):
        hf_dataset = dataset.hf_dataset
    else:
        hf_dataset = dataset
    
    # 获取episode_index列
    episode_indices = hf_dataset['episode_index']
    
    # 找到episode边界
    starts = []
    ends = []
    
    current_ep = None
    for i, ep_idx in enumerate(episode_indices):
        if ep_idx != current_ep:
            if current_ep is not None:
                ends.append(i)
            starts.append(i)
            current_ep = ep_idx
    
    # 最后一个episode的结束
    if current_ep is not None:
        ends.append(len(episode_indices))
    
    return starts, ends


def decode_image_struct_batch(samples: dict[str, Any]) -> dict[str, Any]:
    """解码图像数据
    
    LeRobot 2.1格式中，图像可能以bytes或结构化数据存储
    这个函数将其解码为可用的tensor
    
    Args:
        samples: 包含图像数据的字典
        
    Returns:
        解码后的字典
    """
    # HuggingFace transforms return a list per field; LeRobot stacks it later.
    from lerobot.datasets.utils import hf_transform_to_torch
    return hf_transform_to_torch(samples)


def load_returns_sidecar(
    dataset_path: Path,
    tag: Optional[str] = None,
) -> Optional[dict[int, dict[str, np.ndarray]]]:
    """加载returns sidecar文件
    
    Args:
        dataset_path: 数据集路径
        tag: 可选的标签（如"z0"）
        
    Returns:
        {episode_index: {'return': array, 'reward': array, 'prompt': array}} 或 None
    """
    # 构建sidecar文件路径
    if tag:
        sidecar_path = dataset_path / 'meta' / f'returns_{tag}.parquet'
    else:
        sidecar_path = dataset_path / 'meta' / 'returns.parquet'
    
    if not sidecar_path.exists():
        logger.warning(f"Returns sidecar not found: {sidecar_path}")
        return None
    
    import pyarrow.parquet as pq
    df = pq.read_table(str(sidecar_path)).to_pandas()
    if df.duplicated(['episode_index', 'frame_index']).any():
        raise ValueError(f'Duplicate labels: {sidecar_path}')
    if not np.isfinite(df[['return', 'reward']].to_numpy()).all():
        raise ValueError(f'Non-finite labels: {sidecar_path}')
    sidecar = {}
    for ep, group in df.groupby('episode_index'):
        group = group.sort_values('frame_index')
        if not np.array_equal(group.frame_index, np.arange(len(group))):
            raise ValueError(f'Missing/non-contiguous frames in {sidecar_path}, episode {ep}')
        sidecar[int(ep)] = {key: group[key].to_numpy() for key in ('return', 'reward', 'frame_index')}
        sidecar[int(ep)]['prompt'] = group['prompt'].to_numpy() if 'prompt' in group else None
    return sidecar


def load_task_descriptions(dataset_path: Path) -> Optional[dict[int, str]]:
    """加载任务描述
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        {task_index: task_description} 或 None
    """
    tasks_path = dataset_path / 'meta' / 'tasks.jsonl'
    
    if not tasks_path.exists():
        logger.warning(f"Tasks file not found: {tasks_path}")
        return None
    
    try:
        tasks = {}
        with open(tasks_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                task_idx = entry.get('task_index', len(tasks))
                task_desc = entry.get('task', '')
                tasks[task_idx] = task_desc
        
        logger.info(f"Loaded {len(tasks)} task descriptions from {tasks_path}")
        return tasks
        
    except Exception as e:
        logger.error(f"Failed to load task descriptions: {e}")
        return None
