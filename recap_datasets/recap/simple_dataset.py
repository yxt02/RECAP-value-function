"""
简化版数据集加载器
直接读取Parquet文件和视频帧
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class SimpleValueDataset(Dataset):
    """简化版Value模型数据集
    
    直接读取Parquet文件和视频帧
    """

    def __init__(
        self,
        dataset_path: str,
        robot_type: str = "z02",
        tag: Optional[str] = None,
        action_horizon: int = 10,
        action_dim: int = 32,
        max_samples: Optional[int] = None,
        normalize_returns: bool = True,
        image_size: int = 224,
        cameras: list[str] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.robot_type = robot_type
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.max_samples = max_samples
        self.normalize_returns = normalize_returns
        self.image_size = image_size
        self.cameras = cameras or ['cam2']  # 默认只用头部摄像头

        # 加载parquet文件列表
        data_dir = self.dataset_path / 'data' / 'chunk-000'
        self.parquet_files = sorted(data_dir.glob('episode_*.parquet'))
        
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # 视频目录
        self.video_dirs = {}
        for cam in self.cameras:
            video_dir = self.dataset_path / 'videos' / 'chunk-000' / f'observation.images.{cam}'
            if video_dir.exists():
                self.video_dirs[cam] = video_dir
                logger.info(f"Found video dir for {cam}: {video_dir}")
            else:
                logger.warning(f"Video dir not found for {cam}: {video_dir}")

        # 加载任务描述
        self.tasks = self._load_tasks()

        # 加载returns sidecar
        self.returns_data = self._load_returns(tag)

        # 构建索引映射：(file_idx, frame_idx)
        self._build_index_mapping()

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 缓存视频捕获对象
        self._video_caps = {}

        logger.info(
            f"SimpleValueDataset: {dataset_path}, "
            f"{len(self.parquet_files)} episodes, "
            f"{len(self)} samples, "
            f"cameras={list(self.video_dirs.keys())}"
        )

    def _load_tasks(self) -> dict[int, str]:
        """加载任务描述"""
        tasks_path = self.dataset_path / 'meta' / 'tasks.jsonl'
        tasks = {}
        
        if tasks_path.exists():
            with open(tasks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    task_idx = entry.get('task_index', len(tasks))
                    task_desc = entry.get('task', '')
                    tasks[task_idx] = task_desc
        
        return tasks

    def _load_returns(self, tag: Optional[str]) -> dict:
        """加载returns数据"""
        if tag:
            returns_path = self.dataset_path / 'meta' / f'returns_{tag}.parquet'
        else:
            returns_path = self.dataset_path / 'meta' / 'returns.parquet'
        
        if not returns_path.exists():
            logger.warning(f"Returns file not found: {returns_path}")
            return {}
        
        table = pq.read_table(str(returns_path))
        df = table.to_pandas()
        
        # 按episode_index和frame_index组织数据
        returns_data = {}
        for _, row in df.iterrows():
            ep_idx = int(row['episode_index'])
            fr_idx = int(row['frame_index'])
            if ep_idx not in returns_data:
                returns_data[ep_idx] = {}
            returns_data[ep_idx][fr_idx] = {
                'return': float(row['return']),
                'reward': float(row['reward']),
                'prompt': str(row.get('prompt', 'perform the task')),
            }
        
        # 计算归一化参数
        if self.normalize_returns and returns_data:
            all_returns = []
            for ep_data in returns_data.values():
                for frame_data in ep_data.values():
                    all_returns.append(frame_data['return'])
            
            if all_returns:
                self.return_min = min(all_returns)
                self.return_max = max(all_returns)
                logger.info(f"Return range: [{self.return_min:.2f}, {self.return_max:.2f}]")
        
        return returns_data

    def _build_index_mapping(self):
        """构建索引映射"""
        self.index_mapping = []
        
        for file_idx, parquet_file in enumerate(self.parquet_files):
            # 读取这个episode的帧数
            pf = pq.ParquetFile(str(parquet_file))
            num_frames = pf.metadata.num_rows
            
            for frame_idx in range(num_frames):
                self.index_mapping.append((file_idx, frame_idx))
        
        # 限制样本数
        if self.max_samples and self.max_samples < len(self.index_mapping):
            self.index_mapping = self.index_mapping[:self.max_samples]

    def _get_video_cap(self, cam: str, episode_idx: int):
        """获取视频捕获对象（带缓存）"""
        key = (cam, episode_idx)
        if key not in self._video_caps:
            video_path = self.video_dirs[cam] / f'episode_{episode_idx:06d}.mp4'
            if video_path.exists():
                self._video_caps[key] = cv2.VideoCapture(str(video_path))
            else:
                logger.warning(f"Video file not found: {video_path}")
                return None
        return self._video_caps[key]

    def _read_video_frame(self, cam: str, episode_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        """读取视频帧"""
        cap = self._get_video_cap(cam, episode_idx)
        if cap is None:
            return None
        
        # 设置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, frame_idx = self.index_mapping[idx]
        parquet_file = self.parquet_files[file_idx]
        
        # 读取单帧数据
        df = pq.read_table(str(parquet_file)).to_pandas()
        row = df.iloc[frame_idx]
        
        # 提取episode_index
        ep_idx = int(row['episode_index'])
        
        # 构建样本
        sample = {}
        
        # 图像：读取视频帧
        images = {}
        for cam in self.cameras:
            frame = self._read_video_frame(cam, ep_idx, frame_idx)
            if frame is not None:
                # 应用图像变换
                images[f'observation.images.{cam}'] = self.image_transform(frame)
            else:
                # 返回占位符
                images[f'observation.images.{cam}'] = torch.zeros(3, self.image_size, self.image_size)
        
        sample['images'] = images
        
        # 状态：关节位置
        if 'observation.joint_positions' in row:
            state = np.array(row['observation.joint_positions'], dtype=np.float32)
            # Padding到目标维度
            if len(state) < self.action_dim:
                state = np.pad(state, (0, self.action_dim - len(state)))
            elif len(state) > self.action_dim:
                state = state[:self.action_dim]
            sample['observation/state'] = torch.from_numpy(state)
        
        # 动作
        if 'action.joint_positions' in row:
            action = np.array(row['action.joint_positions'], dtype=np.float32)
            # Padding到目标维度
            if len(action) < self.action_dim:
                action = np.pad(action, (0, self.action_dim - len(action)))
            elif len(action) > self.action_dim:
                action = action[:self.action_dim]
            sample['actions'] = torch.from_numpy(action)
        
        # 任务描述
        task_idx = int(row.get('task_index', 0))
        sample['prompt'] = self.tasks.get(task_idx, 'perform the task')
        
        # Return值
        if ep_idx in self.returns_data and frame_idx in self.returns_data[ep_idx]:
            raw_return = self.returns_data[ep_idx][frame_idx]['return']
            
            if self.normalize_returns:
                # 归一化到 [-1, 0] 范围
                denom = abs(self.return_min) if self.return_min != 0 else 1.0
                sample['target_values'] = raw_return / denom
            else:
                sample['target_values'] = raw_return
        else:
            sample['target_values'] = 0.0
        
        # 元数据
        sample['episode_index'] = ep_idx
        sample['frame_index'] = frame_idx
        
        return sample

    def __del__(self):
        """清理视频捕获对象"""
        for cap in self._video_caps.values():
            if cap is not None:
                cap.release()


def create_simple_dataloader(
    dataset_path: str,
    robot_type: str = "z02",
    tag: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    cameras: list[str] = None,
):
    """创建简化版数据加载器"""
    from torch.utils.data import DataLoader
    
    dataset = SimpleValueDataset(
        dataset_path=dataset_path,
        robot_type=robot_type,
        tag=tag,
        max_samples=max_samples,
        cameras=cameras,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试数据集
    logging.basicConfig(level=logging.INFO)
    
    dataset = SimpleValueDataset(
        dataset_path='data/raw/2026.09.15',
        robot_type='z02',
        tag='z0',
        max_samples=10,
        cameras=['cam2', 'cam3', 'cam4'],
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"样本keys: {list(sample.keys())}")
    print(f"images keys: {list(sample['images'].keys())}")
    for key, img in sample['images'].items():
        print(f"  {key} shape: {img.shape}, min: {img.min():.3f}, max: {img.max():.3f}")
    print(f"state shape: {sample['observation/state'].shape}")
    print(f"actions shape: {sample['actions'].shape}")
    print(f"prompt: {sample['prompt']}")
    print(f"target_values: {sample['target_values']:.4f}")
