"""
简化版Value-model数据集
绕过rlinf和openpi依赖，提供基本的数据加载功能
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # lerobot >= 0.2 layout
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )
except ModuleNotFoundError:  # lerobot < 0.2
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

from recap_datasets.recap.common import BaseDataLoaderImpl, ReCapMixtureDataset
from recap_datasets.recap.utils import (
    decode_image_struct_batch,
    episode_boundaries,
    load_returns_sidecar,
    load_task_descriptions,
)

logger = logging.getLogger(__name__)


# ============== 数据字段映射 ==============
_REPACK_KEYS = {
    "z02": {
        "observation/image": "observation.images.cam2",
        "observation/wrist_image": "observation.images.cam3",
        "observation/left_wrist_image": "observation.images.cam4",
        "observation/state": "observation.joint_positions",
        "actions": "action.joint_positions",
        "prompt": "prompt",
    },
    "libero": {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
    "libero_v2": {
        "observation/image": "observation.images.image",
        "observation/wrist_image": "observation.images.wrist_image",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    },
    "franka": {
        "observation/image": "image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
    "franka_co_train": {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
}


# ============== 数据变换 ==============
class RepackTransform:
    """将数据字段从LeRobot格式重命名为模型期望的格式"""

    def __init__(self, key_mapping: dict[str, str]):
        self.key_mapping = key_mapping

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for target_key, source_key in self.key_mapping.items():
            if source_key in sample:
                result[target_key] = sample[source_key]
            elif '.' in source_key:
                # 尝试从嵌套结构获取
                parts = source_key.split('.')
                value = sample
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                if value is not None:
                    result[target_key] = value
        return result


class InjectDefaultPrompt:
    """注入默认prompt"""

    def __init__(self, default_prompt: Optional[str] = None):
        self.default_prompt = default_prompt or "perform the task"

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if 'prompt' not in sample or not sample['prompt']:
            sample['prompt'] = self.default_prompt
        return sample


class PadStatesAndActions:
    """将状态和动作padding到指定维度"""

    def __init__(self, target_dim: int = 32):
        self.target_dim = target_dim

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # Padding state
        if 'observation/state' in sample:
            state = sample['observation/state']
            if isinstance(state, torch.Tensor):
                current_dim = state.shape[-1]
                if current_dim < self.target_dim:
                    padding = torch.zeros(*state.shape[:-1], self.target_dim - current_dim)
                    sample['observation/state'] = torch.cat([state, padding], dim=-1)
                elif current_dim > self.target_dim:
                    sample['observation/state'] = state[..., :self.target_dim]

        # Padding actions
        if 'actions' in sample:
            actions = sample['actions']
            if isinstance(actions, torch.Tensor):
                current_dim = actions.shape[-1]
                if current_dim < self.target_dim:
                    padding = torch.zeros(*actions.shape[:-1], self.target_dim - current_dim)
                    sample['actions'] = torch.cat([actions, padding], dim=-1)
                elif current_dim > self.target_dim:
                    sample['actions'] = actions[..., :self.target_dim]

        return sample


def compose_transforms(transforms_list):
    """组合多个变换"""
    def composed(sample):
        for transform in transforms_list:
            sample = transform(sample)
        return sample
    return composed


# ============== 归一化统计 ==============
@dataclass
class NormStats:
    """归一化统计量"""
    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None


def _dict_to_norm_stats(data: dict[str, Any]) -> NormStats:
    return NormStats(
        mean=np.array(data["mean"]),
        std=np.array(data["std"]),
        q01=np.array(data["q01"]) if data.get("q01") is not None else None,
        q99=np.array(data["q99"]) if data.get("q99") is not None else None,
        min=np.array(data["min"]) if data.get("min") is not None else None,
        max=np.array(data["max"]) if data.get("max") is not None else None,
    )


def load_stats(norm_stats_path: Path) -> dict[str, NormStats]:
    """加载归一化统计量"""
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {norm_stats_path}")

    with open(norm_stats_path, "r") as f:
        data = json.load(f)

    if "norm_stats" in data:
        data = data["norm_stats"]

    return {key: _dict_to_norm_stats(stats_dict) for key, stats_dict in data.items()}


# ============== Return归一化 ==============
class ReturnNormalizer:
    """Return值归一化"""

    def __init__(
        self,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        norm_stats: Optional[dict[str, NormStats]] = None,
        norm_stats_path: Optional[Path] = None,
        return_key: str = "return",
        keep_continuous: bool = True,
        normalize_to_minus_one_zero: bool = True,
    ):
        self.return_key = return_key
        self.keep_continuous = keep_continuous
        self.normalize_to_minus_one_zero = normalize_to_minus_one_zero

        if return_min is not None and return_max is not None:
            self.return_min = return_min
            self.return_max = return_max
        elif norm_stats is not None:
            self._load_from_norm_stats(norm_stats)
        elif norm_stats_path is not None:
            self._load_from_norm_stats(load_stats(Path(norm_stats_path)))
        else:
            raise ValueError(
                "Must provide either (return_min, return_max), norm_stats, "
                "or norm_stats_path"
            )

        logger.info(
            "ReturnNormalizer: return_min=%.4f, return_max=%.4f, range=%s",
            self.return_min,
            self.return_max,
            "(-1, 0)" if self.normalize_to_minus_one_zero else "(0, 1)",
        )

    def _load_from_norm_stats(self, norm_stats: dict[str, NormStats]):
        if "return" not in norm_stats:
            raise ValueError("norm_stats must contain 'return' key")
        rs = norm_stats["return"]
        self.return_min = float(rs.min[0] if hasattr(rs.min, "__len__") else rs.min)
        self.return_max = float(rs.max[0] if hasattr(rs.max, "__len__") else rs.max)

    def normalize_value(self, value: float) -> float:
        if self.normalize_to_minus_one_zero:
            denom = abs(self.return_min) if self.return_min != 0 else 1.0
            return value / denom
        span = self.return_max - self.return_min
        if span == 0:
            return 0.0
        return (value - self.return_min) / span


# ============== 数据集 ==============
class ValueDataset(Dataset):
    """简化版Value模型数据集"""

    def __init__(
        self,
        dataset_path: str,
        robot_type: str,
        model_type: str = "pi05",
        action_horizon: int = 10,
        action_dim: Optional[int] = None,
        default_prompt: Optional[str] = None,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        normalize_to_minus_one_zero: bool = True,
        max_samples: Optional[int] = None,
        tag: Optional[str] = None,
        episode_percentage: Optional[float] = None,
        shuffle_episodes: bool = False,
        episode_seed: int = 42,
        **kwargs,
    ):
        self.max_samples = max_samples
        local_path = Path(dataset_path).absolute()

        # 加载数据集元信息
        self.dataset_meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
        
        # 确定action字段名
        if "action" in self.dataset_meta.features:
            action_key = "action"
        elif "actions" in self.dataset_meta.features:
            action_key = "actions"
        elif "action.joint_positions" in self.dataset_meta.features:
            action_key = "action.joint_positions"
        else:
            raise ValueError(
                f"No action key in dataset features: "
                f"{list(self.dataset_meta.features.keys())}"
            )

        # 创建delta_timestamps
        delta_timestamps = {
            action_key: [t / self.dataset_meta.fps for t in range(action_horizon)]
        }

        # 加载基础数据集
        self._base = LeRobotDataset(
            local_path.name,
            root=local_path,
            delta_timestamps=delta_timestamps,
            download_videos=False,
        )
        self._base.hf_dataset.set_transform(decode_image_struct_batch)

        # 加载returns sidecar
        self._sidecar = load_returns_sidecar(local_path, tag)
        if self._sidecar is None:
            raise FileNotFoundError(
                f"Returns sidecar not found for {dataset_path}. "
                f"Run compute_returns.py first to generate "
                f"meta/returns{'_' + tag if tag else ''}.parquet"
            )

        # 处理episode子采样
        self._indices = None
        if episode_percentage is not None and episode_percentage < 100:
            if episode_percentage <= 0:
                raise ValueError(
                    f"episode_percentage must be > 0, got {episode_percentage}"
                )
            total = self.dataset_meta.total_episodes
            num = max(1, int(total * episode_percentage / 100.0))
            all_eps = list(range(total))
            if shuffle_episodes:
                rng = np.random.default_rng(episode_seed)
                selected = set(rng.choice(all_eps, size=num, replace=False).tolist())
            else:
                selected = set(all_eps[:num])
            ep_starts, ep_ends = episode_boundaries(self._base)
            self._indices = [
                i for ep in sorted(selected) for i in range(ep_starts[ep], ep_ends[ep])
            ]

        # 构建变换
        self._transform = self._build_transform(
            robot_type=robot_type,
            model_type=model_type,
            action_dim=action_dim or 32,
            default_prompt=default_prompt,
        )

        # 加载任务描述
        self._tasks = load_task_descriptions(local_path) or (
            self.dataset_meta.tasks if hasattr(self.dataset_meta, "tasks") else None
        )

        # Return归一化
        self._normalizer = (
            ReturnNormalizer(
                return_min=return_min,
                return_max=return_max,
                normalize_to_minus_one_zero=normalize_to_minus_one_zero,
            )
            if return_min is not None and return_max is not None
            else None
        )

        n = len(self._indices) if self._indices else len(self._base)
        logger.info(f"ValueDataset: {dataset_path}, {min(n, max_samples or n)} samples")

    @staticmethod
    def _build_transform(robot_type, model_type, action_dim, default_prompt):
        """构建数据变换"""
        robot = robot_type.lower()

        transforms_list = []
        repack_keys = _REPACK_KEYS.get(robot)
        if repack_keys is None:
            raise ValueError(
                f"Unknown robot type: {robot_type}. "
                f"Available: {list(_REPACK_KEYS.keys())}"
            )
        transforms_list.append(RepackTransform(repack_keys))
        transforms_list.append(InjectDefaultPrompt(default_prompt))
        transforms_list.append(PadStatesAndActions(action_dim))

        return compose_transforms(transforms_list)

    def __len__(self) -> int:
        n = len(self._indices) if self._indices else len(self._base)
        return min(n, self.max_samples) if self.max_samples else n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._indices[idx] if self._indices else idx
        sample = self._base[real_idx]

        ep = int(sample.get("episode_index", -1))
        fr = int(sample.get("frame_index", -1))
        if ep < 0 or fr < 0:
            raise KeyError(
                f"LeRobot sample missing episode_index ({ep}) or "
                f"frame_index ({fr}) at real_idx={real_idx}. "
                f"Available keys: {sorted(sample.keys())}"
            )

        # 注入任务描述
        if self._tasks and "task_index" in sample:
            ti = sample["task_index"]
            ti = ti.item() if isinstance(ti, torch.Tensor) else int(ti)
            if ti in self._tasks:
                sample = {**sample, "prompt": self._tasks[ti]}

        # 应用变换
        if self._transform is not None:
            sample = self._transform(sample)

        # 获取return值
        if ep not in self._sidecar:
            raise KeyError(
                f"Episode {ep} not found in returns sidecar at "
                f"real_idx={real_idx}. The sidecar/tag may not match the "
                f"dataset; re-run compute_returns.py with the correct tag."
            )
        raw = float(self._sidecar[ep]["return"][fr])
        target_value = (
            self._normalizer.normalize_value(raw) if self._normalizer else raw
        )

        # 组装输出
        images = sample.get("image", sample.get("images", {}))
        if not isinstance(images, dict):
            images = {}
        masks = sample.get("image_mask", sample.get("image_masks"))

        result: dict[str, Any] = {
            "images": images,
            "prompt": sample.get("prompt", "perform the task"),
            "target_values": target_value,
            "actions": None,
        }
        if isinstance(masks, dict) and masks:
            result["image_masks"] = masks
        return result


# ============== 数据加载器 ==============
class ValueDataLoaderImpl(BaseDataLoaderImpl):
    """轻量级数据加载器包装"""

    def data_config(self) -> dict:
        return {}


class ValueMixtureDataset(ReCapMixtureDataset):
    """混合Value数据集"""

    mixture_name = "ValueMixtureDataset"
