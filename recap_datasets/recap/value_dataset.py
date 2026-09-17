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

from recap_datasets.recap.simple_dataset import SimpleValueDataset

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
                    padding = state.new_zeros(*state.shape[:-1], self.target_dim - current_dim)
                    sample['observation/state'] = torch.cat([state, padding], dim=-1)
                elif current_dim > self.target_dim:
                    sample['observation/state'] = state[..., :self.target_dim]

        # Padding actions
        if 'actions' in sample:
            actions = sample['actions']
            if isinstance(actions, torch.Tensor):
                current_dim = actions.shape[-1]
                if current_dim < self.target_dim:
                    padding = actions.new_zeros(*actions.shape[:-1], self.target_dim - current_dim)
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
        self.return_min = float(np.asarray(rs.min).reshape(-1)[0])
        self.return_max = float(np.asarray(rs.max).reshape(-1)[0])

    def normalize_value(self, value: float) -> float:
        if self.normalize_to_minus_one_zero:
            denom = abs(self.return_min) if self.return_min != 0 else 1.0
            return value / denom
        span = self.return_max - self.return_min
        if span == 0:
            return 0.0
        return (value - self.return_min) / span


# ============== 数据集 ==============
class ValueDataset(SimpleValueDataset):
    """Compatibility entry point sharing the tested local backend.

    model_type is metadata only; this is not an OpenPI/RLinf model transform.
    """
    def __init__(self, dataset_path, robot_type='z02', model_type='pi05',
                 action_horizon=10, action_dim=32, default_prompt=None,
                 return_min=None, return_max=None, normalize_to_minus_one_zero=True,
                 max_samples=None, tag='z02_fail2000_v1', **kwargs):
        if not normalize_to_minus_one_zero:
            raise ValueError('The value head requires [-1, 0] targets')
        if default_prompt is not None:
            raise ValueError('Use the dataset task text; default_prompt is not supported')
        if return_min is not None or return_max is not None:
            if return_min is None or return_min >= 0 or return_max != 0:
                raise ValueError('Require return_min < 0 and return_max == 0')
            kwargs['return_scale'] = abs(return_min)
        super().__init__(dataset_path, robot_type=robot_type, tag=tag,
                         action_horizon=action_horizon, action_dim=action_dim or 32,
                         max_samples=max_samples, **kwargs)


# ============== 数据加载器 ==============
class ValueDataLoaderImpl(BaseDataLoaderImpl):
    """轻量级数据加载器包装"""

    def data_config(self) -> dict:
        return {}


class ValueMixtureDataset(ReCapMixtureDataset):
    """混合Value数据集"""

    mixture_name = "ValueMixtureDataset"
