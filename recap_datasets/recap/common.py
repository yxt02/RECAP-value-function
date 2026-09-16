"""
简化版通用数据加载组件
绕过rlinf依赖，提供基本的数据加载功能
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class BaseDataLoaderImpl(ABC):
    """数据加载器基类"""

    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._data_loader

    def __len__(self) -> int:
        return len(self._data_loader)

    @abstractmethod
    def data_config(self) -> dict:
        """返回数据配置"""
        pass


class ReCapMixtureDataset(Dataset):
    """混合数据集，支持加权采样"""

    mixture_name = "ReCapMixtureDataset"

    def __init__(
        self,
        datasets: list[Dataset],
        weights: Optional[list[float]] = None,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        self.seed = seed

        # 计算每个数据集的长度
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.total_length = sum(self.dataset_lengths)

        # 创建索引映射
        self._build_index_mapping()

        logger.info(
            f"ReCapMixtureDataset: {len(datasets)} datasets, "
            f"total {self.total_length} samples"
        )

    def _build_index_mapping(self):
        """构建索引映射"""
        self.index_mapping = []
        for ds_idx, (ds, length) in enumerate(
            zip(self.datasets, self.dataset_lengths)
        ):
            for sample_idx in range(length):
                self.index_mapping.append((ds_idx, sample_idx))

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx, sample_idx = self.index_mapping[idx]
        return self.datasets[ds_idx][sample_idx]
