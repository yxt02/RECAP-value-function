"""Local z02 value data: strict labels, shared scale, split filtering and video I/O."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .contracts import load_normalization, read_split
from .utils import load_returns_sidecar


class SimpleValueDataset(Dataset):
    def __init__(self, dataset_path, robot_type='z02', tag='z02_fail2000_v1',
                 action_horizon=10, action_dim=32, max_samples=None,
                 normalize_returns=True, image_size=224, cameras=None,
                 split_file=None, episodes=None, return_scale=None,
                 include_state=True, include_actions=True, min_episode_steps=2,
                 image_processor_path=None, cache_episodes=4, cache_videos=6, decoder_threads=1):
        self._video_caps = OrderedDict()
        self._episode_cache = OrderedDict()
        self._pid = os.getpid()
        self.decoder_threads = max(1, int(decoder_threads))
        self.dataset_path = Path(dataset_path).resolve()
        if robot_type != 'z02':
            raise ValueError('This local loader supports z02 only')
        if action_horizon < 1 or action_dim < 22 or (max_samples is not None and max_samples < 1):
            raise ValueError('Invalid horizon, action_dim or max_samples')
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.include_state, self.include_actions = include_state, include_actions
        self.normalize_returns = normalize_returns
        self.cameras = list(cameras if cameras is not None else ['cam2', 'cam3', 'cam4'])
        self.cache_episodes, self.cache_videos = max(1, cache_episodes), max(1, cache_videos)
        self.info = json.loads((self.dataset_path / 'meta/info.json').read_text())
        self.tasks = {int(e['task_index']): e['task'].strip() for e in map(json.loads, (self.dataset_path / 'meta/tasks.jsonl').read_text().splitlines())}
        adaptation = self.dataset_path / 'meta/z02_adaptation.json'
        self.adaptation = json.loads(adaptation.read_text()) if adaptation.exists() else None
        if (include_state or include_actions) and (self.adaptation is None or self.adaptation['joint_indices'] is None):
            raise ValueError(f'{self.dataset_path.name}: joint mapping unverified; use image/text only (include_state=False, include_actions=False)')
        self.joint_indices = self.adaptation['joint_indices'] if self.adaptation else None
        self.return_scale = return_scale
        self.contract = None
        if normalize_returns:
            if return_scale is None:
                self.contract = load_normalization(self.dataset_path, tag)
                self.return_scale = float(self.contract['return_scale'])
            if not np.isfinite(self.return_scale) or self.return_scale <= 0:
                raise ValueError('Return scale must be positive and finite')
        self.returns_data = load_returns_sidecar(self.dataset_path, tag)
        if self.returns_data is None:
            raise FileNotFoundError(f'Missing returns sidecar for tag {tag}: {self.dataset_path}')
        selected = read_split(split_file, self.dataset_path.name) if split_file is not None else None
        if episodes is not None:
            eps = set(map(int, episodes))
            selected = eps if selected is None else selected & eps
        self.parquet_files, self.episode_ids, self.index_mapping = [], [], []
        self._paths_by_episode = {}
        self._task_indices = {}
        found = set()
        excluded = set(self.adaptation.get('excluded_episodes', [])) if self.adaptation else set()
        for path in sorted((self.dataset_path / 'data').rglob('episode_*.parquet')):
            ids = pq.read_table(path, columns=['episode_index', 'frame_index', 'task_index']).to_pandas()
            if len(ids) == 0 or ids.episode_index.nunique() != 1:
                raise ValueError(f'Invalid episode: {path}')
            ep = int(ids.episode_index.iloc[0]); found.add(ep)
            if selected is not None and ep not in selected:
                continue
            if len(ids) < min_episode_steps or ep in excluded:
                continue
            if not np.array_equal(ids.frame_index, np.arange(len(ids))):
                raise ValueError(f'Non-contiguous frame indices: {path}')
            labels = self.returns_data.get(ep)
            if labels is None or len(labels['return']) != len(ids):
                raise ValueError(f'Incomplete labels for {path}')
            if not np.isfinite(labels['return']).all():
                raise ValueError(f'Non-finite labels for {path}')
            if normalize_returns and (np.min(labels['return']) < -self.return_scale or np.max(labels['return']) > 0):
                raise ValueError(f'Labels outside [-return_scale,0]: {path}; use matching contract')
            for cam in self.cameras:
                if not self.video_path(cam, ep).is_file():
                    raise FileNotFoundError(self.video_path(cam, ep))
            self.parquet_files.append(path); self.episode_ids.append(ep)
            self._paths_by_episode[ep] = path
            self._task_indices[ep] = ids.task_index.to_numpy(dtype=np.int32)
            self.index_mapping.extend((ep, fr) for fr in range(len(ids)))
        if selected is not None and selected - found:
            raise ValueError(f'Split references missing episodes: {selected - found}')
        if max_samples is not None:
            self.index_mapping = self.index_mapping[:max_samples]
        if not self.index_mapping:
            raise ValueError(f'No eligible samples: {self.dataset_path}')
        self.index_mapping = np.asarray(self.index_mapping, dtype=np.int64)
        # Match the processor bundled with the SigLIP checkpoint; no ImageNet normalization.
        mean, std = [0.5] * 3, [0.5] * 3
        if image_processor_path:
            processor = json.loads((Path(image_processor_path) / 'preprocessor_config.json').read_text())
            mean, std = processor['image_mean'], processor['image_std']
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    def video_path(self, cam, ep):
        return self.dataset_path / self.info['video_path'].format(
            episode_chunk=ep // self.info.get('chunks_size', 1000), episode_index=ep,
            video_key=f'observation.images.{cam}')

    def _reset_worker_cache(self):
        if os.getpid() != self._pid:
            self.close()
            self._episode_cache.clear()
            self._pid = os.getpid()

    def _episode(self, ep):
        self._reset_worker_cache()
        if ep not in self._episode_cache:
            self._episode_cache[ep] = pq.read_table(self._paths_by_episode[ep]).to_pandas()
            if len(self._episode_cache) > self.cache_episodes:
                self._episode_cache.popitem(last=False)
        self._episode_cache.move_to_end(ep)
        return self._episode_cache[ep]

    def _read_video_frame(self, cam, ep, fr):
        key = (cam, ep)
        if key not in self._video_caps:
            self._video_caps[key] = cv2.VideoCapture(str(self.video_path(cam, ep)), cv2.CAP_FFMPEG,
                                                       [cv2.CAP_PROP_N_THREADS, self.decoder_threads])
            if len(self._video_caps) > self.cache_videos:
                self._video_caps.popitem(last=False)[1].release()
        self._video_caps.move_to_end(key)
        cap = self._video_caps[key]
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != fr:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f'Cannot decode {self.video_path(cam, ep)} frame {fr}')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _joints(self, values):
        a = np.asarray(values, dtype=np.float32)
        if a.shape[-1] != self.adaptation['raw_joint_dim'] or not np.isfinite(a).all():
            raise ValueError('Raw joint vector disagrees with adaptation metadata')
        a = a[..., self.joint_indices]
        return torch.from_numpy(np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, self.action_dim - 22)]))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        ep, fr = map(int, self.index_mapping[idx])
        self._reset_worker_cache()
        if self.include_state or self.include_actions:
            df = self._episode(ep); row = df.iloc[fr]
        value = float(self.returns_data[ep]['return'][fr])
        sample = {
            'images': {f'observation.images.{cam}': self.image_transform(self._read_video_frame(cam, ep, fr)) for cam in self.cameras},
            'image_masks': {f'observation.images.{cam}': True for cam in self.cameras},
            'prompt': self.tasks[int(self._task_indices[ep][fr])],
            'target_values': value / self.return_scale if self.normalize_returns else value,
            'dataset_id': self.dataset_path.name, 'episode_index': ep, 'frame_index': fr,
        }
        if self.include_state:
            sample['observation/state'] = self._joints(row['observation.joint_positions'])
        if self.include_actions:
            indices = np.minimum(np.arange(fr, fr + self.action_horizon), len(df) - 1)
            sample['actions'] = self._joints(np.stack(df['action.joint_positions'].iloc[indices]))
            sample['action_is_pad'] = torch.from_numpy(np.arange(fr, fr + self.action_horizon) >= len(df))
        return sample

    def close(self):
        for cap in getattr(self, '_video_caps', {}).values():
            cap.release()
        if hasattr(self, '_video_caps'):
            self._video_caps.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_video_caps'] = OrderedDict(); state['_episode_cache'] = OrderedDict()
        return state

    def __del__(self):
        self.close()


def create_simple_dataloader(dataset_path, robot_type='z02', tag='z02_fail2000_v1',
                             batch_size=32, num_workers=4, max_samples=None, cameras=None, **kwargs):
    ds = SimpleValueDataset(dataset_path, robot_type=robot_type, tag=tag, max_samples=max_samples, cameras=cameras, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
