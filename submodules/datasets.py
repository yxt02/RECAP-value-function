"""Local z02 value data: strict labels, shared scale, split filtering and video I/O."""
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .contracts import load_normalization, read_split

logger = logging.getLogger(__name__)

MEAN = STD = [0.5] * 3  # Match the processor bundled with SigLIP; no ImageNet normalization.
CONTRACT_KEYS = ('tag', 'gamma', 'failure_reward', 'return_scale', 'max_episode_steps')


def load_returns_sidecar(dataset_path, tag=None):
    """Read `meta/returns[_tag].parquet` as {episode_index: {column: array}}."""
    path = Path(dataset_path) / 'meta' / (f'returns_{tag}.parquet' if tag else 'returns.parquet')
    if not path.exists():
        logger.warning('Returns sidecar not found: %s', path)
        return None
    frame = pq.read_table(path).to_pandas()
    if frame.duplicated(['episode_index', 'frame_index']).any():
        raise ValueError(f'Duplicate labels: {path}')
    if not np.isfinite(frame[['return', 'reward']].to_numpy()).all():
        raise ValueError(f'Non-finite labels: {path}')
    sidecar = {}
    for episode, group in frame.groupby('episode_index'):
        group = group.sort_values('frame_index')
        if not np.array_equal(group.frame_index, np.arange(len(group))):
            raise ValueError(f'Missing/non-contiguous frames in {path}, episode {episode}')
        sidecar[int(episode)] = {key: group[key].to_numpy() for key in ('return', 'reward', 'frame_index')}
        sidecar[int(episode)]['prompt'] = group['prompt'].to_numpy() if 'prompt' in group else None
    return sidecar


class SimpleValueDataset(Dataset):
    """One z02 dataset directory, filtered to a split, with per-frame value labels."""

    def __init__(self, dataset_path, robot_type='z02', tag='z02_fail2000_v1',
                 action_horizon=10, action_dim=32, max_samples=None,
                 normalize_returns=True, image_size=224, cameras=None,
                 split_file=None, episodes=None, return_scale=None,
                 include_state=True, include_actions=True, min_episode_steps=2,
                 image_processor_path=None, cache_episodes=4, cache_videos=6, decoder_threads=1):
        self._video_caps, self._episode_cache = OrderedDict(), OrderedDict()
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
        self.tasks = {int(e['task_index']): e['task'].strip()
                      for e in map(json.loads, (self.dataset_path / 'meta/tasks.jsonl').read_text().splitlines())}
        adaptation = self.dataset_path / 'meta/z02_adaptation.json'
        self.adaptation = json.loads(adaptation.read_text()) if adaptation.exists() else None
        if (include_state or include_actions) and (self.adaptation is None or self.adaptation['joint_indices'] is None):
            raise ValueError(f'{self.dataset_path.name}: joint mapping unverified; '
                             'use image/text only (include_state=False, include_actions=False)')
        self.joint_indices = self.adaptation['joint_indices'] if self.adaptation else None
        self.return_scale, self.contract = return_scale, None
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
            episodes = set(map(int, episodes))
            selected = episodes if selected is None else selected & episodes
        self.parquet_files, self.episode_ids, self.index_mapping = [], [], []
        self._paths_by_episode, self._task_indices = {}, {}
        found = set()
        excluded = set(self.adaptation.get('excluded_episodes', [])) if self.adaptation else set()
        for path in sorted((self.dataset_path / 'data').rglob('episode_*.parquet')):
            ids = pq.read_table(path, columns=['episode_index', 'frame_index', 'task_index']).to_pandas()
            if len(ids) == 0 or ids.episode_index.nunique() != 1:
                raise ValueError(f'Invalid episode: {path}')
            episode = int(ids.episode_index.iloc[0]); found.add(episode)
            if selected is not None and episode not in selected:
                continue
            if len(ids) < min_episode_steps or episode in excluded:
                continue
            if not np.array_equal(ids.frame_index, np.arange(len(ids))):
                raise ValueError(f'Non-contiguous frame indices: {path}')
            labels = self.returns_data.get(episode)
            if labels is None or len(labels['return']) != len(ids):
                raise ValueError(f'Incomplete labels for {path}')
            if not np.isfinite(labels['return']).all():
                raise ValueError(f'Non-finite labels for {path}')
            if normalize_returns and (np.min(labels['return']) < -self.return_scale or np.max(labels['return']) > 0):
                raise ValueError(f'Labels outside [-return_scale,0]: {path}; use matching contract')
            for cam in self.cameras:
                if not self.video_path(cam, episode).is_file():
                    raise FileNotFoundError(self.video_path(cam, episode))
            self.parquet_files.append(path); self.episode_ids.append(episode)
            self._paths_by_episode[episode] = path
            self._task_indices[episode] = ids.task_index.to_numpy(dtype=np.int32)
            self.index_mapping.extend((episode, frame) for frame in range(len(ids)))
        if selected is not None and selected - found:
            raise ValueError(f'Split references missing episodes: {selected - found}')
        if max_samples is not None:
            self.index_mapping = self.index_mapping[:max_samples]
        if not self.index_mapping:
            raise ValueError(f'No eligible samples: {self.dataset_path}')
        self.index_mapping = np.asarray(self.index_mapping, dtype=np.int64)
        mean, std = MEAN, STD
        if image_processor_path:
            processor = json.loads((Path(image_processor_path) / 'preprocessor_config.json').read_text())
            mean, std = processor['image_mean'], processor['image_std']
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    def video_path(self, cam, episode):
        return self.dataset_path / self.info['video_path'].format(
            episode_chunk=episode // self.info.get('chunks_size', 1000), episode_index=episode,
            video_key=f'observation.images.{cam}')

    def _reset_worker_cache(self):
        if os.getpid() != self._pid:
            self.close()
            self._episode_cache.clear()
            self._pid = os.getpid()

    def _episode(self, episode):
        self._reset_worker_cache()
        if episode not in self._episode_cache:
            self._episode_cache[episode] = pq.read_table(self._paths_by_episode[episode]).to_pandas()
            if len(self._episode_cache) > self.cache_episodes:
                self._episode_cache.popitem(last=False)
        self._episode_cache.move_to_end(episode)
        return self._episode_cache[episode]

    def _read_video_frame(self, cam, episode, frame):
        key = (cam, episode)
        if key not in self._video_caps:
            self._video_caps[key] = cv2.VideoCapture(str(self.video_path(cam, episode)), cv2.CAP_FFMPEG,
                                                      [cv2.CAP_PROP_N_THREADS, self.decoder_threads])
            if len(self._video_caps) > self.cache_videos:
                self._video_caps.popitem(last=False)[1].release()
        self._video_caps.move_to_end(key)
        capture = self._video_caps[key]
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(f'Cannot decode {self.video_path(cam, episode)} frame {frame}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _joints(self, values):
        joints = np.asarray(values, dtype=np.float32)
        if joints.shape[-1] != self.adaptation['raw_joint_dim'] or not np.isfinite(joints).all():
            raise ValueError('Raw joint vector disagrees with adaptation metadata')
        joints = joints[..., self.joint_indices]
        return torch.from_numpy(np.pad(joints, [(0, 0)] * (joints.ndim - 1) + [(0, self.action_dim - 22)]))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):
        episode, frame = map(int, self.index_mapping[index])
        self._reset_worker_cache()
        if self.include_state or self.include_actions:
            episode_data = self._episode(episode); row = episode_data.iloc[frame]
        value = float(self.returns_data[episode]['return'][frame])
        sample = {
            'images': {f'observation.images.{cam}': self.image_transform(self._read_video_frame(cam, episode, frame))
                       for cam in self.cameras},
            'image_masks': {f'observation.images.{cam}': True for cam in self.cameras},
            'prompt': self.tasks[int(self._task_indices[episode][frame])],
            'target_values': value / self.return_scale if self.normalize_returns else value,
            'dataset_id': self.dataset_path.name, 'episode_index': episode, 'frame_index': frame,
        }
        if self.include_state:
            sample['observation/state'] = self._joints(row['observation.joint_positions'])
        if self.include_actions:
            indices = np.minimum(np.arange(frame, frame + self.action_horizon), len(episode_data) - 1)
            sample['actions'] = self._joints(np.stack(episode_data['action.joint_positions'].iloc[indices]))
            sample['action_is_pad'] = torch.from_numpy(np.arange(frame, frame + self.action_horizon) >= len(episode_data))
        return sample

    def close(self):
        for capture in getattr(self, '_video_caps', {}).values():
            capture.release()
        if hasattr(self, '_video_caps'):
            self._video_caps.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_video_caps'], state['_episode_cache'] = OrderedDict(), OrderedDict()
        return state

    def __del__(self):
        self.close()
