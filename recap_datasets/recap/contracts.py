"""Shared, explicit contracts for labels, returns and dataset splits."""
from pathlib import Path
import json
import hashlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_split(path, dataset_name):
    entries = json.loads(Path(path).read_text())['episodes']
    keys = [(e['dataset'], int(e['episode_index'])) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError(f'Duplicate episode in split: {path}')
    return {ep for ds, ep in keys if ds == dataset_name}


def resolve_outcome(frame, dataset_type='rollout', override=None):
    if override is not None:
        if not isinstance(override, bool):
            raise ValueError('outcome override must be a boolean')
        return override
    if 'is_success' in frame:
        value = frame['is_success'].iloc[-1]
        if value not in (0, 1, False, True):
            raise ValueError(f'Invalid is_success: {value}')
        return bool(value)
    if 'reward' in frame:
        terminal = float(frame['reward'].iloc[-1])
        if terminal == 0:
            return True
        if terminal < -1 and np.isfinite(terminal):
            return False
        raise ValueError(f'Ambiguous terminal reward: {terminal}')
    if dataset_type == 'sft':
        return True
    raise ValueError('Missing outcome: provide an explicit override or terminal reward')


def episode_returns(length, success, gamma, failure_reward):
    if length < 1 or not 0 <= gamma <= 1 or not np.isfinite(failure_reward) or failure_reward >= 0:
        raise ValueError('Invalid length, discount or failure penalty')
    rewards = np.full(length, -1., dtype=np.float32)
    rewards[-1] = 0. if success else failure_reward
    returns = rewards.copy()
    for t in range(length - 2, -1, -1):
        returns[t] += gamma * returns[t + 1]
    return returns, rewards


def load_normalization(dataset_path, tag):
    path = Path(dataset_path) / 'meta' / f'returns_{tag}.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing shared return contract: {path}; run scripts/adapt_z02.py --all')
    contract = json.loads(path.read_text())
    if contract['tag'] != tag or contract['return_scale'] <= 0:
        raise ValueError(f'Invalid return contract: {path}')
    sidecar = path.with_suffix('.parquet')
    if not sidecar.exists() or hashlib.sha256(sidecar.read_bytes()).hexdigest() != contract['sidecar_sha256']:
        raise ValueError(f'Return sidecar was changed without its contract: {sidecar}')
    return contract
