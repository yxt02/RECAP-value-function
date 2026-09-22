"""Trajectory-local N-step scores; rewards and values share one return scale."""
import numpy as np
import pandas as pd

ADVANTAGE_TABLE_COLUMNS = ['dataset_id', 'episode_index', 'frame_index', 'timestamp',
                           'split', 'horizon', 'threshold', 'advantage_continuous',
                           'advantage', 'advantage_forced']
ADVANTAGE_TABLE_KEYS = ['dataset_id', 'episode_index', 'frame_index', 'horizon']

# RECAP binarizes advantage with a task-level threshold and forces corrections positive.
DEFAULT_PERCENTILE = 30.


def compute_advantage(values, rewards, horizon=50, scale=4000., gamma=1.):
    values, rewards = np.asarray(values, dtype=float), np.asarray(rewards, dtype=float)
    if values.ndim != 1 or len(values) == 0 or rewards.shape != values.shape:
        raise ValueError('Expected aligned nonempty 1-D values and rewards')
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise ValueError('Non-finite values/rewards')
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    if not np.isfinite(scale) or scale <= 0 or not 0 <= gamma <= 1:
        raise ValueError('Invalid scale/gamma')
    n = len(values)
    k = np.minimum(horizon, n - np.arange(n))
    reward_sum = np.zeros(n)
    for step in range(min(horizon, n)):
        reward_sum[:n-step] += gamma**step * rewards[step:]
    next_index = np.arange(n) + k
    next_value = np.zeros(n)
    nonterminal = next_index < n
    next_value[nonterminal] = values[next_index[nonterminal]]
    score = reward_sum / scale + gamma**k * next_value - values
    return dict(effective_horizon=k, reward_sum_raw=reward_sum,
                value_next=next_value, terminal_reached=~nonterminal,
                advantage_continuous=score)


def label_scores(scores, threshold=0.):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Non-finite scores/threshold')
    return np.where(scores > threshold, 'advantage', 'disadvantage')


def value_percentile_threshold(values, percentile=DEFAULT_PERCENTILE):
    """Task-level improvement threshold epsilon, taken from predicted values.

    Mirrors the RECAP rule: epsilon is the configured percentile of the value
    function's own predictions for the task, so the cut adapts per task instead
    of being a fixed number. Multi-task runs must call this once per task.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError('Expected a nonempty 1-D value array')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite values')
    if not np.isfinite(percentile) or not 0. <= percentile <= 100.:
        raise ValueError('percentile must lie in [0, 100]')
    return float(np.percentile(values, percentile, method='linear'))


def label_improvement(advantages, threshold, intervention=None):
    """Binarized improvement indicator and the frames forced positive.

    I = 1[A > threshold], with human-intervention frames forced to True: an
    expert correction during an autonomous rollout counts as an improvement
    whatever the advantage says. Missing (NaN) flags never force a label, so an
    absent intervention column cannot silently turn into "corrected".
    """
    advantages = np.asarray(advantages, dtype=float)
    if advantages.ndim != 1 or len(advantages) == 0:
        raise ValueError('Expected a nonempty 1-D advantage array')
    if not np.isfinite(advantages).all():
        raise ValueError('Non-finite advantages')
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    forced = np.zeros(len(advantages), dtype=bool)
    if intervention is not None:
        flags = np.asarray(intervention, dtype=float)
        if flags.shape != advantages.shape:
            raise ValueError('Intervention flags must align with advantages')
        known = ~np.isnan(flags)
        if not np.isin(flags[known], [0., 1.]).all():
            raise ValueError('Intervention flags must be 0/1 or NaN')
        forced = known & (flags == 1.)
    return (advantages > threshold) | forced, forced


def export_advantage_table(scores):
    """Project scored frames into timestep-level RECAP metadata; never mutates input."""
    if not isinstance(scores, pd.DataFrame):
        raise ValueError('scores must be a pandas DataFrame')
    required = set(ADVANTAGE_TABLE_KEYS) | {'timestamp', 'split', 'threshold',
                                            'advantage_continuous', 'is_advantage',
                                            'advantage_forced'}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f'Missing scores columns: {missing}')
    if len(scores) == 0:
        raise ValueError('Cannot export an empty scores table')
    if scores.duplicated(ADVANTAGE_TABLE_KEYS).any():
        raise ValueError('Duplicate frame/horizon keys in scores')
    continuous = pd.to_numeric(scores['advantage_continuous'], errors='coerce').to_numpy(float)
    threshold = pd.to_numeric(scores['threshold'], errors='coerce').to_numpy(float)
    if not np.isfinite(continuous).all() or not np.isfinite(threshold).all():
        raise ValueError('Non-finite advantage_continuous/threshold')
    if not np.isfinite(pd.to_numeric(scores['timestamp'], errors='coerce')).all():
        raise ValueError('Non-finite timestamps')
    if not scores['is_advantage'].isin([True, False, 0, 1]).all():
        raise ValueError('is_advantage must be boolean')
    if not scores['advantage_forced'].isin([True, False, 0, 1]).all():
        raise ValueError('advantage_forced must be boolean')
    advantage = scores['is_advantage'].astype(bool).to_numpy()
    forced = scores['advantage_forced'].astype(bool).to_numpy()
    if not np.array_equal(advantage, (continuous > threshold) | forced):
        raise ValueError('is_advantage disagrees with advantage_continuous > threshold unless forced')
    table = scores[[c for c in ADVANTAGE_TABLE_COLUMNS if c not in ('advantage', 'advantage_forced')]].copy()
    table['advantage'] = advantage
    table['advantage_forced'] = forced
    return table.sort_values(ADVANTAGE_TABLE_KEYS, kind='mergesort').reset_index(drop=True)


def intervention_windows(times, scores, flags, window_seconds=1., points=61):
    """Descriptive event windows. Missing flags and left-censored starts are excluded."""
    times, scores = np.asarray(times), np.asarray(scores)
    if times.ndim != 1 or not len(times) or scores.shape != times.shape:
        raise ValueError('Expected aligned nonempty time and score arrays')
    if not np.isfinite(times).all() or not np.isfinite(scores).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Times must be finite and strictly increasing; scores must be finite')
    if not np.isfinite(window_seconds) or window_seconds <= 0 or points < 2:
        raise ValueError('Invalid intervention window')
    grid = np.linspace(-window_seconds, window_seconds, points)
    if flags is None:
        return grid, np.empty((0, points))
    flags = np.asarray(flags)
    if flags.shape != times.shape or not np.isin(flags, [0, 1]).all():
        raise ValueError('Intervention flags must be aligned binary values')
    starts = np.flatnonzero((flags[1:] == 1) & (flags[:-1] == 0)) + 1
    rows = [np.interp(times[i] + grid, times, scores) for i in starts
            if times[i] - window_seconds >= times[0] and times[i] + window_seconds <= times[-1]]
    return grid, np.asarray(rows).reshape(-1, points)
