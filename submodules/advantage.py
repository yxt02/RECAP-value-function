"""Trajectory-local N-step scores; rewards and values share one return scale."""
import numpy as np


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
