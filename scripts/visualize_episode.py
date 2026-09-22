"""Four-panel line chart for one scored episode: values, intervention, advantage, labels."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submodules  # noqa: F401

import numpy as np
import pandas as pd

from submodules.contracts import resolve_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='calculate_advantage output directory')
    parser.add_argument('--dataset', required=True, help='dataset_id, e.g. 2026.09.15_2')
    parser.add_argument('--episode', type=int, required=True, help='episode_index')
    parser.add_argument('--horizon', type=int, default=50, help='advantage horizon to plot')
    parser.add_argument('--raw-root', default='data/raw', help='raw datasets root for ground-truth returns')
    parser.add_argument('--output', help='PNG path; default <result>/episode-<dataset>-<ep>.png')
    args = parser.parse_args(argv)

    from submodules.evaluation_workflow import configure_plots
    from submodules.datasets import load_returns_sidecar

    result = resolve_path(args.result)
    scores = pd.read_parquet(result / 'scores.parquet')
    block = scores[(scores.dataset_id == args.dataset)
                   & (scores.episode_index == args.episode)
                   & (scores.horizon == args.horizon)].sort_values('frame_index')
    if block.empty:
        raise SystemExit(f'No rows for {args.dataset}:{args.episode} horizon {args.horizon} in {result}')

    raw = resolve_path(args.raw_root) / args.dataset
    sidecar = load_returns_sidecar(raw, tag='z02_fail2000_v1')
    if sidecar is None or args.episode not in sidecar:
        raise SystemExit(f'Missing returns sidecar for {args.dataset}:{args.episode}')
    labels = sidecar[args.episode]
    if len(labels['return']) != len(block):
        raise SystemExit(f'Frame count mismatch: scores {len(block)} vs returns {len(labels["return"])}')
    import json
    contract = json.loads((raw / 'meta' / 'returns_z02_fail2000_v1.json').read_text())
    scale = float(contract['return_scale'])
    actual = labels['return'].astype(float) / scale

    t = block.timestamp.to_numpy(float)
    model = block.value.to_numpy(float)
    adv = block.advantage_continuous.to_numpy(float)
    thr = float(block.threshold.iloc[0])
    interv = block.intervention.to_numpy(float)
    forced = block.advantage_forced.to_numpy(bool)
    positive = block.is_advantage.to_numpy(bool)
    score_positive = positive & ~forced

    plt = configure_plots()
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    if np.any(interv == 1):
        for ax in axes:
            ax.axvspan(t[interv == 1].min(), t[interv == 1].max(),
                       color='#f0b429', alpha=.18, lw=0, zorder=0)

    axes[0].plot(t, actual, color='#d26121', lw=1.4, label='实际 value（MC 回报 / scale）')
    axes[0].plot(t, model, color='#2463b9', lw=1.4, label='模型估计 value')
    corr = float(np.corrcoef(actual, model)[0, 1])
    mae = float(np.abs(actual - model).mean())
    axes[0].set_ylabel('价值')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].set_title(f'相关系数 {corr:.3f} · MAE {mae:.4f}', fontsize=9, loc='right')

    axes[1].step(t, interv, where='post', color='#5c6b7d', lw=1.2, label='人工接管')
    axes[1].fill_between(t, 0, interv, step='post', color='#f0b429', alpha=.35)
    axes[1].set_ylabel('接管')
    axes[1].set_yticks([0, 1])
    axes[1].legend(loc='best', fontsize=9)

    axes[2].plot(t, adv, color='#2463b9', lw=1.2, label=f'{args.horizon} 帧优势 A(t)')
    axes[2].axhline(thr, color='black', ls='--', lw=1, label=f'阈值 {thr:.5f}（train 70th pct）')
    axes[2].fill_between(t, thr, adv, where=positive, color='#098779', alpha=.25, label='A ≥ 阈值')
    axes[2].fill_between(t, thr, adv, where=~positive, color='#d64545', alpha=.15, label='A < 阈值')
    axes[2].set_ylabel('优势')
    axes[2].legend(loc='best', fontsize=9)

    axes[3].fill_between(t, 0, 1, where=score_positive, step='post',
                         color='#098779', alpha=.75, label='标签 advantage（分数）')
    axes[3].fill_between(t, 0, 1, where=forced & positive, step='post',
                         color='#f0b429', alpha=.85, label='标签 advantage（接管强制）')
    axes[3].fill_between(t, 0, 1, where=~positive, step='post',
                         color='#d64545', alpha=.2, label='标签 disadvantage')
    axes[3].set_ylabel('二值标签')
    axes[3].set_yticks([0, 1])
    axes[3].set_xlabel('时间 / 秒')
    axes[3].legend(loc='upper right', fontsize=9, ncol=3)

    n_score = int(score_positive.sum())
    n_forced = int((forced & positive).sum())
    n_neg = int((~positive).sum())
    fig.suptitle(
        f'{args.dataset} / episode {args.episode} · horizon {args.horizon} · '
        f'{"成功" if bool(block.success.iloc[0]) else "失败"} · {block.split.iloc[0]} · '
        f'{len(block)} 帧 · 正例 {n_score} 分数 + {n_forced} 强制 / 负例 {n_neg}',
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = resolve_path(args.output) if args.output else result / f'episode-{args.dataset}-{args.episode}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
