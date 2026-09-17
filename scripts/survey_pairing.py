#!/usr/bin/env python3
"""Same-session matched-pair feasibility survey (read-only).

Uses only:
  - data/splits/*.json              episode inventory and outcomes
  - data/raw/*/data/chunk-*/...     gripper binary + right-arm state
  - artifacts/.../predictions.npz   existing checkpoint predictions (test split only)

Question this answers: for the planned audit, how many *statistically usable*
same-session good/bad pairs actually exist?

A usable pair must
  1. come from the same collection session (so the batch shortcut is removed),
  2. share the same task phase (gripper open/closed), and
  3. be close in observable robot state, so the comparison is not just a re-read
     of the clock.

Calibration point: if pairing really removes the time/batch confound, the
(batch + frame-index) control must lose its accuracy on the matched pairs and
fall back to ~0.5. Its accuracy is therefore the check that the pairing worked.

Note on matching tightness: matching state *exactly* is degenerate. Under a
Markov assumption V(s) is identical for the same state, so "good > bad" cannot
hold there and 0.5 would be the correct score. The useful band is coarse phase
plus similar-but-not-identical state.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ARM = list(range(8, 15))        # right arm, 7 DOF; always live in every session
HAND = 15                       # right hand binary; live in 3 of 4 sessions
TAIL = 60                       # frames excluded near the terminal (outcome visible)
THRESHOLDS = [0.25, 0.50, 1.00, 1.50, 2.00, 3.00]
MIN_MATCHES = 20                # matched frames required for an episode pair to count
STRIDE = 5                      # 6 Hz; keeps the O(n^2) sweep cheap
BOOTSTRAP = 4000
SEED = 20260917


def episode_path(dataset, episode_index):
    root = ROOT / 'data/raw' / dataset
    hits = sorted(root.glob(f'data/chunk-*/episode_{episode_index:06d}.parquet'))
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected exactly one file for {dataset}/{episode_index}, got {hits}')
    return hits[0]


def load_state(dataset, episode_index):
    """Return (gripper binary, right-arm joints) for one episode."""
    table = pq.read_table(episode_path(dataset, episode_index),
                          columns=['observation.joint_positions'])
    joints = np.stack(table['observation.joint_positions'].to_numpy()).astype(np.float64)
    return joints[:, HAND].copy(), joints[:, ARM].copy()


def inventory():
    splits = {s: json.loads((ROOT / f'data/splits/{s}.json').read_text())['episodes']
              for s in ('train', 'val', 'test')}
    rows = []
    for split, entries in splits.items():
        for entry in entries:
            rows.append(dict(split=split, dataset=entry['dataset'],
                             episode_index=int(entry['episode_index']),
                             frames=int(entry['total_frames']), success=bool(entry['is_success'])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/analysis/pairing_survey.json')
    args = parser.parse_args()

    rows = inventory()
    sessions = sorted({r['dataset'] for r in rows})
    print(f'Loaded {len(rows)} episodes across {len(sessions)} sessions\n')

    # ---- load state once -------------------------------------------------
    state = {}
    for r in rows:
        hand, arm = load_state(r['dataset'], r['episode_index'])
        if len(hand) != r['frames']:
            raise ValueError(f"Frame count mismatch: {r['dataset']}/{r['episode_index']}")
        state[(r['dataset'], r['episode_index'])] = (hand, arm)

    # ---- global normalization over the 7 right-arm joints ----------------
    pooled = np.concatenate([state[(r['dataset'], r['episode_index'])][1] for r in rows])
    mean, std = pooled.mean(0), pooled.std(0)
    if not np.all(std > 0):
        raise ValueError('A right-arm joint is constant across all data')
    print('right-arm joint scale used for matching (mean, std):')
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f'   joint {ARM[i]:2d}  mean {m:+.3f}  std {s:.3f}')

    # ---- section 1: inventory + phase availability -----------------------
    print('\n=== SECTION 1: per-session inventory and gripper-phase channel ===')
    print(f'{"session":16} {"eps":>5} {"ok":>4} {"fail":>5} {"pairs":>6} '
          f'{"hand live":>10} {"hand-closed frac":>17}')
    section1 = {}
    for ds in sessions:
        sub = [r for r in rows if r['dataset'] == ds]
        ok = sum(r['success'] for r in sub)
        fail = len(sub) - ok
        hands = [state[(ds, r['episode_index'])][0] for r in sub]
        live = sum(1 for h in hands if h.std() > 1e-9)
        closed = float(np.mean([h.mean() for h in hands]))
        section1[ds] = dict(episodes=len(sub), success=ok, failure=fail,
                            episode_pairs=ok * fail, hand_channel_live=live, hand_closed_fraction=closed)
        print(f'{ds:16} {len(sub):5d} {ok:4d} {fail:5d} {ok*fail:6d} {live:10d} {closed:17.3f}')
    total_pairs = sum(v['episode_pairs'] for v in section1.values())
    print(f'{"TOTAL":16} {len(rows):5d} '
          f'{sum(v["success"] for v in section1.values()):4d} '
          f'{sum(v["failure"] for v in section1.values()):5d} {total_pairs:6d}')
    print('   -> upper bound on same-session good/bad episode pairs, before any matching')

    # ---- section 2: pairing feasibility sweep ----------------------------
    print('\n=== SECTION 2: pairing feasibility (phase + normalized state distance) ===')
    print('Distance is Euclidean over 7 normalized joints; an independent pair of')
    print('standard normals would sit near 3.74, so ~1.0 is tight and ~3.0 is loose.')
    sweep = {}
    for ds in sessions:
        sub = [r for r in rows if r['dataset'] == ds]
        ok = [r for r in sub if r['success']]
        fail = [r for r in sub if not r['success']]
        if not ok or not fail:
            print(f'\n{ds}: single-outcome session ({len(ok)} ok / {len(fail)} fail) '
                  '-> no within-session pair is possible at any threshold')
            sweep[ds] = dict(ok=len(ok), fail=len(fail), single_outcome=True)
            continue
        print(f'\n{ds}: {len(ok)} ok x {len(fail)} fail = {len(ok)*len(fail)} candidate episode pairs')
        print(f'{"thresh":>7} {"ep pairs >=min":>15} {"matched frames":>15} {"median/frame pair":>18}')
        by_threshold = {}
        for thresh in THRESHOLDS:
            ep_pairs = 0
            frame_pairs = 0
            per_pair = []
            for a in ok:
                ha, aa = state[(ds, a['episode_index'])]
                aa = (aa - mean) / std
                for b in fail:
                    hb, ab = state[(ds, b['episode_index'])]
                    ab = (ab - mean) / std
                    ia = np.arange(0, max(a['frames'] - TAIL, 0), STRIDE)
                    ib = np.arange(0, max(b['frames'] - TAIL, 0), STRIDE)
                    if len(ia) == 0 or len(ib) == 0:
                        continue
                    d = np.sqrt(np.maximum(
                        (aa[ia] ** 2).sum(1)[:, None] + (ab[ib] ** 2).sum(1)[None, :]
                        - 2.0 * aa[ia] @ ab[ib].T, 0.0))
                    phase = (ha[ia][:, None] == hb[ib][None, :])
                    hits = int(((d <= thresh) & phase).sum())
                    if hits >= MIN_MATCHES:
                        ep_pairs += 1
                        frame_pairs += hits
                        per_pair.append(hits)
            by_threshold[str(thresh)] = dict(episode_pairs=ep_pairs, frame_pairs=frame_pairs,
                                             median_frames_per_pair=float(np.median(per_pair)) if per_pair else 0.0)
            print(f'{thresh:7.2f} {ep_pairs:15d} {frame_pairs:15d} '
                  f'{(np.median(per_pair) if per_pair else 0):18.0f}')
        sweep[ds] = dict(ok=len(ok), fail=len(fail), single_outcome=False,
                         candidate_episode_pairs=len(ok) * len(fail), thresholds=by_threshold)

    # ---- section 3: recovery / release event inventory -------------------
    print('\n=== SECTION 3: gripper event inventory (candidate error/recovery events) ===')
    print('A 1->0 transition is a release or drop; a following 0->1 is a re-grasp.')
    print(f'{"session":16} {"eps":>5} {"releases":>9} {"re-grasps":>10} {"eps w/ release":>15}')
    section3 = {}
    for ds in sessions:
        sub = [r for r in rows if r['dataset'] == ds]
        releases = regrasps = 0
        eps_with = 0
        for r in sub:
            hand = state[(ds, r['episode_index'])][0]
            d = np.diff(hand)
            down = np.flatnonzero(d < 0)
            up = np.flatnonzero(d > 0)
            releases += len(down)
            eps_with += int(len(down) > 0)
            for t in down:
                if np.any(up > t):
                    regrasps += 1
        section3[ds] = dict(releases=int(releases), regrasps=int(regrasps), episodes_with_release=eps_with)
        print(f'{ds:16} {len(sub):5d} {releases:9d} {regrasps:10d} {eps_with:15d}')
    print('   -> 09.08 has a constant-0 channel, so it can contribute no such event')

    # ---- section 4: accuracy on matched pairs, where predictions exist ----
    print('\n=== SECTION 4: accuracy on matched pairs (test split, current checkpoint) ===')
    audits = sorted((ROOT / 'artifacts/evaluation').glob('checkpoint-*-test'))
    if not audits:
        print('No evaluation artifacts found; skipping.')
        section4 = None
    else:
        audit = audits[-1]
        data = np.load(audit / 'predictions.npz')
        eps = json.loads((audit / 'episodes.json').read_text())
        prediction, control = data['prediction'], data['dataset_frame_linear']
        print(f'Using {audit.relative_to(ROOT)}')
        section4 = {}
        for ds in sessions:
            sub = [e for e in eps if e['dataset'] == ds]
            ok = [e for e in sub if e['success']]
            fail = [e for e in sub if not e['success']]
            if not ok or not fail:
                print(f'\n{ds}: single outcome in test ({len(ok)} ok / {len(fail)} fail) -> no pair')
                section4[ds] = dict(single_outcome=True)
                continue
            print(f'\n{ds}: {len(ok)} ok x {len(fail)} fail in test')
            print(f'{"thresh":>7} {"ep pairs":>9} {"V acc":>7} {"ctl acc":>8} '
                  f'{"V ep-pair acc":>14} {"V CI95":>16} {"ctl ep-pair acc":>16}')
            by_threshold = {}
            for thresh in THRESHOLDS:
                frame_ok, frame_tot = 0, 0
                ctl_ok, ctl_tot = 0, 0
                groups = []          # one entry per episode pair
                ctl_groups = []
                for a in ok:
                    ha, aa = state[(ds, a['episode_index'])]
                    aa = (aa - mean) / std
                    sa = a['offset']
                    for b in fail:
                        hb, ab = state[(ds, b['episode_index'])]
                        ab = (ab - mean) / std
                        sb = b['offset']
                        ia = np.arange(0, max(a['frames'] - TAIL, 0), STRIDE)
                        ib = np.arange(0, max(b['frames'] - TAIL, 0), STRIDE)
                        if len(ia) == 0 or len(ib) == 0:
                            continue
                        d = np.sqrt(np.maximum(
                            (aa[ia] ** 2).sum(1)[:, None] + (ab[ib] ** 2).sum(1)[None, :]
                            - 2.0 * aa[ia] @ ab[ib].T, 0.0))
                        phase = (ha[ia][:, None] == hb[ib][None, :])
                        mask = (d <= thresh) & phase
                        if int(mask.sum()) < MIN_MATCHES:
                            continue
                        fia, fib = np.nonzero(mask)
                        ga, gb = ia[fia] + sa, ib[fib] + sb   # global frame indices
                        va, vb = prediction[ga], prediction[gb]
                        ca, cb = control[ga], control[gb]
                        frame_ok += int((va > vb).sum()) + 0.5 * int((va == vb).sum())
                        frame_tot += len(va)
                        ctl_ok += int((ca > cb).sum()) + 0.5 * int((ca == cb).sum())
                        ctl_tot += len(ca)
                        groups.append(float((va - vb).mean()))
                        ctl_groups.append(float((ca - cb).mean()))
                if not groups:
                    by_threshold[str(thresh)] = None
                    print(f'{thresh:7.2f} {0:9d} {"-":>7} {"-":>8} {"-":>14} {"-":>16} {"-":>16}')
                    continue
                g = np.asarray(groups)
                acc = float((g > 0).mean() + 0.5 * (g == 0).mean())
                cg = np.asarray(ctl_groups)
                cacc = float((cg > 0).mean() + 0.5 * (cg == 0).mean())
                rng = np.random.default_rng(SEED)
                draws = rng.integers(0, len(g), (BOOTSTRAP, len(g)))
                boot = np.asarray([float((g[d] > 0).mean() + 0.5 * (g[d] == 0).mean()) for d in draws])
                lo, hi = np.quantile(boot, [0.025, 0.975])
                by_threshold[str(thresh)] = dict(
                    episode_pairs=len(groups),
                    frame_level_accuracy=frame_ok / frame_tot,
                    control_frame_level_accuracy=ctl_ok / ctl_tot,
                    episode_pair_accuracy=acc, ci95=[float(lo), float(hi)],
                    control_episode_pair_accuracy=cacc,
                    frames_per_point=frame_tot / len(groups))
                print(f'{thresh:7.2f} {len(groups):9d} {frame_ok/frame_tot:7.3f} {ctl_ok/ctl_tot:8.3f} '
                      f'{acc:14.3f} [{lo:.2f},{hi:.2f}]{"":6} {cacc:16.3f}')
            section4[ds] = dict(ok=len(ok), fail=len(fail), thresholds=by_threshold)
        print('\nReading guide: "V ep-pair acc" is the honest number (independent unit is the')
        print('episode pair). "ctl acc" is the batch+frame-index control; if the pairing removed')
        print('the time/batch confound it must collapse towards 0.5. Frame-level accuracy is')
        print('reported only for reference - its frames are not independent.')

    # ---- section 5: verdict ----------------------------------------------
    print('\n=== SECTION 5: verdict ===')
    usable = {ds: v for ds, v in sweep.items() if not v.get('single_outcome')}
    print(f'episode pairs available before matching : {total_pairs}')
    print(f'sessions able to form any pair          : {len(usable)} of {len(sessions)}')
    for ds, v in usable.items():
        if v['thresholds'].get('1.0'):
            print(f'   {ds:16} at distance<=1.0: {v["thresholds"]["1.0"]["episode_pairs"]} episode pairs')
    print('Independent unit for the audit is the episode pair, not the frame,')
    print('so these counts - not the frame counts - set the confidence interval width.')

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(inventory=section1, pairing_sweep=sweep,
                                   gripper_events=section3, accuracy=section4,
                                   thresholds=THRESHOLDS, min_matches=MIN_MATCHES,
                                   stride=STRIDE, tail_frames=TAIL,
                                   bootstrap=BOOTSTRAP, seed=SEED),
                              indent=2) + '\n')
    print(f'\nWrote {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()