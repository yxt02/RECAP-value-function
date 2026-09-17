#!/usr/bin/env python3
"""Audit and adapt z02 datasets. Raw parquet/video files are never rewritten."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import zipfile
import stat

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from recap_datasets.recap.contracts import episode_returns, resolve_outcome, resolve_path

compute_returns_for_episode = episode_returns
JOINT_NAMES = ([f'left_arm_{i}' for i in range(7)] + ['left_hand_binary'] +
               [f'right_arm_{i}' for i in range(7)] + ['right_hand_binary'] +
               [f'body_source_{i}' for i in range(4)] + ['head_source_18', 'head_source_19'])


def save_json(path, value):
    """Back up differing metadata once; use atomic replacement."""
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        if path.read_text() == content:
            return
        backup = path.with_name(path.name + '.before_z02_v2')
        if not backup.exists():
            shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content)
    temp.replace(path)


def import_archive(archive, data_dir):
    archive = Path(archive).resolve()
    destination = resolve_path(data_dir).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            p = (destination / item.filename).resolve()
            if not p.is_relative_to(destination) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f'Unsafe archive member: {item.filename}')
            if p.exists() and not item.is_dir():
                raise FileExistsError(f'Refusing to overwrite: {p}')
        z.extractall(destination)
    print(f'Imported {archive} into {destination}')


def audit_dataset(cfg, spec, verify_videos=False):
    root = resolve_path(cfg['data_dir']) / spec['name']
    files = sorted((root / 'data').rglob('episode_*.parquet'))
    if not files:
        raise FileNotFoundError(root)
    info = json.loads((root / 'meta/info.json').read_text())
    tasks = {e['task_index']: e['task'].strip() for e in map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
    episodes, tables, videos, schema_names = [], [], [], set()
    for p in files:
        table = pq.read_table(p)
        schema_names.update(table.column_names)
        d = table.to_pandas()
        n = len(d)
        if not n or d.episode_index.nunique() != 1 or not np.array_equal(d.frame_index, np.arange(n)):
            raise ValueError(f'Invalid episode/frame indices: {p}')
        ep = int(d.episode_index.iloc[0])
        if n > cfg['max_episode_steps']:
            raise ValueError(f'{p}: {n} exceeds fixed task horizon; create a new return contract')
        for key in ['observation.joint_positions', 'action.joint_positions']:
            a = np.stack(d[key])
            if a.shape != (n, spec['expected_joint_dim']) or not np.isfinite(a).all():
                raise ValueError(f'{p}: invalid {key} shape/values: {a.shape}')
            if spec.get('expected_extra_tail') is not None and not np.allclose(a[:, 22:], spec['expected_extra_tail'], atol=1e-7):
                raise ValueError(f'{p}: unexpected extra tail; joint mapping needs review')
            if spec['joint_indices'] is not None and not np.isin(a[:, [7, 15]], [0., 1.]).all():
                raise ValueError(f'{p}: hand binary channels invalid')
        if not np.allclose(d.timestamp, np.arange(n) / info['fps'], atol=1e-4):
            raise ValueError(f'{p}: timestamp/frame alignment mismatch')
        if 'intervention' in d and not d.intervention.isin([0, 1]).all():
            raise ValueError(f'{p}: invalid intervention flags')
        success = resolve_outcome(d, override=spec.get('outcome_override'))
        raw_terminal = float(d.reward.iloc[-1]) if 'reward' in d else None
        record = dict(dataset=spec['name'], episode_index=ep, episode_file=p.name,
                      total_frames=n, is_success=success, raw_terminal_reward=raw_terminal,
                      intervention_ratio=float(d.intervention.mean()) if 'intervention' in d else 1.0,
                      joint_dim=spec['expected_joint_dim'], excluded=n < cfg['min_episode_steps'])
        episodes.append(record)
        ret, rew = episode_returns(n, success, cfg['gamma'], cfg['failure_reward'])
        if np.min(ret) < -cfg['return_scale']:
            raise ValueError('Return outside shared support')
        prompts = [tasks[int(t)] for t in d.task_index]
        tables.append(pa.table({'episode_index': d.episode_index.to_numpy(),
                                'frame_index': d.frame_index.to_numpy(), 'return': ret,
                                'reward': rew, 'prompt': prompts}))
        for cam in cfg['cameras']:
            vp = root / info['video_path'].format(episode_chunk=ep // info['chunks_size'],
                                                 episode_index=ep, video_key=f'observation.images.{cam}')
            if not vp.is_file():
                raise FileNotFoundError(vp)
            if verify_videos:
                cap = cv2.VideoCapture(str(vp))
                try:
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if count != n or abs(fps - info['fps']) > 0.05:
                        raise ValueError(f'{vp}: video frames/fps {count}/{fps} != {n}/{info["fps"]}')
                    for pos in sorted({0, n // 2, n - 1}):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                        ok, image = cap.read()
                        if not ok or image is None:
                            raise ValueError(f'{vp}: cannot decode frame {pos}')
                finally:
                    cap.release()
            videos.append(str(vp.relative_to(root)))
    if len({e['episode_index'] for e in episodes}) != len(episodes):
        raise ValueError(f'Duplicate episode ids in {root}')
    return root, info, tasks, episodes, pa.concat_tables(tables), videos, schema_names


def write_dataset(cfg, spec, audited):
    root, info, tasks, episodes, table, videos, schema = audited
    # Sidecars are versioned separately from the original rewards and z0 labels.
    out = root / 'meta' / f'returns_{cfg["tag"]}.parquet'
    temp = out.with_suffix('.parquet.tmp')
    pq.write_table(table, temp)
    temp.replace(out)
    contract = {k: cfg[k] for k in ('tag', 'gamma', 'failure_reward', 'return_scale', 'max_episode_steps')}
    contract.update(task=list(tasks.values()), normalization='raw_return / return_scale',
                    outcome_override=spec.get('outcome_override'), outcome_source=spec.get('outcome_source', 'raw terminal reward'),
                    sidecar_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    save_json(out.with_suffix('.json'), contract)
    save_json(root / 'meta/episode_outcomes.json', {str(e['episode_index']): {
        'is_success': e['is_success'], 'raw_terminal_reward': e['raw_terminal_reward'],
        'source': spec.get('outcome_source', 'raw terminal reward')} for e in episodes})
    save_json(root / 'meta/z02_adaptation.json', {
        'raw_joint_dim': spec['expected_joint_dim'], 'joint_indices': spec['joint_indices'],
        'target_joint_names': JOINT_NAMES, 'raw_arrays_modified': False,
        'extra_dimensions': 'Raw tail retained; explicit model selection only',
        'excluded_episodes': [e['episode_index'] for e in episodes if e['excluded']]})
    # Metadata describes raw storage, not the padded/model-selected array.
    features = {k: v for k, v in info['features'].items() if k in schema or v['dtype'] == 'video'}
    for key in ['observation.joint_positions', 'action.joint_positions']:
        dim = spec['expected_joint_dim']
        features[key]['shape'] = [dim]
        features[key]['names'] = (JOINT_NAMES + ['raw_extra_22', 'raw_extra_23'] if dim == 24 else
                                  JOINT_NAMES if dim == 22 else [f'legacy_joint_{i}' for i in range(dim)])
    info.update(robot_type='z02', features=features, total_episodes=len(episodes),
                total_frames=sum(e['total_frames'] for e in episodes), total_videos=len(videos))
    save_json(root / 'meta/info.json', info)


def split_episodes(cfg, episodes):
    output = resolve_path(cfg['output_dir'])
    eligible = {(e['dataset'], e['episode_index']): e for e in episodes if not e['excluded']}
    splits = {s: [] for s in ('train', 'val', 'test')}
    assigned = set()
    if cfg.get('preserve_existing_splits', True):
        for s in splits:
            p = output / f'{s}.json'
            if p.exists():
                for old in json.loads(p.read_text())['episodes']:
                    key = (old['dataset'], old['episode_index'])
                    if key in assigned:
                        raise ValueError(f'Split overlap: {key}')
                    if key in eligible:
                        splits[s].append(eligible[key]); assigned.add(key)
    rng = random.Random(cfg['random_seed'])
    groups = defaultdict(list)
    for key, e in eligible.items():
        if key not in assigned:
            groups[(e['dataset'], e['is_success'])].append(e)
    for key in sorted(groups):
        group = groups[key]; rng.shuffle(group)
        nt = int(len(group) * cfg['train_ratio']); nv = int(len(group) * cfg['val_ratio'])
        for s, part in zip(splits, (group[:nt], group[nt:nt+nv], group[nt+nv:])):
            splits[s].extend(part)
    summary = {}
    for s, entries in splits.items():
        entries.sort(key=lambda e: (e['dataset'], e['episode_index']))
        counts = dict(total_episodes=len(entries), total_frames=sum(e['total_frames'] for e in entries),
                      success_episodes=sum(e['is_success'] for e in entries), failure_episodes=sum(not e['is_success'] for e in entries))
        save_json(output / f'{s}.json', dict(split_name=s, **counts, episodes=entries))
        summary[s] = counts
        p = output / 'lerobot' / f'{s}_episodes.txt'; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(f'{e["dataset"]}/{e["episode_file"]}\n' for e in entries))
    save_json(output / 'summary.json', summary)
    save_json(output / 'excluded.json', {'episodes': [e for e in episodes if e['excluded']], 'reason': 'Fewer than min_episode_steps; retained in raw storage'})
    return summary


def run(config_path, write=False, split=False, verify_videos=False):
    cfg = OmegaConf.to_container(OmegaConf.load(resolve_path(config_path)), resolve=True)
    if abs(sum(cfg[k] for k in ('train_ratio', 'val_ratio', 'test_ratio')) - 1) > 1e-8:
        raise ValueError('Split ratios must sum to one')
    if cfg['failure_reward'] > -cfg['max_episode_steps'] or cfg['return_scale'] < abs(cfg['failure_reward']) + cfg['max_episode_steps'] - 1:
        raise ValueError('Task contract must separate failures and cover the full return range')
    task_texts = set()
    for spec in cfg['datasets']:
        task_file = resolve_path(cfg['data_dir']) / spec['name'] / 'meta/tasks.jsonl'
        task_texts.update(json.loads(line)['task'].strip() for line in task_file.read_text().splitlines())
    if len(task_texts) != 1:
        raise ValueError('This shared contract covers one task; use separate contracts for different tasks')
    episodes, reports = [], {}
    for spec in cfg['datasets']:
        audited = audit_dataset(cfg, spec, verify_videos)
        root, info, tasks, eps, table, videos, schema = audited
        episodes.extend(eps)
        if write:
            write_dataset(cfg, spec, audited)
        reports[spec['name']] = dict(episodes=len(eps), frames=sum(e['total_frames'] for e in eps),
            successes=sum(e['is_success'] for e in eps), failures=sum(not e['is_success'] for e in eps),
            raw_terminal_rewards=dict(Counter(str(e['raw_terminal_reward']) for e in eps)),
            raw_joint_dim=spec['expected_joint_dim'], videos=len(videos), video_check='count/fps + first/middle/last decode' if verify_videos else 'existence')
        print(spec['name'], reports[spec['name']], flush=True)
    summary = split_episodes(cfg, episodes) if split else None
    if write or split:
        save_json(resolve_path(cfg['output_dir']) / 'adaptation_report.json', dict(datasets=reports, splits=summary if summary is not None else (json.loads((resolve_path(cfg['output_dir']) / 'summary.json').read_text()) if (resolve_path(cfg['output_dir']) / 'summary.json').exists() else None), config=cfg))
    if summary:
        print(json.dumps(summary, indent=2))
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/z02_data.yaml')
    p.add_argument('--all', action='store_true')
    p.add_argument('--analyze', action='store_true')
    p.add_argument('--compute_returns', action='store_true')
    p.add_argument('--split_data', action='store_true')
    p.add_argument('--verify-videos', action='store_true')
    p.add_argument('--import-zip', type=Path)
    args = p.parse_args()
    if args.import_zip:
        cfg = OmegaConf.load(resolve_path(args.config)); import_archive(args.import_zip, cfg.data_dir)
    if any([args.all, args.analyze, args.compute_returns, args.split_data, args.verify_videos]):
        run(args.config, args.all or args.compute_returns, args.all or args.split_data, args.verify_videos)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
