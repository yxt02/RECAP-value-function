#!/usr/bin/env python3
"""Compute real N-step value advantages and descriptive intervention comparisons.

All relative paths are relative to the repository. Does not train or modify inputs.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submodules  # Set runtime threading before importing numeric libraries.


def label_columns(horizons):
    """Output column contract: continuous advantage value + binarized label."""
    multiple = len(horizons) > 1
    return {h: (f'advantage_{h}' if multiple else 'advantage',
                f'advantage_positive_{h}' if multiple else 'advantage_positive')
            for h in horizons}


def write_labeled_frames(destination, scores, frame_files, horizons, write_back=False):
    """Append the advantage columns to every scored frame file.

    Each scored frame parquet gains two columns beside `intervention`:
    `advantage` (float32, continuous N-step advantage) and `advantage_positive`
    (bool, the binarized RECAP label), so a reader needs no join against a
    separate table. Multiple horizons append `_<horizon>` to both names.
    Copies mirror the source layout (data + meta, no videos) under
    `destination`; `write_back` overwrites the source after a one-time `.bak`.
    Columns are appended with pyarrow so untouched columns keep their schema.
    Each dataset's `meta/info.json` features are extended so LeRobot readers
    pick up the new columns.
    """
    import shutil
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    names = label_columns(horizons)
    written = []
    roots = {}
    for (dataset, episode), part in scores.groupby(['dataset_id', 'episode_index'], sort=False):
        entry = frame_files.get(f'{dataset}:{int(episode)}')
        if entry is None:
            raise KeyError(f'No source frame file recorded for {dataset}:{int(episode)}')
        roots[dataset] = Path(entry['root'])
        source = Path(entry['file'])
        table = pq.read_table(source)
        frames = table.num_rows
        if not np.array_equal(table.column('frame_index').to_numpy(), np.arange(frames)):
            raise ValueError(f'Non-contiguous frame indices in source: {source}')
        for horizon in horizons:
            block = part[part.horizon == horizon].sort_values('frame_index')
            if not np.array_equal(block.frame_index.to_numpy(), np.arange(frames)):
                raise ValueError(f'Score/frame misalignment: {dataset}:{int(episode)} horizon {horizon}')
            value_name, label_name = names[horizon]
            columns = {value_name: block.advantage_continuous.to_numpy(np.float32),
                       label_name: block.is_advantage.to_numpy(bool)}
            for name, array in columns.items():
                if name in table.column_names:
                    table = table.set_column(table.column_names.index(name), name, pa.array(array))
                else:
                    table = table.append_column(name, pa.array(array))
        if write_back:
            backup = source.with_name(source.name + '.bak')
            if not backup.exists():
                shutil.copy2(source, backup)
            target = source
        else:
            target = destination / dataset / source.relative_to(entry['root'])
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        written.append(dict(dataset_id=dataset, episode_index=int(episode), frames=frames, path=str(target),
                            backup=str(source.with_name(source.name + '.bak')) if write_back else None))
    for dataset, root in roots.items():
        update_info_features(destination, dataset, root, names, write_back)
    return written


def update_info_features(destination, dataset, root, names, write_back):
    """Mirror `meta/` (no videos) and register the advantage columns in features."""
    import shutil
    if write_back:
        meta_target = root / 'meta'
        info_backup = meta_target / 'info.json.bak'
        if not info_backup.exists():
            shutil.copy2(meta_target / 'info.json', info_backup)
    else:
        meta_target = destination / dataset / 'meta'
        shutil.copytree(root / 'meta', meta_target, dirs_exist_ok=True)
    info_path = meta_target / 'info.json'
    info = json.loads(info_path.read_text(encoding='utf8'))
    features = info.setdefault('features', {})
    for key in [k for k in features if k.split('_')[0] == 'advantage']:
        del features[key]  # Drop stale entries from earlier label contracts.
    for value_name, label_name in names.values():
        features[value_name] = {'dtype': 'float32', 'shape': [1], 'names': None}
        features[label_name] = {'dtype': 'bool', 'shape': [1], 'names': None}
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/optimized/best_model.pt')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--horizon', nargs='+', type=int, default=[50])
    parser.add_argument('--threshold', type=float, default=0., help='Fixed threshold; strict > is advantage')
    parser.add_argument('--episode', action='append', default=[], help='Repeat DATASET:EPISODE; must belong to selected split')
    parser.add_argument('--max-episodes', type=int, help='First N complete selected episodes; never truncates frames')
    parser.add_argument('--output', help='New result directory; existing nonempty directory is rejected')
    parser.add_argument('--reuse-values', help='Prior COMPLETE result directory; recalculate without model inference')
    parser.add_argument('--label-rule', choices=['fixed', 'percentile'], default='fixed',
                        help='fixed: A > --threshold; percentile: A > epsilon, the value percentile below')
    parser.add_argument('--percentile', type=float, default=30.,
                        help='Value percentile used as epsilon under --label-rule percentile')
    parser.add_argument('--reference-split', choices=['train', 'val', 'test'], default='train',
                        help='Split whose predicted values define epsilon; must be scored in this run')
    parser.add_argument('--force-intervention-positive', action=argparse.BooleanOptionalAction, default=True,
                        help='Force I=positive on human-intervention frames (RECAP correction rule)')
    parser.add_argument('--write-labeled-frames', action='store_true',
                        help='Write every scored frame file with the advantage columns appended')
    parser.add_argument('--labeled-dir', help='Destination for labeled frames; defaults to <output>/labeled_frames')
    parser.add_argument('--write-back', action='store_true',
                        help='Overwrite the original dataset frame files instead of writing copies')
    args = parser.parse_args(argv)
    if any(h <= 0 for h in args.horizon) or (args.max_episodes is not None and args.max_episodes < 1):
        parser.error('horizons and max-episodes must be positive')
    if not __import__('math').isfinite(args.threshold):
        parser.error('threshold must be finite')
    if args.label_rule=='percentile' and not 0. <= args.percentile <= 100.:
        parser.error('percentile must lie in [0, 100]')
    if args.write_back:
        if args.labeled_dir:
            parser.error('--write-back targets the original files; --labeled-dir is meaningless')
        args.write_labeled_frames = True

    import logging
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Subset
    from omegaconf import OmegaConf
    from submodules.advantage import (compute_advantage, intervention_windows, export_advantage_table,
                                      value_percentile_threshold, label_improvement)
    from submodules.cache import prepare_features, cache_location, cache_identity, sha256
    from submodules.checkpoint import distribution_bins
    from submodules.contracts import resolve_path, episode_returns
    from submodules.runtime import raw_dataset, configure_runtime, autocast
    from submodules.model import ValueModel
    from submodules.evaluation_workflow import configure_plots, dump

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    horizons = sorted(set(args.horizon))
    output = resolve_path(args.output or 'artifacts/advantage/' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    frame_files=dict()
    if args.reuse_values:
        if args.episode or args.max_episodes or args.split != 'test':
            parser.error('--reuse-values keeps the original data selection; omit selection options')
        source = resolve_path(args.reuse_values)
        previous = json.loads((source/'manifest.json').read_text())
        if previous.get('status') != 'complete' or sha256(source/'values.parquet') != previous['values_sha256']:
            raise ValueError('Incomplete or altered source values')
        values = pd.read_parquet(source/'values.parquet')
        manifest = {k:v for k,v in previous.items() if k not in ('status','files_sha256')}
        manifest['reused_from'] = str(source)
        sidecar = source/'frame_files.json'
        if sidecar.exists():
            frame_files = json.loads(sidecar.read_text())
        elif args.write_labeled_frames:
            raise ValueError(f'Reused run has no frame_files.json; re-score without --reuse-values '
                             f'to write labeled frames: {sidecar}')
    else:
        checkpoint = resolve_path(args.checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        bins = distribution_bins(payload)
        cfg = dict(payload['config'])
        if not cfg['freeze_vlm']:
            raise ValueError('This workflow requires the frozen encoder checkpoint')
        cfg.update(max_samples=None, max_steps=None, val_steps=None, cache_num_workers=0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            cfg['precision'] = 'fp32'
        configure_runtime(cfg)
        adaptation = OmegaConf.to_container(OmegaConf.load(resolve_path(cfg['adaptation_config'])), resolve=True)
        split_root = resolve_path(adaptation['output_dir'])
        splits = {s:json.loads((split_root/f'{s}.json').read_text()) for s in ['train','val','test']}
        keys = {s:{(e['dataset'],e['episode_index']) for e in v['episodes']} for s,v in splits.items()}
        for a,b in [('train','val'),('train','test'),('val','test')]:
            if keys[a] & keys[b]: raise ValueError('Overlapping split episodes')
        # Check current train membership against the membership stored in the checkpoint cache.
        training_identity = (payload.get('feature_cache_manifest') or {}).get('identity', {})
        if not training_identity or training_identity.get('subset_indices') is not None:
            raise ValueError('Checkpoint lacks complete training membership provenance')
        train_data = raw_dataset(cfg, 'train')
        _, current_identity = cache_identity(train_data, cfg, 'train')
        for field in ('weights', 'processor', 'model_config', 'cameras'):
            if current_identity[field] != training_identity[field]:
                raise ValueError(f'Encoder provenance changed: {field}')
        recorded = {s['dataset']:s for s in training_identity['sources']}
        if set(recorded) != {d.dataset_path.name for d in train_data.datasets}:
            raise ValueError('Training dataset membership changed')
        for ds in train_data.datasets:
            saved = recorded[ds.dataset_path.name]
            current = next(source for source in current_identity['sources'] if source['dataset'] == ds.dataset_path.name)
            if json.dumps(current, sort_keys=True) != json.dumps(saved, sort_keys=True):
                raise ValueError('Training data, membership, preprocessing or return contract changed since checkpoint')
            ds.close()
        selected_splits = ['train','val','test'] if args.split == 'all' else [args.split]
        wanted = set()
        for entry in args.episode:
            name, ep = entry.rsplit(':', 1)
            wanted.add((name, int(ep)))
        candidates = set().union(*(keys[s] for s in selected_splits))
        if wanted - candidates: raise ValueError(f'Episodes outside selection: {wanted-candidates}')
        manifest = dict(schema_version=1, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        config=cfg, return_scale=adaptation['return_scale'], gamma=adaptation['gamma'],
                        split=args.split, training_membership_checked=True,
                        interpretation='Model scores, not action ground truth; validation may have selected checkpoint.',
                        splits_sha256={s:sha256(split_root/f'{s}.json') for s in splits}, caches=[])
        rows=[]; count=0; encoder=None
        for split in selected_splits:
            data = raw_dataset(cfg, split)
            metadata = {(e['dataset'],e['episode_index']):e for e in splits[split]['episodes']}
            records=[]; indices=[]; offset=0
            for ds in data.datasets:
                for ep in ds.episode_ids:
                    n=len(ds.returns_data[ep]['reward']); key=(ds.dataset_path.name,ep)
                    take=(not wanted or key in wanted) and (args.max_episodes is None or count < args.max_episodes)
                    if take:
                        entry=metadata[key]
                        if entry['total_frames'] != n: raise ValueError('Manifest frame count mismatch')
                        frame=pd.read_parquet(ds._paths_by_episode[ep])
                        frame_files[f'{key[0]}:{key[1]}']=dict(root=str(ds.dataset_path),
                                                               file=str(ds._paths_by_episode[ep]))
                        times=frame.timestamp.to_numpy(float)
                        if not np.isfinite(times).all() or not np.allclose(times,np.arange(n)/ds.info['fps'],atol=1e-4):
                            raise ValueError(f'Invalid timestamps: {key}')
                        rewards=ds.returns_data[ep]['reward'].astype(float)
                        expected,expected_rewards=episode_returns(n,entry['is_success'],adaptation['gamma'],adaptation['failure_reward'])
                        np.testing.assert_allclose(rewards,expected_rewards,rtol=0,atol=0)
                        np.testing.assert_allclose(ds.returns_data[ep]['return'],expected,rtol=0,atol=1e-5)
                        intervention=frame.intervention.to_numpy() if 'intervention' in frame else np.full(n,np.nan)
                        if 'intervention' in frame and not np.isin(intervention,[0,1]).all():
                            raise ValueError(f'Invalid intervention flags: {key}')
                        records.append(pd.DataFrame(dict(dataset_id=key[0],episode_index=ep,
                            frame_index=np.arange(n),timestamp=times,fps=float(ds.info['fps']),split=split,
                            success=entry['is_success'],reward_raw=rewards,intervention=intervention)))
                        indices.extend(range(offset,offset+n)); count+=1
                    offset+=n
            if not records:
                for ds in data.datasets: ds.close()
                continue
            selected = data if len(indices)==len(data) else Subset(data,indices)
            path,digest,_=cache_location(selected,cfg,split)
            if encoder is None:
                encoder=ValueModel(str(resolve_path(cfg['siglip_path'])),cfg['cameras'],True,
                    cfg['precision'],cfg['projection_dim'],num_bins=bins,
                    load_encoder=not (path/'manifest.json').exists()).to(device).eval()
            if not (path/'manifest.json').exists() and encoder.siglip is None:
                encoder=ValueModel(str(resolve_path(cfg['siglip_path'])),cfg['cameras'],True,
                    cfg['precision'],cfg['projection_dim'],num_bins=bins).to(device).eval()
            cached=prepare_features(selected,encoder,cfg,split,device)
            head=ValueModel(str(resolve_path(cfg['siglip_path'])),cfg['cameras'],True,
                cfg['precision'],cfg['projection_dim'],load_encoder=False,num_bins=bins).to(device).eval()
            head.load_state_dict(payload['model_state_dict'],strict=True)
            predictions=[]
            with torch.inference_mode(),autocast(cfg,device):
                for start in range(0,len(cached),cfg['cached_batch_size']):
                    predictions.append(head(features=cached.features[start:start+cfg['cached_batch_size']].to(device)).float().cpu().numpy().ravel())
            part=pd.concat(records,ignore_index=True); part['value']=np.concatenate(predictions)
            if not np.isfinite(part.value).all() or not part.value.between(-1,0).all(): raise ValueError('Invalid value predictions')
            rows.append(part);manifest['caches'].append(dict(split=split,path=str(path),digest=digest))
            for ds in data.datasets: ds.close()
            del head,cached
        if not rows: raise ValueError('Empty selection')
        values=pd.concat(rows,ignore_index=True)
        if sha256(checkpoint)!=manifest['checkpoint_sha256']: raise ValueError('Checkpoint changed during inference')
        del encoder
    key=['dataset_id','episode_index','frame_index']
    if values.duplicated(key).any(): raise ValueError('Duplicate frame keys')
    values.to_parquet(output/'values.parquet',index=False)
    if args.label_rule=='percentile':
        reference=values[values.split==args.reference_split]
        if reference.empty:
            raise ValueError(f'--reference-split {args.reference_split} is not scored in this run; '
                             f'include it (for example --split all) or reuse a result that does')
        threshold=value_percentile_threshold(reference.value.to_numpy(float),args.percentile)
        threshold_rule=f'epsilon = {args.percentile:g}th percentile of predicted values on {args.reference_split}'
    else:
        threshold=float(args.threshold)
        threshold_rule='fixed strict >; equality is disadvantage'
    threshold_rule+=('; intervention frames forced positive' if args.force_intervention_positive
                     else '; intervention frames labeled by score')
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),status='running',horizons=horizons,
                    threshold=threshold,threshold_rule=threshold_rule,label_rule=args.label_rule,
                    percentile=args.percentile if args.label_rule=='percentile' else None,
                    reference_split=args.reference_split if args.label_rule=='percentile' else None,
                    force_intervention_positive=bool(args.force_intervention_positive),
                    values_sha256=sha256(output/'values.parquet'))
    if frame_files: dump(output/'frame_files.json',frame_files)
    dump(output/'manifest.json',manifest)
    plt=configure_plots();(output/'plots').mkdir();score_frames=[];summaries=[];events={h:[] for h in horizons}
    links=[]
    for (dataset,ep),frame in values.groupby(['dataset_id','episode_index'],sort=False):
        frame=frame.sort_values('frame_index');n=len(frame)
        if not np.array_equal(frame.frame_index,np.arange(n)): raise ValueError('Non-contiguous episode frames')
        flags=None if frame.intervention.isna().all() else frame.intervention.to_numpy()
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
        t=frame.timestamp.to_numpy();axes[0].plot(t,frame.value,label='模型价值 V(t)');axes[0].set_ylabel('价值')
        for horizon in horizons:
            result=compute_advantage(frame.value,frame.reward_raw,horizon,manifest['return_scale'],manifest['gamma'])
            score=result['advantage_continuous']
            positive,forced=label_improvement(score,threshold,flags if args.force_intervention_positive else None)
            part=frame[key+['timestamp','split','success','intervention','value']].copy()
            for name,array in result.items():part[name]=array
            part['horizon']=horizon;part['threshold']=threshold
            part['label']=np.where(positive,'advantage','disadvantage')
            part['is_advantage']=positive;part['advantage_forced']=forced;score_frames.append(part)
            axes[1].plot(t,score,label=f'{horizon} 帧优势')
            if len(horizons)==1:
                axes[1].fill_between(t,threshold,score,where=positive,color='green',alpha=.2)
                axes[1].fill_between(t,threshold,score,where=~positive,color='red',alpha=.15)
            grid,windows=intervention_windows(t,score,flags)
            events[horizon].extend(windows.tolist())
            summaries.append(dict(dataset_id=dataset,episode_index=int(ep),horizon=horizon,frames=n,
                success=bool(frame.success.iloc[0]),split=frame.split.iloc[0],mean_value=float(frame.value.mean()),
                mean_advantage=float(score.mean()),advantage_fraction=float(np.mean(positive)),
                forced_positive_frames=int(forced.sum()),
                intervention_available=flags is not None,intervention_starts=len(windows),
                intervention_mean_advantage=float(score[flags==1].mean()) if flags is not None and np.any(flags==1) else None,
                non_intervention_mean_advantage=float(score[flags==0].mean()) if flags is not None and np.any(flags==0) else None))
        axes[1].axhline(threshold,color='black',linestyle='--',label='标签阈值');axes[1].set_ylabel('多步优势')
        if flags is not None: axes[2].step(t,flags,where='post',label='人工接管（1）')
        else:axes[2].text(.1,.5,'未提供接管标记',transform=axes[2].transAxes)
        axes[2].set_ylabel('接管');axes[2].set_xlabel('时间 / 秒')
        for ax in axes[:2]:ax.legend()
        fig.suptitle(f'{dataset} / {ep} · {"成功" if frame.success.iloc[0] else "失败"} · {frame.split.iloc[0]}')
        fig.tight_layout();name=f'episode-{len(links):04d}.png';fig.savefig(output/'plots'/name,dpi=120);plt.close(fig)
        links.append((f'{dataset}:{ep}',name))
    scores=pd.concat(score_frames,ignore_index=True);scores.to_parquet(output/'scores.parquet',index=False)
    advantages=export_advantage_table(scores);advantages.to_parquet(output/'advantages.parquet',index=False)
    labeled=[]
    if args.write_labeled_frames:
        destination=resolve_path(args.labeled_dir) if args.labeled_dir else output/'labeled_frames'
        if not args.write_back: destination.mkdir(parents=True,exist_ok=True)
        labeled=write_labeled_frames(destination,scores,frame_files,horizons,args.write_back)
        columns=[name for pair in label_columns(horizons).values() for name in pair]
        manifest['labeled_frames']=dict(root=None if args.write_back else str(destination),
                                        write_back=bool(args.write_back),files=len(labeled),
                                        columns=columns,videos_mirrored=False,
                                        note='meta/episodes_stats.jsonl predates the label columns')
        dump(output/'labeled_frames.json',labeled)
    dump(output/'episodes.json',summaries)
    event_report={};fig,ax=plt.subplots(figsize=(10,4))
    for horizon,rows in events.items():
        a=np.asarray(rows);event_report[str(horizon)]=dict(events=len(rows),relative_seconds=grid.tolist(),mean=a.mean(0).tolist() if len(rows) else None)
        if len(rows):ax.plot(grid,a.mean(0),label=f'{horizon} 帧，{len(rows)} 个接管事件')
    ax.axvline(0,color='black',linestyle='--');ax.axhline(threshold,color='grey',linestyle=':')
    ax.set(xlabel='相对接管开始时间 / 秒',ylabel='平均优势',title='接管前后：描述性对比，非因果证据；优势包含未来奖励')
    if any(events.values()):ax.legend()
    else:ax.text(.1,.5,'无完整 ±1 秒接管起始窗口',transform=ax.transAxes)
    fig.tight_layout();fig.savefig(output/'interventions.png',dpi=120);plt.close(fig)
    dump(output/'interventions.json',event_report)
    table = pd.DataFrame(summaries)[['dataset_id','episode_index','split','success','horizon','frames','mean_value','mean_advantage','advantage_fraction','forced_positive_frames','intervention_mean_advantage','non_intervention_mean_advantage']].to_html(index=False, float_format=lambda x: f'{x:.5f}', na_rep='缺失', escape=True)
    items=''.join(f'<details><summary>{html.escape(title)}</summary><img src="plots/{name}"></details>' for title,name in links)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>优势与接管对比</title><style>body{max-width:1100px;margin:30px auto;font-family:sans-serif}img{width:100%}summary{padding:12px;cursor:pointer}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}.table{overflow:auto}</style><h1>模型优势与接管对比</h1>'+f'<p>标签规则：{html.escape(str(manifest.get("threshold_rule")))}，阈值 {manifest.get("threshold"):.6f}。绿色为 advantage，红色为 disadvantage；接管段被强制标为 advantage。接管不是错误真值，失败不必然是负优势。每个分数使用未来 N 帧，不是实时告警。</p>'+'<img src="interventions.png"><h2>轨迹评分比较</h2><p>mean_value：平均价值；mean_advantage：平均优势；advantage_fraction：优势帧比例；forced_positive_frames：被接管强制置正的帧数；最后两列：接管/非接管帧平均优势。</p><div class="table">'+table+'</div><h2>逐轨迹曲线</h2>'+items,encoding='utf8')
    if not np.isfinite(scores.advantage_continuous).all():
        raise ValueError('Non-finite exported scores')
    dump(output/'validation.json',dict(status='passed',frames=len(values),episodes=len(links),score_rows=len(scores),
        advantage_table_rows=len(advantages),labeled_frame_files=len(labeled),
        unique_frame_keys=True,finite_scores=bool(np.isfinite(scores.advantage_continuous).all()),model_quality_validated=False))
    manifest['status']='complete';manifest['files_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    dump(output/'manifest.json',manifest)
    print(f'Completed: {len(links)} episodes, {len(values)} frames. Report: {output / "index.html"}')


if __name__ == '__main__':
    main()
