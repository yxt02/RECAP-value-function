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
    args = parser.parse_args(argv)
    if any(h <= 0 for h in args.horizon) or (args.max_episodes is not None and args.max_episodes < 1):
        parser.error('horizons and max-episodes must be positive')
    if not __import__('math').isfinite(args.threshold):
        parser.error('threshold must be finite')

    import logging
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Subset
    from omegaconf import OmegaConf
    from submodules.advantage import compute_advantage, label_scores, intervention_windows, export_advantage_table
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
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),status='running',horizons=horizons,
                    threshold=args.threshold,threshold_rule='strict >; equality is disadvantage',
                    values_sha256=sha256(output/'values.parquet'))
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
            score=result['advantage_continuous'];labels=label_scores(score,args.threshold)
            part=frame[key+['timestamp','split','success','intervention','value']].copy()
            for name,array in result.items():part[name]=array
            part['horizon']=horizon;part['threshold']=args.threshold;part['label']=labels
            part['is_advantage']=score>args.threshold;score_frames.append(part)
            axes[1].plot(t,score,label=f'{horizon} 帧优势')
            if len(horizons)==1:
                axes[1].fill_between(t,args.threshold,score,where=score>args.threshold,color='green',alpha=.2)
                axes[1].fill_between(t,args.threshold,score,where=score<=args.threshold,color='red',alpha=.15)
            grid,windows=intervention_windows(t,score,flags)
            events[horizon].extend(windows.tolist())
            summaries.append(dict(dataset_id=dataset,episode_index=int(ep),horizon=horizon,frames=n,
                success=bool(frame.success.iloc[0]),split=frame.split.iloc[0],mean_value=float(frame.value.mean()),
                mean_advantage=float(score.mean()),advantage_fraction=float(np.mean(score>args.threshold)),
                intervention_available=flags is not None,intervention_starts=len(windows),
                intervention_mean_advantage=float(score[flags==1].mean()) if flags is not None and np.any(flags==1) else None,
                non_intervention_mean_advantage=float(score[flags==0].mean()) if flags is not None and np.any(flags==0) else None))
        axes[1].axhline(args.threshold,color='black',linestyle='--',label='标签阈值');axes[1].set_ylabel('多步优势')
        if flags is not None: axes[2].step(t,flags,where='post',label='人工接管（1）')
        else:axes[2].text(.1,.5,'未提供接管标记',transform=axes[2].transAxes)
        axes[2].set_ylabel('接管');axes[2].set_xlabel('时间 / 秒')
        for ax in axes[:2]:ax.legend()
        fig.suptitle(f'{dataset} / {ep} · {"成功" if frame.success.iloc[0] else "失败"} · {frame.split.iloc[0]}')
        fig.tight_layout();name=f'episode-{len(links):04d}.png';fig.savefig(output/'plots'/name,dpi=120);plt.close(fig)
        links.append((f'{dataset}:{ep}',name))
    scores=pd.concat(score_frames,ignore_index=True);scores.to_parquet(output/'scores.parquet',index=False)
    advantages=export_advantage_table(scores);advantages.to_parquet(output/'advantages.parquet',index=False)
    dump(output/'episodes.json',summaries)
    event_report={};fig,ax=plt.subplots(figsize=(10,4))
    for horizon,rows in events.items():
        a=np.asarray(rows);event_report[str(horizon)]=dict(events=len(rows),relative_seconds=grid.tolist(),mean=a.mean(0).tolist() if len(rows) else None)
        if len(rows):ax.plot(grid,a.mean(0),label=f'{horizon} 帧，{len(rows)} 个接管事件')
    ax.axvline(0,color='black',linestyle='--');ax.axhline(args.threshold,color='grey',linestyle=':')
    ax.set(xlabel='相对接管开始时间 / 秒',ylabel='平均优势',title='接管前后：描述性对比，非因果证据；优势包含未来奖励')
    if any(events.values()):ax.legend()
    else:ax.text(.1,.5,'无完整 ±1 秒接管起始窗口',transform=ax.transAxes)
    fig.tight_layout();fig.savefig(output/'interventions.png',dpi=120);plt.close(fig)
    dump(output/'interventions.json',event_report)
    table = pd.DataFrame(summaries)[['dataset_id','episode_index','split','success','horizon','frames','mean_value','mean_advantage','advantage_fraction','intervention_mean_advantage','non_intervention_mean_advantage']].to_html(index=False, float_format=lambda x: f'{x:.5f}', na_rep='缺失', escape=True)
    items=''.join(f'<details><summary>{html.escape(title)}</summary><img src="plots/{name}"></details>' for title,name in links)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>优势与接管对比</title><style>body{max-width:1100px;margin:30px auto;font-family:sans-serif}img{width:100%}summary{padding:12px;cursor:pointer}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}.table{overflow:auto}</style><h1>模型优势与接管对比</h1><p>绿色为高于阈值，红色为未高于阈值。接管不是错误真值，失败不必然是负优势。默认 test；全部数据包含训练内评分。每个分数使用未来 N 帧，不是实时告警。</p><img src="interventions.png"><h2>轨迹评分比较</h2><p>mean_value：平均价值；mean_advantage：平均优势；advantage_fraction：优势帧比例；最后两列：接管/非接管帧平均优势。</p><div class="table">'+table+'</div><h2>逐轨迹曲线</h2>'+items,encoding='utf8')
    if not np.isfinite(scores.advantage_continuous).all():
        raise ValueError('Non-finite exported scores')
    dump(output/'validation.json',dict(status='passed',frames=len(values),episodes=len(links),score_rows=len(scores),
        advantage_table_rows=len(advantages),
        unique_frame_keys=True,finite_scores=bool(np.isfinite(scores.advantage_continuous).all()),model_quality_validated=False))
    manifest['status']='complete';manifest['files_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    dump(output/'manifest.json',manifest)
    print(f'Completed: {len(links)} episodes, {len(values)} frames. Report: {output / "index.html"}')


if __name__ == '__main__':
    main()
