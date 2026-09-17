#!/usr/bin/env python3
"""Fixed-checkpoint, full-test evaluation. No fitting of the model or test-selected thresholds."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recap_datasets.recap.contracts import resolve_path
from recap_value.cache import cache_location,prepare_features,FeatureDataset,sha256
from recap_value.model import ValueModel
from recap_value.runtime import configure_runtime,raw_dataset,make_loader,autocast
from recap_value.evaluation import regression,paired_episode_bootstrap,fit_baselines,apply_baselines,temporal_metrics

log=logging.getLogger(__name__)


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def split_metadata(data,split):
    manifest=json.loads((ROOT/f'data/splits/{split}.json').read_text())
    lookup={(e['dataset'],e['episode_index']):e for e in manifest['episodes']}
    episodes=[];targets=[];offset=0
    for ds in data.datasets:
        for ep in ds.episode_ids:
            n=len(ds.returns_data[ep]['return'])
            entry=lookup[(ds.dataset_path.name,ep)]
            if n!=entry['total_frames']:raise ValueError('Split frame count changed')
            episodes.append(dict(dataset=ds.dataset_path.name,episode_index=int(ep),frames=n,offset=offset,
                success=bool(entry['is_success']),fps=float(ds.info['fps']),
                slug=f'{ds.dataset_path.name}-ep{ep:06d}',raw_terminal_reward=entry['raw_terminal_reward']))
            targets.append(np.asarray(ds.returns_data[ep]['return'],dtype=np.float64)/ds.return_scale)
            offset+=n
    if offset!=len(data):raise ValueError('Metadata does not match full dataset')
    return episodes,np.concatenate(targets)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='checkpoints/optimized/best_model.pt')
    parser.add_argument('--output')
    args=parser.parse_args()
    checkpoint=resolve_path(args.checkpoint); checkpoint_hash=sha256(checkpoint)
    output=resolve_path(args.output or f'artifacts/evaluation/checkpoint-{checkpoint_hash[:12]}-test')
    output.mkdir(parents=True,exist_ok=True)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('format_version')!=2 or payload.get('architecture')!='siglip_mean_patch_scalar':
        raise ValueError('Unsupported checkpoint')
    config=dict(payload['config']); config.update(max_samples=None,max_steps=None,val_steps=None)
    if not torch.cuda.is_available():raise RuntimeError('Use the original CUDA/BF16 environment for this audit')
    configure_runtime(config)
    split_json={s:json.loads((ROOT/f'data/splits/{s}.json').read_text()) for s in ['train','val','test']}
    sets={s:{(e['dataset'],e['episode_index']) for e in m['episodes']} for s,m in split_json.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if sets[a]&sets[b]:raise ValueError('Overlapping split episodes')
    # Written before test predictions. These diagnostics are not used to select a model.
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,checkpoint_epoch=payload['epoch'],checkpoint_step=payload['global_step'],
        splits_sha256={s:sha256(ROOT/f'data/splits/{s}.json') for s in split_json},
        primary_metrics=['frame MSE/MAE/RMSE','equal-episode MSE','paired episode-bootstrap 95% intervals'],
        baselines=['train global mean','train per-dataset mean','train per-dataset linear elapsed-frame predictor'],
        diagnostics=['success/failure and dataset strata','fixed 10-bin prediction calibration',
                     'within-episode Spearman','1/10/30-frame TD residuals excluding terminal transitions',
                     'intervention flags and 1-second event windows; descriptive only'],
        bootstrap_repeats=4000,bootstrap_seed=20260917,lag_frames=[1,10,30],
        model_refit=False,test_threshold_selection=False,config=config)
    protocol_path=output/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if previous['checkpoint_sha256']!=checkpoint_hash or previous['splits_sha256']!=protocol['splits_sha256']:
            raise ValueError('Output folder belongs to a different audit')
    else:dump(protocol_path,protocol)
    train=raw_dataset(config,'train'); test=raw_dataset(config,'test')
    train_path,train_digest,train_identity=cache_location(train,config,'train')
    if train_digest!=payload['feature_cache_manifest']['digest']:
        raise ValueError('Training data/encoder identity differs from checkpoint')
    train_episodes,train_targets=split_metadata(train,'train')
    episodes,targets=split_metadata(test,'test')
    fit=fit_baselines(train_episodes,train_targets); baselines=apply_baselines(fit,episodes)
    dump(output/'baseline_fit_train_only.json',fit)
    path,digest,_=cache_location(test,config,'test')
    log.info('Full test: %d episodes, %d frames, checkpoint epoch %d',len(episodes),len(test),payload['epoch'])
    model=ValueModel(str(resolve_path(config['siglip_path'])),config['cameras'],True,config['precision'],
                     config['projection_dim'],load_encoder=not (path/'manifest.json').exists()).cuda().eval()
    cached=prepare_features(test,model,config,'test',torch.device('cuda'))
    model.siglip=None
    model.load_state_dict(payload['model_state_dict'],strict=True)
    model.eval(); torch.cuda.empty_cache()
    loader=make_loader(cached,config,'test',batch_size=config['cached_batch_size'],workers=0)
    predictions=[]
    with torch.inference_mode(),autocast(config,torch.device('cuda')):
        for batch in loader:predictions.append(model(features=batch['features'].cuda()).float().cpu().numpy().ravel())
    prediction=np.concatenate(predictions).astype(np.float64)
    np.testing.assert_allclose(targets,cached.targets.numpy(),atol=6e-8,rtol=0)
    if len(prediction)!=len(targets) or not np.isfinite(prediction).all():raise ValueError('Invalid predictions')
    timestamp=np.empty(len(test));intervention=np.zeros(len(test),dtype=np.int8)
    ds_lookup={d.dataset_path.name:d for d in test.datasets}; events=[]
    (output/'storyboards').mkdir(exist_ok=True);(output/'videos').mkdir(exist_ok=True)
    for entry in episodes:
        ds=ds_lookup[entry['dataset']];ep=entry['episode_index'];n=entry['frames'];start=entry['offset'];sl=slice(start,start+n)
        frame=pq.read_table(ds._paths_by_episode[ep],columns=['timestamp','intervention']).to_pandas()
        times=frame.timestamp.to_numpy(float); times-=times[0]
        if not np.all(np.diff(times)>0):raise ValueError('Nonmonotonic timestamps')
        flags=frame.intervention.to_numpy(dtype=np.int8)
        if not np.isin(flags,[0,1]).all():raise ValueError('Unknown intervention flags')
        timestamp[sl]=times;intervention[sl]=flags
        y=targets[sl];p=prediction[sl]
        entry.update(regression(y,p));entry['temporal']=temporal_metrics(y,p)
        entry['baselines']={key:regression(y,value[sl]) for key,value in baselines.items()}
        entry['intervention_frames']=int(flags.sum());entry['duration_seconds']=float(times[-1])
        entry['terminal_prediction']=float(p[-1]);entry['terminal_target']=float(y[-1])
        entry['max_error_frame']=int(np.abs(p-y).argmax())
        positions=sorted(set([0,n//4,n//2,3*n//4,n-1,entry['max_error_frame']]))
        entry['storyboard_frames']=positions
        sheet=Image.new('RGB',(320*len(positions),268),'#ffffff'); draw=ImageDraw.Draw(sheet)
        for i,fr in enumerate(positions):
            image=Image.fromarray(ds._read_video_frame(config['cameras'][0],ep,fr));image.thumbnail((320,230))
            sheet.paste(image,(i*320+(320-image.width)//2,0))
            draw.text((i*320+6,232),f'f={fr}  t={times[fr]:.2f}s',fill='#111111')
            draw.text((i*320+6,247),f'V={p[fr]:.3f}  R={y[fr]:.3f}',fill='#111111')
        sheet.save(output/'storyboards'/f"{entry['slug']}.jpg",quality=86)
        video=output/'videos'/f"{entry['slug']}.mp4"
        if not video.exists():video.symlink_to(ds.video_path(config['cameras'][0],ep))
        event_starts=np.flatnonzero((flags==1)&np.r_[True,flags[:-1]==0])
        for fr in event_starts:
            h=round(entry['fps'])
            if fr<h or fr+h>=n:continue
            events.append(dict(slug=entry['slug'],dataset=entry['dataset'],success=entry['success'],frame=int(fr),
                seconds=float(times[fr]),before_delta=float(p[fr]-p[fr-h]),after_delta=float(p[fr+h]-p[fr]),
                value_at_start=float(p[fr]),window_frames=h))
        log.info('Trajectory %s MSE %.5f, rho %s',entry['slug'],entry['mse'],entry['spearman'])
    for ds in test.datasets:ds.close()
    rows=[]
    grouping=[('all',episodes),('success',[e for e in episodes if e['success']]),('failure',[e for e in episodes if not e['success']])]
    grouping += [(name,[e for e in episodes if e['dataset']==name]) for name in sorted(ds_lookup)]
    grouping += [(f"{name}/{'success' if outcome else 'failure'}",[e for e in episodes if e['dataset']==name and e['success']==outcome])
                 for name in sorted(ds_lookup) for outcome in [True,False]]
    for name,eps in grouping:
        if not eps:continue
        indices=np.concatenate([np.arange(e['offset'],e['offset']+e['frames']) for e in eps])
        row=dict(group=name,episodes=len(eps),**regression(targets[indices],prediction[indices]),
                 episode_macro_mse=float(np.mean([e['mse'] for e in eps])),
                 episode_macro_mae=float(np.mean([e['mae'] for e in eps])),
                 baselines={k:regression(targets[indices],v[indices]) for k,v in baselines.items()})
        rows.append(row)
    bootstrap={k:paired_episode_bootstrap(targets,prediction,v,episodes) for k,v in baselines.items()}
    flag_groups={str(flag):regression(targets[intervention==flag],prediction[intervention==flag])
                 for flag in [0,1] if np.any(intervention==flag)}
    calibration=[]
    for lo,hi in zip(np.linspace(-1,0,11)[:-1],np.linspace(-1,0,11)[1:]):
        mask=(prediction>=lo)&(prediction<hi if hi<0 else prediction<=hi)
        if np.any(mask):calibration.append(dict(lo=float(lo),hi=float(hi),frames=int(mask.sum()),
                    prediction_mean=float(prediction[mask].mean()),target_mean=float(targets[mask].mean())))
    temporal={}
    for lag in [1,10,30]:
        pairs=[(targets[e['offset']:e['offset']+e['frames']],prediction[e['offset']:e['offset']+e['frames']]) for e in episodes]
        diffs=np.concatenate([p[lag:]-p[:-lag] for y,p in pairs if len(p)>lag])
        expected=np.concatenate([y[lag:]-y[:-lag] for y,p in pairs if len(p)>lag])
        temporal[str(lag)]=dict(transitions=len(diffs),mean_abs_delta=float(np.abs(diffs).mean()),
            median_abs_delta=float(np.median(np.abs(diffs))),mean_abs_target_delta=float(np.abs(expected).mean()),
            delta_to_target_ratio=float(np.abs(diffs).mean()/np.abs(expected).mean()),
            residual_rmse=float(np.sqrt(np.square(diffs-expected).mean())),
            negative_delta_fraction=float(np.mean(diffs<0)),positive_td_residual_fraction=float(np.mean(diffs>expected)))
    result=dict(checkpoint_sha256=checkpoint_hash,checkpoint_epoch=payload['epoch'],checkpoint_step=payload['global_step'],
        test_episodes=len(episodes),test_frames=len(targets),success_episodes=sum(e['success'] for e in episodes),
        failure_episodes=sum(not e['success'] for e in episodes),groups=rows,bootstrap=bootstrap,calibration=calibration,
        intervention_groups=flag_groups,intervention_events=events,temporal=temporal,
        episode_spearman_median=float(np.median([e['spearman'] for e in episodes if e['spearman'] is not None])),
        negative_episode_spearman_count=sum(e['spearman'] is not None and e['spearman']<0 for e in episodes),
        test_feature_cache=str(cached.path))
    np.savez_compressed(output/'predictions.npz',target=targets,prediction=prediction,timestamp=timestamp,intervention=intervention,**baselines)
    dump(output/'episodes.json',episodes);dump(output/'metrics.json',result)
    dump(output/'audit.json',dict(checkpoint_sha256_before=checkpoint_hash,checkpoint_sha256_after=sha256(checkpoint),
         train_identity_matches_checkpoint=True,splits_disjoint=True,full_test_frames=len(prediction),
         test_cache_digest=cached.manifest['digest'],sources_sha256={str(p.relative_to(ROOT)):sha256(p) for p in
         [Path(__file__),ROOT/'recap_value/model.py',ROOT/'recap_value/evaluation.py',ROOT/'recap_value/runtime.py',ROOT/'recap_value/cache.py']},
         finished_utc=datetime.now(timezone.utc).isoformat()))
    if sha256(checkpoint)!=checkpoint_hash:raise RuntimeError('Checkpoint changed during evaluation')
    print(json.dumps({'output':str(output),'metrics':rows[0],'temporal':temporal},indent=2))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');main()
