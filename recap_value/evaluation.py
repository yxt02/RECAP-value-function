"""Read-only evaluation statistics; bootstrap independent episodes, never frames."""
import numpy as np
from scipy.stats import spearmanr


def regression(y, p):
    y,p=np.asarray(y,dtype=np.float64),np.asarray(p,dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y)==0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Expected aligned finite nonempty one-dimensional predictions/targets')
    e=p-y; mse=float(np.mean(e*e)); variance=float(np.var(y))
    rho=float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else None
    return dict(frames=len(y),mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.abs(e).mean()),
                bias=float(e.mean()),p90_absolute_error=float(np.quantile(np.abs(e),.90)),
                r2=1-mse/variance if variance>0 else None,spearman=rho)


def paired_episode_bootstrap(target,prediction,baseline,episodes,repeats=4000,seed=20260917):
    """Paired, unstratified cluster bootstrap over episodes; length weighting explicit."""
    rows=[]
    for ep in episodes:
        sl=slice(ep['offset'],ep['offset']+ep['frames'])
        err=prediction[sl]-target[sl]; base=baseline[sl]-target[sl]
        rows.append([len(err),np.square(err).sum(),np.abs(err).sum(),np.square(base).sum()])
    rows=np.asarray(rows,dtype=np.float64)
    draws=np.random.default_rng(seed).integers(0,len(rows),(repeats,len(rows)))
    summed=rows[draws].sum(axis=1)
    mse=summed[:,1]/summed[:,0]; base_mse=summed[:,3]/summed[:,0]
    macro=np.mean((rows[:,1]/rows[:,0])[draws],axis=1)
    def interval(x): return [float(v) for v in np.quantile(x,[.025,.975])]
    return dict(repeats=repeats,unit='episode',seed=seed,
        frame_mse_ci95=interval(mse),frame_rmse_ci95=interval(np.sqrt(mse)),
        frame_mae_ci95=interval(summed[:,2]/summed[:,0]),episode_macro_mse_ci95=interval(macro),
        mse_difference_model_minus_baseline_ci95=interval(mse-base_mse),
        relative_mse_reduction_ci95=interval(1-mse/base_mse))


def fit_baselines(episodes,target):
    means={}; linear={}
    for name in sorted({e['dataset'] for e in episodes}):
        eps=[e for e in episodes if e['dataset']==name]
        indices=np.concatenate([np.arange(e['offset'],e['offset']+e['frames']) for e in eps])
        frames=np.concatenate([np.arange(e['frames']) for e in eps]).astype(float)
        # Only current elapsed frame index; no outcome, episode length or future observations.
        design=np.column_stack([np.ones(len(frames)),frames/4000.])
        means[name]=float(target[indices].mean())
        linear[name]=np.linalg.lstsq(design,target[indices],rcond=None)[0].tolist()
    return dict(global_mean=float(target.mean()),dataset_means=means,dataset_frame_linear=linear)


def apply_baselines(fit,episodes):
    count=sum(e['frames'] for e in episodes)
    result={k:np.empty(count,dtype=np.float64) for k in ('global_mean','dataset_mean','dataset_frame_linear')}
    for ep in episodes:
        sl=slice(ep['offset'],ep['offset']+ep['frames']); name=ep['dataset']
        result['global_mean'][sl]=fit['global_mean']
        result['dataset_mean'][sl]=fit['dataset_means'].get(name,fit['global_mean'])
        a,b=fit['dataset_frame_linear'].get(name,[fit['global_mean'],0])
        result['dataset_frame_linear'][sl]=np.clip(a+b*np.arange(ep['frames'])/4000.,-1,0)
    return result


def temporal_metrics(y,p,lags=(1,10,30)):
    result={}
    for lag in lags:
        if len(y)<=lag: continue
        delta=np.diff(p) if lag==1 else p[lag:]-p[:-lag]
        expected=y[lag:]-y[:-lag]
        residual=delta-expected
        result[str(lag)]=dict(transitions=len(delta),mean_delta=float(delta.mean()),
            mean_abs_delta=float(np.abs(delta).mean()),median_abs_delta=float(np.median(np.abs(delta))),
            mean_abs_target_delta=float(np.abs(expected).mean()),
            residual_rmse=float(np.sqrt(np.square(residual).mean())),
            residual_mae=float(np.abs(residual).mean()),
            negative_delta_fraction=float(np.mean(delta<0)),
            positive_td_residual_fraction=float(np.mean(residual>0)))
    return result
