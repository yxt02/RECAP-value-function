#!/usr/bin/env python3
"""Bounded GPU and data throughput probes; no long training run."""
import argparse
import gc
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from recap_value.model import ValueModel
from recap_value.runtime import load_config,configure_runtime,raw_dataset,make_loader,autocast,move_batch


def timed_gpu(model,config,batch_size,steps=12):
    steps=max(steps,1024//batch_size)
    images={'observation.images.cam2':torch.rand(batch_size,3,224,224,device='cuda')*2-1}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    target=torch.full((batch_size,1),-.5,device='cuda')
    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast(config,torch.device('cuda')):
            prediction=model(images)
            loss=(prediction.float()-target).square().mean()
        loss.backward();optimizer.step()
    for _ in range(3):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for _ in range(steps):step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return {'batch_size':batch_size,'precision':config['precision'],'steps':steps,
            'samples_per_second':steps*batch_size/elapsed,'seconds':elapsed,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['gpu','loader','head'],required=True)
    p.add_argument('--config',default='config/train_value.yaml')
    args=p.parse_args(argv);cfg=load_config(args.config);configure_runtime(cfg)
    output=ROOT/'artifacts/performance';output.mkdir(parents=True,exist_ok=True)
    results=[]
    if args.mode=='gpu':
        # Historical first-patch/Gemma-allocation implementation, same GPU-resident input.
        spec=importlib.util.spec_from_file_location('historical_training',ROOT/'recap_value/reference/train_value.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=module.ValueModel(str(ROOT/cfg['siglip_path']),str(ROOT/'models/gemma-3-270m'),True).cuda()
        baseline={**cfg,'precision':'fp32'}
        r=timed_gpu(model,baseline,2);r['implementation']='historical_fp32';results.append(r);print(json.dumps(r),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
        model=ValueModel(str(ROOT/cfg['siglip_path']),cfg['cameras'],True,'bf16').cuda()
        for size in [8,16,32,64,128]:
            try:
                r=timed_gpu(model,cfg,size);r['implementation']='optimized_bf16';results.append(r);print(json.dumps(r),flush=True)
                if r['peak_reserved_gib']>10.5:break
            except torch.cuda.OutOfMemoryError:
                results.append({'batch_size':size,'oom':True});torch.cuda.empty_cache();break
    elif args.mode=='loader':
        cfg['max_samples']=2048
        ds=raw_dataset(cfg,'train')
        for workers in [0,2,4,8]:
            loader=make_loader(ds,cfg,'train',batch_size=32,workers=workers,max_batches=28)
            start=time.perf_counter();iterator=iter(loader)
            for _ in range(4):next(iterator)
            setup=time.perf_counter()-start
            start=time.perf_counter();count=0
            for _ in range(24):count+=len(next(iterator)['target_values'])
            elapsed=time.perf_counter()-start
            for _ in iterator: pass
            r={'workers':workers,'batch_size':32,'samples_per_second':count/elapsed,'measured_samples':count,'startup_and_warmup_seconds':setup}
            results.append(r);print(json.dumps(r),flush=True)
            del iterator,loader;gc.collect()
    else:
        from recap_value.cache import cache_location, FeatureDataset
        from recap_value.workflows.training import epoch, build_scheduler
        ds=raw_dataset(cfg,'train')
        path,digest,_=cache_location(ds,cfg,'train')
        ds=FeatureDataset(path,digest)
        for resident in [False,True]:
            cfg['cache_on_device']=resident
            for size in [256,512,1024,2048]:
                model=ValueModel(str(ROOT/cfg['siglip_path']),cfg['cameras'],True,cfg['precision'],cfg['projection_dim'],load_encoder=False).cuda()
                loader=make_loader(ds,cfg,'train',batch_size=size,workers=0)
                opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],fused=True)
                sched=build_scheduler(opt,len(loader)*3,0)
                epoch(model,loader,cfg,torch.device('cuda'),opt,sched)
                trials=[epoch(model,loader,cfg,torch.device('cuda'),opt,sched) for _ in range(2)]
                r={'batch_size':size,'gpu_resident':resident,'samples':len(ds),'trials':trials}
                results.append(r);print(json.dumps(r),flush=True)
                del model,loader,opt,sched;gc.collect();torch.cuda.empty_cache()
    (output/f'{args.mode}_benchmark.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
