#!/usr/bin/env python3
"""Local visual value baseline: measured single-GPU training and frozen features.

This is not the language-conditioned 201-bin RECAP critic. See README.
"""
import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from recap_datasets.recap.contracts import resolve_path
from recap_value.model import ValueModel
from recap_value.runtime import configure_runtime, load_config, raw_dataset, make_loader, autocast, move_batch
from recap_value.cache import prepare_features, cache_location, FeatureDataset

logger = logging.getLogger(__name__)


def epoch(model, loader, config, device, optimizer=None, scheduler=None, max_steps=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    count = steps = 0
    start = time.perf_counter()
    limit = min(len(loader), int(max_steps)) if max_steps is not None else len(loader)
    if limit != len(loader):
        raise ValueError('Set max_batches on make_loader so workers are not left decoding prefetched batches')
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(config, device):
                predictions = model(images=batch.get('images'), features=batch.get('features'))
                loss = nn.functional.mse_loss(predictions.float(), batch['target_values'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['clip_grad_norm'], error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
            n = len(predictions); count += n; steps += 1
            loss_sum += loss.detach() * n
            if training and steps % int(config['log_interval']) == 0:
                logger.info('Step %d/%d, MSE %.6f, LR %.3g', steps, limit, (loss_sum/count).item(), scheduler.get_last_lr()[0])
    if count == 0:
        raise ValueError('No samples processed')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return {'mse':(loss_sum/count).item(), 'samples':count, 'steps':steps,
            'seconds':time.perf_counter()-start, 'samples_per_second':count/(time.perf_counter()-start)}


def build_scheduler(optimizer, total_steps, warmup_steps):
    warmup = min(max(0, int(warmup_steps)), max(0, total_steps-1))
    def multiplier(step):
        if warmup and step < warmup:
            return .01 + .99*step/warmup
        progress = (step-warmup)/max(1, total_steps-warmup)
        return .01 + .99*.5*(1+math.cos(math.pi*min(1., progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/train_value.yaml')
    p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--prepare-cache', action='store_true', help='Build train/val features and exit; no optimizer steps')
    p.add_argument('--no-cache', action='store_true')
    for key, typ in [('batch_size',int),('num_epochs',int),('lr',float),('warmup_steps',int),
                     ('max_total_steps',int),('early_stopping_patience',int),('max_samples',int),('max_steps',int),('val_steps',int),('num_workers',int),('cache_batch_size',int),
                     ('cached_batch_size',int),('save_dir',str)]:
        p.add_argument('--'+key, type=typ)
    args = p.parse_args(argv)
    config = load_config(args.config)
    for key,value in vars(args).items():
        if key not in ('config','smoke_test','prepare_cache','no_cache') and value is not None:
            config[key] = value
    if args.no_cache:
        config['feature_cache'] = False
    if args.smoke_test:
        config.update(num_epochs=1, max_steps=4, val_steps=2,
                      cached_batch_size=min(64, config['cached_batch_size']),
                      max_samples=min(config.get('max_samples') or 256, 256),
                      save_dir='artifacts/performance/smoke-cached' if config['feature_cache'] else 'artifacts/performance/smoke-online')
    configure_runtime(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config['precision']=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support BF16; select fp32')
    if device.type=='cpu':
        config['precision']='fp32'
    for key in ('batch_size','cached_batch_size','cache_batch_size','num_epochs','log_interval'):
        if int(config[key]) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('max_steps','val_steps','max_samples','max_total_steps'):
        if config[key] is not None and int(config[key]) < 1:
            raise ValueError(f'{key} must be positive or null')
    if config['early_stopping_patience'] < 0 or config['early_stopping_min_delta'] < 0:
        raise ValueError('Early stopping patience and min_delta must be nonnegative')
    if args.prepare_cache and not config['feature_cache']:
        raise ValueError('--prepare-cache requires feature_cache=true')
    logger.info('Device %s, configuration %s', device, json.dumps(config))
    train_data, val_data = raw_dataset(config,'train'), raw_dataset(config,'val')
    logger.info('Manifest-filtered samples: train=%d val=%d',len(train_data),len(val_data))
    all_cached = config['feature_cache'] and all((cache_location(d,config,s)[0]/'manifest.json').exists()
                                                for d,s in [(train_data,'train'),(val_data,'val')])
    model = ValueModel(str(resolve_path(config['siglip_path'])), config['cameras'], config['freeze_vlm'],
                       config['precision'], config['projection_dim'], load_encoder=not all_cached).to(device)
    if config['feature_cache']:
        train_data = prepare_features(train_data,model,config,'train',device)
        val_data = prepare_features(val_data,model,config,'val',device)
        if args.prepare_cache:
            logger.info('Feature caches ready; no training performed')
            return
        model.siglip = None  # Frozen encoder not needed during cached epochs or head checkpoints.
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Once at phase boundary, never in the training loop.
        train_loader = make_loader(train_data,config,'train',batch_size=config['cached_batch_size'],workers=config['cached_num_workers'],max_batches=config['max_steps'] or config['max_total_steps'] or sys.maxsize)
        val_loader = make_loader(val_data,config,'val',batch_size=config['cached_batch_size'],workers=config['cached_num_workers'],max_batches=config['val_steps'])
    else:
        train_loader = make_loader(train_data,config,'train',max_batches=config['max_steps'] or config['max_total_steps'] or sys.maxsize)
        val_loader = make_loader(val_data,config,'val',max_batches=config['val_steps'])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config['lr'], weight_decay=config['weight_decay'], fused=device.type=='cuda')
    steps_per_epoch = min(len(train_loader),config['max_steps']) if config['max_steps'] else len(train_loader)
    total_steps = min(config['num_epochs']*steps_per_epoch, config['max_total_steps'] or sys.maxsize)
    scheduler = build_scheduler(optimizer, total_steps, config['warmup_steps'])
    logger.info('Budget: at most %d epochs / %d optimizer steps; %d batches per full epoch', config['num_epochs'], total_steps, steps_per_epoch)
    save_dir = resolve_path(config['save_dir']); save_dir.mkdir(parents=True,exist_ok=True)
    (save_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    best = early_best = float('inf'); history=[]
    global_step = stale_epochs = 0
    initial = {name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad} if args.smoke_test else None
    for ep in range(config['num_epochs']):
        remaining = total_steps-global_step
        if remaining <= 0:
            break
        train_loader.batch_sampler.limit = min(steps_per_epoch, remaining)
        training = epoch(model,train_loader,config,device,optimizer,scheduler)
        global_step += training['steps']
        validation = epoch(model,val_loader,config,device,max_steps=config['val_steps'])
        record = {'epoch':ep+1,'global_step':global_step,'train':training,'val':validation}
        history.append(record); logger.info('%s',json.dumps(record))
        if validation['mse'] < best:
            best = validation['mse']
            # Save frozen encoder identity/path via config/cache manifest, not 1.7GB weights every epoch.
            state = {k:v for k,v in model.state_dict().items() if not (config['freeze_vlm'] and k.startswith('siglip.'))}
            tmp=save_dir/'best_model.pt.tmp'
            torch.save({'format_version':2,'architecture':'siglip_mean_patch_scalar','epoch':ep+1,
                        'global_step':global_step,'config':config,'model_state_dict':state,'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),'val_loss':best,
                        'feature_cache_manifest':train_data.manifest if config['feature_cache'] else None},tmp)
            tmp.replace(save_dir/'best_model.pt')
        (save_dir/'metrics.json').write_text(json.dumps(history,indent=2)+'\n')
        if validation['mse'] < early_best-float(config['early_stopping_min_delta']):
            early_best = validation['mse']; stale_epochs = 0
        else:
            stale_epochs += 1
        if config['early_stopping_patience'] and stale_epochs >= config['early_stopping_patience']:
            logger.info('Early stopping after %d validation checks without sufficient improvement', stale_epochs)
            break
    if args.smoke_test:
        changed=any(not torch.equal(initial[name],param) for name,param in model.named_parameters() if param.requires_grad)
        if not changed:
            raise AssertionError('Optimizer did not update the head')
        logger.info('Smoke passed: real train/val batches, finite backward gradients and parameter update')
    (save_dir/'metrics.json').write_text(json.dumps(history,indent=2)+'\n')


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
