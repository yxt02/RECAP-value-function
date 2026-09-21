#!/usr/bin/env python3
"""Verify real cache/online agreement and checkpoint reload on bounded real frames."""
import json
import argparse
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from recap_value.cache import FeatureDataset,cache_location
from recap_value.runtime import load_config,configure_runtime,raw_dataset,make_loader,autocast
from recap_value.model import ValueModel


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train_value.yaml')
    parser.add_argument('--output', default='artifacts/performance/cache-verification.json')
    args=parser.parse_args(argv)
    cfg=load_config(args.config);configure_runtime(cfg)
    ds=raw_dataset(cfg,'train')
    path,digest,_=cache_location(ds,cfg,'train')
    cached=FeatureDataset(path,digest)
    model=ValueModel(str(ROOT/cfg['siglip_path']),cfg['cameras'],True,cfg['precision'],cfg['projection_dim']).cuda().eval()
    checkpoint=ROOT/cfg['save_dir']/'best_model.pt'
    head=None
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['format_version']==2
        assert not any(k.startswith('siglip.') for k in payload['model_state_dict'])
        result=model.load_state_dict(payload['model_state_dict'],strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith('siglip.') for k in result.missing_keys)
        head=ValueModel(str(ROOT/cfg['siglip_path']),cfg['cameras'],True,cfg['precision'],cfg['projection_dim'],load_encoder=False).cuda().eval()
        head.load_state_dict(payload['model_state_dict'],strict=True)
    # Same extraction batch shape avoids confounding BF16 batch-size kernel rounding.
    from torch.utils.data import Subset
    size=cfg['cache_batch_size']
    starts=sorted({0,(len(ds)//2//size)*size,((len(ds)//size)-1)*size,
                   *[(boundary//size)*size for boundary in ds.cumulative_sizes[:-1]]})
    maximum=prediction_error=0.; verified=0; output_dtypes=set()
    for offset in starts:
        loader=make_loader(Subset(ds,list(range(offset,offset+size))),cfg,'val',
                           batch_size=size,workers=0,sequential=True)
        batch=next(iter(loader)); n=len(batch['target_values'])
        images={k:v.cuda() for k,v in batch['images'].items()}
        with torch.no_grad(),autocast(cfg,torch.device('cuda')):
            encoded=model.encode(images)
            online=model(images=images)
            replay=model(features=cached.features[offset:offset+n].cuda())
        saved=cached.features[offset:offset+n].cuda().float()
        error=(encoded.float()-saved).abs().max().item();maximum=max(maximum,error)
        output_dtypes.add(str(encoded.dtype))
        torch.testing.assert_close(encoded.half().float(),saved,atol=0,rtol=0)
        prediction_error=max(prediction_error,(online-replay).abs().max().item())
        torch.testing.assert_close(online,replay,atol=0,rtol=0)
        if head is not None:
            with torch.no_grad(),autocast(cfg,torch.device('cuda')):
                restored=head(features=cached.features[offset:offset+n].cuda())
            torch.testing.assert_close(online,restored,atol=0,rtol=0)
        torch.testing.assert_close(batch['target_values'].float(),cached.targets[offset:offset+n])
        verified+=n
    report={'verified_frames':verified,'batch_start_indices':starts,
            'max_feature_abs_error':maximum,'max_online_vs_cached_prediction_abs_error':prediction_error,
            'encoder_output_dtypes':sorted(output_dtypes),'cache':str(path)}
    report['checkpoint_reload']='passed' if head is not None else 'not tested'
    output=ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
