#!/usr/bin/env python3
"""Independent replay of every stored prediction; check frame targets and frozen checkpoint."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recap_value.cache import FeatureDataset,sha256
from recap_value.model import ValueModel
from recap_value.runtime import configure_runtime,autocast


def main():
    root=Path(sys.argv[1]).resolve()
    protocol=json.loads((root/'protocol.json').read_text());metrics=json.loads((root/'metrics.json').read_text())
    checkpoint=Path(protocol['checkpoint']);assert sha256(checkpoint)==protocol['checkpoint_sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=payload['config'];configure_runtime(cfg)
    dataset=FeatureDataset(metrics['test_feature_cache'])
    model=ValueModel(str(ROOT/cfg['siglip_path']),cfg['cameras'],True,cfg['precision'],cfg['projection_dim'],load_encoder=False).cuda().eval()
    model.load_state_dict(payload['model_state_dict'],strict=True)
    feature=dataset.features.cuda();predictions=[]
    with torch.inference_mode(),autocast(cfg,torch.device('cuda')):
        for offset in range(0,len(dataset),cfg['cached_batch_size']):
            predictions.append(model(features=feature[offset:offset+cfg['cached_batch_size']]).float().cpu().numpy().ravel())
    actual=np.concatenate(predictions);saved=np.load(root/'predictions.npz')
    np.testing.assert_array_equal(actual,saved['prediction'])
    eps=json.loads((root/'episodes.json').read_text());expected=[]
    for ep in eps:
        expected.extend((np.arange(ep['frames'],dtype=float)-(ep['frames']-1)-(0 if ep['success'] else 2000))/4000)
    np.testing.assert_allclose(expected,saved['target'],atol=1e-15,rtol=0)
    mse=float(np.square(actual-saved['target']).mean())
    np.testing.assert_allclose(mse,metrics['groups'][0]['mse'],atol=1e-15,rtol=0)
    assert sha256(checkpoint)==protocol['checkpoint_sha256']
    output=dict(replayed_frames=len(actual),max_prediction_error=float(np.max(np.abs(actual-saved['prediction']))),
        targets_match_reward_formula=True,checkpoint_unchanged=True,recomputed_mse=mse)
    (root/'independent_replay.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
