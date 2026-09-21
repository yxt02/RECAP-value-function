"""Fast CPU regression tests. Run with unittest discover -s tests -v."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Imported before numpy/torch/transformers: submodules sets single-threaded BLAS, which
# this environment needs for those imports not to crash. See docs/scripts.md.
from submodules.cache import FeatureDataset, sha256, cache_identity, prepare_features
from submodules.model import ValueModel
from submodules.runtime import make_loader
from submodules.feature_loader import DeviceFeatureLoader

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from transformers import SiglipVisionConfig

ROOT = Path(__file__).resolve().parents[1]

# The scripts/ directory holds the single workflow implementation.
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import training_workflow as training


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = nn.Parameter(torch.tensor(1.))
    def forward(self, pixels):
        return types.SimpleNamespace(last_hidden_state=pixels*self.scale)


class TrainingTests(unittest.TestCase):
    def model(self, freeze=True):
        with patch('transformers.SiglipVisionConfig.from_pretrained', return_value=SiglipVisionConfig(hidden_size=4)), \
             patch('submodules.model.SiglipVisionModel.from_pretrained', return_value=FakeVision()):
            return ValueModel('unused', ['cam2','cam3'], freeze, 'fp32', 8)

    def test_mean_pool_camera_order_and_frozen_eval(self):
        model = self.model(); model.train()
        self.assertFalse(model.siglip.training)
        self.assertFalse(model.siglip.scale.requires_grad)
        pixels = torch.arange(24.).reshape(2,3,4)
        images = {'observation.images.cam2':pixels, 'observation.images.cam3':pixels+100}
        expected = torch.cat([pixels.mean(1), pixels.mean(1)+100], -1)
        torch.testing.assert_close(model.encode(images), expected)
        result = model(images)
        self.assertTrue(((result >= -1) & (result <= 0)).all())
        result.sum().backward()
        self.assertIsNone(model.siglip.scale.grad)
        self.assertIsNotNone(model.projection.weight.grad)

    def test_frozen_online_forward_matches_storage_precision(self):
        model=self.model()
        pixels=torch.rand(3,2,4)*3.12345
        images={'observation.images.cam2':pixels,'observation.images.cam3':pixels+.012345}
        expected=model(features=model.encode(images).half())
        torch.testing.assert_close(model(images=images),expected,atol=0,rtol=0)

    def test_trainable_encoder_receives_gradients(self):
        model = self.model(False)
        pixels = torch.rand(2,3,4)
        model({'observation.images.cam2':pixels, 'observation.images.cam3':pixels}).sum().backward()
        self.assertIsNotNone(model.siglip.scale.grad)

    def test_capped_loader_exhausts_and_can_shorten_final_epoch(self):
        cfg = dict(num_workers=0, eval_num_workers=0, pin_memory=False, batch_size=3)
        loader = make_loader(TensorDataset(torch.arange(20)), cfg, 'train', max_batches=4)
        self.assertEqual(len(list(loader)), 4)
        loader.batch_sampler.limit = 2
        self.assertEqual(len(list(loader)), 2)
        self.assertEqual(len(loader), 2)

    def test_resident_loader_alignment_coverage_and_cap(self):
        class Features:
            features = torch.arange(28).reshape(7,4).float()
            targets = torch.arange(7).float()
            def __len__(self): return 7
        ds = Features()
        loader = DeviceFeatureLoader(ds,3,True,42,'cpu')
        batches=list(loader)
        targets=torch.cat([b['target_values'] for b in batches])
        features=torch.cat([b['features'] for b in batches])
        torch.testing.assert_close(features[:,0],targets*4)
        torch.testing.assert_close(targets.sort().values,ds.targets)
        self.assertFalse(torch.equal(targets,torch.cat([b['target_values'] for b in loader])))
        loader.batch_sampler.limit=1
        self.assertEqual(len(list(loader)),1)

    def test_small_budget_warmup_is_bounded(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-4)
        schedule = training.build_scheduler(opt, 4, 500)
        rates=[]
        for _ in range(4):
            rates.append(opt.param_groups[0]['lr']); opt.step(); schedule.step()
        self.assertAlmostEqual(max(rates), 1e-4)
        self.assertGreater(min(rates), 0)
        self.assertAlmostEqual(schedule.get_last_lr()[0], 1e-6)

    def test_cache_integrity_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.save(path/'features.npy', np.ones((3,4), dtype=np.float16))
            np.save(path/'targets.npy', np.arange(3,dtype=np.float32)/-4)
            manifest = dict(count=3,feature_dim=4,digest='expected',
                            files_sha256={name:sha256(path/name) for name in ['features.npy','targets.npy']})
            (path/'manifest.json').write_text(json.dumps(manifest))
            ds=FeatureDataset(path,'expected')
            self.assertEqual(len(ds),3)
            self.assertEqual(ds[1]['target_values'].item(),-.25)
            with self.assertRaisesRegex(ValueError, 'identity'):
                FeatureDataset(path,'stale')
            with (path/'features.npy').open('ab') as stream: stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'Corrupted'):
                FeatureDataset(path,'expected')

    def test_checkpoint_head_reload(self):
        model = self.model(); pixels = torch.rand(5,8)
        expected = model(features=pixels)
        clone = self.model()
        state={k:v for k,v in model.state_dict().items() if not k.startswith('siglip.')}
        result=clone.load_state_dict(state,strict=False)
        self.assertEqual(result.missing_keys,['siglip.scale'])
        torch.testing.assert_close(clone(features=pixels),expected)

    def test_training_entry_global_budget_and_early_stop(self):
        class Features(torch.utils.data.Dataset):
            features=torch.arange(40,dtype=torch.float32).reshape(10,4)/40
            targets=torch.full((10,),-.25)
            manifest={}
            def __len__(self): return 10
            def __getitem__(self,index):
                return {'features':self.features[index],'target_values':self.targets[index]}
        class Head(nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__(); self.projection=nn.Linear(4,1); self.siglip=None
            def forward(self,images=None,features=None):
                return (self.projection(features).tanh()-1)*.5
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'manifest.json').write_text('{}')
            for name,budget,patience,delta,expected in [('budget',7,0,0,[5,7]),('early',30,1,1,[5,10])]:
                cfg=dict(adaptation_config='unused',siglip_path='unused',cameras=['cam2'],freeze_vlm=True,
                    projection_dim=4,precision='fp32',feature_cache=True,cache_on_device=False,
                    batch_size=2,cached_batch_size=2,cache_batch_size=2,num_workers=0,eval_num_workers=0,
                    cached_num_workers=0,pin_memory=False,num_epochs=8,log_interval=100,max_steps=None,
                    val_steps=None,max_samples=None,max_total_steps=budget,early_stopping_patience=patience,
                    early_stopping_min_delta=delta,lr=.001,weight_decay=0,clip_grad_norm=1,warmup_steps=2,
                    save_dir=str(root/name))
                with patch.object(training,'load_config',return_value=cfg), \
                     patch.object(training,'raw_dataset',return_value=Features()), \
                     patch.object(training,'cache_location',return_value=(root,'unused',{})), \
                     patch.object(training,'prepare_features',side_effect=lambda ds,*args:ds), \
                     patch.object(training,'ValueModel',Head), \
                     patch('torch.cuda.is_available',return_value=False):
                    training.main_train(['--config','unused'])
                metrics=json.loads((root/name/'metrics.json').read_text())
                self.assertEqual([m['global_step'] for m in metrics],expected)
                self.assertTrue((root/name/'best_model.pt').exists())
                self.assertTrue(all(m['val']['samples']==10 for m in metrics))

    def test_prepare_cache_order_reuse_and_failure_cleanup(self):
        class Data(torch.utils.data.Dataset):
            def __len__(self): return 5
            def __getitem__(self,index):
                return {'images':{'observation.images.cam2':torch.full((2,4),float(index))},
                        'target_values':-index/5}
        class Encoder(nn.Module):
            freeze_vlm=True
            feature_dim=4
            calls=0
            def encode(self,images):
                self.calls+=1
                return images['observation.images.cam2'].mean(1)
        cfg=dict(freeze_vlm=True,cache_batch_size=2,cache_num_workers=0,
                 batch_size=2,pin_memory=False,precision='fp32')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'complete'
            model=Encoder()
            with patch('submodules.cache.cache_location',return_value=(path,'digest',{})):
                cached=prepare_features(Data(),model,cfg,'train',torch.device('cpu'))
                self.assertEqual(model.calls,3)
                torch.testing.assert_close(cached.features[:,0].float(),torch.arange(5).float())
                torch.testing.assert_close(cached.targets,torch.arange(5).float()/-5)
                prepare_features(Data(),model,cfg,'train',torch.device('cpu'))
                self.assertEqual(model.calls,3)
            bad_path=Path(temporary)/'incomplete'
            with patch('submodules.cache.cache_location',return_value=(bad_path,'digest',{})), \
                 patch.object(model,'encode',side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    prepare_features(Data(),model,cfg,'train',torch.device('cpu'))
            self.assertFalse(bad_path.exists())
            self.assertFalse(list(Path(temporary).glob('.building-*')))

    def test_cache_identity_changes_when_targets_or_precision_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['model.safetensors','config.json','preprocessor_config.json','episode.parquet','video.mp4']:
                (root/name).write_bytes(b'test')
            source = types.SimpleNamespace(dataset_path=root,contract={'return_scale':4000},
                index_mapping=np.array([[0,0],[0,1]],dtype=np.int64),
                returns_data={0:{'return':np.array([-2.,-1.])}}, image_transform='fixed',
                parquet_files=[root/'episode.parquet'], episode_ids=[0],cameras=['cam2'],
                video_path=lambda cam,ep:root/'video.mp4')
            class Data:
                datasets=[source]
                def __len__(self): return 2
            cfg=dict(siglip_path=str(root),cameras=['cam2'],precision='bf16')
            original=cache_identity(Data(),cfg,'train')[0]
            source.returns_data[0]['return'][0]=-3.
            changed=cache_identity(Data(),cfg,'train')[0]
            self.assertNotEqual(original,changed)
            cfg['precision']='fp32'
            self.assertNotEqual(changed,cache_identity(Data(),cfg,'train')[0])

    def test_local_image_only_avoids_repeated_parquet_reads(self):
        from submodules.datasets import SimpleValueDataset
        path=ROOT/'data/raw/2026.09.16_error'
        if not path.is_dir(): self.skipTest('local data absent')
        ds=SimpleValueDataset(path,episodes=[26],cameras=['cam2'],include_state=False,include_actions=False)
        with patch('submodules.datasets.pq.read_table',side_effect=AssertionError('parquet reread')):
            sample=ds[0]
        self.assertEqual(tuple(sample['images']['observation.images.cam2'].shape),(3,224,224))
        from transformers import SiglipImageProcessor
        processor=SiglipImageProcessor.from_pretrained(str(ROOT/'models/siglip2-so400m-patch14-224'),local_files_only=True)
        reference=processor(images=ds._read_video_frame('cam2',26,0),return_tensors='pt')['pixel_values'][0]
        torch.testing.assert_close(sample['images']['observation.images.cam2'],reference,atol=1e-6,rtol=1e-6)
        self.assertEqual(float(ds.returns_data[26]['return'][-1])/ds.return_scale,-.5)
        ds.close()
        joint_ds=SimpleValueDataset(path,episodes=[26],cameras=[],include_state=True,include_actions=True)
        sample=joint_ds[-1]
        self.assertEqual(tuple(sample['observation/state'].shape),(32,))
        self.assertTrue((sample['observation/state'][22:]==0).all())
        self.assertTrue(sample['action_is_pad'][1:].all())
        torch.testing.assert_close(sample['actions'][0],sample['actions'][-1])
        joint_ds.close()

    def test_episode_splits_and_failure_override(self):
        split_dir=ROOT/'data/splits'
        if not split_dir.is_dir(): self.skipTest('local data absent')
        sets=[]; counts=[246913,28359,34759]; failures=[]
        for split,count in zip(['train','val','test'],counts):
            data=json.loads((split_dir/f'{split}.json').read_text())
            self.assertEqual(data['total_frames'],count)
            episodes=data['episodes']
            keys={(e['dataset'],e['episode_index']) for e in episodes}
            self.assertEqual(len(keys),len(episodes)); sets.append(keys)
            failures += [e for e in episodes if e['dataset']=='2026.09.16_error']
        for i in range(3):
            for j in range(i+1,3): self.assertFalse(sets[i]&sets[j])
        self.assertEqual(len(failures),100)
        self.assertTrue(all(not e['is_success'] for e in failures))
        ep26=next(e for e in failures if e['episode_index']==26)
        self.assertEqual(ep26['raw_terminal_reward'],0)


if __name__ == '__main__':
    unittest.main()
