"""Numerical boundary cases for trajectory-local scores and checkpoint compatibility."""
import unittest
import warnings
import submodules
import numpy as np
import torch
from submodules.advantage import compute_advantage, label_scores, intervention_windows
from submodules.contracts import episode_returns
from submodules.checkpoint import distribution_bins, ARCHITECTURE
from submodules.training_workflow import targets_to_bin_indices


class AdvantageTests(unittest.TestCase):
    def test_exact_returns_zero_residual_including_terminal(self):
        for success in (False, True):
            for gamma in (0., .9, 1.):
                returns, rewards = episode_returns(80, success, gamma, -2000.)
                for horizon in (1, 2, 50, 100):
                    result = compute_advantage(returns/4000., rewards, horizon, 4000., gamma)
                    np.testing.assert_allclose(result['advantage_continuous'], 0, atol=1e-7)
                    self.assertEqual(result['effective_horizon'][-1], 1)
                    self.assertEqual(result['value_next'][-1], 0)
                    self.assertTrue(result['terminal_reached'][-1])

    def test_interior_and_failure_terminal(self):
        v=np.linspace(-.8,-.1,100);r=np.full(100,-1.);r[-1]=-2000
        a=compute_advantage(v,r)
        self.assertAlmostEqual(a['advantage_continuous'][0],v[50]-v[0]-50/4000)
        self.assertAlmostEqual(a['advantage_continuous'][-1],-.5-v[-1])
        self.assertEqual(a['effective_horizon'][90],10)
        self.assertEqual(a['reward_sum_raw'][90],-2009)

    def test_labels_strict_threshold(self):
        self.assertEqual(label_scores([-.1,0,.1]).tolist(),['disadvantage','disadvantage','advantage'])
        self.assertEqual(label_scores([.1,.2],.1).tolist(),['disadvantage','advantage'])

    def test_reject_invalid_inputs(self):
        for h in (0,-1,1.5,True):
            with self.assertRaises(ValueError):compute_advantage([0],[-1],h)
        with self.assertRaises(ValueError):compute_advantage([np.nan],[-1])
        with self.assertRaises(ValueError):compute_advantage([0],[-1,-1])
        with self.assertRaises(ValueError):label_scores([1],np.nan)

    def test_intervention_alignment_and_missing_flags(self):
        times=np.arange(0,5,.1);score=times.copy();flags=np.zeros(50);flags[:3]=1;flags[20:25]=1;flags[48:]=1
        grid,events=intervention_windows(times,score,flags)
        self.assertEqual(events.shape,(1,61))
        np.testing.assert_allclose(events[0],2+grid)
        self.assertEqual(intervention_windows(times,score,None)[1].shape,(0,61))
        with self.assertRaises(ValueError):intervention_windows(times,score,np.full(50,2))

    def test_midpoint_bin_labels_and_single_sample_shape(self):
        centers=(torch.arange(201)+.5)/201-1
        torch.testing.assert_close(targets_to_bin_indices(centers,201),torch.arange(201))
        torch.testing.assert_close(targets_to_bin_indices(torch.tensor([[-1.],[0.]]),201),torch.tensor([0,200]))
        self.assertEqual(targets_to_bin_indices(torch.tensor([-.5]),201).shape,(1,))

    def test_checkpoint_legacy_label_but_not_scalar_weights(self):
        p=dict(format_version=2,architecture=ARCHITECTURE,config={'num_bins':3},
               model_state_dict={'bin_centers':torch.tensor([-.8,-.5,-.2]),'value_head.2.weight':torch.zeros(3,256)})
        self.assertEqual(distribution_bins(p),3)
        p['architecture']='siglip_mean_patch_scalar'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');self.assertEqual(distribution_bins(p),3)
            self.assertEqual(len(caught),1)
        del p['model_state_dict']['bin_centers']
        with self.assertRaises(ValueError):distribution_bins(p)


if __name__ == '__main__':unittest.main()
