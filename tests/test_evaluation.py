import unittest

# Imported before numpy: submodules sets single-threaded BLAS, which this environment
# needs for the heavy imports not to crash. See docs/scripts.md.
from submodules.evaluation import regression,paired_episode_bootstrap,fit_baselines,apply_baselines,temporal_metrics

import numpy as np


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_constant_prediction(self):
        y=np.array([-1.,-.5,0.]);p=np.array([-.5,-.5,-.5])
        result=regression(y,p)
        self.assertAlmostEqual(result['mse'],1/6)
        self.assertAlmostEqual(result['mae'],1/3)
        self.assertAlmostEqual(result['r2'],0)
        self.assertIsNone(result['spearman'])
        with self.assertRaises(ValueError):regression(y,p[:2])

    def test_episode_bootstrap_is_paired_and_length_weighted(self):
        y=np.zeros(11);p=np.r_[1.,np.full(10,2.)];baseline=p.copy()
        episodes=[dict(offset=0,frames=1),dict(offset=1,frames=10)]
        stats=paired_episode_bootstrap(y,p,baseline,episodes,1000,3)
        self.assertEqual(stats['mse_difference_model_minus_baseline_ci95'],[0.,0.])
        self.assertEqual(stats['frame_mse_ci95'],[1.,4.])
        self.assertEqual(stats,paired_episode_bootstrap(y,p,baseline,episodes,1000,3))

    def test_elapsed_frame_baseline_does_not_use_test_length_or_outcome(self):
        train=[dict(dataset='A',offset=0,frames=5)]
        y=-.8+.01*np.arange(5)
        fit=fit_baselines(train,y)
        short=apply_baselines(fit,[dict(dataset='A',offset=0,frames=3,success=True)])
        long=apply_baselines(fit,[dict(dataset='A',offset=0,frames=6,success=False)])
        np.testing.assert_allclose(short['dataset_frame_linear'],long['dataset_frame_linear'][:3])
        np.testing.assert_allclose(short['dataset_frame_linear'],y[:3])

    def test_td_residual_zero_for_exact_mc_labels(self):
        y=-.9+np.arange(100)/4000
        stats=temporal_metrics(y,y)
        for lag in [1,10,30]:
            self.assertAlmostEqual(stats[str(lag)]['residual_rmse'],0)
            self.assertAlmostEqual(stats[str(lag)]['mean_abs_delta'],lag/4000)
            self.assertEqual(stats[str(lag)]['negative_delta_fraction'],0)


if __name__=='__main__':unittest.main()
