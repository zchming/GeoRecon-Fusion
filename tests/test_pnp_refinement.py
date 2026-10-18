import unittest
import numpy as np
from georecon.teaching.incremental import pnp, project
from georecon.teaching.pnp_refinement import refined_pnp, ap3p_pnp, residual_state
try:
    import cv2
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'OpenCV teaching extra not installed')
class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(pnp_min_inliers=12, pnp_iterations=1000, pnp_ransac_px=2., max_reprojection_px=2., seed=42)
        self.k = np.array([[700., 25, 320], [0, 710., 240], [0, 0, 1]])
        self.r = cv2.Rodrigues(np.array([.02, .1, -.03]))[0]
        self.t = np.array([-.8, .02, .01])
        rng = np.random.default_rng(20)
        self.x = rng.uniform([-1, -1, 4], [1, 1, 8], (100, 3))
        self.uv = project(self.x, self.k, self.r, self.t)[0]

    def test_exact_pose_skew_and_world_to_camera(self):
        for solver in [pnp, ap3p_pnp]:
            result = refined_pnp(self.x, self.uv, self.k, self.cfg, base_solver=solver)
            self.assertTrue(result['registered'])
            np.testing.assert_allclose(result['R'], self.r, atol=1e-6)
            np.testing.assert_allclose(result['t'], self.t, atol=1e-6)

    def test_noisy_outlier_support_and_cost_guard(self):
        rng = np.random.default_rng(99)
        uv = self.uv+rng.normal(0, .6, self.uv.shape)
        uv[:20] = rng.uniform([0, 0], [640, 480], (20, 2))
        for solver in [pnp, ap3p_pnp]:
            original = solver(self.x, uv, self.k, self.cfg)
            result = refined_pnp(self.x, uv, self.k, self.cfg, base_solver=solver)
            self.assertTrue(result['registered'])
            self.assertGreaterEqual(result['inliers'].sum(), original['inliers'].sum())
            cost = lambda v: residual_state(self.x, uv, self.k, v['R'], v['t'], 2.)[-1]
            self.assertLessEqual(cost(result), cost(original)+1e-10)
            self.assertFalse(result['inliers'][:20].any())

    def test_rejection_never_resurrected(self):
        for solver in [pnp, ap3p_pnp]:
            out = refined_pnp(self.x[:7], self.uv[:7], self.k, self.cfg, base_solver=solver)
            self.assertFalse(out['registered'])
            self.assertEqual(out['reason'], 'insufficient_2d3d')
            self.assertEqual(out['local_optimization']['iterations'], [])

    def test_negative_depth_penalty(self):
        x = np.array([[0., 0, -1], [0, 0, 1]])
        mask, _, _, cost = residual_state(x, np.array([[0., 0], [0, 0]]), np.eye(3), np.eye(3), np.zeros(3), 2.)
        np.testing.assert_array_equal(mask, [False, True])
        self.assertEqual(cost, 4.)
