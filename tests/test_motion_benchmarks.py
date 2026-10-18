import unittest
import numpy as np
from georecon.data.motion_benchmarks import quaternion_matrix, tum_reference, colmap_references


class MotionReferences(unittest.TestCase):
    def test_tum_c2w_optical_center_and_missing_time(self):
        k = np.diag([500., 500., 1.]); text = '# header\n1 1 2 3 0 0 0 1\n'
        camera, gap = tum_reference(text, 1.01, k)
        np.testing.assert_allclose(camera.center, [1, 2, 3])
        np.testing.assert_allclose(camera.t, [-1, -2, -3])
        self.assertAlmostEqual(gap, .01)
        self.assertIsNone(tum_reference(text, 1.03, k)[0])

    def test_colmap_translation_is_not_center(self):
        text = '# header\n1 SIMPLE_RADIAL 640 480 500 320 240 -0.02\n'
        images = '# header\n1 0.70710678 0 0 0.70710678 1 2 3 1 a.jpg\n0 0 -1\n'
        camera, distortion = colmap_references(text, images)['a.jpg']
        np.testing.assert_allclose(camera.t, [1, 2, 3])
        np.testing.assert_allclose(camera.R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-7)
        np.testing.assert_allclose(camera.R@camera.center+camera.t, 0, atol=1e-7)
        self.assertEqual(distortion[0], -.02)

    def test_bad_quaternion_rejected(self):
        with self.assertRaises(ValueError): quaternion_matrix([0, 0, 0, 0])
        with self.assertRaises(ValueError): quaternion_matrix([float('nan'), 0, 0, 1])
