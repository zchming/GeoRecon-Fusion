import unittest
import numpy as np
from numpy.testing import assert_allclose
from georecon.geometry.camera import Camera, resize_crop, relative_pose, triangulate, decompose_projection
from georecon.geometry.alignment import align_points
from georecon.evaluation.poses import rotation_error_deg, translation_direction_error_deg


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 0, 49.5], [0, 120, 39.5], [0, 0, 1]])
        self.first = Camera(self.K, np.eye(3), np.zeros(3), 100, 80)
        self.second = Camera(self.K, np.eye(3), np.array([-1., 0, 0]), 100, 80)

    def test_projection_known_numbers_and_depth(self):
        pixels, z, mask = self.first.project([[1., 2, 4], [0, 0, -1], [0, 0, 0]])
        assert_allclose(pixels[0], [74.5, 99.5])
        assert_allclose(z, [4, -1, 0])
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertTrue(np.isnan(pixels[1:]).all())

    def test_camera_center_and_nonidentity_roundtrip(self):
        R = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        camera = Camera(self.K, R, [1., 2, 3], 100, 80)
        assert_allclose(camera.center, [-2, 1, -3])
        points = np.array([[1., 0, 2], [0, 1, 4]])
        pixels, depth, _ = camera.project(points)
        assert_allclose(camera.unproject(pixels, depth), points, atol=1e-12)

    def test_z_depth_not_euclidean_range(self):
        point = self.first.unproject([[149.5, 39.5]], [2])[0]
        assert_allclose(point, [2, 0, 2])
        self.assertNotAlmostEqual(np.linalg.norm(point), 2)

    def test_invalid_camera_and_unprojection(self):
        for R in (np.diag([1, 1, -1]), np.eye(3)*2):
            with self.assertRaises(ValueError):
                Camera(self.K, R, [0, 0, 0], 100, 80)
        with self.assertRaises(ValueError):
            self.first.unproject([[0, 0]], [-1])
        with self.assertRaises(ValueError):
            Camera(self.K, np.eye(3), [0, 0, float("nan")], 100, 80)

    def test_resize_crop_pixel_centers_and_projection(self):
        transformed, A = resize_crop(self.first, [200, 120], [10, 5, 180, 100])
        assert_allclose(transformed.K, [[200, 0, 89.5], [0, 180, 54.5], [0, 0, 1]])
        p = [[0.2, 0.3, 2.]]
        original = self.first.project(p)[0][0]
        expected = [2*(original[0]+0.5)-0.5-10, 1.5*(original[1]+0.5)-0.5-5]
        assert_allclose(transformed.project(p)[0][0], expected)
        assert_allclose(A @ np.r_[original, 1], np.r_[expected, 1])
        with self.assertRaises(ValueError):
            resize_crop(self.first, [200, 120], [190, 0, 20, 100])

    def test_relative_pose_has_expected_baseline(self):
        R, t = relative_pose(self.first, self.second)
        assert_allclose(R, np.eye(3))
        assert_allclose(t, [-1, 0, 0])

    def test_projection_decomposition_under_positive_negative_scale(self):
        R = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        original = Camera(self.K, R, [2., 3., 4.], 100, 80)
        for scale in (0.2, -3.7):
            rebuilt = decompose_projection(scale*original.projection, 100, 80)
            assert_allclose(rebuilt.K, original.K, atol=1e-10)
            assert_allclose(rebuilt.R, original.R, atol=1e-10)
            assert_allclose(rebuilt.t, original.t, atol=1e-10)
        with self.assertRaises(ValueError):
            decompose_projection(np.zeros((3, 4)), 100, 80)

    def test_triangulation_known_disparity(self):
        # u1-u2 = fx*baseline/Z = 25 pixels -> Z=4.
        points, valid = triangulate(self.first, self.second, [[49.5, 39.5]], [[24.5, 39.5]])
        self.assertTrue(valid[0])
        assert_allclose(points, [[0, 0, 4]], atol=1e-10)

    def test_triangulation_negative_depth_and_zero_parallax(self):
        points, valid = triangulate(self.first, self.second, [[49.5, 39.5]], [[74.5, 39.5]])
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(points).all())
        _, valid = triangulate(self.first, self.first, [[49.5, 39.5]], [[49.5, 39.5]])
        self.assertFalse(valid[0])
        _, valid = triangulate(self.first, self.second, [[49.5, 39.5]], [[49.5, 39.5]])
        self.assertFalse(valid[0])

    def test_sim3_and_se3_known_transform(self):
        source = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Q = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        for scale in (1., 2.5):
            target = scale*source @ Q.T + [3, -2, 5]
            fit = align_points(source, target, estimate_scale=scale != 1)
            self.assertAlmostEqual(fit.scale, scale)
            assert_allclose(fit.apply(source), target, atol=1e-12)
            assert_allclose(fit.R, Q, atol=1e-12)

    def test_alignment_degenerate_and_reflection(self):
        for source in ([[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]):
            with self.assertRaises(ValueError):
                align_points(source, source)
        source = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        target = source*np.array([-1, 1, 1])
        fit = align_points(source, target)
        self.assertAlmostEqual(np.linalg.det(fit.R), 1)
        self.assertGreater(np.linalg.norm(fit.apply(source)-target), 0.1)

    def test_error_units_and_direction_sign(self):
        Q = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertAlmostEqual(rotation_error_deg(Q, np.eye(3)), 90)
        self.assertAlmostEqual(translation_direction_error_deg([1, 0, 0], [-1, 0, 0]), 180)
        self.assertAlmostEqual(translation_direction_error_deg([1, 0, 0], [0, 1, 0]), 90)
        self.assertIsNone(translation_direction_error_deg([0, 0, 0], [1, 0, 0]))
