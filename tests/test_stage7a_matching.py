"""CPU-only controlled-experiment contracts; no DTU/weights/GPU required."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from georecon.geometry.camera import Camera
from georecon.matching.features import (array_digest, matching_inputs, validate_features,
    validate_pairs, ratio_match, extract, ResourceGuard)
from georecon.matching.metrics import support_metrics, pose_metrics, summarize, decision
from georecon.matching.session import load_protocol
from georecon.matching.worker import check_access

ROOT = Path(__file__).resolve().parents[1]
try:
    import cv2
except ImportError:
    cv2 = None


def feature(n=2):
    raw = np.eye(n, 128, dtype=np.float32)
    return dict(keypoints=np.ones((n, 2), np.float32), descriptors_raw=raw,
        descriptors_root=raw.copy(), scales=np.ones(n, np.float32),
        oris=np.zeros(n, np.float32), image_size=np.array([800, 600], np.float32))


class FeatureContract(unittest.TestCase):
    def test_shared_digest_covers_all_root_inputs(self):
        f = feature()
        base = array_digest(matching_inputs(f))
        for key in matching_inputs(f):
            g = copy.deepcopy(f)
            g[key].flat[0] += .1
            self.assertNotEqual(base, array_digest(matching_inputs(g)), key)
        f['descriptors_raw'] *= 2
        self.assertEqual(base, array_digest(matching_inputs(f)))

    def test_empty_features_and_resource_guard(self):
        self.assertEqual(validate_features(feature(0)), 0)
        with self.assertRaises(ResourceGuard):
            validate_features({'keypoints': np.zeros((4609, 2))})

    def test_invalid_coordinates_angles_shapes_and_norms(self):
        for key, value in [('keypoints', -1), ('oris', 7), ('scales', 0), ('descriptors_root', .01)]:
            f = feature()
            f[key].flat[0] = value
            with self.assertRaises(ValueError, msg=key):
                validate_features(f)
        f = feature()
        f['descriptors_raw'] = f['descriptors_raw'].astype(float)
        with self.assertRaises(ValueError):
            validate_features(f)

    def test_indices_unique_integer_bounded_and_sorted(self):
        self.assertEqual(validate_pairs(np.array([[1, 0], [0, 1]]), 2, 2).tolist(), [[0, 1], [1, 0]])
        for a in [np.array([[0, 0], [0, 1]]), np.array([[-1, 0]]), np.array([[0, 2]]), np.array([[0., 1.]])]:
            with self.assertRaises(ValueError):
                validate_pairs(a, 2, 2)

    @unittest.skipIf(cv2 is None, 'OpenCV unavailable; no feature/matcher execution')
    def test_ratio_mutuality_ties_and_empty(self):
        a = np.eye(3, 128, dtype=np.float32)
        b = a[[2, 0, 1]]
        self.assertEqual(ratio_match(a, b)[0].tolist(), [[0, 1], [1, 2], [2, 0]])
        self.assertEqual(len(ratio_match(a[:1], b)[0]), 0)
        self.assertEqual(len(ratio_match(np.zeros_like(a), np.zeros_like(b))[0]), 0)

    @unittest.skipIf(cv2 is None, 'OpenCV unavailable; no feature/matcher execution')
    def test_historical_detector_and_matcher_parity_synthetic(self):
        from georecon.teaching.two_view import features_and_matches
        rgb = np.random.default_rng(42).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        spec = load_protocol(ROOT)['features']
        images = [rgb, np.roll(rgb, 3, axis=1)]
        f = [extract(v, spec)[0] for v in images]
        cfg = json.loads((ROOT/'configs/teaching/stage6_two_view.json').read_text())
        old, _ = features_and_matches(images, cfg)
        np.testing.assert_array_equal(f[0]['keypoints'], old['keypoints_first'])
        np.testing.assert_array_equal(f[1]['keypoints'], old['keypoints_second'])
        np.testing.assert_array_equal(ratio_match(f[0]['descriptors_raw'], f[1]['descriptors_raw'])[0], old['match_indices'])

    def test_gt_and_network_guard(self):
        for path in ['/data/reference.json', '/data/ground_truth/a', '/calibration/cameras.npz', '/x/evaluation.json']:
            with self.assertRaises(PermissionError):
                check_access('open', (path, 'r', 0), set())
        with self.assertRaises(PermissionError):
            check_access('socket.connect', (), set())
        seen = set()
        check_access('open', ('/x/georecon/features.npz', 'r', 0), seen)
        self.assertEqual(len(seen), 1)


class OfflineMetrics(unittest.TestCase):
    def setUp(self):
        self.a = Camera(np.eye(3), np.eye(3), np.zeros(3), 800, 600)
        self.b = Camera(np.eye(3), np.eye(3), np.array([1., 0, 0]), 800, 600)

    def test_epipolar_proxy_and_coverage_not_3d_precision(self):
        p = np.array([[100., 100], [300, 300], [500, 500]])
        q = p + [[20, 0], [50, 0], [10, 20]]
        v, good = support_metrics(self.a, self.b, p, q, np.c_[np.arange(3), np.arange(3)], [3, 3])
        self.assertEqual(good.tolist(), [True, True, False])
        self.assertEqual(v['gt_consistent'], 2)
        self.assertAlmostEqual(v['grid_coverage'], 2/48)
        self.assertIsNone(v['true_3d_correspondence_precision'])

    def test_empty_ratio_and_undefined_gt(self):
        empty = np.empty((0, 2))
        v, _ = support_metrics(self.a, self.b, empty, empty, empty.astype(int), [0, 0])
        self.assertIsNone(v['gt_epipolar_compatibility'])
        self.assertEqual(v['grid_coverage'], 0)
        v, _ = support_metrics(self.a, self.a, empty, empty, empty.astype(int), [0, 0])
        self.assertEqual(v['status'], 'unavailable')

    def test_oriented_translation_and_rejected_pose(self):
        pose = pose_metrics(self.a, self.b, np.eye(3), np.array([-1., 0, 0]), True)
        self.assertEqual(pose['translation_deg'], 180)
        self.assertTrue(pose['false_accept'])
        self.assertEqual(pose['auc']['20'], 0)
        pose = pose_metrics(self.a, self.b, np.eye(3), np.ones(3), False)
        self.assertIsNone(pose['joint_deg'])
        self.assertEqual(pose['auc']['10'], 0)

    def fixture(self):
        p = load_protocol(ROOT)
        cases, seeds = p['input']['cases'], p['input']['geometry_seeds']
        rows = []
        for c in cases:
            for arm in 'ABC':
                for seed in seeds:
                    correct = not c['difficulty_group'] or arm == 'C'
                    rows.append(dict(case_id=c['case_id'], scan=c['scan'], arm=arm, seed=seed,
                        pose=dict(accepted=correct, correct=correct, false_accept=False, GT_pose_defined=True,
                                  auc={str(k): float(correct) for k in [5, 10, 20]}),
                        support=dict(status='evaluated', gt_consistent=30 if arm == 'C' else 20,
                                     grid_coverage=.5), postfeature_seconds=.5 if arm == 'C' else .1))
        return rows, cases, seeds

    def decide(self, rows, cases, seeds, **kwargs):
        return decision(rows, cases, seeds, summarize(rows, cases, seeds),
                        kwargs.get('completed', True), kwargs.get('resources', []), kwargs.get('valid', True))

    def test_planned_denominators_reused_conditions_and_no_best_seed(self):
        rows, cases, seeds = self.fixture()
        target = next(r for r in rows if r['case_id'] == cases[0]['case_id'] and r['arm'] == 'C' and r['seed'] == 43)
        target['pose']['auc'] = {str(k): 0 for k in [5, 10, 20]}
        s = next(s for s in summarize(rows, cases, seeds) if s['scan'] == 1 and s['arm'] == 'C')
        self.assertEqual(s['conditions'][0]['pair_seed_count'], 30)
        self.assertEqual(s['conditions'][1]['pair_seed_count'], 9)
        self.assertAlmostEqual(s['macro_auc10'], (29/30+8/9+1+1)/4)
        self.assertEqual(len(rows), 414)

    def test_all_gates_then_new_false_accept_and_normal_regression(self):
        rows, cases, seeds = self.fixture()
        self.assertEqual(self.decide(rows, cases, seeds)['verdict'], 'GO_FEASIBILITY')
        rows[-1]['pose']['false_accept'] = True
        d = self.decide(rows, cases, seeds)
        self.assertEqual(d['verdict'], 'NO_GO_EVIDENCE')
        self.assertFalse(d['gates']['G2'])
        rows[-1]['pose']['false_accept'] = False
        next(r for r in rows if r['arm'] == 'C')['pose']['correct'] = False
        self.assertTrue(self.decide(rows, cases, seeds)['normal_regressions'])

    def test_per_object_rescue_gate_and_verdict_precedence(self):
        rows, cases, seeds = self.fixture()
        for r in rows:
            if r['scan'] == 6 and r['arm'] == 'C':
                r['pose']['correct'] = False
        self.assertFalse(self.decide(rows, cases, seeds)['gates']['G1'])
        self.assertEqual(self.decide(rows, cases, seeds, resources=['oom'])['verdict'], 'NO_GO_RESOURCE')
        self.assertEqual(self.decide(rows, cases, seeds, completed=False)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(self.decide(rows, cases, seeds, valid=False, resources=['oom'])['verdict'], 'BLOCKED')

    def test_frozen_backends_and_explicit_plan_size(self):
        p = load_protocol(ROOT)
        self.assertEqual(len(p['input']['cases']), 46)
        self.assertEqual(sum(c['difficulty_group'] for c in p['input']['cases']), 26)
        self.assertEqual(p['budget']['main_matcher_runs'], 138)




class DependencyGuardRegression(unittest.TestCase):
    def test_library_code_exception_does_not_allow_calibration_data(self):
        source = Path('/installed/kornia/geometry/calibration')
        check_access('open', (str(source/'pnp.py'), 'r', 0), set(), source)
        check_access('open', (str(source/'__pycache__/pnp.cpython-312.pyc'), 'r', 0), set(), source)
        for path in [source/'cameras.json', source/'reference.json', source/'nested/ground_truth.py',
                     source/'../../../../data/calibration/cameras.py', Path('/data/calibration/cameras.py')]:
            with self.assertRaises(PermissionError):
                check_access('open', (str(path), 'r', 0), set(), source)

    def test_official_rootsift_import_in_guarded_fresh_process(self):
        import importlib.util
        import os
        import subprocess
        import sys
        if importlib.util.find_spec('lightglue') is None or importlib.util.find_spec('torch') is None:
            self.skipTest('isolated LightGlue environment required; no GPU or weights used')
        program = '''
import importlib.metadata
from pathlib import Path
import sys
from georecon.matching.worker import check_access, official_root
import numpy as np
source = Path(importlib.metadata.distribution('kornia').locate_file('kornia/geometry/calibration')).resolve()
sys.addaudithook(lambda event, args: check_access(event, args, set(), source))
x = official_root(np.ones((2, 128), np.float32))
assert x.shape == (2, 128) and x.dtype == np.float32
np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
x = official_root(np.empty((0, 128), np.float32))
assert x.shape == (0, 128)
'''
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8'},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class IncompleteGateRegression(unittest.TestCase):
    def test_missing_execution_never_vacuously_passes_quality_gate(self):
        helper = OfflineMetrics()
        rows, cases, seeds = helper.fixture()
        value = helper.decide(rows, cases, seeds, completed=False)
        self.assertEqual(value['verdict'], 'INCONCLUSIVE')
        for key in ['G1', 'G2', 'G3', 'G4']:
            self.assertIsNone(value['gates'][key])
        self.assertFalse(value['gates']['G5'])


if __name__ == '__main__':
    unittest.main()
