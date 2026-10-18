"""Fixed observations, leakage, GT-free losses and isolated shadow worker."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from georecon.matching.heldout_geometry import prepare_observations, capped_sampson, preference, validation_metrics

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/experiments/stage7a_heldout_geometry_v1.json').read_text())
BACKEND = json.loads((ROOT/'configs/teaching/stage6_two_view.json').read_text())
OPENCV = importlib.util.find_spec('cv2') is not None


class HeldoutObservationTests(unittest.TestCase):
    def test_canonical_validation_is_actual_match_and_source_neutral(self):
        b = np.array([[1, 1], [2, 2]])
        c = np.array([[0, 0], [2, 2]])
        saved = {'raw_to_physical': {'0,0': [0, 0], '1,1': [0, 0], '2,2': [2, 2]},
                 'components': [{'partition': 'validation', 'edges': [[0, 0]]}, {'partition': 'fit', 'edges': [[2, 2]]}]}
        xy = [np.array([[10., 10], [10.5, 10], [100, 100]])]*2
        arrays, counts = prepare_observations(b, c, saved, xy)
        reversed_arrays, _ = prepare_observations(c, b, saved, xy)
        np.testing.assert_array_equal(arrays['validation_indices'], [[0, 0]])
        np.testing.assert_array_equal(arrays['validation_indices'], reversed_arrays['validation_indices'])
        np.testing.assert_array_equal(arrays['validation_p'], [[10, 10]])
        np.testing.assert_array_equal(arrays['fit_B_indices'], [[2, 2]])
        self.assertEqual(counts['B']['distinct_fit'], 1)
        self.assertTrue(arrays['validation_membership'].all())

    def test_fit_keeps_all_raw_aliases_instead_of_filtering(self):
        pairs = np.array([[0, 0], [1, 1], [2, 2]])
        saved = {'raw_to_physical': {'0,0': [0, 0], '1,1': [0, 0], '2,2': [2, 2]},
                 'components': [{'partition': 'fit', 'edges': [[0, 0]]}, {'partition': 'validation', 'edges': [[2, 2]]}]}
        arrays, counts = prepare_observations(pairs, pairs, saved, [np.ones((3, 2))]*2)
        self.assertEqual(counts['B']['raw_fit'], 2)
        self.assertEqual(counts['B']['distinct_fit'], 1)
        self.assertEqual(len(arrays['fit_B_indices']), 2)

    def test_leaking_grouped_endpoint_is_rejected(self):
        pairs = np.array([[0, 0], [1, 1]])
        saved = {'raw_to_physical': {'0,0': [0, 0], '1,1': [0, 1]},
                 'components': [{'partition': 'fit', 'edges': [[0, 0]]}, {'partition': 'validation', 'edges': [[0, 1]]}]}
        with self.assertRaises(ValueError):
            prepare_observations(pairs, pairs, saved, [np.ones((2, 2))]*2)

    def test_pose_preference_unknown_ties_and_no_threshold_fitting(self):
        self.assertEqual(preference(None, 0, 1e-12), 'unknown')
        self.assertEqual(preference(.2, .1, 1e-12), 'C_lower')
        self.assertEqual(preference(.1, .2, 1e-12), 'B_lower')
        self.assertEqual(preference(.1, .1+1e-13, 1e-12), 'tie')

    def test_sampson_translation_sign_is_not_identifiable(self):
        p = np.array([[10., 10], [20, 20]])
        q = p + [2, 0]
        for t in [np.array([1., 0, 0]), np.array([-1., 0, 0])]:
            errors, loss = capped_sampson(p, q, np.eye(3), np.eye(3), np.eye(3), t, 1.5)
            np.testing.assert_allclose(errors, 0)
            np.testing.assert_allclose(loss, 0)
        with self.assertRaises(ValueError):
            capped_sampson(p, q, np.eye(3), np.eye(3), np.eye(3), np.zeros(3), 1.5)

    def test_source_balancing_uses_one_common_row_with_explicit_weights(self):
        p = np.array([[10., 10], [20, 20]])
        q = p + [[2, 0], [2, 1.5*np.sqrt(2)]]
        arrays = {'validation_p': p, 'validation_q': q, 'validation_membership': np.array([[1, 1], [0, 1]], bool)}
        metrics, _ = validation_metrics(arrays, np.eye(3), np.eye(3), np.eye(3), np.array([1., 0, 0]), CFG, BACKEND)
        self.assertEqual(metrics['strata']['union']['count'], 2)
        self.assertAlmostEqual(metrics['primary_loss'], .25)
        self.assertAlmostEqual(metrics['strata']['union']['capped_loss'], .5)
        self.assertIsNone(metrics['strata']['B_only']['capped_loss'])
        self.assertIsNone(metrics['true_3D_match_precision'])

    def test_cheirality_exposes_sign_ambiguity_without_changing_pose(self):
        p = np.array([[10., 10], [20, 20]])
        arrays = {'validation_p': p, 'validation_q': p+[2, 0], 'validation_membership': np.ones((2, 2), bool)}
        plus, _ = validation_metrics(arrays, np.eye(3), np.eye(3), np.eye(3), np.array([1., 0, 0]), CFG, BACKEND)
        minus, _ = validation_metrics(arrays, np.eye(3), np.eye(3), np.eye(3), np.array([-1., 0, 0]), CFG, BACKEND)
        self.assertEqual(plus['primary_loss'], minus['primary_loss'])
        self.assertEqual(plus['dual_positive_fraction'], 1)
        self.assertEqual(minus['dual_positive_fraction'], 0)

    def test_worker_guard_forbids_truth_original_outputs_and_torch(self):
        spec = importlib.util.spec_from_file_location('shadow_runner', ROOT/'scripts/diagnose_stage7a_heldout_geometry.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        for name in ['/x/reference.json', '/x/evaluation.json', '/x/stage7a_v1_after_guard_fix/main/a/geometry_42.npz', '/x/model.pth']:
            with self.assertRaises(PermissionError): runner.guard('open', (name, 'r', 0))
        with self.assertRaises(PermissionError): runner.guard('import', ('torch.cuda',))
        with self.assertRaises(PermissionError): runner.guard('socket.connect', ())
        runner.guard('open', ('/new/inputs/test.npz', 'r', 0))

    @unittest.skipUnless(OPENCV, 'OpenCV unavailable; synthetic shadow worker not executed')
    def test_fresh_worker_fits_only_provided_synthetic_observations(self):
        rng = np.random.default_rng(42)
        xyz = np.c_[rng.uniform(-1, 1, (100, 2)), rng.uniform(4, 8, 100)]
        k = np.array([[700., 0, 400], [0, 700, 300], [0, 0, 1]])
        theta = .08; r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        t = np.array([-.8, 0, .02])
        def project(x):
            h = x@k.T
            return h[:, :2]/h[:, 2:3]
        p, q = project(xyz), project(xyz@r.T+t)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); path = out/'input.npz'
            np.savez(path, fit_B_p=p[:70], fit_B_q=q[:70], fit_C_p=p[:70], fit_C_q=q[:70],
                     validation_p=p[70:], validation_q=q[70:], validation_membership=np.ones((30, 2), bool))
            import hashlib
            cfg = dict(CFG, geometry_seeds=[42])
            bundle = {'config': cfg, 'backend': BACKEND, 'output': str(out), 'cases': [{'case_id': 'synthetic', 'input': str(path),
                'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'K': [k.tolist()]*2, 'eligibility_reasons': []}]}
            (out/'bundle.json').write_text(json.dumps(bundle))
            proc = subprocess.run([sys.executable, str(ROOT/'scripts/diagnose_stage7a_heldout_geometry.py'), 'worker', '--output', str(out)],
                env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            complete = json.loads((out/'worker_complete.json').read_text())
            self.assertEqual(complete['fit_calls'], 2)
            self.assertEqual(complete['validation_DLT_calls'], 2)
            for arm in ['B', 'C']:
                record = json.loads((out/'fits'/'synthetic'/arm/'record.json').read_text())
                self.assertEqual(record['rows'][0]['status'], 'success')
                self.assertLess(record['rows'][0]['validation']['primary_loss'], 1e-6)


if __name__ == '__main__':
    unittest.main()
