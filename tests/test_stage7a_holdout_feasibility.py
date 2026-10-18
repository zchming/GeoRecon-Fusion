"""Synthetic topology checks: no cached data, model, GT or geometry solving."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from georecon.matching.holdout_feasibility import (group_keypoints, correspondence_set,
    source_counts, physical_edges, conflicts, component_partition, spatial, analyze_pair)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/experiments/stage7a_holdout_feasibility_v1.json').read_text())


def points(n):
    return np.array([[5+3*(i % 80), 5+3*(i//80)] for i in range(n)], dtype=float).reshape(-1, 2)


class CachedTopologyTest(unittest.TestCase):
    def test_exact_and_transitive_near_duplicate_groups(self):
        xy = np.array([[0., 0], [0, 0], [.8, 0], [1.6, 0], [3, 0]])
        self.assertEqual(group_keypoints(xy, 0).tolist(), [0, 0, 2, 3, 4])
        self.assertEqual(group_keypoints(xy, 1).tolist(), [0, 0, 0, 0, 4])
        self.assertGreater(np.linalg.norm(xy[0]-xy[3]), 1)  # conservative transitive chain

    def test_nonfinite_and_bad_index_contracts(self):
        with self.assertRaises(ValueError):
            group_keypoints([[np.nan, 0]], 1)
        for a in [np.array([[0., 1.]]), np.array([[0, 3]]), np.array([[-1, 0]]), np.array([[0, 0], [0, 1]])]:
            with self.assertRaises(ValueError):
                correspondence_set(a, [3, 3])

    def test_physical_overlap_hidden_by_orientation_alias_indices(self):
        groups = [group_keypoints([[10, 10], [10, 10]], 1)]*2
        b, c = {(0, 0)}, {(1, 1)}
        self.assertEqual(source_counts(b, c)['common'], 0)
        self.assertEqual(source_counts(physical_edges(b, groups), physical_edges(c, groups))['common'], 1)

    def test_conflicting_chain_never_leaks_endpoints(self):
        edges = {(0, 0), (0, 1), (1, 1), (2, 2), (3, 3)}
        self.assertEqual(conflicts(edges), [1, 1])
        fit, val, components = component_partition(edges, 'synthetic-conflicts', CFG)
        self.assertEqual(len(components), 3)
        self.assertTrue(any(set(map(tuple, r['edges'])) == {(0, 0), (0, 1), (1, 1)} for r in components))
        for side in [0, 1]:
            self.assertFalse({e[side] for e in fit} & {e[side] for e in val})

    def test_partition_invariant_to_order_and_swapping_matcher_labels(self):
        xy = [points(100), points(100)]
        b = np.array([[i, i] for i in range(80)])
        c = np.array([[i, i] for i in range(20, 100)])
        r, mapping = analyze_pair('synthetic-label-neutral', xy, b, c, [[800, 600]]*2, CFG)
        other, mapping2 = analyze_pair('synthetic-label-neutral', xy, c[::-1], b[::-1], [[800, 600]]*2, CFG)
        self.assertEqual(mapping['components'], mapping2['components'])
        self.assertEqual(r['fit']['B'], other['fit']['C'])
        self.assertEqual(r['validation']['B'], other['validation']['C'])
        self.assertFalse(r['endpoint_leakage'])
        self.assertFalse(r['validation_power_established'])

    def test_empty_and_insufficient_sources_remain_unevaluable(self):
        empty = np.empty((0, 2), dtype=np.int64)
        r, _ = analyze_pair('empty', [points(0)]*2, empty, empty, [[800, 600]]*2, CFG)
        self.assertEqual(r['structural_status'], 'not_evaluable_under_this_partition')
        self.assertIn('B_validation_empty', r['reasons'])
        self.assertFalse(r['representative_grid_disjoint'])
        self.assertIsNone(r['spatial_block_independence'])
        self.assertIsNone(r['spatial']['validation_union'][0]['largest_cell_fraction'])
        for value in r['validation_count_levels'].values():
            self.assertFalse(value)

    def test_count_levels_are_reported_not_selected(self):
        xy = [points(160)]*2
        ids = np.c_[np.arange(160), np.arange(160)]
        r, _ = analyze_pair('fixed-levels', xy, ids, ids, [[800, 600]]*2, CFG)
        for n in [1, 5, 10, 20]:
            self.assertEqual(r['validation_count_levels'][str(n)], not r['reasons'] and r['validation_min_per_source'] >= n)
        self.assertEqual(r['equal_exclusive_quota_capacity'], 0)
        self.assertFalse(r['validation_power_established'])

    def test_distinct_edges_do_not_imply_spatial_independence(self):
        xy = [points(160)]*2
        ids = np.c_[np.arange(160), np.arange(160)]
        r, _ = analyze_pair('same-cells', xy, ids, ids, [[800, 600]]*2, CFG)
        self.assertFalse(r['endpoint_leakage'])
        self.assertFalse(r['spatial_block_independence'])
        self.assertTrue(any(r['fit_validation_shared_cells_per_image']))

    def test_spatial_distribution_empty_and_concentrated(self):
        xy = [np.array([[1., 1], [2, 2], [200, 200]])]*2
        stats = spatial({(0, 0), (1, 1), (2, 2)}, xy, [[800, 600]]*2, [8, 6])
        self.assertEqual(stats[0]['occupied_cells'], 2)
        self.assertAlmostEqual(stats[0]['largest_cell_fraction'], 2/3)
        with self.assertRaises(ValueError):
            spatial({(0, 0)}, [np.array([[-1., 0]])]*2, [[800, 600]]*2, [8, 6])

    def test_audit_guard_forbids_solver_model_network_and_truth(self):
        spec = importlib.util.spec_from_file_location('feasibility_runner', ROOT/'scripts/check_stage7a_holdout_feasibility.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ['cv2', 'torch.cuda', 'scipy.optimize', 'georecon.teaching.two_view']:
            with self.assertRaises(PermissionError):
                module.guard('import', (name,))
        for name in ['/data/reference.json', '/data/geometry_42.npz', '/weights/sift.pth']:
            with self.assertRaises(PermissionError):
                module.guard('open', (name, 'r', 0))
        with self.assertRaises(PermissionError):
            module.guard('socket.connect', ())
        module.guard('open', ('/cache/matches.npz', 'r', 0))


if __name__ == '__main__':
    unittest.main()
