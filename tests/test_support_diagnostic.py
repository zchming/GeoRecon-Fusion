"""Identity loss and topology tests; no model, data, geometry or network."""
import unittest
from georecon.evaluation.support_diagnostic import (
    direct_candidates, link_accounting, opportunity_gates, potential_anchors,
    seed_observations, topology,
)


class SupportDiagnosticTests(unittest.TestCase):
    def test_duplicate_votes_and_overlapping_conflicts_are_not_double_counted(self):
        counts, kept = link_accounting([(0, 0), (0, 0), (1, 1), (1, 2), (2, 2), (3, 3)])
        self.assertEqual(kept, {(0, 0), (3, 3)})
        self.assertEqual(counts['duplicate_votes'], 1)
        self.assertEqual(counts['conflict_loss'], 3)
        self.assertEqual(counts['both_conflict_edges'], 1)
        self.assertEqual(sum(counts[k] for k in ('duplicate_votes', 'conflict_loss', 'unique_links')), 6)

    def test_seed_prefix_keeps_identity_and_rejects_missing_or_duplicate_observations(self):
        rows = [(0, 4, 0), (1, 6, 0), (0, 5, 1), (1, 7, 1), (2, 8, 0), (0, 9, 2)]
        obs = seed_observations(rows, [0, 1], 2)
        self.assertEqual(obs, {(0, 4): 0, (1, 6): 0, (0, 5): 1, (1, 7): 1})
        for invalid in (rows[1:], rows+[(0, 4, 0)], rows+[(0, 44, 0)]):
            with self.assertRaises(ValueError):
                seed_observations(invalid, [0, 1], 2)

    def test_reverse_pair_orientation_preserves_candidate_identity(self):
        obs = {(1, 2): 0, (2, 4): 0}
        matches = {(0, 1): [(7, 2)], (0, 2): [(7, 4)]}
        counts, kept = link_accounting(direct_candidates(matches, obs, [1, 2], 0))
        self.assertEqual(counts['candidate_links'], 2)
        self.assertEqual(kept, {(7, 0)})

    def test_closed_missing_and_ambiguous_triangles_stay_separate(self):
        for matches, expected in (
            ({(0, 1): [(0, 0)], (1, 2): [(0, 0)], (0, 2): [(0, 0)]}, (1, 0, 0)),
            ({(0, 1): [(0, 0)], (1, 2): [(0, 0)], (0, 2): []}, (0, 1, 0)),
            ({(0, 1): [(0, 0), (0, 1)], (1, 2): [(0, 0)], (0, 2): [(0, 0)]}, (0, 0, 2)),
        ):
            _, triples, _, _, _ = topology(matches)
            self.assertEqual(tuple(triples[0][k] for k in ('closed', 'missing_edge', 'ambiguous')), expected)
        self.assertEqual(topology({(0, 1): [(0, 0)]})[1], [])
        empty = topology({(0, 1): [], (0, 2): [], (1, 2): []})[1]
        self.assertEqual(empty[0]['candidate_triples'], 0)
        self.assertIsNone(empty[0]['closure_fraction'])

    def test_raw_indirect_anchor_is_not_a_registered_point_and_conflicts_are_excluded(self):
        m = {(0, 1): [(0, 0)], (1, 2): [(0, 0)], (2, 3): [(0, 0)]}
        _, _, groups, lookup, conflicts = topology(m)
        obs = {(0, 0): 0, (1, 0): 0}
        p = potential_anchors(groups, lookup, conflicts, obs, 3, set())
        self.assertEqual(p['topological_anchors'], 1)
        self.assertEqual(p['indirect_missing_pairs'], 1)
        m[0, 3] = [(1, 0)]
        _, _, groups, lookup, conflicts = topology(m)
        p = potential_anchors(groups, lookup, conflicts, obs, 3, set())
        self.assertEqual(p['topological_anchors'], 0)
        self.assertEqual(p['excluded_conflict_components'], 1)

    def test_priority_requires_two_scenes_two_conditions_all_five_seeds_per_ranking(self):
        records, attempts = [], []
        for scene in ('a', 'b'):
            for condition in ('normal5', 'blur5'):
                for seed in range(42, 47):
                    name = f'{scene}/{condition}/{seed}'
                    records.append(dict(record=name, family='tum_rgbd', arm='C', method='legacy',
                                        ranking='quality', scene=scene, condition=condition, seed=seed))
                    attempts.append(dict(record=name, evaluability='exact', conflict_blocks_threshold=True))
        self.assertTrue(opportunity_gates(records, attempts)['quality']['conflict_blocks_threshold']['passed'])
        attempts[0]['evaluability'] = 'unavailable'
        self.assertFalse(opportunity_gates(records, attempts)['quality']['conflict_blocks_threshold']['passed'])
        self.assertFalse(opportunity_gates(records, attempts)['inliers']['conflict_blocks_threshold']['passed'])

    def test_used_inputs_exclude_pool_members_absent_from_frozen_case_hashes(self):
        import importlib.util
        from pathlib import Path
        script = Path(__file__).resolve().parents[1]/'scripts/diagnose_stage7d_support.py'
        spec = importlib.util.spec_from_file_location('stage7d_read_only_script', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bundle = dict(cases=[{'keys':['a','b']}], feature_paths={'a':'fa','b':'fb','unused':'fu'},
                      pair_paths={'B:a:b':'mb','C:a:b':'mc','B:a:unused':'mu'})
        self.assertEqual(module.used_cache_paths(bundle), {'fa','fb','mb','mc'})


if __name__ == '__main__':
    unittest.main()
