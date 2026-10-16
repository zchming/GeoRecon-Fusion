import importlib.util
from pathlib import Path
import unittest
from georecon.data.protocol import read_json
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage3',ROOT/'scripts/stage3_benchmark.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class BenchmarkTests(unittest.TestCase):
    def test_auc_missing_and_threshold_boundary(self):
        self.assertAlmostEqual(module.auc([0,5,10,None],10),.375)
        self.assertEqual(module.auc([None,None],5),0)
        self.assertEqual(module.auc([5,6],5),0)
        self.assertEqual(module.auc([0,0],5),1)
        self.assertIsNone(module.auc([],5))

    def test_split_and_condition_lock(self):
        plan=read_json(ROOT/'configs/benchmarks/dtu_pose_pilot_v1.json')
        self.assertTrue(set(plan['splits']['dev']).isdisjoint(plan['splits']['test']))
        conditions={c['id']:c for c in plan['conditions']}
        self.assertLess(len(conditions['reduced3']['views']),len(conditions['normal5']['views']))
        self.assertEqual(conditions['blur5']['views'],conditions['normal5']['views'])
        self.assertEqual(conditions['blur5']['blur_radius_original_px'],4)
        self.assertEqual(plan['budget']['max_runs'],sum(map(len,plan['splits'].values()))*len(conditions)*len(plan['methods']))
        for config in plan['methods'].values():
            c=read_json(ROOT/config);self.assertEqual(c['timeout_seconds'],180)
            if 'gpu_allocator_limit_gib' in c:self.assertLessEqual(c['gpu_allocator_limit_gib'],10.5)
