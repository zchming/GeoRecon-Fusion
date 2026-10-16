import unittest
import numpy as np
from georecon.ablation.controls import keep_unknown,without_raw_support,remove_evidence_gate


class Stage5AblationTest(unittest.TestCase):
    def test_unknown_control_changes_only_finite_unobserved(self):
        keep=np.array([True,False,False,False]);finite=np.array([True,True,True,False]);observed=np.array([2,0,1,0])
        np.testing.assert_array_equal(keep_unknown(keep,finite,observed),[True,True,False,False])
        self.assertFalse(keep[1])
        with self.assertRaises(ValueError):keep_unknown(keep,finite[:2],observed)

    def test_neutral_support_formula_and_no_input_mutation(self):
        p=np.array([[1.,0,1],[0,0,1]],dtype=np.float32)
        a={'source_point_indices':np.array([0]),'anchor_xyz':np.array([[2.,0,1]]),'classical_weight':np.array([1.]),'raw_weight':np.array([2.])}
        result,beta=without_raw_support(p,a,.5)
        self.assertAlmostEqual(beta[0],1/3);self.assertAlmostEqual(float(result[0,0]),4/3,places=6)
        self.assertEqual(a['raw_weight'][0],2);np.testing.assert_array_equal(result[1],p[1])

    def test_evidence_control_keeps_hash_and_status_metadata(self):
        d={'methods':{'raw':{'status':'unknown','prediction_sha256':'abc','baseline_status':'success','summary':{'evidence_pairs':0,'other':8}}}}
        changed=remove_evidence_gate(d)
        self.assertEqual(changed['methods']['raw']['prediction_sha256'],'abc');self.assertEqual(changed['methods']['raw']['baseline_status'],'success')
        self.assertEqual(d['methods']['raw']['status'],'unknown');self.assertEqual(changed['methods']['raw']['summary']['other'],8)

    def fixture(self):
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('joint_fixture',Path(__file__).parent/'test_joint_optimization.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        if not m.AVAILABLE:self.skipTest('SciPy unavailable')
        return m.JointOptimizationTest().fixture()

    def test_fixed_pose_solver_changes_points_only(self):
        from georecon.ablation.controls import fixed_pose_solve
        p,c=self.fixture();out,record=fixed_pose_solve(p,c,1)
        self.assertTrue(record['accepted'],record);self.assertLess(record['final_objective'],record['initial_objective'])
        np.testing.assert_array_equal(out['R'],p['R0']);np.testing.assert_array_equal(out['t'],p['t0'])
        self.assertTrue(np.all(out['pose_delta']==0));self.assertGreater(np.linalg.norm(out['points']-p['x0']),1e-8)

    def test_fixed_pose_same_point_bounds(self):
        from georecon.ablation.controls import fixed_pose_solve
        p,c=self.fixture();out,record=fixed_pose_solve(p,c,1)
        self.assertTrue(record['accepted']);self.assertLessEqual(np.max(np.abs(out['parameters'])),c['solver']['point_delta_component_bound_over_depth']+1e-12)

    def test_fixed_pose_budget_preserves_rejected_attempt(self):
        from georecon.ablation.controls import fixed_pose_solve
        p,c=self.fixture();c['solver']['max_nfev']=1;out,record=fixed_pose_solve(p,c,1)
        self.assertFalse(record['accepted']);self.assertEqual(record['status'],0);self.assertIsNotNone(out)

if __name__=='__main__':unittest.main()
