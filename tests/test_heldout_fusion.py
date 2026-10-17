import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('heldout_fusion_math',Path(__file__).resolve().parents[1]/'scripts/heldout_fusion_math.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class HeldoutFusionTests(unittest.TestCase):
    def test_groups_never_split_duplicate_witness(self):
        a,b,ok=m.split_groups(np.array(['a','b','b','c']),'a','b');self.assertTrue(ok);np.testing.assert_array_equal(a,[0,3]);np.testing.assert_array_equal(b,[1,2])
    def test_one_remaining_view_not_feasible(self):self.assertFalse(m.split_groups(np.array(['a','b','b']),'a','b')[2])
    def test_source_cannot_be_witness(self):
        with self.assertRaises(ValueError):m.split_groups(np.array(['a','b']),'a','a')
    def cameras(self):return [{'K':np.array([[500.,.2,320],[0,510,240],[0,0,1]]),'R':np.eye(3),'t':np.array([-v,0,0.])} for v in [0,1,2]]
    def test_dlt_uses_only_passed_observations(self):
        cams=self.cameras();x=np.array([.3,.2,5]);uv=np.array([m.project(x,c)[0] for c in cams]);uv[2]+=[100,50];got=m.triangulate(cams[:2],uv[:2]);np.testing.assert_allclose(got,x,atol=1e-11)
    def test_coincident_cameras_rejected(self):
        c=self.cameras()[0];self.assertIsNone(m.triangulate([c,c],np.array([[320,240],[320,240]])))
    def test_score_sign_and_group_aggregation(self):
        c=self.cameras()[0];truth=np.array([0.,0,5]);raw=np.array([.1,0,5]);uv=np.tile(m.project(truth,c)[0],(2,1));s,reason=m.witness_score(raw,truth,uv,c,640,480);self.assertIsNone(reason);self.assertAlmostEqual(s,10.)
    def test_unobservable_is_not_bad_label(self):
        c=self.cameras()[0];s,reason=m.witness_score(np.array([0.,0,5]),np.array([0,0,-1]),np.array([[320,240]]),c,640,480);self.assertIsNone(s);self.assertIsNotNone(reason)
if __name__=='__main__':unittest.main()
