import importlib.util
from pathlib import Path
import unittest
import numpy as np
try:
    spec=importlib.util.spec_from_file_location('fusion_diagnostics_math',Path(__file__).resolve().parents[1]/'scripts/fusion_diagnostics_math.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
except ImportError:m=None
@unittest.skipIf(m is None,'SciPy diagnostics dependency unavailable')
class FusionDiagnosticsTests(unittest.TestCase):
    def test_auc_ties_and_missing(self):
        self.assertEqual(m.auc([1,1,np.nan],[True,False,True])['auc'],.5);self.assertEqual(m.auc([1,2],[False,True])['auc'],1);self.assertIsNone(m.auc([1,2],[True,True])['auc'])
    def test_oracle_includes_raw_and_does_not_change_zero_ties(self):
        d=np.array([[1.,2,3],[2,1,3],[.5,4,5]]);a,b=m.oracle(d);np.testing.assert_array_equal(a,[2,1,0]);self.assertTrue((b<=d[0]).all())
    def test_bilinear_and_border_are_explicit(self):
        g=np.zeros((1,3,3,3));g[0,:,:,0]=np.arange(3);g[0,:,:,1]=np.arange(3)[:,None];x,v=m.bilinear(g,np.array([0,0]),np.array([[.5,1.5],[2,1.]]));np.testing.assert_allclose(x[0],[.5,1.5,0]);self.assertFalse(v[1]);self.assertTrue(np.isnan(x[1]).all())
    def test_frozen_blend_float32(self):
        x=np.array([[1,2,3]],np.float32);a=np.array([[2,4,5]],float);b=np.array([.2]);out=m.candidates(x,a,b,[0,.5,1]);expected=x.copy();expected+=b[:,None]*(a-x);np.testing.assert_array_equal(out[-1],expected);np.testing.assert_array_equal(out[0],x)
    def test_ties_do_not_artificially_select_favorable_examples(self):
        r=m.signal_report(np.ones(4),np.array([-2,-1,1,2]),np.arange(4));self.assertEqual(r['top_fraction_profiles'][0]['actual_selected'],4);self.assertEqual(r['benefit_AUROC']['auc'],.5)
    def test_harm_and_threshold_denominators(self):
        r=m.summary([1,3,np.nan],[3,1,0]);self.assertEqual(r['count'],2);self.assertEqual(r['good_to_bad_2mm'],1);self.assertEqual(r['bad_to_good_2mm'],1)
if __name__=='__main__':unittest.main()
