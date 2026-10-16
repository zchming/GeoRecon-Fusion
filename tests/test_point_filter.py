import unittest
import numpy as np
from georecon.filtering.points import evidence,classify


class PointFilterTest(unittest.TestCase):
    def scene(self):
        n,h,w=3,4,5
        ys,xs=np.mgrid[:h,:w]
        points=np.broadcast_to(np.stack([xs,ys,np.ones_like(xs)],-1),(n,h,w,3)).astype(float).copy()
        depth=np.ones((n,h,w))
        ext=np.broadcast_to(np.eye(4)[:3],(n,3,4)).copy()
        k=np.broadcast_to(np.eye(3),(n,3,3)).copy()
        return points,depth,ext,k

    def test_consistent_points_and_native_boundary(self):
        p,d,e,k=self.scene();ev=evidence(p,d,e,k,[.01],batch=7)
        keep,unknown=classify(ev,0,2)
        expected=np.zeros(d.shape,bool);expected[:,:-1,:-1]=True
        np.testing.assert_array_equal(keep.reshape(d.shape),expected)
        np.testing.assert_array_equal(unknown,~expected.ravel())

    def test_no_self_view_support(self):
        p,d,e,k=self.scene();d[1:]=10
        ev=evidence(p,d,e,k,[.01]);keep,_=classify(ev,0,1)
        self.assertFalse(keep.reshape(d.shape)[0].any())
        self.assertTrue(keep.reshape(d.shape)[1,:-1,:-1].all())

    def test_point_head_not_replaced_by_depth(self):
        p,d,e,k=self.scene();p[0,:,:,2]=2
        ev=evidence(p,d,e,k,[.01]);keep,_=classify(ev,0,1)
        self.assertFalse(keep.reshape(d.shape)[0].any())
        self.assertTrue(keep.reshape(d.shape)[1,:-1,:-1].all())
        self.assertEqual(p[0,0,0,2],2)

    def test_unknown_invalid_and_empty(self):
        p,d,e,k=self.scene();p[:]=np.nan;p[0,0,0]=[1000,1000,1]
        ev=evidence(p,d,e,k,[.01]);keep,unknown=classify(ev,0,1)
        self.assertFalse(keep.any());self.assertEqual(unknown.sum(),1)
        self.assertEqual((~ev['finite']).sum(),p.size//3-1)

    def test_occlusion_disagreement_is_not_support(self):
        p,d,e,k=self.scene();d[:]=.5
        ev=evidence(p,d,e,k,[.01]);keep,unknown=classify(ev,0,1)
        self.assertFalse(keep.any());self.assertGreater((ev['observed']>0).sum(),0)

    def test_batch_invariance_and_no_gt_parameters(self):
        p,d,e,k=self.scene()
        a=evidence(p,d,e,k,[.01,.05],1);b=evidence(p,d,e,k,[.01,.05],1000)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        with self.assertRaises(ValueError):evidence(p[:1],d,e,k,[.01])



class PointEvaluationTest(unittest.TestCase):
    def test_subset_cannot_increase_fixed_sample_recall(self):
        import importlib.util
        from pathlib import Path
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self.skipTest('SciPy evaluation dependency unavailable in this environment')
        spec=importlib.util.spec_from_file_location('stage5_points',Path(__file__).resolve().parents[1]/'scripts/stage5_points.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ev=module.Evaluator.__new__(module.Evaluator)
        ev.xyz=np.array([[0.,0,0],[10,0,0],[100,0,0]])
        ev.ids=np.arange(3);ev.in_mask=np.ones(3,bool)
        ev.recall_gt=np.array([[0.,0,0],[10,0,0]])
        ev.tree=cKDTree(ev.recall_gt)
        before=ev.measure(np.ones(3,bool));after=ev.measure(np.array([True,False,False]))
        self.assertEqual(before['thresholds_mm']['2']['precision'],2/3)
        self.assertEqual(after['thresholds_mm']['2']['precision'],1)
        self.assertEqual(before['thresholds_mm']['2']['recall'],1)
        self.assertEqual(after['thresholds_mm']['2']['recall'],.5)
        empty=ev.measure(np.zeros(3,bool))
        self.assertEqual(empty['thresholds_mm']['2']['f1'],0)
        self.assertIsNone(empty['mean_completeness_mm'])

    def test_runtime_guard_blocks_gt(self):
        import subprocess,sys
        script="""import sys
from georecon.reliability.analyze import deny_ground_truth
sys.addaudithook(deny_ground_truth)
try:
 open('reference.json')
except PermissionError:
 print('blocked')
else:
 raise AssertionError('GT read was allowed')
"""
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'blocked')


if __name__=='__main__':unittest.main()
