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


if __name__=='__main__':unittest.main()
