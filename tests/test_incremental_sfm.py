import unittest
import numpy as np
from georecon.teaching.incremental import pnp,project,collect_links
from georecon.teaching.two_view import triangulate_dlt
try:
    import cv2
except ImportError:cv2=None

@unittest.skipIf(cv2 is None,'OpenCV teaching extra not installed')
class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'pnp_min_inliers':12,'pnp_iterations':1000,'pnp_ransac_px':2.,'max_reprojection_px':2.,'seed':42}
        self.k=np.array([[700.,30,320],[0,710,240],[0,0,1.]])
        rng=np.random.default_rng(20);self.x=rng.uniform([-1,-1,4],[1,1,8],(100,3));self.r=cv2.Rodrigues(np.array([.02,.1,-.03]))[0];self.t=np.array([-.8,.02,.01]);self.uv=project(self.x,self.k,self.r,self.t)[0]
    def test_pose_world_to_camera_and_skew(self):
        out=pnp(self.x,self.uv,self.k,self.cfg);self.assertTrue(out['registered']);np.testing.assert_allclose(out['R'],self.r,atol=1e-6);np.testing.assert_allclose(out['t'],self.t,atol=1e-6)
    def test_noise_outliers(self):
        uv=self.uv.copy();uv[:20]=np.random.default_rng(10).uniform([0,0],[640,480],(20,2));uv[20:]+=.1
        out=pnp(self.x,uv,self.k,self.cfg);self.assertTrue(out['registered']);self.assertGreaterEqual(out['inliers'].sum(),75);self.assertFalse(out['inliers'][:20].any())
    def test_insufficient(self):self.assertEqual(pnp(self.x[:7],self.uv[:7],self.k,self.cfg)['reason'],'insufficient_2d3d')
    def test_collinear(self):
        x=np.c_[np.arange(20),np.zeros((20,2))];self.assertEqual(pnp(x,np.zeros((20,2)),self.k,self.cfg)['reason'],'collinear_3d')
    def test_track_conflicts(self):np.testing.assert_array_equal(collect_links([(1,2),(1,2),(2,3),(2,4),(3,5),(4,5)]),[[1,2]])
    def test_triangulation_world_transform(self):
        ro=self.r;to=self.t;rn=cv2.Rodrigues(np.array([-.03,.05,.02]))[0];tn=np.array([.4,.1,.03]);p=project(self.x,self.k,ro,to)[0];q=project(self.x,self.k,rn,tn)[0]
        tri=triangulate_dlt(p,q,self.k,self.k,rn@ro.T,tn-rn@ro.T@to);np.testing.assert_allclose((tri['xyz']-to)@ro,self.x,atol=1e-10)
if __name__=='__main__':unittest.main()
