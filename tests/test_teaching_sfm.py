import unittest
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json
from georecon.teaching.two_view import known_K,normalize,triangulate_dlt,reconstruct,features_and_matches
try:
    import cv2
    OPENCV=True
except ImportError:OPENCV=False
ROOT=Path(__file__).resolve().parents[1]


class TeachingMathTest(unittest.TestCase):
    def test_intrinsics_and_skew_normalization(self):
        k=np.array([[700.,2,320],[0,710,240],[0,0,1]])
        ray=np.array([[.2,.1,1],[-.3,.4,1]])
        pixel=ray@k.T;np.testing.assert_allclose(normalize(pixel[:,:2],k),ray[:,:2],atol=1e-12)
        for bad in [np.zeros((3,3)),np.eye(2),np.diag([-1,1,1]),np.array([[1,0,0],[.1,1,0],[0,0,1]])]:
            with self.assertRaises(ValueError):known_K(bad)

    def test_dlt_reconstructs_nontrivial_world_to_camera_pose(self):
        x=np.array([[-.4,-.2,4],[.4,.2,5],[.2,-.3,6]],float);k=np.array([[700.,0,320],[0,700,240],[0,0,1]])
        theta=.1;r=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]);t=np.array([-.8,0,.02])
        def project(c):q=c@k.T;return q[:,:2]/q[:,2:3]
        result=triangulate_dlt(project(x),project(x@r.T+t),k,k,r,t)
        np.testing.assert_allclose(result['xyz'],x,atol=1e-10);self.assertTrue(result['positive'].all());self.assertLess(result['reprojection_px'].max(),1e-8)

    def test_tiny_parallax_and_zero_baseline_are_not_good_points(self):
        x=np.array([[.1,.1,5],[-.2,.3,6]],float);k=np.diag([700.,700,1])
        p=(x@k.T);p=p[:,:2]/p[:,2:3];t=np.array([-.0001,0,0]);q=(x+t)@k.T;q=q[:,:2]/q[:,2:3]
        tri=triangulate_dlt(p,q,k,k,np.eye(3),t);self.assertTrue(np.all(tri['parallax_deg']<.5))
        tri=triangulate_dlt(p,p,k,k,np.eye(3),np.zeros(3));self.assertFalse(tri['positive'].any())

    def test_negative_depth_points_remain_explicit_rejections(self):
        k=np.eye(3);r=np.eye(3);t=np.array([-1.,0,0]);x=np.array([[1.,2,-5]])
        p=x[:,:2]/x[:,2:3];y=x+t;q=y[:,:2]/y[:,2:3]
        tri=triangulate_dlt(p,q,k,k,r,t);self.assertFalse(tri['positive'][0]);self.assertLess(tri['depths'][0,0],0)


@unittest.skipUnless(OPENCV,'OpenCV teaching dependency unavailable')
class TeachingCVTest(unittest.TestCase):
    def cfg(self):return read_json(ROOT/'configs/teaching/stage6_two_view.json')
    def scene(self,planar=False,translation=True):
        rng=np.random.default_rng(7);x=np.c_[rng.uniform(-1,1,160),rng.uniform(-.8,.8,160),rng.uniform(4,8,160)]
        if planar:x[:,2]=5
        k1=np.array([[700.,.2,320],[0,710,240],[0,0,1]]);k2=np.array([[680.,-.1,315],[0,705,245],[0,0,1]])
        theta=.08;r=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]);t=np.array([-.7,.02,.01]) if translation else np.zeros(3)
        a=x@k1.T;b=(x@r.T+t)@k2.T
        return a[:,:2]/a[:,2:3],b[:,:2]/b[:,2:3],k1,k2,r,t,x

    def test_calibrated_pose_correct_sign_and_arbitrary_scale(self):
        from georecon.evaluation.poses import rotation_error_deg,translation_direction_error_deg
        p,q,k1,k2,r,t,x=self.scene();out=reconstruct(p,q,k1,k2,self.cfg())
        self.assertEqual(out['status'],'success',out['reason'])
        self.assertLess(rotation_error_deg(out['R'],r),.01);self.assertLess(translation_direction_error_deg(out['t'],t),.01)
        self.assertAlmostEqual(np.linalg.norm(out['t']),1);self.assertEqual(len(out['stats']['hypotheses']),4)
        ids=out['arrays']['kept_match_indices'];np.testing.assert_allclose(out['arrays']['accepted_xyz'],x[ids]/np.linalg.norm(t),atol=1e-5)

    def test_nonplanar_correspondence_outliers_are_filtered(self):
        p,q,k1,k2,r,t,x=self.scene();rng=np.random.default_rng(9);q[:30]=rng.uniform([0,0],[640,480],(30,2))
        out=reconstruct(p,q,k1,k2,self.cfg());self.assertEqual(out['status'],'success',out['reason'])
        self.assertLessEqual(np.sum(out['arrays']['essential_mask'][:30]),5)
        self.assertLessEqual(out['stats']['accepted_max_reprojection_px'],self.cfg()['max_reprojection_px'])

    def test_planar_and_pure_rotation_cases_reject_ambiguous_model(self):
        for planar,translation in [(True,True),(False,False)]:
            p,q,k1,k2,*_=self.scene(planar,translation);out=reconstruct(p,q,k1,k2,self.cfg())
            self.assertEqual(out['status'],'no_reconstruction');self.assertEqual(out['reason'],'homography_dominated_ambiguous_geometry')

    def test_few_matches_and_identity_are_failures_without_pose(self):
        p,q,k1,k2,*_=self.scene();out=reconstruct(p[:7],q[:7],k1,k2,self.cfg());self.assertEqual(out['reason'],'insufficient_matches');self.assertIsNone(out['R'])
        out=reconstruct(p,p,k1,k1,self.cfg());self.assertEqual(out['reason'],'identical_correspondences_no_baseline');self.assertIsNone(out['t'])

    def test_image_matching_recovers_pixel_shift_and_blank_has_no_descriptors(self):
        rng=np.random.default_rng(8);image=rng.integers(0,256,(160,240,3),dtype=np.uint8);moved=np.zeros_like(image);moved[:,12:]=image[:,:-12]
        arrays,stats=features_and_matches([image,moved],self.cfg());self.assertGreater(stats['mutual_matches'],20)
        residual=arrays['pixels_second']-arrays['pixels_first']-[12,0];self.assertLess(float(np.median(np.linalg.norm(residual,axis=1))),.2)
        arrays,stats=features_and_matches([np.zeros_like(image),np.zeros_like(image)],self.cfg());self.assertEqual(stats['mutual_matches'],0)

    def test_runtime_truth_guard_and_intrinsics_permission(self):
        import subprocess,sys
        script="""import sys
from georecon.teaching.run import deny_pose_truth
sys.addaudithook(deny_pose_truth)
for filename in ['data/stage6_truth/test.json','reference.json','calibration/a.txt']:
 try:open(filename)
 except PermissionError:pass
 else:raise AssertionError('pose truth open allowed')
print('blocked')
"""
        proc=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True);self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(proc.stdout.strip(),'blocked')

if __name__=='__main__':unittest.main()
