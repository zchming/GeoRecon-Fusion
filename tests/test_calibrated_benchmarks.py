import unittest
import numpy as np
from PIL import Image
from georecon.data.calibrated_benchmarks import strecha_camera,middlebury_cameras,preprocess,rounded_rotation


class CalibratedBenchmarkTests(unittest.TestCase):
    def test_strecha_transposes_rotation_and_converts_center(self):
        k=np.array([[700,0,320],[0,710,240],[0,0,1.]])
        r=np.array([[0.,0,1],[0,1,0],[-1,0,0]]);center=np.array([2.,3,4])
        text=' '.join(map(str,np.r_[k.ravel(),[0,0,0],r.T.ravel(),center,[640,480]]))
        c,_=strecha_camera(text)
        np.testing.assert_allclose(c.R,r,atol=1e-12);np.testing.assert_allclose(c.center,center,atol=1e-12)
        x=center+r.T@np.array([.2,.3,5.]);p,z,valid=c.project(x[None]);self.assertTrue(valid[0]);np.testing.assert_allclose(p[0],[348,282.6])

    def test_middlebury_translation_is_not_a_center(self):
        k=np.array([[700,0,320],[0,700,240],[0,0,1.]])
        r=np.array([[0.,0,1],[0,1,0],[-1,0,0]]);t=np.array([2.,3,4])
        text='1\na.png '+' '.join(map(str,np.r_[k.ravel(),r.ravel(),t]))
        c,_=middlebury_cameras(text)['a.png'];np.testing.assert_allclose(c.t,t);np.testing.assert_allclose(c.center,-r.T@t)
        with self.assertRaises(ValueError):middlebury_cameras(text.replace('1\n','2\n'))

    def test_no_upscale_and_exact_pixel_center_transform(self):
        from georecon.geometry.camera import Camera
        k=np.array([[1200,0,600],[0,1200,400],[0,0,1.]])
        c=Camera(k,np.eye(3),np.zeros(3),1200,800);im=Image.new('RGB',(1200,800))
        resized,kp,a=preprocess(im,c)
        self.assertEqual(resized.size,(800,533));np.testing.assert_allclose(kp,a@k)
        x=np.array([.1,.2,1.]);p=k@x;p/=p[2];q=kp@x;q/=q[2];np.testing.assert_allclose(q,a@p)
        small=Camera(k,np.eye(3),np.zeros(3),640,480)
        result,_,a=preprocess(Image.new('RGB',(640,480)),small);self.assertEqual(result.size,(640,480));np.testing.assert_array_equal(a,np.eye(3))

    def test_invalid_and_reflected_references_reject(self):
        for r in [np.diag([-1.,1,1]),np.zeros((3,3)),np.eye(3)*1.1]:
            with self.assertRaises(ValueError):rounded_rotation(r)
        for text in ['0 1','nan '*26]:
            with self.assertRaises(ValueError):strecha_camera(text)
        r=np.eye(3);r[0,1]=1e-7;fixed,amount=rounded_rotation(r)
        np.testing.assert_allclose(fixed.T@fixed,np.eye(3),atol=1e-12);self.assertLess(amount,1e-6)

if __name__=='__main__':unittest.main()
