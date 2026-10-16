import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose
from georecon.geometry.camera import Camera
from georecon.reliability.geometry import fundamental,sampson_px,triangulation_checks,components
from georecon.reliability.depth import bilinear,cross_view
from georecon.reliability.analyze import analyze_prediction,deny_ground_truth
from georecon.data.protocol import read_json,result_template
ROOT=Path(__file__).resolve().parents[1]

class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(ROOT/'configs/reliability/stage4_v1.json')
        K=np.array([[300.,0,320],[0,300,240],[0,0,1.]])
        self.a=Camera(K,np.eye(3),np.zeros(3),640,480)
        self.b=Camera(K,np.eye(3),np.array([-1.,0,0]),640,480)
        rng=np.random.default_rng(42);self.xyz=np.c_[rng.uniform(-1,1,(64,2)),rng.uniform(4,8,64)]
        self.p=self.a.project(self.xyz)[0];self.q=self.b.project(self.xyz)[0]

    def test_observed_pixel_error_not_self_projection(self):
        F=fundamental(self.a,self.b)
        assert_allclose(sampson_px(F,self.p,self.q),0,atol=1e-10)
        corrupted=self.q.copy();corrupted[:,1]+=20
        self.assertGreater(sampson_px(F,self.p,corrupted).min(),10)
        assert_allclose(sampson_px(F,self.p,corrupted),sampson_px(100*F,self.p,corrupted))

    def test_epipolar_sign_ambiguity_requires_cheirality(self):
        reverse=Camera(self.b.K,self.b.R,-self.b.t,640,480)
        assert_allclose(sampson_px(fundamental(self.a,reverse),self.p,self.q),0,atol=1e-10)
        self.assertEqual(triangulation_checks(self.a,self.b,self.p,self.q)['positive_depth_ratio'],1)
        self.assertEqual(triangulation_checks(self.a,reverse,self.p,self.q)['positive_depth_ratio'],0)

    def test_world_scale_and_gauge_invariance(self):
        theta=.3;G=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        offset=np.array([10.,-2,3]);scale=7
        changed=[]
        for c in [self.a,self.b]:
            R=c.R@G.T;changed.append(Camera(c.K,R,scale*c.t-R@offset,c.width,c.height))
        assert_allclose(sampson_px(fundamental(*changed),self.p,self.q),0,atol=1e-10)
        self.assertEqual(triangulation_checks(*changed,self.p,self.q)['positive_depth_ratio'],1)

    def test_degenerate_geometry_unknown(self):
        self.assertIsNone(fundamental(self.a,self.a))
        self.assertIsNone(triangulation_checks(self.a,self.a,self.p,self.p)['positive_depth_ratio'])

    def test_depth_checks_other_view_and_boundaries(self):
        depth=np.array([[1.,2,3],[2,3,4],[3,4,5]])
        values,valid=bilinear(depth,np.array([[.5,.5],[2,1],[np.nan,0],[-1,1]]))
        self.assertEqual(valid.tolist(),[True,False,False,False]);self.assertEqual(values[0],2)
        K=np.array([[30.,0,32],[0,30,24],[0,0,1.]])
        E=np.array([np.c_[np.eye(3),[0,0,0]],np.c_[np.eye(3),[-.2,0,0]]],float)
        config={**self.config,'depth_sample_stride':4,'min_depth_samples':4}
        depths=np.full((2,48,64),4.)
        r=cross_view(depths,E,np.stack([K,K]),config)
        self.assertLess(r['median_pair_relative_z_error'],1e-10)
        depths[1]=8
        self.assertGreater(cross_view(depths,E,np.stack([K,K]),config)['median_pair_relative_z_error'],.4)

    def test_missing_evidence_and_method_failure_do_not_get_scores(self):
        manifest={'scene_id':'fixture','images':[{'image_id':'a','width':640,'height':480},{'image_id':'b','width':640,'height':480}],'preprocessing':{'operation':'identity'}}
        prediction=result_template('fixture','0'*64,['a','b'],'fixture')
        observed={'pairs':[{'indices':[0,1],'audit_matches':0,'evidence_status':'insufficient','array_prefix':'0_1'}]}
        r=analyze_prediction(prediction,manifest,observed,{},self.config)
        self.assertEqual(r['status'],'unknown');self.assertIsNone(r['summary']['median_pair_sampson_px']);self.assertIsNone(r['correctness_probability'])
        prediction.update(status='success',failure_reason=None,images=[{'image_id':k,'registered':True,'camera':c.as_dict(),'reason':None} for k,c in zip(['a','b'],[self.a,self.b])])
        r=analyze_prediction(prediction,manifest,observed,{},self.config)
        self.assertEqual(r['pairs'][0]['reason'],'insufficient_matches_or_spatial_coverage')
        prediction['role']='reference'
        with self.assertRaises(ValueError):analyze_prediction(prediction,manifest,observed,{},self.config)

    def test_connectivity_counts_isolated_views(self):
        self.assertEqual(components(3,[(1,2)]),{'components':2,'largest_component_views':2,'isolated_views':1})

    def test_gt_guard_blocks_real_file_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'reference.json';p.write_text('GT sentinel')
            code='import sys;from pathlib import Path;from georecon.reliability.analyze import deny_ground_truth;sys.addaudithook(deny_ground_truth);Path(sys.argv[1]).read_text()'
            process=subprocess.run([sys.executable,'-c',code,str(p)],capture_output=True,text=True)
            self.assertNotEqual(process.returncode,0);self.assertIn('PermissionError',process.stderr)
        for filename in ['reference.json','evaluation.json','calibration/pos_001.txt']:
            with self.assertRaises(PermissionError):deny_ground_truth('open',(filename,'r',0))

    @unittest.skipUnless(importlib.util.find_spec('cv2'),'requires independent OpenCV environment')
    def test_textureless_images_are_unknown_not_perfect(self):
        from georecon.reliability.observations import match_images
        observed,arrays=match_images([np.zeros((64,64,3),np.uint8)]*3,self.config)
        self.assertTrue(all(p['evidence_status']=='insufficient' for p in observed['pairs']))
        self.assertEqual(observed['evidence_graph']['isolated_views'],3)

    @unittest.skipUnless(importlib.util.find_spec('cv2'),'requires independent OpenCV environment')
    def test_image_matches_are_repeatable_without_prediction_or_gt(self):
        from georecon.reliability.observations import match_images
        rng=np.random.default_rng(5);im=rng.integers(0,256,(240,320,3),dtype=np.uint8)
        other=np.roll(im,4,axis=1)
        a,x=match_images([im,other],self.config);b,y=match_images([im,other],self.config)
        self.assertEqual(a,b)
        self.assertGreater(a['pairs'][0]['audit_matches'],0)
        for k in x:assert_allclose(x[k],y[k])
