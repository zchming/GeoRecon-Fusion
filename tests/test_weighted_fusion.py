import unittest
import numpy as np
from georecon.fusion.anchors import predicted_alignment,associate,blend
from georecon.geometry.alignment import Similarity


class WeightedFusionTest(unittest.TestCase):
    def anchors(self):
        return {'source_point_indices':np.array([1]),'anchor_xyz':np.array([[2.,0,1]]),'classical_weight':np.array([1.]),'raw_weight':np.array([2.])}

    def test_weighted_closed_form_and_no_evidence_preservation(self):
        p=np.array([[0.,0,1],[1,0,1],[9,9,1]])
        before=p.copy();out,f=blend(p,self.anchors(),1)
        np.testing.assert_allclose(out[1],[4/3,0,1]);np.testing.assert_array_equal(out[[0,2]],p[[0,2]])
        np.testing.assert_array_equal(p,before);self.assertAlmostEqual(f[0],1/3)
        unweighted,_=blend(p,self.anchors(),None);self.assertEqual(unweighted[1,0],1.5)
        zero,_=blend(p,self.anchors(),0);np.testing.assert_array_equal(zero,p)

    def test_no_anchor_is_exact_fallback(self):
        p=np.array([[1.,2,3]])
        a={k:v[:0] for k,v in self.anchors().items()}
        out,f=blend(p,a,1);np.testing.assert_array_equal(out,p);self.assertEqual(len(f),0)

    def test_source_float32_and_untouched_bytes_preserved(self):
        p=np.array([[.123456789,2,3],[1,0,1],[.876543219,8,9]],dtype=np.float32)
        out,_=blend(p,self.anchors(),.5)
        self.assertEqual(out.dtype,p.dtype)
        self.assertEqual(out[[0,2]].tobytes(),p[[0,2]].tobytes())
        empty={k:v[:0] for k,v in self.anchors().items()}
        untouched,_=blend(p,empty,.5)
        self.assertEqual(untouched.tobytes(),p.tobytes())

    def test_invalid_weight_or_strength_rejected(self):
        p=np.zeros((2,3))
        for value in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):blend(p,self.anchors(),value)
        for value in [0,-1,np.nan]:
            a=self.anchors();a['classical_weight'][0]=value
            with self.assertRaises(ValueError):blend(p,a,1)

    def prediction(self,centers,rotation=None):
        r=np.eye(3) if rotation is None else rotation
        return {'images':[{'image_id':str(i),'registered':True,'camera':{'model':'PINHOLE','K':np.eye(3).tolist(),'R':r.tolist(),'t':(-r@c).tolist(),'width':10,'height':10}} for i,c in enumerate(np.array(centers,float))]}

    def test_predicted_sim3_without_ground_truth(self):
        centers=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
        a=self.prediction(centers);b=self.prediction(2*centers+[3,4,5])
        sim,record=predicted_alignment(a,b,{'max_center_rmse_over_raw_center_diameter':.1,'max_camera_rotation_disagreement_deg':5})
        self.assertEqual(record['status'],'aligned');self.assertAlmostEqual(sim.scale,2)
        np.testing.assert_allclose(sim.apply(centers),2*centers+[3,4,5],atol=1e-12)

    def test_nonidentity_rotation_gauge_alignment(self):
        centers=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        target=2*centers@rotation.T+[3,4,5]
        a=self.prediction(centers);b=self.prediction(target,rotation.T)
        sim,record=predicted_alignment(a,b,{'max_center_rmse_over_raw_center_diameter':.1,'max_camera_rotation_disagreement_deg':5})
        self.assertEqual(record['status'],'aligned')
        np.testing.assert_allclose(sim.R,rotation,atol=1e-12)
        np.testing.assert_allclose(sim.apply(centers),target,atol=1e-12)

    def test_degenerate_missing_and_disagreement_fallback(self):
        cfg={'max_center_rmse_over_raw_center_diameter':.1,'max_camera_rotation_disagreement_deg':5}
        a=self.prediction([[0,0,0],[1,0,0]])
        self.assertIsNone(predicted_alignment(a,a,cfg)[0])
        a=self.prediction([[0,0,0],[1,0,0],[2,0,0]])
        self.assertEqual(predicted_alignment(a,a,cfg)[1]['reason'],'degenerate_camera_alignment')
        a=self.prediction([[0,0,0],[1,0,0],[0,1,0]])
        r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        b=self.prediction([[0,0,0],[1,0,0],[0,1,0]],r)
        self.assertIsNone(predicted_alignment(a,b,cfg)[0])

    def test_observed_track_half_pixel_mapping_and_collision(self):
        n,h,w=3,4,5;ys,xs=np.mgrid[:h,:w]
        p=np.broadcast_to(np.stack([xs,ys,np.ones_like(xs)],-1),(n,h,w,3)).astype(float).copy()
        depth=np.ones((n,h,w));e=np.broadcast_to(np.eye(4)[:3],(n,3,4));k=np.broadcast_to(np.eye(3),(n,3,3))
        # Native observed pixel 1.5,1.5 becomes zero-based native 1,1, NOT 2,2.
        obs={'image_id':np.array(['0','0','0']), 'xy':np.array([[1.5,1.5]]*3),'point_id':np.array([9,2,1]),'xyz':np.array([[1,1,1.01]]*3),'track_length':np.array([3,3,2]),'error':np.array([.1,.1,.1])}
        cm=[{'image_id':str(i),'pixel_transform':np.eye(3).tolist()} for i in range(n)]
        rm=[{'image_id':str(i),'original_to_model_transform':np.eye(3).tolist()} for i in range(n)]
        cfg={'min_track_length':3,'max_colmap_mean_reprojection_error_common_px':2,'max_aligned_anchor_reprojection_error_native_px':2,'max_3d_disagreement_over_raw_source_z':.05}
        a=associate(p,depth,e,k,obs,cm,rm,Similarity(1,np.eye(3),np.zeros(3)),cfg)
        np.testing.assert_array_equal(a['source_point_indices'],[w+1]);np.testing.assert_array_equal(a['point_id'],[2])
        self.assertEqual(a['counts']['collision_discarded'],1)
        self.assertEqual(a['counts']['track_or_error_rejected'],1)

    def test_unsafe_geometry_does_not_invent_correspondence(self):
        n,h,w=3,4,5;ys,xs=np.mgrid[:h,:w]
        p=np.broadcast_to(np.stack([xs,ys,np.ones_like(xs)],-1),(n,h,w,3)).astype(float)
        obs={'image_id':np.array(['0','0']), 'xy':np.array([[1.5,1.5]]*2),'point_id':np.array([1,2]),'xyz':np.array([[1,1,-1],[10,10,1]]),'track_length':np.array([3,3]),'error':np.array([.1,.1])}
        cm=[{'image_id':str(i),'pixel_transform':np.eye(3)} for i in range(n)];rm=[{'image_id':str(i),'original_to_model_transform':np.eye(3)} for i in range(n)]
        a=associate(p,np.ones((n,h,w)),np.broadcast_to(np.eye(4)[:3],(n,3,4)),np.broadcast_to(np.eye(3),(n,3,3)),obs,cm,rm,Similarity(1,np.eye(3),np.zeros(3)),{'min_track_length':3,'max_colmap_mean_reprojection_error_common_px':2,'max_aligned_anchor_reprojection_error_native_px':2,'max_3d_disagreement_over_raw_source_z':.05})
        self.assertEqual(len(a['source_point_indices']),0)

if __name__=='__main__':unittest.main()
