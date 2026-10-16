import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
from georecon.baselines.vggt_ba import square_geometry, independent_square, validate_config, run_vggt_ba
from georecon.geometry.camera import Camera
from georecon.data.protocol import read_json, write_json, sha256

ROOT=Path(__file__).resolve().parents[1]

class OfficialBAAdapterTests(unittest.TestCase):
    def test_square_pixel_centers_and_odd_padding(self):
        assert_allclose(square_geometry(800,600),[[1.28,0,.14],[0,1.28,128.14],[0,0,1]])
        # 5x2 is padded with one black row above and two below, not a fractional center.
        assert_allclose(square_geometry(5,2,10),[[2,0,.5],[0,2,2.5],[0,0,1]])
        im=Image.new('RGB',(5,2),(255,0,0))
        a=independent_square(im,5)
        assert_allclose(a[0,:,0],[0,1,1,0,0])
        self.assertEqual(a.dtype,np.float32)
        with self.assertRaises(ValueError): square_geometry(0,4)

    def test_camera_roundtrip_with_common_resize_and_square(self):
        common=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
        A=square_geometry(800,600)@common
        K=np.array([[900.,0,512],[0,900.,512],[0,0,1.]])
        camera=Camera(np.linalg.solve(A,K),np.eye(3),np.array([.1,.2,.3]),1600,1200)
        xyz=np.array([[1.,2.,9.],[0,0,3]])
        uv,_,_=camera.project(xyz)
        xy=np.c_[uv,np.ones(2)]@A.T
        direct=(xyz+camera.t)@K.T
        assert_allclose(xy[:,:2],direct[:,:2]/direct[:,2:])
        # A second generic COLMAP half-pixel shift would fail this equality.

    def test_reject_silent_resource_and_algorithm_changes(self):
        c=read_json(ROOT/'configs/methods/vggt_ba_smoke.json')
        validate_config(c)
        for k,v in [('max_query_pts',2048),('fine_tracking',False),('preprocessing','crop'),('gpu_allocator_limit_gib',12),('pycolmap_version','4.2.1')]:
            with self.assertRaises(ValueError): validate_config({**c,k:v})

    def test_all_additional_assets_are_hash_pinned(self):
        c=read_json(ROOT/'configs/assets/vggt_ba.json')
        self.assertEqual(len(c['assets']),4)
        for a in c['assets']:
            self.assertEqual(len(a['sha256']),64)
            self.assertTrue(all(x in '0123456789abcdef' for x in a['sha256']))
        sp=next(a for a in c['assets'] if a['name']=='superpoint')
        self.assertIn('NOT Apache',sp['license_note'])

    def test_oom_and_timeout_keep_failure_without_gt_or_fabricated_cameras(self):
        for timeout in [False,True]:
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                Image.new('RGB',(16,12),'red').save(root/'image.png')
                manifest={'schema_version':1,'kind':'synthetic','scene_id':'ba_fixture','split':'smoke','seed':42,'source':'generated','license_note':'fixture','preprocessing':{'operation':'identity'},'budget':{},'images':[{'image_id':str(i),'path':'image.png','width':16,'height':12,'sha256':sha256(root/'image.png')} for i in range(3)]}
                write_json(root/'manifest.json',manifest)
                (root/'reference.json').write_text('INVALID GT SENTINEL: inference must not open')
                out=root/'out'
                def worker(*a,**kw):
                    write_json(out/'worker_state.json',{'status':'process_error','failure_kind':'oom','error':'synthetic OOM fixture; not an actual GPU run'})
                    return {'returncode':1,'timed_out':timeout,'wall_seconds':.01,'peak_worker_rss_sampled_bytes':None}
                with patch('georecon.baselines.vggt_ba.code_provenance',return_value={'data_kind':'synthetic'}), patch('georecon.baselines.vggt_ba.verify_repository',return_value={}), patch('georecon.baselines.vggt_ba.verify_vendor',return_value={}), patch('georecon.baselines.vggt_ba.sha256',return_value='0'*64), patch('georecon.baselines.vggt_ba.execute_worker',side_effect=worker), patch('georecon.baselines.vggt_ba.subprocess.Popen'):
                    result=run_vggt_ba(root/'manifest.json',ROOT/'configs/methods/vggt_ba_smoke.json',ROOT/'experiment03_depth_reprojection/vggt_official','unused','unused','unused','unused',out)
                self.assertEqual(result['status'],'timeout' if timeout else 'oom')
                self.assertTrue(all(not r['registered'] and r['camera'] is None for r in result['images']))
                self.assertTrue(all(v is None for v in result['artifacts'].values()))
