import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from georecon.data.protocol import result_template,write_json,read_json,sha256
from georecon.geometry.camera import Camera
from georecon.selection.policy import choose,validate_policy

POLICY={'schema_version':1,'priority':['vggt_raw','colmap','vggt_ba'],'gate':'complete_cameras_and_nonempty_evidence','partial_evidence':'allow_with_warning','no_candidate':'abstain'}
METHODS=['colmap','vggt_raw','vggt_ba']

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids=['a','b','c'];self.candidates={};self.hashes={m:'a'*64 for m in METHODS}
        for m in METHODS:
            p=result_template('fixture','0'*64,self.ids,m)
            p.update(status='success',failure_reason=None,images=[{'image_id':k,'registered':True,'camera':Camera(np.array([[300.,0,320],[0,300,240],[0,0,1]]),np.eye(3),np.array([-float(i),0,0]),640,480).as_dict(),'reason':None} for i,k in enumerate(self.ids)])
            self.candidates[m]=p
        self.diag={'scene_id':'fixture','input_manifest_sha256':'0'*64,'methods':{m:{'prediction_sha256':self.hashes[m],'baseline_status':'success','status':'evidence_available','summary':{'evidence_pairs':3,'median_pair_sampson_px':.01 if m=='colmap' else 100}} for m in METHODS}}
    def choose(self,**kwargs):
        return choose(self.candidates,self.diag,POLICY,'fixture','0'*64,self.ids,self.hashes,**kwargs)

    def test_priority_not_smallest_residual(self):
        self.assertEqual(self.choose()['selected_method'],'vggt_raw')
        self.assertIsNone(self.choose()['correctness_probability'])

    def test_missing_preferred_and_two_stage_fallback(self):
        self.candidates['vggt_raw']=None
        self.assertEqual(self.choose()['selected_method'],'colmap')
        self.candidates['colmap']=None
        r=self.choose();self.assertEqual(r['selected_method'],'vggt_ba');self.assertTrue(r['fallback_used'])
        self.candidates['vggt_ba']=None
        self.assertEqual(self.choose()['outcome'],'abstain')

    def test_oom_or_partial_never_mixed_with_other_cameras(self):
        p=self.candidates['vggt_raw'];p.update(status='partial',failure_reason='fixture missing camera')
        p['images'][-1].update(registered=False,camera=None,reason='fixture_missing')
        self.assertEqual(self.choose()['selected_method'],'colmap')
        self.candidates['colmap'].update(status='oom',failure_reason='fixture OOM')
        self.assertEqual(self.choose()['selected_method'],'vggt_ba')

    def test_empty_unknown_abstains_partial_warns(self):
        for d in self.diag['methods'].values():d.update(status='unknown');d['summary']['evidence_pairs']=0
        self.assertEqual(self.choose()['outcome'],'abstain')
        self.diag['methods']['vggt_raw'].update(status='partial_evidence');self.diag['methods']['vggt_raw']['summary']['evidence_pairs']=1
        r=self.choose();self.assertEqual(r['selected_method'],'vggt_raw');self.assertIsNotNone(r['warning'])

    def test_stale_or_wrong_input_is_rejected(self):
        self.diag['methods']['vggt_raw']['prediction_sha256']='b'*64
        self.assertEqual(self.choose()['selected_method'],'colmap')
        self.diag['input_manifest_sha256']='1'*64
        with self.assertRaises(ValueError):self.choose()

    def test_reference_and_invalid_predictions_fail_closed(self):
        self.candidates['vggt_raw']['role']='reference'
        self.assertEqual(self.choose()['selected_method'],'colmap')
        self.candidates['colmap']={'broken':'fixture'}
        self.assertEqual(self.choose()['selected_method'],'vggt_ba')

    def test_integrity_failure_causes_fallback(self):
        r=self.choose(availability_errors={'vggt_raw':['artifact_missing_or_corrupt']})
        self.assertEqual(r['selected_method'],'colmap')
        self.assertIn('artifact_missing_or_corrupt',r['trace'][0]['reasons'])

    def test_invalid_policy_cannot_add_new_algorithm(self):
        with self.assertRaises(ValueError):validate_policy({**POLICY,'priority':['colmap','colmap','vggt_raw']})
        with self.assertRaises(ValueError):validate_policy({**POLICY,'gate':'minimum_residual'})

    def test_export_preserves_camera_frames_and_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=root/'bundle';bundle.mkdir()
            manifest={'kind':'synthetic','scene_id':'fixture','images':[{'image_id':i} for i in self.ids]}
            write_json(bundle/'manifest.json',manifest);fingerprint=sha256(bundle/'manifest.json')
            diagnostics=copy.deepcopy(self.diag);diagnostics['input_manifest_sha256']=fingerprint
            for m in METHODS:
                p=copy.deepcopy(self.candidates[m]);p['input_manifest_sha256']=fingerprint
                path=bundle/'artifacts'/m/'cloud.bin';path.parent.mkdir(parents=True);path.write_bytes(('unchanged '+m).encode())
                p['artifacts']['pointcloud']={'path':'cloud.bin','sha256':sha256(path),'semantics':'synthetic bytes fixture'}
                write_json(bundle/(m+'.json'),p);diagnostics['methods'][m]['prediction_sha256']=sha256(bundle/(m+'.json'))
            write_json(bundle/'reliability.json',diagnostics);write_json(root/'policy.json',POLICY)
            for attempt,method in [('first','vggt_raw'),('after_corruption','colmap')]:
                if attempt=='after_corruption':(bundle/'artifacts/vggt_raw/cloud.bin').write_bytes(b'corrupt fixture')
                out=root/attempt
                subprocess.run([sys.executable,'-m','georecon.selection.run',str(bundle),str(root/'policy.json'),str(out)],check=True,capture_output=True,text=True)
                r=read_json(out/'prediction.json');decision=read_json(out/'decision.json')
                self.assertEqual(decision['selected_method'],method)
                self.assertEqual(r['images'],read_json(bundle/(method+'.json'))['images'])
                self.assertEqual((out/r['artifacts']['pointcloud']['path']).read_bytes(),('unchanged '+method).encode())

    def test_fit_guard_forbids_test_data_and_mixed_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'scan6'/'candidate.json';path.parent.mkdir();path.write_text('{}')
            code='import sys;from pathlib import Path;from scripts.fit_stage5 import dev_only_guard;sys.addaudithook(dev_only_guard);Path(sys.argv[1]).read_text()'
            p=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('PermissionError',p.stderr)
