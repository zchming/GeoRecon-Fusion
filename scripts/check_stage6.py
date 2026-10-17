"""Replay the frozen teaching math and verify no truth entered inference."""
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.provenance import code_provenance
from georecon.teaching.run import load_input
from georecon.teaching.two_view import features_and_matches,reconstruct,triangulate_dlt
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6';OUT=ROOT/'outputs/stage6_two_view_v1'


def check():
    lock=read_json(DOC/'input_lock.json');cfgpath=ROOT/'configs/teaching/stage6_two_view.json';cfg=read_json(cfgpath);assert sha256(cfgpath)==lock['protocol_sha256']
    evaluation=read_json(DOC/'evaluation.json');rows=[]
    current_source_digest=code_provenance('synthetic')['source_tree_sha256']
    for case in lock['cases']:
        path=ROOT/case['input'];assert sha256(path)==case['input_sha256'];spec,images,observations,hashes=load_input(path,cfg)
        base=OUT/case['case_id'];pred=validate_result(read_json(base/'prediction.json'));steps=read_json(base/'steps.json')
        for filename,digest in pred['provenance']['input_hashes'].items():assert sha256(filename)==digest
        if images:
            arrays,stats=features_and_matches(images,cfg);assert stats==steps['feature_stage']
            with np.load(base/'features_matches.npz') as old:
                for name,value in arrays.items():np.testing.assert_array_equal(value,old[name])
            p,q=arrays['pixels_first'],arrays['pixels_second']
        else:p,q=observations['pixels_first'],observations['pixels_second']
        reproduced=reconstruct(p,q,spec['images'][0]['K'],spec['images'][1]['K'],cfg)
        assert reproduced['status']==pred['status'] and reproduced['reason']==pred['failure_reason']
        assert reproduced['stats']==steps['geometry_stage']
        with np.load(base/'geometry_steps.npz') as old:
            for name,value in reproduced['arrays'].items():np.testing.assert_allclose(value,old[name],rtol=0,atol=1e-10,equal_nan=True)
        with np.load(base/'sparse_points.npz') as z:xyz=z['xyz'];kept=z['match_indices']
        assert len(xyz)==steps['geometry_stage']['kept_points']
        if pred['status']=='success':
            r=np.array(pred['images'][1]['camera']['R']);t=np.array(pred['images'][1]['camera']['t'])
            np.testing.assert_array_equal(pred['images'][0]['camera']['R'],np.eye(3));np.testing.assert_array_equal(pred['images'][0]['camera']['t'],np.zeros(3))
            assert np.isclose(np.linalg.norm(t),1,atol=1e-12);assert np.allclose(r.T@r,np.eye(3),atol=1e-8) and np.isclose(np.linalg.det(r),1)
            tri=triangulate_dlt(p[kept],q[kept],spec['images'][0]['K'],spec['images'][1]['K'],r,t)
            np.testing.assert_allclose(tri['xyz'],xyz,atol=1e-9,rtol=0);assert tri['positive'].all()
            assert np.all(tri['parallax_deg']>=cfg['min_parallax_deg']) and np.all(tri['reprojection_px']<=cfg['max_reprojection_px'])
        else:
            assert len(xyz)==0 and pred['artifacts']['pointcloud'] is None
            assert all(not i['registered'] and i['camera'] is None for i in pred['images'])
        for a in pred['artifacts'].values():
            if a is not None:assert sha256(base/a['path'])==a['sha256']
        process=read_json(base/'process.json');assert process['returncode']==(0 if pred['status']=='success' else 1)
        invocation=read_json(base/'invocation.json');assert invocation['pose_truth_guard_active'];assert invocation['code']['source_tree_sha256']==current_source_digest
        forbidden=['stage6_truth','reference.json','evaluation.json','results.json','freeze.json','ground_truth','calibration/']
        assert not any(any(t in f.replace('\\','/').lower() for t in forbidden) for f in invocation['project_file_accesses'])
        evaluated=next(r for r in evaluation['rows'] if r['case_id']==case['case_id']);assert evaluated['expected_status_met']
        snap=DOC/'runs'/case['case_id'];snap.mkdir(parents=True,exist_ok=True)
        for name in ['prediction.json','steps.json','invocation.json','process.json']:write_json(snap/name,read_json(base/name))
        rows.append({'case_id':case['case_id'],'feature_and_geometry_replay_identical':True,'unit_baseline_first_frame_gauge_checked':True,'triangulation_depth_angle_reprojection_checked':True,'input_intrinsics_only_and_no_pose_reference_reads':True,'failed_cases_do_not_export_cameras_or_points':True})
    write_json(DOC/'validation.json',{'rows':rows,'frozen_before_runs_commit':'9e8e030','all_8_expected_control_statuses_met':evaluation['all_expected_statuses_met'],
        'known_K_is_explicit_input':True,'teaching_two_view_only':True,'PnP_incremental_BA_implemented':False,'stage7_started':False,'GPU_allocated_bytes':0})

if __name__=='__main__':check()
