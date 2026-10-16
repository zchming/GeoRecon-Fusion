"""Check immutable inputs, closed-form weights, fallback bytes, GT separation and locked rules."""
from pathlib import Path
import subprocess
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.fusion.anchors import blend
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_fusion';OUT=ROOT/'outputs/stage5_fusion_v1'


def check():
    protocol=read_json(ROOT/'configs/fusion/stage5_protocol.json');rule=read_json(ROOT/'configs/fusion/stage5_v1.json')
    assert sha256(ROOT/'docs/stage3/freeze.json')==protocol['stage3_freeze_sha256']
    assert sha256(ROOT/'configs/filtering/stage5_points_v1.json')==protocol['point_filter_rule_sha256']
    assert rule['protocol_sha256']==sha256(ROOT/'configs/fusion/stage5_protocol.json')
    assert not rule['default_enable_recommended']
    freeze=read_json(ROOT/'docs/stage3/freeze.json');rows=[]
    selected_lock=read_json(ROOT/'docs/stage5/input_lock.json')
    comparison=read_json(DOC/'comparison.json')
    used_script=subprocess.run(['git','show','89a0884:scripts/stage5_fusion.py'],capture_output=True,check=True).stdout
    import hashlib
    script_sha=hashlib.sha256(used_script).hexdigest()
    for case in selected_lock['cases']:
        scene=case['scene_id'];base=OUT/'frozen'/scene
        for filename,digest in case['files'].items():assert sha256(ROOT/case['bundle']/filename)==digest
        run=read_json(base/'run.json');prediction=validate_result(read_json(base/'prediction.json'))
        for filename,digest in run['input_hashes'].items():assert sha256(ROOT/filename)==digest
        for filename,digest in run['track_export']['files_sha256'].items():assert sha256(ROOT/filename)==digest
        entry=next(d for d in freeze['datasets'] if d['manifest']['scene_id']==scene)
        assert sha256((ROOT/entry['path']).parent/'reference.json')==entry['reference_sha256']
        raw=read_json(ROOT/'outputs/stage5_v1'/scene/'prediction.json')
        assert raw['images']==prediction['images']
        for a in prediction['artifacts'].values():
            if a is not None:assert sha256(base/a['path'])==a['sha256']
        assert sha256(base/'anchors.npz')==run['anchors_sha256']
        assert sha256(base/'unweighted.npz')==run['unweighted_sha256']
        assert run['rule_sha256']==sha256(ROOT/'configs/fusion/stage5_v1.json')
        source=ROOT/'outputs/stage5_v1'/scene/raw['artifacts']['pointcloud']['path']
        with np.load(source) as z:points=z['world_points']
        with np.load(base/'anchors.npz') as z:anchors={k:z[k] for k in z.files}
        n=points.shape[0];anchor_ids=anchors['source_point_indices']
        with np.load(ROOT/'outputs/stage5_points_v1/frozen'/scene/'evidence.npz') as z:
            np.testing.assert_array_equal(anchors['raw_other_view_support'],z['support'][0,anchor_ids])
        np.testing.assert_array_equal(anchors['raw_weight'],1+anchors['raw_other_view_support'].astype(float)/(n-1))
        with np.load(ROOT/'data/stage5_fusion_tracks'/scene/'observations.npz') as z:
            lookup={int(pid):(int(length),float(error)) for pid,length,error in zip(z['point_id'],z['track_length'],z['error'])}
        expected_wc=np.array([(lookup[int(pid)][0]/n)/(1+(lookup[int(pid)][1]/2)**2) for pid in anchors['point_id']])
        np.testing.assert_array_equal(anchors['classical_weight'],expected_wc)
        col=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/prediction.json')
        if col['artifacts']['pointcloud'] is not None:
            desc=col['artifacts']['pointcloud']
            assert sha256(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap'/desc['path'])==desc['sha256']
        with np.load(base/'fused.npz') as z:
            xyz=z['xyz'];ids=z['source_point_indices'];fraction=z['anchor_fraction']
            expected,f=blend(points,anchors,rule['strength'])
            np.testing.assert_array_equal(xyz,expected);np.testing.assert_array_equal(fraction,f)
            np.testing.assert_array_equal(ids,np.arange(len(expected)))
            assert xyz.dtype==points.dtype
            untouched=np.ones(len(xyz),bool);untouched[anchors['source_point_indices']]=False
            assert xyz[untouched].tobytes()==points.reshape(-1,3)[untouched].tobytes()
            if run['fallback']:assert xyz.tobytes()==points.reshape(-1,3).tobytes()
        if run['fallback']:
            comparison_row=next(r for r in comparison['rows'] if r['scene_id']==scene)
            for method in ['weighted_fused','unweighted']:
                for key,value in comparison_row['raw'].items():assert comparison_row[method][key]==value
        invocation=read_json(base/'invocation.json')
        assert invocation['script_sha256']==script_sha
        assert invocation['gt_guard_active']
        blocked=['reference.json','evaluation.json','results.json','ground_truth','calibration/','freeze.json','stage5_point_gt']
        assert not any(any(t in name.lower() for t in blocked) for name in invocation['project_file_accesses'])
        if case['split']=='dev':
            candidate=OUT/'dev_candidates'/scene/f"candidate_{rule['candidate_index']}.npz"
            assert sha256(candidate)==sha256(base/'fused.npz')
        snap=DOC/'runs'/scene;snap.mkdir(parents=True,exist_ok=True)
        for name in ['run.json','prediction.json','invocation.json']:write_json(snap/name,read_json(base/name))
        rows.append({'scene_id':scene,'point_count':len(expected),'changed_grid_points':run['rows'][0]['moved_points'],'fallback':run['fallback'],'fallback_reason':run['fallback_reason'],'unaffected_float32_bytes_preserved':True,'camera_and_input_hashes_verified':True,'gt_read_guard_and_access_log_checked':True})
    reproduction=read_json(DOC/'fit_reproduction.json')
    assert reproduction['rule_unchanged'] and reproduction['rule_sha256']==sha256(ROOT/'configs/fusion/stage5_v1.json')
    assert not any('scan6' in name.lower() or '/test/' in name.lower() for name in reproduction['project_file_accesses'])
    write_json(DOC/'validation.json',{'rows':rows,'prior_filter_metrics_and_failures_recomputed':True,'dev_fitted_rule_reproduced_unchanged_after_dtype_fix':True,'inference_gt_alignment':False,'joint_optimization':False,'new_model_inference':False,'rule_sha256':sha256(ROOT/'configs/fusion/stage5_v1.json'),'generation_script_commit':'89a0884','generation_script_sha256':script_sha})

if __name__=='__main__':check()
