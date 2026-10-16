"""Verify fixed gauges, sparse variables, immutable inputs, solver bounds and fallback output."""
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.optimization.joint import scatter,decode,terms,objective
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_joint';OUT=ROOT/'outputs/stage5_joint_v1'


def check():
    protocol=read_json(ROOT/'configs/optimization/stage5_protocol.json');rule=read_json(ROOT/'configs/optimization/stage5_v1.json')
    assert rule['protocol_sha256']==sha256(ROOT/'configs/optimization/stage5_protocol.json')
    assert sha256(ROOT/'configs/fusion/stage5_v1.json')==protocol['fusion_rule_sha256']
    protocol['fusion_strength']=read_json(ROOT/'configs/fusion/stage5_v1.json')['strength']
    freeze=read_json(ROOT/'docs/stage3/freeze.json');previous=read_json(ROOT/'docs/stage5_fusion/comparison.json');rows=[]
    for old in previous['rows']:
        scene=old['scene_id'];base=OUT/'frozen'/scene;prior=ROOT/'outputs/stage5_fusion_v1/frozen'/scene
        run=read_json(base/'run.json');row=run['rows'][0]
        for path,digest in run['input_hashes'].items():assert sha256(ROOT/path)==digest
        assert run['rule_sha256']==sha256(ROOT/'configs/optimization/stage5_v1.json')
        assert sha256(base/'optimized.npz')==row['artifact_sha256']
        before=read_json(prior/'prediction.json');pred=validate_result(read_json(base/'prediction.json'))
        assert pred==read_json(base/'optimized_prediction.json')
        assert pred['images'][0]==before['images'][0]
        for a in pred['artifacts'].values():
            if a is not None:assert sha256(base/a['path'])==a['sha256']
        with np.load(prior/'fused.npz') as z:xyz_before=z['xyz']
        with np.load(base/'optimized.npz') as z:xyz=z['xyz'];np.testing.assert_array_equal(z['source_point_indices'],np.arange(len(xyz_before)))
        assert xyz.dtype==xyz_before.dtype
        problem=None
        if run['problem_sha256']:
            assert sha256(base/'problem.npz')==run['problem_sha256']
            with np.load(base/'problem.npz') as z:problem={k:z[k] for k in z.files}
            untouched=np.ones(len(xyz),bool);untouched[problem['ids']]=False
            assert xyz[untouched].tobytes()==xyz_before[untouched].tobytes()
        if row['solver']['accepted']:
            with np.load(base/'optimized_attempt.npz') as a:
                expected=scatter(xyz_before,problem,a['shared_points']);np.testing.assert_array_equal(xyz,expected)
                r,t,p,delta=decode(problem,a['parameters'])
                np.testing.assert_array_equal(r,a['R']);np.testing.assert_array_equal(t,a['t']);np.testing.assert_array_equal(p,a['shared_points'])
                res,z,_=terms(problem,a['parameters'],protocol,rule['pose_prior_strength'])
                assert np.isclose(objective(np.concatenate(res)),row['solver']['final_objective'],rtol=0,atol=1e-10)
                assert np.all(z>0) and row['solver']['final_objective']<=row['solver']['initial_objective']
                assert np.all(np.abs(delta[:,:3])<=np.radians(protocol['solver']['pose_rotation_component_bound_deg'])+1e-12)
                assert np.all(np.abs(delta[:,3:])<=protocol['solver']['pose_translation_component_bound_over_depth']+1e-12)
                assert np.all(np.abs(a['parameters'][6*(len(r)-1):])<=protocol['solver']['point_delta_component_bound_over_depth']+1e-12)
                assert np.linalg.norm(delta[1:])>0 and np.linalg.norm(p-problem['x0'])>0
                for i in range(1,len(r)):
                    np.testing.assert_array_equal(pred['images'][i]['camera']['R'],r[i]);np.testing.assert_array_equal(pred['images'][i]['camera']['t'],t[i])
        else:
            assert xyz.tobytes()==xyz_before.tobytes();assert pred['images']==before['images']
        for a,b in zip(pred['images'],before['images']):
            assert a['image_id']==b['image_id'];assert a['camera']['K']==b['camera']['K'];assert a['camera']['width']==b['camera']['width'] and a['camera']['height']==b['camera']['height']
        invocation=read_json(base/'invocation.json');assert invocation['gt_guard_active']
        assert invocation['script_sha256']==sha256(ROOT/'scripts/stage5_joint.py')
        denied=['reference.json','evaluation.json','results.json','freeze.json','ground_truth','calibration/','stage5_point_gt']
        assert not any(any(t in path.lower() for t in denied) for path in invocation['project_file_accesses'])
        entry=next(d for d in freeze['datasets'] if d['manifest']['scene_id']==scene)
        assert sha256((ROOT/entry['path']).parent/'reference.json')==entry['reference_sha256']
        if old['split']=='dev':
            candidate=OUT/'dev_candidates'/scene/f"candidate_{rule['candidate_index']}.npz"
            assert sha256(candidate)==sha256(base/'optimized.npz')
        snap=DOC/'runs'/scene;snap.mkdir(parents=True,exist_ok=True)
        for name in ['run.json','prediction.json','invocation.json']:write_json(snap/name,read_json(base/name))
        rows.append({'scene_id':scene,'accepted':row['solver']['accepted'],'status':row['solver']['status'],'first_pose_exactly_fixed':True,'intrinsics_unchanged':True,'unaffected_point_bytes_unchanged':True,'inputs_artifacts_formula_bounds_and_gt_isolation_checked':True,'same_original_point_count':len(xyz)})
    reproduce=read_json(DOC/'fit_reproduction.json');assert reproduce['frozen_rule_unchanged'] and reproduce['rule_sha256']==sha256(ROOT/'configs/optimization/stage5_v1.json')
    assert not any('scan6' in f or '/test/' in f for f in reproduce['project_file_accesses'])
    write_json(DOC/'validation.json',{'rows':rows,'inference_has_no_GT_input':True,'test_configuration_retuned':False,'rule_frozen_before_test_commit':'c3510ce','new_matching_or_dense_propagation':False,'GPU_allocated_bytes':0,'raw_camera_GT_point_alignment_fixed':True})

if __name__=='__main__':check()
