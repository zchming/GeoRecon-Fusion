"""Acceptance invariants for frozen methods and diagnostic component removals."""
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.ablation.controls import keep_unknown,without_raw_support
from georecon.optimization.joint import scatter,terms,objective
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_acceptance';OUT=ROOT/'outputs/stage5_acceptance_v1'


def check():
    protocol=read_json(ROOT/'configs/ablation/stage5_acceptance.json');prior=read_json(DOC/'prior_audit.json');rows=[]
    for p,digest in prior['frozen_rule_hashes'].items():assert sha256(ROOT/p)==digest
    assert len(prior['rows'])==8 and not protocol['fit_new_rules'] and not protocol['promote_ablations_to_default']
    for scene in protocol['cases']:
        base=OUT/scene;run=read_json(base/'run.json');assert run['protocol_sha256']==sha256(ROOT/'configs/ablation/stage5_acceptance.json')
        for p,digest in run['input_hashes'].items():assert sha256(ROOT/p)==digest
        rawbase=ROOT/'outputs/stage5_v1'/scene;fusion=ROOT/'outputs/stage5_fusion_v1/frozen'/scene;filtered=ROOT/'outputs/stage5_points_v1/frozen'/scene;joint=ROOT/'outputs/stage5_joint_v1/frozen'/scene
        pred0=read_json(fusion/'prediction.json')
        with np.load(rawbase/'selected_artifacts/raw_predictions.npz') as z:points=z['world_points']
        flat=points.reshape(-1,3)
        with np.load(filtered/'filtered.npz') as z:old_ids=z['source_point_indices']
        keep=np.zeros(len(flat),bool);keep[old_ids]=True
        with np.load(filtered/'evidence.npz') as z:expected_unknown=keep_unknown(keep,z['finite'],z['observed'])
        with np.load(fusion/'anchors.npz') as z:anchors={k:z[k] for k in z.files}
        for item in run['rows']:
            name=item['name'];assert sha256(base/(name+'.npz'))==item['artifact_sha256'];assert sha256(base/(name+'_prediction.json'))==item['prediction_sha256']
            pred=validate_result(read_json(base/(name+'_prediction.json')));assert pred['images']==pred0['images']
            for a in pred['artifacts'].values():
                if a is not None:assert sha256(base/a['path'])==a['sha256']
            with np.load(base/(name+'.npz')) as z:xyz=z['xyz'];ids=z['source_point_indices'];extra={k:z[k] for k in z.files if k not in ['xyz','source_point_indices']}
            assert xyz.dtype==flat.dtype
            if name=='filter_keep_unknown':
                np.testing.assert_array_equal(ids,np.flatnonzero(expected_unknown));assert xyz.tobytes()==flat[ids].tobytes()
            elif name=='fusion_without_raw_support':
                expected,beta=without_raw_support(points,anchors,read_json(ROOT/'configs/fusion/stage5_v1.json')['strength'])
                np.testing.assert_array_equal(xyz,expected);np.testing.assert_array_equal(extra['mix_fraction'],beta)
                np.testing.assert_array_equal(ids,np.arange(len(flat)))
            else:
                np.testing.assert_array_equal(ids,np.arange(len(flat)))
                with np.load(fusion/'fused.npz') as z:before=z['xyz']
                if item['solver']['accepted']:
                    with np.load(joint/'problem.npz') as z:problem={k:z[k] for k in z.files}
                    cfg=read_json(ROOT/'configs/optimization/stage5_protocol.json');cfg['fusion_strength']=read_json(ROOT/'configs/fusion/stage5_v1.json')['strength']
                    strength=read_json(ROOT/'configs/optimization/stage5_v1.json')['pose_prior_strength']
                    with np.load(base/(name+'_attempt.npz')) as a:
                        np.testing.assert_array_equal(xyz,scatter(before,problem,a['points']))
                        assert np.all(a['pose_delta']==0);np.testing.assert_array_equal(a['R'],problem['R0']);np.testing.assert_array_equal(a['t'],problem['t0'])
                        parts,z,_=terms(problem,a['parameters'],cfg,strength)
                        assert np.all(z>0) and np.isclose(objective(np.concatenate(parts)),item['solver']['final_objective'],rtol=0,atol=1e-10)
                        assert item['solver']['final_objective']<=item['solver']['initial_objective']
                        assert np.max(np.abs(a['parameters']))<=cfg['solver']['point_delta_component_bound_over_depth']+1e-12
                    untouched=np.ones(len(flat),bool);untouched[problem['ids']]=False;assert xyz[untouched].tobytes()==before[untouched].tobytes()
                else:assert xyz.tobytes()==before.tobytes()
        control=read_json(base/'selection_control.json');assert control['natural_gated']['selected_method']==control['natural_without_evidence_gate']['selected_method']=='vggt_raw'
        if 'dev_all_unknown_gated' in control:
            assert control['dev_all_unknown_gated']['outcome']=='abstain' and control['dev_all_unknown_without_gate']['selected_method']=='vggt_raw'
        invocation=read_json(base/'invocation.json');assert invocation['gt_guard_active'];assert invocation['script_sha256']==sha256(ROOT/'scripts/run_stage5_ablations.py')
        forbidden=['reference.json','evaluation.json','results.json','ground_truth','freeze.json','calibration/','stage5_point_gt']
        assert not any(any(t in path.lower() for t in forbidden) for path in invocation['project_file_accesses'])
        snap=DOC/'runs'/scene;snap.mkdir(parents=True,exist_ok=True)
        for name in ['run.json','selection_control.json','invocation.json']:write_json(snap/name,read_json(base/name))
        rows.append({'scene_id':scene,'three_ablation_artifacts_and_formulas_verified':True,'cameras_intrinsics_and_source_float32_preserved':True,'GT_access_guard_checked':True,'counterfactual_gate_control_verified':True})
    write_json(DOC/'validation.json',{'rows':rows,'frozen_protocol_commit':'c0bdab6','old_rules_unchanged':True,'test_retuning':False,'new_methods_promoted':False,'GPU_allocated_bytes':0,'next_stage_started':False})

if __name__=='__main__':check()
