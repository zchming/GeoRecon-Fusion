"""Independent post-selection GT evaluation and unchanged-export/cost verification."""
from pathlib import Path
import shutil
import sys
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.evaluation.poses import evaluate_poses
from georecon.provenance import code_provenance
from georecon.selection.policy import choose
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5';OUT=ROOT/'outputs/stage5_v1'
sys.path.insert(0,str(ROOT))
from scripts.fit_stage5 import score


def evaluate():
    lock=read_json(DOC/'input_lock.json');fit=read_json(DOC/'fit_report.json');policy=ROOT/'configs/selection/stage5_v1.json';freeze=read_json(ROOT/'docs/stage3/freeze.json')
    assert sha256(policy)==fit['policy_sha256'] and sha256(DOC/'input_lock.json')==fit['input_lock_sha256']
    assert all('scan6' not in p and 'evaluation.json' not in p and 'results.json' not in p for p in fit['project_file_accesses'])
    rows=[];baseline=[];runs=[]
    source_sha=code_provenance('real')['source_tree_sha256']
    for case in lock['cases']:
        scene=case['scene_id'];root=OUT/scene;bundle=ROOT/case['bundle']
        for name,digest in case['files'].items():assert sha256(bundle/name)==digest
        prediction=validate_result(read_json(root/'prediction.json'));decision=read_json(root/'decision.json');cost=read_json(root/'cost_accounting.json');invocation=read_json(root/'invocation.json')
        assert prediction['config']['policy_sha256']==sha256(policy)
        assert prediction['provenance']['code_commit'].startswith('131962e') and prediction['provenance']['source_tree_sha256']==source_sha
        assert decision['correctness_probability'] is None
        manifest=read_json(bundle/'manifest.json');diagnostics=read_json(bundle/'reliability.json')
        candidates={m:read_json(bundle/(m+'.json')) for m in ['colmap','vggt_raw','vggt_ba']}
        hashes={m:sha256(bundle/(m+'.json')) for m in candidates}
        assert decision==choose(candidates,diagnostics,read_json(policy),scene,sha256(bundle/'manifest.json'),[i['image_id'] for i in manifest['images']],hashes)
        selected=decision['selected_method']
        if selected:
            original=read_json(bundle/(selected+'.json'))
            for key in ['images','coordinate_convention','world_unit']:assert prediction[key]==original[key]
            for name,artifact in prediction['artifacts'].items():
                if artifact is None:assert original['artifacts'][name] is None
                else:
                    assert sha256(root/artifact['path'])==artifact['sha256']==original['artifacts'][name]['sha256']
                    assert artifact['semantics']==original['artifacts'][name]['semantics']
        else:assert not any(i['registered'] for i in prediction['images'])
        expected_cost=sum(b['invocation']['full_cli_wall_seconds'] for b in cost['baseline_branches'].values())+cost['shared_diagnostics_invocation']['full_process_wall_seconds']+invocation['full_process_wall_seconds']
        assert np.isclose(expected_cost,cost['accounted_eager_total_seconds']) and np.isclose(expected_cost,prediction['timing_seconds']['total'])
        entry=next(d for d in freeze['datasets'] if d['manifest']['scene_id']==scene)
        reference=(ROOT/entry['path']).parent/'reference.json';assert sha256(reference)==entry['reference_sha256']
        truth=read_json(reference);e=evaluate_poses(prediction,truth);write_json(root/'evaluation.json',e)
        def row(method,p,e):
            return dict(scene_id=scene,split=case['split'],method=method,status=p['status'],coverage=e['registration_ratio'],joint_auc={str(t):score(e,t) for t in [5,10,20]},rotation_deg=e['rotation_deg_registered_subset']['mean'],translation_deg=e['translation_direction_deg_registered_subset']['mean'],ate_mm=e['camera_center_ate']['rmse'])
        r=row('selector',prediction,e);r.update(selected_method=selected,fallback_used=decision['fallback_used'],warning=decision['warning'],accounted_eager_seconds=cost['accounted_eager_total_seconds'],new_process_seconds=invocation['full_process_wall_seconds']);rows.append(r)
        for method in ['colmap','vggt_raw','vggt_ba']:
            p=read_json(bundle/(method+'.json'));be=evaluate_poses(p,truth);b=row(method,p,be);b['historical_cli_seconds']=cost['baseline_branches'][method]['invocation']['full_cli_wall_seconds'];baseline.append(b)
            if method==selected:
                for key in ['pairs','rotation_deg_registered_subset','translation_direction_deg_registered_subset','camera_center_ate']:assert e[key]==be[key]
        target=DOC/'runs'/scene;target.mkdir(parents=True,exist_ok=True)
        for name in ['prediction.json','decision.json','invocation.json','cost_accounting.json','evaluation.json']:shutil.copyfile(root/name,target/name)
        shutil.copyfile(OUT/(scene+'_worker.txt'),target/'worker_log.txt')
        runs.append({'scene_id':scene,'memory_bytes':prediction['memory_bytes'],'selection_internal_seconds':prediction['timing_seconds']['selection_and_artifact_export']})
    groups=[]
    for split in ['dev','test']:
        for method in ['colmap','vggt_raw','vggt_ba','selector']:
            part=[r for r in rows+baseline if r['split']==split and r['method']==method]
            def stats(key):
                data=[r[key] for r in part if r[key] is not None];return {'count':len(data),'mean':float(np.mean(data)) if data else None,'median':float(np.median(data)) if data else None}
            groups.append({'split':split,'method':method,'cases':len(part),'successes':sum(r['status']=='success' for r in part),'mean_joint_auc':{str(t):float(np.mean([r['joint_auc'][str(t)] for r in part])) for t in [5,10,20]},'rotation_available_subset':stats('rotation_deg'),'translation_available_subset':stats('translation_deg'),'ate_available_subset':stats('ate_mm')})
    assert np.isclose(next(g for g in groups if g['split']=='dev' and g['method']=='selector')['mean_joint_auc']['10'],fit['winning_mean_dev_joint_auc10'],atol=1e-12,rtol=0)
    probe=read_json(DOC/'gt_isolation_probe.json');faults=read_json(DOC/'fault_probes.json')
    assert probe['decision_cameras_and_artifact_hashes_identical'] and len(faults['probes'])==10
    write_json(DOC/'evaluation.json',{'role':'posthoc_GT_evaluation_not_policy_input','policy_sha256':sha256(policy),'rows':rows,'baseline_rows':baseline,'groups':groups,'runs':runs,
                                     'no_fresh_holdout_claim':'scan6 previously evaluated at stages3/4; no scan6 data read during fitting'})
    write_json(DOC/'validation.json',{'cases':8,'original_baseline_diagnostics_reaudited':24,'policy_fit_dev_only_file_audit_passed':True,'policy_sha256':sha256(policy),'source_commit':'131962e','source_and_all_input_hashes_verified':True,
                                     'whole_scene_cameras_frames_and_artifact_bytes_preserved':True,'all_branch_and_diagnostic_costs_accounted':True,'natural_fallbacks':sum(r['fallback_used'] for r in rows),'partial_evidence_warnings':sum(r['warning'] is not None for r in rows),
                                     'dev_only_counterfactual_probes':10,'gt_isolation_probe_passed':True,'geometry_filtering_weighting_or_joint_optimization':False,'additional_GPU_allocation':False})
    print('Independent selection evaluation, unchanged-export and eager-cost checks passed')


if __name__=='__main__':evaluate()
