"""Independent observation/objective/gauge checks; GT is opened only here for evaluation."""
from itertools import combinations
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.geometry.alignment import align_points
from georecon.evaluation.poses import rotation_error_deg,translation_direction_error_deg
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_ba';OUT=ROOT/'outputs/stage6_ba_v1'

def metrics(pred,truth,observations,pixels,xyz,fixed_alignment=None):
    cams={v:item['camera'] for v,item in enumerate(pred['images']) if item['registered']};res=[];depth=[];pairs=[]
    for (v,key,p),uv in zip(observations,pixels):
        c=cams[int(v)];q=np.array(c['R'])@xyz[int(p)]+c['t'];z=q[2];q=np.array(c['K'])@q;res.append(q[:2]/q[2]-uv);depth.append(z)
    res=np.array(res).reshape(-1,2)
    for a,b in combinations(cams,2):
        ca,cb=cams[a],cams[b];ids=[pred['images'][v]['image_id'] for v in [a,b]];ga,gb=[truth['cameras'][i] for i in ids]
        r=np.array(cb['R'])@np.array(ca['R']).T;t=np.array(cb['t'])-r@ca['t'];rg=np.array(gb['R'])@np.array(ga['R']).T;tg=np.array(gb['t'])-rg@ga['t']
        pairs.append({'ids':ids,'rotation_deg':rotation_error_deg(r,rg),'translation_direction_deg':translation_direction_error_deg(t,tg)})
    centers=np.array([-np.array(c['R']).T@c['t'] for c in cams.values()]).reshape(-1,3);target=np.array([-np.array(truth['cameras'][pred['images'][v]['image_id']]['R']).T@truth['cameras'][pred['images'][v]['image_id']]['t'] for v in cams]).reshape(-1,3)
    alignment=None;ate=None;fixed=None;reason=None
    try:
        alignment=align_points(centers,target);ate=float(np.sqrt(np.mean(np.sum((alignment.apply(centers)-target)**2,axis=1))))
        if fixed_alignment is not None:fixed=float(np.sqrt(np.mean(np.sum((fixed_alignment.apply(centers)-target)**2,axis=1))))
    except ValueError as error:reason=str(error)
    return {'registered_images':len(cams),'points':len(xyz),'observations':len(res),'reprojection_rmse_px':float(np.sqrt(np.mean(np.sum(res**2,axis=1)))) if len(res) else None,'reprojection_max_px':float(np.max(np.linalg.norm(res,axis=1))) if len(res) else None,'observations_above_2px':int(np.sum(np.linalg.norm(res,axis=1)>2)),'positive_observations':int(np.sum(np.array(depth)>0)),'robust_objective_scale1':float(np.sum(np.sqrt(1+res**2)-1)) if len(res) else None,'pairs':pairs,'mean_pair_rotation_deg':float(np.mean([p['rotation_deg'] for p in pairs])) if pairs else None,'mean_pair_translation_direction_deg':float(np.mean([p['translation_direction_deg'] for p in pairs])) if pairs else None,'camera_center_ATE_refit_Sim3_mm':ate,'camera_center_ATE_fixed_before_Sim3_mm':fixed,'ATE_unavailable_reason':reason},alignment

def check():
    lock=read_json(DOC/'input_lock.json');source=read_json(DOC/'source_lock.json');cfg=read_json(ROOT/'configs/teaching/stage6_ba.json');assert cfg['loss_scale_px']==1.
    assert sha256(ROOT/'configs/teaching/stage6_ba.json')==lock['config_sha256'];assert sha256(DOC/'input_lock.json')==source['input_lock_sha256']
    for name,h in source['source_sha256'].items():assert sha256(ROOT/name)==h
    artifact_lock=DOC/'output_artifact_hashes.json'
    if artifact_lock.exists():
        for name,h in read_json(artifact_lock).items():assert sha256(ROOT/name)==h
    rows=[];failures=[]
    for case in lock['cases']:
        baseline=ROOT/case['baseline'];out=OUT/case['case_id']
        for name,h in case['baseline_files_sha256'].items():assert sha256(baseline/name)==h
        for key in ['input','reference']:assert sha256(ROOT/case[key])==case[key+'_sha256']
        before=validate_result(read_json(baseline/'prediction.json'));after=validate_result(read_json(out/'prediction.json'));stats=read_json(out/'ba.json');inv=read_json(out/'invocation.json');spec=read_json(ROOT/case['input']);truth=read_json(ROOT/case['reference'])
        assert before['status']==after['status'] and before['failure_reason']==after['failure_reason']
        assert inv['pose_truth_guard_active'] and not any('truth' in name or 'reference.json' in name for name in inv['file_accesses'])
        for a,b in zip(before['images'],after['images']):
            assert a['image_id']==b['image_id'] and a['registered']==b['registered']
            if a['registered']:np.testing.assert_array_equal(a['camera']['K'],b['camera']['K'])
        for artifact in after['artifacts'].values():
            if artifact:assert sha256(out/artifact['path'])==artifact['sha256']
        with np.load(baseline/'sparse_points.npz') as z:old={k:z[k] for k in z.files}
        with np.load(out/'before_points.npz') as z:
            for key,value in old.items():np.testing.assert_array_equal(z[key],value)
        with np.load(out/'sparse_points.npz') as z:new={k:z[k] for k in z.files}
        for key in old:
            if key!='xyz' or not stats['accepted']:np.testing.assert_array_equal(new[key],old[key])
        with np.load(out/'observations.npz') as z:uv=z['pixels'];obs=z['observations'];registered=z['registered_original_indices'];np.testing.assert_array_equal(obs,old['observations'])
        with np.load(baseline/'geometry_steps.npz') as z:
            measured=np.array([z[f'keypoints_{v}'][key] for v,key,p in obs],float).reshape(-1,2);np.testing.assert_array_equal(uv,measured)
        bm,alignment=metrics(before,truth,obs,uv,old['xyz']);am,_=metrics(after,truth,obs,uv,new['xyz'],alignment)
        if stats['accepted']:
            assert am['robust_objective_scale1']<=bm['robust_objective_scale1']+1e-8
            assert am['positive_observations']==len(obs)
            a,b=spec['seed'];assert before['images'][a]==after['images'][a]
            axis=stats['gauge']['fixed_center_axis'];c0=before['images'][b]['camera'];c1=after['images'][b]['camera'];center0=-np.array(c0['R']).T@c0['t'];center1=-np.array(c1['R']).T@c1['t'];np.testing.assert_allclose(center0[axis],center1[axis],rtol=0,atol=1e-12)
            with np.load(out/'ba_candidates.npz') as z:
                np.testing.assert_array_equal(z['xyz'],new['xyz']);np.testing.assert_allclose(z['final_residual_px'].reshape(-1,2),np.array([np.array(after['images'][int(v)]['camera']['K'])@(np.array(after['images'][int(v)]['camera']['R'])@new['xyz'][int(p)]+after['images'][int(v)]['camera']['t']) for v,key,p in obs])[:,:2]/np.array([np.array(after['images'][int(v)]['camera']['R'])[2]@new['xyz'][int(p)]+after['images'][int(v)]['camera']['t'][2] for v,key,p in obs])[:,None]-uv,atol=1e-10)
        else:
            assert before['images']==after['images'];failures.append({'case_id':case['case_id'],'kind':spec['kind'],'reason':stats['reason'],'original_status_retained':after['status']})
        regressions=[]
        for key in ['mean_pair_rotation_deg','mean_pair_translation_direction_deg','camera_center_ATE_refit_Sim3_mm']:
            if bm[key] is not None and am[key]>bm[key]+1e-8:regressions.append(key)
        if am['camera_center_ATE_fixed_before_Sim3_mm'] is not None and am['camera_center_ATE_fixed_before_Sim3_mm']>bm['camera_center_ATE_refit_Sim3_mm']+1e-8:regressions.append('camera_center_ATE_fixed_before_Sim3_mm')
        if bm['reprojection_max_px'] is not None and am['reprojection_max_px']>bm['reprojection_max_px']+1e-8:failures.append({'case_id':case['case_id'],'reason':'accepted_BA_has_worse_max_observation_error','before_px':bm['reprojection_max_px'],'after_px':am['reprojection_max_px'],'observations_above_2px':am['observations_above_2px']})
        if regressions:failures.append({'case_id':case['case_id'],'reason':'accepted_BA_has_GT_metric_regression','metrics':regressions})
        rows.append({'case_id':case['case_id'],'kind':spec['kind'],'BA_accepted':stats['accepted'],'BA_reason':stats['reason'],'converged':stats['converged'],'before':bm,'after':am,'GT_metric_regressions':regressions,'timing_seconds':after['timing_seconds'],'memory_bytes':after['memory_bytes']})
    write_json(DOC/'evaluation.json',rows);write_json(DOC/'failure_catalog.json',failures)
    write_json(DOC/'acceptance.json',{'substep':'stage6 minimal calibrated teaching BA','engineering_acceptance':'passed','all_5_baseline_maps_and_inputs_unchanged':True,'registered_set_K_tracks_and_point_count_unchanged':True,'fixed_first_pose_and_single_scale_coordinate_verified':True,'accepted_outputs_finite_positive_and_nonincreasing_objective':True,'before_after_GT_evaluation_is_offline_only':True,'next_stage_started':False,'stable_general_gain_or_novel_algorithm_proven':False})
    artifacts={str(p.relative_to(ROOT)):sha256(p) for p in OUT.rglob('*') if p.is_file()}
    control_root=ROOT/'outputs/stage6_ba_controls_v1'
    if (DOC/'synthetic_controls.json').exists():
        for row in read_json(DOC/'synthetic_controls.json'):
            p=control_root/(row['case_id']+'.npz');assert sha256(p)==row['artifact_sha256'];assert not row['stats']['accepted'] and row['stats']['reason']==row['expected_reason'];artifacts[str(p.relative_to(ROOT))]=sha256(p)
    if not artifact_lock.exists():write_json(artifact_lock,artifacts)
    print([(r['case_id'],r['BA_reason'],r['before']['reprojection_rmse_px'],r['after']['reprojection_rmse_px'],r['GT_metric_regressions']) for r in rows])
if __name__=='__main__':check()
