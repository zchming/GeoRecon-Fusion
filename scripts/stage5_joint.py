"""Explicit stage5 joint refinement: GT-free run, dev-only configuration, separate evaluation."""
import argparse,copy,resource,sys,time
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.optimization.joint import prepare,solve,scatter
from georecon.provenance import code_provenance
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_joint';OUT=ROOT/'outputs/stage5_joint_v1'
PROTOCOL=ROOT/'configs/optimization/stage5_protocol.json';RULE=ROOT/'configs/optimization/stage5_v1.json';CONDITIONS=['blur5','normal5','reduced3','spaced3']


def load(scene):
    fused=ROOT/'outputs/stage5_fusion_v1/frozen'/scene
    previous=read_json(fused/'run.json');pred=read_json(fused/'prediction.json')
    assert sha256(fused/'fused.npz')==previous['rows'][0]['sha256']
    assert sha256(fused/'anchors.npz')==previous['anchors_sha256']
    for name,digest in previous['input_hashes'].items():assert sha256(ROOT/name)==digest
    raw=ROOT/'outputs/stage5_v1'/scene/'selected_artifacts/raw_predictions.npz'
    obs=ROOT/'data/stage5_fusion_tracks'/scene/'observations.npz'
    cm=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/input_mapping.json'
    rm=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'vggt_raw/input_mapping.json'
    with np.load(fused/'fused.npz') as z:before=z['xyz']
    with np.load(fused/'anchors.npz') as z:anchors={k:z[k] for k in z.files}
    with np.load(obs) as z:tracks={k:z[k] for k in z.files}
    rotations=np.array([i['camera']['R'] for i in pred['images']]);translations=np.array([i['camera']['t'] for i in pred['images']])
    with np.load(raw) as z:
        raw_points=z['world_points'];intrinsics=z['intrinsics'];shape=raw_points.shape[:3]
    problem=prepare(raw_points.reshape(-1,3),before,anchors,tracks,read_json(cm),read_json(rm),rotations,translations,intrinsics,shape)
    files=[fused/'run.json',fused/'prediction.json',fused/'fused.npz',fused/'anchors.npz',raw,obs,cm,rm]
    return before,pred,problem,{str(p.relative_to(ROOT)):sha256(p) for p in files},previous


def camera_prediction(pred,solution,accepted):
    output=copy.deepcopy(pred)
    if accepted:
        # First pose is fixed exactly, including source serialization.
        for i in range(1,len(output['images'])):
            output['images'][i]['camera']['R']=solution['R'][i].tolist()
            output['images'][i]['camera']['t']=solution['t'][i].tolist()
    return output


def generate(scene,dev=False):
    start=time.perf_counter();protocol=read_json(PROTOCOL)
    assert sha256(ROOT/'configs/fusion/stage5_v1.json')==protocol['fusion_rule_sha256']
    protocol=copy.deepcopy(protocol);protocol['fusion_strength']=read_json(ROOT/'configs/fusion/stage5_v1.json')['strength']
    before,pred,problem,hashes,previous=load(scene)
    dest=OUT/('dev_candidates' if dev else 'frozen')/scene
    if dest.exists():raise ValueError('preserve existing run; use a new output root')
    dest.mkdir(parents=True)
    rule=None if dev else read_json(RULE)
    if rule is not None:assert rule['protocol_sha256']==sha256(PROTOCOL)
    strengths=protocol['candidate_pose_prior_strengths'] if dev else [rule['pose_prior_strength']]
    initial=before.copy() if problem is None else scatter(before,problem,problem['x0'])
    np.savez_compressed(dest/'grouped_initial.npz',xyz=initial,source_point_indices=np.arange(len(before)))
    if problem is not None:np.savez_compressed(dest/'problem.npz',**problem)
    rows=[]
    for i,strength in enumerate(strengths):
        solution,record=(None,{'accepted':False,'status':'no_eligible_fusion_correspondence','upstream_reason':previous['fallback_reason']}) if problem is None else solve(problem,protocol,strength)
        name=f'candidate_{i}' if dev else 'optimized'
        accepted=record['accepted'];xyz=scatter(before,problem,solution['points']) if accepted else before.copy()
        if solution is not None:
            attempt=scatter(before,problem,solution['points'])
            np.savez_compressed(dest/(name+'_attempt.npz'),xyz=attempt,parameters=solution['parameters'],R=solution['R'],t=solution['t'],shared_points=solution['points'],pose_delta=solution['pose_delta'])
        np.savez_compressed(dest/(name+'.npz'),xyz=xyz,source_point_indices=np.arange(len(before)))
        out=camera_prediction(pred,solution,accepted)
        from georecon.selection.run import link_or_copy
        for a in out['artifacts'].values():
            if a is not None:
                target=dest/a['path']
                if not target.exists():link_or_copy(ROOT/'outputs/stage5_fusion_v1/frozen'/scene/a['path'],target)
        out['method_id']='joint_pose_sparse_structure_v1';out['config']={'joint_protocol':protocol,'pose_prior_strength':strength,'source_fusion_config':pred['config']}
        out['provenance']['joint_refinement']={'code':code_provenance('real'),'input_hashes':hashes,'protocol_sha256':sha256(PROTOCOL),'rule_sha256':None if dev else sha256(RULE),
            'accepted':accepted,'solver_status':record['status'],'gt_access':'none; open deny guard active','frame':'raw gauge; first pose fixed and raw/anchor priors fix scale','depth_confidence_semantics':'unoptimized source diagnostics; not a regenerated optimized depth map'}
        out['artifacts']['pointcloud']={'path':name+'.npz','sha256':sha256(dest/(name+'.npz')),'semantics':'full float32 source grid; eligible track grid positions share optimized XYZ if accepted, otherwise source fusion XYZ','arrays':{'xyz':[len(xyz),3]}}
        out['timing_seconds']={'joint_incremental_through_candidate':time.perf_counter()-start};out['memory_bytes']={'peak_process_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':0}
        validate_result(out);write_json(dest/(name+'_prediction.json'),out)
        if not dev:write_json(dest/'prediction.json',out)
        rows.append({'name':name,'pose_prior_strength':strength,'solver':record,'artifact_sha256':sha256(dest/(name+'.npz')),'prediction_sha256':sha256(dest/(name+'_prediction.json')),
            'moved_points_vs_fusion':int(np.any(xyz!=before,axis=1).sum()),'full_point_count':len(xyz),'parameters_attempt_preserved':solution is not None})
    write_json(dest/'run.json',{'scene_id':scene,'rows':rows,'input_hashes':hashes,'protocol_sha256':sha256(PROTOCOL),'rule_sha256':None if dev else sha256(RULE),'grouped_initial_sha256':sha256(dest/'grouped_initial.npz'),
        'problem_sha256':sha256(dest/'problem.npz') if problem is not None else None,'wall_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'device':'CPU','gpu_allocated_bytes':0})
    print(scene,[(r['solver']['status'],r['solver']['accepted'],r['moved_points_vs_fusion']) for r in rows],flush=True)


def fit(verify=False):
    from fit_stage5 import dev_only_guard
    reads=[]
    def guard(event,args):
        dev_only_guard(event,args)
        if event=='open' and isinstance(args[0],str) and args[0].startswith(str(ROOT)):reads.append(str(Path(args[0]).relative_to(ROOT)))
    sys.addaudithook(guard)
    from stage5_points import Evaluator
    from stage5_fusion import measure
    if RULE.exists() and not verify:raise ValueError('configuration frozen; use --verify-frozen')
    protocol=read_json(PROTOCOL);ev=Evaluator(1);rows=[]
    for cond in CONDITIONS:
        scene='dtu_scan1_'+cond;ev.prepare(scene);base=OUT/'dev_candidates'/scene
        run=read_json(base/'run.json');assert run['protocol_sha256']==sha256(PROTOCOL)
        row={'scene_id':scene,'raw':ev.measure(np.ones(ev.n,bool)),'before_fused':measure(ev,ROOT/'outputs/stage5_fusion_v1/frozen'/scene/'fused.npz'),'grouped_initial':measure(ev,base/'grouped_initial.npz'),'candidates':[]}
        for c in run['rows']:
            p=base/(c['name']+'.npz');assert sha256(p)==c['artifact_sha256']
            row['candidates'].append(measure(ev,p))
        rows.append(row)
    scores=[float(np.mean([r['candidates'][i]['thresholds_mm']['2']['f1'] for r in rows])) for i in range(len(protocol['candidate_pose_prior_strengths']))]
    best=int(np.argmax(scores));rule={'schema_version':1,'pose_prior_strength':protocol['candidate_pose_prior_strengths'][best],'candidate_index':best,'protocol_sha256':sha256(PROTOCOL),'fit_split':'dev_only','default_enable_recommended':False}
    if verify:
        assert rule==read_json(RULE);write_json(DOC/'fit_reproduction.json',{'frozen_rule_unchanged':True,'rule_sha256':sha256(RULE),'scores':scores,'project_file_accesses':sorted(set(reads))});return
    write_json(RULE,rule);write_json(DOC/'fit_report.json',{'rows':rows,'macro_dev_f1_2mm':scores,'winner':best,'project_file_accesses':sorted(set(reads)),'test_or_mixed_GT_access':False})
    print('frozen',rule,scores,flush=True)


def evaluate():
    from stage5_points import Evaluator
    from stage5_fusion import measure
    from georecon.evaluation.poses import evaluate_poses
    start=time.perf_counter();rows=[]
    for scan in [1,6]:
        ev=Evaluator(scan)
        for cond in CONDITIONS:
            scene=f'dtu_scan{scan}_{cond}';ev.prepare(scene);base=OUT/'frozen'/scene;source=ROOT/'outputs/stage5_fusion_v1/frozen'/scene
            run=read_json(base/'run.json');assert run['rule_sha256']==sha256(RULE)
            p=read_json(base/'prediction.json');before=read_json(source/'prediction.json');truth=read_json(ROOT/'data/stage3_dtu'/scene.removeprefix('dtu_')/'reference.json')
            row={'scene_id':scene,'split':'dev' if scan==1 else 'test','raw':ev.measure(np.ones(ev.n,bool)),'before_fused':measure(ev,source/'fused.npz'),'grouped_initial':measure(ev,base/'grouped_initial.npz'),'optimized':measure(ev,base/'optimized.npz'),
                 'poses_before':evaluate_poses(before,truth),'poses_after':evaluate_poses(p,truth),'evaluation_point_alignment_fixed_from_raw':ev.alignment,'run':run}
            # Also report absolute camera-center drift with the SAME raw evaluation transform.
            from georecon.geometry.alignment import Similarity
            a=ev.alignment;sim=Similarity(a['scale'],np.array(a['R']),np.array(a['t']))
            def centers(pred):return np.array([-np.array(i['camera']['R']).T@np.array(i['camera']['t']) for i in pred['images']])
            gt=centers(truth)
            row['fixed_frame_camera_rmse_mm']={key:float(np.sqrt(np.mean(np.sum((sim.apply(centers(obj))-gt)**2,axis=1)))) for key,obj in [('before',before),('after',p)]}
            rows.append(row);print(scene,'F1@2',row['before_fused']['thresholds_mm']['2']['f1'],row['optimized']['thresholds_mm']['2']['f1'],flush=True)
    write_json(DOC/'comparison.json',{'rows':rows,'rule_sha256':sha256(RULE),'protocol_sha256':sha256(PROTOCOL),'offline_evaluation_wall_seconds':time.perf_counter()-start,'peak_evaluator_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'test_retuning':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','fit','evaluate']);p.add_argument('--scene');p.add_argument('--dev',action='store_true');p.add_argument('--verify-frozen',action='store_true');p.add_argument('--output-root',type=Path);args=p.parse_args()
    if args.output_root is not None:OUT=args.output_root.resolve()
    if args.action=='generate':
        from georecon.reliability.analyze import deny_ground_truth
        if args.dev:
            from fit_stage5 import dev_only_guard
            sys.addaudithook(dev_only_guard)
        sys.addaudithook(deny_ground_truth);accesses=[]
        def trace(event,values):
            if event=='open' and isinstance(values[0],str) and values[0].startswith(str(ROOT)):accesses.append(str(Path(values[0]).relative_to(ROOT)))
        sys.addaudithook(trace);generate(args.scene,args.dev)
        dest=OUT/('dev_candidates' if args.dev else 'frozen')/args.scene
        write_json(dest/'invocation.json',{'argv':sys.argv,'project_file_accesses':sorted(set(accesses)),'gt_guard_active':True,'script_sha256':sha256(Path(__file__)),'code':code_provenance('real')})
    elif args.action=='fit':fit(args.verify_frozen)
    else:evaluate()
