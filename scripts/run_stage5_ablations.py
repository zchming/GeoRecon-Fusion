"""Predeclared diagnostic ablations. Runtime refuses GT reads and overwrites."""
import argparse,copy,resource,sys,time
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.ablation.controls import keep_unknown,without_raw_support,remove_evidence_gate,fixed_pose_solve
from georecon.optimization.joint import scatter
from georecon.selection.policy import choose
from georecon.selection.run import link_or_copy
from georecon.provenance import code_provenance
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stage5_acceptance_v1';PROTOCOL=ROOT/'configs/ablation/stage5_acceptance.json'


def selection_control(scene):
    bundle=ROOT/'data/stage5_bundles'/scene;policy=read_json(ROOT/'configs/selection/stage5_v1.json')
    manifest=read_json(bundle/'manifest.json');diag=read_json(bundle/'reliability.json')
    preds={m:read_json(bundle/(m+'.json')) for m in policy['priority']};digests={m:sha256(bundle/(m+'.json')) for m in policy['priority']}
    for m,p in preds.items():
        for a in p['artifacts'].values():
            if a is not None:assert sha256(bundle/'artifacts'/m/a['path'])==a['sha256']
    args=(policy,scene,sha256(bundle/'manifest.json'),[i['image_id'] for i in manifest['images']],digests)
    result={'natural_gated':choose(preds,diag,*args),'natural_without_evidence_gate':choose(preds,remove_evidence_gate(diag),*args)}
    assert result['natural_gated']==read_json(ROOT/'outputs/stage5_v1'/scene/'decision.json')
    if manifest['split']=='dev':
        unknown=copy.deepcopy(diag)
        for d in unknown['methods'].values():d['status']='unknown';d['summary']['evidence_pairs']=0
        result['dev_all_unknown_gated']=choose(preds,unknown,*args)
        result['dev_all_unknown_without_gate']=choose(preds,remove_evidence_gate(unknown),*args)
    return result


def run(scene):
    start=time.perf_counter();protocol=read_json(PROTOCOL);assert scene in protocol['cases']
    dest=OUT/scene
    if dest.exists():raise ValueError('preserve every prior attempt; output already exists')
    dest.mkdir(parents=True)
    rawbase=ROOT/'outputs/stage5_v1'/scene;fb=ROOT/'outputs/stage5_points_v1/frozen'/scene;fusion=ROOT/'outputs/stage5_fusion_v1/frozen'/scene;joint=ROOT/'outputs/stage5_joint_v1/frozen'/scene
    rawpred=read_json(rawbase/'prediction.json');fusedpred=read_json(fusion/'prediction.json')
    native=rawbase/'selected_artifacts/raw_predictions.npz'
    assert sha256(native)==rawpred['artifacts']['pointcloud']['sha256']
    with np.load(native) as z:points=z['world_points']
    flat=points.reshape(-1,3)
    with np.load(fb/'evidence.npz') as z:finite=z['finite'];observed=z['observed']
    with np.load(fb/'filtered.npz') as z:old_ids=z['source_point_indices']
    keep=np.zeros(len(flat),bool);keep[old_ids]=True
    mask=keep_unknown(keep,finite,observed);ids=np.flatnonzero(mask)
    with np.load(fusion/'anchors.npz') as z:anchors={k:z[k] for k in z.files}
    with np.load(fusion/'fused.npz') as z:before=z['xyz']
    assert sha256(fusion/'fused.npz')==read_json(fusion/'run.json')['rows'][0]['sha256']
    source_files=[rawbase/'prediction.json',native,fb/'evidence.npz',fb/'filtered.npz',fusion/'anchors.npz',fusion/'fused.npz',fusion/'prediction.json',joint/'run.json']
    old_joint=read_json(joint/'run.json');problem=None
    if old_joint['problem_sha256']:
        assert sha256(joint/'problem.npz')==old_joint['problem_sha256']
        with np.load(joint/'problem.npz') as z:problem={k:z[k] for k in z.files}
        source_files.append(joint/'problem.npz')
    input_hashes={str(p.relative_to(ROOT)):sha256(p) for p in source_files}
    controls=selection_control(scene);write_json(dest/'selection_control.json',controls)
    rows=[]
    for name in protocol['new_ablations']:
        begin=time.perf_counter();solver=None;extra={}
        if name=='filter_keep_unknown':xyz=flat[ids].copy();indices=ids;sourcepred=rawpred;sourcebase=rawbase
        elif name=='fusion_without_raw_support':
            xyz,beta=without_raw_support(points,anchors,read_json(ROOT/'configs/fusion/stage5_v1.json')['strength']);indices=np.arange(len(flat));extra={'mix_fraction':beta}
            sourcepred=fusedpred;sourcebase=fusion
        else:
            cfg=read_json(ROOT/'configs/optimization/stage5_protocol.json');cfg['fusion_strength']=read_json(ROOT/'configs/fusion/stage5_v1.json')['strength']
            strength=read_json(ROOT/'configs/optimization/stage5_v1.json')['pose_prior_strength']
            solution,solver=(None,{'accepted':False,'status':'no_eligible_correspondences'}) if problem is None else fixed_pose_solve(problem,cfg,strength)
            xyz=scatter(before,problem,solution['points']) if solver['accepted'] else before.copy();indices=np.arange(len(flat));sourcepred=fusedpred;sourcebase=fusion
            if solution is not None:np.savez_compressed(dest/(name+'_attempt.npz'),**solution)
        path=dest/(name+'.npz');np.savez_compressed(path,xyz=xyz,source_point_indices=indices,**extra)
        pred=copy.deepcopy(sourcepred)
        for a in pred['artifacts'].values():
            if a is not None:
                target=dest/a['path']
                if not target.exists():link_or_copy(sourcebase/a['path'],target)
        pred['method_id']=name+'_ablation';pred['config']={'source':sourcepred['config'],'ablation':name,'protocol_sha256':sha256(PROTOCOL),'diagnostic_only':True}
        pred['provenance']['acceptance_ablation']={'code':code_provenance('real'),'gt_access':'none; runtime deny guard','input_hashes':input_hashes,'not_a_new_fitted_rule':True,'all_cameras_unchanged':True}
        pred['artifacts']['pointcloud']={'path':path.name,'sha256':sha256(path),'semantics':protocol['new_ablations'][name]+'; xyz/source_point_indices in raw gauge, float32'}
        elapsed=time.perf_counter()-begin;pred['timing_seconds']={'ablation_body':elapsed};pred['memory_bytes']={'peak_process_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':0}
        validate_result(pred);write_json(dest/(name+'_prediction.json'),pred)
        rows.append({'name':name,'point_count':len(xyz),'artifact_sha256':sha256(path),'prediction_sha256':sha256(dest/(name+'_prediction.json')),'solver':solver,'body_wall_seconds':elapsed})
    write_json(dest/'run.json',{'scene_id':scene,'protocol_sha256':sha256(PROTOCOL),'input_hashes':input_hashes,'rows':rows,'all_cameras_fixed':True,'wall_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'device':'CPU','gpu_allocated_bytes':0})
    print(scene,[(r['name'],r['point_count'],r['solver']['status'] if r['solver'] else None) for r in rows],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-root',type=Path);args=p.parse_args()
    if args.output_root:OUT=args.output_root.resolve()
    from georecon.reliability.analyze import deny_ground_truth
    sys.addaudithook(deny_ground_truth);accesses=[]
    def trace(event,values):
        if event=='open' and isinstance(values[0],str) and values[0].startswith(str(ROOT)):accesses.append(str(Path(values[0]).relative_to(ROOT)))
    sys.addaudithook(trace);run(args.scene)
    write_json(OUT/args.scene/'invocation.json',{'argv':sys.argv,'script_sha256':sha256(Path(__file__)),'project_file_accesses':sorted(set(accesses)),'gt_guard_active':True,'code':code_provenance('real')})
