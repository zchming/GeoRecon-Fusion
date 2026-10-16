"""Minimal hybrid weighted fusion, dev-only fitting and separate offline evaluation."""
import argparse,copy,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.fusion.anchors import predicted_alignment,associate,blend
from georecon.provenance import code_provenance
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_fusion';OUT=ROOT/'outputs/stage5_fusion_v1'
PROTOCOL=ROOT/'configs/fusion/stage5_protocol.json';RULE=ROOT/'configs/fusion/stage5_v1.json'
CONDITIONS=['blur5','normal5','reduced3','spaced3']


def inputs(scene):
    selected=ROOT/'outputs/stage5_v1'/scene
    decision=read_json(selected/'decision.json');raw=read_json(selected/'prediction.json')
    if decision['selected_method']!='vggt_raw':raise ValueError('not_supported: this fusion requires selected raw dense grid')
    point=selected/raw['artifacts']['pointcloud']['path']
    assert sha256(point)==raw['artifacts']['pointcloud']['sha256']
    colbase=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap'
    rawbase=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'vggt_raw'
    col=read_json(colbase/'prediction.json')
    a=read_json(colbase/'input_mapping.json');b=read_json(rawbase/'input_mapping.json')
    assert [v['image_id'] for v in b]==[i['image_id'] for i in raw['images']]
    export=read_json(ROOT/'data/stage5_fusion_tracks'/scene/'export.json')
    for name,digest in export['files_sha256'].items():assert sha256(ROOT/name)==digest
    obs=ROOT/'data/stage5_fusion_tracks'/scene/'observations.npz'
    assert sha256(obs)==export['artifact_sha256']
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in [selected/'prediction.json',selected/'decision.json',point,colbase/'prediction.json',colbase/'input_mapping.json',rawbase/'input_mapping.json',obs]}
    return point,raw,col,a,b,obs,hashes,export


def generate(scene,dev=False):
    start=time.perf_counter();protocol=read_json(PROTOCOL)
    # freeze.json is evaluation metadata; runtime checks its pinned digest through the published protocol, never reads it.
    p,raw,col,cm,rm,obs,hashes,export=inputs(scene)
    dest=OUT/('dev_candidates' if dev else 'frozen')/scene
    if dest.exists():raise ValueError('output exists; preserve every attempt')
    dest.mkdir(parents=True)
    rule=None if dev else read_json(RULE)
    if rule is not None:assert rule['protocol_sha256']==sha256(PROTOCOL)
    sim,alignment=predicted_alignment(col,raw,protocol['alignment'])
    with np.load(p) as z:
        points=z['world_points']
        if sim is not None:
            with np.load(obs) as tracks:
                anchors=associate(points,z['depth'],z['extrinsics'],z['intrinsics'],tracks,cm,rm,sim,protocol['correspondence'])
        else:
            anchors={'source_point_indices':np.array([],dtype=np.int64),'anchor_xyz':np.empty((0,3)),'point_id':np.array([],dtype=np.int64),
                     'classical_weight':np.array([]),'raw_weight':np.array([]),'raw_other_view_support':np.array([],dtype=np.uint8),'counts':{'eligible_grid_points':0}}
    anchor_arrays={k:v for k,v in anchors.items() if k!='counts'}
    np.savez_compressed(dest/'anchors.npz',**anchor_arrays)
    strengths=protocol['candidate_strengths'] if dev else [rule['strength']]
    rows=[]
    for i,alpha in enumerate(strengths):
        xyz,fraction=blend(points,anchors,alpha)
        name=f'candidate_{i}' if dev else 'fused'
        np.savez_compressed(dest/(name+'.npz'),xyz=xyz,source_point_indices=np.arange(len(xyz),dtype=np.int64),anchor_fraction=fraction)
        delta=np.linalg.norm(xyz-points.reshape(-1,3),axis=1)
        rows.append({'name':name,'strength':alpha,'sha256':sha256(dest/(name+'.npz')),'point_count':len(xyz),'eligible_points':len(fraction),'moved_points':int((delta>0).sum()),'fraction_min':float(fraction.min()) if len(fraction) else None,'fraction_max':float(fraction.max()) if len(fraction) else None,'max_displacement_raw_units':float(delta.max())})
    unweighted,_=blend(points,anchors,None)
    np.savez_compressed(dest/'unweighted.npz',xyz=unweighted,source_point_indices=np.arange(len(unweighted),dtype=np.int64))
    if not dev:
        from georecon.selection.run import link_or_copy
        output=copy.deepcopy(raw)
        for a in output['artifacts'].values():
            if a is not None:
                target=dest/a['path']
                if not target.exists():link_or_copy(ROOT/'outputs/stage5_v1'/scene/a['path'],target)
        output['method_id']='weighted_sparse_anchor_fusion_v1'
        output['config']={'source_selection':raw['config'],'weighted_fusion':rule}
        output['provenance']['weighted_fusion']={'code':code_provenance('real'),'protocol_sha256':sha256(PROTOCOL),'rule_sha256':sha256(RULE),
                'input_hashes':hashes,'alignment':alignment,'gt_access':'none; deny guard active','camera_and_world_frame':'raw camera gauge unchanged; only eligible point XYZ corrected','dense_depth_and_confidence':'unfused source diagnostics; not derived from fused XYZ'}
        output['artifacts']['pointcloud']={'path':'fused.npz','sha256':rows[0]['sha256'],'semantics':'raw gauge full grid flattened C order frame/y/x; weighted sparse-anchor corrected xyz; source_point_indices maps original grid','arrays':{'xyz':[len(unweighted),3],'source_point_indices':[len(unweighted)]}}
        output['timing_seconds']={'incremental_fusion':time.perf_counter()-start}
        output['memory_bytes']={'peak_process_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':0}
        validate_result(output);write_json(dest/'prediction.json',output)
    write_json(dest/'run.json',{'scene_id':scene,'source_shape':list(points.shape),'input_hashes':hashes,'alignment':alignment,'correspondence_counts':anchors['counts'],'rows':rows,
        'unweighted_sha256':sha256(dest/'unweighted.npz'),'anchors_sha256':sha256(dest/'anchors.npz'),'protocol_sha256':sha256(PROTOCOL),'rule_sha256':None if dev else sha256(RULE),
        'fallback':sim is None or len(anchors['source_point_indices'])==0,'fallback_reason':alignment.get('reason','no_eligible_correspondences') if sim is None or len(anchors['source_point_indices'])==0 else None,
        'point_deletion':False,'cameras_unchanged':True,'device':'CPU','gpu_allocated_bytes':0,'wall_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'track_export':export})
    print(scene,alignment['status'],anchors['counts'],flush=True)


def mask_for(ev,points):
    bb=ev.mask['BB'];res=float(ev.mask['Res'].item());obs=ev.mask['ObsMask']
    q=(points-bb[0])/res;safe=np.where(np.isfinite(q),q,-1e9)
    idx=(np.sign(safe)*np.floor(np.abs(safe)+.5)).astype(np.int64)
    inside=(idx>=0).all(1)&(idx<obs.shape).all(1)&np.isfinite(points).all(1)
    result=np.zeros(len(points),bool);result[inside]=obs[tuple(idx[inside].T)].astype(bool)
    return result


def measure(ev,path):
    from georecon.geometry.alignment import Similarity
    old_xyz=ev.xyz.copy();old_mask=ev.in_mask.copy();alignment=ev.alignment
    sim=Similarity(alignment['scale'],np.array(alignment['R']),np.array(alignment['t']))
    with np.load(path) as z:xyz=z['xyz'][ev.ids]
    ev.xyz=sim.apply(xyz)
    fixed=ev.measure(np.ones(ev.n,bool))
    ev.in_mask=mask_for(ev,ev.xyz);current=ev.measure(np.ones(ev.n,bool))
    current['fixed_raw_roi_sensitivity']=fixed
    current['sample_points_changed_roi_membership']=int((old_mask!=ev.in_mask).sum())
    ev.xyz=old_xyz;ev.in_mask=old_mask
    return current


def fit():
    from fit_stage5 import dev_only_guard
    accesses=[]
    def guard(event,args):
        dev_only_guard(event,args)
        if event=='open' and isinstance(args[0],str) and args[0].startswith(str(ROOT)):accesses.append(str(Path(args[0]).relative_to(ROOT)))
    sys.addaudithook(guard)
    from stage5_points import Evaluator
    if RULE.exists():raise ValueError('rule already frozen; no refitting')
    protocol=read_json(PROTOCOL);ev=Evaluator(1);rows=[]
    for cond in CONDITIONS:
        scene='dtu_scan1_'+cond;ev.prepare(scene);dest=OUT/'dev_candidates'/scene
        run=read_json(dest/'run.json');assert run['protocol_sha256']==sha256(PROTOCOL)
        row={'scene_id':scene,'raw':ev.measure(np.ones(ev.n,bool)),'candidates':[],'unweighted':measure(ev,dest/'unweighted.npz'),'alignment':ev.alignment}
        for candidate in run['rows']:
            path=dest/(candidate['name']+'.npz');assert sha256(path)==candidate['sha256']
            row['candidates'].append(measure(ev,path))
        rows.append(row)
    scores=[float(np.mean([r['candidates'][i]['thresholds_mm']['2']['f1'] for r in rows])) for i in range(len(protocol['candidate_strengths']))]
    best=int(np.argmax(scores));baseline=float(np.mean([r['raw']['thresholds_mm']['2']['f1'] for r in rows]))
    rule={'schema_version':1,'strength':protocol['candidate_strengths'][best],'candidate_index':best,'protocol_sha256':sha256(PROTOCOL),'fit_split':'dev_only','dev_beats_raw':scores[best]>baseline,'default_enable_recommended':False,'deployment':'explicit experimental tool; dev gain alone is insufficient for default enable','dev_gain_f1_2mm':scores[best]-baseline}
    write_json(RULE,rule)
    write_json(DOC/'fit_report.json',{'rows':rows,'macro_dev_f1_2mm':scores,'raw_macro_f1_2mm':baseline,'winner':best,'project_file_accesses':sorted(set(accesses)),'test_access_denied_by_audit_hook':True})
    print(rule,scores,flush=True)


def evaluate():
    from stage5_points import Evaluator
    rule=read_json(RULE);rows=[];start=time.perf_counter()
    old=read_json(ROOT/'docs/stage5_points/comparison.json')
    for scan in [1,6]:
        ev=Evaluator(scan)
        for condition in CONDITIONS:
            scene=f'dtu_scan{scan}_{condition}';ev.prepare(scene);dest=OUT/'frozen'/scene
            run=read_json(dest/'run.json');assert run['rule_sha256']==sha256(RULE)
            assert sha256(dest/'fused.npz')==run['rows'][0]['sha256']
            raw=ev.measure(np.ones(ev.n,bool));previous=next(r for r in old['rows'] if r['scene_id']==scene)
            assert raw==previous['before']
            row={'scene_id':scene,'split':'dev' if scan==1 else 'test','raw':raw,'prior_filtered':previous['after'],'weighted_fused':measure(ev,dest/'fused.npz'),
                'unweighted':measure(ev,dest/'unweighted.npz'),'evaluation_alignment':ev.alignment,'run':run}
            rows.append(row)
            print(scene,'F1@2',raw['thresholds_mm']['2']['f1'],row['weighted_fused']['thresholds_mm']['2']['f1'],flush=True)
    write_json(DOC/'comparison.json',{'rows':rows,'rule_sha256':sha256(RULE),'protocol_sha256':sha256(PROTOCOL),'test_retuning':False,'offline_evaluation_wall_seconds':time.perf_counter()-start,'peak_evaluator_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','fit','evaluate']);p.add_argument('--scene');p.add_argument('--dev',action='store_true');p.add_argument('--output-root',type=Path);args=p.parse_args()
    if args.output_root is not None:OUT=args.output_root.resolve()
    if args.action=='generate':
        from georecon.reliability.analyze import deny_ground_truth
        if args.dev:
            from fit_stage5 import dev_only_guard
            sys.addaudithook(dev_only_guard)
        sys.addaudithook(deny_ground_truth);accesses=[]
        def trace(event,values):
            if event=='open' and isinstance(values[0],str) and values[0].startswith(str(ROOT)):accesses.append(str(Path(values[0]).relative_to(ROOT)))
        sys.addaudithook(trace)
        generate(args.scene,args.dev)
        dest=OUT/('dev_candidates' if args.dev else 'frozen')/args.scene
        write_json(dest/'invocation.json',{'argv':sys.argv,'project_file_accesses':sorted(set(accesses)),'gt_guard_active':True,'script_sha256':sha256(Path(__file__)),'code':code_provenance('real')})
    elif args.action=='fit':fit()
    else:evaluate()
