"""Immutable retrospective complementarity/harm/identifiability diagnostics; no new method."""
from pathlib import Path
import argparse,json,resource,subprocess,time
import numpy as np
from scipy.spatial import cKDTree
from georecon.data.protocol import read_json,write_json,sha256
from georecon.geometry.alignment import Similarity
from georecon.fusion.anchors import predicted_alignment
from stage5_points import Evaluator,ply_points
from stage5_fusion import mask_for
from fusion_diagnostics_math import candidates,oracle,summary,signal_report,bilinear,auc
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/fusion_diagnostics_502';OUT=ROOT/'outputs/fusion_diagnostics_502_v1'


def project(x,k,e):
    c=np.asarray(x)@e[:,:3].T+e[:,3];q=c@k.T
    return q[:,:2]/q[:,2:3],c[:,2]


def extract(scene,source,anchors,tracks,cm,rm,alignment):
    """Inference-available features only; no reference or oracle is used here."""
    ids=anchors['source_point_indices'];pids=anchors['point_id'];n,h,w,_=source['world_points'].shape;frame=ids//(h*w);raw=source['world_points'].reshape(-1,3)[ids];anchor=anchors['anchor_xyz'];ks=source['intrinsics'];es=source['extrinsics'];cm={x['image_id']:x for x in cm};rmidx={x['image_id']:i for i,x in enumerate(rm)}
    uv=[];length=[];error=[];track_gain=[];parallax=[];delta=[];duplicates=[]
    # Track residuals use measurements already used by COLMAP: not held-out evidence.
    col=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/prediction.json');colcams={i['image_id']:i['camera'] for i in col['images'] if i['registered']}
    for a,(pid,f) in enumerate(zip(pids,frame)):
        rows=np.flatnonzero(tracks['point_id']==pid);image_id=rm[f]['image_id'];src=[r for r in rows if tracks['image_id'][r]==image_id];T=np.array(rm[f]['original_to_model_transform'])@np.linalg.inv(np.array(cm[image_id]['pixel_transform']));matching=[]
        for r in src:
            q=T@np.r_[tracks['xy'][r]-.5,1.];u=q[:2]/q[2];rounded=np.floor(u+.5).astype(int)
            if int(f*h*w+rounded[1]*w+rounded[0])==int(ids[a]):matching.append((r,u))
        assert matching,'saved anchor must match a measured observation in its saved grid cell'
        # Existing associate() stable sort chooses the first same-grid/same-weight/same-ID row.
        r,u=matching[0];duplicates.append({'source_point_index':int(ids[a]),'point_id':int(pid),'track_observations':len(rows),'unique_view_count':len(set(tracks['image_id'][rows].tolist())),'source_view_observations':len(src),'same_grid_observations':len(matching),'selected_original_track_row':int(r)})
        uv.append(u);length.append(tracks['track_length'][r]);error.append(tracks['error'][r]);old=[];new=[];rays=[];ray_views=[]
        for r in rows:
            name=str(tracks['image_id'][r]);v=rmidx[name];T=np.array(rm[v]['original_to_model_transform'])@np.linalg.inv(np.array(cm[name]['pixel_transform']));q=T@np.r_[tracks['xy'][r]-.5,1.];measured=q[:2]/q[2]
            old.append(float(np.linalg.norm(project(raw[a:a+1],ks[v],es[v])[0][0]-measured)));new.append(float(np.linalg.norm(project(anchor[a:a+1],ks[v],es[v])[0][0]-measured)))
            cc=colcams[name];center=-np.array(cc['R']).T@cc['t'];ray=tracks['xyz'][r]-center;rays.append(ray/np.linalg.norm(ray));ray_views.append(name)
        track_gain.append(float(np.mean(old)-np.mean(new)));dots=np.clip(np.array(rays)@np.array(rays).T,-1,1);ia,ib=np.triu_indices(len(rays),1);different=np.array(ray_views)[ia]!=np.array(ray_views)[ib];angles=np.degrees(np.arccos(dots[ia[different],ib[different]]));parallax.append(float(angles.min()))
        z=project(raw[a:a+1],ks[f],es[f])[1][0];delta.append(float(np.linalg.norm(anchor[a]-raw[a])/z))
    uv=np.array(uv);integer=np.c_[ids%w,(ids//w)%h];rounding=np.linalg.norm(uv-integer,axis=1);assert (rounding<=np.sqrt(.5)+1e-7).all()
    with np.load(ROOT/'outputs/stage5_points_v1/frozen'/scene/'evidence.npz') as z:observed=z['observed'][ids]
    sims=[];loo_records=[];config=read_json(ROOT/'configs/fusion/stage5_protocol.json')['alignment']
    rawpred=read_json(ROOT/'outputs/stage5_v1'/scene/'prediction.json')
    for omitted in alignment['shared_ids']:
        a={**col,'images':[i for i in col['images'] if i['image_id']!=omitted]};b={**rawpred,'images':[i for i in rawpred['images'] if i['image_id']!=omitted]};sim,record=predicted_alignment(a,b,config);loo_records.append({'omitted':omitted,**record})
        if sim is not None:sims.append(sim)
    xyz=np.array([tracks['xyz'][np.flatnonzero(tracks['point_id']==p)[0]] for p in pids])
    sensitivity=np.full(len(ids),np.nan)
    if len(sims)==n:
        offsets=np.stack([s.apply(xyz) for s in sims]);distance=np.sqrt(np.mean(np.sum((offsets-anchor[None])**2,axis=2),axis=0));depth=np.array([project(raw[i:i+1],ks[v],es[v])[1][0] for i,v in enumerate(frame)]);sensitivity=distance/depth
    confidence=source['world_points_conf'].reshape(-1)[ids];depthconf=source['depth_conf'].reshape(-1)[ids]
    signals={'negative_raw_support':-anchors['raw_other_view_support'].astype(float),'negative_raw_observed':-observed.astype(float),'negative_raw_confidence':-confidence.astype(float),'negative_depth_confidence':-depthconf.astype(float),'negative_colmap_error':-np.array(error),'track_length':np.array(length,dtype=float),'negative_relative_disagreement':-np.array(delta),'negative_pixel_rounding':-rounding,'track_reprojection_gain':np.array(track_gain),'minimum_parallax':np.array(parallax),'negative_alignment_LOO_sensitivity':-sensitivity,'negative_raw_weight_over_classical':-anchors['raw_weight']/anchors['classical_weight']}
    interpolated,valid=bilinear(source['world_points'],frame,uv)
    return raw,xyz,frame,uv,interpolated,valid,signals,loo_records,duplicates


def run():
    if OUT.exists():raise ValueError('preserve all diagnostic attempts; output exists')
    OUT.mkdir(parents=True);start=time.perf_counter();lock=read_json(DOC/'input_lock.json');protocol=read_json(DOC/'protocol.json');assert sha256(DOC/'protocol.json')==lock['protocol_sha256']
    for name,h in lock['files_sha256'].items():assert sha256(ROOT/name)==h
    names=['scripts/prepare_fusion_diagnostics.py','scripts/diagnose_fusion_502.py','scripts/fusion_diagnostics_math.py','tests/test_fusion_diagnostics.py'];write_json(DOC/'source_lock.json',{'frozen_before_diagnostic_outcomes':True,'source_sha256':{name:sha256(ROOT/name) for name in names},'input_lock_sha256':sha256(DOC/'input_lock.json')})
    comparisons=read_json(ROOT/'docs/stage5_fusion/comparison.json')['rows'];failures=read_json(ROOT/'docs/stage5_fusion/failure_analysis.json')['rows'];report=[];example_rows=[];gpu_before=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    for scan in [1,6]:
        scene=f'dtu_scan{scan}_normal5';folder=ROOT/'outputs/stage5_fusion_v1/frozen'/scene;dest=OUT/scene;dest.mkdir();row=next(r for r in comparisons if r['scene_id']==scene);a=row['evaluation_alignment'];sim=Similarity(a['scale'],np.array(a['R']),np.array(a['t']));ev=Evaluator(scan)
        with np.load(ROOT/'outputs/stage5_v1'/scene/'selected_artifacts/raw_predictions.npz') as z:source={k:z[k] for k in ['world_points','depth','intrinsics','extrinsics','world_points_conf','depth_conf']}
        with np.load(folder/'anchors.npz') as z:anchors={k:z[k] for k in z.files}
        with np.load(ROOT/'data/stage5_fusion_tracks'/scene/'observations.npz') as z:tracks={k:z[k] for k in z.files}
        cm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/input_mapping.json');rm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'vggt_raw/input_mapping.json');runinfo=read_json(folder/'run.json');ids=anchors['source_point_indices'];groups=anchors['point_id'];anchor=anchors['anchor_xyz']
        raw,classical,frame,uv,interpolated,interp_valid,signals,loo,duplicates=extract(scene,source,anchors,tracks,cm,rm,runinfo['alignment']);assert set(signals)==set(protocol['signal_directions'])
        with np.load(folder/'fused.npz') as z:weighted=z['xyz'][ids];beta=z['anchor_fraction']
        with np.load(folder/'unweighted.npz') as z:unweighted=z['xyz'][ids]
        actions=candidates(raw,anchor,beta,protocol['oracle_betas']);np.testing.assert_array_equal(actions[-1],weighted);np.testing.assert_array_equal(actions[2],unweighted)
        distances=np.stack([ev.tree.query(sim.apply(x),workers=8)[0] for x in actions]);before,after=distances[0],distances[-1];gain=before-after;which,best=oracle(distances);f=next(r for r in failures if r['scene_id']==scene);legacy=summary(before,after)
        for key in ['distance_improved','distance_worsened','good_to_bad_2mm','bad_to_good_2mm']:
            expected=legacy[{'distance_improved':'improved','distance_worsened':'worsened'}.get(key,key)];assert expected==f[key]
        np.testing.assert_allclose([legacy['mean_before_mm'],legacy['mean_after_mm']],[f['mean_before_mm'],f['mean_after_mm']],atol=1e-10,rtol=0)
        col=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/prediction.json');truth=read_json(ROOT/'data/stage3_dtu'/f'scan{scan}_normal5/reference.json');gt_sim,gt_record=predicted_alignment(col,truth,{'max_center_rmse_over_raw_center_diameter':float('inf'),'max_camera_rotation_disagreement_deg':float('inf')});assert gt_sim is not None;gt_anchor=gt_sim.apply(classical);old_anchor_gt=sim.apply(anchor);raw_gt=sim.apply(raw)
        aligned_control=raw_gt+beta[:,None]*(gt_anchor-raw_gt)
        controls={'GT_camera_assisted_alignment':ev.tree.query(aligned_control,workers=8)[0]};sub=sim.apply(interpolated[interp_valid]);subweighted=sub+beta[interp_valid,None]*(old_anchor_gt[interp_valid]-sub);both=sub+beta[interp_valid,None]*(gt_anchor[interp_valid]-sub)
        for name,points in [('bilinear_raw_sensitivity',subweighted),('both_controls',both)]:
            d=np.full(len(ids),np.nan);d[interp_valid]=ev.tree.query(points,workers=8)[0];controls[name]=d
        fixed_roi=mask_for(ev,raw_gt);control_summaries={name:{'vs_raw':summary(before,d),'vs_actual_weighted':summary(after,d),'rescued_harmed_points':int(((after>before+1e-9)&(d<after-1e-9)&np.isfinite(d)).sum())} for name,d in controls.items()}
        # Full-reference check is explicitly separate, with the same camera alignment and action set.
        fullgt=ply_points(ROOT/'data/stage5_point_gt'/f'scan{scan}/stl{scan:03}_total.ply');fulltree=cKDTree(fullgt);full_dist=np.stack([fulltree.query(sim.apply(x),workers=8)[0] for x in actions]);full_which,full_best=oracle(full_dist);fullgain=full_dist[0]-full_dist[-1]
        rankings={name:signal_report(score,gain,groups) for name,score in signals.items()};rank_full={name:signal_report(score,fullgain,groups) for name,score in signals.items()}
        strata=[]
        for v in np.unique(frame):
            m=frame==v;strata.append({'frame':int(v),'image_id':rm[v]['image_id'],'positions':int(m.sum()),'unique_landmarks':int(len(np.unique(groups[m]))),'weighted':summary(before,after,m),'signal_AUROC':{name:auc(score[m],gain[m]>1e-9) for name,score in signals.items()}})
        archive={'source_point_indices':ids,'point_id':groups,'source_frame':frame,'observed_uv':uv,'raw_xyz':raw,'aligned_anchor_xyz':anchor,'frozen_fraction':beta,'actions_xyz':actions,'action_distances_200k_mm':distances,'oracle_action':which,'oracle_distance_200k_mm':best,'full_reference_action_distances_mm':full_dist,'full_reference_oracle_action':full_which,'raw_fixed_ROI':fixed_roi,'bilinear_valid':interp_valid,'GT_assisted_anchor_xyz_mm':gt_anchor,'bilinear_raw_xyz':interpolated,**{'control_'+name:d for name,d in controls.items()},**{'signal_'+name:value for name,value in signals.items()}}
        np.savez_compressed(dest/'diagnostics.npz',**archive)
        info={'scene_id':scene,'historical_split_label':'dev' if scan==1 else 'test_previously_inspected','points':len(ids),'unique_classical_landmarks':len(np.unique(groups)),'primary_all_points':legacy,'primary_fixed_raw_ROI':summary(before,after,fixed_roi),'endpoint_classical':summary(before,distances[4]),'oracle':summary(before,best),'oracle_relative_mean_gain':float((before.mean()-best.mean())/before.mean()),'oracle_action_histogram':np.bincount(which,minlength=len(actions)).tolist(),'primary_actions':[{'action':name,**summary(before,d)} for name,d in zip([str(v) for v in protocol['oracle_betas']]+['frozen_weighted'],distances)],'source_view_strata':strata,'source_observation_audit':duplicates,'positions_with_duplicate_source_view':sum(v['source_view_observations']>1 for v in duplicates),'positions_with_track_length_above_unique_views':sum(v['track_observations']>v['unique_view_count'] for v in duplicates),'counterfactual_sensitivity':control_summaries,'alignment_controls':{'original':runinfo['alignment'],'GT_camera_assisted':gt_record,'LOO_predicted_only':loo},'signals':rankings,'full_reference':{'vertices':len(fullgt),'weighted':summary(full_dist[0],full_dist[-1]),'endpoint_classical':summary(full_dist[0],full_dist[4]),'oracle':summary(full_dist[0],full_best),'benefit_label_flips_from_200k':int(((gain>1e-9)!=(fullgain>1e-9)).sum()),'signals':rank_full},'NPZ_sha256':sha256(dest/'diagnostics.npz')}
        write_json(dest/'summary.json',info);report.append(info)
        # Preserve worst harms, best gains, and oracle-missed opportunities with complete point identities.
        indexes=set(np.argsort(gain)[:10].tolist()+np.argsort(gain)[-10:].tolist()+np.argsort((before-best)-gain)[-10:].tolist())
        for i in sorted(indexes):example_rows.append({'scene_id':scene,'source_point_index':int(ids[i]),'point_id':int(groups[i]),'source_frame':int(frame[i]),'raw_error_mm':float(before[i]),'weighted_error_mm':float(after[i]),'classical_endpoint_error_mm':float(distances[4,i]),'oracle_error_mm':float(best[i]),'oracle_action':int(which[i]),'full_reference_raw_mm':float(full_dist[0,i]),'full_reference_weighted_mm':float(full_dist[-1,i]),'fixed_raw_ROI':bool(fixed_roi[i]),'signals':{name:float(score[i]) if np.isfinite(score[i]) else None for name,score in signals.items()},'counterfactual_errors_mm':{name:float(d[i]) if np.isfinite(d[i]) else None for name,d in controls.items()}})
        del source,ev,fulltree,fullgt
    assert sum(r['points'] for r in report)==502
    matched=[]
    for name in protocol['signal_directions']:
        ok=True
        for r in report:
            s=r['signals'][name];a=s['benefit_AUROC']['auc'];b=s['unique_landmark_mean_AUROC']['auc'];profile=next((p for p in s['top_fraction_profiles'] if p['requested_fraction']==.25),None)
            ok &= a is not None and b is not None and a>=.65 and b>=.6 and profile is not None and profile['mean_gain_mm']>0 and profile['mean_gain_mm']>profile['random_same_coverage_expected_mean_gain_mm']
        if ok:matched.append(name)
    complementarity=all(r['oracle_relative_mean_gain']>=.1 and r['endpoint_classical']['improved']/r['points']>=.2 and r['full_reference']['oracle']['mean_gain_mm']>0 for r in report)
    verdict={'complementarity_gate_passed':complementarity,'identifiability_gate_passed':bool(matched),'qualifying_prespecified_signals':matched,'recommendation':'conditional_limited_identifiability_followup_only' if complementarity and matched else ('complementarity_exists_but_current_signals_insufficient_do_not_expand_weighting' if complementarity else 'stop_current_fusion_design_and_investigate_correspondence_or_complementarity_first'),'not_statistical_confirmation':True,'independent_objects':2,'new_deployment_rule':False,'global_F1_gain_or_novelty_established':False}
    write_json(DOC/'results.json',{'scenes':report,'verdict':verdict,'old_six_fallbacks':[{'scene_id':r['scene_id'],'reason':r['reason']} for r in failures if r['fallback']]});write_json(DOC/'failure_examples.json',example_rows)
    for name,h in lock['files_sha256'].items():assert sha256(ROOT/name)==h
    gpu_after=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    write_json(DOC/'execution.json',{'wall_seconds':time.perf_counter()-start,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'CPU_threads':8,'GPU_task_allocated_bytes':0,'GPU_budget_GiB':12,'whole_card_snapshot_before_MiB':gpu_before,'whole_card_snapshot_after_MiB':gpu_after,'snapshot_not_peak':True,'new_weights_models_downloads':False,'input_and_rule_hashes_unchanged_after_analysis':True,'source_kind':'posthoc evaluation reads GT intentionally; never inference or fitting'})
    print(json.dumps(verdict));print([(r['scene_id'],r['primary_all_points'],r['oracle'],r['endpoint_classical']) for r in report])
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,default=OUT);args=parser.parse_args();OUT=args.output_root.resolve();run()
