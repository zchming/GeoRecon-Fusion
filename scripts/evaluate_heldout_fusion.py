"""Offline GT evaluation and within-source ranking. No candidate changes or fitted policy."""
from pathlib import Path
from collections import Counter
import resource,time
import numpy as np
from scipy.spatial import cKDTree
from georecon.data.protocol import read_json,write_json,sha256
from georecon.geometry.alignment import Similarity
from stage5_points import Evaluator,ply_points
from fusion_diagnostics_math import auc
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/heldout_fusion_502';OUT=ROOT/'outputs/heldout_fusion_502_v1'


def profiles(score,gain,harm,goodbad,fullgain,seed=42):
    score=np.asarray(score,float);gain=np.asarray(gain,float);valid=np.isfinite(score)&np.isfinite(gain);ids=np.flatnonzero(valid);result=[]
    if not len(ids):return result
    rng=np.random.default_rng(seed)
    for fraction in [.25,.5,1.]:
        n=max(1,int(np.ceil(fraction*len(ids))));threshold=np.sort(score[ids])[::-1][n-1];selected=ids[score[ids]>=threshold];permuted=[]
        for _ in range(200):
            order=rng.permutation(score[ids]);chosen=ids[order>=threshold];permuted.append([float(gain[chosen].mean()),float(harm[chosen].mean()),float(goodbad[chosen].mean())])
        a=np.array(permuted)
        result.append({'requested_fraction':fraction,'selected':len(selected),'evaluable':len(ids),'actual_fraction':len(selected)/len(ids),'score_threshold_descriptive_only':float(threshold),'mean_gain_mm':float(gain[selected].mean()),'harm_rate':float(harm[selected].mean()),'good_to_bad_2mm_rate':float(goodbad[selected].mean()),'full_reference_mean_gain_mm':float(fullgain[selected].mean()),'random_same_count_expected_gain_mm':float(gain[ids].mean()),'random_same_count_expected_harm_rate':float(harm[ids].mean()),'random_same_count_expected_good_to_bad_rate':float(goodbad[ids].mean()),'within_stratum_permutation_200':{'gain_mean_mm':float(a[:,0].mean()),'gain_5_95_percentile_mm':np.quantile(a[:,0],[.05,.95]).tolist(),'note':'shuffle variability, not object-level confidence interval or significance test'}})
    return result


def run():
    if (DOC/'evaluation.json').exists():raise ValueError('preserve previous evaluation')
    start=time.perf_counter();lock=read_json(DOC/'input_lock.json')
    for name,h in lock['files_sha256'].items():assert sha256(ROOT/name)==h
    rows=[];examples=[]
    comparison=read_json(ROOT/'docs/stage5_fusion/comparison.json')['rows']
    for scene in read_json(DOC/'protocol.json')['objects']:
        folder=OUT/scene;folds=read_json(folder/'folds.json');positions=read_json(folder/'positions.json');scan=int(scene.split('_')[1].removeprefix('scan'));ev=Evaluator(scan);a=next(r for r in comparison if r['scene_id']==scene)['evaluation_alignment'];sim=Similarity(a['scale'],np.array(a['R']),np.array(a['t']))
        with np.load(folder/'candidates.npz') as z:data={k:z[k] for k in z.files}
        full=cKDTree(ply_points(ROOT/'data/stage5_point_gt'/f'scan{scan}/stl{scan:03}_total.ply'));raw=sim.apply(data['raw_xyz']);d0=ev.tree.query(raw,workers=8)[0];f0=full.query(raw,workers=8)[0]
        scores=np.array([np.nan if f['score'] is None else f['score'] for f in folds]);d=np.full(len(folds),np.nan);fd=d.copy();valid_xyz=np.isfinite(data['crossfit_xyz']).all(1);d[valid_xyz]=ev.tree.query(sim.apply(data['crossfit_xyz'][valid_xyz]),workers=8)[0];fd[valid_xyz]=full.query(sim.apply(data['crossfit_xyz'][valid_xyz]),workers=8)[0]
        full_d=np.full(len(folds),np.nan);q=np.isfinite(data['full_DLT_xyz']).all(1);full_d[q]=ev.tree.query(sim.apply(data['full_DLT_xyz'][q]),workers=8)[0];original_d=ev.tree.query(sim.apply(data['original_frozen_xyz']),workers=8)[0]
        aggregates=[];score_points=[];gain_points=[];harm_points=[];goodbad_points=[];fullgain_points=[];raw_error=[];full_score=[];original_score=[];source=[]
        for p,pos in enumerate(positions):
            allfolds=np.array([i for i,f in enumerate(folds) if f['point_index']==p]);kept=allfolds[np.isfinite(scores[allfolds])&np.isfinite(d[allfolds])];record={**pos,'requested_folds':len(allfolds),'scorable_folds':len(kept),'construction_failure_reasons':dict(Counter(folds[i]['construction']['reason'] for i in allfolds if folds[i]['construction']['reason'] is not None)),'raw_error_mm':float(d0[p]),'original_frozen_error_mm':float(original_d[p])}
            if len(kept):
                score=float(scores[kept].mean());gain=float((d0[p]-d[kept]).mean());harm=float((d[kept]>d0[p]+1e-9).mean());goodbad=float(((d0[p]<=2)&(d[kept]>2)).mean());fg=float((f0[p]-fd[kept]).mean());fc=[folds[i]['full_DLT_score'] for i in kept if folds[i]['full_DLT_score'] is not None];oc=[folds[i]['original_frozen_score'] for i in kept if folds[i]['original_frozen_score'] is not None]
                record.update(mean_heldout_score=score,mean_candidate_error_mm=float(d[kept].mean()),mean_gain_mm=gain,harm_fold_fraction=harm,good_to_bad_fold_fraction=goodbad,full_reference_mean_gain_mm=fg,full_DLT_score_on_same_valid_folds=float(np.mean(fc)) if len(fc)==len(kept) else None,original_frozen_score_on_same_valid_folds=float(np.mean(oc)) if len(oc)==len(kept) else None)
                score_points.append(score);gain_points.append(gain);harm_points.append(harm);goodbad_points.append(goodbad);fullgain_points.append(fg);raw_error.append(float(d0[p]));full_score.append(record['full_DLT_score_on_same_valid_folds']);original_score.append(record['original_frozen_score_on_same_valid_folds']);source.append(pos['source_image'])
            aggregates.append(record)
        score_points=np.array(score_points);gain_points=np.array(gain_points);harm_points=np.array(harm_points);goodbad_points=np.array(goodbad_points);fullgain_points=np.array(fullgain_points);source=np.array(source);raw_error=np.array(raw_error);strata=[]
        for name in sorted({p['source_image'] for p in positions}):
            mask=source==name;g=gain_points[mask];s=score_points[mask];positive=int((g>1e-9).sum());negative=int((g< -1e-9).sum());enough=len(g)>=10 and positive>=5 and negative>=5
            item={'source_image':name,'original_positions':sum(p['source_image']==name for p in positions),'evaluable_positions':len(g),'benefit_positions':positive,'harm_positions':negative,'interpretable_AUROC':enough,'AUROC':auc(s,g>1e-9),'mean_gain_mm':float(g.mean()) if len(g) else None,'mean_harm_fold_fraction':float(harm_points[mask].mean()) if len(g) else None,'rank_profiles':profiles(s,g,harm_points[mask],goodbad_points[mask],fullgain_points[mask]),'source_only_constant_tie_profile':profiles(np.zeros(len(g)),g,harm_points[mask],goodbad_points[mask],fullgain_points[mask])}
            for kind,control_score,control_gain in [('full_DLT_resubstitution',full_score,None),('original_frozen_resubstitution',original_score,None)]:
                values=np.array([np.nan if x is None else x for x in control_score]);local=values[mask]
                if kind=='full_DLT_resubstitution':cg=np.array([float(d0[p['point_index']]-full_d[next(i for i,f in enumerate(folds) if f['point_index']==p['point_index'])]) if p.get('mean_heldout_score') is not None else np.nan for p in aggregates if p.get('mean_heldout_score') is not None])[mask]
                else:cg=np.array([p['raw_error_mm']-p['original_frozen_error_mm'] for p in aggregates if p.get('mean_heldout_score') is not None])[mask]
                common=np.isfinite(local)&np.isfinite(cg);item[kind]={'same_evaluable_positions':int(common.sum()),'AUROC_for_its_own_candidate_gain':auc(local[common],cg[common]>1e-9),'AUROC_for_crossfit_candidate_gain_same_points':auc(local[common],g[common]>1e-9),'warning':'fit-in score, not genuinely held out; candidate differs from crossfit'}
            strata.append(item)
        source_only=np.array([score_points[source==name].mean() for name in source]);row={'scene_id':scene,'positions':len(positions),'requested_folds':len(folds),'geometrically_feasible_folds':sum(f['geometrically_feasible'] for f in folds),'constructed_folds':int(valid_xyz.sum()),'scorable_folds':int(np.isfinite(scores).sum()),'evaluable_positions':len(score_points),'positions_with_no_evaluable_fold':len(positions)-len(score_points),'all_requested_folds_scorable_positions':sum(p['requested_folds']==p['scorable_folds'] for p in aggregates),'fold_construction_failures':dict(Counter(f['construction']['reason'] for f in folds if f['construction']['reason'] is not None)),'witness_unavailable_reasons':dict(Counter(f['witness_unavailable_reason'] for f in folds if f['witness_unavailable_reason'] is not None)),'overall_AUROC_descriptive_only':auc(score_points,gain_points>1e-9),'source_only_mean_score_overall_AUROC':auc(source_only,gain_points>1e-9),'strata':strata,'mean_point_gain_mm':float(gain_points.mean()) if len(gain_points) else None,'eligible_raw_mean_error_mm':float(raw_error.mean()) if len(raw_error) else None,'ineligible_raw_mean_error_mm':float(d0[[p['scorable_folds']==0 for p in aggregates]].mean()) if len(score_points)<len(positions) else None,'GT_reference_label_flips_at_position_mean':int(((gain_points>1e-9)!=(fullgain_points>1e-9)).sum())}
        write_json(folder/'offline_position_evaluation.json',aggregates);np.savez_compressed(folder/'offline_evaluation.npz',fold_distance_mm=d,fold_full_reference_distance_mm=fd,raw_distance_mm=d0,raw_full_reference_distance_mm=f0,full_DLT_distance_mm=full_d,original_frozen_distance_mm=original_d)
        # Preserve no-evidence examples separately from largest observed harms/benefits.
        candidates=[p for p in aggregates if p.get('mean_gain_mm') is not None];worst=sorted(candidates,key=lambda p:p['mean_gain_mm'])[:10];best=sorted(candidates,key=lambda p:p['mean_gain_mm'])[-10:];unknown=[p for p in aggregates if p['scorable_folds']==0][:10]
        examples.extend({'scene_id':scene,'category':kind,**p} for kind,items in [('lowest_gain',worst),('highest_gain',best),('unavailable',unknown)] for p in items);rows.append(row)
    sufficiently_balanced=all(sum(s['interpretable_AUROC'] for s in r['strata'])>=2 for r in rows);checks=[]
    for r in rows:
        for s in r['strata']:
            if not s['interpretable_AUROC']:continue
            for p in s['rank_profiles']:
                if p['requested_fraction']==1:continue
                checks.append(p['mean_gain_mm']>0 and p['mean_gain_mm']>p['random_same_count_expected_gain_mm'] and p['harm_rate']<=p['random_same_count_expected_harm_rate'] and p['good_to_bad_2mm_rate']<=p['random_same_count_expected_good_to_bad_rate'] and p['full_reference_mean_gain_mm']>0)
    decision={'enough_balanced_source_strata_in_each_object':sufficiently_balanced,'all_interpretable_strata_25_and_50_percent_checks_pass':bool(checks) and all(checks),'recommend_limited_followup':sufficiently_balanced and bool(checks) and all(checks),'not_statistical_confirmation':True,'no_new_deployment_rule':True,'conditional_on_frozen_full_input_cameras_and_original_selection':True}
    write_json(DOC/'evaluation.json',{'scenes':rows,'decision':decision});write_json(DOC/'failure_examples.json',examples);write_json(DOC/'evaluation_execution.json',{'wall_seconds':time.perf_counter()-start,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_task_allocated_bytes':0,'scope':'offline GT scoring and within-source permutations, not model inference'})
    print([(r['scene_id'],r['evaluable_positions'],[(s['source_image'],s['evaluable_positions'],s['benefit_positions'],s['harm_positions'],s['AUROC']['auc']) for s in r['strata']]) for r in rows]);print(decision)
if __name__=='__main__':run()
