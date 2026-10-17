"""Replay cross-fit construction and audit disjoint groups/ranks without changing outputs."""
from pathlib import Path
from collections import Counter
from scipy.spatial import cKDTree
from georecon.geometry.alignment import Similarity
from stage5_points import Evaluator,ply_points
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from georecon.fusion.anchors import predicted_alignment
from heldout_fusion_math import split_groups,construct,witness_score,depth_support
from evaluate_heldout_fusion import profiles
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/heldout_fusion_502';OUT=ROOT/'outputs/heldout_fusion_502_v1'
def check():
    lock=read_json(DOC/'input_lock.json');assert sha256(DOC/'protocol.json')==lock['protocol_sha256']
    for name,h in lock['files_sha256'].items():assert sha256(ROOT/name)==h
    for name,h in read_json(OUT/'source_lock.json').items():assert sha256(ROOT/name)==h
    inv=read_json(OUT/'invocation.json');assert inv['GT_guard_active'];assert not any(any(t in path.lower() for t in lock['generation_forbidden_reads']) for path in inv['project_file_accesses'])
    cfg=read_json(ROOT/'configs/fusion/stage5_protocol.json');rule=read_json(ROOT/'configs/fusion/stage5_v1.json');report=[]
    for scene in read_json(DOC/'protocol.json')['objects']:
        folder=OUT/scene;folds=read_json(folder/'folds.json');positions=read_json(folder/'positions.json');offline=read_json(folder/'offline_position_evaluation.json');old=ROOT/'outputs/stage5_fusion_v1/frozen'/scene
        with np.load(folder/'candidates.npz') as z:data={k:z[k] for k in z.files}
        with np.load(folder/'offline_evaluation.npz') as z:ev={k:z[k] for k in z.files}
        with np.load(old/'anchors.npz') as z:a={k:z[k] for k in z.files}
        np.testing.assert_array_equal(data['source_point_indices'],a['source_point_indices']);np.testing.assert_array_equal(data['point_id'],a['point_id'])
        with np.load(ROOT/'outputs/stage5_v1'/scene/'selected_artifacts/raw_predictions.npz') as z:raw={k:z[k] for k in ['depth','intrinsics','extrinsics']}
        with np.load(ROOT/'data/stage5_fusion_tracks'/scene/'observations.npz') as z:tracks={k:z[k] for k in z.files}
        cm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/input_mapping.json');rm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'vggt_raw/input_mapping.json');bycm={x['image_id']:x for x in cm};byraw={x['image_id']:(i,x) for i,x in enumerate(rm)};col=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/prediction.json');pred=read_json(ROOT/'outputs/stage5_v1'/scene/'prediction.json');cc={}
        for item in col['images']:
            c=item['camera'];cc[item['image_id']]={'K':np.array(bycm[item['image_id']]['pixel_transform'])@c['K'],'R':np.array(c['R']),'t':np.array(c['t'])}
        rc={name:{'K':raw['intrinsics'][i].astype(float),'R':raw['extrinsics'][i,:,:3].astype(float),'t':raw['extrinsics'][i,:,3].astype(float)} for name,(i,_) in byraw.items()};cache={}
        for witness in byraw:
            cp={**col,'images':[i for i in col['images'] if i['image_id']!=witness]};rp={**pred,'images':[i for i in pred['images'] if i['image_id']!=witness]};cache[witness],record=predicted_alignment(cp,rp,cfg['alignment']);assert witness not in record.get('shared_ids',[])
        changed_witness_probe=0
        for fold in folds:
            p=fold['point_index'];pos=positions[p];rows=np.array(pos['global_track_rows']);names=tracks['image_id'][rows];xy=tracks['xy'][rows]-.5;source=pos['source_image'];witness=fold['witness_image'];train,test,possible=split_groups(names,source,witness)
            assert possible==fold['geometrically_feasible'];assert set(fold['train_global_rows']).isdisjoint(fold['witness_global_rows']);assert set(fold['train_global_rows'])|set(fold['witness_global_rows'])==set(rows.tolist());assert all(names[i]!=witness for i in train) and all(names[i]==witness for i in test);assert source!=witness and witness not in fold['alignment_camera_ids'];np.testing.assert_array_equal(rows[train],fold['train_global_rows']);np.testing.assert_array_equal(rows[test],fold['witness_global_rows'])
            uv=[]
            for name,point in zip(names,xy):
                i,item=byraw[str(name)];T=np.array(item['original_to_model_transform'])@np.linalg.inv(np.array(bycm[str(name)]['pixel_transform']));q=T@np.r_[point,1.];uv.append(q[:2]/q[2])
            uv=np.array(uv);v=byraw[witness][0];s=pos['source_frame'];depth=raw['depth'][...,0];support,observed=depth_support(data['raw_xyz'][p],s,v,depth,raw['extrinsics'],raw['intrinsics']);assert support==fold['raw_supported_construction_views']
            got,info=construct(train,names,xy,cc,cache[witness],data['raw_xyz'][p],rc[source],uv[pos['source_local_row']],support,5,rule['strength'],cfg['correspondence']);assert info==fold['construction']
            if got is None:assert np.isnan(data['crossfit_xyz'][fold['fold_index']]).all()
            else:
                np.testing.assert_array_equal(got,data['crossfit_xyz'][fold['fold_index']]);score,reason=witness_score(data['raw_xyz'][p],got,uv[test],rc[witness],518,392);assert score==fold['score'] and reason==fold['witness_unavailable_reason']
                if changed_witness_probe<3:
                    changed=xy.copy();changed[test]+=[200,-100];again,again_info=construct(train,names,changed,cc,cache[witness],data['raw_xyz'][p],rc[source],uv[pos['source_local_row']],support,5,rule['strength'],cfg['correspondence']);np.testing.assert_array_equal(again,got);assert again_info==info;changed_depth=depth.copy();changed_depth[v]=np.nan;again_support,again_observed=depth_support(data['raw_xyz'][p],s,v,changed_depth,raw['extrinsics'],raw['intrinsics']);assert (again_support,again_observed)==(support,observed);changed_witness_probe+=1
        # Independent reference-tree replay checks target indexing, gauge and both density profiles.
        scan=int(scene.split('_')[1].removeprefix('scan'));tree=Evaluator(scan).tree;alignment=next(r for r in read_json(ROOT/'docs/stage5_fusion/comparison.json')['rows'] if r['scene_id']==scene)['evaluation_alignment'];metric_sim=Similarity(alignment['scale'],np.array(alignment['R']),np.array(alignment['t']));np.testing.assert_array_equal(tree.query(metric_sim.apply(data['raw_xyz']),workers=8)[0],ev['raw_distance_mm']);finite=np.isfinite(data['crossfit_xyz']).all(1);np.testing.assert_array_equal(tree.query(metric_sim.apply(data['crossfit_xyz'][finite]),workers=8)[0],ev['fold_distance_mm'][finite]);fulltree=cKDTree(ply_points(ROOT/'data/stage5_point_gt'/f'scan{scan}/stl{scan:03}_total.ply'));np.testing.assert_array_equal(fulltree.query(metric_sim.apply(data['raw_xyz']),workers=8)[0],ev['raw_full_reference_distance_mm']);np.testing.assert_array_equal(fulltree.query(metric_sim.apply(data['crossfit_xyz'][finite]),workers=8)[0],ev['fold_full_reference_distance_mm'][finite])
        # Recompute each position's aggregation and each source ranking/permutation exactly.
        evaluated=[x for x in offline if x.get('mean_heldout_score') is not None]
        for x in evaluated:
            ids=np.array([f['fold_index'] for f in folds if f['point_index']==x['point_index'] and f['score'] is not None]);p=x['point_index'];assert x['scorable_folds']==len(ids);assert x['mean_heldout_score']==float(np.mean([folds[i]['score'] for i in ids]));np.testing.assert_allclose(x['mean_gain_mm'],np.mean(ev['raw_distance_mm'][p]-ev['fold_distance_mm'][ids]),rtol=0,atol=0)
        result=next(r for r in read_json(DOC/'evaluation.json')['scenes'] if r['scene_id']==scene)
        for stratum in result['strata']:
            e=[x for x in evaluated if x['source_image']==stratum['source_image']];profile=profiles([x['mean_heldout_score'] for x in e],[x['mean_gain_mm'] for x in e],np.array([x['harm_fold_fraction'] for x in e]),np.array([x['good_to_bad_fold_fraction'] for x in e]),np.array([x['full_reference_mean_gain_mm'] for x in e]));assert profile==stratum['rank_profiles']
            for p in stratum['source_only_constant_tie_profile']:assert p['selected']==stratum['evaluable_positions']
        report.append({'scene_id':scene,'folds':len(folds),'all_witness_groups_disjoint_from_construction':True,'witness_camera_excluded_from_alignment':True,'copied_candidates_weights_gates_and_scores_replayed_exactly':True,'perturbed_witness_coordinates_leave_construction_unchanged_probes':changed_witness_probe,'source_stratum_profiles_and_200_permutations_replayed':True,'source_only_ties_not_split':True,'both_GT_reference_distance_profiles_replayed_exactly':True,'witness_depth_mutation_does_not_change_support':True})
    for name,h in lock['files_sha256'].items():assert sha256(ROOT/name)==h
    write_json(DOC/'verification.json',{'scenes':report,'original_files_and_fusion_rule_unchanged':True,'GT_guard_and_access_log_verified':True,'scope_is_conditional_not_full_pipeline_independence':True,'passed':True})
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in OUT.rglob('*') if p.is_file()};write_json(DOC/'output_artifact_hashes.json',hashes);print('Heldout construction, split isolation and within-source ranking verified.')
if __name__=='__main__':check()
