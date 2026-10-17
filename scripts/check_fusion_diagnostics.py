"""Independent verification of frozen artifacts, NN objectives, oracle and diagnostics."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.spatial import cKDTree
from georecon.data.protocol import read_json,write_json,sha256
from georecon.geometry.alignment import Similarity
from georecon.fusion.anchors import associate
from stage5_points import Evaluator,ply_points
from fusion_diagnostics_math import auc
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/fusion_diagnostics_502';OUT=ROOT/'outputs/fusion_diagnostics_502_v2'
def check():
    lock=read_json(DOC/'input_lock.json');source=read_json(DOC/'source_lock.json');results=read_json(DOC/'results.json');assert sha256(DOC/'protocol.json')==lock['protocol_sha256'];assert sha256(DOC/'input_lock.json')==source['input_lock_sha256']
    for name,h in lock['files_sha256'].items():assert sha256(ROOT/name)==h
    for name,h in source['source_sha256'].items():assert sha256(ROOT/name)==h
    first=DOC/'attempts/attempt01';first_lock=read_json(first/'source_lock.json');assert sha256(first/'diagnose_fusion_502.py')==first_lock['source_sha256']['scripts/diagnose_fusion_502.py'];assert 'AssertionError' in (first/'stderr.txt').read_text()
    # Reuse the unchanged old fusion validator without allowing historical writes.
    spec=importlib.util.spec_from_file_location('old_fusion_check',ROOT/'scripts/check_stage5_fusion.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def compare(path,value):assert read_json(path)==value
    module.write_json=compare;module.check()
    verification=[]
    for r in results['scenes']:
        scene=r['scene_id'];scan=int(scene.split('_')[1].removeprefix('scan'));folder=ROOT/'outputs/stage5_fusion_v1/frozen'/scene;dest=OUT/scene;assert sha256(dest/'diagnostics.npz')==r['NPZ_sha256']
        with np.load(dest/'diagnostics.npz') as z:d={k:z[k] for k in z.files}
        with np.load(folder/'anchors.npz') as z:a={k:z[k] for k in z.files}
        with np.load(ROOT/'outputs/stage5_v1'/scene/'selected_artifacts/raw_predictions.npz') as z:raw={k:z[k] for k in ['world_points','depth','intrinsics','extrinsics']}
        with np.load(ROOT/'data/stage5_fusion_tracks'/scene/'observations.npz') as z:tracks={k:z[k] for k in z.files}
        cm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/input_mapping.json');rm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'vggt_raw/input_mapping.json');alignment=r['alignment_controls']['original'];sim=Similarity(alignment['scale'],np.array(alignment['R']),np.array(alignment['t']));config=read_json(ROOT/'configs/fusion/stage5_protocol.json')['correspondence']
        regenerated=associate(raw['world_points'],raw['depth'],raw['extrinsics'],raw['intrinsics'],tracks,cm,rm,sim,config)
        for k,v in a.items():np.testing.assert_array_equal(v,regenerated[k])
        np.testing.assert_array_equal(d['source_point_indices'],a['source_point_indices']);np.testing.assert_array_equal(d['point_id'],a['point_id']);np.testing.assert_array_equal(d['raw_xyz'],raw['world_points'].reshape(-1,3)[a['source_point_indices']])
        with np.load(folder/'fused.npz') as z:np.testing.assert_array_equal(d['actions_xyz'][-1],z['xyz'][a['source_point_indices']])
        # Full objective replay from local STL, including the published sampled reference.
        ev=Evaluator(scan);e=next(x for x in read_json(ROOT/'docs/stage5_fusion/comparison.json')['rows'] if x['scene_id']==scene)['evaluation_alignment'];eval_sim=Similarity(e['scale'],np.array(e['R']),np.array(e['t']))
        dist=np.array([ev.tree.query(eval_sim.apply(x),workers=8)[0] for x in d['actions_xyz']]);np.testing.assert_array_equal(dist,d['action_distances_200k_mm']);choice=dist.argmin(axis=0);np.testing.assert_array_equal(choice,d['oracle_action']);np.testing.assert_array_equal(dist.min(axis=0),d['oracle_distance_200k_mm']);assert (d['oracle_distance_200k_mm']<=dist[0]).all();assert (d['oracle_distance_200k_mm']<=dist[-1]).all()
        full=ply_points(ROOT/'data/stage5_point_gt'/f'scan{scan}/stl{scan:03}_total.ply');tree=cKDTree(full);fd=np.array([tree.query(eval_sim.apply(x),workers=8)[0] for x in d['actions_xyz']]);np.testing.assert_array_equal(fd,d['full_reference_action_distances_mm']);assert len(full)==r['full_reference']['vertices']
        ids=a['source_point_indices'];n,h,w,_=raw['world_points'].shape;rounding=[]
        for i,record in enumerate(r['source_observation_audit']):
            v=int(d['source_frame'][i]);name=rm[v]['image_id'];mapping=next(c for c in cm if c['image_id']==name);T=np.array(rm[v]['original_to_model_transform'])@np.linalg.inv(np.array(mapping['pixel_transform']));eligible=[]
            for row in np.flatnonzero((tracks['point_id']==d['point_id'][i])&(tracks['image_id']==name)):
                q=T@np.r_[tracks['xy'][row]-.5,1.];uv=q[:2]/q[2];cell=np.floor(uv+.5).astype(int);project=raw['extrinsics'][v,:,:3]@a['anchor_xyz'][i]+raw['extrinsics'][v,:,3];q=raw['intrinsics'][v]@project
                if int(v*h*w+cell[1]*w+cell[0])==int(ids[i]) and np.linalg.norm(q[:2]/q[2]-uv)<=2:eligible.append((int(row),uv))
            assert eligible and record['selected_original_track_row']==eligible[0][0];np.testing.assert_array_equal(d['observed_uv'][i],eligible[0][1])
        # Mann-Whitney rank AUC checked by direct pair counting, preserving ties.
        gain=dist[0]-dist[-1];y=gain>1e-9
        for name,s in r['signals'].items():
            scores=d['signal_'+name];valid=np.isfinite(scores);positive=scores[valid&y];negative=scores[valid&~y]
            expected=float(np.mean((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:]))) if len(positive) and len(negative) else None
            actual=s['benefit_AUROC']['auc'];assert actual==expected or (actual is not None and expected is not None and abs(actual-expected)<1e-12)
        # Auxiliary endpoint and numerical precision controls: not new actions or policies.
        raw_gt=eval_sim.apply(d['raw_xyz']);anchor_gt=eval_sim.apply(d['aligned_anchor_xyz']);beta=d['frozen_fraction'];float64_control=raw_gt+beta[:,None]*(anchor_gt-raw_gt);float_error=ev.tree.query(float64_control,workers=8)[0];gt_anchor_error=ev.tree.query(d['GT_assisted_anchor_xyz_mm'],workers=8)[0]
        verification.append({'scene_id':scene,'original_correspondence_association_replayed_exactly':True,'primary_and_full_reference_distances_replayed_exactly':True,'source_observation_mapping_including_duplicates_matches_old_gate_and_order':True,'oracle_verified_pointwise_not_worse_than_raw_or_frozen':True,'all_AUCs_match_direct_pair_counting':True,'GT_assisted_endpoint_mean_error_mm':float(gt_anchor_error.mean()),'GT_assisted_endpoint_better_than_raw_count':int((gt_anchor_error<dist[0]-1e-9).sum()),'float64_coordinate_control_max_distance_difference_from_original_float32_mm':float(np.abs(float_error-dist[-1]).max()),'float64_coordinate_control_mean_distance_difference_mm':float((float_error-dist[-1]).mean()),'full_reference_points':len(full)})
    for name,h in lock['files_sha256'].items():assert sha256(ROOT/name)==h
    write_json(DOC/'verification.json',{'scenes':verification,'old_rules_inputs_and_outputs_unchanged':True,'old_8_case_fusion_validator_passed_readonly':True,'first_failed_attempt_preserved':True,'production_rule_created_or_changed':False,'GT_is_posthoc_diagnostic_only':True,'checks_passed':True})
    hashes={str(p.relative_to(ROOT)):sha256(p) for root in [OUT,ROOT/'outputs/fusion_diagnostics_502_v1'] for p in root.rglob('*') if p.is_file()};write_json(DOC/'output_artifact_hashes.json',hashes);print('All diagnostics verified; frozen rules and artifacts unchanged.')
if __name__=='__main__':check()
