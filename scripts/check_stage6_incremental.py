"""Independent replay, observation invariants, offline GT evaluation and acceptance."""
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.teaching.incremental import reconstruct_incremental,project
from georecon.geometry.camera import Camera,relative_pose
from georecon.evaluation.poses import rotation_error_deg,translation_direction_error_deg
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_incremental';OUT=ROOT/'outputs/stage6_incremental_v1'
def check():
    lock=read_json(DOC/'input_lock.json');source=read_json(DOC/'source_lock.json');cfg=read_json(ROOT/'configs/teaching/stage6_incremental.json')
    for name,h in source['source_sha256'].items():assert sha256(ROOT/name)==h
    # Verify previously published immutable artifacts, not just their JSON summaries.
    for case in read_json(ROOT/'docs/stage6/input_lock.json')['cases']:
        previous=ROOT/'outputs/stage6_two_view_v1'/case['case_id']
        old=read_json(previous/'prediction.json')
        assert old==read_json(ROOT/'docs/stage6/runs'/case['case_id']/'prediction.json')
        for artifact in old['artifacts'].values():
            if artifact is not None:assert sha256(previous/artifact['path'])==artifact['sha256']
    assert sha256(ROOT/'configs/teaching/stage6_incremental.json')==lock['config_sha256'];rows=[]
    for case in lock['cases']:
        assert sha256(ROOT/case['input'])==case['input_sha256'];assert sha256(ROOT/case['reference'])==case['reference_sha256']
        path=ROOT/case['input'];spec=read_json(path);folder=OUT/case['case_id'];pred=validate_result(read_json(folder/'prediction.json'));steps=read_json(folder/'steps.json');inv=read_json(folder/'invocation.json')
        assert inv['pose_truth_guard_active'] and not any('truth' in p or 'reference.json' in p for p in inv['file_accesses'])
        images=[];ks=[]
        for item in spec['images']:
            assert sha256(path.parent/item['path'])==item['sha256'];images.append(np.array(Image.open(path.parent/item['path']).convert('RGB')));ks.append(np.array(item['K']))
        poses,xyz,obs,resteps,debug,origins=reconstruct_incremental(images,ks,spec['seed'],cfg);assert steps==resteps
        with np.load(folder/'geometry_steps.npz') as z:
            assert set(z.files)==set(debug)
            for key,value in debug.items():np.testing.assert_allclose(z[key],value,rtol=0,atol=0,equal_nan=True)
        with np.load(folder/'sparse_points.npz') as z:
            np.testing.assert_array_equal(z['xyz'],xyz);table=z['observations'];np.testing.assert_array_equal(z['point_creation_view'],origins)
        assert len({(int(v),int(p)) for v,k,p in table})==len(table)
        assert len({(int(v),int(k)) for v,k,p in table})==len(table)
        errors=[]
        for v,k,point in table:
            uv,z=project(xyz[point:point+1],ks[v],*poses[v]);err=float(np.linalg.norm(uv[0]-debug[f'keypoints_{v}'][k]));assert z[0]>0 and err<=cfg['max_reprojection_px']+1e-8;errors.append(err)
        if len(xyz):np.testing.assert_array_equal(xyz[:steps[0]['stats']['kept_points']],debug['seed_accepted_xyz'])
        # Seed replay against previously published two-view arrays, including the preserved blank-view seed.
        oldcase='dtu_scan1_blur_001_003' if 'blur5' in case['case_id'] else ('dtu_scan1_001_020' if 'spaced3' in case['case_id'] else 'dtu_scan1_001_003')
        oldpred=read_json(ROOT/'docs/stage6/runs'/oldcase/'prediction.json')
        if poses:
            for j,v in enumerate(spec['seed']):
                np.testing.assert_array_equal(poses[v][0],oldpred['images'][j]['camera']['R']);np.testing.assert_array_equal(poses[v][1],oldpred['images'][j]['camera']['t'])
        truth=read_json(ROOT/case['reference']);pairs=[]
        for a,b in combinations(sorted(poses),2):
            i,j=[spec['images'][v]['image_id'] for v in [a,b]];ca,cb=truth['cameras'][i],truth['cameras'][j];rg=np.array(cb['R'])@np.array(ca['R']).T;tg=np.array(cb['t'])-rg@ca['t'];ra,ta=poses[a];rb,tb=poses[b];r=rb@ra.T;t=tb-r@ta
            pairs.append({'ids':[i,j],'rotation_deg':rotation_error_deg(r,rg),'translation_direction_deg':translation_direction_error_deg(t,tg)})
        row={'case_id':case['case_id'],'kind':spec['kind'],'status':pred['status'],'registered':len(poses),'input_images':len(images),'points':len(xyz),'pnp_steps':[s for s in steps if s['operation']=='pnp'],'pairs_offline_GT':pairs,'mean_pair_rotation_deg':float(np.mean([p['rotation_deg'] for p in pairs])) if pairs else None,'mean_pair_translation_direction_deg':float(np.mean([p['translation_direction_deg'] for p in pairs])) if pairs else None,'max_observation_reprojection_px':max(errors) if errors else None,'timing_seconds':pred['timing_seconds'],'memory_bytes':pred['memory_bytes']};rows.append(row)
        if spec['kind']=='synthetic':assert pred['status']=='partial' and len(xyz)==404 and len(poses)==2
    write_json(DOC/'evaluation.json',rows)
    write_json(DOC/'acceptance.json',{'scope':'stage6 minimal known-K PnP and incremental registration without BA','engineering_acceptance':'passed','all_5_runs_exactly_replayed':True,'prior_seed_poses_exactly_preserved':True,'all_track_id_uniqueness_and_positive_depth_reprojection_checks_passed':True,'source_input_and_config_hashes_verified':True,'failure_controls_retained':True,'BA':False,'next_stage_started':False,'novelty_or_general_performance_gain_established':False})
    print([(r['case_id'],r['status'],r['registered'],r['points']) for r in rows])
if __name__=='__main__':check()
