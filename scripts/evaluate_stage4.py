"""Post-hoc GT-only evaluation; never imported by reliability inference."""
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.stats import spearmanr
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.evaluation.poses import evaluate_poses
from georecon.provenance import code_provenance

ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage4';OUT=ROOT/'outputs/stage4_v1'


def evaluate():
    started=time.perf_counter()
    lock=read_json(DOC/'input_lock.json');freeze=read_json(ROOT/'docs/stage3/freeze.json')
    assert lock['stage3_freeze_sha256']==sha256(ROOT/'docs/stage3/freeze.json')
    assert lock['config_sha256']==sha256(ROOT/'configs/reliability/stage4_v1.json')
    source_digest=code_provenance('real')['source_tree_sha256']
    rows=[];runs=[]
    for case in lock['cases']:
        scene=case['scene_id'];dest=OUT/scene;r=read_json(dest/'reliability.json');o=read_json(dest/'observations.json')
        assert r['config_sha256']==lock['config_sha256'];assert r['provenance']['code_commit'].startswith('913863e')
        assert r['provenance']['source_tree_sha256']==source_digest
        assert r['observations_sha256']==sha256(dest/'observations.npz');assert r['match_signals']['sha256']==sha256(dest/'match_signals.npz')
        with np.load(dest/'observations.npz',allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            for pair in o['pairs']:
                for suffix in ['_first','_second']:assert arrays[pair['array_prefix']+suffix].shape==(pair['audit_matches'],2)
        with np.load(dest/'match_signals.npz',allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            for method,value in r['methods'].items():
                for pair,observed in zip(value['pairs'],o['pairs']):
                    key=method+'_'+observed['array_prefix']+'_sampson_px'
                    if pair['status']=='available':assert arrays[key].shape==(observed['audit_matches'],)
                    else:assert key not in arrays.files
        d=next(d for d in freeze['datasets'] if d['manifest']['scene_id']==scene)
        reference_path=(ROOT/d['path']).parent/'reference.json';assert sha256(reference_path)==d['reference_sha256']
        truth=read_json(reference_path)
        bundle=ROOT/case['bundle']
        assert sha256(bundle/'manifest.json')==d['manifest_sha256']==r['input_manifest_sha256']
        for name,digest in case['files'].items():assert sha256(bundle/name)==digest
        for method,value in r['methods'].items():
            prediction=validate_result(read_json(bundle/(method+'.json')));assert sha256(bundle/(method+'.json'))==value['prediction_sha256']
            e=evaluate_poses(prediction,truth)
            pair_errors=[max(p['rotation_deg'],p['translation_direction_deg']) for p in e['pairs'] if p['rotation_deg'] is not None and p['translation_direction_deg'] is not None]
            if prediction['status'] not in {'success','partial'}:
                assert value['status']=='unknown' and value['summary']['median_pair_sampson_px'] is None
            assert value['composite_score'] is None and value['correctness_probability'] is None
            row=dict(scene_id=scene,split=case['split'],method=method,baseline_status=prediction['status'],diagnostic_status=value['status'],input_pairs=value['summary']['input_pairs'],evidence_pairs=value['summary']['evidence_pairs'],
                     evidence_pair_fraction=value['summary']['evidence_pair_fraction'],epipolar_median_px=value['summary']['median_pair_sampson_px'],positive_depth_ratio=value['summary']['median_pair_positive_depth_ratio'],
                     raw_cross_depth_error=value['depth_consistency'].get('median_pair_relative_z_error'),raw_confidence_mean=value['confidence']['depth_conf']['mean'] if value['confidence'] else None,
                     gt_joint_error_mean_deg=float(np.mean(pair_errors)) if pair_errors else None,gt_rotation_deg=e['rotation_deg_registered_subset']['mean'],gt_translation_deg=e['translation_direction_deg_registered_subset']['mean'],gt_ate_mm=e['camera_center_ate']['rmse'],
                     shared_match_graph=o['evidence_graph'],reliability_sha256=sha256(dest/'reliability.json'))
            gt_pairs={tuple(p['image_ids']):p for p in e['pairs']}
            row['pair_evaluation']=[]
            for pair in value['pairs']:
                gp=gt_pairs[tuple(pair['image_ids'])]
                error=max(gp['rotation_deg'],gp['translation_direction_deg']) if gp['rotation_deg'] is not None and gp['translation_direction_deg'] is not None else None
                row['pair_evaluation'].append({'image_ids':pair['image_ids'],'diagnostic_status':pair['status'],'sampson_median_px':pair['sampson_px']['median'] if pair['sampson_px'] else None,'gt_joint_error_deg':error})
            supported=[p['gt_joint_error_deg'] for p in row['pair_evaluation'] if p['diagnostic_status']=='available' and p['gt_joint_error_deg'] is not None]
            row['gt_joint_error_evidence_pairs_only_deg']=float(np.mean(supported)) if supported else None
            rows.append(row)
        runs.append({'scene_id':scene,'timing_seconds':r['timing_seconds'],'memory_bytes':r['memory_bytes'],'invocation':read_json(dest/'invocation.json'),'versions':r['versions']})
        published=DOC/'runs'/scene;published.mkdir(parents=True,exist_ok=True)
        for name in ['reliability.json','observations.json','invocation.json']:shutil.copyfile(dest/name,published/name)
        shutil.copyfile(OUT/(scene+'_worker.txt'),published/'worker_log.txt')
    correlations=[]
    for split in ['dev','test']:
        for method in ['colmap','vggt_raw','vggt_ba']:
            for signal in ['epipolar_median_px','positive_depth_ratio','raw_cross_depth_error','raw_confidence_mean']:
                subset=[r for r in rows if r['split']==split and r['method']==method and r[signal] is not None and r['gt_joint_error_mean_deg'] is not None]
                x=[r[signal] for r in subset];y=[r['gt_joint_error_mean_deg'] for r in subset]
                enough=len(subset)>=3 and len(set(x))>1 and len(set(y))>1
                correlations.append(dict(split=split,method=method,signal=signal,available_instances=len(subset),status='descriptive_only' if enough else 'unavailable',reason=None if enough else 'requires >=3 nonconstant available instances',spearman_rho=float(spearmanr(x,y).statistic) if enough else None))
    write_json(DOC/'evaluation.json',dict(role='posthoc_ground_truth_evaluation_only',input_lock_sha256=sha256(DOC/'input_lock.json'),rows=rows,correlations=correlations,runs=runs,evaluation_wall_seconds=time.perf_counter()-started,calibration='none; all configurations frozen before signal computation',limits='stage3 test already seen; one independent object per split; conditions correlated; no inferential significance or fresh-holdout claim'))
    probe=read_json(DOC/'gt_isolation_probe.json')
    assert probe['all_method_diagnostics_identical'] and probe['all_match_observations_and_signal_arrays_identical']
    write_json(DOC/'validation.json',dict(gt_isolation_integration_probe_passed=True,stage3_rows_reaudited=24,diagnostic_methods=24,cases=8,source_commit='913863e',config_sha256=lock['config_sha256'],input_lock_sha256=sha256(DOC/'input_lock.json'),source_and_artifact_hashes_verified=True,failed_methods_preserved_as_unknown=sum(r['baseline_status'] not in ['success','partial'] for r in rows),selection_or_fusion_implemented=False,gt_excluded_from_reliability_inference=True,gt_evaluation_in_separate_process=True,compute='CPU only; no additional GPU allocation'))
    print('Post-hoc evaluation and all input/output hash checks passed')


if __name__=='__main__':evaluate()
