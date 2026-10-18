"""Separate offline Stage7A evaluator; never feeds GT back to the workers."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from ..geometry.camera import Camera
from .session import digest, load_protocol, write
from .metrics import support_metrics, pose_metrics, summarize, decision


def truth_for_case(root, protocol, case):
    condition = case['conditions'][0]
    lock = next(v for v in protocol['input']['input_locks'] if v['scene_id'] == f"dtu_scan{case['scan']}_{condition}")
    value = json.loads((root/lock['reference_evaluation_only_except_K']).read_text())
    mapping = json.loads((root/lock['common_input_mapping']).read_text())
    transform = {v['image_id']: np.array(v['pixel_transform']) for v in mapping}
    cameras = {v['image_id']: v['camera'] for v in value['images']}
    result = []
    for image_id in case['view_ids']:
        camera = cameras[image_id]
        result.append(Camera(transform[image_id] @ np.array(camera['K']), np.array(camera['R']), np.array(camera['t']), 800, 600))
    return result


def export_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_session(root, out, review_out=None):
    start = time.perf_counter()
    target = Path(review_out) if review_out is not None else out
    target.mkdir(parents=True, exist_ok=True)
    if (target/'evaluation.json').exists():
        raise ValueError('evaluation already exists; do not overwrite published evidence')
    protocol = load_protocol(root)
    bundle = json.loads((out/'bundle.json').read_text())
    implementation = json.loads((out/'implementation.json').read_text())
    session = json.loads((out/'session.json').read_text())
    source_drift = []
    for name, expected in implementation['source_sha256'].items():
        if digest(out/'sources'/name) != expected:
            raise ValueError(f'archived inference source bytes changed: {name}')
        if digest(root/name) != expected:
            source_drift.append(name)
    if source_drift and session['complete']:
        raise ValueError(f'implementation bytes changed since completed inference: {source_drift}')
    for name, expected in json.loads((out/'input_hashes.json').read_text()).items():
        if digest(root/name) != expected:
            raise ValueError(f'input bytes changed: {name}')
    if any(set(v) != {'path', 'sha256', 'K', 'width', 'height'} for v in bundle['images'].values()):
        raise ValueError('unexpected inference image metadata')
    rows, pairs, records, resource = [], [], [], list(session['resource_failures'])
    verified_cache_cases = []
    cases, seeds = protocol['input']['cases'], protocol['input']['geometry_seeds']
    for case in cases:
        truth = truth_for_case(root, protocol, case)
        root_inputs = {}
        for arm in 'ABC':
            folder = out/'main'/case['case_id']/arm
            record = json.loads((folder/'record.json').read_text()) if (folder/'record.json').exists() else None
            if record:
                if not record['gt_guard'] or any(any(w in p.lower() for w in ['reference.json', 'evaluation.json', 'ground_truth']) for p in record['file_accesses']):
                    raise ValueError('GT inference boundary failed')
                records.append(record)
                if arm in 'BC' and record.get('root_input_digests'):
                    root_inputs[arm] = record['root_input_digests']
            support, good = {'status': 'unavailable', 'reason': 'missing_match_artifact'}, None
            match_arrays = None
            if record and record.get('matches_sha256'):
                if digest(folder/'matches.npz') != record['matches_sha256']:
                    raise ValueError('match arrays changed')
                with np.load(folder/'matches.npz', allow_pickle=False) as data:
                    match_arrays = {k: data[k] for k in data.files}
                support, good = support_metrics(*truth, match_arrays['pixels_first'], match_arrays['pixels_second'],
                    match_arrays['indices'], record['keypoint_counts'])
                evaluation_folder = out/'offline_support'/case['case_id']
                evaluation_folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(evaluation_folder/f'{arm}.npz', gt_epipolar_consistent=good)
            arm_rows = []
            for seed in seeds:
                geometry = next((g for g in record['geometry'] if g['seed'] == seed), None) if record else None
                arrays, pose = {}, pose_metrics(*truth, None, None, False)
                if geometry:
                    path = folder/f'geometry_{seed}.npz'
                    if digest(path) != geometry['arrays_sha256']:
                        raise ValueError('geometry arrays changed')
                    with np.load(path, allow_pickle=False) as data:
                        arrays = {k: data[k] for k in data.files}
                    pose = pose_metrics(*truth, arrays.get('accepted_R'), arrays.get('accepted_t'), geometry['status'] == 'success')
                e_available = 'essential_candidates' in arrays
                h_available = 'H' in arrays
                stats = geometry['statistics'] if geometry else {}
                row = {'case_id': case['case_id'], 'scan': case['scan'], 'conditions': case['conditions'], 'arm': arm, 'seed': seed,
                    'inference_status': geometry['status'] if geometry else record['status'] if record else 'not_run',
                    'reason': geometry['reason'] if geometry else record.get('error') if record else 'no_worker_record',
                    'support': support, 'pose': pose, 'essential_inliers': stats.get('essential_inliers') if e_available else None,
                    'homography_inliers': stats.get('homography_inliers') if h_available else None,
                    'E_gt_consistent': int(good[arrays['essential_mask']].sum()) if e_available and good is not None else None,
                    'H_gt_consistent': int(good[arrays['homography_mask']].sum()) if h_available and good is not None else None,
                    'filtered_points': stats.get('kept_points'), 'accepted_points': stats.get('kept_points', 0) if pose['accepted'] else 0,
                    'accepted_gt_consistent': int(good[arrays['kept_match_indices']].sum()) if pose['accepted'] and good is not None else 0,
                    'positive_depth_ratio': stats.get('positive_fraction'), 'positive_median_parallax_deg': stats.get('positive_median_parallax_deg'),
                    'accepted_median_parallax_deg': stats.get('accepted_median_parallax_deg'), 'accepted_max_reprojection_px': stats.get('accepted_max_reprojection_px'),
                    'postfeature_seconds': record['timing']['match_total_seconds'] + geometry['seconds'] if geometry and record.get('timing') else None}
                rows.append(row)
                arm_rows.append(row)
            if arm == 'C' and 'B' in root_inputs and 'C' in root_inputs:
                if root_inputs['B'] != root_inputs['C']:
                    raise ValueError('B/C feature cache inputs differ')
                verified_cache_cases.append(case['case_id'])
            finite_R = [r['pose']['rotation_deg'] for r in arm_rows if r['pose']['rotation_deg'] is not None]
            finite_t = [r['pose']['translation_deg'] for r in arm_rows if r['pose']['translation_deg'] is not None]
            pairs.append({'case_id': case['case_id'], 'scan': case['scan'], 'conditions': '|'.join(case['conditions']), 'arm': arm,
                'keypoints_first': record['keypoint_counts'][0] if record and record.get('keypoint_counts') else None,
                'keypoints_second': record['keypoint_counts'][1] if record and record.get('keypoint_counts') else None,
                'candidates': support.get('candidates'), 'gt_epipolar_consistent': support.get('gt_consistent'),
                'gt_epipolar_compatibility': support.get('gt_epipolar_compatibility'), 'grid_coverage': support.get('grid_coverage'),
                'accepted_seeds': sum(r['pose']['accepted'] for r in arm_rows), 'correct_seeds': sum(r['pose']['correct'] for r in arm_rows),
                'false_accept_seeds': sum(r['pose']['false_accept'] for r in arm_rows),
                'E_inliers_by_seed': '|'.join('NA' if r['essential_inliers'] is None else str(r['essential_inliers']) for r in arm_rows),
                'accepted_points_by_seed': '|'.join(str(r['accepted_points']) for r in arm_rows),
                'mean_accepted_rotation_deg': float(np.mean(finite_R)) if finite_R else None,
                'mean_accepted_translation_deg': float(np.mean(finite_t)) if finite_t else None,
                'joint_auc10': float(np.mean([r['pose']['auc']['10'] for r in arm_rows])),
                'seed42_postfeature_seconds': arm_rows[0]['postfeature_seconds'],
                'matcher_seconds': record['timing']['matcher_seconds'] if record and record.get('timing') else None,
                'transfer_seconds': record['timing']['h2d_seconds']+record['timing']['d2h_seconds'] if record and record.get('timing') else None,
                'reasons_by_seed': '|'.join(str(r['reason']) for r in arm_rows)})
    summaries = summarize(rows, cases, seeds)
    contracts_valid = (not session['failure'] or bool(session['resource_failures'])) and not source_drift
    verdict = decision(rows, cases, seeds, summaries, session['complete'], resource, contracts_valid)
    values = [int(s) for s in (out/'gpu_memory.csv').read_text().splitlines() if s.strip().isdigit()]
    operations = json.loads((out/'operations.json').read_text())
    feature_records = [json.loads(p.read_text()) for p in (out/'features').glob('*/record.json')]
    gpu_records = [r for r in records if r.get('matching', {}).get('peak_allocated_bytes')]
    all_records = [json.loads(f.read_text()) for f in out.glob('**/record.json')]
    for r in all_records:
        if not r.get('gt_guard') or any('reference.json' in name or 'evaluation.json' in name for name in r['file_accesses']):
            raise ValueError('GT read boundary failed in feature/warmup/cold worker')
    all_gpu_stats = [v for r in all_records for v in [r.get('matching', {}), r.get('model') or {}] if v.get('peak_allocated_bytes')]
    cold = []
    for op in operations:
        if op['operation'] == 'cold':
            record = json.loads((out/'cold'/op['case_id']/op['arm']/'record.json').read_text())
            cold.append({**op, 'keypoints': record.get('keypoint_counts'), 'matches': record.get('matches'),
                'extract_and_root_seconds': sum(v['decode_extract_root_seconds'] for v in record.get('cold_feature_timings', [])),
                'model_load_seconds': record['model']['model_load_seconds'] if record.get('model') else 0})
    cost = {'session_wall_seconds': session['full_wall_seconds'],
        'shared_feature_seconds': sum(r.get('timing', {}).get('extract_seconds', 0) for r in feature_records),
        'shared_rootsift_seconds': sum(r.get('timing', {}).get('rootsift_seconds', 0) for r in feature_records),
        'peak_gpu_allocated_bytes_main': max((r['matching']['peak_allocated_bytes'] for r in gpu_records), default=0),
        'peak_gpu_reserved_bytes_main': max((r['matching']['peak_reserved_bytes'] for r in gpu_records), default=0),
        'whole_card_sampled_peak_mib': max(values) if values else None,
        'peak_gpu_allocated_bytes_all': max((r['peak_allocated_bytes'] for r in all_gpu_stats), default=0),
        'peak_gpu_reserved_bytes_all': max((r['peak_reserved_bytes'] for r in all_gpu_stats), default=0),
        'peak_worker_rss_bytes_all': max((r['peak_rss_bytes'] for r in all_records), default=0),
        'peak_worker_rss_bytes_main': max((r['peak_rss_bytes'] for r in records+feature_records), default=0),
        'cold_profiles': cold, 'feature_images': sum(r['status'] == 'success' for r in feature_records), 'feature_attempts': len(feature_records),
        'main_pair_arms': len(records), 'geometry_rows': sum(len(r['geometry']) for r in records),
        'warmup_calls': sum(op['operation'] == 'warmup' for op in operations),
        'offline_evaluation_seconds': time.perf_counter()-start}
    write(target/'evaluation.json', {'protocol_sha256': implementation['protocol_sha256'], 'rows': rows,
        'summaries': summaries, 'decision': verdict, 'cost': cost, 'GT_used_only_in_this_offline_evaluator': True,
        'B_C_feature_cache_equal_verified': len(verified_cache_cases) == len(cases),
        'B_C_feature_cache_verified_cases': verified_cache_cases, 'input_hashes_unchanged': True, 'current_source_drift_from_failed_run': source_drift,
        'archived_inference_sources_verified': True, 'quality_interpretation_allowed': session['complete'] and contracts_valid})
    export_csv(target/'pairs.csv', pairs)
    export_csv(target/'conditions.csv', [c for s in summaries for c in s['conditions']])
    print(json.dumps({'decision': verdict, 'summary': [{k: s[k] for k in ['scan', 'arm', 'macro_auc10', 'macro_correct_rate']} for s in summaries], 'cost': cost}, indent=2))
    return verdict
