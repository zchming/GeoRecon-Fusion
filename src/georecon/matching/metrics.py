"""Offline-only geometry proxies, failure-aware aggregation and frozen gates."""
import numpy as np
from ..geometry.camera import relative_pose
from ..reliability.geometry import fundamental, sampson_px
from ..evaluation.poses import rotation_error_deg, translation_direction_error_deg


def support_metrics(first, second, p, q, indices, keypoint_counts, threshold=1.5, grid=(8, 6)):
    F = fundamental(first, second)
    if F is None:
        return {'status': 'unavailable', 'reason': 'zero_reference_baseline'}, np.zeros(len(p), bool)
    errors = sampson_px(F, p, q)
    good = np.isfinite(errors) & (errors <= threshold)
    n, count = len(p), int(good.sum())
    coverage, feature_coverage = [], []
    for camera, pixels, side, total in zip([first, second], [p, q], [0, 1], keypoint_counts):
        points = pixels[good]
        cells = np.floor(points / [camera.width, camera.height] * grid).astype(int)
        valid = (cells >= 0).all(axis=1) & (cells < grid).all(axis=1)
        occupied = len(np.unique(cells[valid], axis=0))
        coverage.append(occupied / (grid[0]*grid[1]))
        feature_coverage.append(len(np.unique(indices[good, side])) / total if total else None)
    return {'status': 'evaluated', 'candidates': n, 'gt_consistent': count,
            'gt_epipolar_compatibility': count/n if n else None,
            'unscorable_matches': int((~np.isfinite(errors)).sum()),
            'grid_coverage_per_image': coverage, 'grid_coverage': min(coverage),
            'feature_coverage_per_image': feature_coverage,
            'true_3d_correspondence_precision': None}, good


def pose_metrics(first, second, R, t, accepted, threshold=5.0):
    if not accepted:
        return {'accepted': False, 'correct': False, 'false_accept': False,
                'rotation_deg': None, 'translation_deg': None, 'joint_deg': None,
                'auc': {str(k): 0.0 for k in [5, 10, 20]},
                'GT_pose_defined': bool(np.linalg.norm(relative_pose(first, second)[1]) > 1e-12)}
    rg, tg = relative_pose(first, second)
    rotation = rotation_error_deg(np.asarray(R), rg)
    translation = translation_direction_error_deg(np.asarray(t), tg)
    joint = max(rotation, translation) if translation is not None else None
    return {'accepted': True, 'correct': joint is not None and joint <= threshold,
            'false_accept': joint is not None and joint > threshold,
            'rotation_deg': rotation, 'translation_deg': translation, 'joint_deg': joint,
            'auc': {str(k): max(0.0, 1-joint/k) if joint is not None else 0.0 for k in [5, 10, 20]},
            'GT_pose_defined': joint is not None}


def summarize(rows, cases, seeds):
    """Four-condition macro per object, preserving each planned pair/seed."""
    lookup = {(r['case_id'], r['arm'], r['seed']): r for r in rows}
    result = []
    for scan in [1, 6]:
        for arm in ['A', 'B', 'C']:
            condition_rows = []
            for condition in ['normal5', 'reduced3', 'spaced3', 'blur5']:
                selected = [lookup[c['case_id'], arm, seed] for c in cases if c['scan'] == scan and condition in c['conditions'] for seed in seeds]
                n = len(selected)
                condition_rows.append({'scan': scan, 'arm': arm, 'condition': condition,
                    'pair_seed_count': n, 'accepted': sum(r['pose']['accepted'] for r in selected),
                    'correct': sum(r['pose']['correct'] for r in selected),
                    'false_accepts': sum(r['pose']['false_accept'] for r in selected),
                    'correct_rate': sum(r['pose']['correct'] for r in selected)/n,
                    'auc5': float(np.mean([r['pose']['auc']['5'] for r in selected])),
                    'auc10': float(np.mean([r['pose']['auc']['10'] for r in selected])),
                    'auc20': float(np.mean([r['pose']['auc']['20'] for r in selected]))})
            result.append({'scan': scan, 'arm': arm, 'conditions': condition_rows,
                'macro_auc10': float(np.mean([r['auc10'] for r in condition_rows])),
                'macro_correct_rate': float(np.mean([r['correct_rate'] for r in condition_rows]))})
    return result


def decision(rows, cases, seeds, summaries, completed, resource_failures, contracts_valid=True):
    lookup = {(r['case_id'], r['arm'], r['seed']): r for r in rows}
    macro = {(s['scan'], s['arm']): s for s in summaries}
    rescues, regressions, new_false = [], [], []
    support, delta_time = {}, []
    for c in cases:
        b = [lookup[c['case_id'], 'B', s] for s in seeds]
        a = [lookup[c['case_id'], 'C', s] for s in seeds]
        if c['difficulty_group'] and not any(r['pose']['correct'] for r in b) and all(r['pose']['correct'] for r in a):
            rescues.append(c['case_id'])
        if 'normal5' in c['conditions'] and all(r['pose']['correct'] for r in b) and not all(r['pose']['correct'] for r in a):
            regressions.append(c['case_id'])
        for rb, rc in zip(b, a):
            if rc['pose']['false_accept'] and not rb['pose']['false_accept']:
                new_false.append({'case_id': c['case_id'], 'seed': rc['seed']})
        if b[0].get('postfeature_seconds') is not None and a[0].get('postfeature_seconds') is not None:
            delta_time.append(a[0]['postfeature_seconds'] - b[0]['postfeature_seconds'])
    for scan in [1, 6]:
        hard = [c for c in cases if c['scan'] == scan and c['difficulty_group']]
        pairs = [(lookup[c['case_id'], 'B', seeds[0]]['support'], lookup[c['case_id'], 'C', seeds[0]]['support']) for c in hard]
        valid = all(b.get('status') == a.get('status') == 'evaluated' for b, a in pairs)
        support[str(scan)] = {'count_delta_median': float(np.median([a['gt_consistent']-b['gt_consistent'] for b, a in pairs])) if valid else None,
                             'coverage_delta_median': float(np.median([a['grid_coverage']-b['grid_coverage'] for b, a in pairs])) if valid else None}
    auc_delta = {str(scan): macro[scan, 'C']['macro_auc10']-macro[scan, 'B']['macro_auc10'] for scan in [1, 6]}
    timing = {'pairs': len(delta_time), 'median_added_seconds': float(np.median(delta_time)) if delta_time else None,
              'p95_added_seconds': float(np.quantile(delta_time, .95, method='linear')) if delta_time else None}
    gates = {
        'G1': len(rescues) >= 2 and all(any(c['scan'] == scan and c['case_id'] in rescues for c in cases) for scan in [1, 6]),
        'G2': all(v >= -1e-12 for v in auc_delta.values()) and not regressions and not new_false,
        'G3': all(s['count_delta_median'] is not None and s['count_delta_median'] > 0 and s['coverage_delta_median'] >= -1e-12 for s in support.values()),
        'G4': len(delta_time) == len(cases) and timing['median_added_seconds'] <= 1+1e-12 and timing['p95_added_seconds'] <= 3+1e-12,
        'G5': completed and not resource_failures and contracts_valid}
    gt_defined = all(r['pose']['GT_pose_defined'] and r['support'].get('status') == 'evaluated' for r in rows if r['arm'] in ['B', 'C'])
    verdict = ('BLOCKED' if not contracts_valid else 'NO_GO_RESOURCE' if resource_failures else
               'INCONCLUSIVE' if not completed or not gt_defined else 'NO_GO_EVIDENCE' if not all(gates[k] for k in ['G1', 'G2', 'G3']) else
               'NO_GO_COST' if not gates['G4'] else 'GO_FEASIBILITY')
    # Partial/invalid runs cannot pass a quality gate by vacuous truth or zero-filled rows.
    if not completed or not contracts_valid or not gt_defined:
        for key in ['G1', 'G2', 'G3', 'G4']:
            gates[key] = None
    return {'verdict': verdict, 'gates': {k: bool(v) if v is not None else None for k, v in gates.items()}, 'rescued_pairs': rescues,
            'normal_regressions': regressions, 'new_false_accepts': new_false, 'auc10_delta_C_minus_B': auc_delta,
            'hard_support_C_minus_B': support, 'cost_C_minus_B': timing}
