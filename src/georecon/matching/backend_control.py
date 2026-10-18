"""Frozen E-estimator contrast; legacy geometry and acceptance stay unchanged."""
from contextlib import contextmanager
import numpy as np

ARMS = ('B_old', 'B_new', 'C_old', 'C_new')


def usac_parameters(spec, seed):
    import cv2
    params = cv2.UsacParams()
    for key, value in spec.items():
        value = seed if key == 'randomGeneratorState' else getattr(cv2, value) if isinstance(value, str) else value
        setattr(params, key, value)
        if getattr(params, key) != value:
            raise ValueError(f'USAC parameter not effective: {key}')
    return params


@contextmanager
def replace_essential(p, q, k1, k2, cfg, spec):
    """Serial worker only. Replace exactly one E call, restore even on error.

    The legacy function still computes H, chooses poses, triangulates and rejects.
    Original raw pixels/K go to USAC; normalized arguments are checked, not reused.
    """
    import cv2
    from ..teaching.two_view import normalize
    original = cv2.findEssentialMat
    calls = []
    p, q, k1, k2 = [np.ascontiguousarray(v, dtype=np.float64) for v in [p, q, k1, k2]]
    params = usac_parameters(spec, cfg['seed'])

    def estimate(a, b, camera, **kwargs):
        if calls:
            raise RuntimeError('expected one legacy essential call')
        np.testing.assert_array_equal(a, normalize(p, k1))
        np.testing.assert_array_equal(b, normalize(q, k2))
        np.testing.assert_array_equal(camera, np.eye(3))
        expected = dict(method=cv2.RANSAC, prob=cfg['essential_ransac_probability'],
                        threshold=cfg['essential_ransac_px']/np.mean([k1[0, 0], k1[1, 1], k2[0, 0], k2[1, 1]]),
                        maxIters=cfg['essential_ransac_iterations'])
        if kwargs != expected:
            raise ValueError('legacy E call contract changed')
        calls.append(True)
        e, mask = original(p, q, k1, k2, None, None, params)
        if e is not None:
            if mask is None or np.asarray(mask).size != len(p) or not np.isin(mask, [0, 1]).all():
                raise ValueError('USAC returned invalid mask')
        return e, mask

    cv2.findEssentialMat = estimate
    try:
        yield calls
    finally:
        cv2.findEssentialMat = original


def reconstruct_new(p, q, k1, k2, cfg, spec):
    from ..teaching.two_view import reconstruct
    with replace_essential(p, q, k1, k2, cfg, spec) as calls:
        result = reconstruct(p, q, k1, k2, cfg)
    result['E_backend'] = 'usac_magsac_seeded'
    result['essential_calls'] = len(calls)
    return result


def compare_legacy(value, arrays, historical, historical_arrays):
    """No GT needed. Never substitute historical predictions for current results."""
    differences = []
    for key in ['status', 'reason']:
        if value[key] != historical[key]:
            differences.append(key)
    oldstats = historical['statistics']
    for key in ['matches', 'essential_inliers', 'homography_inliers', 'kept_points']:
        if value['stats'].get(key) != oldstats.get(key):
            differences.append('stats.'+key)
    # Check every available E/H/triangulation support mask and kept index array.
    keys = {k for k in set(arrays) | set(historical_arrays)
            if 'mask' in k or k.endswith('_indices') or k in ['candidate_positive', 'candidate_finite']}
    for key in sorted(keys):
        if key not in arrays or key not in historical_arrays or not np.array_equal(arrays[key], historical_arrays[key]):
            differences.append('arrays.'+key)
    if value['status'] == 'success':
        for key in ['accepted_R', 'accepted_t']:
            if key not in historical_arrays or not np.allclose(arrays[key], historical_arrays[key], atol=1e-9, rtol=0):
                differences.append(key)
    return differences


def summarize(rows, cases, seeds):
    lookup = {(r['case_id'], r['arm'], r['seed']): r for r in rows}
    summaries = []
    for scan in [1, 6]:
        for arm in ARMS:
            groups = []
            for condition in ['normal5', 'reduced3', 'spaced3', 'blur5']:
                selected = [lookup[c['case_id'], arm, s] for c in cases if c['scan'] == scan and condition in c['conditions'] for s in seeds]
                if not selected:
                    raise ValueError('missing planned condition')
                groups.append({'condition': condition, 'planned': len(selected),
                    'correct_rate': float(np.mean([r['pose']['correct'] for r in selected])),
                    **{f'auc{t}': float(np.mean([r['pose']['auc'][str(t)] for r in selected])) for t in [5, 10, 20]}})
            summaries.append({'scan': scan, 'arm': arm, 'conditions': groups,
                'macro_correct_rate': float(np.mean([g['correct_rate'] for g in groups])),
                **{f'macro_auc{t}': float(np.mean([g[f'auc{t}'] for g in groups])) for t in [5, 10, 20]}})
    return summaries


def decision(rows, cases, seeds, summaries, valid, resource_failure=False, incomplete=False):
    lookup = {(r['case_id'], r['arm'], r['seed']): r for r in rows}
    if len(lookup) != len(cases)*len(ARMS)*len(seeds) or len(rows) != len(lookup):
        raise ValueError('missing or duplicate planned rows')
    macro = {(r['scan'], r['arm']): r for r in summaries}
    def pose(c, a, s): return lookup[c['case_id'], a, s]['pose']
    def stable(c, a): return all(pose(c, a, s)['correct'] for s in seeds)
    normal_regressions, severe_new, new_false, transitions = [], [], [], []
    for c in cases:
        for baseline in ['C_old', 'B_old', 'B_new']:
            for s in seeds:
                old, new = pose(c, baseline, s), pose(c, 'C_new', s)
                entry = {'case_id': c['case_id'], 'baseline': baseline, 'seed': s,
                         'old_joint_deg': old['joint_deg'], 'new_joint_deg': new['joint_deg']}
                if new['false_accept'] and not old['false_accept']:
                    new_false.append(entry)
                if old['correct'] != new['correct']:
                    transitions.append(dict(entry, kind='rescue' if new['correct'] else 'loss'))
                if baseline in ['C_old', 'B_old'] and new['accepted'] and new['joint_deg'] is not None and new['joint_deg'] > 20 and (old['correct'] or not old['accepted']):
                    severe_new.append(entry)
            if baseline in ['C_old', 'B_old'] and 'normal5' in c['conditions'] and stable(c, baseline) and not stable(c, 'C_new'):
                normal_regressions.append({'case_id': c['case_id'], 'baseline': baseline})
    per_object = {}
    for scan in [1, 6]:
        selected = [c for c in cases if c['scan'] == scan]
        per_object[scan] = {
            'stable_correct': {a: sum(stable(c, a) for c in selected) for a in ARMS},
            'false_by_seed': {a: [sum(pose(c, a, s)['false_accept'] for c in selected) for s in seeds] for a in ARMS}}
    def delta(scan, a, b, key): return macro[scan, a][key]-macro[scan, b][key]
    quality1 = all(delta(s, 'C_new', 'C_old', 'macro_auc10') > 1e-12 and
                   delta(s, 'C_new', 'C_old', 'macro_correct_rate') >= -1e-12 and
                   per_object[s]['stable_correct']['C_new']-per_object[s]['stable_correct']['C_old'] >= 1 for s in [1, 6])
    quality2 = not normal_regressions and not severe_new and all(
        all(n <= o for n, o in zip(per_object[s]['false_by_seed']['C_new'], per_object[s]['false_by_seed'][a]))
        for s in [1, 6] for a in ['C_old', 'B_old'])
    quality3 = all(delta(s, 'C_new', 'B_old', 'macro_auc10') >= -1e-12 and
                   delta(s, 'C_new', 'B_old', 'macro_correct_rate') >= -1e-12 for s in [1, 6]) and sum(
                   per_object[s]['stable_correct']['C_new']-per_object[s]['stable_correct']['B_old'] for s in [1, 6]) > 0
    timed = [lookup[c['case_id'], a, seeds[0]].get('geometry_seconds') for c in cases for a in ['C_old', 'C_new']]
    cost = None
    if all(v is not None for v in timed):
        dt = np.array(timed[1::2])-np.array(timed[::2])
        cost = {'median_seconds': float(np.median(dt)), 'p95_seconds': float(np.quantile(dt, .95, method='linear'))}
    gates = {'G1': quality1, 'G2': quality2, 'G3': quality3,
             'G4': cost is not None and cost['median_seconds'] <= 1 and cost['p95_seconds'] <= 3,
             'G5': bool(valid and not resource_failure and not incomplete and all(r['pose']['GT_pose_defined'] for r in rows))}
    verdict = ('NO_GO_RESOURCE' if resource_failure else 'BLOCKED' if not valid else 'INCONCLUSIVE' if incomplete or not gates['G5'] else
               'NO_GO_EVIDENCE' if not all(gates[k] for k in ['G1', 'G2', 'G3']) else
               'NO_GO_COST' if not gates['G4'] else 'GO_FEASIBILITY')
    return {'verdict': verdict, 'gates': gates, 'per_object': per_object, 'cost': cost,
            'normal_regressions': normal_regressions, 'severe_new': severe_new,
            'new_false_accepts': new_false, 'correct_transitions': transitions,
            'no_safe_replacement_policy': True, 'unseen_object_validation': False}
