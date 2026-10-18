"""E-only ranked sampling, with original-row masks and unchanged downstream geometry."""
import numpy as np
from .backend_control import reconstruct_new


def rank_order(scores, indices, mode):
    scores, indices = np.asarray(scores), np.asarray(indices)
    if scores.ndim != 1 or indices.shape != (len(scores), 2) or indices.dtype.kind not in 'iu':
        raise ValueError('one score per integer correspondence required')
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('finite scores in [0,1] required')
    if mode == 'lex':
        return np.arange(len(scores), dtype=np.int64)
    if mode != 'score':
        raise ValueError('unknown frozen ordering')
    return np.lexsort((indices[:, 1], indices[:, 0], -scores.astype(np.float64)))


def restore_mask(mask, order):
    if mask is None:
        return None
    order = np.asarray(order)
    m = np.asarray(mask)
    if order.dtype.kind not in 'iu' or sorted(order.tolist()) != list(range(len(order))):
        raise ValueError('order must be a full permutation')
    if m.size != len(order) or not np.isin(m, [0, 1]).all():
        raise ValueError('native mask shape or values invalid')
    original = np.empty(len(order), dtype=m.dtype)
    original[order] = m.reshape(-1)
    return original.reshape(-1, 1)


def reconstruct_sampling(p, q, k1, k2, cfg, base_params, arm, scores, indices):
    """Serial process only; ranking affects E, never H or triangulation row order."""
    import cv2
    order = rank_order(scores, indices, arm['order'])
    if len(order) != len(p) or np.shape(q) != np.shape(p):
        raise ValueError('match rows and ranking differ')
    params = dict(base_params, sampler=arm['sampler'])
    native = cv2.findEssentialMat
    trace = {}

    def ordered_native(a, b, first_K, second_K, dist1, dist2, usac):
        if trace:
            raise RuntimeError('expected a single E fit')
        e, m = native(np.ascontiguousarray(a[order]), np.ascontiguousarray(b[order]),
                      first_K, second_K, dist1, dist2, usac)
        trace['called'] = True
        if m is not None:
            trace['native_mask'] = np.asarray(m).reshape(-1).copy()
        return e, restore_mask(m, order)

    cv2.findEssentialMat = ordered_native
    try:
        value = reconstruct_new(p, q, k1, k2, cfg, params)
    finally:
        cv2.findEssentialMat = native
    value['arrays']['estimation_order'] = order
    if 'native_mask' in trace:
        value['arrays']['estimation_native_mask'] = trace['native_mask']
    value['E_backend'] = 'msac_'+arm['sampler']+'_'+arm['order']
    return value


def scorecards(rows, cases, config):
    lookup = {(r['case_id'], r['arm'], r['seed']): r for r in rows}
    if len(rows) != len(lookup) or len(rows) != len(cases)*len(config['arms'])*len(config['seeds']):
        raise ValueError('missing/duplicate planned rows')
    cards = []
    for scan in [1, 6]:
        for arm in config['arms']:
            seed_stats = []
            selected_cases = [c for c in cases if c['scan'] == scan]
            for seed in config['seeds']:
                group = [lookup[c['case_id'], arm, seed] for c in selected_cases]
                macro = {}
                for key in ['auc5', 'auc10', 'auc20', 'correct_rate']:
                    macro[key] = float(np.mean([np.mean([r['pose']['correct'] if key == 'correct_rate' else r['pose']['auc'][key[3:]]
                        for r in group if condition in r['conditions']]) for condition in ['normal5', 'reduced3', 'spaced3', 'blur5']]))
                seed_stats.append({'seed':seed, **macro, **{k:sum(r['pose'][k] for r in group) for k in ['correct', 'accepted', 'false_accept']}})
            group = [lookup[c['case_id'], arm, seed] for c in selected_cases for seed in config['seeds']]
            errors = [r['pose']['joint_deg'] for r in group if r['pose']['accepted'] and r['pose']['joint_deg'] is not None]
            cards.append({'scan':scan, 'arm':arm, 'seed_stats':seed_stats,
                'macro':{k:{'mean':float(np.mean([s[k] for s in seed_stats])), 'min':min(s[k] for s in seed_stats), 'max':max(s[k] for s in seed_stats)} for k in ['auc5', 'auc10', 'auc20', 'correct_rate']},
                **{k+'_events':sum(r['pose'][k] for r in group) for k in ['correct', 'accepted', 'false_accept']},
                'severe_events':sum(v > 20 for v in errors), 'severe_pairs':sorted({r['case_id'] for r in group if r['pose']['accepted'] and r['pose']['joint_deg'] is not None and r['pose']['joint_deg'] > 20}),
                'accepted_p95':float(np.quantile(errors, .95, method='linear')) if errors else None,
                'accepted_max':max(errors) if errors else None,
                'all_seeds_correct':sum(all(lookup[c['case_id'],arm,s]['pose']['correct'] for s in config['seeds']) for c in selected_cases),
                'geometry_seconds':sum(r['geometry_seconds'] or 0 for r in group)})
    bykey = {(c['scan'],c['arm']):c for c in cards}
    contrasts = []
    for candidate, baseline in [('P_score','U_lex'), ('P_score','U_score'), ('U_score','U_lex'), ('P_lex','U_lex')]:
        for scan in [1, 6]:
            a,b = bykey[scan,candidate], bykey[scan,baseline]
            changed = []
            for c in cases:
                if c['scan'] != scan: continue
                for s in config['seeds']:
                    x,y = lookup[c['case_id'],candidate,s],lookup[c['case_id'],baseline,s]
                    if x['pose']['correct'] != y['pose']['correct'] or x['pose']['false_accept'] != y['pose']['false_accept']:
                        changed.append({'case_id':c['case_id'], 'seed':s,'old_correct':y['pose']['correct'],'new_correct':x['pose']['correct'],
                                        'old_joint':y['pose']['joint_deg'],'new_joint':x['pose']['joint_deg'],'new_status':x['status']})
            contrasts.append({'scan':scan,'candidate':candidate,'baseline':baseline,
                'auc10_delta':a['macro']['auc10']['mean']-b['macro']['auc10']['mean'],
                'macro_correct_delta':a['macro']['correct_rate']['mean']-b['macro']['correct_rate']['mean'],
                'severe_delta':a['severe_events']-b['severe_events'],'transitions':changed})
    notices = {
        'accuracy':all(bykey[s,'P_score']['macro']['auc10']['mean']-bykey[s,'U_lex']['macro']['auc10']['mean'] >= .01-1e-12 for s in [1,6]),
        'correctness':all(bykey[s,'P_score']['macro']['correct_rate']['mean']-bykey[s,'U_lex']['macro']['correct_rate']['mean'] >= .02-1e-12 for s in [1,6]),
        'tail':sum(bykey[s,'U_lex']['severe_events'] for s in [1,6]) > 0 and
               sum(bykey[s,'P_score']['severe_events'] for s in [1,6]) <= sum(bykey[s,'U_lex']['severe_events'] for s in [1,6])*2/3 and
               all(bykey[s,'P_score']['macro']['auc10']['mean']-bykey[s,'U_lex']['macro']['auc10']['mean'] >= -.01-1e-12 for s in [1,6])}
    return {'cards':cards,'contrasts':contrasts,'notices':notices,
            'priority_candidate':'C_msac_uniform_lex retained per user; PROSAC tradeoffs require human judgment',
            'unseen_objects_tested':False,'safe_replacement_or_fusion':False}
