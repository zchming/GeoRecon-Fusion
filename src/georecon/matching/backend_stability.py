"""Profile scorecards: expose gains and regressions without choosing output poses."""
import numpy as np
from .backend_control import reconstruct_new


def reconstruct_profile(p, q, k1, k2, cfg, profile, parameters):
    from ..teaching.two_view import reconstruct
    if profile == 'old':
        value = reconstruct(p, q, k1, k2, cfg)
    else:
        value = reconstruct_new(p, q, k1, k2, cfg, parameters)
    value['E_backend'] = profile
    return value


def scorecards(rows, cases, config):
    arms, seeds = config['arms'], config['seeds']
    lookup = {(r['case_id'], r['arm'], r['seed']): r for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(cases)*len(arms)*len(seeds):
        raise ValueError('missing/duplicate planned rows')
    cards = []
    for scan in [1, 6]:
        selected_cases = [c for c in cases if c['scan'] == scan]
        for arm in arms:
            seed_stats = []
            for seed in seeds:
                selected = [lookup[c['case_id'], arm, seed] for c in selected_cases]
                macro = {}
                for key in ['auc5', 'auc10', 'auc20', 'correct_rate']:
                    groups = []
                    for condition in ['normal5', 'reduced3', 'spaced3', 'blur5']:
                        group = [r for r in selected if condition in r['conditions']]
                        if not group:
                            raise ValueError('missing condition')
                        groups.append(np.mean([r['pose']['correct'] if key == 'correct_rate' else r['pose']['auc'][key[3:]] for r in group]))
                    macro[key] = float(np.mean(groups))
                seed_stats.append({'seed':seed, **macro,
                    **{k:sum(r['pose'][k] for r in selected) for k in ['correct', 'accepted', 'false_accept']},
                    'severe':sum(r['pose']['accepted'] and r['pose']['joint_deg'] is not None and r['pose']['joint_deg'] > 20 for r in selected)})
            selected = [lookup[c['case_id'],arm,s] for c in selected_cases for s in seeds]
            frequencies = [sum(lookup[c['case_id'],arm,s]['pose']['correct'] for s in seeds) for c in selected_cases]
            severe_pairs = sorted({r['case_id'] for r in selected if r['pose']['accepted'] and r['pose']['joint_deg'] is not None and r['pose']['joint_deg'] > 20})
            accepted_errors = [r['pose']['joint_deg'] for r in selected if r['pose']['accepted'] and r['pose']['joint_deg'] is not None]
            cards.append({'scan':scan,'arm':arm,'seed_stats':seed_stats,
                'macro':{k:{'mean':float(np.mean([r[k] for r in seed_stats])), 'min':min(r[k] for r in seed_stats), 'max':max(r[k] for r in seed_stats)} for k in ['auc5','auc10','auc20','correct_rate']},
                'correct_frequency':{str(k):frequencies.count(k) for k in range(len(seeds)+1)},
                'all_seeds_correct':frequencies.count(len(seeds)),
                'correct_events':sum(r['pose']['correct'] for r in selected),
                'false_accept_events':sum(r['pose']['false_accept'] for r in selected),
                'severe_events':sum(r['severe'] for r in seed_stats),'severe_pairs':severe_pairs,
                'accepted_joint_quantiles':{str(q):float(np.quantile(accepted_errors,q,method='linear')) if accepted_errors else None for q in [.5,.9,.95]},
                'geometry_seconds':sum(r['geometry_seconds'] or 0 for r in selected)})
    bykey = {(r['scan'],r['arm']):r for r in cards}
    contrasts = []
    for candidate, baseline in [('C_msac','C_magsac'),('C_msac','C_old'),('C_magsac','C_old'),('B_msac','B_magsac'),('B_msac','B_old'),('C_msac','B_msac')]:
        for scan in [1, 6]:
            a,b = bykey[scan,candidate],bykey[scan,baseline]
            transitions = []
            for c in cases:
                if c['scan'] != scan: continue
                for seed in seeds:
                    x,y = lookup[c['case_id'],candidate,seed],lookup[c['case_id'],baseline,seed]
                    if x['pose']['correct'] != y['pose']['correct'] or (x['pose']['accepted'] and x['pose']['joint_deg'] is not None and x['pose']['joint_deg'] > 20 and not (y['pose']['accepted'] and y['pose']['joint_deg'] is not None and y['pose']['joint_deg'] > 20)):
                        transitions.append({'case_id':c['case_id'],'seed':seed,'baseline_correct':y['pose']['correct'],
                            'candidate_correct':x['pose']['correct'],'baseline_joint':y['pose']['joint_deg'],'candidate_joint':x['pose']['joint_deg'],
                            'candidate_status':x['status']})
            contrasts.append({'scan':scan,'candidate':candidate,'baseline':baseline,
                'mean_auc10_delta':a['macro']['auc10']['mean']-b['macro']['auc10']['mean'],
                'worst_seed_auc10_delta':a['macro']['auc10']['min']-b['macro']['auc10']['min'],
                'mean_correct_rate_delta':a['macro']['correct_rate']['mean']-b['macro']['correct_rate']['mean'],
                'false_accept_event_delta':a['false_accept_events']-b['false_accept_events'],
                'severe_event_delta':a['severe_events']-b['severe_events'],
                'new_severe_pairs':sorted(set(a['severe_pairs'])-set(b['severe_pairs'])),
                'transitions':transitions})
    notices = {}
    for candidate in ['C_magsac','C_msac']:
        notices[candidate] = {'accuracy_gain_over_old_both_objects':all(bykey[s,candidate]['macro']['auc10']['mean']-bykey[s,'C_old']['macro']['auc10']['mean'] >= .05-1e-12 for s in [1,6])}
    notices['C_msac'].update(
        accuracy_gain_over_magsac_both_objects=all(bykey[s,'C_msac']['macro']['auc10']['mean']-bykey[s,'C_magsac']['macro']['auc10']['mean'] >= .02-1e-12 for s in [1,6]),
        severe_events_halved_within_auc_tolerance=all(bykey[s,'C_magsac']['severe_events'] > 0 and bykey[s,'C_msac']['severe_events'] <= bykey[s,'C_magsac']['severe_events']/2 and bykey[s,'C_msac']['macro']['auc10']['mean']-bykey[s,'C_magsac']['macro']['auc10']['mean'] >= -.02-1e-12 for s in [1,6]))
    return {'cards':cards,'contrasts':contrasts,'notices':notices,
            'candidate_decision':'retain MAGSAC per user; present MSAC tradeoffs to user, no automatic deployment or pose selection',
            'unseen_objects_tested':False,'policy_or_fusion_implemented':False}
