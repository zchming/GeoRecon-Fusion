"""One-factor MSAC effort ablations; report tradeoffs, never select poses."""
import numpy as np


def scorecards(rows, cases, config):
    arms, seeds = list(config['profiles']), config['seeds']
    lookup = {(r['case_id'],r['arm'],r['seed']):r for r in rows}
    if len(rows) != len(lookup) or len(rows) != len(cases)*len(arms)*len(seeds):
        raise ValueError('missing/duplicate planned rows')
    cards = []
    for scan in [1,6]:
        selected_cases = [c for c in cases if c['scan'] == scan]
        for arm in arms:
            seed_stats = []
            for seed in seeds:
                group = [lookup[c['case_id'],arm,seed] for c in selected_cases]
                macro = {}
                for key in ['auc5','auc10','auc20','correct_rate']:
                    values = []
                    for condition in ['normal5','reduced3','spaced3','blur5']:
                        selected = [r for r in group if condition in r['conditions']]
                        if not selected: raise ValueError('missing planned condition')
                        values.append(np.mean([r['pose']['correct'] if key == 'correct_rate' else r['pose']['auc'][key[3:]] for r in selected]))
                    macro[key] = float(np.mean(values))
                seed_stats.append({'seed':seed,**macro,**{k:sum(r['pose'][k] for r in group) for k in ['correct','accepted','false_accept']}})
            group = [lookup[c['case_id'],arm,s] for c in selected_cases for s in seeds]
            errors = [r['pose']['joint_deg'] for r in group if r['pose']['accepted'] and r['pose']['joint_deg'] is not None]
            cards.append({'scan':scan,'arm':arm,'seed_stats':seed_stats,
                'macro':{k:{'mean':float(np.mean([s[k] for s in seed_stats])), 'min':min(s[k] for s in seed_stats), 'max':max(s[k] for s in seed_stats)} for k in ['auc5','auc10','auc20','correct_rate']},
                **{k+'_events':sum(r['pose'][k] for r in group) for k in ['correct','accepted','false_accept']},
                'severe_events':sum(v>20 for v in errors),
                'severe_pairs':sorted({r['case_id'] for r in group if r['pose']['accepted'] and r['pose']['joint_deg'] is not None and r['pose']['joint_deg']>20}),
                'accepted_p95':float(np.quantile(errors,.95,method='linear')) if errors else None,
                'accepted_max':max(errors) if errors else None,
                'all_seeds_correct':sum(all(lookup[c['case_id'],arm,s]['pose']['correct'] for s in seeds) for c in selected_cases),
                'geometry_seconds':sum(r['geometry_seconds'] or 0 for r in group)})
    bykey = {(c['scan'],c['arm']):c for c in cards}
    contrasts,notices = [],{}
    for arm in arms:
        if arm == 'base':continue
        deltas,ranges,base_ranges = [],[],[]
        for scan in [1,6]:
            new,old = bykey[scan,arm],bykey[scan,'base']
            delta = new['macro']['auc10']['mean']-old['macro']['auc10']['mean'];deltas.append(delta)
            ranges.append(new['macro']['auc10']['max']-new['macro']['auc10']['min'])
            base_ranges.append(old['macro']['auc10']['max']-old['macro']['auc10']['min'])
            transitions = []
            for c in cases:
                if c['scan'] != scan:continue
                for seed in seeds:
                    a,b = lookup[c['case_id'],arm,seed],lookup[c['case_id'],'base',seed]
                    if a['pose']['correct'] != b['pose']['correct'] or a['pose']['false_accept'] != b['pose']['false_accept']:
                        transitions.append({'case_id':c['case_id'],'seed':seed,'old_correct':b['pose']['correct'],'new_correct':a['pose']['correct'],
                                            'old_joint':b['pose']['joint_deg'],'new_joint':a['pose']['joint_deg'],'new_status':a['status']})
            contrasts.append({'arm':arm,'scan':scan,'auc10_delta':delta,
                'macro_correct_delta':new['macro']['correct_rate']['mean']-old['macro']['correct_rate']['mean'],
                'false_delta':new['false_accept_events']-old['false_accept_events'],
                'severe_delta':new['severe_events']-old['severe_events'],
                'new_severe_pairs':sorted(set(new['severe_pairs'])-set(old['severe_pairs'])),
                'transitions':transitions})
        false_new=sum(bykey[s,arm]['false_accept_events'] for s in [1,6]);false_old=sum(bykey[s,'base']['false_accept_events'] for s in [1,6])
        accepted_new=sum(bykey[s,arm]['accepted_events'] for s in [1,6]);accepted_old=sum(bykey[s,'base']['accepted_events'] for s in [1,6])
        notices[arm]={'mean':all(d >= .005-1e-12 for d in deltas) or (max(deltas)>=.01-1e-12 and min(deltas)>=-.005-1e-12),
                      'variation':all(b>0 and a<=.9*b+1e-12 for a,b in zip(ranges,base_ranges)),
                      'false_acceptance':false_new<=false_old-3 and accepted_new>=.9*accepted_old}
    return {'cards':cards,'contrasts':contrasts,'notices':notices,
            'priority_candidate':'original C_msac retained; any new tradeoffs require human judgment',
            'unseen_objects_tested':False,'automatically_selected_output':False}


def choose_development(rows, config):
    """Freeze one global configuration using scan1 only, before scan6 GT reads."""
    dev = [r for r in rows if r['scan'] == 1]
    expected = 23*len(config['profiles'])*len(config['seeds'])
    if len(dev) != expected or any(not r['pose']['GT_pose_defined'] for r in dev):
        raise ValueError('development selection requires all planned defined-GT rows')
    metrics = []
    for arm in config['profiles']:
        group = [r for r in dev if r['arm'] == arm]
        auc = float(np.mean([np.mean([r['pose']['auc']['10'] for r in group if cond in r['conditions']]) for cond in ['normal5','reduced3','spaced3','blur5']]))
        correct = float(np.mean([np.mean([r['pose']['correct'] for r in group if cond in r['conditions']]) for cond in ['normal5','reduced3','spaced3','blur5']]))
        metrics.append({'arm':arm,'auc10':auc,'correct_rate':correct,
            'severe':sum(r['pose']['accepted'] and r['pose']['joint_deg'] is not None and r['pose']['joint_deg']>20 for r in group),
            'false_accepts':sum(r['pose']['false_accept'] for r in group)})
    base = metrics[0]
    if base['arm'] != 'base':raise ValueError('baseline must be declared first')
    eligible = [m for m in metrics if m['auc10'] >= base['auc10']-.005-1e-12 and m['correct_rate'] >= base['correct_rate']-.02-1e-12]
    selected = min(eligible,key=lambda m:(m['severe'],m['false_accepts'],-m['auc10'],m['arm']!='base',list(config['profiles']).index(m['arm'])))
    return {'selected_arm':selected['arm'],'development_metrics':metrics,'eligible':[m['arm'] for m in eligible],
            'uses_validation_GT':False,'per_pair_pose_selection':False}
