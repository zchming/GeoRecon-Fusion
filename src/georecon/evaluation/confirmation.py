"""Predeclared nonregression/replication gates; no algorithm or GT-based selection."""

def confirmation_gate(before,after,*,scene_score_deltas,scene_severe_deltas,seed_score_deltas,complete,replication):
    """Zero noninferiority margin; TUM material thresholds fixed before new inputs."""
    checks={'available_and_complete':bool(complete),
            'score_nonregression':after['auc10']>=before['auc10']-1e-12,
            'accurate_map_nonregression':after['accurate_scene']>=before['accurate_scene']-1e-12,
            'false_nonincrease':after['false_pairs']<=before['false_pairs'],
            'severe_nonincrease':after['severe_pairs']<=before['severe_pairs'],
            'no_scene_severe_increase':bool(scene_severe_deltas) and all(v<=0 for v in scene_severe_deltas.values())}
    if replication:
        checks.update(material_tum_gain=(after['auc10']-before['auc10']>=.01-1e-12 or
                        (before['false_pairs']>0 and after['false_pairs']<=.9*before['false_pairs'] and checks['score_nonregression'])),
                      all_tum_scene_scores_nonnegative=bool(scene_score_deltas) and all(v>=-1e-12 for v in scene_score_deltas.values()),
                      four_of_five_seed_scores_positive=len(seed_score_deltas)==5 and sum(v>1e-12 for v in seed_score_deltas.values())>=4)
    return {'passed':all(checks.values()),'checks':checks}
