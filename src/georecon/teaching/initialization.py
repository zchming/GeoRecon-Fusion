"""Bounded seed selection on measured matches; no truth inputs or learned weights."""
from itertools import combinations
import time
import numpy as np
from .incremental import collect_links, reconstruct_incremental
from ..matching.backend_control import reconstruct_new


def directed_pairs(matches, a, b):
    return matches[a,b] if a<b else matches[b,a][:,::-1]


def coverage(points, shape, grid=(8,6)):
    points=np.asarray(points)
    if not len(points) or not np.isfinite(points).all():return None
    cells=np.floor(points/np.asarray(shape)*grid).astype(int)
    valid=(cells>=0).all(1)&(cells<grid).all(1)
    return len(np.unique(cells[valid],axis=0))/(grid[0]*grid[1])


def seed_signals(pair, fit, features, matches, shapes, cfg):
    a,b=pair;stats=fit['stats'];m=matches[pair];ids=fit['arrays']['kept_match_indices']
    essential=fit['arrays']['essential_mask']
    cov=[coverage(features[v][0][m[essential,side]],shapes[v]) for side,v in enumerate(pair)]
    links=[]
    if fit['status']=='success':
        obs={(v,int(m[row,side])):point for point,row in enumerate(ids) for side,v in enumerate(pair)}
        for v in range(len(features)):
            if v in pair:continue
            candidates=[(kn,obs[old,int(ko)]) for old in pair for ko,kn in directed_pairs(matches,old,v) if (old,int(ko)) in obs]
            links.append({'view':v,'unambiguous_landmarks':len(collect_links(candidates))})
    return {'pair':list(pair),'status':fit['status'],'reason':fit['reason'],
        'essential_inliers':stats['essential_inliers'],'inlier_fraction':stats['essential_inliers']/len(m) if len(m) else None,
        'coverage':min(cov) if all(v is not None for v in cov) else None,
        'reachable_other_views':sum(v['unambiguous_landmarks']>=cfg['pnp_min_inliers'] for v in links),
        'other_view_links':links,'kept_points':stats['kept_points'],
        'positive_fraction':stats.get('positive_fraction'),
        'parallax_deg':stats.get('accepted_median_parallax_deg'),
        'homography_fraction':stats['homography_inliers']/len(m) if len(m) else None}


def rank_seeds(signals,strategy):
    if strategy not in ('inliers','quality','quality_without_graph'):raise ValueError('unknown ranking strategy')
    def key(s):
        result=[s['status']!='success']
        if strategy=='quality':result.append(-s['reachable_other_views'])
        if strategy!='inliers':result += [s['coverage'] is None,-s['coverage'] if s['coverage'] is not None else 0]
        return tuple(result+[-s['essential_inliers'],tuple(s['pair'])])
    return sorted(signals,key=key)


def verify_candidates(features,matches,ks,shapes,cfg,msac):
    fits={};signals=[];start=time.perf_counter()
    for pair in combinations(range(len(features)),2):
        a,b=pair;m=matches[pair]
        fit=reconstruct_new(features[a][0][m[:,0]],features[b][0][m[:,1]],ks[a],ks[b],cfg,msac)
        fits[pair]=fit;signals.append(seed_signals(pair,fit,features,matches,shapes,cfg))
    return fits,signals,time.perf_counter()-start


def initialize(images,ks,features,matches,cfg,fits,signals,strategy,fixed,max_attempts):
    """Retry only rejected seeds; successful but wrong poses cannot be recognized by GT."""
    if type(max_attempts)!=int or max_attempts<1:raise ValueError('positive retry budget required')
    order=([next(s for s in signals if s['pair']==list(fixed))] if strategy=='fixed' else rank_seeds(signals,strategy)[:max_attempts])
    attempts=[]
    for s in order:
        pair=tuple(s['pair']);attempts.append({'pair':list(pair),'status':s['status'],'reason':s['reason']})
        if s['status']=='success':
            result=reconstruct_incremental(images,ks,pair,cfg,features=features,pair_matches=matches,seed_result=fits[pair])
            return result,{'selected_seed':list(pair),'attempts':attempts,'ranking':order}
    return ({},np.empty((0,3)),{},[],{},[]),{'selected_seed':None,'attempts':attempts,'ranking':order}


def refine_map(result,ks,features,seed,ba_cfg):
    """Use the existing teaching BA solver, preserving rejected candidates/fallback."""
    from .bundle_adjustment import prepare,solve,decode
    poses,xyz,obs,steps,debug,origins=result
    if not poses:return result,{'accepted':False,'reason':'no_seeded_map','wall_seconds':0.},{}
    views=list(poses);lookup={v:i for i,v in enumerate(views)}
    table=np.array([(lookup[v],key,p) for (v,key),p in sorted(obs.items())],int).reshape(-1,3)
    uv=np.array([features[views[v]][0][key] for v,key,p in table])
    problem=prepare([poses[v][0] for v in views],[poses[v][1] for v in views],[ks[v] for v in views],xyz,table[:,0],table[:,2],uv,[lookup[v] for v in seed])
    delta,stats=solve(problem,ba_cfg);r,t,x=decode(problem,delta)
    candidate={'R':r,'t':t,'xyz':x,'views':np.array(views)}
    if stats['accepted']:result=({v:(r[i],t[i]) for i,v in enumerate(views)},x,obs,steps,debug,origins)
    return result,stats,candidate
