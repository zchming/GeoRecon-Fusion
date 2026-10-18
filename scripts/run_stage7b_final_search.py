"""Bounded final PnP search on explicit, existing calibrated inference bundles.

Predict never reads camera references. Experimental PnP injection is serial and
scoped; all legacy sources and commands remain unchanged. Outputs refuse reuse.
"""
import argparse
from contextlib import nullcontext
from itertools import combinations
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())
def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path, allow_pickle=False) as data: return {k: data[k] for k in data.files}


def predict(args):
    from georecon.teaching.initialization import initialize, verify_candidates, refine_map
    from georecon.teaching.pnp_refinement import refined_pnp, ap3p_pnp
    cfg = read(args.config)
    for path, digest in cfg['algorithm_locks'].items():
        if sha(ROOT/path) != digest: raise ValueError('frozen algorithm changed: '+path)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', cfg)
    sources = list(cfg['algorithm_locks'])+[str(Path(__file__).relative_to(ROOT))]
    for name in sources:
        path = out/'sources'/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, path)
    write(out/'freeze.json', {'config_sha256': sha(args.config), 'source_sha256': {n: sha(ROOT/n) for n in sources}, 'unix': time.time(), 'development_only': args.development, 'workers':args.workers, 'selected_candidate_lock': read(args.selection) if args.selection else None})
    bundles = []
    for source in args.bundle:
        source = source.resolve(); b = read(source/'inference.json')
        for c in b['cases']:
            if 'scan' in c:
                c.update(family='dtu', scene='scan'+str(c['scan']))
            if args.development and (c['family'] != 'dtu' or c['scan'] not in [1, 6]): continue
            if not args.development and c['family']=='dtu' and c['scan'] in [1,6]: continue
            bundles.append((source, b, c))
    sanitized = {'cases': [{'source': str(s), 'case': c} for s,b,c in bundles]}
    write(out/'input_manifest.json', sanitized)
    hashes = {}
    for source, bundle, case in bundles:
        paths = [source/'inference.json']
        paths += [Path(bundle['feature_paths'][k]) for k in case['keys']]
        paths += [Path(bundle['images'][k]['path']) for k in case['keys']]
        paths += [Path(bundle['pair_paths'][arm+':'+a+':'+b]) for a,b in combinations(case['keys'],2) for arm in cfg['frontends']]
        for path in paths:
            if str(path) not in hashes: hashes[str(path)] = sha(path)
    write(out/'input_hashes.json', hashes)
    reads = set()
    def guard(event, data):
        if event in ('socket.connect', 'socket.getaddrinfo'): raise PermissionError('network forbidden in prediction')
        if event=='open' and isinstance(data[0], (str, bytes)):
            name = os.fsdecode(data[0]); lower = name.lower()
            if any(w in lower for w in ['truth.json', 'reference.json', 'evaluation.json', '/raw/', '_par.txt', '.camera', 'groundtruth', 'cameras.txt', 'images.txt']):
                raise PermissionError('reference poses forbidden during prediction')
            reads.add(name)
    # Preload numerical implementation before installing the reference access guard.
    import scipy.optimize
    import cv2
    cv2.setNumThreads(8); cv2.ocl.setUseOpenCL(False)
    sys.addaudithook(guard)
    solvers = {'legacy': None, 'lm': refined_pnp, 'ap3p': ap3p_pnp, 'ap3p_lm': partial(refined_pnp, base_solver=ap3p_pnp)}
    def compute(jobs, worker_id):
        start = time.perf_counter(); done = 0
        for source, bundle, case in jobs:
            keys = case['keys']; n = len(keys)
            features = [(load(bundle['feature_paths'][k])['keypoints'], None) for k in keys]
            ks = [np.array(bundle['images'][k]['K']) for k in keys]
            shapes = [(bundle['images'][k]['width'], bundle['images'][k]['height']) for k in keys]
            for arm in cfg['frontends']:
                matches = {(i,j): load(bundle['pair_paths'][arm+':'+keys[i]+':'+keys[j]])['indices'] for i,j in combinations(range(n),2)}
                for seed in cfg['geometry_seeds']:
                    geometry = {**bundle['geometry'], 'seed': seed}
                    fits, signals, seconds = verify_candidates(features, matches, ks, shapes, geometry, bundle['msac'])
                    for ranking in cfg['rankings']:
                        for method in cfg['pnp_candidates']:
                            folder = out/'predictions'/case['id']/arm/str(seed)/(ranking+'_'+method)
                            folder.mkdir(parents=True, exist_ok=False)
                            ts = time.perf_counter(); audits = []
                            def solver(*a, **kw):
                                fit = solvers[method](*a, **kw)
                                audits.append({k: v for k,v in fit.items() if not isinstance(v,np.ndarray) and k not in ('R','t')})
                                return fit
                            context = patch('georecon.teaching.incremental.pnp', solver) if method != 'legacy' else nullcontext()
                            try:
                                with context:
                                    result, selection = initialize([None]*n, ks, features, matches, geometry, fits, signals, ranking, [0,2], 3)
                                before = {str(v): {'R':r.tolist(), 't':t.tolist()} for v,(r,t) in result[0].items()}
                                result, ba, candidate = refine_map(result, ks, features, selection['selected_seed'], bundle['ba'])
                                poses, xyz, obs, steps, debug, origins = result
                                np.savez_compressed(folder/'map.npz', xyz=xyz, observations=np.array([(v,k,p) for (v,k),p in sorted(obs.items())],int).reshape(-1,3))
                                np.savez_compressed(folder/'ba_candidate.npz', **candidate)
                                record = {'status':'complete', 'case':case, 'source':str(source), 'arm':arm, 'seed':seed, 'ranking':ranking, 'method':method,
                                          'selection':selection, 'signals':signals, 'steps':steps, 'pnp_audits':audits, 'before_poses':before, 'BA':ba,
                                          'poses':{str(v):{'R':r.tolist(), 't':t.tolist()} for v,(r,t) in poses.items()}, 'points':len(xyz),
                                          'map_seconds':time.perf_counter()-ts, 'shared_candidate_seconds':seconds}
                            except Exception as e:
                                record = {'status':'process_error','case':case,'source':str(source),'arm':arm,'seed':seed,'ranking':ranking,'method':method,'poses':{},'before_poses':{},'points':0,'error':repr(e),'traceback':traceback.format_exc()}
                            write(folder/'record.json',record); done += 1
                            if time.perf_counter()-start > cfg['budget']['session_seconds']: raise TimeoutError('fixed session budget exceeded')
                            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > cfg['budget']['rss_gib']*1024**3: raise MemoryError('RSS budget')
                    print(case['id'], arm, seed, flush=True)
        write(out/f'worker_{worker_id}.json', {'complete':True,'runs':done,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu_bytes':0,'gt_guard':True,'file_accesses':sorted(reads)})
    parent_start = time.perf_counter()
    if args.workers > 1:
        import multiprocessing
        context = multiprocessing.get_context('fork')
        children = [context.Process(target=compute, args=(bundles[i::args.workers], i)) for i in range(args.workers)]
        for child in children: child.start()
        for child in children: child.join()
        if any(child.exitcode != 0 for child in children):
            write(out/'session_failure.json', {'reason':'case worker failed','exitcodes':[c.exitcode for c in children]})
            raise RuntimeError('case worker failed; preserve all outputs')
    else:
        compute(bundles, 0)
    workers = [read(out/f'worker_{i}.json') for i in range(args.workers)]
    write(out/'session.json', {'complete':all(w['complete'] for w in workers), 'runs':sum(w['runs'] for w in workers),
                             'seconds':time.perf_counter()-parent_start, 'peak_rss_bytes':max(w['peak_rss_bytes'] for w in workers),
                             'sum_worker_peak_rss_bytes':sum(w['peak_rss_bytes'] for w in workers),'workers':args.workers,
                             'gpu_bytes':0,'gt_guard':True,'file_accesses':sorted(set(n for w in workers for n in w['file_accesses']))})



def evaluate(args):
    from georecon.geometry.camera import Camera, relative_pose
    from georecon.matching.metrics import pose_metrics
    out = args.output.resolve()
    if (out/'evaluation.json').exists(): raise ValueError('evaluation already exists')
    cfg = read(out/'protocol.json'); rows = []
    sources = {x['source'] for x in read(out/'input_manifest.json')['cases']}
    bundles = {s: read(Path(s)/'inference.json') for s in sources}
    truths = {s: read(Path(s)/'truth.json') for s in sources}
    for path in sorted((out/'predictions').glob('*/*/*/*/record.json')):
        rec = read(path); c = rec['case']; b = bundles[rec['source']]; truth = truths[rec['source']]
        truth_key = str(c['scan']) if c['family']=='dtu' else c['scene']
        gt = [Camera.from_dict(truth[truth_key][v]) if truth[truth_key].get(v) is not None else None for v in c['view_ids']]
        cams = {int(v): Camera(np.array(b['images'][c['keys'][int(v)]]['K']), np.array(p['R']), np.array(p['t']), b['images'][c['keys'][int(v)]]['width'], b['images'][c['keys'][int(v)]]['height']) for v,p in rec['poses'].items()}
        pairs=[]
        for i,j in combinations(range(len(gt)),2):
            if gt[i] is None or gt[j] is None:
                pairs.append({'pair':[i,j],'accepted':i in cams and j in cams,'correct':False,'false_accept':False,'joint_deg':None,'auc':{str(t):0. for t in [5,10,20]},'unavailable':True}); continue
            r,t = relative_pose(cams[i],cams[j]) if i in cams and j in cams else (None,None)
            pairs.append({'pair':[i,j],**pose_metrics(gt[i],gt[j],r,t,r is not None),'unavailable':False})
        selected=rec.get('selection',{}).get('selected_seed'); initial=None
        if selected:
            a,bv=selected
            if gt[a] is not None and gt[bv] is not None:
                pa,pb=rec['before_poses'][str(a)],rec['before_poses'][str(bv)]
                rr=np.array(pb['R'])@np.array(pa['R']).T; tt=np.array(pb['t'])-rr@np.array(pa['t'])
                initial=pose_metrics(gt[a],gt[bv],rr,tt,True)
        rows.append({'case':c['id'],'family':c['family'],'scene':c['scene'],'condition':c['condition'],'arm':rec['arm'],'seed':rec['seed'],'ranking':rec['ranking'],'method':rec['method'],'status':rec['status'],
                     'planned_pairs':len(pairs),'unavailable_pairs':sum(p['unavailable'] for p in pairs),'registered':len(cams),'accurate_scene':all(p['correct'] for p in pairs),'correct_init':bool(initial and initial['correct']),
                     'correct_cameras':sum(v in cams and all(p['correct'] for p in pairs if v in p['pair']) for v in range(len(gt))),
                     'accepted_pairs':sum(p['accepted'] for p in pairs),'false_pairs':sum(p['false_accept'] for p in pairs),'severe_pairs':sum(p['joint_deg'] is not None and p['joint_deg']>20 for p in pairs),
                     **{f'auc{t}':float(np.mean([p['auc'][str(t)] for p in pairs])) for t in [5,10,20]},'points':rec['points'],'seconds':rec.get('map_seconds',0)+rec.get('shared_candidate_seconds',0),'BA':rec.get('BA',{}).get('reason'),'selected_seed':selected,'pairs':pairs,'record':str(path)})
    summaries=[]
    for family in sorted({r['family'] for r in rows}):
        for arm in cfg['frontends']:
            for ranking in cfg['rankings']:
                for method in cfg['pnp_candidates']:
                    rs=[r for r in rows if (r['family'],r['arm'],r['ranking'],r['method'])==(family,arm,ranking,method)]
                    angles=[p['joint_deg'] for r in rs for p in r['pairs'] if p['joint_deg'] is not None]
                    summaries.append({'family':family,'arm':arm,'ranking':ranking,'method':method,'runs':len(rs),'scenes':len({r['scene'] for r in rs}),
                                      **{k:float(np.mean([r[k] for r in rs])) for k in ['auc5','auc10','auc20','accurate_scene','correct_cameras','registered','correct_init','seconds']},
                                      **{k:sum(r[k] for r in rs) for k in ['planned_pairs','unavailable_pairs','accepted_pairs','false_pairs','severe_pairs']},'p95_deg':float(np.percentile(angles,95)) if angles else None,'max_deg':max(angles) if angles else None})
    write(out/'evaluation.json',{'rows':rows,'summaries':summaries,'complete':len(rows)==len(read(out/'input_manifest.json')['cases'])*len(cfg['frontends'])*len(cfg['geometry_seeds'])*len(cfg['rankings'])*len(cfg['pnp_candidates']) and all(r['status']=='complete' for r in rows)})
    import csv
    with (out/'summaries.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(json.dumps({'summaries':[s for s in summaries if s['arm']=='C' and s['ranking']=='inliers']},indent=2))


def select(args):
    out=args.output.resolve();e=read(out/'evaluation.json')
    if not read(out/'freeze.json')['development_only']:raise ValueError('only development may select candidate')
    rows=[s for s in e['summaries'] if s['arm']=='C' and s['ranking']=='inliers'];base=next(s for s in rows if s['method']=='legacy')
    eligible=[s for s in rows if s['method']!='legacy' and s['auc10']>base['auc10'] and s['false_pairs']<=base['false_pairs'] and s['severe_pairs']<=base['severe_pairs'] and s['accurate_scene']>=base['accurate_scene']]
    winner=sorted(eligible,key=lambda s:(-s['auc10'],s['method']))[0]['method'] if eligible else 'lm'
    target=out/'selection.json'
    if target.exists():raise ValueError('selection already frozen')
    write(target,{'selected':winner,'development_winner':bool(eligible),'eligible':[r['method'] for r in eligible],'rule':read(out/'protocol.json')['development_selection'],'development_evaluation_sha256':sha(out/'evaluation.json'),'unix':time.time(),'external_predictions_started':False})
    print(target.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['predict','evaluate','select']);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,action='append',default=[]);p.add_argument('--development',action='store_true');p.add_argument('--selection',type=Path);p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if not 1 <= args.workers <= 8:raise ValueError('worker budget1..8')
    if args.mode=='predict' and not args.development and not args.selection:raise ValueError('external prediction requires frozen development selection')
    globals()[args.mode](args)
if __name__=='__main__':main()
