"""Cached four-arm E backend experiment, isolated predictions and offline audit."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from georecon.matching.backend_control import ARMS, reconstruct_new, compare_legacy, summarize, decision

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = '097ada714c3519c26450e91f2d52853d56adb5a87435b79c856530a3fc22fc8f'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def guard(event, args):
    if event in ['socket.connect', 'socket.getaddrinfo']:
        raise PermissionError('offline geometry worker forbids network')
    if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.')):
        raise PermissionError('no model inference in E backend control')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).lower().replace('\\', '/')
        if any(s in name for s in ['reference.json', 'evaluation.json', '/stage3_dtu/', '/stage7a_v1_after_guard_fix/', 'ground_truth', '.pth']):
            raise PermissionError('worker cannot read GT, historical predictions or model weights')


def sources(config):
    return [Path(__file__).resolve(), ROOT/'src/georecon/matching/backend_control.py',
            ROOT/'tests/test_stage7a_backend_control.py', config,
            ROOT/'src/georecon/teaching/two_view.py', ROOT/'configs/teaching/stage6_two_view.json',
            ROOT/'src/georecon/matching/metrics.py', ROOT/'src/georecon/matching/evaluate.py']


def prepare(cache, out, config, contracts):
    import cv2
    cfg = read(config)
    if digest(config) != FROZEN_SHA:
        raise ValueError('frozen backend protocol changed')
    protocol_path = ROOT/'configs/experiments/stage7a_protocol_v1.json'
    if digest(protocol_path) != cfg['input']['case_source_sha256']:
        raise ValueError('original protocol changed')
    for key in ['source', 'config']:
        path = ROOT/cfg['backend']['unchanged_'+key]
        if digest(path) != cfg['backend'][key+'_sha256']:
            raise ValueError('legacy backend changed')
    runtime = {'python': sys.version.split()[0], 'numpy': np.__version__,
               'opencv-python': importlib.metadata.version('opencv-python'), 'cv2': cv2.__version__}
    if any(v != cfg['runtime_lock'][k] for k, v in runtime.items()):
        raise ValueError('runtime differs from frozen design')
    binary = list(Path(cv2.__file__).parent.glob('*.so'))
    if len(binary) != 1 or digest(binary[0]) != cfg['runtime_lock']['cv2_binary_sha256']:
        raise ValueError('cv2 binary differs')
    contract = read(contracts)
    if not contract['passed'] or contract['skipped'] or contract['new_solver_calls'] > 32:
        raise ValueError('synthetic contract suite incomplete/failed')
    for name, sha in contract['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('contract suite tested different sources')
    old, protocol = read(cache/'bundle.json'), read(protocol_path)
    if not read(cache/'session.json')['complete'] or len(old['cases']) != 46:
        raise ValueError('original cache incomplete')
    expected = protocol['input']['cases']
    if [{k: c[k] for k in e} for c, e in zip(old['cases'], expected)] != expected:
        raise ValueError('cached cases/order differ')
    locked = {}
    def lock(p):
        locked[p.relative_to(cache).as_posix()] = digest(p)
    for name in ['bundle.json', 'input_hashes.json', 'implementation.json']:
        lock(cache/name)
    features, records = {}, {}
    for key in sorted(old['images']):
        folder = cache/'features'/key
        lock(folder/'features.npz'); lock(folder/'record.json')
        rec = read(folder/'record.json'); records[key] = rec
        if rec['status'] != 'success' or rec['feature_sha256'] != digest(folder/'features.npz'):
            raise ValueError('feature integrity failed')
        with np.load(folder/'features.npz', allow_pickle=False) as f:
            features[key] = f['keypoints']
    cases = []
    for c in old['cases']:
        arrays = {}
        for arm in ['B', 'C']:
            folder = cache/'main'/c['case_id']/arm
            lock(folder/'matches.npz'); lock(folder/'record.json')
            rec = read(folder/'record.json')
            if rec['status'] != 'success' or rec['matches_sha256'] != digest(folder/'matches.npz') or rec['root_input_digests'] != [records[k]['root_input_digest'] for k in c['image_keys']]:
                raise ValueError('cached match/feature contract failed')
            with np.load(folder/'matches.npz', allow_pickle=False) as f:
                ids = f['indices']
                if ids.dtype.kind not in 'iu' or ids.shape != (len(ids), 2) or np.any(ids < 0):
                    raise ValueError('invalid match indices')
                if len(ids) and not np.array_equal(ids, np.array(sorted(map(tuple, ids)))):
                    raise ValueError('match order changed')
                for side, name in enumerate(['pixels_first', 'pixels_second']):
                    pixels = f[name]
                    np.testing.assert_array_equal(features[c['image_keys'][side]][ids[:, side]], pixels)
                    if not np.isfinite(pixels).all(): raise ValueError('nonfinite pixels')
                    arrays[arm+'_'+name] = pixels
                arrays[arm+'_indices'] = ids
        path = out/'inputs'/f"{c['case_id']}.npz"; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        cases.append({**{k:c[k] for k in ['case_id', 'scan', 'conditions', 'view_ids']},
                      'K': [old['images'][k]['K'] for k in c['image_keys']],
                      'input': str(path), 'input_sha256': digest(path)})
    aggregate = hashlib.sha256(json.dumps(locked, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if len(locked) != 235 or aggregate != cfg['input']['cache_lock']['sha256']:
        raise ValueError('frozen 235-file cache lock differs')
    for name, sha in read(cache/'input_hashes.json').items():
        if digest(ROOT/name) != sha: raise ValueError('historical image/calibration mapping changed')
    source_hashes = {}
    for path in sources(config):
        name = str(path.relative_to(ROOT)); target = out/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        source_hashes[name] = digest(path)
    write(out/'cache_lock.json', locked)
    write(out/'provenance.json', {'base_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': source_hashes, 'runtime': runtime, 'cache_sha256': aggregate,
        'contract_report_sha256': digest(contracts), 'authority':'user authorized continuing frozen backend implementation and controlled run; no commit/push/Stage7B'})
    shutil.copyfile(contracts, out/'contracts.json')
    bundle = {'cases': cases, 'config': cfg, 'backend': read(ROOT/cfg['backend']['unchanged_config']), 'output': str(out)}
    write(out/'bundle.json', bundle)
    return bundle


def worker(bundle):
    from georecon.teaching.two_view import reconstruct
    import cv2
    cv2.ocl.setUseOpenCL(False); cv2.setNumThreads(8)
    out, cfg = Path(bundle['output']), bundle['config']
    count = 0
    for index, case in enumerate(bundle['cases']):
        if digest(case['input']) != case['input_sha256']: raise ValueError('prepared observations changed')
        with np.load(case['input'], allow_pickle=False) as f: arrays = {k:f[k] for k in f.files}
        for arm in cfg['execution_plan']['arm_orders'][index % 4]:
            folder = out/'fits'/case['case_id']/arm; folder.mkdir(parents=True, exist_ok=False)
            write(out/'progress.json', {'case_id': case['case_id'], 'arm':arm, 'started':time.time(), 'done':False})
            rows = []
            match_arm, mode = arm.split('_')
            for seed in cfg['execution_plan']['seeds']:
                start = time.perf_counter()
                p, q = [arrays[match_arm+'_'+key] for key in ['pixels_first', 'pixels_second']]
                settings = dict(bundle['backend'], seed=seed)
                value = (reconstruct(p,q,*case['K'],settings) if mode == 'old' else
                         reconstruct_new(p,q,*case['K'],settings,cfg['backend']['new']['params']))
                seconds = time.perf_counter()-start; count += 1
                saved = dict(value['arrays'])
                if value['status'] == 'success': saved.update(accepted_R=value['R'], accepted_t=value['t'])
                path = folder/f'seed_{seed}.npz'; np.savez_compressed(path, **saved)
                rows.append({'seed':seed, 'status':value['status'], 'reason':value['reason'],
                             'statistics':value['stats'], 'geometry_seconds':seconds, 'arrays_sha256':digest(path)})
                write(folder/'record.json', {'case_id':case['case_id'], 'arm':arm, 'rows':rows, 'input_sha256':case['input_sha256']})
            write(out/'progress.json', {'case_id':case['case_id'], 'arm':arm, 'done':True})
    write(out/'worker_complete.json', {'reconstruction_calls':count, 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'gpu_compute':False, 'gpu_allocated_bytes':0, 'opencv_threads':cv2.getNumThreads(), 'opencl_enabled':cv2.ocl.useOpenCL()})


def run(cache, out, config, contracts):
    if out.exists(): raise ValueError('refuse to overwrite backend experiment output')
    out.mkdir(parents=True); started = time.perf_counter(); failure = None; proc = None
    try:
        bundle = prepare(cache,out,config,contracts)
        budget = bundle['config']['budget']
        env = {**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1',
               'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8'}
        with (out/'worker.stdout.txt').open('w') as stdout, (out/'worker.stderr.txt').open('w') as stderr:
            proc = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--output',str(out)], env=env,cwd=ROOT,stdout=stdout,stderr=stderr)
            while proc.poll() is None:
                if time.perf_counter()-started > budget['session_seconds']: raise TimeoutError('session limit')
                path = out/'progress.json'
                if path.exists():
                    p = read(path)
                    if not p['done'] and time.time()-p['started'] > budget['per_case_arm_all_seeds_seconds']: raise TimeoutError('case-arm limit')
                path = Path(f'/proc/{proc.pid}/status')
                if path.exists():
                    rss = next((int(l.split()[1])*1024 for l in path.read_text().splitlines() if l.startswith('VmRSS:')),0)
                    if rss > budget['process_rss_gib']*1024**3: raise MemoryError('RSS limit')
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > budget['artifacts_gib']*1024**3: raise MemoryError('artifact limit')
                time.sleep(.1)
            if proc.returncode: raise RuntimeError('worker software failure; see preserved stderr')
    except Exception as error:
        failure = {'type':type(error).__name__,'error':repr(error),'traceback':traceback.format_exc()}
    finally:
        if proc is not None and proc.poll() is None: proc.kill(); proc.wait()
    session = {'complete':failure is None,'failure':failure,'wall_seconds':time.perf_counter()-started,
               'worker_exit_code':proc.returncode if proc else None,'gpu_compute':False}
    write(out/'session.json',session); print(json.dumps(session,indent=2))
    return failure is None


def evaluate(cache, out):
    from georecon.matching.evaluate import truth_for_case
    from georecon.matching.metrics import pose_metrics
    started = time.perf_counter()
    if (out/'evaluation.json').exists(): raise ValueError('refuse to overwrite backend evaluation')
    bundle, provenance = read(out/'bundle.json'), read(out/'provenance.json')
    for name, sha in provenance['source_sha256'].items():
        if digest(ROOT/name) != sha or digest(out/'sources'/name) != sha: raise ValueError('implementation changed')
    for name, sha in read(out/'cache_lock.json').items():
        if digest(cache/name) != sha: raise ValueError('cached source changed')
    for name, sha in read(cache/'input_hashes.json').items():
        if digest(ROOT/name) != sha: raise ValueError('historical evaluation input changed')
    protocol = read(ROOT/'configs/experiments/stage7a_protocol_v1.json')
    seeds = bundle['config']['execution_plan']['seeds']; rows = []; mismatches = []
    for case in bundle['cases']:
        if digest(case['input']) != case['input_sha256']: raise ValueError('prepared input changed')
        truth = truth_for_case(ROOT,protocol,case)
        for arm in ARMS:
            folder = out/'fits'/case['case_id']/arm
            rec = read(folder/'record.json') if (folder/'record.json').exists() else {'rows':[]}
            for seed in seeds:
                saved = next((r for r in rec['rows'] if r['seed']==seed),None)
                pose = pose_metrics(*truth,None,None,False)
                arrays = {}
                if saved:
                    path = folder/f'seed_{seed}.npz'
                    if digest(path) != saved['arrays_sha256']: raise ValueError('predictions changed')
                    with np.load(path,allow_pickle=False) as f: arrays = {k:f[k] for k in f.files}
                    if saved['status']=='success': pose = pose_metrics(*truth,arrays['accepted_R'],arrays['accepted_t'],True)
                    if arm.endswith('_old'):
                        historic = cache/'main'/case['case_id']/arm[0]
                        oldrec = next(r for r in read(historic/'record.json')['geometry'] if r['seed']==seed)
                        oldpath = historic/f'geometry_{seed}.npz'
                        if digest(oldpath) != oldrec['arrays_sha256']: raise ValueError('historical geometry changed')
                        with np.load(oldpath,allow_pickle=False) as f: oldarrays = {k:f[k] for k in f.files}
                        differences = compare_legacy({'status':saved['status'],'reason':saved['reason'],'stats':saved['statistics']},arrays,oldrec,oldarrays)
                        if differences: mismatches.append({'case_id':case['case_id'],'arm':arm,'seed':seed,'differences':differences})
                rows.append({'case_id':case['case_id'],'scan':case['scan'],'conditions':case['conditions'],'arm':arm,'seed':seed,
                    'status':saved['status'] if saved else 'not_run','reason':saved['reason'] if saved else 'missing_record',
                    'pose':pose,'geometry_seconds':saved['geometry_seconds'] if saved else None,
                    'statistics':saved['statistics'] if saved else None})
        if time.perf_counter()-started > bundle['config']['budget']['separate_offline_evaluation_seconds']: raise TimeoutError('evaluation limit')
    summary = summarize(rows,bundle['cases'],seeds); session=read(out/'session.json')
    resource_failure = bool(session['failure'] and session['failure']['type'] in ['MemoryError','TimeoutError'])
    verdict = decision(rows,bundle['cases'],seeds,summary,not mismatches and (session['failure'] is None or resource_failure),resource_failure,not session['complete'])
    write(out/'legacy_reproduction.json',{'passed':not mismatches,'compared_slots':sum(r['arm'].endswith('_old') and r['status']!='not_run' for r in rows),'mismatches':mismatches})
    # Keep raw per-slot records, but publish one concise row per case/arm/seed.
    flattened = []
    for r in rows:
        flattened.append({**{k:r[k] for k in ['case_id','scan','arm','seed','status','reason','geometry_seconds']},
            **{k:r['pose'][k] for k in ['accepted','correct','false_accept','rotation_deg','translation_deg','joint_deg']},
            'essential_inliers':r['statistics'].get('essential_inliers') if r['statistics'] else None,
            'homography_inliers':r['statistics'].get('homography_inliers') if r['statistics'] else None,
            'kept_points':r['statistics'].get('kept_points') if r['statistics'] else None})
    with (out/'pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flattened[0]),lineterminator='\n');writer.writeheader();writer.writerows(flattened)
    write(out/'evaluation.json',{'rows':rows,'summary':summary,'decision':verdict,'session':session,'evaluation_seconds':time.perf_counter()-started})
    print(json.dumps({'decision':verdict,'summary':summary},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['run','worker','evaluate']);p.add_argument('--cache');p.add_argument('--output',required=True)
    p.add_argument('--config');p.add_argument('--contract-report')
    a=p.parse_args();out=Path(a.output).resolve()
    if a.operation=='worker':
        bundle=read(out/'bundle.json');sys.addaudithook(guard)
        try: worker(bundle)
        except Exception:
            write(out/'worker_failure.json',{'traceback':traceback.format_exc()});raise
    elif a.operation=='run':
        if not all([a.cache,a.config,a.contract_report]):p.error('run requires explicit cache, config and contract-report')
        raise SystemExit(0 if run(Path(a.cache).resolve(),out,Path(a.config).resolve(),Path(a.contract_report).resolve()) else 1)
    else:
        if not a.cache:p.error('evaluate requires cache')
        evaluate(Path(a.cache).resolve(),out)


if __name__=='__main__':main()
