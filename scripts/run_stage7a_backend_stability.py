"""Fixed ten-seed backend scorecards; candidate tradeoffs are decided by user."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
from georecon.matching.backend_stability import reconstruct_profile, scorecards

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('backend_control_runner',ROOT/'scripts/run_stage7a_backend_control.py')
legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(legacy)
read,write,digest=legacy.read,legacy.write,legacy.digest
FROZEN_SHA='08a5b8f6a213423d08aaafa6d014d0709c7c296bf71eb52f388aa75816b941fc'
DENIED_ROOTS=()
ACCESSES=set()


def guard(event,args):
    legacy.guard(event,args)
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=Path(os.fsdecode(args[0])).resolve()
        if any(name.is_relative_to(p) for p in DENIED_ROOTS):
            raise PermissionError('worker cannot read source-run or original cache')
        ACCESSES.add(str(name))


def source_files(config):
    return [Path(__file__).resolve(),ROOT/'src/georecon/matching/backend_stability.py',
            ROOT/'tests/test_stage7a_backend_stability.py',config,
            ROOT/'scripts/run_stage7a_backend_control.py',ROOT/'src/georecon/matching/backend_control.py',
            ROOT/'src/georecon/teaching/two_view.py',ROOT/'configs/teaching/stage6_two_view.json',
            ROOT/'src/georecon/matching/metrics.py',ROOT/'src/georecon/matching/evaluate.py']


def contracts(out,config):
    if out.exists():raise ValueError('refuse overwrite of contract output')
    out.mkdir(parents=True);start=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_stage7a_backend_stability.py')
    with (out/'tests.txt').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'passed':result.wasSuccessful() and result.testsRun>0,'reported':result.testsRun,
        'skipped':[{'test':t.id(),'reason':r} for t,r in result.skipped],
        'failures':len(result.failures),'errors':len(result.errors),
        'profile_entry_calls':getattr(sys.modules.get('test_stage7a_backend_stability'),'CALLS',0),
        'seconds':time.perf_counter()-start,
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_files(config)}}
    write(out/'contracts.json',report);print(json.dumps(report,indent=2))
    return report['passed'] and not report['skipped']


def prepare(source_run,cache,out,config,report):
    import cv2
    cfg=read(config)
    if digest(config)!=FROZEN_SHA:raise ValueError('frozen profile changed')
    for name,sha in cfg['source_locks'].items():
        if digest(source_run/name)!=sha:raise ValueError('incumbent source changed')
    source=read(source_run/'bundle.json');provenance=read(source_run/'provenance.json')
    for name,sha in provenance['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('incumbent implementation changed')
    if not read(source_run/'session.json')['complete'] or len(source['cases'])!=46:raise ValueError('incomplete incumbent')
    locked=read(source_run/'cache_lock.json')
    if hashlib.sha256(json.dumps(locked,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=cfg['fixed_cache_aggregate_sha256']:raise ValueError('input aggregate changed')
    for name,sha in locked.items():
        if digest(cache/name)!=sha:raise ValueError('original input changed')
    for name,sha in read(cache/'input_hashes.json').items():
        if digest(ROOT/name)!=sha:raise ValueError('original DTU/calibration mapping changed')
    runtime={'python':sys.version.split()[0],'numpy':np.__version__,'cv2':cv2.__version__}
    if any(v!=cfg['runtime_lock'][k] for k,v in runtime.items()):raise ValueError('runtime changed')
    so=list(Path(cv2.__file__).parent.glob('*.so'))
    if len(so)!=1 or digest(so[0])!=cfg['runtime_lock']['cv2_binary_sha256']:raise ValueError('OpenCV binary changed')
    if digest(ROOT/'src/georecon/teaching/two_view.py')!=cfg['geometry_source_sha256'] or digest(ROOT/'configs/teaching/stage6_two_view.json')!=cfg['geometry_config_sha256']:raise ValueError('legacy geometry changed')
    cert=read(report)
    if not cert['passed'] or cert['skipped'] or cert['profile_entry_calls']>cfg['budget']['new_profile_contract_calls_max']:raise ValueError('contracts incomplete')
    for name,sha in cert['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('tested sources changed')
    cases=[]
    for c in source['cases']:
        # Resolve within explicitly provided source-run, not its historical host path.
        path=source_run/'inputs'/f"{c['case_id']}.npz"
        if digest(path)!=c['input_sha256']:raise ValueError('prepared full matches changed')
        target=out/'inputs'/path.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
        cases.append({**c,'input':str(target)})
    bundle={'cases':cases,'backend':source['backend'],'config':cfg,'output':str(out),
            'denied_read_roots':[str(source_run),str(cache)]}
    write(out/'bundle.json',bundle)
    hashes={}
    for p in source_files(config):
        name=str(p.relative_to(ROOT));target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);hashes[name]=digest(p)
    write(out/'provenance.json',{'source_sha256':hashes,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runtime':runtime,'config_sha256':digest(config),'source_run_sha256':cfg['source_locks'],'cache_lock_sha256':cfg['fixed_cache_aggregate_sha256'],
        'authority':cfg['authority'],'contract_sha256':digest(report)})
    shutil.copyfile(report,out/'contracts.json')
    return bundle


def worker(bundle):
    import cv2
    cv2.setNumThreads(8);cv2.ocl.setUseOpenCL(False)
    out=Path(bundle['output']);cfg=bundle['config'];count=0
    for index,c in enumerate(bundle['cases']):
        if digest(c['input'])!=c['input_sha256']:raise ValueError('fit input changed')
        with np.load(c['input'],allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        shift=index%len(cfg['arms']);order=cfg['arms'][shift:]+cfg['arms'][:shift]
        for arm in order:
            matcher,profile=arm.split('_')
            folder=out/'fits'/c['case_id']/arm;folder.mkdir(parents=True,exist_ok=False)
            write(out/'progress.json',{'case_id':c['case_id'],'arm':arm,'started':time.time(),'done':False})
            records=[]
            for seed in cfg['seeds']:
                start=time.perf_counter()
                v=reconstruct_profile(arrays[matcher+'_pixels_first'],arrays[matcher+'_pixels_second'],*c['K'],
                    dict(bundle['backend'],seed=seed),profile,cfg['profiles'][profile])
                elapsed=time.perf_counter()-start;count+=1;saved=dict(v['arrays'])
                if v['status']=='success':saved.update(accepted_R=v['R'],accepted_t=v['t'])
                path=folder/f'seed_{seed}.npz';np.savez_compressed(path,**saved)
                records.append({'seed':seed,'status':v['status'],'reason':v['reason'],'statistics':v['stats'],'geometry_seconds':elapsed,
                                'arrays_sha256':digest(path),'E_backend':v['E_backend']})
                write(folder/'record.json',{'rows':records,'input_sha256':c['input_sha256']})
            write(out/'progress.json',{'case_id':c['case_id'],'arm':arm,'done':True})
    write(out/'worker_complete.json',{'reconstruction_calls':count,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'gpu_compute':False,'cuda_allocated_bytes':0,'opencl':cv2.ocl.useOpenCL(),'opencv_threads':cv2.getNumThreads(),
        'file_accesses':sorted(ACCESSES)})


def run(source_run,cache,out,config,report):
    if out.exists():raise ValueError('refuse to overwrite stability run')
    out.mkdir(parents=True);start=time.perf_counter();proc=None;failure=None
    try:
        bundle=prepare(source_run,cache,out,config,report);budget=bundle['config']['budget']
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8'}
        with (out/'worker.stdout.txt').open('w') as stdout,(out/'worker.stderr.txt').open('w') as stderr:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--output',str(out)],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            while proc.poll() is None:
                if time.perf_counter()-start>budget['session_seconds']:raise TimeoutError('session limit')
                progress=out/'progress.json'
                if progress.exists():
                    p=read(progress)
                    if not p['done'] and time.time()-p['started']>budget['case_arm_seconds']:raise TimeoutError('case-arm limit')
                status=Path(f'/proc/{proc.pid}/status')
                if status.exists():
                    rss=next((int(l.split()[1])*1024 for l in status.read_text().splitlines() if l.startswith('VmRSS:')),0)
                    if rss>budget['rss_gib']*1024**3:raise MemoryError('RSS limit')
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>budget['artifacts_gib']*1024**3:raise MemoryError('artifact limit')
                time.sleep(.1)
            if proc.returncode:raise RuntimeError('worker failed; retain stderr and partial predictions')
    except Exception as err:failure={'type':type(err).__name__,'error':repr(err),'traceback':traceback.format_exc()}
    finally:
        if proc is not None and proc.poll() is None:proc.kill();proc.wait()
    session={'complete':failure is None,'failure':failure,'wall_seconds':time.perf_counter()-start,'worker_exit_code':proc.returncode if proc else None,'gpu_compute':False}
    write(out/'session.json',session);print(json.dumps(session,indent=2));return failure is None


def evaluate(source_run,cache,out):
    from georecon.matching.backend_control import compare_legacy
    from georecon.matching.metrics import pose_metrics
    from georecon.matching.evaluate import truth_for_case
    start=time.perf_counter()
    if (out/'evaluation.json').exists():raise ValueError('refuse overwrite of evaluation')
    bundle=read(out/'bundle.json');cfg=bundle['config'];provenance=read(out/'provenance.json')
    for name,sha in provenance['source_sha256'].items():
        if digest(ROOT/name)!=sha or digest(out/'sources'/name)!=sha:raise ValueError('source changed')
    for name,sha in cfg['source_locks'].items():
        if digest(source_run/name)!=sha:raise ValueError('source-run changed')
    for name,sha in read(source_run/'cache_lock.json').items():
        if digest(cache/name)!=sha:raise ValueError('original cache changed')
    for name,sha in read(cache/'input_hashes.json').items():
        if digest(ROOT/name)!=sha:raise ValueError('truth/input mapping changed')
    protocol=read(ROOT/'configs/experiments/stage7a_protocol_v1.json');rows=[];mismatches=[];compared=0
    for c in bundle['cases']:
        truth=truth_for_case(ROOT,protocol,c)
        for arm in cfg['arms']:
            folder=out/'fits'/c['case_id']/arm
            record=read(folder/'record.json') if (folder/'record.json').exists() else {'rows':[]}
            for seed in cfg['seeds']:
                rec=next((r for r in record['rows'] if r['seed']==seed),None)
                pose=pose_metrics(*truth,None,None,False);arrays={}
                if rec:
                    path=folder/f'seed_{seed}.npz'
                    if digest(path)!=rec['arrays_sha256']:raise ValueError('prediction changed')
                    with np.load(path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
                    if rec['status']=='success':pose=pose_metrics(*truth,arrays['accepted_R'],arrays['accepted_t'],True)
                    if seed in [42,43,44] and not arm.endswith('_msac'):
                        oldarm=arm[0]+('_old' if arm.endswith('_old') else '_new')
                        oldfolder=source_run/'fits'/c['case_id']/oldarm
                        oldrec=next(r for r in read(oldfolder/'record.json')['rows'] if r['seed']==seed)
                        oldpath=oldfolder/f'seed_{seed}.npz'
                        if digest(oldpath)!=oldrec['arrays_sha256']:raise ValueError('historical prediction changed')
                        with np.load(oldpath,allow_pickle=False) as f:oldarrays={k:f[k] for k in f.files}
                        diff=compare_legacy({'status':rec['status'],'reason':rec['reason'],'stats':rec['statistics']},arrays,oldrec,oldarrays);compared+=1
                        if diff:mismatches.append({'case_id':c['case_id'],'arm':arm,'seed':seed,'differences':diff})
                rows.append({'case_id':c['case_id'],'scan':c['scan'],'conditions':c['conditions'],'arm':arm,'seed':seed,
                    'status':rec['status'] if rec else 'not_run','reason':rec['reason'] if rec else 'missing_record','pose':pose,
                    'geometry_seconds':rec['geometry_seconds'] if rec else None,'statistics':rec['statistics'] if rec else None})
        if time.perf_counter()-start>cfg['budget']['evaluation_seconds']:raise TimeoutError('evaluation budget')
    cards=scorecards(rows,bundle['cases'],cfg);session=read(out/'session.json')
    valid=session['complete'] and not mismatches and all(r['pose']['GT_pose_defined'] for r in rows)
    evaluation={'valid_complete':valid,'rows':rows,**cards,'session':session,'reproduction':{'passed':not mismatches,'compared_slots':compared,'mismatches':mismatches},'evaluation_seconds':time.perf_counter()-start}
    write(out/'evaluation.json',evaluation)
    with (out/'pairs.csv').open('w',newline='') as f:
        flat=[{**{k:r[k] for k in ['case_id','scan','arm','seed','status','reason','geometry_seconds']},**{k:r['pose'][k] for k in ['correct','accepted','false_accept','joint_deg','rotation_deg','translation_deg']}} for r in rows]
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    print(json.dumps({'valid_complete':valid,'reproduction':evaluation['reproduction'],'notices':cards['notices']},indent=2))


def main():
    global DENIED_ROOTS
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['contracts','run','worker','evaluate'])
    p.add_argument('--output',required=True);p.add_argument('--source-run');p.add_argument('--cache');p.add_argument('--config');p.add_argument('--contract-report')
    a=p.parse_args();out=Path(a.output).resolve()
    if a.operation=='worker':
        bundle=read(out/'bundle.json');DENIED_ROOTS=tuple(Path(p) for p in bundle['denied_read_roots']);sys.addaudithook(guard)
        try:worker(bundle)
        except Exception:write(out/'worker_failure.json',{'traceback':traceback.format_exc()});raise
    elif a.operation=='contracts':
        if not a.config:p.error('contracts requires config')
        raise SystemExit(0 if contracts(out,Path(a.config).resolve()) else 1)
    elif a.operation=='run':
        if not all([a.source_run,a.cache,a.config,a.contract_report]):p.error('run requires explicit source-run,cache,config,contract-report')
        raise SystemExit(0 if run(Path(a.source_run).resolve(),Path(a.cache).resolve(),out,Path(a.config).resolve(),Path(a.contract_report).resolve()) else 1)
    else:
        if not a.source_run or not a.cache:p.error('evaluate requires source-run and cache')
        evaluate(Path(a.source_run).resolve(),Path(a.cache).resolve(),out)


if __name__=='__main__':main()
