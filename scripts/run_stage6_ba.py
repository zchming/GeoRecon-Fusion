"""Record every immutable BA attempt, including skips and solver limits."""
from pathlib import Path
import os,subprocess,sys,time
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_ba';OUT=ROOT/'outputs/stage6_ba_v1'
def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True);q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    return {'memory_used_total_MiB':p.stdout.strip(),'compute_apps':q.stdout.strip(),'returncode':p.returncode}
def run():
    if OUT.exists():raise ValueError('preserve every attempt')
    lock=read_json(DOC/'input_lock.json');cfg=ROOT/'configs/teaching/stage6_ba.json';assert sha256(cfg)==lock['config_sha256']
    names=['src/georecon/teaching/bundle_adjustment.py','src/georecon/teaching/ba_run.py','scripts/prepare_stage6_ba.py','scripts/run_stage6_ba.py','tests/test_teaching_ba.py']
    write_json(DOC/'source_lock.json',{'source_sha256':{name:sha256(ROOT/name) for name in names},'config_sha256':sha256(cfg),'input_lock_sha256':sha256(DOC/'input_lock.json'),'frozen_before_real_runs':True});rows=[]
    for case in lock['cases']:
        for name,digest in case['baseline_files_sha256'].items():assert sha256(ROOT/case['baseline']/name)==digest
        out=OUT/case['case_id'];log=DOC/'runs'/case['case_id'];log.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-m','georecon.teaching.ba_run','--input',str(ROOT/case['input']),'--baseline',str(ROOT/case['baseline']),'--config',str(cfg),'--output',str(out)]
        before=gpu();samples=[];start=time.perf_counter()
        with (log/'stdout.txt').open('w') as stdout,(log/'stderr.txt').open('w') as stderr:
            proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'})
            while proc.poll() is None:samples.append(gpu());time.sleep(.2)
        record={'case_id':case['case_id'],'command':command,'exit_code':proc.returncode,'controller_seconds':time.perf_counter()-start,'gpu_before':before,'gpu_samples':samples,'gpu_after':gpu(),'profile':'CPU SciPy sparse least squares; no CUDA/model/OpenCL; whole card is not task allocation; WSL process listing limited'}
        write_json(log/'process.json',record);rows.append(record)
        for name in ['prediction.json','ba.json','invocation.json']:
            if (out/name).exists():write_json(log/name,read_json(out/name))
    write_json(DOC/'execution.json',rows)
if __name__=='__main__':run()
