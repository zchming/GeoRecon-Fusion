"""Independent immutable attempts; save actual exits, time, memory and all debug outputs."""
from pathlib import Path
import os,subprocess,sys,time
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_incremental';OUT=ROOT/'outputs/stage6_incremental_v1'
def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    return {'memory_used_total_MiB':p.stdout.strip(),'compute_apps':q.stdout.strip(),'returncode':p.returncode}
def run():
    if OUT.exists():raise ValueError('preserve all attempts')
    lock=read_json(DOC/'input_lock.json');cfg=ROOT/'configs/teaching/stage6_incremental.json';assert sha256(cfg)==lock['config_sha256']
    sources=['src/georecon/teaching/incremental.py','src/georecon/teaching/incremental_run.py','src/georecon/teaching/two_view.py','scripts/run_stage6_incremental.py','scripts/prepare_stage6_incremental.py']
    write_json(DOC/'source_lock.json',{'source_sha256':{p:sha256(ROOT/p) for p in sources},'config_sha256':sha256(cfg),'input_lock_sha256':sha256(DOC/'input_lock.json'),'fixed_before_real_runs':True})
    rows=[]
    for case in lock['cases']:
        assert sha256(ROOT/case['input'])==case['input_sha256'];out=OUT/case['case_id'];log=DOC/'runs'/case['case_id'];log.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-m','georecon.teaching.incremental_run','--input',str(ROOT/case['input']),'--config',str(cfg),'--output',str(out)]
        before=gpu();start=time.perf_counter();samples=[]
        with (log/'stdout.txt').open('w') as stdout,(log/'stderr.txt').open('w') as stderr:
            proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'})
            while proc.poll() is None:
                samples.append(gpu());time.sleep(.1)
        record={'case_id':case['case_id'],'command':command,'exit_code':proc.returncode,'controller_seconds':time.perf_counter()-start,'gpu_before':before,'gpu_samples':samples,'gpu_after':gpu(),'GPU_task_allocated_bytes':0,'profile':'CPU SIFT/EPnP/NumPy; no CUDA/OpenCL; whole-card usage includes desktop; WSL compute listing limited'}
        write_json(log/'process.json',record);rows.append(record)
        if (out/'prediction.json').exists():
            for name in ['prediction.json','steps.json','invocation.json']:write_json(log/name,read_json(out/name))
    write_json(DOC/'execution.json',rows)
if __name__=='__main__':run()
