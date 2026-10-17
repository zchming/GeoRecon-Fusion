"""Run all frozen educational cases as independent guarded CLI processes."""
import os,time,subprocess
from pathlib import Path
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6';OUT=ROOT/'outputs/stage6_two_view_v1'
PYTHON='/home/zchming/.venvs/georecon-reliability/bin/python'


def gpu_snapshot():
    result={}
    for key,args in [('card',['--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits']),('compute',['--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])]:
        try:
            p=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=3)
            result[key]={'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
        except Exception as e:result[key]={'error':type(e).__name__+': '+str(e)}
    return result


def run():
    lock=read_json(DOC/'input_lock.json');cfg=ROOT/'configs/teaching/stage6_two_view.json'
    assert sha256(cfg)==lock['protocol_sha256'];rows=[]
    for case in lock['cases']:
        inp=ROOT/case['input'];assert sha256(inp)==case['input_sha256'];dest=OUT/case['case_id']
        if dest.exists():raise ValueError('preserve existing case, no implicit retry')
        start=time.perf_counter();before=gpu_snapshot()
        cmd=[PYTHON,'-m','georecon.cli','run-teaching-sfm','--input',str(inp),'--config',str(cfg),'--output',str(dest)]
        env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8');proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,cwd=ROOT)
        samples=[]
        # CLI stdout is small (<pipe capacity); capture after completion. GPU never used by CPU SIFT/DLT.
        while proc.poll() is None:
            snapshot=gpu_snapshot();samples.append(snapshot)
            time.sleep(.05)
        stdout,stderr=proc.communicate();elapsed=time.perf_counter()-start;after=gpu_snapshot()
        if not dest.exists():dest.mkdir(parents=True)
        (dest/'worker.stdout.txt').write_text(stdout);(dest/'worker.stderr.txt').write_text(stderr)
        record={'case_id':case['case_id'],'command':cmd,'returncode':proc.returncode,'full_process_wall_seconds_including_sampling':elapsed,'pid':proc.pid,'gpu_before':before,'gpu_samples':samples,'gpu_after':after,
            'measurement_scope':'whole card and exposed compute listing; WSL may not expose per-process memory; CPU algorithm/OpenCL disabled, no CUDA/model allocation','algorithm_GPU_allocated_bytes':0}
        write_json(dest/'process.json',record);rows.append(record)
        if not (dest/'prediction.json').exists():raise RuntimeError('case failed before protocol output; retain process logs')
        pred=read_json(dest/'prediction.json');expected_code=0 if pred['status']=='success' else 1
        if proc.returncode!=expected_code:raise RuntimeError('unexpected worker return code; preserve failure')
        print(case['case_id'],pred['status'],pred['failure_reason'],flush=True)
    write_json(DOC/'execution.json',{'rows':rows,'new_downloads':0,'real_cases':3,'synthetic_or_degeneracy_controls':8,'all_cases_preserved':True})

if __name__=='__main__':run()
