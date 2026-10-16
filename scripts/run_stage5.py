"""Replay the frozen scene policy; all upstream costs are accounted, never hidden."""
from pathlib import Path
import subprocess
import sys
import time
from georecon.data.protocol import read_json,write_json,sha256,validate_result
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5';OUT=ROOT/'outputs/stage5_v1'


def run():
    lock=read_json(DOC/'input_lock.json');fit=read_json(DOC/'fit_report.json');policy=ROOT/'configs/selection/stage5_v1.json'
    assert sha256(policy)==fit['policy_sha256'] and sha256(DOC/'input_lock.json')==fit['input_lock_sha256']
    for case in lock['cases']:
        bundle=ROOT/case['bundle'];output=OUT/case['scene_id']
        if output.exists():raise ValueError('run exists; never retry implicitly')
        for name,digest in case['files'].items():assert sha256(bundle/name)==digest
        OUT.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','georecon.selection.run',str(bundle),str(policy),str(output)]
        start=time.perf_counter()
        with (OUT/(case['scene_id']+'_worker.txt')).open('w') as f:process=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        if process.returncode:raise RuntimeError('selector failed; see preserved worker log')
        wall=time.perf_counter()-start
        branches={}
        for method in ['colmap','vggt_raw','vggt_ba']:
            root=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/case['scene_id']/method
            branches[method]={'invocation':read_json(root/'benchmark_invocation.json'),'memory_bytes':read_json(root/'prediction.json')['memory_bytes']}
        diag=read_json(ROOT/'outputs/stage4_v1'/case['scene_id']/'invocation.json')
        upstream=sum(b['invocation']['full_cli_wall_seconds'] for b in branches.values())
        costs={'execution_model':'eager snapshots; sum of historical measured stages plus new selector process, not a freshly timed end-to-end model run','baseline_branches':branches,'shared_diagnostics_invocation':diag,'new_selector_process_seconds':wall,'accounted_eager_total_seconds':upstream+diag['full_process_wall_seconds']+wall}
        write_json(output/'cost_accounting.json',costs)
        prediction=read_json(output/'prediction.json')
        prediction['timing_seconds'].update(upstream_all_branches_cli=upstream,upstream_shared_diagnostics_process=diag['full_process_wall_seconds'],new_selector_process=wall,total=costs['accounted_eager_total_seconds'])
        write_json(output/'prediction.json',validate_result(prediction))
        write_json(output/'invocation.json',{'command':command,'returncode':process.returncode,'full_process_wall_seconds':wall,'input_lock_sha256':sha256(DOC/'input_lock.json'),'policy_sha256':sha256(policy)})
        print(case['scene_id'],read_json(output/'decision.json')['selected_method'],flush=True)


if __name__=='__main__':run()
