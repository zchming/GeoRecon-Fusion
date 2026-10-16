"""One dev-only invariance probe; invalid GT sentinels must not affect diagnostics."""
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from georecon.data.protocol import read_json,write_json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/stage4_bundles/dtu_scan1_reduced3';bundle=ROOT/'data/stage4_gt_probe'
output=ROOT/'outputs/stage4_gt_isolation_probe';original=ROOT/'outputs/stage4_v1/dtu_scan1_reduced3'
if bundle.exists() or output.exists():raise ValueError('probe outputs already exist; never overwrite')
shutil.copytree(source,bundle)
(bundle/'reference.json').write_text('INVALID_GT_SENTINEL: this must never be read')
(bundle/'evaluation.json').write_text('INVALID_EVALUATION_SENTINEL')
(bundle/'calibration').mkdir();(bundle/'calibration/pos_001.txt').write_text('INVALID_CALIBRATION')
command=[sys.executable,'-m','georecon.reliability.analyze',str(bundle),str(ROOT/'configs/reliability/stage4_v1.json'),str(output)]
start=time.perf_counter();process=subprocess.run(command,capture_output=True,text=True,check=True);wall=time.perf_counter()-start
(output/'probe_worker.txt').write_text(process.stdout+process.stderr)
a,b=read_json(original/'reliability.json'),read_json(output/'reliability.json')
assert a['methods']==b['methods']
assert read_json(original/'observations.json')==read_json(output/'observations.json')
for name in ['observations.npz','match_signals.npz']:
    with np.load(original/name,allow_pickle=False) as x,np.load(output/name,allow_pickle=False) as y:
        assert x.files==y.files
        assert all(np.array_equal(x[k],y[k],equal_nan=True) for k in x.files)
write_json(ROOT/'docs/stage4/gt_isolation_probe.json',dict(scope='dev-only integration invariance probe, not a benchmark retry',command=command,wall_seconds=wall,invalid_reference_evaluation_and_calibration_present=True,all_method_diagnostics_identical=True,all_match_observations_and_signal_arrays_identical=True,configuration_changed=False,raw_provenance_and_timing_excluded_from_numeric_comparison=True))
print('GT isolation: identical diagnostics with invalid GT/evaluation/calibration files present')
