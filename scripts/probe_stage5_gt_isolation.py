"""Prepare invalid sentinels outside the dev-fit guard, then test the guarded selector."""
from pathlib import Path
import shutil
import subprocess
import sys
import time
from georecon.data.protocol import read_json,write_json,sha256
from georecon.selection.run import link_or_copy
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5'
source=ROOT/'data/stage5_bundles/dtu_scan1_reduced3';bundle=ROOT/'data/stage5_gt_probe_attempt02';output=ROOT/'outputs/stage5_gt_probe_attempt02';original=ROOT/'outputs/stage5_v1/dtu_scan1_reduced3'
if bundle.exists() or output.exists():raise ValueError('probe exists; never overwrite')
shutil.copytree(source,bundle,copy_function=lambda a,b:link_or_copy(Path(a),Path(b)))
(bundle/'reference.json').write_text('INVALID_GT_SENTINEL');(bundle/'evaluation.json').write_text('INVALID_EVALUATION_SENTINEL')
(bundle/'calibration').mkdir();(bundle/'calibration/pos_001.txt').write_text('INVALID_CALIBRATION')
command=[sys.executable,'-m','georecon.selection.run',str(bundle),str(ROOT/'configs/selection/stage5_v1.json'),str(output)]
begin=time.perf_counter();p=subprocess.run(command,capture_output=True,text=True,check=True);wall=time.perf_counter()-begin
(output/'probe_worker.txt').write_text(p.stdout+p.stderr)
a,b=read_json(original/'prediction.json'),read_json(output/'prediction.json')
assert read_json(original/'decision.json')==read_json(output/'decision.json')
for key in ['images','world_unit','coordinate_convention','artifacts']:assert a[key]==b[key]
for artifact in b['artifacts'].values():
    if artifact is not None:assert sha256(output/artifact['path'])==artifact['sha256']
write_json(DOC/'gt_isolation_probe.json',{'scope':'DEV-only integration probe, not refitting or benchmark retry','command':command,'wall_seconds':wall,'decision_cameras_and_artifact_hashes_identical':True,'invalid_gt_files_present':True,
                                      'attempt_history':[{'attempt':1,'status':'probe_harness_error_before_selector_execution','reason':'dev-only fit guard blocked writing invalid evaluation.json sentinel; incomplete bundle retained at data/stage5_gt_probe'},{'attempt':2,'status':'success','change':'sentinel preparation in a separate process; selector and frozen policy unchanged'}]})
print('GT-isolation export probe passed; initial harness failure preserved')
