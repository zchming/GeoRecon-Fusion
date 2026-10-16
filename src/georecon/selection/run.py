"""GT-free scene choice and unchanged reconstruction export from sanitized snapshots."""
import copy
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
from ..data.protocol import read_json,write_json,sha256,result_template,validate_result,relative_file
from ..provenance import code_provenance
from ..reliability.analyze import deny_ground_truth
from .policy import choose,validate_policy


def link_or_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError as exc:
        import errno
        if exc.errno!=errno.EXDEV:raise
        shutil.copyfile(source,target)


def run(bundle,policy_path,output):
    begin=time.perf_counter();bundle=Path(bundle).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('output must be new; selection attempts never overwritten')
    output.mkdir(parents=True,exist_ok=True);sys.addaudithook(deny_ground_truth)
    manifest=read_json(bundle/'manifest.json');fingerprint=sha256(bundle/'manifest.json')
    image_ids=[r['image_id'] for r in manifest['images']]
    policy=validate_policy(read_json(policy_path));diagnostics=read_json(bundle/'reliability.json')
    candidates={};digests={};availability_errors={}
    for method in policy['priority']:
        path=bundle/(method+'.json');availability_errors[method]=[]
        try:
            candidates[method]=read_json(path);digests[method]=sha256(path)
            validate_result(candidates[method])
            checked=set()
            for artifact in candidates[method]['artifacts'].values():
                if artifact is not None and artifact['path'] not in checked:
                    checked.add(artifact['path'])
                    asset=relative_file(bundle/'artifacts'/method,artifact['path'])
                    if not asset.is_file() or sha256(asset)!=artifact['sha256']:
                        availability_errors[method].append('artifact_missing_or_corrupt')
        except (OSError,ValueError,KeyError,TypeError):
            candidates[method]=None;digests[method]=None;availability_errors[method].append('candidate_missing_or_invalid')
    decision=choose(candidates,diagnostics,policy,manifest['scene_id'],fingerprint,image_ids,digests,availability_errors)
    chosen=decision['selected_method']
    if chosen:
        source=candidates[chosen];prediction=copy.deepcopy(source)
        copied={}
        for name,artifact in prediction['artifacts'].items():
            if artifact is None:continue
            path=artifact['path']
            if path not in copied:
                original=relative_file(bundle/'artifacts'/chosen,path)
                if sha256(original)!=artifact['sha256']:raise ValueError('selected artifact digest mismatch')
                target=relative_file(output,'selected_artifacts/'+path)
                link_or_copy(original,target)
                if sha256(target)!=artifact['sha256']:raise ValueError('export changed selected artifact')
                copied[path]='selected_artifacts/'+path
            artifact['path']=copied[path]
        prediction['provenance']={'selected_source_method':chosen,'selected_source_prediction_sha256':digests[chosen],
                                  'selected_source_method_id':source['method_id'],'selected_source_config':source['config'],
                                  'selected_source_provenance':source['provenance'],'camera_and_world_frame':'unchanged from selected baseline; no alignment or mixed cameras',**code_provenance(manifest['kind'])}
    else:
        prediction=result_template(manifest['scene_id'],fingerprint,image_ids,'scene_selector_v1',kind=manifest['kind'])
        prediction.update(status='no_reconstruction',failure_reason='no complete successful candidate with usable geometric evidence')
        prediction['provenance']=code_provenance(manifest['kind'])
    prediction.update(method_id='scene_selector_v1',run_id=output.name,config={'policy':policy,'policy_sha256':sha256(policy_path),'execution':'eager_snapshot_selection'},versions={'python':platform.python_version()},logs=['decision.json','invocation.json'])
    prediction['provenance'].update(gt_access='none; runtime deny guard active',policy_sha256=sha256(policy_path),diagnostics_sha256=sha256(bundle/'reliability.json'))
    prediction['timing_seconds']={'selection_and_artifact_export':time.perf_counter()-begin,'total':time.perf_counter()-begin}
    prediction['memory_bytes']={'peak_rss_os':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':None}
    write_json(output/'decision.json',decision);write_json(output/'prediction.json',validate_result(prediction))
    return prediction


if __name__=='__main__':run(sys.argv[1],sys.argv[2],sys.argv[3])
