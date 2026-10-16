"""Dev-only counterfactual fallback probes and one GT-isolation export probe."""
import copy
from pathlib import Path
import shutil
import subprocess
import sys
import time
from georecon.data.protocol import read_json,write_json,sha256
from georecon.selection.policy import choose
from georecon.selection.run import link_or_copy
from georecon.evaluation.poses import evaluate_poses
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5'
sys.path.insert(0,str(ROOT))
from scripts.fit_stage5 import dev_only_guard,score


def probe():
    sys.addaudithook(dev_only_guard)
    lock=read_json(DOC/'input_lock.json');policy=read_json(ROOT/'configs/selection/stage5_v1.json');freeze=read_json(ROOT/'docs/stage3/freeze.json')
    probes=[]
    for case in lock['cases']:
        if case['split']!='dev':continue
        bundle=ROOT/case['bundle'];manifest=read_json(bundle/'manifest.json');diagnostics=read_json(bundle/'reliability.json')
        candidates={m:read_json(bundle/(m+'.json')) for m in policy['priority']};hashes={m:sha256(bundle/(m+'.json')) for m in policy['priority']}
        scenarios=[('raw_missing',['vggt_raw']),('raw_and_colmap_missing',['vggt_raw','colmap'])]
        if case['scene_id']=='dtu_scan1_reduced3':scenarios += [('all_missing',list(policy['priority'])),('all_unknown_evidence',[])]
        for name,masked in scenarios:
            simulated=copy.deepcopy(candidates);diag=copy.deepcopy(diagnostics)
            for m in masked:simulated[m]=None
            if name=='all_unknown_evidence':
                for value in diag['methods'].values():value['status']='unknown';value['summary']['evidence_pairs']=0
            decision=choose(simulated,diag,policy,case['scene_id'],sha256(bundle/'manifest.json'),[r['image_id'] for r in manifest['images']],hashes)
            chosen=decision['selected_method'];evaluation=None
            if chosen:
                d=next(d for d in freeze['datasets'] if d['manifest']['scene_id']==case['scene_id'])
                evaluation=evaluate_poses(candidates[chosen],read_json((ROOT/d['path']).parent/'reference.json'))
            probes.append({'scene_id':case['scene_id'],'scenario':name,'masked_methods':masked,'decision':decision,
                           'gt_joint_auc10':score(evaluation) if evaluation else 0.,'gt_rotation_deg':evaluation['rotation_deg_registered_subset']['mean'] if evaluation else None,'gt_translation_deg':evaluation['translation_direction_deg_registered_subset']['mean'] if evaluation else None})
    assert len(probes)==10
    assert next(p for p in probes if p['scene_id']=='dtu_scan1_reduced3' and p['scenario']=='raw_and_colmap_missing')['decision']['selected_method']=='vggt_ba'
    assert all(p['decision']['outcome']=='abstain' for p in probes if p['scenario'] in ['all_missing','all_unknown_evidence'])
    result={'scope':'DEV only; counterfactual missing inputs, not natural failures or extra benchmark samples','policy_sha256':sha256(ROOT/'configs/selection/stage5_v1.json'),'probes':probes,'source_snapshots_modified':False}
    if (DOC/'fault_probes.json').exists():assert read_json(DOC/'fault_probes.json')==result
    else:write_json(DOC/'fault_probes.json',result)
    subprocess.run([sys.executable,str(ROOT/'scripts/probe_stage5_gt_isolation.py')],check=True)
    print('Ten DEV fault probes and GT-isolation export probe passed')


if __name__=='__main__':probe()
