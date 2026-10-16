"""Audit past benchmark, create GT-free bundles, and execute diagnostics only."""
import argparse
import copy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import time
from georecon.data.protocol import read_json,write_json,sha256,validate_result

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/stage4';DATA=ROOT/'data/stage4_bundles';OUT=ROOT/'outputs/stage4_v1'
CONFIG=ROOT/'configs/reliability/stage4_v1.json'
STAGE3=ROOT/'outputs/stage3_dtu_pose_pilot_v1'


def audit():
    from georecon.evaluation.poses import evaluate_poses
    freeze=read_json(ROOT/'docs/stage3/freeze.json');config=read_json(CONFIG)
    assert sha256(ROOT/'docs/stage3/freeze.json')==config['stage3_freeze_sha256']
    rows=[];native_pair_audit=[]
    for d in freeze['datasets']:
        manifest=ROOT/d['path'];assert sha256(manifest)==d['manifest_sha256']
        reference=manifest.parent/'reference.json';assert sha256(reference)==d['reference_sha256']
        for method in ['colmap','vggt_raw','vggt_ba']:
            root=STAGE3/d['manifest']['scene_id']/method;r=validate_result(read_json(root/'prediction.json'))
            e=evaluate_poses(r,read_json(reference));assert e==read_json(root/'evaluation.json')
            assert r==read_json(ROOT/'docs/stage3/runs'/d['manifest']['scene_id']/method/'prediction.json')
            for a in r['artifacts'].values():
                if a and 'sha256' in a:assert sha256(root/a['path'])==a['sha256']
            if method=='colmap':
                state=read_json(root/'worker_state.json');mapping=read_json(root/'input_mapping.json')
                names={m['worker_name']:m['image_id'] for m in mapping}
                image_ids={i:names[name] for i,name,camera in state['database']['images']}
                for pair_id,count,geometry in state['database']['verified_pairs']:
                    first,second=divmod(pair_id,2147483647)
                    native_pair_audit.append({'scene':r['scene_id'],'native_pair_ids':[first,second],'image_ids':[image_ids[first],image_ids[second]],'verified_matches':count,'configuration':geometry})
            rows.append(dict(scene=r['scene_id'],method=method,status=r['status'],evaluation_matches_published=True,artifact_hashes_verified=True))
    write_json(DOC/'stage3_audit.json',dict(audited_at_utc=datetime.now(timezone.utc).isoformat(),rows=rows,counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},freeze_sha256=sha256(ROOT/'docs/stage3/freeze.json'),native_pair_audit=native_pair_audit,documentation_correction='Stage3 edge labels corrected via database image-ID -> worker filename -> original image-ID mapping; retained wide-spacing edge is 001-010, not 010-020; numerical results unchanged'))


def prepare():
    if (DOC/'input_lock.json').exists():raise ValueError('stage4 input lock exists; never overwrite')
    freeze=read_json(ROOT/'docs/stage3/freeze.json')
    assert sha256(ROOT/'docs/stage3/freeze.json')==read_json(CONFIG)['stage3_freeze_sha256']
    lock={'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage3_freeze_sha256':sha256(ROOT/'docs/stage3/freeze.json'),'config_sha256':sha256(CONFIG),'cases':[]}
    for d in freeze['datasets']:
        scene=d['manifest']['scene_id'];src=ROOT/d['path'];dest=DATA/scene
        if dest.exists():raise ValueError('bundle exists; never replace')
        (dest/'images').mkdir(parents=True)
        shutil.copyfile(src,dest/'manifest.json')
        for record in d['manifest']['images']:
            shutil.copyfile(src.parent/record['path'],dest/record['path'])
        for method in ['colmap','vggt_raw','vggt_ba']:
            shutil.copyfile(STAGE3/scene/method/'prediction.json',dest/(method+'.json'))
        raw=read_json(dest/'vggt_raw.json')
        if raw['status']=='success':shutil.copyfile(STAGE3/scene/'vggt_raw/raw_predictions.npz',dest/'raw_predictions.npz')
        files={str(p.relative_to(dest)):sha256(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        assert not any('reference' in n or 'evaluation' in n or 'calibration' in n for n in files)
        lock['cases'].append(dict(scene_id=scene,split=d['manifest']['split'],bundle=str(dest.relative_to(ROOT)),files=files))
    write_json(DOC/'input_lock.json',lock)


def infer():
    lock=read_json(DOC/'input_lock.json');assert sha256(CONFIG)==lock['config_sha256']
    assert lock['stage3_freeze_sha256']==read_json(CONFIG)['stage3_freeze_sha256']
    python=Path.home()/'.venvs/georecon-reliability/bin/python'
    for case in lock['cases']:
        bundle=ROOT/case['bundle'];dest=OUT/case['scene_id']
        for name,digest in case['files'].items():assert sha256(bundle/name)==digest
        dest.parent.mkdir(parents=True,exist_ok=True)
        command=[str(python),'-m','georecon.reliability.analyze',str(bundle),str(CONFIG),str(dest)]
        begin=time.perf_counter()
        with (dest.parent/(case['scene_id']+'_worker.txt')).open('w') as f:process=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        if process.returncode:raise RuntimeError(f'diagnostics failed for {case["scene_id"]}; see worker log')
        write_json(dest/'invocation.json',dict(command=command,returncode=process.returncode,full_process_wall_seconds=time.perf_counter()-begin,input_lock_sha256=sha256(DOC/'input_lock.json')))
        print(case['scene_id'],'completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','prepare','infer']);a=p.parse_args();{'audit':audit,'prepare':prepare,'infer':infer}[a.action]()
