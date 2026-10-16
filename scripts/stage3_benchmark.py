"""Frozen two-scene DTU pose pilot: preparation, sequential runs, honest failure aggregation."""
import argparse
import copy
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image,ImageFilter
from georecon.data.protocol import read_json,write_json,sha256,validate_manifest,validate_result
from georecon.data.dtu import prepare_dtu
from georecon.evaluation.poses import evaluate_poses

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/benchmarks/dtu_pose_pilot_v1.json'
DATA=ROOT/'data/stage3_dtu'
OUT=ROOT/'outputs/stage3_dtu_pose_pilot_v1'
DOC=ROOT/'docs/stage3'
ENVS={'colmap':'georecon-colmap','vggt_raw':'georecon-vggt','vggt_ba':'georecon-vggt-ba'}


def auc(errors,threshold):
    """Empirical CDF integral / threshold; missing or undefined pairs remain failures."""
    return float(sum(max(0,1-e/threshold) for e in errors if e is not None)/len(errors)) if errors else None


def audit_stage2():
    reference=read_json(ROOT/'data/dtu_smoke_pinned/reference.json')
    roots=['outputs/stage2_colmap/final_validation','outputs/stage2_vggt/acceptance','outputs/stage2_vggt_ba/acceptance']
    records=[]
    for directory in roots:
        root=ROOT/directory;r=validate_result(read_json(root/'prediction.json'));e=evaluate_poses(r,reference)
        assert r['status']=='success' and e['registered_images']==3
        artifacts=[]
        for a in r['artifacts'].values():
            if a is not None and 'sha256' in a:
                assert sha256(root/a['path'])==a['sha256'];artifacts.append(a['path'])
        records.append(dict(output=directory,prediction_sha256=sha256(root/'prediction.json'),artifacts_verified=artifacts,evaluation=e))
    write_json(DOC/'stage2_acceptance_audit.json',dict(audited_at_utc=datetime.now(timezone.utc).isoformat(),status='passed',records=records,limits='single-scene acceptance; previous OOM and preprocessing differences remain applicable'))


def prepare():
    plan=read_json(PLAN);DATA.mkdir(parents=True,exist_ok=True)
    for split,scans in plan['splits'].items():
        for scan in scans:
            pool=DATA/f'scan{scan}_pool'
            if not (pool/'manifest.json').exists():
                c=read_json(ROOT/'configs/datasets/dtu_smoke.json')
                c.update(scene_id=f'dtu_scan{scan}_stage3_pool',scan=scan,view_ids=sorted({v for cond in plan['conditions'] for v in cond['views']}),split=split,max_transfer_bytes=plan['budget']['download_per_scene_bytes'])
                c.pop('member_sha256',None) # New source members pinned by first-use provenance before running methods.
                cfg=DATA/f'scan{scan}_download_config.json';write_json(cfg,c);prepare_dtu(cfg,pool)
            original=read_json(pool/'manifest.json');reference=read_json(pool/'reference.json')
            for cond in plan['conditions']:
                target=DATA/f'scan{scan}_{cond["id"]}'
                if (target/'manifest.json').exists():continue
                (target/'images').mkdir(parents=True)
                manifest=copy.deepcopy(original);manifest.update(scene_id=f'dtu_scan{scan}_{cond["id"]}',split=split)
                manifest['images']=[]
                for view in cond['views']:
                    record=copy.deepcopy(next(i for i in original['images'] if i['image_id']==f'{view:03d}'))
                    src=pool/record['path'];dst=target/record['path']
                    if cond['blur_radius_original_px']:
                        with Image.open(src) as im:im.convert('RGB').filter(ImageFilter.GaussianBlur(cond['blur_radius_original_px'])).save(dst)
                    else:dst.write_bytes(src.read_bytes())
                    record['sha256']=sha256(dst);manifest['images'].append(record)
                manifest['source']['condition']=cond
                write_json(target/'manifest.json',validate_manifest(manifest))
                ref=copy.deepcopy(reference);ref.update(scene_id=manifest['scene_id'],input_manifest_sha256=sha256(target/'manifest.json'))
                ref['images']=[next(i for i in reference['images'] if i['image_id']==r['image_id']) for r in manifest['images']]
                write_json(target/'reference.json',validate_result(ref))
    # Commit the file hashes before any method sees test results.
    lock=dict(plan_sha256=sha256(PLAN),created_at_utc=datetime.now(timezone.utc).isoformat(),methods={k:dict(path=v,sha256=sha256(ROOT/v)) for k,v in plan['methods'].items()},datasets=[])
    for p in sorted(DATA.glob('scan*_*')):
        if p.is_dir() and not p.name.endswith('_pool'):
            lock['datasets'].append(dict(path=str((p/'manifest.json').relative_to(ROOT)),manifest_sha256=sha256(p/'manifest.json'),reference_sha256=sha256(p/'reference.json'),manifest=read_json(p/'manifest.json')))
    if (DOC/'freeze.json').exists():raise ValueError('freeze already exists; never silently regenerate')
    write_json(DOC/'freeze.json',lock)


def run():
    plan,lock=read_json(PLAN),read_json(DOC/'freeze.json')
    assert sha256(PLAN)==lock['plan_sha256']
    for method,item in lock['methods'].items():assert sha256(ROOT/item['path'])==item['sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    for d in lock['datasets']:
        manifest_path=ROOT/d['path'];assert sha256(manifest_path)==d['manifest_sha256'];assert sha256(manifest_path.parent/'reference.json')==d['reference_sha256']
        scene=d['manifest']['scene_id']
        for method,config in plan['methods'].items():
            dest=OUT/scene/method
            if dest.exists():raise ValueError(f'run exists; refusing implicit rerun {dest}')
            dest.parent.mkdir(parents=True,exist_ok=True)
            python=str(Path.home()/'.venvs'/ENVS[method]/'bin/python')
            command=[python,'-m','georecon.cli',{'colmap':'run-colmap','vggt_raw':'run-vggt','vggt_ba':'run-vggt-ba'}[method],'--manifest',str(manifest_path),'--config',str(ROOT/config),'--output',str(dest)]
            if method!='colmap':command+=['--official-repo',str(ROOT/'experiment03_depth_reprojection/vggt_official'),'--checkpoint',str(ROOT/'experiment03_depth_reprojection/model.pt')]
            if method=='vggt_ba':command+=['--assets',str(ROOT/'data/vggt_ba_assets'),'--dinov2-repo',str(ROOT/'data/third_party/dinov2'),'--lightglue-repo',str(ROOT/'data/third_party/LightGlue')]
            begin=time.perf_counter()
            with (dest.parent/(method+'_cli.txt')).open('w') as f:
                try:
                    process=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=210)
                    returncode=process.returncode
                except subprocess.TimeoutExpired:
                    returncode=-999
            if not (dest/'prediction.json').exists():
                raise RuntimeError('runner produced no result protocol; inspect CLI log before any retry')
            prediction=read_json(dest/'prediction.json')
            if prediction['status']=='not_run':
                prediction.update(status='timeout' if returncode==-999 else 'process_error',failure_reason='CLI terminated before final result; see preserved log')
                write_json(dest/'prediction.json',validate_result(prediction))
            write_json(dest/'benchmark_invocation.json',dict(command=command,returncode=returncode,full_cli_wall_seconds=time.perf_counter()-begin,freeze_sha256=sha256(DOC/'freeze.json')))
            print(scene,method,returncode,flush=True)
            # Failures are evaluated too: unavailable pairs remain in the denominator.
            evaluation=evaluate_poses(read_json(dest/'prediction.json'),read_json(manifest_path.parent/'reference.json'))
            write_json(dest/'evaluation.json',evaluation)
    summarize()


def summarize():
    plan,lock=read_json(PLAN),read_json(DOC/'freeze.json');rows=[]
    import shutil
    for d in lock['datasets']:
        manifest=d['manifest'];scene=manifest['scene_id']
        for method in plan['methods']:
            root=OUT/scene/method;r=read_json(root/'prediction.json');e=read_json(root/'evaluation.json')
            errors={k:[] for k in ['rotation','translation','joint']}
            for p in e['pairs']:
                a,b=p['rotation_deg'],p['translation_direction_deg'];errors['rotation'].append(a);errors['translation'].append(b);errors['joint'].append(max(a,b) if a is not None and b is not None else None)
            row=dict(scene_id=scene,split=manifest['split'],condition=manifest['source']['condition']['id'],method=method,status=r['status'],failure_reason=r['failure_reason'],registered=e['registered_images'],input_images=e['input_images'],coverage=e['registration_ratio'],success=r['status']=='success',rotation_deg=e['rotation_deg_registered_subset']['mean'],translation_deg=e['translation_direction_deg_registered_subset']['mean'],ate_mm=e['camera_center_ate']['rmse'],auc={name:{str(t):auc(values,t) for t in plan['metrics']['auc_thresholds_deg']} for name,values in errors.items()},timing_seconds=r['timing_seconds'],memory_bytes=r['memory_bytes'],invocation=read_json(root/'benchmark_invocation.json'))
            row.update(cli_wall_seconds=row['invocation']['full_cli_wall_seconds'],runner_total_seconds=row['timing_seconds']['total'],peak_gpu_allocated_bytes=row['memory_bytes'].get('peak_gpu_allocated'))
            rows.append(row)
            target=DOC/'runs'/scene/method;target.mkdir(parents=True,exist_ok=True)
            for name in ['prediction.json','evaluation.json','worker_state.json','process.json','input_mapping.json','preflight.json','benchmark_invocation.json']:
                if (root/name).exists():shutil.copyfile(root/name,target/name)
            if (root/'worker.log').exists():shutil.copyfile(root/'worker.log',target/'worker_log.txt')
    groups=[]
    for split in plan['splits']:
        for method in plan['methods']:
            part=[r for r in rows if r['split']==split and r['method']==method]
            def stat(key):
                values=[r[key] for r in part if r[key] is not None];return dict(count=len(values),mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None)
            groups.append(dict(split=split,method=method,cases=len(part),successes=sum(r['success'] for r in part),statuses={s:sum(r['status']==s for r in part) for s in sorted({r['status'] for r in part})},coverage=stat('coverage'),cli_wall_seconds=stat('cli_wall_seconds'),runner_total_seconds=stat('runner_total_seconds'),peak_gpu_allocated_bytes=stat('peak_gpu_allocated_bytes'),rotation_registered_subset=stat('rotation_deg'),translation_registered_subset=stat('translation_deg'),ate_available_subset=stat('ate_mm'),joint_auc={str(t):float(np.mean([r['auc']['joint'][str(t)] for r in part])) for t in plan['metrics']['auc_thresholds_deg']},wall_seconds_total=sum(r['invocation']['full_cli_wall_seconds'] for r in part)))
    write_json(DOC/'results.json',dict(freeze_sha256=sha256(DOC/'freeze.json'),rows=rows,groups=groups))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','prepare','run','summarize']);a=p.parse_args()
    {'audit':audit_stage2,'prepare':prepare,'run':run,'summarize':summarize}[a.action]()
