"""Reaudit immutable Stage4 diagnostics and construct GT-free scene selection inputs."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from georecon.selection.run import link_or_copy
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5'


def audit():
    source=read_json(ROOT/'docs/stage4/input_lock.json');rows=[]
    for case in source['cases']:
        scene=case['scene_id'];path=ROOT/'outputs/stage4_v1'/scene
        r=read_json(path/'reliability.json')
        assert r==read_json(ROOT/'docs/stage4/runs'/scene/'reliability.json')
        assert r['config_sha256']==source['config_sha256']
        assert sha256(path/'observations.npz')==r['observations_sha256']
        assert sha256(path/'match_signals.npz')==r['match_signals']['sha256']
        for name,digest in case['files'].items():assert sha256(ROOT/case['bundle']/name)==digest
        for method,value in r['methods'].items():
            assert value['correctness_probability'] is None and value['composite_score'] is None
            assert sha256(ROOT/case['bundle']/(method+'.json'))==value['prediction_sha256']
            if value['baseline_status'] not in ['success','partial']:assert value['status']=='unknown'
            rows.append({'scene_id':scene,'split':case['split'],'method':method,'baseline_status':value['baseline_status'],'diagnostic_status':value['status']})
    write_json(DOC/'stage4_audit.json',{'rows':rows,'counts':dict(Counter(r['diagnostic_status'] for r in rows)),'published_snapshots_and_array_hashes_match':True,
                                      'counterexamples_reviewed':['blur COLMAP low Sampson does not establish correct pose','spaced3 partial geometric evidence cannot certify all cameras','OOM/no model remains unknown','stage3 native image-ID mapping correction retained']})


def prepare():
    if (DOC/'input_lock.json').exists():raise ValueError('Stage5 lock exists; never regenerate implicitly')
    previous=read_json(ROOT/'docs/stage4/input_lock.json');protocol=read_json(ROOT/'configs/selection/stage5_protocol.json')
    assert sha256(ROOT/'docs/stage4/input_lock.json')==protocol['stage4_input_lock_sha256']
    lock={'stage3_freeze_sha256':protocol['stage3_freeze_sha256'],'stage4_input_lock_sha256':protocol['stage4_input_lock_sha256'],'protocol_sha256':sha256(ROOT/'configs/selection/stage5_protocol.json'),'cases':[]}
    for case in previous['cases']:
        scene=case['scene_id'];bundle=ROOT/'data/stage5_bundles'/scene
        if bundle.exists():raise ValueError('selection bundle exists')
        bundle.mkdir(parents=True)
        for filename in ['manifest.json','colmap.json','vggt_raw.json','vggt_ba.json']:
            shutil.copyfile(ROOT/case['bundle']/filename,bundle/filename)
        shutil.copyfile(ROOT/'outputs/stage4_v1'/scene/'reliability.json',bundle/'reliability.json')
        for method in protocol['methods']:
            p=read_json(bundle/(method+'.json'));seen=set()
            for a in p['artifacts'].values():
                if a is None or a['path'] in seen:continue
                seen.add(a['path']);src=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/method/a['path']
                assert sha256(src)==a['sha256'];link_or_copy(src,bundle/'artifacts'/method/a['path'])
        files={str(p.relative_to(bundle)):sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
        assert not any('reference' in n or 'evaluation' in n or 'calibration' in n for n in files)
        lock['cases'].append({'scene_id':scene,'split':case['split'],'bundle':str(bundle.relative_to(ROOT)),'files':files})
    write_json(DOC/'input_lock.json',lock)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','prepare']);args=parser.parse_args();{'audit':audit,'prepare':prepare}[args.action]()
