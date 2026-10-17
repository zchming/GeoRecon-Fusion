"""Audit previous substep, prepare declared K-only sequences; truth stays separate."""
from pathlib import Path
import numpy as np
from PIL import Image
from georecon.data.protocol import read_json,write_json,sha256
from georecon.data.scene import load_scene
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_incremental'

def prepare():
    if (DOC/'input_lock.json').exists():raise ValueError('already frozen')
    old=read_json(ROOT/'docs/stage6/source_lock.json')
    for name,digest in old['source_sha256'].items():assert sha256(ROOT/name)==digest
    lock=read_json(ROOT/'docs/stage6/input_lock.json')
    for case in lock['cases']:
        for key in ['input','reference']:assert sha256(ROOT/case[key])==case[key+'_sha256']
    write_json(DOC/'two_view_audit.json',{'source_hashes_verified':old['source_sha256'],'all_11_input_and_truth_hashes_verified':True,'previous_acceptance':read_json(ROOT/'docs/stage6/acceptance.json'),'previous_summary':read_json(ROOT/'docs/stage6/summary.json'),'wide_pair_failure_preserved':'001/020: 4 mutual matches; insufficient_matches','old_outputs_modified':False})
    cfg=read_json(ROOT/'configs/teaching/stage6_two_view.json');cfg.update(scope='stage6 minimal PnP/incremental only; no BA, stage7 or new benchmark',pnp_min_inliers=12,pnp_iterations=1000,pnp_ransac_px=2.,registration_order='manifest order excluding fixed seed; one attempt per view; no adaptive selection/retries',track_conflicts='reject ambiguous keypoint or landmark identities in both directions',existing_poses_and_points='immutable; no BA or pose refinement',cases=[{'case_id':f'dtu_scan1_{c}','source_scene':f'scan1_{c}','seed_ids':['001','020'] if c=='spaced3' else ['001','003']} for c in ['reduced3','normal5','blur5','spaced3']])
    # These are new sequence declarations, not a change to old pair settings.
    cfg.pop('real_pairs');cfg.pop('image_failure_controls');cfg.pop('synthetic_correspondence_cases')
    write_json(ROOT/'configs/teaching/stage6_incremental.json',cfg);rows=[]
    for case in cfg['cases']+[{'case_id':'blank_added_view','source_scene':'scan1_reduced3','seed_ids':['001','003'],'blank':True}]:
        path=ROOT/'data/stage3_dtu'/case['source_scene']/'manifest.json';manifest,images=load_scene(path,preprocess=True);ref=read_json(path.parent/'reference.json');cams={i['image_id']:i['camera'] for i in ref['images']}
        folder=ROOT/'data/stage6_incremental'/case['case_id'];folder.mkdir(parents=True,exist_ok=False);items=[]
        for item,im in zip(manifest['images'],images):
            c=cams[item['image_id']];a=np.array([[800/c['width'],0,(800/c['width']-1)/2],[0,600/c['height'],(600/c['height']-1)/2],[0,0,1.]])
            if case.get('blank') and item['image_id']=='002':im=np.zeros_like(im)
            out=folder/(item['image_id']+'.png');Image.fromarray(im).save(out)
            items.append({'image_id':item['image_id'],'path':out.name,'sha256':sha256(out),'width':800,'height':600,'K':(a@c['K']).tolist()})
        ids=[i['image_id'] for i in items];spec={'schema_version':1,'case_id':case['case_id'],'kind':'synthetic' if case.get('blank') else 'real','images':items,'seed':[ids.index(i) for i in case['seed_ids']]};write_json(folder/'input.json',spec)
        truth=ROOT/'data/stage6_incremental_truth'/(case['case_id']+'.json');write_json(truth,{'cameras':cams,'synthetic_blank_control':bool(case.get('blank'))})
        rows.append({'case_id':case['case_id'],'input':str((folder/'input.json').relative_to(ROOT)),'input_sha256':sha256(folder/'input.json'),'reference':str(truth.relative_to(ROOT)),'reference_sha256':sha256(truth),'source_manifest_sha256':sha256(path),'source_reference_sha256':sha256(path.parent/'reference.json')})
    write_json(DOC/'input_lock.json',{'config_sha256':sha256(ROOT/'configs/teaching/stage6_incremental.json'),'cases':rows,'no_new_downloads':True,'new_profile':'3/5-view incremental, same 800x600 preprocessing; known K; previous two-view records retained'})
if __name__=='__main__':prepare()
