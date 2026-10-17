"""Separate preparation: known K inputs only, pose/reference truth in another directory."""
from pathlib import Path
import numpy as np
from PIL import Image
from georecon.data.protocol import read_json,write_json,sha256
from georecon.data.scene import load_scene
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6';DATA=ROOT/'data/stage6_pairs';TRUTH=ROOT/'data/stage6_truth'


def audit_stage5():
    accept=read_json(ROOT/'docs/stage5_acceptance/acceptance.json');lock=read_json(ROOT/'docs/stage5_acceptance/input_lock.json')
    assert accept['engineering_and_evidence_acceptance']=='passed' and accept['scientific_stable_gain']=='not_established'
    for name,digest in lock['source_sha256'].items():assert sha256(ROOT/name)==digest
    for rule in lock['rule_chain']:assert sha256(ROOT/rule['file'])==rule['sha256']
    for case in lock['cases']:
        for name,digest in case['input_hashes'].items():assert sha256(ROOT/name)==digest
        for name,digest in case['artifacts'].items():assert sha256(ROOT/'outputs/stage5_acceptance_v1'/case['scene_id']/(name+'.npz'))==digest
    write_json(DOC/'stage5_audit.json',{'acceptance_sha256':sha256(ROOT/'docs/stage5_acceptance/acceptance.json'),'input_lock_sha256':sha256(ROOT/'docs/stage5_acceptance/input_lock.json'),
        'all_source_rules_input_and_24_ablation_artifact_hashes_verified':True,'engineering_acceptance':'passed','stable_gain':'not_established','previous_outputs_changed':False,'stage6_is_teaching_not_a_method_performance_extension':True})


def save_images(case,images,ids,ks,kind,truth,source):
    folder=DATA/case
    if folder.exists():raise ValueError('input exists; preserve prior preparation')
    folder.mkdir(parents=True);records=[]
    for i,(im,image_id,k) in enumerate(zip(images,ids,ks)):
        path=folder/f'{i}.png';Image.fromarray(im).save(path)
        records.append({'image_id':image_id,'path':path.name,'sha256':sha256(path),'width':im.shape[1],'height':im.shape[0],'K':np.asarray(k).tolist()})
    write_json(folder/'input.json',{'schema_version':1,'case_id':case,'kind':kind,'mode':'images','intrinsics_assumption':'known, distortion-free pinhole, zero-based pixel centers','images':records})
    write_json(TRUTH/(case+'.json'),truth)
    return {'case_id':case,'mode':'images','input':str((folder/'input.json').relative_to(ROOT)),'input_sha256':sha256(folder/'input.json'),'reference':str((TRUTH/(case+'.json')).relative_to(ROOT)),'reference_sha256':sha256(TRUTH/(case+'.json')),'source':source}


def prepare():
    cfg=read_json(ROOT/'configs/teaching/stage6_two_view.json');audit_stage5()
    if (DOC/'input_lock.json').exists():raise ValueError('stage6 inputs already locked')
    rows=[];cached=None
    for case in cfg['real_pairs']:
        manifest_path=ROOT/'data/stage3_dtu'/case['source_scene']/'manifest.json'
        manifest,images=load_scene(manifest_path,preprocess=True);ref_path=manifest_path.parent/'reference.json';reference=read_json(ref_path)
        image_by_id={r['image_id']:im for r,im in zip(manifest['images'],images)}
        cams={r['image_id']:r['camera'] for r in reference['images']};ids=case['view_ids']
        # Existing frozen common preprocessing is resize 1600x1200 -> 800x600, no crop.
        assert manifest['preprocessing']['resize_wh']==[800,600] and manifest['preprocessing']['crop_xywh']==[0,0,800,600]
        k=[]
        for image_id in ids:
            c=cams[image_id];a=np.array([[800/c['width'],0,(800/c['width']-1)/2],[0,600/c['height'],(600/c['height']-1)/2],[0,0,1.]])
            k.append(a@np.array(c['K']))
        first,second=[cams[i] for i in ids];r=np.array(second['R'])@np.array(first['R']).T;t=np.array(second['t'])-r@np.array(first['t'])
        truth={'case_id':case['case_id'],'kind':'real','relative_R':r.tolist(),'relative_t':t.tolist(),'pose_truth_role':'evaluation only; K was explicitly supplied to reconstruction','expected_status':None}
        rows.append(save_images(case['case_id'],[image_by_id[i] for i in ids],ids,k,'real',truth,{'source_manifest':str(manifest_path.relative_to(ROOT)),'manifest_sha256':sha256(manifest_path),'calibration_source_sha256':sha256(ref_path),'processed_pixel_transform':'A = diag(sx,sy,1) with translations (s-1)/2; Knew=A*K'}))
        if cached is None:cached=(image_by_id[ids[0]],k[0])
    image,k=cached
    rows.append(save_images('blank_pair',[np.zeros_like(image),np.zeros_like(image)],['blank0','blank1'],[k,k],'synthetic',{'case_id':'blank_pair','kind':'synthetic','expected_status':'no_reconstruction','reason':'no image texture'},{'control':'generated blank images; not DTU benchmark sample'}))
    rows.append(save_images('identical_dtu_image',[image,image.copy()],['copy0','copy1'],[k,k],'synthetic',{'case_id':'identical_dtu_image','kind':'synthetic','expected_status':'no_reconstruction','reason':'same actual DTU view copied, zero baseline'},{'control':'duplicate image degeneracy; not independent real scene'}))
    rng=np.random.default_rng(cfg['seed']);count=160
    xyz=np.c_[rng.uniform(-1.4,1.4,count),rng.uniform(-.9,.9,count),rng.uniform(4,8,count)]
    angle=np.radians(5);r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    t=np.array([-.7,.02,.01]);k1=np.array([[700.,.2,320],[0,710,240],[0,0,1]]);k2=np.array([[680.,-.1,315],[0,705,245],[0,0,1]])
    def project(x,rotation,translation,k):
        c=x@rotation.T+translation;q=c@k.T;return q[:,:2]/q[:,2:3]
    for name in cfg['synthetic_correspondence_cases']:
        x=xyz.copy();rotation=r.copy();translation=t.copy();truth_inlier=np.ones(count,bool)
        if name=='planar':x[:,2]=6
        if name=='pure_rotation':translation[:]=0
        if name=='tiny_baseline':translation=np.array([-.0001,0,0])
        p=project(x,np.eye(3),np.zeros(3),k1);q=project(x,rotation,translation,k2)
        if name=='nonplanar_noise_outliers':
            p+=rng.normal(0,.2,p.shape);q+=rng.normal(0,.2,q.shape);bad=rng.choice(count,32,replace=False);q[bad]=rng.uniform([0,0],[640,480],(len(bad),2));truth_inlier[bad]=False
        if name=='too_few_matches':p=p[:7];q=q[:7];x=x[:7];truth_inlier=truth_inlier[:7]
        folder=DATA/name
        if folder.exists():raise ValueError('synthetic input exists')
        folder.mkdir(parents=True);np.savez_compressed(folder/'observations.npz',pixels_first=p,pixels_second=q)
        write_json(folder/'input.json',{'schema_version':1,'case_id':name,'kind':'synthetic','mode':'correspondences','intrinsics_assumption':'known synthetic pinhole K; measured noisy correspondences','images':[{'image_id':'a','width':640,'height':480,'K':k1.tolist()},{'image_id':'b','width':640,'height':480,'K':k2.tolist()}],
            'observations':{'path':'observations.npz','sha256':sha256(folder/'observations.npz')}})
        write_json(TRUTH/(name+'.json'),{'case_id':name,'kind':'synthetic','relative_R':rotation.tolist(),'relative_t':translation.tolist(),'first_camera_xyz':x.tolist(),'true_match_mask':truth_inlier.tolist(),'expected_status':'success' if name.startswith('nonplanar') else 'no_reconstruction'})
        rows.append({'case_id':name,'mode':'correspondences','input':str((folder/'input.json').relative_to(ROOT)),'input_sha256':sha256(folder/'input.json'),'reference':str((TRUTH/(name+'.json')).relative_to(ROOT)),'reference_sha256':sha256(TRUTH/(name+'.json')),'source':{'control':'seeded synthetic geometric observations; not image-matching benchmark'}})
    write_json(DOC/'input_lock.json',{'protocol_sha256':sha256(ROOT/'configs/teaching/stage6_two_view.json'),'cases':rows,'no_new_downloads':True,'DTU_intrinsics_are_declared_input_not_pose_evaluation_leakage':True})

if __name__=='__main__':prepare()
