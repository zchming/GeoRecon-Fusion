"""Frozen known-K initialization study. Separate prepare/predict/evaluate processes."""
import argparse
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def verify_protocol(config_path,snapshot,freeze):
    # The human-authored config and indented JSON snapshot have different whitespace.
    if sha(config_path)!=read(freeze)['config_sha256'] or read(config_path)!=read(snapshot):
        raise ValueError('frozen protocol changed')


def processed_image(rgb,kind,radius):
    from PIL import Image,ImageFilter
    if kind=='blur':rgb=rgb.filter(ImageFilter.GaussianBlur(radius))
    return rgb.resize((800,600),Image.Resampling.BILINEAR)


def prepare(args):
    from PIL import Image,ImageFilter
    from georecon.data.remote_zip import RangeZip
    from georecon.geometry.camera import decompose_projection,resize_crop
    import zipfile
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=read(args.config);write(out/'protocol.json',cfg)
    write(out/'freeze.json',{'config_sha256':sha(args.config),'created_unix':time.time(),'stage7b_predictions_exist':False})
    original=read(args.cache/'bundle.json');oldcases=original['cases']
    images={};truth={};feature_paths={};pair_paths={};downloads={}
    if args.source:sys.path.insert(0,str(args.source))
    from georecon.matching.worker import Worker
    original['checkpoint']=str(args.weights.resolve())
    worker=Worker(original,gpu=False)
    # Copy original images/features/matches, byte-identical; no fresh extraction on old scenes.
    for key,item in original['images'].items():
        target=out/'images'/f'{key}.png';target.parent.mkdir(exist_ok=True);shutil.copyfile(item['path'],target)
        f=out/'features'/f'{key}.npz';f.parent.mkdir(exist_ok=True);shutil.copyfile(args.cache/'features'/key/'features.npz',f)
        images[key]={**item,'path':str(target)};feature_paths[key]=str(f)
    for c in oldcases:
        for arm in cfg['frontends']:
            target=out/'matches'/arm/(c['case_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.cache/'main'/c['case_id']/arm/'matches.npz',target)
            pair_paths[arm+':'+':'.join(c['image_keys'])]=str(target)
    for scan in cfg['development_objects']:
        ref=read(Path('data/stage3_dtu')/f'scan{scan}_normal5/reference.json')
        extra=read(Path('data/stage3_dtu')/f'scan{scan}_spaced3/reference.json')
        truth[str(scan)]={i['image_id']:i['camera'] for i in ref['images']+extra['images']}
    remote=RangeZip('https://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip',cfg['budget']['new_data_transfer_bytes'])
    with zipfile.ZipFile(remote) as archive:
        members=[archive.getinfo(f'Rectified/scan{s}/rect_{v:03d}_3_r5000.png') for s in cfg['independent_objects'] for v in [1,2,3,4,5,10,20]]
        if sum(i.compress_size for i in members)+remote.transferred+len(members)*256>cfg['budget']['new_data_transfer_bytes']:raise ValueError('bounded transfer unavailable')
        for info in members:
            scan=int(info.filename.split('/')[1][4:]);view=info.filename.split('/')[-1].split('_')[1]
            raw=(args.reuse_data/'download'/f'scan{scan}'/f'{view}.png').read_bytes() if args.reuse_data else archive.read(info)
            import zlib
            if len(raw)!=info.file_size or zlib.crc32(raw)!=info.CRC:raise ValueError('source member CRC/size mismatch')
            p=out/'download'/f'scan{scan}'/f'{view}.png';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            downloads[info.filename]={'sha256':sha(p),'bytes':len(raw),'crc32':info.CRC}
            import PIL.Image
            with PIL.Image.open(p) as im:
                rgb=im.convert('RGB');w,h=rgb.size
                cal=Path('data/stage3_dtu/scan1_pool/calibration')/f'pos_{view}.txt'
                camera=decompose_projection(np.loadtxt(cal),w,h);processed,_=resize_crop(camera,[800,600])
                truth.setdefault(str(scan),{})[view]=camera.as_dict()
                kinds=['sharp','blur'] if int(view)<=5 else ['sharp']
                for kind in kinds:
                    key=f'scan{scan}_{kind}_{view}';target=out/'images'/f'{key}.png'
                    image=processed_image(rgb,kind,cfg['blur_radius_original_px'])
                    image.save(target);images[key]={'path':str(target),'sha256':sha(target),'K':processed.K.tolist(),'width':800,'height':600}
                    f,timing=worker.features(images[key]);fp=out/'features'/f'{key}.npz';np.savez_compressed(fp,**f);feature_paths[key]=str(fp)
                    write(out/'feature_records'/f'{key}.json',timing)
            print('prepared',scan,view,flush=True)
    write(out/'download_provenance.json',{'url':remote.url,'archive_size':remote.size,'etag':remote.etag,'transferred_bytes':remote.transferred,'ranges':remote.ranges,'members':downloads,'calibration_source':'existing SampleSet cal18 projection matrices; shared acquisition positions'})
    cases=[]
    for scan in cfg['development_objects']+cfg['independent_objects']:
        for condition,views in cfg['conditions'].items():
            kind='blur' if condition=='blur5' else 'sharp';keys=[f'scan{scan}_{kind}_{v:03d}' for v in views]
            cases.append({'id':f'scan{scan}_{condition}','scan':scan,'split':'development' if scan in cfg['development_objects'] else 'independent','condition':condition,'keys':keys,'view_ids':[f'{v:03d}' for v in views]})
    worker.load_model()
    gpu_peak=0;new_match_time=0
    for c in cases:
        for a,b in combinations(c['keys'],2):
            for arm in cfg['frontends']:
                pk=arm+':'+a+':'+b
                if pk in pair_paths:continue
                f,g=load(feature_paths[a]),load(feature_paths[b]);pairs,record,timing,extra=worker.match(f,g,arm)
                path=out/'matches'/arm/f'{a}__{b}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,indices=pairs,pixels_first=f['keypoints'][pairs[:,0]],pixels_second=g['keypoints'][pairs[:,1]],**extra)
                pair_paths[pk]=str(path);new_match_time+=timing['match_total_seconds'];gpu_peak=max(gpu_peak,record.get('peak_allocated_bytes',0))
                write(out/'match_records'/arm/f'{a}__{b}.json',{'timing':timing,'resource':record})
    inference={'config':cfg,'images':images,'cases':cases,'feature_paths':feature_paths,'pair_paths':pair_paths,
        'geometry':read(cfg['geometry_config']),'msac':read(cfg['msac_config'])['profiles']['msac'],'ba':{**read(cfg['ba_config']),**cfg['ba_resource_override']}}
    write(out/'inference.json',inference);write(out/'truth.json',truth)
    paths=[Path(v['path']) for v in images.values()]+[Path(v) for v in feature_paths.values()]+[Path(v) for v in pair_paths.values()]
    write(out/'input_hashes.json',{str(p.relative_to(out)):sha(p) for p in paths})
    write(out/'preparation.json',{'new_matching_seconds':new_match_time,'gpu_peak_allocated_bytes':gpu_peak,'gpu_peak_reserved_bytes':__import__('torch').cuda.max_memory_reserved(),'new_objects':cfg['independent_objects'],'known_K':True,'matching_inputs_identical_across_strategies':True})
    print('prepared cases',len(cases),flush=True)


def predict(args):
    from georecon.teaching.initialization import verify_candidates,initialize,refine_map
    out=args.output.resolve();bundle=read(out/'inference.json');cfg=bundle['config']
    verify_protocol(args.config,out/'protocol.json',out/'freeze.json')
    target=out/'predictions';target.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__).resolve(),root/'src/georecon/teaching/initialization.py',root/'src/georecon/teaching/incremental.py',root/'src/georecon/teaching/two_view.py',root/'src/georecon/teaching/bundle_adjustment.py',root/'src/georecon/matching/backend_control.py',root/'tests/test_adaptive_initialization.py']
    hashes={}
    for source in sources:
        relative=source.relative_to(root);copy=out/'sources'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,copy);hashes[str(relative)]=sha(source)
    write(out/'prediction_provenance.json',{'source_sha256':hashes,'inference_sha256':sha(out/'inference.json'),'protocol_sha256':sha(out/'protocol.json'),'started_unix':time.time()})
    for p,digest in read(out/'input_hashes.json').items():
        if sha(out/p)!=digest:raise ValueError('input changed')
    accesses=[]
    def guard(event,data):
        if event=='open' and isinstance(data[0],(str,bytes)):
            name=os.fsdecode(data[0]).lower()
            if any(s in name for s in ['truth.json','reference.json','evaluation.json','/calibration/']):raise PermissionError('truth forbidden in prediction')
            accesses.append(os.fsdecode(data[0]))
    sys.addaudithook(guard);start=time.perf_counter()
    for case in bundle['cases']:
        keys=case['keys'];features=[(load(bundle['feature_paths'][k])['keypoints'],None) for k in keys]
        ks=[np.asarray(bundle['images'][k]['K']) for k in keys];images=[None]*len(keys);shapes=[(800,600)]*len(keys)
        for arm in cfg['frontends']:
            matches={(i,j):load(bundle['pair_paths'][arm+':'+keys[i]+':'+keys[j]])['indices'] for i,j in combinations(range(len(keys)),2)}
            for seed in cfg['geometry_seeds']:
                geom={**bundle['geometry'],'seed':seed}
                fits,signals,geometry_seconds=verify_candidates(features,matches,ks,shapes,geom,bundle['msac'])
                for strategy in cfg['strategies']+['quality_without_graph']:
                    folder=target/case['id']/arm/str(seed)/strategy;folder.mkdir(parents=True,exist_ok=False)
                    ts=time.perf_counter()
                    try:
                        result,selection=initialize(images,ks,features,matches,geom,fits,signals,strategy,cfg['fixed_seed_indices'][case['condition']],cfg['max_seed_attempts'])
                        pre={str(v):{'R':r.tolist(),'t':t.tolist()} for v,(r,t) in result[0].items()}
                        before=result[1].copy();result,ba,candidate=refine_map(result,ks,features,selection['selected_seed'],bundle['ba'])
                        poses,xyz,obs,steps,debug,origins=result
                        np.savez_compressed(folder/'map.npz',xyz=xyz,before_xyz=before,observations=np.array([(v,k,p) for (v,k),p in sorted(obs.items())],int).reshape(-1,3),**{f'keypoints_{v}':f[0] for v,f in enumerate(features)})
                        np.savez_compressed(folder/'ba_candidate.npz',**candidate)
                        # Raw failed hypotheses remain available, once per shared case/frontend/seed.
                        if strategy=='fixed':
                            for pair,fit in fits.items():np.savez_compressed(folder/f'candidate_{pair[0]}_{pair[1]}.npz',**fit['arrays'],**({'accepted_R':fit['R'],'accepted_t':fit['t']} if fit['R'] is not None else {}))
                        record={'status':'complete','case':case,'arm':arm,'seed':seed,'strategy':strategy,'selection':selection,'signals':signals,'steps':steps,'BA':ba,'before_poses':pre,
                            'poses':{str(v):{'R':r.tolist(),'t':t.tolist()} for v,(r,t) in poses.items()},'points':len(xyz),'incremental_and_ba_seconds':time.perf_counter()-ts,
                            'shared_candidate_seconds':geometry_seconds,'candidate_geometry_calls':len(fits),'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu_bytes':0}
                    except Exception as error:
                        record={'status':'process_error','case':case,'arm':arm,'seed':seed,'strategy':strategy,'error':repr(error),'traceback':traceback.format_exc(),'poses':{},'before_poses':{},'points':0}
                    write(folder/'record.json',record)
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['budget']['rss_gib']*1024**3:raise MemoryError('RSS budget reached')
                    if time.perf_counter()-start>cfg['budget']['session_seconds']:raise TimeoutError('session budget reached; preserve incomplete denominator')
                print(case['id'],arm,seed,'done',flush=True)
    write(out/'prediction_session.json',{'seconds':time.perf_counter()-start,'truth_guard':True,'accesses':sorted(set(accesses)),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


def evaluate(args):
    from georecon.geometry.camera import Camera,relative_pose
    from georecon.matching.metrics import pose_metrics
    from georecon.teaching.incremental import project
    out=args.output.resolve();bundle=read(out/'inference.json');truth=read(out/'truth.json');cfg=bundle['config'];rows=[]
    if (out/'evaluation.json').exists():raise ValueError('evaluation exists')
    for c in bundle['cases']:
        ks=[np.asarray(bundle['images'][key]['K']) for key in c['keys']]
        gt=[Camera.from_dict(truth[str(c['scan'])][v]) for v in c['view_ids']]
        for arm in cfg['frontends']:
            for seed in cfg['geometry_seeds']:
                for strategy in cfg['strategies']+['quality_without_graph']:
                    folder=out/'predictions'/c['id']/arm/str(seed)/strategy
                    rec=read(folder/'record.json') if (folder/'record.json').exists() else {'poses':{},'status':'not_run'}
                    poses={int(v):Camera(ks[int(v)],np.asarray(p['R']),np.asarray(p['t']),800,600) for v,p in rec['poses'].items()}
                    pairs=[]
                    for a,b in combinations(range(len(gt)),2):
                        r,t=relative_pose(poses[a],poses[b]) if a in poses and b in poses else (None,None)
                        pairs.append({'pair':[a,b],**pose_metrics(gt[a],gt[b],r,t,r is not None)})
                    selected=rec.get('selection',{}).get('selected_seed');init=None
                    if selected:
                        a,b=selected;p=rec['before_poses'];r=np.array(p[str(b)]['R']);t=np.array(p[str(b)]['t'])
                        init=pose_metrics(gt[a],gt[b],r,t,True)
                    angles=[p['joint_deg'] for p in pairs if p['joint_deg'] is not None]
                    false=sum(p['false_accept'] for p in pairs);severe=sum(p['accepted'] and p['joint_deg'] is not None and p['joint_deg']>20 for p in pairs)
                    anchor=selected[0] if selected else None
                    correct_cameras=(1+sum(p['correct'] for p in pairs if anchor in p['pair'])) if anchor is not None else 0
                    conservative=sum(v in poses and all(p['correct'] for p in pairs if v in p['pair']) for v in range(len(gt)))
                    errors=[];depth=[]
                    if poses:
                        z=load(folder/'map.npz')
                        for v in poses:
                            obs=z['observations'];m=obs[:,0]==v;uv,d=project(z['xyz'][obs[m,2]],ks[v],poses[v].R,poses[v].t)
                            errors.extend(np.linalg.norm(uv-z[f'keypoints_{v}'][obs[m,1]],axis=1).tolist());depth.extend((d>0).tolist())
                    n=len(pairs);accepted=sum(p['accepted'] for p in pairs)
                    rows.append({'case':c['id'],'scan':c['scan'],'split':c['split'],'condition':c['condition'],'arm':arm,'seed':seed,'strategy':strategy,'status':rec['status'],
                        'planned_pairs':n,'registered':len(poses),'correct_cameras_anchor':correct_cameras,'correct_cameras_all_pairs':conservative,'accurate_scene':all(p['correct'] for p in pairs),
                        'correct_init':bool(init and init['correct']),'false_pairs':false,'severe_pairs':severe,'accepted_pairs':accepted,'false_rate':false/n,'severe_rate':severe/n,
                        **{f'auc{t}':float(np.mean([p['auc'][str(t)] for p in pairs])) for t in [5,10,20]},'max_deg':max(angles) if angles else None,
                        'points':rec.get('points',0),'reprojection_median_px':float(np.median(errors)) if errors else None,'positive_observation_fraction':float(np.mean(depth)) if depth else None,
                        'attempts':len(rec.get('selection',{}).get('attempts',[])),'compute_seconds':rec.get('incremental_and_ba_seconds',0)+rec.get('shared_candidate_seconds',0),
                        'BA_reason':rec.get('BA',{}).get('reason'),'pairs':pairs,'selected_seed':selected})
    summaries=[]
    for split in ['development','independent']:
        for arm in cfg['frontends']:
            for strategy in cfg['strategies']+['quality_without_graph']:
                r=[v for v in rows if v['split']==split and v['arm']==arm and v['strategy']==strategy]
                # Balanced design: equal condition counts and seeds per object; verify denominator.
                expected=len(cfg['development_objects'] if split=='development' else cfg['independent_objects'])*len(cfg['conditions'])*len(cfg['geometry_seeds'])
                if len(r)!=expected:raise ValueError('planned denominator mismatch')
                summaries.append({'split':split,'arm':arm,'strategy':strategy,'runs':len(r),
                    **{k:float(np.mean([v[k] for v in r])) for k in ['accurate_scene','registered','correct_cameras_anchor','correct_cameras_all_pairs','correct_init','auc5','auc10','auc20','false_rate','severe_rate','points','attempts','compute_seconds']},
                    'false_pairs':sum(v['false_pairs'] for v in r),'severe_pairs':sum(v['severe_pairs'] for v in r),
                    'worst_deg':max([v['max_deg'] for v in r if v['max_deg'] is not None],default=None)})
    write(out/'evaluation.json',{'rows':rows,'summaries':summaries,'protocol_sha256':sha(out/'protocol.json')})
    import csv
    path=out/'summaries.csv';path.parent.mkdir(exist_ok=True)
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(json.dumps(summaries,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','predict','evaluate']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/experiments/stage7b_initialization_v2.json'));p.add_argument('--cache',type=Path);p.add_argument('--source',type=Path);p.add_argument('--weights',type=Path);p.add_argument('--reuse-data',type=Path)
    a=p.parse_args();globals()[a.mode](a)

if __name__=='__main__':main()
