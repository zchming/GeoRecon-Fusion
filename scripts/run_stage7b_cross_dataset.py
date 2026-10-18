"""Frozen external validation; reuses existing matching, MSAC and incremental/BA code."""
import argparse
from itertools import combinations
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


class Downloads:
    def __init__(self,budget):self.budget=budget;self.transferred=0;self.records=[]
    def get(self,url,blob=None):
        with urllib.request.urlopen(url,timeout=30) as response:
            length=response.headers.get('Content-Length');remaining=min(self.budget['max_single_file_bytes'],self.budget['max_download_bytes']-self.transferred)
            if length and int(length)>remaining:raise ValueError('download size exceeds frozen remaining budget')
            raw=response.read(remaining+1)
        self.transferred+=len(raw)
        if len(raw)>remaining:raise ValueError('download budget exceeded; no fallback')
        digest=hashlib.sha256(raw).hexdigest()
        if blob and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blob:raise ValueError('pinned git blob mismatch')
        self.records.append({'url':url,'bytes':len(raw),'sha256':digest,'git_blob':blob})
        return raw


def view_indices(condition,n):
    return [0,(n-1)//2,n-1] if condition=='spaced3' else [0,1,2] if condition=='reduced3' else [0,1,2,3,4]


def prepare(args):
    from PIL import Image
    from georecon.data.calibrated_benchmarks import strecha_camera,middlebury_cameras,preprocess
    from georecon.matching.worker import Worker
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);cfg=read(args.config)
    write(out/'protocol.json',cfg);write(out/'freeze.json',{'config_sha256':sha(args.config),'before_download_unix':time.time(),'no_external_pose_scores_read':True})
    for p,d in cfg['algorithm_locks'].items():
        if sha(ROOT/p)!=d:raise ValueError('frozen algorithm changed: '+p)
    prior=read(ROOT/cfg['prior_config']);old=read(args.cache/'bundle.json');sys.path.insert(0,str(args.source.resolve()))
    lg=read(ROOT/'configs/experiments/stage7a_protocol_v1.json')['lightglue']
    for p,d in lg['source_sha256'].items():
        if sha(args.source/p)!=d:raise ValueError('LightGlue source differs')
    if sha(args.weights)!=lg['weights_sha256']:raise ValueError('LightGlue checkpoint differs')
    old['checkpoint']=str(args.weights.resolve());worker=Worker(old,gpu=False);downloads=Downloads(cfg['budget'])
    metadata=args.metadata.resolve();revision=subprocess.check_output(['git','-C',str(metadata),'rev-parse','HEAD'],text=True).strip()
    if revision!=cfg['strecha_revision']:raise ValueError('dataset metadata revision differs')
    tree=subprocess.check_output(['git','-C',str(metadata),'ls-tree','-r','HEAD'],text=True)
    blobs={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
    images={};features={};cases=[];truth={};rounding={};download_start=time.perf_counter()
    for dataset in cfg['data']:
        family,scene,n=dataset['family'],dataset['scene'],dataset['views'];sid=family+'_'+scene
        wanted=sorted(set(v for condition in cfg['conditions'] for v in view_indices(condition,n)))
        folder=out/'raw'/sid;folder.mkdir(parents=True,exist_ok=False);cameras={};rasters={}
        if family=='strecha':
            base=f"https://raw.githubusercontent.com/openMVG/SfM_quality_evaluation/{revision}/"
            prefix='Benchmarking_Camera_Calibration_2008/'+scene+'/'
            for v in wanted:
                name=f'{v:04d}.jpg';camera_path=prefix+'gt_dense_cameras/'+name+'.camera';image_path=prefix+'images/'+name
                camera_raw=downloads.get(base+urllib.parse.quote(camera_path),blobs[camera_path]);(folder/(name+'.camera')).write_bytes(camera_raw)
                camera,amount=strecha_camera(camera_raw.decode());cameras[name]=camera;rounding[sid+':'+name]=amount
                raw=downloads.get(base+urllib.parse.quote(image_path),blobs[image_path]);path=folder/name;path.write_bytes(raw);rasters[name]=path
        else:
            raw=downloads.get(dataset['url']);(folder/'source.zip').write_bytes(raw)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                camera_names=[p for p in archive.namelist() if p.endswith('_par.txt')]
                if len(camera_names)!=1:raise ValueError('one official parameter table required')
                camera_raw=archive.read(camera_names[0]);(folder/'parameters.txt').write_bytes(camera_raw);parsed=middlebury_cameras(camera_raw.decode())
                names=sorted(parsed)
                if len(names)!=n:raise ValueError('official view count differs')
                for v in wanted:
                    name=names[v];camera,amount=parsed[name];cameras[name]=camera;rounding[sid+':'+name]=amount
                    path=folder/name;members=[p for p in archive.namelist() if Path(p).name==name]
                    if len(members)!=1:raise ValueError('unique selected image required')
                    path.write_bytes(archive.read(members[0]));rasters[name]=path
        names=sorted(cameras);by_index=dict(zip(wanted,names));truth[sid]={name:camera.as_dict() for name,camera in cameras.items()}
        for index in wanted:
            name=by_index[index];camera=cameras[name]
            with Image.open(rasters[name]) as im:
                for kind in (['sharp','blur'] if index<5 else ['sharp']):
                    key=sid+'_'+kind+'_'+str(index).zfill(4);rgb,k,transform=preprocess(im.convert('RGB'),camera,kind=='blur')
                    path=out/'images'/(key+'.png');path.parent.mkdir(exist_ok=True);rgb.save(path)
                    item={'path':str(path),'sha256':sha(path),'K':k.tolist(),'width':rgb.width,'height':rgb.height}
                    images[key]=item;f,timing=worker.features(item);target=out/'features'/(key+'.npz');target.parent.mkdir(exist_ok=True);np.savez_compressed(target,**f);features[key]=str(target)
                    write(out/'feature_records'/(key+'.json'),{'timing':timing,'pixel_transform':transform.tolist(),'keypoints':len(f['keypoints'])})
        for condition in cfg['conditions']:
            indices=view_indices(condition,n);kind='blur' if condition=='blur5' else 'sharp'
            cases.append({'id':sid+'_'+condition,'scene':sid,'family':family,'condition':condition,'view_ids':[by_index[v] for v in indices],'keys':[sid+'_'+kind+'_'+str(v).zfill(4) for v in indices]})
        write(out/'download_progress.json',{'finished_scene':sid,'transferred':downloads.transferred,'records':downloads.records});print('prepared',sid,flush=True)
    write(out/'truth.json',truth);write(out/'calibration_rounding.json',rounding)
    worker.load_model();pair_paths={};match_seconds=0.;gpu_peak=0;reserved=0
    for c in cases:
        for a,b in combinations(c['keys'],2):
            f,g=load(features[a]),load(features[b])
            for arm in cfg['frontends']:
                key=arm+':'+a+':'+b
                if key in pair_paths:continue
                pairs,record,timing,extras=worker.match(f,g,arm);path=out/'matches'/arm/(a+'__'+b+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,indices=pairs,pixels_first=f['keypoints'][pairs[:,0]],pixels_second=g['keypoints'][pairs[:,1]],**extras);pair_paths[key]=str(path)
                match_seconds+=timing['match_total_seconds'];gpu_peak=max(gpu_peak,record.get('peak_allocated_bytes',0));reserved=max(reserved,record.get('peak_reserved_bytes',0))
                write(out/'match_records'/arm/(a+'__'+b+'.json'),{'timing':timing,'memory':record})
    inference={'config':cfg,'images':images,'cases':cases,'feature_paths':features,'pair_paths':pair_paths,
        'geometry':read(ROOT/prior['geometry_config']),'msac':read(ROOT/prior['msac_config'])['profiles']['msac'],'ba':{**read(ROOT/prior['ba_config']),**prior['ba_resource_override']}}
    write(out/'inference.json',inference)
    paths=[Path(v['path']) for v in images.values()]+[Path(v) for v in features.values()]+[Path(v) for v in pair_paths.values()]
    write(out/'input_hashes.json',{str(p.relative_to(out)):sha(p) for p in paths})
    write(out/'preparation.json',{'body_seconds':time.perf_counter()-download_start,'match_seconds':match_seconds,'gpu_peak_allocated_bytes':gpu_peak,'gpu_peak_reserved_bytes':reserved,'downloads':downloads.records,'download_bytes':downloads.transferred,'scenes':len(cfg['data']),'reference_format_checked_against_primary_sources':True,'weight_sha256':sha(args.weights)})
    print('ready',len(cases),'conditions',downloads.transferred,'bytes',flush=True)


def predict(args):
    from georecon.teaching.initialization import verify_candidates,initialize,refine_map
    out=args.output.resolve();bundle=read(out/'inference.json');cfg=bundle['config']
    if sha(args.config)!=read(out/'freeze.json')['config_sha256'] or read(args.config)!=read(out/'protocol.json'):raise ValueError('frozen configuration changed')
    for p,d in cfg['algorithm_locks'].items():
        if sha(ROOT/p)!=d:raise ValueError('algorithm changed after freeze')
    target=out/'predictions';target.mkdir(exist_ok=False);source_paths=[Path(__file__),ROOT/'src/georecon/data/calibrated_benchmarks.py',ROOT/'tests/test_calibrated_benchmarks.py']+[ROOT/p for p in cfg['algorithm_locks']]
    hashes={}
    for p in source_paths:
        name=p.resolve().relative_to(ROOT);dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);hashes[str(name)]=sha(p)
    write(out/'provenance.json',{'source_sha256':hashes,'inference_sha256':sha(out/'inference.json'),'started':time.time()})
    for p,d in read(out/'input_hashes.json').items():
        if sha(out/p)!=d:raise ValueError('input changed')
    accesses=set()
    def guard(event,data):
        if event=='open' and isinstance(data[0],(str,bytes)):
            name=os.fsdecode(data[0]);lower=name.lower()
            if any(w in lower for w in ['truth.json','reference.json','evaluation.json','/raw/','_par.txt','.camera']):raise PermissionError('pose reference forbidden in reconstruction')
            accesses.add(name)
    sys.addaudithook(guard);start=time.perf_counter();completed=0
    for case in bundle['cases']:
        keys=case['keys'];features=[(load(bundle['feature_paths'][k])['keypoints'],None) for k in keys];ks=[np.array(bundle['images'][k]['K']) for k in keys];shapes=[(bundle['images'][k]['width'],bundle['images'][k]['height']) for k in keys]
        for arm in cfg['frontends']:
            matches={(i,j):load(bundle['pair_paths'][arm+':'+keys[i]+':'+keys[j]])['indices'] for i,j in combinations(range(len(keys)),2)}
            for seed in cfg['geometry_seeds']:
                geometry={**bundle['geometry'],'seed':seed};fits,signals,seconds=verify_candidates(features,matches,ks,shapes,geometry,bundle['msac'])
                for strategy in cfg['strategies']:
                    folder=target/case['id']/arm/str(seed)/strategy;folder.mkdir(parents=True,exist_ok=False);ts=time.perf_counter()
                    try:
                        result,selection=initialize([None]*len(keys),ks,features,matches,geometry,fits,signals,strategy,cfg['fixed_seed_indices'],cfg['max_seed_attempts'])
                        before={str(v):{'R':r.tolist(),'t':t.tolist()} for v,(r,t) in result[0].items()}
                        result,ba,candidate=refine_map(result,ks,features,selection['selected_seed'],bundle['ba']);poses,xyz,obs,steps,debug,origins=result
                        np.savez_compressed(folder/'map.npz',xyz=xyz,observations=np.array([(v,k,p) for (v,k),p in sorted(obs.items())],int).reshape(-1,3),**{f'keypoints_{v}':f[0] for v,f in enumerate(features)})
                        np.savez_compressed(folder/'ba_candidate.npz',**candidate)
                        if strategy=='fixed':
                            for pair,fit in fits.items():np.savez_compressed(folder/f'fit_{pair[0]}_{pair[1]}.npz',**fit['arrays'])
                        record={'status':'complete','case':case,'arm':arm,'seed':seed,'strategy':strategy,'selection':selection,'signals':signals,'steps':steps,'BA':ba,'before_poses':before,
                            'poses':{str(v):{'R':r.tolist(),'t':t.tolist()} for v,(r,t) in poses.items()},'points':len(xyz),'map_seconds':time.perf_counter()-ts,'shared_candidate_seconds':seconds,'candidate_calls':len(fits),'gpu_bytes':0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
                    except Exception as error:record={'status':'process_error','case':case,'arm':arm,'seed':seed,'strategy':strategy,'error':repr(error),'traceback':traceback.format_exc(),'poses':{},'before_poses':{},'points':0}
                    write(folder/'record.json',record);completed+=1
                    if time.perf_counter()-start>cfg['budget']['session_seconds']:raise TimeoutError('session budget; retain partial runs')
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>cfg['budget']['rss_gib']*1024**2:raise MemoryError('RSS budget')
                print(case['id'],arm,seed,flush=True)
    write(out/'session.json',{'complete':True,'runs':completed,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gt_guard':True,'file_accesses':sorted(accesses)})


def evaluate(args):
    from georecon.geometry.camera import Camera,relative_pose
    from georecon.matching.metrics import pose_metrics
    from georecon.teaching.incremental import project
    out=args.output.resolve();bundle=read(out/'inference.json');cfg=bundle['config'];truth=read(out/'truth.json');rows=[]
    if (out/'evaluation.json').exists():raise ValueError('preserve previous evaluation')
    for c in bundle['cases']:
        ks=[np.array(bundle['images'][k]['K']) for k in c['keys']];gt=[Camera.from_dict(truth[c['scene']][v]) for v in c['view_ids']]
        for arm in cfg['frontends']:
            for seed in cfg['geometry_seeds']:
                for strategy in cfg['strategies']:
                    folder=out/'predictions'/c['id']/arm/str(seed)/strategy;rec=read(folder/'record.json') if (folder/'record.json').exists() else {'status':'not_run','poses':{}}
                    cams={int(v):Camera(ks[int(v)],np.array(p['R']),np.array(p['t']),bundle['images'][c['keys'][int(v)]]['width'],bundle['images'][c['keys'][int(v)]]['height']) for v,p in rec['poses'].items()};pairs=[]
                    for i,j in combinations(range(len(gt)),2):
                        r,t=relative_pose(cams[i],cams[j]) if i in cams and j in cams else (None,None)
                        pairs.append({'pair':[i,j],**pose_metrics(gt[i],gt[j],r,t,r is not None)})
                    selected=rec.get('selection',{}).get('selected_seed');init=None
                    if selected:
                        i,j=selected;p=rec['before_poses'][str(j)];init=pose_metrics(gt[i],gt[j],np.array(p['R']),np.array(p['t']),True)
                    accepted=sum(p['accepted'] for p in pairs);false=sum(p['false_accept'] for p in pairs);severe=sum(p['joint_deg'] is not None and p['joint_deg']>20 for p in pairs);angles=[p['joint_deg'] for p in pairs if p['joint_deg'] is not None];errors=[]
                    if cams:
                        z=load(folder/'map.npz');obs=z['observations']
                        for v,cam in cams.items():
                            rows_v=obs[obs[:,0]==v];uv,_=project(z['xyz'][rows_v[:,2]],cam.K,cam.R,cam.t);errors.extend(np.linalg.norm(uv-z[f'keypoints_{v}'][rows_v[:,1]],axis=1).tolist())
                    rows.append({'case':c['id'],'scene':c['scene'],'family':c['family'],'condition':c['condition'],'arm':arm,'seed':seed,'strategy':strategy,'status':rec['status'],'registered':len(cams),
                        'accurate_scene':all(p['correct'] for p in pairs),'correct_cameras':sum(v in cams and all(p['correct'] for p in pairs if v in p['pair']) for v in range(len(gt))),
                        'correct_init':bool(init and init['correct']),'accepted_init':selected is not None,'planned_pairs':len(pairs),'accepted_pairs':accepted,'false_pairs':false,'severe_pairs':severe,
                        'false_rate':false/len(pairs),'severe_rate':severe/len(pairs),**{f'auc{t}':float(np.mean([p['auc'][str(t)] for p in pairs])) for t in [5,10,20]},
                        'max_deg':max(angles) if angles else None,'points':rec.get('points',0),'reprojection_median':float(np.median(errors)) if errors else None,
                        'seconds':rec.get('map_seconds',0)+rec.get('shared_candidate_seconds',0),'attempts':len(rec.get('selection',{}).get('attempts',[])),'BA':rec.get('BA',{}).get('reason'),'selected_seed':selected,'pairs':pairs})
    summaries=[]
    for group in ['strecha','middlebury','all']:
        for arm in cfg['frontends']:
            for strategy in cfg['strategies']:
                selected=[r for r in rows if (group=='all' or r['family']==group) and r['arm']==arm and r['strategy']==strategy]
                vals=[p['joint_deg'] for r in selected for p in r['pairs'] if p['joint_deg'] is not None]
                summaries.append({'family':group,'arm':arm,'strategy':strategy,'runs':len(selected),'scenes':len({r['scene'] for r in selected}),
                    **{k:float(np.mean([r[k] for r in selected])) for k in ['registered','accurate_scene','correct_cameras','correct_init','auc5','auc10','auc20','false_rate','severe_rate','seconds','attempts','points']},
                    'false_pairs':sum(r['false_pairs'] for r in selected),'severe_pairs':sum(r['severe_pairs'] for r in selected),'p95_deg':float(np.percentile(vals,95)) if vals else None,'max_deg':max(vals) if vals else None})
    complete=len(rows)==len(bundle['cases'])*len(cfg['frontends'])*len(cfg['geometry_seeds'])*len(cfg['strategies']) and all(r['status']=='complete' for r in rows)
    look={(r['family'],r['arm'],r['strategy']):r for r in summaries};gates={}
    for family in ['strecha','middlebury']:
        a,b=look[family,'C','inliers'],look[family,'C','quality'];tol=1e-12
        gates[family]=all(b[k]>=a[k]-tol for k in ['auc10','accurate_scene','correct_cameras']) and all(b[k]<=a[k]+tol for k in ['false_rate','severe_rate']) and (any(b[k]>a[k]+tol for k in ['auc10','accurate_scene','correct_cameras']) or any(b[k]<a[k]-tol for k in ['false_rate','severe_rate']))
    scene_severe_nonincrease=all(sum(r['severe_pairs'] for r in rows if r['scene']==scene and r['arm']=='C' and r['strategy']=='quality')<=sum(r['severe_pairs'] for r in rows if r['scene']==scene and r['arm']=='C' and r['strategy']=='inliers') for scene in {r['scene'] for r in rows})
    promoted=complete and all(gates.values()) and scene_severe_nonincrease
    result={'rows':rows,'summaries':summaries,'complete':complete,'decision':{'quality_dominates_both_families':gates,'no_scene_severe_increase':scene_severe_nonincrease,'external_quality_promotion_gate':promoted,'provisional_reference':'quality' if promoted else 'inliers','claim':'candidate external validation only; no universal winner if tradeoffs; preserve both implementations'},'source_provenance_sha256':sha(out/'provenance.json')}
    write(out/'evaluation.json',result)
    import csv
    with (out/'summaries.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(json.dumps({'summaries':summaries,'decision':result['decision']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','predict','evaluate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    for key in ['cache','source','weights','metadata']:p.add_argument('--'+key,type=Path)
    args=p.parse_args()
    if args.mode=='prepare' and any(getattr(args,k) is None for k in ['cache','source','weights','metadata']):p.error('prepare requires explicit --cache --source --weights --metadata')
    try:globals()[args.mode](args)
    except Exception as error:
        if args.output.exists():
            path=args.output/(args.mode+'_failure_'+str(time.time_ns())+'.json');write(path,{'error':repr(error),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
