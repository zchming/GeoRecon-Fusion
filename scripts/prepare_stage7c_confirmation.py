"""Prepare predeclared held-out objects/sequences; reuse Stage7B algorithms unchanged."""
import argparse
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
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


def prepare(a):
    import cv2
    from PIL import Image,ImageFilter
    from georecon.data.remote_zip import RangeZip
    from georecon.geometry.camera import Camera,decompose_projection,resize_crop
    from georecon.data.motion_benchmarks import tum_reference
    from georecon.matching.worker import Worker
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);cfg=read(a.config);start=time.perf_counter()
    write(out/'protocol.json',cfg);write(out/'freeze.json',{'config_sha256':sha(a.config),'unix':time.time(),'candidate':'ap3p','before_new_input_retrieval':True,'preparer_sha256':sha(Path(__file__))})
    for n,h in cfg['algorithm_locks'].items():
        if sha(ROOT/n)!=h:raise ValueError('frozen source changed: '+n)
    for n,h in cfg['calibration_sha256'].items():
        if sha(a.calibration/n)!=h:raise ValueError('official shared calibration changed')
    old=read(a.cache/'bundle.json')
    if sha(a.cache/'bundle.json')!=cfg['baseline_bundle_sha256']:raise ValueError('original matching config changed')
    seen=set()
    for source in a.previous:
        for case in read(source/'inference.json')['cases']:
            if 'scan' in case:seen.add(('dtu',str(case['scan'])))
            elif case['family']=='tum_rgbd':seen.add(('tum_rgbd',case['scene'].removeprefix('tum_rgbd_')))
    requested={('dtu',str(s)) for s in cfg['data']['dtu']['scans']}|{('tum_rgbd',s) for s in cfg['data']['tum']['scenes']}
    if requested & seen:raise ValueError('new objects/sequences overlap historical selection inputs')
    write(out/'split_audit.json',{'requested':sorted(requested),'historical_objects_sequences':sorted(seen),'overlap':[],'previous_inference_sha256':{str(p):sha(p/'inference.json') for p in a.previous},'limitation':'object/recording-level holdout, DTU same trajectory; TUM fr1 recordings may share physical office, not independent geographic sites'})
    sys.path.insert(0,str(a.source.resolve()));lg=read(ROOT/'configs/experiments/stage7a_protocol_v1.json')['lightglue']
    for n,h in lg['source_sha256'].items():
        if sha(a.source/n)!=h:raise ValueError('LightGlue source changed')
    if sha(a.weights)!=lg['weights_sha256']:raise ValueError('LightGlue weight changed')
    old['checkpoint']=str(a.weights.resolve());worker=Worker(old,gpu=False)
    raw_records=[];transferred=0;datasets=[];unavailable=[];planned=[];references={};association={}
    for family,scenes,conditions in [('dtu',cfg['data']['dtu']['scans'],cfg['data']['dtu']['conditions']),('tum_rgbd',cfg['data']['tum']['scenes'],cfg['data']['tum']['conditions'])]:
        for scene in scenes:
            sid='scan'+str(scene) if family=='dtu' else 'tum_rgbd_'+scene
            for condition,indices in conditions.items():planned.append({'id':sid+'_'+condition,'family':family,'scene':sid,'condition':condition,'image_count':len(indices)})
    # Exact byte ranges only; never fall back to a full129GB archive.
    d=cfg['data']['dtu'];remote=RangeZip(d['url'],cfg['budget']['download_bytes'])
    if remote.size!=d['archive_size'] or remote.etag!=d['etag']:raise ValueError('official DTU archive identity changed')
    with zipfile.ZipFile(remote) as archive:
        for scan in d['scans']:
            sid='scan'+str(scan);folder=out/'raw'/sid;folder.mkdir(parents=True,exist_ok=False);rasters={};cameras={}
            try:
                for v in d['views']:
                    name=f'{v:03d}';member=f'Rectified/{sid}/rect_{v:03d}_{d["light"]}_r5000.png';info=archive.getinfo(member)
                    if info.compress_size+remote.transferred>cfg['budget']['download_bytes']:raise ValueError('DTU transfer budget')
                    raw=archive.read(info);p=folder/(name+'.png');p.write_bytes(raw)
                    with Image.open(p) as im:
                        if im.size!=(1600,1200):raise ValueError('unexpected DTU raster profile')
                    camera=decompose_projection(np.loadtxt(a.calibration/f'pos_{v:03d}.txt'),1600,1200)
                    rasters[name]=p;cameras[name]=camera
                    raw_records.append({'url':remote.url,'member':member,'sha256':sha(p),'crc32':info.CRC,'bytes':len(raw)})
                datasets.append(('dtu',scan,sid,rasters,cameras,d['conditions']))
            except Exception as error:
                unavailable.append({'family':'dtu','scene':sid,'status':'data_unavailable','reason':repr(error)})
                write(folder/'failure.json',unavailable[-1])
            print('DTU retrieval',sid,flush=True)
    transferred=remote.transferred;raw_records.append({'url':remote.url,'etag':remote.etag,'transferred':remote.transferred,'ranges':remote.ranges})
    def get(url,size=None):
        nonlocal transferred
        remaining=cfg['budget']['download_bytes']-transferred;limit=size or min(1024**2,remaining)
        if limit>remaining:raise ValueError('total transfer budget')
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes=0-{size-1}'} if size else {}),timeout=30) as response:
            if size and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes 0-{size-1}/')):raise ValueError('exact prefix required, full archive refused')
            raw=response.read(limit+1);etag=response.headers.get('ETag')
        transferred+=len(raw)
        if len(raw)>limit or (size and len(raw)!=size):raise ValueError('download size/limit mismatch')
        raw_records.append({'url':url,'bytes':len(raw),'etag':etag,'sha256':hashlib.sha256(raw).hexdigest(),'prefix_bytes':size})
        return raw
    for scene in cfg['data']['tum']['scenes']:
        t=cfg['data']['tum'];sid='tum_rgbd_'+scene;folder=out/'raw'/sid;folder.mkdir(parents=True,exist_ok=False)
        try:
            base=t['archive_base']+scene;raw=get(base+'.tgz',t['prefix_bytes']);(folder/'source_prefix.tgz').write_bytes(raw)
            truth=get(base+'-groundtruth.txt');(folder/'groundtruth.txt').write_bytes(truth);paths=[]
            with tarfile.open(fileobj=io.BytesIO(raw),mode='r|gz') as tar:
                for member in tar:
                    if member.isfile() and '/rgb/' in member.name and member.name.endswith('.png'):
                        p=folder/Path(member.name).name;p.write_bytes(tar.extractfile(member).read());paths.append(p)
                        if len(paths)==t['images']:break
            if len(paths)!=t['images']:raise ValueError('first8MiB does not contain7 RGB members; no substitutes')
            rasters={p.name:p for p in sorted(paths)};cameras={};gaps={}
            for name,p in rasters.items():
                with Image.open(p) as im:
                    if im.size!=(640,480):raise ValueError('unexpected TUM raster profile')
                cameras[name],gaps[name]=tum_reference(truth.decode(),float(Path(name).stem),t['K'],t['maximum_pose_gap_seconds'])
            write(folder/'selected.json',{'archive_order':[p.name for p in paths],'sorted_filenames':list(rasters),'reference_gaps_seconds':gaps})
            association[sid]=gaps;datasets.append(('tum_rgbd',scene,sid,rasters,cameras,t['conditions']))
        except Exception as error:
            unavailable.append({'family':'tum_rgbd','scene':sid,'status':'data_unavailable','reason':repr(error)});write(folder/'failure.json',unavailable[-1])
        print('TUM retrieval',sid,flush=True)
    write(out/'downloads.json',{'transferred_bytes':transferred,'records':raw_records,'full_archive_downloaded':False});write(out/'unavailable.json',unavailable)
    images={};feature_paths={};cases=[]
    for family,scene,sid,rasters,cameras,conditions in datasets:
        names=list(rasters);references[str(scene) if family=='dtu' else sid]={n:c.as_dict() if c is not None else None for n,c in cameras.items()}
        for index,name in enumerate(names):
            camera=cameras[name] if cameras[name] is not None else Camera(np.array(cfg['data']['tum']['K']),np.eye(3),np.zeros(3),640,480)
            v=int(name) if family=='dtu' else index
            with Image.open(rasters[name]) as im:
                for kind in (['sharp','blur'] if (v<=5 if family=='dtu' else v<5) else ['sharp']):
                    rgb=im.convert('RGB');rgb=rgb.filter(ImageFilter.GaussianBlur(4)) if kind=='blur' else rgb
                    if family=='dtu':processed,_=resize_crop(camera,[800,600]);rgb=rgb.resize((800,600),Image.Resampling.BILINEAR)
                    else:processed=camera;rgb=Image.fromarray(cv2.undistort(np.array(rgb),camera.K,np.array(cfg['data']['tum']['distortion']),None,camera.K))
                    key=sid+'_'+kind+'_'+str(v).zfill(3 if family=='dtu' else 4);path=out/'images'/(key+'.png');path.parent.mkdir(exist_ok=True);rgb.save(path)
                    item={'path':str(path),'sha256':sha(path),'K':processed.K.tolist(),'width':rgb.width,'height':rgb.height};images[key]=item
                    f,timing=worker.features(item);path=out/'features'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**f);feature_paths[key]=str(path)
                    write(out/'feature_records'/(key+'.json'),{'timing':timing,'keypoints':len(f['keypoints'])})
        for condition,views in conditions.items():
            kind='blur' if condition=='blur5' else 'sharp';ids=[f'{v:03d}' for v in views] if family=='dtu' else [names[v] for v in views]
            c={'id':sid+'_'+condition,'family':family,'scene':sid,'condition':condition,'view_ids':ids,'keys':[sid+'_'+kind+'_'+str(v).zfill(3 if family=='dtu' else 4) for v in views]}
            if family=='dtu':c.update(scan=scene,split='independent')
            cases.append(c)
    write(out/'truth.json',references);write(out/'planned_cases.json',planned);write(out/'association.json',association)
    worker.load_model();pair_paths={};gpu_peak=reserved=0;match_seconds=0.
    for c in cases:
        for akey,bkey in combinations(c['keys'],2):
            f,g=load(feature_paths[akey]),load(feature_paths[bkey])
            for arm in cfg['frontends']:
                key=arm+':'+akey+':'+bkey
                if key in pair_paths:continue
                matches,record,timing,extras=worker.match(f,g,arm);p=out/'matches'/arm/(akey+'__'+bkey+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,indices=matches,pixels_first=f['keypoints'][matches[:,0]],pixels_second=g['keypoints'][matches[:,1]],**extras);pair_paths[key]=str(p)
                gpu_peak=max(gpu_peak,record.get('peak_allocated_bytes',0));reserved=max(reserved,record.get('peak_reserved_bytes',0));match_seconds+=timing['match_total_seconds']
                write(out/'match_records'/arm/(akey+'__'+bkey+'.json'),{'timing':timing,'resource':record})
        print('matched',c['id'],flush=True)
    if sha(a.baseline/'inference.json')!=cfg['baseline_inference_sha256']:raise ValueError('baseline geometry/BA profile changed')
    previous=read(a.baseline/'inference.json')
    write(out/'inference.json',{'config':cfg,'cases':cases,'images':images,'feature_paths':feature_paths,'pair_paths':pair_paths,'geometry':previous['geometry'],'msac':previous['msac'],'ba':previous['ba']})
    inputs=[Path(v['path']) for v in images.values()]+[Path(v) for v in feature_paths.values()]+[Path(v) for v in pair_paths.values()]
    write(out/'input_hashes.json',{str(p.relative_to(out)):sha(p) for p in inputs})
    write(out/'preparation.json',{'complete':not unavailable,'ready_cases':len(cases),'planned_cases':len(planned),'seconds':time.perf_counter()-start,'match_seconds':match_seconds,'download_bytes':transferred,'gpu_peak_allocated_bytes':gpu_peak,'gpu_peak_reserved_bytes':reserved,'source_sha256':sha(Path(__file__))})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['output','config','source','weights','cache','baseline','calibration']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--previous',type=Path,action='append',required=True);prepare(p.parse_args())
