"""Bounded official TUM prefixes and COLMAP sample ZIP members; no full downloads.

Requires explicit output/source/weight paths; raw data stay outside Git. COLMAP
sample poses are a reconstruction reference, not independent ground truth.
"""
import argparse
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
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



def metadata(args):
    from georecon.data.remote_zip import RangeZip
    cfg=read(args.config);out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    reader=RangeZip(cfg['data']['south_building']['url'],16*1024**2)
    with zipfile.ZipFile(reader) as archive:
        for name in ['south-building/sparse/cameras.txt','south-building/sparse/images.txt']:
            raw=archive.read(name);(out/Path(name).name).write_bytes(raw)
        write(out/'south_members.json',[{'name':v.filename,'size':v.file_size,'compressed':v.compress_size} for v in archive.infolist()])
    write(out/'metadata_transfer.json',{'url':reader.url,'size':reader.size,'etag':reader.etag,'transferred':reader.transferred,'ranges':reader.ranges,'member_sha256':{n:sha(out/n) for n in ['cameras.txt','images.txt']}})


def prepare(args):
    import cv2
    from PIL import Image,ImageFilter
    from georecon.data.remote_zip import RangeZip
    from georecon.data.calibrated_benchmarks import preprocess
    from georecon.data.motion_benchmarks import colmap_references,tum_reference
    from georecon.matching.worker import Worker
    cfg=read(args.config);out=args.output.resolve()
    if (out/'inference.json').exists() or (out/'preparation_started.json').exists():raise ValueError('preserve previous preparation')
    if not out.is_dir():raise ValueError('explicit output must contain inspected metadata')
    write(out/'preparation_started.json',{'unix':time.time(),'config_sha256':sha(args.config)})
    sys.path.insert(0,str(args.source.resolve()));lg=read(ROOT/'configs/experiments/stage7a_protocol_v1.json')['lightglue']
    for p,d in lg['source_sha256'].items():
        if sha(args.source/p)!=d:raise ValueError('LightGlue source differs')
    if sha(args.weights)!=lg['weights_sha256']:raise ValueError('LightGlue weights differ')
    meta=read(out/'metadata_transfer.json');transferred=meta['transferred'];records=[meta];datasets=[]
    for n,d in meta.get('member_sha256',{}).items():
        if sha(out/n)!=d:raise ValueError('downloaded metadata hash mismatch')
    def get(url,size=None):
        nonlocal transferred
        headers={'Range':f'bytes=0-{size-1}'} if size else {}
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as response:
            if size and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes 0-{size-1}/')):raise ValueError('exact prefix required; full archive refused')
            limit=size or min(1024**2,cfg['budget']['download_bytes']-transferred)
            data=response.read(limit+1)
            if len(data)>limit or (size and len(data)!=size):raise ValueError('transfer budget/length mismatch')
            etag=response.headers.get('ETag')
        transferred+=len(data)
        if transferred>cfg['budget']['download_bytes']:raise ValueError('total data budget exceeded')
        records.append({'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'etag':etag,'range_prefix':size})
        return data
    # The selection pool is determined solely by official archive order.
    for scene in cfg['data']['tum']['scenes']:
        folder=out/'raw'/('tum_rgbd_'+scene);folder.mkdir(parents=True,exist_ok=False)
        base='https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_'+scene
        raw=get(base+'.tgz',8*1024**2);(folder/'source_prefix.tgz').write_bytes(raw)
        truth_raw=get(base+'-groundtruth.txt');(folder/'groundtruth.txt').write_bytes(truth_raw)
        chosen=[]
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r|gz') as tar:
            for member in tar:
                if member.isfile() and '/rgb/' in member.name and member.name.endswith('.png'):
                    path=folder/Path(member.name).name
                    path.write_bytes(tar.extractfile(member).read());chosen.append(path)
                    if len(chosen)==7:break
        if len(chosen)!=7:raise ValueError('fewer than7 planned TUM RGB images; no substitute selection')
        rasters={p.name:p for p in sorted(chosen)};cameras={};gaps={}
        for name in rasters:
            camera,gap=tum_reference(truth_raw.decode(),float(Path(name).stem),cfg['data']['tum']['K']);cameras[name]=camera;gaps[name]=gap
        write(folder/'selected.json',{'archive_order':[p.name for p in chosen],'sorted_filenames':list(rasters),'reference_gaps_seconds':gaps})
        datasets.append(('tum_rgbd',scene,7,rasters,cameras,{name:np.array(cfg['data']['tum']['distortion']) for name in rasters},list(rasters)))
        print('downloaded',scene,transferred,flush=True)
    scene='south_building';folder=out/'raw'/scene;folder.mkdir(parents=True,exist_ok=False)
    parsed=colmap_references((out/'cameras.txt').read_text(),(out/'images.txt').read_text());names=sorted(parsed)
    if len(names)!=128:raise ValueError('128 official COLMAP sample images expected')
    wanted=[0,1,2,3,4,63,127];rasters={};cameras={};distortions={}
    reader=RangeZip(cfg['data']['south_building']['url'],cfg['budget']['download_bytes']-transferred)
    if reader.etag!=meta['etag'] or reader.size!=meta['size']:raise ValueError('official ZIP changed since metadata read')
    with zipfile.ZipFile(reader) as archive:
        for v in wanted:
            name=names[v];data=archive.read('south-building/images/'+name);path=folder/name;path.write_bytes(data)
            rasters[name]=path;cameras[name],distortions[name]=parsed[name]
    transferred+=reader.transferred
    records.append({'url':reader.url,'bytes':reader.transferred,'etag':reader.etag,'ranges':reader.ranges})
    datasets.append(('south_building',scene,128,rasters,cameras,distortions,names))
    write(out/'downloads.json',{'records':records,'transferred':transferred,'exploratory_probe_bytes_not_included':0,'budget_bytes':cfg['budget']['download_bytes'],'full_archives_downloaded':False})
    # GPU model is loaded only from the previously verified local checkpoint.
    worker_bundle=read(args.cache/'bundle.json');worker_bundle['checkpoint']=str(args.weights.resolve());worker=Worker(worker_bundle,gpu=False)
    images={};features={};cases=[];truth={};start=time.perf_counter()
    for family,scene,n,rasters,cameras,distortions,names in datasets:
        sid=family+'_'+scene;truth[sid]={name:camera.as_dict() if camera is not None else None for name,camera in cameras.items()}
        for name,path in rasters.items():
            v=names.index(name);camera=cameras[name]
            # Missing evaluation pose must never drop an inference frame or its K.
            if camera is None:
                from georecon.geometry.camera import Camera
                camera=Camera(np.array(cfg['data']['tum']['K']),np.eye(3),np.zeros(3),640,480)
            with Image.open(path) as im:
                for kind in (['sharp','blur'] if v<5 else ['sharp']):
                    raw=im.convert('RGB');raw=raw.filter(ImageFilter.GaussianBlur(4)) if kind=='blur' else raw
                    raw=Image.fromarray(cv2.undistort(np.array(raw),camera.K,distortions[name],None,camera.K))
                    rgb,k,transform=preprocess(raw,camera,False);key=sid+'_'+kind+'_'+str(v).zfill(4)
                    dest=out/'images'/(key+'.png');dest.parent.mkdir(exist_ok=True);rgb.save(dest)
                    item={'path':str(dest),'sha256':sha(dest),'K':k.tolist(),'width':rgb.width,'height':rgb.height};images[key]=item
                    f,timing=worker.features(item);dest=out/'features'/(key+'.npz');dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**f);features[key]=str(dest)
                    write(out/'feature_records'/(key+'.json'),{'timing':timing,'pixel_transform':transform.tolist(),'keypoints':len(f['keypoints']),'undistorted':True,'original_distortion':distortions[name].tolist()})
        for condition in cfg['data']['conditions']:
            vs=[0,(n-1)//2,n-1] if condition=='spaced3' else [0,1,2] if condition=='reduced3' else [0,1,2,3,4]
            kind='blur' if condition=='blur5' else 'sharp'
            cases.append({'id':sid+'_'+condition,'scene':sid,'family':family,'condition':condition,'view_ids':[names[v] for v in vs],'keys':[sid+'_'+kind+'_'+str(v).zfill(4) for v in vs]})
    write(out/'truth.json',truth);worker.load_model();pair_paths={};match_seconds=0.;gpu_peak=0;reserved=0
    for c in cases:
        for a,b in combinations(c['keys'],2):
            f,g=load(features[a]),load(features[b])
            for arm in cfg['frontends']:
                key=arm+':'+a+':'+b
                if key in pair_paths:continue
                pairs,record,timing,extras=worker.match(f,g,arm);dest=out/'matches'/arm/(a+'__'+b+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,indices=pairs,pixels_first=f['keypoints'][pairs[:,0]],pixels_second=g['keypoints'][pairs[:,1]],**extras);pair_paths[key]=str(dest)
                match_seconds+=timing['match_total_seconds'];gpu_peak=max(gpu_peak,record.get('peak_allocated_bytes',0));reserved=max(reserved,record.get('peak_reserved_bytes',0))
                write(out/'match_records'/arm/(a+'__'+b+'.json'),{'timing':timing,'memory':record})
        print('matched',c['id'],flush=True)
    prior=read(args.prior/'inference.json')
    write(out/'inference.json',{'config':cfg,'images':images,'cases':cases,'feature_paths':features,'pair_paths':pair_paths,'geometry':prior['geometry'],'msac':prior['msac'],'ba':prior['ba']})
    write(out/'input_hashes.json',{str(p.relative_to(out)):sha(p) for p in [Path(v['path']) for v in images.values()]+[Path(p) for p in features.values()]+[Path(p) for p in pair_paths.values()]})
    write(out/'preparation.json',{'body_seconds':time.perf_counter()-start,'match_seconds':match_seconds,'gpu_peak_allocated_bytes':gpu_peak,'gpu_peak_reserved_bytes':reserved,'download_bytes':transferred,'cases':len(cases),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/georecon/data/motion_benchmarks.py']}})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['metadata','prepare','all'],default='all');p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    args=p.parse_args()
    if args.mode in ['metadata','all']:metadata(args)
    if args.mode in ['prepare','all']:prepare(args)
