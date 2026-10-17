"""Teaching run wrapper: permitted known K, no pose/reference access, immutable outputs."""
import platform,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ..data.protocol import read_json,write_json,sha256,relative_file,result_template,validate_result
from ..geometry.camera import Camera
from ..provenance import code_provenance
from .two_view import known_K,features_and_matches,reconstruct


def deny_pose_truth(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]).replace('\\','/').lower()
        if any(t in name for t in ['stage6_truth','reference.json','evaluation.json','results.json','freeze.json','calibration/','ground_truth']):
            raise PermissionError('teaching runtime permits supplied K only; pose/scale/reference reads forbidden')


def load_input(path,cfg):
    path=Path(path).resolve();v=read_json(path)
    if set(v)-{'schema_version','case_id','kind','mode','intrinsics_assumption','images','observations'}:raise ValueError('unexpected teaching input fields')
    if v['schema_version']!=1 or v['kind'] not in ['real','synthetic'] or v['mode'] not in ['images','correspondences'] or len(v['images'])!=2:raise ValueError('unsupported teaching input')
    if len({i['image_id'] for i in v['images']})!=2:raise ValueError('two distinct image IDs required')
    images=[];hashes={str(path):sha256(path)}
    for item in v['images']:
        if set(item)-{'image_id','path','sha256','width','height','K'}:raise ValueError('pose or unexpected image metadata in teaching input')
        known_K(item['K'])
        if min(item['width'],item['height'])<=0 or item['width']>cfg['budget']['image_width'] or item['height']>cfg['budget']['image_height']:raise ValueError('input image dimensions exceed frozen budget; no silent resize')
        if v['mode']=='images':
            p=relative_file(path.parent,item['path']);assert sha256(p)==item['sha256']
            with Image.open(p) as im:
                if im.size!=(item['width'],item['height']):raise ValueError('decoded dimensions differ')
                images.append(np.array(im.convert('RGB')))
            hashes[str(p)]=sha256(p)
    observations=None
    if v['mode']=='correspondences':
        p=relative_file(path.parent,v['observations']['path']);assert sha256(p)==v['observations']['sha256']
        with np.load(p,allow_pickle=False) as z:
            if set(z.files)!={'pixels_first','pixels_second'}:raise ValueError('unexpected arrays in supplied observations')
            observations={k:z[k] for k in z.files}
        hashes[str(p)]=sha256(p)
    return v,images,observations,hashes


def preview(images,p,q,inliers,kept,path):
    if not images:return
    w=images[0].shape[1];canvas=Image.new('RGB',(w+images[1].shape[1],max(i.shape[0] for i in images)))
    canvas.paste(Image.fromarray(images[0]),(0,0));canvas.paste(Image.fromarray(images[1]),(w,0));draw=ImageDraw.Draw(canvas)
    keep=set(kept.tolist());indices=np.linspace(0,len(p)-1,min(80,len(p)),dtype=int) if len(p) else []
    for idx in indices:
        a=tuple(p[idx]);b=tuple(q[idx]+[w,0]);color=(0,240,220) if idx in keep else ((0,200,0) if inliers[idx] else (240,60,40))
        draw.line([a,b],fill=color,width=1)
        for x,y in [a,b]:draw.ellipse((x-2,y-2,x+2,y+2),outline=color)
    canvas.save(path)


def run(input_path,config_path,output):
    import cv2
    start=time.perf_counter();output=Path(output).resolve()
    if output.exists():raise ValueError('output exists; preserve all prior attempts')
    output.mkdir(parents=True);sys.addaudithook(deny_pose_truth);accesses=[]
    def trace(event,args):
        if event=='open' and isinstance(args[0],str):accesses.append(args[0])
    sys.addaudithook(trace)
    cfg=read_json(config_path);spec,images,observations,hashes=load_input(input_path,cfg)
    feature_stats=None;feature_arrays={}
    if images:
        feature_arrays,feature_stats=features_and_matches(images,cfg);p=feature_arrays['pixels_first'];q=feature_arrays['pixels_second']
        np.savez_compressed(output/'features_matches.npz',**feature_arrays)
    else:p=observations['pixels_first'];q=observations['pixels_second']
    result=reconstruct(p,q,spec['images'][0]['K'],spec['images'][1]['K'],cfg)
    np.savez_compressed(output/'geometry_steps.npz',**result['arrays'])
    xyz=result['arrays'].get('accepted_xyz',np.empty((0,3)));kept=result['arrays']['kept_match_indices']
    rgb=np.empty((0,3),dtype=np.uint8)
    if images and len(xyz):
        xy=np.floor(p[kept]+.5).astype(int);xy[:,0]=np.clip(xy[:,0],0,images[0].shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,images[0].shape[0]-1);rgb=images[0][xy[:,1],xy[:,0]]
    elif len(xyz):rgb=np.full((len(xyz),3),200,np.uint8)
    np.savez_compressed(output/'sparse_points.npz',xyz=xyz,rgb=rgb,match_indices=kept)
    preview(images,p,q,result['arrays']['essential_mask'],kept,output/'matches.png')
    pred=result_template(spec['case_id'],sha256(input_path),[i['image_id'] for i in spec['images']],'teaching_two_view_sfm',kind=spec['kind'])
    pred.update(status=result['status'],failure_reason=result['reason'],config=cfg,versions={'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__},logs=['steps.json','invocation.json'])
    if result['status']=='success':
        for i,item in enumerate(spec['images']):
            camera=Camera(np.array(item['K']),np.eye(3) if i==0 else result['R'],np.zeros(3) if i==0 else result['t'],item['width'],item['height'])
            pred['images'][i]={'image_id':item['image_id'],'registered':True,'camera':camera.as_dict(),'reason':None}
        pred['artifacts']['pointcloud']={'path':'sparse_points.npz','sha256':sha256(output/'sparse_points.npz'),'semantics':'finite two-view DLT sparse XYZ in first camera frame; arbitrary unit baseline; rgb and measured match_indices'}
    else:
        for image in pred['images']:image['reason']=result['reason']
    pred['artifacts']['tracks']={'path':'geometry_steps.npz','sha256':sha256(output/'geometry_steps.npz'),'semantics':'measured pixels, RANSAC/hypothesis/triangulation debug arrays; rejected candidates are not accepted geometry'}
    pred['provenance'].update(**code_provenance(spec['kind']),known_intrinsics='supplied K is declared input; no pose or scale truth input',pose_truth_access='none; runtime guard active',input_hashes=hashes,config_sha256=sha256(config_path),scope='educational calibrated two-view, no PnP/incremental mapping/BA')
    elapsed=time.perf_counter()-start;pred['timing_seconds']={'process_body':elapsed};pred['memory_bytes']={'peak_rss_os':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':0}
    write_json(output/'prediction.json',validate_result(pred))
    write_json(output/'steps.json',{'case_id':spec['case_id'],'kind':spec['kind'],'mode':spec['mode'],'feature_stage':feature_stats,'geometry_stage':result['stats'],'status':result['status'],'failure_reason':result['reason'],'known_K':True,'translation_norm':float(np.linalg.norm(result['t'])) if result['t'] is not None else None,'wall_seconds':elapsed,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'GPU_allocated_bytes':0})
    write_json(output/'invocation.json',{'input_path':str(Path(input_path).resolve()),'config_sha256':sha256(config_path),'project_file_accesses':sorted(set(accesses)),'pose_truth_guard_active':True,'code':code_provenance(spec['kind'])})
    return pred


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    try:
        result=run(args.input,args.config,args.output);print(result['scene_id'],result['status'],result['failure_reason'],flush=True)
        if result['status']!='success':raise SystemExit(1)
    except Exception as e:
        out=Path(args.output)
        # Never change an already-existing completed output on overwrite rejection.
        if out.exists() and not (out/'prediction.json').exists():write_json(out/'process_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        raise
