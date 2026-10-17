"""Run a frozen image sequence without reading pose truth; outputs are append-only."""
import argparse,resource,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from ..data.protocol import read_json,write_json,sha256,relative_file,result_template,validate_result
from ..geometry.camera import Camera
from ..provenance import code_provenance
from .run import deny_pose_truth
from .two_view import known_K
from .incremental import reconstruct_incremental


def run(input_path,config_path,output):
    start=time.perf_counter();output=Path(output)
    if output.exists():raise ValueError('preserve prior attempts: output already exists')
    output.mkdir(parents=True);sys.addaudithook(deny_pose_truth)
    def guard(event,args):
        if event=='open' and isinstance(args[0],str) and 'stage6_incremental_truth' in args[0]:raise PermissionError('pose truth forbidden')
    sys.addaudithook(guard)
    accesses=[]
    def trace(event,args):
        if event=='open' and isinstance(args[0],str):accesses.append(args[0])
    sys.addaudithook(trace)
    cfg=read_json(config_path);spec=read_json(input_path)
    if set(spec)!={'schema_version','case_id','kind','images','seed'} or spec['schema_version']!=1 or spec['kind'] not in ['real','synthetic']:raise ValueError('unsupported sequence schema')
    if not 3<=len(spec['images'])<=5 or len({i['image_id'] for i in spec['images']})!=len(spec['images']):raise ValueError('3..5 distinct images required')
    seed=spec['seed']
    if len(seed)!=2 or len(set(seed))!=2 or any(type(i)!=int or not 0<=i<len(spec['images']) for i in seed):raise ValueError('invalid seed indices')
    images=[];ks=[]
    for item in spec['images']:
        if set(item)!={'image_id','path','sha256','width','height','K'}:raise ValueError('unexpected image metadata / pose forbidden')
        ks.append(known_K(item['K']));p=relative_file(Path(input_path).parent,item['path'])
        if sha256(p)!=item['sha256']:raise ValueError('image checksum mismatch')
        with Image.open(p) as im:
            if im.size!=(item['width'],item['height']) or not (0<im.width<=800 and 0<im.height<=600):raise ValueError('no silent resizing')
            images.append(np.array(im.convert('RGB')))
    poses,xyz,obs,steps,debug,origins=reconstruct_incremental(images,ks,seed,cfg)
    np.savez_compressed(output/'geometry_steps.npz',**debug)
    table=np.array([(v,key,point) for (v,key),point in sorted(obs.items())],dtype=np.int64).reshape(-1,3)
    np.savez_compressed(output/'sparse_points.npz',xyz=xyz,rgb=np.full((len(xyz),3),180,np.uint8),observations=table,point_creation_view=np.array(origins,dtype=np.int64))
    pred=result_template(spec['case_id'],sha256(input_path),[i['image_id'] for i in spec['images']],'teaching_incremental_sfm',kind=spec['kind'])
    status='success' if len(poses)==len(images) else ('partial' if poses else 'no_reconstruction')
    pred.update(status=status,failure_reason=None if status=='success' else ('seed_initialization_failed' if not poses else 'one_or_more_pnp_registrations_failed'),config=cfg)
    failures={s.get('view_index'):s['reason'] for s in steps if s['operation']=='pnp' and not s['registered']}
    for v,item in enumerate(spec['images']):
        if v in poses:
            r,t=poses[v];pred['images'][v].update(registered=True,camera=Camera(ks[v],r,t,item['width'],item['height']).as_dict(),reason=None)
        else:pred['images'][v]['reason']=failures.get(v,'seed_initialization_failed')
    if poses:pred['artifacts']['pointcloud']={'path':'sparse_points.npz','sha256':sha256(output/'sparse_points.npz'),'semantics':'unit seed baseline XYZ, immutable existing points; observation rows [view,keypoint,landmark], point_creation_view -1=seed'}
    pred['artifacts']['tracks']={'path':'geometry_steps.npz','sha256':sha256(output/'geometry_steps.npz'),'semantics':'measured pair matches, seed candidates, PnP candidates/masks and new triangulation candidates including rejections'}
    pred['timing_seconds']={'process_body':time.perf_counter()-start};pred['memory_bytes']={'peak_rss_os':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':0}
    pred['provenance'].update(**code_provenance(spec['kind']),config_sha256=sha256(config_path),known_intrinsics='declared known K; no pose truth',BA=False)
    write_json(output/'prediction.json',validate_result(pred));write_json(output/'steps.json',steps)
    write_json(output/'invocation.json',{'input_sha256':sha256(input_path),'config_sha256':sha256(config_path),'pose_truth_guard_active':True,'file_accesses':sorted(set(accesses)),'source':code_provenance(spec['kind'])})
    return pred

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=run(a.input,a.config,a.output);print(result['status']);sys.exit(0 if result['status']=='success' else 1)
