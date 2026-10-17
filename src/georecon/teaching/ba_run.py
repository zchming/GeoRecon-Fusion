"""Immutable teaching BA wrapper over a hashed incremental map, known K only."""
import argparse,copy,platform,resource,sys,time
from pathlib import Path
import numpy as np
import scipy
from ..data.protocol import read_json,write_json,sha256,relative_file,validate_result
from ..geometry.camera import Camera
from ..provenance import code_provenance
from .run import deny_pose_truth
from .bundle_adjustment import prepare,decode,residuals,solve


def load_problem(input_path,baseline):
    input_path=Path(input_path);baseline=Path(baseline);spec=read_json(input_path);pred=validate_result(read_json(baseline/'prediction.json'))
    if set(spec)!={'schema_version','case_id','kind','images','seed'}:raise ValueError('unexpected input fields')
    if pred['input_manifest_sha256']!=sha256(input_path) or pred['scene_id']!=spec['case_id']:raise ValueError('map/input mismatch')
    if [v['image_id'] for v in pred['images']]!=[v['image_id'] for v in spec['images']]:raise ValueError('image order differs')
    for item in spec['images']:
        if set(item)!={'image_id','path','sha256','width','height','K'}:raise ValueError('pose metadata is forbidden in input')
        if sha256(relative_file(input_path.parent,item['path']))!=item['sha256']:raise ValueError('input image checksum mismatch')
    registered=[v for v,item in enumerate(pred['images']) if item['registered']];lookup={v:i for i,v in enumerate(registered)}
    for artifact in pred['artifacts'].values():
        if artifact and sha256(relative_file(baseline,artifact['path']))!=artifact['sha256']:raise ValueError('baseline artifact changed')
    with np.load(baseline/'sparse_points.npz',allow_pickle=False) as z:points={key:z[key] for key in z.files}
    with np.load(baseline/'geometry_steps.npz',allow_pickle=False) as z:
        table=points['observations'];uv=np.array([z[f'keypoints_{v}'][key] for v,key,point in table],float).reshape(-1,2)
    if not registered:return spec,pred,points,None,registered,uv
    if any(v not in lookup for v in spec['seed']):raise ValueError('registered map lacks its gauge seed')
    cams=[pred['images'][v]['camera'] for v in registered]
    for v,c in zip(registered,cams):
        if not np.array_equal(c['K'],spec['images'][v]['K']):raise ValueError('known K differs from baseline')
    problem=prepare([c['R'] for c in cams],[c['t'] for c in cams],[c['K'] for c in cams],points['xyz'],np.array([lookup[int(v)] for v in table[:,0]],int),table[:,2],uv,[lookup[v] for v in spec['seed']])
    return spec,pred,points,problem,registered,uv


def run(input_path,baseline,config_path,output):
    start=time.perf_counter();output=Path(output)
    if output.exists():raise ValueError('preserve prior attempts; output exists')
    output.mkdir(parents=True);sys.addaudithook(deny_pose_truth);accesses=[]
    def guard(event,args):
        if event=='open' and isinstance(args[0],str):
            if 'stage6_incremental_truth' in args[0]:raise PermissionError('pose truth forbidden in BA')
            accesses.append(args[0])
    sys.addaudithook(guard)
    cfg=read_json(config_path);spec,before,points,problem,registered,uv=load_problem(input_path,baseline)
    np.savez_compressed(output/'before_points.npz',**points)
    if problem is None:
        delta=np.array([]);stats={'accepted':False,'reason':'no_seeded_map','converged':False,'wall_seconds':0.};candidate_r=np.empty((0,3,3));candidate_t=np.empty((0,3));candidate_x=points['xyz'];initial=final=np.empty((0,2));zi=zf=np.array([])
    else:
        delta,stats=solve(problem,cfg);candidate_r,candidate_t,candidate_x=decode(problem,delta);initial,zi=residuals(problem,np.zeros(problem['size']));final,zf=residuals(problem,delta)
    np.savez_compressed(output/'ba_candidates.npz',delta=delta,R=candidate_r,t=candidate_t,xyz=candidate_x,initial_residual_px=initial,final_residual_px=final,initial_depth=zi,final_depth=zf)
    np.savez_compressed(output/'observations.npz',observations=points['observations'],pixels=uv,registered_original_indices=np.array(registered,int))
    after=copy.deepcopy(before);after.update(method_id='teaching_incremental_sfm_ba',run_id='teaching_incremental_sfm_ba',config=cfg,logs=['ba.json','invocation.json'])
    if stats['accepted']:
        points={**points,'xyz':candidate_x}
        for local,v in enumerate(registered):
            item=spec['images'][v];after['images'][v]['camera']=Camera(np.array(item['K']),candidate_r[local],candidate_t[local],item['width'],item['height']).as_dict()
    np.savez_compressed(output/'sparse_points.npz',**points)
    if registered:after['artifacts']['pointcloud']={'path':'sparse_points.npz','sha256':sha256(output/'sparse_points.npz'),'semantics':'all original sparse points, same IDs/rgb/observation table; BA candidate only if accepted; arbitrary scale with fixed gauge coordinate'}
    after['artifacts']['tracks']={'path':'observations.npz','sha256':sha256(output/'observations.npz'),'semantics':'unchanged measured keypoint observations and original image-index mapping; no tracks filtered or created'}
    after['provenance'].update(**code_provenance(spec['kind']),BA=True,BA_accepted=stats['accepted'],baseline_prediction_sha256=sha256(Path(baseline)/'prediction.json'),config_sha256=sha256(config_path),pose_truth_access='none; runtime guard active',scope='stage6 teaching calibrated BA only')
    after['versions']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    after['timing_seconds']={'process_body':time.perf_counter()-start,'ba_solver':stats['wall_seconds']};after['memory_bytes']={'peak_rss_os':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':0}
    write_json(output/'prediction.json',validate_result(after));write_json(output/'ba.json',stats)
    write_json(output/'invocation.json',{'pose_truth_guard_active':True,'file_accesses':sorted(set(accesses)),'input_sha256':sha256(input_path),'baseline_prediction_sha256':sha256(Path(baseline)/'prediction.json'),'config_sha256':sha256(config_path),'source':code_provenance(spec['kind'])})
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','baseline','config','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();s=run(a.input,a.baseline,a.config,a.output);print(s['reason']);sys.exit(0 if s['accepted'] else 1)
