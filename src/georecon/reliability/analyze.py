"""Inference diagnostics with a filesystem boundary: input bundle only, no GT/evaluation files."""
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
from ..data.protocol import read_json,write_json,sha256,validate_result
from ..data.scene import load_scene
from ..geometry.camera import Camera
from ..baselines.colmap import pixel_transform
from ..provenance import code_provenance
from .observations import match_images
from .geometry import fundamental,sampson_px,triangulation_checks,distribution
from .depth import cross_view


def analyze_prediction(prediction,manifest,observations,arrays,config,per_match=None):
    validate_result(prediction)
    if prediction['role']!='prediction':raise ValueError('only method predictions, never reference cameras, are accepted')
    ids=[i['image_id'] for i in manifest['images']]
    if [i['image_id'] for i in prediction['images']]!=ids or prediction['scene_id']!=manifest['scene_id']:
        raise ValueError('prediction and observation identity mismatch')
    cameras=[]
    for r,record in zip(prediction['images'],manifest['images']):
        camera=Camera.from_dict(r['camera']) if r['registered'] else None
        if camera is not None:
            A=pixel_transform(manifest,record)
            shape=manifest['preprocessing'].get('crop_xywh',[0,0,record['width'],record['height']])
            camera=Camera(A@camera.K,camera.R,camera.t,shape[2],shape[3])
        cameras.append(camera)
    pairs=[]
    for observed in observations['pairs']:
        i,j=observed['indices'];a,b=cameras[i],cameras[j]
        pair={'image_ids':[ids[i],ids[j]],'audit_matches':observed['audit_matches'],'status':'unknown','reason':None,'sampson_px':None,'fraction_within_tolerance':None,'triangulation':None}
        if prediction['status'] not in {'success','partial'}:
            pair['reason']='method_unavailable_'+prediction['status']
        elif a is None or b is None:pair['reason']='missing_camera'
        elif observed['evidence_status']!='available':pair['reason']='insufficient_matches_or_spatial_coverage'
        else:
            F=fundamental(a,b)
            if F is None:pair['reason']='degenerate_baseline'
            else:
                prefix=observed['array_prefix'];p,q=arrays[prefix+'_first'],arrays[prefix+'_second']
                errors=sampson_px(F,p,q)
                if not np.isfinite(errors).all():pair['reason']='undefined_epipolar_lines'
                else:
                    if per_match is not None:per_match[observed['array_prefix']+'_sampson_px']=errors
                    pair.update(status='available',sampson_px=distribution(errors),fraction_within_tolerance=float(np.mean(errors<=config['epipolar_tolerance_px'])),triangulation=triangulation_checks(a,b,p,q))
        pairs.append(pair)
    available=[p for p in pairs if p['status']=='available']
    views=[]
    for image_id in ids:
        incident=[p for p in available if image_id in p['image_ids']]
        views.append({'image_id':image_id,'evidence_pairs':len(incident),'possible_pairs':len(ids)-1,'median_pair_sampson_px':float(np.median([p['sampson_px']['median'] for p in incident])) if incident else None})
    return {'views':views,'method_id':prediction['method_id'],'baseline_status':prediction['status'],'registered_cameras':sum(c is not None for c in cameras),'input_cameras':len(ids),
            'status':'evidence_available' if len(available)==len(pairs) else 'partial_evidence' if available else 'unknown',
            'pairs':pairs,'summary':{'scope':'available evidence pairs only; always read counts/coverage with values','input_pairs':len(pairs),'evidence_pairs':len(available),'evidence_pair_fraction':len(available)/len(pairs),
                                  'median_pair_sampson_px':float(np.median([p['sampson_px']['median'] for p in available])) if available else None,
                                  'median_pair_positive_depth_ratio':float(np.median([p['triangulation']['positive_depth_ratio'] for p in available if p['triangulation']['positive_depth_ratio'] is not None])) if any(p['triangulation']['positive_depth_ratio'] is not None for p in available) else None},
            'depth_consistency':{'status':'not_supported','reason':'no method-owned optimized dense depth output'},'confidence':None,'composite_score':None,'correctness_probability':None}


def deny_ground_truth(event,args):
    """Runtime audit guard is defense in depth; the sanitized bundle is the main boundary."""
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]).lower()
        if any(word in name for word in ['reference.json','evaluation.json','results.json','freeze.json','ground_truth','calibration/']):
            raise PermissionError('GT/evaluation access forbidden in reliability inference')


def run(bundle,config_path,output):
    start=time.perf_counter();bundle=Path(bundle).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('output must be new; do not overwrite diagnostics')
    output.mkdir(parents=True,exist_ok=True)
    sys.addaudithook(deny_ground_truth)
    config=read_json(config_path)
    if config['schema_version']!=1:raise ValueError('unsupported reliability config')
    manifest,images=load_scene(bundle/'manifest.json',preprocess=True)
    expected=sha256(bundle/'manifest.json')
    import cv2
    extraction_start=time.perf_counter();observations,arrays=match_images(images,config)
    np.savez(output/'observations.npz',**arrays)
    write_json(output/'observations.json',observations)
    match_seconds=time.perf_counter()-extraction_start
    methods={};method_timings={};match_signals={}
    for method in ['colmap','vggt_raw','vggt_ba']:
        begin=time.perf_counter();prediction=read_json(bundle/(method+'.json'))
        if prediction['input_manifest_sha256']!=expected:raise ValueError('prediction fingerprint mismatch')
        method_signals={}
        value=analyze_prediction(prediction,manifest,observations,arrays,config,method_signals)
        match_signals.update({method+'_'+key:value for key,value in method_signals.items()})
        if method=='vggt_raw' and prediction['status']=='success':
            artifact=prediction['artifacts']['depth']
            if artifact is None:raise ValueError('raw depth missing')
            raw=bundle/'raw_predictions.npz'
            if sha256(raw)!=artifact['sha256']:raise ValueError('raw depth artifact hash mismatch')
            with np.load(raw,allow_pickle=False) as a:
                if a['depth'].shape[0]!=len(images):raise ValueError('depth count mismatch')
                value['depth_consistency']=cross_view(a['depth'][...,0],a['extrinsics'],a['intrinsics'],config)
                value['confidence']={'scope':'uncalibrated raw output; not probability; no GT-based mapping','depth_conf':distribution(a['depth_conf'].ravel()),'world_points_conf':distribution(a['world_points_conf'].ravel())}
        value['prediction_sha256']=sha256(bundle/(method+'.json'));methods[method]=value;method_timings[method]=time.perf_counter()-begin
    np.savez(output/'match_signals.npz',**match_signals)
    result={'schema_version':1,'role':'reliability_diagnostics','scene_id':manifest['scene_id'],'split':manifest['split'],'input_manifest_sha256':expected,'config':config,'config_sha256':sha256(config_path),'gt_access':'none; runtime deny guard active; sanitized input bundle contains no calibration, reference or evaluation',
            'provenance':code_provenance('real'),'versions':{'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__},'methods':methods,'observations':'observations.json','observations_sha256':sha256(output/'observations.npz'),'match_signals':{'path':'match_signals.npz','sha256':sha256(output/'match_signals.npz'),'semantics':'per-match square-root Sampson error in common pixels; unknown pairs have no array; no filtering applied'},
            'timing_seconds':{'shared_image_matching':match_seconds,**method_timings,'total':time.perf_counter()-start},'memory_bytes':{'peak_rss_os':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':None},
            'compute':'CPU vectorized geometry and image matching; no model inference or CUDA allocation required'}
    write_json(output/'reliability.json',result)
    return result


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
