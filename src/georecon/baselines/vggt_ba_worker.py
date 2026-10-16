"""Run the pinned, unmodified official demo with offline weights and audit hooks."""
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import zipfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
from ..data.protocol import read_json, write_json, sha256
from ..geometry.camera import Camera
from .vggt_ba import independent_square


def main(output):
    state = {'status':'starting','timing_seconds':{},'memory_bytes':{},'weight_loads':[]}
    def save():
        state['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write_json(output/'worker_state.tmp.json',state)
        (output/'worker_state.tmp.json').replace(output/'worker_state.json')
    save()
    try:
        start = time.perf_counter()
        import torch
        import pyceres  # Register ceres::Solver::Options before inspecting PyCOLMAP options.
        import pycolmap
        import importlib.metadata as metadata
        c, request, mapping = [read_json(output/(n+'.json')) for n in ['config','request','input_mapping']]
        state['versions'] = {k:metadata.version(k) for k in ['torch','torchvision','pycolmap','pyceres','kornia','trimesh','hydra-core','omegaconf','numpy','Pillow','lightglue']}
        state['versions'].update(python=platform.python_version(),cuda_runtime=torch.version.cuda)
        if pycolmap.__version__ != c['pycolmap_version'] or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('pinned PyCOLMAP or CUDA BF16 unavailable')
        torch.set_num_threads(c['num_threads'])
        total = torch.cuda.get_device_properties(0).total_memory
        limit = int(c['gpu_allocator_limit_gib']*1024**3)
        torch.cuda.set_per_process_memory_fraction(limit/total)
        torch.cuda.reset_peak_memory_stats()
        state['device'] = dict(name=torch.cuda.get_device_name(),total_bytes=total,allocator_cap_bytes=limit)
        state['timing_seconds']['runtime_imports_and_cuda_setup'] = time.perf_counter()-start
        def sync():
            torch.cuda.synchronize()
        def measured(name, fn):
            def wrapped(*args,**kwargs):
                state['status'] = name
                save()
                sync()
                begin = time.perf_counter()
                try:
                    return fn(*args,**kwargs)
                finally:
                    sync()
                    state['timing_seconds'][name] = state['timing_seconds'].get(name,0)+time.perf_counter()-begin
                    state['memory_bytes'].update(peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
                    save()
            return wrapped
        # Verify every local asset before constructing models. No implicit network fallback.
        start = time.perf_counter()
        if sha256(request['checkpoint']) != c['checkpoint_sha256']:
            raise ValueError('VGGT checkpoint hash mismatch')
        routes = {'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt':Path(request['checkpoint'])}
        asset_config = read_json(output/'asset_config.json')
        for asset in asset_config['assets']:
            path = Path(request['assets'])/asset['filename']
            if sha256(path) != asset['sha256']:
                raise ValueError(f"BA asset hash mismatch: {asset['name']}")
            routes[asset['requested_url']] = path
        state['timing_seconds']['all_checkpoint_sha256_checks'] = time.perf_counter()-start
        state['verified_weight_sha256'] = {'vggt':c['checkpoint_sha256'],**{a['name']:a['sha256'] for a in asset_config['assets']}}
        original_hub_load = torch.hub.load
        def local_weights(url,*args,**kwargs):
            if url not in routes:
                raise ValueError(f'unapproved weight URL: {url}')
            begin = time.perf_counter()
            weights = torch.load(routes[url],map_location='cpu',weights_only=True,mmap=zipfile.is_zipfile(routes[url]))
            state['weight_loads'].append(dict(url=url,file=routes[url].name,seconds=time.perf_counter()-begin))
            save()
            return weights
        def local_hub(repo,model,*args,**kwargs):
            if repo != 'facebookresearch/dinov2' or model != 'dinov2_vitb14_reg':
                raise ValueError('unapproved hub repository/model')
            return original_hub_load(request['dinov2_repo'],model,*args,source='local',**kwargs)
        torch.hub.load_state_dict_from_url = local_weights
        torch.hub.load = local_hub
        # Enforce the pinned feature extractor, even if another LightGlue is installed.
        sys.path.insert(0,request['lightglue_repo'])
        sys.path.insert(0,request['official_repo'])
        start = time.perf_counter()
        import demo_colmap as demo
        import vggt.dependency.track_predict as tracking
        state['timing_seconds']['official_demo_import'] = time.perf_counter()-start
        # Upstream glob is unsorted; pin its order to the manifest before any pose inference.
        original_glob = demo.glob.glob
        def ordered_glob(pattern):
            paths = original_glob(pattern)
            expected = [str(output/'images'/m['worker_name']) for m in mapping]
            if set(paths) != set(expected):
                raise ValueError('official input file set differs from manifest')
            state['official_glob_order_corrected'] = paths != expected
            return expected
        demo.glob = SimpleNamespace(glob=ordered_glob)
        demo.VGGT = measured('vggt_model_construction',demo.VGGT)
        original_loader = demo.load_and_preprocess_images_square
        def audited_loader(paths,size=1024):
            expected_paths = [str(output/'images'/m['worker_name']) for m in mapping]
            if paths != expected_paths:
                raise ValueError('official glob order differs from manifest; refusing reordered inference')
            images, coords = original_loader(paths,size)
            expected = []
            for path in paths:
                with Image.open(path) as image:
                    expected.append(independent_square(image,size))
            actual = images.numpy()
            error = float(np.max(np.abs(actual-np.stack(expected))))
            if error > 1e-7 or list(images.shape) != [len(mapping),3,1024,1024]:
                raise ValueError('official square pixel audit failed')
            write_json(output/'preflight.json',dict(input_shape=list(images.shape),independent_loader_max_abs=error,range=[float(actual.min()),float(actual.max())],model_grid=[518,518],tracking_grid=[1024,1024],mapping=mapping,pixel_convention='integer tracking pixels inserted unchanged by official converter; no -0.5 adjustment',intrinsic_scaling='official K[:2,:] *= 1024/518; literal upstream rule',export_note='primary export inverses actual PIL pixel-center transform; official final sparse resets principal point separately'))
            return images,coords
        demo.load_and_preprocess_images_square = measured('square_preprocessing_and_pixel_audit',audited_loader)
        original_forward = demo.run_VGGT
        def forward(*args,**kwargs):
            sync()
            forward_started=time.perf_counter()
            values = original_forward(*args,**kwargs)
            sync()
            state['timing_seconds']['vggt_synchronized_forward']=time.perf_counter()-forward_started
            E,K,depth,conf = values
            np.savez(output/'square_initial.npz',extrinsics=E,intrinsics=K,depth=depth,depth_conf=conf)
            records=[]
            # The pre-BA snapshot here is the full PINHOLE output before SIMPLE_PINHOLE conversion.
            scale = np.array([[518/1024,0,(518/1024-1)/2],[0,518/1024,(518/1024-1)/2],[0,0,1.]])
            for i,m in enumerate(mapping):
                A=scale@np.array(m['original_to_square_transform'])
                camera=Camera(np.linalg.solve(A,K[i].astype(float)),E[i,:,:3].astype(float),E[i,:,3].astype(float),*m['original_wh'])
                records.append(dict(image_id=m['image_id'],registered=True,camera=camera.as_dict(),reason=None))
            write_json(output/'pre_ba_export.json',{'images':records})
            if c['memory_schedule'] == 'retire_unused_vggt_after_forward':
                begin=time.perf_counter()
                # The demo never uses this model again. Release its storage without a CPU copy.
                args[0].to('meta')
                torch.cuda.empty_cache()
                sync()
                state['timing_seconds']['retire_unused_vggt_storage']=time.perf_counter()-begin
                state['memory_schedule']=c['memory_schedule']
            return values
        demo.run_VGGT = measured('vggt_forward_and_initial_snapshot',forward)
        original_tracks = demo.predict_tracks
        def predict(*args,**kwargs):
            values=original_tracks(*args,**kwargs)
            names=['xy','visibility','confidence','initial_xyz','rgb']
            np.savez(output/'tracks.npz',**{k:v for k,v in zip(names,values) if v is not None})
            state['track_shapes']={k:list(v.shape) for k,v in zip(names,values) if v is not None}
            return values
        demo.predict_tracks = measured('tracking_including_models_and_export',predict)
        for name in ['build_vggsfm_tracker','generate_rank_by_dino','initialize_feature_extractors','_forward_on_query','_augment_non_visible_frames']:
            setattr(tracking,name,measured(name,getattr(tracking,name)))
        original_ba = pycolmap.bundle_adjustment
        def adjust(reconstruction,options):
            native=output/'native_before_ba'; native.mkdir(); reconstruction.write(str(native))
            (output/'ba_options.txt').write_text(str(options.todict())+'\n')
            state['ba_solver']='official PyCOLMAP 3.10.0 CPU Ceres defaults; no solver substitution'
            result=measured('ceres_bundle_adjustment',original_ba)(reconstruction,options)
            native=output/'native_after_ba';native.mkdir();reconstruction.write(str(native))
            state['optimized_points']=len(reconstruction.points3D)
            ids=np.array(sorted(reconstruction.points3D),dtype=np.int64)
            xyz=np.array([reconstruction.points3D[int(i)].xyz for i in ids])
            rgb=np.array([reconstruction.points3D[int(i)].color for i in ids],dtype=np.uint8)
            np.savez(output/'optimized_points.npz',xyz=xyz,rgb=rgb,point_ids=ids)
            records=[]; errors=[]; observation_errors=[]
            for i,m in enumerate(mapping):
                im=reconstruction.images[i+1];cam=reconstruction.cameras[im.camera_id]
                E=im.cam_from_world.matrix()
                A=np.array(m['original_to_square_transform'])
                K=np.asarray(cam.calibration_matrix(),dtype=float)
                camera=Camera(np.linalg.solve(A,K),E[:,:3],E[:,3],*m['original_wh'])
                records.append(dict(image_id=m['image_id'],registered=True,camera=camera.as_dict(),reason=None))
                associated=[p for p in im.points2D if p.has_point3D()]
                if associated:
                    points=np.array([reconstruction.points3D[p.point3D_id].xyz for p in associated])
                    uv,_,valid=camera.project(points)
                    remapped=np.c_[uv,np.ones(len(uv))]@A.T
                    native_uv=(points@E[:,:3].T+E[:,3])@K.T
                    native_uv=native_uv[:,:2]/native_uv[:,2:]
                    errors.append(float(np.max(np.abs(remapped[:,:2]-native_uv))))
                    observation_errors.extend(np.linalg.norm(native_uv-np.array([p.xy for p in associated]),axis=1).tolist())
            if not errors or max(errors)>1e-6:
                raise ValueError('optimized camera coordinate conversion audit failed')
            state['validation']=dict(camera_export_max_abs_square_px=max(errors),post_ba_mean_observation_error_px=float(np.mean(observation_errors)),observations=len(observation_errors),optimized_points=len(ids),pointcloud_source='reconstruction.points3D after BA; official points.ply is not used')
            write_json(output/'export.json',{'images':records})
            return result
        pycolmap.bundle_adjustment=adjust
        args=SimpleNamespace(scene_dir=str(output),seed=c['seed'],use_ba=True,max_reproj_error=c['max_reproj_error'],shared_camera=c['shared_camera'],camera_type=c['camera_type'],vis_thresh=c['vis_thresh'],query_frame_num=c['query_frame_num'],max_query_pts=c['max_query_pts'],fine_tracking=c['fine_tracking'],conf_thres_value=5.)
        start=time.perf_counter()
        with torch.no_grad():
            demo.demo_fn(args)
        sync()
        state['timing_seconds']['official_demo_wall_including_nested_stages']=time.perf_counter()-start
        # Confirm native model can be read after serialization.
        reread=pycolmap.Reconstruction(str(output/'native_after_ba'))
        if len(reread.points3D)!=state['optimized_points'] or len(reread.images)!=len(mapping):
            raise ValueError('native reconstruction roundtrip failed')
        with np.load(output/'optimized_points.npz',allow_pickle=False) as points:
            actual=np.array([reread.points3D[int(i)].xyz for i in points['point_ids']])
            if not np.array_equal(actual,points['xyz']):
                raise ValueError('optimized point export differs from native model')
        state['validation']['native_point_roundtrip_exact']=True
        state['memory_bytes'].update(peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
        state['status']='complete';save()
    except Exception as exc:
        state.update(status='process_error',failure_kind='oom' if isinstance(exc,MemoryError) or type(exc).__name__=='OutOfMemoryError' else 'process_error',error=f'{type(exc).__name__}: {exc}')
        if 'torch' in locals() and torch.cuda.is_available():
            state['memory_bytes'].update(peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
        save();traceback.print_exc();raise SystemExit(1) from exc


if __name__=='__main__':
    main(Path(sys.argv[1]))
