"""Pinned official BA adapter; no reference data enters the inference worker."""
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from .colmap import execute_worker, pixel_transform, validate_sparse_artifact
from .vggt import verify_repository
from ..data.scene import load_scene
from ..data.protocol import read_json, write_json, sha256, result_template, validate_result
from ..provenance import code_provenance


def square_geometry(width, height, size=1024):
    if min(width, height, size) <= 0:
        raise ValueError('positive dimensions required')
    side = max(width, height)
    left, top = (side-width)//2, (side-height)//2
    s = size/side
    return np.array([[s, 0, s*left+(s-1)/2], [0, s, s*top+(s-1)/2], [0, 0, 1.]])


def independent_square(image, size=1024):
    image = image.convert('RGB')
    side = max(image.size)
    canvas = Image.new('RGB', (side, side))
    canvas.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return np.asarray(canvas.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32).transpose(2, 0, 1)/255


def validate_config(c):
    if c['official_commit'] != 'a288dd0f14786c93483e45524328726ab7b1b4ce' or c['schema_version'] != 1 or c['pycolmap_version'] != '3.10.0':
        raise ValueError('unsupported official BA versions')
    if c.get('memory_schedule') not in {'retain_vggt_like_official_demo','retire_unused_vggt_after_forward'}:
        raise ValueError('memory schedule must be explicit')
    required = dict(device='cuda', autocast='bfloat16', parameter_dtype='float32', preprocessing='official_black_square_1024_then_bilinear_518', camera_type='SIMPLE_PINHOLE', shared_camera=False, fine_tracking=True, max_query_pts=4096, query_frame_num=8, vis_thresh=0.2, max_reproj_error=8.0)
    if any(c.get(k) != v for k, v in required.items()):
        raise ValueError('only explicit official demo defaults supported; resource changes require a named profile')
    if not 0 < c['gpu_allocator_limit_gib'] <= 10.5 or not np.isfinite(c['timeout_seconds']) or c['timeout_seconds'] <= 0 or type(c['num_threads']) is not int or c['num_threads'] <= 0 or type(c['seed']) is not int:
        raise ValueError('invalid resource budget')
    return c


def verify_vendor(path, commit):
    path = Path(path).resolve()
    def git(*a):
        return subprocess.check_output(['git', '-C', str(path), *a], text=True).strip()
    if git('rev-parse','HEAD') != commit or git('status','--porcelain'):
        raise ValueError('vendor checkout is not clean at pinned commit')
    return {'commit': commit, 'license_sha256': {p.name: sha256(p) for p in path.glob('LICENSE*') if p.is_file()}}


def run_vggt_ba(manifest_path, config_path, official_repo, checkpoint, assets, dinov2_repo, lightglue_repo, output):
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; attempts never overwritten')
    c = validate_config(read_json(config_path))
    asset_config = read_json(Path(__file__).resolve().parents[3]/'configs/assets/vggt_ba.json')
    source = verify_repository(official_repo, c['official_commit'])
    source['ba_files_sha256'] = {n: sha256(Path(official_repo)/n) for n in ['demo_colmap.py','vggt/dependency/track_predict.py','vggt/dependency/vggsfm_utils.py','vggt/dependency/np_to_pycolmap.py','requirements_demo.txt']}
    source['dinov2'] = verify_vendor(dinov2_repo, asset_config['dinov2_commit'])
    source['lightglue'] = verify_vendor(lightglue_repo, asset_config['lightglue_commit'])
    manifest, images = load_scene(manifest_path, preprocess=True)
    if len(images) not in c.get('allowed_input_counts', [3]):
        raise ValueError('input count outside explicit profile')
    (output/'images').mkdir(parents=True)
    mapping = []
    for i, (record, image) in enumerate(zip(manifest['images'], images)):
        name = f'{i:06d}.png'
        Image.fromarray(image).save(output/'images'/name)
        A = square_geometry(image.shape[1], image.shape[0]) @ pixel_transform(manifest, record)
        mapping.append(dict(image_id=record['image_id'], worker_name=name, original_wh=[record['width'],record['height']], common_wh=[image.shape[1],image.shape[0]], original_to_square_transform=A.tolist(), common_processed_sha256=sha256(output/'images'/name), source_sha256=record['sha256']))
    request = {k: str(Path(v).resolve()) for k,v in dict(official_repo=official_repo,checkpoint=checkpoint,assets=assets,dinov2_repo=dinov2_repo,lightglue_repo=lightglue_repo).items()}
    for name,value in [('config',c),('asset_config',asset_config),('input_mapping',mapping),('request',request),('source',source)]:
        write_json(output/(name+'.json'),value)
    result = result_template(manifest['scene_id'],sha256(manifest_path),[m['image_id'] for m in mapping],c['method_id'],kind=manifest['kind'])
    result.update(config=c, run_id=output.name, provenance={**code_provenance(manifest['kind']), 'official_source':source, 'gt_access':'none; reference is evaluated separately', 'pixel_convention':'official tracks inserted unchanged as integer centers; no additional COLMAP -0.5 shift'}, logs=['worker.log','worker_state.json','preflight.json','process.json','gpu_memory.csv'])
    write_json(output/'prediction.json',result)
    prep = time.perf_counter()-started
    monitor = None
    with (output/'gpu_memory.csv').open('w') as gpu_log:
        try:
            monitor = subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--loop-ms=100'],stdout=gpu_log,stderr=subprocess.DEVNULL)
        except OSError:
            pass
        try:
            process = execute_worker([sys.executable,'-m','georecon.baselines.vggt_ba_worker',str(output)],c['timeout_seconds'],output/'worker.log',dict(os.environ, HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS=str(c['num_threads']),PYTHONUNBUFFERED='1'))
        finally:
            if monitor:
                monitor.terminate()
                monitor.wait(timeout=5)
    write_json(output/'process.json',process)
    try:
        state = read_json(output/'worker_state.json')
    except (OSError,ValueError):
        state = {'error':'worker did not write state'}
    if process['timed_out']:
        result.update(status='timeout',failure_reason='official BA exceeded budget')
    elif process['returncode'] != 0 or state.get('status') != 'complete':
        result.update(status='oom' if state.get('failure_kind') == 'oom' else 'process_error',failure_reason=state.get('error','incomplete official BA'))
    else:
        try:
            payload = read_json(output/'export.json')
            if [i['image_id'] for i in payload['images']] != [i['image_id'] for i in mapping]:
                raise ValueError('export changed image order')
            count = validate_sparse_artifact(output/'optimized_points.npz')
            if count == 0:
                raise ValueError('BA exported no optimized points')
            result.update(status='success',failure_reason=None,images=payload['images'])
            result['artifacts']['pointcloud'] = {'path':'optimized_points.npz','sha256':sha256(output/'optimized_points.npz'),'semantics':'actual optimized reconstruction.points3D in arbitrary world units; NOT upstream visualization PLY'}
            result['artifacts']['tracks'] = {'path':'tracks.npz','sha256':sha256(output/'tracks.npz'),'semantics':'raw tracker observations/visibility; 1024 square grid; not calibrated confidence or reliability analysis'}
            validate_result(result)
        except (ValueError,KeyError,OSError) as exc:
            result.update(status='process_error',failure_reason=f'export validation: {exc}')
            result['images'] = result_template(manifest['scene_id'],sha256(manifest_path),[m['image_id'] for m in mapping],c['method_id'],kind=manifest['kind'])['images']
            result['artifacts'] = {k:None for k in result['artifacts']}
    result['versions'] = state.get('versions',{})
    result['timing_seconds'] = {'staging_and_source_checks':prep, **state.get('timing_seconds',{}),'worker_wall':process['wall_seconds'],'total':time.perf_counter()-started}
    samples = [int(x)*1024**2 for x in (output/'gpu_memory.csv').read_text().splitlines() if x.strip().isdigit()]
    result['memory_bytes'] = {**state.get('memory_bytes',{}),'peak_worker_rss_os':state.get('peak_rss_bytes'),'peak_worker_rss_sampled':process.get('peak_worker_rss_sampled_bytes'),'peak_device_used_sampled_including_desktop':max(samples) if samples else None}
    write_json(output/'prediction.json',validate_result(result))
    if result['status'] == 'success':
        before = read_json(output/'pre_ba_export.json')
        pre = dict(result,method_id='vggt_square_pre_ba',run_id=output.name+'_pre_ba',images=before['images'],artifacts={k:None for k in result['artifacts']})
        write_json(output/'pre_ba_prediction.json',validate_result(pre))
    return result
