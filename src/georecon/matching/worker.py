"""CPU/GPU workers read only sanitized inputs; reference scoring is separate."""
import argparse
import json
import importlib.metadata
import os
from pathlib import Path
import resource
import sys
import time
import traceback

START = time.perf_counter()


def check_access(event, args, seen, calibration_source=None):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise PermissionError('network forbidden in offline matching worker')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).replace('\\', '/')
        forbidden = any(word in name.lower() for word in ['reference.json', 'evaluation.json', 'results.json', 'ground_truth', 'stage6_truth'])
        if '/calibration/' in name.lower():
            # Kornia imports Python calibration operators, not experiment calibration data.
            # Permit only source/bytecode under this installed dependency's exact directory.
            path = Path(name).resolve()
            library_code = calibration_source is not None and path.is_relative_to(calibration_source) and path.suffix in {'.py', '.pyc'}
            forbidden = forbidden or not library_code
        if forbidden:
            raise PermissionError('reference/evaluation read forbidden in matching worker')
        if 'georecon' in name or '/VisionGeometryLab/' in name:
            seen.add(name)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def read_features(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def official_root(descriptors):
    import torch
    from lightglue.sift import sift_to_rootsift
    return sift_to_rootsift(torch.from_numpy(descriptors)).numpy()


class Worker:
    def __init__(self, bundle, gpu=False):
        import cv2
        self.bundle = bundle
        self.reads = set()
        self.gpu = gpu
        self.model = None
        cv2.setNumThreads(8)
        cv2.ocl.setUseOpenCL(False)
        if gpu:
            self.load_model()

    def load_model(self):
        import torch
        from lightglue import LightGlue
        start = time.perf_counter()
        torch.set_num_threads(8)
        torch.manual_seed(42)
        properties = torch.cuda.get_device_properties(0)
        budget = self.bundle['budget']['gpu_allocator_bytes']
        if budget >= properties.total_memory:
            raise ValueError('physical GPU cannot support frozen allocator ceiling')
        torch.cuda.set_per_process_memory_fraction(budget/properties.total_memory)
        torch.cuda.reset_peak_memory_stats()
        state = torch.load(self.bundle['checkpoint'], map_location='cpu', weights_only=True)
        old = torch.hub.load_state_dict_from_url
        def offline(url, *args, **kwargs):
            if url != self.bundle['weights_source']:
                raise PermissionError('unapproved weight URL')
            return state
        torch.hub.load_state_dict_from_url = offline
        try:
            self.model = LightGlue(features='sift').eval().cuda()
        finally:
            torch.hub.load_state_dict_from_url = old
        canonical = state
        for i in range(9):
            canonical = {k.replace(f'self_attn.{i}', f'transformers.{i}.self_attn').replace(f'cross_attn.{i}', f'transformers.{i}.cross_attn'): v for k, v in canonical.items()}
        checked = self.model.load_state_dict(canonical, strict=False)
        if set(checked.missing_keys) & set(dict(self.model.named_parameters())) or checked.unexpected_keys:
            raise ValueError('weight parameter coverage mismatch')
        if vars(self.model.conf) != self.bundle['matcher_config']:
            raise ValueError('matcher configuration differs from frozen protocol')
        torch.cuda.synchronize()
        self.load_record = {'model_load_seconds': time.perf_counter()-start,
            'weight_missing_buffers': checked.missing_keys, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'device': torch.cuda.get_device_name(0),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32}

    def features(self, item, include_root=True):
        import numpy as np
        from PIL import Image
        from .features import extract
        start = time.perf_counter()
        with Image.open(item['path']) as image:
            rgb = np.array(image.convert('RGB'))
        f, timing = extract(rgb, self.bundle['features'], official_root if include_root else None)
        timing['decode_extract_root_seconds'] = time.perf_counter()-start
        return f, timing

    def match(self, first, second, arm):
        import numpy as np
        from .features import ratio_match, validate_pairs
        if arm != 'C':
            start = time.perf_counter()
            key = 'descriptors_raw' if arm == 'A' else 'descriptors_root'
            pairs, statistics = ratio_match(first[key], second[key])
            elapsed = time.perf_counter()-start
            return pairs, statistics, {'matcher_seconds': elapsed, 'h2d_seconds': 0., 'd2h_seconds': 0., 'match_total_seconds': elapsed}, {}
        if not len(first['keypoints']) or not len(second['keypoints']):
            return np.empty((0, 2), np.int64), {'not_run_empty_features': True}, {'matcher_seconds': 0., 'h2d_seconds': 0., 'd2h_seconds': 0., 'match_total_seconds': 0.}, {}
        import torch
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        start = time.perf_counter()
        gpu = []
        for f in [first, second]:
            gpu.append({k: torch.from_numpy(f['descriptors_root'] if k == 'descriptors' else f[k])[None].cuda()
                        for k in ['keypoints', 'descriptors', 'scales', 'oris', 'image_size']})
        torch.cuda.synchronize()
        h2d = time.perf_counter()-start
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model({'image0': gpu[0], 'image1': gpu[1]})
        torch.cuda.synchronize()
        matcher_seconds = time.perf_counter()-start
        start = time.perf_counter()
        pairs = out['matches'][0].cpu().numpy()
        scores = out['scores'][0].cpu().numpy()
        arrays = {k: out[k][0].cpu().numpy() for k in ['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'prune0', 'prune1']}
        d2h = time.perf_counter()-start
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError('invalid learned matching scores')
        if not np.array_equal(arrays['matches0'][pairs[:, 0]], pairs[:, 1]) or not np.array_equal(arrays['matches1'][pairs[:, 1]], pairs[:, 0]):
            raise ValueError('learned matches are not mutually indexed')
        if (arrays['matches0'] >= 0).sum() != len(pairs) or (arrays['matches1'] >= 0).sum() != len(pairs):
            raise ValueError('learned output counts disagree')
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        arrays['scores'] = scores[order]
        record = {'stop_layer': int(out['stop']), 'baseline_allocated_bytes': baseline_allocated,
            'baseline_reserved_bytes': baseline_reserved, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        pairs = validate_pairs(pairs, len(first['keypoints']), len(second['keypoints']))
        del out, gpu
        return pairs, record, {'matcher_seconds': matcher_seconds, 'h2d_seconds': h2d, 'd2h_seconds': d2h,
                               'match_total_seconds': h2d+matcher_seconds+d2h}, arrays

    def handle(self, request):
        import numpy as np
        from georecon.data.protocol import write_json, sha256
        from .features import array_digest, matching_inputs, validate_pairs, validate_features
        from ..teaching.two_view import reconstruct
        start = time.perf_counter()
        out = Path(request['output'])
        out.mkdir(parents=True, exist_ok=False)
        record = {'status': 'running', 'operation': request['operation'], 'geometry': [], 'model': getattr(self, 'load_record', None)}
        try:
            if request['operation'] == 'feature':
                f, timing = self.features(self.bundle['images'][request['image_key']])
                np.savez_compressed(out/'features.npz', **f)
                record.update(status='success', timing=timing, keypoints=len(f['keypoints']),
                    root_input_digest=array_digest(matching_inputs(f)), feature_sha256=sha256(out/'features.npz'))
            else:
                case = request['case']
                if request.get('cold'):
                    data = [self.features(self.bundle['images'][key], request['arm'] != 'A') for key in case['image_keys']]
                    features = [v[0] for v in data]
                    record['cold_feature_timings'] = [v[1] for v in data]
                    for side, f in enumerate(features):
                        np.savez_compressed(out/f'features_{side}.npz', **f)
                else:
                    features = [read_features(self.bundle['feature_paths'][key]) for key in case['image_keys']]
                for f in features:
                    validate_features(f)
                if request['arm'] in ['B', 'C']:
                    record['root_input_digests'] = [array_digest(matching_inputs(f)) for f in features]
                record['keypoint_counts'] = [len(f['keypoints']) for f in features]
                pairs, statistics, timing, arrays = self.match(*features, request['arm'])
                pairs = validate_pairs(pairs, *record['keypoint_counts'])
                p, q = features[0]['keypoints'][pairs[:, 0]], features[1]['keypoints'][pairs[:, 1]]
                np.savez_compressed(out/'matches.npz', indices=pairs, pixels_first=p, pixels_second=q, **arrays)
                record.update(matches=len(pairs), matching=statistics, timing=timing, matches_sha256=sha256(out/'matches.npz'))
                if request['operation'] != 'warmup':
                    for seed in request['seeds']:
                        cfg = dict(self.bundle['backend_config'], seed=seed)
                        keys = case['image_keys']
                        geom_start = time.perf_counter()
                        value = reconstruct(p, q, self.bundle['images'][keys[0]]['K'], self.bundle['images'][keys[1]]['K'], cfg)
                        elapsed = time.perf_counter()-geom_start
                        geom = {'seed': seed, 'status': value['status'], 'reason': value['reason'], 'statistics': value['stats'], 'seconds': elapsed}
                        arrays = dict(value['arrays'])
                        if value['R'] is not None:
                            arrays['accepted_R'], arrays['accepted_t'] = value['R'], value['t']
                        np.savez_compressed(out/f'geometry_{seed}.npz', **arrays)
                        geom['arrays_sha256'] = sha256(out/f'geometry_{seed}.npz')
                        record['geometry'].append(geom)
                record['status'] = 'success'
        except Exception as error:
            name = type(error).__name__
            record.update(status='oom' if name == 'OutOfMemoryError' else 'resource_guard' if name == 'ResourceGuard' else 'process_error',
                          error=repr(error), traceback=traceback.format_exc())
        record.update(worker_seconds=time.perf_counter()-start, worker_age_seconds=time.perf_counter()-START,
            peak_rss_bytes=rss(), gt_guard=True, file_accesses=sorted(self.reads))
        write_json(out/'record.json', record)
        return {'status': record['status'], 'record': str(out/'record.json'), 'peak_rss_bytes': record['peak_rss_bytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'cold'], required=True)
    parser.add_argument('--request')
    args = parser.parse_args()
    bundle = json.loads(Path(args.bundle).read_text())
    reads = set()
    try:
        calibration_source = Path(importlib.metadata.distribution('kornia').locate_file('kornia/geometry/calibration')).resolve()
    except importlib.metadata.PackageNotFoundError:
        calibration_source = None
    sys.addaudithook(lambda event, data: check_access(event, data, reads, calibration_source))
    request = json.loads(Path(args.request).read_text()) if args.request else None
    worker = Worker(bundle, args.mode == 'gpu' or (request and request['arm'] == 'C'))
    worker.reads = reads
    if args.mode == 'cold':
        reply = worker.handle(request)
        print(json.dumps(reply), flush=True)
        return
    print(json.dumps({'status': 'ready', 'model': getattr(worker, 'load_record', None), 'peak_rss_bytes': rss()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        print(json.dumps(worker.handle(request)), flush=True)


if __name__ == '__main__':
    main()
