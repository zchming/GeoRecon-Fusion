"""Shared feature contracts and traditional matching, independent of GT."""
import hashlib
import numpy as np


class ResourceGuard(RuntimeError):
    pass


def array_digest(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def matching_inputs(features):
    return {k: features[k] for k in ['keypoints', 'descriptors_root', 'scales', 'oris', 'image_size']}


def validate_features(f, guard=4608):
    n = len(f['keypoints'])
    if n > guard:
        raise ResourceGuard('detector exceeds frozen feature guard; no truncation')
    shapes = {'keypoints': (n, 2), 'descriptors_raw': (n, 128),
              'scales': (n,), 'oris': (n,), 'image_size': (2,)}
    if 'descriptors_root' in f:
        shapes['descriptors_root'] = (n, 128)
    for key, shape in shapes.items():
        a = f[key]
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'invalid {key}: finite float32 {shape} required')
    if np.any(f['image_size'] <= 0) or np.any(f['keypoints'] < 0) or np.any(f['keypoints'] >= f['image_size']):
        raise ValueError('invalid image size or pixel coordinates')
    if np.any(f['scales'] <= 0) or np.any(f['oris'] < 0) or np.any(f['oris'] > 2*np.pi):
        raise ValueError('scale/angle convention violation')
    if 'descriptors_root' in f and n and not np.allclose(np.linalg.norm(f['descriptors_root'], axis=1), 1, atol=1e-5):
        raise ValueError('RootSIFT must have unit L2 norm')
    return n


def extract(rgb, spec, root_transform=None):
    import cv2
    import time
    start = time.perf_counter()
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(spec['opencv_threads'])
    cv2.setRNGSeed(spec['sift_seed'])
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 input required')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=spec['nfeatures_target'], nOctaveLayers=spec['n_octave_layers'],
                             contrastThreshold=spec['contrast_threshold'], edgeThreshold=spec['edge_threshold'],
                             sigma=spec['sigma'], enable_precise_upscale=spec['enable_precise_upscale'])
    kp, des = sift.detectAndCompute(gray, None)
    f = {'keypoints': np.array([p.pt for p in kp], np.float32).reshape(-1, 2),
         'descriptors_raw': np.empty((0, 128), np.float32) if des is None else des,
         'scales': np.array([p.size for p in kp], np.float32),
         'oris': np.deg2rad(np.array([p.angle for p in kp], np.float32)),
         'image_size': np.array([rgb.shape[1], rgb.shape[0]], np.float32)}
    validate_features(f)
    extraction = time.perf_counter() - start
    root_seconds = 0.0
    if root_transform is not None:
        start = time.perf_counter()
        f['descriptors_root'] = np.asarray(root_transform(f['descriptors_raw'].copy()), np.float32)
        root_seconds = time.perf_counter() - start
        validate_features(f)
    return f, {'extract_seconds': extraction, 'rootsift_seconds': root_seconds}


def ratio_match(a, b, ratio=.75):
    import cv2
    if min(len(a), len(b)) < 2:
        return np.empty((0, 2), np.int64), {'ratio_forward': 0, 'ratio_backward': 0}
    bf = cv2.BFMatcher(cv2.NORM_L2)
    def direction(x, y):
        return {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(x, y, k=2)
                if len(m) == 2 and m[0].distance < ratio*m[1].distance}
    forward, backward = direction(a, b), direction(b, a)
    pairs = np.array(sorted((i, j) for i, j in forward.items() if backward.get(j) == i), np.int64).reshape(-1, 2)
    return pairs, {'ratio_forward': len(forward), 'ratio_backward': len(backward)}


def validate_pairs(pairs, n, m):
    pairs = np.asarray(pairs)
    if pairs.dtype.kind not in 'iu' or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('integer Nx2 match indices required')
    if np.any(pairs < 0) or np.any(pairs[:, 0] >= n) or np.any(pairs[:, 1] >= m):
        raise ValueError('out-of-range feature index')
    if any(len(np.unique(pairs[:, k])) != len(pairs) for k in [0, 1]):
        raise ValueError('matches must be unique in both directions')
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int64)
