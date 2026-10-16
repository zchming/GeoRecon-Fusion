"""Cross-view consistency of actual point-head XYZ, never depth-derived replacement points."""
import numpy as np
from ..reliability.depth import bilinear


def evidence(points, depths, extrinsics, intrinsics, tolerances, batch=65536):
    points = np.asarray(points)
    depths = np.asarray(depths)
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    n, h, w = depths.shape
    if n < 2 or points.shape != (n,h,w,3) or extrinsics.shape != (n,3,4) or intrinsics.shape != (n,3,3):
        raise ValueError('incompatible native point/depth/camera arrays')
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError('nonfinite cameras')
    if batch < 1 or any(t <= 0 or t >= 1 for t in tolerances):
        raise ValueError('invalid filtering parameters')
    flat = points.reshape(-1,3)
    finite = np.isfinite(flat).all(axis=1)
    observed = np.zeros(len(flat),np.uint8)
    support = np.zeros((len(tolerances),len(flat)),np.uint8)
    for s in range(n):
        for start in range(s*h*w,(s+1)*h*w,batch):
            end = min(start+batch,(s+1)*h*w)
            p = flat[start:end].astype(np.float64)
            good = finite[start:end]
            for j in range(n):
                if j == s:
                    continue
                c = p @ extrinsics[j,:,:3].astype(float).T + extrinsics[j,:,3]
                z = c[:,2]
                q = c @ intrinsics[j].astype(float).T
                with np.errstate(invalid='ignore',divide='ignore'):
                    xy = q[:,:2] / q[:,2:3]
                target, inside = bilinear(depths[j],xy)
                valid = good & (z>0) & inside
                observed[start:end] += valid
                err = np.full(len(p),np.inf)
                err[valid] = np.abs(z[valid]-target[valid])/np.maximum(z[valid],target[valid])
                for k,t in enumerate(tolerances):
                    support[k,start:end] += valid & (err<=t)
    return {'finite':finite,'observed':observed,'support':support}


def classify(ev, tolerance_index, min_support):
    if min_support < 1:
        raise ValueError('support must be positive')
    keep = ev['finite'] & (ev['support'][tolerance_index]>=min_support)
    unknown = ev['finite'] & (ev['observed']==0)
    return keep, unknown
