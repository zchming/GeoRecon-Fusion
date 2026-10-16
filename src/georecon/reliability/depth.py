"""Cross-view Z-depth comparison, explicitly internal consistency, never self-reprojection evidence."""
from itertools import permutations
import numpy as np
from ..geometry.camera import Camera
from .geometry import distribution


def bilinear(depth,xy):
    xy=np.asarray(xy,float);h,w=depth.shape
    valid=np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<w-1)&(xy[:,1]<h-1)
    values=np.full(len(xy),np.nan)
    ids=np.flatnonzero(valid);p=xy[ids];x=np.floor(p[:,0]).astype(int);y=np.floor(p[:,1]).astype(int)
    a=p[:,0]-x;b=p[:,1]-y
    samples=np.stack([depth[y,x],depth[y,x+1],depth[y+1,x],depth[y+1,x+1]],axis=1)
    good=np.isfinite(samples).all(axis=1)&(samples>0).all(axis=1)
    values[ids[good]]=((1-a)*(1-b)*samples[:,0]+a*(1-b)*samples[:,1]+(1-a)*b*samples[:,2]+a*b*samples[:,3])[good]
    return values,np.isfinite(values)


def cross_view(depths,extrinsics,intrinsics,config):
    n,h,w=depths.shape
    if extrinsics.shape!=(n,3,4) or intrinsics.shape!=(n,3,3) or not np.isfinite(depths).all() or (depths<=0).any():
        raise ValueError('invalid native raw depth/camera arrays')
    cameras=[Camera(intrinsics[i].astype(float),extrinsics[i,:,:3].astype(float),extrinsics[i,:,3].astype(float),w,h) for i in range(n)]
    ys,xs=np.mgrid[config['depth_sample_stride']//2:h:config['depth_sample_stride'],config['depth_sample_stride']//2:w:config['depth_sample_stride']]
    xy=np.c_[xs.ravel(),ys.ravel()].astype(float)
    pairs=[]
    for i,j in permutations(range(n),2):
        world=cameras[i].unproject(xy,depths[i,ys,xs].ravel())
        uv,z,positive=cameras[j].project(world)
        target,inside=bilinear(depths[j],uv)
        valid=positive&inside
        residual=np.abs(z[valid]-target[valid])/np.maximum(np.abs(z[valid]),target[valid])
        enough=len(residual)>=config['min_depth_samples']
        pairs.append({'indices':[i,j],'status':'available' if enough else 'insufficient','source_samples':len(xy),'target_valid_samples':int(valid.sum()),'valid_fraction':float(np.mean(valid)),
                      'relative_z_error':distribution(residual) if enough else None,'fraction_within_relative_tolerance':float(np.mean(residual<=config['depth_relative_tolerance'])) if enough else None})
    return {'status':'available' if any(p['status']=='available' for p in pairs) else 'insufficient','scope':'raw depth on its native grid; compare source-projected Z against separately predicted target depth',
            'limitations':'same joint network outputs are correlated; occlusion and padding are NOT removed; low discrepancy is not independent correctness', 'pairs':pairs,
            'median_pair_relative_z_error':float(np.median([p['relative_z_error']['median'] for p in pairs if p['status']=='available'])) if any(p['status']=='available' for p in pairs) else None}
