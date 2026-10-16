"""GT-free, subset of grid positions updated by observed sparse track correspondences."""
import numpy as np
from ..geometry.alignment import Similarity
from ..geometry.camera import Camera
from ..filtering.points import evidence


def predicted_alignment(classical, raw, config):
    a={i['image_id']:Camera.from_dict(i['camera']) for i in classical['images'] if i['registered']}
    b={i['image_id']:Camera.from_dict(i['camera']) for i in raw['images'] if i['registered']}
    keys=sorted(a.keys()&b.keys())
    if len(keys)<3:
        return None,{'status':'fallback','reason':'fewer_than_three_shared_cameras'}
    x=np.array([a[k].center for k in keys]); y=np.array([b[k].center for k in keys])
    if np.linalg.matrix_rank(x-x.mean(0),tol=1e-10)<2 or np.linalg.matrix_rank(y-y.mean(0),tol=1e-10)<2:
        return None,{'status':'fallback','reason':'degenerate_camera_alignment'}
    # Relative predicted orientations determine gauge rotation. Centers alone can be ill-conditioned.
    rotation_mean=np.mean([b[k].R.T@a[k].R for k in keys],axis=0)
    u,_,vh=np.linalg.svd(rotation_mean);sign=np.ones(3);sign[-1]=np.linalg.det(u@vh)
    rotation=u@np.diag(sign)@vh
    xx=(x-x.mean(0))@rotation.T;yy=y-y.mean(0)
    scale=float(np.sum(xx*yy)/np.sum(xx*xx))
    if not np.isfinite(scale) or scale<=1e-12:
        return None,{'status':'fallback','reason':'nonpositive_orientation_conditioned_scale'}
    sim=Similarity(scale,rotation,y.mean(0)-scale*rotation@x.mean(0))
    diameter=float(np.max(np.linalg.norm(y[:,None]-y[None,:],axis=-1)))
    rmse=float(np.sqrt(np.mean(np.sum((sim.apply(x)-y)**2,axis=1))))
    relative=rmse/diameter if diameter>1e-12 else float('inf')
    angles=[float(np.degrees(np.arccos(np.clip((np.trace((a[k].R@sim.R.T)@b[k].R.T)-1)/2,-1,1)))) for k in keys]
    record={'status':'aligned','source':'predicted camera orientations plus centers; closed form; no GT/ICP', 'scale':sim.scale,'R':sim.R.tolist(),'t':sim.t.tolist(),
            'shared_ids':keys,'center_rmse':rmse,'raw_center_diameter':diameter,'relative_center_rmse':relative,'rotation_disagreement_deg':angles}
    if relative>config['max_center_rmse_over_raw_center_diameter'] or max(angles)>config['max_camera_rotation_disagreement_deg']:
        record.update(status='fallback',reason='predicted_camera_disagreement')
        return None,record
    return sim,record


def associate(points, depths, extrinsics, intrinsics, observations, colmap_mapping, raw_mapping, sim, config):
    n,h,w,_=points.shape;flat=points.reshape(-1,3)
    if n<2 or depths.shape[:3]!=(n,h,w) or extrinsics.shape!=(n,3,4) or intrinsics.shape!=(n,3,3):
        raise ValueError('incompatible raw arrays')
    by_raw={v['image_id']:(i,v) for i,v in enumerate(raw_mapping)}
    by_col={v['image_id']:v for v in colmap_mapping}
    support=evidence(points,depths,extrinsics,intrinsics,[.05])['support'][0]
    candidates=[];counts={'observations':len(observations['point_id']),'track_or_error_rejected':0,'out_of_grid':0,'geometry_rejected':0}
    for image_id in sorted(set(observations['image_id'].tolist())):
        if image_id not in by_raw or image_id not in by_col:continue
        frame,item=by_raw[image_id]
        if item['image_id']!=raw_mapping[frame]['image_id']:raise ValueError('frame mapping mismatch')
        rows=np.flatnonzero(observations['image_id']==image_id)
        err=observations['error'][rows];length=observations['track_length'][rows]
        good=np.isfinite(err)&(err>=0)&(err<=config['max_colmap_mean_reprojection_error_common_px'])&(length>=config['min_track_length'])&(length<=n)
        counts['track_or_error_rejected']+=int((~good).sum());rows=rows[good]
        if not len(rows):continue
        # COLMAP native half-pixel convention -> common zero-based centers -> original -> raw native.
        transform=np.array(item['original_to_model_transform'])@np.linalg.inv(np.array(by_col[image_id]['pixel_transform']))
        obsxy=observations['xy'][rows]-.5
        q=np.c_[obsxy,np.ones(len(rows))]@transform.T
        uv=q[:,:2]/q[:,2:3]
        finite=np.isfinite(uv).all(1)
        ix=np.floor(np.where(finite,uv[:,0],-1e6)+.5).astype(int)
        iy=np.floor(np.where(finite,uv[:,1],-1e6)+.5).astype(int)
        inside=finite&(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        counts['out_of_grid']+=int((~inside).sum());rows=rows[inside];ix=ix[inside];iy=iy[inside];uv=uv[inside]
        ids=frame*h*w+iy*w+ix;anchor=sim.apply(observations['xyz'][rows]);raw=flat[ids]
        c=anchor@extrinsics[frame,:,:3].T+extrinsics[frame,:,3];q=c@intrinsics[frame].T
        with np.errstate(divide='ignore',invalid='ignore'):projected=q[:,:2]/q[:,2:3]
        rawz=(raw@extrinsics[frame,:,:3].T+extrinsics[frame,:,3])[:,2]
        distance=np.linalg.norm(anchor-raw,axis=1)
        gate=np.isfinite(anchor).all(1)&np.isfinite(raw).all(1)&(rawz>0)&(c[:,2]>0)
        gate&=np.linalg.norm(projected-uv,axis=1)<=config['max_aligned_anchor_reprojection_error_native_px']
        gate&=distance<=config['max_3d_disagreement_over_raw_source_z']*rawz
        counts['geometry_rejected']+=int((~gate).sum())
        for pos in np.flatnonzero(gate):
            r=rows[pos];wc=(float(observations['track_length'][r])/n)/(1+(float(observations['error'][r])/2)**2)
            candidates.append((int(ids[pos]),-wc,int(observations['point_id'][r]),anchor[pos],wc,1+float(support[ids[pos]])/(n-1)))
    # Deterministic collision resolution. Do not average different sparse anchors into a new correspondence.
    candidates.sort(key=lambda v:(v[0],v[1],v[2]));chosen=[];seen=set()
    for c in candidates:
        if c[0] not in seen:chosen.append(c);seen.add(c[0])
    ids=np.array([c[0] for c in chosen],dtype=np.int64)
    return {'source_point_indices':ids,'anchor_xyz':np.array([c[3] for c in chosen],dtype=float).reshape(-1,3),
            'point_id':np.array([c[2] for c in chosen],dtype=np.int64),'classical_weight':np.array([c[4] for c in chosen]),
            'raw_weight':np.array([c[5] for c in chosen]),'raw_other_view_support':support[ids],
            'counts':dict(counts,eligible_observations=len(candidates),collision_discarded=len(candidates)-len(chosen),eligible_grid_points=len(ids))}


def blend(points, anchors, strength=None):
    """Return a source-dtype full grid: unavailable values/bytes unchanged; no deletion."""
    if strength is not None and (not np.isfinite(strength) or strength<0):raise ValueError('invalid fusion strength')
    out=points.reshape(-1,3).copy();ids=anchors['source_point_indices']
    wc=anchors['classical_weight'];wr=anchors['raw_weight']
    if np.any(~np.isfinite(wc)) or np.any(~np.isfinite(wr)) or np.any(wc<=0) or np.any(wr<=0):raise ValueError('nonpositive or nonfinite weights')
    fraction=np.full(len(ids),.5) if strength is None else strength*wc/(wr+strength*wc)
    out[ids]+=fraction[:,None]*(anchors['anchor_xyz']-out[ids])
    return out,fraction
