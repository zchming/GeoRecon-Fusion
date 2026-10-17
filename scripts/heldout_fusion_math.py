"""Observation-level cross-fitting on fixed cameras; diagnostic copies only, no GT."""
import numpy as np
from georecon.reliability.depth import bilinear


def split_groups(names,source,heldout):
    names=np.asarray(names)
    if source==heldout:raise ValueError('source view cannot be held out')
    train=np.flatnonzero(names!=heldout);test=np.flatnonzero(names==heldout)
    return train,test,bool(len(test) and source in names[train] and len(set(names[train].tolist()))>=2)


def project(x,camera):
    q=np.asarray(x)@camera['R'].T+camera['t'];p=q@camera['K'].T
    with np.errstate(divide='ignore',invalid='ignore'):uv=p[...,:2]/p[...,2:3]
    return uv,q[...,2]


def triangulate(cameras,xy):
    rows=[]
    for camera,p in zip(cameras,xy):
        u=np.linalg.solve(camera['K'],np.r_[p,1.]);u=u[:2]/u[2];P=np.c_[camera['R'],camera['t']];rows.extend([u[0]*P[2]-P[0],u[1]*P[2]-P[1]])
    a=np.array(rows);_,s,v=np.linalg.svd(a)
    if len(s)<4 or s[-2]<=s[0]*1e-12 or abs(v[-1,3])<=1e-12:return None
    x=v[-1,:3]/v[-1,3]
    return x if np.isfinite(x).all() else None


def depth_support(raw,source_frame,omitted,depth,extrinsics,intrinsics):
    supported=0;observed=0
    for v in range(len(depth)):
        if v in [source_frame,omitted]:continue
        c=np.asarray(raw,float)@extrinsics[v,:,:3].astype(float).T+extrinsics[v,:,3];q=c@intrinsics[v].astype(float).T
        if c[2]<=0 or not np.isfinite(q).all():continue
        target,valid=bilinear(depth[v],(q[:2]/q[2])[None])
        if valid[0]:observed+=1;supported+=int(abs(c[2]-target[0])/max(c[2],target[0])<=.05)
    return supported,observed


def construct(rows,names,xy,cameras,sim,raw,source_camera,source_uv,support,n_images,alpha,cfg):
    info={'reason':None,'training_rows':list(map(int,rows)),'training_unique_views':len(set(names[rows].tolist()))}
    def fail(reason):info['reason']=reason;return None,info
    if sim is None:return fail('alignment_rejected')
    if info['training_unique_views']<2:return fail('fewer_than_two_construction_views')
    # Preserve old track-length/error gates, using construction rows only. Duplicates remain rows.
    if len(rows)<cfg['min_track_length'] or len(rows)>n_images:return fail('construction_track_length_gate')
    x=triangulate([cameras[str(names[i])] for i in rows],xy[rows])
    if x is None:return fail('triangulation_degenerate')
    residual=[]
    for i in rows:
        uv,z=project(x,cameras[str(names[i])])
        if z<=0 or not np.isfinite(uv).all():return fail('nonpositive_construction_depth')
        residual.append(float(np.linalg.norm(uv-xy[i])))
    e=float(np.mean(residual));info['construction_mean_error_common_px']=e
    if e>cfg['max_colmap_mean_reprojection_error_common_px']:return fail('construction_reprojection_gate')
    anchor=sim.apply(x[None])[0];uv,z=project(anchor,source_camera);_,rawz=project(raw,source_camera);info.update(classical_xyz=x.tolist(),aligned_xyz=anchor.tolist(),source_reprojection_native_px=float(np.linalg.norm(uv-source_uv)),relative_disagreement=float(np.linalg.norm(anchor-raw)/rawz) if rawz>0 else None)
    if z<=0 or rawz<=0 or not np.isfinite(anchor).all():return fail('nonpositive_source_depth')
    if info['source_reprojection_native_px']>cfg['max_aligned_anchor_reprojection_error_native_px']:return fail('source_reprojection_gate')
    if info['relative_disagreement']>cfg['max_3d_disagreement_over_raw_source_z']:return fail('source_3d_disagreement_gate')
    wc=(len(rows)/n_images)/(1+(e/2)**2);wr=1+support/(n_images-1);beta=alpha*wc/(wr+alpha*wc);fused=np.array(raw,dtype=np.float32,copy=True);fused+=beta*(anchor-fused)
    info.update(classical_weight=wc,raw_weight=wr,beta=float(beta),raw_depth_support=int(support));return fused,info


def witness_score(raw,candidate,uv,camera,width,height):
    if candidate is None:return None,'no_candidate'
    before,zb=project(raw,camera);after,za=project(candidate,camera)
    if zb<=0 or za<=0 or not np.isfinite(before).all() or not np.isfinite(after).all():return None,'nonpositive_or_nonfinite_witness_projection'
    if not (0<=before[0]<width and 0<=before[1]<height and 0<=after[0]<width and 0<=after[1]<height):return None,'witness_projection_outside_image'
    # Aggregate all observations in the held-out image as one dependent group.
    return float(np.linalg.norm(before[None]-uv,axis=1).mean()-np.linalg.norm(after[None]-uv,axis=1).mean()),None
