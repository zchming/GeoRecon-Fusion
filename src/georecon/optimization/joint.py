"""Sparse BA with raw/anchor priors; no reference input, model loading or dense propagation."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def prepare(raw_xyz, fused_xyz, anchors, tracks, colmap_mapping, raw_mapping, rotations, translations, intrinsics, shape):
    n,h,w=shape;ids=anchors['source_point_indices'];pids,inverse=np.unique(anchors['point_id'],return_inverse=True)
    if len(ids)==0:return None
    raw=np.asarray(raw_xyz)[ids].astype(float);fused=np.asarray(fused_xyz)[ids].astype(float)
    x0=np.array([fused[inverse==i].mean(0) for i in range(len(pids))])
    aligned=np.array([anchors['anchor_xyz'][np.flatnonzero(inverse==i)[0]] for i in range(len(pids))])
    wc=np.array([anchors['classical_weight'][np.flatnonzero(inverse==i)[0]] for i in range(len(pids))])
    lookup={int(pid):i for i,pid in enumerate(pids)};by_col={i['image_id']:i for i in colmap_mapping};by_raw={i['image_id']:(f,i) for f,i in enumerate(raw_mapping)}
    obs_frame=[];obs_point=[];uv=[]
    for k,pid in enumerate(tracks['point_id']):
        image_id=str(tracks['image_id'][k])
        if int(pid) not in lookup or image_id not in by_raw:continue
        f,item=by_raw[image_id]
        transform=np.array(item['original_to_model_transform'])@np.linalg.inv(np.array(by_col[image_id]['pixel_transform']))
        q=transform@np.r_[tracks['xy'][k]-.5,1.];xy=q[:2]/q[2]
        if not np.isfinite(xy).all() or not (-.5<=xy[0]<w-.5 and -.5<=xy[1]<h-.5):continue
        obs_frame.append(f);obs_point.append(lookup[int(pid)]);uv.append(xy)
    source_frames=ids//(h*w)
    z=np.einsum('ij,ij->i',raw,rotations[source_frames,2])+translations[source_frames,2]
    if not np.isfinite(z).all() or np.any(z<=0):raise ValueError('nonpositive/nonfinite raw source point Z')
    normal=float(np.median(z))
    return {'ids':ids,'point_ids':pids,'inverse':inverse,'raw_prior':raw,'raw_weight':anchors['raw_weight'].astype(float),
            'anchor_prior':aligned,'classical_weight':wc,'x0':x0,'R0':np.asarray(rotations,float),'t0':np.asarray(translations,float),'K':np.asarray(intrinsics,float),
            'obs_frame':np.array(obs_frame,dtype=int),'obs_point':np.array(obs_point,dtype=int),'uv':np.array(uv,float).reshape(-1,2),'normalization':normal}


def decode(problem,x):
    n=len(problem['R0']);pose=np.zeros((n,6));pose[1:]=x[:6*(n-1)].reshape(-1,6)
    delta=Rotation.from_rotvec(pose[:,:3]).as_matrix()
    r=delta@problem['R0'];t=np.einsum('nij,nj->ni',delta,problem['t0'])+pose[:,3:]*problem['normalization']
    points=problem['x0']+x[6*(n-1):].reshape(-1,3)*problem['normalization']
    return r,t,points,pose


def terms(problem,x,protocol,strength):
    r,t,points,pose=decode(problem,x);f=problem['obs_frame'];j=problem['obs_point'];scale=problem['normalization'];cfg=protocol['residuals']
    c=np.einsum('nij,nj->ni',r[f],points[j])+t[f];q=np.einsum('nij,nj->ni',problem['K'][f],c)
    xy=q[:,:2]/np.maximum(q[:,2:3],scale*1e-6)
    reproj=(xy-problem['uv'])/cfg['reprojection_sigma_native_px']*np.sqrt(problem['classical_weight'][j,None])
    raw=(points[problem['inverse']]-problem['raw_prior'])/(scale*cfg['raw_xyz_sigma_over_depth'])*np.sqrt(problem['raw_weight'][:,None])
    anchor=(points-problem['anchor_prior'])/(scale*cfg['classical_xyz_sigma_over_depth'])*np.sqrt(problem['classical_weight'][:,None]*protocol['fusion_strength'])
    pose_prior=np.concatenate([pose[1:,:3]/np.radians(cfg['pose_rotation_sigma_deg']),pose[1:,3:]/cfg['pose_translation_sigma_over_depth']],axis=1)*np.sqrt(strength)
    depth=np.maximum(scale*.0001-c[:,2],0)/(scale*.001)
    return [reproj.ravel(),raw.ravel(),anchor.ravel(),pose_prior.ravel(),depth],c[:,2],xy


def objective(residual):
    return float(np.sum(np.sqrt(1+np.asarray(residual)**2)-1))


def sparsity(problem):
    n=len(problem['R0']);m=len(problem['x0']);a=len(problem['ids']);o=len(problem['uv']);pc=6*(n-1)
    result=lil_matrix((2*o+3*a+3*m+pc+o,pc+3*m),dtype=np.int8)
    def obs_rows(start,k,width):
        j=problem['obs_point'][k];f=problem['obs_frame'][k]
        result[start:start+width,pc+3*j:pc+3*j+3]=1
        if f:result[start:start+width,6*(f-1):6*f]=1
    for k in range(o):obs_rows(2*k,k,2)
    offset=2*o
    for k,j in enumerate(problem['inverse']):result[offset+3*k:offset+3*k+3,pc+3*j:pc+3*j+3]=1
    offset+=3*a
    for j in range(m):result[offset+3*j:offset+3*j+3,pc+3*j:pc+3*j+3]=1
    offset+=3*m
    for k in range(pc):result[offset+k,k]=1
    offset+=pc
    for k in range(o):obs_rows(offset+k,k,1)
    return result.tocsr()


def solve(problem,protocol,strength):
    cfg=protocol['solver'];start=time.perf_counter();n=len(problem['R0']);m=len(problem['x0']);o=len(problem['uv']);pc=6*(n-1)
    if m<cfg['min_unique_points'] or m>cfg['max_unique_points'] or o>cfg['max_observations'] or o==0:
        return None,{'accepted':False,'status':'insufficient_or_over_budget_constraints','unique_points':m,'observations':o}
    if any(np.sum(problem['obs_point']==j)<3 for j in range(m)):
        return None,{'accepted':False,'status':'insufficient_track_observations'}
    x0=np.zeros(pc+3*m);bounds=np.r_[np.tile(np.r_[np.full(3,np.radians(cfg['pose_rotation_component_bound_deg'])),np.full(3,cfg['pose_translation_component_bound_over_depth'])],n-1),np.full(3*m,cfg['point_delta_component_bound_over_depth'])]
    calls=0;history=[];last=x0.copy()
    class BudgetExceeded(Exception):pass
    def fun(x):
        nonlocal calls,last
        calls+=1;last=x.copy()
        if time.perf_counter()-start>cfg['wall_budget_seconds']:raise BudgetExceeded('wall budget exceeded')
        return np.concatenate(terms(problem,x,protocol,strength)[0])
    initial=objective(fun(x0))
    def callback(x):
        history.append({'iteration':len(history)+1,'objective':objective(np.concatenate(terms(problem,x,protocol,strength)[0])),'elapsed_seconds':time.perf_counter()-start})
    try:
        result=least_squares(fun,x0,jac='2-point',jac_sparsity=sparsity(problem),bounds=(-bounds,bounds),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=1,
            x_scale='jac',max_nfev=cfg['max_nfev'],ftol=cfg['ftol'],xtol=cfg['xtol'],gtol=cfg['gtol'],callback=callback)
        x=result.x;status=int(result.status);message=str(result.message);nfev=int(result.nfev);optimality=float(result.optimality)
    except Exception as e:
        x=last;status=-99;message=type(e).__name__+': '+str(e);nfev=None;optimality=None
    residuals,z,xy=terms(problem,x,protocol,strength);residual=np.concatenate(residuals);final=objective(residual)
    r,t,points,pose=decode(problem,x)
    accepted=status>0 and np.isfinite(x).all() and np.isfinite(residual).all() and final<=initial+1e-9 and np.all(z>0)
    _,_,initial_xy=terms(problem,x0,protocol,strength)
    record={'accepted':bool(accepted),'status':status,'message':message,'nfev':nfev,'residual_calls_including_finite_differences':calls,'optimality':optimality,
            'initial_objective':initial,'final_objective':final,'objective_terms_final':{name:objective(v) for name,v in zip(['reprojection','raw_xyz_prior','classical_xyz_prior','pose_prior','positive_depth'],residuals)},
            'initial_reprojection_rms_native_px':float(np.sqrt(np.mean(np.sum((initial_xy-problem['uv'])**2,axis=1)))),
            'final_reprojection_rms_native_px':float(np.sqrt(np.mean(np.sum((xy-problem['uv'])**2,axis=1)))),
            'min_observation_depth':float(z.min()),'unique_points':m,'observations':o,'variables':len(x),'normalization_raw_units':problem['normalization'],
            'max_rotation_delta_deg':float(np.max(np.linalg.norm(pose[:,:3],axis=1))*180/np.pi),'max_translation_delta_over_depth':float(np.max(np.linalg.norm(pose[:,3:],axis=1))),
            'wall_seconds':time.perf_counter()-start,'history':history}
    return {'R':r,'t':t,'points':points,'parameters':x,'pose_delta':pose},record


def scatter(before,problem,points):
    out=np.array(before,copy=True)
    if problem is not None:out[problem['ids']]=points[problem['inverse']]
    return out
