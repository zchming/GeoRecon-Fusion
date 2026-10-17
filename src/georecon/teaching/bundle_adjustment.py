"""Teaching calibrated sparse BA. No GT, priors, filtering or registration changes."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from .two_view import known_K


def prepare(rotations,translations,intrinsics,xyz,frames,point_ids,pixels,seed):
    r=np.array(rotations,dtype=float,copy=True);t=np.array(translations,dtype=float,copy=True);k=np.array(intrinsics,dtype=float,copy=True);x=np.array(xyz,dtype=float,copy=True)
    f=np.asarray(frames);j=np.asarray(point_ids);uv=np.array(pixels,dtype=float,copy=True);n=len(r)
    if r.shape!=(n,3,3) or t.shape!=(n,3) or k.shape!=(n,3,3) or x.ndim!=2 or x.shape[1]!=3 or uv.shape!=(len(f),2) or j.shape!=f.shape or f.ndim!=1:raise ValueError('invalid BA array shapes')
    if not all(np.isfinite(v).all() for v in [r,t,k,x,uv]):raise ValueError('nonfinite BA input')
    if f.dtype.kind not in 'iu' or j.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=n) or np.any(j<0) or np.any(j>=len(x)):raise ValueError('invalid observation indices')
    if len(seed)!=2 or len(set(seed))!=2 or any(type(v) not in (int,np.int64) or not 0<=v<n for v in seed):raise ValueError('invalid gauge cameras')
    for ki,ri in zip(k,r):
        known_K(ki)
        if not np.allclose(ri@ri.T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(ri),1,atol=1e-6):raise ValueError('proper rotations required')
    if len(set(zip(f.tolist(),j.tolist())))!=len(f):raise ValueError('duplicate point observation in camera')
    centers=-np.einsum('nji,nj->ni',r,t);first,second=seed;baseline=centers[second]-centers[first];axis=int(np.argmax(np.abs(baseline)))
    cols={};cursor=0
    for v in range(n):
        if v==first:continue
        axes=[a for a in range(3) if not (v==second and a==axis)];cols[v]=(cursor,axes);cursor+=3+len(axes)
    depth=np.einsum('nij,nj->ni',r[f],x[j])+t[f]
    scale=float(np.median(np.abs(depth[:,2]))) if len(f) else 1.
    return {'R':r,'t':t,'K':k,'X':x,'f':f.astype(int),'j':j.astype(int),'uv':uv,'C':centers,'seed':list(seed),'anchor_axis':axis,'baseline':float(np.linalg.norm(baseline)),'scale':max(scale,1e-6),'cols':cols,'pc':cursor,'size':cursor+3*len(x)}


def decode(problem,delta):
    r=problem['R'].copy();c=problem['C'].copy();s=problem['scale']
    for v,(start,axes) in problem['cols'].items():
        r[v]=Rotation.from_rotvec(delta[start:start+3]).as_matrix()@r[v]
        c[v,axes]+=s*delta[start+3:start+3+len(axes)]
    t=-np.einsum('nij,nj->ni',r,c)
    # Preserve the fixed first camera byte-for-byte instead of round-tripping C.
    a=problem['seed'][0];t[a]=problem['t'][a]
    return r,t,problem['X']+s*delta[problem['pc']:].reshape(-1,3)


def residuals(problem,delta):
    r,t,x=decode(problem,delta);f,j=problem['f'],problem['j'];camera=np.einsum('nij,nj->ni',r[f],x[j])+t[f];q=np.einsum('nij,nj->ni',problem['K'][f],camera)
    z=q[:,2:3];floor=1e-9*problem['scale'];safe=np.where(np.abs(z)>floor,z,np.where(z<0,-floor,floor))
    return (q[:,:2]/safe-problem['uv']).ravel(),camera[:,2]


def objective(residual,scale):
    return float(scale**2*np.sum(np.sqrt(1+(residual/scale)**2)-1))


def sparsity(problem):
    f,j=problem['f'],problem['j'];a=lil_matrix((2*len(f),problem['size']),dtype=np.int8)
    for i,(v,p) in enumerate(zip(f,j)):
        a[2*i:2*i+2,problem['pc']+3*p:problem['pc']+3*p+3]=1
        if v in problem['cols']:
            start,axes=problem['cols'][v];a[2*i:2*i+2,start:start+3+len(axes)]=1
    return a.tocsr()


def solve(problem,cfg):
    start=time.perf_counter();n=len(problem['R']);m=len(problem['X']);o=len(problem['f']);zero=np.zeros(problem['size']);history=[]
    stats={'accepted':False,'reason':None,'solver_status':None,'converged':False,'cameras':n,'points':m,'observations':o,'gauge':{'fixed_camera':problem['seed'][0],'scale_anchor_camera':problem['seed'][1],'fixed_center_axis':problem['anchor_axis'],'constraint':'one component of second camera center relative to fixed first center is held at its initial value'},'history':history}
    def reject(reason):
        stats.update(reason=reason,wall_seconds=time.perf_counter()-start);return zero.copy(),stats
    if n<3:return reject('requires_at_least_three_registered_cameras')
    if m<cfg['min_points'] or m>cfg['max_points'] or o>cfg['max_observations']:return reject('insufficient_or_over_budget_structure')
    if problem['baseline']<1e-8:return reject('zero_seed_baseline')
    if np.linalg.matrix_rank(problem['X']-problem['X'].mean(0),tol=1e-8)<2:return reject('collinear_structure')
    f,j=problem['f'],problem['j']
    if any(np.unique(f[j==p]).size<2 for p in range(m)):return reject('point_has_fewer_than_two_views')
    reached={problem['seed'][0]}
    while True:
        before=len(reached)
        for p in range(m):
            views=set(f[j==p].tolist())
            if reached&views:reached |= views
        if len(reached)==before:break
    if len(reached)!=n:return reject('disconnected_observation_graph')
    initial,z=residuals(problem,zero)
    if np.any(z<=0):return reject('nonpositive_initial_depth')
    cost=objective(initial,cfg['loss_scale_px']);stats.update(initial_objective=cost,initial_rmse_px=float(np.sqrt(np.mean(initial.reshape(-1,2)**2)*2)))
    calls=0;last=zero.copy()
    class WallBudget(Exception):pass
    def fun(delta):
        nonlocal calls
        calls+=1
        if time.perf_counter()-start>cfg['wall_budget_seconds']:raise WallBudget('wall budget exceeded')
        return residuals(problem,delta)[0]
    def callback(delta):
        nonlocal last
        last=delta.copy();res,depth=residuals(problem,last);history.append({'iteration':len(history)+1,'objective':objective(res,cfg['loss_scale_px']),'positive_observations':int((depth>0).sum()),'elapsed_seconds':time.perf_counter()-start})
    try:
        fit=least_squares(fun,zero,jac='2-point',jac_sparsity=sparsity(problem),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=cfg['loss_scale_px'],x_scale='jac',max_nfev=cfg['max_nfev'],ftol=cfg['ftol'],xtol=cfg['xtol'],gtol=cfg['gtol'],callback=callback)
        last=fit.x;stats.update(solver_status=int(fit.status),converged=bool(fit.success),solver_message=str(fit.message),nfev=int(fit.nfev),optimality=float(fit.optimality))
    except WallBudget:
        stats.update(function_calls=calls);_,stats=reject('wall_budget_exceeded');return last,stats
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        stats.update(solver_error=repr(error),function_calls=calls);_,stats=reject('solver_error');return last,stats
    final,depth=residuals(problem,last);final_cost=objective(final,cfg['loss_scale_px']);r,t,x=decode(problem,last)
    accepted=all(np.isfinite(v).all() for v in [r,t,x,final]) and np.all(depth>0) and final_cost<=cost+1e-9
    stats.update(accepted=bool(accepted),reason=('accepted_converged' if fit.success else 'accepted_evaluation_budget_limited') if accepted else 'candidate_nonfinite_nonpositive_or_objective_increase',final_objective=final_cost if np.isfinite(final_cost) else None,final_rmse_px=float(np.sqrt(np.mean(final.reshape(-1,2)**2)*2)) if np.isfinite(final).all() else None,positive_observations=int((depth>0).sum()),function_calls=calls,wall_seconds=time.perf_counter()-start)
    return last,stats
