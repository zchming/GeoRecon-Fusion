import copy,time
import numpy as np
from ..fusion.anchors import blend


def keep_unknown(keep,finite,observed):
    if keep.shape!=finite.shape or keep.shape!=observed.shape:raise ValueError('mask shapes differ')
    return keep | (finite & (observed==0))


def without_raw_support(points,anchors,strength):
    neutral=dict(anchors,raw_weight=np.ones_like(anchors['raw_weight'],dtype=float))
    return blend(points,neutral,strength)


def remove_evidence_gate(diagnostics):
    """Counterfactual only: leave identity/status/hash checks intact in original choose()."""
    result=copy.deepcopy(diagnostics)
    for d in result['methods'].values():
        d['status']='evidence_available';d.setdefault('summary',{})['evidence_pairs']=1
    return result


def fixed_pose_solve(problem,protocol,strength):
    from scipy.optimize import least_squares
    from ..optimization.joint import terms,decode,sparsity,objective
    cfg=protocol['solver'];n=len(problem['R0']);m=len(problem['x0']);o=len(problem['uv']);pc=6*(n-1)
    if m<cfg['min_unique_points'] or m>cfg['max_unique_points'] or o>cfg['max_observations'] or not o or any(np.sum(problem['obs_point']==j)<3 for j in range(m)):
        return None,{'accepted':False,'status':'insufficient_or_over_budget_constraints'}
    start=time.perf_counter();calls=0;last=np.zeros(3*m);history=[]
    def full(y):return np.r_[np.zeros(pc),y]
    def residual(y):return np.concatenate(terms(problem,full(y),protocol,strength)[0])
    initial=objective(residual(last))
    def fun(y):
        nonlocal calls,last
        calls+=1;last=y.copy()
        if time.perf_counter()-start>cfg['wall_budget_seconds']:raise TimeoutError('fixed-pose solve wall budget exceeded')
        return residual(y)
    def callback(y):history.append({'iteration':len(history)+1,'objective':objective(residual(y)),'wall_seconds':time.perf_counter()-start})
    bound=cfg['point_delta_component_bound_over_depth']
    try:
        result=least_squares(fun,last,jac='2-point',jac_sparsity=sparsity(problem)[:,pc:],bounds=(-bound,bound),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=1,x_scale='jac',
            max_nfev=cfg['max_nfev'],ftol=cfg['ftol'],xtol=cfg['xtol'],gtol=cfg['gtol'],callback=callback)
        y=result.x;status=int(result.status);message=str(result.message);nfev=int(result.nfev)
    except Exception as e:y=last;status=-99;message=type(e).__name__+': '+str(e);nfev=None
    x=full(y);r,t,points,pose=decode(problem,x);parts,z,uv=terms(problem,x,protocol,strength);final=objective(np.concatenate(parts))
    accepted=status>0 and np.isfinite(y).all() and np.isfinite(final) and final<=initial+1e-9 and np.all(z>0)
    _,_,uv0=terms(problem,np.zeros(pc+3*m),protocol,strength)
    record={'accepted':bool(accepted),'status':status,'message':message,'nfev':nfev,'residual_calls_including_finite_differences':calls,'initial_objective':initial,'final_objective':final,'wall_seconds':time.perf_counter()-start,
            'initial_reprojection_rms_native_px':float(np.sqrt(np.mean(np.sum((uv0-problem['uv'])**2,1)))),'final_reprojection_rms_native_px':float(np.sqrt(np.mean(np.sum((uv-problem['uv'])**2,1)))),'variables':len(y),'history':history,'all_poses_fixed':True}
    return {'R':r,'t':t,'points':points,'parameters':x,'pose_delta':pose},record
