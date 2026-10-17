"""Posthoc diagnostic math only; never exports or selects a deployment fusion rule."""
import numpy as np
from scipy.stats import rankdata,spearmanr


def auc(scores,positive):
    scores=np.asarray(scores,float);positive=np.asarray(positive,bool);valid=np.isfinite(scores);scores=scores[valid];y=positive[valid];a=int(y.sum());b=len(y)-a
    return {'valid':len(y),'positive':a,'negative':b,'auc':float((rankdata(scores)[y].sum()-a*(a+1)/2)/(a*b)) if a and b else None}


def signal_report(scores,gain,groups):
    scores=np.asarray(scores,float);gain=np.asarray(gain,float);valid=np.isfinite(scores)&np.isfinite(gain);y=gain>1e-9
    out={'direction':'larger frozen score predicts benefit; no sign fitting','benefit_AUROC':auc(scores,y),'spearman_gain':None,'top_fraction_profiles':[]}
    if valid.sum()>2 and np.ptp(scores[valid])>0 and np.ptp(gain[valid])>0:out['spearman_gain']=float(spearmanr(scores[valid],gain[valid]).statistic)
    # Keep tied scores together: never use GT or point index to split ties.
    for fraction in [.1,.25,.5,1.]:
        ids=np.flatnonzero(valid);n=max(1,int(np.ceil(len(ids)*fraction)))
        if not len(ids):continue
        cutoff=np.sort(scores[ids])[::-1][n-1];chosen=ids[scores[ids]>=cutoff];g=gain[chosen]
        out['top_fraction_profiles'].append({'requested_fraction':fraction,'actual_selected':len(chosen),'valid':len(ids),'score_cutoff':float(cutoff),'mean_gain_mm':float(g.mean()),'benefit_fraction':float((g>1e-9).mean()),'harm_fraction':float((g< -1e-9).mean()),'random_same_coverage_expected_mean_gain_mm':float(gain[ids].mean()),'diagnostic_only_not_a_policy':True})
    keys=np.unique(groups);gs=[];gg=[]
    for key in keys:
        m=(groups==key)&valid
        if m.any():gs.append(float(scores[m].mean()));gg.append(float(gain[m].mean()))
    out['unique_landmark_mean_AUROC']=auc(gs,np.asarray(gg)>1e-9)
    return out


def bilinear(grid,frame,uv):
    n,h,w,_=grid.shape;uv=np.asarray(uv,float);floor=np.floor(uv).astype(int);ix,iy=floor[:,0],floor[:,1];valid=(ix>=0)&(iy>=0)&(ix+1<w)&(iy+1<h)
    result=np.full((len(uv),3),np.nan);ids=np.flatnonzero(valid);a,b=uv[ids,0]-ix[ids],uv[ids,1]-iy[ids];v=frame[ids];x,y=ix[ids],iy[ids]
    result[ids]=((1-a)*(1-b))[:,None]*grid[v,y,x]+(a*(1-b))[:,None]*grid[v,y,x+1]+((1-a)*b)[:,None]*grid[v,y+1,x]+(a*b)[:,None]*grid[v,y+1,x+1]
    return result,valid


def candidates(raw,anchor,fraction,betas):
    # Frozen production blend uses float32 in-place arithmetic; reproduce this for each action.
    result=[]
    for beta in [*betas,None]:
        b=fraction if beta is None else np.full(len(raw),beta);x=raw.copy();x+=b[:,None]*(anchor-x);result.append(x)
    return np.stack(result)


def oracle(distances):
    distances=np.asarray(distances,float)
    if distances.ndim!=2 or not np.isfinite(distances).all():raise ValueError('finite action x point distances required')
    action=np.argmin(distances,axis=0);best=distances[action,np.arange(distances.shape[1])]
    return action,best


def summary(before,after,mask=None):
    before=np.asarray(before,float);after=np.asarray(after,float);valid=np.isfinite(before)&np.isfinite(after)
    if mask is not None:valid&=mask
    a,b=before[valid],after[valid]
    return {'count':int(valid.sum()),'mean_before_mm':float(a.mean()) if len(a) else None,'mean_after_mm':float(b.mean()) if len(b) else None,'median_before_mm':float(np.median(a)) if len(a) else None,'median_after_mm':float(np.median(b)) if len(b) else None,'mean_gain_mm':float((a-b).mean()) if len(a) else None,'improved':int((b<a-1e-9).sum()),'worsened':int((b>a+1e-9).sum()),'unchanged':int((np.abs(a-b)<=1e-9).sum()),'good_to_bad_2mm':int(((a<=2)&(b>2)).sum()),'bad_to_good_2mm':int(((a>2)&(b<=2)).sum())}
