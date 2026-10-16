"""Camera-conditioned checks of independently observed correspondences, without GT."""
import numpy as np
from ..geometry.camera import relative_pose


def fundamental(first, second):
    R,t=relative_pose(first,second)
    if np.linalg.norm(t)<=1e-12:
        return None
    x,y,z=t
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    F=np.linalg.inv(second.K).T@cross@R@np.linalg.inv(first.K)
    norm=np.linalg.norm(F)
    return F/norm if norm>1e-15 else None


def sampson_px(F,first_pixels,second_pixels):
    """Square-root Sampson distance in the common input pixel grid (not squared px)."""
    p=np.c_[first_pixels,np.ones(len(first_pixels))]
    q=np.c_[second_pixels,np.ones(len(second_pixels))]
    fp=p@F.T;fq=q@F
    denom=np.sum(fp[:,:2]**2+fq[:,:2]**2,axis=1)
    out=np.full(len(p),np.nan)
    valid=denom>1e-24
    out[valid]=np.abs(np.sum(q[valid]*fp[valid],axis=1))/np.sqrt(denom[valid])
    return out


def triangulation_checks(first,second,p,q):
    """DLT on actual matched pixels; retain negative depth as evidence, not filtered out."""
    p=np.asarray(p,float);q=np.asarray(q,float)
    a,b=first.projection,second.projection
    if np.linalg.norm(first.center-second.center)<=1e-12 or not len(p):
        return {'finite_points':0,'positive_depth_ratio':None,'angle_median_deg':None}
    design=np.stack([p[:,0,None]*a[2]-a[0],p[:,1,None]*a[2]-a[1],q[:,0,None]*b[2]-b[0],q[:,1,None]*b[2]-b[1]],axis=1)
    _,_,vh=np.linalg.svd(design)
    valid=np.abs(vh[:,-1,3])>1e-12
    xyz=vh[valid,-1,:3]/vh[valid,-1,3,None]
    if not len(xyz):return {'finite_points':0,'positive_depth_ratio':None,'angle_median_deg':None}
    positive=(first.world_to_camera(xyz)[:,2]>1e-12)&(second.world_to_camera(xyz)[:,2]>1e-12)
    r1=xyz-first.center;r2=xyz-second.center
    denom=np.linalg.norm(r1,axis=1)*np.linalg.norm(r2,axis=1)
    good=denom>1e-20
    angles=np.degrees(np.arccos(np.clip(np.sum(r1[good]*r2[good],axis=1)/denom[good],-1,1)))
    return {'finite_points':int(valid.sum()),'positive_depth_ratio':float(np.sum(positive)/len(p)),
            'angle_median_deg':float(np.median(angles)) if len(angles) else None}


def distribution(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return {'count':len(a),'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'mean':float(np.mean(a)) if len(a) else None}


def components(n,edges):
    neighbors=[set() for _ in range(n)]
    for a,b in edges:neighbors[a].add(b);neighbors[b].add(a)
    seen=set();sizes=[]
    for i in range(n):
        if i in seen:continue
        stack=[i];size=0;seen.add(i)
        while stack:
            v=stack.pop();size+=1
            for u in neighbors[v]-seen:seen.add(u);stack.append(u)
        sizes.append(size)
    return {'components':len(sizes),'largest_component_views':max(sizes) if sizes else 0,'isolated_views':sum(not a for a in neighbors)}
