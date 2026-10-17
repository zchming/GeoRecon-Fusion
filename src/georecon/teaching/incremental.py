"""Known-K teaching incremental SfM: immutable poses/points, no BA or GT input."""
import numpy as np
from .two_view import known_K, normalize, reconstruct, triangulate_dlt


def match(a, b, ratio):
    import cv2
    if a is None or b is None or min(len(a),len(b))<2:return np.empty((0,2),dtype=np.int64)
    bf=cv2.BFMatcher(cv2.NORM_L2)
    def one(x,y):
        return {m[0].queryIdx:m[0].trainIdx for m in bf.knnMatch(x,y,k=2) if len(m)==2 and m[0].distance<ratio*m[1].distance}
    f,g=one(a,b),one(b,a)
    return np.array(sorted((i,j) for i,j in f.items() if g.get(j)==i),dtype=np.int64).reshape(-1,2)


def collect_links(candidates):
    """Reject ambiguous keypoint/landmark identities in both directions."""
    by_key={};by_point={}
    for key,point in candidates:
        by_key.setdefault(int(key),set()).add(int(point));by_point.setdefault(int(point),set()).add(int(key))
    return np.array(sorted((key,next(iter(points))) for key,points in by_key.items() if len(points)==1 and len(by_point[next(iter(points))])==1),dtype=np.int64).reshape(-1,2)


def project(x,k,r,t):
    c=np.asarray(x)@r.T+t;q=c@k.T
    with np.errstate(divide='ignore',invalid='ignore'): uv=q[:,:2]/q[:,2:3]
    return uv,c[:,2]


def pnp(x,uv,k,cfg):
    import cv2
    x=np.asarray(x,float);uv=np.asarray(uv,float);k=known_K(k)
    if x.shape!=(len(uv),3) or uv.shape!=(len(x),2) or not np.isfinite(x).all():raise ValueError('finite Nx3/Nx2 required')
    normalized=normalize(uv,k);mask=np.zeros(len(x),bool)
    out={'registered':False,'reason':'insufficient_2d3d','R':None,'t':None,'inliers':mask,'candidate_count':len(x)}
    if len(x)<cfg['pnp_min_inliers']:return out
    if np.linalg.matrix_rank(x-x.mean(0),tol=1e-8)<2:out['reason']='collinear_3d';return out
    cv2.setRNGSeed(cfg['seed'])
    ok,rv,t,ids=cv2.solvePnPRansac(np.ascontiguousarray(x),np.ascontiguousarray(normalized),np.eye(3),None,iterationsCount=cfg['pnp_iterations'],reprojectionError=cfg['pnp_ransac_px']/np.mean([k[0,0],k[1,1]]),confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or ids is None:out['reason']='pnp_ransac_failed';return out
    r=cv2.Rodrigues(rv)[0];t=t.ravel();pred,z=project(x,k,r,t);err=np.linalg.norm(pred-uv,axis=1)
    mask[ids.ravel()]=True;mask &= (z>1e-9)&np.isfinite(err)&(err<=cfg['max_reprojection_px'])
    out.update(R=r,t=t,inliers=mask,reprojection_px=err,depths=z)
    if mask.sum()<cfg['pnp_min_inliers']:out['reason']='insufficient_positive_reprojection_inliers';return out
    out.update(registered=True,reason=None);return out


def reconstruct_incremental(images,ks,seed,cfg):
    import cv2
    cv2.setNumThreads(cfg['opencv_threads']);cv2.ocl.setUseOpenCL(False)
    sift=cv2.SIFT_create(nfeatures=cfg['sift_features']);features=[]
    for im in images:
        kp,des=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),None)
        features.append((np.array([v.pt for v in kp],float).reshape(-1,2),des))
    cache={};debug={};steps=[]
    def pairs(i,j):
        if (i,j) not in cache:
            cache[i,j]=match(features[i][1],features[j][1],cfg['ratio_threshold']);debug[f'matches_{i}_{j}']=cache[i,j]
        return cache[i,j]
    a,b=seed;m=pairs(a,b);base=reconstruct(features[a][0][m[:,0]],features[b][0][m[:,1]],ks[a],ks[b],cfg)
    debug.update({'seed_'+k:v for k,v in base['arrays'].items()})
    poses={};xyz=[];observations={};origins=[]
    steps.append({'view_indices':[a,b],'operation':'seed','status':base['status'],'reason':base['reason'],'stats':base['stats']})
    if base['status']!='success':return poses,np.empty((0,3)),observations,steps,debug,origins
    poses[a]=(np.eye(3),np.zeros(3));poses[b]=(base['R'],base['t'])
    xyz=list(base['arrays']['accepted_xyz'].copy())
    for point,row in enumerate(base['arrays']['kept_match_indices']):
        observations[a,int(m[row,0])]=point;observations[b,int(m[row,1])]=point;origins.append(-1)
    for v in range(len(images)):
        if v in poses:continue
        candidates=[]
        for old in poses:
            for ko,kn in pairs(old,v):
                if (old,int(ko)) in observations:candidates.append((kn,observations[old,int(ko)]))
        links=collect_links(candidates);debug[f'links_{v}']=links
        fit=pnp(np.asarray(xyz)[links[:,1]],features[v][0][links[:,0]],ks[v],cfg)
        for key,value in fit.items():
            if isinstance(value,np.ndarray):debug[f'pnp_{v}_{key}']=value
        record={'view_index':v,'operation':'pnp','candidate_links':len(candidates),'unique_links':len(links),'inliers':int(fit['inliers'].sum()),'registered':fit['registered'],'reason':fit['reason'],'points_before':len(xyz),'new_points':0}
        steps.append(record)
        if not fit['registered']:continue
        previous=list(poses);poses[v]=(fit['R'],fit['t'])
        for key,point in links[fit['inliers']]:observations[v,int(key)]=int(point)
        for old in previous:
            rows=np.array([(ko,kn) for ko,kn in pairs(old,v) if (old,int(ko)) not in observations and (v,int(kn)) not in observations],dtype=np.int64).reshape(-1,2)
            if not len(rows):continue
            ro,to=poses[old];rn,tn=poses[v];rr=rn@ro.T;tt=tn-rr@to
            tri=triangulate_dlt(features[old][0][rows[:,0]],features[v][0][rows[:,1]],ks[old],ks[v],rr,tt)
            keep=tri['positive']&(tri['parallax_deg']>=cfg['min_parallax_deg'])&(tri['reprojection_px'].max(1)<=cfg['max_reprojection_px'])
            debug[f'tri_{old}_{v}_matches']=rows
            for key,value in tri.items():debug[f'tri_{old}_{v}_{key}']=value
            debug[f'tri_{old}_{v}_keep']=keep
            for row,local in zip(rows[keep],tri['xyz'][keep]):
                point=len(xyz);xyz.append((local-to)@ro);origins.append(v)
                observations[old,int(row[0])]=point;observations[v,int(row[1])]=point;record['new_points']+=1
    debug.update({f'keypoints_{i}':f[0] for i,f in enumerate(features)})
    return poses,np.array(xyz),observations,steps,debug,origins
