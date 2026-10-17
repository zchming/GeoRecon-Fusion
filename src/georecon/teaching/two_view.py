"""SIFT -> mutual ratios -> E RANSAC -> four poses -> explicit DLT and checks."""
import numpy as np


def known_K(value):
    k=np.asarray(value,dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]) or abs(np.linalg.det(k))<1e-12 or abs(k[1,0])>1e-12:
        raise ValueError('known finite nonsingular pinhole K with positive focal lengths required')
    return k


def normalize(pixels,k):
    pixels=np.asarray(pixels,dtype=float)
    if pixels.ndim!=2 or pixels.shape[1]!=2 or not np.isfinite(pixels).all():raise ValueError('finite Nx2 pixel coordinates required')
    q=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(known_K(k)).T
    return q[:,:2]/q[:,2:3]


def features_and_matches(images,cfg):
    import cv2
    if len(images)!=2:raise ValueError('exactly two images required')
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(cfg['opencv_threads']);cv2.setRNGSeed(cfg['seed'])
    detector=cv2.SIFT_create(nfeatures=cfg['sift_features']);features=[]
    for image in images:
        kp,des=detector.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
        xy=np.array([v.pt for v in kp],dtype=float).reshape(-1,2)
        features.append((xy,des))
    bf=cv2.BFMatcher(cv2.NORM_L2);pairs=[];forward={};backward={}
    if all(v[1] is not None and len(v[1])>=2 for v in features):
        def ratio(a,b):
            return {v[0].queryIdx:(v[0].trainIdx,float(v[0].distance),float(v[1].distance)) for v in bf.knnMatch(a,b,k=2) if len(v)==2 and v[0].distance<cfg['ratio_threshold']*v[1].distance}
        forward=ratio(features[0][1],features[1][1]);backward=ratio(features[1][1],features[0][1])
        pairs=sorted((i,v[0]) for i,v in forward.items() if v[0] in backward and backward[v[0]][0]==i)
    indices=np.array(pairs,dtype=np.int64).reshape(-1,2)
    p=features[0][0][indices[:,0]];q=features[1][0][indices[:,1]]
    arrays={'keypoints_first':features[0][0],'keypoints_second':features[1][0],'match_indices':indices,'pixels_first':p,'pixels_second':q,
        'nearest_distances':np.array([[forward[i][1],forward[i][2]] for i,j in pairs],dtype=float).reshape(-1,2)}
    stats={'keypoints':[len(v[0]) for v in features],'ratio_forward':len(forward),'ratio_backward':len(backward),'mutual_matches':len(pairs),'sift_cap_is_target_not_strict_maximum':True}
    return arrays,stats


def triangulate_dlt(p,q,k1,k2,r,t):
    """One homogeneous 4x4 DLT system per measured pair; no image synthesis."""
    p=np.asarray(p,float);q=np.asarray(q,float);r=np.asarray(r,float);t=np.asarray(t,float).reshape(3)
    if p.ndim!=2 or p.shape[1]!=2 or p.shape!=q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('equal finite Nx2 matches required')
    a=np.c_[np.eye(3),np.zeros(3)];b=np.c_[r,t];u=normalize(p,k1);v=normalize(q,k2)
    design=np.stack([u[:,0,None]*a[2]-a[0],u[:,1,None]*a[2]-a[1],v[:,0,None]*b[2]-b[0],v[:,1,None]*b[2]-b[1]],axis=1)
    _,_,vh=np.linalg.svd(design);hom=vh[:,-1];finite=np.isfinite(hom).all(1)&(np.abs(hom[:,3])>1e-12)
    xyz=np.full((len(p),3),np.nan);xyz[finite]=hom[finite,:3]/hom[finite,3,None]
    c2=xyz@r.T+t;z1=xyz[:,2];z2=c2[:,2];center2=-r.T@t
    ray1=xyz;ray2=xyz-center2
    den=np.linalg.norm(ray1,axis=1)*np.linalg.norm(ray2,axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        angle=np.degrees(np.arccos(np.clip(np.sum(ray1*ray2,axis=1)/den,-1,1)))
        first=xyz@known_K(k1).T;first=first[:,:2]/first[:,2:3]
        second=c2@known_K(k2).T;second=second[:,:2]/second[:,2:3]
    errors=np.c_[np.linalg.norm(first-p,axis=1),np.linalg.norm(second-q,axis=1)]
    positive=finite&np.isfinite(errors).all(1)&np.isfinite(angle)&(z1>1e-9)&(z2>1e-9)&(np.linalg.norm(t)>1e-12)
    return {'xyz':xyz,'finite':finite,'positive':positive,'depths':np.c_[z1,z2],'parallax_deg':angle,'reprojection_px':errors}


def reconstruct(p,q,k1,k2,cfg):
    import cv2
    k1,k2=known_K(k1),known_K(k2);p=np.asarray(p,float);q=np.asarray(q,float)
    # Validate before early exits so invalid input cannot look like ordinary matching failure.
    n1,n2=normalize(p,k1),normalize(q,k2)
    if p.shape!=q.shape:raise ValueError('correspondence shapes differ')
    stats={'matches':len(p),'essential_inliers':0,'homography_inliers':0,'kept_points':0,'hypotheses':[]}
    arrays={'pixels_first':p,'pixels_second':q,'essential_mask':np.zeros(len(p),bool),'homography_mask':np.zeros(len(p),bool),'kept_match_indices':np.array([],dtype=np.int64)}
    def fail(reason):return {'status':'no_reconstruction','reason':reason,'stats':stats,'arrays':arrays,'R':None,'t':None}
    if len(p)<cfg['min_matches']:return fail('insufficient_matches')
    if np.max(np.linalg.norm(p-q,axis=1))<1e-9 and np.allclose(k1,k2):return fail('identical_correspondences_no_baseline')
    cv2.setRNGSeed(cfg['seed']);cv2.setNumThreads(cfg['opencv_threads'])
    threshold=cfg['essential_ransac_px']/np.mean([k1[0,0],k1[1,1],k2[0,0],k2[1,1]])
    e,mask=cv2.findEssentialMat(n1,n2,np.eye(3),method=cv2.RANSAC,prob=cfg['essential_ransac_probability'],threshold=threshold,maxIters=cfg['essential_ransac_iterations'])
    cv2.setRNGSeed(cfg['seed'])
    h,hm=cv2.findHomography(p,q,cv2.RANSAC,cfg['essential_ransac_px'],maxIters=cfg['essential_ransac_iterations'],confidence=cfg['essential_ransac_probability'])
    if h is not None and np.isfinite(h).all():arrays['H']=h;arrays['homography_mask']=hm.ravel().astype(bool);stats['homography_inliers']=int(arrays['homography_mask'].sum())
    stats['normalized_ransac_threshold']=float(threshold)
    if e is None or mask is None or e.ndim!=2 or e.shape[1]!=3 or e.shape[0]%3 or not np.isfinite(e).all():return fail('essential_estimation_failed')
    arrays['essential_candidates']=e;arrays['essential_mask']=mask.ravel().astype(bool);stats['essential_inliers']=int(arrays['essential_mask'].sum())
    if stats['essential_inliers']<cfg['min_matches']:return fail('insufficient_essential_inliers')
    if stats['homography_inliers']/len(p)>=cfg['homography_dominance_fraction'] and stats['homography_inliers']>=cfg['homography_to_essential_inlier_ratio']*stats['essential_inliers']:
        return fail('homography_dominated_ambiguous_geometry')
    ids=np.flatnonzero(arrays['essential_mask']);best=None
    for eidx in range(e.shape[0]//3):
        matrix=e[eidx*3:eidx*3+3];r1,r2,trans=cv2.decomposeEssentialMat(matrix)
        for ridx,r in enumerate([r1,r2]):
            for sign in [1,-1]:
                t=sign*trans.ravel()/np.linalg.norm(trans);tri=triangulate_dlt(p[ids],q[ids],k1,k2,r,t)
                count=int(tri['positive'].sum());median=float(np.median(tri['reprojection_px'][tri['positive']].max(1))) if count else None
                record={'essential_index':eidx,'rotation_index':ridx,'translation_sign':sign,'dual_positive_points':count,'median_max_view_reprojection_px':median}
                stats['hypotheses'].append(record);score=(count,-median if median is not None else -np.inf)
                if best is None or score>best[0]:best=(score,r,t,tri,matrix,record)
    _,r,t,tri,matrix,record=best;arrays.update({'candidate_'+key:value for key,value in tri.items()});arrays.update(candidate_R=r,candidate_t=t,E=matrix,essential_match_indices=ids)
    stats['selected_hypothesis']=record;stats['positive_fraction']=float(tri['positive'].mean())
    if stats['positive_fraction']<cfg['min_positive_depth_fraction']:return fail('insufficient_dual_positive_support')
    keep=tri['positive']&(tri['parallax_deg']>=cfg['min_parallax_deg'])&(tri['reprojection_px'].max(1)<=cfg['max_reprojection_px'])
    arrays['triangulation_keep_mask']=keep;arrays['kept_match_indices']=ids[keep]
    stats['kept_points']=int(keep.sum());stats['positive_median_parallax_deg']=float(np.median(tri['parallax_deg'][tri['positive']]))
    if stats['kept_points']<cfg['min_kept_points']:return fail('insufficient_parallax_or_reprojection_support')
    arrays['accepted_xyz']=tri['xyz'][keep];stats['accepted_median_parallax_deg']=float(np.median(tri['parallax_deg'][keep]));stats['accepted_max_reprojection_px']=float(tri['reprojection_px'][keep].max())
    return {'status':'success','reason':None,'R':r,'t':t,'stats':stats,'arrays':arrays}
