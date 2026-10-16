"""Fresh image-only SIFT matches; never reuse a baseline's optimized tracks."""
from itertools import combinations
import numpy as np
from .geometry import sampson_px,distribution,components


def coverage(xy,wh,grid_wh):
    if not len(xy):return 0
    grid=np.floor(np.asarray(xy)/np.array(wh)*np.array(grid_wh)).astype(int)
    grid=np.clip(grid,[0,0],np.array(grid_wh)-1)
    return len(np.unique(grid,axis=0))


def match_images(images,config):
    import cv2
    cv2.setNumThreads(config['opencv_threads']);cv2.setRNGSeed(config['seed'])
    detector=cv2.SIFT_create(nfeatures=config['sift_features'])
    features=[]
    for image in images:
        kp,des=detector.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
        features.append((np.array([k.pt for k in kp],float).reshape(-1,2),des))
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    pairs=[];edges=[];arrays={}
    for i,j in combinations(range(len(images)),2):
        p,dp=features[i];q,dq=features[j];matches=[]
        if dp is not None and dq is not None and min(len(dp),len(dq))>=2:
            def good(a,b):
                return {m[0].queryIdx:m[0].trainIdx for m in matcher.knnMatch(a,b,k=2) if len(m)==2 and m[0].distance<config['ratio_threshold']*m[1].distance}
            forward,backward=good(dp,dq),good(dq,dp)
            matches=sorted((a,b) for a,b in forward.items() if backward.get(b)==a)
        # Alternate match IDs: fit image-only F on one half, audit on the other.
        # This does NOT claim that SIFT locations cannot overlap a baseline's training observations.
        a=np.array([p[x] for x,y in matches],float).reshape(-1,2)
        b=np.array([q[y] for x,y in matches],float).reshape(-1,2)
        fit_a,fit_b=a[::2],b[::2];audit_a,audit_b=a[1::2],b[1::2]
        if len(audit_a)>config['max_audit_matches']:
            ix=np.linspace(0,len(audit_a)-1,config['max_audit_matches'],dtype=int);audit_a,audit_b=audit_a[ix],audit_b[ix]
        F=None
        if len(fit_a)>=8:
            cv2.setRNGSeed(config['seed']+i*100+j)
            F,_=cv2.findFundamentalMat(fit_a,fit_b,cv2.FM_RANSAC,config['image_f_ransac_px'],config['image_f_ransac_confidence'],config['image_f_ransac_iterations'])
            if F is not None and (F.shape!=(3,3) or not np.isfinite(F).all() or np.linalg.norm(F)<1e-15):F=None
        image_errors=sampson_px(F,audit_a,audit_b) if F is not None else np.array([])
        cells=[coverage(audit_a,[images[i].shape[1],images[i].shape[0]],config['grid_wh']),coverage(audit_b,[images[j].shape[1],images[j].shape[0]],config['grid_wh'])]
        enough=len(audit_a)>=config['min_evidence_matches'] and min(cells)>=config['min_covered_cells']
        if enough:edges.append((i,j))
        prefix=f'{i}_{j}';arrays[prefix+'_first']=audit_a;arrays[prefix+'_second']=audit_b
        pairs.append({'indices':[i,j],'descriptor_matches':len(matches),'fit_matches':len(fit_a),'audit_matches':len(audit_a),'covered_cells':cells,'evidence_status':'available' if enough else 'insufficient',
                      'image_only_f_fit_status':'available' if F is not None else 'unavailable','image_only_f_audit_residual_px':distribution(image_errors),
                      'image_only_f_audit_support_fraction':float(np.mean(image_errors<=config['image_f_ransac_px'])) if len(image_errors) else None,'array_prefix':prefix})
    return {'detector':'OpenCV SIFT + mutual Lowe ratio; independently extracted; no predicted geometry used',
            'independence_limit':'not held-out images; baseline SIFT/tracker observations may spatially overlap; not statistical independence',
            'keypoints_per_view':[len(p) for p,d in features],'pairs':pairs,'evidence_graph':components(len(images),edges)},arrays
