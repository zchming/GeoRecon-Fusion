"""Experimental bounded PnP local optimization; no reference-pose inputs.

Uses OpenCV LM on normalized observations to retain full/skewed K semantics.
Original RANSAC acceptance is required, and every accepted local step must
preserve positive-depth pixel support and not increase truncated pixel cost.
This is a standard local-optimization component, not a novelty claim.
"""
import numpy as np
from .incremental import pnp, project
from .two_view import normalize, known_K


def residual_state(x, uv, k, r, t, limit):
    pixels, depths = project(x, k, r, t)
    errors = np.linalg.norm(pixels-uv, axis=1)
    valid = np.isfinite(errors) & (depths > 1e-9)
    mask = valid & (errors <= limit)
    cost = float(np.where(valid, np.minimum(errors**2, limit**2), limit**2).sum())
    return mask, errors, depths, cost


def ap3p_pnp(x, uv, k, cfg):
    """Same gates/budget as legacy EPnP; change only the minimal RANSAC solver."""
    import cv2
    x, uv, k = np.asarray(x, float), np.asarray(uv, float), known_K(k)
    if x.shape != (len(uv), 3) or uv.shape != (len(x), 2) or not np.isfinite(x).all():
        raise ValueError('finite Nx3/Nx2 required')
    normalized = normalize(uv, k)
    out = {'registered': False, 'reason': 'insufficient_2d3d', 'R': None, 't': None,
           'inliers': np.zeros(len(x), bool), 'candidate_count': len(x)}
    if len(x) < cfg['pnp_min_inliers']:
        return out
    if np.linalg.matrix_rank(x-x.mean(0), tol=1e-8) < 2:
        out['reason'] = 'collinear_3d'
        return out
    cv2.setRNGSeed(cfg['seed'])
    ok, rv, tv, ids = cv2.solvePnPRansac(
        np.ascontiguousarray(x), np.ascontiguousarray(normalized), np.eye(3), None,
        iterationsCount=cfg['pnp_iterations'],
        reprojectionError=cfg['pnp_ransac_px']/np.mean([k[0, 0], k[1, 1]]),
        confidence=.999, flags=cv2.SOLVEPNP_AP3P)
    if not ok or ids is None:
        out['reason'] = 'pnp_ransac_failed'
        return out
    r, t = cv2.Rodrigues(rv)[0], tv.ravel()
    mask, errors, depths, _ = residual_state(x, uv, k, r, t, cfg['max_reprojection_px'])
    ransac_mask = np.zeros(len(x), bool)
    ransac_mask[ids.ravel()] = True
    mask &= ransac_mask
    out.update(R=r, t=t, inliers=mask, reprojection_px=errors, depths=depths)
    if mask.sum() < cfg['pnp_min_inliers']:
        out['reason'] = 'insufficient_positive_reprojection_inliers'
        return out
    out.update(registered=True, reason=None)
    return out


def refined_pnp(x, uv, k, cfg, *, base_solver=pnp):
    import cv2
    base = base_solver(x, uv, k, cfg)
    audit = {'method': 'bounded_LM', 'iterations': [], 'changed': False}
    base['local_optimization'] = audit
    if not base['registered']:
        audit['stop'] = 'original_registration_rejected'
        return base
    x, uv, k = np.asarray(x, float), np.asarray(uv, float), known_K(k)
    normalized = normalize(uv, k)
    r, t = base['R'].copy(), base['t'].copy()
    mask, errors, depths, cost = residual_state(x, uv, k, r, t, cfg['max_reprojection_px'])
    # Reclassify all available links at the SAME original pixel threshold.
    for iteration in range(3):
        before_mask = mask.copy()
        rv, tv = cv2.solvePnPRefineLM(
            np.ascontiguousarray(x[mask]), np.ascontiguousarray(normalized[mask]),
            np.eye(3), None, cv2.Rodrigues(r)[0], t.reshape(3, 1).copy(),
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-8))
        candidate_r, candidate_t = cv2.Rodrigues(rv)[0], tv.ravel()
        next_mask, next_errors, next_depths, next_cost = residual_state(
            x, uv, k, candidate_r, candidate_t, cfg['max_reprojection_px'])
        accepted = bool(np.isfinite(candidate_r).all() and np.isfinite(candidate_t).all()
                        and next_mask.sum() >= mask.sum() and next_cost <= cost + 1e-10)
        audit['iterations'].append({'round': iteration, 'support_before': int(mask.sum()),
                                    'support_after': int(next_mask.sum()), 'cost_before': cost,
                                    'cost_after': next_cost, 'accepted': accepted})
        if not accepted:
            audit['stop'] = 'support_or_robust_cost_regression'
            break
        audit['changed'] = True
        r, t, mask, errors, depths, cost = candidate_r, candidate_t, next_mask, next_errors, next_depths, next_cost
        if np.array_equal(mask, before_mask):
            audit['stop'] = 'stable_support'
            break
    else:
        audit['stop'] = 'three_round_budget'
    base.update(R=r, t=t, inliers=mask, reprojection_px=errors, depths=depths)
    return base
