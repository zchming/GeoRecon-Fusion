"""GT-free preparation and fixed-observation validation of shadow poses."""
from collections import defaultdict
import numpy as np


def prepare_observations(b, c, saved, xy):
    """Keep all fit rows; use one actual union row per grouped validation edge."""
    b, c = [np.asarray(v, dtype=np.int64).reshape(-1, 2) for v in [b, c]]
    mapping = {tuple(map(int, k.split(','))): tuple(v) for k, v in saved['raw_to_physical'].items()}
    union = set(map(tuple, b)) | set(map(tuple, c))
    if set(mapping) != union:
        raise ValueError('saved mapping differs from B/C union')
    partitions = {name: set() for name in ['fit', 'validation']}
    for component in saved['components']:
        edges = set(map(tuple, component['edges']))
        if edges & (partitions['fit'] | partitions['validation']):
            raise ValueError('component edges repeated')
        partitions[component['partition']].update(edges)
    if partitions['fit'] | partitions['validation'] != set(mapping.values()):
        raise ValueError('partition misses grouped edges')
    for side in [0, 1]:
        if {e[side] for e in partitions['fit']} & {e[side] for e in partitions['validation']}:
            raise ValueError('fit/validation share a grouped endpoint')
    canonical = {}
    for pair in sorted(union):
        canonical.setdefault(mapping[pair], pair)
    physical = [{mapping[tuple(pair)] for pair in arm} for arm in [b, c]]
    edges = sorted(partitions['validation'])
    ids = np.array([canonical[e] for e in edges], np.int64).reshape(-1, 2)
    masks = np.array([[e in arm for arm in physical] for e in edges], bool).reshape(-1, 2)
    arrays = {'validation_indices': ids, 'validation_edges': np.array(edges, np.int64).reshape(-1, 2),
              'validation_membership': masks, 'validation_p': xy[0][ids[:, 0]], 'validation_q': xy[1][ids[:, 1]]}
    counts = {}
    for name, arm, groups in zip(['B', 'C'], [b, c], physical):
        fit = np.array(sorted(tuple(pair) for pair in arm if mapping[tuple(pair)] in partitions['fit']), np.int64).reshape(-1, 2)
        arrays[f'fit_{name}_indices'] = fit
        arrays[f'fit_{name}_p'], arrays[f'fit_{name}_q'] = xy[0][fit[:, 0]], xy[1][fit[:, 1]]
        counts[name] = {'raw_fit': len(fit), 'distinct_fit': len(groups & partitions['fit']),
                        'validation': int(masks[:, ['B', 'C'].index(name)].sum())}
    return arrays, counts


def capped_sampson(p, q, k1, k2, r, t, cap):
    t = np.asarray(t).reshape(3)
    if np.linalg.norm(t) <= 1e-12:
        raise ValueError('nonzero translation required')
    x, y, z = t
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    F = np.linalg.inv(k2).T @ cross @ r @ np.linalg.inv(k1)
    F /= np.linalg.norm(F)
    p, q = np.c_[p, np.ones(len(p))], np.c_[q, np.ones(len(q))]
    fp, fq = p @ F.T, q @ F
    denominator = np.sum(fp[:, :2]**2+fq[:, :2]**2, axis=1)
    error = np.full(len(p), np.nan)
    valid = denominator > 1e-24
    error[valid] = np.abs(np.sum(q[valid]*fp[valid], axis=1))/np.sqrt(denominator[valid])
    loss = np.ones(len(p))
    loss[np.isfinite(error)] = np.minimum(error[np.isfinite(error)]/cap, 1)
    return error, loss


def preference(b, c, tolerance):
    if b is None or c is None:
        return 'unknown'
    return 'C_lower' if c < b-tolerance else 'B_lower' if b < c-tolerance else 'tie'


def validation_metrics(arrays, k1, k2, r, t, cfg, backend):
    from ..teaching.two_view import triangulate_dlt
    p, q, membership = [arrays[k] for k in ['validation_p', 'validation_q', 'validation_membership']]
    error, loss = capped_sampson(p, q, k1, k2, r, t, cfg['epipolar_cap_px'])
    masks = {'B': membership[:, 0], 'C': membership[:, 1],
             'common': membership.all(1), 'B_only': membership[:, 0] & ~membership[:, 1],
             'C_only': membership[:, 1] & ~membership[:, 0], 'union': np.ones(len(p), bool)}
    by_source = {key: {'count': int(mask.sum()), 'capped_loss': float(loss[mask].mean()) if mask.any() else None}
                 for key, mask in masks.items()}
    grid_source = []
    for source in ['B', 'C']:
        mask = masks[source]
        views = []
        for pixels in [p, q]:
            cells = np.floor(pixels[mask]/[800, 600]*cfg['grid_xy']).astype(int)
            groups = defaultdict(list)
            for cell, value in zip(cells, loss[mask]):
                groups[tuple(cell)].append(value)
            views.append(float(np.mean([np.mean(v) for v in groups.values()])) if groups else None)
        grid_source.append(float(np.mean(views)) if all(v is not None for v in views) else None)
    primary = float(np.mean([by_source[s]['capped_loss'] for s in ['B', 'C']])) if all(masks[s].any() for s in ['B', 'C']) else None
    tri = triangulate_dlt(p, q, k1, k2, r, t)
    support = tri['positive'] & (tri['parallax_deg'] >= backend['min_parallax_deg']) & (tri['reprojection_px'].max(1) <= backend['max_reprojection_px'])
    result = {'primary_loss': primary, 'strata': by_source,
        'source_cell_balanced_loss': float(np.mean(grid_source)) if all(v is not None for v in grid_source) else None,
        'nonfinite_epipolar_errors': int((~np.isfinite(error)).sum()),
        'dual_positive_fraction': float(tri['positive'].mean()) if len(p) else None,
        'original_triangulation_support_fraction': float(support.mean()) if len(p) else None,
        'true_3D_match_precision': None}
    return result, {'validation_sampson_px': error, 'validation_capped_loss': loss,
                    **{'validation_'+key: value for key, value in tri.items()}, 'validation_support_mask': support}
