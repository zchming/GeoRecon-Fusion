"""Cached correspondence topology only: no camera, model or solver imports."""
import hashlib
import json
from collections import defaultdict
import numpy as np


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_keypoints(xy, radius):
    """Conservative transitive proximity groups, canonicalized by minimum index."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or radius < 0:
        raise ValueError('finite Nx2 keypoints and nonnegative radius required')
    graph, buckets = Components(), defaultdict(list)
    for i, point in enumerate(xy):
        graph.find(i)
        cell = tuple(point) if radius == 0 else tuple(np.floor(point/radius).astype(int))
        neighbors = [cell] if radius == 0 else [(cell[0]+a, cell[1]+b) for a in [-1, 0, 1] for b in [-1, 0, 1]]
        for neighbor in neighbors:
            for j in buckets[neighbor]:
                if np.linalg.norm(point-xy[j]) <= radius:
                    graph.join(i, j)
        buckets[cell].append(i)
    return np.array([graph.find(i) for i in range(len(xy))], dtype=np.int64)


def correspondence_set(pairs, sizes):
    a = np.asarray(pairs)
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in 'iu':
        raise ValueError('integer Nx2 indices required')
    if np.any(a < 0) or any(np.any(a[:, side] >= sizes[side]) for side in [0, 1]):
        raise ValueError('out-of-range match index')
    if any(len(np.unique(a[:, side])) != len(a) for side in [0, 1]):
        raise ValueError('each original matcher must be one-to-one')
    return {tuple(map(int, row)) for row in a}


def physical_edges(edges, groups):
    return {(int(groups[0][i]), int(groups[1][j])) for i, j in edges}


def source_counts(b, c):
    return {'B': len(b), 'C': len(c), 'union': len(b | c),
            'common': len(b & c), 'B_only': len(b-c), 'C_only': len(c-b)}


def conflicts(edges):
    partners = [defaultdict(set), defaultdict(set)]
    for i, j in edges:
        partners[0][i].add(j)
        partners[1][j].add(i)
    return [sum(len(v) > 1 for v in side.values()) for side in partners]


def spatial(edges, xy, sizes, grid):
    """Each physical correspondence gets one vote; representatives are original XY."""
    result = []
    for side in [0, 1]:
        points = np.array([xy[side][e[side]] for e in sorted(edges)]).reshape(-1, 2)
        cells = np.floor(points/np.array(sizes[side])*grid).astype(int)
        if len(cells) and not ((cells >= 0).all() and (cells < grid).all()):
            raise ValueError('pixel outside declared common image')
        keys = [tuple(map(int, p)) for p in cells]
        counts = defaultdict(int)
        for key in keys:
            counts[key] += 1
        result.append({'occupied_cells': len(counts),
            'grid_coverage': len(counts)/int(np.prod(grid)),
            'largest_cell_fraction': max(counts.values())/len(keys) if keys else None,
            'cell_counts': {f'{x},{y}': v for (x, y), v in sorted(counts.items())}})
    return result


def component_partition(edges, case_id, cfg):
    graph = Components()
    for i, j in sorted(edges):
        graph.join((0, i), (1, j))
    components = defaultdict(set)
    for edge in sorted(edges):
        components[graph.find((0, edge[0]))].add(edge)
    fit, validation, records = set(), set(), []
    for group in sorted(components.values(), key=lambda s: min(s)):
        payload = json.dumps([cfg['partition_salt'], case_id, sorted(group)], separators=(',', ':'))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        selected = int(digest, 16)*cfg['holdout_denominator'] < cfg['holdout_numerator']*2**256
        (validation if selected else fit).update(group)
        records.append({'sha256': digest, 'partition': 'validation' if selected else 'fit',
                        'edges': [list(e) for e in sorted(group)]})
    assert not fit & validation and fit | validation == edges
    assert all(not {e[side] for e in fit} & {e[side] for e in validation} for side in [0, 1])
    return fit, validation, records


def analyze_pair(case_id, xy, b_pairs, c_pairs, sizes, cfg):
    sizes_n = [len(v) for v in xy]
    b, c = [correspondence_set(a, sizes_n) for a in [b_pairs, c_pairs]]
    exact = [group_keypoints(v, 0) for v in xy]
    near = [group_keypoints(v, cfg['near_duplicate_radius_px']) for v in xy]
    eb, ec = [physical_edges(s, exact) for s in [b, c]]
    pb, pc = [physical_edges(s, near) for s in [b, c]]
    fit, val, components = component_partition(pb | pc, case_id, cfg)
    counts = source_counts(pb, pc)
    fit_counts, val_counts = source_counts(pb & fit, pc & fit), source_counts(pb & val, pc & val)
    reasons = []
    floor = cfg['minimum_train_distinct_edges']
    for arm in ['B', 'C']:
        if fit_counts[arm] < floor:
            reasons.append(f'{arm}_fit_distinct_edges_below_{floor}')
        if not val_counts[arm]:
            reasons.append(f'{arm}_validation_empty')
    if len(components) < 2:
        reasons.append('fewer_than_two_components')
    distributions = {name: spatial(s, xy, sizes, cfg['grid_xy']) for name, s in
        [('B_all', pb), ('C_all', pc), ('fit_union', fit), ('validation_union', val),
         ('validation_B', pb & val), ('validation_C', pc & val)]}
    overlap_cells = [len(set(distributions['fit_union'][side]['cell_counts']) &
                         set(distributions['validation_union'][side]['cell_counts'])) for side in [0, 1]]
    raw_to_physical = {f'{i},{j}': [int(near[0][i]), int(near[1][j])] for i, j in sorted(b | c)}
    val_min = min(val_counts['B'], val_counts['C'])
    row = {'case_id': case_id, 'raw': source_counts(b, c), 'exact_XY': source_counts(eb, ec),
        'physical': counts, 'raw_conflicting_endpoints_per_image': conflicts(b | c),
        'physical_conflicting_endpoints_per_image': conflicts(pb | pc),
        'all_feature_duplicate_excess_exact_per_image': [len(x)-len(np.unique(g)) for x, g in zip(xy, exact)],
        'all_feature_duplicate_excess_near_per_image': [len(x)-len(np.unique(g)) for x, g in zip(xy, near)],
        'matched_alias_excess_B_C': [len(b)-len(pb), len(c)-len(pc)],
        'components': len(components), 'largest_component_edges': max((len(r['edges']) for r in components), default=0),
        'fit': fit_counts, 'validation': val_counts,
        'validation_min_per_source': val_min,
        'validation_count_levels': {str(n): not reasons and val_min >= n for n in cfg['validation_count_levels']},
        'equal_exclusive_quota_capacity': min(val_counts['B_only'], val_counts['C_only']),
        'equal_three_strata_quota_capacity': min(val_counts[k] for k in ['common', 'B_only', 'C_only']),
        'structural_status': 'count_feasible_only' if not reasons else 'not_evaluable_under_this_partition',
        'reasons': reasons, 'spatial': distributions,
        'fit_validation_shared_cells_per_image': overlap_cells,
        'endpoint_leakage': False, 'representative_grid_disjoint': bool(fit and val and not any(overlap_cells)),
        'spatial_block_independence': False if any(overlap_cells) else None,
        'validation_power_established': False}
    return row, {'components': components, 'raw_to_physical': raw_to_physical,
                 'group_representatives_per_image': [g.tolist() for g in near]}
