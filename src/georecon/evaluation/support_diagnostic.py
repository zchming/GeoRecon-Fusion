"""Read-only identity/topology accounting. No geometric solver or model imports."""
from collections import Counter, defaultdict
from itertools import combinations, product


def link_accounting(candidates):
    """Duplicates are supporting votes; ambiguity is counted on unique edges."""
    edges = set(map(tuple, candidates))
    by_key, by_point = defaultdict(set), defaultdict(set)
    for k, p in edges:
        by_key[k].add(p)
        by_point[p].add(k)
    key_bad = {e for e in edges if len(by_key[e[0]]) > 1}
    point_bad = {e for e in edges if len(by_point[e[1]]) > 1}
    kept = edges - key_bad - point_bad
    counts = dict(candidate_links=len(candidates), distinct_pairs=len(edges),
                  duplicate_votes=len(candidates)-len(edges),
                  key_conflict_edges=len(key_bad), point_conflict_edges=len(point_bad),
                  both_conflict_edges=len(key_bad & point_bad),
                  conflict_loss=len(key_bad | point_bad), unique_links=len(kept))
    assert counts['candidate_links'] == counts['duplicate_votes'] + counts['conflict_loss'] + len(kept)
    return counts, kept


def topology(matches, tick=lambda: None):
    """Union raw matching edges, preserving all same-image identity conflicts."""
    parent, adjacency = {}, defaultdict(lambda: defaultdict(set))

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), rows in matches.items():
        for ka, kb in rows:
            u, v = (a, int(ka)), (b, int(kb))
            x, y = find(u), find(v)
            if x != y:
                parent[x] = y
            adjacency[u][b].add(v)
            adjacency[v][a].add(u)
        tick()
    groups = defaultdict(set)
    for node in list(parent):
        groups[find(node)].add(node)
    components = list(groups.values())
    node_component = {node: i for i, group in enumerate(components) for node in group}
    conflicts = [len({v for v, k in group}) != len(group) for group in components]
    counts = Counter(len({v for v, k in group}) for group in components)
    triples = []
    # Include declared views with zero matched observations: no wedges is N/A.
    views = sorted({v for pair in matches for v in pair})
    for images in combinations(views, 3):
        wedges = set()
        for node in parent:
            if node[0] not in images:
                continue
            other = [v for v in images if v != node[0]]
            for u, v in product(adjacency[node].get(other[0], ()), adjacency[node].get(other[1], ())):
                wedges.add(tuple(sorted((node, u, v))))
        closed = missing = ambiguous = 0
        for nodes in wedges:
            bad = any(len(adjacency[node].get(v, ())) > 1
                      for node in nodes for v in images if v != node[0])
            if bad:
                ambiguous += 1
            elif all(v in adjacency[u].get(v[0], ()) for u, v in combinations(nodes, 2)):
                closed += 1
            else:
                missing += 1
        triples.append(dict(views=list(images), candidate_triples=len(wedges), closed=closed,
                            missing_edge=missing, ambiguous=ambiguous,
                            decidable=closed+missing,
                            closure_fraction=closed/(closed+missing) if closed+missing else None))
        tick()
    summary = dict(nodes=len(parent), components=len(components), two_view_components=counts[2],
                   multiview_components=sum(n for views, n in counts.items() if views >= 3),
                   multiview_observations=sum(len(g) for g in components if len({v for v, k in g}) >= 3),
                   conflict_components=sum(conflicts),
                   conflict_fraction=sum(conflicts)/len(components) if components else None,
                   conflict_observations=sum(len(g) for g, bad in zip(components, conflicts) if bad))
    return summary, triples, components, node_component, conflicts


def seed_observations(observations, seed, count):
    """Final observations retain seed point IDs; geometry and xyz are not read."""
    found = {}
    for v, k, p in observations:
        v, k, p = int(v), int(k), int(p)
        if v in seed and 0 <= p < count:
            if (v, k) in found:
                raise ValueError('duplicate seed observation identity')
            found[v, k] = p
    expected = {(v, p) for v in seed for p in range(count)}
    actual = [(v, p) for (v, k), p in found.items()]
    if len(actual) != 2*count or set(actual) != expected:
        raise ValueError('seed point prefix is incomplete or not one-to-one')
    return found


def direct_candidates(matches, seed_obs, seed, view):
    candidates = []
    for old in seed:
        rows = matches[min(old, view), max(old, view)]
        for a, b in rows:
            ko, kn = (a, b) if old < view else (b, a)
            if (old, int(ko)) in seed_obs:
                candidates.append((int(kn), seed_obs[old, int(ko)]))
    return candidates


def potential_anchors(components, lookup, conflicts, seed_obs, view, direct_kept):
    anchors = defaultdict(set)
    for node, point in seed_obs.items():
        if node in lookup:
            anchors[lookup[node]].add(point)
    potential = set()
    excluded_conflict = excluded_multi = 0
    for index, points in anchors.items():
        target = [node for node in components[index] if node[0] == view]
        if not target:
            continue
        if conflicts[index]:
            excluded_conflict += 1
        elif len(points) != 1:
            excluded_multi += 1
        else:
            potential.add((target[0][1], next(iter(points))))
    return dict(topological_anchors=len({p for k, p in potential}),
                indirect_missing_pairs=len(potential - direct_kept),
                excluded_conflict_components=excluded_conflict,
                excluded_multi_landmark_components=excluded_multi)


def opportunity_gates(records, attempts):
    """Frozen five-seed, two-condition, two-recording research-priority gate."""
    by_record = defaultdict(list)
    for a in attempts:
        by_record[a['record']].append(a)
    result = {}
    for ranking in ('inliers', 'quality'):
        output = {}
        for cause in ('conflict_blocks_threshold', 'indirect_support_gap'):
            groups = defaultdict(dict)
            for r in records:
                if r['family'] != 'tum_rgbd' or r['arm'] != 'C' or r['method'] != 'legacy' or r['ranking'] != ranking:
                    continue
                aa = by_record[r['record']]
                supported = any(a['evaluability'] == 'exact' and a.get(cause, False) for a in aa)
                groups[r['scene'], r['condition']][r['seed']] = supported
            conditions = {scene+':'+condition: sorted(k for k, v in seeds.items() if v)
                          for (scene, condition), seeds in sorted(groups.items())}
            scene_counts = Counter(scene for (scene, condition), seeds in groups.items()
                                   if set(seeds) == set(range(42, 47)) and all(seeds.values()))
            qualifying = sorted(scene for scene, n in scene_counts.items() if n >= 2)
            output[cause] = dict(passed=len(qualifying) >= 2,
                                 conditions_supported_seeds=conditions,
                                 qualifying_scenes=qualifying)
        result[ranking] = output
    return result
