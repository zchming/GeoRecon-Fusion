"""Audit frozen existing matching caches; no inference or geometric computation."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from georecon.evaluation.support_diagnostic import (
    direct_candidates, link_accounting, opportunity_gates, potential_anchors,
    seed_observations, topology,
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(root, paths):
    text = ''.join(p.relative_to(root).as_posix()+'\0'+sha(p)+'\n' for p in sorted(paths))
    return dict(files=len(paths), sha256=hashlib.sha256(text.encode()).hexdigest())


def used_cache_paths(bundle):
    """Only fixed case inputs were hashed by Stage7C; unused pool members are not runs."""
    paths = set()
    for case in bundle['cases']:
        paths.update(bundle['feature_paths'][key] for key in case['keys'])
        paths.update(bundle['pair_paths'][arm+':'+a+':'+b]
                     for a, b in combinations(case['keys'], 2) for arm in ('B', 'C'))
    return paths


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arg in ('config', 'data', 'study', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    config = read(args.config)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    budget = config['budgets']
    resource.setrlimit(resource.RLIMIT_AS, (budget['rss_gib']*1024**3, budget['rss_gib']*1024**3))
    forbidden = ('cv2', 'torch', 'scipy', 'pycolmap', 'lightglue')
    accesses = set()

    def guard(event, values):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise PermissionError('network forbidden')
        if event == 'import' and values[0].split('.')[0] in forbidden:
            raise PermissionError('model/geometric library import forbidden')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            name = os.fsdecode(values[0])
            if any(x in name.lower() for x in ('truth.json', 'groundtruth', '/raw/', '/weights/')) or name.lower().endswith(('.png', '.jpg', '.jpeg', '.pth', '.pt')):
                raise PermissionError('images, GT and weights forbidden')
            accesses.add(name)
    sys.addaudithook(guard)

    def tick():
        if time.monotonic()-start > budget['wall_seconds']:
            raise TimeoutError('frozen wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > budget['rss_gib']*1024**3:
            raise MemoryError('frozen RSS budget')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > budget['new_output_mib']*1024**2:
            raise RuntimeError('frozen output budget')

    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        tick()

    def table(name, rows):
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (out/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
        tick()

    records, attempts, unavailable, graphs, pairs, loops = [], [], [], [], [], []
    try:
        assert args.data.resolve() == (root/'outputs/stage7c_data_v1').resolve(), 'data scope changed'
        assert args.study.resolve() == (root/'outputs/stage7c_confirmation_v1').resolve(), 'study scope changed'
        for n, h in {**config['anchors_sha256'], **config['algorithm_locks'],
                     config['protocol_document']: config['protocol_document_sha256']}.items():
            assert sha(root/n) == h, 'frozen file changed: '+n
        record_paths = sorted(args.study.glob('predictions/*/*/*/*/record.json'))
        map_paths = sorted(args.study.glob('predictions/*/*/*/*/map.npz'))
        assert digest(root, [p.resolve() for p in record_paths]) == config['record_set'], 'record set changed'
        assert digest(root, [p.resolve() for p in map_paths]) == config['map_set'], 'map set changed'
        bundle = read(args.data/'inference.json')
        old_hashes = read(args.study/'input_hashes.json')
        cache_paths = used_cache_paths(bundle)
        for path in sorted(cache_paths):
            assert path in old_hashes and sha(path) == old_hashes[path], 'input changed: '+path
            tick()
        write('freeze.json', dict(config_sha256=sha(args.config), diagnostic_source_sha256=sha(Path(__file__)),
                                  helper_source_sha256=sha(root/'src/georecon/evaluation/support_diagnostic.py'),
                                  verified_cache_files=len(cache_paths), record_set=config['record_set'], map_set=config['map_set']))
        data_by_case = {}
        for case in bundle['cases']:
            keys = case['keys']
            features = []
            for key in keys:
                with np.load(bundle['feature_paths'][key], allow_pickle=False) as data:
                    xy = data['keypoints']
                assert xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all(), 'invalid keypoints'
                features.append(xy)
            for arm in ('B', 'C'):
                matches = {}
                for i, j in combinations(range(len(keys)), 2):
                    with np.load(bundle['pair_paths'][arm+':'+keys[i]+':'+keys[j]], allow_pickle=False) as data:
                        m = data['indices']
                    assert m.ndim == 2 and m.shape[1] == 2 and np.issubdtype(m.dtype, np.integer)
                    assert not len(m) or (m.min() >= 0 and m[:, 0].max() < len(features[i]) and m[:, 1].max() < len(features[j]))
                    assert np.array_equal(m, np.array(sorted(map(tuple, m.tolist())), dtype=m.dtype).reshape(-1, 2)), 'noncanonical order'
                    row = dict(case=case['id'], arm=arm, first=i, second=j, matches=len(m),
                               duplicate_edges=len(m)-len(set(map(tuple, m.tolist()))))
                    for side, v in enumerate((i, j)):
                        ids = m[:, side]
                        info = bundle['images'][keys[v]]
                        cells = np.floor(features[v][ids]/[info['width'], info['height']]*[8, 6]).astype(int)
                        valid = ((cells >= 0) & (cells < [8, 6])).all(1)
                        degrees = defaultdict(set)
                        for edge in m:
                            degrees[int(edge[side])].add(int(edge[1-side]))
                        row.update({f'unique_{side}': len(degrees), f'one_to_many_{side}': sum(len(n) > 1 for n in degrees.values()),
                                    f'coverage_{side}':len(set(map(tuple, cells[valid].tolist())))/48,
                                    f'outside_image_{side}':int((~valid).sum())})
                    pairs.append(row)
                    matches[i, j] = m.tolist()
                summary, triangles, components, lookup, conflicts = topology(matches, tick)
                graphs.append(dict(case=case['id'], family=case['family'], scene=case['scene'], arm=arm, **summary))
                loops.extend(dict(case=case['id'], arm=arm, **r) for r in triangles)
                data_by_case[case['id'], arm] = matches, components, lookup, conflicts
                tick()
        for path in record_paths:
            r = read(path)
            case = r['case']
            name = path.relative_to(args.study).as_posix()
            common = dict(record=name, case=case['id'], family=case['family'], scene=case['scene'],
                          condition=case['condition'], arm=r['arm'], ranking=r['ranking'], method=r['method'], seed=r['seed'])
            seed = r.get('selection', {}).get('selected_seed')
            pnps = [s for s in r.get('steps', []) if s['operation'] == 'pnp']
            records.append(dict(**common, status=r['status'], selected_seed=json.dumps(seed), seeded=seed is not None,
                                pnp_attempts=len(pnps), registered=len(r.get('poses', {})),
                                BA=r.get('BA', {}).get('reason'), seed_attempts=json.dumps(r.get('selection', {}).get('attempts', [])),
                                signals=json.dumps(r.get('signals', []))))
            if seed is None:
                unavailable.append(dict(**common, evaluability='not_applicable', reason='no_seeded_map'))
                continue
            matches, components, lookup, conflicts = data_by_case[case['id'], r['arm']]
            ss = r['steps'][0]
            nseed = ss['stats']['kept_points']
            try:
                with np.load(path.parent/'map.npz', allow_pickle=False) as data:
                    obs = seed_observations(data['observations'], seed, nseed)
                obs_error = None
            except (ValueError, KeyError) as e:
                obs_error = str(e)
            prior_success = False
            for step_no, s in enumerate(pnps):
                a = dict(**common, attempt=step_no, view=s['view_index'], candidate_links=s['candidate_links'],
                         unique_links=s['unique_links'], inliers=s['inliers'], registered=s['registered'],
                         reason=s['reason'], points_before=s['points_before'], new_points=s['new_points'])
                if prior_success:
                    a.update(evaluability='unavailable', detail='intermediate_state_unavailable')
                elif obs_error:
                    a.update(evaluability='unavailable', detail='seed_identity_unavailable: '+obs_error)
                else:
                    counts, kept = link_accounting(direct_candidates(matches, obs, seed, s['view_index']))
                    if counts['candidate_links'] != s['candidate_links'] or counts['unique_links'] != s['unique_links'] or nseed != s['points_before']:
                        a.update(evaluability='inconsistent', detail='candidate_unique_or_point_count_mismatch')
                    else:
                        potential = potential_anchors(components, lookup, conflicts, obs, s['view_index'], kept)
                        a.update(evaluability='exact', detail='', **{k:v for k,v in counts.items() if k not in a}, **potential,
                                 conflict_blocks_threshold=counts['distinct_pairs'] >= 12 > counts['unique_links'],
                                 indirect_support_gap=counts['unique_links'] < 12 <= potential['topological_anchors'] and potential['indirect_missing_pairs'] > 0)
                attempts.append(a)
                if a['evaluability'] != 'exact':
                    unavailable.append(a.copy())
                prior_success |= s['registered']
            tick()
        assert len(records) == 960 and len(graphs) == 48
        table('graphs.csv', graphs)
        table('pairs.csv', pairs)
        table('loops.csv', [{**r, 'views':json.dumps(r['views'])} for r in loops])
        table('records.csv', records)
        table('attempts.csv', attempts)
        table('unavailable.csv', unavailable)
        # Seal unsupervised diagnostics BEFORE any existing outcome labels are read.
        names = ['graphs.csv', 'pairs.csv', 'loops.csv', 'records.csv', 'attempts.csv', 'unavailable.csv']
        write('features_seal.json', {name:sha(out/name) for name in names})
        gates = opportunity_gates(records, attempts)
        outcomes = read(args.study/'evaluation.json')['rows']
        index = {(r['case'], r['arm'], r['ranking'], r['method'], r['seed']): r for r in outcomes}
        joined = []
        for r in records:
            e = index[r['case'], r['arm'], r['ranking'], r['method'], r['seed']]
            joined.append({k:r[k] for k in common} | {k:e[k] for k in ('auc10', 'accurate_scene', 'correct_init', 'false_pairs', 'severe_pairs')})
        table('existing_outcomes.csv', joined)
        compare = []
        ri = {(r['case'],r['arm'],r['ranking'],r['method'],r['seed']):r for r in records}
        for case in bundle['cases']:
            for ranking, method, seed in product_values():
                b, c = [ri[case['id'],arm,ranking,method,seed] for arm in ('B','C')]
                compare.append(dict(case=case['id'], ranking=ranking, method=method, seed=seed,
                                    B_seed=b['selected_seed'], C_seed=c['selected_seed'],
                                    stratum='both_no_seed' if not b['seeded'] and not c['seeded'] else
                                    'same_seed' if b['seeded'] and b['selected_seed']==c['selected_seed'] else
                                    'different_seed' if b['seeded'] and c['seeded'] else 'one_no_seed'))
        table('frontend_pairing.csv', compare)
        shared = []
        for key, r in ri.items():
            if r['method'] == 'legacy':
                other=ri[key[:3]+('ap3p', key[4])]
                shared.append(r['selected_seed']==other['selected_seed'])
        assert len(shared) == 480 and all(shared), 'solver shared seed mismatch'
        write('decision.json', dict(status='PRIORITY_GATE_MET' if any(x['passed'] for r in gates.values() for x in r.values()) else 'STOP_TRACK_REPAIR_ROUTE',
                                    gates=gates, records=len(records), graphs=len(graphs), attempts=len(attempts),
                                    exact=sum(a['evaluability']=='exact' for a in attempts),
                                    unavailable=dict(Counter(a['detail'] for a in unavailable if 'detail' in a)),
                                    no_seed=sum(not r['seeded'] for r in records), solver_shared_seed_checks=len(shared),
                                    frontend_pairing=dict(Counter(r['stratum'] for r in compare)),
                                    no_new_inference_geometry_network=True, geometry_accuracy_gain_not_tested=True))
        write('session.json', dict(complete=True, seconds=time.monotonic()-start,
                                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                   gpu_bytes=0, network_bytes=0, workers=1,
                                   output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                                   forbidden_imports_loaded=[n for n in forbidden if n in sys.modules],
                                   reads_count=len(accesses)))
        print((out/'decision.json').read_text())
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error), records_completed=len(records),
                                                       graphs_completed=len(graphs), attempts_completed=len(attempts)), indent=2)+'\n')
        raise


def product_values():
    from itertools import product
    return product(('inliers', 'quality'), ('legacy', 'ap3p'), range(42, 47))


if __name__ == '__main__':
    main()
