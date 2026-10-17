"""Read cached topology only; refuse model/solver imports, GT files and network."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from georecon.matching.holdout_feasibility import analyze_pair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise PermissionError('no network in cached feasibility check')
    if event == 'import' and any(args[0] == name or args[0].startswith(name+'.') for name in
                                ['torch', 'cv2', 'scipy', 'georecon.teaching', 'georecon.geometry']):
        raise PermissionError('no model or geometry libraries in cached feasibility check')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = str(args[0]).lower().replace('\\', '/')
        if any(s in name for s in ['reference.json', 'geometry_', 'ground_truth', '/calibration/', '.pth']):
            raise PermissionError('no GT, geometry arrays or weights in cached feasibility check')


def read(path):
    return json.loads(Path(path).read_text())


def inside(cache, name):
    path = Path(name).resolve()
    if not path.is_relative_to(cache):
        raise ValueError('feature cache escapes declared input directory')
    return path


def run(cache, output, config_path):
    start = time.perf_counter()
    if output.exists():
        raise ValueError('refuse to overwrite feasibility output')
    cfg = read(config_path)
    bundle = read(cache/'bundle.json')
    if len(bundle['cases']) != 46 or len(bundle['feature_paths']) != 24:
        raise ValueError('expected the existing complete 46-pair/24-image Stage7A cache')
    if not read(cache/'session.json')['complete']:
        raise ValueError('incomplete source run')
    output.mkdir(parents=True)
    (output/'case_groups').mkdir()
    hashes = {str(p.relative_to(cache)): digest(p) for p in
              [cache/'bundle.json', cache/'session.json', cache/'evaluation.json']}
    features, records = {}, {}
    for key, filename in bundle['feature_paths'].items():
        path = inside(cache, filename)
        record = read(path.parent/'record.json')
        if record['status'] != 'success' or digest(path) != record['feature_sha256']:
            raise ValueError('feature cache integrity failure')
        hashes[str(path.relative_to(cache))] = record['feature_sha256']
        hashes[str((path.parent/'record.json').relative_to(cache))] = digest(path.parent/'record.json')
        with np.load(path, allow_pickle=False) as f:
            features[key] = {'xy': f['keypoints'], 'size': f['image_size']}
        records[key] = record
    rows = []
    for case in bundle['cases']:
        xy = [features[k]['xy'] for k in case['image_keys']]
        sizes = [features[k]['size'] for k in case['image_keys']]
        matches = []
        for arm in 'BC':
            folder = cache/'main'/case['case_id']/arm
            rec = read(folder/'record.json')
            if rec['status'] != 'success' or digest(folder/'matches.npz') != rec['matches_sha256']:
                raise ValueError('match cache integrity failure')
            if rec['root_input_digests'] != [records[k]['root_input_digest'] for k in case['image_keys']]:
                raise ValueError('B/C source cache identity failure')
            for name in ['record.json', 'matches.npz']:
                hashes[str((folder/name).relative_to(cache))] = digest(folder/name)
            with np.load(folder/'matches.npz', allow_pickle=False) as data:
                ids = data['indices']
                for side, name in enumerate(['pixels_first', 'pixels_second']):
                    np.testing.assert_array_equal(data[name], xy[side][ids[:, side]])
                matches.append(ids)
        row, mapping = analyze_pair(case['case_id'], xy, *matches, sizes, cfg)
        row.update(scan=case['scan'], conditions=case['conditions'])
        rows.append(row)
        write(output/'case_groups'/f"{case['case_id']}.json", mapping)
    # Outcome labels are attached only after every deterministic partition is complete.
    decision = read(cache/'evaluation.json')['decision']
    benefit = set(decision['rescued_pairs'])
    harm = {v['case_id'] for v in decision['new_false_accepts']}
    normal = set(decision['normal_regressions'])
    if len(benefit) != 7 or len(harm) != 6 or not normal <= harm or len(normal) != 2:
        raise ValueError('unexpected retrospective focus labels')
    for row in rows:
        cid = row['case_id']
        row['focus'] = 'benefit' if cid in benefit else 'harm' if cid in harm else 'other'
        row['normal_regression'] = cid in normal
    summary = {}
    for group in ['all', 'benefit', 'harm', 'other']:
        chosen = rows if group == 'all' else [r for r in rows if r['focus'] == group]
        summary[group] = {'pairs': len(chosen),
            'count_feasible_only': sum(not r['reasons'] for r in chosen),
            'not_evaluable': [{'case_id': r['case_id'], 'reasons': r['reasons']} for r in chosen if r['reasons']],
            'levels': {str(n): sum(r['validation_count_levels'][str(n)] for r in chosen) for n in cfg['validation_count_levels']},
            'representative_grid_disjoint': sum(r['representative_grid_disjoint'] for r in chosen),
            'median_validation_min_per_source': float(np.median([r['validation_min_per_source'] for r in chosen])),
            'median_exclusive_quota': float(np.median([r['equal_exclusive_quota_capacity'] for r in chosen]))}
    source = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), source/'src/georecon/matching/holdout_feasibility.py',
               source/'tests/test_stage7a_holdout_feasibility.py', config_path]
    provenance = {'scope': cfg['scope'], 'config': cfg, 'source_sha256': {str(p.relative_to(source)): digest(p) for p in sources},
        'cache_input_sha256': hashes, 'source_cache_unchanged': all(digest(cache/name) == sha for name, sha in hashes.items()),
        'new_inference_calls': 0, 'new_geometry_calls': 0, 'gpu_compute': False,
        'restricted_audit_guard_enabled': True, 'wall_seconds': time.perf_counter()-start,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write(output/'analysis.json', {'summary': summary, 'rows': rows, 'provenance': provenance})
    flat = []
    for r in rows:
        row = {k: r[k] for k in ['case_id', 'scan', 'focus', 'normal_regression', 'structural_status']}
        row['reasons'] = '|'.join(r['reasons'])
        for prefix in ['raw', 'exact_XY', 'physical', 'fit', 'validation']:
            row.update({f'{prefix}_{k}': v for k, v in r[prefix].items()})
        for key in ['components', 'largest_component_edges', 'validation_min_per_source',
                    'equal_exclusive_quota_capacity', 'equal_three_strata_quota_capacity']:
            row[key] = r[key]
        for key in ['raw_conflicting_endpoints_per_image', 'physical_conflicting_endpoints_per_image',
                    'all_feature_duplicate_excess_exact_per_image', 'all_feature_duplicate_excess_near_per_image',
                    'matched_alias_excess_B_C', 'fit_validation_shared_cells_per_image']:
            row[key] = '|'.join(map(str, r[key]))
        for name in ['validation_union', 'validation_B', 'validation_C']:
            for side in [0, 1]:
                row[f'{name}_occupied_cells_{side}'] = r['spatial'][name][side]['occupied_cells']
                row[f'{name}_largest_cell_fraction_{side}'] = r['spatial'][name][side]['largest_cell_fraction']
        flat.append(row)
    with (output/'pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({'summary': summary, 'wall_seconds': provenance['wall_seconds'],
                      'peak_rss_bytes': provenance['peak_rss_bytes']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    sys.addaudithook(guard)
    run(Path(args.cache).resolve(), Path(args.output).resolve(), Path(args.config).resolve())


if __name__ == '__main__':
    main()
