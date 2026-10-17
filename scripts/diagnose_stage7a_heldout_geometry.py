"""Fixed-partition shadow geometry and separate offline truth audit; no policy."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from georecon.matching.heldout_geometry import prepare_observations, validation_metrics, preference

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def guard(event, args):
    if event in ['socket.connect', 'socket.getaddrinfo']:
        raise PermissionError('offline diagnostic forbids network')
    if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.')):
        raise PermissionError('no learned inference or CUDA in shadow diagnostic')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).lower().replace('\\', '/')
        if any(s in name for s in ['reference.json', 'evaluation.json', '/stage3_dtu/', '/stage7a_v1_after_guard_fix/', '/stage7a_holdout_feasibility_v1_final/', 'ground_truth', '.pth']):
            raise PermissionError('worker cannot read GT, original full-fit results or model weights')


def prepare(cache, feasibility, output, config):
    from georecon.matching.session import load_protocol
    protocol = load_protocol(ROOT)
    cfg = read(config)
    if output.exists():
        raise ValueError('refuse to overwrite shadow diagnostic output')
    if digest(feasibility/'analysis.json') != cfg['feasibility_analysis_sha256']:
        raise ValueError('feasibility source changed')
    runtime = {'python': sys.version.split()[0], **{k: importlib.metadata.version(k) for k in ['numpy', 'opencv-python']}}
    if any(runtime[k] != protocol['lightglue']['runtime'][k] for k in runtime):
        raise ValueError('geometry runtime differs from Stage7A')
    old = read(cache/'bundle.json')
    if len(old['cases']) != cfg['case_count'] or not read(cache/'session.json')['complete']:
        raise ValueError('incomplete original Stage7A source')
    backend = read(ROOT/protocol['backend']['config_source'])
    output.mkdir(parents=True)
    sources = [Path(__file__).resolve(), ROOT/'src/georecon/matching/heldout_geometry.py',
               ROOT/'tests/test_stage7a_heldout_geometry.py', config,
               ROOT/protocol['backend']['source'], ROOT/protocol['backend']['config_source']]
    source_hashes = {}
    for path in sources:
        name = str(path.relative_to(ROOT))
        target = output/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); source_hashes[name] = digest(path)
    input_hashes = {}
    def lock(path):
        input_hashes[str(path)] = digest(path)
    for path in [cache/'bundle.json', cache/'session.json', cache/'input_hashes.json', feasibility/'analysis.json',
                 ROOT/'configs/experiments/stage7a_protocol_v1.json']:
        lock(path)
    features, feature_records = {}, {}
    for key, filename in old['feature_paths'].items():
        path = Path(filename)
        if not path.resolve().is_relative_to(cache):
            raise ValueError('feature path escapes cache')
        rec = read(path.parent/'record.json')
        if digest(path) != rec['feature_sha256']:
            raise ValueError('feature bytes changed')
        lock(path); lock(path.parent/'record.json')
        with np.load(path, allow_pickle=False) as f:
            features[key] = f['keypoints']
        feature_records[key] = rec
    cases = []
    partition_cfg = read(ROOT/'configs/experiments/stage7a_holdout_feasibility_v1.json')
    lock(ROOT/'configs/experiments/stage7a_holdout_feasibility_v1.json')
    for case in old['cases']:
        cid, keys = case['case_id'], case['image_keys']
        saved_path = feasibility/'case_groups'/f'{cid}.json'; lock(saved_path)
        saved = read(saved_path)
        for group in saved['components']:
            payload = json.dumps([partition_cfg['partition_salt'], cid, sorted(map(tuple, group['edges']))], separators=(',', ':'))
            h = hashlib.sha256(payload.encode()).hexdigest()
            if h != group['sha256'] or group['partition'] != ('validation' if int(h, 16)*4 < 2**256 else 'fit'):
                raise ValueError('saved component assignment differs')
        pairs = []
        for arm in ['B', 'C']:
            folder = cache/'main'/cid/arm; rec = read(folder/'record.json')
            if digest(folder/'matches.npz') != rec['matches_sha256'] or rec['root_input_digests'] != [feature_records[k]['root_input_digest'] for k in keys]:
                raise ValueError('original match contract changed')
            lock(folder/'record.json'); lock(folder/'matches.npz')
            with np.load(folder/'matches.npz', allow_pickle=False) as f:
                ids = f['indices']
                for side, name in enumerate(['pixels_first', 'pixels_second']):
                    np.testing.assert_array_equal(features[keys[side]][ids[:, side]], f[name])
                pairs.append(ids)
        arrays, counts = prepare_observations(*pairs, saved, [features[k] for k in keys])
        reasons = []
        for arm in ['B', 'C']:
            if counts[arm]['distinct_fit'] < partition_cfg['minimum_train_distinct_edges']:
                reasons.append(f'{arm}_fit_distinct_edges_below_20')
            if not counts[arm]['validation']:
                reasons.append(f'{arm}_validation_empty')
        path = output/'inputs'/f'{cid}.npz'; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        cases.append({'case_id': cid, 'scan': case['scan'], 'conditions': case['conditions'],
            'view_ids': case['view_ids'], 'K': [old['images'][k]['K'] for k in keys],
            'input': str(path), 'input_sha256': digest(path), 'counts': counts, 'eligibility_reasons': reasons})
    bundle = {'cases': cases, 'config': cfg, 'backend': backend, 'output': str(output)}
    write(output/'bundle.json', bundle)
    write(output/'provenance.json', {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': source_hashes, 'input_sha256': input_hashes, 'runtime': runtime,
        'authority': 'user approved fixed-partition heldout geometry diagnosis, no replacement policy/Stage7B/commit/push'})
    return bundle


def worker(bundle):
    from georecon.teaching.two_view import reconstruct
    import cv2
    cv2.setNumThreads(8); cv2.ocl.setUseOpenCL(False)
    cfg, out = bundle['config'], Path(bundle['output'])
    fit_calls = validation_calls = 0
    for index, case in enumerate(bundle['cases']):
        with np.load(case['input'], allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        if digest(case['input']) != case['input_sha256']:
            raise ValueError('prepared fit/validation observations changed')
        for arm in ['BC', 'CB'][index % 2]:
            write(out/'progress.json', {'case_id': case['case_id'], 'arm': arm, 'started': time.time(), 'done': False})
            folder = out/'fits'/case['case_id']/arm; folder.mkdir(parents=True, exist_ok=False)
            rows = []
            for seed in cfg['geometry_seeds']:
                row = {'seed': seed, 'status': 'not_evaluable', 'reason': '|'.join(case['eligibility_reasons']),
                       'validation': None, 'geometry_seconds': None, 'validation_seconds': None}
                if not case['eligibility_reasons']:
                    start = time.perf_counter()
                    value = reconstruct(arrays[f'fit_{arm}_p'], arrays[f'fit_{arm}_q'], *case['K'], dict(bundle['backend'], seed=seed))
                    fit_calls += 1
                    row.update(status=value['status'], reason=value['reason'], statistics=value['stats'], geometry_seconds=time.perf_counter()-start)
                    saved = dict(value['arrays'])
                    if value['status'] == 'success':
                        saved['accepted_R'], saved['accepted_t'] = value['R'], value['t']
                        start = time.perf_counter()
                        metrics, extra = validation_metrics(arrays, *[np.array(k) for k in case['K']], value['R'], value['t'], cfg, bundle['backend'])
                        validation_calls += 1; saved.update(extra)
                        row.update(validation=metrics, validation_seconds=time.perf_counter()-start)
                    path = folder/f'seed_{seed}.npz'; np.savez_compressed(path, **saved)
                    row['arrays_sha256'] = digest(path)
                rows.append(row)
            write(folder/'record.json', {'case_id': case['case_id'], 'arm': arm, 'rows': rows,
                'fit_input_sha256': case['input_sha256'], 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
            write(out/'progress.json', {'case_id': case['case_id'], 'arm': arm, 'done': True})
    write(out/'worker_complete.json', {'fit_calls': fit_calls, 'validation_DLT_calls': validation_calls,
                                     'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


def run(cache, feasibility, out, config):
    start = time.perf_counter()
    bundle = prepare(cache, feasibility, out, config)
    budget = bundle['config']['budget']; failure = None
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
    with (out/'worker.stdout.txt').open('w') as stdout, (out/'worker.stderr.txt').open('w') as stderr:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(out)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            while proc.poll() is None:
                if time.perf_counter()-start > budget['session_seconds']:
                    raise TimeoutError('session time limit')
                path = out/'progress.json'
                if path.exists():
                    progress = read(path)
                    if not progress['done'] and time.time()-progress['started'] > budget['pair_arm_seconds']:
                        raise TimeoutError('pair-arm time limit')
                status = Path(f'/proc/{proc.pid}/status')
                if status.exists():
                    rss = next((int(s.split()[1])*1024 for s in status.read_text().splitlines() if s.startswith('VmRSS:')), 0)
                    if rss > budget['process_rss_gib']*1024**3:
                        raise MemoryError('worker RSS limit')
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > budget['artifacts_gib']*1024**3:
                    raise MemoryError('artifact limit')
                time.sleep(.1)
            if proc.returncode:
                raise RuntimeError('worker failed; preserve stderr and partial records')
        except Exception as error:
            failure = {'error': repr(error), 'type': type(error).__name__}
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    write(out/'session.json', {'complete': failure is None, 'failure': failure,
        'wall_seconds': time.perf_counter()-start, 'gpu_compute': False, 'worker_exit_code': proc.returncode})
    print(json.dumps(read(out/'session.json'), indent=2))
    return failure is None


def evaluate(cache, out):
    from georecon.matching.evaluate import truth_for_case
    from georecon.matching.metrics import pose_metrics
    from georecon.matching.session import load_protocol
    if (out/'evaluation.json').exists():
        raise ValueError('refuse to overwrite offline diagnostic evaluation')
    protocol, bundle, provenance = load_protocol(ROOT), read(out/'bundle.json'), read(out/'provenance.json')
    for name, sha in provenance['source_sha256'].items():
        if digest(ROOT/name) != sha or digest(out/'sources'/name) != sha:
            raise ValueError('run implementation changed')
    for name, sha in provenance['input_sha256'].items():
        if digest(name) != sha:
            raise ValueError('source inputs changed')
    if digest(cache/'evaluation.json') != bundle['config']['stage7a_full_evaluation_sha256']:
        raise ValueError('original full-fit evaluation changed')
    for name, sha in read(cache/'input_hashes.json').items():
        if digest(ROOT/name) != sha:
            raise ValueError('original DTU inputs changed')
    old = read(cache/'evaluation.json')
    benefit = set(old['decision']['rescued_pairs']); harm = {v['case_id'] for v in old['decision']['new_false_accepts']}
    normal = set(old['decision']['normal_regressions']); cfg = bundle['config']
    rows, pairs = [], []
    for case in bundle['cases']:
        truth = truth_for_case(ROOT, protocol, case)
        records = {arm: read(out/'fits'/case['case_id']/arm/'record.json') if (out/'fits'/case['case_id']/arm/'record.json').exists() else {'rows': []} for arm in ['B', 'C']}
        focus = 'benefit' if case['case_id'] in benefit else 'harm' if case['case_id'] in harm else 'other'
        case_rows = []
        for seed in cfg['geometry_seeds']:
            arms = {}
            for arm in ['B', 'C']:
                rec = next((r for r in records[arm]['rows'] if r['seed'] == seed), {'status': 'not_run', 'reason': 'missing_record', 'validation': None})
                pose = pose_metrics(*truth, None, None, False)
                if rec.get('arrays_sha256'):
                    path = out/'fits'/case['case_id']/arm/f'seed_{seed}.npz'
                    if digest(path) != rec['arrays_sha256']:
                        raise ValueError('saved geometry arrays changed')
                    with np.load(path, allow_pickle=False) as data:
                        if rec['status'] == 'success':
                            pose = pose_metrics(*truth, data['accepted_R'], data['accepted_t'], True)
                arms[arm] = {'fit': rec, 'pose': pose}
            comparable = all(arms[a]['pose']['accepted'] for a in ['B', 'C'])
            vb, vc = [arms[a]['fit']['validation'] for a in ['B', 'C']]
            signals = {key: preference(vb[key] if vb else None, vc[key] if vc else None, cfg['comparison_tolerance'])
                       for key in ['primary_loss', 'source_cell_balanced_loss']}
            strata = {s: preference(vb['strata'][s]['capped_loss'] if vb else None, vc['strata'][s]['capped_loss'] if vc else None, cfg['comparison_tolerance']) for s in ['B', 'C', 'common', 'B_only', 'C_only', 'union']}
            category = ('useful_C' if not arms['B']['pose']['correct'] and arms['C']['pose']['correct'] else
                        'harmful_C' if arms['B']['pose']['correct'] and not arms['C']['pose']['correct'] else
                        'both_correct' if arms['B']['pose']['correct'] else 'both_wrong') if comparable else 'unknown_unpaired'
            row = {'case_id': case['case_id'], 'scan': case['scan'], 'focus_original_full_fit': focus,
                   'normal_regression_original': case['case_id'] in normal, 'seed': seed, 'arms': arms,
                   'comparable': comparable, 'shadow_GT_category': category, 'diagnostic_preference': signals,
                   'stratum_preference': strata, 'replacement_performed': False}
            rows.append(row); case_rows.append(row)
        flat = {'case_id': case['case_id'], 'scan': case['scan'], 'original_focus': focus,
                'original_normal_regression': case['case_id'] in normal,
                'eligibility_reasons': '|'.join(case['eligibility_reasons'])}
        for arm in ['B', 'C']:
            flat[arm+'_fit_raw'] = case['counts'][arm]['raw_fit']; flat[arm+'_validation_edges'] = case['counts'][arm]['validation']
            for key, fn in [('status', lambda r: r['arms'][arm]['fit']['status']),
                            ('joint_deg', lambda r: r['arms'][arm]['pose']['joint_deg']),
                            ('primary_loss', lambda r: r['arms'][arm]['fit']['validation']['primary_loss'] if r['arms'][arm]['fit']['validation'] else None)]:
                flat[arm+'_'+key+'_by_seed'] = '|'.join(str(fn(r)) for r in case_rows)
        for name, fn in [('primary', lambda r:r['diagnostic_preference']['primary_loss']),
                         ('grid', lambda r:r['diagnostic_preference']['source_cell_balanced_loss']),
                         ('shadow_GT', lambda r:r['shadow_GT_category'])]:
            flat[name+'_by_seed'] = '|'.join(fn(r) for r in case_rows)
        for s in ['B', 'C', 'common', 'B_only', 'C_only', 'union']:
            flat['stratum_'+s+'_by_seed'] = '|'.join(r['stratum_preference'][s] for r in case_rows)
        pairs.append(flat)
    summary = {}
    for group in ['all', 'benefit', 'harm', 'other']:
        selected = [r for r in rows if r['seed'] == 42 and (group == 'all' or r['focus_original_full_fit'] == group)]
        summary[group] = {'planned_pairs': len(selected), 'comparable': sum(r['comparable'] for r in selected),
            'primary': {p: sum(r['diagnostic_preference']['primary_loss'] == p for r in selected) for p in ['B_lower', 'C_lower', 'tie', 'unknown']},
            'shadow_GT': {p: sum(r['shadow_GT_category'] == p for r in selected) for p in ['useful_C', 'harmful_C', 'both_correct', 'both_wrong', 'unknown_unpaired']}}
    write(out/'evaluation.json', {'rows': rows, 'summary': summary, 'session': read(out/'session.json'),
                                'no_policy_or_hybrid_result': True, 'original_stage7a_verdict_unchanged': old['decision']['verdict']})
    with (out/'pairs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]), lineterminator='\n'); w.writeheader(); w.writerows(pairs)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['run', 'worker', 'evaluate'])
    p.add_argument('--cache'); p.add_argument('--feasibility'); p.add_argument('--output', required=True); p.add_argument('--config')
    a = p.parse_args(); out = Path(a.output).resolve()
    if a.operation == 'worker':
        bundle = read(out/'bundle.json')
        sys.addaudithook(guard)
        try:
            worker(bundle)
        except Exception:
            write(out/'worker_failure.json', {'traceback': traceback.format_exc()}); raise
    elif a.operation == 'run':
        if not all([a.cache, a.feasibility, a.config]): p.error('run requires cache, feasibility and config')
        raise SystemExit(0 if run(Path(a.cache).resolve(), Path(a.feasibility).resolve(), out, Path(a.config).resolve()) else 1)
    else:
        if not a.cache: p.error('evaluate requires cache')
        evaluate(Path(a.cache).resolve(), out)


if __name__ == '__main__':
    main()
