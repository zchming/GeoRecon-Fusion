"""Bounded Stage 7A orchestration; immutable local artifacts, no GT scoring."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import subprocess
import sys
import time

PROTOCOL_SHA256 = 'b4e301115a5f90413290eb496baee1e8b48e15e3779e6e6f1c892866146233c8'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_protocol(root):
    path = root/'configs/experiments/stage7a_protocol_v1.json'
    if digest(path) != PROTOCOL_SHA256:
        raise ValueError('frozen protocol bytes changed')
    p = json.loads(path.read_text())
    for name, key in [('source', 'sha256'), ('config_source', 'config_sha256')]:
        if digest(root/p['backend'][name]) != p['backend'][key]:
            raise ValueError('frozen geometric backend changed')
    return p


def prepare(root, out, vendor, checkpoint):
    p = load_protocol(root)
    actual = {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in p['lightglue']['runtime'] if name != 'python'}}
    if actual != p['lightglue']['runtime']:
        raise ValueError(f'frozen runtime mismatch: {actual}')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vendor, text=True).strip() != p['lightglue']['vendor_commit'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=vendor, text=True):
        raise ValueError('LightGlue vendor version or working tree mismatch')
    for path, expected in p['lightglue']['source_sha256'].items():
        if digest(vendor/path) != expected:
            raise ValueError('LightGlue source digest mismatch')
    if digest(checkpoint) != p['lightglue']['weights_sha256']:
        raise ValueError('cached checkpoint digest mismatch')
    images, input_hashes = {}, {}
    for lock in p['input']['input_locks']:
        for key, sha in [('manifest', 'manifest_sha256'), ('reference_evaluation_only_except_K', 'reference_sha256'), ('common_input_mapping', 'common_input_mapping_sha256')]:
            path = root/lock[key]
            if digest(path) != lock[sha]:
                raise ValueError(f'input lock mismatch: {lock[key]}')
            input_hashes[lock[key]] = lock[sha]
        m = json.loads((root/lock['manifest']).read_text())
        ref = json.loads((root/lock['reference_evaluation_only_except_K']).read_text())
        mapping_path = root/lock['common_input_mapping']
        mapping = json.loads(mapping_path.read_text())
        if ref['input_manifest_sha256'] != lock['manifest_sha256']:
            raise ValueError('reference manifest identity mismatch')
        if [i['image_id'] for i in m['images']] != [i['image_id'] for i in mapping]:
            raise ValueError('image mapping order mismatch')
        cameras = {i['image_id']: i['camera'] for i in ref['images']}
        for item, staged in zip(m['images'], mapping):
            raw = root/Path(lock['manifest']).parent/item['path']
            common = mapping_path.parent/'images'/staged['worker_name']
            if digest(raw) != item['sha256'] or staged['source_sha256'] != item['sha256'] or digest(common) != staged['processed_sha256']:
                raise ValueError('raw/common image bytes changed')
            input_hashes[str(raw.relative_to(root))] = item['sha256']
            input_hashes[str(common.relative_to(root))] = staged['processed_sha256']
            # Explicitly allowed calibration K only: no pose value copied into bundle.
            k, a = cameras[item['image_id']]['K'], staged['pixel_transform']
            processed = [[sum(a[r][j]*k[j][c] for j in range(3)) for c in range(3)] for r in range(3)]
            group = 'blur' if lock['condition'] == 'blur5' else 'sharp'
            key = f"scan{m['scene_id'].split('_')[1][4:]}_{group}_{item['image_id']}"
            value = {'path': str(common), 'sha256': staged['processed_sha256'], 'K': processed, 'width': 800, 'height': 600}
            if key in images:
                if any(images[key][name] != value[name] for name in ['sha256', 'K', 'width', 'height']):
                    raise ValueError('deduplicated input differs')
            else:
                images[key] = value
    if len(images) != 24:
        raise ValueError('unique feature input count differs')
    cases = []
    for c in p['input']['cases']:
        group = 'blur' if 'blur5' in c['conditions'] else 'sharp'
        cases.append({**c, 'image_keys': [f"scan{c['scan']}_{group}_{i}" for i in c['view_ids']]})
    bundle = {'images': images, 'cases': cases, 'feature_paths': {}, 'features': p['features'],
        'backend_config': json.loads((root/p['backend']['config_source']).read_text()),
        'budget': p['budget'], 'checkpoint': str(checkpoint), 'weights_source': p['lightglue']['weights_source'],
        'matcher_config': p['arms']['C_rootsift_lightglue']['matcher_config']}
    out.mkdir(parents=True, exist_ok=False)
    write(out/'bundle.json', bundle)
    write(out/'input_hashes.json', input_hashes)
    paths = sorted((root/'src/georecon/matching').glob('*.py')) + [root/'scripts/run_stage7a.py'] + sorted((root/'tests').glob('test_stage7a*.py')) + [root/p['backend']['source'], root/'configs/experiments/stage7a_protocol_v1.json']
    sources = {}
    for path in paths:
        name = str(path.relative_to(root))
        sources[name] = digest(path)
        target = out/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write(out/'implementation.json', {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'implementation_not_yet_committed': True, 'source_sha256': sources, 'protocol_sha256': PROTOCOL_SHA256,
        'runtime': actual, 'authority': 'user explicitly authorized frozen Stage7A implementation and execution; no commit/push/7B'})
    return p, bundle


class Limit(RuntimeError):
    pass


class Client:
    def __init__(self, root, out, mode, env, bundle_path):
        self.log = (out/f'{mode}-{time.time_ns()}.stderr.txt').open('w')
        self.proc = subprocess.Popen([sys.executable, '-u', '-m', 'georecon.matching.worker', '--bundle', str(bundle_path), '--mode', mode],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)

    def receive(self, seconds, check):
        start = time.perf_counter()
        while True:
            check(self.proc.pid)
            if time.perf_counter()-start > seconds:
                raise Limit('per-operation wall timeout')
            if self.selector.select(.1):
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'worker exited {self.proc.poll()}; inspect stderr')
                return json.loads(line)

    def send(self, request, seconds, check):
        start = time.perf_counter()
        self.proc.stdin.write(json.dumps(request)+'\n')
        self.proc.stdin.flush()
        reply = self.receive(seconds, check)
        reply['controller_seconds'] = time.perf_counter()-start
        return reply

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.selector.close()
        self.log.close()


def run_session(root, out, vendor, checkpoint):
    start = time.perf_counter()
    p, bundle = prepare(root, out, vendor, checkpoint)
    budget = p['budget']
    env = {**os.environ, 'PYTHONPATH': str(root/'src'), 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    (out/'logs').mkdir()
    gpu_log = (out/'gpu_memory.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '--loop-ms=100'], stdout=gpu_log, stderr=subprocess.DEVNULL)
    clients, operations, resource_failures = {}, [], []
    def check(pid=None):
        if time.perf_counter()-start > budget['total_session_wall_seconds']:
            raise Limit('session wall budget')
        if pid:
            status = Path(f'/proc/{pid}/status')
            if status.exists():
                memory = next((int(s.split()[1])*1024 for s in status.read_text().splitlines() if s.startswith('VmRSS:')), 0)
                if memory > budget['process_rss_ceiling_gib']*1024**3:
                    raise Limit('process RSS budget')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > budget['local_artifact_ceiling_gib']*1024**3:
            raise Limit('local artifact budget')
        with (out/'gpu_memory.csv').open('rb') as log:
            log.seek(max(0, (out/'gpu_memory.csv').stat().st_size-4096))
            values = [int(s) for s in log.read().splitlines() if s.strip().isdigit()]
        if values and max(values)*1024**2 > budget['whole_card_ceiling_bytes']:
            raise Limit('whole-card GPU budget')
    def get_client(mode):
        if mode not in clients:
            client = Client(root, out/'logs', mode, env, out/'bundle.json')
            clients[mode] = client
            ready = client.receive(180, check)
            operations.append({'operation': f'{mode}_startup', **ready})
        return clients[mode]
    def request(mode, operation, seconds):
        reply = get_client(mode).send(operation, seconds, check)
        operations.append({**operation, **reply})
        if reply['peak_rss_bytes'] > budget['process_rss_ceiling_gib']*1024**3:
            raise Limit('measured RSS high-water budget')
        if reply['status'] != 'success':
            if reply['status'] in ['oom', 'resource_guard']:
                resource_failures.append(reply)
            raise RuntimeError(f"worker {reply['status']}: {reply['record']}")
        write(out/'operations.json', operations)
        return reply
    complete, failure = False, None
    try:
        for key in bundle['images']:
            folder = out/'features'/key
            reply = request('cpu', {'operation': 'feature', 'image_key': key, 'output': str(folder)}, budget['feature_wall_seconds_per_image'])
            bundle['feature_paths'][key] = str(folder/'features.npz')
            print('FEATURE', key, flush=True)
        write(out/'bundle.json', bundle)
        # CPU server reloads the now-complete cache locations, preserving the same files.
        clients['cpu'].close()
        del clients['cpu']
        warm_case = next(c for c in bundle['cases'] if c['case_id'] == budget['cold_profile_case_ids'][0])
        for i in range(budget['lightglue_warmup_calls']):
            request('gpu', {'operation': 'warmup', 'arm': 'C', 'case': warm_case, 'seeds': [], 'output': str(out/'warmup'/str(i))}, 180)
        for index, case in enumerate(bundle['cases']):
            order = ['ABC', 'BCA', 'CAB'][index % 3]
            for arm in order:
                req = {'operation': 'pair', 'arm': arm, 'case': case, 'seeds': p['input']['geometry_seeds'], 'output': str(out/'main'/case['case_id']/arm)}
                request('gpu' if arm == 'C' else 'cpu', req, budget['pair_arm_wall_seconds_including_three_geometry_seeds'])
                print('PAIR', case['case_id'], arm, flush=True)
        for client in clients.values():
            client.close()
        clients.clear()
        for case_id in budget['cold_profile_case_ids']:
            case = next(c for c in bundle['cases'] if c['case_id'] == case_id)
            for arm in 'ABC':
                folder = out/'cold'/case_id/arm
                req = {'operation': 'pair', 'cold': True, 'arm': arm, 'case': case, 'seeds': [42], 'output': str(folder)}
                request_path = out/'cold_requests'/f'{case_id}_{arm}.json'
                write(request_path, req)
                begin = time.perf_counter()
                with (out/'logs'/f'cold-{case_id}-{arm}.stdout.txt').open('w') as stdout, (out/'logs'/f'cold-{case_id}-{arm}.stderr.txt').open('w') as stderr:
                    proc = subprocess.Popen([sys.executable, '-u', '-m', 'georecon.matching.worker', '--bundle', str(out/'bundle.json'), '--mode', 'cold', '--request', str(request_path)], cwd=root, env=env, stdout=stdout, stderr=stderr)
                    try:
                        while proc.poll() is None:
                            check(proc.pid)
                            if time.perf_counter()-begin > budget['cold_profile_wall_seconds_per_run']:
                                raise Limit('cold profile timeout')
                            time.sleep(.1)
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                            proc.wait()
                record = json.loads((folder/'record.json').read_text()) if (folder/'record.json').exists() else {'status': 'process_error'}
                op = {'operation': 'cold', 'case_id': case_id, 'arm': arm, 'seconds': time.perf_counter()-begin, 'exit_code': proc.returncode, 'status': record['status']}
                operations.append(op)
                write(out/'operations.json', operations)
                if record['status'] in ['oom', 'resource_guard']:
                    resource_failures.append(op)
                if proc.returncode or record['status'] != 'success':
                    raise RuntimeError('cold profile failed; preserve original files')
                print('COLD', case_id, arm, flush=True)
        complete = True
    except Exception as error:
        failure = {'error': repr(error), 'type': type(error).__name__}
        if isinstance(error, Limit):
            resource_failures.append(failure)
    finally:
        for client in clients.values():
            client.close()
        monitor.terminate()
        monitor.wait(timeout=5)
        gpu_log.close()
        write(out/'operations.json', operations)
        write(out/'session.json', {'complete': complete, 'failure': failure, 'resource_failures': resource_failures,
            'full_wall_seconds': time.perf_counter()-start, 'protocol_sha256': PROTOCOL_SHA256,
            'implementation_not_yet_committed': True, 'GT_scoring_in_session': False})
    return complete
