"""Explicit source MLA cases and bounded same-input acceptance controls.

[Role] Retain source dispatch, primitive comparison and two complete Body runs.
[Invariants] No fitted split count, service coefficient or whole-Body correction.
[Timeouts & Locks] Body worker 90s/512MiB; supervisor allows 5s cleanup grace.
[Side-Effect Policy] Local case/evidence JSON only; no hardware or model weights.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BATCHES = (1, 2, 4, 6, 8, 12)
CASES = ROOT / 'examples/cases/kimi-a5-paged-host-20261030'
OUTPUT = ROOT / 'data/verification/kimi-a5-paged-host-20261030'
SOURCE_CASE = ROOT.parent / 'profiler/examples/structure/cases/a5-kimi93-tp8-longctx-source.json'
IMPLEMENTATION = 'ascend_paged_mla_decode_arch35'
POLICY = 'source_split_core_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def case_path(batch):
    return CASES / f'c{batch}.json'


def prepare():
    source = json.loads(SOURCE_CASE.read_text())
    rows = []
    for batch in BATCHES:
        raw = copy.deepcopy(source)
        raw['name'] = f'Kimi A5 TP8 q1 128K B{batch} source paged MLA'
        raw['runtime']['kernels']['lowerings']['paged_mla_attention'] = {
            'implementation': IMPLEMENTATION, 'options': {'dispatch': POLICY}}
        raw['runtime']['metadata']['paged_mla_source_control'] = {
            'source_case_sha256': sha(SOURCE_CASE),
            'scope': 'Only the q1 dense MLA source implementation binding changes.',
            'capture_dispatch': 'conditional_source_compatible; binary_revision_unobserved',
            'source_policy': 'SplitCoreV2 M64/KV128, fdTolerance=9; no timing inputs',
            'native_query_capacity': 'existing graph-capture padding; inactive KV length zero'}
        path = case_path(batch)
        write(path, raw)
        rows.append({'batch': batch, 'case': str(path.relative_to(ROOT)), 'sha256': sha(path),
                     'scope': 'body', 'phase': 'decode', 'tokens': 1, 'context': 126000})
    write(OUTPUT / 'inputs.json', {'source_case': {'path': str(SOURCE_CASE),
          'sha256': sha(SOURCE_CASE)}, 'cases': rows,
          'training': 'none; source ownership and existing declared physical services'})


def identity(path, batch):
    files = ('family_paged_mla.py', 'paged_mla.py', 'paged_mla_host_v2.py',
             'paged_mla_dispatch.py', 'engine.py', 'program.py')
    return {'case': {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)},
            'runtime_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                cwd=ROOT, text=True, timeout=3).strip(),
            'runtime_files_sha256': {f: sha(ROOT / 'mindie_sim/prediction' / f) for f in files},
            'geometry': {'scope': 'body', 'phase': 'decode', 'tokens': 1,
                         'context': 126000, 'sequences': batch}}


def core(native_path):
    from mindie_sim import target_graphs, operator_costs
    native = json.loads(native_path.read_text())
    rows = []
    for batch in BATCHES:
        path = case_path(batch); raw = json.loads(path.read_text())
        graph = target_graphs(raw, scope='body', phase='decode', tokens=1,
                              sequences=batch, context=126000)[0]
        op = next(o for o in graph if o.recipe == IMPLEMENTATION)
        started = time.monotonic()
        result = operator_costs(raw, [op], input_ready_ms=7)['operators'][0]
        captured = next(c for c in native['cases'] if c['case_id'].endswith('_c' + str(batch)))
        observed = next(a for a in captured['native_abi_groups']
                        if a['abi']['Type'] == 'FusedInferAttentionScore')
        rows.append({**identity(path, batch), 'status': 'completed',
            'wall_seconds': time.monotonic()-started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'source': captured['source'], 'extraction': captured['extraction'],
            'native_primitive': observed, 'native_body_comparison': captured['comparison'],
            'duration_ms': result['duration_ms'],
            'relative_error': result['duration_ms']*1000/observed['mean_us']-1,
            'active_shape': op.local_shape, 'padded_shape': op.padded_shape,
            'geometry_metadata': dict(op.metadata)['paged_mla_geometry'],
            'program_metadata': result['operator']['metadata']['program_execution'],
            'work': {k: result['operator'][k] for k in
                     ('hbm_bytes', 'compute_flops', 'resource_work') if k in result['operator']},
            'live_tasks': result['live_tasks'], 'remaining_buffers': result['remaining_buffers'],
            'service_before_input': result['service_before_input']})
        write(OUTPUT / 'core.json', {'boundary': 'one native MLA kernel; no Body fitting',
            'capture_dispatch': 'conditional_source_compatible; original source binary unobserved',
            'scope': 'six source controls; not a trained or heldout accuracy claim', 'cases': rows})
        print(batch, round(result['duration_ms']*1000, 3), flush=True)


def worker(batch):
    from mindie_sim import predict
    path = case_path(batch); started = time.monotonic(); cpu = time.process_time()
    record = {**identity(path, batch), 'status': 'running', 'budget_seconds': 90,
              'budget_rss_bytes': 512*1024*1024}
    def deadline(*_):
        raise TimeoutError('complete_body_90s_budget_exhausted')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(90)
    try:
        result = predict(json.loads(path.read_text()), event_limit=0, **record['geometry'])
        record.update(status='completed', body_forward_ms=result['body_forward_ms'],
            execution=result['execution'], result_sha256=hashlib.sha256(
                json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0)
        record.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(OUTPUT / f'body-c{batch}.json', record)
    print(json.dumps({k: record[k] for k in ('status', 'wall_seconds', 'peak_rss_bytes')}), flush=True)
    return 0 if record['status'] == 'completed' else 1


def body(batch):
    started = time.monotonic(); peak = 0; failure = None
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--batch', str(batch)])
    while proc.poll() is None:
        try:
            rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(proc.pid)],
                capture_output=True, text=True, timeout=2, check=False).stdout.strip()
            peak = max(peak, int(rss or 0)*1024)
        except subprocess.TimeoutExpired:
            failure = 'rss_sampling_timeout'; break
        if peak > 512*1024*1024:
            failure = 'process_rss_exceeded_512MiB'; break
        if time.monotonic()-started > 95:
            failure = 'outer_95s_budget_exhausted'; break
        time.sleep(.25)
    if failure:
        proc.terminate()
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=2)
        write(OUTPUT / f'body-c{batch}.json', {**identity(case_path(batch), batch),
            'status': 'failed', 'error': failure, 'wall_seconds': time.monotonic()-started,
            'peak_rss_bytes': peak})
    return proc.returncode or (1 if failure else 0)


def summary():
    native = json.loads((OUTPUT / 'core.json').read_text())
    rows = []
    for result in native['cases']:
        batch = result['geometry']['sequences']
        row = {'batch': batch, 'active_queries': batch,
            'native_query_capacity': result['geometry_metadata']['native_query_capacity'],
            'core_predicted_us': result['duration_ms']*1000,
            'core_observed_mean_us': result['native_primitive']['mean_us'],
            'core_relative_error': result['relative_error'],
            'body_status': 'not_run_in_this_control'}
        path = OUTPUT / f'body-c{batch}.json'
        if path.exists():
            record = json.loads(path.read_text())
            row.update(body_status=record['status'], body_wall_seconds=record['wall_seconds'],
                       body_peak_rss_bytes=record['peak_rss_bytes'])
            if record['status'] == 'completed':
                observed = result['native_body_comparison']['observed_ms']['mean']
                row.update(body_ms=record['body_forward_ms'], native_body_mean_ms=observed,
                           body_relative_error=record['body_forward_ms']/observed-1)
            else:
                row['error'] = record['error']
        rows.append(row)
    write(OUTPUT / 'summary.json', {'schema': 'kimi-a5-source-host-acceptance/1',
        'boundary': 'q1 non-speculative Decode, context126000; primitive and complete93-layer Body separate',
        'accuracy_basis': 'conditional source-compatible binary/host dispatch',
        'training': 'none; M64/KV128/fdTolerance9 ownership; no fitted split count',
        'acceptance': 'Primitive errors do not establish whole-Body acceptance; failed runs have no prediction.',
        'cases': rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'core', 'body', 'worker', 'summary'))
    parser.add_argument('--batch', type=int, choices=(1, 12), default=1)
    parser.add_argument('--native-ledger', type=Path, default=Path('/tmp/kimi-a5-native-ledger-20261030.json'))
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'core': core(args.native_ledger)
    elif args.action == 'summary': summary()
    else: raise SystemExit(worker(args.batch) if args.action == 'worker' else body(args.batch))
