"""Execute complete-model long-context Decode probes from frozen source cases.

[Role] Prepare reusable scenario inputs and run bounded device-body probes.
[Invariants] Prompt/output are 126000/3000. Complete layers and TP/DP/EP survive;
  global concurrency is assigned by an explicit per-DP vector. This is a body
  snapshot, not a 3000-token Serving replay or a client TPOT measurement.
  Existing physical parameters are frozen; no observed duration is consumed.
[Timeouts & Locks] Each child has 90s and 512MiB sampled RSS, plus 5s cleanup.
[Side-Effect Policy] Local input/result files only; no device/remote work.
"""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'profiler/examples/structure/cases'
CASES = ROOT / 'examples/cases/longctx-decode-20261030'
OUT = ROOT / 'data/verification/longctx-decode-20261030'
SCRATCH = Path(tempfile.gettempdir()) / 'mindie-longctx-decode-20261030'
CONTEXTS = (126000, 127500, 128999)
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
BASES = {
    'qwen35-a5-tp4-ep4': 'decode-qwen35-35b-a5-tp4-c4-graph.json',
    'qwen36-a5-tp2': 'decode-qwen36-a5-tp2-c1.json',
    'qwen36-a5-tp8': 'decode-a5_qwen36_highconc_c4.json',
    'qwen38-a5-tp4': 'decode-a5_qwen38_highconc_c4.json',
    'v41-a5-tp1-dp8-ep8': 'decode-a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1.json',
    'v4flash-a3-tp8-ep8': 'decode-v4flash-a3-tp8-c4.json',
    'qwen35-35b-a3-tp4': 'decode-qwen35-35b-a3-tp4-c144.json',
    'qwen35-397b-a3-tp16': 'decode-qwen35-397b-a3-tp16-c48.json',
    'qwen36-a3-tp4': 'decode-qwen36-a3-tp4-c64.json',
    'qwen38-a3-tp4': 'decode-qwen38-a3-tp4-c64.json',
    'v41-a3-tp4-dp4-ep16': 'decode-a3_v41_w8a8_tp4_dp4_ep16_dspark7_c4.json',
}


def identity(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': Path(path).stat().st_size}


def package_identity():
    rows = [(str(path.relative_to(ROOT)), identity(path)['sha256'])
            for path in sorted((ROOT / 'mindie_sim').rglob('*.py'))]
    return {'sha256': sha256(json.dumps(rows).encode()).hexdigest(), 'files': len(rows)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sequences(global_concurrency, dp):
    """Deterministic balanced scenario; it does not assert observed routing."""
    whole, extra = divmod(global_concurrency, dp)
    return [whole + int(replica < extra) for replica in range(dp)]


def prepare():
    sys.path.insert(0, str(ROOT))
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.prediction.decode_probe import decode_windows
    from mindie_sim.prediction.dispatch import cohorts
    windows = decode_windows(126000, 3000, query_tokens=1, output_offsets=[0, 1500, 2999])
    assert tuple(window['context'] for window in windows) == CONTEXTS
    deployments, matrix = [], []
    for name, filename in BASES.items():
        source_path = SOURCE / filename
        path = CASES / (name + '.json')
        prepared = path.exists()
        raw = json.loads((path if prepared else source_path).read_text())
        baseline = deepcopy(raw)
        retained = raw['runtime'].get('metadata', {}).get('longctx_decode_scenario', {})
        source = retained['source'] if prepared else identity(source_path)
        changes = deepcopy(retained.get('source_domain_changes', []))
        raw['name'] = name + '-longctx-decode-126000-3000'
        raw['workload']['requests'] = [{'id': 'r0', 'prompt_tokens': 126000, 'output_tokens': 3000}]
        runtime = raw['runtime']
        runtime['max_model_len'] = max(runtime.get('max_model_len', 0), 129000)
        runtime['max_num_seqs'] = max(runtime.get('max_num_seqs', 0), 64)
        runtime.setdefault('metadata', {})['longctx_decode_scenario'] = {
            'source': source, 'prompt_tokens': 126000, 'output_tokens': 3000,
            'snapshot_contexts': list(CONTEXTS), 'global_concurrency': list(CONCURRENCY),
            'distribution': 'explicit balanced global concurrency across preserved DP replicas',
            'query_tokens_per_sequence': 1,
            'boundary': 'complete target model body, one Decode query per active sequence',
            'not_executed': ['Prefill', 'output head', 'sampling', 'draft model', 'host scheduler', 'client transport'],
            'accuracy': 'prediction scenario; no same-boundary long-context measurement bound',
            'parameters': 'existing source-case physical services unchanged; no new fit',
        }
        assert raw['model'] == baseline['model'] and raw['machine'] == baseline['machine']
        assert runtime.get('kernels') == baseline['runtime'].get('kernels')
        partition = runtime.get('kernels', {}).get('moe_token_partition')
        if not prepared and partition and partition['kind'] == 'vllm_ascend_mc2_tp_slice':
            previous = partition.get('padding', 'none')
            partition['padding'] = 'right_to_tp'
            changes.append({'selector': 'moe_token_partition.padding', 'previous': previous,
                            'selected': 'right_to_tp',
                            'reason': 'existing source MC2 caller pads nondivisible query rows after router/shared work; retained captures used divisible rows',
                            'parameter_changes': 'no service/rate change'})
        runtime['metadata']['longctx_decode_scenario']['source_domain_changes'] = changes
        case = parse_case(raw)
        participants = list(cohorts(case.runtime.parallel)[0])
        assert participants == list(range(case.runtime.parallel.dp)), 'matrix_requires_complete_deployment_EP_cohort'
        path = CASES / (name + '.json')
        write(path, raw)
        deployment = {'id': name, 'input': identity(path), 'source': source,
                      'model': case.model.name, 'layers': len(case.model.layer_plans),
                      'selected_participants': participants, 'unprobed_replicas': [],
                      'parallel': {key: getattr(case.runtime.parallel, key)
                                   for key in ('tp', 'dp', 'ep', 'pp', 'pcp')},
                      'source_domain_changes': changes,
                      'runtime_changes': {'max_model_len': runtime['max_model_len'],
                                          'max_num_seqs': runtime['max_num_seqs']}}
        deployments.append(deployment)
        for concurrency in CONCURRENCY:
            counts = sequences(concurrency, case.runtime.parallel.dp)
            for window in windows:
                context = window['context']
                matrix.append({'id': f'{name}-ctx{context}-c{concurrency}',
                               'deployment': name, 'input': deployment['input'],
                               'context_tokens': context, 'query_tokens_per_sequence': window['tokens'],
                               'generated_before': window['generated_before'], 'kv_length': window['kv_length'],
                               'global_concurrency': concurrency,
                               'sequences_by_replica': counts,
                               'selected_participants': participants, 'unprobed_replicas': [],
                               'sequences_in_executed_cohort': sum(counts),
                               'per_dp_input': None,
                               'symmetric_per_dp': len(set(counts)) == 1,
                               'complete_layers': deployment['layers'],
                               'parallel': deployment['parallel'],
                               'status': 'prepared_not_executed'})
    record = {'schema': 'mindie-longctx-decode-matrix/1', 'deployments': deployments,
              'matrix': matrix, 'producer': identity(Path(__file__)),
              'source_read_policy': 'Existing prepared base cases are authoritative; the mutable source catalogue is read only to materialize a missing base.',
              'workload': {'prompt_tokens': 126000, 'output_tokens': 3000},
              'contexts': list(CONTEXTS), 'global_concurrency': list(CONCURRENCY),
              'concurrency_contract': {
                  'global': 'sum(sequences_by_replica); explicitly distributed across every DP',
                  'symmetric_per_dp': 'all DP counts equal; per-DP count c would mean global c*DP',
                  'idle_replicas': 'kept in EP cohort; no DP reduction to obtain a pass'},
              'limits': {'wall_seconds': 90, 'sampled_rss_bytes': 512 * 2**20, 'cleanup_seconds': 5}}
    write(OUT / 'inputs.json', record)
    return record


def effective_input(entry):
    raw = json.loads(Path(entry['input']['path']).read_text())
    assert identity(Path(entry['input']['path']))['sha256'] == entry['input']['sha256']
    raw['workload']['requests'] = [
        {'id': f'r{i}', 'prompt_tokens': 126000, 'output_tokens': 3000}
        for i in range(entry['global_concurrency'])]
    raw['runtime']['metadata']['longctx_decode_probe'] = {
        key: entry[key] for key in ('context_tokens', 'query_tokens_per_sequence',
                                   'global_concurrency', 'sequences_by_replica')}
    return raw


def worker(entry, output):
    sys.path.insert(0, str(ROOT))
    from mindie_sim import graph_costs, target_graphs
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.prediction.decode_probe import probe_capacity
    raw = effective_input(entry)
    case = parse_case(raw)
    assert len(case.model.layer_plans) == entry['complete_layers']
    assert sum(entry['sequences_by_replica']) == entry['global_concurrency']
    runtime_before = package_identity()
    capacity = probe_capacity(case, context=entry['context_tokens'],
                              tokens=entry['query_tokens_per_sequence'],
                              sequences=entry['sequences_by_replica'])
    started = time.perf_counter()
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    graphs = target_graphs(case, phase='decode', tokens=entry['query_tokens_per_sequence'],
                           context=entry['context_tokens'], sequences=entry['sequences_by_replica'],
                           scope='body')
    assert sorted(graphs) == entry['selected_participants']
    partitions = {}
    for replica, ops in graphs.items():
        for op in ops:
            partition = dict(op.metadata).get('moe_token_partition')
            if partition:
                key = (replica, partition['full_rows'], partition['local_rows'], partition['allocated_rows'])
                partitions[key] = {'replica': replica, **partition}
    print(json.dumps({'event': 'graph_constructed', 'operators_by_replica':
                      {replica: len(ops) for replica, ops in graphs.items()}}), flush=True)
    result = graph_costs(case, graphs, external_inputs={'analysis_input': 0},
                         isolated=False, trace_limit=0)
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    record = {'schema': 'mindie-longctx-decode-body/1', 'status': 'completed',
              'scenario': entry, 'runtime_seconds': time.perf_counter() - started,
              'runtime_commit': commit, 'producer': identity(Path(__file__)),
              'runtime_before': runtime_before, 'runtime_after': package_identity(),
              'duration_ms': result['duration_ms'], 'execution': result['execution'],
              'parameters': {key: result['parameters'][key] for key in (
                  'operator_occurrences', 'calibration', 'regime_selection', 'unrecorded_mapping_invocations')},
              'caller_partitions': list(partitions.values()),
              'operators_by_replica': {replica: len(ops) for replica, ops in graphs.items()},
              'boundary': 'complete_target_body_single_decode_snapshot',
              'comparison': {'status': 'unavailable', 'reason': 'no matching long-context body measurement'},
              'capacity': capacity}
    record['runtime_unchanged'] = record['runtime_before'] == record['runtime_after']
    write(output, record)


def capacity_report(record):
    """Static declared capacity for every scenario; execution remains separate."""
    sys.path.insert(0, str(ROOT))
    from mindie_sim.prediction.decode_probe import probe_capacity
    capacities = []
    for entry in record['matrix']:
        raw = effective_input(entry)
        capacity = probe_capacity(raw, context=entry['context_tokens'],
                                  tokens=entry['query_tokens_per_sequence'],
                                  sequences=entry['sequences_by_replica'])
        capacities.append({'id': entry['id'], 'capacity': capacity})
    write(OUT / 'capacity.json', {'schema': 'mindie-longctx-decode-capacity/1',
                                 'basis': 'declared geometry; not actual allocator admission',
                                 'scenarios': capacities})
    return capacities


def summarize():
    """Retain actual outcomes without copying complete inputs or timeline dumps."""
    from collections import Counter
    inputs = json.loads((OUT / 'inputs.json').read_text())
    results = []
    for path in sorted(OUT.glob('*.receipt.json')):
        receipt = json.loads(path.read_text())
        row = {key: receipt.get(key) for key in ('status', 'returncode', 'elapsed_seconds',
               'peak_sampled_rss_bytes', 'runtime_unchanged', 'runtime_before', 'runtime_after')}
        row.update(id=receipt['scenario']['id'], receipt=identity(path))
        if receipt.get('attempt'):
            row['attempt'] = receipt['attempt']
        if receipt.get('result'):
            output = Path(receipt['result']['path'])
            result = json.loads(output.read_text())
            row.update(body_ms=result['duration_ms'], capacity=result['capacity'],
                       operators_by_replica=result['operators_by_replica'],
                       physical_work=result['execution']['work'],
                       source_service_selection=result['parameters']['regime_selection'])
        else:
            row['error'] = receipt.get('error_tail')
        results.append(row)
    capacities = json.loads((OUT / 'capacity.json').read_text())['scenarios']
    source_changes = []
    for deployment in inputs['deployments']:
        source_path = Path(deployment['source']['path'])
        current = identity(source_path)
        if current['sha256'] != deployment['source']['sha256']:
            kernels = json.loads(source_path.read_text())['runtime'].get('kernels', {})
            source_changes.append({'deployment': deployment['id'],
                'status': 'candidate_declared_not_accepted; frozen input preserved',
                'frozen_source': deployment['source'], 'current_source': current,
                'candidate_hccl': kernels.get('hccl'),
                'candidate_service_regime_count': len(kernels.get('service_regimes', []))})
    record = {'schema': 'mindie-longctx-decode-summary/1',
              'workload': inputs['workload'], 'contexts': inputs['contexts'],
              'deployments': len(inputs['deployments']), 'prepared_scenarios': len(inputs['matrix']),
              'capacity_counts': dict(Counter(row['capacity']['status'] for row in capacities)),
              'actual_runs': results, 'mutable_source_changes_not_promoted': source_changes,
              'boundary': 'full target-body single-Decode snapshot; no Prefill or Serving replay',
              'accuracy': 'unscored; no same-boundary measurement bound to these scenarios',
              'unbound': {'v4pro': 'no current reconciled Decode base identified'},
              'limitations': ['Declared capacity is not real allocator admission.',
                              'Full source-derived and empirical service domains remain explicit; unmatched entries do not gain a measurement.',
                              'Prepared scenarios are not completed executions.',
                              'Current probes use q=1; this is not a speculative q=4/6 verification trajectory.']}
    write(OUT / 'summary.json', record)
    return record


def run(entry):
    path = OUT / (entry['id'] + '.json')
    raw = effective_input(entry)
    effective_hash = sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    stdout, stderr = (SCRATCH / (entry['id'] + suffix) for suffix in ('.stdout.log', '.stderr.log'))
    runtime_before = package_identity()
    started, peak, reason = time.perf_counter(), 0, None
    with stdout.open('w') as out, stderr.open('w') as err:
        process = subprocess.Popen([sys.executable, __file__, 'worker', '--id', entry['id']],
                                   stdout=out, stderr=err)
        try:
            while process.poll() is None:
                sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                                        capture_output=True, text=True, timeout=2)
                if sample.stdout.strip():
                    peak = max(peak, int(sample.stdout.strip()) * 1024)
                if peak > 512 * 2**20:
                    reason = 'memory_limit'
                elif time.perf_counter() - started >= 90:
                    reason = 'timeout'
                if reason:
                    break
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        code = process.wait(timeout=1)
    runtime_after = package_identity()
    receipt = {'scenario': entry, 'effective_input_sha256': effective_hash,
               'effective_input_serialization': 'sorted compact JSON of base + declared global workload and probe metadata',
               'runtime_before': runtime_before, 'runtime_after': runtime_after,
               'runtime_unchanged': runtime_before == runtime_after,
               'status': reason or ('completed' if code == 0 else 'failed'),
               'returncode': code, 'elapsed_seconds': time.perf_counter() - started,
               'peak_sampled_rss_bytes': peak,
               'progress': [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]}
    if code == 0:
        receipt['result'] = identity(path)
        receipt['duration_ms'] = json.loads(path.read_text())['duration_ms']
    else:
        receipt['error_tail'] = stderr.read_text()[-3000:]
    write(path.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'worker', 'run', 'capacity', 'summarize'))
    parser.add_argument('--id')
    args = parser.parse_args()
    if args.command == 'prepare':
        record = prepare()
        print(json.dumps({'deployments': len(record['deployments']), 'matrix': len(record['matrix'])}))
    else:
        record = json.loads((OUT / 'inputs.json').read_text())
        if args.command == 'summarize':
            result = summarize()
            print(json.dumps({'actual_runs': len(result['actual_runs']), 'prepared_scenarios': result['prepared_scenarios']}))
            sys.exit(0)
        if args.command == 'capacity':
            from collections import Counter
            rows = capacity_report(record)
            print(json.dumps(dict(Counter(row['capacity']['status'] for row in rows))))
            sys.exit(0)
        entry = next(row for row in record['matrix'] if row['id'] == args.id)
        if args.command == 'worker':
            worker(entry, OUT / (entry['id'] + '.json'))
        else:
            run(entry)
