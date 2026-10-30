"""Retrospective, source-conditional Kimi TP communication service evidence.

[Role] Freeze source assignments, stream native HCOM calls, fit training only.
[Invariants] One HCOM envelope is one observation; CPU controllers are not
  added. Payload is a source caller declaration, not an observed HCOM argument.
  Neither Body error nor context/concurrency identity is a response coordinate.
[Timeouts & Locks] One source at a time, 25 seconds per parser; bounded samples.
[Side-Effect Policy] Read retained local captures; write requested evidence only.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROFILER = Path('/Users/maoxx241/code/profiler')
sys.path.insert(0, str(PROFILER))
from mindie_profiler.normalized import kernel_rows
from mindie_profiler.decode_segmentation import is_aicpu_controller

TYPES = {'hcom_allGather_': 'allgather', 'hcom_reduceScatter_': 'reduce_scatter'}
COUNTS = {'allgather': 9, 'reduce_scatter': 8}
WIDTH, DTYPE_BYTES = 7168, 2


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = HERE / name
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def source_blob(repository, revision, path, symbols):
    data = subprocess.check_output(['git', 'show', f'{revision}:{path}'],
                                   cwd=repository, timeout=5)
    return {'repository': repository, 'revision': revision, 'path': path,
            'sha256': hashlib.sha256(data).hexdigest(), 'symbols': symbols,
            'captured_binary_identity': 'not_established'}


def freeze():
    census_path = ROOT / 'data/verification/kimi-mla-cache-domain-20261030.json'
    census = read(census_path)['native_census']['cases']
    report_path = PROFILER / 'examples/structure/results/decode.json'
    current = {r['case_id']:r for r in read(report_path)['cases']}
    sources = []
    for case in census:
        tp, batch = case['tp'], case['client_concurrency']
        context = case['context_evidence']['lower_bound']
        capacities = sorted({g['physical_batch'] for g in case['native_abi_groups']})
        score = read(current[case['case_id']]['evidence']['path'])
        assert score['source_identity']['sha256'] == case['source']['sha256']
        extraction = score['extraction_identity']
        sources.append({k: case[k] for k in ('case_id', 'source')} | {
            'extraction': identity(extraction['path']),
            'extraction_selection': 'current source-identical Profiler extraction, including physical graph capacity and unpadded head alignment',
            'tp': tp, 'client_concurrency': batch, 'context_lower_bound': context,
            'native_attention_query_capacities': capacities,
            'assignment': 'train' if context == 4096 and batch in
                ((2, 24, 48) if tp == 8 else (2, 12, 48)) else 'validation',
            'original_assignment': 'unassigned', 'evaluation_relation': 'retrospective',
        })
    assert Counter(s['assignment'] for s in sources) == {'train': 6, 'validation': 40}
    ascend = '/Users/maoxx241/code/vllm-ascend'
    rev = '8c30e4f491f73cba737a1dacc522894e437cdf3d'
    source = [source_blob(ascend, rev, 'vllm_ascend/models/kimi_k3.py', [
        'AscendKimiLinearModel.__init__/forward', 'AscendKimiDecoderLayer.forward_attn_residual',
        'AscendKimiMLP.forward']), source_blob(ascend, rev,
        'vllm_ascend/ops/fused_moe/shared_experts.py',
        ['SharedExperts._gather_sp_input', 'SharedExperts._pad_and_reduce_scatter']),
        source_blob('/Users/maoxx241/code/vllm-v0.27.1-k3-ops-validation',
        '6e448d0ea9bf3d88d898b65449ca6dc2aec170ac',
        'vllm/models/common/ops/sequence_parallel.py',
        ['sp_shard', 'sp_all_gather', 'sp_reduce_scatter'])]
    cases = [identity(PROFILER / f'examples/structure/cases/a3-kimi4-tp{tp}-longctx-source.json')
             for tp in (8, 16)]
    return write('inputs.json', {
        'schema': 'kimi-hccl-retrospective-source-assignment/1',
        'status': 'frozen_before_new_HCOM_duration_extraction',
        'source_inventory': identity(census_path), 'extraction_report_identity': identity(report_path),
        'canonical_cases': cases,
        'assignment_amendment': {
            'original_training': '4K TP8/TP16 each B2/B24/B48',
            'replacement': 'TP16 B24 excluded; TP16 B12 replaces it before B12 HCOM duration extraction',
            'reason': 'TP16 B24 extraction has overlapping forwards and includes lm_head AG in 78/177 body windows; independent Profiler repair pending',
            'causal_counterexample': {'csv_line': 28155, 'name': 'hcom_reduceScatter__503_14_103',
                'start_ns': 1793353592288758635, 'end_ns': 1793353592297256625,
                'dependent_attnres_start_ns': 1793353592288789296,
                'next_forward_start_ns': 1793353592293357847},
            'timing_threshold_used_for_exclusion': False,
            'replacement_source_duration_seen_before_selection': False},
        'caller_source': source, 'sources': sources,
        'physical_contract': {'dtype': 'bf16', 'hidden_width': WIDTH,
            'query_rows': 'native graph storage capacity; inactive rows remain allocated',
            'local_rows': 'ceil(native_query_capacity / tp)',
            'allgather_input_bytes': 'local_rows * hidden_width * 2',
            'reduce_scatter_input_bytes': 'tp * local_rows * hidden_width * 2',
            'calls_per_body': COUNTS,
            'call_structure': 'four attention AG/RS + dense FFN AG/RS + three shared FFN AG/RS + final AG',
            'topology': 'two fixed single-node TP groups, 8 or 16 physical ranks; no unseen topology transfer',
            'payload_basis': 'source-conditional; raw HCOM input/output descriptors are N/A'},
        'training_policy': 'per-collective, per-fixed-rank continuous payload interpolation; equal source weighting; no call ordinal, context or batch selector',
        'validation_policy': 'all remaining sources; retain unsupported payloads and all errors without refitting',
        'limits': ['Historical sources were unassigned; this retrospective split is not blind prospective validation.',
                   'HCOM duration includes the observed collective boundary and peer/control effects, not an isolated physical link rate.',
                   'AI_CPU controllers overlap HCOM and are not added. No Body duration is a fit target.',
                   'No runtime joint-collective execution boundary is installed by this evidence producer.'],
    })


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0}
    return {'count': len(ordered), 'min': ordered[0], 'median': statistics.median(ordered),
            'mean': statistics.mean(ordered), 'p90': ordered[math.ceil(.9*len(ordered))-1],
            'max': ordered[-1]}


def brief(row):
    return {'csv_line': row.line, 'type': row.kind, 'name': row.name,
            'core': row.core, 'state': row.state, 'blocks': row.blocks,
            'input_shapes': row.input_shapes, 'output_shapes': row.output_shapes,
            'input_dtypes': row.input_dtypes, 'output_dtypes': row.output_dtypes,
            'input_formats': row.input_formats, 'output_formats': row.output_formats}


def collect(case):
    started = time.monotonic()
    actual = identity(case['source']['path'])
    assert actual['sha256'] == case['source']['sha256'], case['case_id']
    assert identity(case['extraction']['path'])['sha256'] == case['extraction']['sha256']
    extraction = read(case['extraction']['path'])
    forwards = extraction['forwards']
    starts = [f['start_ns'] for f in forwards]
    calls = [defaultdict(list) for _ in forwards]
    first_head = []
    excluded = Counter()
    signatures = {}
    for row in kernel_rows(case['source']['path'], timeout_s=25):
        i = bisect.bisect_right(starts, row.start_ns)-1
        if (i < 0 or row.start_ns >= forwards[i]['end_ns']
                or row.event.device_id != forwards[i]['device_id']):
            continue
        if is_aicpu_controller(row.event):
            excluded[row.kind] += 1
            continue
        if i == 0:
            first_head.append(brief(row))
        collective = TYPES.get(row.kind)
        if collective is None:
            continue
        calls[i][collective].append((row.start_ns, row.line, (row.end_ns-row.start_ns)/1000))
        key = (collective, row.signature[1:])
        entry = signatures.setdefault(key, brief(row) | {'count': 0})
        entry['count'] += 1
        entry['last_csv_line'] = row.line
    capacities = sorted({f['query_rows'] for f in forwards})
    assert capacities == case['native_attention_query_capacities'], (case['case_id'], capacities)
    assert len(capacities) == 1, (case['case_id'], 'mixed_capacity_requires_separate_domain')
    local_rows = math.ceil(capacities[0]/case['tp'])
    groups = []
    structure_errors = []
    for collective, expected in COUNTS.items():
        counts = Counter(len(f[collective]) for f in calls)
        if counts != {expected: len(forwards)}:
            structure_errors.append({'collective': collective, 'expected': expected,
                'observed_count_histogram': dict(counts)})
        per_ordinal = [[] for _ in range(max(counts))]
        for forward in calls:
            for ordinal, (_, _, elapsed) in enumerate(sorted(forward[collective])):
                per_ordinal[ordinal].append(elapsed)
        samples = [v for ordinal in per_ordinal for v in ordinal]
        groups.append({'collective': collective,
            'declared_input_bytes': local_rows*WIDTH*DTYPE_BYTES*(case['tp'] if collective == 'reduce_scatter' else 1),
            'local_rows': local_rows, 'native_query_capacity': capacities[0],
            'expected_calls_per_forward': expected, 'observed_call_count_histogram': dict(counts),
            'native_envelope_us': stats(samples),
            'ordinal_diagnostics_only': [dict(ordinal=i, native_envelope_us=stats(v))
                                         for i, v in enumerate(per_ordinal)],
            'signatures': [s for (kind, _), s in signatures.items() if kind == collective]})
    neighbors = []
    for i, row in enumerate(first_head):
        if row['type'] in TYPES:
            neighbors.append({'collective': TYPES[row['type']], 'event': row,
                              'before': first_head[max(0, i-2):i], 'after': first_head[i+1:i+3]})
    return {**case, 'forward_count': len(forwards), 'groups': groups,
            'status': 'invalid_extraction_boundary' if structure_errors else 'complete_source_call_structure',
            'structure_errors': structure_errors,
            'excluded_controller_counts': dict(excluded), 'first_forward_tensor_neighbors': neighbors,
            'wall_seconds': time.monotonic()-started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def extract(assignment):
    manifest = read(HERE/'inputs.json')
    rows = []
    for case in manifest['sources']:
        if case['assignment'] != assignment:
            continue
        result = collect(case)
        rows.append(result)
        write(f'{assignment}-native.json', {'schema': 'kimi-hccl-native-ledger/1',
            'assignment_identity': identity(HERE/'inputs.json'), 'status': 'completed' if
            len(rows) == sum(s['assignment'] == assignment for s in manifest['sources']) else 'in_progress',
            'cases': rows})
        print(json.dumps({'case': case['case_id'], 'forwards': result['forward_count'],
                          'seconds': result['wall_seconds'], 'peak_rss': result['peak_rss_bytes']}), flush=True)


def fit():
    sys.path.insert(0, str(ROOT))
    from mindie_sim.prediction.structure import parse_case
    training = read(HERE/'train-native.json')
    assert training['status'] == 'completed'
    groups = defaultdict(list)
    for case in training['cases']:
        assert case['status'] == 'complete_source_call_structure', case['case_id']
        for group in case['groups']:
            groups[case['tp'], group['collective']].append({
                'payload_bytes': group['declared_input_bytes'],
                'mean_us': group['native_envelope_us']['mean'],
                'case_id': case['case_id'], 'source_sha256': case['source']['sha256'],
                'ordinal_means_us': [o['native_envelope_us']['mean'] for o in group['ordinal_diagnostics_only']]})
    rules = []
    domains = {}
    for tp in (8, 16):
        canonical = PROFILER / f'examples/structure/cases/a3-kimi4-tp{tp}-longctx-source.json'
        case = parse_case(read(canonical))
        machine = case.machine.to_dict()
        domains[str(tp)] = {
            'canonical_input': identity(canonical), 'resolved_machine': machine,
            'resolved_machine_sha256': hashlib.sha256(json.dumps(machine,sort_keys=True).encode()).hexdigest(),
            'runtime_declaration': {'runtime': case.runtime.vllm_impl,
                'revision': case.runtime.vllm_version, 'format': case.runtime.kernel_format},
            'identity_strength': 'source-conditional; empty software identity fields are unknown, not captured versions',
            'guard_policy': 'same resolved physical rates, service modes, compute supply, rank count and topology; do not transfer to changed link resources'}
    for (tp, collective), points in sorted(groups.items()):
        points.sort(key=lambda p:p['payload_bytes'])
        pooled = defaultdict(list)
        for point in points:
            pooled[point['payload_bytes']].append(point['mean_us'])
        knots = [[x,statistics.mean(y)] for x,y in sorted(pooled.items())]
        assert len(knots) >= 2
        rules.append({'tp': tp, 'collective': collective,
            'physical_domain': str(tp),
            'response': {'kind': 'continuous_piecewise_linear', 'coordinate': 'declared_input_bytes',
                         'points': knots},
            'domain': {'min': knots[0][0], 'max': knots[-1][0]},
            'extrapolation_allowed': False, 'training': points,
            'monotone_training_means': all(a[1] <= b[1] for a,b in zip(knots,knots[1:])),
            'evaluation_relation': 'retrospective_training',
            'parameter_meaning': 'inclusive native collective envelope, including observed peer/control boundary',
            'pure_link_bandwidth_or_RTT': False})
    return write('rules.json', {'schema': 'kimi-hccl-source-conditional-response/1',
        'training_identity': identity(HERE/'train-native.json'),
        'assignment_identity': identity(HERE/'inputs.json'), 'physical_domains': domains, 'rules': rules,
        'runtime_admission': 'candidate_only; use existing single native-group inclusive response after physical/topology guard verification',
        'heldout_used_for_fit': False})


def predict(rule, value):
    points = rule['response']['points']
    if not points[0][0] <= value <= points[-1][0]:
        return None
    for a, b in zip(points, points[1:]):
        if a[0] <= value <= b[0]:
            return a[1]+(value-a[0])/(b[0]-a[0])*(b[1]-a[1])
    raise ValueError('response_domain_gap')


def validate():
    packet = read(HERE/'rules.json')
    validation = read(HERE/'validation-native.json')
    assert validation['status'] == 'completed'
    rules = {(r['tp'],r['collective']):r for r in packet['rules']}
    results = []
    for case in validation['cases']:
        for group in case['groups']:
            rule = rules[case['tp'],group['collective']]
            value = predict(rule,group['declared_input_bytes']) if not case['structure_errors'] else None
            observed = group['native_envelope_us']['mean']
            error = None if value is None else value/observed-1
            results.append({'case_id':case['case_id'],'tp':case['tp'],
                'collective':group['collective'],'input_bytes':group['declared_input_bytes'],
                'predicted_us':value,'observed_mean_us':observed,'relative_error':error,
                'status':'invalid_extraction_boundary' if case['structure_errors'] else
                    'unsupported_payload' if error is None else 'pass' if abs(error)<=.15 else 'fail',
                'ordinal_relative_errors': [] if value is None else [value/o['native_envelope_us']['mean']-1
                    for o in group['ordinal_diagnostics_only']]})
    return write('validation.json',{'schema':'kimi-hccl-continuous-validation/1',
        'rules_identity':identity(HERE/'rules.json'),'native_identity':identity(HERE/'validation-native.json'),
        'summary':dict(Counter(r['status'] for r in results)),'results':results,
        'boundary':'single HCOM envelope; not whole Body acceptance',
        'refit_after_validation':False})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','train','fit','validation','score'))
    mode=parser.parse_args().mode
    if mode=='freeze':freeze()
    elif mode in ('train','validation'):extract(mode)
    elif mode=='fit':fit()
    else:validate()
