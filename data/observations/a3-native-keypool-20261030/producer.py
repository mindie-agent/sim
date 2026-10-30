"""Assign native KeyPool service from retained operator calls and report residuals.

[Role] Separate native-call calibration, operator checks and whole-Body checks.
[Invariants] Never fit Body or layer spans. KeyPool trains on independent short
  context calls; PoolKeyIndexer uses long-context B1/B2 only. Other long-context
  populations were previously inspected and are not pristine holdouts.
[Timeouts & Locks] Stream CSV files; one bounded 45-layer probe at a time.
[Side-Effect Policy] Local explicit parameters, case and evidence only; no NPU.
"""
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import resource
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mindie_sim import predict, target_graphs, operator_costs

HERE = Path(__file__).resolve().parent
CAPTURES = Path('/Volumes/data/mindie-sim-data/a3/heldout_engine_20261030')
SHORT = CAPTURES/'glm5_3_flash_w8a8_static_kernel_on_c10_p4481_20261030/kernel_details.csv'
LONG = CAPTURES/'glm5_3_flash_w8a8_static_kernel_on_c10_p128k_20261030/kernel_details.csv'
BASE = ROOT.parent/'profiler/examples/structure/cases/a3-glm53flash-heldout-tp16.json'
BODY = Path('/Volumes/data/mindie-sim-data/acceptance-archives-20261030/current/182a821a5620a867118b292bd3930907650607a7ab6935d76f9caafd0f9a764f.json')
CASE = ROOT/'examples/cases/longctx-decode-20261030/glm53flash-a3-native-keypool.json'
IMPLEMENTATION = 'vllm_ascend_cann_keypool_caller'


def identity(path):
    digest = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def load(path):
    assert path.stat().st_size < 50 * 1024**2
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def shapes(value):
    return [tuple(int(v) for v in shape.split(',')) if shape else ()
            for shape in value.strip('"').split(';')]


def capture(path):
    groups, cache = {}, None
    with path.open(newline='') as source:
        for line, row in enumerate(csv.DictReader(source), 2):
            native = row['Type']
            if cache is None and native == '_write_keypool_caches_kernel':
                cache = {'line': line, 'input_shapes': row['Input Shapes'],
                         'output_shapes': row['Output Shapes']}
            if native not in ('KeyPool', 'PoolKeyIndexer') or row['OP State'] != 'static':
                continue
            inputs, outputs = shapes(row['Input Shapes']), shapes(row['Output Shapes'])
            if native == 'KeyPool':
                tokens, hidden = inputs[0]
                if (hidden != 4096 or inputs[1:4] != [(128, 4096), (128, 4096), (4, 128)]
                        or row['Input Data Types'].split(';')[:3] != ['DT_BF16'] * 3):
                    continue
                batch = (inputs[4][0] - 1) // 2
                if tokens != 6 * batch or outputs[0] != (tokens // 4 + batch, 128):
                    continue
            else:
                tokens = inputs[0][0] - 1
                batch = inputs[3][0] - 1
                if (inputs[0][1:] != (32, 128) or tokens != 6 * batch
                        or row['Input Data Types'].split(';')[:3] != ['DT_BF16'] * 3
                        or outputs[0] != (tokens + 1, 2051)):
                    continue
            group = groups.setdefault((native, batch), {'native': native, 'batch': batch,
                'tokens': tokens, 'count': 0, 'sum_us': 0., 'min_us': float('inf'), 'max_us': 0.,
                'first_line': line, 'last_line': line, 'input_shapes': row['Input Shapes'],
                'output_shapes': row['Output Shapes'], 'input_dtypes': row['Input Data Types'],
                'block_num': row['Block Num'], 'mix_block_num': row['Mix Block Num']})
            duration = float(row['Duration(us)'])
            group['count'] += 1
            group['sum_us'] += duration
            group['min_us'] = min(group['min_us'], duration)
            group['max_us'] = max(group['max_us'], duration)
            group['last_line'] = line
    for group in groups.values():
        group['mean_us'] = group['sum_us'] / group['count']
    return groups, cache


def affine(groups):
    """Equal geometry weighting; constrained nonnegative slope and intercept."""
    points = [(g['tokens'], g['mean_us']) for g in groups]
    xbar, ybar = (statistics.mean(p[i] for p in points) for i in (0, 1))
    slope = max(0., sum((x-xbar)*(y-ybar) for x, y in points)
                / sum((x-xbar)**2 for x, _ in points))
    intercept = ybar - slope * xbar
    assert intercept >= 0
    return intercept, slope


def parameters(short, long):
    rules, assignments = [], []
    for native, recipe, groups, training in (
            ('KeyPool', 'cann_key_pool', [g for (n, _), g in short.items() if n == 'KeyPool'],
             'independent_short_context_same_BF16_WK_gate_state_ABI'),
            ('PoolKeyIndexer', 'cann_pool_key_indexer', [long[('PoolKeyIndexer', b)] for b in (1, 2)],
             'long_context_B1_B2_reassigned_from_previously_inspected_capture')):
        intercept, slope = affine(groups)
        provenance = {'scope': 'native_kernel_response_not_body_fit', 'training': training,
            'assignment_batches': sorted(g['batch'] for g in groups),
            'fitting': 'equal_geometry_means_nonnegative_affine_in_input_query_rows',
            'validation_batches': [3, 4, 7, 8], 'validation_history': 'previously_examined_not_pristine_holdout',
            'source': identity(SHORT if native == 'KeyPool' else LONG),
            'source_kernel_binary_identity': 'not_established',
            'reference_schedule': 'logical_envelope_replaced_only_at_this_native_boundary',
            'extrapolation': ('long_context_has_same_producer_ABI' if native == 'KeyPool'
                             else 'B3_to_B8_outside_B1_B2_training_geometry; score_reuse_unidentified')}
        match = {'hardware': 'Ascend910_9362', 'recipe': recipe, 'dtype': 'bf16',
                 'metadata.implementation': IMPLEMENTATION, 'metadata.pool_size': 4}
        coordinates = {'rows': {'path': 'metadata.padded_query_rows'}}
        domain = {'rows': {'min': 6, 'max': 60 if native == 'KeyPool' else 48}}
        if native == 'KeyPool':
            match.update({'local_shape.1': 128, 'local_shape.2': 4096,
                          'metadata.tail_capacity': 9})
        else:
            match.update({'local_shape.1': 32, 'local_shape.2': 128, 'metadata.topk_tokens': 2048})
            coordinates['scanned_pools'] = {'path': 'metadata.scanned_pools'}
            domain['scanned_pools'] = {'min': 196804, 'max': 1574432}
        rules.append({'id': 'a3-native-' + recipe + '-operator-calibration', 'match': match,
            'coordinates': coordinates, 'domain': domain,
            'response': {'fixed_us': intercept, 'terms': [{'coordinate': 'rows', 'us_per_unit': slope}]},
            'execution': {'kind': 'joint_native', 'resource': 'observed_a3_local_native',
                          'reference_rate': 1e9, 'native_kernel': native}, 'provenance': provenance})
        assignments.append({'native': native, 'fixed_us': intercept, 'us_per_query_row': slope,
                            'training_groups': groups})
    return rules, assignments


def timeout(signum, frame):
    raise TimeoutError('native_keypool_body_probe_exceeded_90_seconds')


def main():
    short, _ = capture(SHORT)
    long, cache = capture(LONG)
    rules, assignments = parameters(short, long)
    packet = {'schema': 'mindie-sim.native-keypool-service/1', 'rules': rules,
              'assignments': assignments, 'long_context_operator_groups': list(long.values()),
              'tail_capacity_evidence': cache, 'sources': [identity(SHORT), identity(LONG)],
              'new_parameters': 4, 'independent_validation_claim': False}
    save(HERE/'parameters.json', packet)
    raw = load(BASE)
    raw['name'] = 'glm53flash-a3-tp16-q6-native-keypool-operator-service'
    kernels = raw['runtime']['kernels']
    kernels.setdefault('lowerings', {})['pool_key_indexer'] = {'implementation': IMPLEMENTATION,
        'options': {'tail_capacity': 9, 'block_table_columns': 8195, 'score_mode': 'linear_weighted_query'}}
    kernels.setdefault('service_regimes', []).extend(rules)
    raw['runtime']['graph_buckets'] = list(range(6, 61, 6))
    raw['runtime']['piecewise_capture_keys'] = list(range(6, 61, 6))
    raw['runtime']['metadata'] = {'input_boundary': 'full_45_layer_target_body_only',
        'context_basis': '131201_initial_prompt_scenario; actual_per_call_context_unobserved',
        'calibration_origin': 'explicit_operator_calibration_not_pristine_holdout',
        'native_keypool_parameters': str((HERE/'parameters.json').relative_to(ROOT)),
        'base_declaration': identity(BASE), 'retained_base_parameters': 'unchanged',
        'capture_rows': {'observed_long_context': [6, 12, 18, 24, 42, 48],
                         'B5_B6': 'prediction_only_short_capture_bucket_geometry'},
        'tail_capacity': {'value': 9, 'source': identity(LONG), **cache},
        'limitations': ['CANN source binary and internal score reuse are unbound',
                       'Sparse attention valid indices and native owner schedule remain unbound',
                       'No Body or per-layer timing entered parameter assignment']}
    save(CASE, raw)
    retained = load(BODY)
    observed = {r['geometry']['sequences_per_replica']: r
                for r in [retained, *retained['other_geometry_populations']]}
    report = {'scope': 'full_target_body_and_topology_aligned_layer_spans; not_serving',
        'case': identity(CASE), 'retained_body': identity(BODY), 'model_layers': 45,
        'calibration': packet, 'rows': [], 'operator_validation': [],
        'validation_history': 'all_observed_Body_and_long_operator_populations_previously_examined',
        'operator_threshold': .15, 'body_threshold': .15}
    signal.signal(signal.SIGALRM, timeout)
    for batch in range(1, 9):
        start = time.perf_counter()
        signal.alarm(90)
        result = predict(raw, scope='body', phase='decode', tokens=6, context=131201,
                         sequences=batch, event_limit=0)
        signal.alarm(0)
        actual = observed.get(batch)
        row = {'batch': batch, 'prediction_ms': result['body_forward_ms'],
               'observed_mean_ms': actual['comparison']['observed_ms']['mean'] if actual else None,
               'evaluation_role': 'operator_training_Body_check' if batch < 3 else
                                  ('previously_inspected_validation' if actual else 'prediction_only'),
               'geometry': result['geometry'], 'layers': [], 'elapsed_seconds': time.perf_counter()-start,
               'live_tasks': result['execution']['live_tasks'],
               'live_programs': result['execution']['live_programs']}
        if actual:
            row['relative_error'] = row['prediction_ms']/row['observed_mean_ms']-1
            row['passed'] = abs(row['relative_error']) <= .15
        layer_observed = {r['layer_id']: r for r in actual['per_layer_comparison']['layers']} if actual else {}
        for layer in range(45):
            timeline = [r for r in result['timeline'] if r['layer'] == layer and r['replica'] == 0]
            duration = max(r['complete_ms'] for r in timeline)-min(r['ready_ms'] for r in timeline)
            item = {'layer_id': layer, 'predicted_ms': duration}
            if layer in layer_observed:
                item['observed_ms'] = layer_observed[layer]['observed_ms']
                item['relative_error'] = duration/item['observed_ms']['mean']-1
                item['passed'] = abs(item['relative_error']) <= .15
            row['layers'].append(item)
        report['rows'].append(row)
        tiny = deepcopy(raw)
        tiny['model']['layers'] = 4
        ops = [op for op in target_graphs(tiny, scope='body', phase='decode', tokens=6,
                context=131201, sequences=batch)[0]
               if op.layer == 3 and op.recipe in ('cann_key_pool', 'cann_pool_key_indexer')]
        costs = operator_costs(tiny, ops)['operators']
        for op, cost in zip(ops, costs):
            native = dict(op.metadata)['kernel']
            captured = long.get((native, batch))
            check = {'batch': batch, 'native': native, 'shape': op.local_shape,
                     'predicted_us': cost['duration_ms']*1000, 'program': cost['program'],
                     'observed_us': captured['mean_us'] if captured else None}
            if captured:
                check['relative_error'] = check['predicted_us']/captured['mean_us']-1
                check['passed'] = abs(check['relative_error']) <= .15
            report['operator_validation'].append(check)
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(HERE/'validation.json', report)
        print(json.dumps({k:v for k,v in row.items() if k not in ('layers','geometry')}), flush=True)
    report['summary'] = {'observed_body_passes': sum(r.get('passed', False) for r in report['rows']),
        'observed_body_count': len(observed),
        'native_operator_passes': sum(r.get('passed', False) for r in report['operator_validation']),
        'native_operator_count': sum(r['observed_us'] is not None for r in report['operator_validation'])}
    save(HERE/'validation.json', report)
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
