"""Inspect same-ABI short-context native GMM responses before long-context checks.

[Role] Calibrate only complete native expert calls from independent 4K capture.
[Invariants] Capacity is not valid rows; actual group-list/router values remain
  unknown. The probability-moment reference stays unchanged. No Body fitting.
[Timeouts & Locks] Stream one capture at a time with a 90-second scan budget.
[Side-Effect Policy] Local parameters and diagnostic reports only; no NPU work.
"""
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PROFILER = ROOT.parent/'profiler'
sys.path[:0] = [str(ROOT), str(PROFILER), str(PROFILER/'examples/structure')]
from mindie_sim import predict, target_graphs, operator_costs
from mindie_sim.prediction.structure import parse_case
from mindie_profiler.decode_segmentation import auto_segment_decode_forwards
from mindie_profiler.normalized import kernel_rows, row_from_event
from mindie_profiler.events import SourceRef
from operators import segmentation_config

spec = importlib.util.spec_from_file_location('keypool_parameters', Path(__file__).with_name('producer.py'))
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
HERE = Path(__file__).resolve().parent
EXTRACTION = Path('/Volumes/data/mindie-sim-data/acceptance-archives-20261030/pending/long-context-decode/eb9a5755fea96e96546e93270119f5e194971f5bab79972d1ebff0a5eab87e85.json')
KINDS = ('GroupedMatmulSwigluQuant', 'GroupedMatmul')
OUTPUT_CASE = ROOT/'examples/cases/longctx-decode-20261030/glm53flash-a3-native-keypool-gmm-service.json'


def scan(path, config):
    groups, forwards = {}, defaultdict(list)
    source_id = producer.identity(path)
    source = SourceRef('rank_0', 0, 'csv', str(path), path.name, source_id['sha256'], path.stat().st_size,
                       'kernel_details_csv_46col', 'retained_full_capture', mtime_ns=path.stat().st_mtime_ns)

    def consume(forward, events):
        geometry = forward['geometry']
        if geometry['tokens_per_sequence'] != 6:
            return
        batch = geometry['sequences_per_replica']
        forwards[batch].append(forward['forward_id'])
        for event in events:
            if (event.op_type not in KINDS or not forward['start_ns'] <= event.start_ns < event.end_ns <= forward['end_ns']):
                continue
            row = row_from_event(event)
            signature = (row.input_shapes, row.input_dtypes, row.input_formats,
                         row.output_shapes, row.output_dtypes, row.blocks)
            key = (batch, event.op_type, signature)
            group = groups.setdefault(key, {'batch': batch, 'query_rows': geometry['query_rows'],
                'expected_local_rows': geometry['query_rows'] * 8 / 16, 'native': event.op_type,
                'input_shapes': row.input_shapes, 'input_dtypes': row.input_dtypes,
                'input_formats': row.input_formats, 'output_shapes': row.output_shapes,
                'output_dtypes': row.output_dtypes, 'block_num': row.blocks,
                'count': 0, 'sum_us': 0., 'min_us': float('inf'), 'max_us': 0.,
                'pmu_sums': {}, 'pmu_counts': {}, 'first_locator': event.locator})
            duration = event.duration_ns/1000
            group['count'] += 1
            group['sum_us'] += duration
            group['min_us'], group['max_us'] = min(group['min_us'], duration), max(group['max_us'], duration)
            for name, value in row.counters_us.items():
                group['pmu_sums'][name] = group['pmu_sums'].get(name, 0) + value
                group['pmu_counts'][name] = group['pmu_counts'].get(name, 0) + 1

    result = auto_segment_decode_forwards(kernel_rows(path, rank_id='rank_0', rank_index=0,
        max_rows=5_000_000, timeout_s=90), config, source=source, on_forward=consume,
        max_detailed_forwards=0, max_samples_per_layer=0)
    for group in groups.values():
        group['mean_us'] = group['sum_us']/group['count']
        group['forward_count'] = len(forwards[group['batch']])
        group['calls_per_forward'] = group['count']/group['forward_count']
        group['pmu_means_us'] = {k: v/group['pmu_counts'][k] for k, v in group['pmu_sums'].items()}
    return {'source': source_id, 'groups': list(groups.values()), 'forward_ids': dict(forwards),
            'scan_counts': result['scan_counts']}


def main():
    start = time.monotonic()
    raw = producer.load(producer.CASE)
    case = parse_case(raw)
    config = segmentation_config({**case.model.to_dict(), 'tp': case.runtime.parallel.tp}, producer.load(EXTRACTION))
    report = {'scope': 'same_native_ABI_operator_calibration_inventory',
              'routing_distribution': 'unknown; no independence_or_forceEPLB_claim',
              'model_reference': {'experts': 288, 'local_expert_banks': 18, 'topk': 8, 'tp': 16, 'ep': 16,
                                  'q': 6, 'capacity_is_valid_rows': False}, 'sources': []}
    for path in (producer.SHORT, producer.LONG):
        result = scan(path, config)
        report['sources'].append(result)
        producer.save(HERE/'gmm-inventory.json', report)
        print(path.parent.name)
        for group in result['groups']:
            if group['input_shapes'][0][0] == 384:
                print(json.dumps({k: group[k] for k in ('batch','native','expected_local_rows','mean_us','min_us','max_us','count','calls_per_forward','pmu_means_us')}), flush=True)
    report['elapsed_seconds'] = time.monotonic()-start
    producer.save(HERE/'gmm-inventory.json', report)
    fit_and_check(report)


def fit_and_check(inventory):
    """All parameters use short-context native calls, never Body observations."""
    short, long = inventory['sources']
    assignments, rules = [], []
    signature_keys = ('input_shapes', 'input_dtypes', 'input_formats', 'output_shapes',
                      'output_dtypes', 'block_num')
    for native in KINDS:
        training = [g for g in short['groups'] if g['native'] == native and g['input_shapes'][0][0] == 384]
        validation = [g for g in long['groups'] if g['native'] == native and g['input_shapes'][0][0] == 384]
        assert sorted(g['batch'] for g in training) == [6, 7]
        assert sorted(g['batch'] for g in validation) == [7, 8]
        abi = {k: training[0][k] for k in signature_keys}
        assert all({k: g[k] for k in signature_keys} == abi for g in training + validation)
        assert all(g['calls_per_forward'] == 42 for g in training + validation)
        a, b = sorted(training, key=lambda g: g['expected_local_rows'])
        slope = (b['mean_us'] - a['mean_us'])/(b['expected_local_rows'] - a['expected_local_rows'])
        intercept = a['mean_us'] - slope * a['expected_local_rows']
        assert slope >= 0 and intercept >= 0
        fused = native == 'GroupedMatmulSwigluQuant'
        if fused:
            match = {'recipe': 'declared_native_fusion', 'metadata.fusion_role': 'grouped_gate_up_activation_quant',
                'metadata.fusion_tensor_view.input_width': 4096,
                'metadata.fusion_tensor_view.intermediate_width': 2048,
                'metadata.fusion_tensor_view.local_experts': 18,
                'metadata.fusion_tensor_view.routing_basis': 'probability_moments',
                'metadata.fusion_tensor_view.input_dtype': 'int8',
                'metadata.fusion_tensor_view.output_dtype': 'int8',
                'metadata.fusion_tensor_view.weight_dtype': 'int8',
                'metadata.fusion_tensor_view.input_scale_dtype': 'fp32',
                'metadata.fusion_tensor_view.output_scale_dtype': 'fp32'}
            axis = 'metadata.fusion_phases.0.metadata.expected_rows'
        else:
            match = {'recipe': 'grouped_down', 'dtype': 'int8', 'local_shape.1': 4096, 'local_shape.2': 2048,
                     'metadata.matrix_weight_bank_groups': 18, 'metadata.routing_basis': 'probability_moments',
                     'metadata.activation_format': 'int8', 'metadata.weight_format': 'int8',
                     'metadata.output_format': 'bf16'}
            axis = 'metadata.expected_rows'
        rule = {'id': 'a3-q6-cap384-shortctx-' + native, 'match': {'hardware': 'Ascend910_9362', **match},
            'coordinates': {'expected_local_rows': {'path': axis}},
            'domain': {'expected_local_rows': {'min': 18, 'max': 24.000000001}},
            'response': {'fixed_us': intercept, 'terms': [{'coordinate': 'expected_local_rows', 'us_per_unit': slope}]},
            'execution': {'kind': 'joint_native', 'resource': 'observed_a3_local_native',
                          'reference_rate': 1e9, 'native_kernel': native},
            'provenance': {'basis': 'independent_short_context_complete_native_call_affine',
                'source': short['source'], 'training_batches': [6, 7], 'training_query_rows': [36, 42],
                'training_expected_local_rows': [18, 21], 'same_native_ABI': abi,
                'caller_domain': {'q': 6, 'tp': 16, 'ep': 16, 'input_capacity': 384,
                                  'local_expert_banks': 18, 'actual_group_list': 'unobserved'},
                'conditional_transfer': 'routing_distribution_not_observed; not_independent_routing_generalization',
                'reference_work': 'original_probability_moments_unchanged_and_unexecuted_at_native_boundary',
                'source_schedule': 'not_reconstructed', 'binary_identity': 'not_established',
                'body_timing_used': False, 'validation_history': 'previously_inspected_long_context',
                'extrapolation': 'expected_rows24_from18_21; tested_and_not_assumed_accurate'}}
        rules.append(rule)
        assignments.append({'native': native, 'fixed_us': intercept, 'us_per_expected_local_row': slope,
            'training': [{k:g[k] for k in ('batch','expected_local_rows','count','mean_us','pmu_means_us')} for g in training],
            'validation': [{**{k:g[k] for k in ('batch','expected_local_rows','count','mean_us','pmu_means_us')},
                            'prediction_us': intercept+slope*g['expected_local_rows'],
                            'relative_error': (intercept+slope*g['expected_local_rows'])/g['mean_us']-1,
                            'passed_15percent': abs((intercept+slope*g['expected_local_rows'])/g['mean_us']-1)<=.15}
                           for g in validation]})
    parameters = {'scope': 'conditional_native_GMM_service; no_routing_or_Body_fit',
        'new_parameters': 4, 'rules': rules, 'assignments': assignments,
        'reference_routing_changed': False, 'recorded_tensor_ABI_compatible': True,
        'B8_operator_transfer': 'failed_15percent_for_both_kernels'}
    producer.save(HERE/'gmm-parameters.json', parameters)
    original = producer.load(producer.CASE)
    raw = deepcopy(original)
    raw['name'] += ':shortctx-native-gmm-service'
    raw['runtime']['kernels']['service_regimes'].extend(rules)
    raw['runtime']['metadata']['native_gmm_service'] = {
        'parameters': str((HERE/'gmm-parameters.json').relative_to(ROOT)),
        'scope': 'q6_TP16_EP16_MC2_capacity384_scenario',
        'actual_routing': 'unknown', 'reference_routing': 'unchanged_probability_moments',
        'native_operator_B8_validation': 'fails_15percent; do_not_infer_validity_from_Body_error'}
    producer.save(OUTPUT_CASE, raw)
    retained = producer.load(producer.BODY)
    observed = {r['geometry']['sequences_per_replica']: r
                for r in [retained, *retained['other_geometry_populations']]}
    report = {'scope': 'full_45_layer_q6_B7_B8; not_serving_or_independent_routing_validation',
              'case': producer.identity(OUTPUT_CASE), 'parameters': producer.identity(HERE/'gmm-parameters.json'),
              'primitive_validation': assignments, 'rows': []}
    signal.signal(signal.SIGALRM, producer.timeout)
    for batch in (7, 8):
        signal.alarm(90)
        before = predict(original, scope='body', phase='decode', tokens=6, context=131201, sequences=batch, event_limit=0)
        signal.alarm(90)
        after = predict(raw, scope='body', phase='decode', tokens=6, context=131201, sequences=batch, event_limit=0)
        signal.alarm(0)
        actual = observed[batch]
        row = {'batch': batch, 'body_before_ms': before['body_forward_ms'], 'body_after_ms': after['body_forward_ms'],
            'observed_mean_ms': actual['comparison']['observed_ms']['mean'],
            'relative_error': after['body_forward_ms']/actual['comparison']['observed_ms']['mean']-1,
            'live_tasks': after['execution']['live_tasks'], 'live_programs': after['execution']['live_programs'],
            'layers': [], 'native_services': []}
        for layer in actual['per_layer_comparison']['layers']:
            records = [r for r in after['timeline'] if r['layer'] == layer['layer_id']]
            ms = max(r['complete_ms'] for r in records)-min(r['ready_ms'] for r in records)
            row['layers'].append({'layer_id': layer['layer_id'], 'predicted_ms': ms,
                'observed_ms': layer['observed_ms'], 'relative_error': ms/layer['observed_ms']['mean']-1})
        tiny = deepcopy(raw)
        tiny['model']['layers'] = 4
        ops = [o for o in target_graphs(tiny, scope='body', phase='decode', tokens=6, context=131201, sequences=batch)[0]
               if o.recipe == 'grouped_down' or dict(o.metadata).get('fusion_role') == 'grouped_gate_up_activation_quant']
        for op, cost in zip(ops, operator_costs(tiny, ops)['operators']):
            assert cost['program'] == 'JointNativeService'
            row['native_services'].append({'recipe': op.recipe, 'duration_us': cost['duration_ms']*1000,
                                           'program': cost['program'], 'reference_hbm_bytes': op.hbm_bytes})
        row['passed_15percent'] = abs(row['relative_error']) <= .15
        report['rows'].append(row)
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        producer.save(HERE/'gmm-validation.json', report)
        print(json.dumps({k:v for k,v in row.items() if k != 'layers'}), flush=True)


if __name__ == '__main__':
    main()
