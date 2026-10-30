"""Compare four source-compatible A5 cases with the existing BF16 store queue.

[Role] Bind one missing GDN caller service and expose remaining native residuals.
[Invariants] No coefficients, hardware rates, routes or Body timings are fitted.
  Shape-less helpers inherit only a conditional source/adjacent-call geometry.
[Timeouts & Locks] Stream each capture within 90 s; each Body has a 90 s budget.
[Side-Effect Policy] Write local case copies and evidence; no capture or NPU work.
"""
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PROFILER = ROOT.parent / 'profiler'
sys.path[:0] = [str(ROOT), str(PROFILER), str(PROFILER / 'examples/structure')]
from mindie_sim import predict, target_graphs, operator_costs
from mindie_sim.prediction.structure import parse_case
from mindie_profiler.decode_segmentation import auto_segment_decode_forwards, is_aicpu_controller
from mindie_profiler.normalized import kernel_rows
from mindie_profiler.events import SourceRef
from operators import segmentation_config, deduplicate_communication

HERE = Path(__file__).resolve().parent
CURRENT = Path('/Volumes/data/mindie-sim-data/acceptance-archives-20261030/current')
CASES = ROOT / 'examples/cases/longctx-decode-20261030/a5-gdn-store'
OBSERVED = {
    ('qwen36', 2): 'a344293723f6dfec2ee160ee6fcf44c3fc5fe2daa30f5dfa99281f739fd2c890.json',
    ('qwen36', 4): '26c548ae00bf23ac5a380a28f2995e2b60c24f041015817bf3220324ff66340b.json',
    ('qwen38', 2): 'a9243cad078eb10c26bdd71252b3f99875d30f80ac62fe397b1dedc2f434b473.json',
    ('qwen38', 4): 'f8c04e380b537b4614197ced0f75935d13b30d0e562b7ae5eefeb8a3f1b3affb.json',
}
BINDING = {'bytes_per_element': 2, 'resource': 'scalar_store_issue', 'from_hardware': True}


def load(path):
    assert path.stat().st_size < 50 * 1024**2
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def identity(path):
    digest = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def timeout(signum, frame):
    raise TimeoutError('a5_gdn_store_Body_exceeded_90_seconds')


def native_groups(raw, retained):
    extraction = load(Path(retained['extraction_identity']['path']))
    case = parse_case(raw)
    config = segmentation_config({**case.model.to_dict(), 'tp': case.runtime.parallel.tp}, extraction)
    source_id = retained['source_identity']
    path = Path(source_id['path'])
    assert identity(path)['sha256'] == source_id['sha256']
    source = SourceRef('rank_0', 0, 'csv', str(path), path.name, source_id['sha256'], path.stat().st_size,
                       source_id['schema_version'], 'retained_full_capture', mtime_ns=path.stat().st_mtime_ns)
    groups, forwards = {}, []

    def consume(forward, events):
        geometry = forward['geometry']
        if (geometry['tokens_per_sequence'], geometry['sequences_per_replica']) != (4, 4):
            return
        forwards.append(forward['forward_id'])
        stages = [s for s in forward['stages'] if s['layer_id'] is not None]
        assert len(stages) == 64
        starts = [s['start_ns'] for s in stages]
        selected, _ = deduplicate_communication([e for e in events
            if forward['start_ns'] <= e.start_ns < e.end_ns <= forward['end_ns']
            and not is_aicpu_controller(e)])
        for event in selected:
            stage = stages[bisect_right(starts, event.start_ns)-1]
            # Preserve observed ABI variants; duration sums are diagnostics.
            key = (stage['attention'], stage['layer_id'] == 0, event.op_type,
                   event.input_shapes, event.input_dtypes, event.output_shapes, event.output_dtypes,
                   event.block_dim, event.core)
            row = groups.setdefault(key, {'attention_kind': stage['attention'], 'first_layer': stage['layer_id'] == 0,
                'native': event.op_type, 'input_shapes': event.input_shapes, 'input_dtypes': event.input_dtypes,
                'output_shapes': event.output_shapes, 'output_dtypes': event.output_dtypes,
                'block_num': event.block_dim, 'core': event.core, 'count': 0, 'sum_us': 0.,
                'min_us': float('inf'), 'max_us': 0., 'first_locator': event.locator})
            duration = event.duration_ns / 1000
            row['count'] += 1
            row['sum_us'] += duration
            row['min_us'], row['max_us'] = min(row['min_us'], duration), max(row['max_us'], duration)

    scan = auto_segment_decode_forwards(kernel_rows(path, rank_id='rank_0', rank_index=0,
        max_rows=5_000_000, timeout_s=90), config, source=source, on_forward=consume,
        max_detailed_forwards=0, max_samples_per_layer=0)
    assert len(forwards) == retained['comparison']['observed_forward_count']
    for row in groups.values():
        row['mean_us'] = row['sum_us'] / row['count']
        row['calls_per_forward'] = row['count'] / len(forwards)
        row['per_forward_sum_us'] = row['sum_us'] / len(forwards)
    return {'source': identity(path), 'forward_ids': forwards, 'groups': list(groups.values()),
            'scan_counts': scan['scan_counts'],
            'boundary': 'same complete target Body; excludes scheduler controllers and duplicate HCOM summaries',
            'aggregation': 'native duration sums may overlap; not additive Body components'}


def prediction(raw, retained):
    signal.alarm(90)
    started = time.monotonic()
    result = predict(raw, scope='body', phase='decode', tokens=4, context=126000,
                     sequences=4, event_limit=0)
    signal.alarm(0)
    layers, groups = [], defaultdict(lambda: {'count': 0, 'sum_ms': 0.})
    for observed in retained['per_layer_comparison']['layers']:
        layer = observed['layer_id']
        timeline = [r for r in result['timeline'] if r['layer'] == layer and r['replica'] == 0]
        ms = max(r['complete_ms'] for r in timeline)-min(r['ready_ms'] for r in timeline)
        layers.append({'layer': layer, 'attention_kind': 'gqa' if layer % 4 == 3 else 'gdn',
            'predicted_ms': ms, 'observed_ms': observed['observed_ms']['mean'],
            'relative_error': ms/observed['observed_ms']['mean']-1})
        for row in timeline:
            key = (layers[-1]['attention_kind'], layer == 0, row['recipe'], row.get('fusion_role'))
            group = groups[key]
            group['count'] += 1
            group['sum_ms'] += row['elapsed_ms']
    tiny = deepcopy(raw)
    tiny['model']['layers'] = 1
    ops = target_graphs(tiny, scope='body', phase='decode', tokens=4, context=126000, sequences=4)[0]
    costs = operator_costs(tiny, ops)['operators']
    timeline = {row['op_id']: row for row in result['timeline'] if row['replica'] == 0}
    hccl_calls = []
    for name, row in timeline.items():
        if row['recipe'] == 'hccl_entry':
            exit_row = timeline[name.removesuffix('.hccl.entry')]
            hccl_calls.append({'name': name.removesuffix('.hccl.entry'), 'layer': row['layer'],
                'duration_us': (exit_row['complete_ms']-row['ready_ms'])*1000})
    return {'body_ms': result['body_forward_ms'],
        'body_relative_error': result['body_forward_ms']/retained['comparison']['observed_ms']['mean']-1,
        'layers': layers, 'first_layer_ms': layers[0]['predicted_ms'],
        'hccl_calls': hccl_calls,
        'groups': [{'attention_kind': k[0], 'first_layer': k[1], 'recipe': k[2], 'fusion_role': k[3], **v}
                   for k, v in groups.items()],
        'first_layer_operators': [{'name': op.op_id, 'recipe': op.recipe, 'family': op.family,
            'local_shape': op.local_shape, 'duration_us': cost['duration_ms']*1000, 'program': cost['program'],
            'beta_store': dict(op.metadata).get('beta_store'), 'reference_hbm_bytes': op.hbm_bytes,
            'communication': dict(op.metadata).get('communication')} for op, cost in zip(ops, costs)],
        'elapsed_seconds': time.monotonic()-started, 'live_tasks': result['execution']['live_tasks'],
        'live_programs': result['execution']['live_programs']}


def main():
    signal.signal(signal.SIGALRM, timeout)
    report = {'scope': 'conditional_existing_scalar_store_primitive_binding; no_target_latency_fit',
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'binding': BINDING, 'new_parameters': 0, 'cases': [],
        'representative_geometry': {'batch': 4, 'query_tokens': 4, 'context_tokens': 126000},
        'selection_basis': 'same available B4/q4 geometry across all four deployments, before comparison',
        'limitations': ['captured gating ABI often N/A; conditional source order, not binary identity',
                        'hardware SKU is retained declaration, not measured from delivered deployment',
                        'exact per-call KV increments and rank arrivals remain unobserved']}
    for (model, tp), observed_file in OBSERVED.items():
        source = PROFILER / 'examples/structure/cases' / f'a5-{model}-tp{tp}-long-context-source.json'
        original = load(source)
        # The shared source templates may already carry the binding. This
        # controlled pair always differs by exactly this one caller option.
        original['runtime']['kernels']['lowerings']['gdn_decode']['options']['gating_beta_store'] = {}
        raw = deepcopy(original)
        raw['runtime']['kernels']['lowerings']['gdn_decode']['options']['gating_beta_store'] = BINDING
        raw['runtime']['metadata']['frozen_scalar_store_binding'] = {
            'source_case': identity(source), 'parameter_source': 'Ascend950DT_9582 hardware profile',
            'scope': 'existing BF16 scalar-store primitive applied to source-compatible GDN caller',
            'target_timing_used_to_fit': False, 'captured_binary_identity': 'not_established'}
        output_case = CASES / source.name
        save(output_case, raw)
        retained_path = CURRENT / observed_file
        retained = load(retained_path)
        parsed = parse_case(raw)
        die = parsed.machine.dies[0]
        record = {'model': model, 'tp': tp, 'source_case': identity(source), 'case': identity(output_case),
            'retained_validation': identity(retained_path), 'observed_body_ms': retained['comparison']['observed_ms'],
            'comparison_control': 'before explicitly selects DMA with gating_beta_store={}; after binds frozen SKU service',
            'hardware_scalar_store': {'rate': dict(die.rates)['scalar_store_issue'],
                'owners': dict(die.resource_units)['scalar_store_issue'],
                'queue': dict(dict(die.store_queues)['scalar_store_issue'])},
            'native': native_groups(raw, retained)}
        record['before'] = prediction(original, retained)
        record['after'] = prediction(raw, retained)
        for mode in ('before', 'after'):
            result = record[mode]
            result['body_passed_15percent'] = abs(result['body_relative_error']) <= .15
            result['layer_family_sums'] = {kind: {
                'predicted_ms': sum(r['predicted_ms'] for r in result['layers'] if r['attention_kind'] == kind),
                'observed_ms': sum(r['observed_ms'] for r in result['layers'] if r['attention_kind'] == kind)}
                for kind in ('gdn', 'gqa')}
        report['cases'].append(record)
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(HERE / 'validation.json', report)
        print(json.dumps({'model': model, 'tp': tp, 'observed_ms': record['observed_body_ms']['mean'],
            **{mode: {'body_ms': record[mode]['body_ms'], 'error': record[mode]['body_relative_error'],
                      'first_layer_ms': record[mode]['first_layer_ms']} for mode in ('before', 'after')}}), flush=True)


if __name__ == '__main__':
    main()
