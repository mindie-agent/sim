"""Validate one unchanged native GLM case over all recorded short/long batches.

[Role] Complete the Body matrix and inspect the unusual B7 first-layer span.
[Invariants] One case and parameter set; no per-population choices, layer drops,
  latency fitting, or attribution of missing device events to unobserved CPU work.
[Timeouts & Locks] Stream each capture within 90 seconds; each Body within 90 s.
[Side-Effect Policy] Local evidence JSON only; no NPU work or parameter changes.
"""
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import resource
import signal
import statistics
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

spec = importlib.util.spec_from_file_location('keypool_parameters', Path(__file__).with_name('producer.py'))
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
HERE = Path(__file__).resolve().parent
CASE = ROOT / 'examples/cases/longctx-decode-20261030/glm53flash-a3-native-keypool-gmm-service.json'
EXTRACTION = Path('/Volumes/data/mindie-sim-data/acceptance-archives-20261030/pending/long-context-decode/eb9a5755fea96e96546e93270119f5e194971f5bab79972d1ebff0a5eab87e85.json')


def stats(values):
    return {'count': len(values), 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values)}


def event_item(event, origin):
    return {'native': event.op_type, 'name': event.name, 'core': event.core,
            'stream': event.stream_id, 'start_us': (event.start_ns - origin) / 1000,
            'end_us': (event.end_ns - origin) / 1000, 'duration_us': event.duration_ns / 1000,
            'locator': event.locator}


def layer_zero(forward, events):
    stage = next(s for s in forward['stages'] if s['layer_id'] == 0)
    start, end = stage['start_ns'], stage['end_ns']
    candidates = [e for e in events if start <= e.start_ns < e.end_ns <= end]
    controllers = [e for e in candidates if is_aicpu_controller(e)]
    concrete, duplicates = deduplicate_communication([e for e in candidates if not is_aicpu_controller(e)])
    concrete.sort(key=lambda e: (e.start_ns, e.end_ns))
    cursor, gaps = start, []
    for event in concrete:
        if event.start_ns > cursor:
            gaps.append({'start_us': (cursor-start)/1000, 'end_us': (event.start_ns-start)/1000,
                         'duration_us': (event.start_ns-cursor)/1000})
        cursor = max(cursor, event.end_ns)
    if cursor < end:
        gaps.append({'start_us': (cursor-start)/1000, 'end_us': (end-start)/1000,
                     'duration_us': (end-cursor)/1000})
    gap_us = sum(g['duration_us'] for g in gaps)
    return {'forward_id': forward['forward_id'], 'stage': stage,
            'span_us': (end-start)/1000, 'device_activity_union_us': (end-start)/1000-gap_us,
            'uncovered_device_span_us': gap_us, 'gaps': gaps,
            'events': [event_item(e, start) for e in concrete],
            'controllers': [event_item(e, start) for e in controllers],
            'duplicate_hcom_summaries': len(duplicates),
            'attribution_limit': 'kernel CSV has no CPU/runtime trace; uncovered spans are not CPU-wait proof'}


def extract(path, config, inspect_layer_zero=False):
    source_id = producer.identity(path)
    source = SourceRef('rank_0', 0, 'csv', str(path), path.name, source_id['sha256'], path.stat().st_size,
                       'kernel_details_csv_46col', 'retained_full_capture', mtime_ns=path.stat().st_mtime_ns)
    groups, diagnostics = defaultdict(list), []

    def consume(forward, events):
        geometry = forward['geometry']
        if geometry['tokens_per_sequence'] != 6:
            return
        batch = geometry['sequences_per_replica']
        stages = [s for s in forward['stages'] if s['layer_id'] is not None]
        assert len(stages) == 45
        groups[batch].append({'forward_id': forward['forward_id'],
            'start_ns': forward['start_ns'], 'end_ns': forward['end_ns'],
            'body_ms': forward['body_forward_span_ms'],
            'activity_ms': forward['body_activity_ms'], 'gap_ms': forward['body_internal_gap_ms'],
            'layers': {s['layer_id']: s['span_ms'] for s in stages}})
        if inspect_layer_zero and batch in (7, 8):
            diagnostics.append({'batch': batch, **layer_zero(forward, events)})

    result = auto_segment_decode_forwards(kernel_rows(path, rank_id='rank_0', rank_index=0,
        max_rows=5_000_000, timeout_s=90), config, source=source, on_forward=consume,
        max_detailed_forwards=0, max_samples_per_layer=0)
    return {'source': source_id, 'scan_counts': result['scan_counts'], 'populations': [
        {'batch': batch, 'body_ms': stats([f['body_ms'] for f in forwards]),
         'activity_ms': stats([f['activity_ms'] for f in forwards]),
         'gap_ms': stats([f['gap_ms'] for f in forwards]),
         'forwards': forwards,
         'layers': [{'layer': layer, 'observed_ms': stats([f['layers'][layer] for f in forwards])}
                    for layer in range(45)]} for batch, forwards in sorted(groups.items())]}, diagnostics


def services(raw, context, batch):
    tiny = deepcopy(raw)
    tiny['model']['layers'] = 4
    ops = [o for o in target_graphs(tiny, scope='body', phase='decode', tokens=6, context=context,
                                  sequences=batch)[0]
           if o.recipe in ('grouped_down', 'cann_key_pool', 'cann_pool_key_indexer')
           or dict(o.metadata).get('fusion_role') == 'grouped_gate_up_activation_quant']
    costs = operator_costs(tiny, ops)['operators']
    return [{'recipe': o.recipe, 'fusion_role': dict(o.metadata).get('fusion_role'),
             'program': c['program'], 'duration_us': c['duration_ms']*1000,
             'reference_hbm_bytes': o.hbm_bytes} for o, c in zip(ops, costs)]


def main():
    started = time.monotonic()
    raw = producer.load(CASE)
    case = parse_case(raw)
    config = segmentation_config({**case.model.to_dict(), 'tp': case.runtime.parallel.tp}, producer.load(EXTRACTION))
    report = {'scope': 'same_case_full_45_layer_q6_short_and_long_context_Body_matrix',
              'case': producer.identity(CASE), 'gmm_parameters': producer.identity(HERE / 'gmm-parameters.json'),
              'keypool_parameters': producer.identity(HERE / 'parameters.json'),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'runtime_sources': {name: producer.identity(ROOT / 'mindie_sim/prediction' / name)
                                  for name in ('feedforward.py', 'moe_group_list.py', 'index_pool_native.py')},
              'source_extractions': [], 'rows': [], 'new_parameters': 0,
              'parameter_selection': 'all populations use exactly the same case; no retuning or case selection',
              'context_basis': 'capture scenario initial prompt; actual per-call context not in kernel CSV',
              'validation_history': 'previously inspected captures; short native GMM/KeyPool are training inputs',
              'threshold': .15}
    for label, path, context in (('short_4481', producer.SHORT, 4481), ('long_131201', producer.LONG, 131201)):
        extraction, diagnostics = extract(path, config, inspect_layer_zero=label.startswith('long'))
        report['source_extractions'].append({'label': label, 'context': context, **extraction})
        if diagnostics:
            producer.save(HERE / 'gmm-layer0-boundary.json', {'source': extraction['source'],
                'extraction_contract': producer.identity(EXTRACTION), 'records': diagnostics,
                'no_layers_dropped': True, 'cpu_trace_available': False})
        producer.save(HERE / 'gmm-matrix-validation.json', report)
        print(json.dumps({'extracted': label, 'batches': [p['batch'] for p in extraction['populations']]}), flush=True)
        for population in extraction['populations']:
            batch = population['batch']
            signal.signal(signal.SIGALRM, producer.timeout)
            signal.alarm(90)
            before = time.monotonic()
            result = predict(raw, scope='body', phase='decode', tokens=6, context=context,
                             sequences=batch, event_limit=0)
            signal.alarm(0)
            row = {'capture': label, 'context': context, 'batch': batch,
                   'observed_ms': population['body_ms'], 'predicted_ms': result['body_forward_ms'],
                   'relative_error': result['body_forward_ms']/population['body_ms']['mean']-1,
                   'elapsed_seconds': time.monotonic()-before, 'layers': [],
                   'live_tasks': result['execution']['live_tasks'],
                   'live_programs': result['execution']['live_programs'],
                   'native_services': services(raw, context, batch)}
            row['passed_15percent'] = abs(row['relative_error']) <= .15
            for layer in population['layers']:
                records = [r for r in result['timeline'] if r['layer'] == layer['layer'] and r['replica'] == 0]
                ms = max(r['complete_ms'] for r in records)-min(r['ready_ms'] for r in records)
                error = ms/layer['observed_ms']['mean']-1
                row['layers'].append({**layer, 'predicted_ms': ms, 'relative_error': error,
                                      'passed_15percent': abs(error) <= .15})
            row['layer_passes'] = sum(x['passed_15percent'] for x in row['layers'])
            report['rows'].append(row)
            report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            producer.save(HERE / 'gmm-matrix-validation.json', report)
            print(json.dumps({k: v for k, v in row.items() if k not in ('layers', 'native_services')}), flush=True)
    assert producer.identity(CASE) == report['case']
    assert producer.identity(HERE / 'gmm-parameters.json') == report['gmm_parameters']
    report['elapsed_seconds'] = time.monotonic()-started
    report['summary'] = {label: {'passed': sum(r['passed_15percent'] for r in report['rows'] if r['capture'] == label),
                                'total': sum(r['capture'] == label for r in report['rows'])}
                         for label in ('short_4481', 'long_131201')}
    producer.save(HERE / 'gmm-matrix-validation.json', report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
