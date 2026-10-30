"""Locate B7/B8 native residuals using the existing complete Body boundaries.

[Role] Compare kernel populations inside topology-aligned attention/FFN blocks.
[Invariants] Observed duration sums may overlap and are diagnostic, not service
  fits. Scheduler wrappers and duplicate HCOM summaries do not become kernels.
[Timeouts & Locks] One streamed source scan; one finite simulated Body at a time.
[Side-Effect Policy] Local diagnostic JSON only; no capture or parameter changes.
"""
from bisect import bisect_right
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PROFILER = ROOT.parent/'profiler'
sys.path[:0] = [str(ROOT), str(PROFILER), str(PROFILER/'examples/structure')]
from mindie_sim import predict
from mindie_sim.prediction.structure import parse_case
from mindie_profiler.decode_segmentation import auto_segment_decode_forwards, _norm, _norm_start, _anchor, _Row, is_aicpu_controller
from mindie_profiler.normalized import kernel_rows, shapes
from mindie_profiler.events import SourceRef
from operators import segmentation_config, deduplicate_communication

spec = importlib.util.spec_from_file_location('keypool_parameters', Path(__file__).with_name('producer.py'))
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
HERE = Path(__file__).resolve().parent
EXTRACTION = Path('/Volumes/data/mindie-sim-data/acceptance-archives-20261030/pending/long-context-decode/eb9a5755fea96e96546e93270119f5e194971f5bab79972d1ebff0a5eab87e85.json')


def main():
    start = time.monotonic()
    raw = producer.load(producer.CASE)
    case = parse_case(raw)
    retained = producer.load(EXTRACTION)
    config = segmentation_config({**case.model.to_dict(), 'tp': case.runtime.parallel.tp}, retained)
    expected = {f['forward_id']: f for f in retained['forwards']
                if f['geometry']['sequences_per_replica'] in (7, 8)}
    groups, layers, seen = {}, defaultdict(list), defaultdict(list)

    def consume(forward, events):
        if forward['forward_id'] not in expected:
            return
        original = expected[forward['forward_id']]
        assert all(forward[k] == original[k] for k in ('start_ns', 'end_ns', 'query_rows'))
        batch = forward['geometry']['sequences_per_replica']
        seen[batch].append(forward['forward_id'])
        stages = [s for s in forward['stages'] if s['layer_id'] is not None]
        starts, ffn_start = [s['start_ns'] for s in stages], {}
        for layer in stages:
            candidates = [_Row(e, shapes(e.input_shapes), shapes(e.output_shapes)) for e in events
                if layer['start_ns'] <= e.start_ns < layer['evidence']['successor_boundary_start_ns']]
            anchor = next(row.event.start_ns for row in candidates if _anchor(row, config))
            norm = next(i for i, row in enumerate(candidates)
                        if row.event.start_ns > anchor and _norm(row, config['hidden_size']))
            ffn_start[layer['layer_id']] = candidates[_norm_start(candidates, norm, config['hidden_size'])].event.start_ns
            layers[(batch, layer['layer_id'])].append({'attention': layer['attention'],
                'span_ms': layer['span_ms'], 'attention_ms': (ffn_start[layer['layer_id']] - layer['start_ns'])/1e6,
                'ffn_ms': (layer['end_ns'] - ffn_start[layer['layer_id']])/1e6})
        selected = [e for e in events if forward['start_ns'] <= e.start_ns < e.end_ns <= forward['end_ns']
                    and not is_aicpu_controller(e)]
        selected, _ = deduplicate_communication(selected)
        for event in selected:
            layer = stages[bisect_right(starts, event.start_ns)-1]
            block = 'attention' if event.start_ns < ffn_start[layer['layer_id']] else 'ffn'
            key = (batch, event.op_type, block, layer['attention'], layer['layer_id'] < 3,
                   str(event.input_shapes), str(event.output_shapes))
            group = groups.setdefault(key, {'batch': batch, 'native': event.op_type,
                'block': block, 'attention_kind': layer['attention'], 'dense_layer': layer['layer_id'] < 3,
                'input_shapes': event.input_shapes, 'output_shapes': event.output_shapes,
                'count': 0, 'total_us': 0., 'first_locator': event.locator})
            group['count'] += 1
            group['total_us'] += event.duration_ns/1000

    source_id = retained['source_identity']
    path = Path(source_id['path'])
    assert producer.identity(path)['sha256'] == source_id['sha256']
    source = SourceRef('rank_0', 0, 'csv', str(path), path.name, source_id['sha256'], path.stat().st_size,
                       source_id['schema_version'], 'retained_full_capture', mtime_ns=path.stat().st_mtime_ns)
    actual = auto_segment_decode_forwards(kernel_rows(path, rank_id='rank_0', rank_index=0,
        max_rows=5_000_000, timeout_s=90), config, source=source, on_forward=consume,
        max_detailed_forwards=0, max_samples_per_layer=0)
    assert set(sum(seen.values(), [])) == set(expected)
    report = {'scope': 'topology_aligned_native_kernel_and_layer_residual; duration_sums_not_additive_walltime',
        'source': producer.identity(path), 'extraction': producer.identity(EXTRACTION),
        'case': producer.identity(producer.CASE), 'forward_ids': dict(seen), 'observed_groups': list(groups.values()),
        'layers': [], 'predicted_groups': [], 'scan_counts': actual['scan_counts'], 'new_parameters': 0}
    for group in report['observed_groups']:
        group['mean_us'] = group['total_us']/group['count']
        group['per_forward_sum_us'] = group['total_us']/len(seen[group['batch']])
    for batch in (7, 8):
        result = predict(raw, scope='body', phase='decode', tokens=6, context=131201, sequences=batch, event_limit=0)
        simulated = defaultdict(lambda: {'count': 0, 'total_ms': 0.})
        for row in result['timeline']:
            key = (row['recipe'], row.get('fusion_role'), row['layer'] < 3,
                   'mla' if row['layer'] % 4 == 3 else 'kda')
            group = simulated[key]
            group['count'] += 1
            group['total_ms'] += row['elapsed_ms']
        report['predicted_groups'].extend({'batch': batch, 'recipe': key[0], 'fusion_role': key[1],
            'dense_layer': key[2], 'attention_kind': key[3], **value} for key, value in simulated.items())
        for layer in range(45):
            timeline = [r for r in result['timeline'] if r['layer'] == layer]
            observed = layers[(batch, layer)]
            report['layers'].append({'batch': batch, 'layer': layer, 'attention_kind': observed[0]['attention'],
                'predicted_ms': max(r['complete_ms'] for r in timeline)-min(r['ready_ms'] for r in timeline),
                **{'observed_' + k: sum(r[k] for r in observed)/len(observed)
                   for k in ('span_ms', 'attention_ms', 'ffn_ms')}})
    report['elapsed_seconds'] = time.monotonic()-start
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    producer.save(HERE/'residuals.json', report)
    print(json.dumps({'elapsed_seconds': report['elapsed_seconds'], 'peak_rss_bytes': report['peak_rss_bytes'],
                      'observed_groups': len(groups), 'forwards': {k: len(v) for k,v in seen.items()}}))


if __name__ == '__main__':
    main()
