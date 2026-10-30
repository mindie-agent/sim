"""Validate unchanged HCCL responses against independent long-context captures.

[Role] Apply frozen per-call responses to compatible conditional source graphs.
[Invariants] Parameters, domains and original provenance are copied unchanged.
  Template identity remains unobserved; no Body latency enters a response.
[Timeouts & Locks] Each Body is bounded to 90 s; CSV is streamed within 90 s.
[Side-Effect Policy] Write local cases and compact validation; no device work.
"""
from bisect import bisect_right
from copy import deepcopy
from dataclasses import replace
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import signal
import subprocess
import time

spec = importlib.util.spec_from_file_location('gdn_store', Path(__file__).with_name('produce.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from mindie_sim import graph_costs, predict, target_graphs

MANIFEST = base.PROFILER / 'examples/structure/cases.json'
TP4 = base.ROOT / 'data/observations/a5-hccl-service-regimes-20261030/tp4-parameters.json'
TP2 = base.ROOT / 'examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json'
OUTPUT = base.ROOT / 'examples/cases/longctx-decode-20261030/a5-gdn-hccl-frozen'


def frozen_rule(tp):
    path = TP2 if tp == 2 else TP4
    doc = base.load(path)
    rules = doc['runtime']['kernels']['service_regimes'] if tp == 2 else doc['service_regimes']
    rule = next(r for r in rules if r['id'] == ('hcom_allreduce_-physical-4' if tp == 2
        else 'a5-aiv-tp4-allreduce-mesh_twoshot-observable-service-v1'))
    return deepcopy(rule), base.identity(path)


def selected_captures(model, tp):
    records = []
    for record in base.load(MANIFEST)['cases']:
        if not record['case_id'].startswith(f'a5-{model}-tp{tp}-decode-'):
            continue
        if not record['case_id'].endswith('-p126000'):
            continue
        extraction = base.load(Path(record['extraction_path']))
        assert base.identity(Path(record['extraction_path']))['sha256'] == record['extraction_identity']['sha256']
        for group in extraction['geometry_groups']:
            geometry = group['geometry']
            if geometry['tokens_per_sequence'] == 4:
                records.append((geometry['sequences_per_replica'], record, extraction, group))
    high = max(r[0] for r in records)
    return sorted((r for r in records if r[0] in {1, 4, high}), key=lambda r: r[0])


def native_calls(record, extraction, group, raw):
    """Stream only HCOM rows inside the retained complete Body populations."""
    path = Path(record['source']['path'])
    assert base.identity(path)['sha256'] == record['source']['sha256']
    ids = set(group['forward_ids'])
    populations = {}
    forwards = []

    def consume(forward, events):
        if forward['forward_id'] not in ids:
            return
        forwards.append(forward['forward_id'])
        stages = [s for s in forward['stages'] if s['layer_id'] is not None]
        assert len(stages) == 64
        layer_starts = [s['start_ns'] for s in stages]
        selected, _ = base.deduplicate_communication([event for event in events
            if event.op_type == 'hcom_allReduce_' and not base.is_aicpu_controller(event)
            and forward['start_ns'] <= event.start_ns < event.end_ns <= forward['end_ns']])
        assert len(selected) == 128
        for event in selected:
            layer = stages[bisect_right(layer_starts, event.start_ns) - 1]
            key = (layer['attention'], layer['layer_id'] == 0, event.block_dim, event.core)
            row = populations.setdefault(key, {'attention': key[0], 'first_layer': key[1],
                'blocks': key[2], 'core': key[3], 'count': 0, 'sum_us': 0,
                'min_us': float('inf'), 'max_us': 0, 'locator': event.locator})
            duration = event.duration_ns / 1000
            row['count'] += 1
            row['sum_us'] += duration
            row['min_us'] = min(row['min_us'], duration)
            row['max_us'] = max(row['max_us'], duration)
    case = base.parse_case(raw)
    config = base.segmentation_config({**case.model.to_dict(), 'tp': case.runtime.parallel.tp}, extraction)
    source = extraction['source_identity']
    ref = base.SourceRef('rank_0', 0, 'csv', str(path), path.name, source['sha256'],
        path.stat().st_size, source['schema_version'], 'retained_full_capture', mtime_ns=path.stat().st_mtime_ns)
    base.auto_segment_decode_forwards(base.kernel_rows(path, rank_id='rank_0', rank_index=0,
        max_rows=5_000_000, timeout_s=90), config, source=ref, on_forward=consume,
        max_detailed_forwards=0, max_samples_per_layer=0)
    assert set(forwards) == ids
    for row in populations.values():
        row['mean_us'] = row['sum_us'] / row['count']
        row['calls_per_body'] = row['count'] / len(forwards)
    return {'forwards': len(forwards), 'calls_per_body': 128,
            'groups': list(populations.values()), 'source': record['source'],
            'boundary': 'deduplicated native AivKernel inside complete 64-layer target Body'}


def primitive(case, batch):
    tiny = deepcopy(case)
    tiny['model']['layers'] = 1
    graphs = target_graphs(tiny, scope='body', phase='decode', tokens=4, context=126000, sequences=batch)
    selected, selections = {}, []
    for rank, ops in graphs.items():
        entry = next(op for op in ops if op.recipe == 'hccl_entry')
        group = dict(entry.metadata)['hccl_source_group']
        keep = [op for op in ops if dict(op.metadata).get('hccl_source_group') == group]
        ids = {op.op_id for op in keep}
        selected[rank] = tuple(replace(op, deps=tuple(d for d in op.deps if d in ids)) for op in keep)
        selections.append(dict(entry.metadata)['hccl_selection'])
    result = graph_costs(tiny, selected, isolated=False)
    return {'duration_us': result['duration_ms'] * 1000, 'selection': selections[0],
            'parameters': result['parameters'].get('service_regimes', []),
            'live_tasks': result['execution']['live_tasks'], 'live_programs': result['execution']['live_programs']}


def body(case, batch, observed):
    signal.alarm(90)
    start = time.monotonic()
    try:
        result = predict(case, scope='body', phase='decode', tokens=4, context=126000,
                         sequences=batch, event_limit=0)
    finally:
        signal.alarm(0)
    timeline = {r['op_id']: r for r in result['timeline'] if r['replica'] == 0}
    calls = []
    for name, row in timeline.items():
        if row['recipe'] == 'hccl_entry':
            end = timeline[name.removesuffix('.hccl.entry')]
            calls.append((end['complete_ms'] - row['ready_ms']) * 1000)
    assert len(calls) == 128
    ms = result['body_forward_ms']
    return {'status': 'completed', 'body_ms': ms, 'body_relative_error': ms / observed - 1,
            'body_passed_15percent': abs(ms / observed - 1) <= .15,
            'hccl_call_count': len(calls), 'hccl_mean_us': sum(calls) / len(calls),
            'hccl_min_us': min(calls), 'hccl_max_us': max(calls),
            'parameters': result['parameters'].get('service_regimes', []),
            'elapsed_seconds': time.monotonic() - start,
            'live_tasks': result['execution']['live_tasks'], 'live_programs': result['execution']['live_programs']}


def bounded_body(case, batch, observed):
    start = time.monotonic()
    try:
        return body(case, batch, observed)
    except TimeoutError as error:
        return {'status': 'timed_out', 'error': str(error),
                'elapsed_seconds': time.monotonic() - start, 'body_score': 'unavailable'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Keep already scored or explicitly timed-out rows with unchanged input identities')
    parser.add_argument('--retry-timeouts', action='store_true', help='Retry timed-out Bodies within the same 90-second budget after a separately verified engine fix')
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, base.timeout)
    started = time.monotonic()
    destination = base.HERE / 'hccl-frozen-validation.json'
    previous = base.load(destination) if args.resume and destination.exists() else {}
    retained = {(r['model'], r['tp'], r['batch']): r for r in previous.get('cases', [])}
    report = {'scope': 'unchanged_HCCL_response_transfer_to_independent_long_context',
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.ROOT, text=True).strip(),
        'manifest': base.identity(MANIFEST), 'parameter_changes': 0, 'domain_changes': 0,
        'selection': 'B1, B4 and largest actual q4 batch in each deployment extraction',
        'template_identity': 'conditional source candidate; not captured algorithm proof',
        'hardware_identity': 'declared Ascend950DT_9582; captured SKU not independently established',
        'response_boundary': 'inclusive native call; control is not added to analytic duration',
        'validation_role': 'new 128K captures excluded from old coefficients; already inspected, not blind',
        'context_basis': 'request prompt 126000; exact per-step KV increments unobserved',
        'previous_run': ({'revision': previous['revision'], 'elapsed_seconds': previous['elapsed_seconds'],
                          'resumed_rows': len(retained)} if previous else None),
        'cases': []}
    for model, tp in base.OBSERVED:
        template = 'mesh_oneshot' if tp == 2 else 'mesh_twoshot'
        source = base.ROOT / 'examples/cases/longctx-decode-20261030/a5-gdn-hccl-source' / f'a5-{model}-tp{tp}-{template}.json'
        control = base.load(source)
        raw = deepcopy(control)
        rule, rule_source = frozen_rule(tp)
        assert not raw['runtime']['kernels'].get('service_regimes')
        raw['runtime']['kernels']['service_regimes'] = [rule]
        raw['runtime']['metadata']['frozen_hccl_transfer'] = {
            'parameter_source': rule_source, 'parameter_changes': 0, 'domain_changes': 0,
            'evaluation_relation': 'independent_new_128K_capture_not_used_in_frozen_response',
            'source_provenance_retained_verbatim': True, 'captured_template': 'unobserved',
            'out_of_domain': 'analytic; no extrapolation or cross-TP substitution'}
        output = OUTPUT / f'a5-{model}-tp{tp}-{template}-frozen.json'
        base.save(output, raw)
        assert raw['runtime']['kernels']['service_regimes'][0] == frozen_rule(tp)[0]
        for batch, record, extraction, population in selected_captures(model, tp):
            old = retained.get((model, tp, batch))
            if old is not None:
                assert old.get('base_source_case', old['control_case']) == base.identity(source)
                assert old['control_case'] == base.identity(Path(old['control_case']['path']))
                assert old['case'] == base.identity(Path(old['case']['path']))
                assert old['extraction'] == record['extraction_identity']
                if not args.retry_timeouts or old['after']['status'] == old['before']['status'] == 'completed':
                    report['cases'].append(old)
                    continue
            native = native_calls(record, extraction, population, control)
            observed_blocks, = {r['blocks'] for r in native['groups']}
            local_control, local_raw = deepcopy(control), deepcopy(raw)
            control_identity, case_identity = base.identity(source), base.identity(output)
            if tp == 4 and observed_blocks == 5:
                declaration = {'expansion': 'AIV', 'template': 'mesh_oneshot', 'block_limit': 5, 'axes': ['tp']}
                for case in (local_control, local_raw):
                    case['runtime']['kernels']['hccl']['allreduce'] = declaration
                    case['runtime']['metadata']['hccl_source_candidate'] = {
                        'declaration': declaration, 'native_block_num': 5,
                        'captured_template': 'unobserved',
                        'source_dispatch': 'conditional OneShot R+1 branch matches native5; no timing-based selection'}
                low_source = OUTPUT / f'a5-{model}-tp4-b{batch}-mesh_oneshot-control.json'
                low_case = OUTPUT / f'a5-{model}-tp4-b{batch}-mesh_oneshot-frozen.json'
                base.save(low_source, local_control)
                base.save(low_case, local_raw)
                control_identity, case_identity = base.identity(low_source), base.identity(low_case)
            else:
                assert observed_blocks == (3 if tp == 2 else 60)
            before = primitive(local_control, batch)
            after = primitive(local_raw, batch)
            observed = population['body_forward_span_ms']
            row = {'model': model, 'tp': tp, 'batch': batch, 'geometry': population['geometry'],
                'case': case_identity, 'control_case': control_identity,
                'base_source_case': base.identity(source),
                'capture_id': record['case_id'], 'extraction': record['extraction_identity'],
                'observed_body_ms': observed, 'native': native,
                'primitive_before': before, 'primitive_after': after}
            if old is not None:
                row['prior_body_attempt'] = {'before': old['before'], 'after': old['after'],
                    'revision': previous['revision']}
            row['before'] = bounded_body(local_control, batch, observed['mean'])
            hit = any(p['queries'].get('observable_response', 0) for p in after['parameters'])
            if hit:
                row['after'] = bounded_body(local_raw, batch, observed['mean'])
                row['execution_relation'] = 'both_complete_Bodies_requested; see individual status'
            else:
                assert before['duration_us'] == after['duration_us']
                row['after'] = deepcopy(row['before'])
                row['execution_relation'] = 'one_Body_requested; unchanged_analytic_program_domain_miss_verified_in_primitive'
            report['cases'].append(row)
            report['elapsed_seconds'] = time.monotonic() - started
            report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report['status'] = ('incomplete_body_timeouts' if any(r['after']['status'] != 'completed'
                or r['before']['status'] != 'completed' for r in report['cases']) else 'completed_so_far')
            base.save(base.HERE / 'hccl-frozen-validation.json', report)
            print(json.dumps({'model': model, 'tp': tp, 'batch': batch, 'hit': hit,
                'primitive_before_us': before['duration_us'], 'primitive_after_us': after['duration_us'],
                'before_status': row['before']['status'], 'after_status': row['after']['status'],
                'body_before_ms': row['before'].get('body_ms'), 'body_after_ms': row['after'].get('body_ms'),
                'error': row['after'].get('body_relative_error')}), flush=True)
    report['status'] = ('incomplete_body_timeouts' if any(r['after']['status'] != 'completed'
        or r['before']['status'] != 'completed' for r in report['cases']) else 'completed')
    base.save(base.HERE / 'hccl-frozen-validation.json', report)
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
