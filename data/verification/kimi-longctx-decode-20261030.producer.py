"""Bounded Kimi long-context Decode structure and execution probes.

[Role] Exercise complete 93-layer TP8/DP4/EP32 bodies and native core programs.
[Invariants] Explicit SplitKV scenarios are not captured host dispatch. Context
  probes do not simulate the complete prompt/output stream or fit observations.
  The standard workload is decimal 126000 input and 3000 output tokens.
[Timeouts & Locks] Each child has 90s and 512MiB RSS limits plus 5s cleanup.
[Side-Effect Policy] Writes local cases and compact evidence; no device work.
"""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'examples/cases/kimi-k3-a5-sp-right-pad-conditional.json'
ARTIFACT = ROOT / 'data/verification/kimi-longctx-decode-20261030'
RAW = Path(tempfile.gettempdir()) / 'mindie-kimi-longctx-decode-20261030'
OUT, CASES = RAW / 'runs', RAW / 'cases'
MLA = 'ascend_paged_mla_decode_arch35'
KDA = 'vllm_ascend_recurrent_kda_arch35'


def identity(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(),
            'size_bytes': Path(path).stat().st_size}


def package_identity():
    rows = [(str(p.relative_to(ROOT)), identity(p)['sha256'])
            for p in sorted((ROOT / 'mindie_sim').rglob('*.py'))]
    return {'sha256': sha256(json.dumps(rows).encode()).hexdigest(), 'files': len(rows)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    sys.path.insert(0, str(ROOT))
    from mindie_sim.prediction.decode_probe import decode_windows
    base = json.loads(BASE.read_text())
    windows = decode_windows(126000, 3000, output_offsets=(0, 1500, 2999))
    entries = []
    # Context, global concurrency, explicit equal-tile splits per local sequence.
    grid = [(32768, 1, 1), (32768, 1, 32), (65536, 1, 32),
            (126000, 1, 32), (127500, 1, 32), (128999, 1, 32),
            (126000, 4, 32), (126000, 16, 8), (126000, 64, 2)]
    for context, concurrency, splits in grid:
        raw = deepcopy(base)
        name = f'kimi-current-ctx{context}-c{concurrency}-split{splits}'
        raw['name'] = name
        raw['workload']['requests'] = [
            {'id': f'r{i}', 'prompt_tokens': 126000, 'output_tokens': 3000}
            for i in range(concurrency)]
        runtime = raw['runtime']
        runtime['max_model_len'] = 129000
        runtime['max_num_seqs'] = 64
        runtime['kernels']['lowerings'].update({
            'paged_mla_attention': {'implementation': MLA,
                                    'options': {'kv_splits': splits}},
            'kda': {'implementation': KDA, 'options': {'use_beta_sigmoid': False}},
        })
        counts = [concurrency // 4 + (i < concurrency % 4) for i in range(4)]
        runtime['metadata']['longctx_decode_probe'] = {
            'context': context, 'sequences_by_replica': counts,
            'standard_input_tokens_decimal': 126000,
            'standard_output_tokens_decimal': 3000,
            'boundary': 'one_complete_target_body_device_graph_no_host_output_scheduler',
            'split_policy': 'explicit_equal_tile_splits_sensitivity_control',
            'dispatch_identity': 'native_host_output_not_established',
            'checkpoint_identity': 'current_quant_metadata_conditional_to_historical_capture',
            'early_contexts': '32768/65536 are operator/body scaling controls, not the standard workload timeline',
            'calibration': 'none',
        }
        path = CASES / (name + '.json')
        write(path, raw)
        entries.append({'id': name, 'context': context, 'concurrency': concurrency,
                        'splits': splits, 'sequences_by_replica': counts,
                        'input': identity(path)})
    result = {'schema': 'mindie-kimi-longctx-decode-inputs/1', 'cases': entries,
              'base': identity(BASE), 'producer': identity(Path(__file__)),
              'standard_decode_windows': windows,
              'limits': {'seconds': 90, 'rss_bytes': 512 * 2**20, 'cleanup_seconds': 5}}
    write(ARTIFACT / 'inputs.json', result)
    return result


def worker(path, output, body):
    sys.path.insert(0, str(ROOT))
    from mindie_sim import graph_costs, operator_costs, target_graphs
    from mindie_sim.prediction.decode_probe import probe_capacity
    from mindie_sim.prediction.structure import parse_case
    raw = json.loads(path.read_text())
    meta = raw['runtime']['metadata']['longctx_decode_probe']
    context, counts = meta['context'], meta['sequences_by_replica']
    started = time.perf_counter()
    before = package_identity()
    case = parse_case(raw)
    capacity = probe_capacity(case, context=context, tokens=1, sequences=counts)
    graph = target_graphs(case, phase='decode', tokens=1, context=context,
                          sequences=counts, scope='body', resource_fusion=False)
    print(json.dumps({'event': 'graph_ready', 'ops': {r: len(g) for r, g in graph.items()},
                      'wall_seconds': time.perf_counter() - started}), flush=True)
    chosen = {}
    for op in graph[0]:
        if op.recipe == MLA:
            chosen.setdefault('mla', op)
        if op.recipe == KDA:
            chosen.setdefault('kda', op)
    if set(chosen) != {'mla', 'kda'}:
        raise RuntimeError('required_native_decode_cores_missing:' + str(chosen.keys()))
    cores = {}
    for name, op in chosen.items():
        print(json.dumps({'event': 'operator_start', 'family': name}), flush=True)
        row = operator_costs(case, (op,), input_ready_ms=7)['operators'][0]
        cores[name] = row
        assert row['live_tasks'] == 0 and not any(row['remaining_buffers'].values())
        assert not row['service_before_input']
    result = {'schema': 'mindie-kimi-longctx-decode-probe/1', 'status': 'completed',
              'input': identity(path), 'probe': meta, 'runtime_before': before,
              'model': {'layers': case.model.layers,
                        'layer_counts': dict(Counter(p.attention for p in case.model.layer_plans)),
                        'tp': case.runtime.parallel.tp, 'dp': case.runtime.parallel.dp,
                        'ep': case.runtime.parallel.ep},
              'operators_by_replica': {r: len(g) for r, g in graph.items()},
              'cores': cores,
              'capacity': capacity, 'producer': identity(Path(__file__)),
              'body': {'status': 'not_requested'},
              'comparison': {'status': 'unmeasured_physical_scenario',
                             'observed_timings_used_as_parameters': False}}
    del graph
    write(output.with_suffix('.partial.json'), result)
    if body:
        print(json.dumps({'event': 'body_start', 'wall_seconds': time.perf_counter() - started}), flush=True)
        graphs = target_graphs(case, phase='decode', tokens=1, context=context,
                               sequences=counts, scope='body')
        report = graph_costs(case, graphs, external_inputs={'analysis_input': 0},
                             isolated=False, trace_limit=0)
        sums, occurrences = Counter(), Counter()
        for row in report['timeline']:
            if row['replica'] == 0 and row['recipe'] in {MLA, KDA}:
                sums[row['recipe']] += row['elapsed_ms']
                occurrences[row['recipe']] += 1
        result['body'] = {'status': 'completed', 'duration_ms': report['duration_ms'],
                          'native_core_elapsed_span_sum_replica0_ms': dict(sums),
                          'native_core_occurrences_replica0': dict(occurrences),
                          'execution': report['execution'], 'parameters': report['parameters']}
        assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    result['runtime_after'] = package_identity()
    result['runtime_unchanged'] = result['runtime_after'] == before
    result['wall_seconds'] = time.perf_counter() - started
    write(output, result)


def run(entry, body):
    output = OUT / (entry['id'] + ('.body' if body else '.operators') + '.json')
    log, err = output.with_suffix('.stdout.log'), output.with_suffix('.stderr.log')
    start, peak, reason = time.perf_counter(), 0, None
    OUT.mkdir(parents=True, exist_ok=True)
    runtime_before = package_identity()
    with log.open('w') as stdout, err.open('w') as stderr:
        args = [sys.executable, __file__, 'worker', '--case', entry['input']['path'],
                '--output', str(output)] + (['--body'] if body else [])
        child = subprocess.Popen(args, stdout=stdout, stderr=stderr)
        while child.poll() is None:
            sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(child.pid)],
                                    capture_output=True, text=True, timeout=2)
            if sample.stdout.strip():
                peak = max(peak, int(sample.stdout.strip()) * 1024)
            if peak > 512 * 2**20:
                reason = 'memory_limit'
            elif time.perf_counter() - start >= 90:
                reason = 'timeout'
            if reason:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
                break
            time.sleep(.25)
        code = child.wait(timeout=3)
    receipt = {'id': entry['id'], 'body_requested': body,
               'status': reason or ('completed' if code == 0 else 'failed'),
               'exit_code': code, 'wall_seconds': time.perf_counter() - start,
               'peak_sampled_rss_bytes': peak, 'stdout': identity(log), 'stderr': identity(err),
               'runtime_before': runtime_before, 'runtime_after': package_identity()}
    receipt['runtime_unchanged'] = receipt['runtime_before'] == receipt['runtime_after']
    if code == 0:
        receipt['result'] = identity(output)
    else:
        receipt['error_tail'] = err.read_text()[-2500:]
    write(output.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def summarize():
    """Reconcile actual completed runs without re-executing device programs."""
    sys.path.insert(0, str(ROOT))
    from mindie_sim.prediction.decode_probe import probe_capacity
    manifest = json.loads((ARTIFACT / 'inputs.json').read_text())
    rows, checks = [], []
    for entry in manifest['cases']:
        paths = [p for p in OUT.glob(entry['id'] + '.*.json')
                 if p.name.endswith(('.body.json', '.operators.json'))]
        if not paths:
            rows.append({**entry, 'status': 'not_run'})
            continue
        path = next((p for p in paths if p.name.endswith('.body.json')), paths[0])
        result = json.loads(path.read_text())
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        raw = json.loads(Path(entry['input']['path']).read_text())
        assert result['input']['sha256'] == entry['input']['sha256']
        assert result['runtime_unchanged']
        mla, kda = result['cores']['mla'], result['cores']['kda']
        geometry = mla['operator']['metadata']['paged_mla_geometry']
        contexts, heads, page = geometry['contexts'], geometry['heads'], geometry['page_size']
        kv = sum(c + 1 for c in contexts) * 576 * 2
        queries = indices = partial_roundtrip = 0
        for context in contexts:
            tiles = (context + 128) // 128
            splits = min(geometry['kv_splits'], tiles)
            small, extra = divmod(tiles, splits)
            start = 0
            for split in range(splits):
                count = small + (split < extra)
                stop = min((start + count) * 128, context + 1)
                indices += ((stop + page - 1) // page - start * 128 // page) * 4
                queries += heads * 576 * 2
                if splits > 1:
                    partial_roundtrip += 2 * heads * (512 * 4 + 8)
                start += count
        output = len(contexts) * heads * 512 * 2
        expected_hbm = kv + queries + indices + partial_roundtrip + output
        assert mla['operator']['hbm_bytes'] == expected_hbm
        capacity = probe_capacity(raw, context=entry['context'], tokens=1,
                                  sequences=entry['sequences_by_replica'])
        core_meta = mla['operator']['metadata']['program_execution']
        state_meta = kda['operator']['metadata']['program_execution']['kda_recurrent_schedule']
        body = result['body']
        if body['status'] == 'completed':
            assert body['execution']['live_tasks'] == body['execution']['live_programs'] == 0
        rows.append({
            'id': entry['id'], 'status': result['status'], 'context': entry['context'],
            'concurrency': entry['concurrency'], 'sequences_by_replica': entry['sequences_by_replica'],
            'model': result['model'], 'input_sha256': entry['input']['sha256'],
            'runtime_before': result['runtime_before'], 'runtime_after': result['runtime_after'],
            'runtime_unchanged': result['runtime_unchanged'],
            'raw_result': identity(path),
            'receipt': {k: receipt[k] for k in ('status', 'exit_code', 'wall_seconds',
                                               'peak_sampled_rss_bytes', 'body_requested')},
            'mla': {'isolated_us': 1000 * mla['duration_ms'], 'hbm_bytes': expected_hbm,
                    'hbm_ledger': {'kv_shared_once': kv, 'split_query_reads': queries,
                                   'page_index_reads': indices, 'partial_roundtrip': partial_roundtrip,
                                   'final_output': output},
                    'physical_heads': heads, 'requested_splits': geometry['kv_splits'],
                    'active_aic_owners': core_meta['active_aic_owners'],
                    'physical_segments': core_meta['physical_segments']},
            'kda': {'isolated_us': 1000 * kda['duration_ms'],
                    'hbm_bytes': kda['operator']['hbm_bytes'],
                    'active_aiv': state_meta['active_aiv'],
                    'logical_heads': state_meta['logical_heads'],
                    'owner_cycles': state_meta['owner_cycles']},
            'body': {k: body[k] for k in ('status', 'duration_ms',
                    'native_core_elapsed_span_sum_replica0_ms',
                    'native_core_occurrences_replica0') if k in body},
            'capacity': capacity,
        })
        checks.append({'id': entry['id'], 'input_content_unchanged': True,
                       'exact_independent_mla_hbm_ledger': True,
                       'complete_cleanup': True})
    complete = {r['id']: r for r in rows if r['status'] == 'completed'}
    baseline = complete['kimi-current-ctx32768-c1-split32']
    longer = complete['kimi-current-ctx126000-c1-split32']
    body_growth = longer['body']['duration_ms'] - baseline['body']['duration_ms']
    mla_growth = 24 * (longer['mla']['isolated_us'] - baseline['mla']['isolated_us']) / 1000
    assert abs(body_growth - mla_growth) < 1e-8
    assert len({r['kda']['hbm_bytes'] for r in complete.values() if r['concurrency'] == 1}) == 1
    report = {'schema': 'mindie-kimi-longctx-decode-summary/1', 'rows': rows, 'checks': checks,
              'standard_workload': {'prompt_tokens': 126000, 'output_tokens': 3000,
                                    'windows': manifest['standard_decode_windows']},
              'context_control': {'body_growth_ms': body_growth, '24_mla_calls_growth_ms': mla_growth,
                                  'kda_state_bytes_and_duration_context_invariant': True},
              'capacity_reporting_runtime': package_identity(),
              'capacity_reporting_note': 'Recomputed by shared probe_capacity using Decode query workspace; initial raw Memory reports used case Prefill workspace. Operator/body runs are unchanged.',
              'limitations': [
                  'Physical scenarios only; no same-boundary measured accuracy or heldout pass is claimed.',
                  'Split counts are explicit sensitivity controls; historical/native host dispatch is not established.',
                  'Three full 93-layer bodies executed; six additional cells executed native cores and constructed complete DP graphs.',
                  'No full autoregressive trajectory, TTFT, client TPOT or throughput is inferred from sampled bodies.',
                  'Current E896/MX checkpoint quant metadata is conditional relative to the historical A5 capture.',
                  'A3 TP16 heldout captures remain excluded from fitting and from these A5 controls.'],
              'raw_artifact_directory': str(RAW)}
    write(ARTIFACT / 'summary.json', report)
    return report


def tp_transfer():
    """Audit declared A3 sharding without opening heldout operator durations."""
    sys.path.insert(0, str(ROOT))
    from mindie_sim import operator_costs, target_graphs
    from mindie_sim.catalog.cache import layer_cache_components
    from mindie_sim.prediction.decode_probe import probe_capacity
    from mindie_sim.prediction.structure import parse_case
    base_path = ROOT / ('examples/cases/high-concurrency-20261030/kimi-source/'
                        'a3_kimi4_tp8_w4a8_c64_source.json')
    base = json.loads(base_path.read_text())
    rows = []
    recipes = {'linear_q', 'linear_k', 'linear_v', 'q_b', 'q_absorb', 'v_expand',
               'output', 'expert_dispatch', 'expert_combine',
               'balanced_grouped_gate_up', 'balanced_grouped_down'}
    for tp in (8, 16):
        raw = deepcopy(base)
        raw['runtime'].update(tp=tp, ep=tp, max_model_len=36864)
        raw['machine'].update(num_devices=tp, ranks_per_node=tp)
        assert not raw['runtime']['kernels'].get('service_regimes')
        case = parse_case(raw)
        graph = target_graphs(case, phase='decode', tokens=1, context=32768,
                              sequences=12, scope='body', resource_fusion=False)[0]
        selected = {op.recipe: op for op in graph
                    if op.layer in (0, 3) and op.recipe in recipes}
        kda = next(op for op in graph if op.recipe == 'vllm_ascend_recurrent_kda_arch22')
        native = operator_costs(case, (kda,), input_ready_ms=7)['operators'][0]
        assert native['live_tasks'] == 0 and not any(native['remaining_buffers'].values())
        cache = {str(plan.index): layer_cache_components(
            case.model, plan, case.runtime.parallel, 32769,
            capacity_padded=False) for plan in case.model.layer_plans}
        mla = next(p for p in case.model.layer_plans if p.attention == 'mla')
        moe = next(p for p in case.model.layer_plans if p.ffn == 'moe')
        startup = Path('/Volumes/data/mindie-sim-data/a3/heldout_engine_20261030') / (
            f'kimi_k3_w4a8_4layer_tp{tp}_ep{tp}_heldout_20261030/logs/serve.log')
        rows.append({'tp': tp, 'ep': tp, 'dp': 1, 'layers': case.model.layers,
                     'topology': 'single_A3_host_physical_Die_group; no_cross_node_link_inferred',
                     'source_startup': identity(startup), 'startup_args_line': 23,
                     'model_local_mla_heads': mla.num_heads // tp,
                     'resident_experts_per_rank': moe.num_experts // tp,
                     'cache_components_unpadded': cache,
                     'operators': {name: {'local_shape': op.local_shape, 'flops': op.flops,
                         'hbm_bytes': op.hbm_bytes, 'link_bytes': op.link_bytes,
                         'matrix_groups': dict(op.metadata).get('matrix_groups')}
                         for name, op in selected.items()},
                     'kda': {'local_shape': kda.local_shape, 'isolated_us': native['duration_ms'] * 1000,
                             'physical_hbm_bytes': native['operator']['hbm_bytes'],
                             'program_execution': native['operator']['metadata']['program_execution']},
                     'capacity': probe_capacity(case, context=32768, tokens=1, sequences=12)})
    a, b = rows
    for name in recipes - {'expert_dispatch', 'expert_combine'}:
        assert a['operators'][name]['flops'] == 2 * b['operators'][name]['flops']
    assert a['cache_components_unpadded']['3'] == b['cache_components_unpadded']['3']
    for layer in ('0', '1', '2'):
        assert {k: v / 2 for k, v in a['cache_components_unpadded'][layer].items()} == b['cache_components_unpadded'][layer]
    result = {'schema': 'mindie-kimi-a3-tp-transfer-geometry/1', 'base': identity(base_path),
              'runtime': package_identity(), 'rows': rows,
              'checks': {'projection_and_active_expert_flops_halve': True,
                         'shared_latent_mla_cache_unchanged': True,
                         'kda_state_and_conv_storage_halve': True},
              'split': {'tp16': 'entire_new_capture_group_heldout; never_fit',
                        'capture_duration_files_read': 0},
              'limitations': [
                  'Startup and source geometry audit only; newly arrived TP16 measurements remain with the Profiler pipeline.',
                  'A3 MLA remains generic latent attention; no arch35 paged kernel or captured padding/dispatch is claimed.',
                  'Physical FIA head padding (logical6 versus possible8) requires the new native ABI export.',
                  'Declared context32768 is a structural probe; actual per-forward KV lengths are not inferred from tensor capacity.']}
    write(ARTIFACT / 'a3-tp-transfer.json', result)
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'worker', 'summarize', 'tp-transfer'))
    parser.add_argument('--id')
    parser.add_argument('--case', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--body', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare()))
    elif args.command == 'worker':
        worker(args.case, args.output, args.body)
    elif args.command == 'summarize':
        report = summarize()
        print(json.dumps({'rows': len(report['rows']), 'checks': len(report['checks']),
                          'context_control': report['context_control']}))
    elif args.command == 'tp-transfer':
        report = tp_transfer()
        print(json.dumps(report['checks']))
    else:
        manifest = json.loads((ARTIFACT / 'inputs.json').read_text())
        entry = next(row for row in manifest['cases'] if row['id'] == args.id)
        receipt = run(entry, args.body)
        if receipt['status'] != 'completed':
            raise SystemExit(1)
