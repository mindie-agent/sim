"""Audit retained A3 Qwen FFN contracts through one real target layer.

[Role] Exercise source-bound representation and reduction cutpoint corrections.
[Invariants] Layers are selected from complete model graphs. Native durations
  are read after predictions and never become rates, floors or dispatch inputs.
[Timeouts & Locks] Finite one-layer Engine runs; bounded streaming source reads.
[Side-Effect Policy] Local compact evidence only; no device or remote execution.
"""
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case

OUT = ROOT / 'data/verification/qwen-a3-ffn-source-20261030.json'


def identity(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(),
            'size_bytes': Path(path).stat().st_size}


def probe(raw, query, batch):
    started = time.perf_counter()
    case = parse_case(raw)
    full = target_graphs(case, phase='decode', tokens=query, sequences=batch,
                         context=126976, scope='body')[0]
    stop = next(i for i, op in enumerate(full) if op.layer != 0)
    layer = full[:stop]
    report = graph_costs(case, layer, external_inputs={'analysis_input': 0}, isolated=False)
    priced = operator_costs(case, layer, input_ready_ms=7)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    rows = []
    for row in priced['operators']:
        assert row['live_tasks'] == 0 and not any(row['remaining_buffers'].values())
        assert not row['service_before_input']
        op = row['operator']
        rows.append({key: op[key] for key in ('op_id', 'recipe', 'family', 'dtype', 'local_shape',
                                              'flops', 'hbm_bytes', 'mte2_bytes', 'mte3_bytes', 'link_bytes')}
                    | {'elapsed_us': row['duration_ms'] * 1000,
                       'program': row['program'], 'serviced': row['serviced']})
    terms = layer_weight_terms(case.model.layer_plans[0], case.model)
    shared = sum(term_bytes(term, case.model.quant, case.runtime.parallel)
                 for term in terms if term.category == 'shared_expert')
    return {'full_model_layers': len(case.model.layer_plans), 'selected_layer': 0,
            'query_tokens': query, 'sequence_slots': batch,
            'layer_graph_duration_ms': report['duration_ms'],
            'isolated_operator_sum_ms': priced['operator_sum_ms'],
            'shared_resident_weight_bytes': shared, 'operators': rows,
            'body_declaration': full.default_selection,
            'physical_work': report['execution']['work'],
            'runtime_seconds': time.perf_counter() - started}


def retained_comparison(prediction, path, groups):
    """Compare declared component scopes after all model inputs are fixed."""
    wanted = {line for _, lines, _ in groups for line in lines}
    rows = {}
    with Path(path).open() as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if line > max(wanted):
                break
            if line in wanted:
                rows[line] = row
    operators = {row['op_id']: row for row in prediction['operators']}
    compared = []
    for name, lines, ids in groups:
        observed = sum(float(rows[line]['Duration(us)']) for line in lines)
        estimated = sum(operators['s0.l0.' + op_id]['elapsed_us'] for op_id in ids)
        compared.append({'component': name, 'csv_lines': lines, 'operator_ids': ids,
                         'native_duration_sum_us': observed,
                         'isolated_sim_duration_sum_us': estimated,
                         'relative_error_percent': (estimated / observed - 1) * 100,
                         'native_types': [rows[line]['Type'] for line in lines]})
    return {'source': identity(path), 'components': compared,
            'boundary': 'component elapsed sums, not an additive layer critical path or score',
            'interpretation': 'tensor geometry is retained; compiled dispatch and scalar arguments remain conditional'}


def main():
    source = ROOT / 'examples/cases/longctx-decode-20261030/qwen35-35b-a3-tp4.json'
    corrected = json.loads(source.read_text())
    previous = deepcopy(corrected)
    previous['model']['quant']['category_overrides'].pop('shared_expert')
    previous['runtime']['moe_recipe'] = 'alltoall'
    rows = []
    for label, query, batch in (('retained_q2_b1', 2, 1), ('transfer_q4_b4', 4, 4)):
        rows.append({'scenario': label, 'before': probe(previous, query, batch),
                     'corrected': probe(corrected, query, batch),
                     'accuracy': 'not scored; this probe does not bind whole-layer measured boundaries'})
    control_path = ROOT / 'examples/cases/longctx-decode-20261030/qwen36-a3-tp4.json'
    control = probe(json.loads(control_path.read_text()), 4, 1)
    contracts = json.loads((ROOT / 'tests/fixtures/verification/qwen35-a3-ffn-contract.json').read_text())
    q35_path = contracts['captures'][0]['source']['path']
    q36_path = ('/Volumes/data/mindie-sim-data/a3/2026-10-23/qwen36-rows/prof/p_126976_c1/'
                'rank0_430_20261023224849770_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv')
    common = [('qkvz', ['linear_qkvz']), ('beta_alpha', ['linear_beta_alpha']),
              ('convolution', ['short_convolution']), ('gating', ['delta_gates']),
              ('recurrence', ['decode_recurrent_scan']), ('output_projection', ['linear_output'])]
    q35_groups = [(name, [line], ids) for (name, ids), line in zip(common, [33904,33906,33911,33915,33918,33921])]
    q35_groups += [('shared_ffn', [33934,33935,33936], ['shared.gate_up','shared.swiglu','shared.down']),
                   ('routed_up_act_down', [33930,33931], ['expert_up.quantize','expert_up','expert_act',
                                                        'expert_down.quantize','expert_down'])]
    q36_groups = [(name, [line], ids) for (name, ids), line in zip(common, [45705,45706,45711,45715,45718,45721])]
    q36_groups += [('dense_ffn', [45725,45726,45728,45729,45730],
                    ['gate_up.quantize','gate_up','swiglu','down.quantize','down'])]
    comparisons = {'qwen35': retained_comparison(rows[0]['corrected'], q35_path, q35_groups),
                   'qwen36': retained_comparison(control, q36_path, q36_groups)}
    result = {'schema': 'mindie-qwen-a3-ffn-source-audit/1',
              'input': identity(source), 'control_input': identity(control_path),
              'source_contract': identity(ROOT / 'tests/fixtures/verification/qwen35-a3-ffn-contract.json'),
              'producer': identity(Path(__file__)), 'scenarios': rows, 'qwen36_q4_b1_control': control,
              'retained_comparisons': comparisons,
              'corrections': ['shared expert gate/up/down operands and resident weights use retained BF16',
                              'one token-row TP sum follows routed/shared join'],
              'no_new_parameters': True,
              'unresolved': ['A3 GDN recurrence remains an analytical vector mechanism, without an arch22 source recurrence binding.',
                             'GroupedMatmulSwigluQuant and GroupedMatmul remain an analytical decomposition; no arch35 substitution.',
                             'Native scalar group_list, dispatch and per-call compiled identities are unobserved.',
                             'Finite launch, vector/copy service and routing Cumsum residuals are not fitted from body differences.']}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'scenarios': [
        {key: row[key]['layer_graph_duration_ms'] for key in ('before', 'corrected')} for row in rows],
        'qwen36_control_ms': control['layer_graph_duration_ms']}))


if __name__ == '__main__':
    main()
