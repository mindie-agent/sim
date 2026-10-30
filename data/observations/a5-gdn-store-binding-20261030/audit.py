"""Inspect complete GDN layers and native role boundaries after fixed bindings.

[Role] Retain physical matrix/state/helper contracts and unpriced residuals.
[Invariants] Native duration sums, activity unions and spans are distinct. No
  gap is a fitted parameter; incomplete dispatch evidence remains conditional.
[Timeouts & Locks] Only bounded first-layer simulation and streamed raw rows.
[Side-Effect Policy] Local diagnostic JSON; no case or runtime parameter edits.
"""
import csv
from copy import deepcopy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('gdn_store', Path(__file__).with_name('produce.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from mindie_sim import target_graphs, operator_costs
from mindie_sim.prediction.hccl_contracts import HcclSelection
from mindie_sim.prediction.hccl_aiv import blocks
from mindie_sim.schema.common import ModelingError


def shapes(text):
    if text == 'N/A':
        return []
    return [tuple(int(v) for v in s.split(',')) if s else () for s in text.strip('"').split(';')]


def matches(op, group):
    native = group['native']
    ins, outs = shapes(group['input_shapes']), shapes(group['output_shapes'])
    if op.recipe in ('linear_qkvz', 'linear_beta_alpha', 'linear_output', 'dense_gate_up', 'dense_down'):
        m, n, k = op.local_shape
        return native in ('QuantBatchMatmulV3', 'MatMulV3', 'MatMulV2') and ins and ins[0] == (m, k) and outs and outs[0] == (m, n)
    return native == {
        'ascend_causal_conv1d_update': 'CausalConv1d',
        'ascend_recurrent_gdn_fp32_state': 'RecurrentGatedDeltaRule',
        'ascend_gdn_decode_gating': 'fused_gdn_gating_kernel',
        'linear_output_norm_gate': '_layer_norm_fwd_1pass_kernel_npu',
        'decode_q_l2norm': 'l2norm_fwd_kernel2', 'decode_k_l2norm': 'l2norm_fwd_kernel2',
        'qkv_concatenate': 'ConcatD', 'core_output_zero': 'ZerosLike',
    }.get(op.recipe, '')


def raw_rows(path, wanted):
    result = {}
    columns = ('Name', 'Type', 'Accelerator Core', 'Block Num', 'Mix Block Num', 'Stream ID',
               'Input Shapes', 'Input Data Types', 'Input Formats', 'Output Shapes', 'Output Data Types', 'Output Formats')
    with path.open(newline='') as source:
        for line, row in enumerate(csv.DictReader(source), 2):
            if line in wanted:
                result[line] = {key: row[key] for key in columns}
            if line >= max(wanted):
                break
    assert result.keys() == wanted
    return result


def main():
    binding = base.load(base.HERE / 'validation.json')
    report = {'scope': 'native_role_and_source_work_audit; not_additive_Body_residual_fit', 'cases': []}
    for case in binding['cases']:
        raw = base.load(Path(case['case']['path']))
        raw['model']['layers'] = 1
        ops = target_graphs(raw, scope='body', phase='decode', tokens=4, context=126000, sequences=4)[0]
        costs = operator_costs(raw, ops)['operators']
        retained = base.load(Path(case['retained_validation']['path']))
        groups = case['native']['groups']
        source = Path(case['native']['source']['path'])
        rows = raw_rows(source, {g['first_locator']['csv_line'] for g in groups})
        record = {'model': case['model'], 'tp': case['tp'], 'case': case['case'],
            'native_source': case['native']['source'], 'first_layer_native': [], 'role_comparisons': [],
            'full_layer_stats': [{'layer': layer['layer_id'], 'span_ms': layer['stats'],
                                 'device_union_ms': layer['activity_ms_stats'],
                                 'uncovered_gap_ms': layer['internal_gap_ms_stats']}
                                for layer in retained['observed']['per_layer_ms']],
            'source_work': [], 'known_boundary_differences': [
                'native state-index Slice [256,4] -> [4,1] exists; default stride1 has no materialization',
                'native post-recurrent Slice selects z heads from [M,64,128]; separate core_output_copy event not identified',
                'native AddRmsNormDynamicQuant is one boundary; generic residual/norm/quant are separate calls',
                'N/A helper shapes use source order plus adjacent shaped recurrent/conv, not binary proof',
                'generic per-channel INT8 weight metadata includes scale and offset storage; native ABI exposes scale only']}
        for group in sorted([g for g in groups if g['first_layer']], key=lambda g: g['first_locator']['csv_line']):
            record['first_layer_native'].append({**group, 'raw_abi': rows[group['first_locator']['csv_line']]})
        record['first_layer_native_sum_us'] = sum(g['per_forward_sum_us'] for g in groups if g['first_layer'])
        for op, cost in zip(ops, costs):
            native = [g for g in groups if not g['first_layer'] and g['attention_kind'] == 'gdn' and matches(op, g)]
            if native:
                count = sum(g['count'] for g in native)
                observed = sum(g['sum_us'] for g in native) / count
                record['role_comparisons'].append({'role': op.recipe, 'local_shape': op.local_shape,
                    'predicted_us': cost['duration_ms']*1000, 'native_mean_us': observed,
                    'relative_error': cost['duration_ms']*1000/observed-1,
                    'native_calls_per_forward': sum(g['calls_per_forward'] for g in native),
                    'native_abi': [{'locator': g['first_locator'], 'fields': rows[g['first_locator']['csv_line']]}
                                   for g in native],
                    'pairing': 'M/N/K tensor match' if op.family in ('gemm', 'gate_up_projection', 'down_projection')
                               else 'source role and complete GDN-layer topology'})
            if op.recipe in ('linear_qkvz', 'linear_beta_alpha', 'linear_output', 'dense_gate_up', 'dense_down',
                             'ascend_causal_conv1d_update', 'ascend_recurrent_gdn_fp32_state'):
                record['source_work'].append({'role': op.recipe, 'operator': cost['operator'],
                    'serviced': cost['serviced'], 'declared_service': cost['declared_service'],
                    'service_delta': cost['service_delta'], 'duration_ms': cost['duration_ms']})
        report['cases'].append(record)
    base.save(base.HERE / 'operator-residuals.json', report)
    hccl_path = base.HERE / 'hccl-validation.json'
    if hccl_path.exists():
        hccl = base.load(hccl_path)
        for case in hccl['cases']:
            for candidate in case['candidates']:
                selection = HcclSelection('AIV', 'allreduce', candidate['template'], case['tp'],
                                          case['payload_bytes'], block_limit=case['native_block_num'])
                try:
                    owners = blocks(selection, 64)
                except ModelingError:
                    owners = None
                candidate['modeled_launch_owners'] = owners
                candidate['native_block_count_matches_modeled_owners'] = owners == case['native_block_num']
        base.save(hccl_path, hccl)


if __name__ == '__main__':
    main()
