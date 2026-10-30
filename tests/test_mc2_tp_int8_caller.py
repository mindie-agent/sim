"""Captured Qwen shapes through the real MC2 and INT32 grouped caller chain."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from test_mc2_per_token_int8 import case as small_case, graphs as small_graphs

ROOT = Path(__file__).resolve().parents[1]


def qwen_case(concurrency, layers=40):
    path = ROOT / ('examples/cases/high-concurrency-20261030/qwen35-hccl-conditional/'
                   f'qwen35-35b-a5-tp4-c{concurrency}-hccl-source.json')
    raw = json.loads(path.read_text())
    raw['model']['layers'] = layers
    if 'pp_layer_partition' in raw['runtime']:
        raw['runtime']['pp_layer_partition'] = [layers]
    raw['runtime']['kernels'].update(
        native_moe_protocol={'kind': 'arch35_fullmesh', 'max_inflight_per_rank': 4,
                             'dispatch_quant': 'per_token_int8'},
        moe_token_partition={'kind': 'vllm_ascend_mc2_tp_slice',
                             'gather_output_materialization': 'per_rank_copy' if concurrency == 128 else 'view'})
    return raw


def assert_layer(ops, layer, concurrency):
    prefix = f's0.l{layer}.'
    router, topk, send, received, gate, activation, down = (
        ops[prefix + name] for name in ('router', 'topk', 'pack', 'dispatch',
                                        'expert_up', 'expert_act', 'expert_down'))
    rows = concurrency * 8
    assert router.local_shape[0] == concurrency * 4
    assert topk.local_shape[0] == concurrency
    assert dict(send.metadata)['mc2_phase']['source_tokens'] == concurrency
    assert 'activation_scales' in received.returns
    m = dict(received.metadata)
    assert m['activation_payload_shape'] == (rows, 2048)
    assert m['activation_scale_shape'] == (rows,)
    assert m['activation_scale_format'] == 'fp32' and m['activation_scale_bytes'] == rows * 4
    assert gate.deps == (received.op_id,)
    assert dict(gate.metadata)['activation_payload_producer'] == received.op_id
    assert prefix + 'expert_up.quantize' not in ops
    assert dict(gate.metadata)['output_format'] == 'int32'
    assert activation.recipe == 'grouped_int32_dequant_swiglu'
    assert activation.deps == (gate.op_id, received.op_id)
    assert dict(activation.metadata)['input_row_scale_producer'] == received.op_id
    assert 'activation_scales' in activation.returns
    assert dict(activation.metadata)['output_format'] == 'int8'
    assert dict(activation.metadata)['activation_scale_bytes'] == rows * 4
    assert down.deps == (activation.op_id,)
    assert dict(down.metadata)['activation_scale_producer'] == activation.op_id
    assert dict(down.metadata)['output_format'] == 'bf16'
    # Gate quant is transport-owned; down quant survives inside DequantSwigluQuant.
    assert ops[prefix + 'shared.gate_up.quantize'].family == 'quantize'
    assert ops[prefix + 'shared.down.quantize'].family == 'quantize'
    gather = ops[prefix + 'mc2_tp_gather']
    assert dict(gather.metadata)['hccl_selection']['payload_bytes'] == concurrency * 2048 * 2
    assert any(dict(o.metadata).get('device_kernel_group', {}).get('kernel') ==
               'HCCL_AIV_allgather_mesh' for o in ops.values() if o.layer == layer)


@pytest.mark.parametrize('concurrency', [4, 64, 128])
def test_all_40_real_qwen_layers_bind_local_mc2_and_typed_int32_caller(concurrency):
    raw = qwen_case(concurrency); before = deepcopy(raw)
    ops = {o.op_id: o for o in target_graphs(raw, phase='decode', tokens=4,
            sequences=concurrency, context=4096, scope='body')[0]}
    assert raw == before
    for layer in range(40):assert_layer(ops, layer, concurrency)
    copies = [o for o in ops.values() if '.mc2_gather_copy' in o.op_id]
    assert len(copies) == (160 if concurrency == 128 else 0)


def test_real_qwen_one_layer_waits_for_received_scales_and_gather():
    raw = qwen_case(4, layers=1)
    graphs = target_graphs(raw, phase='decode', tokens=4, sequences=4, context=128, scope='body')
    result = graph_costs(raw, graphs, input_ready_ms=7, external_inputs={'analysis_input': 7})
    assert_layer({o.op_id: o for o in graphs[0]}, 0, 4)
    timeline = {r['op_id']: r for r in result['timeline']}
    dispatch, gate, activation = (timeline['s0.l0.' + k] for k in ('dispatch', 'expert_up', 'expert_act'))
    assert gate['ready_ms'] >= dispatch['complete_ms']
    assert activation['ready_ms'] >= max(dispatch['complete_ms'], gate['complete_ms'])
    gather_entry = timeline['s0.l0.mc2_tp_gather.hccl.entry']
    assert gather_entry['ready_ms'] >= timeline['s0.l0.unpack']['complete_ms']
    assert min(r['ready_ms'] for r in result['timeline']) >= 7
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert not any(result['execution']['live_allocated'].values())


def test_idle_source_explicit_receiver_contract_is_not_source_shape():
    raw = small_case()
    kernels = deepcopy(raw.runtime.kernels)
    kernels['lowerings'] = {'grouped_int8_caller': 'ascend_grouped_int32_dequant_swiglu'}
    raw = replace(raw, runtime=replace(raw.runtime, kernels=kernels))
    graphs = small_graphs(raw, (0, 3))
    ops = {o.op_id: o for o in graphs[0]}
    received = ops['s0.l0.dispatch']; meta = dict(received.metadata)
    assert received.local_shape[0] == 0
    assert meta['activation_payload_shape'] == (3, 2048)
    assert meta['activation_scale_shape'] == (3,) and meta['activation_scale_bytes'] == 12
    assert ops['s0.l0.expert_act'].deps == ('s0.l0.expert_up', received.op_id)
    result = graph_costs(raw, graphs, input_ready_ms=7, external_inputs={'analysis_input': 7})
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
