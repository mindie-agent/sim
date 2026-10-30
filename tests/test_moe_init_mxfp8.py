"""Real ordinary routing producer, source reuse and finite Engine accounting."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.prediction.moe_init_mxfp8 import RECIPE, build, row_runs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]


def case(batch, selected=True, fused=False):
    raw = json.loads((ROOT / 'examples/cases/high-concurrency-20261030/glm-flash-ws9' /
                      f'a5_glm53flash_tp8_c{batch}_p4k.json').read_text())
    bindings = raw['runtime'].setdefault('kernels', {}).setdefault('lowerings', {})
    if selected:
        bindings['moe_init_routing'] = RECIPE
    if fused:
        bindings['native_fusion'] = {
            'implementation': 'declared_native_fusion', 'options': {
                'roles': ['grouped_gate_up_activation_quant'],
                'program_profiles': {'grouped_gate_up_activation_quant': 'ascend_mxfp8_grouped_swiglu_quant'},
                'program_options': {'grouped_gate_up_activation_quant': {'l1_k': 512}}}}
    return raw


def graph(raw, batch):
    return target_graphs(raw, scope='body', phase='decode', tokens=4,
                         context=4096, sequences=batch)[0]


@pytest.mark.parametrize('batch', (16, 32, 64))
def test_full_frontend_replaces_only_pack_and_gate_input_quantizer(batch):
    raw = case(batch)
    old, new = graph(case(batch, selected=False), batch), graph(raw, batch)
    assert raw == case(batch)
    initialized = [op for op in new if op.recipe == RECIPE]
    assert len(initialized) == 42
    assert not any(op.op_id.endswith('.expert_up.quantize') for op in new)
    assert sum(op.op_id.endswith('.expert_down.quantize') for op in new) == 42
    old_by = {op.op_id: op for op in old}
    by_id = {op.op_id: op for op in new}
    for init in initialized:
        config = dict(init.metadata)['moe_init_config']
        assert config == {'tokens': batch * 4, 'topk': 8, 'width': 4096, 'experts': 288}
        matrix = by_id[init.op_id.replace('.pack', '.expert_up')]
        assert matrix.deps == (init.op_id,)
        assert dict(matrix.metadata)['activation_scale_producer'] == init.op_id
        assert matrix.flops == old_by[matrix.op_id].flops
        assert matrix.hbm_bytes == old_by[matrix.op_id].hbm_bytes
        from mindie_sim.structure_contracts import predicted_contract
        contract = predicted_contract(init.to_dict())
        assert contract[2][0] == ((batch * 4, 4096), 'bf16', 'ND')
        assert contract[3][3] == ((batch * 32, 128), 'fp8_e8m0', 'ND')
    for op in new:
        if '.expert_' not in op.op_id and not op.op_id.endswith('.pack'):
            assert op == old_by[op.op_id]


@pytest.mark.parametrize('tokens,topk,units', ((3, 8, 4), (7, 5, 8), (9, 6, 4)))
def test_source_owner_partition_reloads_only_split_tokens(tokens, topk, units):
    per = (tokens * topk + units - 1) // units
    expected = []
    for owner in range((tokens * topk + per - 1) // per):
        assignments = list(range(owner * per, min(tokens * topk, (owner + 1) * per)))
        histogram = {}
        for slot in assignments:
            histogram[slot // topk] = histogram.get(slot // topk, 0) + 1
        assert list(row_runs(tokens * topk, topk, units, owner)) == list(histogram.items())
        expected.extend(histogram.values())
    assert sum(expected) == tokens * topk


@pytest.mark.parametrize('batch', (16, 64))
def test_actual_source_call_conserves_expanded_payload_scale_and_owner_service(batch):
    raw = case(batch)
    op = next(op for op in graph(raw, batch) if op.recipe == RECIPE)
    report = operator_costs(raw, (op,), input_ready_ms=7)['operators'][0]
    physical = report['operator']
    meta = physical['metadata']['program_execution']
    assert meta['quantized_input_rows'] == batch * 4
    assert meta['assignments'] == batch * 4 * 8
    assert meta['output_data_bytes'] == batch * 4 * 8 * 4096
    assert meta['output_scale_bytes'] == batch * 4 * 8 * 128
    assert report['live_tasks'] == 0
    assert not report['service_before_input']
    assert all(abs(value) < 1e-6 for value in report['service_delta'].values())
    assert all(value == 0 for value in report['remaining_buffers'].values())
    # Only source role declarations may charge MTE2; generic MTE2 is replaced.
    assert physical['mte2_bytes'] == 0
    assert physical['metadata']['resource_work']['mte2_aiv'] > batch * 4 * 4096 * 2


def test_source_routing_and_fused_up_keep_single_activation_producer_and_late_dependency():
    raw = case(16, fused=True)
    ops = graph(raw, 16)
    init = next(op for op in ops if op.recipe == RECIPE)
    up = next(op for op in ops if op.layer == init.layer and
              dict(op.metadata).get('fusion_program_profile') == 'ascend_mxfp8_grouped_swiglu_quant')
    assert init.op_id in up.deps
    assert not any(op.op_id.endswith('.expert_up.quantize') for op in ops)
    selected = [init, up]
    external = {dep: 11 for op in selected for dep in op.deps if dep not in {init.op_id, up.op_id}}
    result = graph_costs(raw, selected, external_inputs=external)
    rows = {row['op_id']: row for row in result['timeline']}
    assert rows[init.op_id]['ready_ms'] >= 11
    assert rows[up.op_id]['ready_ms'] >= rows[init.op_id]['complete_ms']
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0


@pytest.mark.parametrize('axis', ('mte2_aiv', 'mte3', 'vector', 'hbm'))
def test_declared_source_resource_slowdown_changes_real_program(axis):
    raw = case(64)
    op = next(op for op in graph(raw, 64) if op.recipe == RECIPE)
    parsed = parse_case(raw)
    results = []
    for multiplier in (.5, 1, 2):
        dies = tuple(replace(die, rates=tuple((key, value * multiplier if key == axis else value)
                                              for key, value in die.rates)) for die in parsed.machine.dies)
        scaled = replace(parsed, machine=replace(parsed.machine, dies=dies))
        row = operator_costs(scaled, (op,))['operators'][0]
        results.append(row['duration_ms'])
    assert results[0] >= results[1] >= results[2]
    assert results[0] > results[2]
