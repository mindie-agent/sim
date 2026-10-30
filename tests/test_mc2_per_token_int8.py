"""Ordinary INT8 MC2 consumes source-produced FP32 row scales exactly once."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.mc2_phase import contract
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from test_native_mc2_graph import case as base_case
from test_mc2_mx_dispatch import protocol_only
from test_joint_collective_execution import wire


def case(dispatch_quant='per_token_int8', width=2048):
    raw = base_case()
    raw['model'].update(hidden_size=width, head_dim=width // 4,
                        intermediate_size=width, moe_intermediate_size=512)
    raw['machine']['device']['compute_peaks']['int8'] = 1e9
    raw['runtime']['kernels']['native_moe_protocol']['dispatch_quant'] = dispatch_quant
    parsed = parse_case(raw)
    quant = QuantSpec('explicit-w8a8', weight_format='int8', expert_weight_format='int8',
                      activation_format='int8', scale_format='fp32', scale_bytes=4,
                      exclusions=('router', 'norm'))
    return replace(parsed, model=replace(parsed.model, quant=quant))


def graphs(raw, sequences=(1, 1)):
    return target_graphs(raw, phase='decode', tokens=1, sequences=sequences,
                         context=128, scope='body')


def test_real_frontend_uses_received_row_scale_once_and_preserves_down_quantizer():
    raw = case()
    original = deepcopy(raw)
    selected, plain = graphs(raw), graphs(case('none'))
    ops, old = ({o.op_id: o for o in g[0]} for g in (selected, plain))
    assert raw == original
    assert 's0.l0.expert_up.quantize' not in ops and 's0.l0.expert_up.quantize' in old
    assert ops['s0.l0.expert_down.quantize'] == old['s0.l0.expert_down.quantize']
    up = ops['s0.l0.expert_up']; meta = dict(up.metadata)
    assert meta['activation_format'] == 'int8'
    assert meta['activation_group_size'] == 2048
    assert meta['activation_scale_shape'] == (2,)
    assert meta['activation_scale_format'] == 'fp32' and meta['activation_scale_bytes'] == 8
    assert up.deps == (meta['activation_payload_producer'],) == (meta['activation_scale_producer'],)
    assert (up.flops, up.hbm_bytes, up.mte2_bytes, up.mte3_bytes) == (
        old[up.op_id].flops, old[up.op_id].hbm_bytes, old[up.op_id].mte2_bytes, old[up.op_id].mte3_bytes)
    assert dict(ops['s0.l0.expert_down'].metadata)['output_format'] == 'bf16'
    phase = dict(ops['s0.l0.mc2_dispatch.send_read_quantize'].metadata)['mc2_phase']
    plan = contract(phase, 4)
    assert plan.layout.payload_bytes == 2048 and plan.layout.scale_bytes == 4
    assert plan.layout.packet_bytes == 2560 and plan.layout.materialized_bytes == 2064
    work = dict(next(p for p in plan.phases if p.name == 'send_read_quantize').work)
    # Source: FP32 cast, abs, max tree, multiply, and three INT8-path casts.
    assert work['vector'] == 2 * (7 * 2048 - 1)
    assert work['mte2_aiv'] == work['hbm'] == 2 * 2048 * 2
    result = graph_costs(raw, selected, input_ready_ms=7, external_inputs={'analysis_input': 7})
    assert min(x['ready_ms'] for x in result['timeline']) >= 7
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0


@pytest.mark.parametrize('sequences', [(1, 0), (0, 3), (1, 3)])
def test_idle_and_unequal_dp_keep_exact_integer_scales_and_return_transpose(sequences):
    raw = case(); declared = graphs(raw, sequences)
    report = graph_costs(raw, protocol_only(declared), isolated=False)
    # Per assignment: INT8 row+FP32scale+triple -> 5*512B; BF16 return->9*512B.
    # topk2 and 2 ranks, one half of assignment traffic remote; count4*32/source.
    assert sum(wire(report).values()) == sum(sequences) * (2560 + 4608) + 128
    vector = 0
    for rank, ops in declared.items():
        for op in ops:
            phase = dict(op.metadata).get('mc2_phase', {})
            if phase.get('phase') == 'send_read_quantize':
                assert phase['source_assignments'] == sequences[rank] * 2
                send = next(p for p in contract(phase, 4).phases if p.name == 'send_read_quantize')
                vector += dict(dict(send.steps)['quantize'])['vector']
            if op.op_id == 's0.l0.expert_up':
                meta = dict(op.metadata)
                assert meta['activation_scale_shape'] == (op.local_shape[0],)
                assert meta['activation_scale_bytes'] == op.local_shape[0] * 4
    assert vector == sum(sequences) * 2 * (7 * 2048 - 1)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_late_route_blocks_payload_and_scale_before_actual_cube_consumer():
    raw = case(); declared = graphs(raw)
    declared[0] = tuple(replace(o, deps=o.deps + ('late_route',))
                        if o.op_id == 's0.l0.pack' else o for o in declared[0])
    report = graph_costs(raw, declared, isolated=False,
                         external_inputs={'analysis_input': 0, 'late_route': 17})
    timeline = {x['id']: x for x in report['timeline']}
    recv = timeline['r0:s0.l0.dispatch']; up = timeline['r0:s0.l0.expert_up']
    assert recv['ready_ms'] >= 17 and up['ready_ms'] >= recv['complete_ms']
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_per_block_scales_cannot_be_hidden_by_per_row_transport():
    raw = case()
    incompatible = replace(raw, model=replace(raw.model,
        quant=replace(raw.model.quant, activation_group_size=128)))
    with pytest.raises(ModelingError, match='matching_int8_fp32_per_row'):
        graphs(incompatible)


def test_one_native_invocation_per_dispatch_or_combine_not_per_protocol_phase():
    raw = case()
    kernels = deepcopy(raw.runtime.kernels)
    kernels['invocations'] = [
        {'match': {'kernel': 'MoeDistributeDispatchV3'}, 'startup_ms': .002, 'minimum_ms': .030},
        {'match': {'kernel': 'MoeDistributeCombineV3'}, 'startup_ms': .002, 'minimum_ms': .030}]
    raw = replace(raw, runtime=replace(raw.runtime, kernels=kernels),
                  machine=replace(raw.machine, dies=tuple(replace(d, launch_us=3) for d in raw.machine.dies)))
    report = graph_costs(raw, protocol_only(graphs(raw)), include_submission=True, isolated=False)
    assert report['execution']['submission']['eager_invocations'] == 4
    host = {k: v for k,v in report['execution']['serviced'].items() if k.startswith('host@')}
    assert host and all(v == pytest.approx(.006) for v in host.values())
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


@pytest.mark.parametrize('role', ['vector', 'hbm', 'mte2_aiv', 'mte3', 'link'])
def test_per_token_work_responds_to_declared_physical_resources(role):
    raw = case(); times, services = [], []
    for factor in (.5, 1, 2):
        if role == 'link':
            changed = replace(raw, machine=replace(raw.machine,
                links=(replace(raw.machine.links[0], capacity_bytes_per_ms=100 * factor),)))
        else:
            dies = tuple(replace(d, rates=tuple({**dict(d.rates), role: 1e7 * factor}.items()))
                         for d in raw.machine.dies)
            changed = replace(raw, machine=replace(raw.machine, dies=dies))
        report = graph_costs(changed, protocol_only(graphs(changed)), isolated=False)
        times.append(report['duration_ms']); services.append(report['execution']['serviced'])
    assert times[0] > times[1] > times[2]
    assert services[0] == pytest.approx(services[1])
    assert services[1] == pytest.approx(services[2])
