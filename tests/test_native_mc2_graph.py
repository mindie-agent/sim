"""Ordinary native MC2 calls checked against framed-row and causal oracles."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.mc2_phase import contract
from test_joint_collective_execution import cohort_case, wire


def case(window=8):
    raw = cohort_case(2)
    raw['model']['layers'] = 1
    raw['machine']['device'].update(
        rates={'memory': 1e9, 'vector': 1e9, 'mte2_aic': 1e9,
               'mte2_aiv': 1e9, 'mte3': 1e9},
        resource_units={'vector': 4, 'mte2_aiv': 4, 'mte3': 4})
    raw['machine']['links'][0].update(capacity_bytes_per_ms=1e6, latency_us=5)
    raw['runtime']['kernels']['native_moe_protocol'] = {
        'kind': 'arch35_fullmesh', 'max_inflight_per_rank': window}
    raw['runtime']['routing'] = {'policy': 'balanced'}
    return raw


def test_ordinary_frontend_frames_each_row_count_and_return_once():
    raw = case()
    graphs = target_graphs(raw, phase='decode', tokens=1, context=128, scope='body')
    report = graph_costs(
        raw, graphs, external_inputs={'analysis_input': 7},
        input_ready_ms=7, isolated=False, compact=True)
    # Hidden32 BF16: one 512B dispatch frame and one512B return frame.
    # Two assignments/source, half local;4 expert records *32B half local.
    # Two physical senders, exactly one source-grouped kernel each direction.
    assert sum(wire(report).values()) == 2 * (512 + 64 + 512)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert report['execution']['work']['link_bytes'] == 2176
    ops = graphs[0]
    native = [o for o in ops
              if o.recipe == 'arch35_fullmesh_resource_phase']
    # One layer, one dispatch and one combine call. No semantic phase is a
    # separately scheduled Work operator.
    assert len(native) == 2
    group_ops = [o for o in ops if 'device_kernel_group' in dict(o.metadata)]
    assert sum(dict(o.metadata)['device_kernel_group']['entry'] for o in group_ops) == 2
    assert sum(dict(o.metadata)['device_kernel_group']['exit'] for o in group_ops) == 2
    indexed = {o.op_id: o for o in ops}
    dispatch = indexed['s0.l0.dispatch']
    combine = indexed['s0.l0.unpack']
    assert dict(dispatch.metadata)['mc2_phase']['kind'] == 'dispatch'
    assert dict(combine.metadata)['mc2_phase']['kind'] == 'combine'
    assert dispatch.deps == ('s0.l0.norm2', 's0.l0.topk')
    assert indexed['s0.l0.expert_up'].deps == ('s0.l0.dispatch',)
    # The one native program still exposes the source phase plan: payload
    # framing does not wait for the independent count/status branch, and the
    # receiver flag check waits both a real payload arrival and metadata.
    dispatch_plan = contract(dict(dispatch.metadata)['mc2_phase'], 4)
    phases = {phase.name: phase for phase in dispatch_plan.phases}
    assert dict(phases['send_read_quantize'].work)['hbm'] == 2 * 32 * 2
    assert phases['payload_exchange'].after == ('frame_payload',)
    assert 'count_status_exchange' not in phases['payload_exchange'].after
    assert {'payload_exchange', 'receive_metadata'} <= set(
        phases['receive_flags'].after)
    assert {'s0.l0.expert_down', 's0.l0.dispatch', 's0.l0.topk'} <= set(combine.deps)
    # Semantic phase details remain attributable without a second schedule.
    output_plan = contract(dict(combine.metadata)['mc2_phase'], 4)
    phase_work = [dict(p.work) for p in output_plan.phases
                  if p.name in {'weighted_reduce', 'publish_output'}]
    reduction = {key: sum(p.get(key, 0) for p in phase_work)
                 for key in set().union(*(set(p) for p in phase_work))}
    assert reduction['mte2_aiv'] == 2 * (4 + 480)  # weights + framed slot payload
    assert reduction['mte3'] == 32 * 2
    assert reduction['vector'] == (3 * 2 + 2) * 32
    bridge = report['profile']['native_bridge']
    assert bridge['status'] == 'actual'
    assert bridge['native_launch_groups'] == 2
    assert bridge['transport_tx_bytes'] == bridge['transport_rx_bytes'] == 2176
    assert bridge['transport_frames'] == bridge['directed_frames'] == 0
    assert bridge['known_vs_expected'] == 'declared_destination_expectation'
    groups = {group['op_id']: group for group in bridge['operators']}
    dispatch_group = groups['s0.l0.dispatch']
    combine_group = groups['s0.l0.unpack']
    dispatch_name = 'group:s0.l0.dispatch:r0'
    combine_name = 'group:s0.l0.unpack:r0'
    assert dispatch_group['launch_mode'] == 'cohort'
    assert dispatch_group['payload_packet_bytes'][dispatch_name] == 512
    assert dispatch_group['link_by_phase_by_rank'][dispatch_name][
        'payload_exchange'] == {'link_hccs': 512.0}
    assert dispatch_group['link_by_phase_by_rank'][dispatch_name][
        'count_status_exchange'] == {'link_hccs': 64.0}
    assert combine_group['link_by_phase_by_rank'][combine_name][
        'payload_exchange'] == {'link_hccs': 512.0}
    # The same dispatched work ledger is public; no second private Engine ran.
    dispatch_work = dict(dispatch_group['work_by_rank'][dispatch_name])
    assert sum(value for pipe, value in dispatch_work.items()
               if 'link' in pipe) == 576
    combine_work = dict(combine_group['work_by_rank'][combine_name])
    assert sum(value for pipe, value in combine_work.items()
               if 'link' in pipe) == 512
    steps = report['profile']['steps_by_replica']
    assert set(steps) == {0, 1}
    assert all(step['start_ms'] == 7 for step in steps.values())
    assert min(entry['ready_ms']
               for entry in report['profile']['by_operator'].values()) >= 7
    assert all(step['device_end_ms'] == step['end_ms']
               for step in steps.values())


def test_one_host_and_inclusive_device_boundary_per_native_call():
    raw = case()
    raw['machine']['device']['launch_us'] = 3
    raw['runtime']['kernels']['invocations'] = [
        {'match': {'kernel': 'MoeDistributeDispatchV3'}, 'startup_ms': .002, 'minimum_ms': .030},
        {'match': {'kernel': 'MoeDistributeCombineV3'}, 'startup_ms': .002, 'minimum_ms': .030}]
    graphs = target_graphs(raw, phase='decode', tokens=1, context=128, scope='body')
    # Isolate complete groups while keeping actual caller phase ordering.
    selected = {}
    for rank, ops in graphs.items():
        group_ops = [o for o in ops if 'device_kernel_group' in dict(o.metadata)]
        names = {o.op_id for o in group_ops}
        selected[rank] = tuple(replace(o, deps=tuple(d for d in o.deps if d in names))
                               for o in group_ops)
    report = graph_costs(raw, selected, include_submission=True, isolated=False)
    assert report['execution']['submission']['eager_invocations'] == 4
    assert all(v == pytest.approx(.006) for k,v in report['execution']['serviced'].items()
               if k.startswith('host@'))
    assert report['duration_ms'] >= .063


@pytest.mark.parametrize('scale', [.5, 1, 2])
def test_transport_window_and_rate_preserve_source_work(scale):
    raw = case()
    raw['machine']['links'][0]['capacity_bytes_per_ms'] *= scale
    report = predict(raw, scope='body', tokens=1, context=128, sequences=1)
    assert report['status'] == 'modeled'
    assert sum(wire(report).values()) == 2176
    slow = deepcopy(raw)
    slow['machine']['links'][0]['capacity_bytes_per_ms'] *= .5
    slower = predict(slow, scope='body', tokens=1, context=128, sequences=1)
    assert slower['body_forward_ms'] > report['body_forward_ms']
    # A real source-shaped flag retry adds physical attempts when arrivals are
    # slower; the successful body is still issued once.  Wire and source body
    # must not shrink.
    for role, amount in report['execution']['serviced'].items():
        assert slower['execution']['serviced'].get(role, 0.0) >= amount - 1e-6


def test_ordinary_selector_uses_common_moe_comm_config_without_kernel_entry():
    from mindie_sim.prediction.moe_comm_selection import (
        ALLGATHER, MC2, MoeCommConfig,
    )
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig

    raw = case()
    raw['runtime']['kernels'].pop('native_moe_protocol')
    raw['machine']['architecture'] = 'A5'
    raw['runtime'].update(
        max_num_seqs=64, max_num_batched_tokens=8192,
        graph_buckets=[1, 2, 4, 8, 16, 32, 64],
        metadata={'arch_family': 'A5',
                  'additional_config': {'enable_fused_mc2': 0}})
    parsed = parse_case(raw)
    features = FeatureConfig.from_runtime(
        parsed.runtime, parsed.model, machine=parsed.machine)
    config = MoeCommConfig.from_features(
        features, ep=parsed.runtime.parallel.ep,
        top_k=parsed.model.top_k or 0,
        experts=parsed.model.num_experts or 0)
    assert config is not None
    assert config.mode == 'dynamic' and config.capacity == 64
    assert config.select(1) == MC2
    assert config.select(1024) == ALLGATHER
    graphs = target_graphs(
        raw, phase='decode', tokens=1, sequences=[1, 1], scope='body')
    indexed = {op.op_id: op for op in graphs[0]}
    dispatch = indexed['s0.l0.dispatch']
    combine = indexed['s0.l0.unpack']
    assert dict(dispatch.metadata)['mc2_phase']['kind'] == 'dispatch'
    assert dict(combine.metadata)['mc2_phase']['kind'] == 'combine'
    assert dict(dispatch.metadata)['moe_comm_selection']['mode'] == 'dynamic'
    assert indexed['s0.l0.expert_up'].deps == ('s0.l0.dispatch',)
    assert 's0.l0.expert_down' in combine.deps
