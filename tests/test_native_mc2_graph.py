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
    report = graph_costs(raw, graphs, external_inputs={'analysis_input': 7}, input_ready_ms=7)
    # Hidden32 BF16: one 512B dispatch frame and one512B return frame.
    # Two assignments/source, half local;4 expert records *32B half local.
    # Two physical senders, exactly one source-grouped kernel each direction.
    assert sum(wire(report).values()) == 2 * (512 + 64 + 512)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    ops = graphs[0]
    group_ops = [o for o in ops if 'device_kernel_group' in dict(o.metadata)]
    assert sum(dict(o.metadata)['device_kernel_group']['entry'] for o in group_ops) == 2
    assert sum(dict(o.metadata)['device_kernel_group']['exit'] for o in group_ops) == 2
    unpack = next(o for o in ops if o.op_id == 's0.l0.unpack')
    output_plan = contract(dict(unpack.metadata)['mc2_phase'], 4)
    phase_work = [dict(p.work) for p in output_plan.phases if p.name in {'weighted_reduce', 'publish_output'}]
    reduction = {key: sum(p.get(key, 0) for p in phase_work)
                 for key in set().union(*(set(p) for p in phase_work))}
    assert reduction['mte2_aiv'] == 2 * (4 + 480) # weights + once-framed slot payload
    assert reduction['mte3'] == 32 * 2
    assert reduction['vector'] == (3 * 2 + 2) * 32
    indexed = {o.op_id: o for o in ops}
    send = indexed['s0.l0.pack']
    phase = dict(send.metadata)['mc2_phase']
    ledger = contract(phase, 4)
    assert dict(next(p for p in ledger.phases if p.name == 'send_read_quantize').work)['hbm'] == 2 * 32 * 2
    # There is no false dependency from counting to sending the payload.
    net = indexed['s0.l0.dispatch.payload_network']
    assert net.deps == ('s0.l0.mc2_dispatch.frame_payload',)
    flags = indexed['s0.l0.mc2_dispatch.receive_flags']
    assert 's0.l0.dispatch.payload_network' in flags.deps
    assert 's0.l0.mc2_dispatch.receive_metadata' in flags.deps
    assert min(row['ready_ms'] for row in report['timeline']) >= 7


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
    assert slower['execution']['serviced'] == pytest.approx(report['execution']['serviced'])
