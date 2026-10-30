"""Two CCU endpoint owners, complete payload and real notification arrival."""
from copy import deepcopy
from dataclasses import replace
from math import ceil

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_ccu_two_die import build, contract, channels, TEMPLATE, SERVICES
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from test_prediction_system import case


def channel_map(ranks, *, swap=False):
    # Explicit cyclic directed channel assignment, not a model's rank grouping.
    return [[None if src == dst else int(((dst - src) % ranks) >= ranks // 2) ^ swap
             for dst in range(ranks)] for src in range(ranks)]


def declaration(ranks, *, signal_bytes=8):
    return {'expansion': 'CCU_MS', 'template': TEMPLATE, 'topology': 'TWO_DIE_REGULAR',
            'channel_die_map': channel_map(ranks), 'signal_bytes': signal_bytes}


def selected(ranks=8, size=256):
    return select(declaration(ranks), collective='allgather', ranks=ranks, payload_bytes=size)


def declared(ranks=8, size=256):
    raw = case(tp=ranks)
    raw['model'].update(hidden_size=128, num_heads=16, num_kv_heads=16,
                        intermediate_size=256, vocab_size=size * ranks // 2)
    raw['machine']['device'].update(
        rates={'memory': 1e12, 'vector': 1e12,
               **{role: 1e12 for role in SERVICES}, 'hccl_control': 1e12},
        resource_units={role: 2 for role in SERVICES})
    raw['machine']['links'][0].update(capacity_bytes_per_ms=1e9)
    raw['runtime']['kernels'] = {'hccl': {'allgather': declaration(ranks)}}
    return raw


def isolated(raw):
    graphs = target_graphs(raw, phase='decode', tokens=1, context=16, scope='model')
    result = {}
    for replica, ops in graphs.items():
        selected_ops = [op for op in ops if 'hccl_source_group' in dict(op.metadata)]
        ids = {op.op_id for op in selected_ops}
        result[replica] = tuple(replace(op, deps=tuple(d for d in op.deps if d in ids))
                                for op in selected_ops)
    return result


def totals(plan):
    result = {}
    for phase in plan.phases:
        for role, value in phase.work:
            result[role] = result.get(role, 0) + value
    return result


@pytest.mark.parametrize('ranks', [4, 8, 16])
@pytest.mark.parametrize('size', [2, 384, 4096])
def test_two_shared_input_loads_and_one_self_output_are_not_mem2mem(ranks, size):
    selection = selected(ranks, size)
    plan = contract(selection)
    work = totals(plan)
    assert work['hbm'] == work['ccu_buffer_bytes'] == (ranks + 2) * size
    assert work['ccu_copy'] == 3 * size
    assert work['ccu_control'] == 5 * (ranks - 1) + 4
    assert work['hccl_control'] == 4
    assert sum(sum(map(sum, p.exchange)) for p in plan.phases if p.exchange) == ranks * (ranks - 1) * size
    assert sum(len(p.signals) for p in plan.phases) == 3 * ranks * (ranks - 1)
    assert all(totals(contract(selection, rank)) == work for rank in range(ranks))
    assert plan.scratch_bytes == 0


@pytest.mark.parametrize('ranks', [8, 16])
def test_public_model_caller_frames_payload_and_three_remote_notifications_once(ranks):
    size = 30
    raw = declared(ranks, size)
    raw['machine']['links'][0].update(transaction_bytes=16, protocol_overhead_bytes=4)
    ops = isolated(raw)
    assert len({dict(op.metadata)['hccl_source_group'] for op in ops[0]}) == 1
    result = graph_costs(raw, ops, input_ready_ms=7, trace_limit=10000, isolated=False)
    serviced = result['execution']['serviced']
    assert serviced['link:fabric'] == pytest.approx(ranks * (ranks - 1) * (ceil(size / 16) * 20 + 3 * 20))
    assert serviced['memory@r0:s0'] == pytest.approx((ranks + 2) * size)
    assert serviced['ccu_copy@r0:s0'] == pytest.approx(3 * size)
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert all(event['start_ms'] >= 7 for event in result['events'] if event['demand'])
    copies = [event for event in result['events'] if event['demand'].get('ccu_copy@r0:s0')]
    owned = {owner: sum(event['demand']['ccu_copy@r0:s0'] for event in copies
                        if tuple(event['placement']['ccu_copy@r0:s0']) == (owner,))
             for owner in (0, 1)}
    assert owned == {0: 2 * size, 1: size}


def test_owner_identity_follows_channel_endpoint_map_not_npu_rank_or_aiv_count():
    raw = declared()
    reports = []
    for swap in (False, True):
        value = deepcopy(raw)
        value['runtime']['kernels']['hccl']['allgather']['channel_die_map'] = channel_map(8, swap=swap)
        result = graph_costs(value, isolated(value), isolated=False, trace_limit=10000)
        copies = [event for event in result['events'] if event['demand'].get('ccu_copy@r0:s0')]
        work = {owner: sum(event['demand']['ccu_copy@r0:s0'] for event in copies
                          if tuple(event['placement']['ccu_copy@r0:s0']) == (owner,))
                for owner in (0, 1)}
        assert work[int(swap)] == 512 and work[int(not swap)] == 256
        reports.append(result)
    assert reports[0]['duration_ms'] == pytest.approx(reports[1]['duration_ms'])
    missing = deepcopy(raw)
    missing['machine']['device']['resource_units']['ccu_copy'] = 1
    with pytest.raises(ModelingError, match='two_explicit_units'):
        graph_costs(missing, isolated(missing), isolated=False)


def test_one_active_endpoint_cannot_borrow_idle_ccu_or_vector_owners():
    s = selected(size=256)
    op = OpSpec('load', 'hccl_local', 'decode', 0, 0, (), (), (), (), (), metadata=(
        ('hccl_phase', {'selection': s.to_dict(), 'phase': 'die0_input_to_ccu'}),))
    units = {'hbm': 1, **dict.fromkeys(SERVICES, 2)}
    geometry = Geometry(128, units)
    plan = build(op, geometry)
    rates = {'hbm': 1e12, 'ccu_copy': 1024, 'ccu_buffer_bytes': 1e12, 'ccu_control': 1e12}
    context = RecipeContext(op, {}, {role: role for role in rates}, rates, 128, resource_units=units)
    engine = Engine([Resource(role, rate, units=units[role]) for role, rate in rates.items()])
    ready = engine.submit('input', release=7)
    done = plan.lower(context).start(engine, 'ccu', ready)
    engine.advance(until=(done,))
    assert done.time == pytest.approx(7 + 256 / (1024 / 2))
    assert engine.serviced['ccu_copy'] == 256


def test_pre_and_post_waits_follow_real_signal_arrival_without_an_extra_rtt():
    raw = declared()
    raw['runtime']['kernels']['hccl']['allgather']['signal_bytes'] = None
    baseline = graph_costs(raw, isolated(raw), input_ready_ms=7, isolated=False)
    delayed = deepcopy(raw)
    delayed['machine']['links'][0]['latency_us'] = 2000
    result = graph_costs(delayed, isolated(delayed), input_ready_ms=7, isolated=False)
    # PreSync messages, payload, PostSync messages cross the link once each.
    # NotifyWait is local issue work, not another round trip.
    assert result['duration_ms'] - baseline['duration_ms'] == pytest.approx(6)
    assert result['execution']['serviced'] == pytest.approx(baseline['execution']['serviced'])


@pytest.mark.parametrize('role', ['memory', 'ccu_copy', 'ccu_buffer_bytes', 'ccu_control', 'link'])
def test_physical_capacity_scan_preserves_each_owner_and_work(role):
    raw = declared()
    ops = isolated(raw)
    reports = []
    for factor in (.5, 1, 2):
        value = deepcopy(raw)
        if role == 'link':
            value['machine']['links'][0]['capacity_bytes_per_ms'] = 1000 * factor
        else:
            value['machine']['device']['rates'][role] = 1e6 * factor
        reports.append(graph_costs(value, ops, isolated=False))
    assert reports[0]['duration_ms'] > reports[1]['duration_ms'] > reports[2]['duration_ms']
    assert reports[0]['execution']['serviced'] == pytest.approx(reports[1]['execution']['serviced'])
    assert reports[1]['execution']['serviced'] == pytest.approx(reports[2]['execution']['serviced'])


def test_noncontiguous_physical_rank_order_preserves_channel_map_and_public_serving():
    raw = declared()
    rank_order = [41, 7, 93, 18, 64, 2, 57, 30]
    physical = deepcopy(raw)
    physical['machine']['devices'] = [{'rank': rank} for rank in rank_order]
    physical['machine']['links'][0]['members'] = rank_order
    baseline = graph_costs(raw, isolated(raw), isolated=False)
    remapped = graph_costs(physical, isolated(physical), isolated=False)
    assert remapped['duration_ms'] == pytest.approx(baseline['duration_ms'])
    assert remapped['execution']['serviced'] == pytest.approx(baseline['execution']['serviced'])
    serving = predict(physical)
    assert serving['status'] == 'modeled'
    assert serving['execution']['live_tasks'] == serving['execution']['live_programs'] == 0


def test_regular_but_skew_endpoint_map_is_outside_the_representative_domain():
    s = selected()
    mapping = channel_map(8)
    mapping[0] = [None if value is None else 1 - value for value in mapping[0]]
    # Every row still has 3/4 peers. Rank zero alone would misprice the other
    # ranks' per-owner issue and input/output work, so this needs another model.
    with pytest.raises(ModelingError, match='homogeneous_endpoint_degrees'):
        channels(replace(s, channel_die_map=tuple(map(tuple, mapping))))
    for incompatible in (replace(s, expansion='CCU_SCHED'), replace(s, payload_bytes=4098)):
        with pytest.raises(ModelingError):
            contract(incompatible)
