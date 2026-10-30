"""Three-dimensional contribution oracles and actual public Engine controls."""
from copy import deepcopy
from dataclasses import replace
from math import prod

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.hccl_analytic_hierarchy import contract, groups, select, stages
from mindie_sim.prediction.hccl_signals import SignalGroup
from test_communication_topology import topology
from test_hccl_source_graph import declared
from test_joint_collective_execution import wire


def declaration(dims=(2, 2, 2), order=None, signal_bytes=None, protocol='ack_data'):
    return {'rank_dimensions': list(dims),
            'levels': [{'reduce_scatter': 'mesh', 'allgather': 'mesh', 'protocol': protocol}
                       for _ in dims], 'signal_bytes': signal_bytes,
            **({'rank_order': order} if order is not None else {})}


def selection(dims=(2, 2, 2), order=None, **kw):
    return select(declaration(dims, order, **kw), collective='allreduce',
                  ranks=prod(dims), payload_bytes=prod(dims) * 128)


@pytest.mark.parametrize('dims,order', [
    ((2, 2, 2), None), ((2, 3, 2), None), ((3, 2, 4), None),
    ((2, 2, 2), [7, 0, 5, 2, 3, 4, 1, 6]),
])
def test_all_output_shards_receive_each_contribution_exactly_once(dims, order):
    s = selection(dims, order)
    p = contract(s)
    # Element-shard values carry original contributor sets. Neither phase work
    # nor a duration estimate is used to infer correctness of the reduction.
    state = {r: {i: {r} for i in range(s.ranks)} for r in range(s.ranks)}
    shard_bytes = s.payload_bytes // s.ranks
    exchanges = {p.name: p.exchange for p in p.phases if p.exchange}
    for level, kind, input_size, output_size in stages(s):
        matrix = exchanges[f'l{level}_{kind}.exchange']
        expected = [[0] * s.ranks for _ in range(s.ranks)]
        updated = {}
        for members in groups(s, level):
            width = len(members)
            if kind == 'rs':
                keys = sorted(state[members[0]])
                assert all(set(state[r]) == set(keys) for r in members)
                chunk = len(keys) // width
                for position, dst in enumerate(members):
                    owned = keys[position * chunk:(position + 1) * chunk]
                    updated[dst] = {}
                    for key in owned:
                        contributions = [state[src][key] for src in members]
                        union = set().union(*contributions)
                        assert len(union) == sum(map(len, contributions))
                        updated[dst][key] = union
                    for src in members:
                        if src != dst:
                            expected[src][dst] = len(owned) * shard_bytes
            else:
                combined = {}
                for src in members:
                    assert not set(combined).intersection(state[src])
                    combined.update(state[src])
                    for dst in members:
                        if src != dst:
                            expected[src][dst] = len(state[src]) * shard_bytes
                for dst in members:
                    updated[dst] = dict(combined)
            assert len(state[members[0]]) * shard_bytes == input_size
            assert len(updated[members[0]]) * shard_bytes == output_size
        assert matrix == tuple(map(tuple, expected))
        state = updated
    assert all(set(values) == set(range(s.ranks)) for values in state.values())
    assert all(contributors == set(range(s.ranks))
               for values in state.values() for contributors in values.values())
    assert sum(sum(map(sum, m)) for m in exchanges.values()) == 2 * s.payload_bytes * (s.ranks - 1)


def test_local_hbm_reduce_work_and_remote_signals_have_independent_owners():
    s = selection((2, 3, 2))
    p = contract(s)
    oracle_hbm = oracle_copy = oracle_reduce = 0
    for level, kind, size, output in stages(s):
        width = s.rank_dimensions[level]
        if kind == 'rs':
            oracle_hbm += (5 * width - 3) * output
            oracle_copy += output
            oracle_reduce += (width - 1) * output
        else:
            oracle_hbm += 2 * width * size
            oracle_copy += size
    work = lambda plan, role: sum(dict(x.work).get(role, 0) for x in plan.phases)
    assert work(p, 'hbm') == oracle_hbm
    assert work(p, 'hccl_reduce') == oracle_reduce == s.payload_bytes * (s.ranks - 1) / s.ranks
    assert work(p, 'hccl_copy') == oracle_copy
    assert sum(len(x.signals) for x in p.phases) == 4 * s.ranks * sum(d - 1 for d in s.rank_dimensions)
    for rank in range(s.ranks):
        assert work(contract(s, rank), 'hbm') == oracle_hbm
    seen = set()
    for phase in p.phases:
        assert set(phase.after) <= seen
        seen.add(phase.name)
    q = contract(selection((2, 3, 2), protocol='data_signal'))
    assert work(q, 'hbm') == work(p, 'hbm')
    assert work(q, 'hccl_control') * 2 == work(p, 'hccl_control')
    assert sum(len(x.signals) for x in q.phases) * 2 == sum(len(x.signals) for x in p.phases)


def raw_case(dims=(2, 2, 2), *, signal_bytes=None, protocol='ack_data'):
    ranks = prod(dims)
    raw = declared(ranks=ranks)
    raw['model'].update(num_heads=ranks, num_kv_heads=ranks, hidden_size=ranks * 8)
    raw['machine']['device'].update(launch_us=3,
        rates={'memory': 1e12, 'vector': 1e12, 'hccl_copy': 1e12,
               'hccl_reduce': 1e12, 'hccl_control': 1e9},
        resource_units={'hccl_copy': 1, 'hccl_reduce': 1, 'hccl_control': 1},
        work_units={'hccl_copy': 'bytes', 'hccl_reduce': 'bytes', 'hccl_control': 'ops'})
    raw['runtime']['kernels'] = {'analytic_hierarchy': {'allreduce':
        declaration(dims, signal_bytes=signal_bytes, protocol=protocol)}}
    s = selection(dims)
    raw['machine']['links'] = [
        {'name': f'l{level}_{i}', 'members': list(members),
         'capacity_bytes_per_ms': (1e6, 1e5, 1e4)[level],
         'latency_us': level + 1, 'semantics': 'per_port'}
        for level in range(3) for i, members in enumerate(groups(s, level))]
    return raw


def isolated(raw):
    result = {}
    for rank, ops in target_graphs(raw, phase='decode', tokens=1, context=16, scope='body').items():
        keep = [op for op in ops if 'analytic_hierarchy_group' in dict(op.metadata)]
        ids = {op.op_id for op in keep}
        result[rank] = tuple(replace(op, deps=tuple(d for d in op.deps if d in ids)) for op in keep)
    return result


def test_public_three_level_composition_single_boundary_late_input_and_no_native_claim():
    raw = raw_case(signal_bytes=4)
    ops = isolated(raw)
    report = graph_costs(raw, ops, isolated=False, input_ready_ms=7, include_submission=True)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert report['execution']['submission']['eager_invocations'] == 4
    assert all(row['ready_ms'] >= 7 for row in report['timeline'])
    for op in ops[0]:
        meta = dict(op.metadata)
        assert 'native_kind' not in meta
        assert 'hccl_source_group' not in meta
        assert 'source_revision' not in meta['hccl_selection']
        assert meta['hccl_selection']['selection_basis'] == 'explicit_analytic_composition'
    # Four calls, N=128B: 2*N*(R-1) data plus 96 directed 4B signals per call.
    # Full-duplex resources charge TX and RX service separately; wire is TX
    # only here, because every message traverses one link.
    assert sum(v for key, v in wire(report).items() if ':tx:' in key) == 4 * (2 * 128 * 7 + 96 * 4)
    faster = deepcopy(raw)
    for link in faster['machine']['links']:
        link['latency_us'] = 0
    fast = graph_costs(faster, ops, isolated=False, input_ready_ms=7, include_submission=True)
    # RS+AG at each level: ACK, payload, DATA propagation; local waits add none.
    assert report['duration_ms'] - fast['duration_ms'] == pytest.approx(4 * 6 * .006, abs=1e-8)


@pytest.mark.parametrize('level', [0, 1, 2])
def test_every_level_bandwidth_changes_duration_without_changing_physical_work(level):
    raw = raw_case((2, 2, 3), signal_bytes=4)
    ops = isolated(raw)
    results = []
    for scale in (.5, 1, 2):
        varied = deepcopy(raw)
        for link in varied['machine']['links']:
            if link['name'].startswith(f'l{level}_'):
                link['capacity_bytes_per_ms'] *= scale
        results.append(graph_costs(varied, ops, isolated=False))
    assert results[0]['duration_ms'] > results[1]['duration_ms'] > results[2]['duration_ms']
    assert results[0]['execution']['serviced'] == pytest.approx(results[1]['execution']['serviced'])
    assert results[1]['execution']['serviced'] == pytest.approx(results[2]['execution']['serviced'])


def test_receiver_incast_and_control_frame_accounting_on_three_level_public_graph():
    raw = raw_case((2, 2, 3), signal_bytes=4)
    ops = isolated(raw)
    reports = {}
    for semantics in ('per_port', 'per_pair'):
        varied = deepcopy(raw)
        for link in varied['machine']['links']:
            link.update(semantics=semantics, latency_us=0)
            if link['name'].startswith('l2_'):
                link['capacity_bytes_per_ms'] = 1000
        reports[semantics] = graph_costs(varied, ops, isolated=False)
    assert reports['per_port']['duration_ms'] > reports['per_pair']['duration_ms']
    service = reports['per_port']['execution']['serviced']
    # Level2 has three ranks/group. Each rank receives two 16B shards in RS
    # and AG, plus two 4B ACKs and two completion signals per primitive.
    assert service['link:l2_0:rx:port0'] == pytest.approx(4 * (2 * 2 * 16 + 2 * 2 * 2 * 4))
    assert sum(v for key, v in wire(reports['per_port']).items() if ':tx:' in key) == pytest.approx(
        sum(wire(reports['per_pair']).values()))
    assert all(r['execution']['live_tasks'] == r['execution']['live_programs'] == 0
               for r in reports.values())


def test_optional_controller_service_uses_declared_rate_and_preserves_unpriced_work():
    raw = raw_case()
    ops = isolated(raw)
    measured = []
    for rate in (1e5, 2e5):
        varied = deepcopy(raw)
        varied['machine']['device']['rates']['hccl_control'] = rate
        measured.append(graph_costs(varied, ops, isolated=False))
    assert measured[0]['duration_ms'] > measured[1]['duration_ms']
    assert measured[0]['execution']['serviced'] == pytest.approx(measured[1]['execution']['serviced'])
    for field in ('rates', 'resource_units', 'work_units'):
        raw['machine']['device'][field].pop('hccl_control')
    lower_bound = graph_costs(raw, ops, isolated=False)
    assert lower_bound['duration_ms'] < measured[1]['duration_ms']
    assert lower_bound['execution']['live_tasks'] == 0


def test_hbm_scan_changes_local_shard_copy_and_reduce_service_only():
    raw = raw_case()
    ops = isolated(raw)
    reports = []
    for factor in (.5, 1, 2):
        varied = deepcopy(raw)
        varied['machine']['device']['rates']['memory'] = 1e8 * factor
        reports.append(graph_costs(varied, ops, isolated=False))
    assert reports[0]['duration_ms'] > reports[1]['duration_ms'] > reports[2]['duration_ms']
    assert reports[0]['execution']['serviced'] == pytest.approx(reports[1]['execution']['serviced'])
    assert reports[1]['execution']['serviced'] == pytest.approx(reports[2]['execution']['serviced'])


def test_one_late_physical_rank_delays_only_its_explicit_remote_notification_group():
    raw = raw_case((2, 2, 3), signal_bytes=4)
    plan = contract(selection((2, 2, 3), signal_bytes=4))
    signal = next(p for p in plan.phases if p.name == 'l2_rs.ack_signal')
    topo = topology(raw)
    eng = Engine(topo.resources())
    ready = {r: eng.submit(f'r{r}', release=7 if r == 8 else 0) for r in range(12)}
    done = SignalGroup(tuple(range(12)), topo, signal.signals, 4).start_participants(
        eng, 'level2_ack', ready)
    eng.advance(until=tuple(done.values()), first=False)
    assert all(done[r].time >= 7.003 for r in (0, 4, 8))
    assert all(done[r].time < .01 for r in range(12) if r not in (0, 4, 8))
    assert eng.summary()['live_tasks'] == eng.summary()['live_programs'] == 0
