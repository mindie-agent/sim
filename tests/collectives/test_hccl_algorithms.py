"""HCCL collective schedules, directed traffic, and caller lowering.

[Role] Exercise integral mesh, detour and NHR schedules and their routing.
[Invariants] Reduce-scatter and allgather conserve shard contributions. Peer
  slots, remote notifications and frames are charged to their physical owners.
[Timeouts & Locks] Bounded in-memory schedules with no external locks.
[Side-Effect Policy] Read-only inputs and locally constructed Engines.
"""
import random
from copy import deepcopy
from dataclasses import replace
from math import ceil, prod

import pytest
from _support.collective_cases import (
    case, declared, exchange, isolated_calls, per_pair_topology, topology, wire,
)

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.hccl_aicpu import contract as aicpu_contract
from mindie_sim.prediction.hccl_alltoallv import AlltoAllVGroup, bind_local_work, peer_schedule
from mindie_sim.prediction.hccl_alltoallv import contract as alltoallv_contract
from mindie_sim.prediction.hccl_analytic_hierarchy import contract as analytic_contract
from mindie_sim.prediction.hccl_analytic_hierarchy import groups, stages
from mindie_sim.prediction.hccl_analytic_hierarchy import select as select_analytic
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_graph import lower
from mindie_sim.prediction.hccl_hierarchy import contract as hierarchical_contract
from mindie_sim.prediction.hccl_nhr import rounds
from mindie_sim.prediction.hccl_source_shapes import rs_chunks, slice_sizes
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0


def source(collective, template, ranks, payload, **options):
    return select({"expansion": "AICPU_TS", "template": template,
                   "dtype_bytes": 2, "buffer_mode": "separate", **options},
                  collective=collective, ranks=ranks, payload_bytes=payload)


def work(plan, role):
    return sum(dict(phase.work).get(role, 0) for phase in plan.phases)


def exchanges(plan):
    return [phase for phase in plan.phases if phase.exchange]


def test_mesh_chunk_allreduce_keeps_direction_and_integer_peer_ledger():
    s = source("allreduce", "mesh_chunk_twoshot", 4, 2006)
    plan = aicpu_contract(s)
    empty_tail = aicpu_contract(source("allreduce", "mesh_chunk_twoshot", 4, 2), rank=3)
    assert work(empty_tail, "hccl_copy") == 0
    assert not any(next(p for p in empty_tail.phases if p.name == "gather").exchange[3])
    assert slice_sizes(2006, 2, 4) == (502, 502, 502, 500)
    assert work(plan, "hccl_copy") == 2 * 502
    assert work(plan, "hccl_reduce") == 3 * 502
    rs = [phase for phase in exchanges(plan) if phase.name.startswith("rs_step")]
    gather = next(phase for phase in exchanges(plan) if phase.name == "gather")
    assert [phase.native_direction for phase in rs] == ["send", "send", "send"]
    assert gather.native_direction == "read"
    assert gather.exchange[0] == (0, 502, 502, 502)
    assert tuple(gather.exchange[source][0] for source in range(4)) == (0, 502, 502, 500)
    remote = {
        peer: [phase.exchange[0][peer] for phase in rs]
        for peer in range(4) if peer
    }
    assert remote == {1: [166, 168, 168], 2: [168, 168, 166],
                      3: [168, 166, 166]}
    # Charge the outgoing read and incoming read-modify-write once.
    for phase in rs:
        assert dict(phase.work)["hbm"] == sum(phase.exchange[0]) + 2 * sum(
            row[0] for row in phase.exchange)


@pytest.mark.parametrize("collective,template,segments,options,phase_names,direction", [
    ("reduce_scatter", "rs_mesh_chunk", (57 * 1024**2, 7 * 1024**2), {},
     ("step0",), "send"),
    ("allgather", "z_axis_detour_allgather", (50 * 1024**2, 14 * 1024**2),
     {"level1_weights": (6, 2)}, ("ch0", "ch1", "ch2"), "read"),
])
def test_segmented_templates_preserve_order_and_per_peer_bytes(
        collective, template, segments, options, phase_names, direction):
    ranks = 8
    reduce = collective == "reduce_scatter"
    plan = aicpu_contract(source(collective, template, ranks, sum(segments) * (ranks if reduce else 1),
                                 segments=segments, **options))
    assert [dict(p.work)["hccl_copy"] for p in plan.phases if dict(p.work).get("hccl_copy")] == [
        size for size in segments for _ in range(2)]
    assert work(plan, "hccl_reduce") == ((ranks - 1) * sum(segments) if reduce else 0)
    named = {phase.name: phase for phase in exchanges(plan)}
    for segment, size in enumerate(segments):
        phases = [named[f"seg{segment}_{name}"] for name in phase_names]
        assert all(phase.native_direction == direction for phase in phases)
        assert all(sum(map(sum, phase.exchange)) == ranks * sum(phase.exchange[0]) for phase in phases)
        per_rank = size if reduce else size * (ranks - 1)
        assert sum(sum(phase.exchange[0]) for phase in phases) == per_rank
        if not reduce:
            assert all(sum(p.exchange[0][peer] for p in phases) == size for peer in range(1, ranks))
    if reduce:
        assert rs_chunks(segments[0], 2, ranks) == (8536064,) * 6 + (8552448,)
        assert rs_chunks(10 * 1024**2, 2, ranks) == (1495040,) * 6 + (1515520,)
    else:
        assert [p.exchange[0][1] for p in named.values()] == [
            24465408, 20971520, 6991872, 6848512, 5873664, 1957888]


def _chunk_case(ranks, collective, template, **options):
    raw = case(tp=ranks)
    raw["machine"]["device"]["rates"].update(
        hbm=4e12, hccl_copy=1.5e12, hccl_reduce=1e12)
    raw["model"]["num_heads"] = ranks
    raw["model"]["num_kv_heads"] = max(1, ranks // 2)
    raw["runtime"]["kernels"] = {"hccl": {
        collective: {"expansion": "AICPU_TS", "template": template, **options}}}
    return raw


@pytest.mark.parametrize("collective,template,options", [
    ("allgather", "z_axis_detour_allgather", {"segments": (8192.75,), "level1_weights": (6, 2)}),
    ("allgather", "z_axis_detour_allgather", {"segments": (8192,), "level1_weights": (6.9, 2.2)}),
    ("reduce_scatter", "rs_mesh_chunk", {}),
    ("allgather", "z_axis_detour_allgather", {"segments": (8192,)}),
    ("allreduce", "mesh_chunk_twoshot", {"segments": (8192,)}),
    ("allreduce", "mesh_chunk_twoshot", {"level1_weights": (6, 2)}),
    ("allreduce", "mesh_oneshot", {"unused_rate": 1}),
    ("reduce_scatter", "z_axis_detour_reduce_scatter",
     {"segments": (8192,) * 2, "level1_weights": (6, 2)}),
])
def test_template_selection_rejects_invalid_or_unused_parameters(collective, template, options):
    with pytest.raises(ModelingError):
        aicpu_contract(source(collective, template, 4, 4 * 8192, **options))


@pytest.mark.parametrize("ranks,collective,template,payload,factor,options,directions", [
    (4, "allreduce", "mesh_oneshot", 1024, 1.5, {}, ["send"]),
    (4, "allreduce", "mesh_chunk_twoshot", 30, 1.5, {}, ["send"] * 3 + ["read"]),
    (8, "allgather", "z_axis_detour_allgather", 64 * 1024**2, 7,
     {"segments": (50 * 1024**2, 14 * 1024**2), "level1_weights": (6, 2)}, ["read"] * 6),
    (8, "reduce_scatter", "rs_mesh_chunk", 512 * 1024**2, 7 / 8,
     {"segments": (57 * 1024**2, 7 * 1024**2)}, ["send"] * 14),
])
def test_chunk_caller_lowers_directed_work_once(ranks, collective, template, payload,
                                               factor, options, directions):
    raw = _chunk_case(ranks, collective, template, **options)
    parsed = parse_case(raw)
    selected = lower((exchange(payload * factor, family=collective),), parsed.runtime, "bf16")
    report = graph_costs(raw, selected, isolated=False, trace_limit=0)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    network = [dict(op.metadata) for op in selected if op.family == "alltoallv"]
    phases = [dict(op.metadata)["hccl_phase"] for op in selected if "hccl_phase" in dict(op.metadata)]
    assert [p["native_direction"] for p in phases if p["native_direction"] != "local"] == directions
    assert len(network) == len(directions)
    assert sum(sum(map(sum, p["peer_traffic"]["payload_bytes"])) for p in network) == (
        ranks * payload * (ranks - 1 if template == "mesh_oneshot" else factor)
    )


def _nhr_selection(collective='allreduce', ranks=4, size=1024):
    return select({'expansion':'AICPU_TS', 'template':'nhr'},
                  collective=collective, ranks=ranks, payload_bytes=size)


@pytest.mark.parametrize('ranks', (2, 4, 8, 16, 32))
def test_source_cyclic_rounds_deliver_each_contribution_exactly_once(ranks):
    # Run integer data provenance, not timing formulas: every output shard must
    # contain all rank contributions with no duplicated reduction operands.
    state = [{shard:{rank} for shard in range(ranks)} for rank in range(ranks)]
    rs = [rounds(ranks, rank, 'reduce_scatter') for rank in range(ranks)]
    for index in range((ranks-1).bit_length()):
        messages = [(step[index].send_to, shard, state[src][shard].copy())
                    for src, step in enumerate(rs) for shard in step[index].send_slices]
        for dst, shard, contributors in messages:
            assert not state[dst][shard] & contributors
            state[dst][shard] |= contributors
    assert all(state[rank][rank] == set(range(ranks)) for rank in range(ranks))
    state = [{rank:state[rank][rank]} for rank in range(ranks)]
    ag = [rounds(ranks, rank, 'allgather') for rank in range(ranks)]
    for index in range((ranks-1).bit_length()):
        messages = [(step[index].send_to, shard, state[src][shard].copy())
                    for src, step in enumerate(ag) for shard in step[index].send_slices]
        for dst, shard, contributors in messages:
            assert shard not in state[dst]
            state[dst][shard] = contributors
    assert all(len(shards) == ranks for shards in state)
    assert all(v == set(range(ranks)) for shards in state for v in shards.values())
    if ranks >= 4:
        assert rs[0][0].send_to == ranks - 1  # source cyclic ring, not XOR partner1.


@pytest.mark.parametrize('ranks', (2, 4, 8, 16))
def test_rounds_preserve_bytes_and_source_copy_elisions(ranks):
    n = ranks * 128
    ar = aicpu_contract(_nhr_selection(ranks=ranks, size=n))
    rs = aicpu_contract(_nhr_selection('reduce_scatter', ranks, n))
    ag = aicpu_contract(_nhr_selection('allgather', ranks, 128))
    # NHR source has remote in-place reduce: one source read, destination RMW.
    assert work(ar, 'hbm') == 7*n-5*n/ranks
    assert work(rs, 'hbm') == 4*n-n/ranks
    assert work(ag, 'hbm') == 3*n
    assert work(ar, 'hccl_reduce') == work(rs, 'hccl_reduce') == n*(ranks-1)/ranks
    assert work(ar, 'hccl_copy') == n  # first TX half and last RX half skip copy.
    assert work(rs, 'hccl_copy') == n/2+n/ranks
    assert work(ag, 'hccl_copy') == n/ranks+n/2
    for plan, expected in ((ar,2*(ranks-1)*n), (rs,(ranks-1)*n), (ag,(ranks-1)*n)):
        exchanges = [p.exchange for p in plan.phases if p.exchange]
        assert sum(sum(sum(row) for row in matrix) for matrix in exchanges) == expected
        assert all(matrix[i][i] == 0 for matrix in exchanges for i in range(ranks))
        assert all(sum(value>0 for value in row) == 1 for matrix in exchanges for row in matrix)
        # Every physical rank has an equal local work ledger in this domain.
        for rank in range(ranks):
            peer = aicpu_contract(plan.selection, rank)
            assert work(peer, 'hbm') == work(plan, 'hbm')
    phases = {p.name:p for p in ag.phases}
    assert phases['scratch_to_output'].after == phases[f'ag_{(ranks-1).bit_length()-1}_ack'].after
    assert ag.thread_count == 2 and rs.thread_count == ar.thread_count == 1


def test_unimplemented_rank_padding_does_not_silently_become_fractional_mesh():
    with pytest.raises(ModelingError, match='non_power_of_two'):
        aicpu_contract(_nhr_selection(ranks=6, size=768))
    with pytest.raises(ModelingError, match='tail_shards'):
        aicpu_contract(_nhr_selection(ranks=8, size=18))
    with pytest.raises(ModelingError, match='distinct_input'):
        aicpu_contract(replace(_nhr_selection(), buffer_mode='symmetric'))


def _hierarchical_selection(local=2, remote=4, n=1024, **options):
    return select(dict(expansion='AICPU_TS', template='parallel_mesh_nhr_route0',
        topology_levels=2, level0_size=local, max_loop_bytes=n, **options),
        collective='allreduce', ranks=local*remote, payload_bytes=n)


@pytest.mark.parametrize('local,remote', [(2,2),(2,4),(4,2),(4,4),(8,4)])
def test_source_hierarchy_exact_directed_work_and_no_alias_copies(local,remote):
    r, n = local*remote, local*remote*128
    plan = hierarchical_contract(_hierarchical_selection(local,remote,n))
    assert work(plan, 'hbm') == 7*n+2*n/local-5*n/r
    assert work(plan, 'hccl_reduce') == n*(r-1)/r
    assert work(plan, 'hccl_copy') == 2*n/local
    assert sum(sum(map(sum,p.exchange)) for p in plan.phases if p.exchange)==2*n*(r-1)
    log = (remote-1).bit_length()
    assert sum(len(p.signals) for p in plan.phases)==r*(4*(local-1)+3*log)
    assert not any(p.name.startswith(('l1_rs.input_to','l1_rs.scratch_to',
                                      'l1_ag.input_to','l1_ag.scratch_to')) for p in plan.phases)
    assert not any(p.signals for p in plan.phases if '_executor_' in p.name)
    fused = [p for p in plan.phases if dict(p.control_counts).get('remote_notify_fused_with_payload')]
    assert len(fused)==log and all(not p.signals for p in fused)
    # Equal shards retain equal local work for every physical rank.
    for rank in (0,local-1,r-1):
        other=hierarchical_contract(plan.selection,rank)
        assert sum(dict(p.work).get('hbm',0) for p in other.phases)==work(plan, 'hbm')
    for p in plan.phases:
        for src,row in enumerate(p.exchange):
            for dst,value in enumerate(row):
                if value:
                    assert (src//local==dst//local if p.name.startswith('l0_')
                            else src%local==dst%local)


def _hierarchy_case(latency=0, signal_bytes=None):
    raw=declared('AICPU_TS','parallel_mesh_nhr_route0',ranks=8)
    raw['model'].update(layers=1, num_heads=8, num_kv_heads=8)
    raw['machine']['device']['rates']['memory']=1e15
    raw['machine']['links'][0].update(latency_us=latency,capacity_bytes_per_ms=1e12)
    raw['runtime']['kernels']['hccl']['allreduce'].update(
        topology_levels=2,level0_size=2,max_loop_bytes=1048576,signal_bytes=signal_bytes)
    return raw


def test_public_caller_one_invocation_remote_rounds_and_frame_bytes_once():
    raw=_hierarchy_case()
    ops=isolated_calls(raw)
    zero=graph_costs(raw,ops,isolated=False,input_ready_ms=7,include_submission=True)
    slow = deepcopy(raw)
    slow['machine']['links'][0]['latency_us'] = 5
    delayed=graph_costs(slow,ops,isolated=False,input_ready_ms=7,include_submission=True)
    # R=8,L=2,H=4: 6 payload + 10 notification propagation waves/call.
    assert delayed['duration_ms']-zero['duration_ms']==pytest.approx(2*16*.005,abs=1e-6)
    assert delayed['execution']['submission']['eager_invocations']==2
    assert delayed['execution']['live_tasks']==delayed['execution']['live_programs']==0
    assert sum(wire(delayed).values())==2*2*7*64
    assert all(row['ready_ms']>=7 for row in delayed['timeline'])
    framed=_hierarchy_case(signal_bytes=4)
    framed['machine']['links'][0].update(transaction_bytes=16,protocol_overhead_bytes=4)
    with_signals=graph_costs(framed,isolated_calls(framed),isolated=False)
    framed['runtime']['kernels']['hccl']['allreduce']['signal_bytes']=None
    payload_only=graph_costs(framed,isolated_calls(framed),isolated=False)
    assert sum(wire(with_signals).values())-sum(wire(payload_only).values())==2*80*20


def _analytic_declaration(dims=(2, 2, 2), order=None, signal_bytes=None, protocol='ack_data'):
    return {'rank_dimensions': list(dims),
            'levels': [{'reduce_scatter': 'mesh', 'allgather': 'mesh', 'protocol': protocol}
                       for _ in dims], 'signal_bytes': signal_bytes,
            **({'rank_order': order} if order is not None else {})}


def _analytic_selection(dims=(2, 2, 2), order=None, **kw):
    return select_analytic(_analytic_declaration(dims, order, **kw), collective='allreduce',
                  ranks=prod(dims), payload_bytes=prod(dims) * 128)


@pytest.mark.parametrize("dims,order", [
    ((2, 2, 2), None), ((2, 3, 2), None), ((3, 2, 4), None),
    ((2, 2, 2), [7, 0, 5, 2, 3, 4, 1, 6]),
])
def test_hierarchy_shards_receive_each_contribution_once(dims, order):
    selected = _analytic_selection(dims, order)
    ranks = range(selected.ranks)
    state = {rank: {shard: {rank} for shard in ranks} for rank in ranks}
    shard_bytes = selected.payload_bytes // selected.ranks
    matrices = {p.name: p.exchange for p in analytic_contract(selected).phases if p.exchange}
    for level, kind, input_size, output_size in stages(selected):
        expected = [[0] * len(ranks) for _ in ranks]
        for members in groups(selected, level):
            prior = {rank: state[rank] for rank in members}
            assert all(len(shards) * shard_bytes == input_size for shards in prior.values())
            if kind == "rs":
                shards = sorted(prior[members[0]])
                width = len(shards) // len(members)
                for position, dst in enumerate(members):
                    state[dst] = {}
                    for shard in shards[position * width:(position + 1) * width]:
                        contributions = [prior[src][shard] for src in members]
                        union = set().union(*contributions)
                        assert len(union) == sum(map(len, contributions))
                        state[dst][shard] = union
            else:
                union = {shard: sources for shards in prior.values() for shard, sources in shards.items()}
                assert len(union) == sum(map(len, prior.values()))
                for dst in members:
                    state[dst] = union
            for dst in members:
                assert len(state[dst]) * shard_bytes == output_size
                for src in members:
                    if src != dst:
                        expected[src][dst] = output_size if kind == "rs" else input_size
        assert matrices[f"l{level}_{kind}.exchange"] == tuple(map(tuple, expected))
    assert all(set(shards) == set(ranks) for shards in state.values())
    assert all(sources == set(ranks) for shards in state.values() for sources in shards.values())
    assert sum(sum(map(sum, m)) for m in matrices.values()) == 2 * selected.payload_bytes * (len(ranks) - 1)


def test_local_hbm_reduce_work_and_remote_signals_have_independent_owners():
    s = _analytic_selection((2, 3, 2))
    p = analytic_contract(s)
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
    assert work(p, 'hbm') == oracle_hbm
    assert work(p, 'hccl_reduce') == oracle_reduce == s.payload_bytes * (s.ranks - 1) / s.ranks
    assert work(p, 'hccl_copy') == oracle_copy
    assert sum(len(x.signals) for x in p.phases) == 4 * s.ranks * sum(d - 1 for d in s.rank_dimensions)
    for rank in range(s.ranks):
        assert work(analytic_contract(s, rank), 'hbm') == oracle_hbm
    seen = set()
    for phase in p.phases:
        assert set(phase.after) <= seen
        seen.add(phase.name)
    q = analytic_contract(_analytic_selection((2, 3, 2), protocol='data_signal'))
    assert work(q, 'hbm') == work(p, 'hbm')
    assert work(q, 'hccl_control') * 2 == work(p, 'hccl_control')
    assert sum(len(x.signals) for x in q.phases) * 2 == sum(len(x.signals) for x in p.phases)


def _analytic_case(dims=(2, 2, 2), *, signal_bytes=None, protocol='ack_data'):
    ranks = prod(dims)
    raw = declared(ranks=ranks)
    raw['model'].update(layers=1, num_heads=ranks, num_kv_heads=ranks, hidden_size=ranks * 8)
    raw['machine']['device'].update(launch_us=3,
        rates={'memory': 1e12, 'vector': 1e12, 'hccl_copy': 1e12,
               'hccl_reduce': 1e12, 'hccl_control': 1e9},
        resource_units={'hccl_copy': 1, 'hccl_reduce': 1, 'hccl_control': 1},
        work_units={'hccl_copy': 'bytes', 'hccl_reduce': 'bytes', 'hccl_control': 'ops'})
    raw['runtime']['kernels'] = {'analytic_hierarchy': {'allreduce':
        _analytic_declaration(dims, signal_bytes=signal_bytes, protocol=protocol)}}
    s = _analytic_selection(dims)
    raw['machine']['links'] = [
        {'name': f'l{level}_{i}', 'members': list(members),
         'capacity_bytes_per_ms': (1e6, 1e5, 1e4)[level],
         'latency_us': level + 1, 'semantics': 'per_port'}
        for level in range(3) for i, members in enumerate(groups(s, level))]
    return raw


def _analytic_ops(raw):
    result = {}
    for rank, ops in target_graphs(raw, phase='decode', tokens=1, context=16, scope='body').items():
        keep = [op for op in ops if 'analytic_hierarchy_group' in dict(op.metadata)]
        ids = {op.op_id for op in keep}
        result[rank] = tuple(replace(op, deps=tuple(d for d in op.deps if d in ids)) for op in keep)
    return result


def test_three_level_composition_keeps_one_causal_boundary():
    raw = _analytic_case(signal_bytes=4)
    ops = _analytic_ops(raw)
    report = graph_costs(raw, ops, isolated=False, input_ready_ms=7, include_submission=True)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert report['execution']['submission']['eager_invocations'] == 2
    assert all(row['ready_ms'] >= 7 for row in report['timeline'])
    for op in ops[0]:
        meta = dict(op.metadata)
        assert 'native_kind' not in meta
        assert 'hccl_source_group' not in meta
        assert 'source_revision' not in meta['hccl_selection']
        assert meta['hccl_selection']['selection_basis'] == 'explicit_analytic_composition'
    # Two calls, N=128B: 2*N*(R-1) data plus 96 directed 4B signals per call.
    # Full-duplex resources charge TX and RX service separately; wire is TX
    # only here, because every message traverses one link.
    assert sum(v for key, v in wire(report).items() if ':tx:' in key) == 2 * (2 * 128 * 7 + 96 * 4)
    faster = deepcopy(raw)
    for link in faster['machine']['links']:
        link['latency_us'] = 0
    fast = graph_costs(faster, ops, isolated=False, input_ready_ms=7, include_submission=True)
    # RS+AG at each level: ACK, payload, DATA propagation; local waits add none.
    assert report['duration_ms'] - fast['duration_ms'] == pytest.approx(2 * 6 * .006, abs=1e-8)


def _alltoallv_selection(matrix, mode='write', rank=0):
    return select({'expansion':'AICPU_TS','template':'mesh_'+mode,'dtype_bytes':1},
                  collective='alltoallv', ranks=len(matrix), payload_bytes=sum(matrix[rank]))


def _run_alltoallv(matrix, mode='write', *, ready=None, memory=1e15, copy=None, control=None,
        bandwidth=1, latency=0, topo=None, ranks=None):
    count=len(matrix)
    ranks=tuple(range(count)) if ranks is None else ranks
    topo = topo or per_pair_topology(count, capacity=bandwidth, latency_us=latency)
    resources = list(topo.resources())
    contexts = {}
    for rank in ranks:
        rates = {k: v for k, v in (('hbm', memory), ('hccl_copy', copy), ('hccl_control', control))
                 if v is not None}
        names={role:f'{role}:{rank}' for role in rates}
        resources.extend(Resource(names[role],rate) for role,rate in rates.items())
        contexts[rank]=RecipeContext(None,{},names,{names[k]:v for k,v in rates.items()},1,
                                    resource_units={name:1 for name in names.values()})
    engine=Engine(resources,trace_limit=30000)
    initial={rank:engine.submit(f'ready{rank}',release=(ready or [0]*count)[index])
             for index,rank in enumerate(ranks)}
    group=AlltoAllVGroup(_alltoallv_selection(matrix,mode),tuple(map(tuple,matrix)),ranks,topo)
    done=group.start_participants(engine,'native',initial,contexts)
    engine.advance(until=tuple(done.values()),first=False)
    assert all(event.done for event in done.values())
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    for plan in group.plans():
        for role, resource in contexts[ranks[plan.rank]].resource_names.items():
            assert engine.serviced.get(resource, 0) == pytest.approx(work(plan, role))
    return done,engine,group


@pytest.mark.parametrize('mode',('read','write'))
def test_nonuniform_integer_matrix_keeps_diagonal_and_copy_direction(mode):
    rng=random.Random(0xA2A)
    for count in (2,3,8,19):
        matrix=tuple(tuple(rng.randrange(5)*16 for _ in range(count)) for _ in range(count))
        for rank in range(count):
            plan=alltoallv_contract(_alltoallv_selection(matrix,mode,rank),matrix,rank)
            out=sum(matrix[rank])-matrix[rank][rank]
            incoming=sum(row[rank] for row in matrix)-matrix[rank][rank]
            diagonal=matrix[rank][rank]
            assert work(plan,'hbm')==2*diagonal+(3*out+incoming if mode=='read' else out+3*incoming)
            assert work(plan,'hccl_copy')==diagonal+(out if mode=='read' else incoming)
            assert sum(p.send_bytes for p in plan.peers)==out
            assert sum(p.recv_bytes for p in plan.peers)==incoming
            assert plan.thread_count==min(16,count-1)+1
            assert all(p.owner_domain==() or p.owner_domain[0]>0 for p in plan.phases)


@pytest.mark.parametrize('mode',('read','write'))
@pytest.mark.parametrize('count', (2, 17, 18))
def test_full_mesh_latency_scales_with_slot_reuse(mode,count):
    schedules = [{peer: (slot, turn) for peer, slot, turn in peer_schedule(count, rank)}
                 for rank in range(count)]
    for rank, schedule in enumerate(schedules):
        assert set(schedule) == set(range(count)) - {rank}
        assert len(set(schedule.values())) == count - 1
        assert all(schedules[peer][rank][1] == turn for peer, (_, turn) in schedule.items())
    matrix = tuple(tuple(int(src != dst) for dst in range(count)) for src in range(count))
    done,engine,_=_run_alltoallv(matrix,mode,latency=5)
    assert [e.time for e in done.values()]==pytest.approx([ceil((count-1)/16)*1.005]*count,abs=1e-9)
    assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==count*(count-1)
    services=[row for row in engine.trace if row['task'].endswith(':hop0:service')]
    assert len(services)==count*(count-1)


@pytest.mark.parametrize('mode',('read','write'))
def test_zero_peer_does_not_create_remote_barrier_and_receive_only_waits_for_input(mode):
    matrix=((4,0,0),(0,0,16),(0,0,0))
    done,engine,group=_run_alltoallv(matrix,mode,ready=[1,2,9],memory=16,copy=8)
    assert done[0].time==pytest.approx(1.5)
    assert done[1].time>=9 and done[2].time>=9
    wire=next(row for row in engine.trace if row['task'].endswith(':hop0:service'))
    assert wire['start_ms']>=9
    assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==16
    assert work(group.plans()[2],'hccl_copy')==(0 if mode=='read' else 16)
    empty=((0,0),(0,0))
    done,_,_=_run_alltoallv(empty,mode,ready=[3,17])
    assert done[0].time==3 and done[1].time==17


def test_later_slot_waits_for_postcopy_without_imposing_whole_round_barrier():
    count=18
    matrix=[[0]*count for _ in range(count)]
    # Rank0 left-distance8(slot0) then opposite-distance9(slot0).
    matrix[10][0]=100
    matrix[9][0]=1
    matrix[1][0]=1
    _,engine,_=_run_alltoallv(matrix,'write',memory=100,copy=1,bandwidth=100)
    rows={r['task']:r for r in engine.trace}
    assert rows['native:9->0:hop0:service']['start_ms']>=rows['native:rank0:postcopy_10']['end_ms']
    assert rows['native:1->0:hop0:service']['start_ms']==0
    assert rows['native:rank0:postcopy_1']['end_ms']<rows['native:rank0:postcopy_10']['end_ms']


def test_return_transpose_and_physical_rank_mapping_preserve_framed_hops():
    raw = case(tp=4)
    raw['model'].update(num_heads=8)
    raw['machine']['links']=[
        {'name':'left','members':[1,0],'semantics':'per_pair','capacity_bytes_per_ms':16,
         'transaction_bytes':8,'protocol_overhead_bytes':2,'latency_us':5},
        {'name':'right','members':[0,3],'semantics':'per_pair','capacity_bytes_per_ms':32,
         'transaction_bytes':16,'protocol_overhead_bytes':4,'latency_us':7},
    ]
    topo=topology(raw)
    matrix=((7,9),(21,3))
    for cells in (matrix,tuple(zip(*matrix))):
        _,engine,_=_run_alltoallv(cells,topo=topo,ranks=(1,3))
        # Each of9 and21bytes traverses two hops. Per-hop framing is retained.
        expected=ceil(9/8)*10+ceil(21/8)*10+ceil(9/16)*20+ceil(21/16)*20
        assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==expected
        assert sum(v for k,v in engine.serviced.items() if k.startswith('hbm:'))==4*(9+21)+2*(7+3)


def test_declared_control_rate_changes_only_control_service_not_payload():
    matrix=((0,8),(8,0))
    values=[]
    for scale in (.5,1,2):
        done,engine,group=_run_alltoallv(matrix,'read',memory=100,copy=100,control=10*scale,bandwidth=100)
        values.append(max(e.time for e in done.values()))
        assert dict(group.plans()[0].control_counts)['notify']==6
        assert dict(group.plans()[0].control_counts)['wait']==6
        assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==16
    assert values[0]>values[1]>values[2]
    ctx=RecipeContext(None,{}, {'hbm':'hbm'}, {'hbm':100},1,resource_units={'hbm':1})
    demand,omitted=bind_local_work((('hbm',32),('hccl_copy',16),('hccl_control',4)),ctx)
    assert demand=={'hbm':32} and omitted=={'hccl_copy':16,'hccl_control':4}


@pytest.mark.parametrize('expansion,template',[
    ('AIV','mesh_oneshot'),('AIV','mesh_twoshot'),
    ('AICPU_TS','mesh_oneshot'),('AICPU_TS','mesh_twoshot'),
    ('AICPU_TS','nhr'),('CCU_MS','mesh_oneshot_v1'),('CCU_MS','mesh_oneshot_v2')])
def test_ordinary_caller_preserves_wire_work_once_and_one_invocation(expansion,template):
    raw=declared(expansion,template)
    # The two-shot source partitions these tensors across 60 slices.
    raw['model'].update(layers=1, hidden_size=960)
    raw['machine']['device']['launch_us']=3
    selected=isolated_calls(raw)
    groups={dict(op.metadata)['hccl_source_group'] for op in selected[0]}
    assert len(groups) == 2  # Attention and FFN each own one invocation.
    report=graph_costs(raw,selected,include_submission=True,isolated=True,input_ready_ms=7)
    assert sum(wire(report).values()) == len(groups) * 2 * (2 * raw["model"]["hidden_size"])
    assert report['execution']['submission']['eager_invocations']==2
    assert report['execution']['live_tasks']==report['execution']['live_programs']==0
    assert all(row['ready_ms']>=7 for row in report['timeline'])
    assert report['bounds']['combined_ms']<=report['duration_ms']


@pytest.mark.parametrize("expansion,template,recipe", [
    ("AIV", "mesh", "hccl_aiv_gather_source_phase"),
    ("AICPU_TS", "nhr", "hccl_aicpu_source_phase"),
    ("CCU_SCHED", "mesh_mem2mem_v1", "hccl_ccu_source_phase"),
    ("CCU_SCHED", "mesh_mem2mem_v2", "hccl_ccu_source_phase"),
])
def test_allgather_dispatch_executes_the_declared_expansion(expansion, template, recipe):
    raw = declared(expansion, template)
    raw["runtime"]["kernels"]["hccl"]["allgather"] = raw["runtime"]["kernels"]["hccl"].pop("allreduce")
    parsed = parse_case(raw)
    graph = lower((exchange(64, family="allgather"),), parsed.runtime, parsed.model.activation_dtype)
    report = graph_costs(raw, graph, input_ready_ms=7, isolated=False)
    assert report["duration_ms"] > 0
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert all(row["ready_ms"] >= 7 for row in report["timeline"])
    assert any(row["recipe"] == recipe for row in report["timeline"])
