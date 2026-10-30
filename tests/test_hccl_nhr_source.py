"""NHR data-contribution oracles, directed wire totals and real Engine phases."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.hccl_aicpu import build, contract
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_nhr import rounds
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def selection(collective='allreduce', ranks=4, size=1024):
    return select({'expansion':'AICPU_TS', 'template':'nhr'},
                  collective=collective, ranks=ranks, payload_bytes=size)


def amount(plan, role):
    return sum(dict(phase.work).get(role, 0) for phase in plan.phases)


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
    ar = contract(selection(ranks=ranks, size=n))
    rs = contract(selection('reduce_scatter', ranks, n))
    ag = contract(selection('allgather', ranks, 128))
    # NHR source has remote in-place reduce: one source read, destination RMW.
    assert amount(ar, 'hbm') == 7*n-5*n/ranks
    assert amount(rs, 'hbm') == 4*n-n/ranks
    assert amount(ag, 'hbm') == 3*n
    assert amount(ar, 'hccl_reduce') == amount(rs, 'hccl_reduce') == n*(ranks-1)/ranks
    assert amount(ar, 'hccl_copy') == n  # first TX half and last RX half skip copy.
    assert amount(rs, 'hccl_copy') == n/2+n/ranks
    assert amount(ag, 'hccl_copy') == n/ranks+n/2
    for plan, expected in ((ar,2*(ranks-1)*n), (rs,(ranks-1)*n), (ag,(ranks-1)*n)):
        exchanges = [p.exchange for p in plan.phases if p.exchange]
        assert sum(sum(sum(row) for row in matrix) for matrix in exchanges) == expected
        assert all(matrix[i][i] == 0 for matrix in exchanges for i in range(ranks))
        assert all(sum(value>0 for value in row) == 1 for matrix in exchanges for row in matrix)
        # Every physical rank has an equal local work ledger in this domain.
        for rank in range(ranks):
            peer = contract(plan.selection, rank)
            assert amount(peer, 'hbm') == amount(plan, 'hbm')
    phases = {p.name:p for p in ag.phases}
    assert phases['scratch_to_output'].after == phases[f'ag_{(ranks-1).bit_length()-1}_ack'].after
    assert ag.thread_count == 2 and rs.thread_count == ar.thread_count == 1


def execute(s, *, rates=None, ready_ms=7):
    rates = rates or {'hbm':1024, 'link':1e9}
    source = contract(s)
    geom = Geometry(1, {key:1 for key in rates})
    engine = Engine([Resource(key, value) for key, value in rates.items()], trace_limit=1000)
    ready = engine.submit('ready', release=ready_ms)
    events = {}
    unpriced = {}
    for phase in source.phases:
        op = OpSpec(phase.name,'collective','decode',0,0,(),(),(),(),(),metadata=(
            ('hccl_phase', {'selection':s.to_dict(), 'phase':phase.name}),))
        plan = build(op, geom)
        unpriced.update(plan.metadata['unpriced_service_work'])
        ctx = RecipeContext(op, {}, {key:key for key in rates}, rates, 1,
                            resource_units=geom.resource_units)
        deps = tuple(events[key] for key in phase.after) or (ready,)
        start = engine.submit(phase.name+':ready', after=deps)
        local = plan.lower(ctx).start(engine, phase.name, start)
        if phase.exchange:
            wire = engine.submit(phase.name+':wire', {'link':sum(phase.exchange[0])}, after=(start,))
            events[phase.name] = engine.submit(phase.name+':join', after=(local,wire))
        else:
            events[phase.name] = local
    done = engine.submit('done', after=tuple(events.values()))
    engine.advance(horizon=ready_ms-1)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()['live_tasks'] == engine.summary()['live_programs'] == 0
    return done.time-ready_ms, engine, source, unpriced


def test_complete_nhr_uses_real_barriers_and_unpriced_controls_do_not_gate_simulation():
    elapsed, engine, source, unpriced = execute(selection())
    assert elapsed == pytest.approx(5.75)
    assert engine.serviced['hbm'] == 5888
    assert engine.serviced['link'] == 1536
    assert unpriced == {'hccl_control':2, 'hccl_reduce':256, 'hccl_copy':512}
    traces = {t['task']:t for t in engine.trace}
    # AllGather cannot consume a shard before the complete ReduceScatter.
    assert traces['ag_0_exchange:wire']['start_ms'] >= traces['rs_1_exchange:wire']['end_ms']
    timed, _, _, _ = execute(selection(), rates={
        'hbm':1024, 'link':1e9, 'hccl_copy':1e9, 'hccl_reduce':1e9, 'hccl_control':10})
    assert timed == pytest.approx(elapsed+8*0.2)


def test_allgather_final_output_copy_overlaps_last_network_round():
    elapsed, engine, _, _ = execute(selection('allgather'), rates={
        'hbm':1024, 'link':1e9, 'hccl_copy':256})
    # input copy4ms, first exchange2ms, final localcopy8ms overlaps4ms memory
    # exchange. A serialized final copy+exchange would incorrectly take18ms.
    assert elapsed == pytest.approx(14)
    traces = {t['task']:t for t in engine.trace}
    copy = traces['scratch_to_output:body:program:service']
    network = traces['ag_1_exchange:wire']
    assert copy['start_ms'] == network['start_ms']


def test_bandwidth_changes_preserve_work_and_all_source_rounds():
    results = [execute(selection(ranks=8), rates={
        'hbm':1024*scale, 'link':256*scale, 'hccl_copy':512*scale,
        'hccl_reduce':512*scale, 'hccl_control':10*scale}) for scale in (.5,1,2)]
    assert results[0][0] == pytest.approx(2*results[1][0])
    assert results[1][0] == pytest.approx(2*results[2][0])
    assert results[0][1].serviced == results[1][1].serviced == results[2][1].serviced


def test_unimplemented_rank_padding_does_not_silently_become_fractional_mesh():
    with pytest.raises(ModelingError, match='non_power_of_two'):
        contract(selection(ranks=6, size=768))
    with pytest.raises(ModelingError, match='tail_shards'):
        contract(selection(ranks=8, size=18))
    with pytest.raises(ModelingError, match='distinct_input'):
        contract(replace(selection(), buffer_mode='symmetric'))
