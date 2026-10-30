"""Independent traffic/work oracles and real Engine controls for HCCL phases."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.hccl_aicpu import build, contract
from mindie_sim.prediction.hccl_contracts import select, legacy_mesh_candidate
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.operators import OpSpec


def selection(collective="allreduce", template="mesh_oneshot", ranks=4, size=1024):
    return select({"expansion": "AICPU_TS", "template": template},
                  collective=collective, ranks=ranks, payload_bytes=size)


def aggregate(plan, role):
    return sum(dict(phase.work).get(role, 0) for phase in plan.phases)


@pytest.mark.parametrize("ranks", (2, 3, 8))
def test_oneshot_keeps_peer_concurrency_and_true_memory_amplification(ranks):
    size = 1024
    plans = [contract(selection(ranks=ranks, size=size), rank) for rank in range(ranks)]
    for rank, plan in enumerate(plans):
        phases = {p.name:p for p in plan.phases}
        assert phases['input_to_scratch'].after == phases['allwrite'].after == ('pre_sync',)
        assert set(phases['post_sync'].after) == {'input_to_scratch', 'allwrite'}
        matrix = phases['allwrite'].exchange
        assert sum(matrix[rank]) == (ranks-1)*size
        assert matrix[rank][rank] == 0
        # 2 local copies (4N), network read+write (2(R-1)N), and
        # R-1 reductions each read src+dst and overwrite dst (3(R-1)N).
        assert aggregate(plan, 'hbm') == (4+5*(ranks-1))*size
        assert aggregate(plan, 'hccl_reduce') == (ranks-1)*size
        assert len(phases['local_reduce'].steps) == ranks-1
        assert plan.scratch_bytes == ranks*size
    assert sum(sum(row) for row in plans[0].phases[2].exchange) == ranks*(ranks-1)*size


@pytest.mark.parametrize("count,ranks", ((17, 4), (2, 8), (128, 3), (1024, 8)))
def test_twoshot_integer_tail_and_empty_rank_oracle(count, ranks):
    s = selection(template='mesh_twoshot', ranks=ranks, size=2*count)
    plans = [contract(s, rank) for rank in range(ranks)]
    chunk = (count+ranks-1)//ranks
    widths = [max(0, min(chunk, count-rank*chunk))*2 for rank in range(ranks)]
    assert sum(widths) == 2*count
    for rank, plan in enumerate(plans):
        phases = {p.name:p for p in plan.phases}
        scatter, gather = phases['scatter'].exchange, phases['gather'].exchange
        assert scatter == tuple(zip(*gather))
        assert sum(scatter[rank]) == 2*count-widths[rank]
        assert sum(row[rank] for row in scatter) == (ranks-1)*widths[rank]
        assert aggregate(plan, 'hccl_copy') == 2*widths[rank]
        assert aggregate(plan, 'hccl_reduce') == (ranks-1)*widths[rank]
        assert phases['ag_pre_sync'].after == ('local_reduce',)
    # Every input contribution traverses RS once, then its reduced shard AG once.
    network = sum(sum(sum(row) for row in p.exchange) for p in plans[0].phases)
    assert network == 2*(ranks-1)*2*count


@pytest.mark.parametrize("ranks", (2, 8))
def test_standalone_allgather_and_reduce_scatter_are_not_ring_aliases(ranks):
    n = 2048
    ag = contract(selection('allgather', 'mesh', ranks, n))
    rs = contract(selection('reduce_scatter', 'mesh', ranks, ranks*n))
    assert ag.thread_count == max(1, ranks-1)
    assert rs.thread_count == ranks
    assert aggregate(ag, 'hbm') == 4*n+2*(ranks-1)*n
    assert aggregate(rs, 'hbm') == 2*n+5*(ranks-1)*n
    assert aggregate(rs, 'hccl_reduce') == (ranks-1)*n
    ag_phases = {p.name:p for p in ag.phases}
    assert ag_phases['pre_sync'].after == ('local_copies',)
    rs_phases = {p.name:p for p in rs.phases}
    assert rs_phases['input_to_output'].after == ('post_sync',)


def execute_local_phase(name, *, factor=1, priced=True, ready_ms=13):
    s = selection(ranks=4, size=1024)
    op = OpSpec('test', 'collective', 'decode', 0, 0, (), (), (), (), (), metadata=(
        ('hccl_phase', {'selection':s.to_dict(), 'phase':name}),))
    rates = {'hbm':1024*factor}
    if priced:
        rates.update(hccl_copy=1024*factor, hccl_reduce=256*factor, hccl_control=10*factor)
    units = {key:1 for key in rates}
    plan = build(op, Geometry(1, units))
    context = RecipeContext(op, {}, {k:k for k in rates}, rates, 1, resource_units=units)
    engine = Engine([Resource(k,v) for k,v in rates.items()], trace_limit=1000)
    ready = engine.submit('ready', release=ready_ms)
    done = plan.lower(context).start(engine, 'local', ready)
    engine.advance(horizon=ready_ms-1)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()['live_tasks'] == engine.summary()['live_programs'] == 0
    assert engine.serviced == pytest.approx({k:plan.work.resources.get(k,0) for k in rates})
    return done.time-ready_ms, plan, engine


def test_serial_reductions_use_declared_service_and_preserve_hbm_lower_bound():
    times = []
    for factor in (.5, 1, 2):
        elapsed, plan, engine = execute_local_phase('local_reduce', factor=factor)
        assert elapsed == pytest.approx(3*max(3,4)/factor)
        times.append(elapsed)
        rows = [r for r in engine.trace if ':reduce_' in r['task']]
        assert rows[0]['end_ms'] <= rows[1]['start_ms'] <= rows[2]['start_ms']
        assert not plan.metadata['unpriced_service_work']
    assert times[0] > times[1] > times[2]
    elapsed, plan, _ = execute_local_phase('local_reduce', priced=False)
    assert elapsed == 9
    assert plan.metadata['unpriced_service_work'] == {'hccl_reduce':3072}
    assert plan.work.resources['hbm'] == 9216
    # Lack of a hardware control parameter stays visible without disabling the
    # physical model or manufacturing a whole-kernel launch constant.
    elapsed, plan, _ = execute_local_phase('pre_sync', priced=False)
    assert elapsed == 0
    assert plan.metadata['unpriced_service_work'] == {'hccl_control':6}


def test_legacy_candidate_has_real_thresholds_but_does_not_override_explicit_dispatch():
    args = dict(collective='allreduce', ranks=2)
    assert legacy_mesh_candidate(**args, expansion='AIV', payload_bytes=128*1024-2) == 'mesh_oneshot'
    assert legacy_mesh_candidate(**args, expansion='AIV', payload_bytes=128*1024) == 'mesh_twoshot'
    assert legacy_mesh_candidate(**args, expansion='AICPU_TS', payload_bytes=8*1024**2) == 'mesh_oneshot'
    # Current old selector's ratio is 8/R^2, not a fabricated 64/R^2.
    assert legacy_mesh_candidate(**args, expansion='AICPU_TS', payload_bytes=16*1024**2) == 'mesh_twoshot'
    assert legacy_mesh_candidate(**args, expansion='AICPU_TS', payload_bytes=16*1024**2+2) == 'mesh_chunk_twoshot'
    explicit = select({'expansion':'AIV', 'template':'mesh_oneshot', 'block_limit':3},
                      **args, payload_bytes=491520)
    assert explicit.template == 'mesh_oneshot' and explicit.block_limit == 3


@pytest.mark.parametrize('collective,template', (('allreduce','mesh_twoshot'), ('reduce_scatter','mesh')))
def test_symmetric_read_reduce_changes_memory_work_not_wire_volume(collective, template):
    s = selection(collective, template, 4, 4096)
    separate = contract(s)
    direct = contract(replace(s, buffer_mode='symmetric'))
    assert [p.exchange for p in separate.phases if p.exchange] == [p.exchange for p in direct.phases if p.exchange]
    assert direct.scratch_bytes == 0
    assert aggregate(separate, 'hccl_reduce') == aggregate(direct, 'hccl_reduce') == 3*1024
    expected_saved = 2*1024+2*3*1024 if collective == 'allreduce' else 2*3*1024
    assert aggregate(separate, 'hbm')-aggregate(direct, 'hbm') == expected_saved
    assert not any(p.name == 'local_reduce' for p in direct.phases)
    exchange = next(p for p in direct.phases if p.name == 'scatter')
    assert dict(exchange.work)['hbm'] == 3*1024+2*3*1024
    if collective == 'allreduce':
        assert next(p for p in direct.phases if p.name == 'ag_pre_sync').after == ('rs_result_to_output',)


def test_allgather_remote_output_removes_only_scratch_copy():
    s = selection('allgather', 'mesh', 8, 256)
    separate, direct = contract(s), contract(replace(s, buffer_mode='remote_output'))
    assert aggregate(separate,'hbm')-aggregate(direct,'hbm') == 512
    assert direct.scratch_bytes == 0
    assert next(p.exchange for p in separate.phases if p.exchange) == next(p.exchange for p in direct.phases if p.exchange)


def test_complete_local_dag_overlaps_copy_and_network_then_folds_serially():
    s = selection(ranks=4, size=1024)
    source = contract(s)
    rates = {'hbm':1024, 'hccl_reduce':256, 'link':512}
    geometry = Geometry(1, {key:1 for key in rates})
    engine = Engine([Resource(k,v) for k,v in rates.items()], trace_limit=1000)
    ready = engine.submit('ready', release=5)
    events = {}
    for phase in source.phases:
        op = OpSpec(phase.name,'collective','decode',0,0,(),(),(),(),(),metadata=(
            ('hccl_phase', {'selection':s.to_dict(), 'phase':phase.name}),))
        plan = build(op, geometry)
        ctx = RecipeContext(op, {}, {k:k for k in rates}, rates, 1, resource_units=geometry.resource_units)
        deps = tuple(events[key] for key in phase.after) or (ready,)
        start = engine.submit(phase.name+':ready', after=deps)
        local = plan.lower(ctx).start(engine, phase.name, start)
        if phase.exchange:
            # Isolated representative rank link service, with the same phase
            # dependency as its endpoint HBM. Full topology is caller-owned.
            wire = engine.submit(phase.name+':wire', {'link':sum(phase.exchange[0])}, after=(start,))
            events[phase.name] = engine.submit(phase.name+':join', after=(local,wire))
        else:
            events[phase.name] = local
    done = engine.submit('done', after=tuple(events.values()))
    engine.advance(horizon=4)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    # Input copy2ms and endpoint HBMs6ms share memory and finish by8ms;
    # link6ms overlaps. Then 3 reductions ×4ms and finalcopy2ms =>22ms.
    assert done.time == pytest.approx(5+22)
    assert engine.serviced == pytest.approx({'hbm':19*1024,'hccl_reduce':3*1024,'link':3*1024})
    trace = {row['task']:row for row in engine.trace}
    assert trace['input_to_scratch:body:program:service']['start_ms'] == 5
    assert trace['allwrite:body:program:service']['start_ms'] == 5
    assert engine.summary()['live_tasks'] == engine.summary()['live_programs'] == 0
