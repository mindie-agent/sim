"""Physical MC2 metadata traffic, token owners and ordered copy/reduce controls."""
import pytest

from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.mc2_phase import build, contract
from mindie_sim.prediction.program import RecipeContext


def run_phase(name, *, kind='dispatch', tokens=1, top_k=8, received=8,
              experts=16, ep=8, units=64, copy_units=None, rate=64000):
    raw = dict(kind=kind, width=240, dtype='bf16', source_tokens=tokens,
               top_k=top_k, source_assignments=tokens * top_k if kind == 'dispatch' else received,
               received_assignments=received if kind == 'dispatch' else tokens * top_k,
               experts=experts, ep=ep, phase=name)
    op = OpSpec('phase', 'mc2_protocol', 'decode', 0, 0, (), (), (), (), (),
                metadata=(('mc2_phase', raw),))
    rates = {'hbm': 1e9, 'mte2_aiv': rate, 'mte3': rate, 'vector': rate}
    physical = {key: 1 if key == 'hbm' else units for key in rates}
    if copy_units is not None:
        physical.update(mte2_aiv=copy_units, mte3=copy_units)
    plan = build(op, Geometry(units, physical))
    context = RecipeContext(op, {}, {key: key for key in rates}, rates, units,
                            resource_units=physical)
    engine = Engine([Resource(key, value, units=physical[key]) for key, value in rates.items()],
                    trace_limit=10000)
    done = plan.lower(context).start(engine, 'phase', engine.submit('ready', release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()['live_tasks'] == engine.summary()['live_programs'] == 0
    assert not any(engine.allocated.values())
    assert engine.serviced == pytest.approx({k: plan.work.resources.get(k, 0) for k in rates})
    return done.time - 7, engine, plan, contract(raw, units)


@pytest.mark.parametrize('tokens', [1, 4, 65])
def test_combine_slots_and_reduction_use_output_token_owners(tokens):
    elapsed, engine, plan, _ = run_phase('receive_slots', kind='combine', tokens=tokens)
    assert plan.metadata['active_owner_domain'] == tuple(range(min(tokens, 64)))
    # Each owner reads all eight slots of each output token it owns. A one-token
    # call cannot split these reads across eight idle AIVs.
    assert elapsed == 0  # boundary only; payload read moved into each top-k slot
    elapsed, engine, plan, _ = run_phase('weighted_reduce', kind='combine', tokens=tokens)
    per_token = 8 * (4 + 480) + (3 * 8 + 1) * 240
    assert elapsed == pytest.approx(((tokens + 63) // 64) * per_token / 1000)
    assert engine.serviced['hbm'] == tokens * 8 * (4 + 480)
    assert engine.serviced['vector'] == tokens * (3 * 8 + 1) * 240
    for owner in range(min(tokens, 64)):
        rows = [r for r in engine.trace if f'owner{owner}:' in r['task']]
        weights = next(r for r in rows if r['task'].endswith(':weights'))
        reduction = next(r for r in rows if r['task'].endswith(':tile0:initialize_accumulator'))
        assert weights['end_ms'] <= reduction['start_ms']
    _, publication, _, plan = run_phase('publish_output', kind='combine', tokens=tokens)
    assert publication.serviced['vector'] == tokens * 240
    assert publication.serviced['mte3'] == pytest.approx(tokens * 480)
    phases = {p.name: p for p in plan.phases}
    assert phases['clear_data_flags'].after == ('weighted_reduce',)
    assert phases['publish_output'].after == ('clear_data_flags',)


def test_two_active_aiv_share_their_actual_copy_pipe():
    # AIV0/1 share MTE pipe0 when64 AIV map to32 copy pipes. They cannot borrow
    # the unused31 other pipes. Each output token owns eight480B return slots.
    elapsed, _, _, _ = run_phase('weighted_reduce', kind='combine', tokens=2,
                                 copy_units=32, rate=32000)
    assert elapsed == pytest.approx((2 * 8 * (4 + 480) + 2 * (3 * 8 + 1) * 240) / 1000)


def test_metadata_prefix_broadcast_includes_idle_experts_and_every_aiv():
    aiv, experts, ep, count = 64, 384, 32, 8
    _, _, _, plan = run_phase('prefix_counts', experts=experts, ep=ep)
    phases = {p.name: p for p in plan.phases}
    sums = dict(phases['receive_counts'].work)
    assert sums['mte2_aiv'] == experts * 32
    assert sums['mte3'] == experts * 32 + count * count * 32
    prefix = dict(phases['prefix_counts'].work)
    assert prefix['mte2_aiv'] == count * count * 32
    assert prefix['mte3'] == count * count * 32 + (aiv + 1) * experts * 4 + count * aiv * 32
    ready = dict(phases['prefix_ready'].work)
    assert ready['mte2_aiv'] == ready['mte3'] == count * aiv * 32
    assert dict(phases['receive_metadata'].work)['mte2_aiv'] == aiv * experts * 4
    assert dict(phases['expert_counts'].work)['mte3'] == experts // ep * 8
    # No query is needed for the count/status protocol on an incoming-only rank.
    _, _, empty, _ = run_phase('routing_metadata', tokens=0, received=8)
    assert not any(empty.work.resources.values())
    _, _, counts, _ = run_phase('prefix_ready', tokens=0, received=8)
    assert counts.metadata['active_owner_domain'] == tuple(range(64))


def test_local_materialization_waits_for_read_before_write_and_scales_by_pipe_rate():
    spans = []
    for factor in (.5, 1, 2):
        elapsed, engine, plan, _ = run_phase('receive_local_copy', received=1, rate=64000 * factor)
        # width240: dispatch triple needs two480B regions; output is480+12B.
        assert elapsed == pytest.approx((960 + 492) / (1000 * factor))
        spans.append(elapsed)
        assert plan.metadata['active_owner_domain'] == (0,)
    assert spans[0] > spans[1] > spans[2]
