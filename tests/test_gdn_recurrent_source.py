"""Native recurrence resource and dependency oracles for real Engine programs."""
import math

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.gdn_recurrent import UB_BYTES, build, head_ownership
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.operators import OpSpec


def run(lengths=(4, 3), *, heads=3, dk=128, dv=128, units=4, rates=None,
        compact=True, contender=None):
    shape = (sum(lengths), heads, dk, dv)
    op = OpSpec("scan", "recurrent_vector", "decode", 0, 0, (), shape, shape, shape, shape,
                dtype="bf16", metadata=(("sequence_tokens", lengths), ("heads", heads),
                     ("key_heads", 1), ("dk", dk), ("dv", dv), ("state_dtype", "fp32")))
    values = {"hbm": 1e12, "mte2_aiv": 1e12, "mte2_aic": 1e12,
              "vector": 1e12, "mte3": 1e12, **(rates or {})}
    counts = {role: units if role != "hbm" else 1 for role in values}
    geometry = Geometry(units // 2 or 1, counts, {"ub": 256 * 1024}, {"ub": units})
    plan = build(op, geometry, compact=compact)
    engine = Engine([Resource(role, value, units=counts[role]) for role, value in values.items()],
                    pools={"ub": 256 * 1024}, pool_units={"ub": units}, trace_limit=20000)
    context = RecipeContext(plan.work.apply(op), {}, {role: role for role in values}, values,
                            units // 2 or 1, resource_units=counts, pool_names={"ub": "ub"},
                            pool_units={"ub": units})
    ready = engine.submit("input", release=3)
    complete = plan.lower(context).start(engine, "scan", ready)
    background = engine.submit("external", contender, after=(ready,)) if contender else None
    engine.advance(horizon=2)
    assert not any(engine.serviced.values())
    engine.advance(until=(complete,))
    if background:
        engine.advance(until=(background,))
    assert not engine._tasks and not engine._promises
    assert not engine.allocations and not any(engine.allocated.values())
    return plan, engine, complete.time - 3


@pytest.mark.parametrize("lengths,dk,dv,units", (
    ((1,), 16, 16, 1), ((4,), 128, 128, 4), ((4, 3), 128, 65, 4),
    ((2, 4, 3), 129, 97, 8), ((8, 8), 32, 257, 4), ((4, 4, 4), 256, 129, 4),
))
def test_native_state_tiles_conserve_payload_and_resident_qkv_work(lengths, dk, dv, units):
    heads = 3
    plan, engine, _ = run(lengths, heads=heads, dk=dk, dv=dv, units=units)
    ak, av = math.ceil(dk / 16) * 16, math.ceil(dv / 16) * 16
    raw_step = 8192 // ak
    value_step = (raw_step if raw_step <= 256 else 255) // 16 * 16
    slices = [min(value_step, dv - offset) for offset in range(0, dv, value_step)]
    incoming = scalar = outgoing = vector = 0
    for q in lengths:
        for _ in range(heads):
            incoming += dk * dv * 4 + q * (2 * dk + dv) * 2
            scalar += q * (4 + 2 * len(slices))
            outgoing += q * (dk * dv * 4 + dv * 2)
            vector += q * ak + q * 2 * ak + q * 3 * ak + q * av
            for n in slices:
                vector += ak * n  # GM state -> resident UB copy, once per tile
                vector += q * (ak*n + (2*ak-1)*n + 2*n + 2*ak*n
                               + ak*n + (2*ak-1)*n + n)
    expected = {"mte2_aiv": incoming, "mte3": outgoing,
                "hbm": incoming + outgoing + scalar, "vector": vector}
    for role, amount in expected.items():
        assert plan.work.resources[role] == amount
        assert engine.serviced[role] == pytest.approx(amount, rel=1e-11)
    assert engine.serviced["mte2_aic"] == 0
    assert engine.peak_allocated["ub"] <= UB_BYTES * units
    assert all(event["start_ms"] >= 3 for event in engine.trace if event["demand"])


def test_state_write_overlaps_query_dot_but_single_output_slot_is_not_reused():
    _, engine, duration = run((1,), heads=1, dk=16, dv=16, units=1,
                             rates={"vector": 1000, "mte3": 1000})
    # Independent scalar oracle: 1920 vector ops before publication, 1024 B
    # state write overlaps 512 query/cast ops, then 32 B attention output.
    assert duration == pytest.approx(1.920 + max(1.024, .512) + .032, abs=1e-7)
    write = next(event for event in engine.trace if event["task"].endswith(":state_write"))
    query = next(event for event in engine.trace if event["task"].endswith(":query_dot"))
    assert max(write["start_ms"], query["start_ms"]) < min(write["end_ms"], query["end_ms"])
    _, engine, _ = run((4,), heads=1, dk=16, dv=16, units=1,
                        rates={"vector": 1000, "mte3": 1000})
    writes = [event for event in engine.trace if event["task"].endswith(":state_write")]
    copies = [event for event in engine.trace if event["task"].endswith(":state_copy")]
    for prior, following in zip(writes, copies[1:]):
        assert prior["end_ms"] <= following["start_ms"]
    attention = [event for event in engine.trace if event["task"].endswith(":attention_write")]
    for prior, following in zip(attention, writes[1:]):
        assert prior["end_ms"] <= following["start_ms"]


def test_ragged_heads_follow_source_accumulated_work_assignment():
    # 30 token-heads / 4 owners gives target 8. A complete head may cross it.
    # Owner 0 gets three q3 heads (9); owner 1 two q4 heads (8); owner 2 gets
    # the remaining q4 head plus two q3 heads (10); owner 3 gets the last q3.
    assert head_ownership((3, 4, 3), 3, 4) == (
        ((0, 3),), ((1, 2),), ((1, 1), (2, 2)), ((2, 1),))


def test_read_and_write_bandwidth_controls_change_only_declared_service_time():
    for role in ("mte2_aiv", "mte3"):
        results = [run((4,), heads=1, dk=128, dv=128, units=1,
                       rates={role: 1000 * scale}) for scale in (.5, 1, 2)]
        durations = [result[2] for result in results]
        assert durations[0] > durations[1] > durations[2]
        for _, engine, _ in results[1:]:
            assert engine.serviced == pytest.approx(results[0][1].serviced, rel=1e-11)
