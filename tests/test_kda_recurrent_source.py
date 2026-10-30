"""Physical work and finite-queue controls for source-selected KDA decode."""
import math

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators.kda_recurrent import (
    Params, buffer_profile, build, make_op, owner_runs,
)
from mindie_sim.prediction.program import RecipeContext


def execute(lengths=(4, 3), *, heads=6, key_heads=3, dv=128, units=4,
            copy_units=None, ub=256*1024, rates=None, compact=True,
            contender=None, setup=0, minimum=0, planner=build, constructor=make_op, **flags):
    op = constructor({"lengths": lengths, "heads": heads, "key_heads": key_heads,
                  "dk": 128, "dv": dv, **flags}, deps=("input",))
    rates = {"hbm": 1e12, "mte2_aiv": 1e12, "mte2_aic": 1,
             "mte2": 1, "mte3": 1e12, "vector": 1e12, **(rates or {})}
    counts = {key: units for key in rates}
    counts["hbm"] = 1
    if copy_units:
        counts["mte2_aiv"] = counts["mte3"] = copy_units
    geometry = Geometry(max(1, units//2), counts, {"ub": ub}, {"ub": units})
    plan = planner(op, geometry, compact=compact)
    engine = Engine([Resource(key, value, units=counts[key]) for key, value in rates.items()],
                    pools={"ub": ub}, pool_units={"ub": units}, trace_limit=30000)
    context = RecipeContext(plan.work.apply(op), {}, {key: key for key in rates}, rates,
                            max(1, units//2), setup_ms=setup, minimum_ms=minimum,
                            resource_units=counts, pool_names={"ub": "ub"}, pool_units={"ub": units})
    ready = engine.submit("input", release=3)
    complete = plan.lower(context).start(engine, "kda", ready)
    other = engine.submit("external", contender, after=(ready,)) if contender else None
    engine.advance(horizon=2.9)
    assert not any(engine.serviced.values())
    engine.advance(until=(complete,))
    duration = complete.time-3
    if other:
        engine.advance(until=(other,))
    assert not engine._tasks and not engine._promises
    assert not engine.allocations and not any(engine.allocated.values())
    assert not engine.serviced["mte2_aic"] and not engine.serviced["mte2"]
    return op, plan, engine, duration


@pytest.mark.parametrize("lengths,heads,key_heads,dv,units,state,gate,beta", (
    ((1,), 1, 1, 128, 1, "fp32", "bf16", "bf16"),
    ((4,), 3, 1, 128, 4, "fp32", "fp32", "fp32"),
    ((1, 4), 6, 3, 256, 4, "bf16", "fp16", "fp16"),
    ((8, 2, 3), 10, 2, 256, 8, "fp32", "bf16", "fp32"),
    ((5, 4, 2), 7, 1, 128, 40, "bf16", "fp32", "bf16"),
))
def test_resident_state_all_head_beta_and_per_query_checkpoint_payload(
        lengths, heads, key_heads, dv, units, state, gate, beta):
    _, plan, engine, _ = execute(lengths, heads=heads, key_heads=key_heads, dv=dv, units=units,
                               state_dtype=state, gate_dtype=gate, beta_dtype=beta)
    width = {"bf16": 2, "fp16": 2, "fp32": 4}
    incoming = outgoing = vector = 0
    for q in lengths:
        for _ in range(heads):
            incoming += q*(2*128+dv)*2 + q*128*width[gate] + q*heads*width[beta]
            incoming += 128*dv*width[state] + 4 + 128*4
            outgoing += q*(128*dv*width[state] + dv*2)
            vector += math.ceil(q*heads/8)*8  # source whole-HV beta convert
            vector += q*(2*128+dv) + 2*q*128*3 + q*128  # casts, two norms, Q scale
            vector += q*128*(1+1+1+7+1) + 3  # zero,convert,bias,safe gate,exp,A_log
            vector += 128*dv  # initial state resident copy, once across q
            for _ in range(q):
                vector += dv*(128 + 128 + 127 + 2 + 3*128 + 127 + 128 + 1)
    expected = {"hbm": incoming+outgoing, "mte2_aiv": incoming, "mte3": outgoing, "vector": vector}
    assert plan.work.resources == expected
    for key, value in expected.items():
        assert engine.serviced[key] == pytest.approx(value, rel=1e-11)
    schedule = plan.metadata["kda_recurrent_schedule"]
    assert engine.peak_allocated["ub"] == schedule["ub_bytes_per_owner"]


@pytest.mark.parametrize("units,heads,lengths", (
    (1, 3, (1, 4, 1)), (4, 7, (4, 4, 1, 4)), (40, 6, (1, 2, 4, 8)),
    (64, 48, (4,)*128), (8, 3, (1, 2, 3, 4, 5, 8)),
))
def test_round_robin_owner_runs_match_independent_task_id_enumeration(units, heads, lengths):
    blocks = min(units, len(lengths)*heads)
    for owner in range(blocks):
        expected = [lengths[task//heads] for task in range(owner, len(lengths)*heads, blocks)]
        actual = [q for q, count in owner_runs(lengths, heads, blocks, owner) for _ in range(count)]
        assert actual == expected
        assert list(owner_runs(lengths, heads, blocks, owner, compact=False)) == [(q, 1) for q in expected]


@pytest.mark.parametrize("ub,dv,state,expected", (
    (262144, 128, "fp32", (64, 2, 2, 2)),
    (196608, 128, "fp32", (64, 2, 2, 2)),
    (262144, 256, "fp32", (88, 3, 1, 2)),
    (196608, 256, "bf16", (88, 3, 1, 2)),
))
def test_dynamic_host_ub_choice_and_exact_kernel_buffer_allocation(ub, dv, state, expected):
    p = Params((4,), 48, 48, dv=dv, state_dtype=state)
    profile = buffer_profile(p, ub)
    result = tuple(profile[key] for key in ("value_step", "parts", "state_slots", "attention_slots"))
    assert result == expected
    k, n = 128, profile["value_step"]
    state_bytes = 4 if state == "fp32" else 2
    actual = (8*k*2*2 + 8*dv*2 + 8*k*4 + math.ceil(8*48/8)*8*4
              + k*n*state_bytes*(1+profile["state_slots"])
              + profile["attention_slots"]*n*2 + profile["ub_rest_bytes"] + 64)
    assert profile["ub_bytes_per_owner"] == actual <= ub


def _event(engine, suffix):
    return next(event for event in engine.trace if event["task"].endswith(":"+suffix))


def test_double_output_queue_waits_two_queries_and_preserves_fifo_and_state_prefetch():
    _, plan, engine, _ = execute((4,), heads=1, key_heads=1, units=1,
                               rates={"hbm": 1e30, "vector": 1e7, "mte2_aiv": 1e6, "mte3": 1e5})
    schedule = plan.metadata["kda_recurrent_schedule"]
    assert schedule["state_slots"] == schedule["attention_slots"] == 2
    # Source double-buffer path submits token 0 exports only after Compute(1).
    assert _event(engine, "attention_write0_0")["start_ms"] >= _event(engine, "attention_cast0_1")["end_ms"]
    assert _event(engine, "state_export0_2")["start_ms"] >= _event(engine, "state_write0_0")["end_ms"]
    assert _event(engine, "state_write0_0")["start_ms"] >= _event(engine, "attention_write0_0")["end_ms"]
    assert _event(engine, "attention_write0_1")["start_ms"] >= _event(engine, "state_write0_0")["end_ms"]
    # Next value-tile initial state DMA overlaps current tile recurrence.
    load, compute = _event(engine, "state_load1"), _event(engine, "recur0_0")
    assert max(load["start_ms"], compute["start_ms"]) < min(load["end_ms"], compute["end_ms"])


@pytest.mark.parametrize("flags", (
    {"use_qk_l2norm": False}, {"use_gate_in_kernel": False},
    {"safe_gate": False}, {"has_dt_bias": False},
    {"use_beta_sigmoid": True, "allow_neg_eigval": True},
))
def test_gate_and_norm_branches_conserve_their_selected_work(flags):
    _, plan, engine, _ = execute((4, 1), **flags)
    for key, amount in plan.work.resources.items():
        assert engine.serviced[key] == pytest.approx(amount, rel=1e-11)


@pytest.mark.parametrize("role", ("vector", "mte2_aiv", "mte3", "hbm"))
def test_independent_bandwidth_axes_change_real_service_time(role):
    runs = [execute((4, 4), heads=3, key_heads=1, units=4, rates={role: 1e6*factor})
            for factor in (.5, 1, 2)]
    assert runs[0][3] > runs[1][3] > runs[2][3]
    for _, plan, engine, _ in runs:
        for key, amount in plan.work.resources.items():
            assert engine.serviced[key] == pytest.approx(amount, rel=1e-11)


def test_single_active_owner_cannot_borrow_idle_vector_or_copy_capacity():
    _, _, one, t1 = execute((1,), heads=1, key_heads=1, units=1,
                           rates={"vector": 1e6, "hbm": 1e30})
    _, _, four, t4 = execute((1,), heads=1, key_heads=1, units=4,
                            rates={"vector": 4e6, "hbm": 1e30})
    assert t4 == pytest.approx(t1, rel=1e-4)
    assert four.serviced == pytest.approx(one.serviced)
    _, _, _, shared = execute((1,), heads=2, key_heads=1, units=4, copy_units=2,
                             rates={"mte2_aiv": 1e6, "hbm": 1e30})
    _, _, _, separate = execute((1,), heads=2, key_heads=1, units=4, copy_units=4,
                               rates={"mte2_aiv": 2e6, "hbm": 1e30})
    assert shared > 1.9*separate


@pytest.mark.parametrize("lengths,heads,units,dv", (
    ((1,)*4, 6, 4, 128), ((4,)*8, 12, 4, 256),
    ((1, 2, 3, 4, 5, 8), 7, 4, 128), ((4, 4, 1, 1, 4), 3, 8, 256),
))
def test_compact_heads_keep_real_work_and_isolated_duration(lengths, heads, units, dv):
    results = [execute(lengths, heads=heads, key_heads=1, units=units, dv=dv, compact=mode)
               for mode in (False, True)]
    assert results[0][1].work == results[1][1].work
    assert results[0][2].serviced == pytest.approx(results[1][2].serviced, rel=1e-10)
    # Multiple owners can be rephased at ragged run boundaries; uniform q
    # runs have identical isolated physical bottlenecks under fluid service.
    if len(set(lengths)) == 1:
        assert results[0][3] == pytest.approx(results[1][3], rel=1e-9)
    assert results[1][2].completed_count <= results[0][2].completed_count


@pytest.mark.parametrize("resource", ("hbm", "mte2_aiv", "mte3", "vector"))
def test_external_competition_preserves_payload_and_work_without_hidden_floor(resource):
    baseline = execute((4,)*4, heads=6, key_heads=3, units=4)
    results = [execute((4,)*4, heads=6, key_heads=3, units=4, compact=mode,
                       contender={resource: 1e8}) for mode in (False, True)]
    for _, plan, engine, _ in results:
        assert plan.work == baseline[1].work
        for key, amount in plan.work.resources.items():
            assert engine.serviced[key] == pytest.approx(amount + (1e8 if key == resource else 0), rel=1e-9)


def test_one_outer_launch_and_diagnostics_preserve_semantic_contract():
    _, _, _, base = execute((4,)*4)
    _, _, _, with_setup = execute((4,)*4, setup=1)
    _, _, _, with_minimum = execute((4,)*4, minimum=100)
    assert with_setup == pytest.approx(base+1)
    assert with_minimum == pytest.approx(100)
    op = make_op({"lengths": (4,), "heads": 6, "key_heads": 3})
    geometry = Geometry(2, {"vector": 4, "mte2_aiv": 4, "mte2_aic": 2, "mte3": 4},
                        {"ub": 262144}, {"ub": 4})
    prepared, _ = PlannedRecipe(build).prepare(op, geometry)
    metadata = dict(prepared.metadata)
    for key, value in op.metadata:
        assert metadata[key] == value
    assert metadata["program_execution"]["execution_class"] == "vector"
