"""Source-derived native FN regressions through the actual Execution path.

The pinned arch35 source is a4a7958. Synthetic rates isolate dependencies and
per-AIV supply; these checks do not claim measured latency accuracy.
"""
from __future__ import annotations

import pytest

from mindie_sim import Case
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.instructions import REGISTER_RESOURCES
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import native_causal_conv_prefill as native
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView


def _params(**changes):
    return {
        "lengths": [8], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "cpu",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6, **changes,
    }


def _hardware(*, aic=1, aiv=1, split_reads=False, read=1e12,
              write=1e12, vector=1e12):
    rates = {"memory": 1e20, "vector": vector, "mte3": write}
    units = {"vector": aiv, "mte3": aiv}
    rates.update(dict.fromkeys(REGISTER_RESOURCES, vector))
    units.update(dict.fromkeys(REGISTER_RESOURCES, aiv))
    if split_reads:
        rates.update(mte2_aic=1e20, mte2_aiv=read)
        units.update(mte2_aic=aic, mte2_aiv=aiv)
    else:
        rates["mte2"] = read
        units["mte2"] = aiv
    return {
        "machine": {"device": {
            "compute_units": aic, "memory_bytes": 2**30,
            "compute_peaks": {"bf16": 1e14, "fp16": 1e14},
            "rates": rates, "resource_units": units,
            "service_modes": {"reg_issue_slots": {"resource": "vector", "rate": vector}},
            "buffers": {"ub": 253952}, "buffer_units": {"ub": aiv},
        }},
        "model": {
            "name": "source-control", "layers": 1, "hidden_size": 8,
            "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
            "intermediate_size": 16, "vocab_size": 16,
        },
        "workload": {"requests": [{"prompt_tokens": 1, "output_tokens": 1}]},
    }


def _start(params, hardware, *, release_ms=0):
    case, op = Case.from_mapping(hardware), native.make_op(params)
    execution = Execution(
        case, CostModel(case),
        recipes=RecipeBook({op.recipe: PlannedRecipe(native.build)}),
        trace_limit=2000,
    )
    work = IterationWork(
        (op,), IterationPlan((RequestView(
            "r", sum(params["lengths"]), 0, True, "prefill"),)),
        1, 1, 1, None, "prefill", "source-control",
    )
    return execution, execution.add(work, 0, release_ms)


def _finish(execution, step):
    execution.engine.advance(until=(step.completion,))
    assert execution.engine.summary()["live_tasks"] == 0
    assert not any(execution.engine.summary()["live_allocated"].values())
    return list(execution.engine.trace)


def _stage(rows, suffix):
    return next(row for row in rows if row["task"].endswith(suffix))


@pytest.mark.parametrize("metadata", ["cpu", "device"])
@pytest.mark.parametrize("bias", [False, True])
def test_token_prefetch_respects_outer_ready_and_history(metadata, bias):
    execution, step = _start(
        _params(metadata_placement=metadata, bias=bias, has_initial_state=[True]),
        _hardware(read=1000, write=1000, vector=1e20), release_ms=7,
    )
    execution.engine.advance(horizon=6.5)
    assert not any(execution.engine.summary()["serviced"].values())
    rows = _finish(execution, step)
    gate = _stage(rows, ":pairs")
    load = _stage(rows, ":pairs:program:tile1:load")
    history = _stage(rows, ":history")
    assert load["start_ms"] == pytest.approx(gate["end_ms"])
    assert gate["end_ms"] >= history["end_ms"] >= 7
    assert all(row["start_ms"] >= 7 for row in rows if row["demand"])


def test_prefetch_holds_source_ub_buffers_until_kernel_drain():
    execution, step = _start(
        _params(lengths=[1], metadata_placement="device", has_initial_state=[True]),
        _hardware(read=1000, write=1000, vector=1e20),
    )
    execution.engine.advance(horizon=1)
    during = execution.engine.summary()
    # InitSharedBuffersAndEvents precedes PrefetchInitStates: five input T
    # slots, two output T slots and eight float calculation slots.
    resident = (5 + 2) * 4096 * 2 + 8 * 4096 * 4
    assert during["serviced"]["mte2@r0:s0"] == pytest.approx(1)
    assert during["live_allocated"]["ub@r0:s0"] == resident
    assert during["peak_allocated_per_unit"]["ub@r0:s0:unit0"] == resident
    rows = _finish(execution, step)
    reserve = _stage(rows, ":reserve0")
    prefetch = [row for row in rows if ":prefetch0:program:" in row["task"] and row["demand"]]
    release, drain = _stage(rows, ":release0"), _stage(rows, ":drain")
    assert reserve["end_ms"] <= min(row["start_ms"] for row in prefetch)
    assert release["start_ms"] >= drain["end_ms"]


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", ["none", "silu"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_regbase_ledger_matches_selected_source_instruction_counts(width, activation, dtype):
    # arch35/causal_conv1d_regbase.h:101-187, BF16/FP16 specialization.
    instructions = {
        "token_load": 1, "token_cast": 1, "weight_load": width,
        "state_load": width - 1, "multiply": width, "add": width - 1,
        "state_store": width - 1, "output_cast": 1, "output_store": 1,
        "silu": 4 if activation == "silu" else 0,
    }
    params = _params(width=width, activation=activation, dtype=dtype)
    op = native.make_op(params)
    geometry = Geometry(1, {"vector": 1}, {}, {})
    ledger = native.source_ledger(op, geometry)
    # One predicate update and double-slot stores accompany these calls.
    # Tail width16 reduces arithmetic lanes, not the number of instructions.
    setup = (5 + 3) * 16
    issue = {2: 7, 3: 15, 4: 25}[width] + 8 * (sum(instructions.values()) + 1 + width)
    assert ledger["work"]["vector"] == setup
    assert ledger["work"]["reg_issue_slots"] == issue
    assert ledger["work"]["reg_ub_accesses"] == 3 * (width - 1) + 8 * 3 * width
    execution, step = _start(params, _hardware())
    _finish(execution, step)
    assert execution.engine.summary()["serviced"]["vector@r0:s0"] == pytest.approx(setup + issue)


@pytest.mark.parametrize("dim,port_calls", [(16, 21), (64, 21), (80, 42)])
@pytest.mark.parametrize("aiv", [1, 4])
def test_rolling_ub_ports_preserve_register_tails_and_cannot_borrow_inactive_owners(dim, port_calls, aiv):
    hardware = _hardware(aiv=aiv, read=1e20, write=1e20, vector=1e20)
    # Each AIV can perform one UB register access/ms. With one token and at
    # most80 channels, the source launches one AIV despite four being available.
    hardware["machine"]["device"]["rates"]["reg_ub_accesses"] = 1000 * aiv
    params = _params(lengths=[1], dim=dim, activation="none")
    execution, step = _start(params, hardware)
    _finish(execution, step)
    assert step.completion.time == pytest.approx(port_calls)
    assert execution.engine.serviced["reg_ub_accesses@r0:s0"] == port_calls
    hardware["machine"]["device"]["rates"]["reg_ub_accesses"] *= 2
    faster, faster_step = _start(params, hardware)
    _finish(faster, faster_step)
    assert faster_step.completion.time == pytest.approx(port_calls / 2)


@pytest.mark.parametrize("aiv", [1, 2, 8])
def test_rolling_issue_mode_owns_its_bound_physical_pipe(aiv):
    hardware = _hardware(aiv=aiv, read=1e20, write=1e20, vector=1e20)
    device = hardware["machine"]["device"]
    device["rates"]["vector_issue"] = 1000 * aiv
    device["resource_units"]["vector_issue"] = aiv
    device["service_modes"]["reg_issue_slots"] = {
        "resource": "vector_issue", "rate": 1000 * aiv,
    }
    execution, step = _start(
        _params(lengths=[1], dim=80, state_rows=0, state_history_slots=0), hardware)
    rows = _finish(execution, step)
    # Width4: 25 restore slots +30 SiLU token slots, twice for80 channels.
    assert step.completion.time == pytest.approx(110)
    assert execution.engine.serviced["vector_issue@r0:s0"] == pytest.approx(110)
    for row in rows:
        if row["demand"].get("vector_issue@r0:s0"):
            assert row["placement"]["vector_issue@r0:s0"] == (0,)


@pytest.mark.parametrize("tokens", [1, 5])
def test_first_token_load_overlaps_restore_before_pair_prefetch(tokens):
    hardware = _hardware(read=128000, write=1e20, vector=1e20)
    hardware["machine"]["device"]["rates"]["reg_ub_accesses"] = 9000
    params = _params(lengths=[tokens], dim=64, activation="none",
                     cache_indices=None, state_rows=0, state_history_slots=0,
                     metadata_placement="device")
    execution, step = _start(params, hardware, release_ms=5)
    rows = _finish(execution, step)
    history = _stage(rows, ":history")
    first = _stage(rows, ":first_load")
    restore = _stage(rows, ":restore")
    compute = _stage(rows, ":pairs:program:tile0:compute")
    # fn_tasks.h:138-150 waits for history, then issues the current input.
    # causal_conv1d.h:827-829 restores state before waiting for that input.
    assert first["start_ms"] == pytest.approx(history["end_ms"])
    assert restore["start_ms"] == pytest.approx(history["end_ms"])
    assert first["end_ms"] - first["start_ms"] == pytest.approx(1)
    assert restore["end_ms"] - restore["start_ms"] == pytest.approx(1)
    assert compute["start_ms"] == pytest.approx(max(first["end_ms"], restore["end_ms"]))
    if tokens == 1:
        # 512B weights +12B metadata, overlapped128B first input/9 UB calls,
        # then12 token UB calls; all other resources have ample supply.
        assert step.completion.time - 5 == pytest.approx(524 / 128 + 1 + 12 / 9)
    else:
        prefetch = _stage(rows, ":pairs:program:tile1:load")
        assert prefetch["start_ms"] == pytest.approx(compute["start_ms"])


def test_rolling_silu_consumes_special_lanes_without_inventing_ub_reads():
    hardware = _hardware(read=1e20, write=1e20, vector=1e20)
    hardware["machine"]["device"]["rates"]["reg_special_lanes"] = 1000
    params = _params(lengths=[1], dim=16)
    active, active_step = _start(params, hardware)
    _finish(active, active_step)
    plain, plain_step = _start({**params, "activation": "none"}, hardware)
    _finish(plain, plain_step)
    assert active_step.completion.time == pytest.approx(32)
    assert plain_step.completion.time < 1e-8
    assert active.engine.serviced["reg_ub_accesses@r0:s0"] == plain.engine.serviced["reg_ub_accesses@r0:s0"]


def test_generic_arithmetic_competes_before_old_output_slot_is_released():
    params = _params(bias=True)
    # Source RunSeq with bias/SiLU: 18 operations/channel, including one
    # output cast. Each token is 1ms vector, .1ms input and 4ms output.
    token_ops = 18 * 16
    hardware = _hardware(read=320000, write=8000, vector=token_ops * 1000)
    baseline, baseline_step = _start(params, hardware)
    start = _stage(_finish(baseline, baseline_step), ":pairs")["end_ms"]

    execution, step = _start(params, hardware)
    companion = execution.engine.submit(
        "companion-vector", {"vector@r0:s0": token_ops},
        release=start + 2.2, placement={"vector@r0:s0": (0,)},
    )
    execution.engine.advance(until=(companion,))
    # Token2 arithmetic begins at 2ms and needs 17/18ms. At 2.2ms the
    # companion shares the AIV until that arithmetic finishes. Output cast2
    # alone waits for store0 at 5ms. This also determines companion latency.
    assert companion.time - start == pytest.approx(3 + 17 / 18)
    rows = _finish(execution, step)
    arithmetic = _stage(rows, ":pairs:program:tile2:compute")
    cast = _stage(rows, ":pairs:program:tile2:cast")
    old_store = _stage(rows, ":pairs:program:tile0:store")
    assert arithmetic["start_ms"] - start == pytest.approx(2)
    assert arithmetic["end_ms"] < old_store["end_ms"]
    assert cast["start_ms"] - start == pytest.approx(5)
    assert arithmetic["demand"]["vector@r0:s0"] == 17 * 16
    assert cast["demand"]["vector@r0:s0"] == 16


def test_one_active_aiv_uses_only_its_split_read_pipe_and_per_unit_supply():
    params = _params(lengths=[1], metadata_placement="device", has_initial_state=[True])
    read_per_aiv = 1e6
    hardware = _hardware(aic=32, aiv=64, split_reads=True,
                         read=64 * read_per_aiv, write=1e20, vector=1e20)
    execution, step = _start(params, hardware)
    rows = _finish(execution, step)
    reads = [row for row in rows if row["demand"].get("mte2_aiv@r0:s0", 0)]
    assert reads
    for row in reads:
        amount = row["demand"]["mte2_aiv@r0:s0"]
        assert row["placement"]["mte2_aiv@r0:s0"] == (0,)
        assert row["end_ms"] - row["start_ms"] == pytest.approx(
            amount / read_per_aiv * 1000)
    summary = execution.engine.summary()
    assert summary["serviced"]["mte2_aic@r0:s0"] == 0
    assert "mte2@r0:s0" not in summary["serviced"]
    geometry = Geometry(
        32, hardware["machine"]["device"]["resource_units"],
        {"ub": 253952}, {"ub": 64},
    )
    ledger = native.source_ledger(native.make_op(params), geometry)
    assert summary["serviced"]["mte2_aiv@r0:s0"] == pytest.approx(
        ledger["work"]["mte2_aiv"])
    assert ledger["work"].get("mte2_aic", 0) == 0
