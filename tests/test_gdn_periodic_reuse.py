"""Independent full-engine controls for the GDN periodic fast paths.

The reference implementations below build and execute the actual native
source plans on a fresh Engine.  They do not call the product fast helpers,
so the comparisons exercise 3/4/5 pack periods directly rather than asserting
the production recurrence against itself.

[Invariants]
  * A matched queue-granule phase makes the one-iteration gating plan
    state-equivalent to the full 3/4/5-iteration plan: elapsed, busy and full
    work agree.
  * A queue granule that does not divide the 64-row address stride falls back
    to the full gating plan; the fast path is not admitted.
  * PF3 one-/two-pack creation remains affine through 3/4/5 packs for actual
    ratio-3, ratio-4 and 32-cube/64-vector occupancy supplies; stage times,
    busy and full Work match the directly executed three-plan pipeline.
[Timeouts & Locks] Bounded to 5 periods; no long token loop.
[Side-Effect Policy] Private Engine instances; no I/O, device or global state.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.gdn_decode_programs import gating_plan
from mindie_sim.prediction.gdn_program_bridge import (
    gating_program_result, pf3_program_result,
)
from mindie_sim.prediction.operators import gdn_output, gdn_prepare, gdn_state
from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.program import RecipeContext

# Actual A5 Qwen3.8 GDN physical supply (same rates/units/queue as the selected
# ordinary target), not a synthetic 1e9 clock.
_RATES = (
    ("cube", 865075200000.0), ("fixpipe", 6758400000.0),
    ("hbm", 4000000000.0), ("link", 200000000.0),
    ("link_hccs", 392000000.0), ("link_roce", 200000000.0),
    ("mte1", 13516800000.0), ("mte2", 4224000000.0),
    ("mte2_aic", 4224000000.0), ("mte2_aiv", 4224000000.0),
    ("mte3", 13516800000.0), ("reg_alu_lanes", 13516800000.0),
    ("reg_data_writes", 211200000.0), ("reg_reduce_lanes", 6758400000.0),
    ("reg_shuffle_lanes", 6758400000.0),
    ("reg_special_lanes", 3379200000.0),
    ("reg_ub_accesses", 211200000.0), ("reg_ub_stores", 105600000.0),
    ("scalar_store_issue", 5657458.563535912),
    ("vector", 6758400000.0),
)
_UNITS = tuple((name, 64 if name in {
    "vector", "mte2_aiv", "mte3", "hbm", "fixpipe", "mte1",
} else 32) for name, _ in _RATES)
_SUPPLY = PipeSupply(
    _RATES,
    resource_units=_UNITS,
    store_queues=(("scalar_store_issue", (
        ("element_bytes", 2), ("granule_bytes", 128),
        ("outstanding", 8), ("response_ms", 9.125e-05),
        ("row_control_ms", 7e-06),
    )),),
    gdn_beta_store=(("queue", {
        "granule_bytes": 128, "outstanding": 8,
        "response_ms": 9.125e-05, "row_control_ms": 7e-06,
    }), ("resource", "scalar_store_issue")),
    gdn_decode_selected=True,
    buffers=(("ub", 253952),), pool_units=(("ub", 64),),
    service_modes=(("reg_issue_slots", ("vector", 211200000000.0)),
                   ("ub_copy", ("vector", 27033600000000.0))),
    compute_units=32,
    compute_peaks=(("bf16", 432537600000000.0),
                   ("fp32", 27033600000000.0)),
)


def _close(left, right):
    return abs(float(left) - float(right)) <= 1e-12 * max(1.0, abs(float(right)))


def _gating_reference(supply, rows, heads):
    rates, units = dict(supply.rates), dict(supply.resource_units)
    queue = dict(supply.store_queues).get("scalar_store_issue")
    queue = dict(queue) if queue else None
    if queue:
        queue = dict(queue)
        queue.setdefault("element_bytes", 2)
    resources = [Resource(name, float(rate), "work", "fair",
                          int(units.get(name, 1))) for name, rate in rates.items()]
    engine = Engine(resources)
    context = RecipeContext(
        op=None, work={}, resource_names={name: name for name in rates},
        rates=rates, core_count=max(1, int(units.get("vector", 1))),
        resource_units=units, pool_names={},
        store_queues=({"scalar_store_issue": dict(queue)} if queue else {}),
    )
    op = SimpleNamespace(
        local_shape=(int(rows), int(heads)),
        metadata={"beta_store": ({"resource": "scalar_store_issue",
                                  "from_hardware": True} if queue else {})},
    )
    geometry = SimpleNamespace(
        resource_units=units,
        compute_units=max(1, int(units.get("vector", 1))),
    )
    plan = gating_plan(op, geometry)
    ready = engine.submit("ready", release=0)
    event = plan.program(context).start(engine, "gdn_gating", ready)
    engine.advance(until=(event,))
    busy = tuple(sorted((name, Fraction.from_float(value))
                        for name, value in engine.busy_ms.items() if value))
    return Fraction.from_float(event.time - ready.time), busy, dict(plan.work.resources)


def _pf3_reference(supply, lengths, heads, keys, dk=128, dv=128, chunk=64):
    rates, units = dict(supply.rates), dict(supply.resource_units)
    units.setdefault("cube", int(supply.compute_units))
    buffers, pool_units = dict(supply.buffers), dict(supply.pool_units)
    modes = dict(supply.service_modes)
    resources = [Resource(name, float(rate), "work", "fair",
                          int(units.get(name, 1))) for name, rate in rates.items()]
    engine = Engine(resources, pools=buffers, pool_units=pool_units)
    context = RecipeContext(
        op=None, work={}, resource_names={r: r for r in set(rates) | set(modes)},
        rates={**rates, **{name: rate for name, (_, rate) in modes.items()}},
        core_count=max(1, int(supply.compute_units)),
        resource_units=units, pool_names={name: name for name in buffers},
        pool_units=pool_units, pool_capacities=buffers, store_queues={},
        service_modes=modes, compute_peaks=dict(supply.compute_peaks),
    )
    lengths = tuple(int(q) for q in lengths)
    params = {"lengths": lengths, "heads": int(heads), "key_heads": int(keys),
              "dk": int(dk), "dv": int(dv), "chunk": int(chunk),
              "dtype": "bf16", "gate_dtype": "fp32", "use_exp2": True}
    prepare = gdn_prepare.make_op({**params, "use_qk_l2norm_in_kernel": True,
                                   "use_gate_in_kernel": False,
                                   "use_beta_sigmoid_in_kernel": False,
                                   "output_a": True, "output_beta": False})
    state = gdn_state.make_op({**params, "use_initial_state": True,
                               "output_final_state": True, "state_v_first": True})
    output = gdn_output.make_op({**params, "state_v_first": True})
    geometry = SimpleNamespace(
        resource_units=units, compute_units=max(1, int(supply.compute_units)),
        buffers=buffers,
    )
    plans = (gdn_prepare.build(prepare, geometry),
             gdn_state.build(state, geometry),
             gdn_output.build(output, geometry))
    ready = engine.submit("ready", release=0)
    previous = ready
    stages, work = [], {}
    for plan in plans:
        start = previous.time if previous.time is not None else engine.now
        event = plan.program(context).start(engine, "pf3", previous)
        engine.advance(until=(event,))
        stages.append(Fraction.from_float(event.time - start))
        previous = event
        for key, value in plan.work.resources.items():
            work[key] = work.get(key, 0) + value
    busy = tuple(sorted((name, Fraction.from_float(value))
                        for name, value in engine.busy_ms.items() if value))
    return Fraction.from_float(previous.time - ready.time), busy, work, tuple(stages)


def _assert_busy(fast, reference):
    fast = dict(fast)
    reference = dict(reference)
    assert set(fast) == set(reference)
    for name, value in reference.items():
        assert _close(fast[name], value)


def test_gating_full_engine_three_to_five_iterations_and_aligned_granule():
    for rounds in (3, 4, 5):
        rows = 64 * 64 * rounds
        reference = _gating_reference(_SUPPLY, rows, 24)
        fast = gating_program_result(_SUPPLY, rows, 24)
        assert _close(fast[0], reference[0])
        _assert_busy(fast[3], reference[1])
        assert dict(fast[4]) == dict(reference[2])


def test_gating_unaligned_granule_falls_back_to_full_plan():
    queue = {"element_bytes": 2, "granule_bytes": 4096, "outstanding": 8,
             "response_ms": 9.125e-05, "row_control_ms": 7e-06}
    supply = replace(_SUPPLY, store_queues=(
        ("scalar_store_issue", tuple(sorted(queue.items()))),
    ))
    for rounds in (3, 4):
        rows = 64 * 64 * rounds
        reference = _gating_reference(supply, rows, 24)
        fast = gating_program_result(supply, rows, 24)
        assert _close(fast[0], reference[0])
        _assert_busy(fast[3], reference[1])
        assert dict(fast[4]) == dict(reference[2])


def test_gating_alternate_outstanding_and_owner_occupancy():
    for outstanding in (1, 4):
        queue = {"element_bytes": 2, "granule_bytes": 128,
                 "outstanding": outstanding, "response_ms": 9.125e-05,
                 "row_control_ms": 7e-06}
        supply = replace(_SUPPLY, store_queues=(
            ("scalar_store_issue", tuple(sorted(queue.items()))),
        ))
        for rounds in (3, 4):
            rows = 64 * 64 * rounds
            reference = _gating_reference(supply, rows, 24)
            fast = gating_program_result(supply, rows, 24)
            assert _close(fast[0], reference[0])
            _assert_busy(fast[3], reference[1])
            assert dict(fast[4]) == dict(reference[2])
    supply32 = replace(
        _SUPPLY,
        resource_units=tuple((name, 32) for name, _ in _UNITS),
        compute_units=32,
    )
    rows = 64 * 32 * 3
    reference = _gating_reference(supply32, rows, 24)
    fast = gating_program_result(supply32, rows, 24)
    assert _close(fast[0], reference[0])
    _assert_busy(fast[3], reference[1])
    assert dict(fast[4]) == dict(reference[2])


def test_gating_4096_period_tail_and_4097_wide_fallback():
    q4096 = {"element_bytes": 2, "granule_bytes": 4096, "outstanding": 8,
             "response_ms": 9.125e-05, "row_control_ms": 7e-06}
    q4097 = dict(q4096); q4097["granule_bytes"] = 4097
    for granule, queue, rounds in (
        (4096, q4096, (4, 5, 8, 9)),
        (4097, q4097, (3, 4)),
    ):
        supply = replace(_SUPPLY, store_queues=(
            ("scalar_store_issue", tuple(sorted(queue.items()))),
        ))
        for k in rounds:
            rows = 64 * 64 * k
            reference = _gating_reference(supply, rows, 24)
            fast = gating_program_result(supply, rows, 24)
            assert _close(fast[0], reference[0]), (granule, k)
            _assert_busy(fast[3], reference[1])
            assert dict(fast[4]) == dict(reference[2]), (granule, k)


def test_pf3_full_engine_three_to_five_pack_creation():
    for rounds in (3, 4, 5):
        lengths = (512 * rounds,)
        reference = _pf3_reference(_SUPPLY, lengths, 24, 8)
        fast = pf3_program_result(_SUPPLY, lengths, 24, 8, 128, 128, 64)
        assert _close(fast[0], reference[0])
        assert len(fast[5]) == len(reference[3])
        for left, right in zip(fast[5], reference[3]):
            assert _close(left, right)
        _assert_busy(fast[3], reference[1])
        assert dict(fast[4]) == dict(reference[2])
        assert dict(fast[4]) == dict(reference[2])


def test_pf3_actual_ratio4_and_occupancy_full_engine_controls():
    for supply, heads, keys, stride in (
        (_SUPPLY, 32, 8, 512),
        (replace(_SUPPLY, resource_units=tuple(
            (name, 64 if name == "vector" else 32)
            for name, _ in _UNITS), compute_units=32), 24, 8, 256),
    ):
        lengths = (stride * 3,)
        reference = _pf3_reference(supply, lengths, heads, keys)
        fast = pf3_program_result(supply, lengths, heads, keys, 128, 128, 64)
        assert _close(fast[0], reference[0])
        for left, right in zip(fast[5], reference[3]):
            assert _close(left, right)
        _assert_busy(fast[3], reference[1])
        assert dict(fast[4]) == dict(reference[2])
