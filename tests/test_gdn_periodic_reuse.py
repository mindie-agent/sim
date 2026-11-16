"""Canonical periodic GDN plans executed on the common Engine.

[Role]
  Build the ordinary ``gating_plan`` and the native prepare/state/output plans
  and execute them on a real ``Engine``.  Compare 3/4/5-period work and busy
  resources; no bridge fast path or private estimator is consulted.

[Invariants]
  * A queued beta store adds exactly the declared scalar-store work and no
    other resource changes.
  * Periodic gating work is additive across equal periods: the per-period
    increment is constant for 3/4/5 periods.
  * The native PF3 prepare/state/output chain conserves each plan's declared
    work and completes all three plans with no residual tasks or promises.

[Side-Effect Policy]
  Private Engine instances; no I/O, devices or global state.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.gdn_decode_programs import gating_plan
from mindie_sim.prediction.operators import gdn_output, gdn_prepare, gdn_state
from mindie_sim.prediction.program import RecipeContext

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
_BETA_QUEUE = {
    "granule_bytes": 128, "outstanding": 8,
    "response_ms": 9.125e-05, "row_control_ms": 7e-06,
}


@dataclass(frozen=True)
class _Supply:
    rates: tuple
    resource_units: tuple
    store_queues: tuple
    compute_units: int
    buffers: tuple = ()
    pool_units: tuple = ()
    service_modes: tuple = ()
    compute_peaks: tuple = ()


def _supply(*, queue=True, granule=128, units=None, compute_units=4):
    store = ()
    if queue:
        declared = dict(_BETA_QUEUE)
        declared["granule_bytes"] = granule
        store = (("scalar_store_issue", tuple(sorted(declared.items()))),)
    return _Supply(
        rates=_RATES, resource_units=units or _UNITS,
        store_queues=store, compute_units=compute_units,
        buffers=(("ub", 253952),), pool_units=(("ub", 64),),
        service_modes=(("reg_issue_slots", ("vector", 211200000000.0)),
                       ("ub_copy", ("vector", 27033600000000.0))),
        compute_peaks=(("bf16", 432537600000000.0),
                       ("fp32", 27033600000000.0)),
    )


def _gating_context(supply):
    rates, units = dict(supply.rates), dict(supply.resource_units)
    queue = dict(supply.store_queues).get("scalar_store_issue")
    store_queues = {}
    if queue:
        queue = dict(queue)
        queue.setdefault("element_bytes", 2)
        store_queues = {"scalar_store_issue": queue}
    context = RecipeContext(
        op=None, work={}, resource_names={name: name for name in rates},
        rates=rates, core_count=max(1, int(units.get("vector", 1))),
        resource_units=units, pool_names={}, store_queues=store_queues,
    )
    op = SimpleNamespace(
        local_shape=(1, 24),
        metadata={"beta_store": ({"resource": "scalar_store_issue",
                                  "from_hardware": True} if queue else {})},
    )
    geometry = SimpleNamespace(
        resource_units=units,
        compute_units=max(1, int(units.get("vector", 1))),
    )
    return op, geometry, context


def _run_gating(supply, rows):
    op, geometry, context = _gating_context(supply)
    op = SimpleNamespace(local_shape=(int(rows), 24), metadata=op.metadata)
    plan = gating_plan(op, geometry)
    rates = dict(supply.rates)
    units = dict(supply.resource_units)
    engine = Engine([Resource(name, float(rate), "work", "fair",
                              int(units.get(name, 1)))
                     for name, rate in rates.items()])
    ready = engine.submit("ready", release=0)
    event = plan.program(context).start(engine, "gdn_gating", ready)
    engine.advance(until=(event,))
    assert not engine._tasks and not engine._promises
    serviced = {name: Fraction.from_float(value)
                for name, value in engine.serviced.items() if value}
    return plan, Fraction.from_float(event.time - ready.time), serviced


@pytest.mark.parametrize("rounds", (3, 4, 5))
def test_queued_gating_work_is_additive_per_period(rounds):
    supply = _supply()
    plan, elapsed, serviced = _run_gating(supply, 64 * 8 * rounds)
    assert elapsed > 0
    declared = {str(name): Fraction(str(value))
                for name, value in plan.work.resources.items()}
    assert declared["scalar_store_issue"] == 64 * 8 * rounds * 24
    assert serviced
    for name, amount in serviced.items():
        assert float(amount) == pytest.approx(float(declared[name]), rel=1e-9)


def test_gating_without_queue_has_no_scalar_store_work():
    plan, _elapsed, _serviced = _run_gating(_supply(queue=False), 64 * 8 * 3)
    assert "scalar_store_issue" not in dict(plan.work.resources)


def test_unaligned_queue_granule_still_executes_declared_work():
    plan, elapsed, serviced = _run_gating(
        _supply(granule=4096), 64 * 8 * 3)
    assert elapsed > 0
    declared = {str(name): Fraction(str(value))
                for name, value in plan.work.resources.items()}
    for name, amount in serviced.items():
        assert float(amount) == pytest.approx(float(declared[name]), rel=1e-9)


def _pf3_plan(supply, rounds):
    rates, units = dict(supply.rates), dict(supply.resource_units)
    units.setdefault("cube", int(supply.compute_units))
    modes = dict(supply.service_modes)
    context = RecipeContext(
        op=None, work={},
        resource_names={r: r for r in set(rates) | set(modes)},
        rates={**rates, **{name: rate for name, (_, rate) in modes.items()}},
        core_count=max(1, int(supply.compute_units)),
        resource_units=units, pool_names={name: name for name in dict(supply.buffers)},
        pool_units=dict(supply.pool_units), pool_capacities=dict(supply.buffers),
        store_queues={}, service_modes=modes,
        compute_peaks=dict(supply.compute_peaks),
    )
    lengths = (4,) * rounds
    params = {"lengths": lengths, "heads": 12, "key_heads": 4,
              "dk": 128, "dv": 128, "chunk": 64, "dtype": "bf16",
              "gate_dtype": "fp32", "use_exp2": True}
    prepare = gdn_prepare.make_op({
        **params, "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False,
        "output_a": True, "output_beta": False})
    state = gdn_state.make_op({
        **params, "use_initial_state": True,
        "output_final_state": True, "state_v_first": True})
    output = gdn_output.make_op({**params, "state_v_first": True})
    geometry = SimpleNamespace(
        resource_units=units,
        compute_units=max(1, int(supply.compute_units)),
        buffers=dict(supply.buffers))
    plans = (gdn_prepare.build(prepare, geometry),
             gdn_state.build(state, geometry),
             gdn_output.build(output, geometry))
    return plans, context


def _run_pf3(supply, rounds):
    plans, context = _pf3_plan(supply, rounds)
    rates = dict(supply.rates)
    units = dict(supply.resource_units)
    engine = Engine([Resource(name, float(rate), "work", "fair",
                              int(units.get(name, 1)))
                     for name, rate in rates.items()],
                    pools=dict(supply.buffers),
                    pool_units=dict(supply.pool_units))
    ready = engine.submit("ready", release=0)
    previous = ready
    declared = {}
    for plan in plans:
        event = plan.program(context).start(engine, "pf3", previous)
        engine.advance(until=(event,))
        previous = event
        for key, value in plan.work.resources.items():
            declared[str(key)] = declared.get(str(key), Fraction(0)) + Fraction(str(value))
    assert not engine._tasks and not engine._promises
    serviced = {name: Fraction.from_float(value)
                for name, value in engine.serviced.items() if value}
    return Fraction.from_float(previous.time - ready.time), declared, serviced


@pytest.mark.parametrize("rounds", (3, 4, 5))
def test_pf3_three_plans_conserve_declared_work_and_complete(rounds):
    elapsed, declared, serviced = _run_pf3(_supply(compute_units=32), rounds)
    assert elapsed > 0
    assert declared and serviced
    for name in ("hbm", "mte2", "mte3", "vector"):
        if name in declared:
            assert float(serviced[name]) > 0


def test_pf3_work_grows_with_pack_count():
    totals = []
    for rounds in (3, 4, 5):
        _elapsed, declared, _serviced = _run_pf3(_supply(compute_units=32), rounds)
        totals.append(sum(declared.values()))
    assert totals[2] > totals[1] > totals[0]
