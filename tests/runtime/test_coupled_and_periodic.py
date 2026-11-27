"""Coupled flows, prepared stages and periodic panels.

[Role] Pin the program-level schedules the Engine executes without expanding
  every tile: coupled multi-resource flows under backpressure, prepared immutable
  ``Stage`` reuse, and ``periodic_matrix_cycle`` jumps checked against a
  literal panel/micro recurrence and the real ``RefillPipeline``.
[Invariants] A coupled flow never reads ahead of its slowest contended
  resource and owners never lend idle units; reused stages give byte-identical
  traces; periodic jumps keep the exact span with stepped + jumped == total
  panels; work is conserved exactly.
[Timeouts & Locks] In-memory Engine runs only; 100k-repeat cases are
  O(1)/O(period) by construction and finish in milliseconds.
[Side-Effect Policy] No filesystem, environment or network access.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import product
import math
import weakref

import pytest

from _support.linear_cases import enumerate_matrix, run_template, template_context, template_datapath
from mindie_sim.prediction.runtime.reload_bounds import compute_compulsory_hbm_cut
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
from mindie_sim.prediction.operators.matrix.matrix_cycle import matrix_cycle, periodic_matrix_cycle
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
from mindie_sim.templates import CatlassProblem, lower_catlass_spec
from mindie_sim.templates import _catlass_periodic, catlass_lower

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("n,k,reads,expect_skip,reserved_mib", [
    (32, 65_632, 8_400_896, True, 0), (512, 4_128, 4_491_264, False, 0),
    (512, 4_128, 4_509_696, False, 126),
])
def test_periodic_matrix_conserves_hbm_and_clock(monkeypatch, n, k, reads, expect_skip, reserved_mib):
    path = template_datapath("Ascend950DT_9582")
    problem = CatlassProblem(32, n, k)
    spec, = enumerate_matrix(
        "basic_pingpong", problem, path, tile=(32, 32, 32), aic_budget=path.aic_count,
        l0_tiles=((32, 32, 32),), stage_options=((2, 2, 2, 2, 1),),
    )
    spec = replace(spec, swizzle_direction="Zn")
    context = template_context(path, problem)
    view = None
    if reserved_mib:
        view = AdmissionMemoryView({0: path.usable_bytes("l2")}, memory_access=path.memory_access, smooth_knee=True)
        assert view.grant_reservation(0, "peer", reserved_mib * 1024**2) == 2 * 1024**2
    accelerated = lower_catlass_spec(spec, problem, path, context=context, memory_view=view)
    default = lower_catlass_spec(spec, problem, path, memory_view=view)
    monkeypatch.setattr(catlass_lower, "STREAMING_PANEL_MICRO_THRESHOLD", math.inf)
    detailed = lower_catlass_spec(spec, problem, path, context=context, memory_view=view)
    literal = run_template(path, context, detailed)
    skipped = run_template(path, context, accelerated)
    if expect_skip:
        assert skipped._periodic_detector.stats()["skipped_cycles"] > 0
    for result in (default, accelerated, detailed):
        assert result.summary.dram_sector_bytes == reads + problem.m * problem.n * 2
        if not reserved_mib:
            cut = compute_compulsory_hbm_cut((result.summary,))[0]
            assert result.summary.dram_sector_bytes == cut.read_bytes + cut.write_bytes
        assert result.summary.turnaround_service_bytes == detailed.summary.turnaround_service_bytes
    assert skipped.delivered["hbm:ctr_r"] == pytest.approx(reads, rel=1e-12)
    assert literal.delivered["hbm:ctr_r"] == pytest.approx(reads, rel=1e-12)
    assert skipped.delivered["hbm:ctr_w"] == literal.delivered["hbm:ctr_w"]
    # Contended cohorts can sum the same service increments in a different order.
    assert (skipped.now == pytest.approx(literal.now, rel=1e-12)) if reserved_mib else (abs(skipped.now - literal.now) <= math.ulp(literal.now))
    if reserved_mib:
        rebound = replace(default, program=default.program.with_context(path, context))
        unbound = run_template(path, context, rebound)
        assert unbound.now == skipped.now and unbound.delivered == skipped.delivered


def test_streaming_panel_encoding_keeps_only_bounded_lookahead(monkeypatch):
    monkeypatch.setattr(_catlass_periodic, "_panel_key", str)
    periods = (1, 2, 4, 8)
    encoder = _catlass_periodic._PanelPhases(periods)
    motif = tuple(range(2 * max(periods)))
    panels = (17, 18) + motif * 4096 + (19,)
    for panel in panels:
        encoder.append(panel)
        assert len(encoder.pending) < 3 * max(periods)
    phases, _ = encoder.finish()
    assert len(phases) == 3
    assert tuple(panel for phase in phases for panel in phase.panels * phase.count) == panels


def test_cascading_cohort_overlap_settles_before_the_timer():
    engine = Engine([Resource("read", 2, units=2), Resource("copy", 2, units=2)], backend="python")
    members = ((0,), (1,))
    both = (0, 1)
    slow = engine.submit("slow", {"read": 4, "copy": 4}, coupled=True, cohort_size=2,
                         placement={"read": both, "copy": both}, cohort_widened=("read", "copy"),
                         cohort_members={"read": members, "copy": members})
    fast = engine.submit("fast", {"read": 4}, cohort_size=2, placement={"read": both},
                         cohort_widened=("read",), cohort_members={"read": members})
    engine.submit("blocker", {"copy": 2}, placement={"copy": (0,)})
    engine.advance(horizon=1)
    slow_members, fast_members = (engine._tasks[event.sequence].split_children for event in (slow, fast))
    assert len(slow_members) == len(fast_members) == 2
    assert engine.serviced == {"read": 2, "copy": 1.5}


# --- Coupled flows: backpressure and owner-scoped arbitration --------------


def test_coupled_progress_cannot_read_ahead_while_compute_is_contended():
    engine = Engine([Resource("read", 10), Resource("compute", 10)])
    done = engine.submit("steady", {"read": 100, "compute": 100}, coupled=True)
    blocker = engine.submit("other-compute", {"compute": 50})
    engine.advance(horizon=5)
    assert engine.serviced == pytest.approx({"read": 25, "compute": 50})
    engine.advance(until=(blocker,))
    assert blocker.time == pytest.approx(10)
    assert engine.serviced == pytest.approx({"read": 50, "compute": 100})
    engine.advance(until=(done,))
    assert done.time == pytest.approx(15)
    assert engine.serviced == pytest.approx({"read": 100, "compute": 150})


def test_complementary_resource_flows_reclaim_the_capacity_of_each_other():
    engine = Engine([Resource("read", 10), Resource("compute", 10)])
    left = engine.submit("left", {"read": 10, "compute": 100}, coupled=True)
    right = engine.submit("right", {"read": 100, "compute": 10}, coupled=True)
    engine.advance(until=(left, right), first=False)
    assert (left.time, right.time) == pytest.approx((11, 11))
    assert engine.serviced == pytest.approx({"read": 110, "compute": 110})


def test_coupled_unit_ownership_and_external_release_change_progress_causally():
    engine = Engine([Resource("read", 20, units=2), Resource("compute", 20, units=2)])
    done = engine.submit("steady", {"read": 100, "compute": 100},
                         placement={"read": (0,), "compute": (0,)}, coupled=True, release=3)
    other = engine.submit("other", {"compute": 40}, release=5, placement={"compute": (0,)})
    independent = engine.submit("independent", {"compute": 100}, placement={"compute": (1,)})
    engine.advance(horizon=3)
    assert engine.serviced["read"] == 0
    engine.advance(until=(done, other, independent), first=False)
    assert (independent.time, other.time, done.time) == pytest.approx((10, 13, 17))
    assert engine.summary()["live_tasks"] == 0
    assert engine.serviced == pytest.approx({"read": 100, "compute": 240})


def test_fifo_eligibility_does_not_consume_other_coupled_resources_early():
    engine = Engine([Resource("queue", 1, policy="fifo"), Resource("read", 10)])
    earlier = engine.submit("earlier", {"queue": 5})
    later = engine.submit("later", {"queue": 2, "read": 20}, coupled=True)
    engine.advance(until=(earlier,))
    assert engine.serviced == pytest.approx({"queue": 5, "read": 0})
    engine.advance(until=(later,))
    assert later.time == pytest.approx(7)


@pytest.mark.parametrize("right_owners", ((1, 3), (3, 1)))
def test_capped_flows_and_independent_components_share_only_their_owners(right_owners):
    # Two of four owners participate: 60 read / 40 compute per ms. The capped
    # flow takes (12, 4); compute's remaining 36 splits 18/18; the ordinary
    # read component reclaims the final 30. A disjoint owner gets its own 30.
    engine = Engine([Resource("read", 120, units=4), Resource("compute", 80, units=4)])
    placement = {"read": (1, 3), "compute": (1, 3)}
    left = engine.submit("capped", {"read": 60, "compute": 20}, placement=placement,
                         caps={"read": 12}, coupled=True)
    right = engine.submit("coupled", {"read": 60, "compute": 60}, coupled=True,
                          placement={"read": right_owners, "compute": right_owners})
    ordinary = engine.submit("independent-components", {"read": 20, "compute": 20}, placement=placement)
    separate = engine.submit("other-owner", {"read": 100}, placement={"read": (0,)})
    engine._settle()
    rates = engine._rates()
    assert rates["read"] == pytest.approx({left.sequence: 12, right.sequence: 18,
                                           ordinary.sequence: 30, separate.sequence: 30})
    assert rates["compute"] == pytest.approx({left.sequence: 4, right.sequence: 18, ordinary.sequence: 18})
    engine.advance(horizon=0.5)
    assert engine.serviced == pytest.approx({"read": 45, "compute": 20})
    engine.advance()
    assert engine.serviced == pytest.approx({"read": 240, "compute": 100})
    assert engine.summary()["live_tasks"] == 0


# --- Prepared immutable stages ---------------------------------------------


def prepared_run(*, reuse, trace_limit=1000):
    engine = Engine([Resource("read", 12, units=2), Resource("cube", 8, units=2)],
                    pools={"l1": 16}, pool_units={"l1": 2}, trace_limit=trace_limit)
    ready = engine.submit("ready", latency=0.25)
    background = engine.submit("background", {"read": 7, "cube": 3}, release=1.5)
    templates = {}

    def tile(index):
        slot, width = index % 2, 1 if index == 8 else 4
        if (slot, width) not in templates:
            allocation = Allocation(f"slot{slot}", "l1", 8, (slot,))
            templates[slot, width] = (
                Stage("load", {"read": width * 6}, caps={"read": 5}, acquire=(allocation,),
                      placement={"read": (slot,)}),
                Stage("compute", {"cube": width * 3}, after=("load",), previous=("publish",),
                      placement={"cube": (slot,)}, coupled=True),
                Stage("publish", after=("compute",), free=(allocation,),
                      program=KernelProgram((Stage("output", {"read": width}),))),
            )
        return templates[slot, width] if reuse else tuple(replace(s) for s in templates[slot, width])

    plan = PipelineProgram(9, 2, tile)
    complete, concurrent = plan.start(engine, "pipeline", ready), plan.start(engine, "concurrent", ready)
    engine.advance(until=(complete, concurrent, background), first=False)
    return engine.summary(), list(engine.trace)


def test_prepared_reuse_and_disabled_trace_preserve_identical_execution():
    reused, fresh = prepared_run(reuse=True), prepared_run(reuse=False)
    assert reused == fresh
    assert reused[0]["live_tasks"] == reused[0]["live_programs"] == 0
    assert reused[0]["live_allocated"] == {"l1": 0}
    quiet, no_events = prepared_run(reuse=True, trace_limit=0)
    assert reused[1] and not no_events
    traced = dict(reused[0])
    for key in ("retained_trace_events", "trace_dropped"):
        traced.pop(key)
        quiet.pop(key)
    assert traced == quiet


def test_prepared_allocation_totals_recheck_live_occupancy_for_each_tile():
    engine = Engine([Resource("read", 1)], pools={"l1": 10}, pool_units={"l1": 2}, trace_limit=100)
    held = Allocation("background", "l1", 4, (0,))
    engine.submit("background", acquire=(held,), free=(held,), latency=3)
    operands = (Allocation("a", "l1", 4, (0, 1)), Allocation("b", "l1", 3, (0,)))
    stages = (Stage("copy", {"read": 1}, acquire=operands, free=operands),)
    complete = PipelineProgram(3, 1, lambda _: stages).start(engine, "p", engine.submit("ready"))
    engine.advance(until=(complete,))
    copies = [row for row in engine.trace if row["task"].endswith(":copy")]
    # Each operand fits beside the background alone, but unit0 needs 7 bytes.
    assert [(row["start_ms"], row["end_ms"]) for row in copies] == [(3, 4), (4, 5), (5, 6)]
    assert complete.time == 6 and engine.serviced == {"read": 3}
    assert engine.summary()["peak_allocated_per_unit"] == {"l1:unit0": 7, "l1:unit1": 4}
    assert engine.summary()["live_allocated"] == {"l1": 0}


def test_prepared_submission_keeps_duplicate_pending_and_resolved_dependencies():
    engine = Engine([Resource("read", 10)], trace_limit=100)
    prior = engine.submit("prior", {"read": 10})
    engine.advance(until=(prior,))
    promised = engine.future("promised")
    gate = engine.submit("gate", latency=2, on_complete=lambda _: engine.resolve(promised))
    done = engine.submit("consumer", {"read": 20}, release=4, after=(prior, promised, gate, promised, prior))
    engine.advance(until=(done,))
    assert (prior.time, gate.time, promised.time, done.time) == (1, 3, 3, 6)
    assert engine.serviced == {"read": 30}
    assert engine.summary()["completed_tasks"] == 3
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_stage_work_and_placement_are_construction_time_snapshots():
    work, caps, units = {"read": 12}, {"read": 3}, [0]
    stage = Stage("read", work, caps=caps, placement={"read": units})
    work["read"], caps["read"], units[0] = 120, 30, 1
    engine = Engine([Resource("read", 12, units=2)])
    complete = PipelineProgram(3, 1, lambda _: (stage,)).start(engine, "p", engine.submit("ready"))
    engine.advance(until=(complete,))
    assert complete.time == 12 and engine.serviced == {"read": 36}


def test_generated_stages_leave_only_the_live_slot_window():
    refs, retained = [], []

    def tile(index):
        retained.append(sum(ref() is not None for ref in refs))
        stage = Stage("varying", {"read": index + 1})
        refs.append(weakref.ref(stage))
        return (stage,)

    engine = Engine([Resource("read", 1)])
    complete = PipelineProgram(100, 3, tile).start(engine, "p", engine.submit("ready"))
    engine.advance(until=(complete,))
    assert complete.time == 5050
    assert max(retained) <= 3
    assert not any(ref() is not None for ref in refs)


def test_pipeline_previous_lag_one_orders_adjacent_tiles():
    """Both static DAG unrolling and dynamic slot windows honor previous_lags=(('p', 1),)."""
    # 1. Two slots, two tiles (static unroll path): tile 0 runs 'p' (3 ms), tile 1 runs 'c' (5 ms) with lag 1 on 'p'.
    for slots in (2, 1):
        engine = Engine([Resource("r0", 1.0), Resource("r1", 1.0)])
        tiles = (
            (Stage("p", {"r0": 3.0}),),
            (Stage("c", {"r1": 5.0}, previous_lags=(("p", 1),)),),
        )
        done = PipelineProgram(2, slots, lambda idx, _t=tiles: _t[idx]).start(
            engine, "lag1", engine.submit("ready"),
        )
        engine.advance(until=(done,))
        assert done.time == pytest.approx(8.0, rel=1e-12)

    # 2. Dynamic multi-slot window (count=3 > slots=2): lag 1 chains across tile 0 -> 1 -> 2.
    engine3 = Engine([Resource("r0", 1.0), Resource("r1", 1.0), Resource("r2", 1.0)])
    tiles3 = (
        (Stage("p", {"r0": 3.0}),),
        (Stage("c", {"r1": 5.0}, previous_lags=(("p", 1),)),),
        (Stage("d", {"r2": 2.0}, previous_lags=(("c", 1),)),),
    )
    done3 = PipelineProgram(3, 2, lambda idx, _t=tiles3: _t[idx]).start(
        engine3, "lag1_dyn", engine3.submit("ready"),
    )
    engine3.advance(until=(done3,))
    assert done3.time == pytest.approx(10.0, rel=1e-12)

    # 3. Mid-stream static-to-dynamic handoff (tile 0..1 static, tile 2 acquires buffer) keeps lag=1 exact.
    alloc = Allocation("buf", "ub", 4)
    engine4 = Engine([Resource("r0", 1.0), Resource("r1", 1.0)], pools={"ub": 8})
    tiles4 = (
        (Stage("p", {"r0": 3.0}),),
        (Stage("p", {"r0": 2.0}),),
        (Stage("c", {"r1": 4.0}, acquire=(alloc,), free=(alloc,), previous_lags=(("p", 1),)),),
    )
    done4 = PipelineProgram(3, 2, lambda idx, _t=tiles4: _t[idx]).start(
        engine4, "lag1_handoff", engine4.submit("ready"),
    )
    engine4.advance(until=(done4,))
    assert done4.time == pytest.approx(9.0, rel=1e-12)
    assert not engine4.allocations


# --- Periodic finite-panel matrix cycles -----------------------------------


def literal_panels(pattern, repeats, tail, refill_slots, consumer_slots):
    """Independent literal panel/micro history oracle in exact fractions."""
    load_done, operand_done, cube_done, panel_done = [Fraction(0)], [Fraction(0)], [Fraction(0)], []
    for load, micros in tuple(pattern) * repeats + tuple(tail):
        retired = panel_done[-refill_slots] if len(panel_done) >= refill_slots else 0
        load_done.append(max(load_done[-1], retired) + Fraction(load))
        for operand, cube in micros:
            reusable = cube_done[-consumer_slots] if len(cube_done) > consumer_slots else 0
            operand_done.append(max(operand_done[-1], load_done[-1], reusable) + Fraction(operand))
            cube_done.append(max(cube_done[-1], operand_done[-1]) + Fraction(cube))
        panel_done.append(cube_done[-1])
    return float(cube_done[-1])


def engine_panels(pattern, repeats, tail, refill_slots, consumer_slots):
    engine = Engine([Resource(role, 1.0) for role in ("load", "operand", "cube")])
    count = len(pattern) * repeats

    def panel(index):
        load, micros = pattern[index % len(pattern)] if index < count else tail[index - count]

        def steps():
            for part, (operand, cube) in enumerate(micros):
                yield (Stage("operand", {"operand": operand}),
                       Stage("cube", {"cube": cube}, after=("operand",))), part + 1 == len(micros)
        return Stage("load", {"load": load}), steps(), ()

    done = RefillPipeline(count + len(tail), refill_slots, consumer_slots, panel).start(
        engine, "periodic", engine.submit("ready", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return done.time - 7


PATTERNS = (
    ((1., ((.25, 2.), (3., .5))), (.125, ((.5, .25),)), (4., ((0., 1.), (2., .5)))),
    ((.1, ((.01, .03),)), (.3, ((.05, .09), (.01, .12)))),
    ((0., ((0., 0.),)), (0., ((0., 1.),))),
)
TAIL = ((1.75, ((2., .25), (.125, 3.))), (.5, ((.25, .125),)), (0., ((1., .5),)))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_periodic_and_finite_tail_match_literal_recurrence_across_slots(pattern):
    for repeats, refill, consumer in product((0, 1, 2, 7, 37), (1, 2, 3), (1, 2, 5)):
        result = periodic_matrix_cycle(pattern, repeats, refill, consumer, tail=TAIL)
        assert result.span_ms == literal_panels(pattern, repeats, TAIL, refill, consumer)
        assert result.stepped_panels + result.jumped_panels == repeats * len(pattern) + len(TAIL)
        if result.period_panels:
            assert result.period_panels % len(pattern) == 0


@pytest.mark.parametrize("refill,consumer", [(1, 1), (2, 2), (3, 4)])
def test_variable_micro_lengths_use_actual_engine_dependencies(refill, consumer):
    result = periodic_matrix_cycle(PATTERNS[0], 9, refill, consumer, tail=TAIL)
    assert result.span_ms == pytest.approx(engine_panels(PATTERNS[0], 9, TAIL, refill, consumer))


def test_phase_index_prevents_a_false_cycle_when_normalized_times_already_match():
    # One slot, zero micro work: normalized state repeats every panel but the
    # next load differs, so a phase-free cycle key would be wrong.
    pattern = tuple((float(load), ((0., 0.),)) for load in (1, 2, 5))
    result = periodic_matrix_cycle(pattern, 100_000, 1, 1)
    assert result.span_ms == 800_000
    assert result.period_panels == 3
    assert result.stepped_panels < 20 and result.jumped_panels > 299_980


def test_pattern_boundary_never_drains_shared_refill_and_consumer_windows():
    pattern = ((1., ((0., 2.),)), (1., ((0., 2.),)))
    result = periodic_matrix_cycle(pattern, 10)
    assert result.span_ms == 41
    assert result.span_ms < periodic_matrix_cycle(pattern, 1).span_ms * 10
    assert result.span_ms == engine_panels(pattern, 10, (), 2, 2)


@pytest.mark.parametrize("length", [15, 16, 80])
def test_large_refill_and_scale_periods_jump_without_expanding_panels(length):
    # Distinct 3/5 or 16/80 source periods are explicit input, not averaged away.
    pattern = tuple((float(10 + (20 if i % 3 == 0 else 0) + (10 if i % 5 == 0 else 0)),
                     ((.25, 1.), (.5, 2.))) for i in range(length))
    tail = ((17., ((.5, 1.),)), (11., ((.25, 2.),)))
    result = periodic_matrix_cycle(pattern, 100_000, tail=tail)
    assert result.span_ms == 100_000 * sum(load for load, _ in pattern) + 28 + 2.25
    assert result.period_panels % length == 0
    assert result.stepped_panels < length * 8 + len(tail)
    assert result.stepped_panels + result.jumped_panels == 100_000 * length + len(tail)


def test_uniform_api_keeps_identical_cycle_witness_and_arbitrary_tail_is_continuous():
    full, tail = (1.5, ((.25, 2.), (1., .5))), (.75, ((.5, .125),))
    uniform = matrix_cycle(*full, 10_000, 3, 2, tail_load_ms=tail[0], tail_micro_steps=tail[1])
    assert uniform == periodic_matrix_cycle((full,), 10_000, 3, 2, tail=(tail,))
    pure_tail = periodic_matrix_cycle((), 0, tail=TAIL)
    assert pure_tail.span_ms == literal_panels((), 0, TAIL, 2, 2)
    assert pure_tail.prologue_panels == pure_tail.prologue_ms == 0
    assert periodic_matrix_cycle((), 0).span_ms == 0
