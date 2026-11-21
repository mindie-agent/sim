"""Coupled flows, compact steady pipelines, prepared stages and periodic panels.

[Role] Pin the program-level schedules the Engine executes without expanding
  every tile: coupled multi-resource flows under backpressure, the compact
  two-stage ``SteadyPipeline`` (which must equal the detailed
  ``PipelineProgram`` and an independent slot recurrence), prepared immutable
  ``Stage`` reuse, and ``periodic_matrix_cycle`` jumps checked against a
  literal panel/micro recurrence and the real ``RefillPipeline``.
[Invariants] A coupled flow never reads ahead of its slowest contended
  resource and owners never lend idle units; compact == detailed == recurrence
  for every full/tail geometry the compact form admits, and it refuses the rest
  explicitly; reused stages give byte-identical traces; periodic jumps keep the
  exact span with stepped + jumped == total panels; work is conserved exactly.
[Timeouts & Locks] In-memory Engine runs only; 10M-item and 100k-repeat cases
  are O(1)/O(period) by construction and finish in milliseconds.
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
from mindie_sim.optimization.reload_bounds import compute_compulsory_hbm_cut
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operators.matrix.matrix_cycle import matrix_cycle, periodic_matrix_cycle
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
from mindie_sim.prediction.operators.matrix.steady_pipeline import SteadyPipeline
from mindie_sim.templates import CatlassProblem, lower_catlass_spec
from mindie_sim.templates import _catlass_periodic, catlass_lower

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("n,k,reads,expect_skip", [(32, 65_632, 8_400_896, True), (512, 4_128, 4_491_264, False)])
def test_periodic_matrix_conserves_hbm_and_clock(monkeypatch, n, k, reads, expect_skip):
    path = template_datapath("Ascend950DT_9582")
    problem = CatlassProblem(32, n, k)
    spec, = enumerate_matrix(
        "basic_pingpong", problem, path, tile=(32, 32, 32), aic_budget=path.aic_count,
        l0_tiles=((32, 32, 32),), stage_options=((2, 2, 2, 2, 1),),
    )
    spec = replace(spec, swizzle_direction="Zn")
    context = template_context(path, problem)
    accelerated = lower_catlass_spec(spec, problem, path, context=context)
    default = lower_catlass_spec(spec, problem, path)
    monkeypatch.setattr(catlass_lower, "STREAMING_PANEL_MICRO_THRESHOLD", math.inf)
    detailed = lower_catlass_spec(spec, problem, path, context=context)
    literal = run_template(path, context, detailed)
    skipped = run_template(path, context, accelerated)
    if expect_skip:
        assert skipped._periodic_detector.stats()["skipped_cycles"] > 0
    for result in (default, accelerated, detailed):
        cut = compute_compulsory_hbm_cut((result.summary,))[0]
        assert result.summary.dram_sector_bytes == cut.read_bytes + cut.write_bytes == reads + problem.m * problem.n * 2
        assert result.summary.turnaround_service_bytes == detailed.summary.turnaround_service_bytes
    assert skipped.delivered["hbm:ctr_r"] == pytest.approx(reads, rel=1e-12)
    assert skipped.delivered["hbm:ctr_w"] == literal.delivered["hbm:ctr_w"]
    assert abs(skipped.now - literal.now) <= math.ulp(literal.now)


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


# --- Compact steady two-stage pipeline -------------------------------------


def two_stage_engine():
    return Engine([Resource("load", 1.0), Resource("compute", 1.0)])


def slot_recurrence(items, slots, ready_at):
    """Independent oracle: each item runs two serial stages; slots retire in order."""
    producer_end = consumer_end = ready_at
    retired = []
    for index, (load_work, compute_work) in enumerate(items):
        slot_ready = retired[index - slots] if index >= slots else ready_at
        producer_end = max(producer_end, slot_ready) + load_work
        consumer_end = max(producer_end, consumer_end) + compute_work
        retired.append(consumer_end)
    return consumer_end


def detailed_two_stage(engine, items, ready, slots=2):
    def tile(index):
        load, compute = items[index]
        return Stage("load", {"load": load}), Stage("compute", {"compute": compute}, after=("load",))
    return PipelineProgram(len(items), slots, tile).start(engine, "detailed", ready)


def test_compact_matches_detailed_and_slot_recurrence_across_full_and_tail_geometries():
    ready_at = 7.0
    for count, load, compute, tail in product(
        (0, 1, 2, 7, 31), (1.0, 3.0, 10.0), (2.0, 3.0, 9.0),
        (None, (1.0, 1.0), (17.0, 2.0), (2.0, 19.0)),
    ):
        items = [(load, compute)] * count + ([] if tail is None else [tail])
        if not items:
            continue
        tails = {} if tail is None else {"tail_producer": Stage("load", {"load": tail[0]}),
                                         "tail_consumer": Stage("compute", {"compute": tail[1]})}
        for slots in ((2, 3, 5) if tail is None else (2,)):
            engine = two_stage_engine()
            ready = engine.submit("ready", latency=ready_at)
            end = SteadyPipeline(count, Stage("load", {"load": load}), Stage("compute", {"compute": compute}),
                                 slots=slots, **tails).start(engine, "recurrence", ready)
            engine.advance(horizon=ready_at - 1e-3)
            assert engine.serviced == {"load": 0.0, "compute": 0.0} and not end.done
            engine.advance(until=(end,))
            assert end.time == pytest.approx(slot_recurrence(items, slots, ready_at))
            assert engine.serviced == pytest.approx({"load": sum(i[0] for i in items),
                                                     "compute": sum(i[1] for i in items)})
            summary = engine.summary()
            assert summary["live_tasks"] == summary["live_programs"] == 0 and not engine.allocations
            if slots == 2:
                reference = two_stage_engine()
                detailed = detailed_two_stage(reference, items, reference.submit("ready", latency=ready_at))
                reference.advance(until=(detailed,))
                assert end.time == pytest.approx(detailed.time)
                assert engine.serviced == pytest.approx(reference.serviced)


def test_empty_pipeline_does_not_reserve_declared_buffers():
    engine = Engine([Resource("load", 1.0), Resource("compute", 1.0)], pools={"scratch": 8.0})
    end = SteadyPipeline(0, Stage("load", {"load": 1.0}), Stage("compute", {"compute": 1.0}),
                         buffers=(Allocation("resident", "scratch", 8.0),),
                         ).start(engine, "empty", engine.submit("ready", latency=3.0))
    engine.advance(until=(end,))
    assert end.time == 3.0
    assert engine.serviced == {"load": 0.0, "compute": 0.0}
    assert not engine.allocations
    assert engine.summary()["completed_tasks"] == 2


def test_extra_slot_can_prefetch_a_long_tail_so_compact_rejects_that_domain():
    items = [(1.0, 10.0), (1.0, 10.0), (100.0, 1.0)]
    for slots, expected in ((2, 112.0), (3, 103.0)):
        engine = two_stage_engine()
        end = detailed_two_stage(engine, items, engine.submit("ready"), slots)
        engine.advance(until=(end,))
        assert end.time == pytest.approx(expected)
    with pytest.raises(ValueError, match="tail_requires_two_slots"):
        SteadyPipeline(2, Stage("load", {"load": 1.0}), Stage("compute", {"compute": 10.0}), slots=3,
                       tail_producer=Stage("load", {"load": 100.0}),
                       tail_consumer=Stage("compute", {"compute": 1.0}))


def test_explicit_buffers_wait_for_ready_and_release_per_owner_unit():
    engine = Engine([Resource("load", 1.0), Resource("compute", 1.0)],
                    pools={"scratch": 10.0}, pool_units={"scratch": 2})
    tails = {"tail_producer": Stage("tail-load", {"load": 2.0}),
             "tail_consumer": Stage("tail-compute", {"compute": 1.0})}
    completed = {}
    for label, unit, ready_at in (("a", 0, 5.0), ("b", 1, 0.0), ("c", 0, 5.5)):
        completed[label] = SteadyPipeline(
            1, Stage("load", {"load": 2.0}), Stage("compute", {"compute": 3.0}),
            buffers=(Allocation(f"resident-{label}", "scratch", 10.0, (unit,)),),
            **(tails if label == "a" else {}),
        ).start(engine, label, engine.submit(f"ready-{label}", latency=ready_at), owner=f"tenant-{label}")
    engine.advance(horizon=4.999)
    assert set(engine.allocations) == {"b:resident-b"}
    engine.advance(until=(completed["b"],))
    assert completed["b"].time == pytest.approx(5.0)
    engine.advance(horizon=6.0)
    assert "a:resident-a" in engine.allocations and "c:resident-c" not in engine.allocations
    engine.advance(horizon=10.5)
    # A's final consumer still owns the buffer after A's drain has started.
    assert "a:resident-a" in engine.allocations and not completed["a"].done
    engine.advance(until=(completed["a"], completed["c"]), first=False)
    # B frees unit1 at 5, but C is pinned to unit0 and waits for A's release at 11.
    assert (completed["a"].time, completed["c"].time) == pytest.approx((11.0, 16.0))
    assert engine.peak_unit_allocated == {("scratch", 0): 10.0, ("scratch", 1): 10.0}
    assert not engine.allocations and not engine.unit_allocated
    assert engine.summary()["live_allocated"] == {"scratch": 0.0}


def test_steady_union_preserves_stage_caps_and_owner_units():
    engine = Engine([Resource("load", 20.0, units=2), Resource("compute", 20.0, units=2)], trace_limit=20)
    producer = Stage("load", {"load": 10.0}, caps={"load": 10.0}, placement={"load": (0,)})
    consumer = Stage("compute", {"compute": 10.0}, caps={"compute": 10.0}, placement={"compute": (1,)})
    idle_load = engine.submit("idle-load", {"load": 40.0}, placement={"load": (1,)})
    idle_compute = engine.submit("idle-compute", {"compute": 40.0}, placement={"compute": (0,)})
    end = SteadyPipeline(3, producer, consumer).start(engine, "pinned", engine.submit("ready"))
    engine.advance(until=(end, idle_load, idle_compute), first=False)
    # Losing a cap would use 20/ms; losing placement would contend with idle work.
    assert (end.time, idle_load.time, idle_compute.time) == pytest.approx((4.0, 4.0, 4.0))
    steady = {row["task"]: row for row in engine.trace}["pinned:steady"]
    assert steady["placement"] == {"load": (0,), "compute": (1,)}
    assert steady["demand"] == pytest.approx({"load": 20.0, "compute": 20.0})
    assert steady["coupled"] is True


def test_external_memory_arrival_conserves_work_and_delays_completion():
    def run(with_contender):
        engine = Engine([Resource("producer", 10.0), Resource("memory", 10.0)])
        end = SteadyPipeline(2, Stage("producer", {"producer": 50.0}), Stage("memory", {"memory": 100.0})
                             ).start(engine, "steady", engine.submit("ready"))
        external = engine.submit("external", {"memory": 50.0}, release=10.0) if with_contender else None
        return engine, end, external

    engine, end, external = run(True)
    engine.advance(horizon=10.0)
    assert engine.serviced == pytest.approx({"producer": 75.0, "memory": 50.0})
    engine.advance(horizon=12.5)
    # Coupled flow ran 5 producer/10 memory per ms; after arrival memory is
    # shared, so the producer cannot finish its remaining work ahead of it.
    assert engine.serviced == pytest.approx({"producer": 81.25, "memory": 75.0})
    engine.advance(until=(external,))
    assert external.time == pytest.approx(20.0) and not end.done
    assert engine.serviced == pytest.approx({"producer": 100.0, "memory": 150.0})
    engine.advance(until=(end,))
    assert end.time == pytest.approx(30.0)
    assert engine.serviced == pytest.approx({"producer": 100.0, "memory": 250.0})
    assert engine.summary()["live_tasks"] == 0 and not engine.allocations
    quiet_engine, quiet_end, _ = run(False)
    quiet_engine.advance(until=(quiet_end,))
    assert quiet_engine.serviced == pytest.approx({"producer": 100.0, "memory": 200.0})
    assert end.time - quiet_end.time == pytest.approx(5.0)


def test_bounded_task_window_is_identical_for_ten_and_ten_million_items():
    def run(count):
        engine = two_stage_engine()
        end = SteadyPipeline(count, Stage("load", {"load": 2.0}), Stage("compute", {"compute": 3.0})
                             ).start(engine, "pipeline", engine.submit("ready"))
        engine.advance(until=(end,))
        assert end.time == pytest.approx(2.0 + (count - 1) * 3.0 + 3.0)
        return engine.summary()

    small, large = run(10), run(10_000_000)
    assert small["completed_tasks"] == large["completed_tasks"] < 20
    assert small["peak_live_tasks"] == large["peak_live_tasks"] < 20
    assert large["live_tasks"] == large["live_programs"] == 0


def test_phase_ranking_counterexample_keeps_detailed_overlap_order():
    # Overlapped: 21 vs 30.1; charging phases serially would reverse it (40 vs 32).
    results = {}
    for label, load_rate, cube_rate in (("balanced", 24.0, 32.0), ("fast_copy", 240.0, 32.0 / 1.5)):
        engine = Engine([Resource("mte2", load_rate), Resource("cube", cube_rate)])
        end = SteadyPipeline(20, Stage("refill", {"mte2": 24.0}), Stage("compute", {"cube": 32.0})
                             ).start(engine, label, engine.submit("ready"))
        engine.advance(until=(end,))
        results[label] = end.time
        assert engine.serviced == pytest.approx({"mte2": 480.0, "cube": 640.0})
    assert results == pytest.approx({"balanced": 21.0, "fast_copy": 30.1})


LOAD, COMPUTE = Stage("load", {"load": 1.0}), Stage("compute", {"compute": 1.0})
PINNED = Stage("load", {"load": 1.0}, placement={"load": (0,)})


@pytest.mark.parametrize("args,kwargs,match", [
    ((1, LOAD, COMPUTE), {"slots": 1}, "slots"),
    ((-1, LOAD, COMPUTE), {}, "count"),
    ((1, LOAD, Stage("compute", {"load": 1.0})), {}, "disjoint"),
    ((1, LOAD, COMPUTE), {"tail_producer": Stage("tail", {"load": 1.0})}, "together"),
    ((1, Stage("load", {"load": 1.0}, latency=1.0), COMPUTE), {}, "latency"),
    ((1, Stage("load", {"load": 1.0}, after=("x",)), COMPUTE), {}, "after"),
    ((1, Stage("load", {"load": 1.0}, coupled=True), COMPUTE), {}, "coupled"),
    ((1, PINNED, COMPUTE), {"tail_producer": Stage("tail", {"compute": 1.0}),
                            "tail_consumer": Stage("tail", {"compute": 1.0})}, "tail_resources_not_in_full"),
    ((1, PINNED, COMPUTE), {"tail_producer": Stage("tail", {"load": 1.0}, placement={"load": (1,)}),
                            "tail_consumer": Stage("tail", {"compute": 1.0})}, "placement_mismatch"),
    ((1, LOAD, COMPUTE), {"buffers": (Allocation("same", "pool", 1.0), Allocation("same", "pool", 1.0))},
     "buffer_name_not_unique"),
])
def test_unsupported_steady_shapes_are_refused_explicitly(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        SteadyPipeline(*args, **kwargs)


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
