"""Executable contracts for the compact steady two-stage pipeline primitive."""
from itertools import product

import pytest

from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.program import PipelineProgram, Stage
from mindie_sim.prediction.steady_pipeline import SteadyPipeline


def _detailed_two_stage(engine, name, ready, count, producer, consumer, slots=2):
    def tile(_):
        return (
            Stage(producer.name, producer.work, caps=producer.caps,
                  placement=producer.placement),
            Stage(consumer.name, consumer.work, after=(producer.name,),
                  caps=consumer.caps, placement=consumer.placement),
        )

    return PipelineProgram(count, slots, tile).start(engine, name, ready)


def test_uniform_compact_matches_detailed_two_stage_and_closed_form():
    producer = Stage("load", {"load": 12.0})
    consumer = Stage("compute", {"compute": 9.0})
    load_ms, compute_ms = 12.0 / 3.0, 9.0 / 1.5

    for count in (1, 2, 1000):
        expected = load_ms + (count - 1) * max(load_ms, compute_ms) + compute_ms
        results = {}
        for label in ("compact", "detailed"):
            engine = Engine([Resource("load", 3.0), Resource("compute", 1.5)])
            ready = engine.submit("ready", latency=7.0)
            if label == "compact":
                end = SteadyPipeline(count, producer, consumer).start(
                    engine, "pipeline", ready)
            else:
                end = _detailed_two_stage(
                    engine, "pipeline", ready, count, producer, consumer)
            engine.advance(horizon=6.999)
            assert engine.serviced == {"load": 0.0, "compute": 0.0}
            assert not end.done
            engine.advance(until=(end,))
            assert end.time == pytest.approx(7.0 + expected)
            assert engine.serviced == pytest.approx(
                {"load": 12.0 * count, "compute": 9.0 * count})
            assert engine.summary()["live_tasks"] == 0
            results[label] = end.time
        assert results["compact"] == pytest.approx(results["detailed"])


def test_empty_pipeline_does_not_reserve_declared_buffers():
    engine = Engine(
        [Resource("load", 1.0), Resource("compute", 1.0)],
        pools={"scratch": 8.0},
    )
    ready = engine.submit("ready", latency=3.0)
    buffer = Allocation("resident", "scratch", 8.0)
    end = SteadyPipeline(
        0, Stage("load", {"load": 1.0}), Stage("compute", {"compute": 1.0}),
        buffers=(buffer,),
    ).start(engine, "empty", ready)
    engine.advance(until=(end,))
    assert end.time == 3.0
    assert engine.serviced == {"load": 0.0, "compute": 0.0}
    assert not engine.allocations
    assert engine.summary()["completed_tasks"] == 2


def test_tail_pair_matches_detailed_full_n_plus_one_and_serial_tail_only():
    producer = Stage("load", {"load": 10.0})
    consumer = Stage("compute", {"compute": 5.0})
    tail_producer = Stage("tail-load", {"load": 7.0})
    tail_consumer = Stage("tail-compute", {"compute": 2.0})

    engines = {}
    ends = {}
    for label in ("compact", "detailed"):
        engine = Engine([Resource("load", 1.0), Resource("compute", 1.0)])
        ready = engine.submit("ready")
        if label == "compact":
            end = SteadyPipeline(
                4, producer, consumer, tail_producer=tail_producer,
                tail_consumer=tail_consumer).start(engine, "tail", ready)
        else:
            def tile(index):
                tail = index == 4
                pwork = tail_producer.work if tail else producer.work
                cwork = tail_consumer.work if tail else consumer.work
                return (
                    Stage("load", pwork),
                    Stage("compute", cwork, after=("load",)),
                )

            end = PipelineProgram(5, 2, tile).start(engine, "tail", ready)
        engine.advance(until=(end,))
        engines[label], ends[label] = engine, end

    assert ends["compact"].time == pytest.approx(
        10.0 + 3 * 10.0 + max(5.0, 7.0) + 2.0)
    assert ends["compact"].time == pytest.approx(ends["detailed"].time)
    assert engines["compact"].serviced == pytest.approx(
        {"load": 4 * 10.0 + 7.0, "compute": 4 * 5.0 + 2.0})
    assert engines["compact"].serviced == pytest.approx(engines["detailed"].serviced)
    summary = engines["compact"].summary()
    assert summary["live_tasks"] == summary["live_programs"] == 0
    assert not engines["compact"].allocations

    engine = Engine([Resource("load", 1.0), Resource("compute", 1.0)])
    end = SteadyPipeline(
        0, producer, consumer, tail_producer=tail_producer,
        tail_consumer=tail_consumer).start(engine, "tail-only", engine.submit("ready"))
    engine.advance(until=(end,))
    assert end.time == pytest.approx(7.0 + 2.0)
    assert engine.serviced == pytest.approx({"load": 7.0, "compute": 2.0})
    assert not engine.allocations


def test_compact_matches_finite_slot_recurrence_across_full_and_tail_geometries():
    """The oracle models each item's two serial stages and slot retirement."""
    for count, load, compute, tail in product(
        (1, 2, 7, 31), (1.0, 3.0, 10.0), (2.0, 3.0, 9.0),
        (None, (1.0, 1.0), (17.0, 2.0), (2.0, 19.0)),
    ):
        slots_values = (2, 3, 5) if tail is None else (2,)
        for slots in slots_values:
            ready_at = 7.0
            producer_end = consumer_end = ready_at
            retired = []
            items = [(load, compute)] * count
            if tail is not None:
                items.append(tail)
            for index, (load_work, compute_work) in enumerate(items):
                slot_ready = retired[index - slots] if index >= slots else ready_at
                producer_end = max(producer_end, slot_ready) + load_work
                consumer_end = max(producer_end, consumer_end) + compute_work
                retired.append(consumer_end)
            engine = Engine([Resource("load", 1.0), Resource("compute", 1.0)])
            tails = {} if tail is None else {
                "tail_producer": Stage("load", {"load": tail[0]}),
                "tail_consumer": Stage("compute", {"compute": tail[1]}),
            }
            end = SteadyPipeline(
                count, Stage("load", {"load": load}),
                Stage("compute", {"compute": compute}), slots=slots, **tails,
            ).start(engine, "recurrence", engine.submit("ready", latency=ready_at))
            engine.advance(until=(end,))
            assert end.time == pytest.approx(consumer_end)
            assert engine.serviced == pytest.approx({
                "load": sum(item[0] for item in items),
                "compute": sum(item[1] for item in items),
            })
            assert engine.summary()["live_tasks"] == 0
            assert not engine.allocations


def test_extra_slot_can_prefetch_a_long_tail_so_compact_rejects_that_domain():
    def detailed(slots):
        engine = Engine([Resource("load", 1.0), Resource("compute", 1.0)])

        def tile(index):
            return (
                Stage("load", {"load": 100.0 if index == 2 else 1.0}),
                Stage("compute", {"compute": 1.0 if index == 2 else 10.0},
                      after=("load",)),
            )

        end = PipelineProgram(3, slots, tile).start(engine, "tail", engine.submit("ready"))
        engine.advance(until=(end,))
        return end.time

    assert detailed(2) == pytest.approx(112.0)
    assert detailed(3) == pytest.approx(103.0)
    with pytest.raises(ValueError, match="tail_requires_two_slots"):
        SteadyPipeline(
            2, Stage("load", {"load": 1.0}), Stage("compute", {"compute": 10.0}),
            slots=3, tail_producer=Stage("load", {"load": 100.0}),
            tail_consumer=Stage("compute", {"compute": 1.0}),
        )


def test_explicit_buffers_wait_for_ready_and_release_per_owner_unit():
    engine = Engine(
        [Resource("load", 1.0), Resource("compute", 1.0)],
        pools={"scratch": 10.0}, pool_units={"scratch": 2},
    )
    producer = Stage("load", {"load": 2.0})
    consumer = Stage("compute", {"compute": 3.0})
    tail_producer = Stage("tail-load", {"load": 2.0})
    tail_consumer = Stage("tail-compute", {"compute": 1.0})
    completed = {}
    for label, unit, ready_at in (
        ("a", 0, 5.0), ("b", 1, 0.0), ("c", 0, 5.5),
    ):
        ready = engine.submit(f"ready-{label}", latency=ready_at)
        tails = ({"tail_producer": tail_producer, "tail_consumer": tail_consumer}
                 if label == "a" else {})
        completed[label] = SteadyPipeline(
            1, producer, consumer,
            buffers=(Allocation(f"resident-{label}", "scratch", 10.0, (unit,)),),
            **tails,
        ).start(engine, label, ready, owner=f"tenant-{label}")

    engine.advance(horizon=4.999)
    assert "b:resident-b" in engine.allocations
    assert "a:resident-a" not in engine.allocations
    assert "c:resident-c" not in engine.allocations
    engine.advance(until=(completed["b"],))
    assert completed["b"].time == pytest.approx(5.0)

    engine.advance(horizon=6.0)
    assert "a:resident-a" in engine.allocations
    assert "c:resident-c" not in engine.allocations
    assert not completed["c"].done
    engine.advance(horizon=10.5)
    # A's final consumer still owns the buffer after A's full drain finished.
    assert "a:resident-a" in engine.allocations
    assert "c:resident-c" not in engine.allocations
    assert not completed["a"].done
    engine.advance(until=(completed["a"], completed["c"]), first=False)
    # B releases unit1 at 5. C is pinned to unit0 and cannot borrow that idle
    # owner capacity; it waits for A's release at 11 and completes at 16.
    assert completed["a"].time == pytest.approx(11.0)
    assert completed["c"].time == pytest.approx(16.0)
    assert engine.peak_unit_allocated == {("scratch", 0): 10.0, ("scratch", 1): 10.0}
    assert engine.summary()["peak_allocated_per_unit"] == {
        "scratch:unit0": 10.0, "scratch:unit1": 10.0}
    assert not engine.allocations
    assert not engine.unit_allocated
    assert engine.summary()["live_allocated"] == {"scratch": 0.0}


def test_steady_union_preserves_stage_caps_and_owner_units():
    engine = Engine(
        [Resource("load", 20.0, units=2), Resource("compute", 20.0, units=2)],
        trace_limit=20,
    )
    producer = Stage(
        "load", {"load": 10.0}, caps={"load": 10.0}, placement={"load": (0,)})
    consumer = Stage(
        "compute", {"compute": 10.0}, caps={"compute": 10.0},
        placement={"compute": (1,)},
    )
    idle_load = engine.submit(
        "idle-load", {"load": 40.0}, placement={"load": (1,)})
    idle_compute = engine.submit(
        "idle-compute", {"compute": 40.0}, placement={"compute": (0,)})
    end = SteadyPipeline(3, producer, consumer).start(
        engine, "pinned", engine.submit("ready"))
    engine.advance(until=(end, idle_load, idle_compute), first=False)

    # Each full stage is capped at 10 work/ms on its own unit. If the steady
    # task lost either cap it would use the 20 work/ms unit rate; if it lost
    # placement it could contend with the independent idle-unit work.
    assert end.time == pytest.approx(4.0)
    assert idle_load.time == pytest.approx(4.0)
    assert idle_compute.time == pytest.approx(4.0)
    steady = {row["task"]: row for row in engine.trace}["pinned:steady"]
    assert steady["placement"] == {"load": (0,), "compute": (1,)}
    assert steady["demand"] == pytest.approx({"load": 20.0, "compute": 20.0})
    assert steady["coupled"] is True


def test_external_memory_arrival_conserves_work_and_delays_completion():
    producer = Stage("producer", {"producer": 50.0})
    consumer = Stage("memory", {"memory": 100.0})

    def run(with_contender):
        engine = Engine([Resource("producer", 10.0), Resource("memory", 10.0)])
        end = SteadyPipeline(2, producer, consumer).start(
            engine, "steady", engine.submit("ready"))
        external = None
        if with_contender:
            external = engine.submit("external", {"memory": 50.0}, release=10.0)
        return engine, end, external

    engine, end, external = run(True)
    engine.advance(horizon=10.0)
    assert engine.serviced == pytest.approx({"producer": 75.0, "memory": 50.0})
    assert not external.done
    engine.advance(horizon=12.5)
    # Before the arrival the coupled flow is at 5 producer/10 memory per ms.
    # Afterwards the external memory work shares the consumer pipe, so the
    # steady producer cannot finish its remaining work ahead of it.
    assert engine.serviced == pytest.approx({"producer": 81.25, "memory": 75.0})
    engine.advance(horizon=15.0)
    assert engine.serviced["producer"] == pytest.approx(87.5)
    assert not end.done
    engine.advance(until=(external,))
    assert external.time == pytest.approx(20.0)
    assert engine.serviced == pytest.approx({"producer": 100.0, "memory": 150.0})
    assert not end.done
    engine.advance(until=(end,))
    assert end.time == pytest.approx(30.0)
    assert engine.serviced == pytest.approx({"producer": 100.0, "memory": 250.0})
    assert engine.summary()["live_tasks"] == 0
    assert not engine.allocations

    quiet_engine, quiet_end, _ = run(False)
    quiet_engine.advance(until=(quiet_end,))
    assert quiet_engine.serviced == pytest.approx({"producer": 100.0, "memory": 200.0})
    assert quiet_engine.now == pytest.approx(25.0)
    assert end.time - quiet_end.time == pytest.approx(5.0)


def test_bounded_task_window_is_identical_for_ten_and_ten_million_items():
    def run(count):
        engine = Engine([Resource("load", 1.0), Resource("compute", 1.0)])
        end = SteadyPipeline(
            count,
            Stage("load", {"load": 2.0}),
            Stage("compute", {"compute": 3.0}),
        ).start(engine, "pipeline", engine.submit("ready"))
        engine.advance(until=(end,))
        return engine.summary(), end.time

    small_summary, small_time = run(10)
    large_summary, large_time = run(10_000_000)
    assert small_time == pytest.approx(2.0 + 9 * 3.0 + 3.0)
    assert large_time == pytest.approx(2.0 + (10_000_000 - 1) * 3.0 + 3.0)
    assert small_summary["completed_tasks"] == large_summary["completed_tasks"]
    assert small_summary["peak_live_tasks"] == large_summary["peak_live_tasks"]
    assert large_summary["completed_tasks"] < 20
    assert large_summary["peak_live_tasks"] < 20
    assert large_summary["live_tasks"] == large_summary["live_programs"] == 0


def test_phase_ranking_counterexample_keeps_detailed_overlap_order():
    results = {}
    for label, load_rate, cube_rate in (
        ("balanced", 24.0, 32.0),
        ("faster_copy_slower_matrix", 240.0, 32.0 / 1.5),
    ):
        engine = Engine([Resource("mte2", load_rate), Resource("cube", cube_rate)])
        end = SteadyPipeline(
            20, Stage("refill", {"mte2": 24.0}),
            Stage("compute", {"cube": 32.0}),
        ).start(engine, label, engine.submit("ready"))
        engine.advance(until=(end,))
        results[label] = end.time
        assert engine.serviced == pytest.approx({"mte2": 480.0, "cube": 640.0})

    # tests/test_phase_ranking.py shows 21 versus 30.1 in the detailed control
    # and the reversed 40 versus 32 ranking when phases are charged serially.
    assert results["balanced"] == pytest.approx(21.0)
    assert results["faster_copy_slower_matrix"] == pytest.approx(30.1)
    assert results["balanced"] < results["faster_copy_slower_matrix"]


def test_unsupported_shapes_are_refused_explicitly():
    producer = Stage("load", {"load": 1.0})
    consumer = Stage("compute", {"compute": 1.0})
    with pytest.raises(ValueError, match="slots"):
        SteadyPipeline(1, producer, consumer, slots=1)
    with pytest.raises(ValueError, match="count"):
        SteadyPipeline(-1, producer, consumer)
    with pytest.raises(ValueError, match="disjoint"):
        SteadyPipeline(1, producer, Stage("compute", {"load": 1.0}))
    with pytest.raises(ValueError, match="together"):
        SteadyPipeline(1, producer, consumer,
                       tail_producer=Stage("tail", {"load": 1.0}))
    with pytest.raises(ValueError, match="latency"):
        SteadyPipeline(1, Stage("load", {"load": 1.0}, latency=1.0), consumer)
    with pytest.raises(ValueError, match="after"):
        SteadyPipeline(1, Stage("load", {"load": 1.0}, after=("x",)), consumer)
    with pytest.raises(ValueError, match="coupled"):
        SteadyPipeline(1, Stage("load", {"load": 1.0}, coupled=True), consumer)
    with pytest.raises(ValueError, match="tail_resources_not_in_full"):
        SteadyPipeline(
            1, Stage("load", {"load": 1.0}, placement={"load": (0,)}), consumer,
            tail_producer=Stage("tail", {"compute": 1.0}),
            tail_consumer=Stage("tail", {"compute": 1.0}))
    with pytest.raises(ValueError, match="placement_mismatch"):
        SteadyPipeline(
            1, Stage("load", {"load": 1.0}, placement={"load": (0,)}), consumer,
            tail_producer=Stage("tail", {"load": 1.0}, placement={"load": (1,)}),
            tail_consumer=Stage("tail", {"compute": 1.0}))
    with pytest.raises(ValueError, match="buffer_name_not_unique"):
        SteadyPipeline(
            1, producer, consumer,
            buffers=(Allocation("same", "pool", 1.0), Allocation("same", "pool", 1.0)))
