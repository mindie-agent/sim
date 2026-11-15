"""Observer-driven exposed attribution on real serial/parallel causal edges."""
from __future__ import annotations

import pytest

from mindie_sim.prediction.exposed_program import (
    StageDescriptor, declarations_from_program, program_descriptors,
)
from mindie_sim.prediction.runtime.exposed_observer import ExposedObserver
from mindie_sim.prediction.runtime.exposed_intervals import (
    bind_interval, interval_clocks, residual_capacity, service_role,
)
from mindie_sim.schema.common import ModelingError


def run(stages):
    observer = ExposedObserver()
    observer.begin_scope("step", 0)
    terminal = observer.program_scope("step", 0, stages)
    observer.finish_scope("step", terminal)
    return observer, terminal


def test_serial_and_concurrent_program_attribution():
    _, terminal = run((StageDescriptor("cube", cube_ms=4),
                       StageDescriptor("vector", vector_ms=3)))
    observer = ExposedObserver()
    observer.begin_scope("serial", 0)
    terminal = observer.program_scope("serial", 0, (
        StageDescriptor("cube", cube_ms=4),
        StageDescriptor("vector", vector_ms=3)))
    result = observer.finish_scope("serial", terminal)
    assert result["exposed_breakdown_ms"] == {
        "cube": 4.0, "vector": 3.0, "memory": 0.0, "communication": 0.0}

    observer = ExposedObserver()
    observer.begin_scope("parallel", 0, inputs={
        "cube": None, "vector": None})
    observer.on_submit("cube", declaration=StageDescriptor("cube", cube_ms=4))
    observer.on_submit("vector", declaration=StageDescriptor("vector", vector_ms=7))
    observer.on_resolve("cube", moment_ms=4)
    observer.on_resolve("vector", moment_ms=7)
    joined = observer.finish_scope("parallel", ("cube", "vector"))
    assert joined["exposed_breakdown_ms"]["cube"] == pytest.approx(4)
    assert joined["exposed_breakdown_ms"]["vector"] == pytest.approx(3)


def test_control_disclosure_is_bounded_by_the_vector_bucket():
    observer = ExposedObserver()
    observer.begin_scope("control", 0)
    terminal = observer.program_scope("control", 0, (
        StageDescriptor("control", control_ms=2, opaque_control_ms=1),
        StageDescriptor("coupled", cube_ms=5, memory_ms=3)))
    result = observer.finish_scope("control", terminal)
    basis = result["exposed_breakdown_basis"]
    assert basis["vector_control_ms"] <= result["exposed_breakdown_ms"]["vector"]
    assert basis["opaque_control_ms"] <= result["exposed_breakdown_ms"]["vector"]
    assert basis["method"] == "ordered_critical_span_v1"


def test_service_interval_uses_offered_capacity_and_records_blockers():
    interval = interval_clocks(
        delivered_work={"hbm": 100.0, "cube": 100.0},
        offered_capacities={"hbm": 25.0, "cube": 100.0})
    assert interval.clocks.cube_ms == pytest.approx(1.0)
    assert (interval.clocks.without_communication_ms
            - interval.clocks.compute_ms) == pytest.approx(3.0)
    blocked = interval_clocks(
        delivered_work={"hbm": 100.0}, offered_capacities={"hbm": 0.0})
    assert blocked.blockers == ("hbm",)
    assert blocked.clocks.all_ms == 0
    assert residual_capacity({"hbm": 100.0}, {"hbm": 30.0}) == {"hbm": 70.0}
    assert service_role("link:fabric") == "communication"
    assert service_role("memory@r0") == "memory"
    clocks, control, opaque = bind_interval(
        interval=blocked, control_ms=2.0, opaque_control_ms=1.0)
    assert clocks.compute_ms - clocks.cube_ms == pytest.approx(2.0)
    assert control == 2.0 and opaque == 1.0
    with pytest.raises(ModelingError, match="control_invalid"):
        bind_interval(interval=interval, control_ms=-1.0)


def test_periodic_skip_advances_all_four_clocks_and_bounded_trace():
    observer = ExposedObserver(max_events=8)
    observer.begin_scope("periodic", 0)
    for index in range(20):
        observer.on_submit(f"extra{index}",
                           declaration=StageDescriptor("cube", cube_ms=1))
    observer.on_submit("stage0", declaration=StageDescriptor(
        "coupled", cube_ms=2, vector_ms=3, memory_ms=1))
    observer.on_periodic_skip(
        correspondence={"stage0": "stage0"}, repeat=3,
        clock_delta={"all_ms": 3, "without_communication_ms": 3,
                     "compute_ms": 3, "cube_ms": 2})
    result = observer.finish_scope("periodic", "stage0")
    assert len(observer.scope_events("periodic")) <= 8
    assert result["exposed_breakdown_basis"]["periodic_skips"] == 3
    assert sum(result["exposed_breakdown_ms"].values()) == pytest.approx(
        result["exposed_breakdown_basis"]["duration_ms"])


def test_program_descriptors_map_semantic_roles_not_op_names():
    class Program:
        stages = (
            type("Stage", (), {"work": {"cube": 100.0}, "latency": 0.0,
                               "after": (), "coupled": False}),
            type("Stage", (), {"work": {"hbm": 100.0}, "latency": 0.0,
                               "after": ("c0",), "coupled": False}),
        )

    descriptors = program_descriptors(
        Program(), {"cube": 100.0, "hbm": 50.0})
    assert [item.role for item in descriptors] == ["cube", "memory"]
    declarations = declarations_from_program(
        Program(), {"cube": 100.0, "hbm": 50.0})
    assert declarations[0]["cube_ms"] == pytest.approx(1.0)
    assert declarations[1]["memory_ms"] == pytest.approx(2.0)
