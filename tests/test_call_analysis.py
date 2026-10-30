"""Independent call-protocol queue arithmetic, distinct from serving costs."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.program import RecipeBook
from test_graph_analysis import op
from test_prediction_system import case


def test_repeated_callback_preserves_intermediate_traffic_and_input_readiness():
    ops = (op("producer", memory=2e6),
           op("consumer", memory=3e6, deps=("producer",)))
    result = call_costs(case(), ops, invocations=3, input_ready_ms=7, trace_limit=100)
    assert result["duration_ms"] == 15
    assert result["complete_ms"] == 22
    assert result["execution"]["serviced"]["memory@r0:s0"] == 15e6
    assert all(row["start_ms"] >= 7 for row in result["events"] if row["demand"])
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_eager_submissions_pipeline_and_replay_is_one_launch_for_the_whole_burst():
    raw = case()
    raw["machine"]["device"]["launch_us"] = 2000
    # These serving costs have no place inside a timed primitive callback.
    raw["runtime"]["submission"] = {"scheduler_ms": 100, "completion_ms": 200}
    ops = (op("a", flops=1e6), op("b", memory=1e6, deps=("a",)))
    eager = call_costs(raw, ops, invocations=3, include_submission=True)
    replay = call_costs(raw, ops, invocations=3, mode="graph",
                        include_submission=True, submission={"graph_replay_ms": 3})
    device = call_costs(raw, ops, invocations=3)
    assert eager["duration_ms"] == 13  # Last of six 2ms launches + 1ms device tail.
    assert replay["duration_ms"] == 9  # One 3ms replay + six serial 1ms kernels.
    assert device["duration_ms"] == 6
    assert eager["execution"]["serviced"]["host@r0:s0"] == 12
    assert replay["execution"]["serviced"]["host@r0:s0"] == 3
    assert device["execution"]["serviced"]["host@r0:s0"] == 0
    assert eager["execution"]["serviced"]["scheduler@r0"] == 0
    assert raw["runtime"]["submission"]["scheduler_ms"] == 100
    # Explicit submission arguments must not smuggle a request scheduler or
    # completion tail into the callback event comparison either.
    with pytest.raises(ValueError, match="outside_callback_boundary"):
        call_costs(raw, ops, include_submission=True,
                   submission={"scheduler_ms": 100, "completion_ms": 200})


def test_source_call_expansion_precedes_repetition_and_budgeting():
    from mindie_sim.prediction.mechanism import PlannedRecipe, OperatorPlan, Work
    from mindie_sim.prediction.program import KernelProgram, Stage
    def children(parent):
        return (replace(op("a", memory=1e6), source_boundary="kernel", recipe="inner"),
                replace(op("b", memory=2e6, deps=("a",)), source_boundary="kernel", recipe="inner"))
    def prepare(child, geometry):
        work = Work(resources={"hbm": child.hbm_bytes})
        return OperatorPlan(work, lambda context: KernelProgram((Stage("work", context.bind(work)),)))
    book = RecipeBook(recipes={
        "outer": PlannedRecipe(prepare, kernels=children), "inner": PlannedRecipe(prepare)})
    call = replace(op("closure"), source_boundary="call", recipe="outer")
    result = call_costs(case(), (call,), recipes=book, invocations=2, max_operators=6)
    assert result["duration_ms"] == 6
    assert result["single_call"]["expanded_operators"] == 3
    assert result["single_call"]["source_boundaries"]["kernel"] == 2
    with pytest.raises(ValueError, match="call_expanded_operator_budget_exceeded"):
        call_costs(case(), (call,), recipes=book, invocations=3, max_operators=6)
