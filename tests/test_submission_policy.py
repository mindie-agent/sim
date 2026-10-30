"""Host enqueue service remains independent from device work and PD transfer."""
import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution, SubmissionPolicy
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_graph_analysis import op
from test_prediction_deployment import pd_case
from test_prediction_system import case, linear_work


def test_explicit_runtime_enqueue_overrides_legacy_device_cost_and_runs_ahead():
    raw = case()
    raw["machine"]["device"]["launch_us"] = 1000
    raw["runtime"]["submission"] = {"kernel_launch_ms": .25}
    operators = (op("a", flops=4e6), op("b", flops=2e6, deps=("a",)))
    report = graph_costs(raw, operators, include_submission=True, trace_limit=100)
    # b is enqueued at .5 while a computes from .25 to 4.25. Charging
    # the enqueue after a's device completion would incorrectly give 6.5.
    assert report["duration_ms"] == 6.25
    summary = report["execution"]["submission"]
    assert summary["host_service_ms"] == .5
    assert summary["eager_invocations"] == 2
    assert summary["graph_replays"] == 0
    submits = [e for e in report["events"] if e["task"].endswith(":submit")]
    assert [e["end_ms"] for e in submits] == [.25, .5]

    raw["runtime"]["submission"]["kernel_launch_ms"] = 0
    excluded = graph_costs(raw, operators, include_submission=True)
    assert excluded["duration_ms"] == 6
    assert excluded["execution"]["submission"]["host_service_ms"] == 0
    del raw["runtime"]["submission"]["kernel_launch_ms"]
    legacy = graph_costs(raw, operators, include_submission=True)
    assert legacy["duration_ms"] == 7
    assert legacy["execution"]["submission"]["host_service_ms"] == 2


def test_capture_charges_one_explicit_replay_per_stage_and_preserves_device_work():
    raw = case(pp=2)
    raw["machine"]["device"]["launch_us"] = 1000
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4],
                          graph_capture_mode="full")
    raw["runtime"]["submission"] = {"kernel_launch_ms": .25, "graph_replay_ms": .1}
    graphs = target_graphs(raw, phase="decode", tokens=1)
    captured = graph_costs(raw, graphs, include_submission=True)
    eager = graph_costs(raw, graphs, include_submission=True, replay_scope="none")
    summary = captured["execution"]["submission"]
    assert summary["eager_invocations"] == 0
    assert summary["graph_replays"] == 2
    assert summary["host_service_ms"] == pytest.approx(.2)
    assert eager["execution"]["submission"]["eager_invocations"] > 2
    assert captured["duration_ms"] < eager["duration_ms"]
    assert captured["execution"]["work"] == eager["execution"]["work"]


def test_pd_keeps_independent_pool_enqueue_costs_in_actual_completion():
    raw = pd_case()
    raw["pools"]["prefill"]["runtime"]["submission"] = {"kernel_launch_ms": .25}
    raw["pools"]["decode"]["runtime"]["submission"] = {"kernel_launch_ms": .5}
    report = predict(raw, lower=linear_work)
    assert report["requests"][0]["finish_ms"] == 9.25
    assert report["execution"]["handoff_bytes"] == 384
    prefill, decode = (report["execution"]["submission"][p]
                       for p in ("prefill", "decode"))
    assert (prefill["host_service_ms"], decode["host_service_ms"]) == (.25, 1)
    assert (prefill["eager_invocations"], decode["eager_invocations"]) == (1, 2)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_device_only_async_execution_does_not_charge_background_scheduler():
    raw = case()
    raw["runtime"].update(async_scheduling=True, submission={
        "scheduler_ms": 5, "kernel_launch_ms": 2, "completion_ms": 3})
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    work = linear_work(plan, parsed.model, parsed.runtime, parsed.machine)
    step = execution.add(work, 0, 0)
    execution.prepare_next(0)
    execution.engine.advance()
    assert step.completion.time == execution.engine.now == 4
    summary = execution.summary()
    assert summary["serviced"]["scheduler@r0"] == 0
    assert summary["submission"]["host_service_ms"] == 0
    assert summary["submission"]["eager_invocations"] == 0
    assert summary["live_tasks"] == summary["live_programs"] == 0


def test_api_submission_override_reports_its_actual_parameter_source():
    raw = case(prompt=4, output=1)
    raw["runtime"]["submission"] = {"kernel_launch_ms": 1}
    result = predict(raw, lower=linear_work,
                     submission=SubmissionPolicy(kernel_launch_ms=.25))
    assert result["metrics"]["makespan_ms"] == 4.25
    stage = result["execution"]["submission"]["stages"][0]
    assert stage["kernel_launch_ms"] == .25
    assert stage["kernel_parameter"] == "api.submission.kernel_launch_ms"
