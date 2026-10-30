"""Independent transfer and delivery oracles on the public serving engine."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case, linear_work
from test_prediction_deployment import pd_case


def io_policy():
    return {"host_prepare_ms": .5, "host_result_ms": .5,
            "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8, "latency_us": 250},
            "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4, "latency_us": 250},
            "client_delivery_ms": 2}


def test_public_transfer_phases_preserve_three_completion_boundaries():
    raw = case(prompt=4, output=3)
    raw["runtime"]["serving_io"] = io_policy()
    result = predict(raw, lower=linear_work, event_limit=200)
    row = result["requests"][0]
    device, host, client = (result["metric_boundaries"][key] for key in (
        "device_token_commit", "host_result_ready", "client_delivery"))
    # Each input/return transfer includes its actual bytes and serial latency.
    assert row["first_token_abs_ms"] == 8.75
    assert row["finish_ms"] == 17.75
    assert host["window_end_ms"] == 19.5
    assert client["window_end_ms"] == 21.5
    assert row["boundaries"]["host_result_ready"]["ttft_ms"] == 10.5
    assert row["boundaries"]["client_delivery"]["ttft_ms"] == 12.5
    assert device["tpot_ms"]["mean"] == host["tpot_ms"]["mean"] == 4.5
    assert client["tpot_ms"]["mean"] == 4.5
    service = result["execution"]["serviced"]
    assert service["h2d@r0"] == 48 and service["d2h@r0"] == 12
    assert service["memory@r0:s0"] == 60
    assert service["scheduler@r0"] == 3
    assert result["metrics"]["window_end_ms"] == 19.5
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_final_d2h_tail_changes_host_tps_without_polluting_device_commit_window():
    raw = case(prompt=4, output=1)
    raw["runtime"]["serving_io"] = io_policy()
    first = predict(raw, lower=linear_work)
    raw["runtime"]["serving_io"]["d2h"]["bytes_per_token"] *= 10
    second = predict(raw, lower=linear_work)
    assert first["metric_boundaries"]["device_token_commit"] == second["metric_boundaries"]["device_token_commit"]
    assert second["metric_boundaries"]["device_token_commit"]["output_tps"] == pytest.approx(1000 / 8.75)
    assert second["metric_boundaries"]["host_result_ready"]["window_end_ms"] == 19.5
    assert second["metric_boundaries"]["client_delivery"]["window_end_ms"] == 21.5
    assert second["metric_boundaries"]["host_result_ready"]["output_tps"] < first["metric_boundaries"]["host_result_ready"]["output_tps"]


def test_input_copy_contends_with_other_work_on_endpoint_memory():
    raw = case(prompt=1, output=1)
    raw["machine"]["device"]["rates"]["memory"] = 8000
    raw["runtime"]["serving_io"] = {"h2d": {"bytes_per_s": 1e9, "bytes_per_token": 8}}
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True, trace_limit=100)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    work = linear_work(plan, parsed.model, parsed.runtime, parsed.machine)
    execution.engine.submit("other_memory", {"memory@r0:s0": 16})
    step = execution.add(work, 0, 0)
    execution.engine.advance()
    assert step.completion.time == 3  # two ms shared HBM copy, then one ms compute
    assert execution.engine.serviced["memory@r0:s0"] == 24
    starts = [e["start_ms"] for e in execution.engine.trace if e["demand"].get("matrix@r0:s0")]
    assert starts and min(starts) == 2


def test_closed_loop_releases_after_explicit_client_delivery_and_think_gap():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1, "gap_ms": 2},
    ])
    raw["runtime"]["serving_io"] = {"host_result_ms": 1, "client_delivery_ms": 5}
    rows = predict(raw, lower=linear_work)["requests"]
    assert rows[0]["finish_ms"] == 1
    assert rows[0]["boundaries"]["client_delivery"]["finish_ms"] == 7
    assert rows[1]["arrival_ms"] == 9
    assert rows[1]["finish_ms"] == 10
    assert rows[1]["boundaries"]["client_delivery"]["finish_ms"] == 16


def test_undeclared_client_latency_has_no_client_metric_and_graph_scope_omits_io():
    raw = case(prompt=2, output=2)
    raw["runtime"]["serving_io"] = {"host_result_ms": 3}
    result = predict(raw, lower=linear_work)
    assert set(result["metric_boundaries"]) == {"device_token_commit", "host_result_ready"}
    assert result["requests"][0]["finish_ms"] == 6
    graphs = target_graphs(raw, phase="decode", tokens=1)
    report = graph_costs(raw, graphs, include_submission=True)
    assert not report["execution"]["serving_io"]["enabled"]
    control = deepcopy(raw)
    control["runtime"].pop("serving_io")
    assert report["duration_ms"] == graph_costs(control, graphs, include_submission=True)["duration_ms"]


def test_pd_orders_client_output_even_with_different_pool_latencies():
    raw = pd_case()
    raw["pools"]["prefill"]["runtime"]["serving_io"] = {"client_delivery_ms": 20}
    raw["pools"]["decode"]["runtime"]["serving_io"] = {"client_delivery_ms": 1}
    result = predict(raw, lower=linear_work)
    row = result["requests"][0]
    assert row["finish_ms"] == 8
    assert row["boundaries"]["host_result_ready"]["finish_ms"] == 8
    delivered = row["boundaries"]["client_delivery"]
    assert delivered["first_token_abs_ms"] == delivered["finish_ms"] == 23
    assert delivered["itl"] == {"count": 2, "mean_ms": 0, "max_ms": 0}
    assert result["execution"]["handoff_bytes"] == 384
    assert result["metric_boundaries"]["device_token_commit"]["output_tps"] == 375
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_speculative_bursts_copy_only_committed_tokens_and_keep_long_gaps():
    raw = case(output=6)
    raw["runtime"]["metadata"] = {"draft_mode": "mtp", "draft_proposal_tokens": 2,
        "draft_accept_trace": [0, 2], "draft_accept_tokens": 1}
    raw["runtime"]["serving_io"] = {"d2h": {"bytes_per_s": 4000, "bytes_per_token": 4}}
    result = predict(raw, lower=linear_work)
    row = result["requests"][0]
    assert row["speculative"]["proposed_tokens"] == 4
    assert result["execution"]["serviced"]["d2h@r0"] == 6 * 4
    host = row["boundaries"]["host_result_ready"]
    assert host["generated_tokens"] == 6 and host["itl"]["count"] == 5
    assert host["itl"]["max_ms"] > host["tpot_ms"]
