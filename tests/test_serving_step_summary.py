"""Public capture dispatch and all-step statistics retain finite domains."""
from copy import deepcopy

import pytest

from mindie_sim import predict
from test_prediction_system import case


def captured_case(count):
    raw = case(requests=[{"id": str(i), "prompt_tokens": 1, "output_tokens": 2}
                         for i in range(count)])
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4],
                          graph_capture_mode="full", submission={
                              "kernel_launch_ms": .1, "graph_replay_ms": .25},
                          serving_io={"h2d": {"bytes_per_s": 8000,
                                               "bytes_per_token": 8, "token_basis": "padded"}})
    return raw


def test_capture_bucket_and_oversize_eager_fallback_keep_input_bytes_and_output_count():
    captured = predict(captured_case(3), trace_limit=0)
    oversized = predict(captured_case(5), trace_limit=0)
    assert captured["execution"]["submission"]["graph_replays"] == 1
    assert oversized["execution"]["submission"]["graph_replays"] == 0
    assert captured["execution"]["serviced"]["h2d@r0"] == (3 + 4) * 8
    assert oversized["execution"]["serviced"]["h2d@r0"] == pytest.approx((5 + 5) * 8)
    assert captured["metrics"]["output_tokens"] == 6
    assert oversized["metrics"]["output_tokens"] == 10
    groups = captured["execution"]["step_summary"]["groups"]
    assert {(r["mode"], r["replay_scope"], r["graph_bucket"]) for r in groups} == {
        ("prefill", "none", None), ("decode", "full", 4)}
    total = captured["execution"]["step_summary"]["total"]
    assert total["steps"] == 2 and total["valid_tokens"] == 6 and total["padded_tokens"] == 7


def test_all_microbatch_statistics_are_independent_of_bounded_trace():
    raw = captured_case(2)
    raw["runtime"]["pipeline_microbatches"] = 2
    without = predict(raw, trace_limit=0, event_limit=0)
    with_trace = predict(raw, trace_limit=1, event_limit=5)
    summary = without["execution"]["step_summary"]
    assert summary == with_trace["execution"]["step_summary"]
    assert without["traces"] == [] and len(with_trace["traces"]) == 1
    assert summary["total"]["steps"] == 4
    assert without["execution"]["iterations"] == 2
    assert summary["total"]["valid_tokens"] == 4
    assert summary["total"]["padded_tokens"] == 10
    assert sum(row["duration_ms"]["sum"] for row in summary["groups"]) == pytest.approx(
        summary["total"]["duration_ms"]["sum"])
    eager = deepcopy(raw)
    eager["runtime"]["graph_mode"] = "eager"
    result = predict(eager, trace_limit=0)
    assert result["execution"]["submission"]["graph_replays"] == 0
    assert result["execution"]["step_summary"]["total"]["padded_tokens"] == 4
    assert result["metrics"]["output_tokens"] == without["metrics"]["output_tokens"]
