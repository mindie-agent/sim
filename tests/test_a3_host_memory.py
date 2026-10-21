"""Regressions for the desktop OOM: bounded history, complete results, admission."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_sim.a3_framework import predict
from mindie_sim.a3_resource_engine import ResourceEngine, ResourceSpec
from mindie_sim.a3_types import A3InputError
from test_a3_modeling import case, dense_model


@pytest.mark.skipif(os.name != "posix", reason="probe uses POSIX CPU/memory accounting")
def test_prediction_has_a_real_cpu_wall_time_and_peak_memory_budget():
    payload = case(dense_model(layers=8), runtime={"max_model_len": 256},
                   workload={"requests": [{"prompt_tokens": 8, "output_tokens": 128}]})
    completed = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("measure_prediction_resources.py"))],
        input=json.dumps(payload), text=True, capture_output=True, timeout=8, check=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "modeled" and report["output_tokens"] == 128
    assert report["cpu_s"] < 5
    assert report["elapsed_s"] < 8
    assert report["peak_rss_bytes"] < 160 * 1024**2
    assert not report["trace_capture"][0]["detail_complete"]


def test_large_prediction_refused_before_any_dag_is_built(monkeypatch):
    import mindie_sim.a3_host_budget as budget
    import mindie_sim.a3_framework as framework

    monkeypatch.setattr(budget, "desktop_memory_bytes", lambda: 16 * 1024**3)
    def unexpected(*args, **kwargs):
        pytest.fail("oversized local prediction reached the event engine")
    monkeypatch.setattr(framework, "predict_global_replicas", unexpected)
    large = case(dense_model(layers=61), runtime={"max_model_len": 2048},
                 workload={"requests": [{"prompt_tokens": 1024, "output_tokens": 128}]})
    with pytest.raises(A3InputError, match="host_capacity_requires_remote"):
        predict(large)


def test_trace_omission_preserves_metrics_and_all_physical_counts(monkeypatch):
    import mindie_sim.a3_trace as trace

    monkeypatch.setattr(trace, "MAX_DETAIL_OPS", 40)
    payload = case(workload={"requests": [{"prompt_tokens": 8, "output_tokens": 8}]})
    full = predict(copy.deepcopy(payload), trace_detail="full")
    for mode in ("bounded", "summary"):
        reduced = predict(copy.deepcopy(payload), trace_detail=mode)
        for key in ("metrics", "requests", "slo_goodput", "physical_calibration", "iterations"):
            assert reduced[key] == full[key]
        report = reduced["trace_capture"][0]
        assert not report["detail_complete"]
        assert report["total_ops"] == sum(step["op_count"] for step in full["traces"])
        assert report["retained_ops"] <= (40 if mode == "bounded" else 0)
        assert sum(len(step["ops"]) for step in reduced["traces"]) == report["retained_ops"]


def test_consumed_history_stays_bounded_over_many_decode_steps(monkeypatch):
    original = ResourceEngine.discard_completed_history
    retained = []
    def observe(self, keep=()):
        original(self, keep)
        retained.append((len(self.tasks), len(self.trace), len(self.completed_at)))
    monkeypatch.setattr(ResourceEngine, "discard_completed_history", observe)
    payload = case(workload={"requests": [{"prompt_tokens": 8, "output_tokens": 48}]})
    compact = predict(payload, trace_detail="summary")
    assert len(retained) >= 48
    assert max(max(row) for row in retained) < 10
    monkeypatch.setattr(ResourceEngine, "discard_completed_history", lambda self, keep=(): None)
    reference = predict(payload, trace_detail="summary")
    assert compact["metrics"] == reference["metrics"]
    assert compact["physical_calibration"] == reference["physical_calibration"]


def test_measurement_requests_refuse_an_incomplete_op_inventory(tmp_path):
    from mindie_sim.a3_calibration_store import SQLiteCalibrationStore
    from mindie_sim.a3_measurement import record_prediction_misses

    prediction = predict(case(), trace_detail="summary")
    store = SQLiteCalibrationStore(tmp_path / "requests.db")
    with pytest.raises(A3InputError, match="record_misses_requires_full_trace"):
        record_prediction_misses(store, prediction)
    assert store.manifest()["requests"] == []


def test_history_release_preserves_live_dependencies_and_prepare_deadlines():
    def run(compact):
        engine = ResourceEngine({"cpu": ResourceSpec("cpu", 1.0)})
        engine.add_task("prepare", {"cpu": 2}, deadline_ms=1)
        engine.add_task("step0", {"cpu": 1}, dependencies=("prepare",))
        engine.add_task("background", {"cpu": 10}, dependencies=("step0",))
        first = engine.run(until=("step0",), compact=True)
        if compact:
            engine.discard_completed_history(("prepare",))
        engine.add_task("step1", {"cpu": 1}, dependencies=("prepare",))
        final = engine.run()
        return first["target_completed_at_ms"], final
    times, compact = run(True)
    original_times, full = run(False)
    assert times == original_times
    for key in ("makespan_ms", "serviced", "deadline_slack_ms", "deadline_misses"):
        assert compact[key] == full[key]
    for task_id in ("prepare", "background", "step1"):
        assert compact["completed_at_ms"][task_id] == full["completed_at_ms"][task_id]


def test_trace_reduction_preserves_framework_stream_backlog_and_client_times(monkeypatch):
    import mindie_sim.a3_trace as trace
    from test_framework_predict import _case, _profile, _backend

    monkeypatch.setattr(trace, "MAX_DETAIL_OPS", 40)
    payload = _case(output=16)
    profile = _profile(stream={"fixed_us": 20000})
    reference = predict(payload, framework_profile=profile, cost_backend=_backend(),
                        trace_detail="full")
    for mode in ("bounded", "summary"):
        reduced = predict(payload, framework_profile=profile, cost_backend=_backend(),
                          trace_detail=mode)
        assert reduced["metrics"] == reference["metrics"]
        assert reduced["framework"] == reference["framework"]
        assert reduced["requests"] == reference["requests"]
        for step, full_step in zip(reduced["traces"], reference["traces"]):
            assert step["framework"] == full_step["framework"]
            if step.get("detail_omitted"):
                assert step["events"] == []
