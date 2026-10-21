"""Framework lifecycle integration using declared CPU-only validation cases."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mindie_sim.a3_framework import TokenLinearCostBackend, predict
from mindie_sim.framework_tasks import FrameworkProfile


def _case(*, prompt=4, output=2, runtime=None):
    return {
        "name": "synthetic-framework-lifecycle-not-measured",
        "machine": {
            "name": "synthetic-test-die", "arch_family": "A3", "num_devices": 1,
            "device": {
                "hbm_gib": 8, "cube_flops_per_s": 1e14,
                "vector_ops_per_s": 1e13, "hbm_bw_bytes_per_s": 1e12,
                "mte_bw_bytes_per_s": 2e12, "launch_us": 1.0,
            },
        },
        "model": {
            "name": "synthetic-dense", "architecture": "gqa", "layers": 1,
            "hidden_size": 16, "num_heads": 2, "num_kv_heads": 1,
            "head_dim": 8, "intermediate_size": 32, "vocab_size": 64,
            "activation_dtype": "bf16", "weight_dtype": "bf16",
            "kv_dtype": "bf16", "compute_dtype": "bf16",
        },
        "workload": {"requests": [{
            "id": "r0", "prompt_tokens": prompt, "output_tokens": output,
        }]},
        "runtime": {
            "tp": 1, "ep": 1, "dp": 1, "pp": 1,
            "max_num_batched_tokens": 16, "max_num_seqs": 1,
            "max_model_len": 64, "kv_block_size": 16,
            "graph_mode": "eager", "gpu_memory_utilization": .9,
            **(runtime or {}),
        },
    }


def _profile(**overrides):
    # Units are microseconds.  A full engine iteration costs 1+2+3+1=7 ms;
    # the independent client stream then needs five additional ms.
    return {
        "source": "synthetic hand calculation, not measured hardware",
        "schedule": {"fixed_us": 500}, "input": {"fixed_us": 500},
        "submit": {"per_kernel_us": 2000}, "output": {"fixed_us": 1000},
        "stream": {"fixed_us": 5000},
        **overrides,
    }


def _backend():
    return TokenLinearCostBackend(base_ms=3, per_token_ms=0)


def _events(result):
    return {
        event["task_id"]: event
        for step in result["traces"] for event in step["events"]
    }


@pytest.mark.parametrize("opaque", [False, True])
def test_no_profile_and_explicit_zero_profile_preserve_entire_legacy_json(opaque):
    kwargs = {"cost_backend": _backend()} if opaque else {}
    case = _case()
    baseline = predict(case, **kwargs)
    for zero in (
        {"source": "zero fixture"},
        FrameworkProfile(source="zero typed fixture"),
        {"source": "explicit zero fixture", "device_launch_us": 0,
         **{kind: {"fixed_us": 0, "per_token_us": 0,
                   "per_request_us": 0, "per_kernel_us": 0}
            for kind in ("schedule", "input", "submit", "output", "stream")}},
    ):
        assert predict(case, framework_profile=zero, **kwargs) == baseline
    assert "framework" not in baseline


def test_one_request_engine_commit_and_client_stream_match_hand_calculation():
    result = predict(_case(), cost_backend=_backend(), framework_profile=_profile())
    assert result["status"] == "modeled"
    first, second = result["traces"]
    assert first["framework"]["device_end_ms"] == pytest.approx(6)
    assert first["framework"]["engine_commit_ms"] == pytest.approx(7)
    assert first["framework"]["client_stream_end_ms"]["r0"] == pytest.approx(12)
    assert second["start_ms"] == pytest.approx(7)
    assert second["framework"]["device_end_ms"] == pytest.approx(13)
    assert second["framework"]["engine_commit_ms"] == pytest.approx(14)
    assert second["framework"]["client_stream_end_ms"]["r0"] == pytest.approx(19)
    assert len(first["framework"]["submit_tasks"]) == 1  # One opaque device call.
    request = result["requests"][0]
    assert request["token_event_times_ms"] == pytest.approx([7, 14])
    assert request["client_token_event_times_ms"] == pytest.approx([12, 19])
    assert request["client_token_times_ms"] == pytest.approx([12, 19])
    assert request["ttft_ms"] == pytest.approx(7)
    assert request["client_ttft_ms"] == pytest.approx(12)
    assert request["client_tpot_ms"] == pytest.approx(7)
    assert request["client_e2e_ms"] == pytest.approx(19)
    assert request["client_finish_ms"] == pytest.approx(19)
    assert result["metrics"]["ttft_ms"]["mean"] == pytest.approx(7)
    assert result["metrics"]["tpot_ms"]["mean"] == pytest.approx(7)
    assert result["framework"]["profile"]["source"] == _profile()["source"]
    assert result["framework"]["client_metrics"]["ttft_ms"]["mean"] == pytest.approx(12)
    events = _events(result)
    for step in result["traces"]:
        lifecycle = step["framework"]
        for field in ("schedule_task", "input_task", "device_gate", "output_task"):
            assert lifecycle[field] in events
        assert all(task in events for task in lifecycle["submit_tasks"])
        assert lifecycle["stream_tasks"]["r0"] in events


def test_stream_backlog_does_not_gate_next_engine_iteration():
    result = predict(
        _case(), cost_backend=_backend(),
        framework_profile=_profile(stream={"fixed_us": 20000}),
    )
    first, second = result["traces"]
    assert first["framework"]["engine_commit_ms"] == pytest.approx(7)
    assert second["start_ms"] == pytest.approx(7)
    assert second["framework"]["engine_commit_ms"] == pytest.approx(14)
    # Both streams contend on one DMA; client completion is a later frontier.
    request = result["requests"][0]
    assert request["client_finish_ms"] > 14
    assert request["finish_ms"] == pytest.approx(14)
    assert request["generated_tokens"] == 2
    assert len(request["client_token_event_times_ms"]) == 2


def test_process_cpu_sharing_is_visible_across_predictive_dp_replicas():
    case = _case(output=1, runtime={"dp": 2})
    case["machine"]["num_devices"] = 2
    case["workload"]["requests"].append({"id": "r1", "prompt_tokens": 4, "output_tokens": 1})
    profile = {"source": "synthetic DP CPU sharing", "schedule": {"fixed_us": 1000}}
    shared = predict(case, framework_profile=profile)
    independent = predict(case, framework_profile={**profile, "cpu_scope": "replica"})
    assert shared["status"] == independent["status"] == "modeled"
    shared_requests = {item["request_id"]: item for item in shared["requests"]}
    independent_requests = {item["request_id"]: item for item in independent["requests"]}
    for rid, request in shared_requests.items():
        assert request["finish_ms"] - independent_requests[rid]["finish_ms"] == pytest.approx(1)
    shared_events, independent_events = _events(shared), _events(independent)
    for step in shared["traces"]:
        event = shared_events[step["framework"]["schedule_task"]]
        assert event["duration_ms"] == pytest.approx(2)
        assert event["demands"] == {"cpu": 1}
    for step in independent["traces"]:
        event = independent_events[step["framework"]["schedule_task"]]
        assert event["duration_ms"] == pytest.approx(1)
        assert event["demands"] == {f"r{step['replica']}:cpu": 1}


def test_chunked_prefill_advances_engine_without_publishing_partial_prompt():
    result = predict(
        _case(prompt=8, output=1, runtime={
            "max_num_batched_tokens": 4, "enable_chunked_prefill": True,
        }),
        cost_backend=_backend(), framework_profile=_profile(),
    )
    assert result["status"] == "modeled"
    partial, final = result["traces"]
    assert partial["requests"][0]["samples"] is False
    assert partial["framework"]["stream_tasks"] == {}
    assert partial["framework"]["client_stream_end_ms"] == {}
    assert partial["framework"]["engine_commit_ms"] == pytest.approx(7)
    assert final["start_ms"] == pytest.approx(7)
    assert final["requests"][0]["samples"] is True
    request = result["requests"][0]
    assert request["token_event_times_ms"] == pytest.approx([14])
    assert request["client_token_event_times_ms"] == pytest.approx([19])
    assert request["computed_tokens"] == 8


def test_eager_submits_one_task_per_operator_not_per_fine_execution_stage():
    from mindie_sim.operator_estimation import create_operator_estimator

    root = Path(__file__).resolve().parents[1]
    mixed = json.loads((root / "examples/operator-plans/mixed.json").read_text(encoding="utf-8"))
    case = _case(prompt=16, output=1)
    case["machine"] = copy.deepcopy(mixed["machine"])
    estimator = create_operator_estimator(dict(mixed["estimators"], strict=False))
    result = predict(
        case, op_estimator=estimator,
        framework_profile={"source": "synthetic submit fixture", "submit": {"per_kernel_us": 20}},
    )
    step = result["traces"][0]
    selected = [op for op in step["ops"] if op.get("execution_plan")]
    assert {op["execution_plan"]["backend"] for op in selected} >= {
        "vector_instruction", "gemm_finite_pipeline",
    }
    submit_ids = step["framework"]["submit_tasks"]
    assert len(submit_ids) == len(step["ops"])
    assert sum(len(op["execution_plan"]["stages"]) for op in selected) > len(submit_ids)
    events = _events(result)
    submit_cpu_work = sum(events[task]["demands"]["cpu"] for task in submit_ids)
    assert submit_cpu_work == pytest.approx(len(submit_ids) * .020)
    # Host submission is a serial sequence even when device nodes can overlap.
    for before, after in zip(submit_ids, submit_ids[1:]):
        assert events[after]["start_ms"] >= events[before]["end_ms"]
    # The next host submit depends on the previous submit, not its device
    # completion.  At least one real device service interval overlaps it.
    overlaps = []
    prefix = f"dp{step['replica']}.it{step['iteration']}"
    for op, next_submit in zip(step["ops"], submit_ids[1:]):
        device_prefix = f"{prefix}:{op['op_id']}"
        host = events[next_submit]
        for task_id, device in events.items():
            if (
                (task_id == device_prefix or task_id.startswith(device_prefix + ":"))
                and device["demands"]
            ):
                overlaps.append(min(device["end_ms"], host["end_ms"])
                                - max(device["start_ms"], host["start_ms"]))
    assert overlaps and max(overlaps) > 1e-12


def test_graph_replay_has_one_submit_without_reusing_eager_kernel_count():
    result = predict(
        _case(prompt=4, output=2, runtime={"graph_mode": "force", "graph_buckets": [4, 8, 16]}),
        framework_profile={"source": "synthetic graph fixture", "submit": {"per_kernel_us": 20}},
    )
    assert result["status"] == "modeled"
    assert all(step["graph_bucket"] is not None for step in result["traces"])
    for step in result["traces"]:
        assert len(step["ops"]) > 1
        assert len(step["framework"]["submit_tasks"]) == 1
        event = _events(result)[step["framework"]["submit_tasks"][0]]
        assert event["demands"]["cpu"] == pytest.approx(.020)


def test_explicit_framework_submit_replaces_legacy_host_launch_intercept():
    low = _case(output=1)
    high = copy.deepcopy(low)
    low["machine"]["device"]["launch_us"] = 1
    high["machine"]["device"]["launch_us"] = 10000
    profile = {
        "source": "synthetic host vs device separation",
        "device_launch_us": 2,
        "submit": {"per_kernel_us": 10},
    }
    low_result = predict(low, framework_profile=profile)
    high_result = predict(high, framework_profile=profile)
    assert low_result["requests"][0]["finish_ms"] == pytest.approx(
        high_result["requests"][0]["finish_ms"],
    )
    assert len(high_result["traces"][0]["framework"]["submit_tasks"]) > 1
    events = _events(high_result)
    for task in high_result["traces"][0]["framework"]["submit_tasks"]:
        assert events[task]["demands"]["cpu"] == pytest.approx(.010)


def test_tp_collective_fabric_setup_latency_survives_host_intercept_removal():
    low = _case(output=1, runtime={"tp": 2})
    low["machine"]["num_devices"] = 2
    low["machine"]["links"] = [{
        "name": "d2d", "channel": "d2d_intra_node", "capacity_gbps": 100,
        "latency_us": 1, "members": [0, 1],
    }]
    high = copy.deepcopy(low)
    high["machine"]["links"][0]["latency_us"] = 101
    profile = {"source": "synthetic fabric separation", "submit": {"per_kernel_us": 1}}
    low_result = predict(low, framework_profile=profile)
    high_result = predict(high, framework_profile=profile)
    assert low_result["status"] == high_result["status"] == "modeled"
    low_communication = {op["op_id"]: op for op in low_result["traces"][0]["ops"]
                         if op["link_bytes"] > 0}
    high_communication = {op["op_id"]: op for op in high_result["traces"][0]["ops"]
                          if op["link_bytes"] > 0}
    assert low_communication.keys() == high_communication.keys()
    assert low_communication
    for op_id, op in low_communication.items():
        following = high_communication[op_id]
        assert op["launch_detail"]["framework_fabric_setup_ms"] == pytest.approx(.001)
        assert following["launch_detail"]["framework_fabric_setup_ms"] == pytest.approx(.101)
        assert following["estimated_ms"] - op["estimated_ms"] == pytest.approx(.100)
    assert high_result["requests"][0]["finish_ms"] > low_result["requests"][0]["finish_ms"]


@pytest.mark.parametrize("scope", ["host_e2e_warm", "block_interval_warm_operator"])
def test_declared_host_or_block_scope_cannot_be_split_into_framework_and_device(scope):
    from mindie_sim.a3_types import A3InputError

    with pytest.raises(A3InputError, match="framework_requires_separable_device_cost_scope"):
        predict(_case(), framework_profile=_profile(), timing_scope=scope)


def _cost_step(*, family="linear", metadata=(), physical=None):
    from types import SimpleNamespace

    from mindie_sim.a3_cost import CalibrationResult, CostedOp
    from mindie_sim.a3_ops import OpSpec

    op = OpSpec(
        op_id="synthetic-cost", family=family, phase="decode", stage=0, layer=0,
        requests=("r0",), semantic_shape=(1, 16, 16), local_shape=(1, 16, 16),
        padded_shape=(1, 16, 16), launch_shape=(1, 16, 16),
        flops=512, metadata=metadata,
    )
    calibration = CalibrationResult("analytic", 1, 0, None, False, False, "synthetic")
    cost = CostedOp(
        op=op, demands=(("cube", 512),), analytic_ms=.1, estimated_ms=.101,
        calibration=calibration, launch_ms=.001, service_ms=.1, physical=physical,
    )
    return {"work": SimpleNamespace(graph_bucket=None), "costed": [cost]}


def test_multi_call_grouped_matmul_is_refused_instead_of_counting_one_host_submit():
    from mindie_sim.a3_types import A3InputError, CaseSpec
    from mindie_sim.framework_runtime import separate_host_intercepts

    step = _cost_step(family="grouped_matmul", metadata=(("group_size", 256),))
    with pytest.raises(A3InputError, match="framework_multiple_launches_not_supported"):
        separate_host_intercepts(step, CaseSpec.from_mapping(_case()),
                                 FrameworkProfile.from_mapping(_profile()))


@pytest.mark.parametrize("fitted_scope", [
    "block_interval_warm_operator", "host_e2e_warm", "", "unknown_scope", "device_event_unknown",
])
def test_applied_block_or_host_physical_alias_is_refused_despite_device_request(fitted_scope):
    from mindie_sim.a3_types import A3InputError, CaseSpec
    from mindie_sim.framework_runtime import separate_host_intercepts

    # The caller requested device time, but the row actually fitted a host/block
    # protocol (or omitted its protocol).  An alias must not hide that boundary.
    physical = {
        "applied": True,
        "fitted_timing_scope": fitted_scope,
        "scope_alias": {"from": "device_event_warm", "to": fitted_scope},
    }
    if fitted_scope:
        physical["timing_scope"] = "device_event_warm"
    step = _cost_step(physical=physical)
    with pytest.raises(A3InputError, match="framework_physical_scope_not_separable"):
        separate_host_intercepts(step, CaseSpec.from_mapping(_case()),
                                 FrameworkProfile.from_mapping(_profile()))


def test_speculative_multi_token_bursts_preserve_actual_output_token_work():
    case = _case(output=4, runtime={"metadata": {
        "draft_mode": "verify", "draft_proposal_tokens": "5", "draft_accept_tokens": "5",
    }})
    result = predict(
        case, cost_backend=_backend(),
        framework_profile={
            "source": "synthetic speculative output accounting",
            "output": {"per_token_us": 100}, "stream": {"per_token_us": 200},
        },
    )
    request = result["requests"][0]
    assert result["status"] == "modeled"
    assert request["generated_tokens"] == 4
    assert len(request["token_event_times_ms"]) == 4
    assert len(request["client_token_event_times_ms"]) == 4
    assert len(request["client_token_times_ms"]) < 4  # A verify burst publishes several tokens.
    assert request["client_token_event_times_ms"][-1] == request["client_finish_ms"]
    events = _events(result)
    output_work = sum(events[step["framework"]["output_task"]]["demands"].get("cpu", 0)
                      for step in result["traces"])
    stream_work = sum(
        events[task]["demands"].get("dma", 0)
        for step in result["traces"] for task in step["framework"]["stream_tasks"].values()
    )
    assert output_work == pytest.approx(4 * .100)
    assert stream_work == pytest.approx(4 * .200)


@pytest.mark.parametrize("profile", [
    [], {"source": ""}, {"source": "fixture", "unknown_stage": {}},
    {"source": "fixture", "input": {"fixed_us": True}},
    {"source": "fixture", "submit": {"fixed_us": -1}},
])
def test_cli_rejects_invalid_framework_profile_as_structured_error(tmp_path, profile):
    from mindie_sim.__main__ import main

    case_path = tmp_path / "case.json"
    profile_path = tmp_path / "framework.json"
    output_path = tmp_path / "result.json"
    case_path.write_text(json.dumps(_case()), encoding="utf-8")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    code = main([
        "predict", "--input", str(case_path), "--framework-config", str(profile_path),
        "--output", str(output_path),
    ])
    assert code == 1
    error = json.loads(output_path.read_text(encoding="utf-8"))
    assert error["status"] == "error"
    assert error["command"] == "predict"
    assert "framework_" in error["reason"]


def test_cli_framework_config_is_integrated_into_predict_json(tmp_path):
    from mindie_sim.__main__ import main

    case_path = tmp_path / "case.json"
    profile_path = tmp_path / "framework.json"
    output_path = tmp_path / "result.json"
    case_path.write_text(json.dumps(_case(output=1)), encoding="utf-8")
    profile = {"source": "CLI integration fixture", "schedule": {"fixed_us": 1000}}
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    assert main([
        "predict", "--input", str(case_path), "--framework-config", str(profile_path),
        "--output", str(output_path),
    ]) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["framework"]["profile"]["source"] == profile["source"]
    assert result["traces"][0]["framework"]["schedule_task"]
    assert result["requests"][0]["client_finish_ms"] is not None
