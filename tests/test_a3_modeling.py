"""Focused A3 dual-model, resource-engine, calibration and should-cost tests.

These tests use normalized declared machines and CPU-backed fixtures.  They do
not claim any A3 measurement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import mindie_sim as sim
from mindie_sim.a3_calibration_store import SQLiteCalibrationStore
from mindie_sim.a3_measurement import record_prediction_misses
from mindie_sim.a3_cost import CalibrationStore, attention_flops, gemm_flops
from mindie_sim.a3_framework import TokenLinearCostBackend, predict
from mindie_sim.a3_measurement import CpuMatmulRunner, run_pending
from mindie_sim.a3_resources import (
    ResourceEngine, ResourceSpec, simulate_microbatch_pipeline,
)
from mindie_sim.a3_should import (
    normalized_chunked_pipeline_search, normalized_shared_cv_search, should_cost,
)
from mindie_sim.a3_types import (
    A3InputError, CaseSpec, MachineSpec, ModelSpec, ParallelSpec,
    RuntimeSpec, WorkloadSpec,
)
from mindie_sim.a3_workload import (
    IterationPlan, RequestView, derive_iteration_work, memory_plan, selected_mode,
)


def machine(devices: int = 2, *, hbm_gib: float = 8.0) -> dict:
    return {
        "name": "normalized-test-die",
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": devices,
        "device": {
            "hbm_gib": hbm_gib,
            "cube_flops_per_s": 1e14,
            "vector_ops_per_s": 1e13,
            "hbm_bw_bytes_per_s": 1e12,
            "mte_bw_bytes_per_s": 2e12,
            "l2_bytes": 1024 * 1024,
            "launch_us": 1.0,
        },
        "links": [{
            "name": "d2d", "channel": "d2d_intra_node", "capacity_gbps": 100,
            "latency_us": 1.0, "members": list(range(devices)),
        }],
    }


def dense_model(**overrides) -> dict:
    payload = {
        "layers": 1, "hidden_size": 64, "num_heads": 4, "num_kv_heads": 1,
        "head_dim": 16, "intermediate_size": 128, "vocab_size": 1000,
        "activation_dtype": "bf16", "weight_dtype": "bf16", "compute_dtype": "bf16",
    }
    payload.update(overrides)
    return payload


def moe_model(**overrides) -> dict:
    payload = dense_model(
        num_experts=8, top_k=2, moe_intermediate_size=64,
        expert_routing_histogram=[0.3, 0.2, 0.12, 0.1, 0.08, 0.08, 0.07, 0.05],
    )
    payload.update(overrides)
    return payload


def case(model: dict | None = None, *, runtime: dict | None = None, workload: dict | None = None, **top) -> dict:
    payload = {
        "machine": machine(),
        "model": model or dense_model(),
        "workload": workload or {"requests": [{"prompt_tokens": 8, "output_tokens": 2}]},
        "runtime": {"max_num_batched_tokens": 16, "max_model_len": 128, **(runtime or {})},
    }
    payload.update(top)
    return payload


# ---------------------------------------------------------------------------
# Typed contracts and bug regressions
# ---------------------------------------------------------------------------


def test_ep_can_span_dp_tp_domain_and_is_not_an_extra_world_axis():
    model = ModelSpec.from_mapping(moe_model())
    parallel = ParallelSpec(dp=8, tp=1, ep=8)
    parallel.validate(model)
    assert parallel.world_size == 8
    assert parallel.ep_domain == 8
    assert parallel.ep_spans_dp is True
    assert parallel.etp == 1

    payload = predict(case(
        moe_model(), runtime={"dp": 8, "tp": 1, "ep": 8},
        workload={"requests": [{"prompt_tokens": 8, "output_tokens": 1}]},
        machine=machine(8),
    ))
    assert payload["status"] in {"modeled", "partial"}
    assert payload["plan"]["runtime"]["parallel"]["ep_domain"] == 8
    assert payload["plan"]["runtime"]["parallel"]["world_size"] == 8


def test_attention_projection_width_need_not_equal_hidden_width():
    model = ModelSpec.from_mapping(dense_model(hidden_size=100, num_heads=4, head_dim=32))
    assert model.hidden_size == 100
    assert model.num_heads * model.head_dim == 128
    parallel = ParallelSpec(tp=2)
    assert model.weight_bytes_per_layer(parallel) > 0


def test_tp_above_kv_heads_replicates_kv_instead_of_dividing_it():
    model = ModelSpec.from_mapping(dense_model(num_heads=8, num_kv_heads=1, head_dim=8))
    assert model.local_kv_heads(8) == 1
    kv_tp1 = memory_plan(model, RuntimeSpec.from_mapping({"tp": 1}), MachineSpec.from_mapping(machine(8))).kv_bytes_per_token
    kv_tp8 = memory_plan(model, RuntimeSpec.from_mapping({"tp": 8}), MachineSpec.from_mapping(machine(8))).kv_bytes_per_token
    assert kv_tp8 == pytest.approx(kv_tp1)

    wide_kv = ModelSpec.from_mapping(dense_model(num_heads=8, num_kv_heads=8, head_dim=8))
    kv_wide_1 = memory_plan(wide_kv, RuntimeSpec.from_mapping({"tp": 1}), MachineSpec.from_mapping(machine(8))).kv_bytes_per_token
    kv_wide_8 = memory_plan(wide_kv, RuntimeSpec.from_mapping({"tp": 8}), MachineSpec.from_mapping(machine(8))).kv_bytes_per_token
    assert kv_wide_8 == pytest.approx(kv_wide_1 / 8)


def test_force_graph_bucket_overflow_falls_back_to_eager_without_dropping_tokens():
    model = ModelSpec.from_mapping(dense_model())
    runtime = RuntimeSpec.from_mapping({
        "max_num_batched_tokens": 16, "max_model_len": 128,
        "graph_mode": "force", "graph_buckets": [4],
    })
    plan = IterationPlan(tuple(RequestView(f"r{i}", 1, 10, True, "decode") for i in range(5)))
    work = derive_iteration_work(plan, model, runtime, MachineSpec.from_mapping(machine()))
    assert work.valid_tokens == 5
    assert work.padded_tokens == 5
    assert work.graph_bucket is None
    guard = next(item for item in work.recipe_guards if item["recipe"] == "attention")
    assert guard["feasible"] is False
    assert "eager_fallback" in guard["reason"]


def test_moe_mc2_preserves_dispatch_plus_combine_bytes_and_dependencies():
    model = ModelSpec.from_mapping(moe_model())
    plan = IterationPlan((RequestView("r", 128, 0, True, "prefill"),))
    alltoall_bytes = 0.0
    mc2_bytes = 0.0
    for recipe in ("alltoall", "mc2"):
        runtime = RuntimeSpec.from_mapping({
            "max_num_batched_tokens": 256, "max_model_len": 512,
            "tp": 2, "ep": 2, "moe_recipe": recipe, "mc2_chunks": 4,
        })
        work = derive_iteration_work(
            plan, model, runtime, MachineSpec.from_mapping(machine(2)),
            ParallelSpec(tp=2, ep=2),
        )
        link = sum(op.link_bytes for op in work.ops if op.family in {"ep_dispatch", "ep_combine"})
        dispatch = [op for op in work.ops if op.family == "ep_dispatch"]
        combine = [op for op in work.ops if op.family == "ep_combine"]
        assert dispatch and combine
        assert sum(op.link_bytes for op in dispatch) == pytest.approx(sum(op.link_bytes for op in combine))
        if recipe == "alltoall":
            alltoall_bytes = link
            assert len(dispatch) == len(combine) == 1
        else:
            mc2_bytes = link
            # The fused path still carries both dispatch and combine, and each
            # combine waits on a grouped GEMM.
            for item in combine:
                assert any(dep.startswith("s0.l0.moe_down") for dep in item.deps)
            assert any(op.family == "barrier" for op in work.ops)
    assert mc2_bytes == pytest.approx(alltoall_bytes)


def test_low_precision_peak_is_declared_not_inferred_from_dtype():
    base = MachineSpec.from_mapping(machine())
    assert base.peak_flops_for("bf16") == base.peak_flops_for("fp16") == 1e14
    # An undeclared low-precision dtype is a loud rejection, never the bf16
    # peak: borrowing it prices an A5 fp8 GEMM 2x too slow (B-1/P5).
    with pytest.raises(A3InputError, match="machine_compute_peak_undeclared:fp8"):
        base.peak_flops_for("fp8")
    with pytest.raises(A3InputError, match="machine_compute_peak_undeclared:int4"):
        base.peak_flops_for("int4")
    declared = MachineSpec.from_mapping({**machine(), "compute_peaks": {"fp8": 2e14}})
    assert declared.peak_flops_for("fp8") == 2e14
    with pytest.raises(A3InputError, match="machine_compute_peak_undeclared:int4"):
        declared.peak_flops_for("int4")


# ---------------------------------------------------------------------------
# Event engine / finite schedules
# ---------------------------------------------------------------------------


def test_shared_and_independent_resource_capacity_conservation():
    shared = ResourceEngine({"mem": ResourceSpec("mem", 1.0)})
    shared.add_task("a", {"mem": 6})
    shared.add_task("b", {"mem": 6})
    result = shared.run()
    assert result["makespan_ms"] == pytest.approx(12.0)
    assert result["serviced"]["mem"] == pytest.approx(12.0)

    independent = ResourceEngine({
        "x": ResourceSpec("x", 1.0), "y": ResourceSpec("y", 1.0),
    })
    independent.add_task("a", {"x": 6})
    independent.add_task("b", {"y": 6})
    assert independent.run()["makespan_ms"] == pytest.approx(6.0)


def test_producer_transfer_consumer_and_chunk_visibility():
    engine = ResourceEngine({
        "compute": ResourceSpec("compute", 1.0),
        "link": ResourceSpec("link", 1.0),
    })
    engine.add_task("producer", {"compute": 8})
    engine.add_task("chunk0", {"link": 3}, dependencies=("producer",))
    engine.add_task("chunk1", {"link": 3}, dependencies=("chunk0",))
    engine.add_task("consumer0", {}, dependencies=("chunk0",))
    engine.add_task("consumer1", {}, dependencies=("chunk1",))
    result = engine.run()
    assert result["completed_at_ms"]["producer"] == pytest.approx(8.0)
    assert result["completed_at_ms"]["chunk0"] == pytest.approx(11.0)
    assert result["completed_at_ms"]["consumer0"] == pytest.approx(11.0)
    assert result["completed_at_ms"]["chunk1"] == pytest.approx(14.0)
    assert result["completed_at_ms"]["consumer1"] == pytest.approx(14.0)


def test_dynamic_active_set_recomputes_when_work_arrives_late():
    engine = ResourceEngine({"mem": ResourceSpec("mem", 1.0)})
    engine.add_task("A", {"mem": 6}, release_ms=0.0)
    engine.add_task("B", {"mem": 6}, release_ms=3.0)
    result = engine.run()
    assert result["completed_at_ms"] == {"A": 9.0, "B": 12.0}


def test_pp_microbatch_finite_schedule_and_transfer():
    plain = simulate_microbatch_pipeline([3, 5], 4)
    assert plain["first_token_ms"] == pytest.approx(8.0)
    assert plain["last_token_ms"] == pytest.approx(23.0)
    transferred = simulate_microbatch_pipeline([3, 5], 4, transfer_ms=1.0)
    assert transferred["first_token_ms"] == pytest.approx(9.0)
    assert transferred["last_token_ms"] == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# 16K scheduler and injected cost oracle
# ---------------------------------------------------------------------------


def test_16k_token_budget_boundaries_with_injected_cost_backend():
    backend = TokenLinearCostBackend(base_ms=1.0, per_token_ms=0.001)
    base = case(
        workload={"requests": [{"id": "r0", "prompt_tokens": 16384, "output_tokens": 1}]},
        runtime={"max_num_batched_tokens": 16384, "max_model_len": 40000},
    )
    exact = predict(base, cost_backend=backend)
    assert exact["requests"][0]["ttft_ms"] == pytest.approx(17.384)
    assert exact["iterations"] == 1

    plus = case(
        workload={"requests": [{"id": "r0", "prompt_tokens": 16385, "output_tokens": 1}]},
        runtime={"max_num_batched_tokens": 16384, "max_model_len": 40000},
    )
    plus_result = predict(plus, cost_backend=backend)
    assert plus_result["requests"][0]["ttft_ms"] == pytest.approx(18.385)
    assert plus_result["iterations"] == 2
    assert [trace["valid_tokens"] for trace in plus_result["traces"]] == [16384, 1]

    mixed = case(
        workload={"requests": [
            {"id": "r0", "prompt_tokens": 16385, "output_tokens": 1},
            {"id": "r1", "prompt_tokens": 16384, "output_tokens": 1},
        ]},
        runtime={"max_num_batched_tokens": 16384, "max_model_len": 40000},
    )
    mixed_result = predict(mixed, cost_backend=backend)
    by_id = {item["request_id"]: item for item in mixed_result["requests"]}
    assert by_id["r0"]["ttft_ms"] == pytest.approx(34.768)
    assert by_id["r1"]["ttft_ms"] == pytest.approx(35.769)
    assert [trace["valid_tokens"] for trace in mixed_result["traces"]] == [16384, 16384, 1]


# ---------------------------------------------------------------------------
# Should-cost / search
# ---------------------------------------------------------------------------


def test_should_cost_L_le_U_and_normalized_algorithms():
    payload = should_cost(case())
    assert payload["status"] == "modeled"
    assert payload["hardware_lower_bound_ms"] is not None
    assert payload["feasible_target_ms"] is not None
    assert payload["hardware_lower_bound_ms"] <= payload["feasible_target_ms"] + 1e-9
    assert payload["bound_consistency"]["lower_bound_le_target"] is True
    assert payload["optimal_in_family"] is True
    assert payload["predictive_not_used_as_input"] is True

    cv = normalized_shared_cv_search()
    assert cv["resource_lower_bound_ms"] == pytest.approx(12.0)
    assert cv["selected_target_ms"] == pytest.approx(14.0)
    assert cv["resource_lower_bound_is_feasible_target"] is False
    assert {plan["target_ms"] for plan in cv["plans"]} == {20.0, 22.0, 14.0}
    assert all(plan["trace"] for plan in cv["plans"])

    chunks = normalized_chunked_pipeline_search()
    assert [(item["chunks"], item["makespan_ms"]) for item in chunks["plans"]] == [
        (1, pytest.approx(18.6)), (2, pytest.approx(13.8)),
        (4, pytest.approx(11.7)), (8, pytest.approx(11.25)),
        (16, pytest.approx(12.225)),
    ]
    assert chunks["selected"]["chunks"] == 8


def test_search_excludes_oom_and_slo_violations_and_reruns_recommendation():
    payload = {
        **case(runtime={"tp": 1, "dp": 1, "pp": 1}),
        "slo": {"ttft_ms": 10_000.0, "tpot_ms": 10_000.0, "e2e_ms": 100_000.0},
        "search": {
            "objective": "throughput", "tp": [1, 2], "dp": [1, 2], "pp": [1],
            "max_num_batched_tokens": [8, 16], "graph_buckets": [[], [16]],
        },
    }
    result = sim.search(payload)
    assert result["status"] == "modeled"
    recommended = result["recommended"]
    assert recommended is not None
    assert result["recheck_matches"] is True
    assert all(
        item["config"]["tp"] * item["config"]["dp"] <= 2
        for item in result["candidates"] if item["feasible"]
    )
    # Impossible SLO: no violating candidate may be recommended.
    impossible = {**payload, "slo": {"ttft_ms": 1e-9, "tpot_ms": 1e-9, "e2e_ms": 1e-9}}
    impossible["search"] = {**payload["search"], "max_candidates": 32}
    blocked = sim.search(impossible)
    assert blocked["status"] == "no_feasible_candidate"
    assert blocked["recommended"] is None


# ---------------------------------------------------------------------------
# Calibration store and measurement loop
# ---------------------------------------------------------------------------


def test_calibration_store_dedup_revision_metadata_and_manifest_roundtrip(tmp_path):
    db = tmp_path / "calibration.sqlite"
    store = SQLiteCalibrationStore(db)
    first = store.record_sample(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 64, "n": 128, "k": 64}, duration_ms=1.2,
        analytic_ms=1.0, source="normalized-cpu", source_class="measured",
        hardware_key="normalized", revision="rev-a",
    )
    assert first["status"] == "inserted"
    duplicate = store.record_sample(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 64, "n": 128, "k": 64}, duration_ms=1.2,
        analytic_ms=1.0, source="normalized-cpu", source_class="measured",
        hardware_key="normalized", revision="rev-b",
    )
    assert duplicate["status"] == "inserted"
    assert duplicate["sample_key"] == first["sample_key"]
    assert duplicate["measurement_key"] != first["measurement_key"]
    idempotent = store.record_sample(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 64, "n": 128, "k": 64}, duration_ms=1.2,
        analytic_ms=1.0, source="normalized-cpu", source_class="measured",
        hardware_key="normalized", run_id="run-a", case_id="case-a",
        repeat_index=0,
    )
    same = store.record_sample(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 64, "n": 128, "k": 64}, duration_ms=1.2,
        analytic_ms=1.0, source="normalized-cpu", source_class="measured",
        hardware_key="normalized", run_id="run-a", case_id="case-a",
        repeat_index=0,
    )
    assert idempotent["status"] == "inserted"
    assert same["status"] == "duplicate"
    assert store.inventory()["measurements"] == 3
    assert store.inventory()["conditions"] == 1
    with pytest.raises(A3InputError):
        store.record_sample(
            op_family="gemm", phase="prefill", recipe="dense",
            shape={"m": 1}, duration_ms=1.0, analytic_ms=1.0,
            source="synthetic", source_class="measured", synthetic=True,
        )
    request = store.record_measurement_request(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 128, "n": 128, "k": 64}, predicted_ms=2.0,
        error_ratio=0.2, extrapolation=0.5, target_impact=0.25,
        hardware_key="normalized",
    )
    again = store.record_measurement_request(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 128, "n": 128, "k": 64}, predicted_ms=2.0,
        hardware_key="normalized",
    )
    assert again["status"] == "deduplicated"
    assert again["query_count"] == 2
    manifest = store.manifest(limit=4)
    assert manifest["status"] == "ready"
    assert manifest["requests"][0]["priority"] > 1.0
    assert manifest["runner_contract"]["no_npu_lease_created_by_sim"] is True
    assert manifest["requests"][0]["request_key"] == request["request_key"]
    # CPU closed loop: manifest -> runner -> sample -> calibration view.
    run = run_pending(store, CpuMatmulRunner(iterations=1), limit=1, run_id="cpu-run-1")
    assert run["completed"]
    assert store.inventory()["calibratable_measurements"] >= 3
    assert store.inventory()["requests"] == {"completed": 1}
    assert len(store.as_calibration_store().observations()) == 1
    assert store.coverage()["groups"]


def test_current_calibration_contract_end_to_end_preserves_identity(tmp_path):
    db = tmp_path / "current-contract.sqlite"
    store = SQLiteCalibrationStore(db)
    with store._connect() as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "measurements", "sample_statistics", "measurement_requests",
    } <= tables
    assert "meta" not in tables
    common = {
        "op_family": "gemm", "phase": "prefill", "recipe": "dense",
        "shape": {"m": 64, "n": 128, "k": 64}, "duration_ms": 1.2,
        "analytic_ms": 1.0, "source": "normalized-capture", "source_class": "measured",
        "hardware_key": "normalized", "dtype": "bf16", "layout": "contiguous",
        "timing_scope": "device_event_warm",
        "protocol": {"timer": "device_event", "warmup": 3, "iterations": 10},
    }
    for run_id in ("run-a", "run-b"):
        store.record_sample(
            **common, run_id=run_id, case_id="case-1", repeat_index=0,
        )
    store.record_sample(
        **common, run_id="run-h", case_id="case-h",
        split="heldout",
    )
    store.record_sample(**{
        **common, "run_id": "run-c", "case_id": "case-c",
        "protocol": {
            "timer": "device_event", "warmup": 3, "iterations": 10,
            "mode": "graph_block", "capture_invocations": 32,
        },
    })
    stats = store.statistics()
    assert len(stats) == 2
    eager = next(item for item in stats if not item["protocol"].get("mode"))
    assert eager["measurement_count"] == 2
    assert eager["run_count"] == 2
    assert eager["run_ids"] == ["run-a", "run-b"]
    assert store.statistics(split="heldout")[0]["run_ids"] == ["run-h"]
    inventory = store.inventory()
    assert inventory["measurements"] == 4
    assert inventory["conditions"] == 3
    assert inventory["requests"] == {}
    coverage = store.coverage()
    coverage_splits = {
        split for group in coverage["groups"] for split in group["splits"]
    }
    assert coverage_splits == {"train", "heldout"}

    request = store.record_measurement_request(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 128, "n": 128, "k": 64}, predicted_ms=2.0,
        analytic_ms=1.0, hardware_key="normalized",
    )
    manifest_path = tmp_path / "manifest.json"
    exported = store.export_manifest(manifest_path, limit=4)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exported["status"] == "written"
    assert exported["requests"] == 1
    assert manifest["status"] == "ready"
    assert manifest["requests"][0]["request_key"] == request["request_key"]
    assert manifest["requests"][0]["request_status"] == "pending"
    assert "schema_version" not in manifest
    assert "schema_version" not in inventory
    assert "schema_version" not in coverage

    ingested = store.ingest_results({"results": [{
        "request_key": manifest["requests"][0]["request_key"],
        "duration_ms": 2.4, "analytic_ms": 1.0,
        "source": "managed-runner", "source_class": "measured",
        "run_id": "run-measured", "split": "train",
        "timing_scope": "device_event_warm",
        "protocol": {"timer": "device_event", "warmup": 3, "iterations": 10},
        "environment": {"device_name": "normalized"},
    }]})
    assert len(ingested["completed"]) == 1
    assert not ingested["failed"]
    completed = store.requests(status="completed")
    assert len(completed) == 1
    assert completed[0]["request_key"] == request["request_key"]
    assert completed[0]["run_id"] == "run-measured"
    assert completed[0]["status"] == "completed"
    measured = [
        sample for sample in store.samples() if sample.run_id == "run-measured"
    ]
    assert len(measured) == 1
    assert measured[0].source_class == "measured"
    assert measured[0].protocol == {
        "iterations": 10, "timer": "device_event", "warmup": 3,
    }
    assert store.manifest()["status"] == "empty"

    reopened = SQLiteCalibrationStore(db)
    assert reopened.inventory()["measurements"] == 5
    assert reopened.requests(status="completed")[0]["request_key"] == (
        request["request_key"]
    )
    assert len(reopened.statistics()) == 3


def test_predict_can_opt_in_to_measurement_requests(tmp_path):
    db = tmp_path / "predict-misses.sqlite"
    payload = predict(case())
    store = SQLiteCalibrationStore(db)
    recorded = record_prediction_misses(store, payload, hardware_key="normalized")
    assert recorded["recorded_requests"] > 0
    assert store.requests(status="pending")
    # Default predict call itself has no DB side effect.
    assert not (tmp_path / "default.sqlite").exists()


# ---------------------------------------------------------------------------
# End-to-end representative models and revision semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [
    dense_model(),
    dense_model(attention="mla", architecture="mla",
                mla={"q_lora_rank": 32, "kv_lora_rank": 16,
                     "qk_nope_head_dim": 12, "qk_rope_head_dim": 4, "v_head_dim": 16}),
    moe_model(),
])
def test_representative_dense_mla_moe_predictions_are_numeric(model):
    payload = predict(case(model, runtime={"tp": 2, "ep": 2 if model.get("num_experts") else 1}))
    assert payload["status"] in {"modeled", "partial"}
    if payload["status"] == "modeled":
        metrics = payload["metrics"]
        assert metrics["ttft_ms"]["p50"] is not None
        assert metrics["throughput_tokens_per_s"] > 0
        assert payload["traces"]
        assert all(trace["op_count"] > 0 for trace in payload["traces"])


def test_same_semantics_different_revision_keeps_prediction_available():
    left = case()
    right = json.loads(json.dumps(left))
    right["revision"] = "different-revision"
    right["source"] = "another metadata label"
    first = predict(left)
    second = predict(right)
    assert first["status"] == second["status"] == "modeled"
    assert first["metrics"] == second["metrics"]
    # No format-version allowance: only actual business/provenance fields parse.
    with pytest.raises(A3InputError, match="unknown_field:case.schema_version"):
        CaseSpec.from_mapping({**left, "schema_version": "historical-label"})


def test_gemm_and_attention_oracles():
    assert gemm_flops(2, 4, 8) == 128
    assert attention_flops(1, 4096, 8, 16) == pytest.approx(4096 * attention_flops(1, 1, 8, 16))
    assert attention_flops(4, 8, 2, 16, causal=True, context=8) > attention_flops(1, 8, 2, 16)


def test_budget_b_minus_one_two_b_and_prefix_cached_first_token():
    backend = TokenLinearCostBackend(base_ms=1.0, per_token_ms=0.001)
    for prompt, expected_iterations, expected_ttft in (
        (16383, 1, 17.383),
        (32768, 2, 34.768),
        (32769, 3, 35.769),
    ):
        payload = predict(case(
            workload={"requests": [{"id": "r", "prompt_tokens": prompt, "output_tokens": 1}]},
            runtime={"max_num_batched_tokens": 16384, "max_model_len": 40000},
        ), cost_backend=backend)
        assert payload["iterations"] == expected_iterations
        assert payload["requests"][0]["ttft_ms"] == pytest.approx(expected_ttft)

    cached = predict(case(
        workload={"requests": [{"id": "cached", "prompt_tokens": 1024,
                               "prefix_tokens": 512, "output_tokens": 1}]},
        runtime={"max_num_batched_tokens": 16384, "max_model_len": 40000},
    ), cost_backend=backend)
    assert cached["requests"][0]["ttft_ms"] == pytest.approx(1.512)
    assert cached["traces"][0]["valid_tokens"] == 512


def test_calibration_changes_prediction_and_revision_is_metadata():
    base = predict(case())
    # A served attention kernel is the fused one (QK+softmax+PV on chip); its
    # launch shape still declares (q, kv, heads, dim), which is the physical key.
    qk = next(
        op for op in base["traces"][0]["ops"]
        if op["family"] == "attention_fused"
    )
    shape = qk["launch_shape"]
    observation = {
        "op_family": "attention_fused", "phase": qk["phase"], "recipe": qk["recipe"],
        "shape": {"q": shape[0], "kv": shape[1], "heads": shape[2], "dim": shape[3]},
        "duration_ms": float(qk["analytic_ms"]) * 8.0,
        "analytic_ms": float(qk["analytic_ms"]),
        "source": "normalized-observed", "source_class": "measured",
        "dtype": "bf16", "layout": "contiguous",
        # A measured residual is hardware-keyed.  This observation used to omit
        # the identity and matched every machine through an empty-machine_key
        # wildcard -- the exact mechanism that let A3 residuals reach A2.
        "machine_key": "normalized-test-die",
        "features": {"weight_dtype": "bf16", "activation_dtype": "bf16",
                     "compute_dtype": "bf16", "accumulation_dtype": "backend_default",
                     "quantization": "none"},
        "timing_scope": "device_event_warm",
    }
    calibrated = predict(case(), calibration={"observations": [observation]})
    assert calibrated["metrics"]["ttft_ms"]["p50"] > base["metrics"]["ttft_ms"]["p50"]
    # Revision labels do not alter the semantic sample key or factor.
    altered = dict(observation)
    altered["revision"] = "rev-other"
    again = predict(case(), calibration={"observations": [altered]})
    assert again["metrics"] == calibrated["metrics"]


def test_normative_best_does_not_depend_on_current_parallel_config():
    left = case(runtime={"tp": 1, "dp": 1, "pp": 1}, machine=machine(4))
    right = case(runtime={"tp": 2, "dp": 2, "pp": 1}, machine=machine(4))
    left_result = should_cost(left)
    right_result = should_cost(right)
    assert left_result["hardware_lower_bound_ms"] == pytest.approx(right_result["hardware_lower_bound_ms"])
    assert left_result["feasible_target_ms"] == pytest.approx(right_result["feasible_target_ms"])
    assert left_result["selected_plan"] == right_result["selected_plan"]


def test_cli_predict_should_search_and_pipeline_microbatches(tmp_path, capsys):
    from mindie_sim.__main__ import main

    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case()), encoding="utf-8")
    assert main(["predict", "--input", str(case_path)]) == 0
    predicted = json.loads(capsys.readouterr().out)
    assert predicted["command"] == "predict"
    assert predicted["status"] == "modeled"
    assert predicted["metrics"]["ttft_ms"]["p50"] is not None
    assert "schema_version" not in predicted

    assert main(["should", "--input", str(case_path)]) == 0
    normative = json.loads(capsys.readouterr().out)
    assert normative["command"] == "should"
    assert normative["hardware_lower_bound_ms"] <= normative["feasible_target_ms"] + 1e-9
    assert "schema_version" not in normative

    search_case = {**case(), "search": {
        "objective": "throughput", "tp": [1], "dp": [1], "pp": [1],
        "max_num_batched_tokens": [8, 16], "graph_buckets": [[]],
    }}
    search_path = tmp_path / "search.json"
    search_path.write_text(json.dumps(search_case), encoding="utf-8")
    assert main(["search", "--input", str(search_path)]) == 0
    searched = json.loads(capsys.readouterr().out)
    assert searched["command"] == "search"
    assert searched["status"] == "modeled"
    assert "schema_version" not in searched

    assert main(["pipeline-microbatches", "--stages", "3,5",
                 "--microbatches", "4", "--transfer-ms", "1"]) == 0
    micro = json.loads(capsys.readouterr().out)
    assert micro["first_token_ms"] == pytest.approx(9.0)
    assert micro["last_token_ms"] == pytest.approx(24.0)


def test_cli_public_errors_are_visible_without_format_labels(tmp_path, capsys):
    from mindie_sim.__main__ import main

    bad = case()
    bad["runtime"]["gpu_memory_utilization"] = 2.0
    bad_path = tmp_path / "bad-case.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    for command in ("predict", "should", "search"):
        assert main([command, "--input", str(bad_path)]) == 1
        error = json.loads(capsys.readouterr().out)
        assert error["status"] == "error"
        assert error["command"] == command
        assert error["reason"]
        assert "schema_version" not in error

    missing = tmp_path / "missing-manifest.json"
    assert main([
        "campaign", "analyze", "--manifest", str(missing),
        "--results", str(missing),
    ]) == 1
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"
    assert "input_unreadable" in error["reason"]
    assert "schema_version" not in error


def test_campaign_analyze_routes_by_business_document_shape(tmp_path):
    from mindie_sim.__main__ import _campaign_analyze_argv

    worker = {
        "campaign_id": "worker-campaign",
        "requests": [{
            "request_key": "r0", "condition": {"family": "gemm"},
            "shape": {"m": 1},
        }],
    }
    wave = {
        "run_id": "wave-run",
        "requests": [{
            "measurement_id": "m0", "point": {"shape": {"tokens": 1}},
            "origins": [{"axis": "tokens", "anchor_width": 128}],
        }],
    }
    worker_path = tmp_path / "worker.json"
    wave_path = tmp_path / "wave.json"
    worker_path.write_text(json.dumps(worker), encoding="utf-8")
    wave_path.write_text(json.dumps(wave), encoding="utf-8")
    worker_argv = ["--manifest", str(worker_path), "--results", "results.json"]
    wave_argv = ["--manifest", str(wave_path), "--results", "results.json"]
    assert _campaign_analyze_argv(worker_argv) == worker_argv
    assert _campaign_analyze_argv(wave_argv) == [
        "--adapter", "campaign", *wave_argv,
    ]
    explicit = ["--adapter", "linearity", "--input", "cases.json"]
    assert _campaign_analyze_argv(explicit) == explicit

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps({"requests": [{"run_id": "r"}]}), encoding="utf-8",
    )
    with pytest.raises(A3InputError):
        _campaign_analyze_argv([
            "--manifest", str(invalid_path), "--results", "results.json",
        ])


def test_cli_calibration_manifest_and_ingest_roundtrip(tmp_path, capsys):
    from mindie_sim.__main__ import main

    db = tmp_path / "cli.sqlite"
    store = SQLiteCalibrationStore(db)
    request = store.record_measurement_request(
        op_family="gemm", phase="prefill", recipe="dense",
        shape={"m": 16, "n": 16, "k": 16}, predicted_ms=1.0,
    )
    assert main(["calibration", "--db", str(db), "inventory"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["requests"] == {"pending": 1}
    manifest_path = tmp_path / "manifest.json"
    assert main(["calibration", "--db", str(db), "manifest",
                 "--limit", "4", "--output", str(manifest_path)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["status"] == "ready"
    assert manifest_path.exists()

    results = tmp_path / "results.json"
    results.write_text(json.dumps({"results": [{
        "request_key": request["request_key"], "duration_ms": 1.5,
        "analytic_ms": 1.0, "source": "external-managed-runner",
        "source_class": "measured", "run_id": "run-42",
    }]}), encoding="utf-8")
    assert main(["calibration", "--db", str(db), "ingest", "--results", str(results)]) == 0
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["completed"]
    assert store.inventory()["calibratable_measurements"] == 1


REAL_TRAINING = Path("/tmp/mindie-a3-dsh.DNIfZY/a3-training-measurements.json")
NATIVE_CAPTURE = Path("/tmp/mindie-a3-dsh.DNIfZY/native-scheduler-capture.json")


@pytest.mark.skipif(not REAL_TRAINING.exists(), reason="real A3 training capture is not mounted")
def test_real_a3_training_import_raw_repeats_calibration_and_cv():
    from mindie_sim.a3_real_data import (
        CvCalibration, import_a3_measurements, machine_from_environment,
    )

    store = SQLiteCalibrationStore(":memory:")
    environment = json.loads(REAL_TRAINING.read_text(encoding="utf-8"))["environment"]
    first = import_a3_measurements(store, REAL_TRAINING)
    second = import_a3_measurements(store, REAL_TRAINING)
    assert first["measurements"] >= 330
    assert second["measurements"] == first["measurements"]
    inventory = store.inventory()
    assert inventory["measurements"] == first["measurements"]
    assert inventory["conditions"] >= 20
    assert inventory["repeats_per_condition"]["max"] >= 11
    stats = store.statistics()
    assert any(item["op_family"] == "gemm" and item["calibratable_count"] == 11 for item in stats)
    assert all(item["source_classes"] == ["measured"] for item in stats)
    calibration = store.as_calibration_store()
    assert len(calibration.observations()) >= 20
    machine = machine_from_environment(environment)
    assert machine.hardware_key == "Ascend910_9362"
    gemm_rows = [item for item in stats if item["op_family"] == "gemm"]
    query = gemm_rows[0]
    result = calibration.factor_for(
        op_family="gemm", phase="operator", recipe="torch_mm_out",
        shape=query["shape"], machine_key=machine.hardware_key,
        dtype="bf16", layout="contiguous",
        features={"weight_dtype": "bf16", "activation_dtype": "bf16",
                  "compute_dtype": "bf16", "accumulation_dtype": "backend_default",
                  "quantization": "none"},
        timing_scope="device_event_warm",
        protocol=query["protocol"],
    )
    assert result.status == "calibrated_exact"
    assert result.exact_matches >= 1
    assert result.factor > 0
    cv = CvCalibration.from_store(store)
    assert cv.conditions
    predicted = cv.predict(
        c_shape_mkn=(512, 2048, 2048), v_elements=4_194_304, machine=machine,
    )
    assert predicted["status"].startswith("calibrated")
    assert predicted["span_ms"] > 0
    assert predicted["slowdowns"]["joint_scale"] > 0


@pytest.mark.skipif(not NATIVE_CAPTURE.exists(), reason="native scheduler capture is not mounted")
def test_native_scheduler_capture_chunking_and_phase_fields():
    from mindie_sim.a3_framework import TokenLinearCostBackend

    capture = json.loads(NATIVE_CAPTURE.read_text(encoding="utf-8"))
    backend = TokenLinearCostBackend()
    for native_case in capture["cases"]:
        requests = [
            {"id": f"r{index}", "prompt_tokens": prompt, "output_tokens": 1}
            for index, prompt in enumerate(native_case["prompt_lengths"])
        ]
        payload = predict(case(
            workload={"requests": requests},
            runtime={
                "max_num_batched_tokens": native_case["token_budget"],
                "max_model_len": 40000,
                "long_prefill_token_threshold": native_case.get("long_prefill_token_threshold", 0),
            },
        ), cost_backend=backend)
        planned = []
        for trace in payload["traces"]:
            scheduled = {}
            for view in trace["requests"]:
                scheduled[view["request_id"]] = scheduled.get(view["request_id"], 0) + view["q_len"]
                if view["q_len"] == 1 and view["semantic_phase"] == "prefill":
                    assert view["execution_mode"] == "decode"
            if scheduled:
                planned.append(scheduled)
        expected = [step["scheduled"] for step in native_case["steps"]
                    if step.get("requires_model_forward")]
        assert planned == expected, native_case["name"]


def test_ttft_is_latency_and_absolute_timestamp_is_separate():
    backend = TokenLinearCostBackend(base_ms=1.0, per_token_ms=0.001)
    payload = predict(case(
        workload={"requests": [{
            "id": "late", "arrival_ms": 100.0,
            "prompt_tokens": 8, "output_tokens": 1,
        }]},
    ), cost_backend=backend)
    request = payload["requests"][0]
    assert request["ttft_ms"] == pytest.approx(1.008)
    assert request["first_token_abs_ms"] == pytest.approx(101.008)
    assert request["e2e_ms"] == pytest.approx(1.008)


def test_nonchunked_admission_honors_budget_and_long_prefill_threshold():
    with pytest.raises(A3InputError):
        CaseSpec.from_mapping(case(
            workload={"requests": [{"prompt_tokens": 8, "output_tokens": 1}]},
            runtime={"max_num_batched_tokens": 4, "enable_chunked_prefill": False},
        ))
    backend = TokenLinearCostBackend()
    payload = predict(case(
        workload={"requests": [{"prompt_tokens": 100, "output_tokens": 1}]},
        runtime={"max_num_batched_tokens": 1000, "long_prefill_token_threshold": 32,
                 "max_model_len": 256},
    ), cost_backend=backend)
    assert [trace["valid_tokens"] for trace in payload["traces"]] == [32, 32, 32, 4]


def test_cross_dp_ep_joins_token_sources_and_per_expert_weights():
    model = ModelSpec.from_mapping(dense_model(
        hidden_size=256, num_heads=4, num_kv_heads=1, head_dim=64,
        intermediate_size=512, num_experts=8, top_k=2,
        moe_intermediate_size=128, num_shared_experts=1,
    ))
    runtime = RuntimeSpec.from_mapping({
        "max_num_batched_tokens": 256, "max_model_len": 512,
        "dp": 2, "tp": 1, "ep": 2, "moe_recipe": "alltoall",
    })
    work = derive_iteration_work(
        IterationPlan((RequestView("r", 256, 0, True, "prefill"),)),
        model, runtime, MachineSpec.from_mapping(machine(2)),
        ParallelSpec(dp=2, tp=1, ep=2),
    )
    # Each expert rank sees both DP token sources: 2*256 routing tokens over 8
    # experts => 4 local experts * 128 tokens * 6*H*moe_dim per rank.
    per_rank_moe = sum(op.flops for op in work.ops if op.family in {"moe_gate_up", "moe_down"})
    assert per_rank_moe == pytest.approx(100_663_296)
    gate_ups = [op for op in work.ops if op.family == "moe_gate_up"]
    assert len(gate_ups) == len(work.routing.local_expert_counts)
    assert sum(op.mte2_bytes for op in gate_ups) >= 4 * 256 * 256 * 2  # four expert gate weights
    assert any(op.family == "shared_expert_down" for op in work.ops)


def test_predict_kv_transfer_changes_ttft_and_shares_link():
    base = predict(case(
        workload={"requests": [{"prompt_tokens": 128, "output_tokens": 2}]},
        runtime={"tp": 1, "max_num_batched_tokens": 256, "max_model_len": 512},
    ))
    kv = predict(case(
        workload={"requests": [{"prompt_tokens": 128, "output_tokens": 2}]},
        runtime={
            "tp": 1, "max_num_batched_tokens": 256, "max_model_len": 512,
            "kv_transfer": {"enabled": True, "chunks": 4, "bytes_per_token": 65536},
        },
    ))
    assert kv["requests"][0]["ttft_ms"] > base["requests"][0]["ttft_ms"]
    assert any(
        trace["end_ms"] >= trace["start_ms"] for trace in kv["traces"]
    )
    assert any(
        any(event["task_id"].endswith("kv_transfer.c0") for event in trace["events"])
        for trace in kv["traces"]
    )
    # Same-link contention control: two transfer tasks share one link, while
    # disjoint links remain parallel.
    shared = ResourceEngine({"link": ResourceSpec("link", 1.0)})
    shared.add_task("a", {"link": 6})
    shared.add_task("b", {"link": 6})
    assert shared.run()["makespan_ms"] == pytest.approx(12.0)
    disjoint = ResourceEngine({
        "a": ResourceSpec("a", 1.0), "b": ResourceSpec("b", 1.0),
    })
    disjoint.add_task("a", {"a": 6})
    disjoint.add_task("b", {"b": 6})
    assert disjoint.run()["makespan_ms"] == pytest.approx(6.0)


def test_future_release_capacity_and_zero_time_trace_contract():
    engine = ResourceEngine({"r": ResourceSpec("r", 10.0)})
    engine.add_task("a", {"r": 100.0}, release_ms=0.0)
    engine.add_task("b", {"r": 100.0}, release_ms=5.0)
    result = engine.run()
    assert result["completed_at_ms"] == {"a": 15.0, "b": 20.0}
    assert result["serviced"]["r"] == pytest.approx(200.0)
    zero = ResourceEngine({"r": ResourceSpec("r", 1.0)})
    zero.add_task("z", {})
    trace = zero.run()["trace"][0]
    assert trace["start_ms"] == 0.0
    assert trace["end_ms"] == 0.0


def test_should_single_request_does_not_scale_with_idle_dp_and_memory_default_independent():
    base = {
        "machine": {**machine(4), "device": {
            **machine(4)["device"], "cube_flops_per_s": 1e12,
            "vector_ops_per_s": 1e12, "launch_us": 0.0,
        }},
        "model": {"layers": 4, "hidden_size": 64, "num_heads": 1,
                  "num_kv_heads": 1, "head_dim": 64, "intermediate_size": 64,
                  "vocab_size": 1000},
        "workload": {"requests": [{"prompt_tokens": 1024, "output_tokens": 32}]},
        "runtime": {"max_num_batched_tokens": 1024, "max_model_len": 4096},
    }
    result = should_cost(base, constraints={
        "tp": [1], "pp": [1], "fusion": False, "graph_padding": False,
    })
    by_dp = {}
    for plan in result["plans"]:
        by_dp[plan["plan"]["dp"]] = plan["target_ms"]
    assert 1 in by_dp and 4 in by_dp
    assert by_dp[1] == pytest.approx(by_dp[4])
    # Normative memory constraints do not inherit a tiny current runtime
    # utilization setting.
    tight = {**base, "runtime": {**base["runtime"], "gpu_memory_utilization": 0.0001}}
    tight_result = should_cost(tight, constraints={
        "tp": [1], "pp": [1], "dp": [1], "fusion": False, "graph_padding": False,
    })
    assert tight_result["status"] == "modeled"


def test_qkv_lowering_uses_local_kv_head_replication():
    model = ModelSpec.from_mapping(dense_model(
        hidden_size=256, num_heads=8, num_kv_heads=2, head_dim=32,
        intermediate_size=512,
    ))
    work = derive_iteration_work(
        IterationPlan((RequestView("r", 8, 0, True, "prefill"),)),
        model,
        RuntimeSpec.from_mapping({"tp": 4, "max_num_batched_tokens": 16, "max_model_len": 128}),
        MachineSpec.from_mapping(machine(4)),
        ParallelSpec(tp=4),
    )
    qkv = next(op for op in work.ops if op.family == "qkv_projection")
    assert qkv.launch_shape == (8, 128, 256)
    assert qkv.flops == pytest.approx(2 * 8 * 128 * 256)


def test_search_uses_peak_resident_kv_not_lifetime_sum():
    payload = {
        "machine": {
            "device": {
                "hbm_gib": 300_000_000 / (1024 ** 3),
                "cube_flops_per_s": 1e14, "vector_ops_per_s": 1e13,
                "hbm_bw_bytes_per_s": 1e12, "mte_bw_bytes_per_s": 2e12,
                "launch_us": 1.0,
            },
            "num_devices": 1, "hardware_key": "seq-test",
        },
        "model": dense_model(hidden_size=64, num_heads=1, head_dim=64, intermediate_size=64),
        "workload": {"requests": [
            {"id": "a", "prompt_tokens": 1881, "output_tokens": 1, "arrival_ms": 0},
            {"id": "b", "prompt_tokens": 1881, "output_tokens": 1, "arrival_ms": 100000},
        ]},
        "runtime": {"max_num_batched_tokens": 1024, "max_model_len": 4096,
                    "kv_block_size": 16},
        "slo": {"ttft_ms": 1e12},
        "search": {"tp": [1], "dp": [1], "pp": [1],
                   "max_num_batched_tokens": [1024], "graph_buckets": [[]]},
    }
    result = sim.search(payload)
    assert result["status"] == "modeled"
    assert result["recommended"] is not None


def test_heldout_samples_are_not_used_by_default_and_details_persist():
    store = SQLiteCalibrationStore(":memory:")
    request = store.record_measurement_request(
        op_family="gemm", phase="operator", recipe="r",
        shape={"m": 1}, predicted_ms=1.0,
    )
    completed = store.complete_request_samples(
        request["request_key"], raw_samples_ms=[1.0, 2.0], analytic_ms=1.0,
        source="managed", run_id="run-x", split="heldout",
        details={"case_id": "held-case", "timing_scope": "warm",
                 "protocol": {"p": 1}, "environment": {"e": 2}},
    )
    assert completed["sample_count"] == 2
    assert store.inventory()["measurements"] == 2
    assert store.as_calibration_store().observations() == ()
    raw = store.samples(split="heldout")
    assert raw[0].details["timing_scope"] == "warm"
    assert raw[0].protocol == {"p": 1}
    assert raw[0].environment == {"e": 2}
    # Failed ingest marks the request failed instead of leaving it running.
    failed_request = store.record_measurement_request(
        op_family="gemm", phase="operator", recipe="r",
        shape={"m": 2}, predicted_ms=1.0,
    )
    store.update_request_status(failed_request["request_key"], status="running", run_id="run-y")
    store.ingest_results({"results": [{
        "request_key": failed_request["request_key"], "status": "failed",
        "reason": "device_error", "run_id": "run-y",
    }]})
    failed = store.requests(status="failed")
    assert any(item["request_key"] == failed_request["request_key"] for item in failed)


def test_generic_dense_and_moe_ffn_materialize_swiglu_activation():
    dense = ModelSpec.from_mapping(dense_model(
        hidden_size=64, intermediate_size=128, num_heads=4, head_dim=16,
        num_kv_heads=1, layers=1,
    ))
    dense_work = derive_iteration_work(
        IterationPlan((RequestView("r", 8, 0, True, "prefill"),)),
        dense,
        RuntimeSpec.from_mapping({
            "tp": 1, "max_num_batched_tokens": 16, "max_model_len": 128,
        }),
        MachineSpec.from_mapping(machine(1)),
        ParallelSpec(tp=1),
    )
    by_id = {op.op_id: op for op in dense_work.ops}
    gate = by_id["s0.l0.gate_up"]
    act = by_id["s0.l0.act"]
    down = by_id["s0.l0.down"]
    assert gate.semantic_shape[1] == 2 * 128
    assert act.family == "swiglu"
    assert act.semantic_shape == (8, 128)
    assert act.local_shape == (8, 128)
    assert act.vector_ops == pytest.approx(8 * 128 * 4.0)
    assert act.mte2_bytes == pytest.approx(8 * 128 * 2 * 2.0)
    assert act.mte3_bytes == pytest.approx(8 * 128 * 2.0)
    assert act.hbm_bytes == pytest.approx(8 * 128 * 3 * 2.0)
    assert act.deps == (gate.op_id,)
    assert down.deps == (act.op_id,)

    def moe_ops(recipe):
        moe = ModelSpec.from_mapping(moe_model(
            hidden_size=64, intermediate_size=128, num_heads=4, head_dim=16,
            num_kv_heads=1, layers=1, moe_intermediate_size=64,
            num_shared_experts=1,
        ))
        work = derive_iteration_work(
            IterationPlan((RequestView("r", 8, 0, True, "prefill"),)),
            moe,
            RuntimeSpec.from_mapping({
                "dp": 2, "tp": 1, "ep": 2, "moe_recipe": recipe,
                "mc2_chunks": 2, "max_num_batched_tokens": 16,
                "max_model_len": 128,
            }),
            MachineSpec.from_mapping(machine(2)),
            ParallelSpec(dp=2, tp=1, ep=2),
        )
        return {op.op_id: op for op in work.ops}

    for recipe in ("alltoall", "mc2"):
        moe_by_id = moe_ops(recipe)
        gates = [
            op for op in moe_by_id.values() if op.family == "moe_gate_up"
        ]
        assert gates
        for gate in gates:
            expert = dict(gate.metadata)["expert"]
            act_id = gate.op_id.replace("moe_gate_up_e", "moe_act_e")
            down_id = gate.op_id.replace("moe_gate_up_e", "moe_down_e")
            act = moe_by_id[act_id]
            down = moe_by_id[down_id]
            assert act.family == "swiglu"
            assert act.deps == (gate.op_id,)
            assert down.deps == (act.op_id,)
            assert act.semantic_shape[1] == 64
            assert dict(act.metadata)["expert"] == expert
            assert act.vector_ops > 0 and act.hbm_bytes > 0
        shared = moe_by_id["s0.l0.shared_experts"]
        shared_act = moe_by_id["s0.l0.shared_expert_act"]
        shared_down = moe_by_id["s0.l0.shared_expert_down"]
        assert shared_act.deps == (shared.op_id,)
        assert shared_down.deps == (shared_act.op_id,)
        assert shared_act.semantic_shape == (8, 64)
        assert shared_act.vector_ops > 0


def test_shared_expert_has_gate_up_and_down_work():
    model = ModelSpec.from_mapping(dense_model(
        hidden_size=256, num_heads=4, head_dim=64, intermediate_size=512,
        num_experts=8, top_k=2, moe_intermediate_size=128, num_shared_experts=1,
    ))
    work = derive_iteration_work(
        IterationPlan((RequestView("r", 256, 0, True, "prefill"),)),
        model,
        RuntimeSpec.from_mapping({"tp": 2, "max_num_batched_tokens": 256, "max_model_len": 512}),
        MachineSpec.from_mapping(machine(2)),
        ParallelSpec(tp=2),
    )
    shared = {op.family: op for op in work.ops if op.family.startswith("shared")}
    assert shared["shared_expert_down"].flops == pytest.approx(8_388_608)
    assert shared["shared_expert_gate_up"].flops + shared["shared_expert_down"].flops > 0


EXPANDED_TRAIN = Path("/Volumes/data/mindie-sim-data/a3/2026-10-20/a3-gemm-expanded-train.json")
COMM_TRAIN = Path("/tmp/mindie-a3-dsh.DNIfZY/communication-train.json")


@pytest.mark.skipif(not EXPANDED_TRAIN.exists(), reason="expanded GEMM training capture is not mounted")
def test_expanded_gemm_train_import_excludes_boundary_holdout():
    from mindie_sim.a3_real_data import import_a3_measurements

    store = SQLiteCalibrationStore(":memory:")
    result = import_a3_measurements(store, EXPANDED_TRAIN)
    assert result["measurements"] == 48 * 7
    stats = store.statistics()
    shapes = {int(item["shape"]["m"]) for item in stats if item["op_family"] == "gemm"}
    assert 127 not in shapes and 129 not in shapes
    assert 4096 in shapes
    assert all(item["split"] == "train" for item in stats)
    assert store.inventory()["repeats_per_condition"]["max"] == 7


@pytest.mark.skipif(not COMM_TRAIN.exists(), reason="communication training capture is not mounted")
def test_communication_training_import_and_affine_shared_lane_predictions():
    from mindie_sim.a3_real_data import (
        CommunicationCalibration, import_communication_measurements,
    )

    store = SQLiteCalibrationStore(":memory:")
    result = import_communication_measurements(store, COMM_TRAIN)
    assert result["measurements"] > 200
    assert result["conditions"] == 12
    stats = store.statistics()
    assert all(item["source_classes"] == ["measured"] for item in stats)
    assert any(item["op_family"] == "all_reduce" and item["calibratable_count"] == 9 for item in stats)
    calibration = CommunicationCalibration.from_store(store)
    assert len(calibration.conditions) == 12
    small = calibration.predict(placement="sio", size_bytes=4 * 2**20)
    assert small["status"] == "calibrated_affine_shared_lane"
    assert small["span_ms"] == pytest.approx(0.248, rel=0.2)
    large = calibration.predict(
        placement="sio", size_bytes=128 * 2**20, requested_delay_ms=1.0,
    )
    assert large["span_ms"] > 1.0
    assert large["span_ms"] > large["per_repetition"]["all_reduce_ms"]
    assert large["all_reduce_ms"] > 0 and large["p2p_ms"] > 0
    # Raw offsets and repetition counts are retained in details/protocol.
    raw = [item for item in store.samples() if item.op_family == "all_reduce"][0]
    assert raw.details["sample"]["interval_overlap_ms"] >= 0
    assert raw.protocol["repetitions_per_lane"] == 16


def test_semantic_counts_prefix_and_position_dependency_oracle():
    from mindie_sim.a3_should import semantic_work

    for prompt, output, prefix, linear, attention in (
        (8, 1, 0, 2_621_440, 36_864),
        (8, 2, 0, 2_949_120, 46_080),
        (8, 1, 7, 327_680, 8_192),
    ):
        parsed = CaseSpec.from_mapping(case(
            model=dense_model(
                layers=4, hidden_size=64, num_heads=1, num_kv_heads=1, head_dim=64,
                intermediate_size=128, vocab_size=1000,
            ),
            workload={"requests": [{
                "prompt_tokens": prompt, "output_tokens": output, "prefix_tokens": prefix,
            }]},
        ))
        work = semantic_work(parsed.model, parsed.workload, ParallelSpec())
        assert work.linear_flops == pytest.approx(linear)
        assert work.attention_flops == pytest.approx(attention)

    # Inactive experts affect residency, not active compulsory reads.
    active = []
    resident = []
    for experts in (8, 64):
        parsed = CaseSpec.from_mapping(case(
            model=dense_model(
                num_experts=experts, top_k=1, moe_intermediate_size=128,
                expert_routing_histogram=[1] + [0] * (experts - 1),
            ),
            workload={"requests": [{"prompt_tokens": 1, "output_tokens": 1}]},
        ))
        work = semantic_work(parsed.model, parsed.workload, ParallelSpec())
        active.append(work.weight_bytes)
        resident.append(work.resident_weight_bytes)
    assert active[0] == pytest.approx(active[1])
    assert resident[1] > resident[0]


def test_normative_rejects_generation_that_cannot_keep_kv_resident():
    payload = case(
        workload={"requests": [{"prompt_tokens": 1024, "output_tokens": 50000}]},
        machine={
            **machine(1),
            "device": {
                **machine(1)["device"], "hbm_bytes": 20_000_000,
                "launch_us": 1.0,
            },
        },
        runtime={"max_model_len": 65536, "kv_block_size": 16},
    )
    result = should_cost(payload, constraints={
        "tp": [1], "dp": [1], "pp": [1], "fusion": False, "graph_padding": False,
        "gpu_memory_utilization": 0.9,
    })
    assert result["status"] == "infeasible"
    assert any(
        item.get("reason") == "replica_peak_kv_infeasible"
        for item in result.get("infeasible_plans", [])
    )


def test_shared_link_is_not_minted_per_dp_replica_and_route_order_is_stable():
    payload = {
        "machine": {
            "num_devices": 4,
            "device": {
                "hbm_gib": 8, "cube_flops_per_s": 1e12, "vector_ops_per_s": 1e12,
                "hbm_bw_bytes_per_s": 1e9, "mte_bw_bytes_per_s": 1e9, "launch_us": 0,
            },
            "links": [{
                "name": "shared", "capacity_gbps": .001, "members": [0, 1, 2, 3],
            }],
        },
        "model": dense_model(layers=2),
        "workload": {"requests": [
            {"prompt_tokens": 64, "output_tokens": 1},
            {"prompt_tokens": 64, "output_tokens": 1},
        ]},
        "runtime": {"max_num_batched_tokens": 128, "max_model_len": 1024},
    }
    normative = should_cost(payload, constraints={
        "tp": [1], "dp": [2], "pp": [2], "fusion": False, "graph_padding": False,
    })
    assert normative["status"] == "modeled"
    assert normative["feasible_target_ms"] >= 16.384 - 1e-9
    predictive = predict({**payload, "runtime": {**payload["runtime"], "tp": 1, "dp": 2, "pp": 2}})
    assert predictive["metrics"]["ttft_ms"]["p50"] >= 16.384 - 1e-9

    slow_fast = [
        {"name": "slow", "capacity_gbps": .001, "members": [0, 1]},
        {"name": "fast", "capacity_gbps": 1000, "members": [2, 3]},
    ]
    results = []
    for links in (slow_fast, list(reversed(slow_fast))):
        candidate = {
            **payload,
            "machine": {**payload["machine"], "links": links},
            "workload": {"requests": [{"prompt_tokens": 64, "output_tokens": 1}]},
            "runtime": {"max_num_batched_tokens": 128, "max_model_len": 1024, "pp": 2},
        }
        results.append(should_cost(candidate, constraints={
            "tp": [1], "dp": [1], "pp": [2], "fusion": False, "graph_padding": False,
        })["feasible_target_ms"])
    assert results[0] == pytest.approx(results[1])


def test_normalized_cv_is_executed_by_resource_engine():
    result = normalized_shared_cv_search()
    for plan in result["plans"]:
        assert plan["trace"]
        assert plan["target_ms"] == pytest.approx(
            max(event["end_ms"] for event in plan["trace"])
        )
    selected = result["selected"]
    assert selected["target_ms"] == pytest.approx(14.0)
    assert {event["task_id"] for event in selected["trace"]} == {
        "c_read", "c_compute", "v_read", "v_compute",
    }


def test_catalog_loads_all_nine_real_models_with_layer_schedules():
    from mindie_sim.model_catalog import load_model

    expected = {
        "deepseek-v4-10t": (61, 0, 61),
        "deepseek-v4-flash-0731": (43, 0, 43),
        "deepseek-v4-pro-0813": (61, 0, 61),
        "deepseek-v4.1-flash": (40, 0, 40),
        "glm-5.3": (78, 0, 78),
        "kimi-k3": (93, 69, 24),
        "kimi-k3-dspark": (5, 0, 0),
        "qwen3.8-2.4t-a95b": (92, 69, 0),
        "qwen3.8-27b": (64, 48, 0),
    }
    for key, (layers, linear, mla) in expected.items():
        model = load_model(key)
        facts = dict(model.catalog_facts)
        assert model.layers == layers
        assert facts["body_layers"] == layers
        assert facts["linear_layers"] == linear
        assert facts["mla_layers"] == mla
        assert len(model.layer_plans) == layers
        assert model.resident_weight_bytes_override > 0
        assert any(plan.ffn == "moe" for plan in model.layer_plans) or model.num_experts == 0
    qwen = load_model("Qwen3.8-2.4T-A95B")
    assert qwen.num_experts == 512 and qwen.top_k == 10
    kimi = load_model("Kimi-K3")
    assert kimi.num_experts == 896 and kimi.top_k == 16
    glm = load_model("GLM-5.3")
    assert any(plan.ffn == "dense" for plan in glm.layer_plans)
    assert any(plan.ffn == "moe" for plan in glm.layer_plans)
    assert dict(load_model("Qwen3.8-27B").catalog_facts)["vision_config"]


def test_catalog_model_predict_should_and_oom_guidance():
    qwen_case = {
        "machine": {
            "device": {
                "hbm_gib": 64, "cube_flops_per_s": 1e14,
                "vector_ops_per_s": 1e13, "hbm_bw_bytes_per_s": 1e12,
                "mte_bw_bytes_per_s": 2e12,
            },
            "num_devices": 8,
        },
        "model": {"catalog_key": "Kimi-K3-DSpark"},
        "workload": {"requests": [{"prompt_tokens": 4, "output_tokens": 1}]},
        "runtime": {"max_num_batched_tokens": 8, "max_model_len": 64,
                    "tp": 8, "dp": 1, "pp": 1},
    }
    predicted = predict(qwen_case)
    assert predicted["status"] == "modeled"
    assert predicted["metrics"]["ttft_ms"]["p50"] is not None
    normative = should_cost(qwen_case, constraints={
        "tp": [8], "pp": [1], "dp": [1], "fusion": False, "graph_padding": False,
    })
    # The catalog lane is still replacing the semantic-work/lowering contracts;
    # keep this integration test honest about either a legal U or an explicit
    # low-bound rejection, without weakening the separate homogeneous oracle.
    if normative["status"] == "modeled":
        assert normative["hardware_lower_bound_ms"] <= normative["feasible_target_ms"] + 1e-9
    else:
        assert normative.get("reason") == "no_legal_plan"
    tiny = {
        **qwen_case,
        "machine": {**qwen_case["machine"], "num_devices": 8},
        "model": {"catalog_key": "Kimi-K3"},
        "runtime": {"max_num_batched_tokens": 8, "max_model_len": 64,
                    "tp": 8, "dp": 1, "pp": 1, "ep": 8},
    }
    oom = predict(tiny)
    assert oom["status"] == "infeasible"
    assert oom["oom_guidance"]["requires_more_devices"] is True
    assert oom["oom_guidance"]["required_world_size_estimate"] > 8


def test_repeat_ordinal_retry_history_and_analytic_baseline():
    store = SQLiteCalibrationStore(":memory:")
    request = store.record_measurement_request(
        op_family="qkv_projection", phase="prefill", recipe="r",
        shape={"m": 8, "n": 16, "k": 16}, predicted_ms=0.04, analytic_ms=0.02,
    )
    complete = store.complete_request_samples(
        request["request_key"], raw_samples_ms=[1.0, 2.0, 3.0],
        analytic_ms=0.02, source="measured-run", run_id="run-1",
        details={"repeat_index": 0, "timing_scope": "device_event_warm"},
    )
    assert complete["sample_count"] == 3
    assert store.inventory()["measurements"] == 3
    assert len(store.as_calibration_store().observations()) == 1
    # Failure then explicit retry: history is retained and manifest reappears.
    failed = store.record_measurement_request(
        op_family="gemm", phase="operator", recipe="r", shape={"m": 1},
        predicted_ms=0.04, analytic_ms=0.02,
    )
    store.update_request_status(failed["request_key"], status="running", run_id="run-2")
    store.ingest_results({"results": [{
        "request_key": failed["request_key"], "status": "failed",
        "reason": "device_error", "run_id": "run-2",
    }]})
    failed_record = store.requests(status="failed")[0]
    assert failed_record["attempt_count"] == 2
    store.requeue_request(failed["request_key"], run_id="run-3")
    pending = store.requests(status="pending")
    assert any(item["request_key"] == failed["request_key"] for item in pending)
    assert store.manifest()["requests"]


def test_public_calibration_features_separate_bf16_and_w8a16():
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_workload import OpSpec

    machine_spec = MachineSpec.from_mapping(machine())
    observation = {
        "op_family": "qkv_projection", "phase": "prefill", "recipe": "prefill_causal:qkv",
        "shape": {"m": 8, "n": 16, "k": 64}, "duration_ms": 2.0,
        "analytic_ms": 1.0, "source_class": "measured", "source": "real-label",
        "dtype": "bf16", "layout": "contiguous",
        # Hardware identity is required to consume a residual (no wildcard).
        "machine_key": "normalized-test-die",
        "features": {"weight_dtype": "bf16", "activation_dtype": "bf16",
                     "compute_dtype": "bf16", "accumulation_dtype": "backend_default",
                     "quantization": "none"},
        "timing_scope": "device_event_warm",
    }
    store = CalibrationStore.from_mapping({"observations": [observation]})
    op = OpSpec(
        op_id="qkv", family="qkv_projection", phase="prefill", stage=0, layer=0,
        requests=(), semantic_shape=(8, 16, 64), local_shape=(8, 16, 64),
        padded_shape=(8, 16, 64), launch_shape=(8, 16, 64), flops=2 * 8 * 16 * 64,
        hbm_bytes=0, dtype="bf16", layout="contiguous", recipe="prefill_causal:qkv",
    )
    bf16 = PredictiveCostModel(
        machine_spec, store, compute_dtype="bf16", weight_dtype="bf16",
        activation_dtype="bf16", quantization="none", timing_scope="device_event_warm",
    ).cost(op)
    assert bf16.calibration.status == "calibrated_exact"
    w8a16 = PredictiveCostModel(
        machine_spec, store, compute_dtype="bf16", weight_dtype="int8",
        activation_dtype="bf16", quantization="w8a16", timing_scope="device_event_warm",
    ).cost(op)
    assert w8a16.calibration.status != "calibrated_exact"


def test_timing_scope_is_part_of_sample_grouping():
    store = SQLiteCalibrationStore(":memory:")
    for scope, duration in (("device_event_warm", 1.0), ("host_wall", 100.0)):
        store.record_sample(
            op_family="gemm", phase="operator", recipe="r", shape={"m": 1},
            duration_ms=duration, analytic_ms=1.0, source="real", source_class="measured",
            split="train", timing_scope=scope, run_id="r1", case_id=scope,
        )
    stats = store.statistics()
    assert len(stats) == 2
    assert sorted(item["timing_scope"] for item in stats) == ["device_event_warm", "host_wall"]


def test_baseline_gemm_matches_projection_compute_baseline():
    from mindie_sim.a3_real_data import baseline_gemm
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_workload import OpSpec

    machine_spec = MachineSpec.from_mapping(machine())
    generic = baseline_gemm(machine_spec, 8192, 8192, 8192)
    op = OpSpec(
        op_id="qkv", family="qkv_projection", phase="operator", stage=0, layer=0,
        requests=(), semantic_shape=(8192, 8192, 8192),
        local_shape=(8192, 8192, 8192), padded_shape=(8192, 8192, 8192),
        launch_shape=(8192, 8192, 8192), flops=2 * 8192 ** 3,
        hbm_bytes=(8192 * 8192 + 8192 * 8192 + 8192 * 8192) * 2,
        mte2_bytes=(8192 * 8192 + 8192 * 8192) * 2,
        mte3_bytes=8192 * 8192 * 2, dtype="bf16", layout="contiguous", recipe="baseline",
    )
    projection = PredictiveCostModel(machine_spec).cost(op).analytic_ms
    assert generic > 0
    assert generic == pytest.approx(projection, rel=1e-9)


def test_mla_attention_waits_for_both_q_and_kv_producers():
    model = ModelSpec.from_mapping(dense_model(
        attention="mla", hidden_size=128, num_heads=4, head_dim=32,
        mla={"q_lora_rank": 64, "kv_lora_rank": 16,
             "qk_nope_head_dim": 20, "qk_rope_head_dim": 12, "v_head_dim": 24},
        intermediate_size=256,
    ))
    work = derive_iteration_work(
        IterationPlan((RequestView("r", 8, 0, True, "prefill"),)),
        model,
        RuntimeSpec.from_mapping({"max_num_batched_tokens": 16, "max_model_len": 128}),
        MachineSpec.from_mapping(machine()),
    )
    qk = next(op for op in work.ops if op.family == "attention_fused")
    assert any("mla_qkv_barrier" in dep for dep in qk.deps)
    barrier = next(op for op in work.ops if op.family == "barrier")
    assert any(".q_b" in dep for dep in barrier.deps)
    assert any(".kv_b" in dep for dep in barrier.deps)


def test_resource_engine_incremental_time_never_regresses():
    from mindie_sim.a3_resources import ResourceEngine, ResourceSpec

    engine = ResourceEngine({"x": ResourceSpec("x", 1.0)})
    engine.add_task("f0", {"x": 10.0})
    first = engine.run()
    assert first["completed_at_ms"]["f0"] == 10.0
    engine.add_task("f1", {"x": 5.0}, dependencies=("f0",))
    second = engine.run()
    assert second["completed_at_ms"]["f1"] == 15.0
    trace = {item["task_id"]: item for item in second["trace"]}
    assert trace["f1"]["start_ms"] == 10.0


def test_resource_engine_token_ready_and_background_drain_are_separate():
    from mindie_sim.a3_resources import ResourceEngine, ResourceSpec

    engine = ResourceEngine({"x": ResourceSpec("x", 1.0)})
    engine.add_task("token", {"x": 5.0})
    engine.add_task("background", {"x": 10.0})
    ready = engine.run(until=("token",))
    assert ready["target_completed_at_ms"]["token"] == 10.0
    assert ready["drain_complete"] is False
    assert ready["remaining_tasks"] == ["background"]
    drained = engine.run()
    assert drained["completed_at_ms"]["token"] == 10.0
    assert drained["completed_at_ms"]["background"] == 15.0


def test_resource_engine_deadline_slack_is_reported():
    from mindie_sim.a3_resources import ResourceEngine, ResourceSpec

    engine = ResourceEngine({"x": ResourceSpec("x", 1.0)})
    engine.add_task("prep", {"x": 8.0}, deadline_ms=5.0)
    result = engine.run()
    assert result["deadline_slack_ms"]["prep"] == pytest.approx(-3.0)
    assert result["deadline_misses"] == ["prep"]


def test_async_prepare_background_is_not_charged_to_ttft():
    async_case = case(runtime={
        **case()["runtime"],
        "prepare": [{
            "name": "future-weight-stream", "duration_ms": 10.0,
            "known_at": "startup", "deadline_ms": 5.0,
        }],
    })
    predicted = predict(async_case)
    assert predicted["status"] == "modeled"
    assert predicted["metrics"]["ttft_ms"]["p50"] < 5.0
    prep = predicted["prepare"]["replicas"][0]
    assert prep["token_ready_ms"] < prep["drain_makespan_ms"]
    assert prep["drain_tail_ms"] > 0
    assert prep["deadline_misses"]


def test_prepare_unknown_future_is_rejected():
    with pytest.raises(A3InputError):
        CaseSpec.from_mapping(case(runtime={
            **case()["runtime"],
            "prepare": [{"name": "spec", "known_at": "future_accept"}],
        }))


def test_should_placement_is_automatic_unless_caller_pins_core_base():
    machine_payload = machine()
    machine_payload["device"] = {
        **machine_payload["device"],
        "aic_count": 20, "aiv_count": 40,
        "l1_bytes": 524288, "ub_bytes": 262144,
        "l0a_bytes": 65536, "l0b_bytes": 65536, "l0c_bytes": 131072,
    }
    payload = case(
        machine=machine_payload,
        # This test exercises the fusion *placement* planner, which composes a
        # kernel out of separate ops: declare the decomposed attention lowering so
        # there is a group to place (the served default is one fused kernel).
        runtime={
            "fusion": True, "fusion_worker_count": 2,
            "metadata": {"attention_decomposition": "qk_softmax_pv"},
        },
    )
    base_constraints = {
        "tp": [1], "dp": [1], "pp": [1], "fusion": [True],
        "graph_padding": False, "worker_count": [2],
    }
    auto = should_cost(payload, constraints=base_constraints)
    fixed = should_cost(
        payload, constraints={**base_constraints, "core_base": 0},
    )
    assert auto["status"] == fixed["status"] == "modeled"
    assert auto["selected_plan"]["placement"]["mode"] == "auto_disjoint"
    assert auto["selected_plan"]["placement"]["core_base"] is None
    assert fixed["selected_plan"]["placement"]["mode"] == "fixed_core_base"
    assert fixed["selected_plan"]["placement"]["core_base"] == 0
    assert auto["selected_witness"]["placement"] == (
        auto["selected_plan"]["placement"]
    )


def test_should_fused_candidate_is_real_and_selected_by_completion():
    # The fusion rules need the decomposed graph to have groups to fuse: the
    # served default already emits the fused attention kernel, so this test
    # declares the decomposed lowering explicitly.
    payload = case(runtime={
        **case()["runtime"],
        "metadata": {"attention_decomposition": "qk_softmax_pv"},
    })
    unfused = should_cost(payload, constraints={
        "tp": [1], "dp": [1], "pp": [1], "fusion": [False],
        "graph_padding": False,
    })
    fused = should_cost(payload, constraints={
        "tp": [1], "dp": [1], "pp": [1], "fusion": [True],
        "graph_padding": False,
    })
    assert unfused["selected_plan"]["fusion"] is False
    assert fused["selected_plan"]["fusion"] is True
    assert fused["feasible_target_ms"] <= unfused["feasible_target_ms"] + 1e-9
    assert "attention_qk_softmax_pv" in fused["selected_witness"]["applied_rules"]
    assert "rmsnorm_gemm_epilogue" in fused["selected_witness"]["applied_rules"]
    assert fused["selected_witness"]["removed_intermediates"]
    assert fused["selected_witness"]["resource_dependency_witness"]["edges"]
    fused_ops = [
        op for step in fused["selected_witness"]["replicas"][0]["steps"]
        for op in step["events"]
        if "fused:" in op["task_id"]
    ]
    assert fused_ops
    assert fused["selected_witness"]["fusion_changes"]["launch"] < 0


def test_mla_absorption_is_rejected_when_projected_dims_differ():
    payload = case()
    payload["model"].update({
        "attention": "mla", "architecture": "mla",
        "num_heads": 4, "head_dim": 32,
        "mla": {"q_lora_rank": 256, "kv_lora_rank": 16,
                "qk_nope_head_dim": 20, "qk_rope_head_dim": 12, "v_head_dim": 24},
    })
    result = should_cost(payload, constraints={
        "tp": [1], "dp": [1], "pp": [1], "fusion": [True],
        "graph_padding": False,
    })
    considered = [
        item for step in result["selected_witness"]["replicas"][0]["steps"]
        for item in ((step.get("fusion") or {}).get("considered_rules") or [])
        if item["rule"] == "mla_projection_absorption"
    ]
    assert considered and considered[0]["eligible"] is False
    assert "mla_projection_absorption" not in result["selected_witness"]["applied_rules"]


def test_search_keeps_fusion_and_reports_pruning():
    from mindie_sim.a3_search import search

    result = search(case(), {
        "max_candidates": 16, "tp": [1], "dp": [1], "pp": [1],
        "fusion": [False, True], "graph_buckets": [[]],
    })
    assert result["status"] == "modeled"
    assert result["recommended"]["config"]["fusion"] is True
    assert result["exhaustive"] is True
    assert result["evaluated_candidates"] == 2
    assert "pruned_by_memory" in result
    assert "unexplored_candidates" in result


def test_predictive_link_reclaims_capacity_for_staggered_replicas():
    payload = {
        "machine": {
            "num_devices": 4,
            "device": {
                "hbm_gib": 8, "cube_flops_per_s": 1e12,
                "vector_ops_per_s": 1e12, "hbm_bw_bytes_per_s": 1e9,
                "mte_bw_bytes_per_s": 1e9, "launch_us": 0,
            },
            "links": [{"name": "shared", "capacity_gbps": .001,
                       "members": [0, 1, 2, 3]}],
        },
        "model": dense_model(layers=2),
        "workload": {"requests": [
            {"prompt_tokens": 64, "output_tokens": 1, "arrival_ms": 0},
            {"prompt_tokens": 64, "output_tokens": 1, "arrival_ms": 20},
        ]},
        "runtime": {"max_num_batched_tokens": 128, "max_model_len": 1024,
                    "tp": 1, "dp": 2, "pp": 2},
    }
    result = predict(payload)
    assert result["link_arbitration"]["final_scales"] == {"0": 1.0, "1": 1.0}
    request_finish = sorted(item["finish_ms"] for item in result["requests"])
    assert request_finish[0] < 16.384
    assert request_finish[1] < 20 + 16.384


def test_public_miss_manifest_worker_ingest_repredict_roundtrip(tmp_path):
    from mindie_sim.a3_calibration_store import SQLiteCalibrationStore
    from mindie_sim.a3_measurement import record_prediction_misses, run_manifest_file

    worker_protocol = {
        "measurement_scope": "event-timed warm repeated calls",
        "timer": "device_event", "warmup": 5,
    }
    payload = case(machine={
        **machine(),
        "hardware_key": "physical-sku-key",
    }, runtime={
        "metadata": {
            "calibration_protocol": json.dumps(worker_protocol),
        },
    })
    base = predict(payload)
    store = SQLiteCalibrationStore(":memory:")
    recorded = record_prediction_misses(
        store, base, hardware_key="physical-sku-key",
    )
    assert recorded["recorded_requests"] > 0
    manifest = store.manifest(limit=256)
    assert manifest["requests"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakeWorker:
        def run(self, request):
            return self.measure(request)

        def measure(self, request):
            return {
                "status": "completed",
                "raw_samples_ms": [float(request["analytic_ms"]) * 2.0],
                "analytic_ms": float(request["analytic_ms"]),
                "source": "leased-a3-worker",
                "source_class": "measured",
                "split": "train",
                "timing_scope": "device_event_warm",
                "protocol": {
                    "measurement_scope": "event-timed warm repeated calls",
                    "timer": "device_event", "warmup": 5,
                },
                "environment": {"device_name": "A3", "cann_version": "test"},
                "details": {"worker": "fake"},
            }

    results_path = tmp_path / "results.json"
    run = run_manifest_file(
        manifest_path, results_path, FakeWorker(), run_id="leased-1",
    )
    assert run["failed"] == 0
    ingest = store.ingest_results(json.loads(results_path.read_text(encoding="utf-8")))
    assert ingest["completed"]
    refreshed = predict(payload, calibration=store.as_calibration_store())
    statuses = {
        op["calibration_status"] for op in refreshed["traces"][0]["ops"]
    }
    assert any(item.startswith("calibrated") for item in statuses)


def test_uncalibrated_communication_and_cv_return_analytic_numeric_or_explicit_machine_gap():
    from mindie_sim.a3_real_data import CommunicationCalibration, CvCalibration

    machine_spec = MachineSpec.from_mapping(machine())
    comm = CommunicationCalibration([])
    no_machine = comm.predict(placement="sio", size_bytes=1024, mode="all_reduce_only")
    assert no_machine["status"] != "calibrated_affine"
    assert no_machine["span_ms"] is None
    assert no_machine["measurement_request"]
    numeric = comm.predict(
        placement="sio", size_bytes=1024, mode="p2p_only",
        machine=machine_spec, repetitions=4,
    )
    assert numeric["status"] == "uncalibrated_analytic_shared_lane"
    assert numeric["span_ms"] > 0

    cv = CvCalibration([])
    cv_no_machine = cv.predict(c_shape_mkn=[8, 8, 8], v_elements=1024)
    assert cv_no_machine["status"] != "calibrated_cv_shared_lane"
    assert cv_no_machine["span_ms"] is None
    cv_numeric = cv.predict(
        c_shape_mkn=[8, 8, 8], v_elements=1024, machine=machine_spec,
    )
    assert cv_numeric["status"] == "uncalibrated_analytic_cv"
    assert cv_numeric["span_ms"] > 0
    assert cv_numeric["measurement_request"]["shape"]["v_elements"] == 1024


def test_worker_free_prose_timing_scope_is_canonicalized_for_calibration():
    from mindie_sim.a3_calibration_store import canonical_timing_scope

    assert canonical_timing_scope(
        "event-timed warm repeated operator calls; includes launch/dispatch gaps"
    ) == "device_event_warm"
    assert canonical_timing_scope("host wall cold") == "host_wall"


def test_fused_group_preserves_cube_work_and_bounded_scratch():
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_fusion import apply_fusion
    from mindie_sim.a3_ops import IterationWork, OpSpec
    from mindie_sim.a3_workload import IterationPlan

    norm = OpSpec(
        "norm", "rmsnorm", "decode", 0, 0, ("r0",), (1, 2), (1, 2),
        (1, 2), (1, 2), vector_ops=2, hbm_bytes=8, dtype="bf16",
    )
    gemm = OpSpec(
        "gemm", "qkv_projection", "decode", 0, 0, ("r0",), (1, 2, 2),
        (1, 2, 2), (1, 2, 2), (1, 2, 2), flops=6, hbm_bytes=16,
        deps=("norm",), dtype="bf16",
    )
    work = IterationWork((norm, gemm), IterationPlan(()), 1, 1, 1, None,
                         "decode", "fixture")
    rejected = apply_fusion(work, scratch_capacity_bytes=3)
    assert "rmsnorm_gemm_epilogue" not in rejected.applied_rules
    fused = apply_fusion(
        work, scratch_capacity_bytes=1024,
        cube_flops_per_s=1000.0, vector_ops_per_s=1000.0,
    )
    assert "rmsnorm_gemm_epilogue" in fused.applied_rules
    fused_op = next(op for op in fused.work.ops if op.coverage == "fused")
    assert fused_op.flops == 6
    meta = dict(fused_op.metadata)
    assert meta["serial_ms"] == 0.0
    assert [stage["name"] for stage in meta["fusion_stages"]] == ["norm", "gemm"]
    assert meta["tile_scratch_bytes"] <= 1024
    # A second consumer that also depends on another input must prevent deletion.
    other = OpSpec(
        "other", "consumer", "decode", 0, 0, ("r0",), (1, 2), (1, 2),
        (1, 2), (1, 2), deps=("norm", "extra"), hbm_bytes=4,
    )
    extra = OpSpec(
        "extra", "metadata", "decode", 0, 0, ("r0",), (1,), (1,),
        (1,), (1,), dtype="bf16",
    )
    escaped = apply_fusion(
        work.__class__((norm, gemm, extra, other), IterationPlan(()), 1, 1, 1,
                       None, "decode", "fixture"),
        scratch_capacity_bytes=1024,
        cube_flops_per_s=1.0, vector_ops_per_s=1.0,
    )
    assert "rmsnorm_gemm_epilogue" not in escaped.applied_rules


def test_finite_fused_stages_preserve_dependencies_and_external_contention():
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_fusion import apply_fusion
    from mindie_sim.a3_ops import IterationWork, OpSpec
    from mindie_sim.a3_resources import (
        ResourceEngine, ResourceSpec, add_costed_ops,
    )
    from mindie_sim.a3_workload import IterationPlan

    qk = OpSpec("qk", "attention_qk", "decode", 0, 0, ("r0",),
                (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                flops=8.0, dtype="bf16")
    soft = OpSpec("soft", "attention_softmax", "decode", 0, 0, ("r0",),
                  (1,), (1,), (1,), (1,), vector_ops=6.0,
                  deps=("qk",), dtype="fp32")
    pv = OpSpec("pv", "attention_pv", "decode", 0, 0, ("r0",),
                (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                deps=("soft",), dtype="bf16")
    work = IterationWork((qk, soft, pv), IterationPlan(()), 1, 1, 1,
                         None, "decode", "fixture")
    fused = apply_fusion(
        work, cube_flops_per_s=1.0, vector_ops_per_s=1.0,
        scratch_capacity_bytes=1024,
    )
    assert "attention_qk_softmax_pv" in fused.applied_rules
    resources = {
        "cube": ResourceSpec("cube", 1.0),
        "vector": ResourceSpec("vector", 1.0),
        "hbm": ResourceSpec("hbm", 1.0),
        "mte2": ResourceSpec("mte2", 1.0), "mte3": ResourceSpec("mte3", 1.0),
        "launch": ResourceSpec("launch", 1.0),
        "serial": ResourceSpec("serial", 1.0),
    }
    stage_map = {0: {name: name for name in resources}}
    engine = ResourceEngine(resources)
    add_costed_ops(
        engine, PredictiveCostModel(MachineSpec.from_mapping(machine())).cost_iteration(fused.work),
        stage_map, prefix="fused", start_ms=0.0,
    )
    engine.add_task("independent", {"vector": 6.0}, release_ms=8.0)
    result = engine.run()
    assert result["completed_at_ms"]["independent"] >= 14.0
    sample = max(
        value for key, value in result["completed_at_ms"].items()
        if key.startswith("fused:")
    )
    assert sample >= result["completed_at_ms"]["independent"]
    assert not any(
        key == "serial"
        for event in result["trace"] for key in event["demands"]
    )
    qk_end = max(
        event["end_ms"] for event in result["trace"]
        if "qk.q0" in event["task_id"]
    )
    soft_start = min(
        event["start_ms"] for event in result["trace"]
        if "softmax.q0" in event["task_id"]
    )
    assert soft_start >= qk_end


def test_communication_backend_is_first_class_for_matching_and_resources():
    from mindie_sim.a3_communication import CommunicationCalibration, CommCondition
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_types import CaseSpec

    # The measurement is bound to the machine it ran on: an unbound observation
    # is refused for every keyed query (the old empty-key wildcard is how A3
    # curves reached other parts).
    conditions = [
        CommCondition("sio", "all_reduce_only", 1024, 0.5, 1,
                      hardware_key="normalized-test-die",
                      ranks=2, dtype="bf16", algorithm="all_reduce_only",
                      execution_backend="aiv"),
        CommCondition("sio", "all_reduce_only", 1024, 0.5, 1,
                      hardware_key="normalized-test-die",
                      ranks=2, dtype="bf16", algorithm="all_reduce_only",
                      execution_backend="aicpu"),
    ]
    calibration = CommunicationCalibration(conditions)
    aiv = calibration.predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        ranks=2, dtype="bf16", algorithm="all_reduce_only",
        execution_backend="aiv", machine=MachineSpec.from_mapping(machine(2)),
    )
    assert aiv["status"] == "calibrated_affine"
    assert aiv["execution_backend"] == "aiv"
    aicpu = calibration.predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        ranks=2, dtype="bf16", algorithm="all_reduce_only",
        execution_backend="aicpu", machine=MachineSpec.from_mapping(machine(2)),
    )
    assert aicpu["execution_backend"] == "aicpu"
    other = calibration.predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        ranks=2, dtype="bf16", algorithm="all_reduce_only",
        execution_backend="unspecified", machine=MachineSpec.from_mapping(machine(2)),
    )
    assert other["status"].startswith("uncalibrated")

    payload = case(machine=machine(2), runtime={
        "max_num_batched_tokens": 16, "max_model_len": 128, "tp": 2,
        "communication_backend": "aiv",
    })
    # The collective placement is a physical fabric.  Renaming an HCCS link to
    # "sio" used to be enough for the cost model to *guess* SIO placement from
    # the name; placement now comes from the declared channel, so the fixture
    # declares a real intra-package SIO pair.
    payload["machine"]["links"] = [{
        "name": "d2d_intra_package", "channel": "d2d_intra_package",
        "capacity_gbps": 100, "latency_us": 1.0, "members": [0, 1],
    }]
    case_spec = CaseSpec.from_mapping(payload)
    baseline = predict(payload)
    with_backend = CaseSpec(
        machine=case_spec.machine, model=case_spec.model,
        workload=case_spec.workload, runtime=case_spec.runtime,
        communication_calibration=calibration,
    )
    predicted = predict(with_backend)
    assert predicted["status"] == "modeled"
    statuses = {
        op["calibration_status"] for op in predicted["traces"][0]["ops"]
        if op["family"] == "allreduce"
    }
    assert any(item.startswith("calibrated") for item in statuses)
    assert predicted["metrics"]["ttft_ms"]["p50"] != baseline["metrics"]["ttft_ms"]["p50"]


PAIRED_STEADY_DB = Path(
    "/Volumes/data/mindie-sim-data/a3/2026-10-20/"
    "comm-paired-steady-20261021/calibration.sqlite"
)


@pytest.mark.skipif(
    not PAIRED_STEADY_DB.exists(),
    reason="paired steady TRAIN communication store is not mounted",
)
def test_actual_paired_steady_typed_whole_span_bridge():
    from mindie_sim.a3_communication import CommunicationCalibration
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_ops import OpSpec
    from mindie_sim.a3_resources import (
        ResourceEngine, ResourceSpec, add_costed_ops,
    )

    store = SQLiteCalibrationStore(PAIRED_STEADY_DB)
    assert store.inventory()["measurements"] == 360
    conditions = CommunicationCalibration.from_store(store).conditions
    assert len(conditions) == 6
    assert all(not condition.execution_stages for condition in conditions)
    assert {condition.observed_execution for condition in conditions} == {
        "aiv", "aicpu",
    }
    machine_spec = MachineSpec.from_mapping({
        "name": "Ascend910_9362", "hardware_key": "Ascend910_9362",
        "num_devices": 2, "arch_family": "A3",
        "device": {
            "hbm_gib": 64, "cube_flops_per_s": 1e14,
            "vector_ops_per_s": 1e13, "hbm_bw_bytes_per_s": 1e12,
            "mte_bw_bytes_per_s": 2e12,
        },
        # A measured "sio" collective ran on the dual-die pair wire; a fixture
        # that only declares an HCCS link now fails link resolution instead of
        # silently pricing SIO traffic on it.
        "links": [
            {"name": "d2d_intra_node", "channel": "d2d_intra_node",
             "capacity_gbps": 196, "members": [0, 1]},
            {"name": "d2d_intra_package", "channel": "d2d_intra_package",
             "capacity_gbps": 190, "latency_us": 0.8, "members": [0, 1]},
        ],
    })
    model = PredictiveCostModel(
        machine_spec,
        communication_calibration=CommunicationCalibration(conditions),
        communication_ranks=2, communication_backend="aiv",
        communication_observed_backend="aiv",
        communication_route="sio:phy2<->phy3",
        communication_timing_completion="device_event_after_collective",
        communication_in_place=True,
        communication_algorithm_regime="RING-RING-NHR",
        communication_topology_class="intra_package",
        communication_placement="sio",
    )
    op = OpSpec(
        "comm", "allreduce", "prefill", 0, 0, ("r",),
        (2048,), (2048,), (2048,), (2048,),
        link_bytes=4096.0, dtype="bf16", layout="contiguous", recipe="r",
    )
    cost = model.cost(op)
    assert cost.calibration.status == "calibrated_affine"
    assert cost.calibration.source == "observed_whole_span_envelope_only"
    assert cost.calibration.extrapolated is False
    assert cost.demand_map()["completion_envelope_ms"] == pytest.approx(
        0.18045, rel=1e-3,
    )
    assert "serial" not in cost.demand_map()
    assert "link" in cost.demand_map()
    assert cost.calibration.evidence["source_status"] == "calibrated_affine"
    assert "comm_stages" not in dict(cost.op.metadata)

    # Two independent opaque spans share no serial resource: each owns its
    # completion envelope, so the makespan is one measured span, not the sum.
    second = OpSpec(
        "comm.2", "allreduce", "prefill", 0, 0, ("r",),
        (2048,), (2048,), (2048,), (2048,),
        link_bytes=4096.0, dtype="bf16", layout="contiguous", recipe="r",
    )
    engine = ResourceEngine({
        "link@s0": ResourceSpec("link@s0", 1e9, "bytes"),
        "launch@s0": ResourceSpec("launch@s0", 1.0),
    })
    add_costed_ops(
        engine, [cost, model.cost(second)],
        {0: {"link": "link@s0", "launch": "launch@s0"}},
        prefix="comm", start_ms=0.0,
    )
    scheduled = engine.run()
    assert scheduled["makespan_ms"] == pytest.approx(0.18045, rel=1e-3)
    assert all(
        value == pytest.approx(0.18045, rel=1e-3)
        for value in scheduled["completed_at_ms"].values()
    )

    # Requested AIV / observed AICPU has only the 4MiB TRAIN point.  A 4KiB
    # query keeps a numeric estimate but exposes singleton extrapolation and the
    # executable missing condition; it is not in-envelope calibrated accuracy.
    extrapolated_model = PredictiveCostModel(
        machine_spec,
        communication_calibration=CommunicationCalibration(conditions),
        communication_ranks=2, communication_backend="aiv",
        communication_observed_backend="aicpu",
        communication_route="sio:phy2<->phy3",
        communication_timing_completion="device_event_after_collective",
        communication_in_place=True,
        communication_algorithm_regime="RING-RING-NHR",
        communication_topology_class="intra_package",
        communication_placement="sio",
    )
    extrapolated = extrapolated_model.cost(op)
    assert extrapolated.calibration.status == "calibrated_affine_extrapolated"
    assert extrapolated.calibration.extrapolated is True
    assert extrapolated.calibration.inside_envelope is False
    evidence = extrapolated.calibration.evidence or {}
    assert evidence["source_status"] == "calibrated_affine_extrapolated"
    assert evidence["size_support"]["measured_size_bytes"] == [4194304]
    assert evidence["size_support"]["singleton_measurement"] is True
    assert evidence["engine_selection"] == "unknown_outside_measured_points"
    assert evidence["in_envelope"] is False
    assert evidence["missing_condition"]["shape"] == {
        "bytes": 4096, "ranks": 2,
    }
    assert extrapolated.demand_map()["completion_envelope_ms"] == pytest.approx(
        0.30081, rel=1e-3,
    )


def test_typed_whole_span_without_stages_is_envelope_only_and_errors_visible():
    from mindie_sim.a3_communication import CommCondition, CommunicationCalibration
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_ops import OpSpec

    machine_spec = MachineSpec.from_mapping({
        **{
            **machine(2),
            # The measured condition is an intra-package SIO collective, so the
            # machine must declare the pair wire it actually ran on.
            "links": [
                {"name": "d2d_intra_node", "channel": "d2d_intra_node",
                 "capacity_gbps": 100, "latency_us": 1.0, "members": [0, 1]},
                {"name": "d2d_intra_package", "channel": "d2d_intra_package",
                 "capacity_gbps": 190, "latency_us": 0.8, "members": [0, 1]},
            ],
        },
        "hardware_key": "Ascend910_9362",
    })
    condition = CommCondition(
        placement="sio", mode="all_reduce_only", size_bytes=2048,
        median_ms=1.5, samples=3, hardware_key="Ascend910_9362",
        timing_scope="communication_lane", ranks=2, dtype="bf16",
        observed_execution="aiv", execution_backend="aiv",
        reduction="sum", group_size=2,
    )
    calibration = CommunicationCalibration([condition])
    predicted = calibration.predict(
        placement="sio", size_bytes=2048, mode="all_reduce_only",
        machine=machine_spec, ranks=2, dtype="bf16",
        execution_backend="aiv", observed_execution="aiv",
        reduction="sum", group_size=2,
    )
    assert predicted["status"] == "calibrated_affine"
    assert predicted["calibrated_scope"] == "typed_whole_span"
    assert predicted["occupancy"] == "unknown"
    assert predicted["stages"] == []

    span_condition = CommCondition(
        placement="sio", mode="all_reduce_plus_p2p", size_bytes=2048,
        median_ms=2.5, samples=3, hardware_key="Ascend910_9362",
        timing_scope="communication_lane", ranks=2, dtype="bf16",
        observed_execution="aiv", execution_backend="aiv",
        reduction="sum", group_size=2,
    )
    span_prediction = CommunicationCalibration([span_condition]).predict(
        placement="sio", size_bytes=2048, mode="all_reduce_plus_p2p",
        machine=machine_spec, ranks=2, dtype="bf16",
        execution_backend="aiv", observed_execution="aiv",
        reduction="sum", group_size=2,
    )
    assert span_prediction["status"] == "calibrated_typed_whole_span"
    assert span_prediction["stages"] == []
    assert span_prediction["occupancy"] == "unknown"
    assert span_prediction["calibrated_scope"] == "typed_whole_span"
    assert span_prediction["span_ms"] == pytest.approx(2.5)
    assert span_prediction["phases"]["all_reduce"] == "analytic_unknown_occupancy"

    op = OpSpec(
        "comm.0", "allreduce", "prefill", 0, 0, ("r",),
        (1024,), (1024,), (1024,), (1024,),
        link_bytes=2048.0, dtype="bf16", layout="contiguous", recipe="r",
    )
    model = PredictiveCostModel(
        machine_spec, communication_calibration=calibration,
        communication_ranks=2, communication_backend="aiv",
        communication_observed_backend="aiv",
        communication_placement="sio",
    )
    cost = model.cost(op)
    assert cost.calibration.status == "calibrated_affine"
    assert cost.calibration.source == "observed_whole_span_envelope_only"
    assert cost.calibration.extrapolated is False
    assert cost.demand_map().get("completion_envelope_ms") == pytest.approx(1.5)
    assert "serial" not in cost.demand_map()
    assert "link" in cost.demand_map()
    assert cost.calibration.evidence["source_status"] == "calibrated_affine"
    assert "comm_stages" not in dict(cost.op.metadata)

    class BrokenCommunication:
        def predict(self, **kwargs):
            raise A3InputError("comm_backend_broken")

    broken = PredictiveCostModel(
        machine_spec, communication_calibration=BrokenCommunication(),
        communication_ranks=2, communication_backend="aiv",
        communication_observed_backend="aiv",
        communication_placement="sio",
    )
    with pytest.raises(A3InputError, match="comm_backend_broken"):
        broken.cost(op)


def test_communication_backend_survives_store_and_controls_matching():
    from mindie_sim.a3_communication import CommunicationCalibration

    store = SQLiteCalibrationStore(":memory:")
    store.record_sample(
        op_family="all_reduce", phase="communication", recipe="toy",
        shape={"bytes": 1024, "ranks": 2}, duration_ms=0.5, analytic_ms=1.0,
        source="synthetic-contract", source_class="measured", split="train",
        hardware_key="A3", dtype="bf16", layout="lane",
        timing_scope="communication_lane",
        features={"placement": "sio", "mode": "all_reduce_only", "ranks": 2,
                  "dtype": "bf16", "algorithm": "all_reduce_only",
                  "execution_backend": "aiv"},
    )
    calibration = CommunicationCalibration.from_store(store)
    assert calibration.conditions[0].execution_backend == "aiv"
    same = calibration.predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        ranks=2, dtype="bf16", algorithm="all_reduce_only",
        execution_backend="aiv", machine=MachineSpec.from_mapping(
            {**machine(2), "hardware_key": "A3"},
        ),
    )
    cross = calibration.predict(
        placement="sio", size_bytes=1024, mode="all_reduce_only",
        ranks=2, dtype="bf16", algorithm="all_reduce_only",
        execution_backend="aicpu", machine=MachineSpec.from_mapping(
            {**machine(2), "hardware_key": "A3"},
        ),
    )
    assert same["status"] == "calibrated_affine"
    assert cross["status"].startswith("uncalibrated")


def test_typed_aiv_communication_expands_to_real_stages_not_scalar_label():
    from mindie_sim.a3_communication import CommunicationCalibration, CommCondition
    from mindie_sim.a3_types import CaseSpec

    stages = (
        {"kind": "control", "engine": "aicpu",
         "work": {"aicpu_service": 1.0}, "deps": ()},
        {"kind": "transfer", "engine": "dma",
         "work": {"link_edge_bytes": 8.0, "local_hbm_read": 8.0},
         "deps": (0,)},
        {"kind": "reduce", "engine": "aiv",
         "work": {"vector_ops": 6.0}, "deps": (1,),
         "reservations": {"aiv_slots": 1.0},
         "reservation_capacities": {"aiv_slots": 2.0}},
    )
    condition = CommCondition(
        "sio", "all_reduce_only", 1024, 0.5, 1,
        hardware_key="normalized-test-die", ranks=2, dtype="bf16",
        algorithm="all_reduce_only", execution_backend="aiv",
        observed_execution="aiv", collective="all_reduce", reduction="sum",
        group_size=2, execution_stages=stages,
    )
    calibration = CommunicationCalibration([condition])
    payload = case(machine=machine(2), model={
        "layers": 1, "hidden_size": 128, "num_heads": 4, "num_kv_heads": 2,
        "head_dim": 32, "intermediate_size": 256, "vocab_size": 256,
        "activation_dtype": "bf16", "weight_dtype": "bf16",
        "compute_dtype": "bf16",
    }, runtime={
        "max_num_batched_tokens": 16, "max_model_len": 128, "tp": 2,
        "communication_backend": "aiv",
        "communication_observed_backend": "aiv",
    })
    payload["machine"]["links"] = [{
        # Declared SIO pair: placement comes from the channel, not the name.
        "name": "d2d_intra_package", "channel": "d2d_intra_package",
        "capacity_gbps": 100, "latency_us": 1.0, "members": [0, 1],
    }]
    spec = CaseSpec.from_mapping(payload)
    typed = CaseSpec(
        machine=spec.machine, model=spec.model, workload=spec.workload,
        runtime=spec.runtime, communication_calibration=calibration,
    )
    result = predict(typed)
    assert result["status"] == "modeled"
    comm_tasks = [
        event for step in result["traces"] for event in step["events"]
        if ":comm" in event["task_id"]
    ]
    assert comm_tasks
    assert any(any("vector" in key for key in event["demands"]) for event in comm_tasks)
    assert any(any("cpu" in key for key in event["demands"]) for event in comm_tasks)
    assert any(event.get("residency") for event in comm_tasks)
    cross = CaseSpec(
        machine=spec.machine, model=spec.model, workload=spec.workload,
        runtime=__import__("dataclasses").replace(
            spec.runtime, communication_observed_backend="aicpu",
        ),
        communication_calibration=calibration,
    )
    cross_result = predict(cross)
    assert all(
        not op["calibration_status"].startswith("calibrated")
        for step in cross_result["traces"] for op in step["ops"]
        if op["family"] == "allreduce"
    )


def _media_case(*, encoded=False, chunked=False, two=False):
    requests = [
        {"request_id": "m0", "prompt_tokens": 1, "media_tokens": 4,
         "media_encoded": encoded, "output_tokens": 2},
    ]
    if two:
        requests.append({
            "request_id": "m1", "prompt_tokens": 2, "media_tokens": 4,
            "output_tokens": 2,
        })
    return {
        "machine": {"num_devices": 8, "device": {
            "hbm_gib": 64, "cube_flops_per_s": 1e14,
            "vector_ops_per_s": 1e13, "hbm_bw_bytes_per_s": 1e12,
            "mte_bw_bytes_per_s": 2e12, "l2_bytes": 1 << 20,
        }},
        "model": {"catalog_key": "Qwen3.8-27B"},
        "workload": {"requests": requests},
        "runtime": {
            "max_num_batched_tokens": 2 if chunked else 16,
            "max_num_seqs": 2 if two else 1,
            "max_model_len": 64, "tp": 8, "dp": 1, "pp": 1,
            "enable_chunked_prefill": chunked,
        },
    }


def test_public_media_state_tower_once_and_already_encoded():
    from mindie_sim.a3_types import CaseSpec

    case = CaseSpec.from_mapping(_media_case())
    assert case.workload.requests[0].prompt_tokens == 5
    assert case.workload.requests[0].media_encoded is False
    metadata = dict(case.runtime.metadata)
    assert metadata["media_positions_pre_expanded"] == "true"
    assert metadata["media_tokens_by_request"] == '{"m0": 4}'
    result = predict(case)
    assert result["status"] == "modeled"
    first, decode = result["traces"][0], result["traces"][1]
    assert first["valid_tokens"] == 5
    assert any("vision.patch_embed" in op["op_id"] for op in first["ops"])
    assert not any("vision" in op["op_id"] for op in decode["ops"])
    assert result["requests"][0]["media_encoded"] is True

    chunked = predict(_media_case(chunked=True))
    assert chunked["status"] == "modeled"
    tower_steps = [
        trace["iteration"] for trace in chunked["traces"]
        if any("vision.patch_embed" in op["op_id"] for op in trace["ops"])
    ]
    assert tower_steps == [0]
    assert chunked["requests"][0]["computed_tokens"] >= 5

    encoded = predict(_media_case(encoded=True))
    assert encoded["requests"][0]["prompt_tokens"] == 5
    assert not any(
        "vision" in op["op_id"]
        for trace in encoded["traces"] for op in trace["ops"]
    )

    two = predict(_media_case(two=True))
    tower_ids = [
        op["op_id"] for trace in two["traces"] for op in trace["ops"]
        if "vision.patch_embed" in op["op_id"]
    ]
    assert len(tower_ids) == 2
    assert len({op_id for op_id in tower_ids}) == 2


def _draft_case(trace, *, output=12):
    import json as _json
    case = json.load(open(
        "examples/model-reference/cases/deepseek-v4.1-flash.json",
        encoding="utf-8",
    ))
    case["machine"]["num_devices"] = 64
    case["runtime"].update(
        tp=64, ep=64, dp=1, pp=1, max_num_seqs=1,
        metadata={
            "draft_mode": "dspark",
            "draft_accept_trace": _json.dumps({"r0": trace}),
        },
    )
    case["workload"]["requests"][0]["output_tokens"] = output
    return case


def test_public_draft_acceptance_changes_state_and_tail():
    fallback = predict(_draft_case([0, 2, 1]))
    q = fallback["requests"][0]
    assert fallback["status"] == "modeled"
    assert (q["generated_tokens"], q["computed_tokens"]) == (12, 19)
    assert q["draft_fallback_tokens"] >= 1
    assert q["draft_rollback_tokens"] > 0
    assert q["draft_temporary_tokens"] > 0
    assert q["draft_steps"][0]["accepted_tokens"] == 0
    assert q["draft_steps"][0]["advanced_tokens"] == 1
    assert q["draft_steps"][1]["accepted_tokens"] == 2
    assert q["draft_steps"][1]["advanced_tokens"] == 3
    assert fallback["traces"][-1]["requests"][0]["draft_verify"] is False

    all_accepted = predict(_draft_case([5, 5, 5]))
    accepted = all_accepted["requests"][0]
    assert accepted["generated_tokens"] == 12
    assert accepted["draft_accepted_total"] > 0
    assert accepted["draft_fallback_tokens"] == 0
    assert len(accepted["draft_steps"]) == 2

    all_fallback = predict(_draft_case([0], output=7))
    fb = all_fallback["requests"][0]
    assert fb["generated_tokens"] == 7
    assert fb["draft_fallback_tokens"] == len(fb["draft_steps"])


def test_public_draft_normative_matches_predictive_and_no_draft_is_unchanged():
    from mindie_sim.a3_should import should_cost

    base = _draft_case([0, 2, 1])
    pred = predict(base)
    norm = should_cost(base, constraints={
        "tp": [64], "dp": [1], "pp": [1], "ep": [64],
        "fusion": [False], "graph_padding": False,
    })
    assert norm["status"] == "modeled"
    assert norm["feasible_target_ms"] == pytest.approx(
        pred["metrics"]["e2e_ms"]["p50"], rel=5e-2,
    )
    decode_steps = [
        step["draft_steps"] for step in norm["plans"][0]["steps"]
        if step.get("draft_steps")
    ]
    assert decode_steps and decode_steps[0][0]["accepted_tokens"] == 0
    assert decode_steps[1][0]["accepted_tokens"] == 2

    no_draft = _draft_case([0, 2, 1])
    no_draft["runtime"].pop("metadata")
    plain = predict(no_draft)
    assert plain["status"] == "modeled"
    assert plain["requests"][0]["generated_tokens"] == 12
    assert plain["requests"][0].get("draft_steps") == []


def test_should_budget_is_caller_controlled_and_counts_attempts():
    import json as _json
    from mindie_sim.a3_should import should_cost

    case = json.load(open("examples/model-reference/cases/qwen3.8-27b.json", encoding="utf-8"))
    full = should_cost(case, constraints={
        "tp": [8], "dp": [1], "pp": [1], "ep": [1],
        "fusion": [False, True], "graph_padding": False,
    })
    assert full["status"] == "modeled"
    assert full["plan_family_truncated"] is False
    assert full["plans_generated"] == full["plans_attempted"]
    assert full["plans_attempted"] == full["plans_evaluated"]
    assert full["plans_feasible"] == full["plans_successful"]
    assert full["plans_unexplored_after_beam"] == 0
    assert full["search_budget"] is None

    budgeted = should_cost(case, constraints={
        "tp": [8], "dp": [1], "pp": [1], "ep": [1],
        "fusion": [False, True], "graph_padding": False,
        "search_budget": 1,
    })
    assert budgeted["status"] == "modeled"
    assert budgeted["search_budget"] == 1
    assert budgeted["search_budget_effective"] == 1
    assert budgeted["plans_attempted"] == 1
    assert budgeted["plans_evaluated"] == 1
    assert budgeted["plans_unexplored_after_beam"] == (
        full["plans_generated"] - budgeted["plans_attempted"]
    )
    assert budgeted["plan_family_truncated"] is True

    # An explicit finite domain is preserved when no budget is requested.
    # The reference machine has real core/pool capacities, so a fusion plan can
    # expose multiple worker variants; pin the caller's intended one-worker
    # domain explicitly rather than assuming fusion=True is sufficient.
    one = should_cost(case, constraints={
        "tp": [8], "dp": [1], "pp": [1], "ep": [1],
        "fusion": [True], "graph_padding": False,
        "worker_count": [1],
    })
    assert one["plans_generated"] == 1
    assert one["plans_attempted"] == 1
    assert one["plans_unexplored_after_beam"] == 0


def test_should_reports_complete_counts_when_no_legal_plan():
    from mindie_sim.a3_should import should_cost

    case = {
        "machine": {"num_devices": 1, "device": {
            "hbm_gib": 1, "cube_flops_per_s": 1e12,
            "vector_ops_per_s": 1e12, "hbm_bw_bytes_per_s": 1e12,
            "mte_bw_bytes_per_s": 2e12, "l2_bytes": 1 << 20,
        }},
        "model": {"catalog_key": "DeepSeek-V4-Pro-0813"},
        "workload": {"requests": [{"prompt_tokens": 8, "output_tokens": 1}]},
        "runtime": {"max_num_batched_tokens": 16, "max_model_len": 64},
    }
    result = should_cost(case, constraints={
        "tp": [1], "dp": [1], "pp": [1], "ep": [1],
        "fusion": [False], "graph_padding": False,
    })
    assert result["status"] == "infeasible"
    assert result["reason"] == "no_legal_plan"
    assert result["plans_generated"] >= 1
    assert result["plans_attempted"] == 0
    assert result["plans_evaluated"] == 0
    assert result["plans_feasible"] == 0
    assert result["plans_successful"] == 0
    assert result["plans_pruned_memory_or_legality"] >= 1
    assert result["plans_unexplored_after_beam"] == 0


def test_should_budget_keeps_topology_diversity():
    import json as _json
    from mindie_sim.a3_should import should_cost

    case = _json.load(open("examples/model-reference/cases/qwen3.8-27b.json", encoding="utf-8"))
    result = should_cost(case, constraints={"search_budget": 2})
    assert result["status"] == "modeled"
    devices = [item["plan"]["devices"] for item in result["plans"]]
    assert len(devices) == 2
    assert max(devices) > min(devices)
    assert result["plans_unexplored_after_beam"] > 0


def test_comm_identity_parser_keeps_requested_and_observed_and_rejects_name_only():
    import copy
    import json as _json
    from mindie_sim.a3_comm_identity import (
        identity_fixture, load_identity, parse_profile_identity,
    )

    payload = _json.load(open("examples/a3_comm_identity_raw.json", encoding="utf-8"))
    records = parse_profile_identity(payload)
    assert len(load_identity(payload)) == 36
    fixture = identity_fixture(payload)
    assert "schema_version" not in fixture
    assert len(load_identity(fixture)) == 36
    with pytest.raises(A3InputError):
        load_identity({"run_id": "unknown-document"})
    assert len(records) == 36
    assert {record.requested_mode for record in records} == {2, 3}
    assert any(
        record.requested_execution == "aiv"
        and record.observed_expansion == "aicpu"
        and record.observed_fallback
        for record in records
    )
    assert all(not record.calibration_eligible for record in records)
    assert all(record.profiling_timing_excluded for record in records)

    bad = copy.deepcopy(payload)
    bad["results"][0]["rows"][0]["duration_ms"] = 1.0
    with pytest.raises(A3InputError):
        parse_profile_identity(bad)

    bad = copy.deepcopy(payload)
    bad["results"][3]["rows"][-1]["observed_fallback"] = False
    with pytest.raises(A3InputError):
        parse_profile_identity(bad)

    bad = copy.deepcopy(payload)
    bad["results"][0]["checksum_reduce_kernels_excluded"] = 0
    with pytest.raises(A3InputError):
        parse_profile_identity(bad)


def test_aiv_core_reservations_convert_to_service_fraction_and_scope_per_die():
    from mindie_sim.a3_cost import CalibrationResult, CostedOp
    from mindie_sim.a3_ops import OpSpec
    from mindie_sim.a3_resources import (
        ResourceEngine, ResourceSpec, add_costed_ops,
    )
    cal = CalibrationResult("declared", 1, 0, None, False, False, "declared")
    def op(name, stage=0, reservations=None):
        shape = (4,)
        return OpSpec(
            name, "allreduce", "prefill", stage, 0, ("r",), shape, shape,
            shape, shape, metadata=(
                ("comm_stages", ({
                    "kind": "aiv", "work": {"link_edge_bytes": 10},
                    "reservations": reservations or {"aiv_cores": 2},
                    "reservation_capacities": {"aiv_cores": 48},
                },)),
            ),
        )
    resources = {
        "vector@s0": ResourceSpec("vector@s0", 48.0),
        "link@s0": ResourceSpec("link@s0", 1.0),
    }
    engine = ResourceEngine(resources)
    ids = add_costed_ops(
        engine,
        [CostedOp(op("comm"), (), 0, 0, cal),
         CostedOp(
             OpSpec("model", "rmsnorm", "prefill", 0, 0, ("r",),
                    (4,), (4,), (4,), (4,), dtype="bf16"),
             (("vector", 48.0 * 3.0),), 0, 0, cal,
         )],
        {0: {"vector": "vector@s0", "link": "link@s0"}},
        prefix="aiv", start_ms=0,
    )
    result = engine.run()
    assert result["completed_at_ms"][ids["model"]] == pytest.approx(
        3.0 / (1.0 - 2.0 / 48.0), rel=1e-9,
    )

    resources = {
        f"{kind}@s{stage}": ResourceSpec(
            f"{kind}@s{stage}", 48.0 if kind == "vector" else 1.0,
        )
        for stage in (0, 1) for kind in ("vector", "link")
    }
    maps = {
        stage: {"vector": f"vector@s{stage}", "link": f"link@s{stage}"}
        for stage in (0, 1)
    }
    stage = {
        "kind": "aiv", "work": {"link_edge_bytes": 10},
        "reservations": {"aiv_cores": 16},
        "reservation_capacities": {"aiv_cores": 48},
    }
    def staged(name, stage_index):
        shape = (4,)
        return OpSpec(
            name, "allreduce", "prefill", stage_index, 0, ("r",), shape,
            shape, shape, shape, metadata=(("comm_stages", (stage,)),),
        )
    engine = ResourceEngine(resources)
    ids = add_costed_ops(
        engine,
        [CostedOp(staged("d0", 0), (), 0, 0, cal),
         CostedOp(staged("d1", 1), (), 0, 0, cal)],
        maps, prefix="dies", start_ms=0,
    )
    result = engine.run()
    assert result["completed_at_ms"][ids["d0"]] == pytest.approx(10.0)
    assert result["completed_at_ms"][ids["d1"]] == pytest.approx(10.0)


def test_prepare_translation_keeps_capacity_directions_request_and_owner():
    from types import SimpleNamespace
    from mindie_sim.a3_async import add_prepare_tasks, prepare_specs_from_candidates
    from mindie_sim.a3_resources import ResourceEngine, ResourceSpec

    candidate = SimpleNamespace(
        candidate_id="prepare.ring", kind="compressor_buffer_capacity",
        known_at="request_arrival", lifetime="request_resident",
        owner="stage1_request_ring", consumer_selector="c2_ring",
        consumer_op_ids=("s1.l14.c2_ring",), first_consumer_op_ids=("s1.l14.c2_ring",),
        replace_op_ids=(), tensor_shape=(32, 2, 512), bytes=0.0,
        operations=0.0, resources={"hbm_reserved_bytes": 131072.0},
        request_id="late", arrival_ms=100.0, iteration=0, reusable=False,
        owner_stage=1, owner_rank="rank_local", capacity_bytes=131072.0,
        read_bytes=0.0, write_bytes=0.0, service_bytes=0.0,
        depends_on=(), cost_once_key="ring:late", host_rates_required=(),
        metadata={"capacity_not_service_bytes": True},
    )
    traffic = SimpleNamespace(
        **{**candidate.__dict__, "candidate_id": "prepare.traffic",
           "kind": "kda_layout", "known_at": "model_load",
           "request_id": None, "arrival_ms": 0.0, "reusable": True,
           "owner_stage": 1, "capacity_bytes": 0.0,
           "read_bytes": 40697856.0, "write_bytes": 20348928.0,
           "service_bytes": 20348928.0, "resources": {}}
    )
    specs, gaps = prepare_specs_from_candidates(
        [candidate, traffic], host_rates={}, max_hbm_bytes=64 * 1024 ** 3,
    )
    assert gaps == ()
    ring, layout = specs
    assert ring.request_id == "late" and ring.owner_stage == 1
    assert ring.capacity_bytes == 131072 and ring.service_bytes == 0
    assert layout.read_bytes == 40697856 and layout.write_bytes == 20348928

    engine = ResourceEngine({
        "hbm@s0": ResourceSpec("hbm@s0", 1.0),
        "mte2@s0": ResourceSpec("mte2@s0", 1.0),
        "mte3@s0": ResourceSpec("mte3@s0", 1.0),
        "hbm@s1": ResourceSpec("hbm@s1", 1.0),
        "mte2@s1": ResourceSpec("mte2@s1", 1.0),
        "mte3@s1": ResourceSpec("mte3@s1", 1.0),
        "hbm_capacity@s1": ResourceSpec("hbm_capacity@s1", 1 << 30, "bytes"),
        "prep": ResourceSpec("prep", 1.0),
    })
    stage_map = {
        0: {"hbm": "hbm@s0", "mte2": "mte2@s0", "mte3": "mte3@s0"},
        1: {"hbm": "hbm@s1", "mte2": "mte2@s1", "mte3": "mte3@s1",
            "hbm_capacity": "hbm_capacity@s1"},
    }
    request = SimpleNamespace(request_id="late", arrival_ms=100.0)
    ids = add_prepare_tasks(
        engine, specs, stage_map, [request], now_ms=0.0, prefix="prep",
    )
    result = engine.run()
    ring_task = engine.tasks[ids[0]]
    layout_task = engine.tasks[ids[1]]
    assert result["completed_at_ms"][ids[0]] >= 100.0
    assert "hbm@s1" not in dict(ring_task.demands)
    assert dict(ring_task.reservations)["hbm_capacity@s1"] == 131072
    assert dict(layout_task.demands)["mte2@s1"] == 40697856
    assert dict(layout_task.demands)["mte3@s1"] == 20348928


def test_fusion_execution_width_reserves_physical_core_fraction():
    from mindie_sim.a3_cost import CalibrationResult, CostedOp
    from mindie_sim.a3_ops import OpSpec
    from mindie_sim.a3_resources import (
        ResourceEngine, ResourceSpec, add_costed_ops,
    )
    cal = CalibrationResult("declared", 1, 0, None, False, False, "declared")
    stage = {
        "name": "tile", "family": "fused_tile",
        "flops": 1000.0, "resources": {"mte1": 5.0},
        "execution_width": {
            "aic_cores": 1, "aiv_cores": 0,
            "scope": "per_die", "partition": "single_core_tile",
        },
        "deps": (),
    }
    op = OpSpec(
        "fused", "fused_tile", "prefill", 0, 0, ("r",), (1,), (1,),
        (1,), (1,),
        metadata=(("fusion_stages", (stage,)),),
    )
    engine = ResourceEngine({
        "cube": ResourceSpec("cube", 1600.0),
        "mte1": ResourceSpec("mte1", 100.0),
    })
    add_costed_ops(
        engine, [CostedOp(op, (), 0, 0, cal)],
        {0: {"cube": "cube", "mte1": "mte1"}},
        prefix="width", start_ms=0, core_counts={"aic": 16, "aiv": 16},
    )
    event = engine.run()["trace"][0]
    assert event["demands"]["mte1"] == 5.0
    assert event["end_ms"] == pytest.approx(10.0)
    assert event["residency"][0]["amount"] == pytest.approx(100.0)


def test_draft_query_tokens_are_charged_to_batch_budget_and_itl_scope_is_explicit():
    case = _draft_case([5], output=7)
    case["runtime"]["max_num_batched_tokens"] = 2
    case["runtime"]["metadata"] = {
        "draft_mode": "dspark", "draft_accept_tokens": 5,
    }
    case["workload"]["requests"][0]["prompt_tokens"] = 2
    result = predict(case)
    assert result["status"] == "modeled"
    q = result["requests"][0]
    assert (q["generated_tokens"], q["computed_tokens"]) == (7, 8)
    assert [item["proposal_tokens"] for item in q["draft_steps"]] == [1, 1, 1]
    assert all(item["accepted_tokens"] <= 1 for item in q["draft_steps"])
    verify_shapes = [
        op["semantic_shape"][0]
        for step in result["traces"] for op in step["ops"]
        if op["op_id"] == "verify.embedding"
    ]
    assert verify_shapes == [1, 1, 1]
    assert q["itl_scope"] == "engine_output_burst_interval"
    assert len(q["token_itl_ms"]) == q["generated_tokens"] - 1


def _curve_op(shape, metadata):
    from mindie_sim.a3_ops import OpSpec
    launch = tuple(shape)
    return OpSpec(
        op_id="curve-op", family="gemm", phase="operator", stage=0, layer=0,
        requests=(), semantic_shape=launch, local_shape=launch,
        padded_shape=launch, launch_shape=launch,
        flops=2.0 * shape[0] * shape[1] * shape[2],
        hbm_bytes=1024.0, dtype="bf16", layout="bf16", recipe="curve",
        metadata=tuple(metadata),
    )


def _curve_artifact(*, quantization="none", transition=False):
    segments = ([{"x_min": 1.0, "x_max": 8.0, "slope_ms_per_unit": 0.01,
                  "intercept_ms": 0.02, "n": 2},
                 {"x_min": 32.0, "x_max": 128.0, "slope_ms_per_unit": 0.1,
                  "intercept_ms": 0.0, "n": 2}]
                if transition else
                [{"x_min": 1.0, "x_max": 128.0, "slope_ms_per_unit": 0.01,
                  "intercept_ms": 0.02, "n": 2}])
    return {
        "curve_count": 1,
        "curves": [{
            "curve_id": "curve:toy",
            "condition_key": "toy",
            "axis": "m",
            "fixed_shape": {"n": 128, "k": 128},
            "physical": {
                "family": "gemm", "recipe": "r", "execution_kind": "primitive",
                "worker_recipe": "w", "backend": "toy",
                "target_match": True, "coverage_scope": "declared",
                "precision": {
                    "input_dtype": "bf16", "weight_dtype": "bf16",
                    "output_dtype": "bf16", "compute_dtype": "bf16",
                    "accumulation_dtype": "backend_default",
                    "quantization": quantization, "weight_format": "bf16",
                    "activation_format": "bf16", "compute_format": "bf16",
                },
                "layout": "contiguous", "quantization": quantization,
                "features": {}, "hardware_key": "Ascend910_9362",
                "source_class": "measured",
            },
            "protocol": {"timer": "device_event", "warmup": 10,
                         "iterations": 30, "trials": 1,
                         "preallocated": True,
                         "cache_policy": "observed_production_run_unspecified",
                         "timing_scope": "device_event_warm",
                         "measurement_scope": "toy"},
            "segments": segments,
            "transition_brackets": [[8.0, 32.0]] if transition else [],
            "train_support": {"x_min": 1.0, "x_max": 128.0, "point_count": 4},
        }],
    }


def _curve_metadata(artifact, *, quantization="none"):
    curve = artifact["curves"][0]
    physical = curve["physical"]
    return (
        ("curve_recipe", physical["recipe"]),
        ("curve_execution_kind", physical["execution_kind"]),
        ("curve_worker_recipe", physical["worker_recipe"]),
        ("curve_backend", physical["backend"]),
        ("curve_target_match", physical["target_match"]),
        ("curve_coverage_scope", physical["coverage_scope"]),
        ("curve_layout", physical["layout"]),
        ("curve_quantization", quantization),
        ("curve_precision", dict(physical["precision"])),
        ("curve_features", {}),
        ("curve_protocol", dict(curve["protocol"])),
        ("curve_axis", "m"),
    )


def test_prediction_consumes_compatible_piecewise_curve_and_never_negative_gap():
    from mindie_sim.a3_cost import PredictiveCostModel
    machine = MachineSpec.from_mapping({
        "name": "Ascend910_9362", "sku": "Ascend910_9391",
        "num_devices": 1, "hardware_key": "Ascend910_9362",
    })
    artifact = _curve_artifact()
    op = _curve_op((32, 128, 128), _curve_metadata(artifact))
    model = PredictiveCostModel(
        machine, curve_artifact=artifact, quantization="bf16_qwen",
    )
    cost = model.cost(op)
    assert cost.calibration.status == "calibrated_piecewise_curve"
    assert cost.estimated_ms == pytest.approx(0.34)

    gap = _curve_artifact(transition=True)
    gap_op = _curve_op((16, 128, 128), _curve_metadata(gap))
    gap_cost = PredictiveCostModel(
        machine, curve_artifact=gap, quantization="bf16_qwen",
    ).cost(gap_op)
    assert gap_cost.calibration.status == "curve_transition_unresolved"
    assert gap_cost.estimated_ms > 0

    # The generic primitive path must not discard the bracket.  It keeps the
    # analytic estimate and reports the actual missing physical condition.
    generic_gap = _curve_op((16, 128, 128), ())
    generic_cost = PredictiveCostModel(
        machine, curve_artifact=gap, quantization="bf16_qwen",
    ).cost(generic_gap)
    assert generic_cost.calibration.status == "curve_transition_unresolved"
    assert generic_cost.curve_result["status"] == "transition_unresolved"
    assert generic_cost.curve_result["lower_bound_ms"] is not None
    assert generic_cost.curve_result["upper_bound_ms"] is not None
    assert generic_cost.curve_result["missing_condition"]["shape"] == {
        "m": 16, "n": 128, "k": 128,
    }
    assert generic_cost.curve_result["missing_condition"]["protocol"]


def test_transition_prediction_materializes_missing_physical_request():
    store = SQLiteCalibrationStore(":memory:")
    protocol = {"timer": "device_event", "warmup": 10, "iterations": 30}
    prediction = {"traces": [{
        "duration_ms": 10.0,
        "ops": [{
            "family": "gemm", "phase": "operator", "recipe": "catalog_down",
            "shape_features": {"m": 16, "n": 128, "k": 128},
            "dtype": "bf16", "layout": "contiguous",
            "estimated_ms": 2.0, "analytic_ms": 2.0,
            "calibration_status": "curve_transition_unresolved",
            "curve_result": {
                "status": "transition_unresolved",
                "matched_curve": "curve:toy",
                "condition_key": "lin:toy",
                "axis": "m", "x": 16.0,
                "lower_bound_ms": 1.5, "upper_bound_ms": 2.5,
                "reason": "no train support inside transition bracket",
                "protocol": protocol,
                "missing_condition": {
                    "hardware_key": "Ascend910_9362",
                    "family": "gemm", "recipe": None, "axis": "m",
                    "value": 16.0,
                    "shape": {"m": 16, "n": 128, "k": 128},
                    "fixed_shape": {"n": 128, "k": 128},
                    "protocol": protocol,
                    "precision": {
                        "input_dtype": "bf16", "weight_dtype": "bf16",
                        "output_dtype": "bf16", "compute_dtype": "bf16",
                        "accumulation_dtype": "backend_default",
                        "quantization": "none",
                    },
                    "layout": "contiguous",
                    "target_match": None,
                    "coverage_scope": "declared",
                },
            },
        }],
    }]}
    recorded = record_prediction_misses(
        store, prediction, hardware_key="Ascend910_9362",
    )
    assert recorded["recorded_requests"] == 1
    request = store.requests()[0]
    assert request["op_family"] == "gemm"
    assert request["shape"] == {"m": 16.0, "n": 128.0, "k": 128.0}
    assert request["details"]["protocol"] == protocol
    assert request["details"]["curve_transition"]["condition_key"] == "lin:toy"
    manifest = store.manifest()
    exported = manifest["requests"][0]
    assert exported["protocol"] == protocol
    assert exported["missing_condition"]["shape"] == {
        "m": 16, "n": 128, "k": 128,
    }




def _native_rms_artifact(
    *, epsilon: float = 1e-6, hardware: str = "Ascend910_9362",
    dtype: str = "bf16",
):
    return {"curves": [{
        "curve_id": "curve:qwen-rms",
        "condition_key": "native:qwen-rms",
        "axis": "tokens",
        "fixed_shape": {"width": 5120},
        "physical": {
            "family": "rmsnorm", "recipe": "npu_rms_norm",
            "execution_kind": "fused_kernel", "worker_recipe": "npu_rms_norm",
            "backend": "torch_npu_installed_api", "target_match": False,
            "coverage_scope": "native_primitive_at_source_declared_2d_geometry",
            "precision": {
                "input_dtype": dtype, "compute_dtype": dtype,
                "accumulation_dtype": "backend_default",
            },
            "layout": "contiguous", "quantization": "none",
            "features": {"epsilon": epsilon}, "hardware_key": hardware,
            "source_class": "measured",
        },
        "protocol": {
            "iterations": 30, "timer": "device_event",
            "trials": 2, "warmup": 10, "preallocated": False,
            "cache_policy": "same_preallocated_buffers_reused_no_cache_flush",
            "timing_scope": "device_event_warm_operator",
        },
        "segments": [{
            "x_min": 1.0, "x_max": 256.0,
            "slope_ms_per_unit": 0.0001, "intercept_ms": 0.01, "n": 2,
        }],
        "train_support": {
            "x_min": 1.0, "x_max": 256.0, "point_count": 3,
            "x_values": [1.0, 128.0, 256.0],
        },
        "transition_brackets": [],
    }]}


def test_explicit_native_rms_primitive_transfer_is_candidate_only():
    from mindie_sim.a3_cost import PredictiveCostModel
    from mindie_sim.a3_workload import (
        IterationPlan, RequestView, derive_iteration_work,
    )

    payload = case(
        model={"catalog_key": "qwen3.8-27b"},
        machine={**machine(8), "hardware_key": "Ascend910_9362"},
        workload={"requests": [{"prompt_tokens": 128, "output_tokens": 1}]},
        runtime={
            "tp": 8, "max_num_batched_tokens": 128, "max_model_len": 1024,
            "curve_allow_primitive_transfer": True,
        },
    )
    case_spec = CaseSpec.from_mapping(payload)
    work = derive_iteration_work(
        IterationPlan((RequestView("r0", 128, 0, True, "prefill"),)),
        case_spec.model, case_spec.runtime, case_spec.machine,
    )
    rms_ops = [
        op for op in work.ops
        if op.family == "rmsnorm"
        and op.local_shape == (128, 5120)
        and float(dict(op.metadata).get("norm_eps", 0.0)) == 1e-6
    ]
    assert rms_ops

    def cost(artifact, *, opt_in=True, protocol=None):
        model = PredictiveCostModel(
            case_spec.machine,
            compute_dtype=case_spec.model.compute_dtype,
            weight_dtype=case_spec.model.weight_dtype,
            activation_dtype=case_spec.model.activation_dtype,
            quantization=case_spec.model.quantization,
            curve_artifact=artifact,
            curve_allow_primitive_transfer=opt_in,
            calibration_protocol=protocol,
        )
        return model.cost(rms_ops[0])

    positive = cost(_native_rms_artifact())
    assert positive.calibration.status == "calibrated_piecewise_curve"
    assert positive.curve_result["transfer_scope"] == (
        "explicit_native_primitive_transfer"
    )
    assert positive.curve_result["production_match"] is False
    assert positive.curve_result["target_match"] is False
    assert positive.calibration.reason.startswith("approximate_primitive_transfer")

    # No opt-in, wrong epsilon/dtype/hardware/protocol remain analytic.
    assert cost(
        _native_rms_artifact(), opt_in=False,
    ).calibration.status == "uncalibrated_analytic"
    assert cost(
        _native_rms_artifact(epsilon=1e-5),
    ).calibration.status == "uncalibrated_analytic"
    assert cost(
        _native_rms_artifact(dtype="fp16"),
    ).calibration.status == "uncalibrated_analytic"
    assert cost(
        _native_rms_artifact(hardware="Ascend910_other"),
    ).calibration.status == "uncalibrated_analytic"
    assert cost(
        _native_rms_artifact(), protocol={"mode": "graph_block"},
    ).calibration.status == "uncalibrated_analytic"


def test_generic_curve_explicit_protocol_is_exact_not_prefix():
    from mindie_sim.a3_cost import PredictiveCostModel
    machine = MachineSpec.from_mapping({
        "name": "Ascend910_9362", "sku": "Ascend910_9391",
        "num_devices": 1, "hardware_key": "Ascend910_9362",
    })
    artifact = _curve_artifact()
    protocol = dict(artifact["curves"][0]["protocol"])
    op = _curve_op((32, 128, 128), ())
    exact = PredictiveCostModel(
        machine, curve_artifact=artifact, calibration_protocol=protocol,
    ).cost(op)
    assert exact.calibration.status == "calibrated_piecewise_curve"

    for partial in (
        {"mode": "eager_block"},
        {**protocol, "warmup": protocol["warmup"] + 1},
        {"timer": protocol["timer"]},
    ):
        rejected = PredictiveCostModel(
            machine, curve_artifact=artifact,
            calibration_protocol=partial,
        ).cost(op)
        assert rejected.calibration.status == "uncalibrated_analytic"
        assert rejected.curve_result is None


def test_predict_retains_curve_artifact_when_calibration_store_is_also_used():
    from dataclasses import replace
    payload = case(
        dense_model(
            layers=1, hidden_size=128, intermediate_size=64,
            num_heads=4, num_kv_heads=1, head_dim=32, vocab_size=1000,
        ),
        machine={**machine(), "hardware_key": "Ascend910_9362"},
        workload={"requests": [{"prompt_tokens": 32, "output_tokens": 1}]},
        runtime={"max_num_batched_tokens": 64, "max_model_len": 128},
    )
    case_spec = replace(
        CaseSpec.from_mapping(payload), curve_artifact=_curve_artifact(),
    )
    result = predict(case_spec, calibration=CalibrationStore())
    assert result["status"] == "modeled"
    statuses = {
        op["calibration_status"]
        for trace in result["traces"] for op in trace["ops"]
    }
    assert "calibrated_piecewise_curve" in statuses
    curve_results = [
        op["curve_result"]
        for trace in result["traces"] for op in trace["ops"]
        if op.get("curve_result")
    ]
    assert curve_results and curve_results[0]["status"] == "ok"


def test_piecewise_curve_does_not_cross_quantization_identity():
    from mindie_sim.a3_cost import PredictiveCostModel
    machine = MachineSpec.from_mapping({
        "name": "Ascend910_9362", "sku": "Ascend910_9391",
        "num_devices": 1, "hardware_key": "Ascend910_9362",
    })
    artifact = _curve_artifact(quantization="fp8")
    op = _curve_op(
        (32, 128, 128),
        tuple(
            item for item in _curve_metadata(artifact, quantization="fp8")
            if item[0] != "curve_quantization"
        ),
    )
    # A BF16 model query must not consume an FP8 physical curve.
    cost = PredictiveCostModel(
        machine, curve_artifact=artifact, quantization="bf16",
    ).cost(op)
    assert cost.calibration.status == "uncalibrated_analytic"


def test_physical_quantization_labels_are_not_model_variant_quantization():
    from mindie_sim.a3_cost import _physical_quantization
    assert _physical_quantization("bf16_qwen") == "none"
    assert _physical_quantization("bf16") == "none"
    assert _physical_quantization("mxfp4") == "mxfp4"
    assert _physical_quantization("fp8_e4m3") == "fp8"


def test_prepare_residency_ledger_holds_rejects_overcommit_and_releases_request():
    from mindie_sim.a3_async import (
        add_prepare_tasks, prepare_residency_report, release_prepare_residency,
    )
    from mindie_sim.a3_types import PrepareSpec

    engine = ResourceEngine({
        "capacity": ResourceSpec("capacity", 100.0, "bytes"),
    })
    specs = (
        PrepareSpec("resident0", owner_stage=0, capacity_bytes=80,
                    lifetime="model_resident"),
        PrepareSpec("resident1", owner_stage=0, capacity_bytes=80,
                    lifetime="model_resident"),
    )
    with pytest.raises(A3InputError, match="prepare_capacity_overcommit"):
        add_prepare_tasks(
            engine, specs, {0: {"hbm_capacity": "capacity"}}, [], now_ms=0,
        )
    report = prepare_residency_report(engine)
    assert report is not None
    assert len(report["overcommit_entries"]) == 1
    assert report["pools"]["capacity"]["occupied_bytes"] == 80

    positive = ResourceEngine({
        "capacity": ResourceSpec("capacity", 100.0, "bytes"),
    })
    add_prepare_tasks(
        positive,
        (PrepareSpec("resident", owner_stage=0, capacity_bytes=80,
                     lifetime="model_resident"),),
        {0: {"hbm_capacity": "capacity"}}, [], now_ms=0,
    )
    assert prepare_residency_report(positive)["pools"][
        "capacity"
    ]["occupied_bytes"] == 80

    request_engine = ResourceEngine({
        "capacity": ResourceSpec("capacity", 100.0, "bytes"),
    })
    add_prepare_tasks(
        request_engine,
        (PrepareSpec("out", owner_stage=0, capacity_bytes=40,
                     lifetime="request_resident", request_id="r0"),),
        {0: {"hbm_capacity": "capacity"}},
        [type("R", (), {"request_id": "r0", "arrival_ms": 0.0})()],
        now_ms=0,
    )
    assert prepare_residency_report(request_engine)["pools"][
        "capacity"
    ]["occupied_bytes"] == 40
    assert release_prepare_residency(
        request_engine, request_id="r0", now_ms=5,
    ) == 1
    assert prepare_residency_report(request_engine)["pools"][
        "capacity"
    ]["occupied_bytes"] == 0

def test_graph_mode_does_not_consume_legacy_eager_curve_protocol():
    from mindie_sim.a3_cost import PredictiveCostModel
    machine = MachineSpec.from_mapping({
        "name": "Ascend910_9362", "sku": "Ascend910_9391",
        "num_devices": 1, "hardware_key": "Ascend910_9362",
    })
    artifact = _curve_artifact()
    op = _curve_op((32, 128, 128), _curve_metadata(artifact))
    model = PredictiveCostModel(
        machine, curve_artifact=artifact, quantization="bf16",
        graph_mode="force",
    )
    assert model.curve_prediction(op) is None
    assert model.cost(op).calibration.status == "uncalibrated_analytic"


def test_calibration_protocol_key_ignores_outcome_evidence_but_not_mode():
    from pathlib import Path as _Path
    db = _Path("/tmp") / f"protocol-key-test-{id(object())}.sqlite"
    try:
        store = SQLiteCalibrationStore(str(db))
        base = {
            "timer": "device_event", "warmup": 10, "iterations": 30,
            "preallocated": True, "measurement_scope": "per_operator",
            "mode": "eager_block", "api_invocations_per_block": 1,
            "host_calls_per_block": 1,
        }
        for run, address, duration in (("run-a", 1, 1000), ("run-b", 2, 2000)):
            protocol = {
                **base, "capture_unique_address_count": address,
                "duration_ms": duration, "run_id": run,
            }
            store.record_sample(
                op_family="gemm", phase="operator", recipe="torch_mm_out",
                shape={"m": 8, "n": 128, "k": 128},
                duration_ms=duration, analytic_ms=10.0,
                dtype="bf16", layout="contiguous",
                hardware_key="Ascend910_9362", features={"quantization": "none"},
                source="toy", source_class="measured", run_id=run,
                case_id="c", split="train", timing_scope="device_event_warm",
                protocol=protocol,
            )
        stats = store.statistics()
        assert len(stats) == 1
        assert sorted(stats[0]["run_ids"]) == ["run-a", "run-b"]
        observations = store.as_calibration_store().observations()
        assert observations and observations[0].protocol
        graph_engine = SQLiteCalibrationStore(
            str(db.with_name(db.name + "-graph"))
        )
        graph_engine.record_sample(
            op_family="gemm", phase="operator", recipe="torch_mm_out",
            shape={"m": 8, "n": 128, "k": 128},
            duration_ms=3000, analytic_ms=10.0,
            dtype="bf16", layout="contiguous",
            hardware_key="Ascend910_9362", features={"quantization": "none"},
            source="toy", source_class="measured", run_id="run-c",
            case_id="c", split="train", timing_scope="device_event_warm",
            protocol={**base, "mode": "graph_block",
                      "api_invocations_per_block": 32,
                      "capture_invocations": 32},
        )
        assert len(graph_engine.statistics()) == 1
    finally:
        for suffix in ("", "-wal", "-shm", "-graph", "-graph-wal", "-graph-shm"):
            try:
                _Path(str(db) + suffix).unlink()
            except OSError:
                pass



def test_calibration_exact_protocol_rejects_wildcards_and_irrelevant_rows():
    base = {
        "op_family": "gemm", "phase": "prefill", "recipe": "plain",
        "shape": {"m": 8, "n": 128, "k": 128},
        "duration_ms": 2.0, "analytic_ms": 1.0,
        "source_class": "measured", "source": "TOY contract; not NPU",
        "machine_key": "toy", "dtype": "bf16", "layout": "contiguous",
        "features": {},
    }
    query = {
        key: base[key] for key in (
            "op_family", "phase", "recipe", "shape", "machine_key",
            "dtype", "layout", "features",
        )
    }
    eager = {
        "mode": "eager_block", "api_invocations_per_block": 1,
        "output_lifetime": "retain_last_only",
    }
    graph = {
        "mode": "graph_single_repeated", "api_invocations_per_block": 32,
        "output_lifetime": "graph_owned",
    }

    def factor(observation_protocol, query_protocol, extra=()):
        store = CalibrationStore.from_mapping([
            {**base, "protocol": observation_protocol}, *extra,
        ])
        return store.factor_for(**query, protocol=query_protocol)

    assert factor(eager, eager).status == "calibrated_exact"
    assert not factor({}, {"mode": "eager_block"}).status.startswith("calibrated")
    assert not factor(
        {"mode": "eager_block"},
        {"mode": "eager_block", "api_invocations_per_block": 32},
    ).status.startswith("calibrated")
    assert not factor(
        {"mode": "eager_block"},
        {"mode": "eager_block", "output_lifetime": "retain_all"},
    ).status.startswith("calibrated")
    assert not factor(
        graph, {**graph, "mode": "graph_block"},
    ).status.startswith("calibrated")
    assert not factor(eager, {}).status.startswith("calibrated")

    unrelated = [
        {**base, "op_family": "rmsnorm", "protocol": {**eager, "warmup": 10}},
        {**base, "machine_key": "other", "protocol": {**eager, "warmup": 10}},
    ]
    assert factor(eager, eager, extra=unrelated).status == "calibrated_exact"

    explicit = CalibrationStore.from_mapping([
        {**base, "protocol": eager},
        {**base, "protocol": {**eager, "warmup": 10}},
    ])
    assert explicit.factor_for(**query, protocol=eager).status == "calibrated_exact"
    assert explicit.factor_for(**query, protocol=None).status == (
        "uncalibrated_ambiguous_protocol"
    )
