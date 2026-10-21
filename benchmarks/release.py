"""[Role]
  Release microbenchmark for deterministic, public-input ``mindie_sim``
  workloads: the four-layer roofline / transfer-boundary / subgraph-screening
  path, the A3 physical-calibration whole-network TTFT/TPOT prediction on the
  checked-in Qwen2.5-7B-Instruct fixture, and the opt-in fine-grained
  Vector/GEMM ``operator-plan`` estimator.

[Invariants]
  * Every workload is deterministic and reads checked-in public fixtures only.
  * Semantic output is asserted before any metric is recorded: a failed
    assertion exits non-zero and no pass result is written.
  * Metrics are CPU wall-clock latency per workload invocation in ``us/op``,
    never a claim about Ascend NPU throughput or a measured device outcome.
  * No network, device, subprocess, provider probe or live-NPU call is made.

[Timeouts & Locks]
  Bounded finite iteration counts; no sleeps, threads, locks or background
  services.  Every case finishes well inside the release-suite timeout.

[Side-Effect Policy]
  Writes exactly one JSON result to ``MINDIE_BENCH_RESULT_PATH`` and nothing
  else; every other input is read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindie_sim import (  # noqa: E402 - repository path must exist first
    analyze_subgraph_fusion,
    evaluate_operator_hap,
    solve_periodic_pipeline,
)
from mindie_sim.a3_framework import predict  # noqa: E402
from mindie_sim.a3_physical_calibration import PhysicalCalibration  # noqa: E402
from mindie_sim.comm import solve_collective_bound, solve_transfer_bound  # noqa: E402
from mindie_sim.hardware import resolve_device_capability_profile  # noqa: E402
from mindie_sim.operator_estimation import operator_plan_report  # noqa: E402

MIB = 1024 * 1024
A3_SKU = "Ascend910_9362"
QWEN_FIXTURE = ROOT / "benchmarks" / "fixtures" / "qwen2.5-7b-instruct.json"
OPERATOR_FIXTURE = ROOT / "examples" / "operator-plans" / "mixed.json"

_PIPELINE_SPEC = {
    "name": "benchmark-three-stage-pipeline",
    "phases": 1,
    "tiles_per_period": 1,
    "stages": [
        {"name": "load", "pipe": "MTE2"},
        {"name": "mma", "pipe": "M"},
        {"name": "store", "pipe": "MTE3"},
    ],
    "edges": [
        {"src": "load", "dst": "mma", "distance": 0},
        {"src": "mma", "dst": "store", "distance": 0},
        {"src": "store", "dst": "load", "distance": 1},
    ],
}
_PIPELINE_DURATIONS = {
    "durations_ns": {"load": 100.0, "mma": 200.0, "store": 50.0},
    "source": "benchmark-public-fixture",
}
_FUSION_DAG = {
    "name": "benchmark-producer-consumer-cast",
    "nodes": [
        {"id": "producer", "kind": "matmul", "duration_us": 10.0},
        {"id": "consumer", "kind": "aiv_cast", "duration_us": 5.0},
        {"id": "tail", "kind": "attention", "duration_us": 20.0},
    ],
    "edges": [
        {
            "src": "producer",
            "dst": "consumer",
            "tensor": {"bytes": 1 * MIB, "dtype": "float32", "layout": "ND"},
        },
        {"src": "consumer", "dst": "tail"},
    ],
    "capacities_kib": {"L1": 512, "L0A": 64, "L0B": 64, "L0C": 256, "UB": 248},
}
_FUSION_CANDIDATE = {
    "name": "elide-cast",
    "producer": "producer",
    "consumer": "consumer",
    "remove_nodes": ["consumer"],
    "remove_edges": [{"src": "producer", "dst": "consumer"}],
    "live_buffers": [{"name": "UB scratch", "buffer": "UB", "size_kib": 128}],
}

_A3_CHANNELS: Mapping[str, Any] | None = None


def _a3_channels() -> Mapping[str, Any]:
    global _A3_CHANNELS
    if _A3_CHANNELS is None:
        profile = resolve_device_capability_profile(soc=A3_SKU)
        _A3_CHANNELS = profile.memory_hierarchy_spec.transfer_channels
    return _A3_CHANNELS


def _close(observed: float, expected: float, *, rel: float = 1e-9) -> bool:
    return math.isclose(float(observed), float(expected), rel_tol=rel, abs_tol=1e-12)


def _four_layer_workload() -> dict[str, Any]:
    """One invocation of the Layer 1-4 roofline/boundary/subgraph path."""
    channels = _a3_channels()
    m = n = k = 4096
    return {
        "hap": evaluate_operator_hap(
            A3_SKU,
            2 * m * n * k,
            (m * k + k * n + m * n) * 2,
            600.0,
            1000.0,
        ),
        "pipeline": solve_periodic_pipeline(
            _PIPELINE_SPEC, _PIPELINE_DURATIONS, total_tiles=10,
        ),
        "fusion": analyze_subgraph_fusion(_FUSION_DAG, _FUSION_CANDIDATE),
        "transfer": solve_transfer_bound("d2d_intra_node", 64 * MIB, channels),
        "collective": solve_collective_bound(
            "all_reduce", 64 * MIB, channels,
            total_ranks=16, ranks_per_node=8, dies_per_package=2,
        ),
    }


def _validate_four_layer(result: Mapping[str, Any]) -> None:
    hap = result["hap"]
    assert hap["status"] == "modeled"
    assert hap["bound_pipe"] == "cube_bound"
    assert _close(hap["t_hw_us"], 508.40048484848484)
    assert _close(hap["hap_score"], 0.8429360983266381)

    pipeline = result["pipeline"]
    assert pipeline["status"] == "modeled"
    assert _close(pipeline["minimum_period_ns"], 350.0)
    assert _close(pipeline["minimum_tile_ii_ns"], 350.0)
    assert "M" in pipeline["bottleneck_pipes"]
    assert _close(pipeline["critical_cycles"][0]["cycle_mean_ns"], 350.0)

    fusion = result["fusion"]
    assert fusion["status"] == "modeled"
    assert fusion["feasibility"]["failures"] == []
    assert _close(fusion["removable_gm_bytes"], 2 * MIB)
    assert _close(fusion["latency"]["critical_path_saving_us"], 5.0)

    transfer = result["transfer"]
    assert transfer["status"] == "modeled"
    assert transfer["flags"] == []
    assert _close(transfer["lower_bound_us"], 344.8921632653061)

    collective = result["collective"]
    assert [tier["tier"] for tier in collective["tiers"]] == [
        "d2d_intra_package", "d2d_intra_node", "d2d_inter_node",
    ]
    assert _close(collective["lower_bound_us"], 2216.780863295949)


def _qwen_workload_factory() -> tuple[Callable[[], dict[str, Any]], Callable[[Mapping[str, Any]], None]]:
    fixture = json.loads(QWEN_FIXTURE.read_text(encoding="utf-8"))
    case = fixture["case"]
    artifact = PhysicalCalibration.from_mapping(fixture["physical_calibration"])

    def workload() -> dict[str, Any]:
        return predict(case, physical_calibration=artifact)

    def validate(payload: Mapping[str, Any]) -> None:
        assert payload["status"] == "modeled"
        assert payload["served_requests"] == 1
        assert payload["unserved_requests"] == 0
        assert payload["output_tokens"] == 4
        calibration = payload["physical_calibration"]
        assert calibration["status"] == "configured"
        assert calibration["timing_scope"] == "device_event_warm"
        assert calibration["applied_op_count"] > 0
        assert calibration["unmodeled_op_count"] > 0

        request = payload["requests"][0]
        assert request["status"] == "served"
        assert request["generated_tokens"] == 4
        ttft = float(request["ttft_ms"])
        tpot = float(request["tpot_ms"])
        assert math.isfinite(ttft) and ttft > 0.0
        assert math.isfinite(tpot) and tpot > 0.0
        expected_e2e = ttft + (request["generated_tokens"] - 1) * tpot
        assert _close(request["e2e_ms"], expected_e2e, rel=1e-9)
        metrics = payload["metrics"]
        assert _close(metrics["ttft_ms"]["mean"], ttft)
        assert _close(metrics["tpot_ms"]["mean"], tpot)
        assert len(payload["traces"]) == 4

    return workload, validate


def _operator_workload_factory() -> tuple[Callable[[], dict[str, Any]], Callable[[Mapping[str, Any]], None]]:
    payload = json.loads(OPERATOR_FIXTURE.read_text(encoding="utf-8"))

    def workload() -> dict[str, Any]:
        return operator_plan_report(payload)

    def validate(result: Mapping[str, Any]) -> None:
        assert result["status"] == "modeled"
        assert result["execution"]["status"] == "modeled"
        assert _close(result["elapsed_ms"], result["execution"]["makespan_ms"])
        assert [op["op_id"] for op in result["operators"]] == ["pre", "gemm", "post"]
        assert all(
            op["operator_estimation"]["status"] == "analytic_plan"
            for op in result["operators"]
        )
        assert {
            op["execution_plan"]["backend"] for op in result["operators"]
        } == {"vector_instruction", "gemm_finite_pipeline"}
        assert all(
            op["operator_estimation"]["not_npu_measured"]
            for op in result["operators"]
        )

    return workload, validate


def _measure(
    workload: Callable[[], Mapping[str, Any]],
    validate: Callable[[Mapping[str, Any]], None],
    iterations: int,
) -> float:
    """Assert determinism, then return mean workload latency in us/op."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    initial = workload()
    validate(initial)
    started = time.perf_counter_ns()
    result = initial
    for _ in range(iterations):
        result = workload()
    elapsed_ns = time.perf_counter_ns() - started
    validate(result)
    return elapsed_ns / iterations / 1000.0


def _cases() -> dict[str, dict[str, Any]]:
    qwen_workload, qwen_validate = _qwen_workload_factory()
    operator_workload, operator_validate = _operator_workload_factory()
    return {
        "four-layer": {
            "metric": "four_layer_roofline_boundary_subgraph",
            "iterations": 200,
            "workload": _four_layer_workload,
            "validate": _validate_four_layer,
        },
        "qwen25-7b-physical": {
            "metric": "a3_qwen25_7b_physical_whole_network_ttft_tpot",
            "iterations": 5,
            "workload": qwen_workload,
            "validate": qwen_validate,
        },
        "operator-plan": {
            "metric": "operator_plan_vector_gemm_estimation",
            "iterations": 20,
            "workload": operator_workload,
            "validate": operator_validate,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="all",
        choices=["all", "four-layer", "qwen25-7b-physical", "operator-plan"],
        help="single benchmark case, or all (default)",
    )
    args = parser.parse_args(argv)

    output = os.environ.get("MINDIE_BENCH_RESULT_PATH")
    if not output:
        parser.error("MINDIE_BENCH_RESULT_PATH is required")

    cases = _cases()
    selected = list(cases) if args.case == "all" else [args.case]
    metrics = []
    for case_id in selected:
        case = cases[case_id]
        value_us = _measure(
            case["workload"], case["validate"], int(case["iterations"]),
        )
        metrics.append({
            "name": case["metric"],
            "unit": "us/op",
            "value": value_us,
            "iterations": int(case["iterations"]),
        })

    Path(output).write_text(
        json.dumps({"status": "pass", "metrics": metrics}) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
