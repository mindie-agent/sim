"""Four-level feature-track waterfall comparison and single-PR What-If attribution.

[Role]
  Evaluate ``(model, hardware, parallel)`` across all three software states
  (``merged`` -> ``unmerged_pr`` -> ``expected_ideal``) plus the 4th hardware
  roofline anchor (``0.8x target`` & ``1.0x physical ideal``) on the single
  canonical simulation spine.

[Invariants]
  * Every level executes through the exact same ``compose_case`` + ``body_forward``
    / ``analyze_network_efficiency`` / ``evaluate_target_efficiency`` pipeline.
  * Monotonicity (``merged >= unmerged_pr >= expected_ideal >= physical_1p0x``)
    is audited explicitly within floating-point tolerance ``1e-6`` ms.
  * Single-PR ablation runs each ``unmerged_pr`` mechanism isolated on top of
    ``merged`` so engineers see exact per-PR ``delta_ms`` and ``delta_pct`` ROI.

[Timeouts & Locks]
  Bounded by 3 + |PRs| deterministic graph evaluations; no I/O or locks.

[Side-Effect Policy]
  Pure in-memory evaluation returning JSON-serializable dictionaries.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.compose import compose_case
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency,
    evaluate_target_efficiency,
)

WATERFALL_SCHEMA = "mindie-sim.feature-waterfall/1"
MONOTONIC_TOLERANCE_MS = 1e-5
DEFAULT_WATERFALL_CONTEXT = 4096
DEFAULT_WATERFALL_SEQUENCES = 8
DEFAULT_ROOFLINE_MBU = 0.80
DEFAULT_ROOFLINE_CUBE_MFU = 0.75
DEFAULT_ROOFLINE_VEC_MFU = 0.65
MS_PRECISION_DIGITS = 6
PCT_PRECISION_DIGITS = 3
RATIO_PRECISION_DIGITS = 4


def _pct_delta(before_ms: float, after_ms: float) -> float:
    if before_ms <= 0.0:
        return 0.0
    return ((after_ms - before_ms) / before_ms) * 100.0


def _speedup(before_ms: float, after_ms: float) -> float:
    if after_ms <= 0.0:
        return 1.0
    return before_ms / after_ms


def _network(analysis: Mapping[str, Any]) -> dict[str, float]:
    """Nominal network efficiency of one level, never a fitted target."""
    return {
        "mbu": round(float(analysis["network_mbu"]), RATIO_PRECISION_DIGITS),
        "cube_mfu": round(float(analysis["network_cube_mfu"]), RATIO_PRECISION_DIGITS),
        "vec_mfu": round(float(analysis["network_vec_mfu"]), RATIO_PRECISION_DIGITS),
        "link_util": round(float(analysis["network_link_util"]), RATIO_PRECISION_DIGITS),
        "roofline_efficiency": round(
            float(analysis["network_roofline_efficiency"]), RATIO_PRECISION_DIGITS),
    }


def compare_feature_tracks(
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    *,
    runtime_stack: str | Mapping[str, Any] = "vllm-ascend",
    phase: str = "decode",
    tokens: int = 1,
    context: int = DEFAULT_WATERFALL_CONTEXT,
    sequences: int = DEFAULT_WATERFALL_SEQUENCES,
    scope: str = "body",
    target_mbu: float = DEFAULT_ROOFLINE_MBU,
    target_cube_mfu: float = DEFAULT_ROOFLINE_CUBE_MFU,
    target_vec_mfu: float = DEFAULT_ROOFLINE_VEC_MFU,
    ablate_prs: bool = True,
    **compose_kwargs: Any,
) -> dict[str, Any]:
    """Return the complete 4-level software-to-silicon performance waterfall."""
    cleaned_kwargs = dict(compose_kwargs)
    cleaned_kwargs.pop("feature_track", None)
    cleaned_kwargs.pop("optimizations", None)
    cleaned_kwargs.pop("exclude_optimizations", None)

    cases: dict[str, dict[str, Any]] = {}
    analyses: dict[str, dict[str, Any]] = {}
    for track in ("merged", "unmerged_pr", "expected_ideal"):
        case = compose_case(
            model, hardware=hardware, runtime_stack=runtime_stack,
            feature_track=track, **cleaned_kwargs,
        )
        cases[track] = case
        analyses[track] = analyze_network_efficiency(
            case, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope,
        )

    roofline_eval = evaluate_target_efficiency(
        cases["expected_ideal"],
        phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope,
        target_mbu=target_mbu,
        target_cube_mfu=target_cube_mfu,
        target_vec_mfu=target_vec_mfu,
        baseline_analysis=analyses["expected_ideal"],
    )

    merged_ms = float(analyses["merged"]["graph_duration_ms"])
    pr_ms = float(analyses["unmerged_pr"]["graph_duration_ms"])
    ideal_ms = float(analyses["expected_ideal"]["graph_duration_ms"])
    roof_0p8_ms = float(roofline_eval["target_step_duration_ms"])
    roof_1p0_ms = float(roofline_eval["physical_ideal_ms"])

    merged_report = cases["merged"]["runtime"]["metadata"]["feature_track_report"]
    pr_report = cases["unmerged_pr"]["runtime"]["metadata"]["feature_track_report"]
    ideal_report = cases["expected_ideal"]["runtime"]["metadata"]["feature_track_report"]

    pr_ablations: list[dict[str, Any]] = []
    if ablate_prs:
        for pr_item in pr_report["unmerged_pr_mechanisms"]:
            mid = pr_item["mechanism_id"]
            single_case = compose_case(
                model, hardware=hardware, runtime_stack=runtime_stack,
                feature_track="merged", optimizations=(mid,), **cleaned_kwargs,
            )
            single_eff = analyze_network_efficiency(
                single_case, phase=phase, tokens=tokens, context=context,
                sequences=sequences, scope=scope,
            )
            single_ms = float(single_eff["graph_duration_ms"])
            pr_ablations.append({
                "mechanism_id": mid,
                "ref": pr_item["ref"],
                "notes": pr_item["notes"],
                "step_ms": round(single_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_ms": round(single_ms - merged_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_pct": round(_pct_delta(merged_ms, single_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_merged": round(_speedup(merged_ms, single_ms), RATIO_PRECISION_DIGITS),
            })
        pr_ablations.sort(key=lambda row: row["delta_vs_merged_ms"])

    monotonic = (
        merged_ms + MONOTONIC_TOLERANCE_MS >= pr_ms
        and pr_ms + MONOTONIC_TOLERANCE_MS >= ideal_ms
        and ideal_ms + MONOTONIC_TOLERANCE_MS >= roof_1p0_ms
    )

    return {
        "schema": WATERFALL_SCHEMA,
        "model": cases["merged"]["model"]["name"],
        "hardware_sku": cases["merged"]["machine"]["sku"],
        "runtime_stack": merged_report["stack"],
        "repository": merged_report["repository"],
        "geometry": {
            "phase": phase, "tokens": tokens,
            "context": context, "sequences": sequences, "scope": scope,
        },
        "monotonic_verified": bool(monotonic),
        "waterfall": {
            "current_merged": {
                "step_ms": round(merged_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["merged"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["merged"]["total_hbm_bytes"]),
                "network": _network(analyses["merged"]),
                "active_mechanisms": [m["id"] for m in merged_report["active_mechanisms"]],
            },
            "current_unmerged_pr": {
                "step_ms": round(pr_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["unmerged_pr"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["unmerged_pr"]["total_hbm_bytes"]),
                "network": _network(analyses["unmerged_pr"]),
                "delta_vs_merged_ms": round(pr_ms - merged_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_pct": round(_pct_delta(merged_ms, pr_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_merged": round(_speedup(merged_ms, pr_ms), RATIO_PRECISION_DIGITS),
                "active_open_prs": [
                    {"mechanism_id": m["mechanism_id"], "ref": m["ref"], "notes": m["notes"]}
                    for m in pr_report["unmerged_pr_mechanisms"]
                ],
                "single_pr_ablation": pr_ablations,
            },
            "expected_ideal": {
                "step_ms": round(ideal_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["expected_ideal"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["expected_ideal"]["total_hbm_bytes"]),
                "network": _network(analyses["expected_ideal"]),
                "delta_vs_unmerged_pr_ms": round(ideal_ms - pr_ms, MS_PRECISION_DIGITS),
                "delta_vs_unmerged_pr_pct": round(_pct_delta(pr_ms, ideal_ms), PCT_PRECISION_DIGITS),
                "delta_vs_merged_ms": round(ideal_ms - merged_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_pct": round(_pct_delta(merged_ms, ideal_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_merged": round(_speedup(merged_ms, ideal_ms), RATIO_PRECISION_DIGITS),
                "unimplemented_architectural_gaps": ideal_report["unimplemented_gaps"],
                "active_mechanisms": [m["id"] for m in ideal_report["active_mechanisms"]],
            },
            "hardware_roofline": {
                "critical_path_0p8x_ms": round(roof_0p8_ms, MS_PRECISION_DIGITS),
                "physical_ideal_1p0x_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
                "remaining_kernel_micro_gap_to_0p8x_ms": round(max(0.0, ideal_ms - roof_0p8_ms), MS_PRECISION_DIGITS),
                "remaining_kernel_micro_gap_to_1p0x_ms": round(max(0.0, ideal_ms - roof_1p0_ms), MS_PRECISION_DIGITS),
                "targets": {
                    "mbu": target_mbu,
                    "cube_mfu": target_cube_mfu,
                    "vec_mfu": target_vec_mfu,
                },
            },
        },
        "efficiency_reference": {
            "target_step_duration_ms": round(roof_0p8_ms, MS_PRECISION_DIGITS),
            "physical_ideal_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
            "target_step_basis": roofline_eval["target_step_basis"],
        },
    }
