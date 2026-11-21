"""Four-rung feature-track waterfall comparison and PR What-If attribution.

[Role]
  Evaluate ``(model, hardware, parallel)`` across all four software states
  (``raw_eager`` -> ``merged`` -> ``unmerged_pr`` -> ``expected_ideal``) plus
  the 1.0x physical roofline from the realized occurrence DAG critical path.

[Invariants]
  * Every level executes through the exact same ``compose_case`` +
    ``analyze_network_efficiency`` pipeline on identical checkpoint quantization.
  * Monotonicity (``raw_eager >= merged >= unmerged_pr >= expected_ideal >= physical_1p0x``)
    is audited explicitly within floating-point tolerance ``1e-5`` ms.
  * PR ablation reports both single-add (``merged + {PR}``) and leave-one-out
    (``unmerged_pr - {PR}``) attribution.

[Timeouts & Locks]
  Bounded by 4 + 2*|unique PR mechanisms| deterministic graph evaluations; no I/O.

[Side-Effect Policy]
  Pure in-memory evaluation returning JSON-serializable dictionaries.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.compose import compose_case
from mindie_sim.prediction.efficiency import analyze_network_efficiency

WATERFALL_SCHEMA = "mindie-sim.feature-waterfall/1"
MONOTONIC_TOLERANCE_MS = 1e-5
DEFAULT_WATERFALL_CONTEXT = 4096
DEFAULT_WATERFALL_SEQUENCES = 8
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
    ablate_prs: bool = True,
    **compose_kwargs: Any,
) -> dict[str, Any]:
    """Return the complete 4-rung software-to-silicon performance waterfall."""
    cleaned_kwargs = dict(compose_kwargs)
    cleaned_kwargs.pop("feature_track", None)
    cleaned_kwargs.pop("optimizations", None)
    cleaned_kwargs.pop("exclude_optimizations", None)
    cleaned_kwargs.pop("raw_eager", None)
    cleaned_kwargs.pop("_stack_phase", None)

    cases: dict[str, dict[str, Any]] = {}
    analyses: dict[str, dict[str, Any]] = {}
    for track in ("merged", "unmerged_pr", "expected_ideal"):
        case = compose_case(
            model, hardware=hardware, runtime_stack=runtime_stack,
            feature_track=track, _stack_phase=phase, **cleaned_kwargs,
        )
        cases[track] = case
        analyses[track] = analyze_network_efficiency(
            case, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope,
        )

    eager_kwargs = dict(cleaned_kwargs)
    if "quant" not in eager_kwargs:
        va_prof = (cases["merged"]["runtime"]["metadata"] or {}).get("va_profile") or {}
        if va_prof.get("quant") is not None:
            eager_kwargs["quant"] = va_prof["quant"]
    cases["raw_eager"] = compose_case(
        model, hardware=hardware, runtime_stack=runtime_stack,
        feature_track="raw_eager", _stack_phase=phase, **eager_kwargs,
    )
    analyses["raw_eager"] = analyze_network_efficiency(
        cases["raw_eager"], phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope,
    )

    eager_ms = float(analyses["raw_eager"]["graph_duration_ms"])
    merged_ms = float(analyses["merged"]["graph_duration_ms"])
    pr_ms = float(analyses["unmerged_pr"]["graph_duration_ms"])
    ideal_ms = float(analyses["expected_ideal"]["graph_duration_ms"])
    ideal_eff = analyses["expected_ideal"]
    roof_1p0_ms = max(
        float(ideal_eff.get("critical_path_roofline_ms") or 0.0),
        float(ideal_eff.get("physical_resource_cut_ms") or 0.0),
    ) or float(ideal_eff.get("total_roofline_ideal_ms") or 0.0)

    eager_report = cases["raw_eager"]["runtime"]["metadata"]["feature_track_report"]
    merged_report = cases["merged"]["runtime"]["metadata"]["feature_track_report"]
    pr_report = cases["unmerged_pr"]["runtime"]["metadata"]["feature_track_report"]
    ideal_report = cases["expected_ideal"]["runtime"]["metadata"]["feature_track_report"]

    pr_ablations: list[dict[str, Any]] = []
    loo_ablations: list[dict[str, Any]] = []
    if ablate_prs:
        single_cache: dict[str, float] = {}
        loo_cache: dict[str, float] = {}
        for pr_item in pr_report["unmerged_pr_mechanisms"]:
            mid = pr_item["mechanism_id"]
            if mid not in single_cache:
                single_case = compose_case(
                    model, hardware=hardware, runtime_stack=runtime_stack,
                    feature_track="merged", optimizations=(mid,),
                    _stack_phase=phase, **cleaned_kwargs,
                )
                single_eff = analyze_network_efficiency(
                    single_case, phase=phase, tokens=tokens, context=context,
                    sequences=sequences, scope=scope,
                )
                single_cache[mid] = float(single_eff["graph_duration_ms"])
            if mid not in loo_cache:
                loo_case = compose_case(
                    model, hardware=hardware, runtime_stack=runtime_stack,
                    feature_track="unmerged_pr", exclude_optimizations=(mid,),
                    _stack_phase=phase, **cleaned_kwargs,
                )
                loo_eff = analyze_network_efficiency(
                    loo_case, phase=phase, tokens=tokens, context=context,
                    sequences=sequences, scope=scope,
                )
                loo_cache[mid] = float(loo_eff["graph_duration_ms"])
            single_ms = single_cache[mid]
            without_ms = loo_cache[mid]
            pr_ablations.append({
                "mechanism_id": mid,
                "ref": pr_item["ref"],
                "notes": pr_item["notes"],
                "step_ms": round(single_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_ms": round(single_ms - merged_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_pct": round(_pct_delta(merged_ms, single_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_merged": round(_speedup(merged_ms, single_ms), RATIO_PRECISION_DIGITS),
            })
            loo_ablations.append({
                "mechanism_id": mid,
                "ref": pr_item["ref"],
                "notes": pr_item["notes"],
                "without_pr_step_ms": round(without_ms, MS_PRECISION_DIGITS),
                "delta_vs_without_pr_ms": round(pr_ms - without_ms, MS_PRECISION_DIGITS),
                "delta_vs_without_pr_pct": round(_pct_delta(without_ms, pr_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_without_pr": round(_speedup(without_ms, pr_ms), RATIO_PRECISION_DIGITS),
            })
        pr_ablations.sort(key=lambda row: row["delta_vs_merged_ms"])
        loo_ablations.sort(key=lambda row: row["delta_vs_without_pr_ms"])

    monotonic = (
        merged_ms + MONOTONIC_TOLERANCE_MS >= pr_ms
        and pr_ms + MONOTONIC_TOLERANCE_MS >= ideal_ms
        and ideal_ms + MONOTONIC_TOLERANCE_MS >= roof_1p0_ms
    )
    eager_monotonic = eager_ms + MONOTONIC_TOLERANCE_MS >= merged_ms

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
        "eager_vs_merged_monotonic": bool(eager_monotonic),
        "waterfall": {
            "raw_eager": {
                "step_ms": round(eager_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["raw_eager"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["raw_eager"]["total_hbm_bytes"]),
                "network": _network(analyses["raw_eager"]),
                "active_mechanisms": [m["id"] for m in eager_report["active_mechanisms"]],
            },
            "current_merged": {
                "step_ms": round(merged_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["merged"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["merged"]["total_hbm_bytes"]),
                "network": _network(analyses["merged"]),
                "delta_vs_raw_eager_ms": round(merged_ms - eager_ms, MS_PRECISION_DIGITS),
                "delta_vs_raw_eager_pct": round(_pct_delta(eager_ms, merged_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_raw_eager": round(_speedup(eager_ms, merged_ms), RATIO_PRECISION_DIGITS),
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
                "leave_one_out_ablation": loo_ablations,
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
                "physical_ideal_1p0x_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
                "remaining_kernel_micro_gap_to_1p0x_ms": round(max(0.0, ideal_ms - roof_1p0_ms), MS_PRECISION_DIGITS),
            },
        },
        "efficiency_reference": {
            "physical_ideal_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
            "basis": "physical_1p0x_critical_path_roofline",
        },
    }
