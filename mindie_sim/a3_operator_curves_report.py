"""Coverage and markdown rendering for operator-campaign analysis.

[Role]
  Render human-readable analysis and explicit missing/pending coverage without
  pooling measurements.

[Invariants]
  * Missing/pending/failed cases stay visible with reasons.
  * Markdown is a secondary view; JSON is authoritative.
  * No latency average across groups is introduced here.

[Timeouts & Locks]
  Pure local formatting; no I/O, locks or devices.

[Side-Effect Policy]
  No writes; callers write explicit paths.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

def _coverage(manifest: Mapping[str, Any], analyses: Sequence[Mapping[str, Any]],
              unmeasured: Sequence[Mapping[str, Any]], counts: Mapping[str, int],
              unselected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inventory = manifest.get("inventory", ())
    statuses: dict[str, int] = {}
    for item in inventory:
        status = str(item.get("collection_status"))
        statuses[status] = statuses.get(status, 0) + 1
    selected_models = {
        model: model_data.get("selected_points", 0)
        for model, model_data in (manifest.get("models") or {}).items()
    }
    return {
        "inventory_entries": len(inventory),
        "inventory_status_counts": dict(sorted(statuses.items())),
        "selected_requests": len(manifest.get("requests", ())),
        "selected_train": manifest.get("coverage", {}).get("selected_train"),
        "selected_heldout": manifest.get("coverage", {}).get("selected_heldout"),
        "model_selected_points": selected_models,
        "measured_groups": len(analyses),
        "unmeasured_results": len(unmeasured),
        "selected_without_result": len(unselected),
        "pending_conditions_in_manifest": manifest.get("coverage", {}).get("pending_conditions"),
        "pending_groups_in_manifest": len(manifest.get("pending_requests", ())),
        "derivation_failures": manifest.get("coverage", {}).get("derivation_failures", []),
        "missing": [
            {
                "condition_key": item.get("condition_key"),
                "family": item.get("family"),
                "recipe": item.get("recipe"),
                "reason": item.get("reason"),
                "required_backend": item.get("required_backend"),
            }
            for item in manifest.get("pending_requests", ())
        ] + [
            {
                "condition_key": None,
                "family": item.get("family"),
                "recipe": item.get("recipe"),
                "reason": item.get("collection_status"),
                "required_backend": ", ".join(item.get("required_backends") or ()),
            }
            for item in inventory
            if item.get("collection_status") in {
                "executable_not_selected_budget", "unsupported", "pending_backend",
            }
        ] + [
            {
                "condition_key": item.get("condition_key"),
                "status": item.get("status"),
                "reason": item.get("reason"),
                "required_backend": item.get("required_backend"),
            }
            for item in unmeasured
        ] + [
            {
                "condition_key": None,
                "family": item.get("model"),
                "recipe": item.get("topology"),
                "reason": item.get("reason"),
                "required_backend": "fix extraction/derivation before coverage",
            }
            for item in manifest.get("coverage", {}).get("derivation_failures", ())
        ],
        "model_selected_points_note": "selected_points counts distinct measurement requests whose lineage includes that model; a deduplicated physical request may cover several models",
        "result_status_counts": dict(counts),
        "pending_results": counts.get("pending", 0),
        "failed_results": counts.get("failed", 0),
        "no_silent_zero": True,
    }


def _markdown(analysis: Mapping[str, Any]) -> str:
    summary = analysis["summary"]
    coverage = analysis["coverage"]
    lines = [
        "# Operator shape-linearity analysis",
        "",
        f"Campaign: `{analysis.get('campaign_id')}`",
        f"Status: `{analysis.get('status')}`",
        f"Groups: {summary['groups']}; completed/pending/failed results: "
        f"{summary['completed_results']}/{summary['pending_results']}/{summary['failed_results']}",
        f"Raw measured samples: {summary['raw_measured_samples']}",
        "",
        "## Coverage",
        "",
        f"- Inventory entries: {coverage['inventory_entries']}",
        f"- Selected requests: {coverage['selected_requests']} "
        f"(train {coverage['selected_train']}, heldout {coverage['selected_heldout']})",
        f"- Measured groups: {coverage['measured_groups']}",
        f"- Unmeasured results: {coverage['unmeasured_results']}; selected without result: "
        f"{coverage['selected_without_result']}",
        f"- Pending conditions in manifest: {coverage['pending_conditions_in_manifest']}",
        "",
        "## Train-only regimes",
        "",
    ]
    if not analysis["analyses"]:
        lines.append("No completed measured samples. Dry-run data is not a measurement.")
    for group in analysis["analyses"][:80]:
        fit = group.get("fit")
        lines.append(
            f"### `{group['condition_key']}` axis `{group['axis']}` "
            f"({group.get('execution_kind')}, {group.get('observed_api')})"
        )
        lines.append(
            f"- independent train/heldout points: {group['independent_train_points']}/"
            f"{group['independent_heldout_points']}; trials: "
            f"{group['independent_train_trials']}/{group['independent_heldout_trials']}"
        )
        lines.append(
            f"- train median range: "
            f"{min((p['median_ms'] for p in group['train_points']), default=0):.6g} .. "
            f"{max((p['median_ms'] for p in group['train_points']), default=0):.6g} ms"
        )
        if fit:
            for segment in fit["segments"]:
                lines.append(
                    f"- segment x=[{segment['x_min']:g},{segment['x_max']:g}] "
                    f"slope={segment['slope_ms_per_unit']:.6g} ms/unit, "
                    f"intercept={segment['intercept_ms']:.6g} ms, n={segment['n']}"
                )
            if fit.get("transition_brackets"):
                lines.append(f"- transition brackets: {fit['transition_brackets']}")
        if group.get("launch_floor_hint", {}).get("hint"):
            lines.append(f"- launch/constant-cost hint: {group['launch_floor_hint']['note']}")
        if group.get("saturation_hint", {}).get("hint"):
            lines.append(f"- saturation/plateau hint: {group['saturation_hint']['note']}")
        heldout = group.get("heldout_evaluation", {})
        if heldout.get("count") and "median_ape" in heldout:
            lines.append(
                f"- heldout APE median/p90/max: {heldout['median_ape']:.3%}/"
                f"{heldout['p90_ape']:.3%}/{heldout['max_ape']:.3%}; "
                f"interp/extrap: {heldout['interpolation_count']}/{heldout['extrapolation_count']}"
            )
        elif heldout.get("status"):
            lines.append(f"- heldout: {heldout['status']} (count {heldout.get('count', 0)})")
        if group.get("split_overlap_x"):
            lines.append(f"- rejected split overlap at x={group['split_overlap_x']}")
        lines.append("")
    lines += ["## Missing/pending", ""]
    if not coverage["missing"]:
        lines.append("None.")
    else:
        for item in coverage["missing"][:80]:
            lines.append(
                f"- `{item.get('family')}/{item.get('recipe')}` {item.get('reason')} "
                f"Required: {item.get('required_backend')}"
            )
        if len(coverage["missing"]) > 80:
            lines.append(f"- ... {len(coverage['missing']) - 80} more in analysis.json")
    lines += [
        "", "## Scope", "",
        "- Piecewise fits use train points only; heldout rows are evaluation-only and root owns thresholds.",
        "- Affine intercepts are constant-cost hints, not proven launch latency; plateaus are not proven saturation.",
        "- No pool across dtype/layout/recipe/backend/protocol; repeats do not increase independent N.",
        "- No A5 collection is performed or fit here.",
        "",
    ]
    return "\n".join(lines)
