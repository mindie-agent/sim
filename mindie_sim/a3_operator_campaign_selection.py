"""Budget-aware condition selection and inventory aggregation.

[Role]
  Choose model/inventory anchors and extra axis points within the explicit
  budget and aggregate every emitted family/recipe into collection status.

[Invariants]
  * Pending and reference variants remain distinct.
  * Budget shortfalls are explicit; selected points never exceed the cap after
    post-dedup truncation.
  * A physical point cannot become holdout if it was train anywhere.

[Timeouts & Locks]
  Pure bounded arithmetic; no I/O, locks or devices.

[Side-Effect Policy]
  No filesystem/network/device mutation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .a3_operator_campaign_conditions import _canonical
from .a3_operator_campaign_sweeps import _axes_for_kind, _point_priority

def _inventory(conditions: Mapping[str, Mapping[str, Any]], models: Mapping[str, Mapping[str, Any]],
               selected_points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected_by_condition: dict[str, int] = {}
    selected_by_point_condition: dict[str, set[str]] = {}
    selected_by_variant: dict[tuple[str, str, str], int] = {}
    for point in selected_points:
        for key in point.get("condition_keys", ()):
            selected_by_condition[key] = selected_by_condition.get(key, 0) + 1
        variant = (point.get("family", ""), point.get("recipe", ""), point.get("execution_kind", ""))
        selected_by_variant[variant] = selected_by_variant.get(variant, 0) + 1
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for condition in conditions.values():
        key = (condition["family"], condition["recipe"])
        entry = groups.setdefault(key, {
            "family": condition["family"], "recipe": condition["recipe"],
            "models": set(), "phases": set(), "shape_kinds": set(),
            "dtypes": set(), "layouts": set(), "quantizations": set(),
            "condition_count": 0, "source_case_count": 0,
            "dimension_samples": [], "execution_variants": {},
        })
        entry["condition_count"] += 1
        entry["phases"].add(condition["phase"])
        entry["shape_kinds"].add(condition["shape_kind"])
        entry["dtypes"].add(str(condition["precision"].get("input_dtype")))
        entry["layouts"].add(str(condition["layout"]))
        entry["quantizations"].add(str(condition["quantization"]))
        origins = list(condition["sources"].values())
        entry["source_case_count"] += sum(len(origin.get("layers", ())) or 1 for origin in origins)
        for origin in origins:
            entry["models"].add(origin["model"])
        if len(entry["dimension_samples"]) < 12:
            sample = {
                "model": origins[0]["model"] if origins else "",
                "phase": condition["phase"], "shape_kind": condition["shape_kind"],
                "shape": condition["shape"], "semantic_shape": condition["semantic_shape"],
                "launch_shape": condition["launch_shape"], "padded_shape": condition["padded_shape"],
                "input_dtype": condition["precision"].get("input_dtype"),
                "weight_dtype": condition["precision"].get("weight_dtype"),
                "compute_dtype": condition["precision"].get("compute_dtype"),
                "layout": condition["layout"], "quantization": condition["quantization"],
                "execution_kind": condition["execution_kind"],
            }
            if sample not in entry["dimension_samples"]:
                entry["dimension_samples"].append(sample)
        vkey = (condition["execution_kind"], str(condition["worker_recipe"]),
                str(condition["backend"]))
        variant = entry["execution_variants"].setdefault(_canonical(vkey), {
            "execution_kind": condition["execution_kind"],
            "worker_recipe": condition["worker_recipe"],
            "backend": condition["backend"],
            "target_match": condition["target_match"],
            "status": condition["status"],
            "reason": condition["reason"],
            "required_backend": condition["required_backend"],
            "condition_count": 0,
            "selected_points": 0,
        })
        variant["condition_count"] += 1
        variant["selected_points"] += selected_by_condition.get(condition["condition_key"], 0)
    result = []
    for key in sorted(groups):
        entry = groups[key]
        variants = [entry["execution_variants"][vkey] for vkey in sorted(entry["execution_variants"])]
        selected = sum(item["selected_points"] for item in variants)
        pending = [item for item in variants if item["status"] == "pending_backend"]
        executable = [item for item in variants if item["status"] in {"executable", "conditional_backend"}]
        if selected:
            status = "selected" if not pending else "selected_with_pending_target"
        elif pending and not executable:
            status = "pending_backend"
        elif executable:
            status = "executable_not_selected_budget"
        else:
            status = "unsupported"
        result.append({
            "family": entry["family"], "recipe": entry["recipe"],
            "models": sorted(entry["models"]),
            "model_count": len(entry["models"]),
            "phases": sorted(entry["phases"]),
            "shape_kinds": sorted(entry["shape_kinds"]),
            "dtypes": sorted(entry["dtypes"]), "layouts": sorted(entry["layouts"]),
            "quantizations": sorted(entry["quantizations"]),
            "condition_count": entry["condition_count"],
            "source_case_count": entry["source_case_count"],
            "dimension_samples": entry["dimension_samples"],
            "execution_variants": variants,
            "selected_points": selected,
            "collection_status": status,
            "pending_variants": len(pending),
            "required_backends": sorted({str(item["required_backend"]) for item in pending if item["required_backend"]}),
        })
    return result


def _choose_condition_points(condition: Mapping[str, Any],
                            points: Sequence[Mapping[str, Any]], *,
                            max_points: int, axes_limit: int) -> list[dict[str, Any]]:
    """Pick a balanced multi-point subset: actual + train/heldout per axis."""
    by_axis: dict[str, list[Mapping[str, Any]]] = {}
    for point in points:
        by_axis.setdefault(str(point["axis"]), []).append(point)
    axes = [axis for axis in _axes_for_kind(condition["shape_kind"]) if axis in by_axis]
    actual = next((point for point in points if point.get("actual")), None)
    chosen: dict[str, dict[str, Any]] = {}
    def add(point: Mapping[str, Any]) -> None:
        chosen.setdefault(str(point["point_key"]), dict(point))
    if actual is not None:
        add(actual)
    def distance(point: Mapping[str, Any]) -> float:
        base = int(condition["shape"].get(str(point["axis"]), 1))
        return abs(math.log((int(point["value"]) + 1.0) / (base + 1.0)))
    for axis in axes[:max(1, axes_limit)]:
        axis_points = sorted(by_axis[axis], key=lambda item: int(item["value"]))
        train = [item for item in axis_points if item.get("split") == "train" and not item.get("actual")]
        held = [item for item in axis_points if item.get("split") == "heldout"]
        interp = [item for item in held if not item.get("extrapolation")]
        outer = [item for item in held if item.get("extrapolation")]
        # Interleave broad train anchors, interpolation holdouts and the
        # outer extrapolation so selection cannot fill every slot with train.
        ordered_groups = (
            sorted(train, key=distance, reverse=True),
            sorted(interp, key=distance, reverse=True),
            sorted(outer, key=distance, reverse=True),
        )
        for index in range(max(len(group) for group in ordered_groups)):
            for group in ordered_groups:
                if index < len(group) and len(chosen) < max_points:
                    add(group[index])
    # Fill any remaining budget with the nearest legal train points.
    if len(chosen) < max_points:
        for item in sorted(points, key=lambda p: (
            p.get("split") != "train", p.get("extrapolation"),
            -_point_priority(p), distance(p), int(p.get("value", 0)),
        )):
            if len(chosen) >= max_points:
                break
            add(item)
    return list(chosen.values())


def _select_conditions(conditions: Mapping[str, Mapping[str, Any]],
                       points_by_condition: Mapping[str, list[dict[str, Any]]],
                       config: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def executable(condition: Mapping[str, Any]) -> bool:
        return condition["status"] in {"executable", "conditional_backend"} and bool(
            condition.get("worker_recipe")
        )

    def score(condition: Mapping[str, Any]) -> tuple[float, str]:
        family = condition["family"]
        base = {
            "gemm": 100.0, "attention_qk": 95.0, "attention_pv": 95.0,
            "attention_softmax": 85.0, "rmsnorm": 80.0, "swiglu": 75.0,
            "elementwise": 55.0, "cache_store": 45.0,
        }.get(family, 20.0)
        origins = list(condition["sources"].values())
        base += min(20.0, math.log2(max(1, len(origins)) + 1) * 4)
        base += min(15.0, math.log2(max(1, sum(len(o.get("layers", ())) or 1 for o in origins)) + 1))
        if condition["execution_kind"] == "reference_decomposition":
            base -= 10.0
        return (base, condition["condition_key"])

    selected: dict[str, dict[str, Any]] = {}
    selected_conditions: set[str] = set()
    covered_models: set[str] = set()
    covered_groups: set[tuple[str, str, str]] = set()

    def add_condition(condition: Mapping[str, Any], *, max_points: int,
                      axes_limit: int, reason: str) -> int:
        candidates = points_by_condition.get(condition["condition_key"], [])
        if not candidates:
            return 0
        chosen = _choose_condition_points(
            condition, candidates, max_points=min(max_points, config.max_points_per_condition),
            axes_limit=axes_limit,
        )
        added = 0
        for point in chosen:
            if len(selected) >= config.budget:
                break
            key = str(point["point_key"])
            if key not in selected:
                selected[key] = {**point, "selection_reason": reason}
                added += 1
            for condition_key in point.get("condition_keys", (condition["condition_key"],)):
                selected_conditions.add(str(condition_key))
        covered_groups.add((condition["family"], condition["recipe"], condition["execution_kind"]))
        for origin in condition["sources"].values():
            covered_models.add(origin["model"])
        return added

    # Phase 1: every model gets a strong GEMM anchor and one non-GEMM anchor.
    for key in config.model_keys:
        model_conditions = [
            cond for cond in conditions.values()
            if executable(cond) and any(o["model"] == key for o in cond["sources"].values())
        ]
        gemm = sorted((c for c in model_conditions if c["family"] == "gemm"), key=score, reverse=True)
        non_gemm = sorted((c for c in model_conditions if c["family"] != "gemm"), key=score, reverse=True)
        for pool, axes_limit in ((gemm[:1], 3), (non_gemm[:1], 2)):
            for condition in pool:
                if len(selected) >= config.budget:
                    break
                add_condition(condition, max_points=6, axes_limit=axes_limit,
                              reason=f"model_anchor:{key}:{condition['family']}")
    # Phase 2: every executable family/recipe/execution group gets an anchor.
    group_best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for condition in conditions.values():
        if not executable(condition):
            continue
        group = (condition["family"], condition["recipe"], condition["execution_kind"])
        if group not in group_best or score(condition) > score(group_best[group]):
            group_best[group] = condition
    for group in sorted(group_best):
        if len(selected) >= config.budget:
            break
        if group in covered_groups:
            continue
        add_condition(group_best[group], max_points=3, axes_limit=1,
                      reason=f"inventory_group:{group[0]}:{group[1]}:{group[2]}")
    # Phase 3: extend already anchored conditions, one point per condition per
    # pass.  Actual train points are always chosen before any holdout.
    anchored = [conditions[key] for key in selected_conditions if key in conditions]
    anchored = [cond for cond in anchored if any(
        point.get("actual") for point in selected.values()
        if any(key == cond["condition_key"] for key in point.get("condition_keys", ()))
    )]
    anchored.sort(key=score, reverse=True)
    while len(selected) < config.budget:
        progressed = False
        for condition in anchored:
            if len(selected) >= config.budget:
                break
            current = {
                str(point["point_key"]) for point in selected.values()
                if any(key == condition["condition_key"] for key in point.get("condition_keys", ()))
            }
            own = [p for p in points_by_condition.get(condition["condition_key"], [])
                   if str(p["point_key"]) not in current]
            if not own or len(current) >= config.max_points_per_condition:
                continue
            own.sort(key=lambda p: (
                -_point_priority(p), int(p["value"]), str(p["point_key"]),
            ))
            point = {**own[0], "selection_reason": "axis_extension"}
            selected[str(point["point_key"])] = point
            for condition_key in point.get("condition_keys", ()):
                selected_conditions.add(str(condition_key))
            progressed = True
        if not progressed:
            break
    ordered = sorted(selected.values(), key=lambda point: (
        -_point_priority(point), point.get("axis", ""), int(point.get("value", 0)),
        str(point["point_key"]),
    ))
    if len(ordered) > config.budget:
        omitted_count = len(ordered) - config.budget
        ordered = ordered[:config.budget]
    else:
        omitted_count = 0
    # A condition selected only through a shared holdout must never survive
    # without its own actual train anchor: drop such orphan heldout points.
    condition_actual: set[str] = set()
    for point in ordered:
        if point.get("actual"):
            condition_actual.update(str(key) for key in point.get("condition_keys", ()))
    filtered = []
    for point in ordered:
        keys = [str(key) for key in point.get("condition_keys", ())]
        if point.get("split") == "heldout" and keys and not any(key in condition_actual for key in keys):
            omitted_count += 1
            continue
        filtered.append(point)
    ordered = filtered
    omitted = {"budget_exhausted": omitted_count} if omitted_count else {}
    for condition_key in sorted(conditions):
        if condition_key in selected_conditions and not any(
            any(key == condition_key for key in point.get("condition_keys", ()))
            for point in ordered
        ):
            omitted[f"condition_not_selected:{conditions[condition_key]['family']}"] = (
                omitted.get(f"condition_not_selected:{conditions[condition_key]['family']}", 0) + 1
            )
    return ordered, {
        "covered_models": sorted(covered_models),
        "covered_groups": sorted([list(group) for group in covered_groups]),
        "omitted": omitted,
    }
