"""Finite legal sweep points, splits, inventory and request materialization.

[Role]
  Turn canonical conditions into legal ladder/boundary/extrapolation points,
  deduplicate physical points globally, precommit train/heldout splits, select
  within the explicit budget, aggregate inventory status and build executable
  measurement requests.

[Invariants]
  * Physical point identity excludes model/layer/family origin.
  * Train anchors win globally; same geometry cannot leak into holdout.
  * Pending conditions materialize explicit points with required backends.
  * No budget shortfall is silently treated as coverage.

[Timeouts & Locks]
  Pure bounded arithmetic; no I/O, locks or devices.

[Side-Effect Policy]
  No filesystem/network/device mutation.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .a3_cost import PredictiveCostModel
from .a3_ops import OpSpec
from .a3_types import A3InputError, MachineSpec, dtype_bytes
from .a3_operator_campaign_conditions import (
    _bytes_for, _canonical, _hash, _origin_summary, _source_summary,
)
from .a3_operator_worker_native import _scaled_expert_counts


def _grouped_histogram(condition: Mapping[str, Any],
                       shape: Mapping[str, int]) -> tuple[list[int], str]:
    """Exact point histogram derived from the actual kernel group."""
    geometry = (condition.get("features") or {}).get("grouped_geometry") or {}
    base = [int(v) for v in (geometry.get("actual_expert_tokens")
                             or (condition.get("features") or {}).get(
                                 "routing", {}).get("local_expert_counts") or ())]
    return _scaled_expert_counts(
        base, int(shape.get("experts", 1)), int(shape.get("routed_tokens", 1)))

def _point_from_shape(condition: Mapping[str, Any], shape: Mapping[str, int], *,
                      axis: str, value: int, reason: str, split_hint: str,
                      split_group: str | None, actual: bool, boundary_probe: bool,
                      extrapolation: bool, config: Any) -> dict[str, Any]:
    kind = condition["shape_kind"]
    semantic = _semantic_for_shape(kind, shape)
    work = _point_work(condition, shape)
    point = {
        "axis": axis, "value": int(value), "reason": reason,
        "shape": dict(shape), "semantic_shape": semantic,
        "launch_shape": semantic, "padded_shape": semantic,
        "actual": bool(actual), "boundary_probe": bool(boundary_probe),
        "extrapolation": bool(extrapolation),
        "split_hint": split_hint, "split_group": split_group,
        "work": work,
        "working_set_bytes": _working_set_bytes(condition, shape),
        "expected_status": condition.get("status", "executable"),
        "required_backend": condition.get("required_backend"),
        "execution_kind": condition.get("execution_kind"),
        "worker_recipe": condition.get("worker_recipe"),
    }
    if kind == "moe_grouped":
        counts, source = _grouped_histogram(condition, shape)
        point["expert_token_counts"] = counts
        point["expert_token_histogram_source"] = source
    point["point_key"] = _point_key(condition, point)
    return point


def _point_key(condition: Mapping[str, Any], point: Mapping[str, Any]) -> str:
    """Physical measurement identity; origin/family metadata is not an axis.

    Two layers/models that execute the same recipe on the same physical shape,
    dtype, layout, quantization and protocol are the same measurement point.
    This is what prevents a cross-model train shape from becoming a heldout
    shape merely because a model or layer name differs.
    """
    precision = condition["precision"]
    identity = {
        "execution_kind": condition["execution_kind"],
        "worker_recipe": condition["worker_recipe"], "backend": condition["backend"],
        "shape": point["shape"],
        "input_dtype": precision.get("input_dtype"),
        "weight_dtype": precision.get("weight_dtype"),
        "output_dtype": precision.get("output_dtype"),
        "compute_dtype": precision.get("compute_dtype"),
        "accumulation_dtype": precision.get("accumulation_dtype"),
        "quantization": condition["quantization"],
        "weight_format": precision.get("weight_format"),
        "weight_representation": precision.get("weight_representation"),
        "weight_block": precision.get("weight_block"),
        "expert_block": precision.get("expert_block"),
        "layout": condition["layout"],
        "protocol": condition["protocol"],
        "expert_token_counts": point.get("expert_token_counts"),
    }
    return _hash("pt:", identity)


def _semantic_for_shape(kind: str, shape: Mapping[str, int]) -> list[int]:
    if kind == "gemm":
        return [int(shape["m"]), int(shape["n"]), int(shape["k"])]
    if kind == "attention":
        return [int(shape["q"]), int(shape["kv"]), int(shape["heads"]), int(shape["dim"])]
    if kind == "attention_softmax":
        if all(key in shape for key in ("q", "heads", "kv")):
            return [int(shape["q"]), int(shape["heads"]), int(shape["kv"])]
        return [int(shape.get("elements", 1))]
    if kind == "router_softmax":
        if all(key in shape for key in ("tokens", "experts")):
            return [int(shape["tokens"]), int(shape["experts"])]
        return [int(shape.get("elements", 1))]
    if kind == "cache":
        return [int(shape["rows"]), int(shape["row_bytes"])]
    if kind == "moe_grouped":
        return [int(shape["experts"]), int(shape["n"]), int(shape["k"])]
    if "tokens" in shape and "width" in shape:
        return [int(shape["tokens"]), int(shape["width"])]
    return [int(shape.get("elements", 1))]


def _point_work(condition: Mapping[str, Any], shape: Mapping[str, int]) -> dict[str, Any]:
    kind = condition["shape_kind"]
    precision = condition["precision"]
    act_bytes = _bytes_for(str(precision.get("input_dtype") or "bf16"))
    out_bytes = _bytes_for(str(precision.get("output_dtype") or precision.get("input_dtype") or "bf16"))
    if kind == "gemm":
        m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
        flops = 2.0 * m * n * k
        declared_weight = float(condition["work"].get("hbm_bytes", 0.0))
        base = condition["work"]
        base_m, base_n, base_k = (int(base["launch_shape"][-3]), int(base["launch_shape"][-2]),
                                  int(base["launch_shape"][-1]))
        weight_bytes = max(0.0, declared_weight - (base_m * base_k + base_m * base_n) * act_bytes)
        weight_bytes *= (n * k) / max(1.0, base_n * base_k)
        hbm = (m * k + m * n) * act_bytes + weight_bytes
        return {"unit": "flops", "flops": flops, "vector_ops": 0.0,
                "hbm_bytes": hbm, "mte2_bytes": m * k * act_bytes + weight_bytes,
                "mte3_bytes": m * n * out_bytes,
                "rate_work": flops, "traffic_bytes": hbm}
    if kind == "moe_grouped":
        base = condition["work"]
        base_shape = dict(condition["shape"])
        routed = max(1, int(shape.get("routed_tokens", 1)))
        base_routed = max(1, int(base_shape.get("routed_tokens", routed)))
        n, k = int(shape["n"]), int(shape["k"])
        base_n, base_k = int(base_shape["n"]), int(base_shape["k"])
        scale = (routed / base_routed) * (n * k) / max(1.0, base_n * base_k)
        return {"unit": "flops", "flops": float(base.get("flops", 0.0)) * scale,
                "vector_ops": float(base.get("vector_ops", 0.0)) * scale,
                "hbm_bytes": float(base.get("hbm_bytes", 0.0)) * scale,
                "mte2_bytes": float(base.get("mte2_bytes", 0.0)) * scale,
                "mte3_bytes": float(base.get("mte3_bytes", 0.0)) * scale,
                "rate_work": float(base.get("flops", 0.0)) * scale,
                "traffic_bytes": float(base.get("hbm_bytes", 0.0)) * scale}
    if kind == "attention":
        q, kv, heads, dim = (int(shape["q"]), int(shape["kv"]), int(shape["heads"]), int(shape["dim"]))
        flops = 2.0 * q * kv * heads * dim
        hbm = (q * heads * dim + kv * heads * dim + heads * q * kv * 2.0) * act_bytes
        return {"unit": "flops", "flops": flops, "vector_ops": 0.0,
                "hbm_bytes": hbm, "mte2_bytes": hbm, "mte3_bytes": q * heads * dim * out_bytes,
                "rate_work": flops, "traffic_bytes": hbm}
    if kind == "attention_softmax":
        if all(key in shape for key in ("q", "heads", "kv")):
            elements = max(1, int(shape["q"]) * int(shape["heads"]) * int(shape["kv"]))
        else:
            elements = max(1, int(shape.get("elements", 1)))
        return {"unit": "vector_ops", "flops": 0.0, "vector_ops": float(elements * 5),
                "hbm_bytes": float(elements * 2 * act_bytes), "mte2_bytes": float(elements * act_bytes),
                "mte3_bytes": float(elements * act_bytes),
                "rate_work": float(elements * 5), "traffic_bytes": float(elements * 2 * act_bytes)}
    if kind == "router_softmax":
        if all(key in shape for key in ("tokens", "experts")):
            elements = max(1, int(shape["tokens"]) * int(shape["experts"]))
        else:
            elements = max(1, int(shape.get("elements", 1)))
        return {"unit": "vector_ops", "flops": 0.0, "vector_ops": float(elements * 5),
                "hbm_bytes": float(elements * 2 * act_bytes), "mte2_bytes": float(elements * act_bytes),
                "mte3_bytes": float(elements * act_bytes),
                "rate_work": float(elements * 5), "traffic_bytes": float(elements * 2 * act_bytes)}
    if kind == "cache":
        # row_bytes is already a byte count, not an element count.
        elements = max(1, int(shape["rows"]) * int(shape["row_bytes"]))
        return {"unit": "bytes", "flops": 0.0, "vector_ops": 0.0,
                "hbm_bytes": float(elements * 2.0),
                "mte2_bytes": float(elements), "mte3_bytes": float(elements),
                "rate_work": float(elements), "traffic_bytes": float(elements * 2.0)}
    elements = max(1, int(shape.get("elements", shape.get("tokens", 1) * shape.get("width", 1))))
    base_ops = float(condition["work"].get("vector_ops") or 0.0)
    base_elements = 1
    for value in condition["work"].get("semantic_shape") or ():
        base_elements *= max(1, int(value))
    ops_per_element = (base_ops / max(1, base_elements)) if base_ops else 1.0
    vector_ops = elements * ops_per_element
    return {"unit": "elements", "flops": 0.0, "vector_ops": float(vector_ops),
            "hbm_bytes": float(elements * 2 * act_bytes), "mte2_bytes": float(elements * act_bytes),
            "mte3_bytes": float(elements * act_bytes),
            "rate_work": float(elements), "traffic_bytes": float(elements * 2 * act_bytes)}


def _working_set_bytes(condition: Mapping[str, Any], shape: Mapping[str, int]) -> int:
    # A conservative allocation estimate: activation, output, dequantized
    # weight for reference recipes, and a possible scale grid.
    precision = condition["precision"]
    act = _bytes_for(str(precision.get("input_dtype") or "bf16"))
    out = _bytes_for(str(precision.get("output_dtype") or precision.get("input_dtype") or "bf16"))
    kind = condition["shape_kind"]
    if kind == "gemm":
        m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
        weight = float(precision.get("packed_weight_bytes") or 0.0)
        if condition["execution_kind"] == "reference_decomposition":
            weight = n * k * act
        return int((m * k + n * k) * act + m * n * out + weight + n * k * 2.0)
    if kind == "moe_grouped":
        # Pending grouped work still reports an explicit estimate; it is never
        # allocated or timed by this campaign.
        return int(int(shape["experts"]) * int(shape["n"]) * int(shape["k"]) * act)
    if kind == "attention":
        q, kv, heads, dim = (int(shape["q"]), int(shape["kv"]), int(shape["heads"]), int(shape["dim"]))
        return int((q + kv) * heads * dim * act + q * heads * dim * out + heads * q * kv * 2.0)
    if kind == "attention_softmax":
        if all(key in shape for key in ("q", "heads", "kv")):
            elements = int(shape["q"]) * int(shape["heads"]) * int(shape["kv"])
        else:
            elements = int(shape.get("elements", 1))
        return int(elements * (act + 4.0))
    if kind == "cache":
        # source + destination byte buffers
        return int(int(shape["rows"]) * int(shape["row_bytes"]) * 2.0)
    elements = int(shape.get("elements", shape.get("tokens", 1) * shape.get("width", 1)))
    return int(elements * (act + out + 4.0))


def _axis_values(condition: Mapping[str, Any], axis: str, config: Any) -> list[dict[str, Any]]:
    shape = dict(condition["shape"])
    if axis not in shape:
        return []
    base = int(shape[axis])
    out: list[dict[str, Any]] = []
    kind = condition["shape_kind"]
    if kind == "gemm" and axis not in {"m", "n", "k"}:
        return []
    if kind in {"attention", "attention_softmax"} and axis not in {"q", "kv", "heads", "dim"}:
        return []
    for factor in config.ladder_factors:
        value = max(1, int(round(base * factor)))
        if value == base or value < 1:
            continue
        if factor < 1.0:
            split = "train" if factor <= 0.25 else "heldout"
        else:
            split = "heldout" if factor <= 2.0 else "train"
        out.append({
            "value": value, "axis": axis, "reason": f"ladder:{axis}:x{factor:g}",
            "split_hint": split, "split_group": None, "actual": False,
            "boundary_probe": False, "extrapolation": factor >= 4.0 and False,
            "factor": factor,
        })
    boundary_values: dict[int, list[int]] = {}
    for boundary in condition.get("boundaries", ()):
        if int(boundary.get("dim", -1)) != list(shape).index(axis):
            continue
        boundary_values.setdefault(int(boundary["value"]), []).append(1)
    for boundary in sorted(boundary_values):
        values = {
            boundary - 2: ("train", None),
            boundary - 1: ("heldout", f"boundary_holdout:{axis}:{boundary}"),
            boundary: ("train", None),
            boundary + 1: ("heldout", f"boundary_holdout:{axis}:{boundary}"),
            boundary + 2: ("train", None),
            max(1, boundary // 2): ("train", None),
            boundary * 2: ("train", None),
        }
        for value, (split, group) in values.items():
            if value < 1 or value == base:
                continue
            out.append({
                "value": int(value), "axis": axis,
                "reason": f"boundary:{axis}:T{boundary}:{value}",
                "split_hint": split, "split_group": group, "actual": False,
                "boundary_probe": True, "extrapolation": False,
            })
    # One outer interval beyond the train envelope, only if it is not already
    # a train anchor and fits the memory budget.
    outer = max(8 * base, base + 4)
    out.append({
        "value": int(outer), "axis": axis,
        "reason": f"outer_extrapolation:{axis}:x8", "split_hint": "heldout",
        "split_group": f"outer:{axis}", "actual": False,
        "boundary_probe": False, "extrapolation": True,
    })
    return out


def _shape_with(condition: Mapping[str, Any], axis: str, value: int) -> dict[str, int]:
    shape = dict(condition["shape"])
    if condition["shape_kind"] == "gemm" and axis == "m":
        shape["m"] = int(value)
    elif condition["shape_kind"] == "gemm":
        shape[axis] = int(value)
    elif condition["shape_kind"] in {"attention", "attention_softmax"}:
        if axis == "q":
            # Keep the existing context fixed: a query-length change is not a
            # silent KV/context change.  Prefill context 0 couples kv=q.
            context = max(0, int(shape["kv"]) - int(shape["q"]))
            shape["q"] = int(value)
            shape["kv"] = int(value) + context
        elif axis == "kv":
            shape["kv"] = int(value)
        elif axis == "heads":
            shape["heads"] = int(value)
        elif axis == "dim":
            shape["dim"] = int(value)
    elif axis in shape:
        shape[axis] = int(value)
    return shape


def _axis_legal(condition: Mapping[str, Any], axis: str, value: int,
                shape: Mapping[str, int], config: Any) -> tuple[bool, str]:
    if value < 1:
        return False, "non_positive"
    kind = condition["shape_kind"]
    if kind == "gemm" and axis not in {"m", "n", "k"}:
        return False, "axis_not_applicable"
    if kind == "moe_grouped" and axis not in {"routed_tokens", "experts", "n", "k"}:
        return False, "axis_not_applicable"
    if kind not in {"gemm", "attention", "attention_softmax", "moe_grouped"} and axis not in {
        "tokens", "width", "elements", "rows", "row_bytes", "experts",
    }:
        return False, "axis_not_applicable"
    if kind in {"attention", "attention_softmax"}:
        q, kv = int(shape["q"]), int(shape["kv"])
        if q < 1 or kv < 1:
            return False, "attention_shape_non_positive"
        if q > kv and condition["phase"] == "prefill" and condition.get("state", {}).get("compress_ratio", 0) == 0:
            # A causal prefill primitive cannot have more queries than KV.
            return False, "causal_prefill_q_exceeds_kv"
        if axis == "heads" and int(shape["heads"]) > 1024:
            return False, "heads_above_cap"
        if axis == "dim" and int(shape["dim"]) > 8192:
            return False, "head_dim_above_cap"
    if kind == "gemm":
        if int(shape["m"]) > 65536 or int(shape["n"]) > 1_048_576 or int(shape["k"]) > 1_048_576:
            return False, "dimension_above_cap"
    if _working_set_bytes(condition, shape) > config.max_working_set_bytes:
        return False, "working_set_above_budget"
    return True, "legal"


def condition_point_materialization(condition: Mapping[str, Any],
                                    config: Any) -> dict[str, Any]:
    """Materialize legal points and explicit rejected-axis evidence for a condition.

    Pending/unsupported conditions are materialized too; their points carry
    ``expected_status`` and ``required_backend`` so absence of an executable
    backend is not represented as an empty condition.
    """
    kind = condition["shape_kind"]
    points: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    primary = _primary_axis(kind)
    actual = _point_from_shape(
        condition, dict(condition["shape"]), axis=primary,
        value=int(condition["shape"].get(primary, 1)),
        reason="actual_demand", split_hint="train", split_group="actual",
        actual=True, boundary_probe=False, extrapolation=False, config=config,
    )
    points.append(actual)
    for axis in _axes_for_kind(kind):
        if axis not in condition["shape"]:
            continue
        for candidate in _axis_values(condition, axis, config):
            shape = _shape_with(condition, axis, candidate["value"])
            legal, why = _axis_legal(condition, axis, candidate["value"], shape, config)
            if not legal:
                rejections.append({"axis": axis, "value": int(candidate["value"]),
                                   "shape": shape, "reason": why,
                                   "expected_status": condition.get("status")})
                continue
            coupled = "kv" if axis == "q" and condition["shape_kind"] in {"attention", "attention_softmax"} else None
            point = _point_from_shape(
                condition, shape, axis=axis, value=int(candidate["value"]),
                reason=f"{candidate['reason']}:{why}" + (f":context_held:{coupled}" if coupled else ""),
                split_hint=candidate["split_hint"], split_group=candidate.get("split_group"),
                actual=False, boundary_probe=bool(candidate.get("boundary_probe")),
                extrapolation=bool(candidate.get("extrapolation")), config=config,
            )
            points.append(point)
    merged: dict[str, dict[str, Any]] = {}
    for point in points:
        key = point["point_key"]
        if key not in merged:
            merged[key] = point
        else:
            current = merged[key]
            current["reason"] = current["reason"] + "|" + point["reason"]
            current["boundary_probe"] = current["boundary_probe"] or point["boundary_probe"]
            current["extrapolation"] = current["extrapolation"] and point["extrapolation"]
            if point["split_hint"] == "train":
                current["split_hint"] = "train"
                current["split_group"] = current.get("split_group") or point.get("split_group")
    ordered = sorted(merged.values(), key=lambda item: (
        0 if item["actual"] else 1, item["axis"], item["value"], item["point_key"],
    ))
    return {"condition_key": condition.get("condition_key"), "points": ordered,
            "rejections": rejections, "status": condition.get("status"),
            "expected_status": condition.get("status"),
            "required_backend": condition.get("required_backend")}


def condition_points(condition: Mapping[str, Any], config: Any) -> list[dict[str, Any]]:
    """Public point materializer, including pending/unsupported conditions."""
    return list(condition_point_materialization(condition, config)["points"])


def materialize_conditions(conditions: Mapping[str, Mapping[str, Any]],
                           condition_keys: Sequence[str], config: Any,
                           *, include_pending: bool = True) -> list[dict[str, Any]]:
    """Public targeted materialization by condition key.

    Returns request-like records for exact conditions, including pending ones.
    The caller can inspect/write them without launching a device.
    """
    machine = config.machine_spec()
    cost_model = PredictiveCostModel(machine)
    output = []
    for key in condition_keys:
        condition = conditions.get(str(key))
        if condition is None:
            output.append({"condition_key": str(key), "status": "missing_condition",
                           "reason": "condition key not present in extracted conditions"})
            continue
        if not include_pending and condition["status"] in {"pending_backend", "unsupported"}:
            continue
        materialization = condition_point_materialization(condition, config)
        prepared_points = []
        for point in materialization["points"]:
            point["condition_keys"] = [key]
            point["lineage"] = [f"{key}:{_canonical(point['shape'])}"]
            prepared_points.append(point)
        for point in assign_splits(prepared_points):
            request = _request_from_point(
                point, config=config, condition=condition, machine=machine,
                cost_model=cost_model,
                source_cases=[_origin_summary(origin) for origin in (
                    condition["sources"].values()
                    if isinstance(condition["sources"], Mapping) else condition["sources"]
                )],
            )
            request["status"] = "pending_backend" if condition["status"] in {
                "pending_backend", "excluded_communication", "unsupported",
            } else "pending"
            request["request_status"] = request["status"]
            request["materialized"] = True
            output.append(request)
    return output


def _primary_axis(kind: str) -> str:
    return {
        "gemm": "m", "attention": "q", "attention_softmax": "q",
        "router_softmax": "tokens", "moe_grouped": "routed_tokens",
        "cache": "rows", "vector": "tokens",
    }.get(kind, "elements")


def _axes_for_kind(kind: str) -> tuple[str, ...]:
    if kind == "gemm":
        return ("m", "n", "k")
    if kind in {"attention", "attention_softmax"}:
        return ("q", "kv", "heads", "dim")
    if kind == "router_softmax":
        return ("tokens", "experts")
    if kind == "cache":
        return ("rows", "row_bytes")
    if kind == "moe_grouped":
        return ("routed_tokens", "experts", "n", "k")
    return ("tokens", "width", "elements")


def assign_splits(points: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate physical points and freeze train/heldout split.

    Any point that is a train anchor in any condition wins globally.  Boundary
    T-1/T+1 pairs share a split group, but a train appearance elsewhere still
    prevents the same physical point from becoming a holdout.
    """
    merged: dict[str, dict[str, Any]] = {}
    for point in points:
        key = str(point["point_key"])
        if key not in merged:
            merged[key] = {
                **point, "lineage": list(point.get("lineage", ())),
                "condition_keys": list(point.get("condition_keys", ())),
            }
        else:
            current = merged[key]
            current["lineage"] = sorted({*current["lineage"], *point.get("lineage", ())})
            current["condition_keys"] = sorted({
                *current["condition_keys"], *point.get("condition_keys", ()),
            })
            current["reason"] = str(current.get("reason", "")) + "|" + str(point.get("reason", ""))
            if point.get("split_hint") == "train":
                current["split_hint"] = "train"
            current["boundary_probe"] = current["boundary_probe"] or bool(point.get("boundary_probe"))
            current["extrapolation"] = current["extrapolation"] or bool(point.get("extrapolation"))
    train_groups: set[str] = set()
    for point in merged.values():
        if point.get("split_hint") == "train" or point.get("actual"):
            group = point.get("split_group")
            if group:
                train_groups.add(str(group))
    for key, point in merged.items():
        group = point.get("split_group")
        if group and str(group) in train_groups:
            point["split_hint"] = "train"
        point["split"] = "train" if point["split_hint"] == "train" else "heldout"
    return [merged[key] for key in sorted(merged)]


def _request_from_point(point: Mapping[str, Any], *, config: CampaignConfig,
                        condition: Mapping[str, Any], machine: MachineSpec,
                        cost_model: PredictiveCostModel,
                        source_cases: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    shape = point["shape"]
    work = point["work"]
    analytic, analytic_basis = _analytic_ms(condition, point, cost_model, machine)
    query_context = {
        "requested_shape": dict(shape),
        "semantic_shape": list(point["semantic_shape"]),
        "dtype": condition["precision"].get("input_dtype"),
        "weight_dtype": condition["precision"].get("weight_dtype"),
        "output_dtype": condition["precision"].get("output_dtype"),
        "compute_dtype": condition["precision"].get("compute_dtype"),
        "accumulation_dtype": condition["precision"].get("accumulation_dtype"),
        "layout": condition["layout"],
        "quantization": condition["quantization"],
        "recipe": condition["recipe"],
        "family": condition["family"],
        "execution_kind": condition["execution_kind"],
        "requested_backend": condition["backend"],
        "worker_recipe": condition["worker_recipe"],
        "target_match": condition["target_match"],
        "parallel": condition["features"].get("parallel"),
        "features": condition["features"].get("metadata", {}),
        "state": condition["state"],
        "protocol": condition["protocol"],
    }
    if point.get("expert_token_counts") is not None:
        query_context["expert_token_counts"] = list(point["expert_token_counts"])
        query_context["expert_token_histogram_source"] = point.get(
            "expert_token_histogram_source")
    request = {
        "request_key": point["point_key"],
        "measurement_id": point["point_key"],
        "request_id": f"campaign:{config.campaign_id}:{point['point_key'][3:15]}",
        "hardware_key": machine.hardware_key or machine.name,
        "op_family": condition["family"],
        "phase": condition["phase"],
        "recipe": condition["recipe"],
        "shape": dict(shape),
        "dtype": condition["precision"].get("input_dtype"),
        "layout": condition["layout"],
        "features": {
            **condition["precision"],
            "execution_kind": condition["execution_kind"],
            "worker_recipe": condition["worker_recipe"],
            "backend": condition["backend"],
            "target_match": condition["target_match"],
            "state": condition["state"],
            "boundaries": condition["boundaries"],
            "expansion": condition["expansion"],
        },
        "competition": None,
        "priority": _point_priority(point),
        "query_count": 1,
        "predicted_ms": analytic,
        "analytic_ms": analytic,
        "extrapolation": 1.0 if point["extrapolation"] else 0.0,
        "target_impact": 1.0,
        "run_id": "",
        "request_status": "pending",
        "condition_key": condition["condition_key"],
        "condition": {key: value for key, value in condition.items() if key != "sources"},
        "point": point,
        "split": point["split"],
        "execution_kind": condition["execution_kind"],
        "worker_recipe": condition["worker_recipe"],
        "backend": condition["backend"],
        "target_match": condition["target_match"],
        "backend_requirement": condition.get("required_backend"),
        "required_backend": condition.get("required_backend"),
        "expected_status": condition.get("status"),
        "status": "pending",
        "query_context": query_context,
        "protocol": dict(config.protocol),
        "source_summary": _source_summary(condition),
        "source_cases": list(source_cases) if source_cases is not None else [
            _origin_summary(origin) for origin in condition["sources"].values()
        ],
        "lineage_condition_keys": list(point["condition_keys"]),
        "work": {**work, "analytic_ms": analytic, "analytic_basis": analytic_basis},
    }
    return request


def _roofline_analytic(condition: Mapping[str, Any], point: Mapping[str, Any],
                      machine: MachineSpec) -> float:
    work = point["work"]
    die = machine.dies[0]
    compute_dtype = str(condition["precision"].get("compute_dtype") or "bf16")
    components = []
    if float(work.get("flops", 0.0)) > 0:
        components.append(float(work["flops"]) / max(1e-9, machine.peak_flops_for(compute_dtype) / 1000.0))
    if float(work.get("vector_ops", 0.0)) > 0:
        components.append(float(work["vector_ops"]) / max(1e-9, die.vector_ops_per_s / 1000.0))
    if float(work.get("hbm_bytes", 0.0)) > 0:
        components.append(float(work["hbm_bytes"]) / max(1e-9, die.hbm_bw_bytes_per_s / 1000.0))
    components.append(max(0.0, die.launch_us) / 1000.0)
    return max(components) if components else 0.0


def _analytic_ms(condition: Mapping[str, Any], point: Mapping[str, Any],
                 cost_model: PredictiveCostModel, machine: MachineSpec) -> tuple[float, str]:
    work = point["work"]
    semantic = tuple(int(v) for v in point["semantic_shape"])
    op = OpSpec(
        op_id="campaign_probe",
        family=str(condition["family"]), phase=str(condition["phase"]),
        stage=0, layer=0, requests=(),
        semantic_shape=semantic, local_shape=semantic, padded_shape=semantic,
        launch_shape=tuple(int(v) for v in point["launch_shape"]),
        flops=float(work.get("flops", 0.0)), hbm_bytes=float(work.get("hbm_bytes", 0.0)),
        mte2_bytes=float(work.get("mte2_bytes", 0.0)),
        mte3_bytes=float(work.get("mte3_bytes", 0.0)),
        vector_ops=float(work.get("vector_ops", 0.0)),
        recipe=str(condition["recipe"]), dtype=str(condition["precision"].get("input_dtype") or "bf16"),
        layout=str(condition["layout"]),
    )
    try:
        value = float(cost_model.cost(op).analytic_ms)
        basis = "predictive_cost_model"
    except Exception as exc:  # noqa: BLE001 - declared analytic fallback, never measurement
        value = _roofline_analytic(condition, point, machine)
        basis = f"declared_roofline_fallback:{type(exc).__name__}"
    if not math.isfinite(value) or value <= 0:
        raise A3InputError(f"campaign_analytic_nonpositive:{condition['condition_key']}:{basis}")
    return value, basis


def _point_priority(point: Mapping[str, Any]) -> int:
    if point.get("actual"):
        return 100
    if point.get("split") == "heldout":
        return 80 if not point.get("extrapolation") else 60
    if point.get("boundary_probe"):
        return 70
    return 50


