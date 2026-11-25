"""Network efficiency, occurrence DAGs and hardware scaling.

[Role] Aggregate operator work, report physical critical-path bounds and
  construct hardware-scaled cases.
[Invariants] Occurrence reports preserve multiplicity, realized dependencies
  and resource identities. Scaling preserves the case's modeling boundary.
[Timeouts & Locks] Uses the simulator's evaluation context and active deadline.
[Side-Effect Policy] Analysis reads input records and scaling returns a new
  case when rates or capacity change.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.hardware import Hardware

from ..program.physical_key import normalize_reference
from ..efficiency.roofline_graph import (
    PhysicalWorkGraph, PhysicalWorkNode, RooflineTargets, critical_path_roofline,
    graph_from_occurrences,
)
from ..structure import parse_case
from .peaks import (
    DEFAULT_CONTEXT, DEFAULT_SEQUENCES, DEFAULT_TOKENS, _BOUNDS, _KINDS,
    _SCALE_KEYS, _check_pos_float, _op_native_call_id, _role_scale_factor,
    _split_link_namespace, _validate_geometry,
    resolve_efficiency_scope,
    analyze_operator_row, extract_hardware_peaks, legacy_service_regime_diagnostics,
)


_OCCURRENCE_IDENTITY = re.compile(r"(?<![\w.])s(\d+)\.l(\d+)")
_WORK_SIGNATURE_PRECISION_DIGITS = 6
_PIPE_EQUIV_TOL = 1e-12
_DURATION_MATCH_REL_TOL = 1e-9
_SITE_RESOURCE_PREFIX = {
    "cube": "matrix",
    "vector": "vector",
    "hbm": "memory",
    "link": "link",
}
_TRANSFER_CHANNEL_BW_KEYS = (
    "uni_peak_bw_gbps",
    "bi_peak_bw_gbps",
    "calibrated_uni_bw_gbps",
    "calibrated_bi_bw_gbps",
    "derived_uni_bw_gbps",
    "derived_bi_bw_gbps",
    "effective_uni_bw_gbps",
    "effective_bi_bw_gbps",
    "pageable_uni_bw_gbps",
    "node_peak_bw_gbps",
    "solo_uni_bw_gbps",
    "solo_effective_uni_bw_gbps",
    "single_peer_flow_cap_gbps",
    "single_peer_flow_cap_bytes_per_ms",
    "capacity_gbps",
    "gbps",
    "capacity_bytes_per_ms",
    "peak_capacity_bytes_per_ms",
    "switch_capacity_bytes_per_ms",
)
_TOPOLOGY_OPTION_BW_KEYS = (
    "intra_bw_4p_gbps",
    "intra_bw_8p_gbps",
    "superpod_bw_gbps",
    "inter_pod_bw_gbps",
)
_HOST_CHANNEL_NAMES = frozenset({"h2d", "d2h"})


def _graph_resource_rates(graph_report: Mapping[str, Any] | None) -> dict[str, float]:
    rates: dict[str, float] = {}
    for raw in ((graph_report or {}).get("resources") or ()):
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("resource") or "")
        cap = float(raw.get("capacity_per_ms", 0.0) or 0.0)
        if name and cap > 0.0:
            rates[name] = cap
    return rates


def _raw_serviced_pipes(
    row: Mapping[str, Any],
    rate_by_resource: Mapping[str, float],
) -> dict[str, Any]:
    serviced = row.get("serviced")
    cube_ms = vector_ms = hbm_ms = link_ms = other_non_hbm_ms = 0.0
    dominant_link: str | None = None
    link_service_by_resource: dict[str, float] = {}
    if isinstance(serviced, Mapping) and not row.get("joint_native_service"):
        for raw_name in sorted(serviced):
            amount = float(serviced[raw_name] or 0.0)
            if amount <= 0.0:
                continue
            name = str(raw_name)
            if name.startswith(("scheduler@", "native_call_service")):
                continue
            split = _split_link_namespace(name)
            if split is not None and split[1].startswith("sdma_flow:"):
                svc_ms = amount
            else:
                rate = float(rate_by_resource.get(name, 0.0) or 0.0)
                if rate <= 0.0:
                    continue
                svc_ms = amount / rate
            if name.startswith(("memory@", "gm_hbm")):
                hbm_ms = max(hbm_ms, svc_ms)
            elif name.startswith("matrix@"):
                cube_ms = max(cube_ms, svc_ms)
            elif name.startswith("vector@"):
                vector_ms = max(vector_ms, svc_ms)
            elif split is not None or name.startswith("ub_inj:") or ":ub_inj:" in name:
                link_service_by_resource[name] = svc_ms
                if svc_ms > link_ms:
                    link_ms = svc_ms
                    dominant_link = name
            else:
                other_non_hbm_ms = max(other_non_hbm_ms, svc_ms)
    basis = row.get("basis") if isinstance(row.get("basis"), Mapping) else {}
    inv_models = basis.get("invocation_models") or ()
    fixed_ms = max(
        (float(item.get("startup_ms", 0.0) or 0.0) for item in inv_models if isinstance(item, Mapping)),
        default=float(row.get("fixed_ms", 0.0) or 0.0),
    )
    minimum_ms = max(
        (float(item.get("minimum_ms", 0.0) or 0.0) for item in inv_models if isinstance(item, Mapping)),
        default=float(row.get("minimum_ms", 0.0) or 0.0),
    )
    return {
        "cube_ms": cube_ms,
        "vector_ms": vector_ms,
        "hbm_ms": hbm_ms,
        "link_ms": link_ms,
        "other_non_hbm_ms": other_non_hbm_ms,
        "fixed_ms": fixed_ms,
        "minimum_ms": minimum_ms,
        "dominant_link_resource": dominant_link,
        "link_service_by_resource": link_service_by_resource,
    }


def _pipe_profile_key(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    link_map = metrics.get("link_service_by_resource") or {}
    return (
        f"{float(metrics['cube_ms']):.12g}",
        f"{float(metrics['vector_ms']):.12g}",
        f"{float(metrics['hbm_ms']):.12g}",
        f"{float(metrics['link_ms']):.12g}",
        f"{float(metrics['other_non_hbm_ms']):.12g}",
        f"{float(metrics['fixed_ms']):.12g}",
        f"{float(metrics['minimum_ms']):.12g}",
        f"{float(metrics['duration_ms']):.12g}",
        metrics.get("dominant_link_resource"),
        tuple((str(k), f"{float(v):.12g}") for k, v in sorted(link_map.items())),
    )


def _record_bucket(
    buckets: dict[Any, list[dict[str, Any]]],
    key: Any,
    metrics: dict[str, Any],
) -> None:
    entries = buckets.setdefault(key, [])
    sig = _pipe_profile_key(metrics)
    if not any(_pipe_profile_key(existing) == sig for existing in entries):
        entries.append(metrics)


def _row_pipe_metrics(
    row: Mapping[str, Any],
    rate_by_resource: Mapping[str, float],
    *,
    prefer_executed: bool,
) -> dict[str, Any]:
    op = row.get("operator") if isinstance(row.get("operator"), Mapping) else row
    count = max(1, int(row.get("occurrences", 1) or 1))
    raw_total = float(row.get("total_ms", 0.0) or 0.0)
    duration_ms = float(row.get("duration_ms", raw_total / count if raw_total else 0.0) or 0.0)
    has_analyzed = any(
        key in row for key in ("t_cube_ideal_ms", "t_vec_ideal_ms", "t_hbm_ideal_ms", "t_link_ideal_ms")
    )
    if has_analyzed:
        is_analytical_only = prefer_executed and row.get("work_basis") == "analytical_reference"

        def _pick(exec_key: str, ideal_key: str) -> float:
            if is_analytical_only:
                return 0.0
            if prefer_executed and row.get(exec_key) is not None:
                return float(row[exec_key] or 0.0)
            return float(row.get(ideal_key, 0.0) or 0.0)

        pipes = {
            "cube_ms": _pick("t_cube_exec_ms", "t_cube_ideal_ms"),
            "vector_ms": _pick("t_vec_exec_ms", "t_vec_ideal_ms"),
            "hbm_ms": _pick("t_hbm_exec_ms", "t_hbm_ideal_ms"),
            "link_ms": _pick("t_link_exec_ms", "t_link_ideal_ms"),
            "other_non_hbm_ms": 0.0 if is_analytical_only else float(row.get("t_other_non_hbm_ms", 0.0) or 0.0),
            "fixed_ms": float(row.get("fixed_ms", 0.0) or 0.0),
            "minimum_ms": float(row.get("minimum_ms", 0.0) or 0.0),
            "dominant_link_resource": None if is_analytical_only else row.get("dominant_link_resource"),
            "link_service_by_resource": (
                {} if is_analytical_only
                else dict(row.get("link_service_by_resource") or {})
            ),
        }
    else:
        pipes = _raw_serviced_pipes(row, rate_by_resource)
    native_call_id = str(row.get("native_call_id") or "")
    if not native_call_id and isinstance(row.get("operator"), Mapping):
        native_call_id = _op_native_call_id(row["operator"])
    is_cohort_priced = bool(
        row.get("is_cross_replica_collective", False)
        or row.get("isolation_boundary") == "complete_collective_cohort"
    )
    return {
        **pipes,
        "duration_ms": duration_ms,
        "row_replica": int(row.get("replica", 0) or 0),
        "is_cross_replica_collective": is_cohort_priced,
        "stage": int(op.get("stage", row.get("stage", 0)) or 0),
        "layer": int(op.get("layer", row.get("layer", 0)) or 0),
        "op_id": str(op.get("op_id") or row.get("op_id") or ""),
        "physical_id": str(row.get("physical_id") or ""),
        "family": str(op.get("family") or row.get("family") or ""),
        "recipe": str(op.get("recipe") or row.get("recipe") or ""),
        "program": str(row.get("program") or ""),
        "phase": str(op.get("phase") or row.get("phase") or ""),
        "dtype": str(op.get("dtype") or row.get("dtype") or ""),
        "local_shape": tuple(int(dim) for dim in (op.get("local_shape") or row.get("local_shape") or ())),
        "native_call_id": native_call_id,
    }


def _build_occurrence_index(
    operator_rows: Sequence[Mapping[str, Any]] | None,
    graph_report: Mapping[str, Any] | None = None,
    *,
    prefer_executed: bool = False,
) -> dict[str, Any]:
    """Build multi-key structural lookup tables without colliding distinct layer shapes."""
    rates = _graph_resource_rates(graph_report)
    index: dict[str, Any] = {
        "op_id": {},
        "physical_id": {},
        "norm_call": {},
        "call": {},
        "norm_shape": {},
        "norm_meta": {},
        "replicas": set(),
    }
    for row in operator_rows or ():
        metrics = _row_pipe_metrics(row, rates, prefer_executed=prefer_executed)
        op_id = metrics["op_id"]
        if not op_id:
            continue
        rep, stage, layer = metrics["row_replica"], metrics["stage"], metrics["layer"]
        index["replicas"].add(rep)
        norm_id = normalize_reference(op_id, stage, layer)
        family, recipe, program, phase = (
            metrics["family"], metrics["recipe"], metrics["program"], metrics["phase"]
        )
        call_id, dtype, shape = metrics["native_call_id"], metrics["dtype"], metrics["local_shape"]
        for rep_key in (rep, None):
            _record_bucket(index["op_id"], (rep_key, op_id), metrics)
            if metrics["physical_id"]:
                _record_bucket(index["physical_id"], (rep_key, stage, metrics["physical_id"]), metrics)
            if call_id:
                _record_bucket(index["norm_call"], (rep_key, stage, norm_id, family, recipe, program, phase, call_id), metrics)
                _record_bucket(index["call"], (rep_key, stage, family, recipe, program, phase, call_id), metrics)
            if shape:
                _record_bucket(index["norm_shape"], (rep_key, stage, norm_id, family, recipe, program, phase, dtype, shape), metrics)
            _record_bucket(index["norm_meta"], (rep_key, stage, norm_id, family, recipe, program, phase), metrics)
    return index


def _pick_unambiguous(
    candidates: Sequence[dict[str, Any]] | None,
    item: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    raw_val = item.get("isolated_ms")
    if raw_val is not None:
        target = float(raw_val)
        tol = max(_PIPE_EQUIV_TOL, _DURATION_MATCH_REL_TOL * max(1.0, abs(target)))
        matches = [cand for cand in candidates if abs(cand["duration_ms"] - target) <= tol]
        if len(matches) == 1:
            return matches[0]
    return None


def _match_for_replica_scope(
    item: Mapping[str, Any],
    index: Mapping[str, Any],
    rep_key: int | None,
    *,
    op_id: str,
    stage: int,
    norm_id: str,
    phys_id: str,
    family: str,
    recipe: str,
    program: str,
    phase: str,
    call_id: str,
    dtype: str,
    shape: tuple[int, ...],
) -> dict[str, Any] | None:
    cands = index["op_id"].get((rep_key, op_id))
    if cands:
        chosen = _pick_unambiguous(cands, item)
        if chosen is not None:
            return chosen
    if phys_id:
        chosen = _pick_unambiguous(index["physical_id"].get((rep_key, stage, phys_id)), item)
        if chosen is not None:
            return chosen
    if call_id:
        for table, key in (
            ("norm_call", (rep_key, stage, norm_id, family, recipe, program, phase, call_id)),
            ("call", (rep_key, stage, family, recipe, program, phase, call_id)),
        ):
            chosen = _pick_unambiguous(index[table].get(key), item)
            if chosen is not None:
                return chosen
    if shape:
        chosen = _pick_unambiguous(
            index["norm_shape"].get((rep_key, stage, norm_id, family, recipe, program, phase, dtype, shape)),
            item,
        )
        if chosen is not None:
            return chosen
    return _pick_unambiguous(
        index["norm_meta"].get((rep_key, stage, norm_id, family, recipe, program, phase)),
        item,
    )


def _match_timeline_item(
    item: Mapping[str, Any],
    index: Mapping[str, Any],
    op_physical_ids: Mapping[tuple[int, str], str] | None = None,
) -> dict[str, Any] | None:
    op_id = str(item.get("op_id") or "")
    if not op_id:
        return None
    has_rep = item.get("replica") is not None
    item_rep = int(item["replica"]) if has_rep else 0
    match = _OCCURRENCE_IDENTITY.search(op_id)
    stage = int(item["stage"]) if item.get("stage") is not None else (int(match.group(1)) if match else 0)
    layer = int(item["layer"]) if item.get("layer") is not None else (int(match.group(2)) if match else 0)
    norm_id = normalize_reference(op_id, stage, layer) if (match is not None or item.get("layer") is not None) else op_id
    family = str(item.get("family") or "")
    recipe = str(item.get("recipe") or "")
    program = str(item.get("program") or "")
    phase = str(item.get("phase") or "")
    native_call = item.get("native_call") if isinstance(item.get("native_call"), Mapping) else {}
    call_id = str(native_call.get("call_id") or "")
    typed = native_call.get("typed") if isinstance(native_call.get("typed"), Mapping) else {}
    shapes = typed.get("shapes") if isinstance(typed.get("shapes"), Mapping) else {}
    raw_shape = item.get("local_shape") or shapes.get("local") or ()
    shape = tuple(int(dim) for dim in raw_shape) if raw_shape else ()
    dtype = str(item.get("dtype") or typed.get("dtype") or "")
    common = dict(
        op_id=op_id, stage=stage, norm_id=norm_id, family=family, recipe=recipe,
        program=program, phase=phase, call_id=call_id, dtype=dtype, shape=shape,
    )
    exact_phys_id = str(item.get("physical_id") or "")
    if not exact_phys_id and op_physical_ids:
        exact_phys_id = str(op_physical_ids.get((item_rep, op_id)) or "")
    if has_rep:
        chosen = _match_for_replica_scope(item, index, item_rep, phys_id=exact_phys_id, **common)
        if chosen is not None:
            return chosen
        if item_rep in index.get("replicas", ()):
            return None
    fallback_phys_id = exact_phys_id
    if not fallback_phys_id and op_physical_ids:
        fallback_phys_id = str(op_physical_ids.get((0, op_id)) or "")
    return _match_for_replica_scope(item, index, None, phys_id=fallback_phys_id, **common)


def _existing_resource_ids(item: Mapping[str, Any]) -> dict[str, str]:
    existing = item.get("resource_ids")
    if isinstance(existing, Mapping):
        return {str(k): str(v) for k, v in existing.items()}
    if existing:
        return {str(k): str(v) for k, v in existing}
    return {}


def _scope_link_identity(
    name: str,
    *,
    rep: int,
    stage: int,
    matched: Mapping[str, Any],
) -> str:
    is_cohort_priced = bool(matched.get("is_cross_replica_collective"))
    same_replica = int(matched.get("row_replica", rep)) == rep
    if not is_cohort_priced and same_replica:
        return str(name)
    return f"{name}@r{rep}:s{stage}"


def _occurrence_resource_ids(
    item: Mapping[str, Any],
    matched: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    if item.get("replica") is None and item.get("stage") is None:
        return None
    rep = int(item.get("replica", 0) or 0)
    stage = int(item.get("stage", 0) or 0)
    resolved = _existing_resource_ids(item)
    for logical in ("cube", "vector", "hbm"):
        resolved.setdefault(logical, f"{_SITE_RESOURCE_PREFIX[logical]}@r{rep}:s{stage}")
    if "link" not in resolved:
        dom = matched.get("dominant_link_resource") if matched is not None else None
        if dom and matched is not None:
            resolved["link"] = _scope_link_identity(str(dom), rep=rep, stage=stage, matched=matched)
        else:
            resolved["link"] = f"link@r{rep}:s{stage}"
    return resolved


def _occurrence_link_services(
    item: Mapping[str, Any],
    matched: Mapping[str, Any] | None,
) -> tuple[tuple[str, float], ...]:
    if matched is None:
        return ()
    if "link" in _existing_resource_ids(item) or item.get("link_services"):
        return ()
    link_map = matched.get("link_service_by_resource")
    if not isinstance(link_map, Mapping) or not link_map:
        return ()
    rep = int(item.get("replica", 0) or 0)
    stage = int(item.get("stage", 0) or 0)
    return tuple(
        (_scope_link_identity(str(name), rep=rep, stage=stage, matched=matched), float(val))
        for name, val in sorted(link_map.items())
        if float(val) > 0.0
    )


def occurrence_work_graph(
    graph_report: Mapping[str, Any] | None,
    operator_rows: Sequence[Mapping[str, Any]],
    *,
    op_physical_ids: Mapping[tuple[int, str], str] | None = None,
):
    """Realized occurrence DAG with per-node physical service fields and site identities."""
    if not graph_report:
        return None
    timeline = tuple(graph_report.get("timeline", ()) or ())
    if not timeline:
        return None
    index = _build_occurrence_index(operator_rows, graph_report, prefer_executed=False)
    rows = []
    for item in timeline:
        matched = _match_timeline_item(item, index, op_physical_ids=op_physical_ids)
        res_ids = _occurrence_resource_ids(item, matched)
        link_svcs = _occurrence_link_services(item, matched)
        pipes = (
            {
                "cube_ms": matched["cube_ms"],
                "vector_ms": matched["vector_ms"],
                "hbm_ms": matched["hbm_ms"],
                "link_ms": matched["link_ms"],
                "fixed_ms": matched["fixed_ms"],
            }
            if matched is not None else {}
        )
        row_out = {**item, **pipes}
        if res_ids is not None:
            row_out["resource_ids"] = res_ids
        if link_svcs:
            row_out["link_services"] = link_svcs
        rows.append(row_out)
    return graph_from_occurrences(rows)


def _causal_split(timeline: Sequence[Mapping[str, Any]], input_ready_ms: float) -> list[tuple[float, float, float]]:
    """Per-row ``(causal wait, executed service, absolute release offset)`` split."""
    complete = {str(item["id"]): float(item["complete_ms"])
                for item in timeline
                if item.get("id") is not None and item.get("complete_ms") is not None}
    split: list[tuple[float, float, float]] = []
    for item in timeline:
        if item.get("ready_ms") is not None and item.get("complete_ms") is not None:
            ready = float(item["ready_ms"])
            done = float(item["complete_ms"])
            floor = max((complete[str(pred)] for pred in item.get("predecessors", ()) or ()
                         if str(pred) in complete), default=float(input_ready_ms))
            split.append((
                max(0.0, ready - floor),
                max(0.0, done - max(ready, floor)),
                max(0.0, ready - float(input_ready_ms)),
            ))
        else:
            split.append((0.0, max(0.0, float(item.get("elapsed_ms", 0.0) or 0.0)), 0.0))
    return split


def _priced_pipes(
    operator_rows: Sequence[Mapping[str, Any]],
    graph_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _build_occurrence_index(operator_rows, graph_report, prefer_executed=True)


def _reconciled_pipes(matched: Mapping[str, Any] | None, elapsed: float, wait: float) -> dict[str, float]:
    """Single-node executed pipe summary without rescaling pipes to elapsed time."""
    if matched is None:
        return {
            "cube_ms": 0.0, "vector_ms": 0.0, "hbm_ms": 0.0, "link_ms": 0.0,
            "fixed_ms": elapsed + wait, "launch_ms": elapsed + wait,
        }
    cube = min(max(0.0, float(matched["cube_ms"])), elapsed)
    vec = min(max(0.0, float(matched["vector_ms"])), elapsed)
    hbm = min(max(0.0, float(matched["hbm_ms"])), elapsed)
    link = min(max(0.0, float(matched["link_ms"])), elapsed)
    other = min(max(0.0, float(matched.get("other_non_hbm_ms", 0.0) or 0.0)), elapsed)
    fixed_in = min(max(0.0, float(matched.get("fixed_ms", 0.0) or 0.0)), elapsed)
    min_in = min(max(0.0, float(matched.get("minimum_ms", 0.0) or 0.0)), elapsed)
    non_hbm_floor = min(elapsed, max(min_in, fixed_in + max(cube, vec, link, other)))
    zero_h = min(elapsed, max(non_hbm_floor, elapsed - hbm))
    exposed_h = max(0.0, elapsed - zero_h)
    if exposed_h <= _PIPE_EQUIV_TOL:
        h_eff = min(hbm, max(cube, vec))
        return {
            "cube_ms": cube, "vector_ms": vec, "hbm_ms": h_eff, "link_ms": link,
            "fixed_ms": wait + max(0.0, elapsed - max(cube, vec, link)),
            "launch_ms": wait + elapsed,
        }
    h_eff = max(cube, vec, link) + exposed_h
    return {
        "cube_ms": cube, "vector_ms": vec, "hbm_ms": min(hbm, h_eff), "link_ms": link,
        "fixed_ms": wait + max(0.0, elapsed - max(cube, vec, min(hbm, h_eff), link)),
        "launch_ms": wait + zero_h,
    }


def _unique_aux_id(base: str, suffix: str, used: set[str]) -> str:
    candidate = f"{base}:{suffix}"
    while candidate in used:
        candidate = f"{candidate}#"
    used.add(candidate)
    return candidate


def _executed_stage_rows(
    item: Mapping[str, Any],
    node_id: str,
    preds: tuple[str, ...],
    res_ids: Mapping[str, str] | None,
    matched: Mapping[str, Any] | None,
    elapsed: float,
    used: set[str],
) -> list[dict[str, Any]]:
    base_extra = {"resource_ids": res_ids} if res_ids is not None else {}
    if matched is None:
        return [{
            **item, "id": node_id, "predecessors": preds, **base_extra,
            "cube_ms": 0.0, "vector_ms": 0.0, "hbm_ms": 0.0, "link_ms": 0.0,
            "fixed_ms": elapsed, "launch_ms": elapsed,
        }]
    cube = min(max(0.0, float(matched["cube_ms"])), elapsed)
    vec = min(max(0.0, float(matched["vector_ms"])), elapsed)
    hbm = min(max(0.0, float(matched["hbm_ms"])), elapsed)
    link = min(max(0.0, float(matched["link_ms"])), elapsed)
    other = min(max(0.0, float(matched.get("other_non_hbm_ms", 0.0) or 0.0)), elapsed)
    fixed_in = min(max(0.0, float(matched.get("fixed_ms", 0.0) or 0.0)), elapsed)
    min_in = min(max(0.0, float(matched.get("minimum_ms", 0.0) or 0.0)), elapsed)
    non_hbm_floor = min(elapsed, max(min_in, fixed_in + max(cube, vec, link, other)))
    zero_h = min(elapsed, max(non_hbm_floor, elapsed - hbm))
    exposed_h = max(0.0, elapsed - zero_h)
    if exposed_h > _PIPE_EQUIV_TOL and zero_h > _PIPE_EQUIV_TOL:
        pre_id = _unique_aux_id(node_id, "non_hbm", used)
        h_cov = min(max(0.0, hbm - exposed_h), max(cube, vec))
        return [
            {
                **item, "id": pre_id, "predecessors": preds, **base_extra,
                "cube_ms": cube, "vector_ms": vec, "hbm_ms": h_cov, "link_ms": link,
                "fixed_ms": max(0.0, zero_h - max(cube, vec, link)),
                "launch_ms": zero_h,
            },
            {
                **item, "id": node_id, "predecessors": (pre_id,), **base_extra,
                "cube_ms": 0.0, "vector_ms": 0.0, "hbm_ms": exposed_h, "link_ms": 0.0,
                "fixed_ms": 0.0, "launch_ms": 0.0,
            },
        ]
    if exposed_h > _PIPE_EQUIV_TOL:
        return [{
            **item, "id": node_id, "predecessors": preds, **base_extra,
            "cube_ms": 0.0, "vector_ms": 0.0, "hbm_ms": exposed_h, "link_ms": 0.0,
            "fixed_ms": max(0.0, elapsed - exposed_h), "launch_ms": 0.0,
        }]
    h_cov = min(hbm, max(cube, vec))
    return [{
        **item, "id": node_id, "predecessors": preds, **base_extra,
        "cube_ms": cube, "vector_ms": vec, "hbm_ms": h_cov, "link_ms": link,
        "fixed_ms": max(0.0, elapsed - max(cube, vec, link)),
        "launch_ms": elapsed,
    }]


def executed_occurrence_work_graph(
    graph_report: Mapping[str, Any] | None,
    operator_rows: Sequence[Mapping[str, Any]],
    *,
    op_physical_ids: Mapping[tuple[int, str], str] | None = None,
):
    """Realized occurrence DAG reconciled to the Engine's executed spans without rescaling pipes."""
    if not graph_report:
        return None
    timeline = tuple(graph_report.get("timeline", ()) or ())
    if not timeline:
        return None
    index = _priced_pipes(operator_rows, graph_report)
    input_ready = float(graph_report.get("input_ready_ms", 0.0) or 0.0)
    used: set[str] = {
        str(item["id"]) for item in timeline if item.get("id") is not None
    }
    rows: list[dict[str, Any]] = []
    for idx, (item, (wait, elapsed, abs_release)) in enumerate(
        zip(timeline, _causal_split(timeline, input_ready))
    ):
        node_id = str(item["id"]) if item.get("id") is not None else f"occurrence{idx}"
        used.add(node_id)
        preds = tuple(str(pred) for pred in (item.get("predecessors") or ()))
        if wait > _PIPE_EQUIV_TOL and abs_release > _PIPE_EQUIV_TOL:
            rel_id = _unique_aux_id(node_id, "release", used)
            rows.append({
                "id": rel_id, "predecessors": (),
                "cube_ms": 0.0, "vector_ms": 0.0, "hbm_ms": 0.0, "link_ms": 0.0,
                "fixed_ms": abs_release, "launch_ms": abs_release,
            })
            preds = (*preds, rel_id)
        matched = _match_timeline_item(item, index, op_physical_ids=op_physical_ids)
        res_ids = _occurrence_resource_ids(item, matched)
        rows.extend(_executed_stage_rows(item, node_id, preds, res_ids, matched, elapsed, used))
    return graph_from_occurrences(rows)


def occurrence_critical_path(graph_report: Mapping[str, Any] | None,
                            operator_rows: Sequence[Mapping[str, Any]],
                            *, graph: Any | None = None
                            ) -> dict[str, Any] | None:
    """Critical-path roofline over actual timeline occurrences, never grouped rows.

    Every timeline node retains its own physical resource service and the
    realized data/stream/state/rendezvous predecessor edges.  Occurrences are
    never deduplicated: repeated layers keep their full multiplicity.
    """
    if graph is None:
        graph = occurrence_work_graph(graph_report, operator_rows)
    if graph is None:
        return None
    roofline = critical_path_roofline(
        graph, RooflineTargets(cube=1.0, vector=1.0,
                               hbm=1.0, link=1.0))
    return {
        "critical_path_ideal_ms": roofline.critical_path_ideal_ms,
        "critical_path_target_ms": roofline.critical_path_target_ms,
        "physical_ideal_ms": roofline.physical_ideal_ms,
        "resource_totals": dict(roofline.resource_totals),
        "resource_cut_bounds": dict(roofline.resource_cut_bounds),
        "critical_path_nodes": tuple(roofline.critical_path_nodes),
        "occurrence_nodes": len(graph.nodes),
        "basis": "actual_execution_occurrences_with_realized_dependency_edges",
    }


OCCURRENCE_DAG_SCHEMA = "mindie-sim.occurrence-dag/1"


def _export_occurrence_dag(graph):
    """Serialize the realized occurrence DAG as public node/edge rows.

    The object graph stays available through ``_retain_graph``. The compact
    public form preserves realized predecessors and physical resource identities
    when a report is serialized.
    """
    if graph is None:
        return None
    return {
        "schema": OCCURRENCE_DAG_SCHEMA,
        "nodes": tuple((
            str(node.node_id),
            float(node.cube_ms), float(node.vector_ms),
            float(node.hbm_ms), float(node.link_ms),
            float(node.launch_ms), float(node.fixed_ms),
            tuple(str(before) for before in node.predecessors),
            int(node.occurrences),
            tuple((str(resource), str(identity))
                  for resource, identity in node.resource_ids),
            tuple((str(identity), float(service))
                  for identity, service in node.link_services),
        ) for node in graph.nodes),
        "edges": tuple((str(before), str(after))
                       for before, after in graph.edges),
    }


def _occurrence_dag_graph(payload):
    """Rebuild the realized occurrence DAG from its public node/edge form."""
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema") != OCCURRENCE_DAG_SCHEMA:
        return None
    nodes = tuple(PhysicalWorkNode(
        node_id=str(node_id),
        cube_ms=float(cube_ms), vector_ms=float(vector_ms),
        hbm_ms=float(hbm_ms), link_ms=float(link_ms),
        launch_ms=float(launch_ms), fixed_ms=float(fixed_ms),
        predecessors=tuple(str(before) for before in predecessors),
        occurrences=int(occurrences),
        resource_ids=tuple((str(resource), str(identity))
                           for resource, identity in resource_ids),
        link_services=(
            tuple((str(identity), float(service)) for identity, service in rest[0])
            if rest else ()
        ),
    ) for (node_id, cube_ms, vector_ms, hbm_ms, link_ms, launch_ms, fixed_ms,
           predecessors, occurrences, resource_ids, *rest) in payload.get("nodes", ()))
    return PhysicalWorkGraph(
        nodes,
        tuple((str(before), str(after))
              for before, after in payload.get("edges", ())))


def _analyze_operator_rows(
    rows: Sequence[Mapping[str, Any]],
    parsed: CaseSpec,
    peaks_by_site: dict[tuple[int, int], dict[str, Any]],
    obs_map: Mapping[str, float],
    pmu_map: Mapping[str, Mapping[str, float]],
    resource_rates: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    analyzed_list: list[dict[str, Any]] = []
    for row in rows:
        op = row["operator"]
        rep, st = int(row.get("replica", 0) or 0), int(op.get("stage", 0) or 0)
        site = (rep, st)
        if site not in peaks_by_site:
            peaks_by_site[site] = extract_hardware_peaks(parsed, stage=st, replica=rep)
        op_id, phys_id = op.get("op_id", ""), row.get("physical_id", "")
        rec_key = f"{op.get('family', '')}:{op.get('recipe', '')}"
        analyzed_list.append(analyze_operator_row(
            row, parsed, peaks=peaks_by_site[site],
            observed_duration_ms=obs_map.get(op_id, obs_map.get(phys_id, obs_map.get(rec_key))),
            pmu=pmu_map.get(op_id, pmu_map.get(phys_id, pmu_map.get(rec_key))),
            resource_rates=resource_rates,
        ))
    return analyzed_list


def _scale_endpoint_rates(raw: Any, scale: float) -> Any:
    if isinstance(raw, Mapping):
        return {k: float(v) * scale for k, v in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [[k, float(v) * scale] for k, v in raw]
    return raw


def _scale_machine_links(machine_dict: dict[str, Any], link_bw_scale: float) -> None:
    if link_bw_scale == 1.0:
        return
    for lnk in machine_dict.get("links", ()):
        if str(lnk.get("scope") or "").strip().lower() == "host":
            continue
        if str(lnk.get("channel") or "").strip().lower() in _HOST_CHANNEL_NAMES:
            continue
        for lk in ("capacity_bytes_per_ms", "switch_capacity_bytes_per_ms", "peak_capacity_bytes_per_ms"):
            if lnk.get(lk) is not None:
                lnk[lk] = float(lnk[lk]) * link_bw_scale
        if lnk.get("endpoint_rates") is not None:
            lnk["endpoint_rates"] = _scale_endpoint_rates(lnk["endpoint_rates"], link_bw_scale)
    for ch_name, ch_spec in (machine_dict.get("transfer_channels") or {}).items():
        if not isinstance(ch_spec, Mapping):
            continue
        if str(ch_name).strip().lower() in _HOST_CHANNEL_NAMES:
            continue
        if str(ch_spec.get("scope") or "").strip().lower() == "host":
            continue
        for bw_key in _TRANSFER_CHANNEL_BW_KEYS:
            if ch_spec.get(bw_key) is not None:
                ch_spec[bw_key] = float(ch_spec[bw_key]) * link_bw_scale
        if ch_spec.get("endpoint_rates") is not None:
            ch_spec["endpoint_rates"] = _scale_endpoint_rates(ch_spec["endpoint_rates"], link_bw_scale)
    for opt_key in _TOPOLOGY_OPTION_BW_KEYS:
        if machine_dict.get(opt_key) is not None:
            machine_dict[opt_key] = float(machine_dict[opt_key]) * link_bw_scale


def analyze_network_efficiency(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = DEFAULT_TOKENS, context: int = DEFAULT_CONTEXT, sequences: int = DEFAULT_SEQUENCES, scope: str | None = None,
    operator_report: Mapping[str, Any] | None = None, graph_report: Mapping[str, Any] | None = None,
    observed_durations_ms: Mapping[str, float] | None = None, pmu_by_op: Mapping[str, Mapping[str, float]] | None = None,
    run_graph: bool = True, _precomputed_graph: tuple[Any, CaseSpec, dict[str, Any]] | None = None,
    evaluation_context: Any | None = None, _retain_graph: bool = False,
    timeline_replicas: Any = None,
) -> dict[str, Any]:
    """Compute whole-network 4D MFU/MBU/CV efficiency and bottleneck breakdowns."""
    parsed = parse_case(case)
    scope = resolve_efficiency_scope(parsed, scope)
    _validate_geometry(phase, tokens, context, sequences, scope)
    # A supplied report is not an escape hatch: its rows must be modeled too.
    from ..efficiency.unmodeled_gate import assert_report_supported, assert_timing_supported
    from .attribution import _fast_operator_and_graph_reports, attribute_program

    pkg = sys.modules.get(__package__ or "")
    fast_reports_fn = (
        getattr(pkg, "_fast_operator_and_graph_reports", _fast_operator_and_graph_reports)
        if pkg is not None else _fast_operator_and_graph_reports
    )
    attribute_fn = (
        getattr(pkg, "attribute_program", attribute_program)
        if pkg is not None else attribute_program
    )

    assert_timing_supported(parsed, entry="analyze_network_efficiency")
    assert_report_supported(operator_report, entry="analyze_network_efficiency")
    assert_report_supported(graph_report, entry="analyze_network_efficiency")
    evaluation = None
    if operator_report is None:
        from ..evaluation import EvaluationContext
        evaluation = (evaluation_context if evaluation_context is not None
                      else EvaluationContext.create(parsed))
        operator_report, auto_graph = fast_reports_fn(
            parsed, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope, run_graph=run_graph,
            precomputed_graph=_precomputed_graph,
            evaluation_context=evaluation,
            **({"timeline_replicas": timeline_replicas} if timeline_replicas is not None else {}))
        graph_report = auto_graph if graph_report is None else graph_report

    all_raw_rows = list(operator_report["operators"])
    lead_replica = (
        min(int(r.get("replica", 0) or 0) for r in all_raw_rows)
        if all_raw_rows else 0
    )
    raw_rows = [r for r in all_raw_rows if int(r.get("replica", 0) or 0) == lead_replica]
    peaks_by_site, obs_map, pmu_map = {}, dict(observed_durations_ms or {}), dict(pmu_by_op or {})
    graph_rates = _graph_resource_rates(graph_report) or None
    analyzed_ops = _analyze_operator_rows(
        raw_rows, parsed, peaks_by_site, obs_map, pmu_map, resource_rates=graph_rates)
    extra_source = operator_report.get("_all_replica_operators") or all_raw_rows
    extra_raw = [r for r in extra_source if int(r.get("replica", 0) or 0) != lead_replica]
    graph_analyzed_ops = (
        [*analyzed_ops, *_analyze_operator_rows(
            extra_raw, parsed, peaks_by_site, obs_map, pmu_map, resource_rates=graph_rates)]
        if extra_raw else analyzed_ops
    )
    op_physical_ids = operator_report.get("_op_physical_ids")
    totals = {"flops": 0.0, "vector_ops": 0.0, "hbm_bytes": 0.0,
              "link_bytes": 0.0, "cube_ms": 0.0, "vec_ms": 0.0,
              "hbm_ms": 0.0, "link_ms": 0.0, "roofline_ms": 0.0}
    stage_pipes: dict[int, dict[str, float]] = {}
    for analyzed in analyzed_ops:
        occurrences = analyzed["occurrences"]
        totals["flops"] += analyzed["flops"] * occurrences
        totals["vector_ops"] += analyzed["vector_ops"] * occurrences
        totals["hbm_bytes"] += analyzed["hbm_bytes"] * occurrences
        totals["link_bytes"] += analyzed["link_bytes"] * occurrences
        totals["cube_ms"] += analyzed["t_cube_ideal_ms"] * occurrences
        totals["vec_ms"] += analyzed["t_vec_ideal_ms"] * occurrences
        totals["hbm_ms"] += analyzed["t_hbm_ideal_ms"] * occurrences
        totals["link_ms"] += analyzed["t_link_ideal_ms"] * occurrences
        totals["roofline_ms"] += analyzed["roofline_ideal_ms"] * occurrences
        st_bucket = stage_pipes.setdefault(
            int(analyzed["stage"]), {"cube": 0.0, "vec": 0.0, "hbm": 0.0, "link": 0.0})
        st_bucket["cube"] += analyzed["t_cube_ideal_ms"] * occurrences
        st_bucket["vec"] += analyzed["t_vec_ideal_ms"] * occurrences
        st_bucket["hbm"] += analyzed["t_hbm_ideal_ms"] * occurrences
        st_bucket["link"] += analyzed["t_link_ideal_ms"] * occurrences

    operator_sum_ms = sum(x["total_ms"] for x in analyzed_ops)
    graph_duration_ms = float(graph_report["duration_ms"]) if (graph_report is not None and observed_durations_ms is None) else operator_sum_ms
    safe_graph_ms = max(graph_duration_ms, 1e-12)
    stage_count = max(1, len(stage_pipes))
    tot_flops, tot_vec_ops = totals["flops"], totals["vector_ops"]
    tot_hbm_b, tot_link_b = totals["hbm_bytes"], totals["link_bytes"]
    tot_cube_id, tot_vec_id = totals["cube_ms"], totals["vec_ms"]
    tot_hbm_id, tot_link_id = totals["hbm_ms"], totals["link_ms"]
    serial_roof_id = totals["roofline_ms"]
    tot_roof_id = max(
        (max(bucket.values(), default=0.0) for bucket in stage_pipes.values()),
        default=0.0,
    )
    cv_ops = [x for x in analyzed_ops if x["operator_kind"] == "cv_fused" and x["cv_overlap_ratio"] is not None]
    cv_ms = sum(x["total_ms"] for x in cv_ops)
    network_cv_overlap = (sum(x["cv_overlap_ratio"] * x["total_ms"] for x in cv_ops) / cv_ms) if cv_ms > 0 else None

    def _agg(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        g_ms, denom = sum(x["total_ms"] for x in items), max(sum(x["total_ms"] for x in items), 1e-12)
        h_id, c_id = sum(x["t_hbm_ideal_ms"] * x["occurrences"] for x in items), sum(x["t_cube_ideal_ms"] * x["occurrences"] for x in items)
        v_id, l_id = sum(x["t_vec_ideal_ms"] * x["occurrences"] for x in items), sum(x["t_link_ideal_ms"] * x["occurrences"] for x in items)
        return {"occurrences": sum(x["occurrences"] for x in items), "duration_ms": g_ms, "total_ms": g_ms, "share_of_step": g_ms / safe_graph_ms, "share_of_operator_sum": g_ms / max(operator_sum_ms, 1e-12), "mbu": h_id / denom, "cube_mfu": c_id / denom, "vec_mfu": v_id / denom, "link_util": l_id / denom, "roofline_efficiency": max(h_id, c_id, v_id, l_id) / denom}

    fam_g, bound_g, kind_g = defaultdict(list), defaultdict(list), defaultdict(list)
    for x in analyzed_ops:
        fam_g[x["family"], x["recipe"]].append(x)
        bound_g[x["dominant_bound"]].append(x)
        kind_g[x["operator_kind"]].append(x)

    occurrence_graph = occurrence_work_graph(
        graph_report, graph_analyzed_ops, op_physical_ids=op_physical_ids)
    attr_kwargs: dict[str, Any] = {"occurrence_graph": occurrence_graph}
    if graph_analyzed_ops is not analyzed_ops:
        attr_kwargs["graph_operator_rows"] = graph_analyzed_ops
    if op_physical_ids:
        attr_kwargs["op_physical_ids"] = op_physical_ids
    attribution = attribute_fn(graph_report, analyzed_ops, **attr_kwargs)
    occurrence_roofline = occurrence_critical_path(
        graph_report, graph_analyzed_ops, graph=occurrence_graph)
    if occurrence_roofline is not None:
        tot_roof_id = occurrence_roofline["physical_ideal_ms"]
    stage_span_ms = stage_count * safe_graph_ms
    nominal = {"cube_mfu": tot_cube_id / stage_span_ms, "vec_mfu": tot_vec_id / stage_span_ms,
               "mbu": tot_hbm_id / stage_span_ms, "link_util": tot_link_id / safe_graph_ms,
               "roofline_efficiency": tot_roof_id / safe_graph_ms,
               "denominator": "graph_span_ms", "semantics": "nominal_peak_relative_work_over_span"}
    graph_accel = graph_report.get("accel_status") if graph_report is not None else None
    from ..frontend.body_spec import describe_included_components
    return {
        "schema": "mindie-sim.efficiency/1",
        "engine_backend": (graph_report or {}).get("engine_backend"),
        "accel_status": graph_accel,
        "geometry": {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope},
        "included_components": describe_included_components(parsed, scope=scope),
        "total_flops": tot_flops, "total_vector_ops": tot_vec_ops, "total_hbm_bytes": tot_hbm_b, "total_link_bytes": tot_link_b,
        "total_cube_ideal_ms": tot_cube_id, "total_vec_ideal_ms": tot_vec_id, "total_hbm_ideal_ms": tot_hbm_id, "total_link_ideal_ms": tot_link_id,
        "total_roofline_ideal_ms": tot_roof_id, "operator_serial_roofline_ideal_ms": serial_roof_id, "operator_sum_ms": operator_sum_ms, "graph_duration_ms": graph_duration_ms,
        "network_cube_mfu": nominal["cube_mfu"], "network_vec_mfu": nominal["vec_mfu"], "network_mbu": nominal["mbu"],
        "network_link_util": nominal["link_util"], "network_roofline_efficiency": nominal["roofline_efficiency"],
        "nominal": nominal, "effective_service_utilization": dict(attribution.get("service_utilization_by_role") or {}),
        "critical_path_roofline_ms": (occurrence_roofline["critical_path_ideal_ms"]
                                      if occurrence_roofline else None),
        "critical_path_roofline_target_ms": (occurrence_roofline["critical_path_target_ms"]
                                             if occurrence_roofline else None),
        "critical_path_nodes": (occurrence_roofline["critical_path_nodes"]
                                if occurrence_roofline else ()),
        "critical_path_occurrence_nodes": (occurrence_roofline["occurrence_nodes"]
                                           if occurrence_roofline else 0),
        "physical_resource_cut_ms": (max(occurrence_roofline["resource_cut_bounds"].values(),
                                         default=0.0)
                                     if occurrence_roofline else None),
        "attribution": attribution,
        **({"legacy_service_regime_diagnostics": diagnostics}
           if (diagnostics := legacy_service_regime_diagnostics(parsed)) is not None else {}),
        "operator_serial_roofline_efficiency": serial_roof_id / max(operator_sum_ms, 1e-12), "network_cv_overlap": network_cv_overlap,
        "breakdown_by_family": sorted([{"family": f, "recipe": r, **_agg(items)} for (f, r), items in fam_g.items()], key=lambda r: -r["total_ms"]),
        "breakdown_by_bound": {b: _agg(bound_g.get(b, ())) for b in _BOUNDS}, "breakdown_by_kind": {k: _agg(kind_g.get(k, ())) for k in _KINDS}, "operators": analyzed_ops,
        # The compact DAG preserves realized edges in serialized reports.
        # A caller retaining the object graph already has those edges.
        **({"occurrence_dag": _export_occurrence_dag(occurrence_graph)}
           if occurrence_graph is not None and not _retain_graph else {}),
        **({"_occurrence_graph": occurrence_graph} if _retain_graph else {}),
    }


def _operator_fit_signature(op: Mapping[str, Any]) -> str:
    return json.dumps({"op_id": op.get("op_id", ""), "family": op.get("family", ""), "recipe": op.get("recipe", ""), "dtype": op.get("dtype", ""), "local_shape": list(op.get("local_shape") or ()), "flops": round(float(op.get("flops", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS), "vector_ops": round(float(op.get("vector_ops", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS), "hbm_bytes": round(float(op.get("hbm_bytes", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS), "link_bytes": round(float(op.get("link_bytes", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS)}, sort_keys=True, separators=(",", ":"))


def _decompose_operator_pipes(op: Mapping[str, Any]) -> dict[str, Any]:
    """Export per-pipe work/nominal bounds without identifying a bottleneck from total duration."""
    t_base, t_min = float(op["duration_ms"]), float(op.get("minimum_ms", 0.0) or 0.0)
    t_fixed = min(max(float(op["fixed_ms"]), t_min), t_base)
    service_ms = max(0.0, t_base - t_fixed)
    ideals = {"hbm": float(op["t_hbm_ideal_ms"]), "cube": float(op["t_cube_ideal_ms"]),
              "vec": float(op["t_vec_ideal_ms"]), "link": float(op["t_link_ideal_ms"])}
    active = {pipe: value for pipe, value in ideals.items() if value > 0.0}
    nominal_lower = {pipe: (value / service_ms if service_ms > 0.0 else None) for pipe, value in ideals.items()}
    return {
        "fixed_ms": t_fixed, "minimum_ms": t_min, "service_ms": service_ms,
        "work_ideal_ms": ideals, "roofline_ideal_ms": max(active.values(), default=0.0),
        "nominal_work_lower_bound_utilization": nominal_lower,
        "candidate_pipes": sorted(active),
        "bottleneck_pipe": None,
        "bottleneck_identified": False,
        "bottleneck_evidence": None,
        "identification": "total_duration_does_not_identify_pipe_bottleneck",
        "unidentified_pipes": sorted(active),
        "t_hbm_eff_ms": None, "t_cube_eff_ms": None, "t_vec_eff_ms": None, "t_link_eff_ms": None,
        "effective_mbu": None, "effective_cube_mfu": None,
        "effective_vec_mfu": None, "effective_link_util": None,
    }


def fit_case_roofline_regimes(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = DEFAULT_CONTEXT, sequences: int = 1, scope: str | None = None, observed_durations_ms: Mapping[str, float] | None = None, baseline_analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit multi-pipe roofline parameters (fixed_us, effective_mbu/cube_mfu/vec_mfu) per operator."""
    pkg = sys.modules.get(__package__ or "")
    analyze_fn = getattr(pkg, "analyze_network_efficiency", analyze_network_efficiency) if pkg is not None else analyze_network_efficiency
    base = dict(baseline_analysis) if baseline_analysis is not None else analyze_fn(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, observed_durations_ms=observed_durations_ms)
    fitted_ops, by_key, by_recipe = [], {}, defaultdict(list)
    for op in base["operators"]:
        dec, sig_key = _decompose_operator_pipes(op), _operator_fit_signature(op)
        by_key[sig_key] = dec
        by_recipe[f"{op['family']}:{op['recipe']}"].append(dec)
        fitted_ops.append({"signature_key": sig_key, "op_id": op["op_id"], "family": op["family"], "recipe": op["recipe"], "operator_kind": op["operator_kind"], "dominant_bound": op["dominant_bound"], "occurrences": op["occurrences"], "duration_ms": op["duration_ms"], "fixed_us": dec["fixed_ms"] * 1000.0, **dec})
    summaries = []
    for rk, items in sorted(by_recipe.items()):
        summaries.append({
            "family": rk.split(":", 1)[0], "recipe": rk.split(":", 1)[1],
            "fixed_us": sum(x["fixed_ms"] for x in items) / len(items) * 1000.0,
            "candidate_pipes": sorted({pipe for x in items for pipe in x["candidate_pipes"]}),
            "nominal_work_lower_bound_utilization": {
                pipe: max((x["nominal_work_lower_bound_utilization"].get(pipe) or 0.0) for x in items)
                for pipe in ("hbm", "cube", "vec", "link")
            },
            "bottleneck_pipe": None, "bottleneck_identified": False,
            "identification": "total_duration_does_not_identify_pipe_bottleneck",
            "effective_mbu": None, "effective_cube_mfu": None,
            "effective_vec_mfu": None, "effective_link_util": None,
        })
    return {"schema": "mindie-sim.roofline-fit/1", "geometry": base["geometry"], "fitted_regimes": summaries, "operators": fitted_ops, "by_key": by_key,
            "identification_scope": "total_duration_exports_nominal_work_bounds_but_does_not_identify_pipe_bottleneck"}


def scale_case_hardware(case: CaseSpec | Mapping[str, Any], *, hbm_bw_scale: float = 1.0, cube_flops_scale: float = 1.0, vector_ops_scale: float = 1.0, link_bw_scale: float = 1.0, hbm_capacity_scale: float = 1.0, baseline_analysis: Mapping[str, Any] | None = None) -> CaseSpec:
    """Scale hardware peaks, explicit/derived rates, service modes, links, and capacity."""
    for name, val in zip(_SCALE_KEYS, (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale)):
        _check_pos_float(f"hardware_scale_{name}", val)
    parsed = parse_case(case)
    if (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale) == (1.0, 1.0, 1.0, 1.0, 1.0):
        return parsed
    kw = dict(hbm_bw_scale=hbm_bw_scale, cube_flops_scale=cube_flops_scale, vector_ops_scale=vector_ops_scale, link_bw_scale=link_bw_scale)
    mode_to_role = {m_name: m_role for die in parsed.machine.dies for m_name, (m_role, _) in die.service_modes}
    # Serialized dies include machine-wide device defaults and per-die overrides.
    machine_dict = deepcopy(parsed.machine.to_dict())
    for die, dev_dict in zip(parsed.machine.dies, machine_dict["devices"]):
        ins_map = dev_dict.get("compute_arch", {}).get("instruction_services", {})
        svc_role = {sn: (ss["scale_with"]["resource"] if isinstance(ss.get("scale_with"), Mapping) else str(ss.get("resource") or ss.get("owner_axis") or sn)) for sn, ss in ins_map.items()}
        dev_dict["memory_bytes"] = max(1, int(round(die.memory_bytes * hbm_capacity_scale)))
        dev_dict["compute_peaks"] = {dt: float(val) * cube_flops_scale for dt, val in dev_dict.get("compute_peaks", {}).items()}
        dev_dict["rates"] = {r: float(val) * _role_scale_factor(svc_role.get(r, r), **kw) for r, val in dev_dict.get("rates", {}).items()}
        for key in ("hbm_read_bytes_per_s", "hbm_write_bytes_per_s"):
            if dev_dict.get(key) is not None:
                dev_dict[key] = float(dev_dict[key]) * hbm_bw_scale
        for m_name, m_spec in dev_dict.get("service_modes", {}).items():
            if m_spec.get("rate") is not None:
                m_role = "hbm" if m_name in {"mte2_aic_dram", "mte2_aiv_dram"} else str(m_spec.get("resource") or svc_role.get(m_name) or mode_to_role.get(m_name, m_name))
                m_spec["rate"] = float(m_spec["rate"]) * _role_scale_factor(m_role, **kw)
        for svc_name, svc_spec in ins_map.items():
            if not svc_spec.get("scale_with"):
                sf = _role_scale_factor(svc_role[svc_name], **kw)
                if svc_spec.get("clock_hz") is not None:
                    svc_spec["clock_hz"] = float(svc_spec["clock_hz"]) * sf
                elif svc_spec.get("per_owner_rate") is not None:
                    svc_spec["per_owner_rate"] = float(svc_spec["per_owner_rate"]) * sf
                elif svc_spec.get("issue_ns") is not None:
                    svc_spec["issue_ns"] = float(svc_spec["issue_ns"]) / sf
    _scale_machine_links(machine_dict, link_bw_scale)
    new_kernels = deepcopy(parsed.runtime.kernels)
    legacy_rules = list(new_kernels.get("service_regimes", ()) or ())
    metadata = dict(parsed.runtime.metadata)
    if legacy_rules:
        metadata["legacy_service_regime_diagnostics"] = {
            "production_prediction": False, "scaled": False,
            "source": "runtime.kernels.service_regimes",
            "rules": deepcopy(legacy_rules),
        }
    return replace(parsed, machine=Hardware.from_mapping(machine_dict),
                   runtime=replace(parsed.runtime, kernels=new_kernels,
                                   metadata=tuple(sorted(metadata.items()))))
