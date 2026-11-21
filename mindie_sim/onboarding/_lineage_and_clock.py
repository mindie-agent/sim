"""Disjoint capture lineage, authoritative clock parity, and planned-grid coordinate validators.

[Role] Enforce cryptographic train-vs-heldout and cross-level capture lineage
  isolation, resolve a single authoritative clock operator (``busy_union`` vs
  ``elapsed_span``) with zero caller/plan/payload conflicts, validate measured
  durations, physical tolerance ``max(rtol * measured, atol_ns)``, event/rank
  integrity, and exact 1:1 planned-grid coordinate & multi-rank participation.
[Invariants]
  * Never accepts caller-supplied ``predicted_*`` fields.
  * Never mixes ``busy_union`` and ``elapsed_span`` across row, boundary,
    payload, or plan coordinates.
  * Rejects train/heldout and cross-level overlap across roots, capture SHA-256
    digests, URIs, and parent digests.
  * Every multi-rank parallel topology requires exact distinct rank count.
[Timeouts & Locks] Pure in-memory validation helpers; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from mindie_sim.schema.common import ModelingError

from .contracts import (
    DEFAULT_PHYSICAL_ATOL_NS,
    FORBIDDEN_CALLER_PREDICTION_KEYS,
    NS_PER_MS,
    NS_PER_US,
    PERCENT_SCALE,
    VALID_CLOCK_OPERATORS,
)

STANDARD_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {
    "L1": "device_kernel",
    "L2": "fx_region",
    "L3": "model_step",
}
REQUIRED_MULTILEVEL_LEVELS = ("L1", "L2", "L3")
_ALLOWED_OK_STATUSES = frozenset({"", "ok", "pass", "passed", "success"})


def exceeds_physical_tolerance(
    predicted_ns: int,
    measured_ns: int,
    *,
    threshold_pct: float,
    atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS,
) -> bool:
    """Return True iff ``|predicted_ns - measured_ns| > max(rtol * measured_ns, atol_ns)``."""
    tol_ns = max((float(threshold_pct) / PERCENT_SCALE) * float(measured_ns), float(atol_ns))
    return abs(int(predicted_ns) - int(measured_ns)) > tol_ns


def _norm_digest(token: str) -> str:
    low = token.strip().lower()
    if low.startswith("sha256:"):
        return low[7:]
    return low


def validate_disjoint_lineage(
    observations: Sequence[Mapping[str, Any]],
    *,
    external_train_roots: Sequence[str] = (),
    external_train_digests: Sequence[str] = (),
    external_train_uris: Sequence[str] = (),
    require_disjoint_level_roots: bool = True,
) -> dict[str, Any]:
    """Fail closed if train vs heldout share roots, capture SHA-256 digests, or URIs."""
    train_roots: set[str] = {str(r).strip() for r in external_train_roots if str(r).strip()}
    train_digests: set[str] = {_norm_digest(str(d)) for d in external_train_digests if str(d).strip()}
    train_uris: set[str] = {str(u).strip() for u in external_train_uris if str(u).strip()}
    heldout_roots: set[str] = set()
    heldout_digests: set[str] = set()
    heldout_uris: set[str] = set()
    level_roots: dict[str, set[str]] = {}
    for idx, row in enumerate(observations):
        cid = str(row.get("case_id") or f"obs-{idx}")
        split = str(row.get("split") or "heldout").strip().lower()
        if split not in ("train", "heldout"):
            raise ModelingError(f"observation_split_invalid:{cid}:{split}")
        boundary = row.get("boundary") if isinstance(row.get("boundary"), Mapping) else {}
        root = str(
            boundary.get("acquisition_uid")
            or row.get("acquisition_root")
            or row.get("acquisition_uid")
            or ""
        ).strip()
        if not root:
            raise ModelingError(f"observation_acquisition_root_missing:{cid}")
        raw_sha = _norm_digest(str(boundary.get("raw_sha256") or row.get("raw_sha256") or ""))
        cap_uri = str(boundary.get("capture_uri") or row.get("capture_uri") or "").strip()
        if split == "train":
            train_roots.add(root)
            if raw_sha:
                train_digests.add(raw_sha)
            if cap_uri:
                train_uris.add(cap_uri)
        else:
            heldout_roots.add(root)
            if raw_sha:
                heldout_digests.add(raw_sha)
            if cap_uri:
                heldout_uris.add(cap_uri)
            level = str(row.get("level") or "L1").strip().upper()
            level_roots.setdefault(level, set()).add(root)
        for parent in row.get("parent_digests") or boundary.get("parent_digests") or ():
            ptok = _norm_digest(str(parent))
            if split == "heldout" and (ptok in train_roots or ptok in train_digests):
                raise ModelingError(f"train_heldout_parent_lineage_leakage:{cid}:{ptok}")

    train_overlap = sorted(train_roots & heldout_roots)
    if train_overlap:
        raise ModelingError(f"train_heldout_lineage_overlap:{','.join(train_overlap)}")
    digest_overlap = sorted(train_digests & heldout_digests)
    if digest_overlap:
        raise ModelingError(f"train_heldout_capture_digest_overlap:{','.join(digest_overlap)}")
    uri_overlap = sorted(train_uris & heldout_uris)
    if uri_overlap:
        raise ModelingError(f"train_heldout_capture_uri_overlap:{','.join(uri_overlap)}")
    if require_disjoint_level_roots:
        root_to_levels: dict[str, set[str]] = {}
        for level, roots in level_roots.items():
            for root in roots:
                root_to_levels.setdefault(root, set()).add(level)
        cross_level = sorted(
            f"{root}({'+'.join(sorted(lvls))})"
            for root, lvls in root_to_levels.items()
            if len(lvls) > 1
        )
        if cross_level:
            raise ModelingError(f"cross_level_lineage_overlap:{','.join(cross_level)}")
    return {
        "train_roots": sorted(train_roots),
        "heldout_roots": sorted(heldout_roots),
        "level_roots": {lvl: sorted(roots) for lvl, roots in sorted(level_roots.items())},
        "disjoint_ok": True,
    }


def _check_no_caller_predictions(row: Mapping[str, Any], cid: str) -> None:
    for key in FORBIDDEN_CALLER_PREDICTION_KEYS:
        if key in row and row[key] is not None:
            raise ModelingError(f"caller_predicted_duration_forbidden:{cid}:{key}")


def resolve_authoritative_clock_operator(
    row: Mapping[str, Any], cid: str, *, plan_pt: Mapping[str, Any] | None = None
) -> str:
    """Resolve one authoritative clock operator, enforcing plan agreement and rejecting conflicts."""
    boundary = row.get("boundary") if isinstance(row.get("boundary"), Mapping) else {}
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    plan_clk = str((plan_pt or {}).get("clock_operator") or "").strip()
    if plan_clk and plan_clk not in VALID_CLOCK_OPERATORS:
        raise ModelingError(f"clock_operator_invalid:{cid}:plan={plan_clk}")
    found: list[tuple[str, str]] = []
    for src, raw in (
        ("row", row.get("clock_operator")),
        ("boundary", boundary.get("clock_operator")),
        ("payload", payload.get("clock_operator")),
    ):
        val = str(raw or "").strip()
        if val:
            if val not in VALID_CLOCK_OPERATORS:
                raise ModelingError(f"clock_operator_invalid:{cid}:{src}={val}")
            found.append((src, val))
    if not found:
        if plan_clk:
            return plan_clk
        raise ModelingError(f"clock_operator_missing:{cid}")
    distinct = {val for _src, val in found}
    if len(distinct) > 1:
        detail = ",".join(f"{src}={val}" for src, val in found)
        raise ModelingError(f"clock_operator_conflict:{cid}:{detail}")
    chosen = found[0][1]
    if plan_clk and chosen != plan_clk:
        raise ModelingError(f"clock_operator_plan_mismatch:{cid}:plan={plan_clk}:got={chosen}")
    return chosen


def extract_measured_p99_ns(row: Mapping[str, Any]) -> int | None:
    """Extract capture-derived ``p99`` duration in ns if explicitly present on ``row``."""
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    for container in (row, payload):
        for key in ("measured_p99_ns", "silicon_p99_ns", "p99_ns"):
            val = container.get(key)
            if val is not None and not isinstance(val, bool):
                num = float(val)
                if math.isfinite(num) and num > 0.0:
                    return int(round(num))
    return None


def measured_ns_from_row(
    row: Mapping[str, Any],
    cid: str,
    *,
    clock_op: str,
    parsed_measured_ns: int | None = None,
) -> int:
    """Extract positive measured duration in ns and verify agreement with ``clock_op``."""
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    busy_raw = row.get("busy_union_ns") if row.get("busy_union_ns") is not None else payload.get("busy_union_ns")
    elapsed_raw = row.get("elapsed_span_ns") if row.get("elapsed_span_ns") is not None else payload.get("elapsed_span_ns")
    expected_from_clock = busy_raw if clock_op == "busy_union" else elapsed_raw
    if parsed_measured_ns is not None:
        val = float(parsed_measured_ns)
    elif row.get("measured_ns") is not None:
        val = float(row["measured_ns"])
    elif row.get("measured_us") is not None:
        val = float(row["measured_us"]) * NS_PER_US
    elif row.get("measured_ms") is not None:
        val = float(row["measured_ms"]) * NS_PER_MS
    elif expected_from_clock is not None:
        val = float(expected_from_clock)
    else:
        raise ModelingError(f"observation_measured_duration_missing:{cid}")
    if not math.isfinite(val) or val <= 0.0:
        raise ModelingError(f"observation_measured_duration_nonpositive:{cid}:{val}")
    measured_int = int(round(val))
    if busy_raw is not None and elapsed_raw is not None and not isinstance(busy_raw, bool) and not isinstance(elapsed_raw, bool):
        b_int = int(round(float(busy_raw)))
        e_int = int(round(float(elapsed_raw)))
        if b_int <= 0 or e_int <= 0 or b_int > e_int:
            raise ModelingError(
                f"observation_busy_union_exceeds_elapsed_span:{cid}:busy={b_int}:elapsed={e_int}"
            )
    if expected_from_clock is not None:
        exp_int = int(round(float(expected_from_clock)))
        if measured_int != exp_int:
            raise ModelingError(
                f"clock_operator_measured_value_mismatch:{cid}:op={clock_op}"
                f":measured_ns={measured_int}:declared_{clock_op}_ns={exp_int}"
            )
    return measured_int


def validate_event_ids_and_ranks(raw_events: Any, raw_ranks: Any, cid: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Require non-empty, non-blank event IDs and non-negative integer device ranks."""
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)) or not raw_events:
        raise ModelingError(f"observation_event_ids_missing:{cid}")
    cleaned_events = tuple(str(e).strip() for e in raw_events)
    if any(not e for e in cleaned_events):
        raise ModelingError(f"observation_event_id_blank:{cid}")
    if not isinstance(raw_ranks, Sequence) or isinstance(raw_ranks, (str, bytes)) or not raw_ranks:
        raise ModelingError(f"observation_device_ranks_missing:{cid}")
    cleaned_ranks: list[int] = []
    for r in raw_ranks:
        if isinstance(r, bool) or not isinstance(r, int) or r < 0:
            raise ModelingError(f"observation_device_rank_invalid:{cid}:{r!r}")
        cleaned_ranks.append(r)
    return cleaned_events, tuple(cleaned_ranks)


def validate_row_matches_plan_point(
    row: Mapping[str, Any],
    plan_pt: Mapping[str, Any],
    *,
    sku: str,
    pid: str,
    device_ranks: Sequence[int] = (),
    expected_level: str = "",
) -> None:
    """Verify every planned coordinate, level, & multi-rank participation matches the observation."""
    from .cliff_planner import validate_parallel_topology

    cohort = row.get("cohort") if isinstance(row.get("cohort"), Mapping) else {}
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    if expected_level:
        row_lvl = str(row.get("level") or cohort.get("level") or "").strip().upper()
        if row_lvl != expected_level:
            raise ModelingError(
                f"observation_plan_level_mismatch:{pid}:expected={expected_level}:got={row_lvl}"
            )
    row_sku = str(row.get("sku") or cohort.get("sku") or "").strip()
    if row_sku and row_sku != sku:
        raise ModelingError(f"observation_sku_mismatch:{pid}:expected={sku}:got={row_sku}")
    axis = str(plan_pt.get("axis") or "").strip()
    expected_val = int(plan_pt["value"])
    if axis:
        got_axis = row.get(axis) if row.get(axis) is not None else payload.get(axis)
        if got_axis is None or isinstance(got_axis, bool) or int(got_axis) != expected_val:
            raise ModelingError(f"observation_plan_coordinate_mismatch:{pid}:{axis}:expected={expected_val}:got={got_axis}")
        if axis == "context_len":
            for pkey in ("context_len", "context"):
                if payload.get(pkey) is not None and int(payload[pkey]) != expected_val:
                    raise ModelingError(f"observation_payload_coordinate_mismatch:{pid}:{pkey}:expected={expected_val}:got={payload[pkey]}")
    for field, exp in (("batch_size", int(plan_pt.get("batch_size", 1))), ("query_tokens", int(plan_pt.get("query_tokens", 1)))):
        got = row.get(field) if row.get(field) is not None else payload.get(field)
        if got is None or isinstance(got, bool) or int(got) != exp:
            raise ModelingError(f"observation_plan_coordinate_mismatch:{pid}:{field}:expected={exp}:got={got}")
    exp_topo = str(plan_pt.get("parallel_topology") or "tp1")
    got_topo = str(row.get("parallel_topology") or cohort.get("parallel_topology") or payload.get("parallel_topology") or "").strip()
    if got_topo != exp_topo:
        raise ModelingError(f"observation_plan_coordinate_mismatch:{pid}:parallel_topology:expected={exp_topo}:got={got_topo}")
    exp_ranks = validate_parallel_topology(exp_topo)
    if exp_ranks > 1 and device_ranks:
        unique_ranks = set(int(r) for r in device_ranks)
        if len(unique_ranks) != exp_ranks or len(device_ranks) != exp_ranks:
            raise ModelingError(
                f"observation_participating_ranks_mismatch:{pid}:topology={exp_topo}:expected_ranks={exp_ranks}:got={list(device_ranks)}"
            )
    if plan_pt.get("topk") is not None:
        got_topk = row.get("topk") if row.get("topk") is not None else payload.get("topk")
        if got_topk is None or isinstance(got_topk, bool) or int(got_topk) != int(plan_pt["topk"]):
            raise ModelingError(f"observation_plan_coordinate_mismatch:{pid}:topk:expected={plan_pt['topk']}:got={got_topk}")


__all__ = [
    "REQUIRED_MULTILEVEL_LEVELS",
    "STANDARD_LEVEL_BOUNDARY_KINDS",
    "_ALLOWED_OK_STATUSES",
    "_check_no_caller_predictions",
    "_norm_digest",
    "exceeds_physical_tolerance",
    "extract_measured_p99_ns",
    "measured_ns_from_row",
    "resolve_authoritative_clock_operator",
    "validate_disjoint_lineage",
    "validate_event_ids_and_ranks",
    "validate_row_matches_plan_point",
]
