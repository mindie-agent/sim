"""Sealed GateCase -> EvaluationRunner -> EngineRunReceipt scoring with clock parity.

[Role] Execute Prevention Lock 2 white-box scoring strictly through
  ``mindie_sim.observations.multilevel_gate.evaluate`` with live HMAC-signed
  ``EngineRunReceipt`` handles bound to the real workload Engine, one
  authoritative clock operator (``busy_union`` vs ``elapsed_span``) applied
  identically across capture and Engine intervals, cryptographic train-vs-
  heldout capture digest isolation, strict 1:1 planned-grid coordinate matching
  for BOTH ``train`` and ``heldout`` splits, operator family/implementation
  identity binding, and mandatory ``<= 10.0%`` pointwise threshold enforcement.
[Invariants] Structurally forbids caller-supplied ``predicted_*`` fields,
  ``service_regimes`` crutches, ``over_budget`` substitutions, missing train
  or heldout planned cells, conflicting/missing clock labels, reversed-event
  capture duplicates, dimension collapses, and thresholds ``> 10.0%``.
[Timeouts & Locks] Pure in-memory scoring and live Engine replay; zero
  subprocesses, network calls, or device contexts.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import statistics
from typing import Any

from mindie_sim.observations.multilevel_gate import (
    CaptureBoundary, CohortKey, EngineInterval, EvaluationRunner,
    FrozenFitManifest, FrozenFitParameter, GateCase, Manifest,
    evaluate as evaluate_multilevel, parse_gate_case, validate_lineage,
)
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError

from ._capture_truth import build_content_digest_index, verify_capture_and_override_case
from ._lean_and_receipts import validate_strict_threshold_pct
from ._whitebox_runner import WhiteboxSpineRunner, check_no_service_regimes
from .contracts import (
    DEFAULT_CLOCK_OPERATOR, DEFAULT_HELDOUT_APE_THRESHOLD_PCT, DISPLAY_ROUND_DIGITS,
    FORBIDDEN_CALLER_PREDICTION_KEYS, NS_PER_MS, NS_PER_US, PERCENT_SCALE,
    VALID_CLOCK_OPERATORS, assert_sanitized_payload, sanitize_public_alias,
    sanitize_public_capture_uri, sanitize_public_run_id,
)

STANDARD_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {
    "L1": "device_kernel",
    "L2": "fx_region",
    "L3": "model_step",
}
REQUIRED_MULTILEVEL_LEVELS = ("L1", "L2", "L3")
_P90_QUANTILE = 0.90
_ALLOWED_OK_STATUSES = frozenset({"", "ok", "pass", "passed", "success"})


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
    row: Mapping[str, Any], plan_pt: Mapping[str, Any], *, sku: str, pid: str, device_ranks: Sequence[int] = ()
) -> None:
    """Verify every planned coordinate & multi-rank participation matches the observation."""
    from .cliff_planner import validate_parallel_topology

    cohort = row.get("cohort") if isinstance(row.get("cohort"), Mapping) else {}
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
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


def _row_to_gate_case(row: Mapping[str, Any], *, sku: str, idx: int, plan_pt: Mapping[str, Any] | None = None) -> GateCase:
    cid = str(row.get("case_id") or row.get("point_id") or f"obs-{idx}")
    plan_axis = str(plan_pt.get("axis") or "") if plan_pt else ""
    plan_value = int(plan_pt["value"]) if plan_pt and plan_pt.get("value") is not None else None
    _check_no_caller_predictions(row, cid)
    check_no_service_regimes(row, path=cid)
    row_status = str(row.get("status") or "").strip().lower()
    if row_status not in _ALLOWED_OK_STATUSES:
        raise ModelingError(f"observation_status_not_ok:{cid}:status={row_status}")
    if row.get("validated") is False or row.get("qualified") is False:
        raise ModelingError(f"observation_unvalidated_or_unqualified:{cid}")
    clock_op = resolve_authoritative_clock_operator(row, cid, plan_pt=plan_pt)
    coord_keys = (
        "sim_case", "target_op_id", "target_family", "target_layer", "target_op_ids",
        "measured_binding", "actual_shape", "plan_key",
        "context_len", "tokens", "batch_size", "query_tokens", "batch_x_kv_heads", "parallel_topology", "topk",
    )
    if isinstance(row.get("boundary"), Mapping) and isinstance(row.get("cohort"), Mapping):
        raw_copy = dict(row)
        payload = dict(raw_copy.get("payload") or {})
        payload["clock_operator"] = clock_op
        if plan_axis:
            payload.setdefault("axis", plan_axis)
        if plan_value is not None:
            payload.setdefault("value", plan_value)
        for extra in coord_keys:
            if row.get(extra) is not None:
                if extra in payload and payload[extra] != row[extra]:
                    raise ModelingError(f"observation_payload_coordinate_mismatch:{cid}:{extra}")
                payload[extra] = row[extra]
        raw_copy["payload"] = payload
        gc = parse_gate_case(raw_copy)
        if gc.cohort.sku != sku:
            raise ModelingError(f"observation_sku_mismatch:{cid}:expected={sku}:got={gc.cohort.sku}")
        validate_event_ids_and_ranks(gc.boundary.event_ids, gc.boundary.device_ranks, cid)
        measured_ns_from_row(row, cid, clock_op=clock_op, parsed_measured_ns=gc.measured_ns)
        return gc

    row_sku = str(row.get("sku") or "").strip()
    if row_sku and row_sku != sku:
        raise ModelingError(f"observation_sku_mismatch:{cid}:expected={sku}:got={row_sku}")
    level = str(row.get("level") or "L1").strip().upper()
    if level not in STANDARD_LEVEL_BOUNDARY_KINDS:
        raise ModelingError(f"observation_level_invalid:{cid}:{level}")
    split = str(row.get("split") or "heldout").strip().lower()
    acq = str(row.get("acquisition_uid") or row.get("acquisition_root") or "").strip()
    run_uid = str(row.get("run_uid") or "").strip()
    raw_sha = str(row.get("raw_sha256") or "").strip()
    if not acq:
        raise ModelingError(f"observation_acquisition_root_missing:{cid}")
    if not run_uid:
        raise ModelingError(f"observation_run_uid_missing:{cid}")
    if not raw_sha:
        raise ModelingError(f"observation_raw_sha256_missing:{cid}")
    cleaned_events, cleaned_ranks = validate_event_ids_and_ranks(row.get("event_ids"), row.get("device_ranks"), cid)
    measured_ns = measured_ns_from_row(row, cid, clock_op=clock_op)
    payload_dict: dict[str, Any] = {"clock_operator": clock_op, "point_id": str(row.get("point_id") or cid)}
    if plan_axis:
        payload_dict["axis"] = plan_axis
    if plan_value is not None:
        payload_dict["value"] = plan_value
    for extra in (*coord_keys, "phase", "sequences"):
        if row.get(extra) is not None:
            payload_dict[extra] = row[extra]
    boundary = CaptureBoundary(
        raw_sha256=raw_sha, acquisition_uid=acq, run_uid=run_uid,
        parent_digests=tuple(str(p) for p in (row.get("parent_digests") or ())),
        event_ids=cleaned_events, device_ranks=cleaned_ranks,
        clock_domain=str(row.get("clock_domain") or "device_monotonic"),
        start_event=str(row.get("start_event") or "k_start"), end_event=str(row.get("end_event") or "k_end"),
        boundary_kind=str(row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS[level]),
        unit="ns", capture_uri=str(row.get("capture_uri") or f"capture://onboarding/{acq}/{raw_sha}"),
    )
    cohort = CohortKey(
        level=level, cohort_name=str(row.get("cohort_name") or row.get("regime_tag") or "onboarding_cliff"),  # type: ignore[arg-type]
        sku=sku, phase=str(row.get("phase") or "decode"), mode=str(row.get("mode") or "eager"),
        shape_regime=str(row.get("shape_regime") or "cliff"), parallel_topology=str(row.get("parallel_topology") or "tp1"),
    )
    regime_dims = tuple(sorted((str(k), str(v)) for k, v in (row.get("regime_dims") or {"clock_operator": clock_op}).items()))
    return GateCase(
        case_id=cid, level=level, split=split, boundary=boundary,  # type: ignore[arg-type]
        input_digest=str(row.get("input_digest") or f"in-{cid}"),
        stack_digest=str(row.get("stack_digest") or "stack-whitebox"),
        implementation=str(row.get("implementation") or "whitebox_spine"),
        cohort=cohort, measured_ns=measured_ns, regime_dims=regime_dims,
        payload=tuple(sorted(payload_dict.items())),
    )


class ClockParityRunner:
    """Wrapper enforcing authoritative clock parity, scored clock interval, & HMAC receipt."""

    def __init__(
        self,
        inner: EvaluationRunner,
        expected_clock: Mapping[str, str],
        *,
        expected_family: str = "",
        plan_by_case: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._inner = inner
        self._expected_clock = expected_clock
        self._expected_family = expected_family.strip()
        self._plan_by_case = dict(plan_by_case or {})
        self.engine_clocks: dict[str, dict[str, Any]] = {}

    def _check(self, case: GateCase, interval: EngineInterval) -> EngineInterval:
        diag = dict(interval.diagnostics)
        eng_op = str(diag.get("clock_operator") or getattr(interval, "clock_operator", "") or "").strip()
        sil_op = self._expected_clock[case.case_id]
        if not eng_op:
            raise ModelingError(f"engine_clock_operator_missing:{case.case_id}")
        if eng_op != sil_op:
            raise ModelingError(f"clock_operator_mismatch:{case.case_id}:silicon={sil_op}:engine={eng_op}")
        if interval.run_receipt is None:
            raise ModelingError(f"engine_run_receipt_missing:{case.case_id}")

        busy_raw = diag.get("busy_union_ns")
        if busy_raw is None:
            busy_raw = getattr(interval, "busy_union_ns", None)
        elapsed_raw = diag.get("elapsed_span_ns")
        if elapsed_raw is None:
            elapsed_raw = getattr(interval, "elapsed_span_ns", None)
        busy_int = int(round(float(busy_raw))) if busy_raw is not None and not isinstance(busy_raw, bool) else None
        elapsed_int = int(round(float(elapsed_raw))) if elapsed_raw is not None and not isinstance(elapsed_raw, bool) else int(interval.predicted_ns)

        if eng_op == "busy_union":
            if busy_int is None or busy_int <= 0:
                raise ModelingError(f"engine_busy_union_missing:{case.case_id}")
            if hasattr(interval, "clock_operator") and int(interval.predicted_ns) != busy_int:
                raise ModelingError(
                    f"engine_clock_operator_value_mismatch:{case.case_id}:op=busy_union"
                    f":predicted_ns={interval.predicted_ns}:diag={busy_int}"
                )
            scored_pred_ns = busy_int
        else:
            if elapsed_int != int(interval.predicted_ns):
                raise ModelingError(
                    f"engine_clock_operator_value_mismatch:{case.case_id}:op=elapsed_span"
                    f":predicted_ns={interval.predicted_ns}:diag={elapsed_int}"
                )
            scored_pred_ns = int(interval.predicted_ns)

        exec_families = diag.get("executed_op_families")
        if self._expected_family and isinstance(exec_families, (list, tuple)) and exec_families:
            if self._expected_family not in exec_families:
                raise ModelingError(
                    f"operator_family_replay_mismatch:{case.case_id}:expected={self._expected_family}:executed={list(exec_families)}"
                )

        lowered = diag.get("lowered_dimensions")
        plan_pt = self._plan_by_case.get(case.case_id)
        if isinstance(lowered, Mapping):
            mat_sku = str(lowered.get("materialized_sku") or "").strip()
            if mat_sku and mat_sku != case.cohort.sku:
                raise ModelingError(f"whitebox_materialized_sku_mismatch:{case.case_id}:expected={case.cohort.sku}:materialized={mat_sku}")
            if plan_pt is not None:
                axis = str(plan_pt.get("axis") or "")
                if axis == "batch_x_kv_heads" and lowered.get("batch_x_kv_heads") != int(plan_pt["value"]):
                    raise ModelingError(f"whitebox_dimension_collapse:{case.case_id}:batch_x_kv_heads:expected={plan_pt['value']}:lowered={lowered.get('batch_x_kv_heads')}")
                if axis == "tokens" and lowered.get("tokens") != int(plan_pt["value"]):
                    raise ModelingError(f"whitebox_dimension_collapse:{case.case_id}:tokens:expected={plan_pt['value']}:lowered={lowered.get('tokens')}")
                if axis != "batch_x_kv_heads" and lowered.get("sequences") is not None and int(lowered["sequences"]) != int(plan_pt.get("batch_size", 1)):
                    raise ModelingError(f"whitebox_dimension_collapse:{case.case_id}:sequences:expected={plan_pt.get('batch_size', 1)}:lowered={lowered.get('sequences')}")

        self.engine_clocks[case.case_id] = {
            "clock_operator": eng_op,
            "busy_union_ns": busy_int,
            "elapsed_span_ns": elapsed_int,
            "scored_predicted_ns": scored_pred_ns,
            "program_dag_sha256": str(diag.get("program_dag_sha256") or ""),
            "executed_op_families": list(exec_families) if isinstance(exec_families, (list, tuple)) else [],
            "lowered_dimensions": dict(lowered) if isinstance(lowered, Mapping) else None,
            "engine_uid": str(interval.run_receipt.engine_uid),
        }
        return interval

    def run_l1(self, case: GateCase) -> EngineInterval: return self._check(case, self._inner.run_l1(case))
    def run_l2(self, case: GateCase) -> EngineInterval: return self._check(case, self._inner.run_l2(case))
    def run_l3(self, case: GateCase) -> EngineInterval: return self._check(case, self._inner.run_l3(case))


def _summarize_apes(apes_pct: Sequence[float], signed_pct: Sequence[float]) -> dict[str, Any]:
    if not apes_pct:
        return {"count": 0, "mape_pct": None, "medape_pct": None, "p90_ape_pct": None, "max_ape_pct": None, "median_signed_pct": None}
    ordered = sorted(apes_pct)
    p90_idx = min(len(ordered) - 1, max(0, math.ceil(_P90_QUANTILE * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "mape_pct": round(float(statistics.fmean(ordered)), DISPLAY_ROUND_DIGITS),
        "medape_pct": round(float(statistics.median(ordered)), DISPLAY_ROUND_DIGITS),
        "p90_ape_pct": round(float(ordered[p90_idx]), DISPLAY_ROUND_DIGITS),
        "max_ape_pct": round(float(ordered[-1]), DISPLAY_ROUND_DIGITS),
        "median_signed_pct": round(float(statistics.median(signed_pct)), DISPLAY_ROUND_DIGITS),
    }


def _bind_active_runner(
    runner: EvaluationRunner | None,
    *,
    primitive_table: PrimitiveTable | None,
    profile_override: Mapping[str, Any] | None,
) -> EvaluationRunner:
    if runner is None:
        return WhiteboxSpineRunner(primitive_table=primitive_table, profile_override=profile_override)
    if isinstance(runner, WhiteboxSpineRunner) and (
        (primitive_table is not None and runner.primitive_table is None)
        or (profile_override is not None and runner.profile_override is None)
    ):
        return WhiteboxSpineRunner(
            clock_operator=runner.clock_operator,
            primitive_table=runner.primitive_table if runner.primitive_table is not None else primitive_table,
            profile_override=runner.profile_override if runner.profile_override is not None else profile_override,
        )
    return runner


def score_sealed_gate_cases(
    observations: Sequence[Mapping[str, Any]],
    *,
    sku: str,
    runner: EvaluationRunner | None = None,
    primitive_table: PrimitiveTable | None = None,
    profile_override: Mapping[str, Any] | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    external_train_roots: Sequence[str] = (),
    external_train_digests: Sequence[str] = (),
    external_train_uris: Sequence[str] = (),
    expected_points: Sequence[Mapping[str, Any]] = (),
    expected_train_points: Sequence[Mapping[str, Any]] = (),
    expected_point_ids: Sequence[str] = (),
    expected_family: str = "",
    expected_implementations: Sequence[str] = (),
    require_disjoint_level_roots: bool = True,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = False,
) -> dict[str, Any]:
    """Score observations strictly via ``GateCase -> EvaluationRunner -> EngineRunReceipt``."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    plan_by_id: dict[str, Mapping[str, Any]] = {
        str(pt["point_id"]): pt for pt in expected_points if isinstance(pt, Mapping) and pt.get("point_id")
    }
    train_plan_by_id: dict[str, Mapping[str, Any]] = {
        str(pt["point_id"]): pt for pt in expected_train_points if isinstance(pt, Mapping) and pt.get("point_id")
    }
    all_expected_ids = sorted(set(expected_point_ids) | set(plan_by_id))
    all_expected_train_ids = sorted(train_plan_by_id)
    if not observations:
        return {
            "ok": False, "status": "insufficient_evidence", "sku": sku,
            "threshold_pct": cap_threshold,
            "lineage": {"train_roots": sorted(set(external_train_roots)), "heldout_roots": [], "level_roots": {}, "disjoint_ok": False},
            "heldout_summary": _summarize_apes([], []), "train_summary": _summarize_apes([], []),
            "missing_plan_points": all_expected_ids,
            "missing_train_points": all_expected_train_ids,
            "over_budget_points": [], "failing_cases": [], "residuals": [],
        }

    over_budget_rows: list[dict[str, Any]] = []
    scorable_rows: list[Mapping[str, Any]] = []
    for idx, row in enumerate(observations):
        cid = str(row.get("case_id") or row.get("point_id") or f"obs-{idx}")
        _check_no_caller_predictions(row, cid)
        if str(row.get("status") or "").strip().lower() == "over_budget":
            over_budget_rows.append({"case_id": cid, "point_id": str(row.get("point_id") or cid), "status": "over_budget", "executed_shape": row.get("executed_shape")})
            continue
        scorable_rows.append(row)

    lineage = validate_disjoint_lineage(
        scorable_rows,
        external_train_roots=external_train_roots,
        external_train_digests=external_train_digests,
        external_train_uris=external_train_uris,
        require_disjoint_level_roots=require_disjoint_level_roots,
    )
    allowed_impls = {str(x).strip() for x in expected_implementations if str(x).strip()}
    exp_fam = expected_family.strip()
    cases: list[GateCase] = []
    expected_clock: dict[str, str] = {}
    case_meta: dict[str, dict[str, Any]] = {}
    plan_by_case: dict[str, Mapping[str, Any]] = {}
    seen_heldout_ids: set[str] = set()
    seen_train_ids: set[str] = set()
    seen_heldout_captures: dict[tuple[str, frozenset[str], tuple[int, ...], str], str] = {}
    captures_verified_flags: list[bool] = []
    digest_index = build_content_digest_index(Path(capture_root)) if capture_root is not None else None
    resolved_files: dict[str, str | Path] = dict(capture_files or {})

    for idx, row in enumerate(scorable_rows):
        raw_pid = str(row.get("point_id") or row.get("case_id") or f"obs-{idx}").strip()
        plan_pt = plan_by_id.get(raw_pid) or train_plan_by_id.get(raw_pid)
        gc = _row_to_gate_case(row, sku=sku, idx=idx, plan_pt=plan_pt)
        pid = str(row.get("point_id") or gc.case_id).strip()
        payload_map = dict(gc.payload)
        clock_op = str(payload_map.get("clock_operator", DEFAULT_CLOCK_OPERATOR))
        gc, cap_verified, auth_p99 = verify_capture_and_override_case(
            gc, row, clock_op=clock_op,
            capture_root=capture_root, capture_files=capture_files,
            require_capture_files=require_capture_files,
            row_p99_ns=extract_measured_p99_ns(row),
            digest_index=digest_index,
            resolved_files=resolved_files,
        )
        captures_verified_flags.append(cap_verified)
        row_fam = str(row.get("target_family") or payload_map.get("target_family") or row.get("op_family") or "").strip()
        if exp_fam and row_fam and row_fam != exp_fam:
            raise ModelingError(f"operator_family_replay_mismatch:{pid}:expected={exp_fam}:got={row_fam}")
        if allowed_impls and gc.implementation not in allowed_impls:
            raise ModelingError(f"operator_implementation_mismatch:{pid}:got={gc.implementation}:allowed={sorted(allowed_impls)}")
        if gc.split == "heldout":
            if pid in seen_heldout_ids:
                raise ModelingError(f"duplicate_heldout_point_id:{pid}")
            seen_heldout_ids.add(pid)
            if gc.case_id != pid and gc.case_id in plan_by_id and pid in plan_by_id:
                raise ModelingError(f"conflicting_case_and_point_id:{gc.case_id}:{pid}")
            if pid in plan_by_id:
                validate_row_matches_plan_point(row, plan_by_id[pid], sku=sku, pid=pid, device_ranks=gc.boundary.device_ranks)
                plan_by_case[gc.case_id] = plan_by_id[pid]
            cap_key = (_norm_digest(gc.boundary.raw_sha256), frozenset(gc.boundary.event_ids), gc.boundary.device_ranks, gc.boundary.run_uid)
            if cap_key in seen_heldout_captures:
                raise ModelingError(f"duplicate_heldout_physical_capture:{pid}:collides_with={seen_heldout_captures[cap_key]}")
            seen_heldout_captures[cap_key] = pid
        elif gc.split == "train":
            if pid in seen_train_ids:
                raise ModelingError(f"duplicate_train_point_id:{pid}")
            seen_train_ids.add(pid)
            if pid in train_plan_by_id:
                validate_row_matches_plan_point(row, train_plan_by_id[pid], sku=sku, pid=pid, device_ranks=gc.boundary.device_ranks)
                plan_by_case[gc.case_id] = train_plan_by_id[pid]
        cases.append(gc)
        expected_clock[gc.case_id] = clock_op
        case_meta[gc.case_id] = {
            "point_id": pid, "split": gc.split, "target_family": row_fam, "implementation": gc.implementation,
            "operator_id": str(row.get("operator_id") or payload_map.get("operator_id") or "").strip(),
            "upstream_abi": str(row.get("upstream_abi") or row.get("abi") or "").strip(),
            "silicon_busy_union_ns": row.get("busy_union_ns"), "silicon_elapsed_span_ns": row.get("elapsed_span_ns"),
            "measured_p99_ns": auth_p99, "raw_sha256": gc.boundary.raw_sha256, "capture_uri": gc.boundary.capture_uri,
            "run_uid": gc.boundary.run_uid, "event_ids": list(gc.boundary.event_ids), "device_ranks": list(gc.boundary.device_ranks),
            "start_event": gc.boundary.start_event, "end_event": gc.boundary.end_event,
            "clock_domain": gc.boundary.clock_domain, "parent_digests": list(gc.boundary.parent_digests),
        }

    manifest = Manifest(
        manifest_id=f"onboarding-{sanitize_public_alias(sku, 'sku')}",
        schema_version="mindie-multilevel-manifest/1",
        cases=tuple(cases),
        strict_three_level_blind=require_disjoint_level_roots,
    )
    all_fit_tokens = sorted(
        {str(t).strip() for t in (*external_train_roots, *external_train_digests) if str(t).strip()}
    )
    fit_params = tuple(
        FrozenFitParameter(parameter_id=f"train-tok-{i}", training_digests=(tok,))
        for i, tok in enumerate(all_fit_tokens)
    )
    fits = FrozenFitManifest("onboarding-frozen-fits", "mindie-frozen-fits/1", fit_params)
    eff_cap_files = resolved_files or capture_files
    validate_lineage(
        manifest, fits, capture_root=capture_root, capture_files=eff_cap_files,
        require_capture_files=require_capture_files,
    )
    active_runner = _bind_active_runner(
        runner, primitive_table=primitive_table, profile_override=profile_override
    )
    checked_runner = ClockParityRunner(
        active_runner, expected_clock, expected_family=exp_fam, plan_by_case=plan_by_case
    )
    raw_residuals = list(
        evaluate_multilevel(
            manifest, fits, checked_runner, target_split="all",
            capture_root=capture_root, capture_files=eff_cap_files,
            require_capture_files=require_capture_files,
        )
    )

    residual_dicts: list[dict[str, Any]] = []
    heldout_apes, heldout_signed, train_apes, train_signed = [], [], [], []
    failing_cases: list[str] = []
    scored_heldout_ids: set[str] = set()
    scored_train_ids: set[str] = set()
    for res in raw_residuals:
        meta = case_meta[res.case_id]
        clk = checked_runner.engine_clocks.get(res.case_id, {})
        scored_pred_ns = int(clk.get("scored_predicted_ns", res.predicted_ns))
        signed_err = (scored_pred_ns - res.measured_ns) / float(res.measured_ns) if res.measured_ns > 0 else res.signed_error
        ape = abs(signed_err)
        ape_pct = ape * PERCENT_SCALE
        signed_pct = signed_err * PERCENT_SCALE
        row_dict = {
            **res.to_dict(),
            "predicted_ns": scored_pred_ns, "signed_error": signed_err, "absolute_percentage_error": ape,
            "point_id": meta["point_id"], "split": meta["split"],
            "target_family": meta["target_family"], "implementation": meta["implementation"],
            "operator_id": meta["operator_id"], "upstream_abi": meta["upstream_abi"],
            "clock_operator": clk.get("clock_operator", expected_clock[res.case_id]),
            "silicon_busy_union_ns": meta["silicon_busy_union_ns"], "silicon_elapsed_span_ns": meta["silicon_elapsed_span_ns"],
            "measured_p99_ns": meta["measured_p99_ns"],
            "engine_busy_union_ns": clk.get("busy_union_ns"), "engine_elapsed_span_ns": clk.get("elapsed_span_ns"),
            "program_dag_sha256": clk.get("program_dag_sha256", ""),
            "executed_op_families": clk.get("executed_op_families", []), "engine_uid": clk.get("engine_uid", ""),
            "raw_sha256": meta["raw_sha256"],
            "capture_uri": sanitize_public_capture_uri(meta["capture_uri"], raw_sha256=meta["raw_sha256"]),
            "run_uid": sanitize_public_run_id(meta["run_uid"]),
            "event_ids": meta["event_ids"], "device_ranks": meta["device_ranks"],
            "start_event": meta["start_event"], "end_event": meta["end_event"],
            "clock_domain": meta["clock_domain"], "parent_digests": meta["parent_digests"],
        }
        residual_dicts.append(row_dict)
        if meta["split"] == "heldout" and res.status == "ok":
            scored_heldout_ids.add(meta["point_id"])
            heldout_apes.append(ape_pct)
            heldout_signed.append(signed_pct)
            if ape_pct >= cap_threshold:
                failing_cases.append(res.case_id)
        elif meta["split"] == "train" and res.status == "ok":
            scored_train_ids.add(meta["point_id"])
            train_apes.append(ape_pct)
            train_signed.append(signed_pct)

    missing_plan_points = sorted(set(all_expected_ids) - scored_heldout_ids)
    missing_train_points = sorted(set(all_expected_train_ids) - scored_train_ids)
    if not heldout_apes or missing_plan_points or missing_train_points:
        ok, status = False, "insufficient_evidence"
    elif failing_cases:
        ok, status = False, "fail"
    else:
        ok, status = True, "pass"

    result = {
        "ok": ok, "status": status, "sku": sku, "threshold_pct": cap_threshold,
        "capture_files_verified": bool(captures_verified_flags) and all(captures_verified_flags),
        "lineage": lineage,
        "heldout_summary": _summarize_apes(heldout_apes, heldout_signed),
        "train_summary": _summarize_apes(train_apes, train_signed),
        "missing_plan_points": missing_plan_points,
        "missing_train_points": missing_train_points,
        "over_budget_points": over_budget_rows,
        "failing_cases": sorted(failing_cases),
        "residuals": residual_dicts,
    }
    assert_sanitized_payload(result, field_name="sealed_whitebox_score_result")
    return result


from ._manifest_verifier import run_multilevel_verdict  # noqa: E402

__all__ = [
    "REQUIRED_MULTILEVEL_LEVELS",
    "STANDARD_LEVEL_BOUNDARY_KINDS",
    "ClockParityRunner",
    "WhiteboxSpineRunner",
    "extract_measured_p99_ns",
    "measured_ns_from_row",
    "resolve_authoritative_clock_operator",
    "run_multilevel_verdict",
    "score_sealed_gate_cases",
    "validate_disjoint_lineage",
    "validate_event_ids_and_ranks",
    "validate_row_matches_plan_point",
]
