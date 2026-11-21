"""Sealed GateCase -> EvaluationRunner -> EngineRunReceipt scoring and multilevel verdicts.

[Role] Execute Prevention Lock 2 white-box scoring strictly through
  ``mindie_sim.observations.multilevel_gate.evaluate`` with live HMAC-signed
  ``EngineRunReceipt`` handles, identical clock-operator value and label parity
  (``busy_union`` vs ``elapsed_span``), cryptographic train-vs-heldout capture
  digest isolation, strict 1:1 planned-grid coordinate matching (no
  double-crediting), mandatory ``<= 10.0%`` threshold enforcement, and
  pointwise release verdicts.
[Invariants] Structurally forbids caller-supplied ``predicted_*`` fields,
  ``service_regimes`` crutches, ``over_budget`` substitutions, missing planned
  cells, fabricated unverified reports, and thresholds ``> 10.0%``.
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
    CaptureBoundary,
    CohortKey,
    EngineInterval,
    EvaluationRunner,
    FrozenFitManifest,
    FrozenFitParameter,
    GateCase,
    Manifest,
    Residual,
    evaluate as evaluate_multilevel,
    gate_decision,
    parse_gate_case,
    stream_manifest,
    summarize_cohorts,
    validate_lineage,
)
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError

from ._lean_and_receipts import (
    validate_strict_threshold_pct,
    verify_stage_report,
)
from ._whitebox_runner import (
    WhiteboxSpineRunner,
    check_no_service_regimes,
)
from .contracts import (
    DEFAULT_CLOCK_OPERATOR,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DISPLAY_ROUND_DIGITS,
    FORBIDDEN_CALLER_PREDICTION_KEYS,
    NS_PER_MS,
    NS_PER_US,
    PERCENT_SCALE,
    VALID_CLOCK_OPERATORS,
    GateCheckOutcome,
    assert_sanitized_payload,
    read_bounded_json,
    sanitize_public_alias,
)

STANDARD_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {
    "L1": "device_kernel",
    "L2": "fx_region",
    "L3": "model_step",
}
REQUIRED_MULTILEVEL_LEVELS = ("L1", "L2", "L3")
_P90_QUANTILE = 0.90
_ENGINE_RECEIPT_PREFIX = "capture://engine-receipt/"


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


def _resolve_clock_operator(row: Mapping[str, Any], cid: str) -> str:
    boundary = row.get("boundary") if isinstance(row.get("boundary"), Mapping) else {}
    op = str(row.get("clock_operator") or boundary.get("clock_operator") or "").strip()
    if not op:
        raise ModelingError(f"clock_operator_missing:{cid}")
    if op not in VALID_CLOCK_OPERATORS:
        raise ModelingError(f"clock_operator_invalid:{cid}:{op}")
    return op


def _measured_ns_from_row(row: Mapping[str, Any], cid: str, *, clock_op: str) -> int:
    busy_raw = row.get("busy_union_ns")
    elapsed_raw = row.get("elapsed_span_ns")
    expected_from_clock = busy_raw if clock_op == "busy_union" else elapsed_raw
    if row.get("measured_ns") is not None:
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


def _validate_row_matches_plan_point(
    row: Mapping[str, Any],
    plan_pt: Mapping[str, Any],
    *,
    sku: str,
    pid: str,
) -> None:
    row_sku = str(row.get("sku") or "").strip()
    if row_sku and row_sku != sku:
        raise ModelingError(f"observation_sku_mismatch:{pid}:expected={sku}:got={row_sku}")
    axis = str(plan_pt.get("axis") or "")
    expected_val = int(plan_pt["value"])
    if axis == "context_len":
        got_ctx = row.get("context_len")
        if got_ctx is None or int(got_ctx) != expected_val:
            raise ModelingError(
                f"observation_plan_coordinate_mismatch:{pid}:context_len:expected={expected_val}:got={got_ctx}"
            )
    elif axis == "tokens":
        got_tok = row.get("tokens")
        if got_tok is None or int(got_tok) != expected_val:
            raise ModelingError(
                f"observation_plan_coordinate_mismatch:{pid}:tokens:expected={expected_val}:got={got_tok}"
            )
    exp_batch = int(plan_pt.get("batch_size", 1))
    got_batch = row.get("batch_size")
    if got_batch is None or int(got_batch) != exp_batch:
        raise ModelingError(
            f"observation_plan_coordinate_mismatch:{pid}:batch_size:expected={exp_batch}:got={got_batch}"
        )
    exp_q = int(plan_pt.get("query_tokens", 1))
    got_q = row.get("query_tokens")
    if got_q is None or int(got_q) != exp_q:
        raise ModelingError(
            f"observation_plan_coordinate_mismatch:{pid}:query_tokens:expected={exp_q}:got={got_q}"
        )
    exp_topo = str(plan_pt.get("parallel_topology") or "tp1")
    got_topo = str(row.get("parallel_topology") or "").strip()
    if got_topo != exp_topo:
        raise ModelingError(
            f"observation_plan_coordinate_mismatch:{pid}:parallel_topology:expected={exp_topo}:got={got_topo}"
        )
    exp_topk = plan_pt.get("topk")
    if exp_topk is not None:
        got_topk = row.get("topk")
        if got_topk is None or int(got_topk) != int(exp_topk):
            raise ModelingError(
                f"observation_plan_coordinate_mismatch:{pid}:topk:expected={exp_topk}:got={got_topk}"
            )


def _row_to_gate_case(row: Mapping[str, Any], *, sku: str, idx: int) -> GateCase:
    cid = str(row.get("case_id") or row.get("point_id") or f"obs-{idx}")
    _check_no_caller_predictions(row, cid)
    check_no_service_regimes(row, path=cid)
    clock_op = _resolve_clock_operator(row, cid)
    if isinstance(row.get("boundary"), Mapping) and isinstance(row.get("cohort"), Mapping):
        raw_copy = dict(row)
        payload = dict(raw_copy.get("payload") or {})
        payload["clock_operator"] = clock_op
        for extra in ("sim_case", "target_op_id", "target_family", "target_layer", "target_op_ids"):
            if row.get(extra) is not None:
                payload[extra] = row[extra]
        raw_copy["payload"] = payload
        gc = parse_gate_case(raw_copy)
        _measured_ns_from_row(row, cid, clock_op=clock_op)
        return gc

    level = str(row.get("level") or "L1").strip().upper()
    if level not in STANDARD_LEVEL_BOUNDARY_KINDS:
        raise ModelingError(f"observation_level_invalid:{cid}:{level}")
    split = str(row.get("split") or "heldout").strip().lower()
    acq = str(row.get("acquisition_uid") or row.get("acquisition_root") or "").strip()
    run_uid = str(row.get("run_uid") or "").strip()
    raw_sha = str(row.get("raw_sha256") or "").strip()
    raw_events = row.get("event_ids")
    raw_ranks = row.get("device_ranks")
    if not acq:
        raise ModelingError(f"observation_acquisition_root_missing:{cid}")
    if not run_uid:
        raise ModelingError(f"observation_run_uid_missing:{cid}")
    if not raw_sha:
        raise ModelingError(f"observation_raw_sha256_missing:{cid}")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)) or not raw_events:
        raise ModelingError(f"observation_event_ids_missing:{cid}")
    if not isinstance(raw_ranks, Sequence) or isinstance(raw_ranks, (str, bytes)) or not raw_ranks:
        raise ModelingError(f"observation_device_ranks_missing:{cid}")
    measured_ns = _measured_ns_from_row(row, cid, clock_op=clock_op)
    payload_dict: dict[str, Any] = {"clock_operator": clock_op, "point_id": str(row.get("point_id") or cid)}
    for extra in (
        "sim_case", "phase", "tokens", "context_len", "sequences",
        "batch_size", "query_tokens", "topk",
        "target_op_id", "target_family", "target_layer", "target_op_ids",
    ):
        if row.get(extra) is not None:
            payload_dict[extra] = row[extra]
    boundary = CaptureBoundary(
        raw_sha256=raw_sha,
        acquisition_uid=acq,
        run_uid=run_uid,
        parent_digests=tuple(str(p) for p in (row.get("parent_digests") or ())),
        event_ids=tuple(str(e) for e in raw_events),
        device_ranks=tuple(int(r) for r in raw_ranks),
        clock_domain=str(row.get("clock_domain") or "device_monotonic"),
        start_event=str(row.get("start_event") or "k_start"),
        end_event=str(row.get("end_event") or "k_end"),
        boundary_kind=str(row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS[level]),
        unit="ns",
        capture_uri=str(row.get("capture_uri") or f"capture://onboarding/{acq}/{raw_sha}"),
    )
    cohort = CohortKey(
        level=level,  # type: ignore[arg-type]
        cohort_name=str(row.get("cohort_name") or row.get("regime_tag") or "onboarding_cliff"),
        sku=sku,
        phase=str(row.get("phase") or "decode"),
        mode=str(row.get("mode") or "eager"),
        shape_regime=str(row.get("shape_regime") or "cliff"),
        parallel_topology=str(row.get("parallel_topology") or "tp1"),
    )
    regime_dims = tuple(
        sorted((str(k), str(v)) for k, v in (row.get("regime_dims") or {"clock_operator": clock_op}).items())
    )
    return GateCase(
        case_id=cid, level=level, split=split, boundary=boundary,  # type: ignore[arg-type]
        input_digest=str(row.get("input_digest") or f"in-{cid}"),
        stack_digest=str(row.get("stack_digest") or "stack-whitebox"),
        implementation=str(row.get("implementation") or "whitebox_spine"),
        cohort=cohort, measured_ns=measured_ns, regime_dims=regime_dims,
        payload=tuple(sorted(payload_dict.items())),
    )


class _ClockParityRunner:
    """Wrapper enforcing identical clock operator and HMAC EngineRunReceipt."""

    def __init__(self, inner: EvaluationRunner, expected_clock: Mapping[str, str]) -> None:
        self._inner = inner
        self._expected_clock = expected_clock
        self.engine_clocks: dict[str, dict[str, Any]] = {}

    def _check(self, case: GateCase, interval: EngineInterval) -> EngineInterval:
        diag = dict(interval.diagnostics)
        eng_op = str(diag.get("clock_operator") or "").strip()
        sil_op = self._expected_clock[case.case_id]
        if not eng_op:
            raise ModelingError(f"engine_clock_operator_missing:{case.case_id}")
        if eng_op != sil_op:
            raise ModelingError(f"clock_operator_mismatch:{case.case_id}:silicon={sil_op}:engine={eng_op}")
        if interval.run_receipt is None:
            raise ModelingError(f"engine_run_receipt_missing:{case.case_id}")
        self.engine_clocks[case.case_id] = {
            "clock_operator": eng_op,
            "busy_union_ns": diag.get("busy_union_ns"),
            "elapsed_span_ns": diag.get("elapsed_span_ns"),
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


def score_sealed_gate_cases(
    observations: Sequence[Mapping[str, Any]],
    *,
    sku: str,
    runner: EvaluationRunner | None = None,
    primitive_table: PrimitiveTable | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    external_train_roots: Sequence[str] = (),
    external_train_digests: Sequence[str] = (),
    external_train_uris: Sequence[str] = (),
    expected_points: Sequence[Mapping[str, Any]] = (),
    expected_point_ids: Sequence[str] = (),
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
    all_expected_ids = sorted(set(expected_point_ids) | set(plan_by_id))
    if not observations:
        return {
            "ok": False, "status": "insufficient_evidence", "sku": sku,
            "threshold_pct": cap_threshold,
            "lineage": {"train_roots": sorted(set(external_train_roots)), "heldout_roots": [], "level_roots": {}, "disjoint_ok": False},
            "heldout_summary": _summarize_apes([], []), "train_summary": _summarize_apes([], []),
            "missing_plan_points": all_expected_ids, "over_budget_points": [],
            "failing_cases": [], "residuals": [],
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
    cases: list[GateCase] = []
    expected_clock: dict[str, str] = {}
    case_meta: dict[str, dict[str, Any]] = {}
    seen_point_ids: set[str] = set()
    for idx, row in enumerate(scorable_rows):
        gc = _row_to_gate_case(row, sku=sku, idx=idx)
        pid = str(row.get("point_id") or gc.case_id).strip()
        if gc.split == "heldout":
            if pid in seen_point_ids:
                raise ModelingError(f"duplicate_heldout_point_id:{pid}")
            seen_point_ids.add(pid)
            if gc.case_id != pid and gc.case_id in plan_by_id and pid in plan_by_id:
                raise ModelingError(f"conflicting_case_and_point_id:{gc.case_id}:{pid}")
            if pid in plan_by_id:
                _validate_row_matches_plan_point(row, plan_by_id[pid], sku=sku, pid=pid)
        cases.append(gc)
        expected_clock[gc.case_id] = str(dict(gc.payload).get("clock_operator", DEFAULT_CLOCK_OPERATOR))
        case_meta[gc.case_id] = {
            "point_id": pid, "split": gc.split,
            "silicon_busy_union_ns": row.get("busy_union_ns"), "silicon_elapsed_span_ns": row.get("elapsed_span_ns"),
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
    validate_lineage(
        manifest, fits, capture_root=capture_root, capture_files=capture_files,
        require_capture_files=require_capture_files,
    )
    active_runner = (
        runner if runner is not None else WhiteboxSpineRunner(primitive_table=primitive_table)
    )
    checked_runner = _ClockParityRunner(active_runner, expected_clock)
    raw_residuals = list(
        evaluate_multilevel(
            manifest, fits, checked_runner, target_split="all",
            capture_root=capture_root, capture_files=capture_files,
            require_capture_files=require_capture_files,
        )
    )

    residual_dicts: list[dict[str, Any]] = []
    heldout_apes, heldout_signed, train_apes, train_signed = [], [], [], []
    failing_cases: list[str] = []
    scored_heldout_ids: set[str] = set()
    for res in raw_residuals:
        meta = case_meta[res.case_id]
        clk = checked_runner.engine_clocks.get(res.case_id, {})
        ape_pct = res.absolute_percentage_error * PERCENT_SCALE
        signed_pct = res.signed_error * PERCENT_SCALE
        residual_dicts.append({
            **res.to_dict(), "point_id": meta["point_id"], "split": meta["split"],
            "clock_operator": clk.get("clock_operator", expected_clock[res.case_id]),
            "silicon_busy_union_ns": meta["silicon_busy_union_ns"],
            "silicon_elapsed_span_ns": meta["silicon_elapsed_span_ns"],
            "engine_busy_union_ns": clk.get("busy_union_ns"),
            "engine_elapsed_span_ns": clk.get("elapsed_span_ns"),
        })
        if meta["split"] == "heldout" and res.status == "ok":
            scored_heldout_ids.add(meta["point_id"])
            heldout_apes.append(ape_pct)
            heldout_signed.append(signed_pct)
            if ape_pct >= cap_threshold:
                failing_cases.append(res.case_id)
        elif meta["split"] == "train" and res.status == "ok":
            train_apes.append(ape_pct)
            train_signed.append(signed_pct)

    missing_plan_points = sorted(set(all_expected_ids) - scored_heldout_ids)
    if not heldout_apes or missing_plan_points:
        ok, status = False, "insufficient_evidence"
    elif failing_cases:
        ok, status = False, "fail"
    else:
        ok, status = True, "pass"

    result = {
        "ok": ok, "status": status, "sku": sku, "threshold_pct": cap_threshold,
        "lineage": lineage, "heldout_summary": _summarize_apes(heldout_apes, heldout_signed),
        "train_summary": _summarize_apes(train_apes, train_signed),
        "missing_plan_points": missing_plan_points, "over_budget_points": over_budget_rows,
        "failing_cases": sorted(failing_cases), "residuals": residual_dicts,
    }
    assert_sanitized_payload(result, field_name="sealed_whitebox_score_result")
    return result


def run_multilevel_verdict(
    *,
    sku: str,
    validation_doc: Mapping[str, Any] | None,
    multilevel_manifest_json: str | Path | None,
    fits_json: str | Path | None,
    runner: EvaluationRunner | None,
    primitive_table: PrimitiveTable | None = None,
    threshold_pct: float,
) -> tuple[GateCheckOutcome, list[dict[str, Any]]]:
    """Evaluate strict pointwise multilevel L1/L2/L3 gate verdict."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    if multilevel_manifest_json is not None:
        manifest = stream_manifest(multilevel_manifest_json)
        if fits_json is not None:
            raw_fits = read_bounded_json(fits_json)
            params = tuple(
                FrozenFitParameter(str(p["parameter_id"]), tuple(str(d) for d in p.get("training_digests", ())), str(p.get("source_uri") or ""))
                for p in raw_fits.get("parameters", ())
            )
            fits = FrozenFitManifest(str(raw_fits.get("fit_id") or "onboard-fit"), str(raw_fits.get("schema_version") or "mindie-frozen-fits/1"), params, tuple((str(e[0]), str(e[1])) for e in raw_fits.get("lineage_edges", ())))
        else:
            fits = FrozenFitManifest("onboard-empty", "mindie-frozen-fits/1", ())
        validate_lineage(manifest, fits)
        expected_clock = {
            c.case_id: str(dict(c.payload).get("clock_operator") or DEFAULT_CLOCK_OPERATOR)
            for c in manifest.cases
        }
        active_runner = runner if runner is not None else WhiteboxSpineRunner(primitive_table=primitive_table)
        checked_runner = _ClockParityRunner(active_runner, expected_clock)
        residuals = list(evaluate_multilevel(manifest, fits, checked_runner, target_split="heldout"))
        summaries = summarize_cohorts(residuals, threshold_pct=cap_threshold)
        verdict = gate_decision(summaries, required_levels=REQUIRED_MULTILEVEL_LEVELS, require_disjoint_level_roots=True)
        pw_fail = sorted(r.case_id for r in residuals if r.status == "ok" and r.absolute_percentage_error * PERCENT_SCALE >= cap_threshold)
        passed = verdict["status"] == "pass" and not pw_fail
        return (
            GateCheckOutcome(
                name="multilevel_gate", passed=passed, status="pass" if passed else ("fail" if pw_fail else verdict["status"]),
                detail=f"status={verdict['status']};levels={verdict['covered_levels']};missing={verdict['missing_levels']};pointwise_failures={len(pw_fail)}",
                metrics=(("covered_levels", verdict["covered_levels"]), ("missing_levels", verdict["missing_levels"]), ("overlapping_level_roots", verdict["overlapping_level_roots"]), ("failed_cohorts", verdict["failed_cohorts"]), ("pointwise_failures", pw_fail), ("threshold_pct", cap_threshold)),
            ),
            [r.to_dict() for r in residuals],
        )

    if validation_doc is None:
        return GateCheckOutcome("multilevel_gate", False, "insufficient_evidence", "no validation report or multilevel manifest supplied"), []
    if not verify_stage_report("sku_validation", validation_doc):
        return GateCheckOutcome("multilevel_gate", False, "fail", "validation_report_hmac_unverified_or_fabricated"), []
    if str(validation_doc.get("sku") or "") != sku:
        return GateCheckOutcome("multilevel_gate", False, "fail", f"validation_report_sku_mismatch:expected={sku}"), []

    heldout_rows = [r for r in (validation_doc.get("residuals") or ()) if str(r.get("split") or "heldout") == "heldout"]
    missing_plan = list(validation_doc.get("missing_plan_points") or ())
    missing_anchors = list(validation_doc.get("missing_anchors") or ())
    missing_cliffs = list(validation_doc.get("missing_cliffs") or ())
    abi_ok = bool((validation_doc.get("abi_rejection_check") or {}).get("ok", False))
    if not heldout_rows:
        return GateCheckOutcome("multilevel_gate", False, "insufficient_evidence", "validation report contains zero heldout residuals"), []
    for row in heldout_rows:
        rcpt = str(row.get("engine_receipt") or "")
        if not rcpt.startswith(_ENGINE_RECEIPT_PREFIX):
            return GateCheckOutcome("multilevel_gate", False, "fail", f"unverified_engine_receipt:{row.get('case_id')}"), heldout_rows
    validate_disjoint_lineage(heldout_rows, require_disjoint_level_roots=True)
    res_objs: list[Residual] = []
    pw_fail: list[str] = []
    for row in heldout_rows:
        lvl = str(row.get("level") or "L1").upper()
        c_raw = row.get("cohort") if isinstance(row.get("cohort"), Mapping) else {}
        cohort = CohortKey(lvl, str(c_raw.get("cohort_name") or "onboarding"), str(c_raw.get("sku") or sku), str(c_raw.get("phase") or "decode"), str(c_raw.get("mode") or "eager"), str(c_raw.get("shape_regime") or "cliff"), str(c_raw.get("parallel_topology") or "tp1"))  # type: ignore[arg-type]
        ape = float(row.get("absolute_percentage_error", 0.0))
        if ape * PERCENT_SCALE >= cap_threshold:
            pw_fail.append(str(row.get("case_id") or ""))
        res_objs.append(Residual(str(row.get("case_id") or ""), lvl, cohort, str(row.get("acquisition_root") or ""), int(row.get("measured_ns") or 1), int(row.get("predicted_ns") or 1), str(row.get("engine_receipt") or ""), float(row.get("signed_error", 0.0)), ape, "ok", boundary_kind=str(row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS.get(lvl, "device_kernel"))))
    summaries = summarize_cohorts(res_objs, threshold_pct=cap_threshold)
    verdict = gate_decision(summaries, required_levels=REQUIRED_MULTILEVEL_LEVELS, require_disjoint_level_roots=True)
    missing_any = bool(missing_plan or missing_anchors or missing_cliffs)
    passed = bool(validation_doc.get("ok")) and abi_ok and verdict["status"] == "pass" and not pw_fail and not missing_any
    eff_status = "pass" if passed else ("fail" if (pw_fail or not abi_ok or not validation_doc.get("ok")) else ("insufficient_evidence" if missing_any else verdict["status"]))
    return (
        GateCheckOutcome(
            name="multilevel_gate", passed=passed, status=eff_status,
            detail=f"status={eff_status};levels={verdict['covered_levels']};missing_levels={verdict['missing_levels']};missing_plan={len(missing_plan)};pointwise_failures={len(pw_fail)};abi_ok={abi_ok}",
            metrics=(("covered_levels", verdict["covered_levels"]), ("missing_levels", verdict["missing_levels"]), ("missing_plan_points", len(missing_plan)), ("missing_anchors", len(missing_anchors)), ("missing_cliffs", len(missing_cliffs)), ("overlapping_level_roots", verdict["overlapping_level_roots"]), ("failed_cohorts", verdict["failed_cohorts"]), ("pointwise_failures", sorted(pw_fail)), ("abi_ok", abi_ok), ("threshold_pct", cap_threshold)),
        ),
        heldout_rows,
    )


__all__ = [
    "REQUIRED_MULTILEVEL_LEVELS",
    "STANDARD_LEVEL_BOUNDARY_KINDS",
    "WhiteboxSpineRunner",
    "run_multilevel_verdict",
    "score_sealed_gate_cases",
    "validate_disjoint_lineage",
]
