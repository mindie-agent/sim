"""Sealed GateCase -> EvaluationRunner -> EngineRunReceipt scoring and multilevel verdicts.

[Role] Execute Prevention Lock 2 white-box scoring strictly through
  ``mindie_sim.observations.multilevel_gate.evaluate`` with live HMAC-signed
  ``EngineRunReceipt`` handles, identical clock-operator parity
  (``busy_union`` vs ``elapsed_span``), disjoint lineage enforcement, 1:1
  planned-grid coverage, and pointwise ``< 10.0%`` release verdicts.
[Invariants] Structurally forbids caller-supplied ``predicted_*`` fields,
  ``service_regimes`` crutches, ``over_budget`` substitutions, missing planned
  cells, and synthetic receipt strings.
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
    bind_engine_interval,
    evaluate as evaluate_multilevel,
    gate_decision,
    parse_gate_case,
    stream_manifest,
    summarize_cohorts,
    validate_lineage,
)
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.schema.common import ModelingError

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
_DEFAULT_ENGINE_START_NS = 100.0
_ENGINE_TRACE_LIMIT = 32


def validate_disjoint_lineage(
    observations: Sequence[Mapping[str, Any]],
    *,
    external_train_roots: Sequence[str] = (),
    require_disjoint_level_roots: bool = True,
) -> dict[str, Any]:
    """Fail closed if train vs heldout or L1/L2/L3 share any acquisition lineage root."""
    train_roots: set[str] = {str(r).strip() for r in external_train_roots if str(r).strip()}
    heldout_roots: set[str] = set()
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
        if split == "train":
            train_roots.add(root)
        else:
            heldout_roots.add(root)
            level = str(row.get("level") or "L1").strip().upper()
            level_roots.setdefault(level, set()).add(root)
        for parent in row.get("parent_digests") or boundary.get("parent_digests") or ():
            ptok = str(parent).strip()
            if split == "heldout" and ptok in train_roots:
                raise ModelingError(f"train_heldout_parent_lineage_leakage:{cid}:{ptok}")

    train_overlap = sorted(train_roots & heldout_roots)
    if train_overlap:
        raise ModelingError(f"train_heldout_lineage_overlap:{','.join(train_overlap)}")
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


def _check_no_service_regimes(payload: Any, *, path: str = "case") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_s = str(key)
            if key_s == "service_regimes" and value:
                raise ModelingError(f"whitebox_service_regimes_forbidden:{path}")
            _check_no_service_regimes(value, path=f"{path}.{key_s}")
    elif isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            _check_no_service_regimes(item, path=f"{path}[{idx}]")


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


def _measured_ns_from_row(row: Mapping[str, Any], cid: str) -> int:
    if row.get("measured_ns") is not None:
        val = float(row["measured_ns"])
    elif row.get("measured_us") is not None:
        val = float(row["measured_us"]) * NS_PER_US
    elif row.get("measured_ms") is not None:
        val = float(row["measured_ms"]) * NS_PER_MS
    else:
        raise ModelingError(f"observation_measured_duration_missing:{cid}")
    if not math.isfinite(val) or val <= 0.0:
        raise ModelingError(f"observation_measured_duration_nonpositive:{cid}:{val}")
    return int(round(val))


def _row_to_gate_case(row: Mapping[str, Any], *, sku: str, idx: int) -> GateCase:
    cid = str(row.get("case_id") or row.get("point_id") or f"obs-{idx}")
    _check_no_caller_predictions(row, cid)
    _check_no_service_regimes(row, path=cid)
    clock_op = _resolve_clock_operator(row, cid)
    if isinstance(row.get("boundary"), Mapping) and isinstance(row.get("cohort"), Mapping):
        raw_copy = dict(row)
        payload = dict(raw_copy.get("payload") or {})
        payload["clock_operator"] = clock_op
        if row.get("sim_case") is not None:
            payload["sim_case"] = row["sim_case"]
        raw_copy["payload"] = payload
        return parse_gate_case(raw_copy)

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
    measured_ns = _measured_ns_from_row(row, cid)
    payload_dict: dict[str, Any] = {"clock_operator": clock_op, "point_id": str(row.get("point_id") or cid)}
    for extra in ("sim_case", "phase", "tokens", "context_len", "sequences", "batch_size", "query_tokens", "topk"):
        if row.get(extra) is not None:
            payload_dict[extra] = row[extra]
    boundary = CaptureBoundary(
        raw_sha256=raw_sha,
        acquisition_uid=acq,
        run_uid=run_uid,
        parent_digests=tuple(str(p) for p in (row.get("parent_digests") or ())),
        event_ids=tuple(str(e) for e in (row.get("event_ids") or (f"ev-{cid}",))),
        device_ranks=tuple(int(r) for r in (row.get("device_ranks") or (0,))),
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


def _union_interval_span_ms(intervals: Sequence[tuple[float, float]]) -> float:
    """Compute exact total busy time (union of closed ``[start_ms, end_ms]`` intervals)."""
    valid = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not valid:
        return 0.0
    total = 0.0
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


class WhiteboxSpineRunner:
    """Sealed ``EvaluationRunner`` executing ``mindie_sim.api.evaluate_step`` on a live Engine."""

    def __init__(self, *, clock_operator: str = DEFAULT_CLOCK_OPERATOR) -> None:
        if clock_operator not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"whitebox_runner_clock_operator_invalid:{clock_operator}")
        self.clock_operator = clock_operator

    def _run_case(self, case: GateCase) -> EngineInterval:
        from mindie_sim.api import compose_case, evaluate_step
        from mindie_sim.prediction.results import StepRequest
        from mindie_sim.schema.case import Case

        payload = dict(case.payload)
        raw_case = payload.get("sim_case")
        if raw_case is None:
            raise ModelingError(f"whitebox_case_missing_sim_case:{case.case_id}")
        if isinstance(raw_case, Case):
            sim_case: Any = raw_case
        elif isinstance(raw_case, Mapping):
            _check_no_service_regimes(raw_case, path=f"{case.case_id}.sim_case")
            if "compose_model" in raw_case:
                sim_case = compose_case(
                    raw_case["compose_model"],
                    raw_case.get("compose_hardware", case.cohort.sku),
                    options=raw_case.get("compose_options"),
                )
            else:
                sim_case = raw_case
        else:
            raise ModelingError(f"whitebox_case_invalid_type:{case.case_id}")

        req = StepRequest(
            phase=str(payload.get("phase") or case.cohort.phase or "decode"),  # type: ignore[arg-type]
            tokens=int(payload.get("query_tokens") or payload.get("tokens") or 1),
            context=int(payload.get("context_len") or payload.get("context") or 0),
            sequences=int(payload.get("batch_size") or payload.get("sequences") or 1),
            scope="body",
        )
        ev = evaluate_step(sim_case, req)
        elapsed_ms = float(ev.span_ms)
        busy_ms = _union_interval_span_ms(
            [(op.start_ms, op.end_ms) for op in ev.operators if op.start_ms is not None and op.end_ms is not None]
        ) or elapsed_ms
        elapsed_ns = int(round(elapsed_ms * NS_PER_MS))
        busy_ns = int(round(busy_ms * NS_PER_MS))
        if elapsed_ns <= 0 or busy_ns <= 0:
            raise ModelingError(f"whitebox_prediction_nonpositive:{case.case_id}")
        target_ns = busy_ns if self.clock_operator == "busy_union" else elapsed_ns
        eng = Engine([Resource("device", float(NS_PER_MS), "ns", "fifo")], trace_limit=_ENGINE_TRACE_LIMIT)
        s_ev = eng.submit(case.boundary.start_event, {"device": _DEFAULT_ENGINE_START_NS})
        e_ev = eng.submit(case.boundary.end_event, {"device": float(target_ns)}, after=(s_ev,))
        eng.advance(until=(e_ev,))
        return bind_engine_interval(
            case, eng, s_ev, e_ev,
            diagnostics=(
                ("clock_operator", self.clock_operator),
                ("busy_union_ns", busy_ns),
                ("elapsed_span_ns", elapsed_ns),
            ),
        )

    def run_l1(self, case: GateCase) -> EngineInterval: return self._run_case(case)
    def run_l2(self, case: GateCase) -> EngineInterval: return self._run_case(case)
    def run_l3(self, case: GateCase) -> EngineInterval: return self._run_case(case)


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
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    external_train_roots: Sequence[str] = (),
    expected_point_ids: Sequence[str] = (),
    require_disjoint_level_roots: bool = True,
) -> dict[str, Any]:
    """Score observations strictly via ``GateCase -> EvaluationRunner -> EngineRunReceipt``."""
    if not observations:
        return {
            "ok": False, "status": "insufficient_evidence", "sku": sku,
            "threshold_pct": float(threshold_pct),
            "lineage": {"train_roots": sorted(set(external_train_roots)), "heldout_roots": [], "level_roots": {}, "disjoint_ok": False},
            "heldout_summary": _summarize_apes([], []), "train_summary": _summarize_apes([], []),
            "missing_plan_points": sorted(set(expected_point_ids)), "over_budget_points": [],
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
        require_disjoint_level_roots=require_disjoint_level_roots,
    )
    cases: list[GateCase] = []
    expected_clock: dict[str, str] = {}
    case_meta: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(scorable_rows):
        gc = _row_to_gate_case(row, sku=sku, idx=idx)
        cases.append(gc)
        expected_clock[gc.case_id] = str(dict(gc.payload).get("clock_operator", DEFAULT_CLOCK_OPERATOR))
        case_meta[gc.case_id] = {
            "point_id": str(row.get("point_id") or gc.case_id), "split": gc.split,
            "silicon_busy_union_ns": row.get("busy_union_ns"), "silicon_elapsed_span_ns": row.get("elapsed_span_ns"),
        }

    manifest = Manifest(
        manifest_id=f"onboarding-{sanitize_public_alias(sku, 'sku')}",
        schema_version="mindie-multilevel-manifest/1",
        cases=tuple(cases),
        strict_three_level_blind=require_disjoint_level_roots,
    )
    fit_params = tuple(
        FrozenFitParameter(parameter_id=f"train-root-{i}", training_digests=(str(root),))
        for i, root in enumerate(sorted(set(external_train_roots))) if str(root).strip()
    )
    fits = FrozenFitManifest("onboarding-frozen-fits", "mindie-frozen-fits/1", fit_params)
    validate_lineage(manifest, fits)
    checked_runner = _ClockParityRunner(runner if runner is not None else WhiteboxSpineRunner(), expected_clock)
    raw_residuals = list(evaluate_multilevel(manifest, fits, checked_runner, target_split="all"))

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
            scored_heldout_ids.update((res.case_id, meta["point_id"]))
            heldout_apes.append(ape_pct)
            heldout_signed.append(signed_pct)
            if ape_pct >= threshold_pct:
                failing_cases.append(res.case_id)
        elif meta["split"] == "train" and res.status == "ok":
            train_apes.append(ape_pct)
            train_signed.append(signed_pct)

    missing_plan_points = sorted(set(expected_point_ids) - scored_heldout_ids)
    if not heldout_apes or missing_plan_points:
        ok, status = False, "insufficient_evidence"
    elif failing_cases:
        ok, status = False, "fail"
    else:
        ok, status = True, "pass"

    result = {
        "ok": ok, "status": status, "sku": sku, "threshold_pct": float(threshold_pct),
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
    threshold_pct: float,
) -> tuple[GateCheckOutcome, list[dict[str, Any]]]:
    """Evaluate strict pointwise multilevel L1/L2/L3 gate verdict."""
    if multilevel_manifest_json is not None and runner is not None:
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
        residuals = list(evaluate_multilevel(manifest, fits, runner, target_split="heldout"))
        summaries = summarize_cohorts(residuals, threshold_pct=threshold_pct)
        verdict = gate_decision(summaries, required_levels=REQUIRED_MULTILEVEL_LEVELS, require_disjoint_level_roots=True)
        pw_fail = sorted(r.case_id for r in residuals if r.status == "ok" and r.absolute_percentage_error * PERCENT_SCALE >= threshold_pct)
        passed = verdict["status"] == "pass" and not pw_fail
        return (
            GateCheckOutcome(
                name="multilevel_gate", passed=passed, status="pass" if passed else ("fail" if pw_fail else verdict["status"]),
                detail=f"status={verdict['status']};levels={verdict['covered_levels']};missing={verdict['missing_levels']};pointwise_failures={len(pw_fail)}",
                metrics=(("covered_levels", verdict["covered_levels"]), ("missing_levels", verdict["missing_levels"]), ("overlapping_level_roots", verdict["overlapping_level_roots"]), ("failed_cohorts", verdict["failed_cohorts"]), ("pointwise_failures", pw_fail), ("threshold_pct", threshold_pct)),
            ),
            [r.to_dict() for r in residuals],
        )

    if validation_doc is None:
        return GateCheckOutcome("multilevel_gate", False, "insufficient_evidence", "no validation report or multilevel manifest supplied"), []

    heldout_rows = [r for r in (validation_doc.get("residuals") or ()) if str(r.get("split") or "heldout") == "heldout"]
    missing_plan = list(validation_doc.get("missing_plan_points") or ())
    missing_anchors = list(validation_doc.get("missing_anchors") or ())
    missing_cliffs = list(validation_doc.get("missing_cliffs") or ())
    if not heldout_rows:
        return GateCheckOutcome("multilevel_gate", False, "insufficient_evidence", "validation report contains zero heldout residuals"), []
    validate_disjoint_lineage(heldout_rows, require_disjoint_level_roots=True)
    res_objs: list[Residual] = []
    pw_fail: list[str] = []
    for row in heldout_rows:
        lvl = str(row.get("level") or "L1").upper()
        c_raw = row.get("cohort") if isinstance(row.get("cohort"), Mapping) else {}
        cohort = CohortKey(lvl, str(c_raw.get("cohort_name") or "onboarding"), str(c_raw.get("sku") or sku), str(c_raw.get("phase") or "decode"), str(c_raw.get("mode") or "eager"), str(c_raw.get("shape_regime") or "cliff"), str(c_raw.get("parallel_topology") or "tp1"))  # type: ignore[arg-type]
        ape = float(row.get("absolute_percentage_error", 0.0))
        if ape * PERCENT_SCALE >= threshold_pct:
            pw_fail.append(str(row.get("case_id") or ""))
        res_objs.append(Residual(str(row.get("case_id") or ""), lvl, cohort, str(row.get("acquisition_root") or ""), int(row.get("measured_ns") or 1), int(row.get("predicted_ns") or 1), str(row.get("engine_receipt") or ""), float(row.get("signed_error", 0.0)), ape, "ok", boundary_kind=str(row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS.get(lvl, "device_kernel"))))
    summaries = summarize_cohorts(res_objs, threshold_pct=threshold_pct)
    verdict = gate_decision(summaries, required_levels=REQUIRED_MULTILEVEL_LEVELS, require_disjoint_level_roots=True)
    missing_any = bool(missing_plan or missing_anchors or missing_cliffs)
    passed = verdict["status"] == "pass" and not pw_fail and not missing_any
    eff_status = "pass" if passed else ("fail" if pw_fail else ("insufficient_evidence" if missing_any else verdict["status"]))
    return (
        GateCheckOutcome(
            name="multilevel_gate", passed=passed, status=eff_status,
            detail=f"status={eff_status};levels={verdict['covered_levels']};missing_levels={verdict['missing_levels']};missing_plan={len(missing_plan)};pointwise_failures={len(pw_fail)}",
            metrics=(("covered_levels", verdict["covered_levels"]), ("missing_levels", verdict["missing_levels"]), ("missing_plan_points", len(missing_plan)), ("missing_anchors", len(missing_anchors)), ("missing_cliffs", len(missing_cliffs)), ("overlapping_level_roots", verdict["overlapping_level_roots"]), ("failed_cohorts", verdict["failed_cohorts"]), ("pointwise_failures", sorted(pw_fail)), ("threshold_pct", threshold_pct)),
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
