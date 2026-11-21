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

from ._capture_truth import verify_capture_and_override_case
from ._lean_and_receipts import validate_strict_atol_ns, validate_strict_threshold_pct
from ._lineage_and_clock import (
    REQUIRED_MULTILEVEL_LEVELS,
    STANDARD_LEVEL_BOUNDARY_KINDS,
    _ALLOWED_OK_STATUSES,
    _check_no_caller_predictions,
    _norm_digest,
    exceeds_physical_tolerance,
    extract_measured_p99_ns,
    measured_ns_from_row,
    resolve_authoritative_clock_operator,
    validate_disjoint_lineage,
    validate_event_ids_and_ranks,
    validate_row_matches_plan_point,
)
from ._whitebox_runner import WhiteboxSpineRunner, check_no_service_regimes, eligible_replay_cases
from .contracts import (
    DEFAULT_CLOCK_OPERATOR, DEFAULT_HELDOUT_APE_THRESHOLD_PCT, DEFAULT_PHYSICAL_ATOL_NS,
    DISPLAY_ROUND_DIGITS, PERCENT_SCALE, SHORT_DIGEST_LEN, assert_sanitized_payload,
    sanitize_public_alias, sanitize_public_capture_uri, sanitize_public_run_id,
)

_P90_QUANTILE = 0.90


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
    if bool(row.get("debug_cross_level")):
        raise ModelingError(f"observation_debug_cross_level_forbidden:{cid}")
    if bool(row.get("is_counterfactual")):
        raise ModelingError(f"observation_counterfactual_forbidden:{cid}")
    clock_op = resolve_authoritative_clock_operator(row, cid, plan_pt=plan_pt)
    coord_keys = (
        "sim_case", "target_op_id", "target_family", "target_layer", "target_op_ids",
        "measured_binding", "actual_shape", "plan_key", "timed_nodes",
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

        if busy_int is not None and (busy_int <= 0 or elapsed_int <= 0 or busy_int > elapsed_int):
            raise ModelingError(
                f"engine_busy_union_exceeds_elapsed_span:{case.case_id}"
                f":busy={busy_int}:elapsed={elapsed_int}"
            )
        if eng_op == "busy_union":
            if busy_int is None or busy_int <= 0:
                raise ModelingError(f"engine_busy_union_missing:{case.case_id}")
            if hasattr(interval, "clock_operator"):
                if int(interval.predicted_ns) != busy_int:
                    raise ModelingError(
                        f"engine_clock_operator_value_mismatch:{case.case_id}:op=busy_union"
                        f":predicted_ns={interval.predicted_ns}:diag={busy_int}"
                    )
            elif elapsed_int != int(interval.predicted_ns):
                raise ModelingError(
                    f"engine_clock_operator_value_mismatch:{case.case_id}:op=busy_union"
                    f":predicted_ns={interval.predicted_ns}:elapsed={elapsed_int}"
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
                exec_ops = diag.get("executed_op_ids")
                if (
                    not self._expected_family
                    and case.level in {"L1", "L2"}
                    and isinstance(exec_ops, Sequence)
                    and not isinstance(exec_ops, (str, bytes))
                    and exec_ops
                ):
                    from ._profiler_plan import resolve_point_measured_binding

                    exp_info = resolve_point_measured_binding({"family": "sku_suite"}, plan_pt, sku=case.cohort.sku)
                    if case.level == "L1":
                        exp_tid = str(exp_info.get("target_op_id") or "").strip()
                        if exp_tid and list(str(x) for x in exec_ops) != [exp_tid]:
                            raise ModelingError(
                                f"whitebox_measured_binding_mismatch:{case.case_id}:l1_plan_target_mismatch"
                                f":executed={list(str(x) for x in exec_ops)}:expected={[exp_tid]}"
                            )
                    else:
                        exp_ops = set(exp_info["measured_binding"].get("timed_op_ids") or ())
                        if exp_ops and set(str(x) for x in exec_ops) != exp_ops:
                            raise ModelingError(
                                f"whitebox_measured_binding_mismatch:{case.case_id}:l2_plan_target_mismatch"
                                f":executed={sorted(str(x) for x in exec_ops)}:expected={sorted(exp_ops)}"
                            )

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
    if isinstance(runner, WhiteboxSpineRunner):
        if (
            primitive_table is not None
            and runner.primitive_table is not None
            and runner.primitive_table.envelope_sha256 != primitive_table.envelope_sha256
        ):
            raise ModelingError(
                f"runner_primitive_table_mismatch:runner={runner.primitive_table.envelope_sha256[:SHORT_DIGEST_LEN]}"
                f":requested={primitive_table.envelope_sha256[:SHORT_DIGEST_LEN]}"
            )
        if (
            profile_override is not None
            and runner.profile_override is not None
            and dict(runner.profile_override) != dict(profile_override)
        ):
            raise ModelingError("runner_profile_override_mismatch")
        if (
            (primitive_table is not None and runner.primitive_table is None)
            or (profile_override is not None and runner.profile_override is None)
        ):
            return WhiteboxSpineRunner(
                clock_operator=runner.clock_operator,
                primitive_table=runner.primitive_table if runner.primitive_table is not None else primitive_table,
                profile_override=runner.profile_override if runner.profile_override is not None else profile_override,
                max_workers=runner.max_workers,
            )
    return runner


def _build_and_verify_gate_cases(
    scorable_rows: Sequence[Mapping[str, Any]],
    *,
    sku: str,
    plan_by_id: Mapping[str, Mapping[str, Any]],
    train_plan_by_id: Mapping[str, Mapping[str, Any]],
    expected_levels: Mapping[str, str] = (),
    exp_fam: str,
    allowed_impls: set[str],
    capture_root: str | Path | None,
    capture_files: Mapping[str, str | Path] | None,
    require_capture_files: bool,
) -> tuple[
    list[GateCase],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, str | Path],
    list[bool],
]:
    cases: list[GateCase] = []
    expected_clock: dict[str, str] = {}
    case_meta: dict[str, dict[str, Any]] = {}
    plan_by_case: dict[str, Mapping[str, Any]] = {}
    seen_heldout_ids: set[str] = set()
    seen_train_ids: set[str] = set()
    seen_capture_events: dict[tuple[str, str, tuple[int, ...]], list[tuple[frozenset[str], str]]] = {}
    captures_verified_flags: list[bool] = []
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
            digest_index=None,
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
                validate_row_matches_plan_point(
                    row, plan_by_id[pid], sku=sku, pid=pid,
                    device_ranks=gc.boundary.device_ranks,
                    expected_level=str(expected_levels.get(pid) or ""),
                )
                plan_by_case[gc.case_id] = plan_by_id[pid]
            cap_grp = ("heldout", _norm_digest(gc.boundary.raw_sha256), tuple(sorted(gc.boundary.device_ranks)))
            ev_set = frozenset(gc.boundary.event_ids)
            for prev_ev, prev_pid in seen_capture_events.get(cap_grp, ()):
                if ev_set & prev_ev:
                    raise ModelingError(f"duplicate_heldout_physical_capture:{pid}:collides_with={prev_pid}")
            seen_capture_events.setdefault(cap_grp, []).append((ev_set, pid))
        elif gc.split == "train":
            if pid in seen_train_ids:
                raise ModelingError(f"duplicate_train_point_id:{pid}")
            seen_train_ids.add(pid)
            if pid in train_plan_by_id:
                validate_row_matches_plan_point(
                    row, train_plan_by_id[pid], sku=sku, pid=pid,
                    device_ranks=gc.boundary.device_ranks,
                    expected_level=str(expected_levels.get(pid) or ""),
                )
                plan_by_case[gc.case_id] = train_plan_by_id[pid]
            cap_grp = ("train", _norm_digest(gc.boundary.raw_sha256), tuple(sorted(gc.boundary.device_ranks)))
            ev_set = frozenset(gc.boundary.event_ids)
            for prev_ev, prev_pid in seen_capture_events.get(cap_grp, ()):
                if ev_set & prev_ev:
                    raise ModelingError(f"duplicate_train_physical_capture:{pid}:collides_with={prev_pid}")
            seen_capture_events.setdefault(cap_grp, []).append((ev_set, pid))
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
    return cases, expected_clock, case_meta, plan_by_case, resolved_files, captures_verified_flags


def _aggregate_scored_residuals(
    raw_residuals: Sequence[Any],
    *,
    case_meta: Mapping[str, Mapping[str, Any]],
    expected_clock: Mapping[str, str],
    engine_clocks: Mapping[str, Mapping[str, Any]],
    cap_threshold: float,
    atol_ns: int,
) -> tuple[list[dict[str, Any]], list[float], list[float], list[float], list[float], list[str], set[str], set[str]]:
    residual_dicts: list[dict[str, Any]] = []
    heldout_apes, heldout_signed, train_apes, train_signed = [], [], [], []
    failing_cases: list[str] = []
    scored_heldout_ids: set[str] = set()
    scored_train_ids: set[str] = set()
    for res in raw_residuals:
        meta = case_meta[res.case_id]
        clk = engine_clocks.get(res.case_id, {})
        scored_pred_ns = int(clk.get("scored_predicted_ns", res.predicted_ns))
        signed_err = (scored_pred_ns - res.measured_ns) / float(res.measured_ns) if res.measured_ns > 0 else res.signed_error
        ape = abs(signed_err)
        ape_pct = ape * PERCENT_SCALE
        signed_pct = signed_err * PERCENT_SCALE
        abs_err_ns = abs(scored_pred_ns - int(res.measured_ns))
        exceeds = exceeds_physical_tolerance(
            scored_pred_ns, int(res.measured_ns), threshold_pct=cap_threshold, atol_ns=atol_ns
        )
        row_dict = {
            **res.to_dict(),
            "predicted_ns": scored_pred_ns, "signed_error": signed_err, "absolute_percentage_error": ape,
            "abs_error_ns": abs_err_ns, "atol_ns": atol_ns, "within_physical_tolerance": not exceeds,
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
            if exceeds:
                failing_cases.append(res.case_id)
        elif meta["split"] == "train" and res.status == "ok":
            scored_train_ids.add(meta["point_id"])
            train_apes.append(ape_pct)
            train_signed.append(signed_pct)
    return residual_dicts, heldout_apes, heldout_signed, train_apes, train_signed, failing_cases, scored_heldout_ids, scored_train_ids


def score_sealed_gate_cases(
    observations: Sequence[Mapping[str, Any]],
    *,
    sku: str,
    runner: EvaluationRunner | None = None,
    primitive_table: PrimitiveTable | None = None,
    profile_override: Mapping[str, Any] | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS,
    external_train_roots: Sequence[str] = (),
    external_train_digests: Sequence[str] = (),
    external_train_uris: Sequence[str] = (),
    expected_points: Sequence[Mapping[str, Any]] = (),
    expected_train_points: Sequence[Mapping[str, Any]] = (),
    expected_point_ids: Sequence[str] = (),
    expected_levels: Mapping[str, str] | None = None,
    expected_family: str = "",
    expected_implementations: Sequence[str] = (),
    require_disjoint_level_roots: bool = True,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = False,
) -> dict[str, Any]:
    """Score observations strictly via ``GateCase -> EvaluationRunner -> EngineRunReceipt``."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    eff_atol_ns = validate_strict_atol_ns(atol_ns)
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
            "threshold_pct": cap_threshold, "atol_ns": eff_atol_ns,
            "lineage": {"train_roots": sorted(set(external_train_roots)), "heldout_roots": [], "level_roots": {}, "disjoint_ok": False},
            "heldout_summary": _summarize_apes([], []), "train_summary": _summarize_apes([], []),
            "missing_plan_points": all_expected_ids, "missing_train_points": all_expected_train_ids,
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
    exp_fam = expected_family.strip()
    allowed_impls = {str(x).strip() for x in expected_implementations if str(x).strip()}
    cases, expected_clock, case_meta, plan_by_case, resolved_files, captures_verified_flags = (
        _build_and_verify_gate_cases(
            scorable_rows, sku=sku, plan_by_id=plan_by_id, train_plan_by_id=train_plan_by_id,
            expected_levels=dict(expected_levels or {}),
            exp_fam=exp_fam, allowed_impls=allowed_impls,
            capture_root=capture_root, capture_files=capture_files,
            require_capture_files=require_capture_files,
        )
    )
    manifest = Manifest(
        manifest_id=f"onboarding-{sanitize_public_alias(sku, 'sku')}",
        schema_version="mindie-multilevel-manifest/1",
        cases=tuple(cases),
        strict_three_level_blind=require_disjoint_level_roots,
    )
    all_fit_tokens = sorted(
        {str(t).strip() for t in (*external_train_roots, *external_train_digests) if str(t).strip()}
    )
    fits = FrozenFitManifest(
        "onboarding-frozen-fits",
        "mindie-frozen-fits/1",
        tuple(
            FrozenFitParameter(parameter_id=f"train-tok-{i}", training_digests=(tok,))
            for i, tok in enumerate(all_fit_tokens)
        ),
    )
    eff_cap_files = resolved_files or capture_files
    validate_lineage(
        manifest, fits, capture_root=capture_root, capture_files=eff_cap_files,
        require_capture_files=require_capture_files,
    )
    from ._sku_calibration import temporary_scaffold_profile

    active_runner = _bind_active_runner(
        runner, primitive_table=primitive_table, profile_override=profile_override
    )
    checked_runner = ClockParityRunner(
        active_runner, expected_clock, expected_family=exp_fam, plan_by_case=plan_by_case
    )
    with temporary_scaffold_profile(sku, profile_override):
        if isinstance(active_runner, WhiteboxSpineRunner):
            replayable = eligible_replay_cases(manifest.cases, target_split="all")
            active_runner.prefetch_cases(replayable)
        raw_residuals = list(
            evaluate_multilevel(
                manifest, fits, checked_runner, target_split="all",
                capture_root=capture_root, capture_files=eff_cap_files,
                require_capture_files=require_capture_files,
            )
        )
    (
        residual_dicts, heldout_apes, heldout_signed, train_apes, train_signed,
        failing_cases, scored_heldout_ids, scored_train_ids,
    ) = _aggregate_scored_residuals(
        raw_residuals, case_meta=case_meta, expected_clock=expected_clock,
        engine_clocks=checked_runner.engine_clocks, cap_threshold=cap_threshold, atol_ns=eff_atol_ns,
    )
    missing_plan_points = sorted(set(all_expected_ids) - scored_heldout_ids)
    missing_train_points = sorted(set(all_expected_train_ids) - scored_train_ids)
    if not heldout_apes or missing_plan_points or missing_train_points:
        ok, status = False, "insufficient_evidence"
    elif failing_cases:
        ok, status = False, "fail"
    else:
        ok, status = True, "pass"

    result = {
        "ok": ok, "status": status, "sku": sku,
        "threshold_pct": cap_threshold, "atol_ns": eff_atol_ns,
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
    "exceeds_physical_tolerance",
    "extract_measured_p99_ns",
    "measured_ns_from_row",
    "resolve_authoritative_clock_operator",
    "run_multilevel_verdict",
    "score_sealed_gate_cases",
    "validate_disjoint_lineage",
    "validate_event_ids_and_ranks",
    "validate_row_matches_plan_point",
]
