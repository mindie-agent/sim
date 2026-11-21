"""Strict multilevel manifest and Stage-3 validation report verdict engine.

[Role] Evaluate ``run_multilevel_verdict`` for ``verify_sku_gates`` across both
  the Stage-3 HMAC-verified validation report path AND the direct
  ``multilevel_manifest_json`` replay path, enforcing the FULL authoritative
  SKU plan (all 7 train anchors, all 8 off-grid cliffs, 2D batch/MTP/chunked
  and multi-rank points, ABI rejections, verified raw capture bytes, strict
  authoritative clock parity, capture-derived ``p99``, and canonical
  ``(program, event_dag)`` digests).
[Invariants] Never accepts an incomplete manifest (e.g. 8 heldout rows with
  missing train anchors, multi-rank cells, ABI rejections, or missing raw
  capture files); never falls back from ``busy_union`` to ``elapsed_span``;
  always validates ``validation_doc["sku"] == sku`` even when a manifest is
  also passed; never lowers the ``<= 10.0%`` release ceiling.
[Timeouts & Locks] Deterministic in-memory & bounded file verification; zero
  subprocesses or device contexts.
[Side-Effect Policy] Strictly read-only over manifest, fits, and capture files.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from mindie_sim.observations.multilevel_gate import (
    CohortKey,
    EvaluationRunner,
    FrozenFitManifest,
    FrozenFitParameter,
    GateCase,
    Manifest,
    Residual,
    evaluate as evaluate_multilevel,
    gate_decision,
    parse_gate_case,
    summarize_cohorts,
    validate_lineage,
)
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die

from ._capture_truth import (
    reverify_report_captures,
    verify_capture_and_override_case,
)
from ._lean_and_receipts import validate_strict_threshold_pct, verify_stage_report
from ._sku_calibration import materialize_sku_hardware
from ._whitebox_runner import check_no_service_regimes
from .contracts import (
    DEFAULT_PHYSICAL_ATOL_NS,
    GateCheckOutcome,
    read_bounded_json,
    sanitize_public_capture_uri,
    sanitize_public_run_id,
)

_ENGINE_RECEIPT_PREFIX = "capture://engine-receipt/"
_ALLOWED_OK_STATUSES = frozenset({"", "ok", "pass", "passed", "success"})


def _parse_fits(fits_json: str | Path | None) -> FrozenFitManifest:
    if fits_json is None:
        return FrozenFitManifest("onboard-empty", "mindie-frozen-fits/1", ())
    raw_fits = read_bounded_json(fits_json)
    params = tuple(
        FrozenFitParameter(
            str(p["parameter_id"]),
            tuple(str(d) for d in p.get("training_digests", ())),
            str(p.get("source_uri") or ""),
        )
        for p in raw_fits.get("parameters", ())
    )
    edges = tuple((str(e[0]), str(e[1])) for e in raw_fits.get("lineage_edges", ()))
    return FrozenFitManifest(
        str(raw_fits.get("fit_id") or "onboard-fit"),
        str(raw_fits.get("schema_version") or "mindie-frozen-fits/1"),
        params,
        edges,
    )


def _parse_manifest_cases(
    raw_cases: Sequence[Mapping[str, Any]],
    *,
    sku: str,
    held_plan_by_id: Mapping[str, Mapping[str, Any]],
    train_plan_by_id: Mapping[str, Mapping[str, Any]],
    capture_root: str | Path | None,
    capture_files: Mapping[str, str | Path] | None,
    require_capture_files: bool,
) -> tuple[
    GateCheckOutcome | None,
    list[GateCase],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, str | Path],
]:
    from ._sealed_scorer import (
        _check_no_caller_predictions,
        _norm_digest,
        extract_measured_p99_ns,
        measured_ns_from_row,
        resolve_authoritative_clock_operator,
        validate_event_ids_and_ranks,
        validate_row_matches_plan_point,
    )

    parsed_cases: list[GateCase] = []
    expected_clock: dict[str, str] = {}
    case_meta: dict[str, dict[str, Any]] = {}
    plan_by_case: dict[str, Mapping[str, Any]] = {}
    seen_heldout_ids: set[str] = set()
    seen_train_ids: set[str] = set()
    seen_captures: dict[tuple[str, frozenset[str], tuple[int, ...], str], str] = {}
    resolved_files: dict[str, str | Path] = dict(capture_files or {})

    for idx, row in enumerate(raw_cases):
        cid = str(row.get("case_id") or f"manifest-{idx}")
        _check_no_caller_predictions(row, cid)
        check_no_service_regimes(row, path=cid)
        row_status = str(row.get("status") or "").strip().lower()
        if row_status not in _ALLOWED_OK_STATUSES:
            raise ModelingError(f"observation_status_not_ok:{cid}:status={row_status}")
        if row.get("validated") is False or row.get("qualified") is False:
            raise ModelingError(f"observation_unvalidated_or_unqualified:{cid}")
        raw_copy = dict(row)
        payload = dict(raw_copy.get("payload") or {})
        pid = str(row.get("point_id") or payload.get("point_id") or cid).strip()
        plan_pt = held_plan_by_id.get(pid) or train_plan_by_id.get(pid)
        clock_op = resolve_authoritative_clock_operator(row, cid, plan_pt=plan_pt)
        payload["clock_operator"] = clock_op
        if plan_pt is not None:
            payload.setdefault("axis", str(plan_pt.get("axis") or ""))
            payload.setdefault("value", int(plan_pt["value"]))
        raw_copy["payload"] = payload
        gc = parse_gate_case(raw_copy)
        if gc.cohort.sku != sku:
            err_outcome = GateCheckOutcome(
                "multilevel_gate", False, "fail",
                f"multilevel_manifest_sku_mismatch:expected={sku}:got={gc.cohort.sku}",
            )
            return err_outcome, [], {}, {}, {}, {}
        validate_event_ids_and_ranks(gc.boundary.event_ids, gc.boundary.device_ranks, cid)
        measured_ns_from_row(row, cid, clock_op=clock_op, parsed_measured_ns=gc.measured_ns)
        gc, _cap_ok, auth_p99 = verify_capture_and_override_case(
            gc, row, clock_op=clock_op,
            capture_root=capture_root, capture_files=capture_files,
            require_capture_files=require_capture_files,
            row_p99_ns=extract_measured_p99_ns(row),
            digest_index=None,
            resolved_files=resolved_files,
        )
        if gc.split == "heldout":
            if pid in seen_heldout_ids:
                raise ModelingError(f"duplicate_heldout_point_id:{pid}")
            seen_heldout_ids.add(pid)
            if pid in held_plan_by_id:
                validate_row_matches_plan_point(row, held_plan_by_id[pid], sku=sku, pid=pid, device_ranks=gc.boundary.device_ranks)
                plan_by_case[gc.case_id] = held_plan_by_id[pid]
            cap_key = (
                _norm_digest(gc.boundary.raw_sha256),
                frozenset(gc.boundary.event_ids),
                gc.boundary.device_ranks,
                gc.boundary.run_uid,
            )
            if cap_key in seen_captures:
                raise ModelingError(f"duplicate_heldout_physical_capture:{pid}:collides_with={seen_captures[cap_key]}")
            seen_captures[cap_key] = pid
        elif gc.split == "train":
            if pid in seen_train_ids:
                raise ModelingError(f"duplicate_train_point_id:{pid}")
            seen_train_ids.add(pid)
            if pid in train_plan_by_id:
                validate_row_matches_plan_point(row, train_plan_by_id[pid], sku=sku, pid=pid, device_ranks=gc.boundary.device_ranks)
                plan_by_case[gc.case_id] = train_plan_by_id[pid]
        parsed_cases.append(gc)
        expected_clock[gc.case_id] = clock_op
        case_meta[gc.case_id] = {
            "point_id": pid, "split": gc.split, "measured_p99_ns": auth_p99,
            "raw_sha256": gc.boundary.raw_sha256,
            "capture_uri": sanitize_public_capture_uri(gc.boundary.capture_uri, raw_sha256=gc.boundary.raw_sha256),
            "run_uid": sanitize_public_run_id(gc.boundary.run_uid),
        }
    return None, parsed_cases, expected_clock, case_meta, plan_by_case, resolved_files


def _summarize_manifest_residuals(
    raw_residuals: Sequence[Residual],
    *,
    case_meta: Mapping[str, Mapping[str, Any]],
    expected_clock: Mapping[str, str],
    engine_clocks: Mapping[str, Mapping[str, Any]],
    held_plan_by_id: Mapping[str, Mapping[str, Any]],
    train_plan_by_id: Mapping[str, Mapping[str, Any]],
    abi_check: Mapping[str, Any],
    cap_threshold: float,
    atol_ns: int,
) -> tuple[GateCheckOutcome, list[dict[str, Any]]]:
    from ._sealed_scorer import REQUIRED_MULTILEVEL_LEVELS, exceeds_physical_tolerance

    heldout_res_objs: list[Residual] = []
    heldout_dicts: list[dict[str, Any]] = []
    scored_heldout_ids: set[str] = set()
    scored_train_ids: set[str] = set()
    pw_fail: list[str] = []
    for res in raw_residuals:
        meta = case_meta[res.case_id]
        clk = engine_clocks.get(res.case_id, {})
        scored_pred_ns = int(clk.get("scored_predicted_ns", res.predicted_ns))
        signed_err = (scored_pred_ns - res.measured_ns) / float(res.measured_ns) if res.measured_ns > 0 else res.signed_error
        ape = abs(signed_err)
        adj_res = replace(
            res, predicted_ns=scored_pred_ns, signed_error=signed_err, absolute_percentage_error=ape,
        )
        if meta["split"] == "train":
            if adj_res.status == "ok":
                scored_train_ids.add(meta["point_id"])
            continue
        heldout_res_objs.append(adj_res)
        exceeds = exceeds_physical_tolerance(
            scored_pred_ns, int(adj_res.measured_ns), threshold_pct=cap_threshold, atol_ns=atol_ns
        )
        if adj_res.status == "ok":
            scored_heldout_ids.add(meta["point_id"])
            if exceeds:
                pw_fail.append(adj_res.case_id)
        heldout_dicts.append({
            **adj_res.to_dict(),
            "abs_error_ns": abs(scored_pred_ns - int(adj_res.measured_ns)),
            "atol_ns": atol_ns, "within_physical_tolerance": not exceeds,
            "point_id": meta["point_id"], "split": "heldout",
            "clock_operator": clk.get("clock_operator", expected_clock[res.case_id]),
            "measured_p99_ns": meta["measured_p99_ns"],
            "engine_busy_union_ns": clk.get("busy_union_ns"), "engine_elapsed_span_ns": clk.get("elapsed_span_ns"),
            "program_dag_sha256": clk.get("program_dag_sha256", ""), "engine_uid": clk.get("engine_uid", ""),
            "raw_sha256": meta["raw_sha256"], "capture_uri": meta["capture_uri"], "run_uid": meta["run_uid"],
        })

    missing_heldout = sorted(set(held_plan_by_id) - scored_heldout_ids)
    missing_train = sorted(set(train_plan_by_id) - scored_train_ids)
    summaries = summarize_cohorts(heldout_res_objs, threshold_pct=cap_threshold)
    verdict = gate_decision(summaries, required_levels=REQUIRED_MULTILEVEL_LEVELS, require_disjoint_level_roots=True)
    missing_any = bool(missing_heldout or missing_train)
    passed = verdict["status"] == "pass" and not pw_fail and not missing_any and bool(abi_check["ok"])
    eff_st = (
        "pass" if passed
        else ("fail" if (pw_fail or not abi_check["ok"]) else ("insufficient_evidence" if missing_any else verdict["status"]))
    )
    return (
        GateCheckOutcome(
            name="multilevel_gate", passed=passed, status=eff_st,
            detail=(
                f"status={eff_st};levels={verdict['covered_levels']};missing={verdict['missing_levels']}"
                f";missing_heldout={len(missing_heldout)};missing_train={len(missing_train)}"
                f";abi_ok={abi_check['ok']};pointwise_failures={len(pw_fail)}"
            ),
            metrics=(
                ("covered_levels", verdict["covered_levels"]),
                ("missing_levels", verdict["missing_levels"]),
                ("missing_plan_points", len(missing_heldout)),
                ("missing_train_points", len(missing_train)),
                ("abi_ok", bool(abi_check["ok"])),
                ("overlapping_level_roots", verdict["overlapping_level_roots"]),
                ("failed_cohorts", verdict["failed_cohorts"]),
                ("pointwise_failures", sorted(pw_fail)),
                ("threshold_pct", cap_threshold),
                ("atol_ns", atol_ns),
            ),
        ),
        heldout_dicts,
    )


def _evaluate_manifest_path(
    *,
    sku: str,
    validation_doc: Mapping[str, Any] | None,
    multilevel_manifest_json: str | Path,
    fits_json: str | Path | None,
    runner: EvaluationRunner | None,
    primitive_table: PrimitiveTable | None,
    profile_override: Mapping[str, Any] | None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None,
    capture_root: str | Path | None,
    capture_files: Mapping[str, str | Path] | None,
    require_capture_files: bool,
    cap_threshold: float,
    atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS,
) -> tuple[GateCheckOutcome, list[dict[str, Any]]]:
    from ._sealed_scorer import ClockParityRunner, _bind_active_runner, _norm_digest
    from .cliff_planner import evaluate_abi_rejections, synthesize_sku_512_400k_plan

    if validation_doc is not None and (
        not verify_stage_report("sku_validation", validation_doc)
        or not validation_doc.get("ok")
        or str(validation_doc.get("sku") or "") != sku
    ):
        return (
            GateCheckOutcome(
                "multilevel_gate", False, "fail",
                "conflicting_failed_unverified_or_wrong_sku_validation_report",
            ),
            [],
        )

    raw_manifest = read_bounded_json(multilevel_manifest_json)
    if not isinstance(raw_manifest, Mapping) or not isinstance(raw_manifest.get("cases"), list):
        return GateCheckOutcome("multilevel_gate", False, "fail", "multilevel_manifest_invalid"), []

    hw = materialize_sku_hardware(sku, profile_override=profile_override)
    die = hw.dies[0]
    dp = datapath_for_die(die)
    l2_bytes = max(1, int(dict(die.buffers).get("l2", 0)))
    auth_plan = synthesize_sku_512_400k_plan(
        sku=sku, core_count=int(dp.aic_count), l2_bytes=l2_bytes, profile_override=profile_override
    ).to_dict()
    held_plan_by_id = {str(pt["point_id"]): pt for pt in auth_plan["heldout_points"]}
    train_plan_by_id = {str(pt["point_id"]): pt for pt in auth_plan["train_points"]}

    raw_cases: list[Mapping[str, Any]] = [c for c in raw_manifest["cases"] if isinstance(c, Mapping)]
    err_out, parsed_cases, expected_clock, case_meta, plan_by_case, resolved_files = _parse_manifest_cases(
        raw_cases, sku=sku, held_plan_by_id=held_plan_by_id, train_plan_by_id=train_plan_by_id,
        capture_root=capture_root, capture_files=capture_files, require_capture_files=require_capture_files,
    )
    if err_out is not None:
        return err_out, []

    if validation_doc is not None:
        manifest_pairs = {(gc.case_id, _norm_digest(gc.boundary.raw_sha256)) for gc in parsed_cases}
        report_pairs = {
            (str(r.get("case_id") or ""), _norm_digest(str(r.get("raw_sha256") or "")))
            for r in (validation_doc.get("residuals") or ())
            if isinstance(r, Mapping) and r.get("case_id") and r.get("raw_sha256")
        }
        if not manifest_pairs or not report_pairs or manifest_pairs != report_pairs:
            return GateCheckOutcome("multilevel_gate", False, "fail", "manifest_validation_report_capture_digest_mismatch"), []
        if primitive_table is not None:
            val_env = str(validation_doc.get("primitives_envelope_sha256") or "").strip()
            if not val_env or val_env != primitive_table.envelope_sha256:
                return GateCheckOutcome("multilevel_gate", False, "fail", "manifest_validation_report_primitives_envelope_mismatch"), []

    manifest = Manifest(
        manifest_id=str(raw_manifest.get("manifest_id") or f"manifest-{sku}"),
        schema_version=str(raw_manifest.get("schema_version") or "mindie-multilevel-manifest/1"),
        cases=tuple(parsed_cases),
        lineage_edges=tuple((str(e[0]), str(e[1])) for e in raw_manifest.get("lineage_edges", ())),
        strict_three_level_blind=True,
    )
    fits = _parse_fits(fits_json)
    eff_cap_files = resolved_files or capture_files
    validate_lineage(
        manifest, fits, capture_root=capture_root, capture_files=eff_cap_files,
        require_capture_files=require_capture_files,
    )
    active_runner = _bind_active_runner(
        runner, primitive_table=primitive_table, profile_override=profile_override
    )
    checked_runner = ClockParityRunner(active_runner, expected_clock, plan_by_case=plan_by_case)
    raw_residuals = list(
        evaluate_multilevel(
            manifest, fits, checked_runner, target_split="all",
            capture_root=capture_root, capture_files=eff_cap_files,
            require_capture_files=require_capture_files,
        )
    )
    eff_abi_results = abi_rejection_results
    if eff_abi_results is None and isinstance(raw_manifest.get("abi_rejection_results"), list):
        eff_abi_results = list(raw_manifest["abi_rejection_results"])
    abi_check = evaluate_abi_rejections(auth_plan["abi_rejection_points"], eff_abi_results)
    return _summarize_manifest_residuals(
        raw_residuals, case_meta=case_meta, expected_clock=expected_clock,
        engine_clocks=checked_runner.engine_clocks, held_plan_by_id=held_plan_by_id,
        train_plan_by_id=train_plan_by_id, abi_check=abi_check,
        cap_threshold=cap_threshold, atol_ns=atol_ns,
    )


def run_multilevel_verdict(
    *,
    sku: str,
    validation_doc: Mapping[str, Any] | None,
    multilevel_manifest_json: str | Path | None,
    fits_json: str | Path | None,
    runner: EvaluationRunner | None,
    primitive_table: PrimitiveTable | None = None,
    profile_override: Mapping[str, Any] | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = True,
    threshold_pct: float,
) -> tuple[GateCheckOutcome, list[dict[str, Any]]]:
    """Evaluate strict pointwise multilevel L1/L2/L3 gate verdict."""
    from ._sealed_scorer import (
        REQUIRED_MULTILEVEL_LEVELS,
        STANDARD_LEVEL_BOUNDARY_KINDS,
        exceeds_physical_tolerance,
        validate_disjoint_lineage,
    )

    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    if not require_capture_files:
        raise ModelingError("release_certification_requires_capture_files")
    if multilevel_manifest_json is not None:
        return _evaluate_manifest_path(
            sku=sku,
            validation_doc=validation_doc,
            multilevel_manifest_json=multilevel_manifest_json,
            fits_json=fits_json,
            runner=runner,
            primitive_table=primitive_table,
            profile_override=profile_override,
            abi_rejection_results=abi_rejection_results,
            capture_root=capture_root,
            capture_files=capture_files,
            require_capture_files=True,
            cap_threshold=cap_threshold,
        )

    if validation_doc is None:
        return GateCheckOutcome("multilevel_gate", False, "insufficient_evidence", "no validation report or multilevel manifest supplied"), []
    if not verify_stage_report("sku_validation", validation_doc):
        return GateCheckOutcome("multilevel_gate", False, "fail", "validation_report_hmac_unverified_or_fabricated"), []
    if str(validation_doc.get("sku") or "") != sku:
        return GateCheckOutcome("multilevel_gate", False, "fail", f"validation_report_sku_mismatch:expected={sku}"), []

    heldout_rows = [r for r in (validation_doc.get("residuals") or ()) if str(r.get("split") or "heldout") == "heldout"]
    all_report_rows = [r for r in (validation_doc.get("residuals") or ()) if isinstance(r, Mapping)]
    missing_plan = list(validation_doc.get("missing_plan_points") or ())
    missing_train = list(validation_doc.get("missing_train_points") or ())
    missing_anchors = list(validation_doc.get("missing_anchors") or ())
    missing_cliffs = list(validation_doc.get("missing_cliffs") or ())
    abi_ok = bool((validation_doc.get("abi_rejection_check") or {}).get("ok", False))
    if not heldout_rows:
        return GateCheckOutcome("multilevel_gate", False, "insufficient_evidence", "validation report contains zero heldout residuals"), []
    if validation_doc.get("capture_files_verified") is not True:
        return GateCheckOutcome("multilevel_gate", False, "fail", "capture_files_unverified_for_release"), heldout_rows
    reverify_report_captures(
        all_report_rows,
        capture_root=capture_root,
        capture_files=capture_files,
        require_capture_files=True,
    )
    for row in heldout_rows:
        rcpt = str(row.get("engine_receipt") or "")
        if not rcpt.startswith(_ENGINE_RECEIPT_PREFIX):
            return GateCheckOutcome("multilevel_gate", False, "fail", f"unverified_engine_receipt:{row.get('case_id')}"), heldout_rows
    validate_disjoint_lineage(heldout_rows, require_disjoint_level_roots=True)
    res_objs: list[Residual] = []
    pw_fail: list[str] = []
    eff_atol_ns = max(0, int(validation_doc.get("atol_ns", DEFAULT_PHYSICAL_ATOL_NS)))
    for row in heldout_rows:
        lvl = str(row.get("level") or "L1").upper()
        c_raw = row.get("cohort") if isinstance(row.get("cohort"), Mapping) else {}
        cohort = CohortKey(lvl, str(c_raw.get("cohort_name") or "onboarding"), str(c_raw.get("sku") or sku), str(c_raw.get("phase") or "decode"), str(c_raw.get("mode") or "eager"), str(c_raw.get("shape_regime") or "cliff"), str(c_raw.get("parallel_topology") or "tp1"))  # type: ignore[arg-type]
        meas_ns = int(row.get("measured_ns") or 1)
        pred_ns = int(row.get("predicted_ns") or 1)
        ape = float(row.get("absolute_percentage_error", 0.0))
        if exceeds_physical_tolerance(pred_ns, meas_ns, threshold_pct=cap_threshold, atol_ns=eff_atol_ns):
            pw_fail.append(str(row.get("case_id") or ""))
        res_objs.append(Residual(str(row.get("case_id") or ""), lvl, cohort, str(row.get("acquisition_root") or ""), meas_ns, pred_ns, str(row.get("engine_receipt") or ""), float(row.get("signed_error", 0.0)), ape, "ok", boundary_kind=str(row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS.get(lvl, "device_kernel"))))
    summaries = summarize_cohorts(res_objs, threshold_pct=cap_threshold)
    verdict = gate_decision(summaries, required_levels=REQUIRED_MULTILEVEL_LEVELS, require_disjoint_level_roots=True)
    missing_any = bool(missing_plan or missing_train or missing_anchors or missing_cliffs)
    passed = bool(validation_doc.get("ok")) and abi_ok and verdict["status"] == "pass" and not pw_fail and not missing_any
    eff_status = "pass" if passed else ("fail" if (pw_fail or not abi_ok or not validation_doc.get("ok")) else ("insufficient_evidence" if missing_any else verdict["status"]))
    return (
        GateCheckOutcome(
            name="multilevel_gate", passed=passed, status=eff_status,
            detail=f"status={eff_status};levels={verdict['covered_levels']};missing_levels={verdict['missing_levels']};missing_plan={len(missing_plan)};missing_train={len(missing_train)};pointwise_failures={len(pw_fail)};abi_ok={abi_ok}",
            metrics=(("covered_levels", verdict["covered_levels"]), ("missing_levels", verdict["missing_levels"]), ("missing_plan_points", len(missing_plan)), ("missing_train_points", len(missing_train)), ("missing_anchors", len(missing_anchors)), ("missing_cliffs", len(missing_cliffs)), ("overlapping_level_roots", verdict["overlapping_level_roots"]), ("failed_cohorts", verdict["failed_cohorts"]), ("pointwise_failures", sorted(pw_fail)), ("abi_ok", abi_ok), ("threshold_pct", cap_threshold), ("atol_ns", eff_atol_ns)),
        ),
        heldout_rows,
    )


__all__ = ["run_multilevel_verdict"]
