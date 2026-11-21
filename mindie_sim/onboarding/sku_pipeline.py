"""Turnkey 4-stage New-SKU Onboarding Pipeline for Ascend hardware.

[Role] Orchestrate deterministic, repeatable onboarding of a new or existing
  Ascend SKU across four governed stages plus ``run-all``:
  (1) ``probe-soc`` (cross-checked multi-source silicon probe),
  (2) ``calibrate-l0-and-fabric`` (exact-grant ``HostQueue`` lease + installed
      ``mindie_sim.primitives.sealer`` + CANN manifest/capture cross-check +
      per-metric conflict diagnostics & coverage matrix),
  (3) ``validate-512-400k`` (disjoint 2D Anchor + Off-Grid Cliff +
      ``ParallelSpec``-validated multi-rank + ABI-guard plan scored strictly
      via ``GateCase -> EvaluationRunner -> EngineRunReceipt`` bound to exact
      required NPUs & ``ttl >= timeout + cleanup_grace``), and
  (4) ``verify-gates`` (HMAC-verified report or full-plan manifest replay with
      verified raw capture bytes, mandatory SKU/CANN-matched primitive envelope,
      ``(program, dag)``-joined & p99-verified Lean 4 non-inversion, strict repo
      gates, automatic ``diagnose_tail_cluster``, and SHA-256 receipt
      distinguishing ``silicon_certified`` vs ``offline_reverify``).
[Invariants] Zero duplicate primitive sealers; zero fabricated report passes;
  thresholds ``> 10.0%`` rejected; ``calibrated_count == 0`` and ``plan_only``
  ALWAYS set ``ok=False``; offline retained runs emit ``offline_reverify`` and
  NEVER ``silicon_certified``; ``TimeoutError`` propagates before ``OSError``.
[Timeouts & Locks] Bounded execution budgets; ``HostQueue`` leases always unwind
  via context managers and bind exact required physical devices into subprocesses.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``; public receipts carry opaque sanitized lease/endpoint digests.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
import tempfile
from typing import Any, Literal

from mindie_sim.backends import ascend_profiles
from mindie_sim.observations.multilevel_gate import EvaluationRunner
from mindie_sim.primitives.schema import PrimitiveTable, parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die

from ._capture_truth import verify_live_silicon_evidence
from ._gate_runners import (
    acquire_coordinator_host_queue_lease,
    resolve_repo_root,
    run_arch_governance_gate,
    run_profiler_bench_subprocess,
    run_tail_cluster_diagnosis,
    run_test_orthogonality_gate,
    validate_host_queue_grant,
)
from ._lean_and_receipts import (
    check_lean_non_inversion,
    seal_onboarding_receipt,
    sign_stage_report,
    validate_strict_atol_ns,
    validate_strict_threshold_pct,
    verify_stage_report,
)
from ._profiler_plan import assign_plan_point_level
from ._sealed_scorer import (
    REQUIRED_MULTILEVEL_LEVELS,
    exceeds_physical_tolerance,
    run_multilevel_verdict,
)
from ._sku_calibration import (
    calibrate_l0_and_fabric,
    materialize_sku_hardware,
    temporary_scaffold_profile,
)
from ._soc_probe import (
    parse_npu_smi_topology,
    parse_platform_config_ini,
    probe_soc,
    verify_idle_die_telemetry,
)
from .cliff_planner import (
    evaluate_abi_rejections,
    required_ranks_for_plan,
    score_sealed_gate_cases,
    synthesize_sku_512_400k_plan,
)
from .contracts import (
    ANCHOR_CONTEXT_GRID,
    DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DEFAULT_L0_BUDGET_S,
    DEFAULT_PHYSICAL_ATOL_NS,
    DISPLAY_ROUND_DIGITS,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    SHORT_DIGEST_LEN,
    GateCheckOutcome,
    assert_sanitized_payload,
    collect_receipt_warnings,
    read_bounded_json,
    sanitize_lease_for_receipt,
    write_atomic_json,
)


def _collect_sku_validation_inputs(
    *,
    sku: str,
    plan_doc: Mapping[str, Any],
    observations_json: str | Path | None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None,
    profiler_runner: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None,
    live_profiler: bool,
    endpoint_alias: str,
    profiler_root: str | Path | None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None,
    coordinator_bin: str,
    eff_capture_root: str | Path | None,
    ssh_runner: Callable[[str, str, int], Any] | None,
) -> tuple[
    list[Mapping[str, Any]] | None,
    Sequence[Mapping[str, Any]] | None,
    dict[str, Any],
    Literal["live_silicon", "offline_reverify"],
]:
    lease_public: dict[str, Any] = {"lease_mode": "offline_replay"}
    acquisition_mode: Literal["live_silicon", "offline_reverify"] = "offline_reverify"
    if observations_json is not None:
        raw_obs = read_bounded_json(observations_json)
        if isinstance(raw_obs, Mapping):
            obs_rows: list[Mapping[str, Any]] | None = list(
                raw_obs.get("observations") or raw_obs.get("cases") or ()
            )
            if abi_rejection_results is None and raw_obs.get("abi_rejection_results") is not None:
                abi_rejection_results = list(raw_obs["abi_rejection_results"])
        elif isinstance(raw_obs, list):
            obs_rows = list(raw_obs)
        else:
            raise ModelingError("validate_512_400k_observations_invalid")
        return obs_rows, abi_rejection_results, lease_public, acquisition_mode

    if profiler_runner is not None:
        if lease_acquirer is None and not live_profiler:
            return list(profiler_runner(plan_doc)), abi_rejection_results, lease_public, acquisition_mode
        req_npus = required_ranks_for_plan(plan_doc)
        lease_ttl_s = DEFAULT_L0_BUDGET_S + DEFAULT_CLEANUP_GRACE_S
        lease_ctx = (
            lease_acquirer(sku, lease_ttl_s)
            if lease_acquirer is not None
            else acquire_coordinator_host_queue_lease(
                endpoint_alias, npu_count=req_npus, ttl_seconds=lease_ttl_s,
                min_subprocess_timeout_s=DEFAULT_L0_BUDGET_S,
                cleanup_grace_s=DEFAULT_CLEANUP_GRACE_S, coordinator_bin=coordinator_bin,
            )
        )
        raw_lease: Mapping[str, Any] = {}
        with lease_ctx as lease_info:
            raw_lease = lease_info if isinstance(lease_info, Mapping) else {}
            validate_host_queue_grant(raw_lease, required_devices=req_npus, min_ttl_seconds=lease_ttl_s)
            obs_rows = list(profiler_runner(plan_doc))
        return obs_rows, abi_rejection_results, sanitize_lease_for_receipt(raw_lease), acquisition_mode

    if live_profiler:
        req_npus = required_ranks_for_plan(plan_doc)
        lease_ttl_s = DEFAULT_L0_BUDGET_S + DEFAULT_CLEANUP_GRACE_S
        lease_ctx = (
            lease_acquirer(sku, lease_ttl_s)
            if lease_acquirer is not None
            else acquire_coordinator_host_queue_lease(
                endpoint_alias, npu_count=req_npus, ttl_seconds=lease_ttl_s,
                min_subprocess_timeout_s=DEFAULT_L0_BUDGET_S,
                cleanup_grace_s=DEFAULT_CLEANUP_GRACE_S, coordinator_bin=coordinator_bin,
            )
        )
        raw_lease = {}
        with lease_ctx as lease_info:
            raw_lease = lease_info if isinstance(lease_info, Mapping) else {}
            _, fence, granted, _ = validate_host_queue_grant(
                raw_lease, required_devices=req_npus, min_ttl_seconds=lease_ttl_s
            )
            obs_rows, live_abi = run_profiler_bench_subprocess(
                plan_doc, sku=sku, profiler_root=profiler_root, endpoint_alias=endpoint_alias,
                allowed_phy_ids=granted, fence_token=fence, timeout_s=DEFAULT_L0_BUDGET_S,
                capture_out_dir=eff_capture_root, ssh_runner=ssh_runner,
            )
            if abi_rejection_results is None and live_abi:
                abi_rejection_results = live_abi
        lease_public = sanitize_lease_for_receipt(raw_lease)
        if lease_public.get("lease_mode") == "coordinator_host_queue":
            acquisition_mode = "live_silicon"
        return obs_rows, abi_rejection_results, lease_public, acquisition_mode

    return None, abi_rejection_results, lease_public, acquisition_mode


def _extract_external_train_tokens(
    plan_train_root: str,
    primitives_json: str | Path | Mapping[str, Any] | None,
    train_manifest_json: str | Path | Mapping[str, Any] | None,
) -> tuple[PrimitiveTable | None, list[str], list[str], list[str]]:
    parsed_table: PrimitiveTable | None = None
    roots: list[str] = [plan_train_root]
    digests: list[str] = []
    uris: list[str] = []
    if primitives_json is not None:
        p_doc = read_bounded_json(primitives_json) if isinstance(primitives_json, (str, Path)) else dict(primitives_json)
        parsed_table = parse_primitive_table(p_doc, verify_envelope=True)
        digests.append(parsed_table.envelope_sha256)
        prim_uri = str(parsed_table.provenance.capture_uri or "").strip()
        if prim_uri:
            uris.append(prim_uri)
            digests.append(prim_uri.rsplit("/", 1)[-1])
        for cell in parsed_table.primitives.values():
            ev_id = str(cell.evidence_id or "").strip()
            if ev_id:
                uris.append(ev_id)
                digests.append(ev_id.rsplit("/", 1)[-1])
    if train_manifest_json is not None:
        t_doc = read_bounded_json(train_manifest_json) if isinstance(train_manifest_json, (str, Path)) else dict(train_manifest_json)
        if t_doc.get("train_lineage_root"):
            roots.append(str(t_doc["train_lineage_root"]))
        digests.extend(str(d) for d in (t_doc.get("train_digests") or ()))
        if t_doc.get("primitives_envelope_sha256"):
            digests.append(str(t_doc["primitives_envelope_sha256"]))
        uris.extend(str(u) for u in (t_doc.get("train_capture_uris") or ()))
        if t_doc.get("capture_uri"):
            uris.append(str(t_doc["capture_uri"]))
    return parsed_table, roots, digests, uris


def _build_sku_plan_only_report(
    *,
    sku: str,
    plan: Any,
    plan_doc: Mapping[str, Any],
    lease_public: Mapping[str, Any],
    expected_heldout_ids: Sequence[str],
    expected_train_ids: Sequence[str],
    report_out: str | Path | None,
) -> dict[str, Any]:
    plan_only: dict[str, Any] = {
        "stage": "validate-512-400k", "sku": sku, "ok": False,
        "status": "plan_only", "verdict": "insufficient_evidence",
        "error": "validate_512_400k_plan_only_no_observations",
        "plan_sha256": plan.plan_sha256, "plan": plan_doc,
        "required_ranks": required_ranks_for_plan(plan_doc),
        "acquisition_mode": "offline_reverify", "capture_files_verified": False,
        "lease": dict(lease_public),
        "missing_plan_points": sorted(expected_heldout_ids),
        "missing_train_points": sorted(expected_train_ids),
        "missing_anchors": list(ANCHOR_CONTEXT_GRID),
        "missing_cliffs": list(OFF_GRID_CLIFF_HELDOUT_GRID),
        "abi_rejection_check": evaluate_abi_rejections(plan_doc["abi_rejection_points"], None),
        "heldout_summary": None, "residuals": [],
    }
    plan_only["report_hmac"] = sign_stage_report("sku_validation", plan_only)
    assert_sanitized_payload(plan_only, field_name="validate_512_400k_plan_only")
    if report_out is not None:
        write_atomic_json(report_out, plan_only)
    return plan_only


def validate_512_400k(
    *,
    sku: str,
    observations_json: str | Path | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    train_manifest_json: str | Path | Mapping[str, Any] | None = None,
    primitives_json: str | Path | Mapping[str, Any] | None = None,
    plan_out: str | Path | None = None,
    report_out: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    campaign_id: str = "camp-onboard-512-400k",
    runner: EvaluationRunner | None = None,
    profiler_runner: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    live_profiler: bool = False,
    endpoint_alias: str = "",
    profiler_root: str | Path | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
    coordinator_bin: str = "",
    profile_override: Mapping[str, Any] | None = None,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = True,
    ssh_runner: Callable[[str, str, int], Any] | None = None,
) -> dict[str, Any]:
    """Stage 3: Orchestrate and score disjoint 2D Anchor + Cliff 512..400K + L1/L2/L3."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    hw = materialize_sku_hardware(sku, profile_override=profile_override)
    die = hw.dies[0]
    dp = datapath_for_die(die)
    l2_bytes = int(dict(die.buffers).get("l2", 0))
    plan = synthesize_sku_512_400k_plan(
        sku=sku, core_count=int(dp.aic_count), l2_bytes=max(1, l2_bytes),
        campaign_id=campaign_id, profile_override=profile_override,
    )
    plan_doc = plan.to_dict()
    assert_sanitized_payload(plan_doc, field_name="sku_cliff_plan")
    if plan_out is not None:
        write_atomic_json(plan_out, plan_doc)

    parsed_table, external_train_roots, external_train_digests, external_train_uris = (
        _extract_external_train_tokens(plan.train_lineage_root, primitives_json, train_manifest_json)
    )
    eff_capture_root: str | Path | None = capture_root
    temp_cap_ctx: AbstractContextManager[str | None] = nullcontext(None)
    if live_profiler and observations_json is None and profiler_runner is None and eff_capture_root is None:
        if report_out is not None:
            eff_capture_root = Path(report_out).parent / "captures"
        else:
            temp_cap_ctx = tempfile.TemporaryDirectory(prefix="msim-sku-cap-")

    expected_heldout_pts = list(plan_doc["heldout_points"])
    expected_train_pts = list(plan_doc["train_points"])
    expected_heldout_ids = [str(pt["point_id"]) for pt in expected_heldout_pts]
    expected_train_ids = [str(pt["point_id"]) for pt in expected_train_pts]

    with temporary_scaffold_profile(sku, profile_override), temp_cap_ctx as tmp_cap_dir:
        if tmp_cap_dir is not None:
            eff_capture_root = Path(tmp_cap_dir)
        obs_rows, abi_rejection_results, lease_public, acquisition_mode = _collect_sku_validation_inputs(
            sku=sku, plan_doc=plan_doc, observations_json=observations_json,
            abi_rejection_results=abi_rejection_results, profiler_runner=profiler_runner,
            live_profiler=live_profiler, endpoint_alias=endpoint_alias, profiler_root=profiler_root,
            lease_acquirer=lease_acquirer, coordinator_bin=coordinator_bin,
            eff_capture_root=eff_capture_root, ssh_runner=ssh_runner,
        )
        if obs_rows is None:
            return _build_sku_plan_only_report(
                sku=sku, plan=plan, plan_doc=plan_doc, lease_public=lease_public,
                expected_heldout_ids=expected_heldout_ids, expected_train_ids=expected_train_ids,
                report_out=report_out,
            )

        expected_levels = {
            str(pt["point_id"]): assign_plan_point_level(plan_doc, pt)
            for pt in (*expected_train_pts, *expected_heldout_pts)
            if isinstance(pt, Mapping) and pt.get("point_id")
        }
        scored = score_sealed_gate_cases(
            obs_rows, sku=sku, runner=runner, primitive_table=parsed_table,
            profile_override=profile_override, threshold_pct=cap_threshold,
            external_train_roots=external_train_roots,
            external_train_digests=external_train_digests,
            external_train_uris=external_train_uris,
            expected_points=expected_heldout_pts, expected_train_points=expected_train_pts,
            expected_point_ids=expected_heldout_ids, expected_levels=expected_levels,
            require_disjoint_level_roots=True,
            capture_root=eff_capture_root, capture_files=capture_files,
            require_capture_files=require_capture_files,
        )
    if not scored.get("capture_files_verified"):
        acquisition_mode = "offline_reverify"
    abi_check = evaluate_abi_rejections(plan_doc["abi_rejection_points"], abi_rejection_results)
    train_plan_by_id = {str(pt["point_id"]): int(pt["value"]) for pt in expected_train_pts if pt.get("axis") == "context_len"}
    held_plan_by_id = {str(pt["point_id"]): int(pt["value"]) for pt in expected_heldout_pts if pt.get("axis") == "context_len"}
    train_ctxs = {
        train_plan_by_id[str(r["point_id"])]
        for r in scored["residuals"]
        if r.get("split") == "train" and r.get("status") == "ok" and str(r.get("point_id")) in train_plan_by_id
    }
    heldout_ctxs = {
        held_plan_by_id[str(r["point_id"])]
        for r in scored["residuals"]
        if r.get("split") == "heldout" and r.get("status") == "ok" and str(r.get("point_id")) in held_plan_by_id
    }
    missing_anchors = [c for c in ANCHOR_CONTEXT_GRID if c not in train_ctxs]
    missing_cliffs = [c for c in OFF_GRID_CLIFF_HELDOUT_GRID if c not in heldout_ctxs]
    missing_plan_points = list(scored["missing_plan_points"])
    missing_train_points = list(scored["missing_train_points"])

    if missing_plan_points or missing_train_points or missing_anchors or missing_cliffs or scored["status"] == "insufficient_evidence":
        final_ok, final_status = False, "insufficient_evidence"
    elif scored["failing_cases"] or not scored["ok"] or not abi_check["ok"]:
        final_ok, final_status = False, "fail"
    else:
        final_ok, final_status = True, "pass"

    report: dict[str, Any] = {
        "stage": "validate-512-400k", "sku": sku, "ok": final_ok,
        "status": final_status, "verdict": final_status,
        "threshold_pct": cap_threshold, "atol_ns": int(scored.get("atol_ns", DEFAULT_PHYSICAL_ATOL_NS)),
        "plan_sha256": plan.plan_sha256, "required_ranks": required_ranks_for_plan(plan_doc),
        "acquisition_mode": acquisition_mode,
        "capture_files_verified": bool(scored.get("capture_files_verified", False)),
        "lease": lease_public,
        "primitives_envelope_sha256": parsed_table.envelope_sha256 if parsed_table is not None else "",
        "anchor_grid": list(ANCHOR_CONTEXT_GRID), "off_grid_cliff_grid": list(OFF_GRID_CLIFF_HELDOUT_GRID),
        "missing_plan_points": missing_plan_points, "missing_train_points": missing_train_points,
        "missing_anchors": missing_anchors, "missing_cliffs": missing_cliffs,
        "over_budget_points": list(scored["over_budget_points"]),
        "failing_cases": list(scored["failing_cases"]),
        "abi_rejection_check": abi_check, "lineage": scored["lineage"],
        "heldout_summary": scored["heldout_summary"], "train_summary": scored["train_summary"],
        "residuals": scored["residuals"], "plan": plan_doc,
    }
    rec_unconf, rec_warn = collect_receipt_warnings(lease_public)
    if rec_unconf or rec_warn:
        report["receipt_unconfirmed"] = rec_unconf
        report["receipt_warning"] = rec_warn
    report["report_hmac"] = sign_stage_report("sku_validation", report)
    assert_sanitized_payload(report, field_name="validate_512_400k_report")
    if report_out is not None:
        write_atomic_json(report_out, report)
    return report


def _evaluate_sku_primitives_gate(
    *,
    sku: str,
    primitives_json: str | Path | Mapping[str, Any] | None,
    expected_cann_version: str,
    val_doc: Mapping[str, Any] | None,
) -> tuple[GateCheckOutcome, PrimitiveTable | None, str]:
    if primitives_json is None:
        return GateCheckOutcome("primitives_envelope", False, "fail", "primitives_json_required_for_sku_certification"), None, ""
    prim_doc = read_bounded_json(primitives_json) if isinstance(primitives_json, (str, Path)) else dict(primitives_json)
    try:
        table = parse_primitive_table(prim_doc, verify_envelope=True)
    except ModelingError as exc:
        return GateCheckOutcome("primitives_envelope", False, "fail", str(exc)), None, ""
    cal_cnt = sum(1 for cell in table.primitives.values() if cell.status == "calibrated")
    sku_match = table.sku == sku
    cann_match = (not expected_cann_version) or (table.cann_version == expected_cann_version)
    stage3_env = str((val_doc or {}).get("primitives_envelope_sha256") or "").strip()
    env_match = True if val_doc is None else (bool(stage3_env) and stage3_env == table.envelope_sha256)
    prim_ok = cal_cnt > 0 and sku_match and cann_match and env_match
    reasons: list[str] = []
    if not sku_match:
        reasons.append(f"primitives_sku_mismatch:table={table.sku}:target={sku}")
    if not cann_match:
        reasons.append(f"primitives_cann_mismatch:table={table.cann_version}:expected={expected_cann_version}")
    if not env_match:
        reasons.append(f"primitives_stage3_stage4_envelope_mismatch:stage3={stage3_env or 'missing'}:stage4={table.envelope_sha256[:SHORT_DIGEST_LEN]}")
    if cal_cnt == 0:
        reasons.append("l0_calibration_zero_calibrated_primitives")
    outcome = GateCheckOutcome(
        name="primitives_envelope", passed=prim_ok, status="pass" if prim_ok else "fail",
        detail=f"calibrated={cal_cnt}/{len(table.primitives)};cann={table.cann_version};envelope={table.envelope_sha256[:SHORT_DIGEST_LEN]}" if prim_ok else "; ".join(reasons),
        metrics=(("calibrated_count", cal_cnt), ("total_primitives", len(table.primitives)), ("sku_match", sku_match), ("cann_match", cann_match), ("stage3_envelope_match", env_match)),
    )
    return outcome, table, table.envelope_sha256


def _evaluate_sku_lean_gate(
    diag_residuals: Sequence[Mapping[str, Any]],
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None,
) -> tuple[GateCheckOutcome, list[Any]]:
    expected_timings = {
        str(r["case_id"]): (
            int(r["predicted_ns"]), int(r["measured_ns"]),
            int(r["measured_p99_ns"]) if r.get("measured_p99_ns") is not None else None,
            str(r.get("program_dag_sha256") or ""),
        )
        for r in diag_residuals
        if r.get("case_id") and r.get("predicted_ns") and r.get("measured_ns")
    }
    if isinstance(lean_bounds_json, (str, Path)):
        raw_lean = read_bounded_json(lean_bounds_json)
        lean_rows = raw_lean.get("bounds", raw_lean) if isinstance(raw_lean, Mapping) else raw_lean
    else:
        lean_rows = list(lean_bounds_json or ())
    lean_ok, lean_checked, lean_violations = check_lean_non_inversion(
        lean_rows, require_non_empty=True, verify_certificates=True,
        require_captured_p99=True, require_program_dag_join=True,
        expected_case_timings=expected_timings if expected_timings else None,
    )
    if not expected_timings:
        lean_ok = False
        lean_violations = (*lean_violations, "lean_no_scored_heldout_cases_to_join")
    outcome = GateCheckOutcome(
        name="lean_optimality_non_inversion", passed=lean_ok, status="pass" if lean_ok else "fail",
        detail="0 inversions (witness joined & verified)" if lean_ok else "; ".join(lean_violations),
        metrics=(("cases_checked", len(lean_checked)), ("violations", len(lean_violations))),
    )
    return outcome, list(lean_checked)


def verify_sku_gates(
    *,
    sku: str,
    validation_report_json: str | Path | Mapping[str, Any] | None = None,
    multilevel_manifest_json: str | Path | None = None,
    fits_json: str | Path | None = None,
    runner: EvaluationRunner | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    primitives_json: str | Path | Mapping[str, Any] | None = None,
    expected_cann_version: str = "",
    profile_override: Mapping[str, Any] | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = True,
    repo_root: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    out_receipt: str | Path | None = None,
    diagnosis_out: str | Path | None = None,
    acquisition_mode: Literal["live_silicon", "offline_reverify"] | None = None,
    stage2_lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 4: Execute multilevel, mandatory SKU/CANN primitives, joined Lean, repo gates, receipt."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    if not require_capture_files:
        raise ModelingError("release_certification_requires_capture_files")
    root = resolve_repo_root(repo_root)
    val_doc = (
        read_bounded_json(validation_report_json)
        if isinstance(validation_report_json, (str, Path))
        else (dict(validation_report_json) if isinstance(validation_report_json, Mapping) else None)
    )
    gates: list[GateCheckOutcome] = []
    artifacts: dict[str, str] = {}

    sku_registered = sku in ascend_profiles() or profile_override is not None
    gates.append(
        GateCheckOutcome(
            name="sku_profile_registered", passed=sku_registered,
            status="pass" if sku_registered else "fail",
            detail="registered" if sku_registered else f"unknown_sku:{sku}",
        )
    )
    prim_gate, parsed_table, prim_env_sha = _evaluate_sku_primitives_gate(
        sku=sku, primitives_json=primitives_json,
        expected_cann_version=expected_cann_version, val_doc=val_doc,
    )
    gates.append(prim_gate)
    if prim_env_sha:
        artifacts["primitives_envelope_sha256"] = prim_env_sha

    ml_outcome, diag_residuals = run_multilevel_verdict(
        sku=sku, validation_doc=val_doc, multilevel_manifest_json=multilevel_manifest_json,
        fits_json=fits_json, runner=runner, primitive_table=parsed_table,
        profile_override=profile_override, abi_rejection_results=abi_rejection_results,
        capture_root=capture_root, capture_files=capture_files,
        require_capture_files=require_capture_files, threshold_pct=cap_threshold,
    )
    gates.append(ml_outcome)

    eff_atol_ns = validate_strict_atol_ns((val_doc or {}).get("atol_ns", DEFAULT_PHYSICAL_ATOL_NS))
    failing_cliffs = [
        r for r in diag_residuals
        if exceeds_physical_tolerance(
            int(r.get("predicted_ns") or 0), int(r.get("measured_ns") or 1),
            threshold_pct=cap_threshold, atol_ns=eff_atol_ns,
        )
    ]
    diagnosis_report: dict[str, Any] | None = None
    if failing_cliffs:
        diagnosis_report = run_tail_cluster_diagnosis(
            diag_residuals, repo_root=root, threshold_pct=cap_threshold, output_path=diagnosis_out,
        )

    lean_gate, lean_checked = _evaluate_sku_lean_gate(diag_residuals, lean_bounds_json)
    gates.append(lean_gate)
    if run_repo_gates:
        gates.append(run_arch_governance_gate(repo_root=root, strict=True))
        gates.append(run_test_orthogonality_gate(repo_root=root, strict=orthogonality_base is None, base=orthogonality_base))

    if val_doc and verify_stage_report("sku_validation", val_doc) and val_doc.get("ok"):
        if val_doc.get("plan_sha256"):
            artifacts["plan_sha256"] = str(val_doc["plan_sha256"])
        if isinstance(val_doc.get("heldout_summary"), Mapping):
            max_ape = val_doc["heldout_summary"].get("max_ape_pct")
            if max_ape is not None:
                artifacts["heldout_max_ape_pct"] = str(round(float(max_ape), DISPLAY_ROUND_DIGITS))

    val_hmac_ok = val_doc is not None and verify_stage_report("sku_validation", val_doc)
    prim_env_matches_val = (
        parsed_table is not None
        and val_doc is not None
        and str(val_doc.get("primitives_envelope_sha256") or "").strip() == parsed_table.envelope_sha256
    )
    live_verified = (
        val_hmac_ok and ml_outcome.passed and prim_env_matches_val
        and verify_live_silicon_evidence(val_doc, stage2_lease=stage2_lease)
    )
    if acquisition_mode == "live_silicon" and not live_verified:
        gates.append(
            GateCheckOutcome(
                name="acquisition_mode_evidence", passed=False, status="fail",
                detail="live_silicon_claim_unverified_by_host_queue_and_captures",
            )
        )
    eff_mode: Literal["live_silicon", "offline_reverify"] = (
        "live_silicon" if (live_verified and acquisition_mode != "offline_reverify") else "offline_reverify"
    )
    rec_unconf, rec_warn = collect_receipt_warnings(val_doc, stage2_lease)
    receipt = seal_onboarding_receipt(
        kind="sku", target_id=sku, sku=sku, gates=gates, artifacts=artifacts,
        diagnosis_report=diagnosis_report, acquisition_mode=eff_mode,
        receipt_unconfirmed=rec_unconf, receipt_warning=rec_warn,
    )
    receipt_dict = receipt.to_dict()
    if out_receipt is not None:
        write_atomic_json(out_receipt, receipt_dict)
    out: dict[str, Any] = {
        "stage": "verify-gates", "sku": sku, "ok": receipt.ok,
        "status": receipt.status, "acquisition_mode": receipt.acquisition_mode,
        "receipt_sha256": receipt.receipt_sha256,
        "lean_checks": [row.to_dict() for row in lean_checked],
        "diagnosis_triggered": diagnosis_report is not None,
        "receipt": receipt_dict,
    }
    if rec_unconf or rec_warn:
        out["receipt_unconfirmed"] = rec_unconf
        out["receipt_warning"] = rec_warn
    return out


def run_sku_onboarding(
    *,
    sku: str | None = None,
    endpoint_alias: str = "",
    probe_json: str | Path | None = None,
    raw_l0_json: str | Path | None = None,
    raw_fabric_json: str | Path | None = None,
    observations_json: str | Path | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    out_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_scaffold: bool = False,
    base_template_sku: str = "",
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    cann_version: str = "unmeasured",
    runner: EvaluationRunner | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
    l0_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
    fabric_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
    profiler_runner: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    live_bench: bool = False,
    live_profiler: bool = False,
    profiler_root: str | Path | None = None,
    coordinator_bin: str = "",
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = True,
    ssh_runner: Callable[[str, str, int], Any] | None = None,
) -> dict[str, Any]:
    """Execute all 4 New-SKU onboarding stages end-to-end with real owners and fail-closed gating."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    if not require_capture_files:
        raise ModelingError("release_certification_requires_capture_files")
    base_out = Path(out_dir) if out_dir is not None else None
    stage1 = probe_soc(
        endpoint_alias=endpoint_alias, probe_json=probe_json,
        allow_scaffold=allow_scaffold, base_template_sku=base_template_sku,
        scaffold_out=(base_out / "scaffold_profile.json") if (base_out and allow_scaffold) else None,
    )
    probed_sku = str(stage1.get("sku") or "")
    resolved_sku = str(sku or probed_sku)
    if sku and probed_sku and str(sku) != probed_sku:
        return {
            "stage": "run-all", "sku": resolved_sku, "ok": False, "status": "failed_at_probe_soc",
            "error": f"probe_sku_mismatch:requested={sku}:probed={probed_sku}", "stages": {"probe-soc": stage1},
        }
    if not stage1["ok"]:
        return {
            "stage": "run-all", "sku": resolved_sku, "ok": False,
            "status": "failed_at_probe_soc", "stages": {"probe-soc": stage1},
        }

    probe_cann = str(stage1.get("probe_cann_version") or "").strip()
    profile_override = (
        stage1["scaffold_profile"][resolved_sku]
        if isinstance(stage1.get("scaffold_profile"), Mapping) and resolved_sku in stage1["scaffold_profile"]
        else None
    )
    stage2 = calibrate_l0_and_fabric(
        sku=resolved_sku, endpoint_alias=endpoint_alias,
        raw_l0_json=raw_l0_json, raw_fabric_json=raw_fabric_json,
        out_primitives=(base_out / "primitives.json") if base_out else None,
        manifest_path=(base_out / "primitives_manifest.json") if base_out else None,
        train_manifest_out=(base_out / "train_manifest.json") if base_out else None,
        cann_version=cann_version, probe_cann_version=probe_cann,
        lease_acquirer=lease_acquirer, l0_runner=l0_runner, fabric_runner=fabric_runner,
        live_bench=live_bench, repo_root=repo_root, profiler_root=profiler_root,
        coordinator_bin=coordinator_bin, profile_override=profile_override,
        ssh_runner=ssh_runner,
    )
    if not stage2["ok"]:
        return {
            "stage": "run-all", "sku": resolved_sku, "ok": False,
            "status": "failed_at_calibrate_l0_and_fabric",
            "stages": {"probe-soc": stage1, "calibrate-l0-and-fabric": stage2},
        }

    eff_cap_root: str | Path | None = capture_root if capture_root is not None else ((base_out / "captures") if (live_profiler and base_out) else None)
    cap_ctx: AbstractContextManager[str | None] = (
        tempfile.TemporaryDirectory(prefix="msim-sku-runall-cap-")
        if (live_profiler and eff_cap_root is None and observations_json is None and profiler_runner is None)
        else nullcontext(None)
    )
    with cap_ctx as tmp_cap_dir:
        if tmp_cap_dir is not None:
            eff_cap_root = Path(tmp_cap_dir)
        stage3 = validate_512_400k(
            sku=resolved_sku, observations_json=observations_json,
            abi_rejection_results=abi_rejection_results,
            train_manifest_json=stage2["train_manifest"],
            primitives_json=stage2["primitives_payload"],
            plan_out=(base_out / "cliff_plan.json") if base_out else None,
            report_out=(base_out / "validation_512_400k.json") if base_out else None,
            threshold_pct=cap_threshold, runner=runner, profiler_runner=profiler_runner,
            live_profiler=live_profiler, endpoint_alias=endpoint_alias,
            profiler_root=profiler_root, lease_acquirer=lease_acquirer,
            coordinator_bin=coordinator_bin, profile_override=profile_override,
            capture_root=eff_cap_root, capture_files=capture_files,
            require_capture_files=True, ssh_runner=ssh_runner,
        )
        stage4 = verify_sku_gates(
            sku=resolved_sku, validation_report_json=stage3,
            lean_bounds_json=lean_bounds_json, primitives_json=stage2["primitives_payload"],
            expected_cann_version=stage2["cann_version"], profile_override=profile_override,
            capture_root=eff_cap_root, capture_files=capture_files,
            require_capture_files=True,
            repo_root=repo_root, threshold_pct=cap_threshold,
            run_repo_gates=run_repo_gates, orthogonality_base=orthogonality_base,
            out_receipt=(base_out / "sku_onboarding_receipt.json") if base_out else None,
            diagnosis_out=(base_out / "tail_cluster_diagnosis.json") if base_out else None,
            stage2_lease=stage2.get("lease"),
        )
    all_ok = bool(stage1["ok"] and stage2["ok"] and stage3["ok"] and stage4["ok"])
    out: dict[str, Any] = {
        "stage": "run-all", "sku": resolved_sku, "ok": all_ok,
        "status": stage4["status"] if all_ok else "failed",
        "acquisition_mode": stage4["acquisition_mode"],
        "receipt_sha256": stage4["receipt_sha256"],
        "stages": {
            "probe-soc": stage1,
            "calibrate-l0-and-fabric": stage2,
            "validate-512-400k": stage3,
            "verify-gates": stage4,
        },
    }
    if stage4.get("receipt_unconfirmed") or stage4.get("receipt_warning"):
        out["receipt_unconfirmed"] = bool(stage4.get("receipt_unconfirmed"))
        out["receipt_warning"] = str(stage4.get("receipt_warning") or "")
    return out


__all__ = [
    "REQUIRED_MULTILEVEL_LEVELS",
    "calibrate_l0_and_fabric",
    "parse_npu_smi_topology",
    "parse_platform_config_ini",
    "probe_soc",
    "run_sku_onboarding",
    "validate_512_400k",
    "verify_idle_die_telemetry",
    "verify_sku_gates",
]
