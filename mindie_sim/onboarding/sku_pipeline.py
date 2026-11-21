"""Turnkey 4-stage New-SKU Onboarding Pipeline for Ascend hardware.

[Role] Orchestrate deterministic, repeatable onboarding of a new or existing
  Ascend SKU across four governed stages plus ``run-all``:
  (1) ``probe-soc`` (delegated to ``_soc_probe``),
  (2) ``calibrate-l0-and-fabric`` (real ``mindie-coordinator`` ``HostQueue``
      lease bound to granted physical devices + canonical
      ``tools/bench_l0_primitives.py`` sealer + CANN provenance checks; fails
      closed when ``calibrated_count == 0``),
  (3) ``validate-512-400k`` (disjoint 2D Anchor + Off-Grid Cliff + ABI-guard
      plan scored strictly via ``GateCase -> EvaluationRunner -> EngineRunReceipt``
      with Stage-2 calibrated ``PrimitiveTable`` threaded into replay; fails
      closed on ``plan_only``, missing cells, ABI-rejection omissions, clock
      mismatch, or any pointwise ``APE >= threshold_pct <= 10.0%``), and
  (4) ``verify-gates`` (HMAC-verified report or live manifest replay, mandatory
      SKU/CANN-matched primitive envelope, 1:1 case-joined witnessed Lean 4
      non-inversion, strict repo governance/orthogonality, automatic
      ``diagnose_tail_cluster``, and SHA-256 onboarding receipt).
[Invariants] Zero duplicate primitive sealers; zero fabricated report passes;
  thresholds ``> 10.0%`` rejected; ``calibrated_count == 0`` and ``plan_only``
  ALWAYS set ``ok=False``; ``TimeoutError`` propagates before ``OSError``.
[Timeouts & Locks] Bounded execution budgets; ``HostQueue`` leases always unwind
  via context managers and bind granted device IDs into device subprocesses.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from mindie_sim.backends import ascend_profiles
from mindie_sim.observations.multilevel_gate import EvaluationRunner
from mindie_sim.primitives.schema import PrimitiveTable, parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die

from ._gate_runners import (
    acquire_coordinator_host_queue_lease,
    resolve_repo_root,
    run_arch_governance_gate,
    run_profiler_bench_subprocess,
    run_tail_cluster_diagnosis,
    run_test_orthogonality_gate,
)
from ._lean_and_receipts import (
    check_lean_non_inversion,
    seal_onboarding_receipt,
    sign_stage_report,
    validate_strict_threshold_pct,
)
from ._sealed_scorer import REQUIRED_MULTILEVEL_LEVELS, run_multilevel_verdict
from ._sku_calibration import calibrate_l0_and_fabric, materialize_sku_hardware
from ._soc_probe import (
    parse_npu_smi_topology,
    parse_platform_config_ini,
    probe_soc,
    verify_idle_die_telemetry,
)
from .cliff_planner import (
    evaluate_abi_rejections,
    score_sealed_gate_cases,
    synthesize_sku_512_400k_plan,
)
from .contracts import (
    ANCHOR_CONTEXT_GRID,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DEFAULT_L0_BUDGET_S,
    DISPLAY_ROUND_DIGITS,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    PERCENT_SCALE,
    SHORT_DIGEST_LEN,
    GateCheckOutcome,
    assert_sanitized_payload,
    read_bounded_json,
    write_atomic_json,
)


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
    require_capture_files: bool = False,
) -> dict[str, Any]:
    """Stage 3: Orchestrate and score disjoint 2D Anchor + Cliff 512..400K + L1/L2/L3."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    hw = materialize_sku_hardware(sku, profile_override=profile_override)
    die = hw.dies[0]
    dp = datapath_for_die(die)
    l2_bytes = int(dict(die.buffers).get("l2", 0))
    plan = synthesize_sku_512_400k_plan(
        sku=sku,
        core_count=int(dp.aic_count),
        l2_bytes=max(1, l2_bytes),
        campaign_id=campaign_id,
    )
    plan_doc = plan.to_dict()
    if plan_out is not None:
        write_atomic_json(plan_out, plan_doc)

    parsed_table: PrimitiveTable | None = None
    if primitives_json is not None:
        p_doc = read_bounded_json(primitives_json) if isinstance(primitives_json, (str, Path)) else dict(primitives_json)
        parsed_table = parse_primitive_table(p_doc, verify_envelope=True)

    external_train_roots: list[str] = [plan.train_lineage_root]
    external_train_digests: list[str] = []
    external_train_uris: list[str] = []
    if train_manifest_json is not None:
        t_doc = read_bounded_json(train_manifest_json) if isinstance(train_manifest_json, (str, Path)) else dict(train_manifest_json)
        if t_doc.get("train_lineage_root"):
            external_train_roots.append(str(t_doc["train_lineage_root"]))
        external_train_digests.extend(str(d) for d in (t_doc.get("train_digests") or ()))
        if t_doc.get("primitives_envelope_sha256"):
            external_train_digests.append(str(t_doc["primitives_envelope_sha256"]))
        external_train_uris.extend(str(u) for u in (t_doc.get("train_capture_uris") or ()))
        if t_doc.get("capture_uri"):
            external_train_uris.append(str(t_doc["capture_uri"]))

    obs_rows: list[Mapping[str, Any]] | None = None
    if observations_json is not None:
        raw_obs = read_bounded_json(observations_json)
        if isinstance(raw_obs, Mapping):
            obs_rows = list(raw_obs.get("observations") or raw_obs.get("cases") or ())
            if abi_rejection_results is None and raw_obs.get("abi_rejection_results") is not None:
                abi_rejection_results = list(raw_obs["abi_rejection_results"])
        elif isinstance(raw_obs, list):
            obs_rows = list(raw_obs)
        else:
            raise ModelingError("validate_512_400k_observations_invalid")
    elif profiler_runner is not None:
        obs_rows = list(profiler_runner(plan_doc))
    elif live_profiler:
        lease_ctx = (
            lease_acquirer(sku, DEFAULT_L0_BUDGET_S)
            if lease_acquirer is not None
            else acquire_coordinator_host_queue_lease(
                endpoint_alias, npu_count=1, ttl_seconds=DEFAULT_L0_BUDGET_S, coordinator_bin=coordinator_bin,
            )
        )
        with lease_ctx as lease_info:
            info = dict(lease_info or {})
            if not info.get("lease_id") or info.get("fence_token") is None:
                raise ModelingError("live_profiler_requires_host_queue_lease_id_and_fence_token")
            granted = [int(d) for d in (info.get("devices") or ()) if not isinstance(d, bool)]
            obs_rows = run_profiler_bench_subprocess(
                plan_doc, sku=sku, profiler_root=profiler_root, allowed_phy_ids=granted,
            )

    expected_heldout_pts = list(plan_doc["heldout_points"])
    expected_heldout_ids = [str(pt["point_id"]) for pt in expected_heldout_pts]
    if obs_rows is None:
        plan_only: dict[str, Any] = {
            "stage": "validate-512-400k",
            "sku": sku,
            "ok": False,
            "status": "plan_only",
            "verdict": "insufficient_evidence",
            "error": "validate_512_400k_plan_only_no_observations",
            "plan_sha256": plan.plan_sha256,
            "plan": plan_doc,
            "missing_plan_points": sorted(expected_heldout_ids),
            "missing_anchors": list(ANCHOR_CONTEXT_GRID),
            "missing_cliffs": list(OFF_GRID_CLIFF_HELDOUT_GRID),
            "abi_rejection_check": evaluate_abi_rejections(plan_doc["abi_rejection_points"], None),
            "heldout_summary": None,
            "residuals": [],
        }
        plan_only["report_hmac"] = sign_stage_report("sku_validation", plan_only)
        if report_out is not None:
            write_atomic_json(report_out, plan_only)
        assert_sanitized_payload(plan_only, field_name="validate_512_400k_plan_only")
        return plan_only

    scored = score_sealed_gate_cases(
        obs_rows,
        sku=sku,
        runner=runner,
        primitive_table=parsed_table,
        threshold_pct=cap_threshold,
        external_train_roots=external_train_roots,
        external_train_digests=external_train_digests,
        external_train_uris=external_train_uris,
        expected_points=expected_heldout_pts,
        expected_point_ids=expected_heldout_ids,
        require_disjoint_level_roots=True,
        capture_root=capture_root,
        capture_files=capture_files,
        require_capture_files=require_capture_files,
    )
    abi_check = evaluate_abi_rejections(plan_doc["abi_rejection_points"], abi_rejection_results)
    train_ctxs = {
        int(row["context_len"])
        for row in obs_rows
        if str(row.get("split") or "heldout") == "train"
        and row.get("context_len") is not None
        and str(row.get("status") or "").lower() != "over_budget"
    }
    heldout_ctxs = {
        int(row["context_len"])
        for row in obs_rows
        if str(row.get("split") or "heldout") == "heldout"
        and row.get("context_len") is not None
        and str(row.get("status") or "").lower() != "over_budget"
    }
    missing_anchors = [c for c in ANCHOR_CONTEXT_GRID if c not in train_ctxs]
    missing_cliffs = [c for c in OFF_GRID_CLIFF_HELDOUT_GRID if c not in heldout_ctxs]
    missing_plan_points = list(scored["missing_plan_points"])

    if missing_plan_points or missing_anchors or missing_cliffs or scored["status"] == "insufficient_evidence":
        final_ok, final_status = False, "insufficient_evidence"
    elif scored["failing_cases"] or not scored["ok"] or not abi_check["ok"]:
        final_ok, final_status = False, "fail"
    else:
        final_ok, final_status = True, "pass"

    report: dict[str, Any] = {
        "stage": "validate-512-400k",
        "sku": sku,
        "ok": final_ok,
        "status": final_status,
        "verdict": final_status,
        "threshold_pct": cap_threshold,
        "plan_sha256": plan.plan_sha256,
        "primitives_envelope_sha256": parsed_table.envelope_sha256 if parsed_table is not None else "",
        "anchor_grid": list(ANCHOR_CONTEXT_GRID),
        "off_grid_cliff_grid": list(OFF_GRID_CLIFF_HELDOUT_GRID),
        "missing_plan_points": missing_plan_points,
        "missing_anchors": missing_anchors,
        "missing_cliffs": missing_cliffs,
        "over_budget_points": list(scored["over_budget_points"]),
        "failing_cases": list(scored["failing_cases"]),
        "abi_rejection_check": abi_check,
        "lineage": scored["lineage"],
        "heldout_summary": scored["heldout_summary"],
        "train_summary": scored["train_summary"],
        "residuals": scored["residuals"],
        "plan": plan_doc,
    }
    report["report_hmac"] = sign_stage_report("sku_validation", report)
    if report_out is not None:
        write_atomic_json(report_out, report)
    assert_sanitized_payload(report, field_name="validate_512_400k_report")
    return report


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
    repo_root: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    out_receipt: str | Path | None = None,
    diagnosis_out: str | Path | None = None,
) -> dict[str, Any]:
    """Stage 4: Execute multilevel, mandatory SKU/CANN primitives, joined Lean, repo gates, receipt."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
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
            name="sku_profile_registered",
            passed=sku_registered,
            status="pass" if sku_registered else "fail",
            detail="registered" if sku_registered else f"unknown_sku:{sku}",
        )
    )

    parsed_table: PrimitiveTable | None = None
    if primitives_json is None:
        gates.append(
            GateCheckOutcome(
                name="primitives_envelope",
                passed=False,
                status="fail",
                detail="primitives_json_required_for_sku_certification",
            )
        )
    else:
        prim_doc = (
            read_bounded_json(primitives_json)
            if isinstance(primitives_json, (str, Path))
            else dict(primitives_json)
        )
        try:
            table = parse_primitive_table(prim_doc, verify_envelope=True)
            parsed_table = table
            cal_cnt = sum(1 for cell in table.primitives.values() if cell.status == "calibrated")
            sku_match = table.sku == sku
            cann_match = (not expected_cann_version) or (table.cann_version == expected_cann_version)
            prim_ok = cal_cnt > 0 and sku_match and cann_match
            reasons: list[str] = []
            if not sku_match:
                reasons.append(f"primitives_sku_mismatch:table={table.sku}:target={sku}")
            if not cann_match:
                reasons.append(f"primitives_cann_mismatch:table={table.cann_version}:expected={expected_cann_version}")
            if cal_cnt == 0:
                reasons.append("l0_calibration_zero_calibrated_primitives")
            gates.append(
                GateCheckOutcome(
                    name="primitives_envelope",
                    passed=prim_ok,
                    status="pass" if prim_ok else "fail",
                    detail=(
                        f"calibrated={cal_cnt}/{len(table.primitives)};cann={table.cann_version};envelope={table.envelope_sha256[:SHORT_DIGEST_LEN]}"
                        if prim_ok
                        else "; ".join(reasons)
                    ),
                    metrics=(("calibrated_count", cal_cnt), ("total_primitives", len(table.primitives)), ("sku_match", sku_match), ("cann_match", cann_match)),
                )
            )
            artifacts["primitives_envelope_sha256"] = table.envelope_sha256
        except ModelingError as exc:
            gates.append(GateCheckOutcome(name="primitives_envelope", passed=False, status="fail", detail=str(exc)))

    ml_outcome, diag_residuals = run_multilevel_verdict(
        sku=sku,
        validation_doc=val_doc,
        multilevel_manifest_json=multilevel_manifest_json,
        fits_json=fits_json,
        runner=runner,
        primitive_table=parsed_table,
        threshold_pct=cap_threshold,
    )
    gates.append(ml_outcome)

    failing_cliffs = [
        r for r in diag_residuals
        if float(r.get("absolute_percentage_error", 0.0)) * PERCENT_SCALE >= cap_threshold
    ]
    diagnosis_report: dict[str, Any] | None = None
    if failing_cliffs:
        diagnosis_report = run_tail_cluster_diagnosis(
            diag_residuals,
            repo_root=root,
            threshold_pct=cap_threshold,
            output_path=diagnosis_out,
        )

    expected_timings = {
        str(r["case_id"]): (int(r["predicted_ns"]), int(r["measured_ns"]))
        for r in diag_residuals
        if r.get("case_id") and r.get("predicted_ns") and r.get("measured_ns")
    }
    if isinstance(lean_bounds_json, (str, Path)):
        raw_lean = read_bounded_json(lean_bounds_json)
        lean_rows = raw_lean.get("bounds", raw_lean) if isinstance(raw_lean, Mapping) else raw_lean
    else:
        lean_rows = list(lean_bounds_json or ())
    lean_ok, lean_checked, lean_violations = check_lean_non_inversion(
        lean_rows,
        require_non_empty=True,
        verify_certificates=True,
        expected_case_timings=expected_timings if expected_timings else None,
    )
    if not expected_timings:
        lean_ok = False
        lean_violations = (*lean_violations, "lean_no_scored_heldout_cases_to_join")
    gates.append(
        GateCheckOutcome(
            name="lean_optimality_non_inversion",
            passed=lean_ok,
            status="pass" if lean_ok else "fail",
            detail="0 inversions (witness joined & verified)" if lean_ok else "; ".join(lean_violations),
            metrics=(("cases_checked", len(lean_checked)), ("violations", len(lean_violations))),
        )
    )

    if run_repo_gates:
        gates.append(run_arch_governance_gate(repo_root=root, strict=True))
        gates.append(
            run_test_orthogonality_gate(
                repo_root=root,
                strict=orthogonality_base is None,
                base=orthogonality_base,
            )
        )

    if val_doc and val_doc.get("plan_sha256"):
        artifacts["plan_sha256"] = str(val_doc["plan_sha256"])
    if val_doc and isinstance(val_doc.get("heldout_summary"), Mapping):
        max_ape = val_doc["heldout_summary"].get("max_ape_pct")
        if max_ape is not None:
            artifacts["heldout_max_ape_pct"] = str(round(float(max_ape), DISPLAY_ROUND_DIGITS))

    receipt = seal_onboarding_receipt(
        kind="sku",
        target_id=sku,
        sku=sku,
        gates=gates,
        artifacts=artifacts,
        diagnosis_report=diagnosis_report,
    )
    receipt_dict = receipt.to_dict()
    if out_receipt is not None:
        write_atomic_json(out_receipt, receipt_dict)
    return {
        "stage": "verify-gates",
        "sku": sku,
        "ok": receipt.ok,
        "status": receipt.status,
        "receipt_sha256": receipt.receipt_sha256,
        "lean_checks": [row.to_dict() for row in lean_checked],
        "diagnosis_triggered": diagnosis_report is not None,
        "receipt": receipt_dict,
    }


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
) -> dict[str, Any]:
    """Execute all 4 New-SKU onboarding stages end-to-end with real owners and fail-closed gating."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    base_out = Path(out_dir) if out_dir is not None else None
    stage1 = probe_soc(
        endpoint_alias=endpoint_alias,
        probe_json=probe_json,
        allow_scaffold=allow_scaffold,
        base_template_sku=base_template_sku,
        scaffold_out=(base_out / "scaffold_profile.json") if (base_out and allow_scaffold) else None,
    )
    resolved_sku = str(sku or stage1["sku"])
    if not stage1["ok"]:
        return {
            "stage": "run-all",
            "sku": resolved_sku,
            "ok": False,
            "status": "failed_at_probe_soc",
            "stages": {"probe-soc": stage1},
        }

    profile_override = (
        stage1["scaffold_profile"][resolved_sku]
        if isinstance(stage1.get("scaffold_profile"), Mapping)
        and resolved_sku in stage1["scaffold_profile"]
        else None
    )
    stage2 = calibrate_l0_and_fabric(
        sku=resolved_sku,
        endpoint_alias=endpoint_alias,
        raw_l0_json=raw_l0_json,
        raw_fabric_json=raw_fabric_json,
        out_primitives=(base_out / "primitives.json") if base_out else None,
        manifest_path=(base_out / "primitives_manifest.json") if base_out else None,
        train_manifest_out=(base_out / "train_manifest.json") if base_out else None,
        cann_version=cann_version,
        lease_acquirer=lease_acquirer,
        l0_runner=l0_runner,
        fabric_runner=fabric_runner,
        live_bench=live_bench,
        repo_root=repo_root,
        coordinator_bin=coordinator_bin,
        profile_override=profile_override,
    )
    if not stage2["ok"]:
        return {
            "stage": "run-all",
            "sku": resolved_sku,
            "ok": False,
            "status": "failed_at_calibrate_l0_and_fabric",
            "stages": {"probe-soc": stage1, "calibrate-l0-and-fabric": stage2},
        }

    stage3 = validate_512_400k(
        sku=resolved_sku,
        observations_json=observations_json,
        abi_rejection_results=abi_rejection_results,
        train_manifest_json=stage2["train_manifest"],
        primitives_json=stage2["primitives_payload"],
        plan_out=(base_out / "cliff_plan.json") if base_out else None,
        report_out=(base_out / "validation_512_400k.json") if base_out else None,
        threshold_pct=cap_threshold,
        runner=runner,
        profiler_runner=profiler_runner,
        live_profiler=live_profiler,
        endpoint_alias=endpoint_alias,
        profiler_root=profiler_root,
        lease_acquirer=lease_acquirer,
        coordinator_bin=coordinator_bin,
        profile_override=profile_override,
    )
    stage4 = verify_sku_gates(
        sku=resolved_sku,
        validation_report_json=stage3,
        lean_bounds_json=lean_bounds_json,
        primitives_json=stage2["primitives_payload"],
        expected_cann_version=stage2["cann_version"],
        profile_override=profile_override,
        repo_root=repo_root,
        threshold_pct=cap_threshold,
        run_repo_gates=run_repo_gates,
        orthogonality_base=orthogonality_base,
        out_receipt=(base_out / "sku_onboarding_receipt.json") if base_out else None,
        diagnosis_out=(base_out / "tail_cluster_diagnosis.json") if base_out else None,
    )
    all_ok = bool(stage1["ok"] and stage2["ok"] and stage3["ok"] and stage4["ok"])
    return {
        "stage": "run-all",
        "sku": resolved_sku,
        "ok": all_ok,
        "status": "certified" if all_ok else "failed",
        "receipt_sha256": stage4["receipt_sha256"],
        "stages": {
            "probe-soc": stage1,
            "calibrate-l0-and-fabric": stage2,
            "validate-512-400k": stage3,
            "verify-gates": stage4,
        },
    }


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
