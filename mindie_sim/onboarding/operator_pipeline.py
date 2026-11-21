"""Turnkey 4-stage New-Operator Onboarding Pipeline for Ascend operators/templates.

[Role] Orchestrate deterministic, repeatable onboarding of a new governed
  operator family or template across four stages plus ``run-all``:
  (1) ``scaffold`` / ``validate-contract`` (real ``builtin_family_registry`` /
      ``builtin_template_catalog.lower`` + ``resolve_builder_strict`` ABI
      cross-check),
  (2) ``generate-cliff-plan`` (Lock-3 family-declared / ISA-dispatched
      ``TilingAxes`` + ``validate_abi`` + ``ParallelSpec``-validated ranks),
  (3) ``bench-and-score`` (sealed ``GateCase -> EvaluationRunner ->
      EngineRunReceipt`` scoring enforcing plan SKU, BOTH train & heldout
      splits, operator family binding, exact multi-rank ``HostQueue`` lease
      with ``ttl >= timeout + cleanup_grace``, and ABI rejections), and
  (4) ``certify`` (HMAC-verified reports, zero skipped mandatory contract
      checks, operator/ABI/implementation identity binding across all scored
      receipts, pointwise ``< threshold_pct <= 10.0%``, ``(program, dag)``-
      joined & p99-verified Lean 4 non-inversion, strict repo gates, automatic
      ``diagnose_tail_cluster``, and SHA-256 onboarding receipt distinguishing
      ``silicon_certified`` vs ``offline_reverify``).
[Invariants] Fails closed on fabricated reports, skipped contract checks,
  cross-operator/cross-SKU replay substitution, dropped train anchors,
  thresholds ``> 10.0%``, missing ABI-rejection checks, ``plan_only``,
  ``train==heldout`` leakage, caller ``predicted_*``, ``service_regimes``,
  clock mismatch, or unrelated/inverted/missing-p99 Lean witnesses.
[Timeouts & Locks] Enforces ``lease_ttl >= subprocess_timeout + cleanup_grace``;
  ``TimeoutError`` propagates before ``OSError``; zero device contexts.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``; public receipts carry opaque sanitized lease/endpoint digests.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
import tempfile
from typing import Any, Literal

from mindie_sim.observations.multilevel_gate import EvaluationRunner
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.registry import FamilyRegistry
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.registry import TemplateCatalog

from ._capture_truth import reverify_report_captures, verify_live_silicon_evidence
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
    validate_strict_threshold_pct,
    verify_stage_report,
)
from ._operator_contract import validate_operator_contract
from ._operator_scaffold import scaffold_operator
from ._sealed_scorer import exceeds_physical_tolerance
from .cliff_planner import (
    evaluate_abi_rejections,
    required_ranks_for_plan,
    score_sealed_gate_cases,
    synthesize_operator_cliff_plan,
    verify_cliff_plan_integrity,
)
from .contracts import (
    CONTEXT_MAX_TOKENS,
    CONTEXT_MIN_TOKENS,
    DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DEFAULT_OUTER_TIMEOUT_S,
    DEFAULT_PHYSICAL_ATOL_NS,
    DISPLAY_ROUND_DIGITS,
    GateCheckOutcome,
    OperatorCliffParameters,
    TilingAxes,
    assert_sanitized_payload,
    collect_receipt_warnings,
    read_bounded_json,
    sanitize_lease_for_receipt,
    write_atomic_json,
)

_ENGINE_RECEIPT_PREFIX = "capture://engine-receipt/"


def generate_operator_cliff_plan(
    *,
    operator_id: str,
    family: str,
    sku: str,
    tile_dim: int,
    core_count: int,
    l2_bytes: int,
    bytes_per_token: int = 0,
    topk_threshold: int | None = None,
    context_min: int = CONTEXT_MIN_TOKENS,
    context_max: int = CONTEXT_MAX_TOKENS,
    tiling_axes: TilingAxes | None = None,
    campaign_id: str = "onboard-op",
    plan_out: str | Path | None = None,
) -> dict[str, Any]:
    """Stage 2: Synthesize and optionally write a disjoint 2D cliff + ABI-rejection plan."""
    params = OperatorCliffParameters(
        operator_id=operator_id,
        family=family,
        tile_dim=int(tile_dim),
        core_count=int(core_count),
        l2_bytes=int(l2_bytes),
        bytes_per_token=int(bytes_per_token),
        topk_threshold=int(topk_threshold) if topk_threshold is not None else None,
        context_min=int(context_min),
        context_max=int(context_max),
        tiling_axes=tiling_axes,
    )
    plan = synthesize_operator_cliff_plan(params, sku=sku, campaign_id=campaign_id)
    payload = plan.to_dict()
    assert_sanitized_payload(payload, field_name="operator_cliff_plan")
    if plan_out is not None:
        write_atomic_json(plan_out, payload)
    return {
        "stage": "generate-cliff-plan",
        "ok": True,
        "status": "ok",
        "plan_sha256": plan.plan_sha256,
        "required_ranks": required_ranks_for_plan(payload),
        "train_point_count": len(plan.train_points),
        "heldout_point_count": len(plan.heldout_points),
        "abi_rejection_point_count": len(plan.abi_rejection_points),
        "plan": payload,
    }


def _collect_operator_observations(
    *,
    plan_doc: Mapping[str, Any],
    sku: str,
    req_npus: int,
    captured_json: str | Path | Sequence[Mapping[str, Any]] | None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None,
    bench_dispatcher: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]] | None,
    live_profiler: bool,
    endpoint_alias: str,
    profiler_root: str | Path | None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None,
    coordinator_bin: str,
    timeout_s: int,
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
    if captured_json is not None:
        if isinstance(captured_json, (str, Path)):
            raw_cap = read_bounded_json(captured_json)
            rows: list[Mapping[str, Any]] | None = list(
                raw_cap.get("observations", raw_cap) if isinstance(raw_cap, Mapping) else raw_cap
            )
            if abi_rejection_results is None and isinstance(raw_cap, Mapping) and raw_cap.get("abi_rejection_results") is not None:
                abi_rejection_results = list(raw_cap["abi_rejection_results"])
        else:
            rows = list(captured_json)
        return rows, abi_rejection_results, lease_public, acquisition_mode

    if bench_dispatcher is not None:
        if lease_acquirer is None and not live_profiler:
            return list(bench_dispatcher(plan_doc, int(timeout_s))), abi_rejection_results, lease_public, acquisition_mode
        lease_ttl_s = int(timeout_s) + DEFAULT_CLEANUP_GRACE_S
        lease_ctx = (
            lease_acquirer(sku, lease_ttl_s)
            if lease_acquirer is not None
            else acquire_coordinator_host_queue_lease(
                endpoint_alias, npu_count=req_npus, ttl_seconds=lease_ttl_s,
                min_subprocess_timeout_s=int(timeout_s), cleanup_grace_s=DEFAULT_CLEANUP_GRACE_S,
                coordinator_bin=coordinator_bin,
            )
        )
        raw_lease: Mapping[str, Any] = {}
        with lease_ctx as lease_info:
            raw_lease = lease_info if isinstance(lease_info, Mapping) else {}
            validate_host_queue_grant(raw_lease, required_devices=req_npus, min_ttl_seconds=lease_ttl_s)
            rows = list(bench_dispatcher(plan_doc, int(timeout_s)))
        return rows, abi_rejection_results, sanitize_lease_for_receipt(raw_lease), acquisition_mode

    if live_profiler:
        lease_ttl_s = int(timeout_s) + DEFAULT_CLEANUP_GRACE_S
        lease_ctx = (
            lease_acquirer(sku, lease_ttl_s)
            if lease_acquirer is not None
            else acquire_coordinator_host_queue_lease(
                endpoint_alias, npu_count=req_npus, ttl_seconds=lease_ttl_s,
                min_subprocess_timeout_s=int(timeout_s), cleanup_grace_s=DEFAULT_CLEANUP_GRACE_S,
                coordinator_bin=coordinator_bin,
            )
        )
        raw_lease = {}
        with lease_ctx as lease_info:
            raw_lease = lease_info if isinstance(lease_info, Mapping) else {}
            _, fence, granted, _ = validate_host_queue_grant(
                raw_lease, required_devices=req_npus, min_ttl_seconds=lease_ttl_s
            )
            rows, live_abi = run_profiler_bench_subprocess(
                plan_doc, sku=sku, profiler_root=profiler_root, endpoint_alias=endpoint_alias,
                allowed_phy_ids=granted, fence_token=fence, timeout_s=int(timeout_s),
                capture_out_dir=eff_capture_root, ssh_runner=ssh_runner,
            )
            if abi_rejection_results is None and live_abi:
                abi_rejection_results = live_abi
        lease_public = sanitize_lease_for_receipt(raw_lease)
        if lease_public.get("lease_mode") == "coordinator_host_queue":
            acquisition_mode = "live_silicon"
        return rows, abi_rejection_results, lease_public, acquisition_mode

    return None, abi_rejection_results, lease_public, acquisition_mode


def bench_and_score_operator(
    *,
    cliff_plan: str | Path | Mapping[str, Any],
    sku: str,
    captured_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    bench_dispatcher: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]] | None = None,
    live_profiler: bool = False,
    endpoint_alias: str = "",
    profiler_root: str | Path | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
    coordinator_bin: str = "",
    runner: EvaluationRunner | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    out_score: str | Path | None = None,
    timeout_s: int = DEFAULT_OUTER_TIMEOUT_S,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = True,
    ssh_runner: Callable[[str, str, int], Any] | None = None,
) -> dict[str, Any]:
    """Stage 3: Dispatch bench plan (under exact-rank HostQueue lease if live) and score."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    plan_raw = read_bounded_json(cliff_plan) if isinstance(cliff_plan, (str, Path)) else dict(cliff_plan)
    plan_doc = dict(plan_raw["plan"]) if isinstance(plan_raw, Mapping) and "plan" in plan_raw else dict(plan_raw)
    verified_plan_sha = verify_cliff_plan_integrity(plan_doc)
    plan_sku = str(plan_doc.get("sku") or "").strip()
    if plan_sku and plan_sku != sku:
        raise ModelingError(f"cliff_plan_sku_mismatch:plan={plan_sku}:requested={sku}")
    expected_heldout_pts = list(plan_doc.get("heldout_points", ()))
    expected_train_pts = list(plan_doc.get("train_points", ()))
    expected_heldout_ids = [str(pt["point_id"]) for pt in expected_heldout_pts]
    expected_train_ids = [str(pt["point_id"]) for pt in expected_train_pts]
    plan_family = str(plan_doc.get("family") or "").strip()
    req_npus = required_ranks_for_plan(plan_doc)

    eff_capture_root: str | Path | None = capture_root
    temp_cap_ctx: AbstractContextManager[str | None] = nullcontext(None)
    if live_profiler and captured_json is None and bench_dispatcher is None and eff_capture_root is None:
        if out_score is not None:
            eff_capture_root = Path(out_score).parent / "captures"
        else:
            temp_cap_ctx = tempfile.TemporaryDirectory(prefix="msim-op-cap-")

    with temp_cap_ctx as tmp_cap_dir:
        if tmp_cap_dir is not None:
            eff_capture_root = Path(tmp_cap_dir)
        rows, abi_rejection_results, lease_public, acquisition_mode = _collect_operator_observations(
            plan_doc=plan_doc, sku=sku, req_npus=req_npus,
            captured_json=captured_json, abi_rejection_results=abi_rejection_results,
            bench_dispatcher=bench_dispatcher, live_profiler=live_profiler,
            endpoint_alias=endpoint_alias, profiler_root=profiler_root,
            lease_acquirer=lease_acquirer, coordinator_bin=coordinator_bin,
            timeout_s=int(timeout_s), eff_capture_root=eff_capture_root, ssh_runner=ssh_runner,
        )
        if rows is None:
            plan_only: dict[str, Any] = {
                "stage": "bench-and-score", "ok": False, "status": "plan_only",
                "verdict": "insufficient_evidence",
                "error": "bench_and_score_plan_only_no_observations",
                "sku": sku, "target_id": str(plan_doc.get("target_id") or ""),
                "family": plan_family, "plan_sha256": verified_plan_sha,
                "required_ranks": req_npus, "acquisition_mode": "offline_reverify",
                "capture_files_verified": False, "lease": lease_public,
                "missing_heldout_points": sorted(expected_heldout_ids),
                "missing_train_points": sorted(expected_train_ids),
                "abi_rejection_check": evaluate_abi_rejections(plan_doc.get("abi_rejection_points") or (), None),
                "heldout_summary": None, "residuals": [],
            }
            plan_only["report_hmac"] = sign_stage_report("operator_score", plan_only)
            if out_score is not None:
                write_atomic_json(out_score, plan_only)
            assert_sanitized_payload(plan_only, field_name="bench_and_score_plan_only")
            return plan_only

        external_train = [str(plan_doc["train_lineage_root"])] if plan_doc.get("train_lineage_root") else []
        scored = score_sealed_gate_cases(
            rows, sku=sku, runner=runner, threshold_pct=cap_threshold,
            external_train_roots=external_train, expected_points=expected_heldout_pts,
            expected_train_points=expected_train_pts, expected_point_ids=expected_heldout_ids,
            expected_family=plan_family, require_disjoint_level_roots=False,
            capture_root=eff_capture_root, capture_files=capture_files,
            require_capture_files=require_capture_files,
        )
    if not scored.get("capture_files_verified"):
        acquisition_mode = "offline_reverify"
    abi_check = evaluate_abi_rejections(list(plan_doc.get("abi_rejection_points") or ()), abi_rejection_results)
    missing_heldout = list(scored["missing_plan_points"])
    missing_train = list(scored["missing_train_points"])
    if missing_heldout or missing_train or scored["status"] == "insufficient_evidence":
        final_ok, final_status = False, "insufficient_evidence"
    elif scored["failing_cases"] or not scored["ok"] or not abi_check["ok"]:
        final_ok, final_status = False, "fail"
    else:
        final_ok, final_status = True, "pass"

    report: dict[str, Any] = {
        "stage": "bench-and-score", "ok": final_ok, "status": final_status, "verdict": final_status,
        "sku": sku, "threshold_pct": cap_threshold, "atol_ns": int(scored.get("atol_ns", DEFAULT_PHYSICAL_ATOL_NS)),
        "target_id": str(plan_doc.get("target_id") or ""), "family": plan_family,
        "plan_sha256": verified_plan_sha, "required_ranks": req_npus,
        "acquisition_mode": acquisition_mode,
        "capture_files_verified": bool(scored.get("capture_files_verified", False)),
        "lease": lease_public,
        "missing_heldout_points": missing_heldout, "missing_train_points": missing_train,
        "over_budget_points": list(scored["over_budget_points"]),
        "failing_cases": list(scored["failing_cases"]),
        "abi_rejection_check": abi_check, "lineage": scored["lineage"],
        "heldout_summary": scored["heldout_summary"], "train_summary": scored["train_summary"],
        "residuals": scored["residuals"],
    }
    rec_unconf, rec_warn = collect_receipt_warnings(lease_public)
    if rec_unconf or rec_warn:
        report["receipt_unconfirmed"] = rec_unconf
        report["receipt_warning"] = rec_warn
    report["report_hmac"] = sign_stage_report("operator_score", report)
    assert_sanitized_payload(report, field_name="bench_and_score_report")
    if out_score is not None:
        write_atomic_json(out_score, report)
    return report


def _check_contract_release_eligibility(c_doc: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    sim_disp = c_doc.get("sim_owner_dispatch") if isinstance(c_doc.get("sim_owner_dispatch"), Mapping) else {}
    if (
        sim_disp.get("checked") is not True
        or sim_disp.get("matched") is not True
        or sim_disp.get("lowered") is not True
        or sim_disp.get("lean_status") != "verified"
    ):
        errs.append("contract_sim_owner_skipped_or_unlowered")
    prof = c_doc.get("profiler_builder") if isinstance(c_doc.get("profiler_builder"), Mapping) else {}
    if prof.get("checked") is not True or prof.get("verified") is not True:
        errs.append("contract_profiler_builder_skipped_or_unverified")
    tax = c_doc.get("taxonomy") if isinstance(c_doc.get("taxonomy"), Mapping) else {}
    if tax.get("registered") is not True:
        errs.append("contract_taxonomy_not_registered")
    if c_doc.get("tiling_axes_declared") is not True:
        errs.append("contract_tiling_axes_not_declared")
    if c_doc.get("semantic_work_declared") is not True or c_doc.get("tensor_obligations_declared") is not True:
        errs.append("contract_semantic_or_obligations_missing")
    return errs


def _evaluate_operator_contract_gate(
    operator_id: str, c_doc: Mapping[str, Any], s_doc: Mapping[str, Any]
) -> tuple[GateCheckOutcome, str, Mapping[str, Any], Mapping[str, Any], set[str]]:
    c_hmac_ok = verify_stage_report("operator_contract", c_doc)
    c_family = str(c_doc.get("family") or "").strip()
    s_family = str(s_doc.get("family") or "").strip()
    s_target_id = str(s_doc.get("target_id") or "").strip()
    op_prefix, op_variant = (
        (operator_id.split(":", 1)[0].strip(), operator_id.split(":", 1)[1].strip())
        if ":" in operator_id
        else (operator_id.strip(), "")
    )
    sim_disp = c_doc.get("sim_owner_dispatch") if isinstance(c_doc.get("sim_owner_dispatch"), Mapping) else {}
    prof_disp = c_doc.get("profiler_builder") if isinstance(c_doc.get("profiler_builder"), Mapping) else {}
    declared_abis = {str(v).strip() for v in (c_doc.get("upstream_abis") or {}).values() if str(v or "").strip()}
    identity_violations = _check_contract_release_eligibility(c_doc)
    if s_target_id != operator_id:
        identity_violations.append(f"score_report_operator_id_mismatch:expected={operator_id}:got={s_target_id or 'missing'}")
    if c_family and s_family and c_family != s_family:
        identity_violations.append(f"contract_score_family_mismatch:contract={c_family}:score={s_family}")
    if c_family and op_prefix and c_family != op_prefix:
        identity_violations.append(f"contract_operator_id_mismatch:contract_family={c_family}:operator_id={operator_id}")
    if sim_disp.get("owner") == "builtin_template_catalog":
        tid = str(sim_disp.get("template_id") or "").strip()
        if tid and operator_id != tid and op_variant != tid:
            identity_violations.append(f"contract_template_operator_id_mismatch:template={tid}:operator_id={operator_id}")
    elif op_variant:
        allowed_variants = {
            str(v).strip()
            for v in (
                *(sim_disp.get("owned_recipe_ids") or ()),
                sim_disp.get("resolved_implementation"),
                prof_disp.get("resolved_abi"),
                *declared_abis,
            )
            if str(v or "").strip()
        }
        if allowed_variants and operator_id not in allowed_variants and op_variant not in allowed_variants:
            identity_violations.append(f"contract_operator_variant_unowned:operator_id={operator_id}:owned={sorted(allowed_variants)}")
    contract_ok = c_hmac_ok and bool(c_doc.get("ok")) and not identity_violations
    c_violations = list(c_doc.get("violations") or ())
    if not c_hmac_ok:
        c_violations.append("contract_report_hmac_unverified_or_fabricated")
    c_violations.extend(identity_violations)
    outcome = GateCheckOutcome(
        name="operator_contract", passed=contract_ok, status="pass" if contract_ok else "fail",
        detail="0 violations" if contract_ok else "; ".join(str(v) for v in c_violations),
        metrics=(
            ("source_refs_checked", int(c_doc.get("source_refs_checked", 0))),
            ("hmac_verified", c_hmac_ok), ("identity_bound", not identity_violations),
            ("violations", len(c_violations)),
        ),
    )
    return outcome, c_family, sim_disp, prof_disp, declared_abis


def _evaluate_operator_score_gate(
    *,
    operator_id: str,
    sku: str,
    s_doc: Mapping[str, Any],
    c_family: str,
    sim_disp: Mapping[str, Any],
    prof_disp: Mapping[str, Any],
    declared_abis: set[str],
    cap_threshold: float,
    capture_root: str | Path | None,
    capture_files: Mapping[str, str | Path] | None,
) -> tuple[GateCheckOutcome, bool, bool, list[Mapping[str, Any]], list[Mapping[str, Any]], Any]:
    s_hmac_ok = verify_stage_report("operator_score", s_doc)
    sku_match = str(s_doc.get("sku") or "") == sku
    target_match = str(s_doc.get("target_id") or "").strip() == operator_id
    all_score_rows = [r for r in s_doc.get("residuals", ()) if isinstance(r, Mapping)]
    heldout_rows = [r for r in all_score_rows if r.get("split", "heldout") == "heldout"]
    captures_ok, capture_err = True, ""
    if heldout_rows:
        if s_doc.get("capture_files_verified") is not True:
            captures_ok, capture_err = False, "capture_files_unverified_for_release"
        else:
            reverify_report_captures(
                all_score_rows, capture_root=capture_root, capture_files=capture_files, require_capture_files=True
            )

    allowed_impls = {
        str(v).strip()
        for v in (sim_disp.get("resolved_implementation"), prof_disp.get("resolved_abi"), *declared_abis)
        if str(v or "").strip()
    }
    prof_abi = str(prof_disp.get("resolved_abi") or "").strip()
    receipt_identity_errs: list[str] = []
    for r in heldout_rows:
        cid = str(r.get("case_id") or "")
        exec_fams = r.get("executed_op_families") or []
        if exec_fams and c_family and c_family not in exec_fams:
            receipt_identity_errs.append(f"receipt_family_mismatch:{cid}:executed={exec_fams}:expected={c_family}")
        row_fam = str(r.get("target_family") or "").strip()
        if not row_fam or (c_family and row_fam != c_family):
            receipt_identity_errs.append(f"receipt_target_family_mismatch:{cid}:got={row_fam or 'missing'}:expected={c_family}")
        row_op_id = str(r.get("operator_id") or "").strip()
        if row_op_id and row_op_id != operator_id:
            receipt_identity_errs.append(f"receipt_operator_id_mismatch:{cid}:got={row_op_id}:expected={operator_id}")
        row_impl = str(r.get("implementation") or "").strip()
        if not row_impl or (allowed_impls and row_impl not in allowed_impls):
            receipt_identity_errs.append(f"receipt_implementation_mismatch:{cid}:got={row_impl or 'missing'}")
        row_abi = str(r.get("upstream_abi") or "").strip()
        if row_abi and ((declared_abis and row_abi not in declared_abis) or (prof_abi and row_abi != prof_abi)):
            receipt_identity_errs.append(f"receipt_abi_mismatch:{cid}:got={row_abi}")

    receipts_ok = bool(heldout_rows) and captures_ok and not receipt_identity_errs and all(
        str(r.get("engine_receipt") or "").startswith(_ENGINE_RECEIPT_PREFIX)
        and int(r.get("predicted_ns") or 0) > 0
        and int(r.get("measured_ns") or 0) > 0
        for r in heldout_rows
    )
    eff_atol_ns = max(0, int(s_doc.get("atol_ns", DEFAULT_PHYSICAL_ATOL_NS)))
    failing_rows = [
        r for r in heldout_rows
        if exceeds_physical_tolerance(
            int(r.get("predicted_ns") or 0), int(r.get("measured_ns") or 1),
            threshold_pct=cap_threshold, atol_ns=eff_atol_ns,
        )
    ]
    missing_pts = list(s_doc.get("missing_heldout_points") or ())
    missing_train = list(s_doc.get("missing_train_points") or ())
    max_ape = (s_doc.get("heldout_summary") or {}).get("max_ape_pct")
    abi_ok = bool((s_doc.get("abi_rejection_check") or {}).get("ok", False))
    cliff_ok = (
        s_hmac_ok and sku_match and target_match and bool(s_doc.get("ok"))
        and receipts_ok and not failing_rows and not missing_pts and not missing_train
        and abi_ok and max_ape is not None
    )
    if cliff_ok and max_ape is not None:
        cliff_detail = f"max_ape_pct={round(float(max_ape), DISPLAY_ROUND_DIGITS)};abi_ok={abi_ok}"
    elif not s_hmac_ok:
        cliff_detail = "score_report_hmac_unverified_or_fabricated"
    elif not sku_match:
        cliff_detail = "score_report_sku_mismatch"
    elif not target_match:
        cliff_detail = "score_report_operator_id_mismatch"
    elif capture_err:
        cliff_detail = capture_err
    elif receipt_identity_errs:
        cliff_detail = "; ".join(receipt_identity_errs[:4])
    elif not receipts_ok:
        cliff_detail = "unverified_engine_receipt"
    elif not s_doc.get("ok"):
        cliff_detail = "score_report_ok_false"
    else:
        cliff_detail = f"failing={len(failing_rows)};missing_heldout={len(missing_pts)};missing_train={len(missing_train)};abi_ok={abi_ok}"
    outcome = GateCheckOutcome(
        name="heldout_cliff_accuracy", passed=cliff_ok,
        status="pass" if cliff_ok else ("insufficient_evidence" if (s_hmac_ok and (not heldout_rows or missing_pts or missing_train)) else "fail"),
        detail=cliff_detail,
        metrics=(
            ("heldout_count", len(heldout_rows)), ("hmac_verified", s_hmac_ok),
            ("failing_cliffs", len(failing_rows)), ("missing_heldout_points", len(missing_pts)),
            ("missing_train_points", len(missing_train)), ("abi_rejection_ok", abi_ok),
            ("max_ape_pct", max_ape), ("threshold_pct", cap_threshold), ("atol_ns", eff_atol_ns),
        ),
    )
    return outcome, s_hmac_ok, cliff_ok, heldout_rows, failing_rows, max_ape


def certify_operator(
    *,
    operator_id: str,
    sku: str,
    contract_report: str | Path | Mapping[str, Any],
    score_report: str | Path | Mapping[str, Any],
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    out_receipt: str | Path | None = None,
    diagnosis_out: str | Path | None = None,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = True,
    acquisition_mode: Literal["live_silicon", "offline_reverify"] | None = None,
) -> dict[str, Any]:
    """Stage 4: Enforce complete contract, bound receipts, verified captures, pointwise <10%, joined Lean."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    if not require_capture_files:
        raise ModelingError("release_certification_requires_capture_files")
    root = resolve_repo_root(repo_root)
    c_doc = read_bounded_json(contract_report) if isinstance(contract_report, (str, Path)) else dict(contract_report)
    s_doc = read_bounded_json(score_report) if isinstance(score_report, (str, Path)) else dict(score_report)

    contract_gate, c_family, sim_disp, prof_disp, declared_abis = _evaluate_operator_contract_gate(
        operator_id, c_doc, s_doc
    )
    score_gate, s_hmac_ok, cliff_ok, heldout_rows, failing_rows, max_ape = _evaluate_operator_score_gate(
        operator_id=operator_id, sku=sku, s_doc=s_doc, c_family=c_family,
        sim_disp=sim_disp, prof_disp=prof_disp, declared_abis=declared_abis,
        cap_threshold=cap_threshold, capture_root=capture_root, capture_files=capture_files,
    )
    gates: list[GateCheckOutcome] = [contract_gate, score_gate]

    diagnosis_report: dict[str, Any] | None = None
    if failing_rows:
        diagnosis_report = run_tail_cluster_diagnosis(
            heldout_rows, repo_root=root, threshold_pct=cap_threshold, output_path=diagnosis_out,
        )

    expected_timings = {
        str(r["case_id"]): (
            int(r["predicted_ns"]), int(r["measured_ns"]),
            int(r["measured_p99_ns"]) if r.get("measured_p99_ns") is not None else None,
            str(r.get("program_dag_sha256") or ""),
        )
        for r in heldout_rows
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
    gates.append(
        GateCheckOutcome(
            name="lean_optimality_non_inversion", passed=lean_ok, status="pass" if lean_ok else "fail",
            detail="0 inversions (witness joined & verified)" if lean_ok else "; ".join(lean_violations),
            metrics=(("cases_checked", len(lean_checked)), ("violations", len(lean_violations))),
        )
    )

    if run_repo_gates:
        gates.append(run_arch_governance_gate(repo_root=root, strict=True))
        gates.append(run_test_orthogonality_gate(repo_root=root, strict=orthogonality_base is None, base=orthogonality_base))

    artifacts: dict[str, str] = {}
    if s_doc.get("plan_sha256"):
        artifacts["plan_sha256"] = str(s_doc["plan_sha256"])
    if max_ape is not None:
        artifacts["heldout_max_ape_pct"] = str(round(float(max_ape), DISPLAY_ROUND_DIGITS))
    if c_doc.get("target"):
        artifacts["operator_target"] = str(c_doc["target"])

    live_verified = s_hmac_ok and cliff_ok and verify_live_silicon_evidence(s_doc)
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
    rec_unconf, rec_warn = collect_receipt_warnings(s_doc)
    receipt = seal_onboarding_receipt(
        kind="operator", target_id=operator_id, sku=sku,
        gates=gates, artifacts=artifacts, diagnosis_report=diagnosis_report,
        acquisition_mode=eff_mode, receipt_unconfirmed=rec_unconf, receipt_warning=rec_warn,
    )
    receipt_dict = receipt.to_dict()
    if out_receipt is not None:
        write_atomic_json(out_receipt, receipt_dict)
    out: dict[str, Any] = {
        "stage": "certify", "operator_id": operator_id, "sku": sku,
        "ok": receipt.ok, "status": receipt.status, "acquisition_mode": receipt.acquisition_mode,
        "receipt_sha256": receipt.receipt_sha256,
        "lean_checks": [row.to_dict() for row in lean_checked],
        "diagnosis_triggered": diagnosis_report is not None,
        "receipt": receipt_dict,
    }
    if rec_unconf or rec_warn:
        out["receipt_unconfirmed"] = rec_unconf
        out["receipt_warning"] = rec_warn
    return out


def run_operator_onboarding(
    *,
    target_path: str | Path,
    operator_id: str,
    family: str,
    sku: str,
    tile_dim: int,
    core_count: int,
    l2_bytes: int,
    bytes_per_token: int = 0,
    topk_threshold: int | None = None,
    tiling_axes: TilingAxes | None = None,
    captured_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
    refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None,
    profiler_root: str | Path | None = None,
    profiler_family: str = "",
    profiler_recipe: str = "",
    profiler_impl: str = "",
    family_registry: FamilyRegistry | None = None,
    template_catalog: TemplateCatalog | None = None,
    template_id: str = "",
    sample_op: OpSpec | None = None,
    planning_context: OperatorPlanningContext | None = None,
    require_sim_owner: bool = True,
    require_taxonomy: bool = True,
    require_profiler_builder: bool = True,
    require_tiling_axes: bool = True,
    require_mounted_refs: bool = False,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    out_dir: str | Path | None = None,
    bench_dispatcher: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]] | None = None,
    live_profiler: bool = False,
    endpoint_alias: str = "",
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
    coordinator_bin: str = "",
    runner: EvaluationRunner | None = None,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = True,
    ssh_runner: Callable[[str, str, int], Any] | None = None,
) -> dict[str, Any]:
    """Execute all 4 New-Operator onboarding stages end-to-end."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    if not require_capture_files:
        raise ModelingError("release_certification_requires_capture_files")
    base_out = Path(out_dir) if out_dir is not None else None
    stage1 = validate_operator_contract(
        target_path=target_path, repo_root=repo_root, refs_root=refs_root, repo_map=repo_map,
        profiler_root=profiler_root, profiler_family=profiler_family,
        profiler_recipe=profiler_recipe, profiler_impl=profiler_impl,
        family_registry=family_registry, template_catalog=template_catalog,
        template_id=template_id, sample_op=sample_op, planning_context=planning_context,
        require_sim_owner=require_sim_owner, require_taxonomy=require_taxonomy,
        require_profiler_builder=require_profiler_builder,
        require_tiling_axes=require_tiling_axes, require_mounted_refs=require_mounted_refs,
    )
    stage2 = generate_operator_cliff_plan(
        operator_id=operator_id, family=family, sku=sku,
        tile_dim=tile_dim, core_count=core_count, l2_bytes=l2_bytes,
        bytes_per_token=bytes_per_token, topk_threshold=topk_threshold, tiling_axes=tiling_axes,
        plan_out=(base_out / "operator_cliff_plan.json") if base_out else None,
    )
    eff_cap_root: str | Path | None = capture_root if capture_root is not None else ((base_out / "captures") if (live_profiler and base_out) else None)
    cap_ctx: AbstractContextManager[str | None] = (
        tempfile.TemporaryDirectory(prefix="msim-op-runall-cap-")
        if (live_profiler and eff_cap_root is None and captured_json is None and bench_dispatcher is None)
        else nullcontext(None)
    )
    with cap_ctx as tmp_cap_dir:
        if tmp_cap_dir is not None:
            eff_cap_root = Path(tmp_cap_dir)
        stage3 = bench_and_score_operator(
            cliff_plan=stage2["plan"], sku=sku, captured_json=captured_json,
            abi_rejection_results=abi_rejection_results, bench_dispatcher=bench_dispatcher,
            live_profiler=live_profiler, endpoint_alias=endpoint_alias, profiler_root=profiler_root,
            lease_acquirer=lease_acquirer, coordinator_bin=coordinator_bin, runner=runner,
            threshold_pct=cap_threshold,
            out_score=(base_out / "operator_score_report.json") if base_out else None,
            capture_root=eff_cap_root, capture_files=capture_files,
            require_capture_files=True, ssh_runner=ssh_runner,
        )
        stage4 = certify_operator(
            operator_id=operator_id, sku=sku, contract_report=stage1, score_report=stage3,
            lean_bounds_json=lean_bounds_json, repo_root=repo_root, threshold_pct=cap_threshold,
            run_repo_gates=run_repo_gates, orthogonality_base=orthogonality_base,
            capture_root=eff_cap_root, capture_files=capture_files,
            require_capture_files=True,
            out_receipt=(base_out / "operator_onboarding_receipt.json") if base_out else None,
            diagnosis_out=(base_out / "operator_tail_diagnosis.json") if base_out else None,
        )
    all_ok = bool(stage1["ok"] and stage2["ok"] and stage3["ok"] and stage4["ok"])
    out: dict[str, Any] = {
        "stage": "run-all", "operator_id": operator_id, "sku": sku,
        "ok": all_ok,
        "status": stage4["status"] if all_ok else "failed",
        "acquisition_mode": stage4["acquisition_mode"],
        "receipt_sha256": stage4["receipt_sha256"],
        "stages": {
            "validate-contract": stage1,
            "generate-cliff-plan": stage2,
            "bench-and-score": stage3,
            "certify": stage4,
        },
    }
    if stage4.get("receipt_unconfirmed") or stage4.get("receipt_warning"):
        out["receipt_unconfirmed"] = bool(stage4.get("receipt_unconfirmed"))
        out["receipt_warning"] = str(stage4.get("receipt_warning") or "")
    return out


__all__ = [
    "bench_and_score_operator",
    "certify_operator",
    "generate_operator_cliff_plan",
    "run_operator_onboarding",
    "scaffold_operator",
    "validate_operator_contract",
]
