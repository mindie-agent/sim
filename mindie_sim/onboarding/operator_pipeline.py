"""Turnkey 4-stage New-Operator Onboarding Pipeline for Ascend operators/templates.

[Role] Orchestrate deterministic, repeatable onboarding of a new governed
  operator family or template across four stages plus ``run-all``:
  (1) ``scaffold`` / ``validate-contract`` (real ``builtin_family_registry`` /
      ``builtin_template_catalog`` + ``resolve_builder_strict`` ABI cross-check),
  (2) ``generate-cliff-plan`` (Lock-3 ``TilingAxes`` + ``validate_abi``),
  (3) ``bench-and-score`` (sealed ``GateCase -> EvaluationRunner ->
      EngineRunReceipt`` scoring with ``HostQueue`` lease on live path and
      mandatory ABI-rejection enforcement), and
  (4) ``certify`` (HMAC-verified contract/score reports, pointwise
      ``< threshold_pct <= 10.0%``, 1:1 case-joined witnessed Lean 4
      non-inversion, strict repo gates, automatic ``diagnose_tail_cluster``,
      and SHA-256 onboarding receipt).
[Invariants] Fails closed on fabricated reports, ``ok=False`` score reports,
  thresholds ``> 10.0%``, missing ABI-rejection checks, missing owners,
  ``plan_only``, ``train==heldout`` leakage, caller ``predicted_*``,
  ``service_regimes`` crutches, clock mismatch, or unrelated/inverted Lean
  witnesses.
[Timeouts & Locks] Bounded execution budgets and subprocess gates;
  ``TimeoutError`` propagates before ``OSError``. Zero device contexts.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from mindie_sim.observations.multilevel_gate import EvaluationRunner
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.registry import FamilyRegistry
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.registry import TemplateCatalog

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
    verify_stage_report,
)
from ._operator_contract import validate_operator_contract
from ._operator_scaffold import scaffold_operator
from .cliff_planner import (
    evaluate_abi_rejections,
    score_sealed_gate_cases,
    synthesize_operator_cliff_plan,
    verify_cliff_plan_integrity,
)
from .contracts import (
    CONTEXT_MAX_TOKENS,
    CONTEXT_MIN_TOKENS,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DEFAULT_OUTER_TIMEOUT_S,
    DISPLAY_ROUND_DIGITS,
    PERCENT_SCALE,
    GateCheckOutcome,
    OperatorCliffParameters,
    TilingAxes,
    assert_sanitized_payload,
    read_bounded_json,
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
        "train_point_count": len(plan.train_points),
        "heldout_point_count": len(plan.heldout_points),
        "abi_rejection_point_count": len(plan.abi_rejection_points),
        "plan": payload,
    }


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
) -> dict[str, Any]:
    """Stage 3: Dispatch bench plan (under HostQueue lease if live) and score via sealed runner."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    plan_raw = read_bounded_json(cliff_plan) if isinstance(cliff_plan, (str, Path)) else dict(cliff_plan)
    plan_doc = dict(plan_raw["plan"]) if isinstance(plan_raw, Mapping) and "plan" in plan_raw else dict(plan_raw)
    verified_plan_sha = verify_cliff_plan_integrity(plan_doc)
    expected_heldout_pts = list(plan_doc.get("heldout_points", ()))
    expected_train_pts = list(plan_doc.get("train_points", ()))
    expected_heldout_ids = [str(pt["point_id"]) for pt in expected_heldout_pts]

    rows: list[Mapping[str, Any]] | None = None
    if captured_json is not None:
        if isinstance(captured_json, (str, Path)):
            raw_cap = read_bounded_json(captured_json)
            rows = list(raw_cap.get("observations", raw_cap) if isinstance(raw_cap, Mapping) else raw_cap)
            if abi_rejection_results is None and isinstance(raw_cap, Mapping) and raw_cap.get("abi_rejection_results") is not None:
                abi_rejection_results = list(raw_cap["abi_rejection_results"])
        else:
            rows = list(captured_json)
    elif bench_dispatcher is not None:
        rows = list(bench_dispatcher(plan_doc, int(timeout_s)))
    elif live_profiler:
        lease_ctx = (
            lease_acquirer(sku, int(timeout_s))
            if lease_acquirer is not None
            else acquire_coordinator_host_queue_lease(
                endpoint_alias, npu_count=1, ttl_seconds=int(timeout_s), coordinator_bin=coordinator_bin,
            )
        )
        with lease_ctx as lease_info:
            info = dict(lease_info or {})
            if not info.get("lease_id") or info.get("fence_token") is None:
                raise ModelingError("live_profiler_requires_host_queue_lease_id_and_fence_token")
            granted = [int(d) for d in (info.get("devices") or ()) if not isinstance(d, bool)]
            rows = run_profiler_bench_subprocess(
                plan_doc, sku=sku, profiler_root=profiler_root, allowed_phy_ids=granted, timeout_s=int(timeout_s),
            )

    if rows is None:
        plan_only: dict[str, Any] = {
            "stage": "bench-and-score",
            "ok": False,
            "status": "plan_only",
            "verdict": "insufficient_evidence",
            "error": "bench_and_score_plan_only_no_observations",
            "sku": sku,
            "target_id": str(plan_doc.get("target_id") or ""),
            "family": str(plan_doc.get("family") or ""),
            "plan_sha256": verified_plan_sha,
            "missing_heldout_points": sorted(expected_heldout_ids),
            "abi_rejection_check": evaluate_abi_rejections(plan_doc.get("abi_rejection_points") or (), None),
            "heldout_summary": None,
            "residuals": [],
        }
        plan_only["report_hmac"] = sign_stage_report("operator_score", plan_only)
        if out_score is not None:
            write_atomic_json(out_score, plan_only)
        assert_sanitized_payload(plan_only, field_name="bench_and_score_plan_only")
        return plan_only

    external_train = [str(plan_doc["train_lineage_root"])] if plan_doc.get("train_lineage_root") else []
    scored = score_sealed_gate_cases(
        rows,
        sku=sku,
        runner=runner,
        threshold_pct=cap_threshold,
        external_train_roots=external_train,
        expected_points=expected_heldout_pts,
        expected_train_points=expected_train_pts,
        expected_point_ids=expected_heldout_ids,
        require_disjoint_level_roots=False,
    )
    abi_check = evaluate_abi_rejections(
        list(plan_doc.get("abi_rejection_points") or ()),
        abi_rejection_results,
    )
    missing_points = list(scored["missing_plan_points"])
    if missing_points or scored["status"] == "insufficient_evidence":
        final_ok, final_status = False, "insufficient_evidence"
    elif scored["failing_cases"] or not scored["ok"] or not abi_check["ok"]:
        final_ok, final_status = False, "fail"
    else:
        final_ok, final_status = True, "pass"

    report: dict[str, Any] = {
        "stage": "bench-and-score",
        "ok": final_ok,
        "status": final_status,
        "verdict": final_status,
        "sku": sku,
        "threshold_pct": cap_threshold,
        "target_id": str(plan_doc.get("target_id") or ""),
        "family": str(plan_doc.get("family") or ""),
        "plan_sha256": verified_plan_sha,
        "missing_heldout_points": missing_points,
        "over_budget_points": list(scored["over_budget_points"]),
        "failing_cases": list(scored["failing_cases"]),
        "abi_rejection_check": abi_check,
        "lineage": scored["lineage"],
        "heldout_summary": scored["heldout_summary"],
        "train_summary": scored["train_summary"],
        "residuals": scored["residuals"],
    }
    report["report_hmac"] = sign_stage_report("operator_score", report)
    assert_sanitized_payload(report, field_name="bench_and_score_report")
    if out_score is not None:
        write_atomic_json(out_score, report)
    return report


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
) -> dict[str, Any]:
    """Stage 4: Enforce HMAC-verified reports, pointwise <10%, joined Lean, strict gates."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    root = resolve_repo_root(repo_root)
    c_doc = read_bounded_json(contract_report) if isinstance(contract_report, (str, Path)) else dict(contract_report)
    s_doc = read_bounded_json(score_report) if isinstance(score_report, (str, Path)) else dict(score_report)

    gates: list[GateCheckOutcome] = []
    c_hmac_ok = verify_stage_report("operator_contract", c_doc)
    c_family = str(c_doc.get("family") or "").strip()
    s_family = str(s_doc.get("family") or "").strip()
    s_target_id = str(s_doc.get("target_id") or "").strip()
    op_prefix = operator_id.split(":", 1)[0].strip() if ":" in operator_id else operator_id.strip()
    identity_violations: list[str] = []
    if s_target_id != operator_id:
        identity_violations.append(f"score_report_operator_id_mismatch:expected={operator_id}:got={s_target_id or 'missing'}")
    if c_family and s_family and c_family != s_family:
        identity_violations.append(f"contract_score_family_mismatch:contract={c_family}:score={s_family}")
    if c_family and op_prefix and c_family != op_prefix:
        identity_violations.append(f"contract_operator_id_mismatch:contract_family={c_family}:operator_id={operator_id}")
    contract_ok = c_hmac_ok and bool(c_doc.get("ok")) and not identity_violations
    c_violations = list(c_doc.get("violations") or ())
    if not c_hmac_ok:
        c_violations.append("contract_report_hmac_unverified_or_fabricated")
    c_violations.extend(identity_violations)
    gates.append(
        GateCheckOutcome(
            name="operator_contract",
            passed=contract_ok,
            status="pass" if contract_ok else "fail",
            detail="0 violations" if contract_ok else "; ".join(str(v) for v in c_violations),
            metrics=(
                ("source_refs_checked", int(c_doc.get("source_refs_checked", 0))),
                ("hmac_verified", c_hmac_ok),
                ("identity_bound", not identity_violations),
                ("violations", len(c_violations)),
            ),
        )
    )

    s_hmac_ok = verify_stage_report("operator_score", s_doc)
    sku_match = str(s_doc.get("sku") or "") == sku
    target_match = s_target_id == operator_id
    heldout_rows = [r for r in s_doc.get("residuals", ()) if r.get("split", "heldout") == "heldout"]
    receipts_ok = bool(heldout_rows) and all(
        str(r.get("engine_receipt") or "").startswith(_ENGINE_RECEIPT_PREFIX)
        and int(r.get("predicted_ns") or 0) > 0
        and int(r.get("measured_ns") or 0) > 0
        for r in heldout_rows
    )
    failing_rows = [
        r for r in heldout_rows
        if float(r.get("absolute_percentage_error", 0.0)) * PERCENT_SCALE >= cap_threshold
    ]
    missing_pts = list(s_doc.get("missing_heldout_points") or ())
    max_ape = (s_doc.get("heldout_summary") or {}).get("max_ape_pct")
    abi_check = s_doc.get("abi_rejection_check") or {}
    abi_ok = bool(abi_check.get("ok", False))
    cliff_ok = (
        s_hmac_ok
        and sku_match
        and target_match
        and bool(s_doc.get("ok"))
        and receipts_ok
        and not failing_rows
        and not missing_pts
        and abi_ok
        and max_ape is not None
        and float(max_ape) < cap_threshold
    )
    cliff_detail = (
        f"max_ape_pct={round(float(max_ape), DISPLAY_ROUND_DIGITS)};abi_ok={abi_ok}"
        if cliff_ok and max_ape is not None
        else (
            "score_report_hmac_unverified_or_fabricated" if not s_hmac_ok
            else ("score_report_sku_mismatch" if not sku_match
                  else ("score_report_operator_id_mismatch" if not target_match
                        else ("unverified_engine_receipt" if not receipts_ok
                              else ("score_report_ok_false" if not s_doc.get("ok")
                                    else f"failing={len(failing_rows)};missing={len(missing_pts)};abi_ok={abi_ok}"))))
        )
    )
    gates.append(
        GateCheckOutcome(
            name="heldout_cliff_accuracy",
            passed=cliff_ok,
            status="pass" if cliff_ok else ("insufficient_evidence" if (s_hmac_ok and (not heldout_rows or missing_pts)) else "fail"),
            detail=cliff_detail,
            metrics=(
                ("heldout_count", len(heldout_rows)),
                ("hmac_verified", s_hmac_ok),
                ("failing_cliffs", len(failing_rows)),
                ("missing_heldout_points", len(missing_pts)),
                ("abi_rejection_ok", abi_ok),
                ("max_ape_pct", max_ape),
                ("threshold_pct", cap_threshold),
            ),
        )
    )

    diagnosis_report: dict[str, Any] | None = None
    if failing_rows:
        diagnosis_report = run_tail_cluster_diagnosis(
            heldout_rows,
            repo_root=root,
            threshold_pct=cap_threshold,
            output_path=diagnosis_out,
        )

    expected_timings = {
        str(r["case_id"]): (
            int(r["predicted_ns"]),
            int(r["measured_ns"]),
            int(r["measured_p99_ns"]) if r.get("measured_p99_ns") is not None else None,
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

    artifacts: dict[str, str] = {}
    if s_doc.get("plan_sha256"):
        artifacts["plan_sha256"] = str(s_doc["plan_sha256"])
    if max_ape is not None:
        artifacts["heldout_max_ape_pct"] = str(round(float(max_ape), DISPLAY_ROUND_DIGITS))
    if c_doc.get("target"):
        artifacts["operator_target"] = str(c_doc["target"])

    receipt = seal_onboarding_receipt(
        kind="operator",
        target_id=operator_id,
        sku=sku,
        gates=gates,
        artifacts=artifacts,
        diagnosis_report=diagnosis_report,
    )
    receipt_dict = receipt.to_dict()
    if out_receipt is not None:
        write_atomic_json(out_receipt, receipt_dict)
    return {
        "stage": "certify",
        "operator_id": operator_id,
        "sku": sku,
        "ok": receipt.ok,
        "status": receipt.status,
        "receipt_sha256": receipt.receipt_sha256,
        "lean_checks": [row.to_dict() for row in lean_checked],
        "diagnosis_triggered": diagnosis_report is not None,
        "receipt": receipt_dict,
    }


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
) -> dict[str, Any]:
    """Execute all 4 New-Operator onboarding stages end-to-end."""
    cap_threshold = validate_strict_threshold_pct(threshold_pct)
    base_out = Path(out_dir) if out_dir is not None else None
    stage1 = validate_operator_contract(
        target_path=target_path,
        repo_root=repo_root,
        refs_root=refs_root,
        repo_map=repo_map,
        profiler_root=profiler_root,
        profiler_family=profiler_family,
        profiler_recipe=profiler_recipe,
        profiler_impl=profiler_impl,
        family_registry=family_registry,
        template_catalog=template_catalog,
        template_id=template_id,
        sample_op=sample_op,
        planning_context=planning_context,
        require_sim_owner=require_sim_owner,
        require_taxonomy=require_taxonomy,
        require_profiler_builder=require_profiler_builder,
        require_tiling_axes=require_tiling_axes,
        require_mounted_refs=require_mounted_refs,
    )
    stage2 = generate_operator_cliff_plan(
        operator_id=operator_id,
        family=family,
        sku=sku,
        tile_dim=tile_dim,
        core_count=core_count,
        l2_bytes=l2_bytes,
        bytes_per_token=bytes_per_token,
        topk_threshold=topk_threshold,
        tiling_axes=tiling_axes,
        plan_out=(base_out / "operator_cliff_plan.json") if base_out else None,
    )
    stage3 = bench_and_score_operator(
        cliff_plan=stage2["plan"],
        sku=sku,
        captured_json=captured_json,
        abi_rejection_results=abi_rejection_results,
        bench_dispatcher=bench_dispatcher,
        live_profiler=live_profiler,
        endpoint_alias=endpoint_alias,
        profiler_root=profiler_root,
        lease_acquirer=lease_acquirer,
        coordinator_bin=coordinator_bin,
        runner=runner,
        threshold_pct=cap_threshold,
        out_score=(base_out / "operator_score_report.json") if base_out else None,
    )
    stage4 = certify_operator(
        operator_id=operator_id,
        sku=sku,
        contract_report=stage1,
        score_report=stage3,
        lean_bounds_json=lean_bounds_json,
        repo_root=repo_root,
        threshold_pct=cap_threshold,
        run_repo_gates=run_repo_gates,
        orthogonality_base=orthogonality_base,
        out_receipt=(base_out / "operator_onboarding_receipt.json") if base_out else None,
        diagnosis_out=(base_out / "operator_tail_diagnosis.json") if base_out else None,
    )
    all_ok = bool(stage1["ok"] and stage2["ok"] and stage3["ok"] and stage4["ok"])
    return {
        "stage": "run-all",
        "operator_id": operator_id,
        "sku": sku,
        "ok": all_ok,
        "status": "certified" if all_ok else "failed",
        "receipt_sha256": stage4["receipt_sha256"],
        "stages": {
            "validate-contract": stage1,
            "generate-cliff-plan": stage2,
            "bench-and-score": stage3,
            "certify": stage4,
        },
    }


__all__ = [
    "bench_and_score_operator",
    "certify_operator",
    "generate_operator_cliff_plan",
    "run_operator_onboarding",
    "scaffold_operator",
    "validate_operator_contract",
]
