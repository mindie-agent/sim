"""Turnkey 4-stage New-Operator Onboarding Pipeline for Ascend operators/templates.

[Role] Orchestrate deterministic, repeatable onboarding of a new governed
  operator family or template across four stages plus ``run-all``:
  (1) ``scaffold`` / ``validate-contract`` (delegated to ``_operator_scaffold``
      and ``_operator_contract``),
  (2) ``generate-cliff-plan`` (disjoint ``train`` + off-grid cliff heldout plan
      from tile, core count, L2 footprint, ``topk``, and ``512..400K``),
  (3) ``bench-and-score`` (pure white-box scoring vs silicon ``p50`` with zero
      ``service_regimes`` crutches), and
  (4) ``certify`` (``< 10.0%`` heldout APE across all cliffs, ``0`` Lean
      inversions, ``0`` strict governance/orthogonality violations, automatic
      ``diagnose_tail_cluster`` on failure, and SHA-256 onboarding receipt).
[Invariants] Fails closed on contract violations, ``train==heldout`` lineage
  leakage, ``service_regimes`` crutches, ``>= 10.0%`` cliff APE, or Lean
  dual-bound inversion.
[Timeouts & Locks] Bounded execution budgets and subprocess gates;
  ``TimeoutError`` propagates before ``OSError``. Zero device contexts.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mindie_sim.schema.common import ModelingError

from ._gate_runners import (
    resolve_repo_root,
    run_arch_governance_gate,
    run_tail_cluster_diagnosis,
    run_test_orthogonality_gate,
)
from ._operator_contract import validate_operator_contract
from ._operator_scaffold import scaffold_operator
from .cliff_planner import (
    score_whitebox_observations,
    synthesize_operator_cliff_plan,
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
    assert_sanitized_payload,
    check_lean_non_inversion,
    read_bounded_json,
    seal_onboarding_receipt,
    write_atomic_json,
)


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
    campaign_id: str = "onboard-op",
    plan_out: str | Path | None = None,
) -> dict[str, Any]:
    """Stage 2: Synthesize and optionally write a disjoint train + off-grid cliff plan."""
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
        "plan": payload,
    }


def bench_and_score_operator(
    *,
    cliff_plan: str | Path | Mapping[str, Any],
    sku: str,
    captured_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    bench_dispatcher: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]] | None = None,
    predictor: Callable[[Mapping[str, Any]], int] | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    out_score: str | Path | None = None,
    timeout_s: int = DEFAULT_OUTER_TIMEOUT_S,
) -> dict[str, Any]:
    """Stage 3: Dispatch bench plan or load captured silicon JSON and score white-box predictions."""
    plan_raw = read_bounded_json(cliff_plan) if isinstance(cliff_plan, (str, Path)) else dict(cliff_plan)
    plan_doc = dict(plan_raw["plan"]) if isinstance(plan_raw, Mapping) and "plan" in plan_raw else dict(plan_raw)
    if captured_json is not None:
        if isinstance(captured_json, (str, Path)):
            raw_cap = read_bounded_json(captured_json)
            rows = raw_cap.get("observations", raw_cap) if isinstance(raw_cap, Mapping) else raw_cap
        else:
            rows = list(captured_json)
    elif bench_dispatcher is not None:
        rows = list(bench_dispatcher(plan_doc, int(timeout_s)))
    else:
        raise ModelingError("bench_and_score_requires_captured_json_or_dispatcher")

    external_train = [str(plan_doc["train_lineage_root"])] if plan_doc.get("train_lineage_root") else []
    scored = score_whitebox_observations(
        list(rows),
        sku=sku,
        threshold_pct=threshold_pct,
        predictor=predictor,
        external_train_roots=external_train,
    )
    expected_heldout_ids = {str(pt["point_id"]) for pt in plan_doc.get("heldout_points", ())}
    observed_ids = {str(r["case_id"]) for r in scored["residuals"] if r.get("split") == "heldout"}
    missing_points = sorted(expected_heldout_ids - observed_ids) if expected_heldout_ids else []
    all_ok = scored["ok"] and not missing_points
    report = {
        "stage": "bench-and-score",
        "ok": all_ok,
        "status": "pass" if all_ok else "fail",
        "target_id": str(plan_doc.get("target_id") or ""),
        "plan_sha256": str(plan_doc.get("plan_sha256") or ""),
        "missing_heldout_points": missing_points,
        **scored,
    }
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
    """Stage 4: Enforce <10% cliff APE, 0 Lean inversions, 0 strict violations, emit receipt."""
    root = resolve_repo_root(repo_root)
    c_doc = read_bounded_json(contract_report) if isinstance(contract_report, (str, Path)) else dict(contract_report)
    s_doc = read_bounded_json(score_report) if isinstance(score_report, (str, Path)) else dict(score_report)

    gates: list[GateCheckOutcome] = []
    contract_ok = bool(c_doc.get("ok"))
    c_violations = list(c_doc.get("violations") or ())
    gates.append(
        GateCheckOutcome(
            name="operator_contract",
            passed=contract_ok,
            status="pass" if contract_ok else "fail",
            detail="0 violations" if contract_ok else "; ".join(str(v) for v in c_violations),
            metrics=(
                ("source_refs_checked", int(c_doc.get("source_refs_checked", 0))),
                ("violations", len(c_violations)),
            ),
        )
    )

    heldout_rows = [r for r in s_doc.get("residuals", ()) if r.get("split", "heldout") == "heldout"]
    failing_rows = [
        r for r in heldout_rows
        if float(r.get("absolute_percentage_error", 0.0)) * PERCENT_SCALE >= threshold_pct
    ]
    missing_pts = list(s_doc.get("missing_heldout_points") or ())
    max_ape = (s_doc.get("heldout_summary") or {}).get("max_ape_pct")
    cliff_ok = (
        bool(heldout_rows)
        and not failing_rows
        and not missing_pts
        and max_ape is not None
        and float(max_ape) < threshold_pct
    )
    gates.append(
        GateCheckOutcome(
            name="heldout_cliff_accuracy",
            passed=cliff_ok,
            status="pass" if cliff_ok else "fail",
            detail=(
                f"max_ape_pct={round(float(max_ape), DISPLAY_ROUND_DIGITS)}"
                if max_ape is not None
                else "missing_heldout_observations"
            ),
            metrics=(
                ("heldout_count", len(heldout_rows)),
                ("failing_cliffs", len(failing_rows)),
                ("missing_heldout_points", len(missing_pts)),
                ("max_ape_pct", max_ape),
                ("threshold_pct", threshold_pct),
            ),
        )
    )

    diagnosis_report: dict[str, Any] | None = None
    if failing_rows:
        diagnosis_report = run_tail_cluster_diagnosis(
            heldout_rows,
            repo_root=root,
            threshold_pct=threshold_pct,
            output_path=diagnosis_out,
        )

    if isinstance(lean_bounds_json, (str, Path)):
        raw_lean = read_bounded_json(lean_bounds_json)
        lean_rows = raw_lean.get("bounds", raw_lean) if isinstance(raw_lean, Mapping) else raw_lean
    else:
        lean_rows = list(lean_bounds_json or ())
    lean_ok, lean_checked, lean_violations = check_lean_non_inversion(lean_rows, require_non_empty=True)
    gates.append(
        GateCheckOutcome(
            name="lean_optimality_non_inversion",
            passed=lean_ok,
            status="pass" if lean_ok else "fail",
            detail="0 inversions" if lean_ok else "; ".join(lean_violations),
            metrics=(
                ("checked_cases", len(lean_checked)),
                ("inversions", sum(1 for c in lean_checked if not c.passed)),
            ),
        )
    )

    if run_repo_gates:
        gates.append(run_arch_governance_gate(root, strict=True))
        gates.append(
            run_test_orthogonality_gate(
                root, strict=(orthogonality_base is None), base=orthogonality_base
            )
        )

    artifacts = {
        "target_rel": str(c_doc.get("target_rel") or ""),
        "plan_sha256": str(s_doc.get("plan_sha256") or ""),
        "profiler_builder": str(c_doc.get("profiler_builder") or ""),
    }
    receipt = seal_onboarding_receipt(
        kind="operator",
        target_id=operator_id,
        sku=sku,
        gates=gates,
        artifacts=artifacts,
        diagnosis_report=diagnosis_report,
    )
    payload = receipt.to_dict()
    if out_receipt is not None:
        write_atomic_json(out_receipt, payload)
    return payload


def run_operator_onboarding(
    *,
    operator_id: str,
    target_path: str | Path,
    sku: str,
    tile_dim: int,
    core_count: int,
    l2_bytes: int,
    captured_json: str | Path | Sequence[Mapping[str, Any]],
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]],
    repo_root: str | Path | None = None,
    refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None,
    profiler_root: str | Path | None = None,
    expected_profiler_builder: str | None = None,
    bytes_per_token: int = 0,
    topk_threshold: int | None = None,
    out_dir: str | Path | None = None,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
) -> dict[str, Any]:
    """Run all 4 operator onboarding stages deterministically and emit a signed receipt."""
    work_dir = Path(out_dir) if out_dir is not None else None
    stage1 = validate_operator_contract(
        target_path=target_path,
        repo_root=repo_root,
        refs_root=refs_root,
        repo_map=repo_map,
        profiler_root=profiler_root,
        expected_profiler_builder=expected_profiler_builder,
    )
    stage2 = generate_operator_cliff_plan(
        operator_id=operator_id,
        family=stage1["family"],
        sku=sku,
        tile_dim=tile_dim,
        core_count=core_count,
        l2_bytes=l2_bytes,
        bytes_per_token=bytes_per_token,
        topk_threshold=topk_threshold,
        plan_out=(work_dir / "operator_cliff_plan.json") if work_dir else None,
    )
    stage3 = bench_and_score_operator(
        cliff_plan=stage2["plan"],
        sku=sku,
        captured_json=captured_json,
        threshold_pct=threshold_pct,
        out_score=(work_dir / "operator_score.json") if work_dir else None,
    )
    stage4 = certify_operator(
        operator_id=operator_id,
        sku=sku,
        contract_report=stage1,
        score_report=stage3,
        lean_bounds_json=lean_bounds_json,
        repo_root=repo_root,
        threshold_pct=threshold_pct,
        run_repo_gates=run_repo_gates,
        orthogonality_base=orthogonality_base,
        out_receipt=(work_dir / "operator_onboarding_receipt.json") if work_dir else None,
        diagnosis_out=(work_dir / "operator_tail_diagnosis.json") if work_dir else None,
    )
    return {
        "stage": "run-all",
        "operator_id": operator_id,
        "sku": sku,
        "ok": bool(stage4["ok"]),
        "status": stage4["status"],
        "stages": {
            "validate_contract": stage1,
            "generate_cliff_plan": {k: v for k, v in stage2.items() if k != "plan"},
            "bench_and_score": {k: v for k, v in stage3.items() if k != "residuals"},
            "certify": stage4,
        },
        "receipt": stage4,
    }


__all__ = [
    "bench_and_score_operator",
    "certify_operator",
    "generate_operator_cliff_plan",
    "run_operator_onboarding",
    "scaffold_operator",
    "validate_operator_contract",
]
