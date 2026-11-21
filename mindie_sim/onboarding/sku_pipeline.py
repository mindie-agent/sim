"""Turnkey 4-stage New-SKU Onboarding Pipeline for Ascend hardware.

[Role] Orchestrate deterministic, repeatable onboarding of a new or existing
  Ascend SKU across four governed stages plus ``run-all``:
  (1) ``probe-soc`` (delegated to ``_soc_probe``),
  (2) ``calibrate-l0-and-fabric`` (``HostQueue`` lease / idle-die precheck +
      ``P01..P20`` ``>= 3x L2`` ``ColdRing`` + fabric/DMA ``train`` sealing),
  (3) ``validate-512-400k`` (disjoint Anchor + Off-Grid Cliff Heldout +
      L1/L2/L3 scoring), and
  (4) ``verify-gates`` (``multilevel_gate``, Lean 4 ``OptimalityCertificate``
      non-inversion, ``check_arch_governance --strict``,
      ``check_test_orthogonality --strict``, automatic ``diagnose_tail_cluster``
      on cliff failure, and signed SHA-256 onboarding receipt).
[Invariants] Fails closed on ``silicon_profile_conflict``, ``train``/``heldout``
  or ``L1/L2/L3`` lineage leakage, ``service_regimes`` crutches, or Lean
  inversion. Enforces ``outer_timeout >= sum(inner_budgets) + cleanup_grace``
  and ``except TimeoutError: raise`` before ``except OSError:``.
[Timeouts & Locks] Bounded execution budgets; leases always unwind via context
  managers.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
import re
from typing import Any

from mindie_sim.observations.multilevel_gate import (
    CohortKey,
    EvaluationRunner,
    FrozenFitManifest,
    FrozenFitParameter,
    Residual,
    evaluate as evaluate_multilevel,
    gate_decision,
    stream_manifest,
    summarize_cohorts,
    validate_lineage,
)
from mindie_sim.primitives.schema import (
    PRIMITIVE_SCHEMA_VERSION,
    parse_primitive_table,
)
from mindie_sim.schema.datapath import datapath_for_die

from ._gate_runners import (
    resolve_repo_root,
    run_arch_governance_gate,
    run_live_l0_bench_subprocess,
    run_tail_cluster_diagnosis,
    run_test_orthogonality_gate,
)
from ._primitive_sealer import materialize_sku_hardware, seal_primitive_table
from ._soc_probe import (
    ARCH_PRESET_35,
    parse_npu_smi_topology,
    parse_platform_config_ini,
    probe_soc,
    verify_idle_die_telemetry,
)
from .cliff_planner import (
    STANDARD_LEVEL_BOUNDARY_KINDS,
    score_whitebox_observations,
    synthesize_sku_512_400k_plan,
    validate_disjoint_lineage,
)
from .contracts import (
    ANCHOR_CONTEXT_GRID,
    COLD_RING_L2_MULTIPLIER,
    DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_FABRIC_BUDGET_S,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DEFAULT_L0_BUDGET_S,
    DISPLAY_ROUND_DIGITS,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    PERCENT_SCALE,
    SHORT_DIGEST_LEN,
    TRAIN_OBSERVATION_MANIFEST_SCHEMA,
    GateCheckOutcome,
    TimeoutBudget,
    assert_sanitized_payload,
    check_lean_non_inversion,
    read_bounded_json,
    seal_onboarding_receipt,
    write_atomic_json,
)

REQUIRED_MULTILEVEL_LEVELS = ("L1", "L2", "L3")
_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def calibrate_l0_and_fabric(
    *,
    sku: str,
    endpoint_alias: str = "",
    raw_l0_json: str | Path | None = None,
    raw_fabric_json: str | Path | None = None,
    out_primitives: str | Path | None = None,
    manifest_path: str | Path | None = None,
    train_manifest_out: str | Path | None = None,
    campaign_id: str = "camp-onboard-l0",
    cann_version: str = "unmeasured",
    l0_budget_s: int = DEFAULT_L0_BUDGET_S,
    fabric_budget_s: int = DEFAULT_FABRIC_BUDGET_S,
    cleanup_grace_s: int = DEFAULT_CLEANUP_GRACE_S,
    outer_timeout_s: int | None = None,
    idle_precheck: Mapping[str, Any] | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
    l0_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
    fabric_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
    live_bench: bool = False,
    repo_root: str | Path | None = None,
    profile_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 2: Acquire lease/idle check, run L0+fabric train calibration, seal table."""
    resolved_outer = (
        l0_budget_s + fabric_budget_s + cleanup_grace_s
        if outer_timeout_s is None
        else int(outer_timeout_s)
    )
    budget = TimeoutBudget(
        inner_budgets_s=(
            ("l0_primitives_p01_p20", int(l0_budget_s)),
            ("fabric_host_dma_train", int(fabric_budget_s)),
        ),
        cleanup_grace_s=int(cleanup_grace_s),
        outer_timeout_s=resolved_outer,
    )
    if idle_precheck is not None:
        verify_idle_die_telemetry(
            str(idle_precheck.get("info_text") or ""),
            str(idle_precheck.get("common_text") or ""),
            npu_id=int(idle_precheck.get("npu_id", 0)),
        )

    lease_ctx = (
        lease_acquirer(sku, budget.outer_timeout_s)
        if lease_acquirer is not None
        else nullcontext({"lease_mode": "offline_replay" if raw_l0_json is not None else "idle_precheck"})
    )
    lease_info: dict[str, Any] = {}
    with lease_ctx as acquired:
        lease_info = dict(acquired or {})
        if raw_l0_json is not None:
            raw_l0: Mapping[str, Any] | None = read_bounded_json(raw_l0_json)
        elif l0_runner is not None:
            raw_l0 = dict(l0_runner(sku, int(l0_budget_s)))
        elif live_bench:
            raw_l0 = run_live_l0_bench_subprocess(
                sku=sku,
                endpoint_alias=endpoint_alias,
                campaign_id=campaign_id,
                cann_version=cann_version,
                repo_root=repo_root,
                timeout_s=int(l0_budget_s),
            )
        else:
            raw_l0 = None
        raw_fabric = read_bounded_json(raw_fabric_json) if raw_fabric_json is not None else (
            dict(fabric_runner(sku, int(fabric_budget_s))) if fabric_runner is not None else None
        )

    payload, capture_uri = seal_primitive_table(
        sku=sku,
        cann_version=cann_version,
        endpoint_alias=endpoint_alias,
        campaign_id=campaign_id,
        raw_l0=raw_l0,
        raw_fabric=raw_fabric,
        profile_override=profile_override,
    )
    clean_sku_slug = _SAFE_TOKEN_RE.sub("_", sku.lower()).strip("_")
    arch_prefix = "arch35" if payload["arch_preset"] == ARCH_PRESET_35 else "arch22"
    default_rel = f"data/primitives/{arch_prefix}_{clean_sku_slug}.json"
    if out_primitives is not None:
        out_path = Path(out_primitives)
        write_atomic_json(out_path, payload)
        if manifest_path is not None:
            mpath = Path(manifest_path)
            manifest = (
                read_bounded_json(mpath)
                if mpath.is_file()
                else {"schema_version": PRIMITIVE_SCHEMA_VERSION, "tables": {}}
            )
            manifest.setdefault("tables", {})[sku] = {
                "file": out_path.name,
                "sku": sku,
                "compatible_skus": [sku],
                "arch_preset": payload["arch_preset"],
                "cann_version": cann_version,
                "calibrated": bool(payload["provenance"]["calibrated"]),
                "calibrated_primitive_count": sum(
                    1 for c in payload["primitives"].values() if c["status"] == "calibrated"
                ),
                "total_primitives": len(payload["primitives"]),
                "envelope_sha256": payload["envelope_sha256"],
                "capture_uri": capture_uri,
            }
            write_atomic_json(mpath, manifest)

    train_root = f"acq-train-{payload['envelope_sha256'][:SHORT_DIGEST_LEN]}"
    train_manifest = {
        "schema": TRAIN_OBSERVATION_MANIFEST_SCHEMA,
        "sku": sku,
        "split": "train",
        "campaign_id": campaign_id,
        "train_lineage_root": train_root,
        "cold_ring_l2_multiplier": COLD_RING_L2_MULTIPLIER,
        "primitives_envelope_sha256": payload["envelope_sha256"],
        "capture_uri": capture_uri,
        "calibrated_primitives": sorted(
            k for k, v in payload["primitives"].items() if v["status"] == "calibrated"
        ),
    }
    if train_manifest_out is not None:
        write_atomic_json(train_manifest_out, train_manifest)

    result = {
        "stage": "calibrate-l0-and-fabric",
        "sku": sku,
        "ok": True,
        "status": "ok",
        "default_table_path": default_rel,
        "timeout_budget": budget.to_dict(),
        "lease": lease_info,
        "envelope_sha256": payload["envelope_sha256"],
        "capture_uri": capture_uri,
        "calibrated_count": len(train_manifest["calibrated_primitives"]),
        "total_primitives": len(payload["primitives"]),
        "train_manifest": train_manifest,
        "primitives_payload": payload,
    }
    assert_sanitized_payload(result, field_name="calibrate_l0_result")
    return result


def validate_512_400k(
    *,
    sku: str,
    observations_json: str | Path | None = None,
    train_manifest_json: str | Path | None = None,
    plan_out: str | Path | None = None,
    report_out: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    campaign_id: str = "camp-onboard-512-400k",
    profiler_runner: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    predictor: Callable[[Mapping[str, Any]], int] | None = None,
    profile_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 3: Orchestrate and score disjoint Anchor + Off-Grid Cliff 512..400K + L1/L2/L3."""
    hw = materialize_sku_hardware(sku, profile_override=profile_override)
    die = hw.dies[0]
    dp = datapath_for_die(die)
    l2_bytes = int(dict(die.buffers).get("l2", 0))
    plan = synthesize_sku_512_400k_plan(
        sku,
        l2_bytes=l2_bytes,
        aic_count=int(dp.aic_count),
        aiv_count=int(dp.aiv_count),
        campaign_id=campaign_id,
    )
    if plan_out is not None:
        write_atomic_json(plan_out, plan)

    external_train_roots: list[str] = []
    if train_manifest_json is not None:
        tman = read_bounded_json(train_manifest_json)
        if isinstance(tman, Mapping) and tman.get("train_lineage_root"):
            external_train_roots.append(str(tman["train_lineage_root"]))

    if observations_json is not None:
        raw_obs = read_bounded_json(observations_json)
        rows = raw_obs.get("observations", raw_obs) if isinstance(raw_obs, Mapping) else raw_obs
    elif profiler_runner is not None:
        rows = profiler_runner(plan)
    else:
        return {
            "stage": "validate-512-400k",
            "sku": sku,
            "ok": True,
            "status": "plan_only",
            "plan": plan,
        }

    scored = score_whitebox_observations(
        list(rows),
        sku=sku,
        threshold_pct=threshold_pct,
        predictor=predictor,
        external_train_roots=external_train_roots,
    )
    missing_anchors = [c for c in ANCHOR_CONTEXT_GRID if c not in scored["covered_anchor_contexts"]]
    missing_cliffs = [c for c in OFF_GRID_CLIFF_HELDOUT_GRID if c not in scored["covered_cliff_contexts"]]
    covered_levels = sorted(scored["heldout_roots_by_level"].keys())
    missing_levels = [lvl for lvl in REQUIRED_MULTILEVEL_LEVELS if lvl not in covered_levels]
    ok = scored["ok"] and not missing_cliffs and not missing_levels
    report = {
        "stage": "validate-512-400k",
        "sku": sku,
        "ok": ok,
        "status": "pass" if ok else "fail",
        "plan_sha256": plan["plan_sha256"],
        "anchor_grid": list(ANCHOR_CONTEXT_GRID),
        "off_grid_cliff_heldout_grid": list(OFF_GRID_CLIFF_HELDOUT_GRID),
        "missing_anchor_contexts": missing_anchors,
        "missing_cliff_contexts": missing_cliffs,
        "covered_levels": covered_levels,
        "missing_levels": missing_levels,
        **scored,
    }
    assert_sanitized_payload(report, field_name="validate_512_400k_report")
    if report_out is not None:
        write_atomic_json(report_out, report)
    return report


def _residuals_from_rows(rows: Sequence[Mapping[str, Any]], sku: str) -> list[Residual]:
    out: list[Residual] = []
    for row in rows:
        if row.get("split", "heldout") != "heldout":
            continue
        c_raw = row.get("cohort") or {}
        level = str(row.get("level") or c_raw.get("level") or "L1")
        cohort = CohortKey(
            level=level,  # type: ignore[arg-type]
            cohort_name=str(c_raw.get("cohort_name") or "onboarding"),
            sku=str(c_raw.get("sku") or sku),
            phase=str(c_raw.get("phase") or "decode"),
            mode=str(c_raw.get("mode") or "eager"),
            shape_regime=str(c_raw.get("shape_regime") or "off_grid_cliff"),
            parallel_topology=str(c_raw.get("parallel_topology") or "tp1_dp1"),
        )
        out.append(
            Residual(
                case_id=str(row["case_id"]),
                level=level,
                cohort=cohort,
                acquisition_root=str(row["acquisition_root"]),
                measured_ns=int(row["measured_ns"]),
                predicted_ns=int(row["predicted_ns"]),
                engine_receipt=str(row.get("engine_receipt") or ""),
                signed_error=float(row["signed_error"]),
                absolute_percentage_error=float(row["absolute_percentage_error"]),
                status=str(row.get("status") or "ok"),  # type: ignore[arg-type]
                boundary_kind=str(
                    row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS.get(level, "single_operator")
                ),
            )
        )
    return out


def _multilevel_outcome(
    res_objs: Sequence[Residual], threshold_pct: float
) -> GateCheckOutcome:
    summaries = summarize_cohorts(res_objs, threshold_pct=threshold_pct)
    decision = gate_decision(
        summaries,
        required_levels=REQUIRED_MULTILEVEL_LEVELS,
        require_disjoint_level_roots=True,
    )
    max_ape = max((s.max_ape_pct for s in summaries.values() if s.max_ape_pct is not None), default=None)
    cliff_ok = max_ape is not None and max_ape < threshold_pct
    ml_ok = decision["status"] == "pass" and cliff_ok
    return GateCheckOutcome(
        name="multilevel_gate",
        passed=ml_ok,
        status="pass" if ml_ok else "fail",
        detail=f"status={decision['status']} max_ape_pct={max_ape}",
        metrics=(
            ("covered_levels", decision["covered_levels"]),
            ("missing_levels", decision["missing_levels"]),
            ("overlapping_level_roots", decision["overlapping_level_roots"]),
            ("max_ape_pct", round(max_ape, DISPLAY_ROUND_DIGITS) if max_ape is not None else None),
            ("threshold_pct", threshold_pct),
        ),
    )


def verify_sku_gates(
    *,
    sku: str,
    validation_report_json: str | Path | Mapping[str, Any] | None = None,
    multilevel_manifest_json: str | Path | None = None,
    fits_json: str | Path | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    primitives_json: str | Path | None = None,
    repo_root: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    out_receipt: str | Path | None = None,
    diagnosis_out: str | Path | None = None,
    runner: EvaluationRunner | None = None,
) -> dict[str, Any]:
    """Stage 4: Run multilevel, Lean non-inversion, governance, and orthogonality gates."""
    root = resolve_repo_root(repo_root)
    gates: list[GateCheckOutcome] = []
    artifacts: dict[str, str] = {}
    residuals_for_diag: list[dict[str, Any]] = []
    diagnosis_report: dict[str, Any] | None = None

    if primitives_json is not None:
        prim_raw = read_bounded_json(primitives_json)
        table = parse_primitive_table(prim_raw, verify_envelope=True)
        artifacts["primitives_envelope_sha256"] = table.envelope_sha256
        artifacts["primitives_capture_uri"] = table.provenance.capture_uri

    if multilevel_manifest_json is not None and fits_json is not None:
        manifest = stream_manifest(multilevel_manifest_json)
        fits_raw = read_bounded_json(fits_json)
        fits = FrozenFitManifest(
            fit_id=str(fits_raw.get("fit_id") or "fits"),
            schema_version=str(fits_raw.get("schema_version") or "mindie-sim.frozen-fits/1"),
            parameters=tuple(
                FrozenFitParameter(
                    parameter_id=str(p["parameter_id"]),
                    training_digests=tuple(str(d) for d in p.get("training_digests", ())),
                    source_uri=str(p.get("source_uri") or ""),
                )
                for p in fits_raw.get("parameters", ())
            ),
            lineage_edges=tuple(
                (str(e[0]), str(e[1])) for e in fits_raw.get("lineage_edges", ())
            ),
        )
        validate_lineage(manifest, fits)
        if runner is not None:
            res_objs = list(evaluate_multilevel(manifest, fits, runner, target_split="heldout"))
            residuals_for_diag = [r.to_dict() for r in res_objs]
        elif validation_report_json is not None:
            val_doc = (
                read_bounded_json(validation_report_json)
                if isinstance(validation_report_json, (str, Path))
                else dict(validation_report_json)
            )
            residuals_for_diag = [
                r for r in val_doc.get("residuals", ()) if r.get("split", "heldout") == "heldout"
            ]
            res_objs = _residuals_from_rows(residuals_for_diag, sku)
        else:
            res_objs = []
        gates.append(_multilevel_outcome(res_objs, threshold_pct))
    elif validation_report_json is not None:
        val_doc = (
            read_bounded_json(validation_report_json)
            if isinstance(validation_report_json, (str, Path))
            else dict(validation_report_json)
        )
        all_rows = list(val_doc.get("residuals") or ())
        train_roots = [str(r["acquisition_root"]) for r in all_rows if r.get("split") == "train"]
        train_roots.extend(str(r) for r in val_doc.get("train_roots", ()))
        heldout_rows = [r for r in all_rows if r.get("split", "heldout") == "heldout"]
        level_roots: dict[str, list[str]] = {}
        for row in heldout_rows:
            level_roots.setdefault(str(row.get("level") or "L1"), []).append(str(row["acquisition_root"]))
        validate_disjoint_lineage(
            train_roots,
            [str(r["acquisition_root"]) for r in heldout_rows],
            level_roots=level_roots,
        )
        residuals_for_diag = heldout_rows
        gates.append(_multilevel_outcome(_residuals_from_rows(heldout_rows, sku), threshold_pct))
    else:
        gates.append(
            GateCheckOutcome(
                name="multilevel_gate",
                passed=False,
                status="insufficient_evidence",
                detail="validation_report_json or multilevel_manifest_json required",
            )
        )

    failing_rows = [
        r for r in residuals_for_diag
        if float(r.get("absolute_percentage_error", 0.0)) * PERCENT_SCALE >= threshold_pct
    ]
    if failing_rows:
        diagnosis_report = run_tail_cluster_diagnosis(
            residuals_for_diag,
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

    receipt = seal_onboarding_receipt(
        kind="sku",
        target_id=sku,
        sku=sku,
        gates=gates,
        artifacts=artifacts,
        diagnosis_report=diagnosis_report,
    )
    payload = receipt.to_dict()
    if out_receipt is not None:
        write_atomic_json(out_receipt, payload)
    return payload


def run_sku_onboarding(
    *,
    sku: str | None = None,
    endpoint_alias: str = "",
    probe_json: str | Path | None = None,
    raw_l0_json: str | Path | None = None,
    raw_fabric_json: str | Path | None = None,
    observations_json: str | Path | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    out_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_scaffold: bool = False,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    cann_version: str = "unmeasured",
) -> dict[str, Any]:
    """Run all 4 SKU onboarding stages deterministically and emit a signed receipt."""
    work_dir = Path(out_dir) if out_dir is not None else None
    stage1 = probe_soc(
        endpoint_alias=endpoint_alias,
        probe_json=probe_json,
        allow_scaffold=allow_scaffold,
        scaffold_out=(work_dir / "scaffold_profile.json") if (work_dir and allow_scaffold) else None,
    )
    resolved_sku = str(sku or stage1["sku"])
    if not stage1["ok"]:
        receipt = seal_onboarding_receipt(
            kind="sku",
            target_id=resolved_sku,
            sku=resolved_sku,
            gates=(
                GateCheckOutcome(
                    name="probe_soc",
                    passed=False,
                    status="fail",
                    detail=stage1.get("error") or "probe_soc_failed",
                ),
            ),
            artifacts={"probe_capture_uri": stage1["probe"]["capture_uri"]},
        ).to_dict()
        if work_dir is not None:
            write_atomic_json(work_dir / "sku_onboarding_receipt.json", receipt)
        return {
            "stage": "run-all",
            "sku": resolved_sku,
            "ok": False,
            "status": "failed",
            "stages": {"probe_soc": stage1},
            "receipt": receipt,
        }

    scaffold_override = stage1.get("scaffold_profile")
    stage2 = calibrate_l0_and_fabric(
        sku=resolved_sku,
        endpoint_alias=endpoint_alias,
        raw_l0_json=raw_l0_json,
        raw_fabric_json=raw_fabric_json,
        cann_version=cann_version,
        out_primitives=(work_dir / "primitives.json") if work_dir else None,
        train_manifest_out=(work_dir / "train_manifest.json") if work_dir else None,
        profile_override=scaffold_override,
    )
    stage3 = validate_512_400k(
        sku=resolved_sku,
        observations_json=observations_json,
        train_manifest_json=(work_dir / "train_manifest.json") if work_dir else None,
        plan_out=(work_dir / "cliff_512_400k_plan.json") if work_dir else None,
        report_out=(work_dir / "validation_512_400k.json") if work_dir else None,
        threshold_pct=threshold_pct,
        profile_override=scaffold_override,
    )
    stage4 = verify_sku_gates(
        sku=resolved_sku,
        validation_report_json=stage3,
        lean_bounds_json=lean_bounds_json,
        primitives_json=(work_dir / "primitives.json") if work_dir else None,
        repo_root=repo_root,
        threshold_pct=threshold_pct,
        run_repo_gates=run_repo_gates,
        orthogonality_base=orthogonality_base,
        out_receipt=(work_dir / "sku_onboarding_receipt.json") if work_dir else None,
        diagnosis_out=(work_dir / "tail_diagnosis.json") if work_dir else None,
    )
    return {
        "stage": "run-all",
        "sku": resolved_sku,
        "ok": bool(stage4["ok"]),
        "status": stage4["status"],
        "stages": {
            "probe_soc": {k: v for k, v in stage1.items() if k != "scaffold_profile"},
            "calibrate_l0_and_fabric": {k: v for k, v in stage2.items() if k != "primitives_payload"},
            "validate_512_400k": {k: v for k, v in stage3.items() if k != "residuals"},
            "verify_gates": stage4,
        },
        "receipt": stage4,
    }


__all__ = [
    "calibrate_l0_and_fabric",
    "parse_npu_smi_topology",
    "parse_platform_config_ini",
    "probe_soc",
    "run_sku_onboarding",
    "validate_512_400k",
    "verify_idle_die_telemetry",
    "verify_sku_gates",
]
