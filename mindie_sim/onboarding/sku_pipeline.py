"""Turnkey 4-stage New-SKU Onboarding Pipeline for Ascend hardware.

[Role] Orchestrate deterministic, repeatable onboarding of a new or existing
  Ascend SKU across four governed stages plus ``run-all``:
  (1) ``probe-soc`` (delegated to ``_soc_probe``),
  (2) ``calibrate-l0-and-fabric`` (real ``HostQueue`` ``lease_id``/``fence_token``
      + canonical ``tools/bench_l0_primitives.py`` sealer; fails closed when
      ``calibrated_count == 0``),
  (3) ``validate-512-400k`` (disjoint 2D Anchor + Off-Grid Cliff + ABI-guard
      plan scored strictly via ``GateCase -> EvaluationRunner -> EngineRunReceipt``;
      fails closed on ``plan_only``, missing grid cells, clock mismatch, or any
      pointwise ``APE >= 10.0%``), and
  (4) ``verify-gates`` (pointwise multilevel ``L1 != L2 != L3``, non-degenerate
      witnessed Lean 4 ``OptimalityCertificate`` non-inversion, strict repo
      governance/orthogonality, automatic ``diagnose_tail_cluster`` on failure,
      and SHA-256 onboarding receipt).
[Invariants] Zero duplicate primitive sealers; zero caller-supplied
  ``predicted_*``; zero ``**scored`` verdict overwrites; ``calibrated_count == 0``
  and ``plan_only`` ALWAYS set ``ok=False`` (non-zero CLI exit). Enforces
  ``outer_timeout >= sum(inner_budgets) + cleanup_grace`` and
  ``except TimeoutError: raise`` before ``except OSError:``.
[Timeouts & Locks] Bounded execution budgets; ``HostQueue`` leases always unwind
  via context managers.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from mindie_sim.backends import ascend
from mindie_sim.observations.multilevel_gate import EvaluationRunner
from mindie_sim.primitives.schema import parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Hardware

from ._gate_runners import (
    acquire_coordinator_host_queue_lease,
    load_canonical_l0_sealer,
    resolve_repo_root,
    run_arch_governance_gate,
    run_live_l0_bench_subprocess,
    run_profiler_bench_subprocess,
    run_tail_cluster_diagnosis,
    run_test_orthogonality_gate,
)
from ._sealed_scorer import REQUIRED_MULTILEVEL_LEVELS, run_multilevel_verdict
from ._soc_probe import (
    ARCH_PRESET_35,
    parse_npu_smi_topology,
    parse_platform_config_ini,
    probe_soc,
    verify_idle_die_telemetry,
)
from .cliff_planner import (
    score_sealed_gate_cases,
    synthesize_sku_512_400k_plan,
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

_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _materialize_sku_hardware(
    sku: str,
    *,
    profile_override: Mapping[str, Any] | None = None,
) -> Hardware:
    if profile_override is not None:
        raw_machine = {
            "sku": sku,
            "architecture": profile_override["architecture"],
            "dies_per_package": profile_override["dies_per_package"],
            "ranks_per_node": int(profile_override.get("ranks_per_node", 1)),
            "num_devices": 1,
            "device": deepcopy(profile_override["device"]),
            "transfer_channels": deepcopy(profile_override.get("transfer_channels", {})),
        }
        return Hardware.from_mapping(raw_machine)
    return Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))


def _merge_l0_and_fabric_measurements(
    raw_l0: Mapping[str, Any] | None,
    raw_fabric: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_l0 is None and raw_fabric is None:
        return None
    merged: dict[str, Any] = dict(raw_l0 or {})
    if raw_fabric:
        for key, value in raw_fabric.items():
            if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
                combined = dict(merged[key])
                combined.update(value)
                merged[key] = combined
            else:
                merged[key] = value
    return merged


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
    coordinator_bin: str = "",
) -> dict[str, Any]:
    """Stage 2: Acquire HostQueue lease, run L0+fabric calibration, seal via canonical sealer."""
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

    use_live = bool(live_bench and raw_l0_json is None and l0_runner is None)
    if lease_acquirer is not None:
        lease_ctx: AbstractContextManager[Mapping[str, Any]] = lease_acquirer(
            sku, budget.outer_timeout_s
        )
    elif use_live:
        lease_ctx = acquire_coordinator_host_queue_lease(
            endpoint_alias,
            devices=1,
            ttl_s=budget.outer_timeout_s,
            coordinator_bin=coordinator_bin,
        )
    else:
        lease_ctx = nullcontext(
            {"lease_mode": "offline_replay" if (raw_l0_json is not None or l0_runner is not None) else "uncalibrated_dry_run"}
        )

    lease_info: dict[str, Any] = {}
    with lease_ctx as acquired:
        lease_info = dict(acquired or {})
        if use_live and (not lease_info.get("lease_id") or lease_info.get("fence_token") is None):
            raise ModelingError("live_calibration_requires_host_queue_lease_id_and_fence_token")
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

    sealer = load_canonical_l0_sealer(repo_root)
    merged_raw = _merge_l0_and_fabric_measurements(raw_l0, raw_fabric)
    rec_alias = sealer.sanitize_record_alias(endpoint_alias or sku, sku)
    payload = sealer.build_primitive_table_payload(
        sku=sku,
        cann_version=cann_version,
        endpoint_alias=rec_alias,
        campaign_id=campaign_id,
        raw_measurements=merged_raw,
    )
    parse_primitive_table(payload, verify_envelope=True)
    capture_uri = str(payload["provenance"]["capture_uri"])
    clean_sku_slug = _SAFE_TOKEN_RE.sub("_", sku.lower()).strip("_")
    arch_prefix = "arch35" if payload["arch_preset"] == ARCH_PRESET_35 else "arch22"
    default_rel = f"data/primitives/{arch_prefix}_{clean_sku_slug}.json"

    if out_primitives is not None:
        out_path = Path(out_primitives)
        write_atomic_json(out_path, payload)
        cap_path = out_path.parent / "captures" / f"{out_path.stem}.capture.json"
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        cap_path.write_bytes(sealer.canonical_capture_bytes(sku, merged_raw))
        if manifest_path is not None:
            sealer.update_manifest(Path(manifest_path), out_path, payload)

    calibrated_list = sorted(
        k for k, v in payload["primitives"].items() if v["status"] == "calibrated"
    )
    calibrated_count = len(calibrated_list)
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
        "calibrated_primitives": calibrated_list,
    }
    if train_manifest_out is not None:
        write_atomic_json(train_manifest_out, train_manifest)

    # Prevention Lock 2: zero calibrated primitives NEVER reports ok=True.
    cal_ok = calibrated_count > 0
    result = {
        "stage": "calibrate-l0-and-fabric",
        "sku": sku,
        "ok": cal_ok,
        "status": "ok" if cal_ok else "uncalibrated",
        "error": "" if cal_ok else "l0_calibration_zero_calibrated_primitives",
        "sealer_owner": "tools/bench_l0_primitives.py",
        "default_table_path": default_rel,
        "timeout_budget": budget.to_dict(),
        "lease": lease_info,
        "envelope_sha256": payload["envelope_sha256"],
        "capture_uri": capture_uri,
        "calibrated_count": calibrated_count,
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
    runner: EvaluationRunner | None = None,
    profiler_runner: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    live_profiler: bool = False,
    endpoint_alias: str = "",
    profiler_root: str | Path | None = None,
    profile_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 3: Orchestrate and score disjoint 2D Anchor + Cliff 512..400K + L1/L2/L3."""
    hw = _materialize_sku_hardware(sku, profile_override=profile_override)
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

    external_train_roots: list[str] = [plan.train_lineage_root]
    if train_manifest_json is not None:
        t_doc = read_bounded_json(train_manifest_json)
        t_root = str(t_doc.get("train_lineage_root") or "").strip()
        if t_root:
            external_train_roots.append(t_root)

    obs_rows: list[Mapping[str, Any]] | None = None
    if observations_json is not None:
        raw_obs = read_bounded_json(observations_json)
        if isinstance(raw_obs, Mapping):
            obs_rows = list(raw_obs.get("observations") or raw_obs.get("cases") or ())
        elif isinstance(raw_obs, list):
            obs_rows = list(raw_obs)
        else:
            raise ModelingError("validate_512_400k_observations_invalid")
    elif profiler_runner is not None:
        obs_rows = list(profiler_runner(plan_doc))
    elif live_profiler:
        obs_rows = run_profiler_bench_subprocess(
            plan_doc,
            sku=sku,
            endpoint_alias=endpoint_alias,
            profiler_root=profiler_root,
        )

    expected_heldout_ids = [pt.point_id for pt in plan.heldout_points]
    if obs_rows is None:
        # Prevention Lock 2: plan_only MUST fail closed (ok=False, non-zero CLI exit).
        plan_only = {
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
            "heldout_summary": None,
            "residuals": [],
        }
        if report_out is not None:
            write_atomic_json(report_out, plan_only)
        assert_sanitized_payload(plan_only, field_name="validate_512_400k_plan_only")
        return plan_only

    scored = score_sealed_gate_cases(
        obs_rows,
        sku=sku,
        runner=runner,
        threshold_pct=threshold_pct,
        external_train_roots=external_train_roots,
        expected_point_ids=expected_heldout_ids,
        require_disjoint_level_roots=True,
    )
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
    elif scored["failing_cases"] or not scored["ok"]:
        final_ok, final_status = False, "fail"
    else:
        final_ok, final_status = True, "pass"

    report = {
        "stage": "validate-512-400k",
        "sku": sku,
        "ok": final_ok,
        "status": final_status,
        "verdict": final_status,
        "threshold_pct": float(threshold_pct),
        "plan_sha256": plan.plan_sha256,
        "anchor_grid": list(ANCHOR_CONTEXT_GRID),
        "off_grid_cliff_grid": list(OFF_GRID_CLIFF_HELDOUT_GRID),
        "missing_plan_points": missing_plan_points,
        "missing_anchors": missing_anchors,
        "missing_cliffs": missing_cliffs,
        "over_budget_points": list(scored["over_budget_points"]),
        "failing_cases": list(scored["failing_cases"]),
        "lineage": scored["lineage"],
        "heldout_summary": scored["heldout_summary"],
        "train_summary": scored["train_summary"],
        "residuals": scored["residuals"],
        "plan": plan_doc,
    }
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
    repo_root: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    run_repo_gates: bool = True,
    orthogonality_base: str | None = None,
    out_receipt: str | Path | None = None,
    diagnosis_out: str | Path | None = None,
) -> dict[str, Any]:
    """Stage 4: Execute multilevel, witnessed Lean, strict repo gates, diagnosis, and SHA-256 receipt."""
    root = resolve_repo_root(repo_root)
    val_doc = (
        read_bounded_json(validation_report_json)
        if isinstance(validation_report_json, (str, Path))
        else (dict(validation_report_json) if isinstance(validation_report_json, Mapping) else None)
    )
    gates: list[GateCheckOutcome] = []
    ml_outcome, diag_residuals = run_multilevel_verdict(
        sku=sku,
        validation_doc=val_doc,
        multilevel_manifest_json=multilevel_manifest_json,
        fits_json=fits_json,
        runner=runner,
        threshold_pct=threshold_pct,
    )
    gates.append(ml_outcome)

    failing_cliffs = [
        r for r in diag_residuals
        if float(r.get("absolute_percentage_error", 0.0)) * PERCENT_SCALE >= threshold_pct
    ]
    diagnosis_report: dict[str, Any] | None = None
    if failing_cliffs:
        diagnosis_report = run_tail_cluster_diagnosis(
            diag_residuals,
            repo_root=root,
            threshold_pct=threshold_pct,
            output_path=diagnosis_out,
        )

    if isinstance(lean_bounds_json, (str, Path)):
        raw_lean = read_bounded_json(lean_bounds_json)
        lean_rows = raw_lean.get("bounds", raw_lean) if isinstance(raw_lean, Mapping) else raw_lean
    else:
        lean_rows = list(lean_bounds_json or ())
    lean_ok, lean_checked, lean_violations = check_lean_non_inversion(
        lean_rows,
        require_non_empty=True,
        verify_certificates=True,
    )
    gates.append(
        GateCheckOutcome(
            name="lean_optimality_non_inversion",
            passed=lean_ok,
            status="pass" if lean_ok else "fail",
            detail="0 inversions (witness verified)" if lean_ok else "; ".join(lean_violations),
            metrics=(("cases_checked", len(lean_checked)), ("violations", len(lean_violations))),
        )
    )

    artifacts: dict[str, str] = {}
    if primitives_json is not None:
        prim_doc = (
            read_bounded_json(primitives_json)
            if isinstance(primitives_json, (str, Path))
            else dict(primitives_json)
        )
        try:
            table = parse_primitive_table(prim_doc, verify_envelope=True)
            cal_cnt = sum(1 for cell in table.primitives.values() if cell.status == "calibrated")
            prim_ok = cal_cnt > 0
            gates.append(
                GateCheckOutcome(
                    name="primitives_envelope",
                    passed=prim_ok,
                    status="pass" if prim_ok else "fail",
                    detail=(
                        f"calibrated={cal_cnt}/{len(table.primitives)};envelope={table.envelope_sha256[:SHORT_DIGEST_LEN]}"
                        if prim_ok
                        else "l0_calibration_zero_calibrated_primitives"
                    ),
                    metrics=(("calibrated_count", cal_cnt), ("total_primitives", len(table.primitives))),
                )
            )
            artifacts["primitives_envelope_sha256"] = table.envelope_sha256
        except ModelingError as exc:
            gates.append(
                GateCheckOutcome(
                    name="primitives_envelope",
                    passed=False,
                    status="fail",
                    detail=str(exc),
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
        train_manifest_json=(base_out / "train_manifest.json") if base_out else None,
        plan_out=(base_out / "cliff_plan.json") if base_out else None,
        report_out=(base_out / "validation_512_400k.json") if base_out else None,
        threshold_pct=threshold_pct,
        runner=runner,
        profiler_runner=profiler_runner,
        live_profiler=live_profiler,
        endpoint_alias=endpoint_alias,
        profiler_root=profiler_root,
        profile_override=profile_override,
    )
    stage4 = verify_sku_gates(
        sku=resolved_sku,
        validation_report_json=stage3,
        lean_bounds_json=lean_bounds_json,
        primitives_json=stage2["primitives_payload"],
        repo_root=repo_root,
        threshold_pct=threshold_pct,
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
