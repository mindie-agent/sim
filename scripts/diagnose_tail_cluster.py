#!/usr/bin/env python3
"""Automatic >=10% tail-cluster root-cause diagnosis and conflict detector.

[Role]
  Pair multilevel gate residuals with acquisition lineage, Engine receipts, and
  optional reference runs; cluster strict >= threshold_pct failures by frozen
  cohort and regime dimensions; run the 5-stage diagnostic ladder; and detect
  cross-regime physical parameter conflicts via Jacobian feasible intervals.
[Invariants]
  * Strict < threshold_pct acceptance (10.0% triggers diagnosis); single-point
    tails and P90 exceedances are never hidden behind sample-count thresholds.
  * Pairing requires (case_id, acquisition_root, boundary_kind); persistent
    failures across independent repeats or empty cross-regime parameter
    intersections escalate immediately as blocking.
  * Strictly read-only over hardware profiles, captures, and frozen fits; never
    mutates profiles or suggests per-model constants, split edits, or floors.
[Timeouts & Locks]
  Streaming JSON/JSONL loading; bounded per-cluster aggregation; zero locks or
  device contexts.
[Side-Effect Policy]
  Reads declared inputs and writes JSON to stdout or explicit --output only.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Literal

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from mindie_sim.observations.multilevel_gate import (
    DEFAULT_THRESHOLD_PCT,
    PERCENT_SCALE,
    TAIL_QUANTILE_P90,
    stream_jsonl,
    validate_provenance_text,
)
from mindie_sim.schema.common import ModelingError

REGIME_DIMENSIONS = (
    "dtype_layout", "shape", "warmth", "parallel_members",
    "message_band", "stream", "stack_revision",
)
SILICON_MECHANISM_KEYS = ("STARS", "D2H", "HCCL", "SIMT", "subtile_wave")
JACOBIAN_ZERO_TOL = 1e-12
PARITY_DIFF_TOL_NS = 1
EXIT_OK = 0
EXIT_ESCALATED = 1


@dataclass(frozen=True, slots=True)
class StageResult:
    stage_number: int
    stage_name: str
    status: Literal["clean", "issue_found", "action_required"]
    findings: tuple[str, ...]
    first_divergence: str
    owner_module: str
    next_check: str


@dataclass(frozen=True, slots=True)
class TailClusterReport:
    cluster_id: str
    cohort: dict[str, str]
    regime_dimensions: dict[str, str]
    n_independent_roots: int
    n_rows: int
    error_direction: Literal["overpredict", "underpredict", "mixed"]
    error_distribution: dict[str, float]
    root_captures: tuple[str, ...]
    first_abi_diff: str
    saturated_physical_row: str
    exposed_wait_ns: int
    counterfactual_replay_summary: str
    owner_module: str
    next_falsifiable_check: str
    trigger_reasons: tuple[str, ...]
    blocking: bool
    escalation_reasons: tuple[str, ...]
    stages: tuple[StageResult, ...]


@dataclass(frozen=True, slots=True)
class ParameterBound:
    parameter_name: str
    current_value: float
    theta_min: float
    theta_max: float
    parameter_source: str


@dataclass(frozen=True, slots=True)
class CohortSensitivity:
    cohort_id: str
    measured_ns: float
    predicted_ns: float
    sensitivity_ns_per_unit: float
    parameter_source: str = ""


@dataclass(frozen=True, slots=True)
class CohortFeasibleInterval:
    cohort_id: str
    lower_delta: float | None
    upper_delta: float | None
    feasible_within_bounds: bool


@dataclass(frozen=True, slots=True)
class ParameterConflict:
    parameter_name: str
    parameter_source: str
    allowed_bounds: tuple[float, float]
    conflicting_cohorts: tuple[str, ...]
    cohort_intervals: tuple[CohortFeasibleInterval, ...]
    conflict_reason: str
    read_only_notice: str = "diagnostic_only_profile_unmodified"


def _load_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return list(stream_jsonl(path))
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if isinstance(doc, list):
        return [dict(item) for item in doc if isinstance(item, Mapping)]
    if isinstance(doc, dict):
        for key in ("residuals", "records", "receipts", "lineage", "cases", "items"):
            if isinstance(doc.get(key), list):
                return [dict(item) for item in doc[key] if isinstance(item, Mapping)]
        return [doc]
    return []


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    boundary = row.get("boundary_kind") or (
        row["boundary"].get("boundary_kind") if isinstance(row.get("boundary"), Mapping) else ""
    )
    return (str(row.get("case_id", "")), str(row.get("acquisition_root", "")), str(boundary))


def _cohort_dict(row: Mapping[str, Any]) -> dict[str, str]:
    raw = row.get("cohort")
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in sorted(raw.items())}
    return {
        "level": str(row.get("level", "")),
        "cohort_name": str(row.get("cohort_name", "unknown")),
        "sku": str(row.get("sku", "unknown")),
        "phase": str(row.get("phase", "unknown")),
        "mode": str(row.get("mode", "unknown")),
        "shape_regime": str(row.get("shape_regime", "unknown")),
        "parallel_topology": str(row.get("parallel_topology", "unknown")),
    }


def _regime_dict(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, str]:
    dims: dict[str, str] = {}
    for src in (receipt.get("regime_dims"), row.get("regime_dims"), receipt, row):
        if not isinstance(src, Mapping):
            continue
        for dim in REGIME_DIMENSIONS:
            if dim not in dims and src.get(dim) not in (None, ""):
                dims[dim] = str(src[dim])
    for dim in REGIME_DIMENSIONS:
        dims.setdefault(dim, "unspecified")
    return dims


def pair_records(
    residuals: Sequence[Mapping[str, Any]],
    lineage: Sequence[Mapping[str, Any]] = (),
    engine_receipts: Sequence[Mapping[str, Any]] = (),
    reference_run: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Pair residuals with lineage, Engine receipts, and reference runs by triple key."""
    lineage_map = {_pair_key(r): dict(r) for r in lineage}
    receipt_map = {_pair_key(r): dict(r) for r in engine_receipts}
    receipt_by_id = {str(r.get("engine_receipt", "")): dict(r) for r in engine_receipts if r.get("engine_receipt")}
    ref_map = {_pair_key(r): dict(r) for r in reference_run}
    ref_by_case = {str(r.get("case_id", "")): dict(r) for r in reference_run if r.get("case_id")}

    paired: list[dict[str, Any]] = []
    for res in residuals:
        key = _pair_key(res)
        rcpt_id = str(res.get("engine_receipt", ""))
        receipt = receipt_map.get(key) or receipt_by_id.get(rcpt_id) or {}
        lin = lineage_map.get(key) or {}
        ref = ref_map.get(key) or ref_by_case.get(key[0]) or {}
        paired.append({
            "pair_key": key,
            "residual": dict(res),
            "lineage": lin,
            "receipt": receipt,
            "reference": ref,
            "cohort": _cohort_dict(res),
            "regime_dims": _regime_dict(res, receipt),
        })
    return paired


def _stage1_capture_purity(rows: Sequence[Mapping[str, Any]]) -> StageResult:
    findings: list[str] = []
    for entry in rows:
        res, lin, rcpt = entry["residual"], entry["lineage"], entry["receipt"]
        cid = str(res.get("case_id", "unknown"))
        if not lin:
            findings.append(f"missing_lineage_pair:{cid}")
        if not rcpt:
            findings.append(f"missing_engine_receipt_pair:{cid}")
        for flag in (
            "sha_mismatch", "boundary_mismatch", "unit_mismatch", "clock_domain_mismatch",
            "warmup_contaminated", "cold_eviction", "dvfs_throttle", "stream_saturation_mismatch",
            "duplicate_or_filtered_rows", "host_timer_union_mismatch", "shape_or_quant_mismatch",
            "rank_table_mismatch",
        ):
            if lin.get(flag) or rcpt.get(flag) or res.get(flag):
                findings.append(f"{flag}:{cid}")
        if lin.get("capture_status") not in (None, "", "ok", "measured"):
            findings.append(f"capture_status_invalid:{cid}:{lin.get('capture_status')}")
        if rcpt.get("external_offset_ns", 0) != 0:
            findings.append(f"external_offset_detected:{cid}:{rcpt.get('external_offset_ns')}")
        for text_val in (res.get("engine_receipt", ""), lin.get("capture_uri", "")):
            if text_val:
                try:
                    validate_provenance_text(str(text_val), field_name="stage1")
                except ModelingError as exc:
                    findings.append(f"{exc.reason}:{cid}")
    first = findings[0] if findings else "none"
    return StageResult(
        stage_number=1, stage_name="capture_purity",
        status="issue_found" if findings else "clean",
        findings=tuple(findings), first_divergence=first,
        owner_module="observations.importer" if findings else "none",
        next_check=(
            f"Fix capture boundary/importer for {first} or acquire independent recapture."
            if findings else "Capture purity verified; proceed to Stage 2 ABI diff."
        ),
    )


def _stage2_upstream_abi(rows: Sequence[Mapping[str, Any]]) -> tuple[StageResult, str]:
    findings: list[str] = []
    first_abi = "none"
    for entry in rows:
        res, rcpt = entry["residual"], entry["receipt"]
        cid = str(res.get("case_id", "unknown"))
        exp_ref, obs_ref = rcpt.get("expected_source_ref"), rcpt.get("observed_source_ref")
        if isinstance(exp_ref, Mapping) and isinstance(obs_ref, Mapping) and dict(exp_ref) != dict(obs_ref):
            findings.append(f"source_ref_mismatch:{cid}")
        exp_abi, obs_abi = rcpt.get("expected_abi"), rcpt.get("observed_abi")
        if exp_abi and obs_abi and str(exp_abi) != str(obs_abi):
            diff = f"caller={rcpt.get('caller_node', cid)}:expected={exp_abi}:observed={obs_abi}"
            if first_abi == "none":
                first_abi = diff
            findings.append(f"abi_mismatch:{diff}")
        for field in (
            "kernel_count", "split_sizes", "guard_signature", "tensor_version",
            "layout", "fusion_members", "graph_capture_mode", "dual_stream_events",
        ):
            exp_val, obs_val = rcpt.get(f"expected_{field}"), rcpt.get(f"observed_{field}")
            if exp_val is not None and obs_val is not None and exp_val != obs_val:
                findings.append(f"{field}_mismatch:{cid}:expected={exp_val}:observed={obs_val}")
    first = findings[0] if findings else "none"
    return StageResult(
        stage_number=2, stage_name="upstream_abi_and_caller",
        status="issue_found" if findings else "clean",
        findings=tuple(findings), first_divergence=first,
        owner_module="prediction.operators" if findings else "none",
        next_check=(
            f"Align upstream stack/caller ABI at {first} with minimal shape reproducer."
            if findings else "Upstream ABI and caller contract aligned; proceed to Stage 3."
        ),
    ), first_abi


def _stage3_engine_contention(rows: Sequence[Mapping[str, Any]]) -> tuple[StageResult, str, str]:
    findings: list[str] = []
    saturated_row = "none"
    ablation_notes: list[str] = []
    for entry in rows:
        res, rcpt = entry["residual"], entry["receipt"]
        cid = str(res.get("case_id", "unknown"))
        py_ns, c_ns = rcpt.get("python_ns"), rcpt.get("c_ns")
        if isinstance(py_ns, (int, float)) and isinstance(c_ns, (int, float)):
            if abs(float(py_ns) - float(c_ns)) > PARITY_DIFF_TOL_NS:
                epoch = rcpt.get("first_parity_diff_epoch", "unknown")
                findings.append(f"python_c_parity_mismatch:{cid}:epoch={epoch}:py_ns={py_ns}:c_ns={c_ns}")
        if rcpt.get("conservation_diff_bytes", 0) != 0:
            findings.append(f"conservation_diff:{cid}:bytes={rcpt['conservation_diff_bytes']}")
        for flag in (
            "lad_imbalance", "hbm_dual_row_contention", "l2_knee_miss", "vector_issue_bound",
            "credit_starvation", "rail_coset_conflict", "ring_generation_stall", "periodic_skip_mismatch",
        ):
            if rcpt.get(flag):
                findings.append(f"{flag}:{cid}")
        if saturated_row == "none" and rcpt.get("saturated_physical_row"):
            saturated_row = str(rcpt["saturated_physical_row"])
        if rcpt.get("counterfactual_ablation"):
            ablation_notes.append(f"{cid}:{rcpt['counterfactual_ablation']}")
    first = findings[0] if findings else "none"
    summary = "; ".join(ablation_notes) if ablation_notes else "no_ablation_recorded"
    return StageResult(
        stage_number=3, stage_name="engine_and_resource_contention",
        status="issue_found" if findings else "clean",
        findings=tuple(findings), first_divergence=first,
        owner_module="prediction.runtime.engine" if findings else "none",
        next_check=(
            f"Run per-epoch Python/C diff and single-mechanism ablation on {first}."
            if findings else "Engine parity and resource contention clean; proceed to Stage 4."
        ),
    ), saturated_row, summary


def _stage4_silicon_mechanism(rows: Sequence[Mapping[str, Any]]) -> tuple[StageResult, int]:
    findings: list[str] = []
    max_wait_ns = 0
    for entry in rows:
        res, rcpt = entry["residual"], entry["receipt"]
        cid = str(res.get("case_id", "unknown"))
        wait_ns = int(rcpt.get("exposed_wait_ns") or rcpt.get("bare_wait_ns") or 0)
        max_wait_ns = max(max_wait_ns, wait_ns)
        if wait_ns > 0:
            findings.append(f"exposed_wait:{cid}:wait_ns={wait_ns}")
        mech_signals = rcpt.get("silicon_mechanisms") or rcpt.get("missing_barriers") or ()
        if isinstance(mech_signals, Mapping):
            active = [k for k, v in mech_signals.items() if v]
        elif isinstance(mech_signals, Sequence) and not isinstance(mech_signals, (str, bytes)):
            active = [str(x) for x in mech_signals]
        else:
            active = []
        for mech in SILICON_MECHANISM_KEYS:
            if rcpt.get(f"suspect_{mech.lower()}") or mech in active or mech.lower() in {a.lower() for a in active}:
                findings.append(f"missing_mechanism:{mech}:{cid}")
    first = findings[0] if findings else "none"
    return StageResult(
        stage_number=4, stage_name="missing_barrier_or_silicon_mechanism",
        status="issue_found" if findings else "clean",
        findings=tuple(findings), first_divergence=first,
        owner_module="prediction.runtime.barriers" if findings else "none",
        next_check=(
            f"Model physical mechanism for {first} from independent primitive train capture and verify unseen counterexamples."
            if findings else "No unmodeled STARS/D2H/HCCL/SIMT/subtile_wave barrier flagged."
        ),
    ), max_wait_ns


def _stage5_closure_checklist(blocking: bool) -> StageResult:
    checks = (
        "rerun_l1_l2_l3_same_frozen_parameters",
        "reverify_previously_accurate_regimes_zero_regression",
        "reverify_independent_catalog_conservation_and_cross_repo_contracts",
        "reserve_fresh_unseen_acquisitions_for_final_blind_acceptance",
        "preserve_prior_failure_evidence_with_new_receipts_and_source_sha256",
        "forbid_per_model_constants_split_edits_external_offsets_or_step_floors",
    )
    return StageResult(
        stage_number=5, stage_name="closure_and_unblocking",
        status="action_required" if blocking else "clean",
        findings=checks, first_divergence="none",
        owner_module="observations.multilevel_gate",
        next_check="Complete all 6 closure checks with fresh unseen blind acquisitions before closing ticket.",
    )


def run_five_stage_ladder(
    paired_rows: Sequence[Mapping[str, Any]], *, blocking: bool = False
) -> tuple[tuple[StageResult, ...], str, str, int, str]:
    """Execute Stages 1..5 in exact Vol 2 §3.2 order for one tail cluster."""
    s1 = _stage1_capture_purity(paired_rows)
    s2, first_abi = _stage2_upstream_abi(paired_rows)
    s3, sat_row, cf_summary = _stage3_engine_contention(paired_rows)
    s4, wait_ns = _stage4_silicon_mechanism(paired_rows)
    s5 = _stage5_closure_checklist(blocking or any(s.status == "issue_found" for s in (s1, s2, s3, s4)))
    return (s1, s2, s3, s4, s5), first_abi, sat_row, wait_ns, cf_summary


def detect_parameter_conflicts(
    parameter_bounds: Sequence[ParameterBound],
    sensitivities_by_param: Mapping[str, Sequence[CohortSensitivity]],
    *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> list[ParameterConflict]:
    """Detect empty cross-regime feasible intervals for frozen physical parameters.

    Strictly read-only diagnostic: reports conflicting cohorts and parameter
    sources; never modifies profiles or fits.
    """
    tau = threshold_pct / PERCENT_SCALE
    conflicts: list[ParameterConflict] = []
    for bound in parameter_bounds:
        cohorts = sensitivities_by_param.get(bound.parameter_name, ())
        if not cohorts:
            continue
        d_min = bound.theta_min - bound.current_value
        d_max = bound.theta_max - bound.current_value
        if d_min > d_max:
            raise ModelingError(f"parameter_bounds_inverted:{bound.parameter_name}")

        inter_lo, inter_lo_inc, lo_owner = d_min, True, "training_lower_bound"
        inter_hi, inter_hi_inc, hi_owner = d_max, True, "training_upper_bound"
        cohort_intervals: list[CohortFeasibleInterval] = []
        infeasible_ids: list[str] = []

        for item in cohorts:
            if item.measured_ns <= 0:
                continue
            eps = tau * item.measured_ns
            req = item.measured_ns - item.predicted_ns
            jac = item.sensitivity_ns_per_unit
            if abs(jac) <= JACOBIAN_ZERO_TOL:
                ok = abs(req) < eps
                cohort_intervals.append(CohortFeasibleInterval(
                    item.cohort_id, d_min if ok else None, d_max if ok else None, ok,
                ))
                if not ok:
                    infeasible_ids.append(item.cohort_id)
                continue
            a = (req - eps) / jac
            b = (req + eps) / jac
            c_lo, c_hi = (a, b) if a <= b else (b, a)
            c_ok = c_lo < c_hi and c_lo < d_max and d_min < c_hi
            cohort_intervals.append(CohortFeasibleInterval(item.cohort_id, c_lo, c_hi, c_ok))
            if not c_ok:
                infeasible_ids.append(item.cohort_id)
            if c_lo > inter_lo or (c_lo == inter_lo and inter_lo_inc):
                inter_lo, inter_lo_inc, lo_owner = c_lo, False, item.cohort_id
            if c_hi < inter_hi or (c_hi == inter_hi and inter_hi_inc):
                inter_hi, inter_hi_inc, hi_owner = c_hi, False, item.cohort_id

        empty_intersection = (
            bool(infeasible_ids)
            or inter_lo > inter_hi
            or (inter_lo == inter_hi and not (inter_lo_inc and inter_hi_inc))
        )
        if empty_intersection:
            culprits = list(dict.fromkeys([
                *infeasible_ids,
                *(x for x in (lo_owner, hi_owner) if not x.startswith("training_")),
            ]))
            reason = (
                "infeasible_within_training_bounds"
                if infeasible_ids and len(culprits) == 1
                else "empty_cross_regime_intersection"
            )
            conflicts.append(ParameterConflict(
                parameter_name=bound.parameter_name,
                parameter_source=bound.parameter_source,
                allowed_bounds=(bound.theta_min, bound.theta_max),
                conflicting_cohorts=tuple(culprits),
                cohort_intervals=tuple(cohort_intervals),
                conflict_reason=reason,
            ))
    return conflicts


def _error_direction(signed_pcts: Sequence[float]) -> Literal["overpredict", "underpredict", "mixed"]:
    if all(v > 0 for v in signed_pcts):
        return "overpredict"
    if all(v < 0 for v in signed_pcts):
        return "underpredict"
    return "mixed"


def diagnose_tail_clusters(
    residuals: Sequence[Mapping[str, Any]],
    *,
    lineage: Sequence[Mapping[str, Any]] = (),
    engine_receipts: Sequence[Mapping[str, Any]] = (),
    reference_run: Sequence[Mapping[str, Any]] = (),
    parameter_bounds: Sequence[ParameterBound] = (),
    sensitivities_by_param: Mapping[str, Sequence[CohortSensitivity]] = (),
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> dict[str, Any]:
    """Group failing/tail residuals, run 5-stage diagnosis, and check conflicts."""
    paired = pair_records(residuals, lineage, engine_receipts, reference_run)
    conflicts = detect_parameter_conflicts(parameter_bounds, sensitivities_by_param, threshold_pct=threshold_pct)
    conflict_cohorts = {cid for c in conflicts for cid in c.conflicting_cohorts}

    grouped: dict[tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}
    for entry in paired:
        res = entry["residual"]
        if res.get("status", "ok") != "ok":
            continue
        ape = res.get("absolute_percentage_error")
        if ape is None or not math.isfinite(float(ape)):
            continue
        key = (tuple(sorted(entry["cohort"].items())), tuple(sorted(entry["regime_dims"].items())))
        grouped.setdefault(key, []).append(entry)

    clusters: list[TailClusterReport] = []
    for idx, ((cohort_items, regime_items), entries) in enumerate(sorted(grouped.items()), start=1):
        apes_pct = sorted(float(e["residual"]["absolute_percentage_error"]) * PERCENT_SCALE for e in entries)
        signed_pct = [float(e["residual"]["signed_error"]) * PERCENT_SCALE for e in entries]
        mape_pct = float(statistics.fmean(apes_pct))
        medape_pct = float(statistics.median(apes_pct))
        p90_idx = min(len(apes_pct) - 1, max(0, math.ceil(TAIL_QUANTILE_P90 * len(apes_pct)) - 1))
        p90_pct = float(apes_pct[p90_idx])
        max_pct = float(apes_pct[-1])
        cohort_map, regime_map = dict(cohort_items), dict(regime_items)
        cohort_label = "|".join(cohort_map.get(k, "") for k in (
            "level", "cohort_name", "sku", "phase", "mode", "shape_regime", "parallel_topology",
        ))

        triggers: list[str] = []
        if mape_pct >= threshold_pct:
            triggers.append("cohort_mape_exceeded")
        if medape_pct >= threshold_pct:
            triggers.append("cohort_medape_exceeded")
        if p90_pct >= threshold_pct:
            triggers.append("p90_tail_exceeded")
        if max_pct >= threshold_pct:
            triggers.append("single_boundary_tail_exceeded")
        if any(e["residual"].get("key_regime") and float(e["residual"]["absolute_percentage_error"]) * PERCENT_SCALE >= threshold_pct for e in entries):
            triggers.append("key_regime_exceeded")
        if cohort_label in conflict_cohorts or cohort_map.get("cohort_name") in conflict_cohorts:
            triggers.append("cross_regime_parameter_conflict")
        if not triggers:
            continue

        persistent = any(
            e["reference"]
            and e["reference"].get("absolute_percentage_error") is not None
            and float(e["reference"]["absolute_percentage_error"]) * PERCENT_SCALE >= threshold_pct
            for e in entries
        )
        escalations: list[str] = []
        if persistent:
            escalations.append("persistent_tail_across_independent_repeat")
        if "cohort_mape_exceeded" in triggers or "cohort_medape_exceeded" in triggers:
            escalations.append("strict_sub_threshold_cohort_gate_failed")
        if "cross_regime_parameter_conflict" in triggers:
            escalations.append("cross_regime_parameter_conflict")
        blocking = bool(escalations)

        stages, first_abi, sat_row, wait_ns, cf_summary = run_five_stage_ladder(entries, blocking=blocking)
        first_active = next((s for s in stages[:4] if s.status == "issue_found"), stages[4])
        roots = tuple(sorted({str(e["residual"].get("acquisition_root", "")) for e in entries if e["residual"].get("acquisition_root")}))
        clusters.append(TailClusterReport(
            cluster_id=f"cluster-{idx:03d}",
            cohort=cohort_map,
            regime_dimensions=regime_map,
            n_independent_roots=len(roots),
            n_rows=len(entries),
            error_direction=_error_direction(signed_pct),
            error_distribution={
                "mape_pct": round(mape_pct, 4),
                "medape_pct": round(medape_pct, 4),
                "median_signed_pct": round(float(statistics.median(signed_pct)), 4),
                "p90_ape_pct": round(p90_pct, 4),
                "max_ape_pct": round(max_pct, 4),
            },
            root_captures=roots,
            first_abi_diff=first_abi,
            saturated_physical_row=sat_row,
            exposed_wait_ns=wait_ns,
            counterfactual_replay_summary=cf_summary,
            owner_module=first_active.owner_module,
            next_falsifiable_check=first_active.next_check,
            trigger_reasons=tuple(triggers),
            blocking=blocking,
            escalation_reasons=tuple(escalations),
            stages=stages,
        ))

    any_blocking = any(c.blocking for c in clusters) or bool(conflicts)
    return {
        "schema": "mindie-sim.tail-cluster-diagnosis/1",
        "threshold_pct": threshold_pct,
        "status": "blocking_escalation" if any_blocking else ("tail_warning" if clusters else "clean"),
        "cluster_count": len(clusters),
        "conflict_count": len(conflicts),
        "clusters": [asdict(c) for c in clusters],
        "parameter_conflicts": [asdict(c) for c in conflicts],
        "read_only_guarantee": "profiles_and_fits_unmodified",
    }


def _load_sensitivities(path: Path | None) -> tuple[list[ParameterBound], dict[str, list[CohortSensitivity]]]:
    if path is None:
        return [], {}
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    bounds = [
        ParameterBound(
            parameter_name=str(b["parameter_name"]),
            current_value=float(b["current_value"]),
            theta_min=float(b["theta_min"]),
            theta_max=float(b["theta_max"]),
            parameter_source=str(b.get("parameter_source", "")),
        )
        for b in doc.get("parameter_bounds", [])
    ]
    sens_map: dict[str, list[CohortSensitivity]] = {}
    for pname, items in (doc.get("sensitivities") or {}).items():
        sens_map[str(pname)] = [
            CohortSensitivity(
                cohort_id=str(s["cohort_id"]),
                measured_ns=float(s["measured_ns"]),
                predicted_ns=float(s["predicted_ns"]),
                sensitivity_ns_per_unit=float(s["sensitivity_ns_per_unit"]),
                parameter_source=str(s.get("parameter_source", "")),
            )
            for s in items
        ]
    return bounds, sens_map


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, required=True, help="Residuals JSON or JSONL file")
    parser.add_argument("--lineage", type=Path, default=None, help="Lineage index JSON or JSONL file")
    parser.add_argument("--engine-receipts", type=Path, default=None, help="Engine receipts JSON or JSONL file")
    parser.add_argument("--reference-run", type=Path, default=None, help="Reference baseline residuals JSON/JSONL")
    parser.add_argument("--sensitivities", type=Path, default=None, help="Optional cross-regime Jacobian sensitivities JSON")
    parser.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT, help="Strict < threshold percentage")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    bounds, sens_map = _load_sensitivities(args.sensitivities)
    report = diagnose_tail_clusters(
        _load_records(args.residuals),
        lineage=_load_records(args.lineage),
        engine_receipts=_load_records(args.engine_receipts),
        reference_run=_load_records(args.reference_run),
        parameter_bounds=bounds,
        sensitivities_by_param=sens_map,
        threshold_pct=float(args.threshold_pct),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered.encode("utf-8"))
    sys.stdout.write(rendered)
    return EXIT_ESCALATED if report["status"] == "blocking_escalation" else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
