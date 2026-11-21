"""Lean 4 OptimalityCertificate non-inversion, stage report HMACs, and receipt sealing.

[Role] Enforce strict ``<= 10.0%`` release threshold ceilings, process-bound
  HMAC sealing/verification for stage reports (blocking fabricated JSON dicts
  in ``verify_sku_gates`` / ``certify_operator``), case-joined Lean 4
  ``OptimalityCertificate`` non-inversion checks (closing witness-free ``L=0``
  and unrelated-case bypasses), and SHA-256 ``OnboardingReceipt`` sealing.
[Invariants] Thresholds ``> 10.0%`` fail closed unconditionally. Every Lean
  witness MUST match a real scored case's ``t_sim_ns``, obey
  ``0 < L*_Lean <= min(T_sim, T_silicon_p99)``, and pass
  ``check_optimality_certificate``.
[Timeouts & Locks] Pure in-memory validation and SHA-256/HMAC hashing; zero
  I/O, locks, or device contexts.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import math
import os
import secrets
from typing import Any, Literal

from mindie_sim.optimization.dual_bound import (
    DagEdge,
    EventDagSpec,
    FarkasCertificate,
    OptimalityCertificate,
    ResourceChoiceProgram,
    check_optimality_certificate,
)
from mindie_sim.schema.common import ModelingError

from .contracts import (
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    OPERATOR_ONBOARDING_RECEIPT_SCHEMA,
    SKU_ONBOARDING_RECEIPT_SCHEMA,
    GateCheckOutcome,
    LeanBoundCheckRow,
    OnboardingReceipt,
    assert_sanitized_payload,
    canonical_json_bytes,
    compute_receipt_sha256,
    utc_now_iso,
)

_REPORT_HMAC_KEY_BYTES = 32
_REPORT_HMAC_ENV_VAR = "MINDIE_ONBOARDING_HMAC_KEY"
_PROCESS_HMAC_KEY = secrets.token_bytes(_REPORT_HMAC_KEY_BYTES)
_REPORT_HMAC_FIELD = "report_hmac"
_MAX_MISSING_CASES_IN_MSG = 8


def _active_hmac_key() -> bytes:
    override = os.environ.get(_REPORT_HMAC_ENV_VAR, "").strip()
    if override:
        return hashlib.sha256(override.encode("utf-8")).digest()
    return _PROCESS_HMAC_KEY


def validate_strict_threshold_pct(threshold_pct: float) -> float:
    """Enforce ``0 < threshold_pct <= DEFAULT_HELDOUT_APE_THRESHOLD_PCT`` (10.0%)."""
    if isinstance(threshold_pct, bool) or not isinstance(threshold_pct, (int, float)):
        raise ModelingError(f"threshold_pct_invalid:{threshold_pct!r}")
    val = float(threshold_pct)
    if not math.isfinite(val) or val <= 0.0:
        raise ModelingError(f"threshold_pct_nonpositive:{val}")
    if val > DEFAULT_HELDOUT_APE_THRESHOLD_PCT:
        raise ModelingError(
            f"threshold_pct_exceeds_strict_release_cap:got={val}:max={DEFAULT_HELDOUT_APE_THRESHOLD_PCT}"
        )
    return val


def sign_stage_report(kind: str, payload: Mapping[str, Any]) -> str:
    """Compute an HMAC-SHA256 attesting that ``payload`` was minted by onboarding."""
    body = {k: v for k, v in payload.items() if k != _REPORT_HMAC_FIELD}
    raw = f"{kind}|".encode("utf-8") + canonical_json_bytes(body)
    return hmac.new(_active_hmac_key(), raw, "sha256").hexdigest()


def verify_stage_report(kind: str, payload: Mapping[str, Any]) -> bool:
    """Return True iff ``payload`` carries an intact in-process HMAC for ``kind``."""
    supplied = payload.get(_REPORT_HMAC_FIELD)
    if not isinstance(supplied, str) or not supplied:
        return False
    expected = sign_stage_report(kind, payload)
    return hmac.compare_digest(supplied, expected)


def _parse_certificate_witness(
    row: Mapping[str, Any]
) -> tuple[ResourceChoiceProgram, EventDagSpec, OptimalityCertificate] | None:
    prog_raw = row.get("program")
    dag_raw = row.get("dag")
    cert_raw = row.get("certificate")
    if prog_raw is None and dag_raw is None and cert_raw is None:
        return None
    if not (isinstance(prog_raw, Mapping) and isinstance(dag_raw, Mapping) and isinstance(cert_raw, Mapping)):
        raise ModelingError("lean_certificate_witness_incomplete")
    prog = ResourceChoiceProgram(
        num_resources=int(prog_raw["num_resources"]),
        global_work=tuple(int(v) for v in prog_raw["global_work"]),
        site_modes=tuple(
            tuple(tuple(int(x) for x in mode) for mode in site)
            for site in prog_raw.get("site_modes", ())
        ),
    )
    edges = tuple(
        DagEdge(int(e["src"]), int(e["dst"]), int(e["wire_ns"]))
        for e in dag_raw.get("edges", ())
    )
    dag = EventDagSpec(
        node_quantum_ns=tuple(int(v) for v in dag_raw["node_quantum_ns"]),
        edges=edges,
    )
    farkas_raw = cert_raw["farkas"]
    farkas = FarkasCertificate(
        scale_d=int(farkas_raw["scale_d"]),
        lambda_num=tuple(int(v) for v in farkas_raw["lambda_num"]),
        mu_num=tuple(int(v) for v in farkas_raw.get("mu_num", ())),
        lower_bound_ns=int(farkas_raw["lower_bound_ns"]),
    )
    cert = OptimalityCertificate(
        farkas=farkas,
        path_edges=tuple(int(v) for v in cert_raw.get("path_edges", ())),
        sink_node_id=int(cert_raw["sink_node_id"]),
        l_star_lean_ns=int(cert_raw["l_star_lean_ns"]),
    )
    return prog, dag, cert


def check_lean_non_inversion(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_non_empty: bool = True,
    verify_certificates: bool = True,
    expected_case_timings: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[bool, tuple[LeanBoundCheckRow, ...], tuple[str, ...]]:
    """Enforce ``0 < L*_Lean <= min(T_sim, T_silicon_p99)`` joined 1:1 to scored cases."""
    if not rows:
        if require_non_empty:
            return False, (), ("lean_bounds_missing",)
        return True, (), ()
    checked: list[LeanBoundCheckRow] = []
    violations: list[str] = []
    seen_case_ids: set[str] = set()
    for idx, row in enumerate(rows):
        cid = str(row.get("case_id") or f"case-{idx}")
        if cid in seen_case_ids:
            violations.append(f"lean_witness_duplicate_case_id:{cid}")
        seen_case_ids.add(cid)
        l_star_raw = row.get("l_star_lean_ns")
        t_sim_raw = row.get("t_sim_ns")
        t_p99_raw = row.get("t_silicon_p99_ns")
        if any(
            v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v))
            for v in (l_star_raw, t_sim_raw, t_p99_raw)
        ):
            violations.append(f"lean_bound_fields_invalid:{cid}")
            checked.append(LeanBoundCheckRow(cid, -1, 0, 0, False, False, True, True))
            continue
        l_star = int(round(float(l_star_raw)))
        t_sim = int(round(float(t_sim_raw)))
        t_p99 = int(round(float(t_p99_raw)))
        cert_checked = False
        cert_valid = l_star > 0 and t_sim > 0 and t_p99 > 0
        try:
            witness = _parse_certificate_witness(row)
        except (ModelingError, KeyError, TypeError, ValueError) as exc:
            cert_valid = False
            witness = None
            violations.append(f"lean_certificate_malformed:{cid}:{exc}")
        if witness is not None:
            cert_checked = True
            prog, dag, cert = witness
            ok_witness = check_optimality_certificate(prog, dag, cert) and cert.l_star_lean_ns == l_star
            if not ok_witness:
                cert_valid = False
                violations.append(f"lean_certificate_invalid:{cid}")
        elif verify_certificates:
            cert_valid = False
            violations.append(f"lean_certificate_missing:{cid}")
        if expected_case_timings is not None:
            if cid not in expected_case_timings:
                cert_valid = False
                violations.append(f"lean_witness_unrelated_case_id:{cid}")
            else:
                exp_sim_ns, exp_meas_ns = expected_case_timings[cid]
                if t_sim != int(exp_sim_ns):
                    cert_valid = False
                    violations.append(
                        f"lean_witness_t_sim_mismatch:{cid}:witness={t_sim}:scored={exp_sim_ns}"
                    )
                if t_p99 < int(exp_meas_ns):
                    cert_valid = False
                    violations.append(
                        f"lean_witness_p99_below_measured:{cid}:p99={t_p99}:measured={exp_meas_ns}"
                    )
        sim_inv = l_star > t_sim
        sil_inv = l_star > t_p99
        if sim_inv:
            violations.append(f"lean_sim_inversion:{cid}:l_star={l_star}>t_sim={t_sim}")
        if sil_inv:
            violations.append(f"lean_silicon_p99_inversion:{cid}:l_star={l_star}>t_p99={t_p99}")
        if t_sim > 0 and l_star <= 0:
            cert_valid = False
            violations.append(f"lean_zero_lower_bound_on_positive_time:{cid}:l_star={l_star}:t_sim={t_sim}")
        if l_star < 0 or t_sim <= 0 or t_p99 <= 0:
            cert_valid = False
            violations.append(f"lean_nonpositive_timing:{cid}")
        checked.append(
            LeanBoundCheckRow(
                case_id=cid, l_star_lean_ns=l_star, t_sim_ns=t_sim, t_silicon_p99_ns=t_p99,
                certificate_checked=cert_checked, certificate_valid=cert_valid,
                sim_inverted=sim_inv, silicon_inverted=sil_inv,
            )
        )
    if expected_case_timings is not None:
        missing_cases = sorted(set(expected_case_timings) - seen_case_ids)
        if missing_cases:
            violations.append(
                f"lean_witness_missing_for_cases:{','.join(missing_cases[:_MAX_MISSING_CASES_IN_MSG])}"
            )
    return not violations, tuple(checked), tuple(violations)


def seal_onboarding_receipt(
    *,
    kind: Literal["sku", "operator"],
    target_id: str,
    sku: str,
    gates: Sequence[GateCheckOutcome],
    artifacts: Mapping[str, str],
    diagnosis_report: Mapping[str, Any] | None = None,
) -> OnboardingReceipt:
    """Build, sanitize, and SHA-256-seal an ``OnboardingReceipt``."""
    schema = (
        SKU_ONBOARDING_RECEIPT_SCHEMA
        if kind == "sku"
        else OPERATOR_ONBOARDING_RECEIPT_SCHEMA
    )
    gate_tuple = tuple(gates)
    all_passed = bool(gate_tuple) and all(g.passed for g in gate_tuple)
    status: Literal["certified", "failed"] = "certified" if all_passed else "failed"
    sorted_artifacts = tuple(sorted((str(k), str(v)) for k, v in artifacts.items()))
    created_utc = utc_now_iso()
    draft = {
        "schema": schema,
        "kind": kind,
        "target_id": str(target_id),
        "sku": str(sku),
        "status": status,
        "ok": all_passed,
        "created_utc": created_utc,
        "gates": [gate.to_dict() for gate in gate_tuple],
        "artifacts": {k: v for k, v in sorted_artifacts},
        "diagnosis_report": dict(diagnosis_report) if diagnosis_report else None,
    }
    assert_sanitized_payload(draft, field_name="onboarding_receipt")
    digest = compute_receipt_sha256(draft)
    return OnboardingReceipt(
        schema=schema,
        kind=kind,
        target_id=str(target_id),
        sku=str(sku),
        status=status,
        ok=all_passed,
        created_utc=created_utc,
        gates=gate_tuple,
        artifacts=sorted_artifacts,
        diagnosis_report=diagnosis_report,
        receipt_sha256=digest,
    )


__all__ = [
    "check_lean_non_inversion",
    "seal_onboarding_receipt",
    "sign_stage_report",
    "validate_strict_threshold_pct",
    "verify_stage_report",
]
