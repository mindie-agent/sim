"""Lean 4 OptimalityCertificate non-inversion checks and SHA-256 receipt sealing.

[Role] Enforce ``0 < L*_Lean <= min(T_sim, T_silicon_p99)`` with mandatory
  verified ``OptimalityCertificate`` witnesses (closing witness-free ``L=0``
  bypasses) and build/seal sanitized SHA-256 ``OnboardingReceipt`` records.
[Invariants] Witness-free rows and ``L*_Lean == 0`` on positive simulated time
  fail closed unconditionally. Every sealed receipt passes sanitization before
  digest calculation.
[Timeouts & Locks] Pure in-memory validation and SHA-256 hashing; zero I/O,
  locks, or device contexts.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
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
    OPERATOR_ONBOARDING_RECEIPT_SCHEMA,
    SKU_ONBOARDING_RECEIPT_SCHEMA,
    GateCheckOutcome,
    LeanBoundCheckRow,
    OnboardingReceipt,
    assert_sanitized_payload,
    compute_receipt_sha256,
    utc_now_iso,
)


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
) -> tuple[bool, tuple[LeanBoundCheckRow, ...], tuple[str, ...]]:
    """Enforce ``0 < L*_Lean <= min(T_sim, T_silicon_p99)`` with mandatory witness verification."""
    if not rows:
        if require_non_empty:
            return False, (), ("lean_bounds_missing",)
        return True, (), ()
    checked: list[LeanBoundCheckRow] = []
    violations: list[str] = []
    for idx, row in enumerate(rows):
        cid = str(row.get("case_id") or f"case-{idx}")
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


__all__ = ["check_lean_non_inversion", "seal_onboarding_receipt"]
