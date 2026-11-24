"""Exact resource-choice and dependent-event lower bounds.

[Role] Own the canonical ``OptimalityCertificate`` and ``certify_lower_bound``
  concepts while delegating all replica-preserving demand assembly, rational
  Farkas dual solving, and checked critical-path verification to
  ``mindie_sim.rewrite.dual_cert``.
[Invariants] One replica-preserving normalization covers all placements keyed
  strictly by ``(replica, op_id)``. Checkers mirror Lean Int/Nat and never
  synthesize phantom cross-replica chains or zero-rate ``sort_topk`` modes.
[Timeouts & Locks] Finite Bland pivots and DAG traversal; no locks or external
  solvers.
[Side-Effect Policy] Read-only analysis with no runtime replay or device access.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.rewrite.dual_cert import (
    BINARY64_ULP_SHIFT,
    C11_ULP_GUARD_FACTOR,
    ENDPOINT_ROUND_GUARD_NS,
    NS_PER_MS,
    NS_PER_SECOND,
    DagEdge,
    DualBoundCertificate,
    DualSubgradientReport,
    EpochAccount,
    EventDagSpec,
    FarkasCertificate,
    ResourceChoiceProgram,
    _Demands,
    _atomic_ops,
    _check_region_summary_bindings,
    _engine_modes,
    _machine,
    _memory_demands,
    _op_contract_key,
    _scenario_rank_ops,
    _semantic_ops,
    _unique_ops,
    build_dual_inputs_from_scenario,
    check_farkas_dual,
    check_optimality_certificate,
    eval_checked_path,
    inward_floor_ns,
    solve_checked_critical_path,
    solve_dual_lower_bound,
    solve_farkas_dual,
    verify_epoch_trace,
)

__all__ = (
    "ResourceChoiceProgram",
    "FarkasCertificate",
    "EpochAccount",
    "DagEdge",
    "EventDagSpec",
    "OptimalityCertificate",
    "DualSubgradientReport",
    "BINARY64_ULP_SHIFT",
    "C11_ULP_GUARD_FACTOR",
    "ENDPOINT_ROUND_GUARD_NS",
    "NS_PER_MS",
    "NS_PER_SECOND",
    "check_farkas_dual",
    "eval_checked_path",
    "check_optimality_certificate",
    "verify_epoch_trace",
    "inward_floor_ns",
    "solve_farkas_dual",
    "solve_checked_critical_path",
    "certify_lower_bound",
    "build_dual_inputs_from_scenario",
)


@dataclass(frozen=True, slots=True)
class OptimalityCertificate(DualBoundCertificate):
    """Canonical optimality certificate pairing Farkas dual and checked critical path."""


def certify_lower_bound(
    prog: ResourceChoiceProgram, dag: EventDagSpec,
) -> tuple[OptimalityCertificate, DualSubgradientReport]:
    """Solve and independently check a formal primal-dual lower-bound certificate."""
    cert, report = solve_dual_lower_bound(prog, dag, cert_cls=OptimalityCertificate)
    return cert, report  # type: ignore[return-value]
