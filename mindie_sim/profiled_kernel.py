"""Exact captured-kernel duration provider for the existing operator engine.

[Role]
  Reuse profiler inventory observations as explicit, scoped kernel cost plans.
[Invariants]
  Match an exact variant and caller-declared lowering contract; no shape, SKU,
  recipe or environment wildcard. Keep profiled durations separate from warm
  microbenchmarks, heldout fitting and simulated CAModel inputs. The captured
  scope is a property of the binding, never of the surrounding run protocol.
  The whole captured duration is a completion floor for the parent service, not
  a second analytic service interval and not a claim of zero occupancy: the
  stage still charges the operator's *declared* work, which is a proxy the
  capture cannot confirm. A capture carries no occupancy, so an operator that
  declares no work is refused rather than admitted as a free shared-stage task.
[Timeouts & Locks]
  Read-only SQLite lookups use a 5s busy budget. Bounded 1024-entry result cache;
  no full event-table load, subprocess, device or network work.
[Side-Effect Policy]
  Reads only the supplied inventory. No writes or measurement submission.
"""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from .a3_types import A3InputError
from .operator_plan import (
    PROFILED_KERNEL_BACKEND, ExecutionStage, KernelEstimate, KernelPlan,
)

#: The only captured scope this provider prices.  It is declared per binding
#: because one run mixes captured kernels with operators priced by the run's
#: own measurement protocol; a run-wide scope would disqualify those.
CAPTURED_TIMING_SCOPE = "profiled_device_kernel"

#: Declared operator work -> the engine's shared stage resources.  ``link`` is
#: absent on purpose: communication stays with the existing topology model.
DECLARED_WORK_RESOURCES = (
    ("cube", "flops"), ("vector", "vector_ops"), ("hbm", "hbm_bytes"),
    ("mte2", "mte2_bytes"), ("mte3", "mte3_bytes"),
)


class ProfiledKernelProvider:
    def __init__(self, config: Mapping[str, Any]):
        self.db = Path(config["db"]).resolve()
        self.environment_key = str(config["environment_key"])
        self.bindings: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for binding in config.get("bindings", ()):
            key = (binding["family"], binding["recipe"], binding["phase"],
                   tuple(binding["launch_shape"]), binding["dtype"], binding["layout"])
            self.bindings.setdefault(key, []).append(binding)
        self._cache: dict[str, dict[str, Any]] = {}

    def lookup(self, variant_id: str) -> dict[str, Any]:
        if variant_id not in self._cache:
            # ``mode=ro`` without ``immutable``: the inventory is an incremental
            # WAL database, and ``immutable=1`` would silently read the pre-WAL
            # snapshot (a superseded median, or a missing freshly ingested
            # variant) instead of failing.
            with closing(sqlite3.connect(self.db.as_uri() + "?mode=ro", uri=True, timeout=5)) as connection:
                connection.execute("PRAGMA busy_timeout=5000")
                row = connection.execute(
                    "SELECT v.signature,v.context,v.eligible,v.reason,c.calls,"
                    "c.median_ns,c.min_ns,c.max_ns,v.capture "
                    "FROM variants v JOIN costs c ON v.id=c.variant WHERE v.id=?",
                    (bytes.fromhex(variant_id),),
                ).fetchone()
            if row is None:
                raise A3InputError(f"profile_variant_missing:{variant_id}")
            result = {"signature": json.loads(row[0]), "context": json.loads(row[1]),
                      "eligible": bool(row[2]), "reason": row[3], "calls": row[4],
                      "median_ns": row[5], "min_ns": row[6], "max_ns": row[7],
                      "capture": row[8]}
            if len(self._cache) >= 1024:
                self._cache.clear()
            self._cache[variant_id] = result
        return self._cache[variant_id]

    def estimate(self, op, machine, context):
        # Bindings are exact lowering contracts reviewed against the actual
        # profile; an equal first-input shape alone does not identify a kernel.
        selected = dict(op.metadata).get("profile_variant")
        key = (op.family, op.recipe, op.phase, op.launch_shape, op.dtype, op.layout)
        matches = [b for b in self.bindings.get(key, ())
                   if selected is None or selected == b["variant_id"]]
        if len(matches) != 1:
            return KernelEstimate(unsupported_reason="profile_exact_binding_missing_or_ambiguous")
        binding = matches[0]
        if binding.get("timing_scope") != CAPTURED_TIMING_SCOPE:
            return KernelEstimate(unsupported_reason="profile_binding_timing_scope_unsupported")
        observation = self.lookup(binding["variant_id"])
        evidence = observation["context"]
        if not observation["eligible"]:
            return KernelEstimate(unsupported_reason="profile_ineligible:" + observation["reason"])
        if evidence.get("hardware_key") != machine.hardware_key or evidence.get("environment_key") != self.environment_key:
            return KernelEstimate(unsupported_reason="profile_hardware_or_environment_mismatch")
        demands = tuple(
            (resource, float(getattr(op, field)))
            for resource, field in DECLARED_WORK_RESOURCES
            if float(getattr(op, field, 0.0) or 0.0) > 0
        )
        if not demands:
            # The capture timed the kernel but recorded no occupancy.  Pricing
            # it as a floor with no demands would hand the shared stage a task
            # that every concurrent operator overlaps for free.
            return KernelEstimate(unsupported_reason="profile_occupancy_undeclared_for_shared_stage")
        provenance = {"source_class": "observed", "inventory": str(self.db),
                      "variant_id": binding["variant_id"], "signature": observation["signature"],
                      "capture": observation["capture"], "context": evidence,
                      "timing_scope": CAPTURED_TIMING_SCOPE,
                      "statistic": "median_captured_kernel_duration",
                      "captured_calls": observation["calls"],
                      "captured_min_ns": observation["min_ns"],
                      "captured_max_ns": observation["max_ns"],
                      "captured_median_ns": observation["median_ns"],
                      # One inventory row aggregates every repeated call of the
                      # variant.  Whether those calls are independent trials is
                      # not recorded by the capture, so it stays unknown here
                      # instead of being asserted either way.
                      "aggregated_observation_rows": 1,
                      "independent_trials": None,
                      "resource_occupancy": "not_captured",
                      "occupancy_basis": "declared_work_proxy",
                      "occupancy_demands": dict(demands),
                      "independent_accuracy_acceptance": False}
        return KernelEstimate(plan=KernelPlan(
            backend=PROFILED_KERNEL_BACKEND, version="observed",
            stages=(ExecutionStage(
                "captured_service", demands=demands,
                minimum_duration_ms=observation["median_ns"] / 1e6,
                notes=("captured duration is a completion floor; the demands are "
                       "the operator's declared work, not measured occupancy",),
            ),),
            provenance=provenance, includes_dispatch=False,
        ))
