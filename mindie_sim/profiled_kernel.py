"""Exact captured-kernel duration provider for the existing operator engine.

[Role]
  Reuse profiler inventory observations as explicit, scoped kernel cost plans.
[Invariants]
  Match an exact variant and caller-declared lowering contract; no shape, SKU,
  recipe or environment wildcard. Keep profiled durations separate from warm
  microbenchmarks, heldout fitting and simulated CAModel inputs. Whole captured
  kernel duration replaces the parent service, never adds analytic HBM again.
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
from .operator_plan import ExecutionStage, KernelEstimate, KernelPlan


class ProfiledKernelProvider:
    def __init__(self, config: Mapping[str, Any]):
        self.db = Path(config["db"]).resolve()
        self.environment_key = str(config["environment_key"])
        self.bindings = tuple(config.get("bindings", ()))
        self._cache: dict[str, dict[str, Any]] = {}

    def lookup(self, variant_id: str) -> dict[str, Any]:
        if variant_id not in self._cache:
            with closing(sqlite3.connect(self.db.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=5)) as connection:
                connection.execute("PRAGMA busy_timeout=5000")
                row = connection.execute(
                    "SELECT v.signature,v.context,v.eligible,v.reason,c.calls,c.median_ns,v.capture "
                    "FROM variants v JOIN costs c ON v.id=c.variant WHERE v.id=?",
                    (bytes.fromhex(variant_id),),
                ).fetchone()
            if row is None:
                raise A3InputError(f"profile_variant_missing:{variant_id}")
            result = {"signature": json.loads(row[0]), "context": json.loads(row[1]),
                      "eligible": bool(row[2]), "reason": row[3], "calls": row[4],
                      "median_ns": row[5], "capture": row[6]}
            if len(self._cache) >= 1024:
                self._cache.clear()
            self._cache[variant_id] = result
        return self._cache[variant_id]

    def estimate(self, op, machine, context):
        # Bindings are exact lowering contracts reviewed against the actual
        # profile; an equal first-input shape alone does not identify a kernel.
        selected = dict(op.metadata).get("profile_variant")
        matches = [b for b in self.bindings if (
            b["family"] == op.family and b["recipe"] == op.recipe
            and b["phase"] == op.phase and tuple(b["launch_shape"]) == op.launch_shape
            and b["dtype"] == op.dtype and b["layout"] == op.layout
            and (selected is None or selected == b["variant_id"])
        )]
        if len(matches) != 1:
            return KernelEstimate(unsupported_reason="profile_exact_binding_missing_or_ambiguous")
        if context.timing_scope != "profiled_device_kernel":
            return KernelEstimate(unsupported_reason="profile_timing_scope_mismatch")
        binding = matches[0]
        observation = self.lookup(binding["variant_id"])
        evidence = observation["context"]
        if not observation["eligible"]:
            return KernelEstimate(unsupported_reason="profile_ineligible:" + observation["reason"])
        if evidence.get("hardware_key") != machine.hardware_key or evidence.get("environment_key") != self.environment_key:
            return KernelEstimate(unsupported_reason="profile_hardware_or_environment_mismatch")
        provenance = {"source_class": "observed", "inventory": str(self.db),
                      "variant_id": binding["variant_id"], "signature": observation["signature"],
                      "capture": observation["capture"], "context": evidence,
                      "captured_calls": observation["calls"], "timing_scope": context.timing_scope,
                      "statistic": "median_captured_kernel_duration", "resource_occupancy": "not_captured",
                      "independent_accuracy_acceptance": False}
        return KernelEstimate(plan=KernelPlan(
            backend="profiled_kernel", version="observed",
            stages=(ExecutionStage("captured_service", minimum_duration_ms=observation["median_ns"] / 1e6),),
            provenance=provenance, includes_dispatch=False,
        ))
