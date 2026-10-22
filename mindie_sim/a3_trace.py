"""Bounded prediction detail with lossless physical-coverage aggregation.

[Role] Keep step summaries and a finite sample of expensive op/event detail.
[Invariants] Metrics and coverage include every simulated step. Omitted detail
  is explicit; full capture is opt-in. No sampling changes simulated execution.
[Timeouts & Locks] In-memory reduction, no waits or locks.
[Side-Effect Policy] Owns only per-prediction trace data.
"""
from __future__ import annotations

from typing import Any, Callable

COUNT_MAPS = (
    "unmodeled_reasons", "transfer_counts", "wiring_counts",
    "large_operator_wiring_counts", "large_operator_family_counts", "scope_alias_counts",
)
TOTALS = (
    "applied_op_count", "low_confidence_op_count", "unmodeled_op_count",
    "applied_transfer_op_count", "mc2_op_count", "modeled_duration_ms",
    "physical_applied_duration_ms", "large_operator_duration_ms",
)
MAX_DETAIL_OPS = 4096


def merge_coverage(rows: Any) -> dict[str, Any]:
    result: dict[str, Any] = {key: {} for key in COUNT_MAPS}
    result.update({key: 0 for key in TOTALS})
    large_applied = 0.0
    for row in rows:
        for key in COUNT_MAPS:
            for name, count in row.get(key, {}).items():
                result[key][name] = result[key].get(name, 0) + count
        for key in TOTALS:
            result[key] += row.get(key, 0)
        large_applied += row.get("large_operator_duration_ms", 0) * row.get(
            "large_operator_applied_duration_share", 0,
        )
    result["physical_applied_duration_share"] = (
        result["physical_applied_duration_ms"] / result["modeled_duration_ms"]
        if result["modeled_duration_ms"] else 0.0
    )
    result["large_operator_applied_duration_share"] = (
        large_applied / result["large_operator_duration_ms"]
        if result["large_operator_duration_ms"] else 0.0
    )
    return result


class PredictionTraces(list):
    def __init__(self, detail: str, summarize: Callable[[dict], dict]) -> None:
        super().__init__()
        self.detail = detail
        self.summarize = summarize
        self.coverage = merge_coverage(())
        self.total_ops = 0
        self.retained_ops = 0
        self.omitted_steps = 0

    def append(self, step: dict) -> None:
        self.coverage = merge_coverage((self.coverage, self.summarize(step)))
        count = step["op_count"]
        self.total_ops += count
        keep = self.detail == "full" or (
            self.detail == "bounded" and self.retained_ops + count <= MAX_DETAIL_OPS
        )
        if keep:
            self.retained_ops += count
        else:
            self.omitted_steps += 1
            step = {**step, "ops": [], "events": [], "fusion": None, "routing": None,
                    "detail_omitted": True}
        super().append(step)

    def report(self) -> dict[str, Any]:
        return {"detail": self.detail, "total_ops": self.total_ops,
                "retained_ops": self.retained_ops, "omitted_steps": self.omitted_steps,
                "detail_complete": self.omitted_steps == 0}
