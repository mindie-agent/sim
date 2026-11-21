"""Cancellation audit for components measured inside a model step.

[Role] Score the step and each operator family or block using the gate's tolerance.
[Invariants] Compute cancellation independently for each level from signed errors.
[Timeouts & Locks] One linear pass over components, with no waits or locks.
[Side-Effect Policy] Pure scoring over caller-supplied same-capture durations.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError

DRILLDOWN_LEVELS = ("L1_in_L3", "L2_in_L3")


@dataclass(slots=True)
class _CancellationAudit:
    component_count: int = 0
    absolute_error_ns: int = 0
    signed_error_ns: int = 0

    def add(self, error_ns: int) -> None:
        self.component_count += 1
        self.absolute_error_ns += abs(error_ns)
        self.signed_error_ns += error_ns

    def to_dict(self) -> dict[str, int | float]:
        cancellation_ns = self.absolute_error_ns - abs(self.signed_error_ns)
        return {
            "component_count": self.component_count,
            "cancellation_ns": cancellation_ns,
            "cancellation_ratio": cancellation_ns / max(1.0, self.absolute_error_ns),
        }


def score_vertical_drilldown(
    step_predicted_ns: int, step_measured_ns: int, components: Sequence[tuple[str, str, int, int]],
    exceeds_tolerance: Callable[[int, int], bool],
) -> dict[str, Any]:
    audits = {level: _CancellationAudit() for level in DRILLDOWN_LEVELS}
    failed_components = []
    for level, name, predicted_ns, measured_ns in components:
        if level not in audits:
            raise ModelingError(f"invalid_drilldown_level:{level}")
        audits[level].add(predicted_ns - measured_ns)
        if exceeds_tolerance(predicted_ns, measured_ns):
            failed_components.append({
                "level_tag": level, "name": name, "predicted_ns": predicted_ns, "measured_ns": measured_ns,
            })
    step_passed = not exceeds_tolerance(step_predicted_ns, step_measured_ns)
    return {
        "passed": step_passed and not failed_components,
        "step_passed": step_passed,
        "failed_components": failed_components,
        "levels": {level: audit.to_dict() for level, audit in audits.items()},
    }
