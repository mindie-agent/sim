"""Streaming summaries of all executed microbatches.

[Role] Retain step counts and timing across phase, capture scope and bucket.
[Invariants] Trace truncation never changes aggregates; sum of overlapping step
  durations is not a serving makespan. Keys contain no iteration/request ids.
[Timeouts & Locks] O(configured modes/scopes/buckets), no history or I/O.
[Side-Effect Policy] Mutates only this run's numeric aggregates.
"""
from dataclasses import dataclass, field


@dataclass
class StepTotals:
    steps: int = 0
    valid_tokens: int = 0
    padded_tokens: int = 0
    operators: int = 0
    duration_sum_ms: float = 0
    duration_min_ms: float = float("inf")
    duration_max_ms: float = 0
    device_duration_sum_ms: float = 0

    def record(self, step):
        duration = step.completion.time - step.start_ms
        self.steps += 1
        self.valid_tokens += step.valid_tokens
        self.padded_tokens += step.padded_tokens
        self.operators += step.operators
        self.duration_sum_ms += duration
        self.duration_min_ms = min(self.duration_min_ms, duration)
        self.duration_max_ms = max(self.duration_max_ms, duration)
        self.device_duration_sum_ms += step.device_complete.time - step.start_ms

    def to_dict(self):
        return {"steps": self.steps, "valid_tokens": self.valid_tokens,
                "padded_tokens": self.padded_tokens, "operators": self.operators,
                "duration_ms": {"sum": self.duration_sum_ms,
                                "mean": self.duration_sum_ms / self.steps if self.steps else None,
                                "min": self.duration_min_ms if self.steps else None,
                                "max": self.duration_max_ms if self.steps else None},
                "device_duration_sum_ms": self.device_duration_sum_ms}


@dataclass
class StepSummary:
    total: StepTotals = field(default_factory=StepTotals)
    groups: dict = field(default_factory=dict)
    selections: dict = field(default_factory=dict)

    def record(self, step):
        self.total.record(step)
        key = step.mode, step.replay_scope, step.graph_bucket
        self.groups.setdefault(key, StepTotals()).record(step)
        for decision in step.implementation_selections:
            key = decision["selector"], decision["case"], decision["implementation"], decision["coverage"]
            self.selections[key] = self.selections.get(key, 0) + 1

    def to_dict(self):
        return {"unit": "replica_microbatch", "total": self.total.to_dict(),
                "groups": [{"mode": mode, "replay_scope": scope, "graph_bucket": bucket,
                            **totals.to_dict()}
                           for (mode, scope, bucket), totals in self.groups.items()],
                "duration_boundary": "scheduled_release_to_step_ready",
                "device_duration_boundary": "scheduled_release_to_all_device_ops_complete",
                "implementation_selections": [
                    {"selector": selector, "case": case, "implementation": implementation,
                     "coverage": coverage, "steps": count}
                    for (selector, case, implementation, coverage), count in self.selections.items()],
                "sum_semantics": "overlapping_step_durations_are_additive_not_makespan"}
