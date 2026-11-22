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
    request_commits: int = 0
    fallback_completions: int = 0
    unclassified_commits: int = 0
    merged_commits: int = 0
    host_results: int = 0
    client_deliveries: int = 0
    completed_request_commits: int = 0
    completed_fallback_completions: int = 0
    completed_unclassified_commits: int = 0
    completed_merged_commits: int = 0
    completed_host_results: int = 0
    completed_client_deliveries: int = 0
    first_commit_ms: float | None = None
    last_commit_ms: float | None = None
    first_delivery_ms: float | None = None
    last_delivery_ms: float | None = None

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
        # Execution.add writes two per-request Event maps. Consume their
        # sources directly; never infer a split from event names or from an
        # assembly-side graph scan. The merged ``commits`` mapping is retained
        # only as a compatibility view, and any request present only there is
        # explicitly unclassified rather than guessed.
        declared_map = getattr(step, "declared_commits", None)
        fallback_map = getattr(step, "fallback_commits", None)
        if declared_map is None or fallback_map is None:
            declared_items = []
            fallback_items = []
            unclassified_items = list(step.commits.items())
        else:
            declared_items = list(declared_map.items())
            fallback_items = list(fallback_map.items())
            known = set(declared_map) | set(fallback_map)
            unclassified_items = [
                (request, event) for request, event in step.commits.items()
                if request not in known]
        declared = [event for _, event in declared_items]
        fallback = [event for _, event in fallback_items]
        unclassified = [event for _, event in unclassified_items]
        commits = [event.time for event in declared]
        completions = [event.time for event in fallback]
        unclassified_times = [event.time for event in unclassified]
        commit_sequences = {event.sequence for event in step.commits.values()}
        host_events = [
            event for event in step.host_results.values()
            if event.sequence not in commit_sequences and event is not step.completion]
        host = [event.time for event in host_events]
        deliveries = [event.time for event in step.client_deliveries.values()]
        self.request_commits += len(commits)
        self.fallback_completions += len(completions)
        self.unclassified_commits += len(unclassified)
        self.merged_commits += len(step.commits)
        self.host_results += len(host)
        self.client_deliveries += len(deliveries)
        self.completed_request_commits += sum(value is not None for value in commits)
        self.completed_fallback_completions += sum(
            value is not None for value in completions)
        self.completed_unclassified_commits += sum(
            value is not None for value in unclassified_times)
        self.completed_merged_commits += sum(
            event.time is not None for event in step.commits.values())
        self.completed_host_results += sum(value is not None for value in host)
        self.completed_client_deliveries += sum(value is not None for value in deliveries)
        commits = [value for value in commits if value is not None]
        completions = [value for value in completions if value is not None]
        host = [value for value in host if value is not None]
        deliveries = [value for value in deliveries if value is not None]
        for value in commits:
            self.first_commit_ms = (value if self.first_commit_ms is None
                                    else min(self.first_commit_ms, value))
            self.last_commit_ms = (value if self.last_commit_ms is None
                                   else max(self.last_commit_ms, value))
        for value in deliveries:
            self.first_delivery_ms = (value if self.first_delivery_ms is None
                                      else min(self.first_delivery_ms, value))
            self.last_delivery_ms = (value if self.last_delivery_ms is None
                                     else max(self.last_delivery_ms, value))

    def to_dict(self):
        return {"steps": self.steps, "valid_tokens": self.valid_tokens,
                "padded_tokens": self.padded_tokens, "operators": self.operators,
                "duration_ms": {"sum": self.duration_sum_ms,
                                "mean": self.duration_sum_ms / self.steps if self.steps else None,
                                "min": self.duration_min_ms if self.steps else None,
                                "max": self.duration_max_ms if self.steps else None},
                "device_duration_sum_ms": self.device_duration_sum_ms,
                "request_events": {
                    "commits": self.request_commits,
                    "fallback_completions": self.fallback_completions,
                    "unclassified_commits": self.unclassified_commits,
                    "merged_commits": self.merged_commits,
                    "host_results": self.host_results,
                    "client_deliveries": self.client_deliveries,
                },
                "request_events_completed_at_step_ready": {
                    "commits": self.completed_request_commits,
                    "fallback_completions": self.completed_fallback_completions,
                    "unclassified_commits": self.completed_unclassified_commits,
                    "merged_commits": self.completed_merged_commits,
                    "host_results": self.completed_host_results,
                    "client_deliveries": self.completed_client_deliveries,
                },
                "request_event_windows_ms": {
                    "first_commit": self.first_commit_ms,
                    "last_commit": self.last_commit_ms,
                    "first_delivery": self.first_delivery_ms,
                    "last_delivery": self.last_delivery_ms,
                }}


@dataclass
class StepSummary:
    total: StepTotals = field(default_factory=StepTotals)
    groups: dict = field(default_factory=dict)
    selections: dict = field(default_factory=dict)

    def record(self, step):
        self.total.record(step)
        key = step.mode, step.replay_scope, step.graph_bucket
        self.groups.setdefault(key, StepTotals()).record(step)
        seen = set()
        for decision in step.implementation_selections:
            key = (decision["selector"], decision["case"], decision["implementation"],
                   decision["coverage"], decision["model_scope"])
            counts = self.selections.setdefault(key, [0, 0])
            counts[0] += key not in seen
            counts[1] += 1
            seen.add(key)

    def to_dict(self):
        return {"unit": "replica_microbatch", "total": self.total.to_dict(),
                "groups": [{"mode": mode, "replay_scope": scope, "graph_bucket": bucket,
                            **totals.to_dict()}
                           for (mode, scope, bucket), totals in self.groups.items()],
                "duration_boundary": "scheduled_release_to_step_ready",
                "device_duration_boundary": "scheduled_release_to_all_device_ops_complete",
                "implementation_selections": [
                    {"selector": selector, "case": case, "implementation": implementation,
                     "coverage": coverage, "model_scope": model_scope,
                     "steps": counts[0], "calls": counts[1]}
                    for (selector, case, implementation, coverage, model_scope), counts
                    in self.selections.items()],
                "sum_semantics": "overlapping_step_durations_are_additive_not_makespan",
                "request_event_semantics": (
                    "counts are per-request declared commit/host/delivery events; "
                    "fallback_completions come from Execution.add's explicit "
                    "fallback_commits map; merged_commits is a compatibility "
                    "view only and never replaces the two source maps")}
