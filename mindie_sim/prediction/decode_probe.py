"""Explicit decode windows and resident capacity for device-body probes.

[Role] Preserve actual KV progress, DP population and resource feasibility.
[Invariants] Verification rows are not accepted output tokens. A sample window
  is not an autoregressive trajectory. Capacity never replaces analytical cost
  or silently reduces the declared batch. Dummy graph rows own no request KV.
[Timeouts & Locks] O(layers + participants + sampled windows), no I/O or locks.
[Side-Effect Policy] Read-only input metadata; no device or request execution.
"""
from .balanced_routing import cohort_padded_tokens
from .capacity import Memory
from .dispatch import cohorts
from .graph import GraphPolicy
from .operator_analysis import _check_geometry, cohort_sequence_counts
from .structure import parse_case
from mindie_sim.schema.workload import RequestView


def decode_windows(prompt_tokens, output_tokens, *, query_tokens=1, output_offsets=None):
    """Sample complete verification windows inside an explicit output horizon.

    ``generated_before`` is a declared cursor, not an acceptance estimate. For
    q=4 and output=3000 the final full window starts at cursor2996. No accepted
    token multiplier changes its four target rows. An actual replay should pass
    its observed cursors through ``output_offsets``.
    """
    if any(type(v) is not int or v < 1 for v in (prompt_tokens, output_tokens, query_tokens)):
        raise ValueError("decode_window_lengths_require_positive_integers")
    if query_tokens > output_tokens:
        raise ValueError("decode_query_exceeds_output_horizon")
    last = output_tokens - query_tokens
    offsets = tuple(dict.fromkeys((0, last // 2, last) if output_offsets is None else output_offsets))
    if not offsets or any(type(v) is not int or not 0 <= v <= last for v in offsets):
        raise ValueError("decode_window_cursor_outside_output_horizon")
    return tuple({"generated_before": offset, "context": prompt_tokens + offset,
                  "tokens": query_tokens, "kv_length": prompt_tokens + offset + query_tokens}
                 for offset in offsets)


def probe_capacity(case, *, context, tokens, sequences=1, phase="decode"):
    """Price resident weights/cache without running or gating a target graph.

    Like target_graphs, scalar sequences means *each* member of the first EP/DP
    cohort. A vector explicitly retains idle peers. Other independent cohorts
    are not projected into a global request count. Per-stage capacity uses the
    limiting physical rank, including declared reserved pools and draft state.
    """
    case = parse_case(case)
    _check_geometry(phase, tokens, 1, context, "body")
    participants = cohorts(case.runtime.parallel)[0]
    counts = cohort_sequence_counts(sequences, participants)
    views = {r: tuple(
        RequestView(f"capacity{r}:{i}", tokens, context, False, phase)
        for i in range(count)) for r, count in counts.items()}
    selections = {r: GraphPolicy().select(group, case.runtime) if group else (None, "none")
                  for r, group in views.items()}
    graph_rows = {r: max(tokens * counts[r], bucket or 0)
                  for r, (bucket, _) in selections.items()}
    padded = {r: cohort_padded_tokens((group,), case.runtime) for r, group in views.items()}
    # Body queries, not the case's potentially 126K prefill request, determine
    # live activation workspace. KV grows with valid contexts, not padding.
    memory = Memory(case, workspace_tokens=max(padded.values(), default=0))
    cache = memory.cache_bytes(context + tokens)
    violations, rows = [], []
    runtime = case.runtime
    if context + tokens > runtime.max_model_len:
        violations.append({"constraint": "max_model_len", "required": context + tokens,
                           "available": runtime.max_model_len})
    for replica, count in counts.items():
        for constraint, required, available in (
                ("max_num_seqs", count, runtime.max_num_seqs),
                ("max_num_batched_tokens", count * tokens, runtime.max_num_batched_tokens)):
            if required > available:
                violations.append({"constraint": constraint, "replica": replica,
                                   "required": required, "available": available})
        for stage, per_request in enumerate(cache):
            required = per_request * count
            available = memory.capacity[replica][stage]
            resident = (memory.weights[stage] + memory.workspace[stage]
                        + memory.policy.reserved + required)
            row = {"replica": replica, "stage": stage, "sequences": count,
                   "cache_bytes": required, "cache_capacity_bytes": available,
                   "headroom_bytes": available - required,
                   "resident_bytes": resident,
                   "weights_bytes": memory.weights[stage],
                   "workspace_bytes": memory.workspace[stage],
                   "reserved_bytes": memory.policy.reserved}
            rows.append(row)
            if required > available:
                violations.append({"constraint": "resident_memory", "replica": replica,
                                   "stage": stage, "required": required, "available": available})
    return {"status": "outside_declared_capacity" if violations else "within_declared_capacity",
            "basis": "declared_resident_geometry_not_serving_admission",
            "context_tokens": context, "query_tokens_per_sequence": tokens,
            "kv_length": context + tokens, "sequences_by_replica": counts,
            "sequences_in_cohort": sum(counts.values()),
            "deployment_dp": runtime.parallel.dp,
            "participating_replicas": list(participants),
            "unprobed_replicas": [r for r in range(runtime.parallel.dp) if r not in counts],
            "padded_query_tokens_by_replica": padded,
            "graph_query_tokens_by_replica": graph_rows,
            "graph_replay_scope_by_replica": {r: scope for r, (_, scope) in selections.items()},
            "cache_bytes_per_sequence_stage": cache, "ranks": rows,
            "violations": violations,
            "assumptions": ["KV prefixes are private per request; ready prefix KV is not free memory",
                            "Workspace uses the maximum padded query rows in this cohort",
                            "Declared graph/runtime/communication reserves remain included",
                            "Analytical body execution remains available outside this capacity"]}
