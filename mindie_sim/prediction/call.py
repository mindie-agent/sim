"""Finite API-call and replay composition through the ordinary graph engine.

[Role] Price a complete declared callback and its timed repetition protocol.
[Invariants] Repeats retain real intermediate work, stream order and launch
  boundaries. Serving scheduler/completion costs do not enter a primitive call.
  No observation, fitted duration or event relabeling enters this interface.
[Timeouts & Locks] Explicit expanded-operator budget; no I/O, locks or devices.
[Side-Effect Policy] Returns a compact report without modifying input cases.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError
from .graph_analysis import graph_costs
from .program import RecipeBook
from .structure import parse_case


def call_costs(case, ops, *, recipes=None, invocations=1, mode="eager",
               include_submission=False, submission=None, input_ready_ms=0,
               max_operators=8192, trace_limit=0):
    """Compose timed invocations, returning a device/submitted graph boundary.

    ``graph`` means one replay containing all invocations; ``eager`` submits
    each expanded kernel/API node separately. This models the declared queue,
    not Python allocator behavior, event-record overhead or unknown dispatch.
    """
    if (not isinstance(invocations, int) or isinstance(invocations, bool)
            or invocations < 1 or max_operators < 1):
        raise ModelingError("call_requires_positive_invocations_and_operator_budget")
    if mode not in {"eager", "graph"}:
        raise ModelingError("call_mode_requires_eager_or_graph")
    submission = dict(submission or {})
    outside = submission.keys() - {"graph_replay_ms"}
    if outside:
        raise ModelingError("call_submission_outside_callback_boundary:" + ",".join(sorted(outside)))
    replay = submission.get("graph_replay_ms")
    if replay is not None and (mode != "graph" or not math.isfinite(replay) or replay < 0):
        raise ModelingError("call_replay_service_requires_graph_mode_and_nonnegative_finite_time")
    case = parse_case(case)
    case = replace(case, runtime=replace(
        case.runtime, submission=dict(submission or {}), async_scheduling=False))
    book = recipes or RecipeBook.from_mapping(case.runtime.kernels)
    single = tuple(book.expand(tuple(ops)))
    if len(single) * invocations > max_operators:
        raise ModelingError("call_expanded_operator_budget_exceeded")
    identifiers = {op.op_id for op in single}
    if len(identifiers) != len(single):
        raise ModelingError("call_duplicate_operator_id")
    if any(set(op.deps) - identifiers for op in single):
        raise ModelingError("call_requires_complete_callback_dependencies")
    repeated = tuple(replace(
        op, op_id=f"call{index}/{op.op_id}",
        deps=tuple(f"call{index}/{dep}" for dep in op.deps),
    ) for index in range(invocations) for op in single)
    report = graph_costs(
        case, repeated, recipes=book, input_ready_ms=input_ready_ms,
        include_submission=include_submission,
        replay_scope="full" if mode == "graph" else "none",
        isolated=False, trace_limit=trace_limit,
    )
    return {
        "schema": "mindie-sim.call-costs/1",
        "boundary": report["boundary"], "duration_ms": report["duration_ms"],
        "input_ready_ms": input_ready_ms, "complete_ms": report["complete_ms"],
        "protocol": {"invocations": invocations, "mode": mode,
                     "include_submission": include_submission,
                     "submission": dict(submission or {})},
        "single_call": {
            "expanded_operators": len(single),
            "families": dict(Counter(op.family for op in single)),
            "source_boundaries": dict(Counter(op.source_boundary or "analytical"
                                              for op in single)),
        },
        "operator_occurrences": report["operator_occurrences"],
        "execution": report["execution"], "resources": report["resources"],
        "parameters": report["parameters"], "events": report["events"],
        "assumptions": [
            "A graph protocol is one replay containing every declared invocation.",
            "Eager nodes queue on their declared streams; host submission can overlap device work.",
            "Repeated calls preserve per-stream order; no implicit cross-stream callback fence is added.",
            "Input preparation and warmup outside the timed callback are excluded.",
            "Undeclared allocator, Python dispatch and event-record costs are not inferred.",
            "Device/submitted graph boundaries are not relabeled as observed event spans.",
        ],
    }
