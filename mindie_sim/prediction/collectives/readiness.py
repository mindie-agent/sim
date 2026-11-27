"""Bind C7 observed context to the existing collective engine.

[Role]
  Validate C7 rank coverage/typed calls and start an existing collective plan
  with the *pipeline's actual* per-rank ready events.  An optional T06 startup
  service becomes an Engine gate on those same events.  C7 entry/completion
  and same-stream order are observed context, never a ready threshold.
[Invariants]
  * Readiness comes from the ordinary engine op/resource pipeline.
  * A rank-local or cross-rank observed timestamp never creates a delay.
  * Missing causal predecessor/clock evidence stays unknown.
  * A missing startup service records an unknown and submits no invented delay.
  * No extra service demand is added; waits and hardware work stay separate.
[Timeouts & Locks]
  Pure record validation and event submission on the caller's Engine.
[Side-Effect Policy]
  Submits the caller's plan events; reads only the supplied C7 report.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError

SCHEMA = "mindie-profiler/collective-readiness/1"


def _rank_index(call: Mapping[str, Any]) -> int:
    value = call.get("rank_index")
    if isinstance(value, int):
        return value
    rank = str(call.get("rank_id") or "")
    suffix = rank.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    raise ModelingError(f"collective_readiness_rank_index_missing:{rank}")


def _normalise_dtype(value):
    text = str(value or "").lower()
    return {"bfp16": "bf16", "bfloat16": "bf16", "fp32": "fp32",
            "float32": "fp32", "fp16": "fp16", "float16": "fp16"}.get(text, text)


def _select_calls(report: Mapping[str, Any], call_id: str | None,
                  connection_id: str | None = None,
                  ordinal: int | None = None,
                  communicator: str | None = None):
    doc = report.get("collective_readiness", report)
    if not isinstance(doc, Mapping) or doc.get("schema") != SCHEMA:
        raise ModelingError("collective_readiness_schema_unknown")
    calls = [item for item in (doc.get("calls") or ()) if isinstance(item, Mapping)]
    if call_id is not None:
        calls = [item for item in calls if item.get("call_id") == call_id
                 or item.get("operation", {}).get("op_name") == call_id]
    if communicator is not None:
        calls = [item for item in calls
                 if str(item.get("operation", {}).get("communicator")
                        or item.get("operation", {}).get("group_name"))
                 == str(communicator)]
    if connection_id is not None:
        calls = [item for item in calls
                 if str(item.get("connection_id")) == str(connection_id)]
    if ordinal is not None:
        calls = [item for item in calls
                 if int(item.get("call_ordinal", -1)) == int(ordinal)]
    if not calls:
        raise ModelingError("collective_readiness_no_matching_call")
    by_rank = {}
    for call in calls:
        rank = _rank_index(call)
        if rank in by_rank:
            raise ModelingError(f"collective_readiness_duplicate_rank:{rank}")
        by_rank[rank] = call
    return by_rank


def _validate(calls, ready_by_rank, require_typed, expected_collective=None,
              expected_bytes=None, expected_dtype=None,
              expected_group_size=None):
    if set(calls) != set(ready_by_rank):
        raise ModelingError(
            "collective_readiness_ready_coverage_mismatch:"
            f"calls={sorted(calls)}:ready={sorted(ready_by_rank)}")
    observed = {}
    for rank, call in calls.items():
        operation = dict(call.get("operation") or {})
        if expected_collective is not None and operation.get(
                "collective") != expected_collective:
            raise ModelingError(
                f"collective_readiness_collective_mismatch:{rank}:"
                f"{operation.get('collective')}!={expected_collective}")
        if (expected_dtype is not None
                and _normalise_dtype(operation.get("dtype"))
                != _normalise_dtype(expected_dtype)):
            raise ModelingError(
                f"collective_readiness_dtype_mismatch:{rank}:"
                f"{operation.get('dtype')}!={expected_dtype}")
        if expected_bytes is not None and operation.get("typed_bytes") != expected_bytes:
            raise ModelingError(
                f"collective_readiness_bytes_mismatch:{rank}:"
                f"{operation.get('typed_bytes')}!={expected_bytes}")
        rank_size = operation.get("rank_size")
        if (expected_group_size is not None and rank_size not in (None, 0)
                and int(rank_size) != int(expected_group_size)):
            raise ModelingError(
                f"collective_readiness_group_mismatch:{rank}:"
                f"{rank_size}!={expected_group_size}")
        if require_typed:
            if (not operation.get("dtype")
                    or not isinstance(operation.get("typed_bytes"), (int, float))
                    or operation.get("typed_bytes") <= 0):
                raise ModelingError(
                    f"collective_readiness_typed_admission_missing:{rank}")
        observed[rank] = {
            "event": ready_by_rank[rank],
            "call_id": call.get("call_id"),
            "operation": operation,
            "entry_ns": (call.get("entry") or {}).get("start_ns"),
            "completion_ns": (call.get("completion") or {}).get("end_ns"),
            "observed_context": dict(call.get("observed_context") or {}),
            "clock": dict(call.get("clock") or {}),
            "dependencies": tuple(call.get("dependencies") or ()),
            "waits": tuple(call.get("waits") or ()),
            "unknowns": tuple(call.get("unknowns") or ()),
            "readiness_basis": "pipeline_engine_event",
        }
    return observed


def bind_pipeline_ready(
    report: Mapping[str, Any], ready_by_rank: Mapping[int, Any], *,
    call_id: str | None = None, require_typed: bool = True,
    expected_collective: str | None = None, expected_bytes=None,
    expected_dtype: str | None = None, expected_group_size: int | None = None,
    connection_id: str | None = None, ordinal: int | None = None,
    communicator: str | None = None,
):
    """Validate C7 against actual pipeline ready events; return context only."""
    calls = _select_calls(report, call_id, connection_id=connection_id,
                          ordinal=ordinal, communicator=communicator)
    return _validate(calls, ready_by_rank, require_typed,
                     expected_collective=expected_collective,
                     expected_bytes=expected_bytes, expected_dtype=expected_dtype,
                     expected_group_size=expected_group_size)


def _startup_value(service):
    if service is None:
        return None
    value = (service.get("startup_us") if isinstance(service, Mapping)
             else getattr(service, "startup_us", None))
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ModelingError(f"collective_readiness_startup_invalid:{value}")
    return number


def protocol_startup_gate(
    engine, ready_by_rank: Mapping[int, Any], service, *,
    name: str = "protocol_startup", owner: str = "collective_readiness",
):
    """Submit one explicit protocol-startup gate per participant rank.

    ``service`` is an already-normalized T06 resource/phase record (mapping or
    object with ``startup_us``).  Its fixed term is an Engine event in the same
    program as the collective, not an additive estimate outside it.  Missing
    service data stays an explicit unknown and submits no invented delay.
    """
    startup_us = _startup_value(service)
    if startup_us is None:
        return dict(ready_by_rank), {
            "status": "unknown_not_submitted",
            "startup_us": None,
            "source": (dict(service) if isinstance(service, Mapping)
                       else getattr(service, "source", None)),
            "boundary": "protocol_startup_gate_not_submitted",
        }
    gates = {}
    for rank in sorted(ready_by_rank):
        gates[rank] = engine.submit(
            f"{name}:rank{rank}", after=(ready_by_rank[rank],),
            latency=startup_us / 1000.0, owner=owner,
            metadata={"service_boundary": "protocol_startup",
                      "startup_us": startup_us,
                      "rank": rank})
    return gates, {
        "status": "submitted",
        "startup_us": startup_us,
        "source": (dict(service) if isinstance(service, Mapping)
                   else getattr(service, "source", None)),
        "boundary": "protocol_startup_gate_once_per_rank",
    }


def readiness_skew(context: Mapping[int, Mapping[str, Any]]) -> dict:
    """Report actual per-rank ready times; never invent skew from C7 stamps."""
    times = {}
    for rank, item in context.items():
        event = item.get("event")
        value = getattr(event, "time", None) if event is not None else None
        if value is not None:
            times[int(rank)] = float(value)
    if not times:
        return {"status": "unknown", "ready_ms": {}, "skew_ms": None,
                "observed_ranks": 0}
    return {
        "status": "pipeline_events",
        "ready_ms": times,
        "skew_ms": max(times.values()) - min(times.values()),
        "observed_ranks": len(times),
        "basis": "actual_engine_event_times_not_C7_clock",
    }


def start_with_pipeline(
    report: Mapping[str, Any], engine, plan, ready_by_rank: Mapping[int, Any], *,
    call_id: str | None = None, require_typed: bool = True,
    expected_collective: str | None = None, expected_bytes=None,
    expected_dtype: str | None = None, expected_group_size: int | None = None,
    connection_id: str | None = None, ordinal: int | None = None,
    communicator: str | None = None, startup_service=None,
    name: str = "collective_readiness",
):
    """Start an existing plan with actual pipeline readiness plus C7 context.

    An explicit T06 startup service becomes a gate in the same Engine before
    the plan entry.  The gate's completion is the rank's new ready event, so a
    slow rank delays the plan through a real dependency rather than a fitted
    collective-wide offset.
    """
    context = bind_pipeline_ready(
        report, ready_by_rank, call_id=call_id, require_typed=require_typed,
        expected_collective=expected_collective, expected_bytes=expected_bytes,
        expected_dtype=expected_dtype, expected_group_size=expected_group_size,
        connection_id=connection_id, ordinal=ordinal,
        communicator=communicator)
    gates, startup = protocol_startup_gate(
        engine, ready_by_rank, startup_service,
        name=f"{name}:startup", owner="collective_readiness")
    for item in context.values():
        item["protocol_startup"] = startup
    completions = plan.start_participants(
        engine, f"{name}:plan", gates, owner="collective_readiness")
    return completions, context
