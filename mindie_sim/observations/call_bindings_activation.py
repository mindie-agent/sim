"""Explicit structural alignment with retained activation event closures.

[Role] Bind audited producer protocols to an independently built call graph.
[Invariants] Read geometry and execution identity only, never event duration,
  fitted parameters, canaries or sample splits. Target observation identity
  remains separate from the analytical prediction boundary.
[Timeouts & Locks] Pure bounded metadata validation; no capture scan.
[Side-Effect Policy] No calibration, source writes or device execution.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mindie_sim.prediction.calls_activation import (
    ACTIVATION_CALLS, ActivationCall, ActivationCallSpec, build_activation_call,
)
from mindie_sim.schema.common import ModelingError


PRODUCER_SHA256 = "ea76c64cd935e2d35c849f957f2179701d81d72c951a738a430260de4b251136"
PRODUCER_PATH = (
    "/Users/maoxx241/code/mindie-agent/.workstreams/composability/scripts/bench_activation.py"
)


@dataclass(frozen=True)
class ActivationBinding:
    call: ActivationCall
    invocations: int
    mode: str
    target_identity: dict
    condition_id: str
    ledger: dict


def bind_activation_condition(condition) -> ActivationBinding:
    """Return a conditional boundary binding, not an accuracy prediction row."""
    identity = condition["identity"]
    implementation = identity["implementation"]
    if implementation.get("producer_sha256") != PRODUCER_SHA256:
        raise ModelingError("activation_binding_requires_audited_producer")
    variant = implementation["variant"]
    spec = ActivationCallSpec(
        variant, identity["shape"]["tokens"], identity["shape"]["intermediate"],
        dtype=identity["dtype"], phase=identity["phase"],
    )
    declaration = ACTIVATION_CALLS[variant]
    if (implementation.get("api") != list(declaration["api"])
            or identity["layout"] != declaration["layout"]):
        raise ModelingError("activation_binding_api_or_layout_differs_from_producer")
    execution = identity["execution"]
    invocations = execution["invocations"]
    if not isinstance(invocations, int) or isinstance(invocations, bool) or not 1 <= invocations <= 100:
        raise ModelingError("activation_binding_invocations_outside_producer_protocol")
    burst = identity["boundary"] == "npu_event_burst"
    if identity["boundary"] not in {"npu_event_call", "npu_event_burst"} or (
            not burst and invocations != 1):
        raise ModelingError("activation_binding_unknown_event_protocol")
    expected_level = "subgraph" if burst or len(declaration["api"]) > 1 else "operator"
    expected_execution = {
        "mode": "eager", "timer": "npu_events",
        "input_values": "zeros", "output_lifetime": "discard_after_call",
        "warmup_calls": 3 if burst else 10,
        "timer_interleaving": "event_calls_event" if burst else "event_call_event",
    }
    if (identity["level"] != expected_level or any(
            execution.get(name) != value for name, value in expected_execution.items())):
        raise ModelingError("activation_binding_execution_differs_from_producer")
    call = build_activation_call(spec)
    ledger = {
        "binding": "retained_activation_api_closure/1",
        "producer": {"path": PRODUCER_PATH, "sha256": PRODUCER_SHA256},
        "timed_region": {
            "starts": "event_record_before_first_api",
            "ends": "event_record_after_last_api",
            "input_allocation_and_zero_fill": "outside",
            "warmup": "outside",
            "output_allocation_and_destruction": "inside_api_closure",
            "invocations": invocations,
            "api_calls_per_invocation": call.coverage["api_calls"],
            "serial_same_device_stream": True,
        },
        "known_structure": call.coverage,
        "unresolved": call.coverage["unresolved"],
        "conditional_comparison_only": True,
        "strict_identity_equivalence": False,
        "observation_role": "unchanged",
        "measured_values_consumed": False,
        "duration_basis": "separately_supplied_hardware_resources_and_call_schedule",
    }
    return ActivationBinding(
        call, invocations, "eager", deepcopy(identity), condition["condition_id"], ledger,
    )
