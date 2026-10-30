"""Boundary binding for retained INT8/FRACTAL_NZ grouped-matmul events.

[Role] Validate the audited producer contract and build the independent call.
[Invariants] Only geometry, dtype, layout, API contract, execution protocol and
  provenance are read. Event durations, canaries and sample splits never enter
  the builder, prediction identity or resources.
[Timeouts & Locks] Pure bounded metadata validation; no capture scan or locks.
[Side-Effect Policy] No calibration, source writes, device work or timing math.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mindie_sim.prediction.call_grouped import (
    API, NZ_LAYOUT, PRODUCER_SHA256, QuantizedGroupedCall,
    build_quantized_grouped_call,
)
from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class GroupedBinding:
    call: QuantizedGroupedCall
    invocations: int
    mode: str
    target_identity: dict
    condition_id: str
    ledger: dict


def _expected_contract(experts):
    return {
        "api": API,
        "x": "int8 [tokens, k]",
        "weight": "int8 [E, k, n] FRACTAL_NZ",
        "scale": "float32 [E, n] per-channel",
        "per_token_scale": "float32 [tokens]",
        "split_item": 2,
        "group_type": 0,
        "group_list_type": 1,
        "group_list": f"int64 counts over E={experts} groups",
        "out_dtype": "bfloat16",
    }


def _expected_execution(mode, invocations):
    if mode == "eager" and invocations == 1:
        return {
            "mode": "eager", "timer": "npu_events", "invocations": 1,
            "warmup_calls": 10, "activation_quant": "outside_timed_region",
            "output_lifetime": "discard_after_call",
            "capture_warmup_calls": 0, "dependency": "serial",
        }
    if mode == "graph" and invocations in {1, 8}:
        return {
            "mode": "graph", "timer": "npu_events", "invocations": invocations,
            "warmup_calls": 10, "activation_quant": "outside_timed_region",
            "output_lifetime": "captured_static_buffers",
            "capture_warmup_calls": 3, "dependency": "serial",
        }
    raise ModelingError("grouped_binding_unknown_execution_protocol")


def _identity_contract(identity):
    """Return validated tensors/contract without reading observation samples."""
    implementation = identity.get("implementation")
    if not isinstance(implementation, dict):
        raise ModelingError("grouped_binding_requires_implementation_identity")
    if implementation.get("producer_sha256") != PRODUCER_SHA256:
        raise ModelingError("grouped_binding_requires_audited_producer")
    if implementation.get("api") != API:
        raise ModelingError("grouped_binding_api_differs_from_producer")
    contract = implementation.get("contract")
    if not isinstance(contract, dict):
        raise ModelingError("grouped_binding_requires_complete_api_contract")

    shape = identity.get("shape")
    if not isinstance(shape, dict) or set(shape) != {
            "tokens", "k", "n", "experts", "expert_counts"}:
        raise ModelingError("grouped_binding_requires_exact_shape_contract")
    counts = shape["expert_counts"]
    if (isinstance(counts, (str, bytes)) or not isinstance(counts, (list, tuple))
            or len(counts) != shape["experts"]
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in counts)):
        raise ModelingError("grouped_binding_requires_explicit_integer_expert_counts")
    if sum(counts) != shape["tokens"]:
        raise ModelingError("grouped_binding_counts_must_sum_to_logical_rows")
    if contract != _expected_contract(shape["experts"]):
        raise ModelingError("grouped_binding_api_contract_differs_from_producer")

    expected_dtype = {
        "input": "int8", "weight": "int8", "scale": "float32",
        "per_token_scale": "float32", "output": "bfloat16",
    }
    expected_layout = {"input": "ND", "weight": NZ_LAYOUT, "output": "ND"}
    if identity.get("dtype") != expected_dtype:
        raise ModelingError("grouped_binding_dtype_contract_differs_from_producer")
    if identity.get("layout") != expected_layout:
        raise ModelingError("grouped_binding_layout_contract_differs_from_producer")
    if identity.get("phase") != "primitive":
        raise ModelingError("grouped_binding_phase_differs_from_producer")
    return shape, counts, contract


def bind_grouped_condition(condition):
    """Bind one retained condition; never consume timings, canary or split."""
    identity = condition.get("identity")
    if not isinstance(identity, dict):
        raise ModelingError("grouped_binding_requires_preserved_identity")
    shape, counts, contract = _identity_contract(identity)

    execution = identity.get("execution")
    if not isinstance(execution, dict):
        raise ModelingError("grouped_binding_requires_execution_protocol")
    mode = execution.get("mode")
    invocations = execution.get("invocations")
    expected = _expected_execution(mode, invocations)
    if execution != expected:
        raise ModelingError("grouped_binding_execution_differs_from_producer")

    level = "subgraph" if mode == "graph" else "operator"
    boundary = ("npu_event_graph_replay" if mode == "graph" else "npu_event_call")
    if identity.get("level") != level or identity.get("boundary") != boundary:
        raise ModelingError("grouped_binding_event_boundary_differs_from_producer")

    call = build_quantized_grouped_call({
        "expert_rows": tuple(counts),
        "k": shape["k"],
        "n": shape["n"],
        "phase": "primitive",
    })
    ledger = deepcopy(call.ledger)
    ledger.update({
        "binding": "retained_grouped_int8_fractal_nz_call/1",
        "target_identity": deepcopy(identity),
        "condition_id": condition.get("condition_id"),
        "timed_protocol": {
            "mode": mode,
            "invocations": invocations,
            "whole_replay_events": mode == "graph",
            "sample_division": False,
            "serial_calls": True,
        },
        "read_scope": {
            "geometry": True,
            "dtype": True,
            "layout": True,
            "api_contract": True,
            "execution": True,
            "provenance": True,
            "duration": False,
            "canary": False,
            "sample_split": False,
        },
        "modeled_api_arguments": dict(contract),
        "conditional_comparison_only": True,
        "strict_identity_equivalence": False,
        "observation_role": "unchanged",
        "duration_basis": "separately_supplied_engine_resources_and_call_schedule",
    })
    return GroupedBinding(
        call, invocations, mode, deepcopy(identity), condition.get("condition_id"), ledger,
    )


__all__ = ["GroupedBinding", "bind_grouped_condition"]
