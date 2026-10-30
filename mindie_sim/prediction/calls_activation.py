"""Analytical device structure of complete activation API closures.

[Role] Build standalone, fused and two-API gated activation call graphs.
[Invariants] Inputs are already resident. Separate activation and multiplication
  materialize their intermediate tensor. API boundaries do not prove a native
  binary's kernel count, fusion, tile mapping or launch latency.
[Timeouts & Locks] Pure geometry, at most two operators per call.
[Side-Effect Policy] No observations, I/O, device context or input mutation.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from .program import RecipeBook


# API topology and expression work, independent of measured event durations.
# SiLU x/(1+exp(-x)): negate, exp, add, divide. Tanh GELU uses two
# multiplications for x^3, three scalar products, two additions, tanh and
# the final product. All count as nominal vector-lane operations here.
ACTIVATION_CALLS = {
    "swiglu_fused": {
        "api": ("torch_npu.npu_swiglu(dim=-1)",),
        "layout": "contiguous_gate_first",
        "steps": (("swiglu", 5, 2),),
    },
    "swiglu_separate": {
        "api": ("torch.nn.functional.silu", "Tensor.mul"),
        "layout": "separate_contiguous_gate_up",
        "steps": (("silu", 4, 1), ("multiply", 1, 2)),
    },
    "geglu_separate": {
        "api": ("torch.nn.functional.gelu(approximate=tanh)", "Tensor.mul"),
        "layout": "separate_contiguous_gate_up",
        "steps": (("gelu_tanh", 9, 1), ("multiply", 1, 2)),
    },
    "silu_standalone": {
        "api": ("torch.nn.functional.silu",),
        "layout": "contiguous",
        "steps": (("silu", 4, 1),),
    },
    "gelu_standalone": {
        "api": ("torch.nn.functional.gelu(approximate=tanh)",),
        "layout": "contiguous",
        "steps": (("gelu_tanh", 9, 1),),
    },
}


@dataclass(frozen=True)
class ActivationCallSpec:
    variant: str
    tokens: int
    intermediate: int
    dtype: str = "bf16"
    phase: str = "prefill"

    def __post_init__(self):
        if self.variant not in ACTIVATION_CALLS:
            raise ModelingError(f"activation_call_variant_unsupported:{self.variant}")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (self.tokens, self.intermediate)):
            raise ModelingError("activation_call_requires_positive_integer_tensor_shape")
        if self.dtype not in {"bf16", "bfloat16"}:
            raise ModelingError(f"activation_call_dtype_unsupported:{self.dtype}")
        if self.phase not in {"prefill", "decode"}:
            raise ModelingError(f"activation_call_phase_unsupported:{self.phase}")


@dataclass(frozen=True)
class ActivationCall:
    spec: ActivationCallSpec
    ops: tuple[OpSpec, ...]
    recipes: RecipeBook
    coverage: dict


def build_activation_call(spec: ActivationCallSpec) -> ActivationCall:
    """Construct one complete closure; the common call runner owns repetition."""
    declaration = ACTIVATION_CALLS[spec.variant]
    elements = spec.tokens * spec.intermediate
    tensor_bytes = elements * 2
    shape = (spec.tokens, spec.intermediate)
    ops = []
    for ordinal, (operation, lane_ops, inputs) in enumerate(declaration["steps"]):
        ops.append(vector_op(
            op_id=f"activation_api{ordinal}", deps=(ops[-1].op_id,) if ops else (),
            stage=0, layer=0, requests=(), elements=elements,
            ops_per_element=lane_ops, dtype="bf16", family=operation, phase=spec.phase,
            recipe=f"analytic_api_{operation}", shape=shape,
            read_bytes_override=inputs * tensor_bytes, write_bytes_override=tensor_bytes,
            metadata=(
                ("api", declaration["api"][ordinal]),
                ("call_variant", spec.variant),
                ("call_layout", declaration["layout"]),
                ("arithmetic_basis", "nominal_expression_lane_operations"),
                ("native_kernel_dispatch", "unresolved"),
                ("materialized_input", "activation_temporary" if ordinal else "resident_operand"),
            ),
        ))
    separate = len(ops) == 2
    coverage = {
        "model": "analytic_activation_call/1",
        "abstraction": "api_resource_envelope",
        "variant": spec.variant,
        "phase": spec.phase,
        "shape": {"tokens": spec.tokens, "intermediate": spec.intermediate},
        "dtype": "bf16",
        "layout": declaration["layout"],
        "api_sequence": list(declaration["api"]),
        "api_calls": len(ops),
        "native_kernel_count": None,
        "input_residency": "allocated_and_initialized_before_timed_call",
        "intermediate": (
            {"shape": list(shape), "dtype": "bf16", "bytes": tensor_bytes,
             "producer": ops[0].op_id, "consumer": ops[1].op_id,
             "lifetime": "activation_return_through_multiply_read"}
            if separate else None),
        "output": {"shape": list(shape), "dtype": "bf16", "bytes": tensor_bytes},
        "hbm_bytes": sum(op.hbm_bytes for op in ops),
        "vector_lane_operations": sum(op.vector_ops for op in ops),
        "unresolved": [
            "loaded_kernel_binary",
            "native_api_internal_dispatch_and_kernel_count",
            "instruction_class_service_for_exp_divide_and_tanh",
            "cache_residency_and_backend_tile_mapping",
            "host_api_allocator_and_event_dispatch_gaps",
        ],
        "interpretation": (
            "Complete declared API dataflow; analytical device service estimate, "
            "not an identified native kernel implementation or event duration."
        ),
    }
    return ActivationCall(spec, tuple(ops), RecipeBook(), coverage)
