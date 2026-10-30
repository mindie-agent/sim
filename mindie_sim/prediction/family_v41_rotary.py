"""Explicit partial-RoPE native boundaries for shared-KV attention callers.

[Role] Declare independent Q, KV and post-attention inverse rotations using the
  existing graph builder and resource engine.
[Invariants] Only the rotary slice is read/written in place; NoPE is untouched.
  Each declaration is one invocation. A Q/KV aggregate must be replaced, never
  retained alongside these nodes. Negative sine is supplied by the caller.
  Coefficient sharing across a token's heads is an explicit analytic assumption,
  not proof of native tiling, cache hits or captured library identity.
[Timeouts & Locks] Constant arithmetic; no I/O, locks or device contexts.
[Side-Effect Policy] Appends one immutable operator to the caller's builder.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, Stage


IMPLEMENTATION = "v41_partial_rotary"
NATIVE_KIND = "InplacePartialRotaryMul"
SOURCE = {
    "caller": {
        "repository": "vllm-ascend",
        "path": "vllm_ascend/attention/dsa_v41.py",
        "sha256": "56d3ce83098babf585289d3e078d1efff0442547f61fb0955a11f0b42bfc6ba3",
        "forward_q_kv_lines": [283, 305],
        "inverse_output_lines": [596, 611],
    },
    "slice_kernel": {
        "repository": "cann-recipes-infer",
        "path": "ops/ascendc/src/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_bsn_mixed.h",
        "sha256": "303cbd132566519a9cede7cc13156c62ea7462be26d71222425638bf0997b457",
        "copy_lines": [145, 216], "compute_lines": [284, 316],
    },
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def recipes():
    return {name: PlannedRecipe(_build) for name in
            ("q_rotary", "kv_rotary", "output_inverse_rotary")}


def _build(op, geometry):
    work = Work(resources={"hbm": op.hbm_bytes, "mte2_aiv": op.mte2_bytes,
                           "mte3": op.mte3_bytes, "vector": op.vector_ops})
    return OperatorPlan(work, lambda context: KernelProgram((
        Stage("partial_slice_service", context.work),)), {
            "execution_abstraction": "partial_slice_resource_envelope",
            "execution_class": "vector",
            "owner_schedule": "analytic_aggregate; native_BlockDim_not_selected",
        })


def validate(plan, *, fused_prolog):
    if (plan.mla_variant != "latent_mqa" or plan.cache_kind != "window_compressed"
            or not plan.qk_rope_head_dim or fused_prolog):
        raise ModelingError("partial_rotary_caller_requires_unfused_shared_kv_mqa")


def declare(builder, *, name, deps, tokens, heads, head_dim, rotary_width, inverse=False):
    """Append one BF16/FP32-coefficient in-place partial interleaved rotation.

    The caller selects this mechanism explicitly and supplies the full active
    tensor view. For Q/KV use independent dependencies and names ``q_rotary`` /
    ``kv_rotary``; output ``output_inverse_rotary`` depends on attention results.
    The shared vector resource counts semantic lanes, not ISA cycles: two
    multiplies, one add, two casts, one permutation and half-lane sign change.
    UB coefficient broadcast lanes are retained separately from HBM traffic.
    """
    dtype = builder.model.activation_dtype
    if (dtype != "bf16" or tokens < 0 or heads < 1 or head_dim < 1
            or rotary_width < 2 or rotary_width > head_dim or rotary_width % 2):
        raise ModelingError("partial_rotary_requires_bf16_even_slice_inside_head")
    elements = tokens * heads * rotary_width
    tensor_bytes = elements * dtype_bytes(dtype)
    coefficient_bytes = 2 * tokens * rotary_width * dtype_bytes("fp32")
    operations = {
        "multiply": 2 * elements, "add": elements,
        "cast": 2 * elements, "interleave_permutation": elements,
        "odd_lane_sign_change": elements // 2,
        "coefficient_broadcast": 2 * tokens * max(0, heads - 1) * rotary_width,
    }
    view = {
        "kind": "inplace_partial_rotary", "native_kind": NATIVE_KIND,
        "shape": (tokens, 1, heads, head_dim),
        "coefficient_shape": (tokens, 1, 1, rotary_width),
        "dtype": dtype, "coefficient_dtype": "fp32", "layout": "ND",
        "partial_slice": (head_dim - rotary_width, head_dim),
        "rotary_mode": "interleave", "inverse": bool(inverse),
        "sine_sign": "caller_negated" if inverse else "caller_forward",
    }
    metadata = (
        ("native_tensor_view", view), ("native_source", SOURCE),
        ("partial_rotary_work", {
            "rotated_elements": elements, "untouched_nope_elements": tokens * heads * (head_dim - rotary_width),
            "input_slice_bytes": tensor_bytes, "output_slice_bytes": tensor_bytes,
            "coefficient_bytes": coefficient_bytes, "vector_lane_operations": operations,
            "coefficient_load_scope": "once_per_token_shared_across_heads",
            "coefficient_sharing_is_analytic_assumption": True,
            "native_owner_split_and_reload_not_selected": True,
            "ub_permutation_and_broadcast_are_vector_lane_work_not_hbm": True,
            "caller_negated_sine_preparation_outside_this_invocation": bool(inverse),
        }),
        ("implementation_assumption", "source_partial_slice_analytic_resource_work"),
        ("native_implementation", {
            "native_kind": NATIVE_KIND, "selection": IMPLEMENTATION,
            "conditions": {"scope": "one_inplace_partial_rotation",
                           "native_launch_count": 1, "capture_scalar_flags_proven": False},
        }),
    )
    result = builder.work(
        name, deps, (tokens, heads, head_dim), family="rotary", dtype=dtype,
        vector=sum(operations.values()), read=tensor_bytes + coefficient_bytes,
        write=tensor_bytes, metadata=metadata,
    )
    # Preserve the caller's invocation across generic fusion passes. This
    # boundary declares one kernel; its timing remains analytic resource work.
    builder.ops[-1] = replace(builder.ops[-1], source_boundary="kernel")
    return result
