"""Common compressed-cache producer, including its normalization and rotation.

[Role] Compose the same semantic compressor for attention and index keys.
[Invariants] Valid completed rows differ from native reserved output capacity.
  Parameter precision is declared independently of captured tensor descriptors.
  A C1 layer has no gated pooling: no ``wgate`` and no ``pool`` exist there.
  Compressor native boundaries dispatch by ``plan.compressor_kind``: gated C2
  ``gated_ring`` lowers to its FP32 projections and Triton kernels
  (``ZerosLike``, ``_pool_kernel``, ``_cache_update_kernel``) with token-aligned
  pool/norm writes, while ``weighted_pool`` (DeepSeek-V4) maps to the custom
  ``torch.ops._C_ascend.compressor`` (``aclnnCompressor``) boundary.
  Cache stores and optional index-key Hadamard/quantization stay outside.
[Timeouts & Locks] Finite arithmetic over request views; no I/O or locks.
[Side-Effect Policy] Adds operators to the caller's builder only.
"""
from collections.abc import Sequence
from typing import Literal

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.structure_groups import declare_group, source_reference

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_REV = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/models/deepseek_v4/compressor.py",
        source_sha256="757e21d9e95a1bdc19c42921939830d8ba8d210cea635b7e5a694ce351c7a5ca",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/models/deepseek_v41/compressor.py",
        source_sha256="b5bda1c7a95952a6271effcd7b2af99110a6b75129d9c232df44f842ec1ab5e1",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/compressor/compressor_triton.py",
        source_sha256="3a4dc00693c7350d0da1bf85125f62d5fd25ce440f403d72bce2e7a13d740474",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=_VLLM_ASCEND_REV,
        relative_path="vllm_ascend/attention/dsa_v41.py",
        source_sha256="1da24c133d0fd473e367aea28419dabd7f42d5e90e4a9215c35e787c182cc195",
    ),
)

COMPRESSION_RECIPE = "compressor_bf16_runtime"
COMPRESSION_FP32_RECIPE = "compressor_fp32_runtime"
GATED_RING_COMPRESSOR = "gated_ring"
WEIGHTED_POOL_COMPRESSOR = "weighted_pool"
V4_COMPRESSOR_NATIVE_KIND = "aclnnCompressor"
V4_COMPRESSOR_NATIVE_OP = "torch.ops._C_ascend.compressor"
CAST_VECTOR_OPS_PER_ELEMENT = 1
ZERO_FILL_VECTOR_OPS_PER_ELEMENT = 1
POOL_VECTOR_OPS_PER_ELEMENT = 6
NORM_VECTOR_OPS_PER_ELEMENT = 5
ROTARY_VECTOR_OPS_PER_ELEMENT = 6
KV_SCORE_PAIR_STREAMS = 2
ROPE_TRIG_TABLES = 2
FP32_BYTES = int(dtype_bytes("fp32"))
TRITON_C2_COMPRESSOR_KERNELS: tuple[str, ...] = (
    "ZerosLike",
    "_pool_kernel",
    "_cache_update_kernel",
)
SOURCE = source_reference("vllm_ascend/models/deepseek_v4/compressor.py",
                          "Compressor.forward")
V41_SOURCE = source_reference("vllm_ascend/models/deepseek_v41/compressor.py",
                              "DeepseekV41Compressor.pool_projected")


def recipes():
    """No monolithic CANN compressor recipe exists (real C2 uses Triton kernels)."""
    return {}


def _compressor_weight_dtype(model, recipe: str = COMPRESSION_RECIPE) -> str:
    quant = getattr(model, "quant", None)
    if quant is not None and hasattr(quant, "format_for"):
        fmt = str(quant.format_for(recipe) or "").strip()
        if fmt:
            return fmt
    if quant is not None and hasattr(quant, "for_category"):
        cat = quant.for_category(recipe)
        fmt = str(getattr(cat, "weight_format", "") or "").strip()
        if fmt:
            return fmt
    act = str(getattr(model, "activation_dtype", "") or "").strip()
    if not act:
        raise ModelingError("compressor_weight_dtype_unresolved")
    return act


def _resolve_compressor_kind(plan) -> str:
    kind = str(getattr(plan, "compressor_kind", "") or "").strip()
    return kind or WEIGHTED_POOL_COMPRESSOR


def _ring_update_rows(
    tokens: int, state_rows: int, tokens_by_view: Sequence[int] | None = None,
) -> int:
    if tokens_by_view is not None:
        if state_rows > 0:
            return sum(min(max(0, int(count)), state_rows) for count in tokens_by_view)
        return sum(max(0, int(count)) for count in tokens_by_view)
    return min(tokens, state_rows) if state_rows > 0 else tokens


def _compressor_implementation(
    *, c2_gated_ring: bool, gated: bool, comp_kind: str, norm_weight_dtype: str,
) -> tuple[dict, dict]:
    conditions = {
        "scope": "projection_gate_pool_norm_rope_state",
        "cache_store_external": True,
        "gated_pool": gated,
        "norm_parameter_dtype": norm_weight_dtype,
    }
    if c2_gated_ring:
        return (
            {
                "native_kind": "_pool_kernel",
                "compressor_kind": comp_kind,
                "triton_c2_kernels": TRITON_C2_COMPRESSOR_KERNELS,
                "conditions": conditions,
            },
            V41_SOURCE,
        )
    if gated:
        return (
            {
                "native_kind": V4_COMPRESSOR_NATIVE_KIND,
                "native_op": V4_COMPRESSOR_NATIVE_OP,
                "compressor_kind": comp_kind,
                "triton_c2_kernels": (),
                "conditions": conditions,
            },
            SOURCE,
        )
    return (
        {
            "native_kind": "compress_values",
            "compressor_kind": comp_kind,
            "triton_c2_kernels": (),
            "conditions": conditions,
        },
        V41_SOURCE if comp_kind == GATED_RING_COMPRESSOR else SOURCE,
    )


def compressor(b, *, name, source, tokens, padded_tokens, valid_rows, width, plan, model,
               return_latent=False, fused=False, arch: str = "A5",
               tokens_by_view: Sequence[int] | None = None):
    expansion, hidden = plan.compressor_width, model.hidden_size
    gated = plan.compress_ratio > 1
    do_rope = plan.mla_do_rope
    comp_kind = _resolve_compressor_kind(plan)
    c2_gated_ring = gated and comp_kind == GATED_RING_COMPRESSOR
    if fused:
        # Fail closed against sealed ProbedAbiRegistry: no monolithic CANN
        # aclnnCompressor exists in shipping CANN packages.
        arch_norm = str(arch or "").strip().upper()
        if arch_norm not in {"A2", "A3", "A5"} or not default_abi_registry().is_shipping_available(
            V4_COMPRESSOR_NATIVE_KIND, arch_norm
        ):
            raise ModelingError(f"compressor_fused_abi_unavailable:{V4_COMPRESSOR_NATIVE_KIND}:{arch_norm or 'unknown'}")
    act_dtype = str(getattr(model, "activation_dtype", "") or "").strip()
    if not act_dtype:
        raise ModelingError("compressor_activation_dtype_unresolved")
    act_bytes = dtype_bytes(act_dtype)
    proj_recipe = COMPRESSION_FP32_RECIPE if c2_gated_ring else COMPRESSION_RECIPE
    weight_dtype = _compressor_weight_dtype(model, proj_recipe)
    # V4.1 slices off capture-bucket padding before hidden_states.float()/wkv/wgate (dsa_v41.py:433).
    proj_tokens = tokens if comp_kind == GATED_RING_COMPRESSOR else padded_tokens
    proj_source = source
    input_cast = None
    if c2_gated_ring:
        input_cast = b.vector(
            name + "_input_cast", (source,), proj_tokens * hidden,
            CAST_VECTOR_OPS_PER_ELEMENT,
            family="cast", dtype="fp32", shape=(proj_tokens, hidden),
            read_bytes_override=proj_tokens * hidden * act_bytes,
            write_bytes_override=proj_tokens * hidden * FP32_BYTES,
            metadata=(("input_dtype", act_dtype), ("output_dtype", "fp32")),
        )
        proj_source = input_cast
        proj_kwargs = {
            "category": proj_recipe,
            "activation_dtype": "fp32",
            "output_dtype": "fp32",
            "metadata": (("compute_format", "fp32"),),
        }
    else:
        proj_kwargs = {"category": proj_recipe}
    projection = b.matrix(
        name + "_values", (proj_source,), proj_tokens, expansion * width, hidden,
        **proj_kwargs)
    pooled = projection
    gates = None
    ring_update = None
    if gated:
        gates = b.matrix(
            name + "_gates", (proj_source,), proj_tokens, expansion * width, hidden,
            **proj_kwargs)
        state_rows = _ring_update_rows(tokens, plan.compressor_state_rows, tokens_by_view)
        state_bytes = state_rows * KV_SCORE_PAIR_STREAMS * expansion * width * FP32_BYTES
        if c2_gated_ring:
            completed_rows = max(0, int(valid_rows))
            pool_read_bytes = (
                completed_rows * plan.compress_ratio * KV_SCORE_PAIR_STREAMS * expansion * width * FP32_BYTES
                if completed_rows > 0 else 0
            )
            pool_write_bytes = (tokens + completed_rows) * width * act_bytes
            pool_vec_ops = (
                tokens * width * ZERO_FILL_VECTOR_OPS_PER_ELEMENT
                + completed_rows * plan.compress_ratio * expansion * width * POOL_VECTOR_OPS_PER_ELEMENT
            )
            pooled = b.vector(
                name + "_pool", (projection, gates), tokens * expansion * width,
                0,
                vector_ops_override=pool_vec_ops,
                read_bytes_override=pool_read_bytes,
                write_bytes_override=pool_write_bytes,
                metadata=(("state_dtype", "fp32"),
                          ("valid_completed_rows", completed_rows),
                          ("token_aligned_rows", tokens),
                          ("native_kinds", ("ZerosLike", "_pool_kernel")),
                          ("kernel", "_pool_kernel")),
            )
            ring_update = b.vector(
                name + "_ring_cache_update", (projection, gates, pooled),
                state_rows * KV_SCORE_PAIR_STREAMS * expansion * width, 0,
                family="cache_store", dtype="fp32",
                shape=(state_rows, KV_SCORE_PAIR_STREAMS * expansion * width),
                read_bytes_override=state_bytes,
                write_bytes_override=state_bytes,
                metadata=(("state_dtype", "fp32"), ("ring_rows", state_rows),
                          ("native_kind", "_cache_update_kernel"),
                          ("kernel", "_cache_update_kernel")),
            )
        else:
            pooled = b.vector(
                name + "_pool", (projection, gates), tokens * expansion * width,
                POOL_VECTOR_OPS_PER_ELEMENT,
                read_bytes_override=(tokens + state_rows)
                                    * expansion * width * FP32_BYTES * KV_SCORE_PAIR_STREAMS
                                    + plan.compress_ratio * expansion * width * FP32_BYTES,
                write_bytes_override=valid_rows * width * act_bytes + state_bytes,
                metadata=(("state_dtype", "fp32"), ("ape_dtype", "fp32"),
                          ("state_write_bytes", state_bytes),
                          ("valid_completed_rows", valid_rows),
                          ("native_kind", V4_COMPRESSOR_NATIVE_KIND),
                          ("native_op", V4_COMPRESSOR_NATIVE_OP)),
            )
    norm_rows = tokens if c2_gated_ring else valid_rows
    norm_weight_dtype = "fp32" if (gated and not c2_gated_ring) else plan.compressor_norm_dtype
    norm_deps = (pooled, ring_update) if ring_update is not None else (pooled,)
    normalized = b.vector(
        name + "_norm", norm_deps, norm_rows * width, NORM_VECTOR_OPS_PER_ELEMENT,
        family="rmsnorm", shape=(norm_rows, width),
        read_bytes_override=norm_rows * width * act_bytes
                            + (width * dtype_bytes(norm_weight_dtype) if norm_rows else 0),
        write_bytes_override=norm_rows * width * act_bytes,
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("norm_weight_dtype", norm_weight_dtype),
                  ("norm_eps", model.norm_eps)),
    )
    rotary_rows = tokens if c2_gated_ring else valid_rows
    rotated = normalized
    if do_rope:
        rope_dim = plan.qk_rope_head_dim
        rotated = b.vector(
            name + "_rotary", (normalized,), rotary_rows * rope_dim,
            ROTARY_VECTOR_OPS_PER_ELEMENT,
            family="rotary", shape=(rotary_rows, width),
            read_bytes_override=rotary_rows * (
                rope_dim * act_bytes
                + rope_dim * FP32_BYTES * ROPE_TRIG_TABLES),
            write_bytes_override=rotary_rows * rope_dim * act_bytes,
        )
    raw_members = [op_id for op_id in (input_cast, projection, gates, pooled, ring_update, normalized, rotated)
                   if op_id is not None]
    members = tuple(dict.fromkeys(raw_members))
    entries = (input_cast,) if input_cast is not None else ((projection, gates) if gated else (projection,))
    implementation, group_source = _compressor_implementation(
        c2_gated_ring=c2_gated_ring, gated=gated,
        comp_kind=comp_kind, norm_weight_dtype=norm_weight_dtype,
    )
    declare_group(
        b.ops, key=b.prefix + "." + name + "or", role="compressor",
        family="compressed_attention",
        members=members,
        entries=entries, anchor=rotated,
        view={"tokens": tokens, "hidden": hidden, "width": width,
              "coff": expansion, "ratio": plan.compress_ratio,
              "compressor_kind": comp_kind,
              "output_valid_rows": valid_rows, "rope_width": plan.qk_rope_head_dim,
              "dtype": act_dtype, "weight_dtype": weight_dtype,
              "norm_dtype": norm_weight_dtype, "state_dtype": "fp32",
              "ape_dtype": "fp32", "has_ape": gated and not c2_gated_ring,
              "rope_dtype": "fp32", "gated_pool": gated},
        implementation=implementation,
        source=group_source,
        coverage={"kind": "complete_subgraph",
                  "limitations": ["native_output_capacity_not_equal_valid_completed_rows",
                                  "native_dispatch_and_scalar_flags_not_observed"]},
    )
    # Some callers branch here: index projection consumes normalized pre-RoPE
    # latent, while cache insertion consumes the separately rotated value.
    return (normalized, rotated) if return_latent else rotated
