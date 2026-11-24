"""Common compressed-cache producer, including its normalization and rotation.

[Role] Compose the same semantic compressor for attention and index keys.
[Invariants] Valid completed rows differ from native reserved output capacity.
  Parameter precision is declared independently of captured tensor descriptors.
  A C1 layer has no gated pooling: no ``wgate`` and no ``pool`` exist there.
  No monolithic CANN ``aclnnCompressor`` exists; gated C2 compression lowers to
  its real Triton/CANN kernels (``ZerosLike``, ``_pool_kernel``, and external
  ``_cache_update_kernel`` on store).
  Cache stores and optional index-key Hadamard/quantization stay outside.
[Timeouts & Locks] Finite arithmetic over request views; no I/O or locks.
[Side-Effect Policy] Adds operators to the caller's builder only.
"""
from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.prediction.structure_groups import declare_group, source_reference


COMPRESSION_RECIPE = "compressor_bf16_runtime"
POOL_VECTOR_OPS_PER_ELEMENT = 6
NORM_VECTOR_OPS_PER_ELEMENT = 5
ROTARY_VECTOR_OPS_PER_ELEMENT = 6
FP32_BYTES = int(dtype_bytes("fp32"))
TRITON_C2_COMPRESSOR_KERNELS: tuple[str, ...] = (
    "ZerosLike",
    "_pool_kernel",
    "_cache_update_kernel",
)
SOURCE = source_reference("vllm_ascend/models/deepseek_v4/compressor.py",
                          "Compressor.forward")


def recipes():
    """No monolithic CANN compressor recipe exists (real C2 uses Triton kernels)."""
    return {}


def _compressor_weight_dtype(model) -> str:
    quant = getattr(model, "quant", None)
    if quant is not None and hasattr(quant, "for_category"):
        cat = quant.for_category(COMPRESSION_RECIPE)
        fmt = str(getattr(cat, "weight_format", "") or "").strip()
        if fmt:
            return fmt
    act = str(getattr(model, "activation_dtype", "") or "").strip()
    if not act:
        raise ModelingError("compressor_weight_dtype_unresolved")
    return act


def compressor(b, *, name, source, tokens, padded_tokens, valid_rows, width, plan, model,
               return_latent=False, fused=False, arch: str = "A5"):
    expansion, hidden = plan.compressor_width, model.hidden_size
    gated = plan.compress_ratio > 1
    do_rope = plan.mla_do_rope
    if fused:
        # Fail closed against sealed ProbedAbiRegistry: no monolithic CANN
        # aclnnCompressor exists in shipping CANN packages.
        arch_norm = str(arch or "").strip().upper()
        if arch_norm not in {"A2", "A3", "A5"} or not default_abi_registry().is_shipping_available(
            "aclnnCompressor", arch_norm
        ):
            raise ModelingError(f"compressor_fused_abi_unavailable:aclnnCompressor:{arch_norm or 'unknown'}")
    act_dtype = str(getattr(model, "activation_dtype", "") or "").strip()
    if not act_dtype:
        raise ModelingError("compressor_activation_dtype_unresolved")
    act_bytes = dtype_bytes(act_dtype)
    weight_dtype = _compressor_weight_dtype(model)
    projection = b.matrix(
        name + "_values", (source,), padded_tokens, expansion * width, hidden,
        category=COMPRESSION_RECIPE)
    pooled = projection
    if gated:
        gates = b.matrix(
            name + "_gates", (source,), padded_tokens, expansion * width, hidden,
            category=COMPRESSION_RECIPE)
        pooled = b.vector(
            name + "_pool", (projection, gates), tokens * expansion * width,
            POOL_VECTOR_OPS_PER_ELEMENT,
            read_bytes_override=(tokens + min(tokens, plan.compressor_state_rows))
                                * expansion * width * FP32_BYTES * 2
                                + plan.compress_ratio * expansion * width * FP32_BYTES,
            write_bytes_override=valid_rows * width * act_bytes,
            metadata=(("state_dtype", "fp32"), ("ape_dtype", "fp32"),
                      ("valid_completed_rows", valid_rows),
                      ("native_kinds", ("ZerosLike", "_pool_kernel")),
                      ("kernel", "_pool_kernel")),
        )
    normalized = b.vector(
        name + "_norm", (pooled,), valid_rows * width, NORM_VECTOR_OPS_PER_ELEMENT,
        family="rmsnorm", shape=(valid_rows, width),
        read_bytes_override=valid_rows * width * act_bytes
                            + width * dtype_bytes(plan.compressor_norm_dtype),
        write_bytes_override=valid_rows * width * act_bytes,
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("norm_weight_dtype", plan.compressor_norm_dtype),
                  ("norm_eps", model.norm_eps)),
    )
    rotated = normalized
    if do_rope:
        rotated = b.vector(
            name + "_rotary", (normalized,), valid_rows * plan.qk_rope_head_dim,
            ROTARY_VECTOR_OPS_PER_ELEMENT,
            family="rotary", shape=(valid_rows, width),
            read_bytes_override=valid_rows * (
                width * act_bytes
                + plan.qk_rope_head_dim * FP32_BYTES * 2),
            write_bytes_override=valid_rows * width * act_bytes,
        )
    members = (projection, pooled, normalized, rotated)
    if gated:
        members = (projection, gates, pooled, normalized, rotated)
    declare_group(
        b.ops, key=b.prefix + "." + name + "or", role="compressor",
        family="compressed_attention",
        members=members,
        entries=(projection, gates) if gated else (projection,), anchor=rotated,
        view={"tokens": tokens, "hidden": hidden, "width": width,
              "coff": expansion, "ratio": plan.compress_ratio,
              "output_valid_rows": valid_rows, "rope_width": plan.qk_rope_head_dim,
              "dtype": act_dtype, "weight_dtype": weight_dtype,
              "norm_dtype": plan.compressor_norm_dtype, "state_dtype": "fp32",
              "ape_dtype": "fp32", "rope_dtype": "fp32", "gated_pool": gated},
        implementation={"native_kind": "_pool_kernel" if gated else "compress_values",
                        "triton_c2_kernels": TRITON_C2_COMPRESSOR_KERNELS if gated else (),
                        "conditions": {"scope": "projection_gate_pool_norm_rope_state",
                                       "cache_store_external": True,
                                       "gated_pool": gated,
                                       "norm_parameter_dtype": plan.compressor_norm_dtype}},
        source=SOURCE,
        coverage={"kind": "complete_subgraph",
                  "limitations": ["native_output_capacity_not_equal_valid_completed_rows",
                                  "native_dispatch_and_scalar_flags_not_observed"]},
    )
    # Some callers branch here: index projection consumes normalized pre-RoPE
    # latent, while cache insertion consumes the separately rotated value.
    return (normalized, rotated) if return_latent else rotated
