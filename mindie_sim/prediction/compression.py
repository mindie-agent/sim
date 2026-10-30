"""Common compressed-cache producer, including its normalization and rotation.

[Role] Compose the same semantic compressor for attention and index keys.
[Invariants] Valid completed rows differ from native reserved output capacity.
  Parameter precision is declared independently of captured tensor descriptors.
  Cache stores and optional index-key Hadamard/quantization stay outside.
[Timeouts & Locks] Finite arithmetic over request views; no I/O or locks.
[Side-Effect Policy] Adds operators to the caller's builder only.
"""
from mindie_sim.schema.common import dtype_bytes

from .structure_groups import declare_group, source_reference


def compressor(b, *, name, source, tokens, padded_tokens, valid_rows, width, plan, model,
               return_latent=False):
    expansion, hidden = plan.compressor_width, model.hidden_size
    projection = b.matrix(
        name + "_values", (source,), padded_tokens, expansion * width, hidden,
        category="compressor_bf16_runtime")
    gates = b.matrix(
        name + "_gates", (source,), padded_tokens, expansion * width, hidden,
        category="compressor_bf16_runtime")
    pooled = b.vector(
        name + "_pool", (projection, gates), tokens * expansion * width, 6,
        read_bytes_override=(tokens + min(tokens, plan.compressor_state_rows))
                            * expansion * width * 8 + plan.compress_ratio * expansion * width * 4,
        write_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype),
        metadata=(("state_dtype", "fp32"), ("ape_dtype", "fp32"),
                  ("valid_completed_rows", valid_rows)),
    )
    normalized = b.vector(
        name + "_norm", (pooled,), valid_rows * width, 5,
        family="rmsnorm", shape=(valid_rows, width),
        read_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype)
                            + width * dtype_bytes(plan.compressor_norm_dtype),
        write_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype),
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("norm_weight_dtype", plan.compressor_norm_dtype),
                  ("norm_eps", model.norm_eps)),
    )
    rotated = b.vector(
        name + "_rotary", (normalized,), valid_rows * plan.qk_rope_head_dim, 6,
        family="rotary", shape=(valid_rows, width),
        read_bytes_override=valid_rows * (
            width * dtype_bytes(model.activation_dtype) + plan.qk_rope_head_dim * 8),
        write_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype),
    )
    declare_group(
        b.ops, key=b.prefix + "." + name + "or", role="compressor",
        family="compressed_attention",
        members=(projection, gates, pooled, normalized, rotated),
        entries=(projection, gates), anchor=rotated,
        view={"tokens": tokens, "hidden": hidden, "width": width,
              "coff": expansion, "ratio": plan.compress_ratio,
              "output_valid_rows": valid_rows, "rope_width": plan.qk_rope_head_dim,
              "dtype": model.activation_dtype, "weight_dtype": "bf16",
              "norm_dtype": plan.compressor_norm_dtype, "state_dtype": "fp32",
              "ape_dtype": "fp32", "rope_dtype": "fp32"},
        implementation={"native_kind": "Compressor",
                        "conditions": {"scope": "projection_gate_pool_norm_rope_state",
                                       "cache_store_external": True,
                                       "norm_parameter_dtype": plan.compressor_norm_dtype}},
        source=source_reference("vllm_ascend/models/deepseek_v4/compressor.py", "Compressor.forward"),
        coverage={"kind": "complete_subgraph",
                  "limitations": ["native_output_capacity_not_equal_valid_completed_rows",
                                  "native_dispatch_and_scalar_flags_not_observed"]},
    )
    # Some callers branch here: index projection consumes normalized pre-RoPE
    # latent, while cache insertion consumes the separately rotated value.
    return (normalized, rotated) if return_latent else rotated
