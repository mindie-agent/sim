"""Common compressed-cache producer, including its normalization and rotation.

[Role] Compose the same semantic compressor for attention and index keys.
[Invariants] Valid completed rows differ from native reserved output capacity.
  Parameter precision is declared independently of captured tensor descriptors.
  A C1 layer has no gated pooling: no ``wgate`` and no ``pool`` exist there.
  The native ``aclnnCompressor`` reads the hidden state once and conserves the
  complete Cube FLOPs and Vector Ops of the members it replaces.
  Cache stores and optional index-key Hadamard/quantization stay outside.
[Timeouts & Locks] Finite arithmetic over request views; no I/O or locks.
[Side-Effect Policy] Adds operators to the caller's builder only.
"""
from dataclasses import dataclass, replace

from mindie_sim.schema.common import dtype_bytes

from .mechanism import Work
from .physical_recipe_base import Phase, io_weight, tiled_phased_program
from .structure_groups import declare_group, source_reference


RECIPE = "aclnn_compressor"
NATIVE_KIND = "aclnnCompressor"
COMPRESSION_RECIPE = "compressor_bf16_runtime"
POOL_VECTOR_OPS_PER_ELEMENT = 6
NORM_VECTOR_OPS_PER_ELEMENT = 5
ROTARY_VECTOR_OPS_PER_ELEMENT = 6
FP32_BYTES = int(dtype_bytes("fp32"))
PROJECTION_CUBE_SHARE = 1.0
POOL_VECTOR_SHARE = 0.6
NORM_VECTOR_SHARE = 0.3
ROTARY_VECTOR_SHARE = 0.1
PIPELINE_SLOTS = 2
SOURCE = source_reference("vllm_ascend/models/deepseek_v4/compressor.py",
                          "Compressor.forward")


@dataclass(frozen=True)
class CompressorRecipe:
    """One submitted kernel: projection, gated pool, RMSNorm and RoPE.

    The phases partition the declared plan work, so the fused operator's Cube
    FLOPs, Vector Ops and HBM bytes stay exactly the conserved member work.
    """

    slots: int = PIPELINE_SLOTS

    def lower(self, context):
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        phases = (
            Phase("hidden_load", {"hbm": read_share, "mte2": 1.0}),
            Phase("project", {"cube": PROJECTION_CUBE_SHARE,
                              "vector": POOL_VECTOR_SHARE}, after=("hidden_load",)),
            Phase("pool_norm_rope", {"vector": NORM_VECTOR_SHARE
                                              + ROTARY_VECTOR_SHARE},
                  after=("project",)),
            Phase("store", {"hbm": write_share, "mte3": 1.0},
                  after=("pool_norm_rope",)),
        )
        return tiled_phased_program(context, phases, slots=self.slots)


def build(op, geometry):
    from .mechanism import OperatorPlan
    del geometry
    matrix = {op.dtype: op.flops} if op.flops else {}
    return OperatorPlan(
        Work(matrix, {"hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
                      "mte3": op.mte3_bytes, "vector": op.vector_ops}),
        CompressorRecipe().lower,
        metadata={"physical_class": type(CompressorRecipe()).__name__,
                  "stage_basis": "declared_phases_partition_conserved_member_work"},
    )


def recipes():
    from .mechanism import PlannedRecipe
    return {RECIPE: PlannedRecipe(build)}


def _native_compiler(b, *, name, source, tokens, padded_tokens, valid_rows, width, plan, model,
                     return_latent, do_rope):
    """One fused operator whose bytes and arithmetic are the member totals.

    The projection covers the padded launch rows exactly as the unfused
    ``b.matrix`` does, while the pooled/normalized output covers valid rows.
    """
    expansion, hidden = plan.compressor_width, model.hidden_size
    activation = model.activation_dtype
    activation_bytes = dtype_bytes(activation)
    gated = plan.compress_ratio > 1
    projected_rows = padded_tokens * expansion * width
    flops = 2 * projected_rows * hidden * (2 if gated else 1)
    vector = (POOL_VECTOR_OPS_PER_ELEMENT * tokens * expansion * width if gated else 0)
    vector += NORM_VECTOR_OPS_PER_ELEMENT * valid_rows * width
    if do_rope:
        vector += ROTARY_VECTOR_OPS_PER_ELEMENT * valid_rows * plan.qk_rope_head_dim
    state_rows = min(tokens, plan.compressor_state_rows)
    hidden_read = padded_tokens * hidden * activation_bytes
    weight_read = b.quant.matrix_bytes(expansion * width, hidden, COMPRESSION_RECIPE)
    if gated:
        weight_read *= 2
    # The rolling state and APE tables exist only on the gated (C>=4) path.
    state_read = state_rows * expansion * width * FP32_BYTES if gated else 0
    state_write = state_rows * expansion * width * FP32_BYTES if gated else 0
    ape_read = plan.compress_ratio * expansion * width * FP32_BYTES if gated else 0
    norm_weight_read = width * dtype_bytes(plan.compressor_norm_dtype)
    rope_table_read = (valid_rows * plan.qk_rope_head_dim * FP32_BYTES * 2
                       if do_rope else 0)
    latent_write = valid_rows * width * activation_bytes
    rotated_write = latent_write
    # ``return_latent`` keeps the pre-RoPE normalized latent visible to the
    # compressed indexer; with no rotary tail both consumers are the same
    # value and it is written once.
    written = latent_write + (rotated_write if (return_latent and do_rope) else 0)
    b.work(
        name + "_compress", (source,), (valid_rows, width), family="compressed_attention",
        dtype=activation, recipe=RECIPE, flops=flops, vector=vector,
        read=hidden_read + weight_read + state_read + ape_read
             + norm_weight_read + rope_table_read,
        write=written + state_write,
        metadata=(("native_kind", NATIVE_KIND),
                  ("native_source", SOURCE),
                  ("fused_members", 2 if gated else 1),
                  ("gated_pool", gated),
                  ("do_rope", do_rope),
                  ("state_rows", plan.compressor_state_rows),
                  ("state_dtype", "fp32"), ("ape_dtype", "fp32"),
                  ("valid_completed_rows", valid_rows),
                  ("return_latent", bool(return_latent)),
                  ("hidden_read_once", True),
                  ("norm_dtype", plan.compressor_norm_dtype),
                  ("compression_ratio", plan.compress_ratio)))
    if return_latent:
        materialized = b.ops[-1]
        b.ops[-1] = replace(
            materialized, returns=("latent",),
            metadata=materialized.metadata + (
                ("latent_shape", (valid_rows, width)),
                ("latent_dtype", activation),
                ("latent_write_bytes", latent_write),))
    declared = b.ops[-1].op_id
    declare_group(
        b.ops, key=b.prefix + "." + name + "or", role="compressor",
        family="compressed_attention", members=(declared,), entries=(declared,),
        anchor=declared,
        view={"tokens": tokens, "hidden": hidden, "width": width,
              "coff": expansion, "ratio": plan.compress_ratio,
              "output_valid_rows": valid_rows, "rope_width": plan.qk_rope_head_dim,
              "dtype": activation, "weight_dtype": "bf16",
              "norm_dtype": plan.compressor_norm_dtype, "state_dtype": "fp32",
              "ape_dtype": "fp32", "rope_dtype": "fp32",
              "gated_pool": gated},
        implementation={"native_kind": NATIVE_KIND,
                        "conditions": {"scope": "projection_gate_pool_norm_rope_state",
                                       "cache_store_external": True,
                                       "hidden_read_once": True,
                                       "norm_parameter_dtype": plan.compressor_norm_dtype}},
        source=SOURCE,
        coverage={"kind": "complete_subgraph",
                  "limitations": ["native_output_capacity_not_equal_valid_completed_rows",
                                  "native_dispatch_and_scalar_flags_not_observed"]},
    )
    return (declared, declared) if return_latent else declared


def compressor(b, *, name, source, tokens, padded_tokens, valid_rows, width, plan, model,
               return_latent=False, fused=False):
    expansion, hidden = plan.compressor_width, model.hidden_size
    gated = plan.compress_ratio > 1
    do_rope = plan.mla_do_rope
    if fused:
        return _native_compiler(
            b, name=name, source=source, tokens=tokens, padded_tokens=padded_tokens,
            valid_rows=valid_rows, width=width, plan=plan, model=model,
            return_latent=return_latent, do_rope=do_rope)
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
            write_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype),
            metadata=(("state_dtype", "fp32"), ("ape_dtype", "fp32"),
                      ("valid_completed_rows", valid_rows)),
        )
    normalized = b.vector(
        name + "_norm", (pooled,), valid_rows * width, NORM_VECTOR_OPS_PER_ELEMENT,
        family="rmsnorm", shape=(valid_rows, width),
        read_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype)
                            + width * dtype_bytes(plan.compressor_norm_dtype),
        write_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype),
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
                width * dtype_bytes(model.activation_dtype)
                + plan.qk_rope_head_dim * FP32_BYTES * 2),
            write_bytes_override=valid_rows * width * dtype_bytes(model.activation_dtype),
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
              "dtype": model.activation_dtype, "weight_dtype": "bf16",
              "norm_dtype": plan.compressor_norm_dtype, "state_dtype": "fp32",
              "ape_dtype": "fp32", "rope_dtype": "fp32", "gated_pool": gated},
        implementation={"native_kind": "Compressor",
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
