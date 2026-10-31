"""Explicit packed-cache INT8 sparse MLA adaptation on the shared pipeline.

[Role] Expand declared per-128 FP32-scaled INT8 rows before BF16 QK/PV.
[Invariants] K and V alias one packed cache. A row contains 512 INT8 values,
  four FP32 scales and zero or 64 BF16 rotary values. UB padding is local.
  The arch22 source admits rope64; rope0 is an analytic adaptation. Query
  ownership spans the batch, never a separate whole-call charge per sequence.
[Timeouts & Locks] Pure finite phase planning; common bounded query pipeline.
[Side-Effect Policy] No captures, timing fits, hardware operations or I/O.
"""
from dataclasses import replace
from functools import partial

from mindie_sim.schema.common import ModelingError
from .indexed_attention import _phases, build_pipeline
from .mechanism import PlannedRecipe, Work


VECTOR_BUFFER_BYTES = (64 + 32 + 32 + 4 + 32 + 8 + 8) * 1024 + (3 * 3 + 2) * 512


SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "caller": "vllm_ascend/attention/utils.py:69-80; attention/sfa_v1.py:1215-1276",
    "phase_reference": "csrc/attention/kv_quant_sparse_flash_attention/op_kernel/"
                       "kv_quant_sparse_flash_attention_service_vector_mla.h:652-851",
    "phase_sha256": "3eed5ab746e056319bf873a488a3306cffbabe8dd19f70294da85685f4e3cfd3",
    "scope": "arch22_packed_latent512_rope64; rope0_is_analytic_adaptation",
    "capture_binary": "not_established",
}


def recipe(*, scratch_policy, compact=True, operand_modes=True):
    """Select explicitly; this recipe never changes ordinary BF16 defaults."""
    return PlannedRecipe(partial(build, scratch_policy=scratch_policy,
                                 compact=compact, operand_modes=operand_modes))


def phases(keys, heads, latent, rope, *, first, last, scratch_policy,
           operand_modes=False):
    """Alias-aware gather plus element work of the declared conversion chain."""
    common = _phases(keys, heads, latent, rope, first=first, last=last,
                     scratch_policy=scratch_policy, operand_modes=operand_modes)
    # DataCopyPad transfers valid bytes, pads only the local destination. The
    # retained kernel's arbitrary-index/page reads are accounted as two INT32
    # fields per selected row; dependent scalar response remains unmodeled.
    packed_bytes = keys * (latent + latent // 128 * 4 + rope * 2)
    gather = Work(resources={"hbm": packed_bytes + keys * 8,
                             "mte2_aiv": packed_bytes + keys * 8})
    # INT8 -> FP16 -> FP32, multiply scale, FP32 -> BF16, ND -> NZ copy.
    # A scale is broadcast to eight FP32 lanes before repeated vector Mul.
    # These are typed element service counts, not a measured instruction CPI.
    # Rotary data is already BF16. Its source Copy to NZ adds a lane move,
    # not an INT8 dequantization pass.
    convert = Work(resources={"vector": keys * (5 * latent + latent // 128 * 8 + rope)})
    return (("gather", gather), ("dequantize", convert), *common[1:])


def phase_unit_weights(name, key, rope=0):
    if name not in {"gather", "dequantize", "publish"}:
        return {}
    keys = key[0]
    # MergeKv assigns pairs: ceil(ceil(rows/2)/2) pairs to AIV0.
    first = min(keys, ((keys + 3) // 4) * 2)
    if name == "publish":
        # Each paired AIV publishes its own valid rows plus one 512-byte
        # valid-size descriptor, including the sibling with no KV payload.
        width = 512 + rope
        total = keys * width * 2 + 1024
        return {"mte3": ((first * width * 2 + 512) / total,
                         ((keys - first) * width * 2 + 512) / total)}
    role = "mte2_aiv" if name == "gather" else "vector"
    return {role: (first / keys, (keys - first) / keys)}


def phase_unit_weights_rope64(name, key):
    return phase_unit_weights(name, key, rope=64)


def build(op, geometry, *, scratch_policy, compact=True, operand_modes=True):
    shape = dict(op.metadata).get("indexed_attention_geometry")
    if (not shape or shape["query_dtype"] != "bf16" or shape["cache_dtype"] != "int8"
            or shape["latent"] != 512 or shape["rope"] not in {0, 64}
            or shape["window"] or shape["compression_ratio"] != 1 or shape["cp"] != 1
            or shape.get("sparse_block_size", 1) != 1 or not 1 <= shape["heads"] <= 64):
        raise ModelingError("packed_int8_indexed_attention_requires_bf16_q_int8_latent512_rope0_or64")
    if scratch_policy not in {"resident", "offchip"}:
        raise ModelingError("indexed_attention_scratch_policy_required")
    if any(geometry.resource_units.get(role) != geometry.compute_units * ratio
           for role, ratio in (("vector", 2), ("mte2_aiv", 2), ("mte3", 2),
                               ("mte2_aic", 1), ("mte1", 1), ("fixpipe", 1))):
        raise ModelingError("packed_int8_indexed_attention_requires_paired_aic_aiv_pipes")
    if (geometry.buffer_units.get("ub") != geometry.compute_units * 2
            or geometry.buffers.get("ub", 0) < VECTOR_BUFFER_BYTES):
        raise ModelingError("packed_int8_indexed_attention_requires_declared_vector_scratch")
    plan = build_pipeline(
        shape, geometry, scratch_policy=scratch_policy, compact=compact,
        operand_modes=operand_modes, phase_factory=phases,
        source_items=tuple(SOURCE.items()), paired_vector_stages=("dequantize",),
        vector_buffer_bytes=VECTOR_BUFFER_BYTES,
        phase_unit_weights=phase_unit_weights_rope64 if shape["rope"] else phase_unit_weights)
    row_bytes = 512 + 16 + shape["rope"] * 2
    return replace(plan, metadata={
        "kind": "indexed_attention_packed_int8",
        "coverage": "source_phases_declared_dispatch" if shape["rope"] else "analytic_adaptation",
        "source_execution_domain": "mixed", "source": SOURCE,
        "packed_cache": {"payload_bytes": 512, "scale_bytes": 16,
                         "scale_dtype": "fp32", "group_size": 128,
                         "rotary_bytes": shape["rope"] * 2,
                         "rotary_dtype": "bf16", "row_bytes": row_bytes,
                         "local_padded_row_bytes": (row_bytes + 31) // 32 * 32,
                         "key_value_alias": True},
        "kernel_schedule": "declared_key512_slots3_contiguous_query_owners",
        "vector_buffer_bytes_per_aiv": VECTOR_BUFFER_BYTES,
        "buffer_basis": "retained_InitBuffers_preload3; no_rope_adaptation_keeps_allocation",
        "unmodeled_features": [
            "captured_kernel_dispatch_and_binary", "captured_valid_index_values",
            "scalar_index_and_page_lookup_response", "cross_core_flag_latency",
            "external_cache_eviction", "conversion_instruction_issue_rate"],
        "abstraction": "shared_finite_pipeline_with_fluid_compact_contention",
    })
