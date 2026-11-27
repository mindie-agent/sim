"""Observed attention and cache descriptor contracts.

[Role]
  Compare supported native attention/cache ABIs against independently
  declared active tensor descriptors, including mixed-quant sparse MLA.

[Invariants]
  * Active projections never identify phase, cache slot values, mask values
    or dispatch flags.
  * Paged cache descriptors never become dense TND tensors by dividing
    dimensions or assuming a model's head configuration.
  * Only source-backed arity, rank and row-byte geometries are compared.
  * PaMla membership alone grants no positive contract match.

[Timeouts & Locks]
  Pure bounded tensor-slot inspection; no waits or locks.

[Side-Effect Policy]
  Import-time declarations only; no I/O, model catalogue, hardware or
  simulation imports.
"""
from __future__ import annotations

from .vocabulary import DTYPES, native_tensor_contract, native_tensor_diagnostic
from .fusion import (
    SUBGRAPH_TYPES, subgraph_contract, subgraph_contract_scope,
)

FIA_COMPACT_INPUT_COUNT = 7
FIA_PAGED_INPUT_COUNTS = frozenset({7, 31})
FIA_PAGED_QUERY_RANK = 3
MLA_PAGED_TENSOR_RANK = 4
MLA_COMPACT_TABLE_SLOT = 4
PAGED_FIA_COMPACT_TABLE_SLOT = 6
PAGED_FIA_FULL_TABLE_SLOT = 14
MLA_COMPACT_REQUIRED_SLOTS = (0, 1, 2, 3, 4, 5, 6)
MLA_FULL_REQUIRED_SLOTS = (0, 1, 2, 6, 14, 24, 25)
PAGED_FIA_COMPACT_REQUIRED_SLOTS = (0, 1, 2, 6)
PAGED_FIA_FULL_REQUIRED_SLOTS = (0, 1, 2, 14)
SPARSE_ATTENTION_QUERY_RANK = 3
SPARSE_ATTENTION_ROPE_RANK = 3
SPARSE_ATTENTION_CACHE_RANK = 4

MQSMLA_INPUT_COUNTS = frozenset({7, 8, 10, 11, 12})
MQSMLA_OUTPUT_COUNTS = frozenset({1, 2})
MQSMLA_QUERY_RANK = 3
MQSMLA_CACHE_RANK = 4
MQSMLA_QUERY_WIDTH = 512
MQSMLA_ORIGINAL_ROW_BYTES = 544
MQSMLA_COMPRESSED_ROW_BYTES = 320
MQSMLA_FP8_ROW_BYTES = 608
MQSMLA_MXFP8_ROW_BYTES = 584
MQSMLA_NATIVE_INPUT_COUNT = 18
MQSMLA_NATIVE_RECORDS = {
    ("fp8_e4m3", MQSMLA_FP8_ROW_BYTES), ("fp8_e4m3", MQSMLA_MXFP8_ROW_BYTES),
    ("uint8", MQSMLA_FP8_ROW_BYTES), ("uint8", MQSMLA_MXFP8_ROW_BYTES),
    ("int8", MQSMLA_FP8_ROW_BYTES),
}
MQSMLA_COMPRESSED_INPUT_COUNT = 10
DSL_MQSMLA_NATIVE_KIND = "tilelang_dsl.mixed_quant_sparse_flash_mla"
DSL_MQSMLA_KINDS = frozenset({
    DSL_MQSMLA_NATIVE_KIND, "MqsmlaKernel___call__", "MqsmlaKernel",
})
ACLNN_MQSMLA_CANONICAL_KIND = "aclnnMixedQuantSparseFlashMla"
ACLNN_MQSMLA_KINDS = frozenset({
    ACLNN_MQSMLA_CANONICAL_KIND, "MixedQuantSparseFlashMla",
})
MQSMLA_KINDS = DSL_MQSMLA_KINDS | ACLNN_MQSMLA_KINDS
MLA_PROLOG_INPUT_ARITIES = frozenset({21})
MLA_PROLOG_OUTPUT_ARITIES = frozenset({7})


# Arity variants come from retained original A3/A5 rows, including explicit
# undefined slots. Compact GQA uses input 6 for the block table; the independent
# seven-input MLA ABI uses input 4 and retains query/key rotary tensors.
_ARITIES = {
    "FusedInferAttentionScore": ({6, 7, 31}, {2}),
    "ReshapeAndCacheNdKernel": ({5}, {2}),
    "ScatterPaKvCache": ({5, 8}, {2}),
    "KvRmsNormRopeCache": ({12}, {4}),
    "PagedCacheLoadNdKernel": ({7}, {2}),
    "RINGMLAPrefillBF16Kernel": ({15}, {2}),
    "SparseAttnSharedkv": ({14}, {2}),
    "ScatterNdUpdateSk": ({3}, {1}),
    "LightningIndexer": ({6}, {2}),
    "SparseFlashAttention": ({9}, {3}),
    "SparseFlashMla": ({18}, {2}),
    "KvQuantSparseFlashAttentionVllm": ({9}, {3}),
    "KvQuantSparseFlashAttention": ({9}, {1}),
    "MlaPrologV3": (MLA_PROLOG_INPUT_ARITIES, MLA_PROLOG_OUTPUT_ARITIES),
    "aclnnMlaPrologV4WeightNz": (MLA_PROLOG_INPUT_ARITIES, MLA_PROLOG_OUTPUT_ARITIES),
    "MlaPrologV4WeightNz": (MLA_PROLOG_INPUT_ARITIES, MLA_PROLOG_OUTPUT_ARITIES),
    "MlaPrologV4": (MLA_PROLOG_INPUT_ARITIES, MLA_PROLOG_OUTPUT_ARITIES),
    "ChunkGatedDeltaRuleFwd": ({9, 10}, {4}),
    "RecurrentGatedDeltaRule": ({9, 10}, {2}),
}
# Mixed-quant sparse MLA profile slots: original-only 7/8 inputs,
# compressed cache 10/11/12 inputs, one or two outputs.
_ARITIES.update({kind: (MQSMLA_INPUT_COUNTS, MQSMLA_OUTPUT_COUNTS) for kind in DSL_MQSMLA_KINDS})
_ARITIES.update({kind: ({MQSMLA_NATIVE_INPUT_COUNT}, {2}) for kind in ACLNN_MQSMLA_KINDS})
ATTENTION_CACHE_TYPES = frozenset(_ARITIES) | {"PaMlaAttention"}


def _dense_fia(inputs, outputs):
    # Full FIA input14 is the block table; the compact six-slot export has
    # Q/K/V, mask, q lengths and kv lengths. The seventh compact slot is paged.
    dense = len(inputs) == 6 or (
        len(inputs) == 31 and inputs[14] == ((), None, None))
    active = (*inputs[:3], outputs[0])
    if not dense or any(len(shape) != 3 or layout != "ND" for shape, _, layout in active):
        return None
    q, k, v, out = (slot[0] for slot in active)
    if (any(value <= 0 for shape in (q, k, v, out) for value in shape)
            or k[:2] != v[:2] or q[-1] != k[-1]
            or out != (q[0], q[1], v[-1])
            or len({slot[1] for slot in active}) != 1):
        return None
    return ("attention_active_qkv", "FusedInferAttentionScore", inputs[:3], outputs[:1])


def _paged_fia(inputs, outputs):
    # attention_v1._get_fia_params views (pages, block, KVheads, dim) as
    # (pages, block, packed_width). Preserve the packed width without dividing
    # it by the query width: GQA's KV-head count is a separate caller argument.
    if len(inputs) not in FIA_PAGED_INPUT_COUNTS:
        return None
    query, key, value, output = *inputs[:3], outputs[0]
    table = inputs[PAGED_FIA_COMPACT_TABLE_SLOT if len(inputs) == FIA_COMPACT_INPUT_COUNT
                  else PAGED_FIA_FULL_TABLE_SLOT]
    floating = {"bf16", "fp16"}
    if (not all(_nd_tensor(slot, FIA_PAGED_QUERY_RANK, floating)
                for slot in (query, key, value, output))
            or not _nd_tensor(table, 2, {"int32"})
            or key[0][:2] != value[0][:2] or output != query
            or len({slot[1] for slot in (query, key, value, output)}) != 1):
        return None
    tails = tuple((slot[0][1:], slot[1], slot[2]) for slot in (key, value))
    return ("attention_paged_active_qkv", "FusedInferAttentionScore",
            query, tails, output, table[0][0])


def _paged_mla_fia(inputs, outputs):
    # mla_v1's BNSD_NBSD path preserves native axes rather than flattening them
    # into the GQA TND contract. Slot 6 is a rotary cache, never a block table.
    if len(inputs) not in FIA_PAGED_INPUT_COUNTS:
        return None
    indices = (MLA_COMPACT_REQUIRED_SLOTS if len(inputs) == FIA_COMPACT_INPUT_COUNT
               else MLA_FULL_REQUIRED_SLOTS)
    query, key, value, lengths, table, rope, key_rope = (inputs[index] for index in indices)
    output = outputs[0]
    floating = {"bf16", "fp16"}
    active = (query, key, value, rope, key_rope, output)
    if (not all(_nd_tensor(slot, MLA_PAGED_TENSOR_RANK, floating) for slot in active)
            or len({slot[1] for slot in active}) != 1
            or not _nd_tensor(table, 2, {"int32"})
            or not _nd_tensor(lengths, 1, {"int64"})
            or key != value or key[0][1] != 1
            or key_rope[0][:3] != key[0][:3]
            or rope[0][:3] != query[0][:3]
            or rope[0][-1] != key_rope[0][-1]
            or query[0][-1] != key[0][-1]
            or query[0][2] != 1
            or table[0][0] != query[0][0] or lengths[0] != (query[0][0],)
            or output[0] != (query[0][1], query[0][0], query[0][2], query[0][3])):
        return None
    tails = tuple((slot[0][1:], slot[1], slot[2]) for slot in (key, value, key_rope))
    return ("attention_paged_mla_qk_rope", "FusedInferAttentionScore",
            (query, rope), tails, output, table[0][0])


def fia_block_table_slot(row):
    """Identify table geometry without demanding unrelated auxiliary metadata.

    This selector does not export a tensor contract. Each required active slot
    is validated; missing descriptors in unrelated optional slots remain unknown
    and may still prevent the full native/active comparison contract.
    """
    if row.kind != "FusedInferAttentionScore" or not row.input_shapes:
        return None
    arity = len(row.input_shapes)
    if arity not in FIA_PAGED_INPUT_COUNTS:
        return None
    def slot(direction, index):
        shapes = getattr(row, direction + "_shapes")
        dtypes = getattr(row, direction + "_dtypes")
        formats = getattr(row, direction + "_formats")
        if not shapes or any(index >= len(values) for values in (shapes, dtypes, formats)):
            return None
        return (shapes[index], DTYPES.get(dtypes[index], dtypes[index]), formats[index])
    output = slot("output", 0)
    if output is None:
        return None
    variants = ((MLA_COMPACT_TABLE_SLOT, MLA_COMPACT_REQUIRED_SLOTS, _paged_mla_fia),
                (PAGED_FIA_COMPACT_TABLE_SLOT, PAGED_FIA_COMPACT_REQUIRED_SLOTS,
                 _paged_fia)) if arity == FIA_COMPACT_INPUT_COUNT else (
                (PAGED_FIA_FULL_TABLE_SLOT, MLA_FULL_REQUIRED_SLOTS, _paged_mla_fia),
                (PAGED_FIA_FULL_TABLE_SLOT, PAGED_FIA_FULL_REQUIRED_SLOTS, _paged_fia))
    for table_index, required, validate in variants:
        inputs = [None] * arity
        for index in required:
            inputs[index] = slot("input", index)
        if all(inputs[index] is not None for index in required) and validate(inputs, (output,)):
            return table_index
    return None


def _reshape_cache(inputs, outputs):
    key, value, key_cache, value_cache, slots = inputs
    if (len(key[0]) != 3 or len(value[0]) != 3 or key[0] != value[0]
            or len(key_cache[0]) != 4 or len(value_cache[0]) != 4
            or key_cache[0][2:] != key[0][1:] or value_cache[0][2:] != value[0][1:]
            or slots[0] != (key[0][0],) or slots[1] not in {"int32", "int64"}
            or any(slot[2] != "ND" for slot in (*inputs, *outputs))
            or outputs != (key_cache, value_cache)
            or len({slot[1] for slot in (key, value, key_cache, value_cache)}) != 1):
        return None
    return ("kv_cache_store_active", "ReshapeAndCacheNdKernel", inputs[:2],
            slots, tuple(slot[1:] for slot in (key_cache, value_cache)))


def _scatter_cache(inputs, outputs):
    # CANN's operator graph orders key, keyCacheRef, slotMapping, value,
    # valueCacheRef, unlike the public PyTorch wrapper's argument order.
    key, key_cache, slots, value, value_cache = inputs[:5]
    if (len(key[0]) != 1 or len(value[0]) != 1 or len(slots[0]) != 1
            or not slots[0][0] or key[0][0] <= 0 or value[0][0] <= 0
            or key[0][0] % slots[0][0] or value[0][0] % slots[0][0]
            or tuple(slot[2] for slot in inputs[:5]) != ("NCL", "NCHW", "ND", "NCL", "NCHW")
            or outputs != (key_cache, value_cache)
            or key[1] != key_cache[1] or value[1] != value_cache[1]
            or slots[1] not in {"int32", "int64"}):
        return None
    return ("kv_cache_store_active_elements", "ScatterPaKvCache",
            ((key[0][0], key[1]), (value[0][0], value[1])), slots[0])


def _nd_tensor(slot, rank, dtypes):
    shape, dtype, layout = slot
    return (len(shape) == rank and all(value > 0 for value in shape)
            and dtype in dtypes and layout == "ND")


def _sparse_attention(kind, inputs, outputs):
    floating = {"bf16", "fp16"}
    query, output = inputs[0], outputs[0]
    if (not _nd_tensor(query, SPARSE_ATTENTION_QUERY_RANK, floating) or output != query):
        return None
    if kind == "SparseFlashAttention":
        key, value, rope, key_rope = inputs[1], inputs[2], inputs[7], inputs[8]
        # PA_BSND is (block_count, block_size, heads, dim); TND query keeps
        # its native axes. A recorded block table establishes the paged ABI.
        if (not all(_nd_tensor(slot, SPARSE_ATTENTION_CACHE_RANK, floating)
                    for slot in (key, value, key_rope))
                or not _nd_tensor(rope, SPARSE_ATTENTION_ROPE_RANK, floating)
                or not _nd_tensor(inputs[4], 2, {"int32"})
                or key != value or key[0][-1] != query[0][-1]
                or key_rope[0][:3] != key[0][:3]
                or rope[0][:2] != query[0][:2]
                or rope[0][-1] != key_rope[0][-1]
                or rope[1] != query[1] or key_rope[1] != key[1]):
            return None
        caches = tuple((slot[0][2:], slot[1], slot[2]) for slot in (key, value, key_rope))
        return ("sparse_attention_active", kind, (query, rope), caches, output)
    original, compressed = inputs[1:3]
    empty = ((), None, None)
    # SharedKV uses PA_ND with the same final (heads, dim) axes. Presence of
    # the compressed tensor remains part of the contract, independent of size.
    if (not _nd_tensor(original, SPARSE_ATTENTION_CACHE_RANK, floating)
            or original[0][-1] != query[0][-1]
            or not _nd_tensor(inputs[5], 2, {"int32"})):
        return None
    original_head = (original[0][2:], original[1], original[2])
    if compressed == empty:
        if inputs[6] != empty:
            return None
        compressed_head = empty
    else:
        if (not _nd_tensor(compressed, SPARSE_ATTENTION_CACHE_RANK, floating)
                or not _nd_tensor(inputs[6], 2, {"int32"})
                or (compressed[0][2:], compressed[1:]) != (original[0][2:], original[1:])):
            return None
        compressed_head = (compressed[0][2:], compressed[1], compressed[2])
    return ("sparse_attention_active", kind, (query,), (original_head, compressed_head), output)


def _mqsmla_attention(kind, inputs, outputs):
    query, output = inputs[0], outputs[0]
    empty = ((), None, None)
    if (not _nd_tensor(query, MQSMLA_QUERY_RANK, {"bf16"})
            or query[0][-1] != MQSMLA_QUERY_WIDTH or output != query):
        return None
    if kind in ACLNN_MQSMLA_KINDS:
        canonical_kind = ACLNN_MQSMLA_CANONICAL_KIND
        original, compressed = inputs[1:3]
        table, compressed_table = inputs[5:7]
        if (not _nd_tensor(original, MQSMLA_CACHE_RANK, {"uint8", "fp8_e4m3", "int8"})
                or (original[1], original[0][-1]) not in MQSMLA_NATIVE_RECORDS):
            return None
        compressed_width = original[0][-1]
    else:
        canonical_kind = DSL_MQSMLA_NATIVE_KIND
        original, table = inputs[2:4]
        compressed, compressed_table = (inputs[7:9] if len(inputs) >= MQSMLA_COMPRESSED_INPUT_COUNT
                                        else (empty, empty))
        if (not _nd_tensor(original, MQSMLA_CACHE_RANK, {"uint8"})
                or original[0][-1] != MQSMLA_ORIGINAL_ROW_BYTES):
            return None
        compressed_width = MQSMLA_COMPRESSED_ROW_BYTES
    if original[0][-2] != 1 or not _nd_tensor(table, 2, {"int32"}):
        return None
    compressed_head = empty
    if (compressed != empty or compressed_table != empty
            or (kind in DSL_MQSMLA_KINDS and len(inputs) >= MQSMLA_COMPRESSED_INPUT_COUNT)):
        if (not _nd_tensor(compressed, MQSMLA_CACHE_RANK, {original[1]})
                or compressed[0][-2:] != (1, compressed_width)
                or not _nd_tensor(compressed_table, 2, {"int32"})):
            return None
        compressed_head = (compressed[0][2:], *compressed[1:])
    return ("sparse_attention_active", canonical_kind, (query,),
            ((original[0][2:], *original[1:]), compressed_head), output)


def _attention_projection(row):
    native = native_tensor_contract(row)
    if native is None:
        return None, native_tensor_diagnostic(row)
    inputs, outputs = native[2:]
    arity = _ARITIES.get(row.kind)
    if arity is None or len(inputs) not in arity[0] or len(outputs) not in arity[1]:
        return None, {"category": "contract_support_gap", "reason": "unsupported_native_tensor_slots",
                      "detail": "This attention/cache ABI arity has no source-backed contract.",
                      "operator_type": row.kind}
    if row.kind in SUBGRAPH_TYPES:
        return subgraph_contract(native)
    if row.kind in MQSMLA_KINDS:
        contract = _mqsmla_attention(row.kind, inputs, outputs)
        if contract is None:
            return None, {"category": "contract_support_gap", "reason": "unsupported_mqsmla_geometry",
                          "detail": "The query, cache records and tables must match the ordered ABI.",
                          "operator_type": row.kind}
        return contract, None
    if row.kind == "FusedInferAttentionScore":
        return (_dense_fia(inputs, outputs) or _paged_fia(inputs, outputs)
                or _paged_mla_fia(inputs, outputs) or native), None
    if row.kind == "ReshapeAndCacheNdKernel":
        return _reshape_cache(inputs, outputs) or native, None
    if row.kind == "ScatterPaKvCache":
        return _scatter_cache(inputs, outputs) or native, None
    if row.kind in {"SparseFlashAttention", "SparseAttnSharedkv"}:
        return _sparse_attention(row.kind, inputs, outputs) or native, None
    return native, None


def observed_attention_contract(row):
    return _attention_projection(row)[0]


def attention_contract_scope(row):
    if row.kind in SUBGRAPH_TYPES:
        return subgraph_contract_scope(row)
    contract = observed_attention_contract(row)
    if contract is None:
        return None
    common = {
        "occurrence_identity": False,
        "unobserved": ["tensor_values", "scalar_dispatch_arguments", "selected_tiling",
                       "module_or_layer_identity"],
    }
    if contract[0] == "attention_active_qkv":
        return {
            **common, "kind": "active_attention_tensor_descriptors",
            "compared": ["query_shape_dtype_format", "key_shape_dtype_format",
                         "value_shape_dtype_format", "main_output_shape_dtype_format"],
            "not_compared": ["mask_descriptor_and_values", "sequence_length_arrays",
                             "optional_auxiliary_tensors", "scale", "sparse_mode",
                             "causal_flag", "logical_input_layout", "phase"],
            "storage_boundary": "nonpaged_qkv_descriptors",
        }
    if contract[0] == "attention_paged_active_qkv":
        return {
            **common, "kind": "active_paged_attention_tensor_descriptors",
            "compared": ["native_operator_type", "query_shape_dtype_format",
                         "cache_block_size_packed_row_width_dtype_format",
                         "main_output_shape_dtype_format", "block_table_batch_count"],
            "not_compared": ["cache_capacity", "block_table_column_count_and_values",
                             "sequence_length_array_values", "kv_head_factorization",
                             "mask_descriptor_and_values", "optional_auxiliary_tensors",
                             "scale", "sparse_mode", "causal_flag", "logical_input_layout",
                             "phase", "context_length", "verify_or_draft_role"],
            "storage_boundary": "paged_BSH_cache_and_TND_query_descriptors",
            "conditions": "Execution geometry and context require an independent declaration; "
                          "query token count does not identify a verify or draft invocation.",
        }
    if contract[0] == "attention_paged_mla_qk_rope":
        return {
            **common, "kind": "active_paged_mla_rotary_tensor_descriptors",
            "compared": ["native_operator_type", "native_BNSD_query_shape_dtype_format",
                         "native_BNSD_query_rope_shape_dtype_format",
                         "cache_heads_block_size_width_dtype_format",
                         "native_NBSD_output_shape_dtype_format", "block_table_batch_count"],
            "not_compared": ["cache_capacity", "block_table_column_count_and_values",
                             "sequence_length_values", "scale", "sparse_mode",
                             "causal_flag", "auxiliary_output", "phase", "context_length",
                             "verify_or_draft_role"],
            "storage_boundary": "BNSD_query_NBSD_output_and_paged_rope_cache",
            "validated_capture_abi": "compact7:lengths3/table4/rope5_6; full31:lengths6/table14/rope24_25",
        }
    if contract[0] == "kv_cache_store_active":
        return {
            **common, "kind": "active_cache_store_tensor_descriptors",
            "compared": ["key_shape_dtype_format", "value_shape_dtype_format",
                         "slot_mapping_shape_dtype_format", "cache_dtype_format"],
            "not_compared": ["cache_capacity", "slot_mapping_values", "cache_block_size",
                             "cache_mode", "compression_scalar_arguments"],
        }
    if contract[0] == "kv_cache_store_active_elements":
        return {
            **common, "kind": "active_cache_store_elements",
            "compared": ["key_element_count_dtype", "value_element_count_dtype",
                         "slot_mapping_shape"],
            "validated_capture_abi": "flattened_NCL_key_value_NCHW_cache_ND_slots",
            "not_compared": ["native_storage_layout", "head_shape_factorization",
                             "cache_capacity", "cache_block_size", "slot_mapping_dtype",
                             "slot_mapping_values", "cache_mode", "compression_scalar_arguments"],
        }
    if contract[0] == "sparse_attention_active":
        return {
            **common, "kind": "active_sparse_attention_tensor_descriptors",
            "compared": ["native_operator_type", "query_shape_dtype_format",
                         "query_rope_shape_dtype_format_when_present",
                         "cache_head_shape_dtype_format", "compressed_cache_presence",
                         "main_output_shape_dtype_format"],
            "validated_capture_abi": "ND_query_rank3_and_paged_cache_rank4_with_block_table",
            "not_compared": ["cache_capacity", "cache_block_size", "block_table_shape_and_values",
                             "sequence_length_tensors", "sparse_index_tensors", "sink_tensor",
                             "metadata_tensor", "compression_ratio", "kv_strides",
                             "mask_modes", "attention_window", "logical_input_layout",
                             "scale", "auxiliary_outputs", "phase"],
        }
    return {
        **common, "kind": "native_tensor_abi",
        "compared": ["all_ordered_tensor_shapes", "all_tensor_dtypes", "all_tensor_formats"],
        "not_compared": ["block_table_values", "slot_mapping_values", "sequence_length_values",
                         "cache_mode", "scale", "sparse_mode", "logical_input_layout"],
    }


def attention_contract_diagnostic(row):
    _, issue = _attention_projection(row)
    if issue is not None:
        return issue
    return {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "detail": "The compared descriptor scope is explicit; occurrence evidence remains independent.",
        "operator_type": row.kind,
    }
