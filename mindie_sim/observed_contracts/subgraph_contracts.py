"""Observed attention-preprocessing and index-selection subgraph contracts.

[Role]
  Map source-defined fused subgraph boundaries onto independently declared
  DAGs while retaining the native operator identity and inclusive timing scope.

[Invariants]
  * A fused event covers its members inclusively; separate member timings and
    occurrence identity are never inferred.
  * Native operator types, parameter geometry and dtypes remain distinct.
  * Only documented MLA leading token axes may be combined; tensor values,
    scalar flags and cache contents stay unobserved.
  * Missing descriptors produce explicit capture-gap issues, never guesses.

[Timeouts & Locks]
  Pure bounded descriptor comparisons over one native tensor contract; no
  waits or locks.

[Side-Effect Policy]
  Import-time declarations only; no I/O, hardware, model catalogue or
  simulation imports.
"""
from __future__ import annotations

import math

LIGHTNING_QUERY_RANK = 3
LIGHTNING_CACHE_RANK = 4
LIGHTNING_INDEX_OUTPUT_RANK = 3

MLA_ABSORPTION_RANK = 3
MLA_FLOATING_CACHE_RANK = 4
MLA_PACKED_CACHE_RANK = 4
MLA_PRIMARY_INPUT_COUNT = 7
MLA_PRIMARY_OUTPUT_COUNT = 3
MLA_PACKED_CACHE_INPUT_INDEX = 9
MLA_PACKED_LATENT_WIDTH = 512
MLA_PACKED_ROPE_WIDTH = 64
MLA_PACKED_CACHE_ROW_WIDTH = 656
MLA_AUXILIARY_INPUT_INDICES = (7, 8)

RECURRENT_GDN_OPERAND_RANK = 3
RECURRENT_GDN_STATE_RANK = 4

CHUNK_GDN_OPERAND_RANK = 4
CHUNK_GDN_STATE_RANK = 4
CHUNK_GDN_CHUNK_SIZES = frozenset({64, 128})


MLA_PROLOG_V3_KIND = "MlaPrologV3"
MLA_PROLOG_V4_CANONICAL_KIND = "aclnnMlaPrologV4WeightNz"
MLA_PROLOG_V4_KINDS = frozenset({
    MLA_PROLOG_V4_CANONICAL_KIND, "MlaPrologV4WeightNz", "MlaPrologV4",
})
MLA_PROLOG_KINDS = frozenset({MLA_PROLOG_V3_KIND}) | MLA_PROLOG_V4_KINDS

SUBGRAPH_TYPES = (
    MLA_PROLOG_KINDS
    | {"LightningIndexer", "ChunkGatedDeltaRuleFwd", "RecurrentGatedDeltaRule"}
)
_FLOAT = {"bf16", "fp16"}
_ACTIVATION = _FLOAT | {"int8", "fp8_e4m3"}
_EMPTY = ((), None, None)


def _tensor(slot, rank, dtypes, formats=("ND",)):
    shape, dtype, layout = slot
    return (len(shape) == rank and all(dim > 0 for dim in shape)
            and dtype in dtypes and layout in formats)


def _issue(kind, reason, detail, missing=()):
    return {
        "category": "capture_metadata_missing" if missing else "contract_support_gap",
        "reason": reason, "detail": detail, "operator_type": kind,
        **({"missing_fields": list(missing)} if missing else {}),
    }


def _lightning(inputs, outputs):
    kind = "LightningIndexer"
    required = (("input[0].shape", inputs[0]), ("input[1].shape", inputs[1]),
                ("input[2].shape", inputs[2]), ("output[0].shape", outputs[0]))
    missing = [name for name, slot in required if not slot[0]]
    if missing:
        return None, _issue(kind, "missing_active_tensor_shapes",
                            "Indexer Q, key, weights and index output shapes are required.", missing)
    query, cache, weights = inputs[:3]
    indices = outputs[0]
    if (not _tensor(query, LIGHTNING_QUERY_RANK, _FLOAT)
            or not _tensor(cache, LIGHTNING_CACHE_RANK, _FLOAT)
            or not _tensor(weights, 2, _FLOAT | {"fp32"})
            or not _tensor(inputs[5], 2, {"int32"})
            or not _tensor(indices, LIGHTNING_INDEX_OUTPUT_RANK, {"int32"})
            or weights[0] != query[0][:2]
            or query[0][-1] != cache[0][-1] or query[1] != cache[1]
            or indices[0][:2] != (query[0][0], cache[0][2])):
        return None, _issue(
            kind, "unsupported_indexer_active_geometry",
            "The verified index-selection boundary requires TND Q, paged ND K, "
            "per-token/head weights, a block table and INT32 sparse-index output.")
    cache_head = (cache[0][2:], cache[1], cache[2])
    return ("lightning_indexer_active", kind, (query, weights), cache_head, indices), None


def _mla(raw_kind, inputs, outputs):
    is_v4 = raw_kind in MLA_PROLOG_V4_KINDS
    canonical_kind = (MLA_PROLOG_V4_CANONICAL_KIND if is_v4
                      else MLA_PROLOG_V3_KIND)
    expected_packed_dtype = "fp8_e4m3" if is_v4 else "int8"
    # The source's tile-128 packed cache stores 512 INT8 latent bytes, 64
    # BF16 rotary values and four FP32 scales in one 656-byte row. Its second
    # cache descriptor is explicitly empty, not missing floating cache data.
    packed = _tensor(inputs[MLA_PACKED_CACHE_INPUT_INDEX], MLA_PACKED_CACHE_RANK, {"int8"})
    required = [(f"input[{i}].shape", inputs[i])
                for i in (*range(MLA_PRIMARY_INPUT_COUNT), MLA_PACKED_CACHE_INPUT_INDEX)]
    required += [(f"output[{i}].shape", outputs[i])
                 for i in range(MLA_PRIMARY_OUTPUT_COUNT)]
    if not packed:
        required += [("input[10].shape", inputs[10]), ("output[3].shape", outputs[3])]
    missing = [name for name, slot in required if not slot[0]]
    if missing:
        return None, _issue(raw_kind, "missing_active_tensor_shapes",
                            "MLA token, parameters, cache and primary Q outputs are required.", missing)
    token = inputs[0]
    dq, uq, uk, dkv, gamma_q, gamma_kv = inputs[1:7]
    query, rope = outputs[:2]
    cache, rope_cache = inputs[9:11]
    if (len(token[0]) not in {2, 3}
            or not _tensor(token, len(token[0]), _ACTIVATION)
            or not all(_tensor(slot, 2, _ACTIVATION, ("FRACTAL_NZ",))
                       for slot in (dq, uq, dkv))
            or not _tensor(uk, MLA_ABSORPTION_RANK, _FLOAT)
            or not all(_tensor(slot, 1, _FLOAT) for slot in (gamma_q, gamma_kv))
            or (not packed and not all(_tensor(slot, MLA_FLOATING_CACHE_RANK, _FLOAT)
                                       for slot in (cache, rope_cache)))
            or not _tensor(query, len(token[0]) + 1, _FLOAT)
            or not _tensor(rope, len(token[0]) + 1, _FLOAT)):
        return None, _issue(
            raw_kind, "unsupported_mla_active_geometry",
            "The verified MLA projection requires logical NZ matrix descriptors, "
            "ND parameters, floating or packed INT8 paged caches and complete token/Q axes.")
    leading = token[0][:-1]
    tokens, hidden = math.prod(leading), token[0][-1]
    heads, nope, latent = uk[0]
    rope_width = rope[0][-1]
    if (dq[0] != (hidden, gamma_q[0][0])
            or uq[0] != (gamma_q[0][0], heads * (nope + rope_width))
            or dkv[0] != (hidden, latent + rope_width)
            or gamma_kv[0] != (latent,)
            or query[0] != (*leading, heads, latent)
            or rope[0] != (*leading, heads, rope_width)
            or outputs[2:4] != (cache, rope_cache)
            or len({slot[1] for slot in (uk, gamma_q, gamma_kv, query, rope)}) != 1):
        return None, _issue(
            raw_kind, "inconsistent_mla_active_parameters",
            "MLA projection, RMS, absorption, output and cache dimensions do not agree.")
    if packed:
        if (latent != MLA_PACKED_LATENT_WIDTH or rope_width != MLA_PACKED_ROPE_WIDTH
                or cache[0][2:] != (1, MLA_PACKED_CACHE_ROW_WIDTH)
                or rope_cache not in {((), "bf16", "ND"), _EMPTY}
                or any(slot[1] != expected_packed_dtype
                       for slot in (token, dq, uq, dkv))
                or query[1] != "bf16"):
            return None, _issue(raw_kind, "unsupported_mla_packed_cache",
                                "The retained packed ABI requires 512 INT8 latent bytes, "
                                "64 BF16 rotary values, four FP32 scales and an empty second cache.")
        cache_heads = (((1, MLA_PACKED_CACHE_ROW_WIDTH), "int8", "ND"), _EMPTY)
    else:
        if (is_v4 or cache[0][2:] != (1, latent) or rope_cache[0][2:] != (1, rope_width)
                or cache[1] != rope_cache[1] or cache[1] != query[1]):
            return None, _issue(raw_kind, "inconsistent_mla_active_parameters",
                                "Floating cache dimensions and precision must match their query outputs.")
        cache_heads = tuple((slot[0][2:], slot[1], slot[2]) for slot in (cache, rope_cache))
    query_norm = outputs[5]
    if query_norm[0]:
        if (query_norm[0] not in {(tokens, gamma_q[0][0]), (*leading, gamma_q[0][0])}
                or not _tensor(query_norm, len(query_norm[0]),
                               {expected_packed_dtype} if packed else _ACTIVATION)):
            return None, _issue(raw_kind, "unsupported_mla_query_norm",
                                "The recorded optional query-norm output must retain its row/rank axes.")
        query_norm = ((tokens, gamma_q[0][0]), query_norm[1], query_norm[2])
    else:
        # Both optional outputs are empty in the source's query_norm_flag=false
        # path. A scale with no corresponding data is incomplete capture evidence.
        if outputs[6][0]:
            return None, _issue(raw_kind, "missing_active_tensor_shapes",
                                "Query-norm scale is present but its data shape is absent.",
                                ("output[5].shape",))
        query_norm = _EMPTY
    parameters = tuple((slot[0], slot[1]) for slot in inputs[1:7])
    output_slots = (((tokens, heads, latent), query[1], query[2]),
                    ((tokens, heads, rope_width), rope[1], rope[2]), query_norm)
    return ("mla_prolog_active", canonical_kind, ((tokens, hidden), token[1], token[2]),
            parameters, cache_heads, output_slots), None


def _recurrent_gdn(inputs, outputs):
    kind = "RecurrentGatedDeltaRule"
    query, key, value, beta, state, lengths, indices, gate = inputs[:8]
    gk, accepted = (_EMPTY, inputs[8]) if len(inputs) == 9 else inputs[8:10]
    if (not all(_tensor(slot, RECURRENT_GDN_OPERAND_RANK, {"bf16"})
                for slot in (query, key, value))
            or key != query or outputs[0] != value
            or not _tensor(state, RECURRENT_GDN_STATE_RANK, {"bf16", "fp32"})
            or outputs[1] != state
            or not _tensor(lengths, 1, {"int32"}) or lengths[0][0] < 2):
        return None, _issue(kind, "inconsistent_recurrent_gdn_tensors",
                            "Recurrent GDN requires equal BF16 Q/K, BF16 value/output, "
                            "the same state input/output, and cumulative sequence descriptors.")
    tokens, key_heads, dk = query[0]
    value_tokens, heads, dv = value[0]
    sequences = lengths[0][0] - 1
    if (value_tokens != tokens or heads % key_heads or state[0][1:] != (heads, dv, dk)
            or beta != ((tokens, heads), "bf16", "ND")
            or indices != ((tokens,), "int32", "ND")
            or gate != ((tokens, heads), "fp32", "ND") or gk != _EMPTY
            or accepted not in {_EMPTY, ((sequences,), "int32", "ND")}):
        return None, _issue(kind, "unsupported_recurrent_gdn_geometry",
                            "The retained scalar-decay ABI requires TND operands, V-first "
                            "state, per-token state indices and optional per-sequence accepted counts.")
    return ("recurrent_vector", (tokens, key_heads, heads, dk, dv), sequences, state[1]), None


def _chunk_gdn(native):
    _, kind, inputs, outputs = native
    query, key, value, beta, workspace, gate = inputs[:6]
    gk, state, lengths, chunks = ((_EMPTY, *inputs[6:]) if len(inputs) == 9 else inputs[6:])
    if (not all(_tensor(slot, CHUNK_GDN_OPERAND_RANK, _FLOAT, ("NCHW", "ND"))
                for slot in (query, key, value, workspace))
            or not _tensor(state, CHUNK_GDN_STATE_RANK, _FLOAT | {"fp32"}, ("NCHW", "ND"))
            or query != key or not _tensor(lengths, 1, {"int64"})
            or not _tensor(chunks, 1, {"int64"}) or lengths[0][0] < 2 or gk != _EMPTY):
        return None, _issue(kind, "unsupported_chunk_gdn_native_geometry",
                            "The retained legacy chunk ABI needs complete BHTD operands, "
                            "state, INT64 cumulative lengths/chunk metadata and no vector gate.")
    batch, key_heads, tokens, dk = query[0]
    vb, heads, vt, dv = value[0]
    chunk = workspace[0][-1]
    state_shape, state_dtype, _ = state
    if (batch != 1 or (vb, vt) != (batch, tokens) or heads % key_heads
            or workspace[0] != (batch, heads, tokens, chunk)
            or chunk not in CHUNK_GDN_CHUNK_SIZES
            or state_shape != (lengths[0][0] - 1, heads, dk, dv)
            or len({query[1], value[1], workspace[1], outputs[0][1], outputs[3][1]}) != 1
            or beta != ((batch, heads, tokens), "fp32", "ND")
            or gate not in {((batch, tokens, heads), "fp32", "ND"),
                            ((batch, heads, tokens), "fp32", "ND")}
            or chunks[0][0] % 2
            or outputs[0][0] != (batch, tokens, heads, dv) or outputs[0][2] != "ND"
            or outputs[1][0] != (math.prod(state_shape),) or outputs[1][1] != state_dtype
            or outputs[2] != ((batch * tokens * heads,), "fp32", "ND")
            or outputs[3][0] != (batch * heads * tokens * chunk,)):
        return None, _issue(kind, "inconsistent_chunk_gdn_native_tensors",
                            "The legacy chunk's Q/K/V, workspace, scalar gates, state and "
                            "materialized auxiliary-output descriptors must agree.")
    # The retained wrapper uses a different TND/BF16-state interface. Preserve
    # this complete native boundary until a declaration selects the same ABI.
    return native, None


def subgraph_contract(native):
    """Return (active contract, issue) after native slot cardinality validation."""
    _, kind, inputs, outputs = native
    if kind == "RecurrentGatedDeltaRule":
        return _recurrent_gdn(inputs, outputs)
    if kind == "ChunkGatedDeltaRuleFwd":
        return _chunk_gdn(native)
    if kind in MLA_PROLOG_KINDS:
        return _mla(kind, inputs, outputs)
    return _lightning(inputs, outputs)


def subgraph_contract_scope(row):
    common = {
        "occurrence_identity": False,
        "unobserved": ["tensor_values", "scalar_dispatch_arguments", "selected_tiling",
                       "module_or_layer_identity"],
        "timing_scope": "inclusive_fused_subgraph; no_member_time_split",
    }
    if row.kind == "RecurrentGatedDeltaRule":
        return {
            **common, "kind": "active_recurrent_gdn_geometry",
            "compared": ["query_token_count", "query_key_heads", "value_heads",
                         "key_width", "value_width", "sequence_count", "state_dtype"],
            "validated_capture_abi": [
                "BF16_ND_query_key_value_beta_and_output", "FP32_ND_scalar_decay",
                "V_first_state_input_output_identity", "INT32_sequence_and_state_indices",
            ],
            "not_compared": ["state_capacity", "state_indices_values", "sequence_length_values",
                             "accepted_token_values", "scale", "phase", "verify_or_draft_role"],
        }
    if row.kind == "ChunkGatedDeltaRuleFwd":
        return {
            **common, "kind": "native_chunk_gdn_tensor_abi",
            "compared": ["all_ordered_tensor_shapes", "all_tensor_dtypes", "all_tensor_formats"],
            "not_compared": ["sequence_length_values", "chunk_indices_values", "scale",
                             "scalar_flags", "selected_kernel_binary", "phase"],
            "boundary_limit": "Legacy BHTD/FP32-state capture is distinct from the "
                              "generic TND/BF16-state wrapper; no default subgraph match is implied.",
        }
    if row.kind == "LightningIndexer":
        return {
            **common, "kind": "active_indexer_selection_descriptors",
            "compared": ["native_operator_type", "query_shape_dtype_format",
                         "weights_shape_dtype_format", "key_cache_head_shape_dtype_format",
                         "sparse_index_output_shape_dtype_format"],
            "not_compared": ["cache_capacity", "cache_block_size", "block_table_shape_and_values",
                             "sequence_length_tensors", "sparse_values_output", "sparse_mode",
                             "attention_window", "logical_input_layout", "phase"],
            "excluded_semantic_work": ["query_projection", "weight_projection", "index_cache_build"],
        }
    shapes = row.input_shapes or ()
    missing_auxiliary = [f"input[{i}].shape" for i in MLA_AUXILIARY_INPUT_INDICES
                         if i >= len(shapes) or not shapes[i]]
    return {
        **common, "kind": "active_mla_prolog_descriptors",
        "compared": ["native_operator_type", "token_rows_hidden_dtype_format",
                     "projection_absorption_and_rms_parameter_shapes_dtypes",
                     "cache_head_shape_dtype_format", "query_and_query_rope_rows_heads_dims_dtypes",
                     "optional_query_norm_descriptor_presence_shape_dtype"],
        "not_compared": ["leading_token_axis_grouping", "parameter_physical_storage_layout",
                         "rope_input_tensors", "quantization_scale_tensors", "quantization_scalar_modes",
                         "epsilon", "query_key_scale", "cache_capacity", "cache_block_size",
                         "cache_index_tensor", "clip_and_smooth_values", "phase"],
        "excluded_capture_missing_fields": missing_auxiliary,
        "optional_query_norm_limit": "Empty recorded outputs are compared; the scalar flag is not observed.",
    }
