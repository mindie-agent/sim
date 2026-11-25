"""Observed fused-kernel, fusion-region and subgraph contracts.

[Role] Validate observed fused kernels and multi-op subgraph contracts.
[Invariants] Reports explicit metadata-missing or support-gap diagnostics.
[Timeouts & Locks] Pure in-memory validation; no I/O or locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

import math

from .moe import (
    MOE_FUSION_TYPES, QUANT_FUSION_TYPES,
    moe_fusion_contract, moe_fusion_scope,
    quant_fusion_contract, quant_fusion_scope,
)
from .vocabulary import native_tensor_contract, native_tensor_diagnostic


NORM_GATE = "_layer_norm_fwd_1pass_kernel_npu"
KIMI_NORM_GATE = "layer_norm_gated_fwd_kernel"
QKV_NORM_ROPE = "split_qkv_rmsnorm_mrope_kernel"
FUSED_TYPES = {
    NORM_GATE, KIMI_NORM_GATE, QKV_NORM_ROPE, "GemmaRmsNorm", "AddRmsNormBias", "SituMxQuant",
}
UNOBSERVED_FIELDS = {
    NORM_GATE: ("eps", "norm_before_gate", "gate_activation", "is_rms_norm", "group_size", "strides"),
    KIMI_NORM_GATE: (
        "eps", "gate_activation", "is_rms_norm", "bias", "residual",
        "mean", "rstd", "strides", "compiled_kernel_flags",
    ),
    QKV_NORM_ROPE: (
        "eps", "mrope_sections", "is_interleaved", "partial_rope",
        "output_slot_identity", "unrecorded_output_tensors",
    ),
    "GemmaRmsNorm": ("eps", "weight_values"),
    "AddRmsNormBias": ("eps", "weight_values"),
    "SituMxQuant": ("beta", "linear_beta", "activate_left", "axis", "rounding_mode"),
}
SITU_MX_CHANNEL_DIVISOR = 128
SITU_MX_SCALE_BLOCK_WIDTH = 64
QKV_NORM_ROPE_REQUIRED_OUTPUT_SLOTS = 4


def _situ_mx_contract(inputs, outputs):
    if len(inputs) != 1 or len(outputs) != 2:
        return None
    x, (y, scales) = inputs[0], outputs
    if (len(x[0]) != 2 or x[1:] != ("bf16", "ND")
            or x[0][-1] <= 0 or x[0][-1] % SITU_MX_CHANNEL_DIVISOR):
        return None
    rows, width = x[0][0], x[0][-1] // 2
    if (rows <= 0 or y[0] != (rows, width) or y[1] not in {"fp8_e4m3", "fp8_e5m2"}
            or y[2] != "ND" or scales != ((rows, width // SITU_MX_SCALE_BLOCK_WIDTH, 2), "fp8_e8m0", "ND")):
        return None
    return "situ_mxquant_active", "SituMxQuant", x, y, scales


def _kimi_gated_rows(inputs, outputs):
    """The KDA wrapper flattens leading axes and exposes only the primary output."""
    if len(inputs) != 3 or len(outputs) != 1:
        return None
    x, gate, weight = inputs
    if (len(x[0]) < 2 or len(gate[0]) < 2 or not x[0][-1]
            or gate[0][-1] != x[0][-1] or math.prod(gate[0]) != math.prod(x[0])
            or gate[1:] != x[1:] or outputs[0] != x
            or weight != ((x[0][-1],), *x[1:])):
        return None
    rows = math.prod(x[0][:-1])
    return ("gated_norm_active_rows", KIMI_NORM_GATE,
            ((rows, x[0][-1]), *x[1:]), weight)


def _normalization_slots(kind, inputs, outputs):
    """Validate the retained native pattern; other valid exporter ABIs stay unknown."""
    if kind == NORM_GATE:
        if len(inputs) != 3 or len(outputs) != 2:
            return False
        x, weight, gate = inputs
        y, statistics = outputs
        if len(x[0]) != 2 or weight[0] != (x[0][-1],) or gate != x or y != x:
            return False
        return (weight[1:] == x[1:]
                and statistics == ((x[0][0],), "fp32", "ND"))
    if kind == "GemmaRmsNorm":
        if len(inputs) != 2 or len(outputs) != 2:
            return False
        x, weight = inputs
        y, statistics = outputs
        return (len(x[0]) == 2 and weight == ((x[0][-1],), *x[1:])
                and y == x and statistics == ((x[0][0], 1), "fp32", "ND"))
    if kind == "AddRmsNormBias":
        if len(inputs) != 4 or len(outputs) != 3:
            return False
        x, residual, weight, bias = inputs
        y, statistics, residual_output = outputs
        return (len(x[0]) == 2 and residual == x and y == x and residual_output == x
                and weight == ((x[0][-1],), *x[1:])
                and bias in {((), None, None), weight}
                and statistics == ((x[0][0], 1), "fp32", "ND"))
    return False


def observed_fused_contract(row):
    if row.kind not in FUSED_TYPES:
        return None
    contract = native_tensor_contract(row)
    if contract is None:
        return None
    _, _, inputs, outputs = contract
    if row.kind == QKV_NORM_ROPE:
        # The retained A3/A5 exports contain one output, while the wrapper
        # returns Q/K/V/gate. Q and gate even share shape, so slot identity
        # cannot be recovered from this descriptor.
        return None
    if row.kind == "SituMxQuant":
        return _situ_mx_contract(inputs, outputs)
    present = [slot for slot in (*inputs, *outputs) if slot != ((), None, None)]
    if any(slot[2] != "ND" or slot[1] not in {"bf16", "fp16", "fp32"} for slot in present):
        return None
    if row.kind == KIMI_NORM_GATE:
        return _kimi_gated_rows(inputs, outputs)
    return contract if _normalization_slots(row.kind, inputs, outputs) else None


def fused_contract_scope(row):
    if row.kind not in FUSED_TYPES:
        return None
    scope = {
        "comparison_scope": "native_tensor_slots; scalar_flags_and_tensor_values_not_observed",
        "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        "basis": "captured_tensor_descriptors",
        "semantic_limits": [
            "Tensor equality does not establish compiled flags, numerical equivalence or layer occurrence.",
        ],
    }
    if row.kind == "SituMxQuant":
        scope["comparison_scope"] = (
            "active_input_output_and_MX_scale_descriptors; "
            "activation_scalar_parameters_and_quantization_mode_not_compared")
        scope["semantic_limits"].append(
            "The native kind establishes a SiTU/MX boundary; beta, linear_beta and "
            "activate_left require an independent caller declaration and are not inferred from shape.")
    elif row.kind == KIMI_NORM_GATE:
        scope["comparison_scope"] = (
            "active_row_geometry_weight_dtype_and_ND_descriptors; "
            "auxiliary_outputs_and_scalar_flags_not_compared; no_full_native_abi_equality")
        scope["semantic_limits"].append(
            "The KDA wrapper reshapes x and gate to rows by final feature dimension, "
            "restores the primary output shape, and discards mean/rstd. This comparison "
            "does not alias the Qwen normalization kernel or establish the compiled RMS/gate flags.")
        scope["compared_input_slots"] = ["x", "gate", "weight"]
        scope["compared_output_slots"] = ["primary_output"]
    elif row.kind == QKV_NORM_ROPE:
        scope["comparison_scope"] = "incomplete_native_output_slots; no_contract_equality"
        scope["required_output_slots"] = QKV_NORM_ROPE_REQUIRED_OUTPUT_SLOTS
        scope["observed_output_slots"] = len(row.output_shapes) if row.output_shapes else 0
        scope["semantic_limits"].append(
            "The referenced wrapper declares Q/K/V/gate outputs; the retained export cannot identify them.")
    elif row.kind == "AddRmsNormBias":
        scope["semantic_limits"].append(
            "The fourth slot distinguishes absent beta from a channel-wise affine beta. "
            "Caller-side weight shifts and the checkpoint configuration remain unknown.")
    elif row.kind == "GemmaRmsNorm":
        scope["semantic_limits"].append(
            "The native Gemma operator identity is retained; it is not aliased to ordinary RmsNorm.")
    return scope


def fused_contract_diagnostic(row):
    if row.kind not in FUSED_TYPES:
        return None
    descriptor = native_tensor_contract(row)
    diagnostic = native_tensor_diagnostic(row)
    if descriptor is None:
        return {
            **diagnostic,
            "operator_type": row.kind,
            "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        }
    if row.kind == QKV_NORM_ROPE and len(descriptor[3]) < 4:
        return {
            "category": "capture_metadata_missing",
            "reason": "incomplete_native_tensor_slots",
            "detail": ("The QKV normalization/rotary wrapper returns Q, K, V and gate tensors, "
                       "but this capture records fewer than four outputs. Q and gate can have "
                       "the same shape; missing slots and output identity cannot be inferred."),
            "operator_type": row.kind,
            "observed_output_slots": len(descriptor[3]),
            "required_output_slots": QKV_NORM_ROPE_REQUIRED_OUTPUT_SLOTS,
            "captured_descriptor": descriptor,
            "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        }
    if observed_fused_contract(row) is None:
        return {
            "category": "contract_support_gap",
            "reason": "unsupported_tensor_contract",
            "detail": "The complete captured descriptors differ from the verified native fusion pattern.",
            "operator_type": row.kind,
            "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        }
    return {
        "category": "tensor_contract_available",
        "reason": "observed_tensor_contract_available",
        "detail": (
            "Active x/gate/weight/output row geometry is available; auxiliary statistics, "
            "scalar flags and tensor values are not compared."
            if row.kind == KIMI_NORM_GATE else
            "All compared native tensor slots are present; scalar flags and tensor values are not compared."),
        "operator_type": row.kind,
        "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
    }


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
            or not _tensor(lengths, 1, {"int32"})):
        return None, _issue(kind, "inconsistent_recurrent_gdn_tensors",
                            "Recurrent GDN requires equal BF16 Q/K, BF16 value/output, "
                            "the same state input/output, and an INT32 sequence descriptor.")
    tokens, key_heads, dk = query[0]
    value_tokens, heads, dv = value[0]
    sequences = lengths[0][0]
    if len(inputs) == 9:
        # Compact captures omit gk. Accepted counts disambiguate the
        # wrapper's leading length entry from the native B-element vector.
        if accepted == _EMPTY:
            return None, _issue(kind, "unresolved_sequence_count",
                                "Compact lengths alone do not identify the sequence convention.")
        if accepted[0] == (sequences - 1,) and sequences > 1:
            sequences -= 1
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


QUANT_INDEXER = "VllmAscendQuantLightningIndexerV2"
QUANT_INDEXER_INPUT_COUNT = 14
QUANT_INDEXER_OUTPUT_COUNT = 3
QUANT_INDEXER_QUERY_RANK = 3
QUANT_INDEXER_KEY_CACHE_RANK = 4
QUANT_INDEXER_KEY_SCALE_RANK = 3
QUANT_INDEXER_INDICES_RANK = 3
QUANT_INDEXER_REQUIRED_INPUT_SLOTS = (0, 1, 2, 3, 4, 10)
HC_PRE_INPUT_COUNT = 4
HC_PRE_OUTPUT_COUNT = 3
HC_PRE_PRE_MIX_INPUT_COUNT = 5
HC_PRE_PRE_MIX_OUTPUT_COUNT = 4
HC_PRE_TOKEN_AXIS_RANKS = frozenset({3, 4})
HC_PRE_MIX_SCALE_WIDTH = 3
HC_POST_INPUT_COUNT = 4
HC_POST_OUTPUT_COUNT = 1
HC_POST_TOKEN_AXIS_RANKS = frozenset({2, 3})
COMPRESSOR_INPUT_COUNT = 12
COMPRESSOR_OUTPUT_COUNT = 2
COMPRESSOR_REQUIRED_INPUT_COUNT = 8
COMPRESSOR_STATE_RANK = 3
RECURRENT_KDA_INPUT_COUNT = 10
RECURRENT_KDA_OUTPUT_COUNT = 3
RECURRENT_KDA_ACCEPTED_TOKENS_INPUT_COUNT = 11
RECURRENT_KDA_ACCEPTED_TOKENS_OUTPUT_COUNT = 3
RECURRENT_KDA_REQUIRED_INPUT_SLOTS = (*range(6), 8, 9)
RECURRENT_KDA_GATE_PARAMETER_INPUT_SLOTS = (3, 4, 5, 8, 9)
RECURRENT_KDA_GATE_RANK = 4
RECURRENT_KDA_BETA_RANK = 3
ATTN_RES_INPUT_SLOT_COUNT = 6
ATTN_RES_OUTPUT_SLOT_COUNT = 3
FUSION_TYPES = {QUANT_INDEXER, "HcPre", "HcPost", "Compressor",
                "ChunkKdaFwd", "RecurrentKda", "AttnResFwd"} | MOE_FUSION_TYPES | QUANT_FUSION_TYPES
_ARITIES = {
    QUANT_INDEXER: {(QUANT_INDEXER_INPUT_COUNT, QUANT_INDEXER_OUTPUT_COUNT)},
    "HcPre": {(HC_PRE_INPUT_COUNT, HC_PRE_OUTPUT_COUNT),
              (HC_PRE_PRE_MIX_INPUT_COUNT, HC_PRE_PRE_MIX_OUTPUT_COUNT)},
    "HcPost": {(HC_POST_INPUT_COUNT, HC_POST_OUTPUT_COUNT)},
    "Compressor": {(COMPRESSOR_INPUT_COUNT, COMPRESSOR_OUTPUT_COUNT)},
    "RecurrentKda": {(RECURRENT_KDA_INPUT_COUNT, RECURRENT_KDA_OUTPUT_COUNT),
                     (RECURRENT_KDA_ACCEPTED_TOKENS_INPUT_COUNT,
                      RECURRENT_KDA_ACCEPTED_TOKENS_OUTPUT_COUNT)},
}
_GAPS = {
    "ChunkKdaFwd": (
        "stage_dispatch_unobserved",
        "The retained 10-input/13-output ABI launches several stages under the "
        "same native operator type. Its stage scalar is absent; one event cannot "
        "be assigned the entire chunk products/solve/scan/output subgraph."),
    "AttnResFwd": (
        "active_view_or_stride_unobserved",
        "The historical image supplies the matching six-input/three-BF16-output "
        "interface, including optional add, output-normalization and bank-write "
        "branches. Nonempty fallback tensors do not reveal those scalar flags; "
        "flattened backing allocations do not establish active views or history depth."),
}
# Qwen4 guard registration: explicit gaps only, never an invented active ABI.
QWEN4_HC_KINDS = frozenset({
    "Qwen4ExpGatedResidualMix", "Qwen4ExpHcCombine",
})
FUSION_TYPES |= QWEN4_HC_KINDS
_GAPS.update({
    kind: (
        "active_native_abi_unverified",
        "The Qwen4 low-rank HC boundary has no verified active ABI projection.",
    )
    for kind in QWEN4_HC_KINDS
})
_ATTN_RES_RUNTIME_FIELDS = (
    "invocation_implementation_and_tiling", "active_prefix_shape", "block_residual_strides",
    "valid_blocks", "mix", "fuse_add", "output_norm_enabled", "block_write_idx",
    "save_materialized", "optimize_prefill",
)
_ATTN_RES_SOURCE_INTERFACE = {
    "identity_basis": "retained_capture_image_source",
    "image_id": "sha256:883670fcc91cbd75a7070b00501be0ecd81a266b81366a21bf02139d8e6a33e1",
    "definition_sha256": "8f9c501cc20adbf0860eba3042ac415d3654400284a7257d617fdb3a5a9b1f17",
    "input_count": ATTN_RES_INPUT_SLOT_COUNT, "output_count": ATTN_RES_OUTPUT_SLOT_COUNT,
    "event_binary_or_dispatch_proven": False,
}






def _tail(slot, axes=1):
    return (slot[0][-axes:], slot[1], slot[2])


def _missing(kind, inputs, outputs, ins, outs):
    missing = [f"input[{i}].shape" for i in ins if not inputs[i][0]]
    missing += [f"output[{i}].shape" for i in outs if not outputs[i][0]]
    return (_issue(kind, "missing_active_tensor_shapes",
                   "Compared fusion inputs and outputs require captured shapes.", missing)
            if missing else None)


def _quant_indexer(inputs, outputs):
    kind = QUANT_INDEXER
    missing = _missing(kind, inputs, outputs, QUANT_INDEXER_REQUIRED_INPUT_SLOTS, (0,))
    if missing:
        return None, missing
    if inputs[13][0] or outputs[2][0]:
        return None, _issue(
            kind, "candidate_indexer_boundary_unsupported",
            "Candidate-filter input/output tensors select a different boundary "
            "from ordinary quantized score-plus-topk selection.")
    query, cache, weights, qscale, kscale = inputs[:5]
    indices = outputs[0]
    if (not _tensor(query, QUANT_INDEXER_QUERY_RANK, {"int8"})
            or not _tensor(cache, QUANT_INDEXER_KEY_CACHE_RANK, {"int8"})
            or not all(_tensor(slot, 2, {"fp16", "fp32"})
                       for slot in (weights, qscale))
            or not _tensor(kscale, QUANT_INDEXER_KEY_SCALE_RANK, {"fp16", "fp32"})
            or not _tensor(inputs[10], 2, {"int32"})
            or not _tensor(indices, QUANT_INDEXER_INDICES_RANK, {"int32"})
            or weights[0] != query[0][:2] or qscale[0] != weights[0]
            or kscale[0] != cache[0][:-1] or qscale[1:] != kscale[1:]
            or query[0][-1] != cache[0][-1]
            or indices[0][:2] != (query[0][0], cache[0][2])):
        return None, _issue(
            kind, "unsupported_quant_indexer_active_geometry",
            "Verified quantized selection requires TND INT8 Q, paged INT8 K, "
            "per-query/head weights and descales, per-cache/head descales, "
            "an INT32 block table and sparse-index output.")
    return ("quant_lightning_indexer_active", kind, (query, weights, qscale),
            (_tail(cache, 2), _tail(kscale)), indices), None


def _hc_pre(inputs, outputs):
    kind = "HcPre"
    missing = _missing(kind, inputs, outputs, range(HC_PRE_INPUT_COUNT),
                       range(HC_PRE_OUTPUT_COUNT))
    if missing:
        return None, missing
    if len(inputs) == HC_PRE_PRE_MIX_INPUT_COUNT and inputs[4] != _EMPTY:
        return None, _issue(
            kind, "precomputed_hyper_connection_mix",
            "A supplied pre_mix is a separate native boundary whose captured "
            "resource program is not established. Its presence alone does not "
            "prove that projection or normalization work is skipped.")
    x, fn, scale, base = inputs[:4]
    if len(x[0]) not in HC_PRE_TOKEN_AXIS_RANKS or not _tensor(x, len(x[0]), {"bf16"}):
        return None, _issue(kind, "unsupported_hyper_connection_geometry",
                            "HcPre requires ND BF16 token/HC/hidden axes.")
    leading, (hc, hidden) = x[0][:-2], x[0][-2:]
    mix = (hc + 2) * hc
    expected_inputs = (((mix, hc * hidden), "fp32", "ND"),
                       ((HC_PRE_MIX_SCALE_WIDTH,), "fp32", "ND"), ((mix,), "fp32", "ND"))
    expected_outputs = (((*leading, hidden), "bf16", "ND"),
                        ((*leading, hc), "fp32", "ND"),
                        ((*leading, hc, hc), "fp32", "ND"))
    if ((fn, scale, base) != expected_inputs or outputs[:3] != expected_outputs
            or len(outputs) == 4 and outputs[3] != expected_outputs[1]):
        return None, _issue(
            kind, "inconsistent_hyper_connection_geometry",
            "HC projection, mixing parameters and output axes must agree before "
            "documented leading token dimensions are combined.")
    tokens = math.prod(leading)
    canonical_inputs = (((tokens, hc, hidden), "bf16", "ND"), fn, scale, base)
    canonical_outputs = (((tokens, hidden), "bf16", "ND"),
                         ((tokens, hc), "fp32", "ND"), ((tokens, hc, hc), "fp32", "ND"))
    return ("hyper_connection_active", kind, canonical_inputs, canonical_outputs), None


def _hc_post(inputs, outputs):
    kind = "HcPost"
    missing = _missing(kind, inputs, outputs, range(HC_POST_INPUT_COUNT), (0,))
    if missing:
        return None, missing
    x, residual, post, comb = inputs
    if (len(x[0]) not in HC_POST_TOKEN_AXIS_RANKS or not _tensor(x, len(x[0]), {"bf16"})
            or len(residual[0]) != len(x[0]) + 1):
        return None, _issue(kind, "unsupported_hyper_connection_geometry",
                            "HcPost requires matching token axes and an explicit HC axis.")
    leading, hidden = x[0][:-1], x[0][-1]
    hc = residual[0][-2]
    expected = (((*leading, hc, hidden), "bf16", "ND"),
                ((*leading, hc), "fp32", "ND"), ((*leading, hc, hc), "fp32", "ND"))
    if hc < 1 or (residual, post, comb) != expected or outputs[0] != residual:
        return None, _issue(
            kind, "inconsistent_hyper_connection_geometry",
            "HcPost input, residual, coefficients and output must share exact leading axes.")
    tokens = math.prod(leading)
    canonical_inputs = (((tokens, hidden), "bf16", "ND"),
                        ((tokens, hc, hidden), "bf16", "ND"),
                        ((tokens, hc), "fp32", "ND"), ((tokens, hc, hc), "fp32", "ND"))
    return ("hyper_connection_active", kind, canonical_inputs, (canonical_inputs[1],)), None


def _compressor(inputs, outputs):
    kind = "Compressor"
    missing = _missing(kind, inputs, outputs, range(COMPRESSOR_REQUIRED_INPUT_COUNT),
                       range(COMPRESSOR_OUTPUT_COUNT))
    if missing:
        return None, missing
    x, wkv, wgate, state, ape, norm, sin, cos = inputs[:8]
    result = outputs[0]
    if (not _tensor(x, 2, _FLOAT)
            or not all(_tensor(slot, 2, _FLOAT) for slot in (wkv, wgate, result))
            or not _tensor(state, COMPRESSOR_STATE_RANK, {"fp32"}) or not _tensor(ape, 2, {"fp32"})
            or not _tensor(norm, 1, _FLOAT | {"fp32"})
            or not all(_tensor(slot, 2, _FLOAT | {"fp32"}) for slot in (sin, cos))
            or wgate != wkv or wkv[1:] != x[1:] or result[1:] != x[1:]
            or wkv[0][1] != x[0][1] or ape[0][1] != wkv[0][0]
            or state[0][-1] != 2 * wkv[0][0] or outputs[1] != state
            or result[0][-1] != norm[0][0] or wkv[0][0] % norm[0][0]
            or sin != cos or sin[0][0] != result[0][0]
            or sin[0][-1] > result[0][-1]):
        return None, _issue(
            kind, "inconsistent_compressor_active_geometry",
            "Compressor projections, state channels, position weights, norm/rope "
            "widths and allocated outputs must agree. Allocated output rows do "
            "not establish the number of completed compression groups.")
    return ("compressor_active", kind, x, (wkv, wgate, ape, norm), _tail(state),
            (_tail(sin), _tail(cos)), _tail(result)), None


def _recurrent_kda(inputs, outputs):
    kind = "RecurrentKda"
    missing = _missing(kind, inputs, outputs, RECURRENT_KDA_REQUIRED_INPUT_SLOTS,
                       range(RECURRENT_KDA_OUTPUT_COUNT))
    if missing:
        return None, missing
    gate, beta, state, a_log, dt_bias = (
        inputs[i] for i in RECURRENT_KDA_GATE_PARAMETER_INPUT_SLOTS)
    if len(inputs) == RECURRENT_KDA_ACCEPTED_TOKENS_INPUT_COUNT and inputs[10][0]:
        return None, _issue(
            kind, "recurrent_accepted_tokens_boundary_unsupported",
            "An accepted-token tensor selects speculative state-update behavior, "
            "which is outside the retained ordinary recurrent boundary.")
    if (not _tensor(gate, RECURRENT_KDA_GATE_RANK, _FLOAT | {"fp32"})
            or not _tensor(beta, RECURRENT_KDA_BETA_RANK, _FLOAT | {"fp32"})
            or beta[0] != gate[0][:-1]
            or a_log != ((gate[0][-2],), "fp32", "ND")
            or dt_bias != ((gate[0][-2] * gate[0][-1],), "fp32", "ND")
            or any(not slot[0] or slot[1] not in _FLOAT or slot[2] != "ND"
                   for slot in (*inputs[:3], outputs[0]))
            or len({slot[1:] for slot in (*inputs[:3], outputs[0])}) != 1
            or state[1] not in {"bf16", "fp32"} or state[2] != "ND"
            or outputs[1:] != (state, state)):
        return None, _issue(
            kind, "unsupported_recurrent_parameter_geometry",
            "The retained recurrent ABI requires explicit gate/beta axes, "
            "matching A_log/dt_bias parameters and consistent operand precisions. "
            "Q/K/V and state backing shapes are not interpreted as active views.")
    return (
        "recurrent_kda_parameters_active", kind, (gate, beta, a_log, dt_bias),
        tuple(slot[1:] for slot in inputs[:3]), state[1:], outputs[0][1:],
    ), None


_PROJECTORS = {
    QUANT_INDEXER: _quant_indexer, "HcPre": _hc_pre, "HcPost": _hc_post,
    "Compressor": _compressor, "RecurrentKda": _recurrent_kda,
}


def _project(row):
    native = native_tensor_contract(row)
    if native is None:
        return None, native_tensor_diagnostic(row)
    if row.kind in MOE_FUSION_TYPES:
        return moe_fusion_contract(native)
    if row.kind in QUANT_FUSION_TYPES:
        return quant_fusion_contract(native)
    if row.kind in _GAPS:
        reason, detail = _GAPS[row.kind]
        issue = _issue(row.kind, reason, detail)
        if row.kind == "AttnResFwd":
            issue["required_runtime_evidence"] = list(_ATTN_RES_RUNTIME_FIELDS)
            issue["recorded_slot_counts"] = {"inputs": len(native[2]), "outputs": len(native[3])}
            issue["source_interface_evidence"] = {
                **_ATTN_RES_SOURCE_INTERFACE,
                "recorded_slot_count_matches": (len(native[2]), len(native[3]))
                == (ATTN_RES_INPUT_SLOT_COUNT, ATTN_RES_OUTPUT_SLOT_COUNT),
            }
        return None, issue
    _, _, inputs, outputs = native
    if (len(inputs), len(outputs)) not in _ARITIES[row.kind]:
        return None, _issue(
            row.kind, "unsupported_native_tensor_slots",
            "The captured slot count differs from the source-verified native ABI.")
    return _PROJECTORS[row.kind](inputs, outputs)


def observed_fusion_contract(row):
    if row.kind not in FUSION_TYPES:
        return None
    return _project(row)[0]


def fusion_contract_diagnostic(row):
    if row.kind not in FUSION_TYPES:
        return None
    contract, issue = _project(row)
    return issue or {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "operator_type": row.kind,
        "detail": "Compared active tensor descriptors are available; values, dispatch "
                  "arguments and model occurrence are not established.",
    }


def fusion_contract_scope(row):
    if row.kind not in FUSION_TYPES:
        return None
    if row.kind in MOE_FUSION_TYPES:
        return moe_fusion_scope(row.kind)
    if row.kind in QUANT_FUSION_TYPES:
        return quant_fusion_scope(row.kind)
    scope = {
        "occurrence_identity": False,
        "timing_scope": "inclusive_fused_subgraph; no_member_time_split",
        "unobserved": ["tensor_values", "scalar_dispatch_arguments", "selected_tiling",
                       "module_or_layer_identity"],
    }
    if row.kind == QUANT_INDEXER:
        return {
            **scope, "kind": "active_quantized_indexer_selection_descriptors",
            "compared": ["native_operator_type", "query_weight_and_query_scale_slots",
                         "key_cache_and_scale_head_slots", "sparse_index_output_slot"],
            "not_compared": ["cache_capacity", "cache_block_size", "block_table_shape_and_values",
                             "sequence_and_metadata_tensor_values", "compression_ratio",
                             "quantization_scalar_modes", "sparse_values_output", "phase"],
            "excluded_semantic_work": ["query_projection", "weight_projection",
                                       "index_cache_build", "candidate_filter"],
        }
    if row.kind in {"HcPre", "HcPost"}:
        return {
            **scope, "kind": "active_hyper_connection_descriptors",
            "compared": ["native_operator_type", "canonical_input_slots", "primary_output_slots"],
            "not_compared": ["leading_token_axis_grouping", "epsilon", "sinkhorn_iterations",
                             "parameter_values", "extra_pre_output"],
            "supported_branch": "pre_mix_absent" if row.kind == "HcPre" else "residual_post_mix",
        }
    if row.kind == "Compressor":
        return {
            **scope, "kind": "active_compressor_descriptors",
            "compared": ["native_operator_type", "token_shape_dtype_format",
                         "projection_position_and_norm_parameter_slots", "state_channel_slot",
                         "rope_width_slots", "output_width_slot"],
            "not_compared": ["state_cache_capacity", "state_cache_block_size",
                             "reserved_output_rows", "number_of_completed_groups",
                             "rope_position_values", "block_table_and_sequence_values",
                             "start_positions", "norm_epsilon", "rotary_mode",
                             "cache_mode", "state_cache_strides", "phase"],
            "excluded_semantic_work": ["compressed_kv_cache_store"],
        }
    if row.kind == "RecurrentKda":
        return {
            **scope, "kind": "recurrent_gate_and_parameter_descriptors",
            "compared": ["native_operator_type", "gate_beta_and_parameter_slots",
                         "query_key_value_dtype_format", "state_dtype_format", "output_dtype_format"],
            "not_compared": ["query_key_value_active_shapes", "output_active_shape",
                             "backing_storage_capacity", "state_shape_and_strides",
                             "state_indices_and_sequence_values", "logical_layout_scalar",
                             "qk_normalization_gate_beta_and_state_update_flags", "scale",
                             "safe_gate_and_lower_bound"],
            "limitation": "The descriptor match is a parameter-domain candidate; "
                          "it does not reconstruct Q/K/V or state active views.",
        }
    if row.kind == "AttnResFwd":
        return {
            **scope, "kind": "native_implementation_boundary_only",
            "compared": [],
            "not_compared": ["active_tensor_geometry", "native_stage_or_view_identity",
                             "history_depth", "add_output_norm_and_writeback_work"],
            "required_runtime_evidence": list(_ATTN_RES_RUNTIME_FIELDS),
            "source_interface_evidence": dict(_ATTN_RES_SOURCE_INTERFACE),
            "limitation": _GAPS[row.kind][1],
        }
    return {
        **scope, "kind": "native_implementation_boundary_only",
        "compared": [], "not_compared": ["active_tensor_geometry", "native_stage_or_view_identity"],
        "limitation": _GAPS[row.kind][1],
    }
