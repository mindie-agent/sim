"""Observed contracts for source-defined model fusion boundaries.

[Role]
  Project captured native tensor descriptors for fused model-body operators
  into active observable contracts, and report the support gaps that block a
  projection instead of filling them with guessed geometry.

[Invariants]
  * Native operator types and optional execution branches stay distinct; one
    stage of a multi-launch implementation never claims the whole subgraph.
  * Backing allocations never become guessed active tensor views, and missing
    active shapes are reported as capture-metadata gaps.
  * ``ChunkKdaFwd`` stays a stage-dispatch gap while ``RecurrentKda`` keeps its
    own parameter-domain projector; neither contract absorbs the other.
  * Qwen4 high-connection kinds are registered only as explicit
    ``active_native_abi_unverified`` gaps; no active ABI is invented for them.
  * ``TimeoutError`` always propagates before any data-error handler.

[Timeouts & Locks]
  Pure bounded work over one captured tensor descriptor; no waits or locks.

[Side-Effect Policy]
  No I/O, simulation imports, hardware contexts or mutation.
"""
from __future__ import annotations

import math

from .abi import native_tensor_contract, native_tensor_diagnostic
from .moe_contracts import MOE_FUSION_TYPES, moe_fusion_contract, moe_fusion_scope
from .quant_contracts import QUANT_FUSION_TYPES, quant_fusion_contract, quant_fusion_scope


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
_FLOAT = {"bf16", "fp16"}
_EMPTY = ((), None, None)
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


def _issue(kind, reason, detail, missing=()):
    return {
        "category": "capture_metadata_missing" if missing else "contract_support_gap",
        "reason": reason, "detail": detail, "operator_type": kind,
        **({"missing_fields": list(missing)} if missing else {}),
    }


def _tensor(slot, rank, dtypes):
    shape, dtype, layout = slot
    return (len(shape) == rank and all(value > 0 for value in shape)
            and dtype in dtypes and layout == "ND")


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
