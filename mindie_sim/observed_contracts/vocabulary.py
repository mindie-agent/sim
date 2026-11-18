"""Canonical observed-contract vocabulary, field lineage and onboarding hints.

[Role]
  Own the finite observed contract-family vocabulary, the descriptor fields
  each family compares, mismatch explanations, and the deterministic
  ``mindie-unmodeled-structure-hint/1`` payload that points an agent at the one
  repository (``mindie-sim``) where a new operator or subgraph is onboarded.

[Invariants]
  * ``FIELDS`` keeps every public family name used by the observed contract
    modules, including the five paged-attention, MX-quant and causal-conv
    families that previously drifted out of the profiler copy.
  * ``NATIVE_FAMILIES`` names only families whose second tuple element is a
    native operator identity; non-native semantic families stay out of it.
  * New vocabulary aliases never rename or delete an existing public name.
  * ``StructureTargetError`` remains the shared carrier failure type.

[Timeouts & Locks]
  Pure declarations plus bounded tuple comparisons; no waits or locks.

[Side-Effect Policy]
  Import-time declarations only; no I/O, catalogue or device import.
"""
from __future__ import annotations

from .errors import StructureTargetError

FIELDS = {
    "matrix": ("family", "geometry", "input_dtype", "weight_dtype", "output_dtype",
               "scale_shape", "scale_dtype"),
    "matrix_bias": ("family", "geometry", "input_dtype", "weight_dtype", "output_dtype",
                    "bias_shape", "bias_dtype"),
    "grouped_matrix": ("family", "geometry", "input_dtype", "weight_dtype", "output_dtype",
                       "scale_shape", "scale_dtype"),
    "slice_copy": ("family", "input_shape", "output_shape", "input_dtype", "output_dtype"),
    "rmsnorm": ("family", "shape", "dtype", "has_weight"),
    "add_rmsnorm": ("family", "shape", "dtype", "has_weight"),
    "DynamicMxQuant": ("family", "shape", "input_dtype", "output_dtype", "scale_shape",
                       "scale_dtype"),
    "AddRmsNormDynamicMxQuant": (
        "family", "shape", "input_dtype", "output_dtype", "scale_shape", "scale_dtype"),
    "native_tensor_abi": ("family", "native_operator", "input_slots", "output_slots"),
    "attention_active_qkv": ("family", "native_operator", "qkv_slots", "main_output_slots"),
    "attention_paged_active_qkv": (
        "family", "native_operator", "query_slot",
        "cache_block_size_packed_row_width_dtype_format", "main_output_slot",
        "block_table_batch_count"),
    "attention_paged_mla_qk_rope": (
        "family", "native_operator", "query_slot", "query_rope_slot",
        "cache_heads_block_size_width_dtype_format", "output_slot",
        "block_table_batch_count"),
    "kv_cache_store_active": ("family", "native_operator", "kv_slots", "slot_mapping",
                              "cache_dtypes_layouts"),
    "kv_cache_store_active_elements": ("family", "native_operator", "kv_elements_dtype",
                                       "slot_mapping_shape"),
    "sparse_attention_active": ("family", "native_operator", "query_slots",
                                "cache_head_slots", "output_slot"),
    "gated_norm_active_rows": ("family", "native_operator", "rows_slot", "weight_slot"),
    "lightning_indexer_active": ("family", "native_operator", "query_and_weight_slots",
                                 "key_cache_head_slot", "sparse_index_output_slot"),
    "mla_prolog_active": ("family", "native_operator", "token_rows_slot",
                          "parameter_shape_dtype_slots", "cache_head_slots", "output_slots"),
    "quant_lightning_indexer_active": (
        "family", "native_operator", "query_weight_scale_slots",
        "key_cache_and_scale_head_slots", "sparse_index_output_slot"),
    "hyper_connection_active": (
        "family", "native_operator", "canonical_input_slots", "primary_output_slots"),
    "compressor_active": (
        "family", "native_operator", "token_slot", "parameter_slots",
        "state_channel_slot", "rope_width_slots", "output_width_slot"),
    "recurrent_kda_parameters_active": (
        "family", "native_operator", "gate_beta_parameter_slots",
        "qkv_dtype_format", "state_dtype_format", "output_dtype_format"),
    "grouped_mlp_parameters_active": (
        "family", "native_operator", "expert_and_channel_geometry",
        "input_weight_output_dtypes"),
    "dequant_swiglu_quant_active": (
        "family", "native_operator", "input_slot", "weight_scale_slot", "token_scale_slot",
        "output_slot", "output_scale_slot"),
    "rmsnorm_mxquant_active": (
        "family", "native_operator", "input_slot", "weight_slot", "output_slot", "scale_slot"),
    "dynamic_quant_active": (
        "family", "native_operator", "input_slot", "output_slot", "scale_slot"),
    "dynamic_quant_v2_active": (
        "family", "native_operator", "input_slot", "output_slot", "scale_slot"),
    "add_rmsnorm_dynamic_quant_active": (
        "family", "native_operator", "input_residual_weight_affine_slots",
        "primary_output_slots"),
    "situ_mxquant_active": (
        "family", "native_operator", "input_slot", "output_slot", "scale_slot"),
    "causal_conv1d_active": (
        "family", "native_operator", "input_slot", "weight_slot", "output_slot"),
    "layernorm_active": (
        "family", "native_operator", "input_slot", "weight_slot", "bias_slot", "output_slot"),
    "transpose_batchmatmul_active": (
        "family", "native_operator", "batch_geometry", "input_dtype", "weight_dtype",
        "output_dtype"),
    "router_selection_active": (
        "family", "native_operator", "selection_mode", "logits_slot", "weights_slot",
        "indices_slot"),
}

NATIVE_FAMILIES = {
    "native_tensor_abi", "attention_active_qkv", "attention_paged_active_qkv",
    "attention_paged_mla_qk_rope", "kv_cache_store_active",
    "kv_cache_store_active_elements",
    "sparse_attention_active", "gated_norm_active_rows",
    "lightning_indexer_active", "mla_prolog_active",
    "quant_lightning_indexer_active", "hyper_connection_active", "compressor_active",
    "recurrent_kda_parameters_active",
    "grouped_mlp_parameters_active", "dequant_swiglu_quant_active", "rmsnorm_mxquant_active",
    "dynamic_quant_active", "dynamic_quant_v2_active", "add_rmsnorm_dynamic_quant_active",
    "situ_mxquant_active", "causal_conv1d_active",
    "layernorm_active", "transpose_batchmatmul_active",
    "router_selection_active",
}

# Public aliases: the canonical vocabulary keeps the historical names above.
CONTRACT_FIELDS = FIELDS
NATIVE_CONTRACT_FAMILIES = NATIVE_FAMILIES

ONBOARDING_SCHEMA = "mindie-unmodeled-structure-hint/1"
ONBOARDING_STATUS = "unmodeled_in_sim"
SSOT_OWNER_REPO = "mindie-sim"
ONBOARDING_TARGETS = (
    "mindie_sim/structure_native_contracts.py",
    "mindie_sim/observed_contracts/attention_contracts.py",
    "mindie_sim/structure_subgraphs.py",
    "mindie_sim/subgraph_math.py",
)
AGENT_DIRECTIVE = (
    "Onboard this operator/subgraph ONCE in mindie-sim (declared contract + "
    "observed_contracts + subgraph_math). Do NOT add local operator/subgraph "
    "contract rules to mindie-profiler."
)
_CANDIDATE_TOKENS = (
    ("attention", "attention"), ("flash", "attention"), ("mla", "attention"),
    ("index", "indexer"), ("topk", "indexer"), ("moe", "moe_ffn"),
    ("expert", "moe_ffn"), ("swiglu", "dense_ffn"), ("rmsnorm", "norm_quant"),
    ("quant", "norm_quant"), ("conv", "recurrent_gdn_kda"), ("gdn", "recurrent_gdn_kda"),
    ("kda", "recurrent_gdn_kda"), ("hc", "mhc_residual"), ("residual", "mhc_residual"),
)


def infer_subgraph_candidate(observed_kernel_name=None, observed_op_type=None):
    """Infer one canonical semantic family from an observed operator name only."""
    text = " ".join(str(value) for value in (observed_op_type, observed_kernel_name) if value)
    lowered = text.casefold()
    for token, candidate in _CANDIDATE_TOKENS:
        if token in lowered:
            return candidate
    return None


def build_unmodeled_onboarding_hint(
    *,
    observed_kernel_name=None,
    observed_op_type=None,
    observed_abi_summary=None,
    inferred_subgraph_candidate=None,
):
    """Build the deterministic single-onboarding-point payload for one gap."""
    candidate = inferred_subgraph_candidate
    if candidate is None:
        candidate = infer_subgraph_candidate(observed_kernel_name, observed_op_type)
    return {
        "schema": ONBOARDING_SCHEMA,
        "status": ONBOARDING_STATUS,
        "observed_kernel_name": observed_kernel_name,
        "observed_op_type": observed_op_type,
        "observed_abi_summary": observed_abi_summary,
        "inferred_subgraph_candidate": candidate,
        "ssot_owner_repo": SSOT_OWNER_REPO,
        "onboarding_targets": list(ONBOARDING_TARGETS),
        "agent_directive": AGENT_DIRECTIVE,
    }


def observed_contract_diagnostic(row):
    """Delegate to the canonical observed diagnostic without an import cycle."""
    from .tensor_contracts import observed_contract_diagnostic as _diagnostic

    return _diagnostic(row)


def mismatch_diagnostic(contract, declared):
    """Explain a declared-contract mismatch without proposing a mapping."""
    family = contract[0]
    same_family = [
        other for other in declared if other[0] == family
        and (family not in NATIVE_FAMILIES or other[1] == contract[1])
    ]
    if not same_family:
        return {
            "category": "declared_structure_mismatch",
            "reason": "declared_operator_family_absent",
            "detail": "No declared contract in the selected rank/scope has this operator family.",
        }
    # Exact equality is handled by the mapper. These comparisons only explain
    # which declaration needs inspection; no nearest node is ever selected.
    names = FIELDS.get(family, ("family", "shape"))
    differences = []
    for other in same_family:
        fields = [names[index] if index < len(names) else f"field_{index}"
                  for index in range(max(len(contract), len(other)))
                  if index >= len(contract) or index >= len(other)
                  or contract[index] != other[index]]
        differences.append((fields, other))
    differences.sort(key=lambda pair: len(pair[0]))
    best = differences[0][0]
    same_geometry = [fields for fields, other in differences if other[1] == contract[1]]
    precision_fields = {
        "input_dtype", "weight_dtype", "output_dtype", "bias_dtype", "scale_shape",
        "scale_dtype",
    }
    if family in {"matrix", "matrix_bias", "grouped_matrix"} and any(
        fields and set(fields) <= precision_fields for fields in same_geometry
    ):
        reason = "declared_dtype_or_quantization_mismatch"
    elif any(field in best for field in ("geometry", "shape", "input_shape", "output_shape")):
        reason = "declared_geometry_or_scope_mismatch"
    else:
        reason = "declared_tensor_descriptor_mismatch"
    return {
        "category": "declared_structure_mismatch", "reason": reason,
        "detail": ("Compare the selected execution geometry, precision and scope with the "
                   "capture; tensor descriptors alone do not establish prefill/decode or "
                   "draft ownership."),
        "near_contracts": [
            {"contract": other, "differing_fields": fields}
            for fields, other in differences[:3]
        ],
    }


__all__ = [
    "AGENT_DIRECTIVE", "CONTRACT_FIELDS", "FIELDS", "NATIVE_CONTRACT_FAMILIES",
    "NATIVE_FAMILIES", "ONBOARDING_SCHEMA", "ONBOARDING_TARGETS", "ONBOARDING_STATUS",
    "SSOT_OWNER_REPO", "StructureTargetError", "build_unmodeled_onboarding_hint",
    "infer_subgraph_candidate", "mismatch_diagnostic", "observed_contract_diagnostic",
]
