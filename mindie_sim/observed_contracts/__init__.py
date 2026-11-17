"""Canonical observed tensor, operator and subgraph contract vocabulary.

[Role]
  Publish the one ``mindie-sim`` surface that projects captured native tensor
  descriptors onto independently declared operator/subgraph contracts. Both
  ``mindie-sim`` parity tests and ``mindie-profiler`` lazy shims consume it.

[Invariants]
  * Every module here is pure metadata processing: no capture I/O, device
    context, profiler import or model-catalogue lookup.
  * Public names are re-exported unchanged; the barrel adds no second schema.
  * ``StructureTargetError`` is the only carrier-validation failure and stays
    a ``ValueError`` for callers that already treat bad payloads as data errors.
  * ``FIELDS``/``CONTRACT_FIELDS`` and ``NATIVE_FAMILIES``/
    ``NATIVE_CONTRACT_FAMILIES`` are alias pairs, never divergent copies.

[Timeouts & Locks]
  Import-time module loading only; no waits, timers or locks.

[Side-Effect Policy]
  Import performs no I/O and mutates no global registry.
"""
from __future__ import annotations

from .abi import (
    ABI_SCHEMA, DTYPES, native_abi_digest, native_abi_from_contract,
    native_abi_from_slots, native_abi_identity, native_abi_record,
    native_contract_issue, native_tensor_contract, native_tensor_diagnostic,
)
from .attention_contracts import (
    ATTENTION_CACHE_TYPES, MQSMLA_KINDS, attention_contract_diagnostic,
    attention_contract_scope, fia_block_table_slot, observed_attention_contract,
)
from .carriers import _snapshot_structure_carrier
from .errors import StructureTargetError
from .fused_contracts import (
    FUSED_TYPES, fused_contract_diagnostic, fused_contract_scope,
    observed_fused_contract,
)
from .fusion_contracts import (
    FUSION_TYPES, QWEN4_HC_KINDS, fusion_contract_diagnostic, fusion_contract_scope,
    observed_fusion_contract,
)
from .layout_contracts import (
    LAYOUT_BODY_TYPES, layout_contract_diagnostic, layout_contract_scope,
    observed_layout_contract,
)
from .matrix_contracts import (
    MATRIX_BODY_TYPES, matrix_contract_diagnostic, matrix_contract_scope,
    observed_matrix_contract,
)
from .moe_contracts import MOE_FUSION_TYPES, moe_fusion_contract, moe_fusion_scope
from .native_signature import (
    event_signature, native_call_signature_match, parse_paired_lanes,
)
from .quant_contracts import (
    QUANT_FUSION_TYPES, quant_fusion_contract, quant_fusion_scope,
)
from .router_contracts import (
    ROUTER_TYPES, observed_router_contract, router_contract_diagnostic,
    router_contract_scope,
)
from .subgraph_contracts import (
    SUBGRAPH_TYPES, subgraph_contract, subgraph_contract_scope,
)
from .targets import StructureTargets
from .tensor_contracts import (
    NATIVE_BODY_TYPES, SUPPORTED_TYPES, observed_contract,
    observed_contract_diagnostic, observed_contract_scope,
)
from .vocabulary import (
    AGENT_DIRECTIVE, CONTRACT_FIELDS, FIELDS, NATIVE_CONTRACT_FAMILIES,
    NATIVE_FAMILIES, ONBOARDING_SCHEMA, ONBOARDING_TARGETS, ONBOARDING_STATUS,
    SSOT_OWNER_REPO, build_unmodeled_onboarding_hint, infer_subgraph_candidate,
    mismatch_diagnostic,
)

__all__ = [
    "ABI_SCHEMA", "AGENT_DIRECTIVE", "ATTENTION_CACHE_TYPES", "CONTRACT_FIELDS",
    "DTYPES", "FIELDS", "FUSED_TYPES", "FUSION_TYPES", "LAYOUT_BODY_TYPES",
    "MATRIX_BODY_TYPES", "MOE_FUSION_TYPES", "MQSMLA_KINDS",
    "NATIVE_BODY_TYPES", "NATIVE_CONTRACT_FAMILIES", "NATIVE_FAMILIES",
    "ONBOARDING_SCHEMA", "ONBOARDING_TARGETS", "ONBOARDING_STATUS",
    "QUANT_FUSION_TYPES", "QWEN4_HC_KINDS", "ROUTER_TYPES", "SSOT_OWNER_REPO",
    "SUBGRAPH_TYPES", "SUPPORTED_TYPES", "StructureTargetError",
    "StructureTargets", "_snapshot_structure_carrier",
    "attention_contract_diagnostic", "attention_contract_scope",
    "build_unmodeled_onboarding_hint", "event_signature",
    "fia_block_table_slot", "fused_contract_diagnostic", "fused_contract_scope",
    "fusion_contract_diagnostic", "fusion_contract_scope",
    "infer_subgraph_candidate", "layout_contract_diagnostic",
    "layout_contract_scope", "matrix_contract_diagnostic",
    "matrix_contract_scope", "mismatch_diagnostic", "moe_fusion_contract",
    "moe_fusion_scope", "native_abi_digest", "native_abi_from_contract",
    "native_abi_from_slots", "native_abi_identity", "native_abi_record",
    "native_call_signature_match", "native_contract_issue",
    "native_tensor_contract", "native_tensor_diagnostic",
    "observed_attention_contract", "observed_contract",
    "observed_contract_diagnostic", "observed_contract_scope",
    "observed_fused_contract", "observed_fusion_contract",
    "observed_layout_contract", "observed_matrix_contract",
    "observed_router_contract", "parse_paired_lanes", "quant_fusion_contract",
    "quant_fusion_scope", "router_contract_diagnostic", "router_contract_scope",
    "subgraph_contract", "subgraph_contract_scope",
]
