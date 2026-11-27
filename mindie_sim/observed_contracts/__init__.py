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

import sys

from . import attention as _attention
from . import fusion as _fusion
from . import matrix as _matrix
from . import moe as _moe
from . import signatures as _signatures
from . import vocabulary as _vocabulary

for _alias_name, _alias_mod in (
    ("abi", _vocabulary),
    ("attention_contracts", _attention),
    ("carriers", _vocabulary),
    ("errors", _vocabulary),
    ("fused_contracts", _fusion),
    ("fusion_contracts", _fusion),
    ("layout_contracts", _vocabulary),
    ("matrix_contracts", _matrix),
    ("moe_contracts", _moe),
    ("native_signature", _signatures),
    ("quant_contracts", _moe),
    ("router_contracts", _matrix),
    ("signature_matching", _signatures),
    ("signature_vocabulary", _vocabulary),
    ("subgraph_contracts", _fusion),
    ("targets", _vocabulary),
    ("tensor_contracts", _signatures),
):
    sys.modules.setdefault(f"{__name__}.{_alias_name}", _alias_mod)

from .vocabulary import (
    ABI_SCHEMA, DTYPES, native_abi_digest, native_abi_from_contract,
    native_abi_from_slots, native_abi_identity, native_abi_record,
    native_contract_issue, native_tensor_contract, native_tensor_diagnostic,
)
from .attention import (
    ATTENTION_CACHE_TYPES, MQSMLA_KINDS, attention_contract_diagnostic,
    attention_contract_scope, fia_block_table_slot, observed_attention_contract,
)
from .vocabulary import _snapshot_structure_carrier
from .vocabulary import StructureTargetError
from .fusion import (
    FUSED_TYPES, fused_contract_diagnostic, fused_contract_scope,
    observed_fused_contract,
)
from .fusion import (
    FUSION_TYPES, QWEN4_HC_KINDS, fusion_contract_diagnostic, fusion_contract_scope,
    observed_fusion_contract,
)
from .vocabulary import (
    LAYOUT_BODY_TYPES, layout_contract_diagnostic, layout_contract_scope,
    observed_layout_contract,
)
from .matrix import (
    MATRIX_BODY_TYPES, matrix_contract_diagnostic, matrix_contract_scope,
    observed_matrix_contract,
)
from .moe import MOE_FUSION_TYPES, moe_fusion_contract, moe_fusion_scope
from .signatures import (
    event_signature, native_call_signature_match, parse_paired_lanes,
)
from .moe import (
    QUANT_FUSION_TYPES, quant_fusion_contract, quant_fusion_scope,
)
from .matrix import (
    ROUTER_TYPES, observed_router_contract, router_contract_diagnostic,
    router_contract_scope,
)
from .fusion import (
    SUBGRAPH_TYPES, subgraph_contract, subgraph_contract_scope,
)
from .vocabulary import StructureTargets
from .signatures import (
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
