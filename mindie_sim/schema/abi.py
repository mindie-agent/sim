"""Unified hardware-usable operator ABI contracts (CANN and non-CANN backends).

[Role] Define immutable L0 ABI query, probed-entry, support-verdict, and sealed
  catalog-snapshot contracts covering every Ascend hardware-usable operator
  backend: CANN packages (``cann_aclnn``, ``hccl_mc2``) AND non-CANN packages
  (``vllm_ascend_csrc``, ``triton``, ``torch_npu``, ``catlass``,
  ``third_party_wheel``), plus open target-synthesis backends (``tilelang``,
  ``ascendc``, ``catlass``).
[Invariants] Zero default-admit or empty-set wildcards. Every on-device entry
  carries explicit ``available_on_device``, backend kind, symbol/kernel, arch,
  stack/CANN version, role, jointly constrained ``dtype_tuple``, algebraic
  ``shape_constraints``, and ``source_ref`` provenance. Zero runtime
  uncalibrated states or enums exist here.
[Timeouts & Locks] Pure immutable validation and SHA-256 hashing; no I/O,
  device calls, or locks.
[Side-Effect Policy] Strictly frozen dataclasses; zero global mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Literal, TypeAlias

from .common import ModelingError
from .datapath import SourceRef
from .stack import STACK_TRACKS, StackTrack

ABI_CATALOG_SCHEMA = "mindie-sim.abi-catalog/1"
SHA256_HEX_LENGTH = 64
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

SocArch: TypeAlias = Literal["A2", "A3", "A5"]
SOC_ARCHS: frozenset[str] = frozenset({"A2", "A3", "A5"})

AbiBackendKind: TypeAlias = Literal[
    "cann_aclnn",
    "vllm_ascend_csrc",
    "triton",
    "torch_npu",
    "catlass",
    "third_party_wheel",
    "hccl_mc2",
    "tilelang",
    "ascendc",
    "logical",
]
CANN_PACKAGE_ABI_BACKENDS: frozenset[str] = frozenset({
    "cann_aclnn",
    "hccl_mc2",
})
NON_CANN_ABI_BACKENDS: frozenset[str] = frozenset({
    "vllm_ascend_csrc",
    "triton",
    "torch_npu",
    "catlass",
    "third_party_wheel",
})
HARDWARE_USABLE_ABI_BACKENDS: frozenset[str] = (
    CANN_PACKAGE_ABI_BACKENDS | NON_CANN_ABI_BACKENDS
)
OPEN_SYNTHESIS_ABI_BACKENDS: frozenset[str] = frozenset({
    "tilelang",
    "ascendc",
    "catlass",
})
ABI_BACKEND_KINDS: frozenset[str] = (
    HARDWARE_USABLE_ABI_BACKENDS | OPEN_SYNTHESIS_ABI_BACKENDS | frozenset({"logical"})
)

RegionBackendTarget: TypeAlias = Literal[
    "cann_aclnn",
    "vllm_ascend_csrc",
    "triton",
    "torch_npu",
    "catlass",
    "third_party_wheel",
    "hccl_mc2",
    "tilelang",
    "ascendc",
]
REGION_BACKEND_TARGETS: frozenset[str] = (
    HARDWARE_USABLE_ABI_BACKENDS | OPEN_SYNTHESIS_ABI_BACKENDS
)

FusionTier: TypeAlias = Literal[
    "K1_shipping",
    "K1_proposed_pr",
    "K2_epilogue",
    "K3_dsl_pipeline",
]
FUSION_TIERS: frozenset[str] = frozenset({
    "K1_shipping",
    "K1_proposed_pr",
    "K2_epilogue",
    "K3_dsl_pipeline",
})

RankPlacement: TypeAlias = Literal[
    "local",
    "hccs_single_node",
    "hccs_supernode_a3",
    "a2_switched_multinode",
    "ub_ccu_intramachine",
    "ub_aicpu_supernode",
    "urma_cross_supernode",
    "layered_ub_roce",
    "roce_flat_cross_node",
]
RANK_PLACEMENTS: frozenset[str] = frozenset({
    "local",
    "hccs_single_node",
    "hccs_supernode_a3",
    "a2_switched_multinode",
    "ub_ccu_intramachine",
    "ub_aicpu_supernode",
    "urma_cross_supernode",
    "layered_ub_roce",
    "roce_flat_cross_node",
})

AbiVerdictStatus: TypeAlias = Literal[
    "admitted_shipping",
    "admitted_proposed_pr",
    "rejected_abi",
    "custom_synthesis",
]
ABI_VERDICT_STATUSES: frozenset[str] = frozenset({
    "admitted_shipping",
    "admitted_proposed_pr",
    "rejected_abi",
    "custom_synthesis",
})

ModelScope: TypeAlias = Literal["target", "draft"]
MODEL_SCOPES: frozenset[str] = frozenset({"target", "draft"})

OperandLayout: TypeAlias = Literal[
    "ND",
    "FRACTAL_NZ",
    "FRACTAL_NZ_C0_32",
    "FRACTAL_NZ_C0_4",
    "FRACTAL_NZ_C0_2",
]
OPERAND_LAYOUTS: frozenset[str] = frozenset({
    "ND",
    "FRACTAL_NZ",
    "FRACTAL_NZ_C0_32",
    "FRACTAL_NZ_C0_4",
    "FRACTAL_NZ_C0_2",
})

MlaVariantKind: TypeAlias = Literal["none", "latent_mla", "latent_mqa"]
MLA_VARIANT_KINDS: frozenset[str] = frozenset({"none", "latent_mla", "latent_mqa"})

ManifestEntryTrack: TypeAlias = Literal["merged", "unmerged_pr"]
MANIFEST_ENTRY_TRACKS: frozenset[str] = frozenset({"merged", "unmerged_pr"})


def _req_str(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


def _int_val(value: Any, code: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelingError(f"{code}:{value!r}")
    if minimum is not None and value < minimum:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _int_tuple(values: Sequence[int], code: str, *, minimum: int = 0) -> tuple[int, ...]:
    return tuple(_int_val(v, code, minimum=minimum) for v in values)


def _str_tuples(
    values: Sequence[Sequence[str]], code: str,
) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for row in values:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ModelingError(code)
        items = tuple(_req_str(item, code) for item in row)
        if not items:
            raise ModelingError(code)
        out.append(items)
    return tuple(out)


def _sha256_canonical(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, code: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_RE.fullmatch(value):
        raise ModelingError(f"{code}:{value!r}")
    return value


def _validate_nonempty_source_ref(ref: SourceRef, code: str) -> None:
    if not isinstance(ref, SourceRef):
        raise ModelingError(code)
    for field_name in ("repository", "revision", "relative_path", "source_sha256"):
        val = getattr(ref, field_name, "")
        if not isinstance(val, str) or not val.strip():
            raise ModelingError(f"{code}:{field_name}_required")
    _validate_sha256(ref.source_sha256, f"{code}:source_sha256_invalid")


def _validate_bounds_pair(pair: Sequence[int], code: str) -> tuple[int, int]:
    vals = _int_tuple(pair, code, minimum=0)
    if len(vals) != 2 or (vals[1] > 0 and vals[1] < vals[0]):
        raise ModelingError(f"{code}:{vals!r}")
    return (vals[0], vals[1])


@dataclass(frozen=True, slots=True)
class OperandPortSig:
    """Exact per-port tensor contract derived from resolved OpSpec bindings."""

    role: str
    dtype: str
    layout: OperandLayout
    transposed: bool
    shape: tuple[int, ...]
    quant_mode: str
    group_shape: tuple[int, ...]
    scale_dtype: str
    scale_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _req_str(self.role, "operand_port_role_required"))
        object.__setattr__(self, "dtype", _req_str(self.dtype, "operand_port_dtype_required").lower())
        if self.layout not in OPERAND_LAYOUTS:
            raise ModelingError(f"operand_port_layout_invalid:{self.layout!r}")
        if not isinstance(self.transposed, bool):
            raise ModelingError("operand_port_transposed_invalid")
        object.__setattr__(self, "shape", _int_tuple(self.shape, "operand_port_shape_invalid", minimum=0))
        object.__setattr__(self, "quant_mode", _req_str(self.quant_mode, "operand_port_quant_mode_required").lower())
        object.__setattr__(self, "group_shape", _int_tuple(self.group_shape, "operand_port_group_shape_invalid", minimum=1))
        object.__setattr__(self, "scale_dtype", _req_str(self.scale_dtype, "operand_port_scale_dtype_required").lower())
        object.__setattr__(self, "scale_shape", _int_tuple(self.scale_shape, "operand_port_scale_shape_invalid", minimum=0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "dtype": self.dtype,
            "layout": self.layout,
            "transposed": self.transposed,
            "shape": list(self.shape),
            "quant_mode": self.quant_mode,
            "group_shape": list(self.group_shape),
            "scale_dtype": self.scale_dtype,
            "scale_shape": list(self.scale_shape),
        }


@dataclass(frozen=True, slots=True)
class AbiShapeConstraints:
    """Pure algebraic legality constraints from C++/Triton tiling and CheckParams."""

    k_min: int = 0
    k_max_exclusive: int = 0
    n_max_inclusive: int = 0
    require_even_n: bool = False
    require_even_k_for_fp4: bool = False
    require_even_n_for_fp4_transb: bool = False
    require_m_divisible_by_ws: bool = False
    max_comm_bytes_formula: str = "none"
    hidden_allowed_set: frozenset[int] = frozenset()
    hidden_range: tuple[int, int] = (0, 0)
    hidden_divisibility: int = 0
    inter_min: int = 0
    inter_max: int = 0
    inter_divisibility: int = 0
    topk_range: tuple[int, int] = (0, 0)
    max_shared_experts_per_rank: int = -1

    def __post_init__(self) -> None:
        k_min = _int_val(self.k_min, "abi_shape_k_min_invalid", minimum=0)
        k_max = _int_val(self.k_max_exclusive, "abi_shape_k_max_invalid", minimum=0)
        if k_max > 0 and k_max <= k_min:
            raise ModelingError(f"abi_shape_k_bounds_invalid:{k_min}>={k_max}")
        n_max = _int_val(self.n_max_inclusive, "abi_shape_n_max_invalid", minimum=0)
        inter_min = _int_val(self.inter_min, "abi_shape_inter_min_invalid", minimum=0)
        inter_max = _int_val(self.inter_max, "abi_shape_inter_max_invalid", minimum=0)
        if inter_max > 0 and inter_max < inter_min:
            raise ModelingError(f"abi_shape_inter_bounds_invalid:{inter_min}>{inter_max}")
        hidden_allowed = frozenset(
            _int_val(v, "abi_shape_hidden_allowed_invalid", minimum=1)
            for v in self.hidden_allowed_set
        )
        hidden_range = _validate_bounds_pair(self.hidden_range, "abi_shape_hidden_range_invalid")
        topk_range = _validate_bounds_pair(self.topk_range, "abi_shape_topk_range_invalid")
        object.__setattr__(self, "k_min", k_min)
        object.__setattr__(self, "k_max_exclusive", k_max)
        object.__setattr__(self, "n_max_inclusive", n_max)
        object.__setattr__(self, "inter_min", inter_min)
        object.__setattr__(self, "inter_max", inter_max)
        object.__setattr__(
            self, "hidden_divisibility",
            _int_val(self.hidden_divisibility, "abi_shape_hidden_div_invalid", minimum=0),
        )
        object.__setattr__(
            self, "inter_divisibility",
            _int_val(self.inter_divisibility, "abi_shape_inter_div_invalid", minimum=0),
        )
        object.__setattr__(
            self, "max_shared_experts_per_rank",
            _int_val(self.max_shared_experts_per_rank, "abi_shape_max_shared_invalid", minimum=-1),
        )
        object.__setattr__(self, "hidden_allowed_set", hidden_allowed)
        object.__setattr__(self, "hidden_range", hidden_range)
        object.__setattr__(self, "topk_range", topk_range)
        object.__setattr__(
            self, "max_comm_bytes_formula",
            _req_str(self.max_comm_bytes_formula, "abi_shape_comm_formula_required"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_min": self.k_min,
            "k_max_exclusive": self.k_max_exclusive,
            "n_max_inclusive": self.n_max_inclusive,
            "require_even_n": self.require_even_n,
            "require_even_k_for_fp4": self.require_even_k_for_fp4,
            "require_even_n_for_fp4_transb": self.require_even_n_for_fp4_transb,
            "require_m_divisible_by_ws": self.require_m_divisible_by_ws,
            "max_comm_bytes_formula": self.max_comm_bytes_formula,
            "hidden_allowed_set": sorted(self.hidden_allowed_set),
            "hidden_range": list(self.hidden_range),
            "hidden_divisibility": self.hidden_divisibility,
            "inter_min": self.inter_min,
            "inter_max": self.inter_max,
            "inter_divisibility": self.inter_divisibility,
            "topk_range": list(self.topk_range),
            "max_shared_experts_per_rank": self.max_shared_experts_per_rank,
        }


@dataclass(frozen=True, slots=True)
class AbiQueryKey:
    """Zero-default composite invocation signature derived strictly from OpSpecs."""

    soc_arch: SocArch
    sku_id: str
    abi_id: str
    graph_motif: str
    model_scope: ModelScope
    draft_quant_mode: str
    pr_track: StackTrack
    operands: tuple[OperandPortSig, ...]
    outputs: tuple[OperandPortSig, ...]
    rank_placement: RankPlacement
    world_size: int
    comm_group_id: str
    comm_domain_width: int
    comm_dtype: str
    reduction_kind: str
    activation_kind: str
    activation_params: tuple[tuple[str, float], ...]
    has_bias: bool
    has_active_mask: bool
    dispatch_quant_mode: int
    shared_experts_per_rank: int
    top_k: int
    moe_expert_num: int
    qk_rope_head_dim: int
    mla_variant: MlaVariantKind
    sfa_dcp_active: bool
    pcp_size: int
    kv_cache_dtype: str
    has_kpool_indexer: bool
    dcp_size: int
    is_act_and_mul: bool
    enabled_stack_passes: frozenset[str]

    def __post_init__(self) -> None:
        if self.soc_arch not in SOC_ARCHS:
            raise ModelingError(f"abi_query_invalid_soc_arch:{self.soc_arch!r}")
        object.__setattr__(self, "sku_id", _req_str(self.sku_id, "abi_query_empty_sku_id").lower())
        object.__setattr__(self, "abi_id", _req_str(self.abi_id, "abi_query_empty_abi_id"))
        object.__setattr__(self, "graph_motif", _req_str(self.graph_motif, "abi_query_empty_motif"))
        if self.model_scope not in MODEL_SCOPES:
            raise ModelingError(f"abi_query_invalid_model_scope:{self.model_scope!r}")
        if self.pr_track not in STACK_TRACKS:
            raise ModelingError(f"abi_query_invalid_pr_track:{self.pr_track!r}")
        if self.rank_placement not in RANK_PLACEMENTS:
            raise ModelingError(f"abi_query_invalid_rank_placement:{self.rank_placement!r}")
        if self.mla_variant not in MLA_VARIANT_KINDS:
            raise ModelingError(f"abi_query_invalid_mla_variant:{self.mla_variant!r}")
        ws = _int_val(self.world_size, "abi_query_invalid_world_size", minimum=1)
        dom_w = _int_val(self.comm_domain_width, "abi_query_invalid_domain_width", minimum=1)
        if ws == 1 and self.rank_placement != "local":
            raise ModelingError(f"abi_query_local_placement_mismatch:{self.rank_placement}")
        if ws > 1 and self.rank_placement == "local":
            raise ModelingError(f"abi_query_remote_placement_required:world_size={ws}")
        operands = tuple(self.operands)
        outputs = tuple(self.outputs)
        if not operands or not all(isinstance(o, OperandPortSig) for o in operands):
            raise ModelingError(f"abi_query_invalid_operands:{self.abi_id}")
        if not outputs or not all(isinstance(o, OperandPortSig) for o in outputs):
            raise ModelingError(f"abi_query_invalid_outputs:{self.abi_id}")
        act_params = []
        for k, v in self.activation_params:
            num = float(v)
            if isinstance(v, bool) or not math.isfinite(num):
                raise ModelingError(f"abi_query_invalid_activation_param:{k}")
            act_params.append((_req_str(k, "abi_query_invalid_activation_param_key"), num))
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "world_size", ws)
        object.__setattr__(self, "comm_domain_width", dom_w)
        object.__setattr__(self, "comm_group_id", _req_str(self.comm_group_id, "abi_query_comm_group_required"))
        object.__setattr__(self, "comm_dtype", _req_str(self.comm_dtype, "abi_query_comm_dtype_required").lower())
        object.__setattr__(self, "draft_quant_mode", _req_str(self.draft_quant_mode, "abi_query_draft_quant_required").lower())
        object.__setattr__(self, "reduction_kind", _req_str(self.reduction_kind, "abi_query_reduction_required").lower())
        object.__setattr__(self, "activation_kind", _req_str(self.activation_kind, "abi_query_activation_required").lower())
        object.__setattr__(self, "kv_cache_dtype", _req_str(self.kv_cache_dtype, "abi_query_kv_dtype_required").lower())
        object.__setattr__(self, "activation_params", tuple(act_params))
        object.__setattr__(self, "dispatch_quant_mode", _int_val(self.dispatch_quant_mode, "abi_query_dispatch_quant_invalid", minimum=-1))
        object.__setattr__(self, "shared_experts_per_rank", _int_val(self.shared_experts_per_rank, "abi_query_shared_experts_invalid", minimum=0))
        object.__setattr__(self, "top_k", _int_val(self.top_k, "abi_query_topk_invalid", minimum=0))
        object.__setattr__(self, "moe_expert_num", _int_val(self.moe_expert_num, "abi_query_experts_invalid", minimum=0))
        object.__setattr__(self, "qk_rope_head_dim", _int_val(self.qk_rope_head_dim, "abi_query_rope_dim_invalid", minimum=0))
        object.__setattr__(self, "pcp_size", _int_val(self.pcp_size, "abi_query_pcp_invalid", minimum=1))
        object.__setattr__(self, "dcp_size", _int_val(self.dcp_size, "abi_query_dcp_invalid", minimum=1))
        object.__setattr__(self, "enabled_stack_passes", frozenset(str(p) for p in self.enabled_stack_passes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "soc_arch": self.soc_arch,
            "sku_id": self.sku_id,
            "abi_id": self.abi_id,
            "graph_motif": self.graph_motif,
            "model_scope": self.model_scope,
            "draft_quant_mode": self.draft_quant_mode,
            "pr_track": self.pr_track,
            "operands": [o.to_dict() for o in self.operands],
            "outputs": [o.to_dict() for o in self.outputs],
            "rank_placement": self.rank_placement,
            "world_size": self.world_size,
            "comm_group_id": self.comm_group_id,
            "comm_domain_width": self.comm_domain_width,
            "comm_dtype": self.comm_dtype,
            "reduction_kind": self.reduction_kind,
            "activation_kind": self.activation_kind,
            "activation_params": [[k, v] for k, v in self.activation_params],
            "has_bias": self.has_bias,
            "has_active_mask": self.has_active_mask,
            "dispatch_quant_mode": self.dispatch_quant_mode,
            "shared_experts_per_rank": self.shared_experts_per_rank,
            "top_k": self.top_k,
            "moe_expert_num": self.moe_expert_num,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "mla_variant": self.mla_variant,
            "sfa_dcp_active": self.sfa_dcp_active,
            "pcp_size": self.pcp_size,
            "kv_cache_dtype": self.kv_cache_dtype,
            "has_kpool_indexer": self.has_kpool_indexer,
            "dcp_size": self.dcp_size,
            "is_act_and_mul": self.is_act_and_mul,
            "enabled_stack_passes": sorted(self.enabled_stack_passes),
        }

    def canonical_digest(self) -> str:
        """SHA-256 over every typed invocation field (zero unhashed loophole)."""
        return _sha256_canonical(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProbedAbiEntry:
    """One jointly constrained real-device ABI entry (CANN or non-CANN backend)."""

    entry_id: str
    abi_id: str
    available_on_device: bool
    backend_kind: AbiBackendKind
    symbol_or_kernel: str
    device_arch: SocArch
    stack_or_cann_version: str
    role: str
    dtype_tuple: tuple[tuple[str, ...], ...]
    shape_constraints: AbiShapeConstraints
    source_ref: SourceRef
    graph_motif: str = ""
    vllm_ascend_track: ManifestEntryTrack = "merged"
    required_stack_pass_id: str = ""
    required_mechanism_id: str = ""
    vllm_ascend_ref: str = ""
    op_plugin_ref: str = ""
    cann_check_params_ref: str = ""
    device_kernel_expansion: tuple[str, ...] = ()
    allowed_output_tuples: tuple[tuple[str, ...], ...] = ()
    allowed_rank_placements: frozenset[RankPlacement] = frozenset({"local"})
    allowed_world_sizes: frozenset[int] = frozenset({1})
    allowed_activations: frozenset[str] = frozenset({"none"})
    allowed_model_scopes: frozenset[ModelScope] = frozenset({"target", "draft"})
    allow_bias: bool = True
    require_no_bias: bool = False
    probe_signature_sha256: str = ""
    probe_domain_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _req_str(self.entry_id, "abi_entry_id_required"))
        object.__setattr__(self, "abi_id", _req_str(self.abi_id, "abi_entry_abi_id_required"))
        if not isinstance(self.available_on_device, bool):
            raise ModelingError("abi_entry_available_flag_invalid")
        if self.backend_kind not in ABI_BACKEND_KINDS:
            raise ModelingError(f"abi_entry_backend_invalid:{self.backend_kind!r}")
        object.__setattr__(
            self, "symbol_or_kernel",
            _req_str(self.symbol_or_kernel, "abi_entry_symbol_required"),
        )
        if self.device_arch not in SOC_ARCHS:
            raise ModelingError(f"abi_entry_arch_invalid:{self.device_arch!r}")
        object.__setattr__(
            self, "stack_or_cann_version",
            _req_str(self.stack_or_cann_version, "abi_entry_version_required"),
        )
        object.__setattr__(self, "role", _req_str(self.role, "abi_entry_role_required"))
        if not isinstance(self.shape_constraints, AbiShapeConstraints):
            raise ModelingError("abi_entry_shape_constraints_invalid")
        if not isinstance(self.source_ref, SourceRef):
            raise ModelingError("abi_entry_source_ref_invalid")
        if self.vllm_ascend_track not in MANIFEST_ENTRY_TRACKS:
            raise ModelingError(f"abi_entry_track_invalid:{self.vllm_ascend_track!r}")
        if self.allow_bias and self.require_no_bias:
            raise ModelingError(f"abi_entry_bias_contradiction:{self.entry_id}")
        dtypes = _str_tuples(self.dtype_tuple, f"abi_entry_dtype_tuple_invalid:{self.entry_id}")
        if not dtypes:
            raise ModelingError(f"abi_entry_empty_dtype_tuple:{self.entry_id}")
        outputs = (
            _str_tuples(self.allowed_output_tuples, f"abi_entry_output_tuple_invalid:{self.entry_id}")
            if self.allowed_output_tuples else ()
        )
        placements = frozenset(self.allowed_rank_placements)
        if not placements or any(p not in RANK_PLACEMENTS for p in placements):
            raise ModelingError(f"abi_entry_rank_placements_invalid:{self.entry_id}")
        world_sizes = frozenset(
            _int_val(ws, f"abi_entry_world_size_invalid:{self.entry_id}", minimum=1)
            for ws in self.allowed_world_sizes
        )
        if not world_sizes:
            raise ModelingError(f"abi_entry_empty_world_sizes:{self.entry_id}")
        activations = frozenset(_req_str(a, "abi_entry_activation_invalid").lower() for a in self.allowed_activations)
        if not activations:
            raise ModelingError(f"abi_entry_empty_activations:{self.entry_id}")
        scopes = frozenset(self.allowed_model_scopes)
        if not scopes or any(s not in MODEL_SCOPES for s in scopes):
            raise ModelingError(f"abi_entry_model_scopes_invalid:{self.entry_id}")
        expansion = tuple(_req_str(k, "abi_entry_kernel_expansion_invalid") for k in self.device_kernel_expansion)
        object.__setattr__(self, "dtype_tuple", dtypes)
        object.__setattr__(self, "allowed_output_tuples", outputs)
        object.__setattr__(self, "allowed_rank_placements", placements)
        object.__setattr__(self, "allowed_world_sizes", world_sizes)
        object.__setattr__(self, "allowed_activations", activations)
        object.__setattr__(self, "allowed_model_scopes", scopes)
        object.__setattr__(self, "device_kernel_expansion", expansion)
        self._validate_device_evidence()

    def _validate_device_evidence(self) -> None:
        if self.vllm_ascend_track == "unmerged_pr" and not self.required_mechanism_id.strip():
            raise ModelingError(f"abi_entry_proposed_missing_mechanism_id:{self.entry_id}")
        if not self.available_on_device:
            return
        if self.backend_kind not in HARDWARE_USABLE_ABI_BACKENDS:
            raise ModelingError(
                f"abi_entry_non_device_backend_marked_available:{self.entry_id}:{self.backend_kind}"
            )
        _validate_nonempty_source_ref(self.source_ref, f"abi_entry_source_ref_incomplete:{self.entry_id}")
        _validate_sha256(self.probe_signature_sha256, f"abi_entry_probe_sha256_invalid:{self.entry_id}")
        if not self.device_kernel_expansion:
            raise ModelingError(f"abi_entry_missing_kernel_expansion:{self.entry_id}")
        if self.backend_kind in CANN_PACKAGE_ABI_BACKENDS and not self.cann_check_params_ref.strip():
            raise ModelingError(f"abi_entry_cann_missing_check_params_ref:{self.entry_id}")

    @property
    def is_cann_package(self) -> bool:
        return self.backend_kind in CANN_PACKAGE_ABI_BACKENDS

    @property
    def is_non_cann_package(self) -> bool:
        return self.backend_kind in NON_CANN_ABI_BACKENDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "abi_id": self.abi_id,
            "available_on_device": self.available_on_device,
            "backend_kind": self.backend_kind,
            "symbol_or_kernel": self.symbol_or_kernel,
            "device_arch": self.device_arch,
            "stack_or_cann_version": self.stack_or_cann_version,
            "role": self.role,
            "dtype_tuple": [list(t) for t in self.dtype_tuple],
            "shape_constraints": self.shape_constraints.to_dict(),
            "source_ref": self.source_ref.to_dict(),
            "graph_motif": self.graph_motif,
            "vllm_ascend_track": self.vllm_ascend_track,
            "required_stack_pass_id": self.required_stack_pass_id,
            "required_mechanism_id": self.required_mechanism_id,
            "vllm_ascend_ref": self.vllm_ascend_ref,
            "op_plugin_ref": self.op_plugin_ref,
            "cann_check_params_ref": self.cann_check_params_ref,
            "device_kernel_expansion": list(self.device_kernel_expansion),
            "allowed_output_tuples": [list(t) for t in self.allowed_output_tuples],
            "allowed_rank_placements": sorted(self.allowed_rank_placements),
            "allowed_world_sizes": sorted(self.allowed_world_sizes),
            "allowed_activations": sorted(self.allowed_activations),
            "allowed_model_scopes": sorted(self.allowed_model_scopes),
            "allow_bias": self.allow_bias,
            "require_no_bias": self.require_no_bias,
            "probe_signature_sha256": self.probe_signature_sha256,
            "probe_domain_note": self.probe_domain_note,
        }


@dataclass(frozen=True, slots=True)
class AbiSupportVerdict:
    """Authoritative ABI admissibility verdict for CANN and non-CANN hardware ops."""

    abi_id: str
    status: AbiVerdictStatus
    available_on_device: bool
    backend_kind: AbiBackendKind
    device_arch: SocArch
    stack_or_cann_version: str
    act_dtype: str
    weight_dtype: str
    quant_mode: str
    probe_manifest_sha256: str
    engine_speedup_verified: bool
    query_digest: str
    matched_entry_id: str = ""
    symbol_or_kernel: str = ""
    matched_lowering_id: str = ""
    matched_lowering_sha256: str = ""
    pr_track: StackTrack = "shipping"
    proposed_pr_ref: str = ""
    rejection_reason: str = ""
    source_ref: SourceRef | None = None
    vllm_ascend_ref: str = ""
    op_plugin_ref: str = ""
    cann_check_params_ref: str = ""
    device_kernel_expansion: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "abi_id", _req_str(self.abi_id, "abi_verdict_abi_id_required"))
        if self.status not in ABI_VERDICT_STATUSES:
            raise ModelingError(f"abi_verdict_status_invalid:{self.status!r}")
        if not isinstance(self.available_on_device, bool):
            raise ModelingError("abi_verdict_available_flag_invalid")
        if self.backend_kind not in ABI_BACKEND_KINDS:
            raise ModelingError(f"abi_verdict_backend_invalid:{self.backend_kind!r}")
        if self.device_arch not in SOC_ARCHS:
            raise ModelingError(f"abi_verdict_arch_invalid:{self.device_arch!r}")
        if self.pr_track not in STACK_TRACKS:
            raise ModelingError(f"abi_verdict_track_invalid:{self.pr_track!r}")
        if not isinstance(self.engine_speedup_verified, bool):
            raise ModelingError("abi_verdict_speedup_flag_invalid")
        object.__setattr__(
            self, "query_digest", _validate_sha256(self.query_digest, "abi_verdict_query_digest_invalid"),
        )
        if self.matched_lowering_sha256:
            _validate_sha256(self.matched_lowering_sha256, "abi_verdict_lowering_sha256_invalid")
        object.__setattr__(
            self, "stack_or_cann_version",
            _req_str(self.stack_or_cann_version, "abi_verdict_version_required"),
        )
        if self.source_ref is not None and not isinstance(self.source_ref, SourceRef):
            raise ModelingError("abi_verdict_source_ref_invalid")
        expansion = tuple(_req_str(k, "abi_verdict_kernel_expansion_invalid") for k in self.device_kernel_expansion)
        object.__setattr__(self, "device_kernel_expansion", expansion)
        self._validate_status_invariants()

    def _validate_status_invariants(self) -> None:
        if self.status in {"admitted_shipping", "admitted_proposed_pr"}:
            if not self.available_on_device:
                raise ModelingError(f"abi_verdict_admitted_requires_available_on_device:{self.abi_id}")
            if self.backend_kind not in HARDWARE_USABLE_ABI_BACKENDS:
                raise ModelingError(f"abi_verdict_admitted_requires_hardware_backend:{self.backend_kind}")
            _validate_sha256(self.probe_manifest_sha256, f"abi_verdict_probe_manifest_sha256_invalid:{self.abi_id}")
            if not (self.matched_entry_id.strip() and self.symbol_or_kernel.strip() and self.device_kernel_expansion):
                raise ModelingError(f"abi_verdict_incomplete_admission_evidence:{self.abi_id}:{self.status}")
            if not (self.act_dtype.strip() and self.weight_dtype.strip() and self.quant_mode.strip()):
                raise ModelingError(f"abi_verdict_admitted_missing_dtype_evidence:{self.abi_id}")
            if self.source_ref is None:
                raise ModelingError(f"abi_verdict_admitted_missing_source_ref:{self.abi_id}")
            _validate_nonempty_source_ref(self.source_ref, f"abi_verdict_source_ref_incomplete:{self.abi_id}")
            if self.backend_kind in CANN_PACKAGE_ABI_BACKENDS and not self.cann_check_params_ref.strip():
                raise ModelingError(f"abi_verdict_cann_missing_check_params_ref:{self.abi_id}")
        if self.status == "admitted_shipping" and self.pr_track == "eager":
            raise ModelingError(f"abi_verdict_eager_cannot_admit_shipping:{self.abi_id}")
        if self.status == "admitted_proposed_pr":
            if self.pr_track not in {"proposed", "target"}:
                raise ModelingError(f"abi_verdict_proposed_disallowed_on_track:{self.pr_track}")
            if not self.proposed_pr_ref.strip():
                raise ModelingError(f"abi_verdict_proposed_missing_pr_ref:{self.abi_id}")
        if self.status == "rejected_abi" and not self.rejection_reason.strip():
            raise ModelingError(f"abi_verdict_rejected_missing_reason:{self.abi_id}")
        if self.status == "custom_synthesis" and self.pr_track != "target":
            raise ModelingError(f"abi_verdict_custom_synthesis_requires_target_track:{self.pr_track}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_id": self.abi_id,
            "status": self.status,
            "available_on_device": self.available_on_device,
            "backend_kind": self.backend_kind,
            "device_arch": self.device_arch,
            "stack_or_cann_version": self.stack_or_cann_version,
            "act_dtype": self.act_dtype,
            "weight_dtype": self.weight_dtype,
            "quant_mode": self.quant_mode,
            "probe_manifest_sha256": self.probe_manifest_sha256,
            "engine_speedup_verified": self.engine_speedup_verified,
            "query_digest": self.query_digest,
            "matched_entry_id": self.matched_entry_id,
            "symbol_or_kernel": self.symbol_or_kernel,
            "matched_lowering_id": self.matched_lowering_id,
            "matched_lowering_sha256": self.matched_lowering_sha256,
            "pr_track": self.pr_track,
            "proposed_pr_ref": self.proposed_pr_ref,
            "rejection_reason": self.rejection_reason,
            "source_ref": self.source_ref.to_dict() if self.source_ref is not None else None,
            "vllm_ascend_ref": self.vllm_ascend_ref,
            "op_plugin_ref": self.op_plugin_ref,
            "cann_check_params_ref": self.cann_check_params_ref,
            "device_kernel_expansion": list(self.device_kernel_expansion),
        }


@dataclass(frozen=True, slots=True)
class AbiCatalogSnapshot:
    """SHA-256 sealed per-SoC catalog of CANN and non-CANN hardware-usable ABIs."""

    device_arch: SocArch
    sku_id: str
    stack_or_cann_version: str
    entries: tuple[ProbedAbiEntry, ...]
    schema_version: str = ABI_CATALOG_SCHEMA
    payload_sha256: str = ""

    def __post_init__(self) -> None:
        if self.device_arch not in SOC_ARCHS:
            raise ModelingError(f"abi_catalog_arch_invalid:{self.device_arch!r}")
        object.__setattr__(self, "sku_id", _req_str(self.sku_id, "abi_catalog_sku_required").lower())
        object.__setattr__(
            self, "stack_or_cann_version",
            _req_str(self.stack_or_cann_version, "abi_catalog_version_required"),
        )
        if self.schema_version != ABI_CATALOG_SCHEMA:
            raise ModelingError(f"abi_catalog_schema_invalid:{self.schema_version!r}")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(e, ProbedAbiEntry) for e in entries):
            raise ModelingError("abi_catalog_entries_invalid")
        seen_ids: set[str] = set()
        for entry in entries:
            if entry.device_arch != self.device_arch:
                raise ModelingError(
                    f"abi_catalog_arch_mismatch:{entry.entry_id}:{entry.device_arch}!={self.device_arch}"
                )
            if entry.entry_id in seen_ids:
                raise ModelingError(f"abi_catalog_duplicate_entry_id:{entry.entry_id}")
            seen_ids.add(entry.entry_id)
        object.__setattr__(self, "entries", entries)
        computed = self._compute_payload_sha256()
        if not self.payload_sha256:
            object.__setattr__(self, "payload_sha256", computed)
        else:
            _validate_sha256(self.payload_sha256, "abi_catalog_sha256_invalid")
            if self.payload_sha256 != computed:
                raise ModelingError(
                    f"abi_catalog_sha256_mismatch:declared={self.payload_sha256}:computed={computed}"
                )

    def _compute_payload_sha256(self) -> str:
        canonical = {
            "schema_version": self.schema_version,
            "device_arch": self.device_arch,
            "sku_id": self.sku_id,
            "stack_or_cann_version": self.stack_or_cann_version,
            "entries": [e.to_dict() for e in sorted(self.entries, key=lambda item: item.entry_id)],
        }
        return _sha256_canonical(canonical)

    def entries_for(self, abi_id: str) -> tuple[ProbedAbiEntry, ...]:
        return tuple(e for e in self.entries if e.abi_id == abi_id)

    def entries_by_backend(self, backend_kind: AbiBackendKind) -> tuple[ProbedAbiEntry, ...]:
        return tuple(e for e in self.entries if e.backend_kind == backend_kind)

    def available_entries(self) -> tuple[ProbedAbiEntry, ...]:
        return tuple(e for e in self.entries if e.available_on_device)

    def verify_seal(self, expected_sha256: str) -> None:
        _validate_sha256(expected_sha256, "abi_catalog_expected_sha256_invalid")
        if self.payload_sha256 != expected_sha256:
            raise ModelingError(
                f"abi_catalog_seal_mismatch:actual={self.payload_sha256}:expected={expected_sha256}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device_arch": self.device_arch,
            "sku_id": self.sku_id,
            "stack_or_cann_version": self.stack_or_cann_version,
            "payload_sha256": self.payload_sha256,
            "entries": [e.to_dict() for e in self.entries],
        }


def validate_fusion_tier_verdict(fusion_tier: FusionTier, verdict: AbiSupportVerdict) -> None:
    """Enforce bidirectional Equation 7 between ``fusion_tier`` and ``AbiSupportVerdict``."""
    if fusion_tier not in FUSION_TIERS:
        raise ModelingError(f"fusion_tier_invalid:{fusion_tier!r}")
    if not isinstance(verdict, AbiSupportVerdict):
        raise ModelingError("fusion_tier_verdict_type_invalid")
    is_verified_shipping = (
        verdict.status == "admitted_shipping"
        and verdict.available_on_device
        and bool(verdict.probe_manifest_sha256)
        and verdict.engine_speedup_verified
    )
    is_verified_proposed = (
        verdict.status == "admitted_proposed_pr"
        and verdict.available_on_device
        and bool(verdict.probe_manifest_sha256)
        and verdict.engine_speedup_verified
    )
    if fusion_tier == "K1_shipping" and not is_verified_shipping:
        raise ModelingError(f"fuse_region_unverified_shipping_claim:{verdict.abi_id}")
    if is_verified_shipping and fusion_tier != "K1_shipping":
        raise ModelingError(f"fuse_region_shipping_tier_downgrade_forbidden:{verdict.abi_id}:{fusion_tier}")
    if fusion_tier == "K1_proposed_pr" and not is_verified_proposed:
        raise ModelingError(f"fuse_region_unverified_proposed_claim:{verdict.abi_id}")
    if is_verified_proposed and fusion_tier != "K1_proposed_pr":
        raise ModelingError(f"fuse_region_proposed_tier_downgrade_forbidden:{verdict.abi_id}:{fusion_tier}")
    if fusion_tier in {"K2_epilogue", "K3_dsl_pipeline"} and verdict.pr_track != "target":
        raise ModelingError(f"fuse_region_custom_tier_requires_target_track:{fusion_tier}:{verdict.pr_track}")


ShippingAbiVariant = ProbedAbiEntry
ShippingAbiVerdict = AbiSupportVerdict

__all__ = [
    "ABI_BACKEND_KINDS",
    "ABI_CATALOG_SCHEMA",
    "ABI_VERDICT_STATUSES",
    "CANN_PACKAGE_ABI_BACKENDS",
    "FUSION_TIERS",
    "HARDWARE_USABLE_ABI_BACKENDS",
    "MANIFEST_ENTRY_TRACKS",
    "MLA_VARIANT_KINDS",
    "MODEL_SCOPES",
    "NON_CANN_ABI_BACKENDS",
    "OPEN_SYNTHESIS_ABI_BACKENDS",
    "OPERAND_LAYOUTS",
    "RANK_PLACEMENTS",
    "REGION_BACKEND_TARGETS",
    "SOC_ARCHS",
    "AbiBackendKind",
    "AbiCatalogSnapshot",
    "AbiQueryKey",
    "AbiShapeConstraints",
    "AbiSupportVerdict",
    "AbiVerdictStatus",
    "FusionTier",
    "ManifestEntryTrack",
    "MlaVariantKind",
    "ModelScope",
    "OperandLayout",
    "OperandPortSig",
    "ProbedAbiEntry",
    "RankPlacement",
    "RegionBackendTarget",
    "ShippingAbiVariant",
    "ShippingAbiVerdict",
    "SocArch",
    "validate_fusion_tier_verdict",
]
