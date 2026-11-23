"""Universal Tier-P2 algebraic graph rewrite passes over typed operator traits.

[Role] Classify operators into typed algebraic traits (``POINTWISE``,
  ``ROW_REDUCTION``, ``BILINEAR_MATMUL``, ``ASSOCIATIVE_COLLECTIVE``,
  ``VIEW_METADATA``, ``STATEFUL_RECURRENT``, ``ATTENTION_CORE``), execute the
  four mathematically provable graph rewrite laws (UB pointwise/reduction
  chains, right-distributive sibling MatMul fusion, MatMul x Collective MC2
  pipelining, and pure view elision + typed-predicate shipping CANN ABI
  matching), emit verified ``RewritePlan`` edits, and spotlight non-pure
  stateful/ABI/precision boundaries as ``SemanticAuditTrigger`` records.
[Invariants] Strictly model-agnostic and suffix-table-free: zero model-family
  branches, zero op-name regexes, and zero unsafe rewrites across nonlinear
  norms or stateful mutations. Every emitted ``FuseRegion`` preserves DAG
  convexity, live-out ports, UB/L1 SRAM fit, and algebraic commutativity.
[Timeouts & Locks] Pure O(N^2) local DAG analysis over representative rank
  graphs; no locks, I/O, or device contexts.
[Side-Effect Policy] Pure functional analysis returning immutable records and
  ``RewritePlan`` values without mutating caller scenarios or graphs.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Literal

from mindie_sim.prediction.runtime import tile_solver
from mindie_sim.prediction.runtime.tile_solver import (
    DEFAULT_POOL_GRANULE_BYTES,
    LiveBufferSpec,
    TileCandidate,
    TilePipelineSpec,
    TileShape,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.operators import OpSpec

from .apply import (
    SIMT_UB_CARVEOUT_BYTES,
    ResolvedScenario,
    _activation_port_signature,
    _activation_quant_key,
    _convex,
    _dependencies,
    _law2_sibling_key,
    _successors,
    _topological,
)
from .schema import (
    Edit,
    FuseRegion,
    FusionTier,
    KernelBoundary,
    OpRef,
    PortRef,
    RewritePlan,
    empty_plan,
)

#: Default double-buffer pipeline depth when sizing UB working sets.
UB_DOUBLE_BUFFER_SLOTS = 2
#: Ascend AIV FP32 upcast working width (bytes/elem) inside UB (VCVT BF16->FP32).
FP32_ACCUM_BYTES = 4
#: Minimum members in a shipping MLA Prolog composite region.
MIN_MLA_PROLOG_MEMBERS = 4
#: Minimum members in a fused region.
MIN_FUSE_MEMBERS = 2
#: Minimum avoidable HBM bytes (4 KiB) to emit a semantic audit trigger.
MIN_TRIGGER_AVOIDABLE_BYTES = 4096
#: Standard GEMM tensor shape rank ``(M, N, K)``.
MATMUL_SHAPE_RANK = 3
#: Default UB tile row block size for row-wise reductions.
DEFAULT_UB_TILE_ROWS = 16
#: Default UB alignment granule (bytes).
UB_ALIGN_BYTES = DEFAULT_POOL_GRANULE_BYTES
#: Maximum chain length for Law 1 UB pointwise/reduction composition.
MAX_LAW1_CHAIN_LEN = 12


class AlgebraicTrait(str, Enum):
    """Typed algebraic classification driving provable Tier-P2 rewrites."""

    POINTWISE = "POINTWISE"
    ROW_REDUCTION = "ROW_REDUCTION"
    BILINEAR_MATMUL = "BILINEAR_MATMUL"
    ASSOCIATIVE_COLLECTIVE = "ASSOCIATIVE_COLLECTIVE"
    VIEW_METADATA = "VIEW_METADATA"
    STATEFUL_RECURRENT = "STATEFUL_RECURRENT"
    ATTENTION_CORE = "ATTENTION_CORE"
    CONTROL_BARRIER = "CONTROL_BARRIER"


_POINTWISE_FAMILIES = frozenset({
    "elementwise", "activation", "swiglu", "situ", "silu", "gelu", "sigmoid",
    "mul", "add", "adds", "cast", "pow", "pows", "rsqrt", "rotary",
    "gdn_gating", "residual", "bias_add",
})
_ROW_REDUCTION_FAMILIES = frozenset({
    "rmsnorm", "layernorm", "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant",
    "l2norm", "softmax", "quantize", "reducemean", "topk", "moe_init_routing",
})
_BILINEAR_MATMUL_FAMILIES = frozenset({
    "gemm", "projection", "qkv_projection", "gate_up_projection",
    "down_projection", "output_projection", "grouped_gemm", "grouped_matmul",
    "grouped_moe", "router", "logits", "index_projection",
})
_SUM_COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "all_reduce", "tp_allreduce", "reduce_scatter", "reducescatter",
    "combine",
})
_CONCAT_COLLECTIVE_FAMILIES = frozenset({
    "allgather", "all_gather",
})
_PERMUTE_COLLECTIVE_FAMILIES = frozenset({
    "alltoall", "alltoallv", "dispatch", "mc2", "mc2_protocol", "mc2_transport",
    "collective", "hccl", "p2p",
})
_VIEW_METADATA_FAMILIES = frozenset({
    "tensor_copy", "copy", "view", "reshape", "slice", "permute", "transpose",
    "cat", "concat", "contiguous", "tensor_fill", "pad", "split",
})
_STATEFUL_FAMILIES = frozenset({
    "short_convolution", "causal_conv1d_fn", "conv1d",
    "recurrent_scan", "recurrent_vector", "gdn_state",
    "cache_store", "kv_compress_epilog", "kv_cache_write",
    "indexer_quant_cache",
})
_ATTENTION_CORE_FAMILIES = frozenset({
    "attention", "attention_merge", "indexer", "key_pool", "mla",
})
_OPAQUE_CUSTOM_FAMILIES = frozenset({
    "attn_res_prefix", "mhc", "engram_apply", "engram_query",
    "engram_rh2d", "engram_gather", "h2d",
})
_CONTROL_BARRIER_FAMILIES = frozenset({
    "barrier", "embedding", "acceptance", "eager_split_barrier",
    "d2h_sync", "host_d2h_sync", "constant",
})
_NONLINEAR_POINTWISE_FAMILIES = frozenset({
    "activation", "swiglu", "situ", "silu", "gelu", "sigmoid", "pow", "pows", "rsqrt",
    "gdn_gating",
})
_NONLINEAR_RECIPE_TOKENS = (
    "swiglu", "situ", "silu", "gelu", "sigmoid", "norm", "softmax", "quant", "l2norm",
)
_VIEW_RECIPE_TOKENS = (
    "contiguous", "_flat", "concatenate", "_copy", "materialize", "zero", "_pack",
)
_TYPED_MLA_ROLE_MAP = {
    "mla.q_a": "q_a",
    "q_a": "q_a",
    "mla.kv_a": "kv_a",
    "kv_a": "kv_a",
    "mla.qkv_a": "qkv_a",
    "qkv_a": "qkv_a",
    "mla.q_b": "q_b",
    "q_b": "q_b",
    "mla_absorb": "q_absorb",
    "mla.q_absorb": "q_absorb",
    "q_absorb": "q_absorb",
    "key_absorption": "q_absorb",
    "mla.q_norm": "q_norm",
    "q_norm": "q_norm",
    "mla.kv_norm": "kv_norm",
    "kv_norm": "kv_norm",
    "mla.rotary": "rotary",
    "rotary": "rotary",
    "q_rotary": "rotary",
    "kv_rotary": "rotary",
    "mla.kv_store": "kv_store",
    "kv_store": "kv_store",
    "kv_scatter": "kv_store",
}


@dataclass(frozen=True, slots=True)
class ShippingAbiPredicate:
    """Typed capability predicate for a shipping Tier-K1 CANN/HCCL composite ABI."""

    abi_id: str
    fusion_tier: FusionTier
    pattern: str
    supported_activations: frozenset[str] = frozenset()
    supported_dtypes: frozenset[str] = frozenset()
    supported_weight_dtypes: frozenset[str] = frozenset()
    supported_arch_families: frozenset[str] = frozenset()
    supports_draft_scope: bool = True

    def supports_activation(self, activation: str) -> bool:
        if not self.supported_activations:
            return True
        if not activation:
            return False
        return activation.strip().lower() in self.supported_activations

    def supports_dtype(self, dtype: str) -> bool:
        if not self.supported_dtypes:
            return True
        if not dtype:
            return False
        return dtype.strip().lower() in self.supported_dtypes

    def supports_weight_dtype(self, weight_dtype: str) -> bool:
        if not self.supported_weight_dtypes:
            return True
        if not weight_dtype:
            return False
        return weight_dtype.strip().lower() in self.supported_weight_dtypes

    def supports_arch(self, arch_family: str) -> bool:
        if not self.supported_arch_families:
            return True
        if not arch_family:
            return False
        return arch_family.strip().upper() in self.supported_arch_families


_SHIPPING_STD_DTYPES = frozenset({
    "bf16", "bfloat16", "fp16", "float16", "int8", "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8",
})
_SHIPPING_STD_ACTIVATIONS = frozenset({"swiglu", "silu", "gelu"})
_ARCH35_ARCH_ID = "npu_arch_3510"
_ARCH22_ARCH_ID = "npu_arch_2201"
_FUSED_MC2_MEGAMOE_MODE = 2
_INT4_WEIGHT_DTYPES = frozenset({"int4", "w4a8"})
_INT8_WEIGHT_DTYPES = frozenset({"int8", "w8a8"})
_MEGAMOE_WEIGHT_DTYPES = frozenset({
    "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "mxfp4", "mx_fp4", "fp4", "fp4_e2m1", "int8",
})

#: Typed capability predicates for shipping Tier-K1 composite CANN/HCCL ABIs.
SHIPPING_ABI_PREDICATES: Mapping[str, ShippingAbiPredicate] = {
    "aclnnMlaPrologV3": ShippingAbiPredicate(
        abi_id="aclnnMlaPrologV3",
        fusion_tier="K1_shipping",
        pattern="mla_prolog",
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supports_draft_scope=True,
    ),
    "aclnnSwiGluQuant": ShippingAbiPredicate(
        abi_id="aclnnSwiGluQuant",
        fusion_tier="K1_shipping",
        pattern="act_quant",
        supported_activations=_SHIPPING_STD_ACTIVATIONS,
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supports_draft_scope=True,
    ),
    "aclnnAddRmsNormDynamicQuant": ShippingAbiPredicate(
        abi_id="aclnnAddRmsNormDynamicQuant",
        fusion_tier="K1_shipping",
        pattern="add_rmsnorm_quant",
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supports_draft_scope=True,
    ),
    "aclnnRmsNormDynamicQuant": ShippingAbiPredicate(
        abi_id="aclnnRmsNormDynamicQuant",
        fusion_tier="K1_shipping",
        pattern="rmsnorm_quant",
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supports_draft_scope=True,
    ),
    "aclnnAllGatherMatmulV2": ShippingAbiPredicate(
        abi_id="aclnnAllGatherMatmulV2",
        fusion_tier="K1_shipping",
        pattern="allgather_matmul",
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supports_draft_scope=True,
    ),
    "aclnnMatmulReduceScatterV2": ShippingAbiPredicate(
        abi_id="aclnnMatmulReduceScatterV2",
        fusion_tier="K1_shipping",
        pattern="matmul_reducescatter",
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supports_draft_scope=True,
    ),
    "aclnnMegaMoe": ShippingAbiPredicate(
        abi_id="aclnnMegaMoe",
        fusion_tier="K1_shipping",
        pattern="moe_dispatch_ffn_combine",
        supported_activations=_SHIPPING_STD_ACTIVATIONS,
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supported_weight_dtypes=_MEGAMOE_WEIGHT_DTYPES,
        supported_arch_families=frozenset({"A5", "A3"}),
        supports_draft_scope=False,
    ),
    "dispatch_ffn_combine": ShippingAbiPredicate(
        abi_id="dispatch_ffn_combine",
        fusion_tier="K1_shipping",
        pattern="moe_dispatch_ffn_combine",
        supported_activations=_SHIPPING_STD_ACTIVATIONS,
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supported_weight_dtypes=_INT8_WEIGHT_DTYPES,
        supported_arch_families=frozenset({"A3"}),
        supports_draft_scope=False,
    ),
    "dispatch_ffn_combine_w4_a8": ShippingAbiPredicate(
        abi_id="dispatch_ffn_combine_w4_a8",
        fusion_tier="K1_shipping",
        pattern="moe_dispatch_ffn_combine",
        supported_activations=_SHIPPING_STD_ACTIVATIONS,
        supported_dtypes=_SHIPPING_STD_DTYPES,
        supported_weight_dtypes=_INT4_WEIGHT_DTYPES,
        supported_arch_families=frozenset({"A3"}),
        supports_draft_scope=False,
    ),
}


def _resolve_arch_family(
    datapath: DatapathCapabilitySpec | None, arch_family: str = "",
) -> str:
    explicit = str(arch_family or "").strip().upper()
    if explicit in {"A2", "A3", "A5"}:
        return explicit
    arch_id = str(getattr(getattr(datapath, "compute_arch", None), "arch_id", "") or "")
    if arch_id == _ARCH35_ARCH_ID:
        return "A5"
    if arch_id == _ARCH22_ARCH_ID:
        return "A3"
    return ""


def resolve_shipping_moe_predicate(
    arch_family: str,
    *,
    weight_dtype: str = "",
    enable_fused_mc2: int | None = None,
) -> ShippingAbiPredicate | None:
    """Resolve the architecture-qualified shipping fused-MoE predicate (zero cross-gen leak)."""
    arch = str(arch_family or "").strip().upper()
    w_fmt = str(weight_dtype or "").strip().lower()
    if arch == "A5":
        return SHIPPING_ABI_PREDICATES["aclnnMegaMoe"]
    if arch == "A3":
        if enable_fused_mc2 == _FUSED_MC2_MEGAMOE_MODE:
            return SHIPPING_ABI_PREDICATES["aclnnMegaMoe"]
        if w_fmt in _INT4_WEIGHT_DTYPES:
            return SHIPPING_ABI_PREDICATES["dispatch_ffn_combine_w4_a8"]
        return SHIPPING_ABI_PREDICATES["dispatch_ffn_combine"]
    if arch == "A2":
        return None
    if w_fmt in _MEGAMOE_WEIGHT_DTYPES - _INT8_WEIGHT_DTYPES:
        return SHIPPING_ABI_PREDICATES["aclnnMegaMoe"]
    return None


@dataclass(frozen=True, slots=True)
class OpTraitInfo:
    """Typed algebraic descriptor extracted solely from ``OpSpec`` structure."""

    op_id: str
    trait: AlgebraicTrait
    collective_op: Literal["", "SUM", "CONCAT", "PERMUTE"] = ""
    collective_axis: str = ""
    is_nonlinear: bool = False
    reduction_axis: int | None = None
    activation_kind: str = ""
    activation_params: tuple[tuple[str, float], ...] = ()
    input_dtype: str = "bf16"
    output_dtype: str = "bf16"
    weight_dtype: str = ""
    group_size: int | None = None
    scale_format: str = ""
    is_stateful: bool = False
    is_precision_sensitive: bool = False
    is_opaque_custom: bool = False
    row_working_bytes: int = 0
    symbolic_op_kind: str = "elementwise"
    is_grouped_matmul: bool = False


@dataclass(frozen=True, slots=True)
class AlgebraicCandidate:
    """One mathematically justified fusion candidate emitted by Laws 1-4."""

    candidate_id: str
    law: Literal["law1_ub_chain", "law2_block_matmul", "law3_mc2", "law4_view_or_abi"]
    abi_id: str
    fusion_tier: FusionTier
    member_ids: tuple[str, ...]
    preserve_port_ids: tuple[str, ...] = ()
    shipping_supported: bool = True
    unsupported_reason: str = ""
    elided_view_ids: tuple[str, ...] = ()
    ub_peak_bytes: int = 0
    ub_budget_bytes: int = 0
    ub_precision_mode: str = "fp32_upcast"
    avoidable_hbm_bytes: int = 0
    traits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticAuditTrigger:
    """Structured spotlight where pure algebra paused at a semantic boundary."""

    trigger_id: str
    kind: Literal[
        "STATE_MUTATION",
        "SPECULATIVE_STATE_COMMIT",
        "ABI_UNSUPPORTED_VARIANT",
        "PRECISION_SENSITIVE_REASSOCIATION",
        "OPAQUE_CUSTOM_OP",
    ]
    blocked_ops: tuple[str, ...]
    avoidable_hbm_bytes: int
    potential_gain_us: float
    potential_gain_us_per_tok_die: float
    blocking_reason: str
    source_refs: tuple[str, ...]
    unlock_opcode: str = "FuseRegion"
    candidate_abi: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "kind": self.kind,
            "blocked_ops": list(self.blocked_ops),
            "avoidable_hbm_bytes": int(self.avoidable_hbm_bytes),
            "potential_gain_us": round(float(self.potential_gain_us), 3),
            "potential_gain_us_per_tok_die": round(float(self.potential_gain_us_per_tok_die), 3),
            "blocking_reason": self.blocking_reason,
            "source_refs": list(self.source_refs),
            "unlock_opcode": self.unlock_opcode,
            "candidate_abi": self.candidate_abi,
        }


@dataclass(frozen=True, slots=True)
class AlgebraicRewriteResult:
    """Output of Tier-P2 algebraic discovery over a rank DAG or ``ResolvedScenario``."""

    plan: RewritePlan
    candidates: tuple[AlgebraicCandidate, ...]
    semantic_audit_triggers: tuple[SemanticAuditTrigger, ...]
    rejected_pairs: tuple[tuple[str, str], ...] = ()
    verification: Any = None


def _safe_dtype_bytes(dtype: str) -> float:
    tok = str(dtype or "bf16").strip().lower()
    if not tok:
        return 2.0
    try:
        return float(dtype_bytes(tok))
    except ModelingError:
        return 2.0


def _extract_activation_params(meta: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    params: dict[str, float] = {}
    raw = meta.get("activation_params")
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
                params[k] = float(v)
    for key in ("beta", "linear_beta", "alpha", "limit"):
        val = meta.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and key not in params:
            params[key] = float(val)
    return tuple(sorted(params.items()))


def _infer_activation_kind(fam: str, rec: str, meta: Mapping[str, Any]) -> str:
    explicit = str(meta.get("activation") or meta.get("ffn_activation") or "").strip().lower()
    if explicit:
        return explicit
    for cand in ("situ", "swiglu", "silu", "gelu", "sigmoid"):
        if fam == cand or cand in rec:
            return cand
    return ""


def _estimate_row_working_bytes(
    shape: Sequence[int],
    in_dt: str,
    out_dt: str,
    *,
    is_reduction: bool,
    is_vector_compute: bool = True,
) -> int:
    """Budget UB row working bytes using Ascend NPU FP32 (4B) AIV upcast."""
    if not shape:
        return 0
    row_elems = int(shape[-1]) if len(shape) >= 2 else int(shape[0])
    if row_elems <= 0:
        return 0
    in_b = _safe_dtype_bytes(in_dt)
    out_b = _safe_dtype_bytes(out_dt)
    if is_vector_compute:
        # Ascend AIV upcasts BF16/FP16 to FP32 (4B/elem) inside UB for compute.
        fp32_work = float(FP32_ACCUM_BYTES)
        accum_b = float(FP32_ACCUM_BYTES) if is_reduction else 0.0
        raw = int(row_elems * (max(in_b, fp32_work) + max(out_b, fp32_work) + accum_b))
    else:
        raw = int(row_elems * (in_b + out_b))
    return ((raw + UB_ALIGN_BYTES - 1) // UB_ALIGN_BYTES) * UB_ALIGN_BYTES


def _resolve_symbolic_kind(
    fam: str, rec: str, meta: Mapping[str, Any], trait: AlgebraicTrait, act_kind: str,
) -> str:
    if fam in {"grouped_gemm", "grouped_matmul", "grouped_moe"}:
        return "grouped_matmul"
    if trait == AlgebraicTrait.BILINEAR_MATMUL:
        return "matmul"
    if fam in {"all_gather_matmul", "matmul_reduce_scatter"}:
        return fam
    if fam == "add_rmsnorm_quant":
        return "add_rmsnorm_quant"
    if fam == "add_rmsnorm":
        return "add_rmsnorm"
    if fam == "rmsnorm_quant":
        return "rmsnorm_quant"
    if fam == "rmsnorm":
        return "rmsnorm"
    if fam == "layernorm":
        return "layernorm"
    if fam == "quantize":
        if str(meta.get("activation_scale_format") or "") == "fp8_e8m0":
            return "dynamic_mx_quant"
        return "dynamic_quant"
    if "l2norm" in rec or fam == "l2norm":
        return "l2norm"
    if "norm_gate" in rec:
        return "norm_gate"
    if act_kind:
        return act_kind
    if fam in _SUM_COLLECTIVE_FAMILIES:
        return "reduce_scatter" if "scatter" in fam or "scatter" in rec else (
            "combine" if fam == "combine" or "combine" in rec else "allreduce"
        )
    if fam in _CONCAT_COLLECTIVE_FAMILIES:
        return "allgather"
    if fam in _PERMUTE_COLLECTIVE_FAMILIES:
        if "combine" in rec:
            return "combine"
        return "dispatch" if fam == "dispatch" or "dispatch" in rec else "alltoall"
    if trait == AlgebraicTrait.VIEW_METADATA:
        return "view_copy"
    if fam in {"short_convolution", "causal_conv1d_fn", "conv1d"} or "conv" in rec:
        return "conv1d"
    if fam == "gdn_gating" or "gating" in rec:
        return "gating"
    if "recover_ssm" in rec:
        return "recover_ssm"
    if fam in {"recurrent_scan", "recurrent_vector"}:
        return "recurrent_scan"
    if fam == "attention_merge":
        return "attention_update"
    if fam == "attn_res_prefix":
        return "attn_res"
    if fam == "mhc":
        return "mhc_mix"
    if fam == "rotary" or "rotary" in rec or "rope" in rec or "partial_rotary_work" in meta:
        return "rotary"
    if fam in {"cache_store", "kv_compress_epilog", "kv_cache_write", "indexer_quant_cache"}:
        return "kv_scatter"
    if fam == "indexer":
        return "indexer"
    if fam == "topk":
        return "topk"
    if fam == "cast":
        return "cast"
    if fam == "add" or "add" in rec:
        return "add"
    if fam == "mul" or "mul" in rec or "gate" in rec:
        return "mul"
    return fam or "elementwise"


def classify_op_traits(op: OpSpec) -> OpTraitInfo:
    """Classify an ``OpSpec`` purely by typed operator family, work, and metadata."""
    fam = str(op.family or "").strip().lower()
    rec = str(op.recipe or "").strip().lower()
    meta = dict(op.metadata) if op.metadata else {}

    in_dt = str(meta.get("activation_format") or meta.get("input_format") or op.dtype or "bf16").strip().lower()
    out_dt = str(meta.get("output_format") or op.dtype or "bf16").strip().lower()
    w_dt = str(meta.get("weight_format") or meta.get("weight_dtype") or op.dtype or "").strip().lower()
    raw_gs = meta.get("activation_group_size", meta.get("quant_group_size"))
    group_size = int(raw_gs) if isinstance(raw_gs, int) and not isinstance(raw_gs, bool) and raw_gs > 0 else None
    scale_fmt = str(meta.get("activation_scale_format") or meta.get("quant_scale_dtype") or "").strip().lower()
    act_kind = _infer_activation_kind(fam, rec, meta)
    act_params = _extract_activation_params(meta)
    col_axis = str(meta.get("group_axis") or "").strip().lower()

    is_stateful = bool(meta.get("state_effect")) or fam in _STATEFUL_FAMILIES or any(
        tok in rec for tok in ("recover_ssm", "short_convolution", "causal_conv", "recurrent_")
    )
    is_opaque = fam in _OPAQUE_CUSTOM_FAMILIES
    is_prec_sens = fam == "attention_merge" or "lse_merge" in rec

    trait, col_op, is_nonlin, red_axis = _resolve_primary_trait(
        op, fam, rec, act_kind, is_stateful=is_stateful, is_opaque=is_opaque,
    )
    row_bytes = _estimate_row_working_bytes(
        op.local_shape,
        in_dt,
        out_dt,
        is_reduction=(trait == AlgebraicTrait.ROW_REDUCTION),
        is_vector_compute=(trait in {AlgebraicTrait.POINTWISE, AlgebraicTrait.ROW_REDUCTION}),
    )
    sym_kind = _resolve_symbolic_kind(fam, rec, meta, trait, act_kind)
    is_grouped = fam in {"grouped_gemm", "grouped_matmul", "grouped_moe"} or "grouped" in rec
    return OpTraitInfo(
        op_id=op.op_id,
        trait=trait,
        collective_op=col_op,
        collective_axis=col_axis,
        is_nonlinear=is_nonlin,
        reduction_axis=red_axis,
        activation_kind=act_kind,
        activation_params=act_params,
        input_dtype=in_dt,
        output_dtype=out_dt,
        weight_dtype=w_dt,
        group_size=group_size,
        scale_format=scale_fmt,
        is_stateful=is_stateful,
        is_precision_sensitive=is_prec_sens,
        is_opaque_custom=is_opaque,
        row_working_bytes=row_bytes,
        symbolic_op_kind=sym_kind,
        is_grouped_matmul=is_grouped,
    )


def _resolve_primary_trait(
    op: OpSpec,
    fam: str,
    rec: str,
    act_kind: str,
    *,
    is_stateful: bool,
    is_opaque: bool,
) -> tuple[AlgebraicTrait, Literal["", "SUM", "CONCAT", "PERMUTE"], bool, int | None]:
    if fam in _CONTROL_BARRIER_FAMILIES:
        return AlgebraicTrait.CONTROL_BARRIER, "", False, None
    col_meta = str(dict(op.metadata).get("collective_op") or "").strip().upper() if op.metadata else ""
    if col_meta == "SUM":
        return AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, "SUM", False, None
    if col_meta == "CONCAT":
        return AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, "CONCAT", False, None
    if col_meta == "PERMUTE":
        return AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, "PERMUTE", False, None
    if fam in _SUM_COLLECTIVE_FAMILIES:
        return AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, "SUM", False, None
    if fam in _CONCAT_COLLECTIVE_FAMILIES:
        return AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, "CONCAT", False, None
    if fam in _PERMUTE_COLLECTIVE_FAMILIES:
        return AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, "PERMUTE", False, None
    if fam in {"all_gather_matmul", "matmul_reduce_scatter"}:
        col_mc2: Literal["", "SUM", "CONCAT", "PERMUTE"] = (
            "CONCAT" if fam == "all_gather_matmul" else "SUM"
        )
        return AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, col_mc2, False, None
    if fam in _VIEW_METADATA_FAMILIES or (
        fam == "elementwise" and op.flops == 0 and op.vector_ops == 0 and op.link_bytes == 0
        and any(tok in rec for tok in _VIEW_RECIPE_TOKENS)
    ):
        return AlgebraicTrait.VIEW_METADATA, "", False, None
    if is_stateful:
        return AlgebraicTrait.STATEFUL_RECURRENT, "", False, None
    if fam in _ATTENTION_CORE_FAMILIES:
        return AlgebraicTrait.ATTENTION_CORE, "", True, -1
    if fam in _ROW_REDUCTION_FAMILIES or "l2norm" in rec or "norm_gate" in rec:
        return AlgebraicTrait.ROW_REDUCTION, "", True, -1
    if fam in _BILINEAR_MATMUL_FAMILIES or (op.flops > 0 and not is_opaque):
        return AlgebraicTrait.BILINEAR_MATMUL, "", False, None
    if is_opaque:
        return AlgebraicTrait.ROW_REDUCTION, "", True, -1
    if fam in _POINTWISE_FAMILIES or op.vector_ops > 0:
        is_nl = bool(
            act_kind
            or fam in _NONLINEAR_POINTWISE_FAMILIES
            or any(tok in rec for tok in _NONLINEAR_RECIPE_TOKENS)
        )
        return AlgebraicTrait.POINTWISE, "", is_nl, None
    return AlgebraicTrait.CONTROL_BARRIER, "", False, None


def _usable_ub_bytes(datapath: DatapathCapabilitySpec | None) -> int:
    if datapath is None:
        return 0
    raw = int(dict(datapath.buffers).get("ub", 0) or 0)
    carveout = max(SIMT_UB_CARVEOUT_BYTES, int(datapath.simt_ub_carveout_bytes or 0))
    return max(0, raw - carveout)


def _check_ub_chain_fits(
    members: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    datapath: DatapathCapabilitySpec | None,
) -> tuple[bool, int, int]:
    """Verify 1-row UB working-set fit via ``TileSolver`` / SRAM budget."""
    budget = _usable_ub_bytes(datapath)
    if budget <= 0 or datapath is None:
        return False, 0, 0
    max_row = max((traits[op.op_id].row_working_bytes for op in members), default=0)
    if max_row <= 0:
        return True, 0, budget
    peak_single = max_row
    from types import SimpleNamespace

    hw = SimpleNamespace(
        datapath=datapath,
        buffers=dict(datapath.buffers),
        buffer_units={"ub": datapath.aiv_count, "l2": 1},
        stream_owners=None,
    )
    cands = tuple(
        TileCandidate(
            algorithm="resident_1pass" if slots == UB_DOUBLE_BUFFER_SLOTS else "online_reduce",
            tile_shape=TileShape(1, max(1, max_row // 2), 1),
            depth=slots,
            owner_partition=(1, 1),
            work_items=1,
            tiles_per_owner=1,
            buffers=(LiveBufferSpec("ub_chain", "ub", max_row, 0, 1, owner_kind="aiv", slots=slots),),
            uses_simt=True,
            owner_axis="aiv",
        )
        for slots in (UB_DOUBLE_BUFFER_SLOTS, 1)
    )
    try:
        solved = tile_solver.solve(
            TilePipelineSpec(
                "algebra_ub_chain",
                cands,
                simt_ub_carveout_bytes=max(SIMT_UB_CARVEOUT_BYTES, int(datapath.simt_ub_carveout_bytes or 0)),
            ),
            members[-1],
            hw,
        )
        ub_peak = max((p.peak_bytes for p in solved.pool_peaks if p.pool == "ub"), default=peak_single)
        return ub_peak <= budget, ub_peak, budget
    except tile_solver.TileInfeasibleError:
        return False, peak_single, budget


def _output_shape_bytes(op: OpSpec, tr: OpTraitInfo) -> float:
    """Return logical output tensor bytes for ``op`` (``(M, N)`` for 3D MatMul)."""
    if not op.local_shape:
        return 0.0
    if tr.trait == AlgebraicTrait.BILINEAR_MATMUL and len(op.local_shape) == MATMUL_SHAPE_RANK:
        elems = int(op.local_shape[0]) * int(op.local_shape[1])
    else:
        elems = prod(int(d) for d in op.local_shape if int(d) > 0)
    return float(max(0, elems)) * _safe_dtype_bytes(tr.output_dtype)


def _avoidable_hbm_bytes(members: Sequence[OpSpec], traits: Mapping[str, OpTraitInfo]) -> int:
    if len(members) < MIN_FUSE_MEMBERS:
        return 0
    saved = 0.0
    for op in members[:-1]:
        tr = traits[op.op_id]
        if tr.trait == AlgebraicTrait.VIEW_METADATA:
            saved += float(op.hbm_bytes or 0.0)
            continue
        shape_bytes = _output_shape_bytes(op, tr)
        op_hbm = float(op.hbm_bytes or 0.0)
        op_mte3 = float(op.mte3_bytes or 0.0)
        out_bytes = min(op_mte3, shape_bytes) if (op_mte3 > 0.0 and shape_bytes > 0.0) else (
            shape_bytes if shape_bytes > 0.0 else op_mte3
        )
        saved += min(op_hbm, out_bytes * 2.0) if out_bytes > 0.0 else op_hbm * 0.5
    return int(round(max(0.0, saved)))


def _law2_avoidable_read_bytes(siblings: Sequence[OpSpec], traits: Mapping[str, OpTraitInfo]) -> int:
    """Compute exact shared input activation bytes ``(S - 1) * M * K * elem_bytes`` saved by Law 2."""
    if len(siblings) < MIN_FUSE_MEMBERS:
        return 0
    first = siblings[0]
    if len(first.local_shape) != MATMUL_SHAPE_RANK:
        return 0
    m_dim, k_dim = int(first.local_shape[0]), int(first.local_shape[2])
    if m_dim <= 0 or k_dim <= 0:
        return 0
    a_dt, _, _ = _activation_quant_key(first)
    act_bytes = float(m_dim * k_dim) * _safe_dtype_bytes(a_dt)
    saved = 0.0
    for sib in siblings[1:]:
        mte2 = float(sib.mte2_bytes or 0.0)
        mte3 = float(sib.mte3_bytes or 0.0)
        hbm = float(sib.hbm_bytes or 0.0)
        if mte2 > 0.0:
            avail = min(mte2, max(0.0, hbm - mte3)) if hbm > 0.0 else mte2
        elif hbm > 0.0:
            avail = max(0.0, hbm - mte3)
        else:
            avail = act_bytes
        saved += min(act_bytes, max(0.0, avail))
    return int(round(max(0.0, saved)))


def _same_domain(a: OpSpec, b: OpSpec) -> bool:
    return (a.stage, a.phase, a.stream or "main", a.layer) == (b.stage, b.phase, b.stream or "main", b.layer)


def _producer_port_key(
    op: OpSpec, op_ids: set[str], by_id: Mapping[str, OpSpec] | None = None,
) -> tuple[tuple[str, str, int], ...] | None:
    sig = _activation_port_signature(op, all_ops=by_id)
    if not sig:
        return None
    if all(prod_id in op_ids for prod_id, _, _ in sig):
        return sig
    return None


def _is_ancestor(src: str, dst: str, succ: Mapping[str, Sequence[str]]) -> bool:
    stack = list(succ.get(src, ()))
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        if cur not in seen:
            seen.add(cur)
            stack.extend(succ.get(cur, ()))
    return False


def _typed_mla_role(op: OpSpec) -> str:
    """Resolve typed MLA Prolog structural role purely from metadata (zero op_id checks)."""
    meta = dict(op.metadata) if op.metadata else {}
    for key in ("mla_prolog_role", "abi_role", "structural_role"):
        raw = str(meta.get(key) or "").strip().lower()
        if raw in _TYPED_MLA_ROLE_MAP:
            return _TYPED_MLA_ROLE_MAP[raw]

    slices = meta.get("packed_projection_slices", ())
    if isinstance(slices, Sequence) and not isinstance(slices, (str, bytes)):
        slice_cats = {
            str(item.get("category") or "").strip().lower()
            for item in slices if isinstance(item, Mapping)
        }
        slice_roles = {
            str(item.get("role") or "").strip().lower()
            for item in slices if isinstance(item, Mapping)
        }
        has_qa = "mla.q_a" in slice_cats or "q_a" in slice_roles
        has_kva = "mla.kv_a" in slice_cats or "kv_a" in slice_roles
        if has_qa and has_kva:
            return "qkv_a"
        if has_qa:
            return "q_a"
        if has_kva:
            return "kv_a"

    cat = str(meta.get("quant_category") or meta.get("catalog_category") or "").strip().lower()
    if cat in _TYPED_MLA_ROLE_MAP:
        return _TYPED_MLA_ROLE_MAP[cat]
    if str(meta.get("role") or "").strip().lower() == "key_absorption":
        return "q_absorb"

    fused_m = meta.get("fused_members", ())
    if isinstance(fused_m, Sequence) and not isinstance(fused_m, (str, bytes)):
        if any(str(item).strip().lower() == "qkv_a" for item in fused_m):
            return "qkv_a"
    return ""


def _match_law4_mla_prolog(
    ordered: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
) -> list[AlgebraicCandidate]:
    """Match shipping ``aclnnMlaPrologV3`` via typed ABI roles and DAG topology."""
    by_scope_layer: dict[tuple[str, int, int, str], list[OpSpec]] = defaultdict(list)
    for op in ordered:
        if op.op_id in claimed:
            continue
        meta = dict(op.metadata) if op.metadata else {}
        is_draft = (
            str(meta.get("model_scope") or "").strip().lower() == "draft"
            or str(meta.get("structure_model_scope") or "").strip().lower() == "draft"
        )
        scope = "draft" if is_draft else "target"
        by_scope_layer[(scope, op.stage, op.layer, op.stream or "main")].append(op)

    pred = SHIPPING_ABI_PREDICATES["aclnnMlaPrologV3"]
    found: list[AlgebraicCandidate] = []
    for (scope, stage, layer, _stream), ops in by_scope_layer.items():
        roles = {op.op_id: _typed_mla_role(op) for op in ops}
        q_entries = {oid for oid, role in roles.items() if role in {"q_a", "qkv_a"}}
        kv_entries = {oid for oid, role in roles.items() if role in {"kv_a", "qkv_a"}}
        q_bs = {oid for oid, role in roles.items() if role == "q_b"}
        if not (q_entries and kv_entries and q_bs):
            continue
        q_norms = {
            op.op_id for op in ops
            if roles[op.op_id] == "q_norm" or (
                traits[op.op_id].symbolic_op_kind in {"rmsnorm", "layernorm"}
                and bool(deps.get(op.op_id, set()) & q_entries)
                and bool(set(succ.get(op.op_id, ())) & q_bs)
            )
        }
        kv_norms = {
            op.op_id for op in ops
            if op.op_id not in q_norms and (
                roles[op.op_id] == "kv_norm" or (
                    traits[op.op_id].symbolic_op_kind in {"rmsnorm", "layernorm"}
                    and bool(deps.get(op.op_id, set()) & kv_entries)
                )
            )
        }
        rot_sources = q_bs | q_norms | kv_norms | kv_entries
        rotaries = {
            op.op_id for op in ops
            if roles[op.op_id] == "rotary" or (
                traits[op.op_id].symbolic_op_kind == "rotary"
                and bool(deps.get(op.op_id, set()) & rot_sources)
            )
        }
        absorb_sources = q_bs | rotaries | q_norms | q_entries
        q_absorbs = {
            op.op_id for op in ops
            if roles[op.op_id] == "q_absorb"
            and (not deps.get(op.op_id) or bool(deps.get(op.op_id, set()) & absorb_sources))
        }
        store_sources = kv_norms | rotaries | kv_entries
        kv_stores = {
            op.op_id for op in ops
            if roles[op.op_id] == "kv_store" or (
                traits[op.op_id].symbolic_op_kind == "kv_scatter"
                and bool(deps.get(op.op_id, set()) & store_sources)
            )
        }
        if not (kv_norms or rotaries):
            continue
        member_set = q_entries | kv_entries | q_norms | q_bs | kv_norms | rotaries | q_absorbs | kv_stores
        members = [op for op in ops if op.op_id in member_set]
        if len(members) < MIN_MLA_PROLOG_MEMBERS:
            continue
        if not all(pred.supports_dtype(traits[op.op_id].input_dtype) for op in members):
            continue
        member_ids = tuple(op.op_id for op in members)
        if not _convex(ordered, member_set):
            continue
        preserve = tuple(
            op.op_id for op in members[:-1]
            if any(op.op_id in deps[other.op_id] for other in ordered if other.op_id not in member_set)
        )
        claimed.update(member_ids)
        found.append(AlgebraicCandidate(
            candidate_id=f"law4_mlapo_{scope}_s{stage}_l{layer}",
            law="law4_view_or_abi",
            abi_id="aclnnMlaPrologV3",
            fusion_tier="K1_shipping",
            member_ids=member_ids,
            preserve_port_ids=preserve,
            shipping_supported=True,
            avoidable_hbm_bytes=_avoidable_hbm_bytes(members, traits),
            traits=tuple(sorted({traits[oid].trait.value for oid in member_ids})),
        ))
    return found


def _discover_law3_mc2(
    ordered: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
    *,
    allow_mc2: bool,
) -> list[AlgebraicCandidate]:
    """Law 3: Bilinear MatMul x Associative Collective Pipelining (MC2)."""
    if not allow_mc2:
        return []
    by_id = {op.op_id: op for op in ordered}
    found: list[AlgebraicCandidate] = []
    ag_pred = SHIPPING_ABI_PREDICATES["aclnnAllGatherMatmulV2"]
    rs_pred = SHIPPING_ABI_PREDICATES["aclnnMatmulReduceScatterV2"]

    for op in ordered:
        if op.op_id in claimed:
            continue
        tr = traits[op.op_id]
        # Pattern 3a: AllGather(x, TP) -> [optional Quantize] -> MatMul(W_col)
        if tr.trait == AlgebraicTrait.ASSOCIATIVE_COLLECTIVE and tr.collective_op == "CONCAT":
            if tr.collective_axis and tr.collective_axis != "tp":
                continue
            users = [u for u in succ.get(op.op_id, ()) if u in by_id and u not in claimed]
            if len(users) != 1:
                continue
            nxt = by_id[users[0]]
            nxt_tr = traits[nxt.op_id]
            if (
                nxt_tr.trait == AlgebraicTrait.BILINEAR_MATMUL
                and not nxt_tr.is_grouped_matmul
                and _same_domain(op, nxt)
                and ag_pred.supports_dtype(nxt_tr.input_dtype)
                and _convex(ordered, {op.op_id, nxt.op_id})
            ):
                claimed.update((op.op_id, nxt.op_id))
                found.append(AlgebraicCandidate(
                    candidate_id=f"law3_agmm_{nxt.op_id}",
                    law="law3_mc2",
                    abi_id="aclnnAllGatherMatmulV2",
                    fusion_tier="K1_shipping",
                    member_ids=(op.op_id, nxt.op_id),
                    shipping_supported=True,
                    avoidable_hbm_bytes=_avoidable_hbm_bytes((op, nxt), traits),
                    traits=("ASSOCIATIVE_COLLECTIVE", "BILINEAR_MATMUL"),
                ))
        # Pattern 3b: MatMul(W_row) -> ReduceScatter(SUM, TP)
        elif tr.trait == AlgebraicTrait.BILINEAR_MATMUL and not tr.is_grouped_matmul:
            users = [u for u in succ.get(op.op_id, ()) if u in by_id and u not in claimed]
            if len(users) != 1:
                continue
            nxt = by_id[users[0]]
            nxt_tr = traits[nxt.op_id]
            if (
                nxt_tr.trait == AlgebraicTrait.ASSOCIATIVE_COLLECTIVE
                and nxt_tr.collective_op == "SUM"
                and (not nxt_tr.collective_axis or nxt_tr.collective_axis == "tp")
                and nxt_tr.symbolic_op_kind == "reduce_scatter"
                and _same_domain(op, nxt)
                and rs_pred.supports_dtype(tr.input_dtype)
                and len(deps.get(nxt.op_id, set())) == 1
                and _convex(ordered, {op.op_id, nxt.op_id})
            ):
                claimed.update((op.op_id, nxt.op_id))
                found.append(AlgebraicCandidate(
                    candidate_id=f"law3_mmrs_{op.op_id}",
                    law="law3_mc2",
                    abi_id="aclnnMatmulReduceScatterV2",
                    fusion_tier="K1_shipping",
                    member_ids=(op.op_id, nxt.op_id),
                    shipping_supported=True,
                    avoidable_hbm_bytes=_avoidable_hbm_bytes((op, nxt), traits),
                    traits=("BILINEAR_MATMUL", "ASSOCIATIVE_COLLECTIVE"),
                ))
    return found


def _discover_law2_block_matmul(
    ordered: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    succ: Mapping[str, list[str]],
) -> tuple[list[AlgebraicCandidate], list[tuple[str, str]]]:
    """Law 2: Block-Matrix Right-Distributivity ``x W1, x W2 -> x [W1 | W2]``."""
    by_id = {op.op_id: op for op in ordered}
    op_ids = set(by_id)
    by_producer: dict[tuple[tuple[str, str, int], ...], list[OpSpec]] = defaultdict(list)
    for op in ordered:
        if op.op_id in claimed:
            continue
        tr = traits[op.op_id]
        if tr.trait != AlgebraicTrait.BILINEAR_MATMUL or tr.is_grouped_matmul:
            continue
        if len(op.local_shape) != MATMUL_SHAPE_RANK:
            continue
        prod_port = _producer_port_key(op, op_ids, by_id=by_id)
        if prod_port is not None:
            by_producer[prod_port].append(op)

    found: list[AlgebraicCandidate] = []
    rejected: list[tuple[str, str]] = []
    for _prod_ports, siblings in by_producer.items():
        if len(siblings) < MIN_FUSE_MEMBERS:
            continue
        buckets: dict[tuple[Any, ...], list[OpSpec]] = defaultdict(list)
        for op in siblings:
            sib_key = _law2_sibling_key(op, all_ops=by_id)
            if sib_key is None:
                continue
            key = (op.stage, op.phase, op.stream or "main", op.layer, sib_key)
            buckets[key].append(op)
        if len(buckets) > 1:
            names = [op.op_id for op in siblings]
            rejected.append((
                "+".join(names[:2]),
                "quant_or_sharding_contract_mismatch",
            ))
        for bucket in buckets.values():
            if len(bucket) < MIN_FUSE_MEMBERS:
                continue
            independent: list[OpSpec] = []
            for cand in bucket:
                if any(
                    _is_ancestor(cand.op_id, prev.op_id, succ) or _is_ancestor(prev.op_id, cand.op_id, succ)
                    for prev in independent
                ):
                    continue
                independent.append(cand)
            if len(independent) < MIN_FUSE_MEMBERS:
                continue
            m_ids = tuple(op.op_id for op in independent)
            if not _convex(ordered, set(m_ids)):
                continue
            claimed.update(m_ids)
            found.append(AlgebraicCandidate(
                candidate_id=f"law2_block_mm_{independent[0].op_id}",
                law="law2_block_matmul",
                abi_id="gemm",
                fusion_tier="K2_epilogue",
                member_ids=m_ids,
                preserve_port_ids=m_ids,
                shipping_supported=True,
                avoidable_hbm_bytes=_law2_avoidable_read_bytes(independent, traits),
                traits=("BILINEAR_MATMUL",),
            ))
    return found, rejected


def _append_audit_trigger(
    triggers: list[SemanticAuditTrigger],
    trigger_keys: set[tuple[str, tuple[str, ...]]],
    traits: Mapping[str, OpTraitInfo],
    kind: Literal[
        "STATE_MUTATION",
        "SPECULATIVE_STATE_COMMIT",
        "ABI_UNSUPPORTED_VARIANT",
        "PRECISION_SENSITIVE_REASSOCIATION",
        "OPAQUE_CUSTOM_OP",
    ],
    ops_group: Sequence[OpSpec],
    reason: str,
    cand_abi: str = "",
) -> None:
    ids = tuple(dict.fromkeys(op.op_id for op in ops_group))
    key = (kind, ids)
    if key in trigger_keys or not ids:
        return
    avoid = _avoidable_hbm_bytes(ops_group, traits)
    if avoid < MIN_TRIGGER_AVOIDABLE_BYTES and kind not in {"ABI_UNSUPPORTED_VARIANT", "OPAQUE_CUSTOM_OP"}:
        avoid = max(avoid, sum(int(op.hbm_bytes or 0) for op in ops_group))
    src_refs = tuple(dict.fromkeys(
        str(dict(op.metadata).get("source_ref") or op.recipe or op.family)
        for op in ops_group
    ))
    trigger_keys.add(key)
    triggers.append(SemanticAuditTrigger(
        trigger_id=f"trig_{kind.lower()}_{len(triggers) + 1}",
        kind=kind,
        blocked_ops=ids,
        avoidable_hbm_bytes=int(avoid),
        potential_gain_us=0.0,
        potential_gain_us_per_tok_die=0.0,
        blocking_reason=reason,
        source_refs=src_refs,
        unlock_opcode="FuseRegion",
        candidate_abi=cand_abi,
    ))


def _has_moe_dispatch_combine_frame(
    op: OpSpec,
    tail: OpSpec,
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    succ: Mapping[str, list[str]],
) -> bool:
    """Require a complete connected ``dispatch -> GMM1 -> gate [-> quant] -> GMM2 -> combine`` frame."""
    dispatch_ops = [
        by_id[dep] for dep in op.deps
        if dep in by_id and traits[dep].symbolic_op_kind in {"dispatch", "alltoall"}
        and _same_domain(by_id[dep], op)
    ]
    if not dispatch_ops:
        return False
    disp_axes = {
        str(dict(d.metadata).get("group_axis") or traits[d.op_id].collective_axis or "").strip().lower()
        for d in dispatch_ops
    }
    disp_axes.discard("")
    if disp_axes and disp_axes != {"ep"}:
        return False
    for nxt_id in succ.get(tail.op_id, ()):
        if nxt_id not in by_id or not traits[nxt_id].is_grouped_matmul:
            continue
        gmm2 = by_id[nxt_id]
        if not _same_domain(tail, gmm2):
            continue
        for comb_id in succ.get(nxt_id, ()):
            if comb_id not in by_id or traits[comb_id].symbolic_op_kind != "combine":
                continue
            comb = by_id[comb_id]
            if not _same_domain(gmm2, comb):
                continue
            comb_axis = str(
                dict(comb.metadata).get("group_axis") or traits[comb_id].collective_axis or ""
            ).strip().lower()
            if comb_axis and comb_axis != "ep":
                continue
            if disp_axes and comb_axis and comb_axis not in disp_axes:
                continue
            return True
    return False


def _discover_gmm_act_candidates(
    ordered: Sequence[OpSpec],
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    succ: Mapping[str, list[str]],
    datapath: DatapathCapabilitySpec | None,
    triggers: list[SemanticAuditTrigger],
    trigger_keys: set[tuple[str, tuple[str, ...]]],
    *,
    include_abi_extensions: bool,
    arch_family: str = "",
    enable_fused_mc2: int | None = None,
) -> list[AlgebraicCandidate]:
    """Match GroupedMatmul -> Activation (-> Quant) via arch-qualified ABI capability predicates."""
    swiglu_pred = SHIPPING_ABI_PREDICATES["aclnnSwiGluQuant"]
    eff_arch = _resolve_arch_family(datapath, arch_family)
    found: list[AlgebraicCandidate] = []
    for op in ordered:
        if op.op_id in claimed or not traits[op.op_id].is_grouped_matmul:
            continue
        users = [u for u in succ.get(op.op_id, ()) if u in by_id and u not in claimed]
        if len(users) != 1:
            continue
        act = by_id[users[0]]
        act_tr = traits[act.op_id]
        if act_tr.trait != AlgebraicTrait.POINTWISE or not act_tr.activation_kind or not _same_domain(op, act):
            continue
        chain = [op, act]
        act_users = [u for u in succ.get(act.op_id, ()) if u in by_id and u not in claimed]
        if len(act_users) == 1 and traits[act_users[0]].symbolic_op_kind in {"dynamic_quant", "dynamic_mx_quant"}:
            q_op = by_id[act_users[0]]
            if _same_domain(act, q_op):
                chain.append(q_op)
        m_ids = tuple(x.op_id for x in chain)
        if not _convex(ordered, set(m_ids)):
            continue
        op_tr = traits[op.op_id]
        in_moe_frame = _has_moe_dispatch_combine_frame(op, chain[-1], by_id, traits, succ)
        moe_pred = (
            resolve_shipping_moe_predicate(
                eff_arch, weight_dtype=op_tr.weight_dtype, enable_fused_mc2=enable_fused_mc2,
            )
            if in_moe_frame else None
        )
        use_moe_abi = in_moe_frame and moe_pred is not None
        pred = moe_pred if use_moe_abi else swiglu_pred
        act_supported = pred.supports_activation(act_tr.activation_kind)
        dtype_supported = pred.supports_dtype(op_tr.input_dtype)
        weight_supported = (
            not use_moe_abi or not op_tr.weight_dtype or pred.supports_weight_dtype(op_tr.weight_dtype)
        )
        supported = act_supported and dtype_supported and weight_supported
        fits, ub_pk, ub_cap = _check_ub_chain_fits(chain[1:], traits, datapath)
        if not supported:
            ext_abi = (
                f"{pred.abi_id}_{act_tr.activation_kind}_ext"
                if use_moe_abi else "GroupedMatmulCustomActQuant"
            )
            reason = (
                f"activation={act_tr.activation_kind!r} not in shipping "
                f"supported_activations={sorted(pred.supported_activations)}"
                if not act_supported else
                f"dtype={op_tr.input_dtype!r}/weight={op_tr.weight_dtype!r} outside shipping {pred.abi_id}"
            )
            _append_audit_trigger(
                triggers,
                trigger_keys,
                traits,
                "ABI_UNSUPPORTED_VARIANT",
                chain,
                reason,
                cand_abi=ext_abi,
            )
            if fits and include_abi_extensions:
                claimed.update(m_ids)
                found.append(AlgebraicCandidate(
                    candidate_id=f"law1_gmm_ext_{op.op_id}",
                    law="law1_ub_chain",
                    abi_id=ext_abi,
                    fusion_tier="K3_dsl_pipeline",
                    member_ids=m_ids,
                    shipping_supported=False,
                    unsupported_reason=f"activation={act_tr.activation_kind}_requires_abi_extension",
                    ub_peak_bytes=ub_pk,
                    ub_budget_bytes=ub_cap,
                    avoidable_hbm_bytes=_avoidable_hbm_bytes(chain, traits),
                    traits=tuple(traits[oid].trait.value for oid in m_ids),
                ))
        elif fits:
            claimed.update(m_ids)
            found.append(AlgebraicCandidate(
                candidate_id=f"law1_gmm_act_{op.op_id}",
                law="law1_ub_chain",
                abi_id="aclnnSwiGluQuant" if len(chain) == MIN_FUSE_MEMBERS else "GroupedMatmulSwiGLUQuant",
                fusion_tier="K1_shipping" if len(chain) == MIN_FUSE_MEMBERS else "K2_epilogue",
                member_ids=m_ids,
                shipping_supported=True,
                ub_peak_bytes=ub_pk,
                ub_budget_bytes=ub_cap,
                avoidable_hbm_bytes=_avoidable_hbm_bytes(chain, traits),
                traits=tuple(traits[oid].trait.value for oid in m_ids),
            ))
    return found


def _collect_semantic_audit_triggers(
    ordered: Sequence[OpSpec],
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
    triggers: list[SemanticAuditTrigger],
    trigger_keys: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Emit structured triggers where pure algebra pauses at state/precision/opaque boundaries."""
    conv_neighbor_traits = {
        AlgebraicTrait.VIEW_METADATA, AlgebraicTrait.POINTWISE, AlgebraicTrait.ROW_REDUCTION,
    }
    for op in ordered:
        tr = traits[op.op_id]
        if tr.is_opaque_custom:
            users = [by_id[u] for u in succ.get(op.op_id, ()) if u in by_id and _same_domain(op, by_id[u])]
            _append_audit_trigger(
                triggers, trigger_keys, traits, "OPAQUE_CUSTOM_OP", [op, *users[:1]],
                f"opaque custom operator family={op.family!r} requires source semantic audit before cross-op fusion",
                cand_abi=f"Fused_{op.family}",
            )
            continue
        if tr.is_precision_sensitive:
            users = [
                by_id[u] for u in succ.get(op.op_id, ())
                if u in by_id and traits[u].trait == AlgebraicTrait.BILINEAR_MATMUL
            ]
            if users:
                _append_audit_trigger(
                    triggers, trigger_keys, traits, "PRECISION_SENSITIVE_REASSOCIATION", [op, users[0]],
                    "online softmax LSE rescale (FP32) -> MatMul cross-Cube reassociation forbidden without numerical proof",
                    cand_abi="FusedAttentionUpdateMatmul",
                )
            continue
        if not tr.is_stateful:
            continue
        if tr.symbolic_op_kind == "recover_ssm":
            preds = [by_id[p] for p in deps.get(op.op_id, set()) if p in by_id and traits[p].is_stateful]
            _append_audit_trigger(
                triggers, trigger_keys, traits, "SPECULATIVE_STATE_COMMIT", [*preds[:1], op],
                "speculative state commit mutates persistent SSM slots; confirm 2D slot-aliasing equivalence in source",
                cand_abi="FusedRecurrentStateCommit",
            )
            continue
        if tr.symbolic_op_kind != "conv1d":
            continue
        preds = [
            by_id[p] for p in deps.get(op.op_id, set())
            if p in by_id and traits[p].trait == AlgebraicTrait.VIEW_METADATA
        ]
        users = [
            by_id[u] for u in succ.get(op.op_id, ())
            if u in by_id and traits[u].trait in conv_neighbor_traits
        ]
        if preds or users:
            _append_audit_trigger(
                triggers, trigger_keys, traits, "STATE_MUTATION", [*preds[:1], op, *users[:2]],
                "stateful rolling conv_state update blocks pure algebraic prologue fusion; audit rolling state contract in source",
                cand_abi="FusedStatefulPrologue",
            )


def _pull_single_consumer_views(
    target: OpSpec,
    chain_ids: set[str],
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
) -> list[OpSpec]:
    return [
        by_id[p] for p in deps.get(target.op_id, set())
        if p in by_id and p not in claimed and p not in chain_ids
        and traits[p].trait == AlgebraicTrait.VIEW_METADATA
        and _same_domain(by_id[p], target)
        and len([u for u in succ.get(p, ()) if u in by_id]) == 1
    ]


def _discover_ub_chains(
    ordered: Sequence[OpSpec],
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
    datapath: DatapathCapabilitySpec | None,
) -> list[AlgebraicCandidate]:
    """Law 1 + Law 4: connected single-consumer UB chains over View/Pointwise/RowReduction."""
    eligible_traits = {AlgebraicTrait.VIEW_METADATA, AlgebraicTrait.POINTWISE, AlgebraicTrait.ROW_REDUCTION}
    found: list[AlgebraicCandidate] = []
    for op in ordered:
        if op.op_id in claimed:
            continue
        tr = traits[op.op_id]
        if tr.trait not in eligible_traits or tr.is_stateful or tr.is_opaque_custom:
            continue
        in_elig = [
            by_id[p] for p in deps.get(op.op_id, set())
            if p in by_id and p not in claimed
            and traits[p].trait in eligible_traits
            and not traits[p].is_stateful and not traits[p].is_opaque_custom
            and _same_domain(by_id[p], op)
            and len([u for u in succ.get(p, ()) if u in by_id]) == 1
        ]
        if in_elig:
            continue
        chain: list[OpSpec] = [op]
        cur = op
        while len(chain) < MAX_LAW1_CHAIN_LEN:
            users = [u for u in succ.get(cur.op_id, ()) if u in by_id and u not in claimed]
            if len(users) != 1:
                break
            nxt = by_id[users[0]]
            nxt_tr = traits[nxt.op_id]
            if (
                nxt_tr.trait not in eligible_traits
                or nxt_tr.is_stateful
                or nxt_tr.is_opaque_custom
                or not _same_domain(cur, nxt)
            ):
                break
            extras = _pull_single_consumer_views(nxt, {x.op_id for x in chain}, by_id, traits, claimed, deps, succ)
            proposed = [*chain, *extras, nxt]
            fits, _, _ = _check_ub_chain_fits(proposed, traits, datapath)
            if not fits or not _convex(ordered, {x.op_id for x in proposed}):
                break
            chain = proposed
            cur = nxt

        head_views = _pull_single_consumer_views(chain[0], {x.op_id for x in chain}, by_id, traits, claimed, deps, succ)
        if head_views and _convex(ordered, {x.op_id for x in (*head_views, *chain)}):
            chain = [*head_views, *chain]

        if len(chain) < MIN_FUSE_MEMBERS:
            continue
        chain_set = {c.op_id for c in chain}
        chain_ordered = [x for x in ordered if x.op_id in chain_set]
        fits, ub_pk, ub_cap = _check_ub_chain_fits(chain_ordered, traits, datapath)
        if not fits:
            continue
        abi_id, tier, shipping, unsup = _classify_ub_chain_abi(chain_ordered, traits)
        m_ids = tuple(x.op_id for x in chain_ordered)
        elided = tuple(x.op_id for x in chain_ordered if traits[x.op_id].trait == AlgebraicTrait.VIEW_METADATA)
        claimed.update(m_ids)
        found.append(AlgebraicCandidate(
            candidate_id=f"law1_ub_{chain_ordered[-1].op_id}",
            law="law4_view_or_abi" if len(elided) == len(chain_ordered) else "law1_ub_chain",
            abi_id=abi_id,
            fusion_tier=tier,
            member_ids=m_ids,
            shipping_supported=shipping,
            unsupported_reason=unsup,
            elided_view_ids=elided,
            ub_peak_bytes=ub_pk,
            ub_budget_bytes=ub_cap,
            avoidable_hbm_bytes=_avoidable_hbm_bytes(chain_ordered, traits),
            traits=tuple(traits[oid].trait.value for oid in m_ids),
        ))
    return found


def _discover_law1_and_law4_chains(
    ordered: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
    datapath: DatapathCapabilitySpec | None,
    *,
    include_abi_extensions: bool = True,
    arch_family: str = "",
    enable_fused_mc2: int | None = None,
) -> tuple[list[AlgebraicCandidate], list[SemanticAuditTrigger]]:
    """Law 1 (UB pointwise/reduction chains) + Law 4 (pure view elision & ABI match)."""
    by_id = {op.op_id: op for op in ordered}
    triggers: list[SemanticAuditTrigger] = []
    trigger_keys: set[tuple[str, tuple[str, ...]]] = set()
    found = _discover_gmm_act_candidates(
        ordered, by_id, traits, claimed, succ, datapath, triggers, trigger_keys,
        include_abi_extensions=include_abi_extensions,
        arch_family=arch_family,
        enable_fused_mc2=enable_fused_mc2,
    )
    _collect_semantic_audit_triggers(ordered, by_id, traits, deps, succ, triggers, trigger_keys)
    found.extend(_discover_ub_chains(ordered, by_id, traits, claimed, deps, succ, datapath))
    return found, triggers


def _classify_ub_chain_abi(
    chain: Sequence[OpSpec], traits: Mapping[str, OpTraitInfo],
) -> tuple[str, FusionTier, bool, str]:
    kinds = [traits[op.op_id].symbolic_op_kind for op in chain if traits[op.op_id].trait != AlgebraicTrait.VIEW_METADATA]
    has_quant = any(k in {"dynamic_quant", "dynamic_mx_quant"} for k in kinds)
    has_mx = "dynamic_mx_quant" in kinds
    acts = [traits[op.op_id].activation_kind for op in chain if traits[op.op_id].activation_kind]
    if "add" in kinds and "rmsnorm" in kinds and has_quant:
        return ("AddRmsNormDynamicMxQuant" if has_mx else "aclnnAddRmsNormDynamicQuant"), "K1_shipping", True, ""
    if "rmsnorm" in kinds and has_quant:
        return ("RmsNormDynamicMxQuant" if has_mx else "aclnnRmsNormDynamicQuant"), "K1_shipping", True, ""
    if "norm_gate" in kinds and has_quant:
        return ("FusedRMSNormGatedMxQuant" if has_mx else "FusedRMSNormGatedQuant"), "K2_epilogue", False, "custom_norm_gate_quant"
    if acts and has_quant:
        pred = SHIPPING_ABI_PREDICATES["aclnnSwiGluQuant"]
        act = acts[0]
        if pred.supports_activation(act) and not has_mx:
            return "aclnnSwiGluQuant", "K1_shipping", True, ""
        return f"Fused{act.capitalize()}{'Mx' if has_mx else ''}Quant", "K3_dsl_pipeline", False, f"activation={act}_abi_extension"
    if not kinds:
        return "StridedViewElided", "K1_shipping", True, ""
    if "gating" in kinds:
        return "FusedDecayGatingView", "K2_epilogue", False, "view_elided_gating"
    if "l2norm" in kinds:
        return "FusedQKL2NormView", "K2_epilogue", False, "view_elided_l2norm"
    if "layernorm" in kinds and has_quant:
        return ("FusedLayerNormMxQuant" if has_mx else "FusedLayerNormQuant"), "K2_epilogue", False, "custom_layernorm_quant"
    return "FusedUBChain", "K2_epilogue", False, "ub_chain_synthesis"


def discover_algebraic_rewrites_for_ops(
    ops: Sequence[OpSpec],
    *,
    datapath: DatapathCapabilitySpec | None = None,
    allow_mc2: bool = True,
    include_abi_extensions: bool = True,
    arch_family: str = "",
    enable_fused_mc2: int | None = None,
    base_graph_sha256: str = "",
    framing_sha256: str = "",
) -> AlgebraicRewriteResult:
    """Run all 4 universal algebraic laws over a rank-local ``OpSpec`` DAG."""
    if not ops:
        return AlgebraicRewriteResult(
            plan=empty_plan(base_graph_sha256, framing_sha256),
            candidates=(),
            semantic_audit_triggers=(),
        )
    by_id = {op.op_id: op for op in ops}
    ordered = _topological(by_id)
    traits = {op.op_id: classify_op_traits(op) for op in ordered}
    deps = _dependencies(ordered)
    succ = _successors(ordered)

    claimed: set[str] = set()
    candidates: list[AlgebraicCandidate] = []
    # Law 4a: Shipping composite MLA Prolog (Target + Draft).
    candidates.extend(_match_law4_mla_prolog(ordered, traits, claimed, deps, succ))
    # Law 3: Bilinear MatMul x Associative Collective Pipelining (MC2).
    candidates.extend(_discover_law3_mc2(ordered, traits, claimed, deps, succ, allow_mc2=allow_mc2))
    # Law 2: Block-Matrix Right-Distributivity.
    law2_cands, rejected = _discover_law2_block_matmul(ordered, traits, claimed, succ)
    candidates.extend(law2_cands)
    # Law 1 + Law 4b: Single-consumer UB pointwise/reduction chains + view elision + triggers.
    law1_cands, triggers = _discover_law1_and_law4_chains(
        ordered, traits, claimed, deps, succ, datapath,
        include_abi_extensions=include_abi_extensions,
        arch_family=arch_family,
        enable_fused_mc2=enable_fused_mc2,
    )
    candidates.extend(law1_cands)

    edits: list[Edit] = []
    used_edit_ids: set[str] = set()
    for idx, cand in enumerate(candidates):
        eid = cand.candidate_id
        if eid in used_edit_ids:
            eid = f"{eid}_{idx}"
        used_edit_ids.add(eid)
        refs = tuple(OpRef("atom", oid) for oid in cand.member_ids)
        preserve = tuple(PortRef(OpRef("atom", oid), "output", 0) for oid in cand.preserve_port_ids)
        edits.append(Edit(
            edit_id=eid,
            operation=FuseRegion(
                members=refs,
                preserve_ports=preserve,
                kernel_boundaries=(KernelBoundary(cand.abi_id, refs),),
                fusion_tier=cand.fusion_tier,
            ),
        ))
    plan = RewritePlan(
        base_graph_sha256=base_graph_sha256,
        framing_sha256=framing_sha256,
        edits=tuple(edits),
    )
    return AlgebraicRewriteResult(
        plan=plan,
        candidates=tuple(candidates),
        semantic_audit_triggers=tuple(triggers),
        rejected_pairs=tuple(rejected),
    )


def discover_algebraic_rewrites(
    scenario: ResolvedScenario,
    *,
    allow_mc2: bool | None = None,
    include_abi_extensions: bool = True,
    verify: bool = True,
) -> AlgebraicRewriteResult:
    """Discover Tier-P2 algebraic rewrites on a ``ResolvedScenario`` and verify via ``verify_plan``."""
    from .apply import apply_plan
    from .verify import verify_plan

    meta = dict(scenario.case.runtime.metadata) if scenario.case.runtime.metadata else {}
    add_cfg = meta.get("additional_config") if isinstance(meta.get("additional_config"), Mapping) else {}
    raw_mc2 = add_cfg.get("enable_fused_mc2")
    fused_mc2_level = int(raw_mc2) if raw_mc2 is not None else None
    if allow_mc2 is None:
        allow_mc2 = (fused_mc2_level != 0) if fused_mc2_level is not None else True
    arch_family = str(getattr(scenario.case.machine, "arch_family", "") or "")

    rep_ops = scenario.atomic_ops[0].ops if scenario.atomic_ops else ()
    raw_res = discover_algebraic_rewrites_for_ops(
        rep_ops,
        datapath=scenario.datapath,
        allow_mc2=bool(allow_mc2),
        include_abi_extensions=include_abi_extensions,
        arch_family=arch_family,
        enable_fused_mc2=fused_mc2_level,
        base_graph_sha256=scenario.base_graph_sha256,
        framing_sha256=scenario.framing_sha256,
    )
    if not verify:
        return raw_res
    try:
        applied = apply_plan(raw_res.plan, scenario)
        receipt = verify_plan(scenario, applied, scope="archetype")
    except ModelingError:
        receipt = None
    if receipt is None or not receipt.passed:
        # Filter down to individually verified edits so the returned plan is always sound.
        safe_edits: list[Edit] = []
        safe_cands: list[AlgebraicCandidate] = []
        for edit, cand in zip(raw_res.plan.edits, raw_res.candidates):
            trial = RewritePlan(
                base_graph_sha256=scenario.base_graph_sha256,
                framing_sha256=scenario.framing_sha256,
                edits=(*safe_edits, edit),
            )
            try:
                t_app = apply_plan(trial, scenario)
                t_rec = verify_plan(scenario, t_app, scope="archetype")
            except ModelingError:
                continue
            if t_rec.passed:
                safe_edits.append(edit)
                safe_cands.append(cand)
        plan = RewritePlan(
            base_graph_sha256=scenario.base_graph_sha256,
            framing_sha256=scenario.framing_sha256,
            edits=tuple(safe_edits),
        )
        final_rec = verify_plan(scenario, apply_plan(plan, scenario), scope="archetype")
        return AlgebraicRewriteResult(
            plan=plan,
            candidates=tuple(safe_cands),
            semantic_audit_triggers=raw_res.semantic_audit_triggers,
            rejected_pairs=raw_res.rejected_pairs,
            verification=final_rec,
        )
    return AlgebraicRewriteResult(
        plan=raw_res.plan,
        candidates=raw_res.candidates,
        semantic_audit_triggers=raw_res.semantic_audit_triggers,
        rejected_pairs=raw_res.rejected_pairs,
        verification=receipt,
    )


__all__ = [
    "AlgebraicTrait",
    "ShippingAbiPredicate",
    "SHIPPING_ABI_PREDICATES",
    "resolve_shipping_moe_predicate",
    "OpTraitInfo",
    "AlgebraicCandidate",
    "SemanticAuditTrigger",
    "AlgebraicRewriteResult",
    "classify_op_traits",
    "discover_algebraic_rewrites_for_ops",
    "discover_algebraic_rewrites",
]
