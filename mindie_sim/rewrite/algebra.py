"""Universal algebraic graph rewrite passes over typed operator traits.

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
from dataclasses import dataclass, replace
from enum import Enum
from math import prod
from typing import Any, Literal

from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
from mindie_sim.prediction.fusion.ports import control_dependency_names, full_operand_ports
from mindie_sim.prediction.runtime import tile_solver
from mindie_sim.prediction.runtime.tile_solver import (
    DEFAULT_POOL_GRANULE_BYTES,
    LiveBufferSpec,
)
from mindie_sim.schema.abi import (
    RANK_PLACEMENTS,
    AbiQueryKey,
    AbiSupportVerdict,
    OperandLayout,
    OperandPortSig,
    RankPlacement,
    RegionBackendTarget,
    SocArch,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.stack import ManifestBinding, StackTrack, StackTrackSpec

from .apply import (
    ResolvedScenario,
    _activation_quant_key,
    _convex,
    _dependencies,
    _law2_sibling_key,
    _successors,
    _topological,
    has_illegal_cross_precision_merge,
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
#: Default UB alignment granule (bytes).
UB_ALIGN_BYTES = DEFAULT_POOL_GRANULE_BYTES
#: Maximum chain length for Law 1 UB pointwise/reduction composition.
MAX_LAW1_CHAIN_LEN = 12
#: Microscaling FP8/FP4/HiF8 group size.
MX_GROUP_SIZE = 32
#: CATLASS dual-level MXFP4 level-0 scale1 block width (columns per FP32 scale1 element).
_DUAL_LEVEL_MX_SCALE1_BLOCK = 512
#: CANN/CATLASS E8M0 microscaling pair alignment (2 scale factors packed per pair).
_MX_SCALE_PAIR_ALIGN = 2
#: Per-block FP8 quantization group size.
BLOCK_QUANT_GROUP_SIZE = 128
#: Minimum collective domain world size when collective metadata omits explicit ranks.
DEFAULT_COLLECTIVE_WORLD_SIZE = 2
#: Exact branch-and-bound ceiling for competing-region conflict components.
MAX_CONFLICT_EXACT_SIZE = 16
#: Micro-epilogue duration threshold (us) for Law 2b MatMul epilogue absorption.
MICRO_EPILOGUE_MAX_US = 2.0
#: Maximum standalone Law 2b epilogue ops absorbed after one MatMul.
MAX_LAW2B_EPILOGUE_OPS = 2
#: Default shipping stack fusion passes enabled on merged shipping/proposed/target tracks.
DEFAULT_SHIPPING_STACK_PASSES: frozenset[str] = frozenset({
    "mm_reduce_scatter_fusion",
    "norm_quant_fusion",
})
#: Families eligible as Law 1 terminal cache-quant sinks (excludes rolling conv/SSM state).
_TERMINAL_SINK_FAMILIES = frozenset({
    "cache_store",
    "indexer_quant_cache",
    "kv_cache_write",
    "kv_compress_epilog",
})
#: Row-reduction families explicitly barred from Law 2b MatMul micro-epilogue absorption.
_MICRO_EPILOGUE_EXCLUDED_FAMILIES = frozenset({"topk", "moe_init_routing"})
#: View families proven zero-copy (excludes copy/cat/pad/contiguous materialization).
_ZERO_COPY_VIEW_FAMILIES = frozenset({"view", "reshape", "slice", "transpose", "permute"})
#: Input binding roles that indicate a non-data control/state port on terminal sinks.
_TERMINAL_SINK_NON_DATA_ROLES = frozenset({
    "slot_mapping", "slots", "cache", "cache_ref", "cache_scale", "cache_scale_ref", "x_scale",
})
#: Dtype canonicalization matching ``ProbedAbiRegistry``.
_ABI_DTYPE_CANON = {
    "fp8_e4m3": "fp8_e4m3fn",
    "float8_e4m3fn": "fp8_e4m3fn",
    "fp8_e4m3fn": "fp8_e4m3fn",
    "mxfp8": "fp8_e4m3fn",
    "mxfp4": "fp4_e2m1",
    "fp4": "fp4_e2m1",
    "fp4_e2m1": "fp4_e2m1",
    "hif8": "hifloat8",
    "hifp8": "hifloat8",
    "hifloat8": "hifloat8",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "e8m0": "fp8_e8m0",
    "fp8_e8m0": "fp8_e8m0",
}
_DEFAULT_SKU_BY_ARCH: dict[SocArch, str] = {
    "A2": "ascend910b4",
    "A3": "ascend910_9362",
    "A5": "ascend950dt",
}
_DEFAULT_REMOTE_PLACEMENT_BY_ARCH: dict[SocArch, RankPlacement] = {
    "A2": "hccs_single_node",
    "A3": "hccs_supernode_a3",
    "A5": "ub_ccu_intramachine",
}
_ALLOWED_QUANT_FORMATS = frozenset({"int8", "int4", "fp8_e4m3fn", "fp8_e5m2", "fp4_e2m1", "hifloat8"})
_MX_DTYPES = frozenset({"fp8_e4m3fn", "fp4_e2m1", "hifloat8"})
_MEGAMOE_CONTEXT_BYTES_SHAPE = (1,)
DEFAULT_MLA_ROPE_HEAD_DIM = 64


class AlgebraicTrait(str, Enum):
    """Typed algebraic classification driving provable graph rewrites."""

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
    input_dtype: str = ""
    output_dtype: str = ""
    weight_dtype: str = ""
    group_size: int | None = None
    scale_format: str = ""
    is_stateful: bool = False
    is_precision_sensitive: bool = False
    is_opaque_custom: bool = False
    row_working_bytes: int = 0
    symbolic_op_kind: str = "elementwise"
    is_grouped_matmul: bool = False
    is_terminal_sink: bool = False
    is_micro_epilogue_eligible: bool = False


@dataclass(frozen=True, slots=True)
class AlgebraicCandidate:
    """One mathematically justified fusion candidate emitted by Laws 1-4."""

    candidate_id: str
    law: Literal["law1_ub_chain", "law2_block_matmul", "law3_mc2", "law4_view_or_abi"]
    abi_id: str
    fusion_tier: FusionTier
    member_ids: tuple[str, ...]
    shipping_supported: bool
    preserve_port_ids: tuple[str, ...] = ()
    unsupported_reason: str = ""
    elided_view_ids: tuple[str, ...] = ()
    ub_peak_bytes: int = 0
    ub_budget_bytes: int = 0
    ub_precision_mode: str = "fp32_upcast"
    avoidable_hbm_bytes: int = 0
    traits: tuple[str, ...] = ()
    abi_verdict: AbiSupportVerdict | None = None
    abi_query_digest: str = ""
    backend_target: RegionBackendTarget | None = None
    lowering_id: str = ""
    lowering_sha256: str = ""
    fused_recipe: str = ""
    preserve_ports: tuple[tuple[str, str, int], ...] = ()
    requires_comm_consensus: bool = False


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
    """Output of algebraic rewrite discovery over a rank DAG or ``ResolvedScenario``."""

    plan: RewritePlan
    candidates: tuple[AlgebraicCandidate, ...]
    semantic_audit_triggers: tuple[SemanticAuditTrigger, ...]
    rejected_pairs: tuple[tuple[str, str], ...] = ()
    verification: Any = None


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


_MEGAMOE_DISPATCH_MODE_PASSTHROUGH = 0
_MEGAMOE_DISPATCH_MODE_INT8 = 2
_MEGAMOE_DISPATCH_MODE_MXFP = 4
_MEGAMOE_MX_SCALE_ALIGN = 64
_MEGAMOE_MX_SCALE_TRAILING = 2
_SWIGLU_WIDTH_DIVISOR = 2
_ADD_RMSNORM_UPCAST_WORK_TILES = 2
_DEFAULT_NORM_WORK_TILES = 1
_SIMT_BACKENDS = frozenset({"tilelang_simt", "tla_simt"})


def _estimate_row_working_bytes(
    shape: Sequence[int],
    in_dt: str,
    out_dt: str,
    *,
    is_reduction: bool,
    is_vector_compute: bool = True,
    sym_kind: str = "",
    declared_work_tiles: int | None = None,
) -> int:
    """Budget UB row working bytes using Ascend NPU FP32 (4B) AIV upcast."""
    if not shape:
        return 0
    row_elems = int(shape[-1]) if len(shape) >= 2 else int(shape[0])
    if row_elems <= 0:
        return 0
    in_b = float(dtype_bytes(in_dt))
    out_b = float(dtype_bytes(out_dt))
    if is_vector_compute:
        # Ascend AIV upcasts BF16/FP16 to FP32 (4B/elem) inside UB for compute.
        fp32_work = float(FP32_ACCUM_BYTES)
        if is_reduction:
            if declared_work_tiles is not None and declared_work_tiles > 0:
                tiles = declared_work_tiles
            elif sym_kind in {"add_rmsnorm", "add_rmsnorm_quant"} and in_b < fp32_work:
                tiles = _ADD_RMSNORM_UPCAST_WORK_TILES
            else:
                tiles = _DEFAULT_NORM_WORK_TILES
            accum_b = float(tiles) * fp32_work
        else:
            accum_b = 0.0
        raw = int(row_elems * (max(in_b, fp32_work) + max(out_b, fp32_work) + accum_b))
    else:
        raw = int(row_elems * (in_b + out_b))
    return ((raw + UB_ALIGN_BYTES - 1) // UB_ALIGN_BYTES) * UB_ALIGN_BYTES


def _resolve_symbolic_kind(
    fam: str, rec: str, meta: Mapping[str, Any], trait: AlgebraicTrait, act_kind: str,
) -> str:
    if rec == "megamoe_arch35" or "megamoe" in rec:
        return "megamoe"
    if fam in {"grouped_gemm", "grouped_matmul", "grouped_moe"}:
        return "grouped_matmul"
    if trait == AlgebraicTrait.BILINEAR_MATMUL:
        return "matmul"
    if fam in {"router", "moe_router", "moe_gating"} or any(
        t in rec for t in ("moe_gating", "topk_softmax", "gating_topk", "router")
    ):
        return "moe_router_topk"
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
    if fam == "gdn_gating" or (
        "gating" in rec
        and not any(t in rec for t in ("moe", "router", "topk"))
        and not any(t in fam for t in ("moe", "router", "topk"))
    ):
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


def _abi_weight_group_size(p_abi: Any) -> int:
    if p_abi is None:
        return 0
    wb = getattr(p_abi, "weight_block", None)
    if isinstance(wb, tuple) and wb:
        vals = [int(x) for x in wb if isinstance(x, int) and not isinstance(x, bool) and x > 0]
        if vals:
            return vals[-1]
    if isinstance(wb, int) and not isinstance(wb, bool) and wb > 0:
        return wb
    return 0


def classify_op_traits(op: OpSpec) -> OpTraitInfo:
    """Classify an ``OpSpec`` purely by typed operator family, work, and metadata."""
    fam = str(op.family or "").strip().lower()
    rec = str(op.recipe or "").strip().lower()
    meta = dict(op.metadata) if op.metadata else {}
    p_abi = getattr(op, "precision_abi", None)

    in_dt = str(
        (getattr(p_abi, "input_dtype", "") if p_abi is not None else "")
        or meta.get("activation_format")
        or meta.get("input_format")
        or meta.get("comm_dtype")
        or op.dtype
        or ""
    ).strip().lower()
    out_dt = str(
        (getattr(p_abi, "output_dtype", "") if p_abi is not None else "")
        or meta.get("output_format")
        or meta.get("comm_dtype")
        or op.dtype
        or ""
    ).strip().lower()
    w_dt = str(
        (getattr(p_abi, "weight_dtype", "") if p_abi is not None else "")
        or meta.get("weight_format")
        or meta.get("weight_dtype")
        or op.dtype
        or ""
    ).strip().lower()
    abi_gs = (
        (getattr(p_abi, "act_group_size", 0) or _abi_weight_group_size(p_abi))
        if p_abi is not None else 0
    )
    raw_gs = abi_gs if (isinstance(abi_gs, int) and not isinstance(abi_gs, bool) and abi_gs > 0) else meta.get(
        "activation_group_size", meta.get("quant_group_size")
    )
    group_size = int(raw_gs) if isinstance(raw_gs, int) and not isinstance(raw_gs, bool) and raw_gs > 0 else None
    scale_fmt = str(
        (
            (getattr(p_abi, "act_scale_dtype", "") or getattr(p_abi, "weight_scale_dtype", ""))
            if p_abi is not None else ""
        )
        or meta.get("activation_scale_format")
        or meta.get("quant_scale_dtype")
        or ""
    ).strip().lower()
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
    if (not in_dt or not out_dt) and not is_opaque and trait not in {
        AlgebraicTrait.ASSOCIATIVE_COLLECTIVE,
        AlgebraicTrait.CONTROL_BARRIER,
        AlgebraicTrait.VIEW_METADATA,
    }:
        raise ModelingError(f"algebra_missing_op_dtype:{op.op_id}")
    sym_kind = _resolve_symbolic_kind(fam, rec, meta, trait, act_kind)
    raw_tiles = meta.get("native_norm_fp32_working_tiles")
    declared_tiles = (
        int(raw_tiles) if isinstance(raw_tiles, int) and not isinstance(raw_tiles, bool) and raw_tiles > 0 else None
    )
    row_bytes = (
        _estimate_row_working_bytes(
            op.local_shape,
            in_dt,
            out_dt,
            is_reduction=(trait == AlgebraicTrait.ROW_REDUCTION),
            is_vector_compute=(trait in {AlgebraicTrait.POINTWISE, AlgebraicTrait.ROW_REDUCTION}),
            sym_kind=sym_kind,
            declared_work_tiles=declared_tiles,
        )
        if (in_dt and out_dt)
        else 0
    )
    is_grouped = fam in {"grouped_gemm", "grouped_matmul", "grouped_moe"} or "grouped" in rec
    is_terminal_sink = fam in _TERMINAL_SINK_FAMILIES
    is_zero_copy_view = trait == AlgebraicTrait.VIEW_METADATA and fam in _ZERO_COPY_VIEW_FAMILIES
    is_micro_epilogue = (
        not is_stateful
        and not is_opaque
        and not is_prec_sens
        and fam not in _MICRO_EPILOGUE_EXCLUDED_FAMILIES
        and (
            trait in {AlgebraicTrait.POINTWISE, AlgebraicTrait.ROW_REDUCTION}
            or is_zero_copy_view
        )
    )
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
        is_terminal_sink=is_terminal_sink,
        is_micro_epilogue_eligible=is_micro_epilogue,
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


def _chain_uses_simt(ops: Sequence[OpSpec]) -> bool:
    for op in ops:
        meta = dict(op.metadata) if op.metadata else {}
        if meta.get("uses_simt") or meta.get("tile_uses_simt") or meta.get("template_param_uses_simt"):
            return True
        backend = str(meta.get("backend_target") or meta.get("backend") or "").strip().lower()
        if backend in _SIMT_BACKENDS:
            return True
    return False


def _data_dependency_graph(
    ordered: Sequence[OpSpec],
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Build tensor data-flow dependencies excluding pure control-only edges."""
    from .apply import _named_inputs, _typed_inputs

    by_id = {op.op_id: op for op in ordered}
    data_deps: dict[str, set[str]] = {}
    data_succ: dict[str, list[str]] = {op.op_id: [] for op in ordered}
    for op in ordered:
        ctrl = control_dependency_names(op, by_id)
        ports = full_operand_ports(op, by_id)
        typed_prods = {
            str(port.get("producer") or "").strip()
            for _, port in _typed_inputs(op)
            if str(port.get("role") or "").strip().lower() not in {"control", "barrier", "event"}
        }
        named_prods = {v for v in _named_inputs(op).values() if v in by_id}
        resolved = {dep for dep, _ in ports if dep in by_id} | (typed_prods & by_id.keys()) | named_prods
        if not resolved and not ctrl and not ports:
            resolved = {dep for dep in op.deps if dep in by_id}
        else:
            resolved = {dep for dep in resolved if dep in by_id}
        resolved.discard(op.op_id)
        data_deps[op.op_id] = resolved
        for src in resolved:
            data_succ[src].append(op.op_id)
    return data_deps, data_succ


def _usable_ub_bytes(
    datapath: DatapathCapabilitySpec | None, *, uses_simt: bool = False,
) -> int:
    if datapath is None:
        return 0
    raw = int(dict(datapath.buffers).get("ub", 0) or 0)
    if raw <= 0:
        return 0
    carveout = int(datapath.simt_ub_carveout_bytes or 0) if uses_simt else 0
    return max(0, raw - carveout)


def _simultaneous_ub_buffers(
    members: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    *,
    slots: int,
) -> tuple[LiveBufferSpec, ...]:
    """Build single-counted simultaneous-lifetime UB buffers (working + skip edges)."""
    pos_by_id = {op.op_id: idx for idx, op in enumerate(members)}
    buffers: list[LiveBufferSpec] = []
    for idx, op in enumerate(members):
        row_b = traits[op.op_id].row_working_bytes
        if row_b > 0:
            buffers.append(LiveBufferSpec(
                f"stage_{idx}", "ub", row_b, idx, idx + 1, owner_kind="aiv", slots=slots,
            ))
    for c_idx, op in enumerate(members):
        for dep in op.deps:
            p_idx = pos_by_id.get(dep)
            if p_idx is None or c_idx <= p_idx + 1:
                continue
            p_op = members[p_idx]
            p_dt = traits[dep].output_dtype
            edge_b = (
                _estimate_row_working_bytes(
                    p_op.local_shape, p_dt, p_dt,
                    is_reduction=False, is_vector_compute=False,
                )
                if p_dt
                else 0
            )
            if edge_b > 0:
                buffers.append(LiveBufferSpec(
                    f"edge_{p_idx}_{c_idx}", "ub", edge_b, p_idx, c_idx + 1,
                    owner_kind="aiv", slots=1,
                ))
    return tuple(buffers)


def _check_ub_chain_fits(
    members: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    datapath: DatapathCapabilitySpec | None,
    *,
    uses_simt: bool = False,
) -> tuple[bool, int, int]:
    """Verify exact single-counted simultaneous-lifetime UB fit via ``tile_solver``."""
    budget = _usable_ub_bytes(datapath, uses_simt=uses_simt)
    if budget <= 0 or datapath is None:
        return False, 0, 0
    for slots in (UB_DOUBLE_BUFFER_SLOTS, 1):
        bufs = _simultaneous_ub_buffers(members, traits, slots=slots)
        if not bufs:
            return True, 0, budget
        peak = tile_solver._pool_peak_bytes(bufs, "aiv", "ub", UB_ALIGN_BYTES)
        if peak <= budget:
            return True, peak, budget
    single_bufs = _simultaneous_ub_buffers(members, traits, slots=1)
    peak_min = tile_solver._pool_peak_bytes(single_bufs, "aiv", "ub", UB_ALIGN_BYTES) if single_bufs else 0
    return False, peak_min, budget


def _output_shape_bytes(op: OpSpec, tr: OpTraitInfo) -> float:
    """Return logical output tensor bytes for ``op`` (``(M, N)`` for 3D MatMul)."""
    if not op.local_shape or not tr.output_dtype:
        return 0.0
    if tr.trait == AlgebraicTrait.BILINEAR_MATMUL and len(op.local_shape) == MATMUL_SHAPE_RANK:
        elems = int(op.local_shape[0]) * int(op.local_shape[1])
    else:
        elems = prod(int(d) for d in op.local_shape if int(d) > 0)
    return float(max(0, elems)) * float(dtype_bytes(tr.output_dtype))


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
    act_bytes = float(m_dim * k_dim) * float(dtype_bytes(a_dt))
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
    scope_a = str(dict(a.metadata).get("model_scope") or "target").strip().lower()
    scope_b = str(dict(b.metadata).get("model_scope") or "target").strip().lower()
    return (
        a.stage, a.phase, a.stream or "main", a.layer, scope_a,
    ) == (
        b.stage, b.phase, b.stream or "main", b.layer, scope_b,
    )


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


def _resolve_soc_arch(
    arch_family: str,
    datapath: DatapathCapabilitySpec | None,
    sku_id: str = "",
) -> SocArch | None:
    raw = str(arch_family or "").strip().upper()
    if raw in {"A2", "A3", "A5"}:
        return raw  # type: ignore[return-value]
    low = f"{raw.lower()} {str(sku_id or '').strip().lower()}"
    if "arch35" in low or "3510" in low or "950" in low:
        return "A5"
    if "910_93" in low or "910c" in low or "a3" in low:
        return "A3"
    if "910b" in low or "a2" in low:
        return "A2"
    if datapath is not None:
        arch_id = str(getattr(datapath.compute_arch, "arch_id", "") or "").lower()
        if "3510" in arch_id or "35" in arch_id:
            return "A5"
        if "22" in arch_id:
            return "A2" if "910b" in low else "A3"
    return None


def _norm_abi_dtype(raw: str) -> str:
    low = str(raw or "").strip().lower()
    return _ABI_DTYPE_CANON.get(low, low)


def _quant_contract_for_port(
    dtype: str,
    scale_format: str,
    group_size: int | None,
    *,
    is_weight: bool,
) -> tuple[str, tuple[int, ...], str]:
    dt = _norm_abi_dtype(dtype)
    sf = _norm_abi_dtype(scale_format) if scale_format else ""
    if dt in {"bf16", "fp16", "fp32"}:
        return "dense", (), "none"
    if dt == "fp4_e2m1":
        return "mx_fp4", (group_size or MX_GROUP_SIZE,), (sf or "fp8_e8m0")
    if dt == "hifloat8":
        return "hif8_mx", (group_size or MX_GROUP_SIZE,), (sf or "fp8_e8m0")
    if dt in {"fp8_e4m3fn", "fp8_e5m2"}:
        if group_size == BLOCK_QUANT_GROUP_SIZE and sf != "fp8_e8m0":
            return "per_block_128", (BLOCK_QUANT_GROUP_SIZE,), (sf or "fp32")
        if sf and sf != "fp8_e8m0":
            return "per_group_fp8", ((group_size,) if group_size else ()), sf
        return "mx_fp8", (group_size or MX_GROUP_SIZE,), "fp8_e8m0"
    if dt == "int4":
        return "w4a8_int", (), (sf or "fp32")
    if dt == "int8":
        return ("per_channel_int8" if is_weight else "per_token_int8"), (), (sf or "fp32")
    return "none", (), "none"


def _resolve_declared_weight_layout(
    op: OpSpec,
    p_abi: Any,
    *,
    default_when_undeclared: OperandLayout = "ND",
) -> OperandLayout:
    meta = dict(op.metadata) if op.metadata else {}
    explicit_meta = str(meta.get("weight_layout") or meta.get("weight_representation") or "").strip().upper()
    if explicit_meta:
        if "FRACTAL_NZ_C0_32" in explicit_meta or "NZ_C0_32" in explicit_meta:
            return "FRACTAL_NZ_C0_32"
        if "FRACTAL_NZ" in explicit_meta or explicit_meta == "NZ":
            return "FRACTAL_NZ"
        if "FRACTAL_Z" in explicit_meta:
            return "FRACTAL_Z"
        return "ND"
    if p_abi is not None:
        abi_wl = str(getattr(p_abi, "weight_layout", "") or "").strip().upper()
        if "FRACTAL_NZ_C0_32" in abi_wl or "NZ_C0_32" in abi_wl:
            return "FRACTAL_NZ_C0_32"
        if "FRACTAL_NZ" in abi_wl or abi_wl == "NZ":
            return "FRACTAL_NZ"
        if "FRACTAL_Z" in abi_wl:
            return "FRACTAL_Z"
        return "ND"
    return default_when_undeclared


def _resolve_op_weight_port_contract(
    op: OpSpec,
    tr: OpTraitInfo,
    *,
    default_layout: OperandLayout = "ND",
) -> tuple[str, str, tuple[int, ...], str, OperandLayout]:
    p_abi = getattr(op, "precision_abi", None)
    meta = dict(op.metadata) if op.metadata else {}
    raw_w = (
        getattr(p_abi, "weight_dtype", "") if (p_abi is not None and getattr(p_abi, "weight_dtype", ""))
        else tr.weight_dtype
    )
    w_dt = _norm_abi_dtype(raw_w)
    abi_wgs = _abi_weight_group_size(p_abi)
    raw_wgs = meta.get("weight_group_size", meta.get("quant_group_size"))
    w_gs = (
        abi_wgs if abi_wgs > 0
        else (int(raw_wgs) if isinstance(raw_wgs, int) and not isinstance(raw_wgs, bool) and raw_wgs > 0 else tr.group_size)
    )
    w_sf = str(
        (getattr(p_abi, "weight_scale_dtype", "") if p_abi is not None else "")
        or meta.get("weight_scale_format")
        or meta.get("weight_scale_dtype")
        or tr.scale_format
        or ""
    ).strip().lower()
    wq, wg, ws_dt = _quant_contract_for_port(w_dt, w_sf, w_gs, is_weight=True)
    w_layout = _resolve_declared_weight_layout(op, p_abi, default_when_undeclared=default_layout)
    return w_dt, wq, wg, ws_dt, w_layout


def _resolve_op_act_port_contract(
    op: OpSpec,
    tr: OpTraitInfo,
) -> tuple[str, str, tuple[int, ...], str]:
    p_abi = getattr(op, "precision_abi", None)
    meta = dict(op.metadata) if op.metadata else {}
    raw_in = (
        getattr(p_abi, "input_dtype", "") if (p_abi is not None and getattr(p_abi, "input_dtype", ""))
        else tr.input_dtype
    )
    in_dt = _norm_abi_dtype(raw_in)
    abi_ags = int(getattr(p_abi, "act_group_size", 0) or 0) if p_abi is not None else 0
    raw_ags = meta.get("activation_group_size", meta.get("quant_group_size"))
    a_gs = (
        abi_ags if abi_ags > 0
        else (int(raw_ags) if isinstance(raw_ags, int) and not isinstance(raw_ags, bool) and raw_ags > 0 else tr.group_size)
    )
    a_sf = str(
        (getattr(p_abi, "act_scale_dtype", "") if p_abi is not None else "")
        or meta.get("activation_scale_format")
        or meta.get("act_scale_dtype")
        or tr.scale_format
        or ""
    ).strip().lower()
    aq, ag, as_dt = _quant_contract_for_port(in_dt, a_sf, a_gs, is_weight=False)
    return in_dt, aq, ag, as_dt


def _port(
    role: str,
    dtype: str,
    shape: tuple[int, ...],
    *,
    layout: OperandLayout = "ND",
    transposed: bool = False,
    quant_mode: str = "dense",
    group_shape: tuple[int, ...] = (),
    scale_dtype: str = "none",
) -> OperandPortSig:
    return OperandPortSig(
        role=role,
        dtype=_norm_abi_dtype(dtype),
        layout=layout,
        transposed=transposed,
        shape=shape,
        quant_mode=quant_mode,
        group_shape=group_shape,
        scale_dtype=scale_dtype,
        scale_shape=(),
    )


def _resolve_placement(arch: SocArch, world_size: int, meta: Mapping[str, Any]) -> RankPlacement:
    if world_size <= 1:
        return "local"
    explicit = str(meta.get("rank_placement") or "").strip().lower()
    if explicit in RANK_PLACEMENTS and explicit != "local":
        return explicit  # type: ignore[return-value]
    return _DEFAULT_REMOTE_PLACEMENT_BY_ARCH[arch]


def _resolve_world_size(
    op: OpSpec, default_ws: int | None, *, collective: bool = False,
) -> int:
    meta = dict(op.metadata) if op.metadata else {}
    for key in ("comm_world_size", "world_size", "tp_size", "ep_size", "moe_ep_size"):
        raw = meta.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
            return raw
    if default_ws is not None and default_ws >= 1:
        return default_ws
    return DEFAULT_COLLECTIVE_WORLD_SIZE if collective else 1


def _verify_abi_query(
    *,
    abi_id: str,
    soc_arch: SocArch,
    sku_id: str,
    graph_motif: str,
    operands: tuple[OperandPortSig, ...],
    outputs: tuple[OperandPortSig, ...],
    pr_track: StackTrack,
    track_spec: StackTrackSpec | None,
    manifest_bindings: tuple[ManifestBinding, ...],
    enabled_stack_passes: frozenset[str],
    registry: ProbedAbiRegistry | None = None,
    model_scope: Literal["target", "draft"] = "target",
    rank_placement: RankPlacement = "local",
    world_size: int = 1,
    comm_group_id: str = "local",
    comm_dtype: str = "none",
    reduction_kind: str = "none",
    activation_kind: str = "none",
    activation_params: tuple[tuple[str, float], ...] = (),
    has_bias: bool = False,
    has_active_mask: bool = False,
    dispatch_quant_mode: int = -1,
    shared_experts_per_rank: int = 0,
    top_k: int = 0,
    moe_expert_num: int = 0,
    qk_rope_head_dim: int = 0,
    mla_variant: Literal["none", "latent_mla", "latent_mqa"] = "none",
    kv_cache_dtype: str = "none",
    has_kpool_indexer: bool = False,
    dcp_size: int = 1,
    pcp_size: int = 1,
    is_act_and_mul: bool = False,
) -> tuple[AbiSupportVerdict, str]:
    reg = registry or default_abi_registry()
    query = AbiQueryKey(
        soc_arch=soc_arch,
        sku_id=sku_id or _DEFAULT_SKU_BY_ARCH[soc_arch],
        abi_id=abi_id,
        graph_motif=graph_motif,
        model_scope=model_scope,
        draft_quant_mode="none",
        pr_track=pr_track,
        operands=operands,
        outputs=outputs,
        rank_placement=rank_placement,
        world_size=world_size,
        comm_group_id=comm_group_id,
        comm_domain_width=world_size,
        comm_dtype=comm_dtype,
        reduction_kind=reduction_kind,
        activation_kind=activation_kind,
        activation_params=activation_params,
        has_bias=has_bias,
        has_active_mask=has_active_mask,
        dispatch_quant_mode=dispatch_quant_mode,
        shared_experts_per_rank=shared_experts_per_rank,
        top_k=top_k,
        moe_expert_num=moe_expert_num,
        qk_rope_head_dim=qk_rope_head_dim,
        mla_variant=mla_variant,
        sfa_dcp_active=False,
        pcp_size=pcp_size,
        kv_cache_dtype=kv_cache_dtype,
        has_kpool_indexer=has_kpool_indexer,
        dcp_size=dcp_size,
        is_act_and_mul=is_act_and_mul,
        enabled_stack_passes=enabled_stack_passes,
    )
    verdict = reg.verify(
        query,
        track_spec=track_spec,
        manifest_bindings=manifest_bindings,
        engine_speedup_verified=False,
    )
    return verdict, query.canonical_digest()


def _verdict_to_tier(
    verdict: AbiSupportVerdict | None,
    *,
    fallback_tier: FusionTier,
    fallback_backend: RegionBackendTarget,
) -> tuple[FusionTier, bool, str, RegionBackendTarget]:
    if verdict is None:
        return fallback_tier, False, "soc_arch_unspecified_for_abi_probe", fallback_backend
    if verdict.status == "admitted_shipping":
        return "K1_shipping", True, "", verdict.backend_kind  # type: ignore[return-value]
    if verdict.status == "admitted_proposed_pr":
        return "K1_proposed_pr", False, "", verdict.backend_kind  # type: ignore[return-value]
    return fallback_tier, False, verdict.rejection_reason or "rejected_abi", fallback_backend


def _resolve_model_scope(ops: Sequence[OpSpec]) -> Literal["target", "draft"]:
    for op in ops:
        scope = str(dict(op.metadata).get("model_scope") or "").strip().lower()
        if scope == "draft":
            return "draft"
    return "target"


def _build_mla_prolog_ports(
    ordered_m: Sequence[OpSpec],
    role_of: Mapping[str, str],
    traits: Mapping[str, OpTraitInfo],
    *,
    soc_arch: SocArch | None,
    rope_dim: int,
) -> tuple[str, tuple[OperandPortSig, ...], tuple[OperandPortSig, ...]] | None:
    """Build independent per-projection MLA prolog operand and output port signatures."""
    entry_op = next((op for op in ordered_m if role_of[op.op_id] in {"qkv_a", "q_a"}), ordered_m[0])
    tr = traits[entry_op.op_id]
    by_role: dict[str, OpSpec] = {}
    packed_by_role: dict[str, OpSpec] = {}
    for op_m in ordered_m:
        r_name = role_of.get(op_m.op_id, "")
        if r_name and r_name not in by_role:
            by_role[r_name] = op_m
        packed_roles = dict(op_m.metadata).get("packed_weight_roles") if op_m.metadata else None
        if isinstance(packed_roles, (tuple, list, set, frozenset)):
            for pr in packed_roles:
                pr_tok = str(pr or "").strip().lower()
                if pr_tok and pr_tok not in packed_by_role:
                    packed_by_role[pr_tok] = op_m
    m_dim = int(entry_op.local_shape[0]) if entry_op.local_shape else 1
    h_dim = int(entry_op.local_shape[-1]) if len(entry_op.local_shape) >= 2 else max(1, m_dim)

    def _weight_port(port_name: str, role_keys: tuple[str, ...]) -> OperandPortSig | None:
        src_op = next((by_role[rk] for rk in role_keys if rk in by_role), None)
        if src_op is None:
            src_op = next((packed_by_role[rk] for rk in (*role_keys, port_name) if rk in packed_by_role), None)
        if src_op is None:
            return None
        w_dt, q_mode, g_sh, s_dt, w_layout = _resolve_op_weight_port_contract(
            src_op, traits[src_op.op_id],
            default_layout="ND" if port_name == "weight_uk" else "FRACTAL_NZ",
        )
        sh = (
            (int(src_op.local_shape[2]), int(src_op.local_shape[1]))
            if len(src_op.local_shape) == MATMUL_SHAPE_RANK else (h_dim, h_dim)
        )
        return _port(port_name, w_dt, sh, layout=w_layout, quant_mode=q_mode, group_shape=g_sh, scale_dtype=s_dt)

    w_dq = _weight_port("weight_dq", ("q_a", "qkv_a"))
    w_uq_qr = _weight_port("weight_uq_qr", ("q_b",))
    w_uk_raw = _weight_port("weight_uk", ("q_absorb",))
    w_dkv_kr = _weight_port("weight_dkv_kr", ("kv_a", "qkv_a"))
    if any(p is None for p in (w_dq, w_uq_qr, w_uk_raw, w_dkv_kr)):
        return None
    assert w_dq is not None and w_uq_qr is not None and w_uk_raw is not None and w_dkv_kr is not None
    uk_nope = int(w_uk_raw.shape[0]) if len(w_uk_raw.shape) >= 2 else max(1, h_dim)
    uk_kv_lora = int(w_uk_raw.shape[1]) if len(w_uk_raw.shape) >= 2 else max(1, h_dim)
    qb_out = int(w_uq_qr.shape[1]) if len(w_uq_qr.shape) >= 2 else 0
    head_span = uk_nope + max(0, int(rope_dim))
    if len(w_uk_raw.shape) == MATMUL_SHAPE_RANK:
        w_uk = w_uk_raw
        n_heads = int(w_uk.shape[0])
        kv_lora_dim = int(w_uk.shape[2])
    else:
        n_heads = max(1, qb_out // head_span) if (qb_out > 0 and head_span > 0 and qb_out % head_span == 0) else 1
        kv_lora_dim = uk_kv_lora
        w_uk = replace(w_uk_raw, shape=(n_heads, uk_nope, kv_lora_dim))
    w_dt = w_dq.dtype
    in_dt = _norm_abi_dtype(tr.input_dtype)
    iq, ig, is_dt = _quant_contract_for_port(in_dt, tr.scale_format, tr.group_size, is_weight=False)
    token_dt = "bf16" if w_dt == "fp8_e4m3fn" else in_dt
    tok_q, tok_g, tok_s = ("dense", (), "none") if token_dt == "bf16" else (iq, ig, is_dt)
    all_w_dts = {w_dq.dtype, w_uq_qr.dtype, w_uk.dtype, w_dkv_kr.dtype}
    abi_id = (
        "aclnnMlaPrologV4WeightNz"
        if (soc_arch == "A5" and "fp8_e4m3fn" in all_w_dts)
        else "aclnnMlaPrologV3WeightNz"
    )
    q_norm_op = by_role.get("q_norm")
    kv_norm_op = by_role.get("kv_norm")
    kv_store_op = by_role.get("kv_store")
    q_gamma_dt = _norm_abi_dtype(traits[q_norm_op.op_id].weight_dtype) if q_norm_op is not None else "bf16"
    kv_gamma_dt = _norm_abi_dtype(traits[kv_norm_op.op_id].weight_dtype) if kv_norm_op is not None else "bf16"
    kv_cache_dt = _norm_abi_dtype(traits[kv_store_op.op_id].output_dtype) if kv_store_op is not None else "bf16"
    q_norm_out_dt = "fp8_e4m3fn" if w_dt == "fp8_e4m3fn" else ("int8" if w_dt == "int8" else "bf16")
    q_norm_scale_dt = "fp8_e8m0" if w_dt == "fp8_e4m3fn" else "fp32"
    if q_norm_op is not None and q_norm_op.local_shape:
        q_lora_dim = max(1, int(q_norm_op.local_shape[-1]))
    elif len(w_uq_qr.shape) >= 2 and int(w_uq_qr.shape[0]) > 0:
        q_lora_dim = int(w_uq_qr.shape[0])
    elif len(w_dq.shape) >= 2 and int(w_dq.shape[1]) > 0:
        q_lora_dim = int(w_dq.shape[1])
    else:
        q_lora_dim = h_dim
    kv_meta = dict(kv_store_op.metadata) if (kv_store_op is not None and kv_store_op.metadata) else {}
    cache_mode = str(kv_meta.get("cache_mode") or "").strip().upper()
    n_kv = max(1, int(kv_meta.get("kv_heads") or kv_meta.get("num_kv_heads") or 1))
    block_num = int(kv_meta.get("num_blocks") or kv_meta.get("block_num") or 0)
    block_size = int(kv_meta.get("block_size") or 0)
    batch_size = int(kv_meta.get("batch_size") or 0)
    cache_seq = int(kv_meta.get("cache_seq_len") or kv_meta.get("max_seq_len") or kv_meta.get("seq_len") or 0)
    if cache_mode.startswith("PA_") or (block_num > 0 and block_size > 0):
        eff_bsz = block_size if block_size > 0 else 1
        eff_bnum = block_num if block_num > 0 else max(1, (m_dim + eff_bsz - 1) // eff_bsz)
        kv_cache_shape: tuple[int, ...] = (eff_bnum, eff_bsz, n_kv, kv_lora_dim)
        kr_cache_shape: tuple[int, ...] = (eff_bnum, eff_bsz, n_kv, rope_dim)
    elif cache_mode == "BSND" or (batch_size > 0 and cache_seq > 0):
        eff_b = batch_size if batch_size > 0 else 1
        eff_s = cache_seq if cache_seq > 0 else max(1, (m_dim + eff_b - 1) // eff_b)
        kv_cache_shape = (eff_b, eff_s, n_kv, kv_lora_dim)
        kr_cache_shape = (eff_b, eff_s, n_kv, rope_dim)
    else:
        cache_tokens = max(
            m_dim,
            int(kv_store_op.local_shape[0]) if (kv_store_op is not None and kv_store_op.local_shape) else m_dim,
        )
        kv_cache_shape = (cache_tokens, n_kv, kv_lora_dim)
        kr_cache_shape = (cache_tokens, n_kv, rope_dim)
    operands = (
        _port("token_x", token_dt, (m_dim, h_dim), quant_mode=tok_q, group_shape=tok_g, scale_dtype=tok_s),
        w_dq, w_uq_qr, w_uk, w_dkv_kr,
        _port("rmsnorm_gamma_cq", q_gamma_dt, (q_lora_dim,)),
        _port("rmsnorm_gamma_ckv", kv_gamma_dt, (kv_lora_dim,)),
        _port("rope_sin", "bf16", (m_dim, rope_dim)),
        _port("rope_cos", "bf16", (m_dim, rope_dim)),
        _port("kv_cache_ref", kv_cache_dt, kv_cache_shape),
        _port("kr_cache_ref", "bf16", kr_cache_shape),
    )
    outputs = (
        _port("query_out", "bf16", (m_dim, n_heads, kv_lora_dim)),
        _port("query_rope_out", "bf16", (m_dim, n_heads, rope_dim)),
        _port("query_norm_out", q_norm_out_dt, (m_dim, q_lora_dim)),
        _port("dequant_scale_q_norm_out", q_norm_scale_dt, (m_dim, 1)),
    )
    return abi_id, operands, outputs


def _match_law4_mla_prolog(
    ordered: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
    *,
    soc_arch: SocArch | None = None,
    sku_id: str = "",
    pr_track: StackTrack = "shipping",
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    registry: ProbedAbiRegistry | None = None,
) -> list[AlgebraicCandidate]:
    """Match shipping ``aclnnMlaPrologV3/V4WeightNz`` strictly via ``ProbedAbiRegistry``."""
    if pr_track == "eager":
        return []
    by_id = {op.op_id: op for op in ordered}
    role_of = {op.op_id: _typed_mla_role(op) for op in ordered}
    domain_ops: dict[tuple[Any, ...], list[OpSpec]] = defaultdict(list)
    by_domain: dict[tuple[Any, ...], list[OpSpec]] = defaultdict(list)
    for op in ordered:
        if op.op_id in claimed:
            continue
        scope = str(dict(op.metadata).get("model_scope") or "target").strip().lower()
        dom = (op.stage, op.phase, op.stream or "main", op.layer, scope)
        domain_ops[dom].append(op)
        if role_of[op.op_id]:
            by_domain[dom].append(op)

    _interior_mla_fams = frozenset({
        "rmsnorm", "rotary", "cache_store", "kv_cache_write", "kv_compress_epilog",
    })
    found: list[AlgebraicCandidate] = []
    for dom, members in by_domain.items():
        m_ids = {op.op_id for op in members}
        # Include intervening rmsnorm/rotary/kv_store ops in the same domain fed by MLA anchors.
        changed = True
        while changed:
            changed = False
            for cand in domain_ops[dom]:
                if cand.op_id in m_ids:
                    continue
                cfam = str(cand.family or "").strip().lower()
                if cfam in _interior_mla_fams and any(dep in m_ids for dep in cand.deps):
                    m_ids.add(cand.op_id)
                    if not role_of[cand.op_id]:
                        role_of[cand.op_id] = "rotary" if cfam == "rotary" else (
                            "kv_store" if cfam in {"cache_store", "kv_cache_write", "kv_compress_epilog"} else "q_norm"
                        )
                    changed = True
        roles = {role_of[oid] for oid in m_ids}
        has_entry = "qkv_a" in roles or ({"q_a", "kv_a"} <= roles)
        has_tail = bool({"q_b", "q_absorb", "rotary", "kv_store"} & roles)
        if not has_entry or not has_tail or len(m_ids) < MIN_MLA_PROLOG_MEMBERS:
            continue
        if not _convex(ordered, m_ids):
            continue
        ordered_m = [op for op in ordered if op.op_id in m_ids]
        metas = [dict(op.metadata) for op in ordered_m]
        if any(str(m.get("mla_variant") or "").lower() == "latent_mqa" for m in metas):
            continue
        rope_dims = [int(m["qk_rope_head_dim"]) for m in metas if isinstance(m.get("qk_rope_head_dim"), int)]
        if rope_dims:
            rope_dim = rope_dims[0]
        else:
            rotary_ops = [op for op in ordered_m if role_of.get(op.op_id) == "rotary" and op.local_shape]
            rope_dim = int(rotary_ops[0].local_shape[-1]) if rotary_ops else DEFAULT_MLA_ROPE_HEAD_DIM
        if rope_dim <= 0:
            continue
        if any(bool(m.get("sfa_c8_w4a8_bypass")) for m in metas):
            continue
        built = _build_mla_prolog_ports(
            ordered_m, role_of, traits, soc_arch=soc_arch, rope_dim=rope_dim,
        )
        if built is None:
            continue
        abi_id, operands, outputs = built
        verdict: AbiSupportVerdict | None = None
        q_digest = ""
        if soc_arch is not None:
            verdict, q_digest = _verify_abi_query(
                abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id, graph_motif="mla_prolog",
                operands=operands, outputs=outputs, pr_track=pr_track, track_spec=track_spec,
                manifest_bindings=manifest_bindings, enabled_stack_passes=enabled_stack_passes,
                registry=registry, model_scope=_resolve_model_scope(ordered_m),
                qk_rope_head_dim=rope_dim, mla_variant="latent_mla",
            )
        tier, shipping, unsup, backend = _verdict_to_tier(
            verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
        )
        if not shipping and pr_track != "target":
            continue
        ext_preserve = tuple(
            op.op_id for op in ordered_m[:-1]
            if any(u not in m_ids for u in succ.get(op.op_id, ()) if u in by_id)
        )
        ids_tup = tuple(op.op_id for op in ordered_m)
        found.append(AlgebraicCandidate(
            candidate_id=f"law4_mla_prolog_{ordered_m[-1].op_id}",
            law="law4_view_or_abi",
            abi_id=abi_id,
            fusion_tier=tier,
            member_ids=ids_tup,
            shipping_supported=shipping,
            preserve_port_ids=ext_preserve,
            unsupported_reason=unsup,
            avoidable_hbm_bytes=_avoidable_hbm_bytes(ordered_m, traits),
            traits=tuple(traits[oid].trait.value for oid in ids_tup),
            abi_verdict=verdict,
            abi_query_digest=q_digest,
            backend_target=backend,
            fused_recipe="mla_prolog",
        ))
    return found


def _build_mc2_gemm_verdict(
    abi_id: str,
    mm_op: OpSpec,
    comm_op: OpSpec,
    tr_mm: OpTraitInfo,
    *,
    soc_arch: SocArch | None,
    sku_id: str,
    ws: int,
    pr_track: StackTrack,
    track_spec: StackTrackSpec | None,
    manifest_bindings: tuple[ManifestBinding, ...],
    enabled_stack_passes: frozenset[str],
    registry: ProbedAbiRegistry | None,
    is_agmm: bool,
) -> tuple[AbiSupportVerdict | None, str]:
    if soc_arch is None or len(mm_op.local_shape) != MATMUL_SHAPE_RANK:
        return None, ""
    m_dim, n_dim, k_dim = (int(mm_op.local_shape[0]), int(mm_op.local_shape[1]), int(mm_op.local_shape[2]))
    in_dt, iq, ig, is_dt = _resolve_op_act_port_contract(mm_op, tr_mm)
    w_dt, wq, wg, ws_dt, w_layout = _resolve_op_weight_port_contract(mm_op, tr_mm, default_layout="ND")
    p_abi = getattr(mm_op, "precision_abi", None)
    out_dt = _norm_abi_dtype(
        getattr(p_abi, "output_dtype", "") if (p_abi is not None and getattr(p_abi, "output_dtype", ""))
        else tr_mm.output_dtype
    )
    comm_m = int(comm_op.local_shape[0]) if comm_op.local_shape else 0
    if is_agmm:
        if comm_m > 0:
            x1_m = comm_m
            out_m = m_dim
        elif m_dim > 0 and m_dim % ws == 0:
            x1_m = m_dim // ws
            out_m = m_dim
        else:
            x1_m = m_dim
            out_m = m_dim
    else:
        x1_m = m_dim
        out_m = comm_m if comm_m > 0 else (m_dim // ws if m_dim % ws == 0 else m_dim)
    operands = (
        _port("x1", in_dt, (x1_m, k_dim), layout="ND", quant_mode=iq, group_shape=ig, scale_dtype=is_dt),
        _port("x2", w_dt, (k_dim, n_dim), layout=w_layout, quant_mode=wq, group_shape=wg, scale_dtype=ws_dt),
    )
    outputs = (_port("y", out_dt, (out_m, n_dim)),)
    meta = dict(comm_op.metadata)
    placement = _resolve_placement(soc_arch, ws, meta)
    axis = str(meta.get("group_axis") or tr_mm.collective_axis or "tp").strip().lower() or "tp"
    has_bias = bool(dict(mm_op.metadata).get("has_bias", False))
    raw_comm_dt = str(meta.get("comm_dtype") or comm_op.dtype or "").strip()
    comm_dt = _norm_abi_dtype(raw_comm_dt) if raw_comm_dt else (in_dt if is_agmm else out_dt)
    return _verify_abi_query(
        abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id,
        graph_motif="all_gather_matmul" if is_agmm else "matmul_reduce_scatter",
        operands=operands, outputs=outputs, pr_track=pr_track, track_spec=track_spec,
        manifest_bindings=manifest_bindings, enabled_stack_passes=enabled_stack_passes,
        registry=registry, model_scope=_resolve_model_scope((comm_op, mm_op)),
        rank_placement=placement, world_size=ws, comm_group_id=axis,
        comm_dtype=comm_dt,
        reduction_kind="concat" if is_agmm else "sum",
        has_bias=has_bias,
    )


def _discover_law3_mc2(
    ordered: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
    *,
    allow_mc2: bool,
    soc_arch: SocArch | None = None,
    sku_id: str = "",
    tp_size: int | None = None,
    pr_track: StackTrack = "shipping",
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    include_abi_extensions: bool = True,
    registry: ProbedAbiRegistry | None = None,
) -> list[AlgebraicCandidate]:
    """Law 3: Bilinear MatMul x Associative Collective Pipelining (AGMM / MMRS) via ``ProbedAbiRegistry``."""
    del deps
    if not allow_mc2 or pr_track == "eager":
        return []
    by_id = {op.op_id: op for op in ordered}
    found: list[AlgebraicCandidate] = []
    for op in ordered:
        if op.op_id in claimed:
            continue
        tr = traits[op.op_id]
        if tr.trait == AlgebraicTrait.ASSOCIATIVE_COLLECTIVE and tr.collective_op == "CONCAT":
            users = [by_id[u] for u in succ.get(op.op_id, ()) if u in by_id and u not in claimed]
            if len(users) != 1 or not _same_domain(op, users[0]):
                continue
            mid = users[0]
            mid_tr = traits[mid.op_id]
            members: list[OpSpec] = [op, mid]
            mm_op: OpSpec | None = mid if (mid_tr.trait == AlgebraicTrait.BILINEAR_MATMUL and not mid_tr.is_grouped_matmul) else None
            # Law 3 AGMM 3-op pattern: AllGather -> Dynamic[Mx]Quant -> QuantMatmul
            if mm_op is None and mid_tr.symbolic_op_kind in {"dynamic_quant", "dynamic_mx_quant"}:
                q_users = [by_id[u] for u in succ.get(mid.op_id, ()) if u in by_id and u not in claimed]
                if len(q_users) == 1 and _same_domain(mid, q_users[0]):
                    cand_mm = q_users[0]
                    cand_tr = traits[cand_mm.op_id]
                    if cand_tr.trait == AlgebraicTrait.BILINEAR_MATMUL and not cand_tr.is_grouped_matmul:
                        mm_op = cand_mm
                        members.append(mm_op)
            if mm_op is None or not _convex(ordered, {m.op_id for m in members}):
                continue
            ws = _resolve_world_size(op, tp_size, collective=True)
            verdict, q_digest = _build_mc2_gemm_verdict(
                "aclnnAllGatherMatmulV2", mm_op, op, traits[mm_op.op_id],
                soc_arch=soc_arch, sku_id=sku_id, ws=ws, pr_track=pr_track,
                track_spec=track_spec, manifest_bindings=manifest_bindings,
                enabled_stack_passes=enabled_stack_passes, registry=registry, is_agmm=True,
            )
            tier, shipping, unsup, backend = _verdict_to_tier(
                verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
            )
            if not shipping and verdict is not None and verdict.status != "admitted_proposed_pr":
                if not (include_abi_extensions and pr_track == "target"):
                    continue
            ids = tuple(m.op_id for m in members)
            found.append(AlgebraicCandidate(
                candidate_id=f"law3_agmm_{mm_op.op_id}",
                law="law3_mc2",
                abi_id="aclnnAllGatherMatmulV2",
                fusion_tier=tier,
                member_ids=ids,
                shipping_supported=shipping,
                unsupported_reason=unsup,
                avoidable_hbm_bytes=_avoidable_hbm_bytes(members, traits),
                traits=tuple(traits[oid].trait.value for oid in ids),
                abi_verdict=verdict,
                abi_query_digest=q_digest,
                backend_target=backend,
                fused_recipe="all_gather_matmul",
                requires_comm_consensus=True,
            ))
        elif tr.trait == AlgebraicTrait.BILINEAR_MATMUL and not tr.is_grouped_matmul:
            users = [by_id[u] for u in succ.get(op.op_id, ()) if u in by_id and u not in claimed]
            if len(users) != 1 or not _same_domain(op, users[0]):
                continue
            col = users[0]
            col_tr = traits[col.op_id]
            if col_tr.trait != AlgebraicTrait.ASSOCIATIVE_COLLECTIVE or col_tr.symbolic_op_kind != "reduce_scatter":
                continue
            members = [op, col]
            if not _convex(ordered, {m.op_id for m in members}):
                continue
            ws = _resolve_world_size(col, tp_size, collective=True)
            verdict, q_digest = _build_mc2_gemm_verdict(
                "aclnnMatmulReduceScatterV2", op, col, tr,
                soc_arch=soc_arch, sku_id=sku_id, ws=ws, pr_track=pr_track,
                track_spec=track_spec, manifest_bindings=manifest_bindings,
                enabled_stack_passes=enabled_stack_passes, registry=registry, is_agmm=False,
            )
            tier, shipping, unsup, backend = _verdict_to_tier(
                verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
            )
            if not shipping and verdict is not None and verdict.status != "admitted_proposed_pr":
                if not (include_abi_extensions and pr_track == "target"):
                    continue
            ids = tuple(m.op_id for m in members)
            found.append(AlgebraicCandidate(
                candidate_id=f"law3_mmrs_{op.op_id}",
                law="law3_mc2",
                abi_id="aclnnMatmulReduceScatterV2",
                fusion_tier=tier,
                member_ids=ids,
                shipping_supported=shipping,
                unsupported_reason=unsup,
                avoidable_hbm_bytes=_avoidable_hbm_bytes(members, traits),
                traits=tuple(traits[oid].trait.value for oid in ids),
                abi_verdict=verdict,
                abi_query_digest=q_digest,
                backend_target=backend,
                fused_recipe="matmul_reduce_scatter",
                requires_comm_consensus=True,
            ))
    return found


def _splitv_unabsorbed_bytes(
    siblings: Sequence[OpSpec],
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    succ: Mapping[str, list[str]],
) -> int:
    """Delegate SplitV materialization traffic to the single canonical ``apply._unabsorbed_splitv_cost``."""
    from .apply import _unabsorbed_splitv_cost

    del traits, succ
    splitv_hbm, _ = _unabsorbed_splitv_cost(siblings, by_id)
    return int(round(splitv_hbm))


def _verify_law2a_matmul_abi(
    siblings: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    *,
    soc_arch: SocArch | None,
    sku_id: str,
    pr_track: StackTrack,
    track_spec: StackTrackSpec | None,
    manifest_bindings: tuple[ManifestBinding, ...],
    enabled_stack_passes: frozenset[str],
    registry: ProbedAbiRegistry | None,
) -> tuple[str, AbiSupportVerdict | None, str]:
    first = siblings[0]
    tr = traits[first.op_id]
    p_abi = getattr(first, "precision_abi", None)
    in_dt, iq, ig, is_dt = _resolve_op_act_port_contract(first, tr)
    w_dt, wq, wg, ws_dt, w_layout = _resolve_op_weight_port_contract(first, tr, default_layout="ND")
    out_dt = _norm_abi_dtype(
        getattr(p_abi, "output_dtype", "") if (p_abi is not None and getattr(p_abi, "output_dtype", ""))
        else tr.output_dtype
    )
    is_quant = in_dt in _ALLOWED_QUANT_FORMATS or w_dt in _ALLOWED_QUANT_FORMATS
    abi_id = "aclnnQuantMatmulV4" if is_quant else "aclnnMatmul"
    if soc_arch is None or len(first.local_shape) != MATMUL_SHAPE_RANK:
        return abi_id, None, ""
    m_dim = int(first.local_shape[0])
    k_dim = int(first.local_shape[2])
    n_sum = sum(int(s.local_shape[1]) for s in siblings if len(s.local_shape) == MATMUL_SHAPE_RANK)
    operands = (
        _port("x1", in_dt, (m_dim, k_dim), layout="ND", quant_mode=iq, group_shape=ig, scale_dtype=is_dt),
        _port("x2", w_dt, (k_dim, n_sum), layout=w_layout, quant_mode=wq, group_shape=wg, scale_dtype=ws_dt),
    )
    outputs = (_port("y", out_dt, (m_dim, n_sum)),)
    verdict, q_digest = _verify_abi_query(
        abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id, graph_motif="sibling_block_matmul",
        operands=operands, outputs=outputs, pr_track=pr_track, track_spec=track_spec,
        manifest_bindings=manifest_bindings, enabled_stack_passes=enabled_stack_passes,
        registry=registry, model_scope=_resolve_model_scope(siblings),
    )
    return abi_id, verdict, q_digest


def _discover_law2b_matmul_epilogue(
    ordered: Sequence[OpSpec],
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    succ: Mapping[str, list[str]],
    datapath: DatapathCapabilitySpec | None,
    *,
    soc_arch: SocArch | None,
    sku_id: str = "",
    pr_track: StackTrack,
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    include_abi_extensions: bool,
    registry: ProbedAbiRegistry | None = None,
) -> list[AlgebraicCandidate]:
    """Law 2b: ``BILINEAR_MATMUL`` + executable ``<2us`` UB epilogue (excludes topk/routing)."""
    if pr_track != "target" or not include_abi_extensions:
        return []
    data_deps_map, _ = _data_dependency_graph(ordered)
    found: list[AlgebraicCandidate] = []
    for op in ordered:
        if op.op_id in claimed:
            continue
        tr = traits[op.op_id]
        if tr.trait != AlgebraicTrait.BILINEAR_MATMUL or tr.is_grouped_matmul:
            continue
        users = [u for u in succ.get(op.op_id, ()) if u in by_id]
        if len(users) != 1 or users[0] in claimed:
            continue
        ep_chain: list[OpSpec] = []
        cur_id = op.op_id
        while len(ep_chain) < MAX_LAW2B_EPILOGUE_OPS:
            nxt_ids = [u for u in succ.get(cur_id, ()) if u in by_id]
            if len(nxt_ids) != 1 or nxt_ids[0] in claimed:
                break
            nxt = by_id[nxt_ids[0]]
            nxt_tr = traits[nxt.op_id]
            if not nxt_tr.is_micro_epilogue_eligible or not _same_domain(op, nxt):
                break
            if (
                nxt_tr.trait == AlgebraicTrait.ROW_REDUCTION
                and nxt_tr.symbolic_op_kind not in {"dynamic_quant", "dynamic_mx_quant"}
            ):
                break
            if ep_chain:
                is_view = str(nxt.family or "").strip().lower() in _ZERO_COPY_VIEW_FAMILIES
                is_quant = nxt_tr.symbolic_op_kind in {"dynamic_quant", "dynamic_mx_quant"}
                if not (is_view or is_quant):
                    break
            ep_chain.append(nxt)
            cur_id = nxt.op_id
        if not ep_chain:
            continue
        members = [op, *ep_chain]
        m_ids = {m.op_id for m in members}
        if not _convex(ordered, m_ids):
            continue
        if datapath is not None:
            fits, ub_pk, ub_cap = _check_ub_chain_fits(
                ep_chain, traits, datapath, uses_simt=_chain_uses_simt(ep_chain),
            )
            if not fits:
                continue
        else:
            ub_pk, ub_cap = 0, 0
        # Preserve BF16 rounding boundaries on MatMul(BF16) -> Cast(FP32) epilogues.
        verdict: AbiSupportVerdict | None = None
        q_digest = ""
        tier: FusionTier = "K2_epilogue"
        shipping = False
        unsup = "custom_k2_matmul_ub_epilogue"
        backend: RegionBackendTarget = "tilelang"
        abi_id = "FusedMatmulUbEpilogue"
        is_direct_dual_level_mx = (
            soc_arch == "A5"
            and len(ep_chain) == 1
            and traits[ep_chain[0].op_id].symbolic_op_kind == "dynamic_mx_quant"
            and data_deps_map.get(ep_chain[0].op_id, set()) <= {op.op_id}
        )
        if is_direct_dual_level_mx:
            quant_ep = ep_chain[0]
            ep_in_dt = _norm_abi_dtype(tr.output_dtype or op.dtype)
            if len(op.local_shape) == MATMUL_SHAPE_RANK:
                m_dim, n_dim = int(op.local_shape[0]), int(op.local_shape[1])
            elif len(quant_ep.local_shape) >= 2:
                m_dim = int(prod(quant_ep.local_shape[:-1]))
                n_dim = int(quant_ep.local_shape[-1])
            else:
                m_dim, n_dim = 1, 1
            y_port, _ = _law1_quant_output_ports(
                quant_ep, traits[quant_ep.op_id], (m_dim,), n_dim,
                y_name="y", scale_name="scale2", override_y_shape=(m_dim, n_dim),
            )
            scale1_cols = max(1, (n_dim + _DUAL_LEVEL_MX_SCALE1_BLOCK - 1) // _DUAL_LEVEL_MX_SCALE1_BLOCK)
            scale2_blocks = max(1, (n_dim + MX_GROUP_SIZE - 1) // MX_GROUP_SIZE)
            scale2_cols = (
                (scale2_blocks + _MX_SCALE_PAIR_ALIGN - 1) // _MX_SCALE_PAIR_ALIGN
            ) * _MX_SCALE_PAIR_ALIGN
            sc1_port = _port("scale1", "fp32", (m_dim, scale1_cols))
            sc2_port = _port("scale2", "fp8_e8m0", (m_dim, scale2_cols))
            verdict, q_digest = _verify_abi_query(
                abi_id="catlass_epilogue_dual_level_quant_mx",
                soc_arch=soc_arch, sku_id=sku_id,
                graph_motif="matmul_dual_level_mx_epilogue",
                operands=(_port("x", ep_in_dt, (m_dim, n_dim)),),
                outputs=(y_port, sc1_port, sc2_port),
                pr_track=pr_track, track_spec=track_spec,
                manifest_bindings=manifest_bindings,
                enabled_stack_passes=enabled_stack_passes,
                registry=registry, model_scope=_resolve_model_scope(members),
            )
            tier, shipping, unsup, backend = _verdict_to_tier(
                verdict, fallback_tier="K2_epilogue", fallback_backend="catlass",
            )
            abi_id = "catlass_epilogue_dual_level_quant_mx"
        ids_tup = tuple(m.op_id for m in members)
        found.append(AlgebraicCandidate(
            candidate_id=f"law2b_ep_{op.op_id}",
            law="law2_block_matmul",
            abi_id=abi_id,
            fusion_tier=tier,
            member_ids=ids_tup,
            shipping_supported=shipping,
            unsupported_reason=unsup,
            ub_peak_bytes=ub_pk,
            ub_budget_bytes=ub_cap,
            avoidable_hbm_bytes=_avoidable_hbm_bytes(members, traits),
            traits=tuple(traits[oid].trait.value for oid in ids_tup),
            abi_verdict=verdict,
            abi_query_digest=q_digest,
            backend_target=backend,
            fused_recipe="gemm_epilogue",
        ))
    return found


def _discover_law2_block_matmul(
    ordered: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    succ: Mapping[str, list[str]],
    *,
    datapath: DatapathCapabilitySpec | None = None,
    soc_arch: SocArch | None = None,
    sku_id: str = "",
    pr_track: StackTrack = "shipping",
    epilogue_track: StackTrack | None = None,
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    include_abi_extensions: bool = True,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[list[AlgebraicCandidate], list[tuple[str, str]]]:
    """Law 2a (sibling block-MatMul) + Law 2b (``BILINEAR_MATMUL`` + ``<2us`` UB epilogue)."""
    if pr_track == "eager":
        return [], []
    by_id = {op.op_id: op for op in ordered}
    buckets: dict[Any, list[OpSpec]] = defaultdict(list)
    coarse_buckets: dict[Any, list[tuple[OpSpec, Any]]] = defaultdict(list)
    rejected: list[tuple[str, str]] = []
    for op in ordered:
        if op.op_id in claimed:
            continue
        tr = traits[op.op_id]
        if tr.trait != AlgebraicTrait.BILINEAR_MATMUL or tr.is_grouped_matmul:
            continue
        key = _law2_sibling_key(op, all_ops=by_id)
        if key is None:
            continue
        buckets[key].append(op)
        coarse_buckets[(key[0], key[1], key[2], key[-1])].append((op, key))

    for coarse_entries in coarse_buckets.values():
        if len({entry_key for _, entry_key in coarse_entries}) >= MIN_FUSE_MEMBERS:
            rejected.append((
                ",".join(entry_op.op_id for entry_op, _ in coarse_entries),
                "quant_or_sharding_contract_mismatch",
            ))

    found: list[AlgebraicCandidate] = []
    for group in buckets.values():
        if len(group) < MIN_FUSE_MEMBERS:
            continue
        # Filter out pairs with ancestor dependencies or illegal cross-precision merges.
        independent: list[OpSpec] = []
        for cand in group:
            if any(_is_ancestor(prev.op_id, cand.op_id, succ) or _is_ancestor(cand.op_id, prev.op_id, succ)
                   for prev in independent):
                rejected.append((cand.op_id, "law2_sibling_dependency"))
                continue
            independent.append(cand)
        if len(independent) < MIN_FUSE_MEMBERS:
            continue
        if has_illegal_cross_precision_merge(independent):
            rejected.append((",".join(op.op_id for op in independent), "law2_cross_precision_merge"))
            continue
        m_ids = {op.op_id for op in independent}
        if not _convex(ordered, m_ids):
            rejected.append((",".join(op.op_id for op in independent), "law2_non_convex"))
            continue
        avoid = _law2_avoidable_read_bytes(independent, traits)
        splitv_cost = _splitv_unabsorbed_bytes(independent, by_id, traits, succ)
        if splitv_cost > 0 and splitv_cost >= avoid:
            rejected.append((",".join(op.op_id for op in independent), "splitv_unabsorbed_cost_exceeds_activation_savings"))
            continue
        abi_id, verdict, q_digest = _verify_law2a_matmul_abi(
            independent, traits, soc_arch=soc_arch, sku_id=sku_id, pr_track=pr_track,
            track_spec=track_spec, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_stack_passes, registry=registry,
        )
        tier, shipping, unsup, backend = _verdict_to_tier(
            verdict, fallback_tier="K2_epilogue", fallback_backend="catlass",
        )
        if not shipping and verdict is not None and pr_track != "target":
            rejected.append((",".join(op.op_id for op in independent), unsup))
            continue
        ids_tup = tuple(op.op_id for op in independent)
        found.append(AlgebraicCandidate(
            candidate_id=f"law2_block_{independent[0].op_id}",
            law="law2_block_matmul",
            abi_id=abi_id,
            fusion_tier=tier,
            member_ids=ids_tup,
            shipping_supported=shipping,
            preserve_port_ids=ids_tup,
            unsupported_reason=unsup,
            avoidable_hbm_bytes=max(0, avoid - splitv_cost),
            traits=tuple(traits[oid].trait.value for oid in ids_tup),
            abi_verdict=verdict,
            abi_query_digest=q_digest,
            backend_target=backend,
            fused_recipe="gemm",
        ))
    found.extend(_discover_law2b_matmul_epilogue(
        ordered, by_id, traits, claimed, succ, datapath,
        soc_arch=soc_arch,
        sku_id=sku_id,
        pr_track=epilogue_track if epilogue_track is not None else pr_track,
        track_spec=track_spec,
        manifest_bindings=manifest_bindings,
        enabled_stack_passes=enabled_stack_passes,
        include_abi_extensions=include_abi_extensions,
        registry=registry,
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


def _locate_moe_dispatch_combine_frame(
    op: OpSpec,
    tail: OpSpec,
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    succ: Mapping[str, list[str]],
) -> tuple[OpSpec, OpSpec, OpSpec] | None:
    """Locate connected ``(dispatch, gmm2, combine)`` around ``op -> ... -> tail``."""
    dispatch_ops = [
        by_id[dep] for dep in op.deps
        if dep in by_id and traits[dep].symbolic_op_kind in {"dispatch", "alltoall"}
        and _same_domain(by_id[dep], op)
    ]
    if not dispatch_ops:
        return None
    disp_axes = {
        str(dict(d.metadata).get("group_axis") or traits[d.op_id].collective_axis or "").strip().lower()
        for d in dispatch_ops
    }
    disp_axes.discard("")
    if disp_axes and disp_axes != {"ep"}:
        return None
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
            return dispatch_ops[0], gmm2, comb
    return None


def _norm_moe_activation(act: str) -> str:
    low = str(act or "").strip().lower()
    if low == "situ":
        return "situglu"
    return low or "swiglu"


def _build_megamoe_ports(
    dispatch_op: OpSpec,
    gmm1: OpSpec,
    gmm2: OpSpec,
    *,
    soc_arch: SocArch,
    ws: int,
    m_dim: int,
    h_dim: int,
    inter_dim: int,
    g2_k: int,
    g2_n: int,
    tk_dim: int,
    in_dt: str,
    iq: str,
    ig: tuple[int, ...],
    is_dt: str,
    w1_dt: str,
    wq1: str,
    wg1: tuple[int, ...],
    ws1_dt: str,
    w2_dt: str,
    wq2: str,
    wg2: tuple[int, ...],
    ws2_dt: str,
    out_dt: str,
) -> tuple[tuple[OperandPortSig, ...], tuple[OperandPortSig, ...], int, bool, int]:
    """Build canonical ``aclnnMegaMoe`` operand/output ports and default dispatch/mask flags."""
    meta_disp = dict(dispatch_op.metadata)
    meta_g1 = dict(gmm1.metadata)
    num_experts = int(
        meta_disp.get("moe_expert_num")
        or meta_disp.get("num_experts")
        or meta_g1.get("num_experts")
        or meta_g1.get("moe_expert_num")
        or 0
    )
    raw_local_e = int(
        meta_disp.get("local_experts")
        or meta_g1.get("local_experts")
        or meta_g1.get("experts_per_rank")
        or (len(meta_g1.get("matrix_group_rows", ())) if isinstance(meta_g1.get("matrix_group_rows"), Sequence) else 0)
        or 0
    )
    local_experts = (
        raw_local_e if raw_local_e > 0
        else (max(1, num_experts // max(1, ws)) if num_experts > 0 else 1)
    )
    moe_expert_num = num_experts if num_experts > 0 else (local_experts * max(1, ws))
    mxfp_dts = _MX_DTYPES | {"fp8_e5m2"}
    if soc_arch == "A5":
        a5_w1_layout = _resolve_declared_weight_layout(
            gmm1, getattr(gmm1, "precision_abi", None),
            default_when_undeclared=(
                "FRACTAL_NZ" if (w1_dt == "fp4_e2m1" and in_dt == "fp4_e2m1")
                else ("FRACTAL_NZ_C0_32" if w1_dt == "fp4_e2m1"
                      else ("ND" if w1_dt in {"bf16", "fp8_e5m2"} else "FRACTAL_NZ"))
            ),
        )
        a5_w2_layout = _resolve_declared_weight_layout(
            gmm2, getattr(gmm2, "precision_abi", None),
            default_when_undeclared=(
                "FRACTAL_NZ_C0_32" if w2_dt == "fp4_e2m1"
                else ("ND" if w2_dt in {"bf16", "fp8_e5m2"} else "FRACTAL_NZ")
            ),
        )
        w1_shape: tuple[int, ...] = (local_experts, inter_dim, h_dim)
        w2_shape: tuple[int, ...] = (local_experts, g2_n, g2_k)
        w1_scale_shape: tuple[int, ...] = (
            (
                local_experts, inter_dim,
                (h_dim + _MEGAMOE_MX_SCALE_ALIGN - 1) // _MEGAMOE_MX_SCALE_ALIGN,
                _MEGAMOE_MX_SCALE_TRAILING,
            )
            if w1_dt in mxfp_dts else ()
        )
        w2_scale_shape: tuple[int, ...] = (
            (
                local_experts, g2_n,
                (g2_k + _MEGAMOE_MX_SCALE_ALIGN - 1) // _MEGAMOE_MX_SCALE_ALIGN,
                _MEGAMOE_MX_SCALE_TRAILING,
            )
            if w2_dt in mxfp_dts else ()
        )
        w1_port = replace(
            _port("weight1", w1_dt, w1_shape, layout=a5_w1_layout, quant_mode=wq1, group_shape=wg1, scale_dtype=ws1_dt),
            scale_shape=w1_scale_shape,
        )
        w2_port = replace(
            _port("weight2", w2_dt, w2_shape, layout=a5_w2_layout, quant_mode=wq2, group_shape=wg2, scale_dtype=ws2_dt),
            scale_shape=w2_scale_shape,
        )
        default_has_mask = False
        default_dq_mode = (
            _MEGAMOE_DISPATCH_MODE_MXFP
            if (w1_dt in mxfp_dts or w2_dt in mxfp_dts) and in_dt in {"bf16", "fp16"}
            else _MEGAMOE_DISPATCH_MODE_PASSTHROUGH
        )
    else:
        a23_w1_layout = _resolve_declared_weight_layout(
            gmm1, getattr(gmm1, "precision_abi", None),
            default_when_undeclared=("ND" if w1_dt in {"bf16", "fp16"} else "FRACTAL_NZ"),
        )
        a23_w2_layout = _resolve_declared_weight_layout(
            gmm2, getattr(gmm2, "precision_abi", None),
            default_when_undeclared=("ND" if w2_dt in {"bf16", "fp16"} else "FRACTAL_NZ"),
        )
        w1_port = _port("weight1", w1_dt, (h_dim, inter_dim), layout=a23_w1_layout, quant_mode=wq1, group_shape=wg1, scale_dtype=ws1_dt)
        w2_port = _port("weight2", w2_dt, (g2_k, g2_n), layout=a23_w2_layout, quant_mode=wq2, group_shape=wg2, scale_dtype=ws2_dt)
        default_has_mask = True
        default_dq_mode = (
            _MEGAMOE_DISPATCH_MODE_INT8
            if (w1_dt in {"int8", "int4"} or w2_dt in {"int8", "int4"})
            else _MEGAMOE_DISPATCH_MODE_PASSTHROUGH
        )
    x_scale_shape: tuple[int, ...] = (
        (m_dim, (h_dim + MX_GROUP_SIZE - 1) // MX_GROUP_SIZE)
        if (soc_arch == "A5" and in_dt in mxfp_dts) else ()
    )
    x_port = replace(
        _port("x", in_dt, (m_dim, h_dim), quant_mode=iq, group_shape=ig, scale_dtype=is_dt),
        scale_shape=x_scale_shape,
    )
    operands = (
        _port("context", "int32", _MEGAMOE_CONTEXT_BYTES_SHAPE, quant_mode="none"),
        x_port,
        _port("topk_ids", "int32", (m_dim, tk_dim), quant_mode="none"),
        _port("topk_weights", "fp32", (m_dim, tk_dim)),
        w1_port,
        w2_port,
    )
    outputs = (
        _port("y", out_dt, (m_dim, g2_n)),
        _port("expert_token_nums", "int32", (local_experts,), quant_mode="none"),
    )
    return operands, outputs, moe_expert_num, default_has_mask, default_dq_mode


def _verify_moe_frame_abi(
    dispatch_op: OpSpec,
    gmm1: OpSpec,
    act_op: OpSpec,
    gmm2: OpSpec,
    combine_op: OpSpec,
    traits: Mapping[str, OpTraitInfo],
    *,
    soc_arch: SocArch | None,
    sku_id: str,
    ep_size: int | None,
    enable_fused_mc2: int | None,
    pr_track: StackTrack,
    track_spec: StackTrackSpec | None,
    manifest_bindings: tuple[ManifestBinding, ...],
    enabled_stack_passes: frozenset[str],
    registry: ProbedAbiRegistry | None,
) -> tuple[str, AbiSupportVerdict | None, str]:
    tr1 = traits[gmm1.op_id]
    tr2 = traits[gmm2.op_id]
    act_norm = _norm_moe_activation(traits[act_op.op_id].activation_kind or act_op.family)
    w1_dt, wq1, wg1, ws1_dt, w1_layout = _resolve_op_weight_port_contract(gmm1, tr1, default_layout="FRACTAL_NZ")
    w2_dt, wq2, wg2, ws2_dt, w2_layout = _resolve_op_weight_port_contract(gmm2, tr2, default_layout="FRACTAL_NZ")
    in_dt, iq, ig, is_dt = _resolve_op_act_port_contract(gmm1, tr1)
    if not in_dt:
        in_dt, iq, ig, is_dt = "bf16", "dense", (), "none"
    p_abi2 = getattr(gmm2, "precision_abi", None)
    tr_comb = traits.get(combine_op.op_id)
    comb_dt = (tr_comb.output_dtype if tr_comb is not None else "") or combine_op.dtype or ""
    out_dt = _norm_abi_dtype(
        getattr(p_abi2, "output_dtype", "") if (p_abi2 is not None and getattr(p_abi2, "output_dtype", ""))
        else (comb_dt or tr2.output_dtype or "bf16")
    )
    use_csrc = (
        soc_arch == "A3"
        and (enable_fused_mc2 == 1 or (enable_fused_mc2 is None and w1_dt in {"int8", "int4"} and w2_dt in {"int8", "int4"}))
    )
    abi_id = "dispatch_ffn_combine" if use_csrc else "aclnnMegaMoe"
    if soc_arch is None:
        return abi_id, None, ""
    ws = _resolve_world_size(dispatch_op, ep_size, collective=True)
    meta_disp = dict(dispatch_op.metadata)
    placement = _resolve_placement(soc_arch, ws, meta_disp)
    if len(dispatch_op.local_shape) >= 2:
        m_dim = int(prod(dispatch_op.local_shape[:-1]))
    elif dispatch_op.local_shape:
        m_dim = int(dispatch_op.local_shape[0])
    elif len(combine_op.local_shape) >= 2:
        m_dim = int(prod(combine_op.local_shape[:-1]))
    elif combine_op.local_shape:
        m_dim = int(combine_op.local_shape[0])
    else:
        m_dim = int(gmm1.local_shape[0]) if gmm1.local_shape else 1
    h_dim = int(gmm1.local_shape[2]) if len(gmm1.local_shape) == MATMUL_SHAPE_RANK else int(
        gmm1.local_shape[-1] if gmm1.local_shape else 0
    )
    inter_dim = int(gmm1.local_shape[1]) if len(gmm1.local_shape) == MATMUL_SHAPE_RANK else 0
    g2_k = int(gmm2.local_shape[2]) if len(gmm2.local_shape) == MATMUL_SHAPE_RANK else inter_dim
    g2_n = int(gmm2.local_shape[1]) if len(gmm2.local_shape) == MATMUL_SHAPE_RANK else h_dim
    top_k = int(meta_disp.get("top_k") or dict(gmm1.metadata).get("top_k") or 0)
    tk_dim = max(1, top_k)
    moe_expert_num = 0
    default_has_mask = True
    default_dq_mode = 0
    if abi_id == "dispatch_ffn_combine":
        base_ops = [
            _port("x", in_dt, (m_dim, h_dim), quant_mode=iq, group_shape=ig, scale_dtype=is_dt),
            _port("weight1", w1_dt, (h_dim, inter_dim), layout=w1_layout, quant_mode=wq1, group_shape=wg1, scale_dtype=ws1_dt),
            _port("weight2", w2_dt, (g2_k, g2_n), layout=w2_layout, quant_mode=wq2, group_shape=wg2, scale_dtype=ws2_dt),
            _port("expert_idx", "int32", (m_dim, tk_dim), quant_mode="none"),
            _port("scale1", "fp32", (1,)),
            _port("scale2", "fp32", (1,)),
        ]
        if w1_dt == "int4" or w2_dt == "int4":
            base_ops.extend((_port("bias1", "fp32", (1,)), _port("bias2", "fp32", (1,))))
        base_ops.append(_port("probs", "fp32", (m_dim, tk_dim)))
        operands = tuple(base_ops)
        outputs = (_port("out", out_dt, (m_dim, g2_n)), _port("expert_token_nums", "int64", (1,)))
    else:
        operands, outputs, moe_expert_num, default_has_mask, default_dq_mode = _build_megamoe_ports(
            dispatch_op, gmm1, gmm2,
            soc_arch=soc_arch, ws=ws, m_dim=m_dim, h_dim=h_dim, inter_dim=inter_dim,
            g2_k=g2_k, g2_n=g2_n, tk_dim=tk_dim,
            in_dt=in_dt, iq=iq, ig=ig, is_dt=is_dt,
            w1_dt=w1_dt, wq1=wq1, wg1=wg1, ws1_dt=ws1_dt,
            w2_dt=w2_dt, wq2=wq2, wg2=wg2, ws2_dt=ws2_dt, out_dt=out_dt,
        )
    shared = int(meta_disp.get("shared_experts_per_rank") or 0)
    comm_dt_raw = str(meta_disp.get("comm_dtype") or dispatch_op.dtype or "").strip()
    comm_dt = _norm_abi_dtype(comm_dt_raw) if comm_dt_raw else in_dt
    has_mask = bool(meta_disp.get("has_active_mask", True if abi_id == "dispatch_ffn_combine" else default_has_mask))
    dq_mode = int(meta_disp.get("dispatch_quant_mode", 0 if abi_id == "dispatch_ffn_combine" else default_dq_mode))
    has_bias = bool(meta_disp.get("has_bias", dict(gmm1.metadata).get("has_bias", False)))
    verdict, q_digest = _verify_abi_query(
        abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id, graph_motif="moe_mc2_frame",
        operands=operands, outputs=outputs, pr_track=pr_track, track_spec=track_spec,
        manifest_bindings=manifest_bindings, enabled_stack_passes=enabled_stack_passes,
        registry=registry, model_scope=_resolve_model_scope((dispatch_op, gmm1, gmm2, combine_op)),
        rank_placement=placement, world_size=ws, comm_group_id="ep",
        comm_dtype=comm_dt, reduction_kind="permute", activation_kind=act_norm,
        has_active_mask=has_mask,
        dispatch_quant_mode=dq_mode,
        has_bias=has_bias,
        shared_experts_per_rank=shared, top_k=top_k,
        moe_expert_num=(0 if abi_id == "dispatch_ffn_combine" else moe_expert_num),
        is_act_and_mul=(act_norm not in {"relu2", "gelu"}),
    )
    return abi_id, verdict, q_digest


def _verify_gmm_act_quant_abi(
    gmm_op: OpSpec,
    act_op: OpSpec,
    quant_op: OpSpec | None,
    traits: Mapping[str, OpTraitInfo],
    *,
    soc_arch: SocArch | None,
    sku_id: str,
    pr_track: StackTrack,
    track_spec: StackTrackSpec | None,
    manifest_bindings: tuple[ManifestBinding, ...],
    enabled_stack_passes: frozenset[str],
    registry: ProbedAbiRegistry | None,
) -> tuple[str, AbiSupportVerdict | None, str]:
    tr_g = traits[gmm_op.op_id]
    act_norm = _norm_moe_activation(traits[act_op.op_id].activation_kind or act_op.family)
    if quant_op is None:
        return "FusedGroupedMatmulActivation", None, ""
    abi_id = "aclnnGroupedMatmulSwigluQuantV2"
    if soc_arch is None or len(gmm_op.local_shape) != MATMUL_SHAPE_RANK:
        return abi_id, None, ""
    m_dim, n_dim, k_dim = (int(gmm_op.local_shape[0]), int(gmm_op.local_shape[1]), int(gmm_op.local_shape[2]))
    in_dt, iq, ig, is_dt = _resolve_op_act_port_contract(gmm_op, tr_g)
    w_dt, wq, wg, ws_dt, w_layout = _resolve_op_weight_port_contract(
        gmm_op, tr_g, default_layout="ND",
    )
    is_act_and_mul = act_norm not in {"relu2", "gelu"}
    if is_act_and_mul and n_dim % _SWIGLU_WIDTH_DIVISOR != 0:
        return abi_id, None, ""
    expected_y_n = n_dim // _SWIGLU_WIDTH_DIVISOR if is_act_and_mul else n_dim
    y_m = int(quant_op.local_shape[0]) if quant_op.local_shape else m_dim
    y_n = (
        int(quant_op.local_shape[-1])
        if (len(quant_op.local_shape) >= 2 and int(quant_op.local_shape[-1]) == expected_y_n)
        else expected_y_n
    )
    y_port, sc_port = _law1_quant_output_ports(
        quant_op, traits[quant_op.op_id], (y_m,), y_n,
        y_name="y", scale_name="y_scale", override_y_shape=(y_m, y_n),
    )
    operands = (
        _port("x", in_dt, (m_dim, k_dim), layout="ND", quant_mode=iq, group_shape=ig, scale_dtype=is_dt),
        _port("w", w_dt, (k_dim, n_dim), layout=w_layout, quant_mode=wq, group_shape=wg, scale_dtype=ws_dt),
    )
    outputs = (y_port, sc_port)
    verdict, q_digest = _verify_abi_query(
        abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id, graph_motif="gmm_act_quant",
        operands=operands, outputs=outputs, pr_track=pr_track, track_spec=track_spec,
        manifest_bindings=manifest_bindings, enabled_stack_passes=enabled_stack_passes,
        registry=registry, model_scope=_resolve_model_scope((gmm_op, act_op, quant_op)),
        activation_kind=act_norm, is_act_and_mul=is_act_and_mul,
    )
    return abi_id, verdict, q_digest


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
    soc_arch: SocArch | None = None,
    sku_id: str = "",
    ep_size: int | None = None,
    allow_mc2: bool = True,
    pr_track: StackTrack = "shipping",
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    registry: ProbedAbiRegistry | None = None,
) -> list[AlgebraicCandidate]:
    """Match MoE MC2 frames and ``GroupedMatmul -> Activation [-> Quant]`` via ``ProbedAbiRegistry``."""
    del datapath
    if pr_track == "eager":
        return []
    eff_arch = soc_arch or _resolve_soc_arch(arch_family, None, sku_id)
    found: list[AlgebraicCandidate] = []
    for op in ordered:
        if op.op_id in claimed or not traits[op.op_id].is_grouped_matmul:
            continue
        users = [by_id[u] for u in succ.get(op.op_id, ()) if u in by_id and u not in claimed]
        if len(users) != 1 or not _same_domain(op, users[0]):
            continue
        act_op = users[0]
        act_tr = traits[act_op.op_id]
        if not act_tr.activation_kind and act_tr.symbolic_op_kind not in {"swiglu", "situ", "gelu", "silu"}:
            continue
        quant_op: OpSpec | None = None
        tail = act_op
        act_users = [by_id[u] for u in succ.get(act_op.op_id, ()) if u in by_id and u not in claimed]
        if len(act_users) == 1 and _same_domain(act_op, act_users[0]):
            if traits[act_users[0].op_id].symbolic_op_kind in {"dynamic_quant", "dynamic_mx_quant"}:
                quant_op = act_users[0]
                tail = quant_op

        # 1) Full MoE MC2 frame (dispatch -> GMM1 -> act [-> quant] -> GMM2 -> combine)
        frame = _locate_moe_dispatch_combine_frame(op, tail, by_id, traits, succ) if allow_mc2 and enable_fused_mc2 != 0 else None
        if frame is not None:
            disp, gmm2, comb = frame
            frame_ops = [disp, op, act_op, *([quant_op] if quant_op is not None else []), gmm2, comb]
            if _convex(ordered, {m.op_id for m in frame_ops}):
                abi_id, verdict, q_digest = _verify_moe_frame_abi(
                    disp, op, act_op, gmm2, comb, traits,
                    soc_arch=eff_arch, sku_id=sku_id, ep_size=ep_size,
                    enable_fused_mc2=enable_fused_mc2, pr_track=pr_track,
                    track_spec=track_spec, manifest_bindings=manifest_bindings,
                    enabled_stack_passes=enabled_stack_passes, registry=registry,
                )
                tier, shipping, unsup, backend = _verdict_to_tier(
                    verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
                )
                if not shipping:
                    act_tag = _norm_moe_activation(act_tr.activation_kind or act_op.family)
                    _append_audit_trigger(
                        triggers, trigger_keys, traits, "ABI_UNSUPPORTED_VARIANT", frame_ops,
                        unsup or f"{abi_id} does not admit activation={act_tag} on arch={eff_arch}",
                        cand_abi=f"{abi_id}_{act_tag}_ext",
                    )
                if shipping or (verdict is not None and verdict.status == "admitted_proposed_pr") or (
                    include_abi_extensions and pr_track == "target"
                ):
                    ids_tup = tuple(m.op_id for m in frame_ops)
                    found.append(AlgebraicCandidate(
                        candidate_id=f"law3_moe_{op.op_id}",
                        law="law3_mc2",
                        abi_id=abi_id,
                        fusion_tier=tier,
                        member_ids=ids_tup,
                        shipping_supported=shipping,
                        unsupported_reason=unsup,
                        avoidable_hbm_bytes=_avoidable_hbm_bytes(frame_ops, traits),
                        traits=tuple(traits[oid].trait.value for oid in ids_tup),
                        abi_verdict=verdict,
                        abi_query_digest=q_digest,
                        backend_target=backend,
                        fused_recipe="megamoe_arch35" if abi_id == "aclnnMegaMoe" else abi_id,
                        requires_comm_consensus=True,
                    ))

        # 2) Inner GMM -> Act [-> Quant] candidate (competes with MoE frame in conflict graph!)
        inner_ops = [op, act_op, *([quant_op] if quant_op is not None else [])]
        if not _convex(ordered, {m.op_id for m in inner_ops}):
            continue
        abi_id, verdict, q_digest = _verify_gmm_act_quant_abi(
            op, act_op, quant_op, traits,
            soc_arch=eff_arch, sku_id=sku_id, pr_track=pr_track,
            track_spec=track_spec, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_stack_passes, registry=registry,
        )
        tier, shipping, unsup, backend = _verdict_to_tier(
            verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
        )
        if not shipping and quant_op is not None:
            _append_audit_trigger(
                triggers, trigger_keys, traits, "ABI_UNSUPPORTED_VARIANT", inner_ops,
                unsup or f"{abi_id} rejected by ProbedAbiRegistry on arch={eff_arch}",
                cand_abi=abi_id,
            )
        if shipping or (verdict is not None and verdict.status == "admitted_proposed_pr") or (
            include_abi_extensions and pr_track == "target"
        ):
            ids_tup = tuple(m.op_id for m in inner_ops)
            found.append(AlgebraicCandidate(
                candidate_id=f"law2_gmm_act_{op.op_id}",
                law="law1_ub_chain",
                abi_id=abi_id,
                fusion_tier=tier,
                member_ids=ids_tup,
                shipping_supported=shipping,
                unsupported_reason=unsup,
                avoidable_hbm_bytes=_avoidable_hbm_bytes(inner_ops, traits),
                traits=tuple(traits[oid].trait.value for oid in ids_tup),
                abi_verdict=verdict,
                abi_query_digest=q_digest,
                backend_target=backend,
                fused_recipe="grouped_matmul_swiglu_quant" if shipping else "custom_gmm_epilogue",
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
        if not tr.is_stateful or tr.is_terminal_sink:
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


def _terminal_sink_admissible(
    cur: OpSpec,
    sink: OpSpec,
    chain: Sequence[OpSpec],
    by_id: Mapping[str, OpSpec],
    succ: Mapping[str, list[str]],
) -> bool:
    """Enforce Law 1's 6-condition terminal cache-quant sink admission contract."""
    from mindie_sim.prediction.fusion.ports import input_port_entries
    from .apply import _named_inputs, _typed_inputs

    # Condition 1: cur must feed a data operand (never slot_mapping/cache/cache_scale/x_scale).
    matched_roles: list[str] = []
    saw_declared_ports = False
    for _, port in _typed_inputs(sink):
        saw_declared_ports = True
        prod = str(port.get("producer") or "").strip()
        if prod == cur.op_id:
            role = str(
                port.get("role") or port.get("input") or port.get("name") or port.get("port") or ""
            ).strip().lower()
            matched_roles.append(role)
    for key, prod in _named_inputs(sink).items():
        saw_declared_ports = True
        if prod == cur.op_id:
            matched_roles.append(key.removesuffix("_producer").strip().lower())
    for dep, port_name in input_port_entries(sink.metadata):
        saw_declared_ports = True
        if dep == cur.op_id:
            matched_roles.append(port_name.removeprefix(dep + ":").strip().lower())
    if matched_roles:
        if any(role in _TERMINAL_SINK_NON_DATA_ROLES for role in matched_roles):
            if not any(role and role not in _TERMINAL_SINK_NON_DATA_ROLES for role in matched_roles):
                return False
    elif saw_declared_ports:
        return False
    # Condition 3: no external alias reader along any path from chain[0] to sink.
    chain_ids = {op.op_id for op in chain} | {sink.op_id}
    sink_meta = dict(sink.metadata) if sink.metadata else {}
    cache_name = str(sink_meta.get("cache_buffer") or sink_meta.get("state_buffer") or "").strip().lower()
    if cache_name:
        for op in chain:
            for u in succ.get(op.op_id, ()):
                if u in chain_ids or u not in by_id:
                    continue
                u_meta = dict(by_id[u].metadata)
                if str(u_meta.get("cache_buffer") or u_meta.get("state_buffer") or "").strip().lower() == cache_name:
                    return False
    return True


def _discover_ub_chains(
    ordered: Sequence[OpSpec],
    by_id: Mapping[str, OpSpec],
    traits: Mapping[str, OpTraitInfo],
    claimed: set[str],
    deps: Mapping[str, set[str]],
    succ: Mapping[str, list[str]],
    datapath: DatapathCapabilitySpec | None,
    *,
    soc_arch: SocArch | None = None,
    sku_id: str = "",
    pr_track: StackTrack = "shipping",
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    include_abi_extensions: bool = True,
    registry: ProbedAbiRegistry | None = None,
) -> list[AlgebraicCandidate]:
    """Law 1 + Law 4: connected UB chains over View/Pointwise/RowReduction + terminal cache-quant sinks."""
    if pr_track == "eager":
        return []
    eligible_traits = {AlgebraicTrait.VIEW_METADATA, AlgebraicTrait.POINTWISE, AlgebraicTrait.ROW_REDUCTION}
    found: list[AlgebraicCandidate] = []
    local_seen: set[str] = set()
    for op in ordered:
        if op.op_id in claimed or op.op_id in local_seen:
            continue
        tr = traits[op.op_id]
        if tr.trait not in eligible_traits or tr.is_stateful or tr.is_opaque_custom:
            continue
        in_elig = [
            by_id[p] for p in deps.get(op.op_id, set())
            if p in by_id and p not in claimed and p not in local_seen
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
            # Check COMPLETE graph fan-out (never filter out already-claimed consumers!).
            all_users = [u for u in succ.get(cur.op_id, ()) if u in by_id]
            if len(all_users) != 1 or all_users[0] in claimed or all_users[0] in local_seen:
                break
            nxt = by_id[all_users[0]]
            nxt_tr = traits[nxt.op_id]
            if not _same_domain(cur, nxt) or nxt_tr.is_opaque_custom:
                break
            # Law 1 6-condition terminal cache-quant sink absorption (strictly terminates chain).
            if nxt_tr.is_terminal_sink:
                if _terminal_sink_admissible(cur, nxt, chain, by_id, succ):
                    proposed = [*chain, nxt]
                    fits, _, _ = _check_ub_chain_fits(
                        proposed, traits, datapath, uses_simt=_chain_uses_simt(proposed),
                    )
                    if fits and _convex(ordered, {x.op_id for x in proposed}):
                        chain = proposed
                break
            if nxt_tr.trait not in eligible_traits or nxt_tr.is_stateful:
                break
            extras = _pull_single_consumer_views(nxt, {x.op_id for x in chain}, by_id, traits, claimed | local_seen, deps, succ)
            proposed = [*chain, *extras, nxt]
            fits, _, _ = _check_ub_chain_fits(
                proposed, traits, datapath, uses_simt=_chain_uses_simt(proposed),
            )
            if not fits or not _convex(ordered, {x.op_id for x in proposed}):
                break
            chain = proposed
            cur = nxt

        head_views = _pull_single_consumer_views(chain[0], {x.op_id for x in chain}, by_id, traits, claimed | local_seen, deps, succ)
        if head_views and _convex(ordered, {x.op_id for x in (*head_views, *chain)}):
            chain = [*head_views, *chain]

        if len(chain) < MIN_FUSE_MEMBERS:
            continue
        chain_set = {c.op_id for c in chain}
        chain_ordered = [x for x in ordered if x.op_id in chain_set]
        fits, ub_pk, ub_cap = _check_ub_chain_fits(
            chain_ordered, traits, datapath, uses_simt=_chain_uses_simt(chain_ordered),
        )
        if not fits:
            continue
        abi_id, tier, shipping, unsup, verdict, q_digest, backend = _classify_ub_chain_abi(
            chain_ordered, traits,
            soc_arch=soc_arch, sku_id=sku_id, pr_track=pr_track,
            track_spec=track_spec, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_stack_passes, registry=registry,
        )
        if not shipping and verdict is not None and verdict.status != "admitted_proposed_pr":
            if not (include_abi_extensions and pr_track == "target"):
                continue
        if not shipping and verdict is None and not (include_abi_extensions and pr_track == "target"):
            continue
        m_ids = tuple(x.op_id for x in chain_ordered)
        elided = tuple(
            x.op_id for x in chain_ordered
            if traits[x.op_id].trait == AlgebraicTrait.VIEW_METADATA
            and str(x.family or "").strip().lower() in _ZERO_COPY_VIEW_FAMILIES
        )
        ext_preserve = tuple(
            x.op_id for x in chain_ordered[:-1]
            if any(u not in chain_set for u in succ.get(x.op_id, ()) if u in by_id)
        )
        local_seen.update(m_ids)
        found.append(AlgebraicCandidate(
            candidate_id=f"law1_ub_{chain_ordered[-1].op_id}",
            law="law4_view_or_abi" if len(elided) == len(chain_ordered) else "law1_ub_chain",
            abi_id=abi_id,
            fusion_tier=tier,
            member_ids=m_ids,
            shipping_supported=shipping,
            preserve_port_ids=ext_preserve,
            unsupported_reason=unsup,
            elided_view_ids=elided,
            ub_peak_bytes=ub_pk,
            ub_budget_bytes=ub_cap,
            avoidable_hbm_bytes=_avoidable_hbm_bytes(chain_ordered, traits),
            traits=tuple(traits[oid].trait.value for oid in m_ids),
            abi_verdict=verdict,
            abi_query_digest=q_digest,
            backend_target=backend,
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
    soc_arch: SocArch | None = None,
    sku_id: str = "",
    ep_size: int | None = None,
    allow_mc2: bool = True,
    pr_track: StackTrack = "shipping",
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[list[AlgebraicCandidate], list[SemanticAuditTrigger]]:
    """Law 1 (UB pointwise/reduction chains + terminal sinks) + Law 4 (view elision & ABI match)."""
    by_id = {op.op_id: op for op in ordered}
    triggers: list[SemanticAuditTrigger] = []
    trigger_keys: set[tuple[str, tuple[str, ...]]] = set()
    found = _discover_gmm_act_candidates(
        ordered, by_id, traits, claimed, succ, datapath, triggers, trigger_keys,
        include_abi_extensions=include_abi_extensions,
        arch_family=arch_family,
        enable_fused_mc2=enable_fused_mc2,
        soc_arch=soc_arch,
        sku_id=sku_id,
        ep_size=ep_size,
        allow_mc2=allow_mc2,
        pr_track=pr_track,
        track_spec=track_spec,
        manifest_bindings=manifest_bindings,
        enabled_stack_passes=enabled_stack_passes,
        registry=registry,
    )
    _collect_semantic_audit_triggers(ordered, by_id, traits, deps, succ, triggers, trigger_keys)
    found.extend(_discover_ub_chains(
        ordered, by_id, traits, claimed, deps, succ, datapath,
        soc_arch=soc_arch, sku_id=sku_id, pr_track=pr_track,
        track_spec=track_spec, manifest_bindings=manifest_bindings,
        enabled_stack_passes=enabled_stack_passes,
        include_abi_extensions=include_abi_extensions, registry=registry,
    ))
    return found, triggers


def _has_active_smooth_scale2(chain: Sequence[OpSpec]) -> bool:
    """Return True iff any op in a fused AddRmsNormDynamicQuant chain supplies smooth_scale2."""
    from .apply import _named_inputs, _typed_inputs

    for op in chain:
        meta = dict(op.metadata) if op.metadata else {}
        if any(bool(meta.get(k)) for k in ("smooth_scale2", "has_smooth_scale2", "smooth_scales2")):
            return True
        if any(k.strip().lower() in {"smooth_scale2", "smooth_scale2_optional"} for k in _named_inputs(op)):
            return True
        for _, port in _typed_inputs(op):
            if str(port.get("role") or "").strip().lower() in {"smooth_scale2", "smooth_scale2_optional"}:
                return True
    return False


def _law1_quant_output_ports(
    quant_op: OpSpec,
    tr_q: OpTraitInfo,
    lead_dims: tuple[int, ...] | int,
    n_dim: int,
    *,
    y_name: str,
    scale_name: str,
    override_y_shape: tuple[int, ...] | None = None,
) -> tuple[OperandPortSig, OperandPortSig]:
    p_abi = getattr(quant_op, "precision_abi", None)
    meta = dict(quant_op.metadata) if quant_op.metadata else {}
    is_mx_op = tr_q.symbolic_op_kind == "dynamic_mx_quant"
    raw_out = (
        getattr(p_abi, "output_dtype", "") if (p_abi is not None and getattr(p_abi, "output_dtype", ""))
        else (meta.get("output_dtype") or meta.get("quant_dtype") or tr_q.output_dtype or "")
    )
    out_dt = _norm_abi_dtype(str(raw_out)) if raw_out else ("fp8_e4m3fn" if is_mx_op else "int8")
    if is_mx_op and out_dt in {"bf16", "fp16", "fp32"}:
        out_dt = "fp8_e4m3fn"
    raw_sf = str(
        (getattr(p_abi, "act_scale_dtype", "") if p_abi is not None else "")
        or meta.get("scale_dtype")
        or meta.get("quant_scale_dtype")
        or tr_q.scale_format
        or ""
    ).strip().lower()
    abi_gs = int(getattr(p_abi, "act_group_size", 0) or 0) if p_abi is not None else 0
    raw_gs_val = meta.get("quant_group_size", meta.get("group_size"))
    raw_gs = (
        abi_gs if abi_gs > 0
        else (int(raw_gs_val) if isinstance(raw_gs_val, int) and not isinstance(raw_gs_val, bool) and raw_gs_val > 0 else tr_q.group_size)
    )
    q_mode, g_shape, s_dt = _quant_contract_for_port(out_dt, raw_sf, raw_gs, is_weight=False)
    is_mx_scale = is_mx_op or out_dt in _MX_DTYPES or s_dt == "fp8_e8m0"
    if s_dt == "none":
        s_dt = "fp8_e8m0" if is_mx_scale else "fp32"
    fallback_lead = (int(lead_dims),) if isinstance(lead_dims, int) else (tuple(int(d) for d in lead_dims) or (1,))
    if override_y_shape is not None and len(override_y_shape) >= 2:
        out_lead = tuple(int(d) for d in override_y_shape[:-1])
        out_n = int(override_y_shape[-1])
    elif len(quant_op.local_shape) >= 2:
        out_lead = tuple(int(d) for d in quant_op.local_shape[:-1])
        out_n = int(quant_op.local_shape[-1])
    else:
        out_lead = fallback_lead
        out_n = n_dim
    y_shape = (*out_lead, out_n)
    if is_mx_scale:
        grp = g_shape[0] if g_shape else (int(raw_gs) if raw_gs and int(raw_gs) > 1 else MX_GROUP_SIZE)
        blocks = max(1, (out_n + grp - 1) // grp)
        pairs = max(1, (blocks + _MX_SCALE_PAIR_ALIGN - 1) // _MX_SCALE_PAIR_ALIGN)
        scale_shape: tuple[int, ...] = (*out_lead, pairs, _MX_SCALE_PAIR_ALIGN)
    else:
        scale_shape = out_lead
    return (
        _port(y_name, out_dt, y_shape, quant_mode=q_mode, group_shape=g_shape, scale_dtype=s_dt),
        _port(scale_name, s_dt, scale_shape),
    )


def _classify_ub_chain_abi(
    chain: Sequence[OpSpec],
    traits: Mapping[str, OpTraitInfo],
    *,
    soc_arch: SocArch | None = None,
    sku_id: str = "",
    pr_track: StackTrack = "shipping",
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] = DEFAULT_SHIPPING_STACK_PASSES,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[str, FusionTier, bool, str, AbiSupportVerdict | None, str, RegionBackendTarget]:
    """Classify a Law 1 UB chain strictly against ``ProbedAbiRegistry`` (zero fake defaults)."""
    first, last = chain[0], chain[-1]
    in_dt = _norm_abi_dtype(traits[first.op_id].input_dtype)
    if len(first.local_shape) >= 2:
        in_lead = tuple(int(d) for d in first.local_shape[:-1])
        n_dim = int(first.local_shape[-1])
    else:
        in_lead = (1,)
        n_dim = max(1, int(first.local_shape[0]) if first.local_shape else 1)
    in_shape = (*in_lead, n_dim)

    # Pure zero-copy view elision (never stamped as fake CANN K1_shipping).
    if all(
        traits[op.op_id].trait == AlgebraicTrait.VIEW_METADATA
        and str(op.family or "").strip().lower() in _ZERO_COPY_VIEW_FAMILIES
        for op in chain
    ):
        return "StridedViewElided", "K2_epilogue", False, "logical_view_elision", None, "", "tilelang"

    # Terminal cache-quant sink chain (e.g. LayerNorm -> IndexerQuantCache): K2_epilogue on target track only.
    if traits[last.op_id].is_terminal_sink:
        return (
            "FusedUbTerminalCacheQuantSink", "K2_epilogue", False,
            "terminal_sink_fusion_requires_custom_k2", None, "", "tilelang",
        )

    material = [
        op for op in chain
        if not (
            traits[op.op_id].trait == AlgebraicTrait.VIEW_METADATA
            and str(op.family or "").strip().lower() in _ZERO_COPY_VIEW_FAMILIES
        )
    ]
    mat_syms = tuple(traits[op.op_id].symbolic_op_kind for op in material)
    term_op = material[-1] if material else last
    term_tr = traits[term_op.op_id]
    if mat_syms in {
        ("add", "rmsnorm", "dynamic_quant"),
        ("add_rmsnorm", "dynamic_quant"),
        ("add", "rmsnorm", "dynamic_mx_quant"),
        ("add_rmsnorm", "dynamic_mx_quant"),
    }:
        is_mx = mat_syms[-1] == "dynamic_mx_quant"
        abi_id = "aclnnAddRmsNormDynamicMxQuant" if is_mx else "aclnnAddRmsNormDynamicQuant"
        if soc_arch is not None:
            operands = (
                _port("x1", in_dt, in_shape),
                _port("x2", in_dt, in_shape),
                _port("gamma", in_dt, (n_dim,)),
            )
            if is_mx:
                y_port, sc_port = _law1_quant_output_ports(
                    term_op, term_tr, in_lead, n_dim, y_name="y", scale_name="scale",
                )
                x_out = _port("x_out", in_dt, in_shape)
                outputs = (y_port, x_out, sc_port)
            else:
                y1_port, sc1_port = _law1_quant_output_ports(
                    term_op, term_tr, in_lead, n_dim, y_name="y1", scale_name="scale1",
                )
                y2_port = (
                    replace(y1_port, role="y2")
                    if _has_active_smooth_scale2(chain)
                    else replace(y1_port, role="y2", shape=(1,))
                )
                x_out = _port("x", in_dt, in_shape)
                sc2_port = replace(sc1_port, role="scale2")
                outputs = (y1_port, y2_port, x_out, sc1_port, sc2_port)
            verdict, q_digest = _verify_abi_query(
                abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id, graph_motif="add_rmsnorm_quant",
                operands=operands, outputs=outputs, pr_track=pr_track, track_spec=track_spec,
                manifest_bindings=manifest_bindings, enabled_stack_passes=enabled_stack_passes,
                registry=registry, model_scope=_resolve_model_scope(chain),
            )
            tier, shipping, unsup, backend = _verdict_to_tier(
                verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
            )
            return abi_id, tier, shipping, unsup, verdict, q_digest, backend
        return abi_id, "K3_dsl_pipeline", False, "soc_arch_unspecified_for_abi_probe", None, "", "tilelang"

    if mat_syms in {("situ", "dynamic_quant"), ("situ", "dynamic_mx_quant")}:
        abi_id = "aclnnDequantSituQuant"
        if soc_arch is not None:
            y_port, sc_port = _law1_quant_output_ports(
                term_op, term_tr, in_lead, n_dim, y_name="y", scale_name="y_scale",
            )
            verdict, q_digest = _verify_abi_query(
                abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id, graph_motif="situ_quant",
                operands=(_port("x", in_dt, in_shape),),
                outputs=(y_port, sc_port),
                pr_track=pr_track, track_spec=track_spec, manifest_bindings=manifest_bindings,
                enabled_stack_passes=enabled_stack_passes, registry=registry,
                model_scope=_resolve_model_scope(chain), activation_kind="situglu", is_act_and_mul=True,
            )
            tier, shipping, unsup, backend = _verdict_to_tier(
                verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
            )
            return abi_id, tier, shipping, unsup, verdict, q_digest, backend
        return abi_id, "K3_dsl_pipeline", False, "soc_arch_unspecified_for_abi_probe", None, "", "tilelang"

    if mat_syms in {("swiglu", "dynamic_quant"), ("swiglu", "dynamic_mx_quant")}:
        abi_id = "aclnnSwiGluQuant"
        if soc_arch is not None:
            y_port, sc_port = _law1_quant_output_ports(
                term_op, term_tr, in_lead, n_dim, y_name="y", scale_name="scale",
            )
            verdict, q_digest = _verify_abi_query(
                abi_id=abi_id, soc_arch=soc_arch, sku_id=sku_id, graph_motif="swiglu_quant",
                operands=(_port("x", in_dt, in_shape),),
                outputs=(y_port, sc_port),
                pr_track=pr_track, track_spec=track_spec, manifest_bindings=manifest_bindings,
                enabled_stack_passes=enabled_stack_passes, registry=registry,
                model_scope=_resolve_model_scope(chain), activation_kind="swiglu", is_act_and_mul=True,
            )
            tier, shipping, unsup, backend = _verdict_to_tier(
                verdict, fallback_tier="K3_dsl_pipeline", fallback_backend="tilelang",
            )
            return abi_id, tier, shipping, unsup, verdict, q_digest, backend
        return abi_id, "K3_dsl_pipeline", False, "soc_arch_unspecified_for_abi_probe", None, "", "tilelang"

    return "FusedUbPointwiseChain", "K3_dsl_pipeline", False, "no_shipping_composite_ub_abi", None, "", "tilelang"


def _candidate_priority_weight(cand: AlgebraicCandidate) -> tuple[int, int, int]:
    return (1 if cand.shipping_supported else 0, cand.avoidable_hbm_bytes, len(cand.member_ids))


def _resolve_conflict_graph(candidates: Sequence[AlgebraicCandidate]) -> list[AlgebraicCandidate]:
    """Select maximum-weight non-overlapping candidates across competing conflict components."""
    n = len(candidates)
    if n <= 1:
        return list(candidates)
    member_sets = [frozenset(c.member_ids) for c in candidates]
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if member_sets[i] & member_sets[j]:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    selected_indices: list[int] = []
    for root in range(n):
        if visited[root]:
            continue
        comp: list[int] = []
        stack = [root]
        visited[root] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) == 1:
            selected_indices.append(comp[0])
            continue
        if len(comp) <= MAX_CONFLICT_EXACT_SIZE:
            comp.sort()
            pos = {node: idx for idx, node in enumerate(comp)}
            conflict_mask = [0] * len(comp)
            for idx, node in enumerate(comp):
                mask = 0
                for nb in adj[node]:
                    mask |= 1 << pos[nb]
                conflict_mask[idx] = mask
            best_score: tuple[int, int, int] = (-1, -1, -1)
            best_mask = 0
            for mask in range(1, 1 << len(comp)):
                valid = True
                ship_s = avoid_s = len_s = 0
                for idx in range(len(comp)):
                    if not (mask & (1 << idx)):
                        continue
                    if mask & conflict_mask[idx]:
                        valid = False
                        break
                    w = _candidate_priority_weight(candidates[comp[idx]])
                    ship_s += w[0]
                    avoid_s += w[1]
                    len_s += w[2]
                if valid and (ship_s, avoid_s, len_s) > best_score:
                    best_score = (ship_s, avoid_s, len_s)
                    best_mask = mask
            for idx in range(len(comp)):
                if best_mask & (1 << idx):
                    selected_indices.append(comp[idx])
        else:
            ranked = sorted(comp, key=lambda i: _candidate_priority_weight(candidates[i]), reverse=True)
            chosen_ops: set[str] = set()
            for idx in ranked:
                if not (member_sets[idx] & chosen_ops):
                    selected_indices.append(idx)
                    chosen_ops.update(member_sets[idx])
    selected_indices.sort()
    return [candidates[i] for i in selected_indices]


def discover_algebraic_rewrites_for_ops(
    ops: Sequence[OpSpec],
    *,
    datapath: DatapathCapabilitySpec | None = None,
    allow_mc2: bool = True,
    include_abi_extensions: bool = True,
    arch_family: str = "",
    sku_id: str = "",
    enable_fused_mc2: int | None = None,
    tp_size: int | None = None,
    ep_size: int | None = None,
    pr_track: StackTrack = "shipping",
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
    enabled_stack_passes: frozenset[str] | None = None,
    base_graph_sha256: str = "",
    framing_sha256: str = "",
    registry: ProbedAbiRegistry | None = None,
) -> AlgebraicRewriteResult:
    """Run all 4 universal algebraic laws over a rank-local ``OpSpec`` DAG."""
    if not ops or pr_track == "eager":
        return AlgebraicRewriteResult(
            plan=empty_plan(base_graph_sha256, framing_sha256),
            candidates=(),
            semantic_audit_triggers=(),
        )
    by_id = {op.op_id: op for op in ops}
    ordered = _topological(by_id)
    traits = {op.op_id: classify_op_traits(op) for op in ordered}
    data_deps, data_succ = _data_dependency_graph(ordered)
    soc_arch = _resolve_soc_arch(arch_family, datapath, sku_id)
    passes = DEFAULT_SHIPPING_STACK_PASSES if enabled_stack_passes is None else frozenset(enabled_stack_passes)
    eff_track: StackTrack = track_spec.track if track_spec is not None else pr_track
    # Allow explicit target-synthesis exploration when include_abi_extensions=True on standalone calls.
    cand_track: StackTrack = "target" if (include_abi_extensions and track_spec is None and eff_track == "shipping") else eff_track

    # Enumerate competing candidates independently across all 4 laws (non-greedy conflict graph).
    raw_candidates: list[AlgebraicCandidate] = []
    raw_candidates.extend(_match_law4_mla_prolog(
        ordered, traits, set(), data_deps, data_succ,
        soc_arch=soc_arch, sku_id=sku_id, pr_track=cand_track, track_spec=track_spec,
        manifest_bindings=manifest_bindings, enabled_stack_passes=passes, registry=registry,
    ))
    raw_candidates.extend(_discover_law3_mc2(
        ordered, traits, set(), data_deps, data_succ,
        allow_mc2=allow_mc2 and enable_fused_mc2 != 0,
        soc_arch=soc_arch, sku_id=sku_id, tp_size=tp_size, pr_track=cand_track,
        track_spec=track_spec, manifest_bindings=manifest_bindings,
        enabled_stack_passes=passes, include_abi_extensions=include_abi_extensions, registry=registry,
    ))
    law2_cands, rejected = _discover_law2_block_matmul(
        ordered, traits, set(), data_succ,
        datapath=datapath, soc_arch=soc_arch, sku_id=sku_id, pr_track=cand_track,
        epilogue_track=eff_track,
        track_spec=track_spec, manifest_bindings=manifest_bindings,
        enabled_stack_passes=passes, include_abi_extensions=include_abi_extensions, registry=registry,
    )
    raw_candidates.extend(law2_cands)
    law1_cands, triggers = _discover_law1_and_law4_chains(
        ordered, traits, set(), data_deps, data_succ, datapath,
        include_abi_extensions=include_abi_extensions,
        arch_family=arch_family,
        enable_fused_mc2=enable_fused_mc2,
        soc_arch=soc_arch, sku_id=sku_id, ep_size=ep_size, allow_mc2=allow_mc2,
        pr_track=cand_track, track_spec=track_spec,
        manifest_bindings=manifest_bindings, enabled_stack_passes=passes, registry=registry,
    )
    raw_candidates.extend(law1_cands)

    candidates = _resolve_conflict_graph(raw_candidates)
    edits: list[Edit] = []
    used_edit_ids: set[str] = set()
    for idx, cand in enumerate(candidates):
        eid = cand.candidate_id
        if eid in used_edit_ids:
            eid = f"{eid}_{idx}"
        used_edit_ids.add(eid)
        refs = tuple(OpRef("atom", oid) for oid in cand.member_ids)
        preserve = (
            tuple(PortRef(OpRef("atom", oid), pname, ver) for oid, pname, ver in cand.preserve_ports)
            if cand.preserve_ports else
            tuple(PortRef(OpRef("atom", oid), "output", 0) for oid in cand.preserve_port_ids)
        )
        edits.append(Edit(
            edit_id=eid,
            operation=FuseRegion(
                members=refs,
                preserve_ports=preserve,
                kernel_boundaries=(KernelBoundary(cand.abi_id, refs),),
                fusion_tier=cand.fusion_tier,
                backend_target=cand.backend_target,
                abi_verdict=None,
                abi_query_digest=cand.abi_query_digest,
                lowering_id=cand.lowering_id,
                lowering_sha256=cand.lowering_sha256,
                ub_peak_bytes=cand.ub_peak_bytes,
                ub_wall_bytes=cand.ub_budget_bytes,
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


def _passes_multi_rank_consensus(cand: AlgebraicCandidate, scenario: ResolvedScenario) -> bool:
    """Require 100% rank consensus (presence + shape/dtype/precision parity) for collective edits."""
    if not cand.requires_comm_consensus or len(scenario.atomic_ops) <= 1:
        return True
    required = tuple(cand.member_ids)
    ref_map = {op.op_id: op for op in scenario.atomic_ops[0].ops}
    if any(oid not in ref_map for oid in required):
        return False
    ref_sigs = {
        oid: (
            ref_map[oid].family,
            ref_map[oid].local_shape,
            ref_map[oid].dtype,
            getattr(ref_map[oid], "precision_abi", None),
            ref_map[oid].link_scope,
        )
        for oid in required
    }
    for rank_ops in scenario.atomic_ops[1:]:
        by_id = {op.op_id: op for op in rank_ops.ops}
        for oid, ref_sig in ref_sigs.items():
            op = by_id.get(oid)
            if op is None:
                return False
            sig = (op.family, op.local_shape, op.dtype, getattr(op, "precision_abi", None), op.link_scope)
            if sig != ref_sig:
                return False
    return True


def discover_algebraic_rewrites(
    scenario: ResolvedScenario,
    *,
    allow_mc2: bool | None = None,
    include_abi_extensions: bool = True,
    verify: bool = True,
) -> AlgebraicRewriteResult:
    """Discover algebraic rewrites on a ``ResolvedScenario`` with consensus and verification."""
    from .apply import apply_plan
    from .verify import verify_plan

    meta = dict(scenario.case.runtime.metadata) if scenario.case.runtime.metadata else {}
    add_cfg = meta.get("additional_config") if isinstance(meta.get("additional_config"), Mapping) else {}
    raw_mc2 = add_cfg.get("enable_fused_mc2")
    fused_mc2_level = int(raw_mc2) if raw_mc2 is not None else None
    if allow_mc2 is None:
        allow_mc2 = (fused_mc2_level != 0) if fused_mc2_level is not None else True
    arch_family = str(getattr(scenario.case.machine, "arch_family", "") or "")
    sku_id = str(getattr(scenario.case.machine, "sku", "") or getattr(scenario.case.machine, "name", "") or "")
    parallel = scenario.case.runtime.parallel
    raw_track = str(meta.get("stack_track") or meta.get("pr_track") or "shipping").strip().lower()
    pr_track: StackTrack = raw_track if raw_track in {"eager", "shipping", "proposed", "target"} else "shipping"  # type: ignore[assignment]

    rep_ops = scenario.atomic_ops[0].ops if scenario.atomic_ops else ()
    raw_res = discover_algebraic_rewrites_for_ops(
        rep_ops,
        datapath=scenario.datapath,
        allow_mc2=bool(allow_mc2),
        include_abi_extensions=include_abi_extensions,
        arch_family=arch_family,
        sku_id=sku_id,
        enable_fused_mc2=fused_mc2_level,
        tp_size=int(parallel.tp),
        ep_size=int(parallel.ep),
        pr_track=pr_track,
        base_graph_sha256=scenario.base_graph_sha256,
        framing_sha256=scenario.framing_sha256,
    )
    # Enforce 100% communicator consensus across all ranks for collective edits.
    consensus_edits: list[Edit] = []
    consensus_cands: list[AlgebraicCandidate] = []
    rejected_pairs = list(raw_res.rejected_pairs)
    for edit, cand in zip(raw_res.plan.edits, raw_res.candidates):
        if _passes_multi_rank_consensus(cand, scenario):
            consensus_edits.append(edit)
            consensus_cands.append(cand)
        else:
            rejected_pairs.append((edit.edit_id, "multi_rank_collective_consensus_failed"))
    consensus_plan = RewritePlan(
        base_graph_sha256=scenario.base_graph_sha256,
        framing_sha256=scenario.framing_sha256,
        edits=tuple(consensus_edits),
    )
    if not verify:
        return AlgebraicRewriteResult(
            plan=consensus_plan,
            candidates=tuple(consensus_cands),
            semantic_audit_triggers=raw_res.semantic_audit_triggers,
            rejected_pairs=tuple(rejected_pairs),
        )
    try:
        applied = apply_plan(consensus_plan, scenario)
        receipt = verify_plan(scenario, applied, scope="archetype")
    except ModelingError:
        receipt = None
    if receipt is None or not receipt.passed:
        safe_edits: list[Edit] = []
        safe_cands: list[AlgebraicCandidate] = []
        for edit, cand in zip(consensus_edits, consensus_cands):
            trial = RewritePlan(
                base_graph_sha256=scenario.base_graph_sha256,
                framing_sha256=scenario.framing_sha256,
                edits=(*safe_edits, edit),
            )
            try:
                t_app = apply_plan(trial, scenario)
                t_rec = verify_plan(scenario, t_app, scope="archetype")
            except ModelingError as exc:
                rejected_pairs.append((edit.edit_id, str(exc)))
                continue
            if t_rec.passed:
                safe_edits.append(edit)
                safe_cands.append(cand)
            else:
                failed = ",".join(f"{g.name}:{g.reason}" for g in t_rec.gates if not g.passed) or "verification_failed"
                rejected_pairs.append((edit.edit_id, failed))
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
            rejected_pairs=tuple(rejected_pairs),
            verification=final_rec,
        )
    return AlgebraicRewriteResult(
        plan=consensus_plan,
        candidates=tuple(consensus_cands),
        semantic_audit_triggers=raw_res.semantic_audit_triggers,
        rejected_pairs=tuple(rejected_pairs),
        verification=receipt,
    )


__all__ = [
    "AlgebraicTrait",
    "OpTraitInfo",
    "AlgebraicCandidate",
    "SemanticAuditTrigger",
    "AlgebraicRewriteResult",
    "classify_op_traits",
    "discover_algebraic_rewrites_for_ops",
    "discover_algebraic_rewrites",
]
