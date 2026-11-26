"""Public entry point for the governed MLA prolog family.

[Role] Bind an explicit compute architecture and ABI to one MLA prolog plan.
[Invariants] Each ISA leaf stays behind the package dispatcher. Legacy recipe
  callers use the same planning and lowering path as governed occurrences.
[Timeouts & Locks] Bounded by the selected strategy. Uses no I/O or locks.
[Side-Effect Policy] Returns plans and registration records without global mutation.
"""
from __future__ import annotations

from typing import Literal, TypeAlias

from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
from mindie_sim.schema.abi import AbiQueryKey, OperandPortSig
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import (
    GENERIC_ARCH_ID, ArchId, ComputeArchIdentity, DatapathCapabilitySpec, SourceRef,
)
from mindie_sim.schema.operators import OpSpec

from ...mechanism import Geometry, OperatorPlan, PlannedRecipe
from ...results import FusedPipelineDescriptor, OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from ..attention.common import with_sequence_context
from . import arch22 as _arch22, arch35 as _arch35
from .common import (
    ACLNN_SUPPORTED_NOPE_DIMS,
    ACLNN_SUPPORTED_Q_LORA_RANKS,
    BF16_BYTES,
    C8_BLOCK_SIZE,
    DEFAULT_KV_LORA_RANK,
    DEFAULT_PROLOG_HEADS,
    DEFAULT_PROLOG_HIDDEN,
    DEFAULT_QK_NOPE_DIM,
    DEFAULT_QK_ROPE_DIM,
    DEFAULT_Q_LORA_RANK,
    FAMILY,
    FP32_BYTES,
    MLA_PROLOG_V4_RECIPE,
    MX_GROUP_SIZE,
    SLOT_INDEX_BYTES,
    SUPPORTED_WEIGHT_MODES,
    WEIGHT_QUANT_BF16,
    WEIGHT_QUANT_INT8,
    WEIGHT_QUANT_MXFP8,
    PrologContract,
    make_op,
    unsupported,
    uses_split_m,
    validate_contract,
    w_uk_ownership,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCE_REF = SourceRef(
    repository="cann/ops-transformer",
    revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    relative_path="attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp",
    source_sha256="0950916bb2813a68a0019b13fda325d929da6fc56616423307a3ff158c19def9",
)
_STRATEGIES = {"npu_arch_2201": _arch22.STRATEGY, "npu_arch_3510": _arch35.STRATEGY}

MlaPreprocessMode: TypeAlias = Literal[
    "prolog_v3",
    "prolog_v3_learned_rope_tail",
    "mlapo",
    "strided_kv_adapter",
    "native_decomposed",
]
MLA_PREPROCESS_MODES: tuple[MlaPreprocessMode, ...] = (
    "prolog_v3",
    "prolog_v3_learned_rope_tail",
    "mlapo",
    "strided_kv_adapter",
    "native_decomposed",
)
MLAPO_MAX_SUPPORTED_TOKENS = 1024
PROLOG_V3_SUPPORTED_INPUT_SIZES: frozenset[int] = frozenset({
    1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192,
})
_DECODE_MLAPO_PHASES = frozenset({"decode", "draft"})
_PROLOG_WEIGHT_UK_PORT = "weight_uk:bf16:ND:dense::none"
_PROLOG_BF16_TAIL_PORTS: tuple[str, ...] = (
    "rmsnorm_gamma_cq:bf16:ND:dense::none",
    "rmsnorm_gamma_ckv:bf16:ND:dense::none",
    "rope_sin:bf16:ND:dense::none",
    "rope_cos:bf16:ND:dense::none",
    "kv_cache_ref:bf16:ND:dense::none",
    "kr_cache_ref:bf16:ND:dense::none",
)
_PROLOG_BF16_PORTS: tuple[str, ...] = (
    "token_x:bf16:ND:dense::none",
    "weight_dq:bf16:FRACTAL_NZ:dense::none",
    "weight_uq_qr:bf16:FRACTAL_NZ:dense::none",
    _PROLOG_WEIGHT_UK_PORT,
    "weight_dkv_kr:bf16:FRACTAL_NZ:dense::none",
    *_PROLOG_BF16_TAIL_PORTS,
)
_PROLOG_INT8_PORTS: tuple[str, ...] = (
    "token_x:int8:ND:per_token_int8::fp32",
    "weight_dq:int8:FRACTAL_NZ:per_channel_int8::fp32",
    "weight_uq_qr:int8:FRACTAL_NZ:per_channel_int8::fp32",
    _PROLOG_WEIGHT_UK_PORT,
    "weight_dkv_kr:int8:FRACTAL_NZ:per_channel_int8::fp32",
    *_PROLOG_BF16_TAIL_PORTS,
)
_PROLOG_MXFP8_PORTS: tuple[str, ...] = (
    "token_x:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0",
    "weight_dq:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0",
    "weight_uq_qr:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0",
    _PROLOG_WEIGHT_UK_PORT,
    "weight_dkv_kr:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0",
    *_PROLOG_BF16_TAIL_PORTS,
)
_PROLOG_JOINT_DTYPE_TUPLES: dict[str, tuple[str, ...]] = {
    "bf16": _PROLOG_BF16_PORTS,
    "bfloat16": _PROLOG_BF16_PORTS,
    "int8": _PROLOG_INT8_PORTS,
    "w8a8": _PROLOG_INT8_PORTS,
    "w8a8_int8": _PROLOG_INT8_PORTS,
    "w8a8_dynamic": _PROLOG_INT8_PORTS,
    "mxfp8": _PROLOG_MXFP8_PORTS,
    "fp8": _PROLOG_MXFP8_PORTS,
    "fp8_e4m3": _PROLOG_MXFP8_PORTS,
    "fp8_e4m3fn": _PROLOG_MXFP8_PORTS,
}
_PROLOG_ALLOWED_ACT_DTYPES = frozenset({"bf16", "bfloat16"})
_SFA_PROLOG_V3_QUANT_MODES = frozenset({
    "int8", "w8a8_int8", "w8a8_dynamic", "mxfp8", "fp8", "fp8_e4m3", "fp8_e4m3fn",
})
_SFA_STATIC_MLAPO_QUANT_MODES = frozenset({"w8a8"})
_PROLOG_ARCH_DEFAULT_SKU: dict[str, str] = {
    "A2": "ascend910b2",
    "A3": "ascend910_9362",
    "A5": "ascend950dt_9582",
}
_PROLOG_JOINT_OUTPUT_TUPLES: dict[str, tuple[tuple[str, str], ...]] = {
    "bf16": (
        ("query_out", "bf16"),
        ("query_rope_out", "bf16"),
        ("query_norm_out", "bf16"),
        ("dequant_scale_q_norm_out", "fp32"),
    ),
    "int8": (
        ("query_out", "bf16"),
        ("query_rope_out", "bf16"),
        ("query_norm_out", "int8"),
        ("dequant_scale_q_norm_out", "fp32"),
    ),
    "mxfp8": (
        ("query_out", "bf16"),
        ("query_rope_out", "bf16"),
        ("query_norm_out", "fp8_e4m3fn"),
        ("dequant_scale_q_norm_out", "fp8_e8m0"),
    ),
}


def _parse_prolog_port(
    spec: str,
    *,
    num_tokens: int,
    hidden_size: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_q_heads: int,
    qk_nope_head_dim: int,
) -> OperandPortSig:
    parts = spec.split(":")
    name, dtype, layout = parts[0], parts[1], parts[2]
    quant_mode = parts[3] if len(parts) > 3 and parts[3] else "dense"
    group_shape = tuple(int(v) for v in parts[4].split("x") if v) if len(parts) > 4 and parts[4] else ()
    scale_dtype = parts[5] if len(parts) > 5 and parts[5] else "none"
    t = max(1, int(num_tokens))
    h = int(hidden_size)
    q_r = max(1, int(q_lora_rank))
    kv_r = max(1, int(kv_lora_rank))
    rope_d = max(1, int(qk_rope_head_dim))
    n_heads = max(1, int(num_q_heads) if int(num_q_heads) > 0 else DEFAULT_PROLOG_HEADS)
    nope_d = max(1, int(qk_nope_head_dim) if int(qk_nope_head_dim) > 0 else DEFAULT_QK_NOPE_DIM)
    q_out_w = n_heads * (nope_d + rope_d)
    kv_out_w = kv_r + rope_d
    shapes: dict[str, tuple[int, ...]] = {
        "token_x": (t, h),
        "weight_dq": (h, q_r),
        "weight_uq_qr": (q_r, q_out_w),
        "weight_uk": (n_heads, nope_d, kv_r),
        "weight_dkv_kr": (h, kv_out_w),
        "rmsnorm_gamma_cq": (q_r,),
        "rmsnorm_gamma_ckv": (kv_r,),
        "rope_sin": (t, rope_d),
        "rope_cos": (t, rope_d),
        "kv_cache_ref": (1, C8_BLOCK_SIZE, 1, kv_r),
        "kr_cache_ref": (1, C8_BLOCK_SIZE, 1, rope_d),
    }
    if name not in shapes:
        raise ModelingError(f"mla_prolog_unknown_port:{name}")
    shape = shapes[name]
    scale_shape: tuple[int, ...] = ()
    if quant_mode == "mx_fp8" and group_shape:
        grp = group_shape[0]
        mx_scales: dict[str, tuple[int, ...]] = {
            "token_x": (t, max(1, h // grp)),
            "weight_dq": (q_r, max(1, h // grp)),
            "weight_uq_qr": (q_out_w, max(1, q_r // grp)),
            "weight_dkv_kr": (kv_out_w, max(1, h // grp)),
        }
        scale_shape = mx_scales.get(name, ())
    elif quant_mode == "per_token_int8" and name == "token_x":
        scale_shape = (t, 1)
    elif quant_mode == "per_channel_int8":
        ch_scales: dict[str, tuple[int, ...]] = {
            "weight_dq": (1, q_r),
            "weight_uq_qr": (1, q_out_w),
            "weight_dkv_kr": (1, kv_out_w),
        }
        scale_shape = ch_scales.get(name, ())
    return OperandPortSig(
        role=name,
        dtype=dtype,
        layout=layout,  # type: ignore[arg-type]
        transposed=False,
        shape=shape,
        quant_mode=quant_mode,
        group_shape=group_shape,
        scale_dtype=scale_dtype,
        scale_shape=scale_shape,
    )


def _prolog_v3_abi_available(
    reg: ProbedAbiRegistry,
    arch_norm: str,
    wq_norm: str,
    joint_tuple: tuple[str, ...],
    *,
    num_tokens: int,
    input_size: int | None,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_q_heads: int,
    qk_nope_head_dim: int,
    is_sparse_sfa: bool,
    is_int8_kv: bool,
    dcp_size: int,
    pcp_size: int,
) -> bool:
    """Verify full 11-input ``aclnnMlaPrologV3WeightNz`` joint ABI via ``ProbedAbiRegistry.verify``."""
    hidden = int(input_size) if input_size is not None else DEFAULT_PROLOG_HIDDEN
    operands = tuple(
        _parse_prolog_port(
            spec,
            num_tokens=num_tokens,
            hidden_size=hidden,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            num_q_heads=num_q_heads,
            qk_nope_head_dim=qk_nope_head_dim,
        )
        for spec in joint_tuple
    )
    out_bucket = "mxfp8" if ("fp8" in wq_norm) else ("int8" if ("int8" in wq_norm or "w8a8" in wq_norm) else "bf16")
    t = max(1, int(num_tokens))
    n_heads = max(1, int(num_q_heads) if int(num_q_heads) > 0 else DEFAULT_PROLOG_HEADS)
    q_r = max(1, int(q_lora_rank))
    kv_r = max(1, int(kv_lora_rank))
    rope_d = max(1, int(qk_rope_head_dim))
    out_shapes: dict[str, tuple[int, ...]] = {
        "query_out": (t, n_heads, kv_r),
        "query_rope_out": (t, n_heads, rope_d),
        "query_norm_out": (t, q_r),
        "dequant_scale_q_norm_out": (
            (t, max(1, q_r // MX_GROUP_SIZE)) if out_bucket == "mxfp8" else (t, 1)
        ),
    }
    outputs = tuple(
        OperandPortSig(
            role=out_name,
            dtype=out_dtype,
            layout="ND",
            transposed=False,
            shape=out_shapes[out_name],
            quant_mode="dense" if out_dtype == "bf16" else ("mx_fp8" if "fp8" in out_dtype else "per_token_int8"),
            group_shape=(MX_GROUP_SIZE,) if "fp8" in out_dtype else (),
            scale_dtype="none",
            scale_shape=(),
        )
        for out_name, out_dtype in _PROLOG_JOINT_OUTPUT_TUPLES[out_bucket]
    )
    query = AbiQueryKey(
        soc_arch=arch_norm,  # type: ignore[arg-type]
        sku_id=_PROLOG_ARCH_DEFAULT_SKU[arch_norm],
        abi_id="aclnnMlaPrologV3WeightNz",
        graph_motif="mla_prolog",
        model_scope="target",
        draft_quant_mode="none",
        pr_track="shipping",
        operands=operands,
        outputs=outputs,
        rank_placement="local",
        world_size=1,
        comm_group_id="none",
        comm_domain_width=1,
        comm_dtype="none",
        reduction_kind="none",
        activation_kind="none",
        activation_params=(),
        has_bias=False,
        has_active_mask=False,
        dispatch_quant_mode=0,
        shared_experts_per_rank=0,
        top_k=0,
        moe_expert_num=0,
        qk_rope_head_dim=int(qk_rope_head_dim),
        mla_variant="latent_mqa" if is_sparse_sfa else "latent_mla",
        sfa_dcp_active=bool(is_sparse_sfa and int(dcp_size) > 1),
        pcp_size=int(pcp_size),
        kv_cache_dtype="int8" if is_int8_kv else "bf16",
        has_kpool_indexer=False,
        dcp_size=int(dcp_size),
        is_act_and_mul=False,
        enabled_stack_passes=frozenset(),
    )
    verdict = reg.verify(query)
    return verdict.status == "admitted_shipping"


def _prolog_v3_ready(
    *,
    arch_norm: str,
    wq_norm: str,
    num_tokens: int,
    input_size: int | None,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_q_heads: int,
    qk_nope_head_dim: int,
    is_sparse_sfa: bool,
    is_int8_kv: bool,
    dcp_size: int,
    pcp_size: int,
    registry: ProbedAbiRegistry | None,
) -> bool:
    if (int(q_lora_rank) not in ACLNN_SUPPORTED_Q_LORA_RANKS
            or int(kv_lora_rank) != DEFAULT_KV_LORA_RANK
            or int(qk_rope_head_dim) != DEFAULT_QK_ROPE_DIM
            or (int(qk_nope_head_dim) > 0 and int(qk_nope_head_dim) not in ACLNN_SUPPORTED_NOPE_DIMS)
            or arch_norm not in _PROLOG_ARCH_DEFAULT_SKU):
        return False
    joint_tuple = _PROLOG_JOINT_DTYPE_TUPLES.get(wq_norm)
    if joint_tuple is None:
        return False
    reg = registry if registry is not None else default_abi_registry()
    return _prolog_v3_abi_available(
        reg,
        arch_norm,
        wq_norm,
        joint_tuple,
        num_tokens=int(num_tokens),
        input_size=input_size,
        q_lora_rank=int(q_lora_rank),
        kv_lora_rank=int(kv_lora_rank),
        qk_rope_head_dim=int(qk_rope_head_dim),
        num_q_heads=int(num_q_heads),
        qk_nope_head_dim=int(qk_nope_head_dim),
        is_sparse_sfa=bool(is_sparse_sfa),
        is_int8_kv=bool(is_int8_kv),
        dcp_size=int(dcp_size),
        pcp_size=int(pcp_size),
    )


def resolve_mla_preprocess_mode(
    *,
    arch: str,
    phase: str,
    num_tokens: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    has_rope: bool = True,
    is_sparse_sfa: bool = False,
    is_kv_consumer: bool = False,
    dcp_size: int = 1,
    pcp_size: int = 1,
    enable_mlapo: bool = False,
    enable_mla_prolog: bool = True,
    qkv_a_gate_fused: bool = False,
    is_w4a8: bool = False,
    is_int8_kv: bool = False,
    has_kv_scale: bool = False,
    has_learned_rope_tail: bool = False,
    is_strided_kv_cache: bool = False,
    is_draft_model: bool = False,
    act_dtype: str = "bf16",
    weight_quant: str = "bf16",
    input_size: int | None = None,
    num_kv_heads: int = 1,
    num_q_heads: int = 0,
    qk_nope_head_dim: int = 0,
    registry: ProbedAbiRegistry | None = None,
) -> MlaPreprocessMode:
    """Exact 5-branch MLA preprocess selector (D1 §2.2 / M1; mla_v1.py & sfa_v1.py).

    Grounds ``MlaPrologV3`` admissibility in ``ProbedAbiRegistry.verify()`` and
    upstream BF16/input-width/quant/strided-cache contracts: FP16 stays
    ``native_decomposed``; sparse SFA routes dynamic quantized layers (and
    unquantized KV consumers) to ``prolog_v3`` across PCP/DCP and static W8A8
    layers to ``mlapo`` up to ``MLAPO_MAX_SUPPORTED_TOKENS`` on any phase; dense
    MLA checks strided ``prolog_v3`` (non-PCP, DCP allowed, supported-width drafts
    included) before A5 draft ``strided_kv_adapter`` and non-strided decode
    ``mlapo`` (``enable_mlapo`` or FA-quant INT8 KV with scale).
    """
    arch_norm = str(arch or "").strip().upper()
    act_norm = str(act_dtype).strip().lower() if act_dtype is not None else ""
    if not act_norm:
        raise ModelingError("mla_preprocess_act_dtype_required")
    wq_norm = str(weight_quant).strip().lower() if weight_quant is not None else ""
    if not wq_norm:
        raise ModelingError("mla_preprocess_weight_quant_required")
    if ((not has_rope and not (has_learned_rope_tail and arch_norm == "A5"))
            or int(qk_rope_head_dim) <= 0
            or qkv_a_gate_fused or is_w4a8
            or int(q_lora_rank) <= 0 or int(kv_lora_rank) <= 0
            or act_norm not in _PROLOG_ALLOWED_ACT_DTYPES):
        return "native_decomposed"
    unsupported_input_width = (
        input_size is not None and int(input_size) not in PROLOG_V3_SUPPORTED_INPUT_SIZES
    )
    prolog_mode: MlaPreprocessMode = (
        "prolog_v3_learned_rope_tail" if has_learned_rope_tail else "prolog_v3"
    )
    if is_sparse_sfa:
        if is_int8_kv and not has_kv_scale:
            return "native_decomposed"
        # Upstream sfa_v1.py:1063-1074, 1109-1117, 2073-2127: dynamic quantized SFA
        # and unquantized non-C8 KV consumers use prolog_v3 across PCP/DCP; static
        # W8A8 uses mlapo up to MLAPO_MAX_SUPPORTED_TOKENS on any phase.
        prolog_eligible = (
            wq_norm in _SFA_PROLOG_V3_QUANT_MODES
            or (wq_norm in _PROLOG_ALLOWED_ACT_DTYPES and is_kv_consumer and not is_int8_kv)
        )
        if (prolog_eligible and enable_mla_prolog and not unsupported_input_width
                and _prolog_v3_ready(
                    arch_norm=arch_norm, wq_norm=wq_norm, num_tokens=int(num_tokens),
                    input_size=input_size, q_lora_rank=int(q_lora_rank),
                    kv_lora_rank=int(kv_lora_rank), qk_rope_head_dim=int(qk_rope_head_dim),
                    num_q_heads=int(num_q_heads), qk_nope_head_dim=int(qk_nope_head_dim),
                    is_sparse_sfa=True, is_int8_kv=bool(is_int8_kv),
                    dcp_size=int(dcp_size), pcp_size=int(pcp_size), registry=registry,
                )):
            return prolog_mode
        if (wq_norm in _SFA_STATIC_MLAPO_QUANT_MODES and enable_mlapo
                and not is_int8_kv and not is_draft_model
                and 0 < int(num_tokens) <= MLAPO_MAX_SUPPORTED_TOKENS):
            return "mlapo"
        return "native_decomposed"

    # Dense MLA dispatch order (mla_v1.py:1921-1958, 3439-3495):
    # strided prolog_v3 (non-PCP, DCP allowed, drafts with supported width included)
    # -> A5 draft strided_kv_adapter -> non-strided decode mlapo (enable_mlapo or FA-quant).
    fa_quant_layer = bool(is_int8_kv and has_kv_scale)
    if (is_strided_kv_cache and enable_mla_prolog and not fa_quant_layer
            and not unsupported_input_width and int(pcp_size) <= 1
            and _prolog_v3_ready(
                arch_norm=arch_norm, wq_norm=wq_norm, num_tokens=int(num_tokens),
                input_size=input_size, q_lora_rank=int(q_lora_rank),
                kv_lora_rank=int(kv_lora_rank), qk_rope_head_dim=int(qk_rope_head_dim),
                num_q_heads=int(num_q_heads), qk_nope_head_dim=int(qk_nope_head_dim),
                is_sparse_sfa=False, is_int8_kv=bool(is_int8_kv),
                dcp_size=int(dcp_size), pcp_size=int(pcp_size), registry=registry,
            )):
        return prolog_mode
    if (is_draft_model and is_strided_kv_cache and arch_norm == "A5"
            and not fa_quant_layer and int(dcp_size) == 1 and int(pcp_size) == 1
            and int(num_kv_heads) == 1 and unsupported_input_width):
        return "strided_kv_adapter"
    if ((fa_quant_layer or (enable_mlapo and not is_draft_model))
            and not is_strided_kv_cache and int(pcp_size) <= 1
            and str(phase).strip().lower() in _DECODE_MLAPO_PHASES
            and 0 < int(num_tokens) <= MLAPO_MAX_SUPPORTED_TOKENS):
        return "mlapo"
    return "native_decomposed"


def _planning_context(op: OpSpec, geometry: Geometry) -> OperatorPlanningContext:
    datapath = geometry.datapath
    if datapath is None:
        declared = geometry.compute_arch
        identity = ComputeArchIdentity(
            ArchId(declared.preset) if (declared and declared.preset) else GENERIC_ARCH_ID,
            declared,
        )
        datapath = DatapathCapabilitySpec(
            compute_arch=identity, aic_count=geometry.compute_units,
            aiv_count=int(geometry.resource_units.get("vector", geometry.compute_units)),
            aic_aiv_pairs=(), usable_paths=(), matrix_output_paths=frozenset(),
            ub_peer_edges=(), stars=None,
        )
    return OperatorPlanningContext(
        occurrence=OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
        geometry=geometry, datapath=datapath, stream_owners=geometry.stream_owners,
    )


def build(op: OpSpec, geometry_or_ctx: Geometry | OperatorPlanningContext) -> OperatorPlan:
    ctx = (_planning_context(op, geometry_or_ctx)
           if isinstance(geometry_or_ctx, Geometry) else geometry_or_ctx)
    ctx = with_sequence_context(op, ctx)
    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)
    if strategy is None:
        raise unsupported(op, ctx, "unsupported_architecture")
    return strategy.build(op, ctx)


def _describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(
        ctx.occurrence, spec.family, spec.implementation, (op.op_id,), (),
        spec.phases, spec.sources, spec.repeats,
    )


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext) -> ResolvedOperator:
    plan = build(op, ctx)
    descriptor = _describe(op, ctx, plan)
    return ResolvedOperator(
        occurrence=ctx.occurrence, op=op, recipe=plan, implementation=descriptor.implementation,
        sources=descriptor.sources, descriptor=descriptor,
    )


def register_family() -> OperatorFamilySpec:
    return OperatorFamilySpec(
        name=FAMILY, families=frozenset({FAMILY}),
        owned_recipe_ids=frozenset({FAMILY, MLA_PROLOG_V4_RECIPE}),
        resolve=resolve_operator_plan,
    )


def recipes() -> dict[str, PlannedRecipe]:
    return {MLA_PROLOG_V4_RECIPE: PlannedRecipe(build)}


__all__ = [
    "MLAPO_MAX_SUPPORTED_TOKENS", "MLA_PREPROCESS_MODES", "MlaPreprocessMode",
    "PROLOG_V3_SUPPORTED_INPUT_SIZES", "PrologContract", "build", "make_op",
    "register_family", "resolve_mla_preprocess_mode", "resolve_operator_plan",
    "uses_split_m", "validate_contract", "w_uk_ownership",
]
