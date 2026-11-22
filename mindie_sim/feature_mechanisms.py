"""Optimization mechanism registry, catalog applicators and 4-track feature waterfall.

[Role] Own optimization mechanism declarations, structural applicability rules,
  and 4-track (raw_eager -> merged -> unmerged_pr -> expected_ideal) waterfall evaluation.
[Invariants] Mechanisms apply strictly by model/hardware structural capability,
  never by hardcoded model names.
[Timeouts & Locks] Delegates step evaluation to canonical Engine; no locks.
[Side-Effect Policy] Pure planning and read-only simulation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mindie_sim.prediction.operators.linear_attention.gdn_domain import (
    GDN_CHUNK32_TOKENS,
    model_supports_gdn_chunk32,
)
from mindie_sim.schema.model import ModelSpec

if TYPE_CHECKING:
    from mindie_sim.feature_mechanisms import MechanismContext, OptimizationMechanism

MX_SCALE_GROUP_SIZE = 32
MSA_MIN_BLOCK_SIZE = 64
FUSED_MC2_ROUTED_MODE = 2
FUSED_MC2_SHARED_MODE = 3
QSA_MROPE_DECODE_MAX_ROWS = 64
TEXT_GATED_RESIDUAL_MIN_ROWS = 513
QUANTIZED_ALLREDUCE_MAX_WORLD = 8

MXFP_EXPERT_WEIGHT_FORMATS = frozenset({"mxfp4", "mxfp8", "fp4_e2m1"})

PASS_PRIORITY_ORDER: tuple[str, ...] = (
    "activation_quantization",
    "norm_quant",
    "residual_norm",
    "moe_tail_residual",
    "mhc_boundary_pr",
    "mhc_boundary_ideal",
    "indexer_online_topk",
    "qkv_norm_rope",
    "kda_in_kernel_l2norm",
    "gdn_projection_overlap",
    "activation_quant",
    "gmm_swiglu_quant",
    "tp_mc2_overlap",
)


def _append_passes(runtime_dict: dict[str, Any], *names: str) -> None:
    current = list(runtime_dict.get("lowering_passes") or ())
    for name in names:
        if name not in current:
            current.append(name)
    order = {name: idx for idx, name in enumerate(PASS_PRIORITY_ORDER)}
    current.sort(key=lambda item: order.get(item, len(PASS_PRIORITY_ORDER)))
    runtime_dict["lowering_passes"] = current


def _set_lowering(runtime_dict: dict[str, Any], selector: str, binding: Any) -> None:
    kernels = dict(runtime_dict.get("kernels") or {})
    lowerings = dict(kernels.get("lowerings") or {})
    lowerings.setdefault(selector, binding)
    kernels["lowerings"] = lowerings
    runtime_dict["kernels"] = kernels


def _set_layout_strided(runtime_dict: dict[str, Any], consumer: str, *args: str) -> None:
    layouts = dict(runtime_dict.get("tensor_layouts") or {})
    entry = dict(layouts.get(consumer) or {})
    for arg in args:
        entry.setdefault(arg, "strided")
    layouts[consumer] = entry
    runtime_dict["tensor_layouts"] = layouts


def _has_quant_activations(model: ModelSpec) -> bool:
    return (model.quant is not None
            and str(model.quant.activation_format) != str(model.activation_dtype))


def _has_quant_moe_activations(model: ModelSpec) -> bool:
    """True when model activations or routed-expert overrides are quantized."""
    if _has_quant_activations(model):
        return True
    q = model.quant
    if q is None:
        return False
    for category, override in (getattr(q, "category_overrides", ()) or ()):
        if "routed_expert" not in str(category):
            continue
        if isinstance(override, dict):
            act_fmt = str(override.get("activation_format") or "")
            if act_fmt and act_fmt != str(model.activation_dtype):
                return True
    return False


def _apply_mhc_fusion_overlap(rt: dict[str, Any], ctx: MechanismContext) -> None:
    existing_raw = list(rt.get("lowering_passes") or ())
    unbounded = (
        bool(ctx.options.get("unbounded_rows", False))
        or "mhc_boundary_ideal" in existing_raw
    )
    pass_name = "mhc_boundary_ideal" if unbounded else "mhc_boundary_pr"
    # Upgrade PR-bounded pass to ideal pass cleanly and never downgrade ideal -> PR.
    existing = [p for p in existing_raw
                if p not in {"mhc_boundary_pr", "mhc_boundary_ideal"}]
    rt["lowering_passes"] = existing
    _append_passes(rt, pass_name)


def _apply_tp_mc2_overlap(rt: dict[str, Any], ctx: MechanismContext) -> None:
    _update_meta(rt, tp_mc2_overlap=True)
    if ctx.arch == "a5" and int(ctx.parallel.get("tp", 1) or 1) > 1:
        _append_passes(rt, "tp_mc2_overlap")


def _apply_shared_expert_overlap(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if "multistream_overlap_shared_expert" in ctx.caller_explicit:
        return
    if int(ctx.parallel.get("ep", 1)) > 1:
        rt["multistream_overlap_shared_expert"] = True
        meta = dict(rt.get("metadata") or {})
        meta["multistream_overlap_shared_expert"] = True
        rt["metadata"] = meta


def _is_mxfp_moe(ctx: MechanismContext) -> bool:
    q = ctx.model.quant
    if q is None:
        return False
    expert_fmt = str(getattr(q, "expert_weight_format", "") or "").lower()
    weight_fmt = str(getattr(q, "weight_format", "") or "").lower()
    if expert_fmt in MXFP_EXPERT_WEIGHT_FORMATS or weight_fmt in MXFP_EXPERT_WEIGHT_FORMATS:
        return True
    overrides = getattr(q, "category_overrides", ()) or ()
    if any(str(fmt).lower() in MXFP_EXPERT_WEIGHT_FORMATS for _, fmt in overrides):
        return True
    return (
        q.activation_format == "fp8_e4m3"
        and q.activation_group_size == MX_SCALE_GROUP_SIZE
        and q.activation_scale_format in {"fp8_e8m0", "e8m0"}
    )


def _update_meta(rt: dict[str, Any], **entries: Any) -> dict[str, Any]:
    meta = dict(rt.get("metadata") or {})
    meta.update(entries)
    rt["metadata"] = meta
    return meta


def _update_additional(rt: dict[str, Any], **entries: Any) -> None:
    meta = dict(rt.get("metadata") or {})
    additional = dict(meta.get("additional_config") or {})
    additional.update(entries)
    meta["additional_config"] = additional
    rt["metadata"] = meta


def _apply_torch_compile_support(rt: dict[str, Any], ctx: MechanismContext) -> None:
    rt["enable_torch_compile"] = True
    meta = dict(rt.get("metadata") or {})
    meta["enable_torch_compile"] = True
    policy = dict(meta.get("execution_policy") or {})
    policy["enable_torch_compile"] = True
    meta["execution_policy"] = policy
    rt["metadata"] = meta
    if _has_quant_activations(ctx.model):
        _append_passes(rt, "activation_quantization", "norm_quant")


def _apply_fp32_residual_norm(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if _has_quant_activations(ctx.model):
        _append_passes(rt, "activation_quantization", "norm_quant")
    _append_passes(rt, "residual_norm")


def _apply_gdn_chunk32(rt: dict[str, Any], ctx: MechanismContext) -> None:
    tp = int(ctx.parallel.get("tp", 1))
    if not model_supports_gdn_chunk32(ctx.model, tp):
        return
    policy = dict(rt.get("attention_policy") or {})
    policy["recurrent_chunk"] = GDN_CHUNK32_TOKENS
    rt["attention_policy"] = policy
    _set_lowering(rt, "gdn_prefill", "gdn_prefill_arch35")


def _ep_within_ub_domain(ctx: MechanismContext) -> bool:
    ep = int(ctx.parallel.get("ep", 1))
    rpn = int((ctx.options or {}).get("ranks_per_node", 0) or 0)
    return ep > 1 and (rpn <= 0 or ep <= rpn)


def _apply_cann_megamoe(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if "moe_comm_mode" in ctx.caller_explicit or "enable_fused_mc2" in ctx.caller_explicit:
        return
    if not _ep_within_ub_domain(ctx):
        return
    if int(ctx.parallel.get("redundant_experts_per_rank", 0) or 0) > 0:
        return
    has_latent_or_situ = any(
        p.ffn == "moe" and (p.expert_latent_size > 0 or p.ffn_activation == "situ")
        for p in ctx.model.layer_plans
    )
    default_mode = FUSED_MC2_ROUTED_MODE if has_latent_or_situ else FUSED_MC2_SHARED_MODE
    level = int(ctx.options.get("enable_fused_mc2", default_mode))
    _update_meta(
        rt,
        moe_comm_mode="dynamic",
        moe_mega_moe_supported=1,
        arch_family=ctx.arch.upper(),
    )
    _update_additional(rt, enable_mega_moe=1, enable_fused_mc2=level)


def _apply_qsa_gbsa_fusion(rt: dict[str, Any], ctx: MechanismContext) -> None:
    phase = str((ctx.options or {}).get("phase") or "decode")
    if phase == "decode":
        _append_passes(rt, "qkv_norm_rope")
    _append_passes(rt, "indexer_online_topk")
    _set_layout_strided(rt, "gdn_gating", "beta", "alpha")
    _set_layout_strided(rt, "short_convolution", "input")
    _update_meta(rt, qsa_gbsa_fused=True, qsa_mrope_decode_max_rows=QSA_MROPE_DECODE_MAX_ROWS)


def _apply_text_gated_residual(rt: dict[str, Any], ctx: MechanismContext) -> None:
    _apply_mhc_fusion_overlap(rt, ctx)
    phase = str((ctx.options or {}).get("phase") or "")
    if phase != "decode":
        _update_meta(rt, text_gated_residual=True, text_gated_residual_min_rows=TEXT_GATED_RESIDUAL_MIN_ROWS)


def _apply_msa_score_v2(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _append_passes(rt, "indexer_online_topk")
    _update_meta(rt, msa_index_score_v2=True)


def _apply_step5_qkvg_indexer(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _append_passes(rt, "qkv_norm_rope", "indexer_online_topk")
    _update_meta(rt, flash_attn_npu_3=True, packed_qkvg_indexer=True)


def _apply_dspark_static_kernel(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _update_additional(rt, enable_static_kernel=True, enable_super_kernel=True)


def _apply_dsv41_epilogue(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _set_lowering(rt, "compressor", "vllm_ascend_cann_compressor")
    _update_meta(rt, dsv41_epilogue_engram_overlap=True)


def _apply_in_kernel_gdn_decode_gate(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _set_layout_strided(rt, "gdn_gating", "beta", "alpha")
    _update_meta(rt, skip_gdn_output_zero=True)


def _apply_kimi_dspark_c8(rt: dict[str, Any], ctx: MechanismContext) -> None:
    _apply_shared_expert_overlap(rt, ctx)
    _update_meta(rt, kimi_dspark_c8_mla=True)


def _has_any_local_tp(parallel: Any) -> bool:
    return any(
        int(parallel.get(axis, 1) or 1) > 1
        for axis in (
            "q_proj_tp", "o_proj_tp", "linear_proj_tp", "dcp_linear_tp",
            "mlp_tp", "shared_expert_tp", "lm_head_tp",
        )
    )


def _parallel_cp_mechanisms() -> tuple[OptimizationMechanism, ...]:
    """Generalized Parallelism, Context-Parallel (DCP/PCP), and Local-TP mechanisms."""
    from mindie_sim.feature_mechanisms import OptimizationMechanism

    return (
        OptimizationMechanism(
            id="universal_dcp_query_gather", category="context_parallel",
            phases=("decode",),
            summary="Universal Q AllGather across DCP/TP/DP cohort conserving 100% attention FLOPs",
            eligible=lambda ctx: (
                int(ctx.parallel.get("cp", ctx.parallel.get("dcp", 1)) or 1) > 1
                and any(p.attention in {"mla", "gqa"} for p in ctx.model.layer_plans)
            ),
            apply=lambda rt, _ctx: _update_meta(rt, universal_dcp_query_gather=True),
        ),
        OptimizationMechanism(
            id="sfa_dcp_sharded_indexer", category="context_parallel",
            phases=("decode",),
            summary="Optional DCP-sharded LightningIndexer KV cache reducing indexer HBM by 1/dcp",
            eligible=lambda ctx: (
                int(ctx.parallel.get("cp", ctx.parallel.get("dcp", 1)) or 1) > 1
                and any(p.attention == "mla" and p.indexer for p in ctx.model.layer_plans)
            ),
            apply=lambda rt, _ctx: _update_meta(rt, enable_sfa_dcp_sharded_indexer=True),
        ),
        OptimizationMechanism(
            id="local_projection_tp", category="parallelism",
            phases=("prefill", "decode"),
            summary="Composable per-sublayer Local-TP across Q/O/linear/MLP/shared-expert/lm_head",
            eligible=lambda ctx: _has_any_local_tp(ctx.parallel),
            apply=lambda rt, _ctx: _update_meta(rt, local_projection_tp_active=True),
        ),
        OptimizationMechanism(
            id="pcp_o_proj_weight_sharding", category="context_parallel",
            phases=("prefill", "decode"),
            summary="Shard o_proj weights across PCP ranks with async weight AllGather on cp_weight_stream",
            eligible=lambda ctx: (
                int(ctx.parallel.get("pcp", 1) or 1) > 1
                and any(p.attention in {"mla", "gqa"} for p in ctx.model.layer_plans)
            ),
            apply=lambda rt, ctx: _update_meta(
                rt,
                pcp_o_proj_weight_sharding_supported=True,
                **({"enable_pcp_o_proj_weight_sharding": True}
                   if ctx.options.get("enable_pcp_o_proj_weight_sharding") else {}),
            ),
        ),
        OptimizationMechanism(
            id="pcp_decode_request_sharding", category="context_parallel",
            phases=("decode",),
            summary="Shard independent decode requests across PCP lanes when DCP=1",
            eligible=lambda ctx: (
                int(ctx.parallel.get("pcp", 1) or 1) > 1
                and int(ctx.parallel.get("cp", ctx.parallel.get("dcp", 1)) or 1) == 1
            ),
            apply=lambda rt, _ctx: _update_meta(rt, pcp_decode_request_sharding=True),
        ),
        OptimizationMechanism(
            id="kda_pcp_head_sharding", category="context_parallel",
            phases=("prefill", "decode"),
            summary="Shard KDA recurrent heads across TP*PCP with prefill sequence-to-head AllGather/Scatter",
            eligible=lambda ctx: (
                int(ctx.parallel.get("pcp", 1) or 1) > 1
                and any(p.attention == "kda" for p in ctx.model.layer_plans)
            ),
            apply=lambda rt, _ctx: _update_meta(rt, kda_pcp_head_sharding=True),
        ),
        OptimizationMechanism(
            id="pcp_w8a8_quantized_allgather", category="context_parallel",
            phases=("prefill", "decode"),
            summary="INT8/FP8 quantized PCP activation AllGather halving inter-rank wire volume",
            eligible=lambda ctx: (
                int(ctx.parallel.get("pcp", 1) or 1) > 1
                and _has_quant_activations(ctx.model)
            ),
            apply=lambda rt, _ctx: _update_meta(rt, pcp_w8a8_quantized_allgather=True),
        ),
        OptimizationMechanism(
            id="quantized_allreduce", category="parallelism",
            phases=("prefill", "decode"),
            summary="Opt-in INT8 quantized TP AllReduce on A2 (BF16, 2<=TP<=8, >=16M elems)",
            eligible=lambda ctx: (
                ctx.arch == "a2"
                and 1 < int(ctx.parallel.get("tp", 1) or 1) <= QUANTIZED_ALLREDUCE_MAX_WORLD
                and str(ctx.model.activation_dtype).strip().lower() in {"bf16", "bfloat16"}
                and not (
                    isinstance(ctx.options.get("quantized_all_reduce_config"), dict)
                    and "enable" in ctx.options["quantized_all_reduce_config"]
                    and str(ctx.options["quantized_all_reduce_config"]["enable"]).strip().lower()
                    in {"false", "0", "no", "off", ""}
                )
                and bool(
                    ctx.options.get("explicit_opt_in")
                    or ctx.options.get("quantized_allreduce")
                    or (isinstance(ctx.options.get("quantized_all_reduce_config"), dict)
                        and bool(ctx.options["quantized_all_reduce_config"].get("enable", False)))
                )
            ),
            apply=lambda rt, _ctx: _update_meta(rt, quantized_allreduce=True),
        ),
        OptimizationMechanism(
            id="dsv41_pcp_c2_ring_gather", category="context_parallel",
            phases=("prefill",),
            summary="V4.1 PCP C2 ring-tail compressor boundary gather and restore unpermute",
            eligible=lambda ctx: (
                int(ctx.parallel.get("pcp", 1) or 1) > 1
                and any(
                    p.cache_kind == "window_compressed" and p.compressor_kind == "gated_ring"
                    for p in ctx.model.layer_plans
                )
            ),
            apply=lambda rt, _ctx: _update_meta(rt, dsv41_pcp_c2_ring_gather=True),
        ),
    )


def source_mechanisms() -> tuple[OptimizationMechanism, ...]:
    """Build the audited mechanism specs in registration order."""
    from mindie_sim.feature_mechanisms import OptimizationMechanism

    return (
        OptimizationMechanism(
            id="torch_compile_support", category="compilation", phases=("prefill", "decode"),
            summary="Enable torch.compile / npugraph_ex FX fusion passes on models with do_not_compile=True",
            eligible=lambda ctx: dict(ctx.model.catalog_facts).get("supports_torch_compile") is False,
            apply=_apply_torch_compile_support,
        ),
        OptimizationMechanism(
            id="fp32_residual_norm_fusion", category="norm_fusion", phases=("prefill", "decode"),
            summary="Fuse FP32 residual accumulate + (1+w) RMSNorm + BF16 cast into one AIV kernel",
            eligible=lambda ctx: bool((dict(ctx.model.catalog_facts).get("normalization") or {}).get("fp32_residual")),
            apply=_apply_fp32_residual_norm,
        ),
        OptimizationMechanism(
            id="gdn_chunk32_prefill", category="recurrent", phases=("prefill",),
            summary="Arch35 fla-npu Chunk32 GDN prefill composite for Hv/Hk in {1,2,3,4}",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and model_supports_gdn_chunk32(ctx.model, int(ctx.parallel.get("tp", 1)))
            ),
            apply=_apply_gdn_chunk32,
        ),
        OptimizationMechanism(
            id="cann_megamoe_fusion", category="moe_comm", phases=("prefill", "decode"),
            summary="CANN 9.3 MegaMoE composite Dispatch+GMM1+ActMxQuant+GMM2+[SharedExpert]+Combine",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and _ep_within_ub_domain(ctx)
                and int(ctx.parallel.get("redundant_experts_per_rank", 0) or 0) == 0
                and any(p.ffn == "moe" for p in ctx.model.layer_plans)
                and _is_mxfp_moe(ctx)
            ),
            apply=_apply_cann_megamoe,
        ),
        OptimizationMechanism(
            id="kv_quant_sfa_scale_precompute", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Precompute A5 C8 FP8 scale addresses for KvQuantSparseFlashAttentionV2",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.attention == "mla" and p.indexer for p in ctx.model.layer_plans)
            ),
            apply=lambda rt, _ctx: _update_meta(rt, kv_quant_sfa_scale_precompute=True),
        ),
        OptimizationMechanism(
            id="msa_index_score_v2_topk", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Fused return_choice=1 in-kernel TopK inside CANN 9.3 MsaIndexScoreV2 (+ MXFP4 A4C4)",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.attention == "gqa" and p.indexer and p.candidate_block_size >= MSA_MIN_BLOCK_SIZE
                        for p in ctx.model.layer_plans)
            ),
            apply=_apply_msa_score_v2,
        ),
        OptimizationMechanism(
            id="qsa_gbsa_prologue_fusion", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Fused QSA QKV/RMSNorm/MRoPE/compress prologue + GBSA([1,4]) metadata reuse",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.attention == "gqa" and p.indexer and getattr(p, "hc_lowrank", 0) > 0
                        for p in ctx.model.layer_plans)
            ),
            apply=_apply_qsa_gbsa_fusion,
        ),
        OptimizationMechanism(
            id="text_gated_residual_fusion", category="mhc_overlap",
            phases=("prefill",),
            summary="CANN 9.3 native text_gated_residual (M > 512) replacing 5-op Triton Low-Rank HC chain",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(getattr(p, "hc_lowrank", 0) > 0 for p in ctx.model.layer_plans)
            ),
            apply=_apply_text_gated_residual,
        ),
        OptimizationMechanism(
            id="tp_mc2_overlap", category="parallelism", phases=("prefill", "decode"),
            summary="AllGatherMatmulV2 + MatmulReduceScatterV2 overlapping TP collectives with Cube GEMMs",
            eligible=lambda ctx: int(ctx.parallel.get("tp", 1)) > 1,
            apply=_apply_tp_mc2_overlap,
        ),
        OptimizationMechanism(
            id="step5_qkvg_indexer_fusion", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Packed SWA qkvg + Triton indexer_qkz_norm_rope + CANN 9.3 FA3 hd=192 SWA",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.q_output_gate_kind == "headwise" and p.indexer
                        for p in ctx.model.layer_plans)
            ),
            apply=_apply_step5_qkvg_indexer,
        ),
        OptimizationMechanism(
            id="dspark_static_kernel_allowlist", category="compilation", phases=("decode",),
            summary="Allow enable_static_kernel and enable_super_kernel when speculative method is dspark",
            eligible=lambda ctx: bool(
                ctx.model.auxiliary is not None
                and (ctx.model.auxiliary.dspark is not None
                     or (ctx.model.auxiliary.mtp is not None
                         and ctx.model.auxiliary.mtp.kind == "dspark"))
            ),
            apply=_apply_dspark_static_kernel,
        ),
        OptimizationMechanism(
            id="dsv41_indexer_epilogue_fusion", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Fused DSV4.1 indexer prologue, MQSMLA epilogue, C1-only DSpark share & Engram overlap",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.mla_variant == "latent_mqa" and p.compressor_kind == "gated_ring"
                        for p in ctx.model.layer_plans)
            ),
            apply=_apply_dsv41_epilogue,
        ),
        OptimizationMechanism(
            id="in_kernel_gdn_decode_gate", category="recurrent", phases=("decode",),
            summary="Fold GDN gating/L2Norm into recurrent_gated_delta_rule and skip output buffer zeroing",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.attention == "gdn" for p in ctx.model.layer_plans)
            ),
            apply=_apply_in_kernel_gdn_decode_gate,
        ),
        OptimizationMechanism(
            id="kimi_dspark_c8_mla", category="moe_overlap", phases=("prefill", "decode"),
            summary="5L MLA DSpark C8 QuantFlashMLA + Kimi Latent-MoE norm/quant & full-width shared overlap",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.expert_latent_size > 0 for p in ctx.model.layer_plans)
            ),
            apply=_apply_kimi_dspark_c8,
        ),
        OptimizationMechanism(
            id="keypool_int8_indexer", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Dynamic INT8 quant_mode=2 pool_key_indexer + long-context sparse_flash_mla plan fix",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.attention == "mla" and p.index_kpool > 1 for p in ctx.model.layer_plans)
            ),
            apply=lambda rt, _ctx: _update_meta(rt, keypool_int8_supported=True),
        ),
        *_parallel_cp_mechanisms(),
    )


from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec

MX_SCALE_PACK_FACTOR = 2
GDN_DEFAULT_CHUNK_SIZE = 64
MAX_RECURRENT_DECODE_QUERY = 8
SERIAL_CONTEXT_DEGREE = 1


@dataclass(frozen=True)
class MechanismContext:
    model: ModelSpec
    sku: str
    arch: str
    parallel: Mapping[str, int]
    caller_explicit: frozenset[str]
    options: Mapping[str, Any]


@dataclass(frozen=True)
class OptimizationMechanism:
    """Framework-neutral physical optimization mechanism."""

    id: str
    category: str
    phases: tuple[str, ...]
    summary: str
    eligible: Callable[[MechanismContext], bool]
    apply: Callable[[dict[str, Any], MechanismContext], None]


def _torch_compile_active(rt: Mapping[str, Any], model: ModelSpec) -> bool:
    if "enable_torch_compile" in rt:
        return bool(rt["enable_torch_compile"])
    meta = dict(rt.get("metadata") or {})
    if "enable_torch_compile" in meta:
        return bool(meta["enable_torch_compile"])
    facts = dict(model.catalog_facts)
    if facts.get("supports_torch_compile") is False:
        return False
    return True


def _apply_aclgraph(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if "graph_capture_mode" in ctx.caller_explicit:
        return
    from mindie_sim.compose import capture_keys

    rt["graph_mode"] = "full"
    rt["graph_capture_mode"] = "full"
    meta = dict(rt.get("metadata") or {})
    draft = int(meta.get("draft_proposal_tokens", 0) or 0)
    rt["graph_capture_keys"] = capture_keys(int(rt["max_num_seqs"]), draft)
    policy = dict(meta.get("execution_policy") or {})
    policy["graph_capture_mode"] = "full"
    compile_on = _torch_compile_active(rt, ctx.model)
    policy.setdefault("enable_torch_compile", compile_on)
    meta.setdefault("enable_torch_compile", compile_on)
    meta["execution_policy"] = policy
    rt["metadata"] = meta


def _apply_sp(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if "sp" in ctx.caller_explicit:
        return
    from mindie_sim.prediction.sharding.sequence_parallel import DECLARATION, _validate_model
    tp = int(ctx.parallel.get("tp", 1))
    ep = int(ctx.parallel.get("ep", 1))
    pp = int(ctx.parallel.get("pp", 1))
    pcp = int(ctx.parallel.get("pcp", 1))
    if pcp >= 1 and pp >= 1 and tp > 1 and ep >= tp and ep % tp == 0:
        try:
            _validate_model(ctx.model, padding="right_to_tp")
        except ModelingError:
            return
        kernels = dict(rt.get("kernels") or {})
        kernels.setdefault("sequence_parallel", dict(DECLARATION, padding="right_to_tp"))
        rt["kernels"] = kernels
        meta = dict(rt.get("metadata") or {})
        meta["sp"] = True
        rt["metadata"] = meta


def _apply_basic_norm_quant(rt: dict[str, Any], ctx: MechanismContext) -> None:
    compile_on = _torch_compile_active(rt, ctx.model)
    if _has_quant_activations(ctx.model):
        _append_passes(rt, "activation_quantization")
        if compile_on:
            _append_passes(rt, "norm_quant")
    if not compile_on:
        return
    normalization = dict(ctx.model.catalog_facts).get("normalization") or {}
    # ``add_rmsnorm`` is the residual boundary and is independent of qk_norm.
    if (ctx.model.activation_dtype == "bf16"
            and not normalization.get("norm_weight_offset", 0.0)):
        _append_passes(rt, "residual_norm")
        _set_lowering(rt, "add_rmsnorm", "vllm_ascend_native_norm")
        q = ctx.model.quant
        if q is not None and _has_quant_activations(ctx.model):
            is_mx = (q.activation_format == "fp8_e4m3"
                     and q.activation_group_size == MX_SCALE_GROUP_SIZE
                     and q.activation_scale_format == "fp8_e8m0"
                     and q.activation_scale_pack == MX_SCALE_PACK_FACTOR)
            if q.activation_format == "int8" or is_mx:
                _set_lowering(rt, "add_rmsnorm_quant", "vllm_ascend_native_norm")
            if is_mx:
                _set_lowering(rt, "rmsnorm_quant", "vllm_ascend_native_norm")


def _apply_unit_offset_norm(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if _has_quant_activations(ctx.model):
        _append_passes(rt, "activation_quantization", "norm_quant")
    _append_passes(rt, "residual_norm")


NORM_QUANT_CONSUMER_FAMILIES = (
    "gemm", "qkv_projection", "gate_up_projection", "projection",
    "latent_projection", "router", "index_projection", "logits",
)


def norm_quant_domain() -> dict[str, Any]:
    """One declared norm+quant fusion domain with a materialized BF16 output."""
    return {
        "consumers": [{"family": fam} for fam in NORM_QUANT_CONSUMER_FAMILIES],
        "external_outputs": "materialize",
    }


def ensure_norm_quant_domain(kernels: dict[str, Any]) -> dict[str, Any]:
    """Install the shared norm+quant domain when its pass is declared."""
    domains = dict(kernels.get("fusion_domains") or {})
    domains.setdefault("norm_quant", norm_quant_domain())
    kernels["fusion_domains"] = domains
    return kernels


def _apply_norm_quant_materialize_all(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if not _has_quant_activations(ctx.model):
        return
    _append_passes(rt, "activation_quantization", "norm_quant", "residual_norm")
    kernels = ensure_norm_quant_domain(dict(rt.get("kernels") or {}))
    rt["kernels"] = kernels


def _apply_mla_prolog(rt: dict[str, Any], ctx: MechanismContext) -> None:
    from mindie_sim.prediction.operators.mla_prolog.family_mla_prolog import validate as validate_mla_prolog
    from mindie_sim.schema.parallel import ParallelSpec
    mla_layers = [p for p in ctx.model.layer_plans if p.attention == "mla" and p.q_lora_rank > 0]
    if not mla_layers:
        return
    p_cfg = ParallelSpec(
        dp=int(ctx.parallel.get("dp", 1)), tp=int(ctx.parallel.get("tp", 1)),
        pp=int(ctx.parallel.get("pp", 1)), ep=int(ctx.parallel.get("ep", 1)),
        cp=int(ctx.parallel.get("cp", ctx.parallel.get("dcp", 1))),
        pcp=int(ctx.parallel.get("pcp", 1)),
    )
    passes = tuple(rt.get("lowering_passes") or ())
    options: dict[str, Any] = {}
    quant = ctx.model.quant
    if ctx.arch == "a5" and quant is not None and quant.kv_dtype == "int8":
        from mindie_sim.prediction.operators.mla_prolog.mla_prolog_packed import LAYOUT
        options["cache_layout"] = LAYOUT
    try:
        for plan in mla_layers:
            validate_mla_prolog(plan=plan, parallel=p_cfg, model=ctx.model,
                                quant=quant, lowering_passes=passes, **options)
        _set_lowering(rt, "mla_prolog", (
            {"implementation": "vllm_ascend_mla_prolog_v3", "options": options}
            if options else "vllm_ascend_mla_prolog_v3"))
    except ModelingError:
        return


def _apply_attn_res(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _set_lowering(rt, "attention_residual", "vllm_ascend_attn_res_prefix")


def _apply_recurrent_decode(rt: dict[str, Any], ctx: MechanismContext) -> None:
    is_a5 = ctx.arch == "a5"
    cp = int(ctx.parallel.get("cp", ctx.parallel.get("dcp", 1)))
    pcp = int(ctx.parallel.get("pcp", 1))
    from mindie_sim.catalog.paired_sfa import execution
    kda_layers = [p for p in ctx.model.layer_plans if p.attention == "kda"]
    if (kda_layers
            and pcp >= SERIAL_CONTEXT_DEGREE
            and (cp == SERIAL_CONTEXT_DEGREE
                 or execution(ctx.model).get("recurrent_context_sharded") is False)):
        kda_options = {}
        if is_a5:
            from mindie_sim.prediction.operators.kda.kda_source import resolve as resolve_kda_source
            bounds = {getattr(p, "kda_gate_lower_bound", None) for p in kda_layers}
            if len(bounds) == 1:
                kda_options = resolve_kda_source(
                    kda_layers[0], arch="A5", phase="decode", kernels={},
                    activation_dtype=ctx.model.activation_dtype) or {}
        _set_lowering(rt, "kda", {
            "cases": [{
                "domain": {
                    "phases": ["decode"],
                    "max_query_tokens": {"min": 1, "max": MAX_RECURRENT_DECODE_QUERY},
                },
                "binding": {
                    "implementation": ("vllm_ascend_recurrent_kda_arch35" if is_a5
                                       else "vllm_ascend_recurrent_kda_arch22"),
                    "options": kda_options,
                },
            }],
            "otherwise": "analytical",
        })
    if any(p.attention == "gdn" for p in ctx.model.layer_plans) and pcp >= 1:
        opts = {"recurrent_implementation": "arch35_fp32"} if is_a5 else {}
        binding = {"implementation": "vllm_ascend_gdn_decode", "options": opts}
        _set_lowering(rt, "gdn_decode", {
            "cases": [
                {"domain": {"phases": ["decode"], "min_query_tokens": {"min": 1},
                            "max_query_tokens": {"max": 1}},
                 "binding": binding},
                {"domain": {"phases": ["decode"], "min_query_tokens": {"min": 2},
                            **({"max_query_tokens": {"max": MAX_RECURRENT_DECODE_QUERY}} if is_a5 else {})},
                 "binding": binding},
            ],
            "otherwise": "analytical",
        })


def _apply_gdn_zero_copy(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _set_layout_strided(rt, "gdn_gating", "beta", "alpha")
    _set_layout_strided(rt, "short_convolution", "input")


def _apply_packed_proj_zero_copy(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _set_layout_strided(rt, "gdn_gating", "beta", "alpha")
    _set_layout_strided(rt, "short_convolution", "input")
    _set_layout_strided(rt, "kda_recurrent", "beta")


def _apply_gdn_chunk64(rt: dict[str, Any], ctx: MechanismContext) -> None:
    from mindie_sim.prediction.operators.linear_attention.gdn_domain import model_supports_gdn_chunk64

    tp = int(ctx.parallel.get("tp", 1))
    policy = dict(rt.get("attention_policy") or {})
    policy.setdefault("recurrent_chunk", GDN_DEFAULT_CHUNK_SIZE)
    rt["attention_policy"] = policy
    if model_supports_gdn_chunk64(ctx.model, tp):
        _set_lowering(rt, "gdn_prefill", "gdn_prefill_arch35")


def _apply_recover_ssm(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if "recover_ssm" in ctx.caller_explicit:
        return
    meta = dict(rt.get("metadata") or {})
    if int(meta.get("draft_proposal_tokens", 0) or 0) > 0:
        rt["recover_ssm"] = True


def _apply_dynamic_mc2(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if "moe_comm_mode" in ctx.caller_explicit:
        return
    if int(ctx.parallel.get("ep", 1)) > 1:
        meta = dict(rt.get("metadata") or {})
        meta["moe_comm_mode"] = "dynamic"
        meta.setdefault("arch_family", ctx.arch.upper())
        rt["metadata"] = meta


MECHANISM_CATALOG: dict[str, OptimizationMechanism] = {}


def register_optimization_mechanism(spec: OptimizationMechanism) -> OptimizationMechanism:
    """Register or replace a stack-agnostic physical optimization mechanism."""
    if not isinstance(spec, OptimizationMechanism) or not spec.id.strip():
        raise ModelingError("invalid_optimization_mechanism_spec")
    MECHANISM_CATALOG[spec.id] = spec
    return spec


def list_optimization_mechanisms() -> tuple[OptimizationMechanism, ...]:
    return tuple(MECHANISM_CATALOG[key] for key in sorted(MECHANISM_CATALOG))


for _item in (
    OptimizationMechanism(
        id="aclgraph_decode_capture", category="graph_capture", phases=("decode",),
        summary="Full device graph capture & replay eliminating host launch gaps in decode",
        eligible=lambda ctx: ctx.model.layers > 0,
        apply=_apply_aclgraph,
    ),
    OptimizationMechanism(
        id="sequence_parallel", category="parallelism", phases=("prefill", "decode"),
        summary="TP sequence-parallel norm/residual/router token sharding (RS + AG)",
        eligible=lambda ctx: int(ctx.parallel.get("tp", 1)) > 1,
        apply=_apply_sp,
    ),
    OptimizationMechanism(
        id="add_rmsnorm_quant_fusion", category="norm_fusion", phases=("prefill", "decode"),
        summary="Fused AddRmsNorm and RmsNormDynamicQuant on standard RMSNorm layers",
        eligible=lambda ctx: ctx.model.layers > 0,
        apply=_apply_basic_norm_quant,
    ),
    OptimizationMechanism(
        id="unit_offset_rmsnorm_fusion", category="norm_fusion", phases=("prefill", "decode"),
        summary="Load-time (1+w) gamma folding enabling AddRmsNorm+Quant on Gemma/Qwen norms",
        eligible=lambda ctx: bool((dict(ctx.model.catalog_facts).get("normalization") or {}).get("norm_weight_offset")),
        apply=_apply_unit_offset_norm,
    ),
    OptimizationMechanism(
        id="add_rmsnorm_quant_materialize_all", category="norm_fusion", phases=("prefill", "decode"),
        summary="Dual-output AddRmsNorm+Quant materializing BF16 norm for router/shared branches",
        eligible=lambda ctx: _has_quant_activations(ctx.model),
        apply=_apply_norm_quant_materialize_all,
    ),
    OptimizationMechanism(
        id="mla_prolog_fusion", category="attention_fusion", phases=("prefill", "decode"),
        summary="Fused MLA Prolog (q_a/kv_a norm, RoPE, KV cache write)",
        eligible=lambda ctx: any(p.attention == "mla" and p.q_lora_rank > 0 for p in ctx.model.layer_plans),
        apply=_apply_mla_prolog,
    ),
    OptimizationMechanism(
        id="attn_res_prefix_fusion", category="residual_fusion", phases=("prefill", "decode"),
        summary="Fused AttentionResidual prefix accumulator (AttnResFwd)",
        eligible=lambda ctx: any(p.residual_block_size > 0 for p in ctx.model.layer_plans),
        apply=_apply_attn_res,
    ),
    OptimizationMechanism(
        id="recurrent_decode_native_kernels", category="recurrent", phases=("decode",),
        summary="Native fused KDA and GDN single/speculative decode recurrence kernels",
        eligible=lambda ctx: any(p.attention in {"kda", "gdn"} for p in ctx.model.layer_plans),
        apply=_apply_recurrent_decode,
    ),
    OptimizationMechanism(
        id="mhc_boundary_fusion_overlap", category="mhc_overlap", phases=("prefill", "decode"),
        summary="Fuse HcPost+HcPreMix+RMSNorm sublayer boundary and overlap HcPre/HC-injection on aux stream",
        eligible=lambda ctx: any(p.hyper_connections or getattr(p, "hc_lowrank", 0) > 0 for p in ctx.model.layer_plans),
        apply=_apply_mhc_fusion_overlap,
    ),
    OptimizationMechanism(
        id="gdn_strided_gate_zero_copy", category="layout_zero_copy", phases=("prefill", "decode"),
        summary="Direct strided BA/QKV addressing in GDN gating and causal conv without Slice copies",
        eligible=lambda ctx: any(p.attention == "gdn" for p in ctx.model.layer_plans),
        apply=_apply_gdn_zero_copy,
    ),
    OptimizationMechanism(
        id="packed_projection_zero_copy", category="layout_zero_copy", phases=("prefill", "decode"),
        summary="Zero-copy strided views across all packed GDN and KDA input projections",
        eligible=lambda ctx: any(p.attention in {"gdn", "kda"} for p in ctx.model.layer_plans),
        apply=_apply_packed_proj_zero_copy,
    ),
    OptimizationMechanism(
        id="gdn_chunk64_prefill", category="recurrent", phases=("prefill",),
        summary="Fused Chunk64 WY+FWD_H+FWD_O GDN prefill composite kernel",
        # The composite is the arch35 (A5) GDN caller; A2/A3 keep the
        # analytical chunk decomposition because the AIV rate table for that
        # kernel does not exist on those SKUs.
        eligible=lambda ctx: (ctx.arch == "a5"
                              and any(p.attention == "gdn" for p in ctx.model.layer_plans)),
        apply=_apply_gdn_chunk64,
    ),
    OptimizationMechanism(
        id="gdn_projection_multistream_overlap", category="recurrent_overlap", phases=("decode",),
        summary="Auxiliary-stream BA projection overlapped with QKVZ projection in GDN decode",
        eligible=lambda ctx: any(p.attention == "gdn" for p in ctx.model.layer_plans),
        apply=lambda rt, _ctx: _append_passes(rt, "gdn_projection_overlap"),
    ),
    OptimizationMechanism(
        id="kda_in_kernel_l2norm", category="recurrent", phases=("prefill",),
        summary="Fold Q/K L2 normalization into chunk KDA prefill kernel",
        eligible=lambda ctx: any(p.attention == "kda" for p in ctx.model.layer_plans),
        apply=lambda rt, _ctx: _append_passes(rt, "kda_in_kernel_l2norm"),
    ),
    OptimizationMechanism(
        id="fused_recover_ssm", category="speculative_state", phases=("decode",),
        summary="Compact RecoverSSM speculative verification records + Triton accepted-prefix commit",
        eligible=lambda ctx: any(p.attention in {"gdn", "kda"} for p in ctx.model.layer_plans),
        apply=_apply_recover_ssm,
    ),
    OptimizationMechanism(
        id="qkv_norm_rope_fusion", category="attention_fusion", phases=("prefill", "decode"),
        summary="Fused QKV split + QK RMSNorm + MRoPE single-launch preparation kernel",
        eligible=lambda ctx: any(p.qk_norm for p in ctx.model.layer_plans),
        apply=lambda rt, _ctx: _append_passes(rt, "qkv_norm_rope"),
    ),
    OptimizationMechanism(
        id="indexer_online_ub_topk", category="attention_fusion",
        phases=("prefill", "decode"),
        summary="On-chip UB online TopK score reduction eliminating scoreGm HBM round-trip",
        # Analytical ideal only: a native source binary owns its score
        # workspace, so this experiment is never installed by a merged/PR
        # runtime track.
        eligible=lambda ctx: any(p.index_topk > 0 for p in ctx.model.layer_plans),
        apply=lambda rt, _ctx: _append_passes(rt, "indexer_online_topk"),
    ),
    OptimizationMechanism(
        id="activation_quant_fusion", category="ffn_fusion", phases=("prefill", "decode"),
        summary="Fused SwiGLU/SiTU/GeGLU + dynamic activation quantization kernel",
        eligible=lambda ctx: _has_quant_moe_activations(ctx.model),
        apply=lambda rt, _ctx: _append_passes(rt, "activation_quantization", "activation_quant"),
    ),
    OptimizationMechanism(
        id="gmm_swiglu_quant_fusion", category="ffn_fusion", phases=("prefill", "decode"),
        summary="On-chip AIC->UB GroupedMatmul + SwiGLU/SiTU + Quant eliminating gate_up HBM round-trip",
        eligible=lambda ctx: (any(p.ffn == "moe" for p in ctx.model.layer_plans)
                              and _has_quant_moe_activations(ctx.model)),
        apply=lambda rt, _ctx: _append_passes(rt, "activation_quantization", "gmm_swiglu_quant"),
    ),
    OptimizationMechanism(
        id="moe_tail_residual_fusion", category="ffn_fusion", phases=("prefill", "decode"),
        summary="Plain/fused 3-way shared+routed+residual add eliminating ForeachAddList launch",
        eligible=lambda ctx: any(p.ffn == "moe" and p.num_shared_experts > 0 for p in ctx.model.layer_plans),
        apply=lambda rt, _ctx: _append_passes(rt, "moe_tail_residual"),
    ),
    OptimizationMechanism(
        id="shared_expert_multistream_overlap", category="moe_overlap", phases=("prefill", "decode"),
        summary="Dedicated NPU stream overlapping shared-expert GEMMs with routed EP AllToAll/MC2",
        eligible=lambda ctx: any(p.ffn == "moe" and p.num_shared_experts > 0 for p in ctx.model.layer_plans),
        apply=_apply_shared_expert_overlap,
    ),
    OptimizationMechanism(
        id="dynamic_mc2_dispatch", category="moe_comm", phases=("decode",),
        summary="Dynamic token-capacity selection between fused MC2 and AllToAll/AllGather",
        eligible=lambda ctx: any(p.ffn == "moe" for p in ctx.model.layer_plans),
        apply=_apply_dynamic_mc2,
    ),
):
    register_optimization_mechanism(_item)

for _source_item in source_mechanisms():
    register_optimization_mechanism(_source_item)


WATERFALL_SCHEMA = "mindie-sim.feature-waterfall/1"
MONOTONIC_TOLERANCE_MS = 1e-5
DEFAULT_WATERFALL_CONTEXT = 4096
DEFAULT_WATERFALL_SEQUENCES = 8
MS_PRECISION_DIGITS = 6
PCT_PRECISION_DIGITS = 3
RATIO_PRECISION_DIGITS = 4


def _pct_delta(before_ms: float, after_ms: float) -> float:
    if before_ms <= 0.0:
        return 0.0
    return ((after_ms - before_ms) / before_ms) * 100.0


def _speedup(before_ms: float, after_ms: float) -> float:
    if after_ms <= 0.0:
        return 1.0
    return before_ms / after_ms


def _network(analysis: Mapping[str, Any]) -> dict[str, float]:
    """Nominal network efficiency of one level, never a fitted target."""
    return {
        "mbu": round(float(analysis["network_mbu"]), RATIO_PRECISION_DIGITS),
        "cube_mfu": round(float(analysis["network_cube_mfu"]), RATIO_PRECISION_DIGITS),
        "vec_mfu": round(float(analysis["network_vec_mfu"]), RATIO_PRECISION_DIGITS),
        "link_util": round(float(analysis["network_link_util"]), RATIO_PRECISION_DIGITS),
        "roofline_efficiency": round(
            float(analysis["network_roofline_efficiency"]), RATIO_PRECISION_DIGITS),
    }


def _compare_pr_ablations(
    model, hardware, runtime_stack, pr_report, merged_ms, pr_ms, *,
    phase, tokens, context, sequences, scope, compose_kwargs,
):
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.efficiency import analyze_network_efficiency

    pr_ablations: list[dict[str, Any]] = []
    loo_ablations: list[dict[str, Any]] = []
    single_cache: dict[str, float] = {}
    loo_cache: dict[str, float] = {}
    for pr_item in pr_report["unmerged_pr_mechanisms"]:
        mid = pr_item["mechanism_id"]
        if mid not in single_cache:
            single_case = compose_case(
                model, hardware=hardware, runtime_stack=runtime_stack,
                feature_track="merged", optimizations=(mid,),
                _stack_phase=phase, **compose_kwargs,
            )
            single_eff = analyze_network_efficiency(
                single_case, phase=phase, tokens=tokens, context=context,
                sequences=sequences, scope=scope,
            )
            single_cache[mid] = float(single_eff["graph_duration_ms"])
        if mid not in loo_cache:
            loo_case = compose_case(
                model, hardware=hardware, runtime_stack=runtime_stack,
                feature_track="unmerged_pr", exclude_optimizations=(mid,),
                _stack_phase=phase, **compose_kwargs,
            )
            loo_eff = analyze_network_efficiency(
                loo_case, phase=phase, tokens=tokens, context=context,
                sequences=sequences, scope=scope,
            )
            loo_cache[mid] = float(loo_eff["graph_duration_ms"])
        single_ms = single_cache[mid]
        without_ms = loo_cache[mid]
        pr_ablations.append({
            "mechanism_id": mid,
            "ref": pr_item["ref"],
            "notes": pr_item["notes"],
            "step_ms": round(single_ms, MS_PRECISION_DIGITS),
            "delta_vs_merged_ms": round(single_ms - merged_ms, MS_PRECISION_DIGITS),
            "delta_vs_merged_pct": round(_pct_delta(merged_ms, single_ms), PCT_PRECISION_DIGITS),
            "speedup_vs_merged": round(_speedup(merged_ms, single_ms), RATIO_PRECISION_DIGITS),
        })
        loo_ablations.append({
            "mechanism_id": mid,
            "ref": pr_item["ref"],
            "notes": pr_item["notes"],
            "without_pr_step_ms": round(without_ms, MS_PRECISION_DIGITS),
            "delta_vs_without_pr_ms": round(pr_ms - without_ms, MS_PRECISION_DIGITS),
            "delta_vs_without_pr_pct": round(_pct_delta(without_ms, pr_ms), PCT_PRECISION_DIGITS),
            "speedup_vs_without_pr": round(_speedup(without_ms, pr_ms), RATIO_PRECISION_DIGITS),
        })
    pr_ablations.sort(key=lambda row: row["delta_vs_merged_ms"])
    loo_ablations.sort(key=lambda row: row["delta_vs_without_pr_ms"])

    return pr_ablations, loo_ablations


def compare_feature_tracks(
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    *,
    runtime_stack: str | Mapping[str, Any] = "vllm-ascend",
    phase: str = "decode",
    tokens: int = 1,
    context: int = DEFAULT_WATERFALL_CONTEXT,
    sequences: int = DEFAULT_WATERFALL_SEQUENCES,
    scope: str = "body",
    ablate_prs: bool = True,
    **compose_kwargs: Any,
) -> dict[str, Any]:
    """Return the complete 4-rung software-to-silicon performance waterfall."""
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.efficiency import analyze_network_efficiency

    cleaned_kwargs = dict(compose_kwargs)
    cleaned_kwargs.pop("feature_track", None)
    cleaned_kwargs.pop("optimizations", None)
    cleaned_kwargs.pop("exclude_optimizations", None)
    cleaned_kwargs.pop("raw_eager", None)
    cleaned_kwargs.pop("_stack_phase", None)

    cases: dict[str, dict[str, Any]] = {}
    analyses: dict[str, dict[str, Any]] = {}
    for track in ("merged", "unmerged_pr", "expected_ideal"):
        case = compose_case(
            model, hardware=hardware, runtime_stack=runtime_stack,
            feature_track=track, _stack_phase=phase, **cleaned_kwargs,
        )
        cases[track] = case
        analyses[track] = analyze_network_efficiency(
            case, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope,
        )

    eager_kwargs = dict(cleaned_kwargs)
    if "quant" not in eager_kwargs:
        va_prof = (cases["merged"]["runtime"]["metadata"] or {}).get("va_profile") or {}
        if va_prof.get("quant") is not None:
            eager_kwargs["quant"] = va_prof["quant"]
    cases["raw_eager"] = compose_case(
        model, hardware=hardware, runtime_stack=runtime_stack,
        feature_track="raw_eager", _stack_phase=phase, **eager_kwargs,
    )
    analyses["raw_eager"] = analyze_network_efficiency(
        cases["raw_eager"], phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope,
    )

    eager_ms = float(analyses["raw_eager"]["graph_duration_ms"])
    merged_ms = float(analyses["merged"]["graph_duration_ms"])
    pr_ms = float(analyses["unmerged_pr"]["graph_duration_ms"])
    ideal_ms = float(analyses["expected_ideal"]["graph_duration_ms"])
    ideal_eff = analyses["expected_ideal"]
    roof_1p0_ms = max(
        float(ideal_eff.get("critical_path_roofline_ms") or 0.0),
        float(ideal_eff.get("physical_resource_cut_ms") or 0.0),
    ) or float(ideal_eff.get("total_roofline_ideal_ms") or 0.0)

    eager_report = cases["raw_eager"]["runtime"]["metadata"]["feature_track_report"]
    merged_report = cases["merged"]["runtime"]["metadata"]["feature_track_report"]
    pr_report = cases["unmerged_pr"]["runtime"]["metadata"]["feature_track_report"]
    ideal_report = cases["expected_ideal"]["runtime"]["metadata"]["feature_track_report"]

    pr_ablations, loo_ablations = [], []
    if ablate_prs:
        pr_ablations, loo_ablations = _compare_pr_ablations(
            model, hardware, runtime_stack, pr_report, merged_ms, pr_ms,
            phase=phase, tokens=tokens, context=context, sequences=sequences,
            scope=scope, compose_kwargs=cleaned_kwargs,
        )

    monotonic = (
        merged_ms + MONOTONIC_TOLERANCE_MS >= pr_ms
        and pr_ms + MONOTONIC_TOLERANCE_MS >= ideal_ms
        and ideal_ms + MONOTONIC_TOLERANCE_MS >= roof_1p0_ms
    )
    eager_monotonic = eager_ms + MONOTONIC_TOLERANCE_MS >= merged_ms

    return {
        "schema": WATERFALL_SCHEMA,
        "model": cases["merged"]["model"]["name"],
        "hardware_sku": cases["merged"]["machine"]["sku"],
        "runtime_stack": merged_report["stack"],
        "repository": merged_report["repository"],
        "geometry": {
            "phase": phase, "tokens": tokens,
            "context": context, "sequences": sequences, "scope": scope,
        },
        "monotonic_verified": bool(monotonic),
        "eager_vs_merged_monotonic": bool(eager_monotonic),
        "waterfall": {
            "raw_eager": {
                "step_ms": round(eager_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["raw_eager"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["raw_eager"]["total_hbm_bytes"]),
                "network": _network(analyses["raw_eager"]),
                "active_mechanisms": [m["id"] for m in eager_report["active_mechanisms"]],
            },
            "current_merged": {
                "step_ms": round(merged_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["merged"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["merged"]["total_hbm_bytes"]),
                "network": _network(analyses["merged"]),
                "delta_vs_raw_eager_ms": round(merged_ms - eager_ms, MS_PRECISION_DIGITS),
                "delta_vs_raw_eager_pct": round(_pct_delta(eager_ms, merged_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_raw_eager": round(_speedup(eager_ms, merged_ms), RATIO_PRECISION_DIGITS),
                "active_mechanisms": [m["id"] for m in merged_report["active_mechanisms"]],
            },
            "current_unmerged_pr": {
                "step_ms": round(pr_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["unmerged_pr"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["unmerged_pr"]["total_hbm_bytes"]),
                "network": _network(analyses["unmerged_pr"]),
                "delta_vs_merged_ms": round(pr_ms - merged_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_pct": round(_pct_delta(merged_ms, pr_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_merged": round(_speedup(merged_ms, pr_ms), RATIO_PRECISION_DIGITS),
                "active_open_prs": [
                    {"mechanism_id": m["mechanism_id"], "ref": m["ref"], "notes": m["notes"]}
                    for m in pr_report["unmerged_pr_mechanisms"]
                ],
                "single_pr_ablation": pr_ablations,
                "leave_one_out_ablation": loo_ablations,
            },
            "expected_ideal": {
                "step_ms": round(ideal_ms, MS_PRECISION_DIGITS),
                "operator_sum_ms": round(float(analyses["expected_ideal"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
                "hbm_bytes": float(analyses["expected_ideal"]["total_hbm_bytes"]),
                "network": _network(analyses["expected_ideal"]),
                "delta_vs_unmerged_pr_ms": round(ideal_ms - pr_ms, MS_PRECISION_DIGITS),
                "delta_vs_unmerged_pr_pct": round(_pct_delta(pr_ms, ideal_ms), PCT_PRECISION_DIGITS),
                "delta_vs_merged_ms": round(ideal_ms - merged_ms, MS_PRECISION_DIGITS),
                "delta_vs_merged_pct": round(_pct_delta(merged_ms, ideal_ms), PCT_PRECISION_DIGITS),
                "speedup_vs_merged": round(_speedup(merged_ms, ideal_ms), RATIO_PRECISION_DIGITS),
                "unimplemented_architectural_gaps": ideal_report["unimplemented_gaps"],
                "active_mechanisms": [m["id"] for m in ideal_report["active_mechanisms"]],
            },
            "hardware_roofline": {
                "physical_ideal_1p0x_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
                "remaining_kernel_micro_gap_to_1p0x_ms": round(max(0.0, ideal_ms - roof_1p0_ms), MS_PRECISION_DIGITS),
            },
        },
        "efficiency_reference": {
            "physical_ideal_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
            "basis": "physical_1p0x_critical_path_roofline",
        },
    }
