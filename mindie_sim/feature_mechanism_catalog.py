"""Source-audited optimization mechanism definitions and shared mutators.

[Role] Build audited physical optimization mechanisms and share case mutators
  with the stack-agnostic mechanism registry. The source cohort was reviewed
  against CANN 9.3 and vllm-ascend at revision 405617bae.
[Invariants] Eligibility uses model structure, quantization and hardware
  capability, with monotonic application across feature tracks.
[Timeouts & Locks] Bounded in-memory checks over layer plans, with no locks.
[Side-Effect Policy] Construct mechanism specs and mutate caller-owned runtime
  dictionaries during application.
"""
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


def _apply_mhc_fusion_overlap(rt: dict[str, Any], ctx: MechanismContext) -> None:
    unbounded = bool(ctx.options.get("unbounded_rows", False))
    pass_name = "mhc_boundary_ideal" if unbounded else "mhc_boundary_pr"
    # Upgrade PR-bounded pass to ideal pass cleanly if both are requested
    existing = [p for p in (rt.get("lowering_passes") or ())
                if p not in {"mhc_boundary_pr", "mhc_boundary_ideal"}]
    rt["lowering_passes"] = existing
    _append_passes(rt, pass_name)


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
    if "moe_comm_mode" in ctx.caller_explicit:
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
            apply=lambda rt, _ctx: _update_meta(rt, tp_mc2_overlap=True),
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
    )
