"""Optimization mechanism registry, stack profiles, software stack registry and 4-track waterfall.

[Role] Own optimization mechanism declarations, structural applicability rules,
  pluggable ``RuntimeStackProfile`` / ``SoftwareStackRegistry`` resolution, and
  4-track (raw_eager -> merged -> unmerged_pr -> expected_ideal) waterfall evaluation.
[Invariants] Mechanisms apply strictly by model/hardware structural capability,
  never by hardcoded model names. ``SoftwareStackRegistry.resolve_base`` never
  mutates caller ``Case`` objects or ``Software.kernels`` dictionaries.
[Timeouts & Locks] Bounded JSON manifest reads (< 50 MiB); delegates step
  evaluation to canonical Engine; no locks.
[Side-Effect Policy] Pure planning and read-only simulation; returns new frozen
  ``ResolvedBase`` and ``Case`` records."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.operators.linear_attention.gdn_domain import (
    GDN_CHUNK32_TOKENS,
    model_supports_gdn_chunk32,
)
from mindie_sim.prediction.results import (
    DEFAULT_STACK_NAME,
    LoweringRequest,
    OriginKind,
    ResolvedBase,
)
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.stacks.vllm_ascend.profiles import VA_RECOMMENDED_PROFILES

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


SHIPPING_MEGAMOE_SUPPORTED_ACTIVATIONS: frozenset[str] = frozenset({
    "swiglu", "silu", "gelu",
})


def _model_supports_shipping_megamoe_activation(ctx: MechanismContext) -> bool:
    if bool((ctx.options or {}).get("allow_extended_activation_abi", False)):
        return True
    moe_plans = [p for p in ctx.model.layer_plans if p.ffn == "moe"]
    if not moe_plans:
        return False
    return all(
        str(getattr(p, "ffn_activation", "silu") or "silu").strip().lower()
        in SHIPPING_MEGAMOE_SUPPORTED_ACTIVATIONS
        for p in moe_plans
    )


def _fused_mc2_disabled(ctx: MechanismContext) -> bool:
    return (ctx.options or {}).get("enable_fused_mc2") == 0


def _fused_mc2_explicitly_disabled(rt: Mapping[str, Any], ctx: MechanismContext) -> bool:
    additional = (dict(rt.get("metadata") or {}).get("additional_config") or {})
    if isinstance(additional, Mapping) and additional.get("enable_fused_mc2") == 0:
        return True
    return _fused_mc2_disabled(ctx)


def _apply_tp_mc2_overlap(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if _fused_mc2_explicitly_disabled(rt, ctx):
        return
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
    for _, fields in overrides:
        if isinstance(fields, str):
            fmt = fields.lower()
        elif isinstance(fields, Mapping):
            fmt = str(fields.get("weight_format") or "").lower()
        elif isinstance(fields, (tuple, list)):
            fmt = str(dict(fields).get("weight_format") or "").lower()
        else:
            fmt = ""
        if fmt in MXFP_EXPERT_WEIGHT_FORMATS:
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
    if "enable_torch_compile" in ctx.caller_explicit:
        return
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
    if _fused_mc2_explicitly_disabled(rt, ctx):
        return
    if not _model_supports_shipping_megamoe_activation(ctx):
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


def _apply_dspark_static_kernel(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if "enable_static_kernel" not in ctx.caller_explicit:
        rt["enable_static_kernel"] = True
    if "enable_super_kernel" not in ctx.caller_explicit:
        rt["enable_super_kernel"] = True
    _update_additional(
        rt,
        enable_static_kernel=bool(rt.get("enable_static_kernel", True)),
        enable_super_kernel=bool(rt.get("enable_super_kernel", True)),
    )


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


def _source_attention_and_moe_mechanisms() -> tuple[OptimizationMechanism, ...]:
    """Attention, indexer, recurrent, and MoE mechanisms in registration order."""
    return (
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
            eligible=lambda ctx: int(ctx.parallel.get("tp", 1)) > 1 and not _fused_mc2_disabled(ctx),
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
    )


def source_mechanisms() -> tuple[OptimizationMechanism, ...]:
    """Build the audited mechanism specs in registration order."""
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
                and not _fused_mc2_disabled(ctx)
                and _ep_within_ub_domain(ctx)
                and int(ctx.parallel.get("redundant_experts_per_rank", 0) or 0) == 0
                and any(p.ffn == "moe" for p in ctx.model.layer_plans)
                and _is_mxfp_moe(ctx)
                and _model_supports_shipping_megamoe_activation(ctx)
            ),
            apply=_apply_cann_megamoe,
        ),
        *_source_attention_and_moe_mechanisms(),
        *_parallel_cp_mechanisms(),
    )


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
    if ctx.arch in {"a3", "a5"} and quant is not None and quant.kv_dtype == "int8":
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
    if _fused_mc2_explicitly_disabled(rt, ctx):
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
        eligible=lambda ctx: (ctx.arch == "a5"
                              and any(p.residual_block_size > 0 for p in ctx.model.layer_plans)),
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
        id="kda_recoverssm_shipping", category="speculative_state", phases=("decode",),
        summary="Shipping KDA RecoverSSM speculative state verification merged with DSpark (#21549)",
        eligible=lambda ctx: (
            any(p.attention == "kda" for p in ctx.model.layer_plans)
            and int(ctx.parallel.get("pcp", 1)) == 1
        ),
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
        eligible=lambda ctx: (
            any(p.ffn == "moe" for p in ctx.model.layer_plans)
            and not _fused_mc2_disabled(ctx)
        ),
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


_UNQUANTIZED_PRESET_NAMES = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32"})


def _build_waterfall_payload(
    *,
    cases: Mapping[str, Mapping[str, Any]],
    analyses: Mapping[str, Mapping[str, Any]],
    geometry: Mapping[str, Any],
    pr_ablations: list[dict[str, Any]],
    loo_ablations: list[dict[str, Any]],
) -> dict[str, Any]:
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
    eager_monotonic = eager_ms + MONOTONIC_TOLERANCE_MS >= merged_ms
    monotonic = (
        eager_monotonic
        and merged_ms + MONOTONIC_TOLERANCE_MS >= pr_ms
        and pr_ms + MONOTONIC_TOLERANCE_MS >= ideal_ms
        and ideal_ms + MONOTONIC_TOLERANCE_MS >= roof_1p0_ms
    )
    return {
        "schema": WATERFALL_SCHEMA,
        "model": cases["merged"]["model"]["name"],
        "hardware_sku": cases["merged"]["machine"]["sku"],
        "runtime_stack": merged_report["stack"],
        "repository": merged_report["repository"],
        "geometry": dict(geometry),
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


def compare_feature_tracks(
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    *,
    runtime_stack: str | Mapping[str, Any] | None = None,
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

    cleaned_kwargs = dict(compose_kwargs)
    for key in ("feature_track", "optimizations", "exclude_optimizations", "raw_eager", "_stack_phase"):
        cleaned_kwargs.pop(key, None)

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
    merged_meta = cases["merged"]["runtime"]["metadata"] or {}
    merged_quant = cases["merged"]["model"].get("quant")
    va_prof = merged_meta.get("va_profile") or {}
    va_flags = dict(va_prof.get("tutorial_flags") or {})
    explicit_quant = eager_kwargs.get("quant")
    unquantized_override = (
        isinstance(explicit_quant, str)
        and explicit_quant.strip().lower() in _UNQUANTIZED_PRESET_NAMES
    )
    if "quant" not in eager_kwargs:
        if isinstance(merged_quant, Mapping) and merged_quant:
            eager_kwargs["quant"] = dict(merged_quant)
        elif va_prof.get("quant") is not None:
            eager_kwargs["quant"] = va_prof["quant"]
    if not unquantized_override:
        if "kv_cache_dtype" not in eager_kwargs:
            resolved_kv = (merged_quant.get("kv_dtype") if isinstance(merged_quant, Mapping) else None) or va_flags.get("kv_cache_dtype")
            if resolved_kv:
                eager_kwargs["kv_cache_dtype"] = str(resolved_kv)
        if "indexer_kv_dtype" not in eager_kwargs:
            resolved_idx = (merged_quant.get("index_dtype") if isinstance(merged_quant, Mapping) else None) or va_flags.get("indexer_kv_dtype")
            if resolved_idx:
                eager_kwargs["indexer_kv_dtype"] = str(resolved_idx)
    cases["raw_eager"] = compose_case(
        model, hardware=hardware, runtime_stack=runtime_stack,
        feature_track="raw_eager", _stack_phase=phase, **eager_kwargs,
    )
    analyses["raw_eager"] = analyze_network_efficiency(
        cases["raw_eager"], phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope,
    )

    pr_ablations, loo_ablations = [], []
    if ablate_prs:
        pr_report = cases["unmerged_pr"]["runtime"]["metadata"]["feature_track_report"]
        pr_ablations, loo_ablations = _compare_pr_ablations(
            model, hardware, runtime_stack, pr_report,
            float(analyses["merged"]["graph_duration_ms"]),
            float(analyses["unmerged_pr"]["graph_duration_ms"]),
            phase=phase, tokens=tokens, context=context, sequences=sequences,
            scope=scope, compose_kwargs=cleaned_kwargs,
        )

    return _build_waterfall_payload(
        cases=cases, analyses=analyses,
        geometry={"phase": phase, "tokens": tokens, "context": context,
                  "sequences": sequences, "scope": scope},
        pr_ablations=pr_ablations, loo_ablations=loo_ablations,
    )


# --- Layer 2 RuntimeStackProfile Registry & JSON Manifest Loader ------------

STACK_MANIFEST_SCHEMA = "mindie-sim.stack-manifest/1"
VLLM_ASCEND_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "stacks" / "vllm_ascend" / "manifest.json"
)
MAX_MANIFEST_BYTES = 50 * 1024 * 1024
MIN_CAPTURE_SEQS = 16

TRACK_ALIASES: dict[str, str] = {
    "raw_eager": "raw_eager",
    "eager": "raw_eager",
    "merged": "merged",
    "current_merged": "merged",
    "unmerged_pr": "unmerged_pr",
    "current_unmerged_pr": "unmerged_pr",
    "expected_ideal": "expected_ideal",
    "ideal": "expected_ideal",
}

_MECHANISM_LOWERING_SELECTORS: dict[str, tuple[str, ...]] = {
    "add_rmsnorm_quant_fusion": (
        "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant"),
    "mla_prolog_fusion": ("mla_prolog",),
    "attn_res_prefix_fusion": ("attention_residual",),
    "recurrent_decode_native_kernels": ("kda", "gdn_decode"),
    "gdn_chunk64_prefill": ("gdn_prefill",),
}
_MECHANISM_LOWERING_PASSES: dict[str, tuple[str, ...]] = {
    "add_rmsnorm_quant_fusion": ("residual_norm",),
    "mhc_boundary_fusion_overlap": ("mhc_boundary_pr", "mhc_boundary_ideal"),
    "gdn_projection_multistream_overlap": ("gdn_projection_overlap",),
    "kda_in_kernel_l2norm": ("kda_in_kernel_l2norm",),
    "qkv_norm_rope_fusion": ("qkv_norm_rope",),
    "indexer_online_ub_topk": ("indexer_online_topk",),
    "activation_quant_fusion": ("activation_quant",),
    "gmm_swiglu_quant_fusion": ("gmm_swiglu_quant",),
    "moe_tail_residual_fusion": ("moe_tail_residual",),
    "tp_mc2_overlap": ("tp_mc2_overlap",),
}


def excluded_lowering_bindings(
    exclude_optimizations: Sequence[str] = (),
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve excluded mechanism IDs into installed selectors and passes."""
    excluded = frozenset(
        str(value).strip() for value in exclude_optimizations if str(value).strip()
    )
    for mechanism_id in excluded:
        if mechanism_id not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{mechanism_id}")
    selectors = frozenset(
        selector for mechanism_id in excluded
        for selector in _MECHANISM_LOWERING_SELECTORS.get(mechanism_id, ())
    )
    passes = frozenset(
        name for mechanism_id in excluded
        for name in _MECHANISM_LOWERING_PASSES.get(mechanism_id, ())
    )
    return selectors, passes


@dataclass(frozen=True)
class StackMechanismBinding:
    """Binds one universal mechanism ID to a stack's merge state and PR ref."""

    mechanism_id: str
    state: str  # "merged" | "unmerged_pr"
    ref: str | None = None
    models: tuple[str, ...] = ()
    archs: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mechanism_id not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{self.mechanism_id}")
        if self.state not in {"merged", "unmerged_pr"}:
            raise ModelingError(f"invalid_stack_binding_state:{self.state}")

    def matches(self, model_key: str, arch: str, phase: str | None = None) -> bool:
        if self.models and model_key not in self.models:
            return False
        if self.archs and arch not in self.archs:
            return False
        if phase and self.phases and phase.lower() not in self.phases:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "state": self.state,
            "ref": self.ref,
            "models": list(self.models),
            "archs": list(self.archs),
            "phases": list(self.phases),
            "options": dict(self.options),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RuntimeStackProfile:
    """Complete feature-track declaration for one serving runtime stack."""

    name: str
    repository: str
    description: str = ""
    model_defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    bindings: tuple[StackMechanismBinding, ...] = ()
    supported_models: tuple[str, ...] = ()
    unmerged_pr_models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STACK_MANIFEST_SCHEMA,
            "name": self.name,
            "repository": self.repository,
            "description": self.description,
            "supported_models": list(self.supported_models),
            "unmerged_pr_models": list(self.unmerged_pr_models),
            "bindings": [b.to_dict() for b in self.bindings],
        }


RUNTIME_STACKS: dict[str, RuntimeStackProfile] = {}


def normalize_feature_track(track: str) -> str:
    key = str(track or "").strip().lower()
    if key not in TRACK_ALIASES:
        raise ModelingError(f"unknown_feature_track:{track}")
    return TRACK_ALIASES[key]


def register_runtime_stack(profile: RuntimeStackProfile) -> RuntimeStackProfile:
    if not isinstance(profile, RuntimeStackProfile) or not profile.name.strip():
        raise ModelingError("invalid_runtime_stack_profile")
    RUNTIME_STACKS[profile.name.strip().lower()] = profile
    return profile


def load_runtime_stack_manifest(source: str | Path | Mapping[str, Any]) -> RuntimeStackProfile:
    """Load a ``RuntimeStackProfile`` from a JSON file or plain Python dict."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise ModelingError(f"stack_manifest_not_found:{path}")
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ModelingError(f"stack_manifest_exceeds_50_mib:{path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(source, Mapping):
        raw = dict(source)
    else:
        raise ModelingError("stack_manifest_requires_path_or_mapping")
    if not isinstance(raw, Mapping):
        raise ModelingError("stack_manifest_invalid_mapping")
    schema = raw.get("schema", STACK_MANIFEST_SCHEMA)
    if schema != STACK_MANIFEST_SCHEMA:
        raise ModelingError(f"stack_manifest_schema_unsupported:{schema}")
    bindings = tuple(
        StackMechanismBinding(
            mechanism_id=str(item["mechanism_id"]),
            state=str(item["state"]),
            ref=item.get("ref"),
            models=tuple(str(m) for m in (item.get("models") or ())),
            archs=tuple(str(a).lower() for a in (item.get("archs") or ())),
            phases=tuple(str(p).lower() for p in (item.get("phases") or ())),
            options=dict(item.get("options") or {}),
            notes=str(item.get("notes") or ""),
        )
        for item in (raw.get("bindings") or ())
    )
    return RuntimeStackProfile(
        name=str(raw["name"]).strip(),
        repository=str(raw.get("repository") or ""),
        description=str(raw.get("description") or ""),
        model_defaults=dict(raw.get("model_defaults") or {}),
        bindings=bindings,
        supported_models=tuple(str(m).strip() for m in (raw.get("supported_models") or ())),
        unmerged_pr_models=tuple(str(m).strip() for m in (raw.get("unmerged_pr_models") or ())),
    )


def resolve_runtime_stack(
    stack: str | Path | Mapping[str, Any] | RuntimeStackProfile,
) -> RuntimeStackProfile:
    if isinstance(stack, RuntimeStackProfile):
        return stack
    if isinstance(stack, (Mapping, Path)):
        return load_runtime_stack_manifest(stack)
    key = str(stack or "").strip()
    lower = key.lower()
    if lower in RUNTIME_STACKS:
        return RUNTIME_STACKS[lower]
    candidate = Path(key).expanduser()
    if candidate.is_file():
        return load_runtime_stack_manifest(candidate)
    raise ModelingError(f"unknown_runtime_stack:{stack}")


def _collapse_binding_group(
    mid: str, state: str, group: Sequence[StackMechanismBinding],
) -> dict[str, Any]:
    merged_options: dict[str, Any] = {}
    refs: list[str] = []
    notes: list[str] = []
    for item in group:
        merged_options.update(dict(item.options))
        if item.ref and item.ref not in refs:
            refs.append(item.ref)
        if item.notes and item.notes not in notes:
            notes.append(item.notes)
    primary_ref = refs[0] if len(refs) == 1 else (", ".join(refs) if refs else None)
    entry: dict[str, Any] = {
        "id": mid,
        "state": state,
        "ref": primary_ref,
        "options": merged_options,
        "notes": "; ".join(notes),
    }
    if len(refs) > 1:
        entry["refs"] = refs
    return entry


def resolve_stack_track(
    stack_profile: RuntimeStackProfile,
    *,
    track: str,
    model_key: str,
    ctx: MechanismContext,
    optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve active mechanisms, PR provenance, and unimplemented gaps."""
    canonical_track = normalize_feature_track(track)
    if stack_profile.supported_models or stack_profile.unmerged_pr_models:
        if canonical_track == "merged" and model_key not in stack_profile.supported_models:
            raise ModelingError(
                f"model_not_supported_by_runtime_stack:{stack_profile.name}:{model_key}:merged")
        if (canonical_track == "unmerged_pr"
                and model_key not in stack_profile.supported_models
                and model_key not in stack_profile.unmerged_pr_models):
            raise ModelingError(
                f"model_not_supported_by_runtime_stack:{stack_profile.name}:{model_key}:unmerged_pr")

    include_ids = tuple(str(x).strip() for x in optimizations if str(x).strip())
    exclude_ids = frozenset(str(x).strip() for x in exclude_optimizations if str(x).strip())
    for mid in (*include_ids, *exclude_ids):
        if mid not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{mid}")
    overlap_conflict = set(include_ids) & exclude_ids
    if overlap_conflict:
        raise ModelingError(
            "optimization_include_exclude_conflict:" + ",".join(sorted(overlap_conflict)))

    model_cfg = dict(stack_profile.model_defaults.get(model_key) or {})
    ctx_phase = str((ctx.options or {}).get("phase") or "").strip().lower() or None
    eligible_ids = [
        mech.id for mech in MECHANISM_CATALOG.values()
        if mech.eligible(ctx) and (not ctx_phase or not mech.phases or ctx_phase in mech.phases)
    ]
    eligible_set = set(eligible_ids)
    merged_list: list[StackMechanismBinding] = []
    pr_list: list[StackMechanismBinding] = []
    merged_by_id: dict[str, list[StackMechanismBinding]] = defaultdict(list)
    pr_by_id: dict[str, list[StackMechanismBinding]] = defaultdict(list)
    for binding in stack_profile.bindings:
        if not binding.matches(model_key, ctx.arch, phase=ctx_phase):
            continue
        if binding.mechanism_id not in eligible_set:
            continue
        if binding.state == "merged":
            if binding.mechanism_id == "sequence_parallel" and model_cfg.get("sp") is False:
                continue
            merged_list.append(binding)
            merged_by_id[binding.mechanism_id].append(binding)
        else:
            pr_list.append(binding)
            pr_by_id[binding.mechanism_id].append(binding)

    active: dict[str, dict[str, Any]] = {}
    if canonical_track in {"merged", "unmerged_pr", "expected_ideal"}:
        for mid, group in merged_by_id.items():
            active[mid] = _collapse_binding_group(mid, "merged", group)
    if canonical_track in {"unmerged_pr", "expected_ideal"}:
        for mid, group in pr_by_id.items():
            prev_opts = dict(active[mid]["options"]) if mid in active else {}
            collapsed = _collapse_binding_group(mid, "unmerged_pr", group)
            collapsed["options"] = {**prev_opts, **collapsed["options"]}
            active[mid] = collapsed
    ideal_eligible_ids: list[str] = eligible_ids
    if canonical_track == "expected_ideal":
        ideal_opts = {**dict(ctx.options or {}), "allow_extended_activation_abi": True}
        if "enable_fused_mc2" not in ctx.caller_explicit and ideal_opts.get("enable_fused_mc2") == 0:
            ideal_opts.pop("enable_fused_mc2", None)
        ideal_ctx = MechanismContext(
            model=ctx.model, sku=ctx.sku, arch=ctx.arch, parallel=ctx.parallel,
            caller_explicit=ctx.caller_explicit,
            options=ideal_opts,
        )
        ideal_eligible_ids = [
            mech.id for mech in MECHANISM_CATALOG.values()
            if mech.eligible(ideal_ctx) and (not ctx_phase or not mech.phases or ctx_phase in mech.phases)
        ]
        for mid in ideal_eligible_ids:
            prev = active.get(mid)
            opts: dict[str, Any] = {}
            if mid == "mhc_boundary_fusion_overlap":
                opts["unbounded_rows"] = True
            elif mid == "cann_megamoe_fusion":
                opts["allow_extended_activation_abi"] = True
            if prev is None:
                active[mid] = {
                    "id": mid, "state": "expected_ideal", "ref": None,
                    "options": opts, "notes": "architectural_ideal",
                }
            elif opts:
                active[mid] = {**prev, "options": {**prev["options"], **opts}}

    for mid in include_ids:
        mech = MECHANISM_CATALOG[mid]
        opt_ctx = MechanismContext(
            model=ctx.model, sku=ctx.sku, arch=ctx.arch, parallel=ctx.parallel,
            caller_explicit=ctx.caller_explicit,
            options={**dict(ctx.options or {}), "explicit_opt_in": True},
        )
        if not mech.eligible(opt_ctx):
            continue
        group = pr_by_id.get(mid) or merged_by_id.get(mid)
        if group:
            collapsed = _collapse_binding_group(mid, group[0].state, group)
            collapsed["notes"] = "caller_enabled"
            active[mid] = collapsed
        else:
            active[mid] = {
                "id": mid, "state": "explicit_override", "ref": None,
                "options": {}, "notes": "caller_enabled",
            }
    for mid in exclude_ids:
        active.pop(mid, None)

    gap_pool = ideal_eligible_ids if canonical_track == "expected_ideal" else eligible_ids
    unimplemented_gaps = [
        mid for mid in gap_pool
        if mid not in merged_by_id and mid not in pr_by_id
    ]
    return {
        "stack": stack_profile.name,
        "repository": stack_profile.repository,
        "feature_track": canonical_track,
        "eligible_mechanisms": eligible_ids,
        "active_mechanisms": [active[key] for key in MECHANISM_CATALOG if key in active],
        "merged_mechanisms": [b.to_dict() for b in merged_list],
        "unmerged_pr_mechanisms": [b.to_dict() for b in pr_list],
        "unimplemented_gaps": unimplemented_gaps,
        "explicit_included": list(include_ids),
        "explicit_excluded": sorted(exclude_ids),
    }


def apply_resolved_mechanisms(
    runtime_dict: dict[str, Any],
    resolved_track: Mapping[str, Any],
    base_ctx: MechanismContext,
) -> None:
    """Apply every active mechanism in deterministic catalog order."""
    active = {entry["id"]: entry for entry in resolved_track["active_mechanisms"]}
    for mid, mech in MECHANISM_CATALOG.items():
        entry = active.get(mid)
        if entry is None:
            continue
        opts = {**dict(base_ctx.options or {}), **dict(entry.get("options") or {})}
        if entry.get("notes") == "caller_enabled":
            opts["explicit_opt_in"] = True
        ctx = MechanismContext(
            model=base_ctx.model,
            sku=base_ctx.sku,
            arch=base_ctx.arch,
            parallel=base_ctx.parallel,
            caller_explicit=base_ctx.caller_explicit,
            options=opts,
        )
        if mech.eligible(ctx):
            mech.apply(runtime_dict, ctx)


def _vllm_ascend_profile() -> RuntimeStackProfile:
    profile = load_runtime_stack_manifest(VLLM_ASCEND_MANIFEST_PATH)
    model_defaults = {
        key: {
            "quant": dict(value.get("quant") or {}),
            "sp": value.get("sp", True),
            "mlapo": "VLLM_ASCEND_ENABLE_MLAPO" in (value.get("env") or {}),
            "kv_block_size": value.get("kv_block_size"),
            "mtp_proposal_tokens": value.get("mtp_proposal_tokens", 0),
            "graph_capture_mode": value.get("graph_capture_mode", "full"),
            "multistream_overlap_shared_expert": bool(
                value.get("multistream_overlap_shared_expert", False)
            ),
            "engram_cpu_offload": bool(value.get("engram_cpu_offload", False)),
            "tutorial_flags": dict(value.get("tutorial_flags") or {}),
        }
        for key, value in VA_RECOMMENDED_PROFILES.items()
    }
    return replace(
        profile,
        description="vLLM-Ascend main baseline and audited open performance PRs",
        model_defaults=model_defaults,
        bindings=profile.bindings,
    )


register_runtime_stack(_vllm_ascend_profile())

register_runtime_stack(RuntimeStackProfile(
    name="raw_eager",
    repository="eager-baseline",
    description="Unfused eager PyTorch/ATen execution baseline with zero optional fusions",
    bindings=(),
))


# --- Unified Software Stack Registry & Lowering-Base Resolver ---------------


@dataclass(frozen=True, slots=True)
class SoftwareStackSpec:
    """Declarative specification of one registered inference software stack."""

    name: str
    repository: str
    description: str
    profile: RuntimeStackProfile
    sources: tuple[SourceRef, ...] = ()
    default_track: str = "merged"
    decode_capture_mode: str = "full"


def _extract_origins(metadata: Mapping[str, object]) -> tuple[tuple[str, OriginKind], ...]:
    raw = metadata.get("recipe_binding_origins")
    if not isinstance(raw, Mapping):
        return ()
    entries: list[tuple[str, OriginKind]] = []
    for section, mapping in sorted(raw.items()):
        if not isinstance(mapping, Mapping):
            continue
        for key, origin in sorted(mapping.items()):
            label = str(origin)
            if label in {"caller", "stack_profile", "body_default"}:
                entries.append((f"{section}.{key}", label))  # type: ignore[arg-type]
    return tuple(entries)


def _stack_name_for_case(case: Case, requested: str | None) -> str:
    if requested:
        return str(requested).strip()
    meta = dict(case.runtime.metadata)
    report = meta.get("feature_track_report")
    if isinstance(report, Mapping) and report.get("stack"):
        return str(report["stack"]).strip()
    return DEFAULT_STACK_NAME


def _mtp_proposal_tokens(case: Case) -> int:
    """Resolve draft/MTP proposal token count matching ``scheduler.py``."""
    dspark = getattr(case.runtime, "dspark", None)
    if dspark is not None:
        step_spec = dspark.step_spec()
        if getattr(step_spec, "draft_len", 0):
            return int(step_spec.draft_len)
    meta = dict(case.runtime.metadata)
    raw_spec = meta.get("speculative")
    if raw_spec:
        if isinstance(raw_spec, SpeculativeSpec):
            spec_obj = raw_spec
        else:
            parsed = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
            spec_obj = SpeculativeSpec.from_mapping(parsed) if isinstance(parsed, Mapping) else None
        if spec_obj is not None and spec_obj.draft_len:
            return int(spec_obj.draft_len)
    if meta.get("draft_proposal_tokens"):
        return int(meta["draft_proposal_tokens"])
    aux = getattr(case.model, "auxiliary", None)
    mtp = getattr(aux, "mtp", None) if aux is not None else None
    if mtp is not None and getattr(mtp, "proposal_tokens", 0):
        return int(mtp.proposal_tokens)
    model_spec = getattr(case.model, "speculative", None)
    if model_spec is not None and getattr(model_spec, "proposal_tokens", 0):
        return int(model_spec.proposal_tokens)
    return 0


def _apply_production_decode_capture(
    case: Case,
    request: LoweringRequest,
    stack: SoftwareStackSpec,
) -> Case:
    if not request.production_capture or request.step is None or request.step.phase != "decode":
        return case
    target_mode = (
        stack.decode_capture_mode
        if stack.decode_capture_mode and stack.decode_capture_mode != "none"
        else "full"
    )
    if case.runtime.graph_capture_mode == target_mode and case.runtime.graph_capture_keys:
        return case
    from mindie_sim.compose.runtime import capture_keys

    max_seqs = max(
        MIN_CAPTURE_SEQS,
        int(case.runtime.max_num_seqs or MIN_CAPTURE_SEQS),
        request.step.sequences if isinstance(request.step.sequences, int)
        else max(request.step.sequences, default=1),
    )
    mtp = _mtp_proposal_tokens(case)
    keys = tuple(case.runtime.graph_capture_keys or capture_keys(max_seqs, mtp))
    runtime = replace(
        case.runtime,
        graph_capture_mode=target_mode,
        graph_capture_keys=keys,
    )
    return replace(case, runtime=runtime)


def _apply_tier_policy(
    case: Case,
    request: LoweringRequest,
    stack: SoftwareStackSpec,
) -> Case:
    if request.tier == "mainline":
        return case
    if request.tier == "unmerge":
        kernels = deepcopy(dict(case.runtime.kernels))
        meta = dict(case.runtime.metadata)
        origins = meta.get("recipe_binding_origins")
        lowerings = dict(kernels.get("lowerings") or {})
        if isinstance(origins, Mapping) and isinstance(origins.get("lowerings"), Mapping):
            low_origins = origins["lowerings"]
            lowerings = {
                key: value for key, value in lowerings.items()
                if low_origins.get(key) == "caller"
            }
        else:
            lowerings = {}
        kernels["lowerings"] = lowerings
        policy = dict(meta.get("execution_policy") or {})
        policy["automatic_optional_fusions"] = False
        meta["execution_policy"] = policy
        runtime = replace(
            case.runtime,
            kernels=kernels,
            lowering_passes=(),
            metadata=tuple(meta.items()),
        )
        return replace(case, runtime=runtime)

    from mindie_sim.compose.runtime import finalize_compilation_spine

    rt_dict = case.runtime.to_dict()
    parallel = case.runtime.parallel
    phase = request.step.phase if request.step is not None else None
    arch = str(case.machine.arch_family or "").lower()
    ctx = MechanismContext(
        model=case.model,
        sku=str(case.machine.sku or ""),
        arch=arch,
        parallel={
            "tp": parallel.tp,
            "dp": parallel.dp,
            "pp": parallel.pp,
            "ep": parallel.ep,
            "pcp": parallel.pcp,
            "dcp": parallel.dcp,
        },
        caller_explicit=frozenset(),
        options={"phase": phase} if phase else {},
    )
    resolved = resolve_stack_track(
        stack.profile,
        track="expected_ideal",
        model_key=str(case.model.name),
        ctx=ctx,
    )
    apply_resolved_mechanisms(rt_dict, resolved, ctx)
    finalize_compilation_spine(rt_dict, raw_eager=False)
    new_runtime = Software.from_mapping(rt_dict)
    return replace(case, runtime=new_runtime)


class SoftwareStackRegistry:
    """Registry of supported inference software stacks and tier resolution."""

    def __init__(self, specs: Iterable[SoftwareStackSpec] = (), /) -> None:
        self._specs: dict[str, SoftwareStackSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SoftwareStackSpec, /) -> SoftwareStackSpec:
        key = spec.name.strip().lower()
        if not key:
            raise ModelingError("software_stack_name_required")
        self._specs[key] = spec
        return spec

    def get(self, name: str, /) -> SoftwareStackSpec:
        key = str(name or "").strip().lower()
        if key in self._specs:
            return self._specs[key]
        profile = resolve_runtime_stack(name)
        return SoftwareStackSpec(
            name=profile.name,
            repository=profile.repository,
            description=profile.description,
            profile=profile,
        )

    def list_stacks(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def resolve_base(self, case: Case, request: LoweringRequest, /) -> ResolvedBase:
        """Resolve stack profile, tier rules, capture policy, and binding origins."""
        stack = self.get(_stack_name_for_case(case, request.stack))
        working = replace(
            case,
            runtime=replace(
                case.runtime,
                kernels=deepcopy(dict(case.runtime.kernels)),
                metadata=tuple(deepcopy(dict(case.runtime.metadata)).items()),
            ),
        )
        working = _apply_tier_policy(working, request, stack)
        working = _apply_production_decode_capture(working, request, stack)
        origins = _extract_origins(dict(working.runtime.metadata))
        return ResolvedBase(
            case=working,
            request=request,
            stack=stack,
            recipe_origins=origins,
            sources=stack.sources,
        )


@lru_cache(maxsize=1)
def default_stack_registry() -> SoftwareStackRegistry:
    """Return the built-in ``SoftwareStackRegistry`` populated from audited stacks."""
    specs = [
        SoftwareStackSpec(
            name=profile.name,
            repository=profile.repository,
            description=profile.description,
            profile=profile,
            default_track="raw_eager" if profile.name == "raw_eager" else "merged",
            decode_capture_mode="none" if profile.name == "raw_eager" else "full",
        )
        for profile in RUNTIME_STACKS.values()
    ]
    return SoftwareStackRegistry(specs)


StackRegistry = SoftwareStackRegistry
