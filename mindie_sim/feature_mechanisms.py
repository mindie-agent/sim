"""Optimization mechanism registry, stack profiles, software stack registry and 4-track waterfall.

[Role] Own optimization mechanism declarations, structural applicability rules,
  and canonical 4-track (eager -> shipping -> proposed -> target) contracts.
[Invariants] Mechanisms apply strictly by model/hardware structural capability,
  never by hardcoded model names (AG014). Zero legacy track wire aliases, zero
  silent ModelingError swallows, zero model-keyed stack defaults.
[Timeouts & Locks] Bounded JSON manifest reads (< 50 MiB); delegates step
  evaluation to canonical Engine; no locks.
[Side-Effect Policy] Pure planning and read-only simulation; returns new frozen
  ``ResolvedBase`` and ``Case`` records."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.catalog.context_parallel import is_v41_replicated_cache_model
from mindie_sim.catalog.draft_spec import validate_fia_speculative_tokens
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
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.stack import STACK_TRACKS
from mindie_sim.stacks.vllm_ascend.profiles import (
    ManifestBinding,
    extract_model_capabilities,
)
from mindie_sim.stacks.vllm_ascend.speculative import build_capture_bucket_rule

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
        and float(getattr(p, "swiglu_limit", 0.0) or 0.0) <= 0.0
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


def _has_unit_offset_norm(model: ModelSpec) -> bool:
    normalization = dict(model.catalog_facts).get("normalization") or {}
    return bool(normalization.get("norm_weight_offset", 0.0))


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
        if _has_unit_offset_norm(ctx.model):
            _append_passes(rt, "activation_quantization")
        else:
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
    routed_only = any(
        p.ffn == "moe"
        and (
            p.expert_latent_size > 0
            or p.ffn_activation == "situ"
            or bool(getattr(p, "shared_expert_gate", False))
            or int(getattr(p, "num_shared_experts", 0) or 0) <= 0
        )
        for p in ctx.model.layer_plans
    )
    default_mode = FUSED_MC2_ROUTED_MODE if routed_only else FUSED_MC2_SHARED_MODE
    raw_level = int(ctx.options.get("enable_fused_mc2", default_mode))
    level = min(raw_level, FUSED_MC2_ROUTED_MODE) if routed_only else raw_level
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


def _apply_region_gbsa_qkvg_indexer(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _append_passes(rt, "qkv_norm_rope", "indexer_online_topk")
    _update_meta(rt, flash_attn_npu_3=True, packed_qkvg_indexer=True)


def _apply_csa2_indexer_epilogue(rt: dict[str, Any], ctx: MechanismContext) -> None:
    if default_abi_registry().is_shipping_available("aclnnCompressor", ctx.arch.upper()):
        _set_lowering(rt, "compressor", "vllm_ascend_cann_compressor")
    _update_meta(rt, csa2_epilogue_engram_overlap=True)


def _apply_latent_moe_dspark_c8(rt: dict[str, Any], ctx: MechanismContext) -> None:
    _apply_shared_expert_overlap(rt, ctx)
    _update_meta(rt, latent_moe_dspark_c8_mla=True)


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


def _apply_in_kernel_gdn_decode_gate(rt: dict[str, Any], _ctx: MechanismContext) -> None:
    _set_layout_strided(rt, "gdn_gating", "beta", "alpha")
    _update_meta(rt, skip_gdn_output_zero=True)


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
            id="region_gbsa_qkvg_indexer_fusion", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Packed SWA qkvg + Triton indexer_qkz_norm_rope + CANN 9.3 FA3 SWA",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(
                    p.q_output_gate_kind == "headwise" and p.indexer
                    for p in ctx.model.layer_plans
                )
            ),
            apply=_apply_region_gbsa_qkvg_indexer,
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
            id="csa2_indexer_epilogue_fusion", category="attention_fusion",
            phases=("prefill", "decode"),
            summary="Fused CSA2 indexer prologue, MQSMLA epilogue, C1-only DSpark share & Engram overlap",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(
                    p.mla_variant == "latent_mqa" and p.compressor_kind == "gated_ring"
                    for p in ctx.model.layer_plans
                )
            ),
            apply=_apply_csa2_indexer_epilogue,
        ),
        OptimizationMechanism(
            id="mqsmla_dsl_544_320", category="attention_fusion",
            phases=("decode",),
            summary="Non-CANN TileLang DSL MixedQuantSparseFlashMla (544B FP8 + 320B FP4 packed KV) on A5",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and (
                    any(
                        p.mla_variant == "latent_mqa"
                        and p.cache_kind == "window_compressed"
                        and int(p.compress_ratio or 0) > 0
                        for p in ctx.model.layer_plans
                    )
                    or is_v41_replicated_cache_model(ctx.model)
                )
            ),
            apply=lambda rt, _ctx: _update_meta(rt, mqsmla_dsl_544_320_active=True),
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
            id="latent_moe_dspark_c8_mla", category="moe_overlap", phases=("prefill", "decode"),
            summary="5L MLA DSpark C8 QuantFlashMLA + Latent-MoE norm/quant & shared expert overlap",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.expert_latent_size > 0 for p in ctx.model.layer_plans)
            ),
            apply=_apply_latent_moe_dspark_c8,
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
    has_moe_layers = any(getattr(p, "ffn", "") == "moe" for p in ctx.model.layer_plans)
    dense_sp_requested = bool(
        ctx.options.get("enable_flashcomm1")
        or ctx.options.get("explicit_opt_in")
        or ctx.options.get("phase") == "prefill"
    )
    ep_compatible = (
        (ep >= tp and ep % tp == 0)
        or (ep == 1 and not has_moe_layers and dense_sp_requested)
    )
    if pcp >= 1 and pp >= 1 and tp > 1 and ep_compatible:
        _validate_model(ctx.model, padding="right_to_tp")
        kernels = dict(rt.get("kernels") or {})
        kernels.setdefault(
            "sequence_parallel",
            dict(DECLARATION, padding="right_to_tp", enable_expert_parallel=bool(has_moe_layers)),
        )
        rt["kernels"] = kernels
        meta = dict(rt.get("metadata") or {})
        meta["sp"] = True
        rt["metadata"] = meta


def _apply_basic_norm_quant(rt: dict[str, Any], ctx: MechanismContext) -> None:
    compile_on = _torch_compile_active(rt, ctx.model)
    normalization = dict(ctx.model.catalog_facts).get("normalization") or {}
    has_unit_offset = bool(normalization.get("norm_weight_offset", 0.0))
    if _has_quant_activations(ctx.model):
        _append_passes(rt, "activation_quantization")
        if compile_on and not has_unit_offset:
            _append_passes(rt, "norm_quant")
    if not compile_on:
        return
    # ``add_rmsnorm`` is the residual boundary and is independent of qk_norm.
    if ctx.model.activation_dtype == "bf16" and not has_unit_offset:
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


def _mla_prolog_fusion_eligible(ctx: MechanismContext) -> bool:
    """Check structural domain preconditions for fused MLA Prolog (zero ModelingError swallow)."""
    from mindie_sim.prediction.operators.mla_prolog.family_mla_prolog import (
        BF16_CACHE_DTYPES,
        MLA_PROLOG_QUERY_LORA_RANKS,
        _indexer_q_category,
        _prolog_port_formats,
    )
    from mindie_sim.prediction.operators.mla_prolog.mla_prolog_packed import SUPPORTED_QUERY_FORMATS
    from mindie_sim.catalog.ops import compute_dtype_for

    mla_layers = [p for p in ctx.model.layer_plans if p.attention == "mla" and p.q_lora_rank > 0]
    if not mla_layers or ctx.model.activation_dtype != "bf16":
        return False
    quant = ctx.model.quant
    if quant is None:
        return False
    w_ports, act_ports = _prolog_port_formats(quant)
    if len(set(w_ports)) != 1 or len(set(act_ports)) != 1:
        return False
    proj_w, proj_act = w_ports[0], act_ports[0]
    if proj_w not in {"bf16", "int8", "fp8_e4m3"}:
        return False
    packed = ctx.arch in {"a3", "a5"} and quant.kv_dtype == "int8"
    if packed and (proj_w not in SUPPORTED_QUERY_FORMATS or proj_act != proj_w):
        return False
    if quant.kv_dtype not in BF16_CACHE_DTYPES and not packed:
        return False
    for plan in mla_layers:
        if (
            plan.mla_variant != "latent_mla"
            or plan.cache_kind != "dense"
            or plan.qk_rope_head_dim <= 0
            or bool(plan.mla_use_nope)
            or bool(plan.mla_qkv_a_gate_fused)
            or bool(plan.mla_output_gate)
            or plan.q_lora_rank not in MLA_PROLOG_QUERY_LORA_RANKS
        ):
            return False
        if proj_act != ctx.model.activation_dtype and (
            plan.indexer or plan.owns_topk or plan.index_query_per_layer
        ):
            if compute_dtype_for(quant, _indexer_q_category(quant))[1] != proj_act:
                return False
    return True


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
    for plan in mla_layers:
        validate_mla_prolog(plan=plan, parallel=p_cfg, model=ctx.model,
                            quant=quant, lowering_passes=passes, **options)
    _set_lowering(rt, "mla_prolog", (
        {"implementation": "vllm_ascend_mla_prolog_v3", "options": options}
        if options else "vllm_ascend_mla_prolog_v3"))


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
        eligible=_mla_prolog_fusion_eligible,
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


NS_PER_MS = 1_000_000


def _ms_to_ns(duration_ms: float) -> int:
    return int(round(float(duration_ms) * NS_PER_MS))


def _compare_pr_ablations(
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any],
    runtime_stack: str | Mapping[str, Any] | None,
    proposed_report: Mapping[str, Any],
    shipping_ns: int,
    proposed_ns: int,
    *,
    phase: str,
    tokens: int,
    context: int,
    sequences: int,
    scope: str,
    compose_kwargs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.efficiency import analyze_network_efficiency

    shipping_ms = shipping_ns / NS_PER_MS
    proposed_ms = proposed_ns / NS_PER_MS
    pr_ablations: list[dict[str, Any]] = []
    loo_ablations: list[dict[str, Any]] = []
    single_cache: dict[str, int] = {}
    loo_cache: dict[str, int] = {}
    for pr_item in proposed_report.get("unmerged_pr_mechanisms", ()):
        mid = str(pr_item["mechanism_id"])
        if mid not in single_cache:
            single_case = compose_case(
                model, hardware=hardware, runtime_stack=runtime_stack,
                feature_track="shipping", optimizations=(mid,),
                _stack_phase=phase, **compose_kwargs,
            )
            single_eff = analyze_network_efficiency(
                single_case, phase=phase, tokens=tokens, context=context,
                sequences=sequences, scope=scope,
            )
            single_cache[mid] = _ms_to_ns(float(single_eff["graph_duration_ms"]))
        if mid not in loo_cache:
            loo_case = compose_case(
                model, hardware=hardware, runtime_stack=runtime_stack,
                feature_track="proposed", exclude_optimizations=(mid,),
                _stack_phase=phase, **compose_kwargs,
            )
            loo_eff = analyze_network_efficiency(
                loo_case, phase=phase, tokens=tokens, context=context,
                sequences=sequences, scope=scope,
            )
            loo_cache[mid] = _ms_to_ns(float(loo_eff["graph_duration_ms"]))
        single_ns = single_cache[mid]
        without_ns = loo_cache[mid]
        single_ms = single_ns / NS_PER_MS
        without_ms = without_ns / NS_PER_MS
        pr_ablations.append({
            "mechanism_id": mid,
            "ref": pr_item.get("ref"),
            "notes": pr_item.get("notes", ""),
            "step_ns": single_ns,
            "step_ms": round(single_ms, MS_PRECISION_DIGITS),
            "delta_vs_shipping_ns": single_ns - shipping_ns,
            "delta_vs_shipping_ms": round((single_ns - shipping_ns) / NS_PER_MS, MS_PRECISION_DIGITS),
            "delta_vs_shipping_pct": round(_pct_delta(shipping_ms, single_ms), PCT_PRECISION_DIGITS),
            "speedup_vs_shipping": round(_speedup(shipping_ms, single_ms), RATIO_PRECISION_DIGITS),
        })
        loo_ablations.append({
            "mechanism_id": mid,
            "ref": pr_item.get("ref"),
            "notes": pr_item.get("notes", ""),
            "without_pr_step_ns": without_ns,
            "without_pr_step_ms": round(without_ms, MS_PRECISION_DIGITS),
            "delta_vs_without_pr_ns": proposed_ns - without_ns,
            "delta_vs_without_pr_ms": round((proposed_ns - without_ns) / NS_PER_MS, MS_PRECISION_DIGITS),
            "delta_vs_without_pr_pct": round(_pct_delta(without_ms, proposed_ms), PCT_PRECISION_DIGITS),
            "speedup_vs_without_pr": round(_speedup(without_ms, proposed_ms), RATIO_PRECISION_DIGITS),
        })
    pr_ablations.sort(key=lambda row: row["delta_vs_shipping_ns"])
    loo_ablations.sort(key=lambda row: row["delta_vs_without_pr_ns"])
    return pr_ablations, loo_ablations


def _build_waterfall_payload(
    *,
    analyses: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    step_ns: Mapping[str, int],
    roof_ns: int,
    roof_1p0_ms: float,
    pr_ablations: list[dict[str, Any]],
    loo_ablations: list[dict[str, Any]],
) -> dict[str, Any]:
    eager_ns, ship_ns = step_ns["eager"], step_ns["shipping"]
    prop_ns, targ_ns = step_ns["proposed"], step_ns["target"]
    eager_ms, ship_ms = eager_ns / NS_PER_MS, ship_ns / NS_PER_MS
    prop_ms, targ_ms = prop_ns / NS_PER_MS, targ_ns / NS_PER_MS
    return {
        "eager": {
            "step_ns": eager_ns,
            "step_ms": round(eager_ms, MS_PRECISION_DIGITS),
            "operator_sum_ms": round(float(analyses["eager"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
            "hbm_bytes": float(analyses["eager"]["total_hbm_bytes"]),
            "network": _network(analyses["eager"]),
            "active_mechanisms": [m["id"] for m in reports["eager"].get("active_mechanisms", ())],
        },
        "shipping": {
            "step_ns": ship_ns,
            "step_ms": round(ship_ms, MS_PRECISION_DIGITS),
            "operator_sum_ms": round(float(analyses["shipping"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
            "hbm_bytes": float(analyses["shipping"]["total_hbm_bytes"]),
            "network": _network(analyses["shipping"]),
            "delta_vs_eager_ns": ship_ns - eager_ns,
            "delta_vs_eager_ms": round((ship_ns - eager_ns) / NS_PER_MS, MS_PRECISION_DIGITS),
            "delta_vs_eager_pct": round(_pct_delta(eager_ms, ship_ms), PCT_PRECISION_DIGITS),
            "speedup_vs_eager": round(_speedup(eager_ms, ship_ms), RATIO_PRECISION_DIGITS),
            "active_mechanisms": [m["id"] for m in reports["shipping"].get("active_mechanisms", ())],
        },
        "proposed": {
            "step_ns": prop_ns,
            "step_ms": round(prop_ms, MS_PRECISION_DIGITS),
            "operator_sum_ms": round(float(analyses["proposed"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
            "hbm_bytes": float(analyses["proposed"]["total_hbm_bytes"]),
            "network": _network(analyses["proposed"]),
            "delta_vs_shipping_ns": prop_ns - ship_ns,
            "delta_vs_shipping_ms": round((prop_ns - ship_ns) / NS_PER_MS, MS_PRECISION_DIGITS),
            "delta_vs_shipping_pct": round(_pct_delta(ship_ms, prop_ms), PCT_PRECISION_DIGITS),
            "speedup_vs_shipping": round(_speedup(ship_ms, prop_ms), RATIO_PRECISION_DIGITS),
            "active_open_prs": [
                {"mechanism_id": m["mechanism_id"], "ref": m.get("ref"), "notes": m.get("notes", "")}
                for m in reports["proposed"].get("unmerged_pr_mechanisms", ())
            ],
            "single_pr_ablation": pr_ablations,
            "leave_one_out_ablation": loo_ablations,
        },
        "target": {
            "step_ns": targ_ns,
            "step_ms": round(targ_ms, MS_PRECISION_DIGITS),
            "operator_sum_ms": round(float(analyses["target"]["operator_sum_ms"]), MS_PRECISION_DIGITS),
            "hbm_bytes": float(analyses["target"]["total_hbm_bytes"]),
            "network": _network(analyses["target"]),
            "delta_vs_proposed_ns": targ_ns - prop_ns,
            "delta_vs_proposed_ms": round((targ_ns - prop_ns) / NS_PER_MS, MS_PRECISION_DIGITS),
            "delta_vs_proposed_pct": round(_pct_delta(prop_ms, targ_ms), PCT_PRECISION_DIGITS),
            "delta_vs_shipping_ns": targ_ns - ship_ns,
            "delta_vs_shipping_ms": round((targ_ns - ship_ns) / NS_PER_MS, MS_PRECISION_DIGITS),
            "delta_vs_shipping_pct": round(_pct_delta(ship_ms, targ_ms), PCT_PRECISION_DIGITS),
            "speedup_vs_shipping": round(_speedup(ship_ms, targ_ms), RATIO_PRECISION_DIGITS),
            "unimplemented_architectural_gaps": list(reports["target"].get("unimplemented_gaps", ())),
            "active_mechanisms": [m["id"] for m in reports["target"].get("active_mechanisms", ())],
        },
        "hardware_roofline": {
            "physical_ideal_1p0x_ns": roof_ns,
            "physical_ideal_1p0x_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
            "remaining_kernel_micro_gap_to_1p0x_ns": max(0, targ_ns - roof_ns),
            "remaining_kernel_micro_gap_to_1p0x_ms": round(max(0.0, (targ_ns - roof_ns) / NS_PER_MS), MS_PRECISION_DIGITS),
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
    scope: str = "model",
    ablate_prs: bool = True,
    **compose_kwargs: Any,
) -> dict[str, Any]:
    """Compute canonical 4-track comparison (eager -> shipping -> proposed -> target) with integer-ns deltas."""
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.efficiency import analyze_network_efficiency

    cases: dict[str, dict[str, Any]] = {}
    analyses: dict[str, dict[str, Any]] = {}
    step_ns: dict[str, int] = {}
    base_kwargs = dict(compose_kwargs)
    ship_case = compose_case(
        model,
        hardware=hardware,
        runtime_stack=runtime_stack,
        feature_track="shipping",
        _stack_phase=phase,
        **base_kwargs,
    )
    pinned_kwargs = dict(base_kwargs)
    if "quant" not in pinned_kwargs:
        ss_ship = ((ship_case.get("runtime") or {}).get("metadata") or {}).get("single_spine") or {}
        if ss_ship.get("checkpoint_origin") == "verified_checkpoint_descriptor" and ss_ship.get("checkpoint_id"):
            pinned_kwargs["quant"] = "auto"
        elif ship_case.get("model", {}).get("quant") is not None:
            pinned_kwargs["quant"] = dict(ship_case["model"]["quant"])
        else:
            pinned_kwargs["quant"] = "bf16"
    ship_req = (
        (((ship_case.get("runtime") or {}).get("metadata") or {}).get("single_spine") or {})
        .get("contract_inputs", {})
        .get("requested_options", {})
    )
    if isinstance(ship_req, Mapping):
        for cache_key in ("kv_cache_dtype", "indexer_kv_dtype"):
            if cache_key not in pinned_kwargs and ship_req.get(cache_key):
                pinned_kwargs[cache_key] = ship_req[cache_key]
    for track_name in ("eager", "shipping", "proposed", "target"):
        case_dict = (
            ship_case
            if track_name == "shipping"
            else compose_case(
                model,
                hardware=hardware,
                runtime_stack=runtime_stack,
                feature_track=track_name,
                _stack_phase=phase,
                **pinned_kwargs,
            )
        )
        eff = analyze_network_efficiency(
            case_dict,
            phase=phase,
            tokens=tokens,
            context=context,
            sequences=sequences,
            scope=scope,
        )
        cases[track_name] = case_dict
        analyses[track_name] = eff
        step_ns[track_name] = _ms_to_ns(float(eff["graph_duration_ms"]))

    eager_ns, ship_ns = step_ns["eager"], step_ns["shipping"]
    prop_ns, targ_ns = step_ns["proposed"], step_ns["target"]
    target_eff = analyses["target"]
    roof_1p0_ms = max(
        float(target_eff.get("critical_path_roofline_ms") or 0.0),
        float(target_eff.get("physical_resource_cut_ms") or 0.0),
    ) or float(target_eff.get("total_roofline_ideal_ms") or 0.0)
    roof_ns = _ms_to_ns(roof_1p0_ms)

    reports = {
        t: (cases[t]["runtime"].get("metadata") or {}).get("feature_track_report") or {}
        for t in ("eager", "shipping", "proposed", "target")
    }
    pr_ablations: list[dict[str, Any]] = []
    loo_ablations: list[dict[str, Any]] = []
    if ablate_prs and reports["proposed"].get("unmerged_pr_mechanisms"):
        pr_ablations, loo_ablations = _compare_pr_ablations(
            model, hardware, runtime_stack, reports["proposed"],
            ship_ns, prop_ns,
            phase=phase, tokens=tokens, context=context, sequences=sequences,
            scope=scope, compose_kwargs=pinned_kwargs,
        )

    tol_ns = _ms_to_ns(MONOTONIC_TOLERANCE_MS)
    eager_monotonic = eager_ns + tol_ns >= ship_ns
    monotonic = (
        eager_monotonic
        and ship_ns + tol_ns >= prop_ns
        and prop_ns + tol_ns >= targ_ns
        and targ_ns + tol_ns >= roof_ns
    )
    return {
        "schema": WATERFALL_SCHEMA,
        "model": cases["shipping"]["model"]["name"],
        "hardware_sku": cases["shipping"]["machine"]["sku"],
        "runtime_stack": reports["shipping"].get("stack", "vllm-ascend"),
        "repository": reports["shipping"].get("repository", ""),
        "geometry": {
            "phase": phase, "tokens": int(tokens), "context": int(context),
            "sequences": int(sequences), "scope": scope,
        },
        "monotonic_verified": bool(monotonic),
        "eager_vs_shipping_monotonic": bool(eager_monotonic),
        "waterfall": _build_waterfall_payload(
            analyses=analyses, reports=reports, step_ns=step_ns,
            roof_ns=roof_ns, roof_1p0_ms=roof_1p0_ms,
            pr_ablations=pr_ablations, loo_ablations=loo_ablations,
        ),
        "efficiency_reference": {
            "physical_ideal_ms": round(roof_1p0_ms, MS_PRECISION_DIGITS),
            "basis": "physical_1p0x_critical_path_roofline",
        },
    }


# --- Layer 2 RuntimeStackProfile Registry & JSON Manifest Loader ------------

STACK_MANIFEST_SCHEMA = "mindie-sim.stack-manifest/1"
VLLM_ASCEND_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "stacks" / "vllm_ascend" / "manifest.json"
)
MAX_MANIFEST_BYTES = 50 * 1024 * 1024
MIN_CAPTURE_SEQS = 16

TRACK_ALIASES: dict[str, str] = {
    "eager": "eager",
    "shipping": "shipping",
    "proposed": "proposed",
    "target": "target",
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
class RuntimeStackProfile:
    """Declarative metadata for one serving runtime stack (zero built-in model-name gates)."""

    name: str
    repository: str
    description: str = ""
    bindings: tuple[Any, ...] = ()
    supported_models: tuple[str, ...] = ()
    unmerged_pr_models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": STACK_MANIFEST_SCHEMA,
            "name": self.name,
            "repository": self.repository,
            "description": self.description,
            "bindings": [b.to_dict() if hasattr(b, "to_dict") else dict(b) for b in self.bindings],
        }
        if self.supported_models:
            payload["supported_models"] = list(self.supported_models)
        if self.unmerged_pr_models:
            payload["unmerged_pr_models"] = list(self.unmerged_pr_models)
        return payload


StackMechanismBinding = ManifestBinding


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
    """Load a ``RuntimeStackProfile`` strictly into canonical ``ManifestBinding`` records."""
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
    bindings: list[ManifestBinding] = []
    for item in (raw.get("bindings") or ()):
        if not isinstance(item, Mapping):
            raise ModelingError("stack_manifest_binding_invalid")
        mid = str(item.get("mechanism_id") or "").strip()
        if mid not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{mid}")
        raw_models = tuple(str(m) for m in (item.get("models") or ()) if str(m).strip())
        if raw_models:
            raise ModelingError(f"stack_manifest_models_forbidden_ag014:{mid}")
        raw_preds = item.get("capability_predicates") or ()
        if isinstance(raw_preds, (str, bytes)) or not isinstance(raw_preds, Sequence):
            raise ModelingError(f"stack_manifest_capability_predicates_invalid:{mid}")
        cap_preds = tuple(str(p).strip() for p in raw_preds if str(p).strip())
        raw_opts = item.get("options") or {}
        if not isinstance(raw_opts, Mapping):
            raise ModelingError(f"stack_manifest_options_invalid:{mid}")
        lowering_pass_id = str(item.get("lowering_pass_id") or "").strip()
        if not lowering_pass_id:
            raise ModelingError(f"stack_manifest_missing_lowering_pass_id:{mid}")
        prov_raw = item.get("provenance") if isinstance(item.get("provenance"), Mapping) else {}
        source_sha = str(prov_raw.get("source_sha256") or item.get("source_sha256") or "").strip()
        if not source_sha:
            raise ModelingError(f"stack_manifest_missing_source_sha256:{mid}")
        prov = SourceRef(
            repository=str(prov_raw.get("repository") or raw.get("repository") or ""),
            revision=str(prov_raw.get("revision") or raw.get("manifest_revision") or ""),
            relative_path=str(prov_raw.get("relative_path") or "manifest.json"),
            source_sha256=source_sha,
        )
        raw_phase_scope = item.get("phase_scope") or item.get("phases") or ()
        if isinstance(raw_phase_scope, str):
            phase_scope = (raw_phase_scope.strip().lower(),) if raw_phase_scope.strip() else ()
        else:
            phase_scope = tuple(str(p).strip().lower() for p in raw_phase_scope if str(p).strip())
        bindings.append(ManifestBinding(
            mechanism_id=mid,
            state=str(item.get("state") or "").strip(),  # type: ignore[arg-type]
            ref=str(item.get("ref") or "").strip(),
            lowering_pass_id=lowering_pass_id,
            provenance=prov,
            models=(),
            archs=tuple(str(a).strip().lower() for a in (item.get("archs") or ()) if str(a).strip()),
            phase_scope=phase_scope,
            capability_predicates=cap_preds,
            options=tuple(sorted((str(k), v) for k, v in raw_opts.items())),
            notes=str(item.get("notes") or ""),
        ))
    supported_models = tuple(
        str(m).strip() for m in (raw.get("supported_models") or ()) if str(m).strip()
    )
    unmerged_pr_models = tuple(
        str(m).strip() for m in (raw.get("unmerged_pr_models") or ()) if str(m).strip()
    )
    return RuntimeStackProfile(
        name=str(raw["name"]).strip(),
        repository=str(raw.get("repository") or ""),
        description=str(raw.get("description") or ""),
        bindings=tuple(bindings),
        supported_models=supported_models,
        unmerged_pr_models=unmerged_pr_models,
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
    if lower == "raw-eager" and "raw_eager" in RUNTIME_STACKS:
        return RUNTIME_STACKS["raw_eager"]
    candidate = Path(key).expanduser()
    if candidate.is_file():
        return load_runtime_stack_manifest(candidate)
    raise ModelingError(f"unknown_runtime_stack:{stack}")


def _collapse_binding_group(
    mid: str, state: str, group: Sequence[ManifestBinding],
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


def assert_model_supported_by_runtime_stack(
    stack_profile: RuntimeStackProfile,
    *,
    track: str,
    model_key: str,
) -> None:
    """Enforce explicit custom-manifest model support lists on shipping/proposed tracks."""
    canonical_track = normalize_feature_track(track)
    if not (stack_profile.supported_models or stack_profile.unmerged_pr_models):
        return
    model_id = str(model_key or "").strip()
    if canonical_track == "shipping" and model_id not in stack_profile.supported_models:
        raise ModelingError(
            f"model_not_supported_by_runtime_stack:{stack_profile.name}:{model_id}:shipping"
        )
    if (
        canonical_track == "proposed"
        and model_id not in stack_profile.supported_models
        and model_id not in stack_profile.unmerged_pr_models
    ):
        raise ModelingError(
            f"model_not_supported_by_runtime_stack:{stack_profile.name}:{model_id}:proposed"
        )


def resolve_stack_track(
    stack_profile: RuntimeStackProfile,
    *,
    track: str,
    model_key: str = "",
    ctx: MechanismContext,
    optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve active mechanisms strictly via canonical ``StackTrack`` & structural predicates."""
    canonical_track = normalize_feature_track(track)
    effective_model_key = str(model_key or getattr(ctx.model, "name", "") or "").strip()
    assert_model_supported_by_runtime_stack(
        stack_profile, track=canonical_track, model_key=effective_model_key,
    )
    include_ids = tuple(str(x).strip() for x in optimizations if str(x).strip())
    exclude_ids = frozenset(str(x).strip() for x in exclude_optimizations if str(x).strip())
    for mid in (*include_ids, *exclude_ids):
        if mid not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{mid}")
    overlap_conflict = set(include_ids) & exclude_ids
    if overlap_conflict:
        raise ModelingError("conflicting_optimization_flags:" + ",".join(sorted(overlap_conflict)))

    phase = str((ctx.options or {}).get("phase") or "").strip().lower() or None
    par_raw = ctx.parallel or {}
    par_spec = ParallelSpec(
        dp=int(par_raw.get("dp", 1)), tp=int(par_raw.get("tp", 1)),
        pp=int(par_raw.get("pp", 1)), ep=int(par_raw.get("ep", 1)),
        cp=int(par_raw.get("cp", par_raw.get("dcp", 1))), pcp=int(par_raw.get("pcp", 1)),
    )
    opts_map = ctx.options or {}
    capabilities = extract_model_capabilities(
        ctx.model,
        arch=ctx.arch,
        parallel=par_spec,
        speculative_k=int(opts_map.get("speculative_k") or 0),
        sp_enabled=bool(opts_map.get("sp", False)),
    )

    merged_list: list[ManifestBinding] = []
    pr_list: list[ManifestBinding] = []
    for binding in stack_profile.bindings:
        if not isinstance(binding, ManifestBinding):
            continue
        if not binding.matches("", ctx.arch, phase=phase, capabilities=capabilities):
            continue
        mech = MECHANISM_CATALOG.get(binding.mechanism_id)
        if mech is None:
            continue
        b_opts = {**dict(ctx.options or {}), **dict(binding.options)}
        b_ctx = MechanismContext(
            model=ctx.model, sku=ctx.sku, arch=ctx.arch, parallel=ctx.parallel,
            caller_explicit=ctx.caller_explicit, options=b_opts,
        )
        if not mech.eligible(b_ctx):
            continue
        if binding.state == "merged":
            merged_list.append(binding)
        elif binding.state == "unmerged_pr":
            pr_list.append(binding)

    grouped_merged: dict[str, list[ManifestBinding]] = {}
    for b in merged_list:
        grouped_merged.setdefault(b.mechanism_id, []).append(b)
    grouped_pr: dict[str, list[ManifestBinding]] = {}
    for b in pr_list:
        grouped_pr.setdefault(b.mechanism_id, []).append(b)

    active_by_id: dict[str, dict[str, Any]] = {}
    if canonical_track in {"shipping", "proposed", "target"}:
        for mid, group in grouped_merged.items():
            if mid not in exclude_ids:
                active_by_id[mid] = _collapse_binding_group(mid, "merged", group)
    if canonical_track in {"proposed", "target"}:
        for mid, group in grouped_pr.items():
            if mid not in exclude_ids:
                if mid in active_by_id:
                    merged_entry = dict(active_by_id[mid])
                    pr_entry = _collapse_binding_group(mid, "unmerged_pr", group)
                    merged_entry["options"] = {**dict(merged_entry["options"]), **dict(pr_entry["options"])}
                    merged_entry["state"] = "unmerged_pr"
                    if pr_entry.get("ref"):
                        merged_entry["ref"] = pr_entry["ref"]
                    active_by_id[mid] = merged_entry
                else:
                    active_by_id[mid] = _collapse_binding_group(mid, "unmerged_pr", group)

    unimplemented_gaps: list[dict[str, Any]] = []
    if canonical_track == "target":
        target_opts = {**dict(ctx.options or {}), "allow_extended_activation_abi": True}
        if "enable_fused_mc2" not in ctx.caller_explicit and target_opts.get("enable_fused_mc2") == 0:
            target_opts.pop("enable_fused_mc2", None)
        target_ctx = MechanismContext(
            model=ctx.model, sku=ctx.sku, arch=ctx.arch, parallel=ctx.parallel,
            caller_explicit=ctx.caller_explicit, options=target_opts,
        )
        for mid, mech in MECHANISM_CATALOG.items():
            if mid in exclude_ids:
                continue
            if phase and mech.phases and phase not in mech.phases:
                continue
            if not mech.eligible(target_ctx):
                continue
            target_entry_opts: dict[str, Any] = {}
            if mid == "mhc_boundary_fusion_overlap":
                target_entry_opts["unbounded_rows"] = True
            elif mid == "cann_megamoe_fusion":
                target_entry_opts["allow_extended_activation_abi"] = True
            if mid in active_by_id:
                if target_entry_opts:
                    prev = active_by_id[mid]
                    active_by_id[mid] = {
                        **prev,
                        "options": {**dict(prev.get("options") or {}), **target_entry_opts},
                    }
            else:
                active_by_id[mid] = {
                    "id": mid,
                    "state": "target_gap",
                    "ref": None,
                    "options": target_entry_opts,
                    "notes": "theoretical_target_gap",
                }
                unimplemented_gaps.append({
                    "mechanism_id": mid,
                    "category": mech.category,
                    "summary": mech.summary,
                })

    for mid in include_ids:
        if mid in active_by_id:
            continue
        mech = MECHANISM_CATALOG[mid]
        opt_ctx = MechanismContext(
            model=ctx.model, sku=ctx.sku, arch=ctx.arch, parallel=ctx.parallel,
            caller_explicit=ctx.caller_explicit,
            options={**dict(ctx.options or {}), "explicit_opt_in": True},
        )
        if mech.eligible(opt_ctx):
            active_by_id[mid] = {
                "id": mid,
                "state": "caller_override",
                "ref": None,
                "options": {},
                "notes": "caller_enabled",
            }

    ordered_active = [active_by_id[mid] for mid in MECHANISM_CATALOG if mid in active_by_id]
    return {
        "stack": stack_profile.name,
        "repository": stack_profile.repository,
        "feature_track": canonical_track,
        "arch": ctx.arch,
        "active_mechanisms": ordered_active,
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
    is_target = str(resolved_track.get("feature_track") or "").strip().lower() == "target"
    unlock_profile_mc2 = is_target and "enable_fused_mc2" not in base_ctx.caller_explicit
    if unlock_profile_mc2:
        meta = dict(runtime_dict.get("metadata") or {})
        additional = meta.get("additional_config")
        if isinstance(additional, Mapping) and additional.get("enable_fused_mc2") == 0:
            cleaned = dict(additional)
            cleaned.pop("enable_fused_mc2", None)
            meta["additional_config"] = cleaned
            runtime_dict["metadata"] = meta
    active = {entry["id"]: entry for entry in resolved_track["active_mechanisms"]}
    for mid, mech in MECHANISM_CATALOG.items():
        entry = active.get(mid)
        if entry is None:
            continue
        opts = {**dict(base_ctx.options or {}), **dict(entry.get("options") or {})}
        if unlock_profile_mc2 and opts.get("enable_fused_mc2") == 0:
            opts.pop("enable_fused_mc2", None)
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


# --- Unified Software Stack Registry & Lowering-Base Resolver ---------------


@dataclass(frozen=True, slots=True)
class SoftwareStackSpec:
    """Declarative specification of one registered inference software stack."""

    name: str
    repository: str
    description: str
    profile: RuntimeStackProfile
    sources: tuple[SourceRef, ...] = ()
    default_track: str = "shipping"
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
    """Resolve draft/MTP proposal token count strictly validated via ``catalog.draft_spec``."""
    dspark = getattr(case.runtime, "dspark", None)
    if dspark is not None:
        step_spec = dspark.step_spec()
        if getattr(step_spec, "draft_len", 0):
            return validate_fia_speculative_tokens(int(step_spec.draft_len))
    meta = dict(case.runtime.metadata)
    raw_spec = meta.get("speculative")
    if raw_spec:
        if isinstance(raw_spec, SpeculativeSpec):
            spec_obj = raw_spec
        else:
            parsed = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
            spec_obj = SpeculativeSpec.from_mapping(parsed) if isinstance(parsed, Mapping) else None
        if spec_obj is not None and spec_obj.draft_len:
            return validate_fia_speculative_tokens(int(spec_obj.draft_len))
    if meta.get("draft_proposal_tokens"):
        return validate_fia_speculative_tokens(int(meta["draft_proposal_tokens"]))
    aux = getattr(case.model, "auxiliary", None)
    mtp = getattr(aux, "mtp", None) if aux is not None else None
    if mtp is not None and getattr(mtp, "proposal_tokens", 0):
        return validate_fia_speculative_tokens(int(mtp.proposal_tokens))
    model_spec = getattr(case.model, "speculative", None)
    if model_spec is not None and getattr(model_spec, "proposal_tokens", 0):
        return validate_fia_speculative_tokens(int(model_spec.proposal_tokens))
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
    max_seqs = max(
        MIN_CAPTURE_SEQS,
        int(case.runtime.max_num_seqs or MIN_CAPTURE_SEQS),
        request.step.sequences if isinstance(request.step.sequences, int)
        else max(request.step.sequences, default=1),
    )
    mtp = _mtp_proposal_tokens(case)
    rule = build_capture_bucket_rule(
        cudagraph_mode="FULL_DECODE_ONLY" if target_mode == "full" else target_mode.upper(),
        num_speculative_tokens=mtp,
        max_num_seqs=max_seqs,
    )
    keys = tuple(case.runtime.graph_capture_keys or rule.full_keys)
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
    resolved = resolve_stack_track(stack.profile, track="target", ctx=ctx)
    apply_resolved_mechanisms(rt_dict, resolved, ctx)
    finalize_compilation_spine(rt_dict, raw_eager=False)
    new_runtime = Software.from_mapping(rt_dict)
    return replace(case, runtime=new_runtime)


class SoftwareStackRegistry:
    """Registry facade delegating stack resolution to canonical ``stacks.registry``."""

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
        """Resolve stack profile, canonical track tier, capture policy, and binding origins."""
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


register_runtime_stack(load_runtime_stack_manifest(VLLM_ASCEND_MANIFEST_PATH))
register_runtime_stack(RuntimeStackProfile(
    name="raw_eager",
    repository="eager-baseline",
    description="Unfused eager PyTorch/ATen execution baseline with zero optional fusions",
    bindings=(),
))


@lru_cache(maxsize=1)
def default_stack_registry() -> SoftwareStackRegistry:
    """Return the built-in ``SoftwareStackRegistry`` facade."""
    specs = [
        SoftwareStackSpec(
            name=profile.name,
            repository=profile.repository,
            description=profile.description,
            profile=profile,
            default_track="eager" if profile.name in {"raw_eager", "raw-eager"} else "shipping",
            decode_capture_mode="none" if profile.name in {"raw_eager", "raw-eager"} else "full",
        )
        for profile in RUNTIME_STACKS.values()
    ]
    return SoftwareStackRegistry(specs)


StackRegistry = SoftwareStackRegistry
