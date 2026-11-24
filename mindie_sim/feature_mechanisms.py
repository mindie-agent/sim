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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

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
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.model import ModelSpec

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
            id="in_kernel_gdn_decode_gate", category="recurrent", phases=("decode",),
            summary="Fold GDN gating/L2Norm into recurrent_gated_delta_rule and skip output buffer zeroing",
            eligible=lambda ctx: (
                ctx.arch == "a5"
                and any(p.attention == "gdn" for p in ctx.model.layer_plans)
            ),
            apply=_apply_in_kernel_gdn_decode_gate,
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
    """Unplugged socket: split-brain waterfall builder and legacy track wire protocol purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: compute canonical 4-track comparison ('eager'|'shipping'|'proposed'|'target') "
        "strictly from PredictionSpine + EngineStepReceipt with exact integer-ns deltas"
    )


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
    """Declarative metadata for one serving runtime stack (zero model-name gates)."""

    name: str
    repository: str
    description: str = ""
    bindings: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STACK_MANIFEST_SCHEMA,
            "name": self.name,
            "repository": self.repository,
            "description": self.description,
            "bindings": [b.to_dict() if hasattr(b, "to_dict") else dict(b) for b in self.bindings],
        }


class StackMechanismBinding:
    """Unplugged socket: split-brain StackMechanismBinding schema purged."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "LEGACY_PURGED: StackMechanismBinding purged; use canonical "
            "stacks/vllm_ascend/profiles.ManifestBinding with capability_predicates"
        )


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
    """Unplugged socket: split-brain StackMechanismBinding manifest loader purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: load runtime stack manifests strictly via "
        "schema.stack.ManifestBinding + stacks/vllm_ascend/profiles.load_manifest_bindings"
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


def resolve_stack_track(
    stack_profile: RuntimeStackProfile,
    *,
    track: str,
    model_key: str = "",
    ctx: MechanismContext,
    optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (),
) -> dict[str, Any]:
    """Unplugged socket: legacy track protocol and model-name gates purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: route stack track resolution strictly through "
        "schema.stack.RuntimeStackAdapter + ManifestBinding capability_predicates "
        "using canonical StackTrack ('eager'|'shipping'|'proposed'|'target')"
    )


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
    """Unplugged socket: shadow MTP proposal calculator purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: consume SpeculativeExecutionContract (catalog/draft_spec.py) "
        "for proposal token counts instead of shadow _mtp_proposal_tokens"
    )


def _apply_production_decode_capture(
    case: Case,
    request: LoweringRequest,
    stack: SoftwareStackSpec,
) -> Case:
    raise NotImplementedError(
        "LEGACY_PURGED: wire capture bucket resolution through RuntimeStackAdapter "
        "+ SpeculativeExecutionContract"
    )


def _apply_tier_policy(
    case: Case,
    request: LoweringRequest,
    stack: SoftwareStackSpec,
) -> Case:
    """Unplugged socket: legacy ideal-track 3-tier shim purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: retire legacy ideal-track 3-tier shim; "
        "route strictly through canonical StackTrack ('eager'|'shipping'|'proposed'|'target')"
    )


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
        """Unplugged socket: duplicate SoftwareStackRegistry base resolver purged."""
        raise NotImplementedError(
            "LEGACY_PURGED: route stack base resolution solely through "
            "stacks.registry + RuntimeStackAdapter (zero duplicate SoftwareStackRegistry)"
        )


@lru_cache(maxsize=1)
def default_stack_registry() -> SoftwareStackRegistry:
    """Return the built-in ``SoftwareStackRegistry`` facade."""
    specs = [
        SoftwareStackSpec(
            name=profile.name,
            repository=profile.repository,
            description=profile.description,
            profile=profile,
            default_track="eager" if profile.name == "raw-eager" else "shipping",
            decode_capture_mode="none" if profile.name == "raw-eager" else "full",
        )
        for profile in RUNTIME_STACKS.values()
    ]
    return SoftwareStackRegistry(specs)
