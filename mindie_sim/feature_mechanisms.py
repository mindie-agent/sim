"""Layer 1 stack-agnostic optimization mechanism catalog and case mutators.

[Role]
  Define universal, framework-neutral physical optimization mechanisms
  (graph capture, sequence parallelism, operator fusions, strided zero-copy
  views, multi-stream compute/comm overlap, compact state recovery) and apply
  selected mechanisms deterministically to a Case's runtime configuration.

[Invariants]
  * Mechanism IDs carry zero framework-specific prefixes (no ``va_`` / ``vllm_``).
  * Structural eligibility inspects ``ModelSpec`` / ``LayerPlan`` / ``ParallelSpec``
    geometry, never model-name strings.
  * Explicit caller arguments always win over track/mechanism defaults.
  * External extensions may register new mechanisms via
    ``register_optimization_mechanism`` without editing core dispatch.

[Timeouts & Locks]
  Pure in-memory dictionary and tuple mutations; O(layers + mechanisms), no I/O.

[Side-Effect Policy]
  Registry mutations occur only via explicit ``register_optimization_mechanism``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mindie_sim.feature_mechanism_catalog import (
    MX_SCALE_GROUP_SIZE,
    PASS_PRIORITY_ORDER,
    _append_passes,
    _apply_mhc_fusion_overlap,
    _apply_shared_expert_overlap,
    _has_quant_activations,
    _set_layout_strided,
    _set_lowering,
    source_mechanisms as _source_mechanisms,
)
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
    from mindie_sim.prediction.sequence_parallel import DECLARATION, _validate_model
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
    from mindie_sim.prediction.family_mla_prolog import validate as validate_mla_prolog
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
        from mindie_sim.prediction.mla_prolog_packed import LAYOUT
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
            from mindie_sim.prediction.kda_source import resolve as resolve_kda_source
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
    from mindie_sim.prediction.gdn_domain import model_supports_gdn_chunk64

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
        eligible=lambda ctx: _has_quant_activations(ctx.model),
        apply=lambda rt, _ctx: _append_passes(rt, "activation_quantization", "activation_quant"),
    ),
    OptimizationMechanism(
        id="gmm_swiglu_quant_fusion", category="ffn_fusion", phases=("prefill", "decode"),
        summary="On-chip AIC->UB GroupedMatmul + SwiGLU/SiTU + Quant eliminating gate_up HBM round-trip",
        eligible=lambda ctx: (any(p.ffn == "moe" for p in ctx.model.layer_plans)
                              and _has_quant_activations(ctx.model)),
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

for _source_item in _source_mechanisms():
    register_optimization_mechanism(_source_item)
