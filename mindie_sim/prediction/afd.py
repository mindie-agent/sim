"""Attention-FFN disaggregation (AFD), M-step pipeline and Pareto solver.

[Role] Split a body iteration into an Attention-pool stage and an FFN-pool
  stage, price cross-pool AF dispatch/combine from explicit physical links,
  evaluate the M-step micro-batch pipeline and AFD feasibility constraints, and
  solve the attention/FFN die-count plus M-step Pareto set under a TPOT SLA.
[Invariants] Pool work comes only from the shared efficiency graph; no vendor,
  model-name or serving-floor branch selects a pool or a duration. Link rates
  are the short board (min effective, min peak, max latency) of the declared
  Attention and FFN physical links. Bytes, units and stage boundaries are
  conserved; a candidate is never silently dropped, it is recorded with its
  exact failure reason.
[Timeouts & Locks] Pure in-memory arithmetic and bounded DAG evaluation; no
  I/O, subprocesses, sleeps, device contexts or locks.
[Side-Effect Policy] Inputs are never mutated; every result is a freshly built
  dictionary and every candidate row owns its own records.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral, Real
import re
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.hardware import Hardware

from .efficiency import analyze_network_efficiency
from .structure import parse_case

_AFD_SCHEMA = "mindie-sim.afd-prediction/1"
_AFD_PARETO_SCHEMA = "mindie-sim.afd-pareto/1"
_PARETO_EPS = 1e-12
_DEFAULT_POOL_LINK_TIER = "inter"
_LINK_TIERS = frozenset({"intra", "inter"})

# Explicit, structure-derived pool membership. Operator IDs and recipes are the
# primary signal (models build them from attention/FFN structure); families are
# the fallback, and declaration order is the final fallback for unrelated
# communication or elementwise rows inside one of the two body pools.
_ATTENTION_FAMILIES = frozenset({
    "attention", "attention_online", "mla", "mla_decode", "mla_prolog",
    "compressed_attention", "qkv_projection", "qkv_preprocess", "rotary",
    "cache_store", "cache_transfer", "output_projection", "attention_residual",
    "attn_res_prefix", "dsa", "indexer", "index_topk", "pooled_indexer",
    "key_pool", "gdn_prepare", "gdn_state", "gdn_output", "gdn_gating",
    "recurrent_local", "recurrent_scan", "recurrent_vector", "recurrent_output",
    "kda", "short_convolution", "state_gather", "state_scatter",
})
_FFN_FAMILIES = frozenset({
    "moe", "moe_align", "moe_combine", "moe_init_routing", "router", "routing",
    "topk", "grouped_gemm", "gate_up_projection", "down_projection", "swiglu",
    "situ", "situ_mx_quant", "dequant_swiglu_quant", "mxfp4_weight_supply",
    "prefix_sum", "histogram", "expert_add",
})
_SHARED_FAMILIES = frozenset({
    "embedding", "logits", "sampling", "acceptance", "token_commit", "confidence", "lm_head", "pp_p2p", "vision",
})
_ATTENTION_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"(?:^|[._\s])norm1(?:$|[._\s])", r"qkv", r"(?:^|[._\s])q_a(?:$|[._\s])", r"(?:^|[._\s])kv_a(?:$|[._\s])",
    r"(?:^|[._\s])q_b(?:$|[._\s])", r"q_absorb", r"q_norm", r"kv_norm", r"v_expand", r"rope|rotary",
    r"cache|kv_store", r"attention|attn", r"(?:^|[._\s])output(?:$|[._\s])",
    r"output_projection|attention_output|output_reduce", r"flash", r"mla|index", r"(?:^|[._\s])o_proj(?:$|[._\s])",
))
_FFN_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"(?:^|[._\s])norm2(?:$|[._\s])", r"(?:^|[._\s])ffn(?:$|[._\s])", r"(?:^|[._\s])moe(?:$|[._\s])",
    r"(?:^|[._\s])router(?:$|[._\s])", r"(?:^|[._\s])rout(?:e|ing)(?:$|[._\s])", r"(?:^|[._\s])topk(?:$|[._\s])",
    r"(?:^|[._\s])expert", r"(?:^|[._\s])dispatch(?:$|[._\s])", r"(?:^|[._\s])combine(?:$|[._\s])",
    r"gate_up", r"swiglu|situ", r"(?:^|[._\s])down(?:_|$|[._\s])", r"(?:^|[._\s])shared(?:$|[._\s])",
    r"(?:^|[._\s])join(?:$|[._\s])", r"(?:^|[._\s])pack(?:$|[._\s])", r"(?:^|[._\s])unpack",
))
_SHARED_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"(?:^|[._\s])final_norm(?:$|[._\s])", r"(?:^|[._\s])embedding(?:$|[._\s])",
    r"(?:^|[._\s])lm_head(?:$|[._\s])", r"(?:^|[._\s])logits(?:$|[._\s])",
    r"(?:^|[._\s])sampling(?:$|[._\s])", r"token_commit|acceptance|confidence",
))


def _check_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name}_must_be_int_ge_{minimum}:{value}")
    return int(value)


def _check_float(name: str, value: Any, *, minimum: float, strict: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name}_must_be_finite_real:{value}")
    result = float(value)
    if result < minimum or (strict and result == minimum):
        raise ValueError(f"{name}_must_be_{'>' if strict else '>='}_{minimum}:{value}")
    return result


def _check_tier(pool_link_tier: Any) -> str:
    tier = str(pool_link_tier).strip().lower()
    if tier not in _LINK_TIERS:
        raise ValueError(f"afd_pool_link_tier_unsupported:{pool_link_tier}")
    return tier


def _link_field(link: Any, name: str, default: Any = None) -> Any:
    return link.get(name, default) if isinstance(link, Mapping) else getattr(link, name, default)


def _link_key(link: Any) -> str:
    return f"{_link_field(link, 'channel', '') or ''} {_link_field(link, 'name', '') or ''}".lower()


def _link_values(link: Any) -> tuple[float, float, float]:
    """Return (effective_bytes_per_ms, peak_bytes_per_ms, latency_ms)."""
    raw_capacity = _link_field(link, "capacity_bytes_per_ms")
    try:
        capacity = float(raw_capacity)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"pool_link_capacity_invalid:{raw_capacity}") from exc
    raw_peak = _link_field(link, "effective_peak_capacity_bytes_per_ms", None)
    if raw_peak is None:
        raw_peak = _link_field(link, "peak_capacity_bytes_per_ms", None)
    raw_latency_ms = _link_field(link, "latency_ms", None)
    try:
        peak = capacity if raw_peak is None else float(raw_peak)
        util = float(_link_field(link, "bw_utilization", 1.0))
        if raw_latency_ms is not None:
            latency_ms = float(raw_latency_ms)
        else:
            prop_us = _link_field(link, "propagation_us", None)
            latency_us = (float(prop_us) if prop_us is not None else (
                float(_link_field(link, "latency_us", 0.0))
                + float(_link_field(link, "switch_hops", 0) or 0)
                * float(_link_field(link, "switch_hop_latency_us", 0.0) or 0.0)))
            latency_ms = latency_us / 1000.0
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"pool_link_rate_invalid:{_link_key(link)}") from exc
    if not all(math.isfinite(value) for value in (capacity, peak, util, latency_ms)):
        raise ModelingError(f"pool_link_rate_not_finite:{_link_key(link)}")
    if capacity <= 0.0 or peak <= 0.0 or latency_ms < 0.0:
        raise ModelingError(f"pool_link_rate_non_positive:{_link_key(link)}")
    if not 0.0 < util <= 1.0:
        raise ModelingError(f"pool_link_bw_utilization_invalid:{_link_key(link)}:{util}")
    # A declared capacity may already be the effective rate, the raw peak, or
    # the effective peak; the peak attribute may be either the raw or effective
    # value. Utilization can only reduce the larger declaration, so both forms
    # resolve to the same effective rate without assuming one particular schema.
    low = min(capacity, peak)
    high = max(capacity, peak)
    effective = min(low, high * util)
    if effective <= 0.0:
        raise ModelingError(f"pool_link_effective_rate_non_positive:{_link_key(link)}")
    return effective, peak, latency_ms


def _matching_links(hardware: Hardware, tier: str) -> list[Any]:
    raw_links = (hardware.get("links") if isinstance(hardware, Mapping) else getattr(hardware, "links", ())) or ()
    name = (hardware.get("name", "") if isinstance(hardware, Mapping) else getattr(hardware, "name", ""))
    links = list(raw_links)
    if not links:
        raise ModelingError(f"pool_link_unavailable:{tier}:{name}")
    groups = (("intra_node",), ("intra_package",), ("intra",)) if tier == "intra" else (
        ("inter_node",), ("inter",), ("intra_node",), ("intra_package",), ("intra",))
    for tokens in groups:
        matched = [link for link in links if any(token in _link_key(link) for token in tokens)]
        if matched:
            return matched
    return links


def _extract_tier_link(hw: Hardware, tier: str) -> tuple[float, float, float]:
    tier = _check_tier(tier)
    candidates = [(_link_values(link), _link_key(link)) for link in _matching_links(hw, tier)]
    (effective, peak, latency_ms), _ = max(candidates, key=lambda item: (item[0][0], -item[0][2], item[1]))
    return effective, peak, latency_ms


def effective_pool_link(
    attn_hardware: Hardware, ffn_hardware: Hardware, *,
    pool_link_tier: str = _DEFAULT_POOL_LINK_TIER,
    link_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the AFD cross-pool link after the Attention/FFN short-board rule."""
    tier = _check_tier(pool_link_tier)
    attn_effective, attn_peak, attn_latency = _extract_tier_link(attn_hardware, tier)
    ffn_effective, ffn_peak, ffn_latency = _extract_tier_link(ffn_hardware, tier)
    effective, peak, latency_ms = min(attn_effective, ffn_effective), min(attn_peak, ffn_peak), max(attn_latency, ffn_latency)
    bw_util = effective / peak if peak > 0.0 else 1.0
    if link_override is not None and not isinstance(link_override, Mapping):
        raise ValueError(f"afd_link_override_must_be_mapping:{type(link_override).__name__}")
    override: Mapping[str, Any] = link_override or {}
    recompute_effective = False
    if override.get("gbps") is not None:
        peak = _check_float("afd_link_override_gbps", override["gbps"], minimum=0.0, strict=True) * 1e6
        recompute_effective = True
    if override.get("bw_utilization") is not None:
        bw_util = _check_float("afd_link_override_bw_utilization", override["bw_utilization"], minimum=0.0, strict=True)
        if bw_util > 1.0:
            raise ValueError(f"afd_link_override_bw_utilization_must_be_le_1:{bw_util}")
        recompute_effective = True
    if recompute_effective:
        effective = peak * bw_util
    if override.get("latency_us") is not None:
        latency_ms = _check_float("afd_link_override_latency_us", override["latency_us"], minimum=0.0, strict=False) / 1000.0
    if effective <= 0.0 or peak <= 0.0:
        raise ValueError(f"afd_link_override_effective_rate_non_positive:{effective}")
    return {
        "pool_link_tier": tier, "effective_gbps": effective / 1e6, "peak_gbps": peak / 1e6,
        "effective_bytes_per_ms": effective, "latency_ms": latency_ms, "latency_us": latency_ms * 1000.0,
    }


def _operator_pool(row: Mapping[str, Any], phase: str) -> tuple[str | None, str]:
    """Classify one efficiency row into the attention/FFN pool or shared work."""
    block = str(row.get("structure_block") or "").strip().lower()
    if block == "attention":
        return "attention", "attention"
    if block in {"ffn", "moe", "mlp"}:
        return "ffn", "ffn"
    if block in {"embed", "embedding", "lm_head", "final_norm", "head"}:
        return None, phase
    family = str(row.get("family") or "").strip().lower()
    text = " ".join(str(row.get(key) or "") for key in ("op_id", "recipe", "family")).lower()
    if family in _SHARED_FAMILIES or any(pattern.search(text) for pattern in _SHARED_PATTERNS):
        return None, phase
    attention = family in _ATTENTION_FAMILIES or any(pattern.search(text) for pattern in _ATTENTION_PATTERNS)
    ffn = family in _FFN_FAMILIES or any(pattern.search(text) for pattern in _FFN_PATTERNS)
    if attention and ffn:
        return ("attention", "attention") if ("index" in text or "mla" in text) else (("ffn" if phase == "ffn" else "attention"), phase)
    if ffn:
        return "ffn", "ffn"
    if attention:
        return "attention", "attention"
    return phase, phase


def _split_pool_stage(report: Mapping[str, Any]) -> dict[str, Any]:
    """Split an efficiency report into attention and FFN-pool stage totals."""
    rows = list(report.get("operators") or ())
    operator_sum_ms = float(report.get("operator_sum_ms", 0.0) or 0.0) or sum(
        float(row.get("total_ms", 0.0) or 0.0) for row in rows)
    graph_ms = float(report.get("graph_duration_ms", operator_sum_ms) or operator_sum_ms)
    scale = (graph_ms / operator_sum_ms) if operator_sum_ms > _PARETO_EPS else 1.0
    attention_sum = ffn_sum = shared_sum = 0.0
    attention_count = ffn_count = shared_count = 0
    phase = "attention"
    for row in rows:
        total_ms = float(row.get("total_ms", 0.0) or 0.0)
        pool, phase = _operator_pool(row, phase)
        if pool == "attention":
            attention_sum, attention_count = attention_sum + total_ms, attention_count + 1
        elif pool == "ffn":
            ffn_sum, ffn_count = ffn_sum + total_ms, ffn_count + 1
        else:
            shared_sum, shared_count = shared_sum + total_ms, shared_count + 1
    return {
        "attention_ms": attention_sum * scale, "ffn_ms": ffn_sum * scale, "shared_ms": shared_sum * scale,
        "attention_operator_sum_ms": attention_sum, "ffn_operator_sum_ms": ffn_sum,
        "shared_operator_sum_ms": shared_sum, "attention_operator_count": attention_count,
        "ffn_operator_count": ffn_count, "shared_operator_count": shared_count,
        "operator_sum_ms": operator_sum_ms, "graph_duration_ms": graph_ms, "scale": scale,
    }


def _pipeline_model(
    attn_stage_ms: float, ffn_stage_ms: float, af_comm_ms: float, m_steps: int,
    *, sequences: int | None = None,
    a_subs: Sequence[float] | None = None, f_subs: Sequence[float] | None = None,
    x_subs: Sequence[float] | None = None,
) -> dict[str, Any]:
    from mindie_sim.prediction.engine import Engine, Resource

    serial_ms = attn_stage_ms + ffn_stage_ms + af_comm_ms
    eff_m = min(m_steps, max(1, int(sequences))) if sequences is not None else m_steps
    a_vec = tuple(a_subs) if a_subs is not None else (attn_stage_ms / eff_m,) * eff_m
    f_vec = tuple(f_subs) if f_subs is not None else (ffn_stage_ms / eff_m,) * eff_m
    x_vec = tuple(x_subs) if x_subs is not None else (af_comm_ms / eff_m,) * eff_m
    if eff_m == 2:
        engine = Engine([Resource("afd_compute", 1.0, "ms", "fifo"), Resource("afd_comm", 1.0, "ms", "fifo")])
        c_prev = x_prev = None
        for mb in range(2):
            c_prev = engine.submit(f"mb{mb}.compute", {"afd_compute": a_vec[mb] + f_vec[mb]}, after=(c_prev,) if c_prev else ())
            x_prev = engine.submit(f"mb{mb}.comm", {"afd_comm": x_vec[mb]}, after=(c_prev,) + ((x_prev,) if x_prev else ()))
        mode, steady_ms = "partial_overlap", max(attn_stage_ms + ffn_stage_ms, af_comm_ms)
    else:
        engine = Engine([Resource("afd_attn", 1.0, "ms", "fifo"), Resource("afd_comm", 1.0, "ms", "fifo"),
                         Resource("afd_ffn", 1.0, "ms", "fifo")])
        a_prev = x_prev = f_prev = None
        for mb in range(eff_m):
            a_prev = engine.submit(f"mb{mb}.attn", {"afd_attn": a_vec[mb]}, after=(a_prev,) if a_prev else ())
            x_prev = engine.submit(f"mb{mb}.comm", {"afd_comm": x_vec[mb]}, after=(a_prev,) + ((x_prev,) if x_prev else ()))
            f_prev = engine.submit(f"mb{mb}.ffn", {"afd_ffn": f_vec[mb]}, after=(x_prev,) + ((f_prev,) if f_prev else ()))
        mode = "serial" if eff_m == 1 else "full_overlap"
        steady_ms = serial_ms if eff_m == 1 else max(attn_stage_ms, ffn_stage_ms, af_comm_ms)
    engine.advance()
    step_ms, bubble_ms = float(engine.now), max(0.0, float(engine.now) - steady_ms)
    return {
        "m_steps": m_steps, "effective_m_steps": eff_m, "mode": mode, "serial_step_ms": serial_ms,
        "request_latency_ms": serial_ms, "steady_state_ms": steady_ms,
        "fill_drain_bubble_ms": bubble_ms, "bubble_ms": bubble_ms, "step_duration_ms": step_ms,
        "speedup_vs_serial": (serial_ms / step_ms) if step_ms > _PARETO_EPS else 1.0,
    }


_FFN_WEIGHT_CATS = frozenset({
    "dense_mlp", "router", "routed_expert", "latent_moe", "shared_expert",
})
_FFN_NORM_NAMES = frozenset({"ffn_norm", "post_ffn_norm", "latent_norm"})


def _validate_shared_model_contract(attn_parsed: CaseSpec, ffn_parsed: CaseSpec) -> None:
    am, fm = attn_parsed.model, ffn_parsed.model
    if (am.layers != fm.layers or am.hidden_size != fm.hidden_size
            or am.attention != fm.attention or am.ffn != fm.ffn):
        raise ValueError(
            f"afd_model_contract_mismatch:attn=({am.layers},{am.hidden_size},{am.attention},{am.ffn})"
            f"!=ffn=({fm.layers},{fm.hidden_size},{fm.attention},{fm.ffn})"
        )


def _pool_memory_check(
    parsed: CaseSpec, *, tokens: int, context: int, seqs_per_dp: int,
    is_attn_pool: bool, scope: str = "body",
) -> tuple[bool, str | None]:
    from mindie_sim.catalog.cache import local_kv_heads
    from mindie_sim.catalog.types import QuantSpec
    from mindie_sim.catalog.weight_terms import term_bytes
    from mindie_sim.catalog.weights import layer_weight_terms
    from mindie_sim.prediction.capacity import Memory

    if context + tokens > int(parsed.runtime.max_model_len):
        return False, "exceeds_max_model_len"
    if seqs_per_dp > int(parsed.runtime.max_num_seqs):
        return False, "exceeds_max_num_seqs"
    if tokens * seqs_per_dp > int(parsed.runtime.max_num_batched_tokens):
        return False, "exceeds_max_num_batched_tokens"
    ws_tokens = tokens * seqs_per_dp
    mem = Memory(parsed, workspace_tokens=ws_tokens)
    model, p = parsed.model, parsed.runtime.parallel
    quant = model.quant or QuantSpec("bf16")
    ffn_gross = [0.0] * p.pp
    pool_ws = [0.0] * p.pp
    act_b = dtype_bytes(model.activation_dtype)
    for layer in model.layer_plans:
        st = mem.layer_stage[layer.index]
        for term in layer_weight_terms(layer, model):
            if term.category in _FFN_WEIGHT_CATS or term.name in _FFN_NORM_NAMES:
                ffn_gross[st] += float(term_bytes(term, quant, p, parsed.machine.ranks_per_node))
        residual = model.hidden_size * (2 + max(1, layer.hc_mult))
        if layer.residual_block_size:
            residual += model.hidden_size * (layer.index // layer.residual_block_size + 1)
        if is_attn_pool:
            if layer.attention in {"kda", "gdn"}:
                sub = (layer.conv_width + layer.linear_value_heads * layer.linear_head_dim) / p.tp
            elif layer.attention == "gqa":
                sub = (layer.num_heads / p.tp + 2 * local_kv_heads(layer.num_kv_heads, p.tp)) * layer.head_dim
            else:
                sub = layer.q_lora_rank + layer.kv_lora_rank + layer.num_heads / p.tp * (
                    layer.qk_nope_head_dim + layer.qk_rope_head_dim + layer.v_head_dim)
        else:
            sub = (3 * layer.intermediate_size / p.tp if layer.ffn == "dense" else
                   layer.top_k * (2 * (layer.expert_latent_size or model.hidden_size)
                                  + 3 * layer.moe_intermediate_size / p.etp))
        pool_ws[st] = max(pool_ws[st], ws_tokens * (residual + sub) * act_b)
    ffn_res = [max(0.0, g - s) for g, s in zip(ffn_gross, mem.expert_hbm_saved)]
    pool_weights = [
        max(0.0, w - f) if is_attn_pool else f
        for w, f in zip(mem.weights, ffn_res)
    ]
    width = p.tp * p.pcp
    kv_per_seq = mem.cache_bytes(max(1, context + tokens)) if is_attn_pool else (0.0,) * p.pp
    for rep in range(p.dp):
        for st in range(p.pp):
            budget = min(
                d.hbm_bytes * parsed.runtime.gpu_memory_utilization
                for d in parsed.machine.dies[(rep * p.pp + st) * width:(rep * p.pp + st + 1) * width]
            )
            avail = budget - pool_weights[st] - mem.policy.reserved - pool_ws[st]
            if avail < 0.0:
                return False, "oom"
            if is_attn_pool and kv_per_seq[st] * seqs_per_dp > avail:
                return False, "kv_cache_oom"
    return True, None


def _build_report(
    *, attn_parsed: CaseSpec, ffn_parsed: CaseSpec,
    attn_stage: Mapping[str, Any], ffn_stage: Mapping[str, Any],
    attn_rep: Mapping[str, Any], ffn_rep: Mapping[str, Any],
    m_steps: int, pool_link_tier: str, link_override: Mapping[str, Any] | None,
    phase: str, tokens: int, context: int, sequences: int, scope: str,
    af_dispatch_dtype: str | None, af_combine_dtype: str | None,
    sla_tpot_ms: float, balance_tolerance: float,
    stage_cache: dict[tuple[int, int, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    attn_stage_ms = float(attn_stage["attention_ms"]) + float(attn_stage["shared_ms"])
    ffn_stage_ms = float(ffn_stage["ffn_ms"])
    attn_dp = max(1, int(attn_parsed.runtime.parallel.dp))
    ffn_dp = max(1, int(ffn_parsed.runtime.parallel.dp))
    attn_seqs_per_dp = max(1, math.ceil(sequences / attn_dp))
    ffn_seqs_per_dp = max(1, math.ceil(sequences / ffn_dp))
    attn_ok, attn_reason = _pool_memory_check(
        attn_parsed, tokens=tokens, context=context,
        seqs_per_dp=attn_seqs_per_dp, is_attn_pool=True)
    ffn_ok, ffn_reason = _pool_memory_check(
        ffn_parsed, tokens=tokens, context=context,
        seqs_per_dp=ffn_seqs_per_dp, is_attn_pool=False)
    memory_feasible = bool(attn_ok and ffn_ok)
    link_info = effective_pool_link(
        attn_parsed.machine, ffn_parsed.machine,
        pool_link_tier=pool_link_tier, link_override=link_override,
    )
    num_layers = _check_int("afd_num_layers", attn_parsed.model.layers, minimum=1)
    hidden_size = _check_int("afd_hidden_size", attn_parsed.model.hidden_size, minimum=1)
    batch_tokens = tokens * sequences
    disp_dtype = af_dispatch_dtype or attn_parsed.model.activation_dtype
    comb_dtype = af_combine_dtype or ffn_parsed.model.activation_dtype
    disp_b, comb_b = dtype_bytes(disp_dtype), dtype_bytes(comb_dtype)
    dispatch_bytes = float(batch_tokens * hidden_size * disp_b)
    combine_bytes = float(batch_tokens * hidden_size * comb_b)
    effective_rate = float(link_info["effective_bytes_per_ms"])
    latency_ms = float(link_info["latency_ms"])
    eff_m = min(m_steps, max(1, int(sequences)))
    a_subs: list[float] | None = None
    f_subs: list[float] | None = None
    x_subs: list[float] | None = None
    if eff_m > 1:
        cache = stage_cache if stage_cache is not None else {}
        def _mb_stage(pool_parsed: CaseSpec, dp: int, mb_s: int, is_attn: bool) -> Mapping[str, Any]:
            s = max(1, math.ceil(mb_s / dp))
            key = (id(pool_parsed), s, is_attn)
            if key not in cache:
                rep = analyze_network_efficiency(
                    pool_parsed, phase=phase, tokens=tokens, context=context,
                    sequences=s, scope=scope, run_graph=False)
                cache[key] = _split_pool_stage(rep)
            return cache[key]
        a_subs, f_subs, x_subs = [], [], []
        for i in range(eff_m):
            mb_s = sequences // eff_m + int(i < sequences % eff_m)
            a_st = _mb_stage(attn_parsed, attn_dp, mb_s, True)
            f_st = _mb_stage(ffn_parsed, ffn_dp, mb_s, False)
            a_subs.append(float(a_st["attention_ms"]) + float(a_st["shared_ms"]))
            f_subs.append(float(f_st["ffn_ms"]))
            mb_wire = float(tokens * mb_s * hidden_size) * (disp_b + comb_b) / effective_rate
            x_subs.append((2.0 * latency_ms + mb_wire) * num_layers)
        attn_stage_ms, ffn_stage_ms = math.fsum(a_subs), math.fsum(f_subs)
    dispatch_per_layer_ms = eff_m * latency_ms + dispatch_bytes / effective_rate
    combine_per_layer_ms = eff_m * latency_ms + combine_bytes / effective_rate
    total_dispatch_ms = dispatch_per_layer_ms * num_layers
    total_combine_ms = combine_per_layer_ms * num_layers
    af_comm_ms = total_dispatch_ms + total_combine_ms

    pipeline = _pipeline_model(
        attn_stage_ms, ffn_stage_ms, af_comm_ms, m_steps,
        sequences=sequences, a_subs=a_subs, f_subs=f_subs, x_subs=x_subs,
    )
    step_duration_ms, request_latency_ms = float(pipeline["step_duration_ms"]), float(pipeline["request_latency_ms"])
    compute_balance_gap = abs(attn_stage_ms - ffn_stage_ms) / max(attn_stage_ms, ffn_stage_ms, 1e-12)
    meets_sla = bool(request_latency_ms <= sla_tpot_ms)
    feasible = bool(memory_feasible and meets_sla)
    infeasible_reason = attn_reason or ffn_reason or (None if meets_sla else "tpot_sla_exceeded")
    constraints = {
        "comm_hidden_ok": bool(af_comm_ms <= max(attn_stage_ms, ffn_stage_ms)),
        "compute_balance_gap": compute_balance_gap,
        "compute_balance_ok": bool(compute_balance_gap <= balance_tolerance),
        "meets_tpot_sla": meets_sla, "memory_feasible": memory_feasible, "feasible": feasible,
    }
    attn_dies, ffn_dies = len(attn_parsed.machine.dies), len(ffn_parsed.machine.dies)
    total_dies, tokens_per_iteration = attn_dies + ffn_dies, tokens * sequences
    tokens_per_sec = tokens_per_iteration * 1000.0 / max(step_duration_ms, 1e-12)
    tokens_per_sec_per_total_die = tokens_per_sec / max(1, total_dies)
    return {
        "schema": _AFD_SCHEMA, "mode": "attention_ffn_disaggregation",
        "geometry": {
            "phase": phase, "scope": scope, "tokens": tokens, "context": context,
            "sequences": sequences, "batch_tokens": batch_tokens, "m_steps": m_steps,
            "pool_link_tier": pool_link_tier, "sla_tpot_ms": sla_tpot_ms,
            "balance_tolerance": balance_tolerance,
        },
        "attn_pool": {
            "machine": attn_parsed.machine.name, "dies": attn_dies,
            "parallel": attn_parsed.runtime.parallel.to_dict(),
            "sequences_per_dp": attn_seqs_per_dp, "memory_feasible": attn_ok,
            "stage_ms": attn_stage_ms, "operator_sum_ms": float(attn_stage["operator_sum_ms"]),
            "graph_duration_ms": float(attn_stage["graph_duration_ms"]),
            "stage_scale": float(attn_stage["scale"]),
            "attention_operator_count": int(attn_stage["attention_operator_count"]),
            "shared_operator_count": int(attn_stage["shared_operator_count"]),
            "shared_ms": float(attn_stage["shared_ms"]),
            "network_cube_mfu": float(attn_rep.get("network_cube_mfu", 0.0) or 0.0),
            "network_mbu": float(attn_rep.get("network_mbu", 0.0) or 0.0),
        },
        "ffn_pool": {
            "machine": ffn_parsed.machine.name, "dies": ffn_dies,
            "parallel": ffn_parsed.runtime.parallel.to_dict(),
            "sequences_per_dp": ffn_seqs_per_dp, "memory_feasible": ffn_ok,
            "stage_ms": ffn_stage_ms, "operator_sum_ms": float(ffn_stage["operator_sum_ms"]),
            "graph_duration_ms": float(ffn_stage["graph_duration_ms"]),
            "stage_scale": float(ffn_stage["scale"]),
            "ffn_operator_count": int(ffn_stage["ffn_operator_count"]),
            "shared_operator_count": int(ffn_stage["shared_operator_count"]),
            "shared_ms": float(ffn_stage["shared_ms"]),
            "network_cube_mfu": float(ffn_rep.get("network_cube_mfu", 0.0) or 0.0),
            "network_mbu": float(ffn_rep.get("network_mbu", 0.0) or 0.0),
        },
        "attn_stage_ms": attn_stage_ms, "ffn_stage_ms": ffn_stage_ms,
        "num_layers": num_layers, "batch_tokens": batch_tokens,
        "hidden_size": hidden_size, "dispatch_dtype": disp_dtype,
        "combine_dtype": comb_dtype, "attn_dies": attn_dies, "ffn_dies": ffn_dies,
        "total_dies": total_dies, "af_dispatch_bytes": dispatch_bytes,
        "af_combine_bytes": combine_bytes, "af_dispatch_ms": dispatch_per_layer_ms,
        "af_combine_ms": combine_per_layer_ms, "af_dispatch_total_ms": total_dispatch_ms,
        "af_combine_total_ms": total_combine_ms, "sla_tpot_ms": sla_tpot_ms,
        "balance_tolerance": balance_tolerance, "pool_link_tier": pool_link_tier,
        "pool_link": link_info, "link_info": link_info, "link": link_info,
        "af_comm": {
            "num_layers": num_layers, "hidden_size": hidden_size, "batch_tokens": batch_tokens,
            "dispatch_dtype": disp_dtype, "combine_dtype": comb_dtype,
            "dispatch_bytes": dispatch_bytes, "combine_bytes": combine_bytes,
            "af_dispatch_bytes": dispatch_bytes, "af_combine_bytes": combine_bytes,
            "dispatch_bytes_per_layer": dispatch_bytes, "combine_bytes_per_layer": combine_bytes,
            "dispatch_ms": dispatch_per_layer_ms, "combine_ms": combine_per_layer_ms,
            "dispatch_ms_per_layer": dispatch_per_layer_ms, "combine_ms_per_layer": combine_per_layer_ms,
            "dispatch_ms_total": total_dispatch_ms, "combine_ms_total": total_combine_ms,
            "af_dispatch_ms": dispatch_per_layer_ms, "af_combine_ms": combine_per_layer_ms,
            "af_dispatch_total_ms": total_dispatch_ms, "af_combine_total_ms": total_combine_ms,
            "af_comm_ms": af_comm_ms,
        },
        "af_comm_ms": af_comm_ms, "pipeline": pipeline, "step_duration_ms": step_duration_ms,
        "request_latency_ms": request_latency_ms, "constraints": constraints,
        "comm_hidden_ok": constraints["comm_hidden_ok"], "compute_balance_gap": compute_balance_gap,
        "compute_balance_ok": constraints["compute_balance_ok"], "meets_tpot_sla": meets_sla,
        "memory_feasible": memory_feasible, "feasible": feasible, "infeasible_reason": infeasible_reason,
        "throughput": {
            "attn_dies": attn_dies, "ffn_dies": ffn_dies, "total_dies": total_dies,
            "tokens_per_iteration": tokens_per_iteration, "tokens_per_sec": tokens_per_sec,
            "tokens_per_sec_per_total_die": tokens_per_sec_per_total_die,
        },
        "tokens_per_sec": tokens_per_sec, "tokens_per_sec_per_total_die": tokens_per_sec_per_total_die,
    }


def predict_afd(
    attn_case: CaseSpec | Mapping[str, Any],
    ffn_case: CaseSpec | Mapping[str, Any], *,
    m_steps: int = 3, pool_link_tier: str = _DEFAULT_POOL_LINK_TIER,
    link_override: Mapping[str, Any] | None = None,
    phase: str = "decode", tokens: int = 1, context: int = 4096,
    sequences: int = 32, af_dispatch_dtype: str | None = None,
    af_combine_dtype: str | None = None, sla_tpot_ms: float = 50.0,
    balance_tolerance: float = 0.35, scope: str = "body",
) -> dict[str, Any]:
    """Predict an AFD iteration from explicit Attention-pool and FFN-pool cases."""
    m_steps, tokens = _check_int("afd_m_steps", m_steps, minimum=1), _check_int("afd_tokens", tokens, minimum=1)
    context, sequences = _check_int("afd_context", context, minimum=0), _check_int("afd_sequences", sequences, minimum=1)
    sla_tpot_ms = _check_float("afd_sla_tpot_ms", sla_tpot_ms, minimum=0.0, strict=True)
    balance_tolerance = _check_float("afd_balance_tolerance", balance_tolerance, minimum=0.0, strict=False)
    pool_link_tier = _check_tier(pool_link_tier)
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"afd_phase_unsupported:{phase}")
    if scope not in {"body", "model"}:
        raise ValueError(f"afd_scope_unsupported:{scope}")

    attn_parsed, ffn_parsed = parse_case(attn_case), parse_case(ffn_case)
    _validate_shared_model_contract(attn_parsed, ffn_parsed)
    attn_seqs = max(1, math.ceil(sequences / max(1, int(attn_parsed.runtime.parallel.dp))))
    ffn_seqs = max(1, math.ceil(sequences / max(1, int(ffn_parsed.runtime.parallel.dp))))
    attn_rep = analyze_network_efficiency(attn_parsed, phase=phase, tokens=tokens, context=context, sequences=attn_seqs, scope=scope, run_graph=False)
    ffn_rep = analyze_network_efficiency(ffn_parsed, phase=phase, tokens=tokens, context=context, sequences=ffn_seqs, scope=scope, run_graph=False)
    return _build_report(
        attn_parsed=attn_parsed, ffn_parsed=ffn_parsed,
        attn_stage=_split_pool_stage(attn_rep), ffn_stage=_split_pool_stage(ffn_rep),
        attn_rep=attn_rep, ffn_rep=ffn_rep,
        m_steps=m_steps, pool_link_tier=pool_link_tier, link_override=link_override,
        phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope,
        af_dispatch_dtype=af_dispatch_dtype, af_combine_dtype=af_combine_dtype,
        sla_tpot_ms=sla_tpot_ms, balance_tolerance=balance_tolerance,
    )


def _dedupe_positive_ints(name: str, values: Sequence[int]) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name}_must_be_sequence_of_positive_ints:{values}")
    result: list[int] = []
    for value in values:
        checked = _check_int(name, value, minimum=1)
        if checked not in result:
            result.append(checked)
    if not result:
        raise ValueError(f"{name}_must_not_be_empty")
    return result


def _scaled_pool_layout(
    parsed: CaseSpec, target_dies: int, *, allow_ep: bool,
) -> dict[str, int]:
    """Pick the structural DP/TP/EP layout closest to the base proportional scale."""
    from mindie_sim.sweep import enumerate_parallel_candidates

    candidates = enumerate_parallel_candidates(
        parsed, num_devices=target_dies, allow_dp=True, allow_ep=allow_ep,
        allow_pp=(parsed.runtime.parallel.pp > 1),
        allow_pcp=(parsed.runtime.parallel.pcp > 1))
    if not candidates:
        raise ModelingError(f"afd_no_parallel_layout:{target_dies}")
    base = parsed.runtime.parallel
    base_world = max(1, int(base.dp) * int(base.tp))
    scale = float(target_dies) / float(base_world)
    # Preserve the base DP:TP ratio when the pool grows: a die-count change is
    # a new world size, not an independent multiplication of both axes.
    base_ratio = float(base.dp) / float(max(1, base.tp))
    target_tp = math.sqrt(float(target_dies) / base_ratio) if base_ratio > 0.0 else float(target_dies)
    target_dp = float(target_dies) / target_tp
    target_ep = float(base.ep) * scale if allow_ep else 1.0
    def _distance(candidate: Mapping[str, int]) -> tuple[float, tuple[int, int, int, int]]:
        dist = (abs(candidate["dp"] - target_dp) / max(target_dp, 1.0)
                + abs(candidate["tp"] - target_tp) / max(target_tp, 1.0))
        if allow_ep:
            dist += abs(candidate["ep"] - target_ep) / max(target_ep, 1.0)
        return dist, (candidate["tp"], candidate["dp"], candidate["ep"], candidate["pp"])
    best = min(candidates, key=_distance)
    return {key: int(best[key]) for key in ("dp", "tp", "ep", "pp", "pcp")}


def _scaled_pool_case(
    parsed: CaseSpec, raw_case: CaseSpec | Mapping[str, Any], target_dies: int,
    *, allow_ep: bool,
) -> CaseSpec:
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.sweep import _materialize_machine_for_devices

    machine = _materialize_machine_for_devices(parsed, raw_case, target_dies)
    layout = _scaled_pool_layout(parsed, target_dies, allow_ep=allow_ep)
    parallel = ParallelSpec.from_mapping(layout)
    if parallel.world_size != target_dies:
        raise ModelingError(f"afd_parallel_world_mismatch:{parallel.world_size}!={target_dies}")
    partition = (parsed.runtime.pp_layer_partition
                 if len(parsed.runtime.pp_layer_partition) == parallel.pp else ())
    return parse_case(replace(
        parsed, machine=machine,
        runtime=replace(parsed.runtime, parallel=parallel,
                        pp_layer_partition=partition)))


def _pareto_optimal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if not row.get("feasible"):
            row["pareto_optimal"] = False
            continue
        dominated = False
        r_tps, r_step = float(row["tokens_per_sec_per_total_die"]), float(row["step_duration_ms"])
        for other in rows:
            if other is row or not other.get("feasible"):
                continue
            o_tps, o_step = float(other["tokens_per_sec_per_total_die"]), float(other["step_duration_ms"])
            if (o_tps >= r_tps - _PARETO_EPS and o_step <= r_step + _PARETO_EPS
                    and (o_tps > r_tps + _PARETO_EPS or o_step < r_step - _PARETO_EPS)):
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    return [row for row in rows if row.get("pareto_optimal")]


def solve_afd_pareto(
    attn_base_case: CaseSpec | Mapping[str, Any],
    ffn_base_case: CaseSpec | Mapping[str, Any], *,
    attn_die_counts: Sequence[int] = (8, 16),
    ffn_die_counts: Sequence[int] = (16, 32, 64),
    m_steps_options: Sequence[int] = (1, 3, 4),
    pool_link_tier: str = _DEFAULT_POOL_LINK_TIER,
    link_override: Mapping[str, Any] | None = None,
    sequences: int = 32, context: int = 4096, sla_tpot_ms: float = 50.0,
) -> dict[str, Any]:
    """Sweep Attention/FFN die counts and M-steps under the TPOT SLA."""
    attn_counts = _dedupe_positive_ints("attn_die_counts", attn_die_counts)
    ffn_counts = _dedupe_positive_ints("ffn_die_counts", ffn_die_counts)
    m_options = _dedupe_positive_ints("m_steps_options", m_steps_options)
    pool_link_tier = _check_tier(pool_link_tier)
    sequences = _check_int("afd_sequences", sequences, minimum=1)
    context = _check_int("afd_context", context, minimum=0)
    sla_tpot_ms = _check_float("afd_sla_tpot_ms", sla_tpot_ms, minimum=0.0, strict=True)
    if link_override is not None and not isinstance(link_override, Mapping):
        raise ValueError(
            f"afd_link_override_must_be_mapping:{type(link_override).__name__}")

    attn_base = parse_case(attn_base_case)
    ffn_base = parse_case(ffn_base_case)
    _validate_shared_model_contract(attn_base, ffn_base)
    rows: list[dict[str, Any]] = []
    attn_cache: dict[int, tuple[CaseSpec | None, dict[str, Any] | None, dict[str, Any] | None, str | None]] = {}
    ffn_cache: dict[int, tuple[CaseSpec | None, dict[str, Any] | None, dict[str, Any] | None, str | None]] = {}
    mb_stage_cache: dict[tuple[int, int, bool], dict[str, Any]] = {}

    def _invalid_row(attn_d: int, ffn_d: int, m_s: int, reason: str,
                     ap: dict[str, Any] | None = None, fp: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "attn_dies": attn_d, "ffn_dies": ffn_d, "total_dies": attn_d + ffn_d,
            "m_steps": m_s, "attn_parallel": ap, "ffn_parallel": fp,
            "attn_stage_ms": None, "ffn_stage_ms": None, "af_comm_ms": None,
            "step_duration_ms": None, "steady_state_ms": None, "bubble_ms": None,
            "tokens_per_sec": 0.0, "tokens_per_sec_per_total_die": 0.0,
            "comm_hidden_ok": False, "compute_balance_ok": False,
            "compute_balance_gap": None, "meets_tpot_sla": False,
            "feasible": False, "status": "invalid",
            "infeasible_reason": reason, "pareto_optimal": False,
        }

    def _pool(
        parsed: CaseSpec, raw_case: CaseSpec | Mapping[str, Any], target: int,
        allow_ep: bool, cache: dict[int, tuple[CaseSpec | None, dict[str, Any] | None, dict[str, Any] | None, str | None]],
    ) -> tuple[CaseSpec | None, dict[str, Any] | None, dict[str, Any] | None, str | None]:
        if target in cache:
            return cache[target]
        try:
            candidate = _scaled_pool_case(parsed, raw_case, target, allow_ep=allow_ep)
            seqs_per_dp = max(1, math.ceil(sequences / max(1, int(candidate.runtime.parallel.dp))))
            report = analyze_network_efficiency(
                candidate, phase="decode", tokens=1, context=context,
                sequences=seqs_per_dp, scope="body", run_graph=False)
            cache[target] = (candidate, report, _split_pool_stage(report), None)
        except (ModelingError, ValueError) as exc:
            cache[target] = (None, None, None, str(exc))
        return cache[target]

    for attn_dies in attn_counts:
        attn_case, _attn_rep, attn_stage, attn_error = _pool(
            attn_base, attn_base_case, attn_dies, False, attn_cache)
        for ffn_dies in ffn_counts:
            ffn_case, _ffn_rep, ffn_stage, ffn_error = _pool(
                ffn_base, ffn_base_case, ffn_dies, True, ffn_cache)
            if attn_case is None or ffn_case is None or attn_stage is None or ffn_stage is None:
                reason = attn_error or ffn_error or "afd_candidate_unavailable"
                for m_steps in m_options:
                    rows.append(_invalid_row(attn_dies, ffn_dies, m_steps, reason))
                continue
            for m_steps in m_options:
                try:
                    report = _build_report(
                        attn_parsed=attn_case, ffn_parsed=ffn_case,
                        attn_stage=attn_stage, ffn_stage=ffn_stage,
                        attn_rep=_attn_rep or {}, ffn_rep=_ffn_rep or {},
                        m_steps=m_steps, pool_link_tier=pool_link_tier,
                        link_override=link_override, phase="decode", tokens=1,
                        context=context, sequences=sequences, scope="body",
                        af_dispatch_dtype=None, af_combine_dtype=None,
                        sla_tpot_ms=sla_tpot_ms, balance_tolerance=0.35,
                        stage_cache=mb_stage_cache,
                    )
                except (ModelingError, ValueError) as exc:
                    rows.append(_invalid_row(
                        attn_dies, ffn_dies, m_steps, str(exc),
                        attn_case.runtime.parallel.to_dict(), ffn_case.runtime.parallel.to_dict()))
                    continue
                meets_sla, mem_ok, feasible = bool(report["meets_tpot_sla"]), bool(report["memory_feasible"]), bool(report["feasible"])
                rows.append({
                    "attn_dies": attn_dies, "ffn_dies": ffn_dies, "total_dies": attn_dies + ffn_dies,
                    "m_steps": m_steps, "attn_parallel": attn_case.runtime.parallel.to_dict(),
                    "ffn_parallel": ffn_case.runtime.parallel.to_dict(),
                    "attn_stage_ms": report["attn_stage_ms"], "ffn_stage_ms": report["ffn_stage_ms"],
                    "af_comm_ms": report["af_comm_ms"], "steady_state_ms": report["pipeline"]["steady_state_ms"],
                    "bubble_ms": report["pipeline"]["fill_drain_bubble_ms"], "step_duration_ms": report["step_duration_ms"],
                    "request_latency_ms": report["request_latency_ms"], "tokens_per_sec": report["tokens_per_sec"],
                    "tokens_per_sec_per_total_die": report["tokens_per_sec_per_total_die"],
                    "comm_hidden_ok": report["comm_hidden_ok"], "compute_balance_ok": report["compute_balance_ok"],
                    "compute_balance_gap": report["compute_balance_gap"], "meets_tpot_sla": meets_sla,
                    "memory_feasible": mem_ok, "feasible": feasible,
                    "status": "feasible" if feasible else ("oom" if not mem_ok else "sla_exceeded"),
                    "infeasible_reason": report["infeasible_reason"], "pareto_optimal": False,
                })

    pareto_frontier = _pareto_optimal_rows(rows)
    sla_rows = [row for row in rows if row.get("feasible")]
    recommended = max(
        sla_rows,
        key=lambda row: (float(row["tokens_per_sec_per_total_die"]),
                         -float(row["step_duration_ms"]),
                         -int(row["total_dies"]), int(row["m_steps"])),
        default=None)
    return {
        "schema": _AFD_PARETO_SCHEMA, "attn_die_counts": attn_counts,
        "ffn_die_counts": ffn_counts, "m_steps_options": m_options,
        "pool_link_tier": pool_link_tier,
        "link_override": dict(link_override) if link_override is not None else None,
        "sequences": sequences, "context": context, "sla_tpot_ms": sla_tpot_ms,
        "search_domain": {
            "attn_die_counts": attn_counts, "ffn_die_counts": ffn_counts,
            "m_steps_options": m_options, "candidate_count": len(rows),
        },
        "candidates": rows, "grid": rows,
        "pareto_frontier": pareto_frontier, "recommended": recommended,
    }


__all__ = ["effective_pool_link", "predict_afd", "solve_afd_pareto"]
