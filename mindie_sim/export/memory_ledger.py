"""Closed-form per-die Weight/Quant/Cache/Engram-PLE/PP HBM memory calculator.

[Role] Compute deterministic, Excel-verifiable per-die memory ledgers honoring
  ``QuantSpec.exclusions``, post-TP/EP/PP/DCP/PCP/Local-TP tensor sharding,
  MXFP4/MXFP8 microscaling and W8A8/W4A8 per-channel aux overhead, KV/Indexer/
  KeyPool/KDA recurrent cache sizing, Engram/PLE Host DRAM offload, and closed-form
  per-DP Max HBM Concurrency.
[Invariants] Strictly pure analytical computation over ``Case`` and ``Memory``;
  zero device contexts, zero weight loading, and exact parity with Sheet 3's
  live OOXML ``VLOOKUP``/``ROUNDUP`` formulas.
[Timeouts & Locks] Single-pass O(Unique Weight Terms + Layers) arithmetic; no
  locks or I/O.
[Side-Effect Policy] Read-only over caller ``Case`` instances.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
from typing import Any

from mindie_sim.catalog.cache import (
    DEFAULT_CACHE_BLOCK_SIZE,
    INDEX_INTERLEAVE_TOKENS,
    layer_cache_components,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, _sharded_shape
from mindie_sim.catalog.weights import engram_table_terms, stage_weight_terms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError

from .models import (
    BF16_BYTES,
    FP32_BYTES,
    KV_QUANT_GROUP_SIZE,
    MAX_EXPLICIT_LAYER_LABELS,
    MIN_DENOM_EPS,
    SUPPORTED_CACHE_DTYPES,
    WEIGHT_QUANT_FORMATS,
    CacheLedgerRow,
    MemoryCalculatorLedger,
    StageBudgetRow,
    WeightLedgerRow,
)

_ENGRAM_CATEGORIES = frozenset({"engram_table", "ple_table", "gemma4_ple"})
_WEIGHT_QUANT_ALIASES: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "fp32": "FP32",
    "float32": "FP32",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "w8a8": "W8A8",
    "int8": "INT8",
    "w4a8": "W4A8",
    "int4": "INT4",
    "mxfp8": "MXFP8",
    "w8a8_mxfp8": "MXFP8",
    "w8a8-mxfp8": "MXFP8",
    "mxfp4": "MXFP4",
    "mxfp4_routed_only": "MXFP4",
    "w4a8_mxfp": "MXFP4",
    "w4a8-mxfp": "MXFP4",
    "w4a8_mxfp4": "MXFP4",
    "w4a8-mxfp4": "MXFP4",
}
_CACHE_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "c8": "C8",
    "int8": "C8",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
}
_QUANT_FORMAT_DISPLAY: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "fp32": "FP32",
    "float32": "FP32",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "int8": "W8A8",
    "w8a8": "W8A8",
    "int4": "W4A8",
    "w4a8": "W4A8",
    "mxfp8": "MXFP8",
    "mx_fp8": "MXFP8",
    "mxfp4": "MXFP4",
    "mx_fp4": "MXFP4",
    "fp4": "MXFP4",
    "fp4_e2m1": "MXFP4",
}


def evaluate_weight_storage_bytes(
    local_rows: int, local_cols: int, quant_format: str,
) -> tuple[int, int, int, int]:
    """Closed-form ``(payload, scale, aux, total)`` bytes for one post-TP local matrix."""
    key = str(quant_format or "BF16").strip().upper()
    spec = WEIGHT_QUANT_FORMATS.get(key)
    if spec is None:
        raise ModelingError(f"unsupported_weight_quant_format:{quant_format}")
    rows = int(local_rows)
    cols = int(local_cols)
    if rows < 1 or cols < 1:
        raise ModelingError(f"invalid_local_weight_shape:({rows},{cols})")
    payload = rows * math.ceil(cols * float(spec["bytes"]))
    group = int(spec["group"])
    scale = (
        rows * ((cols + group - 1) // group) * int(spec["scale"])
        if group > 0 else 0
    )
    aux = rows * int(spec["aux"]) if int(spec["aux"]) > 0 else 0
    return payload, scale, aux, payload + scale + aux


def _normalize_quant_preset(quant: QuantSpec | None, fallback: str = "BF16") -> str:
    if quant is None:
        return fallback
    name = str(getattr(quant, "name", "") or "").strip().lower()
    exp_fmt = str(getattr(quant, "expert_weight_format", "") or "").strip().lower()
    w_fmt = str(getattr(quant, "weight_format", "") or "").strip().lower()
    combined = f"{name} {exp_fmt} {w_fmt}"
    if "mxfp4" in combined or "mx_fp4" in combined:
        return "MXFP4"
    if "mxfp8" in combined or "mx_fp8" in combined:
        return "MXFP8"
    if "w4a8" in combined or "int4" in combined or "fp4" in combined:
        return "W4A8"
    if "w8a8" in combined or "int8" in combined:
        return "W8A8"
    if "fp8" in combined:
        return "FP8"
    return "BF16"


def _describe_quant_display(
    quant: QuantSpec | None, resolved_wq: str, *, explicit_override: bool,
) -> tuple[str, tuple[str, ...]]:
    exclusions = tuple(getattr(quant, "exclusions", ()) or ()) if quant is not None else ()
    if quant is None or not exclusions:
        return resolved_wq, ()
    excl_set = set(exclusions)
    routed_only = (
        resolved_wq == "MXFP4"
        and ({"attn", "mla", "kda"} & excl_set)
        and "shared_expert" in excl_set
    )
    if routed_only:
        return "MXFP4 (Routed-Only, Attn/Shared BF16)", exclusions
    w_fmt = str(getattr(quant, "weight_format", "bf16") or "bf16").lower()
    exp_fmt = str(getattr(quant, "expert_weight_format", "bf16") or "bf16").lower()
    if not explicit_override and w_fmt != exp_fmt and "fp8" in w_fmt and "mxfp4" in exp_fmt:
        return "FP8 Linear + MXFP4 Routed Experts (Router/Norm/Head BF16)", exclusions
    return f"{resolved_wq} (Excl: {','.join(exclusions[:MAX_EXPLICIT_LAYER_LABELS])})", exclusions


def _canonicalize_weight_quant_override(raw: str) -> str:
    token = str(raw).strip().lower()
    return _WEIGHT_QUANT_ALIASES.get(token, str(raw).strip().upper())


def _canonicalize_cache_dtype_override(raw: str) -> str:
    token = str(raw).strip().lower()
    return _CACHE_DTYPE_ALIASES.get(token, str(raw).strip().upper())


def _normalize_cache_dtype(model: Any) -> str:
    quant = getattr(model, "quant", None)
    raw = str(
        (getattr(quant, "kv_dtype", "") if quant is not None else "")
        or getattr(model, "kv_dtype", "")
        or "bf16"
    ).strip().lower()
    if "int8" in raw or raw == "c8":
        return "C8"
    if "fp8" in raw:
        return "FP8"
    return "BF16"


def _resolve_shard_axis_and_rule(term: WeightTerm) -> tuple[str, int]:
    shard = str(term.shard or "replicated")
    role = str(term.placement_role or "")
    axis = 1 if (role == "attention_output" or shard in {"in_tp", "dense_down", "expert_down", "group_in"}) else 0
    if shard in {"expert_gate_up", "expert_down"}:
        return "EP", axis
    if shard == "replicated":
        return "Replicated", axis
    if shard == "node":
        return "Node", axis
    if role == "attention_output":
        return "Local-TP (o_proj)", axis
    if role == "attention_query":
        return "Local-TP (q_proj)", axis
    if term.category == "shared_expert":
        return "Shared-TP", axis
    return f"TP ({shard})", axis


def stage_layer_counts(case: Case) -> tuple[int, ...]:
    """Return decoder layer counts owned by each pipeline stage."""
    p = case.runtime.parallel
    if case.runtime.pp_layer_partition:
        return tuple(int(x) for x in case.runtime.pp_layer_partition)
    return tuple(
        case.model.layers // p.pp + (1 if stage < case.model.layers % p.pp else 0)
        for stage in range(p.pp)
    )


def _cache_dtype_override_ratio(model: Any, target_dtype: str) -> float:
    base = _normalize_cache_dtype(model)
    target = str(target_dtype or base).strip().upper()
    if target == base:
        return 1.0
    width_by_dtype = {
        "BF16": float(BF16_BYTES),
        "C8": 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE),
        "FP8": 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE),
    }
    base_w = width_by_dtype.get(base, float(BF16_BYTES))
    target_w = width_by_dtype.get(target, float(BF16_BYTES))
    return target_w / max(base_w, MIN_DENOM_EPS)


def _compute_cache_ledger_rows(
    case: Case,
    *,
    stage_layers: Sequence[int],
    cache_dtype: str,
    seq_len: int,
    concurrency: int,
    block_size: int,
    spec_tokens: int,
) -> tuple[int, tuple[CacheLedgerRow, ...], list[float], list[float]]:
    model = case.model
    p = case.runtime.parallel
    dcp = max(1, int(p.cp))
    local_tokens = math.ceil(max(1, seq_len) / (max(1, block_size) * dcp)) * max(1, block_size)
    layer_stage = [s for s, count in enumerate(stage_layers) for _ in range(count)]
    comp_stage_bytes: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(stage_layers))
    comp_stage_layers: dict[str, list[int]] = defaultdict(lambda: [0] * len(stage_layers))
    comp_scalable: dict[str, bool] = {}

    scale_override = _cache_dtype_override_ratio(model, cache_dtype)
    for lp in model.layer_plans:
        stage = layer_stage[lp.index] if lp.index < len(layer_stage) else 0
        raw_comps = layer_cache_components(
            model, lp, p, max(1, seq_len),
            capacity_padded=True,
            interleave=INDEX_INTERLEAVE_TOKENS,
            runtime=case.runtime,
            query_tokens=max(1, 1 + spec_tokens),
        )
        for comp_name, val in raw_comps.items():
            if val <= 0.0:
                continue
            scalable = comp_name not in {"recurrent_state", "conv_state", "compressor_state"}
            comp_scalable[comp_name] = scalable
            adjusted = float(val) * (scale_override if scalable else 1.0)
            comp_stage_bytes[comp_name][stage] += adjusted
            comp_stage_layers[comp_name][stage] += 1

    stage_per_req_totals = [0.0] * len(stage_layers)
    stage_totals = [0.0] * len(stage_layers)
    rows: list[CacheLedgerRow] = []
    for comp_name in sorted(comp_stage_bytes):
        per_stage = comp_stage_bytes[comp_name]
        layers_per_stage = tuple(comp_stage_layers[comp_name])
        max_layers = max(layers_per_stage, default=0)
        max_req_bytes = max(per_stage, default=0.0)
        scalable = comp_scalable.get(comp_name, True)
        b_per_tok = (
            max_req_bytes / float(max(1, max_layers * local_tokens))
            if (scalable and max_layers > 0) else 0.0
        )
        for st_idx, st_val in enumerate(per_stage):
            stage_per_req_totals[st_idx] += st_val
            stage_totals[st_idx] += st_val * float(concurrency)
        rows.append(CacheLedgerRow(
            component=comp_name,
            cache_kind=cache_dtype if scalable else "State (Fixed)",
            active_layers_per_stage=layers_per_stage,
            max_stage_layers=max_layers,
            bytes_per_token_per_layer=b_per_tok,
            per_request_bytes=max_req_bytes,
            batch_total_bytes=max_req_bytes * float(concurrency),
            scales_with_tokens=scalable,
        ))
    return local_tokens, tuple(rows), stage_totals, stage_per_req_totals


def _resolve_term_quant_format(
    term: WeightTerm,
    model_quant: QuantSpec | None,
    resolved_wq: str,
    *,
    explicit_override: bool,
) -> tuple[str, bool, str]:
    cat = str(term.category or "")
    fam = cat.split(".", 1)[0]
    unquantized_cats = {"norm", "embedding", "lm_head", "param_fp32", "engram_table", "conv_weight_f32", "a_log", "dt_bias"}
    if fam in {"param_fp32", "conv_weight_f32", "a_log", "dt_bias"}:
        return "FP32", True, "FP32 param"
    if fam in unquantized_cats or cat in unquantized_cats:
        return "BF16", True, "quantization_config.ignore -> BF16"

    term_quant = term.runtime_quant or term.quant or model_quant
    if term_quant is not None:
        exclusions = set(getattr(term_quant, "exclusions", ()) or ())
        if cat in exclusions or fam in exclusions:
            native = _QUANT_FORMAT_DISPLAY.get(str(term_quant.native_format or "bf16").lower(), "BF16")
            return native, True, "quantization_config.ignore -> BF16"
        if not explicit_override:
            eff_raw = str(term_quant.format_for(cat) or "bf16").lower()
            mapped = _QUANT_FORMAT_DISPLAY.get(eff_raw, resolved_wq)
            note = "MXFP4 (group=32)" if mapped == "MXFP4" else mapped
            return mapped, (mapped == "BF16"), note
    return resolved_wq, (resolved_wq == "BF16"), ("MXFP4 (group=32)" if resolved_wq == "MXFP4" else resolved_wq)


def _weight_group_label(category: str) -> str:
    lower = category.lower()
    if lower in _ENGRAM_CATEGORIES or "engram" in lower or "ple" in lower:
        return "Engram / PLE"
    if "expert" in lower or "router" in lower or "moe" in lower:
        return "MoE"
    if "mla" in lower or "indexer" in lower or "attn" in lower or "compressor" in lower:
        return "Attention / MLA"
    if "kda" in lower or "gdn" in lower or "conv" in lower or "linear" in lower:
        return "LinearAttn"
    if "mlp" in lower or "ffn" in lower:
        return "Dense FFN"
    if "vision" in lower or "projector" in lower or "image" in lower:
        return "Vision"
    if lower in {"embedding", "lm_head"}:
        return "Head / Embedding"
    return "Norm / Residual"


def _collect_weight_ledger_rows(
    case: Case,
    mem: Memory,
    stage_layers: Sequence[int],
    resolved_wq: str,
    *,
    explicit_override: bool,
) -> tuple[tuple[WeightLedgerRow, ...], list[float], list[float], list[float]]:
    p = case.runtime.parallel
    stage_terms = stage_weight_terms(
        case.model, p,
        include_draft=bool(mem.draft_kind),
        include_engram=True,
        include_engram_tables=True,
        layer_partition=stage_layers,
        draft_kind=mem.draft_kind or None,
        slot_plans=mem.slot_plans,
    )
    engram_names = {t.name for t in engram_table_terms(case.model)}
    grouped: dict[tuple[str, str, int, int, str, bool], dict[str, Any]] = {}
    for stage_idx, terms in enumerate(stage_terms):
        for term in terms:
            if not term.runtime_owned:
                continue
            local_r, local_c, local_cnt = _sharded_shape(term, p, case.machine.ranks_per_node)
            shard_rule, axis = _resolve_shard_axis_and_rule(term)
            is_engram = (
                term.category in _ENGRAM_CATEGORIES
                or term.name in engram_names
                or "engram_table" in term.name.lower()
                or "ple_table" in term.name.lower()
            )
            key = (term.category, term.name, int(term.rows), int(term.cols), shard_rule, is_engram)
            entry = grouped.get(key)
            if entry is None:
                quant_fmt, is_excl, q_note = _resolve_term_quant_format(
                    term, case.model.quant, resolved_wq, explicit_override=explicit_override,
                )
                entry = {
                    "category": term.category,
                    "name": term.name,
                    "global_rows": int(term.rows),
                    "global_cols": int(term.cols),
                    "shard_rule": shard_rule,
                    "axis": axis,
                    "local_rows": int(local_r),
                    "local_cols": int(local_c),
                    "stage_repeats": [0] * p.pp,
                    "is_engram": is_engram,
                    "quant": quant_fmt,
                    "is_excluded": is_excl,
                    "quant_note": q_note,
                }
                grouped[key] = entry
            entry["stage_repeats"][stage_idx] += max(1, int(local_cnt))

    weight_rows: list[WeightLedgerRow] = []
    stage_w_payload = [0.0] * p.pp
    stage_w_scale = [0.0] * p.pp
    stage_engram_total = [0.0] * p.pp
    for entry in grouped.values():
        payload_1, scale_1, aux_1, total_1 = evaluate_weight_storage_bytes(
            entry["local_rows"], entry["local_cols"], entry["quant"],
        )
        repeats = tuple(entry["stage_repeats"])
        max_rep = max(repeats, default=0)
        for st_idx, rep in enumerate(repeats):
            if rep <= 0:
                continue
            if entry["is_engram"]:
                stage_engram_total[st_idx] += float(total_1 * rep)
            else:
                stage_w_payload[st_idx] += float(payload_1 * rep)
                stage_w_scale[st_idx] += float((scale_1 + aux_1) * rep)
        weight_rows.append(WeightLedgerRow(
            group=_weight_group_label(entry["category"]),
            name=entry["name"],
            category=entry["category"],
            global_rows=entry["global_rows"],
            global_cols=entry["global_cols"],
            shard_rule=entry["shard_rule"],
            shard_axis=entry["axis"],
            local_rows=entry["local_rows"],
            local_cols=entry["local_cols"],
            stage_repeats=repeats,
            max_stage_repeats=max_rep,
            quant_format=entry["quant"],
            payload_bytes=payload_1 * max_rep,
            scale_bytes=scale_1 * max_rep,
            aux_bytes=aux_1 * max_rep,
            per_die_bytes=total_1 * max_rep,
            is_engram_or_ple=entry["is_engram"],
            is_quant_excluded=entry["is_excluded"],
            quant_note=entry["quant_note"],
        ))
    weight_rows.sort(key=lambda r: (r.group, r.name))
    return tuple(weight_rows), stage_w_payload, stage_w_scale, stage_engram_total


def _compute_max_hbm_concurrency(
    stage_budgets: Sequence[StageBudgetRow],
    stage_per_req_cache: Sequence[float],
    fallback_conc: int,
) -> int:
    caps: list[int] = []
    for st, per_req in zip(stage_budgets, stage_per_req_cache):
        fixed = (
            st.weight_payload_bytes
            + st.weight_scale_aux_bytes
            + st.engram_ple_hbm_bytes
            + st.workspace_peak_bytes
            + st.reserve_bytes
        )
        avail = st.usable_hbm_budget_bytes - fixed
        if avail <= 0.0:
            return 0
        if per_req > 0.0:
            caps.append(int(avail // per_req))
    return min(caps) if caps else max(0, int(fallback_conc))


def compute_memory_quant_ledger(
    case: Case,
    *,
    phase: str = "decode",
    seq_len: int | None = None,
    concurrency: int | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
) -> MemoryCalculatorLedger:
    """Derive closed-form per-die Weight/Quant/Cache/Engram-PLE/PP HBM budget ledger."""
    p = case.runtime.parallel
    stage_layers = stage_layer_counts(case)
    resolved_seq = max(1, int(seq_len if seq_len is not None else case.runtime.max_model_len))
    resolved_conc = max(1, int(concurrency if concurrency is not None else max(1, len(case.workload.requests))))
    explicit_override = weight_quant is not None
    resolved_wq = (
        _canonicalize_weight_quant_override(weight_quant)
        if weight_quant is not None else _normalize_quant_preset(case.model.quant)
    )
    if resolved_wq not in WEIGHT_QUANT_FORMATS:
        raise ModelingError(f"unsupported_weight_quant:{resolved_wq}")
    wq_display, q_exclusions = _describe_quant_display(
        case.model.quant, resolved_wq, explicit_override=explicit_override,
    )
    resolved_cd = (
        _canonicalize_cache_dtype_override(cache_dtype)
        if cache_dtype is not None else _normalize_cache_dtype(case.model)
    )
    if resolved_cd not in SUPPORTED_CACHE_DTYPES:
        raise ModelingError(f"unsupported_cache_dtype:{resolved_cd}")
    resolved_offload = (
        bool(case.runtime.engram_cpu_offload)
        if engram_host_offload is None else bool(engram_host_offload)
    )
    block_size = int(getattr(case.runtime, "block_size", DEFAULT_CACHE_BLOCK_SIZE) or DEFAULT_CACHE_BLOCK_SIZE)
    spec_tokens = int(dict(case.runtime.metadata).get("draft_proposal_tokens", 0) or 0)

    mem = Memory(case)
    alloc = mem.allocation_report(resolved_seq)
    weight_rows, stage_w_payload, stage_w_scale, stage_engram_total = _collect_weight_ledger_rows(
        case, mem, stage_layers, resolved_wq, explicit_override=explicit_override,
    )
    local_tokens, cache_rows, stage_cache_bytes, stage_per_req_cache = _compute_cache_ledger_rows(
        case,
        stage_layers=stage_layers,
        cache_dtype=resolved_cd,
        seq_len=resolved_seq,
        concurrency=resolved_conc,
        block_size=block_size,
        spec_tokens=spec_tokens,
    )
    ws_by_stage = [float(x) for x in alloc["workspace"]["total_bytes_per_stage_rank"]]
    deq_by_stage = [float(x) for x in alloc["temporary_dequant"]["bytes_per_stage_rank"]]
    reserve_bytes = float(alloc["reserve"]["total_bytes_per_rank"])
    die_hbm = float(min(d.hbm_bytes for d in case.machine.dies))
    util = float(case.runtime.gpu_memory_utilization)
    usable_hbm = die_hbm * util

    stage_budgets: list[StageBudgetRow] = []
    for st in range(p.pp):
        engram_hbm = 0.0 if resolved_offload else stage_engram_total[st]
        engram_host = stage_engram_total[st] if resolved_offload else 0.0
        ws_peak = (ws_by_stage[st] if st < len(ws_by_stage) else 0.0) + (
            deq_by_stage[st] if st < len(deq_by_stage) else 0.0
        )
        total_hbm = (
            stage_w_payload[st] + stage_w_scale[st] + engram_hbm
            + stage_cache_bytes[st] + ws_peak + reserve_bytes
        )
        stage_budgets.append(StageBudgetRow(
            stage=st,
            owned_layers=stage_layers[st],
            weight_payload_bytes=stage_w_payload[st],
            weight_scale_aux_bytes=stage_w_scale[st],
            engram_ple_hbm_bytes=engram_hbm,
            engram_ple_host_bytes=engram_host,
            cache_bytes=stage_cache_bytes[st],
            workspace_peak_bytes=ws_peak,
            reserve_bytes=reserve_bytes,
            total_hbm_bytes=total_hbm,
            usable_hbm_budget_bytes=usable_hbm,
            headroom_bytes=usable_hbm - total_hbm,
            fits_hbm=total_hbm <= usable_hbm,
        ))

    local_tp = max(
        int(getattr(p, "effective_o_proj_tp", p.tp) or p.tp),
        int(getattr(p, "q_proj_tp", 1) or 1),
        int(getattr(p, "dcp_linear_tp", 1) or 1),
    )
    shared_dp = bool(dict(case.runtime.metadata).get("enable_shared_expert_dp", False))
    shared_tp = 1 if shared_dp else int(p.tp)
    sim_cache_by_stage = tuple(
        float(sum(comp["total"] for comp in entry["allocated"].values()))
        for entry in alloc["cache"]["components"]
    )
    weight_footprint = max(
        (st.weight_payload_bytes + st.weight_scale_aux_bytes + st.engram_ple_hbm_bytes for st in stage_budgets),
        default=0.0,
    )
    max_conc = _compute_max_hbm_concurrency(stage_budgets, stage_per_req_cache, resolved_conc)
    return MemoryCalculatorLedger(
        model_name=str(case.model.name),
        hardware_sku=str(case.machine.sku or case.machine.name),
        phase=phase,
        weight_quant=resolved_wq,
        cache_dtype=resolved_cd,
        engram_host_offload=resolved_offload,
        dp=int(p.dp), tp=int(p.tp), ep=int(p.ep), pp=int(p.pp),
        dcp=int(p.cp), pcp=int(p.pcp), local_tp=int(local_tp), shared_tp=int(shared_tp),
        total_dies=len(case.machine.dies),
        seq_len=resolved_seq, concurrency=resolved_conc,
        block_size=block_size, spec_tokens=spec_tokens, local_cache_tokens=local_tokens,
        die_hbm_capacity_bytes=die_hbm, gpu_memory_utilization=util, usable_hbm_bytes=usable_hbm,
        weight_rows=weight_rows,
        cache_rows=cache_rows,
        stage_budgets=tuple(stage_budgets),
        max_stage_hbm_bytes=max((row.total_hbm_bytes for row in stage_budgets), default=0.0),
        max_stage_host_offload_bytes=max((row.engram_ple_host_bytes for row in stage_budgets), default=0.0),
        fits_hbm=all(row.fits_hbm for row in stage_budgets),
        sim_memory_weight_bytes_by_stage=tuple(float(x) for x in alloc["weights"]["total_bytes_per_stage_rank"]),
        sim_memory_cache_bytes_by_stage=sim_cache_by_stage,
        weight_quant_display=wq_display,
        quant_exclusions=q_exclusions,
        weight_footprint_per_die_bytes=weight_footprint,
        per_request_cache_bytes_max_stage=max(stage_per_req_cache, default=0.0),
        max_hbm_concurrency_per_dp=max_conc,
    )
