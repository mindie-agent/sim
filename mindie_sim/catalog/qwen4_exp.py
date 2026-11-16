"""Qwen4Exp PLE, low-rank gated HC and QSA shared-pool geometry.

[Role]
  Normalize the Qwen4Exp n-gram PLE tables (1-based ``ple_layer_ids``, prime
  head layouts, conv kernel), the low-rank gated hyper-connection matrices,
  and the weight-free QSA indexer into existing weight/op/cache records.

[Invariants]
  * ``ple_layer_ids`` are one-based; a table's ``layer_id`` is ``ple_id - 1``.
  * Table rows are the exact sum of per-head primes rounded to the declared
    divisor; table width is ``ple_embed_dim / ((ngram-1)*heads)``.
  * PLE projections are TP-replicated; table rows shard over TP/ETP.
  * ``hc_lowrank > 0`` selects GatedResidual terms and suppresses Sinkhorn.
  * The QSA shared pool is 12 slot-backed pages, 3 GDN state groups and one
    raw circular page; PLE conv state is private, per table and TP-replicated.

[Timeouts & Locks]
  Pure integer/float arithmetic bounded by layer and head counts; no I/O.

[Side-Effect Policy]
  No filesystem, network, device or global mutation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from mindie_sim.catalog.config import _int
from mindie_sim.catalog.ops import comm_op, gemm_op, vector_op
from mindie_sim.catalog.types import (
    DraftSpec, EngramSpec, EngramTableSpec, merge_source,
)
from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _t
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec

ZERO = 0
ONE = 1
TWO = 2
ONE_FLOAT = 1.0
ZERO_FLOAT = 0.0
BF16_BYTES = 2
KERNEL_BLOCK_TOKENS = 128
DEFAULT_PLE_CONV_KERNEL = 4
DEFAULT_PLE_DIVISOR = 128
PLE_GATE_OPS_PER_ELEM = 6.0
PLE_LOOKUP_OPS_PER_ELEM = 2.0
DRAFT_HC_HEAD_TERMS = 3
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - ONE) // int(alignment)) * int(alignment)


def _is_prime_64(value: int) -> bool:
    if value < TWO:
        return False
    for prime in _SMALL_PRIMES:
        if value % prime == ZERO:
            return value == prime
    exponent = value - ONE
    shifts = ZERO
    while exponent % TWO == ZERO:
        exponent //= TWO
        shifts += ONE
    for base in _MILLER_RABIN_BASES:
        if base % value == ZERO:
            continue
        witness = pow(base, exponent, value)
        if witness in (ONE, value - ONE):
            continue
        for _ in range(shifts - ONE):
            witness = pow(witness, TWO, value)
            if witness == value - ONE:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + ONE
        if candidate <= TWO:
            prime = TWO
            continue
        if candidate % TWO == ZERO:
            candidate += ONE
        while not _is_prime_64(candidate):
            candidate += TWO
        prime = candidate
    return prime


def build_ple_spec(c: Mapping[str, Any]) -> EngramSpec:
    ple_layers = [int(v) for v in c.get("ple_layer_ids", ()) or ()]
    max_ngram = _int(c, "ngram_size")
    heads = _int(c, "heads_per_ngram")
    embed = _int(c, "ple_embed_dim")
    base_rows = _int(c, "ngram_vocab_size_base")
    ngram_heads = (max_ngram - ONE) * heads
    if max_ngram < TWO or heads <= ZERO or embed <= ZERO or ngram_heads <= ZERO:
        raise ModelingError("model_config_ple_geometry_missing")
    if embed % ngram_heads:
        raise ModelingError("model_config_ple_geometry_missing")
    width = embed // ngram_heads
    divisor = _int(c, "make_ngram_vocab_size_divisible_by", DEFAULT_PLE_DIVISOR)
    conv_k = _int(c, "ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL)
    tables = []
    for index, ple_id in enumerate(ple_layers):
        if ple_id <= ZERO:
            raise ModelingError("model_config_ple_layer_id_invalid")
        total = ZERO
        for local_head in range(ngram_heads):
            global_head = index * ngram_heads + local_head
            total += _nth_prime_after(base_rows - ONE, global_head + ONE)
        tables.append(EngramTableSpec(
            layer_id=ple_id - ONE, table_id=index,
            rows=_round_up(total, divisor), width=width))
    return EngramSpec(
        tables=tuple(tables), max_ngram=max_ngram, heads=heads, width=width,
        pad_token_id=_int(c, "engram_pad_token_id", ZERO),
        host_response_bytes=width * BF16_BYTES,
        source=merge_source({
            "kind": "qwen4_exp_ple", "ple_layers": str(len(tables)),
            "ple_conv_kernel_size": str(conv_k), "ple_embed_dim": str(embed),
        }),
    )


def is_qwen4_exp_ple(engram: Any) -> bool:
    return engram is not None and dict(
        getattr(engram, "source", ()) or ()).get("kind") == "qwen4_exp_ple"


def ple_table_terms(model: ModelSpec) -> list[WeightTerm]:
    engram = model.engram
    if engram is None:
        return []
    return [WeightTerm(f"ple_table_{table.table_id}", table.rows, table.width,
                       "engram_table", "out_tp", ONE) for table in engram.tables]


def ple_gate_terms_by_layer(model: ModelSpec) -> dict[int, list[WeightTerm]]:
    engram = model.engram
    if engram is None:
        return {}
    hc = max((p.hc_mult for p in model.layer_plans), default=ONE)
    h = model.hidden_size
    hc_h = hc * h
    embed = engram.columns_per_table * engram.width
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    by_layer: dict[int, list[WeightTerm]] = {}
    for table in engram.tables:
        by_layer.setdefault(table.layer_id, []).extend([
            _t("ple_kv_proj", hc_h + h, embed, "ple_proj", count=ONE),
            _t("ple_norm_key", hc_h, ONE, "norm", count=ONE),
            _t("ple_norm_query", hc_h, ONE, "norm", count=ONE),
            _t("ple_norm_conv", hc_h, ONE, "norm", count=ONE),
            _t("ple_conv1d", hc_h, conv_k, "conv_weight_bf16", count=ONE),
        ])
    return by_layer


def lower_ple(
    *, prefix: str, plan: Any, stage: int, deps: str, model: ModelSpec,
    parallel: ParallelSpec, quant: Any, mode: str, padded_tokens: int,
    views: Any, runtime: Any,
) -> tuple[list[OpSpec], str]:
    engram = model.engram
    if not is_qwen4_exp_ple(engram):
        raise ModelingError(f"catalog_qwen4_exp_ple_missing:layer{plan.index}")
    engram.validate()
    requests = tuple(view.request_id for view in views)
    h = model.hidden_size
    hc_mult = plan.hc_mult or ONE
    embed = engram.columns_per_table * engram.width
    act_bytes = dtype_bytes(model.activation_dtype)
    lookups = max(ONE, padded_tokens * engram.lookups_per_token_per_table)
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    state_bytes = ple_conv_state_bytes(model, runtime) * max(ONE, len(views))
    conv_input = float(padded_tokens * hc_mult * h * act_bytes)
    gated_output = float(padded_tokens * hc_mult * h * act_bytes)
    outer_residual = float(padded_tokens * hc_mult * h * act_bytes)
    ops: list[OpSpec] = []
    last = ""
    for table in engram.tables:
        if table.layer_id != plan.index:
            continue
        query = vector_op(
            op_id=f"{prefix}.ple_ngram_ids_t{table.table_id}", deps=(deps,),
            stage=stage, layer=plan.index, requests=requests,
            elements=lookups, ops_per_element=PLE_LOOKUP_OPS_PER_ELEM,
            dtype="fp32", family="engram_query", phase=mode, recipe="ple_ngram_ids",
            shape=(padded_tokens, engram.lookups_per_token_per_table),
            metadata=(("table_id", table.table_id), ("lookups", lookups)))
        ops.append(query)
        gathered = query.op_id
        has_host = bool(getattr(runtime, "engram_cpu_offload", False))
        if has_host:
            lookup = vector_op(
                op_id=f"{prefix}.ple_host_uva_t{table.table_id}", deps=(query.op_id,),
                stage=stage, layer=plan.index, requests=requests,
                elements=lookups, ops_per_element=PLE_LOOKUP_OPS_PER_ELEM,
                dtype="bf16", family="engram_rh2d", phase=mode,
                recipe="ple_host_uva_lookup", shape=(padded_tokens, embed),
                read_bytes_override=float(lookups * engram.row_bytes / max(ONE, parallel.tp)),
                write_bytes_override=float(padded_tokens * embed * BF16_BYTES))
        else:
            lookup = vector_op(
                op_id=f"{prefix}.ple_device_lookup_t{table.table_id}", deps=(query.op_id,),
                stage=stage, layer=plan.index, requests=requests,
                elements=max(ONE, padded_tokens * embed),
                ops_per_element=PLE_LOOKUP_OPS_PER_ELEM, dtype="bf16",
                family="engram_gather", phase=mode, recipe="ple_device_lookup",
                shape=(padded_tokens, embed),
                read_bytes_override=float(lookups * engram.row_bytes / max(ONE, parallel.tp)),
                write_bytes_override=float(padded_tokens * embed * BF16_BYTES))
        ops.append(lookup)
        gathered = lookup.op_id
        if parallel.tp > ONE:
            reduce = comm_op(
                op_id=f"{prefix}.ple_allreduce_t{table.table_id}", deps=(gathered,),
                stage=stage, layer=plan.index, requests=requests,
                family="tp_allreduce", phase=mode, recipe="ple_tp_allreduce",
                link_bytes=float(TWO * padded_tokens * embed * BF16_BYTES
                                 * (parallel.tp - ONE) / parallel.tp),
                shape=(padded_tokens, embed))
            ops.append(reduce)
            gathered = reduce.op_id
        kv = gemm_op(
            op_id=f"{prefix}.ple_kv_proj_t{table.table_id}", deps=(gathered,),
            stage=stage, layer=plan.index, requests=requests, m=padded_tokens,
            n=(hc_mult + ONE) * h, k=embed, quant=quant, category="ple_proj",
            phase=mode, family="gemm", recipe="ple_kv_proj",
            metadata=(("table_id", table.table_id),
                      ("replicated_projection", True)))
        gate = vector_op(
            op_id=f"{prefix}.ple_gate_t{table.table_id}", deps=(kv.op_id,),
            stage=stage, layer=plan.index, requests=requests,
            elements=max(ONE, padded_tokens * hc_mult * h),
            ops_per_element=PLE_GATE_OPS_PER_ELEM, dtype=model.activation_dtype,
            family="engram_apply", phase=mode, recipe="ple_gate",
            shape=(padded_tokens, TWO * hc_mult * h),
            read_bytes_override=float((TWO * hc_mult + ONE) * h * padded_tokens * act_bytes),
            write_bytes_override=float(TWO * hc_mult * h * padded_tokens * act_bytes))
        conv = vector_op(
            op_id=f"{prefix}.ple_short_conv_t{table.table_id}",
            deps=(gate.op_id, deps), stage=stage, layer=plan.index,
            requests=requests, elements=max(ONE, padded_tokens * hc_mult * h),
            ops_per_element=float(TWO * conv_k + TWO),
            dtype=model.activation_dtype, family="conv1d", phase=mode,
            recipe="ple_short_conv", shape=(padded_tokens, hc_mult * h),
            read_bytes_override=conv_input + outer_residual + state_bytes,
            write_bytes_override=gated_output + state_bytes)
        ops.extend([kv, gate, conv])
        last = conv.op_id
    if not last:
        raise ModelingError(f"catalog_engram_table_not_at_layer:{plan.index}")
    return ops, last


def lowrank_hc_layer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    hc_dim = plan.hc_mult * model.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"hc_{part}_norm", hc_dim, ONE, "norm"),
            _t(f"hc_{part}_down_inject", plan.hc_lowrank + plan.hc_mult, hc_dim, "linear_param_bf16"),
            _t(f"hc_{part}_up", hc_dim, plan.hc_lowrank, "linear_param_bf16"),
        ])
    return terms


def lowrank_hc_head_terms(model: ModelSpec) -> list[WeightTerm]:
    hc = max((p.hc_mult for p in model.layer_plans), default=ZERO)
    lr = max((p.hc_lowrank for p in model.layer_plans), default=ZERO)
    hc_dim = hc * model.hidden_size
    return [_t("hc_head_norm", hc_dim, ONE, "norm"),
            _t("hc_head_down", lr, hc_dim, "linear_param_bf16"),
            _t("hc_head_up", hc_dim, lr, "linear_param_bf16")]


def lowrank_hc_draft_terms(draft: DraftSpec) -> list[WeightTerm]:
    hc_dim = draft.mhc_mult * draft.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"draft_hc_{part}_norm", hc_dim, ONE, "norm"),
            _t(f"draft_hc_{part}_down_inject",
               draft.mhc_lowrank + draft.mhc_mult, hc_dim, "linear_param_bf16"),
            _t(f"draft_hc_{part}_up", hc_dim, draft.mhc_lowrank, "linear_param_bf16"),
        ])
    return terms


def lowrank_hc_draft_head_terms(draft: DraftSpec) -> list[WeightTerm]:
    hc_dim = draft.mhc_mult * draft.hidden_size
    return [
        _t("draft_hc_head_norm", hc_dim, ONE, "norm"),
        _t("draft_hc_head_down", draft.mhc_lowrank, hc_dim, "linear_param_bf16"),
        _t("draft_hc_head_up", hc_dim, draft.mhc_lowrank, "linear_param_bf16"),
    ]


def qsa_indexer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    return [_t("index_qk_proj", (plan.index_heads + ONE) * plan.index_dim,
               model.hidden_size, "indexer"),
            _t("indexer_q_norm", plan.index_dim, ONE, "norm"),
            _t("indexer_k_norm", plan.index_dim, ONE, "norm")]


def ple_conv_state_bytes(model: Any, runtime: Any = None) -> float:
    if not is_qwen4_exp_ple(getattr(model, "engram", None)):
        return ZERO_FLOAT
    engram = model.engram
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    conv_state_len = (conv_k - ONE) * engram.max_ngram
    num_spec = (int(getattr(runtime, "mtp_proposal_tokens", ZERO) or ZERO)
                if runtime is not None else ZERO)
    hc = max((p.hc_mult for p in model.layer_plans), default=ONE)
    return float((conv_state_len + num_spec) * hc * model.hidden_size * BF16_BYTES)


def qwen4_exp_pool_breakdown(
    model: Any, parallel: ParallelSpec, sequence_tokens: int, *, runtime: Any = None,
) -> dict[str, Any] | None:
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    if facts.get("model_type") != "qwen4_exp_text" and not is_qwen4_exp_ple(
            getattr(model, "engram", None)):
        return None
    qsa = [p for p in model.layer_plans if p.attention == "gqa" and p.indexer]
    gdn = [p for p in model.layer_plans if p.attention == "gdn"]
    if not qsa or not gdn:
        return None
    tp = max(ONE, parallel.tp)
    seq = int(sequence_tokens)
    slot_count = len(qsa)
    local_kv = max(ONE, (qsa[ZERO].num_kv_heads + tp - ONE) // tp)
    kv_elem = dtype_bytes(str(getattr(getattr(model, "quant", None), "kv_dtype", "")
                              or getattr(model, "kv_dtype", "bf16")))
    kv_token_bytes = int(TWO * local_kv * qsa[ZERO].head_dim * kv_elem)
    max_ssm_bytes = max(
        int(math.ceil(p.recurrent_state_elements / tp) * dtype_bytes(p.recurrent_state_dtype))
        for p in gdn)
    conv_bytes = max(
        int(math.ceil(p.conv_state_elements / tp) * dtype_bytes(p.conv_state_dtype))
        for p in gdn)
    subpage_alignment = KERNEL_BLOCK_TOKENS * kv_token_bytes
    min_block_tokens = _round_up(_ceil_div(max_ssm_bytes, kv_token_bytes), KERNEL_BLOCK_TOKENS)
    cfg_block = int(getattr(runtime, "kv_block_size", KERNEL_BLOCK_TOKENS)
                    or KERNEL_BLOCK_TOKENS)
    manager_block_size = max(min_block_tokens, _round_up(cfg_block, KERNEL_BLOCK_TOKENS))
    main_page_bytes = manager_block_size * kv_token_bytes
    state_stride = _round_up(max(conv_bytes + max_ssm_bytes, main_page_bytes), subpage_alignment)
    manager_block_size = state_stride // kv_token_bytes
    ratio = max(ONE, qsa[ZERO].index_kpool)
    idx_elem = dtype_bytes(str(getattr(getattr(model, "quant", None), "index_dtype", "")
                               or "bf16"))
    raw_page_bytes = int(TWO * ratio * qsa[ZERO].index_dim * BF16_BYTES)
    compressed_page_bytes = int((manager_block_size // ratio) * qsa[ZERO].index_dim * idx_elem)
    slot_backing_bytes = state_stride + raw_page_bytes + compressed_page_bytes
    pool_bytes_per_block = slot_count * slot_backing_bytes
    qsa_blocks = _ceil_div(max(ONE, seq), manager_block_size) if seq > ZERO else ZERO
    gdn_state_groups = _ceil_div(len(gdn), slot_count)
    raw_blocks = ONE if seq > ZERO else ZERO
    total_pool_blocks = qsa_blocks + raw_blocks + (gdn_state_groups if seq > ZERO else ZERO)
    private_ple_bytes = ple_conv_state_bytes(model, runtime) if seq > ZERO else ZERO_FLOAT
    return {
        "slot_count": slot_count, "manager_block_size": manager_block_size,
        "state_stride": state_stride, "pool_bytes_per_block": pool_bytes_per_block,
        "qsa_blocks": qsa_blocks, "gdn_state_groups": gdn_state_groups,
        "raw_blocks": raw_blocks, "total_pool_blocks": total_pool_blocks,
        "private_ple_bytes": private_ple_bytes,
        "total_pool_bytes": float(total_pool_blocks * pool_bytes_per_block) + private_ple_bytes,
    }


__all__ = [
    "build_ple_spec", "is_qwen4_exp_ple", "lower_ple", "ple_table_terms",
    "ple_gate_terms_by_layer", "lowrank_hc_layer_terms", "lowrank_hc_head_terms",
    "lowrank_hc_draft_terms", "lowrank_hc_draft_head_terms", "qsa_indexer_terms",
    "ple_conv_state_bytes", "qwen4_exp_pool_breakdown",
]
