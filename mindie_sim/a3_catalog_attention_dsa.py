"""DSA compressed-cache, index/TopK/candidate and SWA attention lowering.

[Role]
  Build C4/C128/C1/C2 producers, Index-K stores, selection producers,
  candidate-block ownership and per-consumer selected-row attention.

[Invariants]
  * Compressor input projections execute for all current hidden tokens even
    when zero new compressed output rows complete.
  * Index-K consumes unrotated latent before in-place RoPE/store.
  * Candidate production exists only at the declared source layer; consumers
    retain an explicit dependency on it.
  * Every consumer pays its own selected-row reads.

[Timeouts & Locks]
  Pure arithmetic; no I/O.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .a3_catalog_cache import (
    completed_rows, index_selection_cost, new_completed_rows,
)
from .a3_catalog_attention_common import (
    _causal_pairs, _long_selected_pairs, _requests,
)
from .a3_catalog_ops import (
    barrier_op, cache_store_op, comm_op, gemm_op, vector_op,
)
from .a3_catalog_types import QuantSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_ops import OpSpec, _op
from .a3_types import A3InputError, ParallelSpec, dtype_bytes


def _index_producer(
    *, prefix: str, plan: LayerPlan, stage: int, deps: str, seq: int,
    model: ModelSpec, parallel: ParallelSpec, quant: QuantSpec, mode: str,
    requests: tuple[str, ...], views: Sequence[Any], cache_dep: str | None = None,
    candidate_dep: str | None = None,
) -> tuple[list[OpSpec], str, float, str | None]:
    """Create query projection + score/TopK (+ candidate) selection ops."""
    facts = dict(model.catalog_facts)
    index_heads = int(facts.get("index_n_heads", 32) or 32)
    index_dim = int(facts.get("index_head_dim", 128) or 128)
    topk = int(facts.get("index_topk", 0) or 0)
    native = str(facts.get("profile", ""))
    cost = index_selection_cost(model, plan, seq)
    ops: list[OpSpec] = []
    q_rank = plan.q_lora_rank or model.hidden_size
    q_deps = (deps,) + ((cache_dep,) if cache_dep else ())
    wq = gemm_op(
        op_id=f"{prefix}.index_q", deps=q_deps, stage=stage, layer=plan.index,
        requests=requests, m=1 if not views else max(1, sum(v.q_len for v in views)),
        n=index_heads * index_dim, k=q_rank, quant=quant,
        category="indexer_fp8", phase=mode, family="gemm",
        semantic_family="dsa_index_query",
        metadata=(("index_heads", index_heads), ("index_dim", index_dim)),
    )
    ops.append(wq)
    tokens = max(1, sum(max(1, v.q_len) for v in views))
    score = _op(
        op_id=f"{prefix}.index_score", family="gemm", phase=mode, stage=stage,
        layer=plan.index, requests=requests,
        semantic_shape=(tokens, max(1, cost.scored_rows), index_heads * index_dim),
        local_shape=(tokens, max(1, cost.scored_rows), index_heads * index_dim),
        padded_shape=(tokens, max(1, cost.scored_rows), index_heads * index_dim),
        launch_shape=(tokens, max(1, cost.scored_rows), index_heads * index_dim),
        flops=2.0 * tokens * max(1, cost.scored_rows) * index_heads * index_dim,
        hbm_bytes=cost.index_read_bytes + tokens * index_heads * index_dim * 2.0
        + tokens * max(1, cost.scored_rows) * 2.0,
        mte2_bytes=cost.index_read_bytes + tokens * index_heads * index_dim * 2.0,
        mte3_bytes=tokens * max(1, cost.scored_rows) * 2.0,
        deps=(wq.op_id,), recipe="catalog_dsa_index_score",
        dtype=model.activation_dtype,
        metadata=(("semantic_family", "dsa_index_score"),
                  ("scored_rows", cost.scored_rows),
                  ("candidate_rows", cost.candidate_rows),
                  ("candidate_filter", plan.candidate_filter),
                  ("index_row_bytes", cost.index_row_bytes)),
    )
    ops.append(score)
    select_deps = (score.op_id,) + ((candidate_dep,) if candidate_dep else ())
    select = vector_op(
        op_id=f"{prefix}.index_topk", deps=select_deps, stage=stage,
        layer=plan.index, requests=requests,
        elements=max(1, tokens * max(1, cost.scored_rows)),
        ops_per_element=5.0, dtype="fp32", family="topk_selection", phase=mode,
        recipe="catalog_dsa_topk",
        shape=(tokens, max(1, cost.scored_rows)),
        metadata=(("selected_rows", cost.selected_rows), ("topk", topk),
                  ("index_source_layer_id", plan.index_source_layer_id)),
    )
    ops.append(select)
    last = select.op_id
    candidate_id: str | None = None
    if plan.owns_candidate_blocks:
        # Candidate source selection publishes block IDs for later selections.
        block = int(facts.get("candidate_block_size", 8) or 8)
        blocks = max(1, int(facts.get("candidate_topk_blocks", 2048) or 2048))
        candidate = vector_op(
            op_id=f"{prefix}.candidate_blocks", deps=(select.op_id,), stage=stage,
            layer=plan.index, requests=requests,
            elements=max(1, (completed_rows(seq, plan.compress_ratio or 1) + block - 1) // block),
            ops_per_element=2.0, dtype="fp32", family="candidate_blocks", phase=mode,
            recipe="catalog_dsa_candidate_blocks",
            shape=(blocks, block),
            metadata=(("candidate_blocks", blocks), ("block_size", block)),
        )
        ops.append(candidate)
        candidate_id = candidate.op_id
        last = candidate.op_id
    return ops, last, cost.index_read_bytes, candidate_id


def _long_cache_producer(
    *, prefix: str, plan: LayerPlan, stage: int, deps: str, seq: int,
    new_rows: int, input_tokens: int, model: ModelSpec,
    parallel: ParallelSpec, quant: QuantSpec, mode: str,
    requests: tuple[str, ...],
) -> tuple[list[OpSpec], str, str | None]:
    """C4/C128/C1/C2 long-KV and index-K producers with in-place order.

    The compressor projections consume all input hidden tokens before pooling;
    zero newly completed output rows therefore never replaces input compute.
    """
    ops: list[OpSpec] = []
    h = model.hidden_size
    d = plan.head_dim
    facts = dict(model.catalog_facts)
    native = str(facts.get("profile", ""))
    rows = max(0, int(new_rows))
    tokens = max(1, int(input_tokens))
    compressor_category = (
        "compressor_fp32_runtime"
        if native == "deepseek_v41" and plan.compress_ratio == 2
        else "compressor_bf16_runtime"
    )
    # Input-side compressor projections always execute for the current tokens.
    comp = gemm_op(
        op_id=f"{prefix}.compressor_kv", deps=(deps,), stage=stage,
        layer=plan.index, requests=requests, m=tokens, n=d, k=h,
        quant=quant, category=compressor_category, phase=mode, family="gemm",
        semantic_family="dsa_compressor_kv", weight_count=1,
        semantic_shape=(tokens, d, h),
        metadata=(("compressor", f"ratio{plan.compress_ratio}"),
                  ("input_tokens", tokens), ("completed_new_rows", rows)),
    )
    pool = vector_op(
        op_id=f"{prefix}.compressor_pool", deps=(comp.op_id,), stage=stage,
        layer=plan.index, requests=requests,
        elements=max(1, tokens * d), ops_per_element=2.0,
        dtype=model.activation_dtype, family="compress_pool", phase=mode,
        recipe="catalog_compressor_pool",
        semantic_family="dsa_compressor_pool",
        metadata=(("ratio", plan.compress_ratio), ("completed_new_rows", rows)),
    )
    ops.extend([comp, pool])
    last = pool.op_id
    long_ready = pool.op_id
    gate = None
    ring_id: str | None = None
    if native == "deepseek_v41" and plan.compress_ratio == 2:
        gate = gemm_op(
            op_id=f"{prefix}.compressor_gate", deps=(deps,), stage=stage,
            layer=plan.index, requests=requests, m=tokens, n=d, k=h,
            quant=quant, category=compressor_category, phase=mode, family="gemm",
            semantic_family="dsa_compressor_gate_ring",
            metadata=(("input_tokens", tokens),
                      ("executes_without_new_pairs", True)),
        )
        ring = vector_op(
            op_id=f"{prefix}.c2_ring", deps=(pool.op_id, gate.op_id), stage=stage,
            layer=plan.index, requests=requests,
            elements=max(1, 32 * 2 * d), ops_per_element=2.0,
            dtype="fp32", family="recurrent_state", phase=mode,
            recipe="catalog_c2_ring", shape=(32, 2, d),
            metadata=(("capacity_rows", 32), ("valid_rows", min(seq, 32)),
                      ("ring_dtype", "fp32"),
                      ("ring_updates_without_new_pairs", True)),
        )
        ops.extend([gate, ring])
        last = ring.op_id
        long_ready = ring.op_id
        ring_id = ring.op_id
    index_store = None
    if rows <= 0:
        # No new completed compressed rows: the persistent cache is unchanged.
        # A real readiness barrier preserves the producer/consumer dependency.
        long_barrier = barrier_op(
            op_id=f"{prefix}.long_cache_ready", deps=(long_ready,),
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            recipe="catalog_long_cache_no_new_rows", shape=(1, d),
            metadata=(("new_rows", 0), ("cache_reused", True),
                      ("input_projections_executed", True)),
        )
        ops.append(long_barrier)
        if plan.owns_index_k:
            index_barrier = barrier_op(
                op_id=f"{prefix}.index_cache_ready", deps=(long_barrier.op_id,),
                stage=stage, layer=plan.index, requests=requests, phase=mode,
                recipe="catalog_index_cache_no_new_rows",
                shape=(1, int(facts.get("index_head_dim", 128) or 128)),
                metadata=(("new_rows", 0), ("cache_reused", True)),
            )
            ops.append(index_barrier)
            index_store = index_barrier.op_id
        return ops, long_barrier.op_id, index_store
    norm = vector_op(
        op_id=f"{prefix}.compressor_norm", deps=(pool.op_id,), stage=stage,
        layer=plan.index, requests=requests, elements=max(1, rows * d),
        ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
        phase=mode, recipe="catalog_compressor_norm",
        shape=(max(1, rows), d),
        metadata=(("tokens", max(1, rows)), ("reduction_width", d),
                  ("norm_type", "rmsnorm"), ("norm_eps", model.norm_eps)),
    )
    ops.append(norm)
    last = norm.op_id
    long_ready = norm.op_id
    if gate is not None:
        long_ready = f"{prefix}.c2_ring"
    index_store = None
    if plan.owns_index_k:
        index_dim = int(facts.get("index_head_dim", 128) or 128)
        # Index-K consumes the unrotated latent *before* the in-place long-KV
        # RoPE.  Its dependency is the norm, not the later store.
        wk = gemm_op(
            op_id=f"{prefix}.index_k_proj", deps=(norm.op_id,), stage=stage,
            layer=plan.index, requests=requests, m=max(1, rows),
            n=index_dim, k=d, quant=quant, category="norm", phase=mode,
            family="gemm", semantic_family="dsa_index_k_projection",
            metadata=(("consumes", "unrotated_latent"),),
        )
        k_norm = vector_op(
            op_id=f"{prefix}.index_k_norm", deps=(wk.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=max(1, rows * index_dim),
            ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="catalog_index_k_norm",
            shape=(rows, index_dim),
            metadata=(("tokens", rows), ("reduction_width", index_dim),
                      ("norm_type", "rmsnorm"), ("norm_eps", model.norm_eps)),
        )
        quantize = vector_op(
            op_id=f"{prefix}.index_k_quant", deps=(k_norm.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=max(1, rows * index_dim),
            ops_per_element=2.0, dtype="bf16", family="quantize", phase=mode,
            recipe="catalog_index_k_quantize", semantic_family="int8_per_row_scale",
            read_write=1.0, hbm_bytes_override=float(rows * index_dim * 3),
            shape=(rows, index_dim),
            metadata=(("input_format", "bf16"), ("output_format", "int8"),
                      ("activation_format", "bf16"),
                      ("quant_output_dtype", "int8"),
                      ("quant_axis", "per_row"),
                      ("scale_axis", "per_row"),
                      ("scale_dtype", "fp16"),
                      ("scale_format", "fp16"),
                      ("scale_bytes", 2.0),
                      ("scale_shape", (rows,)),
                      ("scale_elements", rows),
                      ("row_width", index_dim),
                      ("tokens", rows),
                      ("per_row_scale", True)),
        )
        store = cache_store_op(
            op_id=f"{prefix}.index_k_store", deps=(quantize.op_id,), stage=stage,
            layer=plan.index, requests=requests, phase=mode,
            recipe="catalog_index_k_store", rows=rows,
            row_bytes=index_dim + 2, dtype="int8", kind="index_k",
            metadata=(("scale_bytes_per_row", 2), ("shared_owner", True),
                      ("shared_owner_layer", plan.index)),
        )
        ops.extend([wk, k_norm, quantize, store])
        index_store = store.op_id
        last = store.op_id
    rope_deps = (last,) + ((ring_id,) if ring_id and ring_id != last else ())
    rope = vector_op(
        op_id=f"{prefix}.latent_rope", deps=rope_deps, stage=stage,
        layer=plan.index, requests=requests, elements=max(1, rows * d),
        ops_per_element=3.0, dtype=model.activation_dtype, family="rope",
        phase=mode, recipe="catalog_inplace_latent_rope",
        semantic_family="inplace_latent_rope", read_write=2.0,
        metadata=(("in_place", True),
                  ("ordering", "index_k_before_inplace_rope")),
    )
    store = cache_store_op(
        op_id=f"{prefix}.long_kv_store", deps=(rope.op_id,), stage=stage,
        layer=plan.index, requests=requests, phase=mode,
        recipe="catalog_long_kv_store", rows=rows,
        row_bytes=d * 2, dtype=quant.kv_dtype, kind="long_kv",
        metadata=(("compress_ratio", plan.compress_ratio),
                  ("owner_layer", plan.index), ("shared_owner", True),
                  ("shared_owner_layer", plan.index)),
    )
    ops.extend([rope, store])
    return ops, store.op_id, index_store


def _dsa_attention_view(
    *, prefix: str, view: Any, plan: LayerPlan, stage: int, deps: tuple[str, ...],
    local_heads: int, qk_dim: int, v_dim: int, selected: int, swa: int,
    cache_row_bytes: int, model: ModelSpec,
) -> tuple[list[OpSpec], str]:
    q = max(1, view.q_len)
    new_rows = new_completed_rows(view.context, q, plan.compress_ratio) if plan.compress_ratio else 0
    facts = dict(model.catalog_facts)
    topk = int(facts.get("index_topk", 0) or 0)
    if plan.compress_ratio <= 0:
        long_pairs = 0.0
    else:
        cap = 0 if (str(facts.get("profile", "")) == "deepseek_v4"
                    and plan.compress_ratio == 128) else topk
        long_pairs = _long_selected_pairs(q, view.context, plan.compress_ratio, cap)
    swa_pairs = _causal_pairs(q, view.context, plan.window)
    pairs = long_pairs + swa_pairs
    q_bytes = q * local_heads * qk_dim * dtype_bytes(model.activation_dtype)
    kv_bytes = (selected + swa) * cache_row_bytes
    scores = max(1, int(q * local_heads * (selected + swa)))
    qk = _op(
        op_id=f"{prefix}.qk", family="attention_qk", phase=view.execution_mode,
        stage=stage, layer=plan.index, requests=(view.request_id,),
        semantic_shape=(q, selected + swa, local_heads, qk_dim),
        local_shape=(q, max(1, selected + swa), local_heads, qk_dim),
        padded_shape=(q, max(1, selected + swa), local_heads, qk_dim),
        launch_shape=(q, max(1, selected + swa), local_heads, qk_dim),
        flops=2.0 * local_heads * pairs * qk_dim,
        hbm_bytes=q_bytes + kv_bytes + scores * 2.0,
        mte2_bytes=q_bytes + kv_bytes, mte3_bytes=scores * 2.0,
        deps=deps, recipe="catalog_dsa_qk", dtype=model.activation_dtype,
        metadata=(("selected_long_rows", selected), ("swa_rows", swa),
                  ("new_compressed_rows", new_rows), ("q_len", q),
                  ("candidate_filter", plan.candidate_filter)),
    )
    soft = vector_op(
        op_id=f"{prefix}.softmax", deps=(qk.op_id,), stage=stage,
        layer=plan.index, requests=(view.request_id,),
        elements=scores, ops_per_element=5.0, dtype="fp32",
        family="attention_softmax", phase=view.execution_mode,
        recipe="catalog_dsa_softmax",
        shape=(q, local_heads, max(1, selected + swa)),
    )
    pv = _op(
        op_id=f"{prefix}.pv", family="attention_pv", phase=view.execution_mode,
        stage=stage, layer=plan.index, requests=(view.request_id,),
        semantic_shape=(q, selected + swa, local_heads, v_dim),
        local_shape=(q, max(1, selected + swa), local_heads, v_dim),
        padded_shape=(q, max(1, selected + swa), local_heads, v_dim),
        launch_shape=(q, max(1, selected + swa), local_heads, v_dim),
        flops=2.0 * local_heads * pairs * v_dim,
        hbm_bytes=kv_bytes + scores * 2.0 + q * local_heads * v_dim
        * dtype_bytes(model.activation_dtype),
        mte2_bytes=kv_bytes + scores * 2.0,
        mte3_bytes=q * local_heads * v_dim * dtype_bytes(model.activation_dtype),
        deps=(soft.op_id,), recipe="catalog_dsa_pv",
        dtype=model.activation_dtype,
    )
    return [qk, soft, pv], pv.op_id
