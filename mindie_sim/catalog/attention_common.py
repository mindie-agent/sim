"""Common attention geometry/dependency records for catalog lowering.

[Role]
  Exact causal/compressed pair oracles, collective helper, legal TP widths and
  the shared AttentionResult record.

[Invariants]
  * Pair counts are exact integer sums, not fractional averages at boundaries.
  * MLA q_b/kv_b use local heads once; grouped output keeps its group axis.
  * Collective payloads are logical activation bytes.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.ops import comm_op, vector_op
from mindie_sim.catalog.context_parallel import (
    merge_metadata, merge_payload_bytes, merge_update_elements,
)
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec

ZERO = 0
ONE = 1
ZERO_FLOAT = 0.0
LSE_SCALAR_WIDTH = 1
MERGE_OPS_PER_ELEM = 3.0
PACK_LSE_OPS = 4.0
FP32_LSE_BYTES = 4.0
UPDATE_TYPE_ZERO = 0
PREFILL_PHASES = frozenset({"prefill", "draft_prefill"})


def min_positive(*bounds: int) -> int:
    """Minimum of positive bounds, or ``0`` when all bounds are unbounded/disabled."""
    positives = [int(bound) for bound in bounds if int(bound) > 0]
    return min(positives) if positives else 0


def active_dcp_degree(parallel: ParallelSpec, phase: str) -> int:
    """Effective KV-sharding DCP width for attention execution in ``phase``."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1 or str(phase) in PREFILL_PHASES:
        return 1
    cp_plan = str(getattr(parallel, "cp_plan", "dcp") or "dcp")
    if cp_plan in {"dcp", "pcp_dcp"}:
        return cp
    return 1


def store_dcp_degree(parallel: ParallelSpec, phase: str) -> int:
    """Effective KV-cache store sharding degree for ``phase``."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        return 1
    cp_plan = str(getattr(parallel, "cp_plan", "dcp") or "dcp")
    if cp_plan == "dcp":
        return cp
    if cp_plan == "pcp_dcp":
        return 1 if str(phase) in PREFILL_PHASES else cp
    return 1


@dataclass
class AttentionResult:
    ops: list[OpSpec] = field(default_factory=list)
    last: str = ""
    output_ids: tuple[str, ...] = ()
    cache_store_id: str | None = None
    topk_id: str | None = None
    selected_rows: int = 0
    index_read_bytes: float = 0.0
    cache_write_rows: int = 0
    attention_outputs: tuple[str, ...] = ()
    index_store_id: str | None = None
    candidate_id: str | None = None


def _triangular(q: int, window: int) -> float:
    n = min(max(0, q), max(0, window))
    return n * (n + 1) / 2.0


def _causal_pairs(q: int, context: int, window: int) -> float:
    """Exact sum_{i=1..q} min(context+i, window) (window<=0 means no window)."""
    q = max(0, int(q))
    context = max(0, int(context))
    if window <= 0:
        return q * context + q * (q + 1) / 2.0
    window = max(1, int(window))
    if context >= window:
        return float(q * window)
    full = min(q, window - context)
    partial = full * context + full * (full + 1) / 2.0
    return float(partial + (q - full) * window)


def _floor_sum(n: int, m: int, a: int, b: int) -> int:
    """sum_{i=0..n-1} floor((a*i+b)/m), non-negative arguments."""
    n, m, a, b = int(n), int(m), int(a), int(b)
    result = 0
    while True:
        if a >= m:
            result += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            result += n * (b // m)
            b %= m
        y_max = a * n + b
        if y_max < m:
            return result
        n = y_max // m
        b = y_max % m
        m, a = a, m


def _long_selected_pairs(
    q: int, context: int, ratio: int, topk: int, *,
    noncausal: bool = False, kv_len: int | None = None,
) -> float:
    """Exact sum of selected compressed rows; ``topk<=0`` means no cap."""
    q = max(ZERO, int(q))
    context = max(ZERO, int(context))
    ratio = int(ratio)
    if ratio <= ZERO:
        return ZERO_FLOAT
    topk = int(topk)
    if noncausal:
        horizon = max(ZERO, int(kv_len) if kv_len is not None else context + q)
        avail = horizon // ratio
        return float(q * (min(avail, topk) if topk > ZERO else avail))
    if topk <= ZERO:
        return float(_floor_sum(q, ratio, ONE, context + ONE))
    threshold = topk * ratio - context
    unsaturated = min(q, max(ZERO, threshold - ONE))
    floor_sum = _floor_sum(unsaturated, ratio, ONE, context + ONE)
    return float(floor_sum + (q - unsaturated) * topk)


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def _allreduce(
    *, prefix: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], model: ModelSpec, parallel: ParallelSpec,
    mode: str, tokens: int, suffix: str = "attn",
) -> OpSpec:
    h = model.hidden_size
    return comm_op(
        op_id=f"{prefix}.tp_{suffix}", deps=deps, stage=stage, layer=layer,
        requests=requests, family="allreduce", phase=mode,
        recipe="catalog_tp_allreduce",
        link_bytes=2.0 * (parallel.tp - 1) / parallel.tp * tokens * h
        * dtype_bytes(model.activation_dtype),
        shape=(tokens, h),
        metadata=(("collective", "allreduce"), ("group_axis", "tp"),
                  ("steps", 2 * (parallel.tp - 1)),
                  ("payload_semantics", "logical_activation")),
    )


def _group_plan(plan: LayerPlan, parallel: ParallelSpec) -> tuple[bool, int]:
    groups = int(plan.o_groups)
    tp = max(1, parallel.tp)
    if groups <= 0 or plan.num_heads <= 0 or groups > plan.num_heads:
        return False, 1
    group_heads = plan.num_heads // groups
    local_heads = max(1, plan.num_heads // tp)
    if groups % tp == 0 and group_heads * (groups // tp) == local_heads:
        return True, groups // tp
    return False, groups


def _qkv_widths(plan: LayerPlan, parallel: ParallelSpec) -> tuple[int, int, int, int]:
    local_heads = max(1, plan.num_heads // parallel.tp)
    local_kv = local_kv_heads(plan.num_kv_heads, parallel.tp)
    q_width = plan.q_projection_width(local_heads, plan.head_dim)
    kv_width = local_kv * plan.head_dim
    return local_heads, local_kv, q_width, kv_width


def dcp_partial_merge_ops(
    *, prefix: str, view: Any = None, views: Sequence[Any] | None = None,
    cp: int, local_heads: int, v_dim: int,
    stage: int, layer: int, mode: str, deps: tuple[str, ...],
    total_pairs: float, local_pairs: float, total_rows: int, local_rows: int,
    interleave: int,
) -> tuple[list[OpSpec], str]:
    """All-gather the per-rank ``(out, LSE)`` partials and log-sum-exp merge.

    One op carries the collective volume on the declared fabric; the merge
    kernel is a dependent vector op, so the model never hides the merge behind
    the all-gather's ``max`` resource term.
    """
    batch_views = tuple(views) if views is not None else ((view,) if view is not None else ())
    q = max(ONE, sum(int(item.q_len) for item in batch_views))
    requests = tuple(item.request_id for item in batch_views)
    modes = {item.execution_mode for item in batch_views}
    exec_phase = next(iter(modes)) if len(modes) == ONE else mode
    full, link = merge_payload_bytes(q, local_heads, v_dim, cp)
    merge = comm_op(
        op_id=f"{prefix}.dcp_merge", deps=deps, stage=stage, layer=layer,
        requests=requests, family="dcp_allgather", phase=exec_phase,
        recipe="catalog_dcp_lse_merge", link_bytes=link,
        shape=(q, local_heads, v_dim + LSE_SCALAR_WIDTH),
        metadata=(
            ("group_axis", "cp"),
            ("collective", "allgather"),
            ("partial_reduction", "logsumexp"),
            ("payload_semantics", "attention_out_plus_lse_fp32"),
            ("partial_payload_bytes", full),
            ("cp", cp),
            ("cp_context_strategy", "kv_context_shard_lse_merge"),
            ("cp_interleave", interleave),
            ("cp_pairs_total", float(total_pairs)),
            ("cp_pairs_local", float(local_pairs)),
            ("cp_rows_total", int(total_rows)),
            ("cp_rows_local", int(local_rows)),
        ),
    )
    update = vector_op(
        op_id=f"{prefix}.dcp_update", deps=(merge.op_id,), stage=stage,
        layer=layer, requests=requests,
        elements=int(merge_update_elements(q, local_heads, v_dim, cp)),
        ops_per_element=MERGE_OPS_PER_ELEM, dtype="fp32", family="attention_merge",
        phase=exec_phase, recipe="catalog_dcp_attention_update",
        semantic_family="dcp_logsumexp_attention_update",
        shape=(q, local_heads, v_dim),
        metadata=merge_metadata(cp, q, local_heads, v_dim) + (
            ("native_implementation", {
                "native_kind": "AttentionUpdate",
                "selection": None,
                "conditions": {
                    "scope": "partial_output_and_lse_merge_only",
                    "update_type": UPDATE_TYPE_ZERO, "partials": cp,
                    "input_dtype": "fp32", "output_dtype": "fp32",
                    "collective_excluded": True,
                },
            }),
            ("native_tensor_view", {
                "partial_lse_shapes": ((q * local_heads,),) * cp,
                "partial_output_shapes": ((q * local_heads, v_dim),) * cp,
                "output_shape": (q * local_heads, v_dim),
                "dtype": "fp32",
                "comparison_scope": "source_declared_partial_merge; capture_ABI_not_verified",
            }),
            ("native_source", {
                "repository": "vllm-ascend",
                "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
                "path": "vllm_ascend/attention/context_parallel/common_cp.py",
                "sha256": "55b44b753cfe25794f0a9a42463950bd71ec59d2b4b44f0f6bf4e4c1e46bdf24",
                "symbol": "_npu_attention_update",
                "lines": "168-195",
                "source_state": "clean_at_revision",
                "scope": "merge_semantics_only; not_collective_schedule_or_capture_binary_identity",
            }),
        ),
    )
    return [merge, update], update.op_id


def dcp_fia_query_heads(local_heads: int, tp: int, pcp: int, cp: int) -> int:
    """Local FIA query head count after vLLM-Ascend pre-attention Q gather."""
    pcp_deg = max(1, int(pcp))
    tp_deg = max(1, int(tp))
    cp_deg = max(1, int(cp))
    if cp_deg <= 1 or (pcp_deg > 1 and cp_deg == pcp_deg):
        return int(local_heads)
    gather_deg = cp_deg if pcp_deg <= 1 else tp_deg
    return int(local_heads) * max(1, gather_deg)


def dcp_query_gather_ops(
    *, prefix: str, q_ready: str, rotary: str | None, q: int,
    local_heads: int, q_nope_dim: int, q_rope_dim: int, act_dtype: str,
    parallel: ParallelSpec, cp: int, stage: int, layer: int,
    requests: tuple[str, ...], mode: str,
) -> tuple[list[OpSpec], str, str | None, int]:
    """Emit vLLM-Ascend ``_dcp_all_gather_fragments`` pre-attention Q gathers."""
    pcp = int(getattr(parallel, "pcp", 1) or 1)
    tp = int(getattr(parallel, "tp", 1) or 1)
    fia_heads = dcp_fia_query_heads(local_heads, tp, pcp, cp)
    if fia_heads == int(local_heads):
        return [], q_ready, rotary, local_heads
    gather_axis = "cp" if pcp <= 1 else "tp"
    gather_deg = cp if pcp <= 1 else tp
    elem_b = float(dtype_bytes(act_dtype))
    rows = max(1, int(q))
    ops: list[OpSpec] = []
    nope_op = comm_op(
        op_id=f"{prefix}.dcp_q_nope_allgather" if q_rope_dim > 0 else f"{prefix}.dcp_q_allgather",
        deps=(q_ready,), stage=stage, layer=layer, requests=requests,
        family="allgather", phase=mode, recipe="dcp_query_allgather",
        link_bytes=float(rows * local_heads * q_nope_dim * elem_b * (gather_deg - 1)),
        shape=(rows, fia_heads, q_nope_dim),
        metadata=(("group_axis", gather_axis), ("collective", "allgather"),
                  ("gathered_heads", fia_heads)),
    )
    ops.append(nope_op)
    new_rotary = rotary
    if q_rope_dim > 0 and rotary is not None:
        pe_op = comm_op(
            op_id=f"{prefix}.dcp_q_pe_allgather",
            deps=(rotary,), stage=stage, layer=layer, requests=requests,
            family="allgather", phase=mode, recipe="dcp_query_allgather",
            link_bytes=float(rows * local_heads * q_rope_dim * elem_b * (gather_deg - 1)),
            shape=(rows, fia_heads, q_rope_dim),
            metadata=(("group_axis", gather_axis), ("collective", "allgather"),
                      ("gathered_heads", fia_heads)),
        )
        ops.append(pe_op)
        new_rotary = pe_op.op_id
    return ops, nope_op.op_id, new_rotary, fia_heads


def dcp_a2a_combine_ops(
    *, prefix: str, views: Sequence[Any], cp: int, parallel: ParallelSpec,
    local_heads: int, fia_heads: int, v_dim: int, out_dtype: str,
    stage: int, layer: int, mode: str, deps: tuple[str, ...],
    total_pairs: float, local_pairs: float, total_rows: int, local_rows: int,
    interleave: int,
) -> tuple[list[OpSpec], str]:
    """Emit vLLM-Ascend ``dcp_a2a_fused`` across PCP=1, DCP=PCP, and DCP=TP*PCP."""
    pcp = int(getattr(parallel, "pcp", 1) or 1)
    tp = int(getattr(parallel, "tp", 1) or 1)
    q = max(1, sum(int(item.q_len) for item in views))
    requests = tuple(item.request_id for item in views)
    modes = {item.execution_mode for item in views}
    exec_phase = next(iter(modes)) if len(modes) == 1 else mode
    out_b = float(dtype_bytes(out_dtype))
    head_bytes = float(v_dim) * out_b + FP32_LSE_BYTES
    ops: list[OpSpec] = []
    if pcp > 1 and cp == pcp:
        gather = comm_op(
            op_id=f"{prefix}.dcp_merge", deps=deps, stage=stage, layer=layer,
            requests=requests, family="dcp_allgather", phase=exec_phase,
            recipe="catalog_dcp_lse_merge",
            link_bytes=float(q * local_heads * head_bytes * (pcp - 1)),
            shape=(q, local_heads, v_dim + LSE_SCALAR_WIDTH),
            metadata=(("group_axis", "pcp"), ("collective", "allgather"),
                      ("cp", cp), ("cp_interleave", interleave),
                      ("cp_pairs_total", float(total_pairs)),
                      ("cp_pairs_local", float(local_pairs)),
                      ("cp_rows_total", int(total_rows)),
                      ("cp_rows_local", int(local_rows))),
        )
        ops.append(gather)
        combine_deps = (gather.op_id,)
        combine_recipe = "catalog_dcp_attention_update"
    else:
        pack = vector_op(
            op_id=f"{prefix}.dcp_output_pack", deps=deps, stage=stage, layer=layer,
            requests=requests, elements=q * fia_heads, ops_per_element=PACK_LSE_OPS,
            dtype=out_dtype, family="elementwise", phase=exec_phase,
            recipe="dcp_output_pack", shape=(q, fia_heads, v_dim),
            read_bytes_override=float(q * fia_heads * head_bytes),
            write_bytes_override=float(q * fia_heads * head_bytes),
        )
        ops.append(pack)
        a2a_axis = "cp" if pcp <= 1 else "tp"
        a2a_deg = cp if pcp <= 1 else tp
        a2a = comm_op(
            op_id=f"{prefix}.dcp_output_alltoall", deps=(pack.op_id,),
            stage=stage, layer=layer, requests=requests, family="alltoall",
            phase=exec_phase, recipe="dcp_output_alltoall",
            link_bytes=float(q * local_heads * head_bytes * (a2a_deg - 1)),
            shape=(q, local_heads, v_dim + LSE_SCALAR_WIDTH),
            metadata=(("group_axis", a2a_axis), ("collective", "alltoall"),
                      ("source", "dcp_a2a_fused"), ("cp", cp)),
        )
        ops.append(a2a)
        combine_deps = (a2a.op_id,)
        if pcp > 1:
            pcp_ag = comm_op(
                op_id=f"{prefix}.dcp_pcp_allgather", deps=combine_deps,
                stage=stage, layer=layer, requests=requests,
                family="dcp_allgather", phase=exec_phase,
                recipe="catalog_dcp_lse_merge",
                link_bytes=float(q * local_heads * tp * head_bytes * (pcp - 1)),
                shape=(q, local_heads, v_dim + LSE_SCALAR_WIDTH),
                metadata=(("group_axis", "pcp"), ("collective", "allgather"),
                          ("cp", cp), ("cp_interleave", interleave)),
            )
            ops.append(pcp_ag)
            combine_deps = (pcp_ag.op_id,)
        combine_recipe = "dcp_lse_merge"
    elements = int(merge_update_elements(q, local_heads, v_dim, cp))
    combine = vector_op(
        op_id=f"{prefix}.dcp_lse_combine", deps=combine_deps, stage=stage,
        layer=layer, requests=requests, elements=elements,
        ops_per_element=MERGE_OPS_PER_ELEM, dtype="fp32",
        family="attention_merge", phase=exec_phase, recipe=combine_recipe,
        semantic_family="dcp_logsumexp_attention_update",
        shape=(q, local_heads, v_dim),
        read_bytes_override=float(q * local_heads * head_bytes * cp),
        write_bytes_override=float(q * local_heads * v_dim * out_b),
        metadata=merge_metadata(cp, q, local_heads, v_dim),
    )
    ops.append(combine)
    return ops, combine.op_id

