"""GQA tensor views, cache-store adapters and decode/prefill contract records.

[Role]
  Provide ``AttentionPolicy``, QKV split layout helpers, ``split_qkv_rmsnorm_mrope``
  subgraph declarations, paged/contiguous tensor views, and cache representation
  store adapters for ``mindie_sim.prediction.attention``.

[Invariants]
  * Paged vs contiguous KV tensor views reflect explicit storage declarations,
    never query-length heuristics.
  * K and V cache formats resolve once per layer call; ``k_eq_v`` shares only
    the projection matrix, never the post-norm/post-RoPE cache planes.

[Timeouts & Locks]
  Constant/per-request metadata construction; no locks or I/O.

[Side-Effect Policy]
  Pure record construction; ``_store_cache`` appends only to the caller's ``ops``.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Literal

from mindie_sim.catalog.cache_format import gqa_kv_formats
from mindie_sim.catalog.context_parallel import owned_positions
from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.runtime_features import canonical_prefix_cache_key
from mindie_sim.schema.tensors import TensorLayout

from .attention_kv_access import paged_fia_plan
from mindie_sim.prediction.cache_representation import store as representation_store
from mindie_sim.prediction.memory_traffic import require_machine_memory_hierarchy
from mindie_sim.prediction.structure_groups import declare_group

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py",
        source_sha256="ba3c7e740c3a2a9c8850733d3be77fefac79bc3ed02a45b4be076b58d6f7632d",
    ),
)
DEFAULT_QUERY_TILE: int = 128
DEFAULT_RECURRENT_CHUNK: int = 64
QKV_OUTPUT_SLOTS: int = 4

_QKV_NORM_ROPE_SOURCE = {
    "repository": "vllm-ascend",
    "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "path": "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py",
    "sha256": "ba3c7e740c3a2a9c8850733d3be77fefac79bc3ed02a45b4be076b58d6f7632d",
    "symbol": "triton_split_qkv_rmsnorm_mrope",
    "lines": "29-57,83-110,262-278,282-366",
    "source_state": "clean_at_revision",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def _memory_hierarchy(machine, *, hierarchy=None):
    """Declared device hierarchy for traffic planning; fails closed when absent."""
    return require_machine_memory_hierarchy(machine, hierarchy=hierarchy)


def uses_mrope_qkv_kernel(model, plan) -> bool:
    """True only when the layer uses AscendMRotaryEmbedding (split_qkv_rmsnorm_mrope).

    Ordinary text RoPE models (e.g. Qwen3-32B, Qwen3-30B-A3B, Qwen3-235B-A22B,
    Gemma-4, MiniMax) split QKV into zero-copy views and invoke Q/K RMSNorm
    separately rather than the Triton MRoPE fused kernel.
    """
    if getattr(plan, "q_output_gate", False) and not getattr(plan, "q_gate_separate", False):
        return True
    facts = dict(getattr(model, "catalog_facts", ()) or {})
    if not facts:
        return True
    if facts.get("vision_config") or facts.get("has_vision_tower"):
        return True
    rope_params = facts.get("rope_parameters") or {}
    if isinstance(rope_params, Mapping):
        if rope_params.get("mrope_section") or rope_params.get("mrope_interleaved"):
            return True
    return False


def _declare_ordinary_qk_norm_rope(ops, qkv, norm_ids, rope, views, dtype):
    """Declare unfused ordinary text Q-norm + K-norm + RoPE (no MRoPE fusion)."""
    members = (*norm_ids, rope.op_id)
    declare_group(
        ops, key=qkv.op_id + ".norm_rope", role="qkv_norm_rope",
        family="qkv_preprocess", members=members,
        entries=tuple(norm_ids), anchor=rope.op_id,
        view={
            "input_shape": views["query"].storage_shape, "dtype": dtype,
            "input_views": {name: layout.to_dict() for name, layout in views.items()},
        },
        implementation={
            "native_kind": "RmsNorm+Rotary", "selection": None,
            "conditions": {"qk_norm": True, "mrope_fused_kernel": False},
        },
        source=_QKV_NORM_ROPE_SOURCE,
        fusion_status="unfused_decomposed",
        fusion_reason="ordinary_text_rope_no_mrope_kernel",
        native_fusion_eligible=False,
    )


def _qkv_views(tokens, heads, kv_heads, width, has_gate, k_eq_v=False, *, v_width: int | None = None):
    """Q and gate alternate by head; all slices retain the projected row pitch."""
    eff_v_width = width if v_width is None else int(v_width)
    query, key = heads * width, kv_heads * width
    value = kv_heads * eff_v_width
    gate = query if has_gate else 0
    pitch = query + gate + key + (0 if k_eq_v else value)
    storage = (tokens, pitch)
    val_dim = width if k_eq_v else eff_v_width
    views = {
        "query": TensorLayout(
            (tokens, heads, width), (pitch, width * (2 if has_gate else 1), 1),
            storage_shape=storage),
        "key": TensorLayout(
            (tokens, kv_heads, width), (pitch, width, 1),
            offset=query + gate, storage_shape=storage),
        "value": TensorLayout(
            (tokens, kv_heads, val_dim), (pitch, val_dim, 1),
            offset=query + gate if k_eq_v else query + gate + key, storage_shape=storage),
    }
    if has_gate:
        views["gate"] = TensorLayout(
            (tokens, heads, width), (pitch, 2 * width, 1),
            offset=width, storage_shape=storage)
    return views


def _qkv_output_copy(qkv, layout, output, dtype):
    """The native wrapper allocates V/gate outputs and writes the selected values."""
    return vector_op(
        op_id=qkv.op_id + "." + output, deps=(qkv.op_id,), stage=qkv.stage,
        layer=qkv.layer, requests=qkv.requests, elements=layout.elements,
        ops_per_element=0, dtype=dtype, family="tensor_copy", phase=qkv.phase,
        recipe="qkv_split_materialize", shape=layout.shape,
        metadata=(
            ("source_layout", layout.to_dict()),
            ("target_layout", TensorLayout.contiguous(layout.shape).to_dict()),
            ("source_output", output),
            ("materialization_reason", "split_qkv_native_output"),
            ("native_source", _QKV_NORM_ROPE_SOURCE),
        ),
    )


def _declare_qkv_norm_rope(ops, qkv, normalized, rope, copies, views, dtype):
    """The projected tensor is an input; its GEMM and later attention stay outside."""
    members = (normalized.op_id, rope.op_id, *(op.op_id for op in copies))
    declare_group(
        ops, key=qkv.op_id + ".norm_rope", role="qkv_norm_rope",
        family="qkv_preprocess", members=members,
        entries=(normalized.op_id, *(op.op_id for op in copies)),
        anchor=rope.op_id,
        view={
            "input_shape": views["query"].storage_shape, "dtype": dtype,
            "input_views": {name: layout.to_dict() for name, layout in views.items()},
            "output_shapes": {
                name: (layout.shape[0], math.prod(layout.shape[1:]))
                for name, layout in views.items()
            },
            "required_output_slots": QKV_OUTPUT_SLOTS,
        },
        implementation={
            "native_kind": "split_qkv_rmsnorm_mrope_kernel", "selection": None,
            "conditions": {
                "qk_norm": True, "gate_present": "gate" in views,
                "boundary_input": "projected_qkv",
                "boundary_views": {name: layout.to_dict() for name, layout in views.items()},
            },
        },
        source=_QKV_NORM_ROPE_SOURCE,
        coverage={
            "kind": "complete_subgraph", "required_output_slots": QKV_OUTPUT_SLOTS,
            "limitations": [
                "retained_capture_exposes_only_one_of_q_k_v_gate_output_descriptors",
                "mrope_sections_partial_rotary_width_epsilon_bias_and_weight_shift_not_declared",
                "semantic_work_is_not_a_selected_native_dispatch_or_measured_duration",
            ],
        },
    )


@dataclass(frozen=True)
class AttentionPolicy:
    query_tile: int = DEFAULT_QUERY_TILE
    reuse_kv_across_query_heads: bool = True
    fused_batch: bool = True
    recurrent_chunk: int = DEFAULT_RECURRENT_CHUNK
    kv_storage: str = "auto"

    def __post_init__(self):
        if self.kv_storage not in {"auto", "paged"}:
            raise ModelingError("attention_kv_storage_requires_auto_or_paged")


def _format_dict(fmt) -> dict[str, object]:
    """Serialize one resolved ``CacheFormat`` for layer-exact downstream ``kv_access``."""
    return {
        "dtype": fmt.dtype,
        "group_size": fmt.group_size,
        "scale_dtype": fmt.scale_dtype,
        "suffix_elements": fmt.suffix_elements,
        "suffix_dtype": fmt.suffix_dtype,
        "packed_elements": fmt.packed_elements,
    }


def _resolved_cache_regions(key_fmt, value_fmt, shared: bool, regions) -> dict[str, object]:
    """Emit layer/phase-resolved cache representations without losing scoped overrides."""
    has_nondefault = any(
        fmt.group_size or fmt.suffix_elements or fmt.packed_elements > 1
        for fmt in (key_fmt, value_fmt)
    )
    if not regions and not has_nondefault and key_fmt == value_fmt:
        return {}
    if shared:
        return {"gqa_kv": _format_dict(key_fmt)}
    return {"gqa_kv_key": _format_dict(key_fmt), "gqa_kv_value": _format_dict(value_fmt)}


def _gqa_formats(model, *, plan=None, phase=None):
    """Resolve shared or independent K/V cache formats once."""
    regions = dict(getattr(model, "metadata", ())).get("cache_representations", {})
    key_fmt, value_fmt, shared = gqa_kv_formats(model, plan=plan, phase=phase)
    return key_fmt, value_fmt, regions, shared


def _contiguous_tensor_view(
    views, heads, kv_heads, width, model, quant, parallel, *,
    cacheless=False, v_width: int | None = None,
):
    """Fresh prefill has explicit TND operands, before any paged-cache reuse."""
    if parallel.cp != 1 or parallel.pcp != 1 or any(
        (not cacheless and view.phase != "prefill") or view.context
        for view in views if int(view.logical_q_len) > 0
    ):
        return ()
    eff_v_width = width if v_width is None else int(v_width)
    q_total = sum(int(view.logical_q_len) for view in views)
    kv_total = sum(int(view.kv_len) for view in views if int(view.logical_q_len) > 0)
    query = (q_total, heads, width)
    output = (q_total, heads, eff_v_width)
    key = (kv_total, kv_heads, width)
    value = (kv_total, kv_heads, eff_v_width)
    return (("attention_tensor_view", {
        "query_shape": query, "key_shape": key, "value_shape": value, "output_shape": output,
        "query_dtype": model.activation_dtype, "key_dtype": quant.kv_dtype,
        "value_dtype": quant.kv_dtype, "output_dtype": model.activation_dtype,
        "head_dim": width, "v_head_dim": eff_v_width,
        "logical_layout": "TND", "kv_storage": "contiguous",
    }),)


def _attention_tensor_view(
    views, heads, kv_heads, width, model, quant, parallel, policy, runtime,
    causal=True, cacheless=False, *, plan=None, phase=None,
):
    """A paged ABI requires an explicit storage declaration, not a query length."""
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    if cacheless or policy.kv_storage != "paged":
        return _contiguous_tensor_view(
            views, heads, kv_heads, width, model, quant, parallel,
            cacheless=cacheless, v_width=v_width)
    if runtime is None or parallel.cp != 1 or parallel.pcp != 1:
        return ()
    active_views = tuple(view for view in views if int(view.logical_q_len) > 0)
    q_total = sum(int(view.logical_q_len) for view in active_views)
    query = (q_total, heads, width)
    output = (q_total, heads, v_width)
    key_tail = (runtime.kv_block_size, kv_heads * width)
    value_tail = (runtime.kv_block_size, kv_heads * v_width)
    eff_phase = phase or (getattr(views[0], "phase", None) if views else None)
    key_fmt, value_fmt, regions, _shared = _gqa_formats(model, plan=plan, phase=eff_phase)
    window = int(getattr(plan, "window", 0) or 0)
    native = None
    page_size = int(getattr(runtime, "kv_block_size", 0) or 0)
    if page_size > 0 and active_views:
        native = paged_fia_plan(
            requests=tuple((int(view.logical_q_len), int(view.context) + int(view.logical_q_len))
                           for view in active_views),
            query_lengths=tuple(int(view.logical_q_len) for view in active_views),
            kv_lengths=tuple(int(view.context) + int(view.logical_q_len) for view in active_views),
            page_size=page_size, query_heads=heads, kv_heads=kv_heads, head_dim=width,
            v_head_dim=v_width, window=window, shared_format=_shared,
            query_dtype=model.activation_dtype, output_dtype=model.activation_dtype,
            key_format=key_fmt, value_format=value_fmt,
            phase=getattr(active_views[0], "phase", "decode"),
            mask="causal" if causal else "none", return_lse=False, output_prepare=None,
        )
    residency = str(dict(getattr(model, "metadata", ())).get("cache_source_residency", "unresolved"))
    resolved_regions = _resolved_cache_regions(key_fmt, value_fmt, _shared, regions)
    return (("attention_tensor_view", {
        "kind": "paged_gqa", "logical_layout": "TND", "kv_storage": "paged",
        "query_shape": query, "output_shape": output,
        "key_tail_shape": key_tail, "value_tail_shape": value_tail,
        "query_dtype": model.activation_dtype, "output_dtype": model.activation_dtype,
        "key_dtype": key_fmt.dtype, "value_dtype": value_fmt.dtype,
        "batch_count": len(active_views) or len(views), "kv_block_size": runtime.kv_block_size,
        "local_kv_heads": kv_heads, "head_dim": width, "v_head_dim": v_width,
        "cache_row_width": kv_heads * (width + v_width), "cache_head_width": width,
        "cache_source_residency": residency,
        **({"cache_representations": resolved_regions} if resolved_regions else {}),
        **({"native_paged_fia": native} if native is not None else {}),
    }),)


def _cache_tensor_view(
    views, kv_heads, width, key_fmt, value_fmt, shared, model, parallel,
    *, v_head_width: int | None = None, region: str = "gqa_kv",
):
    """Declare newly written tokens without inventing cache allocation or slots."""
    del model
    if parallel.cp != 1 or parallel.pcp != 1:
        return ()
    tokens = sum(int(view.logical_q_len) for view in views)
    v_width = width if v_head_width is None else int(v_head_width)
    total = (
        key_fmt.row_bytes(kv_heads * (width + v_width)) if shared
        else key_fmt.row_bytes(kv_heads * width) + value_fmt.row_bytes(kv_heads * v_width)
    )
    plane_extra: dict[str, object] = {}
    if not shared and region == "gqa_kv_key":
        plane_extra = {"plane": "key", "plane_row_bytes": key_fmt.row_bytes(kv_heads * width)}
    elif not shared and region == "gqa_kv_value":
        plane_extra = {"plane": "value", "plane_row_bytes": value_fmt.row_bytes(kv_heads * v_width)}
    return (("cache_tensor_view", {
        "key_shape": (tokens, kv_heads, width), "value_shape": (tokens, kv_heads, v_width),
        "key_dtype": key_fmt.dtype, "value_dtype": value_fmt.dtype,
        "slot_shape": (tokens,), "cache_row_bytes": total,
        "cache_group_size": key_fmt.group_size if shared else 0,
        "cache_scale_dtype": key_fmt.scale_dtype if shared else "mixed",
        **plane_extra,
    }),)


def _decode_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode, *, window=None, cp=None):
    if plan.cache_kind == "none" or mode not in {"decode", "draft"}:
        return ()
    requests = tuple(
        (int(view.logical_q_len), int(view.context) if int(view.logical_q_len) > 0 else 0)
        for view in views
    )
    queries = {query for query, _ in requests}
    contexts = {context for _, context in requests}
    eff_win = int(plan.window or 0) if window is None else int(window)
    eff_cp = int(parallel.cp) if cp is None else int(cp)
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    return (("gqa_decode", {
        "requests": requests,
        **({"uniform_query_tokens": next(iter(queries))} if len(queries) == 1 else {}),
        **({"uniform_context_tokens": next(iter(contexts))} if len(contexts) == 1 else {}),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width, "v_head_dim": v_width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": eff_win, "cp": eff_cp,
    }),)


def _prefill_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode, *, window=None, cp=None):
    """Declare per-request prefill geometry; the explicit recipe validates it."""
    if plan.cache_kind != "none" and mode not in {"prefill", "draft_prefill"}:
        return ()
    eff_win = int(plan.window or 0) if window is None else int(window)
    eff_cp = int(parallel.cp) if cp is None else int(cp)
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    return (("gqa_prefill", {
        "requests": tuple(
            (int(view.logical_q_len), int(view.context) if int(view.logical_q_len) > 0 else 0)
            for view in views
        ),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width, "v_head_dim": v_width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": eff_win, "cp": eff_cp,
    }),)


def _merge_half_open(intervals: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    valid = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    if not valid:
        return ()
    merged: list[tuple[int, int]] = [valid[0]]
    for lo, hi in valid[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _cp_owned_span_for_interval(
    lo: int, hi: int, cp: int, interleave: int, rank: int,
) -> tuple[tuple[tuple[int, ...], ...], str, int]:
    """Build exact O(1) half_open or strided CP span for rank's rows in [lo, hi)."""
    if hi <= lo:
        return (), "half_open", 0
    if cp <= 1:
        return ((lo, hi),), "half_open", hi - lo
    chunk = max(1, int(interleave))
    stride = cp * chunk
    c0 = lo // stride
    c1 = (hi - 1) // stride
    quads: list[tuple[int, int, int, int]] = []
    # Check cycle c0 directly; interior cycles c0+1..c1-1 are always full width.
    first_lo = max(lo, c0 * stride + rank * chunk)
    first_hi = min(hi, c0 * stride + (rank + 1) * chunk)
    if first_hi > first_lo:
        quads.append((first_lo, first_hi - first_lo, stride, 1))
    full_start = c0 + 1
    full_end = c1 - 1
    if full_end >= full_start:
        full_count = full_end - full_start + 1
        quads.append((full_start * stride + rank * chunk, chunk, stride, full_count))
    if c1 > c0:
        last_lo = max(lo, c1 * stride + rank * chunk)
        last_hi = min(hi, c1 * stride + (rank + 1) * chunk)
        if last_hi > last_lo:
            quads.append((last_lo, last_hi - last_lo, stride, 1))
    if not quads:
        return (), "half_open", 0
    # Coalesce adjacent full-width chunks when head or tail happens to be full width.
    coalesced: list[tuple[int, int, int, int]] = []
    for start, width, step, count in quads:
        if (coalesced and width == chunk and coalesced[-1][1] == chunk
                and coalesced[-1][2] == stride
                and coalesced[-1][0] + coalesced[-1][3] * stride == start):
            p_start, p_w, p_s, p_c = coalesced[-1]
            coalesced[-1] = (p_start, p_w, p_s, p_c + count)
        else:
            coalesced.append((start, width, step, count))
    total_rows = sum(width * count for _, width, _, count in coalesced)
    if all(count == 1 for _, _, _, count in coalesced):
        return tuple((start, start + width) for start, width, _, _ in coalesced), "half_open", total_rows
    normalized = tuple(
        (start, width, width if count == 1 else step, count)
        for start, width, step, count in coalesced
    )
    return normalized, "start_width_stride_count", total_rows


def _span_for_intervals(
    intervals: tuple[tuple[int, int], ...], cp: int, interleave: int, rank: int,
) -> tuple[tuple[tuple[int, ...], ...], str, int]:
    """Project merged historical intervals onto physical rank-local rows and span."""
    if not intervals:
        return (), "half_open", 0
    if cp <= 1:
        return intervals, "half_open", sum(hi - lo for lo, hi in intervals)
    parts: list[tuple[int, ...]] = []
    strided = False
    total_rows = 0
    for lo, hi in intervals:
        seg, fmt, rows = _cp_owned_span_for_interval(lo, hi, cp, interleave, rank)
        if rows <= 0:
            continue
        total_rows += rows
        if fmt == "start_width_stride_count":
            strided = True
        parts.append((fmt, seg))
    if total_rows <= 0:
        return (), "half_open", 0
    if not strided:
        flat_pairs = tuple(item for _, seg in parts for item in seg)
        return flat_pairs, "half_open", total_rows
    flat_quads: list[tuple[int, int, int, int]] = []
    for fmt, seg in parts:
        if fmt == "start_width_stride_count":
            flat_quads.extend(seg)
        else:
            for s_lo, s_hi in seg:
                w = s_hi - s_lo
                flat_quads.append((s_lo, w, w, 1))
    return tuple(flat_quads), "start_width_stride_count", total_rows


def _historical_interval(
    view, *, window: int, noncausal: bool, use_global_q: bool = False,
) -> tuple[int, int]:
    """Historical KV read span strictly inside [0, context); excludes current writes."""
    context = int(getattr(view, "context", 0) or 0)
    if context <= 0:
        return 0, 0
    if window <= 0:
        return 0, context
    if noncausal:
        return max(0, context - window), context
    if use_global_q:
        min_start = 0
    else:
        ranges = getattr(view, "query_ranges", ()) or ((0, int(getattr(view, "q_len", 0) or 0)),)
        min_start = min((begin for begin, end in ranges if end > begin), default=0)
    lo = max(0, context + min_start + 1 - window)
    return (lo, context) if context > lo else (0, 0)


def _row_stride_int(row_bytes: float) -> int:
    return int(round(row_bytes)) if abs(row_bytes - round(row_bytes)) < 1e-6 else 0


def _local_byte_span_for_intervals(
    intervals: tuple[tuple[int, int], ...],
    cp: int,
    interleave: int,
    cp_shard: int,
    row_stride: int,
) -> tuple[tuple[tuple[int, int], ...], str]:
    """Translate logical intervals into rank-local contiguous storage byte spans."""
    if row_stride <= 0 or not intervals:
        return (), "half_open"
    if cp <= 1:
        return tuple((lo * row_stride, hi * row_stride) for lo, hi in intervals), "half_open"
    chunk = max(1, int(interleave))
    local_slots: list[tuple[int, int]] = []
    for lo, hi in intervals:
        if hi <= lo:
            continue
        seg_rows = owned_positions(lo, hi - 1, cp, chunk, cp_shard)
        if seg_rows <= 0:
            continue
        local_lo = owned_positions(0, lo - 1, cp, chunk, cp_shard) if lo > 0 else 0
        local_slots.append((local_lo, local_lo + seg_rows))
    merged_slots = _merge_half_open(local_slots)
    return tuple((lo * row_stride, hi * row_stride) for lo, hi in merged_slots), "half_open"


def _build_plane_access(
    *, op_id: str, request: str, stage: int, layer: int, storage: str,
    span: tuple[tuple[int, ...], ...], span_format: str, rows: int,
    row_bytes: float, cp: int, rank: int | None, scope: str,
    byte_span: tuple[tuple[int, ...], ...] = (),
    byte_span_format: str = "half_open",
    physical_key: str = "",
) -> dict[str, object]:
    row_stride = _row_stride_int(row_bytes)
    record: dict[str, object] = {
        "occurrence": op_id,
        "producer": op_id,
        "request": request,
        "stage": int(stage),
        "layer": int(layer),
        "direction": "read",
        "position": "cache",
        "output": storage,
        "storage": storage,
        "span": span,
        "span_format": span_format,
        "span_unit": "rows",
        "rows": int(rows),
        "persistent": True,
        "temporary": False,
        "scope": scope,
        "bytes": float(rows) * float(row_bytes),
        "source": op_id,
    }
    if row_stride > 0:
        record["row_stride"] = row_stride
    if byte_span:
        record["byte_span"] = byte_span
        record["byte_span_format"] = byte_span_format
    if cp > 1 and rank is not None:
        record["rank"] = int(rank)
    if physical_key:
        record["physical_key"] = physical_key
    return record


def _emit_interval_accesses(
    records: list[dict[str, object]],
    *,
    intervals: tuple[tuple[int, int], ...],
    planes: tuple[tuple[str, float], ...],
    op_id: str,
    request: str,
    stage: int,
    layer: int,
    cp: int,
    cp_interleave: int,
    cp_shard: int,
    record_rank: int | None,
    scope: str,
    physical_key: str = "",
) -> None:
    span, fmt, rows = _span_for_intervals(intervals, cp, cp_interleave, cp_shard)
    if rows <= 0:
        return
    for storage, row_bytes in planes:
        byte_span, byte_fmt = _local_byte_span_for_intervals(
            intervals, cp, cp_interleave, cp_shard, _row_stride_int(row_bytes),
        )
        records.append(_build_plane_access(
            op_id=op_id, request=request, stage=stage, layer=layer, storage=storage,
            span=span, span_format=fmt, rows=rows, row_bytes=row_bytes,
            cp=cp, rank=record_rank, scope=scope,
            byte_span=byte_span, byte_span_format=byte_fmt,
            physical_key=physical_key,
        ))


def _sparse_selected_rows(
    view,
    *,
    view_idx: int,
    owned_hist: int,
    selected_topk: int,
    selected_rows_by_view: tuple[int, ...] | None,
    use_global_q: bool,
) -> int:
    if owned_hist <= 0 or selected_topk <= 0:
        return 0
    if selected_rows_by_view is not None and view_idx < len(selected_rows_by_view):
        return min(owned_hist, max(0, int(selected_rows_by_view[view_idx])))
    if use_global_q:
        sel_q = int(getattr(view, "global_q_len", 0) or getattr(view, "q_len", 0) or 0)
    else:
        ranges = getattr(view, "query_ranges", ()) or ((0, int(getattr(view, "q_len", 0) or 0)),)
        sel_q = sum(end - begin for begin, end in ranges if end > begin)
    return min(owned_hist, max(1, sel_q) * int(selected_topk))


def gqa_kv_memory_accesses(
    views,
    *,
    op_id: str,
    stage: int,
    layer: int,
    kv_heads: int,
    width: int,
    v_width: int,
    key_fmt,
    value_fmt,
    shared_format: bool,
    mode: str,
    window: int = 0,
    cp: int = 1,
    cp_interleave: int = 1,
    cp_rank: int | None = None,
    physical_rank: int | None = None,
    block_size: int = 1,
    prefix_cache_mode: str = "private",
    cacheless: bool = False,
    noncausal: bool = False,
    allow_shared_prefix: bool = True,
    use_global_q: bool = False,
    selected_topk: int = 0,
    selected_rows_by_view: tuple[int, ...] | None = None,
) -> tuple[tuple[str, tuple[dict[str, object], ...]], ...]:
    """Emit physical per-request historical GQA KV read spans for offload binding.

    Covers packed/quantized & asymmetric K/V, sliding windows, CP rank ownership,
    sparse Top-K selection bounds and shared-prefix deduplication while excluding
    padding views and current-token ``cache_store`` writes ``[context, context + q)``.
    """
    if cacheless or not views or kv_heads <= 0 or width <= 0 or v_width <= 0:
        return ()
    if shared_format:
        planes = (("gqa_kv", float(key_fmt.row_bytes(kv_heads * (width + v_width)))),)
    else:
        planes = (
            ("gqa_kv_key", float(key_fmt.row_bytes(kv_heads * width))),
            ("gqa_kv_value", float(value_fmt.row_bytes(kv_heads * v_width))),
        )
    planes = tuple((name, rb) for name, rb in planes if rb > 0.0)
    if not planes:
        return ()
    scope = "draft" if mode in {"draft", "draft_prefill"} else "target"
    cp_shard = int(cp_rank or 0) if cp > 1 else 0
    record_rank = (
        int(physical_rank) if physical_rank is not None else cp_shard
    ) if cp > 1 else None
    eff_block = max(1, int(block_size or 1))
    is_sparse = int(selected_topk or 0) > 0
    use_shared_prefix = (
        not is_sparse
        and allow_shared_prefix
        and str(prefix_cache_mode or "private").strip().lower() == "shared"
    )
    active_entries: list[tuple[int, object, int, int, str, int]] = []
    for view_idx, view in enumerate(views):
        if getattr(view, "padding", False):
            continue
        hist_lo, hist_hi = _historical_interval(
            view, window=window, noncausal=noncausal, use_global_q=use_global_q,
        )
        if hist_hi <= hist_lo:
            continue
        context = int(getattr(view, "context", 0) or 0)
        raw_prefix = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), context))
        aligned = (raw_prefix // eff_block) * eff_block if use_shared_prefix else 0
        pkey = (
            canonical_prefix_cache_key("shared", getattr(view, "prefix_cache_key", ""))
            if (use_shared_prefix and aligned > hist_lo)
            else ""
        )
        active_entries.append((view_idx, view, hist_lo, hist_hi, pkey, aligned))
    if not active_entries:
        return (("memory_accesses", ()),)
    shared_counts = Counter(pkey for _, _, _, _, pkey, _ in active_entries if pkey)
    shared_groups: dict[str, tuple[str, list[tuple[int, int]]]] = {}
    records: list[dict[str, object]] = []
    for view_idx, view, hist_lo, hist_hi, pkey, aligned in active_entries:
        req = str(view.request_id)
        if is_sparse:
            _, _, owned_hist = _span_for_intervals(((hist_lo, hist_hi),), cp, cp_interleave, cp_shard)
            rows = _sparse_selected_rows(
                view, view_idx=view_idx, owned_hist=owned_hist,
                selected_topk=int(selected_topk),
                selected_rows_by_view=selected_rows_by_view,
                use_global_q=use_global_q,
            )
            if rows > 0:
                for storage, row_bytes in planes:
                    records.append(_build_plane_access(
                        op_id=op_id, request=req, stage=stage, layer=layer, storage=storage,
                        span=(), span_format="half_open", rows=rows, row_bytes=row_bytes,
                        cp=cp, rank=record_rank, scope=scope,
                    ))
            continue
        if pkey and shared_counts[pkey] > 1:
            sh_hi = min(hist_hi, aligned)
            if sh_hi > hist_lo:
                if pkey not in shared_groups:
                    shared_groups[pkey] = (req, [])
                shared_groups[pkey][1].append((hist_lo, sh_hi))
            priv_lo = max(hist_lo, aligned)
            priv_intervals = ((priv_lo, hist_hi),) if hist_hi > priv_lo else ()
        else:
            priv_intervals = ((hist_lo, hist_hi),)
        if priv_intervals:
            _emit_interval_accesses(
                records, intervals=priv_intervals, planes=planes, op_id=op_id,
                request=req, stage=stage, layer=layer, cp=cp,
                cp_interleave=cp_interleave, cp_shard=cp_shard,
                record_rank=record_rank, scope=scope,
            )
    for pkey, (owner_req, raw_intervals) in shared_groups.items():
        merged = _merge_half_open(raw_intervals)
        _emit_interval_accesses(
            records, intervals=merged, planes=planes, op_id=op_id,
            request=owner_req, stage=stage, layer=layer, cp=cp,
            cp_interleave=cp_interleave, cp_shard=cp_shard,
            record_rank=record_rank, scope=scope, physical_key=f"prefix:{pkey}",
        )
    return (("memory_accesses", tuple(records)),)


class _CacheProducer:
    """Minimal existing-builder view for T07 cache representation producers."""

    def __init__(self, ops, *, prefix, stage, layer, requests, phase, model, plan=None):
        self.ops, self.prefix, self.model, self.plan = ops, prefix, model, plan
        self.common = {"stage": stage, "layer": layer, "requests": requests, "phase": phase}

    def add(self, op):
        self.ops.append(op)


def _store_cache(
    ops, *, prefix, deps, rows, width, region, stage, layer, requests, phase,
    model, views, kv_heads, head_width, parallel, quant, key_fmt, value_fmt,
    shared_format, name="cache", v_head_width: int | None = None, plan=None,
):
    """Use T07 store: BF16 source and packed output are separate ledgers."""
    del quant
    producer = _CacheProducer(
        ops, prefix=prefix, stage=stage, layer=layer,
        requests=requests, phase=phase, model=model, plan=plan,
    )
    store_id = representation_store(producer, name, deps, rows, width, region)
    view = _cache_tensor_view(
        views, kv_heads, head_width, key_fmt, value_fmt, shared_format, model, parallel,
        v_head_width=v_head_width, region=region,
    )
    if ops:
        op = ops[-1]
        metadata = dict(op.metadata)
        metadata["cache_kind"] = "gqa"
        if view:
            for key, value in view:
                metadata[key] = value
        ops[-1] = replace(op, metadata=tuple(sorted(metadata.items())))
    return store_id
