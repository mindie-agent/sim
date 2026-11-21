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

from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.tensors import TensorLayout

from .attention_kv_access import paged_fia_plan
from mindie_sim.prediction.cache_representation import store as representation_store
from mindie_sim.prediction.memory_traffic import default_hierarchy
from mindie_sim.prediction.structure_groups import declare_group

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


def _memory_hierarchy(machine):
    """Declared device hierarchy for traffic planning, or the frontend default."""
    if machine is not None and getattr(machine, "dies", ()):
        hierarchy = getattr(machine.dies[0].physical_mechanisms, "memory_hierarchy", None)
        if hierarchy is not None and hierarchy.l2_capacity_bytes > 0:
            return hierarchy
    return default_hierarchy()


def _qkv_views(tokens, heads, kv_heads, width, has_gate, k_eq_v=False):
    """Q and gate alternate by head; all slices retain the projected row pitch."""
    query, kv = heads * width, kv_heads * width
    gate = query if has_gate else 0
    pitch = query + gate + (1 if k_eq_v else 2) * kv
    storage = (tokens, pitch)
    views = {
        "query": TensorLayout(
            (tokens, heads, width), (pitch, width * (2 if has_gate else 1), 1),
            storage_shape=storage),
        "key": TensorLayout(
            (tokens, kv_heads, width), (pitch, width, 1),
            offset=query + gate, storage_shape=storage),
        "value": TensorLayout(
            (tokens, kv_heads, width), (pitch, width, 1),
            offset=query + gate if k_eq_v else query + gate + kv, storage_shape=storage),
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


def _gqa_formats(model):
    """Resolve shared or independent K/V cache formats once."""
    regions = dict(getattr(model, "metadata", ())).get("cache_representations", {})
    if "gqa_kv_key" in regions or "gqa_kv_value" in regions:
        return cache_format(model, "gqa_kv_key"), cache_format(model, "gqa_kv_value"), regions, False
    return cache_format(model, "gqa_kv"), cache_format(model, "gqa_kv"), regions, True


def _contiguous_tensor_view(views, heads, kv_heads, width, model, quant, parallel, *, cacheless=False):
    """Fresh prefill has explicit TND operands, before any paged-cache reuse."""
    if parallel.cp != 1 or parallel.pcp != 1 or any(
        (not cacheless and view.phase != "prefill") or view.context for view in views
    ):
        return ()
    query = (sum(view.q_len for view in views), heads, width)
    kv = (sum(view.kv_len for view in views), kv_heads, width)
    return (("attention_tensor_view", {
        "query_shape": query, "key_shape": kv, "value_shape": kv, "output_shape": query,
        "query_dtype": model.activation_dtype, "key_dtype": quant.kv_dtype,
        "value_dtype": quant.kv_dtype, "output_dtype": model.activation_dtype,
        "logical_layout": "TND", "kv_storage": "contiguous",
    }),)


def _attention_tensor_view(
    views, heads, kv_heads, width, model, quant, parallel, policy, runtime,
    causal=True, cacheless=False,
):
    """A paged ABI requires an explicit storage declaration, not a query length."""
    if cacheless or policy.kv_storage != "paged":
        return _contiguous_tensor_view(
            views, heads, kv_heads, width, model, quant, parallel, cacheless=cacheless)
    if runtime is None or parallel.cp != 1 or parallel.pcp != 1:
        return ()
    query = (sum(view.q_len for view in views), heads, width)
    tail = (runtime.kv_block_size, kv_heads * width)
    key_fmt, value_fmt, regions, _shared = _gqa_formats(model)
    native = None
    page_size = int(getattr(runtime, "kv_block_size", 0) or 0)
    if page_size > 0 and all(view.q_len > 0 for view in views):
        native = paged_fia_plan(
            requests=tuple((view.q_len, view.context + view.q_len) for view in views),
            query_lengths=tuple(view.q_len for view in views),
            kv_lengths=tuple(view.context + view.q_len for view in views),
            page_size=page_size, query_heads=heads, kv_heads=kv_heads, head_dim=width,
            query_dtype=model.activation_dtype, output_dtype=model.activation_dtype,
            key_format=key_fmt, value_format=value_fmt,
            phase=getattr(views[0], "phase", "decode"),
            mask="causal" if causal else "none", return_lse=False, output_prepare=None,
        )
    residency = str(dict(getattr(model, "metadata", ())).get("cache_source_residency", "unresolved"))
    return (("attention_tensor_view", {
        "kind": "paged_gqa", "logical_layout": "TND", "kv_storage": "paged",
        "query_shape": query, "output_shape": query,
        "key_tail_shape": tail, "value_tail_shape": tail,
        "query_dtype": model.activation_dtype, "output_dtype": model.activation_dtype,
        "key_dtype": key_fmt.dtype, "value_dtype": value_fmt.dtype,
        "batch_count": len(views), "kv_block_size": runtime.kv_block_size,
        "local_kv_heads": kv_heads, "head_dim": width,
        "cache_row_width": 2 * kv_heads * width, "cache_head_width": width,
        "cache_source_residency": residency,
        **({"cache_representations": regions} if regions else {}),
        **({"native_paged_fia": native} if native is not None else {}),
    }),)


def _cache_tensor_view(views, kv_heads, width, key_fmt, value_fmt, shared, model, parallel):
    """Declare newly written tokens without inventing cache allocation or slots."""
    del model
    if parallel.cp != 1 or parallel.pcp != 1:
        return ()
    tokens = sum(view.q_len for view in views)
    total = (
        key_fmt.row_bytes(2 * kv_heads * width) if shared
        else key_fmt.row_bytes(kv_heads * width) + value_fmt.row_bytes(kv_heads * width)
    )
    return (("cache_tensor_view", {
        "key_shape": (tokens, kv_heads, width), "value_shape": (tokens, kv_heads, width),
        "key_dtype": key_fmt.dtype, "value_dtype": value_fmt.dtype,
        "slot_shape": (tokens,), "cache_row_bytes": total,
        "cache_group_size": key_fmt.group_size if shared else 0,
        "cache_scale_dtype": key_fmt.scale_dtype if shared else "mixed",
    }),)


def _decode_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode, *, window=None, cp=None):
    if plan.cache_kind == "none" or mode not in {"decode", "draft"}:
        return ()
    requests = tuple((view.q_len, view.context) for view in views)
    queries = {query for query, _ in requests}
    contexts = {context for _, context in requests}
    eff_win = int(plan.window or 0) if window is None else int(window)
    eff_cp = int(parallel.cp) if cp is None else int(cp)
    return (("gqa_decode", {
        "requests": requests,
        **({"uniform_query_tokens": next(iter(queries))} if len(queries) == 1 else {}),
        **({"uniform_context_tokens": next(iter(contexts))} if len(contexts) == 1 else {}),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": eff_win, "cp": eff_cp,
    }),)


def _prefill_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode, *, window=None, cp=None):
    """Declare per-request prefill geometry; the explicit recipe validates it."""
    if plan.cache_kind != "none" and mode not in {"prefill", "draft_prefill"}:
        return ()
    eff_win = int(plan.window or 0) if window is None else int(window)
    eff_cp = int(parallel.cp) if cp is None else int(cp)
    return (("gqa_prefill", {
        "requests": tuple((view.q_len, view.context) for view in views),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": eff_win, "cp": eff_cp,
    }),)


class _CacheProducer:
    """Minimal existing-builder view for T07 cache representation producers."""

    def __init__(self, ops, *, prefix, stage, layer, requests, phase, model):
        self.ops, self.prefix, self.model = ops, prefix, model
        self.common = {"stage": stage, "layer": layer, "requests": requests, "phase": phase}

    def add(self, op):
        self.ops.append(op)


def _store_cache(
    ops, *, prefix, deps, rows, width, region, stage, layer, requests, phase,
    model, views, kv_heads, head_width, parallel, quant, key_fmt, value_fmt,
    shared_format, name="cache",
):
    """Use T07 store: BF16 source and packed output are separate ledgers."""
    del quant
    producer = _CacheProducer(
        ops, prefix=prefix, stage=stage, layer=layer,
        requests=requests, phase=phase, model=model,
    )
    store_id = representation_store(producer, name, deps, rows, width, region)
    view = _cache_tensor_view(
        views, kv_heads, head_width, key_fmt, value_fmt, shared_format, model, parallel
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
