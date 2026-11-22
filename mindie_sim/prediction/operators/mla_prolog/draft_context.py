"""Independent context-KV preparation for external block drafters.

[Role] Separate target-feature context rows from proposal query rows.
[Invariants] Each draft layer computes and stores its own BF16 context KV.
  Context projection is not reused as the query embedding.  The layer body is
  one ``aclnnMlaPrologV3WeightNz`` call followed by one
  ``aclnnScatterPaKvCache``; DCP writes only ``ceil(rows / cp)`` rank-owned
  rows of ``1,152 B`` each, never an unsharded copy of every row.
[Timeouts & Locks] O(draft layers); no locks or external work.
[Side-Effect Policy] Appends only to the caller-owned operator list.
"""
from dataclasses import replace
from math import ceil

from mindie_sim.catalog.ops import cache_store_op, gemm_op, vector_op, barrier_op
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.physical_recipe_base import Phase, io_weight, tiled_phased_program

NORM_OPS = 5
ROPE_OPS = 6
PROLOG_RECIPE = "draft_context_mla_prolog_v3"
SCATTER_RECIPE = "draft_context_kv_scatter"
PROLOG_NATIVE_KIND = "aclnnMlaPrologV3WeightNz"
PROLOG_V4_NATIVE_KIND = "aclnnMlaPrologV4WeightNz"
SCATTER_NATIVE_KIND = "aclnnScatterPaKvCache"
KV_ROW_BYTES_PER_ELEMENT = int(dtype_bytes("bf16"))
PROLOG_CUBE_SHARE = 1.0
PROLOG_VECTOR_SHARE = 1.0
PROLOG_MEMORY_SHARE = 1.0
UNIT_CONTEXT_PARALLEL = 1
TWO_CACHE_COMPONENTS = 2
ASCEND_PROLOG_ARCHS = frozenset({"", "A2", "A3", "A5"})
CONTIGUOUS_DRAFT_LAYOUTS = frozenset({
    "split_contiguous", "contiguous_components", "contiguous_tuple",
    "component", "two_contiguous", "contiguous",
})
STRIDED_DRAFT_LAYOUTS = frozenset({
    "strided", "token_fused", "combined", "paired_sfa",
})


def draft_cache_requires_prolog_v3(model, runtime=None, machine=None) -> bool:
    """Reference ``_cache_requires_prolog_v3_writer`` on the draft MLA cache."""
    model_meta = dict(getattr(model, "metadata", ()) or ())
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    arch = str(getattr(machine, "arch_family", "")
               or rt_meta.get("arch_family") or "").strip().upper()
    if arch not in ASCEND_PROLOG_ARCHS:
        return False
    layout = str(rt_meta.get("draft_cache_layout")
                 or model_meta.get("draft_cache_layout") or "").strip().lower()
    if (layout in CONTIGUOUS_DRAFT_LAYOUTS
            or bool(rt_meta.get("draft_cache_contiguous")
                    or model_meta.get("draft_cache_contiguous"))):
        return False
    if layout in STRIDED_DRAFT_LAYOUTS:
        return True
    for key in ("draft_cache_representations", "cache_representations"):
        reps = rt_meta.get(key) if key in rt_meta else model_meta.get(key)
        if not isinstance(reps, dict):
            continue
        if (({"mla_nope", "mla_rope"} <= set(reps)
                or {"nope", "rope"} <= set(reps)
                or (key == "draft_cache_representations"
                    and len(reps) == TWO_CACHE_COMPONENTS
                    and "mla_kv" not in reps))
                and all(not isinstance(v, dict) or v.get("contiguous", True)
                        for v in reps.values())):
            return False
        if key == "draft_cache_representations" and isinstance(reps.get("mla_kv"), dict):
            if bool(reps["mla_kv"].get("contiguous_components", False)):
                return False
    allocator = str(model_meta.get("declared_cache_allocator")
                    or model_meta.get("cache_allocator") or "").strip().lower()
    return allocator == "paired_sfa"
SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "nightly-ci-main-a5-openeuler-cann9.2.1-20261111",
    "path": "vllm_ascend/models/kimi_k3/kimi_k3_dspark.py",
    "symbol": "exec_context_kv_prolog_v3",
    "lines": "211-240",
    "companion": "vllm_ascend/attention/mla_v1.py:2295-2356,2556-2594",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def recipes():
    from mindie_sim.prediction.mechanism import PlannedRecipe
    return {PROLOG_RECIPE: PlannedRecipe(build_prolog),
            SCATTER_RECIPE: PlannedRecipe(build_scatter)}


def _prolog_work(op):
    return Work({op.dtype: op.flops} if op.flops else {},
                {"hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
                 "mte3": op.mte3_bytes, "vector": op.vector_ops})


def _prolog_program(context):
    read_share = io_weight(context, "hbm")
    write_share = io_weight(context, "hbm_write")
    return tiled_phased_program(context, (
        Phase("context_load", {"hbm": read_share, "mte2": PROLOG_MEMORY_SHARE}),
        Phase("qkv_a", {"cube": PROLOG_CUBE_SHARE}, after=("context_load",)),
        Phase("kv_norm_rope", {"vector": PROLOG_VECTOR_SHARE}, after=("qkv_a",)),
        Phase("publish", {"hbm": write_share, "mte3": PROLOG_MEMORY_SHARE},
              after=("kv_norm_rope",)),
    ))


def _scatter_program(context):
    read_share = io_weight(context, "hbm")
    write_share = io_weight(context, "hbm_write")
    return tiled_phased_program(context, (
        Phase("gather_rank_rows", {"hbm": read_share, "mte2": PROLOG_MEMORY_SHARE}),
        Phase("paged_scatter", {"hbm": write_share, "mte3": PROLOG_MEMORY_SHARE},
              after=("gather_rank_rows",)),
    ))


def build_prolog(op, geometry):
    from mindie_sim.prediction.mechanism import OperatorPlan
    del geometry
    return OperatorPlan(_prolog_work(op), _prolog_program,
                        metadata={"physical_class": "DraftContextPrologV3",
                                  "stage_basis": "declared_phases_partition_member_work"})


def build_scatter(op, geometry):
    from mindie_sim.prediction.mechanism import OperatorPlan
    del geometry
    return OperatorPlan(_prolog_work(op), _scatter_program,
                        metadata={"physical_class": "DraftContextKvScatter",
                                  "stage_basis": "declared_phases_partition_member_work"})


def owned_rows(rows, parallel):
    """Rank-owned DCP row shard: ``ceil(rows / max(1, cp))``."""
    cp = max(UNIT_CONTEXT_PARALLEL,
             int(getattr(parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL))
    return int(ceil(int(rows) / cp))


def owned_scatter_rows(rows, rows_by_view, views, contexts, parallel, block):
    """Worst-rank rows one draft layer stores under block ownership.

    A position block is owned atomically by one context rank, so the writer
    charges every row in that block to its owner.  ``ceil(rows / cp)`` would
    split one block across ranks and undercount the owner's write.
    """
    if rows_by_view and views and contexts:
        from mindie_sim.prediction.operators.mla.generic_latent_attention import owned_store_rows
        return int(owned_store_rows(rows_by_view, views, parallel, block,
                                    contexts=contexts))
    return owned_rows(rows, parallel)


def prepare(ops, *, projection, model, draft, rows, query_rows,
            prefix, stage, requests, parallel=None, rows_by_view=None,
            views=None, contexts=None, block=UNIT_CONTEXT_PARALLEL,
            use_prolog_v3=True):
    proj_mode = str(getattr(draft, "context_projection_mode", "") or "none").strip()
    if proj_mode == "shared_compressed_kv":
        common = dict(stage=stage, layer=-1, requests=requests, phase="draft")
        embedding = vector_op(
            op_id=prefix + "query_embedding", deps=(), elements=query_rows * draft.hidden_size,
            ops_per_element=0, dtype="bf16", family="embedding",
            recipe="draft_query_embedding", **common)
        ready = barrier_op(
            op_id=prefix + "context_query_ready", deps=(projection.op_id, embedding.op_id),
            recipe="draft_context_query_ready", shape=(rows, query_rows), **common)
        ops.extend((embedding, ready))
        return ready.op_id
    if getattr(draft, "attention_kind", "mla") == "gqa" or proj_mode in {"fused_kv_bank", "per_layer_qkv"}:
        from .draft_context_gqa import prepare_gqa_context
        return prepare_gqa_context(
            ops,
            projection=projection,
            model=model,
            draft=draft,
            rows=rows,
            query_rows=query_rows,
            prefix=prefix,
            stage=stage,
            requests=requests,
            parallel=parallel,
            owned_scatter_rows_fn=owned_scatter_rows,
            rows_by_view=rows_by_view,
            views=views,
            contexts=contexts,
            block=block,
        )
    common = dict(stage=stage, layer=-1, requests=requests, phase="draft")
    norm = vector_op(
        op_id=prefix + "context_norm", deps=(projection.op_id,),
        elements=rows * draft.hidden_size, ops_per_element=NORM_OPS,
        dtype="bf16", family="rmsnorm", recipe="draft_context_norm", **common)
    ops.append(norm)
    kv_width = draft.kv_lora_rank + draft.qk_rope_head_dim
    row_bytes = kv_width * KV_ROW_BYTES_PER_ELEMENT
    rank_rows = owned_scatter_rows(
        rows, rows_by_view, views, contexts, parallel, block)
    activation_bytes = dtype_bytes(draft.activation_dtype)
    stores = []
    for layer in range(draft.layers):
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
        qkv_width = (draft.q_lora_rank + draft.kv_lora_rank
                     + draft.qk_rope_head_dim)
        if use_prolog_v3:
            mla_format = model.quant.format_for("mla")
            is_mxfp8 = mla_format == "fp8_e4m3"
            native_kind = PROLOG_V4_NATIVE_KIND if is_mxfp8 else PROLOG_NATIVE_KIND
            prolog_dtype = mla_format if is_mxfp8 else draft.activation_dtype
            do_rope = bool(draft.qk_rope_head_dim > 0)
            weight_bytes = model.quant.matrix_bytes(qkv_width, draft.hidden_size, "mla")
            prolog = OpSpec(
                op_id=name + ".mla_prolog_v3", deps=(norm.op_id,),
                recipe=PROLOG_RECIPE, family="mla",
                semantic_shape=(rows, qkv_width), local_shape=(rows, qkv_width),
                padded_shape=(rows, qkv_width), launch_shape=(rows, qkv_width),
                flops=2 * rows * qkv_width * draft.hidden_size,
                vector_ops=rows * (NORM_OPS * (draft.q_lora_rank + draft.kv_lora_rank)
                                   + (ROPE_OPS * draft.qk_rope_head_dim if do_rope else 0)),
                hbm_bytes=(rows * draft.hidden_size * activation_bytes + weight_bytes
                           + rows * qkv_width * activation_bytes),
                mte2_bytes=rows * draft.hidden_size * activation_bytes + weight_bytes,
                mte3_bytes=rows * qkv_width * activation_bytes,
                dtype=prolog_dtype,
                metadata=(("native_kind", native_kind),
                          ("native_source", SOURCE),
                          ("do_rope", do_rope),
                          ("fused_members", ("qkv_a", "kv_norm", "kv_rope") if do_rope else ("qkv_a", "kv_norm")),
                          ("context_rows", rows), ("query_rows", query_rows),
                          ("data_parallel_context", int(getattr(parallel, "cp", 1) or 1)),
                          ("rank_owned_rows", rank_rows),
                          ("kv_row_bytes", row_bytes),
                          ("hidden_read_once", True),
                          ("timing_boundary", "expanded_semantic_operator")),
                **layer_common)
            ops.append(prolog)
            kv_ready = prolog.op_id
        else:
            qkv_proj = gemm_op(
                op_id=name + ".qkv_a", deps=(norm.op_id,),
                m=rows, n=qkv_width, k=draft.hidden_size, quant=model.quant,
                category="mla", family="gemm", recipe="draft_context_qkv_a",
                **layer_common)
            kv_norm = vector_op(
                op_id=name + ".kv_norm", deps=(qkv_proj.op_id,),
                elements=rows * draft.kv_lora_rank, ops_per_element=NORM_OPS,
                dtype=draft.activation_dtype, family="rmsnorm",
                recipe="draft_context_kv_norm", **layer_common)
            ops.extend((qkv_proj, kv_norm))
            kv_ready = kv_norm.op_id
            if draft.qk_rope_head_dim:
                kv_rope = vector_op(
                    op_id=name + ".kv_rope", deps=(kv_norm.op_id,),
                    elements=rows * draft.qk_rope_head_dim,
                    ops_per_element=ROPE_OPS, dtype=draft.activation_dtype,
                    family="rotary", recipe="draft_context_kv_rope",
                    **layer_common)
                ops.append(kv_rope)
                kv_ready = kv_rope.op_id
        stored = cache_store_op(
            op_id=name + ".kv_scatter", deps=(kv_ready,), rows=rank_rows,
            row_bytes=row_bytes, dtype="bf16", kind="draft_context",
            recipe=SCATTER_RECIPE, **layer_common)
        stored = replace(stored, metadata=stored.metadata + (
            ("native_kind", SCATTER_NATIVE_KIND),
            ("cache_epilog", {
                "kind": SCATTER_NATIVE_KIND,
                "context_rows": rows,
                "rank_owned_rows": rank_rows,
                "kv_row_bytes": row_bytes,
                "row_elements": kv_width,
            }),
            ("dcp_row_shard", rank_rows),
            ("dcp_degree", int(getattr(parallel, "cp", 1) or 1))))
        ops.append(stored)
        stores.append(stored.op_id)
    embedding = vector_op(
        op_id=prefix + "query_embedding", deps=(), elements=query_rows * draft.hidden_size,
        ops_per_element=0, dtype="bf16", family="embedding",
        recipe="draft_query_embedding", **common)
    ready = barrier_op(
        op_id=prefix + "context_query_ready", deps=(*stores, embedding.op_id),
        recipe="draft_context_query_ready", shape=(rows, query_rows), **common)
    ops.extend((embedding, ready))
    return ready.op_id
