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

from mindie_sim.catalog.ops import cache_store_op, vector_op, barrier_op
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .mechanism import Work
from .physical_recipe_base import Phase, io_weight, tiled_phased_program

NORM_OPS = 5
ROPE_OPS = 6
PROLOG_RECIPE = "draft_context_mla_prolog_v3"
SCATTER_RECIPE = "draft_context_kv_scatter"
PROLOG_NATIVE_KIND = "aclnnMlaPrologV3WeightNz"
SCATTER_NATIVE_KIND = "aclnnScatterPaKvCache"
KV_ROW_BYTES_PER_ELEMENT = int(dtype_bytes("bf16"))
PROLOG_CUBE_SHARE = 1.0
PROLOG_VECTOR_SHARE = 1.0
PROLOG_MEMORY_SHARE = 1.0
UNIT_CONTEXT_PARALLEL = 1
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
    from .mechanism import PlannedRecipe
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
    from .mechanism import OperatorPlan
    del geometry
    return OperatorPlan(_prolog_work(op), _prolog_program,
                        metadata={"physical_class": "DraftContextPrologV3",
                                  "stage_basis": "declared_phases_partition_member_work"})


def build_scatter(op, geometry):
    from .mechanism import OperatorPlan
    del geometry
    return OperatorPlan(_prolog_work(op), _scatter_program,
                        metadata={"physical_class": "DraftContextKvScatter",
                                  "stage_basis": "declared_phases_partition_member_work"})


def owned_rows(rows, parallel):
    """Rank-owned DCP row shard: ``ceil(rows / max(1, cp))``."""
    cp = max(UNIT_CONTEXT_PARALLEL,
             int(getattr(parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL))
    return int(ceil(int(rows) / cp))


def prepare(ops, *, projection, model, draft, rows, query_rows,
            prefix, stage, requests, parallel=None):
    common = dict(stage=stage, layer=-1, requests=requests, phase="draft")
    norm = vector_op(
        op_id=prefix + "context_norm", deps=(projection.op_id,),
        elements=rows * draft.hidden_size, ops_per_element=NORM_OPS,
        dtype="bf16", family="rmsnorm", recipe="draft_context_norm", **common)
    ops.append(norm)
    kv_width = draft.kv_lora_rank + draft.qk_rope_head_dim
    row_bytes = kv_width * KV_ROW_BYTES_PER_ELEMENT
    rank_rows = owned_rows(rows, parallel)
    activation_bytes = dtype_bytes(draft.activation_dtype)
    stores = []
    for layer in range(draft.layers):
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
        qkv_width = (draft.q_lora_rank + draft.kv_lora_rank
                     + draft.qk_rope_head_dim)
        weight_bytes = model.quant.matrix_bytes(qkv_width, draft.hidden_size, "mla")
        prolog = OpSpec(
            op_id=name + ".mla_prolog_v3", deps=(norm.op_id,),
            recipe=PROLOG_RECIPE, family="mla",
            semantic_shape=(rows, qkv_width), local_shape=(rows, qkv_width),
            padded_shape=(rows, qkv_width), launch_shape=(rows, qkv_width),
            flops=2 * rows * qkv_width * draft.hidden_size,
            vector_ops=rows * (NORM_OPS * (draft.q_lora_rank + draft.kv_lora_rank)
                               + ROPE_OPS * draft.qk_rope_head_dim),
            hbm_bytes=(rows * draft.hidden_size * activation_bytes + weight_bytes
                       + rows * qkv_width * activation_bytes),
            mte2_bytes=rows * draft.hidden_size * activation_bytes + weight_bytes,
            mte3_bytes=rows * qkv_width * activation_bytes,
            dtype=draft.activation_dtype,
            metadata=(("native_kind", PROLOG_NATIVE_KIND),
                      ("native_source", SOURCE),
                      ("do_rope", True),
                      ("fused_members", ("qkv_a", "kv_norm", "kv_rope")),
                      ("context_rows", rows), ("query_rows", query_rows),
                      ("data_parallel_context", int(getattr(parallel, "cp", 1) or 1)),
                      ("rank_owned_rows", rank_rows),
                      ("kv_row_bytes", row_bytes),
                      ("hidden_read_once", True),
                      ("timing_boundary", "expanded_semantic_operator")),
            **layer_common)
        ops.append(prolog)
        stored = cache_store_op(
            op_id=name + ".kv_scatter", deps=(prolog.op_id,), rows=rank_rows,
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
