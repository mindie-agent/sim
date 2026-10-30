"""Declared one-sequence pointer attention on the ordinary GQA model path.

[Role] Translate GQA geometry into the existing 2D pointer source planner.
[Invariants] One source invocation serves one sequence. Batched launches, CP,
  windowed attention and quantized KV require their own implementations.
[Timeouts & Locks] Pure per-sequence metadata work; no locks or subprocesses.
[Side-Effect Policy] Returns records; no device initialization or observations.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from .implementations import preserve_graph
from .mechanism import PlannedRecipe
from .operators import attention_online


def recipes():
    return {attention_online.RECIPE_ID: PlannedRecipe(attention_online.build)}


def validate(*, views, plan, parallel, policy, quant, model, compute_capability=None):
    if parallel.cp != 1 or parallel.pcp != 1:
        raise ModelingError("pointer_attention_requires_unpartitioned_queries_and_kv")
    if plan.window or quant.kv_dtype != model.activation_dtype:
        raise ModelingError("pointer_attention_requires_unwindowed_unquantized_kv")
    if policy.fused_batch and len(views) != 1:
        raise ModelingError("pointer_attention_one_sequence_launch_requires_fused_batch_false")
    for view in views:
        if view.query_ranges and view.query_ranges != ((0, view.q_len),):
            raise ModelingError("pointer_attention_requires_contiguous_query_range")
    # The implementation id explicitly selects pointer 2D: descriptor mode and
    # segmented decode buffers are absent. Capability only selects source tiles.
    if compute_capability is None:
        raise ModelingError("pointer_attention_compute_capability_declaration_required")


def lower_op(op, *, view, plan, key_heads, model, runtime, compute_capability):
    source = attention_online.make_op({
        "query": view.q_len, "context": view.context, "heads": op.local_shape[2],
        "key_heads": key_heads, "head_dim": plan.head_dim, "dtype": model.activation_dtype,
        "causal": not plan.noncausal_attention, "compute_capability": compute_capability,
        "page_block_size": runtime.kv_block_size,
    })
    source = preserve_graph(source, op)
    return replace(source, metadata=source.metadata + (("implementation_binding", {
        "selector": "gqa_attention", "implementation": attention_online.IMPLEMENTATION_ID,
        "call_batching": "one_sequence_per_launch", "segmented_decode": False,
    }),))
