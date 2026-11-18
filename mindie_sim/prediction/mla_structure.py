"""Declared fusion groups over the existing MLA operator DAG.

[Role] Preserve prolog and indexer boundaries without replacing their operators.
[Invariants] Groups use builder geometry and source selection, never captures.
  A group does not add work, collapse dependencies or assert native timing.
[Timeouts & Locks] Linear metadata updates within one layer; no I/O or locks.
[Side-Effect Policy] Replaces records only in the caller-owned operator list.
"""
from .structure_groups import declare_group


def declare_indexer(ops, *, prefix, plan, parallel, model, quant, views,
                    padded_tokens, scores, topk):
    if parallel.cp != 1 or plan.candidate_filter:
        return
    selection = next(op for op in ops if op.op_id == topk)
    quantized = quant.index_dtype not in {"bf16", "fp16"}
    role = "quant_lightning_indexer" if quantized else "lightning_indexer"
    partition = tuple({"query": view.q_len, "context": view.context,
                       "kv_length": view.kv_len} for view in views)
    # A fused native indexer is one operator for both score and TopK.
    fused = scores == topk
    members = (topk,) if fused else (scores, topk)
    entries = (topk,) if fused else (scores,)
    declare_group(
        ops, key=prefix + "." + role, role=role,
        family="dsa", members=members, entries=entries, anchor=topk,
        view={
            "tokens": sum(view.q_len for view in views), "physical_tokens": padded_tokens,
            "sequence_partition": partition, "compression_ratio": max(1, plan.compress_ratio),
            "heads": plan.index_heads, "width": plan.index_dim,
            "topk": plan.index_topk, "query_dtype": quant.index_dtype,
            "weight_dtype": "fp16" if quant.index_dtype == "int8" else model.activation_dtype,
            "cache_dtype": quant.index_dtype,
            "selection_output_dtype": dict(selection.metadata)["selection_output_dtype"],
            "scale_dtype": quant.index_scale_dtype,
        },
        implementation={
            "native_kind": (dict(selection.metadata).get("native_kind")
                            if fused else
                            "VllmAscendQuantLightningIndexerV2" if quantized
                            else "LightningIndexer"),
            "conditions": {
                "cp": 1, "pcp": parallel.pcp, "candidate_filter": False,
                "source_selection": "semantic_score_and_topk_projection",
                "timing_boundary": ("single_fused_quantized_indexer" if fused
                                    else "separate_declared_operators"),
            },
        },
        source={"kind": "semantic_builder", "module": "mindie_sim.prediction.mla"},
    )
