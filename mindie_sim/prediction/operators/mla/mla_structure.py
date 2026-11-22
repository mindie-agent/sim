"""Declared fusion groups over the existing MLA operator DAG.

[Role] Preserve prolog and indexer boundaries without replacing their operators.
[Invariants] Groups use builder geometry and source selection, never captures.
  A group does not add work, collapse dependencies or assert native timing.
[Timeouts & Locks] Linear metadata updates within one layer; no I/O or locks.
[Side-Effect Policy] Replaces records only in the caller-owned operator list.
"""
from mindie_sim.prediction.structure_groups import declare_group


def declare_indexer(ops, *, prefix, plan, parallel, model, quant, views,
                    padded_tokens, scores, topk):
    """Unconditionally declare the DSA indexer subgraph over rank-local KV."""
    selection = next(op for op in ops if op.op_id == topk)
    sel_meta = dict(selection.metadata)
    quantized = quant.index_dtype not in {"bf16", "fp16"}
    role = "quant_lightning_indexer" if quantized else "lightning_indexer"
    partition = tuple({"query": view.q_len, "context": view.context,
                       "kv_length": view.kv_len} for view in views)
    # A fused native indexer is one operator for both score and TopK.
    fused = scores == topk
    members = (topk,) if fused else (scores, topk)
    entries = (topk,) if fused else (scores,)
    is_kernel = selection.source_boundary == "kernel"
    sel_impl = sel_meta.get("native_implementation")
    sel_native_kind = (
        sel_meta.get("native_kind")
        or (sel_impl.get("native_kind") if isinstance(sel_impl, dict) else None)
    )
    default_kind = ("VllmAscendQuantLightningIndexerV2" if quantized
                    else "LightningIndexer")
    fusion_status = "fused_existing" if fused else "unfused_decomposed"
    if is_kernel:
        fusion_reason = "selected_native_indexer_kernel"
    elif fused:
        fusion_reason = "fused_quantized_indexer_operator"
    elif int(parallel.cp) != 1 or bool(plan.candidate_filter):
        fusion_reason = "dcp_or_candidate_filter_decomposed_indexer"
    else:
        fusion_reason = "separate_score_and_topk_operators"
    native_fusion_eligible = bool(
        int(parallel.cp) == 1 and not plan.candidate_filter and not is_kernel)
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
            "selection_output_dtype": sel_meta.get("selection_output_dtype", "int32"),
            "scale_dtype": quant.index_scale_dtype,
        },
        implementation={
            "native_kind": sel_native_kind or default_kind,
            "conditions": {
                "cp": int(parallel.cp), "pcp": int(parallel.pcp),
                "candidate_filter": bool(plan.candidate_filter),
                "source_selection": "semantic_score_and_topk_projection",
                "timing_boundary": ("single_fused_quantized_indexer" if fused
                                    else "separate_declared_operators"),
            },
        },
        source={"kind": "semantic_builder", "module": "mindie_sim.prediction.operators.mla.mla"},
        fusion_status=fusion_status,
        fusion_reason=fusion_reason,
        native_fusion_eligible=native_fusion_eligible,
    )
