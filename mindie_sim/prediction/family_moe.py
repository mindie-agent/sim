"""Ordinary MoE routing bindings to source alignment and combination plans.

[Role] Connect the semantic ``route_pack``/``route_unpack`` calls to the
  existing CUDA alignment and Triton weighted-combination kernels.
[Invariants] Alignment emits indices only; the packing copy and its dependency
  remain. Explicit ``topk_ids`` are required and never reconstructed from a
  histogram. Combination is selected only for an actual
  ``[tokens, topk, hidden]`` tensor contract, a declared router-weight dtype,
  no expert map, no weight-on-input, and the explicit capability-89 source
  branch. Expert-grouped input is materialized by a declared dependent copy,
  not relabeled as token-major.
[Timeouts & Locks] O(tokens*topk) metadata for explicit ids; pure local
  construction, no histories or locks.
[Side-Effect Policy] Returns operator records only; no device or measurement
  access.
"""
from dataclasses import replace
from typing import Mapping

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .implementations import preserve_graph
from .mechanism import PlannedRecipe
from .operators import moe_align, moe_combine


ALIGN_IMPLEMENTATION = "cuda_moe_align_block_size"
COMBINE_IMPLEMENTATION = moe_combine.IMPLEMENTATION
MATERIALIZE_RECIPE = "route_unpack.materialize"
_WEIGHT_DTYPES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
}


def recipes():
    return {
        ALIGN_IMPLEMENTATION: PlannedRecipe(
            moe_align.build, kernels=moe_align.kernels),
        moe_combine.RECIPE: PlannedRecipe(moe_combine.build),
    }


def _source(call, params):
    try:
        return call(params)
    except ModelingError:
        raise
    except ValueError as exc:
        raise ModelingError(str(exc)) from None


def _routing(metadata):
    if metadata.get("routing_basis") != "declared_integer_counts":
        return None
    raw = metadata.get("routing_realization")
    return dict(raw) if isinstance(raw, Mapping) else None


def _explicit_ids(metadata):
    realization = _routing(metadata)
    if realization is None:
        return None, None
    raw = realization.get("topk_ids")
    if raw is None:
        return None, realization
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ModelingError("moe_align_topk_ids_must_be_2d_sequence")
    return tuple(tuple(row) for row in raw), realization


def _merged_metadata(op, **updates):
    return tuple({**dict(op.metadata), **updates}.items())


def _lower_align(op, *, block_size=None, pad_sorted_ids=False,
                 ignore_invalid_experts=False, topk_ids_dtype="int32"):
    metadata = dict(op.metadata)
    ids, realization = _explicit_ids(metadata)
    if ids is None:
        raise ModelingError(
            "moe_align_requires_explicit_topk_ids_in_routing_realization")
    producer = metadata.get("topk_ids_producer")
    if not producer or producer not in op.deps:
        raise ModelingError("moe_align_requires_topk_ids_producer_dependency")
    if block_size is None:
        raise ModelingError("moe_align_requires_explicit_block_size_option")
    declared_dtype = metadata.get("topk_ids_dtype", "int32")
    if topk_ids_dtype != declared_dtype:
        raise ModelingError(
            "moe_align_topk_ids_dtype_must_match_declared_route_pack_ids:"
            f"{topk_ids_dtype}!={declared_dtype}")
    num_experts = metadata.get("num_experts")
    if type(num_experts) is not int or num_experts < 1:
        raise ModelingError("moe_align_requires_declared_positive_num_experts")
    tokens, top_k = op.local_shape
    source = _source(moe_align.make_op, {
        "topk_ids": ids,
        "num_experts": num_experts,
        "block_size": block_size,
        "pad_sorted_ids": pad_sorted_ids,
        "ignore_invalid_experts": ignore_invalid_experts,
        "topk_ids_dtype": topk_ids_dtype,
        "implementation": ALIGN_IMPLEMENTATION,
        "phase": op.phase,
    })
    analysis = dict(dict(source.metadata)["analysis"])
    if (analysis["tokens"], analysis["top_k"]) != (tokens, top_k):
        raise ModelingError("moe_align_token_geometry_mismatch")
    if tuple(analysis["counts"]) != tuple(realization.get("expert_rows", ())):
        raise ModelingError("moe_align_ids_must_agree_with_declared_histogram")
    semantic = replace(
        op, op_id=op.op_id + ".align", family="moe_align",
        recipe=ALIGN_IMPLEMENTATION, source_boundary="call",
        deps=(producer,), semantic_shape=(tokens, top_k),
        dtype="int32", layout="flat",
        metadata=(
            ("routing_basis", "declared_integer_counts"),
            ("routing_realization", realization),
            ("topk_ids_producer", producer),
        ),
    )
    align = preserve_graph(source, semantic)
    align = replace(align, metadata=_merged_metadata(
        align, routing_realization=realization,
        topk_ids_producer=producer,
        semantic_layout="routing_indices_only"))
    # Scan the source's padded index output into compact expert-grouped rows.
    # Sentinels cost index reads, but never cause payload reads or writes.
    original_index_bytes = tokens * top_k * 4
    sorted_index_bytes = analysis["total_padded"] * 4
    expert_index_bytes = analysis["used_blocks"] * 4
    count_bytes = 4
    index_delta = sorted_index_bytes + expert_index_bytes + count_bytes - original_index_bytes
    pack = replace(
        op, deps=tuple(dict.fromkeys((*op.deps, align.op_id))),
        mte2_bytes=op.mte2_bytes + index_delta,
        hbm_bytes=op.hbm_bytes + index_delta,
        vector_ops=op.vector_ops + analysis["total_padded"],
        metadata=_merged_metadata(
            op, alignment_op=align.op_id,
            packed_layout="expert_grouped",
            topk_ids_dtype=topk_ids_dtype,
            pack_contract={
                "implementation": "analytical_aligned_to_compact_copy",
                "index_producer": align.op_id,
                "index_inputs": ("sorted_token_ids", "expert_ids", "num_tokens_post_pad"),
                "sorted_index_bytes": sorted_index_bytes,
                "expert_index_bytes": expert_index_bytes,
                "count_bytes": count_bytes,
                "scanned_index_slots": analysis["total_padded"],
                "sentinel": analysis["sentinel"],
                "sentinel_slots": analysis["total_padded"] - analysis["valid_entries"],
                "compact_rows": analysis["valid_entries"],
                "output_layout": "compact_expert_grouped_with_inverse_token_slot",
                "inverse_index_bytes": tokens * top_k * 4,
                "payload_reads": "original_activation_rows",
                "padded_payload_reads_or_writes": False,
            },
        ),
    )
    return align, pack


def _canonical_weight_dtype(value):
    if not isinstance(value, str):
        return None
    return _WEIGHT_DTYPES.get(value.strip().lower())


def _materialize_expert_grouped(op, metadata, realization, *, producer, weights):
    ids = realization.get("topk_ids")
    if ids is None:
        raise ModelingError(
            "moe_combine_requires_explicit_topk_ids_for_expert_grouped_input")
    inverse = metadata.get("inverse_index_producer")
    if not inverse or inverse not in op.deps:
        raise ModelingError("moe_combine_requires_pack_inverse_index_dependency")
    tokens, top_k, hidden = (int(value) for value in op.local_shape)
    input_rows = metadata.get("input_rows")
    if type(input_rows) is not int or input_rows < 0:
        raise ModelingError("moe_combine_requires_declared_expert_grouped_input_rows")
    expert_rows = tuple(realization.get("expert_rows", ()))
    if sum(expert_rows) != input_rows:
        raise ModelingError(
            "moe_combine_expert_rows_must_match_expert_grouped_input_rows")
    element_bytes = dtype_bytes(op.dtype)
    payload_in = input_rows * hidden * element_bytes
    payload_out = tokens * top_k * hidden * element_bytes
    # The packer's inverse token-slot index travels with the compact rows.
    # Top-k IDs alone cannot reconstruct an atomic sort's within-expert order.
    read = payload_in + tokens * top_k * 4
    write = payload_out
    materialize = vector_op(
        op_id=op.op_id + ".materialize",
        deps=tuple(dict.fromkeys((producer, inverse, weights))),
        stage=op.stage, layer=op.layer, requests=op.requests,
        elements=tokens * top_k * hidden, ops_per_element=1,
        vector_ops_override=tokens * top_k * hidden,
        dtype=op.dtype, family="routing", phase=op.phase,
        recipe=MATERIALIZE_RECIPE, shape=(tokens, top_k, hidden),
        read_bytes_override=read, write_bytes_override=write,
        metadata=(
            ("mapping_basis", "pack_produced_inverse_token_slot"),
            ("inverse_index_producer", inverse),
            ("inverse_index_bytes", tokens * top_k * 4),
            ("input_layout", "expert_grouped"),
            ("output_layout", "tokens_topk_hidden"),
            ("input_rows", input_rows),
            ("expert_rows", expert_rows),
            ("routing_realization", realization),
            ("payload_producer", producer),
            ("topk_weights_producer", weights),
            ("materialization_basis", "declared_copy_not_a_source_kernel"),
        ),
        notes=("explicit expert-grouped to token-major materialization",
               "does not relabel compacted rows as token-major"),
    )
    return materialize


def _lower_combine(op, *, compute_capability=None, weight_dtype=None):
    metadata = dict(op.metadata)
    if op.layout != "contiguous":
        raise ModelingError("moe_combine_requires_contiguous_output_layout")
    if len(op.local_shape) != 3:
        raise ModelingError("moe_combine_requires_tokens_topk_hidden_source_input")
    tokens, top_k, hidden = (int(value) for value in op.local_shape)
    if min(tokens, top_k, hidden) < 1:
        raise ModelingError("moe_combine_requires_positive_tokens_topk_hidden")
    if metadata.get("has_expert_map") is not False:
        raise ModelingError("moe_combine_requires_no_expert_map_branch")
    if metadata.get("apply_router_weight_on_input") is not False:
        raise ModelingError("moe_combine_requires_router_weights_on_input_semantics")
    declared_weight = _canonical_weight_dtype(metadata.get("router_weight_dtype"))
    if declared_weight is None:
        raise ModelingError("moe_combine_requires_declared_router_weight_dtype")
    selected_weight = _canonical_weight_dtype(weight_dtype)
    if selected_weight is None:
        raise ModelingError("moe_combine_requires_explicit_weight_dtype_option")
    if selected_weight != declared_weight:
        raise ModelingError(
            "moe_combine_weight_dtype_must_match_declared_router_weights:"
            f"{selected_weight}!={declared_weight}")
    if compute_capability is None:
        raise ModelingError("moe_combine_requires_explicit_compute_capability_option")
    if type(metadata.get("ep_degree", 1)) is not int or metadata.get("ep_degree", 1) != 1:
        raise ModelingError("moe_combine_ordinary_binding_requires_ep1_local_rows")
    producer = metadata.get("payload_producer")
    weights = metadata.get("topk_weights_producer")
    if not producer or producer not in op.deps:
        raise ModelingError("moe_combine_requires_payload_producer_dependency")
    if not weights or weights not in op.deps:
        raise ModelingError("moe_combine_requires_topk_weights_producer_dependency")
    layout = metadata.get("input_layout")
    realization = _routing(metadata)
    materialize = None
    if layout == "expert_grouped":
        if realization is None:
            raise ModelingError(
                "moe_combine_expert_grouped_input_requires_explicit_realization")
        materialize = _materialize_expert_grouped(
            op, metadata, realization, producer=producer, weights=weights)
        combine_deps = (materialize.op_id, weights)
    elif layout == "tokens_topk_hidden":
        combine_deps = tuple(dict.fromkeys((*op.deps, weights)))
    else:
        raise ModelingError(
            f"moe_combine_unsupported_caller_input_layout:{layout!r}")
    source = _source(moe_combine.make_op, {
        "tokens": tokens, "hidden": hidden, "topk": top_k,
        "dtype": op.dtype, "weight_dtype": selected_weight,
        "compute_capability": compute_capability,
        "implementation": COMBINE_IMPLEMENTATION,
        "apply_router_weight_on_input": False,
        "has_expert_map": False,
    })
    if tuple(source.local_shape) != (tokens, top_k, hidden):
        raise ModelingError("moe_combine_source_input_shape_mismatch")
    if source.mte3_bytes != tokens * hidden * dtype_bytes(op.dtype):
        raise ModelingError("moe_combine_output_layout_or_dtype_mismatch")
    if dict(source.metadata)["dispatch"].get("has_expert_map") is not False:
        raise ModelingError("moe_combine_source_selected_expert_map_branch")
    semantic = replace(
        op, family="moe_combine", recipe=COMBINE_IMPLEMENTATION,
        source_boundary="kernel", deps=combine_deps,
        local_shape=(tokens, top_k, hidden),
        semantic_shape=(tokens, top_k, hidden),
        metadata=(
            ("combine_input_layout", "tokens_topk_hidden"),
            ("combine_output_shape", (tokens, hidden)),
            ("router_weight_dtype", selected_weight),
            ("topk_weights_producer", weights),
        ),
    )
    combine = preserve_graph(source, semantic)
    combine = replace(combine, metadata=_merged_metadata(
        combine, combine_input_layout="tokens_topk_hidden",
        combine_output_shape=(tokens, hidden),
        router_weight_dtype=selected_weight,
        topk_weights_producer=weights,
        routing_realization=realization,
    ))
    return (materialize, combine) if materialize is not None else combine


def lower_op(op, **options):
    """Lower exactly one selected semantic routing recipe."""
    if op.recipe == "route_pack":
        return _lower_align(op, **options)
    if op.recipe == "route_unpack":
        return _lower_combine(op, **options)
    raise ModelingError(f"moe_family_unsupported_recipe:{op.recipe}")


__all__ = [
    "ALIGN_IMPLEMENTATION", "COMBINE_IMPLEMENTATION", "MATERIALIZE_RECIPE",
    "lower_op", "recipes",
]
