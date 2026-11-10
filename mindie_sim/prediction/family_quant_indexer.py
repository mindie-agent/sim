"""One complete quantized lightning-indexer invocation.

[Role] Keep QK, weighted head reduction and top-k on one common-engine kernel.
[Invariants] Query/key quantization, rotary transforms, scale producers and host
  metadata remain caller dependencies. The source performs INT8 QK followed by
  FP16 head reduction; per-head scores remain on chip. A resource envelope does
  not identify the captured tiling, owner split, cache residency or binary.
[Timeouts & Locks] O(requests) integer sums; no I/O or locks.
[Side-Effect Policy] Append one operator to the caller-owned builder only.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import _floor_sum
from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .operators.mixed_core import split_reads
from .program import KernelProgram, Stage


IMPLEMENTATION = "cann_quant_lightning_indexer_v2_complete"
RECIPE = "quant_lightning_indexer_v2"
SOURCE = {
    "repository": "ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "cube": {
        "path": "attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_cube_arch22.h",
        "sha256": "87a50714932a4e8e7bd65a230c7bb32476456ac4aaa029e63c1f660244b9c4a6",
        "lines": [492, 577],
    },
    "vector": {
        "path": "attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h",
        "sha256": "d2ba785ac291998012c3b33671d708ad184970bd5ff896f79aa74dc319ae2906",
        "lines": [271, 420],
    },
    "capture_binary_identity": "not_established",
}


def recipes():
    return {RECIPE: PlannedRecipe(_build)}


def _build(op, geometry):
    ledger = dict(op.metadata)["quant_indexer_work"]
    work = split_reads(Work(dict(op.matrix_flops), {
        "hbm": op.hbm_bytes,
        "mte1": ledger["mte1_bytes"],
        "mte2": ledger["aic_read_bytes"] + ledger["aiv_read_bytes"],
        "mte3": op.mte3_bytes,
        "vector": op.vector_ops, "fixpipe": ledger["fixpipe_bytes"],
    }), ledger["aic_read_bytes"], geometry.resource_units)
    return OperatorPlan(work, lambda context: KernelProgram((
        Stage("complete_indexer_service", context.work),)), {
            "execution_abstraction": "complete_kernel_resource_envelope",
            "native_dispatch_reconstructed": False,
            "cross_owner_topk_traffic": "not_selected_without_owner_metadata",
        })


def _owned_spans(view):
    """Owned query spans. Empty ``query_ranges`` is the whole local query."""
    ranges = view.query_ranges
    if not ranges:
        return ((0, view.q_len),)
    return tuple((start, end) for start, end in ranges if end > start)


def declare(builder, *, deps, views, plan, quant, parallel):
    """Declare the supported complete boundary without inferring native owners."""
    if (parallel.cp != 1 or parallel.pcp < 1 or plan.candidate_filter
            or plan.index_kpool != 1 or plan.compress_ratio != 4
            or (plan.index_heads, plan.index_dim, plan.index_topk) != (64, 128, 512)
            or quant.index_dtype != "int8" or quant.index_scale_dtype != "fp16"):
        raise ModelingError("quant_lightning_indexer_v2_complete_domain_mismatch")
    tokens = sum(view.q_len for view in views)
    # Causal compressed-key counts follow each owned span. PCP chunks pass
    # query_ranges; padding and cache allocation do not create index work.
    ratio = plan.compress_ratio
    spans = tuple((view, start, end) for view in views for start, end in _owned_spans(view))
    pairs = sum(_floor_sum(end - start, ratio, 1, view.context + start + 1)
                for view, start, end in spans)
    heads, width = plan.index_heads, plan.index_dim
    sentinel_fills = sum(
        short * plan.index_topk - _floor_sum(short, ratio, 1, view.context + start + 1)
        for view, start, end in spans
        if (short := max(0, min(end - start, plan.index_topk * ratio - (view.context + start + 1))))
    )
    # QueryNd2Nz, ProcessVec0 and WeightDmaCopy all execute under the same
    # isFirstS2InnerLoop guard for queries with at least one compressed key.
    active_query_rows = sum(
        end - start for view, start, end in spans if (view.context + end) // ratio > 0
    )
    query_bytes = active_query_rows * heads * width
    key_bytes = pairs * width
    scale_weight_bytes = active_query_rows * heads * 4 + pairs * 2
    # Source Vec0 broadcasts (weight * query_scale) over the M=16 matrix
    # axis once per active query owner on isFirstS2InnerLoop, and writes/reads
    # that small operand through GM. Only the reduced FP32 score is materialized
    # in GM per pair, not the 64-head INT32 QK result.
    weight_workspace_bytes = active_query_rows * heads * 16 * 2
    score_workspace_bytes = pairs * 4
    output_bytes = tokens * plan.index_topk * 4
    aic_read_bytes = query_bytes + key_bytes + weight_workspace_bytes
    aiv_read_bytes = scale_weight_bytes + score_workspace_bytes
    read_bytes = aic_read_bytes + aiv_read_bytes
    mte3_write_bytes = weight_workspace_bytes + output_bytes
    # ProcessQk loads query/key from L1 to L0A/L0B; ProcessWs loads the FP16
    # score tensor (written to L1 by FixpSToL1) and broadcasted FP16 weights.
    mte1_bytes = (aic_read_bytes + 2 * pairs * heads) if pairs else 0
    vector = active_query_rows * heads * 17 + pairs * 2 + pairs * math.log2(plan.index_topk) + sentinel_fills
    matrix = {"int8": 2 * pairs * heads * width,
              "fp16": 2 * pairs * heads * 16}
    ledger = {
        "query_key_pairs": pairs, "query_rows": tokens,
        "sequence_partition": tuple((view.q_len, view.context) for view in views),
        "qk_int8_flops": matrix["int8"],
        "weighted_head_reduce_fp16_flops": matrix["fp16"],
        "head_reduce_physical_m": 16,
        "dequant_relu": "source_fixpipe_before_fp16_head_reduce",
        "fixpipe_bytes": 2 * pairs * heads + 4 * pairs,
        "mte1_bytes": mte1_bytes,
        "aic_read_bytes": aic_read_bytes, "aiv_read_bytes": aiv_read_bytes,
        "query_bytes": query_bytes, "key_bytes": key_bytes,
        "scale_and_weight_bytes": scale_weight_bytes,
        "weight_workspace_read_write_bytes": 2 * weight_workspace_bytes,
        "reduced_score_workspace_read_write_bytes": 2 * score_workspace_bytes,
        "selection_output_bytes": output_bytes,
        "sentinel_fill_elements": sentinel_fills,
        "topk_comparisons": pairs * math.log2(plan.index_topk),
        "traffic_scope": "query_and_key_once_per_query; native_cache_reloads_unresolved",
        "topk_scope": "logical_comparisons; cross_owner_partial_lists_unresolved",
    }
    result = builder.work(RECIPE, deps, (tokens, 1, plan.index_topk),
        flops=sum(matrix.values()), vector=vector,
        read=read_bytes, write=mte3_write_bytes,
        family="indexer", dtype="int8", metadata=(
            ("native_source", SOURCE), ("quant_indexer_work", ledger),
            ("mte1_bytes", mte1_bytes), ("fixpipe_bytes", ledger["fixpipe_bytes"]),
            ("native_tensor_view", {"kind": "quant_lightning_indexer_v2",
                "query_shape": (tokens, heads, width), "query_dtype": "int8",
                "cache_dtype": "int8", "cache_head_shape": (1, width),
                "scale_dtype": "fp16", "weight_dtype": "fp16",
                "output_shape": (tokens, 1, plan.index_topk), "output_dtype": "int32",
                "layout": "ND", "compression_ratio": 4}),
            ("native_implementation", {"native_kind": "VllmAscendQuantLightningIndexerV2",
                "selection": IMPLEMENTATION, "conditions": {"submitted_kernels": 1,
                    "source_boundary": "qk_weighted_head_reduce_topk",
                    "upstream_quantization_and_host_metadata": "outside"}}),
        ))
    # Reduced scores are written to GM by Fixpipe (FixpResToGm), so they
    # contribute to HBM and Fixpipe traffic without consuming MTE3 stores.
    builder.ops[-1] = replace(builder.ops[-1], matrix_flops=tuple(matrix.items()),
        hbm_bytes=read_bytes + mte3_write_bytes + score_workspace_bytes,
        source_boundary="kernel", coverage="complete_quantized_indexer_resource_envelope")
    return result
