"""Gathered-head latent attention and candidate-only context parallelism.

[Role] Compose explicit Q gather, batched attention and output AllToAll.
[Invariants] TP projections stay local; KV history and index history shard.
  The FP8 call binds a labeled resource envelope, never the INT8 source model.
[Timeouts & Locks] O(active requests) construction; no locks.
[Side-Effect Policy] Builds immutable operators; no NPU or source execution.
"""
from dataclasses import replace

from mindie_sim.catalog.attention_common import active_dcp_degree
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.paired_sfa import execution, SOURCE
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .attention_pipeline import TiledFlashAndPagedAttentionRecipe
from .mechanism import OperatorPlan, PlannedRecipe, Work

ONE = 1
TWO = 2
FP32_BYTES = 4
MERGE_OPS = 3
MAX_HEADS = 128
PACK_LSE_OPS = 4
FP8_FORMATS = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2"})
IMPLEMENTATION = "paired_fp8_qsfa_analytic"


ZERO = 0


def enabled(model, parallel, views, runtime=None):
    selected = bool(execution(model).get("dcp_query_gather"))
    nonlocal_phase = any(view.phase != "decode" for view in views)
    if selected and nonlocal_phase and (parallel.cp > ONE or parallel.pcp != ONE):
        raise ModelingError("gathered_head_dcp_requires_decode_pcp1")
    if not selected or nonlocal_phase:
        return False
    # Decode may use DCP even when physical PCP participants remain: every
    # request resolves to one effective KV-sharding degree.
    effective_cp = max(active_dcp_degree(parallel, view.phase) for view in views)
    if runtime:
        kernels = runtime.kernels
        if (set(kernels.get("lowerings", {})) & {
                "mla_decode", "paged_mla_attention", "indexed_attention",
                "mqsmla_attention", "sparse_kv_offload"}
                or "indexed_attention" in kernels.get("programs", {})
                or "attention" in kernels.get("families", {})):
            raise ModelingError("gathered_head_attention_conflicts_with_explicit_implementation")
    return selected and effective_cp >= ONE


def query_gather(b, query, rope, rows, heads, width, cp):
    if cp == ONE:
        return query, rope
    size = rows * heads * width * dtype_bytes(b.model.activation_dtype)
    packed = b.work("dcp_query_pack", (query, rope), (rows, heads, width),
                    read=size, write=size, family="elementwise")
    gathered = b.communicate("dcp_query_allgather", (packed,), size * (cp - ONE),
                             "allgather", "cp", gathered_heads=heads * cp)
    return gathered, gathered


def attention_batch(b, outputs, views, *, plan, heads, local_heads, cp, rows):
    ids = set(outputs)
    members = [op for op in b.ops if op.op_id in ids]
    if not members:
        return outputs
    meta = dict(members[ZERO].metadata)
    # The resolved cache representation, not the checkpoint's raw kv_dtype,
    # selects the native QSFA call.
    fp8 = cache_format(b.model, "mla_kv").dtype in FP8_FORMATS and plan.indexer
    if fp8 and (b.model.activation_dtype != "bf16" or not ONE <= heads <= MAX_HEADS):
        raise ModelingError("fp8_qsfa_requires_bf16_queries_and_native_head_domain")
    meta.update({
        "program_kind": "paired_latent_attention",
        "attention_fidelity": "source_call_resource_envelope",
        "gathered_q_heads": heads, "local_projection_heads": local_heads,
        "source": {**SOURCE, "path": "vllm_ascend/device/device_op.py"},
        "native_kernel": "npu_kv_quant_sparse_flash_attention_vllm" if fp8 else "mla_attention",
        "native_layout": {"query": "TND", "kv": "PA_BSND"},
        "unmodeled_mechanisms": ("compiled_tiling", "internal_pipeline_overlap", "host_runtime_calibration"),
        "request_sequences": tuple((view.q_len, view.context) for view in views),
    })
    meta.pop("indexed_attention_geometry", None)
    shape = (sum(view.q_len for view in views), heads, plan.kv_lora_rank)
    out_bytes = dtype_bytes(b.model.activation_dtype)
    statistics = rows * heads * TWO * FP32_BYTES
    meta.update({
        "qsfa_output_bytes": rows * heads * (plan.kv_lora_rank * out_bytes + TWO * FP32_BYTES),
        "qsfa_statistics_bytes": statistics,
        "qsfa_output_dtype": b.model.activation_dtype,
    })
    batch = replace(
        members[ZERO], op_id=b.prefix + ".latent_attention_batch",
        recipe=IMPLEMENTATION if fp8 else "paired_mla_attention_analytic",
        source_boundary="kernel" if fp8 else "",
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
        flops=sum(op.flops for op in members), vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members) + (statistics if fp8 else ZERO),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members) + (statistics if fp8 else ZERO),
        metadata=tuple(meta.items()))
    b.ops[:] = [op for op in b.ops if op.op_id not in ids]
    b.ops.append(batch)
    if cp == ONE:
        return [batch.op_id]
    # BF16 output plus four BF16 lanes containing the FP32 LSE payload.
    lse_lanes = TWO * FP32_BYTES // int(out_bytes) if out_bytes != FP32_BYTES else ONE
    packed_bytes = rows * heads * (plan.kv_lora_rank + lse_lanes) * out_bytes
    pack = b.work(
        "dcp_output_pack", (batch.op_id,), (rows, heads, plan.kv_lora_rank),
        vector=rows * heads * PACK_LSE_OPS,
        read=rows * heads * (plan.kv_lora_rank * out_bytes + TWO * FP32_BYTES),
        write=packed_bytes, family="elementwise")
    wire = rows * local_heads * (plan.kv_lora_rank + lse_lanes) * out_bytes * (cp - ONE)
    gathered = b.communicate("dcp_output_alltoall", (pack,), wire,
                             "alltoall", "cp", source="dcp_a2a_fused")
    elements = rows * local_heads * plan.kv_lora_rank
    merged = b.work("dcp_lse_merge", (gathered,), (rows, local_heads, plan.kv_lora_rank),
                    vector=MERGE_OPS * elements * cp,
                    read=rows * local_heads * (plan.kv_lora_rank + lse_lanes) * out_bytes * cp,
                    write=elements * out_bytes, family="attention_merge")
    return [merged]


def _build(op, geometry):
    if dict(op.metadata).get("native_kernel") != "npu_kv_quant_sparse_flash_attention_vllm":
        raise ModelingError("paired_fp8_qsfa_call_contract_required")
    work = Work(matrix={op.dtype: op.flops}, resources={
        "hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
        "mte3": op.mte3_bytes, "vector": op.vector_ops})
    return OperatorPlan(work, TiledFlashAndPagedAttentionRecipe().lower,
                        {"fidelity": "class_a_tiled_attention", "source": SOURCE})


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(_build)}
