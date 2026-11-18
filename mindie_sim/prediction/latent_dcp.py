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
#: Packed INT8 latent rows carry the C8 tile-128 sparse kernel.
PACKED_CACHE_DTYPES = frozenset({"int8"})
#: C8 kernel domain: 512 latent values, one FP32 scale per 128, BF16 rope tail.
PACKED_C8_GROUP_SIZE = 128
PACKED_C8_LATENT = 512
PACKED_C8_ROPE = frozenset({0, 64})
#: Declared native call split of one gathered-head DCP decode: two sequential
#: ``H/2`` calls over the same packed cache plus the peer exchange below.
PAIRED_SFA_CALLS = 2
IMPLEMENTATION = "paired_fp8_qsfa_analytic"
PACKED_KERNEL = "npu_kv_quant_sparse_flash_attention_vllm"


ZERO = 0


def enabled(model, parallel, views, runtime=None):
    selected = bool(execution(model).get("dcp_query_gather"))
    nonlocal_phase = any(view.phase != "decode" for view in views)
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


def paired_sfa_calls(heads):
    """Declared sequential native calls of one gathered-head C8 SFA decode."""
    if heads < PAIRED_SFA_CALLS or heads % PAIRED_SFA_CALLS:
        raise ModelingError("paired_sfa_requires_even_gathered_heads")
    return (heads // PAIRED_SFA_CALLS,) * PAIRED_SFA_CALLS


def packed_sfa_geometry(meta, heads, plan, views, cp, *, cache_dtype="",
                        cache_fmt=None):
    """Packed-C8 geometry of the declared paired calls, or ``None``.

    Each call keeps the same selected rows and cache; only the gathered head
    half changes, so the pair costs two sequential tile walks over one cache.
    A plain INT8 MLA row is not the C8 tile-128 representation and must not
    select the packed kernel.
    """
    view = dict(meta.get("attention_tensor_view") or {})
    dtype = str(view.get("cache_dtype") or cache_dtype or "")
    if dtype not in PACKED_CACHE_DTYPES:
        return None
    if cache_fmt is not None and not (
            cache_fmt.dtype == "int8"
            and cache_fmt.packed_elements == ONE
            and cache_fmt.group_size == PACKED_C8_GROUP_SIZE
            and cache_fmt.scale_dtype == "fp32"
            and cache_fmt.suffix_elements == int(plan.qk_rope_head_dim)
            and cache_fmt.suffix_dtype == "bf16"):
        return None
    if (heads % PAIRED_SFA_CALLS or plan.cache_row_elements <= 0
            or int(plan.kv_lora_rank) != PACKED_C8_LATENT
            or int(plan.qk_rope_head_dim) not in PACKED_C8_ROPE):
        return None
    degree = max(ONE, int(cp))
    sharded_topk = (max(ONE, -(-int(plan.index_topk) // degree))
                    if int(plan.index_topk) > ZERO else ZERO)
    sequences = tuple(
        (int(v.q_len), max(ZERO, -(-int(v.context) // degree))) for v in views)
    extents = tuple(
        max(ZERO, -(-min(int(v.prefix_tokens or ZERO), int(v.context)) // degree))
        for v in views)
    shared = any(extents) and all(str(v.prefix_cache_key or "") for v in views
                                  if v.prefix_tokens)
    keys = tuple(str(v.prefix_cache_key or "") for v in views)
    return {
        "sequences": sequences,
        "topk": sharded_topk,
        "compression_ratio": ONE, "window": ZERO, "cp": ONE,
        "heads": heads // PAIRED_SFA_CALLS,
        "latent": int(plan.kv_lora_rank), "rope": int(plan.qk_rope_head_dim),
        "query_dtype": "bf16", "cache_dtype": "int8",
        **({"prefix_cache_mode": "shared", "prefix_tokens": extents,
            "prefix_cache_keys": keys} if shared
           else {"prefix_cache_mode": "private"}),
        "paired_calls": PAIRED_SFA_CALLS,
        "dcp_peer_exchange": int(cp),
    }


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
    resolved = cache_format(b.model, "mla_kv")
    fp8 = resolved.dtype in FP8_FORMATS and plan.indexer
    packed = (packed_sfa_geometry(meta, heads, plan, views, cp,
                                  cache_dtype=resolved.dtype, cache_fmt=resolved)
              if plan.indexer else None)
    native_packed = fp8 or packed is not None
    if native_packed and (b.model.activation_dtype != "bf16"
                          or not ONE <= heads <= MAX_HEADS):
        raise ModelingError("fp8_qsfa_requires_bf16_queries_and_native_head_domain")
    meta.update({
        "program_kind": "paired_latent_attention",
        "attention_fidelity": "source_call_resource_envelope",
        "gathered_q_heads": heads, "local_projection_heads": local_heads,
        "source": {**SOURCE, "path": "vllm_ascend/device/device_op.py"},
        "native_kernel": PACKED_KERNEL if native_packed else "mla_attention",
        "native_layout": {"query": "TND", "kv": "PA_BSND"},
        "unmodeled_mechanisms": ("compiled_tiling", "internal_pipeline_overlap", "host_runtime_calibration"),
        "request_sequences": tuple((view.q_len, view.context) for view in views),
    })
    meta.pop("indexed_attention_geometry", None)
    if packed is not None:
        meta.update({"program_kind": "indexed_attention",
                     "indexed_attention_geometry": packed,
                     "paired_sfa": {
                         "calls": PAIRED_SFA_CALLS,
                         "call_heads": packed["heads"],
                         "peer_exchange": packed["dcp_peer_exchange"]}})
    shape = (sum(view.q_len for view in views), heads, plan.kv_lora_rank)
    out_bytes = dtype_bytes(b.model.activation_dtype)
    statistics = rows * heads * TWO * FP32_BYTES
    if packed is not None:
        # The paired C8 calls also write the FP32 LSE statistics the output
        # pack reads.  Carry the exact declared volume into the arch35 plan.
        packed["statistics_bytes"] = statistics
    meta.update({
        "qsfa_output_bytes": rows * (packed["heads"] if packed else heads)
        * (plan.kv_lora_rank * out_bytes + TWO * FP32_BYTES),
        "qsfa_statistics_bytes": statistics,
        "qsfa_output_dtype": b.model.activation_dtype,
    })
    batch = replace(
        members[ZERO], op_id=b.prefix + ".latent_attention_batch",
        recipe=IMPLEMENTATION if native_packed else "paired_mla_attention_analytic",
        source_boundary="kernel" if native_packed else "",
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
        flops=sum(op.flops for op in members), vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members) + (
            statistics if (native_packed or cp > ONE) else ZERO),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members) + (
            statistics if (native_packed or cp > ONE) else ZERO),
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
    meta = dict(op.metadata)
    if meta.get("native_kernel") != PACKED_KERNEL:
        raise ModelingError("paired_fp8_qsfa_call_contract_required")
    if (meta.get("program_kind") == "indexed_attention"
            and meta.get("indexed_attention_geometry")):
        # The packed INT8 cache is the C8 tile-128 program; the gathered head
        # half is one declared sequential call of the pair.
        from .indexed_quant_attention import SCHEDULE_ARCH35, build as build_packed
        return build_packed(op, geometry, scratch_policy="offchip", compact=True,
                            operand_modes=True, schedule=SCHEDULE_ARCH35)
    work = Work(matrix={op.dtype: op.flops}, resources={
        "hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
        "mte3": op.mte3_bytes, "vector": op.vector_ops})
    return OperatorPlan(work, TiledFlashAndPagedAttentionRecipe().lower,
                        {"fidelity": "class_a_tiled_attention", "source": SOURCE})


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(_build)}
