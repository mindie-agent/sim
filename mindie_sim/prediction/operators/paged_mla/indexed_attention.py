"""Compact indexed-attention gather and matrix/vector pipelines.

[Role] Price query-specific sparse MLA phases on a shared batch owner grid.
[Invariants] Query indices are independent. Cache-resident scratch still consumes
  copy service and finite cache capacity; it is not another HBM transaction.
  Matrix padding, QK/softmax/PV ordering and finite producer slots are explicit.
  The finite stage recurrence is compressed, not an instruction simulator.
[Timeouts & Locks] O(query * key_tiles) arithmetic and O(owners + slots) state.
[Side-Effect Policy] Pure plans and shared Engine submissions; no observed times.
"""
from collections import Counter, OrderedDict
from bisect import bisect_right
from dataclasses import replace
from functools import lru_cache, partial
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.catalog.index_pool import selected_tokens
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operators.sharedkv.l0_sync import (
    p03_fixpipe_setup_ms,
    p09_cast_to_alu_op_ratio,
    p13_paired_ms,
    p13_syncall_ms,
)
from mindie_sim.prediction.operators.sparse_indexer.sparse_attention import selected_query_owners
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage

SPLIT_KV_STATE_SCALARS = 2
# sfa_split_kv_merge.h:72-96,146-149: Mul+Add per partial plus 5 scalar stats ops per partial.
SPLIT_KV_MERGE_OPS_PER_ELEM = 2
SPLIT_KV_MERGE_STAT_OPS_PER_PART = 5
PRELOAD_SLOTS_VALID = frozenset({2, 3})


CUBE_PANEL_COLUMNS = 128
DMA_ROW_ALIGNMENT = 32
FP32_ELEMENT_BYTES = 4
FRACTAL_ALIGNMENT = 16
GATHER_INDEX_BYTES_PER_KEY = 8
GATHER_SCRATCH_SLOTS = 4
INDEXED_KEY_TILE = 512
INDEXED_LATENT_DIM = 512
INDEXED_PIPELINE_SLOTS = 3
INDEXED_ROPE_DIM = 64
MAX_INDEXED_HEADS = 128
OUTPUT_FINALIZE_OPS_PER_VALUE = 5
OUTPUT_INIT_COLUMNS = 128
PHASE_CACHE_ENTRIES = 4
SCORE_AND_PROBABILITY_BYTES = 6
SINGLE_AIV_HEAD_LIMIT = 16
SOFTMAX_OPS_PER_SCORE = 8
SOFTMAX_STATE_OPS_PER_HEAD = 26
TILE_CACHE_ENTRIES = 2048
# sparse_flash_attention_kernel_mla.h:1153: 4 Cube<->Vector paired flag handoffs per tile.
INDEXED_PAIRED_SYNC_PHASES = frozenset({"publish", "scores", "probability", "partial"})
# kv_quant_sparse_flash_attention_service_vector_mla.h:2609-2667:
# dequantize = 1.5 ALU/Copy ops (Mul f32 + Copy bf16) + 2.5 Cast ops (i8->f16, f16->f32, f32->bf16).
INDEXED_DEQUANT_NONCAST_OPS = 1.5
INDEXED_DEQUANT_CAST_OPS = 2.5


TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="8fd0bc4885cb52e58791aaa8efacc571b9bd115d",
        relative_path="csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h",
        source_sha256="cec94843ad2144f80e5a504b33f92a682ba8908037435de65da26f5682ce7cdd",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp",
        source_sha256="2c75acce70974165d5b087416d724bcd0d09909b2e8dd693370b5860de9dbc6e",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="8c30e4f491f73cba737a1dacc522894e437cdf3d",
        relative_path="csrc/attention/kv_quant_sparse_flash_attention/op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h",
        source_sha256="3eed5ab746e056319bf873a488a3306cffbabe8dd19f70294da85685f4e3cfd3",
    ),
)
BF16_ELEMENT_BYTES = 2
SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8fd0bc4885cb52e58791aaa8efacc571b9bd115d",
    "branch": "arch22 sparse_flash_attention MLA V_TEMPLATE",
    "scope": "BF16, single latent KV head, per-query indices, no dense window",
}
IMPLEMENTATION = "ascend_sparse_flash_attention_arch22"
#: Seconds per millisecond; the Engine's declared rates are work per millisecond.
MS_PER_SECOND = 1000.0


def recipe(*, scratch_policy, key_tile=INDEXED_KEY_TILE, slots=INDEXED_PIPELINE_SLOTS, compact=True,
           operand_modes=False, rope0_analytic=False):
    return PlannedRecipe(partial(
        build, scratch_policy=scratch_policy, key_tile=key_tile,
        slots=slots, compact=compact, operand_modes=operand_modes,
        rope0_analytic=rope0_analytic))


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(_selected_build)}


def _selected_build(op, geometry):
    sequences = dict(op.metadata).get("indexed_attention_geometry", {}).get("sequences", ())
    if len(sequences) > 1:
        raise ModelingError("indexed_attention_requires_single_contiguous_sequence")
    if any(geometry.resource_units.get(role) != geometry.compute_units * ratio
           for role, ratio in (("vector", 2), ("mte2_aiv", 2), ("mte3", 2),
                               ("mte2_aic", 1), ("mte1", 1), ("fixpipe", 1))):
        raise ModelingError("indexed_attention_source_requires_paired_aic_aiv_pipes")
    return build(op, geometry, operand_modes=True,
                 scratch_policy=dict(op.metadata)["indexed_source_options"]["scratch_policy"])


def lower_op(op, *, selection, view, views, plan, parallel):
    """Bind the declared TND branch without inferring a captured binary."""
    policy = selection.options.get("scratch_policy", "offchip")
    if policy not in {"resident", "offchip"}:
        raise ModelingError("indexed_attention_scratch_policy_required")
    if (len(views) != 1 or parallel.cp != 1
            or view.draft_proposal_tokens
            or (view.query_ranges and view.query_ranges != ((0, view.q_len),))):
        raise ModelingError("indexed_attention_requires_single_contiguous_sequence")
    if (plan.cache_kind != "dense" or plan.mla_variant != "latent_mla"
            or not plan.index_topk or plan.noncausal_attention):
        raise ModelingError("indexed_attention_requires_causal_dense_latent_selection")
    metadata = {**dict(op.metadata), "indexed_source_options": {"scratch_policy": policy},
                "unmodeled_source_mechanisms": (
                    "scalar_index_and_page_lookup_dependencies", "cross_core_flag_latency",
                    "atomic_response_latency", "external_cache_eviction"),
                "implementation_binding": {
                    "selector": "indexed_attention", "implementation": IMPLEMENTATION,
                    "branch_scope": "TND_BF16_block1_MLA_V_TEMPLATE",
                    "capture_dispatch": "not_established",
                }}
    return replace(op, recipe=IMPLEMENTATION, source_boundary="kernel",
                   metadata=tuple(metadata.items()))


def batch_analytic_ops(ops, outputs, *, runtime, views, prefix):
    """Use one query-owner grid for the explicitly selected analytic pipeline.

    Native source bindings retain their narrower admission contract. Explicit
    per-request/family recipes also keep their chosen call boundaries.
    """
    if runtime is None or len(outputs) < 2 or len(outputs) != len(views):
        return outputs
    kernels = runtime.kernels
    program = kernels.get("programs", {}).get("indexed_attention", {})
    if (not isinstance(program, dict) or program.get("kind") not in {
            "indexed_attention_pipeline", "indexed_attention_packed_int8"}
            or "attention" in kernels.get("families", {})
            or "indexed_attention" in kernels.get("lowerings", {})
            or any(view.query_ranges and view.query_ranges != ((0, view.q_len),)
                   for view in views)):
        return outputs
    ids = set(outputs)
    members = [op for op in ops if op.op_id in ids]
    if (len(members) != len(outputs) or any(op.plan_required for op in members)
            or any(op.recipe in kernels.get("recipes", {}) for op in members)):
        return outputs
    shapes = [dict(op.metadata).get("indexed_attention_geometry") for op in members]
    if any(shape is None or shape["window"] or shape["compression_ratio"] != 1
           or shape["cp"] != 1 for shape in shapes):
        return outputs
    common = [{key: value for key, value in shape.items() if key not in {"query", "context"}}
              for shape in shapes]
    if any(shape != common[0] for shape in common[1:]):
        return outputs
    sequences = tuple((shape["query"], shape["context"]) for shape in shapes)
    total_queries = sum(query for query, _ in sequences)
    meta = dict(members[0].metadata)
    # Active TND descriptors concatenate the query axis only. Context/indices
    # remain separate declarations; this descriptor is not a native ABI claim.
    if "attention_tensor_view" in meta:
        view = dict(meta["attention_tensor_view"])
        for key in ("query_shape", "query_rope_shape", "output_shape"):
            if key in view:
                view[key] = (total_queries, *view[key][1:])
        view["batch_count"] = len(sequences)
        meta["attention_tensor_view"] = view
    meta.update({
        "indexed_attention_geometry": {**common[0], "sequences": sequences,
                                       **_prefix_declaration(views, runtime)},
        "indexed_batch_boundary": "analytic_single_kernel_shared_query_owner_grid",
        "indexed_batch_members": tuple(op.recipe for op in members),
        "indexed_sequence_requests": tuple(view.request_id for view in views),
    })
    for key in ("logical_pairs", "kv_sweeps", "indexed_gather_rows",
                "dense_window_sweep_rows", "selection_index_read_bytes"):
        meta[key] = sum(dict(op.metadata).get(key, 0) for op in members)
    shape = (len(sequences), total_queries, max(q + c for q, c in sequences),
             common[0]["heads"], common[0]["latent"] + common[0]["rope"], common[0]["latent"])
    batched = replace(
        members[0], op_id=prefix + ".latent_attention_batch", recipe="latent_attention_batch",
        deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        flops=sum(op.flops for op in members), vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members), metadata=tuple(meta.items()))
    first = next(i for i, op in enumerate(ops) if op.op_id in ids)
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.insert(first, batched)
    return [batched.op_id]


def _prefix_declaration(views, runtime):
    """Declared shared-prefix extent and identity per request for deduplication."""
    shared = str(getattr(runtime, "prefix_cache_mode", "private") or "private")
    extents = tuple(
        max(0, min(int(getattr(view, "prefix_tokens", 0) or 0),
                   int(getattr(view, "context", 0) or 0)))
        for view in views)
    if shared != "shared" or not any(extents):
        return {"prefix_cache_mode": shared}
    keys = tuple(
        str(getattr(view, "prefix_cache_key", "") or "") for view in views)
    return {"prefix_cache_mode": shared, "prefix_tokens": extents,
            "prefix_cache_keys": keys}


def _align(value, size):
    return (value + size - 1) // size * size


def _phases(keys, heads, latent, rope, *, first, last, scratch_policy,
            operand_modes=False, flash_decode=False):
    """Physical tile work; padding is confined to matrix operands/results."""
    padded_h, padded_k = _align(heads, FRACTAL_ALIGNMENT), _align(keys, FRACTAL_ALIGNMENT)
    width = latent + rope
    query_bytes = heads * width * 2
    payload = keys * width * 2
    # Key and rope are separate, 32-byte-aligned row transfers. The reviewed
    # 512+64 geometry has no row-transaction amplification.
    transactions = keys * (_align(latent * 2, DMA_ROW_ALIGNMENT) + _align(rope * 2, DMA_ROW_ALIGNMENT))
    score = padded_h * padded_k * FP32_ELEMENT_BYTES
    probability = heads * padded_k * 2
    partial_output = padded_h * latent * FP32_ELEMENT_BYTES
    output = (
        heads * (latent + SPLIT_KV_STATE_SCALARS) * FP32_ELEMENT_BYTES
        if flash_decode
        else heads * latent * BF16_ELEMENT_BYTES
    ) if last else 0
    offchip = scratch_policy == "offchip"

    def scratch(read=0, write=0, *, vector=False, fixpipe=False):
        resource = "mte2_aiv" if vector else "mte2_aic"
        if not offchip:
            resource += "_cached"
        work = {resource: read} if read else {}
        if write:
            work["fixpipe" if fixpipe else "mte3"] = write
        if offchip:
            work["hbm"] = read + write
        else:
            if read:
                work["l2_read"] = read
            if write:
                work["l2_write"] = write
        return work

    def atomic(write, *, fixpipe=False):
        work = scratch(write=write, fixpipe=fixpipe)
        # Destination-side read/modify/write is not another MTE2 load.
        role = "hbm" if offchip else "l2_read"
        work[role] = work.get(role, 0) + write
        return work

    def operands(a, b):
        # L0A and L0B instructions share MTE1 but have different service rates.
        return ({"mte1_l0a": a, "mte1_l0b": b} if operand_modes
                else {"mte1": a + b})

    qk_inner = operands(padded_h * ((keys + (CUBE_PANEL_COLUMNS - 1)) // CUBE_PANEL_COLUMNS) * width * 2,
                        padded_k * width * 2)
    pv_prob_mult = 2 if flash_decode else 1
    pv_reads = keys * latent * 2 + pv_prob_mult * probability
    pv_inner = operands(pv_prob_mult * padded_h * (latent // CUBE_PANEL_COLUMNS) * padded_k * 2,
                        latent * padded_k * 2)
    return (
        ("gather", Work(resources={
            "hbm": transactions + keys * GATHER_INDEX_BYTES_PER_KEY,
            "mte2_aiv": transactions + keys * GATHER_INDEX_BYTES_PER_KEY})),
        ("publish", Work(resources=scratch(write=payload + 1024))),
        ("qk_load", Work(resources=scratch(read=payload)) + Work(resources={
            "hbm": query_bytes, "mte2_aic": query_bytes, **qk_inner})),
        ("qk", Work(matrix={"bf16": 2 * padded_h * padded_k * width})),
        ("scores", Work(resources=scratch(write=score, fixpipe=True))),
        ("softmax", Work(resources={
            **scratch(read=heads * padded_k * FP32_ELEMENT_BYTES + 1024, vector=True),
            "vector": SOFTMAX_OPS_PER_SCORE * heads * padded_k + SOFTMAX_STATE_OPS_PER_HEAD * heads})),
        ("probability", Work(resources=scratch(write=pv_prob_mult * probability))),
        ("pv_load", Work(resources={**scratch(read=pv_reads), **pv_inner})),
        # sparse_flash_attention_service_cube_mla.h:1339: FLASH_DECODE issues high + residual PV Mmad.
        ("pv", Work(matrix={"bf16": pv_prob_mult * 2 * padded_h * padded_k * latent})),
        ("nupdate", Work(resources={
            **(scratch(write=partial_output) if first else atomic(partial_output)),
            "vector": padded_h * (latent if first else OUTPUT_INIT_COLUMNS)})),
        ("partial", Work(resources=atomic(partial_output, fixpipe=True))),
        ("update", Work(resources={
            **scratch(read=heads * latent * FP32_ELEMENT_BYTES if last else 0, vector=True),
            "vector": heads * latent * OUTPUT_FINALIZE_OPS_PER_VALUE if last else 0})),
        ("output", Work(resources={"hbm": output, "mte3": output})),
    )


_FD_PHASES = partial(_phases, flash_decode=True)


def build(op, geometry, *, scratch_policy, key_tile=INDEXED_KEY_TILE, slots=INDEXED_PIPELINE_SLOTS, compact=True,
          operand_modes=False, rope0_analytic=False):
    meta = dict(op.metadata)
    shape = meta.get("indexed_attention_geometry")
    if not shape:
        raise ModelingError("indexed_attention_requires_frontend_geometry")
    if (scratch_policy not in {"resident", "offchip"} or key_tile != INDEXED_KEY_TILE
            or slots not in PRELOAD_SLOTS_VALID):
        raise ModelingError("indexed_attention_requires_declared_source_tile_and_scratch_policy")
    if (shape["window"] or shape["compression_ratio"] != 1
            or shape["cp"] != 1 or shape["query_dtype"] != "bf16"
            or shape["cache_dtype"] != "bf16"
            or shape.get("sparse_block_size", 1) != 1
            or (shape["latent"], shape["rope"]) not in (
                {(INDEXED_LATENT_DIM, INDEXED_ROPE_DIM), (INDEXED_LATENT_DIM, 0)} if rope0_analytic else {(INDEXED_LATENT_DIM, INDEXED_ROPE_DIM)})
            or not 1 <= shape["heads"] <= MAX_INDEXED_HEADS):
        raise ModelingError("indexed_attention_source_geometry_outside_reviewed_domain")
    extension = shape["rope"] == 0
    source = ({**SOURCE, "scope": "analytic latent512/rope0 geometry extension",
               "capture_binary": "not_established", "native_source_binding": False}
              if extension else SOURCE)
    plan = build_pipeline(shape, geometry, scratch_policy=scratch_policy,
                          key_tile=key_tile, slots=slots, compact=compact,
                          operand_modes=operand_modes, source_items=tuple(source.items()))
    if extension:
        return replace(plan, metadata={
            "kind": "indexed_attention_pipeline", "coverage": "analytic_geometry_extension",
            "geometry_extension": {"latent": INDEXED_LATENT_DIM, "rope": 0},
            "source_reference": SOURCE, "kernel_dispatch": "not_established",
            "unmodeled_features": ["captured_kernel_schedule", "scalar_index_response",
                                   "cross_core_flag_latency", "external_cache_eviction"],
        })
    return plan


def _check_resident_l2(geometry, scratch_policy, owners, per_owner_scratch):
    if scratch_policy == "resident":
        if (geometry.buffer_units.get("l2", 1) != 1
                or geometry.buffers.get("l2", 0) < owners * per_owner_scratch):
            raise ModelingError("indexed_attention_resident_scratch_requires_shared_cache_capacity")


def _build_split_kv_plan(
    shape, sequences, query, topk, heads, latent, rope, geometry,
    scratch_policy, key_tile, slots, compact, operand_modes, phase_factory,
    source_items, paired_vector_stages, vector_buffer_bytes, phase_unit_weights,
    per_seq_tuple, per_owner_scratch,
):
    """CANN >= 9.3 Arch22 split-KV partitioning along half-open 512-key tile boundaries."""
    pool_size = shape.get("index_pool_size", 1)
    pool_tail = shape.get("index_pool_tail", False)
    # sparse_flash_attention_tiling.cpp:414-418: split-KV requires s2Size >= sparseTokens and no sinks.
    sparse_tokens = max(per_seq_tuple) if per_seq_tuple is not None else int(topk)
    max_s2 = max((ctx_len + q_cnt for q_cnt, ctx_len in sequences if q_cnt > 0), default=0)
    if sparse_tokens <= 0 or max_s2 < sparse_tokens or shape.get("sinks"):
        return None
    n_tiles = (sparse_tokens + key_tile - 1) // key_tile
    p_cap = min(geometry.compute_units // query, n_tiles) if n_tiles > 1 else 1
    tiles_per_part = (n_tiles + p_cap - 1) // p_cap if p_cap > 1 else n_tiles
    parts = (n_tiles + tiles_per_part - 1) // tiles_per_part if p_cap > 1 else 1
    if parts <= 1:
        return None
    owner_tiles: dict[int, tuple[tuple[int, bool, bool], ...]] = {}
    merge_specs = []
    empty_rows = []
    q_flat = 0
    for seq_idx, (q_cnt, ctx_len) in enumerate(sequences):
        seq_topk = per_seq_tuple[seq_idx] if per_seq_tuple is not None else topk
        for pos in range(q_cnt):
            k_row = selected_tokens(ctx_len + pos + 1, seq_topk, pool_size, pool_tail)
            merge_owner = q_flat * parts
            if k_row == 0:
                empty_rows.append((q_flat, merge_owner))
            else:
                active_parts = 0
                for p_idx in range(parts):
                    k_start = ((n_tiles * p_idx) // parts) * key_tile
                    k_end = min(k_row, ((n_tiles * (p_idx + 1)) // parts) * key_tile)
                    pk = k_end - k_start
                    if pk > 0:
                        part_tiles = tuple(
                            (min(key_tile, pk - off), off == 0, off + min(key_tile, pk - off) == pk)
                            for off in range(0, pk, key_tile)
                        )
                        owner_tiles[merge_owner + p_idx] = part_tiles
                        active_parts += 1
                if active_parts > 0:
                    merge_specs.append((q_flat, active_parts, merge_owner))
            q_flat += 1
    active_owners = tuple(sorted(owner_tiles))
    _check_resident_l2(geometry, scratch_policy, len(active_owners), per_owner_scratch)
    fd_factory = _FD_PHASES if phase_factory is _phases else partial(phase_factory, flash_decode=True)

    @lru_cache(maxsize=TILE_CACHE_ENTRIES)
    def fd_phases(key):
        keys, first, last = key
        return fd_factory(keys, heads, latent, rope, first=first, last=last,
                          scratch_policy=scratch_policy, operand_modes=operand_modes)

    owner_work: dict[int, Work] = {}
    parts_work = Work()
    for owner in active_owners:
        w = Work()
        for key, count in Counter(owner_tiles[owner]).items():
            for _, phase in fd_phases(key):
                w += phase.scaled(count)
        owner_work[owner] = w
        parts_work += w

    state_bytes = heads * (latent + SPLIT_KV_STATE_SCALARS) * FP32_ELEMENT_BYTES
    row_output_bytes = heads * latent * BF16_ELEMENT_BYTES
    merge_work = Work()
    merge_per_q = []
    for q_idx, active_parts, merge_owner in merge_specs:
        # sfa_split_kv_merge.h:66-158: Mul+Add+stats over all active_parts (incl. split 0) + Cast.
        per_q = Work(resources={
            "mte2_aiv": active_parts * state_bytes,
            "mte3": row_output_bytes,
            "hbm": active_parts * state_bytes + row_output_bytes,
            "vector": (
                active_parts * heads * (SPLIT_KV_MERGE_OPS_PER_ELEM * latent + SPLIT_KV_MERGE_STAT_OPS_PER_PART)
                + max(0, active_parts - 1) * heads
                + heads * latent
            ),
        })
        merge_work += per_q
        merge_per_q.append((q_idx, merge_owner, per_q))
    empty_per_q = Work(resources={"hbm": row_output_bytes, "mte3": row_output_bytes})
    empty_work = empty_per_q.scaled(len(empty_rows)) if empty_rows else Work()
    nominal_cache: OrderedDict = OrderedDict()

    def _split_kv_lower(ctx):
        body_prog = _lower_indexed_attention(
            ctx, scratch_policy, operand_modes, nominal_cache, heads, phase_unit_weights,
            geometry.compute_units, tuple(paired_vector_stages), fd_phases, per_owner_scratch,
            vector_buffer_bytes, compact, owner_tiles.__getitem__, slots, owner_work, source_items,
            len(active_owners), key_tile, sequences, {o: 0 for o in active_owners}, row_output_bytes,
            owner_ids=active_owners,
        )
        branches = []
        for q_idx, merge_owner, per_q in merge_per_q:
            # sfa_split_kv_merge.h:44-46 splits heads across both AIV subblocks for heads >= 2.
            m_place = ctx.owned(merge_owner, geometry.compute_units, {"mte2_aiv", "mte3", "vector"})
            if heads == 1:
                for role in ("mte2_aiv", "mte3", "vector"):
                    res = ctx.resource_names.get(ctx.physical_role(role))
                    if res in m_place:
                        m_place[res] = m_place[res][:1]
            branches.append(KernelProgram((Stage(
                f"merge_q{q_idx}", ctx.bind(per_q), placement=m_place, coupled=True,
            ),)))
        e_work = ctx.bind(empty_per_q) if empty_rows else {}
        for q_idx, merge_owner in empty_rows:
            e_place = ctx.owned(merge_owner, geometry.compute_units, {"mte3"})
            if heads <= SINGLE_AIV_HEAD_LIMIT:
                mte3_res = ctx.resource_names[ctx.physical_role("mte3")]
                if mte3_res in e_place:
                    e_place[mte3_res] = e_place[mte3_res][:1]
            branches.append(KernelProgram((Stage(
                f"empty_q{q_idx}", e_work, placement=e_place, coupled=True,
            ),)))
        barrier_ms = p13_syncall_ms(ctx) if merge_per_q else 0.0
        stages = [Stage("split_kv_parts", program=body_prog)]
        prev = "split_kv_parts"
        if barrier_ms > 0.0:
            stages.append(Stage("split_kv_barrier", after=(prev,), latency=barrier_ms, keep_empty=True))
            prev = "split_kv_barrier"
        if branches:
            stages.append(Stage("split_kv_merge", after=(prev,), program=ParallelProgram(tuple(branches))))
        return KernelProgram(tuple(stages))

    return OperatorPlan(parts_work + merge_work + empty_work, _split_kv_lower)


def build_pipeline(shape, geometry, *, scratch_policy, key_tile=INDEXED_KEY_TILE, slots=INDEXED_PIPELINE_SLOTS,
                   compact=True, operand_modes=False, phase_factory=_phases,
                   source_items=tuple(SOURCE.items()), paired_vector_stages=(),
                   vector_buffer_bytes=0, phase_unit_weights=None):
    """Common finite query pipeline; callers own representation admission."""
    sequences = tuple(tuple(row) for row in shape.get(
        "sequences", ((shape.get("query", 0), shape.get("context", 0)),)))
    if (not sequences or any(len(row) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in row) for row in sequences)):
        raise ModelingError("indexed_attention_requires_integer_sequence_geometry")
    query, topk = sum(row[0] for row in sequences), shape["topk"]
    heads, latent, rope = shape["heads"], shape["latent"], shape["rope"]
    per_seq_topk = shape.get("per_sequence_topk")
    per_seq_tuple = (
        tuple(int(v) for v in per_seq_topk)
        if isinstance(per_seq_topk, (tuple, list)) and len(per_seq_topk) == len(sequences)
        else None
    )
    # The source reserves four KV rings and three score/probability/output
    # rings per owner. Cache residency is an explicit conditional placement.
    hpad = _align(heads, FRACTAL_ALIGNMENT)
    per_owner_scratch = (GATHER_SCRATCH_SLOTS * key_tile * (latent + rope) * 2
                         + INDEXED_PIPELINE_SLOTS * hpad * (key_tile * SCORE_AND_PROBABILITY_BYTES + latent * FP32_ELEMENT_BYTES))
    split_kv = bool(shape.get("split_kv", phase_factory is _phases))
    if split_kv and query > 0 and geometry.compute_units > query:
        split_plan = _build_split_kv_plan(
            shape, sequences, query, topk, heads, latent, rope, geometry,
            scratch_policy, key_tile, slots, compact, operand_modes, phase_factory,
            source_items, paired_vector_stages, vector_buffer_bytes, phase_unit_weights,
            per_seq_tuple, per_owner_scratch,
        )
        if split_plan is not None:
            return split_plan
    owners = selected_query_owners(query, geometry.compute_units)
    _check_resident_l2(geometry, scratch_policy, owners, per_owner_scratch)
    return _plan(sequences, topk, heads, latent, rope, geometry.compute_units,
                 scratch_policy, key_tile, slots, compact, operand_modes,
                 phase_factory, source_items, tuple(paired_vector_stages), vector_buffer_bytes,
                 phase_unit_weights, shape.get("index_pool_size", 1),
                 shape.get("index_pool_tail", False), per_seq_tuple)


def query_positions(owner, query, compute_units):
    """InitCalcParamsEach assigns contiguous ceil(Q/AIC) query intervals."""
    span = (query + compute_units - 1) // compute_units
    return range(owner * span, min(query, (owner + 1) * span))


def sequence_query_positions(owner, sequences, compute_units):
    """Flatten query ownership, never the independent sequence KV/context axes."""
    ends, total = [], 0
    for query, _ in sequences:
        total += query
        ends.append(total)
    for position in query_positions(owner, total, compute_units):
        sequence = bisect_right(ends, position)
        start = ends[sequence - 1] if sequence else 0
        yield sequence, position - start


def owner_row_tiles(owner, sequences, compute_units, topk, tile_size, *,
                    pool_size=1, pool_tail=False, per_sequence_topk=None):
    """Yield one AIC owner's contiguous query rows and their key tiles.

    The arch35 sparse kernel assigns every query row to one ``1c2v`` group and
    walks that row's selected keys in ``tile_size`` tiles. Ownership is the
    contiguous ceil(query/AIC) interval of the source parameter pass, so a
    caller sums exactly the tiles the engine will submit.
    """
    if tile_size < 1:
        raise ModelingError("indexed_attention_requires_positive_key_tile")
    for sequence, position in sequence_query_positions(owner, sequences, compute_units):
        context = sequences[sequence][1]
        seq_topk = (
            int(per_sequence_topk[sequence])
            if per_sequence_topk is not None and sequence < len(per_sequence_topk)
            else int(topk)
        )
        count = selected_tokens(context + position + 1, seq_topk, pool_size, pool_tail)
        tiles = []
        for start in range(0, count, tile_size):
            keys = min(tile_size, count - start)
            tiles.append((keys, start == 0, start + keys == count))
        yield sequence, tuple(tiles)


def bind_sustained_dram(context):
    """Price external DRAM at the declared sustained ceiling, not the raw peak.

    The profile publishes the sustainable DRAM service rate on the private
    MTE2 paths (``mte2_*_dram``).  External bytes still contend on the shared
    HBM bus, so the bus keeps its physical identity and only its declared
    service rate changes.
    """
    dram_rate = context.service_rate("mte2_aiv_dram")
    if dram_rate is None:
        return context
    name = context.resource_names.get("hbm")
    if name is None:
        return context
    ceiling = min(dram_rate, float(context.rates[name]) * MS_PER_SECOND)
    return replace(context, service_modes={
        **context.service_modes, "hbm": ("hbm", ceiling)})


@lru_cache(maxsize=PHASE_CACHE_ENTRIES)
def _plan(sequences, topk, heads, latent, rope, compute_units,
          scratch_policy, key_tile, slots, compact, operand_modes,
          phase_factory, source_items, paired_vector_stages, vector_buffer_bytes,
          phase_unit_weights, index_pool_size, index_pool_tail,
          per_sequence_topk=None):
    # Repeated model layers share geometry, never dependencies or Engine state.
    # Cache only bounded phase histograms and pure nominal timing calculations.
    query = sum(row[0] for row in sequences)
    owners = selected_query_owners(query, compute_units)
    per_owner_scratch = (GATHER_SCRATCH_SLOTS * key_tile * (latent + rope) * 2
                         + INDEXED_PIPELINE_SLOTS * _align(heads, FRACTAL_ALIGNMENT) * (key_tile * SCORE_AND_PROBABILITY_BYTES + latent * FP32_ELEMENT_BYTES))

    def tile_keys(owner):
        for sequence, position in sequence_query_positions(owner, sequences, compute_units):
            context = sequences[sequence][1]
            seq_topk = (
                int(per_sequence_topk[sequence])
                if per_sequence_topk is not None and sequence < len(per_sequence_topk)
                else int(topk)
            )
            count = selected_tokens(context + position + 1, seq_topk,
                                    index_pool_size, index_pool_tail)
            for start in range(0, count, key_tile):
                keys = min(key_tile, count - start)
                yield keys, start == 0, start + keys == count

    def empty_queries(owner):
        empty = 0
        for sequence, position in sequence_query_positions(owner, sequences, compute_units):
            context = sequences[sequence][1]
            seq_topk = (
                int(per_sequence_topk[sequence])
                if per_sequence_topk is not None and sequence < len(per_sequence_topk)
                else int(topk)
            )
            if selected_tokens(context + position + 1, seq_topk,
                               index_pool_size, index_pool_tail) == 0:
                empty += 1
        return empty

    @lru_cache(maxsize=TILE_CACHE_ENTRIES)
    def phases(key):
        keys, first, last = key
        return phase_factory(keys, heads, latent, rope, first=first, last=last,
                             scratch_policy=scratch_policy, operand_modes=operand_modes)

    owner_work = []
    empty_counts = []
    row_output_bytes = heads * latent * BF16_ELEMENT_BYTES
    for owner in range(owners):
        work = Work()
        for key, count in Counter(tile_keys(owner)).items():
            for _, phase in phases(key):
                work += phase.scaled(count)
        empty = empty_queries(owner)
        empty_counts.append(empty)
        if empty:
            empty_bytes = empty * row_output_bytes
            work += Work(resources={"hbm": empty_bytes, "mte3": empty_bytes})
        owner_work.append(work)
    empty_counts = tuple(empty_counts)
    total = Work()
    for work in owner_work:
        total += work

    nominal_cache = OrderedDict()

    def lower(ctx):
        return _lower_indexed_attention(
            ctx, scratch_policy, operand_modes, nominal_cache, heads, phase_unit_weights,
            compute_units, paired_vector_stages, phases, per_owner_scratch,
            vector_buffer_bytes, compact, tile_keys, slots, owner_work, source_items, owners,
            key_tile, sequences, empty_counts, row_output_bytes)

    return OperatorPlan(total, lower)


def _indexed_phase_sync_ms(ctx, name: str, keys: int) -> float:
    """Calibrated per-tile paired CV sync (P13) and Fixpipe panel setup (P03)."""
    sync_ms = p13_paired_ms(ctx) if name in INDEXED_PAIRED_SYNC_PHASES else 0.0
    fix_ms = p03_fixpipe_setup_ms(ctx)
    if fix_ms > 0.0:
        if name == "scores":
            sync_ms += ((keys + CUBE_PANEL_COLUMNS - 1) // CUBE_PANEL_COLUMNS) * fix_ms
        elif name == "partial":
            sync_ms += ((INDEXED_LATENT_DIM + CUBE_PANEL_COLUMNS - 1) // CUBE_PANEL_COLUMNS) * fix_ms
    return sync_ms


def _dequant_cast_scale(ctx) -> float:
    """Ratio of typed Cast+ALU dequant service to pure FP32 ALU service via P09."""
    cast_ratio = p09_cast_to_alu_op_ratio(ctx)
    if cast_ratio <= 0.0 or cast_ratio >= 1.0:
        return 1.0
    total = INDEXED_DEQUANT_NONCAST_OPS + INDEXED_DEQUANT_CAST_OPS
    return (INDEXED_DEQUANT_NONCAST_OPS + INDEXED_DEQUANT_CAST_OPS * cast_ratio) / total


def _compute_compact_nominal(
    ctx, owner, all_work, all_placement, bind, tile_keys, phases, slots,
    phase_caps, duration, _vector_weights, compute_units, empty_phase, empty_key,
):
    vec_res = ctx.resource_names.get("vector")
    dq_scale = _dequant_cast_scale(ctx)
    phase_times = {}
    stream_tail = {}
    retired = [0.0] * slots
    end = 0.0
    for count, key in enumerate(tile_keys(owner)):
        if key not in phase_times:
            times = []
            for name, phase in phases(key):
                work, placement = bind(phase, name, key)
                caps = phase_caps(work, name, key)
                eff_work = (
                    {r: (v * dq_scale if r == vec_res else v) for r, v in work.items()}
                    if (name == "dequantize" and dq_scale < 1.0 and vec_res in work)
                    else work
                )
                base_dur = max(
                    duration(eff_work, placement),
                    max((eff_work[r] / cap for r, cap in caps.items()), default=0.0),
                )
                times.append((name, base_dur + _indexed_phase_sync_ms(ctx, name, key[0])))
            phase_times[key] = tuple(times)
        ready = retired[count % slots]
        prior_partial = stream_tail.get("partial", 0)
        for name, elapsed in phase_times[key]:
            if name == "nupdate" and not key[1]:
                ready = max(ready, prior_partial)
            ready = max(ready, stream_tail.get(name, 0)) + elapsed
            stream_tail[name] = ready
        retired[count % slots] = ready
        end = max(end, ready)
    if empty_phase is not None:
        e_work, e_place = bind(empty_phase, "output", empty_key)
        e_caps = phase_caps(e_work, "output", empty_key)
        end += max(
            duration(e_work, e_place),
            max((e_work[r] / cap for r, cap in e_caps.items()), default=0.0),
        )
    unit_demand = Counter()
    eff_unit_demand = Counter()
    for key, count in Counter(tile_keys(owner)).items():
        for name, phase in phases(key):
            work, placement = bind(phase, name, key)
            weights = {ctx.resource_names[ctx.physical_role(role)]: tuple(
                value for value in fractions if value)
                for role, fractions in _vector_weights(name, key).items()}
            for resource, units in placement.items():
                raw_amt = work.get(resource, 0) * count
                eff_amt = (
                    raw_amt * dq_scale
                    if (name == "dequantize" and resource == vec_res and dq_scale < 1.0)
                    else raw_amt
                )
                fractions = weights.get(resource, (1 / len(units),) * len(units))
                for unit, fraction in zip(units, fractions):
                    unit_demand[resource, unit] += raw_amt * fraction
                    eff_unit_demand[resource, unit] += eff_amt * fraction
    if empty_phase is not None:
        e_work, e_place = bind(empty_phase, "output", empty_key)
        e_weights = {ctx.resource_names[ctx.physical_role(role)]: tuple(
            value for value in fractions if value)
            for role, fractions in _vector_weights("output", empty_key).items()}
        for resource, units in e_place.items():
            raw_amt = e_work.get(resource, 0)
            fractions = e_weights.get(resource, (1 / len(units),) * len(units))
            for unit, fraction in zip(units, fractions):
                unit_demand[resource, unit] += raw_amt * fraction
                eff_unit_demand[resource, unit] += raw_amt * fraction
    for (resource, _), amount in eff_unit_demand.items():
        role = next(role for role, name in ctx.resource_names.items()
                    if name == resource)
        end = max(end, amount * ctx.resource_units.get(
            role, compute_units) / ctx.rates[resource])
    eff_all_work = {
        r: (sum(v for (res, _), v in eff_unit_demand.items() if res == r)
            if r == vec_res and dq_scale < 1.0 else amt)
        for r, amt in all_work.items()
    }
    return max(end, duration(eff_all_work, all_placement)), dict(unit_demand)


def _compact_owner_program(
    ctx, owner, owner_work, bind, phase_unit_weights, heads, tile_keys, phases, nominal, slots,
    phase_caps, duration, _vector_weights, compute_units, buffers,
    empty_count, row_output_bytes,
):
    # Ordered phase recurrence with finite in-flight tiles. Shared
    # physical pipes impose an additional duty bound across streams.
    # Compact mode is a fluid approximation; the expanded control
    # retains exact phase arbitration for counterexample testing.
    all_work, all_placement = bind(owner_work[owner])
    empty_phase = (
        Work(resources={"hbm": empty_count * row_output_bytes,
                        "mte3": empty_count * row_output_bytes})
        if empty_count else None
    )
    empty_key = (0, True, True)
    # The aggregate may use only physical units used by its real phases
    # (including single-AIV empty outputs when <=16 heads and no tiles run).
    actual_units = {}
    for key in set(tile_keys(owner)):
        for name, phase in phases(key):
            _, placement = bind(phase, name, key)
            for resource, units in placement.items():
                actual_units.setdefault(resource, set()).update(units)
    if empty_phase is not None:
        _, placement = bind(empty_phase, "output", empty_key)
        for resource, units in placement.items():
            actual_units.setdefault(resource, set()).update(units)
    if actual_units:
        all_placement = {resource: tuple(sorted(units))
                         for resource, units in actual_units.items()}
    if owner not in nominal:
        nominal[owner] = _compute_compact_nominal(
            ctx, owner, all_work, all_placement, bind, tile_keys, phases, slots,
            phase_caps, duration, _vector_weights, compute_units, empty_phase, empty_key,
        )
    end, u_demand = nominal[owner]
    caps = {key: amount / end for key, amount in all_work.items()} if end else {}
    has_two_vector_aivs = len(all_placement.get(ctx.resource_names.get("vector", "vector"), ())) == 2
    if has_two_vector_aivs:
        branches = []
        for idx in range(2):
            b_work = {}
            b_place = {}
            for res, amt in all_work.items():
                units = all_placement.get(res, ())
                if len(units) == 2:
                    share = u_demand.get((res, units[idx]), 0.0)
                    if share > 0:
                        b_work[res] = share
                        b_place[res] = (units[idx],)
                else:
                    share = amt / 2.0
                    if share > 0:
                        b_work[res] = share
                        if units:
                            b_place[res] = units
            b_caps = {k: v / end for k, v in b_work.items()} if end else {}
            branches.append(KernelProgram((Stage(
                f"aiv{idx}", b_work, caps=b_caps, placement=b_place, coupled=True),)))
        pipe_stage = Stage("pipeline", after=("reserve",), program=ParallelProgram(tuple(branches)))
    else:
        pipe_stage = Stage("pipeline", all_work, after=("reserve",), caps=caps,
                           placement=all_placement, coupled=True)
    return KernelProgram((
        Stage("reserve", acquire=buffers),
        pipe_stage,
        Stage("release", after=("pipeline",), free=buffers)))


def _lower_indexed_owner(
    ctx, owner, compute_units, heads, paired_vector_stages, phase_unit_weights, _vector_weights,
    phases, m0, hpad, aiv_read_role, scratch_policy, per_owner_scratch, vector_buffer_bytes,
    compact, tile_keys, slots, owner_work, nominal, empty_count, row_output_bytes,
):
    def bind(phase, name=None, key=None):
        work = ctx.bind(phase)
        roles = set(phase.resources) | ({"cube"} if phase.matrix else set())
        placement = ctx.owned(
            owner, compute_units, roles - {"hbm", "l2_read", "l2_write"})
        # Gather splits selected KV rows across both AIVs. For <=16
        # heads the source gives all softmax/output rows to AIV0; for
        # >16 heads vecDealM splits softmax/output rows across AIV0/AIV1.
        single_roles = (set() if heads > SINGLE_AIV_HEAD_LIMIT or name in paired_vector_stages
                        or (name is None and paired_vector_stages) else {"vector"})
        if heads <= SINGLE_AIV_HEAD_LIMIT and name not in {None, "gather", "publish"}:
            single_roles |= {"mte2_aiv", "mte2_aiv_cached", "mte3"}
        for role in single_roles & roles:
            resource = ctx.resource_names[ctx.physical_role(role)]
            placement[resource] = placement[resource][:1]
        if (phase_unit_weights or heads > SINGLE_AIV_HEAD_LIMIT) and key is not None:
            for role, fractions in _vector_weights(name, key).items():
                resource = ctx.resource_names[ctx.physical_role(role)]
                if resource in placement:
                    placement[resource] = tuple(
                        unit for unit, fraction in zip(placement[resource], fractions)
                        if fraction)
        return work, placement

    def phase_caps(work, name, key):
        # An uneven local partition has the supply of its busiest
        # physical unit; it cannot borrow an inactive sibling's pipe.
        result = {}
        if phase_unit_weights or heads > SINGLE_AIV_HEAD_LIMIT:
            for role, fractions in _vector_weights(name, key).items():
                resource = ctx.resource_names[ctx.physical_role(role)]
                if work.get(resource, 0):
                    result[resource] = ctx.rates[resource] / (
                        ctx.resource_units[ctx.physical_role(role)] * max(fractions))
        return result

    def duration(work, placement):
        limits = {}
        for resource, amount in work.items():
            units = len(placement.get(resource, ()))
            role = next(role for role, name in ctx.resource_names.items()
                        if name == resource)
            fraction = units / ctx.resource_units.get(role, compute_units) if units else 1
            limits[resource] = amount / (ctx.rates[resource] * fraction)
        return max(limits.values(), default=0)

    def _output_stage(stage_name, work, placement, *, after, previous=(), caps=None):
        if heads > SINGLE_AIV_HEAD_LIMIT and work:
            first = m0 / heads
            branches = []
            for idx, frac in enumerate((first, 1.0 - first)):
                if not frac:
                    continue
                b_place = dict(placement)
                for role in ("vector", aiv_read_role, "mte3"):
                    res = ctx.resource_names.get(ctx.physical_role(role))
                    if res in b_place and len(b_place[res]) > idx:
                        b_place[res] = (b_place[res][idx],)
                branches.append(KernelProgram((Stage(
                    "aiv", {k: v * frac for k, v in work.items()},
                    placement=b_place, coupled=True),)))
            return Stage(stage_name, after=after, previous=previous,
                         program=ParallelProgram(tuple(branches)))
        return Stage(stage_name, work, after=after, previous=previous,
                     placement=placement, caps=caps or {}, coupled=True)

    def stage_list(key):
        result = []
        partial_tail = (
            "partial_sync" if _indexed_phase_sync_ms(ctx, "partial", key[0]) > 0.0 else "partial"
        )
        for name, phase in phases(key):
            work, placement = bind(phase, name, key)
            after = (result[-1].name,) if result else ()
            sync_ms = _indexed_phase_sync_ms(ctx, name, key[0])
            prev = (partial_tail,) if name == "nupdate" and not key[1] else ()
            if sync_ms > 0.0:
                prev = (*prev, f"{name}_sync")
            if heads > SINGLE_AIV_HEAD_LIMIT and name in {"softmax", "probability", "nupdate", "update", "output"} and work:
                first = m0 / hpad if name == "nupdate" else m0 / heads
                branches = []
                for idx, frac in enumerate((first, 1.0 - first)):
                    if not frac:
                        continue
                    b_place = dict(placement)
                    for role in ("vector", aiv_read_role, "mte3"):
                        res = ctx.resource_names.get(ctx.physical_role(role))
                        if res in b_place and len(b_place[res]) > idx:
                            b_place[res] = (b_place[res][idx],)
                    branches.append(KernelProgram((Stage(
                        "aiv", {k: v * frac for k, v in work.items()},
                        placement=b_place, coupled=True),)))
                result.append(Stage(
                    name, after=after, previous=prev,
                    program=ParallelProgram(tuple(branches))))
            else:
                result.append(Stage(
                    name, work, after=after, previous=prev,
                    placement=placement, caps=phase_caps(work, name, key), coupled=True))
            if sync_ms > 0.0:
                result.append(Stage(
                    f"{name}_sync", after=(name,), latency=sync_ms, keep_empty=True))
        return tuple(result)

    buffers = ()
    if scratch_policy == "resident":
        buffers = (ctx.allocate(f"scratch{owner}", "l2", per_owner_scratch, (0,)),)
    if vector_buffer_bytes:
        from mindie_sim.prediction.ownership import unit_group
        buffers += (ctx.allocate(
            f"vector_scratch{owner}", "ub", vector_buffer_bytes,
            unit_group(owner, compute_units, ctx.pool_units["ub"])),)
    if not compact:
        from mindie_sim.prediction.program import PipelineProgram
        # Bounded verification control; input phases are regenerated
        # lazily, so even this path retains no full task history.
        keys_seq = tuple(tile_keys(owner))
        count = len(keys_seq)
        program = PipelineProgram(
            count, slots, lambda idx, keys_seq=keys_seq: stage_list(keys_seq[idx]))
        stages = [
            Stage("reserve", acquire=buffers),
            Stage("tiles", after=("reserve",), program=program),
        ]
        tail = "tiles"
        if empty_count:
            empty_phase = Work(resources={
                "hbm": empty_count * row_output_bytes,
                "mte3": empty_count * row_output_bytes,
            })
            e_work, e_place = bind(empty_phase, "output", (0, True, True))
            stages.append(_output_stage(
                "empty_output", e_work, e_place, after=(tail,),
                caps=phase_caps(e_work, "output", (0, True, True))))
            tail = "empty_output"
        stages.append(Stage("release", after=(tail,), free=buffers))
        return KernelProgram(tuple(stages))

    return _compact_owner_program(
        ctx, owner, owner_work, bind, phase_unit_weights, heads, tile_keys, phases, nominal,
        slots, phase_caps, duration, _vector_weights, compute_units, buffers,
        empty_count, row_output_bytes)


def _lower_indexed_attention(
    ctx, scratch_policy, operand_modes, nominal_cache, heads, phase_unit_weights, compute_units,
    paired_vector_stages, phases, per_owner_scratch, vector_buffer_bytes, compact, tile_keys,
    slots, owner_work, source_items, owners, key_tile, sequences, empty_counts, row_output_bytes,
    *, owner_ids=None,
):
    # Copy latency/issue limits are deliberately not guessed from residuals.
    required = {"mte2_aic", "mte2_aiv", "mte1", "mte3", "fixpipe"}
    if scratch_policy == "resident":
        required |= {"mte2_aic_cached", "mte2_aiv_cached", "l2_read", "l2_write"}
    for role in required:
        if ctx.physical_role(role) not in ctx.resource_names:
            raise ModelingError(f"indexed_attention_physical_copy_rate_required:{role}")
    if operand_modes and any(ctx.physical_role(role) != "mte1"
                             for role in ("mte1_l0a", "mte1_l0b")):
        raise ModelingError("indexed_attention_operands_must_share_mte1")

    signature = (
        *tuple(tuple(sorted(mapping.items())) for mapping in (
            ctx.rates, ctx.resource_names, ctx.resource_units,
            ctx.service_modes, ctx.compute_peaks)),
        p13_paired_ms(ctx), p03_fixpipe_setup_ms(ctx), _dequant_cast_scale(ctx),
    )
    if signature not in nominal_cache:
        nominal_cache[signature] = {}
        if len(nominal_cache) > 4:
            nominal_cache.popitem(last=False)
    nominal_cache.move_to_end(signature)
    nominal = nominal_cache[signature]

    hpad = _align(heads, FRACTAL_ALIGNMENT)
    m0 = (((heads + (FRACTAL_ALIGNMENT - 1)) // FRACTAL_ALIGNMENT + 1) // 2) * FRACTAL_ALIGNMENT
    m1 = max(0, heads - m0)
    aiv_read_role = "mte2_aiv" if scratch_policy == "offchip" else "mte2_aiv_cached"

    def _vector_weights(name, key):
        base = dict(phase_unit_weights(name, key)) if phase_unit_weights else {}
        if heads > SINGLE_AIV_HEAD_LIMIT and name in {"softmax", "probability", "nupdate", "update", "output"}:
            fracs = (m0 / hpad, _align(m1, FRACTAL_ALIGNMENT) / hpad) if name == "nupdate" else (m0 / heads, m1 / heads)
            for role in ("vector", aiv_read_role, "mte3"):
                base.setdefault(role, fracs)
        return base

    def lower_owner(owner):
        return _lower_indexed_owner(
            ctx, owner, compute_units, heads, paired_vector_stages, phase_unit_weights,
            _vector_weights, phases, m0, hpad, aiv_read_role, scratch_policy,
            per_owner_scratch, vector_buffer_bytes, compact, tile_keys, slots, owner_work,
            nominal, empty_counts[owner], row_output_bytes)

    if ctx.record_mapping:
        ctx.record_mapping({
            "program": "indexed_attention_pipeline", "source": dict(source_items),
            "scratch_policy": scratch_policy,
            "scratch_bytes": owners * per_owner_scratch, "key_tile": key_tile,
            "sparse_block_size": 1,
            "owners": owners, "producer_slots": slots, "compact": compact,
            "query_schedule": "contiguous_ceil_query_over_aic",
            "sequences": sequences,
            "batch_boundary": ("analytic_shared_query_owner_grid" if len(sequences) > 1
                               else "single_sequence"),
            "source_binding": ("analytic_batch_adaptation; capture_dispatch_not_established"
                               if len(sequences) > 1 else "declared_single_sequence_pipeline"),
            "operand_modes": operand_modes,
            "abstraction": "finite stage recurrence with shared-pipe duty bound and fluid contention",
            "unmodeled": ["scalar_index_response", "cross_core_flag_latency",
                          "external_cache_eviction", "atomic_response_latency"],
            "output_accumulation": "integer_bias_update_then_atomic_fixpipe; final_vector_read_once",
            "vector_ownership": (
                "two_gather_AIVs; one_softmax_and_output_AIV"
                if heads <= SINGLE_AIV_HEAD_LIMIT else "two_gather_AIVs; two_softmax_and_output_AIVs"
            ),
            "paired_vector_stages": paired_vector_stages,
            "vector_buffer_bytes_per_aiv": vector_buffer_bytes,
        })
    target_owners = range(owners) if owner_ids is None else owner_ids
    return ParallelProgram(tuple(lower_owner(owner) for owner in target_owners))
