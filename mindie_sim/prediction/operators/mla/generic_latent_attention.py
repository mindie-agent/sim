"""Finite generic latent-attention programs and invocation batching.

[Role] Price dense latent decode through an online attention schedule and
  batch compatible latent attention requests into one invocation.
[Invariants] A private query owner retains Q and serially updates its online
  softmax state and FP32 accumulators over finite key tiles before final store.
  A batched invocation preserves each request's work and dependencies.
[Timeouts & Locks] Finite scans of requests, owners and key tiles.
[Side-Effect Policy] Builds plans and rewrites caller-owned operators during
  batching. No I/O, locks or device work.
"""
from __future__ import annotations

from dataclasses import replace
import math as math

from mindie_sim.catalog.attention_common import (
    _causal_pairs as _causal_pairs,
    _long_selected_pairs as _long_selected_pairs,
    active_dcp_degree as active_dcp_degree,
    min_positive as min_positive,
)
from mindie_sim.catalog.cache import (
    completed_rows as completed_rows,
    index_row_bytes as index_row_bytes,
    packed_region_row_bytes as packed_region_row_bytes,
)
from mindie_sim.catalog.cache_format import (
    CacheFormat as CacheFormat,
    cache_format as cache_format,
)
from mindie_sim.catalog.context_parallel import (
    dsa_cp_token_ranges as dsa_cp_token_ranges,
    owned_positions as owned_positions,
    split_causal_pairs as split_causal_pairs,
    split_pcp_pairs as split_pcp_pairs,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes

from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import RECIPE_NAME as BATCH_INDEXER_RECIPE
from mindie_sim.prediction.engine import Allocation
from .generic_latent_attention_views import (
    BF16_OPERAND_FORMAT as BF16_OPERAND_FORMAT,
    CANDIDATE_BYTES as CANDIDATE_BYTES,
    DEFAULT_BLOCK_SIZE as DEFAULT_BLOCK_SIZE,
    EMPTY_INDEX_ROWS as EMPTY_INDEX_ROWS,
    FP32_BYTES as FP32_BYTES,
    INDEX_IDENTITY_COMPRESSION as INDEX_IDENTITY_COMPRESSION,
    INDEX_NO_ROWS as INDEX_NO_ROWS,
    INDEX_QUERY_TILE as INDEX_QUERY_TILE,
    INT32_BYTES as INT32_BYTES,
    MIN_INDEX_FACTOR as MIN_INDEX_FACTOR,
    MIN_LOG_TOPK as MIN_LOG_TOPK,
    PACKED_INDEX_ARITHMETIC_DTYPE as PACKED_INDEX_ARITHMETIC_DTYPE,
    UNQUANTIZED_INDEX_DTYPES as UNQUANTIZED_INDEX_DTYPES,
    _index_owner_rows as _index_owner_rows,
    attention_tensor_view as attention_tensor_view,
    attention_view_record as attention_view_record,
    dsa_cp_prefill_parts as dsa_cp_prefill_parts,
    index_arithmetic_dtype as index_arithmetic_dtype,
    index_cache_dtype as index_cache_dtype,
    index_cache_format as index_cache_format,
    index_cache_is_packed as index_cache_is_packed,
    index_key_window as index_key_window,
    index_operands_quantized as index_operands_quantized,
    kv_sweeps as kv_sweeps,
    owned_completed_rows as owned_completed_rows,
    owned_store_rows as owned_store_rows,
    selected_pairs as selected_pairs,
    semantic_index_selection as semantic_index_selection,
    sharded_index_visible_rows as sharded_index_visible_rows,
    sparse_over_parts as sparse_over_parts,
    stack_compressor_kind as stack_compressor_kind,
    unique_index_key_rows as unique_index_key_rows,
    view_causal_pairs as view_causal_pairs,
    visible_parts as visible_parts,
    window_cache_read as window_cache_read,
    window_tile_rows as window_tile_rows,
)
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from .mla_structure import declare_indexer as declare_indexer
from mindie_sim.prediction.ownership import read_role, unit_group as unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage

KEY_TILE = 128
MIN_BATCH_MEMBERS = 2
_DEFAULT_QUERY_KEY_DIM = 576
_DEFAULT_LATENT_DIM = 512
_CUBE_HEAD_ALIGNMENT = 16
_MAX_PV_OPERAND_BYTES = 64 * 1024
_QK_VECTOR_OPS_PER_HEAD_PAIR = 4
_ONLINE_VECTOR_OPS_PER_HEAD_PAIR = 8
_PV_VECTOR_OPS_PER_HEAD_PAIR = 2
_UPDATE_VECTOR_OPS_PER_OUTPUT_ELEMENT = 4


def _schedule(op):
    raw = dict(op.metadata).get("latent_attention_schedule")
    if not isinstance(raw, dict):
        raise ModelingError("generic_latent_schedule_metadata_required")
    heads = int(raw["heads"])
    qk_dim = int(raw.get("qk_dim", _DEFAULT_QUERY_KEY_DIM))
    latent_dim = int(raw.get("latent_dim", _DEFAULT_LATENT_DIM))
    dv = int(raw.get("dv", latent_dim))
    kv_row = int(raw["kv_row_bytes"])
    if min(heads, qk_dim, latent_dim, dv, kv_row) <= 0:
        raise ModelingError("generic_latent_schedule_geometry_invalid")
    sequences = raw.get("sequences")
    if sequences is None:
        sequences = ((int(raw.get("query_rows", 1)), int(raw.get("keys", 0))),)
    sequences = tuple((int(q), int(keys)) for q, keys in sequences)
    if not sequences or any(q < 0 or keys < 0 for q, keys in sequences):
        raise ModelingError("generic_latent_schedule_sequence_invalid")
    active = tuple((q, keys) for q, keys in sequences if q > 0)
    if not active:
        return None
    return heads, qk_dim, latent_dim, dv, kv_row, active


def _alloc(context, core, cores, tag, heads, q, q_bytes, kv_tile, latent_dim, dv, qk_dim):
    m_rows = q * heads
    ham = ((m_rows + _CUBE_HEAD_ALIGNMENT - 1) // _CUBE_HEAD_ALIGNMENT) * _CUBE_HEAD_ALIGNMENT
    pv_keys = max(1, min(KEY_TILE, _MAX_PV_OPERAND_BYTES // max(1, latent_dim * 2)))
    pv_l0 = pv_keys * latent_dim * 2
    specs = (
        ("q_l1", "l1", q_bytes),
        ("kv_l1", "l1", kv_tile),
        ("acc_ub", "ub", m_rows * dv * FP32_BYTES + m_rows * KEY_TILE * FP32_BYTES),
        ("qk_l0a", "l0a", ham * min(KEY_TILE, max(1, qk_dim)) * 2),
        ("pv_l0b", "l0b", pv_l0),
        ("out_l0c", "l0c", ham * dv * FP32_BYTES),
    )
    items = []
    for label, pool, size in specs:
        if pool not in context.pool_names or size <= 0:
            continue
        p_units = max(1, int(context.pool_units.get(pool, cores)))
        items.append(Allocation(f"{label}_{tag}", context.pool_names[pool], size,
                                (core % p_units,)))
    return tuple(items)


def _item_stages(context, core, cores, tag, heads, qk_dim, latent_dim, dv,
                 q, keys, kv_row, q_bytes, out_bytes, *, after=()):
    read = read_role(context.resource_names)
    place = context.owned(core, cores, (read, "vector", "mte3", "cube"))
    if keys == 0:
        fin = f"zero_out{tag}"
        return [Stage(fin, context.bind(Work(
            resources={"hbm": out_bytes, "mte3": out_bytes})),
            after=after, placement=place)], fin
    kv_tile = min(KEY_TILE, keys) * kv_row
    alloc = _alloc(context, core, cores, tag, heads, q, q_bytes, kv_tile, latent_dim, dv, qk_dim)
    m_rows = q * heads
    ham = ((m_rows + _CUBE_HEAD_ALIGNMENT - 1) // _CUBE_HEAD_ALIGNMENT) * _CUBE_HEAD_ALIGNMENT

    def tile(index):
        start = index * KEY_TILE
        count = min(KEY_TILE, keys - start)
        load = count * kv_row
        qk_i = 2 * count * ham * qk_dim
        pv_i = 2 * count * ham * latent_dim
        return (
            Stage(f"key_load{tag}_{index}", context.bind(Work(
                resources={"hbm": load, read: load})), placement=place),
            Stage(f"qk{tag}_{index}", context.bind(Work(
                matrix={"bf16": qk_i},
                resources={"vector": _QK_VECTOR_OPS_PER_HEAD_PAIR * count * m_rows})),
                  after=(f"key_load{tag}_{index}",), placement=place),
            Stage(f"online{tag}_{index}", context.bind(Work(
                resources={"vector": _ONLINE_VECTOR_OPS_PER_HEAD_PAIR * count * m_rows
                           + 2 * m_rows})),
                  after=(f"qk{tag}_{index}",), placement=place),
            Stage(f"pv{tag}_{index}", context.bind(Work(
                matrix={"bf16": pv_i},
                resources={"vector": _PV_VECTOR_OPS_PER_HEAD_PAIR * count * m_rows})),
                  after=(f"online{tag}_{index}",), placement=place),
            Stage(f"update{tag}_{index}", context.bind(Work(
                resources={"vector": _UPDATE_VECTOR_OPS_PER_OUTPUT_ELEMENT * m_rows * dv})),
                  after=(f"pv{tag}_{index}",), placement=place),
        )

    stages = [Stage(f"alloc{tag}", acquire=alloc, after=after, placement=place),
              Stage(f"q_load{tag}", context.bind(Work(
                  resources={"hbm": q_bytes, read: q_bytes})),
                    after=(f"alloc{tag}",), placement=place)]
    previous = f"q_load{tag}"
    for index in range((keys + KEY_TILE - 1) // KEY_TILE):
        for stage in tile(index):
            # The rolling KV slot may not be overwritten before the preceding
            # update consumed it; each tile's four stages stay causal.
            stages.append(replace(stage, after=(previous,)))
            previous = stage.name
    fin = f"finalize{tag}"
    stages.append(Stage(fin, context.bind(Work(
        resources={"vector": 2 * m_rows * dv,
                   "hbm": out_bytes, "mte3": out_bytes})),
        after=(previous,), placement=place, free=alloc))
    return stages, fin


def _max_heads_per_owner(heads, q, keys, qk_dim, latent_dim, dv, kv_row, geometry):
    buffers = dict(getattr(geometry, "buffers", {}) or {})
    alignment = _CUBE_HEAD_ALIGNMENT

    def cap(pool):
        if pool not in buffers:
            return None
        return int(buffers[pool])

    limit = heads
    ub_cap = cap("ub")
    if ub_cap is not None:
        ub_per_head = q * (dv + KEY_TILE) * FP32_BYTES
        limit = min(limit, max(1, ub_cap // max(1, ub_per_head)))
    l0c_cap = cap("l0c")
    if l0c_cap is not None:
        l0c_per_row = dv * FP32_BYTES
        max_rows = (l0c_cap // max(1, l0c_per_row) // alignment) * alignment
        if l0c_cap < alignment * l0c_per_row:
            max_rows = max(1, l0c_cap // max(1, l0c_per_row))
        limit = min(limit, max(1, max_rows // max(1, q)))
    l0a_cap = cap("l0a")
    if l0a_cap is not None:
        qk_panel = min(KEY_TILE, max(1, qk_dim)) * 2
        max_rows = (l0a_cap // max(1, qk_panel) // alignment) * alignment
        if l0a_cap < alignment * qk_panel:
            max_rows = max(1, l0a_cap // max(1, qk_panel))
        limit = min(limit, max(1, max_rows // max(1, q)))
    l1_cap = cap("l1")
    kv_tile = min(KEY_TILE, keys) * kv_row
    if l1_cap is not None and l1_cap > kv_tile:
        limit = min(limit, max(1, (l1_cap - kv_tile) // max(1, q * qk_dim * 2)))
    if limit >= heads:
        return heads
    return (max(alignment, (limit // alignment) * alignment)
            if limit >= alignment else max(1, limit))


def build(op, geometry):
    parsed = _schedule(op)
    if parsed is None:
        return OperatorPlan(Work(), lambda context: KernelProgram(()),
                            metadata={"generic_latent_attention": {"idle": 1}})
    heads, qk_dim, latent_dim, dv, kv_row, sequences = parsed
    declared_roles = set(getattr(geometry, "resource_units", ()) or ()) or {"mte2"}
    read = read_role(declared_roles)
    total = Work()
    logical = issued = 0
    total_q = total_kv = total_out = 0
    alignment = _CUBE_HEAD_ALIGNMENT
    vector_ops_per_pair = (_QK_VECTOR_OPS_PER_HEAD_PAIR + _ONLINE_VECTOR_OPS_PER_HEAD_PAIR
                           + _PV_VECTOR_OPS_PER_HEAD_PAIR)
    owner_items = []
    for q, keys in sequences:
        if keys == 0:
            out_bytes = q * heads * latent_dim * 2
            owner_items.append((q, 0, heads))
            total_out += out_bytes
            total += Work(resources={"hbm": out_bytes, "mte3": out_bytes})
            continue
        head_step = _max_heads_per_owner(heads, q, keys, qk_dim, latent_dim, dv, kv_row, geometry)
        for h_start in range(0, heads, head_step):
            h_cur = min(head_step, heads - h_start)
            m_cur = q * h_cur
            owner_items.append((q, keys, h_cur))
            q_bytes = m_cur * qk_dim * 2
            out_bytes = m_cur * latent_dim * 2
            total_q += q_bytes
            total_kv += keys * kv_row
            total_out += out_bytes
            logical += 2 * keys * m_cur * (qk_dim + latent_dim)
            ham = ((m_cur + alignment - 1) // alignment) * alignment
            issued += 2 * keys * ham * (qk_dim + latent_dim)
            total += Work(resources={"hbm": q_bytes, read: q_bytes})
            total += Work(resources={"hbm": keys * kv_row, read: keys * kv_row})
            total += Work(matrix={"bf16": 2 * keys * ham * (qk_dim + latent_dim)})
            owners = (keys + KEY_TILE - 1) // KEY_TILE
            vector = (vector_ops_per_pair * keys * m_cur + 2 * m_cur * owners
                      + _UPDATE_VECTOR_OPS_PER_OUTPUT_ELEMENT * m_cur * dv * owners
                      + 2 * m_cur * dv)
            total += Work(resources={"vector": vector})
            total += Work(resources={"hbm": out_bytes, "mte3": out_bytes})

    def program(context):
        cores = max(1, int(context.core_count))
        per_core = []
        for core in range(min(cores, len(owner_items))):
            chain = []
            prev = ()
            for tag in range(core, len(owner_items), cores):
                q, keys, h_cur = owner_items[tag]
                stages, fin = _item_stages(
                    context, core, cores, tag, h_cur, qk_dim, latent_dim, dv,
                    q, keys, kv_row, q * h_cur * qk_dim * 2, q * h_cur * latent_dim * 2,
                    after=prev)
                chain.extend(stages)
                prev = (fin,)
            per_core.append(KernelProgram(tuple(chain)))
        return KernelProgram((Stage("owners", program=ParallelProgram(tuple(per_core))),))

    return OperatorPlan(
        total, program,
        metadata={"generic_latent_attention": {
            "heads": heads, "batch": len(sequences), "sequences": sequences,
            "key_tile": KEY_TILE, "qk_dim": qk_dim, "latent_dim": latent_dim,
            "dv": dv, "kv_row_bytes": kv_row,
            "q_bytes": total_q, "kv_unique_bytes": total_kv,
            "output_bytes": total_out, "logical_flops": logical,
            "issued_flops": issued,
            "source_scope": "generic_one_query_owner_online_qk_softmax_pv",
        }},
    )


def batch_ops(ops, outputs, *, runtime, views, prefix):
    """Aggregate same-DP dense decode requests into one structural invocation."""
    if runtime is None or len(outputs) < MIN_BATCH_MEMBERS or len(outputs) != len(views):
        return outputs
    if runtime.parallel.pcp != 1 and any(v.phase == "prefill" for v in views):
        return outputs
    policy = getattr(runtime, "attention_policy", None)
    if policy is not None:
        fused = (policy.get("fused_batch") if hasattr(policy, "get")
                 else getattr(policy, "fused_batch", None))
        if fused is False:
            return outputs
    if "attention" in runtime.kernels.get("families", {}):
        return outputs
    by_id = {op.op_id: op for op in ops}
    decode_pairs = [
        (v, by_id[out_id])
        for v, out_id in zip(views, outputs)
        if v.q_len == 1 and v.phase == "decode" and out_id in by_id
        and dict(by_id[out_id].metadata).get("program_kind") == "dense_latent_attention_generic"
        and not by_id[out_id].plan_required
        and by_id[out_id].recipe not in runtime.kernels.get("recipes", {})
    ]
    if len(decode_pairs) < 2:
        return outputs
    decode_views = [v for v, _ in decode_pairs]
    members = [op for _, op in decode_pairs]
    ids = {op.op_id for op in members}
    schedules = [dict(op.metadata).get("latent_attention_schedule") for op in members]
    if any(item is None for item in schedules):
        return outputs
    common = [{key: value for key, value in item.items()
               if key not in {"query_rows", "keys", "sequences", "q_bytes",
                              "output_bytes", "kv_unique_bytes", "owners", "batch"}}
              for item in schedules]
    if any(item != common[0] for item in common[1:]):
        return outputs
    sequences = tuple((int(v.q_len), int(item["keys"]))
                      for v, item in zip(decode_views, schedules))
    meta = dict(members[0].metadata)
    schedule = dict(meta["latent_attention_schedule"])
    activation_bytes = dtype_bytes(members[0].dtype or "bf16")
    qk_width = int(schedule["qk_dim"])
    latent_width = int(schedule["latent_dim"])
    schedule.update({"sequences": sequences, "batch": len(sequences),
                     "query_rows": sum(q for q, _ in sequences),
                     "keys": max(keys for _, keys in sequences),
                     "owners": len(sequences),
                     "kv_unique_bytes": sum(keys * schedule["kv_row_bytes"]
                                            for _, keys in sequences),
                     "q_bytes": sum(q * schedule["heads"] * qk_width * activation_bytes
                                    for q, _ in sequences),
                     "output_bytes": sum(q * schedule["heads"] * latent_width * activation_bytes
                                         for q, _ in sequences)})
    meta["latent_attention_schedule"] = schedule
    view = dict(meta.get("attention_tensor_view", {}))
    if view:
        view["query_shape"] = (sum(q for q, _ in sequences), schedule["heads"],
                               qk_width)
        view["output_shape"] = (schedule["heads"], sum(q for q, _ in sequences),
                                latent_width)
        view["batch_count"] = len(sequences)
        meta["attention_tensor_view"] = view
    meta["latent_batch_boundary"] = "generic_single_invocation_private_query_owners"
    meta["latent_sequence_requests"] = tuple(v.request_id for v in decode_views)
    meta["latent_batch_members"] = tuple(op.recipe for op in members)
    shape = (len(sequences), schedule["heads"], qk_width)
    batched = replace(
        members[0], op_id=prefix + ".latent_attention_batch",
        recipe="latent_attention0",
        requests=tuple(dict.fromkeys(req for op in members for req in op.requests)),
        deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        flops=sum(op.flops for op in members), vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members), metadata=tuple(meta.items()))
    first = next(i for i, op in enumerate(ops) if op.op_id in ids)
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.insert(first, batched)
    new_outputs = []
    inserted = False
    for out_id in outputs:
        if out_id in ids:
            if not inserted:
                new_outputs.append(batched.op_id)
                inserted = True
        else:
            new_outputs.append(out_id)
    return new_outputs


def batch_general_mla_ops(ops, outputs, *, views, prefix, runtime, policy):
    """Combine compatible analytic MLA work in a non-eager invocation."""
    runtime_meta = dict(getattr(runtime, "metadata", ()) or ())
    raw_eager = bool(runtime_meta.get("raw_eager")
                     or runtime_meta.get("execution_policy", {}).get("raw_eager"))
    if len(outputs) < MIN_BATCH_MEMBERS or raw_eager:
        return outputs
    cfg_fam = runtime.kernels.get("families", {}) if runtime else {}
    cfg_rec = runtime.kernels.get("recipes", {}) if runtime else {}
    if "attention" in cfg_fam:
        return outputs
    by_id = {op.op_id: op for op in ops}
    members = [by_id[out_id] for out_id in outputs if out_id in by_id]
    if (len(members) != len(outputs)
            or any(op.plan_required or op.source_boundary or op.recipe in cfg_rec
                   for op in members)):
        return outputs
    # Dense schedules use batch_ops; sparse analytic work preserves every
    # query and its geometry below, including explicit indexed programs.
    first_phase = members[0].phase
    if any(op.phase != first_phase for op in members) or any(
            dict(op.metadata).get("program_kind") == "dense_latent_attention_generic"
            for op in members):
        return outputs
    ids = {op.op_id for op in members}
    reqs = tuple(dict.fromkeys(req for op in members for req in op.requests))
    deps = tuple(dict.fromkeys(dep for op in members for dep in op.deps))
    first_meta = dict(members[0].metadata)
    geometries = [dict(op.metadata).get("indexed_attention_geometry") for op in members]
    if all(geometries):
        geometry = dict(geometries[0])
        sequences = tuple(pair for item in geometries for pair in
                          item.get("sequences", ((item.get("query"), item.get("context")),)))
        geometry.pop("query", None)
        geometry.pop("context", None)
        geometry["sequences"] = sequences
        first_meta["indexed_attention_geometry"] = geometry
    for key in ("indexed_gather_rows", "dense_window_sweep_rows",
                "selection_index_read_bytes"):
        first_meta[key] = sum(dict(op.metadata).get(key, 0) for op in members)
    tensor = dict(first_meta.get("attention_tensor_view", {}))
    for key in ("query_shape", "query_rope_shape", "output_shape"):
        if key in tensor:
            axis = 1 if key == "output_shape" and tensor["kind"] == "dense_paged_latent" else 0
            shape = list(tensor[key])
            shape[axis] = sum(dict(op.metadata)["attention_tensor_view"][key][axis]
                              for op in members)
            tensor[key] = tuple(shape)
    if tensor:
        tensor["batch_count"] = len(members)
        first_meta["attention_tensor_view"] = tensor
    traffic = dict(first_meta.get("attention_traffic_plan", {}))
    if traffic:
        for key in traffic:
            if key != "residency_basis":
                traffic[key] = sum(dict(op.metadata)["attention_traffic_plan"][key]
                                   for op in members)
        first_meta["attention_traffic_plan"] = traffic
    first_meta["request_shapes"] = tuple((v.q_len, v.context) for v in views)
    first_meta["logical_pairs"] = sum(
        float(dict(op.metadata).get("logical_pairs", 0.0)) for op in members)
    first_meta["kv_sweeps"] = sum(
        int(dict(op.metadata).get("kv_sweeps", 0)) for op in members)
    first_shape = members[0].semantic_shape
    total_q = sum(int(op.semantic_shape[0]) for op in members)
    max_kv = max(int(op.semantic_shape[1]) for op in members)
    sem_shape = (total_q, max_kv, *first_shape[2:])
    loc_kv = sum(int(op.local_shape[1]) for op in members)
    loc_shape = (total_q, loc_kv, *members[0].local_shape[2:])
    batched = replace(
        members[0], op_id=prefix + ".latent_attention_batch",
        recipe="latent_attention0", requests=reqs, deps=deps,
        semantic_shape=sem_shape, local_shape=loc_shape,
        padded_shape=sem_shape, launch_shape=sem_shape,
        flops=sum(op.flops for op in members),
        vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members),
        metadata=tuple(first_meta.items()),
    )
    first_idx = next(i for i, op in enumerate(ops) if op.op_id in ids)
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.insert(first_idx, batched)
    return [batched.op_id]
