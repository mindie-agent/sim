"""Finite generic dense latent-MLA online-attention schedule.

[Role] Price one independent dense BF16 latent decode request, or the actual
  same-DP contiguous C8 group, without claiming a recovered vendor tiler.
[Invariants] One query owner per request; K is streamed in finite tiles; Q is
  retained on its owner; online max/sum and FP32 [H,Dv] accumulators update
  serially; one final store per request. Batch aggregation is one invocation
  with independent private owners, not two synthetic launch floors.
[Timeouts & Locks] O(requests + key tiles), no I/O or locks.
[Side-Effect Policy] Returns immutable plans; no engine mutation.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, Stage
from .engine import Allocation

KEY_TILE = 128
ZERO = 0
ONE = 1
TWO = 2
ZERO_FLOAT = 0.0
MIN_BATCH_MEMBERS = 2


def _schedule(op):
    raw = dict(op.metadata).get("latent_attention_schedule")
    if not isinstance(raw, dict):
        raise ModelingError("generic_latent_schedule_metadata_required")
    heads = int(raw["heads"])
    qk_dim = int(raw.get("qk_dim", 576))
    latent_dim = int(raw.get("latent_dim", 512))
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
    if any(keys == 0 or q == 0 for q, keys in sequences):
        return None
    return heads, qk_dim, latent_dim, dv, kv_row, sequences


def _alloc(context, core, cores, tag, heads, q_bytes, kv_tile, latent_dim, dv, qk_dim):
    ham = ((heads + 15) // 16) * 16
    pv_keys = max(1, min(KEY_TILE, 65536 // max(1, latent_dim * 2)))
    pv_l0 = pv_keys * latent_dim * 2
    specs = (
        ("q_l1", "l1", q_bytes),
        ("kv_l1", "l1", kv_tile),
        ("acc_ub", "ub", heads * dv * 4 + heads * KEY_TILE * 4),
        ("qk_l0a", "l0a", ham * min(KEY_TILE, max(1, qk_dim)) * 2),
        ("pv_l0b", "l0b", pv_l0),
        ("out_l0c", "l0c", ham * dv * 4),
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
    kv_tile = min(KEY_TILE, keys) * kv_row
    alloc = _alloc(context, core, cores, tag, heads, q_bytes, kv_tile, latent_dim, dv, qk_dim)
    ham = ((heads + 15) // 16) * 16

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
                resources={"vector": 4 * count * heads})),
                  after=(f"key_load{tag}_{index}",), placement=place),
            Stage(f"online{tag}_{index}", context.bind(Work(
                resources={"vector": 8 * count * heads + 2 * heads})),
                  after=(f"qk{tag}_{index}",), placement=place),
            Stage(f"pv{tag}_{index}", context.bind(Work(
                matrix={"bf16": pv_i},
                resources={"vector": 2 * count * heads})),
                  after=(f"online{tag}_{index}",), placement=place),
            Stage(f"update{tag}_{index}", context.bind(Work(
                resources={"vector": 4 * heads * dv})),
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
        resources={"vector": 2 * heads * dv,
                   "hbm": out_bytes, "mte3": out_bytes})),
        after=(previous,), placement=place, free=alloc))
    return stages, fin


def _max_heads_per_owner(heads, q, keys, qk_dim, latent_dim, dv, kv_row, geometry):
    buffers = dict(getattr(geometry, "buffers", {}) or {})

    def cap(pool):
        if pool not in buffers:
            return None
        return int(buffers[pool])

    limit = heads
    ub_cap = cap("ub")
    if ub_cap is not None:
        limit = min(limit, max(1, ub_cap // max(1, (dv + KEY_TILE) * 4)))
    l0c_cap = cap("l0c")
    if l0c_cap is not None:
        max_ham = max(16, (l0c_cap // max(1, dv * 4) // 16) * 16)
        if l0c_cap < 16 * dv * 4:
            max_ham = max(1, l0c_cap // max(1, dv * 4))
        limit = min(limit, max_ham)
    l0a_cap = cap("l0a")
    if l0a_cap is not None:
        qk_panel = min(KEY_TILE, max(1, qk_dim)) * 2
        max_ham = max(16, (l0a_cap // max(1, qk_panel) // 16) * 16)
        if l0a_cap < 16 * qk_panel:
            max_ham = max(1, l0a_cap // max(1, qk_panel))
        limit = min(limit, max_ham)
    l1_cap = cap("l1")
    kv_tile = min(KEY_TILE, keys) * kv_row
    if l1_cap is not None and l1_cap > kv_tile:
        limit = min(limit, max(1, (l1_cap - kv_tile) // max(1, q * qk_dim * 2)))
    if limit >= heads:
        return heads
    return max(16, (limit // 16) * 16) if limit >= 16 else max(1, limit)


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
    owner_items = []
    for q, keys in sequences:
        head_step = _max_heads_per_owner(heads, q, keys, qk_dim, latent_dim, dv, kv_row, geometry)
        for h_start in range(0, heads, head_step):
            h_cur = min(head_step, heads - h_start)
            owner_items.append((q, keys, h_cur))
            q_bytes = q * h_cur * qk_dim * 2
            out_bytes = q * h_cur * latent_dim * 2
            total_q += q_bytes
            total_kv += keys * kv_row
            total_out += out_bytes
            logical += 2 * keys * h_cur * (qk_dim + latent_dim)
            ham = ((h_cur + 15) // 16) * 16
            issued += 2 * keys * ham * (qk_dim + latent_dim)
            total += Work(resources={"hbm": q_bytes, read: q_bytes})
            total += Work(resources={"hbm": keys * kv_row, read: keys * kv_row})
            total += Work(matrix={"bf16": 2 * keys * ham * (qk_dim + latent_dim)})
            owners = (keys + KEY_TILE - 1) // KEY_TILE
            vector = (14 * keys * h_cur + 2 * h_cur * owners
                      + 4 * h_cur * dv * owners + 2 * h_cur * dv)
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
    if runtime.parallel.pcp != ONE and any(v.phase == "prefill" for v in views):
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
    schedule.update({"sequences": sequences, "batch": len(sequences),
                     "query_rows": sum(q for q, _ in sequences),
                     "keys": max(keys for _, keys in sequences),
                     "owners": len(sequences),
                     "kv_unique_bytes": sum(keys * schedule["kv_row_bytes"]
                                            for _, keys in sequences),
                     "q_bytes": sum(q * schedule["heads"] * schedule.get("qk_dim", 576) * 2
                                    for q, _ in sequences),
                     "output_bytes": sum(q * schedule["heads"] * schedule.get("latent_dim", 512) * 2
                                         for q, _ in sequences)})
    meta["latent_attention_schedule"] = schedule
    view = dict(meta.get("attention_tensor_view", {}))
    if view:
        view["query_shape"] = (sum(q for q, _ in sequences), schedule["heads"],
                               schedule.get("qk_dim", 576))
        view["output_shape"] = (schedule["heads"], sum(q for q, _ in sequences),
                                schedule.get("latent_dim", 512))
        view["batch_count"] = len(sequences)
        meta["attention_tensor_view"] = view
    meta["latent_batch_boundary"] = "generic_single_invocation_private_query_owners"
    meta["latent_sequence_requests"] = tuple(v.request_id for v in decode_views)
    meta["latent_batch_members"] = tuple(op.recipe for op in members)
    shape = (len(sequences), schedule["heads"], schedule.get("qk_dim", 576))
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
    """Combine remaining compatible analytic MLA attention ops when fused_batch is enabled."""
    if len(outputs) < MIN_BATCH_MEMBERS or (policy is not None and not policy.fused_batch):
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
    ids = {op.op_id for op in members}
    reqs = tuple(dict.fromkeys(req for op in members for req in op.requests))
    deps = tuple(dict.fromkeys(dep for op in members for dep in op.deps))
    first_meta = dict(members[ZERO].metadata)
    first_meta["request_shapes"] = tuple((v.q_len, v.context) for v in views)
    first_meta["logical_pairs"] = sum(
        float(dict(op.metadata).get("logical_pairs", ZERO_FLOAT)) for op in members)
    first_meta["kv_sweeps"] = sum(
        int(dict(op.metadata).get("kv_sweeps", ZERO)) for op in members)
    first_shape = members[ZERO].semantic_shape
    total_q = sum(int(op.semantic_shape[ZERO]) for op in members)
    max_kv = max(int(op.semantic_shape[ONE]) for op in members)
    sem_shape = (total_q, max_kv, *first_shape[TWO:])
    loc_kv = sum(int(op.local_shape[ONE]) for op in members)
    loc_shape = (total_q, loc_kv, *members[ZERO].local_shape[TWO:])
    batched = replace(
        members[ZERO], op_id=prefix + ".latent_attention_batch",
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
