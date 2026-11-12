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
from .program import KernelProgram, ParallelProgram, Stage
from .engine import Allocation

KEY_TILE = 128


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


def _alloc(context, owner, heads, q_bytes, kv_tile, latent_dim, dv, qk_dim):
    ham = ((heads + 15) // 16) * 16
    pv_keys = max(1, min(KEY_TILE, 65536 // max(1, latent_dim * 2)))
    pv_l0 = pv_keys * latent_dim * 2
    return (
        Allocation(f"q_l1_{owner}", context.pool_names["l1"], q_bytes, (owner,)),
        Allocation(f"kv_l1_{owner}", context.pool_names["l1"], kv_tile, (owner,)),
        Allocation(f"acc_ub_{owner}", context.pool_names["ub"],
                   heads * dv * 4 + heads * KEY_TILE * 4, (owner,)),
        Allocation(f"qk_l0a_{owner}", context.pool_names["l0a"],
                   ham * min(KEY_TILE, max(1, qk_dim)) * 2, (owner,)),
        Allocation(f"pv_l0b_{owner}", context.pool_names["l0b"], pv_l0, (owner,)),
        Allocation(f"out_l0c_{owner}", context.pool_names["l0c"],
                   ham * dv * 4, (owner,)),
    )


def _owner_program(context, owner, cores, heads, qk_dim, latent_dim, dv,
                   q, keys, kv_row, q_bytes, out_bytes):
    place = context.owned(owner, cores, ("mte2_aic", "vector", "mte3", "cube"))
    kv_tile = min(KEY_TILE, keys) * kv_row
    alloc = _alloc(context, owner, heads, q_bytes, kv_tile, latent_dim, dv, qk_dim)
    ham = ((heads + 15) // 16) * 16

    def tile(index):
        start = index * KEY_TILE
        count = min(KEY_TILE, keys - start)
        load = count * kv_row
        qk_i = 2 * count * ham * qk_dim
        pv_i = 2 * count * ham * latent_dim
        return (
            Stage(f"key_load{index}", context.bind(Work(
                resources={"hbm": load, "mte2_aic": load})), placement=place),
            Stage(f"qk{index}", context.bind(Work(
                matrix={"bf16": qk_i},
                resources={"vector": 4 * count * heads})),
                  after=(f"key_load{index}",), placement=place),
            Stage(f"online{index}", context.bind(Work(
                resources={"vector": 8 * count * heads + 2 * heads})),
                  after=(f"qk{index}",), placement=place),
            Stage(f"pv{index}", context.bind(Work(
                matrix={"bf16": pv_i},
                resources={"vector": 2 * count * heads})),
                  after=(f"online{index}",), placement=place),
            Stage(f"update{index}", context.bind(Work(
                resources={"vector": 4 * heads * dv})),
                  after=(f"pv{index}",), placement=place),
        )

    stages = [Stage(f"alloc{owner}", acquire=alloc, placement=place),
              Stage(f"q_load{owner}", context.bind(Work(
                  resources={"hbm": q_bytes, "mte2_aic": q_bytes})),
                    after=(f"alloc{owner}",), placement=place)]
    previous = f"q_load{owner}"
    for index in range((keys + KEY_TILE - 1) // KEY_TILE):
        for stage in tile(index):
            # The rolling KV slot may not be overwritten before the preceding
            # update consumed it; each tile's four stages stay causal.
            stages.append(replace(stage, after=(previous,)))
            previous = stage.name
    stages.append(Stage(f"finalize{owner}", context.bind(Work(
        resources={"vector": 2 * heads * dv,
                   "hbm": out_bytes, "mte3": out_bytes})),
        after=(previous,), placement=place, free=alloc))
    return KernelProgram(tuple(stages))


def build(op, geometry):
    parsed = _schedule(op)
    if parsed is None:
        return OperatorPlan(Work(), lambda context: KernelProgram(()),
                            metadata={"generic_latent_attention": {"idle": 1}})
    heads, qk_dim, latent_dim, dv, kv_row, sequences = parsed
    cores = max(1, int(geometry.compute_units))
    total = Work()
    logical = issued = 0
    total_q = total_kv = total_out = 0
    q_bytes_each = []
    for q, keys in sequences:
        q_bytes = q * heads * qk_dim * 2
        out_bytes = q * heads * latent_dim * 2
        q_bytes_each.append(q_bytes)
        total_q += q_bytes
        total_kv += keys * kv_row
        total_out += out_bytes
        logical += 2 * keys * heads * (qk_dim + latent_dim)
        ham = ((heads + 15) // 16) * 16
        issued += 2 * keys * ham * (qk_dim + latent_dim)
        total += Work(resources={"hbm": q_bytes, "mte2_aic": q_bytes})
        total += Work(resources={"hbm": keys * kv_row, "mte2_aic": keys * kv_row})
        total += Work(matrix={"bf16": 2 * keys * ham * (qk_dim + latent_dim)})
        owners = (keys + KEY_TILE - 1) // KEY_TILE
        vector = (14 * keys * heads + 2 * heads * owners
                  + 4 * heads * dv * owners + 2 * heads * dv)
        total += Work(resources={"vector": vector})
        total += Work(resources={"hbm": out_bytes, "mte3": out_bytes})

    def program(context):
        programs = tuple(
            _owner_program(context, owner, cores, heads, qk_dim, latent_dim, dv,
                           q, keys, kv_row,
                           q * heads * qk_dim * 2, q * heads * latent_dim * 2)
            for owner, (q, keys) in enumerate(sequences))
        return KernelProgram((Stage("owners", program=ParallelProgram(programs)),))

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
    if runtime is None or len(outputs) < 2 or len(outputs) != len(views):
        return outputs
    if (runtime.parallel.cp != 1 or runtime.parallel.pcp != 1
            or any(v.q_len != 1 or v.phase != "decode" for v in views)):
        return outputs
    policy = getattr(runtime, "attention_policy", None)
    if policy is not None:
        fused = (policy.get("fused_batch") if hasattr(policy, "get")
                 else getattr(policy, "fused_batch", None))
        if fused is False:
            return outputs
    ids = set(outputs)
    members = [op for op in ops if op.op_id in ids]
    if (len(members) != len(outputs)
            or any(dict(op.metadata).get("program_kind") != "dense_latent_attention_generic"
                   for op in members)
            or any(op.plan_required for op in members)
            or any(op.recipe in runtime.kernels.get("recipes", {}) for op in members)
            or "attention" in runtime.kernels.get("families", {})):
        return outputs
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
                      for v, item in zip(views, schedules))
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
    meta["latent_sequence_requests"] = tuple(v.request_id for v in views)
    meta["latent_batch_members"] = tuple(op.recipe for op in members)
    shape = (len(sequences), schedule["heads"], schedule.get("qk_dim", 576))
    batched = replace(
        members[0], op_id=prefix + ".latent_attention_batch",
        recipe="latent_attention0",
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
