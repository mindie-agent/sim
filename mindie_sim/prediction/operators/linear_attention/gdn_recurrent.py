"""Source recurrence with resident QKV and finite state-output queues.

[Role] Lower the arch35 FP32-state/BF16-input RecurrentGatedDeltaRule branch.
[Invariants] Whole-sequence QKV is loaded once per value head. State writes
  begin before the query dot; a single output slot prevents reuse until DMA
  completes. Acceptance selects the initial address, never fewer query rows.
  The consumed version is the committed token count; candidate checkpoints
  exist before token commit, and only the accepted offset becomes visible.
  Prefill effects distinguish cache gather, the local H chunk bank and the
  post-H scatter publish; conv effects are tail writebacks, not chunk commits.
  Padding graph slots own no state and retain no candidate.
[Timeouts & Locks] O(sequences + owners) plan, bounded head/value/token windows.
[Side-Effect Policy] Pure physical work and virtual Engine tasks; no I/O.
"""
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import StateEffect
from ...engine import Allocation
from mindie_sim.prediction.operators.linear_attention.gdn_recurrent_contracts import (
    STATE_DIRECTIONS, STATE_TARGET_KINDS, _SLOT_PUBLISH, _SLOT_SCOPES,
    _TARGET_FIELD, _slot_ports, _state_effect, _state_target, bind_state_ports,
    state_contract, tensor_port,
)
from ...mechanism import OperatorPlan, Work
from ...ownership import read_role, unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ..contracts import OperatorPlanningContext
from .common import owner_units, request_layout_metadata, request_lengths

IMPLEMENTATION = "ascend_recurrent_gdn_fp32_state"
STATE_CONTRACT_KEY = "gdn_state_contract"
STATE_EFFECTS_KEY = "gdn_state_effects"
SOURCE = {
    "repository": "Ascend/ops-transformer",
    "path": "attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule_arch35.h",
    "sha256": "5590f9a236f83660df7b216af2d535c4cb07a8ccd96fa07ebc6be3f78c9a03c2",
    "ranges": ((135, 158), (188, 249), (285, 399), (405, 475)),
    "host": {
        "path": "attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp",
        "sha256": "785101bff3cc4454ffdf731661b940b69254850c5eb5fab7232bb118085cb9ed",
        "block_count_lines": (327, 329),
    },
}
UB_BYTES = 247808
_MAX_HEAD_DIM = 512
_MAX_VALUE_HEADS = 256
_MAX_QUERY_ROWS = 8
VECTOR_LANE_ALIGNMENT = 16
_RESIDENT_QKV_BYTES = 16384
_STATE_TILE_ELEMENTS = 8192
_VALUE_TILE_COLUMN_LIMIT = 256
QK_CAST_SCALE_OPS_PER_LANE = 3


def head_ownership(lengths, heads, blocks):
    """Source IsCurrentBlock accumulates whole-head lengths before advancing."""
    target = max(math.ceil(sum(lengths) * heads / blocks), max(lengths))
    owners = [[] for _ in range(blocks)]
    owner, load = 0, 0
    for seq, q in enumerate(lengths):
        if q == 0:
            continue
        remaining = heads
        while remaining:
            count = min(remaining, math.ceil((target - load) / q))
            owners[owner].append((seq, count))
            remaining -= count
            load += count * q
            if load >= target:
                owner += 1
                load = 0
    return tuple(tuple(segments) for segments in owners if segments)


def head_runs(lengths, segments, *, compact=True):
    """Consecutive equal-q heads retain their owner's full repeated work."""
    current, count = None, 0
    for sequence, heads in segments:
        q = lengths[sequence]
        if not compact:
            for _ in range(heads):
                yield q, 1
        elif q == current:
            count += heads
        else:
            if count:
                yield current, count
            current, count = q, heads
    if compact and count:
        yield current, count


def merge_contracts(contracts, *, allow_mixed_directions=False):
    """Concatenate per-request contracts without copying the first request.

    A fused launch keeps one physical op but must retain every request's
    targets, versions, masks, slot mapping and publish policy.
    """
    contracts = tuple(contracts)
    if not contracts:
        return None
    if len(contracts) == 1:
        return contracts[0]
    first = contracts[0]
    for contract in contracts[1:]:
        for key in ("phase", "direction", "target_kind", "scope",
                    "publish_point", "state_dtype", "conv_state_dtype",
                    "conv_state_layout", "conv_kernel", "state_layout",
                    "index_row_stride", "candidate_write_point",
                    "accepted_checkpoint", "rejected_checkpoints",
                    "commit_policy"):
            if contract.get(key) != first.get(key):
                if key == "commit_policy" and (
                        "pending_acceptance_event" in {
                            contract.get(key), first.get(key)}
                        or allow_mixed_directions):
                    continue
                if (key == "direction" and allow_mixed_directions
                        and {contract.get(key), first.get(key)}
                        <= {"token", "conv"}):
                    continue
                if (key == "phase" and allow_mixed_directions
                        and {contract.get(key), first.get(key)}
                        <= {"decode", "prefill", "mixed"}):
                    continue
                if key in {"publish_point", "scope", "accepted_checkpoint",
                           "rejected_checkpoints"} and allow_mixed_directions:
                    continue
                raise ModelingError(f"gdn_state_contract_merge_mismatch:{key}")
    def concat(key):
        return tuple(item for contract in contracts
                     for item in (contract.get(key) or ()))
    def offset(key):
        result, cursor = [], 0
        for contract in contracts:
            result.extend(value + cursor for value in (contract.get(key) or ()))
            cursor += int(contract.get("slots", 0))
        return tuple(result)
    slot_indices = [contract.get("slot_indices") for contract in contracts]
    merged = dict(first)
    merged.update({
        "slots": sum(int(contract.get("slots", 0)) for contract in contracts),
        "index_slots": sum(int(contract.get("index_slots", 0))
                           for contract in contracts),
        "index_copy_required": any(bool(contract.get("index_copy_required"))
                                   for contract in contracts),
        "padding_slots": offset("padding_slots"),
        "active_slots": offset("active_slots"),
        "fresh_slots": offset("fresh_slots"),
        "no_state_io_slots": offset("no_state_io_slots"),
        "pending_slots": offset("pending_slots"),
        "slot_phases": concat("slot_phases"),
        "slot_directions": concat("slot_directions"),
        "slot_publish_points": concat("slot_publish_points"),
        "mixed_directions": any(contract.get("direction") != first.get("direction")
                                or tuple(contract.get("slot_directions") or ())
                                != tuple(first.get("slot_directions") or ())
                                for contract in contracts),
        "direction": ("mixed"
                      if any(contract.get("direction") != first.get("direction")
                             or tuple(contract.get("slot_directions") or ())
                             != tuple(first.get("slot_directions") or ())
                             for contract in contracts)
                      else first.get("direction")),
        "phase": ("mixed" if any(contract.get("phase") != first.get("phase")
                                  for contract in contracts) else first.get("phase")),
        "scope": ("per_slot"
                  if any(contract.get("scope") != first.get("scope")
                         for contract in contracts) else first.get("scope")),
        "publish_point": ("per_slot"
                          if any(contract.get("publish_point")
                                 != first.get("publish_point")
                                 for contract in contracts)
                          else first.get("publish_point")),
        "slot_scopes": concat("slot_scopes"),
        "accepted_checkpoint": ("per_slot"
                                if any(contract.get("accepted_checkpoint")
                                       != first.get("accepted_checkpoint")
                                       for contract in contracts)
                                else first.get("accepted_checkpoint")),
        "request_ids": concat("request_ids"),
        "queries": concat("queries"),
        "global_queries": concat("global_queries"),
        "query_ranges": concat("query_ranges"),
        "contexts": concat("contexts"),
        "accepted_offsets": concat("accepted_offsets"),
        "advances": concat("advances"),
        "checkpoints": concat("checkpoints"),
        "read_versions": concat("read_versions"),
        "commit_versions": concat("commit_versions"),
        "publish_versions": concat("publish_versions"),
        "local_bank_versions": concat("local_bank_versions"),
        "local_chunk_versions": concat("local_chunk_versions"),
        "local_checkpoint_versions": concat("local_checkpoint_versions"),
        "local_commit_versions": concat("local_commit_versions"),
        "global_versions": concat("global_versions"),
        "internal_commit_versions": concat("internal_commit_versions"),
        "slot_commit_policy": concat("slot_commit_policy"),
        "decision_inputs": concat("decision_inputs"),
        "decision_ready": concat("decision_ready"),
        "candidate_stores": concat("candidate_stores"),
        "value_inputs": concat("value_inputs"),
        "value_outputs": concat("value_outputs"),
        "remote_inputs": concat("remote_inputs"),
        "targets": concat("targets"),
        "heads": max((int(contract.get("heads") or 0)
                      for contract in contracts), default=0),
        "head_shards": max((int(contract.get("head_shards") or 1)
                            for contract in contracts), default=1),
        "slot_indices": (None if any(values is None for values in slot_indices)
                         else tuple(item for values in slot_indices
                                    for item in values)),
        "state_capacity": (first.get("state_capacity")
                           if all(contract.get("state_capacity")
                                  == first.get("state_capacity")
                                  for contract in contracts) else
                           (None if any(contract.get("state_capacity") is None
                                        for contract in contracts) else
                            max(contract.get("state_capacity") or 0
                                for contract in contracts))),
        "state_capacity_lower_bound": max(
            (contract.get("state_capacity_lower_bound") or 0
             for contract in contracts), default=0) or None,
        "copy_boundaries": tuple(dict.fromkeys(
            item for contract in contracts
            for item in (contract.get("copy_boundaries") or ()))),
        "publish_pending": (first.get("publish_pending", "")
                            if all(contract.get("publish_pending")
                                   == first.get("publish_pending")
                                   for contract in contracts) else
                            "remote_producer"),
        "commit_policy": ("per_slot"
                          if any(contract.get("commit_policy")
                                 != first.get("commit_policy")
                                 for contract in contracts)
                          else first.get("commit_policy")),
    })
    return merged


def _state_plan_meta(meta):
    """Program-level mirror of the lowering's typed state effects."""
    result = {}
    if "gdn_state_contract" in meta:
        result["state_contract"] = meta["gdn_state_contract"]
    if "gdn_state_effects" in meta:
        result["state_effects"] = meta["gdn_state_effects"]
    return result


def recurrent_lengths(op, ctx: OperatorPlanningContext | None) -> tuple[int, ...]:
    if ctx is not None and ctx.requests:
        return request_lengths(ctx.requests)
    return tuple(dict(op.metadata)["sequence_tokens"])


def state_value_step(aligned_key: int) -> int:
    columns = _STATE_TILE_ELEMENTS // aligned_key
    if columns > _VALUE_TILE_COLUMN_LIMIT:
        columns = _VALUE_TILE_COLUMN_LIMIT - 1
    return columns // VECTOR_LANE_ALIGNMENT * VECTOR_LANE_ALIGNMENT


def token_vector_ops(aligned_key: int, aligned_value: int, dv: int) -> int:
    dot_ops = 2 * (2 * aligned_key - 1)
    state_ops = aligned_key + 2 * aligned_key + aligned_key  # Decay, rank-one update, state copy.
    value_ops = dot_ops + state_ops + 2 + 1  # Delta and output cast.
    key_ops = (1 + 2 + QK_CAST_SCALE_OPS_PER_LANE) * aligned_key
    return value_ops * dv + key_ops + aligned_value


def build(op, geometry, *, compact=True, ctx: OperatorPlanningContext | None = None):
    meta = dict(op.metadata)
    lengths = recurrent_lengths(op, ctx)
    plan_meta = _state_plan_meta(meta)
    if ctx is not None and ctx.requests:
        plan_meta.update(request_layout_metadata(ctx))
        plan_meta["state_contraction_ops"] = 0
    if not any(length > 0 for length in lengths):
        # A graph-padded step can own physical rows without an active sequence.
        # Its host grid and buffers remain, but the source recurrence performs
        # no state work; keep an explicit empty program rather than inventing a
        # token or falling back to a different implementation.
        return OperatorPlan(Work(), lambda context: KernelProgram(()),
                            metadata={"empty_active_sequences": True, **plan_meta})
    heads, dk, dv = (int(meta[key]) for key in ("heads", "dk", "dv"))
    keys = int(meta["key_heads"])
    if not (1 <= dk <= _MAX_HEAD_DIM and 1 <= dv <= _MAX_HEAD_DIM
            and 1 <= heads <= _MAX_VALUE_HEADS
            and keys > 0 and heads % keys == 0):
        raise ValueError("source_recurrent_gdn_head_dimensions_outside_host_domain")
    ak, av = align_up(dk, VECTOR_LANE_ALIGNMENT), align_up(dv, VECTOR_LANE_ALIGNMENT)
    max_query = max(lengths)
    if op.dtype != "bf16" or meta["state_dtype"] != "fp32" or max_query > _MAX_QUERY_ROWS:
        raise ValueError("source_recurrent_gdn_requires_bf16_fp32_state_q_le8")
    state_bytes = int(dtype_bytes(meta["state_dtype"]))
    input_bytes = int(dtype_bytes(op.dtype))
    if max_query * max(ak, av) * state_bytes > _RESIDENT_QKV_BYTES:
        raise ValueError("source_recurrent_gdn_resident_qkv_exceeds_16k_buffer")
    value_step = state_value_step(ak)
    if not value_step:
        raise ValueError("source_recurrent_gdn_state_tile_exceeds_32k_buffer")
    parts = math.ceil(dv / value_step)
    units, core_ids = recurrent_owners(geometry, ctx)
    active_heads = sum(length > 0 for length in lengths) * heads
    blocks = min(len(core_ids), active_heads)
    if not blocks:
        raise ModelingError("source_recurrent_gdn_requires_active_vector_owners")
    ownership = head_ownership(lengths, heads, blocks)
    run_counts = tuple(sum(1 for _ in head_runs(lengths, segments, compact=compact))
                       for segments in ownership)
    read = read_role(geometry.resource_units, aiv=0)
    total = sum(lengths) * heads
    tensor_reads = active_heads * dk * dv * state_bytes + total * (2 * dk + dv) * input_bytes
    scalar_payload = total * (state_bytes + input_bytes * parts)
    writes = total * (dk * dv * state_bytes + dv * input_bytes)
    vector = active_heads * ak * dv + total * token_vector_ops(ak, av, dv)
    work = Work(resources={"hbm": tensor_reads + scalar_payload + writes,
                           read: tensor_reads, "mte3": writes, "vector": vector})

    def program(context):
        def owner_program(owner, segments, run_count):
            owned = context.owned(owner, units, ("vector", read, "mte3"))
            def head(q, repeat):
                # This scales real repeated work, including each distinct
                # head's QKV/state traffic; it does not scale a duration or
                # keep tensors from different heads simultaneously resident.
                def v(amount):
                    return context.bind(Work(resources={"vector": amount}).scaled(repeat))
                def memory(amount, role):
                    return context.bind(Work(resources={"hbm": amount, role: amount}).scaled(repeat))
                def part(part_index):
                    n = min(value_step, dv - part_index * value_step)
                    def token(t):
                        first = part_index == 0 and t == 0
                        # Scalar GetValue payload is not an MTE2 instruction.
                        # Its issue/cache latency needs a separate hardware
                        # declaration; no PMU counter becomes a stage delay.
                        scalar = input_bytes + (state_bytes if part_index == 0 else 0)
                        stages = [
                            Stage("scalar_inputs", context.bind(Work(resources={"hbm": scalar}).scaled(repeat)),
                                  previous=("attention_cast",)),
                        ]
                        prev = "scalar_inputs"
                        if part_index == 0:
                            stages.append(Stage("gamma", v(2 * ak), after=(prev,), placement=owned))
                            prev = "gamma"
                        stages.append(Stage("decay", v(ak * n), after=(prev,), placement=owned))
                        prev = "decay"
                        if first:
                            stages.append(Stage("qk_load", memory(q * 2 * dk * input_bytes, read),
                                                after=(prev,), placement=owned))
                            stages.append(Stage("qk_cast_scale", v(q * QK_CAST_SCALE_OPS_PER_LANE * ak),
                                                after=("qk_load",), placement=owned))
                            prev = "qk_cast_scale"
                        stages.append(Stage("key_dot", v((2 * ak - 1) * n), after=(prev,), placement=owned))
                        prev = "key_dot"
                        if first:
                            stages.append(Stage("v_load", memory(q * dv * input_bytes, read),
                                                after=(prev,), placement=owned))
                            stages.append(Stage("v_cast", v(q * av),
                                                after=("v_load",), placement=owned))
                            prev = "v_cast"
                        stages.extend((
                            Stage("delta", v(2 * n), after=(prev,), placement=owned),
                            Stage("rank_one_update", v(2 * ak * n), after=("delta",), placement=owned),
                            Stage("state_copy", v(ak * n), after=("rank_one_update",),
                                  previous=("state_write",), placement=owned),
                            Stage("state_write", memory(dk * n * state_bytes, "mte3"), after=("state_copy",),
                                  previous=("attention_write",), placement=owned),
                            Stage("query_dot", v((2 * ak - 1) * n), after=("state_copy",), placement=owned),
                            Stage("attention_cast", v(n), after=("query_dot",),
                                  previous=("attention_write",), placement=owned),
                            Stage("attention_write", memory(n * input_bytes, "mte3"),
                                  after=("attention_cast", "state_write"), placement=owned),
                        ))
                        return tuple(stages)
                    return (Stage("state_load", memory(dk * n * state_bytes, read), placement=owned),
                            Stage("state_resident_copy", v(ak * n), after=("state_load",), placement=owned),
                            Stage("tokens", after=("state_resident_copy",), program=PipelineProgram(q, 2, token)))
                return (Stage("gamma_zero", v(q * ak), placement=owned),
                        Stage("value_tiles", after=("gamma_zero",), program=PipelineProgram(parts, 1, part)))
            pending = tuple(head_runs(lengths, segments, compact=compact))
            heads_program = PipelineProgram(run_count, 1,
                                            lambda index, pending=pending: head(*pending[index]))
            pool = context.pool_names.get("ub")
            if pool is None:
                return heads_program
            allocation = (Allocation("recurrent_buffers", pool, UB_BYTES,
                                     unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("heads", after=("buffers",), program=heads_program),
                                  Stage("release", after=("heads",), free=allocation)))
        # Owners stay separate programs: ``tests/test_gdn_recurrent_compaction``
        # requires every demand-carrying task of this operator to keep its own
        # single physical unit, which a summed-work cohort placement would
        # widen.  Only the boundary encoding is collapsed (see program.py).
        return ParallelProgram(tuple(owner_program(owner, segments, count)
                                     for owner, segments, count in zip(core_ids, ownership, run_counts)))
    if ctx is not None and ctx.requests:
        plan_meta["state_contraction_ops"] = 2 * total * dk * dv
    return OperatorPlan(work, program, metadata={**plan_meta,
        "recurrent_head_schedule": {
        "compact": compact, "logical_heads": active_heads,
        "resource_cycles": sum(run_counts),
        "owner_heads": tuple(sum(count for _, count in segments) for segments in ownership),
        "owner_cycles": run_counts,
        "handoff": "conservative_serial_head_and_DV_tile_retirement",
        "contention_approximation": "serial_cycle_fluid_rephasing" if compact else "none",
    }})


def recurrent_owners(geometry, ctx: OperatorPlanningContext | None) -> tuple[int, range | tuple[int, ...]]:
    if ctx is not None and ctx.requests:
        return ctx.datapath.aiv_count, owner_units(ctx)[1]
    units = geometry.resource_units.get("vector", geometry.compute_units)
    return units, range(units)
