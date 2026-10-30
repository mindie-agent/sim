"""Source recurrence with resident QKV and finite state-output queues.

[Role] Lower the arch35 FP32-state/BF16-input RecurrentGatedDeltaRule branch.
[Invariants] Whole-sequence QKV is loaded once per value head. State writes
  begin before the query dot; a single output slot prevents reuse until DMA
  completes. Acceptance selects the initial address, never fewer query rows.
[Timeouts & Locks] O(sequences + owners) plan, bounded head/value/token windows.
[Side-Effect Policy] Pure physical work and virtual Engine tasks; no I/O.
"""
import math

from .engine import Allocation
from .mechanism import OperatorPlan, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

IMPLEMENTATION = "ascend_recurrent_gdn_fp32_state"
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


def head_ownership(lengths, heads, blocks):
    """Source IsCurrentBlock accumulates whole-head lengths before advancing."""
    target = max(math.ceil(sum(lengths) * heads / blocks), max(lengths))
    owners = [[] for _ in range(blocks)]
    owner, load = 0, 0
    for seq, q in enumerate(lengths):
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


def build(op, geometry, *, compact=True):
    meta = dict(op.metadata)
    lengths = tuple(meta["sequence_tokens"])
    heads, dk, dv = (int(meta[key]) for key in ("heads", "dk", "dv"))
    keys = int(meta["key_heads"])
    if not (1 <= dk <= 512 and 1 <= dv <= 512 and 1 <= heads <= 256
            and keys > 0 and heads % keys == 0):
        raise ValueError("source_recurrent_gdn_head_dimensions_outside_host_domain")
    ak, av = math.ceil(dk / 16) * 16, math.ceil(dv / 16) * 16
    if op.dtype != "bf16" or meta["state_dtype"] != "fp32" or max(lengths) > 8:
        raise ValueError("source_recurrent_gdn_requires_bf16_fp32_state_q_le8")
    if max(lengths) * max(ak, av) * 4 > 16384:
        raise ValueError("source_recurrent_gdn_resident_qkv_exceeds_16k_buffer")
    value_step = 8192 // ak
    value_step = (255 if value_step > 256 else value_step) // 16 * 16
    if not value_step:
        raise ValueError("source_recurrent_gdn_state_tile_exceeds_32k_buffer")
    parts = math.ceil(dv / value_step)
    units = geometry.resource_units.get("vector", geometry.compute_units)
    blocks = min(units, len(lengths) * heads)
    ownership = head_ownership(lengths, heads, blocks)
    run_counts = tuple(sum(1 for _ in head_runs(lengths, segments, compact=compact))
                       for segments in ownership)
    read = read_role(geometry.resource_units, aiv=0)
    total = sum(lengths) * heads
    tensor_reads = len(lengths) * heads * dk * dv * 4 + total * (2 * dk + dv) * 2
    scalar_payload = total * (4 + 2 * parts)
    writes = total * (dk * dv * 4 + dv * 2)
    vector = len(lengths) * heads * ak * dv + total * (8 * ak * dv + dv + 6 * ak + av)
    work = Work(resources={"hbm": tensor_reads + scalar_payload + writes,
                           read: tensor_reads, "mte3": writes, "vector": vector})

    def program(context):
        def owner_program(owner, segments):
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
                        scalar = 2 + (4 if part_index == 0 else 0)
                        return (
                            Stage("scalar_inputs", context.bind(Work(resources={"hbm": scalar}).scaled(repeat)),
                                  previous=("attention_cast",)),
                            Stage("gamma", v(2 * ak if part_index == 0 else 0),
                                  after=("scalar_inputs",), placement=owned),
                            Stage("decay", v(ak * n), after=("gamma",), placement=owned),
                            Stage("qk_load", memory(q * 2 * dk * 2 if first else 0, read),
                                  after=("decay",), placement=owned),
                            Stage("qk_cast_scale", v(q * 3 * ak if first else 0),
                                  after=("qk_load",), placement=owned),
                            Stage("key_dot", v((2 * ak - 1) * n), after=("qk_cast_scale",), placement=owned),
                            Stage("v_load", memory(q * dv * 2 if first else 0, read),
                                  after=("key_dot",), placement=owned),
                            Stage("v_cast", v(q * av if first else 0), after=("v_load",), placement=owned),
                            Stage("delta", v(2 * n), after=("v_cast",), placement=owned),
                            Stage("rank_one_update", v(2 * ak * n), after=("delta",), placement=owned),
                            Stage("state_copy", v(ak * n), after=("rank_one_update",),
                                  previous=("state_write",), placement=owned),
                            Stage("state_write", memory(dk * n * 4, "mte3"), after=("state_copy",),
                                  previous=("attention_write",), placement=owned),
                            Stage("query_dot", v((2 * ak - 1) * n), after=("state_copy",), placement=owned),
                            Stage("attention_cast", v(n), after=("query_dot",),
                                  previous=("attention_write",), placement=owned),
                            Stage("attention_write", memory(n * 2, "mte3"), after=("attention_cast", "state_write"), placement=owned),
                        )
                    return (Stage("state_load", memory(dk * n * 4, read), placement=owned),
                            Stage("state_resident_copy", v(ak * n), after=("state_load",), placement=owned),
                            Stage("tokens", after=("state_resident_copy",), program=PipelineProgram(q, 2, token)))
                return (Stage("gamma_zero", v(q * ak), placement=owned),
                        Stage("value_tiles", after=("gamma_zero",), program=PipelineProgram(parts, 1, part)))
            pending = head_runs(lengths, segments, compact=compact)
            heads_program = PipelineProgram(run_counts[owner], 1,
                                            lambda index: head(*next(pending)))
            pool = context.pool_names.get("ub")
            if pool is None:
                return heads_program
            allocation = (Allocation("recurrent_buffers", pool, UB_BYTES,
                                     unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("heads", after=("buffers",), program=heads_program),
                                  Stage("release", after=("heads",), free=allocation)))
        return ParallelProgram(tuple(owner_program(owner, segments) for owner, segments in enumerate(ownership)))
    return OperatorPlan(work, program, metadata={"recurrent_head_schedule": {
        "compact": compact, "logical_heads": len(lengths) * heads,
        "resource_cycles": sum(run_counts),
        "owner_heads": tuple(sum(count for _, count in segments) for segments in ownership),
        "owner_cycles": run_counts,
        "handoff": "conservative_serial_head_and_DV_tile_retirement",
        "contention_approximation": "serial_cycle_fluid_rephasing" if compact else "none",
    }})
