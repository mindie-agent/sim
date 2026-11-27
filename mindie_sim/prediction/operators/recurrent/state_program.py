"""Finite vector recurrence with persistent per-head state.

[Role] Execute decay, state dot, delta, rank-one update and output dot.
[Invariants] A token consumes the preceding token's updated state. State is
  loaded once per head wave; each speculative checkpoint is written.
  Dot products and rank-one updates use vector service in this recipe.
[Timeouts & Locks] A bounded pipeline window, no device or host I/O.
[Side-Effect Policy] Submits only to the shared virtual engine.
"""
from dataclasses import dataclass
from bisect import bisect_right
from typing import Literal
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule_arch35.h",
        source_sha256="5590f9a236f83660df7b216af2d535c4cb07a8ccd96fa07ebc6be3f78c9a03c2",
    ),
)

REPLAY_OPERAND_BYTES = int(dtype_bytes("fp32"))
DEFAULT_AIV_UB_BUDGET_BYTES = 192 * 1024
VECTOR_LANE_ALIGNMENT = 16


def _strip_storage_bytes(dk: int, v_cols: int, working_state_byte: int) -> int:
    return 2 * dk * v_cols * working_state_byte + (2 * dk + 2 * v_cols) * REPLAY_OPERAND_BYTES


def _resolve_value_strips(dk: int, dv: int, working_state_byte: int, ub_budget: int) -> tuple[int, ...]:
    """Tile ``dv`` into UB-resident strips bounded by ``ub_budget``."""
    if _strip_storage_bytes(dk, dv, working_state_byte) <= ub_budget:
        return (dv,)
    min_storage = _strip_storage_bytes(dk, 1, working_state_byte)
    if min_storage > ub_budget:
        raise ModelingError(f"recurrent_vector_ub_capacity_exceeded:{min_storage}:{ub_budget}")
    available = ub_budget - 2 * dk * REPLAY_OPERAND_BYTES
    per_col = 2 * dk * working_state_byte + 2 * REPLAY_OPERAND_BYTES
    raw_v = available // per_col
    v_step = (raw_v // VECTOR_LANE_ALIGNMENT) * VECTOR_LANE_ALIGNMENT if raw_v >= VECTOR_LANE_ALIGNMENT else raw_v
    v_step = max(1, min(dv, v_step))
    full, tail = divmod(dv, v_step)
    return (v_step,) * full + ((tail,) if tail else ())


@dataclass(frozen=True)
class RecurrentVector:
    state_pool: str = "ub"

    def lower(self, context):
        op, names = context.op, context.resource_names
        meta = dict(op.metadata)
        q, heads, dk, dv = (meta[key] for key in ("tokens", "heads", "dk", "dv"))
        sequences = tuple(sequence for sequence in
                           meta.get("sequence_tokens", (q,)) if sequence > 0)
        state_dtype = meta.get("state_dtype", "fp32")
        state_byte = dtype_bytes(state_dtype)
        if meta.get("state_element_bytes", state_byte) != state_byte:
            raise ValueError("recurrent_vector_state_element_bytes_mismatch")
        if not sequences:
            # A physical graph may allocate state rows with no active sequence.
            # The host grid remains; the source recurrence does no state work.
            return KernelProgram(())
        units = context.resource_units.get("vector", context.core_count)
        vector = names["vector"]
        operand_read = read_role(names, aiv=0)
        pool = context.pool_names.get(self.state_pool)
        # Ascend AIV upcasts BF16/FP16 state into FP32 (4B/elem) working UB
        # buffers alongside the state-dot temporary and vector inputs/delta.
        working_state_byte = max(REPLAY_OPERAND_BYTES, int(state_byte))
        if pool is not None:
            pool_cap = int(context.pool_capacities.get(self.state_pool, 0) or 0)
            if pool_cap > 0:
                ub_budget = pool_cap
            elif context.datapath is not None:
                ub_budget = int(context.datapath.usable_bytes(self.state_pool))
            else:
                ub_budget = DEFAULT_AIV_UB_BUDGET_BYTES
            strips = _resolve_value_strips(dk, dv, working_state_byte, ub_budget)
        else:
            strips = (dv,)
        waves, ends = [], [0]
        for wave_idx, start in enumerate(range(0, heads * len(sequences), units)):
            lengths = tuple(sequences[head // heads]
                            for head in range(start, min(start + units, heads * len(sequences))))
            for strip_idx, strip_v in enumerate(strips):
                waves.append((wave_idx, strip_idx, strip_v, lengths))
                ends.append(ends[-1] + max(lengths))
        per_head_state = dk * dv * state_byte
        steps = sum(sequences) * heads
        read_per_step = (op.mte2_bytes - len(sequences) * heads * per_head_state) / steps
        act_byte = dtype_bytes(op.dtype or "bf16")
        v_read_per_step = min(max(read_per_step, 0.0), dv * act_byte)
        shared_read_per_step = read_per_step - v_read_per_step
        write_per_step = op.mte3_bytes / steps

        def copies(label, amount, role, active, after):
            if role not in names:
                return [Stage(label, {names["hbm"]: amount}, after=after)]
            # Shared copy pipes retain each active AIV's work. A union of pipe
            # IDs would distribute a ragged wave equally and borrow idle share.
            loads = {}
            for owner in active:
                targets = unit_group(
                    owner, units, context.resource_units.get(role, context.core_count))
                for target in targets:
                    loads[target] = loads.get(target, 0) + 1 / len(targets)
            groups = {}
            for target, owners in loads.items():
                groups.setdefault(owners, []).append(target)
            return [Stage(
                f"{label}{index}",
                {names["hbm"]: amount * owners * len(targets) / len(active),
                 names[role]: amount * owners * len(targets) / len(active)},
                after=after, placement={names[role]: tuple(targets)})
                for index, (owners, targets) in enumerate(groups.items())]

        def tile(index):
            slot = bisect_right(ends, index) - 1
            wave_idx, strip_idx, strip_v, wave_lengths = waves[slot]
            token = index - ends[slot]
            active = tuple(unit for unit, length in enumerate(wave_lengths) if token < length)
            active_heads = len(active)
            strip_frac = strip_v / dv
            initial_state = active_heads * dk * strip_v * state_byte if token == 0 else 0
            # First strip loads shared Q/K/gate/beta operands before key_dot/query_dot;
            # subsequent strips stream only their dv-slice of V, conserving total MTE2 bytes.
            operand_read_per_head = (shared_read_per_step if strip_idx == 0 else 0.0) + v_read_per_step * strip_frac
            read = operand_read_per_head * active_heads + initial_state
            write = write_per_step * active_heads * strip_frac
            storage = _strip_storage_bytes(dk, strip_v, working_state_byte)
            alloc_name = f"state{wave_idx}" if len(strips) == 1 else f"state{wave_idx}_v{strip_idx}"
            allocations = (() if pool is None else
                           (Allocation(alloc_name, pool, storage,
                                       tuple(range(len(wave_lengths)))),))
            placement = {vector: active}
            counts = (
                ("decay", active_heads * dk * strip_v),
                ("key_dot", active_heads * strip_v * (2 * dk - 1)),
                ("delta", 2 * active_heads * strip_v),
                ("update", 2 * active_heads * dk * strip_v),
                ("query_dot", active_heads * strip_v * (2 * dk - 1)),
            )
            stages = [Stage("state", acquire=allocations if token == 0 else ())]
            loads = copies("load", read, operand_read, active, ("state",))
            stages.extend(loads)
            predecessors = tuple(stage.name for stage in loads)
            for name, work in counts:
                stages.append(Stage(name, {vector: work}, after=predecessors,
                                    placement=placement))
                predecessors = (name,)
            stores = copies("store", write, "mte3", active, ("query_dot",))
            stages.extend(stores)
            stages.append(Stage("checkpoint", after=tuple(stage.name for stage in stores),
                                free=allocations if index == ends[slot + 1] - 1 else ()))
            return stages

        return PipelineProgram(ends[-1], 1, tile, context.minimum_ms, context.setup_ms)
