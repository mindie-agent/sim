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
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage

REPLAY_OPERAND_BYTES = int(dtype_bytes("fp32"))


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
        waves, ends = [], [0]
        for start in range(0, heads * len(sequences), units):
            lengths = tuple(sequences[head // heads]
                            for head in range(start, min(start + units, heads * len(sequences))))
            waves.append(lengths)
            ends.append(ends[-1] + max(lengths))
        per_head_state = dk * dv * state_byte
        steps = sum(sequences) * heads
        read_per_step = (op.mte2_bytes - len(sequences) * heads * per_head_state) / steps
        write_per_step = op.mte3_bytes / steps
        vector = names["vector"]
        operand_read = read_role(names, aiv=0)
        pool = context.pool_names.get(self.state_pool)
        # State plus the state-dot temporary and vector inputs/delta.
        storage = 2 * dk * dv * state_byte + (2 * dk + 2 * dv) * REPLAY_OPERAND_BYTES

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
            wave = bisect_right(ends, index) - 1
            token = index - ends[wave]
            active = tuple(unit for unit, length in enumerate(waves[wave]) if token < length)
            active_heads = len(active)
            initial_state = active_heads * per_head_state if token == 0 else 0
            read = read_per_step * active_heads + initial_state
            write = write_per_step * active_heads
            allocations = (() if pool is None else
                           (Allocation(f"state{wave}", pool, storage,
                                       tuple(range(len(waves[wave])))),))
            placement = {vector: active}
            counts = (
                ("decay", active_heads * dk * dv),
                ("key_dot", active_heads * dv * (2 * dk - 1)),
                ("delta", 2 * active_heads * dv),
                ("update", 2 * active_heads * dk * dv),
                ("query_dot", active_heads * dv * (2 * dk - 1)),
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
                                free=allocations if index == ends[wave + 1] - 1 else ()))
            return stages

        return PipelineProgram(ends[-1], 1, tile, context.minimum_ms, context.setup_ms)
