"""Source-selected vector KDA decode with resident state and finite queues.

[Role] Lower vLLM-Ascend RecurrentKda arch35 to the common physical Engine.
[Invariants] One launch spans all sequences. QKV/gates remain resident per
  value head; initial state is read once, every query writes a checkpoint.
  Acceptance selects the committed initial slot and makes only the accepted
  prefix visible; later verification candidates are rejected individually
  without changing query work. Host-selected AIV ownership and UB queues never
  borrow idle-core supply. Repeated heads scale real work, not measured
  duration or retained state.
[Timeouts & Locks] O(sequences * active owners) bounded runs; q <= 8. No locks.
[Side-Effect Policy] Pure plans and virtual resource tasks; no runtime/source I/O.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
from typing import Literal

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.prediction.operators.recurrent.recover_ssm_work import record_head_bytes, record_slice_bytes
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec, ProgramFragment, StateEffect
from mindie_sim.schema.subgraphs import PortSet
from mindie_sim.schema.tensors import TensorValue

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
from .kda_source import KDA_MAX_QUERY_ROWS
from .kda_recurrent_contracts import (
    STATE_POOL_DIMS, _DEFAULT_HEAD_DIM, _binding_attrs, _binding_span,
    _decision_name, _effect, _fragment, _slot_storage, _state_effect,
    _value_versions, state_accesses, state_contract, state_fragments,
)
from .kda_state_selection import (
    bind_sequence_values, bind_state_value, binding_record, binding_value,
    decision_ports, resolve as resolve_state_selection, state_ports,
    state_records, state_value,
)

KDA_MAX_VALUE_HEADS = 256
KDA_KEY_HEAD_DIM = 128
KDA_VALUE_HEAD_DIMS = (128, 256)
HEAD_TILE_ALIGNMENT = 16
VALUE_TILE_ALIGNMENT = 8
BETA_LANE_ALIGNMENT = 8
SYNC_BUFFER_BYTES = 64
HOST_UB_GUARD_BYTES = 2048
FP32_BYTES = 4
HOST_KEY_WORKSPACE_FP32_ROWS = 3
GATE_QUEUE_OPS_PER_LANE = 3
SAFE_GATE_OPS_PER_LANE = 7
GATE_OPS_PER_LANE = 4
A_LOG_TRANSFORM_OPS = 3
QK_PREP_OPS_PER_LANE = 3
QK_NORM_EXTRA_OPS_PER_LANE = 6
RECURRENCE_OPS_PER_STATE_ELEMENT = 7

IMPLEMENTATION = RECIPE = "vllm_ascend_recurrent_kda_arch35"
SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d", "source_state": "committed",
    "kernel": {"path": "csrc/attention/recurrent_kda/op_kernel/arch35/recurrent_kda.h",
               "sha256": "f35dd2c5fb604d6d9711c11139c755414d49fb27fc6aea0f3a4bec3d3adb0199",
               "lines": ((151, 198), (235, 270), (375, 596), (736, 920))},
    "host": {"path": "csrc/attention/recurrent_kda/op_host/recurrent_kda_tiling_processor.h",
             "sha256": "c9eac28bd528eb03bdeb456c4fdbed4951f716937570ee9d810e0fb5b3ef3fbd",
             "lines": ((592, 621), (648, 756))},
    "caller": {"path": "vllm_ascend/ops/kda.py",
               "sha256": "ca2f56280746f45a347578e7191decda716557c93aca69b5c3baf7b22a578803"},
    "scope": "selected_source_kernel_not_capture_binary_identity",
}
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        SOURCE["repository"],
        SOURCE["revision"],
        SOURCE["kernel"]["path"],
        SOURCE["kernel"]["sha256"],
    ),
    SourceRef(
        SOURCE["repository"],
        SOURCE["revision"],
        SOURCE["host"]["path"],
        SOURCE["host"]["sha256"],
    ),
    SourceRef(
        SOURCE["repository"],
        SOURCE["revision"],
        SOURCE["caller"]["path"],
        SOURCE["caller"]["sha256"],
    ),
)
_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
A5_AIC_UNITS = 32
A5_AIV_UNITS = 64
A5_UB_BYTES = 253952


def frontend_geometry():
    """Declared A5 ownership used before a device cohort is attached.

    The source plan still recomputes the same work against its actual geometry;
    this default only lets the frontend publish nonempty physical state IO.
    """
    return Geometry(
        A5_AIC_UNITS,
        {"vector": A5_AIV_UNITS, "mte2_aiv": A5_AIV_UNITS,
         "mte3": A5_AIV_UNITS, "mte2_aic": A5_AIC_UNITS},
        {"ub": A5_UB_BYTES},
        {"ub": A5_AIV_UNITS},
    )


@dataclass(frozen=True)
class SourceExecution:
    """Physical differences between the two retained source headers."""
    beta_per_head: bool = True
    delta_broadcast: bool = False
    validate_all_sequences: bool = False


@dataclass(frozen=True)
class Params:
    lengths: tuple
    heads: int
    key_heads: int
    dk: int = _DEFAULT_HEAD_DIM
    dv: int = _DEFAULT_HEAD_DIM
    dtype: str = "bf16"
    state_dtype: str = "fp32"
    gate_dtype: str = "bf16"
    beta_dtype: str = "bf16"
    use_qk_l2norm: bool = True
    use_gate_in_kernel: bool = True
    use_beta_sigmoid: bool = False
    safe_gate: bool = True
    allow_neg_eigval: bool = False
    has_a_log: bool = True
    has_dt_bias: bool = True
    has_initial_state: bool = True
    state_v_first: bool = True
    has_state_indices: bool = True
    has_accepted_tokens: bool = False
    metadata_dtype: str = "int32"
    state_selection: dict | None = None
    #: Committed token count before this invocation, one per sequence.  None
    #: keeps the declared version basis invocation-local; no target context is
    #: inferred from a measurement or from another request.
    state_contexts: tuple | None = None
    #: Caller-declared ``(accepted_tokens, advanced_tokens)`` per sequence.
    #: Absence is pending verification, never an implicit all-row commit.
    state_acceptance: tuple | None = None
    #: Stable state identity per sequence.  None derives a layer/request
    #: identity in the lowerer; an occurrence name is never the state key.
    state_targets: tuple | None = None
    recover_ssm: bool = False
    record_stride: int | None = None

    @classmethod
    def parse(cls, raw):
        p = cls(**{**raw, "lengths": tuple(raw["lengths"])})
        if not isinstance(p.recover_ssm, bool):
            raise ModelingError("recurrent_kda_recover_ssm_invalid")
        if not p.lengths or any(not isinstance(q, int) or not 1 <= q <= KDA_MAX_QUERY_ROWS for q in p.lengths):
            raise ModelingError("recurrent_kda_requires_nonempty_q1_to_q8_sequences")
        if p.record_stride is not None:
            if (isinstance(p.record_stride, bool)
                    or not isinstance(p.record_stride, int)
                    or p.record_stride < max(p.lengths)):
                raise ModelingError("recurrent_kda_record_stride_invalid")
        if (not 1 <= p.key_heads <= p.heads <= KDA_MAX_VALUE_HEADS or p.heads % p.key_heads
                or p.dk != KDA_KEY_HEAD_DIM or p.dv not in KDA_VALUE_HEAD_DIMS):
            raise ModelingError("recurrent_kda_source_head_geometry_unsupported")
        if (p.dtype != "bf16" or p.state_dtype not in ("bf16", "fp32")
                or p.gate_dtype not in _BYTES or p.beta_dtype not in _BYTES):
            raise ModelingError("recurrent_kda_source_precision_unsupported")
        if not p.has_initial_state or not p.state_v_first:
            raise ModelingError("recurrent_kda_requires_initial_v_first_state")
        if p.state_selection is not None:
            if not p.has_state_indices:
                raise ModelingError("recurrent_kda_accepted_selection_requires_state_indices")
            resolve_state_selection(p.state_selection, p.lengths)
            p = replace(p, has_accepted_tokens=True)
        elif p.has_accepted_tokens:
            raise ModelingError("recurrent_kda_accepted_state_slot_branch_not_bound")
        if p.state_contexts is not None:
            contexts = tuple(p.state_contexts)
            if len(contexts) != len(p.lengths) or any(
                    type(value) is not int or value < 0 for value in contexts):
                raise ModelingError("recurrent_kda_state_contexts_must_match_sequences")
            p = replace(p, state_contexts=contexts)
        if p.state_acceptance is not None:
            acceptance = tuple(
                None if value is None else tuple(value)
                for value in p.state_acceptance)
            if len(acceptance) != len(p.lengths):
                raise ModelingError("recurrent_kda_state_acceptance_must_match_sequences")
            p = replace(p, state_acceptance=acceptance)
        if p.state_targets is not None:
            targets = tuple(p.state_targets)
            if (len(targets) != len(p.lengths)
                    or any(not isinstance(target, str) or not target
                           for target in targets)
                    or len(set(targets)) != len(targets)):
                raise ModelingError("recurrent_kda_state_targets_invalid")
            p = replace(p, state_targets=targets)
        if p.use_gate_in_kernel and not p.has_a_log:
            raise ModelingError("recurrent_kda_source_host_requires_a_log_for_raw_gate")
        if p.metadata_dtype not in ("int32", "int64"):
            raise ModelingError("recurrent_kda_metadata_dtype_unsupported")
        return p


def buffer_profile(p, ub_bytes):
    """Exact host candidate order, capacity arithmetic and kernel allocations."""
    k, v, hv = p.dk, p.dv, align_up(p.heads, HEAD_TILE_ALIGNMENT)
    fixed = (KDA_MAX_QUERY_ROWS * (FP32_BYTES * (2 * k + hv) + 2 * v)
             + SYNC_BUFFER_BYTES + HOST_UB_GUARD_BYTES)
    working = fixed + KDA_MAX_QUERY_ROWS * FP32_BYTES * (v + HOST_KEY_WORKSPACE_FP32_ROWS * k + hv)
    state_bytes = _BYTES[p.state_dtype]
    candidates = []
    for state_slots, attention_slots in ((1, 1), (1, 2), (2, 2)):
        queues = (1 + state_slots) * state_bytes * k + attention_slots * 2
        step = ((ub_bytes - working) // (queues + KDA_MAX_QUERY_ROWS * (k + 1))
                // VALUE_TILE_ALIGNMENT * VALUE_TILE_ALIGNMENT)
        if step < VALUE_TILE_ALIGNMENT:
            continue
        parts = math.ceil(v / step)
        step = align_up(math.ceil(v / parts), VALUE_TILE_ALIGNMENT)
        candidates.append((parts, -(state_slots + attention_slots), -step,
                           state_slots, attention_slots, queues))
    if not candidates:
        raise ModelingError("recurrent_kda_source_ub_has_no_valid_value_tile")
    parts, _, negative_step, state_slots, attention_slots, queues = min(candidates)
    step = -negative_step
    rest = ub_bytes - fixed - queues * step
    # Kernel beta allocation is ceil8(8*HV)*4; host reserves 8*ceil16(HV)*4.
    # The host's 2048-byte guard is not an actual InitBuffer allocation.
    host_beta_rows = KDA_MAX_QUERY_ROWS * hv
    actual_beta_rows = align_up(KDA_MAX_QUERY_ROWS * p.heads, BETA_LANE_ALIGNMENT)
    actual = ub_bytes - HOST_UB_GUARD_BYTES - (host_beta_rows - actual_beta_rows) * FP32_BYTES
    return {"value_step": step, "parts": parts, "state_slots": state_slots,
            "attention_slots": attention_slots, "ub_rest_bytes": rest,
            "ub_bytes_per_owner": actual, "host_ub_capacity": ub_bytes}


def _owner_segments(lengths, heads, blocks, owner):
    for seq, q in enumerate(lengths):
        start, end = seq*heads, (seq+1)*heads
        first = owner + max(0, math.ceil((start-owner) / blocks)) * blocks
        amount = max(0, (end - 1 - first) // blocks + 1)
        if amount:
            yield q, amount


def scheduled_runs(lengths, heads, blocks, owner, *, compact=True, beta_per_head=True):
    """Count distinct heads and actual beta loads, without tensor sharing."""
    if compact and (beta_per_head or heads <= blocks):
        # Complete launch waves have equal work on every owner. Keep their
        # partial boundaries explicit: scaling one owner's 2 heads alongside
        # another owner's 1 head would otherwise rephase shared HBM service.
        start, previous, sequences = 0, lengths[0], 0
        spans = []
        for q in lengths:
            if q != previous:
                end = start + sequences*heads
                spans.append((previous, start, end))
                start, previous, sequences = end, q, 0
            sequences += 1
        spans.append((previous, start, start+sequences*heads))
        for q, start, end in spans:
            first_full = math.ceil(start/blocks)*blocks
            last_full = end//blocks*blocks
            leading = start//blocks*blocks + owner
            if start <= leading < min(first_full, end):
                yield q, 1, 1
            count = max(0, (last_full-first_full)//blocks)
            if count:
                yield q, count, count
            trailing = end//blocks*blocks + owner
            if max(start, first_full, last_full) <= trailing < end:
                yield q, 1, 1
        return
    # The arch22 within-sequence beta pattern alternates real loads and reuse.
    # These are different physical cycles, so retain their source head order.
    for q, amount in _owner_segments(lengths, heads, blocks, owner):
        for head in range(amount):
            yield q, 1, int(beta_per_head or head == 0)


def owner_runs(lengths, heads, blocks, owner, *, compact=True):
    """Round-robin source task IDs, compressed without expanding B*HV heads."""
    for q, count, _ in scheduled_runs(lengths, heads, blocks, owner, compact=compact):
        yield q, count


def beta_work(p, q, read):
    incoming = q*p.heads*_BYTES[p.beta_dtype]
    return Work(resources={"hbm": incoming, read: incoming, "mte3": 0,
                           "vector": align_up(q * p.heads, BETA_LANE_ALIGNMENT)})


def head_work(p, q, parts, read, *, include_beta=True, delta_broadcast=False):
    """Payload and vector operations for one physical (sequence,value-head)."""
    k, v = p.dk, p.dv
    state = k*v*_BYTES[p.state_dtype]
    incoming = q*((2*k+v)*2 + k*_BYTES[p.gate_dtype]) + state
    gate = q * GATE_QUEUE_OPS_PER_LANE * k  # gate queue zero, gate convert, final exp
    if p.use_gate_in_kernel:
        incoming += FP32_BYTES * (int(p.has_a_log) + k * int(p.has_dt_bias))
        gate += q*k*((SAFE_GATE_OPS_PER_LANE if p.safe_gate else GATE_OPS_PER_LANE) + int(p.has_dt_bias)) + A_LOG_TRANSFORM_OPS * int(p.has_a_log)
    vector = (q * (QK_PREP_OPS_PER_LANE * k + v) + gate
              + (QK_NORM_EXTRA_OPS_PER_LANE * q * k if p.use_qk_l2norm else 0) + k*v
              + q*((RECURRENCE_OPS_PER_STATE_ELEMENT + 1 + int(delta_broadcast))*k+1)*v + 2*q*parts*int(p.use_beta_sigmoid))
    state_out = record_head_bytes(k, v, p.dtype) if (p.recover_ssm and q > 1) else state
    outgoing = q*(state_out + v*2)
    result = Work(resources={"hbm": incoming + outgoing, read: incoming,
                             "mte3": outgoing, "vector": vector})
    return result + beta_work(p, q, read) if include_beta else result


def _head_stages(p, q, repeat, profile, context, placement, read,
                 *, beta_repetitions=None, delta_broadcast=False,
                 cohort_size=1, cohort_widened=(), cohort_members=None):
    """Source queue order; delayed exports preserve their finite slot reuse."""
    stages = []
    c_members = cohort_members or {}
    def add(name, work=Work(), after=(), count=None):
        scale = (repeat if count is None else count) * cohort_size
        stages.append(Stage(
            name, context.bind(work.scaled(scale)),
            after=tuple(after), placement=placement,
            cohort_size=cohort_size, cohort_widened=cohort_widened,
            cohort_members=c_members,
        ))
        return name
    def vec(name, amount, *after, count=None):
        return add(name, Work(resources={"vector": amount}), after, count)
    def mem(name, amount, role, *after, count=None):
        return add(name, Work(resources={"hbm": amount, role: amount}), after, count)
    k, v = p.dk, p.dv
    beta = mem("beta_load", q*p.heads*_BYTES[p.beta_dtype], read, count=beta_repetitions)
    beta = vec("beta_convert", align_up(q * p.heads, BETA_LANE_ALIGNMENT), beta, count=beta_repetitions)
    qkv = mem("qkv_load", q*(2*k+v)*2, read, beta)
    zero = vec("gate_zero", q*k, beta)
    gate = mem("gate_load", q*k*_BYTES[p.gate_dtype], read, qkv, zero)
    gate = vec("gate_convert", q*k, gate)
    ready = vec("qkv_convert", q*(2*k+v), gate)
    ready = vec("qk_norm_scale", q*k*(1 + QK_NORM_EXTRA_OPS_PER_LANE * int(p.use_qk_l2norm)), ready)
    if p.use_gate_in_kernel:
        if p.has_a_log:
            ready = mem("a_log_load", FP32_BYTES, read, ready)
            ready = vec("a_log_exp", A_LOG_TRANSFORM_OPS, ready)
        if p.has_dt_bias:
            ready = mem("dt_bias_load", k * FP32_BYTES, read, ready)
        ready = vec("gate_activation", q*k*((SAFE_GATE_OPS_PER_LANE if p.safe_gate else GATE_OPS_PER_LANE) + int(p.has_dt_bias)), ready)
    ready = vec("gate_exp", q*k, ready)
    step = profile["value_step"]
    widths = tuple(min(step, v-offset) for offset in range(0, v, step))
    state_load = mem("state_load0", k*widths[0]*_BYTES[p.state_dtype], read, ready)
    last_compute = ready
    last_write = None
    attention_writes, state_writes = [], []
    for part, n in enumerate(widths):
        state_slice = (record_slice_bytes(k, n, part, p.dtype)
                       if (p.recover_ssm and q > 1) else k*n*_BYTES[p.state_dtype])
        resident = vec(f"state_resident{part}", k*n, state_load, last_compute)
        if part+1 < len(widths):
            state_load = mem(f"state_load{part+1}", k*widths[part+1]*_BYTES[p.state_dtype], read, resident)
        last_compute = resident
        pending_attn = pending_state = None
        def write(kind, token, producer, amount):
            nonlocal last_write
            deps = (producer, last_compute) + ((last_write,) if last_write else ())
            name = mem(f"{kind}_write{part}_{token}", amount, "mte3", *deps)
            last_write = name
            (attention_writes if kind == "attention" else state_writes).append(name)
        for token in range(q):
            suffix = f"{part}_{token}"
            recurrence = vec(f"recur{suffix}", (RECURRENCE_OPS_PER_STATE_ELEMENT + int(delta_broadcast))*k*n
                             + 2*int(p.use_beta_sigmoid), last_compute)
            # Compute allocates attention first, then state. Queue slots are
            # reused only after the corresponding prior DMA has completed.
            deps = [recurrence]
            completed = part*q + token
            if completed >= profile["attention_slots"]:
                deps.append(attention_writes[completed-profile["attention_slots"]])
            if completed >= profile["state_slots"]:
                deps.append(state_writes[completed-profile["state_slots"]])
            exported = vec(f"state_export{suffix}", k*n, *deps)
            last_compute = vec(f"attention_cast{suffix}", n, exported)
            if profile["attention_slots"] == 1:
                write("attention", token, last_compute, n*2)
            else:
                if pending_attn is not None:
                    previous_token, producer = pending_attn
                    write("attention", previous_token, producer, n*2)
                pending_attn = token, last_compute
            if profile["state_slots"] == 1:
                write("state", token, exported, state_slice)
            else:
                if pending_state is not None:
                    previous_token, producer = pending_state
                    write("state", previous_token, producer, state_slice)
                pending_state = token, exported
        if pending_attn is not None:
            write("attention", *pending_attn, n*2)
        if pending_state is not None:
            write("state", *pending_state, state_slice)
    return tuple(stages)


def summarize_work(params, geometry, *, execution=SourceExecution()):
    """Canonical frontend/source-plan work for one complete KDA launch."""
    p = params if isinstance(params, Params) else Params.parse(params)
    units = geometry.resource_units.get("vector", geometry.compute_units)
    blocks = min(units, len(p.lengths)*p.heads)
    if "ub" not in geometry.buffers:
        raise ModelingError("recurrent_kda_requires_declared_ub_capacity")
    profile = buffer_profile(p, geometry.buffers["ub"])
    read = read_role(geometry.resource_units, aiv=0)
    work = Work()
    for owner in range(blocks):
        for q, heads, beta_count in scheduled_runs(
                p.lengths, p.heads, blocks, owner,
                beta_per_head=execution.beta_per_head):
            work += head_work(
                p, q, profile["parts"], read, include_beta=False,
                delta_broadcast=execution.delta_broadcast).scaled(heads)
            work += beta_work(p, q, read).scaled(beta_count)
    return work


@dataclass(frozen=True)
class _AdaptiveWaveCohortProgram:
    """Run lockstep wave cohort iff every owner's UB admits simultaneously without contention."""

    folded: KernelProgram
    literal_factory: Callable[[], ParallelProgram]
    ub_pool: str
    ub_bytes: float
    ub_units: tuple[int, ...]

    def _can_admit_lockstep(self, engine) -> bool:
        cap = engine.pools.get(self.ub_pool)
        if cap is None or self.ub_bytes > cap:
            return False
        if self.ub_pool in getattr(engine, "memory_pools", ()):
            return False
        pool_mask = getattr(engine, "_pool_unit_masks", {}).get(self.ub_pool, 0)
        if pool_mask:
            unit_alloc = engine.unit_allocated
            for unit in self.ub_units:
                if unit_alloc.get((self.ub_pool, unit), 0.0) + self.ub_bytes > cap:
                    return False
        ub_set = set(self.ub_units)
        for task in engine._tasks.values():
            if task.start is None and task.acquire:
                for alloc in task.acquire:
                    if alloc.pool == self.ub_pool and ub_set.intersection(alloc.units):
                        return False
        return True

    def bind_stream_owners(self, owners, datapath, **kwargs):
        from mindie_sim.prediction.streams import apply_stream_owners_to_program

        return apply_stream_owners_to_program(
            self.literal_factory(), owners, datapath, **kwargs,
        )

    def start(self, engine, name: str, ready, *, owner: str = ""):
        if ready.done and ready.time <= engine.now and not self._can_admit_lockstep(engine):
            return self.literal_factory().start(engine, name, ready, owner=owner)
        complete = engine.future(name + ":kda_complete")

        def _dispatch(trigger) -> None:
            chosen = self.folded if self._can_admit_lockstep(engine) else self.literal_factory()
            finish = chosen.start(engine, name, trigger, owner=owner)
            engine.on_complete(finish, lambda _ev: engine.resolve(complete), native_hook=True)

        if ready.done and ready.time <= engine.now:
            engine.submit(
                name + ":admit", after=(ready,), owner=owner,
                on_complete=_dispatch, native_hook=True,
            )
            return complete
        engine.on_complete(ready, _dispatch, native_hook=True)
        return complete


def build(op, geometry, *, compact=True, execution=SourceExecution()):
    p = Params.parse(dict(op.metadata)["kda_recurrent_params"])
    units = geometry.resource_units.get("vector", geometry.compute_units)
    blocks = min(units, len(p.lengths)*p.heads)
    if "ub" not in geometry.buffers:
        raise ModelingError("recurrent_kda_requires_declared_ub_capacity")
    profile = buffer_profile(p, geometry.buffers["ub"])
    read = read_role(geometry.resource_units, aiv=0)
    def runs(owner):
        # Execute exact per-head cycles so shared-HBM contention and competing
        # tasks match literal Engine replay without fluid stage-scaling drift.
        return scheduled_runs(p.lengths, p.heads, blocks, owner, compact=False,
                              beta_per_head=execution.beta_per_head)
    work = summarize_work(p, geometry, execution=execution)
    cycles = []
    for owner in range(blocks):
        count = sum(1 for _ in runs(owner))
        cycles.append(count)
    cycles = tuple(cycles)
    fragments = state_fragments(op)
    def program(context):
        def _disjoint_cohort_binding(cohort_owners):
            """Merge ``cohort_owners`` into one exact symmetric cohort when placements & UB units are disjoint."""
            roles = ("vector", read, "mte3")
            per_owner_pl = [context.owned(o, units, roles) for o in cohort_owners]
            pl_keys = tuple(per_owner_pl[0].keys())
            merged_pl: dict[str, tuple[int, ...]] = {}
            widened: list[str] = []
            members: dict[str, tuple[tuple[int, ...], ...]] = {}
            for rk in pl_keys:
                rows = tuple(tuple(pl[rk]) for pl in per_owner_pl)
                if len({len(r) for r in rows}) != 1:
                    return None
                flat = tuple(u for r in rows for u in r)
                if len(set(flat)) != len(flat):
                    return None
                merged_pl[rk] = flat
                widened.append(rk)
                members[rk] = rows
            ub_pool_units = context.pool_units.get("ub", units)
            ub_rows = tuple(tuple(unit_group(o, units, ub_pool_units)) for o in cohort_owners)
            ub_flat = tuple(u for r in ub_rows for u in r)
            if len(set(ub_flat)) != len(ub_flat):
                return None
            return merged_pl, tuple(widened), members, tuple(sorted(ub_flat))

        def owner_program(owner):
            placement = context.owned(owner, units, ("vector", read, "mte3"))
            pending = tuple(runs(owner))
            stage_cache: dict[tuple[int, int, int], tuple[Stage, ...]] = {}
            def head(index):
                key = pending[index]
                cached = stage_cache.get(key)
                if cached is not None:
                    return cached
                q, count, beta_count = key
                built = _head_stages(p, q, count, profile, context, placement, read,
                                     beta_repetitions=beta_count,
                                     delta_broadcast=execution.delta_broadcast)
                stage_cache[key] = built
                return built
            heads = PipelineProgram(cycles[owner], 1, head, interior_fold=False)
            allocation = (Allocation("kda_buffers", context.pool_names["ub"],
                                     profile["ub_bytes_per_owner"],
                                     unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("heads", after=("buffers",), program=heads),
                                  Stage("release", after=("heads",), free=allocation)))

        def _wave_cohort_program():
            """Fold lockstep owner waves into one active symmetric cohort at a time.

            When every wave ``w`` has identical ``(q, count, beta_count)`` across
            all active owners and disjoint per-owner pipe/UB placements, running
            wave ``w`` as a single ``cohort_size = len(active_owners)`` task
            preserves exact per-unit UB release times, fair-share HBM weight,
            and zero-ULP duration while avoiding concurrent multi-cohort breaks.
            """
            per_owner_runs = [tuple(runs(o)) for o in range(blocks)]
            max_waves = max((len(r) for r in per_owner_runs), default=0)
            if max_waves <= 0:
                return None
            # Every wave must be homogeneous across all owners still active in that wave.
            wave_specs: list[tuple[tuple[int, int, int], tuple[int, ...]]] = []
            for w_idx in range(max_waves):
                active = [o for o in range(blocks) if w_idx < len(per_owner_runs[o])]
                keys = {per_owner_runs[o][w_idx] for o in active}
                if len(keys) != 1:
                    return None
                wave_specs.append((next(iter(keys)), tuple(active)))
            full_bound = _disjoint_cohort_binding(tuple(range(blocks)))
            if full_bound is None:
                return None
            # Group owners by retirement wave so each owner's UB frees at its exact last wave.
            by_retire: dict[int, list[int]] = {}
            for o in range(blocks):
                by_retire.setdefault(len(per_owner_runs[o]) - 1, []).append(o)
            ub_pool = context.pool_names["ub"]
            ub_pool_units = context.pool_units.get("ub", units)
            ub_bytes = profile["ub_bytes_per_owner"]
            retire_allocs: dict[int, tuple[Allocation, ...]] = {}
            all_allocs: list[Allocation] = []
            for r_wave, r_owners in sorted(by_retire.items()):
                ub_u = tuple(sorted(
                    u for o in r_owners for u in unit_group(o, units, ub_pool_units)
                ))
                alloc = Allocation(f"kda_buffers_w{r_wave}", ub_pool, ub_bytes, ub_u)
                retire_allocs[r_wave] = (alloc,)
                all_allocs.append(alloc)
            # Group consecutive waves with identical active owners into one PipelineProgram.
            binding_cache: dict[tuple[int, ...], tuple] = {tuple(range(blocks)): full_bound}
            prog_stages: list[Stage] = [Stage("buffers", acquire=tuple(all_allocs))]
            prev_stage = "buffers"
            w_pos = 0
            seg_idx = 0
            while w_pos < max_waves:
                seg_owners = wave_specs[w_pos][1]
                seg_start = w_pos
                while w_pos < max_waves and wave_specs[w_pos][1] == seg_owners:
                    w_pos += 1
                seg_keys = tuple(wave_specs[i][0] for i in range(seg_start, w_pos))
                bound = binding_cache.get(seg_owners)
                if bound is None:
                    bound = _disjoint_cohort_binding(seg_owners)
                    if bound is None:
                        return None
                    binding_cache[seg_owners] = bound
                pl_s, wid_s, mem_s, _ = bound
                c_size = len(seg_owners)
                stage_cache: dict[tuple[int, int, int], tuple[Stage, ...]] = {}
                def _make_head(keys_t, pl_t, wid_t, mem_t, cs_t, cache_t):
                    def _head(index):
                        key = keys_t[index]
                        cached = cache_t.get(key)
                        if cached is not None:
                            return cached
                        q, count, beta_count = key
                        built = _head_stages(
                            p, q, count, profile, context, pl_t, read,
                            beta_repetitions=beta_count,
                            delta_broadcast=execution.delta_broadcast,
                            cohort_size=cs_t,
                            cohort_widened=wid_t if cs_t > 1 else (),
                            cohort_members=mem_t if cs_t > 1 else None,
                        )
                        cache_t[key] = built
                        return built
                    return _head
                seg_pipe = PipelineProgram(
                    len(seg_keys), 1,
                    _make_head(seg_keys, pl_s, wid_s, mem_s, c_size, stage_cache),
                    interior_fold=False,
                )
                seg_name = f"heads_seg{seg_idx}"
                prog_stages.append(Stage(seg_name, after=(prev_stage,), program=seg_pipe))
                prev_stage = seg_name
                last_wave = w_pos - 1
                if last_wave in retire_allocs:
                    rel_name = f"release_seg{seg_idx}"
                    prog_stages.append(Stage(rel_name, after=(seg_name,), free=retire_allocs[last_wave]))
                    if w_pos >= max_waves:
                        prev_stage = rel_name
                seg_idx += 1
            return KernelProgram(tuple(prog_stages)), ub_pool, ub_bytes, full_bound[3]

        if compact and blocks > 1:
            wave_result = _wave_cohort_program()
            if wave_result is not None:
                wave_prog, ub_pool, ub_bytes, ub_units = wave_result
                owners = _AdaptiveWaveCohortProgram(
                    wave_prog,
                    lambda: ParallelProgram(tuple(owner_program(owner) for owner in range(blocks))),
                    ub_pool,
                    ub_bytes,
                    ub_units,
                )
            else:
                owners = ParallelProgram(tuple(owner_program(owner) for owner in range(blocks)))
        else:
            owners = (
                owner_program(0)
                if blocks == 1
                else ParallelProgram(tuple(owner_program(owner) for owner in range(blocks)))
            )
        stages = []
        if fragments["read"] is not None and fragments["read"].state_effects:
            stages.append(Stage("state_read", fragment=fragments["read"]))
            stages.append(Stage("recurrent_owners", after=("state_read",),
                                program=owners))
        else:
            stages.append(Stage("recurrent_owners", program=owners))
        if fragments["write"] is not None and fragments["write"].state_effects:
            stages.append(Stage("state_publish", after=("recurrent_owners",),
                                fragment=fragments["write"]))
        return KernelProgram(tuple(stages))
    scalar_bytes = int(dtype_bytes(p.metadata_dtype))
    validated_sequences = blocks*len(p.lengths) if execution.validate_all_sequences else len(p.lengths)*p.heads
    resolved_sequences = (len(p.lengths)*min(p.heads, blocks)
                          if execution.validate_all_sequences else len(p.lengths)*p.heads)
    metadata_values = blocks*(len(p.lengths)+1) + 2*validated_sequences
    if p.has_state_indices:
        metadata_values += (blocks if execution.validate_all_sequences else p.heads)*sum(p.lengths)
        metadata_values += p.heads*sum(p.lengths)*profile["parts"] + resolved_sequences
    accepted_loads = validated_sequences + resolved_sequences if p.has_accepted_tokens else 0
    metadata_values += accepted_loads
    spec_recover = bool(p.recover_ssm and any(q > 1 for q in p.lengths))
    plan_metadata = {"execution_class": "vector", "state_accesses": state_accesses(op)}
    if spec_recover:
        plan_metadata["recover_ssm"] = True
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is not None:
        plan_metadata["state_contract"] = contract
        plan_metadata["state_effects"] = dict(op.metadata).get("kda_state_effects", ())
    plan_metadata["state_fragments"] = {
        "read": "state_read" if fragments["read"] is not None else None,
        "write": "state_publish" if fragments["write"] is not None else None,
        "commits": len(fragments["commits"]), "rollbacks": len(fragments["rollbacks"]),
    }
    plan_metadata["kda_recurrent_schedule"] = {
            **profile, "logical_heads": len(p.lengths)*p.heads,
            "active_aiv": blocks, "physical_aiv": units, "owner_cycles": cycles,
            "resource_cycles": sum(cycles), "compact": compact,
            "compaction_domain": ("complete_owner_waves_with_explicit_partial_tails"
                                  if execution.beta_per_head or p.heads <= blocks
                                  else "distinct_beta_load_reuse_cycles_remain_detailed"),
            "beta_load_scope": "head" if execution.beta_per_head else "owner_sequence",
            "delta_update": "UB_broadcast_MulAddDst_query_Mul" if execution.delta_broadcast else "fused_MicroAPI_ProcessKQ",
            "ownership": "task_id=(sequence*HV+head);owner=task_id%launched_aiv",
            "state_residency": ("one_initial_load_per_head;recover_ssm_compact_record_then_single_commit"
                                if spec_recover else "one_initial_load_per_head;checkpoint_write_every_query"),
            "head_handoff": "conservative_serial_head_retirement",
            "contention_approximation": "none",
            "scalar_metadata_getvalue_bytes": scalar_bytes*metadata_values,
            "accepted_metadata_getvalue_bytes": scalar_bytes*accepted_loads,
            "unmodeled_services": ("scalar_GetValue_cache_and_issue", "queue_flag_and_vector_instruction_issue"),
        }
    return OperatorPlan(work, program, metadata=plan_metadata)


def make_op(params, *, op_id="kda_recurrent", deps=(), requests=(), stage=0,
            layer=0, phase="decode", stream="", notes=(), state_contexts=None,
            state_targets=None, state_values=None, state_slots=None,
            state_ready=(), state_grants=(), state_decisions=(),
            state_generation=None):
    raw_params = dict(params)
    if state_contexts is not None:
        ctx_tuple = tuple(state_contexts)
        if (raw_params.get("state_contexts") is not None
                and tuple(raw_params["state_contexts"]) != ctx_tuple):
            raise ModelingError("recurrent_kda_state_contexts_conflict")
        raw_params["state_contexts"] = ctx_tuple
    if state_targets is not None:
        tgt_tuple = tuple(state_targets)
        if (raw_params.get("state_targets") is not None
                and tuple(raw_params["state_targets"]) != tgt_tuple):
            raise ModelingError("recurrent_kda_state_targets_conflict")
        raw_params["state_targets"] = tgt_tuple
    p = Params.parse(raw_params)
    shape = (sum(p.lengths), p.heads, p.dk, p.dv)
    spec_recover = bool(p.recover_ssm and any(q > 1 for q in p.lengths))
    metadata = {
        "kda_recurrent_params": dict(p.__dict__), "native_kind": "RecurrentKda",
        "implementation": IMPLEMENTATION, "source": SOURCE,
        "invocation_class": "vector", "program_kind": "recurrent_vector",
        "heads": p.heads, "key_heads": p.key_heads, "dk": p.dk, "dv": p.dv,
        "sequence_tokens": p.lengths, "state_dtype": p.state_dtype,
        "state_mutation": "recover_ssm_compact_record" if spec_recover else "inplace_per_query_checkpoint",
        "source_binding": "explicit_arch35_kernel_selection_not_binary_equivalence",
        "caller_materializations": "outside_selected_kernel",
        **({"recover_ssm": True} if spec_recover else {}),
    }
    selection = resolve_state_selection(p.state_selection, p.lengths)
    if selection is not None:
        metadata["state_selection"] = selection
        deps = tuple(dict.fromkeys((*deps, *selection["dependencies"])))
    if p.state_targets is not None:
        targets = p.state_targets
    else:
        request_ids = tuple(requests)
        layer_target = f"kda.layer[{layer}]" if layer >= 0 else "kda.call"
        targets = tuple(
            f"{layer_target}.recurrent_state["
            f"{request_ids[index] if index < len(request_ids) else index}]"
            for index in range(len(p.lengths)))
    if state_slots is None and selection is not None:
        state_slots = selection.get("initial_slots")
    contexts = p.state_contexts
    if contexts is None:
        derived = _value_versions(state_values, targets, len(p.lengths))
        contexts = tuple(0 if value is None else value for value in derived) \
            if derived is not None else (0,) * len(p.lengths)
    owner = f"kda.layer[{layer}]" if layer >= 0 else "kda.call"
    bindings = bind_sequence_values(
        targets=targets, lengths=p.lengths, contexts=contexts,
        shape=(p.heads, p.dv, p.dk), dtype=p.state_dtype, owner=owner,
        values=state_values, slots=state_slots, ready=state_ready,
        grants=state_grants, generation=state_generation, required=True,
        representation="state")
    contract, effects = state_contract(
        lengths=p.lengths, targets=targets, requests=requests,
        selection=selection, contexts=contexts,
        acceptance=p.state_acceptance, bindings=bindings, owner=owner,
        state_dtype=p.state_dtype, record_dtype=p.dtype, decisions=state_decisions,
        ready=state_ready, grants=state_grants, op_id=op_id,
        recover_ssm=p.recover_ssm, state_values=state_values,
        record_stride=p.record_stride)
    metadata.update(
        kda_state_contract=contract, kda_state_effects=effects,
        state_targets=targets, state_bindings=tuple(
            binding_record(binding) for binding in bindings),
        state_checkpoint_versions=contract["checkpoint_versions"],
        state_read_versions=contract["read_versions"],
        state_commit_versions=contract["commit_versions"])
    op = OpSpec(op_id, "recurrent_scan", phase, stage, layer, tuple(requests),
                shape, shape, shape, shape, deps=tuple(deps), dtype="bf16", layout="TND",
                recipe=RECIPE, source_boundary="kernel", coverage="source_selected_recurrent_kernel",
                metadata=tuple(sorted(metadata.items())), stream=stream, notes=tuple(notes))
    summary = summarize_work(p, frontend_geometry())
    op = summary.apply(op)
    physical_read = float(sum(
        amount for role, amount in summary.resources.items()
        if role in {"mte2", "mte2_aiv"}))
    if physical_read:
        op = replace(op, mte2_bytes=max(float(op.mte2_bytes), physical_read))
    return replace(op, metadata=op.metadata + (
        ("state_accesses", state_accesses(op)),))
