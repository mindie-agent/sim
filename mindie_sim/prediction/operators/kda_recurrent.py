"""Source-selected vector KDA decode with resident state and finite queues.

[Role] Lower vLLM-Ascend RecurrentKda arch35 to the common physical Engine.
[Invariants] One launch spans all sequences. QKV/gates remain resident per
  value head; initial state is read once, every query writes a checkpoint.
  Host-selected AIV ownership and UB queues never borrow idle-core supply.
  Repeated heads scale real work, not measured duration or retained state.
[Timeouts & Locks] O(sequences * active owners) bounded runs; q <= 8. No locks.
[Side-Effect Policy] Pure plans and virtual resource tasks; no runtime/source I/O.
"""
from dataclasses import dataclass, replace
import math

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from .kda_state_selection import resolve as resolve_state_selection

IMPLEMENTATION = RECIPE = "vllm_ascend_recurrent_kda_arch35"
SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "source_state": "committed",
    "kernel": {
        "path": "csrc/attention/recurrent_kda/op_kernel/arch35/recurrent_kda.h",
        "sha256": "f35dd2c5fb604d6d9711c11139c755414d49fb27fc6aea0f3a4bec3d3adb0199",
        "lines": ((151, 198), (235, 270), (375, 596), (736, 920)),
    },
    "host": {
        "path": "csrc/attention/recurrent_kda/op_host/recurrent_kda_tiling_processor.h",
        "sha256": "c9eac28bd528eb03bdeb456c4fdbed4951f716937570ee9d810e0fb5b3ef3fbd",
        "lines": ((592, 621), (648, 756)),
    },
    "caller": {
        "path": "vllm_ascend/ops/kda.py",
        "sha256": "ca2f56280746f45a347578e7191decda716557c93aca69b5c3baf7b22a578803",
    },
    "scope": "selected_source_kernel_not_capture_binary_identity",
}
_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


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
    dk: int = 128
    dv: int = 128
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

    @classmethod
    def parse(cls, raw):
        p = cls(**{**raw, "lengths": tuple(raw["lengths"])})
        if not p.lengths or any(not isinstance(q, int) or not 1 <= q <= 8 for q in p.lengths):
            raise ModelingError("recurrent_kda_requires_nonempty_q1_to_q8_sequences")
        if (not 1 <= p.key_heads <= p.heads <= 256 or p.heads % p.key_heads
                or p.dk != 128 or p.dv not in (128, 256)):
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
        if p.use_gate_in_kernel and not p.has_a_log:
            raise ModelingError("recurrent_kda_source_host_requires_a_log_for_raw_gate")
        if p.metadata_dtype not in ("int32", "int64"):
            raise ModelingError("recurrent_kda_metadata_dtype_unsupported")
        return p


def buffer_profile(p, ub_bytes):
    """Exact host candidate order, capacity arithmetic and kernel allocations."""
    k, v, hv = p.dk, p.dv, math.ceil(p.heads / 16) * 16
    fixed = 8 * (4*k + 2*v) + 8*4*k + 8*4*hv + 64 + 2048
    working = fixed + 8 * (4*v + 12*k + 4*hv)
    state_bytes = _BYTES[p.state_dtype]
    candidates = []
    for state_slots, attention_slots in ((1, 1), (1, 2), (2, 2)):
        queues = (1 + state_slots) * state_bytes * k + attention_slots * 2
        step = (ub_bytes - working) // (queues + 8*k + 8) // 8 * 8
        if step < 8:
            continue
        parts = math.ceil(v / step)
        step = math.ceil(math.ceil(v / parts) / 8) * 8
        candidates.append((parts, -(state_slots + attention_slots), -step,
                           state_slots, attention_slots, queues))
    if not candidates:
        raise ModelingError("recurrent_kda_source_ub_has_no_valid_value_tile")
    parts, _, negative_step, state_slots, attention_slots, queues = min(candidates)
    step = -negative_step
    rest = ub_bytes - fixed - queues * step
    # Kernel beta allocation is ceil8(8*HV)*4; host reserves 8*ceil16(HV)*4.
    # The host's 2048-byte guard is not an actual InitBuffer allocation.
    actual = ub_bytes - 2048 - (8*hv - math.ceil(8*p.heads / 8)*8) * 4
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
                           "vector": math.ceil(q*p.heads/8)*8})


def head_work(p, q, parts, read, *, include_beta=True, delta_broadcast=False):
    """Payload and vector operations for one physical (sequence,value-head)."""
    k, v = p.dk, p.dv
    state = k*v*_BYTES[p.state_dtype]
    incoming = q*((2*k+v)*2 + k*_BYTES[p.gate_dtype]) + state
    gate = q*3*k  # gate queue zero, gate convert, final exp
    if p.use_gate_in_kernel:
        incoming += 4*int(p.has_a_log) + k*4*int(p.has_dt_bias)
        gate += q*k*((7 if p.safe_gate else 4) + int(p.has_dt_bias)) + 3*int(p.has_a_log)
    vector = (q*(3*k+v) + gate
              + (6*q*k if p.use_qk_l2norm else 0) + k*v
              + q*((8+int(delta_broadcast))*k+1)*v + 2*q*parts*int(p.use_beta_sigmoid))
    outgoing = q*(state + v*2)
    result = Work(resources={"hbm": incoming + outgoing, read: incoming,
                             "mte3": outgoing, "vector": vector})
    return result + beta_work(p, q, read) if include_beta else result


def _head_stages(p, q, repeat, profile, context, placement, read,
                 *, beta_repetitions=None, delta_broadcast=False):
    """Source queue order; delayed exports preserve their finite slot reuse."""
    stages = []
    def add(name, work=Work(), after=(), count=None):
        stages.append(Stage(name, context.bind(work.scaled(repeat if count is None else count)),
                            after=tuple(after), placement=placement))
        return name
    def vec(name, amount, *after, count=None):
        return add(name, Work(resources={"vector": amount}), after, count)
    def mem(name, amount, role, *after, count=None):
        return add(name, Work(resources={"hbm": amount, role: amount}), after, count)
    k, v = p.dk, p.dv
    beta = mem("beta_load", q*p.heads*_BYTES[p.beta_dtype], read, count=beta_repetitions)
    beta = vec("beta_convert", math.ceil(q*p.heads/8)*8, beta, count=beta_repetitions)
    qkv = mem("qkv_load", q*(2*k+v)*2, read, beta)
    zero = vec("gate_zero", q*k, beta)
    gate = mem("gate_load", q*k*_BYTES[p.gate_dtype], read, qkv, zero)
    gate = vec("gate_convert", q*k, gate)
    ready = vec("qkv_convert", q*(2*k+v), gate)
    ready = vec("qk_norm_scale", q*k*(7 if p.use_qk_l2norm else 1), ready)
    if p.use_gate_in_kernel:
        if p.has_a_log:
            ready = mem("a_log_load", 4, read, ready)
            ready = vec("a_log_exp", 3, ready)
        if p.has_dt_bias:
            ready = mem("dt_bias_load", k*4, read, ready)
        ready = vec("gate_activation", q*k*((7 if p.safe_gate else 4) + int(p.has_dt_bias)), ready)
    ready = vec("gate_exp", q*k, ready)
    step = profile["value_step"]
    widths = tuple(min(step, v-offset) for offset in range(0, v, step))
    state_load = mem("state_load0", k*widths[0]*_BYTES[p.state_dtype], read, ready)
    last_compute = ready
    last_write = None
    attention_writes, state_writes = [], []
    for part, n in enumerate(widths):
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
            recurrence = vec(f"recur{suffix}", (7+int(delta_broadcast))*k*n
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
                write("state", token, exported, k*n*_BYTES[p.state_dtype])
            else:
                if pending_state is not None:
                    previous_token, producer = pending_state
                    write("state", previous_token, producer, k*n*_BYTES[p.state_dtype])
                pending_state = token, exported
        if pending_attn is not None:
            write("attention", *pending_attn, n*2)
        if pending_state is not None:
            write("state", *pending_state, k*n*_BYTES[p.state_dtype])
    return tuple(stages)


def build(op, geometry, *, compact=True, execution=SourceExecution()):
    p = Params.parse(dict(op.metadata)["kda_recurrent_params"])
    units = geometry.resource_units.get("vector", geometry.compute_units)
    blocks = min(units, len(p.lengths)*p.heads)
    if "ub" not in geometry.buffers:
        raise ModelingError("recurrent_kda_requires_declared_ub_capacity")
    profile = buffer_profile(p, geometry.buffers["ub"])
    read = read_role(geometry.resource_units, aiv=0)
    def runs(owner):
        return scheduled_runs(p.lengths, p.heads, blocks, owner, compact=compact,
                              beta_per_head=execution.beta_per_head)
    work, cycles = Work(), []
    for owner in range(blocks):
        count = 0
        for q, heads, beta_count in runs(owner):
            work += head_work(p, q, profile["parts"], read, include_beta=False,
                              delta_broadcast=execution.delta_broadcast).scaled(heads)
            work += beta_work(p, q, read).scaled(beta_count)
            count += 1
        cycles.append(count)
    cycles = tuple(cycles)
    def program(context):
        def owner_program(owner):
            placement = context.owned(owner, units, ("vector", read, "mte3"))
            pending = tuple(runs(owner))
            def head(index):
                q, count, beta_count = pending[index]
                return _head_stages(p, q, count, profile, context, placement, read,
                                    beta_repetitions=beta_count,
                                    delta_broadcast=execution.delta_broadcast)
            heads = PipelineProgram(cycles[owner], 1, head)
            allocation = (Allocation("kda_buffers", context.pool_names["ub"],
                                     profile["ub_bytes_per_owner"],
                                     unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("heads", after=("buffers",), program=heads),
                                  Stage("release", after=("heads",), free=allocation)))
        return ParallelProgram(tuple(owner_program(owner) for owner in range(blocks)))
    scalar_bytes = 4 if p.metadata_dtype == "int32" else 8
    validated_sequences = blocks*len(p.lengths) if execution.validate_all_sequences else len(p.lengths)*p.heads
    resolved_sequences = (len(p.lengths)*min(p.heads, blocks)
                          if execution.validate_all_sequences else len(p.lengths)*p.heads)
    metadata_values = blocks*(len(p.lengths)+1) + 2*validated_sequences
    if p.has_state_indices:
        metadata_values += (blocks if execution.validate_all_sequences else p.heads)*sum(p.lengths)
        metadata_values += p.heads*sum(p.lengths)*profile["parts"] + resolved_sequences
    accepted_loads = validated_sequences + resolved_sequences if p.has_accepted_tokens else 0
    metadata_values += accepted_loads
    return OperatorPlan(work, program, metadata={
        "execution_class": "vector",
        "kda_recurrent_schedule": {
            **profile, "logical_heads": len(p.lengths)*p.heads,
            "active_aiv": blocks, "physical_aiv": units, "owner_cycles": cycles,
            "resource_cycles": sum(cycles), "compact": compact,
            "compaction_domain": ("complete_owner_waves_with_explicit_partial_tails"
                                  if execution.beta_per_head or p.heads <= blocks
                                  else "distinct_beta_load_reuse_cycles_remain_detailed"),
            "beta_load_scope": "head" if execution.beta_per_head else "owner_sequence",
            "delta_update": "UB_broadcast_MulAddDst_query_Mul" if execution.delta_broadcast else "fused_MicroAPI_ProcessKQ",
            "ownership": "task_id=(sequence*HV+head);owner=task_id%launched_aiv",
            "state_residency": "one_initial_load_per_head;checkpoint_write_every_query",
            "head_handoff": "conservative_serial_head_retirement",
            "contention_approximation": "serial_cycle_fluid_rephasing" if compact else "none",
            "scalar_metadata_getvalue_bytes": scalar_bytes*metadata_values,
            "accepted_metadata_getvalue_bytes": scalar_bytes*accepted_loads,
            "unmodeled_services": ("scalar_GetValue_cache_and_issue", "queue_flag_and_vector_instruction_issue"),
        },
    })


def make_op(params, *, op_id="kda_recurrent", deps=(), requests=(), stage=0,
            layer=0, phase="decode", stream="", notes=()):
    p = Params.parse(params)
    shape = (sum(p.lengths), p.heads, p.dk, p.dv)
    metadata = {
        "kda_recurrent_params": dict(p.__dict__), "native_kind": "RecurrentKda",
        "implementation": IMPLEMENTATION, "source": SOURCE,
        "invocation_class": "vector", "program_kind": "recurrent_vector",
        "heads": p.heads, "key_heads": p.key_heads, "dk": p.dk, "dv": p.dv,
        "sequence_tokens": p.lengths, "state_dtype": p.state_dtype,
        "state_mutation": "inplace_per_query_checkpoint",
        "source_binding": "explicit_arch35_kernel_selection_not_binary_equivalence",
        "caller_materializations": "outside_selected_kernel",
    }
    selection = resolve_state_selection(p.state_selection, p.lengths)
    if selection is not None:
        metadata["state_selection"] = selection
        deps = tuple(dict.fromkeys((*deps, *selection["dependencies"])))
    return OpSpec(op_id, "recurrent_scan", phase, stage, layer, tuple(requests),
                  shape, shape, shape, shape, deps=tuple(deps), dtype="bf16", layout="TND",
                  recipe=RECIPE, source_boundary="kernel", coverage="source_selected_recurrent_kernel",
                  metadata=tuple(sorted(metadata.items())), stream=stream, notes=tuple(notes))
