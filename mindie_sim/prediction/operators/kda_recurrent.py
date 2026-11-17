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
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.prediction.recover_ssm_work import record_head_bytes, record_slice_bytes
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec, ProgramFragment, StateEffect
from mindie_sim.schema.subgraphs import PortSet
from mindie_sim.schema.tensors import TensorValue
from .kda_state_selection import (
    bind_sequence_values, bind_state_value, binding_record, binding_value,
    decision_ports, resolve as resolve_state_selection, state_ports,
    state_records, state_value,
)

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
_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
ZERO_WORK = 0.0
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

    @classmethod
    def parse(cls, raw):
        p = cls(**{**raw, "lengths": tuple(raw["lengths"])})
        if not isinstance(p.recover_ssm, bool):
            raise ModelingError("recurrent_kda_recover_ssm_invalid")
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
    state_out = record_head_bytes(k, v, p.dtype) if (p.recover_ssm and q > 1) else state
    outgoing = q*(state_out + v*2)
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
    work = summarize_work(p, geometry, execution=execution)
    cycles = []
    for owner in range(blocks):
        count = sum(1 for _ in runs(owner))
        cycles.append(count)
    cycles = tuple(cycles)
    fragments = state_fragments(op)
    def program(context):
        def owner_program(owner):
            placement = context.owned(owner, units, ("vector", read, "mte3"))
            pending = tuple(runs(owner))
            def head(index):
                q, count, beta_count = pending[index]
                return _head_stages(p, q, count, profile, context, placement, read,
                                    beta_repetitions=beta_count,
                                    delta_broadcast=execution.delta_broadcast)
            heads = PipelineProgram(cycles[owner], 1, head, interior_fold=compact)
            allocation = (Allocation("kda_buffers", context.pool_names["ub"],
                                     profile["ub_bytes_per_owner"],
                                     unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("heads", after=("buffers",), program=heads),
                                  Stage("release", after=("heads",), free=allocation)))
        owners = ParallelProgram(tuple(owner_program(owner) for owner in range(blocks)))
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
            "contention_approximation": "serial_cycle_fluid_rephasing" if compact else "none",
            "scalar_metadata_getvalue_bytes": scalar_bytes*metadata_values,
            "accepted_metadata_getvalue_bytes": scalar_bytes*accepted_loads,
            "unmodeled_services": ("scalar_GetValue_cache_and_issue", "queue_flag_and_vector_instruction_issue"),
        }
    return OperatorPlan(work, program, metadata=plan_metadata)


def _effect(target, action, version, arbiter=""):
    return StateEffect(target, action, version, "", arbiter)


def _state_effect(target, action, version, arbiter=""):
    return _effect(target, action, version, arbiter).to_dict()


def _slot_storage(storage, slot):
    base = str(storage)
    marker = ":slot:"
    if marker in base:
        base = base.rsplit(marker, 1)[0]
    return base if slot is None else f"{base}{marker}{slot}"


def state_contract(*, lengths, targets=(), requests=(), selection=None,
                   contexts=None, acceptance=None, bindings=None,
                   owner="self", state_dtype=None, decisions=(),
                   ready=(), grants=(), op_id=None):
    """Bind per-query versions to explicit state write/commit records.

    ``bindings`` are the caller's actual ``TensorValue`` records.  The contract
    keeps the historical selection, the current invocation geometry and the
    current acceptance decision separate, then projects the same binding into
    reads, candidate writes, decisions and exports.  ``selection`` supplies the
    physical initial/checkpoint slots; an accepted checkpoint overwritten by a
    later same-sequence slot is rejected unless a real materialization plan
    exists.
    """
    if bindings is None:
        raise ModelingError("kda_state_contract_requires_value_bindings")
    targets = tuple(targets)
    bindings = tuple(bindings)
    if len(bindings) != len(tuple(lengths)) or any(
            binding["target"] != target
            for binding, target in zip(bindings, targets)):
        raise ModelingError("kda_state_contract_bindings_must_match_targets")
    checkpoint_slots = None if selection is None else selection.get("checkpoint_slots")
    records = state_records(
        targets=targets, lengths=lengths, contexts=contexts,
        acceptance=acceptance, checkpoint_slots=checkpoint_slots)
    effects = []
    for binding in bindings:
        effects.append(_state_effect(
            binding["target"], "read", binding["state_version"]))
    write_bindings = []
    by_target = {binding["target"]: binding for binding in bindings}
    for target, version, slot in records["write_records"]:
        source = by_target[target]
        value = binding_value(source)
        layout = value.layout
        # A candidate overwrites this sequence's slot with a new generation;
        # the leading capacity axis is already narrowed for a pool tensor.
        write_value = replace(
            value, producer=str(op_id or target),
            output=f"checkpoint:{version}", storage=_slot_storage(
                source["storage"], slot),
            state_version=version, binding="static", binding_name="")
        binding = bind_state_value(
            write_value, target=target, owner=source.get("owner", owner),
            slot=slot, version=version,
            ready=source.get("ready"), grant=source.get("grant"),
            generation=version, decision=("token_commit" if version in
                                          records["commit_versions"] else None),
            last_use=("next_decode_read" if version in
                      records["commit_versions"] else "last_reader"),
            representation=source.get("representation"),
            required=True, span=((0, layout.shape[-2]), (0, layout.shape[-1])))
        write_bindings.append(binding)
        effects.append(_state_effect(target, "write", version))
    for target, version in records["commit_records"]:
        effects.append(_state_effect(target, "commit", version,
                                     "acceptance_trace"))
    for target, version in records["rollback_records"]:
        effects.append(_state_effect(target, "rollback", version,
                                     "acceptance_trace"))
    acceptance_view = {
        "accepted_offsets": records["accepted_offsets"], "advances": records["advances"],
        "commit_versions": records["commit_versions"], "rejected_versions": records["rejected_versions"],
        "pending": records["pending_acceptance"], "commit_policy": records["commit_policy"],
        "candidate_write_point": records["candidate_write_point"],
    }
    history_selection = {
        "basis": "exact_selection" if selection is not None else "caller_context_checkpoint",
        "address_basis": None if selection is None else selection.get("address_basis"),
        "initial_slots": None if selection is None else selection.get("initial_slots"),
        "checkpoint_slots": checkpoint_slots, "accepted_offsets": records["accepted_offsets"],
        "retained_versions": records["retained_versions"],
        "overwritten_versions": records["overwritten_versions"],
    }
    current_geometry = {
        "queries": records["queries"], "contexts": records["contexts"],
        "version_basis": records["version_basis"], "targets": targets,
        "request_ids": tuple(requests), "state_layout": "V,K", "state_dtype": state_dtype,
    }
    contract = {
        **records, "initial_slots": None if selection is None else selection.get("initial_slots"),
        "checkpoint_slots": checkpoint_slots,
        "address_basis": None if selection is None else selection.get("address_basis"),
        "state_dtype": state_dtype, "state_layout": "V,K", "request_ids": tuple(requests),
        "owner": owner, "read_bindings": tuple(binding_record(b) for b in bindings),
        "write_bindings": tuple(binding_record(b) for b in write_bindings),
        "history_selection": history_selection, "current_geometry": current_geometry,
        "acceptance": acceptance_view, "decisions": tuple(decisions),
        "ready": tuple(ready), "grants": tuple(grants),
    }
    return contract, tuple(effects)
def _fragment(name, effects, ports=None):
    return ProgramFragment(name, state_effects=tuple(effects),
                           ports=PortSet() if ports is None else ports)


def _binding_attrs(bindings):
    from mindie_sim.schema.operators import SemanticAttr
    if not bindings:
        return ()
    return (SemanticAttr(
        "state_bindings",
        tuple(binding_record(binding) for binding in bindings)),)


def _decision_name(contract, target, version):
    decisions = tuple(contract.get("decisions", ()))
    targets = tuple(contract.get("targets", ()))
    if isinstance(decisions, Mapping):
        return decisions.get(target, decisions.get((target, version)))
    if target in targets and len(decisions) == len(targets):
        value = decisions[targets.index(target)]
        if isinstance(value, Mapping):
            return value.get(version, value.get(str(version)))
        return value
    return None


def state_fragments(op):
    """Return executable read/write/commit/rollback fragments.

    Ports and their attrs are projected from the same value bindings used by
    ``state_accesses``.  Commit and rollback fragments carry the actual
    decision event name; they are submitted at the caller's token-commit and
    acceptance events, not folded into one unconditional batch.
    """
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is None:
        return {"read": None, "write": None, "commits": (), "rollbacks": ()}
    read_bindings = tuple(contract.get("read_bindings", ()))
    write_bindings = tuple(contract.get("write_bindings", ()))
    read = _fragment(
        f"{op.op_id}:state_read",
        (_effect(binding["target"], "read", binding["state_version"])
         for binding in read_bindings),
        state_ports(bindings=read_bindings, owner=f"kda.layer[{op.layer}]",
                    direction="in", readiness="previous_iteration",
                    optional=False))
    read = replace(read, attrs=_binding_attrs(read_bindings))
    write = _fragment(
        f"{op.op_id}:state_write",
        (_effect(binding["target"], "write", binding["state_version"])
         for binding in write_bindings),
        state_ports(bindings=write_bindings, owner=f"kda.layer[{op.layer}]",
                    direction="out", readiness="producer_complete",
                    optional=False))
    write = replace(write, attrs=_binding_attrs(write_bindings))
    commits = []
    for target, version in contract["commit_records"]:
        event = _decision_name(contract, target, version)
        fragment = _fragment(
            f"{op.op_id}:state_commit:{target}:{version}",
            (_effect(target, "commit", version, "acceptance_trace"),),
            decision_ports(target=target, version=version,
                           owner=f"kda.layer[{op.layer}]"))
        from mindie_sim.schema.operators import SemanticAttr
        commits.append(replace(fragment, attrs=(SemanticAttr(
            "decision", {"action": "commit", "target": target,
                         "version": version, "event": event or ""}),)))
    rollbacks = []
    for target, version in contract["rollback_records"]:
        event = _decision_name(contract, target, version)
        fragment = _fragment(
            f"{op.op_id}:state_rollback:{target}:{version}",
            (_effect(target, "rollback", version, "acceptance_trace"),),
            decision_ports(target=target, version=version,
                           owner=f"kda.layer[{op.layer}]", action="rollback"))
        from mindie_sim.schema.operators import SemanticAttr
        rollbacks.append(replace(fragment, attrs=(SemanticAttr(
            "decision", {"action": "rollback", "target": target,
                         "version": version, "event": event or ""}),)))
    return {"read": read, "write": write,
            "commits": tuple(commits), "rollbacks": tuple(rollbacks)}


def _binding_span(binding, *, head=None):
    shape = tuple(binding["value"]["layout"]["shape"]
                  if isinstance(binding.get("value"), Mapping)
                  else binding["layout"]["shape"])
    slot = binding.get("slot")
    if len(shape) == 4 and shape[0] == 1:
        first = (slot, slot + 1) if slot is not None else (0, 1)
        rest = tuple((0, size) for size in shape[1:])
        return (first, *rest)
    return tuple((0, size) for size in shape)


def state_accesses(op):
    """Project KDA decode state reads/writes into SERVICE access records.

    A physical checkpoint covers every value head in its actual slot.  Each
    head remains a distinct sub-access (or would be one 262144 B access for the
    four-head control); the slot, layout offset, generation and decision event
    come from the same binding as the effects and ports.
    """
    contract = dict(op.metadata).get("kda_state_contract")
    params = dict(op.metadata).get("kda_recurrent_params", {})
    if contract is None:
        return ()
    dk, dv = int(params.get("dk", 128)), int(params.get("dv", 128))
    state_byte = {"fp32": 4, "bf16": 2}.get(str(params.get("state_dtype", "fp32")), 4)
    use_recover = bool(params.get("recover_ssm", False))
    rec_head = record_head_bytes(dk, dv, str(params.get("dtype", "bf16")))
    request_ids = tuple(contract.get("request_ids", ()))
    result = []
    for index, binding in enumerate(tuple(contract.get("read_bindings", ()))) :
        target = binding["target"]
        request = request_ids[index] if index < len(request_ids) else str(index)
        span = _binding_span(binding)
        elements = math.prod(end - start for start, end in span)
        result.append({
            "occurrence": f"{op.op_id}:read:{index}", "producer": binding["producer"],
            "request": request, "stage": op.stage, "layer": op.layer,
            "direction": "read", "position": "state", "output": target,
            "storage": binding["storage"], "owner": binding.get("owner", f"kda.layer[{op.layer}]"),
            "representation": binding.get("representation", "state"),
            "persistent": True, "temporary": False, "state_version": binding["state_version"],
            "span": span, "bytes": float(elements * state_byte),
            "source": binding.get("ready") or binding["producer"], "reader": "",
            "grant": binding.get("grant") or "", "decision": None, "last_use": "next_decode_read",
            "rank": None, "scope": "target", "generation": binding.get("generation"),
            "slot": binding.get("slot"), "window": binding.get("window"),
            "offset": binding.get("offset"), "physical_key": binding["storage"],
            "request_slot": binding.get("slot"), "rows": 1, "row_stride": dk * dv,
            "tier": binding.get("tier", "hbm"), "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"), "last_use_event": "next_decode_read",
        })
    committed = set(contract.get("commit_versions", ()))
    targets = tuple(contract.get("targets", ()))
    for write_index, binding in enumerate(tuple(contract.get("write_bindings", ()))) :
        target = binding["target"]
        sequence = targets.index(target) if target in targets else 0
        request = request_ids[sequence] if sequence < len(request_ids) else str(sequence)
        version, shape = binding["state_version"], tuple(binding["layout"]["shape"])
        heads, head_axis = (shape[1], 1) if (len(shape) == 4 and shape[0] == 1) else (shape[0], 0)
        slot, commit = binding.get("slot"), version in committed
        decision_event = _decision_name(contract, target, version)
        base_span = _binding_span(binding)
        is_spec = use_recover and int(binding.get("window") or 1) > 1
        out_bytes = float(rec_head if is_spec else dk * dv * state_byte)
        out_repr = "recover_ssm_record" if is_spec else binding.get("representation", "state")
        for head in range(heads):
            span = list(base_span)
            span[head_axis] = (head, head + 1)
            result.append({
                "occurrence": f"{op.op_id}:write:{write_index}:{head}",
                "producer": binding["producer"], "request": request,
                "stage": op.stage, "layer": op.layer, "direction": "write", "position": "state",
                "output": target, "storage": binding["storage"],
                "owner": binding.get("owner", f"kda.layer[{op.layer}]"),
                "representation": out_repr, "persistent": True, "temporary": False,
                "state_version": version, "span": tuple(span), "bytes": out_bytes,
                "source": binding["producer"], "reader": "", "grant": binding.get("grant") or "",
                "decision": "token_commit" if commit else "candidate_hold",
                "last_use": "next_decode_read" if commit else "last_reader",
                "rank": None, "scope": "target", "generation": version, "slot": slot,
                "window": binding.get("window"), "offset": binding.get("offset"),
                "physical_key": f"{binding['storage']}:head:{head}", "request_slot": slot,
                "rows": 1, "row_stride": dk * dv, "tier": binding.get("tier", "hbm"),
                "source_ready": binding.get("ready"), "grant_ready": binding.get("grant"),
                "decision_event": decision_event,
                "last_use_event": "next_decode_read" if commit else "last_reader",
            })
    return tuple(result)
def _value_versions(values, targets, count):
    if values is None:
        return None
    if isinstance(values, TensorValue):
        return (values.state_version,) * count
    if isinstance(values, Mapping):
        return tuple((values.get(target) or values.get(index)).state_version
                     if isinstance(values.get(target, values.get(index)), TensorValue)
                     else None
                     for index, target in enumerate(targets))
    values = tuple(values)
    if len(values) == count:
        return tuple(value.state_version
                     if isinstance(value, TensorValue) else None
                     for value in values)
    return None


def make_op(params, *, op_id="kda_recurrent", deps=(), requests=(), stage=0,
            layer=0, phase="decode", stream="", notes=(), state_contexts=None,
            state_targets=None, state_values=None, state_slots=None,
            state_ready=(), state_grants=(), state_decisions=(),
            state_generation=None):
    p = Params.parse(params)
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
        state_dtype=p.state_dtype, decisions=state_decisions,
        ready=state_ready, grants=state_grants, op_id=op_id)
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
