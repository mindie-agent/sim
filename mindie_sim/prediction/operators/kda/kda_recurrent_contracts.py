"""Versioned KDA decode state contracts, fragments and access records.

[Role] Project state checkpoints and accepted versions from real tensor views.
[Invariants] Reads, candidate writes and exports use the same slot and layout
  bindings. Only the accepted prefix publishes a committed version.
[Timeouts & Locks] Metadata work scales with the supplied requests and writes.
[Side-Effect Policy] Pure contract and access records.
"""
from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Literal

from mindie_sim.prediction.operators.recurrent.recover_ssm_work import record_head_bytes
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import ProgramFragment, StateEffect
from mindie_sim.schema.subgraphs import PortSet
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .kda_state_selection import (
    SOURCES as _SELECTION_SOURCES,
    _narrow_value, bind_state_value, binding_record, binding_value,
    decision_ports, state_ports, state_records,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = _SELECTION_SOURCES

STATE_POOL_DIMS = 4
_STATE_TAIL_DIMS = 3
_DEFAULT_HEAD_DIM = 128
_STATE_DTYPE_BYTES = {"fp32": 4, "bf16": 2}


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


def _write_slot_value(state_values, source, value, slot):
    if slot is None:
        return value
    if (isinstance(state_values, TensorValue)
            and len(state_values.layout.shape) == STATE_POOL_DIMS
            and tuple(state_values.layout.shape[1:]) == tuple(value.layout.shape[-_STATE_TAIL_DIMS:])):
        return _narrow_value(state_values, 0, slot, 1)
    read_slot = source.get("slot")
    if (read_slot is not None and slot != read_slot
            and len(value.layout.shape) == STATE_POOL_DIMS
            and value.layout.shape[0] == 1):
        slot_stride = value.layout.strides[0]
        base_offset = value.layout.offset - int(read_slot) * slot_stride
        if base_offset < 0:
            raise ModelingError("kda_state_value_window_outside_layout")
        return replace(
            value,
            layout=replace(value.layout, offset=base_offset + int(slot) * slot_stride),
        )
    return value


def _recover_ssm_record_value(
    *, producer, source, value, version, token_index, record_dtype,
    batch_index: int = 0, record_stride: int = 1,
):
    shape = tuple(value.layout.shape)
    heads = shape[1] if len(shape) == STATE_POOL_DIMS and shape[0] == 1 else shape[0]
    dv, dk = int(shape[-2]), int(shape[-1])
    rec_head = record_head_bytes(dk, dv, str(record_dtype))
    elem_bytes = int(dtype_bytes(str(record_dtype)))
    rec_elems = rec_head // elem_bytes
    stride_tokens = max(1, int(record_stride))
    layout = TensorLayout(
        shape=(heads, rec_elems),
        strides=(rec_elems, 1),
        offset=(int(batch_index) * stride_tokens + int(token_index)) * heads * rec_elems,
    )
    storage = f"{_slot_storage(source['storage'], None)}:recover_ssm_record"
    return TensorValue(
        producer=str(producer), dtype=str(record_dtype), layout=layout,
        output=f"recover_ssm_record:{version}", storage=storage,
        binding="static", binding_name="", state_version=version,
    ), ((0, heads), (0, rec_elems))


def state_contract(*, lengths, targets=(), requests=(), selection=None,
                   contexts=None, acceptance=None, bindings=None,
                   owner="self", state_dtype=None, record_dtype="bf16",
                   decisions=(), ready=(), grants=(), op_id=None,
                   recover_ssm=False, state_values=None,
                   record_stride: int | None = None):
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
    lengths_tuple = tuple(lengths)
    if len(bindings) != len(lengths_tuple) or any(
            binding["target"] != target
            for binding, target in zip(bindings, targets)):
        raise ModelingError("kda_state_contract_bindings_must_match_targets")
    checkpoint_slots = None if selection is None else selection.get("checkpoint_slots")
    if checkpoint_slots is None and bindings and all(
        binding.get("slot") is not None and int(q) == 1
        for binding, q in zip(bindings, lengths_tuple)
    ):
        checkpoint_slots = tuple((binding["slot"],) for binding in bindings)
    records = state_records(
        targets=targets, lengths=lengths_tuple, contexts=contexts,
        acceptance=acceptance, checkpoint_slots=checkpoint_slots)
    effects = []
    for binding in bindings:
        effects.append(_state_effect(
            binding["target"], "read", binding["state_version"]))
    write_bindings = []
    by_target = {binding["target"]: binding for binding in bindings}
    target_index = {target: index for index, target in enumerate(targets)}
    committed_pairs = set(records["commit_records"])
    query_by_target = dict(zip(targets, records["queries"]))
    max_query = max((int(q) for q in records["queries"]), default=1)
    effective_record_stride = (
        max(max_query, int(record_stride)) if record_stride is not None else max_query
    )
    token_pos_by_target: dict[str, int] = {}
    for target, version, slot in records["write_records"]:
        source = by_target[target]
        value = binding_value(source)
        q_len = int(query_by_target[target])
        is_spec = bool(recover_ssm and q_len > 1)
        token_index = token_pos_by_target.get(target, 0)
        token_pos_by_target[target] = token_index + 1
        if is_spec:
            write_value, span = _recover_ssm_record_value(
                producer=str(op_id or target), source=source, value=value,
                version=version, token_index=token_index,
                record_dtype=record_dtype,
                batch_index=target_index[target],
                record_stride=effective_record_stride,
            )
            binding = bind_state_value(
                write_value, target=target, owner=source.get("owner", owner),
                slot=None, version=version,
                ready=source.get("ready"), grant=source.get("grant"),
                generation=version, decision="recover_ssm_record",
                last_use="recover_ssm_commit",
                representation="recover_ssm_record",
                window=(0, q_len), required=True, span=span,
            )
        else:
            write_base = _write_slot_value(state_values, source, value, slot)
            layout = write_base.layout
            is_commit = (target, version) in committed_pairs
            # A candidate overwrites this sequence's slot with a new generation;
            # the leading capacity axis is sliced by the target write slot.
            write_value = replace(
                write_base, producer=str(op_id or target),
                output=f"checkpoint:{version}", storage=_slot_storage(
                    source["storage"], slot),
                state_version=version, binding="static", binding_name="")
            binding = bind_state_value(
                write_value, target=target, owner=source.get("owner", owner),
                slot=slot, version=version,
                ready=source.get("ready"), grant=source.get("grant"),
                generation=version, decision=("token_commit" if is_commit else None),
                last_use=("next_decode_read" if is_commit else "last_reader"),
                representation=source.get("representation"),
                window=(0, q_len),
                required=True, span=((0, layout.shape[-2]), (0, layout.shape[-1])))
        write_bindings.append(binding)
        effects.append(_state_effect(target, "write", version))
    overwritten_records = tuple(
        (target, version)
        for target, q_len, versions in zip(
            targets, records["queries"], records["overwritten_versions"]
        )
        if not (recover_ssm and int(q_len) > 1)
        for version in versions
    )
    for target, version in overwritten_records:
        effects.append(_state_effect(target, "rollback", version, "slot_overwrite"))
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
        "overwritten_records": overwritten_records,
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
    write_effects = [
        _effect(binding["target"], "write", binding["state_version"])
        for binding in write_bindings
    ]
    for target, version in contract.get("overwritten_records", ()):
        write_effects.append(_effect(target, "rollback", version, "slot_overwrite"))
    write = _fragment(
        f"{op.op_id}:state_write",
        write_effects,
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


def _window_width(window, fallback: int) -> int:
    if isinstance(window, (tuple, list)) and len(window) == 2:
        return int(window[1]) - int(window[0])
    if isinstance(window, int) and not isinstance(window, bool):
        return window
    return int(fallback)


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
    dk, dv = int(params.get("dk", _DEFAULT_HEAD_DIM)), int(params.get("dv", _DEFAULT_HEAD_DIM))
    state_byte = _STATE_DTYPE_BYTES.get(
        str(params.get("state_dtype", "fp32")), _STATE_DTYPE_BYTES["fp32"])
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
    targets = tuple(contract.get("targets", ()))
    queries = tuple(contract.get("queries", ()))
    if "commit_records" in contract:
        committed = {(t, v) for t, v in contract["commit_records"]}
    else:
        committed = {
            (t, v) for t, v in zip(targets, contract.get("commit_versions", ()))
            if v is not None
        }
    for write_index, binding in enumerate(tuple(contract.get("write_bindings", ()))) :
        target = binding["target"]
        sequence = targets.index(target) if target in targets else 0
        request = request_ids[sequence] if sequence < len(request_ids) else str(sequence)
        version, shape = binding["state_version"], tuple(binding["layout"]["shape"])
        heads = shape[1] if len(shape) == STATE_POOL_DIMS and shape[0] == 1 else shape[0]
        slot, commit = binding.get("slot"), (target, version) in committed
        decision_event = _decision_name(contract, target, version)
        base_span = _binding_span(binding)
        fallback_q = queries[sequence] if sequence < len(queries) else 1
        is_spec = use_recover and _window_width(binding.get("window"), fallback_q) > 1
        out_bytes = float(rec_head if is_spec else dk * dv * state_byte)
        out_repr = "recover_ssm_record" if is_spec else binding.get("representation", "state")
        out_decision = (
            "recover_ssm_record" if is_spec
            else ("token_commit" if commit else "candidate_hold")
        )
        out_last_use = (
            "recover_ssm_commit" if is_spec
            else ("next_decode_read" if commit else "last_reader")
        )
        result.append({
            "occurrence": f"{op.op_id}:write:{write_index}",
            "producer": binding["producer"], "request": request,
            "stage": op.stage, "layer": op.layer, "direction": "write", "position": "state",
            "output": target, "storage": binding["storage"],
            "owner": binding.get("owner", f"kda.layer[{op.layer}]"),
            "representation": out_repr, "persistent": True, "temporary": False,
            "state_version": version, "span": base_span, "bytes": float(heads * out_bytes),
            "source": binding["producer"], "reader": "", "grant": binding.get("grant") or "",
            "decision": out_decision,
            "last_use": out_last_use,
            "rank": None, "scope": "target", "generation": version, "slot": slot,
            "window": binding.get("window"), "offset": binding.get("offset"),
            "physical_key": binding["storage"], "request_slot": slot,
            "rows": 1, "row_stride": dk * dv, "tier": binding.get("tier", "hbm"),
            "source_ready": binding.get("ready"), "grant_ready": binding.get("grant"),
            "decision_event": decision_event,
            "last_use_event": out_last_use,
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
