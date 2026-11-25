"""State bindings, history windows and access records for causal convolution.

[Role] Resolve state ownership and project native update semantics to the engine.
[Invariants] Padding owns no history. Reads and writes retain their exact row
  windows, and only the accepted prefix commits.
[Timeouts & Locks] Local descriptor and sequence traversal; no I/O or locks.
[Side-Effect Policy] Returns state contracts, semantic fragments and access records.
"""
from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import ProgramFragment, SemanticAttr, StateEffect
from mindie_sim.schema.subgraphs import PortSet
from mindie_sim.schema.tensors import TensorLayout, TensorValue

from mindie_sim.prediction.operators.kda.kda_state_selection import (
    bind_state_value, binding_record, binding_value, decision_ports,
    state_ports, state_records, state_value,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/ops/kimi_kda.py",
        "af37687463f9b34ca4797bdcefb14c333d3b70b2956d6aa1f15217815d5fbc10",
    ),
)

_STATE_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def _per_sequence(value, count, default=None):
    if value is None or value == () or value == []:
        return (default,) * count
    if isinstance(value, Mapping):
        return tuple(value.get(index, default) for index in range(count))
    value = tuple(value)
    if len(value) == 1 and count > 1:
        return value * count
    if len(value) != count:
        raise ModelingError("causal_conv_update_values_must_match_sequences")
    return value


def _slot_storage(storage, slot):
    base = str(storage)
    marker = ":slot:"
    if marker in base:
        base = base.rsplit(marker, 1)[0]
    return base if slot is None else f"{base}{marker}{slot}"


def _narrow_window(value, offset, rows):
    shape = tuple(value.layout.shape)
    if len(shape) == 2:
        return value.narrow(0, offset, rows)
    if len(shape) == 3 and shape[0] == 1:
        return value.narrow(1, offset, rows)
    raise ModelingError(
        f"causal_conv_state_requires_history_axis:{shape}")


def _conv_bindings(*, views, active, targets, contexts, history_slots, width,
                   dtype, values, slots, owner, ready, grants):
    count = len(views)
    slots = _per_sequence(slots, count)
    ready = _per_sequence(ready, count)
    grants = _per_sequence(grants, count)
    shape = (history_slots, width)
    active_set = set(active)
    if values is None:
        produced = [None] * count
    elif isinstance(values, TensorValue):
        layout = tuple(values.layout.shape)
        if layout == shape and count == 1:
            produced = [values]
        elif len(layout) == 3 and layout[1:] == shape:
            produced = []
            for index in range(count):
                if index not in active_set:
                    produced.append(None)
                    continue
                slot = slots[index]
                if slot is None and layout[0] == count:
                    slot = index
                if slot is None or slot < 0 or slot >= layout[0]:
                    raise ModelingError(
                        f"causal_conv_state_slot_out_of_bounds:{slot}:{layout[0]}")
                produced.append(values.narrow(0, slot, 1))
        else:
            raise ModelingError(
                f"causal_conv_state_pool_shape_mismatch:{layout}:{shape}")
    elif isinstance(values, Mapping):
        produced = [values.get(target, values.get(index))
                    for index, target in enumerate(targets)]
    elif isinstance(values, (tuple, list)):
        produced = list(values)
        if len(produced) != count:
            raise ModelingError(
                "causal_conv_update_values_must_match_sequences")
    else:
        raise ModelingError(
            "causal_conv_update_values_require_tensor_or_sequence")
    bindings = []
    for index, view in enumerate(views):
        if index not in set(active):
            bindings.append(None)
            continue
        target, context, slot = targets[index], contexts[index], slots[index]
        provided = produced[index]
        if provided is None:
            storage = _slot_storage(f"{target}.state", slot)
            provided = TensorValue(
                "external", dtype, TensorLayout.contiguous(shape),
                output=(slot if slot is not None else index),
                storage=storage, binding="external",
                binding_name=ready[index] or f"kda.conv_state[{target}]",
                state_version=context)
        else:
            provided = state_value(provided)
            if tuple(provided.layout.shape[-2:]) != shape:
                raise ModelingError(
                    f"causal_conv_state_value_shape_mismatch:"
                    f"{provided.layout.shape}:{shape}")
            if provided.dtype != dtype:
                raise ModelingError(
                    f"causal_conv_state_value_dtype_mismatch:"
                    f"{provided.dtype}:{dtype}")
        bindings.append(bind_state_value(
            provided, target=target, owner=owner, slot=slot,
            version=context, ready=ready[index], grant=grants[index],
            generation=context, required=True,
            representation="history", span=((0, history_slots), (0, width)),
            readiness=("previous_iteration" if context else "external")))
    return tuple(bindings)


def _select_history_bindings(*, bindings, views, active_slots, contexts, kernel_width,
                             history_acceptance, reader_holds, rebase_outputs,
                             q_lens, speculative, slots):
    if bindings is None:
        raise ModelingError("causal_conv_update_requires_value_bindings")
    bindings = tuple(bindings)
    if len(bindings) != len(views):
        raise ModelingError("causal_conv_update_bindings_must_match_views")
    rebase_outputs = dict(rebase_outputs or {})
    read_bindings = []
    history_selection = []
    for position, index in enumerate(active_slots):
        binding = bindings[index]
        if binding is None:
            raise ModelingError("causal_conv_update_active_binding_missing")
        accepted = (0 if history_acceptance is None
                    else int(history_acceptance[index]))
        source_offset = max(0, accepted - 1)
        history_window = (source_offset, source_offset + kernel_width - 1)
        if index in rebase_outputs:
            binding = rebase_outputs[index]
        read_bindings.append(binding)
        history_selection.append({
            "sequence": index,
            "request": str(getattr(views[index], "request_id", index)),
            "accepted_tokens": accepted,
            "source_offset": source_offset,
            "source_context": contexts[position],
            "history_window": history_window,
            "current_window": ((0, 2 + int(q_lens[index]))
                               if speculative else (0, int(q_lens[index]))),
            "slot": (None if slots is None else slots[index]),
            "rebase": index in rebase_outputs,
            "reader_hold": (None if reader_holds is None
                            else reader_holds[index]),
        })
    return read_bindings, history_selection


def _candidate_bindings(*, read_bindings, records, active_targets, active_slots,
                        slots, layer_target, q_lens, speculative, kernel_width, width):
    write_bindings = []
    by_target = {binding["target"]: binding for binding in read_bindings}
    committed_pairs = set(records["commit_records"])
    for target, version, record_slot in records["write_records"]:
        source = by_target[target]
        sequence = active_targets.index(target)
        view_index = active_slots[sequence]
        slot = (record_slot if slots is None else slots[view_index])
        value = binding_value(source)
        write_value = replace(
            value, producer=f"{layer_target}:conv_candidate",
            output=f"checkpoint:{version}",
            storage=_slot_storage(source["storage"], slot),
            state_version=version, binding="static", binding_name="")
        write_rows = (2 + int(q_lens[view_index]) if speculative
                      else kernel_width - 1)
        is_committed = (target, version) in committed_pairs
        write_bindings.append(bind_state_value(
            write_value, target=target, owner=source.get("owner", layer_target),
            slot=slot, version=version, ready=source.get("ready"),
            grant=source.get("grant"), generation=version,
            decision=("token_commit" if is_committed else None),
            last_use=("next_conv_read" if is_committed else "last_reader"),
            representation="history", required=True,
            span=((0, write_rows), (0, width))))
    return write_bindings


def _state_effects(records):
    effects = []
    for target, version in records["read_records"]:
        effects.append(StateEffect(target, "read", version).to_dict())
    for target, version, _slot in records["write_records"]:
        effects.append(StateEffect(target, "write", version).to_dict())
    for target, version in records["commit_records"]:
        effects.append(StateEffect(
            target, "commit", version, "", "acceptance_trace").to_dict())
    for target, version in records["rollback_records"]:
        effects.append(StateEffect(
            target, "rollback", version, "", "acceptance_trace").to_dict())
    return effects


def _state_contract(*, views, q_lens, padding_slots, width, byte,
                    kernel_width, history_slots, speculative, layer,
                    bindings=None, history_acceptance=None, reader_holds=None,
                    slots=None, targets=None, rebase_outputs=None,
                    state_decisions=()):
    """Declare the actual read/write versions for this sequence partition.

    Historical selection (which committed window is copied), this call's
    geometry (query rows, slots, padding) and the current acceptance decision
    stay separate fields.  Every access/effect/port is projected from the same
    caller value binding; a rebase replaces the read producer with the copy's
    actual output, not a second name for the old state.
    """
    active_slots = tuple(index for index in range(len(views))
                         if index not in set(padding_slots))
    layer_target = f"kda.layer[{layer}]" if layer >= 0 else "kda.call"
    if targets is None:
        targets = [None] * len(views)
        for index in active_slots:
            request = str(getattr(views[index], "request_id", index))
            if not request:
                raise ModelingError("causal_conv_update_request_id_invalid")
            targets[index] = f"{layer_target}.conv_state[{request}]"
    targets = tuple(targets)
    active_targets = tuple(targets[index] for index in active_slots)
    if len(set(active_targets)) != len(active_targets):
        raise ModelingError("causal_conv_update_request_id_duplicate")
    if not active_slots:
        empty = {
            "phase": "decode", "queries": q_lens, "padding_slots": padding_slots,
            "active_slots": (), "targets": (None,) * len(views),
            "read_versions": (None,) * len(views),
            "commit_versions": (None,) * len(views),
            "checkpoint_versions": ((),) * len(views),
            "accepted_offsets": (None,) * len(views),
            "advances": (0,) * len(views),
            "rejected_versions": ((),) * len(views),
            "read_records": (), "write_records": (), "commit_records": (),
            "rollback_records": (), "retained_versions": ((),) * len(views),
            "overwritten_versions": ((),) * len(views),
            "pending_acceptance": False, "commit_policy": "no_state_io",
            "candidate_write_point": "device_complete_before_token_commit",
            "state_read_bytes": 0.0, "state_write_bytes": 0.0,
            "version_basis": "invocation_local",
            "history_selection": (), "current_geometry": {},
            "acceptance": {}, "read_bindings": (), "write_bindings": (),
        }
        return empty, (), 0.0, 0.0
    contexts = tuple(int(getattr(views[index], "context", 0))
                     for index in active_slots)
    acceptance = tuple(
        (int(getattr(views[index], "draft_accepted_tokens", 0) or 0),
         int(getattr(views[index], "draft_advanced_tokens", 0) or 0))
        for index in active_slots)
    records = state_records(
        targets=active_targets, lengths=tuple(q_lens[index]
                                              for index in active_slots),
        contexts=contexts, acceptance=acceptance)
    read_bytes = write_bytes = 0.0
    for q in (q_lens[index] for index in active_slots):
        state_read = kernel_width - 1 + (2 + q if speculative else 0)
        state_write = 2 + q if speculative else kernel_width - 1
        read_bytes += state_read * width * byte
        write_bytes += state_write * width * byte
    target_by_view = [None] * len(views)
    read_versions = [None] * len(views)
    commit_versions = [None] * len(views)
    checkpoint_versions = [()] * len(views)
    accepted_offsets = [None] * len(views)
    advances = [0] * len(views)
    retained_versions = [()] * len(views)
    overwritten_versions = [()] * len(views)
    for position, index in enumerate(active_slots):
        target_by_view[index] = active_targets[position]
        read_versions[index] = records["read_versions"][position]
        commit_versions[index] = records["commit_versions"][position]
        checkpoint_versions[index] = records["checkpoint_versions"][position]
        accepted_offsets[index] = records["accepted_offsets"][position]
        advances[index] = records["advances"][position]
        retained_versions[index] = records["retained_versions"][position]
        overwritten_versions[index] = records["overwritten_versions"][position]
    effects = _state_effects(records)
    read_bindings, history_selection = _select_history_bindings(
        bindings=bindings, views=views, active_slots=active_slots, contexts=contexts,
        kernel_width=kernel_width, history_acceptance=history_acceptance,
        reader_holds=reader_holds, rebase_outputs=rebase_outputs,
        q_lens=q_lens, speculative=speculative, slots=slots)
    write_bindings = _candidate_bindings(
        read_bindings=read_bindings, records=records, active_targets=active_targets,
        active_slots=active_slots, slots=slots, layer_target=layer_target,
        q_lens=q_lens, speculative=speculative, kernel_width=kernel_width, width=width)
    contract = {
        **records,
        "phase": "decode",
        "queries": q_lens,
        "padding_slots": padding_slots,
        "active_slots": active_slots,
        "targets": tuple(target_by_view),
        "read_versions": tuple(read_versions),
        "commit_versions": tuple(commit_versions),
        "checkpoint_versions": tuple(checkpoint_versions),
        "accepted_offsets": tuple(accepted_offsets),
        "advances": tuple(advances),
        "rejected_versions": tuple(
            () if index in set(padding_slots)
            else records["rejected_versions"][active_slots.index(index)]
            for index in range(len(views))),
        "retained_versions": tuple(retained_versions),
        "overwritten_versions": tuple(overwritten_versions),
        "pending_acceptance": records["pending_acceptance"],
        "commit_policy": records["commit_policy"],
        "candidate_write_point": records["candidate_write_point"],
        "version_basis": records["version_basis"],
        "state_read_bytes": read_bytes,
        "state_write_bytes": write_bytes,
        "kernel_width": kernel_width,
        "speculative": speculative,
        "state_dtype": None,
        "state_layout": "slots_history_channels",
        "history_slots": history_slots,
        "read_bindings": tuple(binding_record(b) for b in read_bindings),
        "write_bindings": tuple(binding_record(b) for b in write_bindings),
        "history_selection": tuple(history_selection),
        "current_geometry": {
            "queries": tuple(q_lens),
            "active_slots": active_slots,
            "padding_slots": tuple(padding_slots),
            "kernel_width": kernel_width,
            "history_slots": history_slots,
            "width": width,
            "slots": (None if slots is None else tuple(slots)),
        },
        "acceptance": {
            "accepted_offsets": tuple(accepted_offsets),
            "advances": tuple(advances),
            "commit_versions": tuple(commit_versions),
            "rejected_versions": tuple(
                () if index in set(padding_slots)
                else records["rejected_versions"][active_slots.index(index)]
                for index in range(len(views))),
            "pending": records["pending_acceptance"],
            "commit_policy": records["commit_policy"],
        },
        "state_decisions": tuple(state_decisions),
    }
    return contract, tuple(effects), read_bytes, write_bytes


def _fragment(name, effects, ports=None):
    return ProgramFragment(name, state_effects=tuple(effects),
                           ports=PortSet() if ports is None else ports)


def state_fragments(op):
    """Explicit read/write/commit/rollback fragments for one conv op."""
    contract = dict(op.metadata).get("conv_state_contract")
    if contract is None:
        return {"read": None, "write": None, "commits": (), "rollbacks": ()}
    owner = f"kda.layer[{op.layer}]"
    read_bindings = tuple(contract.get("read_bindings", ()))
    write_bindings = tuple(contract.get("write_bindings", ()))
    read = ProgramFragment(
        f"{op.op_id}:state_read",
        state_effects=tuple(
            StateEffect(target, "read", version)
            for target, version in contract["read_records"]),
        ports=state_ports(bindings=read_bindings, owner=owner, direction="in",
                          readiness="previous_iteration", optional=False),
        attrs=(SemanticAttr("state_bindings", tuple(
            binding_record(binding) for binding in read_bindings)),)
        if read_bindings else ())
    write = ProgramFragment(
        f"{op.op_id}:state_write",
        state_effects=tuple(
            StateEffect(target, "write", version)
            for target, version, _slot in contract["write_records"]),
        ports=state_ports(bindings=write_bindings, owner=owner, direction="out",
                          readiness="producer_complete", optional=False),
        attrs=(SemanticAttr("state_bindings", tuple(
            binding_record(binding) for binding in write_bindings)),)
        if write_bindings else ())
    target_index = {
        target: pos
        for pos, target in enumerate(contract.get("targets", ()))
        if target is not None
    }
    commits = []
    for index, (target, version) in enumerate(contract["commit_records"]):
        decision_idx = target_index.get(target, index)
        commits.append(ProgramFragment(
            f"{op.op_id}:state_commit:{index}",
            state_effects=(StateEffect(target, "commit", version, "",
                                       "acceptance_trace"),),
            ports=decision_ports(target=target, version=version, owner=owner),
            attrs=(SemanticAttr("decision", {
                "action": "commit", "target": target, "version": version,
                "event": _decision_name(contract, decision_idx)}),)))
    rollbacks = []
    for index, (target, version) in enumerate(contract["rollback_records"]):
        decision_idx = target_index.get(target, index)
        rollbacks.append(ProgramFragment(
            f"{op.op_id}:state_rollback:{index}",
            state_effects=(StateEffect(target, "rollback", version, "",
                                       "acceptance_trace"),),
            ports=decision_ports(target=target, version=version, owner=owner,
                                action="rollback"),
            attrs=(SemanticAttr("decision", {
                "action": "rollback", "target": target, "version": version,
                "event": _decision_name(contract, decision_idx)}),)))
    return {"read": read, "write": write,
            "commits": tuple(commits), "rollbacks": tuple(rollbacks)}


def _decision_name(contract, index):
    decisions = tuple(contract.get("state_decisions", ()))
    return decisions[index] if index < len(decisions) else None


def state_accesses(op):
    """Project conv UPDATE state reads/writes into SERVICE access records.

    The history window and the speculative reload window are separate ranges;
    a continuous ``kernel_width-1 + 2 + q`` span never exceeds the declared
    storage.  A rebase is a real predecessor: UPDATE reads its output at the
    selected destination window.
    """
    contract = dict(op.metadata).get("conv_state_contract")
    if contract is None:
        return ()
    width = int(contract.get("conv_width", 0) or 0)
    byte = _STATE_DTYPE_BYTES.get(str(contract.get("state_dtype", "bf16")), 2)
    grants = tuple(contract.get("state_grants", ()))
    read_bindings = tuple(contract.get("read_bindings", ()))
    targets = tuple(contract["targets"])
    target_index = {target: index for index, target in enumerate(targets)
                    if target is not None}
    selection = {item["sequence"]: item
                 for item in contract.get("history_selection", ())}
    result = []
    for binding in read_bindings:
        index = target_index[binding["target"]]
        request = str(getattr(op, "requests", ())[index]) if index < len(
            getattr(op, "requests", ())) else str(index)
        q = int(contract["queries"][index])
        history = selection[index]
        source = binding.get("ready") or binding["producer"]
        base = {
            "producer": binding["producer"], "request": request,
            "stage": op.stage, "layer": op.layer, "direction": "read",
            "position": "state", "output": binding["target"],
            "storage": binding["storage"],
            "owner": binding.get("owner", f"kda.layer[{op.layer}]"),
            "representation": "history", "persistent": True,
            "temporary": False, "source": source, "reader": "",
            "grant": binding.get("grant") or (
                grants[index] if index < len(grants) else ""),
            "decision": None, "rank": None, "scope": "target",
            "generation": binding.get("generation"),
            "slot": binding.get("slot"), "offset": binding.get("offset"),
            "physical_key": binding["storage"],
            "request_slot": binding.get("slot"), "rows": 1,
            "row_stride": width, "tier": binding.get("tier", "hbm"),
            "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"),
            "last_use_event": "before_short_convolution",
        }
        if history.get("rebase"):
            rows = int(contract["kernel_width"]) - 1
            start_row = 0
            result.append({
                **base,
                "occurrence": f"{op.op_id}:read:{index}:rebase_output",
                "state_version": binding["state_version"],
                "span": ((start_row, start_row + rows), (0, width)),
                "bytes": float(rows * width * byte),
                "last_use": "before_short_convolution",
                "source_ready": binding.get("ready"),
                "decision_event": None,
                "window": (start_row, start_row + rows),
            })
            continue
        history_rows = int(contract["kernel_width"]) - 1
        source_offset = int(history["source_offset"])
        result.append({
            **base,
            "occurrence": f"{op.op_id}:read:{index}:history",
            "state_version": binding["state_version"],
            "span": ((source_offset, source_offset + history_rows),
                     (0, width)),
            "bytes": float(history_rows * width * byte),
            "last_use": "before_short_convolution",
            "window": (source_offset, source_offset + history_rows),
            "decision_event": None,
        })
        if bool(contract["speculative"]):
            reload_rows = 2 + q
            result.append({
                **base,
                "occurrence": f"{op.op_id}:read:{index}:reload",
                "state_version": binding["state_version"],
                "span": ((0, reload_rows), (0, width)),
                "bytes": float(reload_rows * width * byte),
                "last_use": "before_short_convolution",
                "window": (0, reload_rows),
                "decision_event": None,
            })
    committed_pairs = {
        (target, version)
        for target, version in contract.get("commit_records", ())
    }
    decisions = tuple(contract.get("state_decisions", ()))
    write_groups: dict[str, list[dict]] = {}
    for binding in tuple(contract.get("write_bindings", ())):
        write_groups.setdefault(binding["target"], []).append(binding)
    for target, group in write_groups.items():
        primary = next(
            (item for item in group if (target, item["state_version"]) in committed_pairs),
            group[-1],
        )
        index = target_index[target]
        request = str(getattr(op, "requests", ())[index]) if index < len(
            getattr(op, "requests", ())) else str(index)
        q = int(contract["queries"][index])
        version = primary["state_version"]
        rows = 2 + q if bool(contract["speculative"]) else (
            int(contract["kernel_width"]) - 1)
        slot = primary.get("slot")
        commit = (target, version) in committed_pairs
        decision_event = decisions[index] if index < len(decisions) else None
        result.append({
            "occurrence": f"{op.op_id}:write:{index}:{version}",
            "producer": primary["producer"], "request": request,
            "stage": op.stage, "layer": op.layer, "direction": "write",
            "position": "state", "output": target,
            "storage": primary["storage"],
            "owner": primary.get("owner", f"kda.layer[{op.layer}]"),
            "representation": "history", "persistent": True,
            "temporary": False, "state_version": version,
            "candidate_versions": tuple(item["state_version"] for item in group),
            "span": ((0, rows), (0, width)),
            "bytes": float(rows * width * byte),
            "source": primary["producer"], "reader": "",
            "grant": primary.get("grant") or "",
            "decision": ("token_commit" if commit else "candidate_hold"),
            "last_use": ("next_conv_read" if commit else "last_reader"),
            "rank": None, "scope": "target", "generation": version,
            "slot": slot, "offset": primary.get("offset"),
            "physical_key": _slot_storage(primary["storage"], slot),
            "request_slot": slot, "rows": rows, "row_stride": width,
            "tier": primary.get("tier", "hbm"),
            "source_ready": primary.get("ready"),
            "grant_ready": primary.get("grant"),
            "decision_event": decision_event,
            "window": (0, rows),
            "last_use_event": ("next_conv_read" if commit else "last_reader"),
        })
    return tuple(result)
