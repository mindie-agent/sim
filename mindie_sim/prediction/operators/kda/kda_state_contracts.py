"""Versioned KDA prefill state contracts and layout boundaries.

[Role] Express state initialization, chunk checkpoints and public state commits.
[Invariants] Stores, reads and final publication use real per-head subviews
  and retain their layouts, readiness edges, slots and generation.
[Timeouts & Locks] Metadata work scales with the provided state records.
[Side-Effect Policy] Pure contract, fragment and access records.
"""
from dataclasses import replace
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import ProgramFragment, SemanticAttr, StateEffect
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .kda_state_selection import (
    bind_state_value, binding_record, prefill_records, state_ports, state_value,
)

_K_DIM = 128
_V_DIM = 128
_CHUNK = 64
_SLOTTED_STATE_TRANSPOSE = (0, 1, 3, 2)


def _effect(target, action, version, arbiter=""):
    return StateEffect(target, action, version, "", arbiter)


def _state_effect(target, action, version, arbiter=""):
    return _effect(target, action, version, arbiter).to_dict()


def _per_sequence(value, count, default=None):
    if value is None or value == () or value == []:
        return (default,) * count
    if isinstance(value, Mapping):
        return tuple(value.get(index, default) for index in range(count))
    value = tuple(value)
    if len(value) != count:
        raise ModelingError("kda_prefill_values_must_match_sequences")
    return value


def _sequence_values(value, count, shape):
    """Split one pooled TensorValue on its leading sequence axis."""
    if isinstance(value, TensorValue):
        layout = tuple(value.layout.shape)
        if len(layout) == len(shape) + 1 and layout[0] == count \
                and layout[1:] == tuple(shape):
            return tuple(value.narrow(0, index, 1) for index in range(count))
        if count == 1 and layout == tuple(shape):
            return (value,)
    return _per_sequence(value, count)


def _public_value(value, *, target, shape, dtype, context, ready):
    if value is None:
        return TensorValue(
            "external", dtype, TensorLayout.contiguous(shape), output=0,
            storage=f"public_state:{target}", binding="external",
            binding_name=ready or f"kda.public_state[{target}]",
            state_version=context)
    value = state_value(value)
    if tuple(value.layout.shape[-len(shape):]) != shape:
        raise ModelingError(
            f"kda_prefill_public_value_shape_mismatch:{value.layout.shape}:{shape}")
    if value.dtype != dtype:
        raise ModelingError(
            f"kda_prefill_public_value_dtype_mismatch:{value.dtype}:{dtype}")
    if value.state_version is not None and value.state_version != context:
        raise ModelingError(
            f"kda_prefill_public_value_version_mismatch:"
            f"{value.state_version}:{context}")
    return value


def _head_view(value, head, *, heads):
    shape = tuple(value.layout.shape)
    if len(shape) == 3:
        return value.permute((0, 2, 1)).narrow(0, head, 1)
    if len(shape) == 4 and shape[0] == 1:
        return value.permute(_SLOTTED_STATE_TRANSPOSE).narrow(1, head, 1)
    raise ModelingError(
        f"kda_prefill_public_value_requires_head_axis:{shape}:{heads}")


_LAYOUT_TAIL_RANK = 3


def _build_publication_bindings(targets, records, layouts, shape, state_dtype, owner, heads, dv, dk):
    publication_bindings = []
    for sequence, target in enumerate(targets):
        final = records["publication_write_records"][sequence]["version"]
        layout = layouts[sequence]
        if layout is None:
            layout = TensorValue(
                records["layout_producer"], state_dtype,
                TensorLayout.contiguous(shape), output=sequence,
                storage=f"public_state:{target}", binding="static",
                binding_name="", state_version=final)
        else:
            layout = state_value(layout)
            if tuple(layout.layout.shape[-_LAYOUT_TAIL_RANK:]) != shape:
                raise ModelingError(
                    f"kda_prefill_layout_shape_mismatch:{layout.layout.shape}:{shape}")
            if layout.dtype != state_dtype:
                raise ModelingError(
                    f"kda_prefill_layout_dtype_mismatch:{layout.dtype}:{state_dtype}")
            if layout.state_version is not None and layout.state_version != final:
                raise ModelingError(
                    f"kda_prefill_layout_version_mismatch:{layout.state_version}:{final}")
        publication_bindings.append(bind_state_value(
            layout, target=target, owner=owner, version=final,
            ready=layout.producer, generation=final, required=True,
            representation="layout",
            span=((0, heads), (0, dv), (0, dk)),
            readiness="producer_complete",
            last_use="after_layout_publication"))
    return publication_bindings


def prefill_state_contract(*, lengths, requests=(), chunk=_CHUNK, contexts=None,
                           targets=None, heads=1, dk=_K_DIM, dv=_V_DIM,
                           state_dtype="fp32", op_id="kda_fwd_h",
                           state_values=None, layout_values=None,
                           state_ready=(), state_grants=(), owner=None):
    """Declare per-head FWD_H stores from real public subviews.

    The initialize stage reads a caller ``TensorValue`` subview of the public
    ``[V,K]`` state (or an explicit standalone external port).  Chunk stores,
    internal reads, the final layout producer and the public commit all derive
    from that same binding.  No internal target or public version is prefilled.
    """
    lengths = tuple(lengths)
    if targets is None:
        requests = tuple(requests)
        targets = tuple(
            f"kda.layer.recurrent_state["
            f"{requests[index] if index < len(requests) else index}]"
            for index in range(len(lengths)))
    targets = tuple(targets)
    records = prefill_records(
        targets=targets, lengths=lengths, contexts=contexts, chunk=chunk,
        heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, op_id=op_id)
    contexts = records["contexts"]
    count = len(targets)
    ready = _per_sequence(state_ready, count, "")
    grants = _per_sequence(state_grants, count, None)
    shape = (heads, dv, dk)
    values = _sequence_values(state_values, count, shape)
    layouts = _sequence_values(layout_values, count, shape)
    owner = owner or "kda.layer"
    read_bindings, initialize_bindings = [], []
    internal_bindings = {}
    for sequence, (target, context) in enumerate(zip(targets, contexts)):
        public = _public_value(
            values[sequence], target=target, shape=shape, dtype=state_dtype,
            context=context, ready=ready[sequence])
        read_bindings.append(bind_state_value(
            public, target=target, owner=owner, version=context,
            ready=ready[sequence] or public.producer, grant=grants[sequence],
            generation=context, required=True, representation="state",
            span=((0, heads), (0, dv), (0, dk)),
            readiness=("external" if public.binding == "external"
                       else "producer_complete")))
        for head in range(heads):
            head_value = _head_view(public, head, heads=heads)
            internal = records["internal_targets"][sequence][head]
            initial = replace(
                head_value, producer=f"{op_id}:initialize",
                output=f"{sequence}:{head}",
                storage=f"state:{sequence}:head:{head}", binding="static",
                binding_name="", state_version=context)
            initialize_bindings.append(bind_state_value(
                initial, target=internal, owner=f"{owner}.head[{head}]",
                version=context, generation=context, required=True,
                representation=state_dtype, span=((0, dk), (0, dv))))
            previous = initial
            for chunk_index, version in enumerate(
                    records["checkpoint_versions"][sequence]):
                chunk_value = replace(
                    previous, producer=f"{op_id}:state_chunk:"
                                      f"{sequence}:{head}:{chunk_index}",
                    output=chunk_index, binding="static", binding_name="",
                    state_version=version)
                internal_bindings[(sequence, head, chunk_index)] = (
                    bind_state_value(
                        chunk_value, target=internal,
                        owner=f"{owner}.head[{head}]", version=version,
                        generation=version, required=True,
                        representation=state_dtype,
                        span=((0, dk), (0, dv))))
                previous = chunk_value
    publication_bindings = _build_publication_bindings(
        targets, records, layouts, shape, state_dtype, owner, heads, dv, dk)
    publication_effects = []
    for record in records["publication_write_records"]:
        publication_effects.append(_state_effect(
            record["target"], "write", record["version"],
            "layout_producer_complete"))
    for record in records["publication_records"]:
        publication_effects.append(_state_effect(
            record["target"], "commit", record["version"],
            "layout_publication"))
    contract = {
        **{key: records[key] for key in (
            "phase", "queries", "contexts", "version_basis", "targets",
            "heads", "dk", "dv", "state_dtype", "internal_targets",
            "layout_producer", "layout_bytes",
            "checkpoint_versions", "internal_checkpoint_versions",
            "internal_commit_versions", "commit_versions", "commit_policy",
            "candidate_write_point", "publication_point")},
        "chunk": chunk,
        "request_ids": tuple(requests),
        "owner": owner,
        "state_ready": tuple(ready),
        "state_grants": tuple(grants),
        "read_versions": tuple(records["contexts"]),
        "read_records": tuple(records["read_records"]),
        "initialize_write_records": tuple(records["initialize_write_records"]),
        "internal_read_records": tuple(records["internal_read_records"]),
        "internal_write_records": tuple(records["internal_write_records"]),
        "internal_commit_records": tuple(records["internal_commit_records"]),
        "publication_write_records": tuple(records["publication_write_records"]),
        "publication_records": tuple(records["publication_records"]),
        "publication_write_records": tuple(records["publication_write_records"]),
        "read_bindings": tuple(binding_record(b) for b in read_bindings),
        "initialize_bindings": tuple(
            binding_record(b) for b in initialize_bindings),
        "internal_bindings": tuple(
            (key, binding_record(b))
            for key, b in sorted(internal_bindings.items())),
        "publication_bindings": tuple(
            binding_record(b) for b in publication_bindings),
        "history_selection": {
            "contexts": tuple(records["contexts"]),
            "version_basis": records["version_basis"],
            "public_subviews": tuple(
                (sequence, context) for sequence, context in
                enumerate(records["contexts"])),
        },
        "current_geometry": {
            "queries": records["queries"],
            "heads": heads, "dk": dk, "dv": dv, "chunk": chunk,
            "targets": targets, "request_ids": tuple(requests),
        },
        "internal_record_count": len(records["internal_commit_records"]),
    }
    return contract, tuple(publication_effects)
def _fragment(name, effects, ports=None):
    return ProgramFragment(name, state_effects=tuple(effects),
                           ports=PortSet() if ports is None else ports)


def _prefill_publication_fragments(op, contract):
    result = []
    for index, (write, commit) in enumerate(zip(
            contract["publication_write_records"],
            contract["publication_records"])):
        result.append(_fragment(
            f"{op.op_id}:state_publication:{index}",
            (_effect(write["target"], "write", write["version"]),
             _effect(commit["target"], "commit", commit["version"],
                     "caller_layout_publication"))))
    return tuple(result)


def _binding_attr(bindings):
    if not bindings:
        return ()
    return (SemanticAttr(
        "state_bindings",
        tuple(binding_record(binding) for binding in bindings)),)


def _binding_ports(bindings, *, direction, owner, readiness,
                   optional=False):
    return state_ports(
        bindings=bindings, owner=owner, direction=direction,
        readiness=readiness, optional=optional)


def state_fragments(op):
    """Return per-head FWD_H stores and the final layout publication.

    ``initialize`` reads the public subview and writes the first internal
    values; ``internal`` is keyed by ``(sequence, head, chunk)`` and reads the
    previous internal version before its own write/commit; ``publication`` is
    one fragment per sequence whose write/commit follows the layout producer.
    """
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is None:
        return {"read": None, "initialize": None, "internal": {},
                "publication": ()}
    owner = contract.get("owner", f"kda.layer[{op.layer}]")
    read_bindings = tuple(contract.get("read_bindings", ()))
    read_effects = tuple(
        _effect(record["target"], "read", record["version"])
        for record in contract["read_records"])
    read = ProgramFragment(
        f"{op.op_id}:state_read", state_effects=read_effects,
        ports=_binding_ports(read_bindings, direction="in", owner=owner,
                             readiness="previous_iteration", optional=False),
        attrs=_binding_attr(read_bindings))
    initialize_bindings = tuple(contract.get("initialize_bindings", ()))
    initialize_effects = read_effects + tuple(
        _effect(record["target"], "write", record["version"])
        for record in contract["initialize_write_records"])
    initialize_ports = PortSet(
        tensor_in=_binding_ports(
            read_bindings, direction="in", owner=owner,
            readiness="previous_iteration", optional=False).tensor_in,
        tensor_out=_binding_ports(
            initialize_bindings, direction="out", owner=owner,
            readiness="producer_complete", optional=False).tensor_out)
    initialize = ProgramFragment(
        f"{op.op_id}:state_initialize", state_effects=initialize_effects,
        ports=initialize_ports,
        attrs=_binding_attr((*read_bindings, *initialize_bindings)))
    internal = {}
    by_key = dict(contract.get("internal_bindings", ()))
    reads = {(record["sequence"], record["head"], record["chunk"]):
             record for record in contract.get("internal_read_records", ())}
    writes = {(record["sequence"], record["head"], record["chunk"]):
              record for record in contract.get("internal_write_records", ())}
    commits = {(record["sequence"], record["head"], record["chunk"]):
               record for record in contract.get("internal_commit_records", ())}
    for key, binding in by_key.items():
        read_record = reads[key]
        write_record = writes[key]
        commit_record = commits[key]
        effects = (
            _effect(read_record["target"], "read", read_record["version"]),
            _effect(write_record["target"], "write", write_record["version"]),
            _effect(commit_record["target"], "commit", commit_record["version"]),
        )
        internal[key] = ProgramFragment(
            f"{op.op_id}:state_chunk:{key[0]}:{key[1]}:{key[2]}",
            state_effects=effects,
            ports=_binding_ports(
                (binding,), direction="inout",
                owner=binding.get("owner", owner),
                readiness="producer_complete", optional=False),
            attrs=_binding_attr((binding,)))
    publication = []
    publication_bindings = tuple(contract.get("publication_bindings", ()))
    for sequence, binding in enumerate(publication_bindings):
        final_internal = []
        for head in range(int(contract["heads"])):
            chunk = len(contract["internal_checkpoint_versions"][sequence][head]) - 1
            final_internal.append(by_key[(sequence, head, chunk)])
        effects = []
        for internal_binding in final_internal:
            effects.append(_effect(
                internal_binding["target"], "read",
                internal_binding["state_version"]))
        write = contract["publication_write_records"][sequence]
        commit = contract["publication_records"][sequence]
        effects.append(_effect(write["target"], "write", write["version"]))
        effects.append(_effect(
            commit["target"], "commit", commit["version"], "layout_publication"))
        ports = PortSet(
            tensor_in=_binding_ports(
                tuple(final_internal), direction="in", owner=owner,
                readiness="producer_complete", optional=False).tensor_in,
            tensor_out=_binding_ports(
                (binding,), direction="out", owner=owner,
                readiness="producer_complete", optional=False).tensor_out,
            event_in=(EventPortSpec(
                name=(f"layout:{sequence}:{binding['producer']}:"
                      f"{binding['state_version']}"),
                direction="in", event="complete", scope="request",
                owner=binding["producer"], readiness="producer_complete",
                generation=binding["state_version"], optional=False),))
        publication.append(ProgramFragment(
            f"{op.op_id}:state_publication:{sequence}",
            state_effects=tuple(effects), ports=ports,
            attrs=_binding_attr((*final_internal, binding))))
    return {"read": read, "initialize": initialize, "internal": internal,
            "publication": tuple(publication)}


def layout_fragment(op):
    """Combine per-sequence publication fragments into one real layout stage."""
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is None:
        return None
    fragments = state_fragments(op)["publication"]
    if not fragments:
        return None
    effects = tuple(
        effect for fragment in fragments for effect in fragment.state_effects)
    tensor_in = tuple(
        port for fragment in fragments for port in fragment.ports.tensor_in)
    tensor_out = tuple(
        port for fragment in fragments for port in fragment.ports.tensor_out)
    event_in = tuple(
        port for fragment in fragments for port in fragment.ports.event_in)
    bindings = tuple(contract.get("publication_bindings", ()))
    attrs = ((SemanticAttr("state_bindings", bindings),) if bindings else ())
    return ProgramFragment(
        f"{op.op_id}:state_layout", state_effects=effects,
        ports=PortSet(tensor_in=tensor_in, tensor_out=tensor_out,
                      event_in=event_in),
        attrs=attrs)
def _kda_access_row(
    op, contract, *, occurrence, producer, request, direction, position, record, binding,
    owner, source, grant, decision, last_use, generation, physical_key, sequence, rows,
    decision_event=None,
):
    return {
        "occurrence": occurrence, "producer": producer, "request": request,
        "stage": op.stage, "layer": op.layer, "direction": direction, "position": position,
        "output": record["target"], "storage": binding["storage"], "owner": owner,
        "representation": record["dtype"], "persistent": True, "temporary": False,
        "state_version": record["version"], "span": record["span"],
        "bytes": float(record["bytes"]), "source": source, "reader": "", "grant": grant,
        "decision": decision, "last_use": last_use, "rank": None, "scope": "target",
        "generation": generation, "slot": None, "window": None,
        "offset": binding.get("offset"), "physical_key": physical_key,
        "request_slot": sequence, "rows": rows,
        "row_stride": int(contract["dk"]) * int(contract["dv"]),
        "tier": "hbm", "source_ready": binding.get("ready"),
        "grant_ready": binding.get("grant"), "decision_event": decision_event,
        "last_use_event": last_use,
    }


def state_accesses(op):
    """Project FWD_H physical reads/writes into SERVICE access records."""
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is None:
        return ()
    owner = contract.get("owner", f"kda.layer[{op.layer}]")
    request_ids = tuple(contract.get("request_ids", ()))
    read_bindings = tuple(contract.get("read_bindings", ()))
    initialize_bindings = tuple(contract.get("initialize_bindings", ()))
    publication_bindings = tuple(contract.get("publication_bindings", ()))
    result = []
    for record in contract["read_records"]:
        sequence, head = record["sequence"], record["head"]
        binding = read_bindings[sequence]
        request = request_ids[sequence] if sequence < len(request_ids) else str(sequence)
        result.append(_kda_access_row(
            op, contract, occurrence=f"{op.op_id}:read:{sequence}:{head}",
            producer=binding["producer"], request=request, direction="read", position="state",
            record=record, binding=binding, owner=f"{owner}.head[{head}]",
            source=binding.get("ready") or binding["producer"],
            grant=binding.get("grant") or "", decision=None,
            last_use="before_head_initialize", generation=record["version"],
            physical_key=f"{binding['storage']}:head:{head}", sequence=sequence, rows=1))
    for record, binding in zip(contract["initialize_write_records"], initialize_bindings):
        sequence, head = record["sequence"], record["head"]
        request = request_ids[sequence] if sequence < len(request_ids) else str(sequence)
        result.append(_kda_access_row(
            op, contract, occurrence=f"{op.op_id}:initialize:{sequence}:{head}",
            producer=record["producer"], request=request, direction="write", position="state",
            record=record, binding=binding, owner=f"{owner}.head[{head}]",
            source=record["producer"], grant="", decision="initial_materialization",
            last_use="next_head_chunk_read", generation=record["version"],
            physical_key=f"{binding['storage']}:head:{head}", sequence=sequence, rows=1))
    internal_reads = {(r["sequence"], r["head"], r["chunk"]): r for r in contract["internal_read_records"]}
    internal_writes = {(r["sequence"], r["head"], r["chunk"]): r for r in contract["internal_write_records"]}
    for key, binding in contract.get("internal_bindings", ()):
        sequence, head, chunk = key
        request = request_ids[sequence] if sequence < len(request_ids) else str(sequence)
        read_record, write_record = internal_reads[key], internal_writes[key]
        head_owner, phys_key = f"{owner}.head[{head}]", f"{binding['storage']}:head:{head}"
        result.append(_kda_access_row(
            op, contract, occurrence=f"{op.op_id}:chunk_read:{sequence}:{head}:{chunk}",
            producer=binding["producer"], request=request, direction="read", position="state",
            record=read_record, binding=binding, owner=head_owner,
            source=binding["producer"], grant="", decision=None,
            last_use="before_chunk_store", generation=read_record["version"],
            physical_key=phys_key, sequence=sequence, rows=1))
        result.append(_kda_access_row(
            op, contract, occurrence=f"{op.op_id}:write:{sequence}:{head}:{chunk}",
            producer=write_record["producer"], request=request, direction="write", position="state",
            record=write_record, binding=binding, owner=head_owner,
            source=write_record["producer"], grant="", decision="internal_chunk_commit",
            last_use="next_head_chunk_read", generation=write_record["generation"],
            physical_key=phys_key, sequence=sequence, rows=1))
    for sequence, binding in enumerate(publication_bindings):
        record = contract["publication_write_records"][sequence]
        request = request_ids[sequence] if sequence < len(request_ids) else str(sequence)
        result.append(_kda_access_row(
            op, contract, occurrence=f"{op.op_id}:public:{sequence}",
            producer=binding["producer"], request=request, direction="write", position="layout",
            record=record, binding=binding, owner=owner,
            source=binding["producer"], grant="", decision="layout_complete",
            last_use="after_layout_publication", generation=record["version"],
            physical_key=f"{binding['storage']}:layout", sequence=sequence,
            rows=int(contract["heads"]), decision_event=binding.get("ready")))
    return tuple(result)
