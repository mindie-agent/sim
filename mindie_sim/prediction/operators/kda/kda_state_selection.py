"""Explicit previous-state selection for the retained recurrent KDA ABI.

[Role] Resolve exact state-slot addresses or retain caller-declared validity.
[Invariants] Acceptance selects the initial checkpoint only: every current
  query executes and writes its checkpoint. State pools of independent
  sequences cannot alias. Metadata capacities never supply missing values.
  Version transitions declare one read/write/commit/reject boundary; an
  undeclared multi-row acceptance stays pending instead of committing.
  ``state_records``/``prefill_records`` bind those versions to explicit
  per-target write/commit records; an overwritten same-sequence slot has no
  live version unless a materialization plan is declared.
[Timeouts & Locks] O(batch * query rows), q <= 8 in the consumer; no locks.
[Side-Effect Policy] Pure contract validation and immutable tuple results.
"""
from itertools import accumulate
from typing import Any, Literal, Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.subgraphs import (
    EventPortSpec, PortSet, TensorPortSpec,
)
from mindie_sim.schema.tensors import TensorLayout, TensorValue

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_RECURRENT_REV = "8c30e4f491f73cba737a1dacc522894e437cdf3d"
_VLLM_ASCEND_FWD_H_REV = "ae923905948f360f4b7ab06df52d0789a0c9316e"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        _VLLM_ASCEND_RECURRENT_REV,
        "csrc/attention/recurrent_kda/op_kernel/arch35/recurrent_kda.h",
        "f35dd2c5fb604d6d9711c11139c755414d49fb27fc6aea0f3a4bec3d3adb0199",
    ),
    SourceRef(
        "vllm-ascend-v41-private",
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp",
        "aa9695e2d9eba0dc0963d248199a86ab5cbd321c0f5836a9dd8ee8901944cf13",
    ),
)

PREFILL_DEFAULT_CHUNK_TOKENS = 64
PREFILL_DEFAULT_KEY_DIM = 128
PREFILL_DEFAULT_VALUE_DIM = 128
_PREFILL_SNAPSHOT_DTYPE = "bf16"


def resolve(raw, lengths):
    """Return an address contract, without inventing scalar service latency."""
    if raw is None:
        return None
    fields = {"kind", "index_layout", "index_row_stride", "dependencies",
              "prefix", "state_indices", "accepted_tokens", "state_capacity", "validity"}
    if set(raw) - fields:
        raise ModelingError("recurrent_kda_state_selection_field_unknown")
    kind = raw.get("kind")
    layout = raw.get("index_layout")
    if layout not in ("packed", "speculative"):
        raise ModelingError("recurrent_kda_state_index_layout_required")
    stride = raw.get("index_row_stride", 0)
    if type(stride) is not int or (layout == "packed" and stride != 0) or (
            layout == "speculative" and stride < max(lengths)):
        raise ModelingError("recurrent_kda_state_index_stride_mismatch")
    inputs = raw.get("dependencies", ())
    if not isinstance(inputs, (list, tuple)):
        raise ModelingError("recurrent_kda_state_input_dependencies_must_be_sequence")
    dependencies = tuple(inputs)
    if any(not isinstance(name, str) or not name for name in dependencies):
        raise ModelingError("recurrent_kda_state_input_dependency_invalid")
    prefix = tuple(accumulate(lengths, initial=0))
    result = {
        "kind": kind, "index_layout": layout, "index_row_stride": stride,
        "prefix": prefix, "dependencies": dependencies,
        "initial_state": "indices[b,accepted[b]-1]",
        "checkpoint_state": "indices[b,t] for every current query t",
        "acceptance_changes_query_work": False,
        "state_output_alias": "initial_state_pool; one selected write destination",
        "capture_value_identity": "not_established",
    }
    if kind == "symbolic_valid_selection":
        if any(name in raw for name in ("prefix", "state_indices", "accepted_tokens", "state_capacity")):
            raise ModelingError("recurrent_kda_symbolic_selection_contains_exact_values")
        required = {"prefix", "state_indices", "accepted_tokens", "independent_sequences"}
        if set(raw.get("validity", {})) != required or any(
                raw["validity"][name] is not True for name in required):
            raise ModelingError("recurrent_kda_symbolic_selection_requires_explicit_validity")
        result.update(address_basis="caller_declared_valid_ranges; values_not_inferred",
                      validity_assumptions=tuple(sorted(required)))
        return result
    if kind != "exact":
        raise ModelingError("recurrent_kda_state_selection_kind_unsupported")
    if "validity" in raw:
        raise ModelingError("recurrent_kda_exact_selection_uses_values_not_validity_claim")
    supplied_prefix = tuple(raw.get("prefix", ()))
    if any(type(value) is not int for value in supplied_prefix) or supplied_prefix != prefix:
        raise ModelingError("recurrent_kda_state_prefix_must_match_current_queries")
    capacity = raw.get("state_capacity")
    accepted = tuple(raw.get("accepted_tokens", ()))
    if type(capacity) is not int or capacity < 1:
        raise ModelingError("recurrent_kda_state_capacity_required")
    if len(accepted) != len(lengths) or any(type(a) is not int or not 1 <= a <= q
                                          for a, q in zip(accepted, lengths)):
        raise ModelingError("recurrent_kda_accepted_values_outside_sequence")
    indices = raw.get("state_indices", ())
    if layout == "packed":
        if len(indices) < prefix[-1]:
            raise ModelingError("recurrent_kda_packed_state_indices_too_short")
        rows = tuple(tuple(indices[start:end]) for start, end in zip(prefix, prefix[1:]))
    else:
        if len(indices) != len(lengths) or any(len(row) != stride for row in indices):
            raise ModelingError("recurrent_kda_speculative_state_index_shape_mismatch")
        rows = tuple(tuple(row[:q]) for row, q in zip(indices, lengths))
    owned = set()
    for row in rows:
        if any(type(slot) is not int or not 0 <= slot < capacity for slot in row):
            raise ModelingError("recurrent_kda_state_slot_outside_pool")
        # Rewriting one sequence's same slot is a valid ordered checkpoint.
        # Across sequences it is an unordered read/write race, outside this
        # independent-sequence physical program's declared support domain.
        if owned.intersection(row):
            raise ModelingError("recurrent_kda_cross_sequence_state_alias_unsupported")
        owned.update(row)
    result.update(address_basis="explicit_caller_values", state_capacity=capacity,
                  accepted_tokens=accepted, checkpoint_slots=rows,
                  initial_slots=tuple(row[a-1] for row, a in zip(rows, accepted)))
    return result


def pcp_remote_publish(query_ranges, q_len, global_q_len, *, pcp=1,
                       padding=False):
    """Whether one PCP rank must leave the final global state to its peers.

    A rank owns the whole global sequence only when it carries every query
    token (``q_len == global_q_len``) and its declared ranges tile
    ``[0, global_q_len)`` without a gap.  A rank that covers the local head and
    tail of a longer global sequence still misses the middle owned by another
    rank, so its scatter cannot publish a global commit; the same is true for a
    rank that owns only a prefix/suffix.  Returning ``False`` for a
    single-rank caller keeps the ordinary sequence-tail commit unchanged.
    """
    if pcp <= 1 or padding:
        return False
    global_q = int(global_q_len or q_len)
    if global_q <= 0:
        return True
    if int(q_len) != global_q:
        return True
    ranges = tuple((int(start), int(end)) for start, end in (query_ranges or ()))
    if not ranges:
        return False
    if ranges[0][0] != 0 or ranges[-1][1] != global_q:
        return True
    return any(prev_end != next_start
               for (_start, prev_end), (next_start, _end)
               in zip(ranges, ranges[1:]))


def version_transition(lengths, *, contexts=None, acceptance=None, phase="decode"):
    """Declare one invocation's state read/write/commit/reject versions.

    ``contexts`` is the committed token count before this invocation.  When
    the caller does not supply it, versions are invocation-local: the operator
    is standalone and must not invent a target request's target context.  Each
    query still writes its own checkpoint; only the accepted prefix becomes
    visible, while later candidates are rejected individually.  A multi-row
    verification without a declared advance remains pending rather than
    committing every row.  This is a state contract, not an executor.
    """
    if phase != "decode":
        raise ModelingError("recurrent_kda_version_transition_requires_decode")
    lengths = tuple(lengths)
    if not lengths or any(type(q) is not int or q < 1 for q in lengths):
        raise ModelingError("recurrent_kda_version_lengths_must_be_positive_ints")
    if contexts is None:
        contexts = (0,) * len(lengths)
        basis = "invocation_local"
    else:
        contexts = tuple(contexts)
        if len(contexts) != len(lengths) or any(
                type(value) is not int or value < 0 for value in contexts):
            raise ModelingError("recurrent_kda_version_contexts_must_match_sequences")
        basis = "caller_context"
    if acceptance is None:
        acceptance = (None,) * len(lengths)
    else:
        acceptance = tuple(acceptance)
        if len(acceptance) != len(lengths):
            raise ModelingError("recurrent_kda_version_acceptance_must_match_sequences")
    queries, advances, accepted_offsets = [], [], []
    checkpoint_versions, commit_versions, rollback_versions = [], [], []
    pending = False
    for q, context, declared in zip(lengths, contexts, acceptance):
        slot_pending = False
        if declared is not None:
            if not isinstance(declared, (tuple, list)) or len(declared) != 2:
                raise ModelingError("recurrent_kda_version_acceptance_pair_required")
            accepted_raw, advance_raw = declared
            if any(type(value) is not int for value in (accepted_raw, advance_raw)):
                raise ModelingError("recurrent_kda_version_acceptance_must_be_ints")
        else:
            accepted_raw, advance_raw = 0, 0
        if q == 1:
            if accepted_raw != 0 or advance_raw not in (0, 1):
                raise ModelingError("recurrent_kda_single_row_acceptance_invalid")
            accepted, advance = 0, 1
        elif declared is None or advance_raw <= 0:
            # Verification candidates exist but the acceptance event has not
            # been declared.  Do not infer that every row commits.
            accepted, advance, slot_pending = 0, 0, True
            pending = True
        else:
            if not 1 <= advance_raw <= q:
                raise ModelingError("recurrent_kda_advance_outside_query")
            accepted = advance_raw - 1
            if accepted_raw not in (0, accepted):
                raise ModelingError("recurrent_kda_accepted_advance_mismatch")
            advance = advance_raw
        versions = tuple(range(context + 1, context + q + 1))
        queries.append(q)
        advances.append(advance)
        accepted_offsets.append(accepted)
        checkpoint_versions.append(versions)
        commit_versions.append(None if slot_pending else context + advance)
        rollback_versions.append(() if slot_pending else versions[advance:])
    return {
        "phase": phase,
        "queries": tuple(queries),
        "contexts": tuple(contexts),
        "version_basis": basis,
        "read_versions": tuple(contexts),
        "checkpoint_versions": tuple(checkpoint_versions),
        "accepted_offsets": tuple(accepted_offsets),
        "advances": tuple(advances),
        "commit_versions": tuple(commit_versions),
        "rejected_versions": tuple(rollback_versions),
        "checkpoints": tuple(queries),
        "pending_acceptance": pending,
        "commit_policy": ("pending_acceptance_event" if pending else "token_commit"),
        "candidate_write_point": "device_complete_before_token_commit",
    }


def state_records(*, targets, lengths, contexts=None, acceptance=None,
                  checkpoint_slots=None):
    """Bind one transition to explicit write/commit/rollback records.

    ``targets`` is the stable per-sequence state identity supplied by the
    lowerer (layer/request, not an occurrence).  ``checkpoint_slots`` is the
    caller's physical address map.  A logical candidate overwritten by a later
    checkpoint on the same slot is not retained: it has no live version unless
    a materialization plan is declared.  The source query work is unchanged.
    """
    targets = tuple(targets)
    lengths = tuple(lengths)
    if len(targets) != len(lengths):
        raise ModelingError("kda_state_targets_must_match_sequences")
    if any(not isinstance(target, str) or not target for target in targets):
        raise ModelingError("kda_state_target_required")
    if len(set(targets)) != len(targets):
        raise ModelingError("kda_state_target_duplicate")
    transition = version_transition(lengths, contexts=contexts, acceptance=acceptance)
    if checkpoint_slots is not None:
        checkpoint_slots = tuple(tuple(row) for row in checkpoint_slots)
        if len(checkpoint_slots) != len(lengths):
            raise ModelingError("kda_state_checkpoint_slots_must_match_sequences")
    write_records, commit_records, rollback_records = [], [], []
    retained_versions, overwritten_versions = [], []
    for index, (target, q) in enumerate(zip(targets, transition["queries"])):
        versions = transition["checkpoint_versions"][index]
        slots = None if checkpoint_slots is None else checkpoint_slots[index]
        if slots is not None:
            if len(slots) != q or any(
                    type(slot) is not int or slot < 0 for slot in slots):
                raise ModelingError("kda_state_checkpoint_slots_shape_invalid")
            last_by_slot = {slot: position for position, slot in enumerate(slots)}
            retained = tuple(sorted(last_by_slot.values()))
            overwritten = tuple(position for position in range(q)
                                if position not in set(retained))
        else:
            retained = tuple(range(q))
            overwritten = ()
        pending = transition["commit_versions"][index] is None
        accepted = transition["accepted_offsets"][index]
        if not pending and accepted not in retained:
            raise ModelingError(
                "kda_state_accepted_checkpoint_overwritten_requires_materialization")
        retained_versions.append(tuple(versions[position] for position in retained))
        overwritten_versions.append(tuple(versions[position] for position in overwritten))
        write_records.extend(
            (target, versions[position],
             None if slots is None else slots[position])
            for position in range(q))
        if not pending:
            commit_records.append((target, versions[accepted]))
            for position in range(transition["advances"][index], q):
                if position in retained:
                    rollback_records.append((target, versions[position]))
    return {
        **transition,
        "targets": targets,
        "read_records": tuple(
            (target, transition["read_versions"][index])
            for index, target in enumerate(targets)),
        "write_records": tuple(write_records),
        "commit_records": tuple(commit_records),
        "rollback_records": tuple(rollback_records),
        "retained_versions": tuple(retained_versions),
        "overwritten_versions": tuple(overwritten_versions),
    }


def prefill_records(*, targets, lengths, contexts=None, chunk=PREFILL_DEFAULT_CHUNK_TOKENS, heads=1,
                    dk=PREFILL_DEFAULT_KEY_DIM, dv=PREFILL_DEFAULT_VALUE_DIM, state_dtype="fp32", op_id="kda_fwd_h",
                    layout_producer=None, layout_outputs=None):
    """Per-head FWD_H stores, internal reads and the public layout publication.

    The internal FP32 state is per ``(sequence, head)``.  The initialize stage
    reads the actual public ``[V,K]`` subview and writes the first internal
    ``[K,V]`` value; every chunk then reads the previous internal version and
    writes/commits its own.  The public target is written and committed only
    by the final layout producer, never by the FWD_H occurrence name.
    """
    targets = tuple(targets)
    lengths = tuple(lengths)
    if len(targets) != len(lengths):
        raise ModelingError("kda_prefill_targets_must_match_sequences")
    if any(not isinstance(target, str) or not target for target in targets):
        raise ModelingError("kda_prefill_target_required")
    if len(set(targets)) != len(targets):
        raise ModelingError("kda_prefill_target_duplicate")
    if not lengths or any(type(q) is not int or q < 1 for q in lengths):
        raise ModelingError("kda_prefill_lengths_must_be_positive_ints")
    if type(chunk) is not int or chunk < 1:
        raise ModelingError("kda_prefill_chunk_must_be_positive")
    if type(heads) is not int or heads < 1:
        raise ModelingError("kda_prefill_heads_must_be_positive")
    if any(type(value) is not int or value < 1 for value in (dk, dv)):
        raise ModelingError("kda_prefill_head_dims_must_be_positive")
    if str(state_dtype) not in {"fp32", "bf16"}:
        raise ModelingError("kda_prefill_state_dtype_unsupported")
    state_byte = int(dtype_bytes(state_dtype))
    if contexts is None:
        contexts = (0,) * len(lengths)
        basis = "invocation_local"
    else:
        contexts = tuple(contexts)
        if len(contexts) != len(lengths) or any(
                type(value) is not int or value < 0 for value in contexts):
            raise ModelingError("kda_prefill_contexts_must_match_sequences")
        basis = "caller_context"
    if layout_outputs is not None:
        layout_outputs = tuple(layout_outputs)
        if len(layout_outputs) != len(lengths):
            raise ModelingError("kda_prefill_layout_outputs_must_match_sequences")
    else:
        layout_outputs = tuple(sequence for sequence in range(len(lengths)))
    internal_targets = tuple(
        tuple(f"{target}.head[{head}]" for head in range(heads))
        for target in targets)
    read_records, initialize_write_records = [], []
    internal_read_records = []
    internal_write_records, internal_commit_records = [], []
    checkpoint_versions, internal_checkpoint_versions = [], []
    internal_commit_versions = []
    publication_write_records, publication_records = [], []
    byte = dk * dv * state_byte
    snapshot_dtype = _PREFILL_SNAPSHOT_DTYPE
    snapshot_byte = dk * dv * int(dtype_bytes(snapshot_dtype))
    span = ((0, dk), (0, dv))
    producer = layout_producer or f"{op_id}:layout_publication"
    for sequence, (target, length, context) in enumerate(
            zip(targets, lengths, contexts)):
        count = (length + chunk - 1) // chunk
        versions = tuple(context + min(step * chunk, length)
                         for step in range(1, count + 1))
        checkpoint_versions.append(versions)
        per_head_versions, per_head_commits = [], []
        public_span = ((0, heads), (0, dv), (0, dk))
        public_bytes = heads * byte
        for head in range(heads):
            internal = internal_targets[sequence][head]
            storage = f"state:{sequence}:head:{head}"
            snapshot_storage = f"h_snapshot:{sequence}:head:{head}"
            # The real initialize conversion reads this head's public view.
            read_records.append({
                "target": target, "version": context, "sequence": sequence,
                "head": head, "storage": f"public_state:{sequence}",
                "dtype": state_dtype, "layout": "V,K",
                "span": ((head, head + 1), (0, dv), (0, dk)),
                "bytes": byte, "action": "read",
                "public_subview": (head, 0, 0)})
            initialize_write_records.append({
                "target": internal, "version": context, "sequence": sequence,
                "head": head, "storage": snapshot_storage, "dtype": snapshot_dtype,
                "layout": "K,V", "span": span, "bytes": snapshot_byte,
                "action": "write",
                "producer": f"{op_id}:initialize",
                "output": (sequence, head)})
            per_head_versions.append(versions)
            per_head_commits.append(versions)
            previous = context
            previous_producer = f"{op_id}:initialize"
            for chunk_index, version in enumerate(versions):
                chunk_producer = (
                    f"{op_id}:state_chunk:{sequence}:{head}:{chunk_index}"
                )
                is_first_chunk = chunk_index == 0
                read = {
                    "target": internal, "version": previous,
                    "sequence": sequence, "head": head,
                    "chunk": chunk_index,
                    "storage": snapshot_storage if is_first_chunk else storage,
                    "dtype": snapshot_dtype if is_first_chunk else state_dtype,
                    "layout": "K,V", "span": span,
                    "bytes": snapshot_byte if is_first_chunk else byte,
                    "action": "read",
                    "producer": previous_producer}
                record = {
                    "target": internal, "version": version,
                    "sequence": sequence, "head": head, "chunk": chunk_index,
                    "storage": storage, "dtype": state_dtype,
                    "layout": "K,V", "span": span, "bytes": byte,
                    "generation": version,
                    "producer": chunk_producer,
                    "output": chunk_index}
                internal_read_records.append(read)
                internal_write_records.append({**record, "action": "write"})
                internal_commit_records.append({**record, "action": "commit"})
                previous = version
                previous_producer = chunk_producer
        internal_checkpoint_versions.append(tuple(per_head_versions))
        internal_commit_versions.append(tuple(per_head_commits))
        final = versions[-1]
        publication_write_records.append({
            "target": target, "version": final, "sequence": sequence,
            "storage": f"public_state:{target}", "dtype": state_dtype,
            "layout": "V,K", "span": public_span, "bytes": public_bytes,
            "head_count": heads, "action": "write",
            "producer": producer, "output": layout_outputs[sequence],
            "generation": final, "ready": producer,
            "layout_conversion": True})
        publication_records.append({
            "target": target, "version": final, "sequence": sequence,
            "storage": f"public_state:{target}", "dtype": state_dtype,
            "layout": "V,K", "span": public_span, "bytes": public_bytes,
            "head_count": heads, "action": "commit",
            "producer": producer, "output": layout_outputs[sequence],
            "generation": final, "ready": producer})
    return {
        "phase": "prefill", "queries": lengths, "contexts": tuple(contexts),
        "version_basis": basis, "targets": targets,
        "heads": heads, "dk": dk, "dv": dv, "state_dtype": state_dtype,
        "internal_targets": internal_targets,
        "layout_producer": producer,
        "layout_dtype": state_dtype,
        "layout_bytes": heads * dk * dv * state_byte,
        "read_records": tuple(read_records),
        "initialize_write_records": tuple(initialize_write_records),
        "internal_read_records": tuple(internal_read_records),
        "internal_write_records": tuple(internal_write_records),
        "internal_commit_records": tuple(internal_commit_records),
        "publication_write_records": tuple(publication_write_records),
        "publication_records": tuple(publication_records),
        "checkpoint_versions": tuple(checkpoint_versions),
        "internal_checkpoint_versions": tuple(internal_checkpoint_versions),
        "internal_commit_versions": tuple(internal_commit_versions),
        "commit_versions": tuple(record["version"]
                                 for record in publication_records),
        "commit_policy": "per_head_chunk_internal_then_layout_publication",
        "candidate_write_point": "per_head_chunk_store_before_next_chunk_read",
        "publication_point": "final_layout_output_write_complete",
    }
def state_value(value, *, reason="kda_state_value_required"):
    """Resolve a caller ``TensorValue`` or its portable record.

    The lowerer accepts a real immutable value directly; a previously exported
    ``value`` mapping is reconstructed from producer/output/layout/storage so a
    next invocation can consume the prior candidate record without inventing a
    name-only alias.
    """
    if isinstance(value, TensorValue):
        return value
    if isinstance(value, Mapping):
        record = value.get("value", value)
        if isinstance(record, TensorValue):
            return record
        if isinstance(record, Mapping) and "producer" in record \
                and "layout" in record:
            layout = record["layout"]
            if isinstance(layout, Mapping):
                layout = TensorLayout(
                    tuple(layout["shape"]), tuple(layout["strides"]),
                    int(layout.get("offset", 0)),
                    tuple(layout.get("storage_shape", ())))
            return TensorValue(
                producer=str(record["producer"]),
                dtype=str(record.get("dtype", "")),
                layout=layout,
                output=record.get("output", 0),
                quant=str(record.get("quant", "")),
                layout_kind=str(record.get("layout_kind", "strided")),
                storage=str(record.get("storage", "")),
                alias_of=str(record.get("alias_of", "")),
                state_version=record.get("state_version"),
                binding=str(record.get("binding", "static")),
                binding_name=str(record.get("binding_name", "")),
            )
    raise ModelingError(reason)


def bind_state_value(value, *, target, owner, slot=None, version=None,
                     ready=None, grant=None, generation=None, decision=None,
                     last_use=None, representation=None, required=True,
                     window=None, span=None, readiness=None):
    """Bind one actual state value to the fields a consumer must dereference.

    This is the single origin for the producer/output, owner, storage,
    generation, slot/window/offset/layout, version and ready/grant used by
    effects, ports, accesses and exports.  It is not a second state IR; the
    value remains the existing ``TensorValue``.
    """
    if not isinstance(target, str) or not target:
        raise ModelingError("kda_state_binding_target_invalid")
    value = state_value(value)
    if slot is not None and (
            type(slot) is not int or slot < 0):
        raise ModelingError("kda_state_binding_slot_invalid")
    if generation is not None and (
            type(generation) is not int or generation < 0):
        raise ModelingError("kda_state_binding_generation_invalid")
    effective_version = (value.state_version if version is None else version)
    if generation is None:
        generation = effective_version
    if (version is not None and value.state_version is not None
            and value.state_version != version):
        raise ModelingError(
            f"kda_state_value_version_mismatch:{target}:"
            f"{value.state_version}:{version}")
    if effective_version is not None and (
            type(effective_version) is not int or effective_version < 0):
        raise ModelingError("kda_state_binding_version_invalid")
    if window is not None:
        window = tuple(int(item) for item in window)
        if len(window) != 2 or window[0] < 0 or window[1] < window[0]:
            raise ModelingError("kda_state_binding_window_invalid")
    if span is not None:
        span = tuple((int(item[0]), int(item[1])) for item in span)
    storage = value.storage or target
    binding = {
        "target": target,
        "value": value,
        "producer": value.producer,
        "output": value.output,
        "storage": storage,
        "alias_of": value.alias_of,
        "dtype": value.dtype,
        "layout": value.layout.to_dict(),
        "layout_kind": value.layout_kind,
        "quant": value.quant,
        "binding": value.binding,
        "binding_name": value.binding_name,
        "offset": value.layout.offset,
        "state_version": effective_version,
        "slot": slot,
        "window": window,
        "span": span,
        "ready": ready,
        "grant": grant,
        "generation": generation,
        "decision": decision,
        "last_use": last_use,
        "representation": representation or value.dtype,
        "required": bool(required),
        "owner": owner,
        "readiness": readiness or (
            "external" if value.binding == "external"
            else "producer_complete"),
    }
    return binding


def binding_value(binding):
    return state_value(binding["value"])


def binding_record(binding, *, include_value=True):
    """Portable producer record: no live object, only its actual identity."""
    record = {key: item for key, item in binding.items() if key != "value"}
    if include_value:
        value = binding.get("value")
        record["value"] = (value.to_dict()
                           if hasattr(value, "to_dict") else value)
    return record


def _per_sequence(value, count, *, name, default=None):
    if value is None or value == () or value == []:
        return (default,) * count
    if isinstance(value, Mapping):
        return tuple(value.get(index, default) for index in range(count))
    if isinstance(value, (tuple, list)):
        values = tuple(value)
        if len(values) == 1 and count > 1:
            return values * count
        if len(values) != count:
            raise ModelingError(f"kda_state_{name}_must_match_sequences")
        return values
    return (value,) * count


def _narrow_value(value, axis, start, length):
    if start < 0 or length < 0 or start + length > value.layout.shape[axis]:
        raise ModelingError("kda_state_value_window_outside_layout")
    return value.narrow(axis, start, length)


def bind_sequence_values(*, targets, lengths, contexts, shape, dtype, owner,
                         values=None, slots=None, storage_prefix="",
                         ready=None, grants=None, generation=None,
                         required=True, representation=None, windows=None,
                         spans=None):
    """Bind per-sequence state values, slots and ready/grant references.

    ``values`` may be one pool tensor (leading capacity or sequence axis), one
    packed sequence tensor, or a per-sequence tuple/mapping.  Absence creates an
    explicit standalone external port with a declared ABI layout; it never
    silently reuses another occurrence's name.
    """
    targets = tuple(targets)
    lengths = tuple(lengths)
    contexts = tuple(contexts)
    count = len(targets)
    if not (len(lengths) == len(contexts) == count):
        raise ModelingError("kda_state_sequence_binding_shape_mismatch")
    shape = tuple(int(item) for item in shape)
    slots = _per_sequence(slots, count, name="slots")
    for slot in slots:
        if slot is not None and (type(slot) is not int or slot < 0):
            raise ModelingError("kda_state_slot_invalid")
    ready = _per_sequence(ready, count, name="ready")
    grants = _per_sequence(grants, count, name="grants")
    generation = _per_sequence(generation, count, name="generation")
    windows = _per_sequence(windows, count, name="windows")
    spans = _per_sequence(spans, count, name="spans")
    required = _per_sequence(required, count, name="required", default=True)
    produced = []
    if values is None:
        produced = [None] * count
    elif isinstance(values, TensorValue):
        layout = tuple(values.layout.shape)
        if layout == shape and count == 1:
            produced = [values]
        elif len(layout) >= 1 and layout[1:] == shape:
            if all(slot is not None for slot in slots):
                produced = [_narrow_value(values, 0, slot, 1) for slot in slots]
            elif layout[0] == count:
                produced = [
                    _narrow_value(values, 0, slot if slot is not None else index, 1)
                    for index, slot in enumerate(slots)
                ]
            else:
                raise ModelingError(
                    f"kda_state_pool_shape_mismatch:{layout}:{shape}")
        else:
            raise ModelingError(
                f"kda_state_pool_shape_mismatch:{layout}:{shape}")
    elif isinstance(values, Mapping):
        produced = [values.get(target, values.get(index))
                    for index, target in enumerate(targets)]
    elif isinstance(values, (tuple, list)):
        produced = list(values)
        if len(produced) != count:
            raise ModelingError("kda_state_values_must_match_sequences")
    else:
        raise ModelingError("kda_state_values_require_tensor_or_sequence")
    bindings = []
    for index, target in enumerate(targets):
        provided = produced[index]
        slot = slots[index]
        if provided is None:
            if slot is None:
                storage = f"{storage_prefix or target}.state"
            else:
                storage = f"{storage_prefix or target}:slot:{slot}"
            provided = TensorValue(
                "external", dtype, TensorLayout.contiguous(shape),
                output=(slot if slot is not None else index),
                storage=storage, binding="external",
                binding_name=f"kda.state[{target}]",
                state_version=contexts[index])
        else:
            provided = state_value(provided)
            if tuple(provided.layout.shape[-len(shape):]) != shape:
                raise ModelingError(
                    f"kda_state_value_shape_mismatch:{target}:"
                    f"{provided.layout.shape}:{shape}")
            if provided.dtype != dtype:
                raise ModelingError(
                    f"kda_state_value_dtype_mismatch:{target}:"
                    f"{provided.dtype}:{dtype}")
        bindings.append(bind_state_value(
            provided, target=target, owner=owner, slot=slot,
            version=contexts[index], ready=ready[index], grant=grants[index],
            generation=generation[index], required=required[index],
            representation=representation, window=windows[index],
            span=spans[index]))
    return tuple(bindings)


def state_ports(*, targets=(), dtype="", owner, direction, readiness,
                bindings=None, optional=None):
    """Typed value ports for a physical read/write fragment.

    A binding-backed port carries the actual producer record in the fragment's
    attrs and is required (``optional=False``) unless the caller explicitly
    declares an absent value.  The name, dtype, owner and readiness still come
    from that same binding; no port is guessed from a target string alone.
    """
    if direction not in {"in", "out", "inout"}:
        raise ModelingError("kda_state_port_direction_invalid")
    if bindings is None:
        specs = tuple(TensorPortSpec(
            name=target, direction=direction, representation=dtype,
            dtype=dtype, layout_kind="strided", scope="layer", owner=owner,
            readiness=readiness, optional=True,
        ) for target in targets)
        return PortSet(
            tensor_in=specs if direction in {"in", "inout"} else (),
            tensor_out=specs if direction == "out" else ())
    specs = []
    seen = set()
    for binding in bindings:
        # Several candidate versions share one logical target; the target
        # remains the single port identity, while the per-version binding is
        # carried by the state effect/attr.
        if binding["target"] in seen:
            continue
        seen.add(binding["target"])
        is_optional = (not binding.get("required", True)
                       if optional is None else optional)
        specs.append(TensorPortSpec(
            name=binding["target"], direction=direction,
            representation=binding.get("representation") or binding["dtype"],
            dtype=binding["dtype"],
            layout_kind=binding.get("layout_kind", "strided"),
            scope="layer", owner=binding.get("owner", owner),
            readiness=binding.get("readiness", readiness),
            optional=is_optional,
        ))
    return PortSet(
        tensor_in=tuple(specs) if direction in {"in", "inout"} else (),
        tensor_out=tuple(specs) if direction == "out" else ())


def decision_ports(*, target, version, owner, action="commit", scope="request"):
    """Typed event input naming the actual decision boundary (not a timer)."""
    if action not in {"commit", "rollback"}:
        raise ModelingError("kda_state_decision_action_invalid")
    if type(version) is not int or version < 0:
        raise ModelingError("kda_state_decision_version_invalid")
    return PortSet(event_in=(EventPortSpec(
        name=f"decision:{target}:{version}",
        direction="in", event=("commit" if action == "commit" else "release"),
        scope=scope, owner=owner, readiness="external",
        generation=version, optional=False),))
