"""Explicit previous-state selection for the retained recurrent KDA ABI.

[Role] Resolve exact state-slot addresses or retain caller-declared validity.
[Invariants] Acceptance selects the initial checkpoint only: every current
  query executes and writes its checkpoint. State pools of independent
  sequences cannot alias. Metadata capacities never supply missing values.
[Timeouts & Locks] O(batch * query rows), q <= 8 in the consumer; no locks.
[Side-Effect Policy] Pure contract validation and immutable tuple results.
"""
from itertools import accumulate

from mindie_sim.schema.common import ModelingError


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
