"""Packed input projections and low-rank gates from structural contracts.

[Role] Compose shared-input matrices while preserving typed output slices.
[Invariants] Packing removes repeated input reads and invocations, not weights
  or arithmetic. Different explicit operand representations remain separate.
  Replicated low-rank outputs retain their full width under tensor parallelism.
[Timeouts & Locks] O(number of projection roles); no locks or device work.
[Side-Effect Policy] Appends operators to the caller's Builder only.
"""
from mindie_sim.catalog.ops import compute_dtype_for
from dataclasses import replace


def require_packed_slice(b, tensors, producer, roles, consumer, argument, name):
    """Materialize only a declared consumer requirement on a known packed view."""
    op = next(op for op in b.ops if op.op_id == producer)
    slices = dict(op.metadata).get("packed_projection_slices", ())
    selected = [item for item in slices if item["role"] in roles]
    if not selected:
        return producer
    start = selected[0]["offset_elements"]
    width = sum(item["shape"][1] for item in selected)
    offset = start
    for item in selected:
        if item["offset_elements"] != offset:
            raise ValueError("packed_projection_slice_requires_adjacent_source_roles")
        offset += item["shape"][1]
    shape = op.local_shape[:2]
    value = tensors.result(producer, shape).narrow(1, start, width)
    result = tensors.require(value, consumer, argument, name=name)
    if result.producer != producer:
        index = next(i for i, candidate in enumerate(b.ops) if candidate.op_id == result.producer)
        materialized = b.ops[index]
        b.ops[index] = replace(materialized, metadata=tuple({**dict(materialized.metadata),
            "native_kind": "Slice", "packed_input_shape": shape,
            "packed_source_roles": tuple(roles),
            "slice_offsets": (0, start), "slice_sizes": (shape[0], width),
            "contract_basis": "declared_contiguous_consumer_of_source_packed_roles"}.items()))
    return result.producer


def _representation(quant, category):
    q = quant.for_category(category)
    return (q.format_for(category), *compute_dtype_for(q, category),
            q.weight_block, q.expert_block, q.scale_format, q.scale_bytes,
            q.activation_group_size, q.activation_scale_format, q.activation_scale_pack)


def input_projections(b, *, name, deps, tokens, hidden, roles, packed):
    """Return role->producer; ``roles`` are (role, recipe, width, category).

    A split is a strided view of the common output. Consumers retain their own
    payload reads; no copy is inferred simply because the row stride grows.
    """
    compatible = len({_representation(b.quant, category)
                      for _, _, _, category in roles}) == 1
    if not packed or not compatible:
        metadata = (("projection_packing", "separate_explicit_operand_formats"),) if packed else ()
        return {role: b.matrix(recipe, (deps,), tokens, width, hidden,
                               category=category, metadata=metadata)
                for role, recipe, width, category in roles}
    width = sum(size for _, _, size, _ in roles)
    offset, slices = 0, []
    for role, recipe, size, category in roles:
        slices.append({"role": role, "recipe": recipe, "category": category,
                       "shape": (tokens, size), "strides": (width, 1),
                       "offset_elements": offset})
        offset += size
    producer = b.matrix(
        name, (deps,), tokens, width, hidden, category=roles[0][3],
        metadata=(("projection_packing", "shared_input_output_slices"),
                  ("packed_projection_slices", tuple(slices))))
    return {role: producer for role, _, _, _ in roles}


def kda_projections(b, *, plan, tokens, hidden, key_width, value_width,
                    heads, dk, deps):
    """Return ``(qkv_producers, output_gate, raw_beta, decay)`` for KDA.

    Beta activation belongs to the selected recurrent caller, not this linear
    projection boundary. Decay is per state head/key channel, independent of
    value width; a low-rank output gate is per value channel.
    """
    gate_width = plan.gate_rank or value_width
    gate_role = "g_a" if plan.gate_rank else "g"
    roles = (
        ("q", "linear_q", key_width, "kda.q"),
        ("k", "linear_k", key_width, "kda.k"),
        ("v", "linear_v", value_width, "kda.v"),
        ("b", "linear_beta", heads, "kda.b"),
        ("f_a", "linear_decay_a", dk, "kda.f_a"),
        (gate_role, "linear_gate_a" if plan.gate_rank else "linear_gate",
         gate_width, "kda." + gate_role),
    )
    projected = input_projections(
        b, name="linear_qkvbfg_a", deps=deps, tokens=tokens, hidden=hidden,
        roles=roles, packed=plan.packed_input_projection)
    gate = projected[gate_role]
    if plan.gate_rank:
        gate = b.matrix("linear_gate_b", (gate,), tokens, value_width,
                        plan.gate_rank, category="kda.g_b")
    decay = b.matrix("linear_decay_b", (projected["f_a"],), tokens,
                     heads * dk, dk, category="kda.f_b")
    return [projected[key] for key in ("q", "k", "v")], gate, projected["b"], decay


def mla_input_projections(b, *, plan, tokens, hidden, kv_width, deps, packed=True):
    return input_projections(
        b, name="qkv_a", deps=deps, tokens=tokens, hidden=hidden,
        roles=(("q_a", "q_a", plan.q_lora_rank, "mla.q_a"),
               ("kv_a", "kv_a", kv_width, "mla.kv_a")),
        packed=packed and plan.packed_input_projection)
