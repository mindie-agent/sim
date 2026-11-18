"""Packed input projections and low-rank gates from structural contracts.

[Role] Compose shared-input matrices while preserving typed output slices.
[Invariants] Packing removes repeated input reads and invocations, not weights
  or arithmetic. Different explicit operand representations remain separate
  unless they declare the same representation, in which case one shared matrix
  carries the grouped output with strided consumer views and no extra copies.
  Replicated low-rank outputs retain their full width under tensor parallelism.
[Timeouts & Locks] O(number of projection roles); no locks or device work.
[Side-Effect Policy] Appends operators to the caller's Builder only.
"""
from mindie_sim.catalog.ops import compute_dtype_for
from dataclasses import replace

_REPLICATED_SHARD = "replicated"
_SINGLE_PACKING = "shared_input_output_slices"
_GROUPED_PACKING = "compatible_group_shared_input_output_slices"
_REPLICATED_MEMO_LIMIT = 4096
_REPLICATED_MEMO: dict[tuple[int, int], tuple[object, object, frozenset]] = {}


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


def _spec(item):
    """One declared role as ``(role, recipe, size, category, replicated)``."""
    role, recipe, size, category = item[:4]
    replicated = bool(item[4]) if len(item) > 4 else False
    return role, recipe, int(size), category, replicated


def _compatible_groups(quant, specs):
    """Group roles by identical operand representation, keeping role order."""
    order, buckets = [], {}
    for spec in specs:
        key = _representation(quant, spec[3])
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(spec)
    return [tuple(buckets[key]) for key in order]


def _packed_matrix(b, name, deps, tokens, hidden, group, packing):
    width = sum(spec[2] for spec in group)
    offset, slices = 0, []
    for role, recipe, size, category, replicated in group:
        slices.append({"role": role, "recipe": recipe, "category": category,
                       "shape": (tokens, size), "strides": (width, 1),
                       "offset_elements": offset, "replicated": replicated})
        offset += size
    return b.matrix(
        name, (deps,), tokens, width, hidden, category=group[0][3],
        metadata=(("projection_packing", packing),
                  ("packed_projection_slices", tuple(slices)),
                  ("packed_projection_group_roles",
                   ("+".join(spec[0] for spec in group),))))


def input_projections(b, *, name, deps, tokens, hidden, roles, packed):
    """Return role->producer; ``roles`` are (role, recipe, width, category).

    A split is a strided view of the common output. Consumers retain their own
    payload reads; no copy is inferred simply because the row stride grows.
    When the declared operand representations differ, the roles are grouped by
    identical representation and every multi-role group shares one matrix.
    """
    specs = tuple(_spec(item) for item in roles)
    if not packed:
        return {spec[0]: b.matrix(spec[1], (deps,), tokens, spec[2], hidden,
                                     category=spec[3])
                for spec in specs}
    groups = _compatible_groups(b.quant, specs)
    if len(groups) == 1:
        producer = _packed_matrix(b, name, deps, tokens, hidden, groups[0],
                                  _SINGLE_PACKING)
        return {spec[0]: producer for spec in specs}
    packing = _GROUPED_PACKING
    producers = {}
    for group in groups:
        if len(group) == 1:
            spec = group[0]
            producers[spec[0]] = b.matrix(
                spec[1], (deps,), tokens, spec[2], hidden, category=spec[3],
                metadata=(("projection_packing", packing),))
            continue
        producer = _packed_matrix(
            b, name + "." + "".join(spec[0] for spec in group),
            deps, tokens, hidden, group, packing)
        producers.update({spec[0]: producer for spec in group})
    return producers


def replicated_categories(b):
    """Catalog categories whose weight term is replicated under local TP."""
    model = getattr(b, "model", None)
    plan = getattr(b, "plan", None)
    if model is None or plan is None:
        return frozenset()
    key = (id(model), id(plan))
    hit = _REPLICATED_MEMO.get(key)
    if hit is not None and hit[0] is model and hit[1] is plan:
        return hit[2]
    from mindie_sim.catalog.weights import layer_weight_terms

    value = frozenset(
        str(term.category) for term in layer_weight_terms(plan, model)
        if str(getattr(term, "shard", "")) == _REPLICATED_SHARD)
    if len(_REPLICATED_MEMO) >= _REPLICATED_MEMO_LIMIT:
        _REPLICATED_MEMO.clear()
    _REPLICATED_MEMO[key] = (model, plan, value)
    return value


def kda_projections(b, *, plan, tokens, hidden, key_width, value_width,
                    heads, dk, deps):
    """Return ``(qkv_producers, output_gate, raw_beta, decay)`` for KDA.

    Beta activation belongs to the selected recurrent caller, not this linear
    projection boundary. Decay is per state head/key channel, independent of
    value width; a low-rank output gate is per value channel.  The low-rank
    ``f_a``/``g_a`` factors are replicated on every owner rank, so their packed
    slices keep the declared width while Q/K/V/B split output features.
    """
    replicated = replicated_categories(b)
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
    tagged = tuple((*role, str(role[3]) in replicated) for role in roles)
    projected = input_projections(
        b, name="linear_qkvbfg_a", deps=deps, tokens=tokens, hidden=hidden,
        roles=tagged, packed=plan.packed_input_projection)
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
