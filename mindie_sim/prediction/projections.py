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
from dataclasses import replace

from mindie_sim.catalog.ops import (
    compute_dtype_for,
    matrix_output_format,
    precision_abi_for_gemm,
)
from mindie_sim.schema.operators import TensorPrecisionABI

_REPLICATED_SHARD = "replicated"
_SINGLE_PACKING = "shared_input_output_slices"
_GROUPED_PACKING = "compatible_group_shared_input_output_slices"
_REPLICATED_MEMO_LIMIT = 4096
_REPLICATED_MEMO: dict[tuple[int, int], tuple[object, object, frozenset]] = {}
_FLOAT_FORMATS = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
_MX_GROUP_SIZE = 32
_MX_SCALE_FORMATS = frozenset({"fp8_e8m0", "ue8m0"})
_MXFP4_WEIGHT_FORMATS = frozenset({"mxfp4", "mx_fp4", "mxfp4_e2m1", "fp4", "fp4_e2m1"})
_MXFP8_FORMATS = frozenset({"mxfp8", "mx_fp8", "mxfp8_e4m3", "fp8_e4m3", "fp8_e5m2"})


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


def effective_linear_method(quant, category) -> str:
    """Resolve the effective linear method after role overrides and exclusions."""
    q = quant.for_category(category)
    wf = str(q.format_for(category) or "").lower().replace("-", "_")
    _, act_raw = compute_dtype_for(q, category)
    af = str(act_raw or "").lower().replace("-", "_")
    if wf in _FLOAT_FORMATS and af in _FLOAT_FORMATS:
        return "none"
    sf = str(q.execution_weight_scale_format() or q.scale_format or "").lower().replace("-", "_")
    block = q.weight_block
    group = int(block[1] if isinstance(block, (tuple, list)) and len(block) > 1 else 0)
    mx_scale = (
        sf in _MX_SCALE_FORMATS
        and group == _MX_GROUP_SIZE
        and int(q.activation_group_size or 0) == _MX_GROUP_SIZE
    )
    if mx_scale and wf in _MXFP4_WEIGHT_FORMATS:
        return "mx_fp4"
    if mx_scale and wf in _MXFP8_FORMATS and af in _MXFP8_FORMATS:
        return "mx_fp8"
    if wf in {"int4", "w4a8", "w4a16", "int4_weight"}:
        return "w4a8" if af == "int8" else "w4a16"
    if wf in {"int8", "w8a8"} or af == "int8":
        return "w8a8" if (wf == "w8a8" or af == "int8") else "w8a16"
    if wf in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        return "fp8"
    return "none"


def precision_abi_for(quant, category, *, prequant: str = "") -> TensorPrecisionABI:
    """Build canonical TensorPrecisionABI for one matrix role."""
    return precision_abi_for_gemm(quant, category, prequant_dtype=prequant)


def _representation(quant, category):
    q = quant.for_category(category)
    abi = precision_abi_for_gemm(q, category)
    if abi.linear_method == "dense":
        return (
            abi.linear_method, abi.input_dtype, abi.weight_dtype,
            abi.output_dtype, abi.compute_dtype, abi.accum_dtype,
        )
    return (
        abi.linear_method,
        abi.input_dtype,
        abi.weight_dtype,
        abi.output_dtype,
        abi.compute_dtype,
        abi.accum_dtype,
        abi.weight_layout,
        abi.weight_block,
        tuple(q.weight_block),
        int(q.expert_block),
        abi.weight_scale_dtype,
        float(q.execution_weight_scale_bytes() or q.scale_bytes or 0.0),
        str(q.execution_int8_weight_metadata() or ""),
        abi.act_group_size,
        abi.act_scale_dtype,
        abi.act_scale_pack,
    )


def _attach_precision_abi(b, op_id: str, category: str) -> str:
    if not b.ops or b.ops[-1].op_id != op_id:
        return op_id
    op = b.ops[-1]
    abi = op.precision_abi
    if abi is None:
        abi = precision_abi_for(b.quant, category, prequant=getattr(b.model, "activation_dtype", ""))
    meta = dict(op.metadata)
    if (op.precision_abi is abi
            and meta.get("linear_method") == abi.linear_method
            and meta.get("precision_abi_signature") == abi.abi_signature):
        return op_id
    meta["linear_method"] = abi.linear_method
    meta["precision_abi_signature"] = abi.abi_signature
    b.ops[-1] = replace(op, precision_abi=abi, metadata=tuple(sorted(meta.items())))
    return op_id


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
    op_id = b.matrix(
        name, (deps,), tokens, width, hidden, category=group[0][3],
        metadata=(("projection_packing", packing),
                  ("packed_projection_slices", tuple(slices)),
                  ("packed_projection_group_roles",
                   ("+".join(spec[0] for spec in group),))))
    return _attach_precision_abi(b, op_id, group[0][3])


def input_projections(b, *, name, deps, tokens, hidden, roles, packed):
    """Return role->producer; ``roles`` are (role, recipe, width, category).

    A split is a strided view of the common output. Consumers retain their own
    payload reads; no copy is inferred simply because the row stride grows.
    When the declared operand representations differ, the roles are grouped by
    identical representation and every multi-role group shares one matrix.
    """
    specs = tuple(_spec(item) for item in roles)
    if not packed:
        return {
            spec[0]: _attach_precision_abi(
                b,
                b.matrix(spec[1], (deps,), tokens, spec[2], hidden, category=spec[3]),
                spec[3],
            )
            for spec in specs
        }
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
            op_id = b.matrix(
                spec[1], (deps,), tokens, spec[2], hidden, category=spec[3],
                metadata=(("projection_packing", packing),))
            producers[spec[0]] = _attach_precision_abi(b, op_id, spec[3])
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


def gdn_input_projections(b, tensors, *, deps, tokens, hidden, conv_width, value_width,
                          heads, metadata=()):
    """Emit GDN QKV+Z and beta+alpha projections under homogeneous ABI packing.

    When QKV and Z (or beta and alpha) share identical effective representation,
    keep the single packed ``linear_qkvz`` / ``linear_beta_alpha`` GEMM and
    return strided slices.  When explicit role overrides separate precision
    (e.g. quantized QKV with BF16 Z), split into homogeneous matrices instead
    of mixing precisions in one GEMM.
    """
    explicit_keys = (
        {str(key) for key, _ in getattr(b.quant, "category_overrides", ())}
        | set(getattr(b.quant, "exclusions", ()))
    )
    qkvz_child_declared = bool({"gdn.qkv", "gdn.z"} & explicit_keys)
    qkvz_rep = _representation(b.quant, "gdn.qkvz")
    qkv_rep = _representation(b.quant, "gdn.qkv") if qkvz_child_declared else qkvz_rep
    z_rep = _representation(b.quant, "gdn.z") if qkvz_child_declared else qkvz_rep
    if not qkvz_child_declared or (qkv_rep == z_rep == qkvz_rep):
        qkvz = b.matrix(
            "linear_qkvz", (deps,), tokens, conv_width + value_width, hidden,
            category="gdn.qkvz", metadata=metadata,
        )
        _attach_precision_abi(b, qkvz, "gdn.qkvz")
        proj = tensors.result(qkvz, (tokens, conv_width + value_width))
        conv_view = proj.narrow(1, 0, conv_width)
        z_view = proj.narrow(1, conv_width, value_width)
    else:
        qkv = b.matrix(
            "linear_qkv", (deps,), tokens, conv_width, hidden,
            category="gdn.qkv", metadata=metadata,
        )
        _attach_precision_abi(b, qkv, "gdn.qkv")
        z = b.matrix(
            "linear_z", (deps,), tokens, value_width, hidden,
            category="gdn.z", metadata=metadata,
        )
        _attach_precision_abi(b, z, "gdn.z")
        conv_view = tensors.result(qkv, (tokens, conv_width))
        z_view = tensors.result(z, (tokens, value_width))

    ba_child_declared = bool({"gdn.b", "gdn.a"} & explicit_keys)
    ba_rep = _representation(b.quant, "gdn.beta_alpha")
    b_rep = _representation(b.quant, "gdn.b") if ba_child_declared else ba_rep
    a_rep = _representation(b.quant, "gdn.a") if ba_child_declared else ba_rep
    if not ba_child_declared or (b_rep == a_rep == ba_rep):
        gates = b.matrix(
            "linear_beta_alpha", (deps,), tokens, 2 * heads, hidden,
            category="gdn.beta_alpha", metadata=metadata,
        )
        _attach_precision_abi(b, gates, "gdn.beta_alpha")
        gate_proj = tensors.result(gates, (tokens, 2 * heads))
        beta_view = gate_proj.narrow(1, 0, heads)
        alpha_view = gate_proj.narrow(1, heads, heads)
    else:
        beta_op = b.matrix(
            "linear_beta", (deps,), tokens, heads, hidden,
            category="gdn.b", metadata=metadata,
        )
        _attach_precision_abi(b, beta_op, "gdn.b")
        alpha_op = b.matrix(
            "linear_alpha", (deps,), tokens, heads, hidden,
            category="gdn.a", metadata=metadata,
        )
        _attach_precision_abi(b, alpha_op, "gdn.a")
        beta_view = tensors.result(beta_op, (tokens, heads))
        alpha_view = tensors.result(alpha_op, (tokens, heads))
    return conv_view, z_view, (beta_view, alpha_view)
