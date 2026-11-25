"""Read-only typed projection of the selected canonical OpSpec tuple.

[Role] Turn declared operator fields into one typed graph without inventing a
  tensor, dtype, weight or state fact that the selected OpSpec did not declare.
[Invariants] A source-planner/native operator is opaque; a control dependency
  never becomes a tensor; an absent dtype/shape/storage fact is ineligible;
  live-out roots are operator ids not consumed by any other operator.  Data
  inputs are emitted from the complete global positional operand sequence in
  exact index order through the shared control classifier, never regrouped by
  producer and never promoting an implicit event to a tensor. Every internal
  input, including resolved state, requires scheduled order across streams/stages.
[Timeouts & Locks] Finite graph construction; no I/O, clocks or locks.
[Side-Effect Policy] Pure construction over immutable records.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import Any, Mapping

from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.fusion.graph import (
    FeasibilityError, NATIVE_AXIS_LAYOUTS, OpNode, TensorValue, TypedGraph, dtype_bytes,
)
from mindie_sim.prediction.fusion.ports import (
    control_dependency_names, declared_control_names, full_operand_ports,
    input_port_entries, projected_ports,
)

_STATE_SCALAR_BYTES = 4
_PACKED_LINEAR_LAYOUTS = frozenset({
    "bf16", "bfloat16", "fp16", "float16", "fp32", "float32",
    "int8_per_channel", "int4_packed_per_channel", "fp8_rowwise",
})
_PACKED_BLOCK_LAYOUT_RE = re.compile(
    r"^(?:fp8_block\d+x\d+_[a-z0-9_]+|(?:fp4|int4)_group\d+_packed)$"
)


def _normalize_tensor_layout(raw: str, *, op_id: str, allow_strided: bool = False) -> str:
    """Map canonical physical or catalog-packed weight/op layouts to ``TensorValue.layout``."""
    text = str(raw or "").strip().lower()
    if text in {"nz", "fractal_nz"} or "fractal_nz" in text:
        return "nz"
    if text in {"contiguous", "nd"} or text in NATIVE_AXIS_LAYOUTS:
        return text
    if allow_strided and text in {"strided", "view"}:
        return text
    if text in _PACKED_LINEAR_LAYOUTS or _PACKED_BLOCK_LAYOUT_RE.match(text):
        return "contiguous"
    kind = "output" if allow_strided else "weight"
    raise FeasibilityError(f"projection_{kind}_layout_unsupported:{op_id}:{text}")

_FAMILY_KIND = {
    "rmsnorm": "rms_norm", "layernorm": "rms_norm", "norm": "rms_norm",
    "add_rmsnorm": "add_rms_norm",
    "rmsnorm_quant": "rms_norm_quant",
    "add_rmsnorm_quant": "add_rms_norm_quant",
    "add": "add", "residual_add": "add", "elementwise": "elementwise",
    "quantize": "dynamic_quant", "quant": "dynamic_quant",
    "dequantize": "cast", "cast": "cast", "tensor_copy": "cast",
    "gemm": "matmul", "gate_up_projection": "matmul",
    "down_projection": "matmul", "shared_expert": "matmul",
    "grouped_gemm": "grouped_matmul", "swiglu": "swiglu",
    "silu": "silu", "situ": "situ", "gelu": "gelu", "sigmoid": "sigmoid",
    "softmax": "softmax", "l2_norm": "l2_norm", "l2norm": "l2_norm",
    "rotary": "rope", "topk": "topk", "attention": "attention",
    "indexer": "attention", "native_fusion": "matmul",
}
_NORM_KINDS = frozenset({"rms_norm", "add_rms_norm", "rms_norm_quant",
                         "add_rms_norm_quant"})
_QUANT_KINDS = frozenset({"dynamic_quant", "quant", "rms_norm_quant",
                          "add_rms_norm_quant"})
_CUBE_FAMILIES = frozenset({"gemm", "gate_up_projection", "down_projection",
                            "shared_expert", "grouped_gemm", "attention",
                            "indexer", "native_fusion"})
_STATE_FAMILIES = frozenset({"recurrent_vector", "short_convolution",
                             "cache_store", "kv_compress_epilog",
                             "indexer_quant_cache", "state_update",
                             "kda_recurrent", "gdn_decode", "recurrent_scan"})


def _metadata(op: OpSpec) -> dict:
    """Last-value lookup for scalar facts only; declarations use raw pairs."""
    return dict(op.metadata)


def _scheduled_before(producer: str, op: OpSpec, by_name: Mapping) -> bool:
    """Reach an internal producer or external input through scheduling edges."""
    pending, seen = list(op.deps), set()
    while pending:
        dep = pending.pop()
        if dep == producer:
            return True
        if dep not in seen:
            seen.add(dep)
            parent = by_name.get(dep)
            if parent is not None:
                pending.extend(parent.deps)
    return False


def execution_prerequisites(ops: tuple[OpSpec, ...], graph: TypedGraph
                            ) -> tuple[OpSpec, ...]:
    """Lower a validated projection to transient readiness, preserving callers.

    Static weights and unbound initial state are not external-ready events.
    Only declared source values bind those events. Previously lowered ancestors
    and other required producers can already guarantee a prerequisite.
    """
    by_name = {op.op_id: op for op in ops}
    sources = {f"{name}:source": name for name in graph.external_inputs}
    earlier, lowered = set(), []
    for op in ops:
        values = tuple(graph.values[key] for key in graph.ops[op.op_id].inputs)
        required = tuple(dict.fromkeys((
            *(dep for dep, _port in full_operand_ports(op, by_name) if dep in earlier),
            *(value.producer if value.producer is not None else sources.get(value.value_id)
              for value in values))))
        missing = tuple(dep for dep in required if dep is not None
                        and not _scheduled_before(dep, op, by_name))
        missing = tuple(dep for dep in missing if not any(
            other != dep and other in earlier
            and _scheduled_before(dep, by_name[other], by_name) for other in missing))
        transient = replace(op, deps=(*op.deps, *missing)) if missing else op
        lowered.append(transient)
        by_name[op.op_id] = transient
        earlier.add(op.op_id)
    return tuple(lowered)


def _family_kind(family: str) -> str:
    if family in _FAMILY_KIND:
        return _FAMILY_KIND[family]
    if family.startswith("kda"):
        return "state_update"
    if family in {"allreduce", "allgather", "reduce_scatter", "dispatch", "combine"}:
        return "collective"
    return family or "unknown"


def _normalize(shape: tuple[int, ...], meta: Mapping) -> tuple[int, ...]:
    rows = int(meta.get("graph_active_query_rows", 0) or 0)
    if len(shape) == 1 and rows > 1 and shape[0] % rows == 0:
        return (rows, shape[0] // rows)
    return shape


def _shapes(op: OpSpec, kind: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    meta = _metadata(op)
    shape = tuple(int(dim) for dim in op.local_shape) or (1,)
    if kind in {"matmul", "grouped_matmul"} and len(shape) == 3:
        return ((shape[0], shape[2]), (shape[0], shape[1]))
    shape = _normalize(shape, meta)
    if kind == "swiglu" and len(shape) >= 2:
        return (shape[0], shape[-1] * 2), shape
    return shape, shape


def _activation_bytes(op: OpSpec, meta: dict) -> int:
    """Declared bytes of one external source value.

    ``read_bytes`` is the sum over *all* operator inputs, so using it for each
    source double-counts a multi-input operator.  Prefer a per-input fact, then
    the declared local shape.
    """
    for key in ("input_bytes", "activation_bytes"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    shape, _ = _shapes(op, _family_kind(op.family))
    if not op.dtype:
        raise FeasibilityError(f"projection_activation_dtype_missing:{op.op_id}")
    return max(0, int(math.ceil(int(math.prod(shape)) * dtype_bytes(op.dtype))))


def _declared_state(meta: dict) -> tuple[dict[str, int | None], dict[str, int | None]]:
    """Read/write state targets and their declared versions; never invented.

    A version the operator did not declare stays ``None``.  Defaulting a
    missing write version to ``0`` would fabricate a state transition (``0 ->
    1``) and could make an opaque access look ordered.
    """
    reads: dict[str, int | None] = {}
    writes: dict[str, int | None] = {}
    effects = meta.get("gdn_state_effects")
    if isinstance(effects, (tuple, list)) and effects:
        for effect in effects:
            if not isinstance(effect, Mapping):
                continue
            target = str(effect.get("target") or "")
            if not target:
                continue
            raw = effect.get("version")
            version = int(raw) if isinstance(raw, (int, float)) else None
            action = str(effect.get("action") or "write")
            if action in {"read", "read_write"}:
                previous = reads.get(target)
                reads[target] = (version if previous is None or version is None
                                 else min(previous, version))
            if action in {"write", "read_write", "write_candidate"}:
                previous = writes.get(target)
                writes[target] = (version if previous is None or version is None
                                  else max(previous, version))
        return reads, writes
    contract = meta.get("gdn_state_contract")
    if isinstance(contract, Mapping):
        targets = contract.get("targets") or ()
        versions = contract.get("write_versions") or contract.get("commit_versions") or ()
        for index, target in enumerate(targets):
            version = None
            if index < len(versions) and versions[index] is not None:
                version = int(versions[index])
            previous = writes.get(str(target))
            writes[str(target)] = (version if previous is None or version is None
                                   else max(previous, version))
    return reads, writes


def _state_target(op: OpSpec, meta: dict) -> str:
    contract = meta.get("gdn_state_contract")
    if isinstance(contract, Mapping):
        targets = contract.get("targets")
        if targets:
            return str(targets[0])
    return f"{op.op_id}:state"


def _resolve_weight_dtype(op: OpSpec, meta: Mapping, fallback_dtype: str = "") -> str:
    """Resolve one canonical physical weight dtype from metadata or typed precision ABI."""
    abi = getattr(op, "precision_abi", None)
    weight_dt = str(
        meta.get("weight_format")
        or meta.get("weight_dtype")
        or (abi.weight_dtype if abi is not None else fallback_dtype)
        or ""
    ).strip()
    if not weight_dt:
        raise FeasibilityError(f"projection_weight_dtype_missing:{op.op_id}")
    return weight_dt


def _weight_identity(op: OpSpec, kind: str, in_shape: tuple[int, ...],
                     out_shape: tuple[int, ...], meta: Mapping) -> tuple[int, str]:
    """Local static weight bytes from declared identity, never ``hbm - read``.

    ``OpSpec.local_shape`` is already the rank-local geometry, so a weight
    derived from it is already local and must never be divided by ``TP`` a
    second time.  Only a value explicitly marked global is sharded here.
    """
    explicit = meta.get("weight_bytes")
    packed = meta.get("packed_weight_bytes")
    sharded = bool(meta.get("weight_sharded", False))
    global_marker = bool(meta.get("weight_global") or meta.get("global_weight")
                         or meta.get("weight_shape_global")
                         or meta.get("global_weight_shape")
                         or meta.get("weight_unsharded_global"))
    if explicit is not None:
        base, basis = int(explicit), "declared_weight_bytes"
    elif packed is not None:
        base, basis = int(packed), "packed_weight_bytes"
    elif kind in {"matmul", "grouped_matmul"} and len(in_shape) >= 2 \
            and len(out_shape) >= 2 and op.family != "native_fusion":
        weight_dt = _resolve_weight_dtype(op, meta, op.dtype or "")
        base = int(math.ceil(int(in_shape[-1]) * int(out_shape[-1]) * dtype_bytes(weight_dt)))
        basis = "shape_k_times_n"
    else:
        return 0, "no_static_weight"
    scale = sum(int(meta.get(key, 0) or 0) for key in
                ("weight_scale_bytes", "row_scale_bytes", "weight_zero_point_bytes",
                 "activation_scale_bytes"))
    base += max(0, scale)
    # ``local_shape``-derived weights are local by construction.  Declared
    # bytes are also treated as the selected physical identity unless the
    # caller explicitly marks them global.
    divide = global_marker and not sharded and basis != "shape_k_times_n"
    local = base if not divide else int(math.ceil(base / max(
        1, int(meta.get("tensor_parallel_hint", 1)))))
    return max(0, local), basis


@dataclass(frozen=True)
class Projection:
    """Extra per-operator facts that the typed graph must not invent."""
    weights: Mapping[str, int] = field(default_factory=dict)
    numerical_policy: Mapping[str, str] = field(default_factory=dict)
    requested: Mapping[str, str] = field(default_factory=dict)
    group_sizes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


def _opnode(op: OpSpec, dep_ids: tuple[str, ...], inputs: tuple[str, ...],
            outputs: tuple[str, ...], kind: str, resource: str,
            state_effect: str, *, semantic_complete: bool, opaque: bool,
            shape_mismatch: bool = False,
            unversioned_state: bool = False) -> OpNode:
    meta = _metadata(op)
    work: dict[str, float] = {}
    if op.hbm_bytes:
        work["hbm"] = float(op.hbm_bytes)
    matrix = dict(op.matrix_flops or ())
    declared_matrix = meta.get("matrix_flops")
    if isinstance(declared_matrix, Mapping) and declared_matrix:
        for dtype, amount in declared_matrix.items():
            matrix[str(dtype)] = matrix.get(str(dtype), 0.0) + float(amount)
    if op.flops:
        work["cube"] = float(op.flops) if resource == "cube" else 0.0
    elif matrix and resource == "cube":
        work["cube"] = float(sum(matrix.values()))
    if op.vector_ops:
        work["vector"] = float(op.vector_ops)
    if op.mte2_bytes:
        work["mte2"] = float(op.mte2_bytes)
    if op.mte3_bytes:
        work["mte3"] = float(op.mte3_bytes)
    if op.link_bytes:
        work["link"] = float(op.link_bytes)
    declared_work = meta.get("resource_work")
    if isinstance(declared_work, Mapping):
        # ``CostModel.price`` treats metadata resource work as *additional* to
        # the OpSpec fields.  Projection must conserve the same total.
        for role, amount in declared_work.items():
            if isinstance(amount, (int, float)):
                work[str(role)] = work.get(str(role), 0.0) + float(amount)
    # Dedicated metadata fields are additive OpSpec work exactly as
    # ``CostModel.price`` consumes them; dropping them deletes executed MTE1
    # or FixPipe bytes during a fusion rewrite.
    for role, metadata_field in (("mte1", "mte1_bytes"),
                                 ("fixpipe", "fixpipe_bytes")):
        amount = meta.get(metadata_field)
        if isinstance(amount, (int, float)) and math.isfinite(float(amount)) \
                and float(amount) > 0:
            work[role] = work.get(role, 0.0) + float(amount)
    reduction_order = meta.get("reduction_order") or (
        "declared_width" if meta.get("reduction_width") else None)
    params: list[tuple[str, Any]] = [
        ("epsilon", meta.get("epsilon", meta.get("norm_eps"))
         if kind in _NORM_KINDS else None),
        ("axis", meta.get("quant_axis", meta.get("axis"))
         if kind in _QUANT_KINDS else None),
        ("group_size", meta.get("quant_group_size",
                                meta.get("group_size", meta.get("activation_group_size")))
         if kind in _QUANT_KINDS else None),
        ("rounding", meta.get("rounding")
         if kind in (_QUANT_KINDS | {"cast", "kv_cast"}) else None),
        ("saturation", meta.get("saturation")
         if kind in _QUANT_KINDS else None),
        ("rotate_style", meta.get("rotate_style") if kind == "rope" else None),
        ("target_dtype", op.dtype if kind in {"cast", "kv_cast"} else None),
        ("reduction_order", reduction_order
         if kind in (_NORM_KINDS | {"softmax"}) else None),
        ("state_key", _state_target(op, meta) if state_effect else None),
    ]
    native_norm_kind = meta.get("native_norm_kind")
    if native_norm_kind:
        params.append(("native_norm_kind", str(native_norm_kind)))
    fp32_tiles = meta.get("native_norm_fp32_working_tiles")
    if isinstance(fp32_tiles, int) and not isinstance(fp32_tiles, bool) and fp32_tiles >= 1:
        params.append(("fp32_working_tiles", fp32_tiles))
    uses_simt = bool(
        meta.get("uses_simt")
        or meta.get("tile_uses_simt")
        or meta.get("template_param_uses_simt")
    )
    if uses_simt:
        params.append(("uses_simt", True))
        if meta.get("tile_uses_simt"):
            params.append(("tile_uses_simt", True))
    if matrix:
        params.append(("matrix_flops", dict(sorted(matrix.items()))))
    group = meta.get("group_sizes") or meta.get("expert_rows")
    if group is not None:
        params.append(("group_sizes", tuple(int(item) for item in group)))
        params.append(("group_list", tuple(int(item) for item in group)))
    if shape_mismatch:
        # Keep the declared output shape; mark the operator incompatible so a
        # candidate region rejects instead of rewriting tensor facts.
        params.append(("declared_shape_mismatch", True))
    if unversioned_state:
        # An unversioned access is an opaque, non-fusable state barrier.
        params.append(("unversioned_state", True))
    if meta.get("capture_ineligible"):
        params.append(("capture_ineligible", True))
    if kind in {"matmul", "grouped_matmul"}:
        abi = getattr(op, "precision_abi", None)
        precision_fields = (
            ("linear_method", meta.get("linear_method") or (abi.linear_method if abi else None)),
            ("activation_format", meta.get("activation_format") or (abi.input_dtype if abi else None)),
            ("weight_format", meta.get("weight_format") or meta.get("weight_dtype") or (abi.weight_dtype if abi else None)),
            ("output_format", meta.get("output_format") or (abi.output_dtype if abi else None)),
            ("compute_format", meta.get("compute_format") or (abi.compute_dtype if abi else None)),
            ("weight_layout", meta.get("weight_layout") or (abi.weight_layout if abi else None)),
            ("weight_block", meta.get("weight_block")),
            ("weight_scale_format", meta.get("weight_scale_format") or (abi.weight_scale_dtype if abi else None)),
            ("activation_group_size", meta.get("activation_group_size") or (abi.act_group_size if abi and abi.act_group_size else None)),
            ("activation_scale_format", meta.get("activation_scale_format") or (abi.act_scale_dtype if abi else None)),
            ("precision_abi_signature", meta.get("precision_abi_signature") or (abi.abi_signature if abi else None)),
        )
        for pkey, pval in precision_fields:
            if pval is not None and pval != "":
                params.append((pkey, pval))
    policy = str(meta.get("numerical_policy", "exact"))
    return OpNode(
        op_id=op.op_id, kind=kind, resource=resource, inputs=inputs,
        outputs=outputs, deps=dep_ids, params=tuple(params),
        stream=op.stream or "main", stage=op.stage, phase=op.phase,
        collective=op.family in {"allreduce", "allgather", "reduce_scatter"}
        or op.link_bytes > 0,
        capture_boundary=str(meta.get("capture_boundary", "")),
        state_effect=state_effect, work=tuple(sorted(work.items())),
        family=op.family, source_boundary=op.source_boundary, dtype=op.dtype,
        layout=(op.layout if op.layout in {"contiguous", "nd", "nz", "strided", "view"}
                else "contiguous"),
        numerical_policy=policy if policy in {"exact", "relaxed"} else "exact",
        priority=int(getattr(op, "priority", 0) or 0),
        semantic_complete=semantic_complete, opaque=opaque,
    )


def _project_operands(op, meta, in_shape, dtype, *, values, produced, by_name,
                      external, external_names, external_bindings):
    # Project data operands from the complete global positional sequence,
    # directly in index order.  Never bucket by producer and never drain
    # several ports at the last producer occurrence: that would regroup
    # ``(b, x, b)`` into ``(b, b, x)`` and delete a repeated ``x``.  The
    # shared classifier keeps an implicit caller event a control edge.
    positions = full_operand_ports(op, by_name)
    explicit_positions = set(input_port_entries(op.metadata))
    declared_controls = declared_control_names(op.metadata)
    controls = set(control_dependency_names(op, by_name))
    unknown_control = False
    for dep in op.deps:
        dep = str(dep)
        if dep in controls and dep not in produced and dep not in external:
            external.append(dep)
    input_ids: list[str] = []
    for dep, port in positions:
        if dep in controls:
            controls.add(dep)
            if dep not in produced and dep not in external:
                external.append(dep)
            continue
        if dep.startswith("event:") and (dep, port) not in explicit_positions:
            # A generated wait is a control edge, never a tensor.  Event
            # insertion materializes the positional sequence first, so a
            # wait without an explicit binding carries no data operand.
            controls.add(dep)
            if dep not in produced and dep not in external:
                external.append(dep)
            continue
        if dep in produced:
            if port not in values or values[port].producer != dep:
                raise FeasibilityError(
                    f"graph_input_port_unknown:{op.op_id}:{port}")
            input_ids.append(port)
            continue
        if (dep, port) in explicit_positions and (
                dep in by_name or dep not in external_bindings):
            # A declared data producer must never fall through to the
            # unknown-control path, which intentionally emits no operand.
            raise FeasibilityError(
                f"graph_input_port_unknown:{op.op_id}:{port}")
        if external_names and dep not in external_names:
            if dep not in declared_controls:
                unknown_control = True
            controls.add(dep)
            if dep not in external:
                external.append(dep)
            continue
        if dep not in external:
            external.append(dep)
        source_id = f"{dep}:source"
        values.setdefault(source_id, TensorValue(
            source_id, dtype, in_shape, role="activation",
            storage_bytes=_activation_bytes(op, meta), sharded=True))
        input_ids.append(source_id)
    return input_ids, unknown_control


def _project_output_dtype(op, kind, dtype, out_shape, input_ids, values):
    shape_mismatch = False
    if kind in {"cast", "kv_cast"}:
        source = values[input_ids[0]] if input_ids else None
        out_dtype = op.dtype or (source.dtype if source is not None else dtype)
    else:
        strict_shape = kind in {"add", "mul", "bias", "dynamic_quant",
                                "quant", "release"}
        if strict_shape:
            # These kinds are shape-preserving.  The declared output shape
            # is authoritative; any disagreeing producer is an inconsistent
            # tensor fact, not permission to rewrite the declaration.
            for value_id in (input_ids[0], *input_ids[1:]) if input_ids else ():
                source = values.get(value_id)
                if source is None or source.producer is None:
                    continue
                if tuple(source.shape) != tuple(out_shape):
                    shape_mismatch = True
                    break
        out_dtype = op.dtype or dtype
    return out_dtype, shape_mismatch


def _project_weights(op, meta, kind, in_shape, out_shape, dtype,
                     tensor_parallel, input_ids, values):
    weight_bytes, _basis = _weight_identity(
        op, kind, in_shape, out_shape,
        {**meta, "tensor_parallel_hint": tensor_parallel})
    if weight_bytes:
        weight_id = f"{op.op_id}:weight"
        abi = getattr(op, "precision_abi", None)
        weight_dtype = _resolve_weight_dtype(op, meta, dtype)
        element = dtype_bytes(weight_dtype)
        if kind in {"matmul", "grouped_matmul"} and len(in_shape) >= 2:
            weight_shape = (int(in_shape[-1]), int(out_shape[-1]))
        else:
            weight_shape = (1, int(weight_bytes / element))
        raw_layout = str(
            meta.get("weight_layout")
            or (abi.weight_layout if abi is not None else "contiguous")
        )
        layout = _normalize_tensor_layout(raw_layout, op_id=op.op_id, allow_strided=False)
        values[weight_id] = TensorValue(
            weight_id, weight_dtype, weight_shape, role="weight",
            storage_bytes=weight_bytes, sharded=True, layout=layout)
        input_ids.append(weight_id)
    if kind in _NORM_KINDS and meta.get("norm_has_weight") is not False:
        gamma_id = f"{op.op_id}:gamma"
        hidden = int(in_shape[-1]) if in_shape else 1
        values[gamma_id] = TensorValue(
            gamma_id, dtype, (hidden,), role="weight",
            storage_bytes=int(math.ceil(hidden * dtype_bytes(dtype))), sharded=True)
        input_ids.append(gamma_id)


def _project_outputs(op, meta, dtype, out_dtype, out_shape, values):
    for index, _returned in enumerate(op.returns):
        value_id = f"{op.op_id}:return:{index}"
        values[value_id] = TensorValue(
            value_id, dtype, (1,), role="scale", producer=op.op_id,
            storage_bytes=int(math.ceil(dtype_bytes(dtype))), sharded=True)
    output_ids: list[str] = []
    primary_id = f"{op.op_id}:out"
    port_records = projected_ports(op.metadata)
    primary_record = next(
        (record for record in port_records if record.port_id == primary_id),
        None)
    if primary_record is not None:
        prim_layout = _normalize_tensor_layout(
            primary_record.layout, op_id=op.op_id, allow_strided=True,
        )
        values[primary_id] = TensorValue(
            primary_id, primary_record.dtype, primary_record.shape,
            role=("output" if primary_record.public or not op.deps
                  else "activation"),
            producer=op.op_id, storage_bytes=primary_record.storage_bytes,
            sharded=True,
            layout=prim_layout, public=primary_record.public)
    else:
        op_layout = _normalize_tensor_layout(
            op.layout or "contiguous", op_id=op.op_id, allow_strided=True,
        )
        values[primary_id] = TensorValue(
            primary_id, out_dtype, out_shape,
            role="output" if not op.deps else "activation",
            producer=op.op_id,
            storage_bytes=(int(meta["write_bytes"])
                           if int(meta.get("write_bytes", 0) or 0) > 0
                           else None),
            sharded=True,
            layout=op_layout)
    output_ids.append(primary_id)
    for record in port_records:
        if record.port_id == primary_id:
            continue
        rec_layout = _normalize_tensor_layout(
            record.layout, op_id=op.op_id, allow_strided=True,
        )
        values[record.port_id] = TensorValue(
            record.port_id, record.dtype, record.shape,
            role="output" if record.public else "activation",
            producer=op.op_id, storage_bytes=record.storage_bytes,
            sharded=True,
            layout=rec_layout, public=record.public)
        output_ids.append(record.port_id)
    return output_ids


def _project_state(op, meta, input_ids, output_ids, values, state_outputs):
    reads, writes = _declared_state(meta)
    state_effect = ""
    unversioned_state = any(version is None for version in
                            (*reads.values(), *writes.values()))
    if op.family in _STATE_FAMILIES or reads or writes:
        if not (reads or writes):
            # No declared effect: never synthesize a ``0 -> 1`` write.
            # The access is an opaque, non-fusable state barrier.
            unversioned_state = True
            state_effect = "write"
        else:
            state_effect = ("read_write" if reads and writes else
                            "read" if reads else "write")
        for target, version in sorted(
                reads.items(),
                key=lambda item: (item[0], -1 if item[1] is None else item[1])):
            if version is None:
                key = f"{op.op_id}:state_in:{target}@unversioned"
            else:
                key = state_outputs.get(
                    (target, version), f"{op.op_id}:state_in:{target}")
            values.setdefault(key, TensorValue(
                key, "fp32", (1,), role="state", storage_bytes=_STATE_SCALAR_BYTES,
                sharded=True, state_version=version, state_key=target))
            input_ids.append(key)
        for target, version in sorted(
                writes.items(),
                key=lambda item: (item[0], -1 if item[1] is None else item[1])):
            suffix = "unversioned" if version is None else str(version)
            key = f"{op.op_id}:state_out:{target}@{suffix}"
            values[key] = TensorValue(
                key, "fp32", (1,), role="state", producer=op.op_id,
                storage_bytes=_STATE_SCALAR_BYTES, sharded=True,
                state_version=version, state_key=target)
            output_ids.append(key)
            if version is not None:
                state_outputs[(target, version)] = key
    return state_effect, unversioned_state


def project_graph(case, ops: tuple[OpSpec, ...], *,
                  opaque_ops: tuple[str, ...] = (),
                  external_data: tuple[str, ...] = ()) -> TypedGraph:
    """Build the typed read-only projection of the selected operator tuple.

    ``external_data`` names the declared graph inputs; any other unknown
    dependency is a control edge and never becomes a fabricated tensor.
    Without that list, scheduled external names can bind explicit source ports.
    ``opaque_ops`` are source-planner/native kernels that must not fuse.
    """
    values: dict[str, TensorValue] = {}
    nodes: dict[str, OpNode] = {}
    produced: set[str] = set()
    external: list[str] = []
    opaque = frozenset(opaque_ops)
    external_names = frozenset(external_data)
    dtype = str(getattr(getattr(case, "model", None), "activation_dtype", "") or "").strip()
    if not dtype and case is None and ops and all(getattr(op, "dtype", None) for op in ops):
        dtype = str(ops[0].dtype).strip()
    if not dtype:
        raise FeasibilityError("projection_model_activation_dtype_missing")
    tensor_parallel = int(getattr(getattr(case, "runtime", None), "parallel", None)
                          and getattr(case.runtime.parallel, "tp", 1) or 1)
    state_outputs: dict[tuple[str, int], str] = {}
    by_name = {item.op_id: item for item in ops}
    external_bindings = external_names or frozenset(
        str(dep) for item in ops for dep in item.deps if str(dep) not in by_name)
    for op in ops:
        meta = _metadata(op)
        kind = _family_kind(op.family)
        vec_only_native = op.family == "native_fusion" and not op.flops and bool(op.vector_ops)
        if vec_only_native:
            kind = "elementwise"
        resource = "cube" if (op.flops or (op.family in _CUBE_FAMILIES and not vec_only_native)) else (
            "mix" if op.link_bytes else "vector")
        if op.family in {"allreduce", "allgather", "reduce_scatter"}:
            resource = "mix"
        in_shape, out_shape = _shapes(op, kind)

        input_ids, unknown_control = _project_operands(
            op, meta, in_shape, dtype, values=values, produced=produced,
            by_name=by_name, external=external, external_names=external_names,
            external_bindings=external_bindings)
        out_dtype, shape_mismatch = _project_output_dtype(
            op, kind, dtype, out_shape, input_ids, values)
        _project_weights(
            op, meta, kind, in_shape, out_shape, dtype, tensor_parallel,
            input_ids, values)
        output_ids = _project_outputs(op, meta, dtype, out_dtype, out_shape, values)
        state_effect, unversioned_state = _project_state(
            op, meta, input_ids, output_ids, values, state_outputs)
        if unversioned_state:
            opaque = opaque | {op.op_id}
        if kind in {"dynamic_quant", "quant"}:
            scale_id = f"{op.op_id}:scale"
            values[scale_id] = TensorValue(
                scale_id, "fp32", (max(1, out_shape[0]) if out_shape else 1, 1),
                role="scale", producer=op.op_id,
                storage_bytes=4 * (max(1, out_shape[0]) if out_shape else 1),
                sharded=True)
            output_ids.append(scale_id)
        complete = (bool(op.dtype and tuple(op.local_shape)) and bool(out_shape)
                    and not unknown_control and not shape_mismatch)
        for value_id in input_ids:
            value = values[value_id]
            if value.producer is not None:
                parent = by_name[value.producer]
                if (parent.stage, parent.stream or "main") != (
                        op.stage, op.stream or "main") \
                        and not _scheduled_before(parent.op_id, op, by_name):
                    raise FeasibilityError(
                        f"graph_input_not_ordered:{op.op_id}:{parent.op_id}")
            if value.producer is None and value.storage_bytes is None:
                complete = False
        nodes[op.op_id] = _opnode(
            op, tuple(op.deps), tuple(input_ids), tuple(output_ids),
            kind, resource, state_effect, semantic_complete=complete,
            opaque=op.op_id in opaque, shape_mismatch=shape_mismatch,
            unversioned_state=unversioned_state)
        produced.add(op.op_id)
    if not ops:
        public = ()
    else:
        consumed = {dep for other in ops for dep in other.deps}
        public = tuple(f"{op.op_id}:out" for op in ops if op.op_id not in consumed)
        public = public or (f"{ops[-1].op_id}:out",)
    rows = 1
    if ops:
        first_meta = _metadata(ops[0])
        rows = int(first_meta.get("graph_active_query_rows", 0) or 0)
        if rows <= 0:
            shape = tuple(int(dim) for dim in ops[0].local_shape)
            rows = shape[0] if len(shape) >= 2 else 1
    return TypedGraph(max(1, rows), values, nodes, public,
                      provenance="canonical_op_spec",
                      external_inputs=tuple(external))
