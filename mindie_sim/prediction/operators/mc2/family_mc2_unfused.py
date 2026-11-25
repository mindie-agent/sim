"""Explicit MC2 -> BF16 grouped GEMM -> SwiGLU -> INT8 caller storage.

[Role] Separate valid expert work from the fixed MC2 receive allocation.
[Invariants] Group lists alone determine matrix rows and active weight reads.
  Unfused pointwise calls scan the whole allocated tensor. The per-call FP32
  gate scale cast is distinct from BF16 scale reads by both grouped matrices.
  Selection declares this native sequence; a fusion flag alone cannot select it.
[Timeouts & Locks] Linear graph scans and O(experts) metadata; no waits or locks.
[Side-Effect Policy] Immutable operator transforms; no I/O, devices or timings.
"""
from dataclasses import replace

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError
from .lowering import FP32_SCALE_BYTES

SELECTOR = "grouped_int8_caller"
IMPLEMENTATION = "vllm_ascend_mc2_unfused_w8a8"
METADATA = "mc2_unfused_storage"
_FP32_STORAGE_BYTES = 4.0
_DYNAMIC_QUANT_OPS_PER_ELEMENT = 4
_SWIGLU_INPUT_BYTES_PER_OUTPUT = 4
_SWIGLU_OPS_PER_OUTPUT = 4
SOURCE = {
    "allocation": {
        "repository": "ops-transformer", "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        "path": "mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_infershape.cpp",
        "sha256": "f4bde620edf03211cab4f8473d2ab80e3fe382fc4593d08c977357892e00d31e",
        "lines": (250, 263, 280, 282),
        "formula": "A=(source_rows*EP)*min(local_experts,top_k); global_bs=0; no shared ranks",
    },
    "unfused_operations": {
        "repository": "vllm-ascend", "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
        "path": "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
        "sha256": "8fc67c08812d71da683c84adf2cf4b1411ffb970d82d48958ed51268c2ad758b",
        "lines": (486, 524),
        "activation_path": "vllm_ascend/ops/fused_moe/moe_mlp.py:47-55,95",
        "activation_sha256": "2bf107e3287e3a4c69512e60e65495448f34a4451eab32ae3b23b5c2a46d2558",
    },
    "scope": "independently_selected_unfused_native_sequence; not_default_dispatch_or_capture_binary_identity",
}

CAPABILITIES = frozenset({
    "A5", "decode", "mc2_receive", "int8_w8a8", "bf16_runtime_weight_scale",
    "fp32_storage_weight_scale_cast", "unfused_activation_chain",
    "capacity_padded_pointwise_scan", "masked_source_force_eplb",
})


def capabilities():
    """Machine-readable receive allocation and representation contract."""
    return {
        "selector": SELECTOR,
        "implementation": IMPLEMENTATION,
        "metadata_key": METADATA,
        "soc_families": ("A5",),
        "phases": ("decode",),
        "activation_format": "int8",
        "weight_format": "int8",
        "output_format": "bf16",
        "activation_scale_format": "fp32",
        "weight_scale_runtime_format": "bf16",
        "weight_scale_storage_formats": ("fp32", "bf16"),
        "dynamic_attributes": (
            "matrix_group_rows", "moe_token_partition", "mc2_phase",
            "activation_payload_shape", "activation_scale_shape",
            "graph_active_query_rows", "graph_physical_query_rows",
            "routing_realization", "source_force_eplb",
        ),
        "requires": (
            "explicit_global_bs_zero", "one_dispatch_receiver",
            "exclusive_gate_up_swiglu_quant_down_chain",
            "per_row_fp32_activation_scale", "declared_weight_scale_storage",
        ),
        "source": SOURCE,
    }


def recipes():
    # Existing matrix and vector recipes price the unchanged semantic work.
    return {}


def _one(items, reason):
    if len(items) != 1:
        raise ModelingError("mc2_unfused_" + reason)
    return items[0]


def _scale_storage(meta):
    """Resolve storage representation and the per-call cast that produces BF16.

    ``gemm_op`` may declare the source checkpoint storage (FP32 per channel)
    separately from the runtime BF16 channel scale that the fused native call
    consumes.  Without that separation, a runtime FP32 scale is still storage.
    """
    runtime_format = meta.get("weight_scale_format")
    runtime_bytes = float(meta.get("weight_scale_bytes", 0) or 0)
    storage_format = meta.get("weight_scale_storage_format")
    storage_bytes = float(meta.get("weight_scale_storage_bytes", 0) or 0)
    if storage_format is None:
        # Legacy records carry one representation. An FP32 scale is converted
        # to the BF16 runtime input by this caller; a BF16 scale already is it.
        if runtime_format == "fp32" and runtime_bytes == _FP32_STORAGE_BYTES:
            return "fp32", _FP32_STORAGE_BYTES
        if runtime_format == "bf16" and runtime_bytes == 2.0:
            return "bf16", 2.0
        raise ModelingError(
            "mc2_unfused_requires_bf16_runtime_scale_from_declared_storage")
    if ((storage_format, storage_bytes) in {("fp32", _FP32_STORAGE_BYTES), ("bf16", 2.0)}
            and ((runtime_format, runtime_bytes) == ("bf16", 2.0)
                 or (runtime_format, runtime_bytes) == (storage_format, storage_bytes))):
        return storage_format, storage_bytes
    raise ModelingError(
        "mc2_unfused_requires_bf16_runtime_scale_from_declared_storage")


def _chain(gate, by_id, users):
    activation = _one([by_id[key] for key in users[gate.op_id]
                       if by_id[key].family == "swiglu"], "requires_one_swiglu")
    quant = _one([by_id[key] for key in users[activation.op_id]
                  if by_id[key].family == "quantize"], "requires_one_quantizer")
    down = _one([by_id[key] for key in users[quant.op_id]
                 if by_id[key].family == "grouped_gemm"], "requires_one_down_matrix")
    if (users[gate.op_id] != {activation.op_id}
            or users[activation.op_id] != {quant.op_id} or users[quant.op_id] != {down.op_id}
            or activation.deps != (gate.op_id,) or quant.deps != (activation.op_id,)
            or down.deps != (quant.op_id,)
            or any(op.plan_required or "implementation_binding" in dict(op.metadata)
                   for op in (gate, activation, quant, down))):
        raise ModelingError("mc2_unfused_requires_ordinary_exclusive_activation_chain")
    return activation, quant, down


def _contract(gate, down, dispatch):
    gm, dm, source = dict(gate.metadata), dict(down.metadata), dict(dispatch.metadata)
    rows, columns, hidden = gate.local_shape
    counts = tuple(gm.get("matrix_group_rows", ()))
    caller = gm.get("moe_token_partition", {})
    phase = source.get("mc2_phase", {})
    if (not counts or any(type(n) is not int or n < 0 for n in counts)
            or sum(counts) != rows or columns % 2 or rows < 1
            or tuple(dm.get("matrix_group_rows", ())) != counts
            or down.local_shape != (rows, hidden, columns // 2)):
        raise ModelingError("mc2_unfused_requires_matching_integer_expert_matrices")
    if (caller.get("kind") != "vllm_ascend_mc2_tp_slice"
            or phase.get("kind") != "dispatch" or phase.get("dispatch_quant") != "per_token"
            or source.get("implementation") != "arch35_fullmesh_per_token_int8"
            or gm.get("activation_payload_producer") != dispatch.op_id
            or gm.get("activation_scale_producer") != dispatch.op_id
            or gate.deps != (dispatch.op_id,)
            or "activation_scales" not in dispatch.returns):
        raise ModelingError("mc2_unfused_requires_owned_mc2_int8_receiver")
    ep, top_k, experts = phase["ep"], phase["top_k"], phase["experts"]
    source_rows = caller["local_rows"]
    if (experts % ep or experts // ep != len(counts)
            or ep < caller["tp"] or ep % caller["tp"] != 0
            or phase["source_tokens"] != source_rows or phase["received_assignments"] != rows
            or tuple(source.get("activation_payload_shape", ())) != (rows, hidden)
            or tuple(source.get("activation_scale_shape", ())) != (rows,)):
        raise ModelingError("mc2_unfused_receiver_geometry_disagrees_with_expert_work")
    active_query = gm.get("graph_active_query_rows", caller["full_rows"])
    physical_query = gm.get("graph_physical_query_rows", caller["full_rows"])
    ledger = gm.get("routing_realization", {})
    if active_query != physical_query:
        # A capture buffer does not prove which dummy rows routing discards.
        # Only the explicit source-mask ledger can make this padded call exact.
        if (ledger.get("basis") != "source_force_eplb_masked_integer_counts"
                or ledger.get("valid_global_tokens") != active_query
                or ledger.get("allocated_global_tokens") != source_rows * ep
                or source.get("source_force_eplb") != ledger):
            raise ModelingError("mc2_unfused_padded_call_requires_explicit_source_mask_ledger")
    storage_by_op = {}
    for op, meta in ((gate, gm), (down, dm)):
        storage_by_op[op.op_id] = _scale_storage(meta)
        if (meta.get("activation_format") != "int8" or meta.get("weight_format") != "int8"
                or meta.get("output_format") != "bf16" or meta.get("bias")
                or meta.get("activation_scale_format") != "fp32"
                or meta.get("activation_group_size") != op.local_shape[-1]
                or tuple(meta.get("activation_scale_shape", ())) != (rows,)):
            raise ModelingError("mc2_unfused_requires_int8_bf16_and_per_row_fp32_scales")
    unique_storages = set(storage_by_op.values())
    if len(unique_storages) == 1:
        storage_format, storage_bytes = next(iter(unique_storages))
    else:
        storage_format, storage_bytes = "mixed", None
    storage_map = {
        op_id: {"format": fmt, "bytes": nbytes}
        for op_id, (fmt, nbytes) in storage_by_op.items()
    }
    cast_required_by_op = {
        op_id: fmt == "fp32" for op_id, (fmt, _) in storage_by_op.items()
    }
    capacity = source_rows * ep * min(len(counts), top_k)
    if capacity < rows:
        raise ModelingError("mc2_unfused_valid_rows_exceed_receive_capacity")
    return {"implementation": IMPLEMENTATION, "global_bs": 0,
            "capacity_rows": capacity, "valid_rows": rows,
            "source_rows": source_rows, "ep": ep, "top_k": top_k,
            "local_experts": len(counts), "active_experts": sum(n > 0 for n in counts),
            "graph_active_query_rows": active_query, "graph_physical_query_rows": physical_query,
            "group_list_values": counts, "group_list_basis": gm.get("routing_basis"),
            "weight_scale_storage_format": storage_format,
            "weight_scale_storage_bytes": storage_bytes,
            "weight_scale_cast_required": any(cast_required_by_op.values()),
            "weight_scale_storage_by_op": storage_map,
            "weight_scale_cast_required_by_op": cast_required_by_op,
            "source": SOURCE}


def _matrix(op, contract, *, scale_producer=None):
    meta = dict(op.metadata)
    rows, n, k = op.local_shape
    active, capacity = contract["active_experts"], contract["capacity_rows"]
    payload, scales = active * n * k, active * n * 2
    # Native calls supply one BF16 channel scale and no offset. The cast scans
    # the complete weight-scale bank; the matrix reads only active experts.
    old_weight = meta["packed_weight_bytes"]
    weight = payload + scales
    read = op.mte2_bytes - old_weight + weight
    epilogue = {**meta["matrix_epilogue"], "column_scale_bytes": scales}
    op_storage = contract["weight_scale_storage_by_op"][op.op_id]
    meta.update({METADATA: contract, "native_kind": "GroupedMatmul",
                 "input_capacity_shape": (capacity, k), "output_capacity_shape": (capacity, n),
                 "activation_scale_capacity_shape": (capacity,),
                 "weight_scale_shape": (contract["local_experts"], n),
                 "weight_scale_format": "bf16", "weight_scale_bytes": 2.0,
                 "weight_scale_storage_format": op_storage["format"],
                 "weight_scale_storage_bytes": op_storage["bytes"],
                 "weight_payload_bytes": payload, "packed_weight_bytes": weight,
                 "weight_bytes": weight, "matrix_epilogue": epilogue,
                 "int8_weight_metadata": "scale_only",
                 "column_metadata_policy": "declared_actual_runtime_scale_only"})
    deps = op.deps
    if scale_producer:
        meta["weight_scale_producer"] = scale_producer
        deps = (*deps, scale_producer)
    precision_abi = (
        replace(op.precision_abi, weight_scale_dtype="bf16", weight_scale_bytes=2)
        if op.precision_abi is not None else None
    )
    return replace(op, deps=deps, padded_shape=(capacity, n, k), mte2_bytes=read,
                   hbm_bytes=read + op.mte3_bytes, precision_abi=precision_abi,
                   metadata=tuple(meta.items()))


def _pointwise(op, contract, *, quantize=False):
    meta = dict(op.metadata)
    rows, width = op.local_shape
    capacity = contract["capacity_rows"]
    if rows != contract["valid_rows"] or op.dtype != "bf16":
        raise ModelingError("mc2_unfused_pointwise_input_must_match_expert_rows")
    if quantize:
        if (meta.get("output_format") != "int8" or meta.get("activation_scale_format") != "fp32"
                or meta.get("activation_group_size") != width
                or tuple(meta.get("activation_scale_shape", ())) != (rows,)):
            raise ModelingError("mc2_unfused_requires_per_token_int8_output_quantization")
        read = capacity * width * 2
        write = capacity * width + capacity * FP32_SCALE_BYTES
        vector = capacity * (_DYNAMIC_QUANT_OPS_PER_ELEMENT * width + 2)
        meta.update(activation_scale_shape=(capacity,), activation_scale_bytes=capacity * FP32_SCALE_BYTES,
                    native_kind="DynamicQuantV2")
    else:
        if meta.get("activation_kind") != "silu":
            raise ModelingError("mc2_unfused_requires_unclipped_silu")
        read = capacity * width * _SWIGLU_INPUT_BYTES_PER_OUTPUT
        write = capacity * width * 2
        vector = capacity * width * _SWIGLU_OPS_PER_OUTPUT
        meta["native_kind"] = "SwiGlu"
    meta.update({METADATA: contract, "physical_shape": (capacity, width),
                 "read_bytes": read, "write_bytes": write})
    return replace(op, padded_shape=(capacity, width), launch_shape=(capacity, width),
                   mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write,
                   vector_ops=vector, metadata=tuple(meta.items()))


def _scale_cast(op, contract, columns, *, inherited):
    banks = contract["local_experts"]
    op_storage = contract["weight_scale_storage_by_op"][op.op_id]
    cast = vector_op(
        op_id=op.op_id + ".weight_scale_cast", deps=op.deps, stage=op.stage,
        layer=op.layer, requests=op.requests, phase=op.phase, dtype="bf16",
        family="cast", recipe="expert_weight_scale_cast", elements=banks * columns,
        ops_per_element=1, shape=(banks, columns),
        read_bytes_override=banks * columns * op_storage["bytes"],
        write_bytes_override=banks * columns * 2,
        metadata=tuple({**inherited, METADATA: contract, "native_kind": "Cast",
                        "input_format": op_storage["format"],
                        "output_format": "bf16",
                        "storage_to_runtime_scale_representation": True}.items()))
    return replace(cast, stream=op.stream)


def _rewrite(ops, gate):
    by_id = {op.op_id: op for op in ops}
    users = {op.op_id: set() for op in ops}
    for op in ops:
        for dep in op.deps:
            if dep in users:
                users[dep].add(op.op_id)
    activation, quant, down = _chain(gate, by_id, users)
    dispatch = _one([by_id[key] for key in gate.deps if key in by_id], "requires_one_receiver")
    contract = _contract(gate, down, dispatch)
    inherited = {key: value for key, value in gate.metadata if key in {
        "structure_block", "structure_role", "graph_active_query_rows", "graph_physical_query_rows"}}
    casts = {}
    for proj in (gate, down):
        if contract["weight_scale_cast_required_by_op"][proj.op_id]:
            # W8A8 dynamic loads each expert matrix's per-channel scales from
            # declared storage and casts only FP32 banks to the BF16 runtime scale.
            casts[proj.op_id] = _scale_cast(
                proj, contract, proj.local_shape[1], inherited=inherited)
    contract = {**contract,
                "weight_scale_cast_ops": {key: op.op_id for key, op in casts.items()}}
    dm = dict(dispatch.metadata)
    dm.update({METADATA: contract,
               "activation_payload_capacity_shape": (contract["capacity_rows"], gate.local_shape[-1]),
               "activation_scale_capacity_shape": (contract["capacity_rows"],)})
    replacements = {
        gate.op_id: _matrix(gate, contract,
                            scale_producer=casts[gate.op_id].op_id if gate.op_id in casts else None),
        activation.op_id: _pointwise(activation, contract),
        quant.op_id: _pointwise(quant, contract, quantize=True),
        down.op_id: _matrix(down, contract,
                            scale_producer=casts[down.op_id].op_id if down.op_id in casts else None),
        dispatch.op_id: replace(dispatch, metadata=tuple(dm.items())),
    }
    result = []
    for op in ops:
        if op.op_id in casts:
            result.append(casts[op.op_id])
        result.append(replacements.get(op.op_id, op))
    return tuple(result)


def lower(ops, bindings):
    from mindie_sim.prediction.implementations import select
    selected = select(bindings, SELECTOR)
    if selected is None or selected.name != IMPLEMENTATION:
        return tuple(ops)
    if selected.options != {"global_bs": 0} or type(selected.options["global_bs"]) is not int:
        raise ModelingError("mc2_unfused_requires_explicit_global_bs0")
    fusion = select(bindings, "native_fusion")
    if fusion and "grouped_gate_up_activation_quant" in fusion.options.get("roles", ()):
        raise ModelingError("mc2_unfused_cannot_select_grouped_fusion")
    current = tuple(ops)
    for gate in ops:
        if (gate.family == "grouped_gemm" and gate.recipe in {"grouped_gate_up", "balanced_grouped_gate_up"}
                and METADATA not in dict(gate.metadata)):
            current = _rewrite(current, gate)
    return current


__all__ = ["CAPABILITIES", "IMPLEMENTATION", "METADATA", "SELECTOR", "SOURCE",
           "capabilities", "lower", "recipes"]
