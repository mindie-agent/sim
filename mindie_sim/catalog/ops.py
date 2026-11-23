"""Catalog op builders with exact quantized weight/activation byte accounting.

[Role]
  Construct OpSpec records for catalog lowering.  The ordinary generic
  constructors assume dense BF16 weights; these builders distinguish packed
  FP4/FP8 weight bytes, activation bytes, compute dtype, output bytes and
  collective payload so calibration can identify the semantic path.

[Invariants]
  * GEMM work remains 2*M*N*K; packed weight bytes never change logical FLOPs.
  * FP4/MXFP4 compute dtype follows the paired activation format (mxfp4 for FP4
    activations, fp8_e4m3 for FP8 activations, bf16 for BF16/FP16 W4A16).
  * Every op has distinct semantic/local/padded/launch shapes.
  * A cache store is a real dependent event; a shared consumer must depend on
    the producer store or an explicit cross-stage transfer.
  * Collective payloads are logical bytes and are not multiplied by TP twice.

[Timeouts & Locks]
  Pure record construction; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.operators import LinearMethod, OpSpec, TensorPrecisionABI
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes

#: Fused groupwise INT4 kernels unpack two nibbles per stored byte, widen the
#: integer to the compute dtype, then multiply by one declared group scale.
W4A16_DEQUANT_OPS_PER_WEIGHT = 3
#: Packed integer weights whose activation stays floating point dequantize
#: in-kernel; the op's compute identity is that activation format.
W4A16_WEIGHT_FORMATS = frozenset({"int4", "w4a8"})
FP4_FORMATS = frozenset({"mxfp4", "fp4", "fp4_e2m1"})
FP8_FORMATS = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8"})
INT8_EPILOGUE_FP32_OPS = 3
INT8_EPILOGUE_CAST_OPS = 4
CUBE_TILE_ALIGN = 16


def _safe_bytes(dtype: str) -> float:
    return float(dtype_bytes(dtype))


def compute_dtype_for(quant: QuantSpec, category: str) -> tuple[str, str]:
    """Return (compute dtype, declared activation dtype) for a category."""
    quant = quant.for_category(category)
    fmt = quant.format_for(category)
    if fmt in {"fp32", "float32"}:
        return "fp32", "fp32"
    if fmt in {"bf16", "bfloat16"}:
        return "bf16", "bf16"
    if fmt in {"fp16", "float16"}:
        return "fp16", "fp16"
    if fmt in {"int8", "w8a8"}:
        return "int8", "int8"
    if fmt in {"int4", "w4a8"}:
        # Packed weights do not imply quantized activations or INT8 compute.
        if quant.activation_format == "int8":
            return "int8", "int8"
        return quant.activation_format, quant.activation_format
    if fmt in FP4_FORMATS:
        act = str(quant.activation_format).strip().lower()
        if act in FP4_FORMATS:
            return "mxfp4", quant.activation_format
        if act in FP8_FORMATS:
            return ("fp8_e5m2" if act == "fp8_e5m2" else "fp8_e4m3"), quant.activation_format
        return "bf16", quant.activation_format
    if fmt in FP8_FORMATS:
        act = str(quant.activation_format).strip().lower()
        if act in FP8_FORMATS:
            return "fp8_e4m3", "fp8_e4m3"
        return "bf16", "bf16"
    raise ModelingError(f"compute_format_undeclared:{category}:{fmt}")


_LOW_OUTPUT_ACTIVATIONS = frozenset({
    "int8", "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "fp4", "fp4_e2m1", "mxfp4",
})


def matrix_output_format(
    quant: QuantSpec, category: str, *, activation_dtype: str = "",
    output_dtype: str = "", prequant: str = "",
) -> str:
    """Resolve the direct output representation of one matrix role.

    This is the single pure rule consumed by ``gemm_op`` and the catalog
    format projection.  Caller output wins, then the role's explicit
    QuantSpec output, then the existing MX/4-bit rule, then the normal matrix
    default: low-precision operands produce a BF16 direct result, high
    precision operands produce that operand format.  A downstream BF16/FP16
    stream is not substituted for the direct result; a real cast must be its
    own typed producer.
    """
    resolved = quant.for_category(category)
    if output_dtype:
        return str(output_dtype)
    declared = resolved.output_format_for(category, prequant=prequant)
    if declared:
        return str(declared)
    act = activation_dtype or compute_dtype_for(resolved, category)[1]
    return "bf16" if act in _LOW_OUTPUT_ACTIVATIONS else str(act)


def packed_weight_bytes(
    quant: QuantSpec, rows: int, cols: int, category: str,
    *, count: int = 1, source: bool = False,
) -> float:
    return float(quant.matrix_bytes(rows, cols, category, source=source)) * float(max(1, count))


def packed_weight_format(quant: QuantSpec, category: str) -> str:
    return quant.format_for(category)


def _gemm_packed_and_epilogue_meta(
    *, quant: QuantSpec, category: str, fmt: str, act: str, out_dtype: str,
    compute: str, m: int, n: int, k: int, group: int, scale_bytes: float,
    weight_bytes: float, weight_cols: int | None, unit_shape: tuple[int, int],
    unit_total: float, bytes_per_weight: float, weight_elements: float,
    metadata: tuple[tuple[str, Any], ...],
) -> tuple[tuple[tuple[str, Any], ...], float]:
    """Build packed-weight, fused-dequant, and INT8 epilogue metadata + vector_ops."""
    packed = fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}
    groupwise_dequant = (fmt in W4A16_WEIGHT_FORMATS and compute != "int8"
                         and m > 0 and weight_elements > 0)
    extra: tuple[tuple[str, Any], ...] = ()
    if packed:
        existing_meta = dict(metadata)
        scale_meta: tuple[tuple[str, Any], ...] = ()
        if "weight_payload_bytes" not in existing_meta:
            unit_payload = float(quant.matrix_payload_bytes(*unit_shape, category))
            payload_bytes = weight_bytes * (unit_payload / unit_total) if unit_total > 0 else 0.0
            scale_meta += (("weight_payload_bytes", payload_bytes),)
        if "weight_scale_format" not in existing_meta:
            scale_meta += (("weight_scale_format", quant.execution_weight_scale_format()),)
        if "weight_scale_bytes" not in existing_meta:
            scale_meta += (("weight_scale_bytes", quant.execution_weight_scale_bytes()),
                           ("weight_scale_storage_format", quant.scale_format),
                           ("weight_scale_storage_bytes", float(quant.scale_bytes)))
        extra = extra + scale_meta + (
            ("dequant", "fused" if fmt in {"mxfp4", "fp4", "fp4_e2m1", "int4", "w4a8"} else "none"),
            ("dequant_traffic", 0.0),
            ("weight_bytes", weight_bytes),
            ("implementation_assumption", "declared_matrix_operand_formats"),
        )
    if groupwise_dequant:
        extra = extra + (
            ("dequant", "groupwise_fused"),
            ("dequant_group_size", max(1, int(quant.expert_block))),
            ("dequant_ops_per_weight", float(W4A16_DEQUANT_OPS_PER_WEIGHT)),
            ("dequant_weight_elements", float(weight_elements)),
            ("dequant_bytes_per_element", float(bytes_per_weight)),
            ("dequant_compute_format", compute),
            ("dequant_residency", "on_chip_l1_l0_no_hbm_writeback"),
        )
    vector_ops = (float(W4A16_DEQUANT_OPS_PER_WEIGHT * weight_elements)
                  if groupwise_dequant else 0.0)
    if (fmt in {"int8", "w8a8"} and act == "int8" and group >= k
            and out_dtype in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}):
        runtime_meta = quant.execution_int8_weight_metadata()
        runtime_scale_bytes = quant.execution_weight_scale_bytes()
        metadata_columns = 1 if runtime_meta == "scale_only" else 2
        column_width = metadata_columns * runtime_scale_bytes
        column_metadata = weight_bytes * column_width / (int(weight_cols or k) + column_width)
        operations = (INT8_EPILOGUE_FP32_OPS if out_dtype in {"fp32", "float32"}
                      else INT8_EPILOGUE_CAST_OPS)
        vector_ops = float(operations * m * n)
        extra += (("matrix_epilogue", {
            "kind": "row_column_scale", "row_scale_bytes": float(scale_bytes),
            "column_scale_bytes": column_metadata, "operations_per_output": operations,
        }), ("matrix_output_contract", "int32_cast_row_scale_column_scale_output_cast"),
            ("int8_weight_metadata", runtime_meta),
            ("column_metadata_policy", "declared_scale_only_storage_traffic"
             if runtime_meta == "scale_only"
             else "retains_declared_scale_and_offset_storage_traffic"))
    return extra, vector_ops


_HIGH_PRECISION_GEMM_FORMATS = frozenset({
    "bf16", "bfloat16", "fp16", "float16", "fp32", "float32",
})


def precision_abi_for_gemm(
    quant: QuantSpec,
    category: str,
    *,
    activation_dtype: str = "",
    output_dtype: str = "",
    compute_dtype: str = "",
    prequant_dtype: str = "",
    act_group: int | None = None,
    act_scale_format: str | None = None,
    act_scale_bytes: float | None = None,
) -> TensorPrecisionABI:
    """Resolve the authoritative ``TensorPrecisionABI`` for one GEMM role."""
    resolved = quant.for_category(category)
    fmt = packed_weight_format(resolved, category)
    default_compute, default_act = compute_dtype_for(resolved, category)
    act = str(activation_dtype or default_act)
    compute = str(compute_dtype or default_compute)
    out_dt = matrix_output_format(
        resolved, category, activation_dtype=act, output_dtype=output_dtype,
        prequant=prequant_dtype or resolved.native_format or "",
    )
    high_prec = fmt in _HIGH_PRECISION_GEMM_FORMATS
    if high_prec:
        method: LinearMethod = "dense"
        weight_block: tuple[int, ...] = ()
        scale_dtype = ""
        scale_bytes_int = 0
        weight_meta: tuple[tuple[str, Any], ...] = ()
    else:
        raw_method = resolved.linear_method(category)
        if fmt in {"int4", "w4a8"}:
            raw_method = "w4a8" if act == "int8" else "w4a16"
        method = raw_method  # type: ignore[assignment]
        if fmt in {"mxfp4", "fp4", "fp4_e2m1", "int4", "w4a8"}:
            weight_block = (1, max(1, int(resolved.expert_block)))
        elif (
            len(resolved.weight_block) == 2
            and int(resolved.weight_block[0]) > 0
            and int(resolved.weight_block[1]) > 0
        ):
            weight_block = (int(resolved.weight_block[0]), int(resolved.weight_block[1]))
        else:
            weight_block = ()
        scale_dtype = str(resolved.execution_weight_scale_format())
        scale_bytes_int = max(0, int(round(float(resolved.execution_weight_scale_bytes()))))
        meta_items: list[tuple[str, Any]] = [
            ("storage_scale_format", str(resolved.scale_format)),
            ("storage_scale_bytes", float(resolved.scale_bytes)),
        ]
        if fmt in {"int8", "w8a8", "int4", "w4a8"}:
            meta_items.append(("int8_weight_metadata", str(resolved.execution_int8_weight_metadata())))
        weight_meta = tuple(meta_items)

    has_act_scale = (act_scale_bytes is not None and act_scale_bytes > 0.0) or (
        act_scale_bytes is None and act in _LOW_OUTPUT_ACTIVATIONS
    )
    if has_act_scale:
        eff_act_group = int(act_group if act_group is not None else (resolved.activation_group_size or 0))
        eff_act_scale = str(
            act_scale_format
            if act_scale_format is not None
            else (
                resolved.execution_activation_scale_format(act)
                if act == "int8"
                else (resolved.activation_scale_format or resolved.scale_format)
            )
        )
        eff_act_pack = (
            f"pack{int(resolved.activation_scale_pack)}"
            if int(resolved.activation_scale_pack) > 1
            else ("scalar" if eff_act_scale else "")
        )
    else:
        eff_act_group, eff_act_scale, eff_act_pack = 0, "", ""

    return TensorPrecisionABI(
        linear_method=method,
        input_dtype=act,
        weight_dtype=fmt,
        output_dtype=out_dt,
        compute_dtype=compute,
        accum_dtype="int32" if compute == "int8" else "fp32",
        weight_layout=_layout_name(fmt, resolved),
        weight_block=weight_block,
        weight_scale_dtype=scale_dtype,
        weight_scale_bytes=scale_bytes_int,
        weight_metadata=weight_meta,
        act_group_size=max(0, eff_act_group),
        act_scale_dtype=eff_act_scale,
        act_scale_pack=eff_act_pack,
    )


def gemm_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], m: int, n: int, k: int, quant: QuantSpec,
    category: str, phase: str, recipe: str = "", family: str = "gemm",
    semantic_family: str = "", weight_count: int = 1,
    weight_rows: int | None = None, weight_cols: int | None = None,
    weight_bytes_override: float | None = None,
    semantic_shape: tuple[int, ...] | None = None,
    metadata: tuple[tuple[str, Any], ...] = (),
    activation_dtype: str | None = None, output_dtype: str | None = None,
    prequant_dtype: str | None = None,
    notes: tuple[str, ...] = (),
) -> OpSpec:
    quant = quant.for_category(category)
    m, n, k = int(m), int(n), int(k)
    if m < 0 or min(n, k) < 1:
        raise ModelingError(f"matrix_geometry_invalid:{op_id}:{m},{n},{k}")
    declared_compute = dict(metadata).get("compute_format")
    compute, act = compute_dtype_for(quant, category)
    compute_format_origin = "catalog_default"
    if declared_compute:
        compute = str(declared_compute)
        compute_format_origin = "explicit"
    act = activation_dtype or act
    out_dtype = matrix_output_format(
        quant, category, activation_dtype=act, output_dtype=output_dtype or "",
        prequant=prequant_dtype or quant.native_format or "",
    )
    act_bytes = _safe_bytes(act)
    out_bytes = _safe_bytes(out_dtype)
    weight_bytes = (
        float(weight_bytes_override)
        if weight_bytes_override is not None
        else packed_weight_bytes(
            quant, int(weight_rows or n), int(weight_cols or k), category,
            count=max(1, int(weight_count)),
        )
    )
    if not m:
        weight_bytes = 0
    input_bytes = float(m * k) * act_bytes
    output_bytes = float(m * n) * out_bytes
    group, scale_format, scale_shape, scale_bytes = activation_scales(quant, act, m, k)
    hbm = input_bytes + weight_bytes + output_bytes + scale_bytes
    flops = 2.0 * m * n * k
    fmt = packed_weight_format(quant, category)
    unit_shape = (int(weight_rows or n), int(weight_cols or k))
    unit_total = float(quant.matrix_bytes(*unit_shape, category))
    bytes_per_weight = (unit_total / float(unit_shape[0] * unit_shape[1])
                        if unit_total > 0 else 0.0)
    weight_elements = (weight_bytes / bytes_per_weight
                       if bytes_per_weight > 0 else 0.0)
    recipe = recipe or semantic_family or family
    abi = precision_abi_for_gemm(
        quant, category,
        activation_dtype=act, output_dtype=out_dtype, compute_dtype=compute,
        prequant_dtype=prequant_dtype or quant.native_format or "",
        act_group=group if scale_bytes > 0 else 0,
        act_scale_format=scale_format if scale_bytes > 0 else "",
        act_scale_bytes=scale_bytes,
    )
    meta = tuple(metadata) + (
        ("semantic_family", semantic_family or family),
        ("linear_method", abi.linear_method),
        ("precision_abi_signature", abi.abi_signature),
        ("weight_format", fmt),
        ("weight_dtype", fmt),
        ("activation_format", act),
        ("input_format", act),
        ("output_format", out_dtype),
        ("compute_format", compute),
        ("compute_format_origin", compute_format_origin),
        ("quant_category", category),
        ("catalog_category", category),
        ("packed_weight_bytes", weight_bytes),
        ("matrix_groups", max(1, int(weight_count))),
        ("weight_representation", _layout_name(fmt, quant)),
        ("dequant_before_compute", fmt in {"mxfp4", "fp4", "fp4_e2m1"}),
        *((("activation_group_size", group),
           ("activation_scale_format", scale_format),
           ("activation_scale_shape", scale_shape),
           ("activation_scale_bytes", float(scale_bytes)),
           ("activation_scale_in_mte2", True))
          if scale_bytes > 0 else ()),
    )
    extra_meta, vector_ops = _gemm_packed_and_epilogue_meta(
        quant=quant, category=category, fmt=fmt, act=act, out_dtype=out_dtype,
        compute=compute, m=m, n=n, k=k, group=group, scale_bytes=scale_bytes,
        weight_bytes=weight_bytes, weight_cols=weight_cols,
        unit_shape=unit_shape, unit_total=unit_total,
        bytes_per_weight=bytes_per_weight, weight_elements=weight_elements,
        metadata=metadata,
    )
    meta = meta + extra_meta
    return OpSpec(
        op_id=op_id, family=family, phase=phase, stage=stage, layer=layer,
        requests=requests,
        semantic_shape=semantic_shape or (m, n, k),
        local_shape=(m, n, k),
        padded_shape=(align_up(m, CUBE_TILE_ALIGN), align_up(n, CUBE_TILE_ALIGN),
                      align_up(k, CUBE_TILE_ALIGN)),
        launch_shape=(m, n, k),
        flops=flops, vector_ops=vector_ops, hbm_bytes=hbm,
        mte2_bytes=input_bytes + weight_bytes + scale_bytes,
        mte3_bytes=output_bytes, deps=deps, recipe=recipe,
        notes=tuple(notes) + ("gemm_formula=2*M*N*K",
                              "weight_bytes_use_declared_packing"),
        metadata=meta + (("program_kind", "matmul"),), dtype=compute,
        layout=_layout_name(fmt, quant),
        precision_abi=abi,
    )


def activation_scales(quant, dtype, rows, cols):
    """Activation scales follow the declared quantization, not weight bytes."""
    if dtype == "int8":
        group = quant.activation_group_size or cols
        groups = (cols + group - 1) // group
        scale_format = quant.execution_activation_scale_format(dtype)
        return group, scale_format, (rows,) if groups == 1 else (rows, groups), float(rows * groups * dtype_bytes(scale_format))
    if dtype in (FP8_FORMATS | FP4_FORMATS):
        group = quant.activation_group_size or cols
        groups = (cols + group - 1) // group
        scale_format = quant.activation_scale_format or quant.scale_format
        pack = quant.activation_scale_pack
        shape = (rows, (groups + pack - 1) // pack, pack) if pack > 1 else (rows, groups)
        count = rows * ((groups + pack - 1) // pack) * pack
        return group, scale_format, shape, float(count * dtype_bytes(scale_format))
    return 0, "", (), 0.0


def _layout_name(fmt: str, quant: QuantSpec) -> str:
    if fmt in {"mxfp4", "fp4", "fp4_e2m1"}:
        return f"fp4_group{max(1, quant.expert_block)}_packed"
    if fmt in {"int4", "w4a8"}:
        if int(quant.expert_block) > 1:
            return f"int4_group{int(quant.expert_block)}_packed"
        # W4A8_DYNAMIC: two int4 values per byte, one fp32 scale per output row.
        return "int4_packed_per_channel"
    if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        if quant.weight_block[0] > 0:
            return f"fp8_block{quant.weight_block[0]}x{quant.weight_block[1]}_{quant.scale_format}"
        return "fp8_rowwise"
    if fmt in {"fp32", "float32"}:
        return "fp32"
    if fmt in {"int8", "w8a8"}:
        return "int8_per_channel"
    if fmt in {"fp16", "float16"}:
        return "fp16"
    return "bf16"


def vector_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], elements: int, ops_per_element: float,
    dtype: str, family: str, phase: str, recipe: str,
    semantic_family: str = "", read_write: float = 2.0,
    shape: tuple[int, ...] | None = None,
    metadata: tuple[tuple[str, Any], ...] = (),
    hbm_bytes_override: float | None = None,
    read_bytes_override: float | None = None,
    write_bytes_override: float | None = None,
    vector_ops_override: float | None = None,
    returns: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> OpSpec:
    elements = max(0, int(elements))
    bytes_per = _safe_bytes(dtype)
    vector_ops = (
        float(vector_ops_override)
        if vector_ops_override is not None
        else float(elements) * float(ops_per_element)
    )
    default_read = float(elements) * bytes_per
    default_write = float(elements) * bytes_per
    read_bytes = (
        float(read_bytes_override)
        if read_bytes_override is not None else default_read
    )
    write_bytes = (
        float(write_bytes_override)
        if write_bytes_override is not None else default_write
    )
    if hbm_bytes_override is not None:
        hbm = float(hbm_bytes_override)
    elif read_bytes_override is not None or write_bytes_override is not None:
        hbm = read_bytes + write_bytes
    else:
        hbm = float(elements) * float(read_write) * bytes_per
    shape = shape or (elements,)
    meta_map = dict(metadata)
    input_format = str(meta_map.get("input_format", dtype))
    output_format = str(meta_map.get("output_format", dtype))
    return OpSpec(
        op_id=op_id, family=family, phase=phase, stage=stage, layer=layer,
        requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=(align_up(elements, 256),),
        launch_shape=(align_up(elements, 256),),
        vector_ops=vector_ops, hbm_bytes=hbm,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        deps=deps, recipe=recipe,
        notes=tuple(notes) + ("vector_work=elements*ops_per_element",),
        metadata=tuple(metadata) + (
            ("semantic_family", semantic_family or family),
            ("input_format", input_format),
            ("output_format", output_format),
        ) + (
            (("read_bytes", read_bytes), ("write_bytes", write_bytes))
            if read_bytes_override is not None or write_bytes_override is not None
            else ()
        ),
        dtype=dtype,
        returns=tuple(returns),
    )


_COMM_AXIS_NORMALIZE = {
    "cp": "dcp",
    "dcp": "dcp",
    "dcp_tp": "dcp",
    "dcp_dp": "dcp",
    "dcp_a2a": "dcp",
    "pcp": "pcp",
    "tp": "tp",
    "tp_sp": "tp_sp",
    "ep": "ep",
    "dp": "dp",
    "pp": "pp",
    "etp": "etp",
    "local_tp": "local_tp",
}


def _normalize_comm_metadata(
    family: str, metadata: tuple[tuple[str, Any], ...],
) -> tuple[tuple[str, Any], ...]:
    meta_map = dict(metadata)
    extra: list[tuple[str, Any]] = []
    if "comm_axis" not in meta_map:
        raw_axis = str(meta_map.get("group_axis") or "").strip().lower()
        if raw_axis == "tp" and "sequence_parallel" in meta_map:
            axis = "tp_sp"
        else:
            axis = _COMM_AXIS_NORMALIZE.get(raw_axis, raw_axis)
        if axis:
            extra.append(("comm_axis", axis))
    if "comm_group_size" not in meta_map:
        size = 0
        for key in ("moe_ep_size", "dcp_degree", "dcp_tp_gather", "dcp_dp_width", "local_tp_width"):
            val = meta_map.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and val > 1:
                size = int(val)
                break
        if not size:
            ranks = meta_map.get("group_ranks")
            if isinstance(ranks, (tuple, list)) and len(ranks) > 1:
                size = len(ranks)
        if not size:
            steps = meta_map.get("steps")
            if isinstance(steps, int) and not isinstance(steps, bool) and steps > 0:
                if family == "allreduce" and steps >= 2 and steps % 2 == 0:
                    size = (steps // 2) + 1
                else:
                    size = steps + 1
        if size > 1:
            extra.append(("comm_group_size", int(size)))
    return tuple(metadata) + tuple(extra) if extra else tuple(metadata)


def comm_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], family: str, phase: str, recipe: str,
    link_bytes: float, shape: tuple[int, ...], metadata: tuple[tuple[str, Any], ...] = (),
    hbm_bytes: float = 0.0, notes: tuple[str, ...] = (),
) -> OpSpec:
    return OpSpec(
        op_id=op_id, family=family, phase=phase, stage=stage, layer=layer,
        requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, link_bytes=float(max(0.0, link_bytes)),
        hbm_bytes=float(max(0.0, hbm_bytes)), deps=deps, recipe=recipe,
        notes=tuple(notes), metadata=_normalize_comm_metadata(family, metadata),
    )


def barrier_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], phase: str, recipe: str,
    shape: tuple[int, ...], metadata: tuple[tuple[str, Any], ...] = (),
    notes: tuple[str, ...] = (),
) -> OpSpec:
    return OpSpec(
        op_id=op_id, family="barrier", phase=phase, stage=stage, layer=layer,
        requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, deps=deps, recipe=recipe,
        notes=tuple(notes), metadata=metadata,
    )


def cache_store_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], phase: str, recipe: str,
    rows: int, row_bytes: int, dtype: str, kind: str,
    metadata: tuple[tuple[str, Any], ...] = (),
) -> OpSpec:
    rows = max(0, int(rows))
    elements = rows * max(1, int(row_bytes))
    shape = (max(1, rows), max(1, int(row_bytes)))
    return OpSpec(
        op_id=op_id, family="cache_store", phase=phase, stage=stage, layer=layer,
        requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=(align_up(max(1, rows), 8), max(1, int(row_bytes))),
        launch_shape=shape, vector_ops=float(elements),
        hbm_bytes=2 * float(rows) * float(row_bytes),
        mte2_bytes=float(rows) * float(row_bytes),
        mte3_bytes=float(rows) * float(row_bytes),
        deps=deps, recipe=recipe,
        notes=("cache_store_is_a_real_dependency",),
        metadata=tuple(metadata) + (
            ("cache_kind", kind), ("cache_visible", True),
            ("shared_owner", False),
        ),
        dtype=dtype,
    )


__all__ = [
    "compute_dtype_for", "packed_weight_bytes", "packed_weight_format",
    "precision_abi_for_gemm", "gemm_op", "vector_op", "comm_op", "barrier_op",
    "cache_store_op",
]
