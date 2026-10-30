"""Catalog op builders with exact quantized weight/activation byte accounting.

[Role]
  Construct OpSpec records for catalog lowering.  The ordinary generic
  constructors assume dense BF16 weights; these builders distinguish packed
  FP4/FP8 weight bytes, activation bytes, compute dtype, output bytes and
  collective payload so calibration can identify the semantic path.

[Invariants]
  * GEMM work remains 2*M*N*K; packed weight bytes never change logical FLOPs.
  * FP4 experts are dequantized to a declared compute format; no unsupported
    FP4 hardware peak is inferred.
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

from typing import Any, Mapping, Sequence

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes


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
        # W4A8 streams packed int4 weights through the int8 cube (int32
        # accumulate, bf16 out); the fused kernel dequantizes in-kernel, so the
        # op's compute identity is int8 and only its weight stream is 0.5 B.
        # Leaving int4 on the bf16 fallback priced every W4A8 expert GEMM on
        # the BF16 cube at 2x its real compute rate.
        return "int8", "int8"
    if fmt in {"mxfp4", "fp4", "fp4_e2m1"}:
        # Explicit dequant/accumulation path; not an FP4 peak claim.
        return "bf16", quant.activation_format
    if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        act = quant.activation_format
        if act in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            return "fp8_e4m3", "fp8_e4m3"
        return "bf16", "bf16"
    raise ModelingError(f"compute_format_undeclared:{category}:{fmt}")


def packed_weight_bytes(
    quant: QuantSpec, rows: int, cols: int, category: str,
    *, count: int = 1, source: bool = False,
) -> float:
    return float(quant.matrix_bytes(rows, cols, category, source=source)) * float(max(1, count))


def packed_weight_format(quant: QuantSpec, category: str) -> str:
    return quant.format_for(category)


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
    notes: tuple[str, ...] = (),
) -> OpSpec:
    quant = quant.for_category(category)
    m, n, k = int(m), int(n), int(k)
    if m < 0 or min(n, k) < 1:
        raise ModelingError(f"matrix_geometry_invalid:{op_id}:{m},{n},{k}")
    compute, act = compute_dtype_for(quant, category)
    act = activation_dtype or act
    # MXFP8 computes in FP8 but stores BF16 unless the consumer declares a
    # different output format.  FP8 input dtype is not the output contract.
    out_dtype = output_dtype or (
        "bf16" if act in {"fp8", "fp8_e4m3", "fp8_e5m2", "int8"} else act
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
    recipe = recipe or semantic_family or family
    packed = fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}
    meta = tuple(metadata) + (
        ("semantic_family", semantic_family or family),
        ("weight_format", fmt),
        ("weight_dtype", fmt),
        ("activation_format", act),
        ("input_format", act),
        ("output_format", out_dtype),
        ("compute_format", compute),
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
    if packed:
        existing_meta = dict(metadata)
        scale_meta = ()
        if "weight_scale_format" not in existing_meta:
            scale_meta += (("weight_scale_format", quant.scale_format),)
        if "weight_scale_bytes" not in existing_meta:
            scale_meta += (("weight_scale_bytes", quant.scale_bytes),)
        meta = meta + scale_meta + (
            ("dequant", "fused" if fmt in {"mxfp4", "fp4", "fp4_e2m1", "int4", "w4a8"} else "none"),
            ("dequant_traffic", 0.0),
            ("weight_bytes", weight_bytes),
            ("implementation_assumption", "declared_matrix_operand_formats"),
        )
    vector_ops = 0.0
    if (fmt in {"int8", "w8a8"} and act == "int8" and group >= k
            and out_dtype in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}):
        # Per-row dynamic quantization rescales the completed INT32 reduction.
        # Retain the declared scale+offset storage traffic, but do not invent
        # offset arithmetic: the dynamic caller passes weight/row scales only.
        column_metadata = weight_bytes * (2 * quant.scale_bytes) / (
            int(weight_cols or k) + 2 * quant.scale_bytes)
        operations = 3 if out_dtype in {"fp32", "float32"} else 4
        vector_ops = float(operations * m * n)
        meta += (("matrix_epilogue", {
            "kind": "row_column_scale", "row_scale_bytes": float(scale_bytes),
            "column_scale_bytes": column_metadata, "operations_per_output": operations,
        }), ("matrix_output_contract", "int32_cast_row_scale_column_scale_output_cast"),
            ("column_metadata_policy", "retains_declared_scale_and_offset_storage_traffic"))
    return OpSpec(
        op_id=op_id, family=family, phase=phase, stage=stage, layer=layer,
        requests=requests,
        semantic_shape=semantic_shape or (m, n, k),
        local_shape=(m, n, k),
        padded_shape=(align_up(m, 16), align_up(n, 16), align_up(k, 16)),
        launch_shape=(m, n, k),
        flops=flops, vector_ops=vector_ops, hbm_bytes=hbm,
        mte2_bytes=input_bytes + weight_bytes + scale_bytes,
        mte3_bytes=output_bytes, deps=deps, recipe=recipe,
        notes=tuple(notes) + ("gemm_formula=2*M*N*K",
                              "weight_bytes_use_declared_packing"),
        # The cube op's physical identity is its *compute* dtype: a W8A8 op is
        # an int8 kernel even though it consumes/produces bf16 activations.
        # Byte widths stay on ``act``/``out_dtype`` above.
        metadata=meta + (("program_kind", "matmul"),), dtype=compute, layout=_layout_name(fmt, quant),
    )


def activation_scales(quant, dtype, rows, cols):
    """Activation scales follow the declared quantization, not weight bytes."""
    if dtype == "int8":
        group = quant.activation_group_size or cols
        groups = (cols + group - 1) // group
        return group, "fp32", (rows,) if groups == 1 else (rows, groups), float(rows * groups * 4)
    if dtype in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
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
        notes=tuple(notes), metadata=metadata,
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
    "gemm_op", "vector_op", "comm_op", "barrier_op", "cache_store_op",
]
