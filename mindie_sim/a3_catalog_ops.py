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

from .a3_catalog_types import QuantSpec
from .a3_ops import OpSpec, _op
from .a3_types import align_up, dtype_bytes


def _safe_bytes(dtype: str) -> float:
    try:
        return float(dtype_bytes(dtype))
    except Exception:
        return 2.0


def compute_dtype_for(quant: QuantSpec, category: str) -> tuple[str, str]:
    """Return (compute dtype, declared activation dtype) for a category."""
    fmt = quant.format_for(category)
    if fmt in {"fp32", "float32"}:
        return "fp32", "fp32"
    if fmt in {"bf16", "bfloat16", "fp16", "float16"}:
        return "bf16", "bf16"
    if fmt == "int8":
        return "bf16", "bf16"
    if fmt in {"mxfp4", "fp4", "fp4_e2m1"}:
        # Explicit dequant/accumulation path; not an FP4 peak claim.
        return "bf16", "bf16"
    if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        act = quant.activation_format
        if act in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            return "fp8_e4m3", "fp8_e4m3"
        return "bf16", "bf16"
    return "bf16", "bf16"


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
    m, n, k = max(1, int(m)), max(1, int(n)), max(1, int(k))
    compute, act = compute_dtype_for(quant, category)
    act = activation_dtype or act
    out_dtype = output_dtype or act
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
    input_bytes = float(m * k) * act_bytes
    output_bytes = float(m * n) * out_bytes
    hbm = input_bytes + weight_bytes + output_bytes
    flops = 2.0 * m * n * k
    fmt = packed_weight_format(quant, category)
    recipe = recipe or semantic_family or family
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
        ("weight_representation", _layout_name(fmt, quant)),
        ("dequant_before_compute", fmt in {"mxfp4", "fp4", "fp4_e2m1", "int8"}),
    )
    return _op(
        op_id=op_id, family=family, phase=phase, stage=stage, layer=layer,
        requests=requests,
        semantic_shape=semantic_shape or (m, n, k),
        local_shape=(m, n, k),
        padded_shape=(align_up(m, 16), align_up(n, 16), align_up(k, 16)),
        launch_shape=(m, n, k),
        flops=flops, hbm_bytes=hbm, mte2_bytes=input_bytes + weight_bytes,
        mte3_bytes=output_bytes, deps=deps, recipe=recipe,
        notes=tuple(notes) + ("gemm_formula=2*M*N*K",
                              "weight_bytes_use_declared_packing"),
        metadata=meta, dtype=act, layout=_layout_name(fmt, quant),
    )


def _layout_name(fmt: str, quant: QuantSpec) -> str:
    if fmt in {"mxfp4", "fp4", "fp4_e2m1"}:
        return f"fp4_group{max(1, quant.expert_block)}_packed"
    if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        if quant.weight_block[0] > 0:
            return f"fp8_block{quant.weight_block[0]}x{quant.weight_block[1]}_{quant.scale_format}"
        return "fp8_rowwise"
    if fmt in {"fp32", "float32"}:
        return "fp32"
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
    notes: tuple[str, ...] = (),
) -> OpSpec:
    elements = max(1, int(elements))
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
    return _op(
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
    )


def comm_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], family: str, phase: str, recipe: str,
    link_bytes: float, shape: tuple[int, ...], metadata: tuple[tuple[str, Any], ...] = (),
    hbm_bytes: float = 0.0, notes: tuple[str, ...] = (),
) -> OpSpec:
    return _op(
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
    return _op(
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
    elements = max(1, rows * max(1, int(row_bytes)))
    shape = (max(1, rows), max(1, int(row_bytes)))
    return _op(
        op_id=op_id, family="cache_store", phase=phase, stage=stage, layer=layer,
        requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=(align_up(max(1, rows), 8), max(1, int(row_bytes))),
        launch_shape=shape, vector_ops=float(elements),
        hbm_bytes=float(rows) * float(row_bytes),
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
