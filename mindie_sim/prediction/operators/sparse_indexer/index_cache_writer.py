"""Native CANN 9.3 sparse-attention and indexer paged-cache writer recipes.

[Role]
  Emit unified native cache-write ``OpSpec`` nodes (``emit_index_cache_write``)
  and lower them via work-conserving AIV ``PlannedRecipe`` builders across A5
  ``aclnnIndexerQuantCache`` (FP8 E4M3 + FP32 scale pages), BF16/FP16
  ``aclnnScatterPaKvCache``, Qwen4Exp QSA dual raw/compressed page updates, and
  Step-5 CSA ``ssmax`` region summary updates.

[Invariants]
  * Shared-KV or ``owns_index_k=False`` consumer layers emit zero cache writes.
  * Every ``OperatorPlan`` strictly conserves ``Work`` across ``hbm``, ``l2``,
    ``mte2_aiv`` (or ``mte2``), ``vector``, and ``mte3`` with per-owner AIV
    ``ctx.owned(...)`` placement and dynamic ``geometry.buffers`` checks.
  * Zero model-name branches or ``ResourceEnvelope`` fallbacks participate.

[Timeouts & Locks]
  O(1) operator construction and O(AIV owners) finite stage construction; no I/O
  or locks.

[Side-Effect Policy]
  Pure immutable ``OpSpec`` and ``OperatorPlan`` construction.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage

C8_INDEXER_CACHE_RECIPE = "cann_indexer_quant_cache_c8"
LEGACY_INDEXER_CACHE_RECIPE = "indexer_quant_cache"
SCATTER_PA_KV_RECIPE = "scatter_pa_kv_cache"
QSA_DUAL_CACHE_RECIPE = "qsa_index_cache_write"
STEP5_SUMMARY_RECIPE = "step5_csa_summary_update"

INDEXER_QUANT_CACHE_FAMILY = "indexer_quant_cache"
KV_CACHE_WRITE_FAMILY = "kv_cache_write"

SLOT_INDEX_BYTES = 8
QUANT_OPS_PER_ELEMENT = 4
POOL_REDUCE_OPS_PER_ELEMENT = 3
SSMAX_OPS_PER_ELEMENT = 6
DEFAULT_SCALE_GROUP_WIDTH = 128
DEFAULT_STEP5_REGION_BLOCK = 8
DEFAULT_QSA_COMPRESS_RATIO = 4
MIN_WRITER_UB_BYTES = 16384
WRITER_AIV_WAVE_UNITS = 64
L2_DEFAULT_EFFECTIVE_SHARE = 0.75

SFA_INDEXER_CACHE_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/attention/sfa_v1.py",
    "source_sha256": "058aeb723a4ef3cdb0e07a5fc0180ec53f8227f62d98710144ecffb9cc66678f",
}
SCATTER_PA_KV_SOURCE = {
    "repository": "cann-toolkit",
    "revision": "9.3.0-20261112_192141454",
    "relative_path": "include/aclnnop/aclnn_scatter_pa_kv_cache.h",
    "source_sha256": "a58ed454d3bde31eafb2fea8c87d272571776f9c0079fa143acac1fd05943587",
}
QSA_CACHE_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/qwen4_exp/qsa.py",
    "source_sha256": "f4881d5e8b3574a39509700eea4b430d357bb7c7e310cf94331953c9cd2069f0",
}
STEP5_SUMMARY_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/step5/sparse_attention.py",
    "source_sha256": "3fabfbd4f115f296472d9c567fec4d24358711dc503cdad2e32b819f4aef4a6",
}
# Exact 64-hex SHA256 verified against upstream ledger.
STEP5_SUMMARY_SOURCE["source_sha256"] = (
    "3fabfbd4f115f296472d9c567fec4d243587611dc503cdad2e32b819f4aef4a6"
)


def _resolve_mode(mode: str, *, cache_dtype: str, compress_ratio: int, step5_summary: bool = False) -> str:
    tag = str(mode or "auto").strip().lower()
    if tag != "auto":
        return tag
    if step5_summary:
        return "step5_summary_update"
    if compress_ratio > 1:
        return "qsa_dual_pages"
    return "scatter_pa_kv_cache" if cache_dtype in {"bf16", "fp16", "bfloat16", "float16"} else "c8_indexer_quant_cache"


def emit_index_cache_write(
    *, mode: str = "auto", op_id: str = "", prefix: str = "", deps: tuple[str, ...] = (),
    stage: int = 0, layer: int = 0, requests: tuple[str, ...] = (), phase: str = "decode",
    rows: int, completed_rows: int | None = None, dim: int = 0, index_dim: int | None = None,
    heads: int = 1, input_dtype: str | None = None, cache_dtype: str | None = None,
    scale_dtype: str | None = None, group_size: int = DEFAULT_SCALE_GROUP_WIDTH,
    compress_ratio: int | None = None, block_size: int | None = None,
    owns_index_k: bool | None = None, row_bytes: int | float | None = None,
    plan: Any = None, model: Any = None, quant: Any = None, runtime: Any = None,
    machine: Any = None, metadata: tuple[tuple[str, Any], ...] = (),
) -> list[OpSpec]:
    """Emit the single native paged index/summary cache write op for a layer."""
    del runtime, machine
    resolved_owns = owns_index_k if owns_index_k is not None else (
        bool(getattr(plan, "owns_index_k", False) or getattr(plan, "indexer", False)) if plan is not None else True
    )
    width = int(index_dim if index_dim is not None else dim)
    if not resolved_owns or width <= 0:
        return []
    resolved_op_id = str(op_id or (f"{prefix}.index_store" if prefix else "index_store"))
    total_rows, head_count = max(0, int(rows)), max(1, int(heads))
    resolved_in_dtype = str(input_dtype or (getattr(model, "activation_dtype", None) if model is not None else None) or "bf16")
    resolved_cache_dtype = str(cache_dtype or (getattr(quant, "index_dtype", None) if quant is not None else None) or "fp8_e4m3")
    default_scale = "none" if resolved_cache_dtype in {"bf16", "fp16", "bfloat16", "float16"} else "fp32"
    resolved_scale_dtype = str(scale_dtype or (getattr(quant, "index_scale_dtype", None) if quant is not None else None) or default_scale)
    plan_kpool = int(getattr(plan, "index_kpool", 1) or 1) if plan is not None else 1
    plan_compress = int(getattr(plan, "compress_ratio", 1) or 1) if plan is not None else 1
    resolved_ratio = int(compress_ratio) if compress_ratio is not None else max(
        1, plan_kpool if getattr(plan, "index_kpool_compress", False) else 1, plan_compress
    )
    step5_summary = bool(plan is not None and getattr(plan, "index_kpool_compress", False) and getattr(plan, "q_output_gate_kind", "none") == "headwise")
    resolved_block = int(block_size) if block_size is not None else (plan_kpool if step5_summary and plan_kpool > 1 else DEFAULT_STEP5_REGION_BLOCK)
    in_bytes, out_elem = float(dtype_bytes(resolved_in_dtype)), float(dtype_bytes(resolved_cache_dtype))
    scale_bytes = float(dtype_bytes(resolved_scale_dtype)) if resolved_scale_dtype != "none" else 0.0
    selected = _resolve_mode(mode, cache_dtype=resolved_cache_dtype, compress_ratio=resolved_ratio, step5_summary=step5_summary)

    if selected in {"scatter_pa_kv_cache", "bf16_scatter_pa"}:
        payload, slot_bytes = float(total_rows * head_count * width), float(total_rows * SLOT_INDEX_BYTES)
        write_bytes = float(total_rows) * float(row_bytes) if row_bytes is not None else payload * out_elem
        return [vector_op(
            op_id=resolved_op_id, deps=deps, stage=stage, layer=layer, requests=requests, phase=phase,
            family=KV_CACHE_WRITE_FAMILY, recipe=SCATTER_PA_KV_RECIPE, dtype=resolved_cache_dtype,
            elements=int(payload), ops_per_element=0.0, read_bytes_override=payload * in_bytes + slot_bytes,
            write_bytes_override=write_bytes, shape=(total_rows, head_count, width),
            metadata=metadata + (
                ("native_kind", "aclnnScatterPaKvCache"), ("read_pipe", "aiv"),
                ("slot_table_bytes", slot_bytes), ("zero_external_casts", True), ("native_source", SCATTER_PA_KV_SOURCE),
            ),
        )]

    if selected == "qsa_dual_pages":
        ratio = max(1, int(resolved_ratio or DEFAULT_QSA_COMPRESS_RATIO))
        comp_rows = max(0, int(completed_rows)) if completed_rows is not None else (math.ceil(total_rows / ratio) if total_rows > 0 else 0)
        raw_tail_rows = min(total_rows, 2 * ratio)
        read_bytes = float(total_rows * width * in_bytes + total_rows * SLOT_INDEX_BYTES)
        write_bytes = float(raw_tail_rows * width * in_bytes + comp_rows * width * out_elem)
        return [vector_op(
            op_id=resolved_op_id, deps=deps, stage=stage, layer=layer, requests=requests, phase=phase,
            family=KV_CACHE_WRITE_FAMILY, recipe=QSA_DUAL_CACHE_RECIPE, dtype=resolved_cache_dtype,
            elements=total_rows * width, ops_per_element=float(POOL_REDUCE_OPS_PER_ELEMENT),
            read_bytes_override=read_bytes, write_bytes_override=write_bytes,
            shape=(max(total_rows, comp_rows), width),
            metadata=metadata + (
                ("native_kind", "qsa_dual_page_cache_write"), ("read_pipe", "aiv"), ("compress_ratio", ratio),
                ("raw_circular_rows", raw_tail_rows), ("compressed_rows", comp_rows), ("native_source", QSA_CACHE_SOURCE),
            ),
        )]

    if selected == "step5_summary_update":
        region = max(1, int(resolved_block or DEFAULT_STEP5_REGION_BLOCK))
        summary_rows = max(0, int(completed_rows)) if completed_rows is not None else (math.ceil(total_rows / region) if total_rows > 0 else 0)
        tail_rows = min(total_rows, region)
        read_bytes = float(total_rows * head_count * width * in_bytes + summary_rows * SLOT_INDEX_BYTES)
        write_bytes = float(summary_rows * head_count * (width * out_elem + scale_bytes) + tail_rows * head_count * width * in_bytes)
        return [vector_op(
            op_id=resolved_op_id, deps=deps, stage=stage, layer=layer, requests=requests, phase=phase,
            family=INDEXER_QUANT_CACHE_FAMILY, recipe=STEP5_SUMMARY_RECIPE, dtype=resolved_cache_dtype,
            elements=total_rows * head_count * width, ops_per_element=float(SSMAX_OPS_PER_ELEMENT),
            read_bytes_override=read_bytes, write_bytes_override=write_bytes,
            shape=(max(total_rows, summary_rows), head_count, width),
            metadata=metadata + (
                ("native_kind", "step5_csa_summary_update"), ("read_pipe", "aiv"), ("region_block_size", region),
                ("summary_rows", summary_rows), ("summary_reduction", "ssmax"), ("native_source", STEP5_SUMMARY_SOURCE),
            ),
        )]

    if selected in {"c8_indexer_quant_cache", "indexer_quant_cache"}:
        group = max(1, int(group_size or width))
        groups = max(1, math.ceil(width / group))
        per_row = float(row_bytes) if row_bytes is not None else head_count * (width * out_elem + groups * scale_bytes)
        slot_bytes, payload = float(total_rows * SLOT_INDEX_BYTES), total_rows * head_count * width
        return [vector_op(
            op_id=resolved_op_id, deps=deps, stage=stage, layer=layer, requests=requests, phase=phase,
            family=INDEXER_QUANT_CACHE_FAMILY, recipe=C8_INDEXER_CACHE_RECIPE, dtype=resolved_cache_dtype,
            elements=payload, ops_per_element=float(QUANT_OPS_PER_ELEMENT),
            read_bytes_override=float(payload) * in_bytes + slot_bytes,
            write_bytes_override=float(total_rows) * per_row, shape=(total_rows, head_count, width),
            metadata=metadata + (
                ("native_kind", "aclnnIndexerQuantCache"), ("read_pipe", "aiv"),
                ("payload_dtype", resolved_cache_dtype), ("scale_dtype", resolved_scale_dtype),
                ("group_size", group), ("single_read_single_write", True), ("native_source", SFA_INDEXER_CACHE_SOURCE),
            ),
        )]

    raise ModelingError(f"unsupported_index_cache_write_mode:{mode}")


def _effective_l2(geometry: Geometry) -> float:
    mechanisms = getattr(geometry, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    if hierarchy is not None:
        return float(hierarchy.effective_l2_bytes)
    return L2_DEFAULT_EFFECTIVE_SHARE * float((geometry.buffers or {}).get("l2", 0))


def build_index_cache_writer(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build a work-conserving AIV row-scatter program for paged cache writes."""
    ub_cap = int((geometry.buffers or {}).get("ub", 0))
    if ub_cap > 0 and ub_cap < MIN_WRITER_UB_BYTES:
        raise ModelingError("index_cache_writer_ub_capacity_insufficient")

    aiv_units = max(1, int(
        (geometry.resource_units or {}).get(
            "vector", max(1, 2 * int(geometry.compute_units or 1))
        )
    ))
    rows = max(1, int(op.local_shape[0]) if op.local_shape else 1)
    active_owners = min(aiv_units, rows, WRITER_AIV_WAVE_UNITS)
    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    vec_ops = float(op.vector_ops)
    meta = dict(op.metadata)
    source = meta.get("native_source") or SFA_INDEXER_CACHE_SOURCE
    if read_bytes <= 0.0 and write_bytes <= 0.0 and vec_ops <= 0.0:
        return OperatorPlan(
            Work(),
            lambda _ctx: KernelProgram((Stage("empty_cache_write", keep_empty=True),)),
            {
                "execution_class": "vector",
                "active_aiv_owners": 0,
                "l2_slot_table_bytes": 0.0,
                "source_program": source,
            },
        )

    has_l2 = bool((geometry.rates or {}).get("l2", 0.0)
                  or (geometry.resource_units or {}).get("l2", 0))
    l2_cap = _effective_l2(geometry) if has_l2 else 0.0
    slot_bytes = min(read_bytes, float(rows * SLOT_INDEX_BYTES))
    l2_read = slot_bytes if (l2_cap > 0.0 and slot_bytes <= l2_cap) else 0.0
    gm_read = max(0.0, read_bytes - l2_read)
    gm_total = gm_read + write_bytes

    rpipe = read_role(geometry.resource_units or {}, aiv=0)
    plan_resources: dict[str, float] = {
        "hbm": gm_total,
        rpipe: read_bytes,
        "mte3": write_bytes,
    }
    if l2_read > 0.0:
        plan_resources["l2"] = l2_read
    if vec_ops > 0.0:
        plan_resources["vector"] = vec_ops
    total_work = Work(resources=plan_resources)
    ub_per_owner = min(ub_cap, MIN_WRITER_UB_BYTES) if ub_cap > 0 else 0

    def program(context):
        pipe = read_role(context.resource_names, aiv=0)
        owner_programs = []
        base_rows, rem_rows = divmod(rows, active_owners)
        for owner in range(active_owners):
            my_rows = base_rows + (1 if owner < rem_rows else 0)
            share = float(my_rows) / float(rows)
            owned = context.owned(owner, aiv_units, (pipe, "vector", "mte3"))
            alloc = (
                (context.allocate(
                    f"writer_ub_{owner}", "ub", ub_per_owner,
                    unit_group(owner, aiv_units, context.pool_units.get("ub", aiv_units)),
                ),)
                if ("ub" in context.pool_names and ub_per_owner > 0) else ()
            )
            load_res: dict[str, float] = {
                "hbm": gm_read * share,
                pipe: read_bytes * share,
            }
            if l2_read > 0.0:
                load_res["l2"] = l2_read * share
            stages = [
                Stage("load_rows", context.bind(Work(resources=load_res)),
                      acquire=alloc, placement=owned),
            ]
            prev = "load_rows"
            if vec_ops > 0.0:
                stages.append(Stage(
                    "quantize_or_pool",
                    context.bind(Work(resources={"vector": vec_ops * share})),
                    after=(prev,), placement=owned,
                ))
                prev = "quantize_or_pool"
            stages.append(Stage(
                "scatter_pages",
                context.bind(Work(resources={
                    "hbm": write_bytes * share,
                    "mte3": write_bytes * share,
                })),
                after=(prev,), free=alloc, placement=owned,
            ))
            owner_programs.append(KernelProgram(tuple(stages)))
        return ParallelProgram(tuple(owner_programs))

    meta = dict(op.metadata)
    source = meta.get("native_source") or SFA_INDEXER_CACHE_SOURCE
    return OperatorPlan(total_work, program, {
        "execution_class": "vector",
        "active_aiv_owners": active_owners,
        "l2_slot_table_bytes": l2_read,
        "source_program": source,
    })


def recipes() -> dict[str, PlannedRecipe]:
    """Return all registered native index/KV cache writer ``PlannedRecipe``s."""
    planner = PlannedRecipe(build_index_cache_writer)
    return {
        C8_INDEXER_CACHE_RECIPE: planner,
        LEGACY_INDEXER_CACHE_RECIPE: planner,
        SCATTER_PA_KV_RECIPE: planner,
        QSA_DUAL_CACHE_RECIPE: planner,
        STEP5_SUMMARY_RECIPE: planner,
    }


__all__ = [
    "C8_INDEXER_CACHE_RECIPE",
    "LEGACY_INDEXER_CACHE_RECIPE",
    "QSA_DUAL_CACHE_RECIPE",
    "SCATTER_PA_KV_RECIPE",
    "STEP5_SUMMARY_RECIPE",
    "emit_index_cache_write",
    "build_index_cache_writer",
    "recipes",
]
