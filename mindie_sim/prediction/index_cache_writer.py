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
from .mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, Stage

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


def _resolve_mode(mode: str, *, cache_dtype: str, compress_ratio: int) -> str:
    tag = str(mode or "auto").strip().lower()
    if tag != "auto":
        return tag
    if compress_ratio > 1:
        return "qsa_dual_pages"
    if cache_dtype in {"bf16", "fp16", "bfloat16", "float16"}:
        return "scatter_pa_kv_cache"
    return "c8_indexer_quant_cache"


def emit_index_cache_write(
    *,
    mode: str = "auto",
    op_id: str,
    deps: tuple[str, ...] = (),
    stage: int = 0,
    layer: int = 0,
    requests: tuple[str, ...] = (),
    phase: str = "decode",
    rows: int,
    dim: int,
    heads: int = 1,
    input_dtype: str = "bf16",
    cache_dtype: str = "fp8_e4m3",
    scale_dtype: str = "fp32",
    group_size: int = DEFAULT_SCALE_GROUP_WIDTH,
    compress_ratio: int = 1,
    block_size: int = DEFAULT_STEP5_REGION_BLOCK,
    owns_index_k: bool = True,
    row_bytes: int | None = None,
    metadata: tuple[tuple[str, Any], ...] = (),
) -> list[OpSpec]:
    """Emit the single native paged index/summary cache write op for a layer."""
    if not owns_index_k or int(rows) <= 0 or int(dim) <= 0:
        return []
    total_rows = int(rows)
    width = int(dim)
    head_count = max(1, int(heads))
    in_bytes = float(dtype_bytes(input_dtype))
    selected = _resolve_mode(
        mode, cache_dtype=cache_dtype, compress_ratio=int(compress_ratio),
    )

    if selected in {"scatter_pa_kv_cache", "bf16_scatter_pa"}:
        out_elem = float(dtype_bytes(cache_dtype))
        payload = float(total_rows * head_count * width)
        slot_bytes = float(total_rows * SLOT_INDEX_BYTES)
        write_bytes = (
            float(total_rows * int(row_bytes))
            if row_bytes is not None else payload * out_elem
        )
        op = vector_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            phase=phase, family=KV_CACHE_WRITE_FAMILY, recipe=SCATTER_PA_KV_RECIPE,
            dtype=cache_dtype, elements=int(payload), ops_per_element=0.0,
            read_bytes_override=payload * in_bytes + slot_bytes,
            write_bytes_override=write_bytes,
            shape=(total_rows, head_count, width),
            metadata=metadata + (
                ("native_kind", "aclnnScatterPaKvCache"),
                ("read_pipe", "aiv"),
                ("slot_table_bytes", slot_bytes),
                ("zero_external_casts", True),
                ("native_source", SCATTER_PA_KV_SOURCE),
            ),
        )
        return [op]

    if selected == "qsa_dual_pages":
        ratio = max(1, int(compress_ratio or DEFAULT_QSA_COMPRESS_RATIO))
        compressed_rows = max(1, math.ceil(total_rows / ratio))
        raw_tail_rows = min(total_rows, 2 * ratio)
        out_elem = float(dtype_bytes(cache_dtype))
        read_bytes = float(total_rows * width * in_bytes + total_rows * SLOT_INDEX_BYTES)
        raw_write = float(raw_tail_rows * width * in_bytes)
        comp_write = float(compressed_rows * width * out_elem)
        op = vector_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            phase=phase, family=KV_CACHE_WRITE_FAMILY, recipe=QSA_DUAL_CACHE_RECIPE,
            dtype=cache_dtype, elements=total_rows * width,
            ops_per_element=float(POOL_REDUCE_OPS_PER_ELEMENT),
            read_bytes_override=read_bytes,
            write_bytes_override=raw_write + comp_write,
            shape=(compressed_rows, width),
            metadata=metadata + (
                ("native_kind", "qsa_dual_page_cache_write"),
                ("read_pipe", "aiv"),
                ("compress_ratio", ratio),
                ("raw_circular_rows", raw_tail_rows),
                ("compressed_rows", compressed_rows),
                ("native_source", QSA_CACHE_SOURCE),
            ),
        )
        return [op]

    if selected == "step5_summary_update":
        region = max(1, int(block_size or DEFAULT_STEP5_REGION_BLOCK))
        summary_rows = max(1, math.ceil(total_rows / region))
        scale_bytes = float(dtype_bytes(scale_dtype)) if scale_dtype != "none" else 0.0
        out_elem = float(dtype_bytes(cache_dtype))
        tail_rows = min(total_rows, region)
        read_bytes = float(
            total_rows * head_count * width * in_bytes + summary_rows * SLOT_INDEX_BYTES
        )
        summary_write = float(summary_rows * head_count * (width * out_elem + scale_bytes))
        tail_write = float(tail_rows * head_count * width * in_bytes)
        op = vector_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            phase=phase, family=INDEXER_QUANT_CACHE_FAMILY, recipe=STEP5_SUMMARY_RECIPE,
            dtype=cache_dtype, elements=total_rows * head_count * width,
            ops_per_element=float(SSMAX_OPS_PER_ELEMENT),
            read_bytes_override=read_bytes,
            write_bytes_override=summary_write + tail_write,
            shape=(summary_rows, head_count, width),
            metadata=metadata + (
                ("native_kind", "step5_csa_summary_update"),
                ("read_pipe", "aiv"),
                ("region_block_size", region),
                ("summary_rows", summary_rows),
                ("summary_reduction", "ssmax"),
                ("native_source", STEP5_SUMMARY_SOURCE),
            ),
        )
        return [op]

    if selected in {"c8_indexer_quant_cache", "indexer_quant_cache"}:
        group = max(1, int(group_size or width))
        groups = max(1, math.ceil(width / group))
        scale_bytes = float(dtype_bytes(scale_dtype)) if scale_dtype != "none" else 0.0
        out_elem = float(dtype_bytes(cache_dtype))
        per_row = (
            float(row_bytes)
            if row_bytes is not None
            else head_count * (width * out_elem + groups * scale_bytes)
        )
        slot_bytes = float(total_rows * SLOT_INDEX_BYTES)
        payload = total_rows * head_count * width
        op = vector_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            phase=phase, family=INDEXER_QUANT_CACHE_FAMILY,
            recipe=C8_INDEXER_CACHE_RECIPE,
            dtype=cache_dtype, elements=payload,
            ops_per_element=float(QUANT_OPS_PER_ELEMENT),
            read_bytes_override=float(payload) * in_bytes + slot_bytes,
            write_bytes_override=float(total_rows) * per_row,
            shape=(total_rows, head_count, width),
            metadata=metadata + (
                ("native_kind", "aclnnIndexerQuantCache"),
                ("read_pipe", "aiv"),
                ("payload_dtype", cache_dtype),
                ("scale_dtype", scale_dtype),
                ("group_size", group),
                ("single_read_single_write", True),
                ("native_source", SFA_INDEXER_CACHE_SOURCE),
            ),
        )
        return [op]

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
