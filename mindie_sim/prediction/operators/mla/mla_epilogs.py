"""Latent attention cache epilogs and index selection lowering.

[Role] Own the fused cache epilogs and the index-key path of latent attention.
[Invariants] A fused epilog replaces one producer write plus one cache read; it
  never removes a value that another consumer still reads.  The native indexer
  consumes checkpoint BF16 head weights directly, with no Cast wrapper.  A
  cross-step TopK reuse keeps every per-step KPool producer and skips only the
  selection, preserving arithmetic per produced cache row.
[Timeouts & Locks] O(views + index pools) arithmetic; no I/O or locks.
[Side-Effect Policy] Rewrites only the caller's builder operator list.
"""
from dataclasses import replace
import math
from typing import Literal

from mindie_sim.catalog.cache import completed_rows, index_row_bytes, index_selection_cost
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.context_parallel import dcp_indexer_sharded, layer_dcp_degree
from mindie_sim.catalog.ops import compute_dtype_for, gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.sharding.context_parallel import (
    CP_DECODE_COMM_STREAM,
    SHARDED_A3_CAST_OPS_PER_SORT,
    SHARDED_FUSED_SORT_TOPK,
    SHARDED_SORT_STAGES_PER_LAYER,
    emit_pcp_hidden_gather_restore,
    sharded_candidate_allgather_tiles,
    sharded_fused_sort_eligible,
    sharded_indexer_tile_rows,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef

from .generic_latent_attention import (
    index_cache_dtype, index_cache_is_packed, owned_completed_rows,
    owned_store_rows, sharded_index_visible_rows, semantic_index_selection,
)
from mindie_sim.prediction.implementations import produced_values, select, set_input_ports
from mindie_sim.prediction.operators.sparse_indexer.indexer_representation import index_representation
from .mla_native import index_layer_norm, index_rms_norm
from .mla_structure import declare_indexer

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/models/deepseek_v41/indexer.py",
        "45ec4290a41c25497d3cadfdb40c7466b5a40ae467a76200d82d4fa34b15fa02",
    ),
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/attention/sfa_v1.py",
        "058aeb723a4ef3cdb0e07a5fc0180ec53f8227f62d98710144ecffb9cc66678f",
    ),
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/indexer_quant_cache/op_host/indexer_quant_cache_contract.h",
        "c62bd82555308117e4a601865253ddbdd6c223cc8cd907be28090270816fac4c",
    ),
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/quant_lightning_indexer/op_host/quant_lightning_indexer_tiling.cpp",
        "54fb29851ffb840eb80c92bdf1592c995a3cf7388cefbc6e5ce89ee17753b3fa",
    ),
)

_IQC_INPUT_DTYPES = frozenset({"fp16", "bf16"})
_IQC_FP8_DTYPES = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2"})
_IQC_NORMAL_DTYPES = _IQC_FP8_DTYPES | frozenset({"hifloat8", "hif8", "uint8"})
_IQC_FP4_DTYPES = frozenset({"fp4", "fp4_e2m1", "fp4_e1m2"})
_IQC_HIFP4_DTYPES = frozenset({"hifp4", "hifloat4"})
_IQC_E8M0_SCALES = frozenset({"fp8_e8m0", "e8m0", "uint8"})
_IQC_FP32_SCALES = frozenset({"fp32", "float32"})
_IQC_HIFP4_SCALES = frozenset({"hifp4_scale", "hifloat4_scale", "uint8"})
_PACKED_FP4_ELEMENTS = 2

INDEXER_QUANT_CACHE = "aclnnIndexerQuantCache"
INDEXER_TOP_K_KIND = "aclnnQuantLightningIndexerV2"
KV_COMPRESS_EPILOG = "aclnnKvCompressEpilog"
KV_SCATTER_EPILOG = "aclnnScatterPaKvCache"
# Production-worker families: a fused epilog is not an eager decomposition, so
# it must never appear as the generic ``cache_store`` boundary.
KV_COMPRESS_EPILOG_FAMILY = "kv_compress_epilog"
INDEXER_QUANT_CACHE_FAMILY = "indexer_quant_cache"
BLOCK_QUANT_KIND = "aclnnDynamicBlockQuant"
BLOCK_QUANT_TILE = 128
INDEXER_QUANT_CACHE_RECIPE = "cann_indexer_quant_cache_c8"
FP32_BYTES = int(dtype_bytes("fp32"))
FP16_BYTES = int(dtype_bytes("fp16"))
BF16_BYTES = int(dtype_bytes("bf16"))
FP32_WEIGHT_DTYPE = "fp32"
INT32_BYTES = int(dtype_bytes("int32"))
INT64_BYTES = int(dtype_bytes("int64"))
ROPE_OPS = 6
CANDIDATE_BYTES = 8
SHARDED_PREPARE_VECTOR_OPS = 4
SHARDED_PACK_VECTOR_OPS = 4
SHARDED_GATHER_VECTOR_OPS = 1
SHARDED_FUSED_SORT_BITONIC_HALF = 0.5
SHARDED_RADIX_ID_BITS = 20
SHARDED_RADIX_DIGIT_BITS = 13
SHARDED_SCORE_BITS_BF16 = 16
SHARDED_SCORE_BITS_FP32 = 32
SHARDED_RADIX_GATHER_STREAMS = 2
SHARDED_NON_PRESORTED_UNPACK_READ_MULT = 2
C2_SCATTER_COORD_INTS = 2
MXFP4_INDEXER_NATIVE_KIND = "quantize_mxfp4_indexer"
MIN_LOG_TOPK = 2
M10_INTERLEAVE_ALIGN = 128
SMLA_METADATA_SIZE_A23 = 1024
SMLA_METADATA_SIZE_A5 = 2048
AI_CPU_METADATA_DEFAULT_ELEMS = 512
AI_CPU_METADATA_OPS_PER_ELEM = 4


def validate_m10_mla_legality(
    *,
    plan,
    eff_dcp: int,
    quant_kv_dtype: str,
    arch: str,
    cp_interleave_size: int,
    sharded_indexer: bool = False,
    enforce_shipping_kpool: bool = True,
) -> None:
    """Enforce hardware-verified M10 MLA/SFA structural legality guards."""
    dcp = max(1, int(eff_dcp))
    arch_norm = str(arch or "").strip().upper()
    kv_dt = str(quant_kv_dtype or "").strip().lower()
    if enforce_shipping_kpool and int(getattr(plan, "index_kpool", 1) or 1) > 1 and dcp > 1:
        raise ModelingError(f"m10_kpool_requires_dcp_eq_1:dcp={dcp}")
    if (getattr(plan, "mla_variant", "") == "latent_mqa"
            and getattr(plan, "cache_kind", "") == "window_compressed"):
        if dcp > 1:
            raise ModelingError(f"m10_compressed_csa2_requires_dcp_eq_1:dcp={dcp}")
        if arch_norm in {"A2", "A3"} and kv_dt == "int8":
            raise ModelingError(
                f"m10_sparse_flash_mla_a23_forbids_int8_kv:arch={arch_norm}:kv={kv_dt}"
            )
    if (dcp > 1 and bool(getattr(plan, "indexer", False))
            and not sharded_indexer
            and int(cp_interleave_size) % M10_INTERLEAVE_ALIGN != 0):
        raise ModelingError(
            f"m10_replicated_sfa_dcp_requires_interleave_multiple_128:{cp_interleave_size}"
        )


def _machine_vector_cores(machine) -> int | None:
    if machine is None:
        return None
    dies = getattr(machine, "dies", ()) or ()
    if dies:
        die0 = dies[0]
        aiv = int(getattr(die0, "aiv_count", 0) or 0)
        if aiv > 0:
            return aiv
        units = dict(getattr(die0, "resource_units", None) or ())
        u_aiv = int(units.get("vector", 0) or 0)
        if u_aiv > 0:
            return u_aiv
    aiv = int(getattr(machine, "aiv_count", 0) or getattr(machine, "vector_cores", 0) or 0)
    return aiv if aiv > 0 else None


def _emit_unfused_presorted_pack(
    b, *, deps: tuple[str, ...], suffix: str, tile_idx: int, tile_rows: int, index_topk: int,
    r_tile: int, cand_tiles: int, is_a5: bool, raw_cand_bytes: int, log_k: float,
) -> str:
    """Emit ordered 6-kernel (A5) or 8-kernel (A3 + 2 Cast) presorted local pack sequence.

    Matches ``vllm_ascend/ops/triton/sharded_indexer.py:438-470, 520-539``:
    ``_prepare_local_sort_inputs -> Sort -> [Cast ->] GatherElements[V2]
    -> Sort -> [Cast ->] GatherElements[V2] -> _pack_candidates``.
    """
    tile_elems = tile_rows * index_topk
    gather_kernel = "GatherElements" if is_a5 else "GatherElementsV2"
    idx_bytes = INT64_BYTES if is_a5 else INT32_BYTES
    prep = b.work(
        f"candidate_sort_prepare{suffix}", deps, (tile_rows, index_topk),
        vector=tile_elems * SHARDED_PREPARE_VECTOR_OPS,
        read=tile_elems * raw_cand_bytes,
        write=tile_elems * CANDIDATE_BYTES,
        family="elementwise", dtype="fp32",
        metadata=(
            ("native_kinds", ("_prepare_local_sort_inputs",)),
            ("kernel", "_prepare_local_sort_inputs"),
            ("candidate_tile_index", int(tile_idx)),
        ),
    )
    sort_ids = b.work(
        f"candidate_sort_ids{suffix}", (prep,), (tile_rows, index_topk),
        vector=tile_elems * log_k,
        read=tile_elems * FP32_BYTES,
        write=tile_elems * (FP32_BYTES + INT64_BYTES),
        family="topk", dtype="fp32",
        metadata=(
            ("native_kinds", ("Sort",)),
            ("kernel", "Sort"),
            ("candidate_tile_index", int(tile_idx)),
        ),
    )
    id_idx_dep = sort_ids
    if not is_a5:
        id_idx_dep = b.vector(
            f"candidate_sort_cast{suffix}", (sort_ids,), tile_elems, 1.0,
            dtype="int32", family="cast",
            read_bytes_override=tile_elems * INT64_BYTES,
            write_bytes_override=tile_elems * INT32_BYTES,
            metadata=(
                ("native_kinds", ("Cast",)),
                ("kernel", "Cast"),
                ("sort_cast_launches", 1),
                ("candidate_tile_index", int(tile_idx)),
            ),
        )
    gather_scores = b.work(
        f"candidate_gather_scores{suffix}", (prep, id_idx_dep), (tile_rows, index_topk),
        vector=tile_elems * SHARDED_GATHER_VECTOR_OPS,
        read=tile_elems * (FP32_BYTES + idx_bytes),
        write=tile_elems * FP32_BYTES,
        family="elementwise", dtype="fp32",
        metadata=(
            ("native_kinds", (gather_kernel,)),
            ("kernel", gather_kernel),
            ("candidate_tile_index", int(tile_idx)),
        ),
    )
    sort_scores = b.work(
        f"candidate_sort_scores{suffix}", (gather_scores,), (tile_rows, index_topk),
        vector=tile_elems * log_k,
        read=tile_elems * FP32_BYTES,
        write=tile_elems * (FP32_BYTES + INT64_BYTES),
        family="topk", dtype="fp32",
        metadata=(
            ("native_kinds", ("Sort",)),
            ("kernel", "Sort"),
            ("candidate_tile_index", int(tile_idx)),
        ),
    )
    score_idx_dep = sort_scores
    if not is_a5:
        score_idx_dep = b.vector(
            f"candidate_sort_cast_scores{suffix}", (sort_scores,), tile_elems, 1.0,
            dtype="int32", family="cast",
            read_bytes_override=tile_elems * INT64_BYTES,
            write_bytes_override=tile_elems * INT32_BYTES,
            metadata=(
                ("native_kinds", ("Cast",)),
                ("kernel", "Cast"),
                ("sort_cast_launches", 1),
                ("candidate_tile_index", int(tile_idx)),
            ),
        )
    gather_ids = b.work(
        f"candidate_gather_ids{suffix}", (sort_ids, score_idx_dep), (tile_rows, index_topk),
        vector=tile_elems * SHARDED_GATHER_VECTOR_OPS,
        read=tile_elems * (FP32_BYTES + idx_bytes),
        write=tile_elems * FP32_BYTES,
        family="elementwise", dtype="fp32",
        metadata=(
            ("native_kinds", (gather_kernel,)),
            ("kernel", gather_kernel),
            ("candidate_tile_index", int(tile_idx)),
        ),
    )
    pack_seq = (
        ("_prepare_local_sort_inputs", "Sort", "GatherElements",
         "Sort", "GatherElements", "_pack_candidates")
        if is_a5 else
        ("_prepare_local_sort_inputs", "Sort", "Cast", "GatherElementsV2",
         "Sort", "Cast", "GatherElementsV2", "_pack_candidates")
    )
    cast_launches = 0 if is_a5 else SHARDED_A3_CAST_OPS_PER_SORT
    return b.work(
        f"candidate_pack{suffix}", (gather_ids, sort_scores), (tile_rows, index_topk),
        vector=tile_elems * SHARDED_PACK_VECTOR_OPS,
        read=tile_elems * CANDIDATE_BYTES,
        write=tile_elems * CANDIDATE_BYTES,
        family="elementwise",
        metadata=(
            ("sharded_indexer_tile_rows", int(r_tile)),
            ("candidate_allgather_tiles", int(cand_tiles)),
            ("candidate_tile_index", int(tile_idx)),
            ("candidate_tile_rows", int(tile_rows)),
            ("sharded_presorted", True),
            ("sharded_fused_sort", False),
            ("sort_cast_launches", int(cast_launches)),
            ("native_kinds", pack_seq),
            ("kernel", "_pack_candidates"),
        ),
    )


def _emit_sharded_candidate_merge(
    b, topk: str, *, valid: int, index_topk: int, eff_dcp: int, runtime, machine,
    score_rows: float | None = None, return_local: bool = True,
) -> str:
    """Emit executable per-tile sharded-indexer pack -> AllGather -> merge.

    Matches upstream ``sharded_indexer.py:36-65`` and ``ops/triton/sharded_indexer.py:186-656``:
      * ``valid <= 0`` returns immediately with zero tiles, zero AllGathers, zero Casts.
      * Presorting and fused merge (``_merge_owner_prefix`` for local output /
        ``_merge_candidates`` otherwise) are enabled only when ``index_topk == 2048``.
      * Fused local sort executes ``ceil((20 + score_bits) / 13)`` radix passes
        (3 passes for BF16 on A5, 4 passes for FP32 on A3).
      * Unfused presorted tiles expand into the explicit 6-kernel A5 (8-kernel A3 with
        2 Casts) sequence: prepare -> sort -> gather -> sort -> gather -> pack.
      * Every tile reads ``(ids, scores)`` from ``topk``; empty owner-prefix ranks
        read only their local ``8K`` packed slice without probing peers; non-presorted
        merges account for ``_unpack_candidates`` scratch write/read traffic.
    """
    n_valid = int(valid)
    if n_valid <= 0:
        return topk
    is_a5 = _is_a5_runtime(runtime, machine)
    r_tile = sharded_indexer_tile_rows(eff_dcp, index_topk)
    cand_tiles = sharded_candidate_allgather_tiles(n_valid, eff_dcp, index_topk)
    if cand_tiles <= 0:
        return topk
    vector_cores = _machine_vector_cores(machine)
    presorted = int(index_topk) == SHARDED_FUSED_SORT_TOPK
    raw_cand_bytes = INT32_BYTES + (BF16_BYTES if is_a5 else FP32_BYTES)
    score_bits = SHARDED_SCORE_BITS_BF16 if is_a5 else SHARDED_SCORE_BITS_FP32
    radix_passes = math.ceil((SHARDED_RADIX_ID_BITS + score_bits) / SHARDED_RADIX_DIGIT_BITS)
    log_k = math.log2(max(MIN_LOG_TOPK, index_topk))
    empty_local_shard = score_rows is not None and float(score_rows) <= 0.0
    remaining = n_valid
    dep = topk
    last_merge = topk
    for tile_idx in range(cand_tiles):
        tile_rows = min(r_tile, remaining)
        remaining -= tile_rows
        tile_elems = tile_rows * index_topk
        tile_fused_sort = presorted and return_local and sharded_fused_sort_eligible(
            world_size=eff_dcp, topk=index_topk, rows=tile_rows, vector_cores=vector_cores,
        )
        suffix = "" if tile_idx == 0 else f"_t{tile_idx}"
        tile_deps = (dep,) if tile_idx == 0 else (dep, last_merge)
        if not presorted:
            merge_kernels: tuple[str, ...] = (
                ("_unpack_candidates", "Sort", "GatherElements", "Sort", "GatherElements", "_select_local_candidates")
                if is_a5 else
                ("_unpack_candidates", "Sort", "Cast", "GatherElementsV2", "Sort", "Cast", "GatherElementsV2", "_select_local_candidates")
            )
            cast_launches = 0 if is_a5 else (2 * SHARDED_A3_CAST_OPS_PER_SORT)
            gather_dep = b.work(
                f"candidate_pack{suffix}", tile_deps, (tile_rows, index_topk),
                vector=tile_elems * SHARDED_PACK_VECTOR_OPS,
                read=tile_elems * raw_cand_bytes,
                write=tile_elems * CANDIDATE_BYTES,
                family="elementwise",
                metadata=(
                    ("sharded_indexer_tile_rows", int(r_tile)),
                    ("candidate_allgather_tiles", int(cand_tiles)),
                    ("candidate_tile_index", int(tile_idx)),
                    ("candidate_tile_rows", int(tile_rows)),
                    ("sharded_presorted", False),
                    ("sharded_fused_sort", False),
                    ("sort_cast_launches", 0),
                    ("native_kinds", ("_pack_candidates",)),
                    ("kernel", "_pack_candidates"),
                ),
            )
        elif tile_fused_sort:
            merge_kernels = ("_merge_owner_prefix",) if return_local else ("_merge_candidates",)
            cast_launches = 0
            per_pass_ops = (
                SHARDED_FUSED_SORT_BITONIC_HALF * log_k * (log_k + 1.0)
                + SHARDED_RADIX_GATHER_STREAMS * SHARDED_GATHER_VECTOR_OPS
            )
            radix_ops = tile_elems * (radix_passes * per_pass_ops + SHARDED_PACK_VECTOR_OPS)
            gather_dep = b.work(
                f"candidate_pack{suffix}", tile_deps, (tile_rows, index_topk),
                vector=radix_ops,
                read=tile_elems * raw_cand_bytes,
                write=tile_elems * CANDIDATE_BYTES,
                family="elementwise",
                metadata=(
                    ("sharded_indexer_tile_rows", int(r_tile)),
                    ("candidate_allgather_tiles", int(cand_tiles)),
                    ("candidate_tile_index", int(tile_idx)),
                    ("candidate_tile_rows", int(tile_rows)),
                    ("sharded_presorted", True),
                    ("sharded_fused_sort", True),
                    ("sharded_radix_passes", int(radix_passes)),
                    ("sort_cast_launches", 0),
                    ("native_kinds", ("_pack_sorted_candidates",)),
                    ("kernel", "_pack_sorted_candidates"),
                ),
            )
        else:
            merge_kernels = ("_merge_owner_prefix",) if return_local else ("_merge_candidates",)
            cast_launches = 0
            gather_dep = _emit_unfused_presorted_pack(
                b, deps=tile_deps, suffix=suffix, tile_idx=tile_idx,
                tile_rows=tile_rows, index_topk=index_topk,
                r_tile=r_tile, cand_tiles=cand_tiles, is_a5=is_a5,
                raw_cand_bytes=raw_cand_bytes, log_k=log_k,
            )
        gathered = b.communicate(
            f"candidate_gather{suffix}", (gather_dep,),
            tile_elems * CANDIDATE_BYTES * (eff_dcp - 1),
            "allgather", "cp", dtype="fp32",
            comm_group_size=int(eff_dcp), dcp_degree=int(eff_dcp),
            candidate_allgather_tiles=int(cand_tiles),
            candidate_tile_index=int(tile_idx),
            sharded_indexer_tile_rows=int(r_tile),
        )
        if not presorted:
            merge_read_bytes = tile_elems * eff_dcp * CANDIDATE_BYTES * SHARDED_NON_PRESORTED_UNPACK_READ_MULT
            merge_write_bytes = tile_elems * eff_dcp * CANDIDATE_BYTES + tile_elems * INT32_BYTES
        elif return_local and empty_local_shard:
            merge_read_bytes = tile_elems * CANDIDATE_BYTES
            merge_write_bytes = tile_elems * INT32_BYTES
        else:
            merge_read_bytes = tile_elems * eff_dcp * CANDIDATE_BYTES
            merge_write_bytes = tile_elems * INT32_BYTES
        last_merge = b.vector(
            f"candidate_merge{suffix}", (gathered,),
            tile_elems * eff_dcp,
            log_k,
            dtype="int32", family="topk",
            read_bytes_override=merge_read_bytes,
            write_bytes_override=merge_write_bytes,
            metadata=(
                ("sharded_presorted", bool(presorted)),
                ("sharded_fused_sort", bool(tile_fused_sort)),
                ("candidate_allgather_tiles", int(cand_tiles)),
                ("candidate_tile_index", int(tile_idx)),
                ("sort_cast_launches", int(cast_launches if not presorted else 0)),
                ("native_kinds", merge_kernels),
                ("kernel", merge_kernels[0]),
            ),
        )
    return last_merge


def emit_ai_cpu_metadata_op(
    b,
    *,
    kind: str,
    deps: tuple[str, ...],
    rows: int,
    arch: str,
) -> str:
    """Emit explicit ``AI_CPU`` sparse metadata op (A3=1024 / A5=2048 for SMLA)."""
    arch_norm = str(arch or "").strip().upper()
    if arch_norm and arch_norm not in (ASCEND_ARCH_FAMILIES | frozenset({"GENERIC", "SYNTHETIC"})):
        raise ModelingError(f"ai_cpu_metadata_arch_unresolved:{arch!r}")
    if kind == "SparseFlashMlaMetadata":
        meta_elems = SMLA_METADATA_SIZE_A5 if arch_norm == "A5" else SMLA_METADATA_SIZE_A23
    else:
        meta_elems = max(AI_CPU_METADATA_DEFAULT_ELEMS, int(rows))
    meta_bytes = float(meta_elems * INT32_BYTES)
    return b.work(
        f"ai_cpu_meta_{kind}",
        deps,
        (max(1, int(rows)), meta_elems),
        vector=0.0,
        read=meta_bytes,
        write=meta_bytes,
        dtype="int32",
        family="elementwise",
        recipe=kind,
        metadata=(
            ("execution_core", "aicpu"),
            ("execution_domain", "AI_CPU"),
            ("invocation_class", "aicpu"),
            ("task_type", "AI_CPU"),
            ("metadata_buffer_elems", int(meta_elems)),
            ("native_kinds", (kind,)),
            ("kernel", kind),
        ),
    )


class SharedSelectionRef(str):
    """TopK occurrence id plus its produced per-request selected-token rows.

    Later MTP proposal steps reuse one TopK id.  They must consume the
    producer's selection geometry, not the wider proposal batch that merely
    names the same identifier.
    """

    __slots__ = ("selected_token_rows", "selected_by_request")

    def __new__(cls, op_id, *, selected_token_rows, selected_by_request):
        value = super().__new__(cls, str(op_id))
        value.selected_token_rows = float(selected_token_rows)
        value.selected_by_request = dict(selected_by_request)
        return value

    def rows_for(self, request_id):
        """Producer-selected rows owned by one request, or ``None``."""
        if request_id in self.selected_by_request:
            return self.selected_by_request[request_id]
        # A one-request producer may be replayed by a wider proposal batch
        # only when that request owns the complete declared selection. A
        # partial map must not copy request 0's rows onto every request.
        if (len(self.selected_by_request) == 1
                and self.selected_token_rows
                == next(iter(self.selected_by_request.values()))):
            return next(iter(self.selected_by_request.values()))
        return None


def _selection_reference(builder, op_id):
    """Wrap one TopK occurrence with the selection geometry it produced."""
    op = next((item for item in reversed(builder.ops) if item.op_id == op_id), None)
    if op is None:
        return op_id
    metadata = dict(op.metadata)
    total = metadata.get("selected_token_rows")
    if total is None:
        return op_id
    request_ids = tuple(metadata.get("selected_request_ids") or ())
    rows = tuple(metadata.get("selected_tokens_by_request") or ())
    by_request = dict(zip(request_ids, rows)) if len(request_ids) == len(rows) else {}
    return SharedSelectionRef(op_id, selected_token_rows=total,
                              selected_by_request=by_request)
CANDIDATE_BLOCK_OPS = 2
DEFAULT_BLOCK_SIZE = 128
ASCEND_ARCH_FAMILIES = frozenset({"A2", "A3", "A5"})
NATIVE_KEYPOOL_IMPLEMENTATION = "vllm_ascend_cann_keypool_caller"


def _policy(runtime):
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    policy = metadata.get("execution_policy")
    policy = dict(policy) if isinstance(policy, dict) else {}
    raw_eager = bool(policy.get("raw_eager", metadata.get("raw_eager", False)))
    automatic = bool(policy.get("automatic_optional_fusions", True))
    return metadata, policy, raw_eager, automatic


def native_fusion_active(runtime, machine=None):
    """True on a declared Ascend family unless eager/generic was requested."""
    if runtime is None:
        return False
    metadata, _, raw_eager, automatic = _policy(runtime)
    if raw_eager or not automatic:
        return False
    family = str(getattr(machine, "arch_family", "")
                 or metadata.get("arch_family") or "")
    return family.upper() in ASCEND_ARCH_FAMILIES


def raw_eager_requested(runtime):
    """True only for an explicit PyTorch-eager decomposition request."""
    return _policy(runtime)[2]


def _is_a5_runtime(runtime, machine=None) -> bool:
    if machine is not None:
        m_fam = str(getattr(machine, "arch_family", "") or "").strip().upper()
        if m_fam:
            return m_fam == "A5"
        dies = getattr(machine, "dies", ()) or ()
        if dies:
            ca = getattr(dies[0], "compute_arch", None)
            arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
            if arch_id:
                return "3510" in arch_id or arch_id == "a5"
    if runtime is None:
        return False
    meta = dict(getattr(runtime, "metadata", ()) or ())
    fam = str(meta.get("arch_family") or meta.get("device_arch") or "").strip().upper()
    if fam:
        return fam == "A5"
    return False


NATIVE_LIGHTNING_INDEXER_IMPLEMENTATIONS = frozenset({
    "cann_bf16_lightning_indexer_v2_complete",
    "cann_c8_lightning_indexer",
})


def native_indexer_active(runtime, machine=None):
    """True when a CANN indexer implementation owns this graph.

    Explicit native lowerings (including ``cann_bf16_lightning_indexer_v2_complete``
    and ``cann_c8_lightning_indexer``) own the boundary unless ``raw_eager`` was
    requested.  A caller that explicitly binds a non-CANN PyTorch path keeps it.
    """
    if runtime is None or raw_eager_requested(runtime):
        return False
    lowerings = dict(getattr(runtime, "kernels", {}).get("lowerings", {}) or {})
    if (select(lowerings, "pool_key_indexer") is not None
            or select(lowerings, "quant_lightning_indexer") is not None):
        return True
    bf16_or_c8 = select(lowerings, "lightning_indexer")
    if bf16_or_c8 is not None:
        return bf16_or_c8.name in NATIVE_LIGHTNING_INDEXER_IMPLEMENTATIONS
    return native_fusion_active(runtime, machine)


def kv_cache_epilog(op, fmt, region, *, native, row_bytes=None):
    """Name the grouped KV quantize+scatter epilog without re-deducting bytes.

    ``cache_representation.store`` already writes the grouped quantized row in
    one read and one write, so only the native epilog identity is attached.
    """
    if not native or not fmt.group_size:
        return op
    kind = KV_COMPRESS_EPILOG if fmt.dtype != op.dtype else KV_SCATTER_EPILOG
    width = int(op.local_shape[1]) if len(op.local_shape) > 1 else 0
    metadata = dict(op.metadata)
    resolved_row_bytes = (
        int(row_bytes) if row_bytes is not None else (
            metadata.get("cache_row_bytes")
            or dict(metadata.get("cache_representation") or {}).get("row_bytes")
            or fmt.row_bytes(width)
        ) if width else 0
    )
    return replace(op, family=KV_COMPRESS_EPILOG_FAMILY, metadata=op.metadata + (
        ("native_kind", kind),
        ("cache_epilog", {
            "kind": kind, "region": region,
            "row_bytes": resolved_row_bytes,
            "quantized_payload_dtype": fmt.dtype,
            "scale_dtype": fmt.scale_dtype,
            "group_size": fmt.group_size,
            "single_read_single_write": True,
        }),
        ("native_quantize_and_scatter", True),
    ))


def block_quant_epilog(op, fmt, region, *, native, width, row_bytes=None):
    """Name the NoPE C8 block-quantized cache write without changing bytes."""
    if not native or fmt.dtype != "fp8_e4m3" or fmt.group_size != BLOCK_QUANT_TILE:
        return op
    resolved_row_bytes = int(row_bytes) if row_bytes is not None else fmt.row_bytes(width)
    return replace(op, family=KV_COMPRESS_EPILOG_FAMILY, metadata=op.metadata + (
        ("native_kind", BLOCK_QUANT_KIND),
        ("cache_epilog", {
            "kind": BLOCK_QUANT_KIND, "scatter": KV_SCATTER_EPILOG,
            "region": region, "tile": BLOCK_QUANT_TILE,
            "row_bytes": resolved_row_bytes,
            "row_components": {"fp8_payload": width, "scale": resolved_row_bytes - width},
            "group_size": fmt.group_size, "scale_dtype": fmt.scale_dtype,
        }),
        ("cache_row_bytes", resolved_row_bytes),
    ))


def native_cache_store(op, *, native):
    """Name an unfused-but-native cache store with its production family.

    The packed prolog writer and a plain latent row are already single native
    stores; on a CANN graph they must not remain the generic eager
    ``cache_store`` boundary.  Grouped writers keep their more specific epilog.
    """
    if not native or op.family != "cache_store":
        return op
    return replace(op, family=KV_COMPRESS_EPILOG_FAMILY,
                   metadata=op.metadata + (("native_cache_store", True),))


def _resolve_arch(runtime, machine=None) -> str:
    if machine is not None:
        m_fam = str(getattr(machine, "arch_family", "") or "").strip().upper()
        if m_fam in ASCEND_ARCH_FAMILIES:
            return m_fam
        dies = getattr(machine, "dies", ()) or ()
        if dies:
            ca = getattr(dies[0], "compute_arch", None)
            arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
            if "3510" in arch_id or arch_id == "a5":
                return "A5"
            if "910b" in arch_id or arch_id == "a2":
                return "A2"
            if "2201" in arch_id or arch_id == "a3":
                sku_low = str(getattr(machine, "sku", "") or "").lower()
                return "A2" if "910b" in sku_low else "A3"
    if runtime is not None:
        meta = dict(getattr(runtime, "metadata", ()) or ())
        fam = str(meta.get("arch_family") or meta.get("device_arch") or "").strip().upper()
        if fam in ASCEND_ARCH_FAMILIES:
            return fam
    return ""


def _indexer_quant_cache_available(arch: str) -> bool:
    if not arch:
        return True
    from mindie_sim.backends.abi_registry import default_abi_registry
    return default_abi_registry().is_shipping_available(INDEXER_QUANT_CACHE, arch)


_IQC_MX_GROUP_SIZE = 32
_IQC_HIFP4_GROUP_SIZE = 64


def _indexer_quant_cache_operands_supported(
    input_dtype: str, payload_dtype: str, scale_dtype: str, *, packed_elements: int = 1,
    width: int = 0, scale_count: int = 0,
) -> bool:
    """Enforce CANN ``indexer_quant_cache_contract.h::IsValidQuantTypes`` & scale stride."""
    x = str(input_dtype or "").strip().lower()
    cache = str(payload_dtype or "").strip().lower()
    scale = str(scale_dtype or "").strip().lower()
    if x not in _IQC_INPUT_DTYPES:
        return False
    if cache in _IQC_FP8_DTYPES and scale in _IQC_E8M0_SCALES:
        if width > 0 and scale_count > 0:
            expected_scales = (width + _IQC_MX_GROUP_SIZE - 1) // _IQC_MX_GROUP_SIZE
            return scale_count == expected_scales
        return True
    if packed_elements == 1 and cache in _IQC_NORMAL_DTYPES and scale in _IQC_FP32_SCALES:
        return scale_count <= 1
    if (cache in _IQC_FP4_DTYPES or (cache == "uint8" and packed_elements == _PACKED_FP4_ELEMENTS)) and scale in _IQC_E8M0_SCALES:
        if width > 0 and scale_count > 0:
            expected_scales = (width + _IQC_MX_GROUP_SIZE - 1) // _IQC_MX_GROUP_SIZE
            return scale_count == expected_scales
        return True
    if cache in _IQC_HIFP4_DTYPES and scale in _IQC_HIFP4_SCALES:
        if width > 0 and scale_count > 0:
            return width % _IQC_HIFP4_GROUP_SIZE == 0 and scale_count == width // _IQC_HIFP4_GROUP_SIZE
        return True
    return False


def index_row_quantization(builder, store_id, quantize_id, *, width, native, arch=None):
    """Fuse index-key quantization and cache scatter into one epilog.

    The quantization producer already writes exactly one cache row per row
    (payload plus group scales).  Keeping it external makes the synthetic
    ``cache_store`` read that row back and write it again.  The fused epilog
    keeps the BF16 input read and the cache-row write, removes the intermediate
    pair, and preserves the total vector work.
    """
    if not native or store_id is None or quantize_id is None:
        return store_id
    arch_norm = str(arch or "").strip().upper()
    if arch_norm and not _indexer_quant_cache_available(arch_norm):
        return store_id
    by_id = {op.op_id: op for op in builder.ops}
    store = by_id.get(store_id)
    quantize = by_id.get(quantize_id)
    if store is None or quantize is None or store.family != "cache_store":
        return store_id
    if quantize.family != "quantize" or quantize_id not in store.deps:
        return store_id
    metadata = dict(quantize.metadata)
    packed = int(metadata.get("packed_elements") or 1)
    scale_count = int(metadata.get("scale_count") or 0)
    input_dtype = str(metadata.get("input_format") or quantize.dtype)
    if not _indexer_quant_cache_operands_supported(
        input_dtype, metadata.get("output_format"), metadata.get("scale_format"),
        packed_elements=packed, width=width, scale_count=scale_count,
    ):
        return store_id
    inputs = tuple(quantize.deps)
    extra = tuple(dep for dep in store.deps if dep != quantize_id)
    fused = replace(
        store, deps=tuple(dict.fromkeys((*inputs, *extra))),
        family=INDEXER_QUANT_CACHE_FAMILY,
        mte2_bytes=quantize.mte2_bytes, mte3_bytes=store.mte3_bytes,
        hbm_bytes=quantize.mte2_bytes + store.mte3_bytes,
        vector_ops=quantize.vector_ops, dtype=quantize.dtype,
        recipe=INDEXER_QUANT_CACHE_RECIPE,
        metadata=store.metadata + (
            ("native_kind", INDEXER_QUANT_CACHE),
            ("fused_members", (quantize.op_id, store.op_id)),
            ("payload_dtype", metadata.get("output_format")),
            ("slot_table_bytes", 0.0),
            ("cache_epilog", {
                "kind": INDEXER_QUANT_CACHE,
                "payload_dtype": metadata.get("output_format"),
                "scale_dtype": metadata.get("scale_format"),
                "scale_shape": metadata.get("scale_shape"),
                "row_elements": width,
                "single_read_single_write": True,
            }),
            ("index_topk_kind", INDEXER_TOP_K_KIND),
        ))
    builder.ops = [op for op in builder.ops
                   if op.op_id not in {quantize_id, store_id}]
    return builder.add(fused)


def _indexer_matrix_category(quant, role: str) -> str:
    """Resolve role-specific indexer Q/K category with unscanned ``indexer_fp8`` fallback."""
    overrides = dict(getattr(quant, "category_overrides", ()) or ())
    exclusions = set(getattr(quant, "exclusions", ()) or ())
    family = role.split(".", 1)[0]
    if role in overrides or role in exclusions or family in overrides or family in exclusions:
        return role
    if "indexer_fp8" in overrides or "indexer_fp8" in exclusions:
        return "indexer_fp8"
    return role


def index_head_weights(builder, *, plan, model, quant, deps, pool_weights, tokens,
                       native):
    """Project indexer head weights and never wrap a native consumer in casts."""
    # The per-layer resolved projection format is the operand contract; the
    # model-level preset may still be W8A8 while indexer head weights stay BF16.
    weights_dtype = compute_dtype_for(builder.quant, "indexer.weights")[1]
    weights_input = deps
    if (not native and pool_weights is None and weights_dtype in {"fp32", "fp16", "bf16"}
            and weights_dtype != model.activation_dtype):
        weights_input = builder.vector(
            "index_weights_input_cast", (deps,), tokens * model.hidden_size, 1,
            family="cast", dtype=weights_dtype, shape=(tokens, model.hidden_size),
            read_bytes_override=tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
            write_bytes_override=tokens * model.hidden_size * dtype_bytes(weights_dtype),
            metadata=(("input_format", model.activation_dtype),
                      ("output_format", weights_dtype)))
    # The KPool producer's fused [K, head weights] projection exposes a
    # strided column view. Its legacy consumer cast reads only the head slice.
    weights = pool_weights or builder.matrix(
        "index_weights", (weights_input,), tokens, plan.index_heads, model.hidden_size,
        category="indexer.weights")
    w_op = next((item for item in reversed(builder.ops) if item.op_id == weights), None)
    w_out_dtype = (
        str(dict(w_op.metadata).get("output_format") or w_op.dtype or weights_dtype).strip().lower()
        if w_op is not None else weights_dtype
    )
    phase = builder.common.get("phase")
    mode1 = native and plan.index_kpool == 1 and quant.index_dtype in {"fp8", "fp8_e4m3"}
    packed_fp32 = index_cache_is_packed(model, plan=plan, phase=phase)
    int8_qli_fp16 = (
        plan.index_kpool <= 1 and quant.index_dtype == "int8" and not packed_fp32
    )
    if native and not mode1 and not packed_fp32 and not int8_qli_fp16:
        # ``aclnnPoolKeyIndexer`` and BF16/FP8 non-mode1 consumers consume the
        # checkpoint head-weight projection directly.
        return weights
    if mode1 or packed_fp32:
        if w_out_dtype != FP32_WEIGHT_DTYPE:
            weights = builder.vector(
                "index_weights_fp32", (weights,), tokens * plan.index_heads, 1,
                family="cast", dtype=FP32_WEIGHT_DTYPE,
                shape=(tokens, plan.index_heads),
                read_bytes_override=tokens * plan.index_heads * dtype_bytes(w_out_dtype),
                write_bytes_override=tokens * plan.index_heads * FP32_BYTES,
                metadata=(
                    ("input_format", w_out_dtype),
                    ("output_format", FP32_WEIGHT_DTYPE),
                    ("source_semantics", "packed_indexer_weights_float32"),
                ))
        if packed_fp32:
            # deepseek_v41/indexer.py:184: self._output(weights_proj, hidden).float() * self.weights_scale
            weights = builder.vector(
                "index_weights_scale", (weights,), tokens * plan.index_heads, 1,
                family="elementwise", dtype=FP32_WEIGHT_DTYPE,
                shape=(tokens, plan.index_heads),
                read_bytes_override=tokens * plan.index_heads * FP32_BYTES,
                write_bytes_override=tokens * plan.index_heads * FP32_BYTES,
                metadata=(
                    ("input_format", FP32_WEIGHT_DTYPE),
                    ("output_format", FP32_WEIGHT_DTYPE),
                    ("source_semantics", "packed_indexer_weights_scale_float32"),
                ))
        return weights
    if quant.index_dtype == "int8":
        if w_out_dtype == "fp16":
            return weights
        return builder.vector(
            "index_weights_fp16", (weights,), tokens * plan.index_heads, 1,
            family="cast", dtype="fp16", shape=(tokens, plan.index_heads),
            read_bytes_override=tokens * plan.index_heads * dtype_bytes(w_out_dtype),
            write_bytes_override=tokens * plan.index_heads * FP16_BYTES,
            metadata=(
                ("input_format", w_out_dtype),
                ("output_format", "fp16"),
                ("source_semantics", "quant_indexer_weights_float16"),
            ))
    if plan.index_kpool > 1:
        if w_out_dtype != quant.index_dtype:
            weights = builder.vector(
                "index_weights_cast", (weights,), tokens * plan.index_heads, 1,
                family="cast", dtype=quant.index_dtype,
                shape=(tokens, plan.index_heads),
                read_bytes_override=tokens * plan.index_heads * dtype_bytes(w_out_dtype),
                write_bytes_override=tokens * plan.index_heads * dtype_bytes(quant.index_dtype),
                metadata=(("input_format", w_out_dtype),
                          ("output_format", quant.index_dtype))
                + ((("input_column_slice",
                     (plan.index_dim, plan.index_dim + plan.index_heads)),)
                   if pool_weights else ()))
        return builder.vector(
            "index_weights_scale", (weights,), tokens * plan.index_heads, 1,
            dtype=quant.index_dtype, shape=(tokens, plan.index_heads))
    return weights


def _indexer_rotates_basis(plan, model) -> bool:
    """Gate indexer Hadamard rotation on explicit representation/compressor policy."""
    explicit_plan = getattr(plan, "index_rotate_basis", None)
    if explicit_plan is not None:
        return bool(explicit_plan)
    meta = dict(getattr(model, "metadata", ()) or ())
    exec_meta = meta.get("attention_execution")
    if isinstance(exec_meta, dict) and "index_rotate_basis" in exec_meta:
        return bool(exec_meta["index_rotate_basis"])
    cache_reps = meta.get("cache_representations")
    if isinstance(cache_reps, dict):
        idx_rep = cache_reps.get("index_k")
        if isinstance(idx_rep, dict) and "rotate_basis" in idx_rep:
            return bool(idx_rep["rotate_basis"])
    return bool(
        plan.compress_ratio > 1
        and getattr(plan, "compressor_kind", "") == "weighted_pool"
        and getattr(plan, "index_input", "hidden") != "compressed"
    )


def _indexer_uses_rmsnorm(plan, model) -> bool:
    """True when the indexer k_norm uses BF16 RMSNorm instead of FP32 LayerNorm."""
    explicit = str(getattr(plan, "index_norm_kind", "") or "").strip().lower()
    if explicit in {"rmsnorm", "layernorm"}:
        return explicit == "rmsnorm"
    meta = dict(getattr(model, "metadata", ()) or ())
    exec_meta = meta.get("attention_execution")
    if isinstance(exec_meta, dict) and exec_meta.get("index_norm_kind"):
        return str(exec_meta["index_norm_kind"]).strip().lower() == "rmsnorm"
    return (
        getattr(plan, "compressor_kind", "") == "gated_ring"
        and getattr(plan, "index_input", "") == "compressed"
    )


def _store_rows(
    rows_by_view, views, parallel, block, *, contexts=None,
    rank_candidates=None, storage_degrees=None, ratio: int = 1,
):
    return owned_store_rows(
        rows_by_view, views, parallel, block, contexts=contexts,
        rank_candidates=rank_candidates, storage_degrees=storage_degrees, ratio=ratio,
    )


def lower_index_path(builder, *, plan, model, quant, parallel, views, deps, runtime,
                     padded_tokens, valid, cp, own_block, long_store,
                     compressed_latent, cache_ids, index_cache_ids, candidate_ids,
                     topk_ids, q_norm, stage, stage_of, index_rope, pool_binding,
                     pcp_prefill, dsa_prefill, hierarchy=None, reuse_selection=False,
                     shared_topk=None, fused_compressor=False,
                     cache_hidden_dep=None, cache_views=None,
                     cache_padded_tokens=None, skip_pcp_index_gather=False,
                     machine=None, store_rank_candidates=None, physical_rank=None):
    """Produce the index cache and (optionally) its TopK selection."""
    b = builder
    q = padded_tokens
    native_indexer = native_indexer_active(runtime, machine)
    ratio = max(1, plan.compress_ratio)
    eff_dcp = max(int(cp), layer_dcp_degree(plan, parallel)) if dcp_indexer_sharded(model, runtime) else int(cp)
    sharded_index = dcp_indexer_sharded(model, runtime) and eff_dcp > 1
    idx_block = max(1, own_block // ratio)
    topk = candidate = None

    def owner(mapping, layer, name, volume):
        if layer not in mapping:
            raise ModelingError(f"producer_missing:{b.prefix}:{name}:{layer}")
        ready = mapping[layer]
        producer_stage = stage_of(layer)
        if producer_stage != stage:
            ready = b.communicate(name + "_transfer", (ready,), volume,
                                  "cache_transfer", "pp",
                                  source_stage=producer_stage, destination_stage=stage)
        return ready

    prod_views = cache_views if cache_views is not None else views
    prod_deps = cache_hidden_dep if cache_hidden_dep is not None else deps
    prod_q = cache_padded_tokens if cache_padded_tokens is not None else q
    prod_valid = sum(v.q_len for v in prod_views)
    index_store, pool_weights = _produce_index_cache(
        b, plan, pool_binding, prod_views, prod_deps, model, quant, prod_q, long_store,
        compressed_latent, ratio, prod_valid, fused_compressor, runtime, native_indexer,
        index_rope, parallel, idx_block, sharded_index,
        pcp_prefill=(pcp_prefill and not skip_pcp_index_gather), machine=machine,
        store_rank_candidates=store_rank_candidates, raw_block=own_block)
    if (pcp_prefill and not skip_pcp_index_gather and plan.index_kpool <= 1
            and plan.indexer and plan.owns_index_k and index_store is not None):
        # One stored index vector is index_dim wide, not index_heads copies of it.
        rows = sum(completed_rows(
            view.context + (view.global_q_len or view.q_len), ratio)
            - completed_rows(view.context, ratio)
            for view in views if view.phase == "prefill")
        index_store = b.communicate(
            "pcp_index_gather", (index_store,),
            rows * index_row_bytes(model, plan.index_dim)
            * (parallel.pcp - 1) / parallel.pcp,
            "allgather", "pcp", dtype=quant.index_dtype)
    if reuse_selection:
        if shared_topk is None or plan.index not in shared_topk:
            raise ModelingError("mtp_shared_topk_selection_missing")
        return index_store, shared_topk[plan.index], None, pool_weights
    if plan.indexer or plan.owns_topk or plan.index_query_per_layer:
        if index_store is None:
            if plan.scope == "draft" and plan.index_k_owner_layer_id < 0:
                index_store = deps
            else:
                index_store = owner(
                    index_cache_ids, plan.index_k_owner_layer_id, "index_cache",
                    sum(index_selection_cost(model, plan, view.kv_len).index_read_bytes
                        for view in views))
        shard_sizes, score_q = None, q
        if (plan.index_kpool > 1 and pool_binding
                and pool_binding.name == NATIVE_KEYPOOL_IMPLEMENTATION):
            from mindie_sim.prediction.operators.sparse_indexer.index_pool_native import (
                _row_shard, resolve_row_shard_rank, supported,
            )
            if supported(views=views, plan=plan, model=model, quant=quant):
                opts = pool_binding.options
                shard_sizes = _row_shard(views, plan, options=opts, runtime=runtime)
                if len(shard_sizes) > 1:
                    shard_rank = resolve_row_shard_rank(
                        opts, len(shard_sizes), physical_rank=physical_rank,
                        parallel=parallel, machine=machine,
                    )
                    score_q = shard_sizes[shard_rank]
        q_index = b.matrix(
            "index_q", (q_norm,), score_q,
            plan.index_heads * plan.index_dim,
            plan.q_lora_rank,
            category=_indexer_matrix_category(b.quant, "indexer.wq"),
        )
        if plan.index_kpool == 1 and index_rope:
            q_index = b.vector(
                "index_query_rotary", (q_index,),
                valid * plan.index_heads * index_rope, ROPE_OPS,
                family="rotary", shape=(valid, plan.index_heads, plan.index_dim))
        q_index = index_representation(
            b, "index_query", q_index,
            (score_q if plan.index_kpool > 1 else valid) * plan.index_heads, plan.index_dim,
            quant=quant, model=model, rotate_basis=_indexer_rotates_basis(plan, model))
        weights = index_head_weights(
            b, plan=plan, model=model, quant=quant, deps=deps,
            pool_weights=pool_weights, tokens=score_q, native=native_indexer)
        selection_deps = [q_index, weights, index_store]
        if plan.candidate_filter:
            if plan.scope == "draft" and plan.candidate_source_layer_id < 0:
                selection_deps.append(deps)
            else:
                selection_deps.append(owner(
                    candidate_ids, plan.candidate_source_layer_id, "candidate",
                    valid * plan.candidate_topk_blocks * INT32_BYTES))
        shard = (parallel.tp if dsa_prefill
                 else eff_dcp if sharded_index else 1)
        score_rows = (
            sharded_index_visible_rows(
                views, parallel, own_block if ratio > 1 else idx_block, ratio,
                sharded_dcp=eff_dcp, rank_candidates=store_rank_candidates,
            )
            if sharded_index else None
        )
        topk = _select_index_rows(
            b, plan, pool_binding, views, model, quant, q_index, weights, index_store, q,
            runtime, hierarchy, shard_sizes, selection_deps, parallel, stage, ratio, shard,
            score_rows, machine=machine, cp_interleave_size=idx_block,
            physical_rank=physical_rank)
        if shard > 1 and not dsa_prefill:
            return_local = int(getattr(parallel, "pcp", 1) or 1) <= 1 or not any(
                v.phase == "prefill" for v in views
            )
            topk = _emit_sharded_candidate_merge(
                b, topk, valid=valid, index_topk=plan.index_topk,
                eff_dcp=eff_dcp, runtime=runtime, machine=machine,
                score_rows=score_rows, return_local=return_local,
            )
        if plan.owns_candidate_blocks:
            candidate = b.vector("candidate_blocks", (topk,),
                                 valid * plan.index_topk, CANDIDATE_BLOCK_OPS,
                                 dtype="int32")
    elif plan.topk_owner_layer_id >= 0:
        topk = owner(topk_ids, plan.topk_owner_layer_id, "topk",
                     valid * plan.index_topk * INT32_BYTES)
    return (index_store, _selection_reference(b, topk), candidate, pool_weights)


def _emit_decomposed_index_quant_store(
    b, quantize_op, *, normalized, index_rows, index_store_rows, total_row_b, quant,
    c2_token_aligned: bool = False,
):
    """Emit dynamic-quant FP32 scale output + scale cast + two slot-indexed scatters."""
    q_meta = dict(quantize_op.metadata)
    scale_fmt = str(q_meta.get("scale_format") or "none").strip().lower()
    scale_cnt = int(q_meta.get("scale_count") or 0)
    scale_row_b = (
        scale_cnt * dtype_bytes(scale_fmt)
        if (scale_cnt > 0 and scale_fmt not in {"", "none"}) else 0
    )
    payload_row_b = max(0, total_row_b - scale_row_b)
    payload_dtype = str(q_meta.get("output_format") or quant.index_dtype).strip().lower()
    scale_dep = normalized
    # DynamicQuantKernelNpuOpApi.cpp:188 always writes FP32 scale before caller casts to target.
    if payload_dtype == "int8" and scale_cnt > 0 and scale_fmt not in {"", "none", "fp32", "float32"}:
        fp32_scale_row_b = scale_cnt * FP32_BYTES
        dq_write = float(index_rows * (payload_row_b + fp32_scale_row_b))
        dq_meta = {**q_meta, "scale_format": "fp32", "physical_row_bytes": payload_row_b + fp32_scale_row_b}
        b.ops[-1] = replace(
            quantize_op,
            mte3_bytes=dq_write,
            hbm_bytes=quantize_op.mte2_bytes + dq_write,
            metadata=tuple(dq_meta.items()),
        )
        scale_dep = b.vector(
            "index_scale_cast", (normalized,), index_rows * scale_cnt, 1,
            family="cast", dtype=scale_fmt, shape=(index_rows, scale_cnt),
            read_bytes_override=index_rows * fp32_scale_row_b,
            write_bytes_override=index_rows * scale_row_b,
            metadata=(
                ("input_format", "fp32"),
                ("output_format", scale_fmt),
                ("source_semantics", "dynamic_quant_scale_cast"),
            ),
        )
    slot_rows = index_rows if c2_token_aligned else index_store_rows
    slot_ints = C2_SCATTER_COORD_INTS if c2_token_aligned else 1
    slots_bytes = float(slot_rows * slot_ints * INT32_BYTES)
    store_deps: tuple[str, ...] = (normalized,)
    if scale_row_b > 0:
        scale_store = b.store(
            "index_scale_store", (scale_dep,), index_store_rows, scale_row_b, scale_fmt,
        )
        s_op = b.ops[-1]
        b.ops[-1] = replace(
            s_op,
            mte2_bytes=s_op.mte2_bytes + slots_bytes,
            hbm_bytes=s_op.hbm_bytes + slots_bytes,
            metadata=s_op.metadata + (("scatter_slot_bytes", slots_bytes),),
        )
        store_deps = (normalized, scale_store)
    index_store = b.store(
        "index_store", store_deps, index_store_rows,
        payload_row_b if scale_row_b > 0 else total_row_b, payload_dtype,
    )
    k_op = b.ops[-1]
    b.ops[-1] = replace(
        k_op,
        mte2_bytes=k_op.mte2_bytes + slots_bytes,
        hbm_bytes=k_op.hbm_bytes + slots_bytes,
        metadata=k_op.metadata + (("scatter_slot_bytes", slots_bytes),),
    )
    return index_store


def _emit_packed_mxfp4_index_store(
    b, quantize_op, *, quant_rows: int, index_store_rows: int, total_row_b: int, width: int,
) -> str:
    """Fuse Triton MXFP4 index quantizer + masked cache write (``quantize_mxfp4_indexer.py:127``)."""
    q_meta = dict(quantize_op.metadata)
    slots_bytes = float(quant_rows * C2_SCATTER_COORD_INTS * INT32_BYTES)
    mte2 = float(quantize_op.mte2_bytes) + slots_bytes
    mte3 = float(index_store_rows * total_row_b)
    b.ops = [op for op in b.ops if op.op_id != quantize_op.op_id]
    return b.work(
        "index_store", tuple(quantize_op.deps), (index_store_rows, width),
        vector=float(quantize_op.vector_ops),
        read=mte2,
        write=mte3,
        family="kv_cache_write",
        dtype=quantize_op.dtype,
        recipe="index_store",
        metadata=(
            ("native_kind", MXFP4_INDEXER_NATIVE_KIND),
            ("scatter_slot_bytes", slots_bytes),
            ("payload_dtype", q_meta.get("output_format")),
            ("scale_dtype", q_meta.get("scale_format")),
            ("valid_store_rows", int(index_store_rows)),
            ("token_aligned_rows", int(quant_rows)),
        ),
    )


def _produce_keypool_index_cache(
    b, plan, pool_binding, views, deps, model, quant, q, runtime, parallel, *, pcp_prefill=False,
):
    from mindie_sim.prediction.operators.sparse_indexer import index_pool
    kpool_source, kpool_views, kpool_q = deps, views, q
    if pcp_prefill and int(parallel.pcp) > 1:
        kpool_views = tuple(
            replace(v, q_len=int(v.global_q_len or v.q_len),
                    query_ranges=((0, int(v.global_q_len or v.q_len)),))
            if v.phase == "prefill" else v
            for v in views
        )
        kpool_q = max(q * int(parallel.pcp), sum(v.q_len for v in kpool_views))
        kpool_source = emit_pcp_hidden_gather_restore(
            b.ops, prefix=b.prefix,
            gather_name="pcp_keypool_hidden_gather",
            restore_name="pcp_keypool_hidden_restore",
            stage=b.common["stage"], layer=plan.index,
            requests=b.common["requests"], mode=b.common["phase"],
            dep=deps, local_rows=q, global_rows=kpool_q,
            hidden=model.hidden_size, pcp=int(parallel.pcp),
            act_dtype=model.activation_dtype, runtime=runtime,
            source="vllm_ascend.attention.indexer_kpool",
            stream=CP_DECODE_COMM_STREAM,
        )
    native_pool = bool(
        pool_binding
        and pool_binding.name == "vllm_ascend_cann_keypool_caller")
    if native_pool:
        from mindie_sim.prediction.operators.sparse_indexer.index_pool_native import supported as native_pool_supported
        # Both native cache production and analytical fallback retain the full K/V batch.
        native_pool = native_pool_supported(
            views=kpool_views, plan=plan, model=model, quant=quant)
    pool_module = pool_binding.module if native_pool else index_pool
    producer_options = (pool_binding.options if native_pool else {})
    return pool_module.producer(
        b, source=kpool_source, views=kpool_views, plan=plan, model=model, quant=quant,
        padded_tokens=kpool_q, **producer_options)


def _emit_index_cache_store(
    b, plan, views, model, quant, normalized, *, ratio, index_rows, index_rows_by_view,
    quant_rows, c2_token_aligned, runtime, native_indexer, parallel, idx_block,
    sharded_index, machine, store_rank_candidates, raw_block,
):
    last = b.ops[-1]
    quantize_id = (last.op_id if last.family == "quantize"
                   and last.op_id == b.prefix + ".index_key_quantize" else None)
    if sharded_index and index_rows:
        if ratio > 1:
            index_store_rows = _store_rows(
                [view.q_len for view in views], views, parallel,
                int(raw_block) if raw_block is not None else max(1, idx_block * ratio),
                rank_candidates=store_rank_candidates, ratio=ratio,
            )
        else:
            index_store_rows = _store_rows(
                index_rows_by_view, views, parallel, idx_block,
                rank_candidates=store_rank_candidates,
            )
    else:
        index_store_rows = index_rows
    total_row_b = index_row_bytes(model, plan.index_dim)
    arch = _resolve_arch(runtime, machine)
    q_meta = dict(last.metadata) if quantize_id is not None else {}
    packed_elems = int(q_meta.get("packed_elements") or 1)
    scale_count = int(q_meta.get("scale_count") or 0)
    q_input_dtype = str(q_meta.get("input_format") or last.dtype)
    iqc_supported = (
        quantize_id is not None
        and _indexer_quant_cache_available(arch)
        and _indexer_quant_cache_operands_supported(
            q_input_dtype, q_meta.get("output_format"), q_meta.get("scale_format"),
            packed_elements=packed_elems, width=plan.index_dim, scale_count=scale_count,
        )
    )
    if quantize_id is not None and native_indexer and packed_elems > 1 and arch == "A5":
        return _emit_packed_mxfp4_index_store(
            b, last, quant_rows=quant_rows, index_store_rows=index_store_rows,
            total_row_b=total_row_b, width=plan.index_dim,
        )
    if quantize_id is not None and native_indexer and not iqc_supported:
        # Target arch or operand tuple lacks shipping aclnnIndexerQuantCache:
        # keep dynamic quant (FP32 scale) + scale cast + two slot-indexed scatters.
        return _emit_decomposed_index_quant_store(
            b, last, normalized=normalized, index_rows=quant_rows,
            index_store_rows=index_store_rows, total_row_b=total_row_b, quant=quant,
            c2_token_aligned=c2_token_aligned,
        )
    index_store = b.store("index_store", (normalized,), index_store_rows,
                          total_row_b, quant.index_dtype)
    if quantize_id is not None:
        return index_row_quantization(
            b, index_store, quantize_id, width=plan.index_dim,
            native=native_indexer, arch=arch)
    if native_indexer:
        from mindie_sim.prediction.operators.sparse_indexer.index_cache_writer import SCATTER_PA_KV_RECIPE, SCATTER_PA_KV_SOURCE
        store_op = next((item for item in reversed(b.ops)
                         if item.op_id == index_store), None)
        if store_op is not None and store_op.family == "cache_store":
            upgraded = replace(
                store_op,
                family="kv_cache_write",
                recipe=SCATTER_PA_KV_RECIPE,
                metadata=store_op.metadata + (
                    ("native_kind", KV_SCATTER_EPILOG),
                    ("native_cache_store", True),
                    ("zero_external_casts", True),
                    ("native_source", SCATTER_PA_KV_SOURCE),
                ),
            )
            b.ops = [upgraded if item.op_id == index_store else item
                     for item in b.ops]
    return index_store


def _produce_index_cache(
    b, plan, pool_binding, views, deps, model, quant, q, long_store, compressed_latent, ratio,
    valid, fused_compressor, runtime, native_indexer, index_rope, parallel, idx_block,
    sharded_index, *, pcp_prefill=False, machine=None, store_rank_candidates=None,
    raw_block=None,
):
    index_store = pool_weights = None
    if plan.owns_index_k and plan.index_kpool > 1:
        index_store, pool_weights = _produce_keypool_index_cache(
            b, plan, pool_binding, views, deps, model, quant, q, runtime, parallel,
            pcp_prefill=pcp_prefill,
        )
    elif plan.owns_index_k:
        index_source = long_store if plan.index_input == "compressed" else deps
        if plan.index_input == "compressed":
            if compressed_latent is not None:
                index_source = compressed_latent
            elif cache_format(model, "compressed_kv").group_size:
                raise ModelingError("packed_compressed_indexer_requires_pre_rope_latent_producer")
        index_rows_by_view = [owned_completed_rows(view, ratio) for view in views]
        index_rows = sum(index_rows_by_view)
        c2_token_aligned = (
            plan.index_input == "compressed"
            and getattr(plan, "compressor_kind", "") == "gated_ring"
        )
        producer_rows = valid if (c2_token_aligned or plan.index_input != "compressed") else index_rows
        input_width = plan.head_dim if plan.index_input == "compressed" else model.hidden_size
        if plan.compress_ratio > 1 and plan.index_input == "hidden":
            from mindie_sim.prediction.operators.mla_prolog.compression import compressor
            normalized = compressor(
                b, name="index_compress", source=deps, tokens=valid, padded_tokens=q,
                valid_rows=index_rows, width=plan.index_dim, plan=plan, model=model,
                fused=fused_compressor, tokens_by_view=[v.q_len for v in views])
            quant_rows = index_rows
        else:
            if c2_token_aligned:
                # Upstream ``deepseek_v41/indexer.py:85`` creates ``wk`` as unquantized BF16 nn.Linear.
                wk_quant = QuantSpec(
                    name=model.activation_dtype,
                    weight_format=model.activation_dtype,
                    activation_format=model.activation_dtype,
                    native_format=model.activation_dtype,
                )
                projected = b.add(gemm_op(
                    op_id=b.prefix + ".index_k", deps=(index_source,),
                    m=producer_rows, n=plan.index_dim, k=input_width,
                    quant=wk_quant, category="indexer.wk", recipe="index_k",
                    activation_dtype=model.activation_dtype,
                    output_dtype=model.activation_dtype, **b.common,
                ))
            else:
                projected = b.matrix(
                    "index_k", (index_source,), producer_rows, plan.index_dim,
                    input_width,
                    category=_indexer_matrix_category(b.quant, "indexer.wk"),
                )
            if plan.index_input == "compressed":
                producer = next((op for op in reversed(b.ops)
                                 if op.op_id == index_source), None)
                if producer is not None and any(
                        value["output"] == "latent"
                        for value in produced_values(producer)):
                    # The packed compressor writes the rotated cache value and
                    # the pre-RoPE latent; the indexer consumes the latent port
                    # before any activation quantizer is inserted.
                    b.ops[-1] = set_input_ports(b.ops[-1], ({
                        "op": index_source, "output": "latent",
                        "source_op": index_source},))
            if _indexer_uses_rmsnorm(plan, model):
                normalized = index_rms_norm(
                    b, source=projected, rows=producer_rows, width=plan.index_dim)
            else:
                explicit_quant_indexer = (
                    runtime is not None
                    and select(dict(getattr(runtime, "kernels", {}).get("lowerings", {}) or {}),
                               "quant_lightning_indexer") is not None
                )
                fp32_wrapper = (
                    not _is_a5_runtime(runtime, machine)
                    and not explicit_quant_indexer
                    and quant.index_dtype in {"bf16", "fp16"}
                )
                normalized = index_layer_norm(
                    b, source=projected, rows=producer_rows, width=plan.index_dim,
                    native=not fp32_wrapper,
                    explicit_fp32_wrapper=fp32_wrapper)
            if index_rope:
                act_b = dtype_bytes(model.activation_dtype)
                normalized = b.vector(
                    "index_rotary", (normalized,), producer_rows * index_rope, ROPE_OPS,
                    family="rotary", shape=(producer_rows, plan.index_dim),
                    read_bytes_override=producer_rows * (
                        index_rope * act_b + index_rope * CANDIDATE_BYTES),
                    write_bytes_override=producer_rows * index_rope * act_b)
            quant_rows = producer_rows
        normalized = index_representation(
            b, "index_key", normalized, quant_rows, plan.index_dim,
            quant=quant, model=model, rotate_basis=_indexer_rotates_basis(plan, model))
        index_store = _emit_index_cache_store(
            b, plan, views, model, quant, normalized,
            ratio=ratio, index_rows=index_rows, index_rows_by_view=index_rows_by_view,
            quant_rows=quant_rows, c2_token_aligned=c2_token_aligned,
            runtime=runtime, native_indexer=native_indexer, parallel=parallel,
            idx_block=idx_block, sharded_index=sharded_index, machine=machine,
            store_rank_candidates=store_rank_candidates, raw_block=raw_block,
        )
    return index_store, pool_weights


def _select_index_rows(
    b, plan, pool_binding, views, model, quant, q_index, weights, index_store, q, runtime,
    hierarchy, shard_sizes, selection_deps, parallel, stage, ratio, shard, score_rows,
    *, machine=None, cp_interleave_size=None, physical_rank=None,
):
    if plan.index_kpool > 1:
        from mindie_sim.prediction.operators.sparse_indexer import index_pool
        native_binding = bool(
            pool_binding
            and pool_binding.name == NATIVE_KEYPOOL_IMPLEMENTATION)
        native_pool = native_binding
        if native_pool:
            from mindie_sim.prediction.operators.sparse_indexer.index_pool_native import supported as native_pool_supported
            native_pool = native_pool_supported(
                views=views, plan=plan, model=model, quant=quant)
        if native_pool:
            topk = pool_binding.module.selection(
                b, query=q_index, weights=weights, cache=index_store,
                views=views, plan=plan, quant=quant, padded_tokens=q,
                l2_bytes=getattr(b, "l2_bytes", None), runtime=runtime,
                hierarchy=hierarchy, shard_sizes=shard_sizes,
                physical_rank=physical_rank, parallel=parallel, machine=machine,
                **pool_binding.options)
        else:
            analytic_options = ({} if native_binding else
                                (pool_binding.options if pool_binding else {}))
            topk = index_pool.selection(
                b, query=q_index, weights=weights, cache=index_store,
                views=views, plan=plan, quant=quant, padded_tokens=q,
                l2_bytes=getattr(b, "l2_bytes", None), runtime=runtime,
                **analytic_options)
    else:
        lowerings = (runtime.kernels.get("lowerings", {}) if runtime else {})
        quant_indexer = (select(lowerings, "quant_lightning_indexer")
                         if runtime else None)
        bf16_indexer = select(lowerings, "lightning_indexer") \
            if runtime and not quant_indexer else None
        if quant_indexer:
            topk = quant_indexer.module.declare(
                b, deps=selection_deps, views=views, plan=plan,
                quant=quant, parallel=parallel, prefix_cache_mode=runtime.prefix_cache_mode,
                **quant_indexer.options)
            declare_indexer(
                b.ops, prefix=b.prefix, plan=plan, parallel=parallel,
                model=model, quant=quant, views=views, padded_tokens=q,
                scores=topk, topk=topk)
            # This explicit arch22 source owns its per-query key/scale
            # ledger. Shared capacity does not establish batch read reuse.
        elif (bf16_indexer is not None and len(views) == 1
              and parallel.cp == 1 and not plan.candidate_filter):
            from mindie_sim.prediction.operators.sparse_indexer import bf16_lightning as li
            if bf16_indexer.name != li.IMPLEMENTATION:
                raise ModelingError("mla_lightning_indexer_selected_not_bf16")
            view = views[0]
            w_op = next((item for item in reversed(b.ops) if item.op_id == weights), None)
            w_out_dtype = (
                str(dict(w_op.metadata).get("output_format") or w_op.dtype
                    or model.activation_dtype).strip().lower()
                if w_op is not None else model.activation_dtype
            )
            topk = b.add(li.make_op(
                query_rows=view.q_len, context=view.context,
                op_id=b.prefix + ".index_lightning",
                deps=tuple(selection_deps), stage=stage, layer=plan.index,
                requests=(view.request_id,), phase=b.common["phase"],
                heads=plan.index_heads, head_dim=plan.index_dim,
                topk=plan.index_topk, compression=ratio,
                page_size=int(getattr(runtime, "kv_block_size", 0)
                              or DEFAULT_BLOCK_SIZE),
                kv_heads=1, query_dtype=model.activation_dtype,
                cache_dtype=index_cache_dtype(model), weight_dtype=w_out_dtype,
                query_lengths=(view.q_len,), context_lengths=(view.context,),
                binding_revision=str(getattr(runtime, "vllm_version", "unknown")),
                runtime_version=str(getattr(runtime, "vllm_version", "unknown")),
            ))
            declare_indexer(
                b.ops, prefix=b.prefix, plan=plan, parallel=parallel,
                model=model, quant=quant, views=views, padded_tokens=q,
                scores=topk, topk=topk)
        else:
            topk = semantic_index_selection(
                b, deps=selection_deps, views=views, plan=plan, model=model,
                quant=quant, parallel=parallel, shard=shard, prefix=b.prefix,
                padded_tokens=q, runtime=runtime, score_rows=score_rows,
                l2_bytes=getattr(b, "l2_bytes", None), machine=machine,
                cp_interleave_size=cp_interleave_size)
    return topk
