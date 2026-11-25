"""Shared sparse indexer contract, work ledger, and finite pipeline lowering.

[Role] Convert query and S2 geometry into matrix, transfer, reduction and
  selection work. Bind leaf-specific score handoffs and selection schedules.
[Invariants] The descriptor and executable program use the same phase work.
  Physical tiles can include padding while useful dot FLOPs and the valid
  candidate extent follow the live rows.
[Timeouts & Locks] Planning uses a bounded set of tile shapes. Execution keeps
  only the active owners and their live tiles; no I/O or locks.
[Side-Effect Policy] Builds immutable plans for the caller's virtual engine.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from typing import Any, Literal

from mindie_sim.catalog.attention_common import _floor_sum
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...ownership import read_role
from ...program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.ring_buffer import RingBuffer
from ..attention.common import planning_sequence_spans, sequence_layout_metadata
from ..contracts import OperatorPlanningContext
from ._lowering import S1Partition, S2Partition, _DisjointOwnerProgram, align, ceil_div, request_program, sort_ops

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
FAMILY = "sparse_indexer"

LIGHTNING_ABI = "aclnnLightningIndexer"
LIGHTNING_V2_ABI = "aclnnLightningIndexerV2"
QLI_V2_ABI = "aclnnQuantLightningIndexerV2"
VLLM_QLI_ABI = "VllmQuantLightningIndexer"
VLLM_QLI_ABIS = frozenset({VLLM_QLI_ABI, "vllm_quant_lightning_indexer_v2", "npu_quant_lightning_indexer_v2"})
QSA_ABI = "aclnnLightningIndexerQSA"
# The manifest's weight-free entry records the analytical overlay's provenance.
# The mounted QSA caller materializes uniform weights at the native boundary.
QSA_OVERLAY_SOURCE = SourceRef(
    repository="mindie-agent/sim", revision="c1bc55e59facc21a6a775da346e7ad7ae4909cfd",
    relative_path="mindie_sim/stacks/vllm_ascend/manifest.json",
    source_sha256="3a68c55fe152187f2ac4237622152c6b1be6cd514880ef210e3b6741d9518b7e",
)
MSA_ABI = "aclnnMsaIndexScore"
MSA_V2_ABI = "aclnnMsaIndexScoreV2"
REGION_ABI = "npu_indexer_region_score"
DSL_QSLI_ABI = "tilelang_dsl.quant_sparse_lightning_indexer"
MSA_ABIS = frozenset({MSA_ABI, MSA_V2_ABI})
QUANT_ABIS = VLLM_QLI_ABIS | {QLI_V2_ABI}
SUPPORTED_ABIS = MSA_ABIS | QUANT_ABIS | {
    LIGHTNING_ABI, LIGHTNING_V2_ABI, QSA_ABI, REGION_ABI, DSL_QSLI_ABI,
}

MSA_INDEX_SCORE_RECIPE = "msa_index_score_arch35"
MSA_INDEX_SCORE_V2_RECIPE = "msa_index_score_v2_arch35"
STEP5_REGION_SCORE_RECIPE = "step5_indexer_region_score_arch35"
LIGHTNING_INDEXER_QSA_RECIPE = "lightning_indexer_qsa_arch35"
SCORER_RECIPES = {
    MSA_ABI: MSA_INDEX_SCORE_RECIPE, MSA_V2_ABI: MSA_INDEX_SCORE_V2_RECIPE,
    REGION_ABI: STEP5_REGION_SCORE_RECIPE, QSA_ABI: LIGHTNING_INDEXER_QSA_RECIPE,
}
_CACHE_ALIASES = {"fp8": "fp8_e4m3", "fp4": "mxfp4", "a4c4": "mxfp4"}
_CACHE_DTYPES = frozenset({"bf16", "fp8_e4m3", "int8", "mxfp4"})
BF16_BYTES = 2
FP32_BYTES = 4
INT32_BYTES = 4
PAIR_BYTES = FP32_BYTES + INT32_BYTES
MX_SCALE_GROUP = 32
MSA_SHORT_BLOCK = 16
MSA_SCORE_STRIDE_ALIGNMENT = 16
MSA_RETURN_INDICES_VALUES = 2
SORT_GROUP = 32
MERGE_FAN_IN = 4
ARCH22_FLUSH_BLOCKS = 4
ARCH22_SMALL_QUERY_MAX = 4
ARCH22_TOPK_LIMIT = 2048
ARCH22_SPLIT_RECORD_BYTES = 128  # Sixteen int64 fields per owner.
QLI_SPLIT_LIST_ALIGNMENT = 16
DEFAULT_S2_TILE = 512
SCORE_SLOTS = 2
SCORE_SLOT_NAME = "index_score"
DEFAULT_INDEX_DIM = 128
REGION_INDEX_DIM = 256
LIGHTNING_V2_MAX_HEAD_GROUP = 64
ARCH22_QLI_HEADS = 64
ARCH22_QLI_DIM = 128
ARCH22_QLI_HEAD_REDUCE_M = 16
RADIX_BINS = 256
DSL_TRUNK_ITEMS = 16384
MIN_INTERIOR_TILES = 3
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/lightning_indexer_v2/op_host/lightning_indexer_v2_tiling.cpp",
        source_sha256="b8988f5bdc1ccf8900c2a7853c4edf012f8cb99f6e9b5af40af56167bcd9aade",
    ),
)


def is_block_msa_indexer(plan) -> bool:
    """Single source of truth for GQA block-maxima MSA indexers (Q heads sharded across TP)."""
    if plan is None or str(getattr(plan, "attention", "") or "") != "gqa":
        return False
    cand_blk = int(getattr(plan, "candidate_block_size", 0) or 0)
    cand_topk = int(getattr(plan, "candidate_topk_blocks", 0) or 0)
    idx_topk = int(getattr(plan, "index_topk", 0) or 0)
    return (
        cand_blk > 0
        and cand_topk > 0
        and not bool(getattr(plan, "index_kpool_compress", False))
        and (idx_topk <= 0 or cand_topk < idx_topk)
    )


def is_region_csa_indexer(plan) -> bool:
    """Single source of truth for compressed region-score CSA indexers (cand_block == kpool)."""
    if plan is None or str(getattr(plan, "attention", "") or "") != "gqa":
        return False
    kpool = int(getattr(plan, "index_kpool", 1) or 1)
    cand_blk = int(getattr(plan, "candidate_block_size", 0) or 0)
    cand_topk = int(getattr(plan, "candidate_topk_blocks", 0) or 0)
    idx_topk = int(getattr(plan, "index_topk", 0) or 0)
    return (
        bool(getattr(plan, "index_kpool_compress", False))
        and kpool > 1
        and cand_blk == kpool
        and cand_topk > 0
        and (idx_topk <= 0 or cand_topk * cand_blk == idx_topk)
    )


def is_qsa_pooled_indexer(plan) -> bool:
    """Single source of truth for pooled-K QSA indexers (LightningIndexerV2)."""
    if plan is None or str(getattr(plan, "attention", "") or "") != "gqa":
        return False
    kpool = int(getattr(plan, "index_kpool", 1) or 1)
    return (
        bool(getattr(plan, "index_kpool_compress", False))
        and kpool > 1
        and not is_region_csa_indexer(plan)
    )


def _integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"sparse_indexer_invalid_{name}:{value!r}")


def unsupported(code: str, message: str) -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(
        capability=FAMILY, entry="sparse_indexer.build",
        code=f"sparse_indexer_{code}", message=message,
    )


def _causal_request_pairs(
    q: int, scanned_rows: int, compress_ratio: int, residual_tokens: int,
) -> int:
    """Exact sum of causally visible compressed key rows for one right-aligned request."""
    if q <= 0 or scanned_rows <= 0:
        return 0
    total_orig = scanned_rows * compress_ratio + residual_tokens
    if q <= total_orig:
        return _floor_sum(q, compress_ratio, 1, total_orig - q + 1)
    return _floor_sum(total_orig, compress_ratio, 1, 1)


def _causal_active_queries(
    q: int, scanned_rows: int, compress_ratio: int, residual_tokens: int,
) -> int:
    """Queries in one right-aligned causal request with at least one visible key row."""
    if q <= 0 or scanned_rows <= 0:
        return 0
    total_orig = scanned_rows * compress_ratio + residual_tokens
    return min(q, max(0, total_orig - compress_ratio + 1))


def _causal_block_rows(
    req_q: int, block_end: int, scanned_rows: int, compress_ratio: int, residual_tokens: int,
) -> int:
    """Causally visible compressed S2 extent for a query block ending at ``block_end``."""
    if scanned_rows <= 0 or block_end <= 0:
        return 0
    total_orig = scanned_rows * compress_ratio + residual_tokens
    end_orig = total_orig - (req_q - block_end)
    if end_orig <= 0:
        return 0
    return min(scanned_rows, end_orig // compress_ratio)


def _sum_floor_range(lo: int, hi: int, a: int, b: int, d: int) -> int:
    """Exact sum_{i=lo}^{hi-1} floor((a + b * i) / d) for a + b * lo >= 0, b >= 0, d > 0."""
    if hi <= lo:
        return 0
    return _floor_sum(hi - lo, d, b, a + b * lo)


def _count_floor_mod(lo: int, hi: int, a: int, b: int, d: int, m: int, r: int) -> int:
    """Exact count of i in [lo, hi) with floor((a + b * i) / d) % m == r."""
    if hi <= lo:
        return 0
    return (
        _sum_floor_range(lo, hi, a + (m - r) * d, b, m * d)
        - _sum_floor_range(lo, hi, a + (m - 1 - r) * d, b, m * d)
    )


def _first_ge(lo: int, hi: int, a: int, b: int, d: int, target: int) -> int:
    """Smallest i in [lo, hi] such that (a + b * i) // d >= target."""
    need = target * d - a
    if need <= lo * b:
        return lo
    idx = ceil_div(need, b)
    return hi if idx >= hi else idx


def _span_scanned_rows(
    kv_tokens: int, max_scanned_rows: int, compress_ratio: int, max_residual: int,
) -> tuple[int, int]:
    step = max(1, compress_ratio)
    rows, rem = divmod(max(0, kv_tokens), step)
    if rows < max_scanned_rows or (rows == max_scanned_rows and rem <= max_residual):
        return rows, rem
    return max_scanned_rows, max_residual


def _region_request_pairs(q: int, kv_tokens: int, scanned_rows: int, compress_ratio: int) -> int:
    if q <= 0 or scanned_rows <= 0:
        return 0
    step = max(1, compress_ratio)
    history_base = max(0, kv_tokens - q)
    total = idx = 0
    while idx < q:
        raw_regions = (history_base + idx) // step
        regions = min(scanned_rows, raw_regions)
        until = (
            q if raw_regions >= scanned_rows
            else min(q, (raw_regions + 1) * step - history_base)
        )
        total += (until - idx) * regions
        idx = until
    return total


def _region_active_queries(q: int, kv_tokens: int, scanned_rows: int, compress_ratio: int) -> int:
    if q <= 0 or scanned_rows <= 0:
        return 0
    step = max(1, compress_ratio)
    history_base = max(0, kv_tokens - q)
    if history_base >= step:
        return q
    return max(0, q - (step - history_base))


def _spans_from_request_views(request_views: Sequence[Any]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for req in request_views:
        if getattr(req, "padding", False):
            continue
        raw_ranges = tuple(
            (int(start), int(end))
            for start, end in (getattr(req, "query_ranges", ()) or ())
            if int(end) > int(start)
        )
        if raw_ranges:
            req_ctx = int(req.context)
            for start, end in raw_ranges:
                spans.append((end - start, req_ctx + end))
            continue
        q = int(getattr(req, "logical_q_len", getattr(req, "q_len", 0)))
        kv = int(getattr(req, "kv_len", int(getattr(req, "context", 0)) + q))
        spans.append((q, kv))
    return tuple(spans)


def _kv_lens_to_spans(query_tokens: int, batch_size: int, kv_lens: Sequence[int]) -> tuple[tuple[int, int], ...]:
    q_base, q_rem = divmod(query_tokens, batch_size)
    return tuple(
        (q_base + (1 if i < q_rem else 0), int(kv))
        for i, kv in enumerate(kv_lens)
    )


@dataclass(frozen=True, slots=True)
class SparseIndexerContract:
    query_tokens: int
    scanned_rows: int
    index_heads: int
    index_dim: int
    block_size: int = 1
    topk_items: int = ARCH22_TOPK_LIMIT
    fused_topk: bool = True
    cache_dtype: str = "bf16"
    has_head_weights: bool = True
    has_mask3_sort: bool = False
    abi: str = LIGHTNING_ABI
    s2_splits: int = 1
    s2_tile_size: int = DEFAULT_S2_TILE
    batch_size: int = 1
    shared_prefix: bool = False
    causal: bool = True
    is_sparse_count_over_2k: bool = False
    return_choice: int | None = None
    compress_ratio: int = 1
    residual_tokens: int = 0
    native_split_core: bool = False
    request_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.abi in MSA_ABIS and self.return_choice is not None:
            object.__setattr__(self, "fused_topk", self.return_choice != 0)
        if self.request_spans:
            normalized = tuple(
                tuple(span) if isinstance(span, (list, tuple)) else span
                for span in self.request_spans
            )
            object.__setattr__(self, "request_spans", normalized)

    def _span_pairs(self, q_i: int, kv_i: int) -> tuple[int, int, int]:
        scanned_i, residual_i = _span_scanned_rows(
            kv_i, self.scanned_rows, self.compress_ratio, self.residual_tokens,
        )
        if q_i <= 0 or scanned_i <= 0:
            return 0, scanned_i, residual_i
        if not self.causal:
            return q_i * scanned_i, scanned_i, residual_i
        if self.abi == REGION_ABI:
            return _region_request_pairs(q_i, kv_i, self.scanned_rows, self.compress_ratio), scanned_i, residual_i
        return _causal_request_pairs(q_i, scanned_i, self.compress_ratio, residual_i), scanned_i, residual_i

    @property
    def useful_pairs(self) -> int:
        if self.query_tokens <= 0 or self.scanned_rows <= 0:
            return 0
        if self.request_spans:
            return sum(self._span_pairs(q_i, kv_i)[0] for q_i, kv_i in self.request_spans)
        if not self.causal:
            return self.query_tokens * self.scanned_rows
        q_base, extra = divmod(self.query_tokens, self.batch_size)
        total = 0
        if self.abi == REGION_ABI:
            kv_tokens = self.scanned_rows * max(1, self.compress_ratio) + self.residual_tokens
            if extra:
                total += extra * _region_request_pairs(
                    q_base + 1, kv_tokens, self.scanned_rows, self.compress_ratio,
                )
            if q_base:
                total += (self.batch_size - extra) * _region_request_pairs(
                    q_base, kv_tokens, self.scanned_rows, self.compress_ratio,
                )
            return total
        if extra:
            total += extra * _causal_request_pairs(
                q_base + 1, self.scanned_rows, self.compress_ratio, self.residual_tokens,
            )
        if q_base:
            total += (self.batch_size - extra) * _causal_request_pairs(
                q_base, self.scanned_rows, self.compress_ratio, self.residual_tokens,
            )
        return total

    @property
    def matrix_flops(self) -> int:
        return 2 * self.useful_pairs * self.index_heads * self.index_dim

    @property
    def matrix_dtype(self) -> str:
        return self.cache_dtype

    @property
    def cache_row_bytes(self) -> int:
        if self.cache_dtype == "int8" and self.abi in QUANT_ABIS:
            return self.index_dim + BF16_BYTES
        return cache_row_bytes(self.cache_dtype, self.index_dim)

    @property
    def score_items(self) -> int:
        return ceil_div(self.scanned_rows, self.block_size)

    @property
    def packed_index_items(self) -> int:
        return self.topk_items + self.compress_ratio - 1

    @property
    def arch22_retained_items(self) -> int:
        return self.topk_items if self.is_sparse_count_over_2k else max(self.topk_items, ARCH22_TOPK_LIMIT)

    @property
    def msa_score_heads(self) -> int:
        return 1 if self.block_size == MSA_SHORT_BLOCK else self.index_heads

    @property
    def msa_score_stride(self) -> int:
        return align(max(1, self.score_items), MSA_SCORE_STRIDE_ALIGNMENT)

    @property
    def aiv_scale_bytes(self) -> int:
        if self.abi in QUANT_ABIS and self.cache_dtype in {"fp8_e4m3", "int8"}:
            return self.cache_row_bytes - self.index_dim
        return 0


def cache_row_bytes(cache_dtype: str, dim: int) -> int:
    """Packed E2M1 payload and E8M0 group scales, or the dense row storage."""
    cache_dtype = _CACHE_ALIASES.get(cache_dtype, cache_dtype)
    if cache_dtype == "mxfp4":
        return ceil_div(dim, 2) + ceil_div(dim, MX_SCALE_GROUP)
    if cache_dtype in {"fp8_e4m3", "int8"}:
        return dim + FP32_BYTES  # Per-row quantization scale.
    if cache_dtype == "bf16":
        return dim * BF16_BYTES
    raise unsupported("cache_dtype", f"Unsupported index cache dtype {cache_dtype!r}")


def validate_contract(contract: SparseIndexerContract) -> None:
    for name in ("index_heads", "index_dim", "block_size", "topk_items",
                 "s2_splits", "s2_tile_size", "batch_size"):
        _integer(name, getattr(contract, name))
    _integer("query_tokens", contract.query_tokens, minimum=0)
    _integer("scanned_rows", contract.scanned_rows, minimum=0)
    _integer("compress_ratio", contract.compress_ratio)
    _integer("residual_tokens", contract.residual_tokens, minimum=0)
    if contract.residual_tokens >= contract.compress_ratio:
        raise ModelingError("sparse_indexer_residual_tokens_must_fit_compress_ratio")
    if contract.abi not in SUPPORTED_ABIS:
        raise unsupported("abi", f"Unsupported sparse indexer ABI {contract.abi!r}")
    if contract.cache_dtype not in _CACHE_DTYPES:
        raise unsupported("cache_dtype", f"Unsupported index cache dtype {contract.cache_dtype!r}")
    if contract.abi in QUANT_ABIS | {LIGHTNING_V2_ABI} and contract.has_head_weights is not True:
        raise ModelingError(f"sparse_indexer_v2_requires_head_weights:{contract.abi}")
    if contract.abi == QSA_ABI and type(contract.has_head_weights) is not bool:
        raise ModelingError("sparse_indexer_invalid_has_head_weights")
    choice = contract.return_choice
    if choice is not None and (contract.abi not in MSA_ABIS or type(choice) is not int
                               or not 0 <= choice <= MSA_RETURN_INDICES_VALUES):
        raise ModelingError(f"sparse_indexer_invalid_return_choice:{choice!r}")
    if (contract.abi in MSA_ABIS and contract.block_size == MSA_SHORT_BLOCK
            and contract.fused_topk):
        raise ModelingError("sparse_indexer_msa_block16_disallows_fused_topk")
    if contract.abi == MSA_ABI and contract.fused_topk:
        raise ModelingError("sparse_indexer_msa_v1_returns_scores")
    if contract.request_spans:
        span_q = 0
        for span in contract.request_spans:
            if not isinstance(span, tuple) or len(span) != 2:
                raise ModelingError(f"sparse_indexer_invalid_request_span:{span!r}")
            q_i, kv_i = span
            _integer("request_query_tokens", q_i, minimum=0)
            _integer("request_kv_tokens", kv_i, minimum=0)
            span_q += q_i
        if span_q != contract.query_tokens:
            raise ModelingError(
                f"sparse_indexer_request_spans_query_mismatch:{span_q}!={contract.query_tokens}"
            )


@dataclass(frozen=True, slots=True)
class PhaseWork:
    name: str
    work: Work = field(default_factory=Work)
    aic: bool = False
    metadata: dict[str, int | bool | str] = field(default_factory=dict)


def vector(name: str, ops: int, **metadata: int | bool | str) -> PhaseWork:
    return PhaseWork(name, Work(resources={"vector": ops}), metadata=metadata)


def pair_merge(name: str, incoming: int, retained: int, output: int, **metadata) -> PhaseWork:
    return vector(name, incoming + retained, input_pair_bytes=(incoming + retained) * PAIR_BYTES,
                  output_pair_bytes=output * PAIR_BYTES, **metadata)


def indexer_contract(op: OpSpec, ctx: OperatorPlanningContext, default_abi: str) -> SparseIndexerContract:
    metadata = dict(op.metadata)
    raw = dict(metadata.get("indexer_contract") or metadata.get("scorer_geometry") or {})
    abi = raw.get("abi") or metadata.get("native_kind")
    if abi is not None and ctx.requested_abi is not None and abi != ctx.requested_abi:
        raise unsupported("abi_conflict", f"Requested {ctx.requested_abi!r} for an {abi!r} operator")
    abi = ctx.requested_abi or abi or default_abi
    required = {"query_tokens", "scanned_rows", "index_heads", "index_dim"}
    if not required <= raw.keys():
        raise unsupported("missing_geometry", f"{op.op_id} requires {sorted(required - raw.keys())}")
    raw.setdefault("has_head_weights", abi not in MSA_ABIS)
    dtype = raw.get("cache_dtype", "bf16").lower()
    raw["cache_dtype"] = _CACHE_ALIASES.get(dtype, dtype)
    raw["abi"] = abi
    if not raw.get("request_spans"):
        if ctx.requests:
            req_spans = _spans_from_request_views(ctx.requests)
            if req_spans and sum(q for q, _ in req_spans) == int(raw["query_tokens"]):
                raw["request_spans"] = req_spans
        elif isinstance(metadata.get("kv_lens"), (list, tuple)):
            batch = int(raw.get("batch_size", 1))
            raw_kv = metadata["kv_lens"]
            if len(raw_kv) == batch:
                raw["request_spans"] = _kv_lens_to_spans(int(raw["query_tokens"]), batch, raw_kv)
    contract = SparseIndexerContract(**{f.name: raw[f.name] for f in fields(SparseIndexerContract)
                                       if f.name in raw})
    validate_contract(contract)
    return contract


def _has_varying_causal_rows(contract: SparseIndexerContract, query_block_size: int) -> bool:
    """True iff causal right-aligned masking reduces visible rows on earlier query blocks."""
    if not contract.causal or contract.scanned_rows <= 0:
        return False
    max_q = ceil_div(contract.query_tokens, contract.batch_size)
    if max_q <= query_block_size:
        return False
    return _causal_block_rows(
        max_q, query_block_size, contract.scanned_rows,
        contract.compress_ratio, contract.residual_tokens,
    ) < contract.scanned_rows


def _request_contract_spans(contract: SparseIndexerContract, ctx: OperatorPlanningContext):
    step = max(1, contract.compress_ratio)
    if not ctx.requests:
        if not contract.request_spans:
            return
        q_offset = kv_offset = 0
        for q, kv in contract.request_spans:
            if contract.abi == REGION_ABI:
                history_base = max(0, kv - q)
                idx = 0
                while idx < q:
                    raw_regions = (history_base + idx) // step
                    regions = min(contract.scanned_rows, raw_regions)
                    until = (
                        q if raw_regions >= contract.scanned_rows
                        else min(q, (raw_regions + 1) * step - history_base)
                    )
                    yield (
                        replace(contract, query_tokens=until - idx, batch_size=1,
                                scanned_rows=regions, causal=False, request_spans=()),
                        (q_offset + idx, kv_offset),
                    )
                    idx = until
            else:
                scanned, residual = _span_scanned_rows(
                    kv, contract.scanned_rows, step, contract.residual_tokens,
                )
                yield (
                    replace(contract, query_tokens=q, batch_size=1,
                            scanned_rows=scanned, residual_tokens=residual, request_spans=()),
                    (q_offset, kv_offset),
                )
            q_offset += q
            kv_offset += kv
        return
    spans = planning_sequence_spans(ctx, "TND")
    reqs = tuple(ctx.requests) if ctx.requests and len(ctx.requests) == len(spans) else (None,) * len(spans)
    for req, (q, kv, q_offset, kv_offset, padding) in zip(reqs, spans, strict=True):
        if padding:
            continue
        raw_ranges = tuple(
            (int(start), int(end))
            for start, end in (getattr(req, "query_ranges", ()) or ())
            if int(end) > int(start)
        ) if req is not None else ()
        if raw_ranges:
            req_ctx = int(req.context)
            local_q_offset = q_offset
            for start, end in raw_ranges:
                chunk_q = end - start
                if contract.abi == REGION_ABI:
                    history_base = req_ctx + start
                    idx = 0
                    while idx < chunk_q:
                        raw_regions = (history_base + idx) // step
                        regions = min(contract.scanned_rows, raw_regions)
                        until = (
                            chunk_q if raw_regions >= contract.scanned_rows
                            else min(chunk_q, (raw_regions + 1) * step - history_base)
                        )
                        yield (
                            replace(contract, query_tokens=until - idx, batch_size=1,
                                    scanned_rows=regions, causal=False, request_spans=()),
                            (local_q_offset + idx, kv_offset),
                        )
                        idx = until
                else:
                    chunk_kv = req_ctx + end
                    scanned, residual = _span_scanned_rows(
                        chunk_kv, contract.scanned_rows, step, contract.residual_tokens,
                    )
                    yield (
                        replace(contract, query_tokens=chunk_q, batch_size=1,
                                scanned_rows=scanned, residual_tokens=residual, request_spans=()),
                        (local_q_offset, kv_offset),
                    )
                local_q_offset += chunk_q
            continue
        if contract.abi == REGION_ABI:
            history_base = max(0, kv - q)
            idx = 0
            while idx < q:
                raw_regions = (history_base + idx) // step
                regions = min(contract.scanned_rows, raw_regions)
                until = (
                    q if raw_regions >= contract.scanned_rows
                    else min(q, (raw_regions + 1) * step - history_base)
                )
                yield (
                    replace(contract, query_tokens=until - idx, batch_size=1,
                            scanned_rows=regions, causal=False, request_spans=()),
                    (q_offset + idx, kv_offset),
                )
                idx = until
        else:
            scanned, residual = _span_scanned_rows(
                kv, contract.scanned_rows, step, contract.residual_tokens,
            )
            yield (
                replace(contract, query_tokens=q, batch_size=1,
                        scanned_rows=scanned, residual_tokens=residual, request_spans=()),
                (q_offset, kv_offset),
            )


def request_contracts(contract: SparseIndexerContract, ctx: OperatorPlanningContext):
    for request, _ in _request_contract_spans(contract, ctx):
        yield request


def _spans_match_uniform_contract(contract: SparseIndexerContract) -> bool:
    """Return True when ``request_spans`` carries no shape detail beyond the uniform contract."""
    spans = contract.request_spans
    if not spans or contract.abi == REGION_ABI or len(spans) != contract.batch_size:
        return False
    q_base, q_rem = divmod(contract.query_tokens, contract.batch_size)
    for i, (q_i, kv_i) in enumerate(spans):
        if q_i != q_base + (1 if i < q_rem else 0):
            return False
        scanned_i, residual_i = _span_scanned_rows(
            kv_i, contract.scanned_rows, contract.compress_ratio, contract.residual_tokens,
        )
        if scanned_i != contract.scanned_rows or residual_i != contract.residual_tokens:
            return False
    return True


def _minimum_indexer_io(contract: SparseIndexerContract) -> tuple[float, float, float, float]:
    """Conservative physical (hbm, mte2, mte3, vector_ops) lower bound for L* soundness."""
    pairs = contract.useful_pairs
    if contract.request_spans:
        active_q = 0
        max_scanned = 0
        block_max_raw = 0
        for q_i, kv_i in contract.request_spans:
            span_pairs, scanned_i, residual_i = contract._span_pairs(q_i, kv_i)
            if span_pairs <= 0:
                continue
            if contract.abi == REGION_ABI and contract.causal:
                visible_i = (
                    min(contract.scanned_rows, (kv_i - 1) // max(1, contract.compress_ratio))
                    if kv_i > 0 else 0
                )
            else:
                visible_i = scanned_i
            max_scanned = max(max_scanned, visible_i)
            if not contract.causal:
                active_q += q_i
            elif contract.abi == REGION_ABI:
                active_q += _region_active_queries(q_i, kv_i, contract.scanned_rows, contract.compress_ratio)
            else:
                active_q += _causal_active_queries(q_i, scanned_i, contract.compress_ratio, residual_i)
            if contract.block_size > 1:
                block_max_raw += max(0, span_pairs - q_i * ceil_div(scanned_i, contract.block_size))
        k_rows = max_scanned
    else:
        if pairs and contract.causal:
            q_base, extra = divmod(contract.query_tokens, contract.batch_size)
            if contract.abi == REGION_ABI:
                kv_tokens = contract.scanned_rows * max(1, contract.compress_ratio) + contract.residual_tokens
                active_q = (
                    extra * _region_active_queries(q_base + 1, kv_tokens, contract.scanned_rows, contract.compress_ratio)
                    + (contract.batch_size - extra) * _region_active_queries(q_base, kv_tokens, contract.scanned_rows, contract.compress_ratio)
                )
                k_rows = (
                    min(contract.scanned_rows, (kv_tokens - 1) // max(1, contract.compress_ratio))
                    if kv_tokens > 0 else 0
                )
            else:
                active_q = (
                    extra * _causal_active_queries(q_base + 1, contract.scanned_rows, contract.compress_ratio, contract.residual_tokens)
                    + (contract.batch_size - extra) * _causal_active_queries(q_base, contract.scanned_rows, contract.compress_ratio, contract.residual_tokens)
                )
                k_rows = contract.scanned_rows
        else:
            active_q = contract.query_tokens if pairs else 0
            k_rows = contract.scanned_rows if pairs else 0
        block_max_raw = max(0, pairs - contract.query_tokens * contract.score_items) if (contract.block_size > 1 and pairs) else 0
    q_elem = float(cache_row_bytes(contract.cache_dtype, contract.index_dim))
    q_bytes = float(active_q * contract.index_heads) * q_elem
    k_bytes = float(k_rows) * float(contract.cache_row_bytes) if pairs else 0.0
    out_items = contract.topk_items if contract.fused_topk else contract.score_items
    out_heads = contract.msa_score_heads if contract.abi in MSA_ABIS else 1
    out_elem = float(INT32_BYTES if contract.fused_topk else FP32_BYTES)
    out_bytes = float(contract.query_tokens * out_heads * out_items) * out_elem
    if contract.abi in MSA_ABIS:
        head_max = (contract.index_heads - contract.msa_score_heads) * pairs
        block_max = block_max_raw * contract.msa_score_heads
        vec_ops = float(head_max + block_max)
    elif contract.abi in QUANT_ABIS and contract.cache_dtype == "int8":
        # Arch22 QLI V2 executes head reduction on FP16 Cube, leaving key descale on AIV.
        vec_ops = float(pairs)
    else:
        vec_ops = float(pairs * contract.index_heads)
    read_b = q_bytes + k_bytes
    return read_b + out_bytes, read_b, out_bytes, vec_ops


def make_op(
    contract: SparseIndexerContract, *, op_id: str, deps: tuple[str, ...] = (),
    stage: int = 0, layer: int = 0, requests: tuple[str, ...] = (),
    phase: str = "decode", recipe: str | None = None, source: SourceRef | None = None,
    request_views: Sequence[Any] | None = None, kv_lens: Sequence[int] | None = None,
    query_lens: Sequence[int] | None = None,
) -> OpSpec:
    dtype = _CACHE_ALIASES.get(contract.cache_dtype, contract.cache_dtype)
    if dtype != contract.cache_dtype:
        contract = replace(contract, cache_dtype=dtype)
    if not contract.request_spans:
        if request_views is not None:
            spans = _spans_from_request_views(request_views)
            if spans:
                contract = replace(contract, request_spans=spans)
        elif kv_lens is not None:
            if len(kv_lens) != contract.batch_size:
                raise ModelingError(
                    f"sparse_indexer_kv_lens_batch_mismatch:{len(kv_lens)}!={contract.batch_size}"
                )
            if query_lens is not None:
                if len(query_lens) != contract.batch_size:
                    raise ModelingError(
                        f"sparse_indexer_query_lens_batch_mismatch:{len(query_lens)}!={contract.batch_size}"
                    )
                inferred = tuple((int(q_i), int(kv_i)) for q_i, kv_i in zip(query_lens, kv_lens))
            else:
                inferred = _kv_lens_to_spans(contract.query_tokens, contract.batch_size, kv_lens)
                if contract.causal and any(q_i > kv_i > 0 for q_i, kv_i in inferred):
                    raise ModelingError(
                        "sparse_indexer_kv_lens_requires_query_lens"
                    )
            contract = replace(contract, request_spans=inferred)
    validate_contract(contract)
    items = contract.topk_items if contract.fused_topk else contract.score_items
    shape = (contract.query_tokens, items)
    padded_shape = shape
    returns = ()
    meta = {"indexer_contract": asdict(contract), "native_kind": contract.abi}
    if contract.request_spans and len(contract.request_spans) == contract.batch_size:
        q_base, q_rem = divmod(contract.query_tokens, contract.batch_size)
        if all(q_i == q_base + (1 if i < q_rem else 0) for i, (q_i, _) in enumerate(contract.request_spans)):
            meta["kv_lens"] = tuple(kv_i for _, kv_i in contract.request_spans)
    if contract.abi in MSA_ABIS:
        shape = (contract.msa_score_heads, *shape)
        padded_shape = shape if contract.fused_topk else (*shape[:-1], contract.msa_score_stride)
        meta["output_shape"] = padded_shape
        if contract.return_choice == MSA_RETURN_INDICES_VALUES:
            returns = ("values",)
            meta.update(values_shape=shape, values_dtype="fp32")
    if source is not None:
        meta["native_source"] = source.to_dict()
        # Existing scorer callers expose this structural record in exports.
        meta["scorer_geometry"] = meta["indexer_contract"].copy()
    hbm_bytes, mte2_bytes, mte3_bytes, vector_ops = _minimum_indexer_io(contract)
    return OpSpec(
        op_id=op_id, deps=deps, family="indexer", recipe=recipe or FAMILY,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=padded_shape, launch_shape=padded_shape,
        dtype="int32" if contract.fused_topk else "fp32", flops=contract.matrix_flops,
        matrix_flops=((contract.matrix_dtype, contract.matrix_flops),),
        vector_ops=vector_ops, hbm_bytes=hbm_bytes, mte2_bytes=mte2_bytes, mte3_bytes=mte3_bytes,
        source_boundary="kernel", returns=returns, metadata=tuple(sorted(meta.items())),
    )


TileSelection = Callable[[SparseIndexerContract, int, int, int, int], tuple[PhaseWork, ...]]
OwnerSelection = Callable[[SparseIndexerContract, int], tuple[PhaseWork, ...]]


@dataclass(frozen=True, slots=True)
class Lowering:
    implementation: str
    tiling: str
    tile_size: int
    score_path: str
    tile_selection: TileSelection | None = None
    owner_selection: OwnerSelection | None = None
    flush_period: int = 1
    score_align: int = 1
    slots: int = 1
    gm_score_row: bool = False
    aivs_per_aic: int = 1
    query_block_size: int = 1
    handoff_query_alignment: int = 1
    gm_key_staging: bool = False
    score_row_item_bytes: int = FP32_BYTES
    owner_score_row: bool = False
    handoff_item_bytes: int = FP32_BYTES
    head_weight_item_bytes: int | None = None
    handoff_alignment: int = 1
    handoff_columns: Callable[[int], int] | None = None
    score_slot_bytes: int = 0
    score_row_read_align: int = 1
    cube_head_reduce_fp16: bool = False

    def __post_init__(self) -> None:
        if self.head_weight_item_bytes is None:
            default = FP32_BYTES if self.implementation in {LIGHTNING_V2_ABI, REGION_ABI} else BF16_BYTES
            object.__setattr__(self, "head_weight_item_bytes", default)

    def handoff_bytes(self, rows: int, heads: int, queries: int) -> int:
        if self.gm_key_staging:
            rows = self.tile_size
        columns = self.handoff_columns(rows) if self.handoff_columns else align(rows, self.score_align)
        m = align(queries, self.handoff_query_alignment) * heads
        destination_bytes = ceil_div(columns * m * self.handoff_item_bytes, self.aivs_per_aic)
        return self.aivs_per_aic * align(destination_bytes, self.handoff_alignment)


def owner_pairs(ctx: OperatorPlanningContext, aivs_per_aic: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    owners = ctx.stream_owners
    aics = (owners.aic or owners.pool_aic) if owners else ()
    aivs = (owners.aiv or owners.pool_aiv) if owners else ()
    active_aic = frozenset((aics or range(ctx.datapath.aic_count))[:ctx.effective_aic_cores])
    active_aiv = frozenset((aivs or range(ctx.datapath.aiv_count))[:ctx.effective_aiv_cores])
    return tuple((aic, siblings[:aivs_per_aic]) for aic, paired in ctx.datapath.aic_aiv_pairs
                 if aic in active_aic
                 and len(siblings := tuple(aiv for aiv in paired if aiv in active_aiv)) >= aivs_per_aic)


@dataclass(frozen=True, slots=True)
class _SparseIndexerProgramBuilder:
    contract: SparseIndexerContract
    lowering: Lowering
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    l2_reuse: bool
    aic_read: str
    aiv_read: str
    aic_count: int
    aiv_count: int
    gm_offsets: tuple[int, int] | None = None
    score_stride_override: int | None = None

    @property
    def _msa_score_stride(self) -> int:
        return self.score_stride_override if self.score_stride_override is not None else self.contract.msa_score_stride

    @property
    def _effective_flush_period(self) -> int:
        c = self.contract
        g = self.lowering
        if g.score_path == "l0c->gm->ub" and g.tile_selection is not None:
            act_s1 = ceil_div(c.query_tokens, c.batch_size)
            small = act_s1 <= ARCH22_SMALL_QUERY_MAX and c.topk_items <= ARCH22_TOPK_LIMIT and not c.is_sparse_count_over_2k
            return ARCH22_FLUSH_BLOCKS if small else 1
        return g.flush_period

    @property
    def partition(self) -> S2Partition:
        return S2Partition(self.contract.scanned_rows, self.lowering.tile_size, self.contract.s2_splits)

    @property
    def query_blocks(self) -> S1Partition:
        return S1Partition(self.contract.query_tokens, self.contract.batch_size, self.lowering.query_block_size)

    def owner_entry(self) -> tuple[PhaseWork, ...]:
        c = self.contract
        scales = c.index_heads * c.aiv_scale_bytes
        query = c.index_heads * c.cache_row_bytes - scales
        weights = c.index_heads * self.lowering.head_weight_item_bytes if c.has_head_weights else 0
        stages = [PhaseWork("query_load", Work(resources={"hbm": query, self.aic_read: query}), aic=True)]
        if scales:
            stages.append(PhaseWork("query_scale_load", Work(resources={"hbm": scales, self.aiv_read: scales})))
        if weights and self.lowering.score_path != "l0c->gm->ub":
            stages.append(PhaseWork("head_weight_load", Work(resources={"hbm": weights, self.aiv_read: weights}),
                                    metadata={"head_weight_bytes": weights}))
            if scales:
                stages.append(vector("prescale_head_weights", c.index_heads))
        return tuple(stages)

    def tile(self, rows: int, index: int, *, reuse: bool, final_owner: bool,
             queries: int, block_score_items: int | None = None) -> tuple[PhaseWork, ...]:
        c = self.contract
        g = self.lowering
        valid = min(g.tile_size, rows - index * g.tile_size)
        physical_rows = g.tile_size if g.gm_key_staging else valid
        heads = queries * c.index_heads
        scores = valid * heads
        score_bytes = g.handoff_bytes(valid, c.index_heads, queries)
        operand_bytes = c.cache_row_bytes - c.aiv_scale_bytes
        memory = "l2" if reuse else "hbm"
        score_to_gm = g.score_path == "l0c->gm->ub"
        if g.gm_key_staging:
            # The DSL gather repeats the last valid block to fill the tail tile.
            key = physical_rows * c.cache_row_bytes
            stages = [
                PhaseWork("key_gather", Work(resources={memory: key, self.aiv_read: key})),
                PhaseWork("key_stage_store", Work(resources={"hbm": key, "mte3": key}),
                          metadata={"aiv_subblocks": g.aivs_per_aic}),
                PhaseWork("key_stage_load", Work(resources={"hbm": key, self.aic_read: key}), True),
            ]
        else:
            key = valid * operand_bytes
            stages = [PhaseWork("key_load", Work(resources={memory: key, self.aic_read: key}), True)]
        if (scales := valid * c.aiv_scale_bytes):
            stages.append(PhaseWork("key_scale_load", Work(resources={memory: scales, self.aiv_read: scales})))
        if g.cube_head_reduce_fp16:
            dequant_l1_bytes = valid * heads * BF16_BYTES
            weight_ws_bytes = queries * c.index_heads * ARCH22_QLI_HEAD_REDUCE_M * BF16_BYTES
            reduced_score_bytes = queries * valid * FP32_BYTES
            weights = heads * g.head_weight_item_bytes if c.has_head_weights else 0
            stages.extend((
                PhaseWork("load_operands", Work(resources={"mte1": key + heads * operand_bytes}), True),
                PhaseWork("cube_scores", Work(matrix={c.matrix_dtype: 2 * physical_rows * heads * c.index_dim}), True),
                PhaseWork("score_dequant_l1", Work(resources={"fixpipe": dequant_l1_bytes}), True),
                PhaseWork("head_reduce_fp16", Work(
                    matrix={"fp16": 2 * queries * ARCH22_QLI_HEAD_REDUCE_M * c.index_heads * valid},
                    resources={"mte1": dequant_l1_bytes + weight_ws_bytes},
                ), True),
                PhaseWork("score_handoff", Work(resources={
                    "fixpipe": reduced_score_bytes, "hbm": reduced_score_bytes,
                }), True, {"route": g.score_path, "score_bytes": reduced_score_bytes}),
                PhaseWork("score_read", Work(resources={
                    "hbm": reduced_score_bytes + weights, self.aiv_read: reduced_score_bytes + weights,
                }), metadata={"head_weight_bytes": weights}),
                vector("key_descale", queries * valid),
            ))
            if g.tile_selection is not None:
                stages.extend(g.tile_selection(c, rows, index, valid, g.tile_size))
            return tuple(stages)
        stages.extend((
            PhaseWork("load_operands", Work(resources={"mte1": key + heads * operand_bytes}), True),
            PhaseWork("cube_scores", Work(matrix={c.matrix_dtype: 2 * physical_rows * heads * c.index_dim}), True),
            PhaseWork("score_handoff", Work(resources={
                "fixpipe": score_bytes, **({"hbm": score_bytes} if score_to_gm else {}),
            }), True, {"route": g.score_path, "score_bytes": score_bytes}),
        ))
        if score_to_gm:
            weights = heads * g.head_weight_item_bytes if c.has_head_weights else 0
            stages.append(PhaseWork("score_read", Work(resources={
                "hbm": score_bytes + weights, self.aiv_read: score_bytes + weights,
            }), metadata={"head_weight_bytes": weights}))
        msa = c.abi in MSA_ABIS
        score_heads = c.msa_score_heads if msa else 1
        if not msa:
            stages.append(vector("rectification", scores))
            if c.has_head_weights:
                stages.append(vector("weighting", scores))
            elif scales:
                stages.append(vector("query_descale", scores))
            stages.append(vector("head_reduction", scores - queries * valid))
            if scales:
                stages.append(vector("key_descale", queries * valid))
        elif score_heads < c.index_heads:
            stages.append(vector("head_maxima", scores - valid))
        candidates = ceil_div(valid, c.block_size)
        if c.block_size > 1:
            stages.append(vector("block_maxima", queries * (valid - candidates) * score_heads))
        if c.has_mask3_sort:
            stages.append(vector("mask3", queries * candidates))
        if msa:
            final_tile = final_owner and (index + 1) * g.tile_size >= rows
            total_items = c.score_items if block_score_items is None else block_score_items
            columns = candidates + (self._msa_score_stride - total_items if final_tile else 0)
            stages.append(self._msa_score_store(candidates, columns))
        elif g.gm_score_row or g.owner_score_row or not c.fused_topk:
            score_row = queries * candidates * (g.score_row_item_bytes if c.fused_topk else FP32_BYTES)
            stages.append(PhaseWork("score_row_store", Work(resources={"hbm": score_row, "mte3": score_row})))
        elif g.tile_selection is not None:
            stages.extend(g.tile_selection(c, rows, index, valid, g.tile_size))
        return tuple(stages)

    def _msa_score_store(self, valid: int, columns: int) -> PhaseWork:
        c = self.contract
        stride = self._msa_score_stride
        size = c.msa_score_heads * columns * FP32_BYTES
        return PhaseWork("score_row_store", Work(resources={"hbm": size, "mte3": size}),
                         metadata={"heads": c.msa_score_heads, "valid_items_per_head": valid,
                                   "score_stride": stride})

    def owner_exit(self, rows: int, partition: S2Partition | None = None) -> tuple[PhaseWork, ...]:
        c = self.contract
        g = self.lowering
        if not c.fused_topk or g.gm_score_row:
            return ()
        part = self.partition if partition is None else partition
        items = ceil_div(rows, c.block_size)
        stages = []
        if g.owner_score_row and (items >= c.topk_items or part.owners > 1):
            size = items * g.score_row_item_bytes
            stages.append(PhaseWork("local_score_row_load", Work(resources={"hbm": size, self.aiv_read: size})))
        if g.owner_selection is not None:
            stages.extend(g.owner_selection(c, items))
        if part.owners > 1:
            count, pairs, records = self._split_storage(rows)
            stages.append(PhaseWork("split_list_store", Work(resources={
                "hbm": pairs + records, "mte3": pairs + records,
            }), metadata={"pairs": count, "pair_bytes": pairs, "record_bytes": records}))
        return tuple(stages)

    def _split_storage(self, rows: int) -> tuple[int, int, int]:
        c = self.contract
        g = self.lowering
        if g.score_path == "l0c->gm->ub":
            count = c.arch22_retained_items
            return count, count * PAIR_BYTES, ARCH22_SPLIT_RECORD_BYTES
        if g.owner_score_row:
            count = align(c.topk_items, QLI_SPLIT_LIST_ALIGNMENT)
            return count, count * (g.score_row_item_bytes + INT32_BYTES), 0
        count = min(c.topk_items, ceil_div(rows, c.block_size))
        return count, count * PAIR_BYTES, 0

    def _split_reduction(self, partition: S2Partition | None = None) -> tuple[PhaseWork, ...]:
        part = self.partition if partition is None else partition
        pairs = pair_bytes = record_bytes = 0
        for rows, owners, _ in part.variants():
            count, payload, record = self._split_storage(rows)
            pairs += owners * count
            pair_bytes += owners * payload
            record_bytes += owners * record
        size = pair_bytes + record_bytes
        stages = [PhaseWork("split_list_load", Work(resources={"hbm": size, self.aiv_read: size}),
                            metadata={"pair_bytes": pair_bytes, "record_bytes": record_bytes})]
        merges = part.owners - 1
        if self.lowering.score_path != "l0c->gm->ub":
            return (*stages, vector("split_topk_merge", pairs, merges=merges))
        groups, tail = divmod(merges, MERGE_FAN_IN - 1)
        entries = groups * MERGE_FAN_IN + (tail + 1 if tail else 0)
        retained = self.contract.arch22_retained_items
        passes = groups + bool(tail)
        return (*stages,
                vector("split_topk_merge", entries * retained, merges=merges, mrg_passes=passes,
                       input_pair_bytes=entries * retained * PAIR_BYTES),
                vector("split_topk_copyback", passes * retained, vector_barriers=2 * passes,
                       pair_bytes=passes * retained * PAIR_BYTES))

    def query_exit(
        self, block_rows: int | None = None, partition: S2Partition | None = None,
    ) -> tuple[PhaseWork, ...]:
        c = self.contract
        g = self.lowering
        part = self.partition if partition is None else partition
        score_items = c.score_items if block_rows is None else ceil_div(block_rows, c.block_size)
        msa = c.abi in MSA_ABIS
        stages = []
        if msa and not score_items:
            stages.append(self._msa_score_store(0, self._msa_score_stride))
        if not c.fused_topk:
            return tuple(stages)
        score_heads = c.msa_score_heads if msa else 1
        needs_gm_select = score_items > c.topk_items if c.abi == REGION_ABI else bool(score_items)
        if g.gm_score_row and needs_gm_select:
            size = align(score_items, g.score_row_read_align) * score_heads * FP32_BYTES
            stages.append(PhaseWork("score_row_load", Work(resources={"hbm": size, self.aiv_read: size})))
            if g.owner_selection is not None:
                stages.extend(g.owner_selection(c, score_items))
        elif not g.gm_score_row and part.owners > 1:
            stages.extend(self._split_reduction(part))
        if c.has_mask3_sort and (c.abi != REGION_ABI or score_items > c.topk_items):
            stages.append(vector("index_sort", sort_ops(c.topk_items)))
        output_items = score_heads * c.topk_items
        valid_items = min(score_items, c.topk_items)
        size = output_items * INT32_BYTES
        stages.extend((
            vector("index_extract", output_items, valid_items_per_head=valid_items,
                   pair_bytes=score_heads * valid_items * (g.score_row_item_bytes + INT32_BYTES)),
            PhaseWork("index_store", Work(resources={"hbm": size, "mte3": size})),
        ))
        if c.abi == QSA_ABI:
            # Packing reads the live group IDs and writes the fixed compact row,
            # including the original-token tail and any sentinel slots.
            group_bytes = valid_items * INT32_BYTES
            packed_bytes = c.packed_index_items * INT32_BYTES
            stages.append(PhaseWork("index_pack", Work(resources={
                "vector": c.packed_index_items, "hbm": group_bytes + packed_bytes, "mte3": packed_bytes,
                **({self.aiv_read: group_bytes} if group_bytes else {}),
            }), metadata={"valid_groups": valid_items, "residual_tokens": c.residual_tokens}))
        if c.return_choice == MSA_RETURN_INDICES_VALUES:
            size = output_items * FP32_BYTES
            stages.append(PhaseWork("values_store", Work(resources={"hbm": size, "mte3": size})))
        return tuple(stages)

    @property
    def _has_varying_causal_rows(self) -> bool:
        return _has_varying_causal_rows(self.contract, self.lowering.query_block_size)

    def _block_geometry(self, block: int) -> tuple[int, int]:
        c = self.contract
        qb = self.lowering.query_block_size
        if not self._has_varying_causal_rows:
            return self.query_blocks.live_rows(block), c.scanned_rows
        rem = block
        for req_q, requests in self.query_blocks._request_groups():
            tiles = ceil_div(req_q, qb)
            group_blocks = tiles * requests
            if rem < group_blocks:
                local = rem % tiles
                start = local * qb
                live_q = min(qb, req_q - start)
                rows = _causal_block_rows(
                    req_q, start + live_q, c.scanned_rows, c.compress_ratio, c.residual_tokens,
                )
                return live_q, rows
            rem -= group_blocks
        raise IndexError("S1 block outside query partition")

    def _tile_variants(self, rows: int) -> Iterator[tuple[int, int]]:
        # Interior tiles differ only at the periodic flush. Keep the leading
        # period (first retained list) and the final tile as separate classes.
        period = self._effective_flush_period
        tiles = ceil_div(rows, self.lowering.tile_size)
        first = min(tiles - 1, period)
        for index in range(first):
            yield index, 1
        interior = tiles - 1 - first
        if interior > 0:
            flushes, ordinary = divmod(interior, period)
            if period > 1:
                ordinary += flushes * (period - 1)
                if ordinary:
                    yield first, ordinary
            if flushes:
                yield first + period - 1, flushes
        yield tiles - 1, 1

    def _add_block_totals(
        self, add, entry: dict[str, Work], tiles: dict[str, Work],
        owner_exit: dict[str, Work], query_exit: dict[str, Work],
        entry_phases: tuple[PhaseWork, ...], tile_cache: dict,
        block_rows: int, live_q: int, weight: int,
    ) -> None:
        if not weight:
            return
        c = self.contract
        g = self.lowering
        period = self._effective_flush_period
        msa = c.abi in MSA_ABIS
        block_items = ceil_div(block_rows, c.block_size)
        part = S2Partition(block_rows, g.tile_size, c.s2_splits)
        for rows, owners, final_owner in part.variants():
            add(entry, entry_phases, weight * live_q * owners)
            for index, count in self._tile_variants(rows):
                valid = min(g.tile_size, rows - index * g.tile_size)
                last_tile = (index + 1) * g.tile_size >= rows
                key = (
                    valid, last_tile, final_owner, live_q,
                    index == 0, index < period, index % period,
                    block_items if (msa and final_owner and last_tile) else 0,
                )
                cached = tile_cache.get(key)
                if cached is None:
                    cached = self.tile(
                        rows, index, reuse=False, final_owner=final_owner,
                        queries=live_q, block_score_items=block_items,
                    )
                    tile_cache[key] = cached
                add(tiles, cached, weight * owners * count)
            add(owner_exit, self.owner_exit(rows, part), weight * live_q * owners)
        add(query_exit, self.query_exit(block_rows, part), weight * live_q)

    def _add_affine_selection(
        self, add, target: dict[str, Work], seg_lo: int, seg_hi: int,
        items_a: int, qb: int, items_d: int, lo_x: int, hi_x: int, mult: int,
    ) -> None:
        if seg_hi <= seg_lo:
            return
        g = self.lowering
        c = self.contract
        n = seg_hi - seg_lo
        if lo_x == hi_x:
            add(target, g.owner_selection(c, lo_x), mult * n)
            return
        p0 = g.owner_selection(c, lo_x)
        p1 = g.owner_selection(c, lo_x + 1)
        s = _sum_floor_range(seg_lo, seg_hi, items_a, qb, items_d)
        delta_x = s - lo_x * n
        for ph0, ph1 in zip(p0, p1):
            res = {
                role: (ph0.work.resources[role] * n + (ph1.work.resources[role] - ph0.work.resources[role]) * delta_x) * mult
                for role in ph0.work.resources
            }
            target[ph0.name] = target.get(ph0.name, Work()) + Work(resources=res)

    def _add_ramp_selection(
        self, add, target: dict[str, Work], lo: int, hi: int,
        items_a: int, qb: int, items_d: int, mult: int,
    ) -> None:
        g = self.lowering
        c = self.contract
        if g.owner_selection is None or hi <= lo:
            return
        if c.abi in QUANT_ABIS:
            m_topk = _first_ge(lo, hi, items_a, qb, items_d, c.topk_items)
            if m_topk >= hi:
                return
            k_lo = ceil_div((items_a + m_topk * qb) // items_d, RADIX_BINS)
            k_hi = ceil_div((items_a + (hi - 1) * qb) // items_d, RADIX_BINS)
            for k in range(k_lo, k_hi + 1):
                b_lo = max(m_topk, _first_ge(m_topk, hi, items_a, qb, items_d, (k - 1) * RADIX_BINS + 1))
                b_hi = _first_ge(m_topk, hi, items_a, qb, items_d, k * RADIX_BINS + 1)
                if b_hi > b_lo:
                    sample = (items_a + b_lo * qb) // items_d
                    add(target, g.owner_selection(c, sample), mult * (b_hi - b_lo))
            return
        x_lo = (items_a + lo * qb) // items_d
        x_hi = (items_a + (hi - 1) * qb) // items_d
        if c.abi == REGION_ABI:
            self._add_affine_selection(add, target, lo, hi, items_a, qb, items_d, x_lo, x_hi, mult)
            return
        if c.abi in MSA_ABIS:
            edge = 1
            while True:
                seg_x0 = 1 if edge == 1 else (edge // 2) + 1
                if seg_x0 > x_hi:
                    break
                if edge >= x_lo:
                    a_i = _first_ge(lo, hi, items_a, qb, items_d, seg_x0)
                    b_i = _first_ge(lo, hi, items_a, qb, items_d, min(edge, x_hi) + 1)
                    if b_i > a_i:
                        self._add_affine_selection(
                            add, target, a_i, b_i, items_a, qb, items_d,
                            max(seg_x0, x_lo), min(edge, x_hi), mult,
                        )
                edge *= 2
            return
        if c.abi == DSL_QSLI_ABI:
            cuts = [x_lo, x_hi + 1]
            if x_lo < c.topk_items + 1 <= x_hi:
                cuts.append(c.topk_items + 1)
            k_start = max(1, x_lo // DSL_TRUNK_ITEMS)
            k_end = (x_hi - 1) // DSL_TRUNK_ITEMS
            for k in range(k_start, k_end + 1):
                cut = k * DSL_TRUNK_ITEMS + 1
                if x_lo < cut <= x_hi:
                    cuts.append(cut)
            cuts = sorted(set(cuts))
            for seg_x0, seg_x1 in zip(cuts, cuts[1:]):
                a_i = _first_ge(lo, hi, items_a, qb, items_d, seg_x0)
                b_i = _first_ge(lo, hi, items_a, qb, items_d, seg_x1)
                if b_i > a_i:
                    self._add_affine_selection(
                        add, target, a_i, b_i, items_a, qb, items_d, seg_x0, seg_x1 - 1, mult,
                    )

    def _add_fast_causal_ramp(
        self, add, entry: dict[str, Work], tiles: dict[str, Work],
        owner_exit: dict[str, Work], query_exit: dict[str, Work],
        entry_phases: tuple[PhaseWork, ...], lo: int, hi: int,
        base: int, qb: int, step: int, req_count: int,
    ) -> None:
        c = self.contract
        g = self.lowering
        ts, splits, blk = g.tile_size, c.s2_splits, c.block_size
        t_a, t_d = base + step * (ts - 1), step * ts
        mid_s = _first_ge(lo, hi, t_a, qb, t_d, splits)
        total_owners = _sum_floor_range(lo, mid_s, t_a, qb, t_d) + (hi - mid_s) * splits
        add(entry, entry_phases, req_count * qb * total_owners)
        # Non-final tiles (period == 1) and final tiles of non-final owners.
        n_idx0 = n_idx1 = nf_idx1 = 0
        nf_idx0 = _sum_floor_range(lo, mid_s, t_a - t_d, qb, t_d)
        for owner_i in range(splits):
            oa, od = t_a + (splits - 1 - owner_i) * t_d, splits * t_d
            m2 = _first_ge(mid_s, hi, oa, qb, od, 2)
            m3 = _first_ge(mid_s, hi, oa, qb, od, MIN_INTERIOR_TILES)
            n_idx0 += hi - m2
            n_idx1 += _sum_floor_range(m3, hi, oa - 2 * od, qb, od)
            if owner_i < splits - 1:
                nf_idx0 += m2 - mid_s
                nf_idx1 += hi - m2
        for count, rows_arg, idx_arg in (
            (n_idx0, 2 * ts, 0), (n_idx1, MIN_INTERIOR_TILES * ts, 1), (nf_idx0, ts, 0), (nf_idx1, 2 * ts, 1),
        ):
            if count:
                add(tiles, self.tile(rows_arg, idx_arg, reuse=False, final_owner=False, queries=qb, block_score_items=0), req_count * count)
        # Final tile of final owner grouped by tail valid count in 1..ts.
        mid_t2 = _first_ge(lo, hi, t_a, qb, t_d, 2 * splits)
        for seg_lo, seg_hi, idx_arg, row_base in ((lo, mid_t2, 0, 0), (mid_t2, hi, 1, ts)):
            if seg_hi <= seg_lo:
                continue
            for v in range(1, ts + 1):
                cnt = _count_floor_mod(seg_lo, seg_hi, base - step, qb, step, ts, v - 1)
                if cnt:
                    add(tiles, self.tile(row_base + v, idx_arg, reuse=False, final_owner=True, queries=qb, block_score_items=0), req_count * cnt)
        items_a, items_d = base + step * (blk - 1), step * blk
        if c.abi in MSA_ABIS:
            sum_items = _sum_floor_range(lo, hi, items_a, qb, items_d)
            delta = req_count * c.msa_score_heads * sum_items * FP32_BYTES
            prev = tiles["score_row_store"].resources
            tiles["score_row_store"] = Work(resources={"hbm": prev["hbm"] - delta, "mte3": prev["mte3"] - delta})
        self._add_fast_causal_exits(add, owner_exit, query_exit, lo, hi, t_a, t_d, items_a, items_d, qb, mid_s, req_count)

    def _add_fast_causal_exits(
        self, add, owner_exit: dict[str, Work], query_exit: dict[str, Work],
        lo: int, hi: int, t_a: int, t_d: int, items_a: int, items_d: int,
        qb: int, mid_s: int, req_count: int,
    ) -> None:
        c = self.contract
        g = self.lowering
        mult = req_count * qb
        n = hi - lo
        if not c.fused_topk:
            return
        if c.s2_splits > 1:
            ts, splits = g.tile_size, c.s2_splits
            mid_o2 = _first_ge(lo, hi, t_a, qb, t_d, 2)
            split_owners = _sum_floor_range(mid_o2, mid_s, t_a, qb, t_d) + (hi - mid_s) * splits
            add(owner_exit, self.owner_exit(ts, S2Partition(2 * ts, ts, 2)), mult * split_owners)
            for k in range(2, splits):
                a_k = _first_ge(lo, hi, t_a, qb, t_d, k)
                b_k = _first_ge(lo, hi, t_a, qb, t_d, k + 1)
                if b_k > a_k:
                    add(query_exit, self._split_reduction(S2Partition(k * ts, ts, splits)), mult * (b_k - a_k))
            if hi > mid_s:
                add(query_exit, self._split_reduction(S2Partition(splits * ts, ts, splits)), mult * (hi - mid_s))
        elif not g.gm_score_row:
            if g.owner_score_row:
                m_topk = _first_ge(lo, hi, items_a, qb, items_d, c.topk_items)
                sum_load = _sum_floor_range(m_topk, hi, items_a, qb, items_d)
                size = sum_load * g.score_row_item_bytes
                if size:
                    add(owner_exit, (PhaseWork("local_score_row_load", Work(resources={"hbm": size, self.aiv_read: size})),), mult)
            self._add_ramp_selection(add, owner_exit, lo, hi, items_a, qb, items_d, mult)
        else:
            sel_lo = _first_ge(lo, hi, items_a, qb, items_d, c.topk_items + 1) if c.abi == REGION_ABI else lo
            if hi > sel_lo:
                align_r = g.score_row_read_align
                score_heads = c.msa_score_heads if c.abi in MSA_ABIS else 1
                sum_aligned = _sum_floor_range(sel_lo, hi, items_a + (align_r - 1) * items_d, qb, align_r * items_d) * align_r
                size = sum_aligned * score_heads * FP32_BYTES
                if size:
                    add(query_exit, (PhaseWork("score_row_load", Work(resources={"hbm": size, self.aiv_read: size})),), mult)
                self._add_ramp_selection(add, query_exit, sel_lo, hi, items_a, qb, items_d, mult)
        if c.has_mask3_sort:
            sort_lo = _first_ge(lo, hi, items_a, qb, items_d, c.topk_items + 1) if c.abi == REGION_ABI else lo
            if hi > sort_lo:
                add(query_exit, (vector("index_sort", sort_ops(c.topk_items)),), mult * (hi - sort_lo))
        score_heads = c.msa_score_heads if c.abi in MSA_ABIS else 1
        output_items = score_heads * c.topk_items
        out_bytes = output_items * INT32_BYTES
        add(query_exit, (vector("index_extract", output_items), PhaseWork("index_store", Work(resources={"hbm": out_bytes, "mte3": out_bytes}))), mult * n)
        if c.abi == QSA_ABI:
            m_topk = _first_ge(lo, hi, items_a, qb, items_d, c.topk_items)
            sum_valid = _sum_floor_range(lo, m_topk, items_a, qb, items_d) + (hi - m_topk) * c.topk_items
            group_b = sum_valid * INT32_BYTES
            packed_b = n * c.packed_index_items * INT32_BYTES
            add(query_exit, (PhaseWork("index_pack", Work(resources={
                "vector": n * c.packed_index_items, "hbm": group_b + packed_b, "mte3": packed_b,
                **({self.aiv_read: group_b} if group_b else {}),
            })),), mult)
        if c.return_choice == MSA_RETURN_INDICES_VALUES:
            val_b = output_items * FP32_BYTES
            add(query_exit, (PhaseWork("values_store", Work(resources={"hbm": val_b, "mte3": val_b})),), mult * n)

    def phase_totals(self) -> dict[str, Work]:
        entry, tiles, owner_exit, query_exit = {}, {}, {}, {}

        def add(totals: dict[str, Work], phases: tuple[PhaseWork, ...], count: int) -> None:
            if count:
                for phase in phases:
                    totals[phase.name] = totals.get(phase.name, Work()) + phase.work.scaled(count)

        def merge_buckets() -> dict[str, Work]:
            merged: dict[str, Work] = {}
            for bucket in (entry, tiles, owner_exit, query_exit):
                for name, work in bucket.items():
                    merged[name] = merged.get(name, Work()) + work
            return merged

        c = self.contract
        if not self._has_varying_causal_rows:
            for rows, owners, final_owner in self.partition.variants():
                add(entry, self.owner_entry(), c.query_tokens * owners)
                for index, count in self._tile_variants(rows):
                    for queries, blocks, reuse in self.query_blocks.variants(self.l2_reuse):
                        add(tiles, self.tile(rows, index, reuse=reuse, final_owner=final_owner, queries=queries),
                            blocks * owners * count)
                add(owner_exit, self.owner_exit(rows), c.query_tokens * owners)
            add(query_exit, self.query_exit(), c.query_tokens)
            return merge_buckets()

        g = self.lowering
        qb = g.query_block_size
        step = c.compress_ratio
        total_orig = c.scanned_rows * step + c.residual_tokens
        entry_phases = self.owner_entry()
        tile_cache: dict = {}
        fast_eligible = (
            self._effective_flush_period == 1
            and g.tile_size % c.block_size == 0
            and (c.s2_splits == 1 or (g.owner_selection is None and g.score_path == "l0c->gm->ub"))
        )
        for req_q, req_count in self.query_blocks._request_groups():
            full_blocks, rem_q = divmod(req_q, qb)
            if full_blocks:
                base = total_orig - req_q + qb
                lo = _first_ge(0, full_blocks, base, qb, step, 1)
                hi = _first_ge(lo, full_blocks, base, qb, step, c.scanned_rows)
                if lo:
                    self._add_block_totals(add, entry, tiles, owner_exit, query_exit, entry_phases, tile_cache, 0, qb, req_count * lo)
                if hi > lo:
                    if fast_eligible and (hi - lo) > g.tile_size:
                        self._add_fast_causal_ramp(add, entry, tiles, owner_exit, query_exit, entry_phases, lo, hi, base, qb, step, req_count)
                    else:
                        for local in range(lo, hi):
                            rows = (base + local * qb) // step
                            self._add_block_totals(add, entry, tiles, owner_exit, query_exit, entry_phases, tile_cache, rows, qb, req_count)
                if full_blocks > hi:
                    self._add_block_totals(add, entry, tiles, owner_exit, query_exit, entry_phases, tile_cache, c.scanned_rows, qb, req_count * (full_blocks - hi))
            if rem_q:
                tail_rows = _causal_block_rows(req_q, req_q, c.scanned_rows, step, c.residual_tokens)
                self._add_block_totals(add, entry, tiles, owner_exit, query_exit, entry_phases, tile_cache, tail_rows, rem_q, req_count)
        return merge_buckets()

    def _placement(self, ctx: RecipeContext, owner, phase: PhaseWork):
        aic, aivs = owner
        if phase.aic:
            return ctx.owned(aic, self.aic_count, (*phase.work.resources, "cube"))
        placement = {}
        for aiv in aivs:
            for name, units in ctx.owned(aiv, self.aiv_count, tuple(phase.work.resources)).items():
                placement[name] = tuple(dict.fromkeys((*placement.get(name, ()), *units)))
        return placement

    def _stages(self, ctx: RecipeContext, owner, phases: tuple[PhaseWork, ...],
                queries: int = 1) -> tuple[Stage, ...]:
        stages = []
        for phase in phases:
            work = phase.work.scaled(queries) if queries > 1 else phase.work
            stages.append(Stage(
                phase.name, ctx.bind(work),
                after=(stages[-1].name,) if stages else (),
                placement=self._placement(ctx, owner, phase),
                metadata={**phase.metadata, "query_rows": queries} if queries > 1 else phase.metadata,
            ))
        return tuple(stages)

    def _tile_stages(self, ctx: RecipeContext, owner, rows: int, index: int,
                     reuse: bool, ring: RingBuffer | None, final_owner: bool, queries: int,
                     key_start: int | None = None,
                     block_score_items: int | None = None) -> tuple[Stage, ...]:
        phases = self.tile(
            rows, index, reuse=reuse, final_owner=final_owner,
            queries=queries, block_score_items=block_score_items,
        )
        stages = self._stages(ctx, owner, phases)
        if key_start is not None:
            first, *rest = stages
            stages = (replace(first, metadata={**first.metadata, "key_gm_row": key_start + index * self.lowering.tile_size}), *rest)
        if ring is None:
            return stages
        slot = ring.slot(index)
        release = f"score_release_{slot}"
        dual_dst_ctl = int(self.lowering.handoff_item_bytes == FP32_BYTES)
        metadata = {"score_slot": slot, "score_layout": "row_major",
                    "dual_dst_ctl": dual_dst_ctl, "dual_destination": bool(dual_dst_ctl)}
        bound = []
        alloc = ()
        for stage in stages:
            if stage.name == "cube_scores":
                stage = replace(stage, previous=("score_handoff",))
            elif stage.name == "score_handoff":
                # Credit is returned only after all paired AIVs finish reading
                # the score. The tile window and explicit per-slot edge apply
                # even when two tail tiles retire in a different order.
                slot_bytes = self.lowering.score_slot_bytes
                alloc = tuple(
                    ctx.allocate(f"{ring.allocation}_{slot}_aiv{aiv}", "ub", slot_bytes, (aiv,))
                    for aiv in owner[1]
                ) if "ub" in ctx.pool_names else ()
                stage = replace(stage, previous=(release,), acquire=alloc,
                                metadata={**stage.metadata, **metadata})
                bound.extend((
                    stage,
                    Stage("score_publish", after=(stage.name,), metadata={"score_slot": slot}),
                ))
                continue
            elif stage.name == "rectification":
                stage = replace(stage, after=("score_publish",))
            elif stage.name == "head_reduction":
                bound.append(stage)
                bound.append(Stage(release, after=(stage.name,), free=alloc,
                                   keep_empty=True, metadata={"score_slot": slot}))
                continue
            if bound and bound[-1].name == release:
                stage = replace(stage, after=(release,))
            bound.append(stage)
        return tuple(bound)

    def _owner_program(self, ctx: RecipeContext, owner, split: int, reuse: bool, queries: int, block: int,
                       partition: S2Partition | None = None, block_rows: int | None = None) -> KernelProgram:
        part = self.partition if partition is None else partition
        rows = part.owner_rows(split)
        g = self.lowering
        block_score_items = None if block_rows is None else ceil_div(block_rows, self.contract.block_size)
        ring = RingBuffer(depth=g.slots, allocation=SCORE_SLOT_NAME) if g.aivs_per_aic > 1 else None
        entry = self._stages(ctx, owner, self.owner_entry(), queries)
        key_start = None
        if self.gm_offsets is not None:
            q_offset, kv_offset = self.gm_offsets
            q_offset += block * g.query_block_size
            entry = tuple(replace(stage, metadata={**stage.metadata, "query_gm_row": q_offset}) for stage in entry)
            key_start = kv_offset // self.contract.compress_ratio + part.owner_start(split)
        tiles = PipelineProgram(
            ceil_div(rows, g.tile_size), ring.depth if ring is not None else g.slots,
            lambda index: self._tile_stages(ctx, owner, rows, index, reuse, ring,
                                            split == part.owners - 1, queries, key_start,
                                            block_score_items=block_score_items),
        )
        body = Stage("score_tiles", after=(entry[-1].name,), program=tiles)
        selection_owner = (owner[0], owner[1][:1]) if g.gm_key_staging else owner
        end = self._stages(ctx, selection_owner, self.owner_exit(rows, part), queries)
        if end:
            end = (replace(end[0], after=(body.name,)), *end[1:])
        return KernelProgram((*entry, body, *end))

    def _query_program(self, ctx: RecipeContext, owner_index: int, block: int) -> tuple[Stage, ...]:
        queries, block_rows = self._block_geometry(block)
        part = S2Partition(block_rows, self.lowering.tile_size, self.contract.s2_splits)
        split_count = part.owners
        concurrent = min(split_count, len(self.owners))
        reuse = self.l2_reuse and block > 0
        streams = []
        for lane, owner in enumerate(self.owners[owner_index:owner_index + concurrent]):
            streams.append(PipelineProgram(
                ceil_div(split_count - lane, concurrent), 1,
                lambda index, lane=lane, owner=owner: (
                    Stage("s2_owner", program=self._owner_program(
                        ctx, owner, index * concurrent + lane,
                        reuse, queries, block, partition=part, block_rows=block_rows)),
                ),
            ))
        aic, aivs = self.owners[owner_index]
        selection_owner = (aic, aivs[:1] if self.lowering.gm_key_staging else aivs)
        exit_stages = self._stages(ctx, selection_owner, self.query_exit(block_rows, part), queries)
        if not streams:
            return exit_stages
        body = Stage("s2_owners", program=ParallelProgram(tuple(streams)))
        if exit_stages:
            exit_stages = (replace(exit_stages[0], after=(body.name,)), *exit_stages[1:])
        return (body, *exit_stages)

    def __call__(self, ctx: RecipeContext) -> ParallelProgram | _DisjointOwnerProgram:
        blocks = self.query_blocks.count
        # Shared-prefix reuse is admitted only if the complete key stream
        # fits, with the first query block establishing residency before reuse.
        width = 1 if self.l2_reuse else max(1, len(self.owners) // max(1, self.partition.owners))
        count = min(blocks, width)
        placed = ParallelProgram(tuple(
            PipelineProgram(
                ceil_div(blocks - lane, count), 1,
                lambda index, lane=lane: self._query_program(
                    ctx, lane * max(1, self.partition.owners), index * count + lane),
            )
            for lane in range(count)
        ))
        # The generic stream binder interprets placement units as lane-local
        # ordinals. These stages already target the lane's physical core IDs.
        if ctx.stream_owners is not None and ctx.stream_owners.isolation == "disjoint":
            return _DisjointOwnerProgram(placed)
        return placed


def build_plan(
    op: OpSpec, ctx: OperatorPlanningContext, contract: SparseIndexerContract,
    lowering: Lowering, owners: tuple[tuple[int, tuple[int, ...]], ...],
    sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    if contract.abi in QUANT_ABIS and contract.cache_dtype == "bf16":
        raise unsupported("quant_abi_dtype", f"{contract.abi} requires a quantized cache_dtype, not 'bf16'")
    if contract.abi == LIGHTNING_V2_ABI and (
        contract.index_dim != DEFAULT_INDEX_DIM or contract.index_heads > LIGHTNING_V2_MAX_HEAD_GROUP
    ):
        raise unsupported(
            "lightning_v2_domain",
            f"{contract.abi} requires index_dim={DEFAULT_INDEX_DIM} and index_heads<={LIGHTNING_V2_MAX_HEAD_GROUP}",
        )
    if contract.abi == REGION_ABI and contract.index_dim != REGION_INDEX_DIM:
        raise ModelingError(f"sparse_indexer_region_requires_index_dim_{REGION_INDEX_DIM}:{contract.index_dim}")
    if not owners:
        raise unsupported("missing_owners", "Sparse index scoring needs reachable AIC and AIV owners")
    geometry = ctx.geometry
    hierarchy = getattr(geometry.physical_mechanisms, "memory_hierarchy", None)
    l2_capacity = getattr(hierarchy, "effective_l2_bytes", geometry.buffers.get("l2", 0))
    if (ctx.requests or contract.request_spans) and ctx.admission_memory_view is not None and "l2" in geometry.resource_units:
        l2_capacity = ctx.admission_memory_view.effective_l2_bytes(geometry.rank or 0)

    batch_msa_stride = contract.msa_score_stride if contract.abi in MSA_ABIS else None

    def bind(request, owners_, offsets=None):
        uniform_rows = not _has_varying_causal_rows(request, lowering.query_block_size)
        reuse = (uniform_rows and request.shared_prefix and request.query_tokens > 1 and request.scanned_rows > 0
                 and "l2" in geometry.resource_units
                 and request.scanned_rows * request.cache_row_bytes <= l2_capacity)
        return _SparseIndexerProgramBuilder(
            request, lowering, owners_, reuse, read_role(geometry.resource_units),
            read_role(geometry.resource_units, aiv=0), ctx.datapath.aic_count, ctx.datapath.aiv_count,
            offsets, batch_msa_stride,
        )

    use_per_request = bool(
        ctx.requests
        or (contract.request_spans and not _spans_match_uniform_contract(contract))
    )
    if use_per_request:
        req_spans = tuple(_request_contract_spans(contract, ctx))
        builders, waves = [], []
        start = wave_start = 0
        for request, offsets in req_spans:
            if len(req_spans) == 1:
                lanes = len(owners)
            else:
                lanes = min(len(owners), max(1, S2Partition(
                    request.scanned_rows, lowering.tile_size, request.s2_splits,
                ).owners))
            if start + lanes > len(owners):
                waves.append((wave_start, len(builders)))
                wave_start, start = len(builders), 0
            builders.append(bind(request, owners[start:start + lanes], offsets if ctx.requests else None))
            start += lanes
        if wave_start < len(builders):
            waves.append((wave_start, len(builders)))

        program = partial(request_program, tuple(builders), tuple(waves))
    else:
        program = bind(contract, owners)
        builders = (program,)

    totals = {}
    for builder in builders:
        for name, work in builder.phase_totals().items():
            totals[name] = totals.get(name, Work()) + work
    phases = []
    total = Work()
    for name, work in totals.items():
        total += work
        items = tuple(WorkItem("cube", "flop", n, dtype) for dtype, n in work.matrix.items() if n)
        items += tuple(WorkItem(role, "op" if role == "vector" else "bytes", n)
                       for role, n in work.resources.items() if n)
        phases.append(PipelinePhase(name, (phases[-1].phase_id,) if phases else (), items))
    return OperatorPlan(
        work=total, program=program,
        metadata={
            "implementation": lowering.implementation, "abi": contract.abi,
            "arch_id": ctx.compute_arch.arch_id, "tiling": lowering.tiling,
            "s2_tile_size": lowering.tile_size, "score_path": lowering.score_path,
            "fused_topk": contract.fused_topk, "cache_row_bytes": contract.cache_row_bytes,
            "s2_splits": max((b.partition.owners for b in builders), default=0), "causal": contract.causal,
            "useful_flops": sum(b.contract.matrix_flops for b in builders),
            **sequence_layout_metadata(ctx, "TND"),
            **({"compress_ratio": contract.compress_ratio, "residual_tokens": contract.residual_tokens,
                "packed_index_items": contract.packed_index_items} if contract.abi == QSA_ABI else {}),
            **({"score_stride": contract.msa_score_stride,
                "score_workspace_bytes": sum(b.contract.msa_score_heads * b.contract.query_tokens
                * b._msa_score_stride * FP32_BYTES for b in builders)} if contract.abi in MSA_ABIS
                and contract.fused_topk else {}),
        },
        pipeline_spec=FusedPipelineSpec(FAMILY, lowering.implementation, tuple(phases), sources),
    )


def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(
        ctx.occurrence, spec.family, spec.implementation, (op.op_id,), (),
        spec.phases, spec.sources, spec.repeats,
    )
