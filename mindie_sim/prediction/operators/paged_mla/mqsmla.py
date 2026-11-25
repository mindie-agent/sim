"""Mixed-quant sparse MLA with resident dequantized KV and paired AIV owners.

[Role] Price the retained whole-row, scalar-address source branch.
[Invariants] One query's original and compressed lists share online state.
  A padded tile repeats valid KV reads; QK and PV both execute fixed 128-row
  tiles. UB/L1 exchanges never become invented HBM workspaces.
[Timeouts & Locks] O(active query rows) metadata; two live phase tiles per owner.
[Side-Effect Policy] Pure plans and the common causal Engine, no observations.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.primitives.loader import resolve_dispatch_envelope
from mindie_sim.schema.abi import AbiQueryKey, AbiSupportVerdict, OperandPortSig
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.prediction.instructions import PrimitiveInventory
from .common import DSL_MQS_ABI

TARGET_BACKEND: Literal["ascend"] = "ascend"
CUBE_FRACTAL_ROWS = 16
MIN_MQS_PAGE_ROWS = 16
MQS_TILE_KEYS = 128
MQS_GM_WORKSPACE_SLOTS = 3
OFFSET_ENTRY_BYTES = 4
ARCH22_PRESET = "npu_arch_2201"

SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/op_api/aclnn_mixed_quant_sparse_flash_mla.h",
        source_sha256="4b613b72ec6cb36209328b084b74aa76e32ec86ad872976e564d9e4eea60e671",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_block_vector.h",
        source_sha256="f60e4f44294b0677522bdeec766f6bb66f61fdc3375ac95f547000eff7c10237",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_block_cube.h",
        source_sha256="df3f7ca12e1bad28f2f52d4bdfe5d43326a176af88d00c7d1121d4cc4e62290e",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/README_en.md",
        source_sha256="e5f013aca264fd192f5780455eeb4fa9d378a03d9576e34bb31488cdf39e75c0",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py",
        source_sha256="9fd0ad943a677e630ffb13df255b73a7b8889de0ed5a032996c2a3be6fc5ee36",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_metadata.py",
        source_sha256="51e13bd02d35704f99298dd9fdff36ddd82c941a8106c07d8e668d29664ad40a",
    ),
)

#: CANN 9.3 official source (`aclnnMixedQuantSparseFlashMla`).
SOURCE = {
    "repository": "cann/ops-transformer",
    "revision": "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "base_revision": "nightly-ci-main-a5-openeuler-cann9.2.1-20261111",
    "source_state": "committed",
    "path": "attention/mixed_quant_sparse_flash_mla/op_api/aclnn_mixed_quant_sparse_flash_mla.h",
    "source_sha256": "4b613b72ec6cb36209328b084b74aa76e32ec86ad872976e564d9e4eea60e671",
    "symbol": "aclnnMixedQuantSparseFlashMla",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
DSL_SOURCE = {
    "repository": "tilelang-ascend/kernels",
    "revision": "5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
    "source_state": "unmerged_pr",
    "path": "samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py",
    "source_sha256": "9fd0ad943a677e630ffb13df255b73a7b8889de0ed5a032996c2a3be6fc5ee36",
    "symbol": DSL_MQS_ABI,
    "scope": "tilelang_dsl_544_320; unmerged_pr_requires_manifest_binding",
}
RECIPE = "aclnn_mixed_quant_sparse_flash_mla"
NATIVE_KIND = "aclnnMixedQuantSparseFlashMla"
DSL_NATIVE_KIND = DSL_MQS_ABI
SUPPORTED_PHASES = frozenset({"decode", "draft"})
SUPPORTED_ADDRESSING = frozenset({"scalar"})
_MQS_CANN_MOTIF = "mixed_quant_sparse_flash_mla"
_MQS_DSL_MOTIF = "mixed_quant_sparse_flash_mla_dsl"
_MQS_CANN_PACKED_GROUP = 64
_MQS_DSL_FP8_GROUP = 32
_MQS_DSL_FP4_GROUP = 16
_MQS_DEFAULT_PAGE_SIZE = 128
_MQS_KV_HEADS = 1
_MQS_METADATA_ELEMS = 2048
_MQS_PAIRED_AIV_COUNT = 2
_AIV_ROLES = frozenset({"vector", "mte2_aiv", "mte3"})
_MQS_ADMITTED_STATUSES = frozenset({"admitted_shipping", "admitted_proposed_pr"})
_SOC_DEFAULT_SKU = {
    "A2": "ascend910b2",
    "A3": "ascend910_9392",
    "A5": "ascend950dt_9582",
}
_A3_SKU_MARKERS = ("9362", "9372", "9382", "9391", "9392")

# Source Channel depths. BF16 output aliases the FP32 accumulator. The P NZ
# buffers include one 16-element pad per N1 face; all scalar state is FP32.
BUFFERS = {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 262144,
           "ub": 2 * 32 * 128 * 4 + 2 * 32 * 512 * 4
                 + 2 * (32 * 128 + 8 * 16) * 2
                 + 4 * 16 * 544 + 2 * 32 * 17 * 16 * 2 + 10 * 32 * 4}

# The packaged source shards query heads by TP and replicates the single MQA
# KV head. ``local_heads = num_heads // tp`` therefore selects the QK cube
# N-face count; the values below are the declared TP tilings whose M-tile still
# aligns to the 16-row cube face.
DEFAULT_MQSMLA_HEADS = 64
MQSMLA_HEAD_WIDTH = 512
MQSMLA_ROPE_WIDTH = 64
CUBE_HEAD_TILE_ALIGN = 16
#: Source scalar address prepass: eight 16-row subtiles, one 512-element row
#: pair of GM reads, and four scalar-ALU issues per address row.
ADDRESS_SUBTILES_PER_TILE = 8
ADDRESS_ROWS_PER_SUBTILE = 16
ADDRESS_SCALAR_READS = 2
ADDRESS_SCALAR_ALU_ISSUES = 4
MIN_PADDED_LOCAL_HEADS = 8
HEAD_GROUP_START = 0
# Packed row widths: FP8 values plus one BF16 scale per 32 values, and two FP4
# values per byte plus one BF16 scale per 16 values.
ORIGINAL_CACHE_ROW_BYTES = 544
COMPRESSED_CACHE_ROW_BYTES = 320
PHYSICAL_KV_TILE_ROWS = 128
GATHER_SUBTILE_ROWS = 16
BF16_ELEMENT_BYTES = 2
FP32_ELEMENT_BYTES = 4
ORIGINAL_DEQUANT_OPS_PER_ELEM = 6
COMPRESSED_DEQUANT_OPS_PER_ELEM = 4
SOFTMAX_PAIR_OPS = 8
SOFTMAX_ROW_REDUCE_OPS = 26
ACCUM_NONFIRST_OPS = 3
# Declared AIC ownership bounds of the source wrapper.
MAX_MQSMLA_AIC_CORES = 36
#: ``aclnnMixedQuantSparseFlashMla`` packed-KV modes of the arch35 kernel.
KV_QUANT_BF16 = 0
KV_QUANT_FP8 = 1
KV_QUANT_MXFP8 = 2
KV_QUANT_MODES = frozenset({KV_QUANT_BF16, KV_QUANT_FP8, KV_QUANT_MXFP8})
#: Packed row widths of one 512-wide latent slot plus its 64 rotary lanes.
KV_QUANT_BF16_ROW_BYTES = 512 * BF16_ELEMENT_BYTES
KV_QUANT_FP8_ROW_BYTES = 576 + 16 * BF16_ELEMENT_BYTES
KV_QUANT_MXFP8_ROW_BYTES = 576 + 8
KV_QUANT_ROW_BYTES = {
    KV_QUANT_BF16: KV_QUANT_BF16_ROW_BYTES,
    KV_QUANT_FP8: KV_QUANT_FP8_ROW_BYTES,
    KV_QUANT_MXFP8: KV_QUANT_MXFP8_ROW_BYTES,
}
#: Declared vector pass inventory of the dequantized operand publish.
#: ``qm=1`` keeps the source's FP8 cast + per-32 scale multiply + NZ move.
#: ``qm=2`` adds the UE8M0 group unpack, exponent shift and broadcast multiply.
KV_QUANT_DEQUANT_OPS = {
    KV_QUANT_BF16: 0,
    KV_QUANT_FP8: 5,
    KV_QUANT_MXFP8: 10,
}
#: Live KV/P/alpha slots of the double-buffered inter-tile pipeline.
QUANT_TILE_SLOTS = 2
# Distinct (branch, rows, first, last, addressing, heads) phase payloads.
TILE_PHASE_CACHE_SIZE = 512
OWNER_STAGE_CACHE_SIZE = 16
_ARCH35_PRESETS = frozenset({"npu_arch_3510", "DAV_3510"})


def supported_heads(heads):
    """Keep the padded eight-head shard and complete 16-head cube faces."""
    return (heads == MIN_PADDED_LOCAL_HEADS
            or heads >= CUBE_HEAD_TILE_ALIGN and heads % CUBE_HEAD_TILE_ALIGN == HEAD_GROUP_START)


def _phase_key(stream, head_counts, index):
    return (*stream.key(index), head_counts[bisect_right(stream.ends, index)])


def query_rows(shape):
    """Global query order preserves each sequence's causal context."""
    window, ratio, topk = (shape[key] for key in ("window", "ratio", "topk"))
    for query, context in shape["sequences"]:
        for offset in range(query):
            position = context + offset + 1
            original = min(position, window) if window else 0
            compressed = position // ratio if ratio else 0
            yield original, min(compressed, topk) if topk else compressed


def owner_ranges(rows, cores):
    """Exact metadata SUPPORT_FD=False contiguous q/r row ownership."""
    q, r = divmod(rows, cores)
    start = 0
    result = []
    for core in range(cores):
        count = q + (core < r)
        if count:
            result.append((core, start, start + count))
        start += count
    return tuple(result)


@lru_cache(maxsize=TILE_PHASE_CACHE_SIZE)
def tile_phases(branch, valid, first, last, addressing="scalar",
                heads=DEFAULT_MQSMLA_HEADS):
    """Source phase payloads, parameterized by the local query-head count."""
    sinks_bytes = heads * FP32_ELEMENT_BYTES
    row_io_bytes = heads * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
    if not valid:
        return (
            ("sinks", Work(resources={"hbm": sinks_bytes, "mte2_aiv": sinks_bytes})),
            ("empty", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH})),
            ("output", Work(resources={"hbm": row_io_bytes, "mte3": row_io_bytes})),
        )
    physical = PHYSICAL_KV_TILE_ROWS
    row_bytes = (ORIGINAL_CACHE_ROW_BYTES if branch == "original"
                 else COMPRESSED_CACHE_ROW_BYTES)
    # The QK/PV M face keeps at least one aligned head face; wider TP shards
    # round their local heads up to the same 16-row granularity.
    cube_heads = max(
        CUBE_HEAD_TILE_ALIGN,
        ((heads + CUBE_HEAD_TILE_ALIGN - 1) // CUBE_HEAD_TILE_ALIGN) * CUBE_HEAD_TILE_ALIGN,
    )
    # Four 16-row subtiles per AIV; duplicate tail indices still issue copies.
    q = row_io_bytes if first else 0
    output = row_io_bytes if last else 0
    phases = [
        ("query", Work(resources={"hbm": q, "mte2_aic": q})),
        ("sinks", Work(resources={"hbm": sinks_bytes if first else 0,
                                  "mte2_aiv": sinks_bytes if first else 0})),
    ]
    subtile = GATHER_SUBTILE_ROWS
    for part in range(physical // subtile):
        sub_bytes = subtile * FP32_ELEMENT_BYTES
        row_bytes_part = subtile * row_bytes
        phases.extend((
            (f"addr_index_{part}", Work(resources={"hbm": sub_bytes})),
            (f"addr_page_{part}", Work(resources={"hbm": sub_bytes})),
            (f"gather_{part}", Work(resources={"hbm": row_bytes_part,
                                               "mte2_aiv": row_bytes_part})),
        ))
    phases.append(("addr_join", Work()))
    dequant_ops = (ORIGINAL_DEQUANT_OPS_PER_ELEM if branch == "original"
                   else COMPRESSED_DEQUANT_OPS_PER_ELEM)
    phases.extend((
        ("dequant", Work(resources={"vector": physical * MQSMLA_HEAD_WIDTH * dequant_ops})),
        ("kv_l1", Work(resources={"mte3": physical * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES})),
        ("qk_operands", Work(resources={
            "mte1": (cube_heads + physical) * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES})),
        ("qk", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * physical * MQSMLA_HEAD_WIDTH})),
        ("scores", Work(resources={"fixpipe": cube_heads * physical * FP32_ELEMENT_BYTES})),
        ("softmax", Work(resources={
            "vector": SOFTMAX_PAIR_OPS * heads * valid + SOFTMAX_ROW_REDUCE_OPS * heads})),
        ("probability_l1", Work(resources={
            "mte3": cube_heads * physical * BF16_ELEMENT_BYTES})),
        ("pv_operands", Work(resources={
            "mte1": (cube_heads * physical + physical * MQSMLA_HEAD_WIDTH)
                    * BF16_ELEMENT_BYTES})),
        ("pv", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * physical * MQSMLA_HEAD_WIDTH})),
        ("values", Work(resources={
            "fixpipe": cube_heads * MQSMLA_HEAD_WIDTH * FP32_ELEMENT_BYTES})),
        ("accumulate", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH * (
            (1 if first else ACCUM_NONFIRST_OPS) + (BF16_ELEMENT_BYTES if last else 0))})),
        ("output", Work(resources={"hbm": output, "mte3": output})),
    ))
    return tuple(phases)


@dataclass(frozen=True)
class OwnerTiles:
    """Index finite source tiles using O(query rows), never O(key tiles)."""
    rows: tuple
    ends: tuple

    @classmethod
    def create(cls, rows):
        ends, total = [], 0
        for original, compressed in rows:
            total += max(1, (original + (MQS_TILE_KEYS - 1)) // MQS_TILE_KEYS + (compressed + (MQS_TILE_KEYS - 1)) // MQS_TILE_KEYS)
            ends.append(total)
        return cls(tuple(rows), tuple(ends))

    def key(self, index):
        row = bisect_right(self.ends, index)
        local = index - (self.ends[row - 1] if row else 0)
        original, compressed = self.rows[row]
        nt = (original + (MQS_TILE_KEYS - 1)) // MQS_TILE_KEYS
        count = self.ends[row] - (self.ends[row - 1] if row else 0)
        if local < nt:
            return "original", min(MQS_TILE_KEYS, original - local * MQS_TILE_KEYS), local == 0, local + 1 == count
        return "compressed", max(0, min(MQS_TILE_KEYS, compressed - (local - nt) * MQS_TILE_KEYS)), local == 0, local + 1 == count

    @property
    def count(self):
        return self.ends[-1]


def quant_row_bytes(quant_mode):
    if quant_mode not in KV_QUANT_MODES:
        raise ModelingError("mqsmla_kv_quant_mode_unsupported")
    return {KV_QUANT_BF16: KV_QUANT_BF16_ROW_BYTES,
            KV_QUANT_FP8: KV_QUANT_FP8_ROW_BYTES,
            KV_QUANT_MXFP8: KV_QUANT_MXFP8_ROW_BYTES}[quant_mode]


def is_abi_admitted(verdict: AbiSupportVerdict) -> bool:
    """Return True iff the sealed ABI registry admitted the query."""
    return verdict.status in _MQS_ADMITTED_STATUSES


def _soc_arch_from_geometry(geometry, sku_override: str | None = None) -> tuple[str, str]:
    compute_arch = getattr(geometry, "compute_arch", None) or getattr(
        getattr(geometry, "datapath", None), "compute_arch", None
    )
    preset = str(getattr(compute_arch, "preset", "") or getattr(compute_arch, "arch_id", "") or "").strip()
    sku_id = str(
        sku_override or getattr(geometry, "sku_id", "")
        or getattr(getattr(geometry, "datapath", None), "sku_id", "") or ""
    ).strip().lower()
    arch_family = str(
        getattr(compute_arch, "arch_family", "") or getattr(geometry, "arch_family", "") or ""
    ).strip().upper()
    mem_hier = getattr(getattr(geometry, "physical_mechanisms", None), "memory_hierarchy", None)
    prov = getattr(mem_hier, "provenance", None)
    prov_arch = str(prov.get("arch") or "").strip().upper() if isinstance(prov, dict) else ""
    if arch_family in _SOC_DEFAULT_SKU:
        soc = arch_family
    elif prov_arch in _SOC_DEFAULT_SKU:
        soc = prov_arch
    elif prov_arch in _ARCH35_PRESETS or preset in _ARCH35_PRESETS:
        soc = "A5"
    elif any(marker in sku_id for marker in _A3_SKU_MARKERS):
        soc = "A3"
    elif preset == ARCH22_PRESET:
        soc = "A2"
    else:
        soc = "A5"
    return soc, (sku_id or _SOC_DEFAULT_SKU[soc])


def _has_dsl_cache_contract(meta: dict) -> bool:
    view = meta.get("attention_tensor_view")
    if not isinstance(view, dict):
        return False
    regions = view.get("cache_representations")
    if not isinstance(regions, dict):
        return False
    win = regions.get("window")
    cmp_kv = regions.get("compressed_kv")
    if not (isinstance(win, dict) and isinstance(cmp_kv, dict)):
        return False
    return (
        win.get("dtype") == "fp8_e4m3"
        and win.get("group_size") == _MQS_DSL_FP8_GROUP
        and win.get("scale_dtype") == "bf16"
        and cmp_kv.get("dtype") == "fp4_e2m1"
        and cmp_kv.get("group_size") == _MQS_DSL_FP4_GROUP
        and cmp_kv.get("scale_dtype") == "bf16"
    )


def _resolve_owner_pairs(geometry) -> tuple[tuple[int, tuple[int, int]], ...]:
    """Resolve active physical (aic, (aiv0, aiv1)) pairs honoring ``stream_owners``."""
    total_cores = int(geometry.compute_units)
    datapath = getattr(geometry, "datapath", None)
    raw_pairs = getattr(datapath, "aic_aiv_pairs", None) if datapath is not None else None
    all_pairs = (
        tuple((int(aic), tuple(int(aiv) for aiv in siblings)) for aic, siblings in raw_pairs)
        if raw_pairs
        else tuple((aic, (2 * aic, 2 * aic + 1)) for aic in range(total_cores))
    )
    owners = getattr(geometry, "stream_owners", None)
    if owners is None:
        return tuple((aic, (aivs[0], aivs[1])) for aic, aivs in all_pairs[:total_cores]
                     if len(aivs) == _MQS_PAIRED_AIV_COUNT)
    aic_list = tuple(owners.aic or owners.pool_aic or ())
    aiv_list = tuple(owners.aiv or owners.pool_aiv or ())
    aic_limit = (
        int(owners.aic_limit) if getattr(owners, "aic_limit", None) is not None
        else (len(aic_list) if aic_list else total_cores)
    )
    aiv_limit = (
        int(owners.aiv_limit) if getattr(owners, "aiv_limit", None) is not None
        else (len(aiv_list) if aiv_list else int(geometry.resource_units.get("vector", 2 * total_cores)))
    )
    active_aic = tuple((aic_list or tuple(range(total_cores)))[:aic_limit])
    active_aiv_set = frozenset((aiv_list or tuple(range(2 * total_cores)))[:aiv_limit])
    by_aic = {aic: tuple(aiv for aiv in siblings if aiv in active_aiv_set) for aic, siblings in all_pairs}
    pairs = tuple(
        (int(aic), (matched[0], matched[1]))
        for aic in active_aic
        if len(matched := by_aic.get(int(aic), ())) == _MQS_PAIRED_AIV_COUNT
    )
    if not pairs:
        raise ModelingError("mqsmla_requires_paired_aiv:stream_owners")
    return pairs


def _pair_placement(context, aic: int, aivs: tuple[int, int], roles) -> dict[str, tuple[int, ...]]:
    placed: dict[str, tuple[int, ...]] = {}
    for role in roles:
        if role in ("hbm", "l2"):
            continue
        phys = context.physical_role(role)
        if phys in context.resource_names:
            placed[context.resource_names[phys]] = aivs if role in _AIV_ROLES else (aic,)
    return placed


def _abi_port(
    role: str,
    dtype: str,
    layout: str,
    shape: tuple[int, ...],
    quant_mode: str = "none",
    group_shape: tuple[int, ...] = (),
    scale_dtype: str = "none",
) -> OperandPortSig:
    return OperandPortSig(
        role=role,
        dtype=dtype,
        layout=layout,
        transposed=False,
        shape=shape,
        quant_mode=quant_mode,
        group_shape=group_shape,
        scale_dtype=scale_dtype,
        scale_shape=(),
    )


def _mqsmla_kv_ports(
    quant_mode: int | None,
    pages: int,
    page_size: int,
) -> tuple[str, str, str, OperandPortSig, OperandPortSig]:
    if quant_mode is None:
        ori_kv = _abi_port(
            "ori_kv", "fp8_e4m3fn", "PA_BBND",
            (pages, page_size, _MQS_KV_HEADS, ORIGINAL_CACHE_ROW_BYTES),
            "per_group_fp8", (_MQS_DSL_FP8_GROUP,), "bf16",
        )
        cmp_kv = _abi_port(
            "cmp_kv", "fp4_e2m1", "PA_BBND",
            (pages, page_size, _MQS_KV_HEADS, COMPRESSED_CACHE_ROW_BYTES),
            "per_group_fp4", (_MQS_DSL_FP4_GROUP,), "bf16",
        )
        return DSL_NATIVE_KIND, _MQS_DSL_MOTIF, "fp8_e4m3fn", ori_kv, cmp_kv
    if quant_mode == KV_QUANT_FP8:
        ori_kv = _abi_port(
            "ori_kv", "fp8_e4m3fn", "PA_BBND",
            (pages, page_size, _MQS_KV_HEADS, KV_QUANT_FP8_ROW_BYTES),
            "packed_kv_608", (_MQS_CANN_PACKED_GROUP,), "bf16",
        )
        cmp_kv = _abi_port(
            "cmp_kv", "fp8_e4m3fn", "PA_BBND",
            (pages, page_size, _MQS_KV_HEADS, KV_QUANT_FP8_ROW_BYTES),
            "packed_kv_608", (_MQS_CANN_PACKED_GROUP,), "bf16",
        )
        return NATIVE_KIND, _MQS_CANN_MOTIF, "fp8_e4m3fn", ori_kv, cmp_kv
    if quant_mode == KV_QUANT_MXFP8:
        ori_kv = _abi_port(
            "ori_kv", "fp8_e4m3fn", "PA_BBND",
            (pages, page_size, _MQS_KV_HEADS, KV_QUANT_MXFP8_ROW_BYTES),
            "packed_kv_584", (_MQS_CANN_PACKED_GROUP,), "fp8_e8m0",
        )
        cmp_kv = _abi_port(
            "cmp_kv", "fp8_e4m3fn", "PA_BBND",
            (pages, page_size, _MQS_KV_HEADS, KV_QUANT_MXFP8_ROW_BYTES),
            "packed_kv_584", (_MQS_CANN_PACKED_GROUP,), "fp8_e8m0",
        )
        return NATIVE_KIND, _MQS_CANN_MOTIF, "fp8_e4m3fn", ori_kv, cmp_kv
    ori_kv = _abi_port(
        "ori_kv", "bf16", "PA_BBND",
        (pages, page_size, _MQS_KV_HEADS, KV_QUANT_BF16_ROW_BYTES),
        "dense", (), "none",
    )
    cmp_kv = _abi_port(
        "cmp_kv", "bf16", "PA_BBND",
        (pages, page_size, _MQS_KV_HEADS, KV_QUANT_BF16_ROW_BYTES),
        "dense", (), "none",
    )
    return NATIVE_KIND, _MQS_CANN_MOTIF, "bf16", ori_kv, cmp_kv


def build_mqsmla_abi_query(
    *,
    soc_arch: str = "A5",
    sku_id: str | None = None,
    quant_mode: int | None,
    active_q: int,
    heads: int,
    batch: int,
    compressed_cache: bool,
    topk_present: bool,
    topk: int = 0,
    page_size: int = _MQS_DEFAULT_PAGE_SIZE,
    pages: int = 1,
    return_softmax_lse: bool = False,
    pr_track: str = "shipping",
    model_scope: str = "target",
) -> AbiQueryKey:
    """Build the exact 1:1 positional ``AbiQueryKey`` for CANN or TileLang DSL MQSMLA."""
    eff_pages = max(1, int(pages))
    eff_page_size = max(1, int(page_size))
    eff_batch = max(1, int(batch))
    eff_q = max(1, int(active_q))
    eff_heads = max(1, int(heads))
    abi_id, motif, kv_dtype, ori_kv, cmp_kv = _mqsmla_kv_ports(
        quant_mode, eff_pages, eff_page_size,
    )
    q_port = _abi_port("q", "bf16", "ND", (eff_q, eff_heads, MQSMLA_HEAD_WIDTH), "dense")
    cu_seqlens_q_port = _abi_port("cu_seqlens_q", "int32", "ND", (eff_batch + 1,))
    sinks_port = _abi_port("sinks", "fp32", "ND", (eff_heads,), "dense")
    meta_port = _abi_port("metadata", "int32", "ND", (_MQS_METADATA_ELEMS,))
    if compressed_cache:
        operands = [q_port, ori_kv, cmp_kv]
        if topk_present:
            operands.append(_abi_port(
                "cmp_sparse_indices", "int32", "ND", (eff_q, _MQS_KV_HEADS, max(1, int(topk))),
            ))
        operands.extend((
            _abi_port("ori_block_table", "int32", "ND", (eff_batch, eff_pages)),
            _abi_port("cmp_block_table", "int32", "ND", (eff_batch, eff_pages)),
            cu_seqlens_q_port,
            _abi_port("seqused_ori_kv", "int32", "ND", (eff_batch,)),
            _abi_port("seqused_cmp_kv", "int32", "ND", (eff_batch,)),
            sinks_port,
            meta_port,
        ))
    else:
        operands = [
            q_port,
            ori_kv,
            _abi_port("ori_block_table", "int32", "ND", (eff_batch, eff_pages)),
            cu_seqlens_q_port,
            _abi_port("seqused_ori_kv", "int32", "ND", (eff_batch,)),
            sinks_port,
            meta_port,
        ]
    outputs = [_abi_port("attn_out", "bf16", "ND", (eff_q, eff_heads, MQSMLA_HEAD_WIDTH), "dense")]
    if return_softmax_lse:
        outputs.append(_abi_port("softmax_lse", "fp32", "ND", (eff_q, eff_heads), "dense"))
    arch_norm = str(soc_arch).strip().upper()
    return AbiQueryKey(
        soc_arch=arch_norm,
        sku_id=sku_id or _SOC_DEFAULT_SKU.get(arch_norm, "ascend950dt_9582"),
        abi_id=abi_id,
        graph_motif=motif,
        model_scope=model_scope,
        draft_quant_mode="none",
        pr_track=pr_track,
        operands=tuple(operands),
        outputs=tuple(outputs),
        rank_placement="local",
        world_size=1,
        comm_group_id="none",
        comm_domain_width=1,
        comm_dtype="none",
        reduction_kind="none",
        activation_kind="none",
        activation_params=(),
        has_bias=False,
        has_active_mask=False,
        dispatch_quant_mode=-1,
        shared_experts_per_rank=0,
        top_k=max(0, int(topk)),
        moe_expert_num=0,
        qk_rope_head_dim=MQSMLA_ROPE_WIDTH,
        mla_variant="latent_mla",
        sfa_dcp_active=False,
        pcp_size=1,
        kv_cache_dtype=kv_dtype,
        has_kpool_indexer=False,
        dcp_size=1,
        is_act_and_mul=False,
        enabled_stack_passes=frozenset(),
    )


def verify_mqsmla_abi(
    *,
    soc_arch: str = "A5",
    sku_id: str | None = None,
    quant_mode: int | None,
    active_q: int,
    heads: int,
    batch: int,
    compressed_cache: bool,
    topk_present: bool,
    topk: int = 0,
    page_size: int = _MQS_DEFAULT_PAGE_SIZE,
    pages: int = 1,
    return_softmax_lse: bool = False,
    pr_track: str = "shipping",
    model_scope: str = "target",
    track_spec=None,
    manifest_bindings: tuple = (),
    registry=None,
) -> AbiSupportVerdict:
    """Verify an MQSMLA invocation against the sealed per-SoC ``ProbedAbiRegistry``."""
    query = build_mqsmla_abi_query(
        soc_arch=soc_arch,
        sku_id=sku_id,
        quant_mode=quant_mode,
        active_q=active_q,
        heads=heads,
        batch=batch,
        compressed_cache=compressed_cache,
        topk_present=topk_present,
        topk=topk,
        page_size=page_size,
        pages=pages,
        return_softmax_lse=return_softmax_lse,
        pr_track=pr_track,
        model_scope=model_scope,
    )
    reg = registry if registry is not None else default_abi_registry()
    return reg.verify(query, track_spec=track_spec, manifest_bindings=tuple(manifest_bindings))


def quant_tile_phases(valid, first, last, *, heads, quant_mode, window_bytes=0,
                      ws_hbm=False, ws_l2=False):
    """One 128-key tile of the arch35 double-buffered inter-tile pipeline.

    Address/gather/dequant/publish of tile ``i+1`` overlap the AIC ``kv_refill ->
    QK -> Fixpipe -> online softmax -> PV`` chain of tile ``i``; CANN 950
    ``aclnnMixedQuantSparseFlashMla`` publishes dequantized BF16 KV from AIV UB
    through a 3-slot GM workspace (``mte3``) and refills AIC L1 via ``mte2_aic``.
    """
    row_bytes = quant_row_bytes(quant_mode)
    dequant_ops = KV_QUANT_DEQUANT_OPS[quant_mode]
    sinks_bytes = heads * FP32_ELEMENT_BYTES
    cube_heads = max(CUBE_HEAD_TILE_ALIGN,
                     -(-heads // CUBE_HEAD_TILE_ALIGN) * CUBE_HEAD_TILE_ALIGN)
    if not valid:
        return (
            ("sinks", Work(resources={"hbm": sinks_bytes if first else 0,
                                      "mte2_aiv": sinks_bytes if first else 0}), ()),
            ("empty", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH}), ()),
        )
    publish_bytes = valid * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
    # mixed_quant_sparse_flash_mla_csa_block_cube.h:203 aligns refill DMA rows up to 16.
    refill_rows = ((valid + CUBE_FRACTAL_ROWS - 1) // CUBE_FRACTAL_ROWS) * CUBE_FRACTAL_ROWS
    refill_bytes = refill_rows * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
    publish_res = {"mte3": publish_bytes}
    refill_res = {"mte2_aic": refill_bytes}
    if ws_hbm:
        publish_res["hbm"] = publish_bytes
        refill_res["hbm"] = refill_bytes
    elif ws_l2:
        publish_res["l2"] = publish_bytes
        refill_res["l2"] = refill_bytes
    return (
        ("sinks", Work(resources={"hbm": sinks_bytes if first else 0,
                                  "mte2_aiv": sinks_bytes if first else 0}), ()),
        ("address", Work(resources={
            "mte2_aiv": valid * ADDRESS_SCALAR_READS * FP32_ELEMENT_BYTES}), ()),
        ("gather", Work(resources={
            "mte2_aiv": valid * row_bytes + window_bytes}), ("address",)),
        ("dequant", Work(resources={
            "vector": valid * MQSMLA_HEAD_WIDTH * dequant_ops}), ("gather",)),
        ("publish", Work(resources=publish_res), ("dequant",)),
        ("kv_refill", Work(resources=refill_res), ("publish",)),
        ("qk", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * valid * MQSMLA_HEAD_WIDTH},
            resources={"mte1": (cube_heads + valid) * MQSMLA_HEAD_WIDTH
                              * BF16_ELEMENT_BYTES,
                       "fixpipe": cube_heads * valid * FP32_ELEMENT_BYTES}),
         ("kv_refill",)),
        ("softmax", Work(resources={
            "vector": SOFTMAX_PAIR_OPS * heads * valid
                      + SOFTMAX_ROW_REDUCE_OPS * heads}), ("qk", "sinks")),
        ("pv", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * valid * MQSMLA_HEAD_WIDTH},
            resources={"mte1": (cube_heads + MQSMLA_HEAD_WIDTH) * valid
                              * BF16_ELEMENT_BYTES,
                       "fixpipe": cube_heads * MQSMLA_HEAD_WIDTH * FP32_ELEMENT_BYTES}),
         ("softmax",)),
        ("accumulate", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH * (
            (1 if first else ACCUM_NONFIRST_OPS)
            + (BF16_ELEMENT_BYTES if last else 0))}), ("pv",)),
    )


@lru_cache(maxsize=TILE_PHASE_CACHE_SIZE)
def quant_tile_phases_cached(valid, first, last, heads, quant_mode, window_bytes=0,
                             ws_hbm=False, ws_l2=False):
    """Bounded cache of the pure quant-mode tile payloads."""
    return quant_tile_phases(valid, first, last, heads=heads,
                             quant_mode=quant_mode, window_bytes=window_bytes,
                             ws_hbm=ws_hbm, ws_l2=ws_l2)


def _mqsmla_stage_after(name: str, previous: str | None) -> tuple[str, ...]:
    if name in {"query", "sinks"}:
        return ()
    if name.startswith("addr_index_"):
        return ("query", "sinks")
    if name.startswith("addr_page_"):
        return ("addr_index_" + name.rsplit("_", 1)[-1],)
    if name.startswith("gather_"):
        return ("addr_page_" + name.rsplit("_", 1)[-1],)
    if name == "addr_join":
        return tuple(f"gather_{part}" for part in range(PHYSICAL_KV_TILE_ROWS // GATHER_SUBTILE_ROWS))
    if name == "qk_operands":
        return ("kv_l1", "query")
    if name == "softmax":
        return ("scores", "sinks")
    return (previous,) if previous else ()


def _mqsmla_owner_program(context, aic, aivs, stream, head_counts, *, addressing: str):
    @lru_cache(maxsize=OWNER_STAGE_CACHE_SIZE)
    def cached_stages(key):
        result = []
        previous = None
        for name, work in tile_phases(*key[:4], addressing=addressing, heads=key[-1]):
            after = _mqsmla_stage_after(name, previous)
            prior = {
                "query": ("output",) if key[2] else (),
                "sinks": ("output",) if key[2] else (),
                "addr_join": ("output",) if key[2] else (),
                "qk_operands": ("qk",), "qk": ("scores",),
                "pv_operands": ("pv",), "pv": ("values",),
                "accumulate": ("output",) if key[2] else ("accumulate",),
            }.get(name, ())
            roles = set(work.resources) - {"hbm"} | ({"cube"} if work.matrix else set())
            result.append(Stage(
                name, context.bind(work), after=after, previous=prior,
                placement=_pair_placement(context, aic, aivs, roles)))
            previous = name
        return tuple(result)

    def stages(index):
        return cached_stages(_phase_key(stream, head_counts, index))

    allocations = tuple(context.allocate(
        f"mqsmla{aic}_{pool}", pool, amount,
        aivs if pool == "ub" else (aic,))
        for pool, amount in BUFFERS.items())
    return KernelProgram((
        Stage("reserve", acquire=allocations),
        # Two complete phase tiles are a conservative source subset:
        # P and alpha have two slots. The third source KV slot can
        # prefetch more aggressively, which this compact DAG omits.
        Stage("tiles", after=("reserve",), program=PipelineProgram(stream.count, 2, stages)),
        Stage("release", after=("tiles",), free=allocations),
    ))


def build(op, geometry):
    meta = dict(op.metadata)
    shape = meta["mqsmla_geometry"]
    requested = str(shape.get("addressing", "scalar"))
    heads = int(shape["heads"])
    if (not supported_heads(heads) or shape["width"] != MQSMLA_HEAD_WIDTH
            or shape["metadata_support_fd"]):
        raise ModelingError("mqsmla_source_geometry_or_branch_unsupported")
    if requested in SUPPORTED_ADDRESSING:
        addressing = requested
    else:
        raise ModelingError("mqsmla_addressing_mode_unsupported")
    rows = tuple(query_rows(shape))
    cores = geometry.compute_units
    if not rows or not 1 <= cores <= MAX_MQSMLA_AIC_CORES:
        raise ModelingError("mqsmla_requires_positive_rows_and_at_most36_aic")
    quant_mode = shape.get("kv_quant_mode")
    if quant_mode is None and not _has_dsl_cache_contract(meta):
        raise ModelingError("mqsmla_cache_representation_contract_required")
    compute_arch = getattr(geometry, "compute_arch", None) or getattr(
        getattr(geometry, "datapath", None), "compute_arch", None
    )
    preset = str(getattr(compute_arch, "preset", "") or getattr(compute_arch, "arch_id", "") or "").strip()
    if quant_mode is None or quant_mode != KV_QUANT_BF16:
        if preset == ARCH22_PRESET:
            raise ModelingError("mqsmla_fp8_fp4_unsupported_on_arch22")
        if preset not in _ARCH35_PRESETS:
            raise ModelingError(f"mqsmla_fp8_fp4_requires_arch35:{preset or 'unset'}")
    if quant_mode is not None and quant_mode not in KV_QUANT_MODES:
        raise ModelingError("mqsmla_kv_quant_mode_unsupported")
    for pool, amount in BUFFERS.items():
        units = geometry.resource_units.get("vector", cores) if pool == "ub" else cores
        if (geometry.buffers.get(pool, 0) < amount
                or geometry.buffer_units.get(pool, 0) != units):
            raise ModelingError("mqsmla_source_buffer_capacity_or_ownership:" + pool)
    for role in ("vector", "mte2_aiv", "mte3"):
        if geometry.resource_units.get(role) != 2 * cores:
            raise ModelingError("mqsmla_requires_paired_aiv:" + role)
    for role in ("mte1", "mte2_aic", "fixpipe"):
        if geometry.resource_units.get(role) != cores:
            raise ModelingError("mqsmla_requires_private_aic:" + role)
    owner_pairs = _resolve_owner_pairs(geometry)
    head_groups = tuple(min(DEFAULT_MQSMLA_HEADS, heads - start)
                        for start in range(HEAD_GROUP_START, heads, DEFAULT_MQSMLA_HEADS))
    row_heads = tuple(group for _ in rows for group in head_groups)
    rows = tuple(row for row in rows for _ in head_groups)
    ownership = owner_ranges(len(rows), len(owner_pairs))
    streams = tuple(
        (owner_pairs[slot][0], owner_pairs[slot][1], OwnerTiles.create(rows[start:end]), row_heads[start:end])
        for slot, start, end in ownership
    )
    physical_ownership = tuple((owner_pairs[slot][0], start, end) for slot, start, end in ownership)
    soc_arch, sku_id = _soc_arch_from_geometry(geometry, sku_override=meta.get("abi_sku_id"))
    sequences = tuple(shape["sequences"])
    active_q = sum(int(q) for q, _ in sequences)
    abi_verdict = verify_mqsmla_abi(
        soc_arch=soc_arch,
        sku_id=sku_id,
        quant_mode=int(quant_mode) if quant_mode is not None else None,
        active_q=active_q,
        heads=heads,
        batch=len(sequences),
        compressed_cache=bool(shape.get("ratio")),
        topk_present=bool(shape.get("topk")),
        topk=int(shape.get("topk") or 0),
        pr_track=str(meta.get("abi_pr_track") or "shipping"),
        track_spec=meta.get("abi_track_spec"),
        manifest_bindings=tuple(meta.get("abi_manifest_bindings") or ()),
    )
    if not is_abi_admitted(abi_verdict):
        raise ModelingError(f"mqsmla_abi_not_admitted:{abi_verdict.status}:{abi_verdict.rejection_reason}")
    if quant_mode is not None:
        return _quant_plan(shape, streams, int(quant_mode), addressing, geometry, abi_verdict)
    total = Work()
    counts = Counter(_phase_key(stream, head_counts, index)
                     for _, _, stream, head_counts in streams for index in range(stream.count))
    for key, count in counts.items():
        for _, work in tile_phases(*key[:4], addressing=addressing, heads=key[-1]):
            total += work.scaled(count)

    def program(context):
        return ParallelProgram(tuple(
            _mqsmla_owner_program(context, aic, aivs, stream, head_counts,
                                  addressing=addressing)
            for aic, aivs, stream, head_counts in streams
        ))

    return OperatorPlan(total, program, {
        "source_execution_domain": "mixed",
        "source_program": DSL_SOURCE,
        "native_kind": DSL_NATIVE_KIND,
        "aclnn_symbol": DSL_NATIVE_KIND,
        "abi_registry_verified": is_abi_admitted(abi_verdict),
        "abi_verdict_status": abi_verdict.status,
        "abi_entry_id": abi_verdict.matched_entry_id,
        "abi_symbol": abi_verdict.symbol_or_kernel or DSL_NATIVE_KIND,
        "abi_rejection_reason": abi_verdict.rejection_reason,
        "mqsmla_owner_ranges": physical_ownership, "mqsmla_buffers_per_owner": BUFFERS,
        "mqsmla_schedule": "whole_query_rows;metadata_SUPPORT_FD_false;two_live_phase_tiles",
        "mqsmla_approximation": "coarse_source_phase_DAG;register_operations_not_instruction_latency",
        "mqsmla_addressing": addressing,
        "address_schedule": {
            "subtiles_per_tile": ADDRESS_SUBTILES_PER_TILE,
            "rows_per_subtile": ADDRESS_ROWS_PER_SUBTILE,
            "scalar_reads_bytes": ADDRESS_SCALAR_READS * MQSMLA_HEAD_WIDTH,
            "primitives": PrimitiveInventory({
                "scalar_gm_access": ADDRESS_SCALAR_READS * PHYSICAL_KV_TILE_ROWS,
                "scalar_alu_issue": (ADDRESS_SCALAR_ALU_ISSUES
                                     * PHYSICAL_KV_TILE_ROWS),
            }).to_dict(("hbm", "mte2_aiv", "vector", "mte3")),
        },
        "unmodeled_source_features": ["aicpu_metadata_program",
                                      "scalar_page_lookup_response",
                                      "vector_address_prepass",
                                      "flash_decode_partial_reduction",
                                      "third_KV_slot_prefetch_overlap",
                                      "cross_query_sinks_prefetch_overlap",
                                      "source_compiler_instruction_scheduling"],
    })


def quant_external_fill(shape, quant_mode):
    """One cold compressed selection per request and a private sliding window."""
    from mindie_sim.prediction.operators.attention.batch_prefix_attention import (
        nested_prefix_union, prefix_members, selected_prefix_count,
    )
    sequences = tuple(shape["sequences"])
    window, ratio, topk = (int(shape.get(k) or 0) for k in ("window", "ratio", "topk"))
    groups = {}
    private = 0
    for (query, context), (key, prefix) in zip(sequences, prefix_members(shape, sequences)):
        if not query:
            continue
        if window:
            private += min(context + query, window + query - 1)
        if not ratio:
            continue
        completed = (context + query) // ratio
        selected = min(topk, completed) if topk else completed
        extent = prefix // ratio
        shared = selected_prefix_count(selected, extent, completed, completed)
        private += selected - shared
        if shared:
            groups.setdefault(key, []).append((extent, shared))
    union = sum(nested_prefix_union(group) for group in groups.values())
    return (private + union) * quant_row_bytes(quant_mode), union


def _quant_plan(shape, streams, quant_mode, addressing, geometry, abi_verdict):
    """Arch35 ``aclnnMixedQuantSparseFlashMla``: overlapped 128-key tile DAG."""
    row_bytes = quant_row_bytes(quant_mode)
    ws_bytes = (
        MQS_GM_WORKSPACE_SLOTS * MQS_TILE_KEYS * MQSMLA_HEAD_WIDTH
        * BF16_ELEMENT_BYTES * len(streams)
    )
    l2_cap = int(geometry.buffers.get("l2", 0) or 0)
    ws_l2_resident = l2_cap > 0 and ws_bytes <= l2_cap
    ws_hbm = not ws_l2_resident
    ws_l2 = ws_l2_resident and "l2" in geometry.rates
    counts = Counter(_phase_key(stream, head_counts, index)
                     for _, _, stream, head_counts in streams
                     for index in range(stream.count))
    total = Work()
    for key, count in counts.items():
        for _name, work, _after in quant_tile_phases_cached(
                key[1], key[2], key[3], key[-1], quant_mode, 0,
                ws_hbm=ws_hbm, ws_l2=ws_l2):
            total += work.scaled(count)
    external, shared_rows = quant_external_fill(shape, quant_mode)
    fill = Work(resources={"hbm": external} if external else {})
    # Each head-group stream reads only its own query-head slice. Summing the
    # per-stream head counts charges the single query tensor once, not once
    # per head group.
    streamed = sum(sum(head_counts) for _aic, _aivs, _stream, head_counts in streams) \
        * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
    stream_work = Work(resources={"hbm": 2 * streamed, "mte2_aic": streamed,
                                  "mte3": streamed})

    def program(context):
        from .indexed_attention import bind_sustained_dram
        context = bind_sustained_dram(context)
        has_l2 = "l2" in context.resource_names

        def bind_work(work):
            if not has_l2 and "l2" in work.resources:
                work = Work(
                    matrix=work.matrix,
                    resources={k: v for k, v in work.resources.items() if k != "l2"},
                )
            return context.bind(work)

        def one(aic, aivs, stream, head_counts):
            @lru_cache(maxsize=OWNER_STAGE_CACHE_SIZE)
            def stages(key):
                result = []
                for name, work, after in quant_tile_phases_cached(
                        key[1], key[2], key[3], key[-1], quant_mode, 0,
                        ws_hbm=ws_hbm, ws_l2=ws_l2):
                    roles = set(work.resources) - {"hbm", "l2"}
                    if work.matrix:
                        roles |= {"cube"}
                    result.append(Stage(
                        name, bind_work(work), after=tuple(after),
                        previous=("accumulate",) if name == "accumulate" and not key[2] else (),
                        placement=_pair_placement(context, aic, aivs, roles)))
                return tuple(result)

            allocations = tuple(context.allocate(
                f"mqsmla{aic}_{pool}", pool, amount,
                aivs if pool == "ub" else (aic,))
                for pool, amount in BUFFERS.items())
            query_bytes = sum(head_counts) * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
            q_load = Stage("q_load", context.bind(Work(
                resources={"hbm": query_bytes, "mte2_aic": query_bytes})),
                after=("reserve",),
                placement=_pair_placement(context, aic, aivs, ("mte2_aic",)))
            tiles = Stage(
                "tiles", after=("q_load",),
                program=PipelineProgram(
                    stream.count, QUANT_TILE_SLOTS,
                    lambda index: stages(_phase_key(stream, head_counts, index))),
            )
            # Output MTE3 store strictly follows tile accumulation completion.
            output_store = Stage("output_store", context.bind(Work(
                resources={"hbm": query_bytes, "mte3": query_bytes})),
                after=("tiles",),
                placement=_pair_placement(context, aic, aivs, ("mte3",)))
            return KernelProgram((
                Stage("reserve", acquire=allocations),
                q_load,
                tiles,
                output_store,
                Stage("release", after=("output_store",), free=allocations)))

        owners = ParallelProgram(tuple(one(aic, aivs, stream, head_counts)
                                       for aic, aivs, stream, head_counts in streams))
        tbl = getattr(context, "primitive_table", None)
        dp = getattr(context, "datapath", None)
        env = resolve_dispatch_envelope(tbl, dp) if (tbl is not None and dp is not None) else None
        meta_launch_ms = float(env.aiv_min_block_floor_ms) if env is not None else 0.0
        stages_root = [
            Stage("prefix_fill", work=context.bind(fill)),
            Stage("owners", after=("prefix_fill",), program=owners),
        ]
        if meta_launch_ms > 0.0:
            stages_root = [
                Stage("metadata_launch", latency=meta_launch_ms),
                Stage("prefix_fill", after=("metadata_launch",), work=context.bind(fill)),
                Stage("owners", after=("prefix_fill",), program=owners),
            ]
        return KernelProgram(tuple(stages_root))

    return OperatorPlan(total + fill + stream_work, program, {
        "source_execution_domain": "mixed",
        "source_program": SOURCE,
        "native_kind": NATIVE_KIND,
        "aclnn_symbol": NATIVE_KIND,
        "abi_registry_verified": is_abi_admitted(abi_verdict),
        "abi_verdict_status": abi_verdict.status,
        "abi_entry_id": abi_verdict.matched_entry_id,
        "abi_symbol": abi_verdict.symbol_or_kernel or NATIVE_KIND,
        "abi_rejection_reason": abi_verdict.rejection_reason,
        "workspace_slots": MQS_GM_WORKSPACE_SLOTS,
        "workspace_bytes": ws_bytes,
        "workspace_l2_resident": ws_l2_resident,
        "mqsmla_owner_ranges": tuple((aic, stream.rows) for aic, _aivs, stream, _h in streams),
        "mqsmla_buffers_per_owner": BUFFERS,
        "mqsmla_schedule": "arch35_overlapped_tile_dag;gm_3slot_kv_workspace;two_live_kv_and_accumulator_slots",
        "mqsmla_approximation": "coarse_source_phase_DAG;register_operations_not_instruction_latency",
        "mqsmla_addressing": addressing,
        "mqsmla_kv_quant_mode": quant_mode,
        "mqsmla_packed_row_bytes": row_bytes,
        "mqsmla_dequant_ops_per_element": KV_QUANT_DEQUANT_OPS[quant_mode],
        "mqsmla_prefix_partition": {
            "external_kv_bytes": external,
            "shared_unique_rows": shared_rows,
            "prefix_cache_mode": shape.get("prefix_cache_mode", "private"),
        },
        "unmodeled_source_features": ["aicpu_metadata_program",
                                      "scalar_page_lookup_response",
                                      "flash_decode_partial_reduction",
                                      "source_compiler_instruction_scheduling"],
    })
