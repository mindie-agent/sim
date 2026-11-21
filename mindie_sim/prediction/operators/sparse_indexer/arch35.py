"""Arch35 sparse indexer strategies for QLI, QSA, MSA and radix selection.

Upstream-ABI: aclnnQuantLightningIndexerV2

[Role] Select the caller's tiler and score path, then lower the associated
  selection and score-result boundary.
[Invariants] QLI uses paired AIV score slots. Fused MSA selection consumes
  the FP32 score row written to GM. MXFP4 carries packed group-32 scales.
[Timeouts & Locks] Bounded route checks and tile classes; no I/O or locks.
[Side-Effect Policy] Constructs plans without loading accelerator libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan
from ..contracts import OperatorPlanningContext
from .common import (
    BF16_BYTES, DSL_QSLI_ABI, FP32_BYTES, INT32_BYTES, MSA_ABIS, PAIR_BYTES,
    QSA_ABI, QSA_OVERLAY_SOURCE, QUANT_ABIS, REGION_ABI, SCORE_SLOTS, VLLM_QLI_ABIS, Lowering, PhaseWork,
    SparseIndexerContract, align, build_plan, ceil_div, indexer_contract, request_contracts,
    owner_pairs, pair_merge, sort_ops, unsupported, vector,
)

ARCH_ID = ArchId("npu_arch_3510")
UPSTREAM_ABI = "aclnnQuantLightningIndexerV2"
QLI_TILE_ROWS = 128
QLI_FIXPIPE_BANK_BYTES = 256
QLI_S1_BASE_SIZE = 4
QLI_SMALL_S1_BASE_SIZE = 2
QLI_MAX_M = 256
QLI_SMALL_MAX_M = 128
QLI_SMALL_M_TOPK_THRESHOLD = 6144
FIXPIPE_ALIGNMENT = 32
DSL_TILE_ROWS = 512
DSL_TRUNK_ITEMS = 16384
RADIX_BINS = 256
RADIX_COUNTER_BYTES = 2
DSL_RADIX_PASSES = 2
REGION_RADIX_PASSES = 4
QLI_TRUNKS_BY_TOPK = ((2048, 16384), (3072, 12288), (4096, 8192), (5120, 4096), (6144, 2048))
QLI_LARGE_TOPK_TRUNK = 12288
_SCORE_ROUTE = "l0c->ub"
VLLM_ASCEND_REVISION = "fd7574116d21e21361acc0de0d2aca777e0bc564"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_cube_arch35.h",
        source_sha256="ff30ef9730f3955a3769666aec858c6e6d9c5f2830902a6475518cd1645b39d9",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/msa_index_score/op_kernel/arch35/msa_index_score_kernel_arch35.h",
        source_sha256="a8fbe9e6120129532418f0ab9498a3a2ab19f53e291f60bdc4f230736dba121e",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/msa_index_score/op_kernel/arch35/msa_index_score_topk_arch35.h",
        source_sha256="fe1908c59aa6616d523f8c07f67f79f78164c2e8048c0471f37a681669c7c531",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/msa_index_score/op_kernel/arch35/msa_mxfp4_kernel.h",
        source_sha256="140a54a09528a931e89512c4e2bc4c50b75c458c3b05735a256af8b875b7c28d",
    ),
    SourceRef(
        repository="cann/cannbot-dsl",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/quant_sparse_lightning_indexer/quant_sparse_lightning_indexer.py",
        source_sha256="a6572f3fd8455b6888f1ffb456ae764db2bbc99cf529f9d7cde6ba76c6927c73",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/lightning_indexer_v2/op_host/lightning_indexer_v2_tiling.cpp",
        source_sha256="b8988f5bdc1ccf8900c2a7853c4edf012f8cb99f6e9b5af40af56167bcd9aade",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision=VLLM_ASCEND_REVISION,
        relative_path="csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h",
        source_sha256="1f6e177b31fe2775c52cc402dbcf52772454f7b89cb9a254728b6289edf93d3e",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision=VLLM_ASCEND_REVISION,
        relative_path="csrc/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp",
        source_sha256="500ca594826c422885c61d10b70433bd079eed1df1538d1685364d788cd5044f",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision=VLLM_ASCEND_REVISION,
        relative_path="csrc/attention/indexer_region_score/op_kernel/indexer_region_score_select.h",
        source_sha256="ad9c20cdda5d98bc94dcd0f45295e31130ed0092d0a7506478aae1d70ed30f3b",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_vector_arch35.h",
        source_sha256="4a58ec4ef1c26b9660af56b2d7b711fa390a76893390b774505e4423c33c2924",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision=VLLM_ASCEND_REVISION,
        relative_path="csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_vector_arch35.h",
        source_sha256="9a17a799618d75e1d2196cb53daf3aade5b882c2fe08d9fe4e51038a3e1cdc7e",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision=VLLM_ASCEND_REVISION,
        relative_path="csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_cube_arch35.h",
        source_sha256="335ed4f409314a45581994abbb076cb7e7b36a0389c0dbc4c7a12dbe7f2534f0",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp",
        source_sha256="24197838cdf1b16bc1d73743c99e69c468fa7c25516c36ea9ccc80a8dcc85b19",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_kernel_arch35.h",
        source_sha256="37b78aab1ddb5eb0139e8a8c710610bbbccdc32359c4352ec302057153ed8945",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision=VLLM_ASCEND_REVISION,
        relative_path="csrc/attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_kernel_arch35.h",
        source_sha256="e0225ec301f2a540266552d56aee37fa7e8b73cc52e97509a267837c0b9053de",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision=VLLM_ASCEND_REVISION,
        relative_path="csrc/torch_binding.cpp",
        source_sha256="e0b96525763d58987858dbd099734f587b79981a639e24992941003b2ddd3b90",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/models/qwen4_exp/lightning_indexer.py",
        source_sha256="1d265521c2a44a904342e24a0d917aec575240469fe0f1dbc5ea61133d0e5736",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/msa_index_score/op_host/msa_index_score_tiling.cpp",
        source_sha256="0c7097f4391b2fe3abf9b7778ede388c234aef004c68ec62ca2f62361b42e625",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/msa_index_score/op_kernel/arch35/msa_seg_row_max_epilogue_arch35.h",
        source_sha256="e54b2e1bd508880c6c86fc2fdaab979f15d4747c9a662bd5a981de2e9983830e",
    ),
)
(
    QLI_SOURCE, MSA_SOURCE, MSA_TOPK_SOURCE, MXFP4_SOURCE, DSL_SOURCE, LIGHTNING_V2_TILER,
    VLLM_SOURCE, VLLM_TILER, REGION_SOURCE, QLI_VEC_SOURCE, VLLM_VEC_SOURCE, VLLM_CUBE_SOURCE,
    QLI_TILER, QLI_KERNEL, VLLM_QLI_KERNEL, VLLM_BINDING, QSA_SOURCE, MSA_TILER, MSA_EPILOGUE,
) = SOURCES


def _sort_tile(
    contract: SparseIndexerContract, rows: int, index: int, valid: int, tile_size: int,
) -> tuple[PhaseWork, ...]:
    items = ceil_div(valid, contract.block_size)
    stages = [vector("tile_sort", sort_ops(items), pair_bytes=items * PAIR_BYTES)]
    if index:
        retained = contract.topk_items
        stages.append(pair_merge("retained_topk_merge", items, retained, retained))
    return tuple(stages)


def _msa_select(contract: SparseIndexerContract, items: int) -> tuple[PhaseWork, ...]:
    heads = contract.msa_score_heads
    return (vector("score_row_topk", heads * sort_ops(items), heads=heads,
                   valid_items_per_head=items, pair_bytes=heads * items * PAIR_BYTES),)


def _region_select(contract: SparseIndexerContract, items: int) -> tuple[PhaseWork, ...]:
    return (vector("region_radix_topk", REGION_RADIX_PASSES * (items + RADIX_BINS),
                   histogram_bins=RADIX_BINS, pair_bytes=items * PAIR_BYTES),)


def _dsl_select(contract: SparseIndexerContract, items: int) -> tuple[PhaseWork, ...]:
    trunks = ceil_div(items, DSL_TRUNK_ITEMS)
    merges = max(0, trunks - 1)
    retained = min(items, contract.topk_items)
    candidates = items + merges * align(retained, RADIX_BINS)
    return (vector(
        "radix_histogram_topk", DSL_RADIX_PASSES * (candidates + trunks * RADIX_BINS),
        trunks=trunks, trunk_items=DSL_TRUNK_ITEMS, histogram_bins=RADIX_BINS,
        histogram_bytes=RADIX_BINS * RADIX_COUNTER_BYTES, counter_dtype="uint16",
        pair_bytes=items * PAIR_BYTES,
    ),)


def _qli_select(contract: SparseIndexerContract, items: int) -> tuple[PhaseWork, ...]:
    topk = contract.topk_items
    if items < topk:
        return ()
    trunk = next((size for limit, size in QLI_TRUNKS_BY_TOPK if topk <= limit), QLI_LARGE_TOPK_TRUNK)
    if contract.abi in VLLM_QLI_ABIS:
        # vLLM can balance a short final trunk against the preceding trunks.
        width = contract.scanned_rows
        loops = ceil_div(width, trunk)
        tail = width - (loops - 1) * trunk
        if loops > 1 and tail < trunk / 2:
            balanced = align(ceil_div(width, loops), RADIX_BINS)
            if RADIX_BINS <= balanced < trunk:
                trunk = balanced
    history = align(topk, RADIX_BINS)
    remaining = max(0, items - history) if topk > trunk else items
    passes = ceil_div(remaining, trunk)
    candidates = align(remaining, RADIX_BINS) + (
        passes if topk > trunk else max(0, passes - 1)
    ) * history
    return (vector("qli_radix_topk", DSL_RADIX_PASSES * (candidates + passes * RADIX_BINS),
                   trunk_items=trunk, trunks=passes, histogram_bins=RADIX_BINS,
                   pair_bytes=items * (RADIX_COUNTER_BYTES + INT32_BYTES)),)


def _qli_fixpipe_columns(rows: int) -> int:
    columns = align(rows, FIXPIPE_ALIGNMENT // FP32_BYTES)
    # Each ND block occupies one 256-byte bank in N. Above one bank, the
    # FP32/INT32 path always issues two 64-column ND blocks.
    return columns if columns * FP32_BYTES <= QLI_FIXPIPE_BANK_BYTES else QLI_TILE_ROWS


def _make_lowering(contract: SparseIndexerContract) -> tuple[Lowering, tuple[SourceRef, ...]]:
    abi = contract.abi
    tile = align(min(contract.s2_tile_size, QLI_TILE_ROWS), contract.block_size)
    kwargs = {"score_align": FIXPIPE_ALIGNMENT // FP32_BYTES}
    tiling = "s2_tiles"
    sources = (LIGHTNING_V2_TILER,)
    if abi in QUANT_ABIS:
        tile = QLI_TILE_ROWS
        # The vLLM caller binds its own exported operator and tiler. The
        # CANN and vLLM kernels currently use the same Arch35 S2 base size.
        if abi in VLLM_QLI_ABIS:
            tiling = "vllm_candidate_tiler"
            sources = (VLLM_SOURCE, VLLM_TILER, VLLM_CUBE_SOURCE,
                       VLLM_VEC_SOURCE, VLLM_QLI_KERNEL, VLLM_BINDING)
        else:
            tiling = "cann_qli_v2_tiler"
            sources = (QLI_SOURCE, QLI_VEC_SOURCE, QLI_TILER, QLI_KERNEL)
        handoff_bytes = BF16_BYTES if contract.cache_dtype == "mxfp4" else FP32_BYTES
        small_m = contract.topk_items > QLI_SMALL_M_TOPK_THRESHOLD
        max_m = QLI_SMALL_MAX_M if small_m else QLI_MAX_M
        slot_bytes = (max_m // 2) * tile * handoff_bytes
        kwargs.update(
            aivs_per_aic=2, slots=SCORE_SLOTS, owner_score_row=True,
            query_block_size=QLI_SMALL_S1_BASE_SIZE if small_m else QLI_S1_BASE_SIZE,
            handoff_query_alignment=2,
            score_row_item_bytes=RADIX_COUNTER_BYTES, owner_selection=_qli_select,
            handoff_item_bytes=handoff_bytes, score_align=FIXPIPE_ALIGNMENT // handoff_bytes,
            handoff_columns=_qli_fixpipe_columns if handoff_bytes == FP32_BYTES else None,
            score_slot_bytes=slot_bytes,
            head_weight_item_bytes=BF16_BYTES if contract.cache_dtype == "int8" else FP32_BYTES,
        )
    elif abi in MSA_ABIS:
        tiling = "msa_block_maxima"
        sources = (MSA_SOURCE, MSA_TILER, MSA_EPILOGUE)
        if contract.fused_topk:
            sources += (MSA_TOPK_SOURCE,)
        if contract.cache_dtype == "mxfp4":
            sources += (MXFP4_SOURCE,)
        kwargs.update(gm_score_row=True, owner_selection=_msa_select,
                      score_row_read_align=FIXPIPE_ALIGNMENT // FP32_BYTES)
    elif abi == DSL_QSLI_ABI:
        tiling = "dsl_radix_trunks"
        tile = DSL_TILE_ROWS
        sources = (DSL_SOURCE,)
        kwargs.update(
            gm_key_staging=True, aivs_per_aic=2, slots=SCORE_SLOTS,
            owner_selection=_dsl_select, head_weight_item_bytes=FP32_BYTES,
            handoff_item_bytes=BF16_BYTES, score_align=FIXPIPE_ALIGNMENT // BF16_BYTES,
            score_slot_bytes=tile // 2 * contract.index_heads * BF16_BYTES,
        )
    elif abi == QSA_ABI:
        sources = (QSA_SOURCE,) if contract.has_head_weights else (QSA_OVERLAY_SOURCE,)
        kwargs.update(head_weight_item_bytes=FP32_BYTES)
    elif abi == REGION_ABI:
        tiling = "region_score_mask3"
        sources = (REGION_SOURCE,)
        kwargs.update(gm_score_row=True, owner_selection=_region_select)
    if "owner_selection" not in kwargs:
        kwargs["tile_selection"] = _sort_tile
    return Lowering(abi, tiling, tile, _SCORE_ROUTE, **kwargs), sources


def _contract(op: OpSpec, ctx: OperatorPlanningContext) -> SparseIndexerContract:
    contract = indexer_contract(op, ctx, UPSTREAM_ABI)
    if contract.cache_dtype == "mxfp4" and contract.abi not in MSA_ABIS | QUANT_ABIS | {DSL_QSLI_ABI}:
        raise unsupported("mxfp4_abi", f"{contract.abi} has no MXFP4 lowering")
    return contract


@dataclass(frozen=True, slots=True)
class _Arch35Strategy:
    arch_id: ArchId = ARCH_ID
    abi: str = UPSTREAM_ABI
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        try:
            _contract(op, ctx)
        except ModelingError:
            return False
        return True

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        contract = _contract(op, ctx)
        lowering, sources = _make_lowering(contract)
        needed = lowering.aivs_per_aic
        representation = "fp32"
        if lowering.handoff_item_bytes == BF16_BYTES:
            representation = "bf16"
        elif contract.abi in QUANT_ABIS and contract.cache_dtype == "int8":
            representation = "int32"
        transfer = 0
        if ctx.requests:
            rectangles = tuple((c.query_tokens, c.scanned_rows) for c in request_contracts(contract, ctx))
        else:
            rectangles = ((ceil_div(contract.query_tokens, contract.batch_size), contract.scanned_rows),)
        if any(q and kv for q, kv in rectangles):
            route = ctx.datapath.route("l0c", "ub", representation)
            lowering = replace(lowering, handoff_alignment=route.path.alignment_bytes)
            transfer = max(lowering.handoff_bytes(min(kv, lowering.tile_size), contract.index_heads,
                                                  min(q, lowering.query_block_size))
                           for q, kv in rectangles if q and kv) // needed
            slot_bytes = lowering.score_slot_bytes or transfer
            if (transfer > slot_bytes or ("ub" in dict(ctx.datapath.buffers)
                    and slot_bytes * lowering.slots > ctx.datapath.usable_bytes("ub"))):
                raise unsupported("score_slot_capacity", "Live score slots exceed the paired AIV UB")
        owners = owner_pairs(ctx, needed)
        for aic, aivs in owners:
            if transfer:
                for aiv in aivs:
                    ctx.datapath.route("l0c", "ub", representation, owner_group=(aic, aiv), tile=transfer)
        return build_plan(op, ctx, contract, lowering, owners, sources)


STRATEGY = _Arch35Strategy()
