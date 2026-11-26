"""Tiled sparse indexer, pooled-indexer and quant-indexer physical recipes.

[Role] Execute key/query load, per-head or weighted-query score, activation and
  UB TopK update as finite phases. The native pooled indexer and the arch35
  quantized lightning indexer are 32-AIC wave programs; the KeyPool producer is
  a two-stage AIC projection plus a 64-AIV token-serial state ring.
[Invariants] Score arithmetic remains per query; shared dense prefix key reads
  amortize only through the declared batch traversal. A fused score+TopK path
  writes zero score bytes to GM, while an explicitly unfused path retains its
  materialization declaration. Every program consumes exactly the declared
  plan work: wave, block and selection demands are the same additive pieces.
[Timeouts & Locks] Bounded tile and heap work; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, tiled_phased_program)
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import (
    KERNEL_READINESS_VALUES,
    KernelBinding,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_kernel_arch35.h",
        "37b78aab1ddb5eb0139e8a8c710610bbbccdc32359c4352ec302057153ed8945",
    ),
)

PIPELINE_SLOTS = 2
INDEXER_TILE_CAP = 64

MS_PER_SECOND = 1000
NS_TO_MS = 1e-6
RATE_INDEX = 1
DRAM_SERVICE_MODE = "mte2_aic_dram"
SERIAL_SLOTS = 1
BF16_ELEMENT_BYTES = 2
FP32_ELEMENT_BYTES = 4
INT32_ELEMENT_BYTES = 4
POOL_KEY_BLOCK = 1024
#: Default Arch35 cube-wave row width (32 AIC owners) when ledger wave_rows is built.
INDEXER_WAVE_ROWS = 32
#: 3D `q_descale * k_descale`, per-head weight, ReLU, tree max and accumulate.
SCORE_REDUCE_OPS = 4
#: Per-head ReLU, weight multiply, and pairwise head reduction (`3 * H - 1` ops/pair).
SCORE_HEAD_REDUCE_MULTIPLIER = 3
#: 32-lane bitonic compare/exchange over the running candidate list, two passes.
TOPK_MERGE_OPS = 64
#: Selected-pool to token-index expansion: pool-id multiply, stride add, store.
SELECTION_OPS = 2
#: Per-sequence ring normalization: ape add, mean, variance, scale, gate.
RING_NORM_OPS = 8
_FP8_SCORE_DTYPES = frozenset({"fp8", "fp8_e4m3"})

POOL_LEDGER = "pool_key_indexer_work"
QUANT_LEDGER = "quant_indexer_arch35_work"
KEYPOOL_LEDGER = "cann_key_pool_work"
_ARCH_PRESET_FAMILY = {
    "npu_arch_3510": "A5",
    "npu_arch_2201": "A3",
}


def _primitive_ms(context, primitive_id: str, metric: str) -> float:
    """Read a calibrated primitive latency in ms from ``context.primitive_table``."""
    table = getattr(context, "primitive_table", None)
    if table is None or primitive_id not in getattr(table, "primitives", ()):
        return 0.0
    band = table.cell(primitive_id).metrics.get(metric)
    if band is None or band.p50 is None or band.p50 <= 0.0:
        return 0.0
    return float(band.p50) * NS_TO_MS


def _geometry_arch_family(geometry) -> str:
    if geometry is None:
        return "unknown"
    compute_arch = getattr(geometry, "compute_arch", None) or getattr(
        getattr(geometry, "datapath", None), "compute_arch", None
    )
    preset = str(
        getattr(compute_arch, "preset", "")
        or getattr(compute_arch, "arch_id", "")
        or ""
    ).strip()
    if preset in _ARCH_PRESET_FAMILY:
        return _ARCH_PRESET_FAMILY[preset]
    fam = str(getattr(geometry, "arch_family", "") or "").strip()
    return fam or "unknown"


def _geometry_ub_capacity(geometry) -> int:
    if geometry is None:
        return 0
    raw = dict(getattr(geometry, "buffers", ())).get("ub")
    return int(raw) if raw is not None else 0


def _indexer_owner_schedule(
    *,
    owner_kind: str,
    tiling_strategy: str,
    parallel_axis: str,
    work_tiles: int,
    total_owners: int,
    read_bytes: float,
    write_bytes: float,
    score_flops: float,
    topk_ops: float,
    ub_capacity_bytes: int | None = None,
    dependent_steps: int = 1,
) -> OwnerScheduleSpec:
    ub_cap = int(ub_capacity_bytes) if ub_capacity_bytes is not None else 0
    total = max(1, int(total_owners))
    tiles = max(0, int(work_tiles))
    active = max(1, min(total, tiles)) if tiles > 0 else 0
    max_per_owner = ((tiles + active - 1) // active) if active > 0 else 0
    denom = float(max(1, active))
    phases: tuple[SerialPhaseSpec, ...] = ()
    if active > 0:
        phases = (
            SerialPhaseSpec("kv_query_load", "mte2", read_bytes / denom, "bytes"),
            SerialPhaseSpec("tile_score_dot", "cube", score_flops / denom, "flop",
                            dependent_chain_steps=max(1, int(dependent_steps))),
            SerialPhaseSpec("online_topk_merge", "vec_reduce", topk_ops / denom, "op",
                            dependent_chain_steps=max(1, int(dependent_steps))),
            SerialPhaseSpec("index_store", "mte3", write_bytes / denom, "bytes"),
        )
    ub_est = max(POOL_KEY_BLOCK * FP32_ELEMENT_BYTES, FP32_ELEMENT_BYTES)
    ub_live = min(ub_cap, ub_est) if ub_cap > 0 else ub_est
    return OwnerScheduleSpec(
        owner_kind=owner_kind,
        tiling_strategy=tiling_strategy,
        parallel_axis=parallel_axis,
        work_tiles=tiles,
        launched_owners=active,
        active_owners=active,
        total_owners=total,
        max_tiles_per_owner=max_per_owner,
        residency=OwnerMemoryResidency(
            ub_live_bytes=ub_live,
            ub_capacity_bytes=ub_cap,
            owner_mte2_bytes=(read_bytes / denom) if active > 0 else 0.0,
            owner_mte3_bytes=(write_bytes / denom) if active > 0 else 0.0,
            shared_hbm_read_bytes=read_bytes,
            shared_hbm_write_bytes=write_bytes,
            l2_admission_mode="reuse",
        ),
        serial_phases=phases,
        dispatch_envelope_source="P15_dispatch",
    )


def _indexer_kernel_binding(
    kernel_id: str,
    recipe_id: str,
    *,
    geometry=None,
    op=None,
) -> KernelBinding:
    arch = _geometry_arch_family(geometry)
    explicit = str(dict(getattr(op, "metadata", ()) or ()).get("readiness") or "").strip() if op is not None else ""
    if explicit and explicit in KERNEL_READINESS_VALUES:
        readiness = explicit
    elif arch in {"A2", "A3", "A5"}:
        from mindie_sim.backends.abi_registry import default_abi_registry

        reg = default_abi_registry()
        if reg.is_shipping_available(kernel_id, arch) or (
            recipe_id in {"sparse_indexer", "index_scores"}
            and not (kernel_id == "npu_indexer_region_score" and arch in {"A2", "A3"})
            and reg.is_shipping_available("aclnnLightningIndexer", arch)
        ):
            readiness = "shipping"
        else:
            readiness = "synthesized_target"
    else:
        readiness = "synthesized_target"
    return KernelBinding(
        kernel_id=kernel_id,
        backend="CANN",
        arch_family=arch,
        submodule="sparse_indexer",
        macro_module="attention",
        readiness=readiness,  # type: ignore[arg-type]
        recipe_id=recipe_id,
    )


def bind_sustained_hbm(service_modes, rates, resource_names):
    """Cap the `hbm` service at the declared sustainable DRAM rate."""
    dram = service_modes.get(DRAM_SERVICE_MODE)
    if dram is None or "hbm" not in resource_names:
        return None
    ceiling = min(dram[RATE_INDEX],
                  rates[resource_names["hbm"]] * MS_PER_SECOND)
    return ("hbm", ceiling)


@dataclass(frozen=True)
class TiledSparseIndexerRecipe:
    """Class B: sparse score, activation and UB TopK selection."""

    tile_cap: int = INDEXER_TILE_CAP
    slots: int = PIPELINE_SLOTS
    traffic_rewrite_capable: bool = True

    def build(self, op, geometry):
        plan = build_plan(self, op, geometry)
        total_aic = int(
            getattr(geometry, "compute_units", 0)
            or (getattr(geometry, "resource_units", {}) or {}).get("mte2_aic", 0)
            or 1
        )
        rows = int(op.local_shape[0]) if op.local_shape else 1
        ub_cap = _geometry_ub_capacity(geometry)
        owner_schedule = _indexer_owner_schedule(
            owner_kind="mix",
            tiling_strategy="tiled_sparse_indexer",
            parallel_axis="query_rows",
            work_tiles=max(1, rows),
            total_owners=total_aic,
            read_bytes=float(op.mte2_bytes or 0.0),
            write_bytes=float(op.mte3_bytes or 0.0),
            score_flops=float(op.flops or 0.0),
            topk_ops=float(op.vector_ops or 0.0),
            ub_capacity_bytes=ub_cap,
        )
        meta = dict(getattr(plan, "metadata", {}) or {})
        meta["owner_schedule"] = owner_schedule.to_dict()
        meta["kernel_binding"] = _indexer_kernel_binding(
            str(dict(op.metadata).get("native_kind") or "SparseIndexerScoreTopK"),
            str(op.recipe or "sparse_indexer"),
            geometry=geometry,
            op=op,
        ).to_dict()
        return replace(plan, metadata=meta)

    def lower(self, context):
        read_share = io_weight(context, "hbm_read")
        write_share = io_weight(context, "hbm_write")
        phases = (
            Phase("operand_load", {IO_HBM_CATEGORY: read_share, "mte2": 1.0}),
            Phase("score", {"cube": 1.0}, after=("operand_load",)),
            Phase("topk_update", {"vector": 1.0}, after=("score",)),
            Phase("store", {IO_HBM_CATEGORY: write_share, "mte3": 1.0},
                  after=("topk_update",)),
        )
        return tiled_phased_program(
            context, phases, slots=self.slots, cap=self.tile_cap)


def _wave_rows(rows, index, wave_rows):
    return max(0, min(wave_rows, rows - index * wave_rows))


def _waves(rows, wave_rows):
    return max(1, int(math.ceil(float(rows) / float(wave_rows)))) if rows else 1


def indexer_waves(query_rows, wave_rows=INDEXER_WAVE_ROWS):
    """`W_T = ceil(T / C_AIC)`; the only wave count a 32-AIC owner grid has."""
    return _waves(int(query_rows), int(wave_rows))


def key_blocks(keys):
    """`N_blk = ceil(K / 1024)`; the only key-block count the cache grid has."""
    return max(1, int(math.ceil(float(keys) / float(POOL_KEY_BLOCK))))


def _matrix_free(values, dtype):
    return {key: value for key, value in values.items() if key != dtype}


_AIC_EXECUTION_ROLES = frozenset({"cube", "mte1", "mte2_aic", "fixpipe"})


def _stream_owner_placement(
    context,
    stream,
    active_owners: int,
    total_owners: int,
    roles: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    dp = getattr(context, "datapath", None)
    if dp is not None and hasattr(stream, "paired_owners"):
        pairs = stream.paired_owners(dp)
        if pairs:
            active = max(1, min(int(active_owners), len(pairs)))
            selected_aic = tuple(sorted({aic_id for aic_id, _ in pairs[:active]}))
            selected_aiv = tuple(sorted({aiv_id for _, aivs in pairs[:active] for aiv_id in aivs}))
            merged: dict[str, tuple[int, ...]] = {}
            for role in roles:
                phys = context.physical_role(role)
                name = context.resource_names.get(phys)
                if not name:
                    continue
                if phys in _AIC_EXECUTION_ROLES:
                    units = selected_aic
                else:
                    pool = stream.pool_for_role(phys, datapath=dp)
                    units = selected_aiv if selected_aiv else pool[:active]
                if units:
                    merged[name] = units
            if merged:
                return merged
    aic_pool = stream.units_for("aic")
    aiv_pool = stream.units_for("aiv")
    active = max(1, min(int(active_owners), max(1, len(aic_pool) if aic_pool else total_owners)))
    aiv_per_aic = max(1, len(aiv_pool) // max(1, len(aic_pool))) if (aic_pool and aiv_pool) else 1
    merged_pool: dict[str, tuple[int, ...]] = {}
    for role in roles:
        phys = context.physical_role(role)
        name = context.resource_names.get(phys)
        if not name:
            continue
        if phys in _AIC_EXECUTION_ROLES and aic_pool:
            units = tuple(sorted(aic_pool[:active]))
        elif aiv_pool:
            units = tuple(sorted(aiv_pool[:active * aiv_per_aic]))
        else:
            continue
        if units:
            merged_pool[name] = units
    return merged_pool


def _active_owner_placement(
    context,
    active_owners: int,
    total_owners: int,
    roles: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    """Bind per-owner resources strictly to the active owner slice of this stream."""
    total = max(1, int(total_owners))
    stream = getattr(context, "stream_owners", None)
    if stream is not None:
        stream_place = _stream_owner_placement(context, stream, active_owners, total, roles)
        if stream_place:
            return stream_place
    active = max(1, min(int(active_owners), total))
    merged: dict[str, set[int]] = {}
    for owner in range(active):
        for name, units in context.owned(owner, total, roles).items():
            merged.setdefault(name, set()).update(units)
    return {name: tuple(sorted(units)) for name, units in merged.items() if units}


def wave_key_ranges_from_spans(request_spans, wave_rows):
    """Derive per-wave distinct interval fragments for an explicit core count."""
    wave_rows = max(1, int(wave_rows))
    waves, batch, prefixes, private = [], 0, {}, {}

    def flush():
        waves.append(tuple((0, n) for n in prefixes.values()) + tuple(private.values()))
        prefixes.clear()
        private.clear()

    for request, (runs, extent, key) in enumerate(request_spans):
        for count, n in runs:
            rem = int(count)
            while rem > 0:
                room = wave_rows - (batch % wave_rows)
                take = min(rem, room)
                p = min(n, extent)
                if p:
                    prefixes[key] = max(prefixes.get(key, 0), p)
                if n > p:
                    private[request] = (extent, max(n, private.get(request, (0, 0))[1]))
                batch += take
                rem -= take
                if batch % wave_rows == 0:
                    flush()
    if batch % wave_rows:
        flush()
    return tuple(waves)


def _stream_aic_width(stream_owners) -> int:
    if stream_owners is None:
        return 0
    aic_limit = getattr(stream_owners, "aic_limit", None)
    if aic_limit is not None and int(aic_limit) > 0:
        return int(aic_limit)
    aic_ids = getattr(stream_owners, "aic", ()) or getattr(stream_owners, "pool_aic", ())
    return len(aic_ids) if aic_ids else 0


def _resolve_quant_ledger(ledger: dict, core_count: int) -> dict:
    """Resolve physical wave width/count from active stream/die AIC count."""
    if core_count <= 0 or "query_rows" not in ledger:
        return ledger
    wave_rows = int(core_count)
    query_rows = int(ledger.get("query_rows", 0) or 0)
    waves = indexer_waves(query_rows, wave_rows) if query_rows > 0 else 0
    old_wave_rows = int(ledger.get("wave_rows", 0) or 0)
    if old_wave_rows == wave_rows and int(ledger.get("waves", 0) or 0) == waves:
        return ledger
    updated = {**ledger, "wave_rows": wave_rows, "waves": waves}
    if old_wave_rows != wave_rows:
        if "request_key_spans" in ledger:
            updated["wave_key_ranges"] = wave_key_ranges_from_spans(
                ledger["request_key_spans"], wave_rows,
            )
        else:
            updated.pop("wave_key_ranges", None)
    return updated


def _wave_pool_cache_bytes(ledger, index, *, hbm: bool = False):
    key = "wave_hbm_cache_bytes_by_wave" if hbm else "wave_cache_bytes_by_wave"
    series = ledger.get(key)
    if series is not None and index < len(series):
        return float(series[index])
    if hbm:
        series = ledger.get("wave_cache_bytes_by_wave")
        if series is not None and index < len(series):
            return float(series[index])
    return float(ledger["wave_cache_bytes"])


def _total_pool_cache_bytes(ledger, *, hbm: bool = False):
    waves = int(ledger.get("waves", 0))
    return sum(_wave_pool_cache_bytes(ledger, idx, hbm=hbm) for idx in range(waves))


def _pool_wave_work(ledger, index):
    """Declared per-wave service of one AIC window, in physical units."""
    rows = _wave_rows(int(ledger["query_rows"]), index, int(ledger["wave_rows"]))
    heads, width = int(ledger["heads"]), int(ledger["width"])
    dtype = str(ledger["score_dtype"])
    elem_bytes = float(dtype_bytes(dtype))
    wave_pairs = ledger.get("wave_score_pairs")
    if wave_pairs is not None and index < len(wave_pairs):
        pairs = float(wave_pairs[index])
    else:
        pairs = float(ledger["score_pairs"]) * float(rows) / float(
            max(1, int(ledger["query_rows"])))
    wave_mte2_cache = _wave_pool_cache_bytes(ledger, index, hbm=False)
    wave_hbm_cache = _wave_pool_cache_bytes(ledger, index, hbm=True)
    wave_descale = float(ledger.get("wave_descale_bytes", 0.0))
    query_payload = float(rows) * heads * width * elem_bytes
    query_scale = (float(rows) * heads * FP32_ELEMENT_BYTES
                   if dtype in _FP8_SCORE_DTYPES else 0.0)
    query_bytes = query_payload + query_scale
    pool_row_bytes = float(ledger.get("pool_row_bytes", 0) or 0)
    payload_row_bytes = float(width) * elem_bytes
    if pool_row_bytes > 0.0:
        wave_pools = wave_mte2_cache / pool_row_bytes
        key_payload = wave_pools * min(pool_row_bytes, payload_row_bytes)
        key_scale = max(0.0, wave_mte2_cache - key_payload)
    else:
        key_payload = wave_mte2_cache
        key_scale = 0.0
    aic_read = query_payload + key_payload
    aiv_read = query_scale + key_scale + wave_descale
    scanned_pools = pairs / float(max(1, heads))
    mte1_bytes = (query_payload + scanned_pools * width * elem_bytes) if pairs > 0.0 else 0.0
    return {
        "rows": rows,
        "dtype": dtype,
        "wave_hbm_cache": wave_hbm_cache,
        "aic_read": aic_read,
        "aiv_read": aiv_read,
        "stream": {
            "hbm": wave_hbm_cache + wave_descale + query_bytes,
            "mte2": aic_read + aiv_read,
        },
        "tiles": {
            dtype: 2.0 * pairs * width,
            "mte1": mte1_bytes,
            "vector": pairs * SCORE_REDUCE_OPS,
        },
    }


def _pool_selection_work(ledger):
    output = float(ledger["output_bytes"])
    return {
        "hbm": output + float(ledger["weight_bytes"]),
        "mte3": output,
        "vector": float(ledger["query_rows"]) * int(ledger["output_width"]) * SELECTION_OPS,
    }


def pool_key_indexer_work(ledger):
    """Total declared work of the pooled indexer, summed from its waves."""
    total = Work()
    for index in range(int(ledger["waves"])):
        wave = _pool_wave_work(ledger, index)
        total = total + Work({wave["dtype"]: wave["tiles"][wave["dtype"]]},
                             {**wave["stream"],
                              **_matrix_free(wave["tiles"], wave["dtype"])})
    selection = _pool_selection_work(ledger)
    return total + Work({}, dict(selection))


@dataclass(frozen=True)
class PoolKeyIndexerRecipe:
    """Class B: `aclnnPoolKeyIndexer` `quant_mode=0` on 32 AIC owners.

    One wave covers up to `INDEXER_WAVE_ROWS` query rows; every owner holds one
    row and walks the pool cache in `POOL_KEY_BLOCK` blocks. The AIC cube dot
    and the 2-AIV descale/head-reduce/merge advance together inside one wave
    (concurrent resources), while the next wave waits for the current one's
    drain. The selected pool ids expand to token indices once, after the last
    wave.
    """

    traffic_rewrite_capable: bool = True

    def bind_context(self, context):
        mode = bind_sustained_hbm(
            dict(context.service_modes), context.rates, context.resource_names)
        if mode is None:
            return context
        return replace(context, service_modes={
            **context.service_modes, "hbm": mode})

    def build(self, op, geometry):
        metadata = dict(op.metadata)
        ledger = dict(metadata.get(POOL_LEDGER, {}))
        work = pool_key_indexer_work(ledger)
        res_units = dict(getattr(geometry, "resource_units", {}) or {}) if geometry is not None else {}
        if "mte2_aic" in res_units and "mte2_aiv" in res_units:
            waves = int(ledger.get("waves", 0))
            aic_total = sum(_pool_wave_work(ledger, idx)["aic_read"] for idx in range(waves))
            aiv_total = sum(_pool_wave_work(ledger, idx)["aiv_read"] for idx in range(waves))
            if aiv_total > 0.0:
                resources = dict(work.resources)
                resources.pop("mte2", None)
                resources["mte2_aic"] = aic_total
                resources["mte2_aiv"] = aiv_total
                work = Work(work.matrix, resources)
        # A shared-prefix plan replaces only the KV cache component of the HBM
        # fill with its physical union; the per-wave MTE2 refill service stays
        # complete.  The prepared plan must carry both components unchanged.
        prefix_hbm = metadata.get("prefix_read_hbm_bytes")
        if prefix_hbm is not None:
            kv_component = _total_pool_cache_bytes(ledger, hbm=True)
            resources = dict(work.resources)
            resources["hbm"] = (float(work.resources.get("hbm", 0.0))
                                - kv_component + float(prefix_hbm))
            work = Work(work.matrix, resources)
        ub_cap = _geometry_ub_capacity(geometry)
        total_owners = int(
            getattr(geometry, "compute_units", 0)
            or ledger.get("wave_rows", INDEXER_WAVE_ROWS)
            or INDEXER_WAVE_ROWS
        )
        total_read = (
            float(work.resources.get("mte2", 0.0))
            + float(work.resources.get("mte2_aic", 0.0))
            + float(work.resources.get("mte2_aiv", 0.0))
        )
        owner_schedule = _indexer_owner_schedule(
            owner_kind="mix",
            tiling_strategy="pool_key_indexer_aic_waves",
            parallel_axis="query_rows",
            work_tiles=max(1, int(ledger.get("query_rows", 1) or 1)),
            total_owners=total_owners,
            read_bytes=total_read,
            write_bytes=float(work.resources.get("mte3", 0.0)),
            score_flops=float(sum(work.matrix.values())),
            topk_ops=float(work.resources.get("vector", 0.0)),
            ub_capacity_bytes=ub_cap,
            dependent_steps=max(1, int(ledger.get("blocks", 1) or 1)),
        )
        return OperatorPlan(work, self.lower, metadata={
            "schedule": "pool_key_indexer_aic_waves",
            "native_dispatch_reconstructed": False,
            "wave_rows": ledger.get("wave_rows"),
            "waves": ledger.get("waves"),
            "blocks": ledger.get("blocks"),
            "owner_schedule": owner_schedule.to_dict(),
            "kernel_binding": _indexer_kernel_binding(
                "aclnnPoolKeyIndexer", str(op.recipe or "pool_key_indexer"), geometry=geometry, op=op,
            ).to_dict(),
        })

    def lower(self, context):
        metadata = dict(context.op.metadata)
        ledger = dict(metadata.get(POOL_LEDGER, {}))
        waves = int(ledger["waves"])
        prefix_hbm = metadata.get("prefix_read_hbm_bytes")
        total_hbm_cache = _total_pool_cache_bytes(ledger, hbm=True)
        handoff_ms = _primitive_ms(context, "P13_crosscore_sync", "mode2_paired_ns")
        floor = (
            _primitive_ms(context, "P15_rts_stars", "aic_first_issue_ns")
            + int(ledger["blocks"]) * _primitive_ms(context, "P13_crosscore_sync", "mode4_intra_block_ns")
        )
        aic_owners = max(1, int(context.core_count))
        owner_budget = max(1, min(aic_owners, int(context.effective_aic_cores or aic_owners)))
        has_split_mte2 = (
            "mte2_aic" in context.resource_names
            and "mte2_aiv" in context.resource_names
            and any(_pool_wave_work(ledger, idx)["aiv_read"] > 0.0 for idx in range(waves))
        )

        def wave(index):
            parts = _pool_wave_work(ledger, index)
            stream = dict(parts["stream"])
            if has_split_mte2:
                stream.pop("mte2", None)
                stream["mte2_aic"] = float(parts["aic_read"])
                stream["mte2_aiv"] = float(parts["aiv_read"])
            base_hbm_cache = float(parts["wave_hbm_cache"])
            if prefix_hbm is not None:
                wave_hbm_cache = (
                    float(prefix_hbm) * base_hbm_cache / total_hbm_cache
                    if total_hbm_cache > 0.0
                    else float(prefix_hbm) / float(max(1, waves))
                )
                stream["hbm"] = stream["hbm"] - base_hbm_cache + wave_hbm_cache
            active = max(1, min(int(parts["rows"]), owner_budget))
            stream_roles = ("mte2_aic", "mte2_aiv") if has_split_mte2 else ("mte2",)
            stream_place = _active_owner_placement(
                context, active, aic_owners, stream_roles,
            )
            tiles_place = _active_owner_placement(
                context, active, aic_owners, ("cube", "mte1", "vector"),
            )
            return (
                Stage("stream", context.bind(Work({}, stream)), placement=stream_place),
                Stage("tiles", context.bind(Work(
                    {parts["dtype"]: parts["tiles"][parts["dtype"]]},
                    _matrix_free(parts["tiles"], parts["dtype"]))),
                    after=("stream",), placement=tiles_place),
                Stage("handoff", after=("tiles",), latency=handoff_ms),
            )

        selection = _pool_selection_work(ledger)
        sel_active = max(1, min(int(ledger.get("query_rows", 1) or 1), owner_budget))
        sel_place = _active_owner_placement(context, sel_active, aic_owners, ("vector", "mte3"))
        return KernelProgram((
            Stage("waves", program=PipelineProgram(waves, SERIAL_SLOTS, wave)),
            Stage("selection", context.bind(Work({}, dict(selection))),
                  after=("waves",), latency=floor, placement=sel_place),
        ))


def _wave_row_keys(ledger, wave_index):
    """Completed keys of every query row in one 32-AIC wave."""
    rows = _wave_rows(int(ledger["query_rows"]), wave_index,
                      int(ledger["wave_rows"]))
    runs = ledger.get("row_completed_keys")
    if not runs:
        return (int(ledger.get("completed_keys", ledger["keys"])),) * rows
    begin = wave_index * int(ledger["wave_rows"])
    end = begin + rows
    if isinstance(runs[0], int):
        return tuple(int(value) for value in runs[begin:end])
    values, position = [], 0
    for count, value in runs:
        count = int(count)
        lo, hi = max(begin, position), min(end, position + count)
        if hi > lo:
            values.extend([int(value)] * (hi - lo))
        position += count
        if position >= end:
            break
    return tuple(values)


def _quant_tile_work(ledger, index):
    """Declared per-(wave, key block) service, in physical units."""
    wave = index // int(ledger["blocks"])
    rows = _wave_rows(int(ledger["query_rows"]), wave,
                      int(ledger["wave_rows"]))
    heads, width = int(ledger["heads"]), int(ledger["width"])
    block = index % int(ledger["blocks"])
    # Each query row scans its own completed keys; one wave advances together.
    row_valids = [
        max(0, min(POOL_KEY_BLOCK, key - block * POOL_KEY_BLOCK))
        for key in _wave_row_keys(ledger, wave)]
    active_rows = sum(1 for value in row_valids if value > 0)
    valid = max(row_valids, default=0)
    tile_pairs = sum(row_valids)
    query_bytes = active_rows * heads * (width + 2 * FP32_ELEMENT_BYTES) if block == 0 else 0
    start = block * POOL_KEY_BLOCK
    stop = start + POOL_KEY_BLOCK
    ranges = ledger.get("wave_key_ranges")
    fragments = (
        ranges[wave]
        if ranges is not None and len(ranges) == int(ledger["waves"])
        else ((0, start + valid),)
    )
    unique = sum(max(0, min(hi, stop) - max(lo, start)) for lo, hi in fragments)
    query_heads = active_rows * heads if block == 0 else 0
    cube_input = (query_heads + unique) * width
    vector_input = (2 * query_heads + unique) * FP32_ELEMENT_BYTES
    return {
        "rows": rows,
        "active_rows": active_rows,
        "valid_keys": valid,
        "cube": {
            "fp8_e4m3": float(tile_pairs) * 2.0 * heads * width,
            # The cold cache rows are filled once; a wave rescans them from L2.
            "hbm": query_bytes,
            # The operand fetch includes this block's key rows and scales.
            "mte2_aic": cube_input,
            "mte2_aiv": vector_input,
        },
        "reduce": {
            "vector": float(tile_pairs) * (
                float(SCORE_HEAD_REDUCE_MULTIPLIER * heads - 1) + TOPK_MERGE_OPS
            ),
        },
    }


def arch35_indexer_work(ledger):
    """Total declared work of the arch35 indexer, summed from its key blocks."""
    tiles = int(ledger["waves"]) * int(ledger["blocks"])
    total = Work()
    for index in range(tiles):
        parts = _quant_tile_work(ledger, index)
        total = total + Work({"fp8_e4m3": parts["cube"]["fp8_e4m3"]},
                             {**_matrix_free(parts["cube"], "fp8_e4m3"), **parts["reduce"]})
    payload = int(ledger.get("payload_keys", ledger.get("completed_keys", ledger["keys"])))
    fill = float(payload) * int(ledger["key_row_bytes"])
    total = total + Work({}, {
        "mte3": float(ledger["output_bytes"]),
        "hbm": float(ledger["output_bytes"]) + fill,
    })
    return total


@dataclass(frozen=True)
class Arch35QuantIndexerRecipe:
    """Class B: `aclnnQuantLightningIndexerV2` arch35 FP8 (`quant_mode=1`).

    `W_T = ceil(T / C_AIC)` waves over `ceil(K / 1024)` valid key blocks. Each
    block is one FP8 Cube dot of `2 * H * 128 * valid_keys` followed by the
    `H`-independent rolling per-token AIV TopK merge; the two stages are
    ordered because the merge consumes the published score block.
    """

    traffic_rewrite_capable: bool = True

    def bind_context(self, context):
        mode = bind_sustained_hbm(
            dict(context.service_modes), context.rates, context.resource_names)
        if mode is None:
            return context
        return replace(context, service_modes={
            **context.service_modes, "hbm": mode})

    def build(self, op, geometry):
        raw_ledger = dict(dict(op.metadata).get(QUANT_LEDGER, {}))
        stream_aics = _stream_aic_width(getattr(geometry, "stream_owners", None))
        core_count = int(stream_aics or getattr(geometry, "compute_units", 0) or 0)
        ledger = _resolve_quant_ledger(raw_ledger, core_count)
        work = arch35_indexer_work(ledger)
        ub_cap = _geometry_ub_capacity(geometry)
        total_owners = int(
            core_count
            or ledger.get("wave_rows", INDEXER_WAVE_ROWS)
            or INDEXER_WAVE_ROWS
        )
        owner_schedule = _indexer_owner_schedule(
            owner_kind="mix",
            tiling_strategy="quant_lightning_indexer_v2_arch35_blocks",
            parallel_axis="query_rows",
            work_tiles=max(1, int(ledger.get("query_rows", 1) or 1)),
            total_owners=total_owners,
            read_bytes=float(work.resources.get("mte2_aic", 0.0)) + float(work.resources.get("mte2_aiv", 0.0)),
            write_bytes=float(work.resources.get("mte3", 0.0)),
            score_flops=float(sum(work.matrix.values())),
            topk_ops=float(work.resources.get("vector", 0.0)),
            ub_capacity_bytes=ub_cap,
            dependent_steps=max(1, int(ledger.get("blocks", 1) or 1)),
        )
        return OperatorPlan(work, self.lower, metadata={
            "schedule": "quant_lightning_indexer_v2_arch35_blocks",
            "native_dispatch_reconstructed": False,
            "wave_rows": ledger.get("wave_rows"),
            "waves": ledger.get("waves"),
            "blocks": ledger.get("blocks"),
            "owner_schedule": owner_schedule.to_dict(),
            "kernel_binding": _indexer_kernel_binding(
                "aclnnQuantLightningIndexerV2",
                str(op.recipe or "quant_lightning_indexer"),
                geometry=geometry,
                op=op,
            ).to_dict(),
        })

    def lower(self, context):
        aic_owners = max(1, int(context.core_count))
        stream_aics = _stream_aic_width(getattr(context, "stream_owners", None))
        eff_aics = int(stream_aics or context.effective_aic_cores or aic_owners)
        owner_budget = max(1, min(aic_owners, eff_aics))
        ledger = _resolve_quant_ledger(
            dict(dict(context.op.metadata).get(QUANT_LEDGER, {})), owner_budget,
        )
        tiles = int(ledger["waves"]) * int(ledger["blocks"])
        floor = _primitive_ms(context, "P15_rts_stars", "aic_first_issue_ns")
        payload = int(ledger.get("payload_keys",
                                 ledger.get("completed_keys", ledger["keys"])))
        fill = float(payload) * int(ledger["key_row_bytes"])
        indices_active = max(1, min(int(ledger.get("query_rows", 1) or 1), owner_budget))
        indices_place = _active_owner_placement(context, indices_active, aic_owners, ("mte3",))
        if tiles == 0:
            # No completed key owns a block: only the INT32 index store runs.
            stages = []
            if fill > 0.0:
                stages.append(Stage("cache_fill", context.bind(Work({}, {
                    "hbm": fill}))))
            stages.append(Stage("indices", context.bind(Work({}, {
                "mte3": float(ledger["output_bytes"]),
                "hbm": float(ledger["output_bytes"])})), latency=floor, placement=indices_place))
            return KernelProgram(tuple(stages))

        def tile(index):
            parts = _quant_tile_work(ledger, index)
            cube, reduce = parts["cube"], parts["reduce"]
            active = max(1, min(int(parts["active_rows"]), owner_budget))
            operands_place = _active_owner_placement(
                context, active, aic_owners, ("cube", "mte2_aic", "mte2_aiv"),
            )
            merge_place = _active_owner_placement(context, active, aic_owners, ("vector",))
            return (
                Stage("operands", context.bind(Work(
                    {"fp8_e4m3": cube["fp8_e4m3"]},
                    _matrix_free(cube, "fp8_e4m3"))), placement=operands_place),
                Stage("topk_merge", context.bind(Work({}, dict(reduce))),
                      after=("operands",), placement=merge_place),
            )

        return KernelProgram((
            # One cold fill of every distinct request cache row. Later waves
            # rescan the resident rows through their operand fetch.
            Stage("cache_fill", context.bind(Work({}, {
                "hbm": fill}))),
            Stage("blocks", after=("cache_fill",),
                  program=PipelineProgram(tiles, SERIAL_SLOTS, tile)),
            Stage("indices", context.bind(Work({}, {
                "mte3": float(ledger["output_bytes"]),
                "hbm": float(ledger["output_bytes"])})),
                after=("blocks",), latency=floor, placement=indices_place),
        ))


def _keypool_step_work(ledger):
    pool, width = int(ledger["pool_size"]), int(ledger["width"])
    ring = float(pool) * 2 * width * FP32_ELEMENT_BYTES
    # AIV reads this token's FP32 key & gate projections written by AIC Fixpipe.
    proj_token_read = (
        2.0 * width * FP32_ELEMENT_BYTES
        if float(ledger.get("projection_output_bytes", 0.0)) > 0.0
        else 0.0
    )
    return {
        "read": {"hbm": ring + proj_token_read, "mte2_aiv": ring + proj_token_read},
        "norm": {"vector": float(pool) * width * RING_NORM_OPS},
        "write": {"hbm": ring, "mte3": ring},
    }


def _merged(*parts):
    """Additive merge of physical unit maps; a repeated role sums, never wins."""
    total = {}
    for part in parts:
        for name, amount in part.items():
            total[name] = total.get(name, 0.0) + float(amount)
    return total


def _keypool_request_schedules(ledger):
    """Per-request `(ring_steps, context, completed_pools)` preserving batch totals."""
    raw_lengths = ledger.get("query_lengths")
    if raw_lengths is not None:
        lengths = tuple(max(0, int(q)) for q in raw_lengths)
    else:
        batch = max(0, int(ledger.get("batch", 0)))
        steps = max(0, int(ledger.get("steps", 0)))
        lengths = (steps,) * batch
    count = len(lengths)
    raw_contexts = ledger.get("contexts_by_request")
    if raw_contexts is not None and len(raw_contexts) == count:
        contexts = tuple(max(0, int(c)) for c in raw_contexts)
    else:
        contexts = (0,) * count
    raw_pools = ledger.get("completed_pools_by_request")
    if raw_pools is not None and len(raw_pools) == count:
        pools = tuple(max(0, int(p)) for p in raw_pools)
    elif count > 0:
        total_pools = max(0, int(ledger.get("completed_pools", 0)))
        base, rem = divmod(total_pools, count)
        pools = tuple(base + (1 if idx < rem else 0) for idx in range(count))
    else:
        pools = ()
    return tuple(zip(lengths, contexts, pools))


def _keypool_projection_resources(ledger):
    proj_in = float(ledger["hidden_bytes"]) + float(ledger["weight_bytes"])
    proj_out = float(ledger.get("projection_output_bytes", 0.0))
    res = {"hbm": proj_in + proj_out, "mte2": proj_in}
    if proj_out > 0.0:
        res["fixpipe"] = proj_out
    return res


def key_pool_work(ledger):
    """Total declared work of the KeyPool producer program."""
    schedules = _keypool_request_schedules(ledger)
    total_steps = sum(seq_steps for seq_steps, _, _ in schedules)
    step = _keypool_step_work(ledger)
    per_step = Work({}, _merged(step["read"], step["norm"], step["write"]))
    total = per_step.scaled(float(total_steps))
    total = total + Work(
        {"bf16": float(ledger["projection_flops"])},
        _keypool_projection_resources(ledger),
    )
    commit = _keypool_commit_work(ledger)
    return total + Work({}, dict(commit))


def _keypool_commit_work(ledger, pools=None):
    pool_count = int(ledger["completed_pools"]) if pools is None else int(pools)
    width = int(ledger["width"])
    commit_bytes = float(pool_count) * width * BF16_ELEMENT_BYTES
    return {
        "hbm": commit_bytes,
        "mte3": commit_bytes,
        "vector": float(pool_count) * int(ledger["pool_size"]) * width * RING_NORM_OPS,
    }


def _keypool_crossing_plan(seq_steps: int, context: int, seq_pools: int, pool_size: int):
    """Split one live sequence into causal ring-crossing windows before slot reuse."""
    if seq_steps <= 0 or seq_pools <= 0 or pool_size <= 0:
        return (seq_steps, 0), 0, 0, seq_pools
    first_rem = pool_size - (context % pool_size)
    if seq_steps < first_rem:
        return (seq_steps, 0), 0, 0, seq_pools
    first_steps = first_rem
    rem_steps = seq_steps - first_steps
    full_crossings = min(seq_pools - 1, rem_steps // pool_size)
    tail_steps = rem_steps - full_crossings * pool_size
    extra_pools = max(0, seq_pools - 1 - full_crossings)
    return (first_steps, 1), full_crossings, tail_steps, extra_pools


@dataclass(frozen=True)
class KeyPoolRingRecipe:
    """Class B: `aclnnKeyPool` two-stage AIC projection plus AIV state ring.

    Stage one runs the BF16 `wk`/`wgate` projections on all AIC owners, writes
    FP32 key/gate workspaces through Fixpipe, and joins through a GM/L2 barrier.
    Stage two gives every AIV a bounded sequence window; each live sequence
    walks its tokens serially, reducing and committing each completed pool at
    its exact ring-crossing boundary before subsequent steps reuse those slots.
    """

    traffic_rewrite_capable: bool = True

    def bind_context(self, context):
        mode = bind_sustained_hbm(
            dict(context.service_modes), context.rates, context.resource_names)
        if mode is None:
            return context
        return replace(context, service_modes={
            **context.service_modes, "hbm": mode})

    def build(self, op, geometry):
        ledger = dict(dict(op.metadata).get(KEYPOOL_LEDGER, {}))
        schedules = _keypool_request_schedules(ledger)
        work = key_pool_work(ledger)
        ub_cap = _geometry_ub_capacity(geometry)
        aiv_total = int((getattr(geometry, "resource_units", {}) or {}).get("vector", INDEXER_TILE_CAP) or INDEXER_TILE_CAP) if geometry is not None else INDEXER_TILE_CAP
        max_steps = max((seq_steps for seq_steps, _, _ in schedules), default=0)
        owner_schedule = _indexer_owner_schedule(
            owner_kind="mix",
            tiling_strategy="cann_key_pool_aic_projection_then_aiv_ring",
            parallel_axis="sequence_ring",
            work_tiles=max(1, len(schedules)),
            total_owners=aiv_total,
            read_bytes=float(work.resources.get("mte2", 0.0)) + float(work.resources.get("mte2_aiv", 0.0)),
            write_bytes=float(work.resources.get("mte3", 0.0)),
            score_flops=float(sum(work.matrix.values())),
            topk_ops=float(work.resources.get("vector", 0.0)),
            ub_capacity_bytes=ub_cap,
            dependent_steps=max(1, max_steps),
        )
        return OperatorPlan(work, self.lower, metadata={
            "schedule": "cann_key_pool_aic_projection_then_aiv_ring",
            "native_dispatch_reconstructed": False,
            "completed_pools": ledger.get("completed_pools"),
            "ring_steps": max_steps,
            "owner_schedule": owner_schedule.to_dict(),
            "kernel_binding": _indexer_kernel_binding(
                "aclnnKeyPool", str(op.recipe or "cann_key_pool"), geometry=geometry, op=op,
            ).to_dict(),
        })

    def lower(self, context):
        ledger = dict(dict(context.op.metadata).get(KEYPOOL_LEDGER, {}))
        schedules = _keypool_request_schedules(ledger)
        pool_size = max(1, int(ledger.get("pool_size", 1)))
        owners = max(1, int(context.resource_units.get("vector", context.core_count)))
        step = _keypool_step_work(ledger)
        one_commit = _keypool_commit_work(ledger, pools=1)
        ring_sync_ms = _primitive_ms(context, "P12_intracore_flag", "flag_v_mte3_ns")
        pool_commit_ms = _primitive_ms(context, "P12_intracore_flag", "flag_ns")
        workspace_join_ms = _primitive_ms(context, "P13_crosscore_sync", "mode0_aiv_barrier_ns")

        def ring_step(owner_index):
            owned = context.owned(owner_index, owners,
                                  ("vector", "mte2_aiv", "mte3"))
            return (
                Stage("ring_read", context.bind(Work({}, dict(step["read"]))),
                      placement=owned),
                Stage("ape_norm", context.bind(Work({}, dict(step["norm"]))),
                      after=("ring_read",), placement=owned),
                Stage("ring_write", context.bind(Work({}, dict(step["write"]))),
                      after=("ape_norm",), placement=owned),
                Stage("ring_sync", after=("ring_write",), placement=owned,
                      latency=ring_sync_ms),
            )

        def crossing_window(owner_index, win_steps, commit_pools):
            owned = context.owned(owner_index, owners, ("vector", "mte2_aiv", "mte3"))
            stages = []
            if win_steps > 0:
                stages.append(Stage("steps", program=PipelineProgram(
                    win_steps, SERIAL_SLOTS,
                    lambda index, owner=owner_index: ring_step(owner))))
            if commit_pools > 0:
                commit = one_commit if commit_pools == 1 else _keypool_commit_work(ledger, pools=commit_pools)
                stages.append(Stage(
                    "pool_commit", context.bind(Work({}, dict(commit))),
                    after=("steps",) if win_steps > 0 else (),
                    placement=owned, latency=pool_commit_ms))
            return tuple(stages)

        def build_sequence_stages(owner_index, seq_steps, seq_ctx, seq_pools):
            (first_steps, first_commit), full_crossings, tail_steps, extra_pools = _keypool_crossing_plan(
                seq_steps, seq_ctx, seq_pools, pool_size,
            )
            parts = []
            if first_steps > 0 or first_commit > 0:
                parts.append(Stage(
                    "first_window",
                    program=KernelProgram(crossing_window(owner_index, first_steps, first_commit)),
                ))
            if full_crossings > 0:
                prev = (parts[-1].name,) if parts else ()
                parts.append(Stage(
                    "pool_windows",
                    after=prev,
                    program=PipelineProgram(
                        full_crossings, SERIAL_SLOTS,
                        lambda index, owner=owner_index: crossing_window(owner, pool_size, 1)),
                ))
            if tail_steps > 0 or extra_pools > 0:
                prev = (parts[-1].name,) if parts else ()
                parts.append(Stage(
                    "tail_window",
                    after=prev,
                    program=KernelProgram(crossing_window(owner_index, tail_steps, extra_pools)),
                ))
            return tuple(parts)

        def owner_program(owner_index):
            owned_seqs = [
                schedules[seq_idx]
                for seq_idx in range(owner_index, len(schedules), owners)
            ]
            if not owned_seqs:
                return KernelProgram(())
            seq_stages = []
            for seq_steps, seq_ctx, seq_pools in owned_seqs:
                parts = build_sequence_stages(owner_index, seq_steps, seq_ctx, seq_pools)
                if not parts:
                    continue
                after = (f"seq_{len(seq_stages) - 1}",) if seq_stages else ()
                seq_stages.append(Stage(
                    f"seq_{len(seq_stages)}",
                    program=KernelProgram(parts),
                    after=after))
            return KernelProgram(tuple(seq_stages))

        return KernelProgram((
            Stage("projection", context.bind(Work(
                {"bf16": float(ledger["projection_flops"])},
                _keypool_projection_resources(ledger)))),
            Stage("workspace_join", after=("projection",),
                  latency=workspace_join_ms),
            Stage("ring", after=("workspace_join",), program=ParallelProgram(
                tuple(owner_program(owner) for owner in range(owners)))),
        ))
