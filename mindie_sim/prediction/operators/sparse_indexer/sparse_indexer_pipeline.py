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

from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, tiled_phased_program)
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.schema.operators import (
    KERNEL_READINESS_VALUES,
    KernelBinding,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)

PIPELINE_SLOTS = 2
INDEXER_TILE_CAP = 64

MS_PER_SECOND = 1000
NS_TO_MS = 1e-6
RATE_INDEX = 1
DRAM_SERVICE_MODE = "mte2_aic_dram"
SERIAL_SLOTS = 1
FP32_ELEMENT_BYTES = 4
INT32_ELEMENT_BYTES = 4
POOL_KEY_BLOCK = 1024
#: Default Arch35 cube-wave row width (32 AIC owners) when ledger wave_rows is built.
INDEXER_WAVE_ROWS = 32
#: 3D `q_descale * k_descale`, per-head weight, ReLU, tree max and accumulate.
SCORE_REDUCE_OPS = 4
#: 32-lane bitonic compare/exchange over the running candidate list, two passes.
TOPK_MERGE_OPS = 64
#: Selected-pool to token-index expansion: pool-id multiply, stride add, store.
SELECTION_OPS = 2
#: Per-sequence ring normalization: ape add, mean, variance, scale, gate.
RING_NORM_OPS = 8

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
    # Step 2: refine readiness via mindie_sim.backends.abi_registry.ProbedAbiRegistry once W1 merges.
    arch = _geometry_arch_family(geometry)
    explicit = str(dict(getattr(op, "metadata", ()) or ()).get("readiness") or "").strip() if op is not None else ""
    if explicit and explicit in KERNEL_READINESS_VALUES:
        readiness = explicit
    elif arch == "A5" or (arch in {"A2", "A3"} and recipe_id in {"sparse_indexer", "index_scores"}):
        readiness = "shipping"
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


def _pool_wave_work(ledger, index):
    """Declared per-wave service of one 32-AIC window, in physical units."""
    rows = _wave_rows(int(ledger["query_rows"]), index, int(ledger["wave_rows"]))
    heads, width = int(ledger["heads"]), int(ledger["width"])
    dtype = str(ledger["score_dtype"])
    pairs = float(ledger["score_pairs"]) * float(rows) / float(
        max(1, int(ledger["query_rows"])))
    return {
        "rows": rows,
        "dtype": dtype,
        "stream": {
            "hbm": (float(ledger["wave_cache_bytes"])
                    + float(ledger["wave_descale_bytes"])
                    + float(rows) * heads * width),
            "mte2": (float(ledger["wave_cache_bytes"])
                     + float(ledger["wave_descale_bytes"])
                     + float(rows) * heads * width),
        },
        "tiles": {
            dtype: 2.0 * pairs * width,
            "mte1": pairs * FP32_ELEMENT_BYTES,
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
        # A shared-prefix plan replaces only the KV cache component of the HBM
        # fill with its physical union; the per-wave MTE2 refill service stays
        # complete.  The prepared plan must carry both components unchanged.
        prefix_hbm = metadata.get("prefix_read_hbm_bytes")
        if prefix_hbm is not None:
            kv_component = (float(ledger.get("waves", 0))
                            * float(ledger.get("wave_cache_bytes", 0.0)))
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
        owner_schedule = _indexer_owner_schedule(
            owner_kind="mix",
            tiling_strategy="pool_key_indexer_aic_waves",
            parallel_axis="query_rows",
            work_tiles=max(1, int(ledger.get("query_rows", 1) or 1)),
            total_owners=total_owners,
            read_bytes=float(work.resources.get("mte2", 0.0)),
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
        wave_hbm_cache = (
            float(prefix_hbm) / float(max(1, waves))
            if prefix_hbm is not None
            else float(ledger["wave_cache_bytes"])
        )
        handoff_ms = _primitive_ms(context, "P13_crosscore_sync", "mode2_paired_ns")
        floor = (
            _primitive_ms(context, "P15_rts_stars", "aic_first_issue_ns")
            + int(ledger["blocks"]) * _primitive_ms(context, "P13_crosscore_sync", "mode4_intra_block_ns")
        )

        def wave(index):
            parts = _pool_wave_work(ledger, index)
            stream = dict(parts["stream"])
            stream["hbm"] = (stream["hbm"] - float(ledger["wave_cache_bytes"])
                             + wave_hbm_cache)
            return (
                Stage("stream", context.bind(Work({}, stream))),
                Stage("tiles", context.bind(Work(
                    {parts["dtype"]: parts["tiles"][parts["dtype"]]},
                    _matrix_free(parts["tiles"], parts["dtype"]))),
                    after=("stream",)),
                Stage("handoff", after=("tiles",), latency=handoff_ms),
            )

        selection = _pool_selection_work(ledger)
        return KernelProgram((
            Stage("waves", program=PipelineProgram(waves, SERIAL_SLOTS, wave)),
            Stage("selection", context.bind(Work({}, dict(selection))),
                  after=("waves",), latency=floor),
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
    fragments = ranges[wave] if ranges is not None else ((0, start + valid),)
    unique = sum(max(0, min(hi, stop) - max(lo, start)) for lo, hi in fragments)
    query_heads = active_rows * heads if block == 0 else 0
    cube_input = (query_heads + unique) * width
    vector_input = (2 * query_heads + unique) * FP32_ELEMENT_BYTES
    return {
        "rows": rows,
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
            "vector": float(tile_pairs) * TOPK_MERGE_OPS,
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
        ledger = dict(dict(op.metadata).get(QUANT_LEDGER, {}))
        work = arch35_indexer_work(ledger)
        ub_cap = _geometry_ub_capacity(geometry)
        total_owners = int(
            getattr(geometry, "compute_units", 0)
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
        ledger = dict(dict(context.op.metadata).get(QUANT_LEDGER, {}))
        tiles = int(ledger["waves"]) * int(ledger["blocks"])
        floor = _primitive_ms(context, "P15_rts_stars", "aic_first_issue_ns")
        payload = int(ledger.get("payload_keys",
                                 ledger.get("completed_keys", ledger["keys"])))
        fill = float(payload) * int(ledger["key_row_bytes"])
        if tiles == 0:
            # No completed key owns a block: only the INT32 index store runs.
            stages = []
            if fill > 0.0:
                stages.append(Stage("cache_fill", context.bind(Work({}, {
                    "hbm": fill}))))
            stages.append(Stage("indices", context.bind(Work({}, {
                "mte3": float(ledger["output_bytes"]),
                "hbm": float(ledger["output_bytes"])})), latency=floor))
            return KernelProgram(tuple(stages))

        def tile(index):
            parts = _quant_tile_work(ledger, index)
            cube, reduce = parts["cube"], parts["reduce"]
            return (
                Stage("operands", context.bind(Work(
                    {"fp8_e4m3": cube["fp8_e4m3"]},
                    _matrix_free(cube, "fp8_e4m3")))),
                Stage("topk_merge", context.bind(Work({}, dict(reduce))),
                      after=("operands",)),
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
                after=("blocks",), latency=floor),
        ))


def _keypool_step_work(ledger):
    pool, width = int(ledger["pool_size"]), int(ledger["width"])
    ring = float(pool) * 2 * width * FP32_ELEMENT_BYTES
    return {
        "read": {"hbm": ring, "mte2_aiv": ring},
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


def key_pool_work(ledger):
    """Total declared work of the KeyPool producer program."""
    steps = int(ledger["steps"])
    step = _keypool_step_work(ledger)
    per_step = Work({}, _merged(step["read"], step["norm"], step["write"]))
    batch = int(ledger["batch"])
    total = per_step.scaled(float(batch) * steps)
    total = total + Work({"bf16": float(ledger["projection_flops"])}, {
        "hbm": float(ledger["hidden_bytes"]) + float(ledger["weight_bytes"]),
        "mte2": float(ledger["hidden_bytes"]) + float(ledger["weight_bytes"]),
    })
    commit = _keypool_commit_work(ledger)
    return total + Work({}, dict(commit))


def _keypool_commit_work(ledger):
    pools = int(ledger["completed_pools"])
    width = int(ledger["width"])
    return {
        "hbm": float(pools) * width * FP32_ELEMENT_BYTES,
        "mte3": float(pools) * width * FP32_ELEMENT_BYTES,
        "vector": float(pools) * int(ledger["pool_size"]) * width * RING_NORM_OPS,
    }


@dataclass(frozen=True)
class KeyPoolRingRecipe:
    """Class B: `aclnnKeyPool` two-stage AIC projection plus AIV state ring.

    Stage one runs the BF16 `wk`/`wgate` projections on all AIC owners and joins
    through a GM/L2 workspace barrier. Stage two gives every AIV a bounded
    sequence window; each sequence walks its `S` verify tokens serially, because
    token `i+1` reads the ring that token `i` wrote, and every completed pool
    then runs its FP32 reduction and cache commit on the same owner.
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
        work = key_pool_work(ledger)
        ub_cap = _geometry_ub_capacity(geometry)
        aiv_total = int((getattr(geometry, "resource_units", {}) or {}).get("vector", INDEXER_TILE_CAP) or INDEXER_TILE_CAP) if geometry is not None else INDEXER_TILE_CAP
        owner_schedule = _indexer_owner_schedule(
            owner_kind="mix",
            tiling_strategy="cann_key_pool_aic_projection_then_aiv_ring",
            parallel_axis="sequence_ring",
            work_tiles=max(1, int(ledger.get("batch", 1) or 1)),
            total_owners=aiv_total,
            read_bytes=float(work.resources.get("mte2", 0.0)) + float(work.resources.get("mte2_aiv", 0.0)),
            write_bytes=float(work.resources.get("mte3", 0.0)),
            score_flops=float(sum(work.matrix.values())),
            topk_ops=float(work.resources.get("vector", 0.0)),
            ub_capacity_bytes=ub_cap,
            dependent_steps=max(1, int(ledger.get("steps", 1) or 1)),
        )
        return OperatorPlan(work, self.lower, metadata={
            "schedule": "cann_key_pool_aic_projection_then_aiv_ring",
            "native_dispatch_reconstructed": False,
            "completed_pools": ledger.get("completed_pools"),
            "ring_steps": ledger.get("steps"),
            "owner_schedule": owner_schedule.to_dict(),
            "kernel_binding": _indexer_kernel_binding(
                "aclnnKeyPool", str(op.recipe or "cann_key_pool"), geometry=geometry, op=op,
            ).to_dict(),
        })

    def lower(self, context):
        ledger = dict(dict(context.op.metadata).get(KEYPOOL_LEDGER, {}))
        steps = int(ledger["steps"])
        owners = int(context.resource_units.get("vector", context.core_count))
        batch = int(ledger["batch"])
        step = _keypool_step_work(ledger)
        commit_total = _keypool_commit_work(ledger)
        # `completed_pools` is the batch total; each owned sequence commits its
        # own share of the completed pools inside its own ring.
        commit = {name: value / float(max(1, batch))
                  for name, value in commit_total.items()}
        pools = int(ledger["completed_pools"])
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

        def sequence(owner_index):
            stages = [Stage("steps", program=PipelineProgram(
                steps, SERIAL_SLOTS,
                lambda index, owner=owner_index: ring_step(owner)))]
            if pools:
                owned = context.owned(owner_index, owners,
                                      ("vector", "mte2_aiv", "mte3"))
                stages.append(Stage("pool_commit", context.bind(Work(
                    {}, dict(commit))), after=("steps",), placement=owned,
                    latency=pool_commit_ms))
            return (Stage("sequence", program=KernelProgram(tuple(stages))),)

        def owner_program(owner_index):
            count = (batch - 1 - owner_index) // owners + 1
            if count <= 0:
                return KernelProgram(())
            return KernelProgram((Stage("sequences", program=PipelineProgram(
                count, SERIAL_SLOTS,
                lambda index, owner=owner_index: sequence(owner))),))

        return KernelProgram((
            Stage("projection", context.bind(Work(
                {"bf16": float(ledger["projection_flops"])},
                {"hbm": float(ledger["hidden_bytes"]) + float(ledger["weight_bytes"]),
                 "mte2": (float(ledger["hidden_bytes"])
                          + float(ledger["weight_bytes"]))}))),
            Stage("workspace_join", after=("projection",),
                  latency=workspace_join_ms),
            Stage("ring", after=("workspace_join",), program=ParallelProgram(
                tuple(owner_program(owner) for owner in range(owners)))),
        ))
