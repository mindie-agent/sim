"""Explicit two-launch sparse scoring and top-k programs.

[Role] Export source operators and execute their physical owner programs.
[Invariants] Geometry is normalized separately; launch boundaries, exact work,
  owner partitions, finite buffers and input dependencies remain explicit.
  Unsupported source branches are never replaced by another implementation.
[Timeouts & Locks] Lazy finite programs keep only active tasks; no waits or locks.
[Side-Effect Policy] Return records and submit to the caller Engine; no device I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from mindie_sim.schema.common import align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelineProgram, Stage,
)

from .index_topk_geometry import (
    IMPLEMENTATION, SOURCE_PATH, SOURCE_SHA256, SOURCE_BYTES,
    SUPPLEMENTARY_SOURCES, SPARSE_BLOCK, SCORE_Q_TILE, BLOCK_SIZE_Q,
    TOPK_TILES, SCORE_KERNEL, TOPK_KERNEL, SCORE_PHASE,
    TOPK_PHASE, SCORE_MASK_OPS, SCORE_MAX_OPS, TOPK_CAUSAL_OPS,
    TOPK_NAN_OPS, TOPK_INIT_OPS, TOPK_LOCAL_OPS, TOPK_INDEX_OPS,
    TOPK_COPY_OPS, TOPK_PREPARE_OPS, CAS_OPS, MERGE_COMBINE_OPS,
    INIT_FILL_OPS, EXTRACT_MUL_OPS, EXTRACT_ADD_OPS, STORE_ADDR_OPS,
    _ceil_div, _next_power_of_two, _int, _flag,
    _IndexTopk, _normalize, _floor_sum, _forced_candidates,
    _score_tile_stats, _topk_token_stats, _counts, _dispatch,
)

def make_op(params: dict) -> OpSpec:
    """Build the immutable source dispatch for one campaign geometry."""
    g = _normalize(params)
    counts = _counts(g)
    geometry = {
        "implementation": g.implementation,
        "query": g.query,
        "context": g.context,
        "batch": g.batch,
        "heads": g.heads,
        "head_dim": g.head_dim,
        "topk": g.topk,
        "block_size_q": g.block_size_q,
        "score_block": g.score_block,
        "topk_block": g.topk_block,
        "init_blocks": g.init_blocks,
        "local_blocks": g.local_blocks,
        "dtype": g.dtype,
        "mask_init": g.mask_init,
        "mask_local": g.mask_local,
    }
    metadata = (
        ("dispatch", _dispatch(g)),
        ("geometry", geometry),
        ("work_components", counts),
        ("source", {
            "path": SOURCE_PATH,
            "sha256": SOURCE_SHA256,
            "bytes": SOURCE_BYTES,
            "variant": "common",
            "source_origin": "current_checkout",
            "historical_capture_image_supplied": False,
            "lines": {
                "score_kernel": "81-163",
                "topk_kernel": "185-284",
                "score_wrapper": "647-703",
                "topk_wrapper": "706-756",
                "wrapper_mask_constexpr": "753-754",
                "decode_path_not_selected": "287-641",
            },
            "supplementary": SUPPLEMENTARY_SOURCES,
        }),
        ("buffers", {
            "score_workspace_bytes": counts["score_workspace_bytes"],
            "score_stride": counts["score_stride"],
            "selected_index_bytes": counts["selected_index_bytes"],
            "onchip_score_tile_bytes": counts["onchip_score_tile_bytes"],
            "onchip_sort_bytes_per_program": counts["onchip_sort_bytes_per_program"],
        }),
        ("launch_boundaries", {
            "score": {
                "kernel": SCORE_KERNEL,
                "phase": SCORE_PHASE,
                "programs": counts["score_programs"],
                "stages": ["q_load", "q_addr", "page_load", "k_load",
                           "qk_dot", "mask_max", "score_store"],
                "per_block": ["page_load", "k_load", "qk_dot",
                              "mask_max", "score_store"],
                "dependencies": {
                    "q_addr": ["q_load"],
                    "page_load": ["q_addr"],
                    "k_load": ["page_load"],
                    "qk_dot": ["k_load"],
                    "mask_max": ["qk_dot"],
                    "score_store": ["mask_max"],
                },
                "per_block_carry": (
                    "next page_load follows the previous score_store at slots=1"
                ),
            },
            "topk": {
                "kernel": TOPK_KERNEL,
                "phase": TOPK_PHASE,
                "programs": counts["topk_programs"],
                "stages": ["init", "load", "masks", "bitonic_merge_scan",
                           "bitonic_merge_desc", "bitonic_merge_combine",
                           "bitonic_merge_asc", "extract", "output_store"],
                "per_chunk": ["load", "masks", "bitonic_merge_scan",
                              "bitonic_merge_desc", "bitonic_merge_combine",
                              "bitonic_merge_asc"],
                "dependencies": {
                    "load": ["init"],
                    "masks": ["load"],
                    "bitonic_merge_scan": ["masks"],
                    "bitonic_merge_desc": ["bitonic_merge_scan"],
                    "bitonic_merge_combine": ["bitonic_merge_desc"],
                    "bitonic_merge_asc": ["bitonic_merge_combine"],
                    "extract": ["bitonic_merge_asc"],
                    "output_store": ["extract"],
                },
                "per_chunk_carry": (
                    "previous bitonic_merge_asc carries topk_score/topk_idx "
                    "into the next chunk's merge"
                ),
                "after": [SCORE_PHASE],
            },
        }),
        ("source_cadence", {
            "score": (
                "q_load -> (page_load -> k_load -> qk_dot -> mask_max -> "
                "score_store) per visible block"
            ),
            "topk": (
                "init -> (load -> masks -> bitonic_merge_scan [-> "
                "bitonic_merge_desc -> bitonic_merge_combine -> "
                "bitonic_merge_asc]) per chunk -> extract -> output_store"
            ),
            "score_load_dot_separate_stage": True,
            "topk_merge_carried_per_chunk": True,
            "aggregate_read_matrix_envelope": False,
        }),
        ("launch_contract", {
            "exported_kernels": [SCORE_PHASE, TOPK_PHASE],
            "host_submissions_eager": 2,
            "host_submissions_captured": 1,
            "metadata_alone_changes_execution": False,
        }),
    )
    return OpSpec(
        op_id="index_topk",
        family="index_topk",
        phase="prefill",
        stage=0,
        layer=0,
        requests=("batch",),
        semantic_shape=(g.batch, g.query, g.heads, g.head_dim),
        local_shape=(g.total_q, g.heads, g.head_dim),
        padded_shape=(g.batch * g.score_tiles * SCORE_Q_TILE, g.heads, g.head_dim),
        launch_shape=(counts["score_programs"], counts["topk_programs"]),
        recipe=IMPLEMENTATION, source_boundary="call",
        coverage="decomposed",
        dtype=g.dtype,
        layout="paged_kv",
        metadata=metadata,
        notes=(
            "source-derived common Triton prefill pair: block-score launch then "
            "bitonic top-k launch",
            "topk_block is the case-declared explicit autotune tile, not an "
            "autotune winner",
            "production semantics are the wrapper's MASK_INIT=False/"
            "MASK_LOCAL=False; raw true branches are recorded but not selected "
            "by the wrapper",
            "source CTA cadence: page/K load and its dependent QK dot are "
            "separate stages; the score loop emits per-block history and the "
            "top-k loop emits per-chunk carried merges",
            "current checkout source, not a historical capture image; no "
            "matching capture observation was found in the read-only evidence "
            "paths",
        ),
    )


def kernels(op: OpSpec) -> tuple[OpSpec, ...]:
    """Export the two real host launch boundaries as distinct operators.

    Children use local ids and ``kernels``-local dependencies; ``expand_call``
    prefixes those ids and prepends the caller's dependencies.  Both children
    keep the same recipe id, so ``build`` receives the phase axis in metadata
    rather than selecting by op id or model identity.
    """
    meta = dict(op.metadata)
    dispatch = _require(meta, "dispatch")
    if dispatch.get("implementation") != IMPLEMENTATION:
        raise ValueError(
            f"unsupported_index_topk_implementation:{dispatch.get('implementation')}"
        )
    counts = _require(meta, "work_components")

    def child(phase: str, deps: tuple[str, ...]) -> OpSpec:
        child_meta = dict(op.metadata)
        child_meta["kernel_phase"] = phase
        child_meta["kernel"] = (
            SCORE_KERNEL if phase == SCORE_PHASE else TOPK_KERNEL
        )
        return replace(
            op,
            op_id=phase,
            deps=deps,
            source_boundary="kernel",
            launch_shape=(counts["score_programs"],) if phase == SCORE_PHASE
            else (counts["topk_programs"],),
            metadata=tuple(child_meta.items()),
            notes=(
                f"phase={phase}; recipe={IMPLEMENTATION}; "
                "selected by OpSpec.metadata['kernel_phase']",
            ),
        )

    return (
        child(SCORE_PHASE, ()),
        child(TOPK_PHASE, (SCORE_PHASE,)),
    )


def _require(source: Mapping[str, Any], key: str) -> Any:
    if key not in source:
        raise ValueError(f"index_topk_metadata_missing:{key}")
    return source[key]


def _phase_work(g: _IndexTopk, counts: Mapping[str, Any]) -> tuple[Work, Work]:
    """Return (score launch work, top-k launch work) in declared physical units."""
    score = Work(
        matrix={g.dtype: counts["score_qk_flops"]},
        resources={
            "hbm": (counts["score_q_bytes"] + counts["score_k_bytes"]
                    + counts["score_block_table_bytes"]
                    + counts["score_store_bytes"]),
            "mte2": (counts["score_q_bytes"] + counts["score_k_bytes"]
                     + counts["score_block_table_bytes"]),
            "mte3": counts["score_store_bytes"],
            "vector": counts["score_vector_ops"],
        },
    )
    topk_vector = (
        counts["topk_prepare_ops"] + counts["topk_sort_ops"]
        + counts["topk_merge_ops"] + counts["topk_init_ops"]
        + counts["topk_extract_ops"]
    )
    topk = Work(resources={
        "hbm": counts["score_read_bytes"] + counts["selected_index_bytes"],
        "mte2": counts["score_read_bytes"],
        "mte3": counts["selected_index_bytes"],
        "vector": topk_vector,
    })
    return score, topk


def _owned(context, core: int, roles: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Pin a CTA to one physical partition; owners is the full device width."""
    return context.owned(core, context.core_count, roles)


def _score_cta_stages(g: _IndexTopk, context, core: int, cta: int) -> tuple[Stage, ...]:
    """One source score CTA: Q load, then a lazy per-page K->dot->store loop.

    The source loop (lines 135-163) loads the block-table page, loads that K
    page, consumes it in ``tl.dot``, masks/reduces and stores.  Those are
    dependent operations: the dot cannot start before the page's K bytes have
    arrived.  A single all-read/all-matrix stage would let the engine overlap
    them and undercharge a one-block CTA.
    """
    tile = cta % g.score_tiles
    stats = _score_tile_stats(g, tile)
    rows = stats["rows"]
    width = dtype_bytes(g.dtype)
    q_start = g.context + tile * SCORE_Q_TILE
    first_masked = q_start // g.score_block
    qk_per_block = 2 * SCORE_Q_TILE * g.score_block * g.head_dim
    k_per_block = g.score_block * g.head_dim * width
    mask_per_block = SCORE_MASK_OPS * SCORE_Q_TILE * g.score_block
    max_per_block = SCORE_MAX_OPS * SCORE_Q_TILE * (g.score_block - 1)
    addr_per_block = g.score_block

    q_load = Work(resources={
        "hbm": stats["q_bytes"], "mte2": stats["q_bytes"],
    })
    q_addr = Work(resources={"vector": SCORE_Q_TILE})

    def page(index: int) -> tuple[Stage, ...]:
        page_load = Work(resources={"hbm": 4, "mte2": 4})
        k_load = Work(resources={"hbm": k_per_block, "mte2": k_per_block})
        dot = Work(matrix={g.dtype: qk_per_block})
        mask = Work(resources={"vector": (
            (mask_per_block if index >= first_masked else 0)
            + max_per_block + addr_per_block
        )})
        store = Work(resources={"hbm": rows * 4, "mte3": rows * 4})
        return (
            Stage("page_load", context.bind(page_load),
                  placement=_owned(context, core, ("mte2",))),
            Stage("k_load", context.bind(k_load), after=("page_load",),
                  placement=_owned(context, core, ("mte2",))),
            Stage("qk_dot", context.bind(dot), after=("k_load",),
                  placement=_owned(context, core, ("cube",))),
            Stage("mask_max", context.bind(mask), after=("qk_dot",),
                  placement=_owned(context, core, ("vector",))),
            Stage("score_store", context.bind(store), after=("mask_max",),
                  placement=_owned(context, core, ("mte3",))),
        )

    blocks = PipelineProgram(stats["blocks"], 1, page)
    return (
        Stage("q_load", context.bind(q_load),
              placement=_owned(context, core, ("mte2",))),
        Stage("q_addr", context.bind(q_addr), after=("q_load",),
              placement=_owned(context, core, ("vector",))),
        Stage("block_loop", after=("q_addr",), program=blocks,
              placement=_owned(context, core, ("cube", "vector", "mte2", "mte3"))),
    )


def _topk_cta_stages(g: _IndexTopk, context, core: int, cta: int) -> tuple[Stage, ...]:
    """One source top-k CTA: sentinel fill, per-chunk merge, extraction.

    The source loop (lines 229-264) carries ``topk_score``/``topk_idx`` from
    one ``BLOCK_SIZE_K`` chunk to the next.  Every chunk after the first
    combines the previous top half with the current sorted tile, so the
    per-chunk merge is a real serial edge rather than one aggregate bitonic
    envelope.
    """
    token = cta % g.query
    stats = _topk_token_stats(g, token)
    tile = g.topk_block
    n = g.log2_tile
    scan_stages = n * (n - 1) // 2
    merge_stages = n
    visible = stats["visible"]

    def chunk(index: int) -> tuple[Stage, ...]:
        start = index * tile
        valid = min(tile, visible - start)
        load = Work(resources={"hbm": valid * 4, "mte2": valid * 4})
        masks = Work(resources={"vector": TOPK_PREPARE_OPS * tile})
        scan = Work(resources={"vector": CAS_OPS * tile * scan_stages})
        merge = Work(resources={"vector": CAS_OPS * tile * merge_stages})
        stages = [
            Stage("load", context.bind(load),
                  placement=_owned(context, core, ("mte2",))),
            Stage("masks", context.bind(masks), after=("load",),
                  placement=_owned(context, core, ("vector",))),
            Stage("bitonic_merge_scan", context.bind(scan), after=("masks",),
                  placement=_owned(context, core, ("vector",))),
        ]
        if index:
            # Later iteration: descending merge of the current tile, combine
            # its top half with the previous iteration's carried state, then
            # the ascending merge.  These are distinct source line regions
            # (252-264); the combine cannot precede the descending merge.
            combine = Work(resources={"vector": MERGE_COMBINE_OPS * tile})
            stages.append(Stage("bitonic_merge_desc", context.bind(merge),
                                after=("bitonic_merge_scan",),
                                placement=_owned(context, core, ("vector",))))
            stages.append(Stage("bitonic_merge_combine", context.bind(combine),
                                after=("bitonic_merge_desc",),
                                placement=_owned(context, core, ("vector",))))
            stages.append(Stage("bitonic_merge_asc", context.bind(merge),
                                after=("bitonic_merge_combine",),
                                placement=_owned(context, core, ("vector",))))
        else:
            stages.append(Stage("bitonic_merge_asc", context.bind(merge),
                                after=("bitonic_merge_scan",),
                                placement=_owned(context, core, ("vector",))))
        return tuple(stages)

    chunks = PipelineProgram(stats["iterations"], 1, chunk)
    init = Work(resources={"vector": stats["init_ops"]})
    extract = Work(resources={"vector": stats["extract_ops"]})
    output = Work(resources={
        "hbm": stats["selected_index_bytes"],
        "mte3": stats["selected_index_bytes"],
    })
    return (
        Stage("init", context.bind(init),
              placement=_owned(context, core, ("vector",))),
        Stage("chunks", after=("init",), program=chunks,
              placement=_owned(context, core, ("vector", "mte2"))),
        Stage("extract", context.bind(extract), after=("chunks",),
              placement=_owned(context, core, ("vector",))),
        Stage("output_store", context.bind(output), after=("extract",),
              placement=_owned(context, core, ("mte3",))),
    )


def _unit_streams(total: int, context, builder) -> ParallelProgram:
    """One lazy CTA stream per participating physical unit.

    CTA linear indices follow the source grid flattening (query/tile dimension
    first).  Each stream owns one physical partition and issues its next CTA
    only after the previous CTA completes, so a singleton CTA can never borrow
    idle per-SM capacity.  ``PipelineProgram`` keeps only the live CTA task and
    a finite count, not a list of all grid CTAs.
    """
    units = min(total, context.core_count)
    if units < 1:
        raise ValueError("index_topk_cta_stream_requires_a_physical_unit")
    programs = []
    for core in range(units):
        count = _ceil_div(total - core, units)
        programs.append(PipelineProgram(
            count, 1,
            lambda index, core=core: builder(core, index * units + core),
        ))
    return ParallelProgram(tuple(programs))


def _launch(program: ParallelProgram) -> KernelProgram:
    return KernelProgram((Stage("cta_grid", program=program),))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Declare the selected score and/or top-k launch with CTA ownership."""
    if geometry.compute_units < 1:
        raise ValueError("index_topk_compute_units_must_be_positive")
    meta = dict(op.metadata)
    dispatch = _require(meta, "dispatch")
    raw_geometry = _require(meta, "geometry")
    declared_counts = _require(meta, "work_components")
    if dispatch.get("implementation") != IMPLEMENTATION:
        raise ValueError(
            f"unsupported_index_topk_implementation:{dispatch.get('implementation')}"
        )
    g = _normalize(raw_geometry)
    counts = _counts(g)
    if counts != declared_counts:
        raise ValueError("index_topk_metadata_counts_inconsistent")

    phase = meta.get("kernel_phase")
    if phase is not None and phase not in (SCORE_PHASE, TOPK_PHASE):
        raise ValueError(f"unsupported_index_topk_kernel_phase:{phase}")

    score_work, topk_work = _phase_work(g, counts)
    if phase == SCORE_PHASE:
        total = score_work
    elif phase == TOPK_PHASE:
        total = topk_work
    else:
        total = score_work + topk_work

    def program(context):
        score_launch = Stage(
            "score_launch",
            program=_launch(_unit_streams(
                counts["score_programs"], context,
                lambda core, cta: _score_cta_stages(g, context, core, cta),
            )),
        )
        if phase == SCORE_PHASE:
            return KernelProgram((score_launch,))
        topk_launch = Stage(
            "topk_launch",
            program=_launch(_unit_streams(
                counts["topk_programs"], context,
                lambda core, cta: _topk_cta_stages(g, context, core, cta),
            )),
        )
        if phase == TOPK_PHASE:
            return KernelProgram((topk_launch,))
        return KernelProgram((score_launch, replace(topk_launch, after=("score_launch",))))

    return OperatorPlan(total, program)


__all__ = ["build", "kernels", "make_op"]
