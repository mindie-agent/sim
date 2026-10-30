"""Triton MLA source launch boundaries and finite split-KV programs.

[Role] Export source operators and execute their physical owner programs.
[Invariants] Geometry is normalized separately; launch boundaries, exact work,
  owner partitions, finite buffers and input dependencies remain explicit.
  Unsupported source branches are never replaced by another implementation.
[Timeouts & Locks] Lazy finite programs keep only active tasks; no waits or locks.
[Side-Effect Policy] Return records and submit to the caller Engine; no device I/O.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.schema.operators import OpSpec

from .mla_decode_geometry import (
    IMPLEMENTATION, SOURCE_FILES, _BLOCK_H, _BLOCK_N,
    _STAGE1_NUM_WARPS, _STAGE2_NUM_WARPS, _STAGE2_NUM_STAGES, _CTA_STREAM_SLOTS,
    _TILE_PIPELINE_SLOTS, _MERGE_PIPELINE_SLOTS, _SPECIAL_HEAD_DIMS, _DTYPE_ALIASES,
    _INDEX_BYTES, _PROGRAM_BYTES, _ACCUM_BYTES, _BOUNDARY,
    _KERNEL_PHASES, _required, _integer, _dtype, _split_summary,
    _cdiv, _validate_contexts, _select,
)

def make_op(params: Mapping[str, Any]) -> OpSpec:
    """Build the explicit geometry adapter for one campaign case."""
    if not isinstance(params, Mapping):
        raise ValueError("mla_decode_params_must_be_mapping")
    selected = _select(params)
    dispatch = selected["dispatch"]
    plan = selected["plan"]
    seq_lens = selected["seq_lens"]

    batch = plan["batch"]
    heads = plan["heads"]
    qk_dim = plan["qk_dim"]
    splits = plan["num_kv_splits"]
    block_dv = plan["block_dv"]
    head_blocks = plan["head_blocks"]
    padded_heads = plan["padded_heads"]

    scratch_elems = batch * heads * splits * (block_dv + 1)
    metadata = (
        ("dispatch", dispatch),
        ("plan", plan),
        ("seq_lens", seq_lens),
        ("boundary", _BOUNDARY),
        ("source", SOURCE_FILES),
        ("source_kernel_launches", 2),
        ("kernels", ("stage1_grouped_decode", "stage2_split_merge")),
        ("scratch", {
            "shape": (batch, heads, splits, block_dv + 1),
            "dtype": "fp32",
            "reserved_bytes": scratch_elems * _ACCUM_BYTES,
            "lifetime": "stage1_write -> stage2_read",
        }),
        ("outputs", {
            "o_shape": (batch, heads, block_dv),
            "o_dtype": "bf16",
            "lse_shape": (batch, heads),
            "lse_dtype": "bf16",
        }),
        ("launches", (
            {
                "name": "stage1_grouped_decode",
                "kernel": "_fwd_grouped_kernel_stage1",
                "grid": (batch, head_blocks, splits),
                "block": {
                    "block_h": plan["block_h"],
                    "block_n": plan["block_n"],
                    "block_dmodel": plan["block_dmodel"],
                    "block_dpe": plan["block_dpe"],
                    "block_dv": plan["block_dv"],
                },
                "num_warps": plan["stage1_num_warps"],
                "num_stages": plan["stage1_num_stages"],
                "after": (),
            },
            {
                "name": "stage2_split_merge",
                "kernel": "_fwd_kernel_stage2",
                "grid": (batch, heads),
                "block": {"block_dv": plan["block_dv"]},
                "num_warps": plan["stage2_num_warps"],
                "num_stages": plan["stage2_num_stages"],
                "after": ("stage1_grouped_decode",),
            },
        )),
    )

    return OpSpec(
        op_id="mla_decode",
        family="mla_decode",
        phase="decode",
        stage=0,
        layer=0,
        requests=tuple(f"seq{index}" for index in range(batch)),
        semantic_shape=(batch, heads, qk_dim),
        local_shape=(batch, heads, qk_dim),
        padded_shape=(batch, padded_heads, qk_dim),
        launch_shape=plan["stage1_grid"],
        recipe=IMPLEMENTATION, source_boundary="call",
        dtype="bf16",
        metadata=metadata,
    )


def kernels(op: OpSpec) -> tuple[OpSpec, OpSpec]:
    """Split the source wrapper's two unconditional launches into children.

    Child ids are local; ``expand_call`` prefixes them and rewrites the local
    dependency.  Both children keep the recipe id so the coordinator's
    ``PlannedRecipe(build, kernels)`` binding prepares each one explicitly.
    """
    metadata = dict(op.metadata)
    plan = metadata.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("mla_decode_plan_metadata_required")
    stage1_metadata = {
        **metadata,
        "kernel_phase": "stage1",
        "kernel": "_fwd_grouped_kernel_stage1",
        "launch_grid": tuple(plan["stage1_grid"]),
        "launch_count": 1,
    }
    stage2_metadata = {
        **metadata,
        "kernel_phase": "stage2",
        "kernel": "_fwd_kernel_stage2",
        "launch_grid": tuple(plan["stage2_grid"]),
        "launch_count": 1,
        "split_reduction_owner": "batch_head",
    }
    return (
        replace(
            op,
            op_id="stage1_grouped_decode",
            source_boundary="kernel",
            deps=(),
            launch_shape=tuple(plan["stage1_grid"]),
            metadata=tuple(stage1_metadata.items()),
        ),
        replace(
            op,
            op_id="stage2_split_merge",
            source_boundary="kernel",
            deps=("stage1_grouped_decode",),
            launch_shape=tuple(plan["stage2_grid"]),
            metadata=tuple(stage2_metadata.items()),
        ),
    )




def _read_plan(op: OpSpec) -> tuple[dict[str, Any], tuple[int, ...]]:
    metadata = dict(op.metadata)
    plan = metadata.get("plan")
    raw_lengths = metadata.get("seq_lens")
    if not isinstance(plan, Mapping) or not isinstance(raw_lengths, (list, tuple)):
        raise ValueError("mla_decode_plan_metadata_required")
    seq_lens = tuple(_integer(value, "seq_lens[]", minimum=1) for value in raw_lengths)
    dispatch = dict(metadata.get("dispatch", {}))
    if plan.get("num_kv_splits") != dispatch.get("num_kv_splits"):
        raise ValueError("mla_decode_dispatch_split_count_mismatch")
    if plan.get("batch") != len(seq_lens):
        raise ValueError("mla_decode_plan_batch_mismatch")
    if plan.get("qk_dim") != plan.get("block_dmodel", 0) + plan.get("block_dpe", 0):
        raise ValueError("mla_decode_plan_qk_tile_mismatch")
    if plan.get("block_dv") != plan.get("latent_dim"):
        raise ValueError("mla_decode_plan_latent_tile_mismatch")
    if plan.get("block_h") != _BLOCK_H or plan.get("block_n") != _BLOCK_N:
        raise ValueError("mla_decode_plan_block_dispatch_mismatch")
    if plan.get("stage1_grid") != (
        len(seq_lens), plan.get("head_blocks"), plan.get("num_kv_splits")
    ):
        raise ValueError("mla_decode_plan_stage1_grid_mismatch")
    if plan.get("stage2_grid") != (len(seq_lens), plan.get("heads")):
        raise ValueError("mla_decode_plan_stage2_grid_mismatch")
    return dict(plan), seq_lens


def _stage1_totals(plan: Mapping[str, Any], seq_lens: tuple[int, ...]) -> Work:
    """Closed-form stage-1 work: a conservation oracle for the CTA programs."""
    heads = plan["heads"]
    splits = plan["num_kv_splits"]
    block_h = plan["block_h"]
    block_n = plan["block_n"]
    block_dmodel = plan["block_dmodel"]
    block_dpe = plan["block_dpe"]
    block_dv = plan["block_dv"]
    qk_dim = plan["qk_dim"]
    head_blocks = plan["head_blocks"]

    reads = writes = vector = flops = 0
    for length in seq_lens:
        _, nonempty, tiles = _split_summary(length, splits, block_n)
        # Q is loaded by every launched split program, including empty splits.
        reads += splits * heads * qk_dim * _PROGRAM_BYTES
        # B_Seqlen is loaded by every launched program.
        reads += head_blocks * splits * _INDEX_BYTES
        # k_scale/v_scale are loaded once in each non-empty split branch.
        reads += head_blocks * nonempty * 2 * _ACCUM_BYTES
        # One page-id lane, one latent K row and one PE K row per key, per
        # head block.  PV reuses the latent K load (no V_Buffer read).
        reads += head_blocks * length * _INDEX_BYTES
        reads += head_blocks * length * (block_dmodel + block_dpe) * _PROGRAM_BYTES
        # Every non-empty split stores FP32 partial output and FP32 LSE.
        writes += nonempty * heads * (block_dv * _ACCUM_BYTES + _ACCUM_BYTES)
        vector += head_blocks * (
            tiles * _tile_vector_ops(block_h, block_n, block_dv)
            + nonempty * _epilogue_vector_ops(block_h, block_dv)
        )
        flops += head_blocks * tiles * _tile_flops(
            block_h, block_n, block_dmodel, block_dpe, block_dv
        )
    return Work(
        matrix={"bf16": flops},
        resources={
            "hbm": reads + writes,
            "mte2": reads,
            "mte3": writes,
            "vector": vector,
        },
    )


def _stage2_totals(plan: Mapping[str, Any], seq_lens: tuple[int, ...]) -> Work:
    """Closed-form stage-2 merge work: the per-head CTA conservation oracle."""
    heads = plan["heads"]
    block_dv = plan["block_dv"]
    block_n = plan["block_n"]
    splits = plan["num_kv_splits"]

    reads = writes = vector = 0
    for length in seq_lens:
        _, nonempty, _ = _split_summary(length, splits, block_n)
        # Every head program loads B_Seqlen.
        reads += heads * _INDEX_BYTES
        # Only non-empty split slots are loaded and merged.
        reads += heads * nonempty * (block_dv * _ACCUM_BYTES + _ACCUM_BYTES)
        writes += heads * (block_dv * _PROGRAM_BYTES + _PROGRAM_BYTES)
        vector += heads * (
            nonempty * _merge_vector_ops(block_dv) + _final_vector_ops(block_dv)
        )
    return Work(
        resources={
            "hbm": reads + writes,
            "mte2": reads,
            "mte3": writes,
            "vector": vector,
        },
    )


def _tile_flops(
    block_h: int, block_n: int, block_dmodel: int, block_dpe: int, block_dv: int
) -> int:
    # Both QK dots and the PV dot are 2*M*N*K and execute full tiles.
    return 2 * block_h * block_n * (block_dmodel + block_dpe + block_dv)


def _tile_vector_ops(block_h: int, block_n: int, block_dv: int) -> int:
    # scale, mask, row-max, p sub/exp/cast, sum(p): 7*Bh*Bn.
    # acc rescale: Bh*Dv.  max/re-scale/e_sum update: 5*Bh.
    return 7 * block_h * block_n + block_h * block_dv + 5 * block_h


def _epilogue_vector_ops(block_h: int, block_dv: int) -> int:
    # acc/e_sum divide and LSE add/log over the padded BLOCK_H rows.
    return block_h * block_dv + 2 * block_h


def _merge_vector_ops(block_dv: int) -> int:
    return 3 * block_dv + 7


def _final_vector_ops(block_dv: int) -> int:
    return 2 * block_dv + 3


def _placement(context: RecipeContext, core: int) -> dict[str, tuple[int, ...]]:
    """Pin one CTA's arithmetic and copy service to one physical owner.

    ``hbm`` is global memory and is deliberately not pinned.  Vector/copy
    units are mapped from the matrix owner by the shared ``unit_group`` rule;
    incompatible declared geometry raises rather than silently sharing.
    """
    return context.owned(
        core, context.core_count, ("cube", "vector", "mte2", "mte3")
    )


def _stage1_cta(
    context: RecipeContext,
    plan: Mapping[str, Any],
    length: int,
    head_block: int,
    split_id: int,
    core: int,
) -> KernelProgram:
    """One source stage-1 CTA: Q load, ordered key tiles, scratch store."""
    block_h = plan["block_h"]
    block_n = plan["block_n"]
    block_dmodel = plan["block_dmodel"]
    block_dpe = plan["block_dpe"]
    block_dv = plan["block_dv"]
    qk_dim = plan["qk_dim"]
    heads = plan["heads"]
    splits = plan["num_kv_splits"]

    valid_h = min(block_h, heads - head_block * block_h)
    chunk = _cdiv(length, splits)
    split_start = split_id * chunk
    split_end = min(split_start + chunk, length)
    span = split_end - split_start
    place = _placement(context, core)

    # The source loads Q and B_Seqlen before the empty-split guard (lines
    # 329-355), so empty splits keep this traffic but skip K/PV/scratch.
    q_bytes = valid_h * qk_dim * _PROGRAM_BYTES + _INDEX_BYTES
    stages: list[Stage] = [
        Stage("q_load", context.bind(Work(
            resources={"hbm": q_bytes, "mte2": q_bytes}
        )), placement=place),
    ]
    if span <= 0:
        return KernelProgram(tuple(stages))

    def tile(index: int):
        start_n = split_start + index * block_n
        valid_n = min(block_n, split_end - start_n)
        # First tile also loads k_scale/v_scale inside the non-empty branch.
        load_bytes = valid_n * (_INDEX_BYTES + qk_dim * _PROGRAM_BYTES)
        if index == 0:
            load_bytes += 2 * _ACCUM_BYTES
        load = Work(resources={"hbm": load_bytes, "mte2": load_bytes})
        qk = Work(
            matrix={"bf16": 2 * block_h * block_n * (block_dmodel + block_dpe)},
            resources={"vector": 2 * block_h * block_n},
        )
        softmax = Work(resources={"vector": 4 * block_h * block_n + 2 * block_h})
        pv = Work(
            matrix={"bf16": 2 * block_h * block_n * block_dv},
            resources={
                "vector": block_h * block_n + block_h * block_dv + 3 * block_h
            },
        )
        return (
            Stage("k_load", context.bind(load), placement=place),
            Stage("qk", context.bind(qk), after=("k_load",), placement=place),
            Stage("softmax", context.bind(softmax), after=("qk",),
                  placement=place),
            Stage("pv", context.bind(pv), after=("softmax",), placement=place),
        )

    tiles = _cdiv(span, block_n)
    stages.append(Stage(
        "key_tiles",
        after=("q_load",),
        program=PipelineProgram(tiles, _TILE_PIPELINE_SLOTS, tile),
    ))
    store_bytes = valid_h * (block_dv * _ACCUM_BYTES + _ACCUM_BYTES)
    stages.append(Stage(
        "scratch_store",
        after=("key_tiles",),
        work=context.bind(Work(resources={
            "hbm": store_bytes,
            "mte3": store_bytes,
            "vector": _epilogue_vector_ops(block_h, block_dv),
        })),
        placement=place,
    ))
    return KernelProgram(tuple(stages))


def _stage2_cta(
    context: RecipeContext,
    plan: Mapping[str, Any],
    length: int,
    core: int,
) -> KernelProgram:
    """One source stage-2 CTA: this batch/head reads only non-empty slots."""
    block_dv = plan["block_dv"]
    place = _placement(context, core)
    stages: list[Stage] = [
        Stage("seq_len", context.bind(Work(
            resources={"hbm": _INDEX_BYTES, "mte2": _INDEX_BYTES}
        )), placement=place),
    ]

    chunk = _cdiv(length, plan["num_kv_splits"])
    nonempty = _cdiv(length, chunk)

    def tile(index: int):
        load_bytes = block_dv * _ACCUM_BYTES + _ACCUM_BYTES
        load = Work(resources={"hbm": load_bytes, "mte2": load_bytes})
        merge = Work(resources={"vector": _merge_vector_ops(block_dv)})
        return (
            Stage("partial_load", context.bind(load), placement=place),
            Stage("online_merge", context.bind(merge), after=("partial_load",),
                  placement=place),
        )

    stages.append(Stage(
        "split_merge",
        after=("seq_len",),
        program=PipelineProgram(nonempty, _MERGE_PIPELINE_SLOTS, tile),
    ))
    store_bytes = block_dv * _PROGRAM_BYTES + _PROGRAM_BYTES
    stages.append(Stage(
        "store",
        after=("split_merge",),
        work=context.bind(Work(resources={
            "hbm": store_bytes,
            "mte3": store_bytes,
            "vector": _final_vector_ops(block_dv),
        })),
        placement=place,
    ))
    return KernelProgram(tuple(stages))


def _cta_streams(
    context: RecipeContext,
    total: int,
    make_cta: Callable[[int, int], KernelProgram],
) -> ParallelProgram:
    """Lazy per-physical-owner CTA streams, at most one resident CTA each.

    CTA ``index`` is pinned to owner ``index % core_count``; the owner stream
    visits exactly those indices.  A pipeline cursor, not an eager CTA list,
    keeps live CTA programs bounded by the declared physical owners.
    """
    cores = context.core_count
    streams = []
    for core in range(min(total, cores)):
        count = (total - core + cores - 1) // cores

        def tile(index: int, core: int = core):
            return (Stage("cta", program=make_cta(core + index * cores, core)),)

        streams.append(PipelineProgram(count, _CTA_STREAM_SLOTS, tile))
    return ParallelProgram(tuple(streams))


def _stage1_program(
    context: RecipeContext, plan: Mapping[str, Any], seq_lens: tuple[int, ...]
) -> ParallelProgram:
    splits = plan["num_kv_splits"]
    head_blocks = plan["head_blocks"]
    batch = len(seq_lens)
    total = batch * head_blocks * splits

    def make_cta(index: int, core: int) -> KernelProgram:
        request, rem = divmod(index, head_blocks * splits)
        head_block, split_id = divmod(rem, splits)
        return _stage1_cta(
            context, plan, seq_lens[request], head_block, split_id, core
        )

    return _cta_streams(context, total, make_cta)


def _stage2_program(
    context: RecipeContext, plan: Mapping[str, Any], seq_lens: tuple[int, ...]
) -> ParallelProgram:
    heads = plan["heads"]
    total = len(seq_lens) * heads

    def make_cta(index: int, core: int) -> KernelProgram:
        request, _ = divmod(index, heads)
        return _stage2_cta(context, plan, seq_lens[request], core)

    return _cta_streams(context, total, make_cta)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build one explicit source kernel child selected by metadata."""
    if geometry.compute_units < 1:
        raise ValueError("mla_decode_geometry_requires_positive_compute_units")
    if any(units < 1 for units in geometry.resource_units.values()):
        raise ValueError("mla_decode_geometry_resource_units_must_be_positive")
    if any(capacity < 0 for capacity in geometry.buffers.values()):
        raise ValueError("mla_decode_geometry_buffer_capacity_must_be_nonnegative")
    if any(units < 1 for units in geometry.buffer_units.values()):
        raise ValueError("mla_decode_geometry_buffer_units_must_be_positive")

    plan, seq_lens = _read_plan(op)
    phase = dict(op.metadata).get("kernel_phase")
    if phase not in _KERNEL_PHASES:
        raise ValueError(
            "mla_decode_kernel_phase_required:"
            "expand_with_kernels_before_build:"
            f"selected:{_KERNEL_PHASES}"
        )

    if phase == "stage1":
        work = _stage1_totals(plan, seq_lens)

        def program(context: RecipeContext) -> ParallelProgram:
            return _stage1_program(context, plan, seq_lens)
    else:
        work = _stage2_totals(plan, seq_lens)

        def program(context: RecipeContext) -> ParallelProgram:
            return _stage2_program(context, plan, seq_lens)

    return OperatorPlan(work, program)


# ``kernels`` is the executable launch boundary.  Keeping the name explicit in
# ``__all__`` makes the coordinator's ``PlannedRecipe(build, kernels)`` binding
# reviewable; metadata such as ``source_kernel_launches`` never substitutes for
# this callable.
__all__ = ["IMPLEMENTATION", "SOURCE_FILES", "build", "kernels", "make_op"]
