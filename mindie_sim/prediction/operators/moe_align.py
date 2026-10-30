"""CUDA MoE alignment work and finite CTA programs.

[Role] Export source operators and execute their physical owner programs.
[Invariants] Geometry is normalized separately; launch boundaries, exact work,
  owner partitions, finite buffers and input dependencies remain explicit.
  Unsupported source branches are never replaced by another implementation.
[Timeouts & Locks] Lazy finite programs keep only active tasks; no waits or locks.
[Side-Effect Policy] Return records and submit to the caller Engine; no device I/O.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import unit_group
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelineProgram, RecipeContext, Stage)
from mindie_sim.schema.operators import OpSpec

from .moe_align_geometry import (
    _IMPLEMENTATION, _WRAPPER_PATH, _CUDA_PATH, _MATH_PATH,
    _WARP_SIZE, _SMALL_NUMEL_LIMIT, _SMALL_EXPERT_LIMIT,
    _FILL_THREADS, _SORT_THREADS, _SORT_BLOCK_LIMIT, _ALIGN_THREADS,
    _INT32_BYTES, _PIPE_ROLES, _SHARED_POOL, _INTEGRAL_BYTES,
    _DEFAULT_TOPK_DTYPE, _PHASE_IDS, _SOURCE_IDENTITY, _SOURCE_LINES,
    _UNRESOLVED, _SELECTION_LIMITATIONS, _source_identity, _require_int,
    _require_bool, _require_sequence, _select_implementation, _parse_expert_map,
    _round_up, _analyze,
)

def _work(*, hbm: int = 0, mte2: int = 0, mte3: int = 0, vector: int = 0) -> Work:
    resources = {
        role: float(value)
        for role, value in (("hbm", hbm), ("mte2", mte2), ("mte3", mte3),
                            ("vector", vector))
        if value
    }
    return Work(resources=resources)


def _small_defs(analysis: Mapping[str, Any]) -> tuple[tuple[str, Work, tuple[str, ...], int], ...]:
    numel = int(analysis["numel"])
    num_experts = int(analysis["num_experts"])
    topk_bytes = int(analysis["topk_element_bytes"])
    max_tokens_padded = int(analysis["max_tokens_padded"])
    max_m_blocks = int(analysis["max_m_blocks"])
    valid_entries = int(analysis["valid_entries"])
    stride = max(num_experts, _WARP_SIZE)
    has_map = bool(analysis["has_expert_map_kernel_arg"])
    topk_pass = numel * topk_bytes
    map_pass = numel * _INT32_BYTES if has_map else 0
    return (
        ("init_sorted",
         _work(hbm=max_tokens_padded * _INT32_BYTES,
               mte3=max_tokens_padded * _INT32_BYTES), (), 0),
        ("clear_counts", _work(vector=stride * num_experts), (), 0),
        ("histogram",
         _work(hbm=topk_pass + map_pass, mte2=topk_pass + map_pass,
               vector=numel), ("clear_counts",), 0),
        ("prefix",
         _work(hbm=_INT32_BYTES, mte3=_INT32_BYTES,
               vector=stride * num_experts + 2 * num_experts),
         ("histogram", "init_sorted"), 0),
        ("expert_ids",
         _work(hbm=max_m_blocks * _INT32_BYTES,
               mte3=max_m_blocks * _INT32_BYTES, vector=max_m_blocks),
         ("prefix", "init_sorted"), 0),
        ("scatter",
         _work(hbm=topk_pass + map_pass + valid_entries * _INT32_BYTES,
               mte2=topk_pass + map_pass,
               mte3=valid_entries * _INT32_BYTES, vector=numel),
         ("expert_ids", "init_sorted"), 0),
    )


def _large_align_defs(analysis: Mapping[str, Any]) -> tuple[tuple[str, Work, tuple[str, ...], int], ...]:
    numel = int(analysis["numel"])
    num_experts = int(analysis["num_experts"])
    topk_bytes = int(analysis["topk_element_bytes"])
    max_tokens_padded = int(analysis["max_tokens_padded"])
    max_m_blocks = int(analysis["max_m_blocks"])
    padded_num_experts = int(analysis["padded_num_experts"])
    has_map = bool(analysis["has_expert_map_kernel_arg"])
    topk_pass = numel * topk_bytes
    map_pass = numel * _INT32_BYTES if has_map else 0
    cumsum_write = (num_experts + 1) * _INT32_BYTES
    cumsum_fill_reads = (2 * num_experts + _ALIGN_THREADS) * _INT32_BYTES
    expert_id_bytes = max_m_blocks * _INT32_BYTES
    return (
        ("init_sorted",
         _work(hbm=max_tokens_padded * _INT32_BYTES,
               mte3=max_tokens_padded * _INT32_BYTES), (), 1),
        ("clear_counts", _work(vector=padded_num_experts), (), 0),
        ("histogram",
         _work(hbm=topk_pass + map_pass, mte2=topk_pass + map_pass,
               vector=numel), ("clear_counts",), 0),
        ("scan",
         _work(hbm=cumsum_write + _INT32_BYTES,
               mte3=cumsum_write + _INT32_BYTES), ("histogram",), 0),
        ("expert_ids",
         _work(hbm=expert_id_bytes + cumsum_fill_reads,
               mte2=cumsum_fill_reads, mte3=expert_id_bytes,
               vector=max_m_blocks), ("scan",), 0),
    )


def _sort_work(analysis: Mapping[str, Any]) -> Work:
    numel = int(analysis["numel"])
    valid = int(analysis["valid_entries"])
    topk_bytes = int(analysis["topk_element_bytes"])
    map_bytes = numel * _INT32_BYTES if bool(analysis["has_expert_map_kernel_arg"]) else 0
    topk_pass = numel * topk_bytes
    return _work(
        hbm=topk_pass + map_bytes + 3 * valid * _INT32_BYTES,
        mte2=topk_pass + map_bytes + valid * _INT32_BYTES,
        mte3=2 * valid * _INT32_BYTES,
        vector=numel,
    )


def _sort_cta_work(analysis: Mapping[str, Any], cta: int) -> Work:
    ids = int(analysis["sort_cta_ids"][cta])
    valid = int(analysis["sort_cta_valid"][cta])
    topk_bytes = int(analysis["topk_element_bytes"])
    map_width = _INT32_BYTES if bool(analysis["has_expert_map_kernel_arg"]) else 0
    topk_pass = ids * topk_bytes
    map_pass = ids * map_width
    return _work(
        hbm=topk_pass + map_pass + 3 * valid * _INT32_BYTES,
        mte2=topk_pass + map_pass + valid * _INT32_BYTES,
        mte3=2 * valid * _INT32_BYTES,
        vector=ids,
    )


def _phase_entries(analysis: Mapping[str, Any],
                   phase: str) -> tuple[tuple[str, Work], ...]:
    if phase == "small_batch_expert":
        return tuple((name, work) for name, work, _, _ in _small_defs(analysis))
    if phase == "large_align":
        return tuple((name, work) for name, work, _, _ in _large_align_defs(analysis))
    if phase == "large_sort":
        return (("sort", _sort_work(analysis)),)
    raise ValueError(f"unsupported_moe_align_kernel_phase:{phase!r}")


def _phase_work(analysis: Mapping[str, Any], phase: str) -> Work:
    total = Work()
    for _, work in _phase_entries(analysis, phase):
        total = total + work
    return total


def _stage_work(analysis: Mapping[str, Any]) -> tuple[tuple[str, Work], ...]:
    """Declare aggregate stage work; never enumerate token or scalar events."""
    if bool(analysis["small_batch_expert_mode"]):
        stages = list(_phase_entries(analysis, "small_batch_expert"))
    else:
        stages = list(_phase_entries(analysis, "large_align"))
        stages.extend(_phase_entries(analysis, "large_sort"))
    return tuple(stages)


def _sum_work(stages: tuple[tuple[str, Work], ...]) -> Work:
    total = Work()
    for _, stage in stages:
        total = total + stage
    return total


def _work_summary(stages: tuple[tuple[str, Work], ...]) -> dict[str, Any]:
    totals = {"hbm_bytes": 0.0, "mte2_bytes": 0.0, "mte3_bytes": 0.0,
              "vector_ops": 0.0}
    for _, stage in stages:
        for role, value in stage.resources.items():
            totals[f"{role}_bytes" if role != "vector" else "vector_ops"] += value
    return {
        **totals,
        "stage_work": {
            name: {role: value for role, value in sorted(stage.resources.items())}
            for name, stage in stages
        },
    }


def _dispatch(analysis: Mapping[str, Any]) -> dict[str, Any]:
    small = bool(analysis["small_batch_expert_mode"])
    return {
        "implementation": _IMPLEMENTATION,
        "pad_sorted_ids": bool(analysis["pad_sorted_ids"]),
        "ignore_invalid_experts": bool(analysis["ignore_invalid_experts"]),
        "expert_map_present": bool(analysis["expert_map_present"]),
        "has_expert_map_kernel_arg": bool(analysis["has_expert_map_kernel_arg"]),
        "expert_ids_post_gather": bool(analysis["expert_ids_post_gather"]),
        "expert_map_post_gather_supported": False,
        "path": analysis["path"],
        "symbol": analysis["symbol"],
        "sort_symbol": analysis["sort_symbol"],
        "topk_ids_dtype": analysis["topk_ids_dtype"],
        "output_dtype": "int32",
        "expert_map_dtype": "int32" if analysis["expert_map_present"] else None,
        "num_experts": int(analysis["num_experts"]),
        "block_size": int(analysis["block_size"]),
        "numel": int(analysis["numel"]),
        "numel_below_1024": int(analysis["numel"]) < _SMALL_NUMEL_LIMIT,
        "num_experts_at_most_64": (
            int(analysis["num_experts"]) <= _SMALL_EXPERT_LIMIT),
        "small_batch_expert_mode": small,
        "kernel_phases": tuple(analysis["kernel_phases"]),
        "host_submissions": int(analysis["host_submissions"]),
        "grid": (1,) if small else (2,),
        "align_ctas": int(analysis["align_ctas"]),
        "threads": analysis["small_threads"] if small else analysis["big_threads"],
        "fill_threads": analysis["small_fill_threads"],
        "shared_bytes": analysis["align_shared_bytes"],
        "sort_grid": analysis["sort_grid"],
        "sort_ctas": analysis["sort_ctas"],
        "sort_block_threads": analysis["sort_block_threads"],
        "scan_lanes": analysis["scan_lanes"],
        "residual_writers": int(analysis["residual_writers"]),
        "residual_tail_blocks": int(analysis["tail_blocks"]),
        "expert_ids_blocks_written": int(analysis["expert_ids_blocks_written"]),
    }


def make_op(params: dict) -> OpSpec:
    """Build the selected CUDA alignment OpSpec from explicit routing ids."""
    if not isinstance(params, Mapping):
        raise ValueError("moe_align_params_must_be_mapping")
    analysis = _analyze(params)
    stages = _stage_work(analysis)
    dispatch = _dispatch(analysis)
    phase = params.get("phase", "prefill")
    if phase not in {"prefill", "decode", "mixed"}:
        raise ValueError(f"unsupported_moe_align_phase:{phase!r}")
    metadata = (
        ("dispatch", dispatch),
        ("implementation", dict(dispatch)),
        ("source_identity", _source_identity()),
        ("source_lines", dict(_SOURCE_LINES)),
        ("analysis", analysis),
        ("counts", tuple(analysis["counts"])),
        ("padded_per_expert", tuple(analysis["padded_per_expert"])),
        ("cumsum", tuple(analysis["cumsum"])),
        ("work", _work_summary(stages)),
        ("unresolved", _UNRESOLVED),
        ("selection_limitations", _SELECTION_LIMITATIONS),
    )
    notes = [
        "selected vLLM current-checkout CUDA moe_align_block_size path; "
        "not a historical capture image",
        f"dispatch: {analysis['path']} ({analysis['symbol']})",
        "exact per-expert routing histogram and padded prefix; no mean expert count",
        "sorted_token_ids sentinel writes and every initialized expert_ids block "
        "are charged",
        "the residual expert_ids fill is outside the expert guard, so all "
        "residual stride threads write",
        "routing ids are required in [0, num_experts); the small scatter has no "
        "out-of-range guard",
        "no historical moe_align dispatch or timing is claimed",
    ]
    if not analysis["small_batch_expert_mode"]:
        notes.append(
            "large path is two host kernel launches: align grid 2 then sort grid "
            f"{analysis['sort_grid']}")
        notes.append(
            "CUB BlockScan instruction rate is unsupported and omitted, not fabricated")
    if analysis["has_expert_map_kernel_arg"]:
        notes.append(
            "expert map uses the native filtered kernel; the wrapper post-gather "
            "expert_map[expert_ids] path is rejected at make_op for lack of a "
            "native ATen launch/layout source")
    return OpSpec(
        op_id="moe_align",
        family="moe_align",
        phase=phase,
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=(int(analysis["tokens"]), int(analysis["top_k"])),
        local_shape=(int(analysis["tokens"]), int(analysis["top_k"])),
        padded_shape=(int(analysis["total_padded"]),),
        launch_shape=(int(analysis["max_tokens_padded"]),),
        dtype="int32",
        layout="flat",
        recipe=_IMPLEMENTATION, source_boundary="call",
        metadata=metadata,
        notes=tuple(notes),
    )


def _cta_placement(context: RecipeContext, cta: int,
                   roles: tuple[str, ...] = _PIPE_ROLES) -> dict[str, tuple[int, ...]]:
    """Pin one source CTA to its physical owner slot.

    ``context.owned`` partitions each declared resource's units evenly over
    ``core_count`` owner slots.  A CTA is one owner, never the whole device.
    """
    owners = max(1, int(context.core_count))
    owner = int(cta) % owners
    return context.owned(owner, owners, roles)


def _shared_units(context: RecipeContext, cta: int) -> tuple[int, ...]:
    """Map a CTA to its declared per-unit shared-memory pool units."""
    owners = max(1, int(context.core_count))
    units = max(1, int(context.pool_units.get(_SHARED_POOL, 1)))
    try:
        return unit_group(int(cta) % owners, owners, units)
    except ValueError as error:
        raise ValueError(
            "moe_align_shared_pool_units_cannot_partition_owners:"
            f"{units}:{owners}") from error


def _cta_stage(context: RecipeContext, cta: int,
               defs: tuple[tuple[str, Work, tuple[str, ...], int], ...],
               shared_bytes: int) -> Stage:
    stages = tuple(
        Stage(name, context.bind(work), after=after,
              placement=_cta_placement(context, cta))
        for name, work, after, _ in defs
    )
    if shared_bytes and _SHARED_POOL in context.pool_names:
        allocation = context.allocate(
            f"shared_cta{cta}", _SHARED_POOL, float(shared_bytes),
            _shared_units(context, cta))
        return Stage(f"cta{cta}", acquire=(allocation,), free=(allocation,),
                     program=KernelProgram(stages))
    return Stage(f"cta{cta}", program=KernelProgram(stages))


def _defs_for(defs: tuple[tuple[str, Work, tuple[str, ...], int], ...],
              cta: int) -> tuple[tuple[str, Work, tuple[str, ...], int], ...]:
    return tuple(item for item in defs if item[3] == cta)


def _small_program(analysis: Mapping[str, Any]):
    def program(context: RecipeContext):
        return KernelProgram((_cta_stage(
            context, 0, _defs_for(_small_defs(analysis), 0),
            int(analysis["align_shared_bytes"])),))

    return program


def _align_program(analysis: Mapping[str, Any]):
    def program(context: RecipeContext):
        defs = _large_align_defs(analysis)
        shared = int(analysis["align_shared_bytes"])
        return KernelProgram((
            _cta_stage(context, 0, _defs_for(defs, 0), shared),
            _cta_stage(context, 1, _defs_for(defs, 1), shared),
        ))

    return program


def _sort_stage(context: RecipeContext, analysis: Mapping[str, Any],
                after: tuple[str, ...] = ()) -> Stage:
    units = max(1, int(context.core_count))
    total = int(analysis["sort_ctas"] or 0)
    pipelines = []
    for unit in range(min(units, total)):
        count = (total - 1 - unit) // units + 1

        def tile(index: int, unit: int = unit):
            cta = unit + index * units
            return (Stage(
                "sort_cta", context.bind(_sort_cta_work(analysis, cta)),
                placement=_cta_placement(context, unit)),)

        pipelines.append(PipelineProgram(count, 1, tile))
    return Stage("sort_ctas", after=after,
                 program=ParallelProgram(tuple(pipelines)))


def _sort_program(analysis: Mapping[str, Any]):
    def program(context: RecipeContext):
        return KernelProgram((_sort_stage(context, analysis),))

    return program


def _full_program(analysis: Mapping[str, Any]):
    if bool(analysis["small_batch_expert_mode"]):
        return _small_program(analysis)

    def program(context: RecipeContext):
        defs = _large_align_defs(analysis)
        shared = int(analysis["align_shared_bytes"])
        stages = [
            _cta_stage(context, 0, _defs_for(defs, 0), shared),
            _cta_stage(context, 1, _defs_for(defs, 1), shared),
            Stage("align_join", after=("cta0", "cta1")),
        ]
        stages.append(_sort_stage(context, analysis, after=("align_join",)))
        return KernelProgram(tuple(stages))

    return program


def _child(op: OpSpec, analysis: Mapping[str, Any], dispatch: Mapping[str, Any],
           phase: str, index: int, deps: tuple[str, ...]) -> OpSpec:
    child_dispatch = {
        **dict(dispatch),
        "kernel_phase": phase,
        "kernel_index": index,
    }
    metadata = dict(op.metadata)
    metadata["dispatch"] = child_dispatch
    metadata["implementation"] = dict(child_dispatch)
    metadata["kernel_phase"] = phase
    metadata["work"] = _work_summary(_phase_entries(analysis, phase))
    child = replace(
        op, op_id=_PHASE_IDS[phase], deps=deps, source_boundary="kernel",
        metadata=tuple(metadata.items()))
    return _phase_work(analysis, phase).apply(child)


def kernels(op: OpSpec) -> tuple[OpSpec, ...]:
    """Expand the wrapper call into its actual host kernel launches.

    Small path: one kernel.  Large path: the align kernel (two CTAs) and the
    sort kernel are distinct launches.  There is no post-gather launch: that
    wrapper path is rejected at ``make_op``.
    """
    metadata = dict(op.metadata)
    dispatch = dict(metadata.get("dispatch", {}))
    analysis = dict(metadata.get("analysis", {}))
    if dispatch.get("implementation") != _IMPLEMENTATION:
        raise ValueError("moe_align_kernels_require_selected_implementation")
    if op.recipe != _IMPLEMENTATION:
        raise ValueError("moe_align_kernels_require_selected_recipe")
    if metadata.get("kernel_phase") is not None:
        raise ValueError("moe_align_kernels_require_parent_op")
    if analysis.get("path") != dispatch.get("path"):
        raise ValueError("moe_align_kernels_require_make_op_analysis")
    phases = tuple(dispatch.get("kernel_phases", ()))
    expected = tuple(analysis.get("kernel_phases", ()))
    if phases != expected:
        raise ValueError("moe_align_kernels_phase_table_mismatch")
    children = []
    deps: tuple[str, ...] = ()
    for index, phase in enumerate(phases):
        child = _child(op, analysis, dispatch, phase, index, deps)
        children.append(child)
        deps = (child.op_id,)
    return tuple(children)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Declare staging and dependencies for the selected kernel path."""
    meta = dict(op.metadata)
    dispatch = dict(meta.get("dispatch", {}))
    if dispatch.get("implementation") != _IMPLEMENTATION:
        raise ValueError("moe_align_build_requires_selected_implementation")
    if op.recipe != _IMPLEMENTATION:
        raise ValueError("moe_align_build_requires_selected_recipe")
    analysis = dict(meta.get("analysis", {}))
    if not analysis or analysis.get("path") != dispatch.get("path"):
        raise ValueError("moe_align_build_requires_make_op_analysis")
    phases = tuple(dispatch.get("kernel_phases", ()))
    if phases != tuple(analysis.get("kernel_phases", ())):
        raise ValueError("moe_align_build_dispatch_button_mismatch")
    phase = meta.get("kernel_phase")
    if phase is None:
        stages = _stage_work(analysis)
        return OperatorPlan(_sum_work(stages), _full_program(analysis))
    if phase not in phases:
        raise ValueError(f"moe_align_build_unknown_kernel_phase:{phase!r}")
    if phase == "small_batch_expert":
        program = _small_program(analysis)
    elif phase == "large_align":
        program = _align_program(analysis)
    elif phase == "large_sort":
        program = _sort_program(analysis)
    else:  # pragma: no cover - guarded by the phase table above
        raise ValueError(f"moe_align_build_unknown_kernel_phase:{phase!r}")
    return OperatorPlan(_phase_work(analysis, phase), program)


def iter_blocks(op: OpSpec):
    """Yield exact nonzero expert block ranges lazily; O(num_experts)."""
    meta = dict(op.metadata)
    cumsum = tuple(meta["cumsum"])
    padded = tuple(meta["padded_per_expert"])
    block_size = int(dict(meta["analysis"])["block_size"])
    for expert, (start, rows) in enumerate(zip(cumsum, padded)):
        if not rows:
            continue
        yield {
            "expert": expert,
            "start_row": int(start),
            "end_row": int(start + rows),
            "first_block": int(start // block_size),
            "blocks": int(rows // block_size),
            "padded_rows": int(rows),
        }


def stage_work(op: OpSpec) -> tuple[tuple[str, Work], ...]:
    """Public declarative stage ledger for independent work oracles.

    A parent wrapper op reports every source launch stage.  An expanded kernel
    child reports only the phase selected in its metadata, which is the ledger
    ``build`` uses for that child.
    """
    meta = dict(op.metadata)
    analysis = dict(meta.get("analysis", {}))
    if not analysis:
        raise ValueError("moe_align_stage_work_requires_make_op_analysis")
    phase = meta.get("kernel_phase")
    if phase is None:
        return _stage_work(analysis)
    if phase not in tuple(analysis.get("kernel_phases", ())):
        raise ValueError(f"moe_align_stage_work_unknown_kernel_phase:{phase!r}")
    return _phase_entries(analysis, phase)


__all__ = ["build", "iter_blocks", "kernels", "make_op", "stage_work"]
