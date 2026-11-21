"""Constructive CATLASS and DSL candidate search.

[Role] Translate declared matrix and tile-pipeline contracts into replay candidates.
[Invariants] Explicit pool ownership, supported routes and full verified seed retention.
[Timeouts & Locks] Candidate and oracle trial counts bound search, with no search locks.
[Side-Effect Policy] Read-only capability queries and oracle replays.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from math import prod
from typing import TYPE_CHECKING

from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram
from mindie_sim.prediction.streams import MAIN_STREAM, bind_stream_owners
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.catlass_catalog import CatlassProblem, matrix_problem, streamk_partition
from mindie_sim.templates.dsl_catalog import ATTN_RES, AttnResProblem, DSL_KERNEL_IDS
from mindie_sim.templates.registry import builtin_template_catalog

from .contracts import CatlassKernelSpec, DslKernelSpec, FuseRegionSpec
from .dual_bound import _unique_ops
from .framework_solver import (
    DEFAULT_MAX_TRIALS, FrameworkSolveResult, _TrialSearch, _atomic_ops, _machine,
    _interleave, _search_bases, _stream_candidates, _stream_plan, _wall, _with_operations,
)

if TYPE_CHECKING:
    from mindie_sim.rewrite.oracle import FullPlanEvaluation, ScenarioDossier
    from mindie_sim.rewrite.schema import RewritePlan

MATRIX_RANK = 3
_GROUPED_FAMILIES = frozenset({"grouped_gemm", "grouped_matmul", "grouped_moe", "mc2"})
_MX_MODES = frozenset({"mxfp8", "mxfp4", "a8w4_mx"})
_CATLASS_FIELDS = ("dispatch_policy", "scheduler", "splitk_factor", "swizzle_offset", "swizzle_direction",
                   "l1_tile_shape", "l0_tile_shape", "stages", "epilogue_kind", "simt_thread_block_dim")
DSL_STAGES = ("D1_dsl_fuse", "D2_catlass", "D3_retune")
__all__ = ("DslSolveResult", "streamk_crossing_partials", "select_catlass_candidates",
           "select_dsl_candidates", "solve_dsl")

@dataclass(frozen=True, slots=True)
class DslSolveResult:
    best_plan: RewritePlan
    best_eval: FullPlanEvaluation
    best_wall_ns: int
    catlass_specs: tuple[CatlassKernelSpec, ...]
    dsl_specs: tuple[DslKernelSpec, ...]
    fuse_specs: tuple[FuseRegionSpec, ...]
    stage_history: tuple[tuple[str, int, str], ...]
    tried_candidates: int

def streamk_crossing_partials(b_tot: int, g_k: int, c_aic: int) -> int:
    if b_tot < 0 or g_k < 1 or c_aic < 0:
        raise ValueError("invalid StreamK geometry")
    return streamk_partition(b_tot, g_k, c_aic).partial_count if c_aic else 0


def _matrix_problem(op, meta):
    if op.source_boundary == "call":
        return None
    declared = meta.get("catlass_problem")
    if declared is not None:
        return matrix_problem(declared)
    shape = tuple(meta.get("matrix_shape", op.local_shape))
    if len(shape) != MATRIX_RANK or min(shape, default=0) <= 0 or op.link_bytes or op.vector_ops:
        return None
    if sum(flops for _, flops in op.matrix_flops or ((op.dtype, op.flops),)) != 2 * prod(shape):
        return None
    groups = tuple(meta.get("group_rows", ()))
    if op.family in _GROUPED_FAMILIES and not groups:
        return None
    mode = meta.get("native_matrix_mode", op.dtype)
    dtype_a = meta.get("dtype_a", meta.get("activation_format", op.dtype))
    dtype_b = meta.get("dtype_b", meta.get("weight_format", op.dtype))
    mx_mode = mode if mode in _MX_MODES else None
    if (dtype_a != dtype_b and not mx_mode) or (meta.get("weight_scale_format") and not mx_mode):
        return None
    return CatlassProblem(
        *shape, dtype_a, dtype_b, meta.get("dtype_c", meta.get("output_format", op.dtype)),
        meta.get("dtype_acc", "int32" if dtype_a == "int8" else "fp32"),
        groups=groups, a_id=f"{op.op_id}:a", b_id=meta.get("weight_allocation_id", f"{op.op_id}:b"),
        c_id=f"{op.op_id}:c", has_bias=bool(meta.get("has_bias")), mx_mode=mx_mode,
    )


def _catlass_specs(op, problem, datapath, aic, aiv, template_id=None):
    meta = dict(op.metadata)
    options = {}
    for key, parameter in (("l1_tile_shape", "l1_tiles"), ("l0_tile_shape", "l0_tiles"),
                           ("mutex_stages", "stage_options"), ("simt_thread_block_dim", "simt_thread_blocks")):
        if key in meta:
            options[parameter] = (tuple(meta[key]),)
    if "swizzle_offsets" in meta:
        options["swizzle_offsets"] = tuple(meta["swizzle_offsets"])
    return builtin_template_catalog().enumerate(
        "K2", problem, datapath, template_id=template_id, aic_budget=aic, aiv_budget=aiv,
        evg_tree=tuple(meta.get("evg_tree", ())), enable_unit_flag=bool(meta.get("enable_unit_flag")),
        enable_l1_resident=bool(meta.get("l1_resident_slot_match")), **options,
    )


def select_catlass_candidates(
    op: OpSpec, datapath: DatapathCapabilitySpec, aic_budget: int, aiv_budget: int,
) -> tuple[CatlassKernelSpec, ...]:
    if not aic_budget:
        return ()
    try:
        problem = _matrix_problem(op, dict(op.metadata))
        return _catlass_specs(op, problem, datapath, aic_budget, aiv_budget) if problem else ()
    except ModelingError:
        return ()


def _dsl_declarations(op):
    base = dict(op.metadata)
    for index, override in enumerate(base.get("dsl_variants", ({},))):
        meta = {**base, **override}
        identifier = meta.get("dsl_kernel", op.family if f"dsl.{op.family}" in DSL_KERNEL_IDS else op.recipe)
        identifier = "dsl." + identifier.removeprefix("dsl.")
        kernel = "dsl." + identifier.removeprefix("dsl.").partition(".")[0]
        problem = meta.get("dsl_problem")
        if problem is None and kernel == ATTN_RES and op.local_shape and op.layout == "contiguous":
            problem = AttnResProblem(prod(op.local_shape[:-1]), op.local_shape[-1], op.dtype)
        if problem is not None and kernel in DSL_KERNEL_IDS:
            yield index, identifier, problem, tuple(meta.get("dsl_members", (op.op_id,)))


def _convex_members(ops, ids):
    if len(ids) != len(set(ids)) or not ids:
        return ()
    members = set(ids)
    by_id = {op.op_id: op for op in ops}
    ordered = tuple(op for op in ops if op.op_id in members)
    if tuple(op.op_id for op in ordered) != ids or any(
        op.link_bytes or dict(op.metadata).get("collective_barrier") for op in ordered
    ):
        return ()
    internal = set()
    outside = []
    for op in ordered:
        if any(dep in members and dep not in internal for dep in op.deps):
            return ()
        internal.add(op.op_id)
        outside.extend(dep for dep in op.deps if dep not in members)
    seen = set()
    while outside:
        dep = outside.pop()
        if dep in members:
            return ()
        if dep in by_id and dep not in seen:
            seen.add(dep)
            outside.extend(by_id[dep].deps)
    return ordered

def select_dsl_candidates(
    ops: Sequence[OpSpec], datapath: DatapathCapabilitySpec, aic_budget: int, aiv_budget: int,
) -> tuple[tuple[DslKernelSpec, tuple[str, ...]], ...]:
    return tuple(dict.fromkeys((spec, ids) for spec, ids, _ in _dsl_candidates(
        ops, datapath, aic_budget, aiv_budget,
    )))


def _dsl_candidates(ops, datapath, aic, aiv):
    if not aic or not aiv:
        return
    for op in ops:
        for index, identifier, problem, members in _dsl_declarations(op):
            if op.op_id not in members or not _convex_members(ops, members):
                continue
            try:
                specs = builtin_template_catalog().enumerate(
                    "K3", problem, datapath, template_id=identifier, all_variants=True,
                    aic_budget=aic, aiv_budget=aiv,
                )
            except ModelingError:
                continue
            for spec in specs:
                yield spec, members, index


def _declared_regions(ops):
    for op in ops:
        for declaration in dict(op.metadata).get("fuse_regions", ()):
            if declaration.get("numerically_equivalent") and "compiler_options" in declaration:
                yield declaration


def _region_lowerer(declaration, members):
    options = dict(declaration["compiler_options"])
    options.update(
        dependencies={op.op_id: op.deps for op in members}, edge_routes=tuple(declaration["edge_routes"]),
        live_in_obligations=tuple(declaration.get("live_in_obligations", ())),
        live_out_obligations=tuple(declaration.get("live_out_obligations", ())),
        backend=declaration["backend"],
    )

    def lower(datapath, *, region_id, **binding):
        _, result = builtin_template_catalog().compile_region(
            region_id, tuple(op.op_id for op in members), datapath, **{**options, **binding},
        )
        return result
    return lower


def _dsl_proposals(ops, datapath, aic, aiv):
    from mindie_sim.rewrite.schema import FuseRegion, KernelBoundary, OpRef, SelectTemplate

    for spec, ids, index in _dsl_candidates(ops, datapath, aic, aiv):
        refs = tuple(OpRef("atom", op_id) for op_id in ids)
        meta = next(dict(op.metadata) for op in ops if op.op_id == ids[-1])
        if len(refs) == 1:
            yield (SelectTemplate(refs[0], "K3", spec.kernel_id, (
                ("aic_budget", spec.aic_budget), ("aiv_budget", spec.aiv_budget), ("declaration", index),
            )),), spec
        elif "dsl_preserve_ports" in meta:
            yield (FuseRegion(refs, tuple(meta["dsl_preserve_ports"]), (KernelBoundary(spec.kernel_id, refs),),
                              fusion_tier="K3_dsl_pipeline"),), spec
    for declaration in _declared_regions(ops):
        ids = tuple(declaration["op_ids"])
        members = _convex_members(ops, ids)
        if len(members) < 2:
            continue
        identifier = declaration["region_id"]
        try:
            result = _region_lowerer(declaration, members)(datapath, region_id=identifier)
        except ModelingError:
            continue
        refs = tuple(OpRef("atom", op_id) for op_id in ids)
        preserve = tuple(declaration.get("preserve_ports", ()))
        yield (FuseRegion(refs, preserve, (KernelBoundary(identifier, refs),),
                          fusion_tier="K3_dsl_pipeline"),), result.summary.spec


def _encode_spec_value(value):
    return ",".join(map(str, value)) if isinstance(value, tuple) else "" if value is None else value


def _catlass_proposals(ops, datapath, aic, aiv):
    from mindie_sim.rewrite.schema import OpRef, SelectTemplate

    for op in ops:
        candidates = select_catlass_candidates(op, datapath, aic, aiv)
        first_by_template = {spec.template_id: spec for spec in reversed(candidates)}
        for spec in sorted(candidates, key=lambda item: item != first_by_template[item.template_id]):
            params = tuple((key, _encode_spec_value(getattr(spec, key))) for key in _CATLASS_FIELDS)
            params += (("aic_budget", spec.aic_budget), ("aiv_budget", spec.aiv_budget))
            yield (SelectTemplate(OpRef("atom", op.op_id), "K2", spec.template_id, params),), spec


def _catalog_lowerer(op, members, datapath, tier, identifier, params, limits):
    catalog = builtin_template_catalog()
    options = dict(params)
    aic = options.pop("aic_budget", limits[0])
    aiv = options.pop("aiv_budget", limits[1])
    if any(type(budget) is not int or not 0 <= budget <= limit for budget, limit in zip((aic, aiv), limits)):
        raise ModelingError(f"rewrite_template_core_budget_invalid:{op.op_id}")
    if tier == "K2":
        problem = _matrix_problem(op, dict(op.metadata))
        if problem is not None and options.keys() <= set(_CATLASS_FIELDS):
            for spec in _catlass_specs(op, problem, datapath, aic, aiv, identifier):
                if all(_encode_spec_value(getattr(spec, key)) == value for key, value in options.items()):
                    return partial(catalog.lower, tier, spec, problem)
    elif tier == "K3":
        if not options and op.family == "rewrite_fused":
            ids = tuple(member.op_id for member in members)
            declared_region = next((entry for entry in _declared_regions(members)
                                    if entry["region_id"] == identifier and tuple(entry["op_ids"]) == ids), None)
            if declared_region is not None:
                return _region_lowerer(declared_region, members)
        index = options.pop("declaration", None)
        kernel = "dsl." + identifier.removeprefix("dsl.").partition(".")[0]
        if options.keys() <= {"variant"} and (index is None or type(index) is int):
            choices = []
            member_ids = tuple(member.op_id for member in members)
            for source in members:
                for ordinal, declared, problem, ids in _dsl_declarations(source):
                    if (ids == member_ids and declared in {kernel, identifier}
                            and (index is None or index == ordinal)):
                        choices.append(problem)
            if len(choices) == 1:
                spec, = catalog.enumerate(
                    tier, choices[0], datapath, template_id=identifier, aic_budget=aic, aiv_budget=aiv, **options,
                )
                return partial(catalog.lower, tier, spec, choices[0])
    raise ModelingError(f"rewrite_template_recipe_unbound:{op.op_id}:{identifier}")


def replay_catalog_binding(scenario, entry, op, binding, plan, stream):
    from mindie_sim.rewrite.apply import _die, _path
    from mindie_sim.rewrite.schema import FuseRegion

    if binding and binding.tier in {"K2", "K3"} and binding.template_id.startswith(("catlass.", "dsl.")):
        tier, identifier, params = binding.tier, binding.template_id, binding.params
        members = (op,)
    elif op.family == "rewrite_fused" and dict(op.metadata).get("fusion_tier") == "K3_dsl_pipeline":
        edit = next(edit for edit in plan.edits if op.op_id == f"rewrite:{edit.edit_id}")
        operation = edit.operation
        if (not isinstance(operation, FuseRegion) or len(operation.kernel_boundaries) != 1
                or any(ref.kind != "atom" for ref in operation.members)):
            return None
        ids = {ref.id for ref in operation.members}
        members = tuple(member for member in entry.ops if member.op_id in ids)
        tier, identifier, params = "K3", op.recipe, ()
        if not identifier.startswith("dsl.") and not any(
            decl["region_id"] == identifier for decl in _declared_regions(members)
        ):
            return None
    else:
        return None
    path = _path(scenario, entry.replica, op.stage)
    limits = path.aic_count, path.aiv_count
    if stream is not None:
        owners = bind_stream_owners(stream, path)[dict(stream.assignments).get(op.op_id, MAIN_STREAM)]
        limits = owners.aic_limit, owners.aiv_limit
    return bind_catalog_recipe(op, members, path, tier=tier, identifier=identifier, params=params,
                               limits=limits, die=_die(scenario, entry.replica, op.stage).rank,
                               replica=entry.replica)


def bind_catalog_recipe(op, members, datapath, *, tier, identifier, params, limits, die, replica):
    """Resolve the edit against its problem and use one specialization for audit and execution."""
    lower = _catalog_lowerer(op, members, datapath, tier, identifier, params, limits)
    result = lower(datapath, die=die, region_id=op.op_id)
    raw = result.summary
    summary = replace(
        raw, op_ids=tuple(member.op_id for member in members),
        semantic_work=tuple(replace(item, die=die, replica=replica, pp_stage=op.stage)
                            for item in raw.semantic_work),
    )
    if isinstance(raw.spec, DslKernelSpec):
        summary = replace(summary, **{
            field: tuple(replace(item, allocation_id=f"{op.op_id}:{item.allocation_id}")
                         for item in getattr(raw, field))
            for field in ("read_obligations", "write_obligations")
        })
    resources = Counter()
    roles = {"mte2_aic": "mte2", "mte2_aiv": "mte2",
             "vec_alu": "vector", "vec_reduce": "vector", "vec_sfu": "vector", "sort_topk": "vector",
             "aiv_simt": "vector"}
    for stage in result.program.stages:
        for role, amount in stage.work.items():
            if role != "cube":
                resources[roles.get(role, role)] += amount
    work = Work(dict(summary.issued_cube_flops_by_dtype), resources)

    def build(_op, _geometry):
        def program(context):
            program = lower(context.datapath or datapath, context=context, die=die, region_id=op.op_id).program
            return _share_hbm_row(program, context)
        return OperatorPlan(work, program)

    return PlannedRecipe(build), summary


def _share_hbm_row(program, context):
    """Bind native HBM service bytes to the shared row used by shipping kernels."""
    bus = getattr(program, "hbm_bus", None)
    if bus is None:
        return program
    row = context.resource_names["hbm"]
    directional = {bus.row_r, bus.row_w}
    counters = {bus.ctr_r, bus.ctr_w}
    stages = []
    for stage in program.stages:
        amount = sum(value for key, value in stage.deliveries if key in counters)
        if amount:
            work = {key: value for key, value in stage.work.items() if key not in directional}
            duration = max(stage.work.get(key, 0) for key in directional)
            stage = replace(stage, work={**work, row: amount}, caps={**stage.caps, row: amount / duration},
                            coupled=True)
        stages.append(stage)
    return KernelProgram(tuple(stages), program.minimum_ms)


def solve_dsl(
    dossier: ScenarioDossier, fw_result: FrameworkSolveResult, *, dual_report=None,
    dual_program=None, event_dag=None, dual_certificate=None, max_trials: int = DEFAULT_MAX_TRIALS,
) -> DslSolveResult:
    search = _TrialSearch(dossier, max_trials, dual_program=dual_program, event_dag=event_dag,
                          dual_certificate=dual_certificate)
    search.seed(fw_result.best_plan, fw_result.best_eval)

    def candidates(base, scenario, index, stage):
        ops = _atomic_ops(scenario)
        paths = tuple(datapath_for_die(device) for device in getattr(_machine(scenario), "dies", ()))
        if index == len(DSL_STAGES) - 1:
            _, evaluation = search.frontier.get(base.base_graph_sha256, (base, None))
            applied = getattr(evaluation, "applied_plan", getattr(evaluation, "applied", None))
            placed_ops = _unique_ops(applied.executable) if applied is not None else ops
            atomic_ids = {op.op_id for op in ops}
            handles = ()
            if applied is not None:
                from mindie_sim.rewrite.schema import OpRef
                handles = tuple((op_id, OpRef("result_of", edit_id))
                                for edit_id, output in applied.handles
                                for op_id in output if len(output) == 1 and op_id not in atomic_ids)
            yield from (_stream_plan(base, stage, operation)
                        for operation in _stream_candidates(scenario, placed_ops, handles))
        elif paths:
            path = paths[0]
            aic, aiv = min(p.aic_count for p in paths), min(p.aiv_count for p in paths)
            if index == 0:
                proposals = _dsl_proposals(ops, path, aic, aiv)
            else:
                bottlenecks = dict(getattr(dossier, "bottleneck_ops", ()))
                sites = sorted(ops, key=lambda op: -bottlenecks.get(op.op_id, 0))
                proposals = _catlass_proposals(sites, path, aic, aiv)

            for operations, _ in proposals:
                yield _with_operations(base, stage, operations)

    for index, stage in enumerate(DSL_STAGES):
        limit = search.remaining // (len(DSL_STAGES) - index)
        bases = tuple(_search_bases(dossier, search))
        search.stage(stage, _interleave(candidates(base, scenario, index, stage)
                                        for base, scenario in bases), limit)
    best = search.best_plan
    specs = tuple(dict.fromkeys(summary.spec for summary in getattr(search.best_eval, "region_summaries", ())))
    return DslSolveResult(search.best_plan, search.best_eval, _wall(search.best_eval),
                          tuple(s for s in specs if isinstance(s, CatlassKernelSpec)),
                          tuple(s for s in specs if isinstance(s, DslKernelSpec)),
                          tuple(s for s in specs if isinstance(s, FuseRegionSpec)),
                          tuple(search.history), search.tried)
