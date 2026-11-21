"""Synthesized legal fusion recipes on the existing Stage/KernelGroup records.

[Role] Turn a feasible region into a real ``PlannedRecipe``, a causally ordered
  multi-stage ``KernelProgram``, one ``KernelGroup`` buffer declaration and a
  typed traffic ledger.
[Invariants] A recipe exists only for a legal region. Cube prologue/cube/
  epilogue work runs in dependency order, never as one concurrent body. Every
  declared physical service field survives the rewrite; only a proven
  eliminated HBM access is removed. ``exact_contract`` is asserted only when
  the numerical contract proves it. Group buffers carry a real pool, byte
  count and owner units so the Engine admits them.
[Timeouts & Locks] Finite over the candidate region; no I/O or locks.
[Side-Effect Policy] Pure record construction; no Engine submission.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from mindie_sim.prediction.kernel_groups import KernelGroup
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, Stage

from .fusion_feasibility import (
    CubeResources, ReductionMerge, RegionAnalysis, analyze_region,
)
from .fusion_graph import TILE_ROWS, TypedGraph


@dataclass(frozen=True)
class RecipePattern:
    recipe_id: str
    fused_op_name: str
    signature: tuple[str, ...]
    numerical_semantics: str
    implementation_tasks: tuple[str, ...]


_PATTERNS = (
    RecipePattern("fused_add_rms_norm_quant", "FusedAddRmsNormDynamicQuant",
                  ("add", "rms_norm", "dynamic_quant"),
                  "declared add/norm epsilon and quant group axis; one UB resident value",
                  ("emit one vector kernel", "retain residual live-out write")),
    RecipePattern("fused_add_rms_norm_cast_quant", "FusedAddRmsNormCastQuant",
                  ("add", "rms_norm", "cast", "dynamic_quant"),
                  "declared add/norm/cast/quant rounding points",
                  ("emit one vector kernel", "retain cast dtype and quant scales")),
    RecipePattern("fused_rms_norm_quant", "FusedRmsNormDynamicQuant",
                  ("rms_norm", "dynamic_quant"),
                  "exact norm epsilon, full reduction width and quant group axis",
                  ("emit one vector kernel", "hoist gamma once")),
    RecipePattern("fused_swiglu_quant", "FusedSwiGLUQuant",
                  ("swiglu", "quant"),
                  "gate/up halves and output width retained; quant scale axis retained",
                  ("emit one vector kernel", "halve last output width")),
    RecipePattern("fused_rope_kv_cast", "FusedRoPEKVQuant",
                  ("rope", "kv_cast"),
                  "rotation style and cast dtype retained; KV cache write version kept",
                  ("emit one vector kernel", "retain KV scale/zero-point")),
    RecipePattern("fused_add_rms_norm", "FusedAddRmsNorm",
                  ("add", "rms_norm"),
                  "exact residual add and norm epsilon; residual live-out retained",
                  ("emit one vector kernel", "write residual once")),
    RecipePattern("fused_cast_quant", "FusedCastDynamicQuant",
                  ("cast", "dynamic_quant"),
                  "exact cast rounding and quant group axis",
                  ("emit one vector kernel", "retain scale buffer")),
    RecipePattern("fused_vector_chain", "FusedVectorChain", (),
                  "declared elementwise/activation contracts in original order",
                  ("emit one vector kernel", "retain each public output")),
)
_VECTOR_PATTERN = _PATTERNS[-1]
_BY_SIGNATURE = {p.signature: p for p in _PATTERNS[:-1]}


@dataclass(frozen=True)
class FusionCandidate:
    candidate_id: str
    pass_name: str
    member_ops: tuple[str, ...]
    pattern: RecipePattern
    stage: Stage
    group: KernelGroup
    recipe: PlannedRecipe
    analysis: RegionAnalysis
    tile_rows: int
    hbm_before: int
    hbm_after: int
    launches_before: int
    launches_after: int
    retained_outputs: tuple[str, ...]
    stream_name: str = "main"
    reasons: tuple[str, ...] = ()
    matrix_work: tuple[tuple[str, float], ...] = ()
    phase_work: tuple[tuple[str, dict[str, float]], ...] = ()
    occurrence: str = ""
    conserved_hbm: int | None = None

    @property
    def hbm_saved(self) -> int:
        return self.hbm_before - self.hbm_after

    @property
    def recipe_id(self) -> str:
        # Occurrence-scoped: two independent chains of one pattern never share
        # a recipe/device-kernel-group identity, so a rejected trial cannot
        # overwrite an accepted program.
        if self.occurrence:
            return f"{self.pattern.recipe_id}:{self.occurrence}"
        return self.pattern.recipe_id

    @property
    def stream(self) -> str:
        return self.stream_name

    @property
    def numerical_policy(self) -> str:
        return self.analysis.numerical_policy

    @property
    def group_buffers(self) -> tuple[dict, ...]:
        return _group_buffers(self.analysis, self.member_ops[0])

    def kernel_spec(self) -> dict:
        outputs = [tensor_record(v) for v in _retained_values(self.analysis)]
        for record in outputs:
            record["materialized"] = True
        return {
            "fused_op_name": self.pattern.fused_op_name,
            "recipe_id": self.pattern.recipe_id,
            "member_ops": list(self.member_ops),
            "implementation_status": "analytical_prototype",
            "inputs": [tensor_record(v, "in") for v in _boundary_values(self.analysis)],
            "outputs": outputs,
            "numerical_policy": self.numerical_policy,
            "numerical_semantics": self.pattern.numerical_semantics,
            "schedule": "causally ordered phase stages",
            "phase_work": [{"phase": phase, "work": dict(work)}
                           for phase, work in self.phase_work],
            "ub": self.analysis.worksheet.record(),
            "group_buffers": list(self.group_buffers),
            "constraints": [c.record() for c in self.analysis.legality.constraints],
            "traffic": self.analysis.traffic.record(),
            "padding_overhead_bytes": self.analysis.padding_overhead_bytes,
            "hbm_bytes_saved": self.hbm_saved,
            "stream": self.stream,
            "capture_segment": None,
            "implementation_tasks": list(self.pattern.implementation_tasks),
        }


@dataclass(frozen=True)
class RejectedFusion:
    candidate_id: str
    pass_name: str
    member_ops: tuple[str, ...]
    reasons: tuple[str, ...]
    detail: str


def tensor_record(value, port: str = "out") -> dict:
    return {
        "value_id": value.value_id, "producer": value.producer, "port": port,
        "dtype": value.dtype, "layout": value.layout,
        "logical_shape": list(value.shape),
        "storage_shape": list(value.storage_shape or value.shape),
        "storage_bytes": value.nbytes, "alignment_bytes": value.alignment_bytes,
        "materialized": bool(value.materialized or value.public),
        "alias_of": value.alias_of, "state_version": value.state_version,
        "state_key": value.state_key,
        "role": "weight" if value.role == "parameter" else value.role,
    }


def _boundary_values(analysis: RegionAnalysis):
    ops = [analysis.ops[m] for m in analysis.members]
    boundary = {v for op in ops for v in op.inputs
                if analysis.values[v].producer is None
                or analysis.values[v].producer not in analysis.members}
    return [analysis.values[v] for v in boundary]


def _retained_values(analysis: RegionAnalysis):
    return [analysis.values[v] for v in analysis.retained_outputs]


def _match(graph: TypedGraph, members: tuple[str, ...]) -> RecipePattern:
    kinds = tuple(graph.ops[m].kind for m in members)
    return _BY_SIGNATURE.get(kinds, _VECTOR_PATTERN)


def _work_total(graph: TypedGraph, members: tuple[str, ...], resource: str) -> float:
    return float(sum(dict(graph.ops[m].work).get(resource, 0.0) for m in members))


def _group_buffers(analysis: RegionAnalysis,
                   owner_op: str = "") -> tuple[dict, ...]:
    owner = analysis.ops[owner_op].owner if owner_op in analysis.ops else 0
    return tuple({"id": a.name, "pool": a.pool, "bytes": a.bytes_per_instance,
                  "units": tuple(a.units) or (owner,), "lifetime": a.lifetime,
                  "instances": a.instances}
                 for a in (*analysis.worksheet.allocations,
                           *analysis.worksheet.cube_allocations))


def _matrix_work_ops(ops, members: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    totals: dict[str, float] = {}
    for op_id in members:
        op = ops[op_id]
        declared = op.param("matrix_flops")
        if isinstance(declared, dict) and declared:
            for dtype, amount in declared.items():
                totals[str(dtype)] = totals.get(str(dtype), 0.0) + float(amount)
        else:
            flops = float(dict(op.work).get("cube", 0.0))
            if flops:
                dtype = op.dtype or "bf16"
                totals[dtype] = totals.get(dtype, 0.0) + flops
    return tuple(sorted((dtype, amount) for dtype, amount in totals.items() if amount))


def _resources(phase_work, matrix_work) -> dict[str, float]:
    totals: dict[str, float] = {}
    for _, work in phase_work:
        for resource, amount in work.items():
            if resource == "cube":
                continue
            totals[resource] = totals.get(resource, 0.0) + float(amount)
    return totals


def _rebalance_hbm(phases, total: int | None):
    """Keep the emitted total HBM equal to the conserved declared bytes.

    The ledger's local reconstruction can under-count a globally declared
    weight; the difference is assigned to the causal operand-read phase (or
    scaled down when the declared total is smaller).
    """
    if total is None:
        return phases
    current = sum(float(work.get("hbm", 0.0)) for _, work in phases)
    if abs(current - float(total)) <= 1e-9:
        return phases
    result = [(phase, dict(work)) for phase, work in phases]
    if not result:
        return (("operand_read", {"hbm": float(total)}),)
    if float(total) > current:
        index = next((i for i, (phase, _) in enumerate(result)
                      if phase in {"operand_read", "vector"}), 0)
        result[index][1]["hbm"] = result[index][1].get("hbm", 0.0) + (
            float(total) - current)
    else:
        scale = float(total) / current if current else 0.0
        for _, work in result:
            if "hbm" in work:
                work["hbm"] *= scale
    return tuple(result)


def _planned_recipe(pattern: RecipePattern, analysis: RegionAnalysis,
                    matrix_work: tuple[tuple[str, float], ...],
                    phase_work: tuple[tuple[str, dict[str, float]], ...],
                    conserved_hbm: int | None = None) -> PlannedRecipe:
    resources = _resources(phase_work, matrix_work)
    if conserved_hbm is not None:
        resources["hbm"] = float(conserved_hbm)
    cube_work = Work(matrix=dict(matrix_work))
    work = Work(matrix=cube_work.matrix, resources=resources)

    def meters(work: dict[str, float]) -> dict[str, float]:
        work.pop("cube", None)
        return work

    def program(context):
        names = context.resource_names
        stages = []
        previous = ()
        for index, (phase, work) in enumerate(phase_work):
            payload = meters(dict(work))
            demand = {names[role]: amount for role, amount in payload.items()
                      if role in names and amount}
            if "cube" in work:
                for name, amount in context.bind(cube_work).items():
                    demand[name] = demand.get(name, 0.0) + amount
            if not demand and index != len(phase_work) - 1:
                continue
            name = phase if phase not in [s.name for s in stages] else f"{phase}:{index}"
            stages.append(Stage(name, demand, after=previous))
            previous = (name,)
        if not stages:
            raise ValueError("fused_recipe_has_no_stage")
        return KernelProgram(tuple(stages), minimum_ms=context.minimum_ms)

    def build(op, geometry):
        del geometry
        return OperatorPlan(work, program, metadata={
            "fusion_recipe": pattern.recipe_id, "phases": [p for p, _ in phase_work],
            "numerical_policy": analysis.numerical_policy})

    return PlannedRecipe(build)


def _make(pattern, analysis, members, pass_name, tile_rows, stream) -> FusionCandidate:
    matrix_work = _matrix_work_ops(analysis.ops, members)
    digest = hashlib.sha1(
        (f"{pattern.recipe_id}|{pass_name}|{'+'.join(members)}|{tile_rows}|"
         f"{analysis.traffic.baseline_bytes}|{analysis.traffic.candidate_bytes}|"
         f"{analysis.phase_work}").encode()
    ).hexdigest()[:8]
    occurrence = f"{members[0]}..{members[-1]}:{digest}"
    scoped_id = f"{pattern.recipe_id}:{occurrence}"
    group = KernelGroup([analysis.ops[m] for m in members],
                        buffers=_group_buffers(analysis, members[0]))
    group.entry_recipe = scoped_id
    conserved = analysis.traffic.candidate_bytes
    recipe = _planned_recipe(pattern, analysis, matrix_work, analysis.phase_work,
                             conserved)
    reasons = tuple(v.reason for v in analysis.traffic.values if v.external_live_out)
    return FusionCandidate(
        f"{pass_name}:{'+'.join(members)}:t{tile_rows}", pass_name, members, pattern,
        Stage(pattern.fused_op_name, dict(analysis.phase_work[0][1])
              if analysis.phase_work else {}),
        group, recipe, analysis, tile_rows, analysis.traffic.baseline_bytes,
        analysis.traffic.candidate_bytes, len(members), 1,
        analysis.retained_outputs, stream, reasons, matrix_work, analysis.phase_work,
        occurrence, conserved)


def candidate_with_hbm(candidate: FusionCandidate, conserved_hbm: int):
    """Rebuild a candidate's recipe so its emitted HBM equals declared-saved."""
    phases = _rebalance_hbm(candidate.analysis.phase_work, int(conserved_hbm))
    recipe = _planned_recipe(candidate.pattern, candidate.analysis,
                             candidate.matrix_work, phases, int(conserved_hbm))
    return replace(candidate, recipe=recipe, phase_work=phases,
                   conserved_hbm=int(conserved_hbm))


def synthesize_vector(graph: TypedGraph, members, *, tile_rows: int,
                      capacity_per_owner: int, buffer_factor: int = 2,
                      pass_name: str = "vector_chain",
                      tensor_parallel: int = 1,
                      usable_ub_bytes: int | None = None,
                      owner: int = 0,
                      allow_relaxed_numerics: bool = False,
                      reduction_merge: ReductionMerge | None = None,
                      channel_width: int | None = None
                      ) -> FusionCandidate | RejectedFusion:
    chosen = tuple(dict.fromkeys(members))
    analysis = analyze_region(graph, chosen, tile_rows=tile_rows,
                              channel_width=channel_width,
                              capacity_per_owner=capacity_per_owner,
                              buffer_factor=buffer_factor,
                              tensor_parallel=tensor_parallel,
                              usable_ub_bytes=usable_ub_bytes, owner=owner,
                              reduction_merge=reduction_merge,
                              allow_relaxed_numerics=allow_relaxed_numerics)
    candidate_id = f"{pass_name}:{'+'.join(chosen)}:t{tile_rows}"
    if not analysis.legality.ok:
        return RejectedFusion(candidate_id, pass_name, chosen, analysis.legality.reasons,
                              "region legality rejected")
    if not analysis.fits:
        return RejectedFusion(candidate_id, pass_name, chosen, ("ub_capacity_exceeded",),
                              f"UB {analysis.worksheet.ub_tile_bytes} > "
                              f"{analysis.worksheet.usable_bytes}")
    pattern = _match(graph, chosen)
    return _make(pattern, analysis, chosen, pass_name, tile_rows,
                 graph.ops[chosen[0]].stream or "main")


def _cube_guard(candidate_id, pass_name, chosen, capability, cube_resources,
                reason: str):
    if capability is not True:
        return RejectedFusion(candidate_id, pass_name, chosen,
                              (f"{reason}_capability_missing",),
                              "declared Cube boundary capability is not true")
    if cube_resources is None or not cube_resources.complete:
        return RejectedFusion(candidate_id, pass_name, chosen,
                              ("cube_resource_constraints_missing",),
                              "L1/L0A/L0B/L0C capacities were not declared")
    return None


def _outside_reader(graph: TypedGraph, value_id: str, chosen: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in graph.consumers(value_id) if c not in chosen)


def synthesize_cube_epilogue(graph: TypedGraph, cube_op: str, epilogue: tuple[str, ...], *,
                             tile_rows: int, capacity_per_owner: int,
                             capability: bool, retain_live_out: bool = False,
                             pass_name: str = "cube_boundary",
                             buffer_factor: int = 2,
                             cube_resources: CubeResources | None = None,
                             tensor_parallel: int = 1,
                             usable_ub_bytes: int | None = None,
                             owner: int = 0,
                             allow_relaxed_numerics: bool = False
                             ) -> FusionCandidate | RejectedFusion:
    chosen = tuple(dict.fromkeys((cube_op, *epilogue)))
    candidate_id = f"{pass_name}:{'+'.join(chosen)}:t{tile_rows}"
    guard = _cube_guard(candidate_id, pass_name, chosen, capability, cube_resources,
                        "cube_epilogue")
    if guard is not None:
        return guard
    analysis = analyze_region(graph, chosen, tile_rows=tile_rows,
                              capacity_per_owner=capacity_per_owner,
                              buffer_factor=buffer_factor, allow_cube=True,
                              cube_resources=cube_resources,
                              tensor_parallel=tensor_parallel,
                              usable_ub_bytes=usable_ub_bytes, owner=owner,
                              allow_relaxed_numerics=allow_relaxed_numerics)
    if not analysis.legality.ok:
        reasons = list(analysis.legality.reasons)
        outside = {v: _outside_reader(graph, v, chosen)
                   for v in graph.ops[cube_op].outputs}
        if any(readers for readers in outside.values()) and not retain_live_out:
            reasons.append("single_use_violated")
        return RejectedFusion(candidate_id, pass_name, chosen,
                              tuple(dict.fromkeys(reasons)),
                              "cube boundary legality rejected")
    order = {op: index for index, op in enumerate(analysis.causal_order)}
    if any(order.get(op, -1) <= order.get(cube_op, -1) for op in epilogue):
        return RejectedFusion(candidate_id, pass_name, chosen,
                              ("cube_epilogue_not_after_cube",),
                              "an epilogue operator is not causally after the Cube")
    intermediates = [v for v in graph.ops[cube_op].outputs
                     if any(v in graph.ops[op].inputs for op in epilogue)]
    if not intermediates:
        return RejectedFusion(candidate_id, pass_name, chosen,
                              ("cube_epilogue_not_consumed",),
                              "no Cube output feeds the epilogue")
    outside = {v: _outside_reader(graph, v, chosen) for v in intermediates}
    if any(readers for readers in outside.values()) and not retain_live_out:
        return RejectedFusion(candidate_id, pass_name, chosen, ("single_use_violated",),
                              "an intermediate Cube output also has an outside reader")
    if not analysis.fits:
        return RejectedFusion(candidate_id, pass_name, chosen, ("ub_capacity_exceeded",),
                              f"UB {analysis.worksheet.ub_tile_bytes} > "
                              f"{analysis.worksheet.usable_bytes}")
    pattern = RecipePattern("fused_cube_epilogue", "FusedCubeEpilogue",
                            tuple(graph.ops[m].kind for m in chosen),
                            "Cube projection then declared vector epilogue in causal "
                            "stage order; the L0C->UB handoff is explicit",
                            ("declare FixPipe path", "emit ordered Cube+Vector stages",
                             "retain live-out when externally read"))
    return _make(pattern, analysis, chosen, pass_name, tile_rows,
                 graph.ops[chosen[0]].stream or "main")


def synthesize_cube_prologue(graph: TypedGraph, prologue_op: str, cube_op: str, *,
                             tile_rows: int, capacity_per_owner: int,
                             capability: bool, retain_live_out: bool = False,
                             pass_name: str = "cube_boundary",
                             buffer_factor: int = 2,
                             cube_resources: CubeResources | None = None,
                             tensor_parallel: int = 1,
                             usable_ub_bytes: int | None = None,
                             owner: int = 0,
                             allow_relaxed_numerics: bool = False
                             ) -> FusionCandidate | RejectedFusion:
    chosen = (prologue_op, cube_op)
    candidate_id = f"{pass_name}:{prologue_op}+{cube_op}:t{tile_rows}"
    guard = _cube_guard(candidate_id, pass_name, chosen, capability, cube_resources,
                        "cube_prologue")
    if guard is not None:
        return guard
    if graph.ops[prologue_op].resource != "vector" or graph.ops[cube_op].resource != "cube":
        return RejectedFusion(candidate_id, pass_name, chosen,
                              ("cube_prologue_direction_invalid",),
                              "prologue must be Vector and its consumer Cube")
    analysis = analyze_region(graph, chosen, tile_rows=tile_rows,
                              capacity_per_owner=capacity_per_owner,
                              buffer_factor=buffer_factor, allow_cube=True,
                              cube_resources=cube_resources,
                              tensor_parallel=tensor_parallel,
                              usable_ub_bytes=usable_ub_bytes, owner=owner,
                              allow_relaxed_numerics=allow_relaxed_numerics)
    if not analysis.legality.ok:
        reasons = list(analysis.legality.reasons)
        intermediate = next((v for v in graph.ops[prologue_op].outputs
                             if v in graph.ops[cube_op].inputs), None)
        if intermediate is not None \
                and _outside_reader(graph, intermediate, chosen) \
                and not retain_live_out:
            reasons.append("single_use_violated")
        return RejectedFusion(candidate_id, pass_name, chosen,
                              tuple(dict.fromkeys(reasons)),
                              "cube prologue legality rejected")
    order = {op: index for index, op in enumerate(analysis.causal_order)}
    if order.get(prologue_op, -1) >= order.get(cube_op, -1):
        return RejectedFusion(candidate_id, pass_name, chosen,
                              ("cube_prologue_not_before_cube",),
                              "the Vector prologue is not causally before the Cube")
    intermediate = next((v for v in graph.ops[prologue_op].outputs
                         if v in graph.ops[cube_op].inputs), None)
    if intermediate is None:
        return RejectedFusion(candidate_id, pass_name, chosen,
                              ("cube_prologue_not_consumed",),
                              "Cube does not consume a prologue output")
    if _outside_reader(graph, intermediate, chosen) and not retain_live_out:
        return RejectedFusion(candidate_id, pass_name, chosen, ("single_use_violated",),
                              f"{intermediate} also has an outside reader")
    if not analysis.fits:
        return RejectedFusion(candidate_id, pass_name, chosen, ("ub_capacity_exceeded",),
                              f"UB {analysis.worksheet.ub_tile_bytes} > "
                              f"{analysis.worksheet.usable_bytes}")
    pattern = RecipePattern("fused_cube_prologue", "FusedCubePrologue",
                            tuple(graph.ops[m].kind for m in chosen),
                            "Vector prologue produces the Cube operand in a declared "
                            "AIV->AIC handoff; prologue write retained only for an "
                            "external reader",
                            ("declare prologue handoff path",
                             "emit ordered Vector+Cube stages"))
    return _make(pattern, analysis, chosen, pass_name, tile_rows,
                 graph.ops[chosen[0]].stream or "main")


def choose_vector_region(graph: TypedGraph, members, *, capacity_per_owner: int,
                         tiles: tuple[int, ...] = TILE_ROWS,
                         buffer_factor: int = 2,
                         tensor_parallel: int = 1,
                         usable_ub_bytes: int | None = None,
                         owner: int = 0,
                         allow_relaxed_numerics: bool = False,
                         reduction_merge: ReductionMerge | None = None,
                         channel_width: int | None = None
                         ) -> tuple[FusionCandidate | RejectedFusion, ...]:
    results = []
    for tile in tiles:
        if tile > graph.rows:
            continue
        results.append(synthesize_vector(
            graph, members, tile_rows=tile,
            capacity_per_owner=capacity_per_owner,
            buffer_factor=buffer_factor,
            tensor_parallel=tensor_parallel,
            usable_ub_bytes=usable_ub_bytes,
            owner=owner,
            allow_relaxed_numerics=allow_relaxed_numerics,
            reduction_merge=reduction_merge, channel_width=channel_width))
    return tuple(results)
