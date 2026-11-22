"""Tile liveness, legality and exact HBM accounting for fusion candidates.

[Role] Decide whether a connected operator region may execute as one kernel on
  one AIV owner and state the exact storage/access ledger it implies.
[Invariants] Legality derives from operator kind, shapes, dtypes, state/version,
  collective/capture boundaries, graph convexity and declared hardware
  resources, never from model names. Whole-width reductions cannot be split
  without a priced merge program. Tile bytes partition the physical storage
  exactly, so ``sum(tiles) == storage``; alignment padding is reported apart
  from logical HBM bytes. An interior value with an external reader keeps its
  write while proven internal rereads are eliminated.
[Timeouts & Locks] Finite graph arithmetic; no I/O, clocks, threads or locks.
[Side-Effect Policy] Pure functions over immutable records; no device context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from mindie_sim.prediction.fusion.graph import (
    CUBE_KINDS, Constraint, FeasibilityError, OpNode, PARAM_ROLES,
    REDUCTION_KINDS, TensorValue, TypedGraph, _align, _graph_index, _is_public,
    external_readers, is_connected, is_convex, region_causal_order, state_ports,
)

from mindie_sim.prediction.fusion.contract import (
    NON_DETERMINISTIC_ROUNDING, NUMERICAL_EXACT, ContractCheck, CubeResources,
    HandoffSplit, ReductionMerge, _contract, _cube_operands, handoff_split,
    local_nbytes, tile_partition_padding,
)
from mindie_sim.prediction.fusion.ub import UbWorksheet, ub_worksheet

# Re-exported compatibility surface: callers and the regression suite import
# these from the feasibility module as well as from their defining modules.
from mindie_sim.prediction.fusion.contract import tile_bytes  # noqa: F401
from mindie_sim.prediction.fusion.ub import (  # noqa: F401
    Allocation, LivenessPoint, TileLiveness, single_op_ub, tile_liveness,
)


_ANALYSIS_CAPACITY_PER_OWNER_BYTES = 262144


def relaxed_requires_opt_in(ops: Mapping[str, OpNode],
                            members: Iterable[str]) -> bool:
    """Whether a relaxed policy here is a caller decision, not a coverage gap.

    A reduction rewrite re-associates work and a stochastic rounding mode is
    irreproducible; both require ``allow_relaxed_numerics``.  A plain Cast
    whose operator declaration omits a rounding mode is a coverage gap that
    the existing contract reports as relaxed without claiming nondeterminism.
    """
    for op_id in members:
        op = ops.get(op_id)
        if op is None:
            continue
        if op.numerical_policy != NUMERICAL_EXACT:
            return True
        if op.kind in REDUCTION_KINDS:
            return True
        rounding = op.param("rounding")
        if rounding is not None \
                and str(rounding) in NON_DETERMINISTIC_ROUNDING:
            return True
    return False

@dataclass(frozen=True)
class Legality:
    ok: bool
    reasons: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    numerical_policy: str = NUMERICAL_EXACT
    reduction_width: int | None = None
    group_sizes: tuple[int, ...] = ()
    cube_operands: Mapping[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


def region_legality(graph: TypedGraph, members: Iterable[str], *,
                    allow_cube: bool = False,
                    cube_resources: CubeResources | None = None,
                    reduction_merge: ReductionMerge | None = None,
                    channel_width: int | None = None,
                    tile_rows: int = 1,
                    allow_relaxed_numerics: bool = False) -> Legality:
    chosen = tuple(dict.fromkeys(members))
    if len(chosen) < 2:
        return Legality(False, ("region_requires_two_operators",), ())
    if any(op not in graph.ops for op in chosen):
        raise FeasibilityError("region_member_unknown")
    reasons: list[str] = []
    if not is_connected(graph, chosen):
        reasons.append("region_not_connected")
    elif not is_convex(graph, chosen):
        reasons.append("region_not_convex")
    if len({(graph.ops[o].stage, graph.ops[o].phase) for o in chosen}) != 1:
        reasons.append("stage_or_phase_mismatch")
    if len({graph.ops[o].stream for o in chosen if graph.ops[o].stream}) > 1:
        reasons.append("stream_assignment_mismatch")
    params: dict[str, Any] = {}
    contracts: list[ContractCheck] = []
    operand_demand: dict[str, int] = {}
    for op_id in chosen:
        op = graph.ops[op_id]
        if op.source_boundary or op.opaque:
            reasons.append("source_kernel_boundary_opaque")
        if not op.semantic_complete:
            reasons.append("tensor_semantics_missing")
        if op.param("declared_shape_mismatch"):
            reasons.append("declared_output_shape_mismatch")
        if op.collective:
            reasons.append("cross_collective_boundary")
        if op.state_effect:
            reasons.append("state_version_barrier")
        if state_ports(graph, op):
            reasons.append("state_version_barrier")
        if op.capture_boundary:
            reasons.append("capture_boundary")
        if op.resource == "cube" and not allow_cube:
            reasons.append("cube_requires_boundary_pass")
        if op.resource == "host":
            reasons.append("host_scalar_dependency")
        check = _contract(graph, op)
        contracts.append(check)
        reasons.extend(check.reasons)
        if op.kind in CUBE_KINDS:
            if not allow_cube:
                reasons.append("cube_requires_boundary_pass")
            if cube_resources is None or not cube_resources.complete:
                reasons.append("cube_resource_constraints_missing")
            else:
                demand = _cube_operands(graph, op, tile_rows, cube_resources)
                for pool, amount in demand.items():
                    operand_demand[pool] = max(operand_demand.get(pool, 0), amount)
                    if amount > getattr(cube_resources, pool):
                        reasons.append(f"cube_{pool}_capacity_exceeded")
        for key in ("epsilon", "axis", "group_size", "rounding", "rotate_style"):
            value = op.param(key)
            if value is not None and params.setdefault(key, value) != value:
                reasons.append("numerical_contract_mismatch")
    reduction_width = max((c.reduction_width or 0 for c in contracts), default=0) or None
    reduction_kinds = {graph.ops[o].kind for o in chosen} & REDUCTION_KINDS
    merge_tiles = merge_bytes = merge_ops = None
    if reduction_merge is not None:
        # A supplied merge contract is validated even when the current channel
        # width needs no split: an unused NaN/Inf program is still invalid and
        # must never reach a worksheet or ledger.
        merge_tiles = reduction_merge.tiles
        merge_bytes = reduction_merge.merge_bytes
        merge_ops = reduction_merge.merge_ops
        if not isinstance(merge_tiles, int) or isinstance(merge_tiles, bool) \
                or not math.isfinite(float(merge_tiles)) or merge_tiles < 2:
            reasons.append("reduction_merge_program_invalid")
        if not isinstance(merge_bytes, (int, float)) \
                or isinstance(merge_bytes, bool) \
                or not math.isfinite(float(merge_bytes)) or merge_bytes <= 0:
            reasons.append("reduction_merge_program_invalid")
        if not isinstance(merge_ops, (int, float)) \
                or isinstance(merge_ops, bool) \
                or not math.isfinite(float(merge_ops)) or merge_ops <= 0:
            reasons.append("reduction_merge_work_missing")
    if reduction_kinds and channel_width is not None and reduction_width \
            and channel_width < reduction_width:
        if reduction_merge is None:
            reasons.append("whole_width_reduction_requires_merge")
        else:
            # The merge program must describe the actual channel partition:
            # one cross-tile merge per cut.
            channel_tiles = int(math.ceil(
                reduction_width / max(1, int(channel_width))))
            if not isinstance(merge_tiles, int) or isinstance(merge_tiles, bool) \
                    or not math.isfinite(float(merge_tiles)) \
                    or merge_tiles != channel_tiles:
                reasons.append("reduction_merge_tiles_mismatch")
    # A contraction may expose only the interfaces its members really have.
    # Multiple primary retained outputs of different dtype/shape/layout cannot
    # be represented by one surviving OpSpec port; reject instead of silently
    # changing an external tensor interface.
    retained_external = external_readers(graph, chosen)
    retained = tuple(
        value_id for op_id in chosen for value_id in graph.ops[op_id].outputs
        if value_id in retained_external or _is_public(graph, value_id))
    primary = []
    for value_id in retained:
        value = graph.values[value_id]
        producer = graph.ops.get(value.producer) if value.producer else None
        if producer is not None and producer.outputs \
                and producer.outputs[0] == value_id:
            primary.append(value)
    interfaces = {(value.dtype, tuple(value.shape), value.layout)
                  for value in primary}
    if len(interfaces) > 1:
        reasons.append("retained_output_interface_mismatch")
    unique = tuple(dict.fromkeys(reasons))
    constraints = [Constraint(r, "violated", f"fusion legality:{r}") for r in unique]
    convex_ok = not ({"region_not_connected", "region_not_convex"} & set(unique))
    constraints.append(Constraint("region_convexity", "satisfied" if convex_ok else "violated",
                                  "forward-convex region whose contraction cannot cycle"))
    policies = {c.numerical_policy for c in contracts}
    policy = NUMERICAL_EXACT if policies == {NUMERICAL_EXACT} and not unique else "relaxed"
    # A relaxed reduction rewrite needs opt-in because it changes the
    # association order.  A stochastic Cast/KV Cast is equally
    # nondeterministic but contains no reduction, so the operator's declared
    # rounding mode is checked independently of reduction membership.
    relaxed_needs_opt_in = relaxed_requires_opt_in(graph.ops, chosen)
    if policy != NUMERICAL_EXACT and relaxed_needs_opt_in \
            and not allow_relaxed_numerics:
        unique = (*unique, "relaxed_numerics_not_allowed")
        constraints.append(Constraint("relaxed_numerics", "violated",
                                      "caller did not set allow_relaxed_numerics"))
    if not unique:
        constraints.extend([
            Constraint("stage_phase", "satisfied", "one stage and one phase"),
            Constraint("dtype_and_layout", "satisfied",
                       "declared dtypes, layouts and storage shapes retained"),
            Constraint("numerical_contract",
                       "satisfied" if policy == NUMERICAL_EXACT else "relaxed",
                       "declared epsilon/quant axis/rounding/saturation retained"),
        ])
    return Legality(not unique, unique, tuple(constraints), policy,
                    reduction_width, tuple(next((c.group_sizes for c in contracts
                                                 if c.group_sizes), ())),
                    operand_demand)


@dataclass(frozen=True)
class TrafficValue:
    value_id: str
    baseline_write_bytes: int
    candidate_write_bytes: int
    baseline_read_bytes: int
    candidate_read_bytes: int
    new_spill_bytes: int
    external_live_out: bool
    saved_bytes: int
    reason: str

    def record(self) -> dict:
        return {"value_id": self.value_id, "scope": "changed_physical_accesses",
                "baseline_write_bytes": self.baseline_write_bytes,
                "candidate_write_bytes": self.candidate_write_bytes,
                "baseline_read_bytes": self.baseline_read_bytes,
                "candidate_read_bytes": self.candidate_read_bytes,
                "new_spill_bytes": self.new_spill_bytes,
                "external_live_out": self.external_live_out,
                "saved_bytes": self.saved_bytes, "reason": self.reason}


@dataclass(frozen=True)
class TrafficLedger:
    values: tuple[TrafficValue, ...]
    baseline_bytes: int
    candidate_bytes: int
    saved_bytes: int
    padding_overhead_bytes: int = 0
    merge_overhead_bytes: int = 0

    def record(self) -> list[dict]:
        return [v.record() for v in self.values]

    def assert_conserved(self, selected_baseline: int | None = None) -> None:
        base = sum(v.baseline_write_bytes + v.baseline_read_bytes for v in self.values)
        cand = sum(v.candidate_write_bytes + v.candidate_read_bytes
                   for v in self.values)
        spill = sum(v.new_spill_bytes for v in self.values)
        if base != self.baseline_bytes or cand + spill != self.candidate_bytes:
            raise FeasibilityError("traffic_ledger_not_reconciled")
        # Per-value savings are net of merge/spill overhead, so their sum is
        # exactly the measured traffic delta rather than a pre-overhead total.
        if sum(v.saved_bytes for v in self.values) != self.saved_bytes:
            raise FeasibilityError("traffic_ledger_savings_not_reconciled")
        if self.saved_bytes != self.baseline_bytes - self.candidate_bytes:
            raise FeasibilityError("traffic_ledger_delta_not_conserved")
        if selected_baseline is not None and selected_baseline != base:
            raise FeasibilityError(
                f"traffic_ledger_baseline_mismatch:{selected_baseline}:{base}")


_ZERO_BYTES = 0; _UNIT_COUNT = 1


def traffic_ledger(graph: TypedGraph, members: Iterable[str], *, tile_rows: int,
                   channel_width: int | None = None, spill_bytes: int = _ZERO_BYTES,
                   rows: int | None = None, tensor_parallel: int = _UNIT_COUNT,
                   reduction_merge: ReductionMerge | None = None) -> TrafficLedger:
    chosen = tuple(dict.fromkeys(members))
    total_rows = int(rows or graph.rows)
    tile_rows = max(_UNIT_COUNT, min(int(tile_rows), total_rows))
    external = external_readers(graph, chosen)
    readers: dict[str, int] = {}
    for op_id in chosen:
        for value_id in graph.ops[op_id].inputs:
            readers[value_id] = readers.get(value_id, _ZERO_BYTES) + _UNIT_COUNT
    produced = {v for op_id in chosen for v in graph.ops[op_id].outputs}
    value_rank = _graph_index(graph)["value_rank"]
    touched = sorted(produced | readers.keys(), key=value_rank.__getitem__)
    records: list[TrafficValue] = []
    spill_assigned = False
    padding = 0
    for value_id in touched:
        value = graph.values[value_id]
        per_invocation = value.role in {"weight", "state", "parameter", "scale"}
        local = local_nbytes(value, tensor_parallel)
        # Tiles partition the physical storage exactly, so every sum over
        # tiles is ``local``.  Only alignment padding needs the partition
        # distribution, and that is O(1) arithmetic.
        uses = readers.get(value_id, 0)
        baseline_w = baseline_r = candidate_w = candidate_r = 0
        alias_write = bool(value.alias_of) and value.role not in {"state", "output"}
        if value_id in produced and not alias_write:
            baseline_w += local
        baseline_r += local * uses
        if (value_id in produced and not alias_write
                and (value_id in external or _is_public(graph, value_id)
                     or value.role in {"state", "output"})):
            candidate_w += local
        if uses and (value_id not in produced or value.role == "state"):
            candidate_r += local
        if value.role not in PARAM_ROLES:
            padding += ((_align(local, value.alignment_bytes) - local)
                        if per_invocation else
                        tile_partition_padding(value, tile_rows, total_rows))
        spill = 0
        if value_id in produced and spill_bytes and not spill_assigned:
            spill, spill_assigned = spill_bytes, True
        saved = baseline_w + baseline_r - candidate_w - candidate_r - spill
        if saved < 0:
            raise FeasibilityError(f"fusion_traffic_ledger_would_add_bytes:{value_id}")
        if value_id in produced and (value_id in external or _is_public(graph, value_id)):
            reason = "external_live_out_kept_internal_rereads_eliminated"
        elif per_invocation:
            reason = "parameter_or_state_reads_retained"
        elif value_id in produced:
            reason = "interior_materialization_eliminated"
        else:
            reason = "boundary_input_loaded_once"
        records.append(TrafficValue(value_id, baseline_w, candidate_w, baseline_r,
                                    candidate_r, spill, value_id in external, saved, reason))
    merge_overhead = 0
    if reduction_merge is not None and reduction_merge.merge_bytes > 0:
        merge = int(reduction_merge.merge_bytes)
        merge_overhead = 2 * merge
        records.append(TrafficValue(
            "reduction_merge", 0, merge, 0, merge, 0, False, -merge_overhead,
            "priced_cross_tile_merge"))
    baseline = sum(r.baseline_write_bytes + r.baseline_read_bytes for r in records)
    candidate = sum(r.candidate_write_bytes + r.candidate_read_bytes + r.new_spill_bytes
                    for r in records)
    if baseline - candidate < 0:
        raise FeasibilityError("fusion_traffic_ledger_nonconservative")
    ledger = TrafficLedger(tuple(records), baseline, candidate, baseline - candidate,
                           padding, merge_overhead)
    ledger.assert_conserved()
    return ledger


@dataclass(frozen=True)
class RegionAnalysis:
    members: tuple[str, ...]
    legality: Legality
    external: tuple[str, ...]
    interior: tuple[str, ...]
    retained_outputs: tuple[str, ...]
    tile_rows: int
    channel_width: int
    worksheet: UbWorksheet
    traffic: TrafficLedger
    live_tile_bytes: int
    values: Mapping[str, TensorValue]
    ops: Mapping[str, OpNode]
    numerical_policy: str = NUMERICAL_EXACT
    causal_order: tuple[str, ...] = ()
    phase_work: tuple[tuple[str, dict[str, float]], ...] = ()
    reduction_merge: ReductionMerge | None = None
    handoff_bytes: int = 0
    handoff_operand_bytes: int = 0
    handoff_output_bytes: int = 0

    @property
    def fits(self) -> bool:
        return self.worksheet.fits

    @property
    def padding_overhead_bytes(self) -> int:
        return self.traffic.padding_overhead_bytes


def _phase_work(graph: TypedGraph, chosen: tuple[str, ...],
                traffic: TrafficLedger,
                reduction_merge: ReductionMerge | None = None,
                handoff: HandoffSplit | None = None
                ) -> tuple[tuple[str, dict[str, float]], ...]:
    """Declared service split by causal phase; no resource is invented.

    Cube regions order operand HBM/mte2 reads, the Vector prologue, the
    Vector-to-Cube operand handoff (MTE1/L1), the Cube body, the Cube-to-Vector
    output handoff (FixPipe/UB), the Vector epilogue and the output HBM/mte3
    write.  A whole-width reduction orders partial tile reduction, the
    cross-tile merge, then the quant/Cast finalize that writes the output.
    The traffic ledger remains the only HBM authority; declared non-HBM work
    survives per phase.
    """
    if handoff is None:
        handoff = HandoffSplit(0, 0)
    merge_bytes = 0
    merge_ops = 0.0
    if reduction_merge is not None \
            and isinstance(reduction_merge.merge_ops, (int, float)) \
            and math.isfinite(float(reduction_merge.merge_ops)) \
            and reduction_merge.merge_ops > 0 \
            and isinstance(reduction_merge.merge_bytes, (int, float)) \
            and math.isfinite(float(reduction_merge.merge_bytes)) \
            and reduction_merge.merge_bytes > 0:
        merge_bytes = int(reduction_merge.merge_bytes)
        merge_ops = float(reduction_merge.merge_ops)
    reads = float(sum(value.candidate_read_bytes for value in traffic.values)) \
        - merge_bytes
    writes = float(sum(value.candidate_write_bytes + value.new_spill_bytes
                       for value in traffic.values)) - merge_bytes
    reads = max(0.0, reads)
    writes = max(0.0, writes)
    cube_indices = [index for index, op_id in enumerate(chosen)
                    if graph.ops[op_id].resource == "cube"]
    first_cube = min(cube_indices) if cube_indices else len(chosen)
    last_reduction = max((index for index, op_id in enumerate(chosen)
                          if graph.ops[op_id].kind in REDUCTION_KINDS),
                         default=-1)
    split_reduction = (not cube_indices and reduction_merge is not None
                       and last_reduction >= 0)
    phases: dict[str, dict[str, float]] = {}
    for index, op_id in enumerate(chosen):
        op = graph.ops[op_id]
        if op.resource == "cube":
            phase = "cube"
        elif not cube_indices:
            if split_reduction:
                phase = "reduce" if index <= last_reduction else "finalize"
            else:
                phase = "vector"
        elif index < first_cube:
            phase = "prologue"
        else:
            phase = "epilogue"
        target = phases.setdefault(phase, {})
        for resource, amount in op.work:
            if resource == "hbm":
                continue
            if cube_indices and resource in {"mte2", "mte3"}:
                # Operand copies are part of the causal read, result copies
                # part of the causal write; neither belongs to the Cube body.
                causal = "operand_read" if resource == "mte2" else "output_write"
                bucket = phases.setdefault(causal, {})
                bucket[resource] = bucket.get(resource, 0.0) + float(amount)
                continue
            target[resource] = target.get(resource, 0.0) + float(amount)
    if cube_indices:
        if reads > 0:
            phases.setdefault("operand_read", {})["hbm"] = reads
        if handoff.operand > 0:
            # Without a Vector prologue the operand transfer is the first
            # part of the read phase; with one it must follow the prologue.
            destination = ("prologue_handoff" if "prologue" in phases
                           else "operand_read")
            bucket = phases.setdefault(destination, {})
            bucket["mte1"] = bucket.get("mte1", 0.0) + float(handoff.operand)
        if writes > 0:
            phases.setdefault("output_write", {})["hbm"] = writes
        if handoff.output > 0:
            # Cube -> Vector output handoff: strictly after the Cube body and
            # before the Vector epilogue that consumes it.
            bucket = phases.setdefault("handoff", {})
            bucket["mte1"] = bucket.get("mte1", 0.0) + float(handoff.output)
    elif split_reduction:
        if reads > 0:
            phases.setdefault("operand_read", {})["hbm"] = reads
        if writes > 0:
            phases.setdefault("output_write", {})["hbm"] = writes
    else:
        body = phases.setdefault("vector", {})
        body["hbm"] = body.get("hbm", 0.0) + reads + writes
    if merge_bytes > 0:
        merge = phases.setdefault("merge", {})
        merge["vector"] = merge.get("vector", 0.0) + merge_ops
        merge["mte1"] = merge.get("mte1", 0.0) + 2.0 * float(merge_bytes)
        # The partial-reduction workspace is written at the end of ``reduce``
        # and read by ``merge``; neither access belongs to the boundary input
        # read or the final output write.
        merge["hbm"] = merge.get("hbm", 0.0) + float(merge_bytes)
        partial = phases.setdefault("reduce" if split_reduction else "merge", {})
        partial["hbm"] = partial.get("hbm", 0.0) + float(merge_bytes)
    for work in phases.values():
        for resource in list(work):
            if abs(work[resource]) <= 1e-12:
                del work[resource]
    order = ("operand_read", "prologue", "prologue_handoff", "cube", "handoff",
             "epilogue", "vector", "reduce", "merge", "finalize", "output_write")
    return tuple((phase, phases[phase]) for phase in order if phase in phases)


def analyze_region(graph: TypedGraph, members: Iterable[str], *, tile_rows: int,
                   channel_width: int | None = None, capacity_per_owner: int = _ANALYSIS_CAPACITY_PER_OWNER_BYTES,
                   buffer_factor: int = 2, allow_cube: bool = False,
                   tensor_parallel: int = 1,
                   cube_resources: CubeResources | None = None,
                   usable_ub_bytes: int | None = None,
                   reduction_merge: ReductionMerge | None = None,
                   owner: int = 0,
                   allow_relaxed_numerics: bool = False,
                   uses_simt: bool = False,
                   simt_ub_carveout_bytes: int = 0) -> RegionAnalysis:
    chosen = tuple(dict.fromkeys(members))
    causal = region_causal_order(graph, chosen)
    raw_width = channel_width
    if raw_width is None:
        raw_width = max((graph.values[v].shape[-1] for op in chosen
                         for v in (*graph.ops[op].inputs, *graph.ops[op].outputs)
                         if graph.values[v].role not in {"weight", "parameter", "state"}),
                        default=1)
    legality = region_legality(graph, causal, allow_cube=allow_cube,
                               cube_resources=cube_resources,
                               reduction_merge=reduction_merge,
                               channel_width=raw_width, tile_rows=tile_rows,
                               allow_relaxed_numerics=allow_relaxed_numerics)
    max_width = max((graph.values[v].shape[-1] for op in causal
                     for v in (*graph.ops[op].inputs, *graph.ops[op].outputs)
                     if graph.values[v].role not in PARAM_ROLES), default=1)
    width = max(1, min(int(raw_width), int(max_width)))
    external = external_readers(graph, chosen)
    interior = tuple(v for op in causal for v in graph.ops[op].outputs
                     if v not in external and not _is_public(graph, v))
    retained = tuple(v for op in causal for v in graph.ops[op].outputs
                     if v in external or _is_public(graph, v))
    split_reduction = bool(legality.reduction_width and channel_width is not None
                           and width < legality.reduction_width)
    priced_merge = (reduction_merge if legality.ok and split_reduction else None)
    ledger = traffic_ledger(graph, causal, tile_rows=tile_rows, channel_width=width,
                            tensor_parallel=tensor_parallel,
                            reduction_merge=priced_merge)
    worksheet = ub_worksheet(graph, causal, tile_rows=tile_rows, channel_width=width,
                             capacity_per_owner=capacity_per_owner,
                             buffer_factor=buffer_factor, allow_cube=allow_cube,
                             usable_bytes=usable_ub_bytes, owner=owner,
                             reduction_merge=priced_merge,
                             cube_resources=cube_resources,
                             uses_simt=uses_simt,
                             simt_ub_carveout_bytes=simt_ub_carveout_bytes)
    split = handoff_split(graph, causal, cube_resources,
                          max(1, min(tile_rows, graph.rows)))
    return RegionAnalysis(causal, legality, tuple(external), interior, retained,
                          max(1, min(tile_rows, graph.rows)), width, worksheet, ledger,
                          worksheet.live_tile_bytes, graph.values, graph.ops,
                          legality.numerical_policy, causal,
                          _phase_work(graph, causal, ledger, priced_merge, split),
                          priced_merge, split.total, split.operand, split.output)
