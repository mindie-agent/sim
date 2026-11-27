"""Mechanical verification of rewrite plans and replay evidence.

[Role] Report structural legality, capacity, overlap, conservation and bounds.
[Invariants] Virtual intervals and memory peaks are measured per placement.
  A certified lower bound above replay wall time is a soundness failure.
[Timeouts & Locks] Bounded by the graph and supplied evidence, with no locks.
[Side-Effect Policy] Checks only. No input, device or runtime state is mutated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from itertools import combinations
from math import fsum, isclose, isfinite
from typing import Any, Literal

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.prediction.evaluation import ACTIVE_DEADLINE_MONO
from mindie_sim.prediction.execution.operator_replay import operator_costs
from mindie_sim.prediction.results import FusedPipelineDescriptor, OpOccurrence, OperatorSequenceRow, StepEvaluation
from mindie_sim.prediction.runtime.reload_bounds import compute_compulsory_hbm_cut, measure_logical_bytes
from mindie_sim.prediction.spine import _parse_occurrence, _wave_breakdown_from_report, _work_items
from mindie_sim.prediction.runtime.accel_tables import (
    MAX_L as ACCEL_MAX_L, MAX_R as ACCEL_MAX_R, MAX_T as ACCEL_MAX_T, MAX_U as ACCEL_MAX_U,
)
from mindie_sim.prediction.runtime.tile_solver import TilePipelineSpec
from mindie_sim.prediction.streams import MAIN_STREAM, bind_stream_owners, proportional_coupled_placement
from mindie_sim.prediction.execution.weight_prefetch import residency_peak
from mindie_sim.schema.abi import validate_fusion_tier_verdict
from mindie_sim.schema.common import ModelingError
from mindie_sim.templates.contracts import DslKernelSpec

from ._oracle_prefetch import _displaced_weight_mte2
from .algebra import AlgebraicTrait, _resolve_soc_arch, classify_op_traits
from .apply import (
    AppliedPlan, ResolvedScenario, _check_stream, _convex, _dependencies, _die,
    _lane_pool_unit_ids, _path, _solve_tiles, _successors, _topological,
)
from .catalog_replay import (
    _canon_digest_value,
    _catalog_lowerer,
    _declared_regions,
    _matrix_problem,
    _replay_ops,
    replay_catalog_binding,
    summary_matches_canonical,
)
from .dual_cert import (
    build_dual_inputs_from_scenario,
    check_optimality_certificate,
)
from .schema import FuseRegion, HoistOrDeduplicate, QuantizeContract, SelectTemplate, _flat

FAKE_OVERLAP_THRESHOLD_PCT = 5.0
SHIPPING_REGRESSION_TOLERANCE_NS = 1000
NS_PER_MS = 1_000_000
SERVICE_CONSERVATION_REL_TOL = 1e-12
SERVICE_CONSERVATION_ABS_TOL = 1e-6
_CAPACITY_POOLS = frozenset({"ub", "l1", "l0a", "l0b", "l0c", "l2", "hbm"})
_NON_MATRIX_IMPLEMENTATIONS = frozenset({
    "TiledPointwise", "RingCollective", "PairwiseCollective", "OneShotCollective",
})
_MATRIX_SERVICE_PREFIXES = ("matrix@", "cube@", "simt@")
_TRANSFER_SERVICE = {"mte2": ("mte2@", "mte2_aic@", "mte2_aiv@"), "hbm": ("memory@",)}
_TEMPLATE_SERVICE = {**_TRANSFER_SERVICE, "cube": _MATRIX_SERVICE_PREFIXES,
                     "vector": ("vector@",), "mte3": ("mte3@",), "link": ("link:",)}
_COHORT_PREFIX = "rewrite_cohort"
_MAX_CATALOG_MEMO_ENTRIES = 128
_RANK_LABEL_METADATA_KEYS = frozenset({
    "ep_rank", "rank", "local_rank", "dp_rank", "tp_rank",
    "pp_rank", "replica", "die_id", "die_rank", "owner_rank",
})
_CATLASS_SPEC_META_KEYS = (
    "l1_tile_shape", "l0_tile_shape", "mutex_stages", "simt_thread_block_dim",
    "swizzle_offsets", "evg_tree", "enable_unit_flag", "l1_resident_slot_match",
)


def _norm_catalog_op(item: Any) -> tuple[Any, ...]:
    meta = tuple(
        sorted(
            (k, _canon_digest_value(v))
            for k, v in item.metadata
            if k not in _RANK_LABEL_METADATA_KEYS
        )
    )
    return (
        item.op_id, item.family, item.recipe, item.stage, item.layer,
        item.dtype, item.local_shape, item.padded_shape, item.launch_shape,
        item.layout, item.flops, item.matrix_flops, item.vector_ops,
        item.hbm_bytes, item.mte2_bytes, item.mte3_bytes, item.link_bytes,
        item.source_boundary, meta,
    )


def _norm_catalog_datapath(path: Any) -> Any:
    return replace(path, ub_peer_edges=()) if hasattr(path, "ub_peer_edges") else path


def _can_retag_problem_ids(problem: Any) -> bool:
    raw = (problem.a_id, problem.b_id, problem.c_id, problem.indices_id)
    distinct = tuple(x for x in raw if x)
    if len(set(distinct)) != len(distinct):
        return False
    return not any(x != y and y.startswith(x + ":") for x in distinct for y in distinct)


def _alloc_id_replacements(old_problem: Any, new_problem: Any) -> tuple[tuple[str, str], ...]:
    if old_problem is None or new_problem is None:
        return ()
    pairs = (
        (old_problem.a_id, new_problem.a_id),
        (old_problem.b_id, new_problem.b_id),
        (old_problem.c_id, new_problem.c_id),
        (old_problem.indices_id, new_problem.indices_id),
    )
    return tuple((src, dst) for src, dst in pairs if src and dst and src != dst)


def _remap_alloc_id(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for src, dst in replacements:
        if value == src:
            return dst
        if value.startswith(src + ":"):
            return dst + value[len(src):]
    return value


def _retag_access_record(rec: Any, target_die: int, replacements: tuple[tuple[str, str], ...]) -> Any:
    if rec is None:
        return None
    pat = rec.pattern
    new_tensor = _remap_alloc_id(pat.tensor_id, replacements) if replacements else pat.tensor_id
    rkey = pat.reuse_key
    if rkey is not None:
        rk_backing = _remap_alloc_id(rkey.backing_allocation, replacements) if replacements else rkey.backing_allocation
        rk_stream = (
            _remap_alloc_id(rkey.reuse_stream, replacements)
            if (replacements and rkey.reuse_stream)
            else rkey.reuse_stream
        )
        rkey = replace(rkey, die=target_die, backing_allocation=rk_backing, reuse_stream=rk_stream)
    return replace(rec, pattern=replace(pat, die=target_die, tensor_id=new_tensor, reuse_key=rkey))


def _retag_summary_for_site(
    raw_summary: Any,
    op_ids: tuple[str, ...],
    *,
    die: int,
    replica: int,
    stage: int,
    is_dsl: bool,
    op_id: str,
    alloc_replacements: tuple[tuple[str, str], ...] = (),
) -> Any:
    """Stamp exact per-site ``(die, replica, pp_stage, region_id, alloc_ids)`` on a pure summary."""
    old_rid = raw_summary.region_id
    old_prefix = f"{old_rid}:"
    new_prefix = f"{op_id}:"

    def _retag_work(item: Any) -> Any:
        wid = item.work_id
        if old_rid != op_id and wid.startswith(old_prefix):
            wid = new_prefix + wid[len(old_prefix):]
        return replace(item, work_id=wid, die=die, replica=replica, pp_stage=stage)

    def _retag_obligations(items: tuple[Any, ...]) -> tuple[Any, ...]:
        if is_dsl:
            return tuple(
                replace(item, die=die, allocation_id=f"{op_id}:{item.allocation_id}")
                for item in items
            )
        if alloc_replacements:
            return tuple(
                replace(item, die=die, allocation_id=_remap_alloc_id(item.allocation_id, alloc_replacements))
                for item in items
            )
        return tuple(replace(item, die=die) for item in items)

    return replace(
        raw_summary,
        region_id=op_id,
        op_ids=op_ids,
        semantic_work=tuple(_retag_work(item) for item in raw_summary.semantic_work),
        read_obligations=_retag_obligations(raw_summary.read_obligations),
        write_obligations=_retag_obligations(raw_summary.write_obligations),
    )


def _retag_catalog_program_for_site(
    program: Any,
    target_die: int,
    raw_summary_for_site: Any,
    *,
    new_problem: Any = None,
    alloc_replacements: tuple[tuple[str, str], ...] = (),
) -> Any:
    """Clone a pure lowered ``with_context`` catalog program onto ``(target_die, new_problem)``."""
    same_rid = getattr(getattr(program, "summary", None), "region_id", None) == raw_summary_for_site.region_id
    if getattr(program, "die", None) == target_die and not alloc_replacements and same_rid:
        return program
    if (
        getattr(program, "context", None) is not None
        or getattr(program, "hbm_bus", None) is not None
        or getattr(program, "initial_state", None) is not None
        or getattr(program, "live_state", None) is not None
    ):
        return None
    eff_problem = new_problem if new_problem is not None else getattr(program, "problem", None)
    if hasattr(program, "waves"):
        retagged_waves = tuple(
            tuple(
                replace(
                    cohort,
                    phases=tuple(
                        replace(
                            phase,
                            panels=tuple(
                                replace(
                                    panel,
                                    gm_a=_retag_access_record(panel.gm_a, target_die, alloc_replacements),
                                    scale_a=_retag_access_record(panel.scale_a, target_die, alloc_replacements),
                                    gm_b=_retag_access_record(panel.gm_b, target_die, alloc_replacements),
                                    scale_b=_retag_access_record(panel.scale_b, target_die, alloc_replacements),
                                )
                                for panel in phase.panels
                            ),
                        )
                        for phase in cohort.phases
                    ),
                    drain=_retag_access_record(cohort.drain, target_die, alloc_replacements),
                )
                for cohort in wave_cohorts
            )
            for wave_cohorts in program.waves
        )
        return replace(
            program, die=target_die, problem=eff_problem,
            waves=retagged_waves, summary=raw_summary_for_site,
        )
    if type(program).__name__ == "_StreamingCatlassProgram":
        return replace(program, die=target_die, problem=eff_problem, summary=raw_summary_for_site)
    return None


def _resolve_catalog_request(op: Any, template: Any, applied: AppliedPlan, atomic_entry: Any):
    if template and template.tier in {"K2", "K3"} and template.template_id.startswith(("catlass.", "dsl.")):
        return template.tier, template.template_id, template.params, (op,), False
    if op.family == "rewrite_fused" and dict(op.metadata).get("fusion_tier") == "K3_dsl_pipeline":
        edit = next(edit for edit in applied.plan.edits if op.op_id == f"rewrite:{edit.edit_id}")
        operation = edit.operation
        if (
            not isinstance(operation, FuseRegion)
            or len(operation.kernel_boundaries) != 1
            or any(ref.kind != "atom" for ref in operation.members)
        ):
            return None
        ids = {ref.id for ref in operation.members}
        members = tuple(member for member in atomic_entry.ops if member.op_id in ids)
        boundary, = operation.kernel_boundaries
        identifier = boundary.abi_id
        if identifier.startswith("dsl."):
            return "K3", identifier, boundary.params, members, False
        if any(decl["region_id"] == identifier for _, _, decl in _declared_regions(atomic_entry.ops)):
            return "K3", identifier, boundary.params, members, True
    return None


def _memoized_catalog_lowering(
    scenario: ResolvedScenario,
    atomic_entry: Any,
    op: Any,
    template: Any,
    applied: AppliedPlan,
    stream: Any,
    catalog_memo: dict[Any, Any],
    *,
    need_program: bool,
) -> tuple[Any, Any, Any, Any, tuple[int, int], int, Callable[..., Any], Any] | None:
    """Resolve one K2/K3 site via cross-layer/cross-replica memo with exact per-site retagging."""
    req = _resolve_catalog_request(op, template, applied, atomic_entry)
    if req is None:
        return None
    tier, tpl_id, params, members, is_explicit_region = req
    if is_explicit_region:
        bound = replay_catalog_binding(scenario, atomic_entry, op, template, applied, stream)
        if bound is None:
            return None
        recipe, summary = bound
        return ("explicit", recipe, summary, None, (0, 0), 0, lambda *a, **kw: None, None)
    path = _path(scenario, atomic_entry.replica, op.stage)
    limits = (path.aic_count, path.aiv_count)
    if stream is not None:
        owners = bind_stream_owners(stream, path)[dict(stream.assignments).get(op.op_id, MAIN_STREAM)]
        limits = (owners.aic_limit, owners.aiv_limit)
    die = _die(scenario, atomic_entry.replica, op.stage).rank
    problem = _matrix_problem(op, dict(op.metadata)) if (tier == "K2" and tpl_id.startswith("catlass.")) else None
    if problem is not None and _can_retag_problem_ids(problem):
        meta_dict = dict(op.metadata)
        catlass_opts = tuple((k, _canon_digest_value(meta_dict[k])) for k in _CATLASS_SPEC_META_KEYS if k in meta_dict)
        norm_problem = replace(problem, a_id="__a__", b_id="__b__", c_id="__c__", indices_id="__idx__")
        memo_key: tuple[Any, ...] = (
            "K2_norm", tpl_id, _canon_digest_value(params), _canon_digest_value(norm_problem),
            catlass_opts, _canon_digest_value(_norm_catalog_datapath(path)), limits,
        )
    else:
        problem = None
        memo_key = (
            tier, tpl_id, _canon_digest_value(params),
            _norm_catalog_op(op),
            tuple(_norm_catalog_op(member) for member in members),
            _canon_digest_value(_norm_catalog_datapath(path)), limits,
        )
    cached = catalog_memo.get(memo_key)
    if cached is None:
        lower = _catalog_lowerer(op, members, atomic_entry.ops, path, tier, tpl_id, params, limits)
        result = lower(path, die=die, region_id=op.op_id)
        if len(catalog_memo) >= _MAX_CATALOG_MEMO_ENTRIES:
            catalog_memo.pop(next(iter(catalog_memo)))
        catalog_memo[memo_key] = (lower, result, problem, op.op_id, die)
        src_problem, src_op_id, src_die = problem, op.op_id, die
        site_prog = result.program if need_program else None
    else:
        lower, result, src_problem, src_op_id, src_die = cached
        site_prog = None
    alloc_repls = _alloc_id_replacements(src_problem, problem) if (src_problem is not None and problem is not None) else ()
    is_dsl = isinstance(result.summary.spec, DslKernelSpec)
    if need_program and site_prog is None:
        if hasattr(result.program, "with_context"):
            raw_for_site = _retag_summary_for_site(
                result.summary, (op.op_id,) if not is_dsl else result.summary.op_ids,
                die=die, replica=0, stage=0, is_dsl=False, op_id=op.op_id,
                alloc_replacements=alloc_repls,
            )
            site_prog = _retag_catalog_program_for_site(
                result.program, die, raw_for_site, new_problem=problem, alloc_replacements=alloc_repls,
            )
            if site_prog is None:
                local_lower = (
                    lower if not alloc_repls
                    else _catalog_lowerer(op, members, atomic_entry.ops, path, tier, tpl_id, params, limits)
                )
                site_prog = local_lower(path, die=die, region_id=op.op_id).program
        elif not alloc_repls and src_op_id == op.op_id and src_die == die:
            site_prog = result.program
        elif is_dsl:
            site_prog = result.program
        else:
            # Programs without ``with_context`` are lowered lazily per executing site
            # via ``site.lower(dp, context=context, die=site.die, region_id=site.op_id)``.
            site_prog = None
    op_ids = (
        dict(applied.absorbed_atoms)[op.op_id]
        if (template and op.family == "rewrite_fused")
        else tuple(member.op_id for member in members)
    )
    summary = _retag_summary_for_site(
        result.summary, op_ids, die=die, replica=atomic_entry.replica, stage=op.stage,
        is_dsl=is_dsl, op_id=op.op_id, alloc_replacements=alloc_repls,
    )
    site_lower = (
        lower if (not alloc_repls and src_op_id == op.op_id)
        else _catalog_lowerer(op, members, atomic_entry.ops, path, tier, tpl_id, params, limits)
    )
    return ("memo", site_prog, summary, path, limits, die, site_lower, result)


def _verify_region_summary_bindings(
    scenario: ResolvedScenario,
    applied: AppliedPlan,
    summaries: tuple[Any, ...],
    *,
    require_catalog: bool = False,
    catalog_memo: dict[Any, Any] | None = None,
) -> None:
    """Verify region summaries against canonical K2/K3 lowerings with cross-layer/replica memo."""
    if not summaries:
        return
    fusions = {f"rewrite:{edit.edit_id}" for edit in applied.plan.edits if isinstance(edit.operation, FuseRegion)}
    members = {region: atoms for region, atoms in applied.absorbed_atoms if region in fusions}
    members.update(
        (binding.target_op, (binding.target_op,))
        for binding in applied.template_bindings
        if binding.target_op not in members
    )
    bound: dict[tuple[str, tuple[str, ...]], set[tuple[int, int, int]]] = defaultdict(set)
    for rank in applied.executable:
        for op in rank.ops:
            if op.op_id in members and (op.op_id not in fusions or op.family == "rewrite_fused"):
                die = _die(scenario, rank.replica, op.stage).rank
                bound[op.op_id, members[op.op_id]].add((rank.replica, op.stage, die))
    memo = {} if catalog_memo is None else catalog_memo
    templates = {
        (getattr(binding, "replica", 0), binding.target_op): binding
        for binding in applied.template_bindings
    }
    atomic = {entry.replica: entry for entry in scenario.atomic_ops}
    streams = {stream.rank: stream for stream in applied.streams}
    needed_ids = {s.region_id for s in summaries}
    canonical: dict[tuple[int, int, str], Any] = {}
    for entry in applied.executable:
        rank_atomic = atomic.get(entry.replica)
        if rank_atomic is None:
            continue
        stream = streams.get(entry.replica)
        for op in entry.ops:
            if op.op_id not in needed_ids:
                continue
            resolved = _memoized_catalog_lowering(
                scenario, rank_atomic, op, templates.get((entry.replica, op.op_id)), applied, stream, memo,
                need_program=False,
            )
            if resolved is not None:
                canonical[entry.replica, op.stage, op.op_id] = resolved[2]
    for summary in summaries:
        work_sites = {(item.replica, item.pp_stage, item.die) for item in summary.semantic_work}
        dies = {item.die for group in (summary.read_obligations, summary.write_obligations) for item in group}
        key = summary.region_id, tuple(summary.op_ids)
        matches = sorted(
            site for site in bound.get(key, ())
            if (not work_sites or work_sites == {site}) and (not dies or dies == {site[-1]})
        )
        chosen = next((
            site for site in matches
            if (
                (expected := canonical.get((site[0], site[1], summary.region_id))) is not None
                and summary_matches_canonical(summary, expected)
            )
            or (expected is None and not require_catalog)
        ), None)
        if chosen is None:
            raise ModelingError(f"LEDGER_RECONCILIATION_ERROR:unbound_region_summary:{summary.region_id}")
        bound[key].remove(chosen)


class SoundnessInversionError(ModelingError):
    """A verified physical lower bound exceeds observed candidate wall time."""


@dataclass(frozen=True, slots=True)
class GateReceipt:
    name: str
    passed: bool
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ShippingRegression:
    delta_ns: int
    loo_culprits: tuple[tuple[str, int], ...] = ()
    baseline_u_ns: int | None = None
    baseline_m_ns: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    plan_sha256: str
    scenario_sha256: str
    scope: Literal["archetype", "full"]
    gates: tuple[GateReceipt, ...]
    wall_ns: int | None = None
    l_star_ns: int | None = None
    overlap_pct: float | None = None
    native_coverage_pct: float = 0.0
    native_fallback_reasons: tuple[str, ...] = ()
    shipping_regression: ShippingRegression | None = None

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


def _interval_union(intervals):
    spans = []
    for start, end in intervals:
        if not isfinite(start) or not isfinite(end) or end < start:
            raise ModelingError("rewrite_overlap_interval_invalid")
        if end > start:
            spans.append((start, end))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def compute_overlap_pct(intervals_x, intervals_y) -> float | None:
    """Compute intersection length divided by the smaller busy-union length."""
    x, y = _interval_union(intervals_x), _interval_union(intervals_y)
    denominator = min(fsum(end - start for start, end in x),
                      fsum(end - start for start, end in y))
    if not denominator:
        return None
    i = j = 0
    intersections = []
    while i < len(x) and j < len(y):
        start_x, end_x = x[i]
        start_y, end_y = y[j]
        intersections.append(max(0, min(end_x, end_y) - max(start_x, start_y)))
        if end_x <= end_y:
            i += 1
        if end_y <= end_x:
            j += 1
    return 100 * fsum(intersections) / denominator


def _ancestor_graph(ops, stream=None) -> tuple[tuple, dict[str, set[str]], dict[str, int], dict[str, int]]:
    """Compute topological order, deps, and transitive ancestor bitmasks in O(V + E)."""
    by_id = {op.op_id: op for op in ops}
    ordered = _topological(by_id)
    deps_map = _dependencies(ordered, stream)
    bit = {op.op_id: 1 << idx for idx, op in enumerate(ordered)}
    anc: dict[str, int] = {}
    for op in ordered:
        mask = 0
        for dep in deps_map.get(op.op_id, ()):
            if dep in bit:
                mask |= bit[dep] | anc.get(dep, 0)
        anc[op.op_id] = mask
    return ordered, deps_map, bit, anc


def _boundary_order_by_region(applied) -> dict[str, dict[str, int]]:
    """Map each ``FuseRegion`` id to atom -> kernel_boundary index."""
    handles = dict(applied.handles)
    absorbed = dict(applied.absorbed_atoms)
    order: dict[str, dict[str, int]] = {}
    for edit in applied.plan.edits:
        op = edit.operation
        if not isinstance(op, FuseRegion):
            continue
        rid = f"rewrite:{edit.edit_id}"
        pos: dict[str, int] = {}
        for b_idx, boundary in enumerate(op.kernel_boundaries):
            for ref in boundary.members:
                if ref.kind == "atom":
                    pos.setdefault(ref.id, b_idx)
                    continue
                for resolved_id in handles.get(ref.id, ()):
                    for atom_id in absorbed.get(resolved_id, (resolved_id,)):
                        pos.setdefault(atom_id, b_idx)
        if pos:
            order[rid] = pos
    return order


def _is_non_commuting_pair(tr_a, tr_b) -> bool:
    red_or_nl_a = tr_a.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr_a.is_nonlinear
    red_or_nl_b = tr_b.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr_b.is_nonlinear
    if red_or_nl_a and red_or_nl_b:
        return True
    strict_red_a = tr_a.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr_a.is_precision_sensitive
    strict_red_b = tr_b.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr_b.is_precision_sensitive
    cube_a = tr_a.trait == AlgebraicTrait.BILINEAR_MATMUL
    cube_b = tr_b.trait == AlgebraicTrait.BILINEAR_MATMUL
    return (strict_red_a and cube_b) or (cube_a and strict_red_b) or (cube_a and cube_b)


_DEP_META_KEYS = frozenset({
    "fusion_input_ports", "typed_inputs", "input_ports", "input_port_versions",
    "control_deps", "control_inputs", "barrier_after", "event_deps", "barrier_deps",
})


def _dep_meta(op) -> tuple:
    return tuple(
        pair for pair in op.metadata
        if pair[0] in _DEP_META_KEYS or (isinstance(pair[0], str) and pair[0].endswith("_producer"))
    )


def _propagate_curr_trait_masks(
    orig_ordered, orig_deps, traits, replacements, curr_bit,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Propagate surviving current-graph trait masks along original DAG edges in O(V + E)."""
    own_r: dict[str, int] = {}
    own_s: dict[str, int] = {}
    own_c: dict[str, int] = {}
    own_sum: dict[str, int] = {}
    for op in orig_ordered:
        oid = op.op_id
        tr = traits[oid]
        r_oid = replacements.get(oid, oid)
        c_bit = curr_bit.get(r_oid, 0)
        own_r[oid] = c_bit if (
            tr.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr.is_nonlinear
        ) else 0
        own_s[oid] = c_bit if (
            tr.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr.is_precision_sensitive
        ) else 0
        own_c[oid] = c_bit if tr.trait == AlgebraicTrait.BILINEAR_MATMUL else 0
        own_sum[oid] = c_bit if (
            tr.trait == AlgebraicTrait.ASSOCIATIVE_COLLECTIVE and tr.collective_op == "SUM"
        ) else 0

    anc_r: dict[str, int] = {}
    anc_s: dict[str, int] = {}
    anc_c: dict[str, int] = {}
    anc_sum: dict[str, int] = {}
    for op in orig_ordered:
        oid = op.op_id
        mr = ms = mc = msum = 0
        for dep in orig_deps.get(oid, ()):
            if dep in own_r:
                mr |= own_r[dep] | anc_r.get(dep, 0)
                ms |= own_s[dep] | anc_s.get(dep, 0)
                mc |= own_c[dep] | anc_c.get(dep, 0)
                msum |= own_sum[dep] | anc_sum.get(dep, 0)
        anc_r[oid] = mr
        anc_s[oid] = ms
        anc_c[oid] = mc
        anc_sum[oid] = msum
    return anc_r, anc_s, anc_c, anc_sum


def _check_algebraic_commutativity(
    originals, current_ops, regions, applied, *, orig_stream=None, curr_stream=None,
) -> None:
    """Reject illegal nonlinear-norm / collective-sum and cross-reduction rewrites."""
    if not originals:
        return
    if not regions and orig_stream == curr_stream and len(originals) == len(current_ops) and all(
        a.op_id == b.op_id and a.deps == b.deps and a.family == b.family
        and (a.metadata == b.metadata or _dep_meta(a) == _dep_meta(b))
        for a, b in zip(originals, current_ops)
    ):
        return
    by_id = {op.op_id: op for op in current_ops}
    replacements = {
        atom: region for region, atoms in regions.items()
        if region in by_id for atom in atoms
    }
    traits = {op.op_id: classify_op_traits(op) for op in originals}
    orig_ordered, orig_deps, orig_bit, orig_anc = _ancestor_graph(originals, orig_stream)
    _, _, curr_bit, curr_anc = _ancestor_graph(current_ops, curr_stream)
    boundary_pos = _boundary_order_by_region(applied)
    anc_r, anc_s, anc_c, anc_sum = _propagate_curr_trait_masks(
        orig_ordered, orig_deps, traits, replacements, curr_bit,
    )

    sum_collectives = [
        op.op_id for op in originals
        if traits[op.op_id].trait == AlgebraicTrait.ASSOCIATIVE_COLLECTIVE
        and traits[op.op_id].collective_op == "SUM"
    ]
    nonlinear_or_red = [
        op.op_id for op in originals
        if traits[op.op_id].trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE}
        or traits[op.op_id].is_nonlinear
    ]
    nonlinear_or_red_set = set(nonlinear_or_red)

    # Guard 1: Never fuse or push NONLINEAR_NORM / ROW_REDUCTION across ASSOCIATIVE_COLLECTIVE(SUM).
    for n_id in nonlinear_or_red:
        rn = replacements.get(n_id, n_id)
        if rn not in curr_bit:
            continue
        rn_bit = curr_bit[rn]
        if anc_sum[n_id] & rn_bit:
            for c_id in sum_collectives:
                if c_id != n_id and replacements.get(c_id, c_id) == rn and (orig_anc.get(n_id, 0) & orig_bit[c_id]):
                    raise ModelingError(f"NON_COMMUTATIVE_REWRITE:collective_sum_fusion:{c_id}:{n_id}")
        if (anc_sum[n_id] & ~rn_bit) & ~curr_anc[rn]:
            for c_id in sum_collectives:
                rc = replacements.get(c_id, c_id)
                if (
                    c_id != n_id and rc != rn and rc in curr_bit
                    and (orig_anc.get(n_id, 0) & orig_bit[c_id])
                    and not (curr_anc.get(rn, 0) & curr_bit[rc])
                ):
                    raise ModelingError(f"NON_COMMUTATIVE_REWRITE:collective_sum_reorder:{n_id}_before_{c_id}")

    for c_id in sum_collectives:
        rc = replacements.get(c_id, c_id)
        if rc not in curr_bit:
            continue
        rc_bit = curr_bit[rc]
        if rc in boundary_pos and (anc_r[c_id] & rc_bit):
            pos = boundary_pos[rc]
            for n_id in regions.get(rc, ()):
                if (
                    n_id != c_id and n_id in nonlinear_or_red_set
                    and (orig_anc.get(c_id, 0) & orig_bit[n_id])
                    and c_id in pos and n_id in pos and pos[c_id] < pos[n_id]
                ):
                    raise ModelingError(f"NON_COMMUTATIVE_REWRITE:collective_sum_reorder:{c_id}_before_{n_id}")
        if (anc_r[c_id] & ~rc_bit) & ~curr_anc[rc]:
            for n_id in nonlinear_or_red:
                rn = replacements.get(n_id, n_id)
                if (
                    c_id != n_id and rc != rn and rn in curr_bit
                    and (orig_anc.get(c_id, 0) & orig_bit[n_id])
                    and not (curr_anc.get(rc, 0) & curr_bit[rn])
                ):
                    raise ModelingError(f"NON_COMMUTATIVE_REWRITE:collective_sum_reorder:{c_id}_before_{n_id}")

    # Guard 2: Never reverse causal order across non-commuting reductions or Cube boundaries.
    guarded = [
        op.op_id for op in originals
        if traits[op.op_id].trait in {
            AlgebraicTrait.ROW_REDUCTION,
            AlgebraicTrait.ATTENTION_CORE,
            AlgebraicTrait.BILINEAR_MATMUL,
        }
        or traits[op.op_id].is_nonlinear
        or traits[op.op_id].is_precision_sensitive
    ]
    guarded_set = set(guarded)
    for b_id in guarded:
        rb = replacements.get(b_id, b_id)
        if rb not in curr_bit:
            continue
        tr_b = traits[b_id]
        req = 0
        if tr_b.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr_b.is_nonlinear:
            req |= anc_r[b_id]
        if tr_b.trait in {AlgebraicTrait.ROW_REDUCTION, AlgebraicTrait.ATTENTION_CORE} or tr_b.is_precision_sensitive:
            req |= anc_c[b_id]
        if tr_b.trait == AlgebraicTrait.BILINEAR_MATMUL:
            req |= anc_s[b_id] | anc_c[b_id]
        if (req & ~curr_bit[rb]) & ~curr_anc[rb]:
            for a_id in guarded:
                ra = replacements.get(a_id, a_id)
                if (
                    a_id != b_id and ra != rb and ra in curr_bit
                    and (orig_anc.get(b_id, 0) & orig_bit[a_id])
                    and _is_non_commuting_pair(traits[a_id], tr_b)
                    and not (curr_anc.get(rb, 0) & curr_bit[ra])
                ):
                    raise ModelingError(f"NON_COMMUTATIVE_REWRITE:non_commuting_reorder:{b_id}_before_{a_id}")

    for rid, pos in boundary_pos.items():
        if rid not in by_id:
            continue
        r_atoms = [aid for aid in regions.get(rid, ()) if aid in guarded_set and aid in pos]
        for a_id in r_atoms:
            tr_a = traits[a_id]
            for b_id in r_atoms:
                if (
                    a_id != b_id and pos[b_id] < pos[a_id]
                    and (orig_anc.get(b_id, 0) & orig_bit[a_id])
                    and _is_non_commuting_pair(tr_a, traits[b_id])
                ):
                    raise ModelingError(f"NON_COMMUTATIVE_REWRITE:non_commuting_boundary_swap:{b_id}_before_{a_id}")


def _structure(scenario, applied):
    sorter = TopologicalSorter()
    for source, target, credits in applied.credit_edges:
        if type(credits) is not int or credits < 0:
            raise ModelingError("CREDIT_CYCLE:invalid_credit_count")
        if credits == 0:
            sorter.add(target, source)
    try:
        tuple(sorter.static_order())
    except CycleError as error:
        raise ModelingError("CREDIT_CYCLE:zero_credit_wait") from error

    original = {rank.replica: rank.ops for rank in scenario.atomic_ops}
    orig_streams = {stream.rank: stream for stream in scenario.stream_plans}
    streams = {stream.rank: stream for stream in applied.streams}
    regions = dict(applied.absorbed_atoms)
    seen = set()
    for rank in applied.executable:
        by_id = {op.op_id: op for op in rank.ops}
        if len(by_id) != len(rank.ops):
            raise ModelingError("rewrite_duplicate_operator")
        _topological(by_id)
        originals = original.get(rank.replica, ())
        atom_ids = {op.op_id for op in originals}
        claimed = set()
        for region in regions.keys() & by_id.keys():
            atoms = set(regions[region])
            if not atoms or atoms - atom_ids or claimed & atoms or not _convex(originals, atoms):
                raise ModelingError(f"NON_CONVEX_REGION:{region}")
            claimed.update(atoms)
            seen.add(region)
        _check_algebraic_commutativity(
            originals, rank.ops, regions, applied,
            orig_stream=orig_streams.get(rank.replica),
            curr_stream=streams.get(rank.replica),
        )
        original_users = _successors(originals) if applied.in_place_aliases else {}
        replacements = {atom: region for region, atoms in regions.items()
                        if region in by_id for atom in atoms}
        for source, writer in applied.in_place_aliases:
            if writer not in by_id:
                continue
            pending = list(by_id[writer].deps)
            predecessors = set()
            while pending:
                key = pending.pop()
                if key in by_id and key not in predecessors:
                    predecessors.add(key)
                    pending.extend(by_id[key].deps)
            for reader in original_users.get(source, ()):
                surviving = replacements.get(reader, reader)
                if surviving in by_id and surviving != writer and surviving not in predecessors:
                    raise ModelingError(f"ALIAS_CONFLICT:{source}:{writer}:{reader}")
        if rank.replica in streams:
            _check_stream(rank.ops, streams[rank.replica])
    if seen != regions.keys():
        raise ModelingError("NON_CONVEX_REGION:missing_executable_region")
    if len(streams) != len(applied.streams) or streams.keys() - original.keys():
        raise ModelingError("STREAM_DAG_CYCLE:invalid_rank_stream_binding")


def _capacity(die, pool, simt, path):
    if pool == "hbm":
        return die.memory_bytes
    raw = dict(die.buffers).get(pool, 0)
    if pool == "ub" and simt:
        return max(0, raw - int(path.simt_ub_carveout_bytes))
    return raw


@dataclass(frozen=True, slots=True)
class _ReplayEvaluation(StepEvaluation):
    execution_evidence: dict = field(default_factory=dict, repr=False)
    service_intervals: dict = field(default_factory=dict, repr=False)
    cohort_intervals: dict = field(default_factory=dict, repr=False)
    cohort_replicas: dict = field(default_factory=dict, repr=False)
    scenario_sha256: str = field(default="", repr=False)
    plan_sha256: str = field(default="", repr=False)


def _representative_cohorts(cohorts):
    return {op_id: cohorts[replica, op_id] for replica, op_id in sorted(cohorts, reverse=True)}


def _evaluation(
    scenario, graphs, report, graph_id, tier, prepared=None,
    *, scenario_sha256: str = "", plan_sha256: str = "",
    extra_pipelines: tuple[FusedPipelineDescriptor, ...] = (),
):
    index = {(replica, op.op_id): op for replica, ops in graphs.items() for op in ops}
    rows, occurrences = [], {}
    for row in report["timeline"]:
        replica, op_id, stage = row["replica"], row["op_id"], row["stage"]
        rank = _die(scenario, replica, stage).rank
        occurrences[row["id"]] = OpOccurrence(replica, stage, rank, op_id)
    service, cohorts = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for ev_name, spans in (report.get("evicted_service_spans") or {}).items():
        replica, _, tail = str(ev_name).partition(":")
        if not replica.startswith("r") or not replica[1:].isdigit():
            continue
        _, _, tail = tail.partition(":")
        keys = tail.split(":")
        for count in range(len(keys), 0, -1):
            node = (int(replica[1:]), ":".join(keys[:count]))
            if node in index:
                service[node].extend(spans)
                break
    for event in report.get("events", ()):
        if not event.get("serviced"):
            continue
        event_name = str(event.get("task", ""))
        replica, _, tail = event_name.partition(":")
        if not replica.startswith("r"):
            continue
        _, _, tail = tail.partition(":")
        keys = tail.split(":")
        for count in range(len(keys), 0, -1):
            node = (int(replica[1:]), ":".join(keys[:count])) if replica[1:].isdigit() else None
            if node in index:
                span = (event["start_ms"], event["end_ms"])
                service[node].append(span)
                suffix = keys[count:]
                if suffix and suffix[0] == "body":
                    suffix = suffix[1:]
                prefix, _, cohort = (suffix[0] if suffix else "").rpartition("_")
                if prefix == _COHORT_PREFIX and cohort in {"0", "1"}:
                    cohorts[node][int(cohort)].append(span)
                break
    for ordinal, row in enumerate(report["timeline"]):
        occ = occurrences[row["id"]]
        op = index.get((occ.replica, occ.op_id))
        preds = tuple(occurrences.get(dep) or _parse_occurrence(dep, occ.replica, occ.stage)
                      for dep in row["predecessors"])
        spans = service.get((occ.replica, occ.op_id), ())
        ready = row["ready_ms"]
        rows.append(OperatorSequenceRow(
            occ, ordinal, row["family"], row.get("program") or row["recipe"], preds,
            row["stream"], op.local_shape if op is not None else (),
            op.dtype if op is not None else "", _work_items(op) if op is not None else (),
            ready, min((start for start, _ in spans), default=ready), row["complete_ms"],
            row.get("isolated_ms"),
        ))
    base_pipelines = tuple(
        item.descriptor for item in prepared.operators if item.descriptor is not None
    ) if prepared is not None else ()
    pipelines = (*base_pipelines, *extra_pipelines) if extra_pipelines else base_pipelines
    evidence = report.get("execution", {})
    cohort_replicas = {node: {cohort: tuple(spans) for cohort, spans in groups.items()}
                       for node, groups in cohorts.items()}
    return _ReplayEvaluation(
        graph_id, scenario.step, tier, report["duration_ms"], tuple(rows), pipelines,
        _wave_breakdown_from_report(report), report.get("engine_backend") or "",
        int(evidence.get("native_steps", 0)),
        execution_evidence=evidence,
        service_intervals=dict(service),
        cohort_intervals=_representative_cohorts(cohort_replicas),
        cohort_replicas=cohort_replicas,
        scenario_sha256=scenario_sha256 or getattr(scenario, "scenario_sha256", ""),
        plan_sha256=plan_sha256,
    )

def _combine_replays(rows):
    if not rows:
        return None
    offset, operators, service = 0.0, [], {}
    cohorts, serviced, peaks = {}, defaultdict(float), defaultdict(float)
    for row in rows:
        operators.extend(replace(op, start_ms=op.start_ms + offset, end_ms=op.end_ms + offset)
                         for op in row.operators)
        service.update({key: tuple((start + offset, end + offset) for start, end in intervals)
                        for key, intervals in row.service_intervals.items()})
        cohorts.update({node: {cohort: tuple((start + offset, end + offset) for start, end in spans)
                               for cohort, spans in groups.items()}
                        for node, groups in row.cohort_replicas.items()})
        for key, amount in row.execution_evidence.get("serviced", {}).items():
            serviced[key] += amount
        for key, amount in row.execution_evidence.get("peak_allocated_per_unit", {}).items():
            peaks[key] = max(peaks[key], amount)
        offset += row.span_ms
    evidence = {**rows[0].execution_evidence, "serviced": dict(serviced), "peak_allocated_per_unit": dict(peaks)}
    return replace(rows[0], operators=tuple(operators), span_ms=offset, service_intervals=service,
                   cohort_intervals=_representative_cohorts(cohorts), cohort_replicas=cohorts,
                   execution_evidence=evidence)

def _execution_evidence(step_eval):
    return getattr(step_eval, "execution_evidence", {}) or {}


def _capacity_checks(scenario, applied, step_eval):
    rows = {(row.occurrence.replica, row.occurrence.op_id): row
            for row in getattr(step_eval, "operators", ())}
    streams = {stream.rank: stream for stream in applied.streams}
    candidates = {
        (getattr(binding, "replica", 0), binding.target_op): binding.tile_spec
        for binding in applied.tile_bindings
        if isinstance(binding.tile_spec, TilePipelineSpec)
    }
    leases = defaultdict(list)
    for lease in applied.prefetch_leases:
        leases[(getattr(lease, "replica", 0), lease.target_op)].append(lease)
    scope = {}
    liveness = defaultdict(list)
    for rank in applied.executable:
        all_ops = {op.op_id: op for op in rank.ops}
        stages = {op.stage for op in rank.ops}
        for stage in stages:
            die = _die(scenario, rank.replica, stage)
            path = _path(scenario, rank.replica, stage)
            simt = any(
                dict(op.metadata).get("uses_simt")
                or dict(op.metadata).get("tile_uses_simt")
                or dict(op.metadata).get("template_param_uses_simt")
                or "simt" in str(dict(op.metadata).get("template_id", "")).lower()
                for op in rank.ops if op.stage == stage
            )
            if "l2" in dict(die.buffers) and dict(die.buffer_units).get("l2", 1) != 1:
                raise ModelingError("MULTI_POOL_CAPACITY:shared_l2_buffer_requires_one_die_owner")
            scope[rank.replica, stage] = die, path, simt
        for ordinal, op in enumerate(rank.ops):
            ident = rank.replica, op.stage
            die, path, simt = scope[ident]
            row = rows.get((rank.replica, op.op_id))
            # Without replay, derive a conservative interval among declared lanes.
            lane = op.stream or MAIN_STREAM
            fallback = (ordinal, ordinal + 1) if rank.replica not in streams else (0, 1)
            start, end = _span(row, default=fallback)
            site = (rank.replica, op.op_id)
            if site in candidates:
                solved, _ = _solve_tiles(scenario, rank.replica, op, candidates[site],
                                         streams.get(rank.replica))
                owners_by_lane = (bind_stream_owners(streams[rank.replica], path)
                                  if rank.replica in streams else {})
                lane_owners = owners_by_lane.get(lane)
                for peak in solved.pool_peaks:
                    if peak.owner_kind == "die" or lane_owners is None or lane_owners.isolation == "shared_cap":
                        owner = peak.owner_kind
                    else:
                        owner = (peak.owner_kind, lane)
                    # Include aliases of the same physical owner in the same sweep.
                    liveness[ident, peak.pool, owner].append((start, end, peak.peak_bytes))
            for lease in leases.get(site, ()):
                if lease.window_anchor_op not in all_ops:
                    raise ModelingError(f"MULTI_POOL_CAPACITY:missing_prefetch_anchor:{op.op_id}")
                anchor = rows.get((rank.replica, lease.window_anchor_op))
                anchor_end = 0 if anchor is None else _span(anchor, default=(0, 0))[1]
                lane_owners = (
                    bind_stream_owners(streams[rank.replica], path).get(lane)
                    if rank.replica in streams else None
                )
                owner_kind = "aiv" if lease.pool == "ub" else "aic"
                owner = (
                    (owner_kind, lane)
                    if lane_owners is not None and lane_owners.isolation == "disjoint"
                    else owner_kind
                )
                lane_ids = _lane_pool_unit_ids(lane_owners, lease.pool, max(1, int(die.pool_units(lease.pool))))
                declared_ids = tuple(int(u) for u in (getattr(lease, "physical_unit_ids", ()) or ()))
                if declared_ids:
                    if not set(declared_ids) <= set(lane_ids):
                        raise ModelingError(
                            f"MULTI_POOL_CAPACITY:prefetch_unit_outside_lane:{ident}:{lease.pool}:{lane}"
                        )
                    units = len(declared_ids)
                else:
                    declared_units = int(getattr(lease, "physical_units", 0) or 0)
                    lane_cap_units = len(lane_ids)
                    units = min(declared_units, lane_cap_units) if declared_units > 0 else lane_cap_units
                units = max(1, units)
                per_unit_bytes = (int(lease.max_bytes) + units - 1) // units
                liveness[ident, lease.pool, owner].append((anchor_end, end, per_unit_bytes))
    for (ident, pool, owner), intervals in liveness.items():
        die, path, simt = scope[ident]
        peak, _ = residency_peak(tuple(intervals))
        if peak > _capacity(die, pool, simt, path):
            raise ModelingError(f"MULTI_POOL_CAPACITY:{ident}:{pool}:{owner}:{peak}")

    for key, peak in _execution_evidence(step_eval).get("peak_allocated_per_unit", {}).items():
        physical = key.split(":unit", 1)[0]
        pool, _, placement = physical.partition("@")
        if pool not in _CAPACITY_POOLS:
            continue
        for (replica, stage), (die, path, simt) in scope.items():
            if placement and placement not in {f"r{replica}:s{stage}", f"d{die.rank}"}:
                continue
            if peak > _capacity(die, pool, simt, path):
                raise ModelingError(f"MULTI_POOL_CAPACITY:{key}:{peak}")


def _native_preflight(scenario, applied, step_eval):
    reasons = set()
    eligible = tasks = terms = 0
    stages = set()
    for rank in applied.executable:
        for op in rank.ops:
            tasks += 1
            stages.add((rank.replica, op.stage))
            terms += sum(work > 0 for work in (op.flops, op.vector_ops, op.hbm_bytes,
                                                op.mte2_bytes, op.mte3_bytes, op.link_bytes))
            if all(type(key) is str and key != "occurrence" and _flat(value)
                   for key, value in op.metadata):
                eligible += 1
            else:
                reasons.add("metadata_not_flat")
    evidence = _execution_evidence(step_eval)
    resources = evidence.get("resource_units", ())
    if not resources:
        resources = tuple(
            count for replica, stage in stages
            for _, count in _die(scenario, replica, stage).resource_units
        )
    if len(resources) + len(scenario.case.machine.links) > ACCEL_MAX_R:
        reasons.add("resource_capacity_exceeded")
    if max(resources, default=0) > ACCEL_MAX_U:
        reasons.add("owner_capacity_exceeded")
    if max(tasks, evidence.get("peak_live_tasks", 0)) > ACCEL_MAX_T or terms > ACCEL_MAX_L:
        reasons.add("task_or_link_capacity_exceeded")
    if any(len({lane.priority for lane in stream.lanes}) > 1 for stream in applied.streams):
        reasons.add("mixed_priority")
    fallback = evidence.get("fallback_reason")
    if fallback:
        reasons.add(fallback)
    steps = evidence.get("advance_steps", 0)
    if steps:
        coverage = 100 * min(steps, evidence.get("native_steps", 0)) / steps
    elif reasons - {"metadata_not_flat"}:
        coverage = 0.0
    else:
        coverage = 100 * eligible / tasks if tasks else 100.0
    return coverage, tuple(sorted(reasons))


def _span(row, *, default=None):
    if row is None or row.start_ms is None or row.end_ms is None:
        return default
    return row.start_ms, row.end_ms


def _active_overlap(groups, names):
    measured = defaultdict(list)
    for left, right in combinations(names, 2):
        value = compute_overlap_pct(groups.get(left, ()), groups.get(right, ()))
        if value is not None:
            measured[left].append(value)
            measured[right].append(value)
    if any(not measured[name] for name in names):
        return None
    return min(max(measured[name]) for name in names)


def _core_and_overlap(scenario, applied, step_eval):
    by_rank = {entry.replica: entry.ops for entry in applied.executable}
    intervals = defaultdict(lambda: defaultdict(list))
    measured = getattr(step_eval, "service_intervals", None)
    for row in getattr(step_eval, "operators", ()):
        occurrence = row.occurrence
        span = _span(row)
        busy = ((span,) if span is not None else ()) if measured is None else measured.get(
            (occurrence.replica, occurrence.op_id), ())
        intervals[occurrence.replica, occurrence.stage][row.stream].extend(busy)
    percentages = []
    for stream in applied.streams:
        assignments = dict(stream.assignments)
        by_stage = defaultdict(list)
        for op in by_rank[stream.rank]:
            by_stage[op.stage].append(op)
        stage_results = []
        for stage, ops in by_stage.items():
            path = _path(scenario, stream.rank, stage)
            bound = bind_stream_owners(stream, path)
            for op in ops:
                lane = assignments.get(op.op_id, MAIN_STREAM)
                owners = bound[lane]
                meta = dict(op.metadata)
                simt = meta.get("tile_uses_simt") or meta.get("template_param_uses_simt")
                aic = bool((op.flops or op.matrix_flops or meta.get("requires_aic")) and not simt)
                aiv = bool(op.vector_ops or meta.get("requires_aiv") or simt)
                if aic and owners.aic_limit <= 0:
                    raise ModelingError(f"stream_zero_aic_budget:{lane}:{op.op_id}")
                if aiv and owners.aiv_limit <= 0:
                    raise ModelingError(f"stream_zero_aiv_budget:{lane}:{op.op_id}")
                if aic and aiv:
                    if owners.isolation == "shared_cap":
                        base_aic = owners.pool_aic or tuple(range(path.aic_count))
                        base_aiv = owners.pool_aiv or tuple(range(path.aiv_count))
                        owners = replace(owners, pool_aic=base_aic[:owners.aic_limit],
                                         pool_aiv=base_aiv[:owners.aiv_limit])
                    aic_owners, aiv_owners = proportional_coupled_placement(owners, path, stream_id=lane)
                    if (len(aic_owners) * path.aiv_count != len(aiv_owners) * path.aic_count
                            or min(owners.aic_limit, owners.aiv_limit) <= 0):
                        raise ModelingError(f"stream_coupled_placement_invalid:{lane}")
            if len(stream.lanes) > 1:
                result = _active_overlap(intervals[stream.rank, stage], stream.lane_ids())
                if result is not None:
                    stage_results.append(result)
        if len(stream.lanes) > 1:
            percentages.append(max(stage_results) if stage_results else None)
    cohort_intervals = getattr(step_eval, "cohort_intervals", {}) or {}
    cohort_replicas = getattr(step_eval, "cohort_replicas", None)
    for tile in applied.tile_bindings:
        if tile.cohort_split == "dbo_2cohort":
            tile_rep = getattr(tile, "replica", 0)
            if cohort_replicas is not None:
                intervals = (
                    [cohort_replicas.get((tile_rep, tile.target_op), {})]
                    if tile_rep in by_rank and any(op.op_id == tile.target_op for op in by_rank[tile_rep])
                    else []
                )
            else:
                intervals = [cohort_intervals.get(tile.target_op, {})]
            percentages.extend(compute_overlap_pct(groups.get(0, ()), groups.get(1, ()))
                               for groups in intervals or ({},))
    if any(value is None for value in percentages):
        raise ModelingError("UNPROVED_OVERLAP")
    overlap = min(percentages) if percentages else None
    if overlap is not None and overlap <= FAKE_OVERLAP_THRESHOLD_PCT:
        return overlap, f"FAKE_OVERLAP:{overlap:.6g}"
    return overlap, ""


def _work_totals(ops):
    work = defaultdict(list)
    for op in ops:
        work["flops", ""].append(op.flops)
        work["vector", ""].append(op.vector_ops)
        work["link", op.link_scope].append(op.link_bytes)
        for dtype, amount in op.matrix_flops or (((op.dtype, op.flops),) if op.flops else ()):
            work["matrix", dtype].append(amount)
    return {key: fsum(values) for key, values in work.items() if any(values)}


def _dropped(actual, expected, allowance=0):
    return actual + allowance + SERVICE_CONSERVATION_ABS_TOL < expected * (1 - SERVICE_CONSERVATION_REL_TOL)


def _service_amount(service, prefixes):
    return fsum(value for key, value in service.items() if key.startswith(prefixes))


def _hbm_service_bytes(service, profile):
    name = profile.names["hbm"]
    path, alpha = profile.datapath, profile.datapath.hbm_rw_alpha
    read = service.get(name + ":r", 0)
    write = service.get(name + ":w", 0)
    fallback = profile.rates["hbm"][0]
    read_rate = path.hbm_read_bytes_per_s / 1000 or fallback
    write_rate = path.hbm_write_bytes_per_s / 1000 or fallback
    if alpha == 1 or read_rate == write_rate:
        # Fully coupled rows only identify total time. Use the maximum possible bytes.
        return (read + write) / (1 + alpha) * max(read_rate, write_rate)
    return ((read - alpha * write) * read_rate + (write - alpha * read) * write_rate) / (1 - alpha**2)


def _template_service(scenario, applied, edited, replay):
    graphs, book, baseline_book, context, deadline = replay
    targets = {op_id for _, op_id in edited}
    profiles = []
    token = ACTIVE_DEADLINE_MONO.set(deadline)
    try:
        for ranks, recipes, streams in (
            ({rank.replica: rank.ops for rank in scenario.atomic_ops}, baseline_book, scenario.stream_plans),
            (graphs, book, applied.streams),
        ):
            site_graphs = {op_id: dict.fromkeys(ranks, ()) for op_id in targets}
            for rank, ops in ranks.items():
                for op in ops:
                    if op.op_id in targets:
                        site_graphs[op.op_id][rank] = _replay_ops((op,))
            profiles.append({(row["replica"], op_id): row["serviced"]
                             for op_id, site in site_graphs.items()
                             for row in operator_costs(scenario.atomic_step.case, site, recipes=recipes,
                                 evaluation=context, stream_plans=streams or None)["operators"]})
    finally:
        ACTIVE_DEADLINE_MONO.reset(token)
    return profiles


_ZERO_COPY_VIEW_FAMS = frozenset({"view", "reshape", "slice", "permute", "transpose", "split", "contiguous"})
_LAW1_ABI_ALLOWED_SYMS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "aclnnAddRmsNormDynamicQuant": (
        frozenset({"add", "rmsnorm", "add_rmsnorm", "dynamic_quant"}),
        frozenset({"dynamic_quant"}),
    ),
    "aclnnAddRmsNormDynamicMxQuant": (
        frozenset({"add", "rmsnorm", "add_rmsnorm", "dynamic_mx_quant"}),
        frozenset({"dynamic_mx_quant"}),
    ),
    "aclnnSwiGluQuant": (
        frozenset({"swiglu", "dynamic_quant", "dynamic_mx_quant"}),
        frozenset({"swiglu"}),
    ),
    "aclnnDequantSituQuant": (
        frozenset({"situ", "dynamic_quant", "dynamic_mx_quant"}),
        frozenset({"situ"}),
    ),
}


def _extract_scenario_abi_context(scenario, verdict, matched_entry):
    from mindie_sim.schema.stack import ManifestBinding
    from .algebra import DEFAULT_SHIPPING_STACK_PASSES

    sku_id = str(getattr(scenario.case.machine, "sku", "") or getattr(scenario.case.machine, "name", "") or "")
    lowering_plan = (
        getattr(scenario, "stack_lowering_plan", None)
        or getattr(getattr(scenario, "mainline_step", None), "stack_lowering_plan", None)
        or getattr(getattr(scenario, "atomic_step", None), "stack_lowering_plan", None)
    )
    track_spec = getattr(lowering_plan, "track_spec", None)
    manifest_bindings = tuple(getattr(track_spec, "active_bindings", ()) or ())
    if (
        verdict.status == "admitted_proposed_pr"
        and track_spec is None
        and not manifest_bindings
        and matched_entry.required_mechanism_id
        and verdict.proposed_pr_ref
    ):
        manifest_bindings = (
            ManifestBinding(
                mechanism_id=matched_entry.required_mechanism_id,
                state="proposed_pr",
                ref=verdict.proposed_pr_ref,
            ),
        )
    enabled_passes = (
        frozenset(lowering_plan.lowering_passes)
        if (lowering_plan is not None and getattr(lowering_plan, "lowering_passes", ()))
        else DEFAULT_SHIPPING_STACK_PASSES
    )
    parallel = getattr(getattr(scenario.case, "runtime", None), "parallel", None)
    tp_size = int(parallel.tp) if parallel is not None else None
    ep_size = int(parallel.ep) if parallel is not None else None
    return sku_id, manifest_bindings, enabled_passes, tp_size, ep_size


def _rerun_canonical_law_abi_query(
    edit_id: str,
    abi_id: str,
    absorbed,
    traits,
    verdict,
    *,
    sku_id: str,
    manifest_bindings,
    enabled_passes,
    tp_size,
    ep_size,
    registry,
):
    """Rerun the canonical law-specific registry query on ``absorbed`` ops; return ``(re_abi, re_verdict, re_digest)`` or ``None``."""
    from .algebra import (
        DEFAULT_MLA_ROPE_HEAD_DIM,
        _build_mc2_gemm_verdict,
        _build_mla_prolog_ports,
        _classify_ub_chain_abi,
        _resolve_model_scope,
        _resolve_world_size,
        _typed_mla_role,
        _verify_abi_query,
        _verify_gmm_act_quant_abi,
        _verify_law2a_matmul_abi,
        _verify_moe_frame_abi,
    )

    if abi_id in _LAW1_ABI_ALLOWED_SYMS:
        allowed_syms, req_syms = _LAW1_ABI_ALLOWED_SYMS[abi_id]
        material = [
            atom for atom in absorbed
            if not (
                traits[atom.op_id].trait == AlgebraicTrait.VIEW_METADATA
                and str(atom.family or "").strip().lower() in _ZERO_COPY_VIEW_FAMS
            )
        ]
        mat_syms = [traits[atom.op_id].symbolic_op_kind for atom in material]
        if (
            not material
            or any(traits[atom.op_id].trait in {
                AlgebraicTrait.BILINEAR_MATMUL, AlgebraicTrait.ASSOCIATIVE_COLLECTIVE, AlgebraicTrait.ATTENTION_CORE,
            } for atom in material)
            or any(sym not in allowed_syms for sym in mat_syms)
            or not (req_syms <= set(mat_syms))
        ):
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_motif_mismatch:{edit_id}:{abi_id}")
        re_abi, _, _, _, re_verdict, re_digest, _ = _classify_ub_chain_abi(
            absorbed, traits, soc_arch=verdict.device_arch, sku_id=sku_id,
            pr_track=verdict.pr_track, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_passes, registry=registry,
        )
        return re_abi, re_verdict, re_digest
    if abi_id in {"aclnnMatmul", "aclnnQuantMatmulV4"}:
        if any(traits[atom.op_id].trait != AlgebraicTrait.BILINEAR_MATMUL for atom in absorbed):
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_motif_mismatch:{edit_id}:{abi_id}")
        return _verify_law2a_matmul_abi(
            absorbed, traits, soc_arch=verdict.device_arch, sku_id=sku_id,
            pr_track=verdict.pr_track, track_spec=None, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_passes, registry=registry,
        )
    if abi_id in {"aclnnAllGatherMatmulV2", "aclnnMatmulReduceScatterV2"}:
        comm_ops = [a for a in absorbed if traits[a.op_id].trait == AlgebraicTrait.ASSOCIATIVE_COLLECTIVE]
        mm_ops = [a for a in absorbed if traits[a.op_id].trait == AlgebraicTrait.BILINEAR_MATMUL]
        if len(comm_ops) != 1 or len(mm_ops) != 1:
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_motif_mismatch:{edit_id}:{abi_id}")
        ws = _resolve_world_size(comm_ops[0], tp_size, collective=True)
        re_verdict, re_digest = _build_mc2_gemm_verdict(
            abi_id, mm_ops[0], comm_ops[0], traits[mm_ops[0].op_id],
            soc_arch=verdict.device_arch, sku_id=sku_id, ws=ws,
            pr_track=verdict.pr_track, track_spec=None, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_passes, registry=registry,
            is_agmm=(abi_id == "aclnnAllGatherMatmulV2"),
        )
        return abi_id, re_verdict, re_digest
    if abi_id.startswith("aclnnMlaProlog"):
        role_of = {a.op_id: _typed_mla_role(a) for a in absorbed}
        for atom in absorbed:
            if not role_of[atom.op_id]:
                fam = str(atom.family or "").strip().lower()
                if fam == "rotary":
                    role_of[atom.op_id] = "rotary"
                elif fam in {"cache_store", "kv_cache_write", "kv_compress_epilog"}:
                    role_of[atom.op_id] = "kv_store"
                elif fam == "rmsnorm":
                    role_of[atom.op_id] = "q_norm"
        rope_dims = [int(m["qk_rope_head_dim"]) for a in absorbed if isinstance((m := dict(a.metadata)).get("qk_rope_head_dim"), int)]
        rope_dim = rope_dims[0] if rope_dims else DEFAULT_MLA_ROPE_HEAD_DIM
        built = _build_mla_prolog_ports(absorbed, role_of, traits, soc_arch=verdict.device_arch, rope_dim=rope_dim)
        if built is None:
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_motif_mismatch:{edit_id}:{abi_id}")
        re_abi, operands, outputs = built
        re_verdict, re_digest = _verify_abi_query(
            abi_id=re_abi, soc_arch=verdict.device_arch, sku_id=sku_id, graph_motif="mla_prolog",
            operands=operands, outputs=outputs, pr_track=verdict.pr_track, track_spec=None,
            manifest_bindings=manifest_bindings, enabled_stack_passes=enabled_passes,
            registry=registry, model_scope=_resolve_model_scope(absorbed),
            qk_rope_head_dim=rope_dim, mla_variant="latent_mla",
        )
        return re_abi, re_verdict, re_digest
    if abi_id in {"aclnnMegaMoe", "dispatch_ffn_combine"}:
        gmms = [a for a in absorbed if traits[a.op_id].is_grouped_matmul]
        disp = next((a for a in absorbed if "dispatch" in str(a.family or "").lower() or "dispatch" in str(a.op_id).lower()), None)
        comb = next((a for a in absorbed if "combine" in str(a.family or "").lower() or "combine" in str(a.op_id).lower()), None)
        act_op = next((a for a in absorbed if traits[a.op_id].activation_kind or traits[a.op_id].symbolic_op_kind in {"swiglu", "situ", "gelu", "silu"}), None)
        if disp is None or comb is None or act_op is None or len(gmms) < 2:
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_motif_mismatch:{edit_id}:{abi_id}")
        return _verify_moe_frame_abi(
            disp, gmms[0], act_op, gmms[1], comb, traits,
            soc_arch=verdict.device_arch, sku_id=sku_id, ep_size=ep_size,
            enable_fused_mc2=(1 if abi_id == "dispatch_ffn_combine" else None),
            pr_track=verdict.pr_track, track_spec=None, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_passes, registry=registry,
        )
    if abi_id == "aclnnGroupedMatmulSwigluQuantV2":
        gmm_op = next((a for a in absorbed if traits[a.op_id].is_grouped_matmul), None)
        act_op = next((a for a in absorbed if traits[a.op_id].activation_kind or traits[a.op_id].symbolic_op_kind in {"swiglu", "situ", "gelu", "silu"}), None)
        quant_op = next((a for a in absorbed if traits[a.op_id].symbolic_op_kind in {"dynamic_quant", "dynamic_mx_quant"}), None)
        if gmm_op is None or act_op is None or quant_op is None:
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_motif_mismatch:{edit_id}:{abi_id}")
        return _verify_gmm_act_quant_abi(
            gmm_op, act_op, quant_op, traits, soc_arch=verdict.device_arch, sku_id=sku_id,
            pr_track=verdict.pr_track, track_spec=None, manifest_bindings=manifest_bindings,
            enabled_stack_passes=enabled_passes, registry=registry,
        )
    return None


def _check_entry_hidden_constraints(edit_id: str, matched_entry, absorbed) -> None:
    sc = getattr(matched_entry, "shape_constraints", None)
    if sc is None or not absorbed[0].local_shape:
        return
    hidden = int(absorbed[0].local_shape[-1])
    h_set = getattr(sc, "hidden_allowed_set", frozenset())
    if h_set and hidden not in h_set:
        raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_shape_mismatch:{edit_id}:{hidden}")
    h_lo, h_hi = getattr(sc, "hidden_range", (0, 0))
    if (h_lo > 0 and hidden < h_lo) or (h_hi > 0 and hidden > h_hi):
        raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_shape_mismatch:{edit_id}:{hidden}")
    h_div = int(getattr(sc, "hidden_divisibility", 0) or 0)
    if h_div > 0 and hidden % h_div != 0:
        raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_shape_mismatch:{edit_id}:{hidden}")


def _verify_verdict_matches_absorbed_ops(edit, op, verdict, matched_entry, scenario, applied, registry) -> None:
    """Ensure a catalog-valid ABI verdict matches the actual absorbed ops per replica."""
    from .algebra import _norm_abi_dtype

    absorbed_map = dict(applied.absorbed_atoms)
    fused_id = f"rewrite:{edit.edit_id}"
    member_ids = absorbed_map.get(fused_id) or tuple(
        ref.id for ref in op.members if ref.kind == "atom"
    )
    if not member_ids:
        return
    sku_id, manifest_bindings, enabled_passes, tp_size, ep_size = _extract_scenario_abi_context(
        scenario, verdict, matched_entry,
    )
    exec_by_replica = {entry.replica: {o.op_id for o in entry.ops} for entry in applied.executable}
    abi_id = verdict.abi_id
    for rank_ops in scenario.atomic_ops:
        if rank_ops.replica in exec_by_replica and fused_id not in exec_by_replica[rank_ops.replica]:
            continue
        atomic_by_id = {atom.op_id: atom for atom in rank_ops.ops}
        absorbed = tuple(atomic_by_id[oid] for oid in member_ids if oid in atomic_by_id)
        if len(absorbed) != len(member_ids) or not absorbed:
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_members_missing:{edit.edit_id}:r{rank_ops.replica}")
        traits = {atom.op_id: classify_op_traits(atom) for atom in absorbed}
        rerun = _rerun_canonical_law_abi_query(
            edit.edit_id, abi_id, absorbed, traits, verdict,
            sku_id=sku_id, manifest_bindings=manifest_bindings, enabled_passes=enabled_passes,
            tp_size=tp_size, ep_size=ep_size, registry=registry,
        )
        if rerun is not None:
            re_abi, re_verdict, re_digest = rerun
            if (
                re_abi != abi_id
                or re_verdict is None
                or re_digest != verdict.query_digest
                or re_verdict.status != verdict.status
                or re_verdict.matched_entry_id != verdict.matched_entry_id
            ):
                raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_query_mismatch:{edit.edit_id}:{abi_id}")
            continue
        first_tr = traits[absorbed[0].op_id]
        if first_tr.input_dtype and verdict.act_dtype and _norm_abi_dtype(first_tr.input_dtype) != _norm_abi_dtype(verdict.act_dtype):
            raise ModelingError(f"ABI_WITNESS_INVALID:absorbed_dtype_mismatch:{edit.edit_id}:{abi_id}")
        _check_entry_hidden_constraints(edit.edit_id, matched_entry, absorbed)


def _verify_fuse_region_witnesses(scenario, applied, scope, *, step_eval=None, baseline_eval=None):
    """Enforce bidirectional Equation 7 exact witness parity on every ``FuseRegion`` edit."""
    soc_arch = _resolve_soc_arch(
        str(getattr(scenario.case.machine, "arch_family", "") or ""),
        scenario.datapath,
        str(getattr(scenario.case.machine, "sku", "") or getattr(scenario.case.machine, "name", "") or ""),
    )
    registry = None
    for edit in applied.plan.edits:
        op = edit.operation
        if not isinstance(op, FuseRegion):
            continue
        verdict = op.abi_verdict
        if verdict is None:
            if scope == "full" and op.fusion_tier in {"K1_shipping", "K1_proposed_pr"}:
                raise ModelingError(f"ABI_WITNESS_INVALID:missing_abi_verdict:{edit.edit_id}")
            continue
        validate_fusion_tier_verdict(op.fusion_tier, verdict)
        if soc_arch is not None and verdict.device_arch != soc_arch:
            raise ModelingError(
                f"ABI_WITNESS_INVALID:arch_mismatch:{edit.edit_id}:{verdict.device_arch}!={soc_arch}"
            )
        if op.kernel_boundaries and verdict.abi_id != op.kernel_boundaries[0].abi_id:
            raise ModelingError(
                f"ABI_WITNESS_INVALID:abi_id_mismatch:{edit.edit_id}:{verdict.abi_id}!={op.kernel_boundaries[0].abi_id}"
            )
        if op.abi_query_digest and op.abi_query_digest != verdict.query_digest:
            raise ModelingError(f"ABI_WITNESS_INVALID:query_digest_mismatch:{edit.edit_id}")
        if op.backend_target is not None and op.backend_target != verdict.backend_kind:
            raise ModelingError(
                f"ABI_WITNESS_INVALID:backend_kind_mismatch:{edit.edit_id}:{op.backend_target}!={verdict.backend_kind}"
            )
        if verdict.matched_lowering_id and op.lowering_id != verdict.matched_lowering_id:
            raise ModelingError(f"ABI_WITNESS_INVALID:lowering_id_mismatch:{edit.edit_id}")
        if verdict.matched_lowering_sha256 and op.lowering_sha256 != verdict.matched_lowering_sha256:
            raise ModelingError(f"ABI_WITNESS_INVALID:lowering_sha256_mismatch:{edit.edit_id}")
        if verdict.status in {"admitted_shipping", "admitted_proposed_pr"}:
            if (
                scope == "full"
                and step_eval is not None
                and baseline_eval is not None
                and round(step_eval.span_ms * NS_PER_MS) >= round(baseline_eval.span_ms * NS_PER_MS)
            ):
                raise ModelingError(f"ABI_WITNESS_INVALID:engine_speedup_not_achieved:{edit.edit_id}")
            registry = registry or default_abi_registry()
            snap = registry.get_snapshot(verdict.device_arch)
            if verdict.probe_manifest_sha256 != snap.payload_sha256:
                raise ModelingError(f"ABI_WITNESS_INVALID:probe_manifest_sha256_mismatch:{edit.edit_id}")
            matched_entry = next((e for e in snap.entries if e.entry_id == verdict.matched_entry_id), None)
            expected_track = "merged" if verdict.status == "admitted_shipping" else "unmerged_pr"
            if (
                matched_entry is None
                or not matched_entry.available_on_device
                or matched_entry.abi_id != verdict.abi_id
                or matched_entry.backend_kind != verdict.backend_kind
                or matched_entry.symbol_or_kernel != verdict.symbol_or_kernel
                or matched_entry.vllm_ascend_track != expected_track
                or (verdict.source_ref is not None and matched_entry.source_ref != verdict.source_ref)
            ):
                raise ModelingError(f"ABI_WITNESS_INVALID:entry_witness_mismatch:{edit.edit_id}")
            _verify_verdict_matches_absorbed_ops(edit, op, verdict, matched_entry, scenario, applied, registry)


def _verify_summary_lineage_work(summary, atomic_by_site):
    """Enforce per-``(lineage, multiplicity, dtype, unit)`` useful-work conservation."""
    replicas = {item.replica for item in summary.semantic_work}
    if len(replicas) != 1:
        raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{summary.region_id}:mixed_replicas")
    replica = next(iter(replicas))
    covered = [op for oid in summary.op_ids if (op := atomic_by_site.get((replica, oid))) is not None]
    if not covered:
        return
    exp_by_unit_dtype: dict[tuple[str, str], float] = defaultdict(float)
    exp_by_unit: dict[str, float] = defaultdict(float)
    exp_by_lineage: dict[tuple[str, str, str], float] = defaultdict(float)
    for op in covered:
        if op.matrix_flops:
            for dt, amount in op.matrix_flops:
                val = float(amount)
                key_dt = str(dt or op.dtype).strip().lower()
                exp_by_unit_dtype["matrix_mac", key_dt] += val
                exp_by_unit["matrix_mac"] += val
                exp_by_lineage[op.op_id, "matrix_mac", key_dt] += val
        elif op.flops:
            val = float(op.flops)
            key_dt = str(op.dtype).strip().lower()
            exp_by_unit_dtype["matrix_mac", key_dt] += val
            exp_by_unit["matrix_mac"] += val
            exp_by_lineage[op.op_id, "matrix_mac", key_dt] += val
        if op.vector_ops:
            val = float(op.vector_ops)
            key_dt = str(op.dtype).strip().lower()
            exp_by_unit_dtype["vector", key_dt] += val
            exp_by_unit["vector"] += val
            exp_by_lineage[op.op_id, "vector", key_dt] += val

    got_by_unit_dtype: dict[tuple[str, str], float] = defaultdict(float)
    got_by_unit: dict[str, float] = defaultdict(float)
    got_by_lineage: dict[tuple[str, str, str], float] = defaultdict(float)
    covered_ids = {op.op_id for op in covered}
    all_lineage_exact = True
    for item in summary.semantic_work:
        val = float(item.logical_ops)
        if val < 0:
            raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{summary.region_id}:negative_work")
        unit = "matrix_mac" if item.kind == "matrix_mac" else "vector"
        key_dt = str(item.dtype).strip().lower()
        got_by_unit_dtype[unit, key_dt] += val
        got_by_unit[unit] += val
        if item.work_id in covered_ids:
            got_by_lineage[item.work_id, unit, key_dt] += val
        else:
            all_lineage_exact = False

    for unit in ("matrix_mac", "vector"):
        if not isclose(exp_by_unit.get(unit, 0.0), got_by_unit.get(unit, 0.0), rel_tol=1e-12, abs_tol=1e-9):
            raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{summary.region_id}:semantic_work")
    if all_lineage_exact and exp_by_lineage:
        for key in exp_by_lineage.keys() | got_by_lineage.keys():
            if not isclose(exp_by_lineage.get(key, 0.0), got_by_lineage.get(key, 0.0), rel_tol=1e-12, abs_tol=1e-9):
                raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{summary.region_id}:lineage:{key[0]}:{key[1]}:{key[2]}")
    else:
        for key in exp_by_unit_dtype.keys() | got_by_unit_dtype.keys():
            if key[0] == "matrix_mac" and not isclose(
                exp_by_unit_dtype.get(key, 0.0), got_by_unit_dtype.get(key, 0.0), rel_tol=1e-12, abs_tol=1e-9,
            ):
                raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{summary.region_id}:dtype:{key[0]}:{key[1]}")


_PREFETCH_BYTES_TOL = 1e-6


def _row_resource_amount(row, resources: frozenset[str]) -> float:
    return fsum(
        float(item.amount) for item in getattr(row, "work", ())
        if item.resource in resources
    )


def _verify_applied_prefetch_dmas(
    applied, step_eval, baseline_eval=None,
) -> tuple[dict[tuple[int, str], float], float, float]:
    """Reconcile every applied ``PrefetchLease`` with its executed DMA, release, and consumer rows."""
    rows_by_site = {
        (row.occurrence.replica, row.occurrence.op_id): row
        for row in getattr(step_eval, "operators", ())
    }
    base_by_site = {
        (row.occurrence.replica, row.occurrence.op_id): row
        for row in getattr(baseline_eval, "operators", ())
    } if baseline_eval is not None else {}
    exec_by_site = {(entry.replica, op.op_id): op for entry in applied.executable for op in entry.ops}
    displaced: dict[tuple[int, str], float] = defaultdict(float)
    total_dma_hbm = 0.0
    total_dma_mte2 = 0.0
    mte2_res = frozenset({"mte2", "mte2_aic", "mte2_aiv"})
    hbm_res = frozenset({"hbm"})
    for lease in applied.prefetch_leases:
        if lease.max_bytes <= 0:
            continue
        replica = getattr(lease, "replica", 0)
        prefetch_row = rows_by_site.get((replica, f"prefetch_W_{lease.target_op}"))
        release_row = rows_by_site.get((replica, f"release_W_{lease.target_op}"))
        if prefetch_row is None or prefetch_row.family != "prefetch":
            raise ModelingError(
                f"WORK_CONSERVATION_MISMATCH:missing_prefetch_dma:{replica}:{lease.target_op}"
            )
        if release_row is None or release_row.family != "release":
            raise ModelingError(
                f"WORK_CONSERVATION_MISMATCH:missing_prefetch_release:{replica}:{lease.target_op}"
            )
        executed_bytes = _row_resource_amount(prefetch_row, hbm_res)
        if executed_bytes + _PREFETCH_BYTES_TOL < float(lease.max_bytes):
            raise ModelingError(
                f"WORK_CONSERVATION_MISMATCH:prefetch_dma_underexecuted:{replica}:{lease.target_op}"
            )
        target_op = exec_by_site.get((replica, lease.target_op))
        expected_mte2 = (
            _displaced_weight_mte2(target_op, lease.max_bytes)[0]
            if target_op is not None else 0.0
        )
        executed_mte2 = _row_resource_amount(prefetch_row, mte2_res)
        if expected_mte2 > 0.0 and executed_mte2 + _PREFETCH_BYTES_TOL < expected_mte2:
            raise ModelingError(
                f"WORK_CONSERVATION_MISMATCH:prefetch_dma_mte2_underexecuted:{replica}:{lease.target_op}"
            )
        target_row = rows_by_site.get((replica, lease.target_op))
        base_row = base_by_site.get((replica, lease.target_op))
        if target_row is not None and base_row is not None:
            if _dropped(executed_bytes + _row_resource_amount(target_row, hbm_res), _row_resource_amount(base_row, hbm_res)):
                raise ModelingError(
                    f"WORK_CONSERVATION_MISMATCH:prefetch_hbm_conservation_dropped:{replica}:{lease.target_op}"
                )
            if _dropped(executed_mte2 + _row_resource_amount(target_row, mte2_res), _row_resource_amount(base_row, mte2_res)):
                raise ModelingError(
                    f"WORK_CONSERVATION_MISMATCH:prefetch_mte2_conservation_dropped:{replica}:{lease.target_op}"
                )
        total_dma_hbm += float(lease.max_bytes)
        total_dma_mte2 += expected_mte2
        displaced[replica, lease.target_op] += min(float(lease.max_bytes), executed_bytes)
    return displaced, total_dma_hbm, total_dma_mte2


def _executed_service_conservation(scenario, applied, step_eval, baseline_eval, service_replay, summaries, *, scope="full"):
    edits = applied.plan.edits
    has_rebind = any(isinstance(edit.operation, (FuseRegion, SelectTemplate)) for edit in edits)
    has_prefetch = any(lease.max_bytes > 0 for lease in applied.prefetch_leases)
    if step_eval is None:
        if scope == "full" and (has_rebind or has_prefetch):
            raise ModelingError("WORK_CONSERVATION_MISMATCH:missing_step_eval")
        return
    leases: dict[tuple[int, str], float] = defaultdict(float)
    total_dma_hbm = 0.0
    total_dma_mte2 = 0.0
    if has_prefetch and (scope == "full" or has_rebind):
        leases, total_dma_hbm, total_dma_mte2 = _verify_applied_prefetch_dmas(
            applied, step_eval, baseline_eval,
        )
    if not has_rebind:
        if has_prefetch and scope == "full":
            got_srv = _execution_evidence(step_eval).get("serviced")
            if not isinstance(got_srv, dict):
                raise ModelingError("WORK_CONSERVATION_MISMATCH:missing_executed_service")
            hbm_res, mte2_res = frozenset({"hbm"}), frozenset({"mte2", "mte2_aic", "mte2_aiv"})
            got_ops = getattr(step_eval, "operators", ())
            got_op_hbm = fsum(_row_resource_amount(row, hbm_res) for row in got_ops)
            got_op_mte2 = fsum(_row_resource_amount(row, mte2_res) for row in got_ops)
            if _dropped(_service_amount(got_srv, _TRANSFER_SERVICE["hbm"]), max(total_dma_hbm, got_op_hbm)):
                raise ModelingError("WORK_CONSERVATION_MISMATCH:executed_memory_service_dropped")
            if _dropped(_service_amount(got_srv, _TRANSFER_SERVICE["mte2"]), max(total_dma_mte2, got_op_mte2)):
                raise ModelingError("WORK_CONSERVATION_MISMATCH:executed_mte2_service_dropped")
            base_ops = getattr(baseline_eval, "operators", ()) if baseline_eval is not None else ()
            if base_ops:
                base_op_hbm = fsum(_row_resource_amount(row, hbm_res) for row in base_ops)
                base_op_mte2 = fsum(_row_resource_amount(row, mte2_res) for row in base_ops)
                if _dropped(got_op_hbm, base_op_hbm):
                    raise ModelingError("WORK_CONSERVATION_MISMATCH:executed_memory_service_dropped")
                if _dropped(got_op_mte2, base_op_mte2):
                    raise ModelingError("WORK_CONSERVATION_MISMATCH:executed_mte2_service_dropped")
        return
    exec_by_site = {(entry.replica, op.op_id): op for entry in applied.executable for op in entry.ops}
    for row in getattr(step_eval, "operators", ()):
        op = exec_by_site.get((row.occurrence.replica, row.occurrence.op_id))
        if op is None:
            continue
        if (op.flops > 0 or op.matrix_flops) and row.implementation in _NON_MATRIX_IMPLEMENTATIONS:
            raise ModelingError(f"WORK_CONSERVATION_MISMATCH:recipe_drops_matrix:{op.op_id}:{row.implementation}")
        if op.link_bytes > 0 and row.implementation == "TiledPointwise":
            raise ModelingError(f"WORK_CONSERVATION_MISMATCH:recipe_drops_link:{op.op_id}:{row.implementation}")
    got_serviced = _execution_evidence(step_eval).get("serviced")
    base_serviced = _execution_evidence(baseline_eval or scenario.atomic_step).get("serviced")
    if not isinstance(got_serviced, dict) or not isinstance(base_serviced, dict):
        raise ModelingError("WORK_CONSERVATION_MISMATCH:missing_executed_service")
    catalog_sites = {(summary.semantic_work[0].replica, summary.region_id) for summary in summaries
                     if summary.semantic_work}
    catalog = {(summary.semantic_work[0].replica, summary.region_id): summary for summary in summaries
               if summary.semantic_work} if service_replay else {}
    kinds = {type(edit.operation) for edit in edits}
    checks = {"link": ("link:",)}
    if not kinds & {QuantizeContract, HoistOrDeduplicate}:
        checks["matrix"] = _MATRIX_SERVICE_PREFIXES
    if SelectTemplate in kinds and not catalog_sites:
        checks.update(_TRANSFER_SERVICE)
        if FuseRegion not in kinds:
            checks["vector"] = ("vector@",)
    quantized = {op_id for receipt in applied.edit_receipts
                 if receipt.opcode == QuantizeContract.__name__ and receipt.status == "applied"
                 for op_id in receipt.affected_ops}
    rebound = {(getattr(binding, "replica", 0), binding.target_op) for binding in applied.template_bindings}
    base_rows = {(row.occurrence.replica, row.occurrence.op_id): row
                 for row in getattr(baseline_eval or scenario.atomic_step, "operators", ())
                 if isinstance(row, OperatorSequenceRow)
                 and (row.occurrence.replica, row.occurrence.op_id) in rebound
                 and (row.occurrence.replica, row.occurrence.op_id) in exec_by_site}
    actual_rows = {(row.occurrence.replica, row.occurrence.op_id): row
                   for row in getattr(step_eval, "operators", ())
                   if (row.occurrence.replica, row.occurrence.op_id) in rebound}
    if base_rows.keys() - actual_rows.keys():
        raise ModelingError("WORK_CONSERVATION_MISMATCH:missing_template_work")
    targets = base_rows.keys() | catalog.keys()
    audited = _template_service(scenario, applied, targets, service_replay) if service_replay and targets else None
    profiles = {(item.replica, item.stage): item for item in service_replay[3].bindings.stage_profiles} if catalog else {}
    displaced_matrix = 0
    for occurrence, summary in catalog.items():
        replica, op_id = occurrence
        service = audited[1].get(occurrence, {})
        op = exec_by_site[occurrence]
        read, write = measure_logical_bytes(summary.read_obligations), measure_logical_bytes(summary.write_obligations)
        if len(summary.op_ids) == 1 and (read < op.mte2_bytes or write < op.mte3_bytes):
            raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{op_id}:template_operand_bytes")
        simt = dict(summary.vector_ops_by_pipe).get("aiv_simt", 0)
        if simt and not summary.issued_cube_flops_by_dtype:
            displaced_matrix += _service_amount(audited[0].get(occurrence, {}), _MATRIX_SERVICE_PREFIXES)
        for role, actual, required in (
            ("hbm", _hbm_service_bytes(service, profiles[replica, op.stage]), float(summary.turnaround_service_bytes)),
            ("mte2", _service_amount(service, _TRANSFER_SERVICE["mte2"]), read),
            ("output", _service_amount(service, ("fixpipe@", "mte3@")), write),
            ("aiv_simt", _service_amount(service, ("aiv_simt@", "simt@", "vector@")), simt),
        ):
            if _dropped(actual, required):
                raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{op_id}:executed_{role}")
    for name, prefixes in checks.items():
        expected, actual = (_service_amount(service, prefixes) for service in (base_serviced, got_serviced))
        allowance = sum(leases.values()) if name in _TRANSFER_SERVICE else displaced_matrix if name == "matrix" else 0
        if _dropped(actual, expected, allowance):
            detail = "memory" if name == "hbm" else name
            raise ModelingError(f"WORK_CONSERVATION_MISMATCH:executed_{detail}_service_dropped")
    for occurrence, row in base_rows.items():
        if occurrence in catalog_sites:
            continue
        replica, op_id = occurrence
        actual = Counter()
        expected = Counter()
        for item in actual_rows[occurrence].work:
            actual[item.resource] += item.amount
        for item in row.work:
            expected[item.resource] += item.amount
        for role, prefixes in _TEMPLATE_SERVICE.items():
            allowance = leases.get((replica, op_id), 0.0) if role in _TRANSFER_SERVICE else 0.0
            if op_id in quantized and role in {"hbm", "mte2", "mte3"}:
                continue
            if _dropped(actual[role], expected[role], allowance):
                raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{op_id}:{role}")
            if audited:
                before, after = (_service_amount(profile.get(occurrence, {}), prefixes) for profile in audited)
                if _dropped(after, before, allowance):
                    raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{op_id}:executed_{role}")


def _hoisted_sites(scenario, applied) -> set[tuple[int, str]]:
    scoped = getattr(applied, "hoisted_load_time_sites", ())
    if scoped:
        return {(int(rep), str(oid)) for rep, oid in scoped}
    if not applied.hoisted_load_time:
        return set()
    return {(entry.replica, oid) for entry in scenario.atomic_ops for oid in applied.hoisted_load_time}


def _conservation(scenario, applied, summaries, step_eval=None, baseline_eval=None, baseline_work=None,
                  service_replay=None, *, scope="full", catalog_memo=None):
    _verify_fuse_region_witnesses(scenario, applied, scope, step_eval=step_eval, baseline_eval=baseline_eval)
    _verify_region_summary_bindings(scenario, applied, summaries, catalog_memo=catalog_memo)
    hoist_sites = _hoisted_sites(scenario, applied)
    cse_ids: set[str] = set()
    cse = {edit.edit_id for edit in applied.plan.edits
           if isinstance(edit.operation, HoistOrDeduplicate) and edit.operation.scope == "intra_step"}
    for receipt in applied.edit_receipts:
        if receipt.edit_id in cse and receipt.status == "applied":
            cse_ids.update(receipt.affected_ops)
    retained = {entry.replica: entry.ops for entry in applied.executable}
    if retained.keys() != {entry.replica for entry in scenario.atomic_ops}:
        raise ModelingError("WORK_CONSERVATION_MISMATCH:replica_domain")
    for entry in scenario.atomic_ops:
        after = retained.get(entry.replica, ())
        rank_hoists = {oid for rep, oid in hoist_sites if rep == entry.replica} | cse_ids
        if baseline_work is not None and not rank_hoists:
            expected = baseline_work[entry.replica]
        else:
            retained_ids = {op.op_id for op in after}
            expected = _work_totals(op for op in entry.ops
                                    if op.op_id not in rank_hoists or op.op_id in retained_ids)
        actual = _work_totals(after)
        for key in expected.keys() | actual.keys():
            if not isclose(expected.get(key, 0), actual.get(key, 0), rel_tol=1e-12, abs_tol=1e-9):
                raise ModelingError(f"WORK_CONSERVATION_MISMATCH:{entry.replica}:{key}")
    atomic_by_site = {(entry.replica, op.op_id): op
                      for entry in scenario.atomic_ops for op in entry.ops} if summaries else {}
    for summary in summaries:
        logical, issued, dram, turnaround, hits = (
            summary.logical_transfer_bytes, summary.issued_copy_bytes, summary.dram_sector_bytes,
            summary.turnaround_service_bytes, summary.realized_l2_hit_bytes,
        )
        if (any(type(value) is not int or value < 0 for value in (logical, issued, dram, hits))
                or type(turnaround) is not Fraction or turnaround < dram
                or hits > issued or (issued == hits and (dram or turnaround))):
            raise ModelingError(f"LEDGER_RECONCILIATION_ERROR:{summary.region_id}")
        if summary.op_ids and summary.semantic_work:
            _verify_summary_lineage_work(summary, atomic_by_site)
        cut = compute_compulsory_hbm_cut((summary,))
        if sum(item.read_bytes + item.write_bytes for item in cut.values()) > dram:
            raise ModelingError(f"LEDGER_RECONCILIATION_ERROR:{summary.region_id}:dram_sector_bytes")
        required = measure_logical_bytes(summary.read_obligations) + measure_logical_bytes(summary.write_obligations)
        if logical < required:
            raise ModelingError(f"LEDGER_RECONCILIATION_ERROR:{summary.region_id}:logical_bytes")
        if issued < logical:
            raise ModelingError(f"LEDGER_RECONCILIATION_ERROR:{summary.region_id}:issued_copy_bytes")
    _executed_service_conservation(scenario, applied, step_eval, baseline_eval, service_replay, summaries, scope=scope)


def _capture(name, check, *args, **kwargs):
    try:
        check(*args, **kwargs)
    except ModelingError as error:
        reason, _, detail = str(error).partition(":")
        return GateReceipt(name, False, reason, detail)
    return GateReceipt(name, True)


def _bound(scenario, applied, summaries, primitive_table, rate_mode, certificate, program, dag, wall, u, m, framework, dsl,
           *, catalog_memo=None):
    if program is not None or dag is not None:
        if summaries:
            _verify_region_summary_bindings(
                scenario, applied, summaries, require_catalog=True, catalog_memo=catalog_memo,
            )
        canonical_program, canonical_dag, _ = build_dual_inputs_from_scenario(
            scenario,
            applied=applied,
            region_summaries=summaries,
            primitive_table=primitive_table,
            rate_mode=rate_mode,
            verify_summaries=not summaries,
        )
        if (program is not None and program != canonical_program) or (dag is not None and dag != canonical_dag):
            raise ModelingError("LEAN_CERTIFICATE_INVALID:certificate_unbound_to_scenario")
    if certificate is None or program is None or dag is None:
        raise ModelingError("LEAN_CERTIFICATE_INVALID:missing_certificate")
    if not check_optimality_certificate(canonical_program, canonical_dag, certificate):
        raise ModelingError("LEAN_CERTIFICATE_INVALID")
    l_star = certificate.l_star_lean_ns
    if l_star < 0 or wall is None:
        raise ModelingError("LEAN_CERTIFICATE_INVALID:missing_or_invalid_bound_evidence")
    if any(value is not None and type(value) is not int for value in (wall, u, m, framework, dsl)):
        raise ModelingError("LADDER_MONOTONICITY_VIOLATION:invalid_wall_time")
    for name, value in (("wall", wall), ("dsl", dsl), ("framework", framework)):
        if value is not None and value < l_star:
            raise SoundnessInversionError(
                f"SOUNDNESS_BUG_LOWER_BOUND_INVERTED:{name}_ns={value}<l_star_ns={l_star}")
    if any(value is not None and value < 0 for value in (u, m)):
        raise ModelingError("LADDER_MONOTONICITY_VIOLATION:invalid_wall_time")
    chain = [value for value in (l_star, dsl, framework) if value is not None]
    baselines = [value for value in (u, m) if value is not None]
    if baselines:
        chain.append(min(baselines))
    if any(left > right for left, right in zip(chain, chain[1:])):
        raise ModelingError("LADDER_MONOTONICITY_VIOLATION")
    return l_star


def _elided_d2h_sites(scenario, applied) -> set[tuple[int, str]]:
    scoped = getattr(applied, "elided_d2h_sites", ())
    if scoped:
        return {(int(rep), str(oid)) for rep, oid in scoped}
    if not applied.elided_d2h_ops:
        return set()
    return {(entry.replica, oid) for entry in scenario.atomic_ops for oid in applied.elided_d2h_ops}


def _verify_step_eval_binding(scenario, applied, step_eval) -> None:
    """Bind full-step Engine replay strictly to ``scenario``, ``step``, ``applied``, and occurrences."""
    if getattr(step_eval, "request", None) != scenario.step:
        raise ModelingError("LEAN_CERTIFICATE_INVALID:step_eval_request_mismatch")
    require_digests = isinstance(step_eval, _ReplayEvaluation) or bool(applied.plan.edits)
    eval_scen_sha = getattr(step_eval, "scenario_sha256", "")
    if (require_digests or eval_scen_sha) and eval_scen_sha != scenario.scenario_sha256:
        raise ModelingError("LEAN_CERTIFICATE_INVALID:step_eval_scenario_mismatch")
    eval_plan_sha = getattr(step_eval, "plan_sha256", "")
    if (require_digests or eval_plan_sha) and eval_plan_sha != applied.plan_sha256:
        raise ModelingError("LEAN_CERTIFICATE_INVALID:step_eval_plan_mismatch")
    rows = getattr(step_eval, "operators", ()) or ()
    if rows or applied.plan.edits:
        hoist_sites = _hoisted_sites(scenario, applied)
        elided_sites = _elided_d2h_sites(scenario, applied)
        row_meta = {(row.occurrence.replica, row.occurrence.op_id): (row.shape, row.family) for row in rows}
        for entry in applied.executable:
            for op in entry.ops:
                site = (entry.replica, op.op_id)
                if site in hoist_sites or site in elided_sites:
                    continue
                if site not in row_meta or row_meta[site] != (op.local_shape, op.family):
                    raise ModelingError(f"LEAN_CERTIFICATE_INVALID:step_eval_occurrence_mismatch:{op.op_id}")


def verify_plan(
    scenario: ResolvedScenario, applied: AppliedPlan, *, step_eval=None, wall_ns=None,
    dual_certificate=None, dual_program=None, event_dag=None, baseline_u_ns=None,
    baseline_m_ns=None, framework_ns=None, dsl_ns=None, region_summaries=(), scope="full",
    baseline_eval=None, baseline_work=None, service_replay=None, primitive_table=None, rate_mode="engine_ceiling",
    catalog_memo=None,
) -> VerificationReceipt:
    """Run structural and replay gates for the stated occurrence coverage."""
    if scope not in {"archetype", "full"}:
        raise ModelingError(f"rewrite_verification_scope_invalid:{scope}")
    replay_wall_ns = round(step_eval.span_ms * NS_PER_MS) if step_eval is not None else None
    wall_mismatch = False
    if scope == "full":
        if replay_wall_ns is not None:
            if wall_ns is not None and wall_ns != replay_wall_ns:
                wall_mismatch = True
            else:
                wall_ns = replay_wall_ns
    elif wall_ns is None and replay_wall_ns is not None:
        wall_ns = replay_wall_ns

    region_summaries = tuple(region_summaries)
    gates = [_capture("convexity_and_acyclicity", _structure, scenario, applied),
             _capture("multi_pool_capacity_and_c11", _capacity_checks, scenario, applied, step_eval)]
    coverage, reasons = _native_preflight(scenario, applied, step_eval)
    name = "stream_core_budget_and_overlap"
    overlap = None
    try:
        overlap, defect = _core_and_overlap(scenario, applied, step_eval)
        reason, _, detail = defect.partition(":")
        gates.append(GateReceipt(name, not defect, reason, detail))
    except ModelingError as error:
        reason, _, detail = str(error).partition(":")
        gates.append(GateReceipt(name, False, reason, detail))
    eff_memo = {} if catalog_memo is None else catalog_memo
    gates.append(_capture("work_conservation_and_ledgers", _conservation,
                          scenario, applied, region_summaries, step_eval, baseline_eval, baseline_work,
                          service_replay, scope=scope, catalog_memo=eff_memo))
    l_star = regression = None
    if scope == "full":
        name = "lean_bound_and_monotonicity"
        try:
            if step_eval is None:
                raise ModelingError("LEAN_CERTIFICATE_INVALID:missing_step_eval")
            _verify_step_eval_binding(scenario, applied, step_eval)
            if wall_mismatch:
                raise ModelingError("LEAN_CERTIFICATE_INVALID:wall_ns_step_eval_mismatch")
            if region_summaries and not gates[-1].passed:
                raise ModelingError("LEAN_CERTIFICATE_INVALID:certificate_unbound_to_scenario")
            l_star = _bound(scenario, applied, region_summaries, primitive_table, rate_mode,
                            dual_certificate, dual_program, event_dag, wall_ns,
                            baseline_u_ns, baseline_m_ns, framework_ns, dsl_ns,
                            catalog_memo=eff_memo)
            gates.append(GateReceipt(name, True))
        except SoundnessInversionError:
            raise
        except ModelingError as error:
            text = ("LEAN_CERTIFICATE_INVALID:certificate_unbound_to_scenario"
                    if str(error).startswith("LEDGER_RECONCILIATION_ERROR:unbound_region_summary") else str(error))
            reason, _, detail = text.partition(":")
            gates.append(GateReceipt(name, False, reason, detail))
        if baseline_m_ns is not None and baseline_u_ns is not None and baseline_m_ns > baseline_u_ns + SHIPPING_REGRESSION_TOLERANCE_NS:
            regression = ShippingRegression(baseline_m_ns - baseline_u_ns, baseline_u_ns=baseline_u_ns, baseline_m_ns=baseline_m_ns)
    return VerificationReceipt(applied.plan_sha256, scenario.scenario_sha256, scope, tuple(gates), wall_ns, l_star,
                               overlap, coverage, reasons, regression)


__all__ = [
    "FAKE_OVERLAP_THRESHOLD_PCT", "SHIPPING_REGRESSION_TOLERANCE_NS",
    "ACCEL_MAX_R", "ACCEL_MAX_U", "ACCEL_MAX_T", "ACCEL_MAX_L", "SoundnessInversionError",
    "GateReceipt", "ShippingRegression", "VerificationReceipt", "compute_overlap_pct", "verify_plan",
]
