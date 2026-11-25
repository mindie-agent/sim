"""Replica-preserving primal-dual lower bounds and optimality certificate checking.

[Role] Assemble replica-scoped physical demands, solve exact rational Farkas
  dual programs and checked critical paths, and verify optimality certificates
  with zero dependency on ``mindie_sim.optimization``.
[Invariants] All atomic ops, semantic aliases, and event DAG nodes/edges are
  keyed strictly by ``(replica, op_id)`` so parallel replicas never splice
  phantom cross-replica chains (``L* <= T_feasible`` always). Boundary copy
  obligations track read (``mte2 > 0``) and terminal store (``mte3 > 0``)
  independently; ``sort_topk`` only emits admitted schedulers with positive rate.
[Timeouts & Locks] Finite Bland pivots and linear DAG traversal; no locks or I/O.
[Side-Effect Policy] Pure read-only certificate and demand analysis.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from operator import mul
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from mindie_sim.prediction.fusion.ports import input_port_entries
from mindie_sim.prediction.operators.matrix.matrix_precision import (
    PACKED_FP4_FORMATS, resolve_matrix_precision,
)
from mindie_sim.prediction.runtime.reload_bounds import (
    SECTOR_BYTES, EpochAccount, build_surplus_coupling_sites, compute_compulsory_hbm_cut,
    direct_mac_pebble_reload_bound, verify_epoch_trace,
)
from mindie_sim.primitives.loader import validate_table_sku_compatibility
from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.contracts import FusedRegionSummary, SemanticWorkItem, TensorRangeObligation

from .catalog_replay import canonical_region_summaries, summary_matches_canonical
from .schema import FuseRegion

if TYPE_CHECKING:
    from .apply import AppliedPlan, ResolvedScenario

BINARY64_ULP_SHIFT = 52
C11_ULP_GUARD_FACTOR = 8
ENDPOINT_ROUND_GUARD_NS = 2
NS_PER_MS = 1_000_000
MS_PER_SECOND = 1000
NS_PER_SECOND = MS_PER_SECOND * NS_PER_MS
_GIGA = 1_000_000_000
_GIBI = 1024 * 1024 * 1024
_KIBI = 1024
_MEBI = 1024 * 1024
_MEGA = 1000 * 1000
_CAPACITY_UNIT_SCALE = {
    **dict.fromkeys(("b/s", "byte/s", "bytes/s", "flop/s", "flops/s", "ops/s", "elements/s"), 1),
    **dict.fromkeys(("b/ms", "byte/ms", "bytes/ms", "flop/ms", "flops/ms", "ops/ms",
                     "elements/ms", "kb/s"), MS_PER_SECOND),
    **dict.fromkeys(("gb/s", "gflop/s", "gops/s"), _GIGA),
    **dict.fromkeys(("tb/s", "tflop/s", "tops/s"), MS_PER_SECOND * _GIGA),
    "mb/s": _MEGA,
    "kib/s": _KIBI,
    "mib/s": _MEBI,
    "gib/s": _GIBI,
    "tib/s": _KIBI * _GIBI,
}
_HARDWARE_RATE_METRICS = {
    "hbm_bus_r": (("P01_datacopy", "hbm_read_bytes_per_s", None),),
    "hbm_bus_w": (("P01_datacopy", "hbm_write_bytes_per_s", None),),
    "mte2_total": (
        ("P01_datacopy", "mte2_aic_hit_bytes_per_s_per_core", "aic_count"),
        ("P01_datacopy", "mte2_aiv_hit_bytes_per_s_per_core", "aiv_count"),
    ),
    "write_total": (
        ("P03_fixpipe", "fixpipe_out_bytes_per_s_per_aic", "aic_count"),
        ("P04_mte3_store", "mte3_store_bytes_per_s_per_aiv", "aiv_count"),
    ),
    "l2_bw": (("P05_l2_residency_reuse", "l2_die_bytes_per_s", None),),
    "cube": (("P06_mmad", "cube_only_rate", "aic_count"),),
    "vector": (("P07_vec_alu", "vec_alu_ops_per_s_per_aiv", "aiv_count"),),
    "simt": (("P11_simt", "simt_ops_per_s_per_aiv", "aiv_count"),),
}
_MMAD_MODES = {
    "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
    "w4a8": "int8", "a8w4_mx": "mxfp8", "fp8": "fp8_e4m3",
}
_COLD_WEIGHT_FAMILIES = frozenset({
    "gemm", "linear", "qkv_projection", "output_projection", "gate_up_projection",
    "down_projection", "projection", "lm_head", "logits",
})
_WEIGHT_ID_KEYS = ("weight_allocation_id", "allocation_id")
_ENGINE_PIPE_MAP = {"cube": "cube", "aiv_simd": "vector", "aiv_simt": "simt"}
_PAIR_LEN = 2
_PORT_LINK_SEMANTICS = frozenset({"per_port", "fully_connected", "mesh"})


def _resolve_comm_endpoint_die(
    meta: dict,
    keys: tuple[str, ...],
    default_die: int | None,
    dies_tuple: tuple[int, ...],
) -> int | None:
    for key in keys:
        val = meta.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and (not dies_tuple or val in dies_tuple):
            return val
    return default_die


def _link_switch_pipe(link_spec) -> str | None:
    if getattr(link_spec, "switch_capacity_bytes_per_ms", None) is None:
        return None
    sw_id = str(getattr(link_spec, "switch_id", "") or "").strip()
    return f"link_switch_id:{sw_id}" if sw_id else f"link_switch:{link_spec.name}"


def _eligible_links_for_op(demands, die: int, op) -> tuple:
    eligible = [lnk for lnk in demands.links.values() if not lnk.members or die in lnk.members]
    scope = str(getattr(op, "link_scope", "") or "").strip().lower()
    if scope and scope not in {"any", "undeclared"}:
        scoped = [lnk for lnk in eligible if str(getattr(lnk, "scope", "") or "").strip().lower() == scope]
        if scoped:
            eligible = scoped
    return tuple(eligible)

__all__ = (
    "ResourceChoiceProgram", "FarkasCertificate", "EpochAccount", "DagEdge",
    "EventDagSpec", "OptimalityCertificate", "DualSubgradientReport",
    "BINARY64_ULP_SHIFT", "C11_ULP_GUARD_FACTOR", "ENDPOINT_ROUND_GUARD_NS",
    "check_farkas_dual", "eval_checked_path", "check_optimality_certificate",
    "verify_epoch_trace", "inward_floor_ns", "solve_farkas_dual",
    "solve_checked_critical_path", "certify_lower_bound",
    "build_dual_inputs_from_scenario", "_check_region_summary_bindings",
    "_pure_copy_boundary", "_weight_allocation_id", "_cold_weight_read",
    "derive_default_obligations", "_atomic_ops", "_semantic_ops", "_unique_ops",
    "_engine_modes", "_scenario_rank_ops", "_machine", "certified_capacity_per_second",
)


class _UpperCapacityLike(Protocol):
    @property
    def sku(self) -> str: ...
    def upper_capacity(self, primitive_id: str) -> float | None: ...


def _normalize_capacity_per_second(value: float, unit: str) -> Fraction:
    scale = _CAPACITY_UNIT_SCALE.get(unit.strip().lower())
    if scale is None:
        raise ValueError(f"unknown primitive capacity unit: {unit!r}")
    return Fraction(str(value)) * scale


def _provider_capacity_per_second(table: _UpperCapacityLike, primitive_id: str) -> Fraction | None:
    normalized = getattr(table, "upper_capacity_per_second", None)
    if callable(normalized):
        return normalized(primitive_id)
    capacity = table.upper_capacity(primitive_id)
    if capacity is None:
        return None
    unit_lookup = getattr(table, "unit", None)
    unit = unit_lookup(primitive_id) if callable(unit_lookup) else None
    if unit is None:
        entries = getattr(table, "entries", ())
        if isinstance(entries, Mapping):
            entries = entries.values()
        unit = next((entry.unit for entry in entries if entry.primitive_id == primitive_id), None)
    return _normalize_capacity_per_second(capacity, unit) if unit is not None else None


def certified_capacity_per_second(
    table: PrimitiveTable | _UpperCapacityLike, pipe: str, dtype: str, path: DatapathCapabilitySpec | None,
) -> Fraction | None:
    if not isinstance(table, PrimitiveTable):
        candidates = ([f"{pipe}:{dtype}"] if dtype else []) + [pipe]
        if pipe == "hbm_bus_r":
            candidates.append("P01_hbm_read")
        for primitive in candidates:
            capacity = _provider_capacity_per_second(table, primitive)
            if capacity is not None:
                return capacity
        return None
    if path is None or table.arch_preset != path.arch_id:
        return None
    metrics = _HARDWARE_RATE_METRICS.get(pipe)
    if metrics is None:
        return None
    total = Fraction()
    for primitive, metric, owners in metrics:
        count = getattr(path, owners) if owners else 1
        if not count:
            continue
        cell = table.cell(primitive)
        mode = (dtype if dtype in cell.modes else _MMAD_MODES.get(dtype, dtype)) if pipe == "cube" else None
        if (mode is not None and mode not in cell.modes) or cell.mode_status(mode) == "unsupported":
            return None
        total += Fraction(str(table.certified_upper_capacity(primitive, metric, mode=mode))) * count
    return total if total > 0 else None


@dataclass(frozen=True, slots=True)
class ResourceChoiceProgram:
    num_resources: int
    global_work: tuple[int, ...]
    site_modes: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class FarkasCertificate:
    scale_d: int
    lambda_num: tuple[int, ...]
    mu_num: tuple[int, ...]
    lower_bound_ns: int


@dataclass(frozen=True, slots=True)
class DagEdge:
    src: int
    dst: int
    wire_ns: int


@dataclass(frozen=True, slots=True)
class EventDagSpec:
    node_quantum_ns: tuple[int, ...]
    edges: tuple[DagEdge, ...]


@dataclass(frozen=True, slots=True)
class DualBoundCertificate:
    farkas: FarkasCertificate
    path_edges: tuple[int, ...]
    sink_node_id: int
    l_star_lean_ns: int


@dataclass(frozen=True, slots=True)
class DualSubgradientReport:
    resource_names: tuple[str, ...]
    weights: tuple[Fraction, ...]
    tight_resources: tuple[str, ...]
    lp_optimum_ns: Fraction
    solve_ms: float


def _dot(left, right):
    return sum(map(mul, left, right))


def _sparse_dot(terms, vector):
    return sum(value * vector[i] for i, value in terms)


def check_farkas_dual(prog: ResourceChoiceProgram, cert: FarkasCertificate) -> bool:
    lam = cert.lambda_num
    terms = tuple((i, value) for i, value in enumerate(lam) if value)
    return (
        cert.scale_d > 0
        and len(prog.global_work) == len(lam) == prog.num_resources
        and all(value >= 0 for value in lam)
        and sum(lam) <= cert.scale_d
        and len(cert.mu_num) == len(prog.site_modes)
        and all(
            modes and all(len(mode) == prog.num_resources and mu <= _sparse_dot(terms, mode) for mode in modes)
            for mu, modes in zip(cert.mu_num, prog.site_modes)
        )
        and cert.lower_bound_ns * cert.scale_d <= _sparse_dot(terms, prog.global_work) + sum(cert.mu_num)
    )


def _is_valid_acyclic_dag(dag: EventDagSpec) -> bool:
    quanta, size = dag.node_quantum_ns, len(dag.node_quantum_ns)
    if not size or any(q < 0 for q in quanta):
        return False
    outgoing: list[list[int]] = [[] for _ in quanta]
    indegree = [0] * size
    for e in dag.edges:
        if not (0 <= e.src < size and 0 <= e.dst < size and e.src != e.dst and e.wire_ns >= 0):
            return False
        outgoing[e.src].append(e.dst)
        indegree[e.dst] += 1
    ready, visited = deque(i for i, deg in enumerate(indegree) if not deg), 0
    while ready:
        visited += 1
        for nxt in outgoing[ready.popleft()]:
            indegree[nxt] -= 1
            if not indegree[nxt]:
                ready.append(nxt)
    return visited == size


def eval_checked_path(dag: EventDagSpec, edge_indices: tuple[int, ...], sink_id: int) -> int | None:
    quanta = dag.node_quantum_ns
    if not 0 <= sink_id < len(quanta) or quanta[sink_id] < 0:
        return None
    total, head, seen = quanta[sink_id], sink_id, {sink_id}
    for index in reversed(edge_indices):
        if not 0 <= index < len(dag.edges):
            return None
        edge = dag.edges[index]
        if edge.dst != head or not 0 <= edge.src < len(quanta) or edge.src in seen:
            return None
        if quanta[edge.src] < 0 or edge.wire_ns < 0:
            return None
        total += quanta[edge.src] + edge.wire_ns
        head = edge.src
        seen.add(head)
    return total


def check_optimality_certificate(
    prog: ResourceChoiceProgram, dag: EventDagSpec, cert: DualBoundCertificate,
) -> bool:
    if not _is_valid_acyclic_dag(dag):
        return False
    path = eval_checked_path(dag, cert.path_edges, cert.sink_node_id)
    return (
        path is not None
        and check_farkas_dual(prog, cert.farkas)
        and cert.l_star_lean_ns >= 0
        and cert.l_star_lean_ns == max(cert.farkas.lower_bound_ns, path)
    )


def inward_floor_ns(d_ns: Fraction) -> int:
    if d_ns <= 0:
        return 0
    d_int = d_ns.numerator // d_ns.denominator
    eps = ENDPOINT_ROUND_GUARD_NS + (
        (C11_ULP_GUARD_FACTOR * d_int + (1 << BINARY64_ULP_SHIFT) - 1) >> BINARY64_ULP_SHIFT
    )
    return max(0, d_int - eps)


def _price(weights, base, sites):
    selected = [min(modes, key=lambda mode: _dot(weights, mode)) for modes in sites]
    return tuple(sum(values) for values in zip(base, *selected))


def _solve_master(base, sites):
    size = len(base)
    columns = [None] * size + [_price((0,) * size, base, sites)]
    basis = list(range(size))
    inverse = [[int(i == j) for j in range(size)] for i in range(size)]
    rhs = [1] * size
    denom = 1
    warm_bases = set()
    bland = False
    while True:
        x = [0] * size
        for variable, row in zip(basis, inverse):
            if variable >= size:
                x = [a + b for a, b in zip(x, row)]
        key = tuple(sorted(basis))
        if key in warm_bases:
            bland = True
            warm_bases.clear()
        if not bland:
            warm_bases.add(key)

        column = _price(x, base, sites)
        reduced = _dot(x, column) - denom
        slack_index = min(range(size), key=x.__getitem__)
        if reduced >= 0 and x[slack_index] >= 0:
            slack = {var for var, value in zip(basis, rhs) if var < size and value > 0}
            normalization = sum(x)
            return tuple(Fraction(value, normalization) for value in x), slack, Fraction(denom, normalization)

        if bland:
            entering = next((j for j, value in enumerate(x) if value < 0), None)
            if entering is None:
                entering = next((j for j in range(size, len(columns)) if _dot(x, columns[j]) < denom), None)
        else:
            entering = slack_index if x[slack_index] < min(0, reduced) else None
        if entering is None:
            if column not in columns:
                columns.append(column)
            entering = columns.index(column)

        if entering < size:
            direction = [row[entering] for row in inverse]
        else:
            direction = [_dot(row, columns[entering]) for row in inverse]
        leaving = None
        for i, value in enumerate(direction):
            if value <= 0:
                continue
            if leaving is None:
                leaving = i
                continue
            left, right = rhs[i] * direction[leaving], rhs[leaving] * value
            if left < right or (left == right and basis[i] < basis[leaving]):
                leaving = i
        if leaving is None:
            raise ModelingError("dual_bound_simplex_unbounded")
        pivot = direction[leaving]
        pivot_row = inverse[leaving]
        pivot_rhs = rhs[leaving]
        for i, factor in enumerate(direction):
            if i != leaving:
                inverse[i] = [(a * pivot - b * factor) // denom for a, b in zip(inverse[i], pivot_row)]
                rhs[i] = (rhs[i] * pivot - pivot_rhs * factor) // denom
        basis[leaving] = entering
        denom = pivot


def _independent_masters(base, sites):
    members = [{i for mode in modes for i, work in enumerate(mode) if work} for modes in sites]
    connected = defaultdict(list)
    for site, rows in enumerate(members):
        for row in rows:
            connected[row].append(site)
    unseen = set(connected)
    weights = [Fraction(0)] * len(base)
    optimum = max((base[i] for i in range(len(base)) if i not in unseen), default=0)
    tight = {i for i, value in enumerate(base) if i not in unseen and value == optimum}
    for i in tight:
        weights[i] = Fraction(1, len(tight)) if optimum else Fraction(0)
    while unseen:
        pending = [min(unseen)]
        rows, group = set(), set()
        while pending:
            row = pending.pop()
            if row in rows:
                continue
            rows.add(row)
            for site in connected[row]:
                if site not in group:
                    group.add(site)
                    pending.extend(members[site])
        unseen.difference_update(rows)
        indices = sorted(rows)
        fixed = tuple(base[i] for i in indices)
        modes = [tuple(tuple(mode[i] for i in indices) for mode in sites[j]) for j in sorted(group)]
        shift = max(1, 1 - min(fixed))
        shifted = tuple(value + shift for value in fixed)
        if min(fixed) < 0:
            shifted += (shift,)
            modes = [tuple(mode + (0,) for mode in site) for site in modes]
        solution, slack, objective = _solve_master(shifted, modes)
        value = objective - shift
        if value > optimum:
            weights = [Fraction(0)] * len(base)
            for i, weight in zip(indices, solution):
                weights[i] = weight
            optimum, tight = value, set()
        if value == optimum:
            tight.update(row for i, row in enumerate(indices) if i not in slack)
    return tuple(weights), tight


def solve_farkas_dual(prog: ResourceChoiceProgram) -> tuple[FarkasCertificate, DualSubgradientReport]:
    start = perf_counter()
    size = prog.num_resources
    if len(prog.global_work) != size or any(
        not modes or any(len(mode) != size for mode in modes) for modes in prog.site_modes
    ):
        raise ValueError("inconsistent resource-choice dimensions")
    names = tuple(f"r{i}" for i in range(size))
    zero_modes = []
    if all(w <= 0 for w in prog.global_work):
        for modes in prog.site_modes:
            mode = next((candidate for candidate in modes if max(candidate, default=0) <= 0), None)
            if mode is None:
                break
            zero_modes.append(mode)
    if len(zero_modes) == len(prog.site_modes) and all(w <= 0 for w in prog.global_work):
        weights = (Fraction(0),) * size
        tight = tuple(
            name for name, values in zip(names, zip(prog.global_work, *zero_modes)) if sum(values) == 0
        )
    else:
        base = list(prog.global_work)
        sites = []
        for modes in prog.site_modes:
            minimum = tuple(map(min, zip(*modes)))
            base = [a + b for a, b in zip(base, minimum)]
            residuals = tuple(dict.fromkeys(tuple(a - b for a, b in zip(mode, minimum)) for mode in modes))
            if all(any(mode) for mode in residuals):
                sites.append(residuals)
        if not sites:
            optimum = max(0, *base)
            tied = [i for i, value in enumerate(base) if value == optimum]
            weights = tuple(
                Fraction(1, len(tied)) if i in tied and optimum else Fraction(0) for i in range(size)
            )
            tight = tuple(names[i] for i in tied)
        else:
            weights, active = _independent_masters(base, sites)
            tight = tuple(name for i, name in enumerate(names) if i in active)

    scale = lcm(*(value.denominator for value in weights))
    lam = tuple(int(value * scale) for value in weights)
    terms = tuple((i, value) for i, value in enumerate(lam) if value)
    mu = tuple(min(_sparse_dot(terms, mode) for mode in modes) for modes in prog.site_modes)
    objective = Fraction(_sparse_dot(terms, prog.global_work) + sum(mu), scale)
    cert = FarkasCertificate(scale, lam, mu, max(0, objective.numerator // objective.denominator))
    if not check_farkas_dual(prog, cert):
        raise ModelingError("dual_bound_farkas_certificate_invalid")
    return cert, DualSubgradientReport(names, weights, tight, objective, (perf_counter() - start) * MS_PER_SECOND)


def solve_checked_critical_path(dag: EventDagSpec) -> tuple[tuple[int, ...], int, int]:
    quanta = dag.node_quantum_ns
    size = len(quanta)
    if not size or any(q < 0 for q in quanta):
        raise ValueError("event DAG requires nonnegative node quanta")
    outgoing: list[list[tuple[int, DagEdge]]] = [[] for _ in quanta]
    indegree = [0] * size
    for index, edge in enumerate(dag.edges):
        if not (0 <= edge.src < size and 0 <= edge.dst < size and edge.wire_ns >= 0):
            raise ValueError("invalid event DAG edge")
        outgoing[edge.src].append((index, edge))
        indegree[edge.dst] += 1
    ready = deque(i for i, degree in enumerate(indegree) if not degree)
    distance, parent = list(quanta), [None] * size
    visited = 0
    while ready:
        src = ready.popleft()
        visited += 1
        for index, edge in outgoing[src]:
            dst = edge.dst
            candidate = distance[src] + edge.wire_ns + quanta[dst]
            if candidate > distance[dst]:
                distance[dst], parent[dst] = candidate, index
            indegree[dst] -= 1
            if not indegree[dst]:
                ready.append(dst)
    if visited != size:
        raise ValueError("cyclic event DAG")
    sink = node = max(range(size), key=distance.__getitem__)
    reverse_path = []
    while (index := parent[node]) is not None:
        reverse_path.append(index)
        node = dag.edges[index].src
    path = tuple(reversed(reverse_path))
    if eval_checked_path(dag, path, sink) != distance[sink]:
        raise ModelingError("dual_bound_critical_path_mismatch")
    return path, sink, distance[sink]


def solve_dual_lower_bound(
    prog: ResourceChoiceProgram,
    dag: EventDagSpec,
    *,
    cert_cls: type[DualBoundCertificate] = DualBoundCertificate,
) -> tuple[DualBoundCertificate, DualSubgradientReport]:
    farkas, report = solve_farkas_dual(prog)
    path, sink, path_ns = solve_checked_critical_path(dag)
    cert = cert_cls(farkas, path, sink, max(0, farkas.lower_bound_ns, path_ns))
    if not check_optimality_certificate(prog, dag, cert):
        raise ModelingError("dual_bound_optimality_certificate_invalid")
    return cert, report


def _unique_ops(entries) -> tuple[OpSpec, ...]:
    by_id: dict[str, OpSpec] = {}
    for entry in entries.values() if isinstance(entries, Mapping) else (entries or ()):
        for op in getattr(entry, "ops", (entry,)):
            if hasattr(op, "op_id"):
                by_id.setdefault(op.op_id, op)
    return tuple(by_id.values())


def _scenario_rank_ops(scenario):
    entries = getattr(scenario, "atomic_ops", None)
    if entries is None:
        graph = getattr(scenario, "atomic_graph", ())
        entries = getattr(graph, "op_specs", getattr(graph, "ops", graph))
    if isinstance(entries, Mapping):
        for rep_key, entry in entries.items():
            rep_id = int(getattr(entry, "replica", rep_key if isinstance(rep_key, int) else 0))
            for op in getattr(entry, "ops", (entry,)):
                if hasattr(op, "metadata"):
                    yield rep_id, op
        return
    for entry in entries or ():
        rep_id = int(getattr(entry, "replica", 0))
        for op in getattr(entry, "ops", (entry,)):
            if hasattr(op, "metadata"):
                yield rep_id, op


def _atomic_ops(scenario) -> tuple[tuple[int, OpSpec], ...]:
    """Return deduplicated ``(replica, OpSpec)`` pairs preserving replica identity."""
    by_key: dict[tuple[int, str], OpSpec] = {}
    for replica, op in _scenario_rank_ops(scenario):
        by_key.setdefault((replica, op.op_id), op)
    return tuple((replica, op) for (replica, _op_id), op in by_key.items())


def _op_contract_key(op, metadata):
    if (
        metadata.get("pure") is not True
        or metadata.get("load_time")
        or metadata.get("state_effect")
        or op.link_bytes
    ):
        return None
    ports = metadata.get("input_port_versions")
    contract = metadata.get("semantic_contract", metadata.get("contract"))
    if ports is None or contract is None:
        return None
    return op.family, op.stage, op.layer, op.requests, op.local_shape, op.dtype, op.layout, repr(ports), repr(contract)


def _proven_hoisted_sites(scenario, applied: AppliedPlan | None) -> frozenset[tuple[int, str]]:
    """Return proven replica-scoped ``(replica, op_id)`` sites hoisted to load time."""
    if applied is None:
        return frozenset()
    scoped = getattr(applied, "hoisted_load_time_sites", ())
    if scoped:
        return frozenset((int(replica), str(op_id)) for replica, op_id in scoped)
    if not applied.hoisted_load_time:
        return frozenset()
    replicas = {entry.replica for entry in getattr(scenario, "atomic_ops", ())} or {0}
    return frozenset((int(replica), str(op_id)) for replica in replicas for op_id in applied.hoisted_load_time)


def _resolved_op_deps(op, meta: Mapping[str, object], in_graph: set[str] | None = None) -> tuple[str, ...]:
    """Collect declared data/control prerequisite op IDs from ``deps``, ports, and typed inputs."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(dep: object) -> None:
        if isinstance(dep, str) and dep and dep != op.op_id and dep not in seen:
            if in_graph is None or dep in in_graph:
                seen.add(dep)
                found.append(dep)

    for dep in getattr(op, "deps", ()):
        _add(dep)
    for dep, _ in input_port_entries(getattr(op, "metadata", ())):
        _add(dep)
    for key in ("typed_inputs", "input_ports"):
        ports = meta.get(key)
        if isinstance(ports, (tuple, list)):
            for port in ports:
                if isinstance(port, Mapping):
                    _add(port.get("producer"))
    if in_graph is not None:
        for val in meta.values():
            if isinstance(val, str) and val in in_graph:
                _add(val)
    return tuple(found)


def _hoisted_source_port(op, meta: Mapping[str, object], src_id: str) -> tuple[str, int]:
    """Resolve the ``(port_name, version)`` consumed from a single hoisted source op."""
    for key in ("typed_inputs", "input_ports"):
        ports = meta.get(key)
        if isinstance(ports, (tuple, list)):
            for port in ports:
                if isinstance(port, Mapping) and port.get("producer") == src_id:
                    name = str(port.get("port") or port.get("output") or port.get("name") or "out")
                    ver = port.get("version", 0)
                    return name, (ver if type(ver) is int and ver >= 0 else 0)
    for dep, port_ref in input_port_entries(getattr(op, "metadata", ())):
        if dep == src_id:
            suffix = port_ref.removeprefix(f"{src_id}:") if isinstance(port_ref, str) else "out"
            return (suffix or "out"), 0
    return "out", 0


def _semantic_ops(
    scenario,
    *,
    hoisted_sites: frozenset[tuple[int, str]] = frozenset(),
) -> tuple[list[tuple[int, OpSpec, dict]], dict[tuple[int, str], tuple[int, str]]]:
    """Deduplicate pure identical-contract ops strictly within each replica."""
    raw_ops = list(_atomic_ops(scenario))
    by_replica: dict[int, list[tuple[OpSpec, dict]]] = defaultdict(list)
    for replica, op in raw_ops:
        by_replica[replica].append((op, dict(op.metadata)))
    valid_hoisted: set[tuple[int, str]] = set()
    for replica, items in by_replica.items():
        in_graph = {op.op_id for op, _ in items}
        candidates = [
            (op, meta, set(_resolved_op_deps(op, meta, in_graph)))
            for op, meta in items
            if (meta.get("load_time") or (replica, op.op_id) in hoisted_sites)
            and not getattr(op, "link_bytes", 0)
            and not meta.get("state_effect")
        ]
        ready: set[str] = set()
        progressed = True
        while progressed:
            progressed = False
            for op, _, deps in candidates:
                if op.op_id not in ready and deps <= ready:
                    ready.add(op.op_id)
                    progressed = True
        for oid in ready:
            valid_hoisted.add((replica, oid))
    seen: dict[tuple, tuple[int, str]] = {}
    aliases: dict[tuple[int, str], tuple[int, str]] = {}
    operations: list[tuple[int, OpSpec, dict]] = []
    for replica, op in raw_ops:
        metadata = dict(op.metadata)
        if (replica, op.op_id) in valid_hoisted:
            aliases[replica, op.op_id] = (replica, op.op_id)
            continue
        key = _op_contract_key(op, metadata)
        if key is not None:
            scoped_key = (replica, key)
            if scoped_key in seen:
                aliases[replica, op.op_id] = seen[scoped_key]
                continue
            seen[scoped_key] = (replica, op.op_id)
        operations.append((replica, op, metadata))
    return operations, aliases


def _machine(scenario):
    machine = getattr(scenario, "machine", None)
    if machine is not None:
        return machine
    atomic = getattr(scenario, "atomic_step", None)
    case = getattr(atomic, "case", None) or getattr(scenario, "case", None)
    return getattr(case, "machine", None)


def _weight_allocation_id(meta: Mapping[str, object]) -> str | None:
    return next((
        value for key in _WEIGHT_ID_KEYS
        if isinstance(value := meta.get(key), str) and value.strip()
    ), None)


def _cold_weight_read(op, meta: Mapping[str, object], die: int) -> TensorRangeObligation | None:
    mte2_bytes = getattr(op, "mte2_bytes", 0)
    if mte2_bytes <= 0 or op.family not in _COLD_WEIGHT_FAMILIES:
        return None
    identity = _weight_allocation_id(meta)
    residency = meta.get("weight_initial_residency", meta.get("initial_residency"))
    if identity is None or residency != "hbm_cold":
        return None
    decl = meta.get("catlass_problem")
    raw_base = decl.get("b_base_byte", 0) if isinstance(decl, Mapping) else getattr(decl, "b_base_byte", 0)
    raw_ver = decl.get("b_version", 0) if isinstance(decl, Mapping) else getattr(decl, "b_version", 0)
    decl_base = raw_base if type(raw_base) is int and raw_base >= 0 else 0
    decl_ver = raw_ver if type(raw_ver) is int and raw_ver >= 0 else 0
    ranges = meta.get("weight_ranges", meta.get("ranges"))
    if ranges is None:
        size = (
            meta.get("packed_weight_bytes")
            or meta.get("weight_payload_bytes")
            or meta.get("weight_bytes")
            or meta.get("local_weight_bytes")
        )
        if not isinstance(size, (int, float)):
            return None
        touched = min(size, mte2_bytes)
        if not touched >= 1:
            return None
        ranges = ((decl_base, decl_base + int(touched)),)
    meta_ver = meta.get("weight_tensor_version", meta.get("tensor_version", 0))
    version = decl_ver if decl_ver else (meta_ver if type(meta_ver) is int and meta_ver >= 0 else 0)
    return TensorRangeObligation(die, identity, version, tuple(ranges), "hbm_cold", "owner")


def _pure_copy_boundary(
    op, meta: Mapping[str, object], die: int, replica: int, stage: int, consumed: set[tuple[int, str]],
    aliases: Mapping[tuple[int, str], tuple[int, str]] | None = None,
):
    """Return ``(read_hbm_ob, write_hbm_ob, mte2_copy_bytes, write_copy_bytes)`` independently."""
    hbm = int(getattr(op, "hbm_bytes", 0) or 0)
    mte2 = int(getattr(op, "mte2_bytes", 0) or 0)
    mte3 = int(getattr(op, "mte3_bytes", 0) or 0)
    warm = meta.get("initial_residency") == "l2_warm"
    read_ob = write_ob = None
    mte2_copy = write_copy = 0
    read_claimed = 0
    alias_map = aliases or {}
    full_deps = _resolved_op_deps(op, meta)
    runtime_deps = tuple(
        dep for dep in full_deps
        if (replica, dep) not in alias_map or alias_map[replica, dep] != (replica, dep)
    )
    hoisted_deps = tuple(
        dep for dep in full_deps
        if (replica, dep) in alias_map and alias_map[replica, dep] == (replica, dep)
    )
    if not runtime_deps:
        if len(hoisted_deps) == 1:
            src_port, src_ver = _hoisted_source_port(op, meta, hoisted_deps[0])
            read_alloc_id = f"hoisted:r{replica}:{hoisted_deps[0]}:{src_port}"
            read_ver = src_ver
        else:
            read_alloc_id = f"boundary_in:r{replica}.p{stage}:{op.op_id}"
            read_ver = 0
        if mte2 > 0:
            mte2_copy = mte2
            if hbm > 0 and not warm:
                read_sz = min(mte2, hbm)
                if read_sz > 0:
                    read_claimed = read_sz
                    read_ob = TensorRangeObligation(
                        die, read_alloc_id, read_ver,
                        ((0, read_sz),), "hbm_cold", "owner",
                    )
        elif mte3 == 0 and hbm > 0 and not warm:
            read_claimed = hbm
            read_ob = TensorRangeObligation(
                die, read_alloc_id, read_ver,
                ((0, hbm),), "hbm_cold", "owner",
            )
    if (replica, op.op_id) not in consumed:
        if mte3 > 0:
            write_copy = mte3
            rem_hbm = max(0, hbm - read_claimed)
            if rem_hbm > 0 and not warm:
                write_sz = min(mte3, rem_hbm)
                if write_sz > 0:
                    write_ob = TensorRangeObligation(
                        die, f"boundary_out:r{replica}.p{stage}:{op.op_id}", 0,
                        ((0, write_sz),), "onchip_resident", "device",
                    )
        elif mte2 == 0 and hbm > 0 and not warm and read_ob is None:
            write_ob = TensorRangeObligation(
                die, f"boundary_out:r{replica}.p{stage}:{op.op_id}", 0,
                ((0, hbm),), "onchip_resident", "device",
            )
    return read_ob, write_ob, mte2_copy, write_copy


def _resolve_op_precision(op, stage_devices, policy: str):
    if not stage_devices or not isinstance(op, OpSpec):
        return op
    return resolve_matrix_precision(op, devices=stage_devices, policy=policy, recipe=None)


def derive_default_obligations(scenario, demands, aliases, covered, rank_ops_fn):
    """Derive per-(replica, stage, die, op) default obligations and die placements."""
    atomic = getattr(scenario, "atomic_step", None)
    case = getattr(atomic, "case", None) or getattr(scenario, "case", None)
    dies = tuple(demands.devices)
    parallel = getattr(getattr(case, "runtime", None), "parallel", None)
    tp_width = (int(parallel.tp) * int(parallel.pcp)) if parallel is not None else 1
    pp_stages = int(parallel.pp) if parallel is not None else 1
    default_dt = str(getattr(getattr(case, "model", None), "compute_dtype", "") or "").strip()
    matrix_policy = str(getattr(getattr(case, "runtime", None), "matrix_compute_policy", "auto") or "auto")

    rank_ops = tuple(rank_ops_fn(scenario))
    consumed = {(replica, dep) for replica, op in rank_ops for dep in getattr(op, "deps", ())}
    multi_rank = len(demands.devices) > 1

    matrix_work: dict[tuple[int, int, int, str], int] = defaultdict(int)
    vector_work: dict[tuple[int, int, int], int] = defaultdict(int)
    link_work: dict[tuple[int, int, int], int] = defaultdict(int)
    switch_work: dict[str, int] = defaultdict(int)
    shared_link_work: dict[str, int] = defaultdict(int)
    mte2_copy_work: dict[tuple[int, int, int], int] = defaultdict(int)
    write_copy_work: dict[tuple[int, int, int], int] = defaultdict(int)
    read_obs: list[TensorRangeObligation] = []
    write_obs: list[TensorRangeObligation] = []
    placements: dict[int, tuple[int, int]] = {}
    stage_devices_cache: dict[tuple[int, int], tuple] = {}
    seen: set[tuple[int, int, str]] = set()

    for replica, op in rank_ops:
        if (replica, op.op_id) in aliases or op.op_id in aliases:
            continue
        meta = dict(op.metadata)
        stage = int(getattr(op, "stage", 0))
        die_idx = (replica * pp_stages + stage) * tp_width
        default_die = dies[die_idx] if 0 <= die_idx < len(dies) else (dies[0] if dies else 0)
        die = _resolve_comm_endpoint_die(
            meta, ("source_die", "src_die", "source_rank", "src_rank", "source", "die", "rank"),
            default_die, dies,
        )
        assert die is not None
        placements.setdefault(die, (replica, stage))
        key = (replica, stage, op.op_id)
        has_explicit_work = bool(meta.get("semantic_work"))
        has_matrix = has_vector = has_link = False
        if not has_explicit_work and key not in seen and key not in covered:
            seen.add(key)
            resolved_op = op
            weight_fmt = str(meta.get("weight_format") or meta.get("weight_dtype") or "").strip().lower()
            if weight_fmt in PACKED_FP4_FORMATS and isinstance(op, OpSpec):
                sd_key = (replica, stage)
                stage_devices = stage_devices_cache.get(sd_key)
                if stage_devices is None:
                    stage_ranks = (
                        dies[die_idx:die_idx + tp_width]
                        if 0 <= die_idx < len(dies)
                        else ((die,) if die in demands.devices else ())
                    )
                    stage_devices = tuple(demands.devices[r] for r in stage_ranks if r in demands.devices)
                    stage_devices_cache[sd_key] = stage_devices
                resolved_op = _resolve_op_precision(op, stage_devices, matrix_policy)
            dtype = str(getattr(resolved_op, "dtype", "") or default_dt).strip()
            pairs = getattr(resolved_op, "matrix_flops", ()) or ((dtype, getattr(resolved_op, "flops", 0)),)
            for dt, flops in pairs:
                if flops > 0:
                    if not str(dt).strip():
                        raise ModelingError(f"dual_obligations_dtype_missing:{op.op_id}")
                    matrix_work[replica, stage, die, str(dt)] += int(flops)
                    has_matrix = True
            vec_ops = int(getattr(op, "vector_ops", 0) or 0)
            if vec_ops > 0 and not meta.get("inline_epilogue_absorbable"):
                vector_work[replica, stage, die] += vec_ops
                has_vector = True
            link_bytes = int(getattr(op, "link_bytes", 0) or 0)
            if link_bytes > 0 and not meta.get("mandatory_comm_wire_bytes_by_link") and (
                multi_rank or bool(demands.links)
            ):
                if demands.engine_rate((replica, stage, die), "link_egress", "") > 0:
                    link_work[replica, stage, die] += link_bytes
                    has_link = True
                eligible = _eligible_links_for_op(demands, die, op)
                sw_pipes = {_link_switch_pipe(lnk) for lnk in eligible}
                if len(eligible) >= 1 and len(sw_pipes) == 1:
                    only_sw = next(iter(sw_pipes))
                    if only_sw is not None:
                        switch_work[only_sw] += link_bytes
                if len(eligible) == 1 and eligible[0].semantics not in {"per_pair", *_PORT_LINK_SEMANTICS}:
                    shared_link_work[f"link:{eligible[0].name}"] += link_bytes

        has_explicit_mem = "read_obligations" in meta or "write_obligations" in meta
        if "read_obligations" in meta:
            read_obs.extend(meta["read_obligations"])
        elif key not in covered and (read := _cold_weight_read(op, meta, die)) is not None:
            read_obs.append(read)
            has_explicit_mem = True
        write_obs.extend(meta.get("write_obligations", ()))

        if not (has_explicit_work or has_explicit_mem or has_matrix or has_vector or has_link) and key not in covered:
            b_read, b_write, b_mte2, b_mte3 = _pure_copy_boundary(
                op, meta, die, replica, stage, consumed, aliases,
            )
            if b_read is not None:
                read_obs.append(b_read)
            if b_write is not None:
                write_obs.append(b_write)
            if b_mte2 > 0:
                mte2_copy_work[replica, stage, die] += b_mte2
            if b_mte3 > 0:
                write_copy_work[replica, stage, die] += b_mte3

    for (replica, stage, die), byte_count in link_work.items():
        demands.charge((replica, stage, die), "link_egress", byte_count)
    for sw_pipe, byte_count in switch_work.items():
        demands.charge((0, 0, -1), sw_pipe, byte_count)
    for lnk_pipe, byte_count in shared_link_work.items():
        demands.charge((0, 0, -1), lnk_pipe, byte_count)
    for (replica, stage, die), byte_count in mte2_copy_work.items():
        demands.charge((replica, stage, die), "mte2_total", byte_count)
    for (replica, stage, die), byte_count in write_copy_work.items():
        demands.charge((replica, stage, die), "write_total", byte_count)

    work_items: list[SemanticWorkItem] = []
    for (replica, stage, die, dt), flops in matrix_work.items():
        path = demands.paths.get(die)
        engines = ("cube", "aiv_simt") if path and path.simt_ub_carveout_bytes else ("cube",)
        work_items.append(SemanticWorkItem(
            f"mac:r{replica}.p{stage}.d{die}:{dt}", "matrix_mac", dt, flops, engines,
            die=die, replica=replica, pp_stage=stage,
        ))
    for (replica, stage, die), ops_count in vector_work.items():
        path = demands.paths.get(die)
        engines = ("aiv_simd", "aiv_simt") if path and path.simt_ub_carveout_bytes else ("aiv_simd",)
        work_items.append(SemanticWorkItem(
            f"vec:r{replica}.p{stage}.d{die}", "vector_alu", "fp32", ops_count, engines,
            die=die, replica=replica, pp_stage=stage,
        ))

    if not (work_items or read_obs or write_obs):
        return (), placements
    return (FusedRegionSummary(
        tier="K1", region_id="scenario_compulsory_floor", op_ids=(), semantic_work=tuple(work_items),
        issued_cube_flops_by_dtype=(), vector_ops_by_pipe=(),
        read_obligations=tuple(read_obs), write_obligations=tuple(write_obs),
        logical_transfer_bytes=0, issued_copy_bytes=0, dram_sector_bytes=0, turnaround_service_bytes=Fraction(),
        realized_l2_hit_bytes=0, comm_wire_bytes_by_link=(), stars_launch_count=0,
        aicpu_metadata_ms=Fraction(), sram_peaks_bytes=(), spec=None,
    ),), placements


class _Demands:
    def __init__(self, scenario, primitive_table, rate_mode):
        if rate_mode not in {"engine_ceiling", "certified_upper", "certified_upper_capacity"}:
            raise ValueError(f"unknown lower-bound rate mode {rate_mode}")
        self.certified = rate_mode != "engine_ceiling"
        self.table = primitive_table
        machine = _machine(scenario)
        self.devices = {device.rank: device for device in getattr(machine, "dies", ())}
        self.paths = {die: datapath_for_die(device) for die, device in self.devices.items()}
        self.links = {link.name: link for link in getattr(machine, "links", ())}
        if self.table is not None:
            for device in self.devices.values():
                if getattr(device, "sku", ""):
                    validate_table_sku_compatibility(self.table, device.sku)
        self.fixed = defaultdict(Fraction)
        self.sites = []

    def rate(self, scope, pipe, dtype=""):
        engine = self.engine_rate(scope, pipe, dtype)
        if not self.certified or engine <= 0:
            return engine
        device = self.devices.get(scope[-1])
        if self.table is None:
            return 0
        if device and getattr(device, "sku", ""):
            validate_table_sku_compatibility(self.table, device.sku)
        capacity = certified_capacity_per_second(self.table, pipe, dtype, self.paths.get(scope[-1]))
        return max(capacity, engine) if capacity is not None else 0

    def engine_rate(self, scope, pipe, dtype):
        if pipe == "link_egress":
            die = scope[-1]
            total = Fraction(0)
            for link in self.links.values():
                if die not in link.members and link.members:
                    continue
                base_bpms = Fraction(link.effective_peak_capacity_bytes_per_ms)
                ep_rates = [Fraction(r) for rank, r in link.endpoint_rates if rank == die]
                if link.semantics == "per_pair":
                    peers = (
                        sum(1 for rank in link.members if rank != die)
                        if link.members else max(1, len(self.devices) - 1)
                    )
                    egress_bpms = base_bpms * max(1, peers)
                    if ep_rates:
                        egress_bpms = min(egress_bpms, max(ep_rates))
                elif link.semantics in _PORT_LINK_SEMANTICS:
                    egress_bpms = max(ep_rates) if ep_rates else base_bpms
                else:
                    egress_bpms = max((base_bpms, *ep_rates))
                if link.switch_capacity_bytes_per_ms is not None:
                    egress_bpms = min(egress_bpms, Fraction(link.switch_capacity_bytes_per_ms))
                total += MS_PER_SECOND * egress_bpms
            return total
        if pipe.startswith("link_switch_id:"):
            sw_id = pipe.removeprefix("link_switch_id:")
            caps = [
                Fraction(lnk.switch_capacity_bytes_per_ms)
                for lnk in self.links.values()
                if str(getattr(lnk, "switch_id", "") or "").strip() == sw_id
                and lnk.switch_capacity_bytes_per_ms is not None
            ]
            return MS_PER_SECOND * max(caps) if caps else 0
        if pipe.startswith("link_switch:"):
            link = self.links.get(pipe.removeprefix("link_switch:"))
            return (
                MS_PER_SECOND * Fraction(link.switch_capacity_bytes_per_ms)
                if (link is not None and link.switch_capacity_bytes_per_ms is not None)
                else 0
            )
        if pipe.startswith("link_rx:"):
            link = self.links.get(pipe.removeprefix("link_rx:"))
            if link is None:
                return 0
            die = scope[-1]
            base_bpms = Fraction(link.effective_peak_capacity_bytes_per_ms)
            ep_rates = [Fraction(r) for rank, r in link.endpoint_rates if rank == die]
            cap_bpms = max(ep_rates) if ep_rates else base_bpms
            if link.switch_capacity_bytes_per_ms is not None:
                cap_bpms = min(cap_bpms, Fraction(link.switch_capacity_bytes_per_ms))
            return MS_PER_SECOND * cap_bpms
        if pipe.startswith("link:"):
            link = self.links.get(pipe.removeprefix("link:"))
            if link is None:
                return 0
            die = scope[-1]
            base_bpms = Fraction(link.effective_peak_capacity_bytes_per_ms)
            ep_rates = [Fraction(r) for rank, r in link.endpoint_rates if rank == die]
            if link.semantics == "per_pair" and die >= 0:
                peers = (
                    sum(1 for rank in link.members if rank != die)
                    if link.members else max(1, len(self.devices) - 1)
                )
                cap_bpms = base_bpms * max(1, peers)
                if ep_rates:
                    cap_bpms = min(cap_bpms, max(ep_rates))
                if link.switch_capacity_bytes_per_ms is not None:
                    cap_bpms = min(cap_bpms, Fraction(link.switch_capacity_bytes_per_ms))
                return MS_PER_SECOND * cap_bpms
            if link.semantics in _PORT_LINK_SEMANTICS and die >= 0:
                cap_bpms = max(ep_rates) if ep_rates else base_bpms
                if link.switch_capacity_bytes_per_ms is not None:
                    cap_bpms = min(cap_bpms, Fraction(link.switch_capacity_bytes_per_ms))
                return MS_PER_SECOND * cap_bpms
            cap_bpms = base_bpms
            if link.switch_capacity_bytes_per_ms is not None:
                cap_bpms = min(cap_bpms, Fraction(link.switch_capacity_bytes_per_ms))
            return MS_PER_SECOND * cap_bpms
        device = self.devices.get(scope[-1])
        if device is None:
            return 0
        path = self.paths[scope[-1]]
        rates = dict(device.rates)
        if pipe == "cube":
            return device.peak(dtype)
        if pipe == "vector":
            return rates.get("vector", 0)
        if pipe == "simt":
            return rates.get("simt", rates.get("vector", 0))
        if pipe == "hbm_bus_r":
            return max(Fraction(path.hbm_read_bytes_per_s), Fraction(rates.get("hbm", 0)))
        if pipe == "hbm_bus_w":
            return max(Fraction(path.hbm_write_bytes_per_s), Fraction(rates.get("hbm", 0)))
        if pipe == "mte2_total":
            split = sum(Fraction(rates.get(key, 0)) for key in ("mte2_aic", "mte2_aiv"))
            return split if split > 0 else Fraction(rates.get("mte2", 0))
        if pipe == "write_total":
            return sum(Fraction(rates.get(key, 0)) for key in ("fixpipe", "mte3"))
        return rates.get(pipe, rates.get("l2", 0) if pipe == "l2_bw" else 0)

    def charge(self, scope, pipe, work, *, dtype="", rate_pipe=None, rows=None):
        rate = self.rate(scope, rate_pipe or pipe, dtype)
        if rate > 0 and work > 0:
            destination = self.fixed if rows is None else rows
            destination[(*scope, pipe)] += Fraction(work) * NS_PER_SECOND / Fraction(rate)

    def transfer(self, scope, direction, size, *, rows=None, copy=True):
        read = direction == "read"
        own = "hbm_bus_r" if read else "hbm_bus_w"
        other = "hbm_bus_w" if read else "hbm_bus_r"
        path = self.paths.get(scope[-1])
        self.charge(scope, own, size, rows=rows)
        if path is not None and not self.certified:
            self.charge(scope, other, Fraction(path.hbm_rw_alpha) * size, rate_pipe=own, rows=rows)
        if copy:
            self.charge(scope, "mte2_total" if read else "write_total", size, rows=rows)

    def add_site(self, modes):
        if len(modes) == 1:
            for resource, ns in modes[0].items():
                self.fixed[resource] += ns
        elif modes:
            self.sites.append(modes)

    def program(self):
        resources = tuple(sorted(self.fixed.keys() | {
            resource for modes in self.sites for mode in modes for resource in mode
        }))
        names = tuple(f"{pipe}@r{replica}.p{stage}.d{die}" for replica, stage, die, pipe in resources)
        fixed = tuple(inward_floor_ns(self.fixed[resource]) for resource in resources)
        sites = tuple(
            tuple(tuple(inward_floor_ns(mode.get(resource, Fraction())) for resource in resources) for mode in modes)
            for modes in self.sites
        )
        return ResourceChoiceProgram(len(resources), fixed, sites), names


def _engine_modes(demands, item):
    scope = item.replica, item.pp_stage, item.die
    modes = []
    for engine in dict.fromkeys(item.admitted_engines):
        mode = defaultdict(Fraction)
        pipe = _ENGINE_PIPE_MAP[engine]
        demands.charge(scope, pipe, item.logical_ops, dtype=item.dtype if pipe == "cube" else "", rows=mode)
        modes.append(mode)
    if item.inline_epilogue_absorbable:
        modes.append({})
    path = demands.paths.get(item.die)
    if (
        item.kind == "sort_topk"
        and path
        and path.simt_ub_carveout_bytes
        and "radix_hist256" in item.admissible_schedulers
    ):
        radix = defaultdict(Fraction)
        demands.charge(scope, "vector", item.logical_ops, rate_pipe="radix_hist256", rows=radix)
        if radix:
            modes.append(radix)
    return modes


def _memory_demands(demands, scenario, summaries, placements, contractions):
    by_die = defaultdict(list)
    for spec in contractions:
        by_die[spec.die].append(spec)
    sectors = getattr(scenario, "sector_bytes", SECTOR_BYTES)
    cuts = compute_compulsory_hbm_cut(summaries, sector_bytes=sectors)
    for die, cut in cuts.items():
        specs = by_die.get(die, ())
        default = (specs[0].replica, specs[0].pp_stage) if specs else (0, 0)
        scope = (*placements.get(die, default), die)
        demands.transfer(scope, "read", cut.read_bytes, copy=False)
        demands.transfer(scope, "write", cut.write_bytes, copy=False)
    for die, specs in by_die.items():
        first = specs[0]
        scope = (*placements.get(die, (first.replica, first.pp_stage)), die)
        results = tuple(direct_mac_pebble_reload_bound(spec) for spec in specs)
        for site in build_surplus_coupling_sites(
            cuts, results, write_free_by_die={die: all(spec.write_free for spec in specs)},
        ):
            modes = []
            for name, _ in site.modes:
                direction = "read" if name == "mode_surplus_read" else "write"
                mode = defaultdict(Fraction)
                demands.transfer(scope, direction, site.surplus_bytes, rows=mode)
                modes.append(mode)
            demands.add_site(modes)


def _check_region_summary_bindings(scenario, applied, summaries, *, require_catalog: bool = False):
    if not summaries:
        return
    from .apply import _die

    fusions = {f"rewrite:{edit.edit_id}" for edit in applied.plan.edits if isinstance(edit.operation, FuseRegion)}
    members = {region: atoms for region, atoms in applied.absorbed_atoms if region in fusions}
    members.update(
        (binding.target_op, (binding.target_op,))
        for binding in applied.template_bindings
        if binding.target_op not in members
    )
    bound = defaultdict(set)
    for rank in applied.executable:
        for op in rank.ops:
            if op.op_id in members and (op.op_id not in fusions or op.family == "rewrite_fused"):
                die = _die(scenario, rank.replica, op.stage).rank
                bound[op.op_id, members[op.op_id]].add((rank.replica, op.stage, die))
    canonical = canonical_region_summaries(scenario, applied)
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


def _summary_is_aliased(summary: FusedRegionSummary, aliases: Mapping[tuple[int, str], tuple[int, str]]) -> bool:
    replicas = {item.replica for item in summary.semantic_work} or {0}
    return any((rep, op_id) in aliases for rep in replicas for op_id in summary.op_ids)


def build_dual_inputs_from_scenario(
    scenario: ResolvedScenario,
    *,
    region_summaries: tuple[FusedRegionSummary, ...] = (),
    primitive_table: PrimitiveTable | _UpperCapacityLike | None = None,
    rate_mode: str = "engine_ceiling",
    applied: AppliedPlan | None = None,
    verify_summaries: bool = True,
) -> tuple[ResourceChoiceProgram, EventDagSpec, tuple[str, ...]]:
    """Build replica-preserving resource-choice and event-DAG dual lower-bound inputs."""
    if applied is not None and verify_summaries:
        _check_region_summary_bindings(scenario, applied, region_summaries, require_catalog=True)
    elif applied is None and not region_summaries:
        region_summaries = getattr(scenario, "region_summaries", ())
    demands = _Demands(scenario, primitive_table, rate_mode)
    hoisted_sites = _proven_hoisted_sites(scenario, applied)
    operations, aliases = _semantic_ops(scenario, hoisted_sites=hoisted_sites)
    summaries = tuple(summary for summary in region_summaries if not _summary_is_aliased(summary, aliases))
    contractions = getattr(scenario, "bilinear_contractions", ())
    placements = dict(getattr(scenario, "die_placements", ()))
    covered = {
        (item.replica, item.pp_stage, op_id)
        for summary in summaries
        for item in summary.semantic_work
        for op_id in summary.op_ids
    }
    defaults, derived_places = derive_default_obligations(
        scenario, demands, aliases, covered, _scenario_rank_ops,
    )
    summaries += defaults
    for die, coord in derived_places.items():
        placements.setdefault(die, coord)
    seen: set[tuple[int, int, int, str]] = set()

    def add_work(item: SemanticWorkItem) -> None:
        key = item.replica, item.pp_stage, item.die, item.work_id
        if key not in seen:
            seen.add(key)
            demands.add_site(_engine_modes(demands, item))

    for summary in summaries:
        for item in summary.semantic_work:
            add_work(item)

    dies_tuple = tuple(demands.devices)
    atomic = getattr(scenario, "atomic_step", None)
    case_obj = getattr(atomic, "case", None) or getattr(scenario, "case", None)
    parallel_obj = getattr(getattr(case_obj, "runtime", None), "parallel", None)
    tp_width = (int(parallel_obj.tp) * int(parallel_obj.pcp)) if parallel_obj is not None else 1
    pp_stages = int(parallel_obj.pp) if parallel_obj is not None else 1

    quanta: list[int] = []
    event_ids: dict[tuple[int, str], int] = {}
    for replica, op, meta in operations:
        for item in meta.get("semantic_work", ()):
            add_work(item)
        for link, byte_count in meta.get("mandatory_comm_wire_bytes_by_link", ()):
            link_spec = demands.links.get(link)
            if link_spec is not None and (link_spec.semantics == "per_pair" or link_spec.semantics in _PORT_LINK_SEMANTICS):
                stage = int(getattr(op, "stage", 0))
                die_idx = (replica * pp_stages + stage) * tp_width
                default_die = dies_tuple[die_idx] if 0 <= die_idx < len(dies_tuple) else (dies_tuple[0] if dies_tuple else 0)
                src_die = _resolve_comm_endpoint_die(
                    meta, ("source_die", "src_die", "source_rank", "src_rank", "source", "die", "rank"),
                    default_die, dies_tuple,
                )
                assert src_die is not None
                demands.charge((replica, stage, src_die), f"link:{link}", byte_count)
                if link_spec.semantics in _PORT_LINK_SEMANTICS:
                    dst_die = _resolve_comm_endpoint_die(
                        meta, ("destination_die", "dst_die", "destination_rank", "dst_rank", "destination"),
                        None, dies_tuple,
                    )
                    if dst_die is not None:
                        demands.charge((replica, stage, dst_die), f"link_rx:{link}", byte_count)
                sw_pipe = _link_switch_pipe(link_spec)
                if sw_pipe is not None:
                    demands.charge((0, 0, -1), sw_pipe, byte_count)
            else:
                demands.charge((0, 0, -1), f"link:{link}", byte_count)
                if link_spec is not None:
                    sw_pipe = _link_switch_pipe(link_spec)
                    if sw_pipe is not None:
                        demands.charge((0, 0, -1), sw_pipe, byte_count)
        for scope, direction, copy_bytes in meta.get("schedule_dependent_memory", ()):
            hit = defaultdict(Fraction)
            demands.charge(scope, "l2_bw", copy_bytes, rows=hit)
            demands.charge(scope, "mte2_total" if direction == "read" else "write_total", copy_bytes, rows=hit)
            miss = defaultdict(Fraction)
            demands.transfer(scope, direction, copy_bytes, rows=miss)
            demands.add_site((hit, miss))

        quantum = meta.get("mandatory_event_quantum_ns")
        if demands.certified and quantum is not None:
            certified = meta.get("certified_mandatory_event_quantum_ns")
            quantum = min(quantum, certified) if certified is not None else None
        if quantum is not None and not meta.get("independent_prefetch"):
            event_ids[replica, op.op_id] = len(quanta)
            quanta.append(inward_floor_ns(Fraction(quantum)))

    edges: list[DagEdge] = []
    for replica, op, meta in operations:
        dst_key = (replica, op.op_id)
        if dst_key not in event_ids:
            continue
        deps = meta.get("mandatory_event_deps", ())
        if demands.certified:
            certified_deps = dict(meta.get("certified_mandatory_event_deps", ()))
            deps = ((pred, min(wire, certified_deps[pred])) for pred, wire in deps if pred in certified_deps)
        for predecessor, wire_ns in deps:
            pred_key = (
                (int(predecessor[0]), str(predecessor[1]))
                if isinstance(predecessor, tuple) and len(predecessor) == _PAIR_LEN
                else (replica, str(predecessor))
            )
            source = aliases.get(pred_key, pred_key)
            if source in event_ids and source != dst_key:
                edges.append(DagEdge(event_ids[source], event_ids[dst_key], inward_floor_ns(Fraction(wire_ns))))

    _memory_demands(demands, scenario, summaries, placements, contractions)
    program, names = demands.program()
    return program, EventDagSpec(tuple(quanta) or (0,), tuple(edges)), names
