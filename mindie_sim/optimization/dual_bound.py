"""Exact resource-choice and dependent-event lower bounds.

[Role] Assemble physical demands, solve their rational relaxation and check witnesses.
[Invariants] One normalization covers all placements. Checkers mirror Lean Int/Nat.
[Timeouts & Locks] Finite Bland pivots and DAG traversal. No locks or external solver.
[Side-Effect Policy] Read-only analysis with no runtime replay or device access.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from operator import mul
from time import perf_counter
from typing import TYPE_CHECKING

from mindie_sim.prediction.runtime.reload_bounds import EpochAccount, verify_epoch_trace
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die

from ._dual_obligations import _weight_allocation_id, derive_default_obligations
from .contracts import (
    FusedRegionSummary, SemanticWorkItem, TensorRangeObligation, certified_capacity_per_second,
)

if TYPE_CHECKING:
    from mindie_sim.rewrite.apply import AppliedPlan, ResolvedScenario
    from .contracts import CapacityTable

BINARY64_ULP_SHIFT = 52
C11_ULP_GUARD_FACTOR = 8
ENDPOINT_ROUND_GUARD_NS = 2
NS_PER_MS = 1_000_000
NS_PER_SECOND = 1000 * NS_PER_MS

__all__ = (
    "ResourceChoiceProgram", "FarkasCertificate", "EpochAccount", "DagEdge",
    "EventDagSpec", "OptimalityCertificate", "DualSubgradientReport",
    "BINARY64_ULP_SHIFT", "C11_ULP_GUARD_FACTOR", "ENDPOINT_ROUND_GUARD_NS",
    "check_farkas_dual", "eval_checked_path", "check_optimality_certificate",
    "verify_epoch_trace", "inward_floor_ns", "solve_farkas_dual",
    "solve_checked_critical_path", "certify_lower_bound", "build_dual_inputs_from_scenario",
)

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
class OptimalityCertificate:
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
        and all(modes and all(len(mode) == prog.num_resources and mu <= _sparse_dot(terms, mode)
                              for mode in modes)
                for mu, modes in zip(cert.mu_num, prog.site_modes))
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
    prog: ResourceChoiceProgram, dag: EventDagSpec, cert: OptimalityCertificate,
) -> bool:
    if not _is_valid_acyclic_dag(dag):
        return False
    path = eval_checked_path(dag, cert.path_edges, cert.sink_node_id)
    return (path is not None and check_farkas_dual(prog, cert.farkas)
            and cert.l_star_lean_ns >= 0
            and cert.l_star_lean_ns == max(cert.farkas.lower_bound_ns, path))

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
    """Solve the positive, normalized restricted master by exact column pricing.

    Each row stores a row of the basis inverse times a common denominator.
    Fraction-free pivots use integer minors. This gives the same ratios and
    reduced costs as Fraction arithmetic, without creating a Fraction per cell.
    """
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
                entering = next((j for j in range(size, len(columns))
                                 if _dot(x, columns[j]) < denom), None)
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
        # Bland's leaving rule, with ratios compared by exact cross products.
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
        assert leaving is not None
        pivot = direction[leaving]
        pivot_row = inverse[leaving]
        pivot_rhs = rhs[leaving]
        for i, factor in enumerate(direction):
            if i != leaving:
                inverse[i] = [(a * pivot - b * factor) // denom
                              for a, b in zip(inverse[i], pivot_row)]
                rhs[i] = (rhs[i] * pivot - pivot_rhs * factor) // denom
        basis[leaving] = entering
        denom = pivot

def _independent_masters(base, sites):
    """Solve disconnected resource components with independent site mixtures."""
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
        tight = tuple(name for name, values in zip(names, zip(prog.global_work, *zero_modes))
                      if sum(values) == 0)
    else:
        base = list(prog.global_work)
        sites = []
        for modes in prog.site_modes:
            # Move the coordinate-wise minimum into fixed work so pricing deals
            # only with nonnegative residuals, even for signed Lean programs.
            minimum = tuple(map(min, zip(*modes)))
            base = [a + b for a, b in zip(base, minimum)]
            residuals = tuple(dict.fromkeys(tuple(a - b for a, b in zip(mode, minimum))
                                           for mode in modes))
            if all(any(mode) for mode in residuals):
                sites.append(residuals)

        if not sites:
            optimum = max(0, *base)
            tied = [i for i, value in enumerate(base) if value == optimum]
            weights = tuple(Fraction(1, len(tied)) if i in tied and optimum else Fraction(0)
                            for i in range(size))
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
    assert check_farkas_dual(prog, cert)
    return cert, DualSubgradientReport(names, weights, tight, objective, (perf_counter() - start) * 1000)

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
    assert eval_checked_path(dag, path, sink) == distance[sink]
    return path, sink, distance[sink]

def certify_lower_bound(
    prog: ResourceChoiceProgram, dag: EventDagSpec,
) -> tuple[OptimalityCertificate, DualSubgradientReport]:
    farkas, report = solve_farkas_dual(prog)
    path, sink, path_ns = solve_checked_critical_path(dag)
    cert = OptimalityCertificate(farkas, path, sink, max(0, farkas.lower_bound_ns, path_ns))
    assert check_optimality_certificate(prog, dag, cert)
    return cert, report

def _scenario_rank_ops(scenario):
    entries = getattr(scenario, "atomic_ops", None)
    if entries is None:
        graph = getattr(scenario, "atomic_graph", ())
        entries = getattr(graph, "op_specs", getattr(graph, "ops", graph))
    for entry in entries.values() if isinstance(entries, Mapping) else (entries or ()):
        for op in getattr(entry, "ops", (entry,)):
            if hasattr(op, "metadata"):
                yield int(getattr(entry, "replica", 0)), op

def _op_contract_key(op, metadata):
    if (metadata.get("pure") is not True or metadata.get("load_time")
        or metadata.get("state_effect") or op.link_bytes):
        return None
    ports = metadata.get("input_port_versions")
    contract = metadata.get("semantic_contract", metadata.get("contract"))
    if ports is None or contract is None:
        return None
    return op.family, op.stage, op.layer, op.requests, op.local_shape, op.dtype, op.layout, repr(ports), repr(contract)

def _atomic_ops(scenario):
    del scenario
    raise NotImplementedError("LEGACY_PURGED: wire replica-preserving dual_bound keying by (replica, op_id)")

def _semantic_ops(scenario):
    del scenario
    raise NotImplementedError("LEGACY_PURGED: wire replica-preserving dual_bound keying by (replica, op_id)")

def _machine(scenario):
    machine = getattr(scenario, "machine", None)
    if machine is not None:
        return machine
    atomic = getattr(scenario, "atomic_step", None)
    case = getattr(atomic, "case", None) or getattr(scenario, "case", None)
    return getattr(case, "machine", None)

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
            from mindie_sim.primitives.loader import validate_table_sku_compatibility
            for device in self.devices.values():
                if getattr(device, "sku", ""):
                    validate_table_sku_compatibility(self.table, device.sku)
        self.fixed = defaultdict(Fraction)
        self.sites = []

    def rate(self, scope, pipe, dtype=""):
        engine = self.engine_rate(scope, pipe, dtype)
        # A silicon constraint can only relax service already priced by simulation.
        if not self.certified or engine <= 0:
            return engine
        device = self.devices.get(scope[-1])
        if self.table is None:
            return 0
        if device and getattr(device, "sku", ""):
            from mindie_sim.primitives.loader import validate_table_sku_compatibility
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
                else:
                    egress_bpms = max((base_bpms, *ep_rates))
                if link.switch_capacity_bytes_per_ms is not None:
                    egress_bpms = min(egress_bpms, Fraction(link.switch_capacity_bytes_per_ms))
                total += 1000 * egress_bpms
            return total
        if pipe.startswith("link:"):
            link = self.links.get(pipe.removeprefix("link:"))
            return 1000 * Fraction(link.effective_peak_capacity_bytes_per_ms) if link else 0
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
            self.charge(scope, other, Fraction(path.hbm_rw_alpha) * size,
                        rate_pipe=own, rows=rows)
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
        sites = tuple(tuple(tuple(inward_floor_ns(mode.get(resource, Fraction()))
                                  for resource in resources) for mode in modes)
                      for modes in self.sites)
        return ResourceChoiceProgram(len(resources), fixed, sites), names

def _engine_modes(demands, item):
    scope = item.replica, item.pp_stage, item.die
    modes = []
    for engine in dict.fromkeys(item.admitted_engines):
        mode = defaultdict(Fraction)
        pipe = {"cube": "cube", "aiv_simd": "vector", "aiv_simt": "simt"}[engine]
        demands.charge(scope, pipe, item.logical_ops, dtype=item.dtype if pipe == "cube" else "",
                       rows=mode)
        modes.append(mode)
    if item.inline_epilogue_absorbable:
        modes.append({})
    path = demands.paths.get(item.die)
    if item.kind == "sort_topk" and path and path.simt_ub_carveout_bytes:
        raise NotImplementedError(
            "LEGACY_PURGED: wire admitted-scheduler-only sort_topk dual modes (never synthesize empty 0 ns radix)"
        )
    return modes

def _memory_demands(demands, scenario, summaries, placements, contractions):
    from mindie_sim.prediction.runtime.reload_bounds import (
        SECTOR_BYTES, build_surplus_coupling_sites, compute_compulsory_hbm_cut,
        direct_mac_pebble_reload_bound,
    )

    by_die = defaultdict(list)
    for spec in contractions:
        by_die[spec.die].append(spec)

    sectors = getattr(scenario, "sector_bytes", SECTOR_BYTES)
    cuts = compute_compulsory_hbm_cut(summaries, sector_bytes=sectors)
    for die, cut in cuts.items():
        specs = by_die.get(die, ())
        default = (specs[0].replica, specs[0].pp_stage) if specs else (0, 0)
        scope = (*placements.get(die, default), die)
        # Sector traffic can exceed issued copy traffic on an unaligned run.
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

def _check_region_summary_bindings(scenario, applied, summaries, *, require_catalog=False):
    if not summaries:
        return
    del scenario, applied, require_catalog
    raise NotImplementedError("LEGACY_PURGED: wire rewrite.catalog_replay canonical region summaries")


def build_dual_inputs_from_scenario(
    scenario: ResolvedScenario, *, region_summaries: tuple[FusedRegionSummary, ...] = (),
    primitive_table: CapacityTable | None = None, rate_mode: str = "engine_ceiling",
    applied: AppliedPlan | None = None,
) -> tuple[ResourceChoiceProgram, EventDagSpec, tuple[str, ...]]:
    """Use only universal work obligations and admitted alternatives for the scenario."""
    del scenario, region_summaries, primitive_table, rate_mode, applied
    raise NotImplementedError("LEGACY_PURGED: wire replica-preserving dual_bound")
