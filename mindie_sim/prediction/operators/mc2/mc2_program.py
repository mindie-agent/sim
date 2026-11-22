"""Canonical MC2 construction from the retained phase contract.

[Role]
  Build launch-local MC2 phase programs, peer wire work and directed transport
  bindings from the retained ``mc2_phase`` contract.  This module owns
  construction only; payload state and Engine scheduling live beside it.

[Invariants]
  * Phase ownership, peer classes and link work come from the declared source
    contract; no isolated duration is stored or fabricated.
  * Local HCCS and remote RoCE peers are separate physical classes.
  * A program returned for an ordinary graph is not started here.
  * Work is conserved when destination HBM is moved to another owner.

[Timeouts & Locks]
  Bounded phase counts; one cached program per declared contract.  No I/O,
  device context, lock or background task.

[Side-Effect Policy]
  Pure construction; creates virtual Engine tasks only when a caller starts
  the returned program on its own Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math
from types import SimpleNamespace
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import KernelProgram

__all__ = [
    "Mc2LaunchView", "native_program_from_contract",
]

_EPS = 1e-12
AIV_PER_AIC = 2
MIN_MC2_AIV_UNITS = 2
NATIVE_PROGRAM_CACHE_MAXSIZE = 128
US_PER_MS = 1000


def _resolve_aiv_units(units_map: Mapping[str, int], *, core_count: int | None = None) -> int:
    """Vector AIV count: declared vector units, else AIC count * AIV_PER_AIC."""
    vector = int(units_map.get("vector", 0) or 0)
    if vector > 0:
        return max(MIN_MC2_AIV_UNITS, vector)
    aic = int(core_count or units_map.get("cube", 0) or 0)
    if aic > 0:
        return max(MIN_MC2_AIV_UNITS, aic * AIV_PER_AIC)
    raise ModelingError("mc2_vector_units_undeclared")


def _as_fraction(value: Any) -> Fraction:
    """Exact scalar conversion: ints and Fractions are exact, floats decimal."""
    t = type(value)
    if t is Fraction:
        return value
    if t is int:
        return Fraction(value)
    if t is bool or isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelingError(f"expression_non_finite:{value!r}")
        return Fraction(str(value))
    raise ModelingError(f"expression_non_numeric:{value!r}")


def _link_work(plan, ep: int, node_ranks: int = 0):
    """Add source-directed peer wire on the real intra/inter link classes.

    A fullmesh source sends to every peer.  When the EP group spans nodes, the
    local (HCCS) and remote (RoCE) peer counts are different physical links;
    each peer's equal share is charged to its actual class.  A single-node or
    undeclared topology keeps one intra-node link class.
    """
    phases = []
    peers = max(0, int(ep) - 1)
    local_peers = (max(0, min(int(ep), int(node_ranks)) - 1)
                   if node_ranks and int(node_ranks) > 1 else peers)
    remote_peers = peers - local_peers
    for phase in plan.phases:
        work = dict(phase.work)
        if phase.name in {"payload_exchange", "count_status_exchange"} and peers:
            # The source plan owns send/recv HBM and MTE3.  The peer wire is
            # the same record outside this rank; the diagonal stays local.
            remote = _as_fraction(work.get("mte3", 0)) * Fraction(peers, ep)
            if remote_peers:
                work["link_hccs"] = (work.get("link_hccs", Fraction(0))
                                     + remote * Fraction(local_peers, peers))
                work["link_roce"] = (work.get("link_roce", Fraction(0))
                                     + remote * Fraction(remote_peers, peers))
            else:
                work["link_hccs"] = (work.get("link_hccs", Fraction(0))
                                     + remote)
        phases.append(replace(phase, work=tuple(work.items())))
    return tuple(phases)

def _peer_class_for(rank, index, peer, *, node_ranks, peer_count, classes):
    """Classify one declared EP edge for an arbitrary launch rank.

    The representative graph's own view is not reused for another replica:
    physical rank nodes decide local HCCS versus remote RoCE.
    """
    declared = dict(classes or {})
    if peer in declared:
        return declared[peer]
    if rank >= 0 and node_ranks > 0:
        return ("hccs" if peer // node_ranks == rank // node_ranks
                else "roce")
    local = (min(max(1, int(peer_count)), node_ranks) - 1
             if node_ranks > 0 else max(0, int(peer_count) - 1))
    return "hccs" if index < local else "roce"

@dataclass(frozen=True)
class Mc2LaunchView:
    """Immutable launch-local rank/topology/fabric view."""
    rank: int
    peer_labels: tuple[int, ...]
    peer_classes: tuple[tuple[int, str], ...] = ()
    peer_weights: tuple[Fraction, ...] = ()
    node_ranks: int = 0
    intra_latency_ms: Fraction = Fraction(0)
    inter_latency_ms: Fraction = Fraction(0)
    resource_names: tuple[tuple[str, str], ...] = ()
    payload_frames: tuple[tuple[int, int], ...] = ()
    payload_packet_bytes: int = 0
    payload_basis: str = "unknown"
    payload_local_frames: int = 0

    def class_for(self, index: int, peer: int) -> str:
        return _peer_class_for(
            self.rank, index, peer, node_ranks=self.node_ranks,
            peer_count=len(self.peer_labels) + 1,
            classes=self.peer_classes)

    def latency_for(self, klass: str) -> Fraction:
        return self.inter_latency_ms if klass == "roce" else self.intra_latency_ms



def _subtract_endpoint_hbm(program, resource, amount):
    """Remove moved destination bytes while preserving owner-local balance.

    Returns ``(program, moved)``. The reduction is apportioned over every
    stage that owns the same physical HBM resource instead of depleting the
    first owner. If the phase has less HBM work than the represented inbound
    cells, keep the original aggregate and report no move.
    """
    if amount <= 0:
        return program, 0.0
    hbm_stages = [
        (stage, float(dict(stage.work).get(resource, 0.0)))
        for stage in program.stages
        if float(dict(stage.work).get(resource, 0.0)) > 0]
    total = sum(value for _, value in hbm_stages)
    if total <= 0 or amount > total + _EPS:
        return program, 0.0
    remaining = float(amount)
    last_hbm = max(index for index, stage in enumerate(program.stages)
                   if float(dict(stage.work).get(resource, 0.0)) > 0)
    stages = []
    for index, stage in enumerate(program.stages):
        work = dict(stage.work)
        present = float(work.get(resource, 0.0))
        if present > 0:
            if index == last_hbm:
                take = remaining
            else:
                take = amount * present / total
            take = min(present, take)
            left = present - take
            if left > _EPS:
                work[resource] = left
            else:
                work.pop(resource, None)
            remaining -= take
            stage = replace(stage, work=work)
        stages.append(stage)
    return KernelProgram(tuple(stages)), float(amount)


def _mapped_work(work: dict[str, Any], rates: dict[str, float]) -> dict[str, Fraction]:
    names = set(rates)
    mapped: dict[str, Fraction] = {}
    for pipe, amount in work.items():
        exact = _as_fraction(amount)
        if exact <= 0:
            continue
        key = str(pipe)
        if key not in names and key in {"mte2_aiv", "mte2_aic"} and "mte2" in names:
            key = "mte2"
        mapped[key] = mapped.get(key, Fraction(0)) + exact
    return mapped

@lru_cache(maxsize=NATIVE_PROGRAM_CACHE_MAXSIZE)
def _native_program(raw_items, aiv, rates_items, units_items, service_items,
                    latencies, node_ranks, peer_weights, rank, peer_classes,
                    resource_names, payload_frames, payload_basis,
                    payload_local_frames, exact_payload):
    """Build the real native phase program without running a private Engine.

    ``raw_items`` is the complete phase contract: source and received
    assignments are already the declared active row ledger, not re-derived
    from a capacity token count.  The shared Execution launches the returned
    program on its own Engine through ``launch_native_cohort``.
    """
    from .mc2_phase import build, contract
    from .mc2_program_launch import _Mc2Program
    raw = dict(raw_items)
    plan = contract(raw, aiv)
    phases = _link_work(plan, max(1, int(raw["ep"])), int(node_ranks))
    base_roles = {"vector", "mte2_aiv", "mte3", "hbm", "link"}
    all_roles = (set(dict(rates_items)) | set(dict(units_items)) | base_roles
                 | {str(name) for name, _mode in service_items})
    geometry = Geometry(
        compute_units=max(MIN_MC2_AIV_UNITS, aiv),
        resource_units={role: max(MIN_MC2_AIV_UNITS, int(dict(units_items).get(role, aiv)))
                        for role in all_roles})
    plans = {}
    for phase in phases:
        plans[phase.name] = build(
            SimpleNamespace(metadata={
                "mc2_phase": {**raw, "phase": phase.name}}),
            geometry)
    # Only real nonzero per-peer phases allocate directed wire futures.  A
    # phase with no link bytes is local-only; PCOM must not create an
    # unresolvable (src,dst,phase) promise for it.
    link_by_phase = {}
    for phase in phases:
        links = {
            pipe: _as_fraction(amount) for pipe, amount in phase.work
            if pipe in {"link", "link_hccs", "link_roce"} and amount}
        if links:
            link_by_phase[phase.name] = links
    if exact_payload:
        # The public payload ledger is the same concrete route frame count
        # times the physical packet width used by the launch sends below.
        packet = int(plan.layout.packet_bytes)
        totals = {}
        for index, (peer, frames) in enumerate(payload_frames):
            klass = _peer_class_for(
                int(rank), index, int(peer), node_ranks=int(node_ranks),
                peer_count=max(1, int(raw["ep"])), classes=peer_classes)
            key = "link_hccs" if klass == "hccs" else "link_roce"
            totals[key] = (totals.get(key, Fraction(0))
                           + Fraction(int(frames) * packet))
        # Keep the named phase present even for a zero-frame representative
        # rank: another replica still owns directed frames in this cohort.
        link_by_phase["payload_exchange"] = totals or {"link": Fraction(0)}
    elif not payload_frames and "unknown" in str(payload_basis).lower():
        # No scoped source route: no invented directed payload wire.
        link_by_phase.pop("payload_exchange", None)
    rates = dict(rates_items)
    units = dict(units_items)
    for name in rates:
        units.setdefault(name, max(MIN_MC2_AIV_UNITS, int(aiv)))
    names = {name: name for name in rates}
    for alias in ("mte2_aiv", "mte2_aic"):
        if alias not in names and "mte2" in names:
            names[alias] = "mte2"
        if alias not in units and "mte2" in units:
            units[alias] = units["mte2"]
    # A synthetic/single-link supply keeps the same route work on its one
    # declared link.  Do not delete off-node bytes; map them to the available
    # physical link class.
    link_fallback = ("link" if "link" in rates else
                     "link_hccs" if "link_hccs" in rates else
                     "link_roce" if "link_roce" in rates else "")
    if not link_fallback:
        # A stage resource table may expose only topology-scoped physical
        # ports (for example per-pair link resources).  Keep one real declared
        # link for an isolated representative launch; a cohort launch rebinds
        # each rank/class to its own port through native_resource_names.
        physical_links = sorted(
            (name for name in rates if "link" in str(name).lower()),
            key=lambda name: (":rx:" in str(name).lower(), str(name)))
        link_fallback = physical_links[0] if physical_links else ""
    for alias in ("link_hccs", "link_roce"):
        if alias not in rates and link_fallback:
            names[alias] = link_fallback
            units.setdefault(alias, units.get(link_fallback, max(MIN_MC2_AIV_UNITS, int(aiv))))
    phase_work: dict[str, Fraction] = {}
    for phase in phases:
        for pipe, amount in plans[phase.name].work.resources.items():
            for key, value in _mapped_work({pipe: amount}, rates).items():
                phase_work[key] = phase_work.get(key, Fraction(0)) + value
    work = dict(phase_work)
    for links in link_by_phase.values():
        for pipe, amount in links.items():
            if amount:
                key = names.get(pipe, pipe)
                work[key] = work.get(key, Fraction(0)) + amount
    program = _Mc2Program(
        phases=phases, plans=plans, link_by_phase=link_by_phase,
        latency=latencies[0], rates=rates, units=units, names=names,
        service_modes=dict(service_items), latencies=latencies,
        peer_count=max(1, int(raw["ep"])), rank=int(rank),
        peer_weights=peer_weights, node_ranks=int(node_ranks),
        peer_classes=peer_classes, resource_names=resource_names,
        payload_frames=payload_frames,
        payload_packet_bytes=int(plan.layout.packet_bytes),
        payload_basis=payload_basis,
        payload_local_frames=int(payload_local_frames),
        route_matrix=raw.get("route_matrix"),
        route_reverse=bool(raw.get("route_reverse", False)),
        exact_payload=bool(exact_payload),
        route_matrix_bytes=raw.get("route_matrix_bytes"),
        uniform_payload_bytes=raw.get("uniform_payload_bytes", 0),
        experts=raw.get("experts", 0),
        transport_basis=raw.get("payload_basis", payload_basis))
    program._phase_work = phase_work
    program.work = tuple(sorted(work.items()))
    return program, program.work

def _transport_identity(context):
    identity = getattr(context, "service_identity", None) or {}
    if isinstance(identity, Mapping):
        transport = identity.get("transport")
        if isinstance(transport, Mapping):
            return dict(transport)
    return {}


def _transport_latencies(context, ep: int) -> tuple[Fraction, Fraction]:
    """Declared intra/inter propagation, including the ordinary graph path.

    The piecewise supply exposes explicit invocation timings; a RecipeContext
    exposes the same physical links through its service identity.  Neither is
    a fitted or inferred residual.
    """
    if ep <= 1:
        return Fraction(0), Fraction(0)
    transport = _transport_identity(context)
    links = tuple(transport.get("links") or ())
    def pick(terms):
        values = []
        for link in links:
            if not isinstance(link, Mapping):
                continue
            name = str(link.get("name", "")).lower()
            if not any(term in name for term in terms):
                continue
            latency = link.get("propagation_us", link.get("latency_us", 0))
            values.append(_as_fraction(latency) / US_PER_MS)
        return max(values) if values else None
    intra = pick(("hccs", "intra", "local", "same_node", "same-node"))
    inter = pick(("roce", "inter", "remote", "superpod"))
    if intra is None:
        intra = Fraction(0)
    if inter is None:
        inter = intra
    return intra, inter


def _transport_peer_classes(context, rank, ep):
    transport = _transport_identity(context)
    placements = transport.get("placements") or ()
    nodes = {}
    for placement in placements:
        if not isinstance(placement, Mapping):
            continue
        try:
            nodes[int(placement["rank"])] = int(placement.get("node", 0))
        except (KeyError, TypeError, ValueError):
            continue
    if rank not in nodes:
        return None
    return tuple(sorted(
        (peer, "hccs" if nodes.get(peer) == nodes[rank] else "roce")
        for peer in range(ep) if peer != rank and peer in nodes))


def native_program_from_contract(raw, context):
    """Reuse the real MC2 program construction for an ordinary public graph.

    Returns without starting an Engine or running a private estimate; the
    common Execution owns the launch.  Physical resources, rank placement and
    link classes come from the actual RecipeContext.
    """
    data = dict(raw)
    ep = max(1, int(data["ep"]))
    rank = int(data.get("route_rank", 0))
    matrix = data.get("route_matrix")
    exact = matrix is not None
    frames = ()
    local_frames = 0
    if exact:
        cells = tuple(tuple(int(value) for value in row) for row in matrix)
        reverse = bool(data.get("route_reverse", False))
        if 0 <= rank < len(cells):
            frames = tuple(sorted(
                (peer, cells[peer][rank] if reverse else cells[rank][peer])
                for peer in range(len(cells))
                if peer != rank
                and (cells[peer][rank] if reverse else cells[rank][peer]) > 0))
            local_frames = cells[rank][rank]
    basis = str(data.get("payload_basis") or "unknown")
    if exact and not frames:
        basis = "actual_zero"
    node_ranks = int(_transport_identity(context).get("ranks_per_node", 0) or 0)
    latencies = _transport_latencies(context, ep)
    resources = tuple(sorted(
        (str(name), str(value))
        for name, value in dict(getattr(context, "resource_names", {}) or {}).items()))
    rates = tuple(sorted(
        (str(name), float(rate))
        for name, rate in dict(getattr(context, "rates", {}) or {}).items()))
    units = tuple(sorted(
        (str(name), int(value))
        for name, value in dict(getattr(context, "resource_units", {}) or {}).items()))
    services = tuple(sorted(
        (str(name), (str(role), float(rate)))
        for name, (role, rate) in dict(
            getattr(context, "service_modes", {}) or {}).items()))
    aiv = _resolve_aiv_units(
        dict(units), core_count=int(getattr(context, "core_count", 0) or 0))
    program, _work = _native_program(
        tuple(sorted(data.items())), aiv, rates, units, services, latencies,
        node_ranks, None, rank, _transport_peer_classes(context, rank, ep),
        resources, frames, basis, local_frames, exact)
    return program
