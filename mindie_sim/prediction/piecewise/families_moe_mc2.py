"""Physical piecewise binding of the source fullmesh MC2 dispatch/return.

[Role]
  Bind the existing ``mc2_phase.contract`` phase ledger and
  ``operators.mc2_fullmesh`` row layouts to the ordinary piecewise family
  runtime.  Each native V3 dispatch/combine invocation remains one timing
  owner; its source phases are exposed as stage work and stage statistics.

[Invariants]
  * Row counts are this rank's pre-expansion source tokens and the expected
    received assignments (source tokens * top_k / ep), never a fabricated
    per-expert route matrix.
  * The source phase DAG is preserved.  Parallel count/status and payload
    branches are not flattened into a serial sum; the critical path is taken
    over the declared dependencies.
  * Frame width, quantized payload, E8M0 scale, AIV count, expert/status
    records, local copy and weighted return all come from the existing
    contracts.  A selected native launch is one owner; an unselected one emits
    no work.
  * Generic AllToAll stays a separate retained family and is never renamed to
    this contract.

[Timeouts & Locks]
  Pure bounded arithmetic over at most a few dozen phases; no I/O, device work,
  locks or background tasks.

[Side-Effect Policy]
  Returns immutable family templates and diagnostic expressions only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from typing import Any, Mapping

from types import SimpleNamespace

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mc2_phase import build, contract
from mindie_sim.prediction.program import KernelProgram, RecipeContext
from mindie_sim.prediction.moe_comm_selection import (
    MC2, MoeCommConfig, local_tokens, selection_bindings,
)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import BoundOperator
from .compiler import (
    CompiledOperator, FamilyTemplate, NativeProgram, SegmentAxis, _demand,
)
from .expressions import Const, Var, as_fraction
from .guards import StepGuard

_NATIVE = {
    "dispatch": ("mc2_dispatch", "MoeDistributeDispatchV3"),
    "combine": ("mc2_combine", "MoeDistributeCombineV3"),
}


def _config(operator: BoundOperator) -> MoeCommConfig | None:
    return MoeCommConfig.from_items(operator.theta.find("moe_comm", None))


def _raw(operator: BoundOperator) -> dict[str, Any]:
    raw = operator.theta.find("mc2_phase", None)
    if raw is None:
        raise ModelingError("mc2_transport_requires_phase_contract")
    return dict(raw)


def _aiv_count(ctx: CompiledOperator) -> int:
    units = dict(getattr(ctx.supply, "resource_units", ()) or ())
    value = units.get("vector") or getattr(ctx.policy, "cores", 1) or 1
    return max(2, int(value))


def _rank_index(ctx: CompiledOperator) -> int:
    meta = dict(ctx.feature_metadata or ())
    for key in ("moe_rank_index", "rank_index", "selected_rank"):
        value = meta.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lstrip("+-").isdigit():
                return int(value)
            suffix = value.rsplit("rank", 1)[-1]
            if suffix.isdigit():
                return int(suffix)
    return -1


def _received_rows(ctx: CompiledOperator, geometry: Any, state: Any,
                   raw: dict[str, Any]) -> tuple[Fraction, bool]:
    """Actual local received assignments, or an explicit expected fallback.

    A source group-list/routing ledger gives the local expert shard; the
    uniform row-grid estimate is used only when no local ledger is declared.
    The caller exposes that basis rather than silently claiming equality.
    """
    tokens = max(0, int(local_tokens(geometry)))
    top_k = max(1, int(raw["top_k"]))
    expected = Fraction(tokens * top_k)
    experts = max(1, int(raw["experts"]))
    ep = max(1, int(raw["ep"]))
    local = max(1, experts // ep)
    rank = _rank_index(ctx)
    meta = dict(ctx.feature_metadata or ())
    declared = meta.get("moe_received_rows")
    layer = int(getattr(ctx, "layer", 0) or 0)
    value = None
    if isinstance(declared, Mapping):
        value = declared.get(layer, declared.get(str(layer)))
    elif (isinstance(declared, (tuple, list)) and len(declared) == 1
          and isinstance(declared[0], int)):
        value = declared[0]
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return Fraction(value), True
    counts = None
    try:
        counts = geometry.routing_for(layer)
    except (AttributeError, ModelingError):
        counts = None
    if counts is None and state is not None:
        try:
            counts = state.routing_state(layer).counts
        except (AttributeError, ModelingError):
            counts = None
    if counts is not None and rank >= 0 and len(tuple(counts)) == experts:
        counts = tuple(int(item) for item in counts)
        return Fraction(sum(counts[rank * local:(rank + 1) * local])), True
    return expected, False


def _payload_frames(ctx, geometry, state, raw, source_assignments):
    """Scoped per-destination whole frames; unknown without a real origin."""
    kind = str(raw.get("kind") or "")
    experts = max(1, int(raw["experts"]))
    source = as_fraction(source_assignments)
    if source.denominator != 1:
        return (), "unknown", 0
    meta = dict(ctx.feature_metadata or ())
    rank = _rank_index(ctx)
    if rank < 0:
        return (), "unknown", 0
    if kind == "combine":
        # Combine direction is expert -> original token rank.  Only a retained
        # reverse mapping / expand_idx can prove it; the forward owner map is
        # not a combine destination.
        reverse = (meta.get("moe_source_ranks")
                   or meta.get("moe_reverse_mapping")
                   or meta.get("moe_reverse_source_ranks"))
        if reverse is None:
            return (), "unknown", 0
        try:
            destinations = tuple(int(item) for item in reverse)
        except (TypeError, ValueError):
            return (), "unknown", 0
        if len(destinations) != int(source):
            return (), "unknown", 0
        frames = {}
        local = 0
        for destination in destinations:
            if destination == rank:
                local += 1
            else:
                frames[destination] = frames.get(destination, 0) + 1
        return (tuple(sorted(frames.items())),
                "actual_source_reverse" if frames else "actual_zero", local)
    if kind != "dispatch":
        return (), "unknown", 0
    if source == 0:
        return (), "actual_zero", 0
    owner = meta.get("moe_expert_rank")
    if owner is None:
        owner = meta.get("moe_expert_owner")
    if owner is None:
        return (), "unknown", 0
    if isinstance(owner, Mapping):
        try:
            ownership = tuple(int(owner[expert]) for expert in range(experts))
        except (KeyError, TypeError, ValueError):
            return (), "unknown", 0
    elif isinstance(owner, (tuple, list)) and len(owner) == experts:
        try:
            ownership = tuple(int(item) for item in owner)
        except (TypeError, ValueError):
            return (), "unknown", 0
    else:
        return (), "unknown", 0
    if state is None:
        return (), "unknown", 0
    try:
        routing = state.routing_state(int(ctx.layer))
        counts = tuple(int(item) for item in routing.counts)
        valid_tokens = int(routing.valid_tokens)
    except (AttributeError, ModelingError, TypeError, ValueError):
        return (), "unknown", 0
    if valid_tokens != int(local_tokens(geometry)):
        # Capacity/global valid rows are not this source rank's ownership.
        return (), "unknown", 0
    if len(counts) != experts or sum(counts) != int(source):
        return (), "unknown", 0
    frames = {}
    local = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        destination = ownership[expert]
        if destination == rank:
            local += count
        else:
            frames[destination] = frames.get(destination, 0) + count
    return (tuple(sorted(frames.items())),
            "actual_source_local" if frames else "actual_zero", local)


@lru_cache(maxsize=128)
def _unpriced_counts(raw_items, tokens, aiv, received, rates_items, units_items):
    """Known source scalar/barrier counts with no declared service rate."""
    raw = dict(raw_items)
    top_k = max(1, int(raw["top_k"]))
    phase_raw = {**raw, "source_tokens": max(0, int(tokens)), "top_k": top_k}
    if raw["kind"] == "dispatch":
        phase_raw["source_assignments"] = Fraction(max(0, int(tokens)) * top_k)
        phase_raw["received_assignments"] = as_fraction(received)
    else:
        phase_raw["source_assignments"] = as_fraction(received)
        phase_raw["received_assignments"] = Fraction(max(0, int(tokens)) * top_k)
    rates = dict(rates_items)
    units = dict(units_items)
    for name in rates:
        units.setdefault(name, max(2, int(aiv)))
    roles = set(rates) | {"vector", "mte2_aiv", "mte3", "hbm"}
    geometry = Geometry(
        compute_units=max(2, aiv),
        resource_units={role: max(2, int(units.get(role, aiv)))
                        for role in roles})
    total = {}
    plan = contract(phase_raw, aiv)
    for phase in plan.phases:
        native = build(SimpleNamespace(metadata={
            "mc2_phase": {**phase_raw, "phase": phase.name}}), geometry)
        services = dict(native.metadata.get("primitive_services") or {})
        for owner_items in (services.get("owners") or {}).values():
            for item in (owner_items or {}).values():
                for role, amount in (item.get("unpriced_service_work") or {}).items():
                    total[role] = total.get(role, Fraction(0)) + as_fraction(amount)
    return tuple(sorted(total.items()))


def _counts(raw: dict[str, Any], geometry: Any,
            received: Fraction | None = None) -> tuple[int, int, int, Fraction, Fraction]:
    tokens = max(0, int(local_tokens(geometry)))
    top_k = max(1, int(raw["top_k"]))
    experts = max(1, int(raw["experts"]))
    # Every source keeps its own assignment count; the fullmesh contract
    # conserves total assignments, so the expected incoming rows are the same
    # count.  The local expert share distributes each source's rows across
    # peers, it does not delete the other peers' incoming cells.
    source_assignments = tokens * top_k
    actual = Fraction(source_assignments) if received is None else as_fraction(received)
    return tokens, top_k, experts, Fraction(source_assignments), actual


def _phase_raw(raw: dict[str, Any], geometry: Any,
               received: Fraction | None = None) -> dict[str, Any]:
    tokens, top_k, _experts, source_assignments, received = _counts(
        raw, geometry, received)
    phase_raw = {**raw, "source_tokens": tokens, "top_k": top_k}
    if raw["kind"] == "dispatch":
        phase_raw["source_assignments"] = source_assignments
        phase_raw["received_assignments"] = received
    elif raw["kind"] == "combine":
        phase_raw["source_assignments"] = received
        phase_raw["received_assignments"] = source_assignments
    else:
        raise ModelingError("mc2_transport_kind_invalid")
    return phase_raw


def _plan(raw: dict[str, Any], geometry: Any, aiv: int,
          received: Fraction | None = None):
    return contract(_phase_raw(raw, geometry, received), aiv)


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
            remote = as_fraction(work.get("mte3", 0)) * Fraction(peers, ep)
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


def _placeholder(raw: dict[str, Any], aiv: int, node_ranks: int = 0):
    kind = raw["kind"]
    dummy = {**raw, "source_tokens": 1, "top_k": 1}
    if kind == "dispatch":
        dummy.update(source_assignments=1, received_assignments=1)
    elif kind == "combine":
        dummy.update(source_assignments=1, received_assignments=1)
    return _link_work(contract(dummy, aiv), max(1, int(raw["ep"])),
                      int(node_ranks))


def _map_pipe(ctx: CompiledOperator, pipe: str) -> str:
    """Keep the source role when declared; bind to the shared supplied pipe.

    ``mc2_phase`` names the AIV MTE2 role directly.  A synthetic or older
    supply may expose only the shared ``mte2`` copy class; the work is the same
    copy work and the selected physical rate is the one that exists.
    """
    names = {name for name, _ in getattr(ctx.supply, "rates", ())}
    if pipe in names:
        return pipe
    if pipe in {"mte2_aiv", "mte2_aic"} and "mte2" in names:
        return "mte2"
    if pipe == "link" and "link_hccs" in names:
        return "link_hccs"
    if pipe == "link" and "link_roce" in names:
        return "link_roce"
    if pipe in {"link_hccs", "link_roce"}:
        if "link" in names:
            return "link"
        other = "link_roce" if pipe == "link_hccs" else "link_hccs"
        if other in names:
            return other
    raise ModelingError(f"mc2_phase_pipe_missing:{pipe}")



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
    if total <= 0 or amount > total + 1e-12:
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
            if left > 1e-12:
                work[resource] = left
            else:
                work.pop(resource, None)
            remaining -= take
            stage = replace(stage, work=work)
        stages.append(stage)
    return KernelProgram(tuple(stages)), float(amount)


class _Mc2Program:
    """Execute the selected native phase programs on the common Engine.

    Each source phase keeps its own KernelProgram, active owner placement and
    primitive/readiness dependencies.  Link bytes and peer propagation are
    separate declared services after the phase they belong to.  The chip is
    not assigned wholesale to every phase: an owner occupies one declared
    resource unit.
    """

    def __init__(self, phases, plans, link_by_phase, latency, rates, units,
                 names, service_modes, peer_count=1, rank=-1,
                 peer_weights=None, node_ranks=0, peer_classes=None,
                 latencies=None, resource_names=None, payload_frames=None,
                 payload_packet_bytes=0, payload_basis="unknown",
                 payload_local_frames=0, route_matrix=None,
                 route_reverse=False, exact_payload=False,
                 route_matrix_bytes=None, uniform_payload_bytes=0,
                 experts=0, transport_basis=None):
        self.phases = tuple(phases)
        self.plans = dict(plans)
        self.link_by_phase = dict(link_by_phase)
        self.rates = dict(rates)
        self.units = dict(units)
        self.names = dict(names)
        self.service_modes = dict(service_modes)
        self.peer_count = max(1, int(peer_count))
        peers = max(0, self.peer_count - 1)
        classes = tuple(sorted(
            (int(peer), ("hccs" if str(value).endswith("hccs") or
                         str(value).lower() in {"local", "intra", "same_node"}
                         else "roce"))
            for peer, value in dict(peer_classes or {}).items()))
        if peers:
            if peer_weights is None:
                weights = tuple(Fraction(1, peers) for _ in range(peers))
            else:
                weights = tuple(as_fraction(item) for item in peer_weights)
                if (len(weights) != peers
                        or sum(weights, Fraction(0)) != 1):
                    raise ModelingError("mc2_peer_weights_invalid")
        else:
            weights = ()
        if rank >= 0 and rank < self.peer_count:
            labels = tuple(item for item in range(self.peer_count)
                           if item != int(rank))
        else:
            labels = tuple(range(peers))
        if latencies is None:
            intra = inter = as_fraction(latency)
        else:
            intra, inter = (as_fraction(item) for item in latencies)
        self.view = Mc2LaunchView(
            rank=int(rank), peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=max(0, int(node_ranks)),
            intra_latency_ms=intra, inter_latency_ms=inter,
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(resource_names or {}).items())),
            payload_frames=tuple(payload_frames or ()),
            payload_packet_bytes=int(payload_packet_bytes or 0),
            payload_basis=str(payload_basis),
            payload_local_frames=int(payload_local_frames))
        self.aiv = max(2, int(self.units.get("vector", 64)))
        #: Exact source force-EPLB assignment cells, indexed [source][expert
        #: destination].  They are whole routed rows, never a statistical
        #: expectation promoted to observed destinations.
        self.route_matrix = (
            tuple(tuple(int(value) for value in row) for row in route_matrix)
            if route_matrix else None)
        self.route_reverse = bool(route_reverse)
        self.exact_payload = bool(exact_payload)
        #: Expected directed payload bytes from a declared statistical routing
        #: policy.  These are policy expectations, not integers observed at a
        #: peer, so they never populate payload_frames.
        self.route_matrix_bytes = (
            tuple(tuple(float(value) for value in row)
                  for row in route_matrix_bytes)
            if route_matrix_bytes else None)
        #: Declared useful payload bytes one source sends across all peers for
        #: a uniform peer policy.  Each edge gets an equal share.
        self.uniform_payload_bytes = float(uniform_payload_bytes or 0)
        self.experts = max(0, int(experts))
        #: Whole-call policy basis.  A rank-local frame set may be empty even
        #: when other EP sources are active; that does not make the operation a
        #: known-zero source.
        self.transport_basis = str(transport_basis or payload_basis)
        #: Mapped phase service work, excluding peer link bytes.  A launch-local
        #: rank adds its own exact directed link work below.
        self._phase_work = {}
        self.work = ()

    def frames_for(self, rank):
        """Whole routed frames this launch rank sends to each remote peer."""
        matrix = self.route_matrix
        if not matrix:
            return ()
        rank = int(rank)
        if not 0 <= rank < len(matrix):
            return ()
        frames = []
        for peer in range(len(matrix)):
            if peer == rank:
                continue
            count = (matrix[peer][rank] if self.route_reverse
                     else matrix[rank][peer])
            if count > 0:
                frames.append((peer, int(count)))
        return tuple(sorted(frames))

    def local_frames_for(self, rank):
        matrix = self.route_matrix
        if not matrix:
            return 0
        rank = int(rank)
        if not 0 <= rank < len(matrix):
            return 0
        return int(matrix[rank][rank])

    def payload_basis_for(self, rank):
        if not self.route_matrix:
            return self.view.payload_basis
        return ("actual_zero" if not self.frames_for(rank)
                else self.view.payload_basis)

    @property
    def has_payload_policy(self):
        return bool(self.route_matrix is not None
                    or self.route_matrix_bytes is not None
                    or self.uniform_payload_bytes > 0)

    def payload_bytes(self, source, destination):
        """Useful payload bytes for one directed source->destination cell."""
        source, destination = int(source), int(destination)
        if source == destination:
            return 0.0
        if self.route_matrix:
            count = (self.route_matrix[destination][source]
                     if self.route_reverse else self.route_matrix[source][destination])
            return float(count) * float(self.view.payload_packet_bytes)
        if self.route_matrix_bytes:
            return float(self.route_matrix_bytes[source][destination])
        ep = max(1, self.peer_count)
        return (self.uniform_payload_bytes / ep
                if ep > 1 else 0.0)

    def frame_count(self, source, destination):
        if not self.route_matrix or source == destination:
            return 0
        return int(self.route_matrix[destination][source]
                   if self.route_reverse else self.route_matrix[source][destination])

    def count_bytes(self, source_index, group_size):
        ep = max(1, int(group_size))
        if ep <= 1:
            return 0.0
        return float(self.experts * 32 * (ep - 1)) / ep

    def transport_cells(self, transport, count=False):
        """Directed cells this launch owns under its actual EP membership."""
        group = tuple(transport.get("group") or ())
        sources = tuple(transport.get("sources") or ())
        if len(group) <= 1:
            return ()
        cells = []
        for source in sources:
            if not 0 <= source < len(group):
                continue
            if count:
                # Contiguous expert intervals: every remote peer receives its
                # own (experts/ep)*32 record, not the whole count on a ring
                # neighbour.  The directed arrivals stay independent.
                experts_per_peer = max(0, self.experts // len(group))
                payload = float(experts_per_peer * 32)
                if payload > 0:
                    for destination in range(len(group)):
                        if destination == source:
                            continue
                        cells.append((source, destination, group[source],
                                      group[destination], payload, 0))
                continue
            for destination in range(len(group)):
                if destination == source:
                    continue
                payload = self.payload_bytes(source, destination)
                if payload <= 0:
                    continue
                cells.append((source, destination, group[source],
                              group[destination], payload,
                              self.frame_count(source, destination)))
        return tuple(cells)

    def transport_account(self, transport):
        """Physical coupled work for every directed cell this launch owns."""
        from mindie_sim.prediction.communication import MessageFlow
        route = transport.get("route")
        payload = {"work": {}, "edges": 0, "frames": 0,
                   "tx_bytes": 0.0, "rx_bytes": 0.0, "useful": 0.0,
                   "useful_bytes": 0.0, "sources": tuple(transport.get("sources") or ()),
                   "group": tuple(transport.get("group") or ())}
        for _si, _di, source, destination, useful, frames in self.transport_cells(
                transport):
            if not callable(route) or useful <= 0:
                continue
            message = route(MessageFlow(source, destination, float(useful)))
            for hop in message.hops:
                for name, amount in hop.demand.items():
                    if not amount:
                        continue
                    payload["work"][str(name)] = (
                        payload["work"].get(str(name), 0.0) + float(amount))
                    lowered = str(name).lower()
                    if ":tx:" in lowered:
                        payload["tx_bytes"] += float(amount)
                    elif ":rx:" in lowered:
                        payload["rx_bytes"] += float(amount)
                    else:
                        # A directed per-pair resource is both source TX and
                        # destination RX of the one transfer.
                        payload["tx_bytes"] += float(amount)
                        payload["rx_bytes"] += float(amount)
            payload["edges"] += 1
            payload["frames"] += int(frames)
        payload["useful"] = sum(
            float(useful) for _si, _di, _source, _destination, useful, _frames
            in self.transport_cells(transport) if useful > 0)
        payload["useful_bytes"] = payload["useful"]
        count = {"work": {}, "edges": 0, "frames": 0,
                 "tx_bytes": 0.0, "rx_bytes": 0.0, "useful": 0.0,
                 "useful_bytes": 0.0, "sources": payload["sources"], "group": payload["group"]}
        for _si, _di, source, destination, useful, _frames in self.transport_cells(
                transport, count=True):
            if not callable(route) or useful <= 0:
                continue
            message = route(MessageFlow(source, destination, float(useful)))
            for hop in message.hops:
                for name, amount in hop.demand.items():
                    if not amount:
                        continue
                    count["work"][str(name)] = (
                        count["work"].get(str(name), 0.0) + float(amount))
                    lowered = str(name).lower()
                    if ":tx:" in lowered:
                        count["tx_bytes"] += float(amount)
                    elif ":rx:" in lowered:
                        count["rx_bytes"] += float(amount)
                    else:
                        count["tx_bytes"] += float(amount)
                        count["rx_bytes"] += float(amount)
            count["edges"] += 1
            count["useful"] += float(useful)
            count["useful_bytes"] += float(useful)
        return payload, count

    def _directed_link(self, rank):
        totals = {}
        labels = tuple(peer for peer in range(self.peer_count) if peer != rank)
        if self.route_matrix_bytes:
            # Statistical expectation uses the same directed matrix that drives
            # the native peer program.  It is not a second, independently
            # uniform distribution of the local receive volume.
            if 0 <= int(rank) < self.peer_count:
                for index, peer in enumerate(labels):
                    useful = self.payload_bytes(rank, peer)
                    if useful <= 0:
                        continue
                    klass = _peer_class_for(
                        rank, index, peer, node_ranks=self.view.node_ranks,
                        peer_count=self.peer_count, classes=self.peer_classes)
                    key = "link_hccs" if klass == "hccs" else "link_roce"
                    totals[key] = totals.get(key, Fraction(0)) + Fraction(useful)
            return totals or {"link": Fraction(0)}
        frames = self.frames_for(rank)
        packet = self.view.payload_packet_bytes
        for index, (peer, count) in enumerate(frames):
            klass = _peer_class_for(
                rank, index, peer, node_ranks=self.view.node_ranks,
                peer_count=self.peer_count, classes=self.peer_classes)
            key = "link_hccs" if klass == "hccs" else "link_roce"
            totals[key] = totals.get(key, Fraction(0)) + Fraction(count * packet)
        # A zero-frame rank still owns the named payload phase.  The empty
        # link ledger must not be mistaken for a declaration that the phase
        # has no peers at all; launch_native_cohort allocates the real edges.
        return totals or {"link": Fraction(0)}

    def link_by_phase_for(self, rank):
        links = {phase: dict(values)
                 for phase, values in self.link_by_phase.items()}
        if self.route_matrix or self.route_matrix_bytes:
            links["payload_exchange"] = self._directed_link(rank)
        return links

    def work_for_rank(self, rank):
        work = dict(self._phase_work)
        for values in self.link_by_phase_for(rank).values():
            for pipe, amount in values.items():
                exact = as_fraction(amount)
                if not exact:
                    continue
                key = self.names.get(pipe, pipe)
                work[key] = work.get(key, Fraction(0)) + exact
        return tuple(sorted(work.items()))

    @property
    def rank(self) -> int:
        return self.view.rank

    @property
    def node_ranks(self) -> int:
        return self.view.node_ranks

    @property
    def peer_labels(self) -> tuple[int, ...]:
        return self.view.peer_labels

    @property
    def peer_classes(self) -> tuple[tuple[int, str], ...]:
        return self.view.peer_classes

    @property
    def peer_weights(self) -> tuple[Fraction, ...]:
        return self.view.peer_weights

    def _peer_class(self, index: int, peer: int,
                    view: Mc2LaunchView | None = None) -> str:
        return (view or self.view).class_for(index, peer)

    @property
    def peer_class_basis(self) -> int:
        return int(bool(self.view.peer_classes) or
                   (self.view.rank >= 0 and self.view.node_ranks > 0))

    def _endpoint_moved(self, transport, phase, count, view=None):
        endpoints = {
            int(rank): str(name) for rank, name in
            (dict(transport or {}).get("destination_memory") or {}).items()
            if name}
        group = tuple((transport or {}).get("group") or ())
        destination = int((view or self.view).rank)
        if not endpoints or destination not in group or not endpoints.get(destination):
            return 0.0
        destination_index = group.index(destination)
        total = 0.0
        for source_index in range(len(group)):
            if source_index == destination_index:
                continue
            if count:
                useful = float(self.experts * 32 / max(1, len(group)))
            else:
                useful = self.payload_bytes(source_index, destination_index)
            if useful > 0:
                total += float(useful)
        return total

    def _endpoint_phase_program(self, program, transport, phase, count,
                                 view=None, resource_names=None):
        moved = self._endpoint_moved(transport, phase, count, view)
        if moved <= 0:
            return program, 0.0
        resource = dict(resource_names or {}).get(
            "hbm", self.names.get("hbm", "hbm"))
        return _subtract_endpoint_hbm(program, resource, moved)

    def _context(self, view=None):
        view = view or self.view
        names = dict(self.names)
        names.update(dict(view.resource_names))
        return RecipeContext(
            op=None, work={}, resource_names=names, rates=self.rates,
            core_count=self.aiv, resource_units=self.units, pool_names={},
            store_queues={}, service_modes=self.service_modes)

    def _schedule_transport(self, engine, name, phase, prerequisite, view,
                            transport, *, count=False, placeholders=None):
        """Route this launch's whole-EP directed cells once, coupled TX+RX.

        One transmission reserves every physical hop capacity in ``hop.demand``
        (source TX and destination RX together) and has one arrival Event.  The
        local rank's send handle is the same first-hop service; no second wire
        reservation or second Engine is created.
        """
        from mindie_sim.prediction.communication import MessageFlow
        route = transport.get("route")
        group = tuple(transport.get("group") or ())
        local_indices = {int(item) for item in
                         (transport.get("sources") or ())}
        if not local_indices and 0 <= int(view.rank) < len(group):
            local_indices = {int(view.rank)}
        work = {}
        peer_events = {}
        parent_events = []
        frames = 0
        edges = 0
        tx_bytes = 0.0
        rx_bytes = 0.0
        useful_bytes = 0.0
        endpoint_memory = {
            int(rank): str(name) for rank, name in
            (transport.get("destination_memory") or {}).items() if name}
        endpoint_resources = set(endpoint_memory.values())
        for _si, _di, source, destination, useful, cell_frames in self.transport_cells(
                transport, count=count):
            if not callable(route) or useful <= 0:
                continue
            message = route(MessageFlow(source, destination, float(useful)))
            previous = (prerequisite,)
            first_service = None
            endpoint = endpoint_memory.get(int(destination))
            for hop_index, hop in enumerate(message.hops):
                demand = {str(name): float(amount)
                          for name, amount in hop.demand.items() if amount}
                if (endpoint
                        and hop_index == len(message.hops) - 1
                        and source != destination
                        and (not group or group[source] != group[destination])):
                    demand[endpoint] = demand.get(endpoint, 0.0) + float(useful)
                service = engine.submit(
                    f"{name}:{phase.name}:{source}->{destination}:"
                    f"hop{hop_index}:service",
                    work=demand, after=previous, coupled=True)
                if first_service is None:
                    first_service = service
                for key, amount in demand.items():
                    work[key] = work.get(key, 0.0) + amount
                    if key in endpoint_resources:
                        continue
                    lowered = key.lower()
                    if ":tx:" in lowered:
                        tx_bytes += amount
                    elif ":rx:" in lowered:
                        rx_bytes += amount
                    else:
                        tx_bytes += amount
                        rx_bytes += amount
                arrival = engine.submit(
                    f"{name}:{phase.name}:{source}->{destination}:"
                    f"hop{hop_index}:arrival",
                    latency=float(hop.latency_ms), after=(service,))
                previous = (arrival,)
            arrival = previous[0]
            final = arrival
            promise = dict(placeholders or {}).get(
                (_si, _di, phase.name))
            if promise is not None:
                final = engine.submit(
                    f"{name}:{phase.name}:{source}->{destination}:retire",
                    after=(arrival,),
                    on_complete=lambda _event, item=promise:
                    engine.resolve(item))
            parent_events.append(final)
            frames += int(cell_frames)
            edges += 1
            useful_bytes += float(useful)
            if not count and source in local_indices:
                peer_events[destination] = {
                    "send": first_service, "arrival": arrival,
                    "receive": None, "check": None,
                    "retire": (final if final is not arrival else None),
                    "class": view.class_for(
                        view.peer_labels.index(destination)
                        if destination in view.peer_labels else 0,
                        destination),
                    "pipe": None, "latency_ms": float(message.hops[-1].latency_ms)
                    if message.hops else 0.0,
                    "frames": int(cell_frames), "bytes": float(useful),
                    "send_placeholder": None, "remote_send": None,
                }
        # A local receive/check/copy must wait the actual directed arrivals
        # from every remote source, not just this rank's outgoing sends.  The
        # cohort launcher creates these futures before any program starts and
        # each sender resolves its own cell; no timestamp or global barrier is
        # synthesized here.
        incoming = []
        for peer in range(len(group)):
            for destination in sorted(local_indices):
                if peer == destination:
                    continue
                promise = dict(placeholders or {}).get(
                    (peer, destination, phase.name))
                if promise is not None:
                    incoming.append(promise)
        complete = engine.submit(
            f"{name}:{phase.name}:peer_complete",
            after=tuple(dict.fromkeys(tuple(parent_events) + tuple(incoming)))
            or (prerequisite,))
        return {
            "event": complete, "peer_events": peer_events, "work": work,
            "frames": frames, "edges": edges, "tx_bytes": tx_bytes,
            "rx_bytes": rx_bytes, "useful_bytes": useful_bytes,
            "group": group, "sources": tuple(transport.get("sources") or ()),
            "basis": self.transport_basis, "count": bool(count),
        }

    def start(self, engine, name, ready, records=None, peer_ready=None,
              peer_send=None, launch=None, source_send=None,
              send_placeholders=None, transport=None):
        event, _handles = self.start_with_handles(
            engine, name, ready, records=records, peer_ready=peer_ready,
            peer_send=peer_send, launch=launch, source_send=source_send,
            send_placeholders=send_placeholders, transport=transport)
        return event

    def _with_route(self, view):
        if not self.route_matrix:
            return view
        return replace(
            view, payload_frames=self.frames_for(view.rank),
            payload_basis=self.payload_basis_for(view.rank),
            payload_local_frames=self.local_frames_for(view.rank))

    def _launch_view(self, launch):
        if launch is None:
            return self._with_route(self.view)
        if not isinstance(launch, Mc2LaunchView):
            raise ModelingError("mc2_launch_view_required")
        if len(launch.peer_labels) != max(0, self.peer_count - 1):
            raise ModelingError("mc2_launch_view_peer_count_mismatch")
        if launch.peer_weights:
            if (len(launch.peer_weights) != len(launch.peer_labels)
                    or sum(launch.peer_weights, Fraction(0)) != 1):
                raise ModelingError("mc2_launch_view_peer_weights_invalid")
        elif launch.peer_labels:
            launch = replace(
                launch,
                peer_weights=tuple(Fraction(1, len(launch.peer_labels))
                                   for _ in launch.peer_labels))
        return self._with_route(launch)

    def start_with_handles(self, engine, name, ready, records=None,
                           peer_ready=None, peer_send=None, launch=None,
                           source_send=None, send_placeholders=None,
                           transport=None):
        """Launch once and return launch-local peer/port handles.

        ``launch`` is an immutable :class:`Mc2LaunchView` for this rank and
        topology.  ``peer_ready``/``peer_send`` are phase-scoped maps.  A
        ``source_send=(src, dst, phase) -> Event`` map is consumed only for
        the matching source rank, destination peer and phase.
        """
        view = self._launch_view(launch)
        transport = dict(transport or {})
        events = {}
        handles = {"peer_events": {}, "phase_complete": {},
                   "peer_class": {}, "ports": {}, "launch": view,
                   "transport": {}, "transport_work": {},
                   "resource_names": dict(view.resource_names),
                   "payload_basis": view.payload_basis,
                   "peer_class_basis": self.peer_class_basis}
        ready_map = dict(peer_ready or {})
        send_map = dict(peer_send or {})
        source_map = dict(source_send or {})
        placeholder_map = dict(send_placeholders or {})
        phases = {phase.name for phase in self.phases}
        for label, table in (("peer_ready", ready_map),
                             ("peer_send", send_map)):
            if set(table) - phases:
                raise ModelingError(
                    f"mc2_{label}_phase_required:"
                    + ",".join(map(str, sorted(set(table) - phases))))
        def phase_map(table, phase):
            value = table.get(phase, {})
            if not isinstance(value, Mapping):
                raise ModelingError(f"mc2_{phase}_peer_map_invalid")
            unknown = set(value) - set(view.peer_labels)
            if unknown:
                raise ModelingError(
                    f"mc2_{phase}_peer_rank_not_participant:"
                    + ",".join(map(str, sorted(unknown))))
            return dict(value)
        local_count_event = None
        for phase in self.phases:
            plan = self.plans[phase.name]
            program = plan.program(self._context(view))
            if (phase.name in {"count_status_exchange", "payload_exchange"}
                    and transport.get("destination_memory")):
                program, moved = self._endpoint_phase_program(
                    program, transport, phase,
                    phase.name == "count_status_exchange", view,
                    self._context(view).resource_names)
            retry = False
            count_retry = False
            if phase.name == "receive_flags":
                # The real MTE2/Compare success chain starts after the
                # selected inbound payload arrivals and local prerequisites.
                # Gather them before phase_ready/program.start so the actual
                # read is not launched early and followed by a fake scalar.
                if "receive_metadata" in events and "frame_payload" in events:
                    local_deps = ("frame_payload", "receive_metadata")
                elif "return_read_frame" in events:
                    local_deps = ("return_read_frame",)
                else:
                    local_deps = phase.after
                deps = tuple(events[item] for item in local_deps)
                owners = tuple((plan.metadata or {}).get(
                    "active_owner_domain") or ())
                local_destinations = {
                    int(item) for item in
                    (transport or {}).get("sources", ())}
                if not local_destinations:
                    local_destinations = {int(view.rank)}
                incoming = ()
                if owners:
                    incoming = tuple(dict.fromkeys(
                        event_ for (source, destination, phase_name), event_
                        in placeholder_map.items()
                        if destination in local_destinations
                        and phase_name == "payload_exchange"
                        and source != destination))
                # One genuinely source-shaped complete branch: combine
                # success-flag retry with exactly one active owner.  Other
                # owners/unsupported phases keep the existing causal wait.
                retry = bool(owners == (0,)
                             and any(str(stage.name).startswith("owner0:")
                                     and str(stage.name).endswith(
                                         ("read_flags", "compare_flags"))
                                     for stage in getattr(program, "stages", ())))
                init_program = read_compare_program = None
                success_sff_program = failure_sff_program = None
                if retry:
                    by_name = {str(stage.name): stage
                               for stage in getattr(program, "stages", ())}
                    init_stage = by_name.get("owner0:initialize_flags")
                    read_stage = by_name.get("owner0:read_flags")
                    compare_stage = by_name.get("owner0:compare_flags")
                    success_sff = by_name.get(
                        "owner0:compare_flags:after:primitive")
                    required = (init_stage, read_stage, compare_stage,
                                success_sff)
                    has_pipe_s = any("pipe_s" in str(role)
                                     for role in self.rates)
                    if (all(stage is not None for stage in required)
                            and (not any("pipe_s" in str(role)
                                         for role in success_sff.work)
                                 or has_pipe_s)):
                        init_stage = replace(init_stage, after=())
                        read_stage = replace(read_stage, after=())
                        compare_stage = replace(
                            compare_stage, after=(read_stage.name,))
                        failure_work = {
                            str(role): (1.0 if "pipe_s" in str(role)
                                        else float(amount))
                            for role, amount in dict(success_sff.work).items()}
                        init_program = KernelProgram((init_stage,))
                        read_compare_program = KernelProgram(
                            (read_stage, compare_stage))
                        success_sff_program = KernelProgram((
                            replace(success_sff, after=()),))
                        failure_sff_program = KernelProgram((
                            replace(success_sff, after=(),
                                    work=failure_work),))
                    else:
                        retry = False
                if not retry and incoming:
                    deps = tuple(dict.fromkeys(deps + incoming))
                handles["receive_flags_owners"] = owners
                handles["receive_flags_incoming"] = tuple(
                    item.name for item in incoming)
                handles["receive_flags_retry"] = bool(retry)
            else:
                deps = tuple(events[dependency] for dependency in phase.after)
            original_deps = deps
            if (phase.name == "receive_counts"
                    and local_count_event is not None):
                deps = (local_count_event,)
            count_specs = []
            if phase.name == "receive_counts":
                active = tuple((plan.metadata or {}).get(
                    "active_owner_domain") or ())
                by_name = {str(stage.name): stage
                           for stage in getattr(program, "stages", ())}
                count_local = {
                    int(item) for item in
                    (transport or {}).get("sources", ())}
                if not count_local:
                    count_local = {int(view.rank)}
                count_cells = {
                    (source, destination, pname): ev
                    for (source, destination, pname), ev
                    in placeholder_map.items()
                    if pname == "count_status_exchange"
                    and destination in count_local
                    and source != destination}
                status_records = max(0, int(self.experts))
                quotient, remainder = (
                    divmod(status_records, len(active)) if active else (0, 0))
                for index, owner in enumerate(active):
                    start = (index * quotient
                             + min(index, remainder))
                    end = start + quotient + int(index < remainder)
                    required_sources = {
                        row % max(1, int(self.peer_count))
                        for row in range(start, end)}
                    incoming = tuple(dict.fromkeys(
                        ev for (source, destination, pname), ev
                        in count_cells.items()
                        if source in required_sources))
                    prefix = f"owner{owner}:"
                    owner_stages = [
                        stage for stage in getattr(program, "stages", ())
                        if str(stage.name).startswith(prefix)]
                    read = next((stage for stage in owner_stages
                                 if str(stage.name).endswith(":read_status")),
                                None)
                    check = next((stage for stage in owner_stages
                                  if str(stage.name).endswith(":check_status")),
                                 None)
                    if read is None or check is None:
                        count_specs = []
                        break
                    read = replace(read, after=())
                    check = replace(check, after=(read.name,))
                    stale = {read.name, check.name}
                    tail_stages = []
                    for stage in owner_stages:
                        if stage.name in stale:
                            continue
                        after = tuple(name for name in stage.after
                                      if name not in stale)
                        tail_stages.append(replace(stage, after=after))
                    if not tail_stages:
                        count_specs = []
                        break
                    failed = KernelProgram((read, check))
                    tail = KernelProgram(tuple(tail_stages))
                    if not any(stage.work for stage in
                               (read, check, *tail.stages)):
                        count_specs = []
                        break
                    count_specs.append((owner, incoming, failed, tail))
                count_retry = bool(count_specs)
                if not count_retry:
                    deps = original_deps
            phase_ready = (engine.submit(f"{name}:{phase.name}:ready",
                                         after=deps) if deps else ready)
            if count_retry:
                phase_name = phase.name
                count_done = engine.future(
                    f"{name}:{phase_name}:success")
                remaining = [len(count_specs)]
                def owner_try(spec, after_event, *, phase_name=phase_name,
                              remaining=remaining, count_done=count_done):
                    owner, incoming, failed, tail = spec
                    attempt = [0]
                    def finish(_event):
                        remaining[0] -= 1
                        if remaining[0] == 0:
                            engine.resolve(count_done)
                    def start_gate(previous, *, owner=owner,
                                   incoming=incoming, failed=failed,
                                   tail=tail, attempt=attempt):
                        def gate(_event, *, owner=owner,
                                 incoming=incoming, failed=failed,
                                 tail=tail, attempt=attempt):
                            complete = all(item.done for item in incoming)
                            read_event = failed.start(
                                engine,
                                f"{name}:{phase_name}:owner{owner}:"
                                f"failed{attempt[0]}",
                                _event)
                            def read_done(_read, *, owner=owner,
                                          complete=complete, tail=tail,
                                          attempt=attempt):
                                if complete:
                                    final = tail.start(
                                        engine,
                                        f"{name}:{phase_name}:owner{owner}:"
                                        "success", _read)
                                    engine.submit(
                                        f"{name}:{phase_name}:owner{owner}:"
                                        "finish", after=(final,),
                                        owner="receive_count_success",
                                        on_complete=finish)
                                else:
                                    attempt[0] += 1
                                    start_gate(_read)
                            engine.submit(
                                f"{name}:{phase_name}:owner{owner}:read_done",
                                after=(read_event,), owner="receive_count_try",
                                on_complete=read_done)
                        engine.submit(
                            f"{name}:{phase_name}:owner{owner}:gate{attempt[0]}",
                            after=(previous,), owner="receive_count_try",
                            on_complete=gate)
                    start_gate(after_event)
                for spec in count_specs:
                    owner_try(spec, phase_ready)
                event = count_done
                handles["receive_counts_retry"] = True
            elif retry:
                phase_name = phase.name
                done = engine.future(f"{name}:{phase_name}:success")
                attempts = [0]
                def next_try(after_event, *, phase_name=phase_name,
                             init_program=init_program,
                             read_compare_program=read_compare_program,
                             success_sff_program=success_sff_program,
                             failure_sff_program=failure_sff_program,
                             required_incoming=incoming, done_event=done):
                    attempt = attempts[0]
                    init = init_program.start(
                        engine, f"{name}:{phase_name}:failed{attempt}:init",
                        after_event)
                    def after_init(_event, *, phase_name=phase_name,
                                   attempt=attempt,
                                   init_program=init_program,
                                   read_compare_program=read_compare_program,
                                   success_sff_program=success_sff_program,
                                   failure_sff_program=failure_sff_program,
                                   required_incoming=required_incoming,
                                   done_event=done_event):
                        # Snapshot at the actual MTE2 read start, after this
                        # try's initialize; never at pre-init try admission.
                        complete = all(item.done for item in required_incoming)
                        read_compare = read_compare_program.start(
                            engine,
                            f"{name}:{phase_name}:failed{attempt}:read_compare",
                            _event)
                        def after_compare(_rc):
                            if complete:
                                sff = success_sff_program.start(
                                    engine,
                                    f"{name}:{phase_name}:success:sff", _rc)
                                engine.submit(
                                    f"{name}:{phase_name}:success_finish",
                                    after=(sff,),
                                    owner="receive_flag_success",
                                    on_complete=lambda _done: engine.resolve(
                                        done_event))
                            else:
                                attempts[0] += 1
                                sff = failure_sff_program.start(
                                    engine,
                                    f"{name}:{phase_name}:failed{attempt}:sff",
                                    _rc)
                                next_try(sff)
                        engine.submit(
                            f"{name}:{phase_name}:failed{attempt}:compare_done",
                            after=(read_compare,), owner="receive_flag_try",
                            on_complete=after_compare)
                    engine.submit(
                        f"{name}:{phase_name}:failed{attempt}:init_done",
                        after=(init,), owner="receive_flag_try",
                        on_complete=after_init)
                next_try(phase_ready)
                event = done
            else:
                event = program.start(engine, f"{name}:{phase.name}", phase_ready)
            if phase.name == "count_status_exchange":
                local_count_event = event
            link = self.link_by_phase.get(phase.name)
            ready_by_peer = phase_map(ready_map, phase.name)
            send_by_peer = phase_map(send_map, phase.name)
            if (transport is not None
                    and phase.name in {"payload_exchange", "count_status_exchange"}
                    and (self.has_payload_policy
                         or phase.name == "count_status_exchange")):
                details = self._schedule_transport(
                    engine, name, phase, event, view, transport,
                    count=phase.name == "count_status_exchange",
                    placeholders=placeholder_map)
                event = details["event"]
                handles["transport"][phase.name] = {
                    key: value for key, value in details.items()
                    if key not in {"event", "peer_events"}}
                for key, amount in details["work"].items():
                    handles["transport_work"][key] = (
                        handles["transport_work"].get(key, 0.0) + amount)
                if details["peer_events"]:
                    handles["peer_events"][phase.name] = dict(
                        details["peer_events"])
            elif phase.name == "payload_exchange" and view.payload_frames:
                # Whole-frame projected token payload: exact selected
                # destinations.  The source transmission is one physical issue
                # per real edge; the reservation resolves only on the arrival
                # task after link/propagation, never at send.
                peer_events = []
                for peer, frames in view.payload_frames:
                    if frames <= 0 or peer not in view.peer_labels:
                        continue
                    index = view.peer_labels.index(peer)
                    klass = view.class_for(index, peer)
                    pipe = "link_hccs" if klass == "hccs" else "link_roce"
                    key = dict(view.resource_names).get(
                        pipe, self.names.get(pipe, pipe))
                    amount = as_fraction(frames) * as_fraction(
                        view.payload_packet_bytes)
                    placeholder = placeholder_map.get(
                        (view.rank, peer, phase.name))
                    send = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:send",
                        work={key: float(amount)} if amount else {},
                        after=(event,))
                    arrival = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:arrival",
                        latency=float(view.latency_for(klass)), after=(send,),
                        on_complete=(
                            (lambda _event, promise=placeholder:
                             engine.resolve(promise))
                            if placeholder is not None else None))
                    peer_events.append(send)
                    if placeholder is None:
                        # No peer receive will consume this arrival (isolated
                        # representative launch). The send is complete only
                        # when its declared propagation has elapsed.
                        peer_events.append(arrival)
                    handles["peer_events"].setdefault(
                        phase.name, {})[peer] = {
                            "send": send, "arrival": arrival,
                            "receive": None, "check": None, "retire": None,
                            "class": klass, "pipe": key,
                            "latency_ms": view.latency_for(klass),
                            "frames": frames, "bytes": amount,
                            "send_placeholder": placeholder,
                            "remote_send": None}
                for index, peer in enumerate(view.peer_labels):
                    incoming = placeholder_map.get(
                        (peer, view.rank, phase.name))
                    if incoming is None:
                        continue
                    klass = view.class_for(index, peer)
                    receive = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:receive",
                        after=(event, incoming))
                    check = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:check",
                        after=(receive,))
                    retire = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:retire",
                        after=(check,))
                    peer_events.append(retire)
                    slot = handles["peer_events"].setdefault(
                        phase.name, {}).setdefault(peer, {})
                    slot.update({
                        "receive": receive, "check": check, "retire": retire,
                        "class": klass, "latency_ms": view.latency_for(klass),
                        "remote_send": incoming})
                handles["payload_frames"] = dict(view.payload_frames)
                handles["payload_basis"] = view.payload_basis
                handles["payload_local_frames"] = view.payload_local_frames
                if peer_events:
                    event = engine.submit(
                        f"{name}:{phase.name}:peer_complete",
                        after=tuple(peer_events))
            elif (link and view.peer_labels and
                  not (phase.name == "payload_exchange" and
                       (self.exact_payload or
                        view.payload_basis == "unknown"))):
                class_peers = {"hccs": [], "roce": []}
                for index, peer in enumerate(view.peer_labels):
                    klass = view.class_for(index, peer)
                    class_peers[klass].append((index, peer))
                    handles["peer_class"][peer] = klass
                class_weight = {}
                for klass, members in class_peers.items():
                    class_weight[klass] = sum(
                        (view.peer_weights[index] for index, _ in members),
                        Fraction(0))
                peer_events = []
                for klass, members in class_peers.items():
                    if not members:
                        continue
                    pipe = "link_hccs" if klass == "hccs" else "link_roce"
                    total = as_fraction(link.get(pipe, 0))
                    if (total <= 0 and "link" in link and
                            sum(1 for items in class_peers.values() if items) == 1):
                        # A one-class synthetic supply declares only `link`.
                        total = as_fraction(link.get("link", 0))
                    key = dict(view.resource_names).get(
                        pipe, self.names.get(pipe, pipe))
                    for index, peer in members:
                        weight = (view.peer_weights[index] /
                                  class_weight[klass]
                                  if class_weight[klass] else Fraction(0))
                        send = send_by_peer.get(peer)
                        if send is None:
                            send = source_map.get(
                                (view.rank, peer, phase.name))
                        placeholder = placeholder_map.get(
                            (view.rank, peer, phase.name))
                        arrival = None
                        if send is None:
                            work = ({key: float(total * weight)}
                                    if total * weight else {})
                            send = engine.submit(
                                f"{name}:{phase.name}:peer{peer}:send",
                                work=work, after=(event,))
                            arrival = engine.submit(
                                f"{name}:{phase.name}:peer{peer}:arrival",
                                latency=float(view.latency_for(klass)),
                                after=(send,),
                                on_complete=(
                                    (lambda _event, _placeholder=placeholder:
                                     engine.resolve(_placeholder))
                                    if placeholder is not None else None))
                        incoming = source_map.get(
                            (peer, view.rank, phase.name))
                        if incoming is None:
                            incoming = placeholder_map.get(
                                (peer, view.rank, phase.name))
                        receive = check = retire = None
                        if incoming is not None:
                            receive_deps = (incoming,)
                            if peer in ready_by_peer:
                                receive_deps += (ready_by_peer[peer],)
                            receive = engine.submit(
                                f"{name}:{phase.name}:peer{peer}:receive",
                                after=receive_deps)
                            check = engine.submit(
                                f"{name}:{phase.name}:peer{peer}:check",
                                after=(receive,))
                            retire = engine.submit(
                                f"{name}:{phase.name}:peer{peer}:retire",
                                after=(check,))
                            peer_events.append(retire)
                        peer_events.append(send)
                        slot = handles["peer_events"].setdefault(
                            phase.name, {}).setdefault(peer, {})
                        slot.update({
                            "send": send, "arrival": arrival,
                            "receive": receive, "check": check,
                            "retire": retire, "class": klass, "pipe": key,
                            "latency_ms": view.latency_for(klass),
                            "send_placeholder": placeholder,
                            "remote_send": incoming})
                        handles["ports"][peer] = {
                            "source": view.rank, "peer": peer,
                            "class": klass, "pipe": key,
                            "latency_ms": view.latency_for(klass)}
                event = engine.submit(
                    f"{name}:{phase.name}:peer_complete",
                    after=tuple(peer_events))
                handles["phase_complete"][phase.name] = event
            elif phase.name == "payload_exchange":
                # Idle/zero-source rank: it has no outbound payload wire, but
                # active peers can still send outward payload to it.  Consume
                # only the matching incoming (peer, self, phase) placeholders;
                # do not invent a local send.
                incoming_events = []
                for index, peer in enumerate(view.peer_labels):
                    incoming = placeholder_map.get(
                        (peer, view.rank, phase.name))
                    if incoming is None:
                        continue
                    klass = view.class_for(index, peer)
                    latency = view.latency_for(klass)
                    receive = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:receive",
                        after=(event, incoming))
                    check = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:check",
                        after=(receive,))
                    retire = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:retire",
                        after=(check,))
                    incoming_events.append(retire)
                    handles["peer_events"].setdefault(
                        phase.name, {})[peer] = {
                            "send": None, "receive": receive,
                            "check": check, "retire": retire,
                            "class": klass, "pipe": None,
                            "latency_ms": latency,
                            "send_placeholder": None,
                            "remote_send": incoming}
                if incoming_events:
                    event = engine.submit(
                        f"{name}:{phase.name}:peer_complete",
                        after=tuple(incoming_events))
            elif link:
                link_work = {}
                for pipe, amount in link.items():
                    key = dict(view.resource_names).get(
                        pipe, self.names.get(pipe, pipe))
                    link_work[key] = link_work.get(key, Fraction(0)) + amount
                event = engine.submit(
                    f"{name}:{phase.name}:link",
                    work={pipe: float(amount)
                          for pipe, amount in link_work.items() if amount},
                    after=(event,))
                if view.intra_latency_ms:
                    event = engine.submit(
                        f"{name}:{phase.name}:propagation",
                        latency=float(view.intra_latency_ms), after=(event,))
            # Authored per-owner primitive stages already execute above.
            # Do not replay the same inventory as an aggregate job.
            handles["phase_complete"].setdefault(phase.name, event)
            if records is not None:
                records.append((phase.name, phase_ready, event))
            events[phase.name] = event
        if not events:
            handles["start"] = ready
            handles["parent"] = ready
            handles["operators"] = {"mc2": {"start": ready, "complete": ready}}
            return ready, handles
        parent = engine.submit(f"{name}:complete", after=tuple(events.values()))
        handles["start"] = ready
        handles["parent"] = parent
        handles["operators"] = {"mc2": {"start": ready, "complete": parent}}
        return parent, handles

def _mapped_work(work: dict[str, Any], rates: dict[str, float]) -> dict[str, Fraction]:
    names = set(rates)
    mapped: dict[str, Fraction] = {}
    for pipe, amount in work.items():
        exact = as_fraction(amount)
        if exact <= 0:
            continue
        key = str(pipe)
        if key not in names and key in {"mte2_aiv", "mte2_aic"} and "mte2" in names:
            key = "mte2"
        mapped[key] = mapped.get(key, Fraction(0)) + exact
    return mapped


def _run_private(program, rates, units):
    resources = tuple(
        Resource(name, rate, "work", "fair", max(1, int(units.get(name, 1))))
        for name, rate in rates.items())
    engine = Engine(resources)
    ready = engine.submit("ready", release=0)
    records = []
    event = program.start(engine, "native", ready, records=records)
    engine.advance(until=(event,))
    elapsed = Fraction.from_float(event.time - ready.time)
    busy = {name: Fraction.from_float(value)
            for name, value in engine.busy_ms.items() if value}
    if busy:
        peak = max(busy.values())
        dominant = next(name for name, value in busy.items() if value == peak)
        tied = tuple(name for name, value in busy.items() if value == peak)
    else:
        dominant, tied = "", ()
    stage_stats = tuple(
        (name, Fraction.from_float(finished.time - started.time))
        for name, started, finished in records)
    return elapsed, dominant, tied, busy, stage_stats


@lru_cache(maxsize=128)
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
    raw = dict(raw_items)
    plan = contract(raw, aiv)
    phases = _link_work(plan, max(1, int(raw["ep"])), int(node_ranks))
    base_roles = {"vector", "mte2_aiv", "mte3", "hbm", "link"}
    all_roles = (set(dict(rates_items)) | set(dict(units_items)) | base_roles
                 | {str(name) for name, _mode in service_items})
    geometry = Geometry(
        compute_units=max(2, aiv),
        resource_units={role: max(2, int(dict(units_items).get(role, aiv)))
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
            pipe: as_fraction(amount) for pipe, amount in phase.work
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
        units.setdefault(name, max(2, int(aiv)))
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
            units.setdefault(alias, units.get(link_fallback, max(2, int(aiv))))
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


@lru_cache(maxsize=128)
def _native_execute(raw_items, tokens, aiv, rates_items, units_items,
                    service_items, latencies, node_ranks, received,
                    peer_weights, rank, peer_classes, resource_names,
                    payload_frames, payload_basis, payload_local_frames):
    raw = dict(raw_items)
    top_k = max(1, int(raw["top_k"]))
    source_assignments = Fraction(max(0, int(tokens)) * top_k)
    phase_raw = {**raw, "source_tokens": max(0, int(tokens)), "top_k": top_k}
    actual_received = as_fraction(received)
    if raw["kind"] == "dispatch":
        phase_raw["source_assignments"] = source_assignments
        phase_raw["received_assignments"] = actual_received
    elif raw["kind"] == "combine":
        phase_raw["source_assignments"] = actual_received
        phase_raw["received_assignments"] = source_assignments
    else:
        raise ModelingError("mc2_transport_kind_invalid")
    basis = str(payload_basis)
    if not payload_frames:
        # A missing witnessed route is not zero transferred payload.  The
        # declared source row still owns its physical expected payload work.
        incoming = (phase_raw["source_assignments"]
                    if raw["kind"] == "dispatch"
                    else phase_raw["received_assignments"])
        packet = contract(phase_raw, aiv).layout.packet_bytes
        volume = float(incoming) * packet
        phase_raw["uniform_payload_bytes"] = volume
        if basis.lower() in {"", "unknown"}:
            basis = ("actual_zero" if volume <= 0
                     else "declared_uniform_peer_payload")
    program, work = _native_program(
        tuple(sorted(phase_raw.items())), aiv, rates_items, units_items,
        service_items, latencies, node_ranks, peer_weights, rank,
        peer_classes, resource_names, tuple(payload_frames), basis,
        payload_local_frames, bool(payload_frames))
    rates = dict(rates_items)
    units = dict(units_items)
    for name in rates:
        units.setdefault(name, max(2, int(aiv)))
    elapsed, dominant, tied, busy, stage_stats = _run_private(
        program, rates, units)
    return (elapsed, dominant, tied, busy, work, stage_stats, program)


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
            values.append(as_fraction(latency) / 1000)
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
    aiv = max(2, int(getattr(context, "core_count", 0)
                     or dict(units).get("vector", 64) or 64))
    program, _work = _native_program(
        tuple(sorted(data.items())), aiv, rates, units, services, latencies,
        node_ranks, None, rank, _transport_peer_classes(context, rank, ep),
        resources, frames, basis, local_frames, exact)
    return program


def _peer_latencies(ctx: CompiledOperator, ep: int) -> tuple[Fraction, Fraction]:
    """True intra/inter fabric propagation, one value per declared class.

    Local peers use the source HCCS latency; interhost peers use the source
    RoCE latency.  No global max is applied to every peer.
    """
    if ep <= 1:
        return Fraction(0), Fraction(0)
    inv = dict(getattr(ctx.supply, "invocations", ()) or ())
    intra = as_fraction(inv.get("link_latency_intra_ms",
                                inv.get("link_latency_ms", 0)))
    inter = as_fraction(inv.get("link_latency_inter_ms", intra))
    return intra, inter


def _native_result(ctx: CompiledOperator, config: MoeCommConfig,
                   raw: dict[str, Any], geometry: Any, state: Any, aiv: int):
    if not _selected(config, geometry):
        return None
    received, _declared = _received_rows(ctx, geometry, state, raw)
    meta = dict(ctx.feature_metadata or ())
    weights = meta.get("moe_peer_weights")
    peer_weights = (tuple(as_fraction(item) for item in weights)
                    if isinstance(weights, (tuple, list)) else None)
    rank = _rank_index(ctx)
    resource_names = meta.get("moe_resource_names")
    resource_items = (tuple(sorted(
        (str(name), str(value)) for name, value in resource_names.items()))
        if isinstance(resource_names, Mapping) else None)
    source_assignments = (received if raw["kind"] == "combine" else
                          Fraction(max(0, int(local_tokens(geometry)))
                                   * max(1, int(raw["top_k"]))))
    frames, basis, local_frames = _payload_frames(
        ctx, geometry, state, raw, source_assignments)
    classes = meta.get("moe_peer_classes")
    peer_classes = None
    if isinstance(classes, Mapping):
        peer_classes = tuple(sorted(
            (int(peer), str(value)) for peer, value in classes.items()))
    else:
        nodes = meta.get("moe_peer_nodes")
        if isinstance(nodes, Mapping) and rank >= 0:
            source_node = nodes.get(rank, nodes.get(str(rank)))
            if source_node is not None:
                peer_classes = tuple(sorted(
                    (int(peer), "hccs" if int(node) == int(source_node)
                     else "roce")
                    for peer, node in nodes.items()
                    if int(peer) != rank))
    return _native_execute(
        tuple(sorted(raw.items())), max(0, int(local_tokens(geometry))), aiv,
        tuple(sorted((str(name), float(rate)) for name, rate in ctx.supply.rates)),
        tuple(sorted((str(name), int(value))
                     for name, value in (ctx.supply.resource_units or ()))),
        tuple(sorted((str(name), (str(role), float(rate)))
                     for name, (role, rate) in (ctx.supply.service_modes or ()))),
        _peer_latencies(ctx, max(1, int(raw["ep"]))),
        int(dict(ctx.supply.invocations or ()).get("ranks_per_node", 0) or 0),
        received, peer_weights, rank, peer_classes, resource_items,
        frames, basis, local_frames)


def _selected(config: MoeCommConfig | None, geometry: Any) -> bool:
    if config is None:
        return False
    return config.select(local_tokens(geometry)) == MC2


def compile_mc2_transport(ctx: CompiledOperator,
                          operator: BoundOperator) -> FamilyTemplate:
    raw = _raw(operator)
    config = _config(operator)
    if config is None:
        raise ModelingError("mc2_transport_requires_selected_recipe")
    kind = str(raw.get("kind", ""))
    if kind not in _NATIVE:
        raise ModelingError("mc2_transport_kind_invalid")
    aiv = _aiv_count(ctx)
    node_ranks = int(dict(ctx.supply.invocations or {}).get(
        "ranks_per_node", 0) or 0)
    placeholder = _placeholder(raw, aiv, node_ranks)
    pipes = tuple(tuple(sorted({_map_pipe(ctx, pipe) for pipe, _ in phase.work}))
                  for phase in placeholder)
    mt = max(1, int(getattr(ctx.policy, "mt", 1) or 1))
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("mc2_tiles", "tokens", mt), "max_tokens"),)
    _tx_name, native = _NATIVE[kind]
    rates_key = tuple(sorted(
        (str(name), float(rate)) for name, rate in ctx.supply.rates))
    units_key = tuple(sorted(
        (str(name), int(value))
        for name, value in (ctx.supply.resource_units or ())))

    def binder(geometry, _state) -> dict[str, Any]:
        values = selection_bindings(config, geometry)
        active = int(values.get("MOE_COMM_SELECTED_MC2", 0))
        if not active:
            for index, phase_pipes in enumerate(pipes):
                for pipe in phase_pipes:
                    values[f"MC2_P{index}_{pipe}"] = 0
            values.update({
                "MC2_SOURCE_TOKENS": 0, "MC2_SOURCE_ASSIGNMENTS": 0,
                "MC2_RECEIVED_ASSIGNMENTS": 0, "MC2_PACKET_BYTES": 0,
                "MC2_DATA_REGION_BYTES": 0, "MC2_SCALE_BYTES": 0,
                "MC2_LINK_BYTES": 0, "MC2_PHASE_COUNT": 0,
                "MC2_RECEIVED_BASIS": 0, "MC2_GROUP_READY_DECLARED": 0,
                "MC2_PEER_WEIGHT_DECLARED": 0,
                "MC2_PEER_CLASS_BASIS": 0,
                "MC2_UNPRICED_SCALAR_UB_ACCESS": 0,
                "MC2_UNPRICED_SCALAR_ALU_ISSUE": 0,
                "MC2_UNPRICED_PIPE_BARRIER": 0,
                "MC2_UNPRICED_CROSS_CORE_SYNC": 0,
                "MC2_UNPRICED_VEC_DUPLICATE_OPS": 0,
                "MC2_UNPRICED_VEC_SUB_OPS": 0,
                "MC2_UNPRICED_VEC_ABS_OPS": 0,
                "MC2_UNPRICED_FP32_ABS_OPS": 0,
                "MC2_UNPRICED_VEC_MINS_OPS": 0,
                "MC2_UNPRICED_VEC_REDUCE_SUM_OPS": 0,
                "MC2_UNPRICED_FP32_REDUCE_SUM_OPS": 0,
                "MC2_UNPRICED_INT32_UB_READ_BYTES": 0,
                "MC2_UNPRICED_INT32_UB_WRITE_BYTES": 0,
                "MC2_UNPRICED_PIPE_V_ISSUE": 0,
                "MC2_UNPRICED_V_S_DEPENDENCY": 0,
                "MC2_UNPRICED_GET_SFF_VALUE": 0,
                "MC2_UNPRICED_PIPE_S_ISSUE": 0,
                "MC2_UNPRICED_REDUCE_SUM_ISSUE": 0,
                "MC2_UNPRICED_PIPE_V_LANES": 0,
                "MC2_UNPRICED_UB_READ_BYTES": 0,
                "MC2_UNPRICED_UB_WRITE_BYTES": 0,
                "MC2_UNPRICED_PRIMITIVE_TOTAL": 0,
            })
            return values
        received, received_declared = _received_rows(
            ctx, geometry, _state, raw)
        plan = _plan(raw, geometry, aiv, received)
        phases = _link_work(plan, max(1, int(raw["ep"])), node_ranks)
        for index, phase in enumerate(phases):
            for pipe in pipes[index]:
                values[f"MC2_P{index}_{pipe}"] = 0
            for pipe, amount in phase.work:
                mapped = _map_pipe(ctx, pipe)
                values[f"MC2_P{index}_{mapped}"] = (
                    values.get(f"MC2_P{index}_{mapped}", 0) + amount)
        tokens, _top_k, _experts, source_assignments, received = _counts(
            raw, geometry, received)
        values.update({
            "MC2_SOURCE_TOKENS": tokens,
            "MC2_SOURCE_ASSIGNMENTS": source_assignments,
            "MC2_RECEIVED_ASSIGNMENTS": received,
            "MC2_PACKET_BYTES": plan.layout.packet_bytes,
            "MC2_DATA_REGION_BYTES": plan.layout.data_region_bytes,
            "MC2_SCALE_BYTES": plan.layout.scale_bytes,
            "MC2_LINK_BYTES": sum(
                (as_fraction(value) for phase in phases
                 for pipe, value in phase.work
                 if pipe in {"link", "link_hccs", "link_roce"}),
                Fraction(0)),
            "MC2_PHASE_COUNT": len(phases),
            "MC2_RECEIVED_BASIS": int(received_declared),
        })
        # Full-group sibling events require the PCOM/F17E handle interface.
        # Until those actual handles are supplied, readiness stays local and
        # this is explicitly not a fabricated group skew.
        values["MC2_GROUP_READY_DECLARED"] = 0
        peer_meta = dict(ctx.feature_metadata or ())
        values["MC2_PEER_WEIGHT_DECLARED"] = int(
            isinstance(peer_meta.get("moe_peer_weights"), (tuple, list)))
        rank = _rank_index(ctx)
        nodes = peer_meta.get("moe_peer_nodes")
        node_known = (isinstance(nodes, Mapping) and rank >= 0 and
                      (rank in nodes or str(rank) in nodes))
        values["MC2_PEER_CLASS_BASIS"] = int(
            bool(peer_meta.get("moe_peer_classes") or node_known or
                 (rank >= 0 and node_ranks > 0)))
        unpriced = dict(_unpriced_counts(
            tuple(sorted(raw.items())), tokens, aiv, received,
            rates_key, units_key))
        aliases = {
            "scalar_ub_access": "MC2_UNPRICED_SCALAR_UB_ACCESS",
            "scalar_alu_issue": "MC2_UNPRICED_SCALAR_ALU_ISSUE",
            "pipe_barrier": "MC2_UNPRICED_PIPE_BARRIER",
            "cross_core_sync": "MC2_UNPRICED_CROSS_CORE_SYNC",
            "vec_duplicate_ops": "MC2_UNPRICED_VEC_DUPLICATE_OPS",
            "vec_sub_ops": "MC2_UNPRICED_VEC_SUB_OPS",
            "vec_abs_ops": "MC2_UNPRICED_VEC_ABS_OPS",
            "fp32_abs_ops": "MC2_UNPRICED_FP32_ABS_OPS",
            "vec_mins_ops": "MC2_UNPRICED_VEC_MINS_OPS",
            "vec_reduce_sum_ops": "MC2_UNPRICED_VEC_REDUCE_SUM_OPS",
            "fp32_reduce_sum_ops": "MC2_UNPRICED_FP32_REDUCE_SUM_OPS",
            "int32_ub_read_bytes": "MC2_UNPRICED_INT32_UB_READ_BYTES",
            "int32_ub_write_bytes": "MC2_UNPRICED_INT32_UB_WRITE_BYTES",
            "pipe_v_issue": "MC2_UNPRICED_PIPE_V_ISSUE",
            "v_s_dependency": "MC2_UNPRICED_V_S_DEPENDENCY",
            "get_sff_value_u64_i64": "MC2_UNPRICED_GET_SFF_VALUE",
            "pipe_s_issue": "MC2_UNPRICED_PIPE_S_ISSUE",
            "reduce_sum_issue": "MC2_UNPRICED_REDUCE_SUM_ISSUE",
            "pipe_v_lanes": "MC2_UNPRICED_PIPE_V_LANES",
            "ub_read_bytes": "MC2_UNPRICED_UB_READ_BYTES",
            "ub_write_bytes": "MC2_UNPRICED_UB_WRITE_BYTES",
        }
        for role, key in aliases.items():
            values[key] = as_fraction(unpriced.get(role, 0))
        values["MC2_UNPRICED_PRIMITIVE_TOTAL"] = sum(
            (as_fraction(value) for value in unpriced.values()), Fraction(0))
        return values

    def duration(geometry, _state):
        result = _native_result(ctx, config, raw, geometry, _state, aiv)
        if result is None:
            return None
        elapsed, dominant, tied, busy, work, _stage_stats, program = result
        return (elapsed, dominant, tied, busy, dict(work),
                NativeProgram(program, work, elapsed))

    def stages(geometry, _state):
        result = _native_result(ctx, config, raw, geometry, _state, aiv)
        return () if result is None else result[5]

    stage_template = tuple(
        tuple(_demand(ctx, pipe, Var(f"MC2_P{index}_{pipe}")) for pipe in phase_pipes)
        for index, phase_pipes in enumerate(pipes))

    derived = (
        ("mc2_source_tokens", Var("MC2_SOURCE_TOKENS")),
        ("mc2_source_assignments", Var("MC2_SOURCE_ASSIGNMENTS")),
        ("mc2_received_assignments", Var("MC2_RECEIVED_ASSIGNMENTS")),
        ("mc2_packet_bytes", Var("MC2_PACKET_BYTES")),
        ("mc2_data_region_bytes", Var("MC2_DATA_REGION_BYTES")),
        ("mc2_scale_bytes", Var("MC2_SCALE_BYTES")),
        ("mc2_link_bytes", Var("MC2_LINK_BYTES")),
        ("mc2_phase_count", Var("MC2_PHASE_COUNT")),
        ("mc2_aiv_count", Const(aiv)),
        ("mc2_local_tokens", Var("MOE_LOCAL_TOKENS")),
        ("mc2_capacity", Var("MOE_COMM_CAPACITY")),
        ("mc2_selected", Var("MOE_COMM_SELECTED_MC2")),
        ("mc2_received_declared", Var("MC2_RECEIVED_BASIS")),
        ("mc2_group_ready_declared", Var("MC2_GROUP_READY_DECLARED")),
        ("mc2_peer_count", Const(max(0, int(raw["ep"]) - 1))),
        ("mc2_peer_rank", Const(_rank_index(ctx))),
        ("mc2_peer_rank_declared", Const(int(_rank_index(ctx) >= 0))),
        ("mc2_peer_weight_declared", Var("MC2_PEER_WEIGHT_DECLARED")),
        ("mc2_peer_class_basis", Var("MC2_PEER_CLASS_BASIS")),
        ("mc2_remote_service_declared", Const(0)),
        ("mc2_unpriced_scalar_ub_access",
         Var("MC2_UNPRICED_SCALAR_UB_ACCESS")),
        ("mc2_unpriced_scalar_alu_issue",
         Var("MC2_UNPRICED_SCALAR_ALU_ISSUE")),
        ("mc2_unpriced_pipe_barrier", Var("MC2_UNPRICED_PIPE_BARRIER")),
        ("mc2_unpriced_cross_core_sync", Var("MC2_UNPRICED_CROSS_CORE_SYNC")),
        ("mc2_unpriced_vec_duplicate_ops", Var("MC2_UNPRICED_VEC_DUPLICATE_OPS")),
        ("mc2_unpriced_vec_sub_ops", Var("MC2_UNPRICED_VEC_SUB_OPS")),
        ("mc2_unpriced_vec_abs_ops", Var("MC2_UNPRICED_VEC_ABS_OPS")),
        ("mc2_unpriced_fp32_abs_ops", Var("MC2_UNPRICED_FP32_ABS_OPS")),
        ("mc2_unpriced_vec_mins_ops", Var("MC2_UNPRICED_VEC_MINS_OPS")),
        ("mc2_unpriced_vec_reduce_sum_ops", Var("MC2_UNPRICED_VEC_REDUCE_SUM_OPS")),
        ("mc2_unpriced_fp32_reduce_sum_ops", Var("MC2_UNPRICED_FP32_REDUCE_SUM_OPS")),
        ("mc2_unpriced_int32_ub_read_bytes", Var("MC2_UNPRICED_INT32_UB_READ_BYTES")),
        ("mc2_unpriced_int32_ub_write_bytes", Var("MC2_UNPRICED_INT32_UB_WRITE_BYTES")),
        ("mc2_unpriced_pipe_v_issue", Var("MC2_UNPRICED_PIPE_V_ISSUE")),
        ("mc2_unpriced_v_s_dependency", Var("MC2_UNPRICED_V_S_DEPENDENCY")),
        ("mc2_unpriced_get_sff_value", Var("MC2_UNPRICED_GET_SFF_VALUE")),
        ("mc2_unpriced_pipe_s_issue", Var("MC2_UNPRICED_PIPE_S_ISSUE")),
        ("mc2_unpriced_reduce_sum_issue", Var("MC2_UNPRICED_REDUCE_SUM_ISSUE")),
        ("mc2_unpriced_pipe_v_lanes", Var("MC2_UNPRICED_PIPE_V_LANES")),
        ("mc2_unpriced_ub_read_bytes", Var("MC2_UNPRICED_UB_READ_BYTES")),
        ("mc2_unpriced_ub_write_bytes", Var("MC2_UNPRICED_UB_WRITE_BYTES")),
        ("mc2_unpriced_primitive_total", Var("MC2_UNPRICED_PRIMITIVE_TOTAL")),
    )
    return FamilyTemplate(
        axes=axes, stages=stage_template,
        materialization=(("peer_traffic", Var("MC2_LINK_BYTES")),),
        internal=(), residency=Const(0), derived=derived,
        regime_exprs=(), binder=binder,
        active=Var("MOE_COMM_SELECTED_MC2"),
        program_duration=duration, program_stages=stages,
        annotations=(("family", "transport"),
                     ("collective", _tx_name),
                     ("native_kind", native),
                     ("source_repository", "cann/ops-transformer"),
                     ("source_revision",
                      "a3925fd49cd256ad478d2faf983bf2961b8ae2c8")),
    )
