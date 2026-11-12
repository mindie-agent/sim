"""Routed expert transport: per-source framed wire and group ledgers.

[Role]
  Lower ordinary EP dispatch/combine (and the declared MC2 transport chunks)
  into directed per-rank wire work, receive hotspots and local HBM ledgers.
  Stage-to-stage KV multicast keeps its existing directed-cell declaration.
  Shared collective volume helpers remain in the sibling
  ``families_moe_collective`` module.

[Invariants]
  * A routed all-to-all starts from the genuine source-rank routed rows
    (group routed rows divided by the declared rank count), not the whole
    group payload.  The source's own share is not egress.
  * Each remote destination's useful share is split over all ``ranks`` group
    members and framed at the declared MTU before aggregation.
  * ``send_bytes`` and ``recv_hotspot`` are per-rank bottlenecks;
    ``aggregate_wire_bytes`` is the group wire total.  ``peer_bytes`` remains
    the declared group routed payload.  Per-source and group semantics are
    never substituted for each other.
  * A valid ``traffic_matrix`` is consumed before ragged placement can fail.
    Those cells are already bytes and are not rescaled by a row width.
  * ``routing_matrix`` with unit ``rows`` expands once per exchange leg.
    Payload and scale keep the direction; the return leg reverses it.
  * Ragged ``source_rows`` without peer cells are not a fabricated matrix.
    Paired ingress and egress deficits qualify a useful-byte floor for the
    declared leg.  That floor is not an exact forecast: hop placement and
    the rank-local diagonal stay unknown, and a missing received workload
    does not turn source rows into egress.
  * Same-node and inter-node peers are counted for every declared group
    index; the busiest source selects each link level, never an average.
  * Chunking partitions the same declared work and does not discount it.

[Timeouts & Locks]
  Pure bounded expression construction and O(ranks) placement counting.  No
  I/O, waits, locks or device work.

[Side-Effect Policy]
  Returns an immutable family template and helper records only.
"""
from __future__ import annotations

import math
from typing import Any

from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import BoundOperator
from .compiler import CompiledOperator, FamilyTemplate, SegmentAxis
from .expressions import Add, Const, Div, Expr, Max, Min, Mul, Sub, Var, as_fraction
from .guards import StepGuard
from .families_moe_collective import (
    DEFAULT_MTU, _chunk_stages, _collective_kind, _frac, _int, _link_pipes,
    _moe_comm_active, _moe_comm_bindings, _node_ranks, _ranks, _round_expr,
    _topology, _traffic_matrix, _uniform_alltoallv,
)


def _routed_alltoallv(local_payload: Expr, ranks: int, node_ranks: int, mtu: int
                      ) -> tuple[Expr, Expr, Expr, Expr, Expr, Expr]:
    """Directed routed all-to-all from one source rank's own routed rows.

    ``local_payload`` is the genuine source-rank share, not the whole group's
    routed rows.  The source splits that payload evenly over all ``ranks``
    destinations and keeps its own share, so only ``ranks - 1`` framed peer
    messages leave the rank.  A source's own share is never charged as an
    egress cell.  ``aggregate`` is the group wire ledger; ``send`` is one
    source's egress, so the two are never substituted for each other.

    The receive hotspot is the matching ingress bottleneck for a uniform
    payload (each destination receives one framed share from every other
    source).  Same-node and off-node peers are counted for every declared
    group index and the busiest source selects each link level; a group that
    straddles a node boundary is not flattened to a single average split.
    """
    if ranks <= 1:
        zero = Const(0)
        return zero, zero, zero, zero, zero, zero
    per_peer_useful = Div(local_payload, Const(ranks))
    per_peer_wire = _round_expr(per_peer_useful, mtu)
    peers = ranks - 1
    send = Mul((Const(peers), per_peer_wire))
    hotspot = send
    aggregate = Mul((Const(ranks), send))
    width = max(1, min(ranks, int(node_ranks)))
    intra_peers = max(
        sum(1 for peer in range(ranks)
            if peer != source and source // width == peer // width)
        for source in range(ranks))
    inter_peers = max(
        sum(1 for peer in range(ranks)
            if peer != source and source // width != peer // width)
        for source in range(ranks))
    intra = Mul((Const(intra_peers), per_peer_wire))
    inter = Mul((Const(inter_peers), per_peer_wire))
    # The declared local ledger is the per-rank egress+ingress staging
    # volume; it is a bottleneck pair, never the group total.
    hbm = Add((send, hotspot))
    return intra, inter, send, hotspot, aggregate, hbm


def _transport_matrix_volumes(matrix: tuple[tuple[int, ...], ...],
                              node_ranks: int, mtu: int, per_token: bool,
                              tokens: Expr) -> tuple[Expr, Expr, Expr, Expr, Expr, Expr]:
    """Directed routed cells, with the self share excluded from wire maxima.

    The diagonal cell is this rank's local self-copy: it is not a peer wire
    cell and is not charged to a link or to a receive hotspot.  A caller that
    needs a local self-copy stage must declare it explicitly.  The local HBM
    bottleneck pairs each rank's own egress and ingress first, then takes the
    hotspot, so two unrelated ranks' maxima are never added together.
    """
    ranks = len(matrix)
    if ranks <= 1:
        zero = Const(0)
        return zero, zero, zero, zero, zero, zero

    def framed(value: int) -> Expr:
        base = Mul((Const(int(value)), tokens)) if per_token else Const(int(value))
        return _round_expr(base, mtu)

    cells = tuple(tuple(framed(value) for value in row) for row in matrix)

    def total(values) -> Expr:
        items = tuple(values)
        return Const(0) if not items else (items[0] if len(items) == 1 else Add(items))

    sends = tuple(total(cells[i][j] for j in range(ranks) if i != j)
                  for i in range(ranks))
    recvs = tuple(total(cells[i][j] for i in range(ranks) if i != j)
                  for j in range(ranks))
    send = Max(sends)
    hotspot = Max(recvs)
    aggregate = total(sends)
    hbm = Max(tuple(Add((sends[i], recvs[i])) for i in range(ranks)))

    def level(off_node: bool) -> Expr:
        split = lambda i, j: (i // node_ranks == j // node_ranks) != off_node
        out = tuple(total(cells[i][j] for j in range(ranks)
                          if i != j and split(i, j)) for i in range(ranks))
        inc = tuple(total(cells[i][j] for i in range(ranks)
                          if i != j and split(i, j)) for j in range(ranks))
        return Max((*out, *inc))

    return level(False), level(True), send, hotspot, aggregate, hbm


def _compile_transport(ctx: CompiledOperator,
                       operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    name = operator.name.lower().replace("-", "_")
    kind = _collective_kind(name, theta)
    ranks = _ranks(theta)
    node_ranks = _node_ranks(theta, ranks)
    topology = _topology(theta, ranks, node_ranks)
    if topology == "hccs":
        node_ranks = ranks
    elif topology == "roce":
        node_ranks = 1
    row_bytes = _frac(theta.find("row_bytes", 0.0), 0.0)
    top_k = max(1, _int(theta.find("top_k", theta.find("k", 1)), 1))
    latent = max(1, _int(theta.find("latent", 1), 1))
    wire_bytes = _frac(theta.find("wire_bytes", 0.0), 0.0)
    chunks = max(1, _int(theta.find("chunks"), 1))
    overlap = max(0, _int(theta.find("overlap"), 0))
    mtu = max(1, _int(theta.find("mtu_bytes"), DEFAULT_MTU))
    placement = str(theta.find("source_placement", "unknown") or "unknown").lower()
    if placement not in {"equal", "ragged", "unknown"}:
        raise ModelingError(f"moe_source_placement_invalid:{placement}")
    leg = str(theta.find("exchange_leg", "") or "").lower()
    if leg and leg not in {"payload", "scale", "return"}:
        raise ModelingError(f"moe_exchange_leg_invalid:{leg}")
    # Imported lazily: the exchange module calls back into this file's
    # volume helper, so a module-level import would cycle.
    from .moe_unfused_exchange import (
        exchange_widths, known_local_bounds, marginal_exchange_bound,
        marginal_leg_volumes, routing_leg_matrix,
    )
    leg_width = {"payload": "payload_row_bytes", "scale": "scale_row_bytes",
                 "return": "return_row_bytes"}
    routing_raw = theta.find("routing_matrix", None)
    byte_routing = None
    if routing_raw is not None:
        # Row counts only.  A missing or non-row unit must not be read as
        # the older byte traffic_matrix.
        unit = str(theta.find("routing_unit", "") or "")
        if unit != "rows":
            raise ModelingError(f"moe_routing_unit_invalid:{unit or 'missing'}")
        if not leg:
            raise ModelingError("moe_exchange_leg_required")
        widths = exchange_widths(
            theta.find("payload_row_bytes", None),
            theta.find("scale_row_bytes", None),
            theta.find("return_row_bytes", None))
        byte_routing = routing_leg_matrix(routing_raw, leg=leg, **widths)
        if len(byte_routing) != ranks:
            raise ModelingError("collective_traffic_matrix_rank_count_mismatch")
    # Read an explicit byte matrix before ragged placement can fail.  The
    # cells stay in the units they were declared in.
    matrix = ((byte_routing, False) if byte_routing is not None
              else _traffic_matrix(theta))
    if matrix is not None and len(matrix[0]) != ranks:
        raise ModelingError("collective_traffic_matrix_rank_count_mismatch")
    # Accept either one latent-width row or a row already expanded by top-k;
    # normalize to one routed row and then apply top-k exactly once.
    declared_row = as_fraction(latent) * wire_bytes
    single_row = (declared_row
                  if declared_row > 0 and row_bytes in (declared_row,
                                                        declared_row * top_k)
                  else row_bytes) or declared_row
    if byte_routing is not None or (
            leg and matrix is None and placement == "equal"):
        widths = exchange_widths(
            theta.find("payload_row_bytes", None),
            theta.find("scale_row_bytes", None),
            theta.find("return_row_bytes", None))
        single_row = widths[leg_width[leg]]
    rows = Mul((Var("T"), Const(top_k)))
    message = Mul((rows, Const(single_row)))
    # A prefill F2 may be lifted to the EP cohort, but equal placement is a
    # declared source fact, not an assumption from the request count.  Ragged
    # or unknown placement must not be silently averaged into a uniform split.
    phase_prefill = Var("phase_prefill")
    if placement == "equal":
        source_partitions = Add((
            Mul((phase_prefill, Max((Const(1), Min((Const(ranks), Var("EP_B"))))))),
            Mul((Sub(Const(1), phase_prefill), Const(ranks))),
        ))
    else:
        # No declared source placement/capacity: retain the pre-PCOM1 rank
        # denominator.  PCOM1's active-cohort guess was independently rejected
        # because cohort_batch counts requests, not distinct nonempty ranks.
        source_partitions = Const(ranks)
    if (placement == "equal" and byte_routing is None and matrix is None
            and theta.find("source_rows", None) is not None):
        raise ModelingError("moe_equal_placement_conflicts_with_source_rows")
    if (placement != "ragged"
            and theta.find("return_rows", None) is not None):
        raise ModelingError("moe_return_rows_requires_ragged")
    # A declared stage-to-stage KV multicast is one source publishing to
    # several remote consumers, not an EP routed all-to-all that partitions
    # this rank's routed rows.  Keep its existing directed-cell declaration.
    p2p_transfer = (str(theta.find("transfer", "") or "").lower() == "p2p"
                    or theta.find("destination_stages", None) is not None)
    bound = None
    local_payload = Const(0)
    if matrix is not None:
        intra, inter, send, hotspot, aggregate, hbm = _transport_matrix_volumes(
            matrix[0], node_ranks, mtu, matrix[1], Var("T"))
        if byte_routing is not None:
            # The route is an absolute row declaration, not a per-token rate.
            message = Const(sum(sum(row) for row in byte_routing))
            rows = Const(sum(int(value) for line in routing_raw for value in line))
    elif p2p_transfer:
        intra, inter, send, hotspot, aggregate = _uniform_alltoallv(
            message, ranks, node_ranks, mtu)
        hbm = Add((send, hotspot))
    elif placement == "ragged":
        source_rows = theta.find("source_rows", None)
        if source_rows is None:
            raise ModelingError("moe_source_placement_ragged_requires_source_rows")
        if len(tuple(source_rows)) != ranks:
            raise ModelingError("moe_source_rows_rank_count_mismatch")
        return_raw = theta.find("return_rows", None)
        return_rows = None if return_raw is None else tuple(return_raw)
        if return_rows is not None and len(return_rows) != ranks:
            raise ModelingError("moe_return_rows_rank_count_mismatch")
        received_rows = theta.find("received_rows", None)
        if received_rows is not None and len(tuple(received_rows)) != ranks:
            raise ModelingError("moe_received_rows_rank_count_mismatch")
        widths = exchange_widths(
            theta.find("payload_row_bytes", None),
            theta.find("scale_row_bytes", None),
            theta.find("return_row_bytes", None))
        source_vector = tuple(source_rows)
        received_vector = (None if received_rows is None
                           else tuple(received_rows))
        bound = known_local_bounds(source_vector, received_vector, **widths)
        marginal = marginal_exchange_bound(source_vector, received_vector)
        # The unpacked BF16 return is one row per token, while routed payload
        # and scale rows are token x top-k.  A declared return_rows vector is
        # therefore the return leg's own ledger; without it the EP6 ragged
        # route keeps its previous routed-row semantics.
        if leg == "return" and return_rows is not None:
            leg_bound = known_local_bounds(return_rows, None, **widths)
            leg_marginal = marginal_exchange_bound(return_rows, None)
        else:
            leg_bound = bound
            leg_marginal = marginal
        volumes = marginal_leg_volumes(leg_marginal, leg_bound, widths, leg)
        if leg:
            single_row = widths[leg_width[leg]]
        intra = Const(volumes["intra_bytes"])
        inter = Const(volumes["inter_bytes"])
        send = Const(volumes["send_bytes"])
        hotspot = Const(volumes["recv_hotspot"])
        aggregate = Const(volumes["aggregate_wire_bytes"])
        hbm = Const(volumes["hbm_bytes"])
        message = Const(volumes["message_bytes"])
        rows = Const(leg_bound["known_source_rows"])
        source_partitions = Const(leg_bound["known_source_ranks"])
        local_payload = Const(
            leg_bound["return_bytes"] if leg == "return"
            else leg_bound["payload_bytes"])
    else:
        if leg and placement != "equal":
            raise ModelingError("moe_exchange_leg_requires_route")
        local_payload = Div(message, source_partitions)
        intra, inter, send, hotspot, _, hbm = _routed_alltoallv(
            local_payload, ranks, node_ranks, mtu)
        # Every active source contributes one send ledger.  The group wire
        # total is active_sources * send, not all ranks * send; the latter
        # inflates sparse/unequal placement by the inactive-rank count.
        aggregate = Mul((source_partitions, send))
    mt = max(1, _int(theta.find("mt", ctx.policy.mt), ctx.policy.mt))
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("transport_tiles", "tokens", mt),
                        "max_tokens"),)

    # The dependency model must see the actual enabled directed traffic, not
    # only a global peer count.  The cells are static in theta, so this is
    # O(active cells) at compile time and no per-evaluation all-pairs scan.
    active_cells = (() if matrix is None else tuple(
        (int(source), int(destination))
        for source, row in enumerate(matrix[0])
        for destination, value in enumerate(row)
        if source != destination and int(value) > 0))

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        values = {"transport_ranks": ranks, "transport_chunks": chunks,
                  "transport_overlap": overlap,
                  "phase_prefill": 1 if geometry.phase_name == "prefill" else 0}
        if matrix is not None:
            values["transport_active_cells"] = active_cells
        values.update(_moe_comm_bindings(theta, geometry))
        return values

    intra_pipe, inter_pipe = _link_pipes(ctx)
    stages = _chunk_stages(ctx, intra, inter, hbm, chunks, intra_pipe, inter_pipe, ranks, node_ranks)
    derived = (
        ("routed_rows", rows), ("peer_bytes", message),
        ("transport_row_bytes", Const(single_row)),
    )
    if bound is not None:
        # Known local rows are not a peer split.  A paired deficit is a
        # qualified useful-byte floor for the declared leg.  It is not an
        # exact matrix, and an empty leg does not mix the three dtypes into
        # one wire number.  Hop bytes stay zero.
        derived += (
            ("source_placement", Const(1)),
            ("directed_forecast", Const(0)),
            ("directed_lower_bound", Const(leg_marginal["peer_workload_declared"])),
            ("forecast_complete", Const(0)),
            ("peer_workload_declared", Const(leg_marginal["peer_workload_declared"])),
            ("diagonal_unknown", Const(1)),
            ("outbound_rows_lower", Const(leg_marginal["outbound_rows_lower"])),
            ("inbound_rows_lower", Const(leg_marginal["inbound_rows_lower"])),
            ("directed_rows_lower", Const(leg_marginal["directed_rows_lower"])),
            ("diagonal_upper_rows", Const(leg_marginal["diagonal_upper_rows"])),
            ("source_partitions", source_partitions),
            ("source_routed_rows", rows),
            ("source_payload_bytes", local_payload),
            ("local_scale_bytes", Const(leg_bound["scale_bytes"])),
            ("local_return_bytes", Const(leg_bound["return_bytes"])),
            ("known_source_payload_max", Const(leg_bound["payload_max"])),
            ("known_source_scale_max", Const(leg_bound["scale_max"])),
            ("known_source_return_max", Const(leg_bound["return_max"])),
            ("known_received_rows", Const(bound["known_received_rows"])),
            ("known_received_payload_bytes", Const(bound["received_payload_bytes"])),
            ("known_received_payload_max", Const(bound["received_payload_max"])),
            ("received_rows_declared", Const(bound["received_declared"])),
            ("per_peer_useful_bytes", Const(0)),
        )
    elif matrix is None and not p2p_transfer:
        # Uniform routed split trace.  An explicit directed matrix already
        # exposes the real per-source cells, and a P2P multicast publishes one
        # declared payload rather than partitioning routed rows.
        derived += (
            ("source_placement", Const(0 if placement == "unknown" else 1)),
            ("source_partitions", source_partitions),
            ("source_routed_rows", Div(rows, source_partitions)),
            ("source_payload_bytes", local_payload),
            ("per_peer_useful_bytes",
             (Div(local_payload, Const(ranks)) if ranks > 1 else Const(0))),
        )
        if leg and placement == "equal":
            derived += (("directed_forecast", Const(1)),)
    elif byte_routing is not None or (
            matrix is not None and placement == "ragged"):
        # Row routing and an explicit ragged byte matrix are directed
        # forecasts.  An older equal or unknown byte matrix keeps its
        # previous derived keys and its original cell units.
        derived += (("directed_forecast", Const(1)),)
    derived += (
        ("send_bytes", send), ("recv_hotspot", hotspot),
        ("aggregate_wire_bytes", aggregate),
        ("intra_bytes", intra), ("inter_bytes", inter),
        ("chunks", Const(chunks)), ("overlap", Const(overlap)),
        ("latent", Const(latent)), ("top_k", Const(top_k)),
        ("ranks", Const(ranks)), ("nodes", Const(math.ceil(ranks / node_ranks))),
        ("ranks_per_node", Const(node_ranks)), ("mtu_bytes", Const(mtu)),
        ("local_hbm_bytes", hbm), ("cohort_batch", Var("EP_B")),
    )
    return FamilyTemplate(
        axes=axes, stages=stages,
        materialization=(("peer_traffic", message),), internal=(),
        residency=Const(0), derived=derived, regime_exprs=(), binder=binder,
        active=_moe_comm_active(theta),
        annotations=(("family", "transport"), ("collective", kind),
                     ("topology", topology)),
    )
