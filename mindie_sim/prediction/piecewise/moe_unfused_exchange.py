"""Private directed MoE unfused-exchange ledgers.

[Role]
  Price explicitly declared source-to-destination routed cells for the three
  first-party unfused AllToAllV exchanges: FP8 payload, its E8M0 scale, and
  the BF16 return.  A caller supplies the directed cells; source/receive
  marginals alone are never treated as a unique matrix.

[Invariants]
  * A cell is useful bytes from one source rank to one destination; the
    diagonal self share is not a peer cell.
  * Peer wire bytes are MTU-framed per directed cell; per-rank send/receive
    are independent marginals and the maximum is a bottleneck, never a sum.
  * Group wire is the sum of active source cells, not ranks x a uniform send.
  * Payload (FP8), its separate E8M0 scale, and the BF16 return are distinct
    ledgers.  A payload leg does not absorb the scale bytes.
  * Paired source and received marginals qualify a remote-row floor.  The
    floor is the larger of the egress-deficit sum and the ingress-deficit
    sum.  It is not an exact matrix, and the two sums are not added.
  * Empty ranks contribute zero; non-EP has no peer wire.
  * No duration, model name, phase label, or observed timing is read.

[Timeouts & Locks]
  Bounded O(active cells + ranks) arithmetic; no I/O, locks or device.

[Side-Effect Policy]
  Returns immutable receipt records only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from mindie_sim.schema.common import ModelingError
from .expressions import Const
from .families_moe_transport import _transport_matrix_volumes


@dataclass(frozen=True)
class DirectedCell:
    source: int
    destination: int
    useful_bytes: int


@dataclass(frozen=True)
class ExchangeLedger:
    name: str
    dtype: str
    total_bytes: int
    remote_bytes: int
    self_bytes: int
    send_max: int
    receive_max: int
    aggregate_wire_bytes: int
    intra_bytes: int
    inter_bytes: int
    hbm_bytes: int


@dataclass(frozen=True)
class ThreeExchangeReceipt:
    payload: ExchangeLedger
    scale: ExchangeLedger
    return_: ExchangeLedger

    @property
    def aggregate_wire_bytes(self):
        return (self.payload.aggregate_wire_bytes
                + self.scale.aggregate_wire_bytes
                + self.return_.aggregate_wire_bytes)


def _rank_matrix(cells, ranks):
    if ranks < 1:
        raise ModelingError("moe_unfused_ranks_invalid")
    matrix = [[0 for _ in range(ranks)] for _ in range(ranks)]
    for cell in cells:
        if not (0 <= cell.source < ranks and 0 <= cell.destination < ranks):
            raise ModelingError("moe_unfused_cell_rank_out_of_range")
        if cell.source == cell.destination:
            raise ModelingError("moe_unfused_self_cell_requires_explicit_stage")
        if cell.useful_bytes < 0:
            raise ModelingError("moe_unfused_cell_bytes_invalid")
        matrix[cell.source][cell.destination] += int(cell.useful_bytes)
    return tuple(tuple(row) for row in matrix)


def _ledger(name, dtype, cells, *, ranks, node_ranks, mtu, total_bytes):
    matrix = _rank_matrix(cells, ranks)
    framed = tuple(tuple((math.ceil(value / mtu) * mtu if value else 0)
                         for value in row) for row in matrix)
    sends = [sum(row) for row in framed]
    receives = [sum(framed[source][destination] for source in range(ranks))
                for destination in range(ranks)]
    useful_remote = sum(sum(row) for row in matrix)
    if total_bytes < useful_remote:
        raise ModelingError(
            f"moe_unfused_remote_exceeds_total:{name}:{useful_remote}>{total_bytes}")
    intra = inter = 0
    width = max(1, min(ranks, int(node_ranks)))
    for source in range(ranks):
        for destination in range(ranks):
            if source == destination:
                continue
            value = framed[source][destination]
            if source // width == destination // width:
                intra += value
            else:
                inter += value
    helper = _transport_matrix_volumes(
        matrix, node_ranks, mtu, False, Const(1))
    aggregate = int(helper[4].evaluate({}))
    helper_send = int(helper[2].evaluate({}))
    helper_recv = int(helper[3].evaluate({}))
    hbm = int(helper[5].evaluate({}))
    if aggregate != sum(sends) or helper_send != max(sends, default=0):
        raise ModelingError(f"moe_unfused_helper_disagrees:{name}")
    if helper_recv != max(receives, default=0):
        raise ModelingError(f"moe_unfused_helper_receive_disagrees:{name}")
    return ExchangeLedger(
        name=name, dtype=dtype, total_bytes=int(total_bytes),
        remote_bytes=useful_remote, self_bytes=int(total_bytes) - useful_remote,
        send_max=max(sends, default=0), receive_max=max(receives, default=0),
        aggregate_wire_bytes=aggregate, intra_bytes=intra, inter_bytes=inter,
        hbm_bytes=hbm,
    )


def _uniform_cells(ranks, rows, row_bytes, *, node_ranks):
    """Equal source rows split over every destination, self excluded."""
    if ranks <= 1:
        return ()
    if rows % ranks:
        raise ModelingError("moe_unfused_uniform_rows_not_divisible")
    per_source_rows = rows // ranks
    per_peer = per_source_rows * row_bytes // ranks
    if per_peer * ranks != per_source_rows * row_bytes:
        raise ModelingError("moe_unfused_uniform_cell_not_integral")
    return tuple(DirectedCell(source, destination, per_peer)
                 for source in range(ranks)
                 for destination in range(ranks) if source != destination)


def uniform_three_exchange(*, ranks, rows, payload_row_bytes, scale_row_bytes,
                           return_row_bytes, node_ranks, mtu):
    """Bounded uniform producer used by tests and callers with equal placement."""
    payload_cells = _uniform_cells(ranks, rows, payload_row_bytes,
                                   node_ranks=node_ranks)
    scale_cells = _uniform_cells(ranks, rows, scale_row_bytes,
                                 node_ranks=node_ranks)
    return_cells = _uniform_cells(ranks, rows, return_row_bytes,
                                  node_ranks=node_ranks)
    return price_three_exchange(
        payload_cells, scale_cells, return_cells,
        ranks=ranks, node_ranks=node_ranks, mtu=mtu,
        payload_total=rows * payload_row_bytes,
        scale_total=rows * scale_row_bytes,
        return_total=rows * return_row_bytes,
    )


def exchange_widths(payload_row_bytes, scale_row_bytes, return_row_bytes):
    """Integral useful bytes per routed row for the three typed exchanges."""

    def one(value, name):
        if value is None:
            raise ModelingError(f"moe_exchange_widths_required:{name}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelingError(f"moe_exchange_width_invalid:{name}")
        return int(value)

    return {
        "payload_row_bytes": one(payload_row_bytes, "payload_row_bytes"),
        "scale_row_bytes": one(scale_row_bytes, "scale_row_bytes"),
        "return_row_bytes": one(return_row_bytes, "return_row_bytes"),
    }


def _row_vector(rows, label):
    """Absolute routed/received rows. ``None`` is an unknown rank, not zero."""
    if rows is None:
        return None
    parsed = []
    for value in rows:
        if value is None:
            parsed.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelingError(f"moe_{label}_invalid")
        parsed.append(int(value))
    if not any(value is not None for value in parsed):
        raise ModelingError(f"moe_{label}_requires_a_known_rank")
    return tuple(parsed)


def known_local_bounds(source_rows, received_rows, *, payload_row_bytes,
                       scale_row_bytes, return_row_bytes):
    """Typed local bytes for ranks whose rows are known and whose peers are not.

    Unknown ranks are omitted.  A known zero is an empty rank.  The sums are
    a lower bound on declared local material, not a directed transfer.
    """
    widths = exchange_widths(payload_row_bytes, scale_row_bytes, return_row_bytes)
    source = _row_vector(source_rows, "source_rows")
    received = (_row_vector(received_rows, "received_rows")
                if received_rows is not None else None)

    def total(rows, width):
        if not rows:
            return 0
        return sum(int(value) * width for value in rows if value is not None)

    def peak(rows, width):
        if not rows:
            return 0
        known = [int(value) * width for value in rows if value is not None]
        return max(known, default=0)

    def count(rows):
        if not rows:
            return 0
        return sum(1 for value in rows if value is not None)

    payload = widths["payload_row_bytes"]
    scale = widths["scale_row_bytes"]
    ret = widths["return_row_bytes"]
    return {
        "known_source_rows": sum(value for value in source if value is not None),
        "known_source_ranks": count(source),
        "known_received_rows": (
            sum(value for value in received if value is not None) if received else 0),
        "known_received_ranks": count(received),
        "received_declared": 0 if received is None else 1,
        "payload_bytes": total(source, payload),
        "scale_bytes": total(source, scale),
        "return_bytes": total(source, ret),
        "payload_max": peak(source, payload),
        "scale_max": peak(source, scale),
        "return_max": peak(source, ret),
        "received_payload_bytes": total(received, payload),
        "received_payload_max": peak(received, payload),
    }


def bound_material_bytes(bounds, leg):
    """Known source bytes of one typed leg.

    Payload, scale, and return are separate buffers.  An operator with no
    leg retains the sum of the three source buffers and is not a wire total.
    """
    if leg == "return":
        return int(bounds["return_bytes"])
    if leg == "scale":
        return int(bounds["scale_bytes"])
    if leg == "payload":
        return int(bounds["payload_bytes"])
    if leg:
        raise ModelingError(f"moe_exchange_leg_invalid:{leg}")
    return (int(bounds["payload_bytes"]) + int(bounds["scale_bytes"])
            + int(bounds["return_bytes"]))


def _integral_byte_count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    number = int(value)
    if number < 0:
        return None
    return number


def proven_exchange_widths(spec):
    """Integral FP8, E8M0, and BF16 row widths from a native dispatch spec.

    The spec qualifies only when the producer already proved a quantized
    payload, a separate E8M0 scale, and nonnegative integral byte widths.
    """
    getter = getattr(spec, "get", None)
    payload = scale = returned = None
    scale_dtype = ""
    quantized = False
    if callable(getter):
        payload = _integral_byte_count(getter("payload_bytes_per_row"))
        scale = _integral_byte_count(getter("scale_bytes_per_row"))
        returned = _integral_byte_count(getter("combine_row_bytes"))
        scale_dtype = str(getter("scale_dtype") or "")
        quantized = bool(getter("quantized"))
    if not (quantized and scale_dtype in {"fp8_e8m0", "e8m0"}
            and scale is not None and scale > 0
            and payload is not None and returned is not None):
        raise ModelingError("moe_exchange_widths_required:payload_row_bytes")
    return {
        "payload_row_bytes": payload,
        "scale_row_bytes": scale,
        "return_row_bytes": returned,
    }


def marginal_exchange_bound(source_rows, received_rows):
    """Conservation floor from paired source and received marginals.

    A rank contributes only when both marginals are known.  Its outbound
    deficit is ``max(0, source - received)`` and its inbound deficit is
    ``max(0, received - source)``.  The unknown diagonal is at most
    ``min(source, received)`` and is not a wire cell.  Group remote rows are
    the larger deficit sum, because one directed edge can appear in both an
    egress marginal and an ingress marginal.  A missing received vector
    declares no peer workload, so source rows are not treated as egress.
    """
    source = _row_vector(source_rows, "source_rows")
    received = (_row_vector(received_rows, "received_rows")
                if received_rows is not None else None)
    if received is not None and len(received) != len(source):
        raise ModelingError("moe_received_rows_rank_count_mismatch")
    outbound = []
    inbound = []
    diagonal_upper = []
    if received is not None:
        for src, recv in zip(source, received):
            if src is None or recv is None:
                continue
            outbound.append(max(0, src - recv))
            inbound.append(max(0, recv - src))
            diagonal_upper.append(min(src, recv))
    outbound_sum = sum(outbound)
    inbound_sum = sum(inbound)
    return {
        "peer_workload_declared": 0 if received is None else 1,
        "outbound_rows_lower": outbound_sum,
        "inbound_rows_lower": inbound_sum,
        "directed_rows_lower": (
            max(outbound_sum, inbound_sum) if received is not None else 0),
        "outbound_rows_max": max(outbound, default=0),
        "inbound_rows_max": max(inbound, default=0),
        "diagonal_upper_rows": sum(diagonal_upper),
        "source_vector": source,
        "received_vector": received,
    }


def marginal_rank_hbm(source, received, width):
    """Busiest known rank's partner-independent source and receive buffers.

    The two tensors are distinct, so the unknown diagonal is not subtracted.
    A rank whose receive marginal is unknown contributes its source buffer
    only.  The result is a maximum, not a sum across ranks.
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 0:
        raise ModelingError("moe_exchange_width_invalid:leg")
    received_rows = (None,) * len(source) if received is None else received
    if len(received_rows) != len(source):
        raise ModelingError("moe_received_rows_rank_count_mismatch")
    peaks = []
    for src, recv in zip(source, received_rows):
        if src is None and recv is None:
            continue
        total = (0 if src is None else int(src)) + (0 if recv is None else int(recv))
        peaks.append(total * int(width))
    return max(peaks, default=0)


def marginal_leg_volumes(marginal, bounds, widths, leg):
    """Qualified volumes for one typed leg, or an untyped local record.

    An empty leg keeps the three source buffers in one HBM record and does
    not publish a mixed-dtype wire.  A declared leg multiplies the remote-row
    floor by that leg's width.  The return leg swaps the egress and ingress
    hotspots.  Hop placement is unknown, so intra-node and inter-node bytes
    stay zero.  These floors are not an exact forecast and are not MTU frames.
    """
    if leg and leg not in {"payload", "scale", "return"}:
        raise ModelingError(f"moe_exchange_leg_invalid:{leg}")
    message = bound_material_bytes(bounds, leg)
    if not leg:
        return {
            "intra_bytes": 0, "inter_bytes": 0, "send_bytes": 0,
            "recv_hotspot": 0, "aggregate_wire_bytes": 0,
            "hbm_bytes": message, "message_bytes": message,
        }
    width = widths[{"payload": "payload_row_bytes", "scale": "scale_row_bytes",
                    "return": "return_row_bytes"}[leg]]
    hbm = marginal_rank_hbm(
        marginal["source_vector"], marginal["received_vector"], width)
    send = recv = aggregate = 0
    if marginal["peer_workload_declared"]:
        egress = int(marginal["outbound_rows_max"])
        ingress = int(marginal["inbound_rows_max"])
        if leg == "return":
            send, recv = ingress * width, egress * width
        else:
            send, recv = egress * width, ingress * width
        aggregate = int(marginal["directed_rows_lower"]) * width
    return {
        "intra_bytes": 0, "inter_bytes": 0, "send_bytes": send,
        "recv_hotspot": recv, "aggregate_wire_bytes": aggregate,
        "hbm_bytes": hbm, "message_bytes": message,
    }


def routing_leg_matrix(routing, *, leg, payload_row_bytes, scale_row_bytes,
                       return_row_bytes):
    """One row-count route expanded into one exchange's useful byte matrix.

    ``routing[i][j]`` is routed rows from source ``i`` to destination ``j``.
    The diagonal is local and is not a peer cell.  Payload and scale keep
    that direction.  The return leg reverses each off-diagonal cell and
    applies the BF16 width.  Callers that already hold bytes use
    ``traffic_matrix`` instead of this function.
    """
    if leg not in {"payload", "scale", "return"}:
        raise ModelingError(f"moe_exchange_leg_invalid:{leg}")
    widths = exchange_widths(payload_row_bytes, scale_row_bytes, return_row_bytes)
    width = widths[{"payload": "payload_row_bytes", "scale": "scale_row_bytes",
                    "return": "return_row_bytes"}[leg]]
    rows = tuple(routing or ())
    ranks = len(rows)
    if ranks < 1 or any(not isinstance(row, (tuple, list)) or len(row) != ranks
                        for row in rows):
        raise ModelingError("moe_routing_matrix_requires_square_rank_cells")
    out = [[0 for _ in range(ranks)] for _ in range(ranks)]
    for source, row in enumerate(rows):
        for destination, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError("moe_routing_cell_invalid")
            if source == destination or value == 0:
                continue
            if leg == "return":
                out[destination][source] += int(value) * width
            else:
                out[source][destination] += int(value) * width
    return tuple(tuple(row) for row in out)


def exchanges_from_routing(routing, *, payload_row_bytes, scale_row_bytes,
                           return_row_bytes, node_ranks, mtu):
    """Price all three legs of one row route through ``price_three_exchange``."""
    widths = exchange_widths(payload_row_bytes, scale_row_bytes, return_row_bytes)
    matrices = {
        name: routing_leg_matrix(routing, leg=name, **widths)
        for name in ("payload", "scale", "return")
    }
    ranks = len(matrices["payload"])

    def cells(matrix):
        return tuple(
            DirectedCell(source, destination, int(value))
            for source, row in enumerate(matrix)
            for destination, value in enumerate(row)
            if source != destination and int(value) > 0)

    def mass(width):
        return sum(int(value) * width for row in routing for value in row)

    return price_three_exchange(
        cells(matrices["payload"]), cells(matrices["scale"]),
        cells(matrices["return"]),
        ranks=ranks, node_ranks=node_ranks, mtu=mtu,
        payload_total=mass(widths["payload_row_bytes"]),
        scale_total=mass(widths["scale_row_bytes"]),
        return_total=mass(widths["return_row_bytes"]),
    )


def price_three_exchange(payload_cells, scale_cells, return_cells, *,
                         ranks: int, node_ranks: int, mtu: int,
                         payload_total: int, scale_total: int,
                         return_total: int) -> ThreeExchangeReceipt:
    """Price explicit payload/scale/return cells with conserved masses."""
    if mtu < 1:
        raise ModelingError("moe_unfused_mtu_invalid")
    if payload_total < 0 or scale_total < 0 or return_total < 0:
        raise ModelingError("moe_unfused_total_invalid")
    return ThreeExchangeReceipt(
        payload=_ledger("payload", "fp8_e4m3", payload_cells,
                        ranks=ranks, node_ranks=node_ranks, mtu=mtu,
                        total_bytes=payload_total),
        scale=_ledger("scale", "e8m0_scale", scale_cells,
                      ranks=ranks, node_ranks=node_ranks, mtu=mtu,
                      total_bytes=scale_total),
        return_=_ledger("return", "bf16", return_cells,
                        ranks=ranks, node_ranks=node_ranks, mtu=mtu,
                        total_bytes=return_total),
    )


_SOURCE_MC2_TOKENS_PER_RANK_LIMIT = 512


def mc2_tokens_capacity(*, max_num_reqs: int, uniform_decode_query_len: int,
                        capture_sizes, tp: int,
                        enable_prefill_mc2: bool = False,
                        max_num_batched_tokens: int | None = None,
                        enable_fused_mc2: bool = True) -> int:
    """Exact source-capacity reduction from the running config.

    Mirrors vLLM-ascend set_mc2_tokens_capacity: prefill-MC2 uses the batched
    token budget, graph capture uses the largest capture size, otherwise
    requests x decode query length.  The 512 value is the source per-rank
    limit, not a generic input default.
    """
    if tp < 1:
        raise ModelingError("mc2_tokens_capacity_tp_invalid")
    if max_num_reqs < 0 or uniform_decode_query_len < 0:
        raise ModelingError("mc2_tokens_capacity_request_invalid")
    if enable_prefill_mc2:
        if max_num_batched_tokens is None or max_num_batched_tokens < 1:
            raise ModelingError("mc2_tokens_capacity_prefill_budget_missing")
        max_tokens = int(max_num_batched_tokens)
    elif capture_sizes:
        sizes = tuple(int(value) for value in capture_sizes)
        if any(value < 1 for value in sizes):
            raise ModelingError("mc2_tokens_capacity_capture_invalid")
        max_tokens = max(sizes)
    else:
        max_tokens = max_num_reqs * uniform_decode_query_len
    per_rank = (max_tokens + tp - 1) // tp
    per_rank = min(per_rank, _SOURCE_MC2_TOKENS_PER_RANK_LIMIT)
    return per_rank * tp


def active_source_geometry(requests) -> dict:
    """Pre-expansion rank placement: group total is not active-source count.

    ``requests`` is an iterable of ``(rank, tokens)``.  Two requests on one
    rank have active_sources=1 and max_tokens_per_rank=2*tokens; two requests
    on distinct ranks have active_sources=2 and max_tokens_per_rank=tokens.
    """
    by_rank: dict[int, int] = {}
    total = 0
    for rank, tokens in requests:
        if int(rank) < 0 or int(tokens) < 0:
            raise ModelingError("active_source_geometry_invalid")
        by_rank[int(rank)] = by_rank.get(int(rank), 0) + int(tokens)
        total += int(tokens)
    if not by_rank:
        return {"group_total_tokens": 0, "active_sources": 0,
                "max_tokens_per_rank": 0, "by_rank": {}}
    return {"group_total_tokens": total, "active_sources": len(by_rank),
            "max_tokens_per_rank": max(by_rank.values()),
            "by_rank": dict(sorted(by_rank.items()))}
