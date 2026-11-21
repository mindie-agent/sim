"""Payload collectives for explicitly selected MoE transport.

[Role] Build forward payload and scale exchanges and the reverse BF16 combine
  from the selected row format and optional exact peer ledger.
[Invariants] Peer traffic excludes local copies. Return traffic transposes
  the forward assignment ledger before applying the return row width.
[Timeouts & Locks] Ledger work is bounded by the supplied peer matrix.
  Construction uses no I/O or locks.
[Side-Effect Policy] Builds operator specifications and their local metadata
  without mutating the inputs or starting device work.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from mindie_sim.catalog.ops import comm_op
from mindie_sim.schema.common import ModelingError, dtype_bytes


_EXPERT_ROW_FORMAT = "peer_traffic_row_format"


def _ceil_rows(rows):
    return int(math.ceil(rows - 1e-12))


def _routing_peer_counts(op):
    """Compact integer source->destination assignment matrix from one op.

    The peer matrix is deliberately kept separate from the full routing
    realization (which may carry every token identity).  Only a route with an
    explicit source/destination ledger publishes it; expected moments keep
    their existing uniform declaration.
    """
    metadata = dict(getattr(op, "metadata", ()) or ())
    cells = metadata.get("moe_peer_counts")
    rank = metadata.get("moe_peer_rank")
    basis = metadata.get("moe_peer_basis", metadata.get("routing_basis", ""))
    if cells is None:
        raw = metadata.get("routing_realization")
        if not isinstance(raw, Mapping):
            return None
        cells = raw.get("peer_assignments", raw.get("peer_assignment_rows"))
        rank = raw.get("representative_rank")
        basis = raw.get("basis", metadata.get("routing_basis", ""))
        if cells is None:
            return None
    try:
        matrix = tuple(tuple(int(value) for value in row) for row in cells)
    except (TypeError, ValueError) as exc:
        raise ModelingError("moe_transport_peer_counts_must_be_integer_matrix") from exc
    size = len(matrix)
    if (size < 1 or any(len(row) != size for row in matrix)
            or any(type(value) is bool or value < 0 for row in matrix for value in row)):
        raise ModelingError("moe_transport_peer_counts_must_be_nonnegative_square_matrix")
    if type(rank) is not int or not 0 <= rank < size:
        raise ModelingError("moe_transport_peer_counts_require_valid_representative_rank")
    return matrix, rank, str(basis)


def _peer_payload(cells, row_bytes):
    """Directed bytes for one representative source and its group mean."""
    size = len(cells)
    matrix = tuple(tuple(0 if source == destination else
                         float(row_bytes) * cells[source][destination]
                         for destination in range(size))
                   for source in range(size))
    total = math.fsum(value for row in matrix for value in row)
    return matrix, total / size


def _peer_metadata(op, row_bytes):
    routing = _routing_peer_counts(op)
    if routing is None:
        return (), None
    cells, rank, basis = routing
    matrix, link_bytes = _peer_payload(cells, row_bytes)
    return (
        ("peer_volume_basis", "group_mean"),
        ("peer_traffic", {"kind": "matrix", "payload_bytes": matrix}),
        ("routing_basis", basis),
        ("traffic_model", basis),
        ("peer_source_rank", rank),
    ), link_bytes


class _PeerRouteId(str):
    """Payload id carrying the compact route ledger to its inverse operator.

    The ordinary builder passes only op ids across the dispatch/combine
    boundary.  An explicit source route already owns the exact integer matrix,
    so that pairing carries it as an immutable attribute of the same string
    identity.  No global registry, history or module-level mutable state is
    involved; a caller that constructs combine directly still receives the
    old id-only contract.
    """

    def __new__(cls, op_id, cells, rank, basis):
        value = super().__new__(cls, op_id)
        value.cells = cells
        value.rank = rank
        value.basis = basis
        return value


def dispatch(*, prefix, plan, stage, requests, quantized, transport, rows, width, ep,
             partitions, mode, hist_gather=None):
    """Block/row payload exchange, with row scales split out when declared."""
    ops = []
    if ep <= 1:
        return ops, None, None
    routing = _routing_peer_counts(quantized)
    common = (("group_axis", "ep"),
              ("source_assignment_partitions", partitions),
              ("remote_fraction", (ep - 1) / ep),
              ("traffic_model", "uniform_peer_payload"),
              ("source_rows", rows),
              ("scale_dtype", transport.scale_dtype),
              ("eplb_peak_to_avg", transport.eplb_peak_to_avg),
              ("dispatch_dtype", transport.dispatch_dtype))
    scale = None
    if transport.quantized and not transport.inline_scales:
        scale_row_bytes = dtype_bytes(transport.scale_dtype)
        scale_peer, scale_link = _peer_metadata(quantized, scale_row_bytes)
        scale = comm_op(
            op_id=prefix + ".moe_scale_dispatch",
            deps=tuple(dict.fromkeys((quantized.op_id,) + ((hist_gather,) if hist_gather else ()))),
            stage=stage, layer=plan.index, requests=requests,
            family="alltoall", phase=mode, recipe="expert_scale_dispatch",
            link_bytes=(scale_link if scale_link is not None
                        else rows * scale_row_bytes * (ep - 1) / ep),
            shape=(_ceil_rows(rows),),
            metadata=common + (("collective_id", prefix + ".moe_scale_dispatch"),
                               ("payload_kind", "row_scale"),
                               ("payload_dtype", transport.scale_dtype),
                               ("row_bytes", scale_row_bytes),
                               *scale_peer),
        )
        ops.append(scale)
    payload_row_bytes = transport.row_bytes(width)
    if transport.inline_scales:
        row_format = {"forward_bytes": int(payload_row_bytes),
                      "reverse_bytes": int(width * transport.return_bytes)}
    else:
        row_format = {"elements": width,
                      "forward_dtype": transport.payload_dtype,
                      "reverse_dtype": "bf16"}
    payload_peer, payload_link = _peer_metadata(quantized, payload_row_bytes)
    payload = comm_op(
        op_id=prefix + ".moe_payload_dispatch",
        deps=tuple(dict.fromkeys((quantized.op_id,)
                                 + ((scale.op_id,) if scale else ())
                                 + ((hist_gather,) if hist_gather else ()))),
        stage=stage, layer=plan.index, requests=requests,
        family="alltoall", phase=mode, recipe="expert_quantized_dispatch",
        link_bytes=(payload_link if payload_link is not None
                    else rows * payload_row_bytes * (ep - 1) / ep),
        shape=(_ceil_rows(rows), width),
        metadata=common + (("collective_id", prefix + ".moe_payload_dispatch"),
                           ("payload_kind", "quantized_rows"),
                           ("payload_dtype", transport.payload_dtype),
                           ("row_elements", width),
                           ("row_bytes", payload_row_bytes),
                           ("payload_bytes", width * transport.payload_bytes),
                           ("inline_scale_bytes",
                            transport.scale_bytes(width) if transport.inline_scales else 0.0),
                           ("scale_group", transport.scale_group),
                           (_EXPERT_ROW_FORMAT, row_format),
                           *payload_peer),
    )
    ops.append(payload)
    # The received quantities are real output slots of this collective.  The
    # gate consumes these values; no consumer may infer dtype from the
    # quantized producer's ancestry.
    from .deepep import declare_received_value
    if scale is not None:
        replaced = declare_received_value(
            scale, dtype=transport.scale_dtype, shape=(_ceil_rows(rows),),
            row_bytes=dtype_bytes(transport.scale_dtype),
            storage=f"moe.rx:{prefix}.moe_scale_dispatch",
            binding="static", binding_name=f"{prefix}.moe_scale_dispatch.rx")
        ops = [replaced if op.op_id == scale.op_id else op for op in ops]
        scale = replaced
    replaced = declare_received_value(
        payload, dtype=transport.payload_dtype, shape=(_ceil_rows(rows), width),
        row_bytes=payload_row_bytes,
        storage=f"moe.rx:{prefix}.moe_payload_dispatch",
        binding="static", binding_name=f"{prefix}.moe_payload_dispatch.rx")
    ops = [replaced if op.op_id == payload.op_id else op for op in ops]
    payload = replaced
    payload_id = (_PeerRouteId(payload.op_id, *routing)
                  if routing is not None else payload.op_id)
    return ops, (scale.op_id if scale else None), payload_id


def combine(*, prefix, plan, stage, requests, deps, transport, rows, width, ep,
            partitions, mode, payload_dispatch):
    """BF16 return with dispatch source rows transposed before dtype conversion."""
    if ep <= 1:
        return None
    return_bytes = width * transport.return_bytes
    cells = getattr(payload_dispatch, "cells", None)
    if cells is None:
        link_bytes = rows * return_bytes * (ep - 1) / ep
        reverse = (("reverse_of_collective", payload_dispatch),)
        peer_metadata = ()
    else:
        # The dispatch id carried the exact assignment matrix.  Transpose it at
        # the return row width here, so the inverse op keeps real per-source
        # traffic instead of a group-mean scalar.  Its counterpart is already
        # an explicit forward declaration, so a second inverse binding would
        # only re-check numbers derivable from this same ledger.
        size = len(cells)
        matrix = tuple(tuple(0 if source == destination else
                             float(return_bytes) * cells[destination][source]
                             for destination in range(size))
                       for source in range(size))
        link_bytes = math.fsum(value for row in matrix for value in row) / size
        reverse = ()
        peer_metadata = (
            ("peer_volume_basis", "group_mean"),
            ("peer_traffic", {"kind": "matrix", "payload_bytes": matrix}),
            ("routing_basis", getattr(payload_dispatch, "basis", "")),
            ("traffic_model", getattr(payload_dispatch, "basis", "")),
            ("peer_source_rank", getattr(payload_dispatch, "rank", 0)),
        )
    return comm_op(
        op_id=prefix + ".expert_combine", deps=(deps,), stage=stage,
        layer=plan.index, requests=requests, family="alltoall", phase=mode,
        recipe="expert_combine",
        link_bytes=link_bytes,
        shape=(_ceil_rows(rows), width),
        metadata=(("group_axis", "ep"),
                  ("source_assignment_partitions", partitions),
                  ("traffic_model", "transpose_dispatch_peer_rows"),
                  ("collective_id", prefix + ".expert_combine"),
                  *reverse,
                  ("payload_kind", "bf16_return_rows"),
                  ("payload_dtype", "bf16"),
                  ("combine_dtype", "bf16"),
                  ("row_elements", width),
                  ("row_bytes", width * transport.return_bytes),
                  ("dispatch_dtype", transport.dispatch_dtype),
                  ("eplb_peak_to_avg", transport.eplb_peak_to_avg),
                  (_EXPERT_ROW_FORMAT, (
                      {"forward_bytes": int(transport.row_bytes(width)),
                       "reverse_bytes": int(width * transport.return_bytes)}
                      if transport.inline_scales else
                      {"elements": width,
                       "forward_dtype": transport.payload_dtype,
                       "reverse_dtype": "bf16"})),
                  *peer_metadata),
    )
