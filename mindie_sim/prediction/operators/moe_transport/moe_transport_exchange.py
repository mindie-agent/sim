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
from dataclasses import replace
import math
from typing import Literal

from mindie_sim.catalog.ops import comm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_REVISION = "175ef4d49c2072f9233619636f0309f4f88b4e2c"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-ascend",
        revision=_VLLM_ASCEND_REVISION,
        relative_path="vllm_ascend/ops/fused_moe/token_dispatcher.py",
        source_sha256="6126304014959e8c52d87284ccac078f7622a6d72c7a7fc8be25929dfe644cfb",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision=_VLLM_ASCEND_REVISION,
        relative_path="vllm_ascend/ops/fused_moe/moe_comm_method.py",
        source_sha256="b785ade12a81349d01d4806c95ab17d37537db9548b3aae492135c8f4318fe39",
    ),
)

_EXPERT_ROW_FORMAT = "peer_traffic_row_format"
_INDEX_BYTES = int(dtype_bytes("int32"))


def _ceil_rows(rows):
    return int(math.ceil(rows - 1e-12))


def _scale_shape(rows: int, width: int, *, group: int, pack: int) -> tuple[int, ...]:
    groups = max(1, math.ceil(width / max(1, int(group))))
    pack_width = max(1, int(pack))
    if pack_width > 1:
        return (int(rows), math.ceil(groups / pack_width), pack_width)
    if groups > 1:
        return (int(rows), groups)
    return (int(rows),)


def _token_route_from_op(op):
    """Extract compact exact token-level route when explicitly declared."""
    metadata = dict(getattr(op, "metadata", ()) or ())
    direct = metadata.get("moe_token_route")
    if isinstance(direct, Mapping):
        return dict(direct)
    raw = metadata.get("routing_realization")
    if not isinstance(raw, Mapping):
        return None
    owners = raw.get("source_owners")
    topk = raw.get("topk_ids")
    placement = raw.get("expert_placement")
    if owners is None or topk is None or placement is None:
        return None
    return {
        "source_owners": tuple(int(v) for v in owners),
        "topk_ids": tuple(tuple(int(e) for e in row) for row in topk),
        "expert_placement": tuple((int(r), int(b)) for r, b in placement),
    }


def _resolve_receive_rows(quantized, routing, fallback_send_rows: float,
                          receive_rows: int | None = None) -> int:
    """Resolve destination rank received rows (including diagonal local rows)."""
    if receive_rows is not None:
        return max(0, int(receive_rows))
    if routing is not None:
        cells, rank, _basis = routing
        return sum(int(cells[src][rank]) for src in range(len(cells)))
    metadata = dict(getattr(quantized, "metadata", ()) or ())
    raw = metadata.get("routing_realization")
    if isinstance(raw, Mapping):
        rank_rows = raw.get("rank_assignment_rows")
        rep = int(raw.get("representative_rank", 0) or 0)
        if rank_rows is not None and 0 <= rep < len(rank_rows):
            return int(rank_rows[rep])
        local = raw.get("local_expert_rows")
        if local is not None:
            return sum(int(v) for v in local)
    return _ceil_rows(fallback_send_rows)


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

    def __new__(cls, op_id, cells=None, rank=0, basis="", *,
                token_route=None, dispatch_op_id=None, reorder_op_id=None,
                recv_rows=None):
        value = super().__new__(cls, op_id)
        value.cells = cells
        value.rank = rank
        value.basis = basis
        value.token_route = token_route
        value.dispatch_op_id = dispatch_op_id or op_id
        value.reorder_op_id = reorder_op_id
        value.recv_rows = recv_rows
        return value


def receive_reorder(
    *, prefix, plan, stage, requests, deps, payload_id: str, width: int,
    recv_rows: int, local_experts: int, payload_dtype: str,
    payload_row_bytes: float, mode: str, scale_id: str | None = None,
    scale_dtype: str = "", scale_shape: tuple[int, ...] = (),
    scale_bytes_total: float = 0.0, scale_group: int = 0,
):
    """Post-dispatch local-expert reorder retaining the inverse map (and scales)."""
    from .deepep import declare_received_value

    reorder_payload_bytes = float(recv_rows) * width * dtype_bytes(payload_dtype)
    reorder_scale_bytes = float(scale_bytes_total) if scale_id is not None else 0.0
    reorder_index_bytes = float(recv_rows * _INDEX_BYTES)
    reorder_io = reorder_payload_bytes + reorder_scale_bytes + reorder_index_bytes
    returns = (
        ("inverse_reorder_map", "activation_scales")
        if scale_id is not None else ("inverse_reorder_map",)
    )
    scale_meta = (
        (("activation_scale_format", scale_dtype),
         ("activation_scale_shape", scale_shape),
         ("activation_scale_bytes", reorder_scale_bytes),
         ("activation_group_size", scale_group or width))
        if scale_id is not None else ()
    )
    reorder = vector_op(
        op_id=prefix + ".moe_receive_reorder",
        deps=tuple(dict.fromkeys(deps)),
        stage=stage, layer=plan.index, requests=requests,
        elements=recv_rows, ops_per_element=0, vector_ops_override=recv_rows,
        dtype=payload_dtype, family="routing", phase=mode,
        recipe="moe_receive_reorder", shape=(recv_rows, width),
        read_bytes_override=reorder_io, write_bytes_override=reorder_io,
        returns=returns,
        metadata=(("payload_producer", payload_id),
                  ("scale_producer", scale_id),
                  ("payload_dtype", payload_dtype),
                  ("payload_rows", recv_rows),
                  ("row_elements", width),
                  ("local_experts", local_experts),
                  ("inverse_reorder_map_dtype", "int32"),
                  ("inverse_reorder_map_shape", (recv_rows,)),
                  ("inverse_reorder_map_bytes", reorder_index_bytes)) + scale_meta,
    )
    return declare_received_value(
        reorder, dtype=payload_dtype, shape=(recv_rows, width),
        row_bytes=payload_row_bytes,
        storage=f"moe.reorder:{prefix}.moe_receive_reorder",
        binding="static", binding_name=f"{prefix}.moe_receive_reorder.rx")


def dispatch(*, prefix, plan, stage, requests, quantized, transport, rows, width, ep,
             partitions, mode, hist_gather=None, receive_rows: int | None = None,
             local_experts: int | None = None):
    """Block/row payload exchange, with scale exchange and multi-expert reorder."""
    ops = []
    if ep <= 1:
        return ops, None, None
    routing = _routing_peer_counts(quantized)
    token_route = _token_route_from_op(quantized)
    recv_rows = _resolve_receive_rows(
        quantized, routing, rows, receive_rows=receive_rows)
    token_route_meta = ((("moe_token_route", token_route),)
                        if token_route is not None else ())
    common = (("group_axis", "ep"),
              ("source_assignment_partitions", partitions),
              ("remote_fraction", (ep - 1) / ep),
              ("traffic_model", "uniform_peer_payload"),
              ("source_rows", rows),
              ("scale_dtype", transport.scale_dtype),
              ("eplb_peak_to_avg", transport.eplb_peak_to_avg),
              ("dispatch_dtype", transport.dispatch_dtype))
    scale = None
    recv_scale_shape: tuple[int, ...] = ()
    scale_row_bytes = 0.0
    if transport.quantized and not transport.inline_scales:
        scale_row_bytes = transport.scale_bytes(width)
        scale_group = transport.scale_group or width
        send_scale_shape = _scale_shape(
            _ceil_rows(rows), width, group=scale_group, pack=transport.scale_pack)
        recv_scale_shape = _scale_shape(
            recv_rows, width, group=scale_group, pack=transport.scale_pack)
        scale_peer, scale_link = _peer_metadata(quantized, scale_row_bytes)
        scale = comm_op(
            op_id=prefix + ".moe_scale_dispatch",
            deps=tuple(dict.fromkeys((quantized.op_id,) + ((hist_gather,) if hist_gather else ()))),
            stage=stage, layer=plan.index, requests=requests,
            family="alltoall", phase=mode, recipe="expert_scale_dispatch",
            link_bytes=(scale_link if scale_link is not None
                        else rows * scale_row_bytes * (ep - 1) / ep),
            shape=send_scale_shape,
            metadata=common + (("collective_id", prefix + ".moe_scale_dispatch"),
                               ("payload_kind", "block_scale" if transport.mxfp8 else "row_scale"),
                               ("payload_dtype", transport.scale_dtype),
                               ("row_bytes", scale_row_bytes),
                               ("scale_group", transport.scale_group),
                               ("scale_pack", transport.scale_pack),
                               *scale_peer,
                               *token_route_meta),
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
                           ("payload_kind", "quantized_rows" if transport.quantized else "payload_rows"),
                           ("payload_dtype", transport.payload_dtype),
                           ("row_elements", width),
                           ("row_bytes", payload_row_bytes),
                           ("payload_bytes", width * transport.payload_bytes),
                           ("inline_scale_bytes",
                            transport.scale_bytes(width) if transport.inline_scales else 0.0),
                           ("scale_group", transport.scale_group),
                           (_EXPERT_ROW_FORMAT, row_format),
                           *payload_peer,
                           *token_route_meta),
    )
    ops.append(payload)
    # The received quantities are real output slots of this collective, sized
    # by destination received rows (including diagonal local assignments).
    from .deepep import declare_received_value
    if scale is not None:
        replaced = declare_received_value(
            scale, dtype=transport.scale_dtype, shape=recv_scale_shape,
            row_bytes=scale_row_bytes,
            storage=f"moe.rx:{prefix}.moe_scale_dispatch",
            binding="static", binding_name=f"{prefix}.moe_scale_dispatch.rx")
        ops = [replaced if op.op_id == scale.op_id else op for op in ops]
        scale = replaced
    replaced = declare_received_value(
        payload, dtype=transport.payload_dtype, shape=(recv_rows, width),
        row_bytes=payload_row_bytes,
        storage=f"moe.rx:{prefix}.moe_payload_dispatch",
        binding="static", binding_name=f"{prefix}.moe_payload_dispatch.rx")
    ops = [replaced if op.op_id == payload.op_id else op for op in ops]
    payload = replaced
    eff_local_experts = (
        int(local_experts) if local_experts is not None
        else math.ceil(int(plan.num_experts) / ep)
    )
    reorder_id = None
    out_payload_id = payload.op_id
    out_scale_id = scale.op_id if scale is not None else None
    if eff_local_experts > 1:
        # Upstream vllm-ascend TokenDispatcherWithAlltoAll reorders received
        # rows from rank-contiguous to expert-contiguous layout and retains
        # the inverse map for combine whenever num_local_experts > 1.
        reorder = receive_reorder(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            deps=(payload.op_id,) + ((scale.op_id,) if scale else ()),
            payload_id=payload.op_id, width=width, recv_rows=recv_rows,
            local_experts=eff_local_experts, payload_dtype=transport.payload_dtype,
            payload_row_bytes=payload_row_bytes, mode=mode,
            scale_id=(scale.op_id if scale is not None else None),
            scale_dtype=transport.scale_dtype, scale_shape=recv_scale_shape,
            scale_bytes_total=float(recv_rows) * scale_row_bytes,
            scale_group=transport.scale_group or width,
        )
        ops.append(reorder)
        reorder_id = reorder.op_id
        out_payload_id = reorder.op_id
        if scale is not None:
            out_scale_id = reorder.op_id
    cells, rank, basis = routing if routing is not None else (None, 0, "")
    payload_id = _PeerRouteId(
        out_payload_id, cells, rank, basis,
        token_route=token_route, dispatch_op_id=payload.op_id,
        reorder_op_id=reorder_id, recv_rows=recv_rows,
    )
    return ops, out_scale_id, payload_id


def combine_unorder(*, prefix, plan, stage, requests, deps, transport, rows,
                    width, ep, mode, payload_dispatch, receive_rows=None,
                    local_experts: int | None = None,
                    reorder_op_id: str | None = None,
                    return_dtype: str = "bf16"):
    """Inverse local-expert reorder before combine when num_local_experts > 1."""
    eff_local_experts = (
        int(local_experts) if local_experts is not None
        else math.ceil(int(plan.num_experts) / ep)
    )
    if ep <= 1 or eff_local_experts <= 1:
        return None
    recorded = getattr(payload_dispatch, "recv_rows", None)
    if receive_rows is not None:
        recv_rows = int(receive_rows)
    elif recorded is not None:
        recv_rows = int(recorded)
    else:
        recv_rows = _ceil_rows(rows)
    reorder_id = (
        reorder_op_id if reorder_op_id is not None
        else getattr(payload_dispatch, "reorder_op_id", None)
    )
    unorder_deps = tuple(dict.fromkeys(
        (deps,) + ((reorder_id,) if reorder_id else ())
    ))
    elem_bytes = (
        transport.return_bytes if transport is not None
        else dtype_bytes(return_dtype)
    )
    payload_bytes = float(recv_rows) * width * elem_bytes
    index_bytes = float(recv_rows * _INDEX_BYTES)
    return vector_op(
        op_id=prefix + ".moe_combine_unorder",
        deps=unorder_deps, stage=stage, layer=plan.index, requests=requests,
        elements=recv_rows, ops_per_element=0, vector_ops_override=recv_rows,
        dtype=return_dtype, family="routing", phase=mode,
        recipe="moe_combine_unorder", shape=(recv_rows, width),
        read_bytes_override=payload_bytes + index_bytes,
        write_bytes_override=payload_bytes,
        metadata=(("payload_producer", deps),
                  ("inverse_reorder_map_producer", reorder_id),
                  ("payload_dtype", return_dtype),
                  ("payload_rows", recv_rows),
                  ("row_elements", width),
                  ("local_experts", eff_local_experts),
                  ("inverse_reorder_map_dtype", "int32"),
                  ("inverse_reorder_map_shape", (recv_rows,)),
                  ("inverse_reorder_map_bytes", index_bytes)),
    )


def combine(*, prefix, plan, stage, requests, deps, transport, rows, width, ep,
            partitions, mode, payload_dispatch):
    """BF16 return with dispatch source rows transposed before dtype conversion."""
    if ep <= 1:
        return None
    return_bytes = width * transport.return_bytes
    cells = getattr(payload_dispatch, "cells", None)
    token_route = getattr(payload_dispatch, "token_route", None)
    token_route_meta = ((("moe_token_route", token_route),)
                        if token_route is not None else ())
    dispatch_collective_id = (
        getattr(payload_dispatch, "dispatch_op_id", None) or str(payload_dispatch)
    )
    if cells is None:
        link_bytes = rows * return_bytes * (ep - 1) / ep
        reverse = (("reverse_of_collective", dispatch_collective_id),)
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
    return replace(
        comm_op(
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
                      *peer_metadata,
                      *token_route_meta),
        ),
        dtype="bf16",
    )
