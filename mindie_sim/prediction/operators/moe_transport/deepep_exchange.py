"""DeepEP exchange accounting and communication overlap.

[Role] Resolve exchange options and build the two-stage link ledger, typed
  receive contracts, and dispatch and combine overlap branches.
[Invariants] Inter-node delivery is deduplicated per token and remote node.
  Traffic splits conserve the declared bytes and share complete row boundaries
  with their consumers.
[Timeouts & Locks] Arithmetic and ledger projections are bounded by the
  supplied traffic and use no I/O or locks.
[Side-Effect Policy] Exchange operators and metadata are derived without
  mutating the inputs or starting device work.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes


def _validate_eplb(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError("eplb_peak_to_avg_invalid") from exc
    if not math.isfinite(result) or result < 1.0:
        raise ModelingError("eplb_peak_to_avg_must_be_at_least_one")
    return result


def _validate_overlap(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError("deepep_overlap_ratio_invalid") from exc
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ModelingError("deepep_overlap_ratio_must_be_in_0_1")
    return result


def deepep_group_volume(tokens: float, hidden_size: int, top_k: int, *,
                        ep: int, ranks_per_node: int, dtype_bytes_val: float,
                        eplb_peak_to_avg: float = 1.0) -> float:
    """Inter-node group volume with one delivery per token and remote node."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    nodes = max(1, math.ceil(ep / max(1, ranks_per_node)))
    if nodes <= 1 or ep <= 1 or tokens <= 0:
        return 0.0
    node_hit_prob = min(float(top_k) / float(nodes), 1.0)
    remote_nodes = float(nodes - 1)
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * node_hit_prob * remote_nodes * ratio)


def deepep_intra_volume(tokens: float, hidden_size: int, top_k: int, *,
                        ep: int, ranks_per_node: int, dtype_bytes_val: float,
                        eplb_peak_to_avg: float = 1.0) -> float:
    """Intra-node scatter volume: every top-k assignment stays local."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    intra_ep = min(max(1, int(ep)), max(1, int(ranks_per_node)))
    if intra_ep <= 1 or tokens <= 0:
        return 0.0
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * float(top_k) * (float(intra_ep - 1) / float(intra_ep)) * ratio)


def standard_alltoall_volume(tokens: float, hidden_size: int, top_k: int, *,
                             ep: int, dtype_bytes_val: float,
                             eplb_peak_to_avg: float = 1.0) -> float:
    """Flat AllToAll assignment volume with a uniform remote fraction."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    if ep <= 1 or tokens <= 0:
        return 0.0
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * float(top_k) * (float(ep - 1) / float(ep)) * ratio)


def resolve_exchange_options(runtime, transport, *, layer_scope: str = "body"):
    """Resolve EPLB/DeepEP options independent of a payload selector."""
    kernels = getattr(runtime, "kernels", None) or {}
    cfg = kernels.get("moe_transport") or {}
    if not isinstance(cfg, Mapping):
        raise ModelingError("moe_transport_requires_mapping")
    scopes = cfg.get("layer_scopes")
    if scopes is not None:
        if (not isinstance(scopes, (list, tuple)) or not scopes
                or any(scope not in {"body", "draft"} for scope in scopes)):
            raise ModelingError("moe_transport_layer_scopes_require_declared_model_scopes")
        if layer_scope not in scopes:
            return {"eplb": 1.0, "deepep": False, "overlap": 0.0, "ranks_per_node": 0}
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    eplb = (transport.eplb_peak_to_avg if transport is not None else
            cfg.get("eplb_peak_to_avg", cfg.get("ep_ratio",
                    metadata.get("eplb_peak_to_avg", 1.0))))
    overlap = cfg.get("deepep_overlap_ratio",
                      metadata.get("deepep_overlap_ratio", 0.0))
    ranks = cfg.get("ranks_per_node",
                    metadata.get("deepep_ranks_per_node", 0))
    try:
        ranks = int(ranks or 0)
    except (TypeError, ValueError) as exc:
        raise ModelingError("deepep_ranks_per_node_invalid") from exc
    if ranks < 0:
        raise ModelingError("deepep_ranks_per_node_invalid")
    return {
        "eplb": _validate_eplb(eplb),
        "deepep": cfg.get("mode") == "deepep" or bool(metadata.get("enable_deepep", False)),
        "overlap": _validate_overlap(overlap),
        "ranks_per_node": ranks,
    }


def deepep_exchange_factor(*, options, ep, top_k, tokens, hidden, dtype_bytes_val):
    """DeepEP group+intra deduplication ratio over a flat AllToAll (bytes conserved)."""
    if not options["deepep"]:
        return 1.0
    ranks_per_node = options["ranks_per_node"] or ep
    if ep <= ranks_per_node:
        return 1.0
    eff_tokens = max(1.0, float(tokens))
    common = dict(ep=ep, ranks_per_node=ranks_per_node,
                  dtype_bytes_val=dtype_bytes_val, eplb_peak_to_avg=1.0)
    group = deepep_group_volume(eff_tokens, hidden, top_k, **common)
    intra = deepep_intra_volume(eff_tokens, hidden, top_k, **common)
    standard = standard_alltoall_volume(
        eff_tokens, hidden, top_k, ep=ep, dtype_bytes_val=dtype_bytes_val,
        eplb_peak_to_avg=1.0)
    return (group + intra) / max(standard, 1e-12)


def _deepep_peer_traffic_matrix(total_bytes: float, *, ep: int,
                                ranks_per_node: int, top_k: int) -> dict[str, Any]:
    rpn = max(1, min(int(ep), int(ranks_per_node)))
    nodes = max(1, math.ceil(ep / rpn))
    w_intra = float(top_k) * (float(rpn - 1) / float(rpn)) if rpn > 1 else 0.0
    w_group = min(float(top_k) / float(nodes), 1.0) * float(nodes - 1) if nodes > 1 else 0.0
    w_total = max(w_intra + w_group, 1e-12)
    intra_share = w_intra / w_total
    inter_share = 1.0 - intra_share if w_intra + w_group > 0 else 0.0
    intra_row = float(total_bytes) * intra_share
    inter_row = float(total_bytes) - intra_row
    return {
        "kind": "two_tier",
        "ranks_per_node": int(rpn),
        "intra_share": intra_share,
        "inter_share": inter_share,
        "intra_row_bytes": intra_row,
        "inter_row_bytes": inter_row,
        "direction": "forward",
    }


_UNSCALED_MATRIX_KEYS = frozenset({"ranks_per_node", "intra_share", "inter_share"})


def _scale_matrix(value, factor):
    if isinstance(value, Mapping):
        return {
            key: (item if key in _UNSCALED_MATRIX_KEYS else _scale_matrix(item, factor))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return type(value)(_scale_matrix(item, factor) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * factor
    return value


def scale_exchange_operator(op, factor, options, *, ep: int = 1,
                            top_k: int = 1, is_combine: bool = False):
    """Scale token payload traffic and retain EPLB/DeepEP provenance."""
    if factor == 1.0 and not options["deepep"] and options["eplb"] == 1.0:
        return op
    scaled_bytes = op.link_bytes * factor
    metadata = dict(op.metadata)
    if "peer_traffic" in metadata:
        metadata["peer_traffic"] = _scale_matrix(metadata["peer_traffic"], factor)
    elif (options["deepep"] and not is_combine
          and options["ranks_per_node"] > 0 and ep > options["ranks_per_node"]
          and ep % options["ranks_per_node"] == 0):
        metadata["peer_traffic"] = _deepep_peer_traffic_matrix(
            scaled_bytes, ep=ep, ranks_per_node=options["ranks_per_node"], top_k=top_k)
    metadata["eplb_peak_to_avg"] = options["eplb"]
    if options["deepep"]:
        metadata["deepep_mode"] = True
        metadata["deepep_overlap_ratio"] = options["overlap"]
        metadata["deepep_dedup_scale"] = factor / max(options["eplb"], 1e-12)
    return replace(op, link_bytes=scaled_bytes,
                   metadata=tuple(metadata.items()))


def _scale_op_portion(op, ratio: float, *, suffix: str = "", stream: str | None = None,
                      deps: tuple[str, ...] | None = None,
                      peer_traffic=None, extra_metadata=(),
                      suffix_reverse: bool = True):
    meta = dict(op.metadata)
    if peer_traffic is not None:
        meta["peer_traffic"] = peer_traffic
    elif "peer_traffic" in meta:
        meta["peer_traffic"] = _scale_matrix(meta["peer_traffic"], ratio)
    if suffix:
        if "collective_id" in meta:
            meta["collective_id"] = str(meta["collective_id"]) + suffix
        if suffix_reverse and "reverse_of_collective" in meta:
            meta["reverse_of_collective"] = str(meta["reverse_of_collective"]) + suffix
        meta["deepep_overlap_branch"] = True
    meta.update(dict(extra_metadata))
    return replace(
        op,
        op_id=op.op_id + suffix,
        recipe=(op.recipe + suffix) if suffix else op.recipe,
        deps=op.deps if deps is None else deps,
        stream=op.stream if stream is None else stream,
        link_bytes=op.link_bytes * ratio,
        metadata=tuple(meta.items()),
    )


def overlap_row_cut(rows, overlap) -> tuple[int, int]:
    """The single integer boundary shared by transport, conversion and gate.

    Both sides of the boundary are complete assignment rows.  The cut is
    decided once from the received-row count; callers with a different row
    basis (for example the source payload edge) project it through the same
    fraction instead of re-deriving a second split from the overlap ratio.
    """
    try:
        total = int(rows)
    except (TypeError, ValueError) as exc:
        raise ModelingError("deepep_overlap_rows_invalid") from exc
    value = _validate_overlap(overlap)
    if total <= 0:
        return 0, 0
    if total < 2 or value <= 0.0:
        return total, 0
    first = int(math.floor(total * (1.0 - value) + 0.5))
    first = max(1, min(total - 1, first))
    return first, total - first


def _allocate_rows(byte_rows, *, row_bytes, first, total_rows):
    """Allocate integer rows in source order to the first/second chunk."""
    if not row_bytes:
        return ([], [])
    rows = [float(value) / float(row_bytes) for value in byte_rows]
    declared = math.fsum(rows)
    if declared <= 0.0:
        return ([0.0 for _ in rows], [0.0 for _ in rows])
    scale = float(total_rows) / declared
    rows = [value * scale for value in rows]
    main, tail = [], []
    cursor = 0.0
    for count in rows:
        start, stop = cursor, cursor + count
        taken = max(0.0, min(stop, float(first)) - start)
        main.append(taken * float(row_bytes))
        tail.append((count - taken) * float(row_bytes))
        cursor = stop
    return main, tail


def split_peer_traffic_rows(traffic, *, rows: int, first: int, row_bytes: float):
    """Split directed payload cells by complete assignment rows.

    Uniform peers and two-tier source totals are allocated in source order.
    An explicit matrix allocates each destination column by the same integer
    fraction so a destination's received range is never partly counted.  No
    bytes are dropped or invented: both chunks sum to the original cell.
    """
    if not isinstance(traffic, Mapping):
        return None, None
    kind = traffic.get("kind")
    values = traffic.get("source_bytes")
    if kind in {"uniform_peers", "two_tier"} and isinstance(values, (list, tuple)):
        main, tail = _allocate_rows(values, row_bytes=row_bytes, first=first,
                                    total_rows=rows)
        return ({**traffic, "source_bytes": tuple(main)},
                {**traffic, "source_bytes": tuple(tail)})
    matrix = traffic.get("payload_bytes")
    if kind == "matrix" and isinstance(matrix, (list, tuple)) and matrix:
        size = len(matrix)
        main = [[0.0] * size for _ in range(size)]
        tail = [[0.0] * size for _ in range(size)]
        for destination in range(size):
            column = [float(matrix[source][destination])
                      for source in range(size)]
            column_rows = (math.fsum(column) / float(row_bytes)
                           if row_bytes else 0.0)
            cut = int(math.floor(column_rows * first / rows + 0.5)) if rows else 0
            cut = max(0, min(int(column_rows), cut))
            before, after = _allocate_rows(column, row_bytes=row_bytes,
                                           first=cut, total_rows=int(column_rows) or 1)
            for source in range(size):
                main[source][destination] = before[source]
                tail[source][destination] = after[source]
        return ({**traffic, "payload_bytes": tuple(tuple(row) for row in main)},
                {**traffic, "payload_bytes": tuple(tuple(row) for row in tail)})
    if kind == "two_tier":
        ratio = first / float(rows) if rows else 0.0
        main = dict(traffic)
        tail = dict(traffic)
        for key in ("intra_row_bytes", "inter_row_bytes"):
            if key in traffic:
                main[key] = float(traffic[key]) * ratio
                tail[key] = float(traffic[key]) - main[key]
        return main, tail
    return None, None


def declare_received_value(op, *, dtype, shape, row_bytes, storage: str = "",
                           binding: str = "static", binding_name: str = "",
                           state_version=None, issued_rows=None, capacity_rows=None):
    """Declare the actual value a transport op receives on its output slot.

    The declaration lives on the producing collective, so a downstream view
    aliases this value instead of searching for the first ancestor with a
    dtype.  ``storage``/``binding_name`` are the receive allocation identity,
    not an alias label: every view of the same buffer keeps both.
    """
    declared_shape = tuple(int(value) for value in shape)
    rows = int(declared_shape[0])
    width = int(declared_shape[1]) if len(declared_shape) > 1 else 1
    metadata = dict(op.metadata)
    metadata.update({
        "output_format": str(dtype), "output_dtype": str(dtype),
        "output_shape": declared_shape, "primary_output": metadata.get("primary_output", "output"),
        "storage": str(storage or metadata.get("storage") or ""),
        "binding": str(binding), "binding_name": str(binding_name),
        "state_version": state_version,
        "payload_rows": rows, "payload_row_elements": width,
        "payload_row_bytes": float(row_bytes),
        "payload_edge_bytes": float(rows) * float(row_bytes),
        "issued_rows": int(rows if issued_rows is None else issued_rows),
        "capacity_rows": int(rows if capacity_rows is None else capacity_rows),
        "receive_allocation": str(storage or metadata.get("storage") or ""),
        "receive_value_basis": "transport_output_slot",
    })
    return replace(op, dtype=str(dtype), metadata=tuple(metadata.items()))


def received_value_contract(op):
    """Read a transport op's own declared output; never search its ancestry."""
    metadata = dict(op.metadata)
    dtype = (metadata.get("output_format") or metadata.get("output_dtype")
             or op.dtype)
    if not dtype:
        raise ModelingError(
            f"deepep_received_value_requires_declared_output:{op.op_id}")
    shape = tuple(metadata.get("output_shape") or op.local_shape)
    if not shape:
        raise ModelingError(f"deepep_received_value_requires_declared_shape:{op.op_id}")
    row_bytes = metadata.get("payload_row_bytes")
    if row_bytes is None:
        row_bytes = float(shape[-1]) * dtype_bytes(dtype)
    return {
        "producer": op.op_id,
        "output": metadata.get("primary_output") or "output",
        "dtype": str(dtype),
        "shape": shape,
        "row_bytes": float(row_bytes),
        "storage": str(metadata.get("storage") or ""),
        "alias_of": str(metadata.get("alias_of") or op.op_id),
        "state_version": metadata.get("state_version"),
        "binding": str(metadata.get("binding") or "static"),
        "binding_name": str(metadata.get("binding_name") or ""),
    }


def split_deepep_dispatch_overlap(op, options, *, rows=None, row_bytes=None,
                                  row_cut=None):
    """Split dispatch by complete destination rows, not by the ratio scalar.

    ``row_cut`` is the integer boundary already chosen on the received-row
    basis.  When this payload edge uses another basis, the same boundary is
    projected by row fraction; the result is still an integer row edge.
    """
    overlap = float(options.get("overlap", 0.0))
    if not options.get("deepep") or overlap <= 0.0:
        return (op,), None
    metadata = dict(op.metadata)
    if rows is None:
        rows = metadata.get("payload_rows",
                            metadata.get("assignment_rows",
                                         op.local_shape[0] if op.local_shape else 0))
    rows = int(rows)
    if rows < 2:
        # A zero/one-row edge has no integer second segment.  Keep the declared
        # transport parts so every cohort rank exposes the same completion
        # identities; the gate itself still stays unsplit.
        main_op = _scale_op_portion(
            op, 1.0 - overlap,
            extra_metadata=(("deepep_completion_basis",
                             "overlap_ratio_single_row"),))
        over_op = _scale_op_portion(
            op, overlap, suffix="_overlap", stream="deepep_comm_stream",
            deps=(main_op.op_id,),
            extra_metadata=(("deepep_completion_basis",
                             "overlap_ratio_single_row"),))
        return (main_op, over_op), over_op.op_id
    if row_bytes is None:
        row_bytes = metadata.get("payload_row_bytes", metadata.get("row_bytes"))
    if row_cut is None:
        first, second = overlap_row_cut(rows, overlap)
    else:
        cut_total = int(row_cut[0]) + int(row_cut[1])
        if cut_total == rows and int(row_cut[0]) >= 1 and int(row_cut[1]) >= 1:
            first = int(row_cut[0])
        else:
            fraction = (float(row_cut[0]) / float(cut_total)
                        if cut_total > 0 else 1.0 - overlap)
            first = max(1, min(rows - 1,
                               int(math.floor(rows * fraction + 0.5))))
        second = rows - first
    traffic = metadata.get("peer_traffic")
    main_traffic, tail_traffic = split_peer_traffic_rows(
        traffic, rows=rows, first=first, row_bytes=float(row_bytes or 0.0))
    edge_bytes = float(rows) * float(row_bytes or 0.0)
    common = (
        ("deepep_row_cut", (first, second)),
        ("deepep_payload_rows", rows),
        ("deepep_payload_row_bytes", float(row_bytes or 0.0)),
        ("deepep_payload_edge_bytes", edge_bytes),
        ("deepep_completion_basis", "integer_destination_rows"),
        ("deepep_transport_row_basis", "source_payload_edge"),
    )
    main_op = _scale_op_portion(
        op, first / float(rows), peer_traffic=main_traffic,
        extra_metadata=common + (
            ("deepep_chunk_row_start", 0), ("deepep_chunk_row_stop", first),
            ("deepep_chunk_rows", first),
            ("deepep_chunk_payload_bytes", first * float(row_bytes or 0.0))))
    over_op = _scale_op_portion(
        op, second / float(rows), suffix="_overlap", stream="deepep_comm_stream",
        deps=(main_op.op_id,), peer_traffic=tail_traffic,
        extra_metadata=common + (
            ("deepep_chunk_row_start", first), ("deepep_chunk_row_stop", rows),
            ("deepep_chunk_rows", second),
            ("deepep_chunk_payload_bytes", second * float(row_bytes or 0.0))))
    return (main_op, over_op), over_op.op_id


def split_deepep_combine_overlap(op, options, *, overlap_deps: tuple[str, ...],
                                 main_deps: tuple[str, ...],
                                 dispatch_overlap_id: str | None = None):
    """Split combine into concurrent-stream and critical-path ops conserving bytes."""
    overlap = float(options.get("overlap", 0.0))
    if not options.get("deepep") or overlap <= 0.0:
        return (op,), op.op_id
    over_op = _scale_op_portion(
        op, overlap, suffix="_overlap", stream="deepep_comm_stream",
        deps=overlap_deps or op.deps,
        suffix_reverse=dispatch_overlap_id is not None,
    )
    main_op = _scale_op_portion(
        op, 1.0 - overlap, deps=tuple(dict.fromkeys((*main_deps, over_op.op_id))),
    )
    return (over_op, main_op), main_op.op_id
