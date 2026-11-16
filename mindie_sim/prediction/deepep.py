"""DeepEP two-stage deduplicated communication and shared-expert overlap.

[Role] Price DeepEP's inter-node group exchange plus intra-node scatter against
  a flat AllToAll ledger, expose the EPLB load ratio, and run the causal engine
  with the shared-expert stream enabled or disabled.
[Invariants] Node-level dispatch deduplicates a token to one remote delivery;
  intra-node traffic counts each of its top-k assignments. Link work is
  capacity times bytes, and startup latency is charged once per collective
  stage. Shared-expert overlap changes scheduling only: total FLOPs and HBM
  bytes are conserved. Unsupported topology or ratio declarations raise.
[Timeouts & Locks] Pure arithmetic; the causal-engine calls are bounded by the
  supplied case and start no device work, I/O, locks or remote execution.
[Side-Effect Policy] Returns immutable reports; never mutates the input case.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .implementations import append_audit, typed_value


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


def expert_bank_segments(bank_rows, first: int):
    """Partition one destination bank in slot order at a complete row boundary.

    Both results keep the full bank length: a slot whose rows span the cut
    contributes to both segments, while an untouched slot stays zero.  This is
    the per-segment expert bank handed to the selected Grouped program; it is
    not a global expert ledger and never reorders the local weight slot.
    """
    remaining = int(first)
    main, tail = [], []
    for count in bank_rows:
        count = int(count)
        taken = min(count, max(0, remaining))
        main.append(taken)
        tail.append(count - taken)
        remaining -= taken
    return tuple(main), tuple(tail)


def _view_contract(source: Mapping, *, fallback_dtype: str, rows: int, width: int):
    """Validate the source value a row view aliases; no ancestry probing."""
    if not source:
        return None
    dtype = str(source.get("dtype") or fallback_dtype)
    shape = tuple(int(value) for value in source.get("shape") or (rows, width))
    if len(shape) != 2 or shape[1] != width:
        raise ModelingError(
            "deepep_activation_view_requires_matching_received_row_width:"
            f"{shape}!={(rows, width)}")
    if shape[0] != rows:
        raise ModelingError(
            "deepep_activation_view_requires_matching_received_rows:"
            f"{shape[0]}!={rows}")
    return {
        "producer": str(source.get("producer") or ""),
        "output": str(source.get("output") or "output"),
        "dtype": dtype,
        "shape": shape,
        "row_bytes": float(source.get("row_bytes")
                           or width * dtype_bytes(dtype)),
        "storage": str(source.get("storage") or ""),
        "alias_of": str(source.get("alias_of") or source.get("producer") or ""),
        "state_version": source.get("state_version"),
        "binding": str(source.get("binding") or "static"),
        "binding_name": str(source.get("binding_name") or ""),
    }


def split_deepep_up_overlap(up, options, *, dispatch_overlap_id: str | None,
                            allow_split: bool = True, active_rows: int | None = None,
                            source_contract: Mapping | None = None,
                            row_cut: tuple[int, int] | None = None,
                            activation_id: str | None = None):
    """Split expert_up by the same integer rows the transport chunk uses."""
    overlap = float(options.get("overlap", 0.0))
    orig_meta = dict(up.metadata)
    if (not options.get("deepep") or overlap <= 0.0 or dispatch_overlap_id is None
            or len(up.local_shape) < 3):
        return (up,), up.op_id
    m, n, k = up.local_shape[-3:]
    if (not allow_split or m < 2 or "matrix_group_rows" in orig_meta
            or "moe_transport_kind" in orig_meta):
        gated = replace(up, deps=tuple(dict.fromkeys((*up.deps, dispatch_overlap_id))))
        return (gated,), gated.op_id
    declared_active = orig_meta.get("graph_active_query_rows")
    if declared_active is None:
        declared_active = active_rows
    if declared_active is None:
        declared_active = m
    declared_active = min(m, max(0, int(declared_active)))
    if declared_active < m or declared_active < 2:
        # A padded graph has physical rows that no active token reaches.  Do
        # not invent a second active interval for padding: keep one materialized
        # gate and make its conversion wait for both dispatch parts.
        gated = replace(up, deps=tuple(dict.fromkeys((*up.deps, dispatch_overlap_id))))
        gated = replace(gated, metadata=tuple({
            **dict(gated.metadata), "deepep_unsplit_padded": True,
            "deepep_full_active_rows": declared_active,
        }.items()))
        return (gated,), gated.op_id
    if row_cut is None:
        first, second = overlap_row_cut(m, overlap)
    else:
        first, second = int(row_cut[0]), int(row_cut[1])
        if first + second != m or first < 1 or second < 1:
            raise ModelingError(
                "deepep_activation_row_cut_must_match_received_rows:"
                f"{row_cut}!={(m, m)}")
    if first < 1 or second < 1:
        return (up,), up.op_id
    shapes = ((first, n, k), (second, n, k))
    fracs = (first / float(m), second / float(m))
    source = _view_contract(
        source_contract or {}, fallback_dtype=str(
            orig_meta.get("input_format") or up.dtype or "bf16"),
        rows=m, width=k)
    source_dtype = str(source["dtype"]) if source else str(
        orig_meta.get("input_format") or up.dtype or "bf16")
    source_deps = tuple(up.deps)
    result = []
    last_gate = up.op_id
    offset = 0
    for idx, (shape, frac) in enumerate(zip(shapes, fracs)):
        physical_start = offset
        physical_stop = physical_start + shape[0]
        start = min(physical_start, declared_active)
        stop = min(physical_stop, declared_active)
        rows = max(0, stop - start)
        byte_offset = start * k * dtype_bytes(source_dtype)
        view_id = f"{up.op_id}.rows{start}_{stop}"
        view_deps = tuple(dict.fromkeys(
            (*source_deps, dispatch_overlap_id) if idx == 1 else source_deps))
        source_row_bytes = (source["row_bytes"] if source else float(k * dtype_bytes(source_dtype)))
        source_shape = source["shape"] if source else (m, k)
        view_meta = {
            "source_producer": (source["producer"] if source else ""),
            "source_output": (source["output"] if source else ""),
            "source_alias": (source["alias_of"] if source else ""),
            "source_state_version": (source["state_version"] if source else None),
            "source_storage_shape": tuple(source_shape),
            "source_storage": (source["storage"] if source else ""),
            "source_binding": (source["binding"] if source else "static"),
            "source_binding_name": (source["binding_name"] if source else ""),
            "named_input": (f"{source['producer']}.{source['output']}"
                            if source else ""),
            "row_start": start, "row_stop": stop,
            "physical_row_start": physical_start,
            "physical_row_stop": physical_stop,
            "full_rows": m, "full_active_rows": declared_active,
            "full_expected_rows": float(orig_meta.get("expected_rows", m)),
            "byte_offset": byte_offset,
            "element_offset": start * k,
            "source_storage_bytes": (int(math.prod(source_shape))
                                     * dtype_bytes(source_dtype)),
            "deepep_row_cut": (first, second),
            "deepep_chunk_row_start": physical_start,
            "deepep_chunk_row_stop": physical_stop,
            "deepep_chunk_rows": rows,
            "deepep_chunk_payload_bytes": float(rows) * float(source_row_bytes),
            "deepep_activation_view": "zero_materialization_received_rows",
            "output_format": source_dtype, "materialized": False,
            "output_layout": {
                "logical_shape": [rows, k],
                "storage_shape": list(source_shape),
                "offset": start * k,
                "strides": [k, 1]},
            "memory_accesses": ({
                "producer": (source["producer"] if source else ""),
                "output": (source["output"] if source else "output"),
                "direction": "alias", "position": "local",
                "storage": (source["storage"] if source else ""),
                "bytes": 0.0,
                "span": ((physical_start, physical_stop),),
                "last_use": ((up.op_id if idx == 0 else up.op_id + "_overlap")
                             + (".quantize" if orig_meta.get("activation_scale_bytes")
                                else "")),
                "source": (source["producer"] if source else ""),
                "purpose": "received_activation_rows",
            },),
        }
        view = vector_op(
            op_id=view_id, deps=view_deps, stage=up.stage, layer=up.layer,
            requests=up.requests, elements=0, ops_per_element=0,
            vector_ops_override=0, dtype=source_dtype, family="tensor_view",
            phase=up.phase, recipe="row_view", shape=(rows, k),
            read_bytes_override=0, write_bytes_override=0,
            metadata=tuple(view_meta.items()))
        view_meta = dict(view.metadata)
        append_audit(
            view_meta, kind="source_value_contract", op_id=view.op_id,
            boundary={"typed_outputs": (typed_value(
                producer=view_id, dtype=source_dtype, shape=(rows, k),
                storage_shape=tuple(source_shape),
                output=(source["output"] if source else "output"),
                role="activation",
                source=(f"{source['producer']}.{source['output']}"
                        if source else f"{view_id}:row_view"),
                materialized=False, write_bytes=0,
                storage=(source["storage"] if source else ""),
                alias_of=(source["alias_of"] if source else ""),
                state_version=(source["state_version"] if source else None),
                binding=(source["binding"] if source else "static"),
                binding_name=(source["binding_name"] if source else ""),
                contract="zero_materialization_received_view",
                layout={"shape": [rows, k], "strides": [k, 1],
                        "offset": start * k,
                        "storage_shape": list(source_shape)}),)},
            layer=up.layer, stage=up.stage)
        view = replace(view, metadata=tuple(view_meta.items()))
        meta = dict(up.metadata)
        # Integer row slices, not the overlap ratio/expected_rows fraction.
        meta["expected_rows"] = float(shape[0])
        meta["deepep_row_cut"] = (first, second)
        meta["deepep_gate_row_start"] = start
        meta["deepep_gate_row_stop"] = stop
        meta["deepep_physical_row_start"] = physical_start
        meta["deepep_physical_row_stop"] = physical_stop
        meta["deepep_full_rows"] = m
        meta["deepep_full_active_rows"] = declared_active
        meta["deepep_activation_view"] = view_id
        meta["deepep_received_storage"] = tuple(source_shape)
        if "activation_scale_bytes" in meta:
            scale_per_row = float(meta["activation_scale_bytes"]) / m
            meta["activation_scale_bytes"] = scale_per_row * shape[0]
        if "activation_scale_shape" in meta and meta["activation_scale_shape"]:
            meta["activation_scale_shape"] = (
                shape[0], *tuple(meta["activation_scale_shape"])[1:])
        if idx == 1:
            meta["deepep_overlap_branch"] = True
        for key in ("matrix_weight_bytes", "packed_weight_bytes", "weight_scale_bytes", "weight_bytes"):
            if key in meta and isinstance(meta[key], (int, float)):
                meta[key] = float(meta[key]) * frac
        if "matrix_epilogue" in meta:
            ep_m = dict(meta["matrix_epilogue"])
            ep_m["row_scale_bytes"] = float(ep_m.get("row_scale_bytes", 0.0)) * frac
            ep_m["column_scale_bytes"] = float(ep_m.get("column_scale_bytes", 0.0)) * frac
            ep_m["operations_per_output"] = (up.vector_ops * frac) / (shape[0] * shape[1])
            meta["matrix_epilogue"] = ep_m
        realization = meta.get("routing_realization")
        if isinstance(realization, Mapping) and realization.get(
                "basis") == "explicit_token_routing_integer":
            bank = tuple(int(value) for value in
                         realization.get("local_expert_rows") or ())
            if bank and sum(bank) == m:
                seg_main, seg_tail = expert_bank_segments(bank, first)
                segment = seg_main if idx == 0 else seg_tail
                meta["routing_realization"] = {
                    **dict(realization), "local_expert_rows": segment,
                    "rows": sum(segment),
                    "active": sum(1 for value in segment if value)}
                meta["matrix_group_rows"] = segment
                meta["matrix_groups"] = sum(1 for value in segment if value)
                meta["deepep_expert_bank_rows"] = bank
                meta["deepep_segment_expert_bank_rows"] = segment
        gate_input_format = str(meta.get("activation_format") or "int8")
        gate_read = (shape[0] * k * dtype_bytes(gate_input_format)
                     + float(meta.get("activation_scale_bytes", 0.0)))
        gate_mte2 = float(meta.get("packed_weight_bytes", 0.0)) + gate_read
        gate_mte3 = shape[0] * n * dtype_bytes(str(meta.get("output_format") or "bf16"))
        part = replace(
            up,
            op_id=up.op_id if idx == 0 else up.op_id + "_overlap",
            recipe=up.recipe,
            deps=(view_id,),
            local_shape=shape,
            semantic_shape=shape if len(up.semantic_shape) == 3 else up.semantic_shape,
            launch_shape=shape if len(up.launch_shape) == 3 else up.launch_shape,
            padded_shape=tuple(math.ceil(v / 16) * 16 for v in shape),
            flops=up.flops * frac,
            matrix_flops=tuple((dt, val * frac) for dt, val in up.matrix_flops),
            hbm_bytes=gate_mte2 + gate_mte3,
            vector_ops=up.vector_ops * frac,
            mte2_bytes=gate_mte2,
            mte3_bytes=gate_mte3,
            metadata=tuple(meta.items()),
        )
        part_meta = dict(part.metadata)
        out_dtype = str(part_meta.get("output_format") or "bf16")
        out_bytes = shape[0] * n * dtype_bytes(out_dtype)
        part_meta["memory_accesses"] = ({
            "producer": part.op_id,
            "output": part_meta.get("primary_output") or "output",
            "direction": "write", "position": "local",
            "storage": f"moe.act:{part.op_id}", "bytes": float(out_bytes),
            "span": ((0, shape[0]),),
            "last_use": str(activation_id or ""),
            "purpose": "gate_output_rows",
        },)
        append_audit(
            part_meta, kind="source_value_contract", op_id=part.op_id,
            boundary={"typed_outputs": (typed_value(
                producer=part.op_id, dtype=out_dtype, shape=(shape[0], n),
                storage_shape=(shape[0], n),
                output=part_meta.get("primary_output") or "output",
                role="activation", source=part.op_id + ":output_format",
                materialized=True, write_bytes=out_bytes),)},
            layer=up.layer, stage=up.stage)
        part = replace(part, metadata=tuple(part_meta.items()))
        result.extend((view, part))
        last_gate = part.op_id
        offset = physical_stop
    return tuple(result), last_gate


def split_deepep_down_overlap(down, options):
    """Split expert_down along output columns N so combine_overlap sends complete outputs."""
    overlap = float(options.get("overlap", 0.0))
    if not options.get("deepep") or overlap <= 0.0 or len(down.local_shape) < 3:
        return (down,), down.op_id, down.op_id
    m, n, k = down.local_shape[-3:]
    if n < 2:
        return (down,), down.op_id, down.op_id
    raw_n1 = max(1, min(n - 1, round(n * overlap)))
    n1 = max(64, min(n - 64, (raw_n1 // 64) * 64)) if (n >= 128 and n % 64 == 0) else raw_n1
    n2 = max(1, n - n1)
    shapes = ((m, n1, k), (m, n2, k))
    fracs = (n1 / n, n2 / n)
    parts = []
    for idx, (shape, frac) in enumerate(zip(shapes, fracs)):
        meta = dict(down.metadata)
        if idx == 1:
            meta.pop("input_conversion_subgraphs", None)
            meta["activation_producer"] = "explicit_quantization"
            meta["deepep_overlap_branch"] = True
        for key in ("matrix_weight_bytes", "packed_weight_bytes", "weight_scale_bytes", "weight_bytes"):
            if key in meta and isinstance(meta[key], (int, float)):
                meta[key] = float(meta[key]) * frac
        if "matrix_epilogue" in meta:
            ep_m = dict(meta["matrix_epilogue"])
            ep_m["row_scale_bytes"] = float(ep_m.get("row_scale_bytes", 0.0)) * frac
            ep_m["column_scale_bytes"] = float(ep_m.get("column_scale_bytes", 0.0)) * frac
            meta["matrix_epilogue"] = ep_m
        parts.append(replace(
            down,
            op_id=down.op_id if idx == 0 else down.op_id + "_overlap",
            recipe=down.recipe,
            deps=down.deps if idx == 0 else (down.op_id,),
            local_shape=shape,
            semantic_shape=shape if len(down.semantic_shape) == 3 else down.semantic_shape,
            launch_shape=shape if len(down.launch_shape) == 3 else down.launch_shape,
            padded_shape=tuple(math.ceil(v / 16) * 16 for v in shape),
            flops=down.flops * frac,
            matrix_flops=tuple((dt, val * frac) for dt, val in down.matrix_flops),
            hbm_bytes=down.hbm_bytes * frac,
            vector_ops=down.vector_ops * frac,
            mte2_bytes=down.mte2_bytes * frac,
            mte3_bytes=down.mte3_bytes * frac,
            metadata=tuple(meta.items()),
        ))
    return (parts[0], parts[1]), parts[0].op_id, parts[1].op_id


def analyze_shared_expert_overlap(
    case, *, phase: str = "decode", tokens: int = 1, context: int = 4096,
    sequences: int = 16, scope: str = "body",
) -> dict[str, Any]:
    """Evaluate the causal engine with and without the shared-expert stream."""
    from mindie_sim.prediction.efficiency import analyze_network_efficiency
    from mindie_sim.prediction.structure import parse_case

    parsed = parse_case(case)

    def evaluate(enabled):
        metadata = {**dict(parsed.runtime.metadata),
                    "overlap_shared_expert": bool(enabled)}
        runtime = replace(parsed.runtime, metadata=tuple(sorted(metadata.items())))
        report = analyze_network_efficiency(
            replace(parsed, runtime=runtime), phase=phase, tokens=tokens,
            context=context, sequences=sequences, scope=scope)
        return report

    baseline = evaluate(False)
    overlapped = evaluate(True)
    baseline_ms = float(baseline["graph_duration_ms"])
    overlapped_ms = float(overlapped["graph_duration_ms"])
    baseline_flops = float(baseline["total_flops"])
    overlapped_flops = float(overlapped["total_flops"])
    baseline_hbm = float(baseline["total_hbm_bytes"])
    overlapped_hbm = float(overlapped["total_hbm_bytes"])
    return {
        "schema": "mindie-sim.shared-expert-overlap/1",
        "phase": phase, "tokens": int(tokens), "context": int(context),
        "sequences": int(sequences), "scope": scope,
        "baseline_step_ms": baseline_ms,
        "overlapped_step_ms": overlapped_ms,
        "saved_ms": baseline_ms - overlapped_ms,
        "speedup_ratio": baseline_ms / max(overlapped_ms, 1e-12),
        "baseline_operator_sum_ms": float(baseline["operator_sum_ms"]),
        "overlapped_operator_sum_ms": float(overlapped["operator_sum_ms"]),
        "baseline_total_flops": baseline_flops,
        "overlapped_total_flops": overlapped_flops,
        "baseline_total_hbm_bytes": baseline_hbm,
        "overlapped_total_hbm_bytes": overlapped_hbm,
        "flops_conserved": math.isclose(
            baseline_flops, overlapped_flops, rel_tol=1e-12, abs_tol=0.0),
        "hbm_bytes_conserved": math.isclose(
            baseline_hbm, overlapped_hbm, rel_tol=1e-12, abs_tol=0.0),
    }


__all__ = [
    "deepep_group_volume", "deepep_intra_volume", "standard_alltoall_volume",
    "resolve_exchange_options", "deepep_exchange_factor",
    "scale_exchange_operator",
    "analyze_shared_expert_overlap",
]
