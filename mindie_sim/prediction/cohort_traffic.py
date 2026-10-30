"""Bind forward and inverse expert traffic from one complete active cohort.

[Role] Give dispatch and combine the same directed source/destination ledger.
[Invariants] Combine transposes dispatch; it never reuses a rank's source total
  as its return total. Scope is this graph, stage and explicit collective ID.
  Unequal DP batches do not imply known unequal expert routing. An explicit
  forward/reverse row-width pair converts rows before transposition; absent it
  preserves the historical byte transpose.
[Timeouts & Locks] O(active graphs + participating ranks squared), no I/O.
[Side-Effect Policy] Returns replaced immutable ops; no engine history.
"""
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .peer_traffic import PeerTraffic
from .topology import Topology


def _peer_row_widths(metadata):
    """Read the explicit forward-row -> reverse-row byte conversion, if declared."""
    raw = metadata.get("peer_traffic_row_format")
    if raw is None:
        return None
    if isinstance(raw, dict) and set(raw) == {"forward_bytes", "reverse_bytes"}:
        widths = (raw["forward_bytes"], raw["reverse_bytes"])
        if any(type(value) is not int or value <= 0 for value in widths):
            raise ModelingError("peer_traffic_framed_row_width_requires_positive_bytes")
        return widths
    if not isinstance(raw, dict) or set(raw) != {
            "elements", "forward_dtype", "reverse_dtype"}:
        raise ModelingError("peer_traffic_row_format_requires_elements_and_dtypes")
    elements = raw["elements"]
    if type(elements) is not int or elements < 1:
        raise ModelingError("peer_traffic_row_elements_must_be_positive_integer")
    return (elements * dtype_bytes(raw["forward_dtype"]),
            elements * dtype_bytes(raw["reverse_dtype"]))


def _peer_rows_at_reverse_width(forward, widths):
    """Transpose declared forward rows, then encode each return cell at BF16 width."""
    forward_bytes, reverse_bytes = widths
    ratio = reverse_bytes / forward_bytes
    if forward.get("kind") == "uniform_peers":
        source = forward.get("source_bytes")
        return {**forward, "source_bytes": [float(value) * ratio for value in source]}
    if forward.get("kind") == "matrix":
        rows = forward.get("payload_bytes")
        return {**forward, "payload_bytes": [
            [float(value) * ratio for value in row] for row in rows]}
    raise ModelingError("peer_traffic_row_format_requires_declared_payload")


def bind_cohort_traffic(case, graphs):
    graphs = {replica: tuple(ops) for replica, ops in graphs.items()}
    inverse = [(r, op) for r, ops in graphs.items() for op in ops
               if dict(op.metadata).get("reverse_of_collective")]
    if not inverse:
        return graphs
    topology = Topology(case.machine, case.runtime.parallel)
    declarations = {}
    for replica, ops in graphs.items():
        for op in ops:
            meta = dict(op.metadata)
            if "collective_id" in meta:
                key = replica, int(meta.get("source_stage", op.stage)), meta["collective_id"]
                if key in declarations:
                    raise ModelingError(f"cohort_traffic_duplicate_collective_identity:{key}")
                declarations[key] = op
    replacements = {}
    for replica, reverse in inverse:
        meta = dict(reverse.metadata)
        stage = int(meta.get("source_stage", reverse.stage))
        axis = meta.get("group_axis", "ep")
        source_id = meta["reverse_of_collective"]
        owner = {rank: r for r in graphs for rank in topology.ranks(r, stage)}
        own_ranks = set(topology.ranks(replica, stage))
        descriptors, totals = [], []
        for group in topology.groups(axis, replica, stage):
            source_ops = []
            for rank in group:
                if rank not in owner or (owner[rank], stage, source_id) not in declarations:
                    raise ModelingError("inverse_traffic_requires_complete_dispatch_cohort")
                source_ops.append(declarations[owner[rank], stage, source_id])
            for op in source_ops:
                source_meta = dict(op.metadata)
                if (source_meta.get("group_axis", "ep") != axis
                        or op.family not in {"alltoall", "alltoallv"}):
                    raise ModelingError("inverse_traffic_requires_matching_dispatch_group")
            widths = [_peer_row_widths(dict(op.metadata)) for op in source_ops]
            declared_widths = next((value for value in widths if value is not None), None)
            if declared_widths is not None and any(value != declared_widths for value in widths):
                raise ModelingError("dispatch_peer_traffic_row_widths_differ")
            explicit = [dict(op.metadata).get("peer_traffic") for op in source_ops]
            if any(value is not None for value in explicit):
                if any(value != explicit[0] for value in explicit):
                    raise ModelingError("dispatch_peer_traffic_declarations_differ")
                forward = explicit[0]
            else:
                if any(op.family == "alltoallv" for op in source_ops):
                    raise ModelingError("alltoallv_requires_explicit_peer_matrix")
                forward = {"kind": "uniform_peers",
                           "source_bytes": [op.link_bytes for op in source_ops],
                           "direction": "forward"}
            plan = PeerTraffic.from_mapping(forward, len(group))
            for index, op in enumerate(source_ops):
                if not math.isclose(plan.row_total(index), op.link_bytes, rel_tol=1e-12, abs_tol=1e-9):
                    raise ModelingError("dispatch_peer_traffic_row_total_mismatch")
                key = owner[group[index]], op.op_id
                replacement = replace(op, metadata=tuple({**dict(op.metadata), "peer_traffic": forward}.items()))
                if key in replacements and replacements[key] != replacement:
                    raise ModelingError("representative_dispatch_groups_require_equal_traffic")
                replacements[key] = replacement
            reverse_widths = _peer_row_widths(meta)
            if declared_widths is None and reverse_widths is not None:
                raise ModelingError("reverse_peer_row_width_requires_dispatch_rows")
            if (declared_widths is not None and reverse_widths is not None
                    and reverse_widths != declared_widths):
                raise ModelingError("dispatch_and_reverse_peer_row_widths_differ")
            routed_forward = (forward if declared_widths is None else
                              _peer_rows_at_reverse_width(forward, declared_widths))
            backward = {**routed_forward, "direction": (
                "transpose" if routed_forward.get("direction", "forward") == "forward" else "forward")}
            return_plan = PeerTraffic.from_mapping(backward, len(group))
            descriptors.append(backward)
            totals.extend(return_plan.row_total(i) for i, rank in enumerate(group) if rank in own_ranks)
        if any(item != descriptors[0] for item in descriptors):
            raise ModelingError("representative_combine_groups_require_equal_traffic")
        if any(not math.isclose(value, totals[0], rel_tol=1e-12, abs_tol=1e-9) for value in totals):
            raise ModelingError("representative_tp_sources_require_equal_return_rows")
        if meta.get("peer_traffic") is not None and (
                meta["peer_traffic"] != descriptors[0]
                or not math.isclose(reverse.link_bytes, totals[0], rel_tol=1e-12, abs_tol=1e-9)):
            raise ModelingError("bound_combine_traffic_disagrees_with_dispatch")
        replacements[replica, reverse.op_id] = replace(
            reverse, link_bytes=totals[0],
            metadata=tuple({**meta, "peer_traffic": descriptors[0]}.items()))
    return {r: tuple(replacements.get((r, op.op_id), op) for op in ops)
            for r, ops in graphs.items()}
