"""Layer 3 producer-consumer subgraph and Phase 0 canonical fusion analyzer.

[Role]
  Account removable logical GM round-trip bytes for a candidate fusion or
  canonical Phase 0 Ascend inference subgraph pattern, enforce hard feasibility
  gates (collective barriers, L1/UB live-buffer limits, non-elidable DCP
  normalization, route/layout contracts), and compute the critical-path saving
  interval ``[0, delta_max_us]`` under ``max(P+C, Q+G)``.

[Invariants]
  * ``removable_gm_bytes`` is logical GM traffic.  It is never divided by HBM
    bandwidth to create a latency claim: L2 residency can absorb logical reads.
  * An AICPU ``HcclLaunch`` duration is never substituted for device
    communication time ``C``; doing so returns an explicit invalid/unmodeled
    result instead of a numeric saving.
  * Hard fusion gates are checked before latency is reported.  A gate failure
    keeps the byte accounting but disables the saving interval.
  * Missing node durations keep latency unmodeled.  No empirical efficiency
    discount or guessed duration is introduced.
  * The saving interval is ``[0, max(tail_before - tail_after, 0)]``; zero is the
    no-guarantee bound when the removed work is already hidden.

[Timeouts & Locks]
  Pure DAG computation with bounded traversal.  No locks, sleeps, subprocesses,
  network, or persistent caches.

[Side-Effect Policy]
  Reads caller-provided mappings only.  Never writes files or changes devices.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

from .hardware import validate_buffer_allocations, DEFAULT_ALIGNMENT_BYTES

DEFAULT_CAPACITIES_KIB = {"L1": 512, "L0A": 64, "L0B": 64, "L0C": 256, "UB": 248}
A2_A3_CAPACITIES_KIB = {"L1": 512, "L0A": 64, "L0B": 64, "L0C": 128, "UB": 192}


def _family_capacities(dag: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[dict[str, float], str]:
    family = str(candidate.get("arch_family", dag.get("arch_family", ""))).strip().upper()
    if family in {"A2", "A3"}:
        return dict(A2_A3_CAPACITIES_KIB), f"{family}_family_reference"
    if family == "A5":
        return dict(DEFAULT_CAPACITIES_KIB), "A5_family_reference"
    # Unknown family: use the smaller A2/A3 UB/L0C reference so an omitted
    # platform profile cannot silently inherit A5 headroom.
    return dict(A2_A3_CAPACITIES_KIB), "conservative_unknown_family_a2_a3_reference"
_AICPU_LAUNCH_KINDS = {"aicpu_hccl_launch", "aicpu_launch", "hccl_launch", "aicpu"}
_DEVICE_COMM_KINDS = {"device_comm", "hccl", "device", "collective_device"}


class SubgraphError(ValueError):
    """Stable subgraph-input validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _duration(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not _finite(value) or float(value) < 0:
        return None
    return float(value)


def _bytes_from_tensor(tensor: Mapping[str, Any] | None) -> float | None:
    if not isinstance(tensor, Mapping):
        return None
    for key, multiplier in (("bytes", 1), ("size_bytes", 1), ("size_kib", 1024), ("size_mib", 1024 * 1024)):
        if tensor.get(key) is not None and _finite(tensor[key]) and float(tensor[key]) >= 0:
            return float(tensor[key]) * multiplier
    return None


def _node_map(dag: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw_nodes = dag.get("nodes")
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)) or not raw_nodes:
        raise SubgraphError("nodes_missing")
    nodes: dict[str, dict[str, Any]] = {}
    order: list[dict[str, Any]] = []
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise SubgraphError("node_invalid")
        node_id = str(raw.get("id", "")).strip()
        if not node_id:
            raise SubgraphError("node_id_empty")
        if node_id in nodes:
            raise SubgraphError(f"node_id_duplicate:{node_id}")
        node = dict(raw)
        node["id"] = node_id
        nodes[node_id] = node
        order.append(node)
    return nodes, order


def _edge_list(dag: Mapping[str, Any], nodes: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_edges = dag.get("edges", ())
    if raw_edges is None:
        raw_edges = ()
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raise SubgraphError("edges_invalid")
    edges: list[dict[str, Any]] = []
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            raise SubgraphError("edge_invalid")
        src = str(raw.get("src", "")).strip()
        dst = str(raw.get("dst", "")).strip()
        if src not in nodes or dst not in nodes:
            raise SubgraphError(f"edge_endpoint_unknown:{src}->{dst}")
        edge = dict(raw)
        edge["src"] = src
        edge["dst"] = dst
        edges.append(edge)
    return edges


def _node_kind(node: Mapping[str, Any]) -> str:
    return str(node.get("duration_kind", node.get("kind", ""))).strip().lower().replace("-", "_")


def _is_aicpu_launch(node: Mapping[str, Any]) -> bool:
    kind = _node_kind(node)
    return kind in _AICPU_LAUNCH_KINDS or "hccllaunch" in kind


def _gate(feasible: bool, code: str, detail: Any = None) -> dict[str, Any]:
    result = {"code": code, "feasible": feasible}
    if detail is not None:
        result["detail"] = detail
    return result


def _resolve_remove_nodes(candidate: Mapping[str, Any], nodes: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    remove_nodes = candidate.get("remove_nodes", ())
    if isinstance(remove_nodes, str):
        remove_nodes = [remove_nodes]
    if not isinstance(remove_nodes, Sequence):
        raise SubgraphError("remove_nodes_invalid")
    result: list[str] = []
    for value in remove_nodes:
        node_id = str(value)
        if node_id not in nodes:
            raise SubgraphError(f"remove_node_unknown:{node_id}")
        if node_id not in result:
            result.append(node_id)
    return result, [nodes[node_id] for node_id in result]


def _resolve_remove_edges(
    candidate: Mapping[str, Any],
    edges: Sequence[Mapping[str, Any]],
    producer: str | None,
    consumer: str | None,
    remove_nodes: Sequence[str],
) -> list[dict[str, Any]]:
    raw = candidate.get("remove_edges")
    if raw is None:
        removed_set = set(remove_nodes)
        return [
            dict(edge) for edge in edges
            if (edge["src"] == producer and (edge["dst"] == consumer or edge["dst"] in removed_set))
            or (edge["src"] in removed_set and (edge["dst"] == consumer or edge["dst"] in removed_set))
        ]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SubgraphError("remove_edges_invalid")
    edge_index = {(edge["src"], edge["dst"]): edge for edge in edges}
    result: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            src, dst = str(item.get("src", "")), str(item.get("dst", ""))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            src, dst = str(item[0]), str(item[1])
        else:
            raise SubgraphError("remove_edge_invalid")
        edge = edge_index.get((src, dst))
        if edge is None:
            raise SubgraphError(f"remove_edge_unknown:{src}->{dst}")
        result.append(edge)
    return result


def _gm_bytes_for_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    tensor = edge.get("tensor")
    tensor_bytes = _bytes_from_tensor(tensor if isinstance(tensor, Mapping) else None)
    write = edge.get("gm_write_bytes")
    read = edge.get("gm_read_bytes")
    if write is None and read is None:
        if edge.get("gm_roundtrip", True) is False:
            write = tensor_bytes
            read = 0.0
        else:
            write = tensor_bytes
            read = tensor_bytes
    write_value = _duration(write) if write is not None else None
    read_value = _duration(read) if read is not None else None
    if write_value is None and read_value is None:
        return {"src": edge["src"], "dst": edge["dst"], "bytes": None, "write_bytes": None, "read_bytes": None}
    write_value = write_value or 0.0
    read_value = read_value or 0.0
    return {
        "src": edge["src"],
        "dst": edge["dst"],
        "tensor": dict(tensor) if isinstance(tensor, Mapping) else None,
        "write_bytes": write_value,
        "read_bytes": read_value,
        "bytes": write_value + read_value,
    }


def _topological_order(nodes: Mapping[str, Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> list[str]:
    indegree = {node_id: 0 for node_id in nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["src"]].append(edge["dst"])
        indegree[edge["dst"]] += 1
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(nodes):
        raise SubgraphError("subgraph_cycle_detected")
    return order


def _critical_path(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    removed: set[str] | None = None,
    producer: str | None = None,
    fused_extra_us: float = 0.0,
) -> dict[str, Any]:
    removed = removed or set()
    active = {node_id: node for node_id, node in nodes.items() if node_id not in removed}
    if not active:
        return {"status": "empty_graph", "critical_path_us": None}
    rewired: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        if src in removed and dst in removed:
            continue
        if src in removed:
            if producer is None or producer in removed:
                continue
            src = producer
        if dst in removed:
            continue
        if src == dst or src not in active or dst not in active:
            continue
        key = (src, dst)
        if key not in seen_edges:
            seen_edges.add(key)
            rewired.append({"src": src, "dst": dst})
    duration: dict[str, float | None] = {}
    for node_id, node in active.items():
        value = _duration(node.get("duration_us"))
        duration[node_id] = value
    if any(value is None for value in duration.values()):
        return {
            "status": "unmodeled_durations",
            "critical_path_us": None,
            "missing_duration_nodes": sorted(node_id for node_id, value in duration.items() if value is None),
        }
    if producer in active and fused_extra_us:
        duration[producer] = (duration[producer] or 0.0) + fused_extra_us
    order = _topological_order(active, rewired)
    longest: dict[str, float] = {}
    previous: dict[str, str | None] = {}
    for node_id in order:
        predecessors = [edge["src"] for edge in rewired if edge["dst"] == node_id]
        if predecessors:
            best = max(predecessors, key=lambda item: longest[item])
            longest[node_id] = longest[best] + float(duration[node_id] or 0.0)
            previous[node_id] = best
        else:
            longest[node_id] = float(duration[node_id] or 0.0)
            previous[node_id] = None
    end = max(longest, key=longest.get)
    path: list[str] = []
    current: str | None = end
    while current is not None and len(path) <= len(active):
        path.append(current)
        current = previous[current]
    path.reverse()
    return {
        "status": "modeled",
        "critical_path_us": longest[end],
        "critical_path_nodes": path,
        "node_durations_us": {node_id: duration[node_id] for node_id in order},
    }


def _contract_gate(
    candidate: Mapping[str, Any],
    remove_edges: Sequence[Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if candidate.get("allow_layout_change"):
        return _gate(True, "layout_scale_contract", {"allowed": True})
    explicit = candidate.get("requires_layout_change")
    if explicit is True:
        return _gate(False, "layout_scale_contract_change", {"reason": "candidate_requires_layout_or_scale_change"})
    expected = candidate.get("consumer_input_contract")
    if isinstance(expected, Mapping):
        for edge in remove_edges:
            tensor = edge.get("tensor")
            if not isinstance(tensor, Mapping):
                continue
            for key in ("dtype", "layout", "scale"):
                if key in expected and key in tensor and str(expected[key]) != str(tensor[key]):
                    return _gate(False, "layout_scale_contract_mismatch", {
                        "src": edge["src"], "dst": edge["dst"], "field": key,
                        "producer": tensor[key], "consumer": expected[key],
                    })
    return _gate(True, "layout_scale_contract")


def _buffer_gate(
    candidate: Mapping[str, Any],
    dag: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    remove_nodes: Sequence[str],
) -> dict[str, Any]:
    allocations = candidate.get("live_buffers")
    if allocations is None:
        allocations = []
        for node_id in remove_nodes:
            node = nodes[node_id]
            for allocation in node.get("live_buffers", ()) or ():
                allocations.append(allocation)
    if not allocations:
        return _gate(True, "live_buffer_limit", {"checked": False, "reason": "no_live_buffers_supplied"})
    capacities = candidate.get("capacities_kib")
    capacity_source = "candidate"
    if capacities is None:
        capacities = dag.get("capacities_kib")
        capacity_source = "dag"
    if capacities is None:
        capacities, capacity_source = _family_capacities(dag, candidate)
    if not isinstance(capacities, Mapping):
        return _gate(False, "live_buffer_limit", {"reason": "capacities_invalid"})
    check = validate_buffer_allocations(
        allocations,
        capacities,
        alignment_bytes=int(candidate.get("alignment_bytes", DEFAULT_ALIGNMENT_BYTES)),
        ping_pong=candidate.get("ping_pong"),
        pack_missing_offsets=True,
    )
    return _gate(
        check["ok"],
        "live_buffer_limit",
        {
            "checked": True,
            "capacity_source": capacity_source,
            "status": check["status"],
            "diagnostics": check["diagnostics"],
        },
    )


def _dcp_gate(candidate: Mapping[str, Any], removed_nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elide = bool(candidate.get("elide_normalization")) or bool(candidate.get("elide_full_reduction"))
    for node in removed_nodes:
        text = " ".join(
            str(node.get(key, "")) for key in ("op", "kind", "name", "normalization", "semantic")
        ).lower()
        if ("dcp" in text or "merge" in text) and (
            "normalization" in text or "lse" in text or "full_reduction" in text or "reduce" in text
        ):
            if elide:
                return _gate(False, "non_elidable_full_reduction_normalization", {
                    "node": node["id"],
                    "normalization": node.get("normalization"),
                    "reason": "DCP full-reduction/LSE normalization must remain in the consumer",
                })
            return _gate(True, "non_elidable_full_reduction_normalization", {
                "node": node["id"],
                "normalization_preserved": True,
            })
    return _gate(True, "non_elidable_full_reduction_normalization", {"checked": False})


def _collective_gate(
    candidate: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    remove_nodes: Sequence[str],
    remove_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allowed = bool(candidate.get("allow_collective_barrier"))
    merged = set(remove_nodes)
    if candidate.get("producer"):
        merged.add(str(candidate["producer"]))
    if candidate.get("consumer"):
        merged.add(str(candidate["consumer"]))
    for node_id in merged:
        node = nodes.get(node_id, {})
        if bool(node.get("collective")) or bool(node.get("collective_barrier")):
            if allowed:
                return _gate(True, "collective_barrier_requires_global_sync", {
                    "node": node_id, "barrier_preserved": True,
                })
            return _gate(False, "collective_barrier_requires_global_sync", {"node": node_id})
    for edge in remove_edges:
        if bool(edge.get("collective_barrier")) or bool(edge.get("collective")):
            if allowed:
                return _gate(True, "collective_barrier_requires_global_sync", {
                    "edge": f"{edge['src']}->{edge['dst']}", "barrier_preserved": True,
                })
            return _gate(False, "collective_barrier_requires_global_sync", {"edge": f"{edge['src']}->{edge['dst']}"})
    return _gate(True, "collective_barrier_requires_global_sync", {"allow_collective_barrier": allowed})


def _route_gate(candidate: Mapping[str, Any], removed_nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    preserve = bool(candidate.get("preserve_route", False))
    for node in removed_nodes:
        text = " ".join(str(node.get(key, "")) for key in ("op", "kind", "name", "route")).lower()
        if "zeroexpertrouter" in text or "zero_expert_router" in text or bool(node.get("route_fallback")):
            if not preserve:
                return _gate(False, "route_fallback_contract", {"node": node["id"]})
            return _gate(True, "route_fallback_contract", {"node": node["id"], "preserved": True})
    return _gate(True, "route_fallback_contract", {"checked": False})


def _overlap_analysis(
    dag: Mapping[str, Any],
    candidate: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    overlap = candidate.get("overlap")
    if not isinstance(overlap, Mapping):
        return {"status": "not_supplied", "reason": "overlap_topology_missing"}
    def resolve(key: str, explicit_key: str) -> tuple[float | None, str | None, Mapping[str, Any] | None]:
        if overlap.get(explicit_key) is not None:
            value = _duration(overlap[explicit_key])
            return value, None, None
        node_id = overlap.get(key)
        if node_id is None:
            return None, f"missing_{key}", None
        node = nodes.get(str(node_id))
        if node is None:
            return None, f"unknown_{key}:{node_id}", None
        return _duration(node.get("duration_us")), None, node
    pack, pack_error, _ = resolve("pack", "pack_us")
    comm, comm_error, comm_node = resolve("comm", "comm_us")
    current, current_error, _ = resolve("current", "current_us")
    gate, gate_error, _ = resolve("gate", "gate_us")
    errors = [value for value in (pack_error, comm_error, current_error, gate_error) if value]
    if errors:
        return {"status": "unmodeled_durations", "reasons": errors}
    if comm_node is not None and _is_aicpu_launch(comm_node):
        return {
            "status": "invalid_communication_duration",
            "reason": "aicpu_hccl_launch_is_not_device_communication",
            "communication_node": comm_node.get("id"),
            "communication_kind": _node_kind(comm_node),
            "min_saving_us": None,
            "max_saving_us": None,
        }
    assert pack is not None and comm is not None and current is not None and gate is not None
    pack_comm = pack + comm
    current_gate = current + gate
    tail_before = max(pack_comm, current_gate)
    tail_after = max(comm, current_gate)
    max_saving = max(0.0, tail_before - tail_after)
    return {
        "status": "modeled",
        "formula": "max(P + C, Q + G)",
        "pack_us": pack,
        "communication_us": comm,
        "current_us": current,
        "gate_us": gate,
        "pack_plus_comm_us": pack_comm,
        "current_plus_gate_us": current_gate,
        "tail_before_us": tail_before,
        "tail_after_eliding_pack_us": tail_after,
        "min_saving_us": 0.0,
        "max_saving_us": max_saving,
        "fully_elided_saving_us": max_saving,
        "critical_branch_before": "pack_plus_comm" if pack_comm >= current_gate else "current_plus_gate",
        "communication_source": "device_duration" if comm_node is not None else "explicit",
        "note": "AICPU HcclLaunch is rejected as C; zero is the no-guarantee bound.",
    }


def analyze_subgraph_fusion(
    dag: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(dag, Mapping):
        raise SubgraphError("dag_invalid")
    nodes, order = _node_map(dag)
    edges = _edge_list(dag, nodes)
    candidate = dict(candidate or dag.get("candidate_fusion") or {})
    if not isinstance(candidate, Mapping):
        raise SubgraphError("candidate_invalid")
    producer = str(candidate.get("producer", "")).strip() or None
    consumer = str(candidate.get("consumer", "")).strip() or None
    if producer is not None and producer not in nodes:
        raise SubgraphError(f"producer_unknown:{producer}")
    if consumer is not None and consumer not in nodes:
        raise SubgraphError(f"consumer_unknown:{consumer}")
    remove_nodes, removed_node_objects = _resolve_remove_nodes(candidate, nodes)
    remove_edges = _resolve_remove_edges(candidate, edges, producer, consumer, remove_nodes)
    removed_edges_bytes = [_gm_bytes_for_edge(edge) for edge in remove_edges]
    removable_bytes = sum(item["bytes"] or 0.0 for item in removed_edges_bytes)
    missing_bytes = [f"{item['src']}->{item['dst']}" for item in removed_edges_bytes if item["bytes"] is None]
    gates = [
        _collective_gate(candidate, nodes, remove_nodes, remove_edges),
        _buffer_gate(candidate, dag, nodes, remove_nodes),
        _dcp_gate(candidate, removed_node_objects),
        _route_gate(candidate, removed_node_objects),
        _contract_gate(candidate, remove_edges, nodes),
    ]
    failures = [gate["code"] for gate in gates if not gate["feasible"]]
    latency_before = _critical_path(nodes, edges)
    latency_after = _critical_path(
        nodes,
        edges,
        removed=set(remove_nodes),
        producer=producer,
        fused_extra_us=float(candidate.get("fused_extra_us", 0.0) or 0.0),
    )
    savings = None
    if latency_before.get("status") == "modeled" and latency_after.get("status") == "modeled":
        savings = max(0.0, latency_before["critical_path_us"] - latency_after["critical_path_us"])
    overlap = _overlap_analysis(dag, candidate, nodes)
    capacities = candidate.get("capacities_kib", dag.get("capacities_kib"))
    if capacities is None:
        capacities, _ = _family_capacities(dag, candidate)
    l2_cache_mib = candidate.get("l2_cache_mib", dag.get("l2_cache_mib"))
    working_set_mib = candidate.get("working_set_mib", dag.get("working_set_mib"))
    residency = None
    if _finite(l2_cache_mib) and _finite(working_set_mib):
        residency = float(working_set_mib) <= float(l2_cache_mib)
    status = "modeled"
    if failures:
        status = "infeasible"
    elif missing_bytes:
        status = "unmodeled_gm_bytes"
    elif latency_before.get("status") != "modeled" or latency_after.get("status") != "modeled":
        status = "unmodeled_latency"
    if overlap.get("status") == "invalid_communication_duration":
        status = "invalid_communication_duration"
    payload: dict[str, Any] = {
        "status": status,
        "fusion": {
            "name": candidate.get("name", "candidate"),
            "producer": producer,
            "consumer": consumer,
            "remove_nodes": remove_nodes,
            "remove_edges": [{"src": edge["src"], "dst": edge["dst"]} for edge in remove_edges],
        },
        "removable_gm_bytes": removable_bytes if not missing_bytes else None,
        "removable_gm_read_bytes": sum(item["read_bytes"] or 0.0 for item in removed_edges_bytes),
        "removable_gm_write_bytes": sum(item["write_bytes"] or 0.0 for item in removed_edges_bytes),
        "removable_gm_bytes_by_edge": removed_edges_bytes,
        "removed_kernel_launches": len(remove_nodes),
        "l2_cache_caveat": {
            "logical_gm_bytes": removable_bytes if not missing_bytes else None,
            "hbm_bytes_lower_bound": None,
            "hbm_time_lower_bound_us": None,
            "l2_cache_mib": float(l2_cache_mib) if _finite(l2_cache_mib) else None,
            "working_set_mib": float(working_set_mib) if _finite(working_set_mib) else None,
            "l2_cache_residency_possible": residency,
            "note": "Logical GM reads are not HBM traffic; do not divide by HBM bandwidth when L2 can retain the working set.",
        },
        "feasibility": {
            "feasible": not failures,
            "gates": gates,
            "failures": failures,
        },
        "latency": {
            "before": latency_before,
            "after": latency_after,
            "critical_path_saving_us": savings,
            "min_saving_us": 0.0 if savings is not None else None,
            "max_saving_us": savings,
        },
        "overlap": overlap,
        "assumptions": [
            "GM byte counts are logical write+read requests, not HBM transactions",
            "L2 residency is evaluated when a working set is supplied; otherwise it stays unknown",
            "critical-path savings are theoretical and do not replace a matched A/B measurement",
            "AICPU HcclLaunch is not device communication completion time",
            "buffer capacities use the declared family reference when platform capacities are omitted; unknown family falls back to A2/A3, not A5",
        ],
    }
    if capacities is not None:
        payload["buffer_capacities_kib"] = dict(capacities) if isinstance(capacities, Mapping) else capacities
    return payload


CANONICAL_FUSION_PATTERNS: dict[str, dict[str, Any]] = {
    "add_rmsnorm_dynamic_quant": {"fused_kernel": "npu_add_rms_norm_dynamic_quant", "ops": ("add", "rmsnorm", "dynamic_quant"), "description": "Add + RMSNorm + DynamicQuant fused vector epilogue", "ub_buffers_per_token_factor": 2.0},
    "qkv_rope_reshape_cache": {"fused_kernel": "npu_rope_reshape_and_cache", "ops": ("qkv_split", "rope", "kv_cache_write"), "description": "QKV Split + RoPE + Paged KV Cache Write", "ub_buffers_per_token_factor": 2.0},
    "mla_prologue_epilogue": {"fused_kernel": "npu_mla_prolog_v3", "ops": ("mla_up_down_proj", "rope", "quant_page_write"), "description": "MLA Up/Down Proj + RMSNorm/RoPE + Quant/PageWrite", "ub_buffers_per_token_factor": 2.5},
    "lightning_indexer_topk": {"fused_kernel": "npu_lightning_indexer_topk", "ops": ("query_dequant", "lightning_indexer", "topk"), "description": "Query Dequant + Lightning Indexer Score + TopK Selection", "ub_buffers_per_token_factor": 1.5},
    "moe_gating_topk_dispatch": {"fused_kernel": "npu_moe_gating_top_k_distribute_dispatch", "ops": ("moe_gating_topk", "log2phy", "moe_distribute_dispatch"), "description": "MoE Router TopK + Log2Phy Mapping + MoeDistributeDispatch", "ub_buffers_per_token_factor": 1.0},
    "gmm_swiglu_dynamic_quant": {"fused_kernel": "npu_grouped_matmul_swiglu_quant", "ops": ("grouped_matmul", "swiglu", "dynamic_quant"), "description": "GroupedMatmul + SwiGLU + DynamicQuant", "ub_buffers_per_token_factor": 3.0},
    "dit_usp_teacache_block": {"fused_kernel": "npu_dit_usp_teacache_fused_block", "ops": ("dit_attention_usp", "teacache_skip_gate", "dit_residual_update"), "alt_ops": ("dit_attention_usp", "teacache_skip_gate"), "description": "DiT Attention USP + TeaCache Skip Gate", "ub_buffers_per_token_factor": 2.0},
}

_OP_ALIASES: dict[str, str] = {
    "add": "add", "aten.add": "add", "aten.add.tensor": "add", "rmsnorm": "rmsnorm", "rms_norm": "rmsnorm", "npu_rms_norm": "rmsnorm",
    "dynamic_quant": "dynamic_quant", "dynamicquant": "dynamic_quant", "npu_dynamic_quant": "dynamic_quant",
    "qkv_split": "qkv_split", "split": "qkv_split", "aten.split": "qkv_split", "rope": "rope", "apply_rope": "rope", "rotary_embedding": "rope", "npu_rotary_mul": "rope",
    "kv_cache_write": "kv_cache_write", "reshape_and_cache": "kv_cache_write", "mla_up_down_proj": "mla_up_down_proj", "mla_proj": "mla_up_down_proj",
    "quant_page_write": "quant_page_write", "mla_page_write": "quant_page_write", "query_dequant": "query_dequant", "dequant": "query_dequant",
    "lightning_indexer": "lightning_indexer", "indexer": "lightning_indexer", "topk": "topk", "aten.topk": "topk",
    "moe_gating_topk": "moe_gating_topk", "router_topk": "moe_gating_topk", "moe_gating_top_k": "moe_gating_topk", "log2phy": "log2phy", "log2phy_map": "log2phy",
    "moe_distribute_dispatch": "moe_distribute_dispatch", "distribute_dispatch": "moe_distribute_dispatch",
    "grouped_matmul": "grouped_matmul", "gmm": "grouped_matmul", "npu_grouped_matmul": "grouped_matmul", "swiglu": "swiglu", "npu_swiglu": "swiglu", "m3_swiglu": "swiglu",
    "dit_attention_usp": "dit_attention_usp", "dit_usp_attention": "dit_attention_usp", "dit_modulate_norm": "dit_attention_usp",
    "teacache_skip_gate": "teacache_skip_gate", "teacache_gate": "teacache_skip_gate", "dit_residual_update": "dit_residual_update", "apply_gate": "dit_residual_update",
}


def _dtype_bytes_local(dtype: str) -> float:
    token = str(dtype).strip().lower()
    return 4.0 if token in {"fp32", "float32", "int32"} else (1.0 if token in {"fp8", "int8", "uint8", "mxfp8"} else 2.0)


def _pattern_edge_bytes(pattern_name: str, cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    tokens, hidden = int(cfg.get("num_tokens", 128)), int(cfg.get("hidden_size", 4096))
    heads, kv_heads, head_dim = int(cfg.get("num_heads", 32)), int(cfg.get("num_kv_heads", 8)), int(cfg.get("head_dim", 128))
    moe_inter, top_k = int(cfg.get("moe_intermediate_size", 2048)), int(cfg.get("top_k", 8))
    q_lora, kv_lora, qk_rope = int(cfg.get("q_lora_rank", 1536)), int(cfg.get("kv_lora_rank", 512)), int(cfg.get("qk_rope_head_dim", 64))
    idx_heads, idx_dim, cand_rows = int(cfg.get("index_n_heads", 8)), int(cfg.get("index_head_dim", 128)), int(cfg.get("candidate_rows", 2048))
    b = _dtype_bytes_local(str(cfg.get("dtype", "bf16")))
    if tokens <= 0 or hidden <= 0:
        raise SubgraphError("canonical_pattern_shape_invalid")
    if pattern_name == "add_rmsnorm_dynamic_quant":
        return float(tokens * hidden * b), float(tokens * hidden * b), float(hidden * b)
    if pattern_name == "qkv_rope_reshape_cache":
        return float(tokens * (heads + 2 * kv_heads) * head_dim * b), float(tokens * 2 * kv_heads * head_dim * b), float((heads + 2 * kv_heads) * head_dim * b)
    if pattern_name == "mla_prologue_epilogue":
        return float(tokens * (q_lora + kv_lora + qk_rope) * b), float(tokens * (kv_lora + qk_rope) * b), float((q_lora + kv_lora + qk_rope) * b)
    if pattern_name == "lightning_indexer_topk":
        return float(tokens * idx_heads * idx_dim * b), float(tokens * cand_rows * b), float((idx_heads * idx_dim + cand_rows) * b)
    if pattern_name == "moe_gating_topk_dispatch":
        return float(tokens * top_k * 8.0), float(tokens * top_k * 8.0), float(top_k * 16.0)
    if pattern_name == "gmm_swiglu_dynamic_quant":
        return float(tokens * top_k * 2 * moe_inter * b), float(tokens * top_k * moe_inter * b), float(2 * moe_inter * b)
    return float(tokens * hidden * b), float(tokens * hidden * b), float(hidden * b)


def evaluate_canonical_fusion_opportunity(pattern_name: str, shape: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Deterministically evaluate a canonical Phase 0 fusible subgraph pattern."""
    key = str(pattern_name).strip().lower()
    if key not in CANONICAL_FUSION_PATTERNS:
        raise SubgraphError(f"canonical_pattern_unknown:{pattern_name}")
    meta = CANONICAL_FUSION_PATTERNS[key]
    cfg: dict[str, Any] = {}
    if shape is not None:
        if not isinstance(shape, Mapping):
            raise SubgraphError("canonical_pattern_shape_invalid")
        cfg.update(shape)
    cfg.update(kwargs)
    e1_bytes, e2_bytes, row_bytes = _pattern_edge_bytes(key, cfg)
    tokens = int(cfg.get("num_tokens", 128))
    auto_tile = bool(cfg.get("auto_tile", True))
    max_fit = max(1, int((160.0 * 1024.0) // max(1.0, row_bytes * float(meta["ub_buffers_per_token_factor"]))))
    tile_tokens = int(cfg.get("tile_tokens", min(tokens, 8, max_fit) if auto_tile else tokens))
    ub_live_kib = (tile_tokens * row_bytes * float(meta["ub_buffers_per_token_factor"])) / 1024.0
    l1_live_kib = float(cfg.get("l1_live_kib", min(256.0, ub_live_kib)))
    raw_durations = cfg.get("op_durations_us")
    d0 = d1 = d2 = None
    if isinstance(raw_durations, Sequence) and not isinstance(raw_durations, (str, bytes)) and len(raw_durations) >= 3:
        d0, d1, d2 = _duration(raw_durations[0]), _duration(raw_durations[1]), _duration(raw_durations[2])
    elif isinstance(raw_durations, Mapping):
        ops = meta["ops"]
        d0, d1, d2 = (_duration(raw_durations.get(ops[i])) for i in range(3))
    fused_us = _duration(cfg.get("fused_duration_us"))
    fused_extra_us = 0.0
    if d0 is not None and d1 is not None and d2 is not None:
        fused_extra_us = max(0.0, fused_us - d0) if fused_us is not None else float(cfg.get("fused_extra_us", 0.5 * (d1 + d2)))
    ops = meta["ops"]
    dag = {
        "arch_family": cfg.get("arch_family", "A3"),
        "nodes": [
            {"id": ops[0], "op": ops[0], "duration_us": d0},
            {"id": ops[1], "op": ops[1], "duration_us": d1, "live_buffers": [{"buffer": "UB", "name": "ub_tile", "size_kib": ub_live_kib}]},
            {"id": ops[2], "op": ops[2], "duration_us": d2, "live_buffers": [{"buffer": "L1", "name": "l1_tile", "size_kib": l1_live_kib}]},
        ],
        "edges": [{"src": ops[0], "dst": ops[1], "tensor": {"bytes": e1_bytes}}, {"src": ops[1], "dst": ops[2], "tensor": {"bytes": e2_bytes}}],
    }
    candidate = {"name": key, "producer": ops[0], "remove_nodes": [ops[1], ops[2]], "fused_extra_us": fused_extra_us, "allow_collective_barrier": bool(cfg.get("allow_collective_barrier", True))}
    analysis = analyze_subgraph_fusion(dag, candidate)
    feasible = bool(analysis["feasibility"]["feasible"])
    max_saving = analysis["latency"]["max_saving_us"] if feasible else None
    return {
        "status": analysis["status"], "pattern_name": key, "fused_kernel": meta["fused_kernel"], "description": meta["description"],
        "unfused_ops": list(ops), "eliminated_gm_read_bytes": analysis["removable_gm_read_bytes"], "eliminated_gm_write_bytes": analysis["removable_gm_write_bytes"],
        "eliminated_gm_roundtrip_bytes": analysis["removable_gm_bytes"], "kernel_launches_before": 3, "kernel_launches_after": 1,
        "kernel_launch_reduction": analysis["removed_kernel_launches"], "tile_tokens": tile_tokens, "ub_live_kib": ub_live_kib,
        "ub_l1_feasible": feasible, "feasibility": analysis["feasibility"], "latency_savings_us_interval": [0.0, max_saving] if max_saving is not None else None,
        "unmodeled_reason": None if max_saving is not None else ("infeasible_fusion" if not feasible else "missing_op_durations"),
    }


def scan_fusible_subgraph_patterns(op_sequence: Sequence[Any], shape: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Scan an operator sequence and greedily match canonical Phase 0 fusion patterns."""
    if not isinstance(op_sequence, Sequence) or isinstance(op_sequence, (str, bytes)):
        raise SubgraphError("op_sequence_invalid")
    normalized: list[str] = []
    durations: list[float | None] = []
    for item in op_sequence:
        raw_name = str(item.get("op", item.get("name", "")) if isinstance(item, Mapping) else item).strip().lower()
        durations.append(_duration(item.get("duration_us")) if isinstance(item, Mapping) else None)
        normalized.append(_OP_ALIASES.get(raw_name, raw_name))
    matches: list[dict[str, Any]] = []
    idx, n = 0, len(normalized)
    while idx < n:
        matched_eval: dict[str, Any] | None = None
        step = 1
        for pat_name, meta in CANONICAL_FUSION_PATTERNS.items():
            for seq_ops, pad_zero in ((tuple(meta["ops"]), False), (tuple(meta.get("alt_ops", ())), True)):
                if seq_ops and idx + len(seq_ops) <= n and tuple(normalized[idx:idx + len(seq_ops)]) == seq_ops:
                    span_durs = durations[idx:idx + len(seq_ops)]
                    eval_kwargs = dict(kwargs)
                    if all(d is not None for d in span_durs):
                        eval_kwargs["op_durations_us"] = [*span_durs, 0.0] if pad_zero else span_durs
                    matched_eval = evaluate_canonical_fusion_opportunity(pat_name, shape, **eval_kwargs)
                    matched_eval["span"] = [idx, idx + len(seq_ops)]
                    step = len(seq_ops)
                    break
            if matched_eval is not None:
                break
        if matched_eval is not None:
            matches.append(matched_eval)
        idx += step
    total_gm = sum(float(m["eliminated_gm_roundtrip_bytes"] or 0.0) for m in matches if m["ub_l1_feasible"])
    total_launches = sum(int(m["kernel_launch_reduction"]) for m in matches if m["ub_l1_feasible"])
    all_have_savings = bool(matches) and all(m["latency_savings_us_interval"] is not None for m in matches)
    total_max_us = sum(float(m["latency_savings_us_interval"][1]) for m in matches) if all_have_savings else None
    return {
        "status": "modeled" if matches else "no_matches", "match_count": len(matches), "matches": matches,
        "total_eliminated_gm_bytes": total_gm, "total_kernel_launch_reduction": total_launches,
        "total_latency_savings_us_interval": [0.0, total_max_us] if total_max_us is not None else None,
    }


__all__ = [
    "SubgraphError", "DEFAULT_CAPACITIES_KIB", "CANONICAL_FUSION_PATTERNS",
    "analyze_subgraph_fusion", "evaluate_canonical_fusion_opportunity", "scan_fusible_subgraph_patterns",
]
