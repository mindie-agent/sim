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
from typing import Any, Literal

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

_DOWN_OVERLAP_COLUMN_ALIGNMENT = 64
_VOLUME_EPS = 1e-12
_INT_ROW_TOL = 1e-9
_EXPLICIT_ROUTING_BASES = frozenset({
    "explicit_token_routing_integer",
    "source_force_eplb_masked_integer_counts",
})


def down_overlap_columns(width: int, overlap: float) -> tuple[int, int]:
    """Aligned output-column cut shared by expert_down and early/late combine."""
    n = int(width)
    if n < 2 or overlap <= 0.0:
        return n, 0
    raw_n1 = max(1, min(n - 1, round(n * overlap)))
    alignment = _DOWN_OVERLAP_COLUMN_ALIGNMENT
    n1 = raw_n1
    if n >= 2 * alignment and n % alignment == 0:
        n1 = max(alignment, min(n - alignment, raw_n1 // alignment * alignment))
    return n1, n - n1


def _hypergeometric_node_hit_prob(total_items: int, node_items: int, top_k: int) -> float:
    """Exact P(at least one of top_k distinct draws hits a node of size node_items)."""
    k = max(0, int(top_k))
    if k <= 0 or node_items <= 0 or total_items <= 0:
        return 0.0
    other = total_items - node_items
    if k > other or total_items <= node_items:
        return 1.0
    return 1.0 - (math.comb(other, k) / math.comb(total_items, k))


def _rank_expert_counts(ep: int, num_experts: int | None = None) -> tuple[list[int], int]:
    """Return ``(rank_experts, total_experts)`` matching contiguous ``expert_slot_plan``."""
    size = int(ep)
    total_e = int(num_experts) if (num_experts is not None and int(num_experts) > 0) else size
    if total_e == size:
        return [1] * size, total_e
    base, rem = divmod(total_e, size)
    return [base + (1 if r < rem else 0) for r in range(size)], total_e


def _node_rank_and_expert_sizes(
    ep: int, ranks_per_node: int, num_experts: int | None = None,
) -> tuple[list[int], list[int], int]:
    """Return (rank_sizes_per_node, expert_counts_per_node, total_experts)."""
    rpn = max(1, min(int(ep), int(ranks_per_node)))
    nodes = max(1, math.ceil(int(ep) / rpn))
    rank_sizes = [min(rpn, int(ep) - idx * rpn) for idx in range(nodes)]
    rank_experts, total_e = _rank_expert_counts(ep, num_experts=num_experts)
    if total_e == int(ep):
        return rank_sizes, list(rank_sizes), total_e
    node_experts: list[int] = []
    offset = 0
    for size in rank_sizes:
        node_experts.append(sum(rank_experts[offset:offset + size]))
        offset += size
    return rank_sizes, node_experts, total_e


def _expected_remote_nodes_hit(
    ep: int, ranks_per_node: int, top_k: int, *, num_experts: int | None = None,
) -> float:
    """Expected distinct remote nodes hit per token under uniform top-k expert draws."""
    if ep <= 1 or top_k <= 0:
        return 0.0
    rank_sizes, node_experts, total_e = _node_rank_and_expert_sizes(
        ep, ranks_per_node, num_experts=num_experts,
    )
    if len(rank_sizes) <= 1:
        return 0.0
    hit_probs = [_hypergeometric_node_hit_prob(total_e, e_n, top_k) for e_n in node_experts]
    total_hit = math.fsum(hit_probs)
    return math.fsum(
        (size / float(ep)) * (total_hit - prob)
        for size, prob in zip(rank_sizes, hit_probs)
    )


def _expected_intra_hops_per_assignment(
    ep: int, ranks_per_node: int, *, num_experts: int | None = None,
) -> float:
    """Exact probability an assignment requires an intra-node hop across (possibly uneven) nodes."""
    if ep <= 1:
        return 0.0
    rank_sizes, node_experts, total_e = _node_rank_and_expert_sizes(
        ep, ranks_per_node, num_experts=num_experts,
    )
    if total_e <= 0:
        return 0.0
    return math.fsum(
        (float(e_n) / float(total_e)) * (float(s_n - 1) / float(s_n))
        for s_n, e_n in zip(rank_sizes, node_experts)
        if s_n > 1
    )


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
                        eplb_peak_to_avg: float = 1.0,
                        num_experts: int | None = None) -> float:
    """Inter-node group volume with exact hypergeometric remote-node deduplication."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    rpn = max(1, int(ranks_per_node))
    nodes = max(1, math.ceil(ep / rpn))
    if nodes <= 1 or ep <= 1 or tokens <= 0:
        return 0.0
    expected_remote = _expected_remote_nodes_hit(
        ep, rpn, top_k, num_experts=num_experts,
    )
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * expected_remote * ratio)


def deepep_intra_volume(tokens: float, hidden_size: int, top_k: int, *,
                        ep: int, ranks_per_node: int, dtype_bytes_val: float,
                        eplb_peak_to_avg: float = 1.0,
                        num_experts: int | None = None) -> float:
    """Intra-node scatter volume across (possibly uneven) node groups."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    if ep <= 1 or tokens <= 0:
        return 0.0
    hop_prob = _expected_intra_hops_per_assignment(
        ep, ranks_per_node, num_experts=num_experts,
    )
    if hop_prob <= 0.0:
        return 0.0
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * float(top_k) * hop_prob * ratio)


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
    parallel = getattr(runtime, "parallel", None)
    ep = int(getattr(parallel, "ep", 0) or 0) if parallel is not None else 0
    return {
        "eplb": _validate_eplb(eplb),
        "deepep": cfg.get("mode") == "deepep" or bool(metadata.get("enable_deepep", False)),
        "overlap": _validate_overlap(overlap),
        "ranks_per_node": ranks,
        **({"ep": ep} if ep > 0 else {}),
    }


def deepep_exchange_factor(*, options, ep, top_k, tokens, hidden, dtype_bytes_val,
                           num_experts: int | None = None):
    """DeepEP group+intra deduplication ratio over a flat AllToAll (bytes conserved)."""
    if not options["deepep"]:
        return 1.0
    ranks_per_node = options["ranks_per_node"] or ep
    if ep <= ranks_per_node:
        return 1.0
    eff_tokens = max(1.0, float(tokens))
    common = dict(ep=ep, ranks_per_node=ranks_per_node,
                  dtype_bytes_val=dtype_bytes_val, eplb_peak_to_avg=1.0,
                  num_experts=num_experts)
    group = deepep_group_volume(eff_tokens, hidden, top_k, **common)
    intra = deepep_intra_volume(eff_tokens, hidden, top_k, **common)
    standard = standard_alltoall_volume(
        eff_tokens, hidden, top_k, ep=ep, dtype_bytes_val=dtype_bytes_val,
        eplb_peak_to_avg=1.0)
    return (group + intra) / max(standard, _VOLUME_EPS)


def _deepep_peer_traffic_matrix(total_bytes: float, *, ep: int,
                                ranks_per_node: int, top_k: int,
                                num_experts: int | None = None) -> dict[str, Any]:
    """Exact expected relay peer matrix matching the average of exact DeepEP routes."""
    size = int(ep)
    rpn = max(1, min(size, int(ranks_per_node)))
    rank_sizes, node_experts, total_e = _node_rank_and_expert_sizes(
        size, rpn, num_experts=num_experts,
    )
    nodes = len(rank_sizes)
    w_intra = float(top_k) * _expected_intra_hops_per_assignment(
        size, rpn, num_experts=num_experts,
    )
    hit_probs = [_hypergeometric_node_hit_prob(total_e, e_n, top_k) for e_n in node_experts]
    w_group = (
        _expected_remote_nodes_hit(size, rpn, top_k, num_experts=num_experts)
        if nodes > 1 else 0.0
    )
    w_total = max(w_intra + w_group, _VOLUME_EPS)
    total_all_ranks = float(total_bytes) * float(size)
    intra_total = total_all_ranks * (w_intra / w_total)
    inter_total = total_all_ranks - intra_total
    cells = [[0.0] * size for _ in range(size)]
    intra_weight_sum = math.fsum(
        float(e_n) * float(s_n - 1) / float(s_n)
        for s_n, e_n in zip(rank_sizes, node_experts) if s_n > 1
    )
    inter_weight_sum = math.fsum(
        float(rank_sizes[s_node]) * hit_probs[d_node]
        for s_node in range(nodes)
        for d_node in range(nodes) if d_node != s_node
    )
    rank_experts, _ = _rank_expert_counts(size, num_experts=num_experts)
    for src in range(size):
        src_node = src // rpn
        lane = src % rpn
        s_n = rank_sizes[src_node]
        e_n = node_experts[src_node]
        if s_n > 1 and e_n > 0 and intra_weight_sum > 0.0:
            node_intra = intra_total * ((float(e_n) * float(s_n - 1) / float(s_n)) / intra_weight_sum)
            denom = float((s_n - 1) * e_n)
            node_start = src_node * rpn
            for dst in range(node_start, node_start + s_n):
                if dst != src and rank_experts[dst] > 0:
                    cells[src][dst] = node_intra * (float(rank_experts[dst]) / denom)
        if nodes > 1 and inter_weight_sum > 0.0:
            for dst_node in range(nodes):
                if dst_node == src_node or hit_probs[dst_node] <= 0.0:
                    continue
                relay = min(size - 1, dst_node * rpn + lane)
                if relay != src:
                    cells[src][relay] = inter_total * (hit_probs[dst_node] / inter_weight_sum)
    return {
        "kind": "matrix",
        "payload_bytes": tuple(tuple(row) for row in cells),
        "direction": "forward",
    }


def _matrix_link_bytes(matrix, *, peer_volume_basis: str, rank: int) -> float:
    """Derive scalar link_bytes matching peer_volume_basis from a peer matrix."""
    size = len(matrix)
    if size == 0:
        return 0.0
    if peer_volume_basis == "group_mean":
        return math.fsum(float(cell) for row in matrix for cell in row) / float(size)
    idx = rank if 0 <= rank < size else 0
    return math.fsum(float(cell) for cell in matrix[idx])


def _extract_exact_token_route(metadata: Mapping[str, Any]):
    """Read token-level (source_owners, topk_ids, expert_placement) if declared."""
    route = metadata.get("moe_token_route")
    if isinstance(route, Mapping):
        owners = route.get("source_owners")
        topk = route.get("topk_ids")
        placement = route.get("expert_placement")
        if owners is not None and topk is not None and placement is not None:
            return owners, topk, placement
    mask = metadata.get("source_token_mask")
    if isinstance(mask, Mapping):
        owners = mask.get("source_owners")
        topk = mask.get("topk_ids")
        placement = mask.get("expert_placement")
        if owners is not None and topk is not None and placement is not None:
            return owners, topk, placement
    return None


def _transpose_matrix(cells: list[list[float]]) -> tuple[tuple[float, ...], ...]:
    size = len(cells)
    return tuple(tuple(cells[dst][src] for dst in range(size)) for src in range(size))


def _deepep_exact_stage_matrices(
    source_owners, topk_ids, expert_placement, *,
    ep: int, ranks_per_node: int, row_bytes: float, is_combine: bool,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...] | None]:
    """Build exact causal stage matrices: (single, None) or (stage1, stage2) for relays."""
    size = int(ep)
    rpn = max(1, min(size, int(ranks_per_node)))
    inter = [[0.0] * size for _ in range(size)]
    intra = [[0.0] * size for _ in range(size)]
    has_two_hop = False
    for src_raw, ids in zip(source_owners, topk_ids):
        src = int(src_raw)
        src_node = src // rpn
        by_node: dict[int, list[int]] = {}
        for expert in ids:
            dst = int(expert_placement[int(expert)][0])
            by_node.setdefault(dst // rpn, []).append(dst)
        for dst_node, dests in by_node.items():
            if dst_node == src_node:
                for dst in dests:
                    if dst != src:
                        intra[src][dst] += float(row_bytes)
            else:
                relay = min(size - 1, dst_node * rpn + (src % rpn))
                if relay != src:
                    inter[src][relay] += float(row_bytes)
                for dst in dests:
                    if dst != relay:
                        intra[relay][dst] += float(row_bytes)
                        if relay != src:
                            has_two_hop = True
    if not has_two_hop:
        combined = [
            [inter[s][d] + intra[s][d] for d in range(size)]
            for s in range(size)
        ]
        if is_combine:
            return _transpose_matrix(combined), None
        return tuple(tuple(row) for row in combined), None
    if is_combine:
        # Reverse causal order: intra-node gather into relay first, then inter-node return.
        return _transpose_matrix(intra), _transpose_matrix(inter)
    # Forward causal order: inter-node transfer to relay first, then intra-node fanout.
    return tuple(tuple(row) for row in inter), tuple(tuple(row) for row in intra)


def _resolve_exact_row_bytes(op, metadata: Mapping[str, Any]) -> float | None:
    row_bytes = metadata.get("row_bytes", metadata.get("payload_row_bytes"))
    if row_bytes is not None:
        return float(row_bytes)
    if op.local_shape and op.dtype:
        return float(int(op.local_shape[-1]) * dtype_bytes(op.dtype))
    return None


def _exact_stage_op(
    op, matrix, *, metadata: dict[str, Any], options: Mapping[str, Any],
    orig_bytes: float, suffix: str = "", deps: tuple[str, ...] | None = None,
    stage_role: str = "",
):
    basis = str(metadata.get("peer_volume_basis") or "")
    rank = int(metadata.get("peer_source_rank", 0) or 0)
    scaled_bytes = _matrix_link_bytes(matrix, peer_volume_basis=basis, rank=rank)
    stage_meta = dict(metadata)
    traffic = stage_meta.get("peer_traffic") or {"kind": "matrix"}
    stage_meta["peer_traffic"] = {**traffic, "payload_bytes": matrix}
    stage_meta["eplb_peak_to_avg"] = options["eplb"]
    if suffix and "collective_id" in stage_meta:
        stage_meta["collective_id"] = str(stage_meta["collective_id"]) + suffix
    if stage_role:
        stage_meta["deepep_relay_stage"] = stage_role
    if options["deepep"]:
        stage_meta["deepep_mode"] = True
        stage_meta["deepep_overlap_ratio"] = options["overlap"]
        stage_meta["deepep_dedup_scale"] = (
            scaled_bytes / orig_bytes if orig_bytes > 0.0 else 1.0
        )
    return replace(
        op,
        op_id=(op.op_id + suffix) if suffix else op.op_id,
        deps=op.deps if deps is None else deps,
        link_bytes=scaled_bytes,
        metadata=tuple(stage_meta.items()),
    )


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


def _continuous_hypergeom_hit_prob(total_items: float, hit_items: float, top_k: int) -> float:
    """Continuous generalization of ``1 - C(E - m, k) / C(E, k)`` for expected node hits."""
    k = max(0, int(top_k))
    if k <= 0 or hit_items <= 0.0 or total_items <= 0.0:
        return 0.0
    if k == 1:
        return min(1.0, hit_items / total_items)
    other = total_items - hit_items
    if other < float(k) or total_items <= hit_items:
        return 1.0
    log_ratio = (
        math.lgamma(other + 1.0) - math.lgamma(other - float(k) + 1.0)
        - math.lgamma(total_items + 1.0) + math.lgamma(total_items - float(k) + 1.0)
    )
    return max(0.0, min(1.0, -math.expm1(log_ratio)))


def _deepep_matrix_stage_matrices(
    matrix, *, ep: int, ranks_per_node: int, top_k: int, is_combine: bool,
    eplb: float = 1.0, num_experts: int | None = None,
    row_bytes: float | None = None, valid_per_rank: tuple[int, ...] = (),
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...] | None]:
    """Decompose a directed peer matrix into causal DeepEP inter/intra relay stages."""
    size = min(int(ep), len(matrix))
    rpn = max(1, min(size, int(ranks_per_node)))
    rank_sizes, node_experts, total_e = _node_rank_and_expert_sizes(
        size, rpn, num_experts=num_experts,
    )
    nodes = len(rank_sizes)
    fwd = [
        [float(matrix[d][s] if is_combine else matrix[s][d]) for d in range(size)]
        for s in range(size)
    ]
    inter = [[0.0] * size for _ in range(size)]
    intra = [[0.0] * size for _ in range(size)]
    has_two_hop = False
    for src in range(size):
        src_node = src // rpn
        lane = src % rpn
        src_tokens = (
            float(valid_per_rank[src])
            if 0 <= src < len(valid_per_rank) and valid_per_rank[src] > 0
            else 0.0
        )
        for dst_node in range(nodes):
            node_start = dst_node * rpn
            node_end = min(size, node_start + rank_sizes[dst_node])
            if dst_node == src_node:
                for dst in range(node_start, node_end):
                    if dst != src and fwd[src][dst] > 0.0:
                        intra[src][dst] += fwd[src][dst] * eplb
                continue
            node_bytes = math.fsum(fwd[src][dst] for dst in range(node_start, node_end))
            if node_bytes <= 0.0:
                continue
            relay = min(size - 1, node_start + lane)
            max_dest_bytes = max(fwd[src][dst] for dst in range(node_start, node_end))
            min_relay_bytes = (
                min(max_dest_bytes, src_tokens * float(row_bytes))
                if (src_tokens > 0.0 and row_bytes and row_bytes > 0.0)
                else max_dest_bytes
            )
            if top_k <= 1:
                inter_bytes = node_bytes
            elif src_tokens > 0.0 and row_bytes and row_bytes > 0.0:
                expected_hits = node_bytes / (src_tokens * float(row_bytes))
                eff_node_e = min(float(total_e), max(0.0, expected_hits * float(total_e) / float(top_k)))
                p_hit = _continuous_hypergeom_hit_prob(float(total_e), eff_node_e, top_k)
                inter_bytes = min(node_bytes, src_tokens * float(row_bytes) * p_hit)
            else:
                e_n = node_experts[dst_node]
                uniform_hits = float(top_k) * float(e_n) / float(max(1, total_e))
                dedup = (
                    _hypergeometric_node_hit_prob(total_e, e_n, top_k) / uniform_hits
                    if uniform_hits > 0.0 else 1.0
                )
                inter_bytes = node_bytes * min(1.0, dedup)
            inter_bytes = max(inter_bytes, min_relay_bytes)
            if relay != src:
                inter[src][relay] += inter_bytes * eplb
            for dst in range(node_start, node_end):
                if dst != relay and fwd[src][dst] > 0.0:
                    intra[relay][dst] += fwd[src][dst] * eplb
                    if relay != src:
                        has_two_hop = True
    if not has_two_hop:
        combined = [
            [inter[s][d] + intra[s][d] for d in range(size)]
            for s in range(size)
        ]
        if is_combine:
            return _transpose_matrix(combined), None
        return tuple(tuple(row) for row in combined), None
    if is_combine:
        return _transpose_matrix(intra), _transpose_matrix(inter)
    return tuple(tuple(row) for row in inter), tuple(tuple(row) for row in intra)


def scale_exchange_stages(
    op, factor, options, *, ep: int = 1, top_k: int = 1,
    is_combine: bool = False, num_experts: int | None = None,
) -> tuple[Any, ...]:
    """Scale exchange traffic and split causal two-stage DeepEP relays when needed."""
    if factor == 1.0 and not options["deepep"] and options["eplb"] == 1.0:
        return (op,)
    metadata = dict(op.metadata)
    traffic = metadata.get("peer_traffic")
    has_peer_matrix = isinstance(traffic, Mapping) and traffic.get("kind") == "matrix"
    is_exact_matrix = (
        has_peer_matrix
        and (metadata.get("routing_basis") in _EXPLICIT_ROUTING_BASES
             or metadata.get("traffic_model") in _EXPLICIT_ROUTING_BASES)
    )
    rpn = int(options.get("ranks_per_node", 0) or 0)
    if is_exact_matrix or (has_peer_matrix and options["deepep"] and 0 < rpn < ep):
        orig_bytes = float(op.link_bytes)
        stage1 = traffic["payload_bytes"]
        stage2 = None
        if options["deepep"] and 0 < rpn < ep:
            row_bytes = _resolve_exact_row_bytes(op, metadata)
            exact_route = _extract_exact_token_route(metadata) if is_exact_matrix else None
            if exact_route is not None and row_bytes is not None:
                owners, topk, placement = exact_route
                stage1, stage2 = _deepep_exact_stage_matrices(
                    owners, topk, placement, ep=ep, ranks_per_node=rpn,
                    row_bytes=row_bytes, is_combine=is_combine,
                )
            elif not is_exact_matrix:
                raw_mask = metadata.get("source_token_mask")
                valid_per_rank = (
                    tuple(int(v) for v in raw_mask.get("valid_tokens_per_rank", ()))
                    if isinstance(raw_mask, Mapping) else ()
                )
                stage1, stage2 = _deepep_matrix_stage_matrices(
                    stage1, ep=ep, ranks_per_node=rpn, top_k=top_k,
                    is_combine=is_combine, eplb=float(options["eplb"]),
                    num_experts=num_experts, row_bytes=row_bytes,
                    valid_per_rank=valid_per_rank,
                )
        if stage2 is not None:
            first_suffix = ".intra" if is_combine else ".inter"
            first_role = "intra_gather" if is_combine else "inter_relay"
            second_role = "inter_return" if is_combine else "intra_fanout"
            first_op = _exact_stage_op(
                op, stage1, metadata=metadata, options=options,
                orig_bytes=orig_bytes, suffix=first_suffix, stage_role=first_role,
            )
            second_op = _exact_stage_op(
                op, stage2, metadata=metadata, options=options,
                orig_bytes=orig_bytes,
                deps=tuple(dict.fromkeys((*op.deps, first_op.op_id))),
                stage_role=second_role,
            )
            return (first_op, second_op)
        single = _exact_stage_op(
            op, stage1, metadata=metadata, options=options, orig_bytes=orig_bytes,
        )
        return (single,)
    scaled_bytes = op.link_bytes * factor
    if "peer_traffic" in metadata:
        metadata["peer_traffic"] = _scale_matrix(metadata["peer_traffic"], factor)
    elif (options["deepep"] and not is_combine
          and options["ranks_per_node"] > 0 and ep > options["ranks_per_node"]
          and ep % options["ranks_per_node"] == 0):
        metadata["peer_traffic"] = _deepep_peer_traffic_matrix(
            scaled_bytes, ep=ep, ranks_per_node=options["ranks_per_node"],
            top_k=top_k, num_experts=num_experts,
        )
        metadata.setdefault("peer_volume_basis", "group_mean")
    metadata["eplb_peak_to_avg"] = options["eplb"]
    if options["deepep"]:
        metadata["deepep_mode"] = True
        metadata["deepep_overlap_ratio"] = options["overlap"]
        metadata["deepep_dedup_scale"] = factor / max(options["eplb"], _VOLUME_EPS)
    return (replace(op, link_bytes=scaled_bytes, metadata=tuple(metadata.items())),)


def scale_exchange_operator(op, factor, options, *, ep: int = 1,
                            top_k: int = 1, is_combine: bool = False,
                            num_experts: int | None = None):
    """Single-op wrapper over :func:`scale_exchange_stages` for direct callers."""
    stages = scale_exchange_stages(
        op, factor, options, ep=ep, top_k=top_k, is_combine=is_combine,
        num_experts=num_experts,
    )
    return stages[-1]


def _scale_op_portion(op, ratio: float, *, suffix: str = "", stream: str | None = None,
                      deps: tuple[str, ...] | None = None,
                      peer_traffic=None, extra_metadata=(),
                      suffix_reverse: bool = True,
                      link_bytes_override: float | None = None):
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
        link_bytes=(float(link_bytes_override) if link_bytes_override is not None
                    else op.link_bytes * ratio),
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


def split_peer_traffic_rows(traffic, *, rows: int, first: int, row_bytes: float,
                            overlap: float | None = None):
    """Split directed payload cells by complete assignment rows (or continuous ratio).

    Uniform peers and two-tier source totals are allocated in source order.
    An explicit integer matrix allocates each destination column by complete
    integer rows so a single row is never split fractionally; fractional
    statistical matrix columns split proportionally with zero dropped bytes.
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
        keep_fraction = (
            (1.0 - float(overlap))
            if overlap is not None
            else (first / float(rows) if rows else 0.0)
        )
        for destination in range(size):
            column = [float(matrix[source][destination])
                      for source in range(size)]
            col_total = math.fsum(column)
            if col_total <= 0.0:
                continue
            column_rows = col_total / float(row_bytes) if row_bytes else 0.0
            column_int = int(round(column_rows))
            is_integer_col = (
                bool(row_bytes)
                and column_int > 0
                and math.isclose(column_rows, float(column_int),
                                 rel_tol=0.0, abs_tol=_INT_ROW_TOL)
                and all(
                    math.isclose(
                        cell / float(row_bytes),
                        float(round(cell / float(row_bytes))),
                        rel_tol=0.0, abs_tol=_INT_ROW_TOL,
                    )
                    for cell in column
                )
            )
            if is_integer_col:
                if column_int == 1:
                    cut = 1
                else:
                    cut = int(math.floor(column_int * keep_fraction + 0.5))
                    cut = max(1, min(column_int - 1, cut))
                before, after = _allocate_rows(column, row_bytes=row_bytes,
                                               first=cut, total_rows=column_int)
            else:
                before = [cell * keep_fraction for cell in column]
                after = [cell - b for cell, b in zip(column, before)]
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
    basis.  Cohort-wide explicit matrices split strictly from cohort-wide
    column counts and ``overlap`` so every rank declares identical peer traffic.
    """
    overlap = float(options.get("overlap", 0.0))
    if not options.get("deepep") or overlap <= 0.0 or options.get("ep") == 1:
        return (op,), None
    metadata = dict(op.metadata)
    if rows is None:
        rows = metadata.get("payload_rows",
                            metadata.get("assignment_rows",
                                         op.local_shape[0] if op.local_shape else 0))
    rows = int(rows)
    if row_bytes is None:
        row_bytes = metadata.get("payload_row_bytes", metadata.get("row_bytes"))
    traffic = metadata.get("peer_traffic")
    is_matrix = isinstance(traffic, Mapping) and traffic.get("kind") == "matrix"
    is_explicit_matrix = (
        is_matrix
        and (metadata.get("routing_basis") in _EXPLICIT_ROUTING_BASES
             or metadata.get("traffic_model") in _EXPLICIT_ROUTING_BASES)
    )
    if is_explicit_matrix:
        eff_row_bytes = float(
            row_bytes if row_bytes is not None
            else (_resolve_exact_row_bytes(op, metadata) or 0.0)
        )
        main_traffic, tail_traffic = split_peer_traffic_rows(
            traffic, rows=rows, first=0, row_bytes=eff_row_bytes, overlap=overlap,
        )
        basis = str(metadata.get("peer_volume_basis") or "")
        rank = int(metadata.get("peer_source_rank", 0) or 0)
        main_link = _matrix_link_bytes(
            main_traffic["payload_bytes"], peer_volume_basis=basis, rank=rank,
        )
        tail_link = _matrix_link_bytes(
            tail_traffic["payload_bytes"], peer_volume_basis=basis, rank=rank,
        )
        common = (
            ("deepep_completion_basis", "cohort_integer_destination_rows"),
            ("deepep_transport_row_basis", "cohort_peer_matrix"),
        )
        main_op = _scale_op_portion(
            op, 1.0 - overlap, peer_traffic=main_traffic,
            link_bytes_override=main_link, extra_metadata=common,
        )
        over_op = _scale_op_portion(
            op, overlap, suffix="_overlap", stream="deepep_comm_stream",
            deps=(main_op.op_id,), peer_traffic=tail_traffic,
            link_bytes_override=tail_link, extra_metadata=common,
        )
        return (main_op, over_op), over_op.op_id
    if rows < 2:
        # A zero/one-row statistical edge has no integer second segment. Keep
        # the declared transport parts so every cohort rank exposes the same
        # completion identities; the gate itself still stays unsplit.
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
    if is_matrix:
        # Statistical peer matrix (e.g. two-stage DeepEP relay expectation):
        # scale proportionally by the declared row_cut fraction so fractional
        # expected rows are conserved exactly across main and overlap chunks.
        main_traffic = _scale_matrix(traffic, first / float(rows))
        tail_traffic = _scale_matrix(traffic, second / float(rows))
    else:
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


def _with_column_width(op, new_width: int, *, ratio: float):
    """Update a column-split combine op's declared tensor width to its chunk span."""
    meta = dict(op.metadata)
    local_shape = (
        (*op.local_shape[:-1], int(new_width))
        if op.local_shape else op.local_shape
    )
    sem_shape = (
        (*op.semantic_shape[:-1], int(new_width))
        if op.semantic_shape and len(op.semantic_shape) == len(op.local_shape)
        else op.semantic_shape
    )
    launch_shape = (
        (*op.launch_shape[:-1], int(new_width))
        if op.launch_shape and len(op.launch_shape) == len(op.local_shape)
        else op.launch_shape
    )
    if isinstance(meta.get("output_shape"), (list, tuple)) and meta["output_shape"]:
        meta["output_shape"] = (*tuple(meta["output_shape"])[:-1], int(new_width))
    if "payload_row_elements" in meta:
        meta["payload_row_elements"] = int(new_width)
    if "payload_row_bytes" in meta:
        meta["payload_row_bytes"] = float(meta["payload_row_bytes"]) * ratio
    if "payload_edge_bytes" in meta:
        meta["payload_edge_bytes"] = float(meta["payload_edge_bytes"]) * ratio
    return replace(
        op, local_shape=local_shape, semantic_shape=sem_shape,
        launch_shape=launch_shape, metadata=tuple(meta.items()),
    )


def split_deepep_combine_overlap(op, options, *, overlap_deps: tuple[str, ...],
                                 main_deps: tuple[str, ...],
                                 dispatch_overlap_id: str | None = None,
                                 column_split: tuple[int, int] | None = None):
    """Split combine along the actual aligned down-projection output columns."""
    overlap = float(options.get("overlap", 0.0))
    meta = dict(op.metadata)
    orig_reverse = meta.get("reverse_of_collective")
    if not options.get("deepep") or overlap <= 0.0 or options.get("ep") == 1:
        return (op,), op.op_id
    if column_split is not None:
        n1, n2 = int(column_split[0]), int(column_split[1])
        width = n1 + n2
    elif op.local_shape:
        width = int(op.local_shape[-1])
        n1, n2 = down_overlap_columns(width, overlap)
    else:
        width = n1 = n2 = 0
    if width < 2 or n1 < 1 or n2 < 1:
        if orig_reverse is not None and dispatch_overlap_id is not None:
            meta["reverse_of_collective_parts"] = (
                str(orig_reverse), str(dispatch_overlap_id),
            )
            updated = replace(
                op,
                deps=tuple(dict.fromkeys((*op.deps, dispatch_overlap_id))),
                metadata=tuple(meta.items()),
            )
            return (updated,), updated.op_id
        return (op,), op.op_id
    early_ratio = n1 / float(width)
    late_ratio = n2 / float(width)
    reverse_extra_early: tuple[tuple[str, Any], ...] = (
        ("deepep_column_cut", (n1, n2)),
        ("deepep_column_span", (0, n1)),
    )
    reverse_extra_late: tuple[tuple[str, Any], ...] = (
        ("deepep_column_cut", (n1, n2)),
        ("deepep_column_span", (n1, width)),
    )
    if orig_reverse is not None:
        parts = (
            (str(orig_reverse), str(dispatch_overlap_id))
            if dispatch_overlap_id is not None else (str(orig_reverse),)
        )
        reverse_extra_early += (
            ("reverse_of_collective", str(orig_reverse)),
            ("reverse_of_collective_parts", parts),
            ("reverse_column_fraction", early_ratio),
        )
        reverse_extra_late += (
            ("reverse_of_collective", str(orig_reverse)),
            ("reverse_of_collective_parts", parts),
            ("reverse_column_fraction", late_ratio),
        )
    over_op = _scale_op_portion(
        op, early_ratio, suffix="_overlap", stream="deepep_comm_stream",
        deps=overlap_deps or op.deps,
        suffix_reverse=False,
        extra_metadata=reverse_extra_early,
    )
    over_op = _with_column_width(over_op, n1, ratio=early_ratio)
    main_op = _scale_op_portion(
        op, late_ratio, deps=tuple(dict.fromkeys((*main_deps, over_op.op_id))),
        extra_metadata=reverse_extra_late,
    )
    main_op = _with_column_width(main_op, n2, ratio=late_ratio)
    return (over_op, main_op), main_op.op_id
