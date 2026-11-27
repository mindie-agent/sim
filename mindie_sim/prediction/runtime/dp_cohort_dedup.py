"""Symmetric homogeneous DP-cohort deduplication for multi-replica execution.

[Role] Detect provably symmetric homogeneous ``dp > 1`` cohorts in
  ``Execution.add_group`` / ``graph_costs``, execute one canonical DP replica
  while preserving exact cross-DP collective contention width (EP, DCP, PCP,
  Local-TP, MC2), and conserve cluster-wide work/service/timeline accounting.
[Invariants] Fail-closed: any rank-asymmetric workload skew, differing
  ``RequestView``/``OpSpec``/metadata (including ``peer_traffic``,
  ``source_force_eplb``, ``source_token_mask``), heterogeneous hardware die
  signature, asymmetric external release time, active memory ledger, offload
  policy, readiness/participant observation, stream plan, or bounded event
  trace disables deduplication and falls back transparently to full
  multi-replica simulation.  When deduplicated, per-step ``duration_ms`` and
  conserved cluster work/bytes match full multi-replica expansion to ``< 1e-9``.
[Timeouts & Locks] Pure bounded in-memory structural checks and dict scaling;
  no threads, locks, sleeps or I/O.
[Side-Effect Policy] Read-only over input ``IterationWork``/``OpSpec`` graphs;
  mutates only the owning ``Execution``/``Engine`` accounting structures on
  explicit post-pass calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import os
import re
from typing import Any

_ENV_TOGGLE = "MINDIE_SIM_DP_DEDUP"
_FALSE_TOKENS = frozenset({"0", "off", "false", "no"})
_SYMMETRIC_LINK_SEMANTICS = frozenset({"per_pair", "per_port", "fully_connected", "mesh"})
_PORT_SEMANTICS = frozenset({"per_port", "fully_connected", "mesh"})
_MIN_DEDUP_REPLICAS = 2
_CANONICAL_INDEX = 0
_BYTE_RECIPROCITY_REL_TOL = 1e-9
_BYTE_SIG_ROUND_DIGITS = 6
_NATIVE_MC2_FAMILIES = frozenset({"mc2_protocol", "mc2_transport"})
_RANK_LABEL_METADATA_KEYS = frozenset({"ep_rank", "dp_rank", "replica", "rank_id"})
_PAIR_RE = re.compile(r"^(?P<pfx>.+):pair(?P<src>\d+)>(?P<dst>\d+)$")
_PORT_RE = re.compile(r"^(?P<pfx>.+):(?P<dir>tx|rx):port(?P<rk>\d+)$")
_UB_INJ_RE = re.compile(r"^(?P<pfx>.*)ub_inj:(?P<dir>tx|rx):r(?P<rk>\d+)$")


def dp_dedup_enabled() -> bool:
    """True unless explicitly disabled via ``MINDIE_SIM_DP_DEDUP=0``."""
    raw = os.environ.get(_ENV_TOGGLE, "1").strip().lower()
    return raw not in _FALSE_TOKENS


def _norm_str(value: str, self_prefix: str, req_map: Mapping[str, str] | None) -> str:
    out = value.replace(self_prefix, "__self__:") if self_prefix and self_prefix in value else value
    if req_map:
        mapped = req_map.get(out)
        if mapped is not None:
            return mapped
        for raw_id, alias in req_map.items():
            if raw_id in out:
                out = out.replace(raw_id, alias)
    return out


def _values_symmetric(
    v0: Any,
    vr: Any,
    self0: str,
    req0: Mapping[str, str] | None,
    self_r: str,
    req_r: Mapping[str, str] | None,
) -> bool:
    """Zero-allocation recursive symmetry comparison across two DP replicas."""
    if v0 == vr:
        return True
    t0 = type(v0)
    tr = type(vr)
    if t0 is str and tr is str:
        return _norm_str(v0, self0, req0) == _norm_str(vr, self_r, req_r)
    if (t0 is tuple or t0 is list) and (tr is tuple or tr is list):
        if len(v0) != len(vr):
            return False
        for a, b in zip(v0, vr):
            if a != b and not _values_symmetric(a, b, self0, req0, self_r, req_r):
                return False
        return True
    if t0 is dict and tr is dict:
        if len(v0) != len(vr) or v0.keys() != vr.keys():
            return False
        for k, a in v0.items():
            b = vr[k]
            if a != b and not _values_symmetric(a, b, self0, req0, self_r, req_r):
                return False
        return True
    return _freeze_value(v0, self0, req0) == _freeze_value(vr, self_r, req_r)


def _freeze_value(
    value: Any,
    self_prefix: str = "",
    req_map: Mapping[str, str] | None = None,
) -> Any:
    kind = type(value)
    if kind is str:
        return _norm_str(value, self_prefix, req_map)
    if kind is int or kind is float or kind is bool or value is None:
        return value
    if kind is tuple or kind is list:
        return tuple(_freeze_value(item, self_prefix, req_map) for item in value)
    if kind is dict:
        return tuple(
            sorted((str(k), _freeze_value(v, self_prefix, req_map)) for k, v in value.items())
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(k), _freeze_value(v, self_prefix, req_map)) for k, v in value.items())
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze_value(item, self_prefix, req_map) for item in value))
    return value


def _replica_req_map(work: Any) -> dict[str, str]:
    reqs = getattr(work.plan, "requests", ()) or ()
    out: dict[str, str] = {}
    for idx, view in enumerate(reqs):
        rid = getattr(view, "request_id", None)
        if isinstance(rid, str) and rid:
            out[rid] = f"__req_{idx}__"
    return out


def _request_view_signature(view: Any) -> tuple[Any, ...]:
    """Structural signature of one RequestView ignoring its request_id."""
    if dataclasses.is_dataclass(view):
        return tuple(
            (f.name, _freeze_value(getattr(view, f.name)))
            for f in dataclasses.fields(view)
            if f.name != "request_id"
        )
    return (
        getattr(view, "q_len", None),
        getattr(view, "logical_q_len", None),
        getattr(view, "kv_len", None),
        getattr(view, "phase", None),
        getattr(view, "prefix_len", None),
    )


def _ops_symmetric(
    op0: Any,
    op_r: Any,
    self0: str,
    req0: Mapping[str, str],
    self_r: str,
    req_r: Mapping[str, str],
) -> bool:
    """Fast fail-closed structural equality of two OpSpecs across DP replicas."""
    if op0 == op_r:
        return True
    if (op0.op_id != op_r.op_id or op0.family != op_r.family
            or op0.stage != op_r.stage or op0.layer != op_r.layer
            or op0.deps != op_r.deps or op0.flops != op_r.flops
            or op0.matrix_flops != op_r.matrix_flops
            or op0.vector_ops != op_r.vector_ops
            or op0.hbm_bytes != op_r.hbm_bytes
            or op0.mte2_bytes != op_r.mte2_bytes
            or op0.mte3_bytes != op_r.mte3_bytes
            or op0.link_bytes != op_r.link_bytes
            or op0.link_scope != op_r.link_scope
            or op0.local_shape != op_r.local_shape
            or op0.padded_shape != op_r.padded_shape
            or op0.launch_shape != op_r.launch_shape
            or op0.layout != op_r.layout
            or op0.returns != op_r.returns
            or op0.dtype != op_r.dtype or op0.recipe != op_r.recipe
            or op0.stream != op_r.stream or op0.phase != op_r.phase
            or op0.source_boundary != op_r.source_boundary):
        return False
    if op0.semantic_shape != op_r.semantic_shape:
        if not _values_symmetric(
            op0.semantic_shape, op_r.semantic_shape, self0, req0, self_r, req_r,
        ):
            return False
    m0 = op0.metadata
    mr = op_r.metadata
    if m0 == mr:
        return True
    d0 = dict(m0) if m0 else {}
    dr = dict(mr) if mr else {}
    if d0.keys() != dr.keys():
        return False
    for key, v0 in d0.items():
        vr = dr[key]
        if v0 == vr:
            continue
        if (key == "group_ranks"
                and isinstance(v0, (tuple, list))
                and isinstance(vr, (tuple, list))):
            if len(v0) != len(vr):
                return False
            continue
        if (key in _RANK_LABEL_METADATA_KEYS
                and isinstance(v0, int) and not isinstance(v0, bool)
                and isinstance(vr, int) and not isinstance(vr, bool)):
            continue
        if not _values_symmetric(v0, vr, self0, req0, self_r, req_r):
            return False
    return True


def _stage_hardware_signature(execution: Any, replica: int) -> tuple[Any, ...]:
    """Exact per-stage die count and Engine resource tuple for one replica."""
    pp = max(1, int(execution.case.runtime.parallel.pp))
    engine = execution.engine
    stages = []
    for stage in range(pp):
        ranks = tuple(execution.topology.ranks(replica, stage))
        dies = execution.stage_dies.get((replica, stage), ())
        die_sigs = tuple(getattr(d, "resource_signature", None) for d in dies)
        stage_names = execution.stage_names.get((replica, stage), {})
        res_sigs = tuple(
            (
                role,
                engine.resources[rname].rate,
                engine.resources[rname].units,
                engine.resources[rname].policy,
            )
            for role, rname in sorted(stage_names.items())
            if rname in engine.resources
        )
        stages.append((len(ranks), die_sigs, res_sigs))
    return tuple(stages)


def _external_input_signature(engine: Any, mapping: Mapping | None) -> tuple[Any, ...]:
    if not mapping:
        return ()
    out = []
    for key in sorted(mapping):
        ev = mapping[key]
        if ev.time is not None:
            out.append((key, True, float(ev.time), 0.0))
        else:
            task = engine._tasks.get(ev.sequence)
            if task is None:
                out.append((key, "unresolved_promise", ev.sequence))
            elif (
                task.remaining
                or task.waiting
                or task.acquire
                or task.callback is not None
                or task.start_callback is not None
            ):
                out.append((key, "unresolved_task", ev.sequence))
            else:
                eff_start = float(task.start) if task.start is not None else float(task.release)
                out.append((key, False, eff_start, float(task.latency)))
    return tuple(out)


def _work_symmetry_signature(work: Any) -> tuple[Any, ...]:
    return (
        work.mode,
        work.recipe,
        work.replay_scope,
        work.stage_count,
        work.valid_tokens,
        work.padded_tokens,
        work.graph_bucket,
        work.local_tokens,
        len(work.ops),
        tuple(_request_view_signature(v) for v in work.plan.requests),
    )


def _rank_topology_profile(topology: Any, engine: Any, ranks: tuple[int, ...]) -> tuple[Any, ...]:
    """Per-replica interconnect profile verifying identical link rates, rails & fanout."""
    rank_set = set(ranks)
    entries = []
    for link in topology.links.values():
        touched = [rk for rk in link.members if rk in rank_set]
        if not touched:
            continue
        tx_rates = []
        rx_rates = []
        for rk in touched:
            tx_name = f"{topology.namespace}link:{link.name}:tx:port{rk}"
            rx_name = f"{topology.namespace}link:{link.name}:rx:port{rk}"
            tx_r = engine.resources.get(tx_name)
            rx_r = engine.resources.get(rx_name)
            tx_rates.append(None if tx_r is None else (tx_r.rate, tx_r.units, tx_r.policy))
            rx_rates.append(None if rx_r is None else (rx_r.rate, rx_r.units, rx_r.policy))
        rail_val = (
            topology.link_rail(link.name)
            if hasattr(topology, "link_rail")
            else getattr(link, "rail_id", None)
        )
        entries.append((
            link.semantics,
            str(getattr(link, "channel", "") or ""),
            repr(rail_val),
            len(link.members),
            len(touched),
            float(link.capacity_bytes_per_ms),
            float(getattr(link, "latency_ms", 0.0) or 0.0),
            tuple(sorted(tx_rates, key=repr)),
            tuple(sorted(rx_rates, key=repr)),
        ))
    entries.sort()
    return tuple(entries)


def _dag_reachability_masks(ops: Sequence[Any]) -> tuple[list[int], list[int]]:
    """Exact ancestor and descendant bitmasks for one replica's op DAG."""
    n = len(ops)
    id_to_idx = {op.op_id: idx for idx, op in enumerate(ops)}
    preds_by_idx: list[list[int]] = [[] for _ in range(n)]
    succs_by_idx: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    in_order = True
    for idx, op in enumerate(ops):
        for dep in op.deps:
            pred = id_to_idx.get(dep)
            if pred is None or pred == idx:
                continue
            preds_by_idx[idx].append(pred)
            succs_by_idx[pred].append(idx)
            indeg[idx] += 1
            if pred > idx:
                in_order = False
    if in_order:
        order = list(range(n))
    else:
        order = [idx for idx in range(n) if indeg[idx] == 0]
        head = 0
        while head < len(order):
            cur = order[head]
            head += 1
            for nxt in succs_by_idx[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    order.append(nxt)
        if len(order) < n:
            order = list(range(n))
    anc = [0] * n
    for idx in order:
        bits = 0
        for pred in preds_by_idx[idx]:
            bits |= anc[pred] | (1 << pred)
        anc[idx] = bits
    desc = [0] * n
    for idx in reversed(order):
        d_bits = desc[idx] | (1 << idx)
        for pred in preds_by_idx[idx]:
            desc[pred] |= d_bits
    return anc, desc


def _bytes_close(lhs: float, rhs: float) -> bool:
    return abs(lhs - rhs) <= _BYTE_RECIPROCITY_REL_TOL * max(1.0, abs(lhs), abs(rhs))


def _replica_pair_blocks_reciprocal(
    matrix: Sequence[Mapping[int, int]],
    byte_rows: Sequence[Mapping[int, float]],
    has_byte_matrix: bool,
    base1: int,
    base2: int,
    local_k: int,
) -> bool:
    """True when cross-replica directed block (r1 -> r2) matches (r2 -> r1) by local position."""
    for u0 in range(local_k):
        src12 = base1 + u0
        src21 = base2 + u0
        for v0 in range(local_k):
            dst12 = base2 + v0
            dst21 = base1 + v0
            if matrix[src12].get(dst12, 0) != matrix[src21].get(dst21, 0):
                return False
            if has_byte_matrix and not _bytes_close(
                byte_rows[src12].get(dst12, 0.0), byte_rows[src21].get(dst21, 0.0),
            ):
                return False
    return True


def _intra_replica_block_matches(
    matrix: Sequence[Mapping[int, int]],
    byte_rows: Sequence[Mapping[int, float]],
    has_byte_matrix: bool,
    base: int,
    local_k: int,
) -> bool:
    """True when replica ``base`` has the same intra-replica directed flows as replica 0."""
    for u0 in range(local_k):
        for v0 in range(local_k):
            if u0 == v0:
                continue
            if matrix[base + u0].get(base + v0, 0) != matrix[u0].get(v0, 0):
                return False
            if has_byte_matrix and not _bytes_close(
                byte_rows[base + u0].get(base + v0, 0.0), byte_rows[u0].get(v0, 0.0),
            ):
                return False
    return True


def _matrices_are_replica_reciprocal(
    matrix: Sequence[Mapping[int, int]],
    byte_rows: Sequence[Mapping[int, float]],
    has_byte_matrix: bool,
    width: int,
    local_k: int,
) -> bool:
    """True when cross-replica blocks are reciprocal, intra-blocks isomorphic, and per-rank totals match."""
    if width <= 0 or local_k <= 0 or width % local_k != 0:
        return False
    n_replicas = width // local_k
    for r1 in range(n_replicas):
        base1 = r1 * local_k
        for r2 in range(r1 + 1, n_replicas):
            if not _replica_pair_blocks_reciprocal(
                matrix, byte_rows, has_byte_matrix, base1, r2 * local_k, local_k,
            ):
                return False
    if local_k > 1:
        for rep in range(1, n_replicas):
            if not _intra_replica_block_matches(
                matrix, byte_rows, has_byte_matrix, rep * local_k, local_k,
            ):
                return False
    if n_replicas <= 1:
        return True
    col_f = [0] * width
    col_b = [0.0] * width if has_byte_matrix else None
    for src, row in enumerate(matrix):
        for dst, frames in row.items():
            if src != dst and frames > 0:
                col_f[dst] += frames
    if col_b is not None:
        for src, row_b in enumerate(byte_rows):
            for dst, b_val in row_b.items():
                if src != dst and b_val > 0.0:
                    col_b[dst] += b_val
    for pos in range(local_k):
        ref_rf, ref_cf = sum(matrix[pos].values()), col_f[pos]
        ref_rb = sum(byte_rows[pos].values()) if col_b is not None else 0.0
        ref_cb = col_b[pos] if col_b is not None else 0.0
        for rep in range(1, n_replicas):
            rk = rep * local_k + pos
            if sum(matrix[rk].values()) != ref_rf or col_f[rk] != ref_cf:
                return False
            if col_b is not None and not (
                _bytes_close(sum(byte_rows[rk].values()), ref_rb) and _bytes_close(col_b[rk], ref_cb)
            ):
                return False
    return True


def _raw_mc2_matrices_reciprocal(mc2_phase: Mapping[str, Any], ep_width: int, local_k: int) -> bool:
    """Pure read-only replica-block reciprocity and per-rank total check over raw ``mc2_phase`` metadata."""
    route_matrix = mc2_phase.get("route_matrix")
    route_bytes = mc2_phase.get("route_matrix_bytes")
    if not route_matrix and not route_bytes:
        return True
    if route_matrix and len(route_matrix) != ep_width:
        return False
    if route_bytes and len(route_bytes) != ep_width:
        return False
    reverse = bool(mc2_phase.get("route_reverse", False))
    matrix: list[dict[int, int]] = [{} for _ in range(ep_width)]
    if route_matrix:
        for u in range(ep_width):
            row = route_matrix[u]
            if len(row) != ep_width:
                return False
            for v in range(ep_width):
                val = int(route_matrix[v][u] if reverse else row[v])
                if u != v and val > 0:
                    matrix[u][v] = val
        n_replicas = ep_width // local_k
        for pos in range(local_k):
            ref_diag = int(route_matrix[pos][pos])
            for rep in range(1, n_replicas):
                rk = rep * local_k + pos
                if int(route_matrix[rk][rk]) != ref_diag:
                    return False
    byte_rows: list[dict[int, float]] = [{} for _ in range(ep_width)]
    if route_bytes:
        for u in range(ep_width):
            row_b = route_bytes[u]
            if len(row_b) != ep_width:
                return False
            for v in range(ep_width):
                b_val = float(row_b[v] or 0.0)
                if u != v and b_val > 0.0:
                    byte_rows[u][v] = b_val
    return _matrices_are_replica_reciprocal(
        matrix, byte_rows, bool(route_bytes), ep_width, local_k,
    )


def _mc2_op_can_run_canonical(
    execution: Any,
    op: Any,
    r0: int,
    source_stage: int,
    *,
    can_r0_only: bool,
    replicas: tuple[int, ...],
) -> bool:
    """True only when an MC2 op is guaranteed to execute canonical r0-only."""
    if not can_r0_only or len(replicas) < _MIN_DEDUP_REPLICAS:
        return False
    if not getattr(execution.topology, "links", None):
        return False
    if not hasattr(execution, "_verify_canonical_cohort_reciprocity"):
        return False
    meta = op.metadata
    mc2_phase = None
    if meta:
        items = meta.items() if isinstance(meta, Mapping) else meta
        for key, value in items:
            if key == "mc2_phase":
                mc2_phase = value
                break
    if not isinstance(mc2_phase, Mapping):
        return False
    r0_ranks = tuple(execution.topology.ranks(r0, source_stage))
    local_k = len(r0_ranks)
    ep_width = int(mc2_phase.get("ep", 0) or 0)
    if local_k <= 0 or ep_width <= 1 or ep_width % local_k != 0:
        return False
    peer_replicas = replicas
    ep_groups = tuple(execution.topology.groups("ep", r0, source_stage)) if hasattr(execution.topology, "groups") else ()
    r0_rank_set = set(r0_ranks)
    ep_group = next((grp for grp in ep_groups if r0_rank_set & set(grp)), ())
    if ep_group:
        ep_rank_set = {int(rk) for rk in ep_group}
        matched = tuple(
            r for r in replicas
            if any(int(rk) in ep_rank_set for rk in execution.topology.ranks(r, source_stage))
        )
        if matched:
            peer_replicas = matched
    if len(peer_replicas) * local_k != ep_width:
        return False
    for rep in peer_replicas:
        if len(execution.topology.ranks(rep, source_stage)) != local_k:
            return False
    return _raw_mc2_matrices_reciprocal(mc2_phase, ep_width, local_k)


def _has_unsynced_peer_die_overlap(
    execution: Any,
    w0: Any,
    r0: int,
    *,
    replicas: tuple[int, ...] = (),
    requires_peer_collectives: bool = False,
) -> bool:
    """True when multi-stream r0 ops can genuinely overlap synthesized peer cohorts.

    A multi-stream replica can still deduplicate bit-identically whenever every
    non-barrier op on another stream in the same PP stage is strictly ordered
    (ancestor or descendant in the op DAG) with every non-r0-only collective,
    or when an overlapping MC2 op is provably eligible for canonical r0-only
    cohort execution (zero synthesized peer replicas).
    """
    ops0 = w0.ops
    if len({op.stream for op in ops0}) <= 1:
        return False
    can_r0_only = can_dedup_collective_to_canonical(
        execution.topology, execution.engine,
        requires_peer_collectives=requires_peer_collectives,
    )
    stage_non_barrier: dict[int, int] = {}
    stage_stream_non_barrier: dict[tuple[int, Any], int] = {}
    collectives: list[tuple[int, Any, int, Any, bool, bool]] = []
    for idx, op in enumerate(ops0):
        if op.family != "barrier":
            bit = 1 << idx
            stage_non_barrier[op.stage] = stage_non_barrier.get(op.stage, 0) | bit
            ss_key = (op.stage, op.stream)
            stage_stream_non_barrier[ss_key] = stage_stream_non_barrier.get(ss_key, 0) | bit
        meta = op.metadata
        has_mc2_phase = False
        has_group_axis = False
        source_stage = op.stage
        if meta:
            items = meta.items() if isinstance(meta, Mapping) else meta
            for key, value in items:
                if key == "mc2_phase" and value:
                    has_mc2_phase = True
                elif key == "group_axis" and value is not None:
                    has_group_axis = True
                elif key == "source_stage" and value is not None:
                    source_stage = int(value)
        is_mc2 = op.family in _NATIVE_MC2_FAMILIES or has_mc2_phase
        if is_mc2 or has_group_axis:
            collectives.append((idx, op, source_stage, op.stream, is_mc2, has_group_axis))
    if not collectives:
        return False
    anc, desc = _dag_reachability_masks(ops0)
    for idx, op, source_stage, stream, is_mc2, has_group_axis in collectives:
        other_mask = (
            stage_non_barrier.get(source_stage, 0)
            & ~stage_stream_non_barrier.get((source_stage, stream), 0)
        )
        if not (other_mask & ~(anc[idx] | desc[idx] | (1 << idx))):
            continue
        if not can_r0_only and (is_mc2 or has_group_axis):
            return True
        if is_mc2 and not _mc2_op_can_run_canonical(
            execution, op, r0, source_stage,
            can_r0_only=can_r0_only, replicas=replicas,
        ):
            return True
    return False


def _has_megamoe_cross_dp_group(ops: Sequence[Any]) -> bool:
    """Fail-closed on fused MegaMoE ops whose peer RX links require full cohort."""
    for op in ops:
        meta = op.metadata
        if not meta:
            continue
        items = meta.items() if isinstance(meta, Mapping) else meta
        is_megamoe = False
        has_group_axis = False
        for key, value in items:
            if key == "megamoe_geometry" and value:
                is_megamoe = True
            elif key == "device_kernel_group" and isinstance(value, Mapping):
                if value.get("kernel") == "megamoe":
                    is_megamoe = True
            elif key == "group_axis" and value is not None:
                has_group_axis = True
        if is_megamoe and (has_group_axis or float(getattr(op, "link_bytes", 0.0) or 0.0) > 0.0):
            return True
    return False


@dataclasses.dataclass(frozen=True)
class CohortDedupEligibility:
    """Typed DP-cohort deduplication eligibility without global side-channels."""

    symmetric: bool
    requires_peer_collectives: bool = False
    reason: str = ""

    def __bool__(self) -> bool:
        return self.symmetric


def evaluate_dp_cohort_dedup(
    execution: Any,
    works: Mapping[int, Any],
    *,
    external_inputs: Mapping[int, Any] | None = None,
    emitted_tokens: Any = None,
    memory_bindings: Any = None,
    state_accesses: Any = None,
) -> CohortDedupEligibility:
    """Fail-closed typed check that all DP replicas in ``works`` are strictly symmetric."""
    if not dp_dedup_enabled() or len(works) < _MIN_DEDUP_REPLICAS:
        return CohortDedupEligibility(False, reason="disabled_or_singleton")
    engine = execution.engine
    if (execution.readiness_report or execution.participant_profile
            or execution.legacy_diagnostics
            or getattr(getattr(execution, "regimes", None), "rules", None)
            or getattr(execution, "stream_owners", None)
            or getattr(execution, "stream_quota_by_res", None)
            or getattr(execution, "stars_queues", None)
            or execution.offload_policy is not None
            or execution._joint_collective_trace is not None
            or engine.memory is not None or engine._memory_grants
            or engine._trace_maxlen > 0 or engine._wave_recorder is not None):
        return CohortDedupEligibility(False, reason="execution_state_active")
    if (emitted_tokens is not None or memory_bindings is not None
            or state_accesses is not None):
        return CohortDedupEligibility(False, reason="per_replica_io_bindings")

    replicas = tuple(sorted(works))
    r0 = replicas[_CANONICAL_INDEX]
    w0 = works[r0]
    if w0.mode != "graph":
        return CohortDedupEligibility(False, reason="non_graph_mode")
    requires_peer_collectives = _has_megamoe_cross_dp_group(w0.ops)
    if _has_unsynced_peer_die_overlap(
        execution, w0, r0, replicas=replicas,
        requires_peer_collectives=requires_peer_collectives,
    ):
        return CohortDedupEligibility(False, reason="unsynced_peer_die_overlap")

    w0_sig = _work_symmetry_signature(w0)
    hw0 = _stage_hardware_signature(execution, r0)
    ext0 = _external_input_signature(engine, (external_inputs or {}).get(r0))
    pp = max(1, int(execution.case.runtime.parallel.pp))
    ranks0 = tuple(
        rk for stg in range(pp) for rk in execution.topology.ranks(r0, stg)
    )
    topo0 = _rank_topology_profile(execution.topology, engine, ranks0)
    self0 = f"analysis{r0}:"
    req_map0 = _replica_req_map(w0)
    ops0 = w0.ops

    for rep in replicas[1:]:
        wr = works[rep]
        if _work_symmetry_signature(wr) != w0_sig:
            return CohortDedupEligibility(False, reason="work_signature_mismatch")
        if _stage_hardware_signature(execution, rep) != hw0:
            return CohortDedupEligibility(False, reason="hardware_signature_mismatch")
        if _external_input_signature(engine, (external_inputs or {}).get(rep)) != ext0:
            return CohortDedupEligibility(False, reason="external_input_mismatch")
        ranks_r = tuple(
            rk for stg in range(pp) for rk in execution.topology.ranks(rep, stg)
        )
        if _rank_topology_profile(execution.topology, engine, ranks_r) != topo0:
            return CohortDedupEligibility(False, reason="topology_profile_mismatch")
        rep_prefix = f"analysis{rep}:"
        req_map_r = _replica_req_map(wr)
        for op0, op_r in zip(ops0, wr.ops):
            if not _ops_symmetric(op0, op_r, self0, req_map0, rep_prefix, req_map_r):
                return CohortDedupEligibility(False, reason="op_asymmetry")
    return CohortDedupEligibility(
        True,
        requires_peer_collectives=requires_peer_collectives,
    )


def is_symmetric_dp_cohort(
    execution: Any,
    works: Mapping[int, Any],
    *,
    external_inputs: Mapping[int, Any] | None = None,
    emitted_tokens: Any = None,
    memory_bindings: Any = None,
    state_accesses: Any = None,
) -> bool:
    """Boolean convenience wrapper over ``evaluate_dp_cohort_dedup``."""
    return evaluate_dp_cohort_dedup(
        execution, works,
        external_inputs=external_inputs,
        emitted_tokens=emitted_tokens,
        memory_bindings=memory_bindings,
        state_accesses=state_accesses,
    ).symmetric


def _row_frame_map(raw: Any, peer_count: int) -> dict[int, int]:
    """Normalize a rank's outgoing directed frame map ``dst -> frames``."""
    out: dict[int, int] = {}
    items = raw.items() if isinstance(raw, Mapping) else (raw or ())
    for peer, frames in items:
        p_idx = int(peer)
        f_cnt = int(frames)
        if 0 <= p_idx < peer_count and f_cnt > 0:
            out[p_idx] = out.get(p_idx, 0) + f_cnt
    return out


def is_symmetric_native_program(
    program: Any,
    peer_count: int,
    *,
    local_rank_count: int = 1,
) -> bool:
    """True iff ``program`` is strictly reciprocal and isomorphic across all EP peers.

    Verifies full directed edge reciprocity ``M[u][v] == M[v][u]`` (rejecting
    directed cycles such as ``0->1->2->0`` that pass histogram-only checks),
    identical per-rank local frames/basis, and isomorphic intra/cross-replica
    slices when ``local_rank_count > 1``.
    """
    inner = getattr(program, "program", None)
    if inner is None or not hasattr(inner, "start_with_handles"):
        return False
    width = int(getattr(inner, "peer_count", 0) or 0)
    if width <= 0 or width != int(peer_count):
        return False
    local_k = max(1, int(local_rank_count))
    if width % local_k != 0:
        return False
    route_matrix = getattr(inner, "route_matrix", None)
    route_bytes = getattr(inner, "route_matrix_bytes", None)
    payload_bytes_fn = getattr(inner, "payload_bytes", None)
    has_byte_matrix = bool(route_bytes) or callable(payload_bytes_fn)
    if not route_matrix and not has_byte_matrix:
        return True
    frames_for = getattr(inner, "frames_for", None)
    basis_for = getattr(inner, "payload_basis_for", None)
    local_for = getattr(inner, "local_frames_for", None)
    if route_matrix and not callable(frames_for):
        return False

    matrix = (
        [_row_frame_map(frames_for(rk), width) for rk in range(width)]
        if route_matrix else [{} for _ in range(width)]
    )
    # 1) Replica-block reciprocity M[r1*K+u0][r2*K+v0] == M[r2*K+u0][r1*K+v0]
    #    (reduces to M[u][v] == M[v][u] when local_k == 1; rejects directed cycles).
    byte_rows: list[dict[int, float]] = [{} for _ in range(width)]
    if has_byte_matrix:
        for u in range(width):
            for v in range(width):
                if u == v:
                    continue
                if callable(payload_bytes_fn):
                    b_uv = float(payload_bytes_fn(u, v) or 0.0)
                else:
                    b_uv = float(route_bytes[u][v] or 0.0)
                if b_uv > 0.0:
                    byte_rows[u][v] = b_uv
    if not _matrices_are_replica_reciprocal(matrix, byte_rows, has_byte_matrix, width, local_k):
        return False

    # 2) Identical per-rank basis, local frames, and off-diagonal frame/byte histograms.
    def _rank_sig(rk: int) -> tuple[Any, ...]:
        rep_base = (rk // local_k) * local_k
        rep_end = rep_base + local_k
        intra = tuple(sorted(
            (dst - rep_base, f) for dst, f in matrix[rk].items()
            if dst != rk and rep_base <= dst < rep_end
        ))
        cross = tuple(sorted(
            (dst % local_k, f) for dst, f in matrix[rk].items()
            if dst != rk and not (rep_base <= dst < rep_end)
        ))
        intra_b = tuple(sorted(
            (dst - rep_base, round(b, _BYTE_SIG_ROUND_DIGITS)) for dst, b in byte_rows[rk].items()
            if dst != rk and rep_base <= dst < rep_end
        ))
        cross_b = tuple(sorted(
            (dst % local_k, round(b, _BYTE_SIG_ROUND_DIGITS)) for dst, b in byte_rows[rk].items()
            if dst != rk and not (rep_base <= dst < rep_end)
        ))
        basis = str(basis_for(rk)) if callable(basis_for) else ""
        loc = int(local_for(rk)) if callable(local_for) else 0
        return (basis, loc, intra, cross, intra_b, cross_b)

    n_replicas = width // local_k
    for pos in range(local_k):
        ref_sig = _rank_sig(pos)
        for rep in range(1, n_replicas):
            if _rank_sig(rep * local_k + pos) != ref_sig:
                return False
    return True


def can_dedup_collective_to_canonical(
    topology: Any,
    engine: Any,
    *,
    requires_peer_collectives: bool = False,
) -> bool:
    """True when all topology links are symmetric full-duplex and peer collectives are uncoupled."""
    if requires_peer_collectives:
        return False
    if getattr(topology, "_shared_switches", None):
        return False
    if getattr(topology, "_link_shared_ports", None):
        return False
    ns = topology.namespace
    for link in topology.links.values():
        if link.semantics not in _SYMMETRIC_LINK_SEMANTICS:
            return False
        if link.semantics in _PORT_SEMANTICS:
            for rk in link.members:
                tx = engine.resources.get(f"{ns}link:{link.name}:tx:port{rk}")
                rx = engine.resources.get(f"{ns}link:{link.name}:rx:port{rk}")
                if tx is not None and rx is not None and tx.rate != rx.rate:
                    return False
    return True


def mirror_deduplicated_step(step0: Any, replica: int, work: Any) -> Any:
    """Build a peer replica's ScheduledIteration sharing canonical events."""
    req_ids_0 = [v.request_id for v in work.plan.requests]

    def _remap_events(mapping: Mapping[str, Any]) -> dict[str, Any]:
        if not mapping:
            return {}
        vals = list(mapping.values())
        if len(vals) == len(req_ids_0):
            return dict(zip(req_ids_0, vals))
        return dict(mapping)

    return dataclasses.replace(
        step0,
        replica=replica,
        commits=_remap_events(step0.commits),
        host_results=_remap_events(step0.host_results),
        client_deliveries=_remap_events(step0.client_deliveries),
        declared_commits=_remap_events(step0.declared_commits),
        fallback_commits=_remap_events(step0.fallback_commits),
    )


def scale_dedup_accounting(
    execution: Any,
    base_work: Mapping[str, float],
    base_recipes: Mapping[str, int],
    base_cost_counts: Mapping[str, int],
    factor: int,
) -> None:
    """Conserve cluster-wide aggregate work, recipe counts, and cost counts."""
    if factor <= 1:
        return
    extra = factor - 1
    for key, before in base_work.items():
        delta = execution.work[key] - before
        if delta:
            execution.work[key] += delta * extra
    for key, cur in list(execution.recipes_used.items()):
        delta = cur - base_recipes.get(key, 0)
        if delta:
            execution.recipes_used[key] = cur + delta * extra
    for key, cur in list(execution.cost.counts.items()):
        delta = cur - base_cost_counts.get(key, 0)
        if delta:
            execution.cost.counts[key] = cur + delta * extra


def _build_link_mirror_classes(
    topology: Any,
    cohort_ranks: frozenset[int],
) -> dict[str, tuple[str, str, str, str, int, str]]:
    """Map each topology link name to its symmetric mirror equivalence class."""
    by_partition: dict[tuple[frozenset[int], str, str, str, str], list[str]] = {}
    meta_by_name: dict[str, tuple[frozenset[int], str, str, str, str]] = {}
    for link_name, link in topology.links.items():
        channel = str(getattr(link, "channel", "") or "")
        scope = str(getattr(link, "scope", "") or "")
        semantics = str(getattr(link, "semantics", "") or "")
        rail_val = (
            topology.link_rail(link_name)
            if hasattr(topology, "link_rail")
            else getattr(link, "rail_id", None)
        )
        members = frozenset(int(m) for m in link.members)
        part_key = (members, channel, scope, semantics, repr(rail_val))
        meta_by_name[link_name] = part_key
        by_partition.setdefault(part_key, []).append(link_name)
    for names in by_partition.values():
        if len(names) > 1:
            names.sort()
    classes: dict[str, tuple[str, str, str, str, int, str]] = {}
    for link_name, (members, channel, scope, semantics, rail_repr) in meta_by_name.items():
        if cohort_ranks and cohort_ranks <= members:
            # Global link covering every participating cohort rank: keep exact link identity
            # so parallel cluster-wide links never cross-contaminate one another.
            tag = link_name
        else:
            # Partitioned link (e.g. intra-package D2D or intra-node slice): match symmetric
            # sibling partitions by (channel, scope, semantics, rail, fanout, parallel ordinal).
            siblings = by_partition[(members, channel, scope, semantics, rail_repr)]
            tag = f"part:{siblings.index(link_name)}"
        classes[link_name] = (channel, scope, semantics, rail_repr, len(members), tag)
    return classes


def _link_mirror_class(
    pfx: str,
    link_classes: Mapping[str, tuple[str, str, str, str, int, str]],
) -> tuple[Any, ...]:
    """Classify a link resource prefix preserving channel, scope, rail, and partition identity."""
    if "link:" not in pfx:
        return (pfx,)
    ns_part, link_name = pfx.rsplit("link:", 1)
    cls = link_classes.get(link_name)
    if cls is None:
        return (ns_part, link_name)
    return (ns_part, *cls)


def _mirror_replica_resources(
    engine: Any, topology: Any, r0: int, peers: tuple[int, ...],
    pp: int, *, mirror_links: bool,
) -> None:
    """Mirror canonical die-local, UB die-injection, and symmetric link counters onto peer ranks."""
    src_tag = f"@r{r0}:"
    serviced = engine.serviced
    busy = engine.busy_ms
    for res_name, amt in list(serviced.items()):
        if not amt or src_tag not in res_name:
            continue
        b_val = busy.get(res_name, 0.0)
        for rep in peers:
            dst_name = res_name.replace(src_tag, f"@r{rep}:", 1)
            if dst_name in serviced:
                serviced[dst_name] = amt
                busy[dst_name] = b_val
    r0_ranks = tuple(rk for s in range(pp) for rk in topology.ranks(r0, s))
    if not r0_ranks:
        return
    if not mirror_links and len(r0_ranks) <= 1:
        return
    all_ranks_by_rep = {
        rep: tuple(rk for s in range(pp) for rk in topology.ranks(rep, s))
        for rep in (r0, *peers)
    }
    cohort_ranks = frozenset(rk for rk_list in all_ranks_by_rep.values() for rk in rk_list)
    link_classes = _build_link_mirror_classes(topology, cohort_ranks)
    port_by_class: dict[tuple[Any, str, int, float], tuple[float, float]] = {}
    pair_by_class: dict[tuple[Any, float], tuple[float, float]] = {}
    intra_pair_by_pos: dict[tuple[Any, int, int, float], tuple[float, float]] = {}
    ub_inj_by_pos: dict[tuple[str, str, int, float], tuple[float, float]] = {}
    resources = engine.resources
    rank_to_rep_pos = {
        rk: (rep, pos)
        for rep, rk_list in all_ranks_by_rep.items()
        for pos, rk in enumerate(rk_list)
    }
    r0_pos = {rk: pos for pos, rk in enumerate(r0_ranks)}
    for res_name, amt in list(serviced.items()):
        if not amt:
            continue
        res_obj = resources.get(res_name)
        rate_val = float(res_obj.rate) if res_obj is not None else 0.0
        b_val = busy.get(res_name, 0.0)
        m_ub = _UB_INJ_RE.match(res_name)
        if m_ub:
            pfx = m_ub.group("pfx")
            direction = m_ub.group("dir")
            rk_id = int(m_ub.group("rk"))
            if direction == "rx" and mirror_links:
                info = rank_to_rep_pos.get(rk_id)
                if info is not None:
                    ukey = (pfx, "rx", info[1], rate_val)
                    prev_amt, prev_busy = ub_inj_by_pos.get(ukey, (0.0, 0.0))
                    ub_inj_by_pos[ukey] = (prev_amt + amt, prev_busy + b_val)
            elif rk_id in r0_pos:
                ub_inj_by_pos[(pfx, direction, r0_pos[rk_id], rate_val)] = (amt, b_val)
            continue
        if "link:" not in res_name:
            continue
        m_port = _PORT_RE.match(res_name)
        if m_port:
            pfx = m_port.group("pfx")
            direction = m_port.group("dir")
            rk_id = int(m_port.group("rk"))
            lclass = _link_mirror_class(pfx, link_classes)
            if direction == "rx" and mirror_links:
                info = rank_to_rep_pos.get(rk_id)
                if info is not None:
                    pkey = (lclass, "rx", info[1], rate_val)
                    prev_amt, prev_busy = port_by_class.get(pkey, (0.0, 0.0))
                    port_by_class[pkey] = (prev_amt + amt, prev_busy + b_val)
            elif rk_id in r0_pos:
                port_by_class[(lclass, direction, r0_pos[rk_id], rate_val)] = (amt, b_val)
            continue
        m_pair = _PAIR_RE.match(res_name)
        if m_pair:
            pfx = m_pair.group("pfx")
            src_rk = int(m_pair.group("src"))
            dst_rk = int(m_pair.group("dst"))
            if src_rk in r0_pos:
                lclass = _link_mirror_class(pfx, link_classes)
                if dst_rk in r0_pos:
                    intra_pair_by_pos[(lclass, r0_pos[src_rk], r0_pos[dst_rk], rate_val)] = (
                        amt, b_val,
                    )
                elif mirror_links:
                    pair_by_class.setdefault((lclass, rate_val), (amt, b_val))
    if not (port_by_class or pair_by_class or intra_pair_by_pos or ub_inj_by_pos):
        return
    for res_name in list(serviced):
        res_obj = resources.get(res_name)
        rate_val = float(res_obj.rate) if res_obj is not None else 0.0
        m_ub = _UB_INJ_RE.match(res_name)
        if m_ub:
            pfx = m_ub.group("pfx")
            direction = m_ub.group("dir")
            rk_id = int(m_ub.group("rk"))
            info = rank_to_rep_pos.get(rk_id)
            if info is None or (info[0] == r0 and not (direction == "rx" and mirror_links)):
                continue
            val = ub_inj_by_pos.get((pfx, direction, info[1], rate_val))
            if val is not None:
                serviced[res_name], busy[res_name] = val
            continue
        if "link:" not in res_name:
            continue
        m_port = _PORT_RE.match(res_name)
        if m_port:
            pfx = m_port.group("pfx")
            direction = m_port.group("dir")
            rk_id = int(m_port.group("rk"))
            info = rank_to_rep_pos.get(rk_id)
            if info is None or (info[0] == r0 and not (direction == "rx" and mirror_links)):
                continue
            lclass = _link_mirror_class(pfx, link_classes)
            val = port_by_class.get((lclass, direction, info[1], rate_val))
            if val is not None:
                serviced[res_name], busy[res_name] = val
            continue
        m_pair = _PAIR_RE.match(res_name)
        if m_pair and serviced[res_name] == 0.0:
            pfx = m_pair.group("pfx")
            src_rk = int(m_pair.group("src"))
            dst_rk = int(m_pair.group("dst"))
            src_info = rank_to_rep_pos.get(src_rk)
            dst_info = rank_to_rep_pos.get(dst_rk)
            if src_info is None or dst_info is None:
                continue
            if src_info[0] == r0 and dst_info[0] == r0:
                continue
            lclass = _link_mirror_class(pfx, link_classes)
            if src_info[0] == dst_info[0]:
                val = intra_pair_by_pos.get((lclass, src_info[1], dst_info[1], rate_val))
            elif mirror_links:
                val = pair_by_class.get((lclass, rate_val))
            else:
                val = None
            if val is not None:
                serviced[res_name], busy[res_name] = val


def expand_dedup_graph_artifacts(
    execution: Any,
    records: list[dict[str, Any]],
    graphs: Mapping[int, tuple[Any, ...]],
    *,
    timeline_replicas: frozenset[int] | None = None,
) -> int:
    """Reconstruct peer-replica timeline records and conserved resource counters."""
    dedup_meta = getattr(execution, "_dedup_cohorts", None)
    if not dedup_meta:
        return 0
    engine = execution.engine
    topology = execution.topology
    pp = max(1, int(execution.case.runtime.parallel.pp))
    r0_records_by_rep: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        r0_records_by_rep.setdefault(rec["replica"], []).append(rec)
    new_records: list[dict[str, Any]] = []
    skipped_occurrences = 0
    for entry in dedup_meta:
        r0 = entry["canonical"]
        peers = entry["peers"]
        mirror_links = entry["mirror_links"]
        _mirror_replica_resources(
            engine, topology, r0, peers, pp, mirror_links=mirror_links,
        )
        base_recs = r0_records_by_rep.get(r0, ())
        if not base_recs:
            continue
        base_non_barrier = sum(rec["op"].family != "barrier" for rec in base_recs)
        r0_prefix = f"r{r0}:"
        for rep in peers:
            if timeline_replicas is not None and rep not in timeline_replicas:
                skipped_occurrences += base_non_barrier
                continue
            rep_ops = {op.op_id: op for op in graphs.get(rep, ())}
            rep_prefix = f"r{rep}:"
            for rec in base_recs:
                op0 = rec["op"]
                op_r = rep_ops.get(op0.op_id, op0)
                preds = tuple(
                    (rep_prefix + p[len(r0_prefix):]) if p.startswith(r0_prefix) else p
                    for p in rec["predecessors"]
                )
                new_records.append({
                    **rec,
                    "id": f"r{rep}:{op_r.op_id}",
                    "op": op_r,
                    "replica": rep,
                    "predecessors": preds,
                })
    if new_records:
        records.extend(new_records)
    return skipped_occurrences
