"""Homogeneous cohort proof, replica rebinding and bounded quotient eligibility.

[Role]
  Separate two optimizations: immutable semantic template reuse (always safe
  after an isomorphism proof, with normal cohort traffic binding left to the
  caller) and an execution quotient (only for a proved symmetric domain).
  Everything unproved keeps full execution and reports the reason.

[Invariants]
  * Template reuse never removes participants, link work or reverse traffic.
  * A quotient proof requires identical per-rank structure, requests, state
    generations, supplies/caps and a peer-flow-preserving placement.
  * ``require`` needs template proof only; an unproved quotient still retains
    full execution rather than inventing a rank-zero run.

[Timeouts & Locks]
  Bounded by cohort size x operators; no I/O or locks.

[Side-Effect Policy]
  Returns immutable graphs/proofs; the caller owns the cache and Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import re
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.graph_analysis import SemanticGraphCache, semantic_graph_key
from mindie_sim.prediction.sharding.local_tp import local_tp_groups


RANK_GROUP = 1
REUSE_OFF = "off"
REUSE_AUTO = "auto"
REUSE_REQUIRE = "require"
REUSE_MODES = frozenset({REUSE_OFF, REUSE_AUTO, REUSE_REQUIRE})
RANK_TOKEN = re.compile(r"@r(\d+)")
_TRIE_END = ""


def _trie_regex_body(node: Mapping[str, Any]) -> str:
    """Emit a deterministic, non-backtracking prefix-trie regex expression."""
    has_end = _TRIE_END in node
    branches = sorted((key, child) for key, child in node.items() if key != _TRIE_END)
    if not branches:
        return ""
    parts: list[str] = []
    for ch, child in branches:
        tail = _trie_regex_body(child)
        parts.append(re.escape(ch) + tail)
    body = parts[0] if len(parts) == 1 else "(?:" + "|".join(parts) + ")"
    if has_end:
        return f"(?:{body})?" if len(parts) == 1 and len(parts[0]) > 1 else f"{body}?"
    return body


def _compile_token_pattern(tokens: Sequence[str]) -> re.Pattern | None:
    """Compile whole-request tokens into one O(len(text)) prefix-trie matcher."""
    trie: dict[str, Any] = {}
    for raw in tokens:
        token = str(raw)
        if not token:
            continue
        cursor = trie
        for ch in token:
            cursor = cursor.setdefault(ch, {})
        cursor[_TRIE_END] = True
    if not trie:
        return None
    return re.compile(r"(?<![\w@])(?:" + _trie_regex_body(trie) + r")(?![\w@])")


@dataclass(frozen=True)
class CohortReuseProof:
    homogeneous: bool
    reasons: tuple[str, ...] = ()
    template_reused: bool = False
    execution_quotient: bool = False
    fallback_reason: str = ""
    replicas: int = 0


@dataclass(frozen=True)
class PreparedCohort:
    graphs: Mapping[int, tuple]
    proof: CohortReuseProof
    template_reused: bool
    execution_quotient: bool
    fallback_reason: str
    cache_stats: Mapping[str, int] = field(default_factory=dict)


def _ops(value: Any) -> tuple:
    if hasattr(value, "ops"):
        return tuple(value.ops)
    return tuple(value)


_SKIP_NORM_META_KEYS = frozenset({"rank", "ep_rank", "routing_price_owner", "group_ranks", "replica"})
_REBIND_ATOMIC = (int, float, bool, type(None))


def _common_prefix(tokens: Sequence[str]) -> str:
    non_empty = [t for t in tokens if t]
    if not non_empty:
        return ""
    pfx = non_empty[0]
    for tok in non_empty[1:]:
        while pfx and not tok.startswith(pfx):
            pfx = pfx[:-1]
        if not pfx:
            break
    return pfx


def _freeze_attr(value: Any) -> Any:
    """Return a deterministic hashable representation of a typed OpSpec field."""
    if value is None or type(value) in _REBIND_ATOMIC or type(value) is str:
        return value
    try:
        hash(value)
        return value
    except TypeError:
        pass
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple((f.name, _freeze_attr(getattr(value, f.name))) for f in fields(value)),
        )
    if isinstance(value, Mapping):
        return tuple(sorted(((str(k), _freeze_attr(v)) for k, v in value.items()), key=lambda pair: pair[0]))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_attr(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_attr(item) for item in value), key=repr))
    return repr(value)


def _request_slot_map(graph: Sequence[Any]) -> tuple[dict[str, str], re.Pattern | None, str]:
    """Map one graph's distinct request IDs to canonical first-appearance slots."""
    req_ids = _request_ids(graph)
    if not req_ids:
        return {}, None, ""
    slots = {req: f"@req{idx}" for idx, req in enumerate(req_ids)}
    return slots, _compile_token_pattern(req_ids), _common_prefix(req_ids)


def _normalized(
    value: Any,
    request_pattern: re.Pattern | None = None,
    request_slots: Mapping[str, str] = (),  # type: ignore[assignment]
    request_prefix: str = "",
    local_rank_slots: Mapping[int, str] | None = None,
) -> Any:
    t = type(value)
    if t in _REBIND_ATOMIC:
        return value
    if t is str:
        if request_slots and value in request_slots:
            return request_slots[value]
        if "@r" in value:
            if local_rank_slots is not None:
                text = RANK_TOKEN.sub(
                    lambda match: (
                        f"@r_local_{local_rank_slots[int(match.group(RANK_GROUP))]}"
                        if int(match.group(RANK_GROUP)) in local_rank_slots
                        else match.group(0)
                    ),
                    value,
                )
            else:
                text = RANK_TOKEN.sub("@r*", value)
        else:
            text = value
        if request_pattern is not None and (not request_prefix or request_prefix in text):
            text = request_pattern.sub(lambda match: request_slots[match.group()], text)
        return text
    if isinstance(value, Mapping):
        return tuple(sorted(
            ((str(key), _normalized(item, request_pattern, request_slots, request_prefix, local_rank_slots))
             for key, item in value.items()
             if str(key) not in _SKIP_NORM_META_KEYS),
            key=lambda pair: pair[0]))
    if isinstance(value, (tuple, list)):
        return tuple(
            _normalized(item, request_pattern, request_slots, request_prefix, local_rank_slots)
            for item in value
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(
            (_normalized(item, request_pattern, request_slots, request_prefix, local_rank_slots)
             for item in value),
            key=repr))
    return value


def _op_signature(
    op: Any,
    request_pattern: re.Pattern | None = None,
    request_slots: Mapping[str, str] = (),  # type: ignore[assignment]
    request_prefix: str = "",
    local_rank_slots: Mapping[int, str] | None = None,
    producer_index: Mapping[str, int] | None = None,
) -> tuple:
    metadata = dict(getattr(op, "metadata", ()) or {})
    op_requests = tuple(getattr(op, "requests", ()) or ())
    raw_deps = tuple(getattr(op, "deps", ()) or ())
    if producer_index is not None:
        dep_sig = tuple(
            ("internal", producer_index[str(dep)]) if str(dep) in producer_index
            else ("external", _normalized(
                str(dep), request_pattern, request_slots, request_prefix, local_rank_slots,
            ))
            for dep in raw_deps
        )
    else:
        dep_sig = _normalized(
            raw_deps, request_pattern, request_slots, request_prefix, local_rank_slots,
        )
    return (
        _normalized(
            str(getattr(op, "op_id", "")),
            request_pattern, request_slots, request_prefix, local_rank_slots,
        ),
        str(getattr(op, "family", "")), str(getattr(op, "recipe", "")),
        str(getattr(op, "phase", "")), int(getattr(op, "stage", 0)),
        str(getattr(op, "dtype", "")), str(getattr(op, "layout", "")),
        tuple(getattr(op, "semantic_shape", ()) or ()),
        tuple(getattr(op, "local_shape", ()) or ()),
        tuple(getattr(op, "padded_shape", ()) or ()),
        tuple(getattr(op, "launch_shape", ()) or ()),
        float(getattr(op, "flops", 0) or 0),
        float(getattr(op, "hbm_bytes", 0) or 0),
        float(getattr(op, "mte2_bytes", 0) or 0),
        float(getattr(op, "mte3_bytes", 0) or 0),
        float(getattr(op, "vector_ops", 0) or 0),
        float(getattr(op, "link_bytes", 0) or 0),
        str(getattr(op, "link_scope", "") or ""),
        str(getattr(op, "source_boundary", "")),
        tuple(getattr(op, "returns", ()) or ()),
        str(getattr(op, "stream", "") or ""),
        _freeze_attr(getattr(op, "precision_abi", None)),
        _freeze_attr(getattr(op, "kernel_binding", None)),
        _freeze_attr(getattr(op, "owner_schedule", None)),
        tuple(getattr(op, "logical_precision_reqs", ()) or ()),
        str(getattr(op, "coverage", "") or ""),
        dep_sig,
        _normalized(
            tuple(getattr(op, "matrix_flops", ()) or ()),
            request_pattern, request_slots, request_prefix, local_rank_slots,
        ),
        _normalized(
            metadata.get("matrix_flops"),
            request_pattern, request_slots, request_prefix, local_rank_slots,
        ),
        tuple(request_slots.get(str(r), str(r)) for r in op_requests) if request_slots else op_requests,
        _normalized(metadata, request_pattern, request_slots, request_prefix, local_rank_slots),
    )


def _values_isomorphic(
    v0: Any,
    vr: Any,
    pat0: re.Pattern | None,
    slots0: Mapping[str, str],
    pfx0: str,
    pat_r: re.Pattern | None,
    slots_r: Mapping[str, str],
    pfx_r: str,
    rank_slots0: Mapping[int, str] | None = None,
    rank_slots_r: Mapping[int, str] | None = None,
) -> bool:
    """Short-circuit structural equality matching per-graph slot-normalized values."""
    t0 = type(v0)
    tr = type(vr)
    if t0 in _REBIND_ATOMIC and tr in _REBIND_ATOMIC:
        return v0 == vr
    if t0 is str and tr is str:
        return (
            _normalized(v0, pat0, slots0, pfx0, rank_slots0)
            == _normalized(vr, pat_r, slots_r, pfx_r, rank_slots_r)
        )
    if (t0 is tuple or t0 is list) and (tr is tuple or tr is list):
        if len(v0) != len(vr):
            return False
        for a, b in zip(v0, vr):
            if type(a) in _REBIND_ATOMIC and type(b) in _REBIND_ATOMIC:
                if a != b:
                    return False
            elif not _values_isomorphic(
                a, b, pat0, slots0, pfx0, pat_r, slots_r, pfx_r, rank_slots0, rank_slots_r,
            ):
                return False
        return True
    if t0 is dict and tr is dict:
        keys0 = {str(k) for k in v0 if str(k) not in _SKIP_NORM_META_KEYS}
        keys_r = {str(k) for k in vr if str(k) not in _SKIP_NORM_META_KEYS}
        if keys0 != keys_r:
            return False
        for k in keys0:
            a, b = v0[k], vr[k]
            if type(a) in _REBIND_ATOMIC and type(b) in _REBIND_ATOMIC:
                if a != b:
                    return False
            elif not _values_isomorphic(
                a, b, pat0, slots0, pfx0, pat_r, slots_r, pfx_r, rank_slots0, rank_slots_r,
            ):
                return False
        return True
    return (
        _normalized(v0, pat0, slots0, pfx0, rank_slots0)
        == _normalized(vr, pat_r, slots_r, pfx_r, rank_slots_r)
    )


def _deps_isomorphic(
    deps0: tuple[Any, ...],
    deps_r: tuple[Any, ...],
    pat0: re.Pattern | None,
    slots0: Mapping[str, str],
    pfx0: str,
    pat_r: re.Pattern | None,
    slots_r: Mapping[str, str],
    pfx_r: str,
    seen0: Mapping[str, int] | None,
    seen_r: Mapping[str, int] | None,
    rank_slots0: Mapping[int, str] | None = None,
    rank_slots_r: Mapping[int, str] | None = None,
) -> bool:
    if len(deps0) != len(deps_r):
        return False
    ranks_differ = rank_slots0 != rank_slots_r
    if seen0 is not None and seen_r is not None:
        for d0, dr in zip(deps0, deps_r):
            s0, sr = str(d0), str(dr)
            in0, in_r = s0 in seen0, sr in seen_r
            if in0 != in_r:
                return False
            if in0:
                if seen0[s0] != seen_r[sr]:
                    return False
            elif (s0 != sr or slots0 != slots_r or ranks_differ) and not _values_isomorphic(
                s0, sr, pat0, slots0, pfx0, pat_r, slots_r, pfx_r,
                rank_slots0, rank_slots_r,
            ):
                return False
        return True
    if deps0 != deps_r or slots0 != slots_r or ranks_differ:
        return _values_isomorphic(
            deps0, deps_r, pat0, slots0, pfx0, pat_r, slots_r, pfx_r,
            rank_slots0, rank_slots_r,
        )
    return True


def _ops_isomorphic(
    op0: Any,
    op_r: Any,
    pat0: re.Pattern | None,
    slots0: Mapping[str, str],
    pfx0: str,
    pat_r: re.Pattern | None,
    slots_r: Mapping[str, str],
    pfx_r: str,
    *,
    req_slots_r: Mapping[str, str] | None = None,
    seen0: Mapping[str, int] | None = None,
    seen_r: Mapping[str, int] | None = None,
    rank_slots0: Mapping[int, str] | None = None,
    rank_slots_r: Mapping[int, str] | None = None,
) -> bool:
    """Structural isomorphism check enforcing request bijection, producer wiring, and typed ABI equality."""
    eff_req_r = slots_r if req_slots_r is None else req_slots_r
    ranks_differ = rank_slots0 != rank_slots_r
    id0 = str(getattr(op0, "op_id", ""))
    id_r = str(getattr(op_r, "op_id", ""))
    if (id0 != id_r or slots0 != slots_r or ranks_differ) and not _values_isomorphic(
        id0, id_r, pat0, slots0, pfx0, pat_r, slots_r, pfx_r,
        rank_slots0, rank_slots_r,
    ):
        return False
    deps0 = tuple(getattr(op0, "deps", ()) or ())
    deps_r = tuple(getattr(op_r, "deps", ()) or ())
    if not _deps_isomorphic(
        deps0, deps_r, pat0, slots0, pfx0, pat_r, slots_r, pfx_r,
        seen0, seen_r, rank_slots0, rank_slots_r,
    ):
        return False
    if op0 == op_r and slots0 == eff_req_r and not ranks_differ:
        return True
    req0 = tuple(getattr(op0, "requests", ()) or ())
    req_r = tuple(getattr(op_r, "requests", ()) or ())
    if (
        str(getattr(op0, "family", "")) != str(getattr(op_r, "family", ""))
        or str(getattr(op0, "recipe", "")) != str(getattr(op_r, "recipe", ""))
        or str(getattr(op0, "phase", "")) != str(getattr(op_r, "phase", ""))
        or int(getattr(op0, "stage", 0)) != int(getattr(op_r, "stage", 0))
        or str(getattr(op0, "dtype", "")) != str(getattr(op_r, "dtype", ""))
        or str(getattr(op0, "layout", "")) != str(getattr(op_r, "layout", ""))
        or tuple(getattr(op0, "semantic_shape", ()) or ()) != tuple(getattr(op_r, "semantic_shape", ()) or ())
        or tuple(getattr(op0, "local_shape", ()) or ()) != tuple(getattr(op_r, "local_shape", ()) or ())
        or tuple(getattr(op0, "padded_shape", ()) or ()) != tuple(getattr(op_r, "padded_shape", ()) or ())
        or tuple(getattr(op0, "launch_shape", ()) or ()) != tuple(getattr(op_r, "launch_shape", ()) or ())
        or float(getattr(op0, "flops", 0) or 0) != float(getattr(op_r, "flops", 0) or 0)
        or float(getattr(op0, "hbm_bytes", 0) or 0) != float(getattr(op_r, "hbm_bytes", 0) or 0)
        or float(getattr(op0, "mte2_bytes", 0) or 0) != float(getattr(op_r, "mte2_bytes", 0) or 0)
        or float(getattr(op0, "mte3_bytes", 0) or 0) != float(getattr(op_r, "mte3_bytes", 0) or 0)
        or float(getattr(op0, "vector_ops", 0) or 0) != float(getattr(op_r, "vector_ops", 0) or 0)
        or float(getattr(op0, "link_bytes", 0) or 0) != float(getattr(op_r, "link_bytes", 0) or 0)
        or str(getattr(op0, "link_scope", "") or "") != str(getattr(op_r, "link_scope", "") or "")
        or str(getattr(op0, "source_boundary", "")) != str(getattr(op_r, "source_boundary", ""))
        or tuple(getattr(op0, "returns", ()) or ()) != tuple(getattr(op_r, "returns", ()) or ())
        or str(getattr(op0, "stream", "") or "") != str(getattr(op_r, "stream", "") or "")
        or getattr(op0, "precision_abi", None) != getattr(op_r, "precision_abi", None)
        or getattr(op0, "kernel_binding", None) != getattr(op_r, "kernel_binding", None)
        or getattr(op0, "owner_schedule", None) != getattr(op_r, "owner_schedule", None)
        or tuple(getattr(op0, "logical_precision_reqs", ()) or ()) != tuple(getattr(op_r, "logical_precision_reqs", ()) or ())
        or str(getattr(op0, "coverage", "") or "") != str(getattr(op_r, "coverage", "") or "")
        or len(req0) != len(req_r)
    ):
        return False
    if req0 and tuple(slots0.get(str(r), str(r)) for r in req0) != tuple(eff_req_r.get(str(r), str(r)) for r in req_r):
        return False
    mf0 = tuple(getattr(op0, "matrix_flops", ()) or ())
    mf_r = tuple(getattr(op_r, "matrix_flops", ()) or ())
    if (mf0 != mf_r or slots0 != slots_r or ranks_differ) and not _values_isomorphic(
        mf0, mf_r, pat0, slots0, pfx0, pat_r, slots_r, pfx_r,
        rank_slots0, rank_slots_r,
    ):
        return False
    raw_m0 = getattr(op0, "metadata", ()) or ()
    raw_mr = getattr(op_r, "metadata", ()) or ()
    if raw_m0 == raw_mr and (not raw_m0 or (slots0 == slots_r and not ranks_differ)):
        return True
    return _values_isomorphic(
        dict(raw_m0), dict(raw_mr), pat0, slots0, pfx0, pat_r, slots_r, pfx_r,
        rank_slots0, rank_slots_r,
    )


def _graph_signature(
    ops: Sequence[Any],
    request_pattern: re.Pattern | None = None,
    request_slots: Mapping[str, str] = (),  # type: ignore[assignment]
    request_prefix: str = "",
    local_rank_slots: Mapping[int, str] | None = None,
) -> tuple:
    seen_producers: dict[str, int] = {}
    sigs = []
    for idx, op in enumerate(ops):
        sigs.append(_op_signature(
            op, request_pattern, request_slots, request_prefix,
            local_rank_slots, producer_index=seen_producers,
        ))
        seen_producers[str(getattr(op, "op_id", ""))] = idx
    return tuple(sigs)


def _replica_hardware_signature(case: Any, replica: int, placement: Any) -> tuple:
    """Physical die, datapath capability, and incident link signature for quotient verification."""
    from mindie_sim.schema.datapath import _arch_capability_key, datapath_for_devices

    parallel = case.runtime.parallel
    pp = max(1, int(parallel.pp))
    links = getattr(case.machine, "links", ()) or ()
    transfer_channels = getattr(case.machine, "transfer_channels", ()) or ()
    stage_sigs = []
    rank_to_pos: dict[int, int] = {}
    for stage in range(pp):
        ranks = placement.stage_ranks(int(replica), stage)
        for rk in ranks:
            if int(rk) not in rank_to_pos:
                rank_to_pos[int(rk)] = len(rank_to_pos)
        dies = tuple(placement.die_for_rank(rank) for rank in ranks)
        stage_rank_set = frozenset(int(d.rank) for d in dies)
        compute_arch, datapath = datapath_for_devices(
            dies, links=links, transfer_channels=transfer_channels,
        )
        ub_edges = getattr(datapath, "ub_peer_edges", ()) or ()
        in_stage_ub = sum(1 for src, dst in ub_edges if src in stage_rank_set and dst in stage_rank_set)
        out_stage_ub = sum(1 for src, dst in ub_edges if src in stage_rank_set and dst not in stage_rank_set)
        dp_sig = datapath._signature((in_stage_ub, out_stage_ub))
        arch_sig = (
            _arch_capability_key(compute_arch),
            tuple(
                (rt.name, rt.provides, _freeze_attr(rt.conditions))
                for rt in compute_arch.runtimes
            ),
        )
        die_sig = tuple(
            (
                str(d.sku), d.rates, d.buffers, d.compute_peaks, d.compute_peak_aliases,
                d.service_modes, d.store_queues, d.work_units, d.resource_units,
                d.buffer_units, float(d.launch_us), _freeze_attr(d.kernel_invocations),
                d.memory_access, d.hbm_read_bytes_per_s, d.hbm_write_bytes_per_s,
                d.hbm_rw_alpha, int(d.simt_ub_carveout_bytes),
                _freeze_attr(d.subtile_wave), _freeze_attr(d.dispatch_sync),
                getattr(d.physical_mechanisms, "signature", None),
            )
            for d in dies
        )
        stage_sigs.append((die_sig, dp_sig, arch_sig))
    replica_ranks = frozenset(rank_to_pos)
    link_sigs = []
    for link in links:
        members = tuple(int(m) for m in link.members)
        if not any(m in replica_ranks for m in members):
            continue
        is_intra = all(m in replica_ranks for m in members)
        local_pos = tuple(sorted(rank_to_pos[m] for m in members if m in rank_to_pos))
        local_eps = tuple(sorted(
            (rank_to_pos[int(rk)], float(rate))
            for rk, rate in link.endpoint_rates
            if int(rk) in rank_to_pos
        ))
        link_sigs.append((
            is_intra,
            local_pos,
            len(members),
            str(link.channel),
            str(link.protocol),
            str(link.semantics),
            str(link.scope),
            float(link.capacity_bytes_per_ms),
            link.peak_capacity_bytes_per_ms,
            float(link.bw_utilization),
            float(link.latency_us),
            int(link.switch_hops),
            float(link.switch_hop_latency_us),
            link.switch_capacity_bytes_per_ms,
            link.transaction_bytes,
            int(link.alignment_bytes),
            int(link.protocol_overhead_bytes),
            link.protocol_bands,
            local_eps,
        ))
    link_sigs.sort(key=repr)
    return (tuple(stage_sigs), tuple(link_sigs))


def _cohort_hardware_homogeneous(case: Any, ordered_keys: Sequence[int]) -> bool:
    """Verify all participating cohort replicas share identical physical service supply."""
    if len(ordered_keys) <= 1:
        return True
    machine = getattr(case, "machine", None)
    runtime = getattr(case, "runtime", None)
    if machine is None or runtime is None or not hasattr(machine, "validate_placement"):
        return True
    try:
        placement = machine.validate_placement(runtime.parallel)
        base_sig = _replica_hardware_signature(case, int(ordered_keys[0]), placement)
        return all(
            _replica_hardware_signature(case, int(rep), placement) == base_sig
            for rep in ordered_keys[1:]
        )
    except (ModelingError, IndexError, KeyError):
        return bool(getattr(machine, "homogeneous", False))


def _replica_local_rank_slots(case: Any, replica: int, topology: Any = None) -> dict[int, str] | None:
    if case is None or getattr(case, "machine", None) is None or getattr(case, "runtime", None) is None:
        return None
    from mindie_sim.prediction.topology import Topology

    try:
        parallel = case.runtime.parallel
        topo = topology if topology is not None else Topology(case.machine, parallel)
        slots: dict[int, str] = {}
        ordinal = 0
        for stage in range(max(1, int(parallel.pp))):
            for rk in topo.ranks(int(replica), stage):
                if int(rk) not in slots:
                    slots[int(rk)] = str(ordinal)
                    ordinal += 1
        return slots or None
    except (ModelingError, ValueError, KeyError, IndexError):
        return None


def prove_homogeneous_cohort(
    case: Any, plans: Mapping[int, Any], *, selected_implementations: Any = None,
    initial_cache_states: Any = None,
) -> CohortReuseProof:
    """Prove structural isomorphism of a complete replica cohort."""
    if not isinstance(plans, Mapping) or not plans:
        return CohortReuseProof(False, ("cohort_empty",), replicas=0)
    replicas = len(plans)
    ordered_keys = sorted(plans)
    ordered = [_ops(plans[key]) for key in ordered_keys]
    reasons: list[str] = []
    if any(len(graph) != len(ordered[0]) for graph in ordered[1:]):
        reasons.append("asymmetric_replica_graph")
    elif len(ordered) > 1:
        base_ops = ordered[0]
        slots0, pat0, pfx0 = _request_slot_map(base_ops)
        shared_topo = None
        if case is not None and getattr(case, "machine", None) is not None and getattr(case, "runtime", None) is not None:
            from mindie_sim.prediction.topology import Topology
            try:
                shared_topo = Topology(case.machine, case.runtime.parallel)
            except (ModelingError, ValueError, KeyError, IndexError):
                shared_topo = None
        rank_slots0 = _replica_local_rank_slots(case, int(ordered_keys[0]), shared_topo)
        for rep_key, graph in zip(ordered_keys[1:], ordered[1:]):
            slots_r, pat_r, pfx_r = _request_slot_map(graph)
            if len(slots0) != len(slots_r):
                reasons.append("asymmetric_replica_graph")
                break
            meta_slots_r = {**slots0, **slots_r} if slots0 else slots_r
            if meta_slots_r != slots_r:
                meta_tokens_r = tuple(meta_slots_r)
                meta_pat_r = _compile_token_pattern(meta_tokens_r)
                meta_pfx_r = _common_prefix(meta_tokens_r)
            else:
                meta_pat_r, meta_pfx_r = pat_r, pfx_r
            rank_slots_r = _replica_local_rank_slots(case, int(rep_key), shared_topo)
            seen0: dict[str, int] = {}
            seen_r: dict[str, int] = {}
            is_iso = True
            for idx, (o0, o_r) in enumerate(zip(base_ops, graph)):
                if not _ops_isomorphic(
                    o0, o_r, pat0, slots0, pfx0,
                    meta_pat_r, meta_slots_r, meta_pfx_r,
                    req_slots_r=slots_r, seen0=seen0, seen_r=seen_r,
                    rank_slots0=rank_slots0, rank_slots_r=rank_slots_r,
                ):
                    is_iso = False
                    break
                seen0[str(getattr(o0, "op_id", ""))] = idx
                seen_r[str(getattr(o_r, "op_id", ""))] = idx
            if not is_iso:
                reasons.append("asymmetric_replica_graph")
                break
    if selected_implementations is not None:
        selections = [selected_implementations.get(key) for key in ordered_keys]
        if any(item != selections[0] for item in selections[1:]):
            reasons.append("asymmetric_implementation_selection")
    if initial_cache_states is not None:
        states = [initial_cache_states.get(key) for key in ordered_keys]
        if any(item != states[0] for item in states[1:]):
            reasons.append("asymmetric_cache_state")
    template_reused = not reasons
    if not template_reused:
        return CohortReuseProof(False, tuple(reasons), replicas=replicas)
    quotient = True
    if not _cohort_hardware_homogeneous(case, ordered_keys):
        quotient = False
        reasons.append("heterogeneous_replica_hardware")
    for op in ordered[0]:
        if getattr(op, "source_boundary", ""):
            quotient = False
            reasons.append("source_bound_collective")
        if float(getattr(op, "link_bytes", 0) or 0) > 0:
            quotient = False
            reasons.append("shared_link_demand")
        metadata = dict(getattr(op, "metadata", ()) or {})
        if metadata.get("mc2_phase") or metadata.get("device_kernel_group"):
            quotient = False
            reasons.append("fused_source_mechanism")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return CohortReuseProof(
        True, unique_reasons, template_reused=True,
        execution_quotient=quotient,
        fallback_reason=("" if quotient else "full_execution_retained"),
        replicas=replicas,
    )


def _request_binding_pattern(binding: Mapping[str, str]) -> re.Pattern | None:
    """One whole-request matcher for nested metadata string leaves."""
    if not binding:
        return None
    return _compile_token_pattern(tuple(str(key) for key in binding if str(key)))


def _scan_meta_prefix(value: Any, prefix: str, memo: dict[int, bool]) -> bool:
    t = type(value)
    if t in _REBIND_ATOMIC:
        return False
    if t is str:
        return prefix in value
    oid = id(value)
    hit = memo.get(oid)
    if hit is not None:
        return hit
    found = False
    if t is tuple or t is list:
        for item in value:
            it = type(item)
            if it is str:
                if prefix in item:
                    found = True
            elif it not in _REBIND_ATOMIC and _scan_meta_prefix(item, prefix, memo):
                found = True
    elif t is dict:
        for item in value.values():
            it = type(item)
            if it is str:
                if prefix in item:
                    found = True
            elif it not in _REBIND_ATOMIC and _scan_meta_prefix(item, prefix, memo):
                found = True
    elif isinstance(value, Mapping):
        for item in value.values():
            if _scan_meta_prefix(item, prefix, memo):
                found = True
    elif isinstance(value, (set, frozenset)):
        for item in value:
            if _scan_meta_prefix(item, prefix, memo):
                found = True
    memo[oid] = found
    return found


def _rebind_nested_requests(
    value: Any,
    pattern: re.Pattern | None,
    binding: Mapping[str, str],
    prefix: str = "",
    has_prefix: Mapping[int, bool] | None = None,
) -> Any:
    """Rename request identities nested in metadata without touching types."""
    if pattern is None:
        return value
    t = type(value)
    if t in _REBIND_ATOMIC:
        return value
    if t is str:
        if prefix and prefix not in value:
            return value
        if value in binding:
            return binding[value]
        return pattern.sub(lambda match: binding[match.group()], value)
    if has_prefix is not None and not has_prefix.get(id(value), True):
        return value
    if t is dict:
        changed = False
        out = {}
        for k, item in value.items():
            ni = _rebind_nested_requests(item, pattern, binding, prefix, has_prefix)
            if ni is not item:
                changed = True
            out[k] = ni
        return out if changed else value
    if t is tuple:
        changed = False
        out_t = []
        for item in value:
            ni = _rebind_nested_requests(item, pattern, binding, prefix, has_prefix)
            if ni is not item:
                changed = True
            out_t.append(ni)
        return tuple(out_t) if changed else value
    if t is list:
        changed = False
        out_l = []
        for item in value:
            ni = _rebind_nested_requests(item, pattern, binding, prefix, has_prefix)
            if ni is not item:
                changed = True
            out_l.append(ni)
        return out_l if changed else value
    if isinstance(value, Mapping):
        return type(value)((key, _rebind_nested_requests(item, pattern, binding, prefix, has_prefix))
                           for key, item in value.items())
    if isinstance(value, (set, frozenset)):
        return type(value)(_rebind_nested_requests(item, pattern, binding, prefix, has_prefix)
                           for item in value)
    return value


def _rank_map(
    case: Any, replica: int, *, template_replica: int = 0, topology: Any = None,
) -> dict[int, int]:
    from mindie_sim.prediction.topology import Topology

    parallel = case.runtime.parallel
    topo = topology if topology is not None else Topology(case.machine, parallel)
    mapping: dict[int, int] = {}
    for stage in range(max(1, int(parallel.pp))):
        base = topo.ranks(int(template_replica), stage)
        target = topo.ranks(int(replica), stage)
        mapping.update({int(left): int(right)
                        for left, right in zip(base, target)})
    return mapping


def _rebind_group_ranks(
    metadata: dict[str, Any], op: Any, replica: int, rank_map: Mapping[int, int],
    case: Any, topology: Any, local_groups_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], ...]],
) -> Any:
    raw_group = tuple(int(rank) for rank in metadata["group_ranks"])
    width = int(metadata.get("local_tp_width", 1) or 1)
    stride = int(metadata.get("local_tp_stride", 1) or 1)
    if raw_group and all(rank in rank_map for rank in raw_group):
        metadata["group_ranks"] = tuple(rank_map[rank] for rank in raw_group)
        return topology
    if case is not None and width > 1:
        if topology is None:
            from mindie_sim.prediction.topology import Topology
            topology = Topology(case.machine, case.runtime.parallel)
        stage = int(getattr(op, "stage", 0))
        anchor = topology.ranks(int(replica), stage)[0]
        cache_key = (stage, width, stride)
        groups = local_groups_cache.get(cache_key)
        if groups is None:
            groups = local_tp_groups(
                case.machine, case.runtime.parallel,
                stage=stage, width=width, stride=stride)
            local_groups_cache[cache_key] = groups
        metadata["group_ranks"] = next(
            (group for group in groups if anchor in group), raw_group)
        return topology
    mapped = tuple(rank_map.get(rank, rank) for rank in raw_group)
    metadata["group_ranks"] = (
        mapped if len(set(mapped)) == len(raw_group) else raw_group)
    return topology


def _ep_owner_map(
    case: Any,
    rank_map: Mapping[int, int],
    replica: int,
) -> dict[int, int]:
    """Map template EP-local ordinals to target replica EP-local ordinals."""
    if case is None or not rank_map:
        return dict(rank_map)
    from mindie_sim.prediction.execution.rank_ledger import ep_groups

    parallel = case.runtime.parallel
    ep_degree = max(1, int(getattr(parallel, "ep", 1) or 1))
    owner_by_rank: dict[int, int] = {}
    for stage in range(max(1, int(getattr(parallel, "pp", 1) or 1))):
        try:
            _, groups = ep_groups(parallel, case.machine, replica, stage)
        except (ModelingError, ValueError, KeyError, IndexError):
            continue
        for group in groups:
            for ordinal, phys_rank in enumerate(group):
                owner_by_rank[int(phys_rank)] = int(ordinal)
    mapping: dict[int, int] = {}
    for left, right in rank_map.items():
        left_owner = owner_by_rank.get(int(left), int(left) % ep_degree)
        right_owner = owner_by_rank.get(int(right), int(right) % ep_degree)
        mapping[left_owner] = right_owner
    return mapping


def rebind_replica_graph(
    template: Sequence[Any], *, replica: int, placement: Any,
    request_binding: Mapping[str, str] | None,
    state_bindings: Mapping[str, str] | None,
    physical_rank: int | None = None,
    case: Any = None,
    topology: Any = None,
    local_groups_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], ...]] | None = None,
    meta_prefix_cache: Mapping[int, bool] | None = None,
) -> tuple[Any, ...]:
    """Rebind rank/request/state identities of one semantic template."""
    rank_map = {int(key): int(value) for key, value in dict(placement or {}).items()}
    ep_map = _ep_owner_map(case, rank_map, replica)
    fallback = int(replica if physical_rank is None else physical_rank)
    requests = {str(key): str(value)
                for key, value in dict(request_binding or {}).items()}
    request_pattern = _request_binding_pattern(requests)
    request_prefix = _common_prefix(tuple(requests)) if requests else ""
    states = dict(state_bindings or {})
    groups_cache = {} if local_groups_cache is None else local_groups_cache
    def _rebind_ident(text: str) -> str:
        out = text
        if "@r" in out:
            if rank_map:
                out = RANK_TOKEN.sub(
                    lambda match: (
                        f"@r{rank_map.get(int(match.group(RANK_GROUP)), int(match.group(RANK_GROUP)))}"
                    ),
                    out,
                )
            else:
                out = RANK_TOKEN.sub(lambda _match: f"@r{fallback}", out)
        if request_pattern is not None:
            out = _rebind_nested_requests(out, request_pattern, requests, request_prefix)
        return out

    rebound = []
    for op in template:
        op_id = _rebind_ident(str(getattr(op, "op_id", "")))
        raw_meta = getattr(op, "metadata", ()) or ()
        meta_may_have_req = (
            request_pattern is not None
            and (meta_prefix_cache is None or meta_prefix_cache.get(id(raw_meta), True))
        )
        metadata = dict(raw_meta)
        meta_changed = False
        for key in ("rank", "destination_rank", "source_rank"):
            if key in metadata and isinstance(metadata[key], int) and not isinstance(metadata[key], bool):
                nv = rank_map.get(int(metadata[key]), metadata[key])
                if nv != metadata[key]:
                    metadata[key] = nv
                    meta_changed = True
        for key in ("ep_rank", "routing_price_owner"):
            if key in metadata and isinstance(metadata[key], int) and not isinstance(metadata[key], bool):
                nv = ep_map.get(int(metadata[key]), metadata[key])
                if nv != metadata[key]:
                    metadata[key] = nv
                    meta_changed = True
        if "group_ranks" in metadata:
            before_gr = metadata["group_ranks"]
            topology = _rebind_group_ranks(
                metadata, op, replica, rank_map, case, topology, groups_cache)
            if metadata["group_ranks"] != before_gr:
                meta_changed = True
        for key in ("state_owner", "recurrent_state_owner", "input_state_version",
                    "published_state_version"):
            if key in metadata and str(metadata[key]) in states:
                nv = states[str(metadata[key])]
                if nv != metadata[key]:
                    metadata[key] = nv
                    meta_changed = True
        if "replica" in metadata and metadata["replica"] != int(replica):
            metadata["replica"] = int(replica)
            meta_changed = True
        if meta_may_have_req:
            rebound_meta = _rebind_nested_requests(
                metadata, request_pattern, requests, request_prefix, meta_prefix_cache)
            if rebound_meta is not metadata:
                metadata = dict(rebound_meta)
                meta_changed = True
        deps = tuple(
            _rebind_ident(str(dep))
            for dep in (getattr(op, "deps", ()) or ()))
        new_meta = (
            tuple(sorted(metadata.items(), key=lambda pair: pair[0]))
            if meta_changed else raw_meta
        )
        rebound.append(replace(
            op, op_id=op_id, deps=deps,
            requests=tuple(requests.get(str(request), str(request))
                           for request in (getattr(op, "requests", ()) or ())),
            metadata=new_meta,
        ))
    return tuple(rebound)


def _request_ids(graph: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(request) for op in graph
        for request in (getattr(op, "requests", ()) or ())))



def prepare_cohort_graphs(
    case: Any, plans: Mapping[int, Any], *, reuse: str,
    graph_cache: SemanticGraphCache | None = None,
) -> PreparedCohort:
    """Prepare full or template-reused cohort graphs without dropping traffic."""
    mode = str(reuse)
    if mode not in REUSE_MODES:
        raise ModelingError(f"homogeneous_graph_reuse_unsupported:{reuse}")
    graphs = {int(replica): _ops(value) for replica, value in plans.items()}
    if mode == REUSE_OFF:
        proof = CohortReuseProof(
            False, ("reuse_off",), fallback_reason="reuse_off",
            replicas=len(graphs))
        return PreparedCohort(graphs, proof, False, False, "reuse_off")
    proof = prove_homogeneous_cohort(case, graphs)
    if mode == REUSE_REQUIRE and not proof.template_reused:
        raise ModelingError(
            "homogeneous_graph_reuse_proof_required:" + ",".join(proof.reasons))
    if not proof.template_reused:
        return PreparedCohort(graphs, proof, False, False,
                              ",".join(proof.reasons) or "proof_failed")
    from mindie_sim.prediction.topology import Topology

    shared_topology = Topology(case.machine, case.runtime.parallel)
    local_groups_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], ...]] = {}
    template_rep = min(graphs)
    request_ids = {replica: _request_ids(graph)
                   for replica, graph in graphs.items()}
    rank_maps = {
        replica: _rank_map(
            case, replica, template_replica=template_rep, topology=shared_topology,
        )
        for replica in graphs
    }
    ranks_by_replica = {
        replica: (min(mapping.values()) if mapping else int(replica))
        for replica, mapping in rank_maps.items()
    }
    cache_stats = {}
    if graph_cache is not None:
        tpl_slots, tpl_pat, tpl_pfx = _request_slot_map(graphs[template_rep])
        tpl_rank_slots = {
            int(rk): str(idx)
            for idx, rk in enumerate(sorted(rank_maps[template_rep]))
        }
        key = semantic_graph_key(
            case, None, scope="cohort",
            selection_signature=(proof.reasons, tuple(sorted(graphs)),
                                 _graph_signature(
                                     graphs[template_rep], tpl_pat, tpl_slots, tpl_pfx,
                                     local_rank_slots=tpl_rank_slots)))
        template = graph_cache.get_or_build(key, lambda: graphs[template_rep])
        cache_stats = graph_cache.stats()
    else:
        template = graphs[template_rep]
    template_req_ids = _request_ids(template)
    template_req_pfx = _common_prefix(template_req_ids)
    meta_prefix_cache: dict[int, bool] | None = None
    if template_req_pfx:
        meta_prefix_cache = {}
        for op in template:
            raw_meta = getattr(op, "metadata", ()) or ()
            _scan_meta_prefix(raw_meta, template_req_pfx, meta_prefix_cache)
    rebound: dict[int, tuple[Any, ...]] = {}
    for replica in sorted(graphs):
        replica_requests = request_ids[replica]
        if replica == template_rep and replica_requests == template_req_ids:
            rebound[replica] = tuple(template)
            continue
        request_binding = (
            dict(zip(template_req_ids, replica_requests))
            if len(template_req_ids) == len(replica_requests)
            else {})
        rebound[replica] = rebind_replica_graph(
            template, replica=replica, placement=rank_maps[replica],
            request_binding=request_binding, state_bindings=None,
            physical_rank=ranks_by_replica.get(replica, replica),
            case=case, topology=shared_topology,
            local_groups_cache=local_groups_cache,
            meta_prefix_cache=meta_prefix_cache)
    return PreparedCohort(
        rebound, proof, True, proof.execution_quotient,
        proof.fallback_reason, cache_stats)
