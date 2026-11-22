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

from dataclasses import dataclass, field, replace
import re
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from .graph_cache import SemanticGraphCache, semantic_graph_key
from .local_tp_groups import local_tp_groups


ZERO = 0
ONE = 1
FIRST = 0
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
    body = parts[FIRST] if len(parts) == ONE else "(?:" + "|".join(parts) + ")"
    if has_end:
        return f"(?:{body})?" if len(parts) == ONE and len(parts[FIRST]) > ONE else f"{body}?"
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
    replicas: int = ZERO


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


def _normalized(
    value: Any,
    request_pattern: re.Pattern | None = None,
    request_set: frozenset[str] = frozenset(),
) -> Any:
    if isinstance(value, str):
        if request_set and value in request_set:
            return "@request"
        text = RANK_TOKEN.sub("@r*", value) if "@r" in value else value
        if request_pattern is not None:
            text = request_pattern.sub("@request", text)
        return text
    if isinstance(value, Mapping):
        return tuple(sorted(
            ((str(key), _normalized(item, request_pattern, request_set))
             for key, item in value.items()
             if str(key) not in {"rank", "ep_rank", "group_ranks", "replica"}),
            key=lambda pair: pair[FIRST]))
    if isinstance(value, (tuple, list)):
        return tuple(_normalized(item, request_pattern, request_set) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(
            (_normalized(item, request_pattern, request_set) for item in value),
            key=repr))
    return value


def _op_signature(
    op: Any,
    request_pattern: re.Pattern | None = None,
    request_set: frozenset[str] = frozenset(),
) -> tuple:
    metadata = dict(getattr(op, "metadata", ()) or {})
    return (
        str(getattr(op, "family", "")), str(getattr(op, "recipe", "")),
        str(getattr(op, "phase", "")), int(getattr(op, "stage", ZERO)),
        str(getattr(op, "dtype", "")), str(getattr(op, "layout", "")),
        tuple(getattr(op, "semantic_shape", ()) or ()),
        tuple(getattr(op, "local_shape", ()) or ()),
        tuple(getattr(op, "padded_shape", ()) or ()),
        tuple(getattr(op, "launch_shape", ()) or ()),
        float(getattr(op, "flops", ZERO) or ZERO),
        float(getattr(op, "hbm_bytes", ZERO) or ZERO),
        float(getattr(op, "mte2_bytes", ZERO) or ZERO),
        float(getattr(op, "mte3_bytes", ZERO) or ZERO),
        float(getattr(op, "vector_ops", ZERO) or ZERO),
        float(getattr(op, "link_bytes", ZERO) or ZERO),
        str(getattr(op, "source_boundary", "")),
        str(getattr(op, "stream", "") or ""),
        _normalized(tuple(getattr(op, "deps", ()) or ()), request_pattern, request_set),
        _normalized(tuple(getattr(op, "matrix_flops", ()) or ()), request_pattern, request_set),
        _normalized(metadata.get("matrix_flops"), request_pattern, request_set),
        len(tuple(getattr(op, "requests", ()) or ())),
        _normalized(metadata, request_pattern, request_set),
    )


def _graph_signature(
    ops: Sequence[Any],
    request_pattern: re.Pattern | None = None,
    request_set: frozenset[str] = frozenset(),
) -> tuple:
    return tuple(_op_signature(op, request_pattern, request_set) for op in ops)


def _request_tokens(graphs: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    tokens = {str(request) for graph in graphs for op in graph
              for request in (getattr(op, "requests", ()) or ())}
    return tuple(sorted(tokens, key=lambda item: (-len(item), item)))


def prove_homogeneous_cohort(
    case: Any, plans: Mapping[int, Any], *, selected_implementations: Any = None,
    initial_cache_states: Any = None,
) -> CohortReuseProof:
    """Prove structural isomorphism of a complete replica cohort."""
    if not isinstance(plans, Mapping) or not plans:
        return CohortReuseProof(False, ("cohort_empty",), replicas=ZERO)
    replicas = len(plans)
    ordered_keys = sorted(plans)
    ordered = [_ops(plans[key]) for key in ordered_keys]
    reasons: list[str] = []
    if any(len(graph) != len(ordered[ZERO]) for graph in ordered[ONE:]):
        reasons.append("asymmetric_replica_graph")
    else:
        request_tokens = _request_tokens(ordered)
        req_set = frozenset(tok for tok in request_tokens if tok)
        req_pat = _compile_token_pattern(request_tokens)
        base_sig = _graph_signature(ordered[ZERO], req_pat, req_set)
        for graph in ordered[ONE:]:
            if _graph_signature(graph, req_pat, req_set) != base_sig:
                reasons.append("asymmetric_replica_graph")
                break
    if selected_implementations is not None:
        selections = [selected_implementations.get(key) for key in ordered_keys]
        if any(item != selections[ZERO] for item in selections[ONE:]):
            reasons.append("asymmetric_implementation_selection")
    if initial_cache_states is not None:
        states = [initial_cache_states.get(key) for key in ordered_keys]
        if any(item != states[ZERO] for item in states[ONE:]):
            reasons.append("asymmetric_cache_state")
    template_reused = not reasons
    if not template_reused:
        return CohortReuseProof(False, tuple(reasons), replicas=replicas)
    quotient = True
    for op in ordered[ZERO]:
        if getattr(op, "source_boundary", ""):
            quotient = False
            reasons.append("source_bound_collective")
        if float(getattr(op, "link_bytes", ZERO) or ZERO) > ZERO:
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


def _rebind_nested_requests(value: Any, pattern: re.Pattern | None,
                            binding: Mapping[str, str]) -> Any:
    """Rename request identities nested in metadata without touching types."""
    if pattern is None:
        return value
    if isinstance(value, str):
        return pattern.sub(lambda match: binding[match.group()], value)
    if isinstance(value, Mapping):
        return type(value)((key, _rebind_nested_requests(item, pattern, binding))
                           for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_rebind_nested_requests(item, pattern, binding)
                     for item in value)
    if isinstance(value, list):
        return [_rebind_nested_requests(item, pattern, binding)
                for item in value]
    if isinstance(value, (set, frozenset)):
        return type(value)(_rebind_nested_requests(item, pattern, binding)
                           for item in value)
    return value


def _rank_map(case: Any, replica: int, *, topology: Any = None) -> dict[int, int]:
    from .topology import Topology

    parallel = case.runtime.parallel
    topo = topology if topology is not None else Topology(case.machine, parallel)
    mapping: dict[int, int] = {}
    for stage in range(max(ONE, int(parallel.pp))):
        base = topo.ranks(ZERO, stage)
        target = topo.ranks(replica, stage)
        mapping.update({int(left): int(right)
                        for left, right in zip(base, target)})
    return mapping


def _rebind_group_ranks(
    metadata: dict[str, Any], op: Any, replica: int, rank_map: Mapping[int, int],
    case: Any, topology: Any, local_groups_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], ...]],
) -> Any:
    raw_group = tuple(int(rank) for rank in metadata["group_ranks"])
    width = int(metadata.get("local_tp_width", ONE) or ONE)
    stride = int(metadata.get("local_tp_stride", ONE) or ONE)
    if raw_group and all(rank in rank_map for rank in raw_group):
        metadata["group_ranks"] = tuple(rank_map[rank] for rank in raw_group)
        return topology
    if case is not None and width > ONE:
        if topology is None:
            from .topology import Topology
            topology = Topology(case.machine, case.runtime.parallel)
        stage = int(getattr(op, "stage", ZERO))
        anchor = topology.ranks(int(replica), stage)[FIRST]
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


def rebind_replica_graph(
    template: Sequence[Any], *, replica: int, placement: Any,
    request_binding: Mapping[str, str] | None,
    state_bindings: Mapping[str, str] | None,
    physical_rank: int | None = None,
    case: Any = None,
    topology: Any = None,
    local_groups_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], ...]] | None = None,
) -> tuple[Any, ...]:
    """Rebind rank/request/state identities of one semantic template."""
    rank_map = {int(key): int(value) for key, value in dict(placement or {}).items()}
    fallback = int(replica if physical_rank is None else physical_rank)
    requests = {str(key): str(value)
                for key, value in dict(request_binding or {}).items()}
    request_pattern = _request_binding_pattern(requests)
    states = dict(state_bindings or {})
    groups_cache = {} if local_groups_cache is None else local_groups_cache
    rebound = []
    for op in template:
        op_id = str(getattr(op, "op_id", ""))
        if "@r" in op_id:
            op_id = RANK_TOKEN.sub(
                lambda match: f"@r{rank_map.get(int(match.group(RANK_GROUP)), fallback)}", op_id)
        metadata = dict(getattr(op, "metadata", ()) or {})
        for key in ("rank", "ep_rank", "destination_rank", "source_rank"):
            if key in metadata and isinstance(metadata[key], int):
                metadata[key] = rank_map.get(int(metadata[key]), metadata[key])
        if "group_ranks" in metadata:
            topology = _rebind_group_ranks(
                metadata, op, replica, rank_map, case, topology, groups_cache)
        for key in ("state_owner", "recurrent_state_owner", "input_state_version",
                    "published_state_version"):
            if key in metadata and str(metadata[key]) in states:
                metadata[key] = states[str(metadata[key])]
        if "replica" in metadata:
            metadata["replica"] = int(replica)
        if request_pattern is not None:
            metadata = dict(_rebind_nested_requests(
                metadata, request_pattern, requests))
        deps = tuple(
            RANK_TOKEN.sub(
                lambda match: f"@r{rank_map.get(int(match.group(RANK_GROUP)), fallback)}",
                str(dep)) if "@r" in str(dep) else str(dep)
            for dep in (getattr(op, "deps", ()) or ()))
        rebound.append(replace(
            op, op_id=op_id, deps=deps,
            requests=tuple(requests.get(str(request), str(request))
                           for request in (getattr(op, "requests", ()) or ())),
            metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[FIRST])),
        ))
    return tuple(rebound)


def _request_ids(graph: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(request) for op in graph
        for request in (getattr(op, "requests", ()) or ())))


def _template_rank(case: Any, replica: int, *, topology: Any = None) -> int:
    mapping = _rank_map(case, replica, topology=topology)
    return min(mapping.values()) if mapping else int(replica)


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
    from .topology import Topology

    shared_topology = Topology(case.machine, case.runtime.parallel)
    local_groups_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], ...]] = {}
    template_rep = min(graphs)
    request_ids = {replica: _request_ids(graph)
                   for replica, graph in graphs.items()}
    rank_maps = {replica: _rank_map(case, replica, topology=shared_topology)
                 for replica in graphs}
    ranks_by_replica = {
        replica: (min(mapping.values()) if mapping else int(replica))
        for replica, mapping in rank_maps.items()
    }
    cache_stats = {}
    if graph_cache is not None:
        all_tokens = _request_tokens(tuple(graphs.values()))
        req_set = frozenset(tok for tok in all_tokens if tok)
        req_pat = _compile_token_pattern(all_tokens)
        key = semantic_graph_key(
            case, None, scope="cohort",
            selection_signature=(proof.reasons, tuple(sorted(graphs)),
                                 _graph_signature(
                                     graphs[template_rep], req_pat, req_set)))
        template = graph_cache.get_or_build(key, lambda: graphs[template_rep])
        cache_stats = graph_cache.stats()
    else:
        template = graphs[template_rep]
    rebound = {template_rep: tuple(template)}
    for replica in sorted(graphs):
        if replica == template_rep:
            continue
        replica_requests = request_ids[replica]
        request_binding = (
            dict(zip(request_ids[template_rep], replica_requests))
            if len(request_ids[template_rep]) == len(replica_requests)
            else {})
        rebound[replica] = rebind_replica_graph(
            template, replica=replica, placement=rank_maps[replica],
            request_binding=request_binding, state_bindings=None,
            physical_rank=ranks_by_replica.get(replica, replica),
            case=case, topology=shared_topology,
            local_groups_cache=local_groups_cache)
    return PreparedCohort(
        rebound, proof, True, proof.execution_quotient,
        proof.fallback_reason, cache_stats)
