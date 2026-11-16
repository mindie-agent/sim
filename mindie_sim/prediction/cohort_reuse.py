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


ZERO = 0
ONE = 1
FIRST = 0
RANK_GROUP = 1
REUSE_OFF = "off"
REUSE_AUTO = "auto"
REUSE_REQUIRE = "require"
REUSE_MODES = frozenset({REUSE_OFF, REUSE_AUTO, REUSE_REQUIRE})
RANK_TOKEN = re.compile(r"@r(\d+)")
REQUEST_PATTERN_CACHE_LIMIT = 4096
_REQUEST_PATTERNS: dict[str, re.Pattern] = {}


def _request_pattern(request: str) -> re.Pattern:
    """A whole request identity, never the ``r0`` inside ``memory@r0:s0``."""
    pattern = _REQUEST_PATTERNS.get(request)
    if pattern is None:
        pattern = re.compile(
            rf"(?<![\w@]){re.escape(request)}(?![\w@])")
        if len(_REQUEST_PATTERNS) >= REQUEST_PATTERN_CACHE_LIMIT:
            _REQUEST_PATTERNS.clear()
        _REQUEST_PATTERNS[request] = pattern
    return pattern


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


def _normalized(value: Any, requests: Sequence[str] = ()) -> Any:
    if isinstance(value, str):
        text = RANK_TOKEN.sub("@r*", value)
        for request in requests:
            if request:
                text = _request_pattern(request).sub("@request", text)
        return text
    if isinstance(value, Mapping):
        return tuple(sorted(
            ((str(key), _normalized(item, requests))
             for key, item in value.items()
             if str(key) not in {"rank", "ep_rank", "group_ranks", "replica"}),
            key=lambda pair: pair[FIRST]))
    if isinstance(value, (tuple, list)):
        return tuple(_normalized(item, requests) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_normalized(item, requests) for item in value),
                            key=repr))
    return value


def _op_signature(op: Any, requests: Sequence[str] = ()) -> tuple:
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
        _normalized(tuple(getattr(op, "deps", ()) or ()), requests),
        _normalized(tuple(getattr(op, "matrix_flops", ()) or ()), requests),
        _normalized(dict(getattr(op, "metadata", ()) or {}).get("matrix_flops"),
                     requests),
        len(tuple(getattr(op, "requests", ()) or ())),
        _normalized(dict(getattr(op, "metadata", ()) or {}), requests),
    )


def _graph_signature(ops: Sequence[Any], requests: Sequence[str] = ()) -> tuple:
    return tuple(_op_signature(op, requests) for op in ops)


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
    ordered = [_ops(plans[key]) for key in sorted(plans)]
    request_tokens = _request_tokens(ordered)
    signatures = [_graph_signature(graph, request_tokens) for graph in ordered]
    reasons: list[str] = []
    if any(signature != signatures[ZERO] for signature in signatures[ONE:]):
        reasons.append("asymmetric_replica_graph")
    if selected_implementations is not None:
        selections = [selected_implementations.get(key)
                      for key in sorted(plans)]
        if any(item != selections[ZERO] for item in selections[ONE:]):
            reasons.append("asymmetric_implementation_selection")
    if initial_cache_states is not None:
        states = [initial_cache_states.get(key) for key in sorted(plans)]
        if any(item != states[ZERO] for item in states[ONE:]):
            reasons.append("asymmetric_cache_state")
    template_reused = not reasons
    if not template_reused:
        return CohortReuseProof(False, tuple(reasons), replicas=replicas)
    quotient = True
    for value in ordered:
        for op in _ops(value):
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


def _rank_map(case: Any, replica: int) -> dict[int, int]:
    from .topology import Topology

    parallel = case.runtime.parallel
    topology = Topology(case.machine, parallel)
    mapping: dict[int, int] = {}
    for stage in range(max(ONE, int(parallel.pp))):
        base = topology.ranks(ZERO, stage)
        target = topology.ranks(replica, stage)
        mapping.update({int(left): int(right)
                        for left, right in zip(base, target)})
    return mapping


def rebind_replica_graph(
    template: Sequence[Any], *, replica: int, placement: Any,
    request_binding: Mapping[str, str] | None,
    state_bindings: Mapping[str, str] | None,
    physical_rank: int | None = None,
) -> tuple[Any, ...]:
    """Rebind rank/request/state identities of one semantic template."""
    rank_map = {int(key): int(value) for key, value in dict(placement or {}).items()}
    fallback = int(replica if physical_rank is None else physical_rank)
    requests = dict(request_binding or {})
    states = dict(state_bindings or {})
    rebound = []
    for op in template:
        op_id = str(getattr(op, "op_id", ""))
        op_id = RANK_TOKEN.sub(
            lambda match: f"@r{rank_map.get(int(match.group(RANK_GROUP)), fallback)}", op_id)
        metadata = dict(getattr(op, "metadata", ()) or {})
        for key in ("rank", "ep_rank", "destination_rank", "source_rank"):
            if key in metadata and isinstance(metadata[key], int):
                metadata[key] = rank_map.get(int(metadata[key]), metadata[key])
        if "group_ranks" in metadata:
            metadata["group_ranks"] = tuple(
                rank_map.get(int(rank), int(rank))
                for rank in metadata["group_ranks"])
        for key in ("state_owner", "recurrent_state_owner", "input_state_version",
                    "published_state_version"):
            if key in metadata and str(metadata[key]) in states:
                metadata[key] = states[str(metadata[key])]
        if "replica" in metadata:
            metadata["replica"] = int(replica)
        deps = tuple(
            RANK_TOKEN.sub(
                lambda match: f"@r{rank_map.get(int(match.group(RANK_GROUP)), fallback)}",
                str(dep)) for dep in (getattr(op, "deps", ()) or ()))
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


def _template_rank(case: Any, replica: int) -> int:
    mapping = _rank_map(case, replica)
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
    template_rep = min(graphs)
    request_ids = {replica: _request_ids(graph)
                   for replica, graph in graphs.items()}
    ranks_by_replica = {replica: _template_rank(case, replica)
                        for replica in graphs}
    cache_stats = {}
    if graph_cache is not None:
        key = semantic_graph_key(
            case, None, scope="cohort",
            selection_signature=(proof.reasons, tuple(sorted(graphs)),
                                 _graph_signature(
                                     graphs[template_rep],
                                     _request_tokens(tuple(graphs.values())))))
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
            template, replica=replica, placement=_rank_map(case, replica),
            request_binding=request_binding, state_bindings=None,
            physical_rank=ranks_by_replica.get(replica, replica))
    return PreparedCohort(
        rebound, proof, True, proof.execution_quotient,
        proof.fallback_reason, cache_stats)
