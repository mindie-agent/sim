"""MoE expert placement, balanced routing, EPLB and routing realizations.

[Role] Compute expert-parallel token routing distributions, EPLB slot maps,
  and deterministic routing realizations for MoE layers.
[Invariants] Conserves total routed token assignments across EP ranks.
[Timeouts & Locks] Pure deterministic placement; no I/O or locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from mindie_sim.schema.common import ModelingError


def _rank_rows(counts, ep, rank):
    """Expert rows owned by one EP rank.

    Stride across the ledger when ``ep`` does not exceed the expert count.
    When more ranks than experts (replicated draft experts, for example 16
    experts under EP32/EP64), split that expert's rows across
    ``ep // len(counts)`` replicas so the rank ledger conserves the total.
    """
    width = len(counts)
    if ep <= width:
        return counts[rank::ep]
    expert = rank % width
    replicas = 1 + (ep - 1 - expert) // width
    rep = rank // width
    return (counts[expert] // replicas + int(rep < counts[expert] % replicas),)


@dataclass(frozen=True)
class BalancedRouting:
    expert_rows: tuple[int, ...]
    local_expert_rows: tuple[int, ...]
    ep: int
    source_partitions: int
    global_tokens: int
    valid_local_tokens: int
    padded_local_tokens: int
    representative_rank: int = 0
    represented_ranks: tuple[int, ...] = (0,)
    expected_rows: Fraction | None = None

    @property
    def rows(self):
        return sum(self.local_expert_rows)

    @property
    def rank_assignment_rows(self):
        return tuple(sum(_rank_rows(self.expert_rows, self.ep, rank))
                     for rank in range(self.ep))

    @property
    def active(self):
        return sum(value > 0 for value in self.local_expert_rows)

    @property
    def basis(self):
        return "forced_balanced_integer_counts"

    def _expert_ids(self):
        """Strided expert ids, or the one replicated expert when EP is wider."""
        count = len(self.expert_rows)
        if self.ep <= count:
            return tuple(range(self.representative_rank, count, self.ep))
        return (self.representative_rank % count,)

    def to_dict(self):
        return {
            "basis": self.basis, "global_tokens": self.global_tokens,
            "global_expert_rows": self.expert_rows,
            "local_expert_rows": self.local_expert_rows,
            "representative_expert_ids": self._expert_ids(),
            "expert_placement": "strided_ep", "matrix_groups": self.active,
            "rows_per_active_expert": (self.rows // self.active
                                       if self.active and len(set(self.local_expert_rows) - {0}) == 1
                                       else None),
            "row_classes": tuple(sorted(Counter(self.local_expert_rows).items())),
            "rank_assignment_rows": tuple(sum(_rank_rows(self.expert_rows, self.ep, rank))
                                          for rank in range(self.ep)),
            "representative_rank": self.representative_rank,
            "represented_ep_ranks": self.represented_ranks,
            "rank_timing_assumption": ("exact_physical_rank" if len(self.represented_ranks) == 1
                                       else "heaviest_local_tp_shard_upper_bound"),
            "cohort_counts_basis": "exact_integer_expert_assignment_ledger",
            "expected_rows_per_expert": (float(self.expected_rows)
                                         if self.expected_rows is not None else None),
            "valid_local_tokens": self.valid_local_tokens,
            "padded_local_tokens": self.padded_local_tokens,
            "source_partitions": self.source_partitions,
            "communication_assumption": "uniform_peer_payload",
            "token_identities": "not_declared",
        }


def balanced_realization(runtime, layer, *, tokens, valid_tokens, global_tokens,
                         token_shards=1, valid_source_tokens=None, replica=0, owner=None):
    p = runtime.parallel
    if runtime.moe_recipe not in {"alltoall", "tp_join_reduce", "local_expert_reduce"}:
        raise ModelingError("balanced_routing_requires_pcp1_unchunked_alltoall")
    width = p.ep_dp_width
    if width > 1 and global_tokens is None:
        raise ModelingError("balanced_routing_requires_complete_dp_cohort:use_target_graphs")
    total = tokens * token_shards if global_tokens is None else global_tokens
    if token_shards != 1 and (
            token_shards != p.tp or (owner is None and total < tokens * token_shards)):
        raise ModelingError("balanced_routing_sequence_shards_require_complete_cohort")
    assignments = total * layer.top_k
    q, r = divmod(assignments, layer.num_experts)
    counts = tuple(q + (index < r) for index in range(layer.num_experts))
    local_ep = min(p.tp * p.pcp, p.ep) if p.ep_spans_pcp else min(p.tp, p.ep)
    first = (replica % width) * local_ep if p.ep_spans_dp else 0
    ranks = tuple(range(first, first + local_ep))
    if owner is not None:
        ranks = (owner,)
    representative = max(ranks, key=lambda rank: sum(_rank_rows(counts, p.ep, rank)))
    local_rows = _rank_rows(counts, p.ep, representative)
    # PCP prefill tokens are already split across PCP ranks. Dividing by the
    # PCP-extended local EP width would shrink dispatch/combine a second time.
    return BalancedRouting(
        counts, local_rows, p.ep,
        max(1, min(p.tp, p.ep) // token_shards), total,
        (valid_tokens * token_shards if valid_source_tokens is None
         else valid_source_tokens), tokens * token_shards,
        representative, ranks,
        Fraction(total * max(1, layer.top_k), p.ep))


def cohort_padded_tokens(groups, runtime):
    """Count every router row, including declared per-replica capture padding."""
    from dataclasses import replace
    from mindie_sim.prediction.graph import GraphPolicy
    total = 0
    for views in groups:
        expanded = tuple(replace(v, q_len=v.q_len + v.draft_proposal_tokens) for v in views)
        valid = sum(v.q_len for v in expanded)
        bucket, _ = GraphPolicy().select(expanded, runtime) if expanded else (None, "none")
        physical = max(valid, bucket or 0)
        sp = runtime.kernels.get("sequence_parallel", {})
        if sp.get("padding") == "right_to_tp":
            physical += (-physical) % runtime.parallel.tp
        total += physical
    return total


from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from mindie_sim.schema.common import ModelingError

SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "cursor": "vllm_ascend/ops/fused_moe/force_eplb.py:130-142",
    "mask": "vllm_ascend/ascend_forward_context.py:217-226",
    "mask_slice": "vllm_ascend/ops/fused_moe/prepare_finalize.py:273-284",
    "dispatch_mask": "vllm_ascend/ops/fused_moe/token_dispatcher.py:127-144,177-186",
    "capture_binary_identity": "not_established; explicit_source_scenario",
}


@dataclass(frozen=True)
class SourceForceEplb:
    expert_rows: tuple[int, ...]
    rank_expert_rows: tuple[tuple[int, ...], ...]
    peer_assignments: tuple[tuple[int, ...], ...]
    valid_tokens_per_rank: tuple[int, ...]
    allocated_tokens_per_rank: int
    top_k: int
    ep: int
    valid_global_tokens: int
    representative_rank: int = 0
    source_partitions: int = 1
    allocated_by_rank: tuple[int, ...] = ()

    @property
    def local_expert_rows(self):
        return self.rank_expert_rows[self.representative_rank]

    @property
    def valid_local_tokens(self):
        return self.valid_tokens_per_rank[self.representative_rank]

    @property
    def rows(self):
        return sum(self.local_expert_rows)

    @property
    def active(self):
        return sum(count > 0 for count in self.local_expert_rows)

    @property
    def basis(self):
        return "source_force_eplb_masked_integer_counts"

    def payload_matrix(self, row_bytes, *, reverse=False):
        """Local cells have source work but never consume a network link."""
        return tuple(tuple(0 if src == dst else row_bytes * (
            self.peer_assignments[dst][src] if reverse else self.peer_assignments[src][dst])
            for dst in range(self.ep)) for src in range(self.ep))

    def to_dict(self):
        banks = len(self.local_expert_rows)
        return {
            "basis": self.basis, "source": SOURCE,
            "expert_placement": "contiguous_ep",
            "global_expert_rows": self.expert_rows,
            "local_expert_rows": self.local_expert_rows,
            "rank_expert_rows": self.rank_expert_rows,
            "rank_assignment_rows": tuple(map(sum, self.rank_expert_rows)),
            "peer_assignment_rows": self.peer_assignments,
            "representative_expert_ids": tuple(range(self.representative_rank * banks,
                                                      (self.representative_rank + 1) * banks)),
            "representative_rank": self.representative_rank,
            "rank_timing_assumption": "declared_physical_rank_compute; complete_peer_network",
            "matrix_groups": self.active,
            "row_classes": tuple(sorted(Counter(self.local_expert_rows).items())),
            "valid_global_tokens": self.valid_global_tokens,
            "allocated_global_tokens": (sum(self.allocated_by_rank) if self.allocated_by_rank
                                        else self.allocated_tokens_per_rank * self.ep),
            "valid_local_tokens": self.valid_local_tokens,
            "source_local_rows": self.allocated_tokens_per_rank,
            "valid_tokens_per_rank": self.valid_tokens_per_rank,
            "allocated_tokens_per_rank": self.allocated_tokens_per_rank,
            "top_k": self.top_k,
            "ep": self.ep,
            "source_partitions": self.source_partitions,
            "communication_assumption": "source_cursor_and_active_mask_exact_integer_matrix",
            "cursor": "(allocated_local_rows*top_k+1)*source_rank + local_assignment",
            "mask": "contiguous_global_valid_prefix; global_bs=0",
        }

    def token_order(self):
        """Expand the declared cursor/mask into concrete local top-k rows.

        The expansion is derived from the same source cursor and valid-prefix
        mask as :func:`ledger`; it is not a captured token identity.  It gives
        T12 a concrete occurrence while the source scenario label remains.
        """
        banks = len(self.expert_rows) // self.ep
        rows = []
        for source, count in enumerate(self.valid_tokens_per_rank):
            for token in range(count):
                ids = []
                width = (self.allocated_by_rank[source] if self.allocated_by_rank
                         else self.allocated_tokens_per_rank)
                for slot in range(self.top_k):
                    cursor = (width * self.top_k + 1) * source + token * self.top_k + slot
                    destination = cursor % self.ep
                    bank = (cursor // self.ep) % banks
                    ids.append(destination * banks + bank)
                rows.append((source, tuple(ids)))
        counts = [0] * len(self.expert_rows)
        peer = [[0] * self.ep for _ in range(self.ep)]
        for source, ids in rows:
            for expert in ids:
                counts[expert] += 1
                peer[source][expert // banks] += 1
        if tuple(counts) != self.expert_rows:
            raise ModelingError("source_force_eplb_token_order_expert_rows_mismatch")
        if tuple(tuple(row) for row in peer) != self.peer_assignments:
            raise ModelingError("source_force_eplb_token_order_peer_mismatch")
        return tuple(rows)

    def typed_occurrence(self, *, payload_scale_producer, native_top_k=1):
        """One concrete source occurrence for a T12 typed-call binding."""
        if type(native_top_k) is not int or native_top_k < 1:
            raise ModelingError("source_force_eplb_native_top_k_invalid")
        order = self.token_order()
        banks = len(self.expert_rows) // self.ep
        return {
            "basis": self.basis,
            "source": SOURCE,
            "capture_identity": SOURCE["capture_binary_identity"],
            "source_owners": tuple(source for source, _ids in order),
            "token_order": tuple(ids for _source, ids in order),
            "token_order_basis": "derived_from_source_cursor",
            "expert_placement": tuple(
                (expert // banks, expert % banks)
                for expert in range(len(self.expert_rows))),
            "source_destination_rows": self.peer_assignments,
            "destination_bank_rows": self.rank_expert_rows,
            "expert_rows": self.expert_rows,
            "valid_tokens_per_rank": self.valid_tokens_per_rank,
            "allocated_tokens_per_rank": self.allocated_tokens_per_rank,
            "representative_rank": self.representative_rank,
            "model_top_k": self.top_k,
            "native_top_k": native_top_k,
            "payload_scale_producer": dict(payload_scale_producer),
        }


FORCE_LEDGER_CACHE_SIZE = 128


@lru_cache(maxsize=FORCE_LEDGER_CACHE_SIZE)
def ledger(experts, ep, top_k, allocated_local_tokens, valid_global_tokens, tp=0,
           valid_by_rank=(), allocated_by_rank=(), owner=0):
    """Count source cursor residues without expanding token/expert assignments."""
    if (ep < 2 or experts % ep or not 0 < top_k <= experts
            or allocated_local_tokens < 0
            or not 0 <= valid_global_tokens <= (sum(allocated_by_rank) if allocated_by_rank
                                                else allocated_local_tokens * ep)):
        raise ModelingError("source_force_eplb_requires_divisible_banks_and_bounded_valid_prefix")
    banks = experts // ep
    if valid_by_rank:
        valid = valid_by_rank
        if (len(valid) != ep or len(allocated_by_rank) != ep
                or sum(valid) != valid_global_tokens
                or any(not 0 <= n <= w for n, w in zip(valid, allocated_by_rank))):
            raise ModelingError("source_force_eplb_invalid_rank_ownership")
    elif 0 < tp < ep and ep % tp == 0:
        groups = ep // tp
        base, extra = divmod(valid_global_tokens, groups)
        valid = tuple(max(0, min(allocated_local_tokens, base + (d < extra)
                                 - t * allocated_local_tokens))
                      for d in range(groups) for t in range(tp))
    else:
        valid = tuple(max(0, min(allocated_local_tokens,
                                valid_global_tokens - src * allocated_local_tokens)) for src in range(ep))
    expert_counts = [0] * experts
    traffic = [[0] * ep for _ in range(ep)]
    for src, rows in enumerate(valid):
        rounds, tail = divmod(rows * top_k, experts)
        width = allocated_by_rank[src] if allocated_by_rank else allocated_local_tokens
        start = ((width * top_k + 1) * src) % experts
        for residue in range(experts):
            count = rounds + ((residue - start) % experts < tail)
            dest, bank = residue % ep, residue // ep
            expert_counts[dest * banks + bank] += count
            traffic[src][dest] += count
    return SourceForceEplb(tuple(expert_counts),
                          tuple(tuple(expert_counts[r*banks:(r+1)*banks]) for r in range(ep)),
                          tuple(map(tuple, traffic)), valid, allocated_local_tokens,
                          top_k, ep, sum(valid), representative_rank=owner,
                          allocated_by_rank=allocated_by_rank)


def select(runtime, layer, *, tokens, valid_tokens, global_tokens,
           token_shards, valid_source_tokens, replica, source_geometry=None):
    raw = runtime.kernels.get("force_eplb_source")
    if raw is None:
        return None
    p = runtime.parallel
    wrapper = runtime.kernels.get("moe_token_partition")
    if raw == {"kind": "vllm_ascend_mc2_tp", "global_bs": 0}:
        if (runtime.routing != {"policy": "balanced"} or p.dp < 1 or p.ep < p.tp
                or p.ep % p.tp != 0 or p.ep < 2 or p.cp != 1 or p.pcp < 1
                or p.pp < 1 or replica < 0
                or runtime.moe_recipe != "alltoall" or token_shards != 1
                or not wrapper or wrapper.get("kind") != "vllm_ascend_mc2_tp_slice"
                or runtime.kernels.get("sequence_parallel")):
            raise ModelingError("source_force_eplb_requires_explicit_mc2_tp_caller")
        if source_geometry is not None:
            g = source_geometry
            return ledger(layer.num_experts, p.ep, layer.top_k, g.allocated[g.owner],
                          sum(g.valid), p.tp, g.valid, g.allocated, g.owner)
        if (global_tokens not in (None, tokens, tokens * (p.ep // p.tp))
                or not 0 < valid_tokens <= tokens):
            raise ModelingError("source_force_eplb_requires_declared_caller_valid_prefix")
        return ledger(layer.num_experts, p.ep, layer.top_k,
                      (tokens + p.tp - 1) // p.tp, valid_tokens * (p.ep // p.tp), p.tp)
    if raw != {"kind": "vllm_ascend_sp_mc2", "global_bs": 0}:
        raise ModelingError("force_eplb_source_requires_explicit_vllm_ascend_sp_mc2_global_bs0")
    if (runtime.routing != {"policy": "balanced"} or p.dp < 1 or p.ep < p.tp
            or p.ep % p.tp != 0 or p.ep < 2 or p.cp != 1 or p.pcp < 1
            or p.pp < 1 or replica < 0
            or runtime.moe_recipe != "alltoall" or token_shards != p.tp
            or wrapper
            or runtime.kernels.get("native_moe_protocol", {}).get("kind") != "arch35_fullmesh"):
        raise ModelingError("source_force_eplb_requires_sp_dp1_ep_tp_mc2_alltoall")
    if source_geometry is not None:
        g = source_geometry
        return ledger(layer.num_experts, p.ep, layer.top_k, g.allocated[g.owner],
                      sum(g.valid), p.tp, g.valid, g.allocated, g.owner)
    if (valid_source_tokens is None
            or global_tokens not in (None, tokens * p.tp, tokens * p.ep)
            or valid_tokens != min(tokens, valid_source_tokens)):
        raise ModelingError("source_force_eplb_requires_explicit_sp_valid_prefix")
    return ledger(layer.num_experts, p.ep, layer.top_k, tokens, valid_source_tokens * (p.ep // p.tp), p.tp)


from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

from mindie_sim.schema.common import ModelingError


_BASE_KEYS = {"scope", "layer", "phase", "tokens", "expert_rows"}


@dataclass(frozen=True)
class RoutingRealization:
    scope: str
    layer: int
    phase: str
    tokens: int
    expert_rows: tuple[int, ...]
    #: Explicit token -> expert identities. ``None`` retains histogram-only
    #: realizations; a source needing ids must not invent this mapping.
    topk_ids: tuple[tuple[int, ...], ...] | None = None

    @property
    def local_expert_rows(self):
        # Explicit histograms currently declare an EP=1 scenario; every bank
        # belongs to this rank. Forced-balance cohorts supply their own slice.
        return self.expert_rows

    @property
    def rows(self):
        return sum(self.expert_rows)

    @property
    def active(self):
        return sum(value > 0 for value in self.expert_rows)

    @property
    def basis(self):
        return "declared_integer_counts"

    def to_dict(self):
        payload = {
            "basis": "declared_integer_counts", "scope": self.scope,
            "layer": self.layer, "phase": self.phase, "tokens": self.tokens,
            "expert_rows": self.expert_rows,
            "expert_ids": tuple(range(len(self.expert_rows))),
            "reuse": "same_scope_layer_phase_tokens",
        }
        if self.topk_ids is not None:
            payload["topk_ids"] = self.topk_ids
        return payload


def _explicit_topk_ids(value, *, tokens, top_k, experts, rows):
    """Validate ids against the exact declared histogram without inference."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ModelingError("routing_realization_topk_ids_must_be_2d_sequence")
    if len(value) != tokens:
        raise ModelingError(
            "routing_realization_topk_ids_must_match_tokens:"
            f"{len(value)}!={tokens}")
    counts = [0] * experts
    result = []
    for token, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)):
            raise ModelingError(
                f"routing_realization_topk_ids_row_must_be_sequence:{token}")
        if len(row) != top_k:
            raise ModelingError(
                "routing_realization_topk_ids_must_match_top_k:"
                f"{token}:{len(row)}!={top_k}")
        seen = set()
        for expert in row:
            if type(expert) is not int or not 0 <= expert < experts:
                raise ModelingError(
                    "routing_realization_topk_ids_must_be_expert_identities:"
                    f"{token}:{expert!r}")
            if expert in seen:
                raise ModelingError(
                    f"routing_realization_topk_ids_duplicate_expert:{token}:{expert}")
            seen.add(expert)
            counts[expert] += 1
        result.append(tuple(row))
    if tuple(counts) != rows:
        raise ModelingError(
            "routing_realization_topk_ids_must_agree_with_expert_rows:"
            f"{tuple(counts)}!={rows}")
    return tuple(result)


def select_realization(runtime, layer, *, phase, tokens, valid_tokens, global_tokens=None,
                       token_shards=1, valid_source_tokens=None, replica=0, views=None,
                       physical_rank=None, query_position=None, source_geometry=None):
    raw = runtime.routing
    from mindie_sim.prediction.sharding.routing import select as source_realization
    selected = source_realization(
        runtime, layer, tokens=tokens, valid_tokens=valid_tokens,
        global_tokens=global_tokens, token_shards=token_shards,
        valid_source_tokens=valid_source_tokens, replica=replica, source_geometry=source_geometry)
    if selected is not None:
        return selected
    if raw.get("policy") == "balanced" and set(raw) == {"policy"}:
        from mindie_sim.prediction.sharding.routing import balanced_realization
        return balanced_realization(runtime, layer, tokens=tokens,
                                    valid_tokens=valid_tokens, global_tokens=global_tokens,
                                    token_shards=token_shards,
                                    valid_source_tokens=valid_source_tokens, replica=replica,
                                    owner=source_geometry.owner if source_geometry else None)
    if set(raw) - {"realizations"}:
        raise ModelingError("routing_configuration_requires_explicit_realizations")
    declarations = raw.get("realizations", ())
    matching = []
    for item in declarations:
        if set(item) - _EXPLICIT_KEYS:
            raise ModelingError("routing_realization_requires_scope_layer_phase_tokens_counts")
        if item["scope"] == layer.scope and item["layer"] == layer.index:
            matching.append(item)
    if not matching:
        return None
    explicit = [item for item in matching
                if any(key in item for key in (
                    "requests", "request_views", "expert_placement",
                    "source_owners", "source_owner", "query_position"))]
    if explicit:
        actual = _norm_views(views)
        selected_explicit = []
        # The realization's ``tokens`` is the logical query row count; a
        # captured graph may carry a larger physical bucket.  Match the
        # logical rows here and keep the physical capacity for the graph
        # geometry, so a template hit rebinds the current route.
        logical_tokens = (int(valid_tokens) if valid_tokens is not None
                          else int(tokens))
        for item in explicit:
            cohort = bool(item.get("cohort", False))
            if (not cohort and item.get("replica") is not None
                    and int(item["replica"]) != int(replica)):
                continue
            if item.get("phase") != phase:
                continue
            if not cohort and int(item.get("tokens", -1)) != logical_tokens:
                continue
            item_position = item.get("query_position")
            if item_position is not None:
                actual_positions = {
                    int(_view_field(view, "context", -1))
                    for view in actual.values()}
                if query_position is not None:
                    actual_positions.add(int(query_position))
                if (int(item_position) not in actual_positions
                        and not (cohort and query_position is not None
                                 and int(item_position) == int(query_position))):
                    continue
            request_views = item.get("requests", item.get("request_views"))
            if request_views is not None:
                if not actual:
                    continue
                declared = [str(entry.get("request_id", ""))
                            for entry in request_views if isinstance(entry, Mapping)]
                if not all(name in actual for name in declared):
                    continue
                if any(entry.get("query_position") is not None
                       and int(_view_field(actual[str(entry["request_id"])],
                                           "context", -1)) != int(entry["query_position"])
                       for entry in request_views if isinstance(entry, Mapping)):
                    continue
            selected_explicit.append(item)
        if len(selected_explicit) > 1:
            raise ModelingError(
                "routing_realization_ambiguous_query_position_or_source")
        if len(selected_explicit) == 1:
            return _explicit_routing(
                selected_explicit[0], layer, phase, logical_tokens, views, replica,
                physical_rank=physical_rank, query_position=query_position)
        if (int(runtime.parallel.ep) > 1 and logical_tokens > 0
                and any(item.get("phase") == phase for item in explicit)):
            raise ModelingError(
                "routing_realization_requires_matching_explicit_token_ledger")
        return None
    local_caller = runtime.moe_recipe == "local_expert_reduce"
    # PCP splits queries before this histogram is applied, and PP only places
    # layers. Neither changes the local top-k ledger, so both may compose.
    plain = [item for item in matching
             if not any(key in item for key in (
                 "requests", "request_views", "expert_placement",
                 "source_owners", "source_owner", "query_position"))]
    if not plain:
        return None
    matching = plain
    if ((runtime.parallel.ep != 1 and not local_caller) or runtime.parallel.pcp < 1
            or runtime.parallel.pp < 1
            or runtime.moe_recipe not in {"alltoall", "tp_join_reduce",
                                           "local_expert_reduce"}):
        raise ModelingError("routing_realization_requires_ep1_pcp1_unchunked_alltoall")
    if layer.router_groups > 1 and layer.router_group_topk < layer.router_groups:
        raise ModelingError("routing_realization_requires_unrestricted_expert_groups")
    if tokens != valid_tokens or global_tokens not in (None, tokens):
        raise ModelingError("routing_realization_requires_unpadded_local_tokens")
    selected = [item for item in matching
                if item["phase"] == phase and item["tokens"] == tokens]
    if len(selected) != 1:
        raise ModelingError(
            f"routing_realization_requires_one_matching_scenario:"
            f"{layer.scope}:{layer.index}:{phase}:{tokens}")
    item = selected[0]
    rows = tuple(item["expert_rows"])
    if (type(item["tokens"]) is not int or type(item["layer"]) is not int
            or len(rows) != layer.num_experts
            or any(type(count) is not int or not 0 <= count <= tokens for count in rows)
            or sum(rows) != tokens * layer.top_k):
        raise ModelingError("routing_realization_must_conserve_integer_topk_assignments")
    topk_ids = None
    if "topk_ids" in item:
        topk_ids = _explicit_topk_ids(
            item["topk_ids"], tokens=tokens, top_k=layer.top_k,
            experts=layer.num_experts, rows=rows)
    return RoutingRealization(layer.scope, layer.index, phase, tokens, rows, topk_ids)


_EXPLICIT_KEYS = _BASE_KEYS | {
    "topk_ids", "requests", "request_views", "expert_placement",
    "source_owners", "source_owner", "replica", "representative_rank",
    "query_position", "cohort",
}


@dataclass(frozen=True)
class ExplicitTokenRouting:
    """Exact source-token assignments and physical EP peer cells.

    The ledger carries logical top-k expert identity, the logical->physical
    expert placement, and one physical EP source owner per token.  It is not a
    statistical moment: peer_assignments are integer full-frame assignment
    counts and payload_matrix is their local-diagonal-free byte projection.
    """
    scope: str
    layer: int
    phase: str
    tokens: int
    topk_ids: tuple[tuple[int, ...], ...]
    expert_placement: tuple[tuple[int, int], ...]
    source_owners: tuple[int, ...]
    expert_rows: tuple[int, ...]
    rank_expert_rows: tuple[tuple[int, ...], ...]
    peer_assignments: tuple[tuple[int, ...], ...]
    representative_rank: int = 0
    replica: int = 0
    request_ids: tuple[str, ...] = ()
    query_positions: tuple[int, ...] = ()
    basis: str = "explicit_token_routing_integer"

    @property
    def local_expert_rows(self):
        if 0 <= self.representative_rank < len(self.rank_expert_rows):
            return self.rank_expert_rows[self.representative_rank]
        return tuple(0 for _ in range(len(self.expert_rows) // max(1, len(self.peer_assignments))))

    @property
    def ep(self):
        return len(self.peer_assignments)

    @property
    def local_weight_expert_ids(self):
        """Contiguous weight-bank slots owned by the representative rank."""
        return tuple(range(len(self.local_expert_rows)))

    @property
    def representative_expert_ids(self):
        """Logical expert ids in this rank's bank, ordered by physical bank."""
        placed = sorted(
            (bank, expert) for expert, (rank, bank) in enumerate(self.expert_placement)
            if rank == self.representative_rank)
        return tuple(expert for _bank, expert in placed)

    @property
    def rows(self):
        return sum(self.local_expert_rows)

    @property
    def active(self):
        return sum(value > 0 for value in self.local_expert_rows)

    @property
    def rank_assignment_rows(self):
        return tuple(sum(row[destination] for row in self.peer_assignments)
                     for destination in range(len(self.peer_assignments)))

    @property
    def valid_local_tokens(self):
        return sum(1 for owner in self.source_owners
                   if owner == self.representative_rank)

    @property
    def source_partitions(self):
        return 1

    def payload_matrix(self, row_bytes, *, reverse=False):
        """Integer assignment cells as physical bytes; self cells are local."""
        size = len(self.peer_assignments)
        return tuple(tuple(0 if src == dst else int(row_bytes) * (
            self.peer_assignments[dst][src] if reverse else
            self.peer_assignments[src][dst])
            for dst in range(size)) for src in range(size))

    def receipt(self):
        """Immutable consumed-object coverage for profiler/caller checks."""
        return {
            "basis": self.basis, "scope": self.scope, "layer": self.layer,
            "phase": self.phase, "tokens": self.tokens,
            "request_ids": self.request_ids,
            "query_positions": self.query_positions,
            "source_owners": self.source_owners,
            "expert_placement": self.expert_placement,
            "representative_rank": self.representative_rank,
            "rank_assignment_rows": self.rank_assignment_rows,
            "rows": self.rows, "active": self.active,
            "valid_local_tokens": self.valid_local_tokens,
            "topk_ids": self.topk_ids,
            "peer_assignments": self.peer_assignments,
        }

    def to_dict(self):
        return {
            "basis": self.basis, "scope": self.scope, "layer": self.layer,
            "phase": self.phase, "tokens": self.tokens,
            "global_tokens": self.tokens,
            "expert_rows": self.expert_rows,
            "global_expert_rows": self.expert_rows,
            "expert_ids": tuple(range(len(self.expert_rows))),
            "expert_placement": self.expert_placement,
            "source_owners": self.source_owners,
            "topk_ids": self.topk_ids,
            "rank_expert_rows": self.rank_expert_rows,
            "peer_assignments": self.peer_assignments,
            "rank_assignment_rows": self.rank_assignment_rows,
            "rows": self.rows, "active": self.active,
            "local_expert_rows": self.local_expert_rows,
            "local_weight_expert_ids": self.local_weight_expert_ids,
            "representative_expert_ids": self.representative_expert_ids,
            "valid_local_tokens": self.valid_local_tokens,
            "valid_global_tokens": self.tokens,
            "representative_rank": self.representative_rank,
            "replica": self.replica,
            "request_ids": self.request_ids,
            "query_positions": self.query_positions,
            "source_partitions": self.source_partitions,
            "rank_timing_assumption": "declared_physical_rank_compute",
            "token_identities": "explicit",
            "expert_rows_basis": "explicit_token_route",
            "reuse": "explicit_token_routing_positions",
            "communication_assumption": "integer_token_source_to_physical_ep_destination",
        }


def _norm_views(views):
    result = {}
    for view in views or ():
        getattr_id = getattr(view, "request_id", None)
        if getattr_id is None and isinstance(view, Mapping):
            getattr_id = view.get("request_id")
        if getattr_id is not None:
            result[str(getattr_id)] = view
    return result


def _view_field(view, key, default=None):
    if isinstance(view, Mapping):
        return view.get(key, default)
    return getattr(view, key, default)


_EXPLICIT_LEDGER_CACHE_SIZE = 128


@lru_cache(maxsize=_EXPLICIT_LEDGER_CACHE_SIZE)
def _explicit_ledger(topk, placement, source_owners, experts, top_k):
    """Build the shared cohort matrix once; per-replica projection is cheap."""
    if type(experts) is not int or experts < 1 or type(top_k) is not int or top_k < 1:
        raise ModelingError("explicit_token_routing_requires_positive_expert_geometry")
    if len(source_owners) != len(topk):
        raise ModelingError(
            "explicit_token_routing_source_owner_count_must_match_tokens:"
            f"{len(source_owners)}!={len(topk)}")
    ranks = sorted({rank for rank, _bank in placement})
    ep = max(ranks) + 1 if ranks else 1
    if ranks and ranks != list(range(ep)):
        raise ModelingError("explicit_token_routing_physical_ranks_must_be_contiguous")
    if ep > experts or experts % ep:
        raise ModelingError(
            "explicit_token_routing_requires_divisible_physical_expert_banks:"
            f"{experts}%{ep}")
    local = experts // ep
    occupied = set()
    for expert, (rank, bank) in enumerate(placement):
        actual_bank = expert % local if bank is None else bank
        if not 0 <= actual_bank < local:
            raise ModelingError(
                f"explicit_token_routing_bank_out_of_range:{expert}:{actual_bank}")
        if (rank, actual_bank) in occupied:
            raise ModelingError(
                f"explicit_token_routing_duplicate_physical_bank:{rank}:{actual_bank}")
        occupied.add((rank, actual_bank))
    rank_rows = [[0] * local for _ in range(ep)]
    expert_rows = [0] * experts
    peer = [[0] * ep for _ in range(ep)]
    for token, ids in enumerate(topk):
        if len(ids) != top_k:
            raise ModelingError(
                f"explicit_token_routing_topk_row_length_mismatch:{token}")
        seen = set()
        src = source_owners[token]
        if type(src) is not int or not 0 <= src < ep:
            raise ModelingError(f"explicit_token_routing_source_owner_out_of_ep:{src}")
        for expert in ids:
            if type(expert) is not int or not 0 <= expert < experts:
                raise ModelingError(
                    f"explicit_token_routing_logical_expert_invalid:{token}:{expert!r}")
            if expert in seen:
                raise ModelingError(
                    f"explicit_token_routing_duplicate_expert:{token}:{expert}")
            seen.add(expert)
            rank, bank = placement[expert]
            bank = expert % local if bank is None else bank
            dst = rank
            expert_rows[expert] += 1
            rank_rows[dst][bank] += 1
            peer[src][dst] += 1
    return (tuple(expert_rows), tuple(tuple(row) for row in rank_rows),
            tuple(tuple(row) for row in peer), ep, local)


def _normalize_placement(placement_raw, experts):
    """Resolve the accepted rank / (rank, bank) shorthand once.

    The original interface accepts ``[0, [0, 1], 1, ...]`` as well as all
    ``(rank, bank)`` pairs.  An omitted bank means the expert's position in
    its local contiguous bank.  Binding must produce one concrete table so
    validation, exported local ids and every consumer see the same physical
    slots; leaving ``None`` in the stored tuple makes later sorting or slot
    comparison depend on the caller.
    """
    if len(placement_raw) != experts:
        raise ModelingError("explicit_token_routing_expert_placement_length_mismatch")
    parsed = []
    for value in placement_raw:
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ModelingError("explicit_token_routing_expert_placement_pair_required")
            rank, bank = value
        else:
            rank, bank = value, None
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ModelingError("explicit_token_routing_physical_rank_invalid")
        if rank < 0:
            raise ModelingError("explicit_token_routing_physical_rank_invalid")
        if bank is not None:
            if isinstance(bank, bool) or not isinstance(bank, int):
                raise ModelingError("explicit_token_routing_physical_bank_invalid")
            if bank < 0:
                raise ModelingError("explicit_token_routing_physical_bank_invalid")
        parsed.append((rank, bank))
    ranks = sorted({rank for rank, _bank in parsed})
    ep = max(ranks) + 1 if ranks else 1
    if ranks != list(range(ep)):
        raise ModelingError("explicit_token_routing_physical_ranks_must_be_contiguous")
    if ep > experts or experts % ep:
        raise ModelingError(
            "explicit_token_routing_requires_divisible_physical_expert_banks:"
            f"{experts}%{ep}")
    local = experts // ep
    concrete = []
    for expert, (rank, bank) in enumerate(parsed):
        slot = expert % local if bank is None else bank
        if slot >= local:
            raise ModelingError(
                f"explicit_token_routing_bank_out_of_range:{expert}:{slot}")
        concrete.append((rank, slot))
    return tuple(concrete)


def _explicit_routing(item, layer, phase, tokens, views, replica,
                      physical_rank=None, query_position=None):
    """Build one exact token ledger, or None when this item is not explicit."""
    if not any(key in item for key in ("requests", "request_views",
                                       "expert_placement", "source_owners",
                                       "source_owner", "query_position")):
        return None
    requests = item.get("requests", item.get("request_views"))
    request_ids, positions = (), ()
    owner_missing = False
    if requests is not None:
        if isinstance(requests, (str, bytes)) or not isinstance(requests, (list, tuple)):
            raise ModelingError("explicit_token_routing_requests_must_be_sequence")
        actual = _norm_views(views)
        request_ids, positions, topk_rows, owners = [], [], [], []
        for entry in requests:
            if not isinstance(entry, Mapping):
                raise ModelingError("explicit_token_routing_request_must_be_mapping")
            request_id = str(entry.get("request_id", ""))
            if not request_id:
                raise ModelingError("explicit_token_routing_request_id_required")
            q_len = entry.get("q_len")
            if type(q_len) is not int or q_len < 0:
                raise ModelingError("explicit_token_routing_request_q_len_invalid")
            position = entry.get("query_position", entry.get("query_start"))
            if position is not None and type(position) is not int:
                raise ModelingError("explicit_token_routing_query_position_invalid")
            context = entry.get("context")
            ranges = entry.get("query_ranges", ())
            selected_view = actual.get(request_id)
            if selected_view is not None:
                if q_len != int(_view_field(selected_view, "q_len", q_len)):
                    raise ModelingError(
                        f"explicit_token_routing_request_q_len_mismatch:{request_id}")
                if position is not None and position != int(
                        _view_field(selected_view, "context", position)):
                    raise ModelingError(
                        f"explicit_token_routing_query_position_mismatch:{request_id}")
                if context is not None and int(_view_field(selected_view, "context", context)) != int(context):
                    raise ModelingError(
                        f"explicit_token_routing_request_context_mismatch:{request_id}")
                actual_ranges = tuple((int(a), int(b)) for a, b in
                                      (_view_field(selected_view, "query_ranges", ()) or ()))
                declared_ranges = tuple((int(a), int(b)) for a, b in (ranges or ()))
                if declared_ranges and actual_ranges and declared_ranges != actual_ranges:
                    raise ModelingError(
                        f"explicit_token_routing_request_ranges_mismatch:{request_id}")
            ids = entry.get("topk_ids")
            single_row = (q_len == 1 and ids is not None
                          and all(type(value) is int for value in ids))
            if ids is None or (len(ids) != q_len and not single_row):
                raise ModelingError(
                    f"explicit_token_routing_request_topk_rows_required:{request_id}")
            owner = entry.get("source_owner", entry.get("source_rank"))
            if owner is None:
                owner = item.get("source_owner", item.get("source_rank"))
            if owner is None:
                owner_missing = True
                owner = int(replica)
            if q_len == 1 and ids and all(type(value) is int for value in ids):
                topk_rows.append(tuple(ids))
            elif len(ids) == q_len and all(
                    isinstance(row, (list, tuple)) for row in ids):
                topk_rows.extend(tuple(row) for row in ids)
            else:
                raise ModelingError(
                    f"explicit_token_routing_request_topk_shape_invalid:{request_id}")
            # A request may declare one owner for all rows or per-row owners.
            declared_owners = entry.get("source_owners")
            if declared_owners is not None:
                if len(declared_owners) != q_len:
                    raise ModelingError(
                        f"explicit_token_routing_request_source_owner_count_mismatch:{request_id}")
                owners.extend(int(value) for value in declared_owners)
            else:
                owners.extend([int(owner)] * q_len)
            request_ids.extend([request_id] * q_len)
            positions.extend([int(position) if position is not None else -1] * q_len)
        topk = tuple(topk_rows)
        source_owners = tuple(owners)
    else:
        ids = item.get("topk_ids")
        if ids is None:
            raise ModelingError("explicit_token_routing_topk_ids_required")
        if ids and all(type(value) is int for value in ids):
            topk = (tuple(ids),)
        else:
            topk = tuple(tuple(row) for row in ids)
        owners = item.get("source_owners")
        if owners is None:
            owner = item.get("source_owner", item.get("source_rank"))
            if owner is None:
                owner_missing = True
                owner = int(replica)
            owners = [int(owner)] * len(topk)
        elif type(owners) is int:
            owners = [int(owners)] * len(topk)
        source_owners = tuple(int(value) for value in owners)
    expected_tokens = (int(item.get("tokens", len(topk)))
                       if item.get("cohort") else int(tokens))
    if len(topk) != expected_tokens:
        raise ModelingError(
            f"explicit_token_routing_token_count_mismatch:{len(topk)}!={expected_tokens}")
    experts = int(layer.num_experts)
    top_k = int(layer.top_k)
    # Placement is authoritative; bind its shorthand to concrete rank/bank
    # slots once, then every ledger/export/consumer shares that table.
    placement = _normalize_placement(
        item.get("expert_placement", ()), experts)
    if owner_missing and any(rank for rank, _bank in placement):
        raise ModelingError(
            "explicit_token_routing_source_owner_required_for_ep")
    expert_rows, rank_rows, peer, ep, local = _explicit_ledger(
        topk, placement, source_owners, experts, top_k)
    representative = (physical_rank if physical_rank is not None else
                      item.get("representative_rank",
                               item.get("replica", replica)))
    representative = int(representative)
    if not 0 <= representative < ep:
        raise ModelingError("explicit_token_routing_representative_rank_invalid")
    # A cohort ledger spans every physical source rank; the selected graph may
    # own only its own source rows. Keep the global token count for the ledger
    # and let valid_local_tokens identify this rank's share.
    ledger_tokens = len(topk) if item.get("cohort") else int(tokens)
    return ExplicitTokenRouting(
        layer.scope, int(layer.index), phase, ledger_tokens, topk,
        placement, source_owners, tuple(expert_rows),
        tuple(tuple(row) for row in rank_rows),
        tuple(tuple(row) for row in peer), representative, int(replica),
        tuple(request_ids if requests is not None else ()),
        tuple(positions if requests is not None else ()))
