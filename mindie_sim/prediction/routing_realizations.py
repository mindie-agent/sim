"""Explicit integer expert assignments for a declared forward scenario.

[Role] Keep repeated routing realizations separate from probability moments.
[Invariants] Counts conserve top-k assignments and preserve empty experts.
  Optional explicit ``topk_ids`` must agree exactly with the histogram, are
  never synthesized from counts, and retain token identities for source
  alignment. Matching selects computation inputs, never timings. A declared
  layer has no implicit statistical fallback when its phase/token scenario is
  unavailable.
[Timeouts & Locks] O(declared scenarios + experts + declared ids), no histories
  or locks.
[Side-Effect Policy] Pure input interpretation; no captures or device activity.
"""
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
                       physical_rank=None, query_position=None):
    raw = runtime.routing
    from .source_force_eplb import select as source_realization
    selected = source_realization(
        runtime, layer, tokens=tokens, valid_tokens=valid_tokens,
        global_tokens=global_tokens, token_shards=token_shards,
        valid_source_tokens=valid_source_tokens, replica=replica)
    if selected is not None:
        return selected
    if raw.get("policy") == "balanced" and set(raw) == {"policy"}:
        from .balanced_routing import balanced_realization
        return balanced_realization(runtime, layer, tokens=tokens,
                                    valid_tokens=valid_tokens, global_tokens=global_tokens,
                                    token_shards=token_shards,
                                    valid_source_tokens=valid_source_tokens, replica=replica)
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
        for item in explicit:
            cohort = bool(item.get("cohort", False))
            if (not cohort and item.get("replica") is not None
                    and int(item["replica"]) != int(replica)):
                continue
            if item.get("phase") != phase:
                continue
            if not cohort and int(item.get("tokens", -1)) != tokens:
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
                selected_explicit[0], layer, phase, tokens, views, replica,
                physical_rank=physical_rank, query_position=query_position)
        if (int(runtime.parallel.ep) > 1 and tokens > 0
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
            "expert_rows": self.expert_rows,
            "expert_ids": tuple(range(len(self.expert_rows))),
            "expert_placement": self.expert_placement,
            "source_owners": self.source_owners,
            "topk_ids": self.topk_ids,
            "rank_expert_rows": self.rank_expert_rows,
            "peer_assignments": self.peer_assignments,
            "rank_assignment_rows": self.rank_assignment_rows,
            "rows": self.rows, "active": self.active,
            "valid_local_tokens": self.valid_local_tokens,
            "representative_rank": self.representative_rank,
            "replica": self.replica,
            "request_ids": self.request_ids,
            "query_positions": self.query_positions,
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


@lru_cache(maxsize=128)
def _explicit_ledger(topk, placement, source_owners, experts, top_k):
    """Build the shared cohort matrix once; per-replica projection is cheap."""
    ranks = sorted({rank for rank, _bank in placement})
    ep = max(ranks) + 1 if ranks else 1
    local = max(1, experts // max(1, ep))
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
        src = source_owners[token] if token < len(source_owners) else 0
        if not 0 <= src < ep:
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
    # Placement is authoritative; ep is its distinct physical rank count.
    placement_raw = item.get("expert_placement", ())
    if len(placement_raw) != experts:
        raise ModelingError("explicit_token_routing_expert_placement_length_mismatch")
    placement = []
    for value in placement_raw:
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ModelingError("explicit_token_routing_expert_placement_pair_required")
            rank, bank = int(value[0]), int(value[1])
        else:
            rank = int(value)
            bank = None
        if rank < 0:
            raise ModelingError("explicit_token_routing_physical_rank_invalid")
        placement.append((rank, bank))
    if owner_missing and placement and max(rank for rank, _bank in placement) > 0:
        raise ModelingError(
            "explicit_token_routing_source_owner_required_for_ep")
    expert_rows, rank_rows, peer, ep, local = _explicit_ledger(
        topk, tuple(placement), source_owners, experts, top_k)
    representative = (physical_rank if physical_rank is not None else
                      item.get("representative_rank",
                               item.get("replica", replica)))
    representative = int(representative)
    if not 0 <= representative < ep:
        raise ModelingError("explicit_token_routing_representative_rank_invalid")
    return ExplicitTokenRouting(
        layer.scope, int(layer.index), phase, int(tokens), topk,
        tuple(placement), source_owners, tuple(expert_rows),
        tuple(tuple(row) for row in rank_rows),
        tuple(tuple(row) for row in peer), representative, int(replica),
        tuple(request_ids if requests is not None else ()),
        tuple(positions if requests is not None else ()))
