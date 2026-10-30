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
                       token_shards=1, valid_source_tokens=None, replica=0):
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
        if set(item) - {"topk_ids"} != _BASE_KEYS:
            raise ModelingError("routing_realization_requires_scope_layer_phase_tokens_counts")
        if item["scope"] == layer.scope and item["layer"] == layer.index:
            matching.append(item)
    if not matching:
        return None
    if (runtime.parallel.ep != 1 or runtime.parallel.pcp != 1
            or runtime.moe_recipe not in {"alltoall", "tp_join_reduce"}):
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
