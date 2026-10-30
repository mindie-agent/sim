"""Integer forced-balance scenarios with per-expert row tails.

[Role] Separate a declared balancing scenario from sampled router assignments.
[Invariants] Count each declared router row once per EP cohort; capture/SP
  padding is physical routing work and stays separate from valid queries. Preserve zero experts,
  distinct expert weights and one grouped launch. The representative strided
  rank is the heaviest shard within this DP replica's local TP group. TP1
  cohorts retain exact per-rank counts; TP representatives remain visible bounds.
  The communication scenario partitions outgoing assignments across local EP
  ranks and assumes uniform peer payload, not an observed token permutation.
[Timeouts & Locks] O(experts + ranks) arithmetic, no locks or I/O.
[Side-Effect Policy] Immutable structural descriptions only.
"""
from collections import Counter
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError


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

    @property
    def rows(self):
        return sum(self.local_expert_rows)

    @property
    def active(self):
        return sum(value > 0 for value in self.local_expert_rows)

    @property
    def basis(self):
        return "forced_balanced_integer_counts"

    def to_dict(self):
        return {
            "basis": self.basis, "global_tokens": self.global_tokens,
            "global_expert_rows": self.expert_rows,
            "local_expert_rows": self.local_expert_rows,
            "representative_expert_ids": tuple(range(self.representative_rank, len(self.expert_rows), self.ep)),
            "expert_placement": "strided_ep", "matrix_groups": self.active,
            "rows_per_active_expert": (self.rows // self.active
                                       if self.active and len(set(self.local_expert_rows) - {0}) == 1
                                       else None),
            "row_classes": tuple(sorted(Counter(self.local_expert_rows).items())),
            "rank_assignment_rows": tuple(sum(self.expert_rows[rank::self.ep])
                                          for rank in range(self.ep)),
            "representative_rank": self.representative_rank,
            "represented_ep_ranks": self.represented_ranks,
            "rank_timing_assumption": ("exact_physical_rank" if len(self.represented_ranks) == 1
                                       else "heaviest_local_tp_shard_upper_bound"),
            "cohort_counts_basis": "exact_integer_expert_assignment_ledger",
            "valid_local_tokens": self.valid_local_tokens,
            "padded_local_tokens": self.padded_local_tokens,
            "source_partitions": self.source_partitions,
            "communication_assumption": "uniform_peer_payload",
            "token_identities": "not_declared",
        }


def balanced_realization(runtime, layer, *, tokens, valid_tokens, global_tokens,
                         token_shards=1, valid_source_tokens=None, replica=0):
    p = runtime.parallel
    if p.pcp != 1 or runtime.moe_recipe not in {"alltoall", "tp_join_reduce"}:
        raise ModelingError("balanced_routing_requires_pcp1_unchunked_alltoall")
    width = p.ep // p.tp if p.ep_spans_dp else 1
    if width > 1 and global_tokens is None:
        raise ModelingError("balanced_routing_requires_complete_dp_cohort:use_target_graphs")
    total = tokens * token_shards if global_tokens is None else global_tokens
    if token_shards != 1 and (
            token_shards != p.tp or total < tokens * token_shards):
        raise ModelingError("balanced_routing_sequence_shards_require_complete_cohort")
    assignments = total * layer.top_k
    q, r = divmod(assignments, layer.num_experts)
    counts = tuple(q + (index < r) for index in range(layer.num_experts))
    first = (replica % width) * p.tp if p.ep_spans_dp else 0
    ranks = tuple(range(first, first + min(p.tp, p.ep)))
    representative = max(ranks, key=lambda rank: sum(counts[rank::p.ep]))
    return BalancedRouting(counts, counts[representative::p.ep], p.ep,
                           min(p.tp, p.ep) // token_shards, total,
                           (valid_tokens * token_shards if valid_source_tokens is None
                            else valid_source_tokens), tokens * token_shards,
                           representative, ranks)


def cohort_padded_tokens(groups, runtime):
    """Count every router row, including declared per-replica capture padding."""
    from dataclasses import replace
    from .graph import GraphPolicy
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
