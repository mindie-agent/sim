"""Masked source force-EPLB assignments for an explicit SP/MC2 caller.

[Role] Retain allocated top-k cursor geometry and valid MC2 payload separately.
[Invariants] Source rank offsets precede masking; physical experts are contiguous
  EP banks. Every valid token owns top_k assignments. Forward/return traffic
  shares this integer ledger. Generic balanced routing is unchanged. One graph
  describes declared EP rank zero's compute and the complete peer transport;
  unequal per-rank compute is reported, never asserted to be a whole-rank bound.
[Timeouts & Locks] O(EP * experts) closed arithmetic, cache bounded to 128 entries.
[Side-Effect Policy] Pure immutable records; no source, capture or device reads.
"""
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
            "allocated_global_tokens": self.allocated_tokens_per_rank * self.ep,
            "valid_local_tokens": self.valid_local_tokens,
            "source_local_rows": self.allocated_tokens_per_rank,
            "valid_tokens_per_rank": self.valid_tokens_per_rank,
            "allocated_tokens_per_rank": self.allocated_tokens_per_rank,
            "source_partitions": self.source_partitions,
            "communication_assumption": "source_cursor_and_active_mask_exact_integer_matrix",
            "cursor": "(allocated_local_rows*top_k+1)*source_rank + local_assignment",
            "mask": "contiguous_global_valid_prefix; global_bs=0",
        }


@lru_cache(maxsize=128)
def ledger(experts, ep, top_k, allocated_local_tokens, valid_global_tokens):
    """Count source cursor residues without expanding token/expert assignments."""
    if (ep < 2 or experts % ep or not 0 < top_k <= experts
            or allocated_local_tokens < 1
            or not 0 < valid_global_tokens <= allocated_local_tokens * ep):
        raise ModelingError("source_force_eplb_requires_divisible_banks_and_bounded_valid_prefix")
    banks = experts // ep
    valid = tuple(max(0, min(allocated_local_tokens,
                            valid_global_tokens - src * allocated_local_tokens)) for src in range(ep))
    expert_counts = [0] * experts
    traffic = [[0] * ep for _ in range(ep)]
    for src, rows in enumerate(valid):
        rounds, tail = divmod(rows * top_k, experts)
        start = ((allocated_local_tokens * top_k + 1) * src) % experts
        for residue in range(experts):
            count = rounds + ((residue - start) % experts < tail)
            dest, bank = residue % ep, residue // ep
            expert_counts[dest * banks + bank] += count
            traffic[src][dest] += count
    return SourceForceEplb(tuple(expert_counts),
                          tuple(tuple(expert_counts[r*banks:(r+1)*banks]) for r in range(ep)),
                          tuple(map(tuple, traffic)), valid, allocated_local_tokens,
                          top_k, ep, valid_global_tokens)


def select(runtime, layer, *, tokens, valid_tokens, global_tokens,
           token_shards, valid_source_tokens, replica):
    raw = runtime.kernels.get("force_eplb_source")
    if raw is None:
        return None
    p = runtime.parallel
    wrapper = runtime.kernels.get("moe_token_partition")
    if raw == {"kind": "vllm_ascend_mc2_tp", "global_bs": 0}:
        if (runtime.routing != {"policy": "balanced"} or p.dp != 1 or p.ep != p.tp
                or p.ep < 2 or p.cp != 1 or p.pcp != 1 or p.pp != 1 or replica != 0
                or runtime.moe_recipe != "alltoall" or token_shards != 1
                or not wrapper or wrapper.get("kind") != "vllm_ascend_mc2_tp_slice"
                or runtime.kernels.get("sequence_parallel")):
            raise ModelingError("source_force_eplb_requires_explicit_mc2_tp_caller")
        if global_tokens not in (None, tokens) or not 0 < valid_tokens <= tokens:
            raise ModelingError("source_force_eplb_requires_declared_caller_valid_prefix")
        return ledger(layer.num_experts, p.ep, layer.top_k,
                      (tokens + p.tp - 1) // p.tp, valid_tokens)
    if raw != {"kind": "vllm_ascend_sp_mc2", "global_bs": 0}:
        raise ModelingError("force_eplb_source_requires_explicit_vllm_ascend_sp_mc2_global_bs0")
    if (runtime.routing != {"policy": "balanced"} or p.dp != 1 or p.ep != p.tp
            or p.ep < 2 or p.cp != 1 or p.pcp != 1 or p.pp != 1 or replica != 0
            or runtime.moe_recipe != "alltoall" or token_shards != p.tp
            or wrapper
            or runtime.kernels.get("native_moe_protocol", {}).get("kind") != "arch35_fullmesh"):
        raise ModelingError("source_force_eplb_requires_sp_dp1_ep_tp_mc2_alltoall")
    if (valid_source_tokens is None or global_tokens not in (None, tokens * p.ep)
            or valid_tokens != min(tokens, valid_source_tokens)):
        raise ModelingError("source_force_eplb_requires_explicit_sp_valid_prefix")
    return ledger(layer.num_experts, p.ep, layer.top_k, tokens, valid_source_tokens)
