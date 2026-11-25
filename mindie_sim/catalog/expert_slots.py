"""Physical redundant-expert slots and runtime-copy weight terms.

[Role]
  Model ``redundant_experts_per_rank=R`` as extra resident physical slots on
  each EP owner while preserving the logical expert population.  Base slots and
  extra runtime copies get exact per-rank shapes, quantization and ETP rules.

[Invariants]
  * Logical experts remain ``E``; only physical residency grows.
  * ``R <= E - base_owned_experts[rank]`` under the distinct-extra policy, and
    an EP1 rank that already owns every expert cannot add a distinct nonlocal
    expert.
  * Extra terms are ``source_owned=False`` runtime copies; counts are not
    divided by EP again.
  * The pre-existing explicit EPLB cohort-total vector is not reinterpreted as
    per-rank R and its surcharge is not added a second time.

[Timeouts & Locks]
  Pure rank/slot arithmetic; no I/O or locks.

[Side-Effect Policy]
  Returns immutable records.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1
POLICY_BALANCED = "balanced_expectation"
POLICY_EXPLICIT = "explicit"
POLICIES = frozenset({POLICY_BALANCED, POLICY_EXPLICIT})
SCOPE_ALL = "all"
SCOPE_TARGET = "target"
SCOPES = frozenset({SCOPE_ALL, SCOPE_TARGET})
ROUTED_TERMS = frozenset({"expert_gate", "expert_up", "expert_down",
                          "draft_expert_gate", "draft_expert_up",
                          "draft_expert_down"})


@dataclass(frozen=True)
class ExpertSlotPlan:
    logical_experts: int
    base_slots_by_rank: tuple[int, ...]
    redundant_slots_by_rank: tuple[int, ...]
    expert_ids_by_rank: tuple[tuple[int, ...], ...]
    policy: str
    base_expert_ids_by_rank: tuple[tuple[int, ...], ...] = ()
    redundant_expert_ids_by_rank: tuple[tuple[int, ...], ...] = ()

    @property
    def ranks(self) -> int:
        return len(self.base_slots_by_rank)

    def copies_per_expert(self) -> dict[int, int]:
        copies: dict[int, int] = {}
        for ids in self.expert_ids_by_rank:
            for expert in ids:
                copies[int(expert)] = copies.get(int(expert), ZERO) + ONE
        return copies

    def split_expert_ids(self, ep_rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return exact ``(base_ids, redundant_ids)`` for ``ep_rank``."""
        if (self.base_expert_ids_by_rank
                and len(self.base_expert_ids_by_rank) == self.ranks
                and len(self.redundant_expert_ids_by_rank) == self.ranks):
            return (
                self.base_expert_ids_by_rank[ep_rank],
                self.redundant_expert_ids_by_rank[ep_rank],
            )
        base_count = int(self.base_slots_by_rank[ep_rank])
        ids = self.expert_ids_by_rank[ep_rank]
        return ids[:base_count], ids[base_count:]


def _base_layout(parallel: Any, experts: int, *, padded: bool = False) -> tuple[int, ...]:
    ep = max(ONE, int(parallel.ep))
    if experts <= ZERO:
        raise ModelingError("expert_slot_plan_requires_routed_experts")
    if padded:
        # EPLB convention when redundancy is active: uniform ceil(E/EP) base
        # slots per rank so global placement forms a rectangular [EP, slots] map.
        base = (experts + ep - ONE) // ep
        return tuple(base for _ in range(ep))
    # Non-EPLB linear placement (vllm-ascend determine_expert_map): exact
    # quotient/remainder E // EP + (rank < E % EP) with zero padding duplicates.
    quotient, remainder = divmod(int(experts), ep)
    return tuple(quotient + (ONE if rank < remainder else ZERO) for rank in range(ep))


def _base_ids(rank: int, base: int, experts: int, ep: int, *, padded: bool = False) -> list[int]:
    """Logical ids for one rank's contiguous base slot range."""
    if padded:
        start = int(rank) * int(base)
        return [int((start + offset) % experts) for offset in range(int(base))]
    quotient, remainder = divmod(int(experts), max(ONE, int(ep)))
    start = int(rank) * quotient + min(int(rank), remainder)
    return [int(start + offset) for offset in range(int(base))]


def expert_slot_plan(
    model: Any, layer: Any, parallel: Any, *, placement: Any = None,
    routing: Mapping[str, Any] | None = None,
) -> ExpertSlotPlan:
    """Resolve base and extra physical slots for one routed MoE layer."""
    del model, placement
    experts = int(getattr(layer, "num_experts", ZERO) or ZERO)
    ep = max(ONE, int(parallel.ep))
    routing = dict(routing or {})
    policy = str(routing.get("expert_slot_policy", POLICY_BALANCED))
    if policy not in POLICIES:
        raise ModelingError(f"expert_slot_policy_unsupported:{policy}")
    extra_total = int(getattr(parallel, "redundant_experts_per_rank", ZERO) or ZERO)
    if extra_total < ZERO:
        raise ModelingError("redundant_experts_per_rank_invalid")
    padded_base = extra_total > ZERO
    base = _base_layout(parallel, experts, padded=padded_base)
    if policy == POLICY_EXPLICIT:
        explicit = routing.get("expert_slot_map")
        if not isinstance(explicit, Mapping):
            raise ModelingError("explicit_expert_slot_policy_requires_map")
        raw = explicit.get(int(layer.index), explicit.get(str(layer.index)))
        if raw is None:
            raise ModelingError(f"explicit_expert_slot_map_missing_layer:{layer.index}")
        if not isinstance(raw, (list, tuple)) or len(raw) != ep:
            raise ModelingError("explicit_expert_slot_map_rank_count_mismatch")
        ids_by_rank = tuple(tuple(sorted(set(int(item) for item in group)))
                            for group in raw)
        redundant = tuple(len(ids) - base[rank] for rank, ids in enumerate(ids_by_rank))
        if any(value != extra_total for value in redundant):
            raise ModelingError("explicit_expert_slot_map_redundant_count_mismatch")
        base_ids_by_rank: list[tuple[int, ...]] = []
        redundant_ids_by_rank: list[tuple[int, ...]] = []
        for rank, ids in enumerate(ids_by_rank):
            if any(expert < ZERO or expert >= experts for expert in ids):
                raise ModelingError("explicit_expert_slot_map_expert_unknown")
            expected_base = set(_base_ids(rank, base[rank], experts, ep, padded=padded_base))
            if not expected_base <= set(ids):
                raise ModelingError("explicit_expert_slot_map_base_slot_missing")
            base_ids_by_rank.append(tuple(sorted(expected_base)))
            redundant_ids_by_rank.append(tuple(eid for eid in ids if eid not in expected_base))
        return ExpertSlotPlan(
            experts, base, redundant, ids_by_rank, policy,
            tuple(base_ids_by_rank), tuple(redundant_ids_by_rank),
        )
    for rank, base_count in enumerate(base):
        if extra_total > experts - base_count:
            raise ModelingError(
                f"redundant_experts_exceed_distinct_local_experts:"
                f"rank{rank}:R={extra_total},base={base_count},E={experts}")
    ids_by_rank_list: list[tuple[int, ...]] = []
    base_ids_list: list[tuple[int, ...]] = []
    redundant_ids_list: list[tuple[int, ...]] = []
    for rank, base_count in enumerate(base):
        base_ids = _base_ids(rank, base_count, experts, ep, padded=padded_base)
        base_set = set(base_ids)
        candidates = [expert for expert in range(experts) if expert not in base_set]
        extra_ids = tuple(candidates[:extra_total])
        base_ids_list.append(tuple(sorted(base_ids)))
        redundant_ids_list.append(extra_ids)
        ids_by_rank_list.append(tuple(sorted((*base_ids, *extra_ids))))
    redundant = tuple(extra_total for _ in range(ep))
    return ExpertSlotPlan(
        experts, base, redundant, tuple(ids_by_rank_list), policy,
        tuple(base_ids_list), tuple(redundant_ids_list),
    )


def routed_expert_weight_terms(
    terms: Sequence[Any], slots: ExpertSlotPlan, *, ep_rank: int,
) -> tuple[Any, ...]:
    """Attach resolved base counts, exact expert IDs and runtime-copy terms."""
    if ep_rank < ZERO or ep_rank >= slots.ranks:
        raise ModelingError(f"expert_slot_ep_rank_invalid:{ep_rank}")
    base_count = int(slots.base_slots_by_rank[ep_rank])
    extra = int(slots.redundant_slots_by_rank[ep_rank])
    base_ids, extra_ids = slots.split_expert_ids(ep_rank)
    result: list[Any] = []
    for term in terms:
        if str(getattr(term, "name", "")) in ROUTED_TERMS:
            result.append(replace(term, resident_count=base_count, expert_ids=base_ids))
            if extra > ZERO:
                result.append(replace(
                    term, name=f"{term.name}_redundant",
                    count=extra, resident_count=extra, expert_ids=extra_ids,
                    source_owned=False, placement_role="", placement_scope="target"))
        else:
            result.append(term)
    return tuple(result)
