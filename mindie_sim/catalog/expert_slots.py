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

    @property
    def ranks(self) -> int:
        return len(self.base_slots_by_rank)

    def copies_per_expert(self) -> dict[int, int]:
        copies: dict[int, int] = {}
        for ids in self.expert_ids_by_rank:
            for expert in ids:
                copies[int(expert)] = copies.get(int(expert), ZERO) + ONE
        return copies


def _base_layout(parallel: Any, experts: int) -> tuple[int, ...]:
    ep = max(ONE, int(parallel.ep))
    if experts <= ZERO:
        raise ModelingError("expert_slot_plan_requires_routed_experts")
    if experts % ep == ZERO:
        base = experts // ep
        return tuple(base for _ in range(ep))
    if ep % experts == ZERO:
        return tuple(ONE for _ in range(ep))
    raise ModelingError(f"experts_not_divisible_by_ep:{experts}%{ep}")


def _uniform_ids(rank: int, base: int, experts: int, ep: int) -> list[int]:
    if experts % ep == ZERO:
        start = rank * base
        return list(range(start, start + base))
    return [rank % experts]


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
    base = _base_layout(parallel, experts)
    extra_total = int(getattr(parallel, "redundant_experts_per_rank", ZERO) or ZERO)
    if extra_total < ZERO:
        raise ModelingError("redundant_experts_per_rank_invalid")
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
        for rank, ids in enumerate(ids_by_rank):
            if any(expert < ZERO or expert >= experts for expert in ids):
                raise ModelingError("explicit_expert_slot_map_expert_unknown")
            expected_base = set(_uniform_ids(rank, base[rank], experts, ep))
            if not expected_base <= set(ids):
                raise ModelingError("explicit_expert_slot_map_base_slot_missing")
        return ExpertSlotPlan(experts, base, redundant, ids_by_rank, policy)
    for rank, base_count in enumerate(base):
        if extra_total > experts - base_count:
            raise ModelingError(
                f"redundant_experts_exceed_distinct_local_experts:"
                f"rank{rank}:R={extra_total},base={base_count},E={experts}")
    ids_by_rank = []
    for rank, base_count in enumerate(base):
        ids = _uniform_ids(rank, base_count, experts, ep)
        candidates = [expert for expert in range(experts) if expert not in ids]
        ids.extend(candidates[:extra_total])
        ids_by_rank.append(tuple(sorted(ids)))
    redundant = tuple(extra_total for _ in range(ep))
    return ExpertSlotPlan(experts, base, redundant, tuple(ids_by_rank), policy)


def routed_expert_weight_terms(
    terms: Sequence[Any], slots: ExpertSlotPlan, *, ep_rank: int,
) -> tuple[Any, ...]:
    """Attach resolved base counts and add exact runtime-copy expert terms."""
    if ep_rank < ZERO or ep_rank >= slots.ranks:
        raise ModelingError(f"expert_slot_ep_rank_invalid:{ep_rank}")
    base_count = int(slots.base_slots_by_rank[ep_rank])
    extra = int(slots.redundant_slots_by_rank[ep_rank])
    result: list[Any] = []
    for term in terms:
        if str(getattr(term, "name", "")) in ROUTED_TERMS:
            result.append(replace(term, resident_count=base_count))
            if extra > ZERO:
                result.append(replace(
                    term, name=f"{term.name}_redundant",
                    count=extra, resident_count=extra, source_owned=False,
                    placement_role="", placement_scope="target"))
        else:
            result.append(term)
    return tuple(result)
