"""Distinct active physical-expert slot work.

[Role]
  Convert logical routing inclusions or an exact integer realization into
  per-rank/per-slot expected rows, active probabilities, copy mapping and
  routed weight reads.  One result feeds dispatch peer traffic, grouped GEMMs
  and cost pricing so they cannot recompute different active-slot estimates.

[Invariants]
  * Logical token assignment, dispatch/combine totals and aggregate FLOPs are
    conserved; physical copies change residency and weight reads only.
  * At small batch active slots cannot exceed local assignments/resident slots;
    at saturation reads approach ``(L+R)*W_expert_local``.
  * Exact realization assigns each logical expert's tokens deterministically
    round-robin among its sorted physical copies without duplicating a token.
  * R=0 returns the existing logical result unchanged.

[Timeouts & Locks]
  Bounded by experts x tokens; no I/O or locks.

[Side-Effect Policy]
  Pure immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.catalog.expert_slots import ExpertSlotPlan
from mindie_sim.schema.common import (
    EPLB_VARIANCE_DAMPING,
    ModelingError,
    inclusion_probability,
    normal_max_order_z,
)


POLICY_BALANCED = "balanced_expectation"
POLICY_EXPLICIT = "explicit"
_NORMAL_ORDER_EPS = 1e-12


def _straggler_slot_weight_bytes(
    *,
    probabilities: Sequence[float],
    weights: Sequence[float],
    mean_bytes: float,
    ep: int,
    total_rows: float,
    total_active: float,
    has_redundant: bool,
    global_tokens: int | None = None,
    total_mean_bytes: float | None = None,
) -> float:
    """Busiest-rank reachable-slot cold weight read (zero-probability banks excluded).

    Obeys two exact probability theorems:
      1. Only when a single token carries at most one assignment globally
         (``global_tokens == 1`` and ``total_rows <= 1.0``) are per-rank reads
         strictly mutually exclusive (``E[max_r W_r] = sum_r E[W_r]``).
      2. For any ``total_rows > 0``, ``E[max_r W_r] <= sum_r E[W_r]`` (Boole's bound).
    """
    if total_rows <= 0.0 or total_active <= 0.0:
        return 0.0
    reachable = [
        (float(p), float(w))
        for p, w in zip(probabilities, weights)
        if float(p) > 0.0 and float(w) > 0.0
    ]
    if not reachable:
        return 0.0
    max_bytes = float(math.fsum(w for _, w in reachable))
    mu_act = float(math.fsum(p for p, _ in reachable))
    reachable_mean_bytes = float(math.fsum(p * w for p, w in reachable))
    boole_cap = (
        min(max_bytes, float(total_mean_bytes))
        if total_mean_bytes is not None else max_bytes
    )
    if global_tokens == 1 and total_rows <= 1.0 + _NORMAL_ORDER_EPS:
        if total_mean_bytes is not None:
            return min(max_bytes, float(total_mean_bytes))
        return min(max_bytes, reachable_mean_bytes * float(ep))
    damping = EPLB_VARIANCE_DAMPING if has_redundant else 1.0
    active_ranks = min(ep, max(1, math.ceil(total_rows - _NORMAL_ORDER_EPS)))
    z_order = normal_max_order_z(active_ranks)
    var_act = (
        math.fsum(p * (1.0 - p) for p, _ in reachable)
        * (1.0 - 1.0 / float(max(2, ep)))
        * damping
    )
    sigma_act = math.sqrt(max(0.0, var_act))
    avg_rank_active = float(total_active) / float(max(1, ep))
    if total_active < 1.0:
        rank_fire = min(1.0, max(0.0, mu_act))
        union_prob = 1.0 - (1.0 - rank_fire) ** max(1, ep)
        act_lower = min(float(len(reachable)), max(mu_act, union_prob))
    else:
        act_lower = max(1.0, float(math.ceil(avg_rank_active - _NORMAL_ORDER_EPS)))
    row_upper = (
        float(total_rows)
        if total_rows < 1.0
        else max(1.0, float(math.ceil(total_rows - _NORMAL_ORDER_EPS)))
    )
    a_max = min(
        float(len(reachable)),
        row_upper,
        max(act_lower, mu_act + z_order * sigma_act),
    )
    unit_bytes = reachable_mean_bytes / mu_act
    return min(boole_cap, a_max * unit_bytes)


@dataclass(frozen=True)
class PhysicalSlotWork:
    """Resolved active physical-slot work for one routed layer."""

    per_rank_slot_rows: tuple[tuple[float, ...], ...]
    active_probabilities: tuple[tuple[float, ...], ...]
    logical_to_copies: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    packed_weight_bytes: tuple[float, ...]
    routing_mode: str
    global_tokens: int
    expected_rows_by_rank: tuple[float, ...]
    active_slots_by_rank: tuple[int, ...]
    #: ``[rank][slot_pos]`` logical-expert weight bytes; empty keeps the
    #: historical uniform ``packed_weight_bytes`` fallback.
    slot_weight_bytes: tuple[tuple[float, ...], ...] = ()

    def rank_mean_weight_read_bytes(self, rank: int) -> float:
        """Additive per-rank expected (or exact realized) cold weight bytes (M2b)."""
        rows = self.per_rank_slot_rows[rank]
        probabilities = self.active_probabilities[rank]
        if self.routing_mode == POLICY_EXPLICIT:
            if self.slot_weight_bytes:
                weights = self.slot_weight_bytes[rank]
                return float(math.fsum(
                    float(weights[index]) * probability
                    for index, probability in enumerate(probabilities)))
            return float(math.fsum(
                float(self.packed_weight_bytes[index])
                for index, value in enumerate(rows) if value > 0))
        weights = (
            self.slot_weight_bytes[rank]
            if self.slot_weight_bytes
            else self.packed_weight_bytes[:len(probabilities)]
        )
        return float(math.fsum(
            float(weights[index]) * probability
            for index, probability in enumerate(probabilities)))

    def rank_barrier_weight_read_bytes(self, rank: int) -> float:
        """Order-statistic straggler barrier cold weight read for one EP rank.

        Matches the straggler-priced EP grouped-GEMM convention used when no
        redundant slot plan is active, while collapsing identically to
        ``rank_mean_weight_read_bytes(rank)`` under explicit realizations,
        single-rank EP, or full saturation.
        """
        mean_bytes = self.rank_mean_weight_read_bytes(rank)
        if self.routing_mode == POLICY_EXPLICIT:
            return mean_bytes
        probabilities = self.active_probabilities[rank]
        weights = (
            self.slot_weight_bytes[rank]
            if self.slot_weight_bytes
            else self.packed_weight_bytes[:len(probabilities)]
        )
        ep = len(self.per_rank_slot_rows)
        total_rows = math.fsum(self.expected_rows_by_rank)
        if ep <= 1 or self.global_tokens <= 0 or total_rows <= 0.0 or not probabilities:
            return mean_bytes
        total_mean = math.fsum(self.rank_mean_weight_read_bytes(r) for r in range(ep))
        return _straggler_slot_weight_bytes(
            probabilities=probabilities,
            weights=weights,
            mean_bytes=mean_bytes,
            ep=ep,
            global_tokens=int(self.global_tokens),
            total_rows=total_rows,
            total_active=math.fsum(
                math.fsum(rank_probs) for rank_probs in self.active_probabilities
            ),
            has_redundant=(
                sum(len(rank_rows) for rank_rows in self.per_rank_slot_rows)
                > len(self.packed_weight_bytes)
            ),
            total_mean_bytes=total_mean,
        )

    def critical_rank_weight_read_bytes(self) -> float:
        """Busiest-rank order-statistic cold weight bytes across EP ranks (M2b barrier)."""
        ep = len(self.per_rank_slot_rows)
        if ep <= 0:
            return 0.0
        rank_means = [self.rank_mean_weight_read_bytes(rank) for rank in range(ep)]
        if self.routing_mode == POLICY_EXPLICIT:
            return max(rank_means)
        total_rows = math.fsum(self.expected_rows_by_rank)
        if total_rows <= 0.0 or self.global_tokens <= 0:
            return 0.0
        total_mean = float(math.fsum(rank_means))
        if self.global_tokens == 1 and total_rows <= 1.0 + _NORMAL_ORDER_EPS:
            # Single token with at most one global assignment -> ranks are strictly
            # mutually exclusive, so E[max_r W_r] == sum_r E[W_r] identically.
            return total_mean
        return max(
            max(rank_means),
            min(
                total_mean,
                max(self.rank_barrier_weight_read_bytes(rank) for rank in range(ep)),
            ),
        )

    def weight_read_bytes(self, rank: int) -> float:
        """True physical per-rank expected (or exact realized) cold weight read bytes.

        Preserves symmetry and exact global conservation
        (``sum_r weight_read_bytes(r) == sum_r rank_mean_weight_read_bytes(r)``)
        without shifting synchronization-barrier skew onto rank 0; callers
        pricing the EP synchronization barrier query ``critical_rank_weight_read_bytes()``.
        """
        return self.rank_mean_weight_read_bytes(rank)

    def to_dict(self) -> dict[str, Any]:
        rank_count = len(self.per_rank_slot_rows)
        return {
            "routing_mode": self.routing_mode,
            "global_tokens": int(self.global_tokens),
            "expected_rows_by_rank": [float(value) for value in self.expected_rows_by_rank],
            "active_slots_by_rank": [int(value) for value in self.active_slots_by_rank],
            "weight_read_bytes_by_rank": [
                self.weight_read_bytes(rank) for rank in range(rank_count)
            ],
            "mean_weight_read_bytes_by_rank": [
                self.rank_mean_weight_read_bytes(rank) for rank in range(rank_count)
            ],
            "critical_rank_weight_read_bytes": self.critical_rank_weight_read_bytes(),
        }


def _inclusion_map(logical_inclusions: Any, experts: int) -> tuple[float, ...]:
    if isinstance(logical_inclusions, Mapping):
        values = [float(logical_inclusions.get(expert, 0))
                  for expert in range(experts)]
    elif isinstance(logical_inclusions, Sequence) and not isinstance(
            logical_inclusions, (str, bytes)):
        values = [float(value) for value in logical_inclusions]
        if len(values) != experts:
            raise ModelingError("logical_inclusion_count_mismatch")
    else:
        values = [float(logical_inclusions) for _ in range(experts)]
    for value in values:
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ModelingError(f"logical_inclusion_invalid:{value}")
    return tuple(values)


def _expert_bytes(value: Any, experts: int) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        return tuple(float(value.get(expert, 0)) for expert in range(experts))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(float(item) for item in value)
        if len(values) != experts:
            raise ModelingError("physical_slot_weight_byte_count_mismatch")
        return values
    return tuple(float(value) for _ in range(experts))


def _copy_index(slots: ExpertSlotPlan) -> dict[int, tuple[tuple[int, int], ...]]:
    result: dict[int, list[tuple[int, int]]] = {}
    for rank, ids in enumerate(slots.expert_ids_by_rank):
        for position, expert in enumerate(ids):
            result.setdefault(int(expert), []).append((rank, position))
    return {expert: tuple(sorted(copies)) for expert, copies in result.items()}


def _balanced(
    slots: ExpertSlotPlan, inclusions: Sequence[float], tokens: int,
    weights: Sequence[float],
) -> PhysicalSlotWork:
    copies = slots.copies_per_expert()
    copy_index = _copy_index(slots)
    slot_bytes = tuple(tuple(float(weights[expert]) for expert in ids)
                       for ids in slots.expert_ids_by_rank)
    rows = [[0] * len(ids) for ids in slots.expert_ids_by_rank]
    probabilities = [[0] * len(ids) for ids in slots.expert_ids_by_rank]
    for expert, count in copies.items():
        inclusion = inclusions[expert] if expert < len(inclusions) else 0.0
        per_copy = inclusion / float(max(1, count))
        active = inclusion_probability(float(tokens), per_copy)
        expected = float(tokens) * per_copy
        for rank, position in copy_index.get(expert, ()):
            rows[rank][position] += expected
            probabilities[rank][position] = max(probabilities[rank][position], active)
    ep = len(probabilities)
    total_rows = math.fsum(math.fsum(row) for row in rows)
    active_slots = tuple(
        max(1, int(round(math.fsum(row))))
        if (tokens > 0 and total_rows > 0.0 and ep > 1 and row and math.fsum(row) > 0.0)
        else int(round(math.fsum(row)))
        for row in probabilities
    )
    return PhysicalSlotWork(
        tuple(tuple(float(value) for value in row) for row in rows),
        tuple(tuple(float(value) for value in row) for row in probabilities),
        tuple((expert, copy_index.get(expert, ())) for expert in sorted(copies)),
        tuple(float(value) for value in weights),
        POLICY_BALANCED, int(tokens),
        tuple(float(sum(row)) for row in rows),
        active_slots,
        slot_bytes,
    )


def _realization_counts(
    realization: Mapping[Any, Any], experts: int,
) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
    by_expert: dict[int, int] = {}
    by_rank: dict[int, dict[int, int]] = {}
    for key, value in realization.items():
        if isinstance(key, int) and isinstance(value, Mapping):
            by_rank[int(key)] = {int(expert): int(count)
                                 for expert, count in value.items()}
        else:
            by_expert[int(key)] = int(value)
    for counts in by_rank.values():
        for expert, count in counts.items():
            by_expert[expert] = by_expert.get(expert, 0) + count
    for expert, count in by_expert.items():
        if expert < 0 or expert >= experts or count < 0:
            raise ModelingError(f"physical_slot_realization_invalid:{expert}:{count}")
    return by_expert, by_rank


def physical_slot_work(
    *, slot_plan: ExpertSlotPlan, logical_inclusions: Any,
    global_tokens: int, routing_realization: Mapping[Any, Any] | None = None,
    packed_weight_bytes: Any = 1,
) -> PhysicalSlotWork:
    """Resolve expected or exact active physical-slot rows."""
    if isinstance(global_tokens, bool) or not isinstance(global_tokens, int) or global_tokens < 0:
        raise ModelingError(f"physical_slot_global_tokens_invalid:{global_tokens}")
    experts = slot_plan.logical_experts
    inclusions = _inclusion_map(logical_inclusions, experts)
    weights = _expert_bytes(packed_weight_bytes, experts)
    if routing_realization is None:
        return _balanced(slot_plan, inclusions, global_tokens, weights)
    by_expert, by_rank = _realization_counts(routing_realization, experts)
    copy_index = _copy_index(slot_plan)
    rows = [[0] * len(ids) for ids in slot_plan.expert_ids_by_rank]
    if by_rank:
        for rank, counts in by_rank.items():
            if rank >= len(rows):
                raise ModelingError(f"physical_slot_realization_rank_invalid:{rank}")
            for expert, count in counts.items():
                copies = copy_index.get(expert, ())
                if not copies:
                    raise ModelingError(f"physical_slot_realization_unknown_expert:{expert}")
                position = next((pos for r, pos in copies if r == rank), None)
                if position is None:
                    raise ModelingError(f"physical_slot_realization_wrong_owner:{rank}:{expert}")
                rows[rank][position] += float(count)
    else:
        for expert, count in by_expert.items():
            copies = copy_index.get(expert, ())
            if not copies:
                raise ModelingError(f"physical_slot_realization_unknown_expert:{expert}")
            for offset in range(count):
                rank, position = copies[offset % len(copies)]
                rows[rank][position] += 1
    probabilities = tuple(
        tuple(1 if value > 0 else 0 for value in row) for row in rows)
    slot_bytes = tuple(tuple(float(weights[expert]) for expert in ids)
                       for ids in slot_plan.expert_ids_by_rank)
    return PhysicalSlotWork(
        tuple(tuple(float(value) for value in row) for row in rows),
        probabilities,
        tuple((expert, copy_index[expert]) for expert in sorted(copy_index)),
        tuple(float(value) for value in weights),
        POLICY_EXPLICIT, int(global_tokens),
        tuple(float(sum(row)) for row in rows),
        tuple(sum(1 for value in row if value > 0) for row in rows),
        slot_bytes,
    )


def bind_expert_slot_work(ops: Sequence[Any], *, slot_work: PhysicalSlotWork) -> tuple[Any, ...]:
    """Attach resolved physical-slot reads to routed grouped-GEMM records."""
    result = []
    for op in ops:
        metadata = dict(getattr(op, "metadata", ()) or ())
        rank = metadata.get("ep_rank", metadata.get("rank"))
        if rank is None or not any(key in metadata for key in (
                "routed_weight_read_bytes", "matrix_weight_bank_groups")):
            result.append(op)
            continue
        rank = int(rank)
        if rank < 0 or rank >= len(slot_work.per_rank_slot_rows):
            raise ModelingError(f"expert_slot_work_rank_invalid:{rank}")
        active_bytes = slot_work.rank_barrier_weight_read_bytes(rank)
        mean_bytes = slot_work.rank_mean_weight_read_bytes(rank)
        previous = float(metadata.get("routed_weight_read_bytes", 0) or 0)
        metadata.update({
            "physical_slot_rows": slot_work.per_rank_slot_rows[rank],
            "physical_slot_active": slot_work.active_slots_by_rank[rank],
            "physical_slot_weight_bytes": active_bytes,
            "physical_slot_mean_weight_bytes": mean_bytes,
            "routed_weight_read_bytes": active_bytes,
            "packed_weight_bytes": active_bytes,
            "expert_slot_active_weight_bytes": active_bytes,
            "expert_slot_routing_mode": slot_work.routing_mode,
        })
        result.append(replace(
            op,
            mte2_bytes=max(0.0, float(getattr(op, "mte2_bytes", 0))
                           - previous + active_bytes),
            hbm_bytes=max(0.0, float(getattr(op, "hbm_bytes", 0))
                          - previous + active_bytes),
            metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[0])),
        ))
    return tuple(result)
