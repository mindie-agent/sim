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
from mindie_sim.schema.common import ModelingError


POLICY_BALANCED = "balanced_expectation"
POLICY_EXPLICIT = "explicit"
#: Variance damping factor when redundant EPLB slots mitigate EP rank skew.
EPLB_VARIANCE_DAMPING = 0.35
#: Expected maximum of two standard normals (sqrt(2 / pi)).
ORDER_STAT_M2_Z = math.sqrt(2.0 / math.pi)
#: Precomputed ln(4 * pi) term in the asymptotic normal maximum order statistic.
ORDER_STAT_LN_FOUR_PI = math.log(4.0 * math.pi)


def _normal_max_order_z(rank_count: int) -> float:
    """Closed-form expected maximum of ``rank_count`` standard normals."""
    if rank_count <= 1:
        return 0.0
    if rank_count == 2:
        return ORDER_STAT_M2_Z
    log_m = math.log(float(rank_count))
    root = math.sqrt(2.0 * log_m)
    asymptotic = root - (math.log(log_m) + ORDER_STAT_LN_FOUR_PI) / (2.0 * root)
    return max(ORDER_STAT_M2_Z, asymptotic)


def _straggler_slot_weight_bytes(
    *,
    probabilities: Sequence[float],
    weights: Sequence[float],
    mean_bytes: float,
    ep: int,
    total_rows: float,
    total_active: float,
    has_redundant: bool,
) -> float:
    """Busiest-rank active-slot cold weight read with >=1 bank floor when active."""
    max_bytes = float(math.fsum(float(w) for w in weights))
    if max_bytes <= 0.0:
        return 0.0
    mu_act = math.fsum(probabilities)
    damping = EPLB_VARIANCE_DAMPING if has_redundant else 1.0
    active_ranks = min(ep, max(1, math.ceil(total_rows - 1e-12)))
    z_order = _normal_max_order_z(active_ranks)
    var_act = (
        math.fsum(p * (1.0 - p) for p in probabilities)
        * (1.0 - 1.0 / float(max(2, ep)))
        * damping
    )
    sigma_act = math.sqrt(max(0.0, var_act))
    act_lower = max(1.0, float(math.ceil(total_active / float(ep) - 1e-12)))
    a_max = min(
        float(len(probabilities)),
        max(1.0, float(math.ceil(total_rows - 1e-12))),
        max(act_lower, mu_act + z_order * sigma_act),
    )
    unit_bytes = (
        mean_bytes / mu_act if mu_act > 0.0 else max_bytes / float(len(probabilities))
    )
    return min(max_bytes, a_max * unit_bytes)


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

    def weight_read_bytes(self, rank: int) -> float:
        """Routed weight bytes read by one rank under this realization."""
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
        mean_bytes = float(math.fsum(
            float(weights[index]) * probability
            for index, probability in enumerate(probabilities)))
        ep = len(self.per_rank_slot_rows)
        total_rows = math.fsum(self.expected_rows_by_rank)
        if ep <= 1 or self.global_tokens <= 0 or total_rows <= 0.0 or not probabilities:
            return mean_bytes
        return _straggler_slot_weight_bytes(
            probabilities=probabilities,
            weights=weights,
            mean_bytes=mean_bytes,
            ep=ep,
            total_rows=total_rows,
            total_active=math.fsum(
                math.fsum(rank_probs) for rank_probs in self.active_probabilities
            ),
            has_redundant=(
                sum(len(rank_rows) for rank_rows in self.per_rank_slot_rows)
                > len(self.packed_weight_bytes)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_mode": self.routing_mode,
            "global_tokens": int(self.global_tokens),
            "expected_rows_by_rank": [float(value) for value in self.expected_rows_by_rank],
            "active_slots_by_rank": [int(value) for value in self.active_slots_by_rank],
            "weight_read_bytes_by_rank": [
                self.weight_read_bytes(rank) for rank in range(len(self.per_rank_slot_rows))
            ],
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
    weight_reads = [[0] * len(ids) for ids in slots.expert_ids_by_rank]
    for expert, count in copies.items():
        inclusion = inclusions[expert] if expert < len(inclusions) else 0
        per_copy = inclusion / float(max(1, count))
        if per_copy <= 0:
            active = 0
        elif per_copy >= 1:
            active = 1
        else:
            active = -math.expm1(float(tokens) * math.log1p(-per_copy))
        expected = float(tokens) * per_copy
        for rank, position in copy_index.get(expert, ()):
            rows[rank][position] += expected
            probabilities[rank][position] = max(probabilities[rank][position], active)
            weight_reads[rank][position] = float(weights[expert])
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
        tuple((expert, copies[expert]) for expert in sorted(copies)),
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
        active_bytes = slot_work.weight_read_bytes(rank)
        previous = float(metadata.get("routed_weight_read_bytes", 0) or 0)
        metadata.update({
            "physical_slot_rows": slot_work.per_rank_slot_rows[rank],
            "physical_slot_active": slot_work.active_slots_by_rank[rank],
            "physical_slot_weight_bytes": active_bytes,
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
