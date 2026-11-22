"""Physical attention and index-key traffic planning.

[Role] Separate logical tensor reads, unique HBM fills, L2/private refills and
  selected-row gathers for attention and indexer operators.
[Invariants] One physical KV row is charged once per unique cache line that must
  leave HBM. Private tile refills remain real but are L2 hits when resident.
  Unknown dynamic TopK selections remain per-query gathers; shared capacity
  alone never collapses them. Output and query payloads stay explicit.
[Timeouts & Locks] Bounded sequence/tile arithmetic; no I/O or locks.
[Side-Effect Policy] Pure immutable traffic plans and operator replacements.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.physical_mechanisms import MemoryHierarchySpec


DEFAULT_QUERY_TILE = 128
BYTES_PER_ELEMENT = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1,
                     "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}


#: Declared attention traffic phases.  A speculative draft step reads and
#: writes the same cache geometry as the target phase it prepares, so it is the
#: same traffic contract under a distinct declared phase name.
ATTENTION_ACCESS_PHASES = frozenset({
    "prefill", "decode", "mixed", "verify",
    "draft", "draft_prefill",
})


@dataclass(frozen=True)
class AttentionAccessSpec:
    q_lengths: tuple[int, ...]
    context_lengths: tuple[int, ...]
    kv_heads: int = 1
    q_heads: int = 1
    key_dim: int = 1
    value_dim: int = 1
    key_bytes: float | None = None
    value_bytes: float | None = None
    row_bytes: float | None = None
    unique_rows: int = 0
    refill_rows: int = 0
    selected_rows: int = 0
    prefix_tokens: int = 0
    prefix_key: str = ""
    cache_identity: str = ""
    representation: str = "gqa"
    phase: str = "decode"
    topk: int = 0
    window: int = 0
    noncausal: bool = False
    selected_rows_known: bool = False

    def __post_init__(self):
        if len(self.q_lengths) != len(self.context_lengths):
            raise ModelingError("attention_access_requires_matching_sequence_geometry")
        if any(value < 0 for value in (*self.q_lengths, *self.context_lengths)):
            raise ModelingError("attention_access_lengths_must_be_nonnegative")
        if (self.kv_heads < 1 or self.q_heads < 1 or self.key_dim < 1
                or self.value_dim < 1):
            raise ModelingError("attention_access_positive_geometry_required")
        if self.row_bytes is not None:
            if self.row_bytes <= 0.0:
                raise ModelingError("attention_access_positive_geometry_required")
        elif (self.key_bytes is None or self.value_bytes is None
              or self.key_bytes <= 0.0 or self.value_bytes <= 0.0
              or self.representation not in {"gqa", "absorbed_mla"}):
            raise ModelingError("attention_access_row_or_element_bytes_required")
        if self.phase not in ATTENTION_ACCESS_PHASES:
            raise ModelingError("attention_access_phase_invalid")


@dataclass(frozen=True)
class AttentionSchedule:
    query_tile: int = DEFAULT_QUERY_TILE
    key_tile: int = DEFAULT_QUERY_TILE
    resident: bool = True
    page_bytes: float = 0.0
    index_bytes: float = 0.0

    def __post_init__(self):
        if self.query_tile < 1 or self.key_tile < 1:
            raise ModelingError("attention_schedule_tile_must_be_positive")


@dataclass(frozen=True)
class IndexAccessSpec:
    query_rows: int = 0
    prefix_tokens: int = 0
    private_tokens: tuple[int, ...] = ()
    row_bytes: float = 0.0
    layer: int = 0
    component: str = "index_k"
    cache_identity: str = ""
    prefix_key: str = ""
    dynamic_selection: bool = False
    selected_rows: int = 0
    index_bytes: float = 0.0
    scale_bytes: float = 0.0

    def __post_init__(self):
        if self.query_rows < 0 or self.prefix_tokens < 0:
            raise ModelingError("index_access_tokens_must_be_nonnegative")
        if any(value < 0 for value in self.private_tokens):
            raise ModelingError("index_access_private_tokens_must_be_nonnegative")
        if self.row_bytes <= 0.0:
            raise ModelingError("index_access_row_bytes_must_be_positive")


@dataclass(frozen=True)
class IndexSchedule:
    query_tile: int = DEFAULT_QUERY_TILE
    batch_tile: bool = True
    shared_key_scan: bool = True


@dataclass(frozen=True)
class TrafficPlan:
    logical_read_bytes: float
    hbm_read_bytes: float
    hbm_write_bytes: float
    l2_hit_bytes: float
    private_refill_bytes: float
    index_bytes: float
    scale_bytes: float
    working_set_bytes: float
    residency_basis: str

    def __post_init__(self):
        values = (self.logical_read_bytes, self.hbm_read_bytes,
                  self.hbm_write_bytes, self.l2_hit_bytes,
                  self.private_refill_bytes, self.index_bytes,
                  self.scale_bytes, self.working_set_bytes)
        if any(not math.isfinite(value) or value < 0.0
               for value in values):
            raise ModelingError("traffic_plan_must_be_finite_and_nonnegative")

    def to_dict(self) -> dict:
        return {
            "logical_read_bytes": float(self.logical_read_bytes),
            "hbm_read_bytes": float(self.hbm_read_bytes),
            "hbm_write_bytes": float(self.hbm_write_bytes),
            "l2_hit_bytes": float(self.l2_hit_bytes),
            "private_refill_bytes": float(self.private_refill_bytes),
            "index_bytes": float(self.index_bytes),
            "scale_bytes": float(self.scale_bytes),
            "working_set_bytes": float(self.working_set_bytes),
            "residency_basis": self.residency_basis,
        }


def _hierarchy(value) -> MemoryHierarchySpec:
    if not isinstance(value, MemoryHierarchySpec):
        raise ModelingError("traffic_planner_requires_memory_hierarchy")
    return value


def _default_row_bytes(access: AttentionAccessSpec) -> float:
    key_bytes = float(access.key_bytes or 0.0)
    value_bytes = float(access.value_bytes or 0.0)
    if access.representation == "absorbed_mla":
        return float(access.key_dim * key_bytes + access.value_dim * value_bytes)
    if access.representation == "gqa":
        return float(access.kv_heads * (access.key_dim * key_bytes
                                        + access.value_dim * value_bytes))
    return 0.0


def _causal_pairs(q_len: int, context: int, window: int, noncausal: bool,
                  topk: int) -> float:
    if noncausal:
        horizon = context + q_len
        if topk > 0:
            return float(q_len * min(horizon, topk))
        if window > 0:
            return float(q_len * min(horizon, window))
        return float(q_len * horizon)
    total = 0
    for index in range(q_len):
        horizon = context + index + 1
        if topk > 0:
            horizon = min(horizon, topk)
        elif window > 0:
            horizon = min(horizon, window)
        total += horizon
    return float(total)


def plan_attention_traffic(access: AttentionAccessSpec,
                           hierarchy: MemoryHierarchySpec,
                           schedule: AttentionSchedule) -> TrafficPlan:
    """Plan unique KV HBM fills and private refills for one attention view."""
    hierarchy = _hierarchy(hierarchy)
    row_bytes = (
        float(access.row_bytes) if access.row_bytes is not None
        else _default_row_bytes(access)
    )
    pairs = math.fsum(
        _causal_pairs(int(q), int(c), access.window, access.noncausal, access.topk)
        for q, c in zip(access.q_lengths, access.context_lengths))
    if access.unique_rows > 0:
        unique_rows = int(access.unique_rows)
    else:
        unique_rows = sum(int(c) + int(q)
                          for q, c in zip(access.q_lengths, access.context_lengths))
        if access.window > 0 and not access.noncausal:
            unique_rows = sum(min(int(c) + int(q), access.window)
                              for q, c in zip(access.q_lengths, access.context_lengths))
    logical_rows = float(access.refill_rows) if access.refill_rows > 0 else pairs
    if access.selected_rows > 0:
        logical_rows = float(access.selected_rows)
    logical = logical_rows * row_bytes
    working_set = float(unique_rows) * row_bytes
    private_refill = logical
    effective_l2 = hierarchy.effective_l2_bytes
    sparse_decode = (access.topk > 0 and access.phase in {"decode", "draft"}
                     and not access.selected_rows_known)
    if sparse_decode:
        hbm_kv = logical
        basis = "unknown_dynamic_topk_per_query_gather"
    elif working_set <= effective_l2 and schedule.resident:
        hbm_kv = working_set
        basis = "dense_unique_working_set_resident"
    else:
        retained = min(working_set, effective_l2)
        spill = 1.0
        if working_set > 0.0:
            spill = max(0.0, 1.0 - retained / working_set)
        hbm_kv = min(logical, working_set + max(0.0, logical - working_set) * spill)
        basis = "capacity_spill_window"
    hbm_kv = min(hbm_kv, logical)
    l2_hit = max(0.0, private_refill - hbm_kv)
    return TrafficPlan(
        logical_read_bytes=logical,
        hbm_read_bytes=hbm_kv,
        hbm_write_bytes=0.0,
        l2_hit_bytes=l2_hit,
        private_refill_bytes=private_refill,
        index_bytes=float(schedule.index_bytes),
        scale_bytes=0.0,
        working_set_bytes=working_set,
        residency_basis=basis,
    )


def plan_index_key_traffic(access: IndexAccessSpec,
                           hierarchy: MemoryHierarchySpec,
                           schedule: IndexSchedule) -> TrafficPlan:
    """Plan shared dense prefix keys once and private per-query keys once."""
    hierarchy = _hierarchy(hierarchy)
    private = float(sum(max(0, int(value)) for value in access.private_tokens))
    logical_rows = (float(access.prefix_tokens) * max(1, access.query_rows)
                    + private)
    logical = logical_rows * access.row_bytes
    if access.dynamic_selection:
        selected = float(access.selected_rows) if access.selected_rows > 0 else logical_rows
        hbm = selected * access.row_bytes
        basis = "unknown_dynamic_selection_per_query"
    elif schedule.batch_tile and schedule.shared_key_scan:
        unique_rows = float(access.prefix_tokens) + private
        hbm = unique_rows * access.row_bytes
        if hbm > hierarchy.effective_l2_bytes:
            spill = max(0.0, 1.0 - hierarchy.effective_l2_bytes / hbm)
            hbm += (logical - hbm) * spill
        basis = "shared_prefix_batch_tile_scan"
    else:
        hbm = logical
        basis = "per_query_index_scan"
    hbm = min(hbm, logical)
    return TrafficPlan(
        logical_read_bytes=logical,
        hbm_read_bytes=hbm,
        hbm_write_bytes=0.0,
        l2_hit_bytes=max(0.0, logical - hbm),
        private_refill_bytes=logical,
        index_bytes=float(access.index_bytes),
        scale_bytes=float(access.scale_bytes),
        working_set_bytes=hbm,
        residency_basis=basis,
    )


def apply_traffic_plan(op, traffic: TrafficPlan):
    """Replace one operator's external traffic with a typed physical plan."""
    metadata = dict(getattr(op, "metadata", ()) or ())
    metadata["traffic_plan"] = traffic.to_dict()
    metadata["traffic_plan_basis"] = traffic.residency_basis
    return replace(
        op,
        hbm_bytes=traffic.hbm_read_bytes + traffic.hbm_write_bytes,
        mte2_bytes=traffic.private_refill_bytes + traffic.index_bytes
        + traffic.scale_bytes,
        metadata=tuple(sorted(metadata.items())),
    )


def require_machine_memory_hierarchy(
    machine, *, hierarchy: MemoryHierarchySpec | None = None,
) -> MemoryHierarchySpec:
    """Return explicit machine/caller memory hierarchy or fail closed."""
    if hierarchy is not None:
        if not isinstance(hierarchy, MemoryHierarchySpec) or hierarchy.l2_capacity_bytes < 0:
            raise ModelingError("machine_memory_hierarchy_missing")
        return hierarchy
    if machine is not None and getattr(machine, "dies", ()):
        die_hierarchy = getattr(
            machine.dies[0].physical_mechanisms, "memory_hierarchy", None
        )
        if isinstance(die_hierarchy, MemoryHierarchySpec) and die_hierarchy.l2_capacity_bytes >= 0:
            return die_hierarchy
    raise ModelingError("machine_memory_hierarchy_missing")
