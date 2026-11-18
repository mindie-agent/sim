"""Placement-aware resident memory and dynamic request ownership.

[Role] Derive weight/cache geometry and account for live allocations.
[Invariants] Stored observations cannot overwrite geometry. Each replica owns
  its cache; PP stages and physical ranks retain separate capacities. A reserved
  pool and the bytes used inside it are not added twice. Cache allocation
  (payload/page tail/scale/page metadata) is reported separately from accesses.
  Every grant/selected/shared/reader holder owns an explicit physical backing
  (rank/stage/storage/tier/generation/range); one batch commits only when all
  physical-union candidates fit, and failed/admitted decisions never leak state.
[Timeouts & Locks] Bounded by layers and active requests, no I/O or locks.
[Side-Effect Policy] Mutates only this prediction's cache ledger.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.cache import (
    cache_component_split, cache_row_split, layer_cache_components,
    local_kv_heads, owned_token_rows,
)
from mindie_sim.catalog.cache_pages import Allocation, AllocationPlan, CachePages
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes, term_payload_bytes
from mindie_sim.catalog.weights import (
    layer_weight_terms, stage_weight_bytes, stage_weight_terms,
)
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.case import Case as CaseSpec
from .runtime.contracts import (
    AllocationBinding, AllocationKey, BackingKey, CapacityDomainState,
    EventKey, OccurrenceKey, PoolKey,
)
from .runtime.ledger import Ledger

ZERO = 0
ONE = 1
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
FIRST = 0


@dataclass(frozen=True)
class _LedgerBoundary:
    """A real completed boundary token for legacy name-based grant/reader calls."""

    sequence: int
    time: float = 0.0

    @property
    def done(self) -> bool:
        return True


@dataclass(frozen=True)
class _HoldSnapshot:
    """One consumer's registered backing range, version and binding facts."""

    stages: tuple[float, ...]
    host_dram: float
    version: int
    staging: tuple[float, ...] = ()
    binding: tuple[tuple[str, object], ...] = ()
    #: The same backing split by physical rank.  Empty keeps the legacy
    #: stage-level call shape; a placement-aware caller fills one row per rank.
    rank_stages: tuple[tuple[float, ...], ...] = ()
    rank_staging: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class _PhysicalRef:
    """One holder's reference to an explicit physical allocation.

    Backing identity is ``(replica, rank, stage, storage, tier, key,
    generation)``.  Two holders with the same identity share one physical
    allocation and their half-open ranges are unioned.  A different live
    generation, or a different key on the same rank/stage, is a second
    allocation and is charged separately.
    """

    rank: int
    stage: int
    storage: str
    tier: str
    physical_key: str
    generation: object
    ranges: tuple[tuple[float, float], ...]
    host_ranges: tuple[tuple[float, float], ...] = ()


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}_invalid")
    return number


def _hashable_generation(value):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _half_open_ranges(value, size: float = 0.0) -> tuple[tuple[float, float], ...]:
    """Normalise explicit half-open byte ranges.

    Accepts ``None`` (meaning ``[0, size)``), a single number, a flat even
    sequence such as ``(0, 4096)``, a flat odd sequence of legacy extents, or a
    sequence of ``(start, end)`` pairs.  Zero-width intervals are dropped.
    """
    size = max(0.0, _finite_number(size, "range_size"))
    if value is None:
        return ((0.0, size),) if size > 0.0 else ()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError("range_sequence_invalid") from exc
    if not items:
        return ((0.0, size),) if size > 0.0 else ()
    if all(isinstance(item, (int, float)) and not isinstance(item, bool)
           for item in items):
        numbers = [_finite_number(item, "range_bound") for item in items]
        if any(number < 0.0 for number in numbers):
            raise ValueError("range_bound_negative")
        if len(numbers) == 1:
            return ((0.0, numbers[0]),) if numbers[0] > 0.0 else ()
        if len(numbers) % 2:
            return tuple((0.0, number) for number in numbers if number > 0.0)
        pairs = [(numbers[index], numbers[index + 1])
                 for index in range(0, len(numbers), 2)]
    else:
        pairs = []
        for item in items:
            try:
                lo, hi = item
            except (TypeError, ValueError) as exc:
                raise ValueError("range_pair_invalid") from exc
            pairs.append((_finite_number(lo, "range_bound"),
                          _finite_number(hi, "range_bound")))
    result = []
    for lo, hi in pairs:
        if lo < 0.0:
            raise ValueError("range_start_negative")
        if hi < lo:
            raise ValueError("range_end_before_start")
        if hi > lo:
            result.append((lo, hi))
    return tuple(result)


def _merge_ranges(ranges) -> tuple[tuple[float, float], ...]:
    ordered = sorted((float(lo), float(hi)) for lo, hi in ranges if hi > lo)
    merged: list[tuple[float, float]] = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _union_bytes(ranges) -> float:
    return float(sum(hi - lo for lo, hi in _merge_ranges(ranges)))


def _normalise_owner_spans(raw, width: int) -> tuple[tuple[int, int], ...] | None:
    """Normalise the T05 token-owner partition for ``width`` physical ranks.

    ``raw`` may be per-rank row counts (implicitly cumulative token spans) or
    explicit half-open ``(start, end)`` token intervals.  ``None`` keeps the
    legacy stage-level behaviour; an explicit partition is never replaced by
    a stage average.
    """
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = raw.get("token_spans") or raw.get("spans") or raw.get("rows")
    if raw is None:
        return None
    items = tuple(raw)
    if len(items) != int(width):
        raise ValueError("memory_token_owner_partition_width_mismatch")
    if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        spans = []
        start = 0
        for count in items:
            if count < 0:
                raise ValueError("memory_token_owner_partition_negative")
            spans.append((start, start + int(count)))
            start += int(count)
        return tuple(spans)
    spans = []
    for item in items:
        try:
            lo, hi = item
        except (TypeError, ValueError) as exc:
            raise ValueError("memory_token_owner_partition_invalid") from exc
        lo, hi = int(lo), int(hi)
        if lo < 0 or hi < lo:
            raise ValueError("memory_token_owner_partition_invalid")
        spans.append((lo, hi))
    return tuple(spans)


@dataclass(frozen=True)
class MemoryPolicy:
    workspace_bytes: float = 0
    graph_pool_bytes: float = 0
    communication_buffer_bytes: float = 0
    runtime_bytes: float = 0
    derive_workspace: bool = True
    cache_page_groups: tuple[dict, ...] = ()
    async_delayed_release_steps: int | None = None
    async_in_flight_lookahead_steps: int | None = None
    async_workspace_buffer_multiplier: float | None = None
    expert_ddr_ratio: float = 0.0
    kvc_ddr_ratio: float = 0.0
    kvc_ssd_ratio: float = 0.0
    #: Number of implementation workspaces that are simultaneously live
    #: (for example independent streams).  The base workspace is multiplied by
    #: this count; async/expert staging reserves remain their own components.
    concurrent_workspace_copies: int = 1
    #: Explicit transient representation bytes (for example a selected
    #: dequantisation tile).  Zero means "not selected"; it is never inferred
    #: from a dtype or a row size.
    temporary_dequant_bytes: float = 0.0
    #: T05 placement fact: token rows owned by each physical rank, either as
    #: per-rank counts or explicit half-open token spans.  ``None`` keeps the
    #: legacy stage-level split; an explicit uneven partition is never
    #: replaced by a stage average or a group minimum.
    rank_token_owners: tuple | None = None

    @property
    def reserved(self) -> float:
        return sum((self.workspace_bytes, self.graph_pool_bytes,
                    self.communication_buffer_bytes, self.runtime_bytes))


def _offload_ratio(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"memory_offload_invalid_ratio:{name}:{value}")
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"memory_offload_invalid_ratio:{name}:{value}")
    return ratio


def _resolve_offload_ratios(case: CaseSpec, policy: MemoryPolicy) -> dict[str, float]:
    """Prefer explicit policy ratios, then the case's offload metadata."""
    raw = dict(case.runtime.metadata).get("offload", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("memory_offload_metadata_mapping_required")
    metadata = dict(raw)

    def pick(name: str) -> float:
        declared = getattr(policy, name)
        if declared != 0.0:
            return _offload_ratio(name, declared)
        if name in metadata:
            return _offload_ratio(name, metadata[name])
        return 0.0

    ratios = {name: pick(name) for name in
              ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio")}
    if ratios["kvc_ddr_ratio"] + ratios["kvc_ssd_ratio"] > 1.0 + 1e-12:
        raise ValueError("memory_offload_kvc_ratios_exceed_unity")
    return ratios


def _routed_expert_bytes_by_stage(
    case: CaseSpec, layer_stage: tuple[int, ...], *, workspace_tokens: int = ONE,
    slot_plans=None,
) -> tuple[list[float], list[float], list[float]]:
    """Routed-expert total, payload and active staging bytes per PP stage."""
    parallel = case.runtime.parallel
    quant = case.model.quant or QuantSpec("bf16")
    result = [0.0] * max(1, parallel.pp)
    payload = [0.0] * max(1, parallel.pp)
    staging = [0.0] * max(1, parallel.pp)
    ep = max(1, int(parallel.ep))
    tokens = max(1, int(workspace_tokens))
    for plan in case.model.layer_plans:
        if plan.ffn != "moe" or int(plan.num_experts) <= 0 or int(plan.moe_intermediate_size) <= 0:
            continue
        stage = layer_stage[plan.index] if plan.index < len(layer_stage) else 0
        terms = [term for term in layer_weight_terms(plan, case.model)
                 if term.category == "routed_expert"]
        slots = (slot_plans or {}).get(plan.index)
        base_count = extra_count = ZERO
        if slots is not None:
            from mindie_sim.catalog.expert_slots import routed_expert_weight_terms
            terms = list(routed_expert_weight_terms(terms, slots, ep_rank=ZERO))
            base_count = int(slots.base_slots_by_rank[FIRST])
            extra_count = int(slots.redundant_slots_by_rank[FIRST])
        routed = float(sum(
            term_bytes(term, quant, parallel, case.machine.ranks_per_node)
            for term in terms
        ))
        routed_payload = float(sum(
            term_payload_bytes(term, quant, parallel, case.machine.ranks_per_node)
            for term in terms
        ))
        result[stage] += routed
        payload[stage] += routed_payload
        num_exp = int(plan.num_experts)
        k = min(num_exp, max(1, int(plan.top_k)))
        if slots is not None:
            # ``routed`` is already this rank's physical slot bank, so the active
            # staging of this rank is
            #   sum_{slot in rank} W[logical(slot)] * p[logical(slot)]
            # with NO further EP division: ``active_total`` is already local.
            copies = slots.copies_per_expert()
            local_slots = max(ONE, base_count + extra_count)
            per_slot = routed / float(local_slots)
            row_share = float(k) / float(num_exp)
            best_active = ZERO_FLOAT
            for ids in slots.expert_ids_by_rank:
                active = ZERO_FLOAT
                for expert in ids:
                    share = row_share / float(copies.get(int(expert), ZERO) or ONE)
                    if ZERO_FLOAT < share < ONE_FLOAT:
                        active += -math.expm1(float(tokens) * math.log1p(-share))
                    else:
                        active += float(share >= ONE_FLOAT)
                best_active = max(best_active, active)
            active_layer_bytes = min(routed, per_slot * best_active)
        else:
            active_total = float(num_exp) if k >= num_exp else float(num_exp) * (
                ONE_FLOAT - ((ONE_FLOAT - float(k) / float(num_exp)) ** float(tokens))
            )
            local_exp = max(ONE, (num_exp + ep - ONE) // ep)
            active_layer_bytes = (active_total / float(ep)) * (
                routed / float(local_exp))
        staging[stage] = max(staging[stage], min(routed, active_layer_bytes))
    return result, payload, staging


def _resident_weight_parallel(term, parallel, enable_shared_expert_dp: bool):
    """Mirror the weight ledger's shared-expert placement without duplicating it."""
    if enable_shared_expert_dp and term.category == "shared_expert":
        return replace(parallel, tp=1)
    return parallel


def _weight_ledger_by_stage(
    case: CaseSpec, layer_stage: tuple[int, ...], *, draft_kind: str,
    include_draft: bool, include_engram_tables: bool, enable_shared_expert_dp: bool,
    slot_plans=None, ep_rank: int | None = None,
) -> tuple[list[float], list[float]]:
    """Persistent weight payload and quant metadata per PP stage.

    The total is checked against ``stage_weight_bytes`` for the same inputs;
    this is a decomposition, never an alternate resident-weight quantity.
    """
    model, parallel = case.model, case.runtime.parallel
    quant = model.quant or QuantSpec("bf16")
    total_by_stage: list[float] = []
    payload_by_stage: list[float] = []
    for terms in stage_weight_terms(
        model, parallel, include_draft=include_draft, draft_kind=draft_kind or None,
        include_engram_tables=include_engram_tables, layer_partition=layer_stage,
        slot_plans=slot_plans, ep_rank=ep_rank,
    ):
        total = payload = 0.0
        for term in terms:
            placed = _resident_weight_parallel(term, parallel, enable_shared_expert_dp)
            total += term_bytes(term, quant, placed, case.machine.ranks_per_node)
            payload += term_payload_bytes(
                term, quant, placed, case.machine.ranks_per_node)
        total_by_stage.append(float(total))
        payload_by_stage.append(float(payload))
    return total_by_stage, payload_by_stage


class Memory:
    def __init__(self, case: CaseSpec, policy: MemoryPolicy | None = None, *,
                 workspace_tokens: int | None = None):
        self.case = case
        self.policy = policy or MemoryPolicy(**case.runtime.memory)
        p = case.runtime.parallel
        partition = case.runtime.pp_layer_partition
        self.layers = tuple(partition or (
            (case.model.layers // p.pp + (stage < case.model.layers % p.pp))
            for stage in range(p.pp)
        ))
        self.layer_stage = tuple(stage for stage, count in enumerate(self.layers) for _ in range(count))
        from mindie_sim.catalog.speculative_state import resolve_speculative_query_tokens
        _draft_meta = dict(case.runtime.metadata)
        _declared = int(_draft_meta.get("draft_proposal_tokens", ZERO) or ZERO)
        self.query_tokens = resolve_speculative_query_tokens(case.runtime)
        if dict(case.model.metadata).get("cache_allocator") == "paired_sfa" and _declared > ZERO:
            self.query_tokens = _declared + ONE
        from mindie_sim.catalog.paired_sfa import page_groups
        paired_groups = page_groups(case, self.layer_stage, self.query_tokens)
        self.paired_sfa = bool(paired_groups)
        if paired_groups and self.policy.cache_page_groups:
            raise ValueError("paired_sfa_and_explicit_cache_page_groups_conflict")
        self.pages = CachePages(
            paired_groups or self.policy.cache_page_groups,
            {layer.index: set(layer_cache_components(
                case.model, layer, p, case.runtime.max_model_len,
                capacity_padded=False, runtime=case.runtime,
                query_tokens=self.query_tokens)) for layer in case.model.layer_plans},
            self.layer_stage,
        )
        self.draft_kind = (dict(case.runtime.metadata).get("draft_mode") or
                           ("dspark" if case.runtime.dspark else "mtp" if
                            case.runtime.mtp_acceptance_length > 1 or any(
                                value > 1 for _, value in case.runtime.mtp_acceptance_by_output_tokens)
                            or dict(case.runtime.metadata).get("speculative")
                            else ""))
        draft = bool(self.draft_kind)
        self.offload = _resolve_offload_ratios(case, self.policy)
        self.kvc_hbm_ratio = max(
            0.0, 1.0 - self.offload["kvc_ddr_ratio"] - self.offload["kvc_ssd_ratio"]
        )
        shared_expert_dp = bool(dict(case.runtime.metadata).get(
            "enable_shared_expert_dp", False))
        self.slot_plans = None
        if (int(getattr(p, "redundant_experts_per_rank", ZERO) or ZERO) > ZERO
                or any(plan.ffn == "moe" and int(plan.num_experts) > ZERO
                       and int(plan.num_experts) % max(ONE, int(p.ep)) != ZERO
                       for plan in case.model.layer_plans)):
            from mindie_sim.catalog.expert_slots import expert_slot_plan
            from mindie_sim.schema.runtime_features import split_routing_features
            _logical, slot_routing = split_routing_features(
                dict(case.runtime.routing or {}))
            self.slot_plans = {
                plan.index: expert_slot_plan(
                    case.model, plan, p, routing=slot_routing)
                for plan in case.model.layer_plans
                if plan.ffn == "moe" and int(plan.num_experts) > ZERO
            }
        self.gross_weights = stage_weight_bytes(
            case.model, p, case.machine.ranks_per_node, include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            layer_partition=self.layers, draft_kind=self.draft_kind or None,
            enable_shared_expert_dp=shared_expert_dp,
            slot_plans=self.slot_plans,
        )
        weight_totals, weight_payloads = _weight_ledger_by_stage(
            case, self.layers, draft_kind=self.draft_kind,
            include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            enable_shared_expert_dp=shared_expert_dp,
            slot_plans=self.slot_plans,
        )
        if any(abs(total - gross) > max(1.0, abs(gross) * 1e-9)
               for total, gross in zip(weight_totals, self.gross_weights)):
            raise ValueError("memory_weight_decomposition_inconsistent")
        resolved_ws_tokens = self._resolve_workspace_tokens(workspace_tokens)
        if self.offload["expert_ddr_ratio"] > ZERO_FLOAT:
            (self.routed_expert_bytes, self.routed_expert_payload_bytes,
             stage_staging_base) = _routed_expert_bytes_by_stage(
                case, self.layer_stage, workspace_tokens=max(ONE, resolved_ws_tokens),
                slot_plans=self.slot_plans,
            )
            self.expert_staging_workspace = [
                self.offload["expert_ddr_ratio"] * value for value in stage_staging_base
            ]
        else:
            self.routed_expert_bytes = [0.0] * p.pp
            self.routed_expert_payload_bytes = [0.0] * p.pp
            self.expert_staging_workspace = [0.0] * p.pp
        self.expert_hbm_saved = [
            self.offload["expert_ddr_ratio"] * value for value in self.routed_expert_bytes
        ]
        self.expert_payload_hbm_saved = [
            self.offload["expert_ddr_ratio"] * value
            for value in self.routed_expert_payload_bytes
        ]
        self.weights = [
            max(0.0, gross - saved)
            for gross, saved in zip(self.gross_weights, self.expert_hbm_saved)
        ]
        self.weights_payload = [
            max(0.0, payload - saved)
            for payload, saved in zip(weight_payloads, self.expert_payload_hbm_saved)
        ]
        self.weights_quant_metadata = [
            max(0.0, total - payload)
            for total, payload in zip(self.weights, self.weights_payload)
        ]
        async_enabled = bool(case.runtime.async_scheduling)
        self.async_delayed_release_steps = (
            (1 if async_enabled else 0) if self.policy.async_delayed_release_steps is None
            else max(0, int(self.policy.async_delayed_release_steps))
        )
        self.async_in_flight_lookahead_steps = (
            (1 if async_enabled else 0) if self.policy.async_in_flight_lookahead_steps is None
            else max(0, int(self.policy.async_in_flight_lookahead_steps))
        )
        self.async_workspace_buffer_multiplier = (
            (1.0 if async_enabled else 0.0) if self.policy.async_workspace_buffer_multiplier is None
            else max(0.0, float(self.policy.async_workspace_buffer_multiplier))
        )
        copies = self.policy.concurrent_workspace_copies
        if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
            raise ValueError("memory_concurrent_workspace_copies_requires_positive_integer")
        self.concurrent_workspace_copies = copies
        temporary = self.policy.temporary_dequant_bytes
        if (isinstance(temporary, bool) or not isinstance(temporary, (int, float))
                or not math.isfinite(float(temporary)) or float(temporary) < 0.0):
            raise ValueError("memory_temporary_dequant_bytes_requires_nonnegative")
        self.temporary_dequant_bytes = float(temporary)
        width = p.tp * p.pcp
        self.async_workspace = (
            self._async_inflight_workspace(workspace_tokens)
            if self.policy.derive_workspace and self.async_workspace_buffer_multiplier > 0
            else [0.0] * p.pp
        )
        base_workspace = (self._workspace(workspace_tokens) if self.policy.derive_workspace
                          else [0.0] * p.pp)
        self.workspace_base = [value * copies for value in base_workspace]
        staging_workspace = (self.expert_staging_workspace if self.policy.derive_workspace
                             else [0.0] * p.pp)
        self.workspace_expert_staging = list(staging_workspace)
        self.sparse_offload = dict(case.runtime.metadata).get("sparse_kv_offload")
        static_offload = (
            list(self.sparse_offload.get("static_npu_bytes_per_stage_rank", (0.0,) * p.pp))
            if self.sparse_offload else [0.0] * p.pp
        )
        self.workspace_static_offload = list(static_offload)
        self.workspace = [
            base + extra + stage_buf + offload
            for base, extra, stage_buf, offload in zip(
                self.workspace_base, self.async_workspace, staging_workspace, static_offload
            )
        ]
        self.temporary_dequant = [self.temporary_dequant_bytes] * p.pp
        self.width = max(1, int(width))
        owner_placement = self.policy.rank_token_owners
        if owner_placement is None:
            metadata = dict(case.runtime.metadata)
            owner_placement = (metadata.get("rank_token_owners")
                               or metadata.get("token_owner_partition"))
        self.rank_token_owners = _normalise_owner_spans(
            owner_placement, self.width)
        fixed_per_rank = [
            self.weights[stage] + self.policy.reserved + self.workspace[stage]
            + self.temporary_dequant[stage]
            for stage in range(p.pp)
        ]
        self.fixed_per_rank = fixed_per_rank
        self.rank_capacity = [
            [
                [die.hbm_bytes * case.runtime.gpu_memory_utilization
                 - fixed_per_rank[stage] for die in self._stage_dies(replica, stage)]
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        self.capacity = [
            [min(row) for row in replica_ranks]
            for replica_ranks in self.rank_capacity
        ]
        self.peak = [[0.0] * p.pp for _ in range(p.dp)]
        self.live = [[0.0] * p.pp for _ in range(p.dp)]
        self.live_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.staging_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.selected_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.shared_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.granted_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.reader_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.owner_ranks: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self.logical_owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.logical_owner_ranks: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self.logical_host_owners: dict[tuple[int, str], float] = {}
        self.logical_staging_owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.logical_staging_rank: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self.staging_owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.staging_owner_ranks: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self.staging_live = [0.0] * p.dp
        self.async_consumer_holds: dict[
            tuple[int, str], dict[str, _HoldSnapshot]] = {}
        self.retired_requests: set[tuple[int, str]] = set()
        self.request_versions: dict[tuple[int, str], int] = {}
        self.grants: dict[tuple[int, str], tuple[float, ...]] = {}
        self.granted = [[0.0] * p.pp for _ in range(p.dp)]
        self.grant_bindings: dict[tuple[int, str], dict] = {}
        self.effect_bindings: dict[tuple[int, str], dict] = {}
        self.onchip_effects: dict[tuple[int, str], dict] = {}
        self.granted_host = [0.0] * p.dp
        self.grant_hosts: dict[tuple[int, str], float] = {}
        self.grant_records: dict[tuple[int, str], dict] = {}
        self.grant_backings: dict[tuple, dict] = {}
        self.grant_readers: dict[
            tuple[int, str], dict[str, dict]] = {}
        self.budget_version = 0
        self.release_events: list[dict] = []
        self.selected_plans: dict[tuple[int, int], AllocationPlan] = {}
        self.selected_items: dict[tuple[int, int], dict[str, Allocation]] = {}
        self.selected_reserved = [[0.0] * p.pp for _ in range(p.dp)]
        self.selected_released: set[tuple[int, int, str]] = set()
        self.selected_physical: dict[tuple[int, int, str], float] = {}
        self.peak_combined = [[0.0] * p.pp for _ in range(p.dp)]
        self.shared_pages: dict[tuple[int, str], dict] = {}
        self.shared_reserved = [[0.0] * p.pp for _ in range(p.dp)]
        self.delayed_owners: dict[tuple[int, str], tuple[int, tuple[float, ...]]] = {}
        self.peak_delayed_release = [[0.0] * p.pp for _ in range(p.dp)]
        self.peak_kv_resident = [[0.0] * p.pp for _ in range(p.dp)]
        self.peak_kv_staging = [[0.0] * p.pp for _ in range(p.dp)]
        self.host_dram_capacity = [
            float(self.sparse_offload["usable_dram_bytes_per_dp"]) if self.sparse_offload else math.inf
            for _ in range(p.dp)
        ]
        self.host_dram_live = [0.0 for _ in range(p.dp)]
        self.host_dram_peak = [0.0 for _ in range(p.dp)]
        self.host_dram_owners: dict[tuple[int, str], float] = {}
        #: Unified physical holder ledger.  Key: (replica, kind, name).
        self._refs: dict[
            tuple[int, str, str], tuple[_PhysicalRef, ...]] = {}
        #: Canonical RuntimeState capacity domain once adopted.  The detached
        #: domain is constructed here and adopted without copying its payload.
        self.runtime_state = None
        self.ledger_domain: CapacityDomainState | None = None
        self._typed_grants: dict[
            tuple[int, str], tuple[AllocationKey, ...]] = {}
        self._ledger_serial = 0
        self._ledger_generations: dict[object, int] = {}

    # ------------------------------------------------------------------
    # Canonical RuntimeState capacity domain
    # ------------------------------------------------------------------
    def ledger_pool(self, replica: int, stage: int, rank: int, *,
                    tier: str = "hbm", domain: str | None = None) -> PoolKey:
        """Physical pool identity for one request allocation rank/stage."""
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        if not 0 <= stage < p.pp:
            raise ValueError("memory_stage_out_of_range")
        if not 0 <= rank < self.width:
            raise ValueError("memory_rank_out_of_range")
        domain_id = domain or (self.ledger_domain.domain_id
                               if self.ledger_domain is not None else "memory")
        return PoolKey(physical_domain=domain_id, rank=int(rank), tier=tier,
                       name=f"stage{int(stage)}", unit=int(replica))

    def _ledger_generation(self, value) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
        key = _hashable_generation(value)
        if key not in self._ledger_generations:
            self._ledger_generations[key] = len(self._ledger_generations)
        return self._ledger_generations[key]

    def _ledger_occurrence(self, replica: int, stage: int, rank: int):
        self._ledger_serial += 1
        return OccurrenceKey(
            runtime_uid=str(getattr(self.runtime_state, "uid", "memory")),
            launch=self._ledger_serial, ordinal=self._ledger_serial,
            replica=int(replica), rank=int(rank), stage=int(stage))

    def _ledger_boundary(self) -> _LedgerBoundary:
        self._ledger_serial += 1
        return _LedgerBoundary(self._ledger_serial)

    def _typed_spec_bindings(self, replica: int, spec: dict):
        """Translate one legacy grant spec onto canonical typed identities.

        The declared ``rank`` is a storage-holder identity, not a new physical
        die: a producer may name a KV owner rank that the placed envelope does
        not enumerate (for example a TP1 case whose binding still names the
        source rank).  Byte accounting already normalises that legacy value
        through ``_account_rank``; the canonical pool/backing identity must use
        the same accounting rank so a grant cannot fail with
        ``memory_rank_out_of_range`` while the legacy ledger admits it.  The
        raw declared rank stays in ``grant_records``/``grant_bindings``.
        """
        if self.ledger_domain is None:
            return ()
        stage = int(spec.get("stage", 0))
        rank = spec.get("rank")
        rank = 0 if rank is None else int(rank)
        account_rank = self._account_rank(rank)
        amount = float(spec.get("bytes", 0.0))
        generation = self._ledger_generation(spec.get("generation"))
        occurrence = self._ledger_occurrence(replica, stage, rank)
        backing = BackingKey(
            physical_domain=self.ledger_domain.domain_id, rank=account_rank,
            tier=str(spec.get("tier", "hbm")),
            storage_id=str(spec.get("physical_key", "")),
            generation=generation)
        ranges = tuple(
            (int(math.floor(lo)), int(math.ceil(hi)))
            for lo, hi in (spec.get("_ranges") or ()))
        if not ranges and amount > 0:
            ranges = ((0, int(math.floor(amount))),)
        key = AllocationKey(
            occurrence=occurrence, scope_kind="invocation",
            scope_ordinal=int(spec.get("stage", 0)), symbol=self._ledger_serial,
            generation=generation)
        result = [AllocationBinding(
            key=key, pool=self.ledger_pool(
                replica, stage, account_rank, tier=str(spec.get("tier", "hbm")),
                domain=self.ledger_domain.domain_id),
            amount=amount, backing=backing, byte_ranges=ranges,
            owner=str(spec.get("owner", "")),
            storage=str(spec.get("storage", "")), access="readwrite")]
        host = float(spec.get("host_bytes", 0.0))
        if host > 0:
            self._ledger_serial += 1
            host_key = AllocationKey(
                occurrence=self._ledger_occurrence(replica, stage, -1),
                scope_kind="external", scope_ordinal=self._ledger_serial,
                symbol=self._ledger_serial, generation=generation)
            result.append(AllocationBinding(
                key=host_key,
                pool=PoolKey(
                    physical_domain=self.ledger_domain.domain_id, rank=-1,
                    tier="host", name="host", unit=int(replica)),
                amount=host,
                backing=BackingKey(
                    physical_domain=self.ledger_domain.domain_id, rank=-1,
                    tier="host",
                    storage_id=str(spec.get("physical_key", "")) + "#host",
                    generation=generation),
                byte_ranges=((0, int(math.floor(host))),),
                owner=str(spec.get("owner", "")),
                storage=str(spec.get("storage", "")), access="readwrite"))
        return tuple(result)

    def bind_runtime(self, state, *, domain_id: str = "memory") -> CapacityDomainState:
        """Adopt a detached ledger domain into the one runtime root.

        Capacities are the same per-rank envelopes used by ``grant_batch``;
        an unbounded host tier is left absent (no configured ceiling) rather
        than encoded as a fake finite number.  Binding a second, different
        runtime to the same live domain raises ``memory_runtime_conflict``.
        """
        if self.ledger_domain is not None and self.runtime_state is not state:
            raise ValueError("memory_runtime_conflict")
        p = self.case.runtime.parallel
        capacities: dict[PoolKey, float] = {}
        for replica in range(p.dp):
            for stage in range(p.pp):
                for rank, cap in enumerate(self.rank_capacity[replica][stage]):
                    if math.isfinite(float(cap)):
                        capacities[self.ledger_pool(
                            replica, stage, rank, domain=domain_id)] = float(cap)
            host = float(self.host_dram_capacity[replica])
            if math.isfinite(host):
                capacities[PoolKey(
                    physical_domain=domain_id, rank=-1, tier="host",
                    name="host", unit=replica)] = host
        domain = Ledger.ensure_domain(
            state, domain_id, capacities=capacities, adopt=False)
        state.claim_capacity_domain(domain)
        self.runtime_state = state
        self.ledger_domain = domain
        return domain

    def typed_binding(
        self, replica: int, name: str, *, tier: str = "hbm",
        rank: int | None = None, stage: int = 0, scope_kind: str = "invocation",
        scope_ordinal: int = 0, generation: int = 0,
        backing_key: str | None = None, ranges=None, amount: float | None = None,
        acquire: bool = True, parent: AllocationKey | None = None,
        access: str = "readwrite", storage: str = "", owner: str = "",
        occurrence: OccurrenceKey | None = None,
    ) -> AllocationBinding:
        """Typed binding for the canonical ledger; names remain provenance."""
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        record = self.grant_records.get((replica, str(name)), {})
        if rank is None:
            rank = record.get("rank")
            if rank is None:
                rank = 0
        if stage == 0:
            stage = int(record.get("stage", stage))
        if not 0 <= stage < p.pp:
            raise ValueError("memory_stage_out_of_range")
        value = float(record.get("bytes", 0.0) if amount is None else amount)
        domain_id = self.ledger_domain.domain_id
        pool = self.ledger_pool(replica, stage, rank, tier=tier, domain=domain_id)
        occurrence = occurrence or OccurrenceKey(
            runtime_uid=str(getattr(self.runtime_state, "uid", "memory")),
            launch=0, ordinal=scope_ordinal, replica=replica, rank=rank,
            stage=stage)
        key = AllocationKey(
            occurrence=occurrence, scope_kind=scope_kind,
            scope_ordinal=scope_ordinal, symbol=int(scope_ordinal),
            generation=int(generation))
        backing = None
        if backing_key is not None:
            backing = BackingKey(
                physical_domain=domain_id, rank=int(rank), tier=tier,
                storage_id=str(backing_key), generation=int(generation))
        raw = ranges if ranges is not None else record.get("ranges")
        byte_ranges = _half_open_ranges(raw, value)
        return AllocationBinding(
            key=key, pool=pool, amount=value, backing=backing,
            byte_ranges=byte_ranges, reservation_parent=parent,
            owner=str(owner or record.get("owner", "")),
            storage=str(storage or record.get("storage", "")),
            access=access, acquire=acquire)

    def grant_typed(self, bindings, *, holder, at):
        """Atomic typed grant against the adopted RuntimeState domain."""
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        return Ledger.try_acquire(self.runtime_state, tuple(bindings),
                                  holder=holder, at=at)

    def retain_reader_typed(self, allocation: AllocationKey, *, reader,
                            ranges=(), expected_generation=None):
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        return Ledger.retain_reader(
            self.runtime_state, allocation, reader=reader, ranges=ranges,
            expected_generation=expected_generation)

    def release_typed(self, allocation: AllocationKey, *, after,
                      expected_generation=None):
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        return Ledger.release(
            self.runtime_state, allocation, after=after,
            expected_generation=expected_generation)

    def ledger_usage(self) -> dict[PoolKey, float]:
        if self.ledger_domain is None:
            return {}
        return dict(self.ledger_domain.current)

    # ------------------------------------------------------------------
    # Unified physical reference ledger
    # ------------------------------------------------------------------
    def _account_rank(self, rank) -> int:
        """Rank used for byte accounting; legacy out-of-range values clamp to 0."""
        value = 0 if rank is None else int(rank)
        return value if 0 <= value < self.width else 0

    def _make_ref(
        self, replica: int, *, rank, stage, storage, tier, physical_key,
        generation, size, ranges=None, host_bytes=0.0, host_ranges=None,
    ) -> _PhysicalRef:
        if not 0 <= int(stage) < self.case.runtime.parallel.pp:
            raise ValueError("memory_physical_stage_out_of_range")
        tier = str(tier or "hbm").lower()
        ranges = _half_open_ranges(ranges, size)
        host = _half_open_ranges(host_ranges, host_bytes) if host_ranges is not None else (
            ((0.0, float(host_bytes)),) if float(host_bytes) > 0.0 else ())
        if tier == "host":
            host = ranges or host
            ranges = ()
        return _PhysicalRef(
            rank=self._account_rank(rank), stage=int(stage),
            storage=str(storage or ""), tier=tier,
            physical_key=str(physical_key or ""),
            generation=_hashable_generation(generation),
            ranges=tuple(ranges), host_ranges=tuple(host),
        )

    def _set_ref(self, holder: tuple[int, str, str],
                 refs: tuple[_PhysicalRef, ...]) -> None:
        if refs:
            self._refs[holder] = tuple(refs)
        else:
            self._refs.pop(holder, None)

    def _selected_refs(
        self, replica: int, stage: int, items: dict[str, Allocation],
    ) -> dict[str, _PhysicalRef]:
        """One physical reference per selected holder, aliasing explicit roots."""
        by_name = dict(items)
        refs: dict[str, _PhysicalRef] = {}
        for item in items.values():
            source = item
            if item.pool:
                # A suballocation is inside its declared pool reservation.
                # Both are holders on the same backing; the reservation's
                # bytes already cover the suballocation's range.
                pool = by_name.get(item.pool)
                if pool is not None:
                    source = pool
                key = str(getattr(source, "physical_key", None) or item.pool)
            elif item.alias_of:
                seen = set()
                current = item
                while current.alias_of:
                    if current.name in seen:
                        break
                    seen.add(current.name)
                    target = by_name.get(current.alias_of)
                    if target is None:
                        break
                    current = target
                source = current
                key = str(getattr(source, "physical_key", None)
                          or getattr(item, "physical_key", None)
                          or item.alias_of)
            else:
                key = str(getattr(item, "physical_key", None) or item.name)
            ranges = getattr(item, "ranges", None)
            if ranges is None:
                ranges = getattr(source, "ranges", None)
            refs[item.name] = self._make_ref(
                replica, rank=getattr(source, "rank", None),
                stage=stage,
                storage=getattr(source, "storage", None),
                tier=getattr(source, "tier", "hbm"),
                physical_key=key,
                generation=getattr(source, "generation", None),
                size=item.bytes, ranges=ranges,
            )
        return refs

    def _grant_ref(self, spec: dict) -> _PhysicalRef:
        return self._make_ref(
            spec["replica"], rank=spec.get("rank"), stage=spec["stage"],
            storage=spec.get("storage"), tier=spec.get("tier", "hbm"),
            physical_key=spec.get("physical_key") or spec["name"],
            generation=spec.get("generation"), size=spec["bytes"],
            ranges=spec.get("_ranges"),
            host_bytes=spec.get("host_bytes", 0.0),
        )

    def _shared_ref(self, replica: int, physical: str, record: dict):
        refs = []
        for stage, amount in enumerate(record["stages"]):
            if amount <= 0.0:
                continue
            refs.append(self._make_ref(
                replica, rank=record.get("rank"), stage=stage,
                storage=record.get("storage"), tier=record.get("tier", "hbm"),
                physical_key=physical, generation=None,
                size=float(amount),
            ))
        return tuple(refs)

    def _request_usage_ranks(self) -> list[list[list[float]]]:
        result = [
            [[0.0] * self.width for _ in range(self.case.runtime.parallel.pp)]
            for _ in range(self.case.runtime.parallel.dp)
        ]
        for (replica, _request), rows in self.owner_ranks.items():
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    result[replica][stage][rank] += float(value)
        return result

    def _staging_usage_ranks(self) -> list[list[list[float]]]:
        result = [
            [[0.0] * self.width for _ in range(self.case.runtime.parallel.pp)]
            for _ in range(self.case.runtime.parallel.dp)
        ]
        for (replica, _request), rows in self.staging_owner_ranks.items():
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    result[replica][stage][rank] += float(value)
        return result

    def _state_usage(
        self, *, extra_holders=(), drop_holders=(), exclude_request=None,
        request_usage=None, staging_usage=None, host_request=None,
    ):
        """Candidate HBM/host usage without mutating any ledger state."""
        p = self.case.runtime.parallel
        hbm = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        host = [0.0] * p.dp
        holders = dict(self._refs)
        for holder in drop_holders:
            holders.pop(holder, None)
        for holder, refs in extra_holders:
            holders[holder] = tuple(refs)
        backing_ranges: dict[tuple, list] = {}
        host_ranges: dict[tuple, list] = {}
        for holder, refs in holders.items():
            replica = holder[0]
            for ref in refs:
                if ref.tier == "host":
                    if ref.host_ranges:
                        host_ranges.setdefault((
                            replica, ref.rank, ref.stage, ref.storage,
                            ref.physical_key, ref.generation), []).extend(
                                ref.host_ranges)
                    continue
                if ref.tier not in {"hbm", ""}:
                    # On-chip/L2/SSD pools are owned by their real allocator;
                    # this ledger only budgets attached HBM and host DRAM.
                    continue
                backing_ranges.setdefault((
                    replica, ref.rank, ref.stage, ref.storage, ref.tier,
                    ref.physical_key, ref.generation), []).extend(ref.ranges)
                if ref.host_ranges:
                    host_ranges.setdefault((
                        replica, ref.rank, ref.stage, ref.storage,
                        ref.physical_key, ref.generation), []).extend(
                            ref.host_ranges)
        for (replica, rank, stage, _storage, _tier, _key, _generation), ranges in \
                backing_ranges.items():
            hbm[replica][stage][rank] += _union_bytes(ranges)
        for (replica, _request), rows in self.owner_ranks.items():
            if (replica, _request) == exclude_request:
                continue
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    hbm[replica][stage][rank] += float(value)
        for (replica, _request), rows in self.staging_owner_ranks.items():
            if (replica, _request) == exclude_request:
                continue
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    hbm[replica][stage][rank] += float(value)
        if request_usage is not None:
            for replica, stages in enumerate(request_usage):
                for stage, row in enumerate(stages):
                    for rank, value in enumerate(row):
                        hbm[replica][stage][rank] += float(value)
        if staging_usage is not None:
            for replica, stages in enumerate(staging_usage):
                for stage, row in enumerate(stages):
                    for rank, value in enumerate(row):
                        hbm[replica][stage][rank] += float(value)
        for (replica, request), value in self.host_dram_owners.items():
            if (replica, request) == exclude_request:
                continue
            host[replica] += float(value)
        if host_request is not None:
            for replica, value in enumerate(host_request):
                host[replica] += float(value)
        for (replica, _rank, _stage, _storage, _key, _generation), ranges in \
                host_ranges.items():
            host[replica] += _union_bytes(ranges)
        return hbm, host

    def _usage_budget_ok(self, hbm, host) -> bool:
        for replica in range(self.case.runtime.parallel.dp):
            for stage, row in enumerate(hbm[replica]):
                capacities = self.rank_capacity[replica][stage]
                for rank, used in enumerate(row):
                    cap = capacities[rank]
                    if used > cap + max(1e-9, abs(cap) * 1e-12):
                        return False
            cap = self.host_dram_capacity[replica]
            if host[replica] > cap + max(1e-9, abs(cap) * 1e-12):
                return False
        return True

    def _ref_category_usage(self):
        p = self.case.runtime.parallel
        selected = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        shared = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        granted = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        readers = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        backings: dict[tuple, dict] = {}
        for holder, refs in self._refs.items():
            kind = holder[1]
            if kind == "grant_reader":
                category = "reader"
            elif kind == "selected":
                category = "selected"
            elif kind == "shared":
                category = "shared"
            elif kind == "grant":
                category = "grant"
            else:
                category = kind
            for ref in refs:
                if ref.tier not in {"hbm", ""} or not ref.ranges:
                    continue
                key = (holder[0], ref.rank, ref.stage, ref.storage,
                       ref.tier, ref.physical_key, ref.generation)
                entry = backings.setdefault(key, {"ranges": [], "kinds": set()})
                entry["ranges"].extend(ref.ranges)
                entry["kinds"].add(category)
        for (replica, rank, stage, _storage, _tier, _key, _generation), entry in \
                backings.items():
            value = _union_bytes(entry["ranges"])
            kinds = entry["kinds"]
            if "selected" in kinds:
                selected[replica][stage][rank] += value
            elif "shared" in kinds:
                shared[replica][stage][rank] += value
            elif "grant" in kinds:
                granted[replica][stage][rank] += value
            elif "reader" in kinds:
                readers[replica][stage][rank] += value
        return selected, shared, granted, readers

    def _sync_usage(self) -> None:
        """Recompute committed counters and peaks from the current holders."""
        p = self.case.runtime.parallel
        hbm, host = self._state_usage()
        request = self._request_usage_ranks()
        staging = self._staging_usage_ranks()
        selected, shared, granted, readers = self._ref_category_usage()
        self.selected_rank = selected
        self.shared_rank = shared
        self.granted_rank = granted
        self.reader_rank = readers
        self.live_rank = request
        self.staging_rank = staging
        for replica in range(p.dp):
            self.live[replica] = [max(row) for row in request[replica]]
            self.staging_live[replica] = float(sum(
                max(row) for row in staging[replica]))
            self.host_dram_live[replica] = float(host[replica])
            self.host_dram_peak[replica] = max(
                self.host_dram_peak[replica], self.host_dram_live[replica])
            for stage in range(p.pp):
                total = max(hbm[replica][stage])
                resident = max(request[replica][stage])
                stage_staging = max(staging[replica][stage])
                self.peak_combined[replica][stage] = max(
                    self.peak_combined[replica][stage], total)
                self.peak[replica][stage] = max(
                    self.peak[replica][stage], resident + stage_staging)
                self.peak_kv_resident[replica][stage] = max(
                    self.peak_kv_resident[replica][stage], resident)
                self.peak_kv_staging[replica][stage] = max(
                    self.peak_kv_staging[replica][stage], stage_staging)
                self.selected_reserved[replica][stage] = max(
                    selected[replica][stage])
                self.shared_reserved[replica][stage] = max(
                    shared[replica][stage])
                self.granted[replica][stage] = max(granted[replica][stage])
        self._rebuild_grant_backings()

    def _rebuild_grant_backings(self) -> None:
        backings: dict[tuple, dict] = {}
        for (replica, name), record in self.grant_records.items():
            binding = self.grant_bindings.get((replica, name), {})
            key = (replica, binding.get("storage"), binding.get("tier"),
                   binding.get("physical_key"), binding.get("rank"))
            row = backings.setdefault(key, {
                "stages": [0.0] * self.case.runtime.parallel.pp,
                "host": 0.0, "names": [],
            })
            for stage, value in enumerate(self.grants.get((replica, name),
                                                          ())):
                row["stages"][stage] = max(row["stages"][stage], float(value))
            row["host"] = max(row["host"], float(
                self.grant_hosts.get((replica, name), 0.0)))
            row["names"].append(name)
        self.grant_backings = backings
        self.granted_host = [0.0] * self.case.runtime.parallel.dp
        for key, row in backings.items():
            self.granted_host[key[0]] += float(row["host"])

    def occupied_bytes(self, replica: int, stage: int, *, rank: int | None = None) -> float:
        """Committed HBM bytes (excluding fixed weights/reserve/workspace)."""
        hbm, _host = self._state_usage()
        if rank is None:
            return float(max(hbm[replica][stage]))
        return float(hbm[replica][stage][self._account_rank(rank)])

    def host_occupied_bytes(self, replica: int) -> float:
        _hbm, host = self._state_usage()
        return float(host[replica])

    def physical_usage(self, replica: int) -> dict:
        """Explain committed per-rank usage from the same candidate ledger."""
        hbm, host = self._state_usage()
        return {
            "replica": replica,
            "hbm_bytes_per_stage_rank": hbm[replica],
            "host_bytes": host[replica],
            "rank_capacity_bytes_per_stage_rank": self.rank_capacity[replica],
            "budget_version": self.budget_version,
        }

    def _replica_occupied(self, replica: int) -> float:
        hbm, _host = self._state_usage()
        return float(sum(max(row) for row in hbm[replica]))

    def _resolve_workspace_tokens(self, tokens=None) -> int:
        runtime = self.case.runtime
        if tokens is None:
            return min(runtime.max_num_batched_tokens, sum(
                max(1, min(r.prompt_tokens, runtime.max_num_batched_tokens)) for r in self.case.workload.requests))
        if type(tokens) is not int or tokens < 0:
            raise ValueError("memory_workspace_tokens_require_nonnegative_integer")
        return tokens

    def _async_inflight_workspace(self, tokens=None) -> list[float]:
        """Ping-pong step input/residual/output buffer retained across overlapped async steps."""
        model, runtime = self.case.model, self.case.runtime
        p = runtime.parallel
        resolved = self._resolve_workspace_tokens(tokens)
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            stage = self.layer_stage[layer.index]
            residual = model.hidden_size * (2 + max(1, layer.hc_mult))
            step_buffer = resolved * residual * dtype_bytes(model.activation_dtype)
            result[stage] = max(result[stage], step_buffer)
        if self.draft_kind and model.auxiliary:
            from mindie_sim.catalog.draft import optional_draft_spec
            draft = optional_draft_spec(model, self.draft_kind)
            feature_bytes = resolved * model.hidden_size * len(draft.target_layer_ids) * dtype_bytes(model.activation_dtype)
            result = [value + feature_bytes for value in result]
        return [value * self.async_workspace_buffer_multiplier for value in result]

    def _workspace(self, tokens=None):
        """Resident simultaneous tensors for one layer, not lifetime traffic."""
        model, runtime = self.case.model, self.case.runtime
        p = runtime.parallel
        tokens = self._resolve_workspace_tokens(tokens)
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            residual = model.hidden_size * (2 + max(1, layer.hc_mult))
            if layer.residual_block_size:
                residual += model.hidden_size * (layer.index // layer.residual_block_size + 1)
            if layer.attention in {"kda", "gdn"}:
                attention = (layer.conv_width + layer.linear_value_heads * layer.linear_head_dim) / p.tp
            elif layer.attention == "gqa":
                attention = (layer.num_heads / p.tp + 2 * local_kv_heads(layer.num_kv_heads, p.tp)) * layer.head_dim
            else:
                attention = layer.q_lora_rank + layer.kv_lora_rank + layer.num_heads / p.tp * (
                    layer.qk_nope_head_dim + layer.qk_rope_head_dim + layer.v_head_dim)
            ffn = (3 * layer.intermediate_size / p.tp if layer.ffn == "dense" else
                   layer.top_k * (2 * (layer.expert_latent_size or model.hidden_size)
                                  + 3 * layer.moe_intermediate_size / p.etp))
            footprint = tokens * (residual + max(attention, ffn)) * dtype_bytes(model.activation_dtype)
            stage = self.layer_stage[layer.index]
            result[stage] = max(result[stage], footprint)
        if self.draft_kind and model.auxiliary:
            from mindie_sim.catalog.draft import optional_draft_spec
            draft = optional_draft_spec(model, self.draft_kind)
            feature_bytes = tokens * model.hidden_size * len(draft.target_layer_ids) * dtype_bytes(model.activation_dtype)
            result = [value + feature_bytes for value in result]
        if model.vision:
            from mindie_sim.catalog.media import media_geometry
            for request in self.case.workload.requests:
                if not request.media_tokens or request.media_encoded:
                    continue
                patches, *_ = media_geometry(runtime, model.vision, request.media_tokens)
                vision = model.vision
                # The default vision recipe materializes FP32 attention scores.
                scores = 2 * patches * patches * vision.num_heads * 4
                tensors = patches * (vision.qkv_width + 2 * vision.hidden_size
                                     + 2 * vision.intermediate_size) * dtype_bytes(model.activation_dtype)
                result[0] = max(result[0], scores + tensors)
        return result

    def cache_bytes(self, tokens: int, *, padded=True,
                    query_tokens: int | None = None) -> tuple[float, ...]:
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        declared = int(dict(case.runtime.metadata).get(
            "draft_proposal_tokens", ZERO) or ZERO)
        q_tok = (self.query_tokens if query_tokens is None
                 else max(ONE, int(query_tokens)))
        if (query_tokens is None and self.draft_kind == "dspark"
                and not self.paired_sfa and declared <= ZERO):
            # A bare DSpark spec widens the draft pool/page geometry; the
            # dense target state keeps its single-row reservation.
            q_tok = ONE
        result = (self.pages.allocation_stage_bytes(tokens, case.runtime.max_model_len, p.pp)
                  if padded else [0.0] * p.pp)
        if model.layer_plans:
            for layer in model.layer_plans:
                components = layer_cache_components(
                    model, layer, p, tokens, capacity_padded=padded,
                    interleave=case.runtime.kv_block_size,
                    runtime=case.runtime, query_tokens=q_tok)
                covered = self.pages.covered.get(layer.index, ()) if padded else ()
                result[self.layer_stage[layer.index]] += sum(
                    value for name, value in components.items() if name not in covered)
        from mindie_sim.catalog.draft import optional_draft_spec
        mode = self.draft_kind
        if self.paired_sfa and mode and not padded:
            result[-ONE] += self._paired_draft_bytes(tokens)
        elif mode and model.auxiliary is not None and not self.paired_sfa:
            if self._uses_inherited_draft_cache(model):
                result[-ONE] += self._inherited_draft_cache_bytes(
                    model, tokens, padded=padded, query_tokens=q_tok)
            else:
                draft = optional_draft_spec(model, mode)
                row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim) if draft.attention_kind == "mla"
                       else 2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
                draft_tokens = (math.ceil(tokens / case.runtime.kv_block_size) * case.runtime.kv_block_size
                                if padded else tokens)
                result[-1] += draft_tokens * row * dtype_bytes(model.kv_dtype) * draft.layers
        return tuple(result)

    def _uses_inherited_draft_cache(self, model) -> bool:
        """True when draft KV/state follows the declared draft LayerPlans."""
        from mindie_sim.catalog.cache import uses_inherited_draft_cache
        return (not self.paired_sfa
                and uses_inherited_draft_cache(model, self.draft_kind))

    def _inherited_draft_cache_bytes(
        self, model, tokens: int, *, padded: bool, rank: int | None = None,
        query_tokens: int | None = None,
    ) -> float:
        """Draft cache payload from the inherited plans, with T05 ownership."""
        from mindie_sim.catalog.cache_ownership import draft_cache_components

        values = draft_cache_components(
            model, self.case.runtime.parallel, draft_kind=self.draft_kind,
            sequence_tokens=tokens, capacity_padded=padded,
            runtime=self.case.runtime, rank=rank,
            owner_spans=self.rank_token_owners if rank is not None else None,
            query_tokens=(self.query_tokens if query_tokens is None
                          else query_tokens),
        )
        return float(values["draft_kv"])

    def _group_cache_parts(self, tokens: int, rank: int | None = None) -> list[dict]:
        """Known physical and valid components inside declared page groups.

        Token-paged components use the group's allocated token slots for
        capacity and its valid/window span for content.  Compressed/index and
        fixed-state components use their own valid geometry; the undeclared
        remainder of the group quote is kept as ``reserve``, never asserted as
        precise payload.  With ``rank`` the valid components are that physical
        rank's owner rows, while every rank still holds the full page slots.
        """
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        stages = [
            {"alloc_payload": 0.0, "alloc_scale": 0.0,
             "valid_payload": 0.0, "valid_scale": 0.0, "components": {}}
            for _ in range(p.pp)
        ]
        by_index = {plan.index: plan for plan in model.layer_plans}
        for group in self.pages.groups:
            owned = self.pages.group_components(group)
            valid_tokens = group.valid_tokens(tokens, case.runtime.max_model_len)
            alloc_tokens = group.allocated_token_slots(tokens, case.runtime.max_model_len)
            for layer_index in group.layers:
                layer = by_index[layer_index]
                components = layer_cache_components(
                    model, layer, p, valid_tokens, capacity_padded=False,
                    interleave=case.runtime.kv_block_size,
                    rank=rank, owner_spans=self.rank_token_owners,
                    runtime=case.runtime, query_tokens=self.query_tokens,
                )
                stage = stages[self.layer_stage[layer.index]]
                for name in owned.get(layer_index, ()):
                    part = None
                    if not self.paired_sfa and name in {"gqa_kv", "mla_kv", "window"}:
                        part = self._group_token_component(
                            model, layer, p, group, name, valid_tokens,
                            alloc_tokens, components,
                        )
                    if part is None:
                        value = components.get(name, 0.0)
                        if value <= 0.0:
                            continue
                        payload, scale = cache_component_split(
                            model, layer, p, name, value)
                        alloc_payload = valid_payload = payload
                        alloc_scale = valid_scale = scale
                    else:
                        alloc_payload, alloc_scale, valid_payload, valid_scale = part
                    stage["alloc_payload"] += alloc_payload
                    stage["alloc_scale"] += alloc_scale
                    stage["valid_payload"] += valid_payload
                    stage["valid_scale"] += valid_scale
                    detail = stage["components"].setdefault(name, {
                        "alloc_payload": 0.0, "alloc_scale": 0.0,
                        "valid_payload": 0.0, "valid_scale": 0.0,
                    })
                    detail["alloc_payload"] += alloc_payload
                    detail["alloc_scale"] += alloc_scale
                    detail["valid_payload"] += valid_payload
                    detail["valid_scale"] += valid_scale
        if self.paired_sfa and self.draft_kind:
            value = self._paired_draft_bytes(tokens, rank=rank)
            stage = stages[-1]
            for name in ("alloc_payload", "valid_payload"):
                stage[name] += value
            stage["components"]["draft_kv"] = {
                "alloc_payload": value, "alloc_scale": ZERO,
                "valid_payload": value, "valid_scale": ZERO}
        return stages

    def _paired_draft_bytes(self, tokens: int, *, rank=None) -> float:
        from mindie_sim.catalog.draft import optional_draft_spec
        from mindie_sim.catalog.paired_sfa import draft_row
        draft = optional_draft_spec(self.case.model, self.draft_kind)
        rows = tokens / self.case.runtime.parallel.cp
        if rank is not None and self.rank_token_owners is not None:
            rows = owned_token_rows(tokens, self.rank_token_owners, rank)
        return rows * draft_row(self.case.model, draft) * draft.layers

    def _group_token_component(
        self, model, layer, parallel, group, name: str, valid_tokens: int,
        alloc_tokens: int, components: dict[str, float],
    ):
        """Rows/bytes for a token-paged component inside one group page."""
        if name == "gqa_kv":
            if (layer.attention not in {"gqa", "projection_only"}
                    or 0 <= layer.kv_owner_layer_id < layer.index):
                return None
            region = "gqa_kv"
            width = 2 * local_kv_heads(layer.num_kv_heads, parallel.tp) * layer.head_dim
            cap = layer.window or 0
        elif name == "mla_kv":
            if layer.attention != "mla" or layer.cache_kind != "dense":
                return None
            region = "mla_kv"
            width = layer.cache_row_elements
            cap = layer.window or 0
        else:  # window
            if layer.attention != "mla" or not layer.window:
                return None
            region = "window"
            width = layer.cache_row_elements
            cap = layer.window
        if name not in components and valid_tokens > 0:
            return None
        # The physical page owns all allocated slots.  Valid content comes from
        # the already CP/TP-local layer_cache_components range, not a second
        # global valid_tokens calculation that would overwrite the local shard.
        alloc_rows = alloc_tokens
        payload, scale = cache_row_split(model, region, width)
        valid_value = components.get(name, 0.0)
        if valid_value > 0.0:
            valid_payload, valid_scale = cache_component_split(
                model, layer, parallel, name, valid_value)
        else:
            valid_payload = valid_scale = 0.0
        return (alloc_rows * payload, alloc_rows * scale,
                valid_payload, valid_scale)

    def _kv_residency(self, tokens: int) -> tuple[list[float], list[float], list[float]]:
        """(gross, HBM-resident, staging) KV bytes, same basis as resize."""
        gross = list(self.cache_bytes(tokens))
        if self.kvc_hbm_ratio == 1.0:
            return gross, list(gross), [0.0] * len(gross)
        resident = [value * self.kvc_hbm_ratio for value in gross]
        staging = [
            value * (1.0 - self.kvc_hbm_ratio)
            for value in self._max_layer_cache_bytes(tokens)
        ]
        return gross, resident, staging

    def _cache_component_breakdown(
        self, tokens: int, *, padded: bool, skip_covered: bool,
        rank: int | None = None,
    ) -> list[dict]:
        """Per-stage component payload/scale split for one liveness view."""
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        stages = [
            {"payload": 0.0, "scale": 0.0, "total": 0.0, "components": {}}
            for _ in range(p.pp)
        ]
        for layer in model.layer_plans:
            components = layer_cache_components(
                model, layer, p, tokens, capacity_padded=padded,
                interleave=case.runtime.kv_block_size,
                rank=rank, owner_spans=self.rank_token_owners,
                runtime=case.runtime, query_tokens=self.query_tokens,
            )
            covered = self.pages.covered.get(layer.index, ()) if skip_covered else ()
            stage = stages[self.layer_stage[layer.index]]
            for name, value in components.items():
                if name in covered:
                    continue
                payload, scale = cache_component_split(
                    model, layer, p, name, value)
                stage["payload"] += payload
                stage["scale"] += scale
                stage["total"] += float(value)
                detail = stage["components"].setdefault(
                    name, {"payload": 0.0, "scale": 0.0, "total": 0.0})
                detail["payload"] += payload
                detail["scale"] += scale
                detail["total"] += float(value)
        from mindie_sim.catalog.draft import optional_draft_spec
        mode = self.draft_kind
        if mode and model.auxiliary is not None and not self.paired_sfa:
            if self._uses_inherited_draft_cache(model):
                value = self._inherited_draft_cache_bytes(
                    model, tokens, padded=padded, rank=rank)
            else:
                draft = optional_draft_spec(model, mode)
                row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim)
                       if draft.attention_kind == "mla" else
                       2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
                draft_tokens = (math.ceil(tokens / case.runtime.kv_block_size)
                                * case.runtime.kv_block_size if padded else tokens)
                if rank is not None and self.rank_token_owners is not None:
                    draft_tokens = owned_token_rows(
                        draft_tokens, self.rank_token_owners, rank)
                value = draft_tokens * row * dtype_bytes(model.kv_dtype) * draft.layers
            stage = stages[-ONE]
            stage["payload"] += float(value)
            stage["total"] += float(value)
            detail = stage["components"].setdefault(
                "draft_kv", {"payload": 0.0, "scale": 0.0, "total": 0.0})
            detail["payload"] += float(value)
            detail["total"] += float(value)
        return stages

    def _rank_cache_bytes(self, tokens: int, *, padded: bool, rank: int) -> list[float]:
        """Allocation bytes for one physical rank.

        With an explicit T05 owner partition the rank owns its exact rows.
        Without one the legacy stage-level per-rank value is retained, so old
        callers and even CP=2 slices are unchanged.
        """
        if self.paired_sfa or self.rank_token_owners is None:
            return list(self.cache_bytes(tokens, padded=padded))
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        result = (
            self.pages.allocation_stage_bytes(tokens, case.runtime.max_model_len, p.pp)
            if padded else [0.0] * p.pp
        )
        for layer in model.layer_plans:
            components = layer_cache_components(
                model, layer, p, tokens, capacity_padded=padded,
                interleave=case.runtime.kv_block_size,
                rank=int(rank), owner_spans=self.rank_token_owners,
                runtime=case.runtime, query_tokens=self.query_tokens,
            )
            covered = self.pages.covered.get(layer.index, ()) if padded else ()
            result[self.layer_stage[layer.index]] += sum(
                value for name, value in components.items() if name not in covered)
        from mindie_sim.catalog.draft import optional_draft_spec
        mode = self.draft_kind
        if mode and model.auxiliary is not None:
            if self._uses_inherited_draft_cache(model):
                result[-1] += self._inherited_draft_cache_bytes(
                    model, tokens, padded=padded, rank=int(rank))
            else:
                draft = optional_draft_spec(model, mode)
                row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim)
                       if draft.attention_kind == "mla"
                       else 2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
                draft_tokens = (math.ceil(tokens / case.runtime.kv_block_size)
                                * case.runtime.kv_block_size if padded else tokens)
                rows = owned_token_rows(draft_tokens, self.rank_token_owners, int(rank))
                result[-1] += rows * row * dtype_bytes(model.kv_dtype) * draft.layers
        return result

    def _rank_max_layer_cache_bytes(self, tokens: int, rank: int) -> tuple[float, ...]:
        if self.rank_token_owners is None:
            return self._max_layer_cache_bytes(tokens)
        model, p = self.case.model, self.case.runtime.parallel
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            comps = layer_cache_components(
                model, layer, p, tokens, capacity_padded=False,
                interleave=self.case.runtime.kv_block_size,
                rank=int(rank), owner_spans=self.rank_token_owners,
                runtime=self.case.runtime, query_tokens=self.query_tokens,
            )
            st = self.layer_stage[layer.index]
            result[st] = max(result[st], float(sum(comps.values())))
        return tuple(result)

    def _kv_residency_ranks(
        self, tokens: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """(HBM-resident, staging) per stage and physical rank."""
        p = self.case.runtime.parallel
        if self.rank_token_owners is None:
            _gross, resident, staging = self._kv_residency(tokens)
            return (
                [[float(resident[stage])] * self.width for stage in range(p.pp)],
                [[float(staging[stage])] * self.width for stage in range(p.pp)],
            )
        resident = [[0.0] * self.width for _ in range(p.pp)]
        staging = [[0.0] * self.width for _ in range(p.pp)]
        for rank in range(self.width):
            gross = self._rank_cache_bytes(tokens, padded=True, rank=rank)
            layer_max = self._rank_max_layer_cache_bytes(tokens, rank)
            for stage in range(p.pp):
                if self.kvc_hbm_ratio == 1.0:
                    resident[stage][rank] = float(gross[stage])
                    staging[stage][rank] = 0.0
                else:
                    resident[stage][rank] = float(gross[stage]) * self.kvc_hbm_ratio
                    staging[stage][rank] = (
                        float(layer_max[stage]) * (1.0 - self.kvc_hbm_ratio))
        return resident, staging

    def _allocation_view(
        self, tokens: int, *, rank: int | None = None,
        include_components: bool = True,
    ) -> dict:
        """One physical rank's (or the legacy stage's) cache allocation view."""
        case, p = self.case, self.case.runtime.parallel
        allocated = self._cache_component_breakdown(
            tokens, padded=True, skip_covered=True, rank=rank)
        valid = self._cache_component_breakdown(
            tokens, padded=False, skip_covered=True, rank=rank)
        group_parts = self._group_cache_parts(tokens, rank=rank)
        group_payload = self.pages.stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        group_metadata = self.pages.metadata_stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        group_access_metadata = self.pages.access_metadata_stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        result = {
            "sequence_tokens": int(tokens),
            "basis": "padded_cache_pages_separate_from_valid_access",
            "allocated": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "reserve_bytes_per_stage_rank": [],
                "metadata_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "valid": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "page_tail": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "reserve_bytes_per_stage_rank": [],
                "metadata_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "access": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "metadata_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "components": [],
            "page_groups": {
                "payload_bytes_per_stage_rank": list(group_payload),
                "metadata_bytes_per_stage_rank": list(group_metadata),
                "access_metadata_bytes_per_stage_rank": list(group_access_metadata),
                "component_payload_bytes_per_stage_rank":
                    [stage["alloc_payload"] for stage in group_parts],
                "component_scale_bytes_per_stage_rank":
                    [stage["alloc_scale"] for stage in group_parts],
                "valid_component_payload_bytes_per_stage_rank":
                    [stage["valid_payload"] for stage in group_parts],
                "valid_component_scale_bytes_per_stage_rank":
                    [stage["valid_scale"] for stage in group_parts],
            },
        }
        if rank is not None:
            result["physical_rank"] = int(rank)
        for index in range(p.pp):
            known_payload = group_parts[index]["alloc_payload"]
            known_scale = group_parts[index]["alloc_scale"]
            reserve = max(0.0, group_payload[index] - known_payload - known_scale)
            alloc_payload = allocated[index]["payload"] + known_payload
            alloc_scale = allocated[index]["scale"] + known_scale
            metadata = group_metadata[index]
            access_metadata = group_access_metadata[index]
            alloc_total = alloc_payload + alloc_scale + reserve + metadata
            valid_payload = valid[index]["payload"] + group_parts[index]["valid_payload"]
            valid_scale = valid[index]["scale"] + group_parts[index]["valid_scale"]
            valid_total = valid_payload + valid_scale
            tail_payload = max(0.0, alloc_payload - valid_payload)
            tail_scale = max(0.0, alloc_scale - valid_scale)
            tail_metadata = max(0.0, metadata - access_metadata)
            result["allocated"]["payload_bytes_per_stage_rank"].append(alloc_payload)
            result["allocated"]["scale_bytes_per_stage_rank"].append(alloc_scale)
            result["allocated"]["reserve_bytes_per_stage_rank"].append(reserve)
            result["allocated"]["metadata_bytes_per_stage_rank"].append(metadata)
            result["allocated"]["total_bytes_per_stage_rank"].append(alloc_total)
            result["valid"]["payload_bytes_per_stage_rank"].append(valid_payload)
            result["valid"]["scale_bytes_per_stage_rank"].append(valid_scale)
            result["valid"]["total_bytes_per_stage_rank"].append(valid_total)
            result["page_tail"]["payload_bytes_per_stage_rank"].append(tail_payload)
            result["page_tail"]["scale_bytes_per_stage_rank"].append(tail_scale)
            result["page_tail"]["reserve_bytes_per_stage_rank"].append(reserve)
            result["page_tail"]["metadata_bytes_per_stage_rank"].append(tail_metadata)
            result["page_tail"]["total_bytes_per_stage_rank"].append(
                tail_payload + tail_scale + reserve)
            result["access"]["payload_bytes_per_stage_rank"].append(valid_payload)
            result["access"]["scale_bytes_per_stage_rank"].append(valid_scale)
            result["access"]["metadata_bytes_per_stage_rank"].append(access_metadata)
            result["access"]["total_bytes_per_stage_rank"].append(
                valid_payload + valid_scale + access_metadata)
            if include_components:
                result["components"].append({
                    "stage": index,
                    "allocated": allocated[index]["components"],
                    "group_allocated": group_parts[index]["components"],
                    "valid": valid[index]["components"],
                })
        expected = (list(self.cache_bytes(tokens)) if rank is None else
                    self._rank_cache_bytes(tokens, padded=True, rank=rank))
        if any(abs(left - right) > max(1.0, abs(right) * 1e-9) for left, right in zip(
                result["allocated"]["total_bytes_per_stage_rank"], expected)):
            raise ValueError("memory_cache_allocation_not_conserved")
        return result

    def rank_cache_allocation(self, tokens: int) -> tuple[dict, ...]:
        """Per-physical-rank allocated/valid/page-tail view (same ledger basis)."""
        if self.rank_token_owners is None:
            view = self._allocation_view(tokens, rank=None)
            return tuple(dict(view) for _ in range(self.width))
        return tuple(
            self._allocation_view(tokens, rank=rank)
            for rank in range(self.width)
        )

    def cache_allocation(self, tokens: int, *, include_components: bool = True) -> dict:
        """Physical cache allocation versus valid logical access.

        Page groups keep their known component row scales instead of being
        flattened to opaque payload.  Unknown group internals remain a declared
        group price.  Allocation metadata is charged for every reserved page;
        access metadata counts only pages touched by this valid span or a fixed
        ring's declared touched page.  ``physical_ranks`` carries each rank's
        exact allocated/valid/tail view when a T05 owner partition is supplied.
        """
        result = self._allocation_view(
            tokens, rank=None, include_components=include_components)
        if self.rank_token_owners is not None:
            result["physical_ranks"] = [
                self._allocation_view(tokens, rank=rank,
                                      include_components=include_components)
                for rank in range(self.width)
            ]
        return result

    def _max_layer_cache_bytes(self, tokens: int) -> tuple[float, ...]:
        model, p = self.case.model, self.case.runtime.parallel
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            comps = layer_cache_components(
                model, layer, p, tokens, capacity_padded=False,
                interleave=self.case.runtime.kv_block_size,
                runtime=self.case.runtime, query_tokens=self.query_tokens)
            st = self.layer_stage[layer.index]
            result[st] = max(result[st], float(sum(comps.values())))
        return tuple(result)

    def host_dram_cache_bytes(self, tokens: int, *, padded=True) -> float:
        if not self.sparse_offload:
            return 0.0
        block_size = int(self.sparse_offload["block_size"])
        eff_tokens = math.ceil(tokens / block_size) * block_size if padded else tokens
        offload_layers = len(self.sparse_offload["offload_layers"])
        return float(eff_tokens * int(self.sparse_offload["kv_row_bytes"]) * offload_layers)

    def pd_handoff_volume_and_stage_hbm(self, tokens: int) -> tuple[float, tuple[float, ...]]:
        """Compute SfaRemoteD2HConnector wire volume (D2D index_k + 1x RD2H mla_kv) and stage HBM write."""
        p = self.case.runtime.parallel
        if not self.sparse_offload or self.sparse_offload["options"].get("keep_device_kv_cache"):
            local = self.cache_bytes(tokens, padded=False)
            return sum(local) * p.tp * p.pcp, local
        offload_set = set(self.sparse_offload["offload_layers"])
        topk_buf = int(self.sparse_offload["topk_buffer_size"])
        use_copy_sfa = bool(self.sparse_offload["options"].get("use_fused_copy_sfa", True))
        hot_tokens = min(tokens, topk_buf) if use_copy_sfa else 0
        stage_hbm = [0.0] * p.pp
        rd2h_bytes = 0.0
        for layer in self.case.model.layer_plans:
            components = layer_cache_components(
                self.case.model, layer, p, tokens, capacity_padded=False,
                interleave=self.case.runtime.kv_block_size,
                runtime=self.case.runtime, query_tokens=self.query_tokens,
            )
            st = self.layer_stage[layer.index]
            for name, val in components.items():
                if name == "mla_kv" and layer.index in offload_set:
                    rd2h_bytes += val
                    if hot_tokens > 0 and tokens > 0:
                        stage_hbm[st] += val * (hot_tokens / tokens)
                else:
                    stage_hbm[st] += val
        wire_volume = sum(stage_hbm) * p.tp * p.pcp + rd2h_bytes
        return wire_volume, tuple(stage_hbm)

    def _stage_dies(self, replica: int, stage: int):
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not 0 <= stage < p.pp:
            raise ValueError("memory_rank_plan_index_out_of_range")
        width = p.tp * p.pcp
        first = (replica * p.pp + stage) * width
        return self.case.machine.dies[first:first + width]

    def _rank_allocations(
        self, replica: int, stage: int, *, tokens: int | None, include_cache: bool,
        rank: int | None = None,
    ) -> tuple[Allocation, ...]:
        if rank is None:
            selected = self.selected_reserved[replica][stage]
            shared = self.shared_reserved[replica][stage]
            granted = self.granted[replica][stage]
            readers = max(self.reader_rank[replica][stage])
        else:
            selected = self.selected_rank[replica][stage][int(rank)]
            shared = self.shared_rank[replica][stage][int(rank)]
            granted = self.granted_rank[replica][stage][int(rank)]
            readers = self.reader_rank[replica][stage][int(rank)]
        allocations = [
            Allocation("weights", self.weights[stage], "weights",
                       note="persistent resident weight bytes"),
            Allocation("activation_workspace", self.workspace[stage], "workspace",
                       note="simultaneous implementation workspace"),
            Allocation("temporary_dequant", self.temporary_dequant[stage], "temporary",
                       note="declared transient representation"),
            Allocation("selected_plan", selected,
                       "workspace", note="selected implementation allocation"),
            Allocation("shared_pages", shared,
                       "shared_page", note="physical prefix/page owner and refs"),
            Allocation("grants", granted, "grant",
                       note="short-lived batch grants"),
            Allocation("grant_readers", readers, "grant_reader",
                       note="last-use readers of released grants"),
            Allocation("reserve.workspace", self.policy.workspace_bytes, "reserve"),
            Allocation("reserve.graph_pool", self.policy.graph_pool_bytes, "reserve"),
            Allocation("reserve.communication", self.policy.communication_buffer_bytes,
                       "reserve"),
            Allocation("reserve.runtime", self.policy.runtime_bytes, "reserve"),
        ]
        if include_cache:
            span = self.case.runtime.max_model_len if tokens is None else tokens
            if rank is None:
                _gross, resident, staging = self._kv_residency(span)
                resident_value = resident[stage]
                staging_value = staging[stage]
            else:
                gross = self._rank_cache_bytes(span, padded=True, rank=int(rank))
                layer_max = self._rank_max_layer_cache_bytes(span, int(rank))
                resident_value = (
                    float(gross[stage]) if self.kvc_hbm_ratio == 1.0
                    else float(gross[stage]) * self.kvc_hbm_ratio)
                staging_value = (
                    0.0 if self.kvc_hbm_ratio == 1.0
                    else float(layer_max[stage]) * (1.0 - self.kvc_hbm_ratio))
            if self.kvc_hbm_ratio == 1.0:
                allocations.append(Allocation(
                    "cache_allocation", resident_value, "cache",
                    note="physical pages at declared token span"))
            else:
                allocations.append(Allocation(
                    "kv_hbm_resident", resident_value, "cache",
                    note="declared KVC HBM residency ratio"))
                allocations.append(Allocation(
                    "kv_hbm_staging", staging_value, "staging",
                    note="required staging for the offloaded KV tier"))
        return tuple(allocations)

    def rank_capacity_plan(
        self, replica: int, stage: int, *, tokens: int | None = None,
        include_cache: bool = True, rank: int | None = None,
    ) -> AllocationPlan:
        """Explicit allocation for one replica/stage, optionally one Rank.

        Without ``rank`` this remains the conservative stage view (minimum
        mounted capacity in the TP/CP group).  ``rank_capacity_plans`` exposes
        every physical Rank's own capacity and its own T05 owner rows so
        heterogeneous dies and uneven token partitions are not represented by
        a single minimum or a stage average.
        """
        dies = self._stage_dies(replica, stage)
        if rank is not None:
            if not 0 <= int(rank) < len(dies):
                raise ValueError("memory_rank_index_out_of_range")
            mounted = dies[int(rank)].hbm_bytes * self.case.runtime.gpu_memory_utilization
            basis = f"physical_rank_{int(rank)}_allocations"
        else:
            mounted = min(
                die.hbm_bytes * self.case.runtime.gpu_memory_utilization
                for die in dies)
            basis = "per_rank_stage_physical_allocations"
        return AllocationPlan(
            self._rank_allocations(replica, stage, tokens=tokens,
                                   include_cache=include_cache, rank=rank),
            capacity_bytes=mounted, basis=basis)

    def rank_capacity_plans(
        self, replica: int, stage: int, *, tokens: int | None = None,
        include_cache: bool = True,
    ) -> tuple[AllocationPlan, ...]:
        """One capacity plan per physical Rank/die in this replica/stage."""
        return tuple(
            self.rank_capacity_plan(
                replica, stage, tokens=tokens, include_cache=include_cache, rank=index)
            for index in range(len(self._stage_dies(replica, stage)))
        )

    def allocation_report(self, tokens: int | None = None) -> dict:
        """Public explainable allocation view for one token span."""
        case, p = self.case, self.case.runtime.parallel
        tokens = case.runtime.max_model_len if tokens is None else int(tokens)
        if tokens < 0:
            raise ValueError("memory_allocation_tokens_requires_nonnegative")
        cache = self.cache_allocation(tokens)
        gross_kv, resident_kv, staging_kv = self._kv_residency(tokens)
        width = p.tp * p.pcp
        mounted = [
            [
                min(d.hbm_bytes * case.runtime.gpu_memory_utilization
                    for d in case.machine.dies[
                        (replica * p.pp + stage) * width:
                        (replica * p.pp + stage + 1) * width])
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        persistent = [
            [
                self.weights[stage] + self.policy.reserved + self.workspace[stage]
                + self.temporary_dequant[stage] + resident_kv[stage]
                + staging_kv[stage] + self.selected_reserved[replica][stage]
                + self.shared_reserved[replica][stage] + self.granted[replica][stage]
                + max(self.reader_rank[replica][stage])
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        plans = [
            [self.rank_capacity_plan(replica, stage, tokens=tokens).to_dict()
             for stage in range(p.pp)]
            for replica in range(p.dp)
        ]
        physical_plans = [
            [[plan.to_dict() for plan in self.rank_capacity_plans(
                replica, stage, tokens=tokens)]
             for stage in range(p.pp)]
            for replica in range(p.dp)
        ]
        return {
            "basis": "declared_physical_allocations_not_access_traffic",
            "sequence_tokens": tokens,
            "weights": {
                "total_bytes_per_stage_rank": list(self.weights),
                "payload_bytes_per_stage_rank": list(self.weights_payload),
                "quant_metadata_bytes_per_stage_rank": list(self.weights_quant_metadata),
                "gross_bytes_per_stage_rank": list(self.gross_weights),
                "offload_saved_bytes_per_stage_rank": list(self.expert_hbm_saved),
            },
            "reserve": {
                "workspace_bytes": self.policy.workspace_bytes,
                "graph_pool_bytes": self.policy.graph_pool_bytes,
                "communication_buffer_bytes": self.policy.communication_buffer_bytes,
                "runtime_bytes": self.policy.runtime_bytes,
                "total_bytes_per_rank": self.policy.reserved,
            },
            "workspace": {
                "base_bytes_per_stage_rank": list(self.workspace_base),
                "async_bytes_per_stage_rank": list(self.async_workspace),
                "expert_staging_bytes_per_stage_rank": list(self.workspace_expert_staging),
                "static_offload_bytes_per_stage_rank": list(self.workspace_static_offload),
                "concurrent_copies": self.concurrent_workspace_copies,
                "total_bytes_per_stage_rank": list(self.workspace),
            },
            "temporary_dequant": {
                "bytes_per_stage_rank": list(self.temporary_dequant),
                "basis": "explicitly declared transient representation",
            },
            "cache": cache,
            "grants_bytes_per_replica_stage": self.granted,
            "peak_combined_bytes_per_replica_stage": self.peak_combined,
            "budget_version": self.budget_version,
            "grant_bindings": [
                {"replica": replica, "name": name, **self._binding_report(binding)}
                for (replica, name), binding in sorted(self.grant_bindings.items())
            ],
            "onchip_effects": [
                {"replica": replica, "name": name, **binding}
                for (replica, name), binding in sorted(self.onchip_effects.items())
            ],
            "selected_plan_bytes_per_replica_stage": self.selected_reserved,
            "shared_page_bytes_per_replica_stage": self.shared_reserved,
            "kv_offload": {
                "kvc_hbm_ratio": self.kvc_hbm_ratio,
                "gross_kv_bytes_per_stage_rank": gross_kv,
                "hbm_resident_kv_bytes_per_stage_rank": resident_kv,
                "staging_bytes_per_stage_rank": staging_kv,
                "basis": "same_residency_and_staging_as_resize",
            },
            "mounted_capacity_bytes_per_replica_stage": mounted,
            "peak_resident_bytes_per_replica_stage": persistent,
            "headroom_bytes_per_replica_stage": [
                [cap - used for cap, used in zip(cap_row, used_row)]
                for cap_row, used_row in zip(mounted, persistent)
            ],
            "rank_plans": plans,
            "physical_rank_plans": physical_plans,
            "physical_rank_live_bytes_per_replica_stage": self.live_rank,
            "physical_rank_staging_bytes_per_replica_stage": self.staging_rank,
            "physical_rank_selected_bytes_per_replica_stage": self.selected_rank,
            "physical_rank_shared_bytes_per_replica_stage": self.shared_rank,
            "physical_rank_grant_bytes_per_replica_stage": self.granted_rank,
            "physical_rank_reader_bytes_per_replica_stage": self.reader_rank,
            "physical_rank_headroom_bytes_per_replica_stage": [
                [
                    [cap - used for cap, used in zip(
                        self.rank_capacity[replica][stage],
                        [a + b + c + d + e + f for a, b, c, d, e, f in zip(
                            self.live_rank[replica][stage],
                            self.staging_rank[replica][stage],
                            self.selected_rank[replica][stage],
                            self.shared_rank[replica][stage],
                            self.granted_rank[replica][stage],
                            self.reader_rank[replica][stage])])]
                    for stage in range(p.pp)
                ]
                for replica in range(p.dp)
            ],
            "physical_rank_cache_allocation": (
                [list(self.rank_cache_allocation(tokens))]
                if self.rank_token_owners is not None else []
            ),
            "rank_token_owners": list(self.rank_token_owners or ()),
            "fixed_plan": True,
            "fallback": "none",
        }

    def _grant_item(self, item, defaults: dict) -> dict | None:
        if isinstance(item, Mapping):
            raw = dict(item)
            name = raw.pop("name", None)
            amount = raw.pop("bytes", raw.pop("size", None))
        else:
            try:
                name, amount = item
            except (TypeError, ValueError):
                return None
            raw = {}
        name = str(name or "")
        if not name or isinstance(amount, bool) or not isinstance(amount, (int, float)):
            return None
        value = float(amount)
        if not math.isfinite(value) or value < 0.0:
            return None
        raw_ranges = raw.get("ranges", defaults["ranges"])
        try:
            ref_ranges = _half_open_ranges(raw_ranges, value)
        except ValueError:
            return None
        host = raw.get("host_bytes", defaults["host_bytes"])
        if isinstance(host, bool) or not isinstance(host, (int, float)):
            return None
        host = float(host)
        if not math.isfinite(host) or host < 0.0:
            return None
        rank = raw.get("rank", defaults["rank"])
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, int):
                return None
            if rank < 0:
                return None
        stage = int(raw.get("stage", defaults["stage"]))
        if not 0 <= stage < self.case.runtime.parallel.pp:
            return None
        result = {
            "name": name, "bytes": value,
            "owner": raw.get("owner", defaults["owner"]),
            "storage": raw.get("storage", defaults["storage"]),
            "tier": str(raw.get("tier", defaults["tier"]) or "hbm"),
            "rank": rank,
            "stage": stage,
            "generation": raw.get("generation", defaults["generation"]),
            "ranges": raw_ranges,
            "event": raw.get("event", defaults["event"]),
            "host_bytes": host,
            "physical_key": str(raw.get("physical_key", name) or name),
            "_ranges": ref_ranges,
        }
        return result

    def _grant_candidate(
        self, replica: int, grants, *, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
        host_bytes: float = 0.0,
    ) -> tuple[list[dict], dict]:
        """Resolve one batch into a pure decision plus the parsed holders."""
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        defaults = {
            "owner": owner, "storage": storage,
            "tier": "hbm", "rank": None, "stage": int(stage),
            "generation": generation,
            "ranges": tuple(ranges) if ranges is not None else (),
            "event": event, "host_bytes": host_bytes,
        }
        parsed = []
        names = set()
        for item in grants:
            spec = self._grant_item(item, defaults)
            if (spec is None or (replica, spec["name"]) in self.grants
                    or spec["name"] in names):
                return [], {
                    "ok": False, "reason": "invalid_or_duplicate",
                    "replica": replica,
                    "budget_version": self.budget_version,
                    "requested": (), "wakeable": False, "permanent": False,
                    "shortfall_bytes_per_stage_rank": [],
                    "releasable_bytes_per_stage_rank": [],
                    "host_shortfall_bytes": 0.0,
                    "host_releasable_bytes": 0.0,
                    "wake_events": (),
                }
            names.add(spec["name"])
            spec["replica"] = replica
            parsed.append(spec)
        if not parsed:
            return [], {
                "ok": True, "reason": "empty", "replica": replica,
                "budget_version": self.budget_version,
                "requested": (), "wakeable": False, "permanent": False,
                "shortfall_bytes_per_stage_rank": [
                    [0.0] * self.width for _ in range(p.pp)],
                "releasable_bytes_per_stage_rank": [
                    [0.0] * self.width for _ in range(p.pp)],
                "host_shortfall_bytes": 0.0,
                "host_releasable_bytes": 0.0,
                "wake_events": (),
            }
        new_holders = [
            ((replica, "grant", spec["name"]), (self._grant_ref(spec),))
            for spec in parsed
        ]
        hbm, host = self._state_usage(extra_holders=new_holders)
        current_hbm, current_host = self._state_usage()
        return parsed, self._grant_decision_payload(
            replica, hbm, host, current_hbm, current_host, parsed)

    def _grant_decision_payload(
        self, replica: int, hbm, host, current_hbm, current_host,
        parsed: list[dict], *, exclude_request: tuple[int, str] | None = None,
    ) -> dict:
        """Explain an atomic grant decision without mutating the ledger."""
        p = self.case.runtime.parallel
        ok = self._usage_budget_ok(hbm, host)
        shortfall = [[0.0] * self.width for _ in range(p.pp)]
        releasable = [[0.0] * self.width for _ in range(p.pp)]
        recoverable = True
        for stage in range(p.pp):
            capacities = self.rank_capacity[replica][stage]
            for rank in range(self.width):
                cap = capacities[rank]
                used = float(hbm[replica][stage][rank])
                base = max(0.0, float(current_hbm[replica][stage][rank]))
                if used > cap + max(1e-9, abs(cap) * 1e-12):
                    shortfall[stage][rank] = used - cap
                    releasable[stage][rank] = base
                    if shortfall[stage][rank] > base + max(
                            1e-9, abs(cap) * 1e-12):
                        recoverable = False
        host_cap = self.host_dram_capacity[replica]
        host_used = float(host[replica])
        host_base = max(0.0, float(current_host[replica]))
        host_shortfall = 0.0
        host_releasable = 0.0
        if host_used > host_cap + max(1e-9, abs(host_cap) * 1e-12):
            host_shortfall = host_used - host_cap
            host_releasable = host_base
            if host_shortfall > host_base + max(1e-9, abs(host_cap) * 1e-12):
                recoverable = False
        wakeable = (not ok) and recoverable
        permanent = (not ok) and not recoverable
        wake_events = []
        if not ok:
            for (rep, name), binding in sorted(self.grant_bindings.items()):
                if rep != replica or binding.get("event") is None:
                    continue
                wake_events.append({
                    "kind": "grant", "name": name,
                    "stage": binding.get("stage"),
                    "rank": binding.get("rank"),
                    "event": binding["event"],
                })
            for (rep, grant), readers in sorted(self.grant_readers.items()):
                if rep != replica:
                    continue
                for consumer, snapshot in sorted(readers.items()):
                    if snapshot.get("event") is None:
                        continue
                    wake_events.append({
                        "kind": "grant_reader", "grant": grant,
                        "consumer": consumer,
                        "event": snapshot["event"],
                    })
            for (rep, request), holds in sorted(
                    self.async_consumer_holds.items()):
                if rep != replica or (rep, request) == exclude_request:
                    continue
                for consumer, snapshot in sorted(holds.items()):
                    binding = dict(snapshot.binding)
                    if binding.get("event") is None:
                        continue
                    wake_events.append({
                        "kind": "async_consumer", "request": request,
                        "consumer": consumer, "event": binding["event"],
                    })
        return {
            "ok": ok,
            "reason": ("admitted" if ok else
                       "waitable_release" if wakeable else
                       "permanent_budget"),
            "replica": replica,
            "budget_version": self.budget_version,
            "requested": tuple(spec["name"] for spec in parsed),
            "wakeable": wakeable,
            "permanent": permanent,
            "shortfall_bytes_per_stage_rank": shortfall,
            "releasable_bytes_per_stage_rank": releasable,
            "host_shortfall_bytes": host_shortfall,
            "host_releasable_bytes": host_releasable,
            "wake_events": tuple(wake_events),
        }

    def grant_decision(
        self, replica: int, grants, *, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
        host_bytes: float = 0.0,
    ) -> dict:
        """Pure admission decision for ``grant_batch``.

        The result separates a refusal that a real holder release can repair
        from a permanent demand/available deficit.  Known holder Events are
        returned so a scheduler can wait on the actual completion instead of a
        timer.  This method never changes refs, peaks, version or callbacks.
        """
        _parsed, decision = self._grant_candidate(
            replica, grants, stage=stage, owner=owner, storage=storage,
            ranges=ranges, generation=generation, event=event,
            host_bytes=host_bytes)
        return decision

    def grant_batch(
        self, replica: int, grants, *, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
        host_bytes: float = 0.0,
    ) -> bool:
        """Atomically grant per-item physical references in one budget.

        Each item may be the legacy ``(name, bytes)`` tuple or a mapping with
        ``name/bytes/owner/storage/tier/rank/stage/generation/ranges/event/
        host_bytes/physical_key``.  Uniform keyword defaults preserve old
        calls.  All holders in the call are resolved into one candidate:
        same-backing ranges are unioned, distinct live generations or keys are
        charged separately, and every physical rank/stage plus host is checked
        before one commit.  Failure changes no ledger, peak, version or
        callback.  ``grant_decision`` exposes the same candidate decision.
        """
        p = self.case.runtime.parallel
        parsed, decision = self._grant_candidate(
            replica, grants, stage=stage, owner=owner, storage=storage,
            ranges=ranges, generation=generation, event=event,
            host_bytes=host_bytes)
        if not decision["ok"]:
            return False
        typed: dict[tuple[int, str], tuple[AllocationKey, ...]] = {}
        if self.ledger_domain is not None:
            from .runtime.ledger import Ledger as _Ledger
            plan = []
            for spec in parsed:
                bindings = self._typed_spec_bindings(replica, spec)
                if bindings:
                    plan.append((spec["name"], bindings))
            flat = tuple(
                binding for _name, bindings in plan for binding in bindings)
            if flat:
                grant = _Ledger.try_acquire(
                    self.runtime_state, flat,
                    holder=self._ledger_occurrence(
                        replica, int(parsed[0]["stage"]),
                        int(parsed[0].get("rank") or 0)),
                    at=self._ledger_boundary())
                if not grant.granted:
                    return False
                granted = set(grant.allocations)
                for name, bindings in plan:
                    typed[(replica, name)] = tuple(
                        binding.key for binding in bindings
                        if binding.key in granted)
        for spec in parsed:
            spec_ref = self._grant_ref(spec)
            self._set_ref((replica, "grant", spec["name"]), (spec_ref,))
            self.grants[(replica, spec["name"])] = tuple(
                spec["bytes"] if index == spec["stage"] else 0.0
                for index in range(p.pp))
            self.grant_hosts[(replica, spec["name"])] = spec["host_bytes"]
            self.grant_bindings[(replica, spec["name"])] = {
                "owner": spec["owner"], "storage": spec["storage"],
                "tier": spec["tier"], "rank": spec["rank"],
                "stage": spec["stage"], "ranges": spec["ranges"],
                "generation": spec["generation"], "event": spec["event"],
                "host_bytes": spec["host_bytes"],
                "physical_key": spec["physical_key"],
            }
            self.grant_records[(replica, spec["name"])] = dict(spec)
        if parsed:
            self._sync_usage()
            self.budget_version += 1
        self._typed_grants.update(typed)
        return True

    def _remove_grant(
        self, replica: int, name: str, binding: dict, *,
        sync: bool = True,
    ) -> float:
        typed = self._typed_grants.pop((replica, str(name)), ())
        if typed and self.ledger_domain is not None:
            from .runtime.ledger import Ledger as _Ledger
            for allocation in typed:
                _Ledger.release(self.runtime_state, allocation,
                                after=self._ledger_boundary())
        before = self._replica_occupied(replica)
        before_host = self.host_occupied_bytes(replica)
        self._set_ref((replica, "grant", name), ())
        self.grant_records.pop((replica, name), None)
        self.grants.pop((replica, name), None)
        self.grant_bindings.pop((replica, name), None)
        self.grant_hosts.pop((replica, name), None)
        after = self._replica_occupied(replica)
        after_host = self.host_occupied_bytes(replica)
        released = max(0.0, before - after)
        released_total = released + max(0.0, before_host - after_host)
        if sync:
            self._sync_usage()
            self.budget_version += 1
        else:
            self._rebuild_grant_backings()
        self.release_events.append({
            "kind": "grant", "replica": replica, "name": name,
            **binding, "released_bytes": released_total,
        })
        return released

    def release_grant(
        self, replica: int, name: str, *, generation: int | None = None,
        event=None, rank: int | None = None,
    ) -> bool:
        key = (replica, str(name))
        record = self.grant_records.get(key)
        if record is None:
            return False
        if generation is not None:
            record_generation = record.get("generation")
            if (record_generation is None
                    or int(generation) != int(record_generation)):
                return False
        if rank is not None:
            record_rank = record.get("rank")
            if record_rank is None or int(rank) != int(record_rank):
                return False
        if (event is not None and event != record.get("event")):
            return False
        self._remove_grant(replica, str(name), self.grant_bindings.get(key, {}))
        return True

    def release_grants(self, replica: int, names) -> float:
        """Release named grants; returns the summed physical released bytes."""
        released = 0.0
        changed = False
        for name in names:
            key = (replica, str(name))
            if key not in self.grant_records:
                continue
            released += self._remove_grant(
                replica, str(name), self.grant_bindings.get(key, {}),
                sync=False)
            changed = True
        if changed:
            self._sync_usage()
            self.budget_version += 1
        return released

    def outstanding_holders(self) -> dict:
        """Known live grants/selected/shared/async holders for T14 wakeup."""
        grants = []
        for (replica, name), record in sorted(self.grant_records.items()):
            grants.append({"replica": replica, "name": name,
                           **self.grant_bindings.get((replica, name), {})})
        return {
            "budget_version": self.budget_version,
            "grants": grants,
            "selected": [
                {"replica": replica, "stage": stage,
                 "bytes": self.selected_reserved[replica][stage]}
                for replica in range(self.case.runtime.parallel.dp)
                for stage in range(self.case.runtime.parallel.pp)
                if self.selected_reserved[replica][stage]
            ],
            "shared": [
                {"replica": replica, "owner": owner, **record}
                for (replica, owner), record in sorted(self.shared_pages.items())
            ],
            "async_consumers": [
                {"replica": key[0], "request": key[1], "consumer": consumer,
                 **snapshot.__dict__}
                for key, holds in sorted(self.async_consumer_holds.items())
                for consumer, snapshot in sorted(holds.items())
            ],
            "grant_readers": [
                {"replica": key[0], "grant": key[1], "consumer": consumer,
                 "storage": snapshot.get("storage"),
                 "generation": snapshot.get("generation"),
                 "event": self._binding_report(
                     {"event": snapshot.get("event")}).get("event"),
                 "ranges": snapshot.get("ranges")}
                for key, readers in sorted(self.grant_readers.items())
                for consumer, snapshot in sorted(readers.items())
            ],
        }

    def drain_release_events(self) -> tuple[dict, ...]:
        """Only real byte releases can wake another waiter."""
        events = tuple(
            event for event in self.release_events
            if float(event.get("released_bytes", 0.0)) > 0.0)
        self.release_events.clear()
        return events

    def live_grants(self, replica: int, stage: int) -> float:
        return float(self.granted[replica][stage])

    def _install_selected(
        self, replica: int, stage: int, items: dict[str, Allocation],
    ) -> bool:
        plan = AllocationPlan(tuple(items.values()))
        # Resolve once to preserve alias-cycle/pool-overcommit validation; the
        # unified physical refs below decide the actual byte union.
        plan.resolve()
        refs = self._selected_refs(replica, stage, items)
        old_names = set(self.selected_items.get((replica, stage), {}))
        drop = {(replica, "selected", name) for name in old_names}
        extra = [
            ((replica, "selected", name), (ref,))
            for name, ref in refs.items()
        ]
        hbm, host = self._state_usage(
            extra_holders=extra, drop_holders=drop)
        if not self._usage_budget_ok(hbm, host):
            return False
        for name in old_names - set(refs):
            self._set_ref((replica, "selected", name), ())
        for name, ref in refs.items():
            self._set_ref((replica, "selected", name), (ref,))
        self.selected_items[(replica, stage)] = dict(items)
        self.selected_plans[(replica, stage)] = plan
        self.selected_physical = {
            key: value for key, value in self.selected_physical.items()
            if not (key[0] == replica and key[1] == stage)
        }
        for name, ref in refs.items():
            self.selected_physical[(replica, stage, ref.physical_key)] = (
                _union_bytes(ref.ranges))
        self._sync_usage()
        self.budget_version += 1
        for name in list(self.selected_released):
            if (name[0] == replica and name[1] == stage
                    and name[2] not in items):
                self.selected_released.discard(name)
        return True

    def attach_allocation_plan(
        self, plan: AllocationPlan, *, replica: int = 0, stage: int = 0,
        replace: bool = False,
    ) -> bool:
        """Attach/replace a selected implementation's alias/pool allocations.

        ``replace=False`` is an incremental acquire: existing live acquisitions
        remain and duplicate names are rejected.  ``replace=True`` treats the
        supplied plan as a complete replacement and commits only its peak
        difference.  All checks precede the single commit.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not 0 <= stage < p.pp:
            raise ValueError("memory_selected_plan_index_out_of_range")
        incoming = {item.name: item for item in plan.allocations}
        if len(incoming) != len(plan.allocations):
            return False
        if replace:
            items = incoming
        else:
            items = dict(self.selected_items.get((replica, stage), {}))
            for name, item in incoming.items():
                if name in items:
                    return False
                items[name] = item
        if not self._install_selected(replica, stage, items):
            return False
        for name in items:
            self.selected_released.discard((replica, stage, name))
        return True

    @staticmethod
    def _binding_report(binding: dict) -> dict:
        row = dict(binding)
        event = row.get("event")
        if event is not None and not isinstance(event, (str, int, float, bool, type(None))):
            row["event"] = getattr(event, "sequence", repr(event))
        return row

    @staticmethod
    def _effect_is_onchip(effect) -> bool:
        pool = str(getattr(effect, "pool", "")).strip().lower()
        return pool in {"l0a", "l0b", "l0c", "l1", "l2", "ub", "smp", "aic",
                        "aiv", "onchip", "local"}

    def attach_allocation_effects(
        self, effects, *, replica: int = 0, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
    ) -> bool:
        """Apply a selected program's acquire/release list atomically.

        The name, size, unit, pool, lifetime and condition are retained from
        the existing typed effect.  This ledger only decides the byte budget;
        T04 remains the owner of the actual pool mechanics.  On-chip effects
        are staged in the same transaction, so a rejected attach changes no
        effect either.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not 0 <= stage < p.pp:
            raise ValueError("memory_selected_plan_index_out_of_range")
        items = dict(self.selected_items.get((replica, stage), {}))
        bindings = {}
        staged_onchip = dict(self.onchip_effects)
        for effect in effects:
            name = f"selected:{getattr(effect, 'name', '')}"
            action = getattr(effect, "action", "")
            if action == "release":
                staged_onchip.pop((replica, name), None)
                items = self._release_selected_item(replica, stage, items, name)
                continue
            if action != "acquire":
                return False
            if getattr(effect, "unit", "bytes") != "bytes":
                return False
            if self._effect_is_onchip(effect):
                staged_onchip[(replica, name)] = {
                    "pool": getattr(effect, "pool", ""),
                    "size": float(getattr(effect, "size", 0.0)),
                    "lifetime": getattr(effect, "lifetime", ""),
                    "owner": owner, "storage": storage,
                    "ranges": tuple(ranges) if ranges is not None else (),
                    "generation": generation, "event": event,
                }
                continue
            if name in items:
                return False
            items[name] = Allocation(
                name, float(getattr(effect, "size", 0.0)), "workspace",
                note=f"pool={getattr(effect, 'pool', '')};"
                     f"lifetime={getattr(effect, 'lifetime', '')};"
                     f"condition={getattr(effect, 'condition', '')}",
                physical_key=getattr(effect, "physical_key", None),
                rank=getattr(effect, "rank", None),
                storage=storage,
                tier=getattr(effect, "tier", "hbm"),
                generation=generation,
                ranges=ranges,
            )
            bindings[name] = {
                "owner": owner, "storage": storage,
                "ranges": tuple(ranges) if ranges is not None else (),
                "generation": generation, "event": event,
                "pool": getattr(effect, "pool", ""),
                "lifetime": getattr(effect, "lifetime", ""),
                "condition": getattr(effect, "condition", ""),
            }
        if not self._install_selected(replica, stage, items):
            return False
        self.onchip_effects = staged_onchip
        for name, binding in bindings.items():
            if name in items:
                self.effect_bindings[(replica, name)] = binding
                self.selected_released.discard((replica, stage, name))
        for name in list(self.effect_bindings):
            if name[0] == replica and name[1] not in items:
                self.effect_bindings.pop(name, None)
        return True

    def _release_selected_item(
        self, replica: int, stage: int, items: dict[str, Allocation], name: str,
    ) -> dict[str, Allocation]:
        """Release a name while aliases keep their physical backing live."""
        if name not in items:
            return items
        target = items[name]
        replaced = dict(items)
        aliases = [value.name for value in replaced.values()
                   if value.alias_of == name and value.name != name]
        if aliases:
            # The alias is the remaining reader of the backing allocation.
            self.selected_released.add((replica, stage, name))
            return replaced
        del replaced[name]
        if target.alias_of:
            root = target.alias_of
            root_released = (replica, stage, root) in self.selected_released
            other_aliases = [value for value in replaced.values()
                             if value.alias_of == root]
            if root_released and not other_aliases:
                replaced.pop(root, None)
        return replaced

    def release_selected_plan(
        self, replica: int = 0, stage: int = 0, name: str | None = None,
    ) -> bool:
        key = (replica, stage)
        items = dict(self.selected_items.get(key, {}))
        if not items:
            return False
        if name is None:
            items = {}
            self.selected_released = {
                record for record in self.selected_released
                if not (record[0] == replica and record[1] == stage)
            }
        else:
            if name not in items:
                return False
            items = self._release_selected_item(replica, stage, items, name)
        before = self._replica_occupied(replica)
        if not self._install_selected(replica, stage, items):
            return False
        released = max(0.0, before - self._replica_occupied(replica))
        if name is not None:
            self.release_events.append({
                "kind": "selected", "replica": replica, "stage": stage,
                "name": name, "released_bytes": released,
            })
        for binding_key in list(self.effect_bindings):
            if binding_key[0] == replica and binding_key[1] not in items:
                self.effect_bindings.pop(binding_key, None)
        return True

    def release_selected_allocation(
        self, name: str, *, replica: int = 0, stage: int = 0,
    ) -> bool:
        return self.release_selected_plan(replica, stage, name=name)

    def retain_shared_pages(
        self, replica: int, owner: str, stages, *, version: int = 0,
        physical_key: str | None = None, storage: str | None = None,
        rank: int | None = None, tier: str = "hbm", stage: int = 0,
    ) -> bool:
        """Retain a physical shared-page owner used by another request.

        The first retain charges the page range; subsequent readers only bump
        the reference count.  ``physical_key`` identifies the underlying page
        owner, so two request/string owners pointing at the same physical range
        are charged once and keep independent references.  SERVICE-CORE
        supplies owner, range and version; this ledger never derives them from
        a logical prefix length.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not owner:
            raise ValueError("memory_shared_pages_index_invalid")
        ranges = tuple(float(value) for value in stages)
        if len(ranges) != p.pp or any(value < 0.0 for value in ranges):
            raise ValueError("memory_shared_pages_range_invalid")
        physical = str(physical_key or owner)
        key = (replica, physical)
        existing = self.shared_pages.get(key)
        if existing is not None:
            if ranges != existing["stages"]:
                return False
            existing["owners"][str(owner)] = existing["owners"].get(str(owner), 0) + 1
            existing["refs"] += 1
            existing["version"] = max(existing["version"], int(version))
            self.budget_version += 1
            return True
        record = {
            "stages": ranges, "version": int(version), "refs": 1,
            "owners": {str(owner): 1}, "storage": storage,
            "rank": rank, "tier": tier, "charged": False,
        }
        holder = (replica, "shared", physical)
        refs = self._shared_ref(replica, physical, record)
        hbm, host = self._state_usage(extra_holders=[(holder, refs)])
        if not self._usage_budget_ok(hbm, host):
            return False
        self.shared_pages[key] = record
        self._set_ref(holder, refs)
        self._sync_usage()
        self.budget_version += 1
        return True

    def release_shared_pages(self, replica: int, owner: str) -> bool:
        released = False
        for key, existing in list(self.shared_pages.items()):
            if key[0] != replica or str(owner) not in existing.get("owners", {}):
                continue
            released = True
            existing["owners"][str(owner)] -= 1
            if existing["owners"][str(owner)] <= 0:
                del existing["owners"][str(owner)]
            existing["refs"] -= 1
            if existing["refs"] > ZERO:
                self.budget_version += 1
                continue
            before = self._replica_occupied(replica)
            self._set_ref((replica, "shared", key[1]), ())
            del self.shared_pages[key]
            self._sync_usage()
            self.release_events.append({
                "kind": "shared", "replica": replica, "owner": owner,
                "physical_key": key[1], **existing,
                "released_bytes": max(0.0, before - self._replica_occupied(replica)),
            })
            self.budget_version += 1
        return released

    def _update_combined_peak(self, replica: int, stage: int) -> None:
        """Compatibility wrapper: peaks are committed-state only."""
        self._sync_usage()

    def _replicate_ranks(self, values) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(values[stage]) for _ in range(self.width))
            for stage in range(self.case.runtime.parallel.pp)
        )

    def _physical_union(
        self, key: tuple[int, str], demand: tuple[float, ...], host_demand: float,
        staging_demand: tuple[float, ...],
        demand_ranks=None, staging_ranks=None,
    ):
        """Physical backing/staging union across ranks, host and stages."""
        p = self.case.runtime.parallel
        if demand_ranks is None:
            demand_ranks = self._replicate_ranks(demand)
        if staging_ranks is None:
            staging_ranks = self._replicate_ranks(staging_demand)
        holds = self.async_consumer_holds.get(key)
        physical_ranks = [
            [float(demand_ranks[stage][rank]) for rank in range(self.width)]
            for stage in range(p.pp)
        ]
        physical_staging_ranks = [
            [float(staging_ranks[stage][rank]) for rank in range(self.width)]
            for stage in range(p.pp)
        ]
        hosts = [float(host_demand)]
        if holds:
            for snapshot in holds.values():
                rank_stages = snapshot.rank_stages or self._replicate_ranks(
                    snapshot.stages)
                rank_staging = snapshot.rank_staging or self._replicate_ranks(
                    snapshot.staging)
                for stage in range(p.pp):
                    for rank in range(self.width):
                        physical_ranks[stage][rank] = max(
                            physical_ranks[stage][rank],
                            float(rank_stages[stage][rank]))
                        physical_staging_ranks[stage][rank] = max(
                            physical_staging_ranks[stage][rank],
                            float(rank_staging[stage][rank]))
                hosts.append(float(snapshot.host_dram))
        physical = tuple(max(row) for row in physical_ranks)
        host_physical = max(hosts)
        physical_staging = tuple(max(row) for row in physical_staging_ranks)
        return (physical, host_physical, physical_staging,
                tuple(tuple(row) for row in physical_ranks),
                tuple(tuple(row) for row in physical_staging_ranks))

    def _set_physical(
        self, replica: int, key: tuple[int, str], physical: tuple[float, ...],
        host_physical: float, staging_physical: tuple[float, ...] | None = None,
        *, physical_ranks=None, staging_ranks=None,
    ) -> None:
        changed = self.owners.get(key) != tuple(physical)
        self.owners[key] = tuple(physical)
        self.owner_ranks[key] = (
            tuple(tuple(row) for row in physical_ranks)
            if physical_ranks is not None else self._replicate_ranks(physical))
        host_before = self.host_dram_owners.get(key, 0.0)
        self.host_dram_owners[key] = float(host_physical)
        changed = changed or float(host_before) != float(host_physical)
        if staging_physical is not None:
            before = self.staging_owners.get(key)
            self.staging_owners[key] = tuple(staging_physical)
            self.staging_owner_ranks[key] = (
                tuple(tuple(row) for row in staging_ranks)
                if staging_ranks is not None else
                self._replicate_ranks(staging_physical))
            changed = changed or before != tuple(staging_physical)
        self._sync_usage()

    def _resident_admission(
        self, replica: int, request: str, tokens: int,
    ) -> dict:
        key = replica, request
        resident_ranks, staging_ranks = self._kv_residency_ranks(tokens)
        demand = tuple(max(row) for row in resident_ranks)
        demand_staging = tuple(max(row) for row in staging_ranks)
        host_demand = self.host_dram_cache_bytes(tokens) if self.sparse_offload else 0.0
        (physical, host_physical, physical_staging,
         physical_ranks, physical_staging_ranks) = self._physical_union(
            key, demand, host_demand, demand_staging,
            demand_ranks=resident_ranks, staging_ranks=staging_ranks)
        p = self.case.runtime.parallel
        request_usage = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        staging_usage = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        for stage in range(p.pp):
            for rank in range(self.width):
                request_usage[replica][stage][rank] = float(
                    physical_ranks[stage][rank])
                staging_usage[replica][stage][rank] = float(
                    physical_staging_ranks[stage][rank])
        host_request = [0.0] * p.dp
        host_request[replica] = float(host_physical)
        hbm, host = self._state_usage(
            exclude_request=key, request_usage=request_usage,
            staging_usage=staging_usage, host_request=host_request)
        return {
            "demand": demand, "physical": physical,
            "physical_ranks": physical_ranks,
            "demand_ranks": tuple(tuple(row) for row in resident_ranks),
            "demand_staging": demand_staging,
            "demand_staging_ranks": tuple(tuple(row) for row in staging_ranks),
            "physical_staging": physical_staging,
            "physical_staging_ranks": physical_staging_ranks,
            "host_demand": float(host_demand),
            "host_physical": float(host_physical),
            "candidate_hbm": hbm, "candidate_host": host,
            "ok": self._usage_budget_ok(hbm, host),
        }

    def admissible_tokens(
        self, replica: int, request: str, tokens: int, *, granularity: int = 1,
    ) -> int:
        """Largest requested KV span that fits without changing the ledger.

        This is a capacity feedback query, not a dispatch decision.  The
        Scheduler/Engine owner decides whether to use a smaller chunk and must
        record the actual chosen span.  A fixed infeasible plan is never
        rewritten to a different dtype/algorithm.  HBM and host DRAM use the
        same candidate calculation as ``resize``, including every physical
        rank's own budget.
        """
        if not 0 <= replica < self.case.runtime.parallel.dp:
            raise ValueError("memory_admission_replica_out_of_range")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("memory_admission_tokens_requires_nonnegative_integer")
        if isinstance(granularity, bool) or not isinstance(granularity, int) or granularity < 1:
            raise ValueError("memory_admission_granularity_requires_positive_integer")
        lo, hi, best = 0, int(tokens), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid % granularity:
                mid -= mid % granularity
            if mid < lo:
                break
            candidate = self._resident_admission(replica, request, mid)
            if candidate["ok"]:
                best = max(best, mid)
                lo = mid + granularity
            else:
                hi = mid - granularity
        return int(best)

    def admission_decision(
        self, replica: int, request: str, tokens: int,
    ) -> dict:
        """Pure request-admission decision with the same refusal vocabulary.

        ``admissible_tokens`` answers "how much fits"; this answer separates a
        shortfall that another live holder's real release can repair from a
        demand that can never fit in this replica.  It consumes the identical
        ``_resident_admission`` candidate used by ``resize``.
        """
        if not 0 <= replica < self.case.runtime.parallel.dp:
            raise ValueError("memory_admission_replica_out_of_range")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("memory_admission_tokens_requires_nonnegative_integer")
        key = (replica, request)
        candidate = self._resident_admission(replica, request, tokens)
        base_hbm, base_host = self._state_usage(exclude_request=key)
        decision = self._grant_decision_payload(
            replica, candidate["candidate_hbm"], candidate["candidate_host"],
            base_hbm, base_host, [], exclude_request=key)
        decision.update({
            "request": request, "tokens": int(tokens),
            "physical_ranks": candidate["physical_ranks"],
        })
        return decision

    def resize(self, replica: int, request: str, tokens: int) -> bool:
        key = replica, request
        candidate = self._resident_admission(replica, request, tokens)
        if not candidate["ok"]:
            return False
        changed = (
            self.owners.get(key) != candidate["physical"]
            or self.owner_ranks.get(key) != candidate["physical_ranks"]
            or self.logical_owners.get(key) != candidate["demand"]
            or self.host_dram_owners.get(key) != candidate["host_physical"]
            or self.staging_owners.get(key) != candidate["physical_staging"]
        )
        self._set_physical(
            replica, key, candidate["physical"], candidate["host_physical"],
            candidate["physical_staging"],
            physical_ranks=candidate["physical_ranks"],
            staging_ranks=candidate["physical_staging_ranks"])
        self.logical_owners[key] = candidate["demand"]
        self.logical_owner_ranks[key] = candidate["demand_ranks"]
        self.logical_host_owners[key] = candidate["host_demand"]
        self.logical_staging_owners[key] = candidate["demand_staging"]
        self.logical_staging_rank[key] = candidate["demand_staging_ranks"]
        self.request_versions[key] = self.request_versions.get(key, 0) + 1
        if changed:
            self.budget_version += 1
        return True

    def _free_request(self, replica: int, request: str) -> None:
        """Remove the request's physical cache once it is safe to do so."""
        key = (replica, request)
        before = self._replica_occupied(replica)
        self.async_consumer_holds.pop(key, None)
        self.logical_owners.pop(key, None)
        self.logical_owner_ranks.pop(key, None)
        self.logical_host_owners.pop(key, None)
        self.logical_staging_owners.pop(key, None)
        self.logical_staging_rank.pop(key, None)
        self.staging_owners.pop(key, None)
        self.staging_owner_ranks.pop(key, None)
        self.request_versions.pop(key, None)
        self.retired_requests.discard(key)
        self.delayed_owners.pop(key, None)
        sizes = self.owners.pop(key, (0.0,) * len(self.weights))
        self.owner_ranks.pop(key, None)
        self.host_dram_owners.pop(key, None)
        self._sync_usage()
        released = max(0.0, before - self._replica_occupied(replica))
        if any(size > 0.0 for size in sizes):
            self.budget_version += 1
            self.release_events.append({
                "kind": "request", "replica": replica, "request": request,
                "stages": sizes, "released_bytes": released,
            })

    def release(self, replica: int, request: str) -> None:
        """Retire a request without clearing unfinished async consumers.

        A live request's current consumer completing does not retire the
        request; a retired request's final consumer is the actual free point.
        Retirement immediately recomputes the physical union of the remaining
        readers, so an unreferenced old tail is not retained.
        """
        key = (replica, request)
        if key in self.async_consumer_holds and self.async_consumer_holds[key]:
            self.retired_requests.add(key)
            self.logical_owners[key] = (0.0,) * len(self.weights)
            self.logical_owner_ranks[key] = tuple(
                (0.0,) * self.width for _ in range(self.case.runtime.parallel.pp))
            self.logical_host_owners[key] = 0.0
            self.logical_staging_owners[key] = (0.0,) * len(self.weights)
            self.logical_staging_rank[key] = tuple(
                (0.0,) * self.width for _ in range(self.case.runtime.parallel.pp))
            self.delayed_owners.pop(key, None)
            zeros = (0.0,) * len(self.weights)
            (physical, host_physical, physical_staging,
             physical_ranks, staging_ranks) = self._physical_union(
                key, zeros, 0.0, zeros)
            self._set_physical(
                replica, key, physical, host_physical, physical_staging,
                physical_ranks=physical_ranks,
                staging_ranks=staging_ranks)
            self.budget_version += 1
            return
        self._free_request(replica, request)

    def force_release(self, replica: int, request: str) -> None:
        """Cancel/rollback escape hatch: drop holds and free immediately."""
        key = (replica, request)
        self.async_consumer_holds.pop(key, None)
        self.retired_requests.discard(key)
        self._free_request(replica, request)

    def _hold_binding(self, storage, ranges, generation, event) -> tuple:
        return tuple(sorted({
            "storage": storage, "generation": generation, "event": event,
            "ranges": tuple(ranges) if ranges is not None else (),
        }.items(), key=lambda item: item[0]))

    def _retain_typed_readers(self, replica: int, grant: str, consumer: str,
                              event):
        """Register the canonical reader lease for one named grant reader.

        Returns an empty tuple when no runtime ledger is adopted, or the real
        ReaderKey tuple.  Returns ``None`` if a lease cannot be registered so
        the caller leaves its own holder projection unchanged.
        """
        if self.ledger_domain is None:
            return ()
        keys = self._typed_grants.get((replica, grant), ())
        if not keys:
            return ()
        from .runtime.ledger import Ledger as _Ledger, LedgerError
        reader = event if event is not None else self._ledger_boundary()
        retained = []
        try:
            for allocation in keys:
                retained.append(_Ledger.retain_reader(
                    self.runtime_state, allocation, reader=reader))
        except LedgerError:
            for reader_key in retained:
                try:
                    _Ledger.retire_reader(self.runtime_state, reader_key)
                except LedgerError:
                    pass
            return None
        return tuple(retained)

    def hold_for_async_consumer(
        self, replica: int, request: str, consumer: str, *,
        storage: str | None = None, ranges=None, generation: int | None = None,
        event=None, grant: str | None = None, rank: int | None = None,
    ) -> bool:
        """Register one async consumer that still reads a real holder.

        With ``grant`` the reader references that grant's physical backing, so
        releasing the grant handle does not free bytes before this reader's
        last-use event.  Without it the legacy request-cache hold is unchanged.
        Repeating the same handle/binding is idempotent; a different binding
        under the same consumer name is refused and leaves the first reader
        untouched.
        """
        if not consumer:
            raise ValueError("memory_async_consumer_name_required")
        if grant is not None:
            record = self.grant_records.get((replica, str(grant)))
            if record is None:
                return False
            if rank is not None:
                record_rank = record.get("rank")
                if record_rank is None or int(rank) != int(record_rank):
                    return False
            if generation is not None and record.get("generation") is not None \
                    and int(generation) != int(record.get("generation")):
                return False
            ref_ranges = (
                _half_open_ranges(ranges, record["bytes"])
                if ranges is not None else record.get("_ranges"))
            ref = self._make_ref(
                replica, rank=record.get("rank"), stage=record["stage"],
                storage=record.get("storage"), tier=record.get("tier", "hbm"),
                physical_key=record.get("physical_key"),
                generation=record.get("generation"),
                size=record["bytes"], ranges=ref_ranges,
                host_bytes=record.get("host_bytes", 0.0))
            readers = self.grant_readers.get((replica, str(grant)))
            existing = readers.get(consumer) if readers else None
            if existing is not None:
                return (
                    existing.get("ref") == ref
                    and existing.get("generation") == generation
                    and existing.get("event") == event
                    and existing.get("storage") == storage
                    and tuple(existing.get("ranges") or ())
                    == tuple(ref_ranges or ()))
            holder = (replica, "grant_reader", f"{grant}::{consumer}")
            hbm, host = self._state_usage(extra_holders=[(holder, (ref,))])
            if not self._usage_budget_ok(hbm, host):
                return False
            typed_readers = self._retain_typed_readers(
                replica, str(grant), consumer, event)
            if typed_readers is None:
                return False
            self._set_ref(holder, (ref,))
            self.grant_readers.setdefault(
                (replica, str(grant)), {})[consumer] = {
                "ref": ref, "generation": generation, "event": event,
                "storage": storage, "ranges": tuple(ref_ranges or ()),
                "rank": ref.rank, "readers": typed_readers,
            }
            self._sync_usage()
            self.budget_version += 1
            return True
        key = (replica, request)
        if key not in self.owners:
            return False
        stages = self.logical_owners.get(
            key, self.owners.get(key, (0.0,) * len(self.weights)))
        rank_stages = self.logical_owner_ranks.get(key)
        if rank_stages is None:
            rank_stages = self._replicate_ranks(stages)
        host = self.logical_host_owners.get(
            key, self.host_dram_owners.get(key, 0.0))
        staging = self.logical_staging_owners.get(
            key, (0.0,) * len(self.weights))
        rank_staging = self.logical_staging_rank.get(key)
        if rank_staging is None:
            rank_staging = self._replicate_ranks(staging)
        binding = self._hold_binding(storage, ranges, generation, event)
        snapshot = _HoldSnapshot(
            tuple(stages), float(host), self.request_versions.get(key, 0),
            tuple(staging), binding,
            tuple(tuple(row) for row in rank_stages),
            tuple(tuple(row) for row in rank_staging),
        )
        holds = self.async_consumer_holds.get(key)
        existing = holds.get(consumer) if holds else None
        if existing is not None:
            return existing == snapshot
        self.async_consumer_holds.setdefault(key, {})[consumer] = snapshot
        self.budget_version += 1
        return True

    def release_async_consumer(
        self, replica: int, request: str, consumer: str, *,
        generation: int | None = None, event=None, rank: int | None = None,
    ) -> bool:
        """Complete one consumer; only the real last reference frees bytes."""
        key = (replica, request)
        holds = self.async_consumer_holds.get(key)
        if holds and consumer in holds:
            if rank is not None:
                # A request-cache hold has no single physical rank; a
                # rank-qualified completion belongs to an explicit reader.
                return False
            snapshot = holds[consumer]
            binding = dict(snapshot.binding)
            bound_generation = binding.get("generation")
            if (generation is not None and bound_generation is not None
                    and int(generation) != int(bound_generation)):
                return False
            bound_event = binding.get("event")
            if event is not None and event != bound_event:
                return False
            before = self._replica_occupied(replica)
            del holds[consumer]
            if not holds:
                del self.async_consumer_holds[key]
            if key in self.retired_requests and not holds:
                self._free_request(replica, request)
            elif key in self.owners:
                demand = self.logical_owners.get(key, (0.0,) * len(self.weights))
                demand_ranks = self.logical_owner_ranks.get(key)
                host_demand = self.logical_host_owners.get(key, 0.0)
                staging_demand = self.logical_staging_owners.get(
                    key, (0.0,) * len(self.weights))
                staging_rank = self.logical_staging_rank.get(key)
                (physical, host_physical, physical_staging,
                 physical_ranks, staging_ranks) = self._physical_union(
                    key, demand, host_demand, staging_demand,
                    demand_ranks=demand_ranks, staging_ranks=staging_rank)
                self._set_physical(
                    replica, key, physical, host_physical, physical_staging,
                    physical_ranks=physical_ranks,
                    staging_ranks=staging_ranks)
                self.budget_version += 1
            self.release_events.append({
                "kind": "async_consumer", "replica": replica,
                "request": request, "consumer": consumer,
                "stages": snapshot.stages, "host_dram": snapshot.host_dram,
                "staging": snapshot.staging, "version": snapshot.version,
                "binding": dict(snapshot.binding),
                "released_bytes": max(
                    0.0, before - self._replica_occupied(replica)),
            })
            return True
        readers = self.grant_readers.get((replica, str(request)))
        if readers and consumer in readers:
            snapshot = readers[consumer]
            if rank is not None and int(snapshot["ref"].rank) != int(rank):
                return False
            bound_generation = snapshot.get("generation")
            if (generation is not None and bound_generation is not None
                    and int(generation) != int(bound_generation)):
                return False
            bound_event = snapshot.get("event")
            if event is not None and event != bound_event:
                return False
            if self.ledger_domain is not None:
                from .runtime.ledger import Ledger as _Ledger, LedgerError
                for reader_key in snapshot.get("readers", ()):
                    try:
                        _Ledger.retire_reader(self.runtime_state, reader_key)
                    except LedgerError:
                        pass
            before = self._replica_occupied(replica)
            self._set_ref((replica, "grant_reader", f"{request}::{consumer}"), ())
            del readers[consumer]
            if not readers:
                del self.grant_readers[(replica, str(request))]
            self._sync_usage()
            self.release_events.append({
                "kind": "grant_reader", "replica": replica,
                "grant": request, "consumer": consumer,
                "binding": snapshot,
                "released_bytes": max(
                    0.0, before - self._replica_occupied(replica)),
            })
            self.budget_version += 1
            return True
        return False

    def release_grant_reader(
        self, replica: int, grant: str, consumer: str, *,
        generation: int | None = None, event=None, rank: int | None = None,
    ) -> bool:
        return self.release_async_consumer(
            replica, grant, consumer, generation=generation, event=event,
            rank=rank)

    def is_retired(self, replica: int, request: str) -> bool:
        return (replica, request) in self.retired_requests

    def pending_async_consumers(self, replica: int, request: str) -> tuple[str, ...]:
        return tuple(sorted(self.async_consumer_holds.get((replica, request), ())))

    def defer_release(self, replica: int, request: str, steps: int | None = None) -> None:
        """Retain KV ownership for finished requests across async scheduling lookahead steps."""
        delay = self.async_delayed_release_steps if steps is None else max(0, int(steps))
        if delay <= 0 or (replica, request) not in self.owners:
            self.release(replica, request)
            return
        self.delayed_owners[replica, request] = (delay, self.owners[replica, request])
        pinned = [0.0] * len(self.weights)
        for (rep, _), (_, sizes) in self.delayed_owners.items():
            if rep == replica:
                pinned = [cur + sz for cur, sz in zip(pinned, sizes)]
        self.peak_delayed_release[replica] = [
            max(old, new) for old, new in zip(self.peak_delayed_release[replica], pinned)
        ]

    def has_delayed_releases(self, replica: int) -> bool:
        return any(rep == replica for rep, _ in self.delayed_owners)

    def step_delayed_releases(self, replica: int) -> None:
        for (rep, req), (rem, sizes) in list(self.delayed_owners.items()):
            if rep != replica:
                continue
            if rem <= 1:
                self.release(replica, req)
            else:
                self.delayed_owners[rep, req] = (rem - 1, sizes)

    def flush_delayed_releases(self, replica: int | None = None) -> None:
        for rep, req in list(self.delayed_owners):
            if replica is None or rep == replica:
                self.release(rep, req)

    def to_dict(self) -> dict:
        result = {
            "status": ("feasible" if all(value >= 0 for row in self.capacity for value in row)
                       else "oom"),
            "basis": "declared_geometry_and_live_allocations",
            "allocation_schema": "mindie-sim.memory-allocation/1",
            "weights_bytes_per_stage_rank": self.weights,
            "weights_payload_bytes_per_stage_rank": self.weights_payload,
            "weights_quant_metadata_bytes_per_stage_rank": self.weights_quant_metadata,
            "reserve_bytes_per_rank": self.policy.reserved,
            "reserve_breakdown": {
                "workspace_bytes": self.policy.workspace_bytes,
                "graph_pool_bytes": self.policy.graph_pool_bytes,
                "communication_buffer_bytes": self.policy.communication_buffer_bytes,
                "runtime_bytes": self.policy.runtime_bytes,
            },
            "activation_workspace_bytes_per_stage_rank": self.workspace,
            "workspace_breakdown": {
                "base_bytes_per_stage_rank": self.workspace_base,
                "async_bytes_per_stage_rank": self.async_workspace,
                "expert_staging_bytes_per_stage_rank": self.workspace_expert_staging,
                "static_offload_bytes_per_stage_rank": self.workspace_static_offload,
                "concurrent_copies": self.concurrent_workspace_copies,
            },
            "temporary_dequant_bytes_per_stage_rank": self.temporary_dequant,
            "async_scheduling": bool(self.case.runtime.async_scheduling),
            "async_workspace_overhead_bytes_per_stage_rank": self.async_workspace,
            "async_delayed_release_steps": self.async_delayed_release_steps,
            "async_in_flight_lookahead_steps": self.async_in_flight_lookahead_steps,
            "peak_delayed_release_cache_bytes_per_replica_stage": self.peak_delayed_release,
            "kv_capacity_bytes_per_replica_stage": self.capacity,
            "peak_cache_bytes_per_replica_stage": self.peak,
            "peak_kv_hbm_resident_bytes_per_replica_stage": self.peak_kv_resident,
            "peak_kv_staging_bytes_per_replica_stage": self.peak_kv_staging,
            "peak_combined_bytes_per_replica_stage": self.peak_combined,
            "budget_version": self.budget_version,
            "grant_bindings": [
                {"replica": replica, "name": name, **self._binding_report(binding)}
                for (replica, name), binding in sorted(self.grant_bindings.items())
            ],
            "onchip_effects": [
                {"replica": replica, "name": name, **binding}
                for (replica, name), binding in sorted(self.onchip_effects.items())
            ],
            "physical_rank_capacity_bytes_per_replica_stage": [
                [[die.hbm_bytes * self.case.runtime.gpu_memory_utilization
                  for die in self._stage_dies(replica, stage)]
                 for stage in range(self.case.runtime.parallel.pp)]
                for replica in range(self.case.runtime.parallel.dp)
            ],
            "physical_rank_live_bytes_per_replica_stage": self.live_rank,
            "physical_rank_staging_bytes_per_replica_stage": self.staging_rank,
            "physical_rank_selected_bytes_per_replica_stage": self.selected_rank,
            "physical_rank_shared_bytes_per_replica_stage": self.shared_rank,
            "physical_rank_grant_bytes_per_replica_stage": self.granted_rank,
            "physical_rank_reader_bytes_per_replica_stage": self.reader_rank,
            "physical_rank_headroom_bytes_per_replica_stage": [
                [
                    [cap - used for cap, used in zip(
                        self.rank_capacity[replica][stage],
                        [a + b + c + d + e + f for a, b, c, d, e, f in zip(
                            self.live_rank[replica][stage],
                            self.staging_rank[replica][stage],
                            self.selected_rank[replica][stage],
                            self.shared_rank[replica][stage],
                            self.granted_rank[replica][stage],
                            self.reader_rank[replica][stage])])]
                    for stage in range(self.case.runtime.parallel.pp)
                ]
                for replica in range(self.case.runtime.parallel.dp)
            ],
            "rank_token_owners": list(self.rank_token_owners or ()),
            "live_cache_bytes_per_replica_stage": self.live,
            "live_grants_bytes_per_replica_stage": self.granted,
            "selected_plan_bytes_per_replica_stage": self.selected_reserved,
            "shared_page_bytes_per_replica_stage": self.shared_reserved,
            "shared_page_owners": [
                {"replica": rep, "owner": owner, **record}
                for (rep, owner), record in sorted(self.shared_pages.items())
            ],
            "peak_resident_bytes_per_replica_stage": [
                [weight + self.policy.reserved + workspace + temporary
                 + self.peak_combined[replica][index]
                 for index, (weight, workspace, temporary) in enumerate(zip(
                     self.weights, self.workspace, self.temporary_dequant))]
                for replica in range(self.case.runtime.parallel.dp)
            ],
            "layer_partition": self.layers,
            "workspace_basis": "simultaneous_layer_tensors_plus_explicit_reserves",
            "fixed_plan": True,
            "fallback": "none",
        }
        max_tokens = self.case.runtime.max_model_len
        cache_allocation = self.cache_allocation(max_tokens, include_components=False)
        gross_kv, resident_kv, staging_kv = self._kv_residency(max_tokens)
        result["cache_allocation_at_max_model_len"] = cache_allocation
        result["kv_offload_at_max_model_len"] = {
            "kvc_hbm_ratio": self.kvc_hbm_ratio,
            "gross_kv_bytes_per_stage_rank": gross_kv,
            "hbm_resident_kv_bytes_per_stage_rank": resident_kv,
            "staging_bytes_per_stage_rank": staging_kv,
            "basis": "same_residency_and_staging_as_resize",
        }
        result["allocation_totals_at_max_model_len"] = {
            "weights_bytes_per_stage_rank": list(self.weights),
            "reserve_bytes_per_rank": self.policy.reserved,
            "workspace_bytes_per_stage_rank": list(self.workspace),
            "temporary_dequant_bytes_per_stage_rank": list(self.temporary_dequant),
            "cache_gross_bytes_per_stage_rank": gross_kv,
            "cache_hbm_resident_bytes_per_stage_rank": resident_kv,
            "cache_staging_bytes_per_stage_rank": staging_kv,
            "cache_unclassified_reserve_bytes_per_stage_rank":
                cache_allocation["allocated"]["reserve_bytes_per_stage_rank"],
            "cache_allocated_bytes_per_stage_rank": resident_kv,
            "cache_valid_bytes_per_stage_rank":
                cache_allocation["valid"]["total_bytes_per_stage_rank"],
            "resident_bytes_per_stage_rank": [
                weight + self.policy.reserved + workspace + temporary + resident + staging
                for weight, workspace, temporary, resident, staging in zip(
                    self.weights, self.workspace, self.temporary_dequant,
                    resident_kv, staging_kv)
            ],
        }
        if any(value > 0.0 for value in self.offload.values()):
            result["offload"] = {
                "expert_ddr_ratio": self.offload["expert_ddr_ratio"],
                "kvc_ddr_ratio": self.offload["kvc_ddr_ratio"],
                "kvc_ssd_ratio": self.offload["kvc_ssd_ratio"],
                "kvc_hbm_ratio": self.kvc_hbm_ratio,
                "gross_weights_bytes_per_stage_rank": self.gross_weights,
                "routed_expert_bytes_per_stage_rank": self.routed_expert_bytes,
                "expert_hbm_saved_bytes_per_stage_rank": self.expert_hbm_saved,
            }
        if self.sparse_offload:
            block_size = int(self.sparse_offload["block_size"])
            idx_row_bytes = int(self.sparse_offload["index_row_bytes"])
            full_idx_layers = len(self.sparse_offload["full_indexer_layers"])
            npu_per_block = max(1, block_size * idx_row_bytes * max(1, full_idx_layers))
            min_stage_cap = max(0.0, min(val for row in self.capacity for val in row))
            npu_limit_blocks = int(min_stage_cap // npu_per_block)
            dram_limit_blocks = int(self.sparse_offload["dram_limit_blocks"])
            workload_limit_blocks = int(self.sparse_offload["workload_limit_blocks"])
            limits = {
                "npu_hbm": npu_limit_blocks,
                "host_dram": dram_limit_blocks,
                "workload": workload_limit_blocks,
            }
            limiting_factor = min(limits, key=limits.__getitem__)
            result["sparse_kv_offload"] = {
                **self.sparse_offload,
                "npu_limit_blocks": npu_limit_blocks,
                "dram_limit_blocks": dram_limit_blocks,
                "workload_limit_blocks": workload_limit_blocks,
                "effective_blocks": limits[limiting_factor],
                "limiting_factor": limiting_factor,
                "host_dram_capacity_bytes_per_replica": self.host_dram_capacity,
                "peak_host_dram_bytes_per_replica": self.host_dram_peak,
                "live_host_dram_bytes_per_replica": self.host_dram_live,
            }
        return result
