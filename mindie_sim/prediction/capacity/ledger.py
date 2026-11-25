"""Runtime capacity bindings, simultaneous per-die peaks, and resident receipts.

[Role] Bind typed pools, charge live physical backing ranges, and emit the
  authoritative per-die ``ResidentMemoryReceipt`` and ``RecoverSSM`` snapshot
  ring HBM headroom calculations consumed by admission, solvers, and export.
[Invariants] Each physical ``(backing_id, generation, die, tier)`` is billed
  once for the union of live ranges. Invalid ranks raise
  ``ValueError("memory_rank_out_of_range")`` instead of coercing to rank 0.
  Peaks are true simultaneous per-die maxima, never cross-rank sums of separate
  resident and staging maxima.
[Timeouts & Locks] Bounded by physical ranks and live holders. No locks or I/O.
[Side-Effect Policy] Mutates only this prediction's capacity domain and usage
  counters; receipt builders are pure read-only projections.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.catalog.cache_pages import Allocation
from mindie_sim.catalog.cache_pools import exact_element_bytes
from mindie_sim.schema.scenario import (
    VISION_ACCOUNTING_SCOPES,
    SpaceTimeTradeoffSpec,
)
from mindie_sim.schema.stack import PreparedWeightSpec

from ..runtime.contracts import (
    AllocationBinding,
    AllocationKey,
    BackingKey,
    CapacityDomainState,
    GrantResult,
    OccurrenceKey,
    PoolKey,
)
from ..runtime.ledger import Ledger
from .contracts import (
    _LedgerBoundary,
    _PhysicalRef,
    _half_open_ranges,
    _hashable_generation,
    _union_bytes,
)


@dataclass(frozen=True)
class ResidentGrantEntry:
    """One physical backing grant inside a per-die ``ResidentMemoryReceipt``."""

    grant_id: str
    category: str
    physical_backing_id: str
    alias_of: str = ""
    generation: int = 0
    tier: str = "hbm"
    checkpoint_bytes: int = 0
    runtime_bytes: int = 0
    allocated_bytes: int = 0
    valid_bytes: int = 0
    lifetime: str = "persistent"
    billed_once_bytes: int = 0
    reclaimable: bool = False
    release_event: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "category": self.category,
            "physical_backing_id": self.physical_backing_id,
            "alias_of": self.alias_of,
            "generation": int(self.generation),
            "tier": self.tier,
            "checkpoint_bytes": int(self.checkpoint_bytes),
            "runtime_bytes": int(self.runtime_bytes),
            "allocated_bytes": int(self.allocated_bytes),
            "valid_bytes": int(self.valid_bytes),
            "lifetime": self.lifetime,
            "billed_once_bytes": int(self.billed_once_bytes),
            "reclaimable": bool(self.reclaimable),
            "release_event": str(self.release_event),
        }


@dataclass(frozen=True)
class DieResidentReceipt:
    """Simultaneous live, reclaimable, and peak HBM residency on one physical die."""

    replica: int
    stage: int
    rank: int
    die_id: int
    usable_hbm_bytes: int
    checkpoint_weight_bytes: int
    prepared_weight_bytes: int
    prepared_load_peak_bytes: int
    vision_weight_bytes: int
    kv_cache_allocated_bytes: int
    kv_cache_valid_bytes: int
    index_cache_allocated_bytes: int
    index_cache_valid_bytes: int
    recurrent_state_bytes: int
    draft_cache_bytes: int
    workspace_bytes: int
    reserve_bytes: int
    staging_bytes: int
    dynamic_grant_bytes: int
    simultaneous_live_hbm_bytes: int
    simultaneous_peak_hbm_bytes: int
    load_peak_hbm_bytes: int
    free_hbm_headroom_bytes: int
    host_dram_bytes: int
    grants: tuple[ResidentGrantEntry, ...] = ()
    reclaimable_hbm_bytes: int = 0
    reclaimable_release_events: tuple[str, ...] = ()

    @property
    def usable_bytes(self) -> int:
        return int(self.usable_hbm_bytes)

    @property
    def current_live_bytes(self) -> int:
        return int(self.simultaneous_live_hbm_bytes)

    @property
    def reclaimable_bytes(self) -> int:
        return int(self.reclaimable_hbm_bytes)

    @property
    def historical_peak_bytes(self) -> int:
        return int(self.simultaneous_peak_hbm_bytes)

    @property
    def immediate_free_hbm_headroom_bytes(self) -> int:
        return int(self.usable_hbm_bytes - self.simultaneous_live_hbm_bytes)

    @property
    def post_reclaim_free_hbm_headroom_bytes(self) -> int:
        return int(
            self.usable_hbm_bytes
            - (self.simultaneous_live_hbm_bytes - self.reclaimable_hbm_bytes)
        )

    @property
    def historical_free_hbm_headroom_bytes(self) -> int:
        return int(self.usable_hbm_bytes - self.simultaneous_peak_hbm_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica": int(self.replica),
            "stage": int(self.stage),
            "rank": int(self.rank),
            "die_id": int(self.die_id),
            "usable_hbm_bytes": int(self.usable_hbm_bytes),
            "checkpoint_weight_bytes": int(self.checkpoint_weight_bytes),
            "prepared_weight_bytes": int(self.prepared_weight_bytes),
            "prepared_load_peak_bytes": int(self.prepared_load_peak_bytes),
            "vision_weight_bytes": int(self.vision_weight_bytes),
            "kv_cache_allocated_bytes": int(self.kv_cache_allocated_bytes),
            "kv_cache_valid_bytes": int(self.kv_cache_valid_bytes),
            "index_cache_allocated_bytes": int(self.index_cache_allocated_bytes),
            "index_cache_valid_bytes": int(self.index_cache_valid_bytes),
            "recurrent_state_bytes": int(self.recurrent_state_bytes),
            "draft_cache_bytes": int(self.draft_cache_bytes),
            "workspace_bytes": int(self.workspace_bytes),
            "reserve_bytes": int(self.reserve_bytes),
            "staging_bytes": int(self.staging_bytes),
            "dynamic_grant_bytes": int(self.dynamic_grant_bytes),
            "simultaneous_live_hbm_bytes": int(self.simultaneous_live_hbm_bytes),
            "reclaimable_hbm_bytes": int(self.reclaimable_hbm_bytes),
            "reclaimable_release_events": list(self.reclaimable_release_events),
            "simultaneous_peak_hbm_bytes": int(self.simultaneous_peak_hbm_bytes),
            "load_peak_hbm_bytes": int(self.load_peak_hbm_bytes),
            "free_hbm_headroom_bytes": int(self.free_hbm_headroom_bytes),
            "immediate_free_hbm_headroom_bytes": self.immediate_free_hbm_headroom_bytes,
            "post_reclaim_free_hbm_headroom_bytes": self.post_reclaim_free_hbm_headroom_bytes,
            "historical_free_hbm_headroom_bytes": self.historical_free_hbm_headroom_bytes,
            "host_dram_bytes": int(self.host_dram_bytes),
            "grants": [grant.to_dict() for grant in self.grants],
        }


@dataclass(frozen=True)
class ResidentMemoryReceipt:
    """Authoritative per-die resident memory receipt keyed by scenario/phase/occurrence."""

    scenario: str
    phase: str
    occurrence: str
    vision_accounting: str
    dies: tuple[DieResidentReceipt, ...]

    @property
    def live_die_hbm_bytes(self) -> int:
        return max((die.simultaneous_live_hbm_bytes for die in self.dies), default=0)

    @property
    def reclaimable_die_hbm_bytes(self) -> int:
        return max((die.reclaimable_hbm_bytes for die in self.dies), default=0)

    @property
    def peak_die_hbm_bytes(self) -> int:
        return max((die.simultaneous_peak_hbm_bytes for die in self.dies), default=0)

    @property
    def min_free_hbm_headroom_bytes(self) -> int:
        return min((die.free_hbm_headroom_bytes for die in self.dies), default=0)

    @property
    def min_immediate_free_hbm_headroom_bytes(self) -> int:
        return min((die.immediate_free_hbm_headroom_bytes for die in self.dies), default=0)

    @property
    def min_post_reclaim_free_hbm_headroom_bytes(self) -> int:
        return min((die.post_reclaim_free_hbm_headroom_bytes for die in self.dies), default=0)

    @property
    def min_historical_free_hbm_headroom_bytes(self) -> int:
        return min((die.historical_free_hbm_headroom_bytes for die in self.dies), default=0)

    def die_receipt(self, replica: int, stage: int, rank: int) -> DieResidentReceipt:
        for die in self.dies:
            if die.replica == replica and die.stage == stage and die.rank == rank:
                return die
        raise KeyError((replica, stage, rank))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "phase": self.phase,
            "occurrence": self.occurrence,
            "vision_accounting": self.vision_accounting,
            "live_die_hbm_bytes": self.live_die_hbm_bytes,
            "reclaimable_die_hbm_bytes": self.reclaimable_die_hbm_bytes,
            "peak_die_hbm_bytes": self.peak_die_hbm_bytes,
            "min_free_hbm_headroom_bytes": self.min_free_hbm_headroom_bytes,
            "min_immediate_free_hbm_headroom_bytes": self.min_immediate_free_hbm_headroom_bytes,
            "min_post_reclaim_free_hbm_headroom_bytes": self.min_post_reclaim_free_hbm_headroom_bytes,
            "min_historical_free_hbm_headroom_bytes": self.min_historical_free_hbm_headroom_bytes,
            "dies": [die.to_dict() for die in self.dies],
        }


@dataclass(frozen=True)
class CacheLedgerProjection:
    """Internal per-die / per-stage / per-component exact integer cache projection.

    Pairs the frozen D0 Contract 3 ``ResidentMemoryReceipt`` with exact integer
    component allocations per PP stage so ``export/memory_ledger.py`` projects
    with 0 B multi-key bijection and zero shadow HBM recomputation.
    """

    receipt: ResidentMemoryReceipt
    bottleneck_rank_by_stage: tuple[int, ...]
    per_stage_component_batch_bytes: tuple[dict[str, int], ...]
    per_stage_component_per_req_bytes: tuple[dict[str, float], ...]
    active_layers_by_component: dict[str, tuple[int, ...]]
    cache_kind_by_component: dict[str, str]
    fixed_per_die_components: frozenset[str]
    normalized_block_size: int
    hybrid_pool_receipt: Any = None


def _exact_int_bound(value: Any, name: str) -> int:
    """Validate an exact integer byte bound without float floor/ceil smearing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_invalid")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name}_must_be_integer")
        return int(value)
    return int(value)


def _exact_half_open_int_ranges(
    raw_ranges: Any, amount: Any,
) -> tuple[tuple[int, int], ...]:
    """Build exact integer half-open ``[lo, hi)`` byte ranges directly."""
    if raw_ranges:
        out: list[tuple[int, int]] = []
        for lo_raw, hi_raw in raw_ranges:
            lo = _exact_int_bound(lo_raw, "memory_byte_range")
            hi = _exact_int_bound(hi_raw, "memory_byte_range")
            if lo < 0 or hi < lo:
                raise ValueError("memory_byte_range_invalid")
            if hi > lo:
                out.append((lo, hi))
        return tuple(out)
    amt = _exact_int_bound(amount, "memory_byte_amount")
    if amt < 0:
        raise ValueError("memory_byte_amount_negative")
    return ((0, amt),) if amt > 0 else ()


@dataclass(frozen=True)
class RecoverSsmRingBreakdown:
    """Exact HBM cost of storing ``K`` speculative SSM snapshots vs ``RecoverSSM``."""

    batch_per_dp: int
    k_proposals: int
    verify_width: int
    linear_layers: int
    max_stage_linear_layers: int
    snapshot_ring_bytes_per_die: int
    base_state_bytes_per_die: int
    replay_records_bytes_per_die: int
    delta_hbm_vs_recover_ssm_bytes: int
    stage_snapshot_ring_bytes: tuple[int, ...]
    stage_delta_vs_recover_ssm_bytes: tuple[int, ...]


def prepared_weight_bytes(spec: PreparedWeightSpec) -> int:
    """Exact runtime HBM bytes for one ``PreparedWeightSpec`` (0 when aliased)."""
    if spec.alias_of:
        return 0
    elements = math.prod(spec.local_shape) if spec.local_shape else 0
    payload = exact_element_bytes(elements, spec.dtype) if elements > 0 else 0
    return int(payload + int(spec.extra_metadata_bytes))


def recover_ssm_snapshot_ring_breakdown(
    case: Any,
    batch_per_dp: int,
    k_proposals: int,
    *,
    runtime: Any = None,
) -> RecoverSsmRingBreakdown:
    """Exact per-die HBM cost of keeping ``K`` SSM snapshot ring slots per request."""
    from mindie_sim.catalog.cache import layer_stage_map
    from mindie_sim.catalog.cache_pools import LINEAR_ATTENTION_KINDS
    from mindie_sim.catalog.speculative_state import speculative_state_plan

    batch = int(batch_per_dp)
    k = int(k_proposals)
    if batch < 0 or k < 0:
        raise ValueError("recover_ssm_ring_args_must_be_nonnegative")
    rt = runtime if runtime is not None else case.runtime
    model = case.model
    parallel = rt.parallel
    pp = max(1, int(parallel.pp))
    stage_map = layer_stage_map(model, parallel, rt)
    stage_extra = [0] * pp
    stage_base = [0] * pp
    stage_replay = [0] * pp
    stage_delta = [0] * pp
    stage_layers = [0] * pp
    verify_width = k + 1
    for plan in model.layer_plans:
        if plan.attention not in LINEAR_ATTENTION_KINDS:
            continue
        stage = stage_map[plan.index] if plan.index < len(stage_map) else pp - 1
        stage_layers[stage] += 1
        if batch == 0:
            continue
        ckpt = speculative_state_plan(
            model, plan, parallel, query_tokens=verify_width,
            recover_ssm=False, capacity_padded=True,
            state_contract=getattr(plan, "state_contract", None), runtime=rt,
        )
        replay = speculative_state_plan(
            model, plan, parallel, query_tokens=verify_width,
            recover_ssm=True, capacity_padded=True,
            state_contract=getattr(plan, "state_contract", None), runtime=rt,
        )
        base_b = int(round(replay.recurrent_bytes))
        extra_b = max(0, int(round(ckpt.recurrent_bytes)) - base_b)
        replay_b = int(round(replay.replay_bytes))
        delta_b = int(round(ckpt.total_bytes - replay.total_bytes))
        stage_base[stage] += batch * base_b
        stage_extra[stage] += batch * extra_b
        stage_replay[stage] += batch * replay_b
        stage_delta[stage] += batch * delta_b
    return RecoverSsmRingBreakdown(
        batch_per_dp=batch,
        k_proposals=k,
        verify_width=verify_width,
        linear_layers=sum(stage_layers),
        max_stage_linear_layers=max(stage_layers, default=0),
        snapshot_ring_bytes_per_die=max(stage_extra, default=0),
        base_state_bytes_per_die=max(stage_base, default=0),
        replay_records_bytes_per_die=max(stage_replay, default=0),
        delta_hbm_vs_recover_ssm_bytes=max(stage_delta, default=0),
        stage_snapshot_ring_bytes=tuple(stage_extra),
        stage_delta_vs_recover_ssm_bytes=tuple(stage_delta),
    )


def estimate_recover_ssm_snapshot_ring_bytes(
    case: Any,
    batch_per_dp: int,
    k_proposals: int,
    *,
    net_over_replay: bool = False,
    runtime: Any = None,
) -> int:
    """Exact bottleneck-die HBM bytes required for ``RecoverSSM`` snapshot ring."""
    bd = recover_ssm_snapshot_ring_breakdown(
        case, batch_per_dp, k_proposals, runtime=runtime,
    )
    return (
        bd.delta_hbm_vs_recover_ssm_bytes
        if net_over_replay else bd.snapshot_ring_bytes_per_die
    )


def _deduplicate_grant_entries(
    entries: Sequence[ResidentGrantEntry],
) -> tuple[tuple[ResidentGrantEntry, ...], int]:
    """Resolve ``alias_of`` chains and bill each ``(backing_id, gen, tier)`` ONCE."""
    by_id = {entry.grant_id: entry for entry in entries}
    billed_peaks: dict[tuple[str, int, str], int] = {}
    resolved_entries: list[ResidentGrantEntry] = []
    for entry in entries:
        backing = entry.physical_backing_id or entry.grant_id
        seen: set[str] = set()
        alias = entry.alias_of
        while alias and alias not in seen:
            seen.add(alias)
            target = by_id.get(alias)
            if target is None:
                backing = alias
                break
            backing = target.physical_backing_id or target.grant_id
            alias = target.alias_of
        raw_bytes = int(entry.allocated_bytes or entry.runtime_bytes)
        if entry.tier.lower() in {"hbm", ""} and entry.lifetime != "load_peak_only":
            key = (backing, int(entry.generation), entry.tier.lower() or "hbm")
            prev = billed_peaks.get(key, 0)
            delta = max(0, raw_bytes - prev)
            if raw_bytes > prev:
                billed_peaks[key] = raw_bytes
        else:
            delta = 0
        resolved_entries.append(ResidentGrantEntry(
            grant_id=entry.grant_id,
            category=entry.category,
            physical_backing_id=backing,
            alias_of=entry.alias_of,
            generation=entry.generation,
            tier=entry.tier,
            checkpoint_bytes=entry.checkpoint_bytes,
            runtime_bytes=entry.runtime_bytes,
            allocated_bytes=entry.allocated_bytes,
            valid_bytes=entry.valid_bytes,
            lifetime=entry.lifetime,
            billed_once_bytes=delta,
            reclaimable=bool(entry.reclaimable),
            release_event=str(entry.release_event),
        ))
    return tuple(resolved_entries), sum(billed_peaks.values())


class _MemoryLedgerMixin:
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

    def _backing_physical_rank(self, replica: int, stage: int, rank: int, tier: str) -> int:
        """Globally unique physical die rank for ``BackingKey`` so distinct DP/PP pools never collide."""
        if str(tier or "hbm").lower() == "host":
            return -(int(replica) + 1)
        pp = max(1, int(self.case.runtime.parallel.pp))
        return (int(replica) * pp + int(stage)) * self.width + int(rank)

    def _typed_spec_bindings(self, replica: int, spec: dict):
        """Construct typed ``AllocationBinding`` records with exact integer half-open ranges."""
        if self.ledger_domain is None:
            return ()
        stage = int(spec.get("stage", 0))
        rank_raw = spec.get("rank")
        rank = 0 if rank_raw is None else int(rank_raw)
        account_rank = self._account_rank(rank)
        amount = float(spec.get("bytes", 0.0))
        generation = self._ledger_generation(spec.get("generation"))
        occurrence = self._ledger_occurrence(replica, stage, rank)
        tier = str(spec.get("tier", "hbm")).lower()
        backing_id = str(
            spec.get("alias_of")
            or spec.get("physical_backing_id")
            or spec.get("physical_key")
            or spec.get("name", "")
        )
        backing_rank = self._backing_physical_rank(replica, stage, account_rank, tier)
        backing = BackingKey(
            physical_domain=self.ledger_domain.domain_id,
            rank=backing_rank,
            tier=tier,
            storage_id=backing_id,
            generation=generation,
        )
        if tier == "host":
            primary_pool = PoolKey(
                physical_domain=self.ledger_domain.domain_id,
                rank=-1,
                tier="host",
                name="host",
                unit=int(replica),
            )
        else:
            primary_pool = self.ledger_pool(
                replica, stage, account_rank, tier=tier,
                domain=self.ledger_domain.domain_id,
            )
        ranges = _exact_half_open_int_ranges(spec.get("_ranges"), spec.get("bytes", 0))
        key = AllocationKey(
            occurrence=occurrence,
            scope_kind="invocation",
            scope_ordinal=stage,
            symbol=self._ledger_serial,
            generation=generation,
        )
        result = [AllocationBinding(
            key=key,
            pool=primary_pool,
            amount=amount,
            backing=backing,
            byte_ranges=ranges,
            owner=str(spec.get("owner") or ""),
            storage=str(spec.get("storage") or ""),
            access="readwrite",
        )]
        host_raw = spec.get("host_bytes", 0)
        host = float(host_raw)
        if host > 0.0:
            host_int = _exact_int_bound(host_raw, "memory_host_bytes")
            self._ledger_serial += 1
            host_key = AllocationKey(
                occurrence=self._ledger_occurrence(replica, stage, -1),
                scope_kind="external",
                scope_ordinal=self._ledger_serial,
                symbol=self._ledger_serial,
                generation=generation,
            )
            result.append(AllocationBinding(
                key=host_key,
                pool=PoolKey(
                    physical_domain=self.ledger_domain.domain_id,
                    rank=-1,
                    tier="host",
                    name="host",
                    unit=int(replica),
                ),
                amount=host,
                backing=BackingKey(
                    physical_domain=self.ledger_domain.domain_id,
                    rank=self._backing_physical_rank(replica, stage, -1, "host"),
                    tier="host",
                    storage_id=backing_id + "#host",
                    generation=generation,
                ),
                byte_ranges=((0, host_int),),
                owner=str(spec.get("owner") or ""),
                storage=str(spec.get("storage") or ""),
                access="readwrite",
            ))
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
        self._sync_domain_occupancy()
        return domain

    def _sync_domain_occupancy(self) -> None:
        """Mirror non-domain occupancy (KV, staging, selected, prepared weights) into ``ledger_domain``."""
        domain = self.ledger_domain
        if domain is None or self.runtime_state is None:
            return
        shadow_map: dict[PoolKey, tuple[AllocationKey, float]] = getattr(
            self, "_domain_shadow_grants", None,
        )
        if shadow_map is None:
            shadow_map = {}
            self._domain_shadow_grants = shadow_map
        hbm_full, host_full = self._state_usage()
        for pool, cap in list(domain.capacities.items()):
            parsed = self._parse_ledger_pool(pool)
            if parsed is None:
                continue
            tier_kind, rep, st, rk = parsed
            prev_key, prev_amt = shadow_map.get(pool, (None, 0.0))
            prev_live = (
                prev_key is not None
                and prev_key in domain.grants
                and not domain.grants[prev_key].released
            )
            eff_prev = float(prev_amt) if prev_live else 0.0
            domain_real = max(0.0, float(domain.current.get(pool, 0.0)) - eff_prev)
            if tier_kind == "host":
                target_total = float(host_full[rep])
                prep_delta = 0.0
            else:
                target_total = float(hbm_full[rep][st][rk])
                prep_delta = max(0.0, float(cap) - float(self.rank_capacity[rep][st][rk]))
            missing = max(0.0, (target_total + prep_delta) - domain_real)
            if abs(missing - eff_prev) <= 1e-9 and (prev_live or missing <= 1e-9):
                continue
            if prev_live and prev_key is not None:
                Ledger.release(
                    self.runtime_state, prev_key, after=self._ledger_boundary(),
                )
            shadow_map.pop(pool, None)
            if missing <= 1e-9:
                continue
            avail = max(0.0, float(cap) - float(domain.current.get(pool, 0.0)))
            acq_amt = min(missing, avail) if math.isfinite(float(cap)) else missing
            if acq_amt <= 1e-9:
                continue
            occ = self._ledger_occurrence(rep, st, rk)
            s_key = AllocationKey(
                occurrence=occ,
                scope_kind="memory_occupancy",
                scope_ordinal=st,
                symbol=self._ledger_serial,
                generation=0,
            )
            binding = AllocationBinding(
                key=s_key,
                pool=pool,
                amount=acq_amt,
                backing=None,
                byte_ranges=(),
                owner="memory:occupancy",
                storage="occupancy",
                access="readwrite",
            )
            res = Ledger.try_acquire(
                self.runtime_state, (binding,), holder=occ, at=self._ledger_boundary(),
            )
            if res.granted:
                shadow_map[pool] = (s_key, acq_amt)

    def _domain_peak_hbm(self, replica: int, stage: int, rank: int) -> float:
        """Read historical dynamic HBM peak recorded on ``self.ledger_domain`` for one die."""
        domain = self.ledger_domain
        if domain is None:
            return 0.0
        pool = self.ledger_pool(replica, stage, rank, domain=domain.domain_id)
        raw_peak = float(domain.peaks.get(pool, 0.0))
        if raw_peak <= 0.0:
            return 0.0
        cap = domain.capacities.get(pool)
        prep_delta = (
            max(0.0, float(cap) - float(self.rank_capacity[replica][stage][rank]))
            if cap is not None and math.isfinite(float(cap)) else 0.0
        )
        return max(0.0, raw_peak - prep_delta)

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
        tier_norm = str(tier or "hbm").lower()
        if tier_norm == "host":
            pool = PoolKey(
                physical_domain=domain_id, rank=-1, tier="host",
                name="host", unit=int(replica),
            )
        else:
            pool = self.ledger_pool(replica, stage, rank, tier=tier_norm, domain=domain_id)
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
                physical_domain=domain_id,
                rank=self._backing_physical_rank(replica, stage, int(rank), tier_norm),
                tier=tier_norm,
                storage_id=str(backing_key), generation=int(generation))
        raw = ranges if ranges is not None else record.get("ranges")
        byte_ranges = _half_open_ranges(raw, value)
        return AllocationBinding(
            key=key, pool=pool, amount=value, backing=backing,
            byte_ranges=byte_ranges, reservation_parent=parent,
            owner=str(owner or record.get("owner", "")),
            storage=str(storage or record.get("storage", "")),
            access=access, acquire=acquire)

    def _parse_ledger_pool(self, pool: PoolKey) -> tuple[str, int, int, int] | None:
        if self.ledger_domain is None or pool.physical_domain != self.ledger_domain.domain_id:
            return None
        p = self.case.runtime.parallel
        tier = str(pool.tier or "hbm").lower()
        rep = int(pool.unit)
        if not 0 <= rep < p.dp:
            return None
        if tier == "host":
            return ("host", rep, 0, -1)
        if tier in {"hbm", ""} and str(pool.name).startswith("stage"):
            try:
                stage = int(str(pool.name)[5:])
            except ValueError:
                return None
            rank = int(pool.rank)
            if 0 <= stage < p.pp and 0 <= rank < self.width:
                return ("hbm", rep, stage, rank)
        return None

    def _mirrored_typed_keys(self) -> tuple[set, set]:
        mirrored_allocs = {
            alloc for keys in self._typed_grants.values() for alloc in keys
        }
        shadow_map = getattr(self, "_domain_shadow_grants", None)
        if shadow_map:
            for s_key, _amt in shadow_map.values():
                mirrored_allocs.add(s_key)
        mirrored_readers = {
            rk
            for readers in self.grant_readers.values()
            for snap in readers.values()
            for rk in snap.get("readers", ())
        }
        return mirrored_allocs, mirrored_readers

    def _typed_preflight_blocked(self, items: tuple) -> list[PoolKey]:
        """Check unified physical HBM/host budget for incoming typed bindings."""
        extra_holders = []
        unbacked_hbm: list[tuple[int, int, int, float, PoolKey]] = []
        unbacked_host: list[tuple[int, float, PoolKey]] = []
        pool_for_loc: dict[tuple[str, int, int, int], PoolKey] = {}
        for idx, item in enumerate(items):
            if not isinstance(item, AllocationBinding) or not item.acquire:
                continue
            if item.reservation_parent is not None:
                continue
            parsed = self._parse_ledger_pool(item.pool)
            if parsed is None:
                continue
            tier_kind, rep, st, rk = parsed
            pool_for_loc[(tier_kind, rep, st, rk)] = item.pool
            if item.backing is None:
                if tier_kind == "host":
                    unbacked_host.append((rep, float(item.amount), item.pool))
                else:
                    unbacked_hbm.append((rep, st, rk, float(item.amount), item.pool))
                continue
            ref_ranges = _half_open_ranges(item.byte_ranges, float(item.amount))
            if not ref_ranges:
                continue
            ref = _PhysicalRef(
                rank=rk if tier_kind == "hbm" else 0,
                stage=st,
                storage=str(item.storage or ""),
                tier=tier_kind,
                physical_key=str(item.backing.storage_id),
                generation=_hashable_generation(item.backing.generation),
                ranges=ref_ranges if tier_kind == "hbm" else (),
                host_ranges=ref_ranges if tier_kind == "host" else (),
            )
            extra_holders.append(((rep, "typed_cand", str(idx)), (ref,)))
        if not extra_holders and not unbacked_hbm and not unbacked_host:
            return []
        hbm, host = self._state_usage(extra_holders=extra_holders)
        for rep, st, rk, amt, _pool in unbacked_hbm:
            hbm[rep][st][rk] += amt
        for rep, amt, _pool in unbacked_host:
            host[rep] += amt
        blocked: list[PoolKey] = []
        for (tier_kind, rep, st, rk), pool in pool_for_loc.items():
            if tier_kind == "host":
                cap = self.host_dram_capacity[rep]
                if host[rep] > cap + max(1e-9, abs(cap) * 1e-12):
                    blocked.append(pool)
            else:
                cap = self.rank_capacity[rep][st][rk]
                if hbm[rep][st][rk] > cap + max(1e-9, abs(cap) * 1e-12):
                    blocked.append(pool)
        return blocked

    def grant_typed(self, bindings, *, holder, at):
        """Atomic typed grant against the adopted RuntimeState domain."""
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        items = tuple(bindings)
        blocked = self._typed_preflight_blocked(items)
        if blocked:
            return GrantResult(
                granted=False, allocations=(), blocked_pools=tuple(blocked),
            )
        result = Ledger.try_acquire(
            self.runtime_state, items, holder=holder, at=at,
        )
        if result.granted and result.allocations:
            self._sync_usage()
            self.budget_version += 1
        return result

    def retain_reader_typed(self, allocation: AllocationKey, *, reader,
                            ranges=(), expected_generation=None):
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        lease = Ledger.retain_reader(
            self.runtime_state, allocation, reader=reader, ranges=ranges,
            expected_generation=expected_generation)
        self._sync_usage()
        self.budget_version += 1
        return lease

    def release_typed(self, allocation: AllocationKey, *, after,
                      expected_generation=None):
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        result = Ledger.release(
            self.runtime_state, allocation, after=after,
            expected_generation=expected_generation)
        if result.released or result.deferred:
            self._sync_usage()
            self.budget_version += 1
        return result

    def ledger_usage(self) -> dict[PoolKey, float]:
        if self.ledger_domain is None:
            return {}
        return dict(self.ledger_domain.current)

    # ------------------------------------------------------------------
    # Unified physical reference ledger
    # ------------------------------------------------------------------
    def _account_rank(self, rank) -> int:
        """Validate physical rank; never coerce invalid ranks silently to 0."""
        if rank is None:
            return 0
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("memory_rank_out_of_range")
        if not 0 <= rank < self.width:
            raise ValueError("memory_rank_out_of_range")
        return int(rank)

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
            if not item.live:
                continue
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
        backing_key = (
            spec.get("alias_of")
            or spec.get("physical_backing_id")
            or spec.get("physical_key")
            or spec["name"]
        )
        return self._make_ref(
            spec["replica"], rank=spec.get("rank"), stage=spec["stage"],
            storage=spec.get("storage"), tier=spec.get("tier", "hbm"),
            physical_key=backing_key,
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

    def _is_cross_dp_dcp_cohort_active(self) -> bool:
        p = self.case.runtime.parallel
        return (
            bool(getattr(p, "cp_spans_dp", False))
            and int(getattr(p, "dcp_dp_width", 1) or 1) > 1
            and self.rank_token_owners is None
        )

    def _accumulate_owner_map(
        self,
        target: list[list[list[float]]],
        owners: Mapping[tuple[int, str], tuple[tuple[float, ...], ...]],
        cohort_owners: Mapping[tuple[int, str], Mapping[int, tuple[tuple[float, ...], ...]]],
        *,
        exclude_request: tuple[int, str] | None = None,
    ) -> None:
        for key, rows in owners.items():
            if key == exclude_request:
                continue
            cohort_map = cohort_owners.get(key)
            if cohort_map is not None:
                for peer_rep, peer_rows in cohort_map.items():
                    for stage, row in enumerate(peer_rows):
                        for rank, value in enumerate(row):
                            target[peer_rep][stage][rank] += float(value)
                continue
            replica = key[0]
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    target[replica][stage][rank] += float(value)

    def _request_usage_ranks(self) -> list[list[list[float]]]:
        result = [
            [[0.0] * self.width for _ in range(self.case.runtime.parallel.pp)]
            for _ in range(self.case.runtime.parallel.dp)
        ]
        self._accumulate_owner_map(result, self.owner_ranks, self._cohort_owner_ranks)
        return result

    def _staging_usage_ranks(self) -> list[list[list[float]]]:
        result = [
            [[0.0] * self.width for _ in range(self.case.runtime.parallel.pp)]
            for _ in range(self.case.runtime.parallel.dp)
        ]
        self._accumulate_owner_map(result, self.staging_owner_ranks, self._cohort_staging_ranks)
        return result

    def _cohort_logical_maps(self) -> tuple[dict, dict, dict]:
        log_res = getattr(self, "_cohort_logical_owner_ranks", None)
        if log_res is None:
            log_res = {}
            self._cohort_logical_owner_ranks = log_res
        log_stg = getattr(self, "_cohort_logical_staging_ranks", None)
        if log_stg is None:
            log_stg = {}
            self._cohort_logical_staging_ranks = log_stg
        holds = getattr(self, "_cohort_hold_snapshots", None)
        if holds is None:
            holds = {}
            self._cohort_hold_snapshots = holds
        return log_res, log_stg, holds

    def _union_cohort_maps(
        self,
        base_map: Mapping[int, tuple[tuple[float, ...], ...]],
        extra_maps: Sequence[Mapping[int, tuple[tuple[float, ...], ...]]],
    ) -> dict[int, tuple[tuple[float, ...], ...]]:
        """Element-wise max across all peer replicas, stages, and local ranks."""
        p = self.case.runtime.parallel
        merged: dict[int, list[list[float]]] = {}
        for mapping in (base_map, *extra_maps):
            for peer_rep, rows in mapping.items():
                dst = merged.get(peer_rep)
                if dst is None:
                    dst = [[0.0] * self.width for _ in range(p.pp)]
                    merged[peer_rep] = dst
                for st, row in enumerate(rows):
                    for rk, val in enumerate(row):
                        fv = float(val)
                        if fv > dst[st][rk]:
                            dst[st][rk] = fv
        return {rep: tuple(tuple(row) for row in rows) for rep, rows in merged.items()}

    def _freed_total_bytes(self, before_hbm, before_host, after_hbm, after_host) -> float:
        """Sum positive per-die HBM and per-replica host byte reductions across all replicas."""
        p = self.case.runtime.parallel
        freed = 0.0
        for rep in range(p.dp):
            freed += max(0.0, float(before_host[rep]) - float(after_host[rep]))
            for st in range(p.pp):
                for rk in range(self.width):
                    freed += max(0.0, float(before_hbm[rep][st][rk]) - float(after_hbm[rep][st][rk]))
        return freed

    def _resident_admission(
        self, replica: int, request: str, tokens: int,
        *, query_tokens: int | None = None, proposal_tokens: int | None = None,
        phase: str = "decode",
    ) -> dict:
        if not self._is_cross_dp_dcp_cohort_active():
            return super()._resident_admission(
                replica, request, tokens,
                query_tokens=query_tokens, proposal_tokens=proposal_tokens,
                phase=phase,
            )
        key = (replica, request)
        eff_q = self._resolve_request_query_tokens(
            key, query_tokens=query_tokens, proposal_tokens=proposal_tokens,
        )
        p = self.case.runtime.parallel
        if p.pp > 1:
            dp_width = max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
            cohort_start = (int(replica) // dp_width) * dp_width
            cohort_end = min(int(p.dp), cohort_start + dp_width)
            res_map: dict[int, tuple[tuple[float, ...], ...]] = {}
            stg_map: dict[int, tuple[tuple[float, ...], ...]] = {}
            for peer_rep in range(cohort_start, cohort_end):
                peer_res = [[0.0] * self.width for _ in range(p.pp)]
                peer_stg = [[0.0] * self.width for _ in range(p.pp)]
                for rk in range(self.width):
                    cohort_rk = ((peer_rep - cohort_start) * p.pp) * self.width + rk
                    if peer_rep == int(replica):
                        gross = self._rank_cache_bytes(
                            tokens, padded=True, rank=cohort_rk,
                            query_tokens=eff_q, phase=phase,
                        )
                    else:
                        gross = self._dp_cohort_extra_by_stage(
                            tokens, padded=True, rank=cohort_rk,
                            query_tokens=eff_q, phase=phase, peer_count_override=1,
                        )
                    layer_max = (
                        self._rank_max_layer_cache_bytes(
                            tokens, cohort_rk, query_tokens=eff_q, phase=phase,
                        )
                        if self.kvc_hbm_ratio < 1.0 else (0.0,) * p.pp
                    )
                    for st in range(p.pp):
                        if self.kvc_hbm_ratio == 1.0:
                            peer_res[st][rk] = float(gross[st])
                        else:
                            peer_res[st][rk] = float(gross[st]) * self.kvc_hbm_ratio
                            peer_stg[st][rk] = float(layer_max[st]) * (1.0 - self.kvc_hbm_ratio)
                res_map[peer_rep] = tuple(tuple(row) for row in peer_res)
                stg_map[peer_rep] = tuple(tuple(row) for row in peer_stg)
        else:
            _ra, _sa, res_map, stg_map = self._kv_residency_cohort(
                replica, tokens, query_tokens=eff_q, phase=phase,
            )
        logical_res_map = dict(res_map)
        logical_stg_map = dict(stg_map)
        _, _, cohort_holds = self._cohort_logical_maps()
        hold_snaps = cohort_holds.get(key, {})
        if hold_snaps:
            res_map = self._union_cohort_maps(
                logical_res_map, [snap[0] for snap in hold_snaps.values()],
            )
            stg_map = self._union_cohort_maps(
                logical_stg_map, [snap[1] for snap in hold_snaps.values()],
            )
        own_logical_res = logical_res_map[int(replica)]
        own_logical_stg = logical_stg_map[int(replica)]
        demand = tuple(max(row) for row in own_logical_res)
        demand_staging = tuple(max(row) for row in own_logical_stg)
        host_demand = self.host_dram_cache_bytes(tokens, query_tokens=eff_q, phase=phase)
        (physical, host_physical, physical_staging,
         physical_ranks, physical_staging_ranks) = self._physical_union(
            key, demand, host_demand, demand_staging,
            demand_ranks=res_map[int(replica)], staging_ranks=stg_map[int(replica)],
        )
        res_map[int(replica)] = physical_ranks
        stg_map[int(replica)] = physical_staging_ranks
        res_all = [[[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        stg_all = [[[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        for peer_rep, rows in res_map.items():
            res_all[peer_rep] = [list(row) for row in rows]
        for peer_rep, rows in stg_map.items():
            stg_all[peer_rep] = [list(row) for row in rows]
        host_request = [0.0] * p.dp
        host_request[replica] = float(host_physical)
        hbm, host = self._state_usage(
            exclude_request=key, request_usage=res_all,
            staging_usage=stg_all, host_request=host_request,
        )
        resolved_q = self._effective_cache_query_tokens(eff_q)
        return {
            "demand": demand, "physical": physical,
            "physical_ranks": physical_ranks,
            "demand_ranks": own_logical_res,
            "demand_staging": demand_staging,
            "demand_staging_ranks": own_logical_stg,
            "physical_staging": physical_staging,
            "physical_staging_ranks": physical_staging_ranks,
            "cohort_owner_ranks": res_map,
            "cohort_staging_ranks": stg_map,
            "cohort_logical_owner_ranks": logical_res_map,
            "cohort_logical_staging_ranks": logical_stg_map,
            "host_demand": float(host_demand),
            "host_physical": float(host_physical),
            "query_tokens": int(resolved_q),
            "candidate_hbm": hbm, "candidate_host": host,
            "ok": self._usage_budget_ok(hbm, host),
        }

    def resize(
        self, replica: int, request: str, tokens: int,
        *, query_tokens: int | None = None, proposal_tokens: int | None = None,
        phase: str = "decode", valid_tokens: int | None = None,
    ) -> bool:
        if not self._is_cross_dp_dcp_cohort_active():
            return super().resize(
                replica, request, tokens,
                query_tokens=query_tokens, proposal_tokens=proposal_tokens,
                phase=phase, valid_tokens=valid_tokens,
            )
        key = (replica, request)
        candidate = self._resident_admission(
            replica, request, tokens,
            query_tokens=query_tokens, proposal_tokens=proposal_tokens,
            phase=phase,
        )
        if not candidate["ok"]:
            return False
        changed = (
            self.owners.get(key) != candidate["physical"]
            or self.owner_ranks.get(key) != candidate["physical_ranks"]
            or self._cohort_owner_ranks.get(key) != candidate["cohort_owner_ranks"]
            or self.logical_owners.get(key) != candidate["demand"]
            or self.host_dram_owners.get(key) != candidate["host_physical"]
            or self.staging_owners.get(key) != candidate["physical_staging"]
            or self.request_query_tokens.get(key) != candidate["query_tokens"]
        )
        log_res, log_stg, _ = self._cohort_logical_maps()
        log_res[key] = candidate["cohort_logical_owner_ranks"]
        log_stg[key] = candidate["cohort_logical_staging_ranks"]
        self._cohort_owner_ranks[key] = candidate["cohort_owner_ranks"]
        self._cohort_staging_ranks[key] = candidate["cohort_staging_ranks"]
        self._set_physical(
            replica, key, candidate["physical"], candidate["host_physical"],
            candidate["physical_staging"],
            physical_ranks=candidate["physical_ranks"],
            staging_ranks=candidate["physical_staging_ranks"],
        )
        self.logical_owners[key] = candidate["demand"]
        self.logical_owner_ranks[key] = candidate["demand_ranks"]
        self.logical_host_owners[key] = candidate["host_demand"]
        self.logical_staging_owners[key] = candidate["demand_staging"]
        self.logical_staging_rank[key] = candidate["demand_staging_ranks"]
        eff_valid = int(tokens) if valid_tokens is None else max(0, min(int(tokens), int(valid_tokens)))
        self.request_tokens[key] = int(tokens)
        self.request_valid_tokens[key] = eff_valid
        self.request_phases[key] = str(phase or "decode")
        self.request_query_tokens[key] = int(candidate["query_tokens"])
        self.request_versions[key] = self.request_versions.get(key, 0) + 1
        if changed:
            self.budget_version += 1
        return True

    def hold_for_async_consumer(
        self, replica: int, request: str, consumer: str, *,
        storage: str | None = None, ranges=None, generation: int | None = None,
        event=None, grant: str | None = None, rank: int | None = None,
    ) -> bool:
        key = (replica, request)
        log_res, log_stg, cohort_holds = self._cohort_logical_maps()
        had_snap = grant is None and consumer in cohort_holds.get(key, {})
        ok = super().hold_for_async_consumer(
            replica, request, consumer, storage=storage, ranges=ranges,
            generation=generation, event=event, grant=grant, rank=rank,
        )
        if ok and grant is None and not had_snap and key in self._cohort_owner_ranks:
            snap_res = dict(log_res.get(key, self._cohort_owner_ranks[key]))
            snap_stg = dict(log_stg.get(key, self._cohort_staging_ranks.get(key, {})))
            cohort_holds.setdefault(key, {})[consumer] = (snap_res, snap_stg)
        return ok

    def release(self, replica: int, request: str) -> None:
        key = (replica, request)
        log_res, log_stg, cohort_holds = self._cohort_logical_maps()
        if key in self._cohort_owner_ranks and self.async_consumer_holds.get(key):
            p = self.case.runtime.parallel
            zeros = tuple((0.0,) * self.width for _ in range(p.pp))
            log_res[key] = {rep: zeros for rep in self._cohort_owner_ranks[key]}
            log_stg[key] = {rep: zeros for rep in self._cohort_staging_ranks.get(key, self._cohort_owner_ranks[key])}
            snaps = cohort_holds.get(key, {})
            if snaps:
                self._cohort_owner_ranks[key] = self._union_cohort_maps(
                    log_res[key], [s[0] for s in snaps.values()],
                )
                self._cohort_staging_ranks[key] = self._union_cohort_maps(
                    log_stg[key], [s[1] for s in snaps.values()],
                )
        super().release(replica, request)

    def _free_request(self, replica: int, request: str) -> None:
        key = (replica, request)
        before_hbm, before_host = self._state_usage()
        sizes = self.owners.get(key, (0.0,) * len(self.weights))
        events_before = len(self.release_events)
        log_res, log_stg, cohort_holds = self._cohort_logical_maps()
        self._cohort_owner_ranks.pop(key, None)
        self._cohort_staging_ranks.pop(key, None)
        log_res.pop(key, None)
        log_stg.pop(key, None)
        cohort_holds.pop(key, None)
        super()._free_request(replica, request)
        after_hbm, after_host = self._state_usage()
        freed_all = self._freed_total_bytes(before_hbm, before_host, after_hbm, after_host)
        if len(self.release_events) > events_before:
            self.release_events[-1]["released_bytes"] = freed_all
        elif freed_all > 0.0:
            self.budget_version += 1
            self.release_events.append({
                "kind": "request", "replica": replica, "request": request,
                "stages": sizes, "released_bytes": freed_all,
            })

    def release_async_consumer(
        self, replica: int, request: str, consumer: str, *,
        generation: int | None = None, event=None, rank: int | None = None,
    ) -> bool:
        key = (replica, request)
        before_hbm, before_host = self._state_usage()
        events_before = len(self.release_events)
        ok = super().release_async_consumer(
            replica, request, consumer, generation=generation, event=event, rank=rank,
        )
        if not ok:
            return False
        log_res, log_stg, cohort_holds = self._cohort_logical_maps()
        snaps = cohort_holds.get(key)
        if snaps is not None:
            snaps.pop(consumer, None)
            if not snaps:
                cohort_holds.pop(key, None)
        if key in self._cohort_owner_ranks:
            rem_snaps = cohort_holds.get(key, {})
            new_res = self._union_cohort_maps(
                log_res.get(key, {}), [s[0] for s in rem_snaps.values()],
            )
            new_stg = self._union_cohort_maps(
                log_stg.get(key, {}), [s[1] for s in rem_snaps.values()],
            )
            self._cohort_owner_ranks[key] = new_res
            self._cohort_staging_ranks[key] = new_stg
            if int(replica) in new_res:
                own_res = new_res[int(replica)]
                own_stg = new_stg.get(int(replica), self.staging_owner_ranks.get(key))
                self.owner_ranks[key] = own_res
                self.owners[key] = tuple(max(row) for row in own_res)
                if own_stg is not None:
                    self.staging_owner_ranks[key] = own_stg
                    self.staging_owners[key] = tuple(max(row) for row in own_stg)
            self._sync_usage()
        after_hbm, after_host = self._state_usage()
        freed_all = self._freed_total_bytes(before_hbm, before_host, after_hbm, after_host)
        if len(self.release_events) > events_before:
            self.release_events[-1]["released_bytes"] = freed_all
        return True

    def _add_typed_domain_usage(
        self,
        backing_ranges: dict[tuple, list],
        host_ranges: dict[tuple, list],
        hbm: list[list[list[float]]],
        host: list[float],
        *,
        skip_reclaimable: bool = False,
    ) -> None:
        """Fold live non-mirrored ``self.ledger_domain`` grants & reader pins into usage."""
        if self.ledger_domain is None:
            return
        mirrored_allocs, mirrored_readers = self._mirrored_typed_keys()
        backing_storage: dict[BackingKey, str] = {}
        for grant in self.ledger_domain.grants.values():
            if grant.released or grant.key in mirrored_allocs:
                continue
            if skip_reclaimable and (grant.deferred or grant.last_use is not None):
                continue
            parsed = self._parse_ledger_pool(grant.pool)
            if parsed is None:
                continue
            tier_kind, rep, st, rk = parsed
            if grant.backing is None:
                if grant.acquire:
                    if tier_kind == "host":
                        host[rep] += float(grant.amount)
                    else:
                        hbm[rep][st][rk] += float(grant.amount)
                continue
            storage_str = str(grant.storage or "")
            backing_storage.setdefault(grant.backing, storage_str)
            if grant.acquire and grant.ranges:
                pairs = [(float(lo), float(hi)) for lo, hi in grant.ranges]
                gen = _hashable_generation(grant.backing.generation)
                if tier_kind == "host":
                    host_ranges.setdefault(
                        (rep, 0, st, storage_str, str(grant.backing.storage_id), gen), []
                    ).extend(pairs)
                else:
                    backing_ranges.setdefault(
                        (rep, rk, st, storage_str, tier_kind, str(grant.backing.storage_id), gen), []
                    ).extend(pairs)
        for backing_rec in self.ledger_domain.backings.values():
            parsed = self._parse_ledger_pool(backing_rec.pool)
            if parsed is None:
                continue
            tier_kind, rep, st, rk = parsed
            storage_str = backing_storage.get(backing_rec.key, "")
            gen = _hashable_generation(backing_rec.key.generation)
            for reader_key, r_ranges in backing_rec.reader_pins.items():
                if reader_key in mirrored_readers or not r_ranges:
                    continue
                if skip_reclaimable:
                    continue
                pairs = [(float(lo), float(hi)) for lo, hi in r_ranges]
                if tier_kind == "host":
                    host_ranges.setdefault(
                        (rep, 0, st, storage_str, str(backing_rec.key.storage_id), gen), []
                    ).extend(pairs)
                else:
                    backing_ranges.setdefault(
                        (rep, rk, st, storage_str, tier_kind, str(backing_rec.key.storage_id), gen), []
                    ).extend(pairs)

    def _candidate_only_usage(self, extra_holders):
        """Evaluate HBM/host usage of ONLY ``extra_holders`` (all live holders released)."""
        p = self.case.runtime.parallel
        hbm = [[[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        host = [0.0] * p.dp
        backing_ranges: dict[tuple, list] = {}
        host_ranges: dict[tuple, list] = {}
        for holder, refs in extra_holders:
            replica = holder[0]
            for ref in refs:
                if ref.tier == "host":
                    if ref.host_ranges:
                        host_ranges.setdefault((
                            replica, ref.rank, ref.stage, ref.storage,
                            ref.physical_key, ref.generation), []).extend(ref.host_ranges)
                    continue
                if ref.tier not in {"hbm", ""}:
                    continue
                backing_ranges.setdefault((
                    replica, ref.rank, ref.stage, ref.storage, ref.tier,
                    ref.physical_key, ref.generation), []).extend(ref.ranges)
                if ref.host_ranges:
                    host_ranges.setdefault((
                        replica, ref.rank, ref.stage, ref.storage,
                        ref.physical_key, ref.generation), []).extend(ref.host_ranges)
        for (replica, rank, stage, _storage, _tier, _key, _generation), ranges in backing_ranges.items():
            hbm[replica][stage][rank] += _union_bytes(ranges)
        for (replica, _rank, _stage, _storage, _key, _generation), ranges in host_ranges.items():
            host[replica] += _union_bytes(ranges)
        return hbm, host

    def _freed_replica_hbm(self, replica: int, before_hbm, after_hbm) -> float:
        """Sum positive per-(stage, rank) physical HBM byte reductions on ``replica``."""
        p = self.case.runtime.parallel
        freed = 0.0
        for st in range(p.pp):
            for rk in range(self.width):
                freed += max(0.0, float(before_hbm[replica][st][rk]) - float(after_hbm[replica][st][rk]))
        return freed

    def _state_usage(
        self, *, extra_holders=(), drop_holders=(), exclude_request=None,
        request_usage=None, staging_usage=None, host_request=None,
        include_typed_domain: bool = True,
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
        if include_typed_domain:
            self._add_typed_domain_usage(backing_ranges, host_ranges, hbm, host)
        for (replica, rank, stage, _storage, _tier, _key, _generation), ranges in \
                backing_ranges.items():
            hbm[replica][stage][rank] += _union_bytes(ranges)
        self._accumulate_owner_map(
            hbm, self.owner_ranks, self._cohort_owner_ranks, exclude_request=exclude_request,
        )
        self._accumulate_owner_map(
            hbm, self.staging_owner_ranks, self._cohort_staging_ranks, exclude_request=exclude_request,
        )
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
        if self.ledger_domain is not None:
            mirrored_allocs, mirrored_readers = self._mirrored_typed_keys()
            backing_storage: dict[BackingKey, str] = {}
            for grant in self.ledger_domain.grants.values():
                if grant.released or grant.key in mirrored_allocs:
                    continue
                parsed = self._parse_ledger_pool(grant.pool)
                if parsed is None or parsed[0] != "hbm":
                    continue
                _tk, rep, st, rk = parsed
                if grant.backing is None:
                    if grant.acquire:
                        granted[rep][st][rk] += float(grant.amount)
                    continue
                storage_str = str(grant.storage or "")
                backing_storage.setdefault(grant.backing, storage_str)
                if grant.acquire and grant.ranges:
                    key = (rep, rk, st, storage_str, "hbm",
                           str(grant.backing.storage_id),
                           _hashable_generation(grant.backing.generation))
                    entry = backings.setdefault(key, {"ranges": [], "kinds": set()})
                    entry["ranges"].extend((float(lo), float(hi)) for lo, hi in grant.ranges)
                    entry["kinds"].add("grant")
            for backing_rec in self.ledger_domain.backings.values():
                parsed = self._parse_ledger_pool(backing_rec.pool)
                if parsed is None or parsed[0] != "hbm":
                    continue
                _tk, rep, st, rk = parsed
                storage_str = backing_storage.get(backing_rec.key, "")
                gen = _hashable_generation(backing_rec.key.generation)
                for reader_key, r_ranges in backing_rec.reader_pins.items():
                    if reader_key in mirrored_readers or not r_ranges:
                        continue
                    key = (rep, rk, st, storage_str, "hbm",
                           str(backing_rec.key.storage_id), gen)
                    entry = backings.setdefault(key, {"ranges": [], "kinds": set()})
                    entry["ranges"].extend((float(lo), float(hi)) for lo, hi in r_ranges)
                    entry["kinds"].add("reader")
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
                stage_hbm = hbm[replica][stage]
                stage_req = request[replica][stage]
                stage_stg = staging[replica][stage]
                for rank_idx, die_used in enumerate(stage_hbm):
                    self.peak_rank[replica][stage][rank_idx] = max(
                        self.peak_rank[replica][stage][rank_idx], float(die_used),
                    )
                total = max(stage_hbm, default=0.0)
                resident = max(stage_req, default=0.0)
                stage_staging = max(stage_stg, default=0.0)
                # True simultaneous per-die peak: NEVER sum resident and staging
                # maxima that occurred on different physical ranks.
                simultaneous_req_stage = max(
                    (float(r) + float(s) for r, s in zip(stage_req, stage_stg)),
                    default=0.0,
                )
                self.peak_combined[replica][stage] = max(
                    self.peak_combined[replica][stage], total)
                self.peak[replica][stage] = max(
                    self.peak[replica][stage], simultaneous_req_stage)
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
        self._sync_domain_occupancy()

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

    # ------------------------------------------------------------------
    # Prepared weights, authoritative per-die receipt & HBM headroom
    # ------------------------------------------------------------------
    def _stage_prepared_weight_totals(
        self,
        specs: Sequence[PreparedWeightSpec],
    ) -> tuple[list[int], list[int], list[list[ResidentGrantEntry]]]:
        """Project ``PreparedWeightSpec`` allocations onto PP stages, billing aliases once."""
        p = self.case.runtime.parallel
        pp = max(1, int(p.pp))
        persistent = [0] * pp
        load_peak = [0] * pp
        per_stage_grants: list[list[ResidentGrantEntry]] = [[] for _ in range(pp)]
        seen_backings: dict[tuple[int, str], int] = {}
        for spec in specs:
            if not isinstance(spec, PreparedWeightSpec):
                raise ValueError("memory_prepared_weight_spec_invalid")
            raw_bytes = (
                exact_element_bytes(math.prod(spec.local_shape), spec.dtype)
                if spec.local_shape else 0
            ) + int(spec.extra_metadata_bytes)
            target_stages: list[int] = []
            if spec.layer_indices:
                for layer_idx in spec.layer_indices:
                    st = (
                        self.layer_stage[layer_idx]
                        if 0 <= layer_idx < len(self.layer_stage) else pp - 1
                    )
                    target_stages.append(st)
            else:
                target_stages.append(pp - 1 if spec.scope == "draft" else 0)
            for idx, st in enumerate(target_stages):
                backing = spec.alias_of or f"{spec.allocation_id}:s{st}:i{idx}"
                if spec.alias_of:
                    billed = 0
                else:
                    key = (st, backing)
                    prev = seen_backings.get(key, 0)
                    billed = max(0, raw_bytes - prev)
                    if raw_bytes > prev:
                        seen_backings[key] = raw_bytes
                if spec.lifetime == "load_peak_only":
                    load_peak[st] += billed
                    billed_steady = 0
                else:
                    persistent[st] += billed
                    billed_steady = billed
                per_stage_grants[st].append(ResidentGrantEntry(
                    grant_id=f"prepared:{spec.allocation_id}:s{st}:i{idx}",
                    category="prepared_weights",
                    physical_backing_id=backing,
                    alias_of=spec.alias_of,
                    generation=0,
                    tier="hbm",
                    checkpoint_bytes=0,
                    runtime_bytes=raw_bytes,
                    allocated_bytes=raw_bytes,
                    valid_bytes=raw_bytes,
                    lifetime=spec.lifetime,
                    billed_once_bytes=billed_steady,
                ))
        return persistent, load_peak, per_stage_grants

    def register_prepared_weights(self, specs: Sequence[PreparedWeightSpec]) -> None:
        """Register post-load ``PreparedWeightSpec`` copies on the unified ledger."""
        for spec in specs:
            if not isinstance(spec, PreparedWeightSpec):
                raise ValueError("memory_prepared_weight_spec_invalid")
            self._prepared_weights[spec.allocation_id] = spec
        persistent, load_peak, _ = self._stage_prepared_weight_totals(
            tuple(self._prepared_weights.values()),
        )
        self._prepared_persistent_by_stage = [float(v) for v in persistent]
        self._prepared_load_peak_by_stage = [float(v) for v in load_peak]
        p = self.case.runtime.parallel
        for replica in range(p.dp):
            for stage in range(p.pp):
                for rank in range(self.width):
                    self.rank_capacity[replica][stage][rank] = (
                        self._base_rank_capacity[replica][stage][rank]
                        - self._prepared_persistent_by_stage[stage]
                    )
            self.capacity[replica] = [
                min(row) for row in self.rank_capacity[replica]
            ]
        self._sync_domain_occupancy()
        self.budget_version += 1

    @staticmethod
    def _describe_release_event(event: Any) -> str:
        if event is None:
            return ""
        if isinstance(event, str):
            return event.strip()
        seq = getattr(event, "sequence", None)
        if seq is not None:
            return f"event:{seq}"
        return repr(event)

    def _collect_reclaimable_holders_and_events(self):
        """Identify live holders backed by explicit release events/boundaries."""
        p = self.case.runtime.parallel
        drop_holders: set = set()
        events_by_die = [
            [[set() for _ in range(self.width)] for _ in range(p.pp)]
            for _ in range(p.dp)
        ]

        def _add_ev(rep: int, st: int, rk: int, label: str) -> None:
            if label and 0 <= rep < p.dp and 0 <= st < p.pp and 0 <= rk < self.width:
                events_by_die[rep][st][rk].add(label)

        for (rep, name), binding in self.grant_bindings.items():
            ev_desc = self._describe_release_event(binding.get("event"))
            if not ev_desc:
                continue
            holder = (rep, "grant", name)
            drop_holders.add(holder)
            for ref in self._refs.get(holder, ()):
                if ref.tier in {"hbm", ""} and ref.ranges:
                    _add_ev(rep, ref.stage, ref.rank, f"grant:{name}:{ev_desc}")
        for (rep, grant), readers in self.grant_readers.items():
            for consumer, snap in readers.items():
                ev_desc = self._describe_release_event(snap.get("event"))
                if not ev_desc:
                    continue
                holder = (rep, "grant_reader", f"{grant}::{consumer}")
                drop_holders.add(holder)
                for ref in self._refs.get(holder, ()):
                    if ref.tier in {"hbm", ""} and ref.ranges:
                        _add_ev(rep, ref.stage, ref.rank, f"grant_reader:{grant}:{consumer}:{ev_desc}")
        for (rep, st, name), binding in self.effect_bindings.items():
            ev_desc = self._describe_release_event(binding.get("event"))
            if not ev_desc:
                continue
            holder = (rep, "selected", st, name)
            drop_holders.add(holder)
            for ref in self._refs.get(holder, ()):
                if ref.tier in {"hbm", ""} and ref.ranges:
                    _add_ev(rep, ref.stage, ref.rank, f"selected:{name}:{ev_desc}")
        if self.ledger_domain is not None:
            mirrored_allocs, _mirrored_readers = self._mirrored_typed_keys()
            for grant in self.ledger_domain.grants.values():
                if grant.released or grant.key in mirrored_allocs:
                    continue
                if not (grant.deferred or grant.last_use is not None):
                    continue
                parsed = self._parse_ledger_pool(grant.pool)
                if parsed is None or parsed[0] != "hbm":
                    continue
                _tk, rep, st, rk = parsed
                ev_label = (
                    f"typed_grant:{grant.key.symbol}:{grant.last_use}"
                    if grant.last_use is not None else f"typed_grant:{grant.key.symbol}:deferred"
                )
                _add_ev(rep, st, rk, ev_label)
        return drop_holders, events_by_die

    @staticmethod
    def _record_nonzero_rank_events(
        events_by_die, replica: int, rows, labels: Sequence[str],
    ) -> None:
        for st, row in enumerate(rows):
            for rk, val in enumerate(row):
                if float(val) > 0.0:
                    events_by_die[replica][st][rk].update(labels)

    @staticmethod
    def _add_rank_rows(target, replica: int, rows) -> None:
        for st, row in enumerate(rows):
            for rk, val in enumerate(row):
                target[replica][st][rk] += float(val)

    def _split_request_async_holds(self, key: tuple[int, str]):
        """Split ``async_consumer_holds[key]`` into unbounded vs explicit event-bound holds."""
        _rep, req = key
        holds = self.async_consumer_holds.get(key, {})
        unbounded: list[tuple[str, Any]] = []
        event_labels: list[str] = []
        for cons, snap in sorted(holds.items()):
            ev_desc = self._describe_release_event(dict(snap.binding).get("event"))
            if ev_desc:
                event_labels.append(f"async_consumer:{req}:{cons}:{ev_desc}")
            else:
                unbounded.append((cons, snap))
        return unbounded, tuple(event_labels)

    def _max_rank_rows(self, rows_list) -> list[list[float]]:
        p = self.case.runtime.parallel
        out = [[0.0] * self.width for _ in range(p.pp)]
        for rows in rows_list:
            for st, row in enumerate(rows):
                for rk, val in enumerate(row):
                    fv = float(val)
                    if fv > out[st][rk]:
                        out[st][rk] = fv
        return out

    def _record_reclaim_diff_events(
        self, events_by_die, replica: int, phys_rows, ret_rows, labels: Sequence[str],
    ) -> None:
        if not labels:
            return
        for st, (p_row, r_row) in enumerate(zip(phys_rows, ret_rows)):
            for rk, (pv, rv) in enumerate(zip(p_row, r_row)):
                if float(pv) > float(rv) + 1e-9:
                    events_by_die[replica][st][rk].update(labels)

    def _fold_single_request_retention(
        self, key: tuple[int, str], phys_rows, retained_hbm, events_by_die,
    ) -> None:
        """Retain logical + eventless hold ranges for one request and collect bound events."""
        rep, req = key
        p = self.case.runtime.parallel
        zeros = tuple((0.0,) * self.width for _ in range(p.pp))
        cohort_map = self._cohort_owner_ranks.get(key)
        stg_cohort = self._cohort_staging_ranks.get(key)
        log_res_all, log_stg_all, cohort_holds = self._cohort_logical_maps()
        unbounded, ev_labels = self._split_request_async_holds(key)
        if key in self.delayed_owners:
            rem, _sizes = self.delayed_owners[key]
            labels = (f"delayed_release:{req}:steps={int(rem)}",) + ev_labels
            if not unbounded:
                if cohort_map is not None:
                    for peer_rep, peer_rows in cohort_map.items():
                        self._record_nonzero_rank_events(events_by_die, peer_rep, peer_rows, labels)
                else:
                    self._record_nonzero_rank_events(events_by_die, rep, phys_rows, labels)
                return
        else:
            labels = ev_labels
        retired = key in self.retired_requests or key in self.delayed_owners
        if cohort_map is not None:
            base_res = {r: zeros for r in cohort_map} if retired else log_res_all.get(key, cohort_map)
            base_stg = (
                {r: zeros for r in (stg_cohort or cohort_map)}
                if retired else log_stg_all.get(key, stg_cohort or {})
            )
            snaps = cohort_holds.get(key, {})
            unb_res = [snaps[c][0] for c, _ in unbounded if c in snaps]
            unb_stg = [snaps[c][1] for c, _ in unbounded if c in snaps]
            ret_res = self._union_cohort_maps(base_res, unb_res)
            ret_stg = self._union_cohort_maps(base_stg, unb_stg)
            for peer_rep, peer_rows in cohort_map.items():
                r_rows = ret_res.get(peer_rep, zeros)
                s_rows = ret_stg.get(peer_rep, zeros)
                self._add_rank_rows(retained_hbm, peer_rep, r_rows)
                self._add_rank_rows(retained_hbm, peer_rep, s_rows)
                p_stg = stg_cohort.get(peer_rep, zeros) if stg_cohort is not None else zeros
                phys_sum = [[float(a) + float(b) for a, b in zip(pr, ps)] for pr, ps in zip(peer_rows, p_stg)]
                ret_sum = [[float(a) + float(b) for a, b in zip(rr, rs)] for rr, rs in zip(r_rows, s_rows)]
                self._record_reclaim_diff_events(events_by_die, peer_rep, phys_sum, ret_sum, labels)
            return
        base_r = zeros if retired else self.logical_owner_ranks.get(key, phys_rows)
        stg_rows = self.staging_owner_ranks.get(key, zeros)
        base_s = zeros if retired else self.logical_staging_rank.get(key, stg_rows)
        unb_r = [snap.rank_stages or self._replicate_ranks(snap.stages) for _, snap in unbounded]
        unb_s = [snap.rank_staging or self._replicate_ranks(snap.staging) for _, snap in unbounded]
        ret_r = self._max_rank_rows((base_r, *unb_r))
        ret_s = self._max_rank_rows((base_s, *unb_s))
        self._add_rank_rows(retained_hbm, rep, ret_r)
        self._add_rank_rows(retained_hbm, rep, ret_s)
        phys_sum = [[float(a) + float(b) for a, b in zip(pr, ps)] for pr, ps in zip(phys_rows, stg_rows)]
        ret_sum = [[float(a) + float(b) for a, b in zip(rr, rs)] for rr, rs in zip(ret_r, ret_s)]
        self._record_reclaim_diff_events(events_by_die, rep, phys_sum, ret_sum, labels)

    def _fold_request_reclaimable_retention(self, retained_hbm, events_by_die) -> None:
        """Retain non-reclaimable request demand and record explicit release events."""
        for key, phys_rows in self.owner_ranks.items():
            self._fold_single_request_retention(key, phys_rows, retained_hbm, events_by_die)

    @staticmethod
    def _norm_receipt_gen(value) -> Any:
        if value is None:
            return 0
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        return _hashable_generation(value)

    def _collect_non_request_backings(self):
        """Per-die live & retained non-request HBM backing maps and explicit release events."""
        p = self.case.runtime.parallel
        drop_holders, _holder_events = self._collect_reclaimable_holders_and_events()
        per_die = [
            [
                [
                    {
                        "unbacked_live": 0.0,
                        "unbacked_ret": 0.0,
                        "unbacked_events": set(),
                        "all_ranges": {},
                        "ret_ranges": {},
                        "events": {},
                    }
                    for _ in range(self.width)
                ]
                for _ in range(p.pp)
            ]
            for _ in range(p.dp)
        ]
        holder_ev_label: dict[Any, str] = {}
        for (rep, name), binding in self.grant_bindings.items():
            ev_desc = self._describe_release_event(binding.get("event"))
            if ev_desc:
                holder_ev_label[(rep, "grant", name)] = f"grant:{name}:{ev_desc}"
        for (rep, grant), readers in self.grant_readers.items():
            for consumer, snap in readers.items():
                ev_desc = self._describe_release_event(snap.get("event"))
                if ev_desc:
                    holder_ev_label[(rep, "grant_reader", f"{grant}::{consumer}")] = (
                        f"grant_reader:{grant}:{consumer}:{ev_desc}"
                    )
        for (rep, st, name), binding in self.effect_bindings.items():
            ev_desc = self._describe_release_event(binding.get("event"))
            if ev_desc:
                holder_ev_label[(rep, "selected", st, name)] = f"selected:{name}:{ev_desc}"

        for holder, refs in self._refs.items():
            rep = holder[0]
            is_drop = holder in drop_holders
            ev_lbl = holder_ev_label.get(holder, "")
            for ref in refs:
                if ref.tier not in {"hbm", ""} or not ref.ranges:
                    continue
                slot = per_die[rep][ref.stage][ref.rank]
                bkey = (
                    str(ref.storage or ""),
                    str(ref.physical_key or ""),
                    self._norm_receipt_gen(ref.generation),
                    str(ref.tier or "hbm").lower() or "hbm",
                )
                slot["all_ranges"].setdefault(bkey, []).extend(ref.ranges)
                if not is_drop:
                    slot["ret_ranges"].setdefault(bkey, []).extend(ref.ranges)
                elif ev_lbl:
                    slot["events"].setdefault(bkey, set()).add(ev_lbl)

        if self.ledger_domain is not None:
            self._fold_typed_domain_non_request_backings(per_die)
        return per_die

    def _fold_typed_domain_non_request_backings(self, per_die) -> None:
        """Fold non-mirrored ``ledger_domain`` grants and reader pins into ``per_die``."""
        mirrored_allocs, mirrored_readers = self._mirrored_typed_keys()
        backing_storage: dict[BackingKey, str] = {}
        for grant in self.ledger_domain.grants.values():
            if grant.released or grant.key in mirrored_allocs:
                continue
            parsed = self._parse_ledger_pool(grant.pool)
            if parsed is None or parsed[0] != "hbm":
                continue
            _tk, rep, st, rk = parsed
            slot = per_die[rep][st][rk]
            is_recl = bool(grant.deferred or grant.last_use is not None)
            ev_lbl = (
                f"typed_grant:{grant.key.symbol}:{grant.last_use}"
                if grant.last_use is not None else f"typed_grant:{grant.key.symbol}:deferred"
            )
            if grant.backing is None:
                if grant.acquire:
                    slot["unbacked_live"] += float(grant.amount)
                    if not is_recl:
                        slot["unbacked_ret"] += float(grant.amount)
                    else:
                        slot["unbacked_events"].add(ev_lbl)
                continue
            storage_str = str(grant.storage or "")
            backing_storage.setdefault(grant.backing, storage_str)
            if grant.acquire and grant.ranges:
                pairs = [(float(lo), float(hi)) for lo, hi in grant.ranges]
                bkey = (storage_str, str(grant.backing.storage_id), self._norm_receipt_gen(grant.backing.generation), "hbm")
                slot["all_ranges"].setdefault(bkey, []).extend(pairs)
                if not is_recl:
                    slot["ret_ranges"].setdefault(bkey, []).extend(pairs)
                else:
                    slot["events"].setdefault(bkey, set()).add(ev_lbl)
        for backing_rec in self.ledger_domain.backings.values():
            parsed = self._parse_ledger_pool(backing_rec.pool)
            if parsed is None or parsed[0] != "hbm":
                continue
            _tk, rep, st, rk = parsed
            slot = per_die[rep][st][rk]
            storage_str = backing_storage.get(backing_rec.key, "")
            bkey = (storage_str, str(backing_rec.key.storage_id), self._norm_receipt_gen(backing_rec.key.generation), "hbm")
            for reader_key, r_ranges in backing_rec.reader_pins.items():
                if reader_key in mirrored_readers or not r_ranges:
                    continue
                pairs = [(float(lo), float(hi)) for lo, hi in r_ranges]
                slot["all_ranges"].setdefault(bkey, []).extend(pairs)
                slot["events"].setdefault(bkey, set()).add(f"typed_reader:{reader_key}")

    def _request_reclaimable_usage_and_events(self):
        """Per-die request-only ``(reclaimable_bytes, sorted_events)`` from active/retired requests."""
        p = self.case.runtime.parallel
        req_live = self._request_usage_ranks()
        stg_live = self._staging_usage_ranks()
        retained_req = [[[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        events_by_die = [
            [[set() for _ in range(self.width)] for _ in range(p.pp)]
            for _ in range(p.dp)
        ]
        self._fold_request_reclaimable_retention(retained_req, events_by_die)
        reclaimable = [
            [
                [
                    max(0, int(round(
                        float(req_live[rep][st][rk]) + float(stg_live[rep][st][rk])
                        - float(retained_req[rep][st][rk])
                    )))
                    for rk in range(self.width)
                ]
                for st in range(p.pp)
            ]
            for rep in range(p.dp)
        ]
        sorted_events = [
            [
                [
                    tuple(sorted(events_by_die[rep][st][rk])) if reclaimable[rep][st][rk] > 0 else ()
                    for rk in range(self.width)
                ]
                for st in range(p.pp)
            ]
            for rep in range(p.dp)
        ]
        return reclaimable, sorted_events

    def _reclaimable_usage_and_events(self):
        """Per-die ``(reclaimable_hbm_bytes, sorted_release_events)`` via exact backing union."""
        p = self.case.runtime.parallel
        req_recl, req_events = self._request_reclaimable_usage_and_events()
        non_req = self._collect_non_request_backings()
        reclaimable = [[[0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        sorted_events = [[[() for _ in range(self.width)] for _ in range(p.pp)] for _ in range(p.dp)]
        for rep in range(p.dp):
            for st in range(p.pp):
                for rk in range(self.width):
                    slot = non_req[rep][st][rk]
                    nr_live = slot["unbacked_live"] + sum(_union_bytes(r) for r in slot["all_ranges"].values())
                    nr_ret = slot["unbacked_ret"] + sum(_union_bytes(r) for r in slot["ret_ranges"].values())
                    nr_recl = max(0, int(round(nr_live - nr_ret)))
                    total = int(req_recl[rep][st][rk]) + nr_recl
                    reclaimable[rep][st][rk] = total
                    if total <= 0:
                        continue
                    evs = set(req_events[rep][st][rk])
                    if slot["unbacked_live"] > slot["unbacked_ret"] + 1e-9:
                        evs.update(slot["unbacked_events"])
                    for bkey, b_evs in slot["events"].items():
                        if _union_bytes(slot["all_ranges"].get(bkey, ())) > _union_bytes(slot["ret_ranges"].get(bkey, ())) + 1e-9:
                            evs.update(b_evs)
                    sorted_events[rep][st][rk] = tuple(sorted(evs))
        return reclaimable, sorted_events

    def free_hbm_headroom_bytes(
        self,
        replica: int = 0,
        stage: int | None = None,
        rank: int | None = None,
        *,
        mode: str = "historical",
    ) -> int:
        """Authoritative free HBM headroom on the bottleneck (or addressed) die.

        ``mode`` selects the margin basis:
          * ``"historical"`` (default): ``usable - max(current_live, historical_peak)``
          * ``"immediate"``: ``usable - current_live``
          * ``"post_reclaim"``: ``usable - (current_live - reclaimable)``
        """
        if mode not in {"historical", "immediate", "post_reclaim"}:
            raise ValueError(f"memory_headroom_mode_invalid:{mode!r}")
        p = self.case.runtime.parallel
        if not 0 <= int(replica) < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        hbm, _ = self._state_usage()
        reclaimable = (
            self._reclaimable_usage_and_events()[0]
            if mode == "post_reclaim" else None
        )
        stages = (int(stage),) if stage is not None else tuple(range(p.pp))
        ranks = (self._account_rank(rank),) if rank is not None else tuple(range(self.width))
        headrooms: list[int] = []
        for st in stages:
            if not 0 <= st < p.pp:
                raise ValueError("memory_stage_out_of_range")
            for rk in ranks:
                cap = float(self.rank_capacity[replica][st][rk])
                live = float(hbm[replica][st][rk])
                if mode == "immediate":
                    used = live
                elif mode == "post_reclaim":
                    recl = float(reclaimable[replica][st][rk]) if reclaimable is not None else 0.0
                    used = max(0.0, live - recl)
                else:
                    used = max(
                        live,
                        float(self.peak_rank[replica][st][rk]),
                        self._domain_peak_hbm(replica, st, rk),
                    )
                headrooms.append(int(math.floor(cap - used)))
        return min(headrooms, default=0)

    def _resolve_vision_delta_by_stage(self, vision_accounting: str) -> tuple[list[int], list[int]]:
        """Return ``(vision_bytes_by_stage, weight_delta_by_stage)`` for vision scope."""
        from mindie_sim.catalog.draft_weights import require_model_quant
        from mindie_sim.catalog.weight_terms import term_bytes
        from mindie_sim.catalog.weights import vision_weight_terms

        p = self.case.runtime.parallel
        pp = max(1, int(p.pp))
        if vision_accounting not in VISION_ACCOUNTING_SCOPES:
            raise ValueError(f"vision_accounting_invalid:{vision_accounting!r}")
        model = self.case.model
        quant = require_model_quant(model)
        vision_all = int(round(sum(
            term_bytes(term, quant, p, self.case.machine.ranks_per_node)
            for term in vision_weight_terms(
                model, include_vision=True, parallel=p, runtime=self.case.runtime,
            )
        )))
        currently_included = (self.vision_accounting == "full_checkpoint") and (vision_all > 0)
        want_included = (vision_accounting == "full_checkpoint") and (vision_all > 0)
        vision_stage = [0] * pp
        delta_stage = [0] * pp
        if want_included:
            vision_stage[0] = vision_all
        if want_included and not currently_included:
            delta_stage[0] = vision_all
        elif currently_included and not want_included:
            delta_stage[0] = -vision_all
        return vision_stage, delta_stage

    def _stage_checkpoint_vs_prepared_split(
        self,
        stage: int,
        *,
        ep_rank: int | None = None,
        weight_bytes: float | None = None,
        all_stage_terms: Sequence[Sequence[Any]] | None = None,
    ) -> tuple[int, int]:
        """Split resident stage weights into source checkpoint vs runtime-only prepared bytes."""
        from mindie_sim.catalog.draft_weights import require_model_quant
        from mindie_sim.catalog.weight_terms import term_bytes
        from mindie_sim.catalog.weights import _expand_resident_terms, stage_weight_terms

        model = self.case.model
        p = self.case.runtime.parallel
        quant = require_model_quant(model)
        if all_stage_terms is None:
            include_vision = self.vision_accounting == "full_checkpoint"
            all_stage_terms = stage_weight_terms(
                model, p,
                include_draft=bool(self.draft_kind),
                draft_kind=self.draft_kind or None,
                include_engram_tables=not self.case.runtime.engram_cpu_offload,
                include_vision=include_vision,
                layer_partition=self.layers,
                slot_plans=self.slot_plans,
                ep_rank=ep_rank,
                runtime=self.case.runtime,
            )
        shared_dp = bool(getattr(self, "enable_shared_expert_dp", False))
        pcp_o_proj = bool(getattr(self, "enable_pcp_o_proj_weight_sharding", False))
        dcp_q_rep = bool(getattr(self, "dcp_q_replicate", False))
        off_ratio = float(getattr(self, "offload", {}).get("expert_ddr_ratio", 0.0) or 0.0)
        runtime_only = 0.0
        for term in all_stage_terms[stage]:
            if term.source_owned:
                continue
            expanded = _expand_resident_terms(
                term, p, shared_dp,
                pcp_o_proj_weight_sharding=pcp_o_proj,
                dcp_q_replicate=dcp_q_rep,
                model=model,
            )
            for t_sub, placed in expanded:
                tb = term_bytes(t_sub, quant, placed, self.case.machine.ranks_per_node)
                if off_ratio > 0.0 and t_sub.category == "routed_expert":
                    tb *= 1.0 - off_ratio
                runtime_only += tb
        total_w = max(0, int(round(self.weights[stage] if weight_bytes is None else weight_bytes)))
        prep_b = min(total_w, max(0, int(round(runtime_only))))
        ckpt_b = max(0, total_w - prep_b)
        return ckpt_b, prep_b

    def _resolve_die_weight_splits(
        self,
        vision_stage: Sequence[int],
        vision_delta: Sequence[int],
    ) -> dict[tuple[int, int, int], tuple[int, int]]:
        """Resolve per-die ``(ckpt_b, term_prep_b)`` from per-die ``rank_weights`` and EP rank."""
        from mindie_sim.catalog.weights import stage_weight_terms
        from mindie_sim.schema.common import ModelingError

        p = self.case.runtime.parallel
        weights_by_ep = tuple(getattr(self, "weights_by_ep_rank", ()) or ())
        multi_ep = len(weights_by_ep) > 1
        placement = None
        if multi_ep:
            try:
                placement = self.case.machine.validate_placement(p)
            except ModelingError:
                placement = None
        default_ep = 0 if self.slot_plans is not None else None
        include_vision = self.vision_accounting == "full_checkpoint"
        rank_weights = getattr(self, "rank_weights", None)
        terms_by_ep: dict[int | None, list[list[Any]]] = {}
        split_cache: dict[tuple[int, int | None, int], tuple[int, int]] = {}
        per_die: dict[tuple[int, int, int], tuple[int, int]] = {}
        for replica in range(p.dp):
            for stage in range(p.pp):
                for rank in range(self.width):
                    ep_id = (
                        self._ep_rank_for_die(replica, stage, rank, placement)
                        if multi_ep else default_ep
                    )
                    die_w = (
                        float(rank_weights[replica][stage][rank])
                        if (rank_weights is not None and replica < len(rank_weights)
                            and stage < len(rank_weights[replica])
                            and rank < len(rank_weights[replica][stage]))
                        else float(self.weights[stage])
                    )
                    die_w_int = max(0, int(round(die_w)))
                    skey = (stage, ep_id, die_w_int)
                    split = split_cache.get(skey)
                    if split is None:
                        all_terms = terms_by_ep.get(ep_id)
                        if all_terms is None:
                            all_terms = stage_weight_terms(
                                self.case.model, p,
                                include_draft=bool(self.draft_kind),
                                draft_kind=self.draft_kind or None,
                                include_engram_tables=not self.case.runtime.engram_cpu_offload,
                                include_vision=include_vision,
                                layer_partition=self.layers,
                                slot_plans=self.slot_plans,
                                ep_rank=ep_id,
                                runtime=self.case.runtime,
                            )
                            terms_by_ep[ep_id] = all_terms
                        ck, tp_b = self._stage_checkpoint_vs_prepared_split(
                            stage, ep_rank=ep_id, weight_bytes=die_w, all_stage_terms=all_terms,
                        )
                        split = (max(0, ck + vision_delta[stage] - vision_stage[stage]), tp_b)
                        split_cache[skey] = split
                    per_die[(replica, stage, rank)] = split
        return per_die

    def _unified_pool_stride_reserve_by_stage(
        self,
        sequence_tokens: int,
        *,
        query_tokens: int | None = None,
    ) -> list[float]:
        """Per-stage per-request unified hybrid-pool stride alignment reserve (all 4 regimes)."""
        p = self.case.runtime.parallel
        seq = int(sequence_tokens)
        if seq <= 0:
            return [0.0] * p.pp
        receipt = self._hybrid_pool_receipt(seq, num_requests=1, query_tokens=query_tokens)
        if receipt is None:
            return [0.0] * p.pp
        return [float(v) for v in receipt.stride_reserve_by_stage]

    def _projected_die_cache_pools(
        self,
        sequence_tokens: int,
        batch_per_dp: int,
        query_tokens: int | None,
        *,
        rank: int | None = None,
        phase: str = "decode",
        include_dp_cohort_extra: bool = True,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Exact per-stage integer cache pools and component breakdowns for one physical rank."""
        from .cache import _COMPONENT_POOL_MAP

        p = self.case.runtime.parallel
        seq = int(sequence_tokens)
        b_int = int(batch_per_dp)
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        hybrid_b = (
            self._hybrid_pool_receipt(seq, num_requests=b_int, query_tokens=q_tok, phase=phase)
            if include_dp_cohort_extra else None
        )
        hybrid_1 = (
            self._hybrid_pool_receipt(seq, num_requests=1, query_tokens=q_tok, phase=phase)
            if hybrid_b is not None else None
        )
        eff_block = hybrid_b.contract.normalized_block_size if hybrid_b is not None else None
        alloc_stages = self._cache_component_breakdown(
            seq, padded=True, skip_covered=True, rank=rank, query_tokens=q_tok,
            block_size_override=eff_block, phase=phase,
        )
        valid_stages = self._cache_component_breakdown(
            seq, padded=False, skip_covered=True, rank=rank, query_tokens=q_tok,
            block_size_override=eff_block, phase=phase,
        )
        group_parts = (
            self._group_cache_parts(seq, rank=rank, query_tokens=q_tok, phase=phase)
            if self.pages.groups else None
        )
        group_payload = (
            self.pages.stage_bytes(seq, self.case.runtime.max_model_len, p.pp)
            if self.pages.groups else None
        )
        group_metadata = (
            self.pages.metadata_stage_bytes(seq, self.case.runtime.max_model_len, p.pp)
            if self.pages.groups else None
        )
        if include_dp_cohort_extra:
            cohort_alloc_parts = self._dp_cohort_extra_parts_by_stage(
                seq, padded=True, rank=rank, query_tokens=q_tok, phase=phase,
            )
            cohort_valid_parts = self._dp_cohort_extra_parts_by_stage(
                seq, padded=False, rank=rank, query_tokens=q_tok, phase=phase,
            )
            rank_base_per_req = (
                None if hybrid_b is not None
                else self._rank_dp_cohort_cache_bytes(
                    seq, padded=True, rank=rank, query_tokens=q_tok, phase=phase,
                )
            )
        else:
            cohort_alloc_parts = [{"kv": 0.0, "index": 0.0, "draft": 0.0} for _ in range(p.pp)]
            cohort_valid_parts = [{"kv": 0.0, "index": 0.0, "draft": 0.0} for _ in range(p.pp)]
            rank_base_per_req = (
                list(self.cache_bytes(seq, padded=True, query_tokens=q_tok, phase=phase))
                if rank is None
                else self._rank_cache_bytes(
                    seq, padded=True, rank=int(rank), query_tokens=q_tok, phase=phase,
                )
            )
        c8_static_stages = tuple(getattr(self, "c8_static_channel_scale_by_stage", (0,) * p.pp))
        hbm_ratio = float(self.kvc_hbm_ratio)
        if hbm_ratio < 1.0:
            layer_max = (
                self._rank_max_layer_cache_bytes(seq, int(rank), query_tokens=q_tok, phase=phase)
                if rank is not None
                else self._max_layer_cache_bytes(seq, query_tokens=q_tok, phase=phase)
            )
            proj_staging_stages = tuple(
                int(round(float(layer_max[st]) * (1.0 - hbm_ratio) * b_int))
                for st in range(p.pp)
            )
        else:
            proj_staging_stages = (0,) * p.pp

        per_stage: list[dict[str, Any]] = []
        for stage in range(p.pp):
            entry = self._assemble_stage_die_cache_pool(
                stage=stage,
                b_int=b_int,
                alloc_stage=alloc_stages[stage],
                valid_stage=valid_stages[stage],
                group_part=group_parts[stage] if group_parts is not None else None,
                group_payload_b=float(group_payload[stage]) if group_payload is not None else 0.0,
                group_meta_b=float(group_metadata[stage]) if group_metadata is not None else 0.0,
                cohort_alloc=cohort_alloc_parts[stage],
                cohort_valid=cohort_valid_parts[stage],
                hybrid_b=hybrid_b,
                hybrid_1=hybrid_1,
                rank_base_per_req_b=float(rank_base_per_req[stage]) if rank_base_per_req is not None else 0.0,
                c8_static_b=int(c8_static_stages[stage]) if stage < len(c8_static_stages) else 0,
                hbm_ratio=hbm_ratio,
                proj_staging_b=int(proj_staging_stages[stage]),
                pool_map=_COMPONENT_POOL_MAP,
            )
            per_stage.append(entry)
        return per_stage, hybrid_b

    def _assemble_stage_die_cache_pool(
        self,
        *,
        stage: int,
        b_int: int,
        alloc_stage: dict[str, Any],
        valid_stage: dict[str, Any],
        group_part: dict[str, Any] | None,
        group_payload_b: float,
        group_meta_b: float,
        cohort_alloc: dict[str, float],
        cohort_valid: dict[str, float],
        hybrid_b: Any,
        hybrid_1: Any,
        rank_base_per_req_b: float,
        c8_static_b: int,
        hbm_ratio: float,
        proj_staging_b: int,
        pool_map: dict[str, str],
    ) -> dict[str, Any]:
        """Conserve exact integer per-stage cache pools and component breakdowns."""
        alloc_pools = {
            "target_kv_bytes": 0.0, "indexer_k_bytes": 0.0,
            "recurrent_state_bytes": 0.0, "draft_kv_bytes": 0.0,
            "draft_state_bytes": 0.0,
        }
        valid_pools = {"target_kv_bytes": 0.0, "indexer_k_bytes": 0.0}
        comp_per_req: dict[str, float] = {}
        comp_batch: dict[str, int] = {}

        for name, detail in alloc_stage["components"].items():
            val = float(detail["total"])
            if val <= 0.0:
                continue
            comp_per_req[name] = comp_per_req.get(name, 0.0) + val
            key = pool_map.get(name)
            if key in alloc_pools:
                alloc_pools[key] += val
        for name, detail in valid_stage["components"].items():
            key = pool_map.get(name)
            if key in valid_pools:
                valid_pools[key] += float(detail["total"])

        group_reserve_per_req = 0.0
        if group_part is not None:
            known_group = 0.0
            for name, detail in group_part["components"].items():
                part_alloc = float(detail["alloc_payload"]) + float(detail["alloc_scale"])
                part_valid = float(detail["valid_payload"]) + float(detail["valid_scale"])
                if part_alloc > 0.0:
                    comp_per_req[name] = comp_per_req.get(name, 0.0) + part_alloc
                key = pool_map.get(name)
                if key in alloc_pools:
                    alloc_pools[key] += part_alloc
                    known_group += part_alloc
                if key in valid_pools:
                    valid_pools[key] += part_valid
            group_reserve_per_req = max(0.0, group_payload_b - known_group)
            if group_reserve_per_req > 0.0:
                comp_per_req["page_group_reserve"] = (
                    comp_per_req.get("page_group_reserve", 0.0) + group_reserve_per_req
                )
            if group_meta_b > 0.0:
                comp_per_req["page_metadata"] = (
                    comp_per_req.get("page_metadata", 0.0) + group_meta_b
                )

        cohort_alloc_kv = float(cohort_alloc.get("kv", 0.0))
        cohort_alloc_idx = float(cohort_alloc.get("index", 0.0))
        cohort_alloc_drf = float(cohort_alloc.get("draft", 0.0))
        if cohort_alloc_kv > 0.0:
            comp_per_req["dcp_dp_cohort_kv"] = comp_per_req.get("dcp_dp_cohort_kv", 0.0) + cohort_alloc_kv
            alloc_pools["target_kv_bytes"] += cohort_alloc_kv
        if cohort_alloc_idx > 0.0:
            comp_per_req["dcp_dp_cohort_index"] = comp_per_req.get("dcp_dp_cohort_index", 0.0) + cohort_alloc_idx
            alloc_pools["indexer_k_bytes"] += cohort_alloc_idx
        if cohort_alloc_drf > 0.0:
            comp_per_req["dcp_dp_cohort_draft"] = comp_per_req.get("dcp_dp_cohort_draft", 0.0) + cohort_alloc_drf
            alloc_pools["draft_kv_bytes"] += cohort_alloc_drf
        valid_pools["target_kv_bytes"] += float(cohort_valid.get("kv", 0.0))
        valid_pools["indexer_k_bytes"] += float(cohort_valid.get("index", 0.0))

        for name, val in list(comp_per_req.items()):
            scaled = val * hbm_ratio
            comp_per_req[name] = scaled
            comp_batch[name] = int(round(b_int * scaled))

        if hybrid_b is not None:
            stride_res_b = int(round(int(hybrid_b.stride_reserve_by_stage[stage]) * hbm_ratio))
            stride_res_1 = (
                float(hybrid_1.stride_reserve_by_stage[stage]) * hbm_ratio
                if hybrid_1 is not None else 0.0
            )
            if stride_res_b > 0:
                comp_batch["page_group_reserve"] = comp_batch.get("page_group_reserve", 0) + stride_res_b
            if stride_res_1 > 0.0:
                comp_per_req["page_group_reserve"] = comp_per_req.get("page_group_reserve", 0.0) + stride_res_1
            outside_subsumed = (
                int(hybrid_b.total_pool_bytes_by_stage[stage])
                + int(hybrid_b.outside_replay_bytes_by_stage[stage])
            )
            sub_draft_kv = bool(hybrid_b.contract.subsumed_draft_kv)
            expected_non_c8 = int(round(outside_subsumed * hbm_ratio))
            if not sub_draft_kv:
                expected_non_c8 += comp_batch.get("draft_kv", 0)
            expected_non_c8 += comp_batch.get("draft_state", 0)
            expected_total = expected_non_c8 + c8_static_b
            page_reserve_total = stride_res_b
        else:
            page_reserve_total = int(round(b_int * group_reserve_per_req * hbm_ratio))
            expected_total = int(round(b_int * rank_base_per_req_b * hbm_ratio)) + c8_static_b

        if c8_static_b > 0:
            comp_batch["c8_static_channel_scale"] = c8_static_b
            comp_per_req.setdefault("c8_static_channel_scale", 0.0)

        idx_a = sum(comp_batch.get(c, 0) for c, k in pool_map.items() if k == "indexer_k_bytes")
        rec_a = sum(comp_batch.get(c, 0) for c, k in pool_map.items() if k == "recurrent_state_bytes")
        drf_a = comp_batch.get("draft_kv", 0) + comp_batch.get("draft_state", 0) + comp_batch.get("dcp_dp_cohort_draft", 0)
        non_kv = idx_a + rec_a + drf_a
        residual = expected_total - sum(comp_batch.values())
        if residual != 0:
            if hybrid_b is not None or group_part is not None:
                comp_batch["page_group_reserve"] = max(0, comp_batch.get("page_group_reserve", 0) + residual)
            else:
                target_key = next((c for c in ("mla_kv", "gqa_kv", "window", "compressed_kv") if c in comp_batch), "page_group_reserve")
                comp_batch[target_key] = max(0, comp_batch.get(target_key, 0) + residual)
        kv_a = (
            expected_total - non_kv
            if expected_total >= non_kv
            else int(round(b_int * alloc_pools["target_kv_bytes"] * hbm_ratio)) + c8_static_b
        )
        return {
            "kv_alloc": kv_a,
            "kv_valid": int(round(b_int * valid_pools["target_kv_bytes"] * hbm_ratio)) + c8_static_b,
            "idx_alloc": idx_a,
            "idx_valid": int(round(b_int * valid_pools["indexer_k_bytes"] * hbm_ratio)),
            "rec": rec_a,
            "draft": drf_a,
            "page_group_reserve": page_reserve_total,
            "proj_staging": int(proj_staging_b),
            "components_batch": {k: v for k, v in comp_batch.items() if v > 0},
            "components_per_req": {
                k: v for k, v in comp_per_req.items() if v > 0.0 or k == "c8_static_channel_scale"
            },
        }

    def _projected_stage_cache_pools(
        self,
        sequence_tokens: int | None,
        batch_per_dp: int,
        query_tokens: int | None,
    ) -> list[dict[str, int]] | None:
        """Per-stage projected KV/Index/State/Draft bytes for ``batch_per_dp`` requests."""
        if sequence_tokens is None or int(batch_per_dp) <= 0:
            return None
        stages, _ = self._projected_die_cache_pools(
            int(sequence_tokens), int(batch_per_dp), query_tokens, rank=None,
        )
        return [
            {
                "kv_alloc": int(st["kv_alloc"]),
                "kv_valid": int(st["kv_valid"]),
                "idx_alloc": int(st["idx_alloc"]),
                "idx_valid": int(st["idx_valid"]),
                "rec": int(st["rec"]),
                "draft": int(st["draft"]),
                "page_group_reserve": int(st["page_group_reserve"]),
            }
            for st in stages
        ]

    def _extract_ragged_request_specs(
        self,
        views: Sequence[Any],
        *,
        default_phase: str,
    ) -> tuple[tuple[int, int, int, str, int], ...]:
        """Extract logical ``(seq_len, q_len, prefix_tokens, prefix_key, gen)`` per active view."""
        specs: list[tuple[int, int, int, str, int]] = []
        norm_default = str(default_phase or "decode").strip().lower()
        for view in views:
            if isinstance(view, Mapping):
                if bool(view.get("padding", False)):
                    continue
                v_phase = str(view.get("phase", norm_default) or norm_default).strip().lower()
                global_q = int(view.get("global_q_len") or 0)
                local_q = int(view.get("q_len") or 0)
                prop = int(view.get("draft_proposal_tokens") or 0)
                ctx = int(view.get("context") or 0)
                raw_kv = int(view.get("kv_len") or 0)
                p_tok = int(view.get("prefix_tokens") or 0)
                p_key = str(view.get("prefix_cache_key") or "")
                p_gen = int(view.get("prefix_generation") or view.get("generation") or 0)
            else:
                if bool(getattr(view, "padding", False)):
                    continue
                v_phase = str(getattr(view, "phase", norm_default) or norm_default).strip().lower()
                global_q = int(getattr(view, "global_q_len", 0) or 0)
                local_q = int(getattr(view, "q_len", 0) or 0)
                prop = int(getattr(view, "draft_proposal_tokens", 0) or 0)
                ctx = int(getattr(view, "context", 0) or 0)
                raw_kv = int(getattr(view, "kv_len", 0) or 0)
                p_tok = int(getattr(view, "prefix_tokens", 0) or 0)
                p_key = str(getattr(view, "prefix_cache_key", "") or "")
                p_gen = int(getattr(view, "prefix_generation", None) or getattr(view, "generation", 0) or 0)
            if v_phase == "decode":
                q_v = max(1, global_q) if global_q > 0 else max(1, (local_q or 1) + max(0, prop))
                seq_v = max(0, max(raw_kv, ctx + q_v))
            else:
                eff_q = global_q if global_q > 0 else local_q
                q_v = 1
                seq_v = max(0, max(raw_kv, ctx + max(1, eff_q)))
            specs.append((seq_v, q_v, max(0, min(p_tok, seq_v)), p_key, p_gen))
        return tuple(specs)

    def _ragged_shared_prefix_dedup(
        self,
        acc: list[dict[str, Any]],
        specs: Sequence[tuple[int, int, int, str, int]],
        *,
        rank: int | None,
        phase: str,
        include_dp_cohort_extra: bool = True,
        peer_dcp_only: bool = False,
    ) -> None:
        """Deduplicate ONLY live, window-resident, shareable physical prefix pages (never draft)."""
        if (
            getattr(self.case.runtime, "prefix_cache_mode", "private") != "shared"
            or len(specs) <= 1
            or not self.case.model.layer_plans
        ):
            return
        from mindie_sim.catalog.cache import DEFAULT_CACHE_BLOCK_SIZE
        from mindie_sim.catalog.cache_ownership import (
            SHARING_SHARED,
            _union_elementary_spans,
            cache_component_ranges,
        )
        from mindie_sim.catalog.context_parallel import (
            cp_interleave,
            dcp_indexer_sharded,
            layer_dcp_factors,
        )
        from mindie_sim.schema.runtime_features import canonical_prefix_cache_key
        from .cache import _COMPONENT_POOL_MAP, _DCP_KV_SHARDED_COMPONENTS

        model, p = self.case.model, self.case.runtime.parallel
        block = max(1, int(getattr(self.case.runtime, "kv_block_size", 0) or DEFAULT_CACHE_BLOCK_SIZE))
        interleave = cp_interleave(self.case.runtime)
        sharded_idx = dcp_indexer_sharded(model, self.case.runtime)
        hbm_ratio = float(self.kvc_hbm_ratio)
        shared_spans: dict[tuple, list[tuple[int, int, float]]] = {}
        gross_shared: dict[tuple, float] = {}
        for seq_v, q_v, p_tok, p_key, p_gen in specs:
            if seq_v <= 0 or p_tok <= 0:
                continue
            ckey = canonical_prefix_cache_key("shared", p_key)
            for layer in model.layer_plans:
                _, _, layer_dp = layer_dcp_factors(layer, p)
                if peer_dcp_only and layer_dp <= 1:
                    continue
                ranges = cache_component_ranges(
                    model, layer, p, sequence_tokens=int(seq_v),
                    prefix_tokens=int(p_tok), interleave=interleave,
                    block_size=block, rank=rank, capacity_padded=False,
                    runtime=self.case.runtime, query_tokens=int(q_v),
                    generation=int(p_gen), phase=phase,
                )
                st = self.layer_stage[int(layer.index)]
                for item in ranges:
                    if item.sharing != SHARING_SHARED or item.valid_bytes <= 0.0:
                        continue
                    is_dcp_comp = (
                        item.component in _DCP_KV_SHARDED_COMPONENTS
                        or (item.component == "index_k" and sharded_idx)
                    )
                    if peer_dcp_only and not is_dcp_comp:
                        continue
                    dp_mult = float(layer_dp) if (include_dp_cohort_extra and not peer_dcp_only and is_dcp_comp and layer_dp > 1) else 1.0
                    span_len = max(1, item.end_token - item.start_token)
                    rate = (float(item.valid_bytes) * dp_mult * hbm_ratio) / float(span_len)
                    pool_kind = "idx" if _COMPONENT_POOL_MAP.get(item.component) == "indexer_k_bytes" else "kv"
                    grp = (int(item.layer), item.component, ckey, int(p_gen), st, pool_kind)
                    shared_spans.setdefault(grp, []).append((int(item.start_token), int(item.end_token), rate))
                    gross_shared[grp] = gross_shared.get(grp, 0.0) + float(item.valid_bytes) * dp_mult * hbm_ratio
        for (_lyr, comp, _ckey, _gen, st, pool_kind), spans in shared_spans.items():
            saved = int(round(max(0.0, gross_shared[(_lyr, comp, _ckey, _gen, st, pool_kind)] - _union_elementary_spans(spans))))
            if saved <= 0:
                continue
            if pool_kind == "idx":
                acc[st]["idx_alloc"] = max(0, acc[st]["idx_alloc"] - saved)
                acc[st]["idx_valid"] = max(0, acc[st]["idx_valid"] - saved)
            else:
                acc[st]["kv_alloc"] = max(0, acc[st]["kv_alloc"] - saved)
                acc[st]["kv_valid"] = max(0, acc[st]["kv_valid"] - saved)
            if comp in acc[st]["components_batch"]:
                acc[st]["components_batch"][comp] = max(0, acc[st]["components_batch"][comp] - saved)

    def _empty_projected_stage_acc(self) -> list[dict[str, Any]]:
        return [
            {
                "kv_alloc": 0, "kv_valid": 0, "idx_alloc": 0, "idx_valid": 0,
                "rec": 0, "draft": 0, "page_group_reserve": 0, "proj_staging": 0,
                "components_batch": {}, "components_per_req": {},
            }
            for _ in range(self.case.runtime.parallel.pp)
        ]

    def _merge_stage_acc(self, dst: list[dict[str, Any]], src: list[dict[str, Any]]) -> None:
        for st in range(self.case.runtime.parallel.pp):
            for key in ("kv_alloc", "kv_valid", "idx_alloc", "idx_valid", "rec", "draft", "page_group_reserve", "proj_staging"):
                dst[st][key] += int(src[st][key])
            for comp_name, val in src[st].get("components_batch", {}).items():
                dst[st]["components_batch"][comp_name] = dst[st]["components_batch"].get(comp_name, 0) + int(val)

    def _projected_ragged_die_cache_pools(
        self,
        specs: Sequence[tuple[int, int, int, str, int]],
        *,
        rank: int | None,
        phase: str,
        include_dp_cohort_extra: bool = True,
        include_c8: bool = True,
    ) -> list[dict[str, Any]]:
        """Sum exact integer cache pools over ragged logical KV lengths (charging C8 once)."""
        p = self.case.runtime.parallel
        c8_stages = tuple(getattr(self, "c8_static_channel_scale_by_stage", (0,) * p.pp))
        buckets: dict[tuple[int, int], int] = {}
        for seq_v, q_v, _p_tok, _p_key, _p_gen in specs:
            if seq_v <= 0:
                continue
            buckets[(int(seq_v), int(q_v))] = buckets.get((int(seq_v), int(q_v)), 0) + 1
        acc = self._empty_projected_stage_acc()
        for (seq_v, q_v), count in sorted(buckets.items()):
            stages, _ = self._projected_die_cache_pools(
                seq_v, count, q_v, rank=rank, phase=phase,
                include_dp_cohort_extra=include_dp_cohort_extra,
            )
            for st in range(p.pp):
                c8_b = int(c8_stages[st]) if st < len(c8_stages) else 0
                entry = stages[st]
                acc[st]["kv_alloc"] += max(0, int(entry["kv_alloc"]) - c8_b)
                acc[st]["kv_valid"] += max(0, int(entry["kv_valid"]) - c8_b)
                acc[st]["idx_alloc"] += int(entry["idx_alloc"])
                acc[st]["idx_valid"] += int(entry["idx_valid"])
                acc[st]["rec"] += int(entry["rec"])
                acc[st]["draft"] += int(entry["draft"])
                acc[st]["page_group_reserve"] += int(entry["page_group_reserve"])
                acc[st]["proj_staging"] += int(entry["proj_staging"])
                for comp_name, val in entry.get("components_batch", {}).items():
                    if comp_name == "c8_static_channel_scale":
                        continue
                    acc[st]["components_batch"][comp_name] = (
                        acc[st]["components_batch"].get(comp_name, 0) + int(val)
                    )
        self._ragged_shared_prefix_dedup(
            acc, specs, rank=rank, phase=phase,
            include_dp_cohort_extra=include_dp_cohort_extra,
        )
        if include_c8:
            for st in range(p.pp):
                c8_b = int(c8_stages[st]) if st < len(c8_stages) else 0
                if c8_b > 0:
                    acc[st]["kv_alloc"] += c8_b
                    acc[st]["kv_valid"] += c8_b
                    acc[st]["components_batch"]["c8_static_channel_scale"] = c8_b
        return acc

    def _projected_peer_dcp_specs(
        self,
        specs: Sequence[tuple[int, int, int, str, int]],
        *,
        rank: int,
        phase: str,
    ) -> list[dict[str, Any]]:
        """Per-stage DCP-sharded bytes placed on a peer die for one source replica's specs."""
        p = self.case.runtime.parallel
        hbm_ratio = float(self.kvc_hbm_ratio)
        acc = self._empty_projected_stage_acc()
        buckets: dict[tuple[int, int], int] = {}
        for seq_v, q_v, _p_tok, _p_key, _p_gen in specs:
            if seq_v > 0:
                buckets[(int(seq_v), int(q_v))] = buckets.get((int(seq_v), int(q_v)), 0) + 1
        for (seq_v, q_v), count in sorted(buckets.items()):
            a_parts = self._dp_cohort_extra_parts_by_stage(
                seq_v, padded=True, rank=rank, query_tokens=q_v,
                phase=phase, peer_count_override=1,
            )
            v_parts = self._dp_cohort_extra_parts_by_stage(
                seq_v, padded=False, rank=rank, query_tokens=q_v,
                phase=phase, peer_count_override=1,
            )
            layer_max = (
                self._rank_max_layer_cache_bytes(seq_v, rank, query_tokens=q_v, phase=phase)
                if hbm_ratio < 1.0 else (0.0,) * p.pp
            )
            for st in range(p.pp):
                kv_a = int(round(float(a_parts[st]["kv"]) * hbm_ratio * count))
                kv_v = int(round(float(v_parts[st]["kv"]) * hbm_ratio * count))
                idx_a = int(round(float(a_parts[st]["index"]) * hbm_ratio * count))
                idx_v = int(round(float(v_parts[st]["index"]) * hbm_ratio * count))
                drf_a = int(round(float(a_parts[st]["draft"]) * hbm_ratio * count))
                acc[st]["kv_alloc"] += kv_a
                acc[st]["kv_valid"] += kv_v
                acc[st]["idx_alloc"] += idx_a
                acc[st]["idx_valid"] += idx_v
                acc[st]["draft"] += drf_a
                if hbm_ratio < 1.0:
                    acc[st]["proj_staging"] += int(round(float(layer_max[st]) * (1.0 - hbm_ratio) * count))
                if kv_a > 0:
                    acc[st]["components_batch"]["dcp_dp_cohort_kv"] = (
                        acc[st]["components_batch"].get("dcp_dp_cohort_kv", 0) + kv_a
                    )
                if idx_a > 0:
                    acc[st]["components_batch"]["dcp_dp_cohort_index"] = (
                        acc[st]["components_batch"].get("dcp_dp_cohort_index", 0) + idx_a
                    )
                if drf_a > 0:
                    acc[st]["components_batch"]["dcp_dp_cohort_draft"] = (
                        acc[st]["components_batch"].get("dcp_dp_cohort_draft", 0) + drf_a
                    )
        self._ragged_shared_prefix_dedup(
            acc, specs, rank=rank, phase=phase,
            include_dp_cohort_extra=False, peer_dcp_only=True,
        )
        return acc

    def _scatter_dp_cohort_projected(
        self,
        per_rep_specs: Mapping[int, tuple[tuple[int, int, int, str, int], ...]],
        *,
        dp_width: int,
        phase: str,
        include_c8: bool = True,
    ) -> dict[tuple[int, int], list[dict[str, Any]]]:
        """Scatter each source replica's requests onto its true destination cohort dies."""
        p = self.case.runtime.parallel
        c8_stages = tuple(getattr(self, "c8_static_channel_scale_by_stage", (0,) * p.pp))
        resolved: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for src_rep, specs in per_rep_specs.items():
            if not specs:
                continue
            cohort_start = (int(src_rep) // dp_width) * dp_width
            cohort_end = min(int(p.dp), cohort_start + dp_width)
            for dst_rep in range(cohort_start, cohort_end):
                for local_rk in range(self.width):
                    phys_rk = ((dst_rep - cohort_start) * p.pp) * self.width + local_rk
                    key = (dst_rep, local_rk)
                    if key not in resolved:
                        resolved[key] = self._empty_projected_stage_acc()
                    if dst_rep == int(src_rep):
                        part = self._projected_ragged_die_cache_pools(
                            specs, rank=phys_rk, phase=phase,
                            include_dp_cohort_extra=False, include_c8=False,
                        )
                    else:
                        part = self._projected_peer_dcp_specs(
                            specs, rank=phys_rk, phase=phase,
                        )
                    self._merge_stage_acc(resolved[key], part)
        if include_c8:
            for stages in resolved.values():
                for st in range(p.pp):
                    c8_b = int(c8_stages[st]) if st < len(c8_stages) else 0
                    if c8_b > 0:
                        stages[st]["kv_alloc"] += c8_b
                        stages[st]["kv_valid"] += c8_b
                        stages[st]["components_batch"]["c8_static_channel_scale"] = c8_b
        return resolved

    def _merge_die_non_request_and_extra_grants(
        self,
        die_backings: Mapping[str, Any],
        dedup_grants: Sequence[ResidentGrantEntry],
        base_gids: set[str],
    ) -> tuple[int, int, set[str]]:
        """Unify live non-request ledger backings with ``extra_grants`` by physical identity."""
        all_ranges: dict[tuple, list] = {k: list(v) for k, v in die_backings["all_ranges"].items()}
        ret_ranges: dict[tuple, list] = {k: list(v) for k, v in die_backings["ret_ranges"].items()}
        events_map: dict[tuple, set[str]] = {k: set(v) for k, v in die_backings["events"].items()}
        fixed_backings = {
            (str(g.physical_backing_id), int(g.generation), str(g.tier or "hbm").lower() or "hbm")
            for g in dedup_grants
            if g.grant_id in base_gids or g.grant_id.startswith("prepared:")
        }
        key_by_bgt: dict[tuple[str, Any, str], tuple] = {}
        for bkey in all_ranges:
            key_by_bgt.setdefault((bkey[1], bkey[2], bkey[3]), bkey)
        for g in dedup_grants:
            if g.grant_id in base_gids or g.grant_id.startswith("prepared:"):
                continue
            tier_norm = str(g.tier or "hbm").lower() or "hbm"
            if tier_norm != "hbm" or g.lifetime == "load_peak_only":
                continue
            bgt = (str(g.physical_backing_id), int(g.generation), tier_norm)
            if bgt in fixed_backings and int(g.billed_once_bytes) <= 0:
                continue
            raw_b = float(int(g.allocated_bytes or g.runtime_bytes))
            if raw_b <= 0.0:
                continue
            bkey = key_by_bgt.get(bgt, ("", bgt[0], bgt[1], bgt[2]))
            key_by_bgt.setdefault(bgt, bkey)
            all_ranges.setdefault(bkey, []).append((0.0, raw_b))
            ev = str(g.release_event).strip()
            if g.reclaimable and ev:
                events_map.setdefault(bkey, set()).add(ev)
            else:
                ret_ranges.setdefault(bkey, []).append((0.0, raw_b))
        unb_live = float(die_backings["unbacked_live"])
        unb_ret = float(die_backings["unbacked_ret"])
        live_total = unb_live + sum(_union_bytes(r) for r in all_ranges.values())
        ret_total = unb_ret + sum(_union_bytes(r) for r in ret_ranges.values())
        recl_total = max(0, int(round(live_total - ret_total)))
        evs: set[str] = set()
        if unb_live > unb_ret + 1e-9:
            evs.update(die_backings["unbacked_events"])
        for bkey, b_evs in events_map.items():
            if _union_bytes(all_ranges.get(bkey, ())) > _union_bytes(ret_ranges.get(bkey, ())) + 1e-9:
                evs.update(b_evs)
        return int(round(live_total)), recl_total, evs

    def _eval_live_phase_specs(
        self,
        phase_map: Mapping[str, Mapping[int, Sequence[tuple[int, int, int, str, int]]]],
        *,
        dp_width: int,
        spans_dp: bool,
        rank_divergent: bool,
    ) -> dict[tuple[int, int], list[dict[str, Any]]]:
        """Evaluate live request specs grouped by ``phase -> replica`` onto physical dies."""
        p = self.case.runtime.parallel
        out: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for req_phase, rep_specs in phase_map.items():
            frozen = {rep: tuple(vals) for rep, vals in rep_specs.items() if vals}
            if not frozen:
                continue
            if spans_dp:
                part = self._scatter_dp_cohort_projected(
                    frozen, dp_width=dp_width, phase=req_phase, include_c8=False,
                )
                for key, stages in part.items():
                    if key not in out:
                        out[key] = self._empty_projected_stage_acc()
                    self._merge_stage_acc(out[key], stages)
                continue
            for rep, specs in frozen.items():
                if rank_divergent:
                    for local_rk in range(self.width):
                        part_st = self._projected_ragged_die_cache_pools(
                            specs, rank=local_rk, phase=req_phase,
                            include_dp_cohort_extra=True, include_c8=False,
                        )
                        key = (rep, local_rk)
                        if key not in out:
                            out[key] = self._empty_projected_stage_acc()
                        self._merge_stage_acc(out[key], part_st)
                else:
                    part_st = self._projected_ragged_die_cache_pools(
                        specs, rank=None, phase=req_phase,
                        include_dp_cohort_extra=True, include_c8=False,
                    )
                    for local_rk in range(self.width):
                        key = (rep, local_rk)
                        if key not in out:
                            out[key] = self._empty_projected_stage_acc()
                        self._merge_stage_acc(out[key], part_st)
        return out

    def _live_die_cache_pools(
        self,
        *,
        default_phase: str,
        dp_width: int,
        spans_dp: bool,
    ) -> dict[tuple[int, int], list[dict[str, Any]]]:
        """Decompose live admitted requests (including held readers) into per-die cache pools."""
        p = self.case.runtime.parallel
        keys = set(self.owners) | set(self.owner_ranks) | set(self._cohort_owner_ranks)
        if not keys:
            return {}
        rank_divergent = self.rank_token_owners is not None or int(getattr(p, "cp", 1) or 1) > 1
        alloc_by_phase: dict[str, dict[int, list[tuple[int, int, int, str, int]]]] = {}
        valid_by_phase: dict[str, dict[int, list[tuple[int, int, int, str, int]]]] = {}
        untracked = [[[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        has_untracked = False
        for key in keys:
            rep, _req = key
            if key not in self.request_tokens:
                if key in self.retired_requests:
                    continue
                has_untracked = True
                cohort_rows = self._cohort_owner_ranks.get(key)
                if cohort_rows is not None:
                    for peer_rep, rows in cohort_rows.items():
                        for st in range(p.pp):
                            for rk in range(self.width):
                                untracked[peer_rep][st][rk] += float(rows[st][rk])
                else:
                    rows = self.owner_ranks.get(key) or self._replicate_ranks(self.owners[key])
                    for st in range(p.pp):
                        for rk in range(self.width):
                            untracked[rep][st][rk] += float(rows[st][rk])
                continue
            retired = key in self.retired_requests
            active_alloc = 0 if retired else int(self.request_tokens.get(key, 0))
            active_valid = 0 if retired else int(self.request_valid_tokens.get(key, active_alloc))
            active_q = int(self.request_query_tokens.get(key, 1) or 1)
            req_phase = str(self.request_phases.get(key, default_phase) or default_phase)
            holds = self._request_hold_tokens.get(key, {})
            eff_alloc = max([active_alloc, *(int(h[0]) for h in holds.values())])
            eff_valid = max([active_valid, *(int(h[1]) for h in holds.values())])
            if eff_alloc > 0:
                alloc_by_phase.setdefault(req_phase, {}).setdefault(int(rep), []).append(
                    (eff_alloc, active_q, 0, "", 0)
                )
            if eff_valid > 0:
                valid_by_phase.setdefault(req_phase, {}).setdefault(int(rep), []).append(
                    (eff_valid, active_q, 0, "", 0)
                )
        resolved = self._eval_live_phase_specs(
            alloc_by_phase, dp_width=dp_width, spans_dp=spans_dp, rank_divergent=rank_divergent,
        )
        if valid_by_phase != alloc_by_phase:
            valid_resolved = self._eval_live_phase_specs(
                valid_by_phase, dp_width=dp_width, spans_dp=spans_dp, rank_divergent=rank_divergent,
            )
            for dkey, stages in resolved.items():
                v_stages = valid_resolved.get(dkey)
                for st in range(p.pp):
                    stages[st]["kv_valid"] = int(v_stages[st]["kv_valid"]) if v_stages is not None else 0
                    stages[st]["idx_valid"] = int(v_stages[st]["idx_valid"]) if v_stages is not None else 0
        if has_untracked:
            for rep in range(p.dp):
                for rk in range(self.width):
                    if any(untracked[rep][st][rk] > 0.0 for st in range(p.pp)):
                        dkey = (rep, rk)
                        if dkey not in resolved:
                            resolved[dkey] = self._empty_projected_stage_acc()
                        for st in range(p.pp):
                            extra_b = int(round(untracked[rep][st][rk]))
                            resolved[dkey][st]["kv_alloc"] += extra_b
                            resolved[dkey][st]["kv_valid"] += extra_b
        return resolved

    def _build_die_receipt(
        self,
        *,
        replica: int,
        stage: int,
        rank: int,
        die_obj: Any,
        ckpt_b: int,
        vision_b: int,
        term_prep_b: int,
        prep_persist_b: int,
        prep_load_b: int,
        stage_prep_grants: Sequence[ResidentGrantEntry],
        extra_grants: Sequence[ResidentGrantEntry],
        projected: dict[str, int] | None,
        live_pool: dict[str, Any] | None = None,
        dyn_live_b: int,
        dyn_peak_b: int,
        req_b: int,
        stg_b: int,
        host_b: int,
        die_backings: Mapping[str, Any],
        req_reclaimable_b: int = 0,
        req_reclaim_events: Sequence[str] = (),
    ) -> DieResidentReceipt:
        usable = int(math.floor(die_obj.hbm_bytes * self.case.runtime.gpu_memory_utilization))
        rank_ws = getattr(self, "rank_workspace", None)
        die_ws = (
            rank_ws[replica][stage][rank]
            if (rank_ws is not None and replica < len(rank_ws)
                and stage < len(rank_ws[replica]) and rank < len(rank_ws[replica][stage]))
            else self.workspace[stage]
        )
        ws_b = int(round(die_ws + self.temporary_dequant[stage]))
        res_b = int(round(self.policy.reserved))
        c8_stages = tuple(getattr(self, "c8_static_channel_scale_by_stage", ()))
        c8_static_b = int(c8_stages[stage]) if stage < len(c8_stages) else 0
        dyn_grant_only = max(0, dyn_live_b - req_b - stg_b)
        if projected is not None:
            kv_a = projected["kv_alloc"]
            kv_v = projected["kv_valid"]
            idx_a = projected["idx_alloc"]
            idx_v = projected["idx_valid"]
            rec_b = projected["rec"]
            drf_b = projected["draft"]
        elif live_pool is not None:
            pool_kv_a = int(live_pool["kv_alloc"])
            pool_kv_v = int(live_pool["kv_valid"])
            idx_a = int(live_pool["idx_alloc"])
            idx_v = int(live_pool["idx_valid"])
            rec_b = int(live_pool["rec"])
            drf_b = int(live_pool["draft"])
            non_kv = idx_a + rec_b + drf_b
            if req_b > pool_kv_a + non_kv:
                pool_kv_a += req_b - (pool_kv_a + non_kv)
            kv_a = pool_kv_a + c8_static_b
            kv_v = min(pool_kv_a, pool_kv_v) + c8_static_b
            idx_v = min(idx_a, idx_v)
        else:
            kv_a = req_b + c8_static_b
            kv_v = req_b + c8_static_b
            idx_a = 0
            idx_v = 0
            rec_b = 0
            drf_b = 0

        ckpt_gid = f"r{replica}:s{stage}:k{rank}:checkpoint_weights"
        vis_gid = f"r{replica}:s{stage}:k{rank}:vision_weights"
        term_gid = f"r{replica}:s{stage}:k{rank}:prepared_terms"
        c8_gid = f"r{replica}:s{stage}:k{rank}:c8_static_channel_scale"
        raw_entries: list[ResidentGrantEntry] = [
            ResidentGrantEntry(
                grant_id=ckpt_gid, category="checkpoint_weights",
                physical_backing_id=ckpt_gid, checkpoint_bytes=ckpt_b,
                runtime_bytes=ckpt_b, allocated_bytes=ckpt_b, valid_bytes=ckpt_b,
            ),
        ]
        if vision_b > 0:
            raw_entries.append(ResidentGrantEntry(
                grant_id=vis_gid, category="vision_weights",
                physical_backing_id=vis_gid, checkpoint_bytes=vision_b,
                runtime_bytes=vision_b, allocated_bytes=vision_b, valid_bytes=vision_b,
            ))
        if term_prep_b > 0:
            raw_entries.append(ResidentGrantEntry(
                grant_id=term_gid, category="prepared_weights",
                physical_backing_id=term_gid, checkpoint_bytes=0,
                runtime_bytes=term_prep_b, allocated_bytes=term_prep_b, valid_bytes=term_prep_b,
            ))
        if c8_static_b > 0:
            raw_entries.append(ResidentGrantEntry(
                grant_id=c8_gid, category="c8_static_channel_scale",
                physical_backing_id=c8_gid, checkpoint_bytes=c8_static_b,
                runtime_bytes=c8_static_b, allocated_bytes=c8_static_b, valid_bytes=c8_static_b,
            ))
        raw_entries.extend(stage_prep_grants)
        raw_entries.extend(extra_grants)
        dedup_grants, _ = _deduplicate_grant_entries(raw_entries)
        base_gids = {ckpt_gid, vis_gid, term_gid, c8_gid}
        combined_grant_live, non_req_recl, non_req_evs = (
            self._merge_die_non_request_and_extra_grants(die_backings, dedup_grants, base_gids)
        )
        extra_delta = max(0, combined_grant_live - dyn_grant_only)
        prep_total = term_prep_b + prep_persist_b
        fixed_total = ckpt_b + vision_b + prep_total + ws_b + res_b
        if projected is not None:
            eff_stg_b = int(projected.get("proj_staging", 0)) if self.kvc_hbm_ratio < 1.0 else stg_b
            live_dyn = kv_a + idx_a + rec_b + drf_b + eff_stg_b + combined_grant_live
            total_reclaimable = min(live_dyn, max(0, int(non_req_recl)))
            combined_reclaim_events = tuple(sorted(non_req_evs)) if total_reclaimable > 0 else ()
        else:
            eff_stg_b = stg_b
            live_dyn = req_b + stg_b + c8_static_b + combined_grant_live
            total_reclaimable = min(
                live_dyn,
                max(0, int(req_reclaimable_b)) + max(0, int(non_req_recl)),
            )
            combined_reclaim_events = (
                tuple(sorted(set(req_reclaim_events) | non_req_evs))
                if total_reclaimable > 0 else ()
            )
        peak_dyn = max(live_dyn, dyn_peak_b + c8_static_b + extra_delta)
        sim_live = fixed_total + live_dyn
        sim_peak = fixed_total + peak_dyn
        load_peak = fixed_total + c8_static_b + prep_load_b
        return DieResidentReceipt(
            replica=replica,
            stage=stage,
            rank=rank,
            die_id=int(getattr(die_obj, "rank", rank)),
            usable_hbm_bytes=usable,
            checkpoint_weight_bytes=ckpt_b,
            prepared_weight_bytes=prep_total,
            prepared_load_peak_bytes=prep_load_b,
            vision_weight_bytes=vision_b,
            kv_cache_allocated_bytes=kv_a,
            kv_cache_valid_bytes=kv_v,
            index_cache_allocated_bytes=idx_a,
            index_cache_valid_bytes=idx_v,
            recurrent_state_bytes=rec_b,
            draft_cache_bytes=drf_b,
            workspace_bytes=ws_b,
            reserve_bytes=res_b,
            staging_bytes=eff_stg_b,
            dynamic_grant_bytes=combined_grant_live,
            simultaneous_live_hbm_bytes=sim_live,
            simultaneous_peak_hbm_bytes=sim_peak,
            load_peak_hbm_bytes=load_peak,
            free_hbm_headroom_bytes=usable - sim_peak,
            host_dram_bytes=host_b,
            grants=dedup_grants,
            reclaimable_hbm_bytes=total_reclaimable,
            reclaimable_release_events=combined_reclaim_events,
        )

    def _resolve_receipt_projected_by_replica(
        self,
        *,
        phase: str,
        sequence_tokens: int | None,
        batch_per_dp: int | Sequence[int],
        query_tokens: int | None,
        request_views: Sequence[Any] | None,
        request_views_by_replica: Mapping[int, Sequence[Any]] | Sequence[Sequence[Any]] | None,
        sequence_lengths: Sequence[int] | None,
        query_lengths: Sequence[int] | None,
        dp_width: int,
        spans_dp: bool,
    ) -> dict[tuple[int, int], list[dict[str, Any]]]:
        """Resolve projected per-stage cache pools keyed by ``(replica, rank_slot)``."""
        p = self.case.runtime.parallel
        rank_divergent = (
            self.rank_token_owners is not None or int(getattr(p, "cp", 1) or 1) > 1
        )
        per_rep_specs: dict[int, tuple[tuple[int, int, int, str, int], ...]] = {}
        is_asymmetric_views = False
        if request_views_by_replica is not None:
            is_asymmetric_views = True
            for rep in range(p.dp):
                if isinstance(request_views_by_replica, Mapping):
                    raw_views = request_views_by_replica.get(rep, ())
                else:
                    raw_views = request_views_by_replica[rep] if rep < len(request_views_by_replica) else ()
                specs = self._extract_ragged_request_specs(raw_views, default_phase=phase)
                if specs:
                    per_rep_specs[rep] = specs
        elif request_views is not None:
            specs = self._extract_ragged_request_specs(request_views, default_phase=phase)
            if specs:
                for rep in range(p.dp):
                    per_rep_specs[rep] = specs
        elif sequence_lengths is not None:
            seq_list = [int(s) for s in sequence_lengths]
            if query_lengths is not None:
                q_list = [int(q) for q in query_lengths]
                if len(q_list) != len(seq_list):
                    raise ValueError("memory_receipt_query_lengths_mismatch")
            else:
                default_q = 1 if str(phase or "decode").strip().lower() == "prefill" else self._effective_cache_query_tokens(query_tokens)
                q_list = [default_q] * len(seq_list)
            specs = tuple((max(0, s), max(1, q), 0, "", 0) for s, q in zip(seq_list, q_list) if s > 0)
            if specs:
                for rep in range(p.dp):
                    per_rep_specs[rep] = specs

        if per_rep_specs:
            if spans_dp and is_asymmetric_views:
                return self._scatter_dp_cohort_projected(
                    per_rep_specs, dp_width=dp_width, phase=str(phase),
                )
            resolved: dict[tuple[int, int], list[dict[str, Any]]] = {}
            cache_by_key: dict[tuple, list[dict[str, Any]]] = {}
            for rep, specs in per_rep_specs.items():
                if rank_divergent:
                    for local_rk in range(self.width):
                        phys_rk = ((rep % dp_width) * p.pp) * self.width + local_rk if spans_dp else local_rk
                        ckey = ("ragged", specs, phys_rk)
                        if ckey not in cache_by_key:
                            cache_by_key[ckey] = self._projected_ragged_die_cache_pools(
                                specs, rank=phys_rk, phase=str(phase),
                            )
                        resolved[(rep, local_rk)] = cache_by_key[ckey]
                else:
                    ckey = ("ragged", specs, None)
                    if ckey not in cache_by_key:
                        cache_by_key[ckey] = self._projected_ragged_die_cache_pools(
                            specs, rank=None, phase=str(phase),
                        )
                    for local_rk in range(self.width):
                        resolved[(rep, local_rk)] = cache_by_key[ckey]
            return resolved

        if sequence_tokens is None:
            return {}
        if isinstance(batch_per_dp, Sequence) and not isinstance(batch_per_dp, (str, bytes)):
            if len(batch_per_dp) != p.dp:
                raise ValueError("memory_receipt_batch_per_dp_length_mismatch")
            batches = [int(v) for v in batch_per_dp]
        else:
            batches = [int(batch_per_dp)] * p.dp
        seq_int = int(sequence_tokens)
        if spans_dp and any(
            batches[rep] != batches[(rep // dp_width) * dp_width]
            for rep in range(p.dp)
        ):
            eff_q = 1 if str(phase or "decode").strip().lower() == "prefill" else self._effective_cache_query_tokens(query_tokens)
            asym_specs = {
                rep: ((seq_int, int(eff_q), 0, "", 0),) * b_int
                for rep, b_int in enumerate(batches) if b_int > 0 and seq_int > 0
            }
            return self._scatter_dp_cohort_projected(
                asym_specs, dp_width=dp_width, phase=str(phase),
            )
        resolved = {}
        cache_by_key = {}
        for rep, b_int in enumerate(batches):
            if b_int <= 0:
                continue
            if rank_divergent:
                for local_rk in range(self.width):
                    phys_rk = ((rep % dp_width) * p.pp) * self.width + local_rk if spans_dp else local_rk
                    ckey = ("uniform", seq_int, b_int, query_tokens, phys_rk)
                    if ckey not in cache_by_key:
                        cache_by_key[ckey] = self._projected_die_cache_pools(
                            seq_int, b_int, query_tokens, rank=phys_rk, phase=str(phase),
                        )[0]
                    resolved[(rep, local_rk)] = cache_by_key[ckey]
            else:
                ckey = ("uniform", seq_int, b_int, query_tokens, None)
                if ckey not in cache_by_key:
                    cache_by_key[ckey] = self._projected_die_cache_pools(
                        seq_int, b_int, query_tokens, rank=None, phase=str(phase),
                    )[0]
                for local_rk in range(self.width):
                    resolved[(rep, local_rk)] = cache_by_key[ckey]
        return resolved

    def _projected_host_by_replica(
        self,
        *,
        phase: str,
        sequence_tokens: int | None,
        batch_per_dp: int | Sequence[int],
        query_tokens: int | None,
        request_views: Sequence[Any] | None,
        request_views_by_replica: Mapping[int, Sequence[Any]] | Sequence[Sequence[Any]] | None,
        sequence_lengths: Sequence[int] | None,
        query_lengths: Sequence[int] | None,
    ) -> dict[int, float]:
        """Projected host-DRAM cache bytes per replica when offload tiers are active."""
        if not self.sparse_offload and float(self.offload.get("kvc_ddr_ratio", 0.0) or 0.0) <= 0.0:
            return {}
        p = self.case.runtime.parallel
        host_map: dict[int, float] = {}
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        default_q = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        if request_views_by_replica is not None:
            for rep in range(p.dp):
                raw_views = (
                    request_views_by_replica.get(rep, ())
                    if isinstance(request_views_by_replica, Mapping)
                    else (request_views_by_replica[rep] if rep < len(request_views_by_replica) else ())
                )
                specs = self._extract_ragged_request_specs(raw_views, default_phase=phase)
                if specs:
                    host_map[rep] = float(sum(
                        self.host_dram_cache_bytes(s, padded=True, query_tokens=q, phase=str(phase))
                        for s, q, _pt, _pk, _pg in specs if s > 0
                    ))
            return host_map
        if request_views is not None:
            specs = self._extract_ragged_request_specs(request_views, default_phase=phase)
            if specs:
                val = float(sum(
                    self.host_dram_cache_bytes(s, padded=True, query_tokens=q, phase=str(phase))
                    for s, q, _pt, _pk, _pg in specs if s > 0
                ))
                return {rep: val for rep in range(p.dp)}
            return {}
        if sequence_lengths is not None:
            seq_list = [int(s) for s in sequence_lengths]
            q_list = (
                [int(q) for q in query_lengths] if query_lengths is not None
                else [default_q] * len(seq_list)
            )
            val = float(sum(
                self.host_dram_cache_bytes(s, padded=True, query_tokens=q, phase=str(phase))
                for s, q in zip(seq_list, q_list) if s > 0
            ))
            return {rep: val for rep in range(p.dp)} if val > 0.0 else {}
        if sequence_tokens is None or int(sequence_tokens) <= 0:
            return {}
        batches = (
            [int(v) for v in batch_per_dp]
            if isinstance(batch_per_dp, Sequence) and not isinstance(batch_per_dp, (str, bytes))
            else [int(batch_per_dp)] * p.dp
        )
        per_req = self.host_dram_cache_bytes(
            int(sequence_tokens), padded=True, query_tokens=default_q, phase=str(phase),
        )
        for rep, b_int in enumerate(batches):
            if b_int > 0:
                host_map[rep] = float(per_req) * float(b_int)
        return host_map

    def resident_memory_receipt(
        self,
        *,
        scenario: str = "",
        phase: str = "decode",
        occurrence: str = "steady",
        sequence_tokens: int | None = None,
        batch_per_dp: int | Sequence[int] = 0,
        query_tokens: int | None = None,
        request_views: Sequence[Any] | None = None,
        request_views_by_replica: Mapping[int, Sequence[Any]] | Sequence[Sequence[Any]] | None = None,
        sequence_lengths: Sequence[int] | None = None,
        query_lengths: Sequence[int] | None = None,
        vision_accounting: str | None = None,
        prepared_weights: Sequence[PreparedWeightSpec] = (),
        extra_grants: Sequence[ResidentGrantEntry] = (),
    ) -> ResidentMemoryReceipt:
        """Build the single authoritative per-die ``ResidentMemoryReceipt``."""
        p = self.case.runtime.parallel
        vis_scope = vision_accounting or self.vision_accounting
        vision_stage, vision_delta = self._resolve_vision_delta_by_stage(vis_scope)
        combined_prepared = list(self._prepared_weights.values())
        known_ids = set(self._prepared_weights)
        for spec in prepared_weights:
            if spec.allocation_id not in known_ids:
                combined_prepared.append(spec)
                known_ids.add(spec.allocation_id)
        prep_persist, prep_load, prep_grants = self._stage_prepared_weight_totals(combined_prepared)
        hbm_live, host_live = self._state_usage()
        request_ranks = self._request_usage_ranks()
        staging_ranks = self._staging_usage_ranks()
        req_recl_ranks, req_recl_events = self._request_reclaimable_usage_and_events()
        non_req_backings = self._collect_non_request_backings()
        die_weight_splits = self._resolve_die_weight_splits(vision_stage, vision_delta)
        dp_width = max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
        spans_dp = bool(getattr(p, "cp_spans_dp", False)) and dp_width > 1 and self.rank_token_owners is None
        projected_map = self._resolve_receipt_projected_by_replica(
            phase=str(phase),
            sequence_tokens=sequence_tokens,
            batch_per_dp=batch_per_dp,
            query_tokens=query_tokens,
            request_views=request_views,
            request_views_by_replica=request_views_by_replica,
            sequence_lengths=sequence_lengths,
            query_lengths=query_lengths,
            dp_width=dp_width,
            spans_dp=spans_dp,
        )
        proj_host_map = self._projected_host_by_replica(
            phase=str(phase),
            sequence_tokens=sequence_tokens,
            batch_per_dp=batch_per_dp,
            query_tokens=query_tokens,
            request_views=request_views,
            request_views_by_replica=request_views_by_replica,
            sequence_lengths=sequence_lengths,
            query_lengths=query_lengths,
        )
        live_pool_map = (
            self._live_die_cache_pools(
                default_phase=str(phase), dp_width=dp_width, spans_dp=spans_dp,
            )
            if len(projected_map) < p.dp * self.width else {}
        )
        dies: list[DieResidentReceipt] = []
        for replica in range(p.dp):
            for stage in range(p.pp):
                stage_dies = self._stage_dies(replica, stage)
                host_req = (
                    proj_host_map[replica] if replica in proj_host_map else host_live[replica]
                )
                host_b = int(round(host_req + self.engram_host_dram_bytes[stage]))
                for rank in range(self.width):
                    proj_stages = projected_map.get((replica, rank))
                    projected = proj_stages[stage] if proj_stages is not None else None
                    live_stages = live_pool_map.get((replica, rank)) if projected is None else None
                    live_pool = live_stages[stage] if live_stages is not None else None
                    dyn_live_b = int(round(hbm_live[replica][stage][rank]))
                    dyn_peak_b = max(
                        dyn_live_b,
                        int(round(self.peak_rank[replica][stage][rank])),
                        int(round(self._domain_peak_hbm(replica, stage, rank))),
                    )
                    die_ckpt_b, die_term_prep_b = die_weight_splits[(replica, stage, rank)]
                    dies.append(self._build_die_receipt(
                        replica=replica, stage=stage, rank=rank, die_obj=stage_dies[rank],
                        ckpt_b=die_ckpt_b, vision_b=vision_stage[stage],
                        term_prep_b=die_term_prep_b,
                        prep_persist_b=prep_persist[stage], prep_load_b=prep_load[stage],
                        stage_prep_grants=prep_grants[stage], extra_grants=extra_grants,
                        projected=projected, live_pool=live_pool,
                        dyn_live_b=dyn_live_b, dyn_peak_b=dyn_peak_b,
                        req_b=int(round(request_ranks[replica][stage][rank])),
                        stg_b=int(round(staging_ranks[replica][stage][rank])),
                        host_b=host_b,
                        die_backings=non_req_backings[replica][stage][rank],
                        req_reclaimable_b=req_recl_ranks[replica][stage][rank],
                        req_reclaim_events=req_recl_events[replica][stage][rank],
                    ))
        return ResidentMemoryReceipt(
            scenario=scenario or str(getattr(self.case.model, "name", "")),
            phase=str(phase),
            occurrence=str(occurrence),
            vision_accounting=vis_scope,
            dies=tuple(dies),
        )

    def _component_layer_counts_and_kinds(self) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
        """Derive per-stage active layer counts and structural kind labels per component."""
        model = self.case.model
        p = self.case.runtime.parallel
        pp = max(1, int(p.pp))
        counts: dict[str, list[int]] = {}
        kinds: dict[str, str] = {}

        def _bump(comp: str, st: int, kind: str) -> None:
            row = counts.setdefault(comp, [0] * pp)
            row[st] += 1
            kinds.setdefault(comp, kind)

        for plan in model.layer_plans:
            st = self.layer_stage[plan.index] if plan.index < len(self.layer_stage) else pp - 1
            if plan.attention == "mla" and plan.cache_kind == "dense":
                _bump("mla_kv", st, "MLA Latent KV")
            elif plan.attention in {"gqa", "projection_only"} and not (0 <= plan.kv_owner_layer_id < plan.index):
                _bump("gqa_kv", st, "GQA K/V")
            if int(plan.window or 0) > 0 and plan.attention == "mla":
                _bump("window", st, "Sliding-Window KV")
            if plan.owns_long_kv and plan.cache_kind == "window_compressed":
                _bump("compressed_kv", st, "Compressed Long KV")
            if plan.owns_index_k:
                _bump("index_k", st, "Sparse Indexer K")
                if int(plan.index_kpool or 0) > 0:
                    _bump("index_pool_tail", st, "KeyPool Ring Tail")
            if int(plan.compressor_state_rows or 0) > 0:
                _bump("compressor_state", st, "C2 Compressor Ring")
            if plan.attention in {"linear", "kda", "gdn"}:
                _bump("recurrent_state", st, "Linear Recurrent SSM")
                _bump("conv_state", st, "Causal Conv1D State")
                _bump("replay_records", st, "RecoverSSM Replay Records")
                _bump("state_transient", st, "RecoverSSM Transient")
        engram = getattr(model, "engram", None)
        if engram is not None and getattr(engram, "layer_ids", ()):
            for lid in engram.layer_ids:
                if 0 <= int(lid) < len(self.layer_stage):
                    _bump("ple_conv_state", self.layer_stage[int(lid)], "PLE Conv1D State")
        if self.draft_kind and model.auxiliary is not None:
            from mindie_sim.catalog.draft import optional_draft_spec
            draft = optional_draft_spec(model, self.draft_kind)
            for _ in range(max(1, int(draft.layers))):
                _bump("draft_kv", pp - 1, f"Draft KV ({draft.attention_kind.upper()})")
                _bump("draft_state", pp - 1, "Draft Recurrent State")
        c8_stages = tuple(getattr(self, "c8_static_channel_scale_by_stage", (0,) * pp))
        for st, val in enumerate(c8_stages):
            if val > 0:
                gqa_cnt = counts.get("gqa_kv", [0] * pp)[st]
                counts.setdefault("c8_static_channel_scale", [0] * pp)[st] = gqa_cnt
                kinds.setdefault("c8_static_channel_scale", "GQA Static C8 Channel Scales (Fixed/Die)")
        kinds.setdefault("page_group_reserve", "Hybrid Pool / Page-Group Reserve")
        kinds.setdefault("page_metadata", "Page Table Metadata")
        kinds.setdefault("dcp_dp_cohort_kv", "DCP DP-Cohort Shared KV")
        kinds.setdefault("dcp_dp_cohort_index", "DCP DP-Cohort Shared Indexer K")
        kinds.setdefault("dcp_dp_cohort_draft", "DCP DP-Cohort Shared Draft KV")
        return {k: tuple(v) for k, v in counts.items()}, kinds

    def cache_ledger_projection(
        self,
        *,
        scenario: str = "",
        phase: str = "decode",
        occurrence: str = "steady",
        sequence_tokens: int,
        batch_per_dp: int,
        query_tokens: int | None = None,
        vision_accounting: str | None = None,
        prepared_weights: Sequence[PreparedWeightSpec] = (),
        extra_grants: Sequence[ResidentGrantEntry] = (),
    ) -> CacheLedgerProjection:
        """Build authoritative ``ResidentMemoryReceipt`` paired with exact per-component integers."""
        from mindie_sim.catalog.cache import DEFAULT_CACHE_BLOCK_SIZE

        p = self.case.runtime.parallel
        seq = max(0, int(sequence_tokens))
        b_int = max(0, int(batch_per_dp))
        receipt = self.resident_memory_receipt(
            scenario=scenario,
            phase=phase,
            occurrence=occurrence,
            sequence_tokens=seq if b_int > 0 else None,
            batch_per_dp=b_int,
            query_tokens=query_tokens,
            vision_accounting=vision_accounting,
            prepared_weights=prepared_weights,
            extra_grants=extra_grants,
        )
        rank_divergent = self.rank_token_owners is not None or int(getattr(p, "cp", 1) or 1) > 1
        dp_width = max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
        spans_dp = bool(getattr(p, "cp_spans_dp", False)) and dp_width > 1 and self.rank_token_owners is None
        bottleneck_ranks: list[int] = []
        stage_comp_batch: list[dict[str, int]] = []
        stage_comp_per_req: list[dict[str, float]] = []
        hybrid_receipt: Any = None
        cached_uniform: list[dict[str, Any]] | None = None
        for st in range(p.pp):
            st_dies = [d for d in receipt.dies if d.stage == st]
            if st_dies:
                best_die = max(
                    st_dies,
                    key=lambda d: (d.simultaneous_peak_hbm_bytes, d.simultaneous_live_hbm_bytes),
                )
                best_rk = int(best_die.rank)
                phys_rk = (
                    ((int(best_die.replica) % dp_width) * p.pp) * self.width + best_rk
                    if spans_dp else best_rk
                )
            else:
                best_rk = max(
                    range(self.width),
                    key=lambda rk: receipt.die_receipt(0, st, rk).simultaneous_live_hbm_bytes,
                )
                phys_rk = best_rk
            bottleneck_ranks.append(best_rk)
            if seq <= 0 or b_int <= 0:
                c8_b = int(self.c8_static_channel_scale_by_stage[st])
                stage_comp_batch.append({"c8_static_channel_scale": c8_b} if c8_b > 0 else {})
                stage_comp_per_req.append({"c8_static_channel_scale": 0.0} if c8_b > 0 else {})
                continue
            if rank_divergent:
                rank_stages, h_rec = self._projected_die_cache_pools(
                    seq, b_int, query_tokens, rank=phys_rk, phase=str(phase),
                )
                entry = rank_stages[st]
                if hybrid_receipt is None and h_rec is not None:
                    hybrid_receipt = h_rec
            else:
                if cached_uniform is None:
                    cached_uniform, hybrid_receipt = self._projected_die_cache_pools(
                        seq, b_int, query_tokens, rank=None, phase=str(phase),
                    )
                entry = cached_uniform[st]
            stage_comp_batch.append(dict(entry["components_batch"]))
            stage_comp_per_req.append(dict(entry["components_per_req"]))

        layer_counts, kinds = self._component_layer_counts_and_kinds()
        raw_block = max(1, int(getattr(self.case.runtime, "kv_block_size", 0) or DEFAULT_CACHE_BLOCK_SIZE))
        norm_block = (
            int(hybrid_receipt.contract.normalized_block_size)
            if hybrid_receipt is not None else raw_block
        )
        return CacheLedgerProjection(
            receipt=receipt,
            bottleneck_rank_by_stage=tuple(bottleneck_ranks),
            per_stage_component_batch_bytes=tuple(stage_comp_batch),
            per_stage_component_per_req_bytes=tuple(stage_comp_per_req),
            active_layers_by_component=layer_counts,
            cache_kind_by_component=kinds,
            fixed_per_die_components=frozenset({"c8_static_channel_scale"}),
            normalized_block_size=norm_block,
            hybrid_pool_receipt=hybrid_receipt,
        )

    def evaluate_space_time_tradeoff(
        self,
        *,
        mechanism: str,
        delta_hbm_bytes: int,
        delta_step_ms: float = 0.0,
        delta_concurrency_per_dp: int = 0,
        delta_tps_per_die: float = 0.0,
        replica: int = 0,
        stage: int | None = None,
        rank: int | None = None,
    ) -> SpaceTimeTradeoffSpec:
        """Evaluate a space-time trade-off against the authoritative free HBM headroom."""
        headroom = self.free_hbm_headroom_bytes(replica=replica, stage=stage, rank=rank)
        direction = "space_for_time" if int(delta_hbm_bytes) >= 0 else "time_for_space"
        feasible = bool(int(delta_hbm_bytes) <= headroom)
        return SpaceTimeTradeoffSpec(
            direction=direction,
            mechanism=mechanism,
            delta_hbm_bytes=int(delta_hbm_bytes),
            free_hbm_headroom_bytes=int(headroom),
            delta_step_ms=float(delta_step_ms),
            delta_concurrency_per_dp=int(delta_concurrency_per_dp),
            delta_tps_per_die=float(delta_tps_per_die),
            feasible_under_hbm_and_sla=feasible,
        )
