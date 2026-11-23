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

from collections.abc import Sequence
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
        }


@dataclass(frozen=True)
class DieResidentReceipt:
    """Simultaneous live and peak HBM residency on one physical die."""

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
            "simultaneous_peak_hbm_bytes": int(self.simultaneous_peak_hbm_bytes),
            "load_peak_hbm_bytes": int(self.load_peak_hbm_bytes),
            "free_hbm_headroom_bytes": int(self.free_hbm_headroom_bytes),
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
    def peak_die_hbm_bytes(self) -> int:
        return max((die.simultaneous_peak_hbm_bytes for die in self.dies), default=0)

    @property
    def min_free_hbm_headroom_bytes(self) -> int:
        return min((die.free_hbm_headroom_bytes for die in self.dies), default=0)

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
            "peak_die_hbm_bytes": self.peak_die_hbm_bytes,
            "min_free_hbm_headroom_bytes": self.min_free_hbm_headroom_bytes,
            "dies": [die.to_dict() for die in self.dies],
        }


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

    def _update_combined_peak(self, replica: int, stage: int) -> None:
        """Compatibility wrapper: peaks are committed-state only."""
        self._sync_usage()

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
        self.budget_version += 1

    def free_hbm_headroom_bytes(
        self,
        replica: int = 0,
        stage: int | None = None,
        rank: int | None = None,
    ) -> int:
        """Authoritative free HBM headroom on the bottleneck (or addressed) die."""
        p = self.case.runtime.parallel
        if not 0 <= int(replica) < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        hbm, _ = self._state_usage()
        stages = (int(stage),) if stage is not None else tuple(range(p.pp))
        ranks = (self._account_rank(rank),) if rank is not None else tuple(range(self.width))
        headrooms: list[int] = []
        for st in stages:
            if not 0 <= st < p.pp:
                raise ValueError("memory_stage_out_of_range")
            for rk in ranks:
                cap = float(self.rank_capacity[replica][st][rk])
                used = max(float(hbm[replica][st][rk]), float(self.peak_rank[replica][st][rk]))
                headrooms.append(int(math.floor(cap - used)))
        return min(headrooms, default=0)

    def _resolve_vision_delta_by_stage(self, vision_accounting: str) -> tuple[list[int], list[int]]:
        """Return ``(vision_bytes_by_stage, weight_delta_by_stage)`` for vision scope."""
        from mindie_sim.catalog.types import QuantSpec
        from mindie_sim.catalog.weight_terms import term_bytes
        from mindie_sim.catalog.weights import vision_weight_terms

        p = self.case.runtime.parallel
        pp = max(1, int(p.pp))
        if vision_accounting not in VISION_ACCOUNTING_SCOPES:
            raise ValueError(f"vision_accounting_invalid:{vision_accounting!r}")
        model = self.case.model
        quant = model.quant or QuantSpec("bf16")
        vision_all = int(round(sum(
            term_bytes(term, quant, p, self.case.machine.ranks_per_node)
            for term in vision_weight_terms(model, include_vision=True)
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

    def _stage_checkpoint_vs_prepared_split(self, stage: int) -> tuple[int, int]:
        """Split ``self.weights[stage]`` into source checkpoint vs runtime-only prepared bytes."""
        from mindie_sim.catalog.types import QuantSpec
        from mindie_sim.catalog.weight_terms import term_bytes
        from mindie_sim.catalog.weights import _expand_resident_terms, stage_weight_terms

        model = self.case.model
        p = self.case.runtime.parallel
        quant = model.quant or QuantSpec("bf16")
        include_vision = self.vision_accounting == "full_checkpoint"
        all_stage_terms = stage_weight_terms(
            model, p, include_draft=bool(self.draft_kind),
            draft_kind=self.draft_kind or None,
            include_engram_tables=not self.case.runtime.engram_cpu_offload,
            include_vision=include_vision, layer_partition=self.layers,
            slot_plans=self.slot_plans,
        )
        runtime_only = 0.0
        for term in all_stage_terms[stage]:
            if term.source_owned:
                continue
            for t_sub, placed in _expand_resident_terms(term, p, False, model=model):
                runtime_only += term_bytes(t_sub, quant, placed, self.case.machine.ranks_per_node)
        prep_b = int(round(runtime_only))
        ckpt_b = max(0, int(round(self.weights[stage])) - prep_b)
        return ckpt_b, prep_b

    def _unified_pool_stride_reserve_by_stage(
        self,
        sequence_tokens: int,
        *,
        query_tokens: int | None = None,
    ) -> list[float]:
        """Per-stage per-request unified hybrid-pool stride alignment reserve."""
        p = self.case.runtime.parallel
        zeros = [0.0] * p.pp
        seq = int(sequence_tokens)
        if seq <= 0:
            return zeros
        from mindie_sim.catalog.qwen4_exp import qwen4_exp_pool_breakdown

        q_tok = self._effective_cache_query_tokens(query_tokens)
        cp = max(1, int(getattr(p, "cp", 1) or 1))
        eff_seq = (seq + cp - 1) // cp if cp > 1 else seq
        qwen4_pool = qwen4_exp_pool_breakdown(
            self.case.model, p, eff_seq,
            runtime=self.case.runtime, query_tokens=q_tok,
        )
        if not qwen4_pool:
            return zeros
        alloc_stages = self._cache_component_breakdown(
            seq, padded=True, skip_covered=True, query_tokens=q_tok,
        )
        target_base = 0.0
        for st in range(p.pp):
            for name, detail in alloc_stages[st]["components"].items():
                if not name.startswith("draft_"):
                    target_base += float(detail.get("total", 0.0) or 0.0)
        if self.pages.groups:
            group_payload = self.pages.stage_bytes(seq, self.case.runtime.max_model_len, p.pp)
            group_parts = self._group_cache_parts(seq, query_tokens=q_tok)
            for st in range(p.pp):
                target_base += max(
                    float(group_payload[st]),
                    float(group_parts[st]["alloc_payload"]) + float(group_parts[st]["alloc_scale"]),
                )
        stride_reserve = max(0.0, float(qwen4_pool["total_pool_bytes"]) - target_base)
        if stride_reserve <= 0.0:
            return zeros
        qsa_per_stage = [0] * p.pp
        for lp in self.case.model.layer_plans:
            if lp.attention == "gqa" and lp.indexer:
                st = self.layer_stage[lp.index]
                if 0 <= st < p.pp:
                    qsa_per_stage[st] += 1
        total_qsa = sum(qsa_per_stage)
        if total_qsa <= 0:
            zeros[0] = stride_reserve
            return zeros
        return [stride_reserve * (float(cnt) / float(total_qsa)) for cnt in qsa_per_stage]

    def _projected_stage_cache_pools(
        self,
        sequence_tokens: int | None,
        batch_per_dp: int,
        query_tokens: int | None,
    ) -> list[dict[str, int]] | None:
        """Per-stage projected KV/Index/State/Draft bytes for ``batch_per_dp`` requests."""
        if sequence_tokens is None or int(batch_per_dp) <= 0:
            return None
        from .cache import _COMPONENT_POOL_MAP

        p = self.case.runtime.parallel
        b_int = int(batch_per_dp)
        alloc_stages = self._cache_component_breakdown(
            int(sequence_tokens), padded=True, skip_covered=True, query_tokens=query_tokens,
        )
        valid_stages = self._cache_component_breakdown(
            int(sequence_tokens), padded=False, skip_covered=True, query_tokens=query_tokens,
        )
        group_parts = (
            self._group_cache_parts(int(sequence_tokens), query_tokens=query_tokens)
            if self.pages.groups else None
        )
        group_payload = (
            self.pages.stage_bytes(int(sequence_tokens), self.case.runtime.max_model_len, p.pp)
            if self.pages.groups else None
        )
        group_metadata = (
            self.pages.metadata_stage_bytes(int(sequence_tokens), self.case.runtime.max_model_len, p.pp)
            if self.pages.groups else None
        )
        stage_cache_bytes = self.cache_bytes(
            int(sequence_tokens), padded=True, query_tokens=query_tokens,
        )
        stride_reserves = self._unified_pool_stride_reserve_by_stage(
            int(sequence_tokens), query_tokens=query_tokens,
        )
        per_stage: list[dict[str, int]] = []
        for stage in range(p.pp):
            alloc_pools = {"target_kv_bytes": 0.0, "indexer_k_bytes": 0.0,
                           "recurrent_state_bytes": 0.0, "draft_kv_bytes": 0.0,
                           "draft_state_bytes": 0.0}
            valid_pools = {"target_kv_bytes": 0.0, "indexer_k_bytes": 0.0}
            for name, detail in alloc_stages[stage]["components"].items():
                key = _COMPONENT_POOL_MAP.get(name)
                if key in alloc_pools:
                    alloc_pools[key] += float(detail["total"])
            for name, detail in valid_stages[stage]["components"].items():
                key = _COMPONENT_POOL_MAP.get(name)
                if key in valid_pools:
                    valid_pools[key] += float(detail["total"])
            group_reserve = 0.0
            if group_parts is not None and group_payload is not None and group_metadata is not None:
                known_group = 0.0
                for name, detail in group_parts[stage]["components"].items():
                    key = _COMPONENT_POOL_MAP.get(name)
                    part_alloc = float(detail["alloc_payload"]) + float(detail["alloc_scale"])
                    if key in alloc_pools:
                        alloc_pools[key] += part_alloc
                        known_group += part_alloc
                    if key in valid_pools:
                        valid_pools[key] += float(detail["valid_payload"]) + float(detail["valid_scale"])
                group_reserve = max(0.0, float(group_payload[stage]) - known_group)
                alloc_pools["target_kv_bytes"] += group_reserve + float(group_metadata[stage])
            stride_reserve = float(stride_reserves[stage]) if stage < len(stride_reserves) else 0.0
            if stride_reserve > 0.0:
                alloc_pools["target_kv_bytes"] += stride_reserve
            idx_a = int(round(b_int * alloc_pools["indexer_k_bytes"]))
            rec_a = int(round(b_int * alloc_pools["recurrent_state_bytes"]))
            drf_a = int(round(b_int * (alloc_pools["draft_kv_bytes"] + alloc_pools["draft_state_bytes"])))
            expected_total = int(round(b_int * (float(stage_cache_bytes[stage]) + stride_reserve)))
            non_kv = idx_a + rec_a + drf_a
            kv_a = (
                expected_total - non_kv
                if expected_total >= non_kv
                else int(round(b_int * alloc_pools["target_kv_bytes"]))
            )
            per_stage.append({
                "kv_alloc": kv_a,
                "kv_valid": int(round(b_int * valid_pools["target_kv_bytes"])),
                "idx_alloc": idx_a,
                "idx_valid": int(round(b_int * valid_pools["indexer_k_bytes"])),
                "rec": rec_a,
                "draft": drf_a,
                "page_group_reserve": int(round(b_int * (group_reserve + stride_reserve))),
            })
        return per_stage

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
        dyn_live_b: int,
        dyn_peak_b: int,
        req_b: int,
        stg_b: int,
        host_b: int,
    ) -> DieResidentReceipt:
        usable = int(math.floor(die_obj.hbm_bytes * self.case.runtime.gpu_memory_utilization))
        ws_b = int(round(self.workspace[stage] + self.temporary_dequant[stage]))
        res_b = int(round(self.policy.reserved))
        dyn_grant_only = max(0, dyn_live_b - req_b - stg_b)
        kv_a = projected["kv_alloc"] if projected is not None else req_b
        kv_v = projected["kv_valid"] if projected is not None else req_b
        idx_a = projected["idx_alloc"] if projected is not None else 0
        idx_v = projected["idx_valid"] if projected is not None else 0
        rec_b = projected["rec"] if projected is not None else 0
        drf_b = projected["draft"] if projected is not None else 0

        ckpt_gid = f"r{replica}:s{stage}:k{rank}:checkpoint_weights"
        vis_gid = f"r{replica}:s{stage}:k{rank}:vision_weights"
        term_gid = f"r{replica}:s{stage}:k{rank}:prepared_terms"
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
        raw_entries.extend(stage_prep_grants)
        raw_entries.extend(extra_grants)
        dedup_grants, _ = _deduplicate_grant_entries(raw_entries)
        base_gids = {ckpt_gid, vis_gid, term_gid}
        extra_only_billed = sum(
            g.billed_once_bytes for g in dedup_grants
            if g.grant_id not in base_gids and not g.grant_id.startswith("prepared:")
        )
        prep_total = term_prep_b + prep_persist_b
        fixed_total = ckpt_b + vision_b + prep_total + ws_b + res_b
        if projected is not None:
            live_dyn = kv_a + idx_a + rec_b + drf_b + dyn_grant_only + stg_b + extra_only_billed
        else:
            live_dyn = dyn_live_b + extra_only_billed
        peak_dyn = max(live_dyn, dyn_peak_b + extra_only_billed)
        sim_live = fixed_total + live_dyn
        sim_peak = fixed_total + peak_dyn
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
            staging_bytes=stg_b,
            dynamic_grant_bytes=dyn_grant_only + extra_only_billed,
            simultaneous_live_hbm_bytes=sim_live,
            simultaneous_peak_hbm_bytes=sim_peak,
            load_peak_hbm_bytes=sim_peak + prep_load_b,
            free_hbm_headroom_bytes=usable - sim_peak,
            host_dram_bytes=host_b,
            grants=dedup_grants,
        )

    def resident_memory_receipt(
        self,
        *,
        scenario: str = "",
        phase: str = "decode",
        occurrence: str = "steady",
        sequence_tokens: int | None = None,
        batch_per_dp: int = 0,
        query_tokens: int | None = None,
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
        ckpt_by_stage: list[int] = []
        term_prep_by_stage: list[int] = []
        for st in range(p.pp):
            ck, tp_b = self._stage_checkpoint_vs_prepared_split(st)
            ckpt_by_stage.append(max(0, ck + vision_delta[st] - vision_stage[st]))
            term_prep_by_stage.append(tp_b)
        projected_stages = self._projected_stage_cache_pools(
            sequence_tokens, batch_per_dp, query_tokens,
        )
        dies: list[DieResidentReceipt] = []
        for replica in range(p.dp):
            for stage in range(p.pp):
                stage_dies = self._stage_dies(replica, stage)
                projected = projected_stages[stage] if projected_stages is not None else None
                host_b = int(round(host_live[replica] + self.engram_host_dram_bytes[stage]))
                for rank in range(self.width):
                    dyn_live_b = int(round(hbm_live[replica][stage][rank]))
                    dyn_peak_b = max(dyn_live_b, int(round(self.peak_rank[replica][stage][rank])))
                    dies.append(self._build_die_receipt(
                        replica=replica, stage=stage, rank=rank, die_obj=stage_dies[rank],
                        ckpt_b=ckpt_by_stage[stage], vision_b=vision_stage[stage],
                        term_prep_b=term_prep_by_stage[stage],
                        prep_persist_b=prep_persist[stage], prep_load_b=prep_load[stage],
                        stage_prep_grants=prep_grants[stage], extra_grants=extra_grants,
                        projected=projected, dyn_live_b=dyn_live_b, dyn_peak_b=dyn_peak_b,
                        req_b=int(round(request_ranks[replica][stage][rank])),
                        stg_b=int(round(staging_ranks[replica][stage][rank])),
                        host_b=host_b,
                    ))
        return ResidentMemoryReceipt(
            scenario=scenario or str(getattr(self.case.model, "name", "")),
            phase=str(phase),
            occurrence=str(occurrence),
            vision_accounting=vis_scope,
            dies=tuple(dies),
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
