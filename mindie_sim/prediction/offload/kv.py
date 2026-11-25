"""KV and recurrent-state offload transfer scheduling.

[Role] Schedule layer and iteration reads and producer-gated writebacks.
[Invariants] Draft and target accesses retain their declared scopes.
[Timeouts & Locks] Bounded by active accesses. No I/O, waits or locks.
[Side-Effect Policy] Submits transfers and records events on the caller's execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .policy import _host_ddr_work, _submit_offload_transfer
from .accesses import (
    StateAccess,
    _access_bytes,
    _as_accesses,
    _unique_accesses,
    is_persistent_offload_access,
)
from .events import (
    _access_registry,
    _accesses_for_work,
    _event_for_access,
    _event_table,
    _has_event_bindings,
    _merge_event_tables,
    _register_access_events,
    _registered_events,
    _work_event_table,
)


def is_kv_or_state_entry_op(op: Any) -> bool:
    if getattr(op, "layer", -1) < 0:
        return False
    # A draft producer can share a numeric layer index with the target.  It
    # must not enter the target layer fetch/writeback path; T04 schedules its
    # aggregate writeback separately with schedule_draft_kv_writeback.
    if dict(getattr(op, "metadata", ()) or ()).get("producer_scope") == "draft":
        return False
    return getattr(op, "family", "") in {
        "attention", "attention_online", "mla", "mla_decode",
        "compressed_attention", "dsa", "indexer", "recurrent_scan",
        "recurrent_vector", "gdn_state", "gdn_recurrent", "kda",
        "kda_recurrent", "short_convolution", "cache_store",
    } or getattr(op, "recipe", "") in {"short_convolution", "causal_conv1d_update"}


def _rank_bytes(accesses) -> dict[int | None, float]:
    by_rank: dict[int | None, float] = {}
    for access in _unique_accesses(accesses):
        if access.bytes > 0.0:
            by_rank[access.rank] = by_rank.get(access.rank, 0.0) + access.bytes
    return by_rank


def _submit_kv_tier_transfers(
    execution: Any, stem: str, replica: int, stage: int, kv_bytes: float,
    anchor: tuple[Any, ...], *, discount_hit: bool, direction: str = "h2d",
    rank_bytes: Mapping[int | None, float] | None = None,
) -> tuple[Any, ...]:
    if kv_bytes <= 0.0:
        return ()
    policy = execution.offload_policy
    scale = (1.0 - policy.mtp_kvc_hit_rate) if discount_hit else 1.0
    hbm_res = execution.stage_names[replica, stage]["hbm"]
    events = []
    ddr_factor = policy.kvc_ddr_ratio * scale
    ddr_bytes = kv_bytes * ddr_factor
    if ddr_bytes > 0.0:
        rank_ddr = (
            {rank: amt * ddr_factor for rank, amt in rank_bytes.items() if amt * ddr_factor > 0.0}
            if rank_bytes else None
        )
        ddr_work = _host_ddr_work(
            execution, replica, stage, direction, ddr_bytes,
            rank_ddr_bytes=rank_ddr,
        )
        events.append(_submit_offload_transfer(
            execution.engine, f"{stem}_ddr", ddr_work,
            after=anchor, latency_us=policy.ddr_startup_latency_us(direction),
            owner=f"replica:{replica}", coupled=len(ddr_work) > 1))
    ssd_bytes = kv_bytes * policy.kvc_ssd_ratio * scale
    if ssd_bytes > 0.0:
        ssd_res = f"{execution.namespace}offload_ssd@r{replica}:s{stage}"
        ssd_work = {ssd_res: ssd_bytes, hbm_res: ssd_bytes}
        events.append(_submit_offload_transfer(
            execution.engine, f"{stem}_ssd", ssd_work,
            after=anchor, latency_us=policy.ssd_latency_us,
            owner=f"replica:{replica}", coupled=len(ssd_work) > 1))
    return tuple(events)


_STATE_WRITER_FAMILIES = frozenset({
    "cache_store", "state_scatter", "state", "kv_store", "gdn_state",
    "gdn_recurrent", "recurrent_vector", "recurrent_scan",
    "short_convolution",
})


def _writer_occurrences(physical_ops) -> set[str]:
    """Ops whose own completion can be the actual state/output writer."""
    writers: set[str] = set()
    for op in physical_ops or ():
        occurrence = str(getattr(op, "op_id", ""))
        meta = dict(getattr(op, "metadata", ()) or ())
        if (str(getattr(op, "family", "")) in _STATE_WRITER_FAMILIES
                or meta.get("cache_kind")
                or float(meta.get("state_write_bytes", 0.0) or 0.0) > 0.0
                or "state_accesses" in meta
                or "memory_accesses" in meta):
            writers.add(occurrence)
    return writers


def _writeback_groups(accesses, producer_done: Any, table: Mapping | None, *,
                      explicit: bool, error: str, batch_outer: bool = False):
    """Group pending writes by their actual producer.output Event.

    A single outer completion is reused only for an explicitly selected
    single-producer write set, or when the selected occurrence itself is the
    declared producer.  Multiple producers without per-access bindings stay
    unresolved; they are not sent on one early Event.
    """
    producers = {access.producer for access in accesses}
    if (batch_outer and explicit and producer_done is not None
            and not _has_event_bindings(table)):
        # The documented draft aggregate writeback is one batch-level write
        # after the last draft op; its producer handle is the batch completion,
        # not one component label.  Per-output bindings still take precedence.
        groups: dict[tuple[Any, int], list[StateAccess]] = {}
        for access in accesses:
            groups.setdefault((producer_done, access.stage), []).append(access)
        return groups
    allow_outer = (
        producer_done is not None and len(producers) == 1
        and (explicit or all(access.occurrence == access.producer
                             for access in accesses))
    )
    groups: dict[tuple[Any, int], list[StateAccess]] = {}
    unresolved: list[StateAccess] = []
    for access in accesses:
        event = _event_for_access(access, table)
        if event is None and allow_outer:
            event = producer_done
        if event is None:
            unresolved.append(access)
            continue
        groups.setdefault((event, access.stage), []).append(access)
    if (unresolved and len(producers) > 1
            and not _has_event_bindings(table)):
        raise ValueError(error)
    return groups


def _submit_writeback_groups(execution: Any, stem: str, replica: int,
                             work: Any, groups) -> tuple[Any, ...]:
    """Submit one tier copy set per actual producer Event and register it."""
    if not groups:
        return ()
    multiple = len(groups) > 1
    events: list[Any] = []
    for index, ((event, stage), group) in enumerate(sorted(
            groups.items(), key=lambda item: (item[0][1], repr(item[0][0])))):
        write_bytes = sum(access.bytes for access in group)
        if write_bytes <= 0.0:
            continue
        name = stem if not multiple else f"{stem}:{index}"
        submitted = _submit_kv_tier_transfers(
            execution, name, replica, stage, write_bytes, (event,),
            discount_hit=False, direction="d2h", rank_bytes=_rank_bytes(group))
        _register_access_events(execution, work, group, submitted)
        events.extend(submitted)
    return tuple(events)


def schedule_kv_layer_offload(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    work: Any, input_ready: Any, consumer_deps: tuple[Any, ...],
    layer_entries: Mapping[int, Any], *, prev_kv_done: Any = None,
    state_accesses: Any = None, source_ready: Any = None,
    grant_ready: Any = None, decision: Any = None,
) -> tuple[Any, ...]:
    """Submit per-layer KV cache DDR/SSD fetches before the layer's attention consumer."""
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    accesses = _unique_accesses(_as_accesses(state_accesses) if state_accesses is not None
                                else _accesses_for_work(execution, work))
    selected = [access for access in accesses
                if access.direction == "read" and access.stage == stage
                and is_persistent_offload_access(access)
                and (state_accesses is not None or access.scope == "target")
                and (access.layer == layer or access.layer < 0)]
    registry = _access_registry(execution, work)
    existing = _registered_events(execution, work, selected)
    pending = [access for access in selected if access.key not in registry]
    kv_bytes = _access_bytes(pending, "read")
    if kv_bytes <= 0.0:
        return existing
    mandatory = tuple(item for item in (source_ready, grant_ready, decision)
                      if item is not None)
    if policy.kvc_prefetch_version == "v1":
        anchor = (input_ready,) if mandatory else (tuple(consumer_deps) or (input_ready,))
    elif policy.kvc_prefetch_version == "v2":
        anchor = (layer_entries.get(layer, input_ready),)
    elif policy.kvc_prefetch_version == "v3":
        anchor = (layer_entries.get(layer - 1, input_ready),)
    else:
        anchor = (input_ready,)
    anchor = tuple(dict.fromkeys((*anchor, *mandatory)))
    if prev_kv_done is not None:
        anchor = tuple(dict.fromkeys((*anchor, prev_kv_done)))
    events = _submit_kv_tier_transfers(
        execution, f"{prefix}:s{stage}.l{layer}:kv_offload", replica, stage,
        kv_bytes, anchor, discount_hit=True, direction="h2d",
        rank_bytes=_rank_bytes(pending))
    _register_access_events(execution, work, pending, events)
    return tuple(dict.fromkeys((*existing, *events)))


def schedule_kv_layer_writeback(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    work: Any, producer_done: Any, *, state_accesses: Any = None,
    access_events: Any = None,
) -> tuple[Any, ...]:
    """Submit per-layer KV/state writebacks to DDR/SSD after their writers.

    Each pending physical write uses the same batch's actual
    producer.output->Event binding.  A single outer completion is reused only
    for one explicitly selected producer.  A write whose real output Event is
    not available is left unissued and unregistered for a later supplement.
    """
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    accesses = _unique_accesses(_as_accesses(state_accesses) if state_accesses is not None
                                else _accesses_for_work(execution, work))
    selected = [access for access in accesses
                if access.direction == "write" and access.stage == stage
                and is_persistent_offload_access(access)
                and access.scope == "target"
                and (access.layer == layer or access.layer < 0)]
    registry = _access_registry(execution, work)
    existing = _registered_events(execution, work, selected)
    pending = [access for access in selected if access.key not in registry]
    if not pending:
        return existing
    table = _merge_event_tables(
        _work_event_table(execution, work), _event_table(access_events))
    groups = _writeback_groups(
        pending, producer_done, table,
        explicit=state_accesses is not None,
        error="state_writeback_requires_one_producer_event_per_call")
    events = _submit_writeback_groups(
        execution, f"{prefix}:s{stage}.l{layer}:kv_writeback", replica,
        work, groups)
    return tuple(dict.fromkeys((*existing, *events)))


def schedule_draft_kv_writeback(
    execution: Any, prefix: str, replica: int, stage: int,
    work: Any, producer_done: Any, *, state_accesses: Any = None,
    access_events: Any = None,
) -> tuple[Any, ...]:
    """Write actual draft-materialized rows after the draft producer completes.

    This is a separate producer identity from the target layer with the same
    numeric index.  T04 should call it once per executed batch after the last
    draft op completes, not once per target layer.  ``proposal=0`` and a
    disabled prefill preparation produce no record and no event.
    """
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    accesses = _unique_accesses(_as_accesses(state_accesses) if state_accesses is not None
                                else _accesses_for_work(execution, work))
    selected = [access for access in accesses
                if access.direction == "write" and access.stage == stage
                and is_persistent_offload_access(access)
                and access.scope == "draft"]
    registry = _access_registry(execution, work)
    existing = _registered_events(execution, work, selected)
    pending = [access for access in selected if access.key not in registry]
    if not pending:
        return existing
    table = _merge_event_tables(
        _work_event_table(execution, work), _event_table(access_events))
    groups = _writeback_groups(
        pending, producer_done, table,
        explicit=state_accesses is not None,
        error="draft_writeback_requires_one_producer_event_per_call",
        batch_outer=True)
    events = _submit_writeback_groups(
        execution, f"{prefix}:s{stage}:draft_kv_writeback", replica,
        work, groups)
    return tuple(dict.fromkeys((*existing, *events)))



def schedule_kv_iteration_offload(
    execution: Any, prefix: str, replica: int, work: Any, input_ready: Any,
    physical_ops: tuple[Any, ...] | list[Any], ids: Mapping[str, Any],
    scheduled_layers: set[tuple[int, int]] | None = None, *,
    state_accesses: Any = None, scheduled_accesses: Any = None,
    access_events: Any = None,
) -> tuple[Any, ...]:
    """Submit fallback writebacks for real, still-unissued physical writes.

    The fallback uses the same per-work issued registry and the same actual
    access->producer.output Event bindings as the layer/draft paths.  It never
    substitutes an unrelated outer occurrence ID or matches Event.name.
    """
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    registry = _access_registry(execution, work)
    if state_accesses is not None:
        accesses = _unique_accesses(_as_accesses(state_accesses))
    else:
        base = _accesses_for_work(execution, work)
        allowed = {str(getattr(op, "op_id", "")) for op in physical_ops}
        accesses = tuple(access for access in base
                         if access.occurrence in allowed)
    table = _merge_event_tables(
        _work_event_table(execution, work), _event_table(access_events))
    submitted = {str(getattr(op, "op_id", "")) for op in physical_ops}
    writers = _writer_occurrences(physical_ops)
    done = set(scheduled_accesses or ()) | set(registry)
    pending = [access for access in accesses
               if access.key not in done and access.direction == "write"
               and is_persistent_offload_access(access)]
    pp = max(1, int(execution.case.runtime.parallel.pp))
    groups: dict[tuple[Any, int], list[StateAccess]] = {}
    for access in pending:
        event = _event_for_access(access, table)
        if event is None and access.producer in ids:
            event = ids.get(access.producer)
        if event is None and access.producer not in submitted:
            # The declared producer is a port not submitted as its own op.
            # Only the occurrence that is itself a real state writer can stand
            # for that output; a generic call/outer ID cannot.
            if (access.occurrence == access.producer
                    or access.occurrence in writers):
                event = ids.get(access.occurrence)
        if event is None:
            # Real unissued write without its own producer.output completion:
            # preserve it for the supplement, do not emit or register it.
            continue
        stage = min(max(0, access.stage), pp - 1)
        groups.setdefault((event, stage), []).append(access)

    events: list[Any] = []
    multiple = len(groups) > 1
    for index, ((producer_event, stage), group) in enumerate(sorted(
            groups.items(), key=lambda item: (
                item[0][1], repr(item[0][0])))):
        write_bytes = sum(access.bytes for access in group)
        if write_bytes <= 0.0:
            continue
        stem = f"{prefix}:s{stage}:kv_writeback"
        if multiple:
            stem = f"{stem}:{index}"
        submitted_events = _submit_kv_tier_transfers(
            execution, stem, replica, stage, write_bytes, (producer_event,),
            discount_hit=False, direction="d2h", rank_bytes=_rank_bytes(group))
        _register_access_events(execution, work, group, submitted_events)
        events.extend(submitted_events)
    return tuple(events)
