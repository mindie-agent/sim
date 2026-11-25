"""Directional HBM bus aliasing, work expansion, and live task retrofit helpers.

[Role] Map scalar shipping HBM work onto bound directional ``:r``/``:w`` bus
  rows while keeping live incremental scalar HBM accounting and delivery
  progress tied to independent read and write directional bus progress across
  Python and C11 execution.
[Invariants] Directional bus rows govern bus contention exclusively once bound;
  scalar HBM work order is preserved in-place for delivery reference semantics;
  cross-charged 2-row HBM transfers advance their read/write controller rows in
  lockstep while preserving caller independence for unrelated resources;
  split children inherit parent occurrence identity for live observers.
[Timeouts & Locks] Pure bounded in-memory dict/task rewrites; no locks or I/O.
[Side-Effect Policy] Mutates only live task work/accounting fields and Engine
  service counters on step sync, split-cohort partition, or retrofit.
"""
from __future__ import annotations

import math
import re
from typing import Mapping

from .memory_physics import HbmBusSpec

# (base, total_bytes, rem_r_at_bind, rem_w_at_bind, orig_cap,
#  row_r, t_r_at_bind, row_w, t_w_at_bind, dir_mode)
HbmAccountingEntry = tuple[str, float, float, float, float | None, str, float, str, float, str]

_READ_PIPE_PREFIXES = frozenset({"mte2", "mte2_aic", "mte2_aiv"})
_WRITE_PIPE_PREFIXES = frozenset({"mte3", "fixpipe"})
_READ_NAME_TOKENS = frozenset({
    "read", "load", "fetch", "prefetch", "gather", "reload", "in", "input", "counts",
})
_WRITE_NAME_TOKENS = frozenset({
    "write", "store", "spill", "writeback", "evict", "scatter", "out", "output",
})
_NAME_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_ROW_READ_SUFFIX = ":r"
_ROW_WRITE_SUFFIX = ":w"
_UNIT_ROW_RATE = 1.0


def _replace_keys_in_place(
    mapping: Mapping[str, float],
    replace_keys: tuple[str, ...],
    row_r: str, val_r: float, row_w: str, val_w: float,
) -> dict[str, float]:
    """Replace any key in ``replace_keys`` in-place preserving first-match order."""
    match = set(replace_keys)
    match.add(row_r)
    match.add(row_w)
    out: dict[str, float] = {}
    inserted = False
    for k, v in mapping.items():
        if k in match:
            if not inserted:
                if val_r > 0.0:
                    out[row_r] = val_r
                if val_w > 0.0:
                    out[row_w] = val_w
                inserted = True
            continue
        out[k] = v
    if not inserted:
        if val_r > 0.0:
            out[row_r] = val_r
        if val_w > 0.0:
            out[row_w] = val_w
    return out


def _clamp01(value: float) -> float:
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else value)


def _hbm_entry_remaining_bytes(
    remaining: Mapping[str, float], entry: HbmAccountingEntry,
) -> tuple[float, float]:
    """Return exact current remaining ``(rem_r_bytes, rem_w_bytes)`` from the entry's bind snapshot."""
    _, _, rem_r, rem_w, _, row_r, t_r, row_w, t_w, dir_mode = entry
    if dir_mode == "mixed":
        cur_r = rem_r * _clamp01(remaining.get(row_r, 0.0) / t_r) if (rem_r > 0.0 and t_r > 0.0) else 0.0
        cur_w = rem_w * _clamp01(remaining.get(row_w, 0.0) / t_w) if (rem_w > 0.0 and t_w > 0.0) else 0.0
        return cur_r, cur_w
    ratios: list[float] = []
    if t_r > 0.0:
        ratios.append(remaining.get(row_r, 0.0) / t_r)
    if t_w > 0.0:
        ratios.append(remaining.get(row_w, 0.0) / t_w)
    frac = _clamp01(max(ratios)) if ratios else 0.0
    if dir_mode == "mixed_coupled":
        return rem_r * frac, rem_w * frac
    if dir_mode == "write":
        return 0.0, rem_w * frac
    return rem_r * frac, 0.0


def _hbm_entry_done_since_bind(
    remaining: Mapping[str, float], entry: HbmAccountingEntry,
) -> float:
    """Return total scalar HBM bytes transferred since this entry was bound."""
    rem_at_bind = entry[2] + entry[3]
    cur_r, cur_w = _hbm_entry_remaining_bytes(remaining, entry)
    return max(0.0, rem_at_bind - (cur_r + cur_w))


def _hbm_entry_done_fraction(entry: HbmAccountingEntry, remaining: Mapping[str, float]) -> float:
    """Return total task-lifetime completion fraction ``[0.0, 1.0]`` for one bound HBM entry."""
    total_b = entry[1]
    if total_b <= 0.0:
        return 1.0
    cur_r, cur_w = _hbm_entry_remaining_bytes(remaining, entry)
    return _clamp01(1.0 - (cur_r + cur_w) / total_b)


def _hbm_delivery_fraction(
    task, live_remaining: Mapping[str, float] | None = None,
) -> float | None:
    """Return directional HBM progress fraction when the task's delivery reference is HBM-bound."""
    if not task.hbm_accounting:
        return None
    ref_r = next(iter(task.demand), None)
    if ref_r is None:
        return None
    rem_map = task.remaining if live_remaining is None else live_remaining
    for entry in task.hbm_accounting:
        if ref_r == entry[5] or ref_r == entry[7]:
            return _hbm_entry_done_fraction(entry, rem_map)
    return None


def _sync_task_hbm_accounting(
    engine, task, live_remaining: Mapping[str, float] | None = None,
) -> None:
    """Credit incremental scalar HBM bytes derived from live directional bus progress."""
    if not task.hbm_accounting:
        return
    rem_map = task.remaining if live_remaining is None else live_remaining
    credited = task.hbm_credited
    if credited is None:
        credited = task.hbm_credited = {}
    acct_serviced = engine._hbm_accounting_serviced
    serviced = engine.serviced
    task_svc = task.service
    for entry in task.hbm_accounting:
        base = entry[0]
        done_since_bind = _hbm_entry_done_since_bind(rem_map, entry)
        prev = credited.get(base, 0.0)
        delta = done_since_bind - prev
        if delta > 0.0:
            credited[base] = done_since_bind
            task_svc[base] = task_svc.get(base, 0.0) + delta
            serviced[base] = serviced.get(base, 0.0) + delta
            acct_serviced[base] = acct_serviced.get(base, 0.0) + delta


def _classify_hbm_split(
    base: str, name: str, work: Mapping[str, float], amount: float,
) -> tuple[str, float, float]:
    """Classify scalar HBM demand into ``(dir_mode, read_bytes, write_bytes)``."""
    suffix = "@" + base.split("@", 1)[1] if "@" in base else ""
    read_pipe = 0.0
    write_pipe = 0.0
    for key, value in work.items():
        if not value or key == base:
            continue
        if suffix and "@" in key and not key.endswith(suffix):
            continue
        prefix = key.split("@", 1)[0]
        if prefix in _READ_PIPE_PREFIXES:
            read_pipe += float(value)
        elif prefix in _WRITE_PIPE_PREFIXES:
            write_pipe += float(value)
    if read_pipe > 0.0 and write_pipe > 0.0:
        share = read_pipe / (read_pipe + write_pipe)
        return "mixed", amount * share, amount * (1.0 - share)
    if read_pipe > 0.0:
        return "read", amount, 0.0
    if write_pipe > 0.0:
        return "write", 0.0, amount
    tokens = {tok for tok in _NAME_SPLIT_RE.split(name.lower()) if tok}
    has_r = bool(tokens & _READ_NAME_TOKENS)
    has_w = bool(tokens & _WRITE_NAME_TOKENS)
    if has_r and not has_w:
        return "read", amount, 0.0
    if has_w and not has_r:
        return "write", 0.0, amount
    return "unknown", amount, 0.0


def _hbm_terms_for_split(
    bus, dir_mode: str, rem_r: float, rem_w: float,
) -> tuple[float, float]:
    """Compute directional ``(t_r, t_w)`` bus times for a classified split."""
    if dir_mode in ("mixed", "mixed_coupled"):
        t_r = t_w = 0.0
        if rem_r > 0.0:
            tr_r, tr_w = bus.terms_for("read", rem_r)
            t_r += tr_r.ms_per_progress
            t_w += tr_w.ms_per_progress
        if rem_w > 0.0:
            tw_r, tw_w = bus.terms_for("write", rem_w)
            t_r += tw_r.ms_per_progress
            t_w += tw_w.ms_per_progress
        return t_r, t_w
    if dir_mode == "read":
        tr_r, tr_w = bus.terms_for("read", rem_r)
        return tr_r.ms_per_progress, tr_w.ms_per_progress
    if dir_mode == "write":
        tw_r, tw_w = bus.terms_for("write", rem_w)
        return tw_r.ms_per_progress, tw_w.ms_per_progress
    tr_r, tr_w = bus.terms_for("read", rem_r)
    tw_r, tw_w = bus.terms_for("write", rem_r)
    return (
        max(tr_r.ms_per_progress, tw_r.ms_per_progress),
        max(tr_w.ms_per_progress, tw_w.ms_per_progress),
    )


def _hbm_bus_alias_terms(
    bus, base: str, name: str, work: Mapping[str, float], amount: float,
) -> tuple[float, float]:
    """Resolve directional ``(row_r, row_w)`` work via ``bus.terms_for``."""
    dir_mode, rem_r, rem_w = _classify_hbm_split(base, name, work, amount)
    return _hbm_terms_for_split(bus, dir_mode, rem_r, rem_w)


def _expand_hbm_bus_work(
    aliases: Mapping[str, object], name: str,
    work: Mapping[str, float], caps: Mapping[str, float] | None,
) -> tuple[Mapping[str, float], Mapping[str, float] | None, tuple[HbmAccountingEntry, ...]]:
    """Route shipping scalar HBM demands onto directional bus rows preserving work order."""
    if not any(base in work and work[base] for base in aliases):
        return work, caps, ()
    expanded = dict(work)
    expanded_caps = dict(caps) if caps else None
    accounting: list[HbmAccountingEntry] = []
    for base, bus in aliases.items():
        amount = expanded.get(base)
        if not amount or bus.row_r in expanded or bus.row_w in expanded:
            continue
        amt = float(amount)
        if amt <= 0.0 or not math.isfinite(amt):
            continue
        dir_mode, rem_r, rem_w = _classify_hbm_split(base, name, expanded, amt)
        if dir_mode == "mixed" and bus.alpha_rw > 0.0:
            dir_mode = "mixed_coupled"
        t_r, t_w = _hbm_terms_for_split(bus, dir_mode, rem_r, rem_w)
        if t_r <= 0.0 and t_w <= 0.0:
            continue
        expanded = _replace_keys_in_place(expanded, (base,), bus.row_r, t_r, bus.row_w, t_w)
        orig_cap: float | None = None
        if expanded_caps is not None and base in expanded_caps:
            cap_val = float(expanded_caps[base])
            if cap_val > 0.0 and math.isfinite(cap_val):
                orig_cap = cap_val
                if t_r > 0.0:
                    expanded_caps.setdefault(bus.row_r, t_r * cap_val / amt)
                if t_w > 0.0:
                    expanded_caps.setdefault(bus.row_w, t_w * cap_val / amt)
                expanded_caps.pop(base, None)
        accounting.append((base, amt, rem_r, rem_w, orig_cap, bus.row_r, t_r, bus.row_w, t_w, dir_mode))
    return expanded, expanded_caps, tuple(accounting)


def _discover_hbm_bus_aliases(resources: Mapping, aliases: dict) -> bool:
    """Register default symmetric bus aliases for pre-existing ``:r``/``:w`` rows."""
    discovered = False
    for name, res_r in resources.items():
        if not name.endswith(_ROW_READ_SUFFIX) or res_r.rate != _UNIT_ROW_RATE:
            continue
        base = name.removesuffix(_ROW_READ_SUFFIX)
        row_w = base + _ROW_WRITE_SUFFIX
        if base in aliases or base not in resources or row_w not in resources:
            continue
        base_res = resources[base]
        res_w = resources[row_w]
        if base_res.rate <= 0.0 or res_w.rate != _UNIT_ROW_RATE:
            continue
        aliases[base] = HbmBusSpec(
            0, float(base_res.rate), float(base_res.rate), 0.0,
            row_r=name, row_w=row_w,
        )
        discovered = True
    return discovered


def _hbm_coupled_entries(
    remaining: Mapping[str, float],
    hbm_accounting: tuple[HbmAccountingEntry, ...],
) -> tuple[HbmAccountingEntry, ...]:
    """Return accounting entries whose directional rows must progress in lockstep."""
    if not hbm_accounting:
        return ()
    return tuple(
        entry for entry in hbm_accounting
        if entry[9] != "mixed"
        and remaining.get(entry[5], 0.0) > 0.0
        and remaining.get(entry[7], 0.0) > 0.0
    )


def _is_pure_coupled_hbm_work(
    work: Mapping[str, float],
    hbm_accounting: tuple[HbmAccountingEntry, ...],
) -> bool:
    """True when all positive work belongs to one coupled 2-row HBM bus."""
    coupled = _hbm_coupled_entries(work, hbm_accounting)
    if len(coupled) != 1:
        return False
    row_r, row_w = coupled[0][5], coupled[0][7]
    return all(k == row_r or k == row_w for k, v in work.items() if v)


def _task_needs_hbm_bus_split(task) -> bool:
    """True when an uncoupled task mixes coupled 2-row HBM work with other resources."""
    if task.coupled or not task.hbm_accounting:
        return False
    coupled = _hbm_coupled_entries(task.remaining, task.hbm_accounting)
    return bool(coupled) and not _is_pure_coupled_hbm_work(task.remaining, task.hbm_accounting)


def _snapshot_hbm_entry(
    remaining: Mapping[str, float], entry: HbmAccountingEntry,
) -> HbmAccountingEntry | None:
    """Re-anchor one HBM accounting entry to live remaining bytes at a split boundary."""
    base, _, _, _, orig_cap, row_r, _, row_w, _, dir_mode = entry
    cur_r, cur_w = _hbm_entry_remaining_bytes(remaining, entry)
    rem_total = cur_r + cur_w
    if rem_total <= 0.0:
        return None
    new_mode = dir_mode
    if dir_mode in ("mixed", "mixed_coupled"):
        if cur_r <= 0.0:
            new_mode = "write"
        elif cur_w <= 0.0:
            new_mode = "read"
    return (
        base, rem_total, cur_r, cur_w, orig_cap,
        row_r, remaining.get(row_r, 0.0), row_w, remaining.get(row_w, 0.0), new_mode,
    )


def _hbm_bus_split_start_hook(_event) -> None:
    """Marker start hook keeping uncoupled 2-row HBM split tasks out of C11 adoption."""
    return None


def _hbm_split_child_names(name: str, remaining: Mapping[str, float], hbm_accounting: tuple[HbmAccountingEntry, ...]) -> tuple[str, ...]:
    """Return ordered child names for an uncoupled task splitting coupled 2-row HBM entries."""
    coupled = _hbm_coupled_entries(remaining, hbm_accounting)
    if not coupled:
        return ()
    names = [f"{name}#hbm{idx}" for idx in range(len(coupled))]
    coupled_rows = {row for entry in coupled for row in (entry[5], entry[7])}
    if any(r not in coupled_rows and amt > 0.0 for r, amt in remaining.items()):
        names.append(f"{name}#rest")
    return tuple(names)


def _notify_split_child_admit(engine, parent_task, child_ev, placement, *, generation: int = 0) -> None:
    """Emit ``on_admit`` for a split child using the parent's allocation generation."""
    wave_rec = engine._wave_recorder
    if wave_rec is None:
        return
    meta = parent_task.metadata
    owners = tuple(
        sorted({u for units in (placement or {}).values() for u in units})
    ) or (0,)
    gen = int(meta["generation"]) if (meta and "generation" in meta) else int(generation)
    wave_rec.on_admit(
        child_ev.sequence, engine.now, owners=owners, generation=gen, blockers=(),
    )


def _notify_split_child_observer(engine, parent_task, child_ev, placement, *, generation: int = 0) -> None:
    """Register an internal split child with the live observer preserving occurrence."""
    wave_rec = engine._wave_recorder
    if wave_rec is None:
        return
    meta = parent_task.metadata
    occ = meta.get("occurrence") if meta else None
    op_occ = meta.get("op_occ") if meta else None
    wave_rec.on_submit(
        child_ev.sequence, child_ev.name,
        aliases=(parent_task.owner,) if parent_task.owner else (),
        predecessors=(),
        occurrence=occ,
        op_occ=op_occ,
        parent_task_id=parent_task.event.sequence,
    )
    _notify_split_child_admit(engine, parent_task, child_ev, placement, generation=generation)


def _remaining_task_deliveries(task) -> tuple[tuple[str, float], ...]:
    """Return coalesced outstanding ``(ctr_id, remaining_amount)`` pairs for ``task``."""
    if not task.deliveries:
        return ()
    totals: dict[str, float] = {}
    for cid, amt in task.deliveries:
        if amt > 0.0:
            totals[cid] = totals.get(cid, 0.0) + amt
    acc = task.delivered_acc or {}
    return tuple(
        (cid, rem_d) for cid, tot in totals.items()
        if (rem_d := tot - acc.get(cid, 0.0)) > 0.0
    )


def _split_uncoupled_hbm_bus_task(engine, task, *, admit_gen: int = 0) -> None:
    """Isolate coupled 2-row HBM bus work into coupled child(ren) without coupling peer resources."""
    from .engine_types import Event, _EMPTY_FROZENSET, _EMPTY_METADATA, _Task
    key = task.event.sequence
    if task.hbm_accounting:
        _sync_task_hbm_accounting(engine, task)
    if task.deliveries:
        engine._update_task_deliveries(task)
    coupled_entries = _hbm_coupled_entries(task.remaining, task.hbm_accounting)
    if not coupled_entries:
        return
    ref_r = next(iter(task.demand), None)
    rem_delivs = _remaining_task_deliveries(task)
    parts: list[tuple[str, dict[str, float], bool, tuple[HbmAccountingEntry, ...]]] = []
    coupled_rows: set[str] = set()
    for idx, entry in enumerate(coupled_entries):
        row_r, row_w = entry[5], entry[7]
        coupled_rows.add(row_r)
        coupled_rows.add(row_w)
        c_rem = {row_r: task.remaining[row_r], row_w: task.remaining[row_w]}
        snap = _snapshot_hbm_entry(task.remaining, entry)
        parts.append((f"{task.event.name}#hbm{idx}", c_rem, True, (snap,) if snap is not None else ()))
    rest_rem = {r: amt for r, amt in task.remaining.items() if r not in coupled_rows and amt > 0.0}
    if rest_rem:
        rest_acct = tuple(
            snap for entry in task.hbm_accounting
            if entry not in coupled_entries
            and (snap := _snapshot_hbm_entry(task.remaining, entry)) is not None
        )
        parts.append((f"{task.event.name}#rest", rest_rem, False, rest_acct))
    if key in engine._active:
        engine._active.discard(key)
        if task.latency > 0.0:
            engine._latency_active.discard(key)
        if task.placement:
            engine._placed_active -= 1
    if task.hbm_accounting:
        task.hbm_accounting = ()
        task.hbm_credited = None
        if engine._hbm_acct_live > 0:
            engine._hbm_acct_live -= 1
    if task.deliveries:
        task.deliveries = ()
        task.delivered_acc = None
        if engine._delivery_live > 0:
            engine._delivery_live -= 1
    task.remaining.clear()
    remaining_parts = [len(parts)]
    orig_task = task

    def _part_done(part_ev, part_task_ref, orig=orig_task, rem_box=remaining_parts):
        for rn, amt in part_task_ref.service.items():
            orig.service[rn] = orig.service.get(rn, 0.0) + amt
        if part_task_ref.first_service is not None:
            if orig.first_service is None or part_task_ref.first_service < orig.first_service:
                orig.first_service = part_task_ref.first_service
        rem_box[0] -= 1
        if rem_box[0] == 0:
            orig.remaining.clear()
            orig.split_children = ()
            anchor = orig.first_service if orig.latency_from_first_service else orig.start
            if orig.latency > 0.0 and anchor is not None and part_ev.time < anchor + orig.latency:
                okey = orig.event.sequence
                engine._active.add(okey)
                engine._latency_active.add(okey)
                if orig.placement:
                    engine._placed_active += 1
            else:
                engine._complete(orig, moment=part_ev.time)

    pre_children = task.split_children if len(task.split_children) == len(parts) else ()
    base_fifo = task.fifo_order[0] if task.fifo_order is not None else key
    child_meta = dict(task.metadata) if task.metadata else _EMPTY_METADATA
    deliv_owner = next((i for i, (_, rem_d, _, _) in enumerate(parts) if ref_r in rem_d), 0)
    children: list[Event] = []
    for idx, (c_name, c_rem, c_coupled, c_acct) in enumerate(parts):
        c_caps = {r: task.caps[r] for r in c_rem if r in task.caps}
        c_place = {r: task.placement[r] for r in c_rem if r in task.placement}
        c_widened = tuple(r for r in task.cohort_widened if r in c_rem)
        c_members = {r: members for r, members in task.cohort_members if r in c_rem}
        c_delivs = rem_delivs if (rem_delivs and idx == deliv_owner) else ()
        if pre_children:
            c_ev = pre_children[idx]
            c_ev.started = task.start
        else:
            engine._sequence += 1
            c_ev = Event(engine._sequence, c_name, started=task.start, runtime_uid=engine.uid)
        c_prep = engine._prepare(
            c_ev.name, c_rem, latency=0.0, caps=c_caps or None,
            placement=c_place or None, coupled=c_coupled, priority=task.priority,
            cohort_size=task.cohort_size,
            cohort_widened=c_widened or None,
            cohort_members=c_members or None,
            latency_from_first_service=task.latency_from_first_service,
        )
        c_task = _Task(
            c_ev, dict(c_rem), c_prep.demand, 0, task.release,
            0.0, c_prep.caps, (), (), task.owner, child_meta,
            c_prep.placement, (), task.priority, c_coupled, None, None,
            task.start, {}, False, None, 0, 0, _EMPTY_FROZENSET, None, (),
            task.cohort_size, c_prep, None, False,
            c_prep.cohort_widened, c_prep.cohort_members, c_delivs,
            {} if c_delivs else None, True,
            fifo_order=(base_fifo, idx),
            latency_from_first_service=task.latency_from_first_service,
            first_service=task.first_service,
            hbm_accounting=c_acct,
            hbm_credited={} if c_acct else None,
        )
        c_task.callback = lambda ev, ct=c_task, fn=_part_done: fn(ev, ct)
        engine._tasks[c_ev.sequence] = c_task
        children.append(c_ev)
        if pre_children:
            _notify_split_child_admit(engine, task, c_ev, c_prep.placement, generation=admit_gen)
        else:
            _notify_split_child_observer(engine, task, c_ev, c_prep.placement, generation=admit_gen)
        if c_acct:
            engine._hbm_acct_live += 1
        if c_delivs:
            engine._delivery_live += 1
        if engine.task_uses_fifo(c_task):
            engine._fifo_live += 1
        if engine.task_blocks_native(c_task):
            engine._inline_blocked += 1
        engine._active.add(c_ev.sequence)
        if c_coupled:
            engine._coupled_active += 1
        if c_prep.placement:
            engine._placed_active += 1
    task.split_children = tuple(children)


def _promote_or_queue_hbm_split(engine, seq: int, task, to_split: list) -> None:
    """Promote a pure 2-row HBM task to coupled or schedule a multi-resource split."""
    if task.coupled or not task.hbm_accounting:
        return
    if _is_pure_coupled_hbm_work(task.remaining, task.hbm_accounting):
        task.coupled = True
        if engine is not None and seq in engine._active:
            engine._coupled_active += 1
    elif _task_needs_hbm_bus_split(task):
        if engine is not None and seq in engine._active:
            to_split.append(task)
        elif task.start_callback is None:
            had_block = engine.task_blocks_native(task) if engine is not None else True
            task.start_callback = _hbm_bus_split_start_hook
            if engine is not None and not had_block:
                engine._inline_blocked += 1


def _retrofit_hbm_bus_tasks(
    tasks: Mapping, native_slots: Mapping, base: str, bus, *,
    engine=None, replacing: bool = False, include_native: bool = False,
) -> bool:
    """Bind already-submitted shipping HBM tasks onto ``bus`` directional rows."""
    deferred = False
    to_split: list = []
    for seq, task in list(tasks.items()):
        rem = task.remaining.get(base, 0.0)
        acct_entry = next((item for item in task.hbm_accounting if item[0] == base), None)
        if rem > 0.0:
            if not replacing and (bus.row_r in task.remaining or bus.row_w in task.remaining):
                continue
            if not include_native and seq in native_slots:
                deferred = True
                continue
            had_acct = bool(task.hbm_accounting)
            total_b = float(task.demand.get(base, rem))
            dir_mode, rem_r, rem_w = _classify_hbm_split(base, task.event.name, task.demand, rem)
            if dir_mode == "mixed" and bus.alpha_rw > 0.0:
                dir_mode = "mixed_coupled"
            t_r, t_w = _hbm_terms_for_split(bus, dir_mode, rem_r, rem_w)
            task.demand = _replace_keys_in_place(task.demand, (base,), bus.row_r, t_r, bus.row_w, t_w)
            task.remaining = _replace_keys_in_place(task.remaining, (base,), bus.row_r, t_r, bus.row_w, t_w)
            orig_cap: float | None = None
            if base in task.caps:
                cap_val = float(task.caps[base])
                new_caps = {k: v for k, v in task.caps.items() if k not in (base, bus.row_r, bus.row_w)}
                if cap_val > 0.0 and math.isfinite(cap_val):
                    orig_cap = cap_val
                    if t_r > 0.0:
                        new_caps[bus.row_r] = t_r * cap_val / rem
                    if t_w > 0.0:
                        new_caps[bus.row_w] = t_w * cap_val / rem
                task.caps = new_caps
            other_acct = tuple(item for item in task.hbm_accounting if item[0] != base)
            task.hbm_accounting = (
                *other_acct,
                (base, total_b, rem_r, rem_w, orig_cap, bus.row_r, t_r, bus.row_w, t_w, dir_mode),
            )
            if task.hbm_credited is None:
                task.hbm_credited = {base: 0.0}
            else:
                task.hbm_credited[base] = 0.0
            if engine is not None and not had_acct:
                engine._hbm_acct_live += 1
            task.prepared = None
            _promote_or_queue_hbm_split(engine, seq, task, to_split)
        elif replacing and acct_entry is not None:
            if not include_native and seq in native_slots:
                deferred = True
                continue
            _, total_b, rem_r_bind, rem_w_bind, orig_cap, old_r, t_r_bind, old_w, t_w_bind, dir_mode = acct_entry
            check_mode = (
                ("mixed_coupled" if bus.alpha_rw > 0.0 else "mixed")
                if dir_mode in ("mixed", "mixed_coupled") else dir_mode
            )
            if ((old_r, old_w) == (bus.row_r, bus.row_w)
                    and check_mode == dir_mode
                    and _hbm_terms_for_split(bus, check_mode, rem_r_bind, rem_w_bind) == (t_r_bind, t_w_bind)):
                continue
            if engine is not None:
                _sync_task_hbm_accounting(engine, task)
            cur_r, cur_w = _hbm_entry_remaining_bytes(task.remaining, acct_entry)
            rem_now = cur_r + cur_w
            if rem_now <= 0.0:
                continue
            new_mode = dir_mode
            if dir_mode in ("mixed", "mixed_coupled"):
                if cur_r <= 0.0:
                    new_mode = "write"
                elif cur_w <= 0.0:
                    new_mode = "read"
                else:
                    new_mode = "mixed_coupled" if bus.alpha_rw > 0.0 else "mixed"
            t_r, t_w = _hbm_terms_for_split(bus, new_mode, cur_r, cur_w)
            task.demand = _replace_keys_in_place(
                task.demand, (base, old_r, old_w), bus.row_r, t_r, bus.row_w, t_w,
            )
            task.remaining = _replace_keys_in_place(
                task.remaining, (base, old_r, old_w), bus.row_r, t_r, bus.row_w, t_w,
            )
            if orig_cap is not None and rem_now > 0.0:
                new_caps = {
                    k: v for k, v in task.caps.items()
                    if k not in (base, old_r, old_w, bus.row_r, bus.row_w)
                }
                if t_r > 0.0:
                    new_caps[bus.row_r] = t_r * orig_cap / rem_now
                if t_w > 0.0:
                    new_caps[bus.row_w] = t_w * orig_cap / rem_now
                task.caps = new_caps
            other_acct = tuple(item for item in task.hbm_accounting if item[0] != base)
            task.hbm_accounting = (
                *other_acct,
                (base, total_b, cur_r, cur_w, orig_cap, bus.row_r, t_r, bus.row_w, t_w, new_mode),
            )
            if task.hbm_credited is None:
                task.hbm_credited = {base: 0.0}
            else:
                task.hbm_credited[base] = 0.0
            task.prepared = None
            if (task.coupled and task.is_split_child and new_mode == "mixed"
                    and set(task.remaining).issubset({bus.row_r, bus.row_w})):
                task.coupled = False
                if engine is not None and seq in engine._active:
                    engine._coupled_active -= 1
            else:
                _promote_or_queue_hbm_split(engine, seq, task, to_split)
    for split_task in to_split:
        _split_uncoupled_hbm_bus_task(engine, split_task)
    return deferred


def _partition_hbm_accounting_for_split(
    task, cohort: int,
) -> tuple[HbmAccountingEntry, ...]:
    """Split a live parent task's outstanding HBM accounting evenly across ``cohort`` children."""
    if not task.hbm_accounting or cohort <= 1:
        return ()
    child_entries: list[HbmAccountingEntry] = []
    for entry in task.hbm_accounting:
        base, _, _, _, orig_cap, row_r, _, row_w, _, dir_mode = entry
        cur_r, cur_w = _hbm_entry_remaining_bytes(task.remaining, entry)
        rem_total = cur_r + cur_w
        if rem_total <= 0.0:
            continue
        new_mode = dir_mode
        if dir_mode in ("mixed", "mixed_coupled"):
            if cur_r <= 0.0:
                new_mode = "write"
            elif cur_w <= 0.0:
                new_mode = "read"
        child_r = cur_r / cohort
        child_w = cur_w / cohort
        child_tot = rem_total / cohort
        child_cap = (orig_cap / cohort) if orig_cap is not None else None
        child_tr = task.remaining.get(row_r, 0.0) / cohort
        child_tw = task.remaining.get(row_w, 0.0) / cohort
        child_entries.append((
            base, child_tot, child_r, child_w, child_cap,
            row_r, child_tr, row_w, child_tw, new_mode,
        ))
    return tuple(child_entries)


def _refresh_prepared_for_hbm_bus(engine, prepared, name: str):
    """Re-prepare cached shipping tasks whenever HBM directional bindings changed."""
    aliases = engine._hbm_bus_aliases
    if not aliases or prepared.hbm_bus_epoch == engine._hbm_bus_epoch:
        return prepared
    acct_map = {entry[0]: (entry[1], entry[4]) for entry in prepared.hbm_accounting}
    if not prepared.demand and not acct_map:
        return prepared
    if not any(base in prepared.demand or base in acct_map for base in aliases):
        return prepared
    drop = {
        row for base, bus in aliases.items()
        if base in prepared.demand or base in acct_map
        for row in (bus.row_r, bus.row_w)
    }
    raw_demand: dict[str, float] = {}
    restored: set[str] = set()
    row_to_base = {
        row: base for base, bus in aliases.items()
        if base in acct_map
        for row in (bus.row_r, bus.row_w)
    }
    for k, v in prepared.demand.items():
        if k in row_to_base:
            b = row_to_base[k]
            if b not in restored:
                raw_demand[b] = acct_map[b][0]
                restored.add(b)
            continue
        if k not in drop:
            raw_demand[k] = v
    for b, (amt, _) in acct_map.items():
        if b not in restored:
            raw_demand[b] = amt
    raw_caps = {k: v for k, v in prepared.caps.items() if k not in drop} if prepared.caps else {}
    for base, (_, cap) in acct_map.items():
        if cap is not None:
            raw_caps[base] = cap
    return engine._prepare(
        name, raw_demand,
        latency=prepared.latency,
        caps=raw_caps or None,
        acquire=prepared.acquire,
        free=prepared.free,
        placement=prepared.placement or None,
        coupled=prepared.coupled,
        priority=prepared.priority,
        cohort_size=prepared.cohort_size,
        cohort_widened=prepared.cohort_widened or None,
        cohort_members=dict(prepared.cohort_members) if prepared.cohort_members else None,
        latency_from_first_service=prepared.latency_from_first_service,
    )


def _execute_asymmetric_cohort_splits(engine, to_split: list[int]) -> bool:
    """Decompose active cohort tasks into per-member tasks preserving live accounting."""
    from .engine_types import Event, _EMPTY_FROZENSET, _EMPTY_METADATA, _Task
    tasks = engine._tasks
    active = engine._active
    for key in to_split:
        task = tasks[key]
        cohort = task.cohort_size
        members_dict = dict(task.cohort_members)
        if task.hbm_accounting:
            _sync_task_hbm_accounting(engine, task)
        if task.deliveries:
            engine._update_task_deliveries(task)
        child_hbm_acct = _partition_hbm_accounting_for_split(task, cohort)
        child_delivs = tuple(
            (cid, rem_d / cohort) for cid, rem_d in _remaining_task_deliveries(task)
        )
        if task.hbm_accounting:
            task.hbm_accounting = ()
            task.hbm_credited = None
            if engine._hbm_acct_live > 0:
                engine._hbm_acct_live -= 1
        if task.deliveries:
            task.deliveries = ()
            task.delivered_acc = None
            if engine._delivery_live > 0:
                engine._delivery_live -= 1
        active.discard(key)
        if task.latency > 0.0:
            engine._latency_active.discard(key)
        if task.coupled:
            engine._coupled_active -= 1
        if task.placement:
            engine._placed_active -= 1
        remaining_members = [cohort]
        orig_task = task

        def _member_done(member_ev, m_task_ref, orig=orig_task, rem_box=remaining_members):
            for rn, amt in m_task_ref.service.items():
                orig.service[rn] = orig.service.get(rn, 0.0) + amt
            rem_box[0] -= 1
            if rem_box[0] == 0:
                orig.remaining.clear()
                orig.split_children = ()
                engine._complete(orig, moment=member_ev.time)

        base_fifo = task.fifo_order[0] if task.fifo_order is not None else key
        child_meta = dict(task.metadata) if task.metadata else _EMPTY_METADATA
        children: list[Event] = []
        for idx in range(cohort):
            m_rem = {r: amt / cohort for r, amt in task.remaining.items()}
            m_caps = {r: cap / cohort for r, cap in task.caps.items()}
            m_place = {
                r: (members_dict[r][idx] if r in members_dict else units)
                for r, units in task.placement.items()
            }
            engine._sequence += 1
            m_ev = Event(engine._sequence, f"{task.event.name}#member{idx}",
                         started=task.start, runtime_uid=engine.uid)
            m_prep = engine._prepare(
                m_ev.name, m_rem, latency=task.latency, caps=m_caps,
                placement=m_place, coupled=task.coupled, priority=task.priority,
                latency_from_first_service=task.latency_from_first_service,
            )
            m_acct = child_hbm_acct or m_prep.hbm_accounting
            m_task = _Task(
                m_ev, dict(m_rem), m_prep.demand, 0, task.release,
                task.latency, m_prep.caps, (), (), task.owner, child_meta,
                m_prep.placement, (), task.priority, task.coupled, None, None,
                task.start, {}, False, None, 0, 0, _EMPTY_FROZENSET, None, (),
                1, m_prep, None, False, (), (), child_delivs,
                {} if child_delivs else None, True,
                fifo_order=(base_fifo, idx),
                latency_from_first_service=task.latency_from_first_service,
                first_service=task.first_service,
                hbm_accounting=m_acct,
                hbm_credited={} if m_acct else None,
            )
            m_task.callback = lambda ev, mt=m_task, fn=_member_done: fn(ev, mt)
            tasks[m_ev.sequence] = m_task
            children.append(m_ev)
            _notify_split_child_observer(engine, task, m_ev, m_prep.placement)
            if m_acct:
                engine._hbm_acct_live += 1
            if child_delivs:
                engine._delivery_live += 1
            if engine.task_uses_fifo(m_task):
                engine._fifo_live += 1
            if engine.task_blocks_native(m_task):
                engine._inline_blocked += 1
            active.add(m_ev.sequence)
            if m_task.latency > 0.0:
                engine._latency_active.add(m_ev.sequence)
            if m_task.coupled:
                engine._coupled_active += 1
            if m_task.placement:
                engine._placed_active += 1
        task.split_children = tuple(children)
    return True
