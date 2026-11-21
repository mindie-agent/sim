"""Directional HBM bus aliasing, work expansion, and live task retrofit helpers.

[Role] Map scalar shipping HBM work onto bound directional ``:r``/``:w`` bus
  rows while keeping live incremental scalar HBM accounting and delivery
  progress tied to directional bus progress across Python and C11 execution.
[Invariants] Directional bus rows govern bus contention exclusively once bound;
  scalar HBM work order is preserved in-place for delivery reference semantics;
  live scalar HBM service and rebinds compute remaining fractions strictly from
  each entry's own binding snapshot with exact total byte conservation.
[Timeouts & Locks] Pure bounded in-memory dict/task rewrites; no locks or I/O.
[Side-Effect Policy] Mutates only live task work/accounting fields and Engine
  service counters on step sync or retrofit.
"""
from __future__ import annotations

import math
import re
from typing import Mapping

from .memory_physics import HbmBusSpec

HbmAccountingEntry = tuple[str, float, float, float | None, str, float, str, float]

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


def _hbm_rem_fraction(remaining: Mapping[str, float], entry: HbmAccountingEntry) -> float:
    """Return outstanding directional fraction in ``[0.0, 1.0]`` against the entry's bind snapshot."""
    _, _, _, _, row_r, t_r, row_w, t_w = entry
    ratios: list[float] = []
    if t_r > 0.0:
        ratios.append(remaining.get(row_r, 0.0) / t_r)
    if t_w > 0.0:
        ratios.append(remaining.get(row_w, 0.0) / t_w)
    if not ratios:
        return 0.0
    return min(1.0, max(0.0, max(ratios)))


def _hbm_entry_done_fraction(entry: HbmAccountingEntry, remaining: Mapping[str, float]) -> float:
    """Return total task-lifetime completion fraction ``[0.0, 1.0]`` for one bound HBM entry."""
    _, total_b, rem_at_bind, _, _, _, _, _ = entry
    if total_b <= 0.0:
        return 1.0
    rem_now = rem_at_bind * _hbm_rem_fraction(remaining, entry)
    return min(1.0, max(0.0, 1.0 - rem_now / total_b))


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
        if ref_r == entry[4] or ref_r == entry[6]:
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
        rem_at_bind = entry[2]
        done_since_bind = rem_at_bind * (1.0 - _hbm_rem_fraction(rem_map, entry))
        prev = credited.get(base, 0.0)
        delta = done_since_bind - prev
        if delta > 0.0:
            credited[base] = done_since_bind
            task_svc[base] = task_svc.get(base, 0.0) + delta
            serviced[base] = serviced.get(base, 0.0) + delta
            acct_serviced[base] = acct_serviced.get(base, 0.0) + delta


def _hbm_bus_alias_terms(
    bus, base: str, name: str, work: Mapping[str, float], amount: float,
) -> tuple[float, float]:
    """Resolve directional ``(row_r, row_w)`` work via ``bus.terms_for``."""
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
        tr_r, tr_w = bus.terms_for("read", amount * share)
        tw_r, tw_w = bus.terms_for("write", amount * (1.0 - share))
        return (
            tr_r.ms_per_progress + tw_r.ms_per_progress,
            tr_w.ms_per_progress + tw_w.ms_per_progress,
        )
    if read_pipe > 0.0:
        tr_r, tr_w = bus.terms_for("read", amount)
        return tr_r.ms_per_progress, tr_w.ms_per_progress
    if write_pipe > 0.0:
        tw_r, tw_w = bus.terms_for("write", amount)
        return tw_r.ms_per_progress, tw_w.ms_per_progress
    tokens = {tok for tok in _NAME_SPLIT_RE.split(name.lower()) if tok}
    has_r = bool(tokens & _READ_NAME_TOKENS)
    has_w = bool(tokens & _WRITE_NAME_TOKENS)
    if has_r and not has_w:
        tr_r, tr_w = bus.terms_for("read", amount)
        return tr_r.ms_per_progress, tr_w.ms_per_progress
    if has_w and not has_r:
        tw_r, tw_w = bus.terms_for("write", amount)
        return tw_r.ms_per_progress, tw_w.ms_per_progress
    tr_r, tr_w = bus.terms_for("read", amount)
    tw_r, tw_w = bus.terms_for("write", amount)
    return (
        max(tr_r.ms_per_progress, tw_r.ms_per_progress),
        max(tr_w.ms_per_progress, tw_w.ms_per_progress),
    )


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
        t_r, t_w = _hbm_bus_alias_terms(bus, base, name, expanded, amt)
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
        accounting.append((base, amt, amt, orig_cap, bus.row_r, t_r, bus.row_w, t_w))
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


def _retrofit_hbm_bus_tasks(
    tasks: Mapping, native_slots: Mapping, base: str, bus, *,
    engine=None, replacing: bool = False, include_native: bool = False,
) -> bool:
    """Bind already-submitted shipping HBM tasks onto ``bus`` directional rows."""
    deferred = False
    for seq, task in tasks.items():
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
            t_r, t_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, rem)
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
                (base, total_b, rem, orig_cap, bus.row_r, t_r, bus.row_w, t_w),
            )
            if task.hbm_credited is None:
                task.hbm_credited = {base: 0.0}
            else:
                task.hbm_credited[base] = 0.0
            if engine is not None and not had_acct:
                engine._hbm_acct_live += 1
            task.prepared = None
        elif replacing and acct_entry is not None:
            if not include_native and seq in native_slots:
                deferred = True
                continue
            if engine is not None:
                _sync_task_hbm_accounting(engine, task)
            _, total_b, rem_at_bind, orig_cap, old_r, _, old_w, _ = acct_entry
            rem_frac = _hbm_rem_fraction(task.remaining, acct_entry)
            if rem_frac <= 0.0:
                continue
            rem_now = rem_at_bind * rem_frac
            t_r, t_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, rem_now)
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
                (base, total_b, rem_now, orig_cap, bus.row_r, t_r, bus.row_w, t_w),
            )
            if task.hbm_credited is None:
                task.hbm_credited = {base: 0.0}
            else:
                task.hbm_credited[base] = 0.0
            task.prepared = None
    return deferred


def _refresh_prepared_for_hbm_bus(engine, prepared, name: str):
    """Re-prepare cached shipping tasks whenever HBM directional bindings changed."""
    aliases = engine._hbm_bus_aliases
    acct_map = {entry[0]: (entry[1], entry[3]) for entry in prepared.hbm_accounting}
    if (not aliases or prepared.hbm_bus_epoch == engine._hbm_bus_epoch
            or (not prepared.demand and not acct_map)):
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
