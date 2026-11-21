"""Directional HBM bus aliasing, work expansion, and live task retrofit helpers.

[Role] Map scalar shipping HBM work onto bound directional ``:r``/``:w`` bus
  rows while keeping the scalar HBM byte total as completion accounting only
  so concurrent read/write shipping stages achieve full-duplex throughput.
[Invariants] Directional bus rows govern bus contention exclusively once bound;
  scalar HBM byte totals are conserved in ``task.service`` and ``Engine.serviced``
  across Python and C11 execution without double counting or serialization.
[Timeouts & Locks] Pure bounded in-memory dict/task rewrites; no locks or I/O.
[Side-Effect Policy] Mutates only live task work/accounting fields on retrofit.
"""
from __future__ import annotations

import math
import re
from typing import Mapping

from .memory_physics import HbmBusSpec

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
) -> tuple[Mapping[str, float], Mapping[str, float] | None, tuple[tuple[str, float, float | None], ...]]:
    """Route shipping scalar HBM demands onto directional bus rows without scalar serialization."""
    if not any(base in work and work[base] for base in aliases):
        return work, caps, ()
    expanded = dict(work)
    expanded_caps = dict(caps) if caps else None
    accounting: list[tuple[str, float, float | None]] = []
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
        if t_r > 0.0:
            expanded[bus.row_r] = t_r
        if t_w > 0.0:
            expanded[bus.row_w] = t_w
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
        expanded.pop(base, None)
        accounting.append((base, amt, orig_cap))
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
    replacing: bool = False, include_native: bool = False,
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
            dem = task.demand.get(base, rem)
            t_r, t_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, rem)
            d_r, d_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, dem)
            new_demand = {k: v for k, v in task.demand.items() if k not in (base, bus.row_r, bus.row_w)}
            if d_r > 0.0:
                new_demand[bus.row_r] = d_r
            if d_w > 0.0:
                new_demand[bus.row_w] = d_w
            task.demand = new_demand
            task.remaining.pop(base, None)
            if t_r > 0.0:
                task.remaining[bus.row_r] = t_r
            else:
                task.remaining.pop(bus.row_r, None)
            if t_w > 0.0:
                task.remaining[bus.row_w] = t_w
            else:
                task.remaining.pop(bus.row_w, None)
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
            task.hbm_accounting = (*other_acct, (base, rem, orig_cap))
            task.prepared = None
        elif replacing and acct_entry is not None:
            if not include_native and seq in native_slots:
                deferred = True
                continue
            _, orig_rem, orig_cap = acct_entry
            ratios = [
                task.remaining.get(row, 0.0) / task.demand[row]
                for row in (bus.row_r, bus.row_w)
                if task.demand.get(row, 0.0) > 0.0
            ]
            frac = max(ratios) if ratios else 0.0
            if frac <= 0.0:
                continue
            rem_now = orig_rem * frac
            t_r, t_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, rem_now)
            d_r, d_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, orig_rem)
            new_demand = {k: v for k, v in task.demand.items() if k not in (base, bus.row_r, bus.row_w)}
            if d_r > 0.0:
                new_demand[bus.row_r] = d_r
            if d_w > 0.0:
                new_demand[bus.row_w] = d_w
            task.demand = new_demand
            if t_r > 0.0:
                task.remaining[bus.row_r] = t_r
            else:
                task.remaining.pop(bus.row_r, None)
            if t_w > 0.0:
                task.remaining[bus.row_w] = t_w
            else:
                task.remaining.pop(bus.row_w, None)
            if orig_cap is not None and rem_now > 0.0:
                new_caps = {k: v for k, v in task.caps.items() if k not in (base, bus.row_r, bus.row_w)}
                if t_r > 0.0:
                    new_caps[bus.row_r] = t_r * orig_cap / rem_now
                if t_w > 0.0:
                    new_caps[bus.row_w] = t_w * orig_cap / rem_now
                task.caps = new_caps
            task.prepared = None
    return deferred


def _refresh_prepared_for_hbm_bus(engine, prepared, name: str):
    """Re-prepare cached shipping tasks whenever HBM directional bindings changed."""
    aliases = engine._hbm_bus_aliases
    acct_map = {base: (amt, cap) for base, amt, cap in prepared.hbm_accounting}
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
    raw_demand = {k: v for k, v in prepared.demand.items() if k not in drop}
    raw_caps = {k: v for k, v in prepared.caps.items() if k not in drop} if prepared.caps else {}
    for base, (amt, cap) in acct_map.items():
        raw_demand[base] = amt
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
