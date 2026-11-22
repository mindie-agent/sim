"""Pure component-local service arbitration for the causal Engine.

[Role] Compute per-resource flow rates from immutable live-task views.
[Invariants] Capped weighted max-min for fair resources, non-preemptive FIFO
  ownership for fifo resources, exact finite unit partitions for pinned
  placement, and strict cross-tier ``math.fsum`` conservation when priority
  classes or per-unit pinned FIFO/coupled constraints apply.
[Timeouts & Locks] Finite progressive filling; no iteration outside caps.
[Side-Effect Policy] Pure arithmetic over supplied records.
"""
from __future__ import annotations

import math
from typing import Mapping

from ..ownership import unit_rates
from .arbiter_tiered import (
    _lockstep_coupled, _max_min_rates, _priority_tiers, _progressive_unit_fill,
    _residual_unit_rates, _tier_flows, _tiered_coupled, _tiered_simple,
    _unit_span, fair_rates,
)


def fifo_winner(caps: Mapping[int, float], tasks) -> int:
    """Non-preemptive owner: actual start, then priority, then preserved FIFO order."""
    return min(caps, key=lambda key: (
        tasks[key].start is None,
        tasks[key].start if tasks[key].start is not None else math.inf,
        tasks[key].priority,
        tasks[key].fifo_order if getattr(tasks[key], "fifo_order", None) is not None else (key, 0)))


def _fast_eligible_task(resources, task) -> bool:
    """True when single-tier fast arbitration matches the per-unit solver."""
    if getattr(task, "latency_from_first_service", False):
        return False
    rem = task.remaining
    if not rem:
        return True
    pl = task.placement
    if task.coupled:
        if getattr(task, "cohort_size", 1) > 1:
            return False
        first_u = first_tot = 0
        for rname in rem:
            res = resources[rname]
            if res.policy == "fifo":
                return False
            if pl and len(rem) > 1:
                units = pl.get(rname)
                u_len = res.units if units is None else len(units)
                if first_tot == 0:
                    first_u, first_tot = u_len, res.units
                elif u_len * first_tot != first_u * res.units:
                    return False
        return True
    if pl:
        for rname in rem:
            units = pl.get(rname)
            if units is not None:
                res = resources[rname]
                if res.policy == "fifo" and len(units) < res.units:
                    return False
    return True


def _needs_unit_tier_solver(resources, tasks, active) -> bool:
    """Detect multi-priority fair/coupled cohorts or sub-unit FIFO/coupled pins."""
    first_prio = None
    multi_prio = False
    any_coupled = False
    any_fair = False
    for key in active:
        task = tasks[key]
        rem = task.remaining
        if not rem:
            continue
        prio = task.priority
        if first_prio is None:
            first_prio = prio
        elif prio != first_prio:
            multi_prio = True
        pl = task.placement
        if task.coupled:
            any_coupled = True
            if pl and len(rem) > 1:
                first_u = first_tot = 0
                for rname, amt in rem.items():
                    if amt <= 0.0:
                        continue
                    res = resources[rname]
                    units = pl.get(rname)
                    u_len = res.units if units is None else len(units)
                    if first_tot == 0:
                        first_u, first_tot = u_len, res.units
                    elif u_len * first_tot != first_u * res.units:
                        return True
        for rname, amt in rem.items():
            if amt <= 0.0:
                continue
            res = resources[rname]
            if res.policy == "fair":
                any_fair = True
            elif pl:
                units = pl.get(rname)
                if units is not None and len(units) < res.units:
                    return True
    return multi_prio and (any_coupled or any_fair)


def _partition_coupled_components(resources, tasks, active):
    """Split active tasks into coupled-connected and disconnected-uncoupled sets."""
    coupled_keys = {
        k for k in active if tasks[k].remaining and tasks[k].coupled
    }
    if not coupled_keys or len(coupled_keys) == len(active):
        return coupled_keys, set()
    uncoupled_keys = {
        k for k in active if tasks[k].remaining and not tasks[k].coupled
    }
    if not uncoupled_keys:
        return coupled_keys, set()
    changed = True
    while changed and uncoupled_keys:
        changed = False
        hot_fifo: set[str] = set()
        hot_units: dict[str, set[int]] = {}
        for k in coupled_keys:
            t = tasks[k]
            pl = t.placement
            for rname, amt in t.remaining.items():
                if amt <= 0.0:
                    continue
                res = resources[rname]
                if res.policy == "fifo":
                    hot_fifo.add(rname)
                else:
                    us = pl.get(rname) if pl else None
                    hot_units.setdefault(rname, set()).update(
                        range(res.units) if us is None else us
                    )
        promoted: list[int] = []
        for k in uncoupled_keys:
            t = tasks[k]
            pl = t.placement
            for rname, amt in t.remaining.items():
                if amt <= 0.0:
                    continue
                if rname in hot_fifo:
                    promoted.append(k)
                    break
                hu = hot_units.get(rname)
                if hu is not None:
                    us = pl.get(rname) if pl else None
                    if us is None or not hu.isdisjoint(us):
                        promoted.append(k)
                        break
        if promoted:
            coupled_keys.update(promoted)
            uncoupled_keys.difference_update(promoted)
            changed = True
    return coupled_keys, uncoupled_keys


def _independent_component_rates(resources, tasks, active, placed_active):
    contenders: dict[str, dict[int, float]] = {}
    for key in sorted(active):
        task = tasks[key]
        caps = task.caps
        for name in task.remaining:
            row = contenders.get(name)
            val = caps.get(name, math.inf) if caps else math.inf
            if row is None:
                contenders[name] = {key: val}
            else:
                row[key] = val
    probe_placed = placed_active != 0
    result = {}
    for name, caps in contenders.items():
        resource = resources[name]
        if resource.policy == "fifo":
            winner = fifo_winner(caps, tasks)
            result[name] = {winner: min(resource.rate, caps[winner])}
            continue
        n_c = len(caps)
        if n_c == 1:
            key, cap = next(iter(caps.items()))
            t_one = tasks[key]
            prep = getattr(t_one, "prepared", None)
            if prep is not None and prep.solo_rates is not None:
                result[name] = {key: prep.solo_rates[name]}
                continue
            if probe_placed:
                pl = t_one.placement
                if pl and name in pl:
                    lim = resource.rate / resource.units * len(pl[name])
                    result[name] = {key: cap if cap < lim else lim}
                    continue
            r_rate = resource.rate
            result[name] = {key: cap if cap < r_rate else r_rate}
            continue
        if not probe_placed or resource.units == 1:
            result[name] = _max_min_rates(resource.rate, caps)
            continue
        if n_c == 2:
            (k0, c0), (k1, c1) = caps.items()
            p0, p1 = tasks[k0].placement, tasks[k1].placement
            u0 = p0.get(name) if p0 else None
            u1 = p1.get(name) if p1 else None
            if u0 is None and u1 is None:
                result[name] = _max_min_rates(resource.rate, caps)
                continue
            if u0 is not None and u1 is not None:
                if u0 == u1:
                    result[name] = _max_min_rates(
                        resource.rate / resource.units * len(u0), caps)
                    continue
                if set(u0).isdisjoint(u1):
                    ur = resource.rate / resource.units
                    l0, l1 = ur * len(u0), ur * len(u1)
                    result[name] = {k0: c0 if c0 < l0 else l0,
                                    k1: c1 if c1 < l1 else l1}
                    continue
        pinned = {}
        for key in caps:
            pl = tasks[key].placement
            if pl:
                u = pl.get(name)
                if u is not None:
                    pinned[key] = u
        result[name] = (unit_rates(resource.rate, resource.units, caps, pinned)
                        if pinned else _max_min_rates(resource.rate, caps))
    return result


def component_rates(resources, tasks, active, *, coupled_solver=None,
                    coupled_active: int | None = None,
                    placed_active: int | None = None,
                    allow_native: bool = True) -> dict[str, dict[int, float]]:
    """Return {physical_resource: {task_sequence: native_work/ms}}."""
    if not active:
        return {}
    if coupled_active != 0 and coupled_solver is not None and len(active) >= 2:
        c_keys, u_keys = _partition_coupled_components(resources, tasks, active)
        if u_keys:
            out: dict[str, dict[int, float]] = {}
            if c_keys:
                r_c = component_rates(
                    resources, tasks, c_keys,
                    coupled_solver=coupled_solver,
                    coupled_active=len(c_keys),
                    placed_active=placed_active,
                    allow_native=allow_native,
                )
                for rname, sub in r_c.items():
                    out[rname] = dict(sub)
            r_u = component_rates(
                resources, tasks, u_keys,
                coupled_solver=None,
                coupled_active=0,
                placed_active=placed_active,
                allow_native=allow_native,
            )
            for rname, sub in r_u.items():
                out.setdefault(rname, {}).update(sub)
            return out
    if coupled_active == 0:
        first_prio = None
        multi_prio = False
        for key in active:
            t_chk = tasks[key]
            if t_chk.remaining:
                p_chk = t_chk.priority
                if first_prio is None:
                    first_prio = p_chk
                elif p_chk != first_prio:
                    multi_prio = True
                    break
        need_tier = (
            _needs_unit_tier_solver(resources, tasks, active)
            if (multi_prio or (placed_active != 0 and any(r.policy != "fair" for r in resources.values())))
            else False
        )
    else:
        need_tier = _needs_unit_tier_solver(resources, tasks, active)
    if need_tier:
        if coupled_solver is not None and any(
                tasks[key].coupled and tasks[key].remaining for key in active):
            return _tiered_coupled(
                resources, tasks, active, coupled_solver,
                allow_native=allow_native)
        return _tiered_simple(resources, tasks, active)
    if coupled_active != 0 and coupled_solver is not None:
        n_active = len(active)
        if n_active == 1:
            key = next(iter(active))
            task = tasks[key]
            if task.coupled and task.remaining:
                single_fn = getattr(coupled_solver, "_single", None)
                if single_fn is not None and _fast_eligible_task(resources, task):
                    return single_fn(resources, key, task)
                proxy = (task.start is None,
                         task.start if task.start is not None else math.inf,
                         task.priority,
                         task.fifo_order if getattr(task, "fifo_order", None) is not None else (key, 0))
                rates = coupled_solver(resources, {proxy: task})
                return {name: {key: components[proxy]}
                        for name, components in rates.items()
                        if proxy in components}
        elif n_active == 2:
            it = iter(active)
            k0, k1 = next(it), next(it)
            t0, t1 = tasks[k0], tasks[k1]
            if (t0.coupled and t0.remaining) or (t1.coupled and t1.remaining):
                pair_fn = getattr(coupled_solver, "_pair", None)
                if (pair_fn is not None and t0.priority == t1.priority
                        and _fast_eligible_task(resources, t0)
                        and _fast_eligible_task(resources, t1)):
                    fast = pair_fn(resources, k0, t0, k1, t1)
                    if fast is not None:
                        return fast
                p0 = (t0.start is None,
                      t0.start if t0.start is not None else math.inf,
                      t0.priority,
                      t0.fifo_order if getattr(t0, "fifo_order", None) is not None else (k0, 0))
                p1 = (t1.start is None,
                      t1.start if t1.start is not None else math.inf,
                      t1.priority,
                      t1.fifo_order if getattr(t1, "fifo_order", None) is not None else (k1, 0))
                reverse = {p0: k0, p1: k1}
                rates = coupled_solver(resources, {p0: t0, p1: t1})
                return {name: {reverse[proxy]: value
                               for proxy, value in components.items()}
                        for name, components in rates.items()}
        else:
            for key in active:
                task = tasks[key]
                if task.coupled and task.remaining:
                    view = {}
                    reverse = {}
                    for active_key in sorted(active):
                        active_task = tasks[active_key]
                        proxy = (active_task.start is None,
                                 active_task.start if active_task.start is not None else math.inf,
                                 active_task.priority,
                                 active_task.fifo_order if getattr(active_task, "fifo_order", None) is not None else (active_key, 0))
                        view[proxy] = active_task
                        reverse[proxy] = active_key
                    rates = coupled_solver(resources, view)
                    return {name: {reverse[proxy]: value
                                   for proxy, value in components.items()}
                            for name, components in rates.items()}
    return _independent_component_rates(resources, tasks, active, placed_active)


def service_step(rates: Mapping[str, Mapping[int, float]], tasks, dt: float):
    """Conserve and apply one integration step without inventing work."""
    used = []
    for name, consumers in rates.items():
        for key, rate in consumers.items():
            task = tasks[key]
            remaining = task.remaining[name]
            amount = min(remaining, dt * rate)
            residual = remaining - amount
            if residual <= 4 * math.ulp(remaining):
                amount, residual = remaining, 0.0
            used.append((name, key, amount, residual))
    return used


__all__ = [
    "component_rates", "fair_rates", "fifo_winner", "service_step",
    "_fast_eligible_task", "_lockstep_coupled", "_max_min_rates",
    "_needs_unit_tier_solver", "_priority_tiers", "_progressive_unit_fill",
    "_residual_unit_rates", "_tier_flows", "_tiered_coupled", "_tiered_simple",
    "_unit_span",
]
