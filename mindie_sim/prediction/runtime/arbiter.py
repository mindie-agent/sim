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
    """Non-preemptive owner: actual start, then priority, then sequence."""
    return min(caps, key=lambda key: (
        tasks[key].start is None,
        tasks[key].start if tasks[key].start is not None else math.inf,
        tasks[key].priority, key))


def _fast_eligible_task(resources, task) -> bool:
    """True when single-tier fast arbitration matches the per-unit solver."""
    rem = task.remaining
    if not rem:
        return True
    pl = task.placement
    if task.coupled:
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


_ZERO_COUNT = 0
_MULTI_RESOURCE_CUTOFF = 1


def _has_overlapping_subunit_coupled(resources, tasks, active) -> bool:
    """True when a coupled task shares a strict sub-unit owner with any peer.

    C11 solves every active coupled cohort jointly.  An unpinned coupled peer
    must therefore not take the closed-form two-task path while an uncoupled
    peer owns only part of one of its resources: the shared owner unit meters
    both flows, and only the per-unit residual solver preserves that capacity.
    An unplaced participant spans every declared unit, so any strict sub-unit
    placement on the resource overlaps it.
    """
    for key in active:
        task = tasks[key]
        rem = task.remaining
        pl = task.placement
        if not (task.coupled and len(rem) > _MULTI_RESOURCE_CUTOFF):
            continue
        for rname, amt in rem.items():
            if amt <= 0.0:
                continue
            res = resources[rname]
            units = pl.get(rname) if pl else None
            unit_set = set(units) if units is not None else None
            own_subunit = units is not None and len(units) < res.units
            for peer_key in active:
                if peer_key == key:
                    continue
                peer = tasks[peer_key]
                if peer.remaining.get(rname, 0.0) <= 0.0:
                    continue
                peer_units = (peer.placement.get(rname)
                              if peer.placement else None)
                peer_subunit = (peer_units is not None
                                and len(peer_units) < res.units)
                if not (own_subunit or peer_subunit):
                    continue
                if unit_set is None or peer_units is None:
                    return True
                if not unit_set.isdisjoint(peer_units):
                    return True
    return False


def component_rates(resources, tasks, active, *, coupled_solver=None,
                    coupled_active: int | None = None,
                    placed_active: int | None = None) -> dict[str, dict[int, float]]:
    """Return {physical_resource: {task_sequence: native_work/ms}}."""
    if not active:
        return {}
    if coupled_active == _ZERO_COUNT:
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
            if (multi_prio or (placed_active != _ZERO_COUNT and any(r.policy != "fair" for r in resources.values())))
            else False
        )
    else:
        need_tier = _needs_unit_tier_solver(resources, tasks, active)
    if need_tier:
        if coupled_solver is not None and any(
                tasks[key].coupled and tasks[key].remaining for key in active):
            return _tiered_coupled(resources, tasks, active, coupled_solver)
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
                         task.priority, key)
                rates = coupled_solver(resources, {proxy: task})
                return {name: {key: components[proxy]}
                        for name, components in rates.items()
                        if proxy in components}
        elif ((placed_active is None or placed_active != _ZERO_COUNT)
              and _has_overlapping_subunit_coupled(resources, tasks, active)):
            return _tiered_coupled(resources, tasks, active, coupled_solver)
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
                      t0.priority, k0)
                p1 = (t1.start is None,
                      t1.start if t1.start is not None else math.inf,
                      t1.priority, k1)
                reverse = {p0: k0, p1: k1}
                rates = coupled_solver(resources, {p0: t0, p1: t1})
                return {name: {reverse[proxy]: value
                               for proxy, value in components.items()}
                        for name, components in rates.items()}
        elif n_active > 8 and any(
                tasks[key].coupled and len(tasks[key].remaining) > 1
                for key in active):
            return _tiered_coupled(resources, tasks, active, coupled_solver)
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
                                 active_task.priority, active_key)
                        view[proxy] = active_task
                        reverse[proxy] = active_key
                    rates = coupled_solver(resources, view)
                    return {name: {reverse[proxy]: value
                                   for proxy, value in components.items()}
                            for name, components in rates.items()}
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
                if u0[-1] < u1[0] or u1[-1] < u0[0]:
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
