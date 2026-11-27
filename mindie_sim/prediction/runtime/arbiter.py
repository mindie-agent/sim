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
from typing import AbstractSet, Mapping

from ..ownership import _unit_mask, unit_rates
from .arbiter_fast_coupled import fast_single_tier_coupled
from .arbiter_tiered import (
    _FLOW_UNPLACED_IDX, _SAT_ULPS, _expand_simple_tier, _fifo_rank,
    _lockstep_coupled, _max_min_rates, _member_placements, _priority_tiers,
    _progressive_unit_fill, _residual_unit_rates, _tier_flows, _tiered_coupled,
    _tiered_simple, _unit_span, fair_rates,
)


class _ProjectedTask:
    """Read-only projection of an uncoupled task onto one coupled resource component."""

    __slots__ = (
        "remaining", "caps", "placement", "priority", "start", "fifo_order",
        "service", "coupled", "cohort_size", "cohort_widened",
        "cohort_members", "latency_from_first_service", "prepared",
    )

    def __init__(self, task, comp_res: AbstractSet[str]):
        self.remaining = {
            r: amt for r, amt in task.remaining.items()
            if amt > 0.0 and r in comp_res
        }
        caps = task.caps
        self.caps = {r: caps[r] for r in self.remaining if r in caps} if caps else {}
        pl = task.placement
        self.placement = {r: pl[r] for r in self.remaining if r in pl} if pl else {}
        self.priority = task.priority
        self.start = task.start
        self.fifo_order = getattr(task, "fifo_order", None)
        self.service = task.service
        self.coupled = False
        self.cohort_size = getattr(task, "cohort_size", 1)
        self.cohort_widened = getattr(task, "cohort_widened", ())
        self.cohort_members = getattr(task, "cohort_members", ())
        self.latency_from_first_service = getattr(task, "latency_from_first_service", False)
        self.prepared = None


def fifo_winner(caps: Mapping[int, float], tasks, resource_name: str | None = None) -> int:
    """Non-preemptive owner: active FIFO holder, then priority, then arrival/FIFO order."""
    if resource_name is None:
        return min(caps, key=lambda key: (
            tasks[key].start is None,
            tasks[key].start if tasks[key].start is not None else math.inf,
            tasks[key].priority,
            tasks[key].fifo_order if getattr(tasks[key], "fifo_order", None) is not None else (key, 0)))
    return min(caps, key=lambda key: (
        tasks[key].service.get(resource_name, 0.0) <= 0.0,
        tasks[key].priority,
        tasks[key].start is None,
        tasks[key].start if tasks[key].start is not None else math.inf,
        tasks[key].fifo_order if getattr(tasks[key], "fifo_order", None) is not None else (key, 0)))


def _fast_eligible_task(resources, task) -> bool:
    """True when single-tier fast arbitration matches the per-unit solver."""
    if getattr(task, "latency_from_first_service", False):
        return False
    rem = task.remaining
    if not rem:
        return True
    if getattr(task, "cohort_size", 1) > 1:
        return False
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
    any_uncoupled_cohort = False
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
        elif getattr(task, "cohort_size", 1) > 1:
            any_uncoupled_cohort = True
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
    if any_coupled and any_uncoupled_cohort:
        return True
    return multi_prio and (any_coupled or any_fair)


def _expand_cohort_contenders(tasks, caps: Mapping[int, float], name: str):
    """Expand uncoupled ``cohort_size > 1`` tasks into literal member contenders."""
    exp_caps: dict[tuple[int, int], float] = {}
    exp_pinned: dict[tuple[int, int], tuple[int, ...]] = {}
    for key, cap in caps.items():
        task = tasks[key]
        cohort = max(1, int(getattr(task, "cohort_size", 1) or 1))
        if cohort == 1:
            exp_caps[(key, 0)] = cap
            pl = task.placement
            if pl and name in pl:
                exp_pinned[(key, 0)] = pl[name]
            continue
        member_cap = cap / cohort
        for idx, m_units in enumerate(_member_placements(task, name, cohort)):
            m_key = (key, idx)
            exp_caps[m_key] = member_cap
            if m_units is not None:
                exp_pinned[m_key] = m_units
    return exp_caps, exp_pinned


def _independent_resource_rates(resources, tasks, contenders, probe_placed: bool):
    result = {}
    for name, caps in contenders.items():
        resource = resources[name]
        if resource.policy == "fifo":
            winner = fifo_winner(caps, tasks, name)
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
        if any(getattr(tasks[key], "cohort_size", 1) > 1 for key in caps):
            exp_caps, exp_pinned = _expand_cohort_contenders(tasks, caps, name)
            exp_rates = (
                unit_rates(resource.rate, resource.units, exp_caps, exp_pinned)
                if (exp_pinned and resource.units > 1)
                else _max_min_rates(resource.rate, exp_caps)
            )
            agg: dict[int, float] = {}
            for (orig_key, _idx), rate in exp_rates.items():
                agg[orig_key] = agg.get(orig_key, 0.0) + rate
            result[name] = agg
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


def _independent_component_rates(resources, tasks, active, placed_active):
    contenders: dict[str, dict[int, float]] = {}
    for key in sorted(active):
        task = tasks[key]
        caps = task.caps
        for name, amt in task.remaining.items():
            if amt <= 0.0:
                continue
            row = contenders.get(name)
            val = caps.get(name, math.inf) if caps else math.inf
            if row is None:
                contenders[name] = {key: val}
            else:
                row[key] = val
    return _independent_resource_rates(resources, tasks, contenders, placed_active != 0)


def _coupled_resource_components(tasks, active) -> list[set[str]]:
    """Group resources touched by coupled tasks using coupled-task edges only."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for key in active:
        task = tasks[key]
        if not task.coupled or not task.remaining:
            continue
        first_res: str | None = None
        for rname, amt in task.remaining.items():
            if amt <= 0.0:
                continue
            if rname not in parent:
                parent[rname] = rname
            if first_res is None:
                first_res = rname
            else:
                ra, rb = find(first_res), find(rname)
                if ra != rb:
                    parent[rb] = ra
    if not parent:
        return []
    groups: dict[str, set[str]] = {}
    for rname in parent:
        groups.setdefault(find(rname), set()).add(rname)
    return list(groups.values())


def _distribute_unplaced_residual_owners(
    tier_shares: Mapping,
    consumed_u: list[float],
    has_cohort: bool,
    owner_map: dict[int, dict[int, float]],
) -> None:
    """Recover exact per-unit rates for an unplaced-only tier absorbing non-uniform residual."""
    rem_flows = {k: float(v) for k, v in tier_shares.items() if v > 0.0}
    rem_u = {u: float(v) for u, v in enumerate(consumed_u) if v > 0.0}
    max_iters = len(rem_flows) + len(rem_u) + 2
    for _ in range(max_iters):
        if not rem_flows or not rem_u:
            break
        n_f = len(rem_flows)
        n_u = len(rem_u)
        min_f = min(rem_flows.values())
        min_u = min(rem_u.values())
        u_step = min_u * (float(n_u) / float(n_f))
        step = min_f if min_f <= u_step else u_step
        if step <= 0.0:
            break
        per_u = step / float(n_u)
        u_dec = per_u * float(n_f)
        for c_key in rem_flows:
            orig_key = c_key[0] if has_cohort else c_key
            u_acc = owner_map.setdefault(orig_key, {})
            for u in rem_u:
                u_acc[u] = u_acc.get(u, 0.0) + per_u
        drop_f = [
            c_key for c_key, val in rem_flows.items()
            if val - step <= _SAT_ULPS * math.ulp(max(1.0, float(tier_shares[c_key])))
        ]
        for c_key in list(rem_flows):
            rem_flows[c_key] -= step
        for c_key in drop_f:
            rem_flows.pop(c_key, None)
        drop_u = [
            u for u, val in rem_u.items()
            if val - u_dec <= _SAT_ULPS * math.ulp(max(1.0, consumed_u[u]))
        ]
        for u in list(rem_u):
            rem_u[u] -= u_dec
        for u in drop_u:
            rem_u.pop(u, None)


def _tiered_coupled_with_owners(
    resources, tasks, active, owner_out: dict[str, dict[int, tuple[tuple[int, float], ...]]],
) -> dict[str, dict[int, float]]:
    """Priority-tiered per-unit coupled solver recording exact per-owner rate weights."""
    orig_capacity = {
        name: [float(resource.rate) / max(1, int(resource.units))] * max(1, int(resource.units))
        for name, resource in resources.items()
    }
    cumulative_terms = {name: [[] for _ in units] for name, units in orig_capacity.items()}
    unit_remaining = {name: list(units) for name, units in orig_capacity.items()}
    result: dict[str, dict[int, float]] = {}
    owner_acc: dict[str, dict[int, dict[int, float]]] = {}
    fifo_winners = {
        name: min(
            (key for key in active if tasks[key].remaining.get(name, 0.0) > 0.0),
            key=lambda key, rname=name: _fifo_rank(tasks, key, rname),
        )
        for name, resource in resources.items()
        if resource.policy == "fifo"
        and any(tasks[key].remaining.get(name, 0.0) > 0.0 for key in active)
    }
    for _priority, tier_keys in _priority_tiers(active, tasks).items():
        tier = sorted(key for key in tier_keys if tasks[key].remaining)
        if not tier:
            continue
        flows = _tier_flows(resources, tasks, tier, unit_remaining, fifo_winners)
        if not flows:
            continue
        committed_tier: list[dict[tuple[str, int], float]] = []
        alphas = _progressive_unit_fill(
            flows, unit_remaining, orig_capacity=orig_capacity,
            cumulative_terms=cumulative_terms, committed_out=committed_tier,
        )
        for index, alpha in enumerate(alphas):
            flow = flows[index]
            key = flow[0]
            for (rname, unit), u_rate in committed_tier[index].items():
                if u_rate > 0.0:
                    u_map = owner_acc.setdefault(rname, {}).setdefault(key, {})
                    u_map[unit] = u_map.get(unit, 0.0) + u_rate
            if not flow[1] and len(flow) > _FLOW_UNPLACED_IDX and flow[_FLOW_UNPLACED_IDX]:
                rate_val = math.fsum(committed_tier[index].values())
                if rate_val > 0.0:
                    name = flow[4][0][0]
                    target = result.setdefault(name, {})
                    target[key] = target.get(key, 0.0) + rate_val
                continue
            if alpha <= 0.0:
                continue
            for name, demand in flow[4]:
                target = result.setdefault(name, {})
                target[key] = target.get(key, 0.0) + alpha * demand
    for name, units in cumulative_terms.items():
        for unit, terms in enumerate(units):
            if math.fsum(terms) > orig_capacity[name][unit]:
                raise AssertionError(f"unit_service_overallocated:{name}:{unit}")
    for rname, by_key in owner_acc.items():
        dest = owner_out.setdefault(rname, {})
        for key, u_map in by_key.items():
            pairs = tuple(sorted((u, v) for u, v in u_map.items() if v > 0.0))
            if pairs:
                dest[key] = pairs
    return {name: rows for name, rows in result.items() if rows}


def _tiered_fair_resource_rates(
    resource, tasks, name: str, caps: Mapping[int, float],
    *, owner_out: dict[str, dict[int, tuple[tuple[int, float], ...]]] | None = None,
) -> dict[int, float]:
    """Strict priority tiers on one fair resource matching ``unit_rates`` within each tier."""
    count = max(1, int(resource.units))
    orig_caps = [float(resource.rate) / count] * count
    unit_remaining = list(orig_caps)
    cumulative_terms: list[list[float]] = [[] for _ in range(count)]
    shares: dict[int, float] = {}
    owner_map: dict[int, dict[int, float]] | None = {} if owner_out is not None else None
    for _priority, tier_keys in _priority_tiers(caps, tasks).items():
        tier = {key: caps[key] for key in tier_keys if caps[key] > 0.0}
        remaining = math.fsum(unit_remaining)
        if not tier or remaining <= 0.0:
            continue
        available = tuple(unit for unit in range(count) if unit_remaining[unit] > 0.0)
        if not available:
            break
        scoped, placements, explicit, has_cohort = _expand_simple_tier(
            tasks, tier, name, count, available, unit_remaining,
        )
        if not placements:
            continue
        max_u = max(unit_remaining)
        uniform = max_u - min(unit_remaining) <= _SAT_ULPS * math.ulp(max_u)
        if uniform and not explicit:
            tier_shares = fair_rates(remaining, scoped)
            tier_sat = any(rate < scoped[c_key] for c_key, rate in tier_shares.items())
            for c_key, rate in tier_shares.items():
                if rate <= 0.0:
                    continue
                orig_key = c_key[0] if has_cohort else c_key
                shares[orig_key] = shares.get(orig_key, 0.0) + rate
                u_tuple = placements.get(c_key, available)
                per_u = rate / len(u_tuple)
                for u in u_tuple:
                    cumulative_terms[u].append(per_u)
                if owner_map is not None:
                    u_acc = owner_map.setdefault(orig_key, {})
                    for u in u_tuple:
                        u_acc[u] = u_acc.get(u, 0.0) + per_u
            for u in available:
                unit_remaining[u] = (
                    0.0 if tier_sat
                    else max(0.0, orig_caps[u] - math.fsum(cumulative_terms[u]))
                )
        else:
            # When explicit placements appear in a tier, freezing every flow touching
            # a saturated unit keeps single-tier rates identical to ``ownership.unit_rates``
            # so adding a lower-priority waiter never perturbs higher-priority completions;
            # an unplaced-only lower tier still absorbs non-uniform residual capacity.
            before_rem = list(unit_remaining) if (owner_map is not None and not explicit) else None
            freeze_set = set(scoped) if explicit else explicit
            tier_shares = _residual_unit_rates(
                unit_remaining, scoped, placements, freeze_set,
                orig_capacity=orig_caps, cumulative_terms=cumulative_terms,
            )
            for c_key, rate in tier_shares.items():
                if rate <= 0.0:
                    continue
                orig_key = c_key[0] if has_cohort else c_key
                shares[orig_key] = shares.get(orig_key, 0.0) + rate
                if owner_map is not None and explicit:
                    u_tuple = placements.get(c_key, available)
                    per_u = rate / len(u_tuple)
                    u_acc = owner_map.setdefault(orig_key, {})
                    for u in u_tuple:
                        u_acc[u] = u_acc.get(u, 0.0) + per_u
            if owner_map is not None and before_rem is not None:
                consumed_u = [max(0.0, before_rem[u] - unit_remaining[u]) for u in range(count)]
                _distribute_unplaced_residual_owners(tier_shares, consumed_u, has_cohort, owner_map)
    if owner_out is not None and owner_map:
        dest = owner_out.setdefault(name, {})
        for key, u_acc in owner_map.items():
            pairs = tuple(sorted((u, v) for u, v in u_acc.items() if v > 0.0))
            if pairs:
                dest[key] = pairs
    return {key: rate for key, rate in shares.items() if rate > 0.0}


def _uncoupled_contenders_rates(
    resources, tasks, contenders: dict[str, dict[int, float]], probe_placed: bool,
    *, owner_rates_out: dict[str, dict[int, tuple[tuple[int, float], ...]]] | None = None,
) -> dict[str, dict[int, float]]:
    """Select single-tier vs priority-tiered arbitration independently per uncoupled resource."""
    if not contenders:
        return {}
    tiered_fair: set[str] = set()
    tiered_fifo: set[str] = set()
    for name, caps in contenders.items():
        resource = resources[name]
        if resource.policy == "fair":
            it = iter(caps)
            base_prio = tasks[next(it)].priority
            if any(tasks[k].priority != base_prio for k in it):
                tiered_fair.add(name)
        elif probe_placed:
            for k in caps:
                pl = tasks[k].placement
                units = pl.get(name) if pl else None
                if units is not None and len(units) < resource.units:
                    tiered_fifo.add(name)
                    break
    if not tiered_fair and not tiered_fifo:
        return _independent_resource_rates(resources, tasks, contenders, probe_placed)
    fast_contenders = {
        name: caps for name, caps in contenders.items()
        if name not in tiered_fair and name not in tiered_fifo
    }
    out = (
        _independent_resource_rates(resources, tasks, fast_contenders, probe_placed)
        if fast_contenders else {}
    )
    for name in tiered_fair:
        rows = _tiered_fair_resource_rates(
            resources[name], tasks, name, contenders[name], owner_out=owner_rates_out,
        )
        if rows:
            out[name] = rows
    if tiered_fifo:
        fifo_keys = {key for name in tiered_fifo for key in contenders[name]}
        fifo_out = _tiered_simple(resources, tasks, fifo_keys)
        for name in tiered_fifo:
            if name in fifo_out:
                out[name] = fifo_out[name]
    return out


def _uncoupled_component_rates(
    resources, tasks, active, placed_active,
    *, owner_rates_out: dict[str, dict[int, tuple[tuple[int, float], ...]]] | None = None,
) -> dict[str, dict[int, float]]:
    """Select single-tier vs priority-tiered arbitration independently per resource."""
    probe_placed = placed_active != 0
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
    has_nonfair_placed = probe_placed and any(r.policy != "fair" for r in resources.values())
    if not multi_prio and not has_nonfair_placed:
        return _independent_component_rates(resources, tasks, active, placed_active)
    contenders: dict[str, dict[int, float]] = {}
    for key in sorted(active):
        task = tasks[key]
        caps = task.caps
        for name, amt in task.remaining.items():
            if amt <= 0.0:
                continue
            row = contenders.get(name)
            val = caps.get(name, math.inf) if caps else math.inf
            if row is None:
                contenders[name] = {key: val}
            else:
                row[key] = val
    return _uncoupled_contenders_rates(
        resources, tasks, contenders, probe_placed, owner_rates_out=owner_rates_out,
    )


def _solve_coupled_component(
    resources, tasks, active, coupled_solver, placed_active, allow_native: bool,
    owner_rates_out: dict[str, dict[int, tuple[tuple[int, float], ...]]] | None,
) -> dict[str, dict[int, float]]:
    """Solve one coupled-resource connected component cleanly."""
    need_tier = _needs_unit_tier_solver(resources, tasks, active)
    if need_tier:
        if coupled_solver is not None:
            if owner_rates_out is not None:
                fast = fast_single_tier_coupled(
                    resources, tasks, active, allow_native=allow_native,
                )
                if fast is not None:
                    return fast
                return _tiered_coupled_with_owners(resources, tasks, active, owner_rates_out)
            return _tiered_coupled(
                resources, tasks, active, coupled_solver, allow_native=allow_native,
            )
        return _tiered_simple(resources, tasks, active)
    if coupled_solver is not None:
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


def _task_res_unit_mask(task, rname: str, full_mask: int) -> int:
    """Physical unit bitmask occupied by ``task`` on ``rname``."""
    cohort = max(1, int(getattr(task, "cohort_size", 1) or 1))
    if not task.coupled and cohort > 1:
        mask = 0
        for m_units in _member_placements(task, rname, cohort):
            if m_units is None:
                return full_mask
            mask |= _unit_mask(m_units)
        return mask
    pl = task.placement
    units = pl.get(rname) if pl else None
    return full_mask if units is None else _unit_mask(units)


def _coupled_uncoupled_links(
    resources, tasks, active, coupled_res_all: AbstractSet[str], placed_active,
) -> dict[str, set[int]]:
    """Return per-resource uncoupled task keys sharing physical units with coupled flows."""
    coupled_on_res: dict[str, list[int]] = {}
    uncoupled_on_res: dict[str, list[int]] = {}
    for key in active:
        task = tasks[key]
        rem = task.remaining
        if not rem:
            continue
        target = coupled_on_res if task.coupled else uncoupled_on_res
        for rname, amt in rem.items():
            if amt > 0.0 and rname in coupled_res_all:
                target.setdefault(rname, []).append(key)
    promoted: dict[str, set[int]] = {}
    probe_placed = placed_active != 0
    for rname, u_keys in uncoupled_on_res.items():
        res = resources[rname]
        if res.policy == "fifo" or res.units <= 1 or not probe_placed:
            promoted[rname] = set(u_keys)
            continue
        full_mask = (1 << res.units) - 1
        hot_mask = 0
        for c_key in coupled_on_res.get(rname, ()):
            hot_mask |= _task_res_unit_mask(tasks[c_key], rname, full_mask)
            if hot_mask == full_mask:
                break
        if hot_mask == full_mask:
            promoted[rname] = set(u_keys)
            continue
        if not hot_mask:
            continue
        remaining_u = [
            (k, _task_res_unit_mask(tasks[k], rname, full_mask))
            for k in u_keys
        ]
        hit_keys: set[int] = set()
        while remaining_u:
            next_rem = []
            grew = False
            for k, mask in remaining_u:
                if hot_mask & mask:
                    hit_keys.add(k)
                    if mask & ~hot_mask:
                        hot_mask |= mask
                        grew = True
                else:
                    next_rem.append((k, mask))
            if not grew:
                break
            remaining_u = next_rem
        if hit_keys:
            promoted[rname] = hit_keys
    return promoted


def component_rates(
    resources, tasks, active, *, coupled_solver=None,
    coupled_active: int | None = None,
    placed_active: int | None = None,
    allow_native: bool = True,
    owner_rates_out: dict[str, dict[int, tuple[tuple[int, float], ...]]] | None = None,
) -> dict[str, dict[int, float]]:
    """Return {physical_resource: {task_sequence: native_work/ms}}."""
    if not active:
        return {}
    if coupled_active == 0:
        return _uncoupled_component_rates(
            resources, tasks, active, placed_active, owner_rates_out=owner_rates_out,
        )
    coupled_comps = _coupled_resource_components(tasks, active)
    if not coupled_comps:
        return _uncoupled_component_rates(
            resources, tasks, active, placed_active, owner_rates_out=owner_rates_out,
        )
    coupled_res_all = (
        coupled_comps[0] if len(coupled_comps) == 1
        else {r for comp in coupled_comps for r in comp}
    )
    coupled_u_links = _coupled_uncoupled_links(
        resources, tasks, active, coupled_res_all, placed_active,
    )
    uncoupled_contenders: dict[str, dict[int, float]] = {}
    for key in sorted(active):
        task = tasks[key]
        if task.coupled:
            continue
        caps = task.caps
        for name, amt in task.remaining.items():
            if amt <= 0.0:
                continue
            if name in coupled_res_all and key in coupled_u_links.get(name, ()):
                continue
            row = uncoupled_contenders.get(name)
            val = caps.get(name, math.inf) if caps else math.inf
            if row is None:
                uncoupled_contenders[name] = {key: val}
            else:
                row[key] = val
    out = (
        _uncoupled_contenders_rates(
            resources, tasks, uncoupled_contenders, placed_active != 0,
            owner_rates_out=owner_rates_out,
        )
        if uncoupled_contenders else {}
    )
    if len(coupled_comps) == 1 and not uncoupled_contenders:
        # Single coupled component covering every active resource: solve directly.
        comp_res = coupled_comps[0]
        comp_keys = {
            k for k in active
            if any(amt > 0.0 and r in comp_res for r, amt in tasks[k].remaining.items())
        }
        sub_res = {r: resources[r] for r in comp_res}
        return _solve_coupled_component(
            sub_res, tasks, comp_keys, coupled_solver, placed_active, allow_native, owner_rates_out,
        )
    for comp_res in coupled_comps:
        comp_keys: set[int] = set()
        sub_tasks: dict[int, object] = {}
        for key in active:
            task = tasks[key]
            rem = task.remaining
            if not rem:
                continue
            if task.coupled:
                if any(amt > 0.0 and r in comp_res for r, amt in rem.items()):
                    comp_keys.add(key)
                    sub_tasks[key] = task
                continue
            keep_res: set[str] = set()
            outside = False
            for rname, amt in rem.items():
                if amt <= 0.0:
                    continue
                if rname in comp_res and key in coupled_u_links.get(rname, ()):
                    keep_res.add(rname)
                else:
                    outside = True
            if keep_res:
                comp_keys.add(key)
                sub_tasks[key] = _ProjectedTask(task, keep_res) if outside else task
        if not comp_keys:
            continue
        sub_res = {r: resources[r] for r in comp_res}
        sub_rates = _solve_coupled_component(
            sub_res, sub_tasks, comp_keys, coupled_solver,
            placed_active, allow_native, owner_rates_out,
        )
        for rname, rows in sub_rates.items():
            out.setdefault(rname, {}).update(rows)
    return out


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
