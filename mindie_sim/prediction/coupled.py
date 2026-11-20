"""Capacity sharing for resource-coupled steady-state progress.

[Role] Assign service to finite fluid pipelines on the existing Engine.
[Invariants] A coupled task consumes every resource in a fixed proportion.
  Dominant-resource progressive filling preserves physical owner capacities,
  explicit cohort widening symmetry, supply caps and FIFO eligibility.
  Uncoupled-only scheduling is unchanged.
[Timeouts & Locks] Finite filling over active flows and resources; no I/O.
[Side-Effect Policy] Pure allocation arithmetic; no clock or work mutation.
"""
from __future__ import annotations

import math

_TRIPLE_ARITY = 3
_QUAD_ARITY = 4
_PAIR_MIXED_SHORT_MODE = 3
_PAIR_UNCOUPLED_DISJOINT_MODE = 4
_PAIR_MIXED_SHARED_MODE = 5


def _span_scale(resources, name: str, amount: float, placement, cohort: int,
                widened: tuple[str, ...]) -> float:
    """Single-member bottleneck duration on ``name`` honoring explicit widening."""
    res = resources[name]
    decl = placement.get(name) if placement else None
    n_u = res.units if decl is None else len(decl)
    denom = (res.rate / res.units) * n_u
    if cohort > 1 and name not in widened:
        denom *= cohort
    return amount / denom if denom > 0.0 else math.inf


def _solve_two_flow_slot(
    cap_s: float, pu0: float, pu1: float, limit0: float, limit1: float,
) -> tuple[float, float]:
    """Exact closed-form progressive fill for two flows sharing one slot."""
    if limit0 <= 0.0 and limit1 <= 0.0:
        return 0.0, 0.0
    if limit1 <= 0.0:
        if pu0:
            b0 = cap_s / pu0
            if b0 < limit0:
                limit0 = b0
        return (limit0 if 0.0 < limit0 < math.inf else 0.0), 0.0
    if limit0 <= 0.0:
        if pu1:
            b1 = cap_s / pu1
            if b1 < limit1:
                limit1 = b1
        return 0.0, (limit1 if 0.0 < limit1 < math.inf else 0.0)
    slope = pu0 + pu1
    min_bound = cap_s / slope if slope else math.inf
    delta1 = min_bound
    if limit0 < delta1:
        delta1 = limit0
    if limit1 < delta1:
        delta1 = limit1
    if min_bound <= delta1:
        return delta1, delta1
    if delta1 < limit0:
        rem_s = max(0.0, cap_s - delta1 * slope) if slope else cap_s
        b2 = rem_s / pu0 if pu0 else math.inf
        min_b2 = limit0 - delta1
        if b2 < min_b2:
            min_b2 = b2
        return delta1 + min_b2, delta1
    if delta1 < limit1:
        rem_s = max(0.0, cap_s - delta1 * slope) if slope else cap_s
        b2 = rem_s / pu1 if pu1 else math.inf
        min_b2 = limit1 - delta1
        if b2 < min_b2:
            min_b2 = b2
        return delta1, delta1 + min_b2
    return delta1, delta1


def _coupled_demand_and_cap_limit(resources, task):
    rem = task.remaining
    prep = getattr(task, "prepared", None)
    if prep is not None and not task.service:
        c_dem = prep.coupled_demand
        if c_dem is not None and len(rem) == len(c_dem):
            return c_dem, prep.coupled_cap_limit
    cohort = getattr(task, "cohort_size", 1)
    widened = getattr(task, "cohort_widened", ())
    placement = task.placement
    n_rem = len(rem)
    if n_rem == 1:
        n0 = next(iter(rem))
        r0 = rem[n0]
        scale = _span_scale(resources, n0, r0, placement, cohort, widened)
        demand = {n0: r0 / scale}
    elif n_rem == 2:
        it = iter(rem)
        n0, n1 = next(it), next(it)
        r0, r1 = rem[n0], rem[n1]
        s0 = _span_scale(resources, n0, r0, placement, cohort, widened)
        s1 = _span_scale(resources, n1, r1, placement, cohort, widened)
        scale = s0 if s0 >= s1 else s1
        demand = {n0: r0 / scale, n1: r1 / scale}
    elif n_rem == _TRIPLE_ARITY:
        it = iter(rem)
        n0, n1, n2 = next(it), next(it), next(it)
        r0, r1, r2 = rem[n0], rem[n1], rem[n2]
        scale = max(
            _span_scale(resources, n0, r0, placement, cohort, widened),
            _span_scale(resources, n1, r1, placement, cohort, widened),
            _span_scale(resources, n2, r2, placement, cohort, widened),
        )
        demand = {n0: r0 / scale, n1: r1 / scale, n2: r2 / scale}
    elif n_rem == _QUAD_ARITY:
        it = iter(rem)
        n0, n1, n2, n3 = next(it), next(it), next(it), next(it)
        r0, r1, r2, r3 = rem[n0], rem[n1], rem[n2], rem[n3]
        scale = max(
            _span_scale(resources, n0, r0, placement, cohort, widened),
            _span_scale(resources, n1, r1, placement, cohort, widened),
            _span_scale(resources, n2, r2, placement, cohort, widened),
            _span_scale(resources, n3, r3, placement, cohort, widened),
        )
        demand = {n0: r0 / scale, n1: r1 / scale, n2: r2 / scale, n3: r3 / scale}
    else:
        scale = max(
            _span_scale(resources, name, r, placement, cohort, widened)
            for name, r in rem.items()
        )
        demand = {name: r / scale for name, r in rem.items()}
    limit = math.inf
    caps = task.caps
    if caps:
        for name, amount in demand.items():
            cap = caps.get(name)
            if cap is not None:
                c_lim = cap / amount
                if c_lim < limit:
                    limit = c_lim
    return demand, limit


def _coupled_single(resources, key, task):
    rem = task.remaining
    if not rem:
        return {}
    demand, limit = _coupled_demand_and_cap_limit(resources, task)
    placement = task.placement
    for name, amount in demand.items():
        if amount:
            res = resources[name]
            u_cnt = len(placement[name]) if placement and name in placement else res.units
            per_unit = amount if u_cnt == 1 else amount / u_cnt
            if per_unit:
                cap_lim = (res.rate / res.units) / per_unit
                if cap_lim < limit:
                    limit = cap_lim
    if 0.0 < limit < math.inf:
        return {name: {key: amount * limit} for name, amount in demand.items()}
    return {}


def _uncoupled_component_metrics(resources, name: str, r_u: float, tu):
    res = resources[name]
    u_cap = res.rate / res.units
    pu_map = tu.placement
    uu = pu_map.get(name) if pu_map else None
    uc_u = res.units if uu is None else len(uu)
    amount_u = r_u / (r_u / (u_cap * uc_u)) if u_cap > 0.0 else 0.0
    lim_u = math.inf
    caps_u = tu.caps
    if caps_u and amount_u > 0.0:
        cap_val = caps_u.get(name)
        if cap_val is not None:
            cl = cap_val / amount_u
            if cl < lim_u:
                lim_u = cl
    pu_u = amount_u if uc_u == 1 else amount_u / uc_u
    return u_cap, amount_u, pu_u, lim_u, uu


def _coupled_pair_mixed_fast(resources, kc, tc, ku, tu):
    """Exact closed-form solver for 1 coupled task + 1 uncoupled task on fair pipes."""
    rem_c = tc.remaining
    rem_u = tu.remaining
    prep_c = tc.prepared
    if (
        prep_c is not None and not tc.service and prep_c.coupled_res_info is not None
        and len(rem_c) == len(prep_c.coupled_demand) and len(rem_u) == 1
    ):
        name_u, r_u = next(iter(rem_u.items()))
        prep_u = tu.prepared
        u_info = prep_u.uncoupled_res_info if prep_u is not None else None
        u_row = u_info.get(name_u) if u_info is not None else None
        if name_u not in prep_c.coupled_demand:
            if prep_c.coupled_solo_rates is not None:
                if u_row is not None and r_u == u_row[0]:
                    return (
                        _PAIR_MIXED_SHORT_MODE, kc, tc, prep_c.coupled_solo_rates,
                        ku, tu, name_u, u_row[_QUAD_ARITY],
                    )
                u_cap_u, amount_u, pu_u, lim_u, _ = _uncoupled_component_metrics(
                    resources, name_u, r_u, tu
                )
                if pu_u:
                    c_lim_u = u_cap_u / pu_u
                    if c_lim_u < lim_u:
                        lim_u = c_lim_u
                rate_u = amount_u * lim_u if 0.0 < lim_u < math.inf else 0.0
                return (
                    _PAIR_MIXED_SHORT_MODE, kc, tc, prep_c.coupled_solo_rates,
                    ku, tu, name_u, rate_u,
                )
        else:
            c_info_d, c_items_t, c_excl_d = prep_c.coupled_res_info
            _, cap_s, pu_c, _, uc = c_info_d[name_u]
            if u_row is not None and r_u == u_row[0]:
                _, amount_u, pu_u, lim_u, _, uu = u_row
            else:
                _, amount_u, pu_u, lim_u, uu = _uncoupled_component_metrics(
                    resources, name_u, r_u, tu
                )
            if uc == uu:
                fc, fu_s = _solve_two_flow_slot(
                    cap_s, pu_c, pu_u, c_excl_d[name_u], lim_u
                )
                rate_u = amount_u * fu_s if fu_s else 0.0
                return (
                    _PAIR_MIXED_SHARED_MODE, kc, tc, c_items_t, fc,
                    ku, tu, name_u, rate_u,
                )
    c_info = None
    c_excl = None
    if prep_c is None or prep_c.coupled_demand is None or tc.service:
        for name in rem_c:
            if resources[name].policy == "fifo":
                return None
        demand_c, limit_c = _coupled_demand_and_cap_limit(resources, tc)
    elif len(rem_c) == len(prep_c.coupled_demand):
        demand_c = prep_c.coupled_demand
        limit_c = prep_c.coupled_cap_limit
        if prep_c.coupled_res_info is not None:
            c_info, _, c_excl = prep_c.coupled_res_info
    else:
        demand_c, limit_c = _coupled_demand_and_cap_limit(resources, tc)
    pc = tc.placement
    prep_u = tu.prepared
    u_info = prep_u.uncoupled_res_info if prep_u is not None else None
    indep_u = []
    shared_one = None
    for name, r_u in rem_u.items():
        if u_info is not None:
            u_row = u_info.get(name)
            if u_row is not None and r_u == u_row[0]:
                _, amount_u, pu_u, lim_u, solo_rate_u, uu = u_row
                if name not in demand_c:
                    if solo_rate_u:
                        indep_u.append((name, solo_rate_u))
                    continue
                if c_info is not None:
                    _, u_cap, pu_c, c_lim_c, uc = c_info[name]
                    if uc == uu:
                        if shared_one is not None:
                            return None
                        shared_one = (name, u_cap, pu_c, amount_u, pu_u, lim_u)
                        continue
                    if uc is None or uu is None:
                        return None
                    if not set(uc).isdisjoint(uu):
                        return None
                    if c_lim_c < limit_c:
                        limit_c = c_lim_c
                    if solo_rate_u:
                        indep_u.append((name, solo_rate_u))
                    continue
        u_cap, amount_u, pu_u, lim_u, uu = _uncoupled_component_metrics(
            resources, name, r_u, tu
        )
        if name not in demand_c:
            if pu_u:
                c_lim = u_cap / pu_u
                if c_lim < lim_u:
                    lim_u = c_lim
            fu = lim_u if 0.0 < lim_u < math.inf else 0.0
            if fu:
                indep_u.append((name, amount_u * fu))
            continue
        if c_info is not None:
            _, _, pu_c, c_lim_c, uc = c_info[name]
        else:
            res = resources[name]
            a_c = demand_c[name]
            uc = pc.get(name) if pc else None
            uc_c = res.units if uc is None else len(uc)
            pu_c = a_c if uc_c == 1 else a_c / uc_c
            c_lim_c = (u_cap / pu_c) if pu_c else math.inf
        if uc == uu:
            if shared_one is not None:
                return None
            shared_one = (name, u_cap, pu_c, amount_u, pu_u, lim_u)
            continue
        if uc is None or uu is None:
            return None
        if not set(uc).isdisjoint(uu):
            return None
        if c_lim_c < limit_c:
            limit_c = c_lim_c
        if pu_u:
            c_lim = u_cap / pu_u
            if c_lim < lim_u:
                lim_u = c_lim
        fu = lim_u if 0.0 < lim_u < math.inf else 0.0
        if fu:
            indep_u.append((name, amount_u * fu))
    if c_info is not None:
        if shared_one is not None:
            limit_c = c_excl[shared_one[0]]
        else:
            for name, (_, _, _, c_lim_c, _) in c_info.items():
                if name not in rem_u and c_lim_c < limit_c:
                    limit_c = c_lim_c
    else:
        for name, a_c in demand_c.items():
            if name not in rem_u:
                res = resources[name]
                uc = pc.get(name) if pc else None
                uc_c = res.units if uc is None else len(uc)
                pu_c = a_c if uc_c == 1 else a_c / uc_c
                if pu_c:
                    c_lim = (res.rate / res.units) / pu_c
                    if c_lim < limit_c:
                        limit_c = c_lim
    if shared_one is None:
        fc = limit_c if 0.0 < limit_c < math.inf else 0.0
    else:
        sname, cap_s, pu_c, amount_u, pu_u, lim_u = shared_one
        fc, fu_s = _solve_two_flow_slot(cap_s, pu_c, pu_u, limit_c, lim_u)
        if fu_s:
            indep_u.append((sname, amount_u * fu_s))
    return (0, kc, tc, demand_c, fc, ku, tu, indep_u)


def _uncoupled_pair_fast(resources, k0, t0, k1, t1):
    """Exact two-task solver for 2 uncoupled tasks returning zero-dict rate lists."""
    rem0, rem1 = t0.remaining, t1.remaining
    if not rem0 or not rem1:
        return None
    prep0, prep1 = t0.prepared, t1.prepared
    s0 = prep0.solo_rates if prep0 is not None else None
    s1 = prep1.solo_rates if prep1 is not None else None
    if s0 is not None and s1 is not None and len(rem0) == 1 and len(rem1) == 1:
        n0, n1 = next(iter(rem0)), next(iter(rem1))
        if n0 != n1:
            return (_PAIR_UNCOUPLED_DISJOINT_MODE, k0, t0, n0, s0[n0], k1, t1, n1, s1[n1])
    rates0, rates1 = [], []

    def _solo_cap(task, name):
        res = resources[name]
        if res.policy == "fifo":
            lim = res.rate
        elif task.placement and name in task.placement:
            lim = res.rate / res.units * len(task.placement[name])
        else:
            lim = res.rate
        cap = task.caps.get(name, math.inf) if task.caps else math.inf
        return cap if cap < lim else lim

    for name in rem0:
        if name not in rem1:
            rates0.append((name, s0[name] if s0 is not None else _solo_cap(t0, name)))
            continue
        res = resources[name]
        c0 = t0.caps.get(name, math.inf) if t0.caps else math.inf
        c1 = t1.caps.get(name, math.inf) if t1.caps else math.inf
        if res.policy == "fifo":
            key0 = (t0.start if t0.start is not None else math.inf, t0.priority, k0)
            key1 = (t1.start if t1.start is not None else math.inf, t1.priority, k1)
            if key0 < key1:
                rates0.append((name, c0 if c0 < res.rate else res.rate))
            else:
                rates1.append((name, c1 if c1 < res.rate else res.rate, True))
            continue
        u0 = t0.placement.get(name) if t0.placement else None
        u1 = t1.placement.get(name) if t1.placement else None
        if u0 is None and u1 is None:
            cap = res.rate
        elif u0 is not None and u1 is not None:
            if u0 == u1:
                cap = res.rate / res.units * len(u0)
            elif u0[-1] < u1[0] or u1[-1] < u0[0]:
                ur = res.rate / res.units
                l0, l1 = ur * len(u0), ur * len(u1)
                rates0.append((name, c0 if c0 < l0 else l0))
                rates1.append((name, c1 if c1 < l1 else l1, False))
                continue
            else:
                return None
        else:
            return None
        share = cap * 0.5
        if (c1, k1) < (c0, k0):
            r1 = c1 if c1 < share else share
            rem_cap = cap - r1
            r0 = c0 if c0 < rem_cap else rem_cap
        else:
            r0 = c0 if c0 < share else share
            rem_cap = cap - r0
            r1 = c1 if c1 < rem_cap else rem_cap
        rates0.append((name, r0))
        rates1.append((name, r1, False))
    for name in rem1:
        if name not in rem0:
            rates1.append((name, s1[name] if s1 is not None else _solo_cap(t1, name), True))
    return (2, k0, t0, rates0, k1, t1, rates1)


def _coupled_pair_fast(resources, k0, t0, k1, t1):
    """Exact two-task closed-form solver returning zero-dict rate tuple."""
    rem0, rem1 = t0.remaining, t1.remaining
    if not rem0 or not rem1:
        return None
    if not t0.coupled:
        if t1.coupled:
            return _coupled_pair_mixed_fast(resources, k1, t1, k0, t0)
        return _uncoupled_pair_fast(resources, k0, t0, k1, t1)
    if not t1.coupled:
        return _coupled_pair_mixed_fast(resources, k0, t0, k1, t1)
    prep0, prep1 = t0.prepared, t1.prepared
    c_info0 = c_info1 = None
    if prep0 is None or prep0.coupled_demand is None or t0.service:
        if any(resources[name].policy == "fifo" for name in rem0):
            return None
        demand0, limit0 = _coupled_demand_and_cap_limit(resources, t0)
    elif len(rem0) == len(prep0.coupled_demand):
        demand0, limit0 = prep0.coupled_demand, prep0.coupled_cap_limit
        if prep0.coupled_res_info is not None:
            c_info0 = prep0.coupled_res_info[0]
    else:
        demand0, limit0 = _coupled_demand_and_cap_limit(resources, t0)
    if prep1 is None or prep1.coupled_demand is None or t1.service:
        if any(name not in rem0 and resources[name].policy == "fifo" for name in rem1):
            return None
        demand1, limit1 = _coupled_demand_and_cap_limit(resources, t1)
    elif len(rem1) == len(prep1.coupled_demand):
        demand1, limit1 = prep1.coupled_demand, prep1.coupled_cap_limit
        if prep1.coupled_res_info is not None:
            c_info1 = prep1.coupled_res_info[0]
    else:
        demand1, limit1 = _coupled_demand_and_cap_limit(resources, t1)
    shared_one = None
    shared_more = None
    if c_info0 is not None and c_info1 is not None:
        for name, (_, cap, pu0, clim0, u0) in c_info0.items():
            row1 = c_info1.get(name)
            if row1 is None:
                if clim0 < limit0:
                    limit0 = clim0
                continue
            _, _, pu1, clim1, u1 = row1
            if u0 == u1:
                if shared_one is None:
                    shared_one = (cap, pu0, pu1)
                elif shared_more is None:
                    shared_more = [shared_one, (cap, pu0, pu1)]
                else:
                    shared_more.append((cap, pu0, pu1))
                continue
            if u0 is None or u1 is None:
                return None
            if not set(u0).isdisjoint(u1):
                return None
            if clim0 < limit0:
                limit0 = clim0
            if clim1 < limit1:
                limit1 = clim1
        for name, (_, _, _, clim1, _) in c_info1.items():
            if name not in c_info0 and clim1 < limit1:
                limit1 = clim1
    else:
        p0, p1 = t0.placement, t1.placement
        for name, a0 in demand0.items():
            res = resources[name]
            cap = res.rate / res.units
            u0 = p0.get(name) if p0 else None
            uc0 = res.units if u0 is None else len(u0)
            pu0 = a0 if uc0 == 1 else a0 / uc0
            if name not in demand1:
                if pu0:
                    c_lim = cap / pu0
                    if c_lim < limit0:
                        limit0 = c_lim
                continue
            a1 = demand1[name]
            u1 = p1.get(name) if p1 else None
            uc1 = res.units if u1 is None else len(u1)
            pu1 = a1 if uc1 == 1 else a1 / uc1
            if u0 == u1:
                if shared_one is None:
                    shared_one = (cap, pu0, pu1)
                elif shared_more is None:
                    shared_more = [shared_one, (cap, pu0, pu1)]
                else:
                    shared_more.append((cap, pu0, pu1))
                continue
            if u0 is None or u1 is None:
                return None
            if not set(u0).isdisjoint(u1):
                return None
            if pu0:
                c_lim = cap / pu0
                if c_lim < limit0:
                    limit0 = c_lim
            if pu1:
                c_lim = cap / pu1
                if c_lim < limit1:
                    limit1 = c_lim
        for name, a1 in demand1.items():
            if name not in demand0:
                res = resources[name]
                u1 = p1.get(name) if p1 else None
                uc1 = res.units if u1 is None else len(u1)
                pu1 = a1 if uc1 == 1 else a1 / uc1
                if pu1:
                    c_lim = (res.rate / res.units) / pu1
                    if c_lim < limit1:
                        limit1 = c_lim
    if shared_one is None:
        f0 = limit0 if 0.0 < limit0 < math.inf else 0.0
        f1 = limit1 if 0.0 < limit1 < math.inf else 0.0
    elif shared_more is None:
        f0, f1 = _solve_two_flow_slot(
            shared_one[0], shared_one[1], shared_one[2], limit0, limit1
        )
    elif limit0 <= 0.0 and limit1 <= 0.0:
        f0 = f1 = 0.0
    elif limit1 <= 0.0:
        for cap_s, pu0_s, _ in shared_more:
            if pu0_s:
                b0 = cap_s / pu0_s
                if b0 < limit0:
                    limit0 = b0
        f0, f1 = (limit0 if 0.0 < limit0 < math.inf else 0.0), 0.0
    elif limit0 <= 0.0:
        for cap_s, _, pu1_s in shared_more:
            if pu1_s:
                b1 = cap_s / pu1_s
                if b1 < limit1:
                    limit1 = b1
        f0, f1 = 0.0, (limit1 if 0.0 < limit1 < math.inf else 0.0)
    else:
        min_bound = math.inf
        for cap_s, pu0_s, pu1_s in shared_more:
            slope = pu0_s + pu1_s
            if slope:
                b = cap_s / slope
                if b < min_bound:
                    min_bound = b
        delta1 = min(min_bound, limit0, limit1)
        f0 = f1 = delta1
        if min_bound > delta1:
            if delta1 < limit0:
                min_b2 = limit0 - delta1
                for cap_s, pu0_s, pu1_s in shared_more:
                    slope = pu0_s + pu1_s
                    rem_s = max(0.0, cap_s - delta1 * slope) if slope else cap_s
                    if pu0_s:
                        b2 = rem_s / pu0_s
                        if b2 < min_b2:
                            min_b2 = b2
                f0 = delta1 + min_b2
            elif delta1 < limit1:
                min_b2 = limit1 - delta1
                for cap_s, pu0_s, pu1_s in shared_more:
                    slope = pu0_s + pu1_s
                    rem_s = max(0.0, cap_s - delta1 * slope) if slope else cap_s
                    if pu1_s:
                        b2 = rem_s / pu1_s
                        if b2 < min_b2:
                            min_b2 = b2
                f1 = delta1 + min_b2
    return (1, k0, t0, demand0, f0, k1, t1, demand1, f1)


def _coupled_pair(resources, k0, t0, k1, t1):
    """Exact two-task closed-form solver when >=1 task is coupled on fair pipes."""
    if not t0.coupled and not t1.coupled:
        return None
    fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
    if fast is None:
        return None
    result: dict = {}
    mode = fast[0]
    if mode == 1:
        _, ck0, _, demand0, f0, ck1, _, demand1, f1 = fast
        if f0:
            for name, amount in demand0.items():
                result[name] = {ck0: amount * f0}
        if f1:
            for name, amount in demand1.items():
                result.setdefault(name, {})[ck1] = amount * f1
        return result
    if mode == _PAIR_MIXED_SHORT_MODE:
        _, kc, _, c_rates, ku, _, name_u, rate_u = fast
        for name, rate_c in c_rates:
            result[name] = {kc: rate_c}
        if rate_u:
            result[name_u] = {ku: rate_u}
        return result
    if mode == _PAIR_MIXED_SHARED_MODE:
        _, kc, _, c_items_t, fc, ku, _, name_u, rate_u = fast
        if fc:
            for name, amount in c_items_t:
                result[name] = {kc: amount * fc}
        if rate_u:
            result.setdefault(name_u, {})[ku] = rate_u
        return result
    _, kc, _, demand_c, fc, ku, _, indep_u = fast
    if fc:
        for name, amount in demand_c.items():
            result[name] = {kc: amount * fc}
    for name, rate_u in indep_u:
        result.setdefault(name, {})[ku] = rate_u
    return result


def coupled_rates(resources, tasks):
    """Return work/ms using normalized dominant-resource max-min fairness."""
    n_tasks = len(tasks)
    if n_tasks == 1:
        key, task = next(iter(tasks.items()))
        if task.coupled:
            return _coupled_single(resources, key, task)
        if not task.remaining:
            return {}
    elif n_tasks == 2:
        (k0, t0), (k1, t1) = tasks.items()
        fast = _coupled_pair(resources, k0, t0, k1, t1)
        if fast is not None:
            return fast

    fifo_first = {}
    for key, task in tasks.items():
        for name in task.remaining:
            if resources[name].policy == "fifo":
                prev = fifo_first.get(name)
                if prev is None or key < prev:
                    fifo_first[name] = key
    keys, coefficients, limits, owners = [], [], [], []
    consumers, capacities, members, default_units = [], [], {}, {}
    for key, task in tasks.items():
        rem = task.remaining
        if not rem:
            continue
        groups = (tuple(rem),) if task.coupled else tuple((name,) for name in rem)
        caps = task.caps
        placement = task.placement
        cohort = getattr(task, "cohort_size", 1) if task.coupled else 1
        widened = getattr(task, "cohort_widened", ()) if task.coupled else ()
        for names in groups:
            flow = len(keys)
            keys.append(key)
            owners.append([])
            if len(names) == 1:
                n0 = names[0]
                scale = _span_scale(resources, n0, rem[n0], placement, cohort, widened)
                demand = {n0: rem[n0] / scale}
            elif len(names) == 2:
                n0, n1 = names
                s0 = _span_scale(resources, n0, rem[n0], placement, cohort, widened)
                s1 = _span_scale(resources, n1, rem[n1], placement, cohort, widened)
                scale = s0 if s0 >= s1 else s1
                demand = {n0: rem[n0] / scale, n1: rem[n1] / scale}
            else:
                scale = max(
                    _span_scale(resources, name, rem[name], placement, cohort, widened)
                    for name in names
                )
                demand = {name: rem[name] / scale for name in names}
            coefficients.append(demand)
            limit = math.inf
            for name, amount in demand.items():
                resource = resources[name]
                if caps:
                    cap = caps.get(name)
                    if cap is not None:
                        limit = min(limit, cap / amount)
                if fifo_first and name in fifo_first and fifo_first[name] != key:
                    limit = 0
                units = placement.get(name) if placement else None
                u_cnt = resource.units if units is None else len(units)
                per_unit = amount if u_cnt == 1 else amount / u_cnt
                members.setdefault(name, []).append((flow, per_unit, units))
            limits.append(limit)

    for name, rows in members.items():
        resource = resources[name]
        capacity = resource.rate / resource.units
        if len(rows) == 1:
            flow, amount, _ = rows[0]
            if amount:
                limits[flow] = min(limits[flow], capacity / amount)
            continue
        common = rows[0][2]
        if all(units == common for _, _, units in rows):
            groups = [tuple((flow, amount) for flow, amount, _ in rows)]
        else:
            full_units = None
            resolved_rows = []
            for flow, amount, units in rows:
                if units is None:
                    if full_units is None:
                        full_units = default_units.get(name)
                        if full_units is None:
                            full_units = default_units[name] = tuple(range(resource.units))
                    units = full_units
                resolved_rows.append((flow, amount, units))
            total_u = sum(len(units) for _, _, units in resolved_rows)
            seen = {u for _, _, units in resolved_rows for u in units}
            if len(seen) == total_u:
                for flow, amount, _ in resolved_rows:
                    if amount:
                        limits[flow] = min(limits[flow], capacity / amount)
                continue
            by_unit: dict[int, list[tuple[int, float]]] = {}
            for flow, amount, units in resolved_rows:
                for unit in units:
                    by_unit.setdefault(unit, []).append((flow, amount))
            groups = by_unit.values()
        for grp in groups:
            if len(grp) == 1:
                flow, amount = grp[0]
                if amount:
                    limits[flow] = min(limits[flow], capacity / amount)
                continue
            slot = len(consumers)
            consumers.append(grp)
            capacities.append(capacity)
            for flow, _ in grp:
                owners[flow].append(slot)

    if not consumers:
        result = {}
        for flow, limit in enumerate(limits):
            if 0 < limit < math.inf:
                for name, amount in coefficients[flow].items():
                    result.setdefault(name, {})[keys[flow]] = amount * limit
        return result

    def _slot_slope(rows):
        s = c = 0.0
        for flow, x in rows:
            if active[flow]:
                t = s + x
                c += ((s - t) + x) if abs(s) >= abs(x) else ((x - t) + s)
                s = t
        return s + c

    remaining = capacities
    assigned = [0.0] * len(keys)
    active = [limit > 0 for limit in limits]
    pending = [flow for flow, enabled in enumerate(active) if enabled]
    slopes = [_slot_slope(rows) for rows in consumers]
    while pending:
        bounds = [amount / slope if slope else math.inf for amount, slope in zip(remaining, slopes)]
        delta = min(min(bounds), min(limits[flow] - assigned[flow] for flow in pending))
        for slot, slope in enumerate(slopes):
            if slope:
                remaining[slot] = max(0.0, remaining[slot] - delta * slope)
            if bounds[slot] <= delta:
                for flow, _ in consumers[slot]:
                    active[flow] = False
        survivors, changed = [], set()
        for flow in pending:
            new_val = assigned[flow] + delta
            assigned[flow] = new_val
            if active[flow] and new_val < limits[flow]:
                survivors.append(flow)
            else:
                active[flow] = False
                changed.update(owners[flow])
        pending = survivors
        if pending:
            for slot in changed:
                slopes[slot] = _slot_slope(consumers[slot])

    result = {}
    for flow, fraction in enumerate(assigned):
        if fraction:
            for name, amount in coefficients[flow].items():
                result.setdefault(name, {})[keys[flow]] = amount * fraction
    return result


coupled_rates._single = _coupled_single
coupled_rates._pair = _coupled_pair
