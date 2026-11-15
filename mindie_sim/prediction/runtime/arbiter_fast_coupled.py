"""Single-tier fast per-unit progressive filling for coupled/placed tasks.

[Role] Solve the common single-priority, non-FIFO case of ``_tiered_coupled``
  without per-step dictionary/set allocations by folding private physical units
  into per-flow ``alpha`` upper bounds and running progressive filling only over
  genuinely shared physical units.
[Invariants] Bit-identical ``demand`` and ``weight`` scaling to ``_tier_flows``,
  exact ``math.fsum`` per-unit summation, 16-ULP unit saturation, 8-ULP cap
  retirement, and an independent final ``unit_service_overallocated`` guard on
  every private and shared unit. Returns ``None`` whenever multiple priority
  tiers or FIFO resources are present so ``_tiered_coupled`` handles them.
[Timeouts & Locks] Bounded progressive filling; no threads, sleeps, locks or I/O.
[Side-Effect Policy] Pure arithmetic over caller-supplied task and resource views.
"""
from __future__ import annotations

import math

_MIN_RATE_PER_MS = 1.0 / 1000000 / 1000000
_HALF_STEP = 1.0 / 2.0
_CAP_ULPS = float(2 * 2 * 2)
_SAT_ULPS = _CAP_ULPS * 2.0
_NEXTAFTER_STEPS = int(_SAT_ULPS) * 2 * 2
_MAX_ATTEMPTS = _NEXTAFTER_STEPS * 2 * 2
_UNIT_RANGE_CACHE: dict[int, tuple[int, ...]] = {}


def fast_single_tier_coupled(resources, tasks, active) -> dict[str, dict[int, float]] | None:
    """Fast path for single-priority non-FIFO cohorts; ``None`` delegates out."""
    first_prio = None
    tier: list[int] = []
    tier_append = tier.append
    for key in active:
        task = tasks[key]
        if not task.remaining:
            continue
        prio = task.priority
        if first_prio is None:
            first_prio = prio
        elif prio != first_prio:
            return None
        tier_append(key)
    if not tier:
        return {}
    if len(tier) > 1:
        tier.sort()

    res_info: dict[str, tuple[float, int, float, float, list, tuple[int, ...]]] = {}
    res_get = res_info.get
    range_get = _UNIT_RANGE_CACHE.get
    flow_keys: list[int] = []
    flow_limits: list[float] = []
    flow_rates: list[tuple[tuple[str, float], ...]] = []
    shared_units: list[tuple[str, int, float, float, float, list[tuple[int, float]]]] = []
    fk_append = flow_keys.append
    fl_append = flow_limits.append
    fr_append = flow_rates.append
    su_append = shared_units.append

    nextafter = math.nextafter
    fsum = math.fsum
    ulp = math.ulp
    inf = math.inf
    isfinite = math.isfinite

    for key in tier:
        task = tasks[key]
        remaining = task.remaining
        placement = task.placement
        caps_map = task.caps
        if task.coupled:
            spans: list[tuple[str, float, tuple[int, ...], int, float, list, bool]] = []
            spans_append = spans.append
            blocked = False
            scale = 0.0
            for name, amount in remaining.items():
                if amount <= 0.0:
                    continue
                info = res_get(name)
                if info is None:
                    res = resources[name]
                    if res.policy == "fifo":
                        return None
                    cnt = max(1, int(res.units))
                    u_cap = float(res.rate) / cnt
                    cap_scale = 1.0 if u_cap < 1.0 else u_cap
                    sat_tol = max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(cap_scale))
                    bind_thresh = u_cap - max(_MIN_RATE_PER_MS, _SAT_ULPS * 2.0 * ulp(cap_scale))
                    full_u = range_get(cnt)
                    if full_u is None:
                        full_u = _UNIT_RANGE_CACHE[cnt] = tuple(range(cnt))
                    info = res_info[name] = (u_cap, cnt, sat_tol, bind_thresh, [None, None], full_u)
                u_cap, cnt, _sat_tol, _bind_thresh, sbox, full_u = info
                if u_cap <= _MIN_RATE_PER_MS:
                    blocked = True
                    break
                decl = placement.get(name) if placement else None
                if decl is None:
                    units = full_u
                    n_u = cnt
                    is_full = True
                else:
                    units = (decl if (decl and 0 <= decl[0] and decl[-1] < cnt
                                      and (len(decl) <= 2 or all(0 <= u < cnt for u in decl)))
                             else tuple(u for u in decl if 0 <= u < cnt))
                    n_u = len(units)
                    if not n_u:
                        blocked = True
                        break
                    is_full = units == full_u
                amt_f = float(amount)
                denom = u_cap * n_u
                if denom <= 0.0:
                    scale = inf
                elif scale != inf:
                    val = amt_f / denom
                    if val > scale:
                        scale = val
                spans_append((name, amt_f, units, n_u, u_cap, sbox, is_full))
            if blocked or not spans:
                continue
            if scale == 0.0:
                raise ZeroDivisionError("coupled_scale_underflow")
            if not isfinite(scale) or scale < 0.0:
                continue
            cohort = task.cohort_size or 1
            if cohort < 1:
                cohort = 1
            inv_scale_cohort = cohort / scale
            limit = 1.0
            if caps_map:
                for name, amt_f, _units, _n_u, _u_cap, _sbox, _is_full in spans:
                    cap = caps_map.get(name)
                    if cap is not None:
                        demand = amt_f * inv_scale_cohort
                        c_lim = (max(0.0, float(cap)) / demand) if demand > 0.0 else 0.0
                        if c_lim < limit:
                            limit = c_lim
                if limit <= 0.0:
                    continue
            f_idx = len(flow_keys)
            fk_append(key)
            fl_append(limit)
            fr_append([(name, amt_f * inv_scale_cohort)
                       for name, amt_f, _u, _nu, _uc, _sb, _if in spans])
            for name, amt_f, units, n_u, _u_cap, sbox, is_full in spans:
                weight = (amt_f * inv_scale_cohort) / n_u
                item = (f_idx, weight)
                if is_full:
                    fslot = sbox[0]
                    if fslot is None:
                        sbox[0] = item
                    elif type(fslot) is tuple:
                        sbox[0] = [fslot, item]
                    else:
                        fslot.append(item)
                else:
                    umap = sbox[1]
                    if umap is None:
                        umap = sbox[1] = {}
                    for u in units:
                        cur = umap.get(u)
                        if cur is None:
                            umap[u] = item
                        elif type(cur) is tuple:
                            umap[u] = [cur, item]
                        else:
                            cur.append(item)
            continue

        for name, amount in remaining.items():
            if amount <= 0.0:
                continue
            info = res_get(name)
            if info is None:
                res = resources[name]
                if res.policy == "fifo":
                    return None
                cnt = max(1, int(res.units))
                u_cap = float(res.rate) / cnt
                cap_scale = 1.0 if u_cap < 1.0 else u_cap
                sat_tol = max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(cap_scale))
                bind_thresh = u_cap - max(_MIN_RATE_PER_MS, _SAT_ULPS * 2.0 * ulp(cap_scale))
                full_u = range_get(cnt)
                if full_u is None:
                    full_u = _UNIT_RANGE_CACHE[cnt] = tuple(range(cnt))
                info = res_info[name] = (u_cap, cnt, sat_tol, bind_thresh, [None, None], full_u)
            u_cap, cnt, _sat_tol, _bind_thresh, sbox, full_u = info
            if u_cap <= _MIN_RATE_PER_MS:
                continue
            decl = placement.get(name) if placement else None
            if decl is None:
                units = full_u
                n_u = cnt
                is_full = True
            else:
                units = (decl if (decl and 0 <= decl[0] and decl[-1] < cnt
                                  and (len(decl) <= 2 or all(0 <= u < cnt for u in decl)))
                         else tuple(u for u in decl if 0 <= u < cnt))
                n_u = len(units)
                if not n_u:
                    continue
                is_full = units == full_u
            demand = u_cap * n_u
            if demand <= 0.0:
                continue
            limit = 1.0
            if caps_map:
                cap = caps_map.get(name)
                if cap is not None:
                    c_lim = max(0.0, float(cap)) / demand
                    if c_lim < limit:
                        limit = c_lim
                if limit <= 0.0:
                    continue
            f_idx = len(flow_keys)
            fk_append(key)
            fl_append(limit)
            fr_append(((name, demand),))
            weight = demand / n_u
            item = (f_idx, weight)
            if is_full:
                fslot = sbox[0]
                if fslot is None:
                    sbox[0] = item
                elif type(fslot) is tuple:
                    sbox[0] = [fslot, item]
                else:
                    fslot.append(item)
            else:
                umap = sbox[1]
                if umap is None:
                    umap = sbox[1] = {}
                for u in units:
                    cur = umap.get(u)
                    if cur is None:
                        umap[u] = item
                    elif type(cur) is tuple:
                        umap[u] = [cur, item]
                    else:
                        cur.append(item)

    n_flows = len(flow_keys)
    if not n_flows:
        return {}

    private_checks: list[tuple[str, int, float, int, float]] = []
    pc_append = private_checks.append
    for name, (u_cap, cnt, sat_tol, bind_thresh, sbox, full_u) in res_info.items():
        fslot = sbox[0]
        umap = sbox[1]
        if umap is None:
            if type(fslot) is tuple:
                f_idx, w = fslot
                pc_append((name, 0, u_cap, f_idx, w))
                if w > 0.0:
                    b = u_cap / w
                    if b < flow_limits[f_idx]:
                        flow_limits[f_idx] = b
                    while flow_limits[f_idx] > 0.0 and flow_limits[f_idx] * w > u_cap:
                        flow_limits[f_idx] = nextafter(flow_limits[f_idx], 0.0)
            elif fslot is not None:
                su_append((name, 0, u_cap, sat_tol, bind_thresh, fslot))
            continue
        if fslot is None:
            for u in (sorted(umap) if len(umap) > 1 else umap):
                cur = umap[u]
                if type(cur) is tuple:
                    f_idx, w = cur
                    pc_append((name, u, u_cap, f_idx, w))
                    if w > 0.0:
                        b = u_cap / w
                        if b < flow_limits[f_idx]:
                            flow_limits[f_idx] = b
                        while flow_limits[f_idx] > 0.0 and flow_limits[f_idx] * w > u_cap:
                            flow_limits[f_idx] = nextafter(flow_limits[f_idx], 0.0)
                else:
                    su_append((name, u, u_cap, sat_tol, bind_thresh, cur))
            continue
        if len(umap) < cnt:
            u_rep = next(u for u in full_u if u not in umap)
            if type(fslot) is tuple:
                f_idx, w = fslot
                pc_append((name, u_rep, u_cap, f_idx, w))
                if w > 0.0:
                    b = u_cap / w
                    if b < flow_limits[f_idx]:
                        flow_limits[f_idx] = b
                    while flow_limits[f_idx] > 0.0 and flow_limits[f_idx] * w > u_cap:
                        flow_limits[f_idx] = nextafter(flow_limits[f_idx], 0.0)
            else:
                su_append((name, u_rep, u_cap, sat_tol, bind_thresh, fslot))
        f_is_tuple = type(fslot) is tuple
        for u in (sorted(umap) if len(umap) > 1 else umap):
            cur = umap[u]
            if f_is_tuple and type(cur) is tuple:
                terms = [fslot, cur] if fslot[0] <= cur[0] else [cur, fslot]
            else:
                f_part = (fslot,) if f_is_tuple else fslot
                u_part = (cur,) if type(cur) is tuple else cur
                terms = sorted([*f_part, *u_part])
            su_append((name, u, u_cap, sat_tol, bind_thresh, terms))

    n_shared = len(shared_units)
    if n_shared == 0:
        alphas = flow_limits
    else:
        binding_units: list[tuple[float, float, list[tuple[int, float]]]] = []
        seen_binding: set[tuple[float, tuple[tuple[int, float], ...]]] = set()
        for _name, _u, cap, sat_tol, bind_thresh, terms in shared_units:
            n_t = len(terms)
            max_tot = (
                terms[0][1] * flow_limits[terms[0][0]] if n_t == 1
                else (terms[0][1] * flow_limits[terms[0][0]] + terms[1][1] * flow_limits[terms[1][0]]) if n_t == 2
                else fsum([flow_limits[f_idx] * w for f_idx, w in terms])
            )
            if max_tot > bind_thresh:
                key_u = (cap, tuple(terms))
                if key_u not in seen_binding:
                    seen_binding.add(key_u)
                    binding_units.append((cap, sat_tol, terms))
        if not binding_units:
            alphas = flow_limits
        else:
            alphas = [0.0] * n_flows
            deg = [0] * n_flows
            for _cap, _sat_tol, terms in binding_units:
                for f_idx, _w in terms:
                    deg[f_idx] += 1
            active_flags = [False] * n_flows
            for f_idx in range(n_flows):
                lim = flow_limits[f_idx]
                if deg[f_idx] == 0:
                    alphas[f_idx] = lim
                elif lim > 0.0:
                    active_flags[f_idx] = True
            rem_units: list[tuple[float, float, list[tuple[int, float]]]] = []
            for cap, sat_tol, terms in binding_units:
                if all(deg[f_idx] == 1 and active_flags[f_idx] for f_idx, _w in terms):
                    n_t = len(terms)
                    slope = (
                        terms[0][1] if n_t == 1
                        else (terms[0][1] + terms[1][1]) if n_t == 2
                        else fsum([w for _f, w in terms])
                    )
                    denom = slope if n_t == 1 else nextafter(slope, inf)
                    b = cap / denom
                    if all(b <= flow_limits[f_idx] for f_idx, _w in terms):
                        delta = b
                        solved = False
                        for attempt in range(_MAX_ATTEMPTS):
                            if delta <= 0.0:
                                break
                            tot = (
                                (delta * terms[0][1]) if n_t == 1
                                else (delta * terms[0][1] + delta * terms[1][1]) if n_t == 2
                                else fsum([delta * w for _f, w in terms])
                            )
                            if tot <= cap:
                                res_v = cap - tot
                                tol = (sat_tol if 0.0 <= res_v <= cap
                                       else max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(max(1.0, abs(cap), abs(res_v)))))
                                if res_v <= tol:
                                    for f_idx, _w in terms:
                                        alphas[f_idx] = delta
                                        active_flags[f_idx] = False
                                    solved = True
                                break
                            delta = nextafter(delta, 0.0) if attempt < _NEXTAFTER_STEPS else delta * _HALF_STEP
                        if solved:
                            continue
                rem_units.append((cap, sat_tol, terms))
            if rem_units:
                n_rem = len(rem_units)
                act_indices = [i for i, ok in enumerate(active_flags) if ok]
                committed = [[0.0] * len(rem_units[s_idx][2]) for s_idx in range(n_rem)]
                totals = [0.0] * n_rem
                open_shared = list(range(n_rem))
                while act_indices and open_shared:
                    bound_limit = inf
                    still_open: list[int] = []
                    for s_idx in open_shared:
                        cap, _sat_tol, terms = rem_units[s_idx]
                        if len(terms) == 2:
                            (f0, w0), (f1, w1) = terms
                            a0, a1 = active_flags[f0], active_flags[f1]
                            if not a0 and not a1:
                                continue
                            still_open.append(s_idx)
                            if a0 and a1:
                                slope = w0 + w1
                                denom = nextafter(slope, inf) if slope > 0.0 else 0.0
                            else:
                                slope = w0 if a0 else w1
                                denom = slope
                            if denom > 0.0:
                                b = (cap - totals[s_idx]) / denom
                                if b < bound_limit:
                                    bound_limit = b
                        else:
                            weights = [w for f_idx, w in terms if active_flags[f_idx]]
                            if weights:
                                still_open.append(s_idx)
                                cnt_w = len(weights)
                                slope = weights[0] if cnt_w == 1 else fsum(weights)
                                if slope > 0.0:
                                    denom = slope if cnt_w == 1 else nextafter(slope, inf)
                                    b = (cap - totals[s_idx]) / denom
                                    if b < bound_limit:
                                        bound_limit = b
                    open_shared = still_open
                    if not open_shared:
                        for f_idx in act_indices:
                            alphas[f_idx] = flow_limits[f_idx]
                        break
                    headroom = inf
                    for f_idx in act_indices:
                        hr = flow_limits[f_idx] - alphas[f_idx]
                        if hr < headroom:
                            headroom = hr
                    delta = bound_limit if bound_limit < headroom else headroom
                    if not isfinite(delta) or delta < 0.0:
                        break
                    cand_sub: list[tuple[int, float]] = []
                    for attempt in range(_MAX_ATTEMPTS):
                        if delta <= 0.0:
                            cand_sub = []
                            break
                        c_sub: list[tuple[int, float]] = []
                        unsafe = False
                        for s_idx in open_shared:
                            cap, _sat_tol, terms = rem_units[s_idx]
                            comm_s = committed[s_idx]
                            if len(terms) == 2:
                                (f0, w0), (f1, w1) = terms
                                v0 = (alphas[f0] + delta) * w0 if active_flags[f0] else comm_s[0]
                                v1 = (alphas[f1] + delta) * w1 if active_flags[f1] else comm_s[1]
                                tot = v0 + v1
                            else:
                                vals = [((alphas[f_idx] + delta) * w if active_flags[f_idx] else comm_s[j])
                                        for j, (f_idx, w) in enumerate(terms)]
                                tot = vals[0] if len(vals) == 1 else fsum(vals)
                            if tot > cap:
                                unsafe = True
                                break
                            c_sub.append((s_idx, tot))
                        if not unsafe:
                            cand_sub = c_sub
                            break
                        delta = nextafter(delta, 0.0) if attempt < _NEXTAFTER_STEPS else delta * _HALF_STEP
                    else:
                        delta = 0.0
                        cand_sub = []
                    advanced = False
                    if delta > 0.0:
                        for f_idx in act_indices:
                            val = alphas[f_idx] + delta
                            if val != alphas[f_idx]:
                                advanced = True
                            alphas[f_idx] = val
                        for s_idx in open_shared:
                            comm_s = committed[s_idx]
                            for j, (f_idx, w) in enumerate(rem_units[s_idx][2]):
                                if active_flags[f_idx]:
                                    comm_s[j] = alphas[f_idx] * w
                        for s_idx, tot in cand_sub:
                            totals[s_idx] = tot
                    next_open: list[int] = []
                    froze_any = False
                    for s_idx in open_shared:
                        cap, sat_tol, terms = rem_units[s_idx]
                        res_v = cap - totals[s_idx]
                        tol = (sat_tol if 0.0 <= res_v <= cap
                               else max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(max(1.0, abs(cap), abs(res_v)))))
                        if res_v <= tol:
                            for f_idx, _w in terms:
                                if active_flags[f_idx]:
                                    active_flags[f_idx] = False
                                    froze_any = True
                        else:
                            next_open.append(s_idx)
                    open_shared = next_open
                    no_step = not advanced
                    next_act: list[int] = []
                    for f_idx in act_indices:
                        if not active_flags[f_idx]:
                            continue
                        lim = flow_limits[f_idx]
                        al = alphas[f_idx]
                        cap_tol = _CAP_ULPS * ulp(max(abs(lim), abs(al)))
                        if al >= lim - cap_tol or (no_step and (lim - al) <= headroom + cap_tol):
                            active_flags[f_idx] = False
                            froze_any = True
                        else:
                            next_act.append(f_idx)
                    if not froze_any:
                        break
                    act_indices = next_act

    for name, u, cap, f_idx, w in private_checks:
        if alphas[f_idx] * w > cap:
            raise AssertionError(f"unit_service_overallocated:{name}:{u}")
    for name, u, cap, _sat_tol, _bind_thresh, terms in shared_units:
        n_t = len(terms)
        tot_u = (
            alphas[terms[0][0]] * terms[0][1] if n_t == 1
            else (alphas[terms[0][0]] * terms[0][1] + alphas[terms[1][0]] * terms[1][1]) if n_t == 2
            else fsum([alphas[f_idx] * w for f_idx, w in terms])
        )
        if tot_u > cap:
            raise AssertionError(f"unit_service_overallocated:{name}:{u}")

    result: dict[str, dict[int, float]] = {}
    for f_idx, alpha in enumerate(alphas):
        if alpha <= 0.0:
            continue
        key = flow_keys[f_idx]
        for name, demand in flow_rates[f_idx]:
            target = result.get(name)
            if target is None:
                result[name] = {key: alpha * demand}
            else:
                target[key] = target.get(key, 0.0) + alpha * demand
    return result
