"""Priority-tiered per-unit progressive filling for the causal Engine.

[Role] Solve multi-priority and per-unit pinned/FIFO service rates with exact
  ``math.fsum`` cross-tier conservation across physical resource owners.
[Invariants] Every physical unit's cumulative allocated service across all
  priority tiers stays within its declared capacity.  Non-preemptive FIFO
  ownership holds across tiers; priority orders waiting contenders only.
[Timeouts & Locks] Bounded progressive filling; no threads, sleeps, locks or I/O.
[Side-Effect Policy] Pure arithmetic over caller-supplied task and resource views.
"""
from __future__ import annotations

import math
from typing import Mapping

from .arbiter_fast_coupled import fast_single_tier_coupled


_TRIPLE_CONTENDER_ARITY = 3
_CAP_ULPS = 8.0
_SAT_ULPS = 16.0
_NEXTAFTER_STEPS = 64
_MAX_ATTEMPTS = 256
_FLOW_UNPLACED_IDX = 5
_EMPTY_FROZENSET: frozenset[str] = frozenset()


def _fifo_rank(tasks, key: int, name: str):
    """Non-preemptive FIFO order: active holder on ``name``, priority, arrival."""
    task = tasks[key]
    return (
        task.service.get(name, 0.0) <= 0.0,
        task.priority,
        task.start is None,
        task.start if task.start is not None else math.inf,
        task.fifo_order if getattr(task, "fifo_order", None) is not None else (key, 0),
    )


def _unit_sat_tol(ref_cap: float, residual: float) -> float:
    """Scale-invariant 16-ULP unit saturation tolerance."""
    return _SAT_ULPS * math.ulp(max(abs(ref_cap), abs(residual)))


def _member_placements(task, name: str, cohort: int) -> tuple[tuple[int, ...] | None, ...]:
    """Resolve per-member unit tuples for an uncoupled cohort on ``name``."""
    members_raw = getattr(task, "cohort_members", ())
    if members_raw:
        for rname, members in members_raw:
            if rname == name and len(members) == cohort:
                return members
    pl = task.placement
    units = pl.get(name) if pl else None
    if (units is not None
            and name in getattr(task, "cohort_widened", ())
            and len(units) % cohort == 0):
        width = len(units) // cohort
        return tuple(units[idx * width:(idx + 1) * width] for idx in range(cohort))
    return (units,) * cohort


def _max_min_rates(capacity: float, caps: Mapping[int, float]) -> dict[int, float]:
    """Unprioritized capped max-min water filling."""
    size = len(caps)
    if size == 1:
        key, cap = next(iter(caps.items()))
        return {key: cap if cap < capacity else capacity}
    if size == 2:
        (key1, cap1), (key2, cap2) = caps.items()
        if (cap2, key2) < (cap1, key1):
            key1, cap1, key2, cap2 = key2, cap2, key1, cap1
        share = capacity * 0.5
        rate1 = cap1 if cap1 < share else share
        remaining = capacity - rate1
        return {key1: rate1, key2: cap2 if cap2 < remaining else remaining}
    if size == _TRIPLE_CONTENDER_ARITY:
        (k0, c0), (k1, c1), (k2, c2) = caps.items()
        if (c1, k1) < (c0, k0):
            k0, c0, k1, c1 = k1, c1, k0, c0
        if (c2, k2) < (c1, k1):
            k1, c1, k2, c2 = k2, c2, k1, c1
            if (c1, k1) < (c0, k0):
                k0, c0, k1, c1 = k1, c1, k0, c0
        s0 = capacity / float(_TRIPLE_CONTENDER_ARITY)
        if c0 >= s0:
            return {k0: s0, k1: s0, k2: s0}
        rem1 = capacity - c0
        s1 = rem1 * 0.5
        if c1 >= s1:
            return {k0: c0, k1: s1, k2: s1}
        rem2 = rem1 - c1
        return {k0: c0, k1: c1, k2: c2 if c2 < rem2 else rem2}
    result: dict[int, float] = {}
    consumed: list[float] = []
    remaining = capacity
    ordered = sorted(caps.items(), key=lambda item: (item[1], item[0]))
    total_n = len(ordered)
    for index, (key, cap) in enumerate(ordered):
        share = remaining / (total_n - index)
        if cap >= share:
            for rem_key, _ in ordered[index:]:
                result[rem_key] = share
            break
        result[key] = cap
        consumed.append(cap)
        remaining = max(0.0, capacity - math.fsum(consumed))
    return result


def fair_rates(capacity: float, caps: Mapping[int, float],
               priorities: Mapping[int, int] | None = None) -> dict[int, float]:
    """Max-min allocation with optional strict priority classes."""
    if priorities:
        tiers: dict[int, dict[int, float]] = {}
        for key, cap in caps.items():
            tiers.setdefault(int(priorities.get(key, 0)), {})[key] = cap
        if len(tiers) > 1:
            result: dict[int, float] = {}
            consumed_all: list[float] = []
            remaining = capacity
            for priority in sorted(tiers):
                tier = tiers[priority]
                if remaining <= 0.0:
                    result.update({key: 0.0 for key in tier})
                    continue
                share = _max_min_rates(remaining, tier)
                result.update(share)
                consumed_all.extend(share.values())
                if any(rate < tier[k] for k, rate in share.items()):
                    remaining = 0.0
                else:
                    remaining = max(0.0, capacity - math.fsum(consumed_all))
            return result
    return _max_min_rates(capacity, caps)


def _priority_tiers(active, tasks) -> dict[int, list[int]]:
    tiers: dict[int, list[int]] = {}
    for key in active:
        tiers.setdefault(int(tasks[key].priority), []).append(key)
    return dict(sorted(tiers.items()))


def _unit_span(resource, declared, unit_remaining) -> tuple[int, ...]:
    """Usable physical units for one resource link, or ``()`` if starved.

    A declared placement may never borrow a sibling unit; an unplaced flow
    fans out only over units that still have residual capacity.  Any declared
    unit at zero residual makes the whole link unusable, which is exactly the
    ``alpha`` upper bound of zero in the water-filling formulation.
    """
    count = max(1, int(resource.units))
    if declared is None:
        units = tuple(unit for unit in range(count)
                      if unit_remaining[unit] > 0.0)
    else:
        units = tuple(unit for unit in declared if 0 <= unit < count)
    if not units:
        return ()
    if any(unit_remaining[unit] <= 0.0 for unit in units):
        return ()
    return units


def _tier_flows(resources, tasks, tier, unit_remaining, fifo_winners) -> list:
    """Build one tier's flow vectors against the actual per-unit residuals.

    A coupled task is one flow whose ``alpha`` is a common progress fraction;
    an uncoupled task contributes one independent flow per resource link.  The
    demand vector is normalized to the declared per-unit capacity of the
    flow's dominant resource, so equal ``alpha`` means equal progress and the
    existing coupled fairness contract is preserved while per-owner
    constraints stay exact.
    """
    flows: list = []
    for key in tier:
        task = tasks[key]
        remaining = task.remaining
        if not remaining:
            continue
        placement = task.placement or {}
        if task.coupled:
            spans: dict[str, tuple[int, ...]] = {}
            blocked = False
            for name, amount in remaining.items():
                if amount <= 0.0:
                    continue
                resource = resources[name]
                if resource.policy == "fifo" \
                        and fifo_winners.get(name) != key:
                    blocked = True
                    break
                units = _unit_span(resource, placement.get(name),
                                   unit_remaining[name])
                if not units:
                    blocked = True
                    break
                spans[name] = units
            if blocked or not spans:
                continue
            cohort = max(1, int(getattr(task, "cohort_size", 1) or 1))
            widened_set = getattr(task, "cohort_widened", ())
            scale = 0.0
            for name, units in spans.items():
                capacity = resources[name].rate / max(
                    1, int(resources[name].units))
                denominator = capacity * len(units)
                if cohort > 1 and name not in widened_set:
                    denominator *= cohort
                if denominator <= 0.0:
                    scale = math.inf
                    break
                scale = max(scale, float(remaining[name]) / denominator)
            if scale == 0.0:
                raise ZeroDivisionError("coupled_scale_underflow")
            if not math.isfinite(scale) or scale < 0.0:
                continue
            limit = 1.0
            rates: list[tuple[str, float]] = []
            terms: list[tuple[str, int, float]] = []
            for name, units in spans.items():
                demand = float(remaining[name]) / scale
                rates.append((name, demand))
                weight = demand / len(units)
                terms.extend((name, unit, weight) for unit in units)
                cap = task.caps.get(name)
                if cap is not None:
                    limit = min(limit, (max(0.0, float(cap)) / demand
                                        if demand > 0.0 else 0.0))
            if limit > 0.0:
                unplaced_res = frozenset(
                    name for name, units in spans.items()
                    if placement.get(name) is None and len(units) > 1
                )
                flows.append((key, True, limit, tuple(terms), tuple(rates), unplaced_res))
            continue
        cohort = max(1, int(getattr(task, "cohort_size", 1) or 1))
        for name, amount in remaining.items():
            if amount <= 0.0:
                continue
            resource = resources[name]
            if resource.policy == "fifo" and fifo_winners.get(name) != key:
                continue
            cap = task.caps.get(name)
            entries = (
                tuple((m_decl, None if cap is None else max(0.0, float(cap)) / cohort)
                      for m_decl in _member_placements(task, name, cohort))
                if cohort > 1 else ((placement.get(name), None if cap is None else max(0.0, float(cap))),)
            )
            unit_cap = resource.rate / max(1, int(resource.units))
            for declared, member_cap in entries:
                units = _unit_span(resource, declared, unit_remaining[name])
                if not units:
                    continue
                demand = unit_cap * len(units)
                if demand <= 0.0:
                    continue
                limit = 1.0 if member_cap is None else min(1.0, member_cap / demand)
                if limit <= 0.0:
                    continue
                weight = demand / len(units)
                unplaced_res = (
                    frozenset((name,)) if (declared is None and len(units) > 1)
                    else _EMPTY_FROZENSET
                )
                flows.append((key, False, limit,
                              tuple((name, unit, weight) for unit in units),
                              ((name, demand),), unplaced_res))
    return flows


def _term_value(idx: int, pair: tuple[str, int], alpha: float,
                base_committed, base_alpha, cur_weight) -> float:
    """Committed value on ``pair`` for active flow ``idx`` at progress ``alpha``."""
    rname = pair[0]
    w = cur_weight[idx][rname]
    ba = base_alpha[idx].get(rname, 0.0)
    if ba:
        return base_committed[idx].get(pair, 0.0) + (alpha - ba) * w
    return alpha * w


def _prospective_terms(pair_terms, active, act_units, alphas, base_committed,
                       base_alpha, cur_weight, committed, step: float):
    """Exact tier terms and totals for every unit at ``alpha + step``."""
    terms_by_unit: dict[tuple[str, int], list[float]] = {}
    totals: dict[tuple[str, int], float] = {}
    for pair, terms in pair_terms.items():
        rname, unit = pair
        values = []
        for idx, _weight in terms:
            au = act_units[idx].get(rname)
            if idx in active and (au is None or unit in au):
                values.append(_term_value(
                    idx, pair, alphas[idx] + step,
                    base_committed, base_alpha, cur_weight,
                ))
            else:
                values.append(committed[idx].get(pair, 0.0))
        terms_by_unit[pair] = values
        totals[pair] = math.fsum(values)
    return terms_by_unit, totals


def _tighten_fill_step(delta: float, pair_terms, active, act_units, alphas,
                       base_committed, base_alpha, cur_weight, committed,
                       totals, capacities, cumulative: bool,
                       orig_capacity, cumulative_terms):
    """Tighten ``delta`` via ``nextafter``/halving until every unit sum is safe."""
    blocking: set[tuple[str, int]] = set()
    for attempt in range(_MAX_ATTEMPTS):
        if delta <= 0.0:
            return 0.0, totals, blocking
        candidate_terms, candidate_totals = _prospective_terms(
            pair_terms, active, act_units, alphas, base_committed,
            base_alpha, cur_weight, committed, delta,
        )
        failed: set[tuple[str, int]] = set()
        if cumulative:
            for pair, values in candidate_terms.items():
                if (math.fsum(cumulative_terms[pair[0]][pair[1]] + values)
                        > float(orig_capacity[pair[0]][pair[1]])):
                    failed.add(pair)
        else:
            for pair, total in candidate_totals.items():
                if total > capacities[pair]:
                    failed.add(pair)
        if not failed:
            return delta, candidate_totals, blocking
        blocking = failed
        delta = (math.nextafter(delta, 0.0) if attempt < _NEXTAFTER_STEPS
                 else delta * 0.5)
    return 0.0, totals, blocking


def _retire_or_redistribute(flows, active, alphas, act_units, flow_demand,
                            base_committed, base_alpha, cur_weight, committed,
                            saturated, headrooms, headroom: float,
                            no_step: bool) -> tuple[set[int], bool]:
    """Freeze capped/pinned flows and redistribute unplaced spans on remaining units."""
    frozen: set[int] = set()
    redistributed: list[tuple[int, str]] = []
    shrunk = False
    for idx in active:
        limit = flows[idx][2]
        alpha = alphas[idx]
        cap_tol = _CAP_ULPS * math.ulp(max(abs(limit), abs(alpha)))
        if alpha >= limit - cap_tol:
            frozen.add(idx)
            continue
        au_map = act_units[idx]
        pinned_hit = any(
            name not in au_map and (name, unit) in saturated
            for name, unit, _w in flows[idx][3]
        )
        if pinned_hit:
            frozen.add(idx)
            continue
        exhausted = False
        for rname, u_set in au_map.items():
            drop = [u for u in u_set if (rname, u) in saturated]
            if not drop:
                continue
            shrunk = True
            u_set.difference_update(drop)
            if not u_set:
                exhausted = True
                break
            redistributed.append((idx, rname))
        if exhausted:
            frozen.add(idx)
            continue
        if no_step and headrooms[idx] <= headroom + cap_tol:
            frozen.add(idx)
    for idx, rname in redistributed:
        if idx in frozen:
            continue
        base_alpha[idx][rname] = alphas[idx]
        cur_weight[idx][rname] = flow_demand[idx][rname] / len(act_units[idx][rname])
        for u in act_units[idx][rname]:
            pair = (rname, u)
            base_committed[idx][pair] = committed[idx].get(pair, 0.0)
    return frozen, shrunk


def _progressive_unit_fill(flows, unit_remaining, *, orig_capacity=None,
                           cumulative_terms=None, committed_out=None) -> list[float]:
    """Equal-alpha progressive filling with unified unplaced-span redistribution."""
    cumulative = orig_capacity is not None and cumulative_terms is not None
    n_flows = len(flows)
    alphas = [0.0] * n_flows
    active = set(range(n_flows))
    flow_demand = [dict(flow[4]) for flow in flows]
    act_units: list[dict[str, set[int]]] = [{} for _ in flows]
    cur_weight: list[dict[str, float]] = [{} for _ in flows]
    base_alpha: list[dict[str, float]] = [{} for _ in flows]
    base_committed: list[dict[tuple[str, int], float]] = [{} for _ in flows]
    pair_terms: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for idx, flow in enumerate(flows):
        unplaced = flow[_FLOW_UNPLACED_IDX] if len(flow) > _FLOW_UNPLACED_IDX else _EMPTY_FROZENSET
        for name, unit, weight in flow[3]:
            pair_terms.setdefault((name, unit), []).append((idx, weight))
            cur_weight[idx][name] = weight
            if name in unplaced:
                act_units[idx].setdefault(name, set()).add(unit)
    capacities = {
        (name, unit): (
            float(orig_capacity[name][unit]) - math.fsum(cumulative_terms[name][unit])
            if cumulative else float(value)
        )
        for name, units in unit_remaining.items()
        for unit, value in enumerate(units)
    }
    committed: list[dict[tuple[str, int], float]] = [{} for _ in flows]
    totals = dict.fromkeys(pair_terms, 0.0)
    tier_saturated: set[tuple[str, int]] = set()

    while active:
        bounds_by_pair: dict[tuple[str, int], float] = {}
        for pair, terms in pair_terms.items():
            rname, unit = pair
            weights = [
                cur_weight[idx][rname] for idx, _w in terms
                if idx in active and (rname not in act_units[idx] or unit in act_units[idx][rname])
            ]
            if weights:
                slope = math.fsum(weights)
                if slope > 0.0:
                    denom = slope if len(weights) == 1 else math.nextafter(slope, math.inf)
                    bounds_by_pair[pair] = (capacities[pair] - totals[pair]) / denom
        headrooms = {idx: flows[idx][2] - alphas[idx] for idx in active}
        headroom = min(headrooms.values())
        raw_delta = min(min(bounds_by_pair.values()) if bounds_by_pair else math.inf, headroom)
        if not math.isfinite(raw_delta) or raw_delta < 0.0:
            break
        delta, totals, blocking = _tighten_fill_step(
            raw_delta, pair_terms, active, act_units, alphas, base_committed,
            base_alpha, cur_weight, committed, totals, capacities,
            cumulative, orig_capacity, cumulative_terms,
        )
        advanced = False
        if delta > 0.0:
            for idx in active:
                lim = flows[idx][2]
                value = min(lim, alphas[idx] + delta)
                if value != alphas[idx]:
                    advanced = True
                alphas[idx] = value
                au_map = act_units[idx]
                for name, unit, _w in flows[idx][3]:
                    au = au_map.get(name)
                    if au is None or unit in au:
                        pair = (name, unit)
                        committed[idx][pair] = _term_value(
                            idx, pair, value, base_committed, base_alpha, cur_weight,
                        )
        saturated: set[tuple[str, int]] = set()
        for pair, total in totals.items():
            name, unit = pair
            residual = capacities[pair] - total
            ref_cap = float(orig_capacity[name][unit]) if cumulative else capacities[pair]
            has_uncapped = any(
                idx in active
                and (name not in act_units[idx] or unit in act_units[idx][name])
                and alphas[idx] < flows[idx][2] - _CAP_ULPS * math.ulp(
                    max(abs(flows[idx][2]), abs(alphas[idx])))
                for idx, _w in pair_terms[pair]
            )
            unit_bound = residual <= 0.0 or (
                (pair in blocking or has_uncapped)
                and residual <= _unit_sat_tol(ref_cap, residual)
            )
            if unit_bound:
                unit_remaining[name][unit] = 0.0
                saturated.add(pair)
                tier_saturated.add(pair)
            else:
                unit_remaining[name][unit] = residual
        frozen, shrunk = _retire_or_redistribute(
            flows, active, alphas, act_units, flow_demand, base_committed,
            base_alpha, cur_weight, committed, saturated, headrooms, headroom,
            not advanced,
        )
        if not frozen and not shrunk:
            break
        active -= frozen

    assigned = {
        pair: math.fsum(committed[idx].get(pair, 0.0) for idx, _w in terms)
        for pair, terms in pair_terms.items()
    }
    for pair, total in assigned.items():
        name, unit = pair
        if cumulative:
            capacity = float(orig_capacity[name][unit])
            accumulated = math.fsum(
                cumulative_terms[name][unit]
                + [committed[idx].get(pair, 0.0) for idx, _w in pair_terms[pair]])
            if accumulated > capacity:
                raise AssertionError(f"unit_service_overallocated:{name}:{unit}")
            cumulative_terms[name][unit].extend(
                committed[idx].get(pair, 0.0) for idx, _w in pair_terms[pair])
            unit_remaining[name][unit] = (
                0.0 if pair in tier_saturated else max(0.0, capacity - accumulated)
            )
            continue
        capacity = capacities.get((name, unit), total)
        if total > capacity + _unit_sat_tol(capacity, total):
            raise AssertionError(f"unit_service_overallocated:{name}:{unit}")
    if committed_out is not None:
        committed_out.extend(committed)
    return alphas


def _tiered_coupled(
    resources, tasks, active, solver, *, allow_native: bool = True,
) -> dict[str, dict[int, float]]:
    """Strict priority tiers solved collectively against per-unit residuals.

    Coupled contenders in one tier are never clipped independently: the whole
    tier is filled by ``_progressive_unit_fill`` over the real
    ``unit_remaining[r][u]`` capacities and each task's declared caps.  A
    coupled task's resource rates therefore remain in the exact lockstep
    proportion ``rate_r = alpha * W_r / scale`` while the sum on every owner
    unit provably stays within that unit's residual.  ``solver`` remains part
    of the call contract for the Engine's shared arbiter signature; the
    per-unit fill must own the computation to see the owner dimension.
    """
    del solver
    fast = fast_single_tier_coupled(
        resources, tasks, active, allow_native=allow_native)
    if fast is not None:
        return fast
    orig_capacity = {
        name: [float(resource.rate) / max(1, int(resource.units))]
        * max(1, int(resource.units))
        for name, resource in resources.items()}
    cumulative_terms = {
        name: [[] for _unit in units] for name, units in orig_capacity.items()}
    unit_remaining = {name: list(units) for name, units in orig_capacity.items()}
    result: dict[str, dict[int, float]] = {}
    fifo_winners = {
        name: min((key for key in active
                   if tasks[key].remaining.get(name, 0.0) > 0.0),
                  key=lambda key, rname=name: _fifo_rank(tasks, key, rname))
        for name, resource in resources.items()
        if resource.policy == "fifo"
        and any(tasks[key].remaining.get(name, 0.0) > 0.0 for key in active)}
    for _priority, tier_keys in _priority_tiers(active, tasks).items():
        tier = sorted(key for key in tier_keys if tasks[key].remaining)
        if not tier:
            continue
        flows = _tier_flows(resources, tasks, tier, unit_remaining,
                            fifo_winners)
        if not flows:
            continue
        committed_tier: list[dict[tuple[str, int], float]] = []
        alphas = _progressive_unit_fill(
            flows, unit_remaining, orig_capacity=orig_capacity,
            cumulative_terms=cumulative_terms, committed_out=committed_tier)
        for index, alpha in enumerate(alphas):
            flow = flows[index]
            key = flow[0]
            if (not flow[1] and len(flow) > _FLOW_UNPLACED_IDX
                    and flow[_FLOW_UNPLACED_IDX]):
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
            total = math.fsum(terms)
            capacity = orig_capacity[name][unit]
            if total > capacity:
                raise AssertionError(
                    f"unit_service_overallocated:{name}:{unit}")
    return {name: rows for name, rows in result.items() if rows}


def _lockstep_coupled(result: dict[str, dict[int, float]], tasks) -> None:
    """Apply one common progress fraction to every coupled task's resources.

    Independent per-resource clipping can leave a task with zero HBM but a
    positive Vector rate.  A coupled pipeline advances on the slowest supplied
    resource, so its progress is ``min_r(rate_r / remaining_r)`` and every
    resource consumes exactly that fraction of its own remaining work.
    """
    per_task: dict[int, dict[str, float]] = {}
    for name, rows in result.items():
        for key, rate in rows.items():
            per_task.setdefault(key, {})[name] = rate
    for key, rates in per_task.items():
        task = tasks[key]
        if not getattr(task, "coupled", False) or not task.remaining:
            continue
        fractions = [float(rates.get(name, 0.0)) / float(amount)
                     for name, amount in task.remaining.items()
                     if amount > 0.0]
        progress = min(fractions) if fractions else 0.0
        for name in list(rates):
            amount = float(task.remaining.get(name, 0.0))
            scaled = max(0.0, progress) * amount
            if scaled <= 0.0:
                result.get(name, {}).pop(key, None)
            else:
                result[name][key] = scaled


def _tighten_residual_step(delta: float, active, units, result, base_result,
                           base_committed, committed, caps, count: int,
                           orig_cap: list[float], cumulative_terms: list[list[float]]):
    """Tighten ``delta`` so every unit's exact ``fsum`` stays within ``orig_cap``."""
    blocking: set[int] = set()
    for attempt in range(_MAX_ATTEMPTS):
        if delta <= 0.0:
            return 0.0, None, blocking
        by_unit: list[list[float]] = [[] for _ in range(count)]
        for key, comm_map in committed.items():
            is_act = key in active
            if is_act:
                next_r = min(caps[key], result[key] + delta)
                u_list = units[key]
                n_u = len(u_list)
                br = base_result[key]
                step_u = (next_r - br) / n_u
                for u, val in comm_map.items():
                    if u in u_list:
                        by_unit[u].append(base_committed[key].get(u, 0.0) + step_u)
                    elif val > 0.0:
                        by_unit[u].append(val)
                for u in u_list:
                    if u not in comm_map:
                        by_unit[u].append(base_committed[key].get(u, 0.0) + step_u)
            else:
                for u, val in comm_map.items():
                    if val > 0.0:
                        by_unit[u].append(val)
        failed = {
            u for u in range(count)
            if math.fsum(cumulative_terms[u] + by_unit[u]) > orig_cap[u]
        }
        if not failed:
            return delta, by_unit, blocking
        blocking = failed
        delta = math.nextafter(delta, 0.0) if attempt < _NEXTAFTER_STEPS else delta * 0.5
    return 0.0, None, blocking


def _residual_unit_rates(unit_remaining: list[float],
                         caps: Mapping,
                         placements: Mapping,
                         explicit: set | None = None,
                         *,
                         orig_capacity: list[float] | None = None,
                         cumulative_terms: list[list[float]] | None = None
                         ) -> dict:
    """Max-min with a distinct residual capacity on each physical unit."""
    count = len(unit_remaining)
    explicit = set(explicit or ())
    orig_cap = [float(v) for v in (orig_capacity if orig_capacity is not None else unit_remaining)]
    cum_terms = cumulative_terms if cumulative_terms is not None else [[] for _ in range(count)]
    result = dict.fromkeys(caps, 0.0)
    base_result = dict.fromkeys(caps, 0.0)
    base_committed: dict = {key: {} for key in caps}
    committed: dict = {key: {} for key in caps}
    units: dict = {}
    for key in caps:
        declared = placements.get(key)
        if declared is None:
            units[key] = [unit for unit in range(count) if unit_remaining[unit] > 0.0]
        else:
            owned = [unit for unit in declared if 0 <= unit < count]
            units[key] = ([] if any(unit_remaining[unit] <= 0.0 for unit in owned) else owned)
    active = {key for key in caps if caps[key] > 0.0 and units[key]}
    tier_saturated: set[int] = set()
    while active:
        slopes: dict[int, float] = {}
        for unit in range(count):
            if unit_remaining[unit] <= 0.0:
                continue
            weights = [1.0 / len(units[key]) for key in active if unit in units[key]]
            if weights:
                slope = math.fsum(weights)
                if slope > 0.0:
                    slopes[unit] = slope if len(weights) == 1 else math.nextafter(slope, math.inf)
        if not slopes:
            break
        bounds = {unit: unit_remaining[unit] / slope for unit, slope in slopes.items()}
        headrooms = {key: caps[key] - result[key] for key in active}
        min_hr = min(headrooms.values())
        raw_delta = min(min(bounds.values()), min_hr)
        if not math.isfinite(raw_delta) or raw_delta < 0.0:
            break
        delta, by_unit, blocking = _tighten_residual_step(
            raw_delta, active, units, result, base_result, base_committed,
            committed, caps, count, orig_cap, cum_terms,
        )
        advanced = False
        if delta > 0.0 and by_unit is not None:
            for key in active:
                nv = min(caps[key], result[key] + delta)
                if nv != result[key]:
                    advanced = True
                result[key] = nv
                u_list = units[key]
                step_u = (nv - base_result[key]) / len(u_list)
                for u in u_list:
                    committed[key][u] = base_committed[key].get(u, 0.0) + step_u
        saturated: set[int] = set()
        for unit in slopes:
            tier_vals = [committed[k].get(unit, 0.0) for k in committed if committed[k].get(unit, 0.0) > 0.0]
            rem = orig_cap[unit] - math.fsum(cum_terms[unit] + tier_vals)
            has_uncapped = any(
                unit in units[k] and result[k] < caps[k] - (
                    _CAP_ULPS * math.ulp(max(abs(caps[k]), abs(result[k])))
                    if math.isfinite(caps[k]) else 0.0)
                for k in active
            )
            unit_bound = rem <= 0.0 or (
                (unit in blocking or has_uncapped)
                and rem <= _unit_sat_tol(orig_cap[unit], rem)
            )
            if unit_bound:
                unit_remaining[unit] = 0.0
                saturated.add(unit)
                tier_saturated.add(unit)
            else:
                unit_remaining[unit] = rem
        progressed = False
        for key in list(active):
            cap_k = caps[key]
            cap_tol = (_CAP_ULPS * math.ulp(max(abs(cap_k), abs(result[key])))
                       if math.isfinite(cap_k) else 0.0)
            if result[key] >= cap_k - cap_tol or (not advanced and headrooms[key] <= min_hr + cap_tol):
                active.discard(key)
                progressed = True
                continue
            if not any(unit in saturated for unit in units[key]):
                continue
            progressed = True
            if key in explicit:
                active.discard(key)
            else:
                units[key] = [unit for unit in units[key] if unit not in saturated]
                if not units[key]:
                    active.discard(key)
                else:
                    base_result[key] = result[key]
                    for u in units[key]:
                        base_committed[key][u] = committed[key].get(u, 0.0)
        if not progressed:
            break
    for unit in range(count):
        tier_vals = [committed[k].get(unit, 0.0) for k in committed if committed[k].get(unit, 0.0) > 0.0]
        if tier_vals:
            cum_terms[unit].extend(tier_vals)
        unit_remaining[unit] = (
            0.0 if unit in tier_saturated
            else max(0.0, orig_cap[unit] - math.fsum(cum_terms[unit]))
        )
    return result


def _expand_simple_tier(tasks, tier, name: str, count: int,
                        available: tuple[int, ...], unit_remaining: list[float]):
    """Build placement/cap maps for one fair tier, expanding ``cohort_size > 1``."""
    has_cohort = any(getattr(tasks[key], "cohort_size", 1) > 1 for key in tier)
    placements: dict = {}
    explicit: set = set()
    scoped: dict = {}
    for key, cap in tier.items():
        task = tasks[key]
        cohort = max(1, int(getattr(task, "cohort_size", 1) or 1)) if has_cohort else 1
        entries = (
            tuple(((key, idx), cap / cohort, m_decl)
                  for idx, m_decl in enumerate(_member_placements(task, name, cohort)))
            if has_cohort else ((key, cap, task.placement.get(name)),)
        )
        for c_key, c_cap, declared in entries:
            if declared is None:
                placements[c_key] = available
                scoped[c_key] = c_cap
                continue
            owned = tuple(unit for unit in declared if 0 <= unit < count)
            if not owned or any(unit_remaining[unit] <= 0.0 for unit in owned):
                continue
            placements[c_key] = owned
            explicit.add(c_key)
            scoped[c_key] = c_cap
    return scoped, placements, explicit, has_cohort


def _tiered_simple(resources, tasks, active) -> dict[str, dict[int, float]]:
    """Strict tiers for fair/fifo and placed/unplaced water-filling.

    Residual capacity is tracked *per physical unit*, so a better priority
    class that pins unit 0 cannot leave its residual as if it were available
    on every unit to a worse class.  An unplaced task fans out only over units
    that actually retain residual capacity; it can never dilute a pinned
    higher-priority owner on an exhausted unit.  FIFO is non-preemptive: an
    already-started holder outranks every waiting task regardless of priority,
    and priority only orders the waiting queue for the next admission.
    """
    contenders: dict[str, dict[int, float]] = {}
    for key in sorted(active):
        for name, amount in tasks[key].remaining.items():
            if amount > 0.0:
                contenders.setdefault(name, {})[key] = tasks[key].caps.get(name, math.inf)
    result: dict[str, dict[int, float]] = {}
    for name, caps in contenders.items():
        resource = resources[name]
        count = max(1, int(resource.units))
        orig_caps = [float(resource.rate) / count] * count
        unit_remaining = list(orig_caps)
        cumulative_terms: list[list[float]] = [[] for _ in range(count)]
        shares: dict[int, float] = {}

        if resource.policy == "fifo":
            eligible = {key: cap for key, cap in caps.items() if cap > 0.0}
            if eligible:
                winner = min(eligible, key=lambda key, rname=name: _fifo_rank(tasks, key, rname))
                declared = tasks[winner].placement.get(name)
                placements = {winner: declared}
                tier_shares = _residual_unit_rates(
                    unit_remaining, {winner: eligible[winner]}, placements,
                    {winner} if declared is not None else set(),
                    orig_capacity=orig_caps, cumulative_terms=cumulative_terms)
                rate = tier_shares.get(winner, 0.0)
                if rate > 0.0:
                    shares[winner] = rate
            result[name] = {key: rate for key, rate in shares.items() if rate > 0.0}
            continue
        for _priority, tier_keys in _priority_tiers(caps, tasks).items():
            tier = {key: caps[key] for key in tier_keys if caps[key] > 0.0}
            remaining = math.fsum(unit_remaining)
            if not tier or remaining <= 0.0:
                continue
            available = tuple(unit for unit in range(count)
                              if unit_remaining[unit] > 0.0)
            if not available:
                break
            scoped, placements, explicit, has_cohort = _expand_simple_tier(
                tasks, tier, name, count, available, unit_remaining)
            if not placements:
                continue
            max_u = max(unit_remaining)
            uniform = max_u - min(unit_remaining) <= _SAT_ULPS * math.ulp(max_u)
            if uniform and not explicit:
                tier_shares = fair_rates(remaining, scoped)
                tier_sat = any(rate < scoped[c_key] for c_key, rate in tier_shares.items())
                for c_key, rate in tier_shares.items():
                    if rate > 0.0:
                        orig_key = c_key[0] if has_cohort else c_key
                        shares[orig_key] = shares.get(orig_key, 0.0) + rate
                        u_tuple = placements.get(c_key, available)
                        per_u = rate / len(u_tuple)
                        for u in u_tuple:
                            cumulative_terms[u].append(per_u)
                for u in available:
                    unit_remaining[u] = (
                        0.0 if tier_sat
                        else max(0.0, orig_caps[u] - math.fsum(cumulative_terms[u]))
                    )
            else:
                tier_shares = _residual_unit_rates(
                    unit_remaining, scoped, placements, explicit,
                    orig_capacity=orig_caps, cumulative_terms=cumulative_terms)
                for c_key, rate in tier_shares.items():
                    if rate > 0.0:
                        orig_key = c_key[0] if has_cohort else c_key
                        shares[orig_key] = shares.get(orig_key, 0.0) + rate
        result[name] = {key: rate for key, rate in shares.items() if rate > 0.0}
    return result


