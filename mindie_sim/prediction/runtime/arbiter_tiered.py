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

from ..ownership import unit_rates
from .arbiter_fast_coupled import fast_single_tier_coupled


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
    if size == 3:
        (k0, c0), (k1, c1), (k2, c2) = caps.items()
        if (c1, k1) < (c0, k0):
            k0, c0, k1, c1 = k1, c1, k0, c0
        if (c2, k2) < (c1, k1):
            k1, c1, k2, c2 = k2, c2, k1, c1
            if (c1, k1) < (c0, k0):
                k0, c0, k1, c1 = k1, c1, k0, c0
        s0 = capacity / 3.0
        r0 = c0 if c0 < s0 else s0
        rem1 = capacity - r0
        s1 = rem1 * 0.5
        r1 = c1 if c1 < s1 else s1
        rem2 = rem1 - r1
        return {k0: r0, k1: r1, k2: c2 if c2 < rem2 else rem2}
    result: dict[int, float] = {}
    remaining = capacity
    ordered = sorted(caps.items(), key=lambda item: (item[1], item[0]))
    for index, (key, cap) in enumerate(ordered):
        share = remaining / (len(ordered) - index)
        rate = min(cap, share)
        result[key] = rate
        remaining -= rate
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
            remaining = capacity
            for priority in sorted(tiers):
                tier = tiers[priority]
                if remaining <= 0:
                    result.update({key: 0.0 for key in tier})
                    continue
                share = _max_min_rates(remaining, tier)
                result.update(share)
                remaining = max(0.0, remaining - sum(share.values()))
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
                      if unit_remaining[unit] > 1e-12)
    else:
        units = tuple(unit for unit in declared if 0 <= unit < count)
    if not units:
        return ()
    if any(unit_remaining[unit] <= 1e-12 for unit in units):
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
                flows.append((key, True, limit, tuple(terms), tuple(rates)))
            continue
        for name, amount in remaining.items():
            if amount <= 0.0:
                continue
            resource = resources[name]
            if resource.policy == "fifo" and fifo_winners.get(name) != key:
                continue
            units = _unit_span(resource, placement.get(name),
                               unit_remaining[name])
            if not units:
                continue
            demand = (resource.rate / max(1, int(resource.units))) * len(units)
            if demand <= 0.0:
                continue
            limit = 1.0
            cap = task.caps.get(name)
            if cap is not None:
                limit = min(limit, max(0.0, float(cap)) / demand)
            if limit <= 0.0:
                continue
            weight = demand / len(units)
            flows.append((key, False, limit,
                          tuple((name, unit, weight) for unit in units),
                          ((name, demand),)))
    return flows


def _progressive_unit_fill(flows, unit_remaining, *, orig_capacity=None,
                           cumulative_terms=None) -> list[float]:
    """Equal-alpha progressive filling bounded by every physical unit.

    At each step every unfrozen flow advances by the same ``alpha`` increment.
    The increment stops at the first task cap or the first ``(resource, unit)``
    residual that would be exhausted.  A *cap-limited* step freezes only the
    flow(s) that actually reached their own declared cap and keeps their
    allocated service; every uncapped peer continues on the residual capacity,
    even when the common increment is as small as ``1e-13`` or a representable
    zero.  A *unit-limited* step uses the scale-aware 16-ULP physical-capacity
    saturation check to retire only the flows blocked on that saturated unit
    while independent non-blocked flows continue.

    Every per-unit total is a ``math.fsum`` of the *actual* committed
    ``alpha * weight`` terms.  The active slope is the ``math.fsum`` of the
    active weights, upper-rounded once with ``nextafter`` before the division,
    so a candidate increment cannot overshoot the exact residual.  Each
    candidate is then re-summed exactly before it is committed: if the rounded
    per-peer products would cross the unit capacity, ``alpha`` is tightened by
    ``nextafter`` (then halved only if no representable step suffices).  A
    genuine oversubscription still raises from the final independent guard, so
    the invariant is never removed.

    When ``orig_capacity`` and ``cumulative_terms`` are supplied the caller owns
    one cross-tier ledger.  The residual is then always
    ``orig_capacity[u] - math.fsum(cumulative_terms[u])`` rather than a
    previously rounded per-tier remainder, the prospective candidate is checked
    as ``math.fsum(cumulative_terms[u] + new_tier_terms[u]) <=
    orig_capacity[u]``, and the committed terms are appended to the shared list
    before the next tier starts.  A locally correct tier sum therefore can never
    hide drift already embedded in an incoming residual.
    """
    cumulative = orig_capacity is not None and cumulative_terms is not None
    alphas = [0.0] * len(flows)
    active = set(range(len(flows)))
    # Per-unit capacity at entry to this tier: the exact bound every committed
    # term is checked against.  Residual and capacity accounting stay separate.
    capacities: dict[tuple[str, int], float] = {}
    for name, units in unit_remaining.items():
        for unit, value in enumerate(units):
            pair = (name, unit)
            if cumulative:
                capacities[pair] = float(orig_capacity[name][unit]) - math.fsum(
                    cumulative_terms[name][unit])
            else:
                capacities[pair] = float(value)
    pair_terms: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for index, flow in enumerate(flows):
        for name, unit, weight in flow[3]:
            pair_terms.setdefault((name, unit), []).append((index, weight))
    committed: list[dict[tuple[str, int], float]] = [dict() for _ in flows]

    def prospective(step: float = 0.0
                    ) -> tuple[dict[tuple[str, int], list[float]],
                               dict[tuple[str, int], float]]:
        """Exact tier terms and totals for every unit at ``alpha + step``.

        An active peer's term is evaluated as ``(alpha + step) * weight``, the
        exact expression the returned rate uses; inactive peers keep their last
        committed term.  This makes the prospective capacity check, the stored
        committed term and the independent final guard agree bit-for-bit.
        """
        terms_by_unit: dict[tuple[str, int], list[float]] = {}
        totals: dict[tuple[str, int], float] = {}
        for pair, terms in pair_terms.items():
            values = []
            for index, weight in terms:
                if index in active:
                    values.append((alphas[index] + step) * weight)
                else:
                    values.append(committed[index].get(pair, 0.0))
            terms_by_unit[pair] = values
            totals[pair] = math.fsum(values)
        return terms_by_unit, totals

    totals = dict.fromkeys(pair_terms, 0.0)
    while active:
        slopes: dict[tuple[str, int], tuple[float, int]] = {}
        for pair, terms in pair_terms.items():
            weights = [weight for index, weight in terms if index in active]
            if weights:
                slope = math.fsum(weights)
                if slope > 0.0:
                    slopes[pair] = (slope, len(weights))
        bounds = [(capacities[pair] - totals[pair])
                  / (slope if count == 1 else math.nextafter(slope, math.inf))
                  for pair, (slope, count) in slopes.items()]
        headrooms = {index: flows[index][2] - alphas[index]
                     for index in active}
        headroom = min(headrooms.values())
        bound_limit = min(bounds) if bounds else math.inf
        delta = min(bound_limit, headroom)
        if not math.isfinite(delta) or delta < 0.0:
            break
        # Verify the exact committed sum of the rounded per-peer products
        # before touching any alpha or residual.  Nextafter steps at capacity
        # scale are sub-ULP; halving is the conservative fallback for an
        # increment that no representable step can make safe.  In cumulative
        # mode the authoritative check is the exact all-tier sum against the
        # original physical capacity, never the rounded local remainder.
        candidate_totals = totals
        for attempt in range(256):
            if delta <= 0.0:
                candidate_totals = totals
                break
            candidate_terms, candidate_totals = prospective(delta)
            if cumulative:
                unsafe = any(
                    math.fsum(cumulative_terms[pair[0]][pair[1]] + values)
                    > float(orig_capacity[pair[0]][pair[1]])
                    for pair, values in candidate_terms.items())
            else:
                unsafe = any(total > capacities[pair]
                             for pair, total in candidate_totals.items())
            if not unsafe:
                break
            delta = (math.nextafter(delta, 0.0) if attempt < 64
                     else delta * 0.5)
        else:
            delta = 0.0
            candidate_totals = totals
        # Advance only by a representable positive increment.  A zero or an
        # absorbed increment is not a reason to abandon unrelated peers: the
        # flows responsible are frozen below and the remaining set continues.
        advanced = False
        if delta > 0.0:
            for index in active:
                value = alphas[index] + delta
                if value != alphas[index]:
                    advanced = True
                alphas[index] = value
                for name, unit, weight in flows[index][3]:
                    committed[index][(name, unit)] = value * weight
        totals = candidate_totals
        for pair, total in totals.items():
            name, unit = pair
            residual = capacities[pair] - total
            ulp_tol = max(1e-12, 16.0 * math.ulp(
                max(1.0, abs(capacities[pair]), abs(residual))))
            # A saturated unit transitions cleanly to exactly zero residual;
            # pure sub-16-ULP rounding dust never leaves a phantom remainder.
            unit_remaining[name][unit] = (
                0.0 if residual <= ulp_tol else residual)
        saturated: set[tuple[str, int]] = set()
        for pair in pair_terms:
            name, unit = pair
            residual = unit_remaining[name][unit]
            unit_tol = max(1e-12, 16.0 * math.ulp(
                max(1.0, abs(capacities[pair]), abs(residual))))
            if residual <= unit_tol:
                unit_remaining[name][unit] = 0.0
                saturated.add(pair)
        frozen: set[int] = set()
        no_step = not advanced
        for index in active:
            limit = flows[index][2]
            alpha = alphas[index]
            # A dimensionless cap tolerance scaled to its own magnitude: an
            # absolute 1e-12 would wrongly pre-freeze a valid 1e-13 cap.
            cap_tol = 8.0 * math.ulp(max(abs(limit), abs(alpha)))
            if alpha >= limit - cap_tol:
                frozen.add(index)
                continue
            if any((name, unit) in saturated
                   for name, unit, _weight in flows[index][3]):
                frozen.add(index)
                continue
            if no_step and headrooms[index] <= headroom + cap_tol:
                # No representable common increment remains; retire only the
                # flow(s) closest to their cap so an independent peer can use
                # the residual instead of the whole tier stalling.
                frozen.add(index)
        if not frozen:
            break
        active -= frozen
    # Independent final bound check.  In cumulative mode it is the exact
    # all-tier ``math.fsum`` against the original physical capacity; otherwise
    # it is the exact sum of this tier's committed ``alpha * weight`` terms
    # against the tier entry capacity (sub-16-ULP dust allowed, as before).
    assigned = {pair: math.fsum(committed[index].get(pair, 0.0)
                                for index, _weight in terms)
                for pair, terms in pair_terms.items()}
    for pair, total in assigned.items():
        name, unit = pair
        if cumulative:
            capacity = float(orig_capacity[name][unit])
            accumulated = math.fsum(
                cumulative_terms[name][unit]
                + [committed[index].get(pair, 0.0)
                   for index, _weight in pair_terms[pair]])
            if accumulated > capacity:
                raise AssertionError(
                    f"unit_service_overallocated:{name}:{unit}")
            # ``committed`` holds this tier's running total per flow, so the
            # tier is appended exactly once at exit; appending per step would
            # double count the earlier alpha increments of the same tier.
            cumulative_terms[name][unit].extend(
                committed[index].get(pair, 0.0)
                for index, _weight in pair_terms[pair])
            # The next tier derives its headroom from the same ledger, never
            # from the rounded per-tier remainder that caused the drift.
            unit_remaining[name][unit] = capacity - accumulated
            continue
        capacity = capacities.get((name, unit), total)
        ulp_tol = max(1e-12, 16.0 * math.ulp(
            max(1.0, abs(capacity), abs(total))))
        if total > capacity + ulp_tol:
            raise AssertionError(
                f"unit_service_overallocated:{name}:{unit}")
    return alphas


def _tiered_coupled(resources, tasks, active, solver) -> dict[str, dict[int, float]]:
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
    fast = fast_single_tier_coupled(resources, tasks, active)
    if fast is not None:
        return fast
    # One immutable physical ledger for the whole tier walk.  ``orig_capacity``
    # is the per-unit capacity declared by the resource; ``cumulative_terms`` is
    # the single shared list of every committed ``alpha * weight`` term from all
    # priority tiers.  Residuals are always ``orig - math.fsum(all terms)``, so
    # a rounded per-tier remainder can never become the next tier's capacity.
    orig_capacity = {
        name: [float(resource.rate) / max(1, int(resource.units))]
        * max(1, int(resource.units))
        for name, resource in resources.items()}
    cumulative_terms = {
        name: [[] for _unit in units] for name, units in orig_capacity.items()}
    unit_remaining = {name: list(units) for name, units in orig_capacity.items()}
    result: dict[str, dict[int, float]] = {}
    # FIFO ownership is non-preemptive across every priority tier: the active
    # holder owns its physical queue until it retires.  Only waiting tasks are
    # ordered by priority for the *next* admission.
    fifo_winners = {
        name: min((key for key in active
                   if tasks[key].remaining.get(name, 0.0) > 0.0),
                  key=lambda key: (
                      tasks[key].start is None,
                      tasks[key].start if tasks[key].start is not None else math.inf,
                      tasks[key].priority, key))
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
        alphas = _progressive_unit_fill(
            flows, unit_remaining, orig_capacity=orig_capacity,
            cumulative_terms=cumulative_terms)
        for index, alpha in enumerate(alphas):
            if alpha <= 0.0:
                continue
            key, _coupled, _limit, _terms, rates = flows[index]
            for name, demand in rates:
                target = result.setdefault(name, {})
                target[key] = target.get(key, 0.0) + alpha * demand
    # Independent all-tier guard: the exact sum of every committed term from
    # every priority tier may never exceed the original physical unit capacity.
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


def _residual_unit_rates(unit_remaining: list[float],
                         caps: Mapping[int, float],
                         placements: Mapping[int, tuple[int, ...]],
                         explicit: set[int] | None = None
                         ) -> dict[int, float]:
    """Max-min with a distinct residual capacity on each physical unit.

    Unplaced (fanned-out) tasks may consume only units that still have
    residual capacity.  When such a unit saturates, a fanned task redistributes
    over its remaining available units; an explicitly pinned task instead ends
    its allocation because it cannot borrow a sibling unit.
    """
    count = len(unit_remaining)
    explicit = set(explicit or ())
    tolerance = 1e-12
    result = dict.fromkeys(caps, 0.0)
    units: dict[int, list[int]] = {}
    for key in caps:
        declared = placements.get(key)
        if declared is None:
            units[key] = [unit for unit in range(count)
                          if unit_remaining[unit] > tolerance]
        else:
            owned = [unit for unit in declared if 0 <= unit < count]
            if any(unit_remaining[unit] <= tolerance for unit in owned):
                units[key] = []
            else:
                units[key] = owned
    active = {key for key in caps
              if caps[key] > 0.0 and units[key]}
    while active:
        slopes: dict[int, float] = {}
        for unit in range(count):
            if unit_remaining[unit] <= tolerance:
                continue
            slope = 0.0
            for key in active:
                if unit in units[key]:
                    slope += 1.0 / len(units[key])
            if slope > 0.0:
                slopes[unit] = slope
        if not slopes:
            break
        bounds = {unit: unit_remaining[unit] / slope
                  for unit, slope in slopes.items()}
        delta = min(min(bounds.values()),
                    min(caps[key] - result[key] for key in active))
        if delta <= tolerance or not math.isfinite(delta):
            break
        for key in active:
            result[key] += delta
        for unit, slope in slopes.items():
            unit_remaining[unit] = max(0.0, unit_remaining[unit] - delta * slope)
        saturated = {unit for unit, bound in bounds.items()
                     if bound <= delta + tolerance}
        for key in list(active):
            if result[key] >= caps[key] - tolerance:
                active.discard(key)
                continue
            hit = [unit for unit in saturated if unit in units[key]]
            if not hit:
                continue
            if key in explicit:
                active.discard(key)
            else:
                units[key] = [unit for unit in units[key]
                              if unit not in saturated]
                if not units[key]:
                    active.discard(key)
    return result


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
        unit_remaining = [float(resource.rate) / count] * count
        shares: dict[int, float] = {}

        def consume(key: int, rate: float, units: tuple[int, ...]) -> None:
            if not units:
                return
            share = rate / len(units)
            for unit in units:
                if 0 <= unit < count:
                    unit_remaining[unit] = max(0.0, unit_remaining[unit] - share)

        if resource.policy == "fifo":
            # Non-preemptive: the active holder is the unique owner.  Priority
            # only selects among tasks that have not started yet.
            eligible = {key: cap for key, cap in caps.items() if cap > 0.0}
            if eligible:
                winner = min(eligible, key=lambda key: (
                    tasks[key].start is None,
                    tasks[key].start if tasks[key].start is not None else math.inf,
                    tasks[key].priority, key))
                declared = tasks[winner].placement.get(name)
                placements = ({winner: declared} if declared is not None
                              else {winner: None})
                # ``_residual_unit_rates`` already debits the exact per-unit
                # residual; consuming the returned rate again double-charged
                # the physical owner.
                tier_shares = _residual_unit_rates(
                    unit_remaining, {winner: eligible[winner]}, placements,
                    {winner} if declared is not None else set())
                rate = tier_shares.get(winner, 0.0)
                if rate > 0.0:
                    shares[winner] = rate
            result[name] = {key: rate for key, rate in shares.items() if rate > 0.0}
            continue
        for _priority, tier_keys in _priority_tiers(caps, tasks).items():
            tier = {key: caps[key] for key in tier_keys if caps[key] > 0.0}
            remaining = sum(unit_remaining)
            if not tier or remaining <= 0.0:
                continue
            available = tuple(unit for unit in range(count)
                              if unit_remaining[unit] > 1e-12)
            if not available:
                break
            placements: dict[int, tuple[int, ...]] = {}
            explicit: set[int] = set()
            for key in tier:
                declared = tasks[key].placement.get(name)
                if declared is None:
                    placements[key] = available
                    continue
                owned = tuple(unit for unit in declared if 0 <= unit < count)
                if any(unit_remaining[unit] <= 1e-12 for unit in owned):
                    # Its exact owner is exhausted; it cannot borrow a peer's
                    # residual capacity in this tier.
                    continue
                placements[key] = owned
                explicit.add(key)
            if not placements:
                continue
            scoped = {key: tier[key] for key in placements}
            uniform = max(unit_remaining) - min(unit_remaining) <= 1e-9
            if uniform and not explicit:
                # Preserve the exact aggregate fair-fill arithmetic when every
                # unit is equally free and every contender fans out uniformly.
                tier_shares = fair_rates(remaining, scoped)
                debit = consume
            elif uniform:
                tier_shares = unit_rates(remaining, count, scoped, placements)
                debit = consume
            else:
                # The residual allocator debits each unit itself, exactly once.
                tier_shares = _residual_unit_rates(
                    unit_remaining, scoped, placements, explicit)
                debit = None
            for key, rate in tier_shares.items():
                if rate > 0.0:
                    shares[key] = shares.get(key, 0.0) + rate
                    if debit is not None:
                        debit(key, rate, placements.get(
                            key, tuple(range(count))))
        result[name] = {key: rate for key, rate in shares.items() if rate > 0.0}
    return result


