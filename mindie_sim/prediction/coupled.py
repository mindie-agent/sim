"""Capacity sharing for resource-coupled steady-state progress.

[Role] Assign service to finite fluid pipelines on the existing Engine.
[Invariants] A coupled task consumes every resource in a fixed proportion.
  Dominant-resource progressive filling preserves physical owner capacities,
  supply caps and FIFO eligibility. Uncoupled-only scheduling is unchanged.
[Timeouts & Locks] Finite filling over active flows and resources; no I/O.
[Side-Effect Policy] Pure allocation arithmetic; no clock or work mutation.
"""
import math


def coupled_rates(resources, tasks):
    """Return work/ms using normalized dominant-resource max-min fairness.

    Ordinary task/resource pairs remain independent flows. Coupled tasks are
    one flow each, normalized by their largest aggregate-resource service time.
    This explicitly models fluid backpressure, not discrete tile availability.
    """
    fifo_resources = {name for name, res in resources.items() if res.policy == "fifo"}
    fifo_first = {}
    if fifo_resources:
        for key, task in tasks.items():
            for name in task.remaining:
                if name in fifo_resources:
                    fifo_first[name] = min(key, fifo_first.get(name, key))
    # Dense ordinals are local to this allocation. They avoid repeatedly
    # hashing (task, component) and (resource, owner) in progressive filling;
    # no completed task, prior allocation or clock state is retained.
    keys, coefficients, limits, owners = [], [], [], []
    consumers, capacities, members, default_units = [], [], {}, {}
    for key, task in tasks.items():
        rem = task.remaining
        if not rem:
            continue
        groups = (tuple(rem),) if task.coupled else tuple((name,) for name in rem)
        caps = task.caps
        placement = task.placement
        for names in groups:
            flow = len(keys)
            keys.append(key)
            owners.append([])
            if len(names) == 1:
                n0 = names[0]
                scale = rem[n0] / resources[n0].rate
                demand = {n0: rem[n0] / scale}
            elif len(names) == 2:
                n0, n1 = names
                s0 = rem[n0] / resources[n0].rate
                s1 = rem[n1] / resources[n1].rate
                scale = s0 if s0 >= s1 else s1
                demand = {n0: rem[n0] / scale, n1: rem[n1] / scale}
            else:
                scale = max([rem[name] / resources[name].rate for name in names])
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
                if units is None:
                    units = default_units.get(name)
                    if units is None:
                        units = default_units[name] = tuple(range(resource.units))
                per_unit = amount if len(units) == 1 else amount / len(units)
                if name in members:
                    members[name].append((flow, per_unit, units))
                else:
                    members[name] = [(flow, per_unit, units)]
            limits.append(limit)

    for name, rows in members.items():
        capacity = resources[name].rate / resources[name].units
        if len(rows) == 1:
            flow, amount, _ = rows[0]
            if amount:
                limits[flow] = min(limits[flow], capacity / amount)
            continue
        common = rows[0][2]
        if all(units == common for _, _, units in rows):
            # Identical placement gives every participating owner the exact
            # same constraint. One representative preserves its per-unit
            # capacity; it neither pools idle units nor changes contention.
            groups = [tuple((flow, amount) for flow, amount, _ in rows)]
        else:
            total_u = 0
            seen = set()
            for _, _, units in rows:
                total_u += len(units)
                seen.update(units)
            if len(seen) == total_u:
                for flow, amount, _ in rows:
                    if amount:
                        limits[flow] = min(limits[flow], capacity / amount)
                continue
            by_unit = {}
            for flow, amount, units in rows:
                for unit in units:
                    by_unit.setdefault(unit, []).append((flow, amount))
            groups = by_unit.values()
        for rows in groups:
            if len(rows) == 1:
                flow, amount = rows[0]
                if amount:
                    limits[flow] = min(limits[flow], capacity / amount)
                continue
            slot = len(consumers)
            consumers.append(rows)
            capacities.append(capacity)
            for flow, _ in rows:
                owners[flow].append(slot)

    if not consumers:
        result = {}
        for flow, limit in enumerate(limits):
            if limit > 0 and limit < math.inf:
                for name, amount in coefficients[flow].items():
                    result.setdefault(name, {})[keys[flow]] = amount * limit
        return result

    remaining = capacities
    assigned = [0.0] * len(keys)
    active = [limit > 0 for limit in limits]
    pending = [flow for flow, enabled in enumerate(active) if enabled]
    slopes = [sum(value for flow, value in rows if active[flow])
              for rows in consumers]
    while pending:
        # Keep each owner's insertion order and sum from zero, preserving
        # floating-point arithmetic as flows retire. Subtracting retired
        # coefficients from a cached sum would introduce cancellation drift.
        bounds = [amount / slope if slope else math.inf
                  for amount, slope in zip(remaining, slopes)]
        delta = min(min(bounds), min([limits[flow] - assigned[flow]
                                      for flow in pending]))
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
                slopes[slot] = sum([value for flow, value in consumers[slot] if active[flow]])

    result = {}
    for flow, fraction in enumerate(assigned):
        if fraction:
            for name, amount in coefficients[flow].items():
                result.setdefault(name, {})[keys[flow]] = amount * fraction
    return result
