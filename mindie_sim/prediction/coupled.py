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
    fifo_first = {}
    for key, task in tasks.items():
        for name in task.remaining:
            if resources[name].policy == "fifo":
                fifo_first[name] = min(key, fifo_first.get(name, key))
    # Dense ordinals are local to this allocation. They avoid repeatedly
    # hashing (task, component) and (resource, owner) in progressive filling;
    # no completed task, prior allocation or clock state is retained.
    keys, coefficients, limits, owners = [], [], [], []
    consumers, capacities, members, default_units = [], [], {}, {}
    for key, task in tasks.items():
        if not task.remaining:
            continue
        groups = [tuple(task.remaining)] if task.coupled else [
            (name,) for name in task.remaining]
        for names in groups:
            flow = len(keys)
            keys.append(key)
            owners.append([])
            scale = (task.remaining[names[0]] / resources[names[0]].rate if len(names) == 1
                     else max(task.remaining[name] / resources[name].rate for name in names))
            demand = {name: task.remaining[name] / scale for name in names}
            coefficients.append(demand)
            limit = math.inf
            for name, amount in demand.items():
                resource = resources[name]
                cap = task.caps.get(name)
                if cap is not None:
                    limit = min(limit, cap / amount)
                if name in fifo_first and fifo_first[name] != key:
                    limit = 0
                units = task.placement.get(name)
                if units is None:
                    if name not in default_units:
                        default_units[name] = tuple(range(resource.units))
                    units = default_units[name]
                per_unit = amount / len(units)
                members.setdefault(name, []).append((flow, per_unit, units))
            limits.append(limit)

    for name, rows in members.items():
        common = rows[0][2]
        if all(units == common for _, _, units in rows):
            # Identical placement gives every participating owner the exact
            # same constraint. One representative preserves its per-unit
            # capacity; it neither pools idle units nor changes contention.
            groups = [tuple((flow, amount) for flow, amount, _ in rows)]
        else:
            by_unit = {}
            for flow, amount, units in rows:
                for unit in units:
                    by_unit.setdefault(unit, []).append((flow, amount))
            groups = by_unit.values()
        capacity = resources[name].rate / resources[name].units
        for rows in groups:
            slot = len(consumers)
            consumers.append(rows)
            capacities.append(capacity)
            for flow, _ in rows:
                owners[flow].append(slot)

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
        delta = min(min(bounds), min(limits[flow] - assigned[flow]
                                     for flow in pending))
        for flow in pending:
            assigned[flow] += delta
        for slot, slope in enumerate(slopes):
            if slope:
                remaining[slot] = max(0.0, remaining[slot] - delta * slope)
            if bounds[slot] <= delta:
                for flow, _ in consumers[slot]:
                    active[flow] = False
        survivors, changed = [], set()
        for flow in pending:
            if active[flow] and assigned[flow] < limits[flow]:
                survivors.append(flow)
            else:
                active[flow] = False
                changed.update(owners[flow])
        pending = survivors
        if pending:
            for slot in changed:
                slopes[slot] = sum(value for flow, value in consumers[slot] if active[flow])

    result = {}
    for flow, fraction in enumerate(assigned):
        if fraction:
            for name, amount in coefficients[flow].items():
                result.setdefault(name, {})[keys[flow]] = amount * fraction
    return result
