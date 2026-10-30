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
    coefficients, limits, consumers = {}, {}, {}
    for key, task in tasks.items():
        if not task.remaining:
            continue
        groups = [tuple(task.remaining)] if task.coupled else [
            (name,) for name in task.remaining]
        for index, names in enumerate(groups):
            flow = key, index
            scale = max(task.remaining[name] / resources[name].rate for name in names)
            demand = {name: task.remaining[name] / scale for name in names}
            coefficients[flow] = demand
            limit = math.inf
            for name, amount in demand.items():
                resource = resources[name]
                limit = min(limit, task.caps.get(name, math.inf) / amount)
                if name in fifo_first and fifo_first[name] != key:
                    limit = 0
                units = task.placement.get(name, tuple(range(resource.units)))
                for unit in units:
                    consumers.setdefault((name, unit), {})[flow] = amount / len(units)
            limits[flow] = limit

    remaining = {pair: resources[pair[0]].rate / resources[pair[0]].units
                 for pair in consumers}
    assigned = dict.fromkeys(coefficients, 0.0)
    active = {flow for flow in coefficients if limits[flow] > 0}
    while active:
        slopes = {pair: sum(value for flow, value in rows.items() if flow in active)
                  for pair, rows in consumers.items()}
        bounds = {pair: remaining[pair] / slope
                  for pair, slope in slopes.items() if slope}
        delta = min(min(bounds.values()), min(limits[flow] - assigned[flow]
                                             for flow in active))
        for flow in active:
            assigned[flow] += delta
        for pair, slope in slopes.items():
            remaining[pair] = max(0.0, remaining[pair] - delta * slope)
        saturated = {pair for pair, bound in bounds.items() if bound <= delta}
        # Use the existing resource-to-flow index once per saturated resource.
        # Testing every active flow against every saturated unit is quadratic
        # for multi-rank pipelines, although the exclusion set is identical.
        blocked = set()
        for pair in saturated:
            blocked.update(consumers[pair])
        active -= {flow for flow in active
                   if assigned[flow] >= limits[flow]
                   or flow in blocked}

    result = {}
    for (key, index), fraction in assigned.items():
        if fraction:
            for name, amount in coefficients[key, index].items():
                result.setdefault(name, {})[key] = amount * fraction
    return result
