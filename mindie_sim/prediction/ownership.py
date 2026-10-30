"""Physical unit constraints for shared resource service.

[Role] Allocate service to programs with explicit physical-unit ownership.
[Invariants] A group divides its work equally over its declared units. Idle
  units cannot supply a pinned program. Aggregate service is counted once.
[Timeouts & Locks] Finite progressive filling over active tasks and units.
[Side-Effect Policy] Pure arithmetic with bounded value-only memoization; no I/O.
"""
import math
from functools import lru_cache


def read_role(roles, aiv=None, *, prefer_generic=False):
    """Select an existing physical read pipe from the declared consumer owner."""
    if prefer_generic and "mte2" in roles:
        return "mte2"
    split = {"mte2_aic", "mte2_aiv"}.intersection(roles)
    if split and len(split) != 2:
        raise ValueError("mixed_read_topology_requires_both_aic_and_aiv_pipes")
    return ("mte2_aiv" if aiv is not None else "mte2_aic") if split else "mte2"


def unit_group(index, owners, units):
    """Map an owner's fixed physical partition to another resource's units."""
    if units >= owners and units % owners == 0:
        width = units // owners
        return tuple(range(index * width, (index + 1) * width))
    if owners % units == 0:
        return (index // (owners // units),)
    raise ValueError(f"resource_ownership_geometry_incompatible:{owners}:{units}")


def _partition_rates(rate, count, caps, placements):
    """Disjoint owner partitions are independent equal-weight resource pools."""
    groups, occupied = {}, set()
    for key in caps:
        units = placements.get(key) or tuple(range(count))
        group = groups.get(units)
        if group is not None:
            group.append(key)
        else:
            if any(unit in occupied for unit in units):
                return None
            occupied.update(units)
            groups[units] = [key]
    result = {}
    for units, keys in groups.items():
        capacity = rate / count * len(units)
        if len(keys) == 1:
            key = keys[0]
            result[key] = min(capacity, caps[key])
            continue
        ordered = sorted(keys, key=caps.__getitem__)
        for index, key in enumerate(ordered):
            assigned = min(caps[key], capacity / (len(ordered) - index))
            result[key] = assigned
            capacity -= assigned
    return result


@lru_cache(maxsize=512)
def _partition_template(rate, count, signature):
    """Cache pure ordinal arithmetic, never task IDs, work or Engine state.

    Keep source order for equal caps and resource-unit tuples. Only disjoint
    partitions are cached: overlap arbitration retains the original task keys
    because set iteration can affect floating-point reduction order.
    """
    caps = {index: row[0] for index, row in enumerate(signature)}
    placements = {index: row[1] for index, row in enumerate(signature)}
    result = _partition_rates(rate, count, caps, placements)
    return None if result is None else tuple(result.items())


def _cached_partition_rates(rate, count, caps, placements):
    # Bound retained signature size as well as entry count. Large dynamic
    # cohorts still use exactly the same partition arithmetic without retention.
    if len(caps) > 256:
        return _partition_rates(rate, count, caps, placements)
    keys = tuple(caps)
    signature = tuple((caps[key], placements.get(key) or ()) for key in keys)
    result = _partition_template(rate, count, signature)
    return None if result is None else {keys[index]: value for index, value in result}


def unit_rates(rate, count, caps, placements):
    """Max-min task rates subject to per-unit capacities and task supply caps."""
    partitioned = _cached_partition_rates(rate, count, caps, placements)
    if partitioned is not None:
        return partitioned
    weights = {}
    for key in caps:
        units = placements.get(key) or tuple(range(count))
        for unit in units:
            weights.setdefault(unit, {})[key] = 1 / len(units)
    remaining = dict.fromkeys(weights, rate / count)
    result = dict.fromkeys(caps, 0.0)
    active = set(caps)
    while active:
        slopes = {unit: sum(weight for key, weight in consumers.items() if key in active)
                  for unit, consumers in weights.items()}
        bounds = {unit: remaining[unit] / slope
                  for unit, slope in slopes.items() if slope}
        delta = min(min(bounds.values(), default=math.inf),
                    min(caps[key] - result[key] for key in active))
        for key in active:
            result[key] += delta
        for unit, slope in slopes.items():
            remaining[unit] = max(0, remaining[unit] - delta * slope)
        saturated = {unit for unit, bound in bounds.items() if bound <= delta}
        active -= {key for key in active
                   if result[key] >= caps[key]
                   or any(key in weights[unit] for unit in saturated)}
    return result
