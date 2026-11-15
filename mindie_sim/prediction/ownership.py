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


@lru_cache(maxsize=4096)
def unit_group(index, owners, units):
    """Map an owner's fixed physical partition to another resource's units."""
    if units >= owners and units % owners == 0:
        width = units // owners
        return tuple(range(index * width, (index + 1) * width))
    if owners % units == 0:
        return (index // (owners // units),)
    raise ValueError(f"resource_ownership_geometry_incompatible:{owners}:{units}")


@lru_cache(maxsize=None)
def _full_units(count):
    return tuple(range(count))


_ZERO_MASK = 0; _ONE_BIT = 1


@lru_cache(maxsize=None)
def _unit_mask(units):
    mask = _ZERO_MASK
    for unit in units:
        mask |= _ONE_BIT << unit
    return mask


def _partition_rates(rate, count, caps, placements):
    """Disjoint owner partitions are independent equal-weight resource pools."""
    groups = {}
    occupied = _ZERO_MASK
    full_u = None
    full_m = _ZERO_MASK
    for key in caps:
        units = placements.get(key)
        if not units:
            if full_u is None:
                full_u = _full_units(count)
                full_m = (_ONE_BIT << count) - _ONE_BIT
            units = full_u
            u_mask = full_m
        else:
            u_mask = _unit_mask(units)
        group = groups.get(units)
        if group is not None:
            group.append(key)
        else:
            if occupied & u_mask:
                return None
            occupied |= u_mask
            groups[units] = [key]
    result = {}
    unit_rate = rate / count
    for units, keys in groups.items():
        capacity = unit_rate * len(units)
        if len(keys) == _ONE_BIT:
            key = keys[0]
            cap = caps[key]
            result[key] = capacity if capacity < cap else cap
            continue
        ordered = sorted(keys, key=caps.__getitem__)
        n_ord = len(ordered)
        for index, key in enumerate(ordered):
            cap = caps[key]
            share = capacity / (n_ord - index)
            assigned = cap if cap < share else share
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
    partitioned = _partition_rates(rate, count, caps, placements)
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
