"""Explicit device-invocation parameters bound at physical kernel boundaries.

[Role] Select declared startup and completion constraints by implementation axes.
[Invariants] No model-label dispatch or inferred residual parameters. Logical
  joins and nested resource phases do not acquire device startup. Host launch
  and transport propagation keep their separate causal boundaries.
[Timeouts & Locks] Linear in the finite user-declared rule list; no I/O.
[Side-Effect Policy] Returns timing plus provenance, without changing work.
"""
from dataclasses import replace
from copy import deepcopy
import hashlib
import json

from mindie_sim.schema.common import ModelingError
from .invocation import InvocationProgram, InvocationTiming
from .services import ResourceService


class InvocationPolicy:
    def __init__(self, rules=()):
        self.rules = []
        allowed = {"hardware", "runtime", "revision", "format", "kernel", "implementation",
                   "dtype", "layout", "replay_scope", "recipe", "family", "phase"}
        for raw in rules:
            match = dict(raw.get("match", {}))
            if match.keys() - allowed:
                raise ModelingError("invocation_unknown_identity_axes:" +
                                    ",".join(sorted(match.keys() - allowed)))
            timing = InvocationTiming.from_mapping({k: v for k, v in raw.items() if k != "match"})
            payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            self.rules.append((match, timing, hashlib.sha256(payload.encode()).hexdigest()))

    def select(self, context, recipe, *, devices=()):
        op = context.op
        metadata = dict(op.metadata)
        explicit_class = metadata.get("invocation_class") or metadata.get(
            "program_execution", {}).get("execution_class")
        # A receive-only rank still enters an explicitly declared native
        # group. It may match a kernel rule even without locally charged work;
        # absent that rule, hardware class selection still requires evidence.
        if (context.service_boundary != "device_kernel"
                or (not context.work and not getattr(recipe, "algorithm", None)
                    and not (op.source_boundary == "kernel" and explicit_class)
                    and not metadata.get("device_kernel_group"))
                or op.family in {"barrier", "token_commit"}):
            return None
        identity = {**context.service_identity,
                    "hardware": context.service_identity["hardware"]["key"],
                    "recipe": op.recipe, "family": op.family, "phase": op.phase}
        matches = [(len(match), timing, key, match) for match, timing, key in self.rules
                   if all(identity.get(axis) == value for axis, value in match.items())]
        if not matches:
            selected = hardware_invocation(context, devices, explicit_class=explicit_class)
            if selected is None:
                return None
            timing, record = selected
        else:
            specificity = max(row[0] for row in matches)
            selected = [row for row in matches if row[0] == specificity]
            if len(selected) != 1:
                raise ModelingError("invocation_ambiguous_matching_rules")
            _, timing, key, match = selected[0]
            record = {"id": key, "match": match, **timing.to_dict()}
        if (isinstance(recipe, ResourceService) and timing.startup_ms
                and recipe.model.estimate(context.native_work.get(recipe.model.resource, 0))
                is not None):
            # An inclusive response already owns its invocation startup. A
            # second opaque device startup cannot be separated from that fit.
            raise ModelingError("invocation_startup_duplicates_resource_service_boundary")
        return timing, record


_HW_INV_CACHE = {}


def hardware_invocation(context, devices, *, explicit_class=None):
    """An enclosing physical kernel pays its declared execution classes once."""
    work = context.native_work
    if explicit_class:
        classes = (explicit_class,)
    elif work.get("link", 0) or context.op.family == "collective":
        # Communication libraries and pure DMA do not imply an AIV kernel.
        return None
    else:
        classes = tuple(role for role in ("cube", "vector") if work.get(role, 0))
    cache_key = (
        tuple(
            (device.rank, tuple((role, timing._cache_sig)
                                for role, timing in device.kernel_invocations))
            for device in devices
        ),
        classes,
    )
    if cache_key in _HW_INV_CACHE:
        return _HW_INV_CACHE[cache_key]
    selected = [
        {"rank": device.rank, "class": role, **timing.to_dict()}
        for device in devices for role, timing in device.kernel_invocations if role in classes
    ]
    if not selected:
        if len(_HW_INV_CACHE) >= 1024:
            _HW_INV_CACHE.pop(next(iter(_HW_INV_CACHE)))
        _HW_INV_CACHE[cache_key] = None
        return None
    # A representative TP/PCP shard conservatively uses its slowest member.
    # Mixed cube/vector is one invocation, not two stacked dispatch delays.
    timing = InvocationTiming(
        startup_ms=max(row["startup_ms"] for row in selected),
        minimum_ms=max(row["minimum_ms"] for row in selected),
        provenance={"basis": "hardware_execution_class", "declarations": selected},
    )
    payload = json.dumps(timing.to_dict(), sort_keys=True, separators=(",", ":"))
    res = timing, {
        "id": hashlib.sha256(payload.encode()).hexdigest(),
        "match": {"execution_classes": classes}, "source": "device.kernel_invocations",
        **timing.to_dict(),
    }
    if len(_HW_INV_CACHE) >= 1024:
        _HW_INV_CACHE.pop(next(iter(_HW_INV_CACHE)))
    _HW_INV_CACHE[cache_key] = res
    return res


def lower_invocation(recipe, context):
    timing = context.invocation_timing
    if timing is None:
        return recipe.lower(context)
    body = recipe.lower(replace(context, minimum_ms=0, invocation_timing=None))
    if context.minimum_ms > timing.minimum_ms:
        timing = replace(timing, minimum_ms=context.minimum_ms)
    return InvocationProgram(body, timing)
