"""Communication curve engine binding and diagnostic regime projection.

[Role] Bind calibrated communication curves to engine service phases and emit
  source-bearing diagnostic ``service_regimes`` rules while preserving measured
  phase boundaries and physical byte coordinates.
[Invariants] Physical link work is never deleted, shrunk, or clamped. Captured
  per-rank payload knots convert to wire coordinates only through an explicit
  per-collective factor, and only matching hardware identities may bind. A bound
  measurement keeps its measured phase separate from the consumer role and is
  never relabeled as a per-hop link price.
[Timeouts & Locks] Pure in-memory binding; no timers, locks, or subprocesses.
[Side-Effect Policy] Pure computation; never launches a device, writes
  calibration, or mutates the primitive store.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Mapping

from mindie_sim.prediction.services import ServiceCurve, service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from .golden import A5_HARDWARE_KEY
from .normalization import (
    INTRA_NODE_MAX_RANKS,
    _collective,
    _dtype,
    _explicit_pair,
    _explicit_scope,
    _require_a5,
)
from .normalized import (
    NormalizedServiceStage,
    build_normalized_service_recipe,
    build_normalized_service_stage,
    normalized_service_duration_ms,
    service_curve_from_normalized,
)
from .streaming import (
    _fold_timings,
    _iter_json_events,
    _read_stream_value,
    _stream_json,
    _weighted_median,
)

__all__ = [
    "_A5CommPlan",
    "_stream_json",
    "_wire_factor",
    "bind_service_curve",
    "build_a5_comm_recipe",
    "build_normalized_service_recipe",
    "build_normalized_service_stage",
    "normalized_service_duration_ms",
    "NormalizedServiceStage",
    "service_curve_from_normalized",
    "to_service_regime",
]

_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_BYTE_RESOURCES = frozenset({"link", "hbm", "mte1", "mte2", "mte3", "fixpipe"})
_SUPPORTED_COLLECTIVES = frozenset({
    "all_reduce", "reduce_scatter", "all_gather", "all_to_all", "all_to_allv",
    "p2p_send_recv",
})
_COLLECTIVE_DIRECTIONS = {
    "all_gather": "read", "all_reduce": "send", "reduce_scatter": "send",
    "all_to_all": "send", "all_to_allv": "send", "p2p_send_recv": "directed",
}


def __getattr__(name: str):
    if name in {"_A5CommPlan", "build_a5_comm_recipe"}:
        from . import recipes as _recipes
        value = getattr(_recipes, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _wire_factor(collective, world_size, curve=None):
    """Captured payload per rank -> the declared collective's physical work.

    This is a coordinate conversion for the ring/tree recipe, not a hop price.
    A whole-collective completion envelope is never converted into a per-hop
    link rate by this factor; its provenance records
    ``link_transit_identified=False``.
    """
    rank_count = max(1, int(world_size or 1))
    name = _collective(collective)
    if rank_count <= 1:
        return 1.0
    if name in {"all_gather", "reduce_scatter"}:
        # The worker's payload_bytes is per-rank ag_in / rs_out while the model
        # accounts the complete ring traffic, (r - 1) * local_bytes.
        return float(rank_count - 1)
    if name == "all_reduce":
        return 2.0 * (rank_count - 1) / rank_count
    if name == "all_to_all":
        return float(rank_count - 1) / rank_count
    if name == "all_to_allv":
        provenance = getattr(curve, "provenance", None) or {}
        identity = getattr(curve, "identity", None) or {}
        if (dict(provenance).get("derived_from") == "all_to_all"
                or dict(identity).get("derived_from") == "all_to_all"):
            return float(rank_count - 1) / rank_count
        # A measured mc2-quant comm-lane row already recorded link_bytes.
        return 1.0
    return 1.0


def _fit_bundle(ingested=None):
    from .calibration import fit_a5_service_curves
    return fit_a5_service_curves(ingested)


def _resolve_bundle(bundle_or_case, case_or_op, op):
    if op is None:
        bundle, case, operation = _fit_bundle(), bundle_or_case, case_or_op
    else:
        bundle, case, operation = bundle_or_case, case_or_op, op
    if not hasattr(bundle, "lookup_curve"):
        bundle = _fit_bundle(bundle)
    return bundle, case, operation


def _validate_hardware(source, *, identity=False):
    source = dict(source or {})
    if identity:
        keys = (source.get("key"), source.get("sku"))
        architecture = source.get("architecture")
    else:
        keys = (source.get("hardware_key"), source.get("sku"))
        architecture = source.get("arch_family")
    for key in keys:
        if key or architecture:
            _require_a5(key or A5_HARDWARE_KEY, architecture)


def _require_case_a5(parsed):
    machine = parsed.machine
    _require_a5(machine.hardware_key or machine.sku, machine.arch_family)
    if machine.hardware_key:
        _require_a5(machine.hardware_key, machine.arch_family)
    if machine.sku:
        _require_a5(machine.sku, machine.arch_family)


def _world_size(case, op):
    metadata = dict(op.metadata)
    for key in ("world_size", "group_size", "ranks", "group_ranks"):
        if metadata.get(key):
            return int(metadata[key])
    axis = metadata.get("group_axis")
    parallel = case.runtime.parallel
    if axis and hasattr(parallel, axis):
        return int(getattr(parallel, axis))
    return int(parallel.tp * parallel.cp or len(case.machine.dies) or 1)


def _placement_for(case, op, collective, world_size, pair):
    del collective
    metadata = dict(op.metadata)
    if "placement_scope" in metadata and metadata["placement_scope"] is not None:
        explicit = _explicit_scope(metadata["placement_scope"])
        if explicit is None:
            raise ModelingError(
                f"a5_placement_scope_unknown:{metadata['placement_scope']}")
        return explicit
    if pair == "0->24":
        return "cross_node_urma_far"
    if pair == "0->1":
        return "intra_node_ub"
    if pair and int(world_size) > INTRA_NODE_MAX_RANKS:
        return "cross_node_urma"
    if int(world_size) > INTRA_NODE_MAX_RANKS:
        return "cross_node_urma"
    if case is not None:
        for link in case.machine.links:
            if link.members and max(link.members) >= INTRA_NODE_MAX_RANKS:
                return "cross_node_urma"
    return "intra_node_ub"


def _resource_for(op, metadata, collective):
    explicit = metadata.get("a5_service_resource") or metadata.get("service_resource")
    if explicit:
        return str(explicit)
    if op.link_bytes or collective in _SUPPORTED_COLLECTIVES:
        return "link"
    return "hbm" if op.hbm_bytes else "link"


def _measurement_boundary(curve, collective):
    """Preserve the measured phase; never relabel it as another boundary."""
    declared = curve.provenance.get("measurement_boundary")
    descriptor = dict(declared) if isinstance(declared, Mapping) else {}
    semantics = curve.provenance.get("service_semantics")
    if not descriptor:
        if semantics == "aicpu_child_collective_phase" or curve.resource == "aicpu_child":
            phase, complete = "aicpu_child", False
        elif curve.boundary == "resource_phase" or curve.resource == "collective_phase":
            phase, complete = "collective_phase", False
        elif collective == "p2p_send_recv":
            phase, complete = "directed_p2p_call", False
        else:
            phase, complete = "collective_call", True
        descriptor = {
            "kind": "native_target_call" if phase == "aicpu_child" else "device_call",
            "phase": phase, "resource": curve.resource,
            "work_coordinate": ("per_rank_input_bytes" if phase == "aicpu_child"
                                else "captured_per_rank_payload"),
            "duration_unit": "ms",
            "source_kind": ("native_ai_cpu_child" if phase == "aicpu_child"
                            else "campaign_device_call"),
            "service_semantics": semantics or (
                "aicpu_child_collective_phase" if phase == "aicpu_child" else
                "directed_p2p_call" if phase == "directed_p2p_call" else
                "collective_phase_service" if phase == "collective_phase" else
                "whole_collective_call_envelope"),
            "complete_collective_call": complete,
            "complete_directed_call": phase == "directed_p2p_call",
            "link_transit_identified": False, "hop_scaling": "not_derived",
            "algorithm_identified": False,
            "direction_observed": phase == "directed_p2p_call",
        }
    descriptor.setdefault("complete_collective_call", False)
    descriptor.setdefault("complete_directed_call", False)
    descriptor.setdefault("link_transit_identified", False)
    descriptor.setdefault("hop_scaling", "not_derived")
    descriptor.setdefault("direction_observed", False)
    descriptor.setdefault("algorithm_identified", False)
    descriptor.setdefault("work_coordinate", "captured_per_rank_payload")
    descriptor.setdefault("phase", "unknown")
    descriptor.setdefault("service_semantics", semantics)
    return descriptor


def _consumer_role(descriptor):
    phase = descriptor.get("phase")
    if phase == "aicpu_child":
        return "aicpu_child_completion_floor"
    if phase == "collective_phase":
        return "resource_phase_envelope"
    if phase == "directed_p2p_call":
        return "directed_p2p_call_envelope"
    if phase == "collective_call":
        return "collective_call_completion_envelope"
    return "measured_call_envelope"


def _require_complete_call(descriptor, required):
    if required and not descriptor.get("complete_collective_call"):
        raise ModelingError(
            "a5_comm_measurement_boundary_cannot_serve_complete_collective_call:"
            + str(descriptor.get("phase") or "unknown"))


def _bind_to_bundle(bundle, case, op, scope="none", *, device=None,
                    resource=None, boundary=None, active_units=None,
                    require_complete_call=False, production=False,
                    **lookup_kwargs):
    if production:
        raise ModelingError("measured_a5_call_binding_is_diagnostic_only")
    parsed = (case if hasattr(case, "machine") and not isinstance(case, Mapping)
              else parse_case(case))
    _require_case_a5(parsed)
    metadata = dict(op.metadata)
    selectors = dict(lookup_kwargs)
    explicit = {key: selectors.get(key) for key in (
        "world_size", "expansion_mode", "placement_scope", "pair",
        "collective", "dtype", "stack", "cann", "algorithm",
        "protocol") if selectors.get(key) is not None}
    hardware_key = selectors.pop("hardware_key", None)
    sku = selectors.pop("sku", None)
    arch_family = selectors.pop("arch_family", None) or parsed.machine.arch_family
    _validate_hardware({"hardware_key": hardware_key, "sku": sku,
                        "arch_family": arch_family})
    hardware_key = (hardware_key or sku or parsed.machine.hardware_key
                    or parsed.machine.sku)
    placement = selectors.pop("placement_scope", None)
    world = selectors.pop("world_size", None) or _world_size(parsed, op)
    mode = selectors.pop("expansion_mode", None) or metadata.get(
        "expansion_mode", metadata.get("expansion_regime", "unspecified"))
    dtype = selectors.pop("dtype", None) or _dtype(
        op.dtype or parsed.model.compute_dtype)
    pair = selectors.pop("pair", None) or _explicit_pair(metadata.get("p2p_pair"))
    collective = _collective(
        selectors.pop("collective", None) or metadata.get("collective")
        or metadata.get("a5_collective") or op.family)
    if collective not in _SUPPORTED_COLLECTIVES:
        collective = _collective(metadata.get("kernel") or op.recipe)
    if collective not in _SUPPORTED_COLLECTIVES:
        raise ModelingError(f"a5_curve_unavailable:{collective}")
    placement = placement or _placement_for(parsed, op, collective, world, pair)
    boundary = (boundary or selectors.pop("boundary", None)
                or ("resource_phase" if metadata.get("device_kernel_group")
                    else "device_kernel"))
    for key in ("stack", "cann", "algorithm", "protocol"):
        if selectors.get(key) is None and metadata.get(key) is not None:
            selectors[key] = metadata[key]
    curve = bundle.lookup_curve(
        collective, world, placement_scope=placement, dtype=dtype,
        expansion_mode=mode, pair=pair, hardware_key=hardware_key,
        arch_family=arch_family, boundary=boundary, **selectors)
    if device is None:
        device = parsed.machine.dies[
            op.stage * parsed.runtime.parallel.tp * parsed.runtime.parallel.pcp]
    bound_metadata = dict(metadata)
    if "world_size" in explicit:
        bound_metadata["world_size"] = int(world)
        for alias in ("group_size", "ranks", "group_ranks", "steps"):
            bound_metadata.pop(alias, None)
    if "expansion_mode" in explicit:
        bound_metadata["expansion_mode"] = mode
        if "expansion_regime" in bound_metadata:
            bound_metadata["expansion_regime"] = mode
    for k, v in (("placement_scope", placement), ("pair", pair),
                 ("collective", collective)):
        if k in explicit:
            bound_metadata["p2p_pair" if k == "pair" else k] = v
    if "collective" in explicit and "a5_collective" in bound_metadata:
        bound_metadata["a5_collective"] = collective
    for key in ("algorithm", "protocol", "stack", "cann"):
        if key in explicit:
            bound_metadata[key] = selectors[key]
    bound_op = op
    if bound_metadata != metadata:
        bound_op = replace(op, metadata=tuple(sorted(bound_metadata.items())))
    if "dtype" in explicit and dtype != bound_op.dtype:
        bound_op = replace(bound_op, dtype=dtype)
    identity = service_identity(parsed, bound_op, scope, device=device)
    resource = resource or _resource_for(op, metadata, collective)
    if resource not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{resource}")
    unit = "bytes"
    units = active_units or device.units(resource)
    measurement = _measurement_boundary(curve, collective)
    _require_complete_call(measurement, require_complete_call)
    wire_factor = _wire_factor(collective, world, curve) if resource == "link" else 1.0
    knots = curve.knots
    source_semantics = (curve.provenance.get("service_semantics")
                        or measurement.get("service_semantics")
                        or ("directed_p2p_call" if collective == "p2p_send_recv"
                            else "whole_collective_call_envelope"))
    provenance = {
        **curve.provenance, "bound_a5_curve": curve.model_id,
        "boundary": boundary, "bound_stack": curve.identity.get("stack"),
        "bound_metadata": dict(bound_metadata),
        "measurement_boundary": measurement,
        "measured_resource": measurement.get("resource"),
        "consumption_resource": resource,
        "source_service_semantics": source_semantics,
        "service_semantics": source_semantics,
        "consumer_role": _consumer_role(measurement),
        "complete_collective_call_response": bool(
            measurement.get("complete_collective_call")),
        "complete_directed_call_response": bool(
            measurement.get("complete_directed_call")),
        "direction": curve.provenance.get(
            "direction", _COLLECTIVE_DIRECTIONS.get(collective, "unspecified")),
        "direction_observed": bool(measurement.get("direction_observed", False)),
        "algorithm_identified": bool(measurement.get("algorithm_identified", False)),
        "shared_domain": curve.provenance.get("shared_domain"),
        "transport_window": curve.provenance.get("transport_window"),
        "work_scope": measurement.get(
            "work_coordinate", "captured_per_rank_payload"),
        # A measured phase/call is a completion envelope, not a per-hop transit
        # price and not a licence to compose an invented hop count.
        "link_transit_identified": bool(
            measurement.get("link_transit_identified", False)),
        "hop_scaling": measurement.get("hop_scaling", "not_derived"),
    }
    if wire_factor != 1.0:
        knots = tuple((float(work) * wire_factor, float(duration))
                      for work, duration in curve.knots)
        provenance["raw_knots"] = curve.knots
        provenance["wire_factor"] = wire_factor
        provenance["wire_coordinate_semantics"] = (
            "captured per-rank payload converted to the declared collective's "
            "physical work coordinate; this is not a hop price")
    return ServiceCurve(resource, unit, identity, boundary, units,
                        curve.startup_ms, knots, provenance)


def bind_service_curve(bundle_or_case, case_or_op, op=None, scope="none",
                       production=False, **kwargs):
    """Bind one calibrated curve to a case/operator; the bundle argument is optional."""
    bundle, case, operation = _resolve_bundle(bundle_or_case, case_or_op, op)
    return _bind_to_bundle(bundle, case, operation, scope,
                           production=production, **kwargs)


def _regime_from_bundle(bundle, case, op, scope="none", *, regime_id=None,
                        coordinate="link", outside="analytic",
                        require_complete_call=False, production=False,
                        **lookup_kwargs):
    if production:
        raise ModelingError("to_service_regime_is_diagnostic_only")
    if coordinate not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{coordinate}")
    if (lookup_kwargs.get("boundary") not in (None, "device_kernel")
            or dict(op.metadata).get("device_kernel_group")):
        raise ModelingError("service_regime_requires_device_kernel_boundary")
    curve = _bind_to_bundle(bundle, case, op, scope,
                            require_complete_call=require_complete_call,
                            **lookup_kwargs)
    if curve.boundary != "device_kernel":
        raise ModelingError("service_regime_requires_device_kernel_boundary")
    identity = curve.identity
    match = {"hardware": identity["hardware"]["key"], "family": op.family,
             **{key: identity[key] for key in ("runtime", "revision", "format",
                                               "dtype", "layout", "replay_scope",
                                               "implementation")}}
    if "transport" in identity:
        match["transport"] = deepcopy(identity["transport"])
    if op.recipe:
        match["recipe"] = op.recipe
    else:
        match["kernel"] = identity["kernel"]
    metadata = dict(curve.provenance.get("bound_metadata") or op.metadata)
    for key in ("collective", "a5_collective", "group_axis", "steps", "world_size",
                "group_size", "ranks", "group_ranks", "placement_scope",
                "p2p_pair", "expansion_mode", "expansion_regime",
                "algorithm", "protocol", "stack", "cann"):
        if key in metadata:
            match[f"metadata.{key}"] = metadata[key]
    raw_knots = curve.provenance.get("raw_knots") or curve.knots
    wire_factor = float(curve.provenance.get("wire_factor") or 1.0)
    knots = [[float(work), float(duration * 1000.0)]
             for work, duration in raw_knots]
    coord_spec = {"path": f"work.{coordinate}"}
    if coordinate == "link" and wire_factor != 1.0:
        coord_spec["scale"] = 1.0 / wire_factor
    return {
        "id": regime_id or (f"a5_comm:{op.family}:{op.recipe or identity['kernel']}:"
                            f"{curve.model_id[:12]}"),
        "match": match,
        "coordinates": {"bytes": coord_spec},
        "domain": {"bytes": {"min": knots[0][0], "max": knots[-1][0]}},
        "response": {"kind": "piecewise_linear", "coordinate": "bytes",
                     "knots": knots, "outside": outside},
        "provenance": {
            "boundary": ("measured A5 "
                         + str(curve.provenance.get("measurement_boundary", {}).get(
                             "phase", "unknown")) + " response"),
            # T20 C07: this rule is a source-bearing diagnostic projection of
            # one measured curve.  Ordinary production pricing uses physical
            # work + normalized services, not an injected old whole-call rule.
            "diagnostic_only": True,
            "source_bearing": True,
            "production_eligible": False,
            "model_id": curve.model_id,
            "stack": curve.provenance.get("bound_stack"),
            "measurement_boundary": curve.provenance.get("measurement_boundary"),
            "measured_resource": curve.provenance.get("measured_resource"),
            "consumption_resource": curve.provenance.get("consumption_resource"),
            "source_service_semantics": curve.provenance.get(
                "source_service_semantics"),
            "service_semantics": curve.provenance.get("service_semantics"),
            "consumer_role": curve.provenance.get("consumer_role"),
            "complete_collective_call_response": bool(curve.provenance.get(
                "complete_collective_call_response", False)),
            "complete_directed_call_response": bool(curve.provenance.get(
                "complete_directed_call_response", False)),
            "direction": curve.provenance.get("direction"),
            "direction_observed": bool(curve.provenance.get(
                "direction_observed", False)),
            "direction_basis": curve.provenance.get("direction_basis"),
            "algorithm_identified": bool(curve.provenance.get(
                "algorithm_identified", False)),
            "shared_domain": curve.provenance.get("shared_domain"),
            "transport_window": curve.provenance.get("transport_window"),
            "work_scope": curve.provenance.get("work_scope"),
            "link_transit_identified": bool(curve.provenance.get(
                "link_transit_identified", False)),
            "hop_scaling": curve.provenance.get("hop_scaling", "not_derived"),
            "coordinate_basis": ("captured_per_rank_payload_not_link_transit"
                                 if coordinate == "link" else
                                 f"captured_{coordinate}_work"),
        },
    }


def to_service_regime(bundle_or_case, case_or_op, op=None, scope="none",
                      require_complete_call=False, production=False, **kwargs):
    """Return a ``service_regimes`` rule bound to the calibrated response."""
    bundle, case, operation = _resolve_bundle(bundle_or_case, case_or_op, op)
    return _regime_from_bundle(
        bundle, case, operation, scope,
        require_complete_call=require_complete_call,
        production=production, **kwargs)


def _duration_for(curve, lookup_bytes):
    estimate = curve.estimate(lookup_bytes)
    if estimate is not None:
        return estimate.duration_ms
    if lookup_bytes < curve.knots[0][0]:
        x0, t0 = curve.knots[0]
        return curve.startup_ms + (t0 - curve.startup_ms) * max(
            0.0, lookup_bytes) / x0
    x0, t0 = curve.knots[-2]
    x1, t1 = curve.knots[-1]
    slope = max((t1 - t0) / (x1 - x0), (t1 - curve.startup_ms) / x1)
    return t1 + slope * (lookup_bytes - x1)
