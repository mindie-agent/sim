"""Communication diagnostic recipe construction for calibrated curves.

[Role] Construct explicit link/HBM/vector/MTE stage ``PlannedRecipe`` instances
  from calibrated communication curves while preserving physical work and
  measured completion-floor boundaries.
[Invariants] Physical link, HBM, MTE, and existing vector work are never deleted
  or clamped. AIV expansion binds explicit vector blocks while AICPU/unspecified
  leaves vector demand untouched. Measured call recipes remain diagnostic-only.
[Timeouts & Locks] Pure in-memory recipe construction; no timers, locks, or I/O.
[Side-Effect Policy] Pure functions; never launches a device or mutates global
  stores.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.schema.common import ModelingError
from .binding import (
    _BYTE_RESOURCES,
    _EXPANSION_MODES,
    _consumer_role,
    _duration_for,
    _fit_bundle,
    _measurement_boundary,
    _placement_for,
    _require_complete_call,
    _validate_hardware,
    _wire_factor,
)
from .golden import A5_HARDWARE_KEY
from .normalization import (
    INTRA_NODE_MAX_RANKS,
    _collective,
    _dtype,
    _explicit_pair,
    _mode,
)

DEFAULT_ACTIVE_AIV_BLOCKS = 16
_DTYPE_ELEMENT_BYTES = {
    "int8": 1.0,
    "fp8-e4m3": 1.0,
    "fp32": 4.0,
    "int32": 4.0,
}
_DEFAULT_ELEMENT_BYTES = 2.0

__all__ = [
    "DEFAULT_ACTIVE_AIV_BLOCKS",
    "_A5CommPlan",
    "build_a5_comm_recipe",
]


@dataclass(frozen=True)
class _A5CommPlan(OperatorPlan):
    """A physical communication program with a measured completion floor.

    ``metadata`` remains the ordinary engine selector record.  The measured
    phase and consumer role are typed attributes so a consumer can require a
    complete collective call without parsing free text or a filename.
    """

    minimum_ms: float = 0.0
    measurement_boundary: Mapping = field(default_factory=dict)
    consumer_role: str = ""
    source_service_semantics: str = ""
    complete_collective_call_response: bool = False
    complete_directed_call_response: bool = False
    direction_observed: bool = False
    direction_basis: str = ""
    algorithm_identified: bool = False
    link_transit_identified: bool = False
    hop_scaling: str = "not_derived"
    diagnostic_only: bool = False
    whole_call_floor: bool = False
    production_eligible: bool = False

    def lower(self, context):
        if context.service_boundary != "device_kernel":
            raise ModelingError("service_curve_boundary_mismatch")
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",),
                  program=self.program(replace(context, setup_ms=0, minimum_ms=0))),
        ), minimum_ms=max(context.minimum_ms, self.minimum_ms))


def _recipe_from_bundle(bundle, collective=None, *, world_size=None,
                        expansion_mode="unspecified",
                        active_aiv_blocks=DEFAULT_ACTIVE_AIV_BLOCKS,
                        allow_unspecified_fallback=False,
                        require_complete_call=False, production=False,
                        **lookup_kwargs):
    if production:
        raise ModelingError("measured_call_recipe_is_diagnostic_only")
    mode = _mode(expansion_mode)
    if mode not in _EXPANSION_MODES and mode != "unspecified":
        raise ModelingError(f"a5_expansion_mode_unknown:{expansion_mode}")
    spec = dict(lookup_kwargs)
    coordinate = spec.pop("coordinate", "link")
    if coordinate not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{coordinate}")
    if spec.pop("boundary", None) not in (None, "device_kernel"):
        raise ModelingError("service_curve_boundary_mismatch")
    placement = spec.pop("placement_scope", None)
    hardware_key = spec.pop("hardware_key", None)
    sku = spec.pop("sku", None)
    arch_family = spec.pop("arch_family", None)
    _validate_hardware({"hardware_key": hardware_key, "sku": sku,
                        "arch_family": arch_family})
    hardware_key = hardware_key or sku

    def build(op, geometry):
        del geometry
        meta = dict(op.metadata)
        if meta.get("device_kernel_group"):
            raise ModelingError("service_curve_boundary_mismatch")
        _validate_hardware(meta)
        op_mode = mode
        if op_mode == "unspecified" and (
            meta.get("expansion_mode") or meta.get("expansion_regime")
        ):
            op_mode = _mode(
                meta.get("expansion_mode") or meta.get("expansion_regime"))
            if op_mode not in _EXPANSION_MODES and op_mode != "unspecified":
                raise ModelingError(f"a5_expansion_mode_unknown:{op_mode}")
        name = _collective(
            collective or meta.get("collective")
            or meta.get("a5_collective") or op.family)
        world = (world_size or meta.get("world_size")
                 or meta.get("group_size") or meta.get("ranks")
                 or INTRA_NODE_MAX_RANKS)
        dtype = _dtype(
            spec.get("dtype") or getattr(op, "dtype", None)
            or meta.get("dtype") or "bf16")
        pair = spec.get("pair") or _explicit_pair(meta.get("p2p_pair"))
        scope = placement or _placement_for(None, op, name, world, pair)
        software = {
            k: spec[k] if k in spec else meta.get(k)
            for k in ("stack", "cann", "algorithm", "protocol")
        }
        extra = {
            k: v for k, v in spec.items()
            if k not in {"dtype", "pair", "stack", "cann", "algorithm", "protocol"}
        }
        curve = bundle.lookup_curve(
            name, world, placement_scope=scope, dtype=dtype,
            expansion_mode=op_mode,
            allow_unspecified_fallback=allow_unspecified_fallback, pair=pair,
            hardware_key=hardware_key or A5_HARDWARE_KEY,
            arch_family=arch_family, boundary="device_kernel",
            **software, **extra)
        if curve.boundary != "device_kernel":
            raise ModelingError("service_curve_boundary_mismatch")
        measurement = _measurement_boundary(curve, name)
        _require_complete_call(measurement, require_complete_call)
        curve_mode = (curve.identity.get("expansion_regime")
                      or curve.identity.get("expansion_mode") or "unspecified")
        metadata = {
            "a5_collective": name, "world_size": int(world),
            "placement_scope": scope, "dtype": dtype,
            "requested_expansion_mode": op_mode,
            "curve_expansion_mode": curve_mode,
            "unspecified_fallback": bool(curve_mode != op_mode),
            "model_id": curve.model_id,
            **{k: software[k] if software[k] is not None else curve.identity.get(k)
               for k in ("stack", "cann", "algorithm", "protocol")},
        }
        wire_factor = (
            _wire_factor(name, world, curve) if coordinate == "link" else 1.0)
        raw_amount = float(
            getattr(op, coordinate + "_bytes", 0)
            or getattr(op, "link_bytes", 0)
            or getattr(op, "hbm_bytes", 0) or 1.0)
        link_bytes = float(
            getattr(op, "link_bytes", 0)
            or (getattr(op, coordinate + "_bytes", 0)
                if coordinate == "link" else 0) or 0.0)
        lookup_bytes = raw_amount / wire_factor if coordinate == "link" else raw_amount
        duration = float(_duration_for(curve, lookup_bytes) or 0.0)
        hbm, mte2, mte3 = (
            float(getattr(op, k, 0) or 0)
            for k in ("hbm_bytes", "mte2_bytes", "mte3_bytes"))
        elem_bytes = _DTYPE_ELEMENT_BYTES.get(dtype, _DEFAULT_ELEMENT_BYTES)
        vector = (
            (float(getattr(op, "vector_ops", 0) or 0)
             or max(1.0, raw_amount / elem_bytes))
            if op_mode == "aiv" else float(getattr(op, "vector_ops", 0) or 0))
        if op_mode != "aiv":
            hbm = max(1.0, hbm or raw_amount)
        resources = {
            "hbm": hbm,
            **{k: v for k, v in (
                ("vector", vector), ("link", link_bytes),
                ("mte2", mte2), ("mte3", mte3)) if v > 0},
        }
        work = Work(resources=resources)

        def program(context):
            _validate_hardware(
                context.service_identity.get("hardware", {}), identity=True)
            link_work = {res: val for res, val in context.work.items()
                         if res not in context.resource_names.values()}
            stages = [Stage("link", link_work) if link_work
                      else Stage("link", latency=duration)]
            for res_name, amount in (("hbm", hbm), ("mte2", mte2), ("mte3", mte3)):
                if amount > 0:
                    stages.append(Stage(
                        res_name, context.bind(Work(resources={res_name: amount}))))
            if vector > 0:
                place = {}
                if op_mode == "aiv":
                    count = context.resource_units.get("vector", context.core_count)
                    units = tuple(range(min(max(1, int(active_aiv_blocks)), count)))
                    place = {context.resource_names["vector"]: units}
                stages.append(Stage(
                    "vector", context.bind(Work(resources={"vector": vector})),
                    placement=place))
            return KernelProgram(tuple(stages))

        return _A5CommPlan(
            work, program, minimum_ms=duration, metadata=metadata,
            measurement_boundary=measurement,
            consumer_role=_consumer_role(measurement),
            source_service_semantics=(
                curve.provenance.get("service_semantics")
                or measurement.get("service_semantics") or ""),
            complete_collective_call_response=bool(
                measurement.get("complete_collective_call")),
            complete_directed_call_response=bool(
                measurement.get("complete_directed_call")),
            direction_observed=bool(measurement.get("direction_observed", False)),
            direction_basis=str(curve.provenance.get("direction_basis") or ""),
            algorithm_identified=bool(
                measurement.get("algorithm_identified", False)),
            link_transit_identified=bool(
                measurement.get("link_transit_identified", False)),
            hop_scaling=measurement.get("hop_scaling", "not_derived"),
            diagnostic_only=True, whole_call_floor=True,
            production_eligible=False)

    return PlannedRecipe(build)


def build_a5_comm_recipe(bundle_or_collective=None, collective=None,
                         allow_unspecified_fallback=False,
                         require_complete_call=False, production=False,
                         **kwargs):
    """Build a PlannedRecipe from the calibrated curve; bundle is optional."""
    if hasattr(bundle_or_collective, "lookup_curve"):
        bundle, name = bundle_or_collective, collective
    else:
        bundle, name = _fit_bundle(), (collective or bundle_or_collective)
    return _recipe_from_bundle(
        bundle, name,
        allow_unspecified_fallback=allow_unspecified_fallback,
        require_complete_call=require_complete_call,
        production=production, **kwargs)
