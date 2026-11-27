"""Source-derived operator plans on the common resource engine.

[Role] Bind explicit typed work and executable programs as one operator extension.
[Invariants] Precision changes service demand on the same matrix resource.
  A plan owns physical work, including workspace traffic; semantic input shape
  remains on OpSpec. Calibration and observations are not planner arguments.
[Timeouts & Locks] Pure metadata operations; planners must stay bounded.
[Side-Effect Policy] No imports of device runtimes, measurements or external I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Callable, Mapping

from mindie_sim.schema.compute_arch import ComputeArchSpec
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_devices
from mindie_sim.schema.operators import OpSpec
from .program.binding import RecipeContext
from .program.stages import KernelProgram, Program, Stage
from .runtime.l2_residency import (
    AdmissionMemoryView,
    RawDieL2Bytes,
    UsableL2Bytes,
    resolve_usable_l2_bytes,
)


@dataclass(frozen=True)
class Work:
    """Matrix FLOPs by compute precision; other resources retain physical units."""

    matrix: Mapping[str, float] = field(default_factory=dict)
    resources: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if "cube" in self.resources:
            raise ValueError("matrix_work_requires_explicit_precision")
        if any(not math.isfinite(v) or v < 0
               for v in (*self.matrix.values(), *self.resources.values())):
            raise ValueError("physical_work_must_be_finite_and_nonnegative")

    def __add__(self, other: Work) -> Work:
        def merge(a, b):
            return {k: a.get(k, 0) + b.get(k, 0) for k in a.keys() | b.keys()}
        return Work(merge(self.matrix, other.matrix), merge(self.resources, other.resources))

    def scaled(self, count: float) -> Work:
        return Work({k: v * count for k, v in self.matrix.items()},
                    {k: v * count for k, v in self.resources.items()})

    @classmethod
    def from_op(cls, op: OpSpec, *, default_dtype: str = "") -> Work:
        """Extract canonical typed work from an untouched ``OpSpec`` declaration."""
        abi = getattr(op, "precision_abi", None)
        resolved_dtype = (
            (getattr(abi, "compute_dtype", "") if abi else "")
            or str(getattr(op, "dtype", "") or "")
            or str(default_dtype or "")
        )
        matrix: dict[str, float] = {}
        for dtype, amount in getattr(op, "matrix_flops", ()) or ():
            val = float(amount)
            if val:
                matrix[str(dtype)] = matrix.get(str(dtype), 0.0) + val
        if not matrix:
            total_flops = float(getattr(op, "flops", 0.0) or 0.0)
            if total_flops:
                matrix[resolved_dtype] = total_flops
        resources: dict[str, float] = {}
        for field_name, role in (
            ("hbm_bytes", "hbm"), ("mte2_bytes", "mte2"), ("mte3_bytes", "mte3"),
            ("vector_ops", "vector"), ("link_bytes", "link"),
        ):
            value = float(getattr(op, field_name, 0.0) or 0.0)
            if value:
                resources[role] = resources.get(role, 0.0) + value
        metadata = dict(getattr(op, "metadata", ()) or ())
        for key, role in (("mte1_bytes", "mte1"), ("fixpipe_bytes", "fixpipe")):
            val = float(metadata.get(key, 0.0) or 0.0)
            if val:
                resources[role] = resources.get(role, 0.0) + val
        for role, value in dict(metadata.get("resource_work", {})).items():
            val = float(value)
            if not val:
                continue
            role_name = str(role)
            if role_name == "cube":
                if not matrix:
                    matrix[resolved_dtype] = val
                continue
            resources[role_name] = resources.get(role_name, 0.0) + val
        return cls(matrix, resources)

    def apply(self, op: OpSpec) -> OpSpec:
        resources = dict(self.resources)
        fields = {name: resources.pop(role, 0) for role, name in (
            ("hbm", "hbm_bytes"), ("mte2", "mte2_bytes"), ("mte3", "mte3_bytes"),
            ("vector", "vector_ops"), ("link", "link_bytes"),
        )}
        metadata = dict(op.metadata)
        # Replace the analytical envelope completely, including extended roles.
        for key in ("resource_work", "mte1_bytes", "fixpipe_bytes"):
            metadata.pop(key, None)
        if resources:
            metadata["resource_work"] = resources
        total_flops = sum(v for v in self.matrix.values() if v)
        matrix_items = tuple(sorted((k, v) for k, v in self.matrix.items() if k and v))
        return replace(op, **fields, flops=total_flops,
                       matrix_flops=matrix_items,
                       metadata=tuple(sorted(metadata.items())))


@dataclass(frozen=True)
class Geometry:
    """Physical ownership inputs without vendor labels or calibration timings."""

    compute_units: int
    resource_units: Mapping[str, int] = field(default_factory=dict)
    buffers: Mapping[str, int] = field(default_factory=dict)
    buffer_units: Mapping[str, int] = field(default_factory=dict)
    physical_mechanisms: object = None
    #: Declared per-second pipe rates and precision peaks.  A planner that must
    #: screen the same mapping the executor will select needs the same declared
    #: physical supply; these are source hardware facts, not observations.
    rates: Mapping[str, float] = field(default_factory=dict)
    compute_peaks: Mapping[str, float] = field(default_factory=dict)
    compute_arch: ComputeArchSpec | None = None
    datapath: DatapathCapabilitySpec | None = None
    replica: int = 0
    rank: int | None = None
    stream_owners: object = None
    admission_memory_view: AdmissionMemoryView | None = None
    requests: tuple = ()
    padded_tokens: int = 0
    layout: str = ""

    @property
    def raw_l2_bytes(self) -> RawDieL2Bytes:
        """Installed physical L2 capacity on this die before reservations or stream sharing."""
        if "l2" in self.buffers:
            return RawDieL2Bytes(max(0, int(self.buffers["l2"])))
        if self.datapath is not None:
            dp_buffers = dict(self.datapath.buffers)
            if "l2" in dp_buffers:
                return RawDieL2Bytes(max(0, int(dp_buffers["l2"])))
        hierarchy = getattr(self.physical_mechanisms, "memory_hierarchy", None)
        if hierarchy is not None and hasattr(hierarchy, "l2_capacity_bytes"):
            return RawDieL2Bytes(max(0, int(hierarchy.l2_capacity_bytes)))
        return RawDieL2Bytes(0)

    def usable_l2_bytes(self, die: int | None = None) -> UsableL2Bytes:
        """Live usable L2 capacity bounded by ``admission_memory_view`` and ``raw_l2_bytes``."""
        resolved_die = (0 if self.rank is None else int(self.rank)) if die is None else int(die)
        return resolve_usable_l2_bytes(
            self.admission_memory_view, resolved_die, fallback_raw_l2=self.raw_l2_bytes,
        )

    @classmethod
    def from_devices(cls, devices, *, links=(), transfer_channels=()):
        """Use the same representative-shard ownership in export and execution."""
        first = devices[0]
        compute_arch, datapath = datapath_for_devices(
            devices, links=links, transfer_channels=transfer_channels
        )
        roles = dict(first.rates).keys() | {"cube", "host"}
        modes = dict(first.service_modes)
        pools = dict(first.buffers)
        return cls(
            first.compute_units,
            {**{role: first.units(role) for role in roles},
             **{name: first.units(role) for name, (role, _) in modes.items()}},
            {pool: min(dict(device.buffers)[pool] for device in devices) for pool in pools},
            {pool: first.pool_units(pool) for pool in pools},
            first.physical_mechanisms,
            dict(first.rates),
            dict(first.compute_peaks),
            compute_arch,
            datapath,
            0,
            int(first.rank),
        )


@dataclass(frozen=True)
class OperatorPlan:
    work: Work
    program: Callable[[RecipeContext], Program]
    metadata: Mapping = field(default_factory=dict)
    #: A native peer program owns its own phase events and launch handles. The
    #: common Execution launches it through launch_native_cohort instead of
    #: wrapping it in a synthetic KernelProgram stage.
    native: bool = False
    #: Structured pipeline specification attached before Engine lowering (never
    #: placed inside C11 Stage metadata).
    pipeline_spec: object = None
    #: Optional physical service context binder applied before work binding.
    context_binder: Callable[[RecipeContext], RecipeContext] | None = None
    traffic_rewrite_capable: bool = False

    def bind_context(self, context: RecipeContext) -> RecipeContext:
        if self.context_binder is not None:
            return self.context_binder(context)
        return context

    def lower(self, context: RecipeContext) -> Program:
        if self.native:
            from .program.stages import NativeProgram
            inner = self.program(replace(context, setup_ms=0, minimum_ms=0))
            return NativeProgram(
                inner, tuple(getattr(inner, "work", ()) or ()))
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",),
                  program=self.program(replace(context, setup_ms=0, minimum_ms=0))),
        ), minimum_ms=context.minimum_ms)


@dataclass(frozen=True)
class PlannedRecipe:
    """A scoped module supplies a planner; registry ownership stays with the caller."""

    build: Callable[[OpSpec, Geometry], OperatorPlan]
    kernels: Callable[[OpSpec], tuple[OpSpec, ...]] | None = None
    traffic_rewrite_capable: bool = False

    @staticmethod
    def declared_work(op: OpSpec) -> Work:
        """Capture canonical typed work from ``op`` before ``prepare()`` mutates work fields."""
        return Work.from_op(op)

    def prepare(self, op: OpSpec, geometry: Geometry):
        plan = self.build(op, geometry)
        prepared = plan.work.apply(op)
        if plan.metadata:
            # A planner may annotate its execution approximation, never
            # overwrite the caller's tensor, layout or source contract.
            metadata = dict(prepared.metadata)
            metadata["program_execution"] = {
                **dict(metadata.get("program_execution", {})), **plan.metadata}
            prepared = replace(prepared, metadata=tuple(sorted(metadata.items())))
        return prepared, plan

    def prepare_with_declared_work(self, op: OpSpec, geometry: Geometry) -> tuple[Work, OpSpec, OperatorPlan]:
        """Capture canonical pre-``prepare()`` typed work alongside ``(prepared, plan)``."""
        floor = Work.from_op(op)
        prepared, plan = self.prepare(op, geometry)
        return floor, prepared, plan
