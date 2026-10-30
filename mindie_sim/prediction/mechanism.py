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

from mindie_sim.schema.operators import OpSpec
from .program import KernelProgram, Program, RecipeContext, Stage


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
        return replace(op, **fields, flops=sum(self.matrix.values()),
                       matrix_flops=tuple(sorted(self.matrix.items())),
                       metadata=tuple(sorted(metadata.items())))


@dataclass(frozen=True)
class Geometry:
    """Physical ownership inputs without vendor labels or calibration timings."""

    compute_units: int
    resource_units: Mapping[str, int] = field(default_factory=dict)
    buffers: Mapping[str, int] = field(default_factory=dict)
    buffer_units: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_devices(cls, devices):
        """Use the same representative-shard ownership in export and execution."""
        first = devices[0]
        roles = dict(first.rates).keys() | {"cube", "host"}
        modes = dict(first.service_modes)
        pools = dict(first.buffers)
        return cls(
            first.compute_units,
            {**{role: first.units(role) for role in roles},
             **{name: first.units(role) for name, (role, _) in modes.items()}},
            {pool: min(dict(device.buffers)[pool] for device in devices) for pool in pools},
            {pool: first.pool_units(pool) for pool in pools},
        )


@dataclass(frozen=True)
class OperatorPlan:
    work: Work
    program: Callable[[RecipeContext], Program]
    metadata: Mapping = field(default_factory=dict)

    def lower(self, context: RecipeContext) -> Program:
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
