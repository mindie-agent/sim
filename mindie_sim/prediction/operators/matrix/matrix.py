"""Finite matrix tiles, operand traffic and on-chip residency.

[Role] Implement an explicit M/N wave and K-panel matrix program.
[Invariants] Tails conserve arithmetic. Ideal shared-cache reuse changes HBM
  traffic, never per-tile private-buffer copies or FLOPs. A panel buffer remains
  live until consumed; accumulators span K panels. Tiles own units and buffers.
[Timeouts & Locks] Lazy O(compute_units * slots) tile construction, no I/O.
[Side-Effect Policy] Submits through the shared program engine only.
"""
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.program import (KernelProgram, ParallelProgram, PipelinePhase,
                      PipelineProgram, PipelineTemplate,
                      PipelineTemplateProgram, Stage,
                      indexed_pipeline_consumer_available)
from .refill import RefillPipeline
from .matrix_epilogue import MatrixEpilogue
from mindie_sim.prediction.ownership import read_role
from .matrix_traffic import FP32_BYTES

DEFAULT_TILE_EDGE = 128
DEFAULT_REFILL_TILE_MN = 256
DEFAULT_REFILL_TILE_K = 128
DEFAULT_COMPUTE_TILE_MN = 128

DEFAULT_SPLIT_K = 1
MAX_SPLIT_K = 32


def directional_mte1_available(context):
    """Check whether both operand transfer modes have a declared service."""
    return all(context.service_rate(role) is not None for role in ("mte1_l0a", "mte1_l0b"))


def operand_transfer(context, left, right, *, directional=False):
    """A/B instructions share one MTE1 pipe; a source may select its modes."""
    if directional and directional_mte1_available(context):
        from mindie_sim.prediction.mechanism import Work
        return context.bind(Work(resources={"mte1_l0a": left, "mte1_l0b": right}))
    name = context.resource_names.get("mte1")
    return {name: left + right} if name is not None else {}


class _TiledBindings:
    """Producer-side lifecycle registry populated where allocations are built."""

    def __init__(self):
        self._rows = {}

    def record(self, allocation, scope, ordinal):
        row = self._rows.get(allocation.name)
        if row is None:
            self._rows[allocation.name] = {
                "name": allocation.name, "scope": scope,
                "ordinal": int(ordinal), "pool": allocation.pool,
                "size": allocation.size, "units": tuple(allocation.units),
                "action": "acquire", "phase": 0, "repeat": 1,
            }

    def rows(self, owner):
        return tuple({**row, "owner": owner}
                     for row in sorted(self._rows.values(),
                                       key=lambda item: item["name"]))


def _owner_template(pipe, owner, panels, k_tail):
    template = pipe.template(owner=owner, panels_per_output=panels,
                             k_tail=k_tail)
    waves = pipe.count // panels if panels else 0
    motif = tuple(template.phases[:3])
    registry = getattr(pipe, "producer_bindings", None)
    bindings = (registry.rows(owner) if registry is not None
                else _producer_bindings(template))
    return replace(template, bindings=bindings,
                   motif_phases=motif, motif_repeats=max(1, waves),
                   static_phase_classes=len(motif))


def _producer_bindings(template) -> tuple[dict, ...]:
    """TiledMatmul's own allocation lifecycle, supplied to the template.

    The producer (not T04) knows that ``panel*`` is a panel-local operand and
    ``accumulator*`` spans one output's K panels.  Setup/drain are scanned too
    so an invocation-scoped reservation can cross the whole template.
    """
    rows = []
    seen = set()
    stages = (*template.setup, *(stage for phase in template.phases
                                 for stage in phase.stages), *template.drain)
    for stage in stages:
        for allocation, action in (
                *((item, "acquire") for item in stage.acquire),
                *((item, "free") for item in stage.free)):
            key = (allocation.name, action)
            if key in seen:
                continue
            seen.add(key)
            name = allocation.name
            if name.startswith("panel"):
                scope = "panel"
            elif name.startswith("accumulator"):
                scope = "output"
            else:
                scope = "invocation"
            digits = "".join(ch for ch in name if ch.isdigit())
            rows.append({
                "name": name, "scope": scope, "owner": template.owner,
                "ordinal": int(digits or 0), "pool": allocation.pool,
                "size": allocation.size, "units": tuple(allocation.units),
                "action": action, "phase": 0, "repeat": 1,
            })
    return tuple(rows)


_INDEXED_OPERAND_SYMBOL = "panel"
_INDEXED_ACCUMULATOR_SYMBOL = "accumulator"
_INDEXED_SYMBOL_SCOPES = {
    _INDEXED_OPERAND_SYMBOL: "panel",
    _INDEXED_ACCUMULATOR_SYMBOL: "output",
}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor if value > 0 else 0


def _tile_lengths(total: int, tile: int) -> tuple[int, ...]:
    """Quotient/remainder gives each axis at most one full and one tail length."""
    quotient, remainder = divmod(total, tile)
    return ((tile,) if quotient else ()) + ((remainder,) if remainder else ())


def _k_panel_classes(k: int, tile_k: int):
    """Finite K panel lengths and their first/middle/last roles."""
    panels = _ceil_div(k, tile_k)
    if not panels:
        return ()
    if panels == 1:
        # The only panel is simultaneously first and last; its real length may
        # still be shorter than tile_k.
        return ((min(tile_k, k), True, True),)
    last = k - (panels - 1) * tile_k
    if panels == 2:
        return ((tile_k, True, False), (last, False, True))
    # Keep last/full and middle/full as distinct topologies.
    return ((tile_k, True, False), (tile_k, False, False),
            (last, False, True))


def _indexed_bindings(owner, phases, *, setup=(), drain=()):
    """Declare stable symbol lifetimes once; never infer from digits."""
    rows = {}
    stages = [*setup, *drain]
    for phase in phases:
        stages.extend(phase.stages)
        for group in phase.residue_stages:
            stages.extend(group)
    for stage in stages:
        for allocation in (*stage.acquire, *stage.free):
            scope = _INDEXED_SYMBOL_SCOPES.get(allocation.name)
            if scope is None:
                raise ValueError(
                    "pipeline_indexed_allocation_symbol_unbound:"
                    f"{allocation.name}")
            key = (allocation.name, scope)
            if key not in rows:
                rows[key] = {
                    "name": allocation.name, "scope": scope,
                    "owner": owner, "action": "acquire", "phase": 0,
                    "repeat": 1,
                }
    return tuple(rows[key] for key in sorted(rows))


class _TiledGeometry:
    """Shared geometry and finite class construction for TiledMatmul.

    The detailed path and the indexed catalog both use the same formula,
    panel-stage core and allocation symbols.  Only the indexed mode builds the
    bounded class catalog; it never enumerates real waves and de-duplicates.
    """

    def __init__(self, recipe, context):
        if min(recipe.tile_m, recipe.tile_n, recipe.tile_k, recipe.slots) < 1:
            raise ValueError("matrix_tile_dimensions_and_slots_must_be_positive")
        self.recipe = recipe
        self.context = context
        op = context.op
        self.op = op
        self.total_m, self.n, self.k = (int(value) for value in
                                        op.local_shape[-3:])
        metadata = dict(op.metadata)
        self.groups = int(metadata.get("matrix_groups", 1))
        if self.groups < 1 or self.total_m % self.groups:
            raise ValueError("matrix_groups_require_equal_integer_rows")
        self.m = self.total_m // self.groups
        self.nm = _ceil_div(self.m, recipe.tile_m)
        self.nn = _ceil_div(self.n, recipe.tile_n)
        self.nk = _ceil_div(self.k, recipe.tile_k)
        self.tiles = self.groups * self.nm * self.nn
        effective_cores = int(getattr(context, "effective_aic_cores", context.core_count))
        if getattr(context, "stream_owners", None) is not None and effective_cores <= 0:
            raise ModelingError(
                f"stream_zero_aic_budget:{op.stream or 'main'}:{op.op_id}"
            )
        self.physical_cores = int(context.core_count)
        self.cores = max(1, effective_cores)
        self.epilogue = MatrixEpilogue.from_op(op)
        self.input_bytes, self.weight = self.epilogue.operands(op)
        names = context.resource_names
        self.operand_read = read_role(names)
        self.scale_read = read_role(names, aiv=0)
        self.act_width = dtype_bytes(
            metadata.get("activation_format", op.dtype or "bf16"))
        self.output_width = dtype_bytes(
            metadata.get("output_format", "bf16"))
        self.operand_pool = context.pool_names.get(recipe.operand_pool)
        self.accumulator_pool = context.pool_names.get(recipe.accumulator_pool)
        excluded = {
            names["hbm"], names.get("mte2"), names.get("mte3"),
            names.get(self.operand_read), names.get(self.scale_read),
        }
        if self.epilogue.operations_per_output:
            excluded.add(names.get("vector"))
        self._compute_work_items = tuple(
            (key, value) for key, value in context.work.items()
            if key not in excluded
        )
        self._acc_elem_bytes = max(FP32_BYTES, self.output_width)
        self._panel_class_cache: dict = {}

    def indexed_guard(self) -> bool:
        """Only the proven equal-group, strided owner placement is indexable."""
        recipe = self.recipe
        return (self.groups >= 1
                and self.total_m % self.groups == 0
                and self.m == self.total_m // self.groups
                and self.cores >= 1
                and self.nm == _ceil_div(self.m, recipe.tile_m)
                and self.nn == _ceil_div(self.n, recipe.tile_n)
                and self.nk == _ceil_div(self.k, recipe.tile_k)
                and self.tiles == self.groups * self.nm * self.nn)

    def indexed_panels(self, owner: int) -> int:
        if self.cores <= 0 or self.nk <= 0 or owner < 0 or owner >= self.tiles:
            return 0
        waves = 1 + (self.tiles - 1 - owner) // self.cores
        return waves * self.nk

    def coordinates(self, owner: int, index: int):
        """The contract's absolute-index mapping for one logical panel."""
        panels = self.indexed_panels(owner)
        if index < 0 or index >= panels:
            raise IndexError("pipeline_index_out_of_range")
        wave, panel = divmod(index, self.nk)
        tile = owner + wave * self.cores
        group, within = divmod(tile, self.nm * self.nn)
        row, column = divmod(within, self.nn)
        a = min(self.recipe.tile_m, self.m - row * self.recipe.tile_m)
        b = min(self.recipe.tile_n, self.n - column * self.recipe.tile_n)
        kd = min(self.recipe.tile_k, self.k - panel * self.recipe.tile_k)
        return (a, b, kd, panel == 0, panel + 1 == self.nk,
                wave, panel, group, row, column)

    def catalog(self):
        """Cartesian product of the finite per-axis lengths and K roles."""
        a_lengths = _tile_lengths(self.m, self.recipe.tile_m)
        b_lengths = _tile_lengths(self.n, self.recipe.tile_n)
        k_classes = _k_panel_classes(self.k, self.recipe.tile_k)
        return tuple((a, b, kd, first, last)
                     for a in a_lengths
                     for b in b_lengths
                     for kd, first, last in k_classes)

    def index_of(self, owner: int, class_ids):
        """Pure integer closure: no Engine, registry, request or event."""
        panels = self.indexed_panels(owner)
        nk, slots, cores = self.nk, self.recipe.slots, self.cores
        nm_nn, nn = self.nm * self.nn, self.nn
        tile_m, tile_n = self.recipe.tile_m, self.recipe.tile_n
        tile_k, m, n, k = (self.recipe.tile_k, self.m, self.n, self.k)
        ids = MappingProxyType(dict(class_ids))

        def mapping(index, _panels=panels, _nk=nk, _slots=slots,
                    _cores=cores, _nm_nn=nm_nn, _nn=nn, _tile_m=tile_m,
                    _tile_n=tile_n, _tile_k=tile_k, _m=m, _n=n, _k=k,
                    _owner=owner, _ids=ids):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("pipeline_index_requires_integer")
            if index < 0 or index >= _panels:
                raise IndexError("pipeline_index_out_of_range")
            wave, panel = divmod(index, _nk)
            tile = _owner + wave * _cores
            _group, within = divmod(tile, _nm_nn)
            row, column = divmod(within, _nn)
            a = min(_tile_m, _m - row * _tile_m)
            b = min(_tile_n, _n - column * _tile_n)
            kd = min(_tile_k, _k - panel * _tile_k)
            first, last = panel == 0, panel + 1 == _nk
            phase_id = _ids[(a, b, kd, first, last)]
            stop = (index + 1) if first or last else wave * _nk + _nk - 1
            return (phase_id, stop, wave, panel, index % _slots)

        return mapping

    def panel_stages(self, owner, a, b, kd, *, first, last,
                     operand_symbol, accumulator_symbol):
        """Shared real-body formula for detailed and indexed modes."""
        total_m, n, k, groups = self.total_m, self.n, self.k, self.groups
        if total_m <= 0 or n <= 0 or k <= 0:
            raise ValueError("matrix_panel_requires_positive_geometry")
        ckey = (owner, a, b, kd, first, last)
        cached = self._panel_class_cache.get(ckey)
        if cached is not None:
            proto_stages, resident, acc_size, owner_units = cached
            op_pool = self.operand_pool
            acc_pool = self.accumulator_pool
            operand = (() if op_pool is None else (
                Allocation(operand_symbol, op_pool, resident, owner_units),))
            accumulator = (() if acc_pool is None else (
                Allocation(accumulator_symbol, acc_pool, acc_size, owner_units),))
            s0 = proto_stages[0]._with_buffers(operand, ())
            s1 = proto_stages[1]._with_buffers(
                accumulator if first else (), operand)
            if not last:
                return (s0, s1, proto_stages[2]), operand, accumulator
            s_last = proto_stages[-1]._with_buffers((), accumulator)
            if len(proto_stages) == 3:
                return (s0, s1, s_last), operand, accumulator
            return (s0, s1, *proto_stages[2:-1], s_last), operand, accumulator
        recipe, context = self.recipe, self.context
        op, names = context.op, context.resource_names
        fraction = a * b / (total_m * n)
        k_fraction = kd / k
        # The flat tile index is group-major. A tile reads its own group's B
        # only; A is private per N tile. Ideal shared-cache reuse suppresses
        # only HBM reads, not cache-to-owner copy service.
        a_copy = self.input_bytes * a / total_m * k_fraction
        b_copy = self.weight / groups * b / n * k_fraction
        a_read = (self.input_bytes * fraction * k_fraction
                  if recipe.reuse_inputs else a_copy)
        b_read = (self.weight * fraction * k_fraction
                  if recipe.reuse_weights else b_copy)
        reads = a_read + b_read
        writes = op.mte3_bytes * fraction if last else 0
        load = {names["hbm"]: reads}
        store = {names["hbm"]: writes}
        for role, volume, target in (
                (self.operand_read, a_copy + b_copy, load),
                ("mte3", writes, store)):
            if role in names:
                target[names[role]] = volume
        # Exact panel share: multiplying by an integer numerator before the
        # single division keeps tile work additive over the whole matrix, so
        # the served FLOP total conserves the operator total bit-for-bit.
        k_numerator = a * b * kd
        k_denominator = total_m * n * k
        compute = {key: value * k_numerator / k_denominator
                   for key, value in self._compute_work_items}
        # Explicit equal groups have physical A/B panel domains. A one-group
        # declaration retains the prior buffer approximation, including
        # aggregate expert moments whose individual shapes are unspecified.
        resident = (a_copy + b_copy if groups > 1
                    else (a + b) * kd * self.act_width)
        acc_size = a * b * self._acc_elem_bytes
        owner_units = (owner,)
        operand = (() if self.operand_pool is None else (
            Allocation(operand_symbol, self.operand_pool, resident,
                       owner_units),))
        accumulator = (() if self.accumulator_pool is None else (
            Allocation(accumulator_symbol, self.accumulator_pool,
                       acc_size, owner_units),))
        stages = [
            Stage("load", load, acquire=operand,
                  placement=context.owned(owner, self.physical_cores,
                                          (self.operand_read,))
                  if recipe.distributed_transfers else {}),
            Stage("compute", compute, after=("load",),
                  caps=context.mac_caps,
                  placement=context.owned(owner, self.physical_cores,
                                          ("cube", "vector")),
                  acquire=accumulator if first else (), free=operand),
        ]
        predecessor = "compute"
        if last and self.epilogue.operations_per_output:
            scale_copy, scale_hbm = self.epilogue.scales(
                a, b, total_m, n, groups,
                reuse_inputs=recipe.reuse_inputs,
                reuse_weights=recipe.reuse_weights)
            read = self.scale_read
            scale_work = {names["hbm"]: scale_hbm}
            if read in names:
                scale_work[names[read]] = (
                    scale_work.get(names[read], 0) + scale_copy)
            stages.extend((
                Stage("scales", scale_work, after=("compute",),
                      placement=context.owned(owner, self.physical_cores, (read,))),
                Stage("epilogue", {
                    names["vector"]:
                        self.epilogue.operations_per_output * a * b},
                    after=("scales",),
                    placement=context.owned(owner, self.physical_cores,
                                            ("vector",))),
            ))
            predecessor = "epilogue"
        stages.append(Stage(
            "store", store, after=(predecessor,),
            placement=context.owned(owner, self.physical_cores, ("mte3",))
            if recipe.distributed_transfers else {},
            free=accumulator if last else ()))
        proto_stages = tuple(stages)
        self._panel_class_cache[ckey] = (
            proto_stages, resident, acc_size, owner_units,
        )
        return proto_stages, operand, accumulator

    def indexed_templates(self, *, setup=()):
        """Build the bounded catalog, or ``None`` for unsupported geometry."""
        if not self.indexed_guard():
            return None
        if not self.tiles:
            return ()
        classes = self.catalog()
        owners = min(self.cores, self.tiles)
        templates = []
        for owner in range(owners):
            phases = []
            class_ids = {}
            for phase_id, (a, b, kd, first, last) in enumerate(classes):
                stages, _operand, _accumulator = self.panel_stages(
                    owner, a, b, kd, first=first, last=last,
                    operand_symbol=_INDEXED_OPERAND_SYMBOL,
                    accumulator_symbol=_INDEXED_ACCUMULATOR_SYMBOL)
                kind = ("single" if first and last else
                        "first" if first else "last" if last else "middle")
                phases.append(PipelinePhase(
                    kind, stages, 1, key=(a, b, kd, first, last)))
                class_ids[(a, b, kd, first, last)] = phase_id
            panels = self.indexed_panels(owner)
            phases = tuple(phases)
            templates.append(PipelineTemplate(
                owner, self.recipe.slots, phases,
                setup=tuple(setup),
                bindings=_indexed_bindings(owner, phases, setup=setup),
                motif_phases=(), motif_repeats=1,
                tile_factory_calls=len(phases),
                stage_objects=sum(len(phase.stages) for phase in phases),
                phase_objects=len(phases),
                static_phase_classes=len(phases),
                dynamic_instances=panels,
                indexed_panels=panels,
                index_of=self.index_of(owner, class_ids),
            ))
        return tuple(templates)


@dataclass(frozen=True)
class TiledMatmul:
    tile_m: int = DEFAULT_TILE_EDGE
    tile_n: int = DEFAULT_TILE_EDGE
    tile_k: int = DEFAULT_TILE_EDGE
    slots: int = 2
    reuse_inputs: bool = True
    reuse_weights: bool = True
    operand_pool: str = "l1"
    accumulator_pool: str = "l0c"
    distributed_transfers: bool = True
    #: Production path consumes the finite template through T04.  ``False``
    #: keeps the original detailed PipelineProgram as the reference/fallback.
    template_execution: bool = True

    def finite_templates(self, context, *, setup=()):
        """Build the bounded indexed catalog for the supported geometry.

        This is construction only.  Normal ``lower``/``pipeline_templates``
        select it only after checking the actual consumer ABI.
        """
        return _TiledGeometry(self, context).indexed_templates(setup=setup)

    def pipeline_templates(self, context):
        """Return per-owner finite phase templates for the T04 repeat executor.

        With the real indexed consumer ABI this is the bounded class catalog.
        Without it the original exact expanded template path is retained, so an
        old reader never receives a unique class catalog as if it were 3 tiles.
        """
        if not context.op.local_shape[-3:][0]:
            return ()
        if indexed_pipeline_consumer_available():
            geometry = _TiledGeometry(self, context)
            templates = geometry.indexed_templates(
                setup=(Stage("setup", latency=context.setup_ms),))
            if templates is not None:
                return templates
        # Original exact path: detailed program then representative phases.
        program = self.lower(context) if not self.template_execution else (
            replace(self, template_execution=False).lower(context))
        cores = next((stage.program for stage in program.stages
                      if stage.name == "cores"), None)
        if cores is None or not hasattr(cores, "programs"):
            return ()
        k = int(context.op.local_shape[-1])
        panels = max(1, math.ceil(k / self.tile_k)) if k else 1
        tail = bool(k and k % self.tile_k)
        setup = tuple(stage for stage in program.stages if stage.name == "setup")
        return tuple(
            replace(_owner_template(pipe, owner, panels, tail), setup=setup)
            for owner, pipe in enumerate(cores.programs))

    def lower(self, context):
        geometry = _TiledGeometry(self, context)
        if self.template_execution and indexed_pipeline_consumer_available():
            templates = geometry.indexed_templates()
            if templates is not None:
                programs = tuple(PipelineTemplateProgram(template)
                                 for template in templates)
                return KernelProgram((
                    Stage("setup", latency=context.setup_ms, keep_empty=True),
                    Stage("cores", after=("setup",),
                          program=ParallelProgram(programs)),
                ), minimum_ms=context.minimum_ms)
        registry = _TiledBindings()

        def panel(core, index):
            (a, b, kd, first, last, wave, _panel, _group,
             _row, _column) = geometry.coordinates(core, index)
            stages, operand, accumulator = geometry.panel_stages(
                core, a, b, kd, first=first, last=last,
                operand_symbol=f"panel{index}",
                accumulator_symbol=f"accumulator{wave}")
            if operand:
                registry.record(operand[0], "panel", index)
            if accumulator:
                registry.record(accumulator[0], "output", wave)
            return stages

        pipelines = tuple(
            PipelineProgram(
                geometry.indexed_panels(core), self.slots,
                lambda index, core=core: panel(core, index),
                # The detailed reference path keeps its unrolled per-tile
                # program on ``start()`` (``template_start=False``) while still
                # declaring panel-class homogeneity for explicit ``template()``
                # inspection.
                homogeneous=True,
                template_start=self.template_execution,
                producer_bindings=registry)
            for core in range(min(geometry.cores, geometry.tiles)))
        if self.template_execution:
            panels = max(1, geometry.nk)
            k_tail = bool(geometry.k and geometry.k % self.tile_k)
            programs = tuple(
                PipelineTemplateProgram(
                    _owner_template(pipe, owner, panels, k_tail))
                for owner, pipe in enumerate(pipelines))
        else:
            programs = pipelines
        return KernelProgram((
            Stage("setup", latency=context.setup_ms, keep_empty=True),
            Stage("cores", after=("setup",), program=ParallelProgram(programs)),
        ), minimum_ms=context.minimum_ms)


@dataclass(frozen=True)
class HierarchicalMatmul:
    """A private refill tile containing smaller matrix-consumption tiles.

    The L1 panel remains resident until every inner tile consumes it. Each
    inner tile still transfers operands to its private matrix operand buffers.
    No cache hit removes service at a different memory boundary.
    """

    tile_m: int = DEFAULT_REFILL_TILE_MN
    tile_n: int = DEFAULT_REFILL_TILE_MN
    tile_k: int = DEFAULT_REFILL_TILE_K
    compute_m: int = DEFAULT_COMPUTE_TILE_MN
    compute_n: int = DEFAULT_COMPUTE_TILE_MN
    slots: int = 2
    compute_slots: int = 2
    reuse_inputs: bool = True
    reuse_weights: bool = True
    k_splits: int = DEFAULT_SPLIT_K

    def _append_panel_output(
        self, stages, *, output, epilogue, context, core, cores, a, b, total_m, n,
        groups, names, scale_read, transfer, writes, accumulator,
    ):
        if output is not None:
            stages.append(Stage(
                "output", after=("matrix",), free=accumulator,
                program=output.lower(
                    context, core=core, rows=a, columns=b,
                    buffer_shape=(self.tile_m, self.tile_n), epilogue=epilogue,
                    reuse_inputs=self.reuse_inputs, reuse_weights=self.reuse_weights)))
            return
        if epilogue.operations_per_output:
            scale_copy, scale_hbm = epilogue.scales(
                a, b, total_m, n, groups,
                reuse_inputs=self.reuse_inputs, reuse_weights=self.reuse_weights)
            stages.extend((
                Stage("scales", transfer(scale_read, scale_copy, hbm=scale_hbm),
                      after=("matrix",), placement=context.owned(core, cores, (scale_read,))),
                Stage("epilogue", {names["vector"]: epilogue.operations_per_output * a * b},
                      after=("scales",), placement=context.owned(core, cores, ("vector",))),
            ))
        stages.append(Stage(
            "store", transfer("mte3", writes, hbm=writes),
            after=("epilogue",) if epilogue.operations_per_output else ("matrix",),
            free=accumulator, placement=context.owned(core, cores, ("mte3",))))

    def lower(self, context, *, owner_tiles=None, group_rows=None, output=None,
              serial_tiles=False, directional_mte1=False):
        if min(self.tile_m, self.tile_n, self.tile_k, self.compute_m,
               self.compute_n, self.slots, self.compute_slots) < 1:
            raise ValueError("matrix_tile_dimensions_and_slots_must_be_positive")
        if self.tile_m % self.compute_m or self.tile_n % self.compute_n:
            raise ValueError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
        from .matrix_mapping import (freeze_owner_tiles, grouped_owner_rectangles,
                                     matrix_geometry, owner_records)
        owner_tiles = (freeze_owner_tiles(owner_tiles)
                       if owner_tiles is not None else None)
        op, names = context.op, context.resource_names
        if (owner_tiles is None and group_rows is not None
                and len(op.local_shape) >= 3):
            # A declared unequal group must launch each group's own row tail,
            # not reconstruct every rectangle from the widest group.
            owner_tiles = freeze_owner_tiles(grouped_owner_rectangles(
                group_rows, op.local_shape[-2], self.tile_m, self.tile_n,
                context.core_count))
        explicit = (owner_records(owner_tiles) if owner_tiles is not None else None)
        geometry = matrix_geometry(context, self, owners=explicit,
                                   group_rows=group_rows)
        total_m, n, k = geometry.local_shape
        metadata = dict(op.metadata)
        groups, m = geometry.groups, geometry.m_per_group
        if not total_m:
            return KernelProgram((Stage("setup", latency=context.setup_ms),),
                                 minimum_ms=context.minimum_ms)
        nm, nn, nk = geometry.nm, geometry.nn, geometry.nk
        cores = context.core_count
        epilogue = MatrixEpilogue.from_op(op)
        inputs, weight = epilogue.operands(op)
        operand_read = read_role(names)
        scale_read = read_role(names, aiv=0)
        compute_width = dtype_bytes(metadata.get("compute_format", op.dtype or "bf16"))
        output_width = dtype_bytes(metadata.get("output_format", "bf16"))
        work = {name: value for name, value in context.work.items()
                if name not in {names["hbm"], names.get("mte1"),
                                names.get("mte2"), names.get("mte3"),
                                names.get(operand_read), names.get(scale_read)}}
        if epilogue.operations_per_output:
            work.pop(names["vector"], None)
        if owner_tiles is None:
            counts = tuple(record[0] for record in geometry.owners)
            tile_sources = None
        else:
            # A source adapter supplies finite owner-local rectangles. K-panel
            # execution and all resource/buffer semantics remain shared here.
            counts = tuple(count for count, _ in owner_tiles)
            tile_sources = [tuple(tiles) for _, tiles in owner_tiles]

        def allocation(name, pool, size, core):
            return (() if pool not in context.pool_names else
                    (Allocation(name, context.pool_names[pool], size, (core,)),))

        def transfer(role, volume, *, hbm=None):
            result = {names[role]: volume} if role in names else {}
            if hbm is not None:
                result[names["hbm"]] = hbm
            return result

        def panel(core, index):
            wave, kk = divmod(index, nk)
            if tile_sources is None:
                tile = wave * cores + core
                a = min(self.tile_m, m - ((tile // nn) % nm) * self.tile_m)
                b = min(self.tile_n, n - (tile % nn) * self.tile_n)
            else:
                a, b = tile_sources[core][wave]
            kd = min(self.tile_k, k - kk * self.tile_k)
            fraction, k_fraction = a * b / (total_m * n), kd / k
            a_copy = inputs * a / total_m * k_fraction
            b_copy = weight * b / (n * groups) * k_fraction
            reads = ((inputs * fraction * k_fraction if self.reuse_inputs else a_copy)
                     + (weight * fraction * k_fraction if self.reuse_weights else b_copy))
            writes = op.mte3_bytes * fraction if kk == nk - 1 else 0
            operand = allocation(f"panel{index}", "l1", a_copy + b_copy, core)
            accumulator = allocation(f"accumulator{wave}", "l0c",
                                     a * b * max(FP32_BYTES, output_width), core)
            inner_n = math.ceil(b / self.compute_n)

            def steps():
                count = math.ceil(a / self.compute_m) * inner_n
                for part in range(count):
                    ai = min(self.compute_m, a - part // inner_n * self.compute_m)
                    bi = min(self.compute_n, b - part % inner_n * self.compute_n)
                    left, right = a_copy * ai / a, b_copy * bi / b
                    # Keep packed payload/scales and the widened representation
                    # bound. Conversion latency is not inferred from these bytes.
                    left = max(left, ai * kd * compute_width)
                    right = max(right, bi * kd * compute_width)
                    local = (
                        *allocation(f"a{index}:{part}", "l0a", left, core),
                        *allocation(f"b{index}:{part}", "l0b", right, core),
                    )
                    demand = {name: value * ai * bi / (total_m * n) * k_fraction
                              for name, value in work.items()}
                    stages = [
                        Stage("operand", operand_transfer(context, left, right,
                              directional=directional_mte1), acquire=local,
                              placement=context.owned(core, cores, ("mte1",))),
                        Stage("matrix", demand, after=("operand",), free=local,
                              acquire=accumulator if kk == 0 and part == 0 else (),
                              placement=context.owned(core, cores, ("cube", "vector"))),
                    ]
                    if part == count - 1 and kk == nk - 1:
                        self._append_panel_output(
                            stages, output=output, epilogue=epilogue, context=context,
                            core=core, cores=cores, a=a, b=b, total_m=total_m, n=n,
                            groups=groups, names=names, scale_read=scale_read,
                            transfer=transfer, writes=writes, accumulator=accumulator)
                    yield tuple(stages), part == count - 1

            return (Stage("load", transfer(operand_read, a_copy + b_copy, hbm=reads),
                          acquire=operand, placement=context.owned(core, cores, (operand_read,))),
                    steps(), operand)

        programs = []
        for core, count in enumerate(counts):
            if not count:
                continue
            program = RefillPipeline(
                count * nk, self.slots, self.compute_slots,
                lambda index, core=core: panel(core, index), nk if serial_tiles else None)
            if output is not None:
                buffers = output.allocations(
                    context, core, (self.tile_m, self.tile_n), epilogue)
                program = KernelProgram((
                    Stage("reserve_output", acquire=buffers),
                    Stage("body", after=("reserve_output",), program=program),
                    Stage("release_output", after=("body",), free=buffers),
                ))
            programs.append(program)
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("cores", after=("setup",), program=ParallelProgram(tuple(programs))),
        ), minimum_ms=context.minimum_ms)
