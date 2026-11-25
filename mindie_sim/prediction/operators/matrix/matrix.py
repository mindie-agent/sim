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
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.program import (KernelProgram, ParallelProgram, PipelinePhase,
                      PipelineProgram, PipelineTemplate,
                      PipelineTemplateProgram, Stage,
                      indexed_pipeline_consumer_available)
from .refill import RefillPipeline
from .matrix_epilogue import MatrixEpilogue
from mindie_sim.prediction.ownership import read_role

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_OPS_TRANSFORMER_REV = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
_CATLASS_REV = "dd03bc0998cbcd67c982e7ab13df6ee15780c6f3"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer", _OPS_TRANSFORMER_REV,
        "mc2/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp",
        "472db5c1bb12fd01289005f636dfa7b86cb0d49c9ce8ee77313fe8c430d618a0",
    ),
    SourceRef(
        "cann/catlass", _CATLASS_REV,
        "include/catlass/gemm/block/block_mmad_mx_tla.hpp",
        "a946988a9f3a049e8702d01e1c8453c047cf6ab9f4f2fbe7ead94a490dfa974a",
    ),
)

FP32_BYTES = 4

DEFAULT_TILE_EDGE = 128
DEFAULT_REFILL_TILE_MN = 256
DEFAULT_REFILL_TILE_K = 128
DEFAULT_COMPUTE_TILE_MN = 128

DEFAULT_SPLIT_K = 1
MAX_SPLIT_K = 32


def plain_store_role(names, epilogue) -> str:
    """Select direct L0C->GM ``fixpipe`` for plain Cube output, ``mte3`` for AIV epilogue."""
    if epilogue.operations_per_output:
        return "mte3"
    return "fixpipe" if "fixpipe" in names else "mte3"


def _active_aic_cores(context) -> int:
    """Active AIC owner count respecting ``stream_owners`` without inflating ``core_count``."""
    owners = getattr(context, "stream_owners", None)
    if owners is not None and getattr(owners, "aic_limit", None) is not None:
        limit = int(owners.aic_limit)
        if limit <= 0:
            op = context.op
            raise ModelingError(
                f"stream_zero_aic_budget:{getattr(op, 'stream', '') or 'main'}:{getattr(op, 'op_id', 'op')}"
            )
        return min(int(context.core_count), limit)
    return int(context.core_count)


def _stream_aic_units(context) -> tuple[int, ...]:
    """Physical AIC unit IDs assigned to the active stream, or empty when unrestricted."""
    owners = getattr(context, "stream_owners", None)
    if owners is None:
        return ()
    units = context.stream_owner_units("cube")
    if units:
        return tuple(int(u) for u in units)
    return tuple(int(u) for u in owners.units_for("aic"))


def l2_hbm_read_scale(op, *, inputs: float, weight: float, groups: int,
                      group_rows: tuple[int, ...], nn: int, tile_m: int,
                      reuse_inputs: bool, reuse_weights: bool,
                      row_scale_bytes: float = 0.0,
                      column_scale_bytes: float = 0.0) -> float:
    """Fraction of nominal HBM reads remaining after admitted L2 read hits."""
    admitted = max(0.0, float(dict(op.metadata).get("admitted_l2_read_hit_bytes", 0.0) or 0.0))
    if admitted <= 0.0:
        return 1.0
    active_groups = sum(1 for row in group_rows if row > 0)
    weight_factor = (
        active_groups / groups if reuse_weights
        else sum(math.ceil(row / tile_m) for row in group_rows if row > 0) / groups
    )
    full_reads = (
        (inputs + row_scale_bytes) * (1 if reuse_inputs else nn)
        + (weight + column_scale_bytes) * weight_factor
    )
    if full_reads <= 0.0:
        return 1.0
    return max(0.0, 1.0 - min(admitted, full_reads) / full_reads)


def _grouped_owner_rectangles_with_rows(group_rows, n, tile_m, tile_n, cores):
    """Group-major owner rectangles annotated with ``(a, b, group_row_count)``."""
    owners = [[] for _ in range(max(1, int(cores)))]
    index = 0
    for row in group_rows:
        row_int = int(row)
        for start in range(0, row_int, int(tile_m)):
            a = min(int(tile_m), row_int - start)
            for col in range(0, int(n), int(tile_n)):
                b = min(int(tile_n), int(n) - col)
                owners[index % len(owners)].append((a, b, row_int))
                index += 1
    return tuple(tuple(shapes) for shapes in owners)


@dataclass(frozen=True)
class _StreamRefillProgram:
    """RefillPipeline adapter exposing ``bind_stream_owners`` for lazy panels."""

    pipeline: RefillPipeline

    def start(self, engine, name, ready, *, owner=""):
        return self.pipeline.start(engine, name, ready, owner=owner)

    def bind_stream_owners(self, owners, datapath, **kwargs):
        from mindie_sim.prediction.streams import (
            bind_allocations_to_stream, bind_stage_to_stream,
        )
        orig_panel = self.pipeline.panel
        alloc_kw = dict(
            pool_names=kwargs.get("pool_names"),
            pool_units=kwargs.get("pool_units"),
            stream_id=kwargs.get("stream_id", "main"),
        )

        def bound_panel(index):
            load, steps, release = orig_panel(index)
            bound_load = bind_stage_to_stream(load, owners, datapath, **kwargs)
            bound_release = (
                bind_allocations_to_stream(
                    release, owners, datapath, label=f"panel{index}:release",
                    **alloc_kw,
                )
                if release else ()
            )

            def bound_steps():
                for stages, is_last in steps:
                    yield (
                        tuple(bind_stage_to_stream(stage, owners, datapath, **kwargs)
                              for stage in stages),
                        is_last,
                    )

            return bound_load, bound_steps(), bound_release

        return _StreamRefillProgram(replace(self.pipeline, panel=bound_panel))


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
_INDEXED_L0A_SYMBOL = "panel_l0a"
_INDEXED_L0B_SYMBOL = "panel_l0b"
_INDEXED_ACCUMULATOR_SYMBOL = "accumulator"
_INDEXED_SYMBOL_SCOPES = {
    _INDEXED_OPERAND_SYMBOL: "panel",
    _INDEXED_L0A_SYMBOL: "panel",
    _INDEXED_L0B_SYMBOL: "panel",
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
        self.physical_cores = int(context.core_count)
        self.cores = max(1, _active_aic_cores(context))
        self._stream_aic = _stream_aic_units(context)
        self.epilogue = MatrixEpilogue.from_op(op)
        self.input_bytes, self.weight = self.epilogue.operands(op)
        names = context.resource_names
        self.operand_read = read_role(names)
        self.scale_read = read_role(names, aiv=0)
        self.store_role = plain_store_role(names, self.epilogue)
        self.act_width = dtype_bytes(
            metadata.get("activation_format", op.dtype or "bf16"))
        self.compute_width = dtype_bytes(
            metadata.get("compute_format", op.dtype or "bf16"))
        self.output_width = dtype_bytes(
            metadata.get("output_format", "bf16"))
        self._is_aggregate_experts = (
            metadata.get("matrix_mapping_domain") == "aggregate_experts"
        )
        self.hbm_read_scale = l2_hbm_read_scale(
            op, inputs=self.input_bytes, weight=self.weight, groups=self.groups,
            group_rows=(self.m,) * self.groups, nn=self.nn, tile_m=recipe.tile_m,
            reuse_inputs=recipe.reuse_inputs, reuse_weights=recipe.reuse_weights,
            row_scale_bytes=self.epilogue.row_scale_bytes,
            column_scale_bytes=self.epilogue.column_scale_bytes,
        )
        self.operand_pool = context.pool_names.get(recipe.operand_pool)
        self.l0a_pool = context.pool_names.get("l0a")
        self.l0b_pool = context.pool_names.get("l0b")
        self.accumulator_pool = context.pool_names.get(recipe.accumulator_pool)
        excluded = {
            names["hbm"], names.get("mte1"), names.get("mte2"), names.get("mte3"),
            names.get(self.operand_read), names.get(self.scale_read),
            names.get(self.store_role),
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

    def _l0_allocations(self, l0a_symbol, l0b_symbol, left_l0, right_l0, owner_units):
        allocs = []
        if self.l0a_pool is not None and left_l0 > 0:
            allocs.append(Allocation(l0a_symbol, self.l0a_pool, left_l0, owner_units))
        if self.l0b_pool is not None and right_l0 > 0:
            allocs.append(Allocation(l0b_symbol, self.l0b_pool, right_l0, owner_units))
        return tuple(allocs)

    def panel_stages(self, owner, a, b, kd, *, first, last,
                     operand_symbol, accumulator_symbol,
                     l0a_symbol=_INDEXED_L0A_SYMBOL,
                     l0b_symbol=_INDEXED_L0B_SYMBOL):
        """Shared real-body formula for detailed and indexed modes."""
        total_m, n, k, groups = self.total_m, self.n, self.k, self.groups
        if total_m <= 0 or n <= 0 or k <= 0:
            raise ValueError("matrix_panel_requires_positive_geometry")
        ckey = (owner, a, b, kd, first, last)
        cached = self._panel_class_cache.get(ckey)
        if cached is not None:
            (proto_stages, resident, left_l0, right_l0,
             acc_size, owner_units, has_mte1) = cached
            op_pool = self.operand_pool
            acc_pool = self.accumulator_pool
            operand = (() if op_pool is None else (
                Allocation(operand_symbol, op_pool, resident, owner_units),))
            accumulator = (() if acc_pool is None else (
                Allocation(accumulator_symbol, acc_pool, acc_size, owner_units),))
            s0 = proto_stages[0]._with_buffers(operand, ())
            if has_mte1:
                local_l0 = self._l0_allocations(
                    l0a_symbol, l0b_symbol, left_l0, right_l0, owner_units)
                s1 = proto_stages[1]._with_buffers(local_l0, operand)
                s2 = proto_stages[2]._with_buffers(
                    accumulator if first else (), local_l0)
                if not last:
                    return (s0, s1, s2, proto_stages[3]), operand, accumulator, local_l0
                s_last = proto_stages[-1]._with_buffers((), accumulator)
                return (s0, s1, s2, *proto_stages[3:-1], s_last), operand, accumulator, local_l0
            s1 = proto_stages[1]._with_buffers(
                accumulator if first else (), operand)
            if not last:
                return (s0, s1, proto_stages[2]), operand, accumulator, ()
            s_last = proto_stages[-1]._with_buffers((), accumulator)
            if len(proto_stages) == 3:
                return (s0, s1, s_last), operand, accumulator, ()
            return (s0, s1, *proto_stages[2:-1], s_last), operand, accumulator, ()
        recipe, context = self.recipe, self.context
        op, names = context.op, context.resource_names
        phys_owner = (
            self._stream_aic[owner % len(self._stream_aic)]
            if self._stream_aic else owner
        )
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
        reads = (a_read + b_read) * self.hbm_read_scale
        writes = op.mte3_bytes * fraction if last else 0
        load = {names["hbm"]: reads}
        store = {names["hbm"]: writes}
        for role, volume, target in (
                (self.operand_read, a_copy + b_copy, load),
                (self.store_role, writes, store)):
            if role in names:
                target[names[role]] = volume
        # Exact panel share: multiplying by an integer numerator before the
        # single division keeps tile work additive over the whole matrix, so
        # the served FLOP total conserves the operator total bit-for-bit.
        k_numerator = a * b * kd
        k_denominator = total_m * n * k
        compute = {key: value * k_numerator / k_denominator
                   for key, value in self._compute_work_items}
        # Aggregate-expert moments omit per-expert physical shapes; all physical
        # dense and grouped tiles reserve their full operand + scale L1 panel.
        resident = ((a + b) * kd * self.act_width
                    if (groups == 1 and self._is_aggregate_experts)
                    else a_copy + b_copy)
        left_l0 = max(a_copy, a * kd * self.compute_width)
        right_l0 = max(b_copy, b * kd * self.compute_width)
        mte1_work = operand_transfer(context, left_l0, right_l0)
        has_mte1 = bool(mte1_work)
        acc_size = a * b * self._acc_elem_bytes
        owner_units = (phys_owner,)
        operand = (() if self.operand_pool is None else (
            Allocation(operand_symbol, self.operand_pool, resident,
                       owner_units),))
        local_l0 = (
            self._l0_allocations(
                l0a_symbol, l0b_symbol, left_l0, right_l0, owner_units)
            if has_mte1 else ()
        )
        accumulator = (() if self.accumulator_pool is None else (
            Allocation(accumulator_symbol, self.accumulator_pool,
                       acc_size, owner_units),))
        stages = [
            Stage("load", load, acquire=operand,
                  placement=context.owned(phys_owner, self.physical_cores,
                                          (self.operand_read,))
                  if recipe.distributed_transfers else {}),
        ]
        if has_mte1:
            stages.append(Stage(
                "operand", mte1_work, after=("load",),
                acquire=local_l0, free=operand,
                placement=context.owned(phys_owner, self.physical_cores, ("mte1",))
                if recipe.distributed_transfers else {}))
            stages.append(Stage(
                "compute", compute, after=("operand",),
                caps=context.mac_caps,
                placement=context.owned(phys_owner, self.physical_cores,
                                        ("cube", "vector")),
                acquire=accumulator if first else (), free=local_l0))
        else:
            stages.append(Stage(
                "compute", compute, after=("load",),
                caps=context.mac_caps,
                placement=context.owned(phys_owner, self.physical_cores,
                                        ("cube", "vector")),
                acquire=accumulator if first else (), free=operand))
        predecessor = "compute"
        if last and self.epilogue.operations_per_output:
            scale_copy, scale_hbm = self.epilogue.scales(
                a, b, total_m, n, groups,
                reuse_inputs=recipe.reuse_inputs,
                reuse_weights=recipe.reuse_weights)
            read = self.scale_read
            scale_work = {names["hbm"]: scale_hbm * self.hbm_read_scale}
            if read in names:
                scale_work[names[read]] = (
                    scale_work.get(names[read], 0) + scale_copy)
            stages.extend((
                Stage("scales", scale_work, after=("compute",),
                      placement=context.owned(phys_owner, self.physical_cores, (read,))),
                Stage("epilogue", {
                    names["vector"]:
                        self.epilogue.operations_per_output * a * b},
                    after=("scales",),
                    placement=context.owned(phys_owner, self.physical_cores,
                                            ("vector",))),
            ))
            predecessor = "epilogue"
        stages.append(Stage(
            "store", store, after=(predecessor,),
            placement=context.owned(phys_owner, self.physical_cores, (self.store_role,))
            if recipe.distributed_transfers else {},
            free=accumulator if last else ()))
        proto_stages = tuple(stages)
        self._panel_class_cache[ckey] = (
            proto_stages, resident, left_l0, right_l0,
            acc_size, owner_units, has_mte1,
        )
        return proto_stages, operand, accumulator, local_l0

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
                stages, _operand, _accumulator, _local_l0 = self.panel_stages(
                    owner, a, b, kd, first=first, last=last,
                    operand_symbol=_INDEXED_OPERAND_SYMBOL,
                    accumulator_symbol=_INDEXED_ACCUMULATOR_SYMBOL,
                    l0a_symbol=_INDEXED_L0A_SYMBOL,
                    l0b_symbol=_INDEXED_L0B_SYMBOL)
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
            stages, operand, accumulator, local_l0 = geometry.panel_stages(
                core, a, b, kd, first=first, last=last,
                operand_symbol=f"panel{index}",
                accumulator_symbol=f"accumulator{wave}",
                l0a_symbol=f"panel_l0a_{index}",
                l0b_symbol=f"panel_l0b_{index}")
            if operand:
                registry.record(operand[0], "panel", index)
            for alloc in local_l0:
                registry.record(alloc, "panel", index)
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
        groups, names, scale_read, store_role, hbm_read_scale, transfer, writes,
        accumulator,
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
                Stage("scales", transfer(scale_read, scale_copy, hbm=scale_hbm * hbm_read_scale),
                      after=("matrix",), placement=context.owned(core, cores, (scale_read,))),
                Stage("epilogue", {names["vector"]: epilogue.operations_per_output * a * b},
                      after=("scales",), placement=context.owned(core, cores, ("vector",))),
            ))
        stages.append(Stage(
            "store", transfer(store_role, writes, hbm=writes),
            after=("epilogue",) if epilogue.operations_per_output else ("matrix",),
            free=accumulator, placement=context.owned(core, cores, (store_role,))))

    def _build_panel(
        self, context, *, core, phys_core, physical_cores, active_cores, index,
        tile_sources, nm, nn, nk, m, n, k, total_m, groups, active_groups,
        inputs, weight, hbm_read_scale, epilogue, output, operand_read,
        scale_read, store_role, compute_width, output_width, work, names,
        directional_mte1,
    ):
        op = context.op
        wave, kk = divmod(index, nk)
        group_row_count = None
        if tile_sources is None:
            tile = wave * active_cores + core
            a = min(self.tile_m, m - ((tile // nn) % nm) * self.tile_m)
            b = min(self.tile_n, n - (tile % nn) * self.tile_n)
            group_row_count = m
        else:
            shape = tile_sources[core][wave]
            if len(shape) == 3:
                a, b, group_row_count = shape
            else:
                a, b = shape
        kd = min(self.tile_k, k - kk * self.tile_k)
        fraction, k_fraction = a * b / (total_m * n), kd / k
        a_copy = inputs * a / total_m * k_fraction
        b_copy = weight * b / (n * groups) * k_fraction
        a_read = inputs * fraction * k_fraction if self.reuse_inputs else a_copy
        if self.reuse_weights:
            if group_row_count is not None and group_row_count > 0:
                b_read = b_copy * (a / group_row_count)
            else:
                b_read = weight * (active_groups / groups) * fraction * k_fraction
        else:
            b_read = b_copy
        reads = (a_read + b_read) * hbm_read_scale
        writes = op.mte3_bytes * fraction if kk == nk - 1 else 0

        def allocation(name, pool, size):
            return (() if pool not in context.pool_names else
                    (Allocation(name, context.pool_names[pool], size, (phys_core,)),))

        def transfer(role, volume, *, hbm=None):
            result = {names[role]: volume} if role in names else {}
            if hbm is not None:
                result[names["hbm"]] = hbm
            return result

        operand = allocation(f"panel{index}", "l1", a_copy + b_copy)
        accumulator = allocation(
            f"accumulator{wave}", "l0c", a * b * max(FP32_BYTES, output_width))
        inner_n = math.ceil(b / self.compute_n)

        def steps():
            count = math.ceil(a / self.compute_m) * inner_n
            for part in range(count):
                ai = min(self.compute_m, a - part // inner_n * self.compute_m)
                bi = min(self.compute_n, b - part % inner_n * self.compute_n)
                left = max(a_copy * ai / a, ai * kd * compute_width)
                right = max(b_copy * bi / b, bi * kd * compute_width)
                local = (
                    *allocation(f"a{index}:{part}", "l0a", left),
                    *allocation(f"b{index}:{part}", "l0b", right),
                )
                demand = {name: value * ai * bi / (total_m * n) * k_fraction
                          for name, value in work.items()}
                stages = [
                    Stage("operand", operand_transfer(
                        context, left, right, directional=directional_mte1),
                        acquire=local,
                        placement=context.owned(phys_core, physical_cores, ("mte1",))),
                    Stage("matrix", demand, after=("operand",), free=local,
                          acquire=accumulator if kk == 0 and part == 0 else (),
                          placement=context.owned(phys_core, physical_cores,
                                                  ("cube", "vector"))),
                ]
                if part == count - 1 and kk == nk - 1:
                    self._append_panel_output(
                        stages, output=output, epilogue=epilogue, context=context,
                        core=phys_core, cores=physical_cores, a=a, b=b,
                        total_m=total_m, n=n, groups=groups, names=names,
                        scale_read=scale_read, store_role=store_role,
                        hbm_read_scale=hbm_read_scale,
                        transfer=transfer, writes=writes, accumulator=accumulator)
                yield tuple(stages), part == count - 1

        load_stage = Stage(
            "load", transfer(operand_read, a_copy + b_copy, hbm=reads),
            acquire=operand,
            placement=context.owned(phys_core, physical_cores, (operand_read,)))
        return load_stage, steps(), operand

    def lower(self, context, *, owner_tiles=None, group_rows=None, output=None,
              serial_tiles=False, directional_mte1=False):
        if min(self.tile_m, self.tile_n, self.tile_k, self.compute_m,
               self.compute_n, self.slots, self.compute_slots) < 1:
            raise ValueError("matrix_tile_dimensions_and_slots_must_be_positive")
        if self.tile_m % self.compute_m or self.tile_n % self.compute_n:
            raise ValueError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
        from .matrix_mapping import (freeze_owner_tiles, matrix_geometry,
                                     owner_records, splitk_executable)
        if (self.k_splits > DEFAULT_SPLIT_K and output is None
                and owner_tiles is None
                and (group_rows is None or len(group_rows) == 1)
                and splitk_executable(context, self)):
            from .matrix_splitk import splitk_program
            return splitk_program(context, self.k_splits, recipe=self)
        physical_cores = int(context.core_count)
        active_cores = max(1, _active_aic_cores(context))
        stream_aic = _stream_aic_units(context)
        geometry_ctx = (
            replace(context, core_count=active_cores)
            if active_cores != physical_cores else context
        )
        owner_tiles = (freeze_owner_tiles(owner_tiles)
                       if owner_tiles is not None else None)
        op, names = context.op, context.resource_names
        tile_sources = None
        if group_rows is not None and len(op.local_shape) >= 3:
            annotated = _grouped_owner_rectangles_with_rows(
                group_rows, op.local_shape[-2], self.tile_m, self.tile_n,
                active_cores)
            nonempty_annotated = tuple(shapes for shapes in annotated if shapes)
            plain_grouped = freeze_owner_tiles(tuple(
                (len(shapes), tuple((a, b) for a, b, _ in shapes))
                for shapes in nonempty_annotated
            ))
            if owner_tiles is None:
                owner_tiles = plain_grouped
                tile_sources = list(nonempty_annotated)
            elif owner_tiles == plain_grouped:
                tile_sources = list(nonempty_annotated)
        explicit = (owner_records(owner_tiles) if owner_tiles is not None else None)
        geometry = matrix_geometry(geometry_ctx, self, owners=explicit,
                                   group_rows=group_rows)
        total_m, n, k = geometry.local_shape
        metadata = dict(op.metadata)
        groups, m = geometry.groups, geometry.m_per_group
        if not total_m:
            return KernelProgram((Stage("setup", latency=context.setup_ms),),
                                 minimum_ms=context.minimum_ms)
        nm, nn, nk = geometry.nm, geometry.nn, geometry.nk
        active_groups = sum(1 for row in geometry.group_rows if row > 0)
        epilogue = MatrixEpilogue.from_op(op)
        inputs, weight = epilogue.operands(op)
        hbm_read_scale = l2_hbm_read_scale(
            op, inputs=inputs, weight=weight, groups=groups,
            group_rows=geometry.group_rows, nn=nn, tile_m=self.tile_m,
            reuse_inputs=self.reuse_inputs, reuse_weights=self.reuse_weights,
            row_scale_bytes=epilogue.row_scale_bytes,
            column_scale_bytes=epilogue.column_scale_bytes,
        )
        operand_read = read_role(names)
        scale_read = read_role(names, aiv=0)
        store_role = plain_store_role(names, epilogue)
        compute_width = dtype_bytes(metadata.get("compute_format", op.dtype or "bf16"))
        output_width = dtype_bytes(metadata.get("output_format", "bf16"))
        excluded = {
            names["hbm"], names.get("mte1"), names.get("mte2"), names.get("mte3"),
            names.get(operand_read), names.get(scale_read), names.get(store_role),
        }
        work = {name: value for name, value in context.work.items()
                if name not in excluded}
        if epilogue.operations_per_output:
            work.pop(names["vector"], None)
        if owner_tiles is None:
            counts = tuple(record[0] for record in geometry.owners)
        else:
            counts = tuple(count for count, _ in owner_tiles)
            if tile_sources is None:
                tile_sources = [tuple(tiles) for _, tiles in owner_tiles]
        programs = []
        for core, count in enumerate(counts):
            if not count:
                continue
            phys_core = stream_aic[core % len(stream_aic)] if stream_aic else core
            program = _StreamRefillProgram(RefillPipeline(
                count * nk, self.slots, self.compute_slots,
                lambda index, core=core, phys_core=phys_core: self._build_panel(
                    context, core=core, phys_core=phys_core,
                    physical_cores=physical_cores, active_cores=active_cores,
                    index=index, tile_sources=tile_sources, nm=nm, nn=nn, nk=nk,
                    m=m, n=n, k=k, total_m=total_m, groups=groups,
                    active_groups=active_groups, inputs=inputs, weight=weight,
                    hbm_read_scale=hbm_read_scale, epilogue=epilogue,
                    output=output, operand_read=operand_read,
                    scale_read=scale_read, store_role=store_role,
                    compute_width=compute_width, output_width=output_width,
                    work=work, names=names, directional_mte1=directional_mte1,
                ), nk if serial_tiles else None))
            if output is not None:
                buffers = output.allocations(
                    context, phys_core, (self.tile_m, self.tile_n), epilogue)
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
