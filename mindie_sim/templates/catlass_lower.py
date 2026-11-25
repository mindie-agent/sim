"""Executable Tier K2 CATLASS schedules.

[Role] Lower tiled matrix, split-K, epilogue and SIMT specializations.
[Invariants] K slices partition the product. GM, on-chip and engine work
  retain distinct boundaries and actual tile tails.
[Timeouts & Locks] Finite declared tile/ring schedules. No I/O or locks.
[Side-Effect Policy] Uses the shared CPU runtime and optional admission view.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import lru_cache
from math import prod

from mindie_sim.prediction.program.stages import Stage
from mindie_sim.prediction.runtime.memory_physics import AddressRun
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import (
    DEFAULT_INST_BLOCK_DIM, LiveBufferSpec, TileCandidate, TileShape,
)
from mindie_sim.schema.common import ModelingError

from .catlass_catalog import (
    GATHER_SCATTER_AIV_MAX_K, GATHER_SCATTER_FULL_LOAD_A, GATHER_SCATTER_POLICIES, GATHER_SCATTER_SIMT_UB_BYTES,
    MX_SCALE_GROUP, STREAMK_PARTIAL_SLOTS_PER_CORE, _workspace_bytes, dtype_name, grouped_start_cores,
    matrix_problem, mx_scale_bytes, packed_bytes, should_elide_l1_reload,
    streamk_partition, swizzle_block_coord, validate_catlass_spec,
)
from ._catlass_streaming import STREAMING_PANEL_MICRO_THRESHOLD, lower_catlass_streaming
from .contracts import SemanticWorkItem, ceil_div, eligible_matrix_engines, evg_order
from .fuse_region import Lowering, ring_credit, solve_template_tile

_NOMINAL_BUFFER_ALIGNMENT = 32
_PURE_LOWER_CACHE_SIZE = 128
SIMT_WARP_SIZE = 32
A8W4_PROLOGUE_SUBBLOCK_ALIGNMENT = 32
A8W4_PROLOGUE_UB_SLOT_BYTES = 2 * 32 * 1024
GATHER_SCATTER_FULL_LOAD_UB_BYTES = 216 * 1024
_SFU_NODES = {"exp": 1, "sqrt": 1, "rsqrt_fast": 1, "silu": 1}
_ALU_NODES = {"relu": 1, "leaky_relu": 2, "silu": 2, "cast": 1, "mul": 1, "muls": 1, "add": 1}


def _paired_aiv(spec, datapath, owner):
    candidates = tuple(aiv for aiv in dict(datapath.aic_aiv_pairs).get(owner, ()) if aiv < spec.aiv_budget)
    if not candidates:
        raise ModelingError(f"catlass_epilogue_owner_unpaired:{owner}")
    return candidates[0]


def _is_simt_matrix(spec):
    return spec.simt_thread_block_dim is not None and spec.dispatch_policy != "GroupedMxMatmulFinalizeRoutingTla"


def _mx_scale_group_span(k0: int, k_len: int) -> tuple[int, int]:
    """Return ``(first_group, group_count)`` touched by ``[k0, k0 + k_len)``."""
    if k_len <= 0:
        return 0, 0
    g_lo = k0 // MX_SCALE_GROUP
    g_hi = (k0 + k_len - 1) // MX_SCALE_GROUP
    return g_lo, g_hi - g_lo + 1


def _scale(spec, elements, *, l1=False):
    if not spec.mx_scale_group:
        return 0
    if l1:
        return mx_scale_bytes(1, elements, spec.l1_scale_factor_k)
    return ceil_div(elements, MX_SCALE_GROUP)


def _compute_b(spec):
    return dtype_name(spec.dtype_a) if spec.prologue_b == "DecompressFp4ToFp8" else dtype_name(spec.dtype_b)


def _cube_dtype(spec):
    return dtype_name(spec.dtype_a)


def _visitor_buffers(spec, tm, tn):
    nodes = evg_order(spec.evg_tree)
    alias = {node.node_id: "accumulator" if node.kind == "acc_load" else node.node_id for node in nodes}
    ends = defaultdict(int)
    for index, node in enumerate(nodes, 1):
        for source in node.inputs:
            ends[alias[source]] = max(ends[alias[source]], index + 1)
    final = len(nodes) + 1
    elements = tm * tn
    buffers = [LiveBufferSpec(
        "visitor:accumulator", "ub", packed_bytes(elements, spec.dtype_acc), 0,
        ends.get("accumulator", final), owner_kind="aiv", slots=2,
        alignment_bytes=_NOMINAL_BUFFER_ALIGNMENT,
    )]
    for index, node in enumerate(nodes, 1):
        if node.kind in {"acc_load", "aux_store"}:
            continue
        count = tn if node.kind == "row_broadcast" else tm if node.kind == "col_broadcast" else elements
        buffers.append(LiveBufferSpec(
            f"visitor:{node.node_id}", "ub", packed_bytes(count, node.dtype), index,
            ends.get(node.node_id, final), owner_kind="aiv", slots=2,
            alignment_bytes=_NOMINAL_BUFFER_ALIGNMENT,
        ))
    return tuple(buffers)


def _axis_tile_counts(extent: int, tile: int) -> tuple[tuple[int, int], ...]:
    full, rem = divmod(extent, tile)
    out = []
    if full > 0:
        out.append((tile, full))
    if rem > 0:
        out.append((rem, 1))
    return tuple(out)


def _total_issued_ops(spec, p, datapath) -> int:
    """Compute exact total issued Cube FLOPs across full and tail microtiles."""
    if _is_simt_matrix(spec):
        return 0
    tm, tn, tk = spec.l1_tile_shape
    micro_k = spec.l0_tile_shape[-1]
    m_counts: Counter[int] = Counter()
    for rows in (p.groups if p.groups else (p.m,)):
        if rows > 0:
            for dim, mult in _axis_tile_counts(rows, tm):
                m_counts[dim] += mult
    n_counts: Counter[int] = Counter()
    if spec.epilogue_kind == "grouped_swiglu_mx_quant":
        for dim, mult in _axis_tile_counts(p.n // 2, tn):
            n_counts[dim] += 2 * mult
    else:
        for dim, mult in _axis_tile_counts(p.n, tn):
            n_counts[dim] += mult
    k_counts: Counter[int] = Counter()
    for k_len, panel_mult in _axis_tile_counts(p.k, tk):
        for u_k, u_mult in _axis_tile_counts(k_len, micro_k):
            k_counts[u_k] += panel_mult * u_mult
    total = 0
    for m_dim, mc in m_counts.items():
        for n_dim, nc in n_counts.items():
            for u_k, kc in k_counts.items():
                total += mc * nc * kc * _issued_ops(m_dim, n_dim, u_k, spec, datapath)
    return total


def solve_catlass_tile(spec, problem, datapath, *, memory_view=None, die=0):
    p = matrix_problem(problem)
    validate_catlass_spec(spec, p, datapath)
    tm, tn, tk = spec.l1_tile_shape
    lm, ln, lk = spec.l0_tile_shape
    g_m = sum(ceil_div(rows, tm) for rows in p.groups) if p.groups else ceil_div(p.m, tm)
    g_n = ceil_div(p.n, tn)
    simt = _is_simt_matrix(spec)
    slots = spec.stages
    sizes = (
        ("l1a", "l1", tm * tk, spec.dtype_a, slots[0]),
        ("l1b", "l1", tn * tk, _compute_b(spec), slots[1]),
        ("l0a", "l0a", lm * lk, spec.dtype_a, slots[2]),
        ("l0b", "l0b", ln * lk, _compute_b(spec), slots[-2]),
        ("l0c", "l0c", lm * ln, spec.dtype_acc, slots[-1]),
    )
    buffers = tuple(
        LiveBufferSpec(name, pool, packed_bytes(elements, dtype), 0, 1, owner_kind="aic", slots=depth,
                       scale_bytes=_scale(spec, elements, l1=pool == "l1") if pool != "l0c" else 0)
        for name, pool, elements, dtype, depth in sizes
    ) if not simt else ()
    total_elements = tm * tn
    if spec.has_bias and not simt:
        buffers += (LiveBufferSpec("bias", "l1", packed_bytes(tn, spec.dtype_acc), 0, 1, owner_kind="aic"),)
    if spec.evg_tree:
        buffers += _visitor_buffers(spec, tm, tn)
    if spec.prologue_b == "DecompressFp4ToFp8":
        buffers += (LiveBufferSpec(
            "packed_and_decompressed_b", "ub", A8W4_PROLOGUE_UB_SLOT_BYTES, 0, 1,
            owner_kind="aiv", slots=slots[1],
        ),)
        for aic in range(min(spec.aic_budget, g_m * g_n)):
            lanes = tuple(v for v in dict(datapath.aic_aiv_pairs).get(aic, ()) if v < spec.aiv_budget)
            if len(lanes) != 2:
                raise ModelingError(f"catlass_a8w4_requires_two_subblocks:{aic}")
            for lane in lanes:
                datapath.route("ub", "l1", _compute_b(spec), (lane, aic))
    if spec.epilogue_kind == "grouped_swiglu_mx_quant":
        peak = 2 * packed_bytes(total_elements, spec.dtype_acc) + packed_bytes(total_elements, spec.dtype_c)
        buffers += (LiveBufferSpec("gate_up", "ub", peak, 0, 1, owner_kind="aiv", slots=2),)
    if simt:
        threads = prod(spec.simt_thread_block_dim)
        pinned = spec.dispatch_policy == "MatmulGatherScatterSimt"
        ub_bytes = GATHER_SCATTER_SIMT_UB_BYTES if pinned else packed_bytes(threads, spec.dtype_acc)
        buffers += (LiveBufferSpec("simt_kernel_ub", "ub", ub_bytes, 0, 1, owner_kind="aiv"),)
    if spec.dispatch_policy == GATHER_SCATTER_FULL_LOAD_A:
        buffers += (LiveBufferSpec("full_load_gather_scatter", "ub", GATHER_SCATTER_FULL_LOAD_UB_BYTES,
                                   0, 1, owner_kind="aiv"),)
        for core in range(min(spec.aic_budget, g_m * g_n)):
            lane = _paired_aiv(spec, datapath, core)
            datapath.route("l0c", "ub", spec.dtype_acc, (core, lane), (tm, tn))
            if p.k <= GATHER_SCATTER_AIV_MAX_K:
                datapath.route("ub", "l1", spec.dtype_a, (lane, core))
    eff_g_n = ceil_div(p.n // 2, tn) if spec.epilogue_kind == "grouped_swiglu_mx_quant" else g_n
    if spec.epilogue_kind in {"evg_ub", "grouped_swiglu_mx_quant"}:
        owners = min(spec.aic_budget, g_m * eff_g_n)
        for aic in range(owners):
            datapath.route("l0c", "ub", dtype_name(spec.dtype_acc),
                           (aic, _paired_aiv(spec, datapath, aic)), (tm, tn))
    work_items = (
        p.m * ceil_div(p.n, SIMT_WARP_SIZE)
        if spec.dispatch_policy == "MatmulGatherScatterSimt"
        else g_m * eff_g_n * spec.splitk_factor
    )
    owner_budget = spec.aiv_budget if simt else spec.aic_budget
    candidate = TileCandidate(
        "resident_1pass", TileShape(lm, ln, lk), max(slots), (spec.aic_budget, spec.aiv_budget),
        work_items, ceil_div(work_items, owner_budget),
        buffers, logical_flops=2 * p.m * p.n * p.k,
        issued_flops=_total_issued_ops(spec, p, datapath),
        uses_simt=spec.simt_thread_block_dim is not None,
        native_mode=_cube_dtype(spec), owner_axis="aiv" if simt else "aic",
    )
    return solve_template_tile(datapath, spec.template_id, candidate, state=memory_view, die=die)


@dataclass(frozen=True, slots=True)
class _ProductTile:
    group: int
    row: int
    group_row: int
    col: int
    m: int
    n: int
    owner: int
    group_rows: int


class _MatrixLowerer:
    def __init__(self, plan, spec, problem):
        self.plan = plan
        self.spec = spec
        self.p = problem
        self.tm, self.tn, self.tk = spec.l1_tile_shape
        self.micro_k = spec.l0_tile_shape[-1]
        self.gk = ceil_div(problem.k, self.tk)
        self.resident = {}
        self.l1_generations = defaultdict(int)
        self.owner_rings = {}
        self.owner_generation = defaultdict(int)
        self.owner_micro = defaultdict(int)
        self.mma_tail = {}
        self.fixpipe_tail = {}
        self._tile_index = 0

    def tile(self, index, *, group=0, group_start=0, rows=None, start_core=0, n_cols=None):
        rows = self.p.m if rows is None else rows
        n_cols = self.p.n if n_cols is None else n_cols
        mi, ni = swizzle_block_coord(
            index, ceil_div(rows, self.tm), ceil_div(n_cols, self.tn),
            self.spec.swizzle_direction, self.spec.swizzle_offset,
        )
        row, col = mi * self.tm, ni * self.tn
        owner_budget = self.spec.aiv_budget if _is_simt_matrix(self.spec) else self.spec.aic_budget
        return _ProductTile(group, group_start + row, row, col,
                            min(self.tm, rows - row), min(self.tn, n_cols - col),
                            (index + start_core) % owner_budget, rows)

    def _rect(self, tensor, dtype, row, col, rows, cols, full_row, **kwargs):
        layout = self.spec.layout_a if tensor == self.p.a_id else self.spec.layout_b
        if layout == "ColumnMajor":
            row, col, rows, cols = col, row, cols, rows
            full_row = self.p.m if tensor == self.p.a_id else self.p.k
        elif layout != "RowMajor":
            raise ModelingError(f"catlass_address_layout_unsupported:{layout}")
        # Each packed row starts at its own independently aligned bit offset.
        stride = packed_bytes(full_row, dtype)
        width = packed_bytes(col + cols, dtype) - packed_bytes(col, dtype)
        if width != packed_bytes(cols, dtype):
            raise ModelingError("catlass_packed_tile_unaligned")
        base = self.p.b_base_byte if tensor == self.p.b_id else 0
        run = AddressRun(base + row * stride + packed_bytes(col, dtype), rows, width, stride, layout)
        if tensor == self.p.b_id and "version" not in kwargs:
            kwargs["version"] = self.p.b_version
        return self.plan.access(kwargs.pop("name"), kwargs.pop("allocation", tensor), width * rows, runs=(run,), **kwargs)

    def _fill(self, name, tile, k_start, k_len, operand, after, ring, generation):
        p, s, core = self.p, self.spec, tile.owner
        for field_name in ("dtype_a", "dtype_b", "dtype_c", "dtype_acc"):
            if dtype_name(getattr(s, field_name)) != dtype_name(getattr(p, field_name)):
                raise ModelingError(f"catlass_spec_problem_{field_name}_mismatch")
        if bool(s.has_bias) != bool(p.has_bias):
            raise ModelingError("catlass_spec_problem_bias_mismatch")
        a = operand == "a"
        tensor, dtype, row, col, rows, cols, width = (
            (p.a_id, p.dtype_a, tile.row, k_start, tile.m, k_len, p.k) if a else
            (p.b_id, p.dtype_b, k_start, tile.col, k_len, tile.n, p.n)
        )
        allocation = tensor if a or not p.groups else f"{tensor}:expert:{tile.group}"
        cache_bypass = not a and (s.l2_cache_hint_b == "disable" or (
            s.scheduler == "grouped_slice_m" and ceil_div(tile.group_rows, self.tm) == 1
        ))
        location = (allocation, row, col, rows, cols)
        lane = (core, operand)
        slot = self.l1_generations[lane] % ring.depth
        self.l1_generations[lane] += 1
        previous = self.resident.get((*lane, slot))
        credit = ring_credit(ring, generation)
        deps = (*after, credit)
        elide = previous is not None and should_elide_l1_reload(
            previous[0], previous[1:], location[0], location[1:], s.enable_l1_resident,
        )
        if elide:
            return deps
        self.resident[*lane, slot] = location
        if a and s.dispatch_policy == GATHER_SCATTER_FULL_LOAD_A:
            vector = self._aiv(tile, direct=True)
            indices = self._index_read(f"{name}:gather", tile, deps, vector)
            aiv_gather = p.k <= GATHER_SCATTER_AIV_MAX_K
            stride = packed_bytes(k_len, dtype)
            read = self.plan.access(
                f"{name}:gm_a", tensor, tile.m * stride, after=(indices,),
                owner=vector if aiv_gather else core, pipe="mte2_aiv" if aiv_gather else "mte2_aic",
                runs=tuple(AddressRun(i * stride, 1, stride, stride) for i in self._physical_rows(tile)),
            )
            if aiv_gather:
                read = self.plan.work(f"{name}:gather_to_l1", "mte3", tile.m * stride, after=(read,), owner=vector)
            return (read,)
        if not a and s.prologue_b == "DecompressFp4ToFp8":
            deps = self._expand_b(name, tile, row, rows, allocation, deps, cache_bypass)
        else:
            deps = (self._rect(
                tensor, dtype, row, col, rows, cols, width,
                name=f"{name}:gm_{operand}", allocation=allocation, owner=core, pipe="mte2_aic",
                after=deps, bypass=cache_bypass,
            ),)
        if s.mx_scale_group:
            per_row = ceil_div(width, MX_SCALE_GROUP) if a else ceil_div(p.k, MX_SCALE_GROUP)
            g_lo, sc = _mx_scale_group_span(col, cols) if a else _mx_scale_group_span(row, rows)
            sr = rows if a else cols
            off = row * per_row + g_lo if a else col * per_row + g_lo
            deps = (self.plan.access(
                f"{name}:scale_{operand}", f"{allocation}:scale", sr * sc, owner=core,
                pipe="mte2_aic", after=deps, bypass=cache_bypass,
                runs=(AddressRun(off, sr, sc, per_row),),
            ),)
        return deps

    def _expand_b(self, name, tile, row, rows, allocation, after, bypass):
        plan, spec = self.plan, self.spec
        granule = A8W4_PROLOGUE_SUBBLOCK_ALIGNMENT
        first = min(rows, ceil_div(rows // 2, granule) * granule) if row + rows == self.p.k else rows // 2
        lanes = tuple(v for v in dict(plan.datapath.aic_aiv_pairs).get(tile.owner, ()) if v < spec.aiv_budget)
        outputs = []
        for lane, offset, count in zip(lanes, (0, first), (first, rows - first)):
            if count <= 0:
                continue
            tag = f"{name}:b:{lane}"
            source = self._rect(
                self.p.b_id, spec.dtype_b, row + offset, tile.col, count, tile.n, self.p.n,
                name=tag + ":gm", allocation=allocation, owner=lane, pipe="mte2_aiv", after=after, bypass=bypass,
            )
            convert = plan.work(tag + ":decompress_b", "vec_alu", count * tile.n,
                                owner=lane, after=(source,), metadata={"prologue": spec.prologue_b})
            outputs.append(plan.work(tag + ":ub_to_l1", "mte3", packed_bytes(count * tile.n, _compute_b(spec)),
                                     owner=lane, after=(convert,)))
        return tuple(outputs)

    def product(self, tile, panels, after):
        p, s, plan = self.p, self.spec, self.plan
        name = f"tile:{self._tile_index}"
        self._tile_index += 1
        core = tile.owner
        l1m, l1n, l1k = s.l1_tile_shape
        l0m, l0n, l0k = s.l0_tile_shape
        mutex = s.dispatch_policy == "MmadPingpongMutex"
        rings = self.owner_rings.get(core) if mutex else None
        if rings is None:
            prefix = f"owner:{core}" if mutex else name
            rings = tuple(
                RingBuffer(
                    die=plan.die, owner=core, pool=pool, allocation=f"{prefix}:{tag}", depth=depth,
                    slot_bytes=packed_bytes(elements, dtype)
                    + (_scale(s, elements, l1=pool == "l1") if pool != "l0c" else 0)
                    if plan.context is not None else 0,
                )
                for (tag, pool, elements, dtype), depth in zip((
                    ("l1a", "l1", l1m * l1k, s.dtype_a), ("l1b", "l1", l1n * l1k, _compute_b(s)),
                    ("l0a", "l0a", l0m * l0k, s.dtype_a), ("l0b", "l0b", l0n * l0k, _compute_b(s)),
                    ("l0c", "l0c", l0m * l0n, s.dtype_acc),
                ), s.stages)
            )
            if mutex:
                self.owner_rings[core] = rings
        l1a, l1b, l0a, l0b, accum = rings
        generation = self.owner_generation[core] if mutex else 0
        panel_start = self.l1_generations[core, "a"] if mutex else 0
        micro_start = self.owner_micro[core] if mutex else 0
        acc_credit = ring_credit(accum, generation)
        previous = (*after, acc_credit)
        if mutex:
            self.owner_generation[core] += 1
            if core in self.mma_tail:
                previous += (self.mma_tail[core],)
        if s.has_bias:
            previous = (plan.access(
                f"{name}:bias", p.c_id + ":bias", packed_bytes(tile.n, s.dtype_acc),
                owner=core, pipe="mte2_aic", offset=packed_bytes(tile.col, s.dtype_acc), after=previous,
            ),)
        g = 0
        for outer, panel in enumerate(panels, panel_start):
            k_start = panel * self.tk
            k_len = min(self.tk, p.k - k_start)
            last_load = f"{name}:micro:{g + ceil_div(k_len, self.micro_k) - 1}:loaded"
            r1 = []
            for operand, ring in (("a", l1a), ("b", l1b)):
                deps = self._fill(f"{name}:k{panel}", tile, k_start, k_len, operand, after, ring, outer)
                r1.append(lower_generation(
                    ring, outer, "mte2", ("mte1",), producer_extra_after=deps,
                    consumer_extra_after={"mte1": (last_load,)},
                ).publish)
            for start in range(0, k_len, self.micro_k):
                micro = min(k_len - start, self.micro_k)
                mma = f"{name}:micro:{g}:mma"
                slot_generation = micro_start + g
                r0 = []
                for rows, dtype, ring, prior in (
                    (tile.m, s.dtype_a, l0a, r1[0]), (tile.n, _compute_b(s), l0b, r1[1]),
                ):
                    moved = packed_bytes(rows * micro, dtype) + (
                        rows * _mx_scale_group_span(k_start + start, micro)[1] if s.mx_scale_group else 0
                    )
                    load = plan.work(f"{ring.allocation}:g{slot_generation}:load", "mte1", moved, owner=core,
                                     after=(prior, ring_credit(ring, slot_generation)))
                    r0.append(lower_generation(
                        ring, slot_generation, "mte1", ("mma",), producer_extra_after=(load,),
                        consumer_extra_after={"mma": (mma,)},
                    ).publish)
                loaded = plan.add(Stage(f"{name}:micro:{g}:loaded", after=tuple(r0)))
                issued = _issued_ops(tile.m, tile.n, micro, s, plan.datapath)
                previous = (plan.work(mma, "cube", issued, after=(*previous, loaded), owner=core,
                                      dtype=_cube_dtype(s), metadata={"logical_flops": 2 * tile.m * tile.n * micro}),)
                g += 1
        event = lower_generation(
            accum, generation, "mma", ("fixpipe",), producer_extra_after=previous,
            consumer_extra_after={"fixpipe": (f"{name}:accumulator_consumed",)},
        )
        if mutex:
            self.owner_micro[core] += g
            self.mma_tail[core] = previous[0]
        else:
            for ring in rings:
                plan.add_ring(ring, after=after)
        return name, event.publish

    def _consume(self, name, dependency):
        return self.plan.add(Stage(f"{name}:accumulator_consumed", after=(dependency,)))

    def _write_output(self, name, tile, after, *, pipe="fixpipe", owner=None, width=None, physical_rows=None):
        dtype = self.spec.dtype_c
        full_n = self.p.n if width is None else width
        if self.spec.layout_c == "RowMajor":
            stride = packed_bytes(full_n, dtype)
            row_bytes = packed_bytes(tile.n, dtype)
            col = packed_bytes(tile.col, dtype)
            runs = ((AddressRun(tile.row * stride + col, tile.m, row_bytes, stride),) if physical_rows is None
                    else tuple(AddressRun(row * stride + col, 1, row_bytes, stride) for row in physical_rows))
            amount = row_bytes * tile.m
        elif self.spec.layout_c == "ColumnMajor":
            stride = packed_bytes(self.p.m, dtype)
            col_bytes = packed_bytes(tile.m, dtype)
            runs = (AddressRun(tile.col * stride + packed_bytes(tile.row, dtype), tile.n, col_bytes, stride),)
            amount = col_bytes * tile.n
        else:
            raise ModelingError(f"catlass_address_layout_unsupported:{self.spec.layout_c}")
        return self.plan.access(name, self.p.c_id, amount, runs=runs, direction="write",
                                pipe=pipe, owner=tile.owner if owner is None else owner, after=after)

    def _aiv(self, tile, direct=False):
        return _paired_aiv(self.spec, self.plan.datapath, tile.owner) if direct else tile.owner % self.spec.aiv_budget

    def finish(self, tile, name, ready):
        s, p, plan = self.spec, self.p, self.plan
        epilogue = s.epilogue_kind
        if s.dispatch_policy == GATHER_SCATTER_FULL_LOAD_A:
            aiv = self._aiv(tile, direct=True)
            local = plan.work(name + ":fixpipe_ub", "fixpipe", packed_bytes(tile.m * tile.n, s.dtype_acc),
                              after=(ready,), owner=tile.owner)
            self._consume(name, local)
            return self._write_output(name + ":scatter", tile, (local,), pipe="mte3", owner=aiv,
                                      physical_rows=self._physical_rows(tile))
        if epilogue == "none":
            after = (ready,)
            if s.dispatch_policy == "MmadPingpongMutex" and tile.owner in self.fixpipe_tail:
                after += (self.fixpipe_tail[tile.owner],)
            out = self._write_output(name + ":fixpipe", tile, after)
            if s.dispatch_policy == "MmadPingpongMutex":
                self.fixpipe_tail[tile.owner] = out
            self._consume(name, out)
            return out
        if epilogue == "finalize_routing":
            aiv = self._aiv(tile)
            amount = packed_bytes(tile.m * tile.n, s.dtype_c)
            tensor = p.c_id + ":routing"
            write = plan.access(name + ":routing_write", tensor, amount, private=True, direction="write",
                                pipe="fixpipe", owner=tile.owner, after=(ready,), working_set=s.workspace_bytes,
                                offset=tile.owner * packed_bytes(self.tm * self.tn, s.dtype_c))
            self._consume(name, write)
            read = plan.access(name + ":routing_read", tensor, amount, private=True, owner=aiv, after=(write,),
                               working_set=s.workspace_bytes,
                               offset=tile.owner * packed_bytes(self.tm * self.tn, s.dtype_c))
            route = plan.work(name + ":finalize_routing", "aiv_simt", tile.m * tile.n, owner=aiv, after=(read,))
            return self._write_output(name + ":output", tile, (route,), pipe="mte3", owner=aiv)
        aiv = self._aiv(tile, epilogue == "evg_ub")
        size = packed_bytes(tile.m * tile.n, s.dtype_acc)
        if epilogue == "evg_ub":
            transfer = plan.work(name + ":fixpipe_ub", "fixpipe", size, owner=tile.owner, after=(ready,),
                                 metadata={"saved_logical_bytes": 2 * size})
            self._consume(name, transfer)
        else:
            tensor = p.c_id + ":evg"
            stride = packed_bytes(p.n, s.dtype_acc)
            run = AddressRun(tile.row * stride + packed_bytes(tile.col, s.dtype_acc), tile.m,
                             packed_bytes(tile.n, s.dtype_acc), stride)
            write = plan.access(name + ":fixpipe_gm", tensor, size, direction="write", pipe="fixpipe",
                                owner=tile.owner, after=(ready,), private=True, runs=(run,), working_set=s.workspace_bytes)
            self._consume(name, write)
            transfer = plan.access(name + ":evg_reload", tensor, size, after=(write,), owner=aiv,
                                   private=True, runs=(run,), working_set=s.workspace_bytes)
        events = {}
        for node in evg_order(s.evg_tree):
            node_name = f"{name}:evg:{node.node_id}"
            after = tuple(events[parent] for parent in node.inputs) or (transfer,)
            elements = tile.m * tile.n
            if node.kind in {"aux_load", "row_broadcast", "col_broadcast"}:
                elems = tile.n if node.kind == "row_broadcast" else tile.m if node.kind == "col_broadcast" else elements
                self_input = node.kind == "aux_load"
                full = packed_bytes(p.n, node.dtype)
                offset = packed_bytes(tile.col if node.kind == "row_broadcast" else tile.row, node.dtype)
                runs = (AddressRun(tile.row * full + packed_bytes(tile.col, node.dtype), tile.m,
                                   packed_bytes(tile.n, node.dtype), full),) if self_input else None
                dep = plan.access(node_name, node.aux_tensor_id, packed_bytes(elems, node.dtype),
                                  after=after, owner=aiv, offset=offset, runs=runs)
            elif node.kind == "aux_store":
                offset = packed_bytes(tile.row * p.n + tile.col, node.dtype)
                stride = packed_bytes(p.n, node.dtype)
                runs = (AddressRun(offset, tile.m, packed_bytes(tile.n, node.dtype), stride),)
                dep = plan.access(node_name, node.aux_tensor_id, packed_bytes(elements, node.dtype),
                                  after=after, owner=aiv, direction="write", pipe="mte3", runs=runs)
            elif node.kind == "acc_load":
                dep = transfer
            else:
                dep = plan.work(node_name + ":alu", "vec_alu", elements * _ALU_NODES.get(node.kind, 0),
                                owner=aiv, after=after)
                if node.kind in _SFU_NODES:
                    dep = plan.work(node_name + ":sfu", "vec_sfu", elements * _SFU_NODES[node.kind],
                                    owner=aiv, after=(dep,))
            events[node.node_id] = dep
        terminals = tuple(events.values())
        if not any(node.kind == "aux_store" and node.aux_tensor_id == p.c_id for node in s.evg_tree):
            return self._write_output(name + ":output", tile, terminals, pipe="mte3", owner=aiv)
        return plan.add(Stage(name + ":evg_complete", after=terminals))

    def grouped_fused(self, tile, after):
        plan, spec = self.plan, self.spec
        aiv = self._aiv(tile, direct=True)
        gate, ready = self.product(tile, range(self.gk), after)
        size = packed_bytes(tile.m * tile.n, spec.dtype_acc)
        gate_ub = plan.work(gate + ":fixpipe_ub", "fixpipe", size, owner=tile.owner, after=(ready,))
        self._consume(gate, gate_ub)
        up, ready = self.product(replace(tile, col=tile.col + self.p.n // 2), range(self.gk), (gate_ub,))
        up_ub = plan.work(up + ":fixpipe_ub", "fixpipe", size, owner=tile.owner, after=(ready,))
        self._consume(up, up_ub)
        elems = tile.m * tile.n
        silu = plan.work(gate + ":silu", "vec_sfu", elems, owner=aiv, after=(gate_ub, up_ub))
        quant = plan.work(gate + ":swiglu_mx_quant", "vec_alu", 2 * elems, owner=aiv, after=(silu,),
                          metadata={"saved_logical_bytes": 2 * (size + size)})
        store = self._write_output(gate + ":output", tile, (quant,), pipe="mte3", owner=aiv, width=self.p.n // 2)
        stride = ceil_div(self.p.n // 2, MX_SCALE_GROUP)
        width = ceil_div(tile.n, MX_SCALE_GROUP)
        return plan.access(gate + ":scale", self.p.c_id + ":scale", tile.m * width, owner=aiv, after=(store,),
                           pipe="mte3", direction="write",
                           runs=(AddressRun(tile.row * stride + tile.col // MX_SCALE_GROUP, tile.m, width, stride),))

    def ordinary(self, tile, after):
        if self.spec.epilogue_kind == "grouped_swiglu_mx_quant":
            return self.grouped_fused(tile, after)
        if _is_simt_matrix(self.spec):
            return self.simt_product(tile, after)
        return self.finish(tile, *self.product(tile, range(self.gk), after))

    def simt_product(self, tile, after):
        plan, p, s = self.plan, self.p, self.spec
        name = f"simt:{self._tile_index}"
        self._tile_index += 1
        owner = tile.owner % s.aiv_budget
        indexed = s.dispatch_policy == "MatmulGatherScatterSimt"
        physical_rows = None
        if indexed:
            index = self._index_read(name, tile, after, owner)
            physical_rows = self._physical_rows(tile)
            stride = packed_bytes(p.k, s.dtype_a)
            a = plan.access(name + ":gather", p.a_id, tile.m * stride, owner=owner, after=(index,),
                            runs=tuple(AddressRun(row * stride, 1, stride, stride) for row in physical_rows))
        else:
            a = self._rect(p.a_id, s.dtype_a, tile.row, 0, tile.m, p.k, p.k, name=name + ":gather",
                           pipe="mte2_aiv", owner=owner, after=after)
        b = self._rect(p.b_id, s.dtype_b, 0, tile.col, p.k, tile.n, p.n, name=name + ":weights",
                       pipe="mte2_aiv", owner=owner, after=after)
        x, y, z = s.simt_thread_block_dim
        simt = plan.work(name + ":simt_mac", "aiv_simt", 2 * tile.m * tile.n * p.k,
                         owner=owner, after=(a, b),
                         metadata={"thread_block_x": x, "thread_block_y": y, "thread_block_z": z})
        return self._write_output(name + ":scatter", tile, (simt,), pipe="mte3", owner=owner,
                                  physical_rows=physical_rows)

    def _simt_b_runs(self, col_lo: int, col_hi: int, repeats: int = 1) -> tuple[list[AddressRun], int]:
        """Build B-operand warp address runs for contiguous column tiles ``[col_lo, col_hi)``."""
        p, s = self.p, self.spec
        runs: list[AddressRun] = []
        total_bytes = 0
        if s.layout_b == "RowMajor":
            stride_b = packed_bytes(p.n, s.dtype_b)
            align = max(1, int(self.plan.sector))
            warp_w = packed_bytes(SIMT_WARP_SIZE, s.dtype_b)
            tail_cols = (p.n % SIMT_WARP_SIZE) or SIMT_WARP_SIZE
            tail_w = packed_bytes(tail_cols, s.dtype_b)
            full_row = (
                col_lo == 0 and col_hi * SIMT_WARP_SIZE >= p.n
                and not self.plan.burst_rules
                and (warp_w % align == 0) and (tail_w % align == 0)
            )
            if full_row:
                mat_bytes = p.k * stride_b
                runs.append(AddressRun(
                    p.b_base_byte, repeats, mat_bytes, mat_bytes if repeats == 1 else 0, "RowMajor",
                ))
                total_bytes += repeats * mat_bytes
            else:
                for col_tile in range(col_lo, col_hi):
                    col_start = col_tile * SIMT_WARP_SIZE
                    cols = min(SIMT_WARP_SIZE, p.n - col_start)
                    w_b = packed_bytes(cols, s.dtype_b)
                    run = AddressRun(p.b_base_byte + packed_bytes(col_start, s.dtype_b), p.k, w_b, stride_b, "RowMajor")
                    for _ in range(repeats):
                        runs.append(run)
                    total_bytes += repeats * p.k * w_b
        elif s.layout_b == "ColumnMajor":
            stride_b = packed_bytes(p.k, s.dtype_b)
            col_start = col_lo * SIMT_WARP_SIZE
            cols = min(col_hi * SIMT_WARP_SIZE, p.n) - col_start
            run = AddressRun(p.b_base_byte + col_start * stride_b, cols, stride_b, stride_b, "ColumnMajor")
            for _ in range(repeats):
                runs.append(run)
            total_bytes += repeats * cols * stride_b
        else:
            raise ModelingError(f"catlass_address_layout_unsupported:{s.layout_b}")
        return runs, total_bytes

    def simt_gather_scatter(self, after):
        """Lower ``MatmulGatherScatterSimt`` across full ``aiv_budget`` warp partitions."""
        plan, p, s = self.plan, self.p, self.spec
        column_tiles = ceil_div(p.n, SIMT_WARP_SIZE)
        total_warps = p.m * column_tiles
        active_workers = min(s.aiv_budget, total_warps)
        index_bytes = packed_bytes(1, "int32")
        stride_a = packed_bytes(p.k, s.dtype_a)
        stride_c = packed_bytes(p.n, s.dtype_c)
        x, y, z = s.simt_thread_block_dim
        done = []
        for worker in range(active_workers):
            warp_begin = total_warps * worker // active_workers
            warp_end = total_warps * (worker + 1) // active_workers
            name = f"simt:{worker}"
            row_lo = warp_begin // column_tiles
            row_hi = (warp_end - 1) // column_tiles + 1
            idx_stage = plan.access(
                name + ":indices", p.indices_id, (row_hi - row_lo) * index_bytes,
                owner=worker, after=after, offset=row_lo * index_bytes,
                obligation=p.indices is not None or p.physical_m is not None,
            )
            a_runs: list[AddressRun] = []
            b_runs: list[AddressRun] = []
            c_runs: list[AddressRun] = []
            a_bytes = 0
            b_bytes = 0
            c_bytes = 0
            worker_ops = 0
            full_rows = 0
            for logical_row in range(row_lo, row_hi):
                col_lo = warp_begin % column_tiles if logical_row == row_lo else 0
                col_hi = ((warp_end - 1) % column_tiles + 1) if logical_row == row_hi - 1 else column_tiles
                row_warps = col_hi - col_lo
                phys_row = logical_row if p.indices is None else p.indices[logical_row]
                a_runs.append(AddressRun(
                    phys_row * stride_a, row_warps, stride_a, stride_a if row_warps == 1 else 0,
                ))
                a_bytes += row_warps * stride_a
                c_start = col_lo * SIMT_WARP_SIZE
                c_cols = min(col_hi * SIMT_WARP_SIZE, p.n) - c_start
                c_width = packed_bytes(c_cols, s.dtype_c)
                c_runs.append(AddressRun(
                    phys_row * stride_c + packed_bytes(c_start, s.dtype_c), 1, c_width, stride_c,
                ))
                c_bytes += c_width
                worker_ops += 2 * c_cols * p.k
                if col_lo == 0 and col_hi == column_tiles:
                    full_rows += 1
                else:
                    part_runs, part_b = self._simt_b_runs(col_lo, col_hi)
                    b_runs.extend(part_runs)
                    b_bytes += part_b
            if full_rows > 0:
                full_runs, full_b = self._simt_b_runs(0, column_tiles, repeats=full_rows)
                b_runs.extend(full_runs)
                b_bytes += full_b
            a_stage = plan.access(
                name + ":gather", p.a_id, a_bytes, owner=worker, pipe="mte2_aiv",
                after=(idx_stage,), runs=tuple(a_runs),
            )
            b_stage = plan.access(
                name + ":weights", p.b_id, b_bytes, owner=worker, pipe="mte2_aiv",
                after=after, runs=tuple(b_runs), version=p.b_version,
            )
            mac_stage = plan.work(
                name + ":simt_mac", "aiv_simt", worker_ops, owner=worker, after=(a_stage, b_stage),
                metadata={"thread_block_x": x, "thread_block_y": y, "thread_block_z": z},
            )
            done.append(plan.access(
                name + ":scatter", p.c_id, c_bytes, runs=tuple(c_runs),
                direction="write", pipe="mte3", owner=worker, after=(mac_stage,),
            ))
        return tuple(done)

    def _physical_rows(self, tile):
        indices = self.p.indices
        return range(tile.row, tile.row + tile.m) if indices is None else indices[tile.row:tile.row + tile.m]

    def _index_read(self, name, tile, after, owner):
        p = self.p
        index_bytes = packed_bytes(1, "int32")
        return self.plan.access(name + ":indices", p.indices_id, tile.m * index_bytes, owner=owner,
                                after=after, offset=tile.row * index_bytes,
                                obligation=p.indices is not None or p.physical_m is not None)

    def zero_destination(self, launch):
        p, s, plan = self.p, self.spec, self.plan
        sector = plan.sector
        total = packed_bytes(p.output_rows * p.n, s.dtype_c)
        sectors = ceil_div(total, sector)
        writes = []
        for lane in range(s.aiv_budget):
            start = sectors * lane // s.aiv_budget * sector
            end = min(total, sectors * (lane + 1) // s.aiv_budget * sector)
            if start < end:
                writes.append(plan.access(
                    f"simt:zero_output:{lane}", p.c_id, end - start, direction="write",
                    pipe="mte3", owner=lane, offset=start, after=(launch,),
                ))
        return plan.add(Stage("simt:zero_output:barrier", after=tuple(writes)))


def _issued_ops(m, n, k, spec, datapath):
    native = next((wave for wave in datapath.subtile_wave if wave.native_mode == _cube_dtype(spec)), None)
    blocks = (native.independently_issuable_m, native.tile_n, native.tile_k) if native else (
        DEFAULT_INST_BLOCK_DIM, DEFAULT_INST_BLOCK_DIM, DEFAULT_INST_BLOCK_DIM
    )
    return 2 * prod(ceil_div(dim, block) * block for dim, block in zip((m, n, k), blocks))


def _reduce_partials(lowerer, part_tiles, signal):
    plan, spec = lowerer.plan, lowerer.spec
    for block, items in part_tiles.items():
        tile = items[0][0]
        owner = block % spec.aiv_budget
        scratch = lowerer.p.c_id + ":splitk"
        loads = []
        for index, (_, _, size, run) in enumerate(items):
            loads.append(plan.access(
                f"reduce:{block}:partial:{index}", scratch, size, owner=owner,
                runs=(run,), after=(signal,), private=True,
            ))
        add = plan.work(f"reduce:{block}:add", "vec_reduce", (len(loads) - 1) * tile.m * tile.n,
                        owner=owner, after=tuple(loads))
        lowerer._write_output(f"reduce:{block}:output", tile, (add,), pipe="mte3", owner=owner)


def _partial(lowerer, parts, block, tile, panels, after, offset, row_stride):
    plan, spec = lowerer.plan, lowerer.spec
    name, ready = lowerer.product(tile, panels, after)
    row_bytes = packed_bytes(tile.n, spec.dtype_acc)
    run = AddressRun(offset, tile.m, row_bytes, row_stride)
    if offset + (tile.m - 1) * row_stride + row_bytes > spec.workspace_bytes:
        raise ModelingError("catlass_partial_exceeds_workspace")
    size = tile.m * row_bytes
    partial = plan.access(
        name + ":partial", lowerer.p.c_id + ":splitk", size, runs=(run,), owner=tile.owner,
        after=(ready,), direction="write", private=True, pipe="fixpipe",
    )
    lowerer._consume(name, partial)
    parts[block].append((tile, partial, size, run))


def _normal_waves_by_owner(lowerer, start, stop, after) -> dict[int, str]:
    owner_done: dict[int, str] = {}
    if lowerer.spec.dispatch_policy == "MmadPingpongMutex":
        for i in range(start, stop):
            tile = lowerer.tile(i)
            owner_done[tile.owner] = lowerer.ordinary(tile, after)
        return owner_done
    for i in range(start, stop):
        tile = lowerer.tile(i)
        deps = (owner_done[tile.owner],) if tile.owner in owner_done else after
        owner_done[tile.owner] = lowerer.ordinary(tile, deps)
    return owner_done


def _normal_waves(lowerer, start, stop, after):
    owner_done = _normal_waves_by_owner(lowerer, start, stop, after)
    return tuple(owner_done.values()) or after


def _split_schedule(lowerer, launch, blocks):
    spec, plan = lowerer.spec, lowerer.plan
    parts = defaultdict(list)
    if spec.scheduler == "streamk":
        partition = streamk_partition(blocks, lowerer.gk, spec.aic_budget)
        if not partition.b_sk:
            _normal_waves(lowerer, 0, blocks, (launch,))
            return
        final = max(0, partition.b_norm - spec.aic_budget)
        pre_by_owner = _normal_waves_by_owner(lowerer, 0, final, (launch,))
        stride = packed_bytes(lowerer.tn, spec.dtype_acc)
        tile_bytes = lowerer.tm * stride
        for core, (start, end) in enumerate(partition.intervals):
            after = (pre_by_owner[core],) if core in pre_by_owner else (launch,)
            for part in range(partition.partial_counts[core]):
                block, panel = divmod(start, lowerer.gk)
                stop = min(end, (block + 1) * lowerer.gk)
                tile = replace(lowerer.tile(partition.b_norm + block), owner=core)
                offset = (core * STREAMK_PARTIAL_SLOTS_PER_CORE + part) * tile_bytes
                _partial(lowerer, parts, block, tile, range(panel, panel + stop - start), after, offset, stride)
                after = (parts[block][-1][1],)
                start = stop
        signal = plan.add(Stage("cross_core_set_flag", after=tuple(row[1] for group in parts.values() for row in group),
                                metadata={"streamk_partial_count": partition.partial_count}))
        _normal_waves(lowerer, final, partition.b_norm, (signal,))
    else:
        base, rem = divmod(lowerer.gk, spec.splitk_factor)
        owner_done = {}
        stride = packed_bytes(lowerer.p.n, spec.dtype_acc)
        matrix_bytes = lowerer.p.m * stride
        for split in range(spec.splitk_factor):
            start = split * base + min(split, rem)
            stop = start + base + (split < rem)
            for block in range(blocks):
                owner = (split * blocks + block) % spec.aic_budget
                tile = replace(lowerer.tile(block), owner=owner)
                offset = split * matrix_bytes + tile.row * stride + packed_bytes(tile.col, spec.dtype_acc)
                _partial(lowerer, parts, block, tile, range(start, stop),
                         (owner_done.get(owner, launch),), offset, stride)
                owner_done[owner] = parts[block][-1][1]
        signal = plan.add(Stage("cross_core_set_flag", after=tuple(owner_done.values())))
    _reduce_partials(lowerer, parts, signal)


def _estimated_panel_micros(spec, p) -> int:
    """Estimate total unrolled K-panel micro steps for streaming threshold selection."""
    tm, tn, tk = spec.l1_tile_shape
    l0k = spec.l0_tile_shape[-1]
    n_cols = p.n // 2 if spec.epilogue_kind == "grouped_swiglu_mx_quant" else p.n
    if p.groups:
        blocks = sum(ceil_div(rows, tm) for rows in p.groups) * ceil_div(n_cols, tn)
    else:
        blocks = ceil_div(p.m, tm) * ceil_div(n_cols, tn)
    return blocks * ceil_div(p.k, tk) * ceil_div(tk, l0k)


_PURE_LOWER_CACHE: dict[object, object] = {}


def _freeze_cache_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool, bytes)) or value is None:
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_cache_value(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_cache_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(_freeze_cache_value(getattr(value, f.name)) for f in fields(value)),
        )
    return value


def _datapath_cache_key(datapath) -> object:
    if is_dataclass(datapath) and not isinstance(datapath, type):
        return _freeze_cache_value(datapath)
    return _freeze_cache_value(vars(datapath)) if hasattr(datapath, "__dict__") else datapath


def lower_catlass_spec(
    spec, problem, datapath, *, context=None, memory_view=None, die=0, live_admission=False, region_id=None,
):
    """Lower a concrete specialization and return its executable audit record."""
    p = matrix_problem(problem)
    if context is None and memory_view is None and not live_admission:
        key = (_freeze_cache_value(spec), _freeze_cache_value(p), _datapath_cache_key(datapath), die, region_id)
        try:
            cached = _PURE_LOWER_CACHE.get(key)
        except TypeError:
            cached = None
            key = None
        if cached is not None:
            return cached
        result = _lower_catlass_impl(spec, p, datapath, context=None, memory_view=None, die=die,
                                     live_admission=False, region_id=region_id)
        if key is not None:
            if len(_PURE_LOWER_CACHE) >= _PURE_LOWER_CACHE_SIZE:
                _PURE_LOWER_CACHE.pop(next(iter(_PURE_LOWER_CACHE)))
            _PURE_LOWER_CACHE[key] = result
        return result
    return _lower_catlass_impl(spec, p, datapath, context=context, memory_view=memory_view, die=die,
                               live_admission=live_admission, region_id=region_id)


def _lower_catlass_impl(
    spec, p, datapath, *, context=None, memory_view=None, die=0, live_admission=False, region_id=None,
):
    solution = solve_catlass_tile(spec, p, datapath, memory_view=memory_view, die=die)
    expected = _workspace_bytes(spec.template_id, p, spec.l1_tile_shape, spec.aic_budget,
                                spec.splitk_factor, spec.dispatch_policy, spec.epilogue_kind)
    if spec.workspace_bytes != expected:
        raise ModelingError(f"catlass_workspace_mismatch:{spec.workspace_bytes}:{expected}")
    if _estimated_panel_micros(spec, p) >= STREAMING_PANEL_MICRO_THRESHOLD:
        return lower_catlass_streaming(
            spec, p, datapath, solution,
            context=context, memory_view=memory_view, die=die,
            live_admission=live_admission, region_id=region_id,
        )
    plan = Lowering(datapath, context=context, memory_view=memory_view, die=die, live_admission=live_admission)
    lowerer = _MatrixLowerer(plan, spec, p)
    launch = plan.launch()
    if spec.dispatch_policy in GATHER_SCATTER_POLICIES:
        launch = lowerer.zero_destination(launch)
    blocks = ceil_div(p.m, lowerer.tm) * ceil_div(p.n, lowerer.tn)
    if spec.scheduler == "grouped_slice_m":
        n_cols = p.n // 2 if spec.epilogue_kind == "grouped_swiglu_mx_quant" else p.n
        row_start, owner_done = 0, {}
        schedules = iter(grouped_start_cores(p.groups, n_cols, lowerer.tm, lowerer.tn, spec.aic_budget))
        for rows in p.groups:
            if not rows:
                continue
            index, start, loops = next(schedules)
            for task in range(loops):
                tile = lowerer.tile(task, group=index, group_start=row_start, rows=rows, start_core=start, n_cols=n_cols)
                owner_done[tile.owner] = lowerer.ordinary(tile, (owner_done.get(tile.owner, launch),))
            row_start += rows
    elif spec.dispatch_policy == "MatmulGatherScatterSimt":
        lowerer.simt_gather_scatter((launch,))
    elif spec.scheduler in {"streamk", "multi_core_splitk"}:
        _split_schedule(lowerer, launch, blocks)
    else:
        _normal_waves(lowerer, 0, blocks, (launch,))
    for rings in lowerer.owner_rings.values():
        for ring in rings:
            plan.add_ring(ring, after=(launch,))
    rid = region_id or spec.template_id
    semantic = [SemanticWorkItem(
        rid + ":matrix", "matrix_mac", _cube_dtype(spec), 2 * p.m * p.n * p.k,
        eligible_matrix_engines(p, datapath), (spec.scheduler,),
    )]
    kinds = {"vec_alu": "vector_alu", "vec_sfu": "vector_sfu", "vec_reduce": "vector_reduce",
             "aiv_simt": "vector_alu"}
    semantic.extend(
        SemanticWorkItem(rid + ":" + pipe, kinds[pipe], dtype_name(spec.dtype_acc), int(ops),
                         ("aiv_simt",) if pipe == "aiv_simt" else ("aiv_simd",), (spec.scheduler,))
        for pipe, ops in plan.vector.items() if ops
        and not (pipe == "aiv_simt" and _is_simt_matrix(spec))
        and not (pipe == "vec_reduce" and spec.scheduler in {"streamk", "multi_core_splitk"})
    )
    return plan.result("K2", rid, (rid,), spec, semantic, solution)
