"""MoE routing, grouped GEMM and collective/expert-transport families.

[Role]
  Derive explicit per-expert ragged tile work, matrix tiling capacity steps and
  topology-correct collective byte volumes as guarded piecewise expressions.
  This module provides the shared physical primitives used by the matrix family
  and owns routing, grouped GEMM, collective and MC2 transport lowering.

[Invariants]
  * Shapes, expert counts, tile capacities, rank/topology facts and directed
    traffic matrices are structural inputs; no model-name or shape-keyed table
    selects a latency, discount or tile candidate.
  * A zero-token expert contributes no weight bank read.  Routed rows conserve
    ``sum_e n_e == k * T_valid``.  Explicit ragged counts drive per-expert
    ``sum_e ceil(n_e / M_t)`` tiles; the uniform expectation is reported
    separately as a fractional moment.
  * Collective volume is a function of rank count, node split, message bytes,
    frame/MTU alignment and directed ``{M_uv}`` cells.  Receive hotspot is
    ``max_v sum_u M_uv`` and is not replaced by a per-model coefficient.
  * Tile buffers are screened against declared L1/L0A/L0B/L0C bytes.  Pipeline
    fill plus ``q-1`` steady intervals is distinct from ``q`` serial tiles.

[Timeouts & Locks]
  Pure bounded expression construction and O(ranks^2) matrix validation.  No
  I/O, waits, locks or device work.

[Side-Effect Policy]
  Returns immutable family templates and helper records only.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any

from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.subgraphs import BoundOperator
from .compiler import (
    CompiledOperator, FamilyTemplate, SegmentAxis, _demand, _out_name,
    _uniform_counts,
)
from .expressions import (
    Add, CeilDiv, Const, Div, Expr, Max, Min, Mul, Sub, Var, as_fraction,
)
from .guards import StepGuard

FRACTAL = 16
UB_ALIGN = 32
# Declared physical capacities.  A caller may override any of them through
# theta; they are hardware structure, never a shape/model lookup.
DEFAULT_L1_BYTES = 512 * 1024
DEFAULT_L0A_BYTES = 64 * 1024
DEFAULT_L0B_BYTES = 64 * 1024
DEFAULT_L0C_BYTES = 128 * 1024
DEFAULT_MTE2_GRANULE = 512
DEFAULT_MTU = 4096
DEFAULT_NODE_RANKS = 8

_WEIGHT_WIDTHS = {
    "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0, "float16": 2.0,
    "fp32": 4.0, "float32": 4.0, "int8": 1.0, "w8a8": 1.0,
    "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "int4": 0.5, "w4a8": 0.5, "fp4": 0.5, "fp4_e2m1": 0.5,
    "mxfp8": 1.0, "mx_fp8": 1.0, "mxfp8_e4m3": 1.0,
    "mxfp4": 0.5, "mx_fp4": 0.5, "mxfp4_e2m1": 0.5,
}
_NON_QUANT = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32",
                        "float32", ""})

def _int(value: Any, default: int) -> int:
    return default if value is None else int(value)

def _frac(value: Any, default: Any = 0) -> Any:
    return as_fraction(default) if value is None else as_fraction(value)

def _width(fmt: Any, default: float = 2.0) -> float:
    text = str(fmt or "").strip().lower()
    if text in _WEIGHT_WIDTHS:
        return _WEIGHT_WIDTHS[text]
    try:
        return float(dtype_bytes(text))
    except ModelingError:
        return float(default)

def _is_quant(fmt: Any) -> bool:
    return str(fmt or "").strip().lower() not in _NON_QUANT

def _round_expr(value: Expr, granule: int) -> Expr:
    if granule <= 1:
        return value
    return Mul((CeilDiv(value, Const(granule)), Const(granule)))

@dataclass(frozen=True)
class TilePlan:
    """One capacity-screened cube tile geometry."""

    m_tile: int
    n_tile: int
    k_tile: int
    k_inner: int
    m_pad: int
    n_pad: int
    k_pad: int
    k_align: int
    n_tiles: int
    k_panels: int
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    slots: int
    compute_slots: int
    fits: bool

    @property
    def tile_load_bytes(self) -> int:
        return self.m_tile * self.k_tile + self.k_tile * self.n_tile

    def footprint(self, act_bytes: float, weight_bytes: float,
                  out_bytes: float = 4.0) -> dict[str, float]:
        return {
            "l1": self.slots * (self.m_tile * self.k_tile * act_bytes
                                + self.k_tile * self.n_tile * weight_bytes),
            "l0a": self.compute_slots * self.m_tile * self.k_inner * act_bytes,
            "l0b": self.compute_slots * self.k_inner * self.n_tile * weight_bytes,
            "l0c": self.compute_slots * self.m_tile * self.n_tile
            * max(4.0, out_bytes),
        }

def _shrink(value: int, unit: int) -> int:
    if value <= unit:
        return value
    return max(unit, (value // 2) // unit * unit)

def tile_plan(m: int, n: int, k: int, *, mt: int = 128, nt: int = 256,
              kt: int = 128, act_bytes: float = 2.0,
              weight_bytes: float = 2.0, out_bytes: float = 4.0,
              l1_bytes: int = DEFAULT_L1_BYTES,
              l0a_bytes: int = DEFAULT_L0A_BYTES,
              l0b_bytes: int = DEFAULT_L0B_BYTES,
              l0c_bytes: int = DEFAULT_L0C_BYTES,
              slots: int = 2, compute_slots: int = 2) -> TilePlan:
    """Screen one declared tile step against physical buffer capacities."""
    m, n, k = max(1, int(m)), max(1, int(n)), max(1, int(k))
    slots, compute_slots = max(1, int(slots)), max(1, int(compute_slots))
    k_align = 32 if min(float(act_bytes), float(weight_bytes)) <= 1.0 else FRACTAL
    m_pad, n_pad = align_up(m, FRACTAL), align_up(n, FRACTAL)
    k_pad = align_up(k, k_align)
    m_tile = min(m_pad, align_up(max(FRACTAL, int(mt)), FRACTAL))
    n_tile = min(n_pad, align_up(max(FRACTAL, int(nt)), FRACTAL))
    k_tile = min(k_pad, align_up(max(k_align, int(kt)), k_align))
    k_inner = min(k_tile, align_up(max(k_align, 64), k_align))
    l1_cap, l0a_cap = max(1, int(l1_bytes)), max(1, int(l0a_bytes))
    l0b_cap, l0c_cap = max(1, int(l0b_bytes)), max(1, int(l0c_bytes))

    def footprint() -> tuple[int, int, int, int]:
        return (
            slots * (m_tile * k_tile * act_bytes + k_tile * n_tile * weight_bytes),
            compute_slots * m_tile * k_inner * act_bytes,
            compute_slots * k_inner * n_tile * weight_bytes,
            compute_slots * m_tile * n_tile * max(4.0, out_bytes),
        )

    l1 = l0a = l0b = l0c = 0
    for _ in range(40):
        l1, l0a, l0b, l0c = footprint()
        if l1 <= l1_cap and l0a <= l0a_cap and l0b <= l0b_cap and l0c <= l0c_cap:
            break
        if (l1 > l1_cap or l0c > l0c_cap) and (m_tile > FRACTAL or n_tile > FRACTAL):
            if m_tile >= n_tile and m_tile > FRACTAL:
                m_tile = _shrink(m_tile, FRACTAL)
            elif n_tile > FRACTAL:
                n_tile = _shrink(n_tile, FRACTAL)
            elif k_tile > k_align:
                k_tile = _shrink(k_tile, k_align)
        elif (l0a > l0a_cap or l0b > l0b_cap) and k_inner > k_align:
            k_inner = _shrink(k_inner, k_align)
        elif k_tile > k_align:
            k_tile = _shrink(k_tile, k_align)
        else:
            break
    l1, l0a, l0b, l0c = footprint()
    fits = l1 <= l1_cap and l0a <= l0a_cap and l0b <= l0b_cap and l0c <= l0c_cap
    k_inner = min(k_inner, k_tile)
    return TilePlan(
        m_tile, n_tile, k_tile, k_inner, m_pad, n_pad, k_pad, k_align,
        math.ceil(n_pad / n_tile), math.ceil(k_pad / k_tile),
        int(l1), int(l0a), int(l0b), int(l0c), slots, compute_slots, fits,
    )

def _scale(group: tuple[tuple[str, Expr], ...], factor: Expr) -> tuple[tuple[str, Expr], ...]:
    return tuple((pipe, Mul((work, factor))) for pipe, work in group)

def _merge_groups(*groups: tuple[tuple[str, Expr], ...]) -> tuple[tuple[str, Expr], ...]:
    merged: dict[str, Expr] = {}
    for group in groups:
        for pipe, work in group:
            merged[pipe] = Add((merged[pipe], work)) if pipe in merged else work
    return tuple(merged.items())

def _stage(ctx: CompiledOperator,
           group: tuple[tuple[str, Expr], ...]) -> tuple[Any, ...]:
    return tuple(_demand(ctx, pipe, work) for pipe, work in group)

def pipeline_stages(ctx: CompiledOperator, *,
                    load: tuple[tuple[str, Expr], ...],
                    compute: tuple[tuple[str, Expr], ...],
                    store: tuple[tuple[str, Expr], ...],
                    q: Expr, pipeline: bool = True, parallel: int = 1,
                    tiles: Expr | None = None, gate: Expr | None = None,
                    fill_extra: tuple[tuple[str, Expr], ...] = (),
                    ) -> tuple[tuple[Any, ...], ...]:
    """Fill plus steady pipeline over ``tiles`` real jobs; cube stays per core."""
    width = Const(parallel)
    gate = Const(1) if gate is None else gate
    tiles = Mul((q, width)) if tiles is None else tiles
    active = Min((width, tiles))
    rest = Sub(tiles, active)
    stages = [
        _stage(ctx, _merge_groups(_scale(load, Mul((gate, active))),
                                  _scale(fill_extra, Mul((gate, active))))),
        _stage(ctx, _scale(compute, Mul((gate, width)))),
        _stage(ctx, _scale(store, Mul((gate, active)))),
    ]
    remaining_waves = Sub(q, Const(1))
    if pipeline:
        stages.append(_stage(ctx, _merge_groups(
            _scale(load, Mul((gate, rest))),
            _scale(compute, Mul((gate, Mul((remaining_waves, width))))),
            _scale(store, Mul((gate, rest))))))
    else:
        stages.extend((
            _stage(ctx, _scale(load, Mul((gate, rest)))),
            _stage(ctx, _scale(compute, Mul((gate, Mul((remaining_waves, width)))))),
            _stage(ctx, _scale(store, Mul((gate, rest)))),
        ))
    return tuple(stages)

def _counts_for(ctx: CompiledOperator, geometry: Any, state: Any,
                local: int, top_k: int,
                experts: int | None = None) -> tuple[int, ...]:
    """Local expert rows; explicit global vectors localize before arithmetic."""
    experts = max(local, int(experts if experts is not None else local))
    counts = geometry.routing_for(int(ctx.layer))
    if counts is None:
        try:
            counts = state.routing_state(int(ctx.layer)).counts
        except ModelingError:
            counts = None
    if counts is None:
        ep = max(1, math.ceil(experts / max(1, local))) if experts > local else 1
        counts = _uniform_counts(geometry.tokens, top_k, experts)[::ep][:local]
        counts = tuple(counts) + (0,) * (local - len(counts))
    elif len(counts) == experts and experts != local:
        counts = tuple(counts)[:local]
    if len(counts) != local:
        raise ModelingError(f"moe_routing_local_experts_mismatch:{len(counts)}!={local}")
    if any(int(value) < 0 for value in counts):
        raise ModelingError("moe_routing_count_invalid")
    return tuple(int(value) for value in counts)

def _moe_route_shape(ctx: CompiledOperator, theta) -> tuple[int, int, int]:
    """Local experts/top-k for ETP; the grouped producer is authoritative."""
    local = theta.find("local", theta.find("E_local", None))
    top_k = theta.find("k", theta.find("top_k", None))
    experts = theta.find("E", None)
    for sibling in getattr(ctx, "siblings", ()):
        if sibling.name in {"expert_down", "expert_gate_up"}:
            local = local if local is not None else sibling.theta.find("local", sibling.theta.find("E_local", None))
            top_k = top_k if top_k is not None else sibling.theta.find("k", sibling.theta.find("top_k", 1))
            experts = experts if experts is not None else sibling.theta.find("E", None)
            break
    local = max(1, int(local if local is not None else 1))
    top_k = max(1, int(top_k if top_k is not None else 1))
    experts = max(local, int(experts if experts is not None else local))
    return local, top_k, experts

def _compile_grouped_gemm(ctx: CompiledOperator,
                          operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    k, n = int(theta.require("K")), int(theta.require("N"))
    raw_experts = theta.find("E", None)
    if raw_experts is None:
        raw_experts = theta.require("experts")
    experts = int(raw_experts)
    raw_local = theta.find("local", theta.find("E_local", None))
    local = experts if raw_local is None else int(raw_local)
    top_k = int(theta.find("k", theta.find("top_k", 1)) or 1)
    raw_tile = theta.find("row_tile", theta.find("moe_rows", None))
    row_tile = int(raw_tile if raw_tile is not None else ctx.policy.moe_row_tile)
    nt = int(theta.find("nt", ctx.policy.nt) or ctx.policy.nt)
    cores = max(1, int(theta.find("cores", ctx.policy.cores) or ctx.policy.cores))
    act_bytes = _frac(theta.find("act_bytes", 2.0), 2.0)
    weight_format = theta.find("weight_format", "bf16")
    weight_bytes = _width(weight_format)
    weight_total = _frac(theta.find("weight_total",
                                    n * k * weight_bytes), n * k * weight_bytes)
    weight_payload = _frac(theta.find("weight_payload", weight_total), weight_total)
    weight_scale = _frac(theta.find("weight_scale", 0.0), 0.0)
    weight_zero = _frac(theta.find("weight_zero", 0.0), 0.0)
    out_format = theta.find("output_format", theta.find("act_format", "bf16"))
    out_bytes = _width(out_format)
    plan = tile_plan(row_tile, n, k, mt=row_tile, nt=nt,
                     act_bytes=float(act_bytes), weight_bytes=weight_bytes,
                     out_bytes=out_bytes,
                     l1_bytes=_int(theta.find("l1_bytes"), DEFAULT_L1_BYTES),
                     l0a_bytes=_int(theta.find("l0a_bytes"), DEFAULT_L0A_BYTES),
                     l0b_bytes=_int(theta.find("l0b_bytes"), DEFAULT_L0B_BYTES),
                     l0c_bytes=_int(theta.find("l0c_bytes"), DEFAULT_L0C_BYTES),
                     slots=_int(theta.find("pipeline_slots",
                                           theta.find("slots", None)),
                                ctx.policy.pipeline_slots),
                     compute_slots=max(1, _int(theta.find("compute_slots", None), 2)))
    if not plan.fits:
        raise ModelingError("tile_plan_capacity_exceeded")
    m_tile, n_tile, n_tiles, k_panels = (plan.m_tile, plan.n_tile,
                                         plan.n_tiles, plan.k_panels)
    axes = (SegmentAxis("EXPERT_TILES", "moe_tiles",
                        StepGuard("moe_tiles", "moe_tiles", 1, lower_bound=0),
                        "max_moe_rows"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        counts = _counts_for(ctx, geometry, state, local, top_k, experts)
        rows = sum(counts)
        active = sum(1 for value in counts if value)
        padded = sum(math.ceil(value / m_tile) * m_tile for value in counts if value)
        fractal_rows = sum(math.ceil(value / FRACTAL) * FRACTAL for value in counts if value)
        row_tiles = sum(math.ceil(value / m_tile) for value in counts if value)
        tiles = row_tiles * n_tiles
        routed_tokens = rows / top_k if top_k else 0
        if top_k <= 0 or experts <= 0 or routed_tokens <= 0:
            expected = 0.0
        elif top_k >= experts:
            expected = float(experts)
        else:
            expected = experts * (1.0 - (1.0 - top_k / experts)
                                  ** routed_tokens)
        row_bins = [0] * cores
        tile_bins = [0] * cores
        for value in sorted((item for item in counts if item), reverse=True):
            slot = min(range(cores), key=tile_bins.__getitem__)
            row_bins[slot] += value
            tile_bins[slot] += math.ceil(value / m_tile)
        busy = max(range(cores), key=tile_bins.__getitem__)
        return {"moe_rows": rows, "MOE_ROWS": rows, "moe_active": active,
                "MOE_ACTIVE": active, "moe_padded": padded,
                "MOE_PADDED": padded, "MOE_FRACTAL_ROWS": fractal_rows,
                "moe_tiles": tiles, "MOE_ROW_TILES": row_tiles,
                "EXPERT_TILES": tiles, "moe_counts": counts,
                "EXPECTED_ACTIVE": as_fraction(expected),
                "expert_peak": max(counts) if counts else 0,
                "expert_imbalance": (max(counts) / (rows / max(1, len(counts)))
                                     if rows else 0.0),
                "max_owner_rows": row_bins[busy], "max_count": tile_bins[busy]}

    rows, active, padded = Var("MOE_ROWS"), Var("MOE_ACTIVE"), Var("MOE_PADDED")
    crit_m = Div(Var("max_owner_rows"), Max((Var("max_count"), Const(1))))
    tiles = Var("EXPERT_TILES")
    waves = Mul((Var("max_count"), Const(n_tiles)))
    gate = Min((tiles, Const(1)))
    k_job = Const(plan.k_pad)
    metadata_total = weight_scale + weight_zero
    metadata_tile = (Div(Mul((Const(metadata_total), k_job,
                              Const(n_tile))),
                         Const(max(1, k * n)))
                     if metadata_total else Const(0))
    load_tile = Add((Mul((Const(m_tile), k_job, Const(act_bytes))),
                     Mul((k_job, Const(n_tile), Const(weight_bytes))),
                     metadata_tile))
    compute_tile = Mul((Const(2), Const(m_tile), Const(n_tile), k_job))
    store_tile = Mul((Const(m_tile), Const(n_tile), Const(out_bytes)))
    stage_load = Add((Mul((crit_m, k_job, Const(act_bytes))),
                      Mul((k_job, Const(n_tile), Const(weight_bytes))),
                      metadata_tile))
    stage_compute = Mul((Const(2), crit_m, Const(n_tile), k_job))
    stage_store = Mul((crit_m, Const(n_tile), Const(out_bytes)))
    quant = _is_quant(weight_format) or _is_quant(theta.find("act_format", "bf16"))
    vector_tile = Mul((crit_m, k_job)) if quant else Const(0)
    fill_extra = (("vector", Add((Const(4096 * 128), Mul((active, Const(512 * 128)))))),)
    weight_read = Mul((active, Const(weight_total)))
    act_read = Mul((rows, Const(k), Const(act_bytes)))
    out = Mul((padded, Const(n), Const(out_bytes)))
    stages = pipeline_stages(
        ctx, load=(("mte2", stage_load), ("mte1", stage_load),
                   ("hbm", stage_load)),
        compute=(("cube", stage_compute),
                 ("fixpipe", Mul((crit_m, Const(n_tile), Const(4)))),
                 ("vector", vector_tile)),
        store=(("mte3", stage_store), ("hbm", stage_store)),
        q=waves, tiles=tiles, gate=gate, fill_extra=fill_extra,
        pipeline=plan.slots >= 2, parallel=cores)
    derived = (
        ("moe_rows", rows), ("active_experts", active),
        ("padded_rows", padded), ("tiles", tiles), ("waves", waves),
        ("row_tiles", Var("MOE_ROW_TILES")),
        ("zero_experts", Sub(Const(local), active)),
        ("expected_active_experts", Var("EXPECTED_ACTIVE")),
        ("expert_peak", Var("expert_peak")),
        ("expert_imbalance", Var("expert_imbalance")),
        ("max_owner_rows", Var("max_owner_rows")),
        ("max_count", Var("max_count")),
        ("flops", Mul((Const(2), rows, Const(n), Const(k)))),
        ("padded_flops", Mul((Const(2), padded, Const(plan.n_pad), Const(plan.k_pad)))),
        ("weight_payload", Mul((active, Const(weight_payload)))),
        ("weight_scale", Mul((active, Const(weight_scale)))),
        ("weight_bytes", weight_read), ("weight_total", Const(weight_total)),
        ("row_tile", Const(m_tile)), ("n_tile", Const(n_tiles)),
        ("k_panels", Const(k_panels)), ("tile_load_bytes", load_tile),
        ("tile_compute_flops", compute_tile), ("tile_store_bytes", store_tile),
        ("conserved_rows", rows), ("act_read_bytes", act_read),
        ("output_bytes", out),
    )
    return FamilyTemplate(
        axes=axes, stages=stages,
        materialization=((_out_name(ctx), out),), internal=(),
        residency=Const(weight_total * local), derived=derived,
        regime_exprs=(), binder=binder,
        annotations=(("family", "grouped_gemm"), ("E", str(experts)),
                     ("local", str(local)), ("top_k", str(top_k)),
                     ("m_tile", str(m_tile)), ("weight_format", str(weight_format))),
    )

def _compile_router(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    if theta.find("K", None) is not None and theta.find("N", None) is not None:
        from .families_matrix import _compile_gemm
        return _compile_gemm(ctx, operator)
    raw_experts = theta.find("E", theta.find("experts", None))
    if raw_experts is None:
        raise ModelingError("moe_routing_requires_expert_count")
    experts = max(1, int(raw_experts))
    top_k = max(1, _int(theta.find("k", theta.find("top_k", 1)), 1))
    width = max(1, _int(theta.find("elements", experts), experts))
    mt = max(1, _int(theta.find("mt", ctx.policy.mt), ctx.policy.mt))
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("routing_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        tokens = geometry.tokens
        expected = (float(experts) if top_k >= experts else
                    experts * (1.0 - (1.0 - top_k / experts) ** tokens))
        return {"routing_tokens": tokens, "routing_expected_active":
                as_fraction(expected)}

    tokens = Var("T")
    scores = Mul((tokens, Const(width * 4)))
    topk = Mul((tokens, Const(top_k * 8)))
    histogram = Add((Const(experts * 8), Mul((tokens, Const(top_k * 4)))))
    sorted_rows = Mul((tokens, Const(top_k * 8)))
    permuted = Mul((tokens, Const(top_k * 8)))
    vector = Add((Mul((tokens, Const(width))),
                  Mul((tokens, Const(width))),
                  Mul((tokens, Const(top_k))),
                  Mul((tokens, Const(top_k * max(1, math.ceil(math.log2(max(2, top_k))))))),
                  Mul((tokens, Const(top_k)))))
    writes = Add((topk, histogram, sorted_rows, permuted))
    main = (_demand(ctx, "mte2", scores), _demand(ctx, "mte3", writes),
            _demand(ctx, "hbm", Add((scores, writes))), _demand(ctx, "vector", vector))
    derived = (
        ("routing_tokens", tokens), ("experts", Const(experts)),
        ("top_k", Const(top_k)), ("histogram_bins", Const(experts)),
        ("sort_ops", Mul((tokens, Const(top_k)
                          * max(1, math.ceil(math.log2(max(2, top_k))))))),
        ("permute_indices", Mul((tokens, Const(top_k)))),
        ("expected_active_experts", Var("routing_expected_active")),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,),
        materialization=(("moe_routing_ledger", writes),), internal=(),
        residency=Const(0), derived=derived, regime_exprs=(), binder=binder,
        annotations=(("family", "router"), ("mode", "routing_init")),
    )

# ---------------------------------------------------------------------------
# Collective and MC2 expert transport

def _collective_kind(name: str, theta: Any) -> str:
    text = (str(theta.find("collective", "") or name)).lower().replace("-", "_")
    if "allreduce" in text or "all_reduce" in text:
        return "allreduce"
    if "reduce_scatter" in text or "reducescatter" in text:
        return "reduce_scatter"
    if "all_gather" in text or "allgather" in text:
        return "all_gather"
    if "alltoall" in text or "all_to_all" in text or "dispatch" in text \
            or "combine" in text:
        return "alltoallv"
    return "allreduce"

def _ranks(theta: Any) -> int:
    # Explicit axis membership wins; "world" never outranks a parallel group.
    axis = theta.find("axis", theta.find("group_axis", None))
    if axis is not None:
        value = theta.find("members", theta.find(str(axis), None))
        if value is not None and int(value) > 0:
            return int(value)
    for key in ("members", "ranks", "R", "ep", "tp", "etp", "world"):
        value = theta.find(key, None)
        if value is not None and int(value) > 0:
            return int(value)
    return 1

def _node_ranks(theta: Any, ranks: int) -> int:
    value = None
    for key in ("ranks_per_node", "hccs_ranks", "node_ranks", "local_ranks"):
        value = theta.find(key, None)
        if value is not None:
            break
    return max(1, min(ranks, int(value or DEFAULT_NODE_RANKS)))

def _topology(theta: Any, ranks: int, node_ranks: int) -> str:
    raw = str(theta.find("topology", "") or "").lower()
    if raw in {"hccs", "intra", "intra_node"}:
        return "hccs"
    if raw in {"roce", "inter", "inter_node"}:
        return "roce"
    if raw:
        return raw
    return "hccs" if ranks <= node_ranks else "hierarchical"

def _link_pipes(ctx: CompiledOperator) -> tuple[str, str]:
    names = {name for name, _ in ctx.supply.rates}
    intra = "link_hccs" if "link_hccs" in names else "link"
    inter = "link_roce" if "link_roce" in names else intra
    return intra, inter

def _coeff(value: Any, message: Expr) -> Expr:
    exact = as_fraction(value)
    return Const(0) if exact == 0 else Mul((Const(exact), message))

def _collective_volumes(kind: str, message: Expr, ranks: int,
                        node_ranks: int) -> tuple[Expr, Expr]:
    nodes = max(1, math.ceil(ranks / node_ranks))
    intra_ranks = min(ranks, node_ranks)
    if ranks <= 1:
        return Const(0), Const(0)
    shard = Div(message, Const(intra_ranks))
    if kind == "allreduce":
        intra = _coeff(Fraction(2 * (intra_ranks - 1), intra_ranks), message)
        inter = _coeff(Fraction(2 * (nodes - 1), nodes), shard) if nodes > 1 else Const(0)
    elif kind == "reduce_scatter":
        intra = _coeff(Fraction(intra_ranks - 1, intra_ranks), message)
        inter = _coeff(Fraction(nodes - 1, nodes), shard) if nodes > 1 else Const(0)
    elif kind == "all_gather":
        intra = _coeff(Fraction(intra_ranks - 1), message)
        inter = (_coeff(Fraction((nodes - 1) * intra_ranks), message)
                 if nodes > 1 else Const(0))
    else:
        raise ModelingError("collective_uniform_volume_kind_unsupported")
    return intra, inter

def _uniform_alltoallv(message: Expr, ranks: int, node_ranks: int, mtu: int
                       ) -> tuple[Expr, Expr, Expr, Expr, Expr]:
    """Frame per-peer cells exactly as the directed matrix does."""
    if ranks <= 1:
        return Const(0), Const(0), Const(0), Const(0), Const(0)
    peers = ranks - 1
    intra_peers = max(0, min(ranks, node_ranks) - 1)
    inter_peers = max(0, ranks - min(ranks, node_ranks))
    per_peer = Mul((CeilDiv(Div(message, Const(peers)), Const(mtu)), Const(mtu)))
    send = Mul((Const(peers), per_peer))
    intra = Mul((Const(intra_peers), per_peer))
    inter = Mul((Const(inter_peers), per_peer))
    return intra, inter, send, send, Mul((send, Const(ranks)))

def _traffic_matrix(theta: Any) -> tuple[tuple[tuple[int, ...], ...], bool] | None:
    raw = theta.find("traffic_matrix", None)
    if raw is None:
        raw = theta.find("send_matrix", None)
    if raw is None:
        return None
    matrix = tuple(tuple(int(value) for value in row) for row in raw)
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ModelingError("collective_traffic_matrix_requires_square_rank_cells")
    if any(value < 0 for row in matrix for value in row):
        raise ModelingError("collective_traffic_matrix_cell_invalid")
    per_token = bool(int(theta.find("matrix_per_token", 0) or 0))
    return matrix, per_token

def _matrix_volumes(matrix: tuple[tuple[int, ...], ...], node_ranks: int,
                    mtu: int, per_token: bool,
                    tokens: Expr) -> tuple[Expr, ...]:
    """Scale logical cells by tokens, frame each peer, then take bottlenecks."""
    ranks = len(matrix)

    def framed(value: int) -> Expr:
        base = Mul((Const(int(value)), tokens)) if per_token else Const(int(value))
        return _round_expr(base, mtu)

    cells = tuple(tuple(framed(value) for value in row) for row in matrix)
    def total(values) -> Expr:
        items = tuple(values)
        return Const(0) if not items else (items[0] if len(items) == 1 else Add(items))
    sends = tuple(total(cells[i]) for i in range(ranks))
    hotspots = tuple(total(cells[i][j] for i in range(ranks)) for j in range(ranks))
    aggregate = Sub(total(sends), total(cells[i][i] for i in range(ranks)))
    def bottleneck(send: tuple[Expr, ...], recv: tuple[Expr, ...]) -> Expr:
        return Max((*send, *recv))
    def node_split(off_node: bool) -> Expr:
        split = lambda i, j: (i // node_ranks == j // node_ranks) != off_node
        send = tuple(total(cells[i][j] for j in range(ranks)
                           if i != j and split(i, j)) for i in range(ranks))
        recv = tuple(total(cells[i][j] for i in range(ranks)
                           if i != j and split(i, j)) for j in range(ranks))
        return bottleneck(send, recv)
    logical = Const(max((sum(matrix[i][j] for i in range(ranks) if i != j)
                         for j in range(ranks)), default=0))
    if per_token:
        logical = Mul((logical, tokens))
    return (node_split(False), node_split(True), Max(sends), Max(hotspots),
            aggregate, logical)

def _chunk_stages(ctx: CompiledOperator, intra: Expr, inter: Expr,
                  hbm: Expr, chunks: int, intra_pipe: str, inter_pipe: str,
                  ranks: int = 1, node_ranks: int = 1) -> tuple[tuple[Any, ...], ...]:
    """One fill chunk, then ``chunks - 1`` overlapped steady chunks.

    Fill is ``1/chunks`` and steady is the remainder, so the stages sum to
    the full payload and ``chunks`` does not discount the work.
    """
    chunks = max(1, int(chunks))

    def one() -> tuple[tuple[str, Expr], ...]:
        share = Div(Const(1), Const(chunks))
        if intra_pipe == inter_pipe:
            links = ((intra_pipe, Mul((Add((intra, inter)), share))),)
        else:
            links = ((intra_pipe, Mul((intra, share))),
                     (inter_pipe, Mul((inter, share))))
        return links + (("hbm", Mul((hbm, share))),
                        ("vector", Mul((Div(hbm, Const(2)), share))))

    fill = one()
    if chunks == 1:
        return (_stage(ctx, fill),)
    return (_stage(ctx, fill), _stage(ctx, _scale(fill, Const(chunks - 1))))

def _compile_collective(ctx: CompiledOperator,
                        operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    name = operator.name.lower().replace("-", "_")
    kind = _collective_kind(name, theta)
    ranks = _ranks(theta)
    node_ranks = _node_ranks(theta, ranks)
    topology = _topology(theta, ranks, node_ranks)
    if topology == "hccs":
        node_ranks = ranks
    elif topology == "roce":
        node_ranks = 1
    wire = _frac(theta.find("wire_bytes", theta.find("bytes_per_token", 0.0)), 0.0)
    mtu = max(1, _int(theta.find("mtu_bytes"), DEFAULT_MTU))
    chunks = max(1, _int(theta.find("chunks"), 1))
    etp = name == "etp_allreduce" or ctx.kind == "F3"
    route = _moe_route_shape(ctx, theta) if etp else None
    message = (Mul((Var("MOE_ROWS"), Const(wire))) if etp
               else Mul((Var("T"), Const(wire))))
    matrix = _traffic_matrix(theta)
    if matrix is not None and len(matrix[0]) != ranks:
        raise ModelingError("collective_traffic_matrix_rank_count_mismatch")
    if matrix is not None:
        intra, inter, send, hotspot, aggregate, output = _matrix_volumes(
            matrix[0], node_ranks, mtu, matrix[1], Var("T"))
        hbm = Add((send, hotspot))
    elif kind == "alltoallv":
        intra, inter, send, hotspot, aggregate = _uniform_alltoallv(
            message, ranks, node_ranks, mtu)
        output = message
        hbm = Add((send, hotspot))
    else:
        intra, inter = _collective_volumes(kind, message, ranks, node_ranks)
        send, hotspot = message, message
        aggregate = Mul((Add((intra, inter)), Const(ranks)))
        if kind == "all_gather":
            output = Mul((message, Const(ranks)))
        elif kind == "reduce_scatter":
            output = Div(message, Const(ranks))
        else:
            output = message
        hbm = Add((message, output))
    intra, inter = _round_expr(intra, mtu), _round_expr(inter, mtu)
    mt = max(1, _int(theta.find("mt", ctx.policy.mt), ctx.policy.mt))
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("collective_tiles", "tokens", mt),
                        "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        values = {"collective_ranks": ranks, "collective_chunks": chunks,
                  "collective_bytes": (sum(sum(row) for row in matrix[0])
                                       if matrix is not None else 0)}
        if route is not None:
            rows = sum(_counts_for(ctx, geometry, state, *route))
            values.update(MOE_ROWS=rows, moe_rows=rows, etp_rows=rows)
        return values

    intra_pipe, inter_pipe = _link_pipes(ctx)
    stages = _chunk_stages(ctx, intra, inter, hbm, chunks, intra_pipe, inter_pipe, ranks, node_ranks)
    derived = (
        ("message_bytes", message), ("wire_bytes", Const(wire)),
        ("intra_bytes", intra), ("inter_bytes", inter),
        ("link_bytes", Add((intra, inter))),
        ("send_bytes", send), ("recv_hotspot", hotspot),
        ("aggregate_wire_bytes", aggregate),
        ("ranks", Const(ranks)), ("nodes", Const(math.ceil(ranks / node_ranks))),
        ("ranks_per_node", Const(node_ranks)), ("mtu_bytes", Const(mtu)),
        ("chunks", Const(chunks)), ("local_hbm_bytes", hbm),
        ("output_bytes", output),
        ("alltoall_matrix_ranks", Const(len(matrix[0]) if matrix else 0)),
        ("alltoall_matrix_cells", Const(len(matrix) * len(matrix[0])
                                        if matrix else 0)),
    )
    return FamilyTemplate(
        axes=axes, stages=stages, materialization=(), internal=(),
        residency=Const(0), derived=derived, regime_exprs=(), binder=binder,
        annotations=(("family", "collective"), ("collective", kind),
                     ("topology", topology)),
    )

def _compile_transport(ctx: CompiledOperator,
                       operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    name = operator.name.lower().replace("-", "_")
    kind = _collective_kind(name, theta)
    ranks = _ranks(theta)
    node_ranks = _node_ranks(theta, ranks)
    topology = _topology(theta, ranks, node_ranks)
    if topology == "hccs":
        node_ranks = ranks
    elif topology == "roce":
        node_ranks = 1
    row_bytes = _frac(theta.find("row_bytes", 0.0), 0.0)
    top_k = max(1, _int(theta.find("top_k", theta.find("k", 1)), 1))
    latent = max(1, _int(theta.find("latent", 1), 1))
    wire_bytes = _frac(theta.find("wire_bytes", 0.0), 0.0)
    chunks = max(1, _int(theta.find("chunks"), 1))
    overlap = max(0, _int(theta.find("overlap"), 0))
    mtu = max(1, _int(theta.find("mtu_bytes"), DEFAULT_MTU))
    # Accept either one latent-width row or a row already expanded by top-k;
    # normalize to one routed row and then apply top-k exactly once.
    declared_row = as_fraction(latent) * wire_bytes
    single_row = (declared_row
                  if declared_row > 0 and row_bytes in (declared_row,
                                                        declared_row * top_k)
                  else row_bytes) or declared_row
    rows = Mul((Var("T"), Const(top_k)))
    message = Mul((rows, Const(single_row)))
    matrix = _traffic_matrix(theta)
    if matrix is not None and len(matrix[0]) != ranks:
        raise ModelingError("collective_traffic_matrix_rank_count_mismatch")
    if matrix is not None:
        intra, inter, send, hotspot, aggregate, _ = _matrix_volumes(
            matrix[0], node_ranks, mtu, matrix[1], Var("T"))
        hbm = Add((send, hotspot))
    else:
        intra, inter, send, hotspot, aggregate = _uniform_alltoallv(
            message, ranks, node_ranks, mtu)
        hbm = Add((send, hotspot))
    intra, inter = _round_expr(intra, mtu), _round_expr(inter, mtu)
    mt = max(1, _int(theta.find("mt", ctx.policy.mt), ctx.policy.mt))
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("transport_tiles", "tokens", mt),
                        "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        return {"transport_ranks": ranks, "transport_chunks": chunks,
                "transport_overlap": overlap}

    intra_pipe, inter_pipe = _link_pipes(ctx)
    stages = _chunk_stages(ctx, intra, inter, hbm, chunks, intra_pipe, inter_pipe, ranks, node_ranks)
    derived = (
        ("routed_rows", rows), ("peer_bytes", message),
        ("transport_row_bytes", Const(single_row)),
        ("send_bytes", send), ("recv_hotspot", hotspot),
        ("aggregate_wire_bytes", aggregate),
        ("intra_bytes", intra), ("inter_bytes", inter),
        ("chunks", Const(chunks)), ("overlap", Const(overlap)),
        ("latent", Const(latent)), ("top_k", Const(top_k)),
        ("ranks", Const(ranks)), ("nodes", Const(math.ceil(ranks / node_ranks))),
        ("ranks_per_node", Const(node_ranks)), ("mtu_bytes", Const(mtu)),
        ("local_hbm_bytes", hbm),
    )
    return FamilyTemplate(
        axes=axes, stages=stages,
        materialization=(("peer_traffic", message),), internal=(),
        residency=Const(0), derived=derived, regime_exprs=(), binder=binder,
        annotations=(("family", "transport"), ("collective", kind),
                     ("topology", topology)),
    )
