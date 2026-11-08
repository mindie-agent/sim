"""Matrix, memory, indexer and elementwise piecewise family models.

[Role]
  Derive dense/quantised GEMM, KV access, norm/activation/copy and embedding
  resource demands from declared structure and physical tiles.  MoE grouped
  GEMM, collective and expert transport live in the sibling
  ``families_moe_collective`` module and are re-exported here for the registry.
  The refined indexer/compression programs live in
  ``families_recurrent_indexer``.

[Invariants]
  * Cube work uses 16/32-aligned fractal extents and a capacity-screened
    ``(M_t, N_t, K_t)`` step.  Short M charges ``align(M, 16)`` rows.  HBM
    weights move once per N-tile/split; MTE1 may replay the resident panel.
  * One output tile is serial load/compute/store; many tiles use
    ``fill + (q-1) * max_pipe``.  MTE2 bytes round to the transaction granule.
  * Elementwise, norm, quant and copy pay at least one AIV repeat per core
    per token wave.  UB rows still round to 32 bytes.  No model name,
    shape-keyed lookup or measurement discount selects a tile or a latency.

[Timeouts & Locks]
  Pure bounded expression construction.  No I/O, waits or locks.

[Side-Effect Policy]
  Returns immutable ``FamilyTemplate`` records; no global mutation.
"""
from __future__ import annotations

import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from .compiler import (
    CompiledOperator, FamilyTemplate, SegmentAxis, _demand, _frac, _out_name,
    _zero,
)
from .expressions import (
    Add, CeilDiv, Const, Div, Expr, Max, Min, Mul, Sub, Var, as_fraction,
)
from .guards import SplitKGuard, StepGuard
from .families_moe_collective import (
    DEFAULT_L0A_BYTES, DEFAULT_L0B_BYTES, DEFAULT_L0C_BYTES,
    DEFAULT_L1_BYTES, DEFAULT_MTE2_GRANULE, FRACTAL, UB_ALIGN,
    _compile_collective, _compile_grouped_gemm, _compile_router,
    _compile_transport, _counts_for, _is_quant, _merge_groups, _round_expr,
    _scale, _stage, _width, tile_plan,
)

# One vector repeat is eight 32-byte UB datablocks.  Counted as element-ops
# on the vector pipe, never as a fitted microsecond.
AIV_REPEAT_BYTES = UB_ALIGN * 8


def _theta_int(theta: Any, name: str, default: int) -> int:
    value = theta.find(name, None)
    return default if value is None else int(value)


def _theta_frac(theta: Any, name: str, default: Any) -> Any:
    value = theta.find(name, None)
    return as_fraction(default) if value is None else as_fraction(value)


def _row_bytes(value: Any) -> int:
    exact = as_fraction(value)
    return int(math.ceil(exact / UB_ALIGN)) * UB_ALIGN


def _aiv_quantum(elem_bytes: Any) -> int:
    width = as_fraction(elem_bytes)
    if width <= 0:
        return AIV_REPEAT_BYTES
    return max(1, int(math.ceil(AIV_REPEAT_BYTES / width)))


def _aiv_floor(rows: Expr, tile: int, cores: int, elem_bytes: Any) -> Expr:
    waves = CeilDiv(rows, Const(max(1, int(tile))))
    c = max(1, int(cores))
    return Mul((waves, Const(c * _aiv_quantum(elem_bytes))))


_MX_FP8 = {"mxfp8", "mx_fp8", "mxfp8_e4m3"}
_MX_FP4 = {"mxfp4", "mx_fp4", "mxfp4_e2m1"}


def _payload_width(fmt: Any) -> float:
    """Packed element width.  Group scales stay on their own tensor."""
    text = str(fmt or "").strip().lower().replace("-", "_")
    if text in _MX_FP8:
        return 1.0
    if text in _MX_FP4:
        return 0.5
    return float(_width(fmt))


def _quant_output_width(theta: Any) -> Any:
    out_format = str(theta.find("output_format", "") or "").strip()
    if out_format:
        return _payload_width(out_format)
    raw = theta.find("quant_bytes", None)
    return as_fraction(1 if raw is None else raw)


def _dynamic_quant_vector(rows: Expr, cols: int) -> Expr:
    """Group reduction, scale encode, rescale, cast and pack.

    ``round(H*59/8)`` is the per-row arithmetic inventory.  The 1024-term is
    one helper constant per 256-wide row tile, never a rate-selected floor.
    """
    width = max(1, int(cols))
    arithmetic = round(width * 59.0 / 8.0)
    tiles = math.ceil(width / 256)
    return Add((
        Mul((rows, Const(arithmetic))),
        Mul((Max((rows, Const(1))), Const(1024 * tiles))),
    ))


def _gemm_mode(name: str, theta: Any) -> str:
    text = name.lower().replace("_", "").replace("-", "")
    weight_format = str(theta.find("weight_format", "bf16") or "bf16").lower()
    if "transpose" in text or bool(int(theta.find("transpose_b", 0) or 0)):
        return "transpose_batch_matmul"
    if weight_format in {"mxfp8", "mx_fp8", "mxfp8_e4m3"}:
        return "quant_batch_matmul_v3_mxfp8"
    if "weightquant" in text:
        return "weight_quant_batch_matmul"
    if "quantbatch" in text or weight_format in {
            "w8a8", "w4a8", "int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2"}:
        return "quant_batch_matmul_v3"
    return "matmul_v2"


def _compile_gemm(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    k, n = int(theta.require("K")), int(theta.require("N"))
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    nt = _theta_int(theta, "nt", ctx.policy.nt)
    kt = _theta_int(theta, "kt", ctx.policy.kt)
    cores = max(1, _theta_int(theta, "cores", ctx.policy.cores))
    act_bytes = _theta_frac(theta, "act_bytes", 2.0)
    act_format = str(theta.find("act_format", "bf16") or "bf16")
    weight_format = str(theta.find("weight_format", "bf16") or "bf16")
    weight_bytes = _payload_width(weight_format)
    out_format = str(theta.find("output_format", act_format) or act_format)
    out_bytes = _payload_width(out_format)
    weight_total = _theta_frac(theta, "weight_total", n * k * weight_bytes)
    weight_payload = _theta_frac(theta, "weight_payload", weight_total)
    weight_scale = _theta_frac(theta, "weight_scale", 0.0)
    weight_zero = _theta_frac(theta, "weight_zero", 0.0)
    granule = max(1, _theta_int(theta, "mte2_granule", DEFAULT_MTE2_GRANULE))
    slots = max(1, _theta_int(theta, "pipeline_slots",
                             _theta_int(theta, "slots", ctx.policy.pipeline_slots)))
    plan = tile_plan(
        mt, n, k, mt=mt, nt=nt, kt=kt, act_bytes=float(act_bytes),
        weight_bytes=weight_bytes, out_bytes=out_bytes,
        l1_bytes=_theta_int(theta, "l1_bytes", DEFAULT_L1_BYTES),
        l0a_bytes=_theta_int(theta, "l0a_bytes", DEFAULT_L0A_BYTES),
        l0b_bytes=_theta_int(theta, "l0b_bytes", DEFAULT_L0B_BYTES),
        l0c_bytes=_theta_int(theta, "l0c_bytes", DEFAULT_L0C_BYTES),
        slots=slots, compute_slots=max(1, _theta_int(theta, "compute_slots", 2)))
    if not plan.fits:
        raise ModelingError("tile_plan_capacity_exceeded")
    mode = _gemm_mode(operator.name, theta)
    batch = max(1, _theta_int(theta, "batch", 1))
    m_tile, n_tile, k_tile = plan.m_tile, plan.n_tile, plan.k_tile
    n_tiles, k_panels = plan.n_tiles, plan.k_panels
    split_policy = ctx.policy.split_policy(
        target=_theta_int(theta, "split_target", ctx.policy.split_target),
        max_splits=_theta_int(theta, "max_splits", ctx.policy.max_splits))
    selected_rows = bool(int(theta.find("selected_rows", 0) or 0))
    axes = (
        SegmentAxis("M_TILES", "gemm_tokens",
                    StepGuard("gemm_m_tiles", "gemm_tokens", m_tile),
                    "max_tokens"),
        SegmentAxis("SPLITS", "gemm_work",
                    SplitKGuard(split_policy, coordinate="gemm_work"), "max_work"),
    )

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        if selected_rows:
            # One externally consumed row per request.  An unfinished prefill
            # with no explicit ranges emits each request's last valid row, so
            # the batch width wins over the full prompt row count.
            ranged = any(request.query_ranges for request in geometry.requests)
            if not ranged and geometry.phase_name == "prefill":
                tokens = geometry.batch
            elif hasattr(geometry, "selected_tokens"):
                tokens = int(geometry.selected_tokens)
            else:
                tokens = geometry.batch
        else:
            tokens = geometry.effective_tokens
        # Split-K beyond the physical K panel count duplicates work without
        # shortening a panel.  Cap the guard coordinate on that lattice.
        panel_cap = max(1, int(as_fraction(split_policy.target) * plan.k_panels))
        return {"gemm_work": max(1, min(tokens * n * k, panel_cap)),
                "gemm_tokens": tokens}

    m_tiles, splits = Var("M_TILES"), Var("SPLITS")
    tokens = Var("gemm_tokens")
    m_pad = Mul((m_tiles, Const(m_tile)))
    fractal_m = Mul((CeilDiv(tokens, Const(FRACTAL)), Const(FRACTAL)))
    # Tail tiles issue only their aligned remainder.  Averages conserve
    # align(M), n_pad and k_pad without charging a full macro-tile per tail.
    tile_rows = Div(fractal_m, m_tiles)
    n_issue = Div(Const(plan.n_pad), Const(n_tiles))
    k_issue = Div(Const(plan.k_pad), Const(k_panels))
    k_split = Div(Const(k_panels), splits)
    k_chunk = Mul((k_split, k_issue))
    tiles = Mul((m_tiles, Const(n_tiles), splits, Const(batch)))
    waves = CeilDiv(tiles, Const(cores))
    gate = Min((tiles, Const(1)))
    active = Min((Const(cores), tiles))
    rest_tiles = Sub(tiles, active)
    internal = Const(1 if slots >= 2 else 0)
    metadata_total = weight_scale + weight_zero
    metadata_panel = (Div(Mul((Const(metadata_total), k_issue, n_issue)),
                          Const(max(1, k * n)))
                      if metadata_total else Const(0))
    act_panel = Mul((tile_rows, k_issue, Const(act_bytes)))
    weight_panel = Add((
        Mul((k_issue, n_issue, Const(weight_bytes))),
        metadata_panel,
    ))
    combined_panel = Add((act_panel, weight_panel))
    combined_granule = _round_expr(combined_panel, granule)
    act_granule = _round_expr(act_panel, granule)
    extra_tiles = Sub(m_tiles, Const(1))
    # First scheduled M tile of an N-tile/split pays the weight granule.
    # Later M tiles pay the activation granule only.  m_tiles == 1 is the
    # combined granule, so single-tile rounding stays exact.
    mte2_panel = Div(Add((combined_granule, Mul((extra_tiles, act_granule)))),
                     m_tiles)
    hbm_panel = Div(Add((combined_panel, Mul((extra_tiles, act_panel)))),
                    m_tiles)
    mte1_panel = combined_granule
    act_chunk = Mul((tile_rows, k_chunk, Const(act_bytes)))
    weight_chunk = Add((
        Mul((k_chunk, n_issue, Const(weight_bytes))),
        Mul((metadata_panel, k_split)),
    ))
    load_tile = Div(Add((Add((act_chunk, weight_chunk)),
                         Mul((extra_tiles, act_chunk)))), m_tiles)
    load_granule = Mul((k_split, mte2_panel))
    panel_compute = Mul((Const(2), tile_rows, n_issue, k_issue))
    compute_tile = Mul((Const(2), tile_rows, n_issue, k_chunk))
    partial_tile = Mul((tile_rows, n_issue, Const(4),
                        Min((Sub(splits, Const(1)), Const(1)))))
    fixpipe = Mul((tile_rows, n_issue, Const(4)))
    act_quant = Const(int(_is_quant(act_format)))
    weight_quant = Const(int(_is_quant(weight_format)))
    panel_vector = Add((
        Mul((tile_rows, k_issue, act_quant)),
        Mul((k_issue, n_issue, weight_quant)),
    ))
    transpose_bytes = (batch * n * k * weight_bytes * 2
                       if mode == "transpose_batch_matmul" else 0.0)
    # Double buffer: one panel fills, the rest overlap, one panel drains.
    # Never replay every panel just because a later output wave exists.
    if slots >= 2:
        fill_panels = Min((Const(1), k_split))
        steady_panels = Max((Const(0), Sub(k_split, Const(1))))
    else:
        fill_panels = k_split
        steady_panels = Const(0)
    load_mem = (("mte2", mte2_panel), ("mte1", mte1_panel))
    load_hbm = (("hbm", hbm_panel),)
    compute = (("cube", panel_compute), ("vector", panel_vector))
    fixpipe_group = (("fixpipe", fixpipe),)
    store = (("mte3", partial_tile), ("hbm", partial_tile))
    load_stage = _stage(ctx, _merge_groups(
        _scale(load_mem, Mul((gate, active, fill_panels))),
        _scale(load_hbm, Mul((gate, active, fill_panels)))))
    compute_stage = _stage(ctx, _merge_groups(
        _scale(compute, Mul((gate, Const(cores), fill_panels))),
        _scale(fixpipe_group, Mul((gate, Const(cores))))))
    first_store = _stage(ctx, _scale(store, Mul((gate, active))))
    rest_mem = _scale(load_mem, Mul((gate, rest_tiles, k_split)))
    rest_hbm = _scale(load_hbm, Mul((gate, rest_tiles, k_split)))
    rest_compute = _scale(compute, Mul((gate, rest_tiles, k_split)))
    rest_fixpipe = _scale(fixpipe_group, Mul((gate, rest_tiles)))
    rest_store = _scale(store, Mul((gate, rest_tiles)))
    internal_factor = Mul((gate, internal, steady_panels))
    internal_stage = _stage(ctx, _merge_groups(
        _scale(load_mem, Mul((active, internal_factor))),
        _scale(load_hbm, Mul((active, internal_factor))),
        _scale(compute, Mul((Const(cores), internal_factor)))))
    if slots >= 2:
        if k_panels > 1:
            # Every split accumulates all of its K panels before its partial
            # tile is published; only then does the merge consume it.
            stages = [
                load_stage, compute_stage,
                _stage(ctx, _merge_groups(rest_mem, rest_hbm, rest_compute,
                                          rest_fixpipe)),
                internal_stage,
                _stage(ctx, _scale(store, Mul((gate, Add((active,
                                                          rest_tiles)))))),
            ]
        else:
            stages = [
                load_stage, compute_stage, first_store,
                _stage(ctx, _merge_groups(rest_mem, rest_hbm, rest_compute,
                                          rest_fixpipe, rest_store)),
            ]
    else:
        stages = [
            load_stage, compute_stage, first_store,
            _stage(ctx, _merge_groups(rest_mem, rest_hbm)),
            _stage(ctx, _merge_groups(rest_compute, rest_fixpipe)),
            _stage(ctx, rest_store),
        ]
    if transpose_bytes:
        stages.insert(0, (
            _demand(ctx, "mte2", Const(transpose_bytes)),
            _demand(ctx, "hbm", Const(transpose_bytes)),
        ))
    out = Mul((Const(batch), tokens, Const(n), Const(out_bytes)))
    partial = Mul((Sub(splits, Const(1)), Const(batch),
                   tokens, Const(n), Const(4)))
    partial_write = Mul((tiles, partial_tile))
    merge = (
        _demand(ctx, "mte2", partial),
        _demand(ctx, "mte3", out),
        _demand(ctx, "hbm", Add((partial, out))),
        _demand(ctx, "vector", Mul((Sub(splits, Const(1)), Const(batch),
                                    tokens, Const(n)))),
    )
    stages.append(merge)
    useful = Mul((tiles, load_tile))
    padded = Mul((tiles, load_granule))
    footprint = plan.footprint(float(act_bytes), weight_bytes, out_bytes)
    derived = (
        ("flops", Mul((Const(2 * batch), tokens, Const(n), Const(k)))),
        ("padded_flops", Mul((Const(2 * batch), m_pad, Const(plan.n_pad),
                              Const(plan.k_pad)))),
        ("fractal_flops", Mul((Const(2 * batch), fractal_m, Const(plan.n_pad),
                               Const(plan.k_pad)))),
        ("fractal_m", fractal_m), ("fractal_n", Const(plan.n_pad)),
        ("fractal_k", Const(plan.k_pad)), ("k_align", Const(plan.k_align)),
        ("tile_padded_m", m_pad), ("charged_m", tile_rows),
        ("tile_m", Const(m_tile)), ("tile_n", Const(n_tile)),
        ("tile_k", Const(k_tile)), ("k_inner", Const(plan.k_inner)),
        ("tile_n_tiles", Const(n_tiles)), ("n_tiles", Const(n_tiles)),
        ("k_panels", Const(k_panels)), ("k_split_panels", k_split),
        ("k_chunk", k_chunk), ("splits", splits),
        ("j_tiles", tiles), ("m_tiles", m_tiles), ("waves", waves),
        ("active_cores", active),
        ("tail_rows", Sub(tokens, Mul((Sub(m_tiles, Const(1)),
                                       Const(m_tile))))),
        ("weight_payload", Const(weight_payload)),
        ("weight_scale", Const(weight_scale)),
        ("weight_zero", Const(weight_zero)),
        ("weight_total", Const(weight_total)),
        ("output_bytes", out), ("padded_output_bytes",
                                Mul((Const(batch), m_pad, Const(plan.n_pad),
                                     Const(out_bytes)))),
        ("partial_bytes", Add((partial_write, partial))),
        ("partial_write_bytes", partial_write),
        ("partial_read_bytes", partial),
        ("transpose_bytes", Const(transpose_bytes)),
        ("tile_load_bytes", load_tile), ("tile_compute_flops", compute_tile),
        ("tile_load_granule_bytes", load_granule),
        ("tile_store_bytes", partial_tile), ("tile_fixpipe_bytes", fixpipe),
        ("mte2_granule_bytes", Const(granule)),
        ("mte2_useful_bytes", useful), ("mte2_padded_bytes", padded),
        ("mte2_efficiency", Div(useful, Max((padded, Const(1))))),
        ("l1_tile_bytes", Const(footprint["l1"])),
        ("l0a_tile_bytes", Const(footprint["l0a"])),
        ("l0b_tile_bytes", Const(footprint["l0b"])),
        ("l0c_tile_bytes", Const(footprint["l0c"])),
        ("pipeline_slots", Const(slots)), ("pipeline_q", waves),
        ("pipeline_fill_tiles", Const(1)), ("pipeline_drain_tiles", Const(0)),
        ("tile_capacity_fits", Const(int(plan.fits))), ("batch", Const(batch)),
        ("quantized", Const(int(_is_quant(weight_format)
                                 or _is_quant(act_format)))),
    )
    return FamilyTemplate(
        axes=axes, stages=tuple(stages),
        materialization=((_out_name(ctx), out),), internal=(),
        residency=Const(weight_total), derived=derived,
        regime_exprs=(), binder=binder,
        annotations=(("family", "gemm"), ("mode", mode), ("K", str(k)),
                     ("N", str(n)), ("weight_format", weight_format),
                     ("act_format", act_format)),
    )


def _compile_kv_access(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    row_bytes = _frac(theta.require("row_bytes"))
    coordinate = "kv_positions" if operator.name == "kv_read" else "kv_rows"
    axes = (SegmentAxis("ROWS", coordinate,
                        StepGuard("kv_rows", coordinate, 1),
                        "max_s_kv" if operator.name == "kv_read" else "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        positions = max(1, sum(request.kv_len for request in geometry.requests))
        rows = max(1, geometry.effective_tokens)
        owner = theta.find("owner_layer", -1)
        if operator.name == "kv_read" and owner is not None and int(owner) >= 0:
            state.require_kv(int(owner))
        return {"kv_positions": positions, "kv_rows": rows}

    rows = Var("ROWS")
    traffic = Mul((rows, Const(row_bytes)))
    if operator.name == "kv_read":
        stage = (_demand(ctx, "mte2", traffic), _demand(ctx, "hbm", traffic),
                 _demand(ctx, "link", _zero()))
        materialization = ()
    else:
        stage = (_demand(ctx, "mte3", traffic), _demand(ctx, "hbm", traffic))
        materialization = (("kv_cache", traffic),)
    return FamilyTemplate(
        axes=axes, stages=(stage,), materialization=materialization,
        internal=(), residency=Const(0),
        derived=(("kv_positions", rows), ("kv_bytes", traffic)),
        regime_exprs=(), binder=binder,
        state_effects=(("kv", operator.name),),
        annotations=(("family", "kv_access"), ("row_bytes", str(row_bytes))),
    )


def _compile_elementwise(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    elements = int(theta.require("elements"))
    ops = int(theta.find("ops_per_element", 1) or 1)
    act_bytes = _theta_frac(theta, "act_bytes", 2.0)
    read = _theta_frac(theta, "read_bytes", elements * act_bytes)
    write = _theta_frac(theta, "write_bytes", elements * act_bytes)
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    cores = max(1, _theta_int(theta, "cores", ctx.policy.cores))
    moe_top_k = max(1, _theta_int(theta, "k",
                                  _theta_int(theta, "top_k", 1)))
    moe_local = max(1, _theta_int(theta, "local",
                                  _theta_int(theta, "E_local", 1)))
    text = (operator.name + " " + operator.role).lower().replace("_", "")
    router_op = any(word in text for word in ("topk", "histogram", "sort",
                                              "router"))
    f3_domain = (ctx.kind == "F3"
                 or theta.find("expert_width", None) is not None
                 or (theta.find("k", None) is not None and not router_op))
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("token_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        rows = geometry.effective_tokens
        if f3_domain:
            # Routed activation lives on the same local-expert domain as the
            # adjacent grouped GEMM: an explicit local shard is consumed as
            # delivered, otherwise the global uniform distribution is
            # localized to this EP shard.
            experts = max(moe_local, _theta_int(theta, "E", moe_local))
            counts = geometry.routing_for(ctx.layer)
            if counts is None:
                try:
                    counts = state.routing_state(ctx.layer).counts
                except ModelingError:
                    counts = None
            if counts is None:
                counts = _counts_for(ctx, geometry, state, moe_local,
                                     moe_top_k, experts)
            elif len(counts) == experts and experts != moe_local:
                counts = tuple(counts)[:moe_local]
            rows = sum(int(value) for value in counts)
        return {"elements_per_token": elements, "routed_rows": rows,
                "ROUTED_ROWS": rows, "vector_ops_per_token": elements * ops,
                "read_per_token": read, "write_per_token": write}

    tokens = Var("ROUTED_ROWS") if f3_domain else Var("T")
    quantized = (bool(int(theta.find("quantized", 0) or 0)) or "quant" in text
                 or operator.family in {"quantize", "quant"})
    read_row = _row_bytes(read)
    if quantized:
        out_fmt = str(theta.find("out_format", theta.find("output_format", "")) or "").lower()
        mx_scale = math.ceil(elements / 32) if out_fmt in (_MX_FP8 | _MX_FP4) else 0
        write_row = _row_bytes(as_fraction(elements) * as_fraction(_quant_output_width(theta)) + mx_scale)
    else:
        write_row = _row_bytes(write)
    gated = 2 if ("glu" in text or "geglu" in text or "swiglu" in text) else 1
    reduction = int(math.ceil(elements / 64))
    quant_vector = (_dynamic_quant_vector(tokens, elements) if quantized
                    else Const(0))
    if "norm" in text or "rms" in text or "layer" in text:
        vector = Add((Mul((tokens, Const(elements * ops))),
                      Mul((tokens, Const(reduction)))))
        if quantized:
            vector = Add((vector, quant_vector))
    elif "histogram" in text or "sort" in text or "topk" in text:
        vector = Mul((tokens, Const(elements * max(1, ops))))
    elif quantized and gated == 1:
        vector = quant_vector
    else:
        vector = Mul((tokens, Const(elements * ops * gated)))
        if quantized:
            vector = Add((vector, quant_vector))
    if "layoutconvert" in text or ("cast" in text and "broadcast" not in text):
        vector = Add((vector, Mul((tokens, Const(elements * 2)))))
    reads = Mul((tokens, Const(read_row)))
    writes = Mul((tokens, Const(write_row)))
    gather = "gather" in text
    if gather:
        reads = Add((reads, Mul((tokens, Const(
            _theta_frac(theta, "index_bytes", 8.0))))))
    vector_pipe = Max((vector, _aiv_floor(tokens, mt, cores, act_bytes)))
    main = (
        _demand(ctx, "vector", vector_pipe), _demand(ctx, "mte2", reads),
        _demand(ctx, "mte3", writes), _demand(ctx, "hbm", Add((reads, writes))),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=((_out_name(ctx), writes),),
        internal=(), residency=Const(0),
        derived=(("elements", Mul((tokens, Const(elements)))),
                 ("vector_ops", vector),
                 ("ub_row_bytes", Const(read_row)),
                 ("ub_write_row_bytes", Const(write_row)),
                 ("ub_padded_bytes", reads),
                 ("vector_reduction_ops", Mul((tokens, Const(reduction)))),
                 ("gated_inputs", Mul((tokens, Const(elements * gated)))),
                 ("gather_indices", tokens if gather else Const(0)),
                 ("quantized_output", Const(int(quantized)))),
        regime_exprs=(), binder=binder,
        annotations=(("family", "elementwise"), ("ops_per_element", str(ops))),
    )


def _compile_copy(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    elements = int(theta.require("elements"))
    elem_bytes = _theta_frac(theta, "act_bytes", 2.0)
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    cores = max(1, _theta_int(theta, "cores", ctx.policy.cores))
    text = (operator.name + " " + operator.role).lower()
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("copy_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        return {"copy_elements": elements, "copy_bytes": elements * elem_bytes}

    read_bytes = _theta_frac(theta, "read_bytes", elements * elem_bytes)
    write_bytes = _theta_frac(theta, "write_bytes", elements * elem_bytes)
    traffic = Mul((Var("T"), Const(write_bytes)))
    read = Mul((Var("T"), Const(read_bytes)))
    if "gather" in text:
        index_bytes = _theta_frac(theta, "index_bytes", 8.0)
        read = Add((read, Mul((Var("T"), Const(index_bytes)))))
    lane = Mul((Var("T"), Const(elements * 2)))
    main = (
        _demand(ctx, "vector", Max((lane, _aiv_floor(
            Var("T"), mt, cores, elem_bytes)))),
        _demand(ctx, "mte2", read), _demand(ctx, "mte3", traffic),
        _demand(ctx, "hbm", Add((read, traffic))),
    )
    effects = (("residual", "push"),) if ctx.kind == "R1" and ctx.name == "slot_push" else ()
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=((_out_name(ctx), traffic),),
        internal=(), residency=Const(0),
        derived=(("copy_bytes", traffic),
                 ("gather_indices", Var("T") if "gather" in text else Const(0))),
        regime_exprs=(), binder=binder, state_effects=effects,
        annotations=(("family", "copy"),),
    )


def _compile_embedding(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    width = int(theta.require("K")) if theta.find("K", None) is not None else int(theta.require("width"))
    elem_bytes = _theta_frac(theta, "act_bytes", 2.0)
    weight_total = _theta_frac(theta, "weight_total", 0.0)
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("embed_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        return {"embedding_width": width, "embedding_bytes": width * elem_bytes}

    traffic = Mul((Var("T"), Const(width * elem_bytes)))
    main = (
        _demand(ctx, "mte2", traffic), _demand(ctx, "mte3", traffic),
        _demand(ctx, "hbm", Add((traffic, traffic))),
        _demand(ctx, "vector", Mul((Var("T"), Const(width)))),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=(("hidden", traffic),),
        internal=(), residency=Const(weight_total),
        derived=(("gather_bytes", traffic), ("gather_indices", Var("T")),
                 ("embedding_width", Const(width))),
        regime_exprs=(), binder=binder,
        annotations=(("family", "embedding"),),
    )
