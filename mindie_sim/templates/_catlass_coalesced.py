"""Bounded-stage exact K-panel coalesced lowering for large-grid CATLASS GEMMs.

[Role] Collapse inner K-panel stage repetition per output tile while preserving
  exact ``_normal_waves`` tile order, logical/issued FLOPs, directional HBM/L2
  byte ledgers, ping-pong pipeline overlap, and ``b_base_byte`` / ``b_version``
  obligations when ``blocks * gk`` exceeds ``COALESCED_PANEL_THRESHOLD``.
[Invariants] Activates only for regular ``dataparallel`` / ``swizzle`` RowMajor
  schedules without epilogues, MX scale groups, or gather/scatter when L2
  residency is provably invariant under K-coalescing; split-K/Stream-K schedules
  always use exact unrolled panel lowering.
[Timeouts & Locks] O(blocks) stage emission; no I/O, threads or locks.
[Side-Effect Policy] Appends stages to the caller's ``Lowering`` builder only.
"""
from __future__ import annotations

from mindie_sim.prediction.runtime.memory_physics import AddressRun

from .catlass_catalog import GATHER_SCATTER_POLICIES, packed_bytes
from .contracts import ceil_div

COALESCED_PANEL_THRESHOLD = 1024
_PINGPONG_MIN_STAGES = 2
_FMA_OPS_PER_MAC = 2
_SUPPORTED_COALESCED_SCHEDULERS = frozenset({"dataparallel", "swizzle"})


def _l2_coalesce_invariant(lowerer, blocks: int) -> bool:
    """True when coalescing K panels per tile preserves exact L2 hits/misses."""
    p, spec = lowerer.p, lowerer.spec
    l2_cap = dict(lowerer.plan.datapath.buffers).get("l2", 0)
    state = getattr(lowerer.plan, "state", None)
    if state is not None and hasattr(state, "effective_l2_bytes"):
        eff = int(state.effective_l2_bytes(lowerer.plan.die))
        if eff > 0:
            l2_cap = min(l2_cap, eff) if l2_cap > 0 else eff
    if l2_cap <= 0:
        return True
    scale_g = spec.mx_scale_group
    a_scale = p.m * ceil_div(p.k, scale_g) if scale_g else 0
    b_scale = p.n * ceil_div(p.k, scale_g) if scale_g else 0
    a_total = packed_bytes(p.m * p.k, spec.dtype_a) + a_scale
    b_total = packed_bytes(p.k * p.n, spec.dtype_b) + b_scale
    if a_total + b_total <= l2_cap:
        return True
    m_tiles = ceil_div(p.m, lowerer.tm)
    n_tiles = ceil_div(p.n, lowerer.tn)
    if m_tiles == 1:
        # Every tile reads the same A and a unique B column: A stays resident in LRU
        # L2 across tiles iff A plus one tile's full-K B column fits within L2.
        tile_b_scale = lowerer.tn * ceil_div(p.k, scale_g) if scale_g else 0
        tile_b = packed_bytes(lowerer.tn * p.k, spec.dtype_b) + tile_b_scale
        return a_total + tile_b <= l2_cap
    if n_tiles == 1:
        tile_a_scale = lowerer.tm * ceil_div(p.k, scale_g) if scale_g else 0
        tile_a = packed_bytes(lowerer.tm * p.k, spec.dtype_a) + tile_a_scale
        return b_total + tile_a <= l2_cap
    return False


def can_coalesce(lowerer, blocks: int) -> bool:
    """Return True when exact per-tile K-coalesced lowering applies."""
    spec = lowerer.spec
    return (
        blocks * lowerer.gk > COALESCED_PANEL_THRESHOLD
        and lowerer.gk > 1
        and spec.scheduler in _SUPPORTED_COALESCED_SCHEDULERS
        and spec.epilogue_kind == "none"
        and spec.dispatch_policy not in GATHER_SCATTER_POLICIES
        and spec.dispatch_policy != "MmadPingpongMutex"
        and not spec.prologue_b
        and (not spec.mx_scale_group or lowerer.tk % spec.mx_scale_group == 0)
        and not spec.simt_thread_block_dim
        and spec.layout_a == "RowMajor"
        and spec.layout_b == "RowMajor"
        and spec.layout_c == "RowMajor"
        and _l2_coalesce_invariant(lowerer, blocks)
    )


def _k_micro_issued(lowerer, tile_m: int, tile_n: int, k_total: int, issued_fn) -> int:
    full_panels, rem_k = divmod(k_total, lowerer.tk)
    panel_issued = 0
    panel_k = min(lowerer.tk, k_total)
    for start in range(0, panel_k, lowerer.micro_k):
        micro = min(panel_k - start, lowerer.micro_k)
        panel_issued += issued_fn(tile_m, tile_n, micro, lowerer.spec, lowerer.plan.datapath)
    total = full_panels * panel_issued
    if rem_k:
        for start in range(0, rem_k, lowerer.micro_k):
            micro = min(rem_k - start, lowerer.micro_k)
            total += issued_fn(tile_m, tile_n, micro, lowerer.spec, lowerer.plan.datapath)
    return total


def _lower_coalesced_tile(
    lowerer, idx: int, tile, after: tuple[str, ...], *,
    scale_fn, compute_b_fn, cube_dtype_fn, issued_fn,
    dtype_b_comp: str, cube_dt: str, a_stride: int, b_stride: int, c_stride: int,
    pingpong: bool, first_k: int, rest_k: int,
) -> str:
    p, s, plan = lowerer.p, lowerer.spec, lowerer.plan
    core = tile.owner
    tag = f"coal:t{idx}"
    a_first_w = packed_bytes(first_k, s.dtype_a)
    b_col_off = packed_bytes(tile.col, s.dtype_b)
    b_row_w = packed_bytes(tile.n, s.dtype_b)
    b_bypass = s.l2_cache_hint_b == "disable"
    scale_g = s.mx_scale_group
    scale_per_row = ceil_div(p.k, scale_g) if scale_g else 0

    a_prologue = plan.access(
        f"{tag}:a_prologue", p.a_id, tile.m * a_first_w, owner=core, pipe="mte2_aic",
        runs=(AddressRun(tile.row * a_stride, tile.m, a_first_w, a_stride),),
        after=after,
    )
    if scale_g:
        a_sc_cols = ceil_div(first_k, scale_g)
        a_prologue = plan.access(
            f"{tag}:a_prologue:scale_a", p.a_id + ":scale", tile.m * a_sc_cols,
            runs=(AddressRun(tile.row * scale_per_row, tile.m, a_sc_cols, scale_per_row),),
            owner=core, pipe="mte2_aic", after=(a_prologue,),
        )
    b_prologue = plan.access(
        f"{tag}:b_prologue", p.b_id, first_k * b_row_w, owner=core, pipe="mte2_aic",
        runs=(AddressRun(p.b_base_byte + b_col_off, first_k, b_row_w, b_stride),),
        after=after, version=p.b_version, bypass=b_bypass,
    )
    if scale_g:
        b_sc_cols = ceil_div(first_k, scale_g)
        b_prologue = plan.access(
            f"{tag}:b_prologue:scale_b", p.b_id + ":scale", tile.n * b_sc_cols,
            runs=(AddressRun(tile.col * scale_per_row, tile.n, b_sc_cols, scale_per_row),),
            owner=core, pipe="mte2_aic", after=(b_prologue,), bypass=b_bypass,
        )
    compute_after: tuple[str, ...] = (a_prologue, b_prologue)
    a_tail = a_prologue
    b_tail = b_prologue
    if rest_k > 0:
        a_rest_w = packed_bytes(rest_k, s.dtype_a)
        a_tail = plan.access(
            f"{tag}:a_rest", p.a_id, tile.m * a_rest_w, owner=core, pipe="mte2_aic",
            runs=(AddressRun(tile.row * a_stride + a_first_w, tile.m, a_rest_w, a_stride),),
            after=(a_prologue,),
        )
        if scale_g:
            a_sc_rest = ceil_div(rest_k, scale_g)
            a_tail = plan.access(
                f"{tag}:a_rest:scale_a", p.a_id + ":scale", tile.m * a_sc_rest,
                runs=(AddressRun(tile.row * scale_per_row + first_k // scale_g, tile.m, a_sc_rest, scale_per_row),),
                owner=core, pipe="mte2_aic", after=(a_tail,),
            )
        b_tail = plan.access(
            f"{tag}:b_rest", p.b_id, rest_k * b_row_w, owner=core, pipe="mte2_aic",
            runs=(AddressRun(p.b_base_byte + first_k * b_stride + b_col_off, rest_k, b_row_w, b_stride),),
            after=(b_prologue,), version=p.b_version, bypass=b_bypass,
        )
        if scale_g:
            b_sc_rest = ceil_div(rest_k, scale_g)
            b_tail = plan.access(
                f"{tag}:b_rest:scale_b", p.b_id + ":scale", tile.n * b_sc_rest,
                runs=(AddressRun(tile.col * scale_per_row + first_k // scale_g, tile.n, b_sc_rest, scale_per_row),),
                owner=core, pipe="mte2_aic", after=(b_tail,), bypass=b_bypass,
            )
        if not pingpong:
            compute_after = (a_tail, b_tail)

    mte1_bytes = (
        packed_bytes(tile.m * p.k, s.dtype_a) + scale_fn(s, tile.m * p.k)
        + packed_bytes(tile.n * p.k, dtype_b_comp) + scale_fn(s, tile.n * p.k)
    )
    mte1_stage = plan.work(f"{tag}:mte1", "mte1", mte1_bytes, owner=core, after=compute_after)
    logical_flops = _FMA_OPS_PER_MAC * tile.m * tile.n * p.k
    issued_flops = _k_micro_issued(lowerer, tile.m, tile.n, p.k, issued_fn)
    mma_after = compute_after if pingpong else (mte1_stage,)
    if s.has_bias:
        bias_stage = plan.access(
            f"{tag}:bias", p.c_id + ":bias", packed_bytes(tile.n, s.dtype_acc),
            owner=core, pipe="mte2_aic", offset=packed_bytes(tile.col, s.dtype_acc), after=after,
        )
        mma_after = (*mma_after, bias_stage)
    mma_stage = plan.work(
        f"{tag}:mma", "cube", issued_flops, owner=core, dtype=cube_dt,
        after=mma_after, metadata={"logical_flops": logical_flops},
    )
    c_row_w = packed_bytes(tile.n, s.dtype_c)
    c_run = AddressRun(tile.row * c_stride + packed_bytes(tile.col, s.dtype_c), tile.m, c_row_w, c_stride)
    out_after = tuple(dict.fromkeys((a_tail, b_tail, mte1_stage, mma_stage)))
    return plan.access(
        f"{tag}:fixpipe", p.c_id, tile.m * c_row_w, owner=core, pipe="fixpipe",
        direction="write", runs=(c_run,), after=out_after,
    )


def lower_coalesced(
    lowerer, launch: str, blocks: int, *, scale_fn, compute_b_fn, cube_dtype_fn, issued_fn,
) -> None:
    """Emit exact wave-ordered K-coalesced stages for a large regular GEMM."""
    p, s = lowerer.p, lowerer.spec
    dtype_b_comp = compute_b_fn(s)
    cube_dt = cube_dtype_fn(s)
    a_stride = packed_bytes(p.k, s.dtype_a)
    b_stride = packed_bytes(p.n, s.dtype_b)
    c_stride = packed_bytes(p.n, s.dtype_c)
    pingpong = s.stages[0] >= _PINGPONG_MIN_STAGES
    first_k = min(lowerer.tk, p.k)
    rest_k = p.k - first_k
    cores = max(1, s.aic_budget)
    after: tuple[str, ...] = (launch,)
    for first in range(0, blocks, cores):
        stop = min(first + cores, blocks)
        after = tuple(
            _lower_coalesced_tile(
                lowerer, idx, lowerer.tile(idx), after,
                scale_fn=scale_fn, compute_b_fn=compute_b_fn,
                cube_dtype_fn=cube_dtype_fn, issued_fn=issued_fn,
                dtype_b_comp=dtype_b_comp, cube_dt=cube_dt,
                a_stride=a_stride, b_stride=b_stride, c_stride=c_stride,
                pingpong=pingpong, first_k=first_k, rest_k=rest_k,
            )
            for idx in range(first, stop)
        )
