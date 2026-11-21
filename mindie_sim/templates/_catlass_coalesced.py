"""Bounded-stage exact coalesced lowering for large-grid CATLASS GEMM schedules.

[Role] Preserve exact logical FLOPs, issued cube ops, directional HBM/L2 byte
  ledgers, and per-core ping-pong pipeline overlap when ``blocks * gk`` exceeds
  the per-panel unroll threshold, preventing 100k+ Python stage explosions on
  production LLM projection shapes.
[Invariants] Conserves every FLOP, MTE2/MTE1/Fixpipe byte, ``AddressRun``
  coverage, cold/warm L2 reuse on repeated A rows, and ``b_base_byte`` /
  ``b_version`` obligations identically to unrolled panel lowering.
[Timeouts & Locks] O(blocks + aic_budget) stage emission; no I/O or locks.
[Side-Effect Policy] Appends stages to the caller's ``Lowering`` builder only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from mindie_sim.prediction.runtime.memory_physics import AddressRun

from .catlass_catalog import (
    GATHER_SCATTER_POLICIES,
    packed_bytes,
    streamk_partition,
)
from .contracts import ceil_div

COALESCED_PANEL_THRESHOLD = 256
_SUPPORTED_COALESCED_SCHEDULERS = frozenset({
    "dataparallel", "swizzle", "streamk", "multi_core_splitk",
})


def can_coalesce(lowerer, blocks: int) -> bool:
    """Return True when exact per-core coalesced lowering applies."""
    spec = lowerer.spec
    return (
        blocks * lowerer.gk > COALESCED_PANEL_THRESHOLD
        and spec.scheduler in _SUPPORTED_COALESCED_SCHEDULERS
        and spec.epilogue_kind == "none"
        and spec.dispatch_policy not in GATHER_SCATTER_POLICIES
        and not spec.prologue_b
        and not spec.simt_thread_block_dim
        and spec.layout_a == "RowMajor"
        and spec.layout_b == "RowMajor"
        and spec.layout_c == "RowMajor"
    )


def _merge_rect_runs(tiles, *, row_attr, col_attr, h_attr, w_attr, full_cols, dtype, base_byte=0):
    by_band: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for t in tiles:
        by_band[(getattr(t, row_attr), getattr(t, h_attr))].append((getattr(t, col_attr), getattr(t, w_attr)))
    stride = packed_bytes(full_cols, dtype)
    runs: list[AddressRun] = []
    for (r0, h), cols in sorted(by_band.items()):
        cols.sort()
        c_start, c_end = cols[0][0], cols[0][0] + cols[0][1]
        for c0, w in cols[1:]:
            if c0 <= c_end:
                c_end = max(c_end, c0 + w)
            else:
                runs.append(AddressRun(
                    base_byte + r0 * stride + packed_bytes(c_start, dtype),
                    h, packed_bytes(c_end - c_start, dtype), stride,
                ))
                c_start, c_end = c0, c0 + w
        runs.append(AddressRun(
            base_byte + r0 * stride + packed_bytes(c_start, dtype),
            h, packed_bytes(c_end - c_start, dtype), stride,
        ))
    return tuple(runs)


def _k_micro_issued(lowerer, tile_m: int, tile_n: int, k_total: int, issued_fn) -> int:
    full_panels, rem_k = divmod(k_total, lowerer.tk)
    panel_issued = 0
    for start in range(0, min(lowerer.tk, k_total), lowerer.micro_k):
        micro = min(min(lowerer.tk, k_total) - start, lowerer.micro_k)
        panel_issued += issued_fn(tile_m, tile_n, micro, lowerer.spec, lowerer.plan.datapath)
    total = full_panels * panel_issued
    if rem_k:
        for start in range(0, rem_k, lowerer.micro_k):
            micro = min(rem_k - start, lowerer.micro_k)
            total += issued_fn(tile_m, tile_n, micro, lowerer.spec, lowerer.plan.datapath)
    return total


def lower_coalesced(lowerer, launch: str, blocks: int, *, scale_fn, compute_b_fn, cube_dtype_fn, issued_fn) -> None:
    """Emit exact per-core ping-pong coalesced stages for a large regular GEMM."""
    p, s, plan = lowerer.p, lowerer.spec, lowerer.plan
    by_core: dict[int, list] = defaultdict(list)
    for idx in range(blocks):
        tile = lowerer.tile(idx)
        by_core[tile.owner].append(tile)

    dtype_b_comp = compute_b_fn(s)
    cube_dt = cube_dtype_fn(s)
    a_stride = packed_bytes(p.k, s.dtype_a)
    b_stride = packed_bytes(p.n, s.dtype_b)
    c_stride = packed_bytes(p.n, s.dtype_c)
    pingpong = s.stages[0] >= 2
    first_k = min(lowerer.tk, p.k)
    rest_k = p.k - first_k

    # Precompute streamk / splitk extra workspace traffic if applicable.
    split_partials = 0
    if s.scheduler == "streamk":
        part = streamk_partition(blocks, lowerer.gk, s.aic_budget)
        split_partials = part.partial_count if part.b_sk else 0
    elif s.scheduler == "multi_core_splitk":
        split_partials = blocks * s.splitk_factor

    core_done: list[str] = []
    for core, tiles in sorted(by_core.items()):
        tag = f"coal:c{core}"
        # 1. Exact A reads with cold first-touch + warm reuse on repeated M-bands.
        a_bands: dict[tuple[int, int], int] = defaultdict(int)
        for t in tiles:
            a_bands[(t.row, t.m)] += 1
        a_first_runs = tuple(
            AddressRun(r0 * a_stride, h, a_stride, a_stride) for r0, h in sorted(a_bands)
        )
        a_first_bytes = sum(h * a_stride for _, h in a_bands)
        a_first_scale = sum(scale_fn(s, h * p.k, l1=True) for _, h in a_bands)
        a_first = plan.access(
            f"{tag}:a_first", p.a_id, a_first_bytes, owner=core, pipe="mte2_aic",
            runs=a_first_runs, scale_bytes=a_first_scale, after=(launch,),
        )
        a_deps = [a_first]
        a_reuse_bytes = sum((cnt - 1) * h * a_stride for (_, h), cnt in a_bands.items() if cnt > 1)
        if a_reuse_bytes > 0:
            a_reuse_scale = sum((cnt - 1) * scale_fn(s, h * p.k, l1=True) for (_, h), cnt in a_bands.items() if cnt > 1)
            a_reuse_runs = tuple(
                AddressRun(r0 * a_stride, h, a_stride, a_stride)
                for (r0, h), cnt in sorted(a_bands.items()) if cnt > 1
                for _ in range(cnt - 1)
            )
            a_deps.append(plan.access(
                f"{tag}:a_reuse", p.a_id, a_reuse_bytes, owner=core, pipe="mte2_aic",
                runs=a_reuse_runs, scale_bytes=a_reuse_scale, after=(a_first,),
            ))

        # 2. Exact B reads split into first-panel prologue + steady-state rest for ping-pong overlap.
        b_col_bands: dict[tuple[int, int], int] = defaultdict(int)
        for t in tiles:
            b_col_bands[(t.col, t.n)] += 1
        b_first_runs = tuple(
            AddressRun(p.b_base_byte + packed_bytes(c0, s.dtype_b), first_k, packed_bytes(w, s.dtype_b), b_stride)
            for (c0, w), cnt in sorted(b_col_bands.items()) for _ in range(cnt)
        )
        b_first_bytes = sum(first_k * packed_bytes(w, s.dtype_b) * cnt for (_, w), cnt in b_col_bands.items())
        b_first_scale = sum(scale_fn(s, w * first_k, l1=True) * cnt for (_, w), cnt in b_col_bands.items())
        b_prologue = plan.access(
            f"{tag}:b_prologue", p.b_id, b_first_bytes, owner=core, pipe="mte2_aic",
            runs=b_first_runs, scale_bytes=b_first_scale, after=(launch,), version=p.b_version,
        )
        compute_after = (a_first, b_prologue)
        b_tail = b_prologue
        if rest_k > 0:
            b_rest_runs = tuple(
                AddressRun(
                    p.b_base_byte + first_k * b_stride + packed_bytes(c0, s.dtype_b),
                    rest_k, packed_bytes(w, s.dtype_b), b_stride,
                )
                for (c0, w), cnt in sorted(b_col_bands.items()) for _ in range(cnt)
            )
            b_rest_bytes = sum(rest_k * packed_bytes(w, s.dtype_b) * cnt for (_, w), cnt in b_col_bands.items())
            b_rest_scale = sum(scale_fn(s, w * rest_k, l1=True) * cnt for (_, w), cnt in b_col_bands.items())
            b_tail = plan.access(
                f"{tag}:b_rest", p.b_id, b_rest_bytes, owner=core, pipe="mte2_aic",
                runs=b_rest_runs, scale_bytes=b_rest_scale, after=(b_prologue,), version=p.b_version,
            )
            if not pingpong:
                compute_after = (*a_deps, b_tail)

        # 3. Exact MTE1 and Cube work on this core (overlapped with b_rest when ping-pong).
        mte1_bytes = sum(
            packed_bytes(t.m * p.k, s.dtype_a) + scale_fn(s, t.m * p.k)
            + packed_bytes(t.n * p.k, dtype_b_comp) + scale_fn(s, t.n * p.k)
            for t in tiles
        )
        mte1_stage = plan.work(f"{tag}:mte1", "mte1", mte1_bytes, owner=core, after=compute_after)
        logical_flops = sum(2 * t.m * t.n * p.k for t in tiles)
        issued_flops = sum(_k_micro_issued(lowerer, t.m, t.n, p.k, issued_fn) for t in tiles)
        mma_stage = plan.work(
            f"{tag}:mma", "cube", issued_flops, owner=core, dtype=cube_dt,
            after=compute_after if pingpong else (mte1_stage,),
            metadata={"logical_flops": logical_flops},
        )
        if s.has_bias:
            bias_bytes = sum(packed_bytes(t.n, s.dtype_acc) for t in tiles)
            plan.access(
                f"{tag}:bias", p.c_id + ":bias", bias_bytes, owner=core, pipe="mte2_aic",
                offset=packed_bytes(tiles[0].col, s.dtype_acc), after=(launch,),
            )

        # 4. Exact output write (Fixpipe).
        c_runs = _merge_rect_runs(
            tiles, row_attr="row", col_attr="col", h_attr="m", w_attr="n",
            full_cols=p.n, dtype=s.dtype_c,
        )
        c_bytes = sum(packed_bytes(t.m * t.n, s.dtype_c) for t in tiles)
        out_after = tuple(dict.fromkeys((*a_deps, b_tail, mte1_stage, mma_stage)))
        core_done.append(plan.access(
            f"{tag}:fixpipe", p.c_id, c_bytes, owner=core, pipe="fixpipe",
            direction="write", runs=c_runs, after=out_after,
        ))

    if split_partials > 0 and s.aiv_budget > 0:
        partial_bytes = packed_bytes(lowerer.tm * lowerer.tn, s.dtype_acc)
        per_aiv = ceil_div(split_partials, s.aiv_budget)
        signal = plan.add(replace(
            plan.stages[0], name="coal:cross_core_set_flag", work={}, after=tuple(core_done),
            placement={}, metadata={"streamk_partial_count": split_partials},
        ))
        for lane in range(min(s.aiv_budget, split_partials)):
            count = min(per_aiv, split_partials - lane * per_aiv)
            if count <= 0:
                continue
            amt = count * partial_bytes
            rd = plan.access(
                f"coal:reduce:{lane}:read", p.c_id + ":splitk", amt, owner=lane,
                pipe="mte2_aiv", private=True, after=(signal,), working_set=s.workspace_bytes,
            )
            plan.work(
                f"coal:reduce:{lane}:add", "vec_reduce", count * lowerer.tm * lowerer.tn,
                owner=lane, after=(rd,),
            )
