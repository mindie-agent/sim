"""Finite RMSNorm + GEMM row/weight/output pipeline.

[Role]
  Lower a single-consumer RMSNorm feeding one GEMM into complete-row norm
  tasks plus finite N/K GEMM and output stages.  A row slot holds BF16 input
  normalized in place; if even one full row cannot fit, an explicit two-pass
  RMS reduction and reread plan replaces whole-tensor rejection.

[Invariants]
  * RMS reduction spans the full K row; normalized values are BF16 at the
    unfused boundary and GEMM consumes those BF16 values.
  * A row slot is reused only after every N tile consumed it.  Weight slots are
    released at their GEMM consumer; output slots after their write.  These are
    task dependencies, not report-only notes.
  * Useful FLOPs and the original norm vector reduction are retained.  Output
    casting, repeated reread/normalization and finite-slot weight reloads are
    charged; no large-L2 cache is granted for free.
  * Full-row admission charges complete row slots, finite weight/output slots,
    one FP32 accumulator and RMS state.  The two-pass plan charges extra reads.

[Timeouts & Locks]
  Bounded finite row/N/K/tile construction.

[Side-Effect Policy]
  No I/O, device, global mutation, or scheduler engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .a3_fusion_common import (
    MemoryBudget, PipelinePlan, add_release_join, ceil_div, pressure,
    tile_sizes,
)
from .a3_types import dtype_bytes

_EPS = 1e-12


@dataclass(frozen=True)
class NormTiling:
    mode: str
    row_tile: int
    n_tile: int
    k_tile: int
    row_slots: int
    weight_slots: int
    output_slots: int
    group_tiles: int
    row_tiles: int
    n_tiles: int
    k_tiles: int
    n_groups: int
    required_by_pool: Mapping[str, float]
    stage_estimate: int


@dataclass(frozen=True)
class NormRequest:
    m: int
    n: int
    k: int
    dtype: str
    norm: Any
    gemm: Any


def _request(norm: Any, gemm: Any) -> NormRequest | None:
    shape = tuple(int(value) for value in getattr(gemm, "launch_shape", ()))
    if len(shape) < 3:
        return None
    m, n, k = max(1, shape[-3]), max(1, shape[-2]), max(1, shape[-1])
    return NormRequest(
        m=m, n=n, k=k, dtype=str(getattr(gemm, "dtype", "") or "bf16"),
        norm=norm, gemm=gemm,
    )


def _empty() -> Mapping[str, float]:
    return {}


def _full_pressure(r: NormRequest, row: int, nt: int, kt: int,
                   row_slots: int, out_slots: int,
                   weight_slots: int = 2, physical: bool = False) -> dict[str, float]:
    b = float(dtype_bytes(r.dtype))
    l1 = row_slots * row * r.k * b + weight_slots * kt * nt * b
    ub = out_slots * row * nt * b + row_slots * row * 4.0
    l0c = row * nt * 4.0
    pools = {"l1": l1, "ub": ub, "l0c": l0c}
    if physical:
        # UB input/normalized row, then explicit UB->L1 handoff; bounded MMAD
        # operands with two concurrent inner tiles and one L0C accumulator.
        pools["ub"] = ub + row_slots * row * r.k * b + out_slots * row * nt * 4.0
        # One MMAD inner tile is active per fixed worker; L0A/L0B operands are
        # refilled per K step.  Two concurrent operand buffers would halve the
        # legal B tile without a second physical MMAD.
        pools["l0a"] = row * kt * b
        pools["l0b"] = kt * nt * b
    return pressure(pools, 1)


def _two_pass_pressure(r: NormRequest, row: int, nt: int, kt: int,
                       group_tiles: int, chunk_slots: int = 2,
                       out_slots: int = 2,
                       weight_slots: int = 2,
                       physical: bool = False) -> dict[str, float]:
    b = float(dtype_bytes(r.dtype))
    l1 = weight_slots * kt * nt * b
    ub = (
        chunk_slots * row * kt * b
        + out_slots * row * nt * b
        + 2 * row * 4.0
    )
    l0c = group_tiles * row * nt * 4.0
    pools = {"l1": l1, "ub": ub, "l0c": l0c}
    if physical:
        pools["l1"] = l1 + chunk_slots * row * kt * b
        pools["ub"] = ub + out_slots * row * nt * 4.0
        # One MMAD inner tile is active per fixed worker; L0A/L0B operands are
        # refilled per K step.  Two concurrent operand buffers would halve the
        # legal B tile without a second physical MMAD.
        pools["l0a"] = row * kt * b
        pools["l0b"] = kt * nt * b
    return pressure(pools, 1)


def _full_stage_estimate(r: NormRequest, row: int, nt: int, kt: int) -> int:
    rows = ceil_div(r.m, row)
    n_tiles = ceil_div(r.n, nt)
    k_tiles = ceil_div(r.k, kt)
    return rows * (3 + n_tiles * k_tiles * 2 + n_tiles * 2)


def _two_pass_stage_estimate(
    r: NormRequest, row: int, nt: int, kt: int, group_tiles: int,
) -> int:
    rows = ceil_div(r.m, row)
    n_tiles = ceil_div(r.n, nt)
    k_tiles = ceil_div(r.k, kt)
    groups = ceil_div(n_tiles, group_tiles)
    return (
        rows * k_tiles * 2
        + rows * groups * k_tiles * (2 + group_tiles)
        + rows * n_tiles * 2
    )


def _choose_tiling(r: NormRequest, budget: MemoryBudget, *,
                   physical: bool = False) -> tuple[NormTiling | None, dict[str, Any]]:
    row_candidates = tile_sizes(min(16, r.m), (16, 8, 4, 2))
    if r.m > 16:
        row_candidates = [16, 8, 4, 2, 1]
    n_candidates = tile_sizes(r.n, (512, 256, 128, 64, 32, 16, 8, 4, 2))
    k_candidates = tile_sizes(r.k, (512, 256, 128, 64, 32, 16, 8, 4, 2))
    minimal: dict[str, Any] | None = None

    # Full complete-row plans first: reduction and BF16 cast stay in one row.
    best_full: tuple[float, NormTiling] | None = None
    for row in row_candidates:
        for nt in n_candidates:
            for kt in k_candidates:
                for row_slots in (2, 1):
                    for out_slots in (2, 1):
                        for weight_slots in (2, 1):
                            required = _full_pressure(
                                r, row, nt, kt, row_slots, out_slots,
                                weight_slots=weight_slots, physical=physical,
                            )
                            estimate = _full_stage_estimate(r, row, nt, kt)
                            if minimal is None or sum(required.values()) < sum(
                                minimal.get("required_by_pool", {}).values()
                            ):
                                minimal = {
                                    "mode": "full_row",
                                    "required_by_pool": required,
                                    "row_tile": row, "n_tile": nt, "k_tile": kt,
                                    "row_slots": row_slots,
                                    "weight_slots": weight_slots,
                                    "output_slots": out_slots,
                                }
                            if not budget.details(required)["fits"]:
                                continue
                            # Prefer real double buffering: one complete-row
                            # slot or one weight slot removes the pipeline
                            # overlap that the finite plan exists to expose.
                            penalty = 1.0
                            if row_slots <= 1:
                                penalty *= 4.0
                            if weight_slots <= 1:
                                penalty *= 3.0
                            if out_slots <= 1:
                                penalty *= 1.5
                            score = estimate * penalty
                            tiling = NormTiling(
                                mode="full_row", row_tile=row, n_tile=nt,
                                k_tile=kt, row_slots=row_slots,
                                weight_slots=weight_slots,
                                output_slots=out_slots, group_tiles=0,
                                row_tiles=ceil_div(r.m, row),
                                n_tiles=ceil_div(r.n, nt),
                                k_tiles=ceil_div(r.k, kt),
                                n_groups=1, required_by_pool=required,
                                stage_estimate=estimate,
                            )
                            if best_full is None or score < best_full[0]:
                                best_full = (score, tiling)
    if best_full is not None:
        return best_full[1], {}

    # No complete row fits: explicit two-pass reduction with counted rereads.
    best_two: tuple[float, NormTiling] | None = None
    for row in row_candidates:
        for nt in n_candidates:
            for kt in k_candidates:
                n_tiles = ceil_div(r.n, nt)
                group_candidates = sorted({
                    n_tiles, min(8, n_tiles), min(4, n_tiles),
                    min(2, n_tiles), 1,
                }, reverse=True)
                for group_tiles in group_candidates:
                    for weight_slots in (2, 1):
                        required = _two_pass_pressure(
                            r, row, nt, kt, group_tiles,
                            weight_slots=weight_slots, physical=physical,
                        )
                        estimate = _two_pass_stage_estimate(
                            r, row, nt, kt, group_tiles,
                        )
                        if minimal is None or sum(required.values()) < sum(
                            minimal.get("required_by_pool", {}).values()
                        ):
                            minimal = {
                                "mode": "two_pass",
                                "required_by_pool": required,
                                "row_tile": row, "n_tile": nt, "k_tile": kt,
                                "group_tiles": group_tiles,
                                "weight_slots": weight_slots,
                            }
                        if not budget.details(required)["fits"]:
                            continue
                        tiling = NormTiling(
                            mode="two_pass", row_tile=row, n_tile=nt,
                            k_tile=kt, row_slots=2,
                            weight_slots=weight_slots, output_slots=2,
                            group_tiles=group_tiles,
                            row_tiles=ceil_div(r.m, row),
                            n_tiles=n_tiles,
                            k_tiles=ceil_div(r.k, kt),
                            n_groups=ceil_div(n_tiles, group_tiles),
                            required_by_pool=required,
                            stage_estimate=estimate,
                        )
                        if best_two is None or estimate < best_two[0]:
                            best_two = (estimate, tiling)
    if best_two is not None:
        return best_two[1], {}
    return None, {
        "reason": "live_buffer_capacity_exceeded",
        "minimal_required_by_pool": dict(
            (minimal or {}).get("required_by_pool", {})
        ),
        "minimal_tiling": minimal,
        "budget": budget.details((minimal or {}).get("required_by_pool", {})),
    }

def _norm_split(norm_ops: float, fraction: float) -> float:
    return float(max(0.0, norm_ops)) * float(fraction)


def plan_norm_gemm_pipeline(
    norm: Any, gemm: Any, *, budget: MemoryBudget,
    physical: bool = False,
) -> tuple[PipelinePlan | None, dict[str, Any]]:
    """Build a finite row-tiled RMSNorm+GEMM plan or exact rejection."""
    r = _request(norm, gemm)
    if r is None:
        return None, {"reason": "gemm_extent_missing"}
    tiling, rejection = _choose_tiling(r, budget, physical=physical)
    if tiling is None:
        return None, rejection
    if tiling.row_tiles == 1 and tiling.n_tiles == 1 and tiling.k_tiles == 1 \
            and not physical:
        # Preserve the legacy two-stage logical fixture only when no physical
        # L0 path is requested.  Physical single tiles must execute the same
        # UB/L1/L0/cast path as every larger tile.
        return _plan_single_tile(r, tiling)
    if tiling.mode == "full_row":
        return _plan_full_row(r, tiling, physical=physical)
    return _plan_two_pass(r, tiling, physical=physical)


def _stage_common(plan: PipelinePlan, *, launch: bool, name: str, family: str,
                  deps: Sequence[int], kind: str, tile_shape: Sequence[int],
                  extents: Mapping[str, Any], **work: Any) -> int:
    return plan.add_stage(
        name, family, deps=deps, launch=launch, kind=kind,
        tile_shape=tile_shape, extents=extents, **work,
    )


def _plan_single_tile(
    r: NormRequest, tiling: NormTiling,
) -> tuple[PipelinePlan, dict[str, Any]]:
    """One row/N/K tile remains two real dependent tasks (legacy names kept)."""
    b = float(dtype_bytes(r.dtype))
    norm_ops = float(getattr(r.norm, "vector_ops", 0.0) or 0.0)
    gemm_flops = float(getattr(r.gemm, "flops", 0.0) or 0.0)
    x_mte2 = float(max(0.0, getattr(r.norm, "mte2_bytes", 0.0) or 0.0))
    x_hbm = max(0.0, float(getattr(r.norm, "hbm_bytes", 0.0) or 0.0)
                - float(getattr(r.norm, "mte3_bytes", 0.0) or 0.0))
    if x_hbm <= 0.0:
        x_hbm = x_mte2
    weight_bytes = float(r.k * r.n * b)
    output_bytes = float(r.m * r.n * b)
    plan = PipelinePlan(kind="norm_gemm_single_tile")
    row = plan.add_buffer(
        "norm.row.slot0", size_bytes=r.m * r.k * b, pool="l1",
        owner="single", shape=(r.m, r.k), dtype=r.dtype, slot=0,
        category="normalized_row",
    )
    state = plan.add_buffer(
        "norm.state.slot0", size_bytes=r.m * 4.0, pool="ub",
        owner="single", shape=(r.m,), dtype="fp32", slot=0,
        category="rms_state",
    )
    weight = plan.add_buffer(
        "gemm.weight.slot0", size_bytes=weight_bytes, pool="l1",
        owner="single", shape=(r.k, r.n), dtype=r.dtype, slot=0,
        category="weight_tile",
    )
    accum = plan.add_buffer(
        "gemm.accum.slot0", size_bytes=r.m * r.n * 4.0, pool="l0c",
        owner="single", shape=(r.m, r.n), dtype="fp32", slot=0,
        category="output_accumulator",
    )
    out = plan.add_buffer(
        "gemm.output.slot0", size_bytes=output_bytes, pool="ub",
        owner="single", shape=(r.m, r.n), dtype=r.dtype, slot=0,
        category="output_tile",
    )
    norm_index = _stage_common(
        plan, launch=True, name="norm", family="rmsnorm", deps=(),
        kind="norm_full_row",
        tile_shape=(r.m, r.k),
        extents={"m": [0, r.m], "k": [0, r.k], "full_k": True},
        vector_ops=norm_ops, mte2_bytes=x_mte2, hbm_bytes=x_hbm,
        creates=(row, state), reads=(), writes=(row, state),
        releases=(state,),
    )
    plan.add_stage(
        "gemm", str(getattr(r.gemm, "family", "gemm")), deps=(norm_index,),
        launch=False, kind="gemm_full_tile",
        tile_shape=(r.m, r.n, r.k),
        extents={"m": [0, r.m], "n": [0, r.n], "k": [0, r.k]},
        flops=gemm_flops,
        vector_ops=float(r.m * r.n),
        mte2_bytes=weight_bytes, mte3_bytes=output_bytes,
        hbm_bytes=weight_bytes + output_bytes,
        reads=(row, weight, accum), writes=(out,),
        creates=(weight, accum, out),
        releases=(row, weight, accum, out),
    )
    plan.finish(required_by_pool=tiling.required_by_pool)
    plan.notes.update({
        "tiling": {"mode": "single_tile", "row": r.m, "n": r.n, "k": r.k},
        "bf16_boundary": "row normalized in place and consumed as BF16",
        "weight_reloads": 1,
    })
    return plan, {}


def _plan_full_row(
    r: NormRequest, tiling: NormTiling, *, physical: bool = False,
) -> tuple[PipelinePlan, dict[str, Any]]:
    b = float(dtype_bytes(r.dtype))
    norm_ops = float(getattr(r.norm, "vector_ops", 0.0) or 0.0)
    gemm_flops = float(getattr(r.gemm, "flops", 0.0) or 0.0)
    gemm_family = str(getattr(r.gemm, "family", "gemm"))
    x_mte2 = float(max(0.0, getattr(r.norm, "mte2_bytes", 0.0) or 0.0))
    x_hbm = max(0.0, float(getattr(r.norm, "hbm_bytes", 0.0) or 0.0)
                - float(getattr(r.norm, "mte3_bytes", 0.0) or 0.0))
    if x_hbm <= 0.0:
        x_hbm = x_mte2
    plan = PipelinePlan(kind="norm_gemm_full_row")
    row_slot_free = [None] * max(1, tiling.row_slots)
    weight_slot_free = [None] * max(1, tiling.weight_slots)
    output_slot_free = [None] * max(1, tiling.output_slots)
    accumulator_free: int | None = None
    weight_use = 0
    output_use = 0
    total_m = float(r.m)
    # Explicit finite inner-K load/MMAD/release loops are emitted.  The builder
    # is indexed O(1), so no whole-K preload or compact metadata-only loop is
    # used; a load stage never serves more bytes than its created buffer.
    compact_gemm = False
    previous_norm_scale: int | None = None
    for row_index in range(tiling.row_tiles):
        row_off = row_index * tiling.row_tile
        row_size = min(tiling.row_tile, r.m - row_off)
        row_slot = row_index % max(1, tiling.row_slots)
        owner = f"r{row_off}"
        row_buffer = plan.add_buffer(
            f"norm.row.{owner}", size_bytes=row_size * r.k * b, pool="l1",
            owner=owner, shape=(row_size, r.k), dtype=r.dtype, slot=row_slot,
            category="normalized_row",
        )
        state_buffer = plan.add_buffer(
            f"norm.state.{owner}", size_bytes=row_size * 4.0, pool="ub",
            owner=owner, shape=(row_size,), dtype="fp32", slot=row_slot,
            category="rms_state",
        )
        ub_row = plan.add_buffer(
            f"norm.ubx.{owner}", size_bytes=row_size * r.k * b, pool="ub",
            owner=owner, shape=(row_size, r.k), dtype=r.dtype, slot=row_slot,
            category="norm_input_ub",
        ) if physical else None
        norm_deps = () if row_slot_free[row_slot] is None else (row_slot_free[row_slot],)
        if previous_norm_scale is not None:
            norm_deps = tuple(dict.fromkeys((*norm_deps, previous_norm_scale)))
        load_stage = _stage_common(
            plan, launch=(row_index == 0), name=f"norm.load.{owner}",
            family="rmsnorm", deps=norm_deps, kind="norm_load_full_row",
            tile_shape=(row_size, r.k),
            extents={"m": [row_off, row_size], "k": [0, r.k], "full_k": True},
            mte2_bytes=x_mte2 * row_size / total_m,
            hbm_bytes=x_hbm * row_size / total_m,
            creates=(ub_row if physical else row_buffer, state_buffer),
            writes=(ub_row if physical else row_buffer, state_buffer),
        )
        reduce_stage = _stage_common(
            plan, launch=False, name=f"norm.reduce.{owner}", family="rmsnorm",
            deps=(load_stage,), kind="rms_reduce_full_k",
            tile_shape=(row_size, r.k),
            extents={"m": [row_off, row_size], "k": [0, r.k], "full_k": True},
            vector_ops=_norm_split(norm_ops * row_size / total_m, 0.4),
            reads=(ub_row if physical else row_buffer,),
            writes=(state_buffer,),
        )
        scale_stage = _stage_common(
            plan, launch=False, name=f"norm.scale.{owner}", family="rmsnorm",
            deps=(reduce_stage,), kind="rms_scale_cast_bf16",
            tile_shape=(row_size, r.k),
            extents={"m": [row_off, row_size], "k": [0, r.k], "full_k": True},
            vector_ops=_norm_split(norm_ops * row_size / total_m, 0.6),
            reads=(ub_row if physical else row_buffer, state_buffer),
            writes=(ub_row if physical else row_buffer,),
            releases=(state_buffer,),
        )
        if physical:
            handoff_stage = _stage_common(
                plan, launch=False, name=f"norm.handoff.{owner}",
                family="rmsnorm", deps=(scale_stage,),
                kind="ub_to_l1_handoff",
                tile_shape=(row_size, r.k),
                extents={"m": [row_off, row_size], "k": [0, r.k],
                         "full_k": True, "src_pool": "ub", "dst_pool": "l1",
                         "src_dtype": r.dtype, "dst_dtype": r.dtype},
                vector_ops=float(row_size * r.k),
                reads=(ub_row,), writes=(row_buffer,), creates=(row_buffer,),
                releases=(ub_row,),
                resources={"mte1": float(row_size * r.k * b)},
            )
        else:
            handoff_stage = scale_stage
        previous_norm_scale = handoff_stage
        last_gemm: int | None = None
        row_gemms: list[int] = []
        for n_index in range(tiling.n_tiles):
            n_off = n_index * tiling.n_tile
            n_size = min(tiling.n_tile, r.n - n_off)
            accum_buffer = plan.add_buffer(
                f"gemm.accum.{owner}.n{n_off}",
                size_bytes=row_size * n_size * 4.0, pool="l0c", owner=owner,
                shape=(row_size, n_size), dtype="fp32", slot=0,
                category="output_accumulator",
            )
            prev_k: int | None = None
            compact_weight_bytes = float(
                sum(
                    min(tiling.k_tile, r.k - kk)
                    for kk in range(0, r.k, tiling.k_tile)
                ) * n_size * b
            ) if compact_gemm else 0.0
            for k_index in ((0,) if compact_gemm else range(tiling.k_tiles)):
                if compact_gemm:
                    k_off = 0
                    k_extent = [0, r.k]
                    inner_k = tiling.k_tile
                    work_k = r.k
                    weight_bytes = compact_weight_bytes
                    flops = 2.0 * row_size * n_size * r.k
                else:
                    k_off = k_index * tiling.k_tile
                    k_size = min(tiling.k_tile, r.k - k_off)
                    k_extent = [k_off, k_size]
                    inner_k = k_size
                    work_k = k_size
                    weight_bytes = float(k_size * n_size * b)
                    flops = 2.0 * row_size * n_size * k_size
                ws = weight_use % max(1, tiling.weight_slots)
                weight_use += 1
                weight_buffer = plan.add_buffer(
                    f"gemm.weight.{owner}.n{n_off}.k{k_off}",
                    size_bytes=inner_k * n_size * b, pool="l1", owner=owner,
                    shape=(inner_k, n_size), dtype=r.dtype, slot=ws,
                    category="weight_tile",
                )
                weight_deps = (
                    () if weight_slot_free[ws] is None else (weight_slot_free[ws],)
                )
                weight_load = _stage_common(
                    plan, launch=False,
                    name=f"weight.load.{owner}.n{n_off}.k{k_off}",
                    family=gemm_family, deps=weight_deps, kind="weight_load",
                    tile_shape=(row_size, inner_k, n_size),
                    extents={
                        "m": [row_off, row_size], "n": [n_off, n_size],
                        "k": k_extent, "inner_k_tile": inner_k,
                        "k_tiles": tiling.k_tiles,
                        "k_tail": r.k % tiling.k_tile or tiling.k_tile,
                        "compact_inner_loop": (
                            True if compact_gemm else "explicit_single_k_step"
                        ),
                        "explicit_k_steps": 1,
                        "whole_k_tiles": tiling.k_tiles,
                    },
                    mte2_bytes=weight_bytes, hbm_bytes=weight_bytes,
                    creates=(weight_buffer,), writes=(weight_buffer,),
                )
                if k_index == 0:
                    base_deps = (
                        () if accumulator_free is None else (accumulator_free,)
                    )
                    creates = (accum_buffer,)
                else:
                    base_deps = (prev_k,) if prev_k is not None else ()
                    creates = ()
                gemm_deps = tuple(dict.fromkeys(
                    (*base_deps, weight_load, handoff_stage)
                ))
                if physical:
                    l0a = plan.add_buffer(
                        f"gemm.l0a.{owner}.n{n_off}.k{k_off}",
                        size_bytes=row_size * inner_k * b, pool="l0a",
                        owner=owner, shape=(row_size, inner_k), dtype=r.dtype,
                        slot=0, category="mmad_a",
                    )
                    l0b = plan.add_buffer(
                        f"gemm.l0b.{owner}.n{n_off}.k{k_off}",
                        size_bytes=inner_k * n_size * b, pool="l0b",
                        owner=owner, shape=(inner_k, n_size), dtype=r.dtype,
                        slot=0, category="mmad_b",
                    )
                    mmad_creates = (*creates, l0a, l0b)
                    mmad_reads = (row_buffer, weight_buffer, l0a, l0b, accum_buffer)
                    mmad_releases = (weight_buffer, l0a, l0b)
                    mmad_resources = {
                        "mte1": float(
                            (tiling.k_tiles if compact_gemm else 1)
                            * (row_size * inner_k * b + inner_k * n_size * b)
                        ),
                    }
                else:
                    mmad_creates = creates
                    mmad_reads = (row_buffer, weight_buffer, accum_buffer)
                    mmad_releases = (weight_buffer,)
                    mmad_resources = {}
                gemm_stage = _stage_common(
                    plan, launch=False,
                    name=f"gemm.{owner}.n{n_off}.k{k_off}", family=gemm_family,
                    deps=gemm_deps, kind="gemm_k_accumulate",
                    tile_shape=(row_size, n_size, inner_k),
                    extents={
                        "m": [row_off, row_size], "n": [n_off, n_size],
                        "k": k_extent, "inner_k_tile": inner_k,
                        "k_tiles": tiling.k_tiles,
                        "k_tail": r.k % tiling.k_tile or tiling.k_tile,
                        "compact_inner_loop": (
                            True if compact_gemm else "explicit_single_k_step"
                        ),
                        "explicit_k_steps": 1,
                        "whole_k_tiles": tiling.k_tiles,
                        "inner_accumulation": "sequential_l0c",
                        "a_pool": "l1", "b_pool": "l1",
                        "l0a_pool": "l0a", "l0b_pool": "l0b",
                        "l0c_pool": "l0c", "src_dtype": r.dtype,
                        "l0a_bytes": row_size * inner_k * b if physical else 0.0,
                        "l0b_bytes": inner_k * n_size * b if physical else 0.0,
                    },
                    flops=flops,
                    reads=mmad_reads, writes=(accum_buffer,),
                    creates=mmad_creates, releases=mmad_releases,
                    resources=mmad_resources,
                )
                weight_slot_free[ws] = gemm_stage
                prev_k = gemm_stage
                last_gemm = gemm_stage
                row_gemms.append(gemm_stage)
            assert prev_k is not None
            output_slot = output_use % max(1, tiling.output_slots)
            output_use += 1
            out_bytes = float(row_size * n_size * b)
            output_buffer = plan.add_buffer(
                f"gemm.output.{owner}.n{n_off}", size_bytes=out_bytes, pool="ub",
                owner=owner, shape=(row_size, n_size), dtype=r.dtype,
                slot=output_slot, category="output_tile",
            )
            if physical:
                fp32_buffer = plan.add_buffer(
                    f"gemm.output_fp32.{owner}.n{n_off}",
                    size_bytes=row_size * n_size * 4.0, pool="ub",
                    owner=owner, shape=(row_size, n_size), dtype="fp32",
                    slot=output_slot, category="output_l0c_copy",
                )
                copy_deps = tuple(dict.fromkeys(
                    (prev_k, output_slot_free[output_slot])
                    if output_slot_free[output_slot] is not None else (prev_k,)
                ))
                copy_stage = _stage_common(
                    plan, launch=False,
                    name=f"output.copy.{owner}.n{n_off}", family=gemm_family,
                    deps=copy_deps, kind="l0c_to_ub_output",
                    tile_shape=(row_size, n_size),
                    extents={"m": [row_off, row_size], "n": [n_off, n_size],
                             "src_pool": "l0c", "dst_pool": "ub",
                             "src_dtype": "fp32", "dst_dtype": "fp32"},
                    vector_ops=float(row_size * n_size),
                    reads=(accum_buffer,), writes=(fp32_buffer,),
                    creates=(fp32_buffer,), releases=(accum_buffer,),
                    resources={"mte1": float(row_size * n_size * 4.0)},
                )
                cast_deps = (copy_stage,)
                cast_reads = (fp32_buffer,)
                cast_releases = (fp32_buffer,)
            else:
                fp32_buffer = None
                copy_stage = prev_k
                cast_deps = tuple(dict.fromkeys(
                    (prev_k, output_slot_free[output_slot])
                    if output_slot_free[output_slot] is not None else (prev_k,)
                ))
                cast_reads = (accum_buffer,)
                cast_releases = (accum_buffer,)
            cast_stage = _stage_common(
                plan, launch=False, name=f"output.cast.{owner}.n{n_off}",
                family=gemm_family, deps=cast_deps,
                kind=f"output_cast_{r.dtype}",
                tile_shape=(row_size, n_size),
                extents={
                    "m": [row_off, row_size], "n": [n_off, n_size],
                    "k": [0, r.k], "full_k": True,
                    "src_pool": "ub", "dst_pool": "ub",
                    "src_dtype": "fp32", "dst_dtype": r.dtype,
                },
                vector_ops=float(row_size * n_size),
                reads=cast_reads, writes=(output_buffer,),
                creates=(output_buffer,), releases=cast_releases,
            )
            output_write = _stage_common(
                plan, launch=False, name=f"output.write.{owner}.n{n_off}",
                family=gemm_family, deps=(cast_stage,), kind="output_write",
                tile_shape=(row_size, n_size),
                extents={"m": [row_off, row_size], "n": [n_off, n_size],
                         "dtype": r.dtype},
                mte3_bytes=out_bytes, hbm_bytes=out_bytes,
                reads=(output_buffer,), releases=(output_buffer,),
            )
            output_slot_free[output_slot] = output_write
            accumulator_free = copy_stage if physical else cast_stage
        if last_gemm is not None:
            row_release = add_release_join(
                plan, name=f"release.row.{owner}", family="rmsnorm",
                consumers=row_gemms, releases=(row_buffer,),
                kind="row_slot_release_join",
                extents={"m": [row_off, row_size], "k": [0, r.k],
                         "full_k": True},
            )
            row_slot_free[row_slot] = row_release
    plan.finish(required_by_pool=tiling.required_by_pool)
    weight_bytes_total = float(
        sum(
            min(tiling.k_tile, r.k - kk) * min(tiling.n_tile, r.n - nn) * b
            for _ in range(tiling.row_tiles)
            for nn in range(0, r.n, tiling.n_tile)
            for kk in range(0, r.k, tiling.k_tile)
        )
    )
    plan.notes.update({
        "tiling": {
            "mode": "full_row", "row_tile": tiling.row_tile,
            "n_tile": tiling.n_tile, "k_tile": tiling.k_tile,
            "row_slots": tiling.row_slots, "weight_slots": tiling.weight_slots,
            "output_slots": tiling.output_slots,
            "row_tiles": tiling.row_tiles, "n_tiles": tiling.n_tiles,
            "k_tiles": tiling.k_tiles,
            "row_tail": r.m % tiling.row_tile or tiling.row_tile,
            "n_tail": r.n % tiling.n_tile or tiling.n_tile,
            "k_tail": r.k % tiling.k_tile or tiling.k_tile,
        },
        "bf16_boundary": "row normalized in place and consumed as BF16",
        "weight_bytes_reloaded": weight_bytes_total,
        "weight_reloads": tiling.row_tiles,
        "two_pass": False,
        "extra_x_read_bytes": 0.0,
        "physical_path": bool(physical),
        "physical_primitives": (
            "mte2->ub, rms_reduce(ub), rms_scale(ub), ub->l1, "
            f"l1->l0a/l0b, mmad->l0c, fp32 l0c->ub, output cast->{r.dtype} ub"
        ) if physical else "logical_unified_no_physical_l0_path",
    })
    return plan, {}


def _plan_two_pass(
    r: NormRequest, tiling: NormTiling, *, physical: bool = False,
) -> tuple[PipelinePlan, dict[str, Any]]:
    b = float(dtype_bytes(r.dtype))
    norm_ops = float(getattr(r.norm, "vector_ops", 0.0) or 0.0)
    gemm_flops = float(getattr(r.gemm, "flops", 0.0) or 0.0)
    gemm_family = str(getattr(r.gemm, "family", "gemm"))
    x_mte2 = float(max(0.0, getattr(r.norm, "mte2_bytes", 0.0) or 0.0))
    x_hbm = max(0.0, float(getattr(r.norm, "hbm_bytes", 0.0) or 0.0)
                - float(getattr(r.norm, "mte3_bytes", 0.0) or 0.0))
    if x_hbm <= 0.0:
        x_hbm = x_mte2
    plan = PipelinePlan(kind="norm_gemm_two_pass")
    chunk_slot_free = [None, None]
    weight_slot_free = [None] * max(1, tiling.weight_slots)
    output_slot_free = [None, None]
    state_slot_free = [None, None]
    accumulator_free: int | None = None
    weight_use = 0
    output_use = 0
    total_m = float(r.m)
    x_group_reads = max(1, tiling.n_groups)
    state_consumers: list[int] = []
    for row_index in range(tiling.row_tiles):
        row_off = row_index * tiling.row_tile
        row_size = min(tiling.row_tile, r.m - row_off)
        state_slot = row_index % 2
        owner = f"r{row_off}"
        state_buffer = plan.add_buffer(
            f"norm.state.{owner}", size_bytes=row_size * 4.0, pool="ub",
            owner=owner, shape=(row_size,), dtype="fp32", slot=state_slot,
            category="rms_state",
        )
        reduce_deps = (
            () if state_slot_free[state_slot] is None
            else (state_slot_free[state_slot],)
        )
        reduce_loads: list[int] = []
        reduce_stages: list[int] = []
        for k_index in range(tiling.k_tiles):
            k_off = k_index * tiling.k_tile
            k_size = min(tiling.k_tile, r.k - k_off)
            xs = k_index % len(chunk_slot_free)
            chunk_bytes = row_size * k_size * b
            chunk = plan.add_buffer(
                f"norm.pass1.x.{owner}.k{k_off}", size_bytes=chunk_bytes,
                pool="ub", owner=owner, shape=(row_size, k_size),
                dtype=r.dtype, slot=xs, category="x_chunk",
            )
            deps = (
                () if chunk_slot_free[xs] is None else (chunk_slot_free[xs],)
            )
            if k_index == 0:
                deps = tuple(dict.fromkeys((*deps, *reduce_deps)))
            load_stage = _stage_common(
                plan, launch=(row_index == 0 and k_index == 0),
                name=f"norm.reduce.load.{owner}.k{k_off}", family="rmsnorm",
                deps=deps, kind="rms_first_pass_load",
                tile_shape=(row_size, k_size),
                extents={"m": [row_off, row_size], "k": [k_off, k_size]},
                mte2_bytes=x_mte2 * row_size * k_size / max(_EPS, total_m * r.k),
                hbm_bytes=x_hbm * row_size * k_size / max(_EPS, total_m * r.k),
                creates=(chunk,),
                writes=(chunk,),
            )
            reduce_stage = _stage_common(
                plan, launch=False, name=f"norm.reduce.{owner}.k{k_off}",
                family="rmsnorm",
                deps=(load_stage,) + ((reduce_stages[-1],) if reduce_stages else ()),
                kind="rms_reduce_partial",
                tile_shape=(row_size, k_size),
                extents={"m": [row_off, row_size], "k": [k_off, k_size]},
                vector_ops=_norm_split(
                    norm_ops * row_size / total_m, 0.4 / tiling.k_tiles
                ),
                reads=(chunk,) + ((state_buffer,) if reduce_stages else ()),
                writes=(state_buffer,),
                creates=(state_buffer,) if not reduce_stages else (),
                releases=(chunk,),
            )
            chunk_slot_free[xs] = reduce_stage
            reduce_loads.append(load_stage)
            reduce_stages.append(reduce_stage)
        last_reduce = reduce_stages[-1]
        for group_index in range(tiling.n_groups):
            n_start = group_index * tiling.group_tiles
            n_end = min(tiling.n_tiles, n_start + tiling.group_tiles)
            group_n = n_end - n_start
            accumulators: dict[int, str] = {}
            for n_index in range(n_start, n_end):
                n_off = n_index * tiling.n_tile
                n_size = min(tiling.n_tile, r.n - n_off)
                accumulators[n_index] = plan.add_buffer(
                    f"gemm.accum.{owner}.g{group_index}.n{n_off}",
                    size_bytes=row_size * n_size * 4.0, pool="l0c", owner=owner,
                    shape=(row_size, n_size), dtype="fp32", slot=n_index - n_start,
                    category="output_accumulator",
                )
            prev_k_by_n: dict[int, int | None] = {n: None for n in accumulators}
            last_chunk_gemm: int | None = None
            for k_index in range(tiling.k_tiles):
                k_off = k_index * tiling.k_tile
                k_size = min(tiling.k_tile, r.k - k_off)
                xs = k_index % len(chunk_slot_free)
                chunk = plan.add_buffer(
                    f"norm.pass2.x.{owner}.g{group_index}.k{k_off}",
                    size_bytes=row_size * k_size * b, pool="ub", owner=owner,
                    shape=(row_size, k_size), dtype=r.dtype, slot=xs,
                    category="x_chunk_normalized",
                )
                reread_deps = (
                    () if chunk_slot_free[xs] is None else (chunk_slot_free[xs],)
                )
                reread_stage = _stage_common(
                    plan, launch=False,
                    name=f"norm.reread.{owner}.g{group_index}.k{k_off}",
                    family="rmsnorm", deps=reread_deps, kind="rms_second_pass_load",
                    tile_shape=(row_size, k_size),
                    extents={"m": [row_off, row_size], "k": [k_off, k_size]},
                    mte2_bytes=x_mte2 * row_size * k_size / max(_EPS, total_m * r.k),
                    hbm_bytes=x_hbm * row_size * k_size / max(_EPS, total_m * r.k),
                    creates=(chunk,),
                    writes=(chunk,),
                )
                scale_stage = _stage_common(
                    plan, launch=False,
                    name=f"norm.scale.{owner}.g{group_index}.k{k_off}",
                    family="rmsnorm", deps=(reread_stage, last_reduce),
                    kind="rms_scale_cast_partial",
                    tile_shape=(row_size, k_size),
                    extents={"m": [row_off, row_size], "k": [k_off, k_size]},
                    vector_ops=_norm_split(
                        norm_ops * row_size / total_m,
                        0.6 / tiling.k_tiles,
                    ),
                    reads=(chunk, state_buffer), writes=(chunk,),
                )
                state_consumers.append(scale_stage)
                if physical:
                    l1_chunk = plan.add_buffer(
                        f"norm.pass2.l1.{owner}.g{group_index}.k{k_off}",
                        size_bytes=row_size * k_size * b, pool="l1",
                        owner=owner, shape=(row_size, k_size), dtype=r.dtype,
                        slot=xs, category="x_chunk_l1",
                    )
                    handoff_stage = _stage_common(
                        plan, launch=False,
                        name=f"norm.handoff.{owner}.g{group_index}.k{k_off}",
                        family="rmsnorm", deps=(scale_stage,),
                        kind="ub_to_l1_handoff",
                        tile_shape=(row_size, k_size),
                        extents={"m": [row_off, row_size], "k": [k_off, k_size],
                                 "src_pool": "ub", "dst_pool": "l1",
                                 "src_dtype": r.dtype, "dst_dtype": r.dtype},
                        vector_ops=float(row_size * k_size),
                        reads=(chunk,), writes=(l1_chunk,),
                        creates=(l1_chunk,), releases=(chunk,),
                        resources={"mte1": float(row_size * k_size * b)},
                    )
                    gemm_input = l1_chunk
                    chunk_to_release = l1_chunk
                else:
                    handoff_stage = scale_stage
                    gemm_input = chunk
                    chunk_to_release = chunk
                chunk_gemms: list[int] = []
                for n_index in range(n_start, n_end):
                    n_off = n_index * tiling.n_tile
                    n_size = min(tiling.n_tile, r.n - n_off)
                    ws = weight_use % max(1, tiling.weight_slots)
                    weight_use += 1
                    weight_bytes = float(k_size * n_size * b)
                    weight_buffer = plan.add_buffer(
                        f"gemm.weight.{owner}.g{group_index}.n{n_off}.k{k_off}",
                        size_bytes=weight_bytes, pool="l1", owner=owner,
                        shape=(k_size, n_size), dtype=r.dtype, slot=ws,
                        category="weight_tile",
                    )
                    weight_deps = (
                        () if weight_slot_free[ws] is None else (weight_slot_free[ws],)
                    )
                    weight_load = _stage_common(
                        plan, launch=False,
                        name=f"weight.load.{owner}.g{group_index}.n{n_off}.k{k_off}",
                        family=gemm_family, deps=weight_deps, kind="weight_load",
                        tile_shape=(row_size, k_size, n_size),
                        extents={"m": [row_off, row_size], "n": [n_off, n_size],
                                 "k": [k_off, k_size]},
                        mte2_bytes=weight_bytes, hbm_bytes=weight_bytes,
                        creates=(weight_buffer,), writes=(weight_buffer,),
                    )
                    previous = prev_k_by_n[n_index]
                    if previous is None:
                        base_deps = (
                            () if accumulator_free is None else (accumulator_free,)
                        )
                        creates = (accumulators[n_index],)
                    else:
                        base_deps = (previous,)
                        creates = ()
                    if physical:
                        l0a = plan.add_buffer(
                            f"gemm.l0a.{owner}.g{group_index}.n{n_off}.k{k_off}",
                            size_bytes=row_size * k_size * b, pool="l0a",
                            owner=owner, shape=(row_size, k_size),
                            dtype=r.dtype, slot=0, category="mmad_a",
                        )
                        l0b = plan.add_buffer(
                            f"gemm.l0b.{owner}.g{group_index}.n{n_off}.k{k_off}",
                            size_bytes=k_size * n_size * b, pool="l0b",
                            owner=owner, shape=(k_size, n_size),
                            dtype=r.dtype, slot=0, category="mmad_b",
                        )
                        mmad_creates = (*creates, l0a, l0b)
                        mmad_reads = (gemm_input, weight_buffer, l0a, l0b,
                                      accumulators[n_index])
                        mmad_releases = (weight_buffer, l0a, l0b)
                        mmad_resources = {
                            "mte1": float(row_size * k_size * b
                                          + k_size * n_size * b),
                        }
                    else:
                        mmad_creates = creates
                        mmad_reads = (gemm_input, weight_buffer,
                                      accumulators[n_index])
                        mmad_releases = (weight_buffer,)
                        mmad_resources = {}
                    gemm_stage = _stage_common(
                        plan, launch=False,
                        name=f"gemm.{owner}.g{group_index}.n{n_off}.k{k_off}",
                        family=gemm_family,
                        deps=tuple(dict.fromkeys(
                            (*base_deps, weight_load, handoff_stage)
                        )),
                        kind="gemm_k_accumulate",
                        tile_shape=(row_size, n_size, k_size),
                        extents={"m": [row_off, row_size], "n": [n_off, n_size],
                                 "k": [k_off, k_size],
                                 "l0a_pool": "l0a", "l0b_pool": "l0b",
                                 "l0c_pool": "l0c", "src_dtype": r.dtype,
                                 "l0a_bytes": row_size * k_size * b if physical else 0.0,
                                 "l0b_bytes": k_size * n_size * b if physical else 0.0},
                        flops=2.0 * row_size * n_size * k_size,
                        reads=mmad_reads, writes=(accumulators[n_index],),
                        creates=mmad_creates, releases=mmad_releases,
                        resources=mmad_resources,
                    )
                    weight_slot_free[ws] = gemm_stage
                    prev_k_by_n[n_index] = gemm_stage
                    last_chunk_gemm = gemm_stage
                    chunk_gemms.append(gemm_stage)
                if last_chunk_gemm is not None:
                    chunk_release = add_release_join(
                        plan, name=f"release.x.{owner}.g{group_index}.k{k_off}",
                        family="rmsnorm", consumers=chunk_gemms,
                        releases=(chunk_to_release,), kind="x_slot_release_join",
                        extents={"m": [row_off, row_size],
                                 "k": [k_off, k_size]},
                    )
                    chunk_slot_free[xs] = chunk_release
            for n_index in range(n_start, n_end):
                n_off = n_index * tiling.n_tile
                n_size = min(tiling.n_tile, r.n - n_off)
                output_slot = output_use % 2
                output_use += 1
                out_bytes = float(row_size * n_size * b)
                output_buffer = plan.add_buffer(
                    f"gemm.output.{owner}.g{group_index}.n{n_off}",
                    size_bytes=out_bytes, pool="ub", owner=owner,
                    shape=(row_size, n_size), dtype=r.dtype, slot=output_slot,
                    category="output_tile",
                )
                if physical:
                    fp32_buffer = plan.add_buffer(
                        f"gemm.output_fp32.{owner}.g{group_index}.n{n_off}",
                        size_bytes=row_size * n_size * 4.0, pool="ub",
                        owner=owner, shape=(row_size, n_size), dtype="fp32",
                        slot=output_slot, category="output_l0c_copy",
                    )
                    copy_deps = tuple(dict.fromkeys(
                        (prev_k_by_n[n_index], output_slot_free[output_slot])
                        if output_slot_free[output_slot] is not None
                        else (prev_k_by_n[n_index],)
                    ))
                    copy_stage = _stage_common(
                        plan, launch=False,
                        name=f"output.copy.{owner}.g{group_index}.n{n_off}",
                        family=gemm_family, deps=copy_deps,
                        kind="l0c_to_ub_output",
                        tile_shape=(row_size, n_size),
                        extents={"m": [row_off, row_size], "n": [n_off, n_size],
                                 "src_pool": "l0c", "dst_pool": "ub",
                                 "src_dtype": "fp32", "dst_dtype": "fp32"},
                        vector_ops=float(row_size * n_size),
                        reads=(accumulators[n_index],), writes=(fp32_buffer,),
                        creates=(fp32_buffer,),
                        releases=(accumulators[n_index],),
                        resources={"mte1": float(row_size * n_size * 4.0)},
                    )
                    cast_deps = (copy_stage,)
                    cast_reads = (fp32_buffer,)
                    cast_releases = (fp32_buffer,)
                else:
                    fp32_buffer = None
                    copy_stage = prev_k_by_n[n_index]
                    cast_deps = tuple(dict.fromkeys(
                        (prev_k_by_n[n_index], output_slot_free[output_slot])
                        if output_slot_free[output_slot] is not None
                        else (prev_k_by_n[n_index],)
                    ))
                    cast_reads = (accumulators[n_index],)
                    cast_releases = (accumulators[n_index],)
                cast_stage = _stage_common(
                    plan, launch=False,
                    name=f"output.cast.{owner}.g{group_index}.n{n_off}",
                    family=gemm_family, deps=cast_deps,
                    kind=f"output_cast_{r.dtype}",
                    tile_shape=(row_size, n_size),
                    extents={"m": [row_off, row_size], "n": [n_off, n_size],
                             "k": [0, r.k], "full_k": True,
                             "src_pool": "ub", "dst_pool": "ub",
                             "src_dtype": "fp32", "dst_dtype": r.dtype},
                    vector_ops=float(row_size * n_size),
                    reads=cast_reads, writes=(output_buffer,),
                    creates=(output_buffer,), releases=cast_releases,
                )
                output_write = _stage_common(
                    plan, launch=False,
                    name=f"output.write.{owner}.g{group_index}.n{n_off}",
                    family=gemm_family, deps=(cast_stage,), kind="output_write",
                    tile_shape=(row_size, n_size),
                    extents={"m": [row_off, row_size], "n": [n_off, n_size]},
                    mte3_bytes=out_bytes, hbm_bytes=out_bytes,
                    reads=(output_buffer,), releases=(output_buffer,),
                )
                output_slot_free[output_slot] = output_write
                accumulator_free = copy_stage if physical else cast_stage
        state_release = add_release_join(
            plan, name=f"release.state.{owner}", family="rmsnorm",
            consumers=state_consumers, releases=(state_buffer,),
            kind="state_slot_release_join",
            extents={"m": [row_off, row_size]},
        )
        state_slot_free[state_slot] = state_release
        state_consumers = []
    plan.finish(required_by_pool=tiling.required_by_pool)
    plan.notes.update({
        "tiling": {
            "mode": "two_pass", "row_tile": tiling.row_tile,
            "n_tile": tiling.n_tile, "k_tile": tiling.k_tile,
            "group_tiles": tiling.group_tiles, "n_groups": tiling.n_groups,
            "row_tiles": tiling.row_tiles, "n_tiles": tiling.n_tiles,
            "k_tiles": tiling.k_tiles,
            "row_tail": r.m % tiling.row_tile or tiling.row_tile,
            "n_tail": r.n % tiling.n_tile or tiling.n_tile,
            "k_tail": r.k % tiling.k_tile or tiling.k_tile,
        },
        "bf16_boundary": "normalized K chunks cast to BF16 before GEMM",
        "weight_bytes_reloaded": float(
            tiling.row_tiles * r.n * r.k * b
        ),
        "weight_reloads": tiling.row_tiles,
        "two_pass": True,
        "extra_x_read_bytes": x_hbm * tiling.n_groups,
        "x_passes": 1 + tiling.n_groups,
        "physical_path": bool(physical),
        "physical_primitives": (
            "mte2->ub chunks, rms_reduce(ub), rms_scale(ub), ub->l1 chunk, "
            f"l1->l0a/l0b, mmad->l0c, fp32 l0c->ub, output cast->{r.dtype} ub"
        ) if physical else "logical_unified_no_physical_l0_path",
    })
    return plan, {}


__all__ = ["NormRequest", "NormTiling", "plan_norm_gemm_pipeline"]
