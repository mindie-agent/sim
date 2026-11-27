#!/usr/bin/env python3
"""Native AscendC JIT micro-benchmark worker for Domains 1-4 L0 primitives.

[Role] Compile and execute bounded ``bisheng -xasc`` AscendC microkernels in
  isolated subprocesses (with ``TASK_QUEUE_ENABLE=0``) to calibrate pure
  on-chip pipes (P02, P03, P06.pipe_m_barrier, P07..P10 pure UB, P11 SIMT on
  Arch35, P12 intra-core flags, P13 cross-core sync, P15/P17 graph vs eager
  launch) and hardware MTE2 DMA physics (P01 1-pass streaming Cold-HBM, 2D
  BurstTurnRule surface, MTE2_AIC Cold/Warm scaling, P05 L2 2^k stride).
[Invariants] Strictly offline Plane-1 tool outside ``mindie_sim/``. Disables
  PyTorch async task queue before direct stream launches so hardware events
  bracket kernels deterministically; cleans all temp build dirs on exit;
  never emits raw host paths, IPs, hostnames, or usernames.
[Timeouts & Locks] Bounded per-subtask subprocess timeouts (60s-180s); zero
  external locks or persistent state.
[Side-Effect Policy] Writes only ephemeral ``TemporaryDirectory`` objects
  removed on return; prints one JSON line when invoked as CLI subprocess.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_l0_ascendc_kernels import (
    ARCH35_SIMT_CPP,
    DMA_HW_CPP,
    P02_P03_CPP,
    P06_MMAD_CPP,
    P10_SORT_CPP,
    P13_SYNC_CPP,
    VEC_AND_P12_CPP,
)

SUB_MARKER = "__MINDIE_ASCENDC_SUB__:"
ARCH35_PRESET = "npu_arch_3510"
ARCH22_BISHENG = "dav-2201"
ARCH35_BISHENG = "dav-3510"
MIB = 1024 * 1024
DEFAULT_REPEATS = 20
COMPILE_TIMEOUT_S = 120
SYNC_SUBTASK_TIMEOUT_S = 60
DEFAULT_SUBTASK_TIMEOUT_S = 180
WARMUP_DIM = 4096
WARMUP_GEMM_ITERS = 25
BLOCKER_TARGET_MS = 6.0
MIN_SINGLE_GEMM_MS = 0.05
MIN_BLOCKER_REPS = 4
CHUNK_REPEATS = 2
DMA_CHUNK_REPEATS = 8
DMA_SETTLE_ITERS = 3
WARMUP_DIFF_ITERS = 8
WARMUP_MS_ITERS = 6
MIN_VALID_BARRIER_NS = 0.1
DEFAULT_MS_REPS = 16
MIN_DELTA_MS = 1e-6
MS_PER_S = 1e-3
NS_PER_MS = 1e6
NS_PER_S = 1e9
IN_BUF_STRIDE = 16384
OUT_BUF_STRIDE = 32768
IN_SCALE = 0.01
FLAG_N1 = 4000
FLAG_N2 = 36000
FLAG_DELTA = float(FLAG_N2 - FLAG_N1)
VEC_N1 = 200
VEC_N2 = 600
EXP_N1 = 100
EXP_N2 = 300
UB_ALU_FMA_FLOPS_FACTOR = 2.0
UB_ALU_LANE_OPS_PER_ITER = UB_ALU_FMA_FLOPS_FACTOR * 8192.0  # MulAddDst: 2 elementary ALU ops (mul + add) per elem
UB_EXP_ELEMS = 4096
UB_EXP_WORK_PER_ITER = 2.0 * float(UB_EXP_ELEMS)
UB_RED_WORK_PER_ITER = 128.0 * 64.0
UB_CAST_BYTES_PER_ITER = 2.0 * 8192.0 * 6.0
SYNC_N1 = 500
SYNC_N2 = 1500
SYNC_DELTA = float(SYNC_N2 - SYNC_N1)
SYNC_PAIRED_DENOM = float(2 * (SYNC_N2 - SYNC_N1))
LOAD_LARGE_FRACTALS = 64
LOAD_SMALL_FRACTALS = 16
FRACTAL_BYTES = 512.0
LOAD_BYTES_PER_ITER = float(LOAD_LARGE_FRACTALS) * FRACTAL_BYTES
LOAD_SMALL_BYTES = float(LOAD_SMALL_FRACTALS) * FRACTAL_BYTES
LOAD_N1 = 500
LOAD_N2 = 2500
LOAD_PIPE_N = 4000
MMAD_N2 = 6500
MMAD_BF16_FLOPS_PER_ITER = 2.0 * 64.0 * 128.0 * 128.0
MMAD_FP32_FLOPS_PER_ITER = 2.0 * 64.0 * 128.0 * 64.0
MMAD_INT8_FLOPS_PER_ITER = 2.0 * 64.0 * 128.0 * 256.0
MMAD_MXFP4_FLOPS_PER_ITER = 2.0 * 64.0 * 128.0 * 512.0
MX_SCALE_BUF_BYTES = 16384
E8M0_UNITY_BYTE = 127
FIXPIPE_M = 64
FIXPIPE_N = 128
FIXPIPE_PIPE_M = 128
FIXPIPE_PIPE_N = 128
FIXPIPE_BF16_BYTES = 2
FIXPIPE_BYTES_PER_ITER = float(FIXPIPE_M * FIXPIPE_N * FIXPIPE_BF16_BYTES)
FIXPIPE_PIPE_BYTES_PER_ITER = float(FIXPIPE_PIPE_M * FIXPIPE_PIPE_N * FIXPIPE_BF16_BYTES)
FIXPIPE_SMALL_M = 32
FIXPIPE_SMALL_N = 64
FIXPIPE_SMALL_BYTES = float(FIXPIPE_SMALL_M * FIXPIPE_SMALL_N * FIXPIPE_BF16_BYTES)
FIXPIPE_N1 = 400
FIXPIPE_N2 = 1400
FIXPIPE_PIPE_ITERS = 2000
SORT_N1 = 50
SORT_N2 = 150
SORT_ELEMS_PER_ITER = 1024.0
SORT_PASSES_PER_ITER = 4.0  # 1 Sort32 + 3 MrgSort4 passes for N=1024
SIMT_N1 = 500
SIMT_N2 = 1500
SIMT_LAUNCH_N1 = 50
SIMT_LAUNCH_N2 = 450
SIMT_WARP_INSTS_PER_ITER = 16.0 * 4.0
SIMT_SCALAR_FLOPS_PER_ITER = 512.0 * 8.0
CACHE_LINE_BYTES = 512
MIN_VALID_DELTAS = 8
COLD_SLOT_MIB = 128
GATHER_SLAB_MIN_MIB = 640
GATHER_SLAB_MAX_MIB = 1024
GATHER_XFER_MIB = 32
GATHER_ROW_BYTES_GRID = (128, 256, 512, 576, 1024, 2048, 4096)
GATHER_BF16_BYTES = 2
GATHER_INT64_BYTES = 8
GATHER_MAX_SLOTS = 8
GATHER_SEED_BASE = 2026
GATHER_PENALTY_DIGITS = 4
GATHER_RW_MULT = 2.0
COLD_L2_MULT = 4
MIN_RING_SLOTS = 5
DMA_CHUNK_BYTES = 32768
ALIGN_OFFSETS = (0, 32, 64, 128, 256)
DMA_2D_SURFACE_PAIRS = (
    (32, 512), (32, 1024), (32, 4096),
    (64, 512), (64, 1024), (64, 4096),
    (128, 512), (128, 1024), (128, 4096),
    (256, 512), (256, 1024), (256, 4096),
    (512, 1024), (512, 2048), (512, 4096),
    (1024, 2048), (1024, 4096), (2048, 4096),
)
DMA_2D_UB_CAP = 65536
DMA_2D_MAX_RPB = 256
L2_CONFLICT_BUF_MIB = 64
L2_CONFLICT_WS_MIB = 16
L2_CONFLICT_ROW_BYTES = 128
L2_CONFLICT_MAX_ROWS = 65536
L2_CONFLICT_MAX_RPB = 128
L2_CONFLICT_MAX_REPS = 200
L2_CONFLICT_REPEATS = 12
L2_STRIDE_GRID = (
    (128, "contiguous"), (512, "pow2"), (544, "coprime"),
    (1024, "pow2"), (1056, "coprime"), (4096, "pow2"), (4128, "coprime"),
    (16384, "pow2"), (16416, "coprime"), (32768, "pow2"), (32800, "coprime"),
    (65536, "pow2"), (65568, "coprime"), (131072, "pow2"),
)
AIC_COLD_SLOT_MIB = 64
AIC_WARM_SLOT_MIB = 32
AIC_CHUNK_BYTES = 65536
AIC_WARM_REPS = 16
AIC_CORE_SWEEP = (1, 2, 4, 8, 12, 16)
AIC_REPEATS = 12
GRAPH_BURST_OPS = 64
GRAPH_WARMUP_ITERS = 16
GRAPH_REPLAY_WARMUP = 4
GRAPH_TENSOR_ELEMS = 1024
ROUND_BPS_DIGITS = 2
CANN_GLOB = "/usr/local/Ascend/cann-9*"
CANN_FALLBACK = "/usr/local/Ascend/ascend-toolkit/latest"
ERR_TAIL_CHARS = 1200


def _compile_so(src: str, arch: str, cann_root: str, tmpdir: Path, name: str) -> tuple[ctypes.CDLL | None, str]:
    cpp = tmpdir / f"{name}.cpp"
    so = tmpdir / f"{name}.so"
    cpp.write_text(src, encoding="utf-8")
    defs = ["-D_MINDIE_ARCH35=1"] if arch == ARCH35_BISHENG else []
    cmd = [
        "bisheng", "-xasc", f"--npu-arch={arch}", *defs, "-std=c++17", "-O3", "-fPIC", "-shared",
        str(cpp), f"-L{cann_root}/aarch64-linux/lib64", "-lruntime", "-lascendcl", "-o", str(so),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S, check=False)
    if proc.returncode != 0 or not so.is_file():
        return None, (proc.stdout + "\n" + proc.stderr)[-ERR_TAIL_CHARS:]
    return ctypes.CDLL(str(so)), ""


def _run_p17_launch_subtask(
    torch: Any, dev: Any, repeats: int, res: dict[str, Any],
) -> None:
    x = torch.ones((GRAPH_TENSOR_ELEMS,), dtype=torch.bfloat16, device=dev)
    y = torch.empty_like(x)
    for _ in range(GRAPH_WARMUP_ITERS):
        torch.add(x, x, out=y)
    torch.npu.synchronize()
    host_ns: list[float] = []
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(GRAPH_BURST_OPS):
            torch.add(x, x, out=y)
        t1 = time.perf_counter()
        torch.npu.synchronize()
        host_ns.append((t1 - t0) * NS_PER_S / float(GRAPH_BURST_OPS))
    res["samples"]["P15.host_submit_burst_ns"] = host_ns
    try:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(GRAPH_BURST_OPS):
                torch.add(x, x, out=y)
        for _ in range(GRAPH_REPLAY_WARMUP):
            graph.replay()
        torch.npu.synchronize()
        g_ns: list[float] = []
        for _ in range(repeats):
            se, ee = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            se.record()
            graph.replay()
            ee.record()
            torch.npu.synchronize()
            g_ns.append(float(se.elapsed_time(ee)) * NS_PER_MS / float(GRAPH_BURST_OPS))
        res["samples"]["P15.graph_issue_interval_ns"] = g_ns
    except Exception as exc:
        res["errors"]["npu_graph"] = str(exc)[:ERR_TAIL_CHARS]


def _run_dma_hw_subtask(
    torch: Any, dev: Any, lib: Any, out_p: Any, st_p: Any,
    *, aic: int, aiv: int, l2_bytes: int, repeats: int, time_ms: Any, res: dict[str, Any],
) -> None:
    slot_b = COLD_SLOT_MIB * MIB
    flush_slots = max(1, math.ceil((COLD_L2_MULT * l2_bytes) / slot_b))
    n_slots = max(MIN_RING_SLOTS, flush_slots + 1)
    ring = [torch.ones((slot_b,), dtype=torch.uint8, device=dev) for _ in range(n_slots)]
    ring_p = [ctypes.c_void_p(t.data_ptr()) for t in ring]
    align_bps: dict[str, float] = {}
    for off_b in ALIGN_OFFSETS:
        per_core = ((slot_b - off_b) // aiv // DMA_CHUNK_BYTES) * DMA_CHUNK_BYTES
        tot_eff = float(per_core * aiv)
        fn = lambda i, ob=off_b: lib.run_dma_align(ring_p[i % n_slots], out_p, ctypes.c_longlong(slot_b), ctypes.c_int(ob), ctypes.c_int(DMA_CHUNK_BYTES), ctypes.c_int(1), st_p, ctypes.c_int(aiv))
        ts = time_ms(fn, reps=repeats, use_blocker=False)
        bws = [tot_eff / (t * MS_PER_S) for t in ts]
        align_bps[f"off_{off_b}B"] = round(statistics.median(bws), ROUND_BPS_DIGITS)
        if off_b == 0:
            ts_blk = time_ms(fn, reps=repeats, use_blocker=True)
            bws_blk = [tot_eff / (t * MS_PER_S) for t in ts_blk]
            res["samples"]["P01.hbm_read_datacopy_bytes_per_s"] = bws_blk
            res["samples"]["P01.hbm_read_datacopy_eager_bytes_per_s"] = bws
            res["metadata"]["hbm_read_datacopy_blocked_bps_p50"] = round(statistics.median(bws_blk), ROUND_BPS_DIGITS)
            res["metadata"]["hbm_read_datacopy_eager_bps_p50"] = round(statistics.median(bws), ROUND_BPS_DIGITS)
    res["metadata"]["gm_alignment_cold_bps"] = align_bps

    def _evict_l2(seed: int) -> None:
        for k in range(1, flush_slots + 1):
            lib.run_dma_align(ring_p[(seed + k) % n_slots], out_p, ctypes.c_longlong(slot_b), ctypes.c_int(0), ctypes.c_int(DMA_CHUNK_BYTES), ctypes.c_int(1), st_p, ctypes.c_int(aiv))

    surf: list[dict[str, Any]] = []
    for rb, sb in DMA_2D_SURFACE_PAIRS:
        rpb = max(1, min(DMA_2D_MAX_RPB, DMA_2D_UB_CAP // rb))
        max_rows = slot_b // sb
        rows_core = (max_rows // aiv // rpb) * rpb
        if rows_core <= 0:
            rpb = max(1, max_rows // aiv)
            rows_core = rpb
        tot_bytes = float(rows_core * aiv * rb)
        touched_lines = n_slots * rows_core * aiv * min(sb, max(rb, CACHE_LINE_BYTES))
        evict = _evict_l2 if touched_lines < COLD_L2_MULT * l2_bytes else None
        ts = time_ms(lambda i, nr=rows_core * aiv, r=rb, s=sb, b=rpb: lib.run_dma_2d(ring_p[i % n_slots], ctypes.c_int(nr), ctypes.c_int(r), ctypes.c_int(s), ctypes.c_int(b), ctypes.c_int(1), st_p, ctypes.c_int(aiv)), reps=AIC_REPEATS, pre_evict=evict)
        surf.append({"row_bytes": rb, "stride_bytes": sb, "bps_p50": round(statistics.median([tot_bytes / (t * MS_PER_S) for t in ts]), ROUND_BPS_DIGITS), "cold_evicted": bool(evict is not None or touched_lines >= COLD_L2_MULT * l2_bytes)})
    res["metadata"]["strided_cold_surface"] = surf
    del ring, ring_p
    torch.npu.empty_cache()

    aic_slot = AIC_COLD_SLOT_MIB * MIB
    n_aic = max(MIN_RING_SLOTS, math.ceil((COLD_L2_MULT * l2_bytes) / aic_slot))
    aring = [torch.ones((aic_slot,), dtype=torch.uint8, device=dev) for _ in range(n_aic)]
    aring_p = [ctypes.c_void_p(t.data_ptr()) for t in aring]
    aic_cold: dict[str, float] = {}
    core_grid = sorted({c for c in AIC_CORE_SWEEP if c <= aic} | {aic})
    for nc in core_grid:
        tot_b = float(((aic_slot // nc // AIC_CHUNK_BYTES) * AIC_CHUNK_BYTES) * nc)
        ts = time_ms(lambda i, c=nc: lib.run_aic_mte2(aring_p[i % n_aic], ctypes.c_longlong(aic_slot), ctypes.c_int(AIC_CHUNK_BYTES), ctypes.c_int(1), st_p, ctypes.c_int(c)), reps=AIC_REPEATS, use_blocker=False)
        aic_cold[str(nc)] = round(statistics.median([tot_b / (t * MS_PER_S) for t in ts]), ROUND_BPS_DIGITS)
    del aring, aring_p
    torch.npu.empty_cache()

    l2_buf = torch.ones((L2_CONFLICT_BUF_MIB * MIB,), dtype=torch.uint8, device=dev)
    l2_p = ctypes.c_void_p(l2_buf.data_ptr())
    aic_warm: dict[str, float] = {}
    warm_slot = AIC_WARM_SLOT_MIB * MIB
    for nc in core_grid:
        tot_b = float(((warm_slot // nc // AIC_CHUNK_BYTES) * AIC_CHUNK_BYTES) * nc * AIC_WARM_REPS)
        ts = time_ms(lambda _i, c=nc: lib.run_aic_mte2(l2_p, ctypes.c_longlong(warm_slot), ctypes.c_int(AIC_CHUNK_BYTES), ctypes.c_int(AIC_WARM_REPS), st_p, ctypes.c_int(c)), reps=AIC_REPEATS)
        bws = [tot_b / (t * MS_PER_S) for t in ts]
        aic_warm[str(nc)] = round(statistics.median(bws), ROUND_BPS_DIGITS)
        if nc == aic:
            res["samples"]["P01.mte2_aic_hit_bytes_per_s_per_core"] = [bw / float(aic) for bw in bws]
    res["metadata"]["mte2_aic_cold_scaling_bps"] = aic_cold
    res["metadata"]["mte2_aic_warm_scaling_bps"] = aic_warm

    aiv_warm_bytes = float(((warm_slot // aiv // DMA_CHUNK_BYTES) * DMA_CHUNK_BYTES) * aiv * AIC_WARM_REPS)
    ts_aiv_warm = time_ms(lambda _i: lib.run_dma_align(l2_p, out_p, ctypes.c_longlong(warm_slot), ctypes.c_int(0), ctypes.c_int(DMA_CHUNK_BYTES), ctypes.c_int(AIC_WARM_REPS), st_p, ctypes.c_int(aiv)), reps=repeats)
    aiv_warm_die_bps = [aiv_warm_bytes / (t * MS_PER_S) for t in ts_aiv_warm]
    res["samples"]["P01.mte2_aiv_hit_bytes_per_s_per_core"] = [bw / float(aiv) for bw in aiv_warm_die_bps]

    l2_conf: list[dict[str, Any]] = [
        {"stride_bytes": L2_CONFLICT_ROW_BYTES, "kind": "contiguous", "bps_p50": round(statistics.median(aiv_warm_die_bps), ROUND_BPS_DIGITS)},
    ]
    for sb, kind in L2_STRIDE_GRID:
        max_rows = min((L2_CONFLICT_WS_MIB * MIB) // sb, L2_CONFLICT_MAX_ROWS)
        rpb = max(1, min(L2_CONFLICT_MAX_RPB, max_rows // max(1, aiv)))
        tot_rows = rpb * aiv
        reps = max(1, min(L2_CONFLICT_MAX_REPS, (L2_CONFLICT_WS_MIB * MIB) // max(1, tot_rows * L2_CONFLICT_ROW_BYTES)))
        tot_b = float(tot_rows * L2_CONFLICT_ROW_BYTES * reps)
        ts = time_ms(lambda _i, nr=tot_rows, s=sb, b=rpb, rp=reps: lib.run_dma_2d(l2_p, ctypes.c_int(nr), ctypes.c_int(L2_CONFLICT_ROW_BYTES), ctypes.c_int(s), ctypes.c_int(b), ctypes.c_int(rp), st_p, ctypes.c_int(aiv)), reps=L2_CONFLICT_REPEATS)
        entry_kind = "row128_2d" if kind == "contiguous" else kind
        l2_conf.append({"stride_bytes": sb, "kind": entry_kind, "bps_p50": round(statistics.median([tot_b / (t * MS_PER_S) for t in ts]), ROUND_BPS_DIGITS)})
    res["metadata"]["l2_stride_conflict"] = l2_conf
    del l2_buf
    torch.npu.empty_cache()

    g_slab = min(GATHER_SLAB_MAX_MIB * MIB, max(COLD_L2_MULT * l2_bytes, GATHER_SLAB_MIN_MIB * MIB))
    g_xfer = GATHER_XFER_MIB * MIB
    gather_surf: dict[str, dict[str, Any]] = {}
    for rb in GATHER_ROW_BYTES_GRID:
        cols, n_tot, n_pick = rb // GATHER_BF16_BYTES, g_slab // rb, g_xfer // rb
        g_src = torch.ones((n_tot, cols), dtype=torch.bfloat16, device=dev)
        g_dst = torch.empty((n_pick, cols), dtype=torch.bfloat16, device=dev)
        n_sl = max(MIN_BLOCKER_REPS, min(GATHER_MAX_SLOTS, n_tot // max(1, n_pick)))
        seq_i = [torch.arange(k * n_pick, (k + 1) * n_pick, dtype=torch.int64, device=dev) for k in range(n_sl)]
        gen = torch.Generator(device="cpu").manual_seed(GATHER_SEED_BASE + rb)
        rnd_i = [torch.randint(0, n_tot, (n_pick,), generator=gen, dtype=torch.int64).to(dev) for _ in range(n_sl)]
        t_seq = statistics.median(time_ms(lambda i, s=g_src, d=g_dst, ix=seq_i, n=n_sl: torch.index_select(s, 0, ix[i % n], out=d), reps=AIC_REPEATS))
        t_rnd = statistics.median(time_ms(lambda i, s=g_src, d=g_dst, ix=rnd_i, n=n_sl: torch.index_select(s, 0, ix[i % n], out=d), reps=AIC_REPEATS))
        tot_rw = float(n_pick) * (GATHER_RW_MULT * float(rb) + float(GATHER_INT64_BYTES))
        gather_surf[f"{rb}B"] = {
            "row_bytes": rb,
            "sequential_rw_bps": round(tot_rw / (t_seq * MS_PER_S), ROUND_BPS_DIGITS),
            "random_scatter_rw_bps": round(tot_rw / (t_rnd * MS_PER_S), ROUND_BPS_DIGITS),
            "scatter_penalty_ratio": round(t_rnd / max(MIN_DELTA_MS, t_seq), GATHER_PENALTY_DIGITS),
        }
        del g_src, g_dst, seq_i, rnd_i
        torch.npu.empty_cache()
    res["metadata"]["gather_v2_cold_surface"] = gather_surf


def _fit_two_point_pipe(
    ds_large: list[float], ds_small: list[float],
    *, n_delta_large: float, n_delta_small: float, bytes_large: float, bytes_small: float,
) -> tuple[list[float], list[float]]:
    if not ds_large or not ds_small:
        return [], []
    tl_list = [(dl / n_delta_large) * MS_PER_S for dl in ds_large]
    ts_list = [(ds_v / n_delta_small) * MS_PER_S for ds_v in ds_small]
    tl_med, ts_med = statistics.median(tl_list), statistics.median(ts_list)
    dt_med = tl_med - ts_med
    if dt_med <= 0.0:
        return [], []
    slope_med = (bytes_large - bytes_small) / dt_med
    setup_med_s = max(0.0, ts_med - bytes_small / slope_med)
    rates = [bytes_large / (tl - setup_med_s) for tl in tl_list if tl > setup_med_s]
    setups = [max(0.0, (ts_v - bytes_small / slope_med) * NS_PER_S) for ts_v in ts_list]
    return rates, setups


def _run_cube_subtask(
    sub: str, *, arch: str, cann_root: str, tmp: Path,
    in_p: Any, mx_p: Any, out_p: Any, st_p: Any, aic: int, repeats: int,
    time_diff: Any, time_ms: Any, lock_dvfs: Any, res: dict[str, Any],
) -> None:
    if sub == "p02_p03":
        lib, err = _compile_so(P02_P03_CPP, arch, cann_root, tmp, sub)
        if err or lib is None:
            res["errors"]["compile"] = err
            return
        lock_dvfs()
        load_delta = float(LOAD_N2 - LOAD_N1)
        load_pipe_tot = LOAD_BYTES_PER_ITER * float(LOAD_PIPE_N)
        for key, runner, tr_flag, store_setup in (
            ("P02.l1_to_l0a_bytes_per_s_per_aic", lib.run_p02_l0a, 0, True),
            ("P02.l1_to_l0a_trans_bytes_per_s_per_aic", lib.run_p02_l0a, 1, False),
            ("P02.l1_to_l0b_bytes_per_s_per_aic", lib.run_p02_l0b, 0, False),
            ("P02.l1_to_l0b_trans_bytes_per_s_per_aic", lib.run_p02_l0b, 1, False),
        ):
            ts_pipe = time_ms(
                lambda _i, f=runner, tr=tr_flag: f(in_p, ctypes.c_int(LOAD_PIPE_N), ctypes.c_int(tr), ctypes.c_int(LOAD_LARGE_FRACTALS), ctypes.c_int(0), st_p, ctypes.c_int(aic)),
                reps=repeats,
            )
            if len(ts_pipe) >= MIN_VALID_DELTAS:
                res["samples"][key] = [load_pipe_tot / (t * MS_PER_S) for t in ts_pipe]
            if store_setup:
                ds_sm = time_diff(lambda n, f=runner, tr=tr_flag: f(in_p, ctypes.c_int(n), ctypes.c_int(tr), ctypes.c_int(LOAD_SMALL_FRACTALS), ctypes.c_int(1), st_p, ctypes.c_int(aic)), LOAD_N1, LOAD_N2)
                ds_lg = time_diff(lambda n, f=runner, tr=tr_flag: f(in_p, ctypes.c_int(n), ctypes.c_int(tr), ctypes.c_int(LOAD_LARGE_FRACTALS), ctypes.c_int(1), st_p, ctypes.c_int(aic)), LOAD_N1, LOAD_N2)
                _, setups = _fit_two_point_pipe(
                    ds_lg, ds_sm, n_delta_large=load_delta, n_delta_small=load_delta,
                    bytes_large=LOAD_BYTES_PER_ITER, bytes_small=LOAD_SMALL_BYTES,
                )
                if len(setups) >= MIN_VALID_DELTAS:
                    res["samples"]["P02.setup_ns"] = setups
        fix_pipe_tot = FIXPIPE_PIPE_BYTES_PER_ITER * float(FIXPIPE_PIPE_ITERS)
        ts_fix = time_ms(
            lambda _i: lib.run_p03_fix_gm(out_p, ctypes.c_int(FIXPIPE_PIPE_ITERS), ctypes.c_int(FIXPIPE_PIPE_M), ctypes.c_int(FIXPIPE_PIPE_N), ctypes.c_int(0), st_p, ctypes.c_int(aic)),
            reps=repeats,
        )
        if len(ts_fix) >= MIN_VALID_DELTAS:
            res["samples"]["P03.fixpipe_out_bytes_per_s_per_aic"] = [fix_pipe_tot / (t * MS_PER_S) for t in ts_fix]
        fix_delta = float(FIXPIPE_N2 - FIXPIPE_N1)
        ds_large = time_diff(lambda n: lib.run_p03_fix_gm(out_p, ctypes.c_int(n), ctypes.c_int(FIXPIPE_M), ctypes.c_int(FIXPIPE_N), ctypes.c_int(1), st_p, ctypes.c_int(aic)), FIXPIPE_N1, FIXPIPE_N2)
        ds_small = time_diff(lambda n: lib.run_p03_fix_gm(out_p, ctypes.c_int(n), ctypes.c_int(FIXPIPE_SMALL_M), ctypes.c_int(FIXPIPE_SMALL_N), ctypes.c_int(1), st_p, ctypes.c_int(aic)), FIXPIPE_N1, FIXPIPE_N2)
        _, fix_setups = _fit_two_point_pipe(
            ds_large, ds_small, n_delta_large=fix_delta, n_delta_small=fix_delta,
            bytes_large=FIXPIPE_BYTES_PER_ITER, bytes_small=FIXPIPE_SMALL_BYTES,
        )
        if len(fix_setups) >= MIN_VALID_DELTAS:
            res["samples"]["P03.fixpipe_setup_ns"] = fix_setups
    elif sub == "p06_mmad":
        lib, err = _compile_so(P06_MMAD_CPP, arch, cann_root, tmp, sub)
        if err or lib is None:
            res["errors"]["compile"] = err
            return
        lock_dvfs()
        modes = [
            ("bf16", "run_p06_bf16", MMAD_BF16_FLOPS_PER_ITER),
            ("fp16", "run_p06_fp16", MMAD_BF16_FLOPS_PER_ITER),
            ("fp32", "run_p06_fp32", MMAD_FP32_FLOPS_PER_ITER),
            ("hf32", "run_p06_hf32", MMAD_FP32_FLOPS_PER_ITER),
            ("int8", "run_p06_int8", MMAD_INT8_FLOPS_PER_ITER),
        ]
        if arch == ARCH35_BISHENG:
            modes.extend([
                ("fp8_e4m3", "run_p06_fp8e4", MMAD_INT8_FLOPS_PER_ITER),
                ("fp8_e5m2", "run_p06_fp8e5", MMAD_INT8_FLOPS_PER_ITER),
                ("hifloat8", "run_p06_hif8", MMAD_INT8_FLOPS_PER_ITER),
            ])
        for mode_name, sym, flops_iter in modes:
            if not hasattr(lib, sym):
                continue
            fn_sym = getattr(lib, sym)
            ts = time_ms(lambda _i, f=fn_sym: f(in_p, ctypes.c_int(MMAD_N2), st_p, ctypes.c_int(aic)), reps=repeats)
            if len(ts) >= MIN_VALID_DELTAS:
                tot = flops_iter * float(MMAD_N2)
                res["samples"][f"P06.cube_only.{mode_name}"] = [tot / (t * MS_PER_S) for t in ts]
        if arch == ARCH35_BISHENG:
            for mode_name, sym, flops_iter in (
                ("mxfp8", "run_p06_mxfp8", MMAD_INT8_FLOPS_PER_ITER),
                ("mxfp4", "run_p06_mxfp4", MMAD_MXFP4_FLOPS_PER_ITER),
            ):
                if not hasattr(lib, sym):
                    continue
                fn_sym = getattr(lib, sym)
                ts = time_ms(
                    lambda _i, f=fn_sym: f(in_p, mx_p, ctypes.c_int(MMAD_N2), st_p, ctypes.c_int(aic)),
                    reps=repeats,
                )
                if len(ts) >= MIN_VALID_DELTAS:
                    tot = flops_iter * float(MMAD_N2)
                    res["samples"][f"P06.cube_only.{mode_name}"] = [tot / (t * MS_PER_S) for t in ts]


def _run_kernel_subtask(
    sub: str, *, torch: Any, out_buf: Any, arch: str, cann_root: str, tmp: Path,
    in_p: Any, mx_p: Any, out_p: Any, st_p: Any, aic: int, aiv: int, repeats: int,
    time_diff: Any, time_ms: Any, lock_dvfs: Any, res: dict[str, Any],
) -> None:
    if sub in ("p02_p03", "p06_mmad"):
        _run_cube_subtask(
            sub, arch=arch, cann_root=cann_root, tmp=tmp,
            in_p=in_p, mx_p=mx_p, out_p=out_p, st_p=st_p, aic=aic, repeats=repeats,
            time_diff=time_diff, time_ms=time_ms, lock_dvfs=lock_dvfs, res=res,
        )
        return
    if sub == "vec_p12":
        lib, err = _compile_so(VEC_AND_P12_CPP, arch, cann_root, tmp, sub)
        if err or lib is None:
            res["errors"]["compile"] = err
            return
        lock_dvfs()
        flag_probes: list[tuple[str, Any]] = [
            ("P12.flag_mte2_v_ns", lambda n: lib.run_p12_mte2_v(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv))),
            ("P12.flag_v_mte3_ns", lambda n: lib.run_p12_v_mte3(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv))),
            ("P12.pipe_v_barrier_ns", lambda n: lib.run_p12_pipe_v(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv))),
            ("P12.flag_mte1_m_ns", lambda n: lib.run_p12_mte1_m(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic))),
            ("P12.flag_m_fix_ns", lambda n: lib.run_p12_m_fix(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic))),
            ("P06.pipe_m_barrier_ns", lambda n: lib.run_p06_pipe_m(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic))),
        ]
        if arch == ARCH35_BISHENG and hasattr(lib, "run_p14_mutex"):
            flag_probes.append(("P14.mutex_overhead_ns", lambda n: lib.run_p14_mutex(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic))))
        if arch == ARCH35_BISHENG and hasattr(lib, "run_p14_mutex_xpipe"):
            flag_probes.append(("P14.mutex_xpipe_handoff_ns", lambda n: lib.run_p14_mutex_xpipe(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic))))
        for key, fn in flag_probes:
            ds = time_diff(fn, FLAG_N1, FLAG_N2)
            vals = [d * (NS_PER_MS / FLAG_DELTA) for d in ds]
            if len(vals) >= MIN_VALID_DELTAS and statistics.median(vals) >= MIN_VALID_BARRIER_NS:
                res["samples"][key] = vals
        for key, fn, n1, n2, work in (
            ("P07.ub_alu_ops_per_s_per_aiv", lambda n: lib.run_p07_alu(in_p, out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), VEC_N1, VEC_N2, UB_ALU_LANE_OPS_PER_ITER),
            ("P08.ub_sfu_ops_per_s_per_aiv", lambda n: lib.run_p08_exp(in_p, out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), EXP_N1, EXP_N2, UB_EXP_WORK_PER_ITER),
            ("P09.ub_reduce_ops_per_s_per_aiv", lambda n: lib.run_p09_red(in_p, out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), VEC_N1, VEC_N2, UB_RED_WORK_PER_ITER),
            ("P09.ub_cast_bytes_per_s_per_aiv", lambda n: lib.run_p09_cast(in_p, out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), VEC_N1, VEC_N2, UB_CAST_BYTES_PER_ITER),
        ):
            ds = time_diff(fn, n1, n2)
            if key == "P08.ub_sfu_ops_per_s_per_aiv" and not bool(torch.isfinite(out_buf[: aiv * UB_EXP_ELEMS]).all()):
                res["errors"]["p08_nonfinite"] = "exp_output_overflow"
                continue
            if len(ds) >= MIN_VALID_DELTAS:
                tot = work * float(n2 - n1)
                res["samples"][key] = [(tot / (d * MS_PER_S)) if d > 0.0 else 0.0 for d in ds]
                if key == "P07.ub_alu_ops_per_s_per_aiv":
                    res["metadata"]["p07_work_unit"] = "elementary_alu_ops_per_s"
    elif sub in ("p13_aiv", "p13_aic", "p13_mode2", "p13_mode4"):
        lib, err = _compile_so(P13_SYNC_CPP, arch, cann_root, tmp, sub)
        if err or lib is None:
            res["errors"]["compile"] = err
            return
        lock_dvfs()
        table: dict[str, tuple[str, Any, float]] = {
            "p13_aiv": ("P13.mode0_aiv_barrier_ns", lambda n: lib.run_p13_aiv(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), NS_PER_MS / SYNC_DELTA),
            "p13_aic": ("P13.mode0_aic_barrier_ns", lambda n: lib.run_p13_aic(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic)), NS_PER_MS / SYNC_DELTA),
            "p13_mode2": ("P13.mode2_paired_ns", lambda n: lib.run_p13_mode2(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic)), NS_PER_MS / SYNC_PAIRED_DENOM),
        }
        if arch == ARCH35_BISHENG and hasattr(lib, "run_p13_mode4"):
            table["p13_mode4"] = ("P13.mode4_intra_block_ns", lambda n: lib.run_p13_mode4(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aic)), NS_PER_MS / SYNC_PAIRED_DENOM)
        if sub not in table:
            return
        key, fn, scale = table[sub]
        vals = [d * scale for d in time_diff(fn, SYNC_N1, SYNC_N2)]
        if len(vals) >= MIN_VALID_DELTAS and statistics.median(vals) >= MIN_VALID_BARRIER_NS:
            res["samples"][key] = vals
    elif sub == "p10_sort":
        lib, err = _compile_so(P10_SORT_CPP, arch, cann_root, tmp, sub)
        if err or lib is None:
            res["errors"]["compile"] = err
            return
        lock_dvfs()
        ds = time_diff(lambda n: lib.run_p10_sort(in_p, out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), SORT_N1, SORT_N2)
        if len(ds) >= MIN_VALID_DELTAS:
            work_pass = SORT_PASSES_PER_ITER * SORT_ELEMS_PER_ITER * float(SORT_N2 - SORT_N1)
            per_aiv = [(work_pass / (d * MS_PER_S)) if d > 0.0 else 0.0 for d in ds]
            res["samples"]["P10.sort_elems_per_s_per_aiv"] = per_aiv
            res["metadata"]["p10_work_unit"] = "elem_passes_per_s"
            res["metadata"]["sort_die_elems_per_s_p50"] = round(statistics.median(per_aiv) * float(aiv), ROUND_BPS_DIGITS)
    elif sub == "p11_simt" and arch == ARCH35_BISHENG:
        lib, err = _compile_so(ARCH35_SIMT_CPP, arch, cann_root, tmp, sub)
        if err or lib is None:
            res["errors"]["compile"] = err
            return
        lock_dvfs()
        ds_alu = time_diff(lambda n: lib.run_p11_alu(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), SIMT_N1, SIMT_N2)
        if len(ds_alu) >= MIN_VALID_DELTAS:
            res["samples"]["P11.simt_warp_instructions_per_s_per_aiv"] = [(SIMT_WARP_INSTS_PER_ITER * float(SIMT_N2 - SIMT_N1)) / (d * MS_PER_S) for d in ds_alu]
            res["metadata"]["simt_scalar_flops_per_s_per_aiv_p50"] = round(statistics.median([(SIMT_SCALAR_FLOPS_PER_ITER * float(SIMT_N2 - SIMT_N1)) / (d * MS_PER_S) for d in ds_alu]), ROUND_BPS_DIGITS)
        ds_bar = time_diff(lambda n: lib.run_p11_bar(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), SIMT_N1, SIMT_N2)
        if len(ds_bar) >= MIN_VALID_DELTAS:
            res["samples"]["P11.syncthreads_ns"] = [d * NS_PER_MS / float(SIMT_N2 - SIMT_N1) for d in ds_bar]
        ds_lch = time_diff(lambda n: lib.run_p11_launch(out_p, ctypes.c_int(n), st_p, ctypes.c_int(aiv)), SIMT_LAUNCH_N1, SIMT_LAUNCH_N2)
        if len(ds_lch) >= MIN_VALID_DELTAS:
            res["samples"]["P11.simt_launch_ns"] = [d * NS_PER_MS / float(SIMT_LAUNCH_N2 - SIMT_LAUNCH_N1) for d in ds_lch]


def _run_subtask(
    sub: str, *, device_id: int, arch: str, aic: int, aiv: int, l2_bytes: int, repeats: int,
) -> dict[str, Any]:
    os.environ["TASK_QUEUE_ENABLE"] = "0"
    import torch
    import torch_npu  # noqa: F401

    env_cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME") or ""
    cann_roots = sorted(glob.glob(CANN_GLOB))
    cann_root = env_cann if (env_cann and Path(env_cann).is_dir()) else (cann_roots[-1] if cann_roots else CANN_FALLBACK)
    torch.npu.set_device(device_id)
    dev = torch.device(f"npu:{device_id}")

    wa = torch.randn((WARMUP_DIM, WARMUP_DIM), dtype=torch.bfloat16, device=dev)
    wb = torch.randn((WARMUP_DIM, WARMUP_DIM), dtype=torch.bfloat16, device=dev)

    def _lock_dvfs() -> None:
        for _ in range(WARMUP_GEMM_ITERS):
            _ = torch.matmul(wa, wb)
        torch.npu.synchronize()

    _lock_dvfs()
    s0, e0 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s0.record()
    _ = torch.matmul(wa, wb)
    e0.record()
    torch.npu.synchronize()
    blocker_reps = max(MIN_BLOCKER_REPS, math.ceil(BLOCKER_TARGET_MS / max(MIN_SINGLE_GEMM_MS, float(s0.elapsed_time(e0)))))

    def _time_diff(fn_call: Any, n1: int, n2: int) -> list[float]:
        for _ in range(WARMUP_DIFF_ITERS):
            fn_call(n1)
            fn_call(n2)
        torch.npu.synchronize()
        deltas: list[float] = []
        for start in range(0, repeats, CHUNK_REPEATS):
            k = min(CHUNK_REPEATS, repeats - start)
            evs = [
                (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True),
                 torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                for _ in range(k)
            ]
            for _ in range(blocker_reps):
                _ = torch.matmul(wa, wb)
            fn_call(n1)
            for s1, e1, s2, e2 in evs:
                s1.record(); fn_call(n1); e1.record()
                s2.record(); fn_call(n2); e2.record()
            torch.npu.synchronize()
            for s1, e1, s2, e2 in evs:
                deltas.append(float(s2.elapsed_time(e2)) - float(s1.elapsed_time(e1)))
        return deltas

    def _time_ms(
        fn_call: Any, *, reps: int = DEFAULT_MS_REPS, use_blocker: bool = True, pre_evict: Any = None,
    ) -> list[float]:
        for i in range(WARMUP_MS_ITERS):
            fn_call(i)
        torch.npu.synchronize()
        out: list[float] = []
        step = 1 if pre_evict is not None else (CHUNK_REPEATS if use_blocker else DMA_CHUNK_REPEATS)
        for start in range(0, reps, step):
            k = min(step, reps - start)
            evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(k)]
            if pre_evict is not None:
                pre_evict(start)
            if use_blocker:
                for _ in range(blocker_reps):
                    _ = torch.matmul(wa, wb)
            if pre_evict is None:
                for s_off in range(DMA_SETTLE_ITERS):
                    fn_call(start + s_off)
            for off, (se, ee) in enumerate(evs):
                se.record(); fn_call(start + (0 if pre_evict is not None else DMA_SETTLE_ITERS) + off); ee.record()
            torch.npu.synchronize()
            out.extend(float(se.elapsed_time(ee)) for se, ee in evs)
        return out

    in_buf = torch.randn((aiv * IN_BUF_STRIDE,), dtype=torch.float32, device=dev) * IN_SCALE
    mx_buf = torch.full((MX_SCALE_BUF_BYTES,), E8M0_UNITY_BYTE, dtype=torch.uint8, device=dev)
    out_buf = torch.zeros((max(aiv, aic) * OUT_BUF_STRIDE,), dtype=torch.float32, device=dev)
    in_p = ctypes.c_void_p(in_buf.data_ptr())
    mx_p = ctypes.c_void_p(mx_buf.data_ptr())
    out_p = ctypes.c_void_p(out_buf.data_ptr())
    st_p = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
    res: dict[str, Any] = {"subtask": sub, "samples": {}, "metadata": {}, "errors": {}}
    with tempfile.TemporaryDirectory(prefix="mindie_l0_ac_") as td:
        tmp = Path(td)
        if sub == "p17_launch":
            _run_p17_launch_subtask(torch, dev, repeats, res)
        elif sub == "dma_hw":
            lib, err = _compile_so(DMA_HW_CPP, arch, cann_root, tmp, sub)
            if err or lib is None:
                res["errors"]["compile"] = err
                return res
            _lock_dvfs()
            _run_dma_hw_subtask(
                torch, dev, lib, out_p, st_p,
                aic=aic, aiv=aiv, l2_bytes=l2_bytes, repeats=repeats, time_ms=_time_ms, res=res,
            )
        else:
            _run_kernel_subtask(
                sub, torch=torch, out_buf=out_buf, arch=arch, cann_root=cann_root, tmp=tmp,
                in_p=in_p, mx_p=mx_p, out_p=out_p, st_p=st_p, aic=aic, aiv=aiv, repeats=repeats,
                time_diff=_time_diff, time_ms=_time_ms, lock_dvfs=_lock_dvfs, res=res,
            )
    return res


def run_ascendc_suite(
    *, device_id: int, arch_preset: str, aic_count: int, aiv_count: int,
    l2_bytes: int, repeats: int = DEFAULT_REPEATS, selected_primitives: set[str] | None = None,
) -> dict[str, Any]:
    """Run native AscendC subtasks in isolated child processes and merge results."""
    arch = ARCH35_BISHENG if arch_preset == ARCH35_PRESET else ARCH22_BISHENG
    subtask_map = [
        ("vec_p12", {"P06", "P07", "P08", "P09", "P12", "P14"}),
        ("p13_aiv", {"P13"}), ("p13_aic", {"P13"}), ("p13_mode2", {"P13"}),
        ("p02_p03", {"P02", "P03"}), ("p06_mmad", {"P06"}),
        ("p10_sort", {"P10"}), ("p17_launch", {"P15"}), ("dma_hw", {"P01", "P05"}),
    ]
    if arch == ARCH35_BISHENG:
        subtask_map.extend([("p13_mode4", {"P13"}), ("p11_simt", {"P11"})])
    subtasks = [
        sub for sub, tags in subtask_map
        if selected_primitives is None or bool(tags & selected_primitives)
    ]
    merged: dict[str, Any] = {"arch": arch, "samples": {}, "metadata": {}, "errors": {}}
    env = dict(os.environ)
    env["TASK_QUEUE_ENABLE"] = "0"
    for sub in subtasks:
        timeout_s = SYNC_SUBTASK_TIMEOUT_S if sub.startswith("p13_") else DEFAULT_SUBTASK_TIMEOUT_S
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--subtask", sub, "--device-id", str(device_id), "--arch", arch,
            "--aic", str(aic_count), "--aiv", str(aiv_count),
            "--l2-bytes", str(l2_bytes), "--repeats", str(repeats),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env, check=False)
        except subprocess.TimeoutExpired:
            merged["errors"][sub] = f"timeout_{timeout_s}s"
            continue
        found = False
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(SUB_MARKER):
                data = json.loads(line[len(SUB_MARKER):])
                merged["samples"].update(data.get("samples", {}))
                merged["metadata"].update(data.get("metadata", {}))
                if data.get("errors"):
                    merged["errors"][sub] = data["errors"]
                found = True
                break
        if not found:
            merged["errors"][sub] = f"rc={proc.returncode}"
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AscendC JIT L0 microkernel subtask runner")
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--arch", default=ARCH22_BISHENG)
    ap.add_argument("--aic", type=int, required=True)
    ap.add_argument("--aiv", type=int, required=True)
    ap.add_argument("--l2-bytes", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = ap.parse_args(argv)
    out = _run_subtask(
        args.subtask, device_id=args.device_id, arch=args.arch,
        aic=args.aic, aiv=args.aiv, l2_bytes=args.l2_bytes, repeats=args.repeats,
    )
    print(SUB_MARKER + json.dumps(out, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
