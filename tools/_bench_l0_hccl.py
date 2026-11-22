#!/usr/bin/env python3
"""Two-rank HCCL collective and duplex P2P micro-benchmark probe worker.

[Role] Plane-1 offline multi-rank probe worker executing lockstep 2-rank HCCL
  AllReduce (P18) and duplex P2P (P19 diagnostic) with 30-GEMM DVFS lock,
  paired eager/StreamBlocker timing, per-trial rank-max aggregation, and
  explicit collective numerical verification.
[Invariants] Strictly offline Plane-1 tool outside ``mindie_sim/``. Bounds HBM
  <= 1 GiB/rank, resets rank-coded inputs outside timing to prevent BF16
  overflow, gathers both ranks before taking per-trial max, and fails loudly
  on verification mismatch.
[Timeouts & Locks] Ephemeral loopback TCP store port; zero persistent files
  or external locks.
[Side-Effect Policy] Spawns two short-lived worker processes freed on join.
"""
from __future__ import annotations

import gc
import math
import os
import socket
from typing import Any

MIB = 1024 * 1024
BF16_BYTES = 2
WARMUP_GEMM_DIM = 4096
WARMUP_GEMM_ITERS = 30
BLOCKER_TARGET_MS = 3.0
MIN_SINGLE_GEMM_MS = 0.05
HCCL_BIG_MIB = 64
HCCL_ARCH35_P2P_MIB = 4
HCCL_SMALL_ELEMS = 2048
HCCL_WARMUP_ITERS = 32
HCCL_BIG_INNER = 8
HCCL_SMALL_INNER = 32
HCCL_BLOCKER_GEMMS = 6
HCCL_WORLD_SIZE = 2
HCCL_REPEATS = 24
HCCL_SMALL_FILL_SCALE = 0.01
HCCL_EXPECTED_AR_SUM = 0.03
HCCL_VERIFY_ATOL = 1e-2
MS_PER_S = 1e-3
NS_PER_MS = 1e6


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _hccl_two_rank_worker(
    rank: int, devs: list[int], port: int, repeats: int, arch_preset: str, out_q: Any,
) -> None:
    os.environ.pop("TASK_QUEUE_ENABLE", None)
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dev_id = int(devs[rank])
    torch.npu.set_device(dev_id)
    dev = torch.device(f"npu:{dev_id}")
    dist.init_process_group("hccl", rank=rank, world_size=HCCL_WORLD_SIZE)

    wa = torch.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    wb = torch.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    for _ in range(WARMUP_GEMM_ITERS):
        _ = torch.matmul(wa, wb)
    torch.npu.synchronize()
    s0, e0 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s0.record()
    for _ in range(HCCL_BIG_INNER):
        _ = torch.matmul(wa, wb)
    e0.record()
    torch.npu.synchronize()
    gemm_ms = max(MIN_SINGLE_GEMM_MS, float(s0.elapsed_time(e0)) / float(HCCL_BIG_INNER))
    blk_reps = max(HCCL_BLOCKER_GEMMS, math.ceil(BLOCKER_TARGET_MS / gemm_ms))

    big_elems = (HCCL_BIG_MIB * MIB) // BF16_BYTES
    big_bytes = float(HCCL_BIG_MIB * MIB)
    p2p_mib = HCCL_ARCH35_P2P_MIB if arch_preset == "npu_arch_3510" else HCCL_BIG_MIB
    p2p_elems = (p2p_mib * MIB) // BF16_BYTES
    p2p_bytes = float(p2p_mib * MIB)
    big_buf = torch.full((big_elems,), float(rank + 1), dtype=torch.bfloat16, device=dev)
    p2p_buf = torch.full((p2p_elems,), float(rank + 1), dtype=torch.bfloat16, device=dev)
    sml_fill = float(rank + 1) * HCCL_SMALL_FILL_SCALE
    sml_ar = torch.full((HCCL_SMALL_ELEMS,), sml_fill, dtype=torch.bfloat16, device=dev)
    sml_p2p = torch.full((HCCL_SMALL_ELEMS,), float(rank + 1), dtype=torch.bfloat16, device=dev)
    rcv_big = torch.empty_like(p2p_buf)
    rcv_sml = torch.empty_like(sml_p2p)
    peer = 1 - rank

    def _p2p(sb: Any, rb: Any) -> None:
        ops = (
            [dist.P2POp(dist.isend, sb, peer), dist.P2POp(dist.irecv, rb, peer)]
            if rank == 0
            else [dist.P2POp(dist.irecv, rb, peer), dist.P2POp(dist.isend, sb, peer)]
        )
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    for _ in range(HCCL_WARMUP_ITERS):
        big_buf.fill_(float(rank + 1)); dist.all_reduce(big_buf)
        sml_ar.fill_(sml_fill); dist.all_reduce(sml_ar)
        _p2p(p2p_buf, rcv_big)
        _p2p(sml_p2p, rcv_sml)
    torch.npu.synchronize()
    ar_ok = abs(float(sml_ar[0].item()) - HCCL_EXPECTED_AR_SUM) <= HCCL_VERIFY_ATOL
    p2p_ok = abs(float(rcv_sml[0].item()) - float(peer + 1)) <= HCCL_VERIFY_ATOL
    dist.barrier()

    def _measure_trials(fn: Any, *, inner: int, blocker: bool, reset_fn: Any = None) -> list[float]:
        ms_list: list[float] = []
        gc.collect()
        was_gc = gc.isenabled()
        gc.disable()
        try:
            for _ in range(repeats):
                if reset_fn is not None:
                    reset_fn()
                torch.npu.synchronize()
                dist.barrier()
                if blocker:
                    for _ in range(blk_reps):
                        _ = torch.matmul(wa, wb)
                fn()
                s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                s.record()
                for _ in range(inner):
                    fn()
                e.record()
                torch.npu.synchronize()
                val = float(s.elapsed_time(e)) / float(inner)
                ms_list.append(val)
        finally:
            if was_gc:
                gc.enable()
        return ms_list

    ar_big_ms = _measure_trials(lambda: dist.all_reduce(big_buf), inner=HCCL_BIG_INNER, blocker=False, reset_fn=lambda: big_buf.fill_(float(rank + 1)))
    ar_sml_ms = _measure_trials(lambda: dist.all_reduce(sml_ar), inner=HCCL_SMALL_INNER, blocker=False, reset_fn=lambda: sml_ar.fill_(sml_fill))
    ar_sml_blk_ms = _measure_trials(lambda: dist.all_reduce(sml_ar), inner=HCCL_SMALL_INNER, blocker=True, reset_fn=lambda: sml_ar.fill_(sml_fill))
    p2p_big_ms = _measure_trials(lambda: _p2p(p2p_buf, rcv_big), inner=HCCL_BIG_INNER, blocker=False)
    p2p_sml_ms = _measure_trials(lambda: _p2p(sml_p2p, rcv_sml), inner=HCCL_SMALL_INNER, blocker=False)
    p2p_sml_blk_ms = _measure_trials(lambda: _p2p(sml_p2p, rcv_sml), inner=HCCL_SMALL_INNER, blocker=True)
    finite_ok = bool(torch.isfinite(sml_ar[0]).item() and torch.isfinite(big_buf[0]).item())
    out_q.put({
        "rank": rank, "verified": bool(ar_ok and p2p_ok and finite_ok),
        "big_bytes": big_bytes, "p2p_bytes": p2p_bytes,
        "ar_big_ms": ar_big_ms, "ar_sml_ms": ar_sml_ms, "ar_sml_blk_ms": ar_sml_blk_ms,
        "p2p_big_ms": p2p_big_ms, "p2p_sml_ms": p2p_sml_ms, "p2p_sml_blk_ms": p2p_sml_blk_ms,
    })
    dist.destroy_process_group()


def run_hccl_two_rank(device_id: int, peer_device_id: int, repeats: int, arch_preset: str) -> dict[str, Any]:
    """Run lockstep 2-rank HCCL AllReduce and duplex P2P probes with rank-max aggregation."""
    import torch.multiprocessing as mp

    port = _free_loopback_port()
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    mp.spawn(
        _hccl_two_rank_worker,
        args=([device_id, peer_device_id], port, min(repeats, HCCL_REPEATS), arch_preset, out_q),
        nprocs=HCCL_WORLD_SIZE,
        join=True,
    )
    ranks = sorted((dict(out_q.get()) for _ in range(HCCL_WORLD_SIZE)), key=lambda d: int(d["rank"]))
    if not all(bool(r.get("verified")) for r in ranks):
        raise RuntimeError("hccl_collective_result_verify_failed")

    def _max_trials(key: str) -> list[float]:
        r0, r1 = ranks[0][key], ranks[1][key]
        if len(r0) != len(r1):
            raise RuntimeError(f"hccl_trial_count_mismatch:{key}:{len(r0)}!={len(r1)}")
        return [
            (max(a, b) if (a > 0.0 and b > 0.0) else min(a, b))
            for a, b in zip(r0, r1, strict=True)
        ]

    def _bps_trials(num_bytes: float, key: str) -> list[float]:
        return [(num_bytes / (t * MS_PER_S) if t > 0.0 else -1.0) for t in _max_trials(key)]

    big_b, p2p_b = float(ranks[0]["big_bytes"]), float(ranks[0]["p2p_bytes"])
    return {
        "P18_hccl_collective_band": {
            "link_bytes_per_s": _bps_trials(big_b, "ar_big_ms"),
            "startup_ns": [t * NS_PER_MS for t in _max_trials("ar_sml_ms")],
            "startup_blocked_ns": [t * NS_PER_MS for t in _max_trials("ar_sml_blk_ms")],
            "measured_world_size": HCCL_WORLD_SIZE, "measured_collective": "all_reduce",
            "rank_max_aggregated": True,
        },
        "P19_direct_peer_window": {
            "hccl_duplex_bytes_per_s": _bps_trials(p2p_b, "p2p_big_ms"),
            "hccl_duplex_4k_ns": [t * NS_PER_MS for t in _max_trials("p2p_sml_ms")],
            "hccl_duplex_4k_blocked_ns": [t * NS_PER_MS for t in _max_trials("p2p_sml_blk_ms")],
            "rank_max_aggregated": True,
        },
    }
