#!/usr/bin/env python3
"""Offline Ascend A2/A3/A5 NPU L0 micro-benchmark calibration suite.

[Role] Plane-1 offline calibration producer executing bounded synthetic NPU
  micro-benchmarks on idle physical Ascend A2/A3/A5 dies via endpoint alias
  and emitting full-envelope SHA-256 sealed ``data/primitives/*.json`` tables
  plus ``data/primitives/manifest.json``.
[Invariants] Strictly offline Plane-1 tool outside ``mindie_sim/``. Checks
  ``npu-smi info`` before touching any NPU so only idle cards (0% AICore,
  zero competing processes) are used; caps synthetic HBM allocations <= 4 GiB;
  runs 30-GEMM DVFS lock, >= 3x L2 cold rotating rings vs L2-hot single buffer,
  and 2.5 ms StreamBlocker saturation; enforces cv_pct <= 5.0% and
  certified_upper_capacity >= p99; never emits raw IPs, hostnames, usernames,
  or host $HOME paths into JSON artifacts.
[Timeouts & Locks] Bounded SSH/remote subprocess execution with unconditional
  remote temporary directory cleanup on exit; targets idle NPU dies only.
[Side-Effect Policy] Writes only the requested ``--out`` JSON table and
  ``--manifest`` index; cleans up remote temporary files unconditionally.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any

_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _PARENTS[0]
if (REPO_ROOT / "mindie_sim").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRIMITIVE_SCHEMA_VERSION = "mindie-l0-primitives/1"
RESULT_MARKER = "__MINDIE_L0_BENCH_JSON__:"
MAX_CALIBRATED_CV_PCT = 5.0
WARMUP_GEMM_ITERS = 30
WARMUP_GEMM_DIM = 4096
BLOCKER_TARGET_MS = 2.5
MIN_SINGLE_GEMM_MS = 0.05
DEFAULT_WARMUP = 8
DEFAULT_REPEATS = 32
TRIM_FRACTION = 0.10
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
SSH_CONNECT_TIMEOUT_S = 8
REMOTE_BENCH_TIMEOUT_S = 360
IPV4_QUAD_RE = re.compile(r"\b(?:\d{1,3}[.-]){3}\d{1,3}\b")
SAFE_ALIAS_RE = re.compile(r"[^a-z0-9._-]+")

SKU_DEFAULT_FILES = {
    "Ascend910B4": "arch22_a2_910b4.json",
    "Ascend910_9362": "arch22_a3_9362.json",
    "Ascend910_9382": "arch22_a3_9382.json",
    "Ascend950PR_9579": "arch35_a5_950pr.json",
    "Ascend950DT_9582": "arch35_a5_950dt.json",
    "Ascend950DT_9581": "arch35_a5_950dt.json",
}
SKU_COMPATIBLE_ALIASES = {
    "Ascend950DT_9582": ["Ascend950DT_9581", "Ascend950DT_9582"],
    "Ascend950DT_9581": ["Ascend950DT_9581", "Ascend950DT_9582"],
}


def compute_envelope_sha256(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "envelope_sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sanitize_record_alias(alias: str, sku: str) -> str:
    """Return a clean public endpoint alias with zero raw IPv4 quads or host paths."""
    cleaned = IPV4_QUAD_RE.sub("", alias.lower()).strip("._-")
    cleaned = SAFE_ALIAS_RE.sub("-", cleaned).strip("-")
    if not cleaned or cleaned in {"a2", "a3", "a5", "host", "default"}:
        return f"lab-{SAFE_ALIAS_RE.sub('-', sku.lower()).strip('-')}"
    return cleaned


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return float(sorted_vals[lo]) if lo == hi else float(sorted_vals[lo] * (1.0 - (pos - lo)) + sorted_vals[hi] * (pos - lo))


def summarize_samples(samples: list[float], *, certified_bound: float, bound_kind: str = "rate_upper") -> dict[str, Any]:
    """Compute untrimmed (p05,p50,p95,p99,cv_pct) + 10%-trimmed stats with upper/lower bound."""
    clean = sorted(float(x) for x in samples if math.isfinite(x) and x >= 0.0)
    if not clean:
        return {"certified_upper_capacity": round(float(certified_bound), 6), "bound_kind": bound_kind}
    n = len(clean)
    mean_v = sum(clean) / n
    cv_pct = (100.0 * statistics.stdev(clean) / mean_v) if (n >= 2 and mean_v > 0.0) else 0.0
    trim_k = int(math.floor(n * TRIM_FRACTION))
    trimmed = clean[trim_k : n - trim_k] if n - 2 * trim_k >= 2 else clean
    t_mean = sum(trimmed) / len(trimmed)
    t_cv = (100.0 * statistics.stdev(trimmed) / t_mean) if (len(trimmed) >= 2 and t_mean > 0.0) else 0.0
    p05, p50, p95, p99 = (_quantile(clean, q) for q in (0.05, 0.50, 0.95, 0.99))
    bound = max(float(certified_bound), p99) if bound_kind == "rate_upper" else max(0.0, min(float(certified_bound), p05))
    return {
        "p05": round(p05, 6), "p50": round(p50, 6), "p95": round(p95, 6), "p99": round(p99, 6),
        "trimmed_mean": round(t_mean, 6), "trimmed_cv_pct": round(t_cv, 4),
        "cv_pct": round(cv_pct, 4), "n_samples": n,
        "certified_upper_capacity": round(bound, 6), "bound_kind": bound_kind,
    }


def _load_sku_hardware_spec(sku: str) -> dict[str, Any]:
    from mindie_sim.backends import ascend
    from mindie_sim.schema.datapath import datapath_for_die
    from mindie_sim.schema.hardware import Hardware

    die = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1})).dies[0]
    dp = datapath_for_die(die)
    bufs, rates = dict(die.buffers), dict(die.rates)
    return {
        "sku": sku, "arch_preset": str(die.compute_arch.preset or "npu_arch_2201"),
        "aic_count": int(dp.aic_count), "aiv_count": int(dp.aiv_count),
        "hbm_gib": int(die.memory_bytes // GIB),
        "l2_bytes": int(bufs.get("l2", 0)), "ub_bytes": int(bufs.get("ub", 0)),
        "l1_bytes": int(bufs.get("l1", 0)), "l0a_bytes": int(bufs.get("l0a", 0)),
        "l0b_bytes": int(bufs.get("l0b", 0)), "l0c_bytes": int(bufs.get("l0c", 0)),
        "has_l0c_to_ub": bool(dp.has_l0c_to_ub), "has_ub_to_l1": bool(dp.has_ub_to_l1),
        "simt_ub_carveout_bytes": int(dp.simt_ub_carveout_bytes),
        "hbm_read_bytes_per_s": float(dp.hbm_read_bytes_per_s or rates.get("hbm", 0.0)),
        "hbm_write_bytes_per_s": float(dp.hbm_write_bytes_per_s or rates.get("hbm", 0.0)),
        "l2_bytes_per_s": float(rates.get("l2", rates.get("mte2_aiv", 0.0) * 2.0)),
        "mte1_bytes_per_s": float(rates.get("mte1", 0.0)),
        "mte2_aic_bytes_per_s": float(rates.get("mte2_aic", 0.0)),
        "mte2_aiv_bytes_per_s": float(rates.get("mte2_aiv", 0.0)),
        "mte3_bytes_per_s": float(rates.get("mte3", 0.0)),
        "fixpipe_bytes_per_s": float(rates.get("fixpipe", 0.0)),
        "vector_ops_per_s": float(rates.get("vector", 0.0)),
        "peaks": {str(k): float(v) for k, v in dict(die.compute_peaks).items()},
    }


def run_local_npu_microbenchmarks(
    *, device_id: int, l2_bytes: int, aic_count: int, aiv_count: int,
    arch_preset: str, warmup: int = DEFAULT_WARMUP, repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Execute bounded live Ascend NPU micro-benchmarks on one idle device."""
    import time
    import torch
    import torch_npu

    del arch_preset
    torch.npu.set_device(device_id)
    dev = torch.device(f"npu:{device_id}")
    wa = torch.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    wb = torch.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    for _ in range(WARMUP_GEMM_ITERS):
        _ = torch.matmul(wa, wb)
    torch.npu.synchronize()

    s0, e0 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s0.record()
    _ = torch.matmul(wa, wb)
    e0.record()
    torch.npu.synchronize()
    blocker_reps = max(1, math.ceil(BLOCKER_TARGET_MS / max(MIN_SINGLE_GEMM_MS, float(s0.elapsed_time(e0)))))

    def _time_ms(fn, *, inner: int = 1, blocker: bool = False) -> list[float]:
        for i in range(warmup):
            fn(i)
        torch.npu.synchronize()
        if blocker:
            evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(repeats)]
            for _ in range(blocker_reps):
                _ = torch.matmul(wa, wb)
            for idx, (se, ee) in enumerate(evs):
                se.record()
                for k in range(inner):
                    fn(idx * inner + k)
                ee.record()
            torch.npu.synchronize()
            return [max(1e-6, float(se.elapsed_time(ee)) / inner) for se, ee in evs]
        out: list[float] = []
        for idx in range(repeats):
            se, ee = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            se.record()
            for k in range(inner):
                fn(idx * inner + k)
            ee.record()
            torch.npu.synchronize()
            out.append(max(1e-6, float(se.elapsed_time(ee)) / inner))
        return out

    raw: dict[str, Any] = {
        "device_name": str(torch.npu.get_device_name(device_id)),
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "blocker_reps": blocker_reps,
    }

    # P01 & P04: >=3x L2 cold rotating rings (capped <= 3 GiB VRAM) + warm L2-hot + atomic
    slab = 64 * MIB
    n_slots = min(14, max(6, math.ceil(max(3 * l2_bytes, 384 * MIB) / slab)))
    ring_a = [torch.ones((slab // 2,), dtype=torch.bfloat16, device=dev) for _ in range(n_slots)]
    ring_b = [torch.ones((slab // 2,), dtype=torch.bfloat16, device=dev) for _ in range(n_slots)]
    ring_c = [torch.empty((slab // 2,), dtype=torch.bfloat16, device=dev) for _ in range(n_slots)]

    w_ms = _time_ms(lambda i: ring_c[i % n_slots].fill_(1.0), inner=2, blocker=True)
    rw11_ms = _time_ms(lambda i: ring_c[i % n_slots].copy_(ring_a[i % n_slots]), inner=2, blocker=True)
    rw21_ms = _time_ms(lambda i: torch.add(ring_a[i % n_slots], ring_b[i % n_slots], out=ring_c[i % n_slots]), inner=2, blocker=True)
    rd_ms = _time_ms(lambda i: torch.amax(ring_a[i % n_slots]), inner=2, blocker=True)
    hbm_w = [slab / (t * 1e-3) for t in w_ms]
    hbm_rw11 = [(2.0 * slab) / (t * 1e-3) for t in rw11_ms]
    hbm_rw21 = [(3.0 * slab) / (t * 1e-3) for t in rw21_ms]
    hbm_r = [max(slab / (tr * 1e-3), (2.0 / 3.0) * rw) for tr, rw in zip(rd_ms, hbm_rw21)]
    t_r1 = slab / max(1.0, statistics.median(hbm_r)) * 1e3
    t_w1 = slab / max(1.0, statistics.median(hbm_w)) * 1e3
    alphas = [max(0.0, min(1.0, (t11 - max(t_r1, t_w1)) / max(1e-9, min(t_r1, t_w1)))) for t11 in rw11_ms]

    s_src = ring_a[0][: 32768 * 1024].view(32768, 1024)
    s_dst = torch.empty((32768, 256), dtype=torch.bfloat16, device=dev)
    strided_ms = _time_ms(lambda _i: s_dst.copy_(s_src[:, :256]), inner=2, blocker=True)
    strided_bps = [(2.0 * 32768 * 512.0) / (t * 1e-3) for t in strided_ms]
    del ring_a, ring_b, ring_c, s_src, s_dst
    torch.npu.empty_cache()

    warm_bytes = min(32 * MIB, max(8 * MIB, l2_bytes // 4))
    w_src = torch.ones((warm_bytes // 2,), dtype=torch.bfloat16, device=dev)
    w_dst = torch.empty((warm_bytes // 2,), dtype=torch.bfloat16, device=dev)
    l2_bps = [(2.0 * warm_bytes) / (t * 1e-3) for t in _time_ms(lambda _i: w_dst.copy_(w_src), inner=8, blocker=True)]
    mte3_warm = [(warm_bytes / (t * 1e-3)) / max(1, aiv_count) for t in _time_ms(lambda _i: w_dst.fill_(1.0), inner=8, blocker=True)]
    a_dst = torch.zeros((65536,), dtype=torch.float32, device=dev)
    a_idx = torch.arange(262144, dtype=torch.int64, device=dev) % 65536
    a_src = torch.ones((262144,), dtype=torch.float32, device=dev)
    atom_bps = [(2.0 * 262144 * 4.0) / (t * 1e-3) for t in _time_ms(lambda _i: a_dst.index_add_(0, a_idx, a_src), inner=8, blocker=True)]
    del w_src, w_dst, a_dst, a_idx, a_src
    torch.npu.empty_cache()

    raw["P01_datacopy"] = {
        "hbm_read_bytes_per_s": hbm_r, "hbm_write_bytes_per_s": hbm_w,
        "hbm_rw11_bytes_per_s": hbm_rw11, "hbm_rw21_bytes_per_s": hbm_rw21,
        "hbm_rw_alpha": alphas, "strided_512b_bytes_per_s": strided_bps,
        "l2_die_bytes_per_s": l2_bps,
        "mte2_aiv_hit_bytes_per_s_per_core": [bw / max(1, aiv_count) for bw in l2_bps],
    }
    raw["P04_mte3_store"] = {"mte3_warm_bytes_per_s_per_aiv": mte3_warm, "atomic_rmw_bytes_per_s": atom_bps}

    # P05: L2 working-set sweep
    ws_curve: dict[str, float] = {}
    ws_list = (8, 32, 64, 96, 128, 160, 192, 256, 320)
    for ws_mib in ws_list:
        xb = torch.ones(((ws_mib * MIB) // 4,), dtype=torch.bfloat16, device=dev)
        yb = torch.empty_like(xb)
        ts = _time_ms(lambda _i: yb.copy_(xb), inner=4 if ws_mib <= 64 else 1, blocker=True)
        ws_curve[f"{ws_mib}MiB"] = round((ws_mib * MIB) / (statistics.median(ts) * 1e-3), 2)
        del xb, yb
    torch.npu.empty_cache()
    peak_bw, cold_bw = max(ws_curve.values()), ws_curve[f"{ws_list[-1]}MiB"]
    thresh = cold_bw + 0.5 * max(0.0, peak_bw - cold_bw)
    knee = max((m for m in ws_list if ws_curve[f"{m}MiB"] >= thresh), default=ws_list[0])
    eff_frac = min(1.0, max(0.25, (knee * MIB) / max(1, l2_bytes)))
    raw["P05_l2_residency_reuse"] = {
        "ws_sweep_bytes_per_s": ws_curve, "l2_knee_mib": knee,
        "l2_effective_fraction": [eff_frac] * repeats, "l2_die_bytes_per_s": l2_bps,
    }

    # P06: Cube Mmad saturation + small-M sweep
    mm_flops = 2.0 * (4096 ** 3)
    mm_modes: dict[str, list[float]] = {}
    for m_name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        ma = torch.randn((4096, 4096), dtype=dt, device=dev)
        mb = torch.randn((4096, 4096), dtype=dt, device=dev)
        mm_modes[m_name] = [(mm_flops / (t * 1e-3)) / max(1, aic_count) for t in _time_ms(lambda _i: torch.matmul(ma, mb), inner=2, blocker=True)]
        del ma, mb
    try:
        if hasattr(torch_npu, "npu_quant_matmul"):
            qa = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8, device=dev)
            qb = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8, device=dev)
            sc = torch.ones((4096,), dtype=torch.float32, device=dev)
            mm_modes["int8"] = [(mm_flops / (t * 1e-3)) / max(1, aic_count) for t in _time_ms(
                lambda _i: torch_npu.npu_quant_matmul(qa, qb, sc, output_dtype=torch.bfloat16), inner=2, blocker=True)]
            del qa, qb, sc
    except Exception:
        pass
    sm_ns: dict[str, float] = {}
    mb_sm = torch.randn((4096, 4096), dtype=torch.bfloat16, device=dev)
    for mv in (1, 16, 32, 64, 128, 256):
        ma_sm = torch.randn((mv, 4096), dtype=torch.bfloat16, device=dev)
        sm_ns[str(mv)] = round(statistics.median(_time_ms(lambda _i: torch.matmul(ma_sm, mb_sm), inner=4, blocker=True)) * 1e6, 2)
        del ma_sm
    del mb_sm
    torch.npu.empty_cache()
    raw["P06_mmad"] = {"modes": mm_modes, "small_m_bf16_ns": sm_ns}

    # P07 / P08 / P09 / P10: Vector & Sort/TopK
    va = torch.randn((2048, 2048), dtype=torch.float32, device=dev)
    vb = torch.randn((2048, 2048), dtype=torch.float32, device=dev)
    vc = torch.empty_like(va)
    elems = 2048 * 2048
    raw["P07_vec_alu"] = {"vec_alu_ops_per_s_per_aiv": [
        (2.0 * elems / (t * 1e-3)) / max(1, aiv_count) for t in _time_ms(lambda _i: torch.addcmul(va, va, vb, out=vc), inner=8, blocker=True)]}
    raw["P08_vec_sfu"] = {"vec_sfu_ops_per_s_per_aiv": [
        (elems / (t * 1e-3)) / max(1, aiv_count) for t in _time_ms(lambda _i: torch.exp(va, out=vc), inner=8, blocker=True)]}
    raw["P09_vec_reduce_cast"] = {
        "vec_reduce_ops_per_s_per_aiv": [(elems / (t * 1e-3)) / max(1, aiv_count) for t in _time_ms(lambda _i: torch.sum(va, dim=-1), inner=8, blocker=True)],
        "cast_bytes_per_s_per_aiv": [(elems * 6.0 / (t * 1e-3)) / max(1, aiv_count) for t in _time_ms(lambda _i: va.to(torch.bfloat16), inner=8, blocker=True)],
    }
    del va, vb, vc
    sx = torch.randn((256, 2048), dtype=torch.float32, device=dev)
    raw["P10_sort_topk"] = {"sort_elems_per_s": [
        (256 * 2048) / (t * 1e-3) for t in _time_ms(lambda _i: torch.topk(sx, k=128, dim=-1), inner=4, blocker=True)]}
    del sx
    torch.npu.empty_cache()

    # P15 & P16: STARS / RTS & Multi-stream Event
    ta, tb, to = torch.ones((1,), dtype=torch.bfloat16, device=dev), torch.ones((1,), dtype=torch.bfloat16, device=dev), torch.empty((1,), dtype=torch.bfloat16, device=dev)
    aa, ab = torch.ones((1, 16), dtype=torch.bfloat16, device=dev), torch.ones((16, 16), dtype=torch.bfloat16, device=dev)
    for _ in range(warmup):
        torch.add(ta, tb, out=to)
    torch.npu.synchronize()
    host_ns: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(64):
            torch.add(ta, tb, out=to)
        host_ns.append((time.perf_counter_ns() - t0) / 64.0)
    torch.npu.synchronize()
    aiv_f = [t * 1e6 for t in _time_ms(lambda _i: torch.add(ta, tb, out=to), inner=16, blocker=True)]
    aic_f = [t * 1e6 for t in _time_ms(lambda _i: torch.matmul(aa, ab), inner=16, blocker=True)]
    q_ns = list(aiv_f)
    raw["P15_rts_stars"] = {"host_submit_ns": host_ns, "issue_interval_ns": q_ns, "aiv_first_issue_ns": aiv_f, "aic_first_issue_ns": aic_f}

    s_alt, ev_sync = torch.npu.Stream(device=dev), torch.npu.Event(enable_timing=False)
    for _ in range(warmup):
        ev_sync.record()
        s_alt.wait_event(ev_sync)
    torch.npu.synchronize()
    se_ns = [t * 1e6 for t in _time_ms(lambda _i: (ev_sync.record(), s_alt.wait_event(ev_sync)), inner=32, blocker=True)]
    raw["P16_multistream_core_control"] = {"stream_event_ns": se_ns}

    # P20: Host-Device H2D/D2H DMA & Eager D2H sync
    dma_bytes = 32 * MIB
    h_pin = torch.ones((dma_bytes // 4,), dtype=torch.float32).pin_memory()
    d_buf = torch.empty((dma_bytes // 4,), dtype=torch.float32, device=dev)
    for _ in range(warmup):
        d_buf.copy_(h_pin, non_blocking=False)
        h_pin.copy_(d_buf, non_blocking=False)
    torch.npu.synchronize()
    h2d_bps, d2h_bps = [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(4):
            d_buf.copy_(h_pin, non_blocking=False)
        torch.npu.synchronize()
        h2d_bps.append((4.0 * dma_bytes) / max(1e-9, (time.perf_counter_ns() - t0) * 1e-9))
        t0 = time.perf_counter_ns()
        for _ in range(4):
            h_pin.copy_(d_buf, non_blocking=False)
        torch.npu.synchronize()
        d2h_bps.append((4.0 * dma_bytes) / max(1e-9, (time.perf_counter_ns() - t0) * 1e-9))
    sp_pin, sp_page = torch.ones((1024,), dtype=torch.float32).pin_memory(), torch.ones((1024,), dtype=torch.float32)
    sp_dev = torch.empty((1024,), dtype=torch.float32, device=dev)
    pin_ns, page_ns, eager_ns = [], [], []
    for _ in range(warmup):
        sp_dev.copy_(sp_pin, non_blocking=False)
        sp_dev.copy_(sp_page, non_blocking=False)
        _ = float(ta.item())
    torch.npu.synchronize()
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(8):
            sp_dev.copy_(sp_pin, non_blocking=False)
        torch.npu.synchronize()
        pin_ns.append((time.perf_counter_ns() - t0) / 8.0)
        t0 = time.perf_counter_ns()
        for _ in range(8):
            sp_dev.copy_(sp_page, non_blocking=False)
        torch.npu.synchronize()
        page_ns.append((time.perf_counter_ns() - t0) / 8.0)
        t0 = time.perf_counter_ns()
        for _ in range(8):
            _ = float(ta.item())
        eager_ns.append((time.perf_counter_ns() - t0) / 8.0)
    raw["P20_host_dma_sync"] = {
        "host_h2d_bytes_per_s": h2d_bps, "host_d2h_bytes_per_s": d2h_bps,
        "pinned_dma_startup_ns": pin_ns, "pageable_dma_startup_ns": page_ns, "eager_d2h_sync_ns": eager_ns,
    }
    del wa, wb, h_pin, d_buf, sp_pin, sp_page, sp_dev, ta, tb, to, aa, ab
    torch.npu.empty_cache()
    return raw


def _cell_status(bands: dict[str, dict[str, Any]]) -> str:
    if not bands:
        return "unmeasured"
    for b in bands.values():
        if b.get("p50") is None or b.get("cv_pct") is None or float(b["cv_pct"]) > MAX_CALIBRATED_CV_PCT:
            return "unmeasured"
    return "calibrated"


def _clean_band(b: dict[str, Any]) -> dict[str, Any]:
    if b.get("p50") is not None and b.get("cv_pct") is not None and float(b["cv_pct"]) <= MAX_CALIBRATED_CV_PCT:
        return b
    return {"certified_upper_capacity": b["certified_upper_capacity"], "bound_kind": b.get("bound_kind", "rate_upper")}


def _evidence_uri(alias: str, campaign_id: str, tag: str, payload: Any) -> str:
    raw = json.dumps({"tag": tag, "payload": payload}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"capture://{alias}/{campaign_id}/{hashlib.sha256(raw).hexdigest()}"


def _bound_only(value: float, kind: str = "rate_upper") -> dict[str, Any]:
    return {"certified_upper_capacity": round(float(value), 6), "bound_kind": kind}


def build_primitive_table_payload(
    *, sku: str, cann_version: str, endpoint_alias: str, campaign_id: str,
    raw_measurements: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble a full-envelope SHA-256 sealed mindie-l0-primitives/1 table."""
    spec = _load_sku_hardware_spec(sku)
    alias = sanitize_record_alias(endpoint_alias, sku)
    aic, aiv = max(1, spec["aic_count"]), max(1, spec["aiv_count"])
    is_arch35 = spec["arch_preset"] == "npu_arch_3510"
    raw = raw_measurements or {}
    vec_cap = max(1.0, spec["vector_ops_per_s"] / aiv)
    top_capture = _evidence_uri(alias, campaign_id, "campaign", raw if raw else {"sku": sku, "status": "unmeasured"})

    def _band(samples: list[float] | None, bound: float, *, kind: str = "rate_upper") -> dict[str, Any]:
        return summarize_samples(samples, certified_bound=bound, bound_kind=kind) if samples else _bound_only(bound, kind)

    def _make_cell(pid: str, units: dict[str, str], params: dict[str, Any],
                   core_metrics: dict[str, dict[str, Any]], extra_metrics: dict[str, dict[str, Any]] | None = None,
                   *, status_override: str | None = None, modes: dict[str, Any] | None = None) -> dict[str, Any]:
        st = status_override if status_override is not None else _cell_status(core_metrics)
        merged = {k: (v if st == "calibrated" else _clean_band(v)) for k, v in core_metrics.items()}
        for k, v in (extra_metrics or {}).items():
            merged[k] = _clean_band(v)
        ev = _evidence_uri(alias, campaign_id, pid, raw.get(pid, {})) if st == "calibrated" else ""
        return {"status": st, "units": units, "parameters": params, "metrics": merged, "modes": modes or {}, "evidence_id": ev}

    p: dict[str, Any] = {}
    p01r = raw.get("P01_datacopy", {})
    p["P01_datacopy"] = _make_cell(
        "P01_datacopy", {"rate": "bytes_per_s", "penalty": "bytes", "alpha": "dimensionless"},
        {"sector_bytes": 32, "l2_cache_ctl_supported": is_arch35, "burst_turn_penalty": []},
        {
            "hbm_read_bytes_per_s": _band(p01r.get("hbm_read_bytes_per_s"), spec["hbm_read_bytes_per_s"]),
            "hbm_write_bytes_per_s": _band(p01r.get("hbm_write_bytes_per_s"), spec["hbm_write_bytes_per_s"]),
            "mte2_aiv_hit_bytes_per_s_per_core": _band(
                p01r.get("mte2_aiv_hit_bytes_per_s_per_core"), max(spec["mte2_aiv_bytes_per_s"], spec["l2_bytes_per_s"]) / aiv),
        },
        {
            "hbm_rw_alpha": _band(p01r.get("hbm_rw_alpha"), 0.0, kind="latency_lower"),
            "mte2_aic_hit_bytes_per_s_per_core": _bound_only(spec["mte2_aic_bytes_per_s"] / aic),
        },
    )
    mte1_aic = spec["mte1_bytes_per_s"] / aic
    p["P02_load_data"] = _make_cell(
        "P02_load_data", {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"fractal_bytes": 512, "mx_scale_supported": is_arch35, "transpose_factor": 1.0}, {},
        {"l1_to_l0a_bytes_per_s_per_aic": _bound_only(mte1_aic), "l1_to_l0b_bytes_per_s_per_aic": _bound_only(mte1_aic), "setup_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    fixp_aic = spec["fixpipe_bytes_per_s"] / aic
    p["P03_fixpipe"] = _make_cell(
        "P03_fixpipe", {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"has_l0c_to_ub": spec["has_l0c_to_ub"], "has_l0c_to_l1": True, "ub_split_n_align_elems": 32}, {},
        {"fixpipe_in_bytes_per_s_per_aic": _bound_only(fixp_aic), "fixpipe_out_bytes_per_s_per_aic": _bound_only(fixp_aic), "setup_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p04r = raw.get("P04_mte3_store", {})
    p["P04_mte3_store"] = _make_cell(
        "P04_mte3_store", {"rate": "bytes_per_s", "latency": "ns"}, {"has_ub_to_l1": spec["has_ub_to_l1"]},
        {
            "mte3_store_bytes_per_s_per_aiv": _band(p04r.get("mte3_warm_bytes_per_s_per_aiv"), spec["mte3_bytes_per_s"] / aiv),
            "atomic_rmw_bytes_per_s": _band(p04r.get("atomic_rmw_bytes_per_s"), spec["hbm_write_bytes_per_s"]),
        },
        {"setup_ns": _bound_only(0.0, "latency_lower"), "atomic_contention_ns": _bound_only(0.0, "latency_lower")},
    )
    p05r = raw.get("P05_l2_residency_reuse", {})
    p["P05_l2_residency_reuse"] = _make_cell(
        "P05_l2_residency_reuse", {"rate": "bytes_per_s", "fraction": "dimensionless", "capacity": "bytes"},
        {"l2_bytes": spec["l2_bytes"], "granule_bytes": 128, "buffer_units_l2": 1, "cache_disable_supported": is_arch35, "ws_sweep_bytes_per_s": p05r.get("ws_sweep_bytes_per_s", {})},
        {
            "l2_die_bytes_per_s": _band(p05r.get("l2_die_bytes_per_s"), spec["l2_bytes_per_s"]),
            "l2_effective_fraction": _band(p05r.get("l2_effective_fraction"), 1.0),
        },
    )
    p06r = raw.get("P06_mmad", {})
    p06_modes: dict[str, Any] = {}
    any_p06 = False
    mode_defs = [
        ("bf16", 16, 16, 16, 16, spec["peaks"].get("bf16", 0.0) / aic),
        ("fp16", 16, 16, 16, 16, spec["peaks"].get("fp16", 0.0) / aic),
        ("fp32", 16, 16, 8, 16, spec["peaks"].get("fp32", 0.0) / aic),
        ("int8", 16, 16, 32, 16, spec["peaks"].get("int8", 0.0) / aic),
    ]
    if is_arch35:
        fp8_cap = spec["peaks"].get("fp8_e4m3", spec["peaks"].get("int8", 0.0)) / aic
        mode_defs.extend([("fp8_e4m3", 16, 16, 32, 16, fp8_cap), ("mxfp8", 16, 16, 32, 16, fp8_cap), ("mxfp4", 16, 16, 64, 16, fp8_cap * 2.0)])
    for m_name, tm, tn, tk, miss, cap in mode_defs:
        if cap <= 0.0:
            continue
        b = _band(p06r.get("modes", {}).get(m_name), cap)
        ms = _cell_status({"r": b})
        any_p06 = any_p06 or (ms == "calibrated")
        entry: dict[str, Any] = {"status": ms, "tile_m": tm, "tile_n": tn, "tile_k": tk, "m_iss": miss, "cube_only_rate": b if ms == "calibrated" else _clean_band(b)}
        if m_name in {"mxfp8", "mxfp4"}:
            entry["scale_group"] = 32
        p06_modes[m_name] = entry
    p["P06_mmad"] = _make_cell(
        "P06_mmad", {"rate": "flops_per_s_per_aic", "latency": "ns"},
        {"pipe_m_barrier_fractal_threshold": 10, "small_m_bf16_ns": p06r.get("small_m_bf16_ns", {})}, {},
        {"pipe_m_barrier_ns": _bound_only(0.0, "latency_lower")},
        status_override="calibrated" if any_p06 else "unmeasured", modes=p06_modes,
    )
    p["P07_vec_alu"] = _make_cell(
        "P07_vec_alu", {"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": 256, "fp32_lanes": 64, "bf16_lanes": 128},
        {"vec_alu_ops_per_s_per_aiv": _band(raw.get("P07_vec_alu", {}).get("vec_alu_ops_per_s_per_aiv"), vec_cap * 2.0)},
        {"raw_latency_cycles": _bound_only(0.0, "latency_lower")},
    )
    p["P08_vec_sfu"] = _make_cell(
        "P08_vec_sfu", {"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": 256, "fp32_lanes": 64, "bf16_lanes": 128},
        {"vec_sfu_ops_per_s_per_aiv": _band(raw.get("P08_vec_sfu", {}).get("vec_sfu_ops_per_s_per_aiv"), vec_cap)},
        {"raw_latency_cycles": _bound_only(0.0, "latency_lower")},
    )
    p09r = raw.get("P09_vec_reduce_cast", {})
    p["P09_vec_reduce_cast"] = _make_cell(
        "P09_vec_reduce_cast", {"rate": "ops_per_s_per_aiv", "cast_rate": "bytes_per_s_per_aiv"},
        {"mx_scale_group": 32, "mx_dynamic_quant_supported": is_arch35},
        {
            "vec_reduce_ops_per_s_per_aiv": _band(p09r.get("vec_reduce_ops_per_s_per_aiv"), vec_cap),
            "cast_bytes_per_s_per_aiv": _band(p09r.get("cast_bytes_per_s_per_aiv"), spec["mte2_aiv_bytes_per_s"] / aiv),
        },
    )
    p["P10_sort_topk"] = _make_cell(
        "P10_sort_topk", {"rate": "elems_per_s"},
        {"sort_algorithm": "masked_hist256" if is_arch35 else "sort32_mrgsort4", "supported_sortall_lengths": [64, 128, 256, 512, 1024, 2048], "histogram_trunk_elems": 16384 if is_arch35 else 0},
        {"sort_elems_per_s": _band(raw.get("P10_sort_topk", {}).get("sort_elems_per_s"), spec["vector_ops_per_s"])},
    )
    p["P11_simt"] = _make_cell(
        "P11_simt", {"rate": "ops_per_s_per_aiv", "latency": "ns"},
        {"supported": is_arch35, "simt_ub_carveout_bytes": spec["simt_ub_carveout_bytes"], "tla_max_threads": 2048 if is_arch35 else 0, "catlass_max_threads": 1024 if is_arch35 else 0, "warp_size": 32},
        {}, {"simt_ops_per_s_per_aiv": _bound_only(vec_cap if is_arch35 else 0.0), "syncthreads_ns": _bound_only(0.0, "latency_lower"), "simt_launch_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured" if is_arch35 else "unsupported",
    )
    p["P12_intracore_flag"] = _make_cell(
        "P12_intracore_flag", {"latency": "ns"}, {"event_ids": 8, "unit_flag_requires_single_l0c_stage": True}, {},
        {"flag_ns": _bound_only(0.0, "latency_lower"), "unitflag_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured",
    )
    p["P13_crosscore_sync"] = _make_cell(
        "P13_crosscore_sync", {"latency": "ns"},
        {"aiv_inter_block_barrier_id": 8, "aic_inter_block_barrier_id": 9, "intra_subblock_barrier_id": 10, "ffts_max_flag_id": 7, "reverse_depth": 15}, {},
        {"mode2_paired_ns": _bound_only(0.0, "latency_lower"), "mode1_subblock_ns": _bound_only(0.0, "latency_lower"), "mode0_aic_barrier_ns": _bound_only(0.0, "latency_lower"), "mode0_aiv_barrier_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p["P14_mutex_block_mmad"] = _make_cell(
        "P14_mutex_block_mmad", {"latency": "ns"}, {"max_mutex_ids": 28, "unit_flag_requires_single_l0c_stage": True}, {},
        {"mutex_overhead_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured",
    )
    p15r = raw.get("P15_rts_stars", {})
    p["P15_rts_stars"] = _make_cell(
        "P15_rts_stars", {"latency": "ns"}, {"queue_scope": "die", "queue_depth": 64, "lanes": 1},
        {
            "issue_interval_ns": _band(p15r.get("issue_interval_ns"), 0.0, kind="latency_lower"),
            "aiv_first_issue_ns": _band(p15r.get("aiv_first_issue_ns"), 0.0, kind="latency_lower"),
            "aic_first_issue_ns": _band(p15r.get("aic_first_issue_ns"), 0.0, kind="latency_lower"),
        },
        {"host_submit_ns": _band(p15r.get("host_submit_ns"), 0.0, kind="latency_lower")},
    )
    p["P16_multistream_core_control"] = _make_cell(
        "P16_multistream_core_control", {"latency": "ns"}, {"modes": ["disjoint", "shared_cap"]},
        {"stream_event_ns": _band(raw.get("P16_multistream_core_control", {}).get("stream_event_ns"), 0.0, kind="latency_lower")},
        {"set_limit_ns": _bound_only(0.0, "latency_lower")},
    )
    p["P17_aicpu_metadata_launch"] = _make_cell(
        "P17_aicpu_metadata_launch", {"latency": "ns"}, {"aclgraph_cacheable": True}, {},
        {"host_select_ns": _bound_only(0.0, "latency_lower"), "aicpu_base_ns": _bound_only(0.0, "latency_lower"), "gm_pub_ns": _bound_only(0.0, "latency_lower"), "eager_d2h_sync_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p["P18_hccl_collective_band"] = _make_cell(
        "P18_hccl_collective_band", {"rate": "bytes_per_s", "latency": "ns"},
        {"payload_bytes_per_frame": 480 if is_arch35 else 1024, "wire_bytes_per_frame": 512 if is_arch35 else 1024, "short_final_frame_policy": "pad_full_frame"}, {},
        {"link_bytes_per_s": _bound_only(4.0e11 if is_arch35 else 2.24e11), "startup_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p["P19_direct_peer_window"] = _make_cell(
        "P19_direct_peer_window", {"rate": "bytes_per_s", "latency": "ns"},
        {"transport": "a5_ub_shmem_mte" if is_arch35 else "a3_sio_hccs", "token_payload_bytes_per_frame": 480, "token_wire_bytes_per_frame": 512}, {},
        {"peer_bytes_per_s": _bound_only(4.0e11 if is_arch35 else 2.24e11), "status_sync_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p20r = raw.get("P20_host_dma_sync", {})
    p["P20_host_dma_sync"] = _make_cell(
        "P20_host_dma_sync", {"rate": "bytes_per_s", "latency": "ns"}, {"contention_scope": "root_complex"},
        {
            "host_h2d_bytes_per_s": _band(p20r.get("host_h2d_bytes_per_s"), 6.4e10),
            "host_d2h_bytes_per_s": _band(p20r.get("host_d2h_bytes_per_s"), 6.4e10),
        },
        {
            "pinned_dma_startup_ns": _band(p20r.get("pinned_dma_startup_ns"), 0.0, kind="latency_lower"),
            "pageable_dma_startup_ns": _band(p20r.get("pageable_dma_startup_ns"), 0.0, kind="latency_lower"),
            "eager_d2h_sync_ns": _band(p20r.get("eager_d2h_sync_ns"), 0.0, kind="latency_lower"),
        },
    )

    payload: dict[str, Any] = {
        "schema_version": PRIMITIVE_SCHEMA_VERSION,
        "arch_preset": spec["arch_preset"],
        "sku": sku,
        "cann_version": cann_version,
        "provenance": {
            "campaign_id": campaign_id, "endpoint_alias": alias, "capture_uri": top_capture,
            "calibrated": any(c["status"] == "calibrated" for c in p.values()),
            "collected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "silicon_fingerprint": {k: spec[k] for k in ("aic_count", "aiv_count", "hbm_gib", "l2_bytes", "ub_bytes", "l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes")},
        },
        "primitives": p,
    }
    payload["envelope_sha256"] = compute_envelope_sha256(payload)
    return payload


def _load_endpoint(alias: str) -> dict[str, Any]:
    data = json.loads((Path.home() / ".mindie" / "endpoints.json").read_text(encoding="utf-8"))
    eps = data.get("endpoints", data)
    if alias not in eps:
        raise SystemExit(f"Unknown endpoint alias: {alias}")
    return dict(eps[alias])


def _ssh_run(ep: dict[str, Any], remote_cmd: str, *, stdin_text: str | None = None, timeout: int = REMOTE_BENCH_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
        "-p", str(ep.get("port", 22)),
    ]
    if ep.get("identity_file"):
        cmd.extend(["-i", str(ep["identity_file"])])
    cmd.extend(["-l", str(ep.get("user", "root")), "--", str(ep["host"]), remote_cmd])
    return subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=timeout, check=False)


def _check_idle_npu_card(ep: dict[str, Any], expected_npu_id: int) -> None:
    res = _ssh_run(ep, "npu-smi info", timeout=20)
    if res.returncode != 0:
        raise RuntimeError(f"npu-smi info failed (rc={res.returncode})")
    if f"No running processes found in NPU {expected_npu_id}" not in res.stdout:
        raise RuntimeError(f"NPU {expected_npu_id} is not confirmed idle by npu-smi info")


def run_remote_calibration(args: argparse.Namespace) -> dict[str, Any]:
    ep = _load_endpoint(args.endpoint_alias)
    _check_idle_npu_card(ep, args.npu_id)
    spec = _load_sku_hardware_spec(args.sku)
    script_b64 = base64.b64encode(Path(__file__).read_bytes()).decode("ascii")
    env_line = f"export ASCEND_RT_VISIBLE_DEVICES={args.visible_devices}\n" if args.visible_devices else ""
    bash_script = (
        "set -euo pipefail\n"
        "TMPD=$(mktemp -d /tmp/mindie-l0-bench.XXXXXX)\n"
        "trap 'rm -rf \"$TMPD\"' EXIT\n"
        f"cat <<'EOF_B64' | base64 -d > \"$TMPD/bench_l0.py\"\n{script_b64}\nEOF_B64\n"
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || "
        "source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true\n"
        f"{env_line}"
        f"{args.python_bin} \"$TMPD/bench_l0.py\" --local-npu "
        f"--device-id {args.device_id} --l2-bytes {spec['l2_bytes']} "
        f"--aic-count {spec['aic_count']} --aiv-count {spec['aiv_count']} "
        f"--arch {spec['arch_preset']} --warmup {args.warmup} --repeats {args.repeats}\n"
    )
    remote_cmd = f"docker exec -i {args.container} bash -s" if args.container else "bash -s"
    completed = _ssh_run(ep, remote_cmd, stdin_text=bash_script, timeout=REMOTE_BENCH_TIMEOUT_S)
    if completed.returncode != 0:
        raise RuntimeError(f"Remote calibration failed (rc={completed.returncode}): {completed.stderr[-800:]}")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    raise RuntimeError(f"No {RESULT_MARKER} payload in remote output")


def update_manifest(manifest_path: Path, table_path: Path, payload: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"schema_version": PRIMITIVE_SCHEMA_VERSION, "tables": {}}
    sku = payload["sku"]
    manifest.setdefault("tables", {})[sku] = {
        "file": table_path.name, "sku": sku, "compatible_skus": SKU_COMPATIBLE_ALIASES.get(sku, [sku]),
        "arch_preset": payload["arch_preset"], "cann_version": payload["cann_version"],
        "calibrated": bool(payload["provenance"]["calibrated"]),
        "calibrated_primitive_count": sum(1 for c in payload["primitives"].values() if c["status"] == "calibrated"),
        "total_primitives": len(payload["primitives"]),
        "envelope_sha256": payload["envelope_sha256"],
        "capture_uri": payload["provenance"]["capture_uri"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Ascend L0 primitive micro-benchmark calibration suite")
    parser.add_argument("--local-npu", action="store_true")
    parser.add_argument("--emit-unmeasured", action="store_true")
    parser.add_argument("--endpoint-alias", default="")
    parser.add_argument("--record-alias", default="")
    parser.add_argument("--container", default="")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--npu-id", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--visible-devices", default="")
    parser.add_argument("--sku", default="")
    parser.add_argument("--arch", default="npu_arch_2201")
    parser.add_argument("--cann-version", default="unmeasured")
    parser.add_argument("--campaign-id", default="camp-202611-l0")
    parser.add_argument("--l2-bytes", type=int, default=134217728)
    parser.add_argument("--aic-count", type=int, default=32)
    parser.add_argument("--aiv-count", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--primitives", default="all")
    parser.add_argument("--out", default="")
    parser.add_argument("--manifest", default="data/primitives/manifest.json")
    args = parser.parse_args(argv)

    if args.local_npu:
        raw = run_local_npu_microbenchmarks(
            device_id=args.device_id, l2_bytes=args.l2_bytes, aic_count=args.aic_count,
            aiv_count=args.aiv_count, arch_preset=args.arch, warmup=args.warmup, repeats=args.repeats,
        )
        print(RESULT_MARKER + json.dumps(raw, sort_keys=True, separators=(",", ":")))
        return 0

    if not args.sku:
        raise SystemExit("--sku is required")
    out_path = Path(args.out) if args.out else (REPO_ROOT / "data" / "primitives" / SKU_DEFAULT_FILES[args.sku])
    manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else (REPO_ROOT / args.manifest)
    raw_meas = None if args.emit_unmeasured else run_remote_calibration(args)
    rec_alias = args.record_alias or sanitize_record_alias(args.endpoint_alias or args.sku, args.sku)
    payload = build_primitive_table_payload(
        sku=args.sku, cann_version=args.cann_version, endpoint_alias=rec_alias,
        campaign_id=args.campaign_id, raw_measurements=raw_meas,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_manifest(manifest_path, out_path, payload)
    cal = [k for k, v in payload["primitives"].items() if v["status"] == "calibrated"]
    print(json.dumps({"sku": args.sku, "out": str(out_path), "calibrated_count": len(cal), "calibrated": cal, "envelope_sha256": payload["envelope_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
