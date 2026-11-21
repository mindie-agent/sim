#!/usr/bin/env python3
"""Offline Ascend A2/A3/A5 NPU L0 micro-benchmark calibration suite.

[Role] Thin Plane-1 offline calibration CLI executing bounded synthetic NPU
  micro-benchmarks on idle physical Ascend A2/A3/A5 dies via endpoint alias
  and delegating full-envelope SHA-256 sealing & manifest updates to the
  single canonical installed sealer ``mindie_sim.primitives.sealer``.
[Invariants] Checks ``npu-smi info`` before touching any NPU so only idle
  cards (<=2% AICore, zero competing processes) are used; caps synthetic HBM
  allocations <= 4 GiB; runs 30-GEMM DVFS lock, >= 3x L2 cold rotating rings,
  and 2.5 ms StreamBlocker saturation; supports newly scaffolded SKUs via
  ``--profile-json``; remains self-contained in ``--local-npu`` remote mode;
  never leaks raw IPs, hostnames, usernames, or host ``$HOME`` paths.
[Timeouts & Locks] Bounded SSH/remote subprocess execution with unconditional
  remote temporary directory cleanup on exit; targets idle NPU dies only.
[Side-Effect Policy] Writes only the requested ``--out`` JSON table, raw
  capture sidecar, and ``--manifest`` index; cleans up remote temp files.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from typing import Any

_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _PARENTS[0]
if (REPO_ROOT / "mindie_sim").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULT_MARKER = "__MINDIE_L0_BENCH_JSON__:"
MAX_IDLE_AICORE_UTIL_PCT = 2
MAX_IDLE_HBM_UTIL_PCT = 10
WARMUP_GEMM_ITERS = 30
WARMUP_GEMM_DIM = 4096
BLOCKER_TARGET_MS = 2.5
MIN_SINGLE_GEMM_MS = 0.05
DEFAULT_WARMUP = 8
DEFAULT_REPEATS = 32
MIB = 1024 * 1024
SSH_CONNECT_TIMEOUT_S = 8
REMOTE_BENCH_TIMEOUT_S = 360
SAFE_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_VISIBLE_RE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")

if "--local-npu" not in sys.argv:
    from mindie_sim.primitives.sealer import (  # noqa: F401
        MAX_CALIBRATED_CV_PCT,
        PRIMITIVE_SCHEMA_VERSION,
        PRIVATE_CAPTURE_RE,
        SKU_DEFAULT_FILES,
        build_primitive_table_payload,
        build_primitive_table_with_diagnostics,
        canonical_capture_bytes,
        compute_envelope_sha256,
        load_sku_hardware_spec as _load_sku_hardware_spec,
        sanitize_record_alias,
        summarize_samples,
        summarize_samples_with_reason,
        update_manifest,
    )


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
    hbm_rw11, hbm_rw21 = [(2.0 * slab) / (t * 1e-3) for t in rw11_ms], [(3.0 * slab) / (t * 1e-3) for t in rw21_ms]
    hbm_r = [max(slab / (tr * 1e-3), (2.0 / 3.0) * rw) for tr, rw in zip(rd_ms, hbm_rw21)]
    t_r1, t_w1 = slab / max(1.0, statistics.median(hbm_r)) * 1e3, slab / max(1.0, statistics.median(hbm_w)) * 1e3
    alphas = [max(0.0, min(1.0, (t11 - max(t_r1, t_w1)) / max(1e-9, min(t_r1, t_w1)))) for t11 in rw11_ms]
    s_src, s_dst = ring_a[0][: 32768 * 1024].view(32768, 1024), torch.empty((32768, 256), dtype=torch.bfloat16, device=dev)
    strided_bps = [(2.0 * 32768 * 512.0) / (t * 1e-3) for t in _time_ms(lambda _i: s_dst.copy_(s_src[:, :256]), inner=2, blocker=True)]
    ring_a = ring_b = ring_c = s_src = s_dst = None
    torch.npu.empty_cache()

    warm_bytes = min(32 * MIB, max(8 * MIB, l2_bytes // 4))
    w_src, w_dst = torch.ones((warm_bytes // 2,), dtype=torch.bfloat16, device=dev), torch.empty((warm_bytes // 2,), dtype=torch.bfloat16, device=dev)
    l2_bps = [(2.0 * warm_bytes) / (t * 1e-3) for t in _time_ms(lambda _i: w_dst.copy_(w_src), inner=8, blocker=True)]
    mte3_warm = [(warm_bytes / (t * 1e-3)) / max(1, aiv_count) for t in _time_ms(lambda _i: w_dst.fill_(1.0), inner=8, blocker=True)]
    a_dst, a_idx, a_src = torch.zeros((65536,), dtype=torch.float32, device=dev), torch.arange(262144, dtype=torch.int64, device=dev) % 65536, torch.ones((262144,), dtype=torch.float32, device=dev)
    atom_bps = [(2.0 * 262144 * 4.0) / (t * 1e-3) for t in _time_ms(lambda _i: a_dst.index_add_(0, a_idx, a_src), inner=8, blocker=True)]
    w_src = w_dst = a_dst = a_idx = a_src = None
    torch.npu.empty_cache()
    raw["P01_datacopy"] = {
        "hbm_read_bytes_per_s": hbm_r, "hbm_write_bytes_per_s": hbm_w,
        "hbm_rw11_bytes_per_s": hbm_rw11, "hbm_rw21_bytes_per_s": hbm_rw21,
        "hbm_rw_alpha": alphas, "strided_512b_bytes_per_s": strided_bps,
        "l2_die_bytes_per_s": l2_bps, "mte2_aiv_hit_bytes_per_s_per_core": [bw / max(1, aiv_count) for bw in l2_bps],
    }
    raw["P04_mte3_store"] = {"mte3_warm_bytes_per_s_per_aiv": mte3_warm, "atomic_rmw_bytes_per_s": atom_bps}

    ws_curve: dict[str, float] = {}
    ws_trials: dict[int, list[float]] = {}
    ws_list = (8, 32, 64, 96, 128, 160, 192, 256, 320)
    for ws_mib in ws_list:
        xb = torch.ones(((ws_mib * MIB) // 4,), dtype=torch.bfloat16, device=dev)
        yb = torch.empty_like(xb)
        ts = _time_ms(lambda _i: yb.copy_(xb), inner=4 if ws_mib <= 64 else 1, blocker=True)
        bws = [(ws_mib * MIB) / (t * 1e-3) for t in ts]
        ws_trials[ws_mib] = bws
        ws_curve[f"{ws_mib}MiB"] = round(statistics.median(bws), 2)
        xb = yb = None
    torch.npu.empty_cache()
    eff_fracs: list[float] = []
    for idx in range(repeats):
        trial_bws = [ws_trials[m][idx] for m in ws_list]
        t_peak, t_cold = max(trial_bws), trial_bws[-1]
        t_thr = t_cold + 0.5 * max(0.0, t_peak - t_cold)
        knee_mib = float(ws_list[0])
        for pos in range(len(ws_list) - 1):
            b0, b1 = trial_bws[pos], trial_bws[pos + 1]
            if b0 >= t_thr:
                knee_mib = float(ws_list[pos])
                if b1 < t_thr and b0 > b1:
                    frac = (b0 - t_thr) / (b0 - b1)
                    knee_mib = float(ws_list[pos]) + frac * float(ws_list[pos + 1] - ws_list[pos])
        eff_fracs.append(min(1.0, max(0.05, (knee_mib * MIB) / max(1, l2_bytes))))
    raw["P05_l2_residency_reuse"] = {
        "ws_sweep_bytes_per_s": ws_curve, "ws_sweep_grid_mib": list(ws_list),
        "l2_knee_mib": round(statistics.median(eff_fracs) * l2_bytes / MIB, 2),
        "l2_effective_fraction": eff_fracs, "l2_die_bytes_per_s": l2_bps,
    }

    mm_flops = 2.0 * (4096 ** 3)
    mm_modes: dict[str, list[float]] = {}
    for m_name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        ma = torch.randn((4096, 4096), dtype=dt, device=dev)
        mb = torch.randn((4096, 4096), dtype=dt, device=dev)
        mm_modes[m_name] = [(mm_flops / (t * 1e-3)) / max(1, aic_count) for t in _time_ms(lambda _i: torch.matmul(ma, mb), inner=2, blocker=True)]
        ma = mb = None
    try:
        if hasattr(torch_npu, "npu_quant_matmul"):
            qa = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8, device=dev)
            qb = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8, device=dev)
            sc = torch.ones((4096,), dtype=torch.float32, device=dev)
            mm_modes["int8"] = [(mm_flops / (t * 1e-3)) / max(1, aic_count) for t in _time_ms(
                lambda _i: torch_npu.npu_quant_matmul(qa, qb, sc, output_dtype=torch.bfloat16), inner=2, blocker=True)]
            qa = qb = sc = None
    except Exception:
        pass
    sm_ns: dict[str, float] = {}
    mb_sm = torch.randn((4096, 4096), dtype=torch.bfloat16, device=dev)
    for mv in (1, 16, 32, 64, 128, 256):
        ma_sm = torch.randn((mv, 4096), dtype=torch.bfloat16, device=dev)
        sm_ns[str(mv)] = round(statistics.median(_time_ms(lambda _i: torch.matmul(ma_sm, mb_sm), inner=4, blocker=True)) * 1e6, 2)
        ma_sm = None
    mb_sm = None
    torch.npu.empty_cache()
    raw["P06_mmad"] = {"modes": mm_modes, "small_m_bf16_ns": sm_ns}

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
    va = vb = vc = None
    sx = torch.randn((256, 2048), dtype=torch.float32, device=dev)
    raw["P10_sort_topk"] = {"sort_elems_per_s": [
        (256 * 2048) / (t * 1e-3) for t in _time_ms(lambda _i: torch.topk(sx, k=128, dim=-1), inner=4, blocker=True)]}
    sx = None
    torch.npu.empty_cache()

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
    aiv_f = [t * 1e6 for t in _time_ms(lambda _i: torch.add(ta, tb, out=to), inner=1, blocker=True)]
    aic_f = [t * 1e6 for t in _time_ms(lambda _i: torch.matmul(aa, ab), inner=1, blocker=True)]
    b16 = _time_ms(lambda _i: torch.add(ta, tb, out=to), inner=16, blocker=True)
    b64 = _time_ms(lambda _i: torch.add(ta, tb, out=to), inner=64, blocker=True)
    q_ns = [max(1.0, ((t64 * 64.0) - (t16 * 16.0)) / 48.0 * 1e6) for t16, t64 in zip(b16, b64)]
    raw["P15_rts_stars"] = {"host_submit_ns": host_ns, "issue_interval_ns": q_ns, "aiv_first_issue_ns": aiv_f, "aic_first_issue_ns": aic_f}

    s_alt, ev_sync = torch.npu.Stream(device=dev), torch.npu.Event(enable_timing=False)
    for _ in range(warmup):
        ev_sync.record()
        s_alt.wait_event(ev_sync)
    torch.npu.synchronize()
    se_ns = [t * 1e6 for t in _time_ms(lambda _i: (ev_sync.record(), s_alt.wait_event(ev_sync)), inner=32, blocker=True)]
    raw["P16_multistream_core_control"] = {"stream_event_ns": se_ns}

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
    wa = wb = h_pin = d_buf = sp_pin = sp_page = sp_dev = ta = tb = to = aa = ab = None
    torch.npu.empty_cache()
    return raw


def _load_endpoint(alias: str) -> dict[str, Any]:
    data = json.loads((Path.home() / ".mindie" / "endpoints.json").read_text(encoding="utf-8"))
    eps = data.get("endpoints", data)
    if alias not in eps:
        raise SystemExit(f"Unknown endpoint alias: {alias}")
    return dict(eps[alias])


def _ssh_run(ep: dict[str, Any], remote_cmd: str, *, stdin_text: str | None = None, timeout: int = REMOTE_BENCH_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        "-o", "StrictHostKeyChecking=yes", "-o", "LogLevel=ERROR",
        "-p", str(ep.get("port", 22)),
    ]
    if ep.get("identity_file"):
        cmd.extend(["-i", str(ep["identity_file"])])
    cmd.extend(["-l", str(ep.get("user", "root")), "--", str(ep["host"]), remote_cmd])
    try:
        return subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"bench_ssh_timeout:timeout_s={timeout}") from None


def parse_npu_smi_idle_die(
    info_text: str, map_text: str, common_text: str, *,
    npu_id: int, chip_id: int, device_id: int, visible_devices: str = "",
) -> dict[str, Any]:
    """Strictly verify physical die idle state, mapping, util, HBM, and clock."""
    if not re.search(rf"No running processes found in NPU {npu_id}(?!\d)", info_text):
        raise RuntimeError(f"bench_npu_not_idle:npu={npu_id}")
    logic_id: int | None = None
    has_slot = "Slot ID" in map_text
    for line in map_text.splitlines():
        cols = line.split()
        if len(cols) >= 5 and cols[0].isdigit() and cols[1].isdigit() and cols[2].isdigit() and cols[3].isdigit():
            c_idx, l_idx = (2, 3) if has_slot else (1, 2)
            if int(cols[0]) == npu_id and int(cols[c_idx]) == chip_id:
                logic_id = int(cols[l_idx])
                break
    if logic_id is None:
        raise ValueError(f"bench_npu_mapping_missing:npu={npu_id}:chip={chip_id}")
    if visible_devices:
        if not SAFE_VISIBLE_RE.fullmatch(visible_devices):
            raise ValueError("bench_visible_devices_invalid")
        vis_list = [int(x) for x in visible_devices.split(",")]
        if device_id < 0 or device_id >= len(vis_list):
            raise ValueError("bench_npu_die_mismatch")
        target_logic = vis_list[device_id]
    else:
        target_logic = device_id
    if target_logic != logic_id:
        raise ValueError(f"bench_npu_die_mismatch:expected_logic={logic_id}:got={target_logic}")
    block = common_text
    if "Chip ID" in common_text:
        parts = re.split(r"(?m)^\s*Chip ID\s*:\s*", common_text)
        matched = [part for part in parts[1:] if re.match(rf"^{chip_id}(?!\d)", part.strip())]
        if not matched:
            raise ValueError(f"bench_npu_common_missing:chip={chip_id}")
        block = matched[0]

    def _int_field(label: str) -> int | None:
        m = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", block)
        return int(m.group(1)) if m else None

    aic_util, hbm_util = _int_field("Aicore Usage Rate(%)"), _int_field("HBM Usage Rate(%)")
    aic_cnt, aic_freq, cur_freq = _int_field("Aicore Count"), _int_field("Aicore Freq(MHZ)"), _int_field("Aicore curFreq(MHZ)")
    if aic_util is None or hbm_util is None:
        raise RuntimeError("bench_npu_telemetry_missing")
    if aic_cnt is None or aic_cnt <= 0 or aic_freq is None or aic_freq <= 0:
        raise RuntimeError("bench_npu_clock_missing")
    if aic_util > MAX_IDLE_AICORE_UTIL_PCT:
        raise RuntimeError(f"bench_npu_busy:aicore_util={aic_util}")
    if hbm_util > MAX_IDLE_HBM_UTIL_PCT:
        raise RuntimeError(f"bench_npu_busy:hbm_util={hbm_util}")
    return {
        "npu_id": npu_id, "chip_id": chip_id, "chip_logic_id": target_logic,
        "aicore_util_pct": aic_util, "hbm_util_pct": hbm_util,
        "observed_aic_count": aic_cnt, "observed_aicore_freq_mhz": aic_freq, "observed_cur_freq_mhz": cur_freq,
    }


def _check_idle_npu_card(ep: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cmd = f"npu-smi info; echo '__MAP__'; npu-smi info -m; echo '__COMMON__'; npu-smi info -t common -i {int(args.npu_id)}"
    res = _ssh_run(ep, cmd, timeout=20)
    if res.returncode != 0:
        raise RuntimeError(f"bench_npu_smi_failed:rc={res.returncode}")
    parts = res.stdout.split("__MAP__", 1)
    if len(parts) != 2 or "__COMMON__" not in parts[1]:
        raise RuntimeError("bench_npu_smi_parse_failed")
    info_text = parts[0]
    map_text, common_text = parts[1].split("__COMMON__", 1)
    return parse_npu_smi_idle_die(
        info_text, map_text, common_text,
        npu_id=int(args.npu_id), chip_id=int(args.chip_id),
        device_id=int(args.device_id), visible_devices=str(args.visible_devices or ""),
    )


@contextmanager
def _register_profile_json(sku: str, profile_json: str):
    if not profile_json:
        yield
        return
    from mindie_sim.backends import ascend_profiles

    raw = json.loads(Path(profile_json).read_text(encoding="utf-8"))
    entry = raw[sku] if isinstance(raw, dict) and sku in raw and isinstance(raw[sku], dict) else raw
    if not isinstance(entry, dict) or "device" not in entry:
        raise ValueError(f"bench_profile_json_invalid:{sku}")
    reg = ascend_profiles()
    already = sku in reg
    if not already:
        reg[sku] = deepcopy(entry)
    try:
        yield
    finally:
        if not already:
            reg.pop(sku, None)


def run_remote_calibration(args: argparse.Namespace) -> dict[str, Any]:
    ep = _load_endpoint(args.endpoint_alias)
    obs = _check_idle_npu_card(ep, args)
    spec = _load_sku_hardware_spec(args.sku)
    if args.container and not SAFE_CONTAINER_RE.fullmatch(args.container):
        raise ValueError("bench_container_invalid")
    script_b64 = base64.b64encode(Path(__file__).read_bytes()).decode("ascii")
    env_line = f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(args.visible_devices)}\n" if args.visible_devices else ""
    py_cmd = " ".join([
        shlex.quote(args.python_bin), '"$TMPD/bench_l0.py"', "--local-npu",
        "--device-id", str(int(args.device_id)), "--l2-bytes", str(int(spec["l2_bytes"])),
        "--aic-count", str(int(spec["aic_count"])), "--aiv-count", str(int(spec["aiv_count"])),
        "--arch", shlex.quote(spec["arch_preset"]), "--warmup", str(int(args.warmup)), "--repeats", str(int(args.repeats)),
    ])
    bash_script = (
        "set -euo pipefail\n"
        "TMPD=$(mktemp -d /tmp/mindie-l0-bench.XXXXXX)\n"
        "trap 'rm -rf \"$TMPD\"' EXIT\n"
        f"cat <<'EOF_B64' | base64 -d > \"$TMPD/bench_l0.py\"\n{script_b64}\nEOF_B64\n"
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || "
        "source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true\n"
        f"{env_line}{py_cmd}\n"
    )
    remote_cmd = f"docker exec -i {shlex.quote(args.container)} bash -s" if args.container else "bash -s"
    completed = _ssh_run(ep, remote_cmd, stdin_text=bash_script, timeout=REMOTE_BENCH_TIMEOUT_S)
    if completed.returncode != 0:
        raise RuntimeError(f"remote_calibration_failed:rc={completed.returncode}")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            data = json.loads(line[len(RESULT_MARKER):])
            data["observed_silicon"] = obs
            return data
    raise RuntimeError("remote_calibration_missing_result")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Ascend L0 primitive micro-benchmark calibration suite")
    parser.add_argument("--local-npu", action="store_true")
    parser.add_argument("--emit-unmeasured", action="store_true")
    parser.add_argument("--endpoint-alias", default="")
    parser.add_argument("--record-alias", default="")
    parser.add_argument("--container", default="")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--npu-id", type=int, default=0)
    parser.add_argument("--chip-id", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--visible-devices", default="")
    parser.add_argument("--sku", default="")
    parser.add_argument("--profile-json", default="")
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
    with _register_profile_json(args.sku, args.profile_json):
        out_path = Path(args.out) if args.out else (REPO_ROOT / "data" / "primitives" / SKU_DEFAULT_FILES[args.sku])
        manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else (REPO_ROOT / args.manifest)
        raw_meas = None if args.emit_unmeasured else run_remote_calibration(args)
        rec_alias = args.record_alias or sanitize_record_alias(args.endpoint_alias or args.sku, args.sku)
        payload = build_primitive_table_payload(
            sku=args.sku, cann_version=args.cann_version, endpoint_alias=rec_alias,
            campaign_id=args.campaign_id, raw_measurements=raw_meas,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap_path = out_path.parent / "captures" / f"{out_path.stem}.capture.json"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    cap_path.write_bytes(canonical_capture_bytes(args.sku, raw_meas))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_manifest(manifest_path, out_path, payload)
    cal = [k for k, v in payload["primitives"].items() if v["status"] == "calibrated"]
    rel_out = os.path.relpath(out_path, REPO_ROOT) if out_path.resolve().is_relative_to(REPO_ROOT.resolve()) else out_path.name
    print(json.dumps({"sku": args.sku, "out": rel_out, "calibrated_count": len(cal), "calibrated": cal, "envelope_sha256": payload["envelope_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
