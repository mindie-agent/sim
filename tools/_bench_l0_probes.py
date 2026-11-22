#!/usr/bin/env python3
"""Remote Ascend NPU L0 micro-benchmark probe worker (single-die + 2-rank HCCL).

[Role] Execute bounded synthetic single-die (P01, P04..P10, P15, P16, P20) and
  2-rank HCCL (P18, P19) micro-benchmarks on idle physical Ascend dies and
  extract host CANN ``platform_config/<SocName>.ini`` SHA-256 provenance.
[Invariants] Caps synthetic HBM allocations <= 4 GiB; applies 30-GEMM DVFS
  lock, >= 3x L2 ColdRing rotation, and chunked dynamic StreamBlocker timing;
  never emits raw host paths, IPs, hostnames, or usernames into output JSON.
[Timeouts & Locks] Bounded per-probe warmup/repeat loops; ephemeral loopback
  port for 2-rank HCCL only; zero persistent files or external locks.
[Side-Effect Policy] Allocates temporary device buffers freed before return;
  emits one JSON line prefixed with ``RESULT_MARKER`` to stdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

RESULT_MARKER = "__MINDIE_L0_BENCH_JSON__:"
WARMUP_GEMM_ITERS = 30
WARMUP_GEMM_DIM = 4096
MMAD_K_DIM = 8192
BLOCKER_TARGET_MS = 3.0
P15_BLOCKER_TARGET_MS = 4.0
BLOCKER_CHUNK_REPEATS = 8
SMALL_CHUNK_REPEATS = 4
BLOCKER_SETTLE_ITERS = 2
DISPATCH_MARGIN_MS = 0.08
MIN_SINGLE_GEMM_MS = 0.05
DEFAULT_WARMUP = 16
DEFAULT_REPEATS = 32
MIB = 1024 * 1024
KIB = 1024
COLD_SLAB_MIB = 128
MIN_RING_BYTES_MIB = 512
MIN_RING_SLOTS = 6
MAX_RING_SLOTS = 8
L2_RING_MULT = 4
BF16_BYTES = 2
FP32_BYTES = 4
CAST_RW_BYTES = 6.0
STRIDED_ROWS = 32768
STRIDED_COLS = 1024
STRIDED_SLICE = 256
STRIDED_SLICE_BYTES = 512.0
WARM_MIN_MIB = 8
WARM_MAX_MIB = 32
WARM_L2_DIV = 4
ATOMIC_BUCKETS = 65536
ATOMIC_UPDATES = 262144
WS_SWEEP_MIB = (4, 8, 16, 32, 64, 96, 112, 128, 144, 160, 192, 208, 224, 256, 320)
WS_INNER_THRESHOLD_MIB = 64
PAYLOAD_RAMP_KIB = (16, 64, 256, 1024, 4096, 16384, 65536, 131072)
PAYLOAD_RAMP_SMALL_KIB = 1024
PAYLOAD_RAMP_SMALL_INNER = 8
SMALL_M_RING_MIN_SLOTS = 4
SMALL_M_L2_RING_MULT = 3.2
SMALL_M_ROUND_DIGITS = 2
FIXPIPE_F32_BF16_RATIO = 2.0
ERR_PREVIEW_CHARS = 120


def _merge_ascendc_into_raw(
    raw: dict[str, Any], ac: dict[str, Any], *, slab: int, rw11_ms: list[float], hbm_w: list[float],
) -> None:
    """Merge pure on-chip and DMA AscendC microkernel samples into raw calibration dict."""
    s = ac.get("samples", {})
    meta = ac.get("metadata", {})
    if s.get("P01.hbm_read_datacopy_bytes_per_s") and "P01_datacopy" in raw:
        rd_ac = s["P01.hbm_read_datacopy_bytes_per_s"]
        raw["P01_datacopy"]["hbm_read_bytes_per_s"] = rd_ac
        if s.get("P01.hbm_read_datacopy_eager_bytes_per_s"):
            raw["P01_datacopy"]["hbm_read_eager_bytes_per_s"] = s["P01.hbm_read_datacopy_eager_bytes_per_s"]
        if rw11_ms and hbm_w:
            t_r1 = float(slab) / max(1.0, statistics.median(rd_ac)) / MS_PER_S
            t_w1 = float(slab) / max(1.0, statistics.median(hbm_w)) / MS_PER_S
            raw["P01_datacopy"]["hbm_rw_alpha"] = [
                (t11 - max(t_r1, t_w1)) / max(MIN_DIVISOR, min(t_r1, t_w1)) for t11 in rw11_ms
            ]
    if s.get("P01.mte2_aiv_hit_bytes_per_s_per_core") and "P01_datacopy" in raw:
        raw["P01_datacopy"]["mte2_aiv_hit_bytes_per_s_per_core"] = s["P01.mte2_aiv_hit_bytes_per_s_per_core"]
    if s.get("P01.mte2_aic_hit_bytes_per_s_per_core") and "P01_datacopy" in raw:
        raw["P01_datacopy"]["mte2_aic_hit_bytes_per_s_per_core"] = s["P01.mte2_aic_hit_bytes_per_s_per_core"]
    for mk in (
        "gm_alignment_cold_bps", "strided_cold_surface", "gather_v2_cold_surface",
        "mte2_aic_cold_scaling_bps", "mte2_aic_warm_scaling_bps",
        "hbm_read_datacopy_blocked_bps_p50", "hbm_read_datacopy_eager_bps_p50",
    ):
        if mk in meta and "P01_datacopy" in raw:
            raw["P01_datacopy"][mk] = meta[mk]
    if "l2_stride_conflict" in meta and "P05_l2_residency_reuse" in raw:
        raw["P05_l2_residency_reuse"]["l2_stride_conflict"] = meta["l2_stride_conflict"]
    if s.get("P02.l1_to_l0a_bytes_per_s_per_aic") and s.get("P02.l1_to_l0b_bytes_per_s_per_aic"):
        l0a, l0b = s["P02.l1_to_l0a_bytes_per_s_per_aic"], s["P02.l1_to_l0b_bytes_per_s_per_aic"]
        l0a_t = s.get("P02.l1_to_l0a_trans_bytes_per_s_per_aic") or []
        l0b_t = s.get("P02.l1_to_l0b_trans_bytes_per_s_per_aic") or []
        tf_a = statistics.median(l0a) / max(1.0, statistics.median(l0a_t)) if l0a_t else 1.0
        tf_b = statistics.median(l0b) / max(1.0, statistics.median(l0b_t)) if l0b_t else 1.0
        raw["P02_load_data"] = {
            "l1_to_l0a_bytes_per_s_per_aic": l0a, "l1_to_l0b_bytes_per_s_per_aic": l0b,
            "l1_to_l0a_transpose_bytes_per_s_per_aic": l0a_t, "l1_to_l0b_transpose_bytes_per_s_per_aic": l0b_t,
            "setup_ns": s.get("P02.setup_ns") or [],
            "measured_l0a_transpose_ratio": round(tf_a, SIX),
            "measured_l0b_transpose_ratio": round(tf_b, SIX),
            "transpose_factor": round(max(1.0, tf_a), FOUR),
        }
    if s.get("P03.fixpipe_out_bytes_per_s_per_aic"):
        fout = s["P03.fixpipe_out_bytes_per_s_per_aic"]
        raw["P03_fixpipe"] = {
            "fixpipe_out_bytes_per_s_per_aic": fout,
            "fixpipe_in_bytes_per_s_per_aic": [v * FIXPIPE_F32_BF16_RATIO for v in fout],
            "setup_ns": s.get("P03.fixpipe_setup_ns") or [],
        }
    p06_cube = {k[len("P06.cube_only."):]: v for k, v in s.items() if k.startswith("P06.cube_only.") and v}
    if p06_cube or s.get("P06.pipe_m_barrier_ns"):
        p06_dst = raw.setdefault("P06_mmad", {})
        if p06_cube:
            p06_dst["cube_only_modes"] = p06_cube
        if s.get("P06.pipe_m_barrier_ns"):
            p06_dst["pipe_m_barrier_ns"] = s["P06.pipe_m_barrier_ns"]
    if s.get("P07.ub_alu_ops_per_s_per_aiv"):
        raw.setdefault("P07_vec_alu", {})["vec_alu_ops_per_s_per_aiv"] = s["P07.ub_alu_ops_per_s_per_aiv"]
    if s.get("P08.ub_sfu_ops_per_s_per_aiv"):
        raw.setdefault("P08_vec_sfu", {})["vec_sfu_ops_per_s_per_aiv"] = s["P08.ub_sfu_ops_per_s_per_aiv"]
    if s.get("P09.ub_reduce_ops_per_s_per_aiv"):
        raw.setdefault("P09_vec_reduce_cast", {})["vec_reduce_ops_per_s_per_aiv"] = s["P09.ub_reduce_ops_per_s_per_aiv"]
    if s.get("P09.ub_cast_bytes_per_s_per_aiv"):
        raw.setdefault("P09_vec_reduce_cast", {})["cast_bytes_per_s_per_aiv"] = s["P09.ub_cast_bytes_per_s_per_aiv"]
    if s.get("P10.sort_elems_per_s_per_aiv"):
        p10_dst = raw.setdefault("P10_sort_topk", {})
        p10_dst["sort_elems_per_s_per_aiv"] = s["P10.sort_elems_per_s_per_aiv"]
        p10_dst["measured_algorithm"] = "sort32_mrgsort4"
        if "sort_die_elems_per_s_p50" in meta:
            p10_dst["sort_die_elems_per_s_p50"] = meta["sort_die_elems_per_s_p50"]
    if s.get("P11.simt_warp_instructions_per_s_per_aiv"):
        raw["P11_simt"] = {
            "simt_warp_instructions_per_s_per_aiv": s["P11.simt_warp_instructions_per_s_per_aiv"],
            "syncthreads_ns": s.get("P11.syncthreads_ns") or [],
            "simt_launch_ns": s.get("P11.simt_launch_ns") or [],
            "simt_scalar_flops_per_s_per_aiv_p50": meta.get("simt_scalar_flops_per_s_per_aiv_p50"),
        }
    if s.get("P12.flag_mte1_m_ns"):
        raw["P12_intracore_flag"] = {
            "flag_ns": s["P12.flag_mte1_m_ns"],
            "flag_mte2_v_ns": s.get("P12.flag_mte2_v_ns") or [],
            "flag_v_mte3_ns": s.get("P12.flag_v_mte3_ns") or [],
            "flag_m_fix_ns": s.get("P12.flag_m_fix_ns") or [],
            "pipe_v_barrier_ns": s.get("P12.pipe_v_barrier_ns") or [],
        }
    if any(s.get(k) for k in ("P13.mode0_aiv_barrier_ns", "P13.mode0_aic_barrier_ns", "P13.mode2_paired_ns", "P13.mode4_intra_block_ns")):
        raw["P13_crosscore_sync"] = {
            "mode0_aiv_barrier_ns": s.get("P13.mode0_aiv_barrier_ns") or [],
            "mode0_aic_barrier_ns": s.get("P13.mode0_aic_barrier_ns") or [],
            "mode2_paired_ns": s.get("P13.mode2_paired_ns") or [],
            "mode4_intra_block_ns": s.get("P13.mode4_intra_block_ns") or [],
        }
    if s.get("P15.graph_issue_interval_ns"):
        raw.setdefault("P15_rts_stars", {})["npugraph_issue_ns"] = s["P15.graph_issue_interval_ns"]
    if s.get("P15.host_submit_burst_ns"):
        raw.setdefault("P15_rts_stars", {})["host_submit_burst_ns"] = s["P15.host_submit_burst_ns"]
    if ac.get("errors"):
        raw["ascendc_subtask_errors"] = ac["errors"]
VEC_DIM = 2048
TRITON_VEC_BLOCK = 2048
TRITON_VEC_BLOCK_FALLBACK = 1024
TRITON_VEC_ACCUMS = 4
TRITON_VEC_ITERS = 512
TRITON_VEC_OPS_PER_ITER = 4
TRITON_VEC_WARMUP = 4
TRITON_VEC_INNER = 4
TRITON_VEC_INIT_SCALE = 0.001
TRITON_VEC_MUL_VAL = 0.9999
SORT_ROWS = 256
SORT_COLS = 2048
SORT_TOPK = 128
SMALL_M_GRID = (1, 16, 32, 64, 128, 256)
INT8_MIN_VAL = -8
INT8_MAX_VAL = 8
TINY_GEMM_DIM = 16
HOST_SUBMIT_BURST = 128
P15_SMALL_BURST = 16
P15_LARGE_BURST = 64
P15_MARGINAL_DENOM = 48.0
P15_FIRST_INNER = 1
P16_INNER_STEPS = 32
P16_HALF_NS_SCALE = 0.5e6
DMA_SLAB_MIB = 64
DMA_SMALL_ELEMS = 1024
DMA_SMALL_BYTES = DMA_SMALL_ELEMS * FP32_BYTES
BISHENG_VER_TIMEOUT_S = 10
CANN_VER_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?(?:\.[A-Za-z0-9]+)?)")
PRIMITIVE_TAG_RE = re.compile(r"^P(?:0[1-9]|1[0-9]|20)$")
NS_PER_MS = 1e6
NS_PER_S = 1e9
MS_PER_S = 1e-3
MIN_LATENCY_MS = 1e-6
MIN_DIVISOR = 1e-9
MIN_KNEE_FRAC = 0.05
KNEE_HALF_RATIO = 0.5
HALF = 0.5
TWO = 2.0
THREE = 3.0
THREE_INT = 3
FOUR = 4
SIX = 6
EIGHT = 8
SIXTEEN = 16


def _detect_runtime_toolchain(soc_name: str) -> dict[str, Any]:
    """Locate active CANN root INI SHA-256 plus runtime CANN and bisheng versions."""
    env_roots = [
        os.environ.get("ASCEND_HOME_PATH", ""),
        os.environ.get("ASCEND_TOOLKIT_HOME", ""),
        os.environ.get("ASCEND_OPP_PATH", ""),
    ]
    patterns = [f"{r.rstrip('/')}/**/platform_config/{soc_name}.ini" for r in env_roots if r and Path(r).is_dir()]
    patterns.extend([
        f"/usr/local/Ascend/cann*/opp/**/platform_config/{soc_name}.ini",
        f"/usr/local/Ascend/ascend-toolkit/**/platform_config/{soc_name}.ini",
        f"/usr/local/Ascend/**/platform_config/{soc_name}.ini",
    ])
    ini_info: dict[str, Any] = {"ini_filename": f"{soc_name}.ini", "ini_sha256": "", "ini_bytes": 0}
    active_root = next((r for r in env_roots[:2] if r and Path(r).is_dir()), "")
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True))
        if matches:
            ini_path = matches[-1]
            try:
                raw_bytes = Path(ini_path).read_bytes()
            except OSError:
                continue
            ini_info = {
                "ini_filename": os.path.basename(ini_path),
                "ini_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "ini_bytes": len(raw_bytes),
            }
            if not active_root:
                parts = Path(ini_path).parts
                for idx, part in enumerate(parts):
                    if part.startswith("cann") or part == "ascend-toolkit":
                        active_root = str(Path(*parts[: idx + 1]))
                        break
            break
    rt_cann = ""
    if active_root:
        for cfg in (Path(active_root) / "version.cfg", Path(active_root) / "ascend_toolkit_install.info"):
            if cfg.is_file():
                m = CANN_VER_RE.search(cfg.read_text(encoding="utf-8", errors="ignore"))
                if m:
                    rt_cann = m.group(1)
                    break
        if not rt_cann:
            m = CANN_VER_RE.search(Path(active_root).name)
            rt_cann = m.group(1) if m else ""
    bs_ver = ""
    if shutil.which("bisheng"):
        try:
            proc = subprocess.run(["bisheng", "--version"], capture_output=True, text=True, timeout=BISHENG_VER_TIMEOUT_S, check=False)
            first = (proc.stdout or proc.stderr).splitlines()[:1]
            bs_ver = first[0].strip()[:ERR_PREVIEW_CHARS] if first else ""
        except Exception:
            bs_ver = ""
    return {
        **ini_info,
        "runtime_cann_version": rt_cann,
        "bisheng_version": bs_ver,
        "collected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _parse_selected_primitives(spec: str) -> set[str] | None:
    raw = str(spec or "all").strip()
    if not raw or raw.lower() == "all":
        return None
    tags: set[str] = set()
    for tok in raw.split(","):
        head = tok.strip().upper().split("_", 1)[0]
        if not PRIMITIVE_TAG_RE.fullmatch(head):
            raise ValueError(f"bench_primitive_filter_invalid:{tok.strip()}")
        tags.add(head)
    return tags


def _probe_vec_alu_triton(dev: Any, aiv_count: int, time_fn: Any) -> list[float] | None:
    """Measure UB-resident 4-accumulator fused FP32 tl.fma vector ALU FLOPs/s/AIV."""
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception:
        return None

    @triton.jit
    def _vec_alu_ub_kernel(
        x_ptr, s_ptr, c_ptr, out_ptr,
        BLOCK: tl.constexpr,
        STRIDE4: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * STRIDE4 + tl.arange(0, BLOCK)
        a0 = tl.load(x_ptr + offs)
        a1 = tl.load(x_ptr + offs + BLOCK)
        a2 = tl.load(x_ptr + offs + 2 * BLOCK)
        a3 = tl.load(x_ptr + offs + 3 * BLOCK)
        s0 = tl.load(s_ptr + offs)
        c0 = tl.load(c_ptr + offs)
        for _ in range(ITERS):
            a0 = tl.fma(a0, s0, c0)
            a1 = tl.fma(a1, s0, c0)
            a2 = tl.fma(a2, s0, c0)
            a3 = tl.fma(a3, s0, c0)
        tl.store(out_ptr + offs, a0 + a1 + a2 + a3)

    n_progs = max(1, int(aiv_count))
    for blk in (TRITON_VEC_BLOCK, TRITON_VEC_BLOCK_FALLBACK):
        s4 = TRITON_VEC_ACCUMS * blk
        total_elems = n_progs * s4
        x_buf = torch.randn((total_elems,), dtype=torch.float32, device=dev) * TRITON_VEC_INIT_SCALE
        s_buf = torch.full((total_elems,), TRITON_VEC_MUL_VAL, dtype=torch.float32, device=dev)
        c_buf = torch.full((total_elems,), TRITON_VEC_INIT_SCALE, dtype=torch.float32, device=dev)
        out_buf = torch.empty((total_elems,), dtype=torch.float32, device=dev)
        grid = (n_progs,)
        try:
            for _ in range(TRITON_VEC_WARMUP):
                _vec_alu_ub_kernel[grid](
                    x_buf, s_buf, c_buf, out_buf,
                    BLOCK=blk, STRIDE4=s4, ITERS=TRITON_VEC_ITERS,
                )
            torch.npu.synchronize()
            ops_per_aiv = float(TRITON_VEC_ITERS * TRITON_VEC_OPS_PER_ITER * blk)
            ts = time_fn(
                lambda _i, _blk=blk, _s4=s4: _vec_alu_ub_kernel[grid](
                    x_buf, s_buf, c_buf, out_buf,
                    BLOCK=_blk, STRIDE4=_s4, ITERS=TRITON_VEC_ITERS,
                ),
                inner=TRITON_VEC_INNER,
                blocker=True,
            )
            return [ops_per_aiv / (t * MS_PER_S) for t in ts]
        except Exception:
            continue
        finally:
            del x_buf, s_buf, c_buf, out_buf
    return None


def _probe_hbm_l2_domain(
    torch: Any, dev: Any, *, l2_bytes: int, aiv_count: int, repeats: int, time_ms: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, list[float], list[float]]:
    slab = COLD_SLAB_MIB * MIB
    n_slots = min(MAX_RING_SLOTS, max(MIN_RING_SLOTS, math.ceil(max(L2_RING_MULT * l2_bytes, MIN_RING_BYTES_MIB * MIB) / slab)))
    ring_a = [torch.ones((slab // BF16_BYTES,), dtype=torch.bfloat16, device=dev) for _ in range(n_slots)]
    ring_b = [torch.ones((slab // BF16_BYTES,), dtype=torch.bfloat16, device=dev) for _ in range(n_slots)]
    ring_c = [torch.empty((slab // BF16_BYTES,), dtype=torch.bfloat16, device=dev) for _ in range(n_slots)]
    for i in range(n_slots * BF16_BYTES):
        ring_c[i % n_slots].copy_(ring_a[i % n_slots])
    torch.npu.synchronize()
    w_ms = time_ms(lambda i: ring_c[i % n_slots].fill_(1.0), inner=1, blocker=True)
    rw11_ms = time_ms(lambda i: ring_c[i % n_slots].copy_(ring_a[i % n_slots]), inner=1, blocker=True)
    rw21_ms = time_ms(lambda i: torch.add(ring_a[i % n_slots], ring_b[i % n_slots], out=ring_c[i % n_slots]), inner=1, blocker=True)
    rd_ms = time_ms(lambda i: torch.amax(ring_a[i % n_slots]), inner=1, blocker=True)
    hbm_w = [slab / (t * MS_PER_S) for t in w_ms]
    hbm_rw11 = [(TWO * slab) / (t * MS_PER_S) for t in rw11_ms]
    hbm_rw21 = [(THREE * slab) / (t * MS_PER_S) for t in rw21_ms]
    hbm_r = [max(slab / (tr * MS_PER_S), (TWO / THREE) * rw) for tr, rw in zip(rd_ms, hbm_rw21)]
    t_r1 = slab / max(1.0, statistics.median(hbm_r)) / MS_PER_S
    t_w1 = slab / max(1.0, statistics.median(hbm_w)) / MS_PER_S
    alphas = [(t11 - max(t_r1, t_w1)) / max(MIN_DIVISOR, min(t_r1, t_w1)) for t11 in rw11_ms]
    s_dst = torch.empty((STRIDED_ROWS, STRIDED_SLICE), dtype=torch.bfloat16, device=dev)
    s_rings = [a[: STRIDED_ROWS * STRIDED_COLS].view(STRIDED_ROWS, STRIDED_COLS) for a in ring_a]
    strided_bps = [
        (TWO * STRIDED_ROWS * STRIDED_SLICE_BYTES) / (t * MS_PER_S)
        for t in time_ms(lambda i: s_dst.copy_(s_rings[i % n_slots][:, :STRIDED_SLICE]), inner=BF16_BYTES, blocker=True)
    ]
    payload_ramp: dict[str, dict[str, Any]] = {}
    ring_total_bytes = n_slots * slab
    for kb in PAYLOAD_RAMP_KIB:
        n_b = kb * KIB
        el = n_b // BF16_BYTES
        chunks_per_slot = max(1, slab // n_b)
        tot_pos = n_slots * chunks_per_slot
        inn = PAYLOAD_RAMP_SMALL_INNER if kb <= PAYLOAD_RAMP_SMALL_KIB else 1
        def _slice_a(i: int, _e: int = el, _cps: int = chunks_per_slot, _tp: int = tot_pos) -> Any:
            p = i % _tp
            off = (p % _cps) * _e
            return ring_a[p // _cps][off : off + _e]
        def _slice_c(i: int, _e: int = el, _cps: int = chunks_per_slot, _tp: int = tot_pos) -> Any:
            p = i % _tp
            off = (p % _cps) * _e
            return ring_c[p // _cps][off : off + _e]
        t_rd = statistics.median(time_ms(lambda i, sa=_slice_a: torch.amax(sa(i)), inner=inn, blocker=True))
        t_wr = statistics.median(time_ms(lambda i, sc=_slice_c: sc(i).fill_(1.0), inner=inn, blocker=True))
        t_rw = statistics.median(time_ms(lambda i, sa=_slice_a, sc=_slice_c: sc(i).copy_(sa(i)), inner=inn, blocker=True))
        payload_ramp[f"{kb}KB"] = {
            "read_bps": round(n_b / (t_rd * MS_PER_S), SMALL_M_ROUND_DIGITS),
            "write_bps": round(n_b / (t_wr * MS_PER_S), SMALL_M_ROUND_DIGITS),
            "rw11_bps": round((TWO * n_b) / (t_rw * MS_PER_S), SMALL_M_ROUND_DIGITS),
            "cold_ring_bytes": ring_total_bytes,
            "cold_verified": bool(ring_total_bytes >= THREE_INT * l2_bytes),
        }
    del ring_a, ring_b, ring_c, s_rings, s_dst
    torch.npu.empty_cache()

    warm_bytes = min(WARM_MAX_MIB * MIB, max(WARM_MIN_MIB * MIB, l2_bytes // WARM_L2_DIV))
    w_src = torch.ones((warm_bytes // BF16_BYTES,), dtype=torch.bfloat16, device=dev)
    w_dst = torch.empty((warm_bytes // BF16_BYTES,), dtype=torch.bfloat16, device=dev)
    mte3_warm = [(warm_bytes / (t * MS_PER_S)) / max(1, aiv_count) for t in time_ms(lambda _i: w_dst.fill_(1.0), inner=EIGHT, blocker=True)]
    a_dst = torch.zeros((ATOMIC_BUCKETS,), dtype=torch.float32, device=dev)
    a_idx = torch.arange(ATOMIC_UPDATES, dtype=torch.int64, device=dev) % ATOMIC_BUCKETS
    a_src = torch.ones((ATOMIC_UPDATES,), dtype=torch.float32, device=dev)
    idx_upd_bps = [
        (TWO * ATOMIC_UPDATES * FP32_BYTES) / (t * MS_PER_S)
        for t in time_ms(lambda _i: a_dst.index_add_(0, a_idx, a_src), inner=EIGHT, blocker=True, chunk_size=SMALL_CHUNK_REPEATS)
    ]
    del w_src, w_dst, a_dst, a_idx, a_src
    torch.npu.empty_cache()

    ws_curve: dict[str, float] = {}
    ws_trials: dict[int, list[float]] = {}
    for ws_mib in WS_SWEEP_MIB:
        xb = torch.ones(((ws_mib * MIB) // BF16_BYTES,), dtype=torch.bfloat16, device=dev)
        ts = time_ms(lambda _i, _x=xb: torch.amax(_x), inner=FOUR if ws_mib <= WS_INNER_THRESHOLD_MIB else 1, blocker=True)
        bws = [(ws_mib * MIB) / (t * MS_PER_S) for t in ts]
        ws_trials[ws_mib] = bws
        ws_curve[f"{ws_mib}MiB"] = round(statistics.median(bws), BF16_BYTES)
        del xb
    torch.npu.empty_cache()
    eff_fracs: list[float] = []
    for idx in range(repeats):
        trial_bws = [ws_trials[m][idx] for m in WS_SWEEP_MIB]
        t_peak, t_cold = max(trial_bws), trial_bws[-1]
        t_thr = t_cold + KNEE_HALF_RATIO * max(0.0, t_peak - t_cold)
        knee_mib = float(WS_SWEEP_MIB[0])
        for pos in range(len(WS_SWEEP_MIB) - 1):
            b0, b1 = trial_bws[pos], trial_bws[pos + 1]
            if b0 >= t_thr:
                knee_mib = float(WS_SWEEP_MIB[pos])
                if b1 < t_thr and b0 > b1:
                    knee_mib = float(WS_SWEEP_MIB[pos]) + ((b0 - t_thr) / (b0 - b1)) * float(WS_SWEEP_MIB[pos + 1] - WS_SWEEP_MIB[pos])
        eff_fracs.append(max(MIN_KNEE_FRAC, (knee_mib * MIB) / max(1, l2_bytes)))
    plateau_mib = max((m for m in WS_SWEEP_MIB if m * MIB <= l2_bytes), key=lambda m: statistics.median(ws_trials[m]), default=WS_SWEEP_MIB[0])
    l2_plateau_bps = ws_trials[plateau_mib]
    p01 = {
        "hbm_read_bytes_per_s": hbm_r, "hbm_write_bytes_per_s": hbm_w,
        "hbm_rw11_bytes_per_s": hbm_rw11, "hbm_rw21_bytes_per_s": hbm_rw21,
        "hbm_rw_alpha": alphas, "strided_512b_bytes_per_s": strided_bps,
        "payload_ramp_bytes_per_s": payload_ramp,
        "l2_die_bytes_per_s": l2_plateau_bps,
        "l2_copy_bytes_per_s": l2_plateau_bps,
        "mte2_aiv_hit_bytes_per_s_per_core": [(HALF * bw) / max(1, aiv_count) for bw in l2_plateau_bps],
    }
    p04 = {"mte3_warm_bytes_per_s_per_aiv": mte3_warm, "indexed_update_bytes_per_s": idx_upd_bps}
    p05 = {
        "ws_sweep_bytes_per_s": ws_curve, "ws_sweep_grid_mib": list(WS_SWEEP_MIB),
        "l2_knee_mib": round(statistics.median(eff_fracs) * l2_bytes / MIB, BF16_BYTES),
        "l2_plateau_mib": plateau_mib, "l2_effective_fraction": eff_fracs, "l2_die_bytes_per_s": l2_plateau_bps,
    }
    return p01, p04, p05, slab, rw11_ms, hbm_w


def _probe_compute_domain(
    torch: Any, torch_npu: Any, dev: Any, *, aic_count: int, aiv_count: int, l2_bytes: int, arch_preset: str, time_ms: Any,
) -> dict[str, Any]:
    mm_flops = TWO * float(WARMUP_GEMM_DIM * WARMUP_GEMM_DIM * MMAD_K_DIM)
    mm_modes: dict[str, list[float]] = {}
    prev_hf32 = getattr(getattr(torch.npu, "matmul", None), "allow_hf32", None)
    for m_name, dt, k_dim, hf32_flag in (
        ("bf16", torch.bfloat16, MMAD_K_DIM, None),
        ("fp16", torch.float16, MMAD_K_DIM, None),
        ("fp32", torch.float32, WARMUP_GEMM_DIM, False),
        ("hf32", torch.float32, WARMUP_GEMM_DIM, True),
    ):
        if hf32_flag is not None and hasattr(torch.npu, "matmul"):
            torch.npu.matmul.allow_hf32 = hf32_flag
        ma = torch.randn((WARMUP_GEMM_DIM, k_dim), dtype=dt, device=dev)
        mb = torch.randn((k_dim, WARMUP_GEMM_DIM), dtype=dt, device=dev)
        flops_m = TWO * float(WARMUP_GEMM_DIM * WARMUP_GEMM_DIM * k_dim)
        mm_modes[m_name] = [(flops_m / (t * MS_PER_S)) / max(1, aic_count) for t in time_ms(lambda _i: torch.matmul(ma, mb), inner=BF16_BYTES, blocker=True)]
        del ma, mb
    if prev_hf32 is not None and hasattr(torch.npu, "matmul"):
        torch.npu.matmul.allow_hf32 = prev_hf32
    if hasattr(torch_npu, "npu_quant_matmul"):
        try:
            qa = torch.randint(INT8_MIN_VAL, INT8_MAX_VAL, (WARMUP_GEMM_DIM, MMAD_K_DIM), dtype=torch.int8, device=dev)
            qb = torch.randint(INT8_MIN_VAL, INT8_MAX_VAL, (MMAD_K_DIM, WARMUP_GEMM_DIM), dtype=torch.int8, device=dev)
            sc = torch.ones((WARMUP_GEMM_DIM,), dtype=torch.float32, device=dev)
            mm_modes["int8"] = [(mm_flops / (t * MS_PER_S)) / max(1, aic_count) for t in time_ms(
                lambda _i: torch_npu.npu_quant_matmul(qa, qb, sc, output_dtype=torch.bfloat16), inner=BF16_BYTES, blocker=True)]
            del qa, qb, sc
        except Exception:
            pass
        if arch_preset == "npu_arch_3510" and hasattr(torch, "float8_e4m3fn"):
            try:
                fa = torch.randn((WARMUP_GEMM_DIM, MMAD_K_DIM), dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn)
                fb = torch.randn((MMAD_K_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn)
                fsc = torch.ones((WARMUP_GEMM_DIM,), dtype=torch.float32, device=dev)
                _ = torch_npu.npu_quant_matmul(fa, fb, fsc, output_dtype=torch.bfloat16)
                torch.npu.synchronize()
                mm_modes["fp8_e4m3"] = [(mm_flops / (t * MS_PER_S)) / max(1, aic_count) for t in time_ms(
                    lambda _i: torch_npu.npu_quant_matmul(fa, fb, fsc, output_dtype=torch.bfloat16), inner=BF16_BYTES, blocker=True)]
                del fa, fb, fsc
            except Exception:
                pass
    sm_cold_ns: dict[str, float] = {}
    sm_warm_ns: dict[str, float] = {}
    w_bytes = WARMUP_GEMM_DIM * WARMUP_GEMM_DIM * BF16_BYTES
    w_slots = max(SMALL_M_RING_MIN_SLOTS, math.ceil((SMALL_M_L2_RING_MULT * l2_bytes) / max(1, w_bytes)))
    mb_ring = [torch.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev) for _ in range(w_slots)]
    for mv in SMALL_M_GRID:
        ma_sm = torch.randn((mv, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev)
        sm_cold_ns[str(mv)] = round(statistics.median(time_ms(lambda i, _a=ma_sm: torch.matmul(_a, mb_ring[i % w_slots]), inner=1, blocker=True)) * NS_PER_MS, SMALL_M_ROUND_DIGITS)
        sm_warm_ns[str(mv)] = round(statistics.median(time_ms(lambda _i, _a=ma_sm: torch.matmul(_a, mb_ring[0]), inner=FOUR, blocker=True)) * NS_PER_MS, SMALL_M_ROUND_DIGITS)
        del ma_sm
    del mb_ring
    torch.npu.empty_cache()

    va = torch.randn((VEC_DIM, VEC_DIM), dtype=torch.float32, device=dev)
    vb = torch.randn((VEC_DIM, VEC_DIM), dtype=torch.float32, device=dev)
    vc = torch.empty_like(va)
    vc_bf16 = torch.empty((VEC_DIM, VEC_DIM), dtype=torch.bfloat16, device=dev)
    elems = VEC_DIM * VEC_DIM
    triton_vec_alu = _probe_vec_alu_triton(dev, aiv_count, time_ms)
    p07 = {"vec_alu_ops_per_s_per_aiv": triton_vec_alu if triton_vec_alu is not None else [
        (float(elems) / (t * MS_PER_S)) / max(1, aiv_count) for t in time_ms(lambda _i: torch.addcmul(va, va, vb, out=vc), inner=EIGHT, blocker=True)]}
    p08 = {"vec_sfu_ops_per_s_per_aiv": [(elems / (t * MS_PER_S)) / max(1, aiv_count) for t in time_ms(lambda _i: torch.exp(va, out=vc), inner=EIGHT, blocker=True)]}
    p09 = {
        "vec_reduce_ops_per_s_per_aiv": [(elems / (t * MS_PER_S)) / max(1, aiv_count) for t in time_ms(lambda _i: torch.sum(va, dim=-1), inner=EIGHT, blocker=True)],
        "cast_bytes_per_s_per_aiv": [(elems * CAST_RW_BYTES / (t * MS_PER_S)) / max(1, aiv_count) for t in time_ms(lambda _i: vc_bf16.copy_(va), inner=EIGHT, blocker=True)],
    }
    del va, vb, vc, vc_bf16
    sx = torch.randn((SORT_ROWS, SORT_COLS), dtype=torch.float32, device=dev)
    p10 = {"sort_elems_per_s_per_aiv": [(SORT_ROWS * SORT_COLS) / (t * MS_PER_S) / max(1, aiv_count) for t in time_ms(lambda _i: torch.topk(sx, k=SORT_TOPK, dim=-1), inner=FOUR, blocker=True)]}
    del sx
    torch.npu.empty_cache()
    return {
        "P06_mmad": {"gemm_e2e_modes": mm_modes, "small_m_bf16_ns": sm_cold_ns, "small_m_bf16_warm_ns": sm_warm_ns},
        "P07_vec_alu": p07, "P08_vec_sfu": p08, "P09_vec_reduce_cast": p09, "P10_sort_topk": p10,
    }


def _probe_runtime_and_host_dma(
    torch: Any, dev: Any, *, warmup: int, repeats: int, time_ms: Any,
) -> dict[str, Any]:
    ta = torch.ones((1,), dtype=torch.bfloat16, device=dev)
    tb = torch.ones((1,), dtype=torch.bfloat16, device=dev)
    to = torch.empty((1,), dtype=torch.bfloat16, device=dev)
    aa = torch.ones((1, TINY_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    ab = torch.ones((TINY_GEMM_DIM, TINY_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    for _ in range(warmup * BF16_BYTES):
        torch.add(ta, tb, out=to)
        _ = torch.matmul(aa, ab)
    torch.npu.synchronize()
    host_ns: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(HOST_SUBMIT_BURST):
            torch.add(ta, tb, out=to)
        host_ns.append((time.perf_counter_ns() - t0) / float(HOST_SUBMIT_BURST))
    torch.npu.synchronize()
    def _first_issue_ns(fn: Any) -> list[float]:
        raw_ms = time_ms(lambda _i: fn(), inner=P15_FIRST_INNER, blocker=True, chunk_size=1, target_ms=P15_BLOCKER_TARGET_MS)
        out_ns: list[float] = []
        for pos in range(0, len(raw_ms), BF16_BYTES):
            grp = raw_ms[pos : pos + BF16_BYTES]
            if grp:
                out_ns.append((sum(grp) / float(len(grp))) * NS_PER_MS)
        return out_ns
    aiv_f = _first_issue_ns(lambda: torch.add(ta, tb, out=to))
    aic_f = _first_issue_ns(lambda: torch.matmul(aa, ab))
    b16 = time_ms(lambda _i: torch.add(ta, tb, out=to), inner=P15_SMALL_BURST, blocker=True, chunk_size=SMALL_CHUNK_REPEATS, target_ms=P15_BLOCKER_TARGET_MS)
    b64 = time_ms(lambda _i: torch.add(ta, tb, out=to), inner=P15_LARGE_BURST, blocker=True, chunk_size=SMALL_CHUNK_REPEATS, target_ms=P15_BLOCKER_TARGET_MS)
    q_ns = [((t64 * float(P15_LARGE_BURST)) - (t16 * float(P15_SMALL_BURST))) / P15_MARGINAL_DENOM * NS_PER_MS for t16, t64 in zip(b16, b64)]

    s_alt = torch.npu.Stream(device=dev)
    ev_sync, ev_back = torch.npu.Event(enable_timing=False), torch.npu.Event(enable_timing=False)
    def _sync_rt(_i: int) -> None:
        ev_sync.record()
        s_alt.wait_event(ev_sync)
        ev_back.record(s_alt)
        torch.npu.current_stream(dev).wait_event(ev_back)
    for idx in range(warmup * BF16_BYTES):
        _sync_rt(idx)
    torch.npu.synchronize()
    se_ns = [t * P16_HALF_NS_SCALE for t in time_ms(_sync_rt, inner=P16_INNER_STEPS, blocker=True, chunk_size=SMALL_CHUNK_REPEATS, target_ms=P15_BLOCKER_TARGET_MS)]

    dma_bytes = DMA_SLAB_MIB * MIB
    h_pin = torch.ones((dma_bytes // FP32_BYTES,), dtype=torch.float32).pin_memory()
    h_page = torch.ones((dma_bytes // FP32_BYTES,), dtype=torch.float32)
    d_buf = torch.empty((dma_bytes // FP32_BYTES,), dtype=torch.float32, device=dev)
    sp_pin = torch.ones((DMA_SMALL_ELEMS,), dtype=torch.float32).pin_memory()
    sp_page = torch.ones((DMA_SMALL_ELEMS,), dtype=torch.float32)
    sp_dev = torch.empty((DMA_SMALL_ELEMS,), dtype=torch.float32, device=dev)
    for _ in range(SIXTEEN):
        d_buf.copy_(h_pin, non_blocking=False)
        h_pin.copy_(d_buf, non_blocking=False)
    torch.npu.synchronize()
    def _single_dma_ns(fn: Any, sync: bool = True) -> list[float]:
        for _ in range(warmup):
            fn()
            if sync:
                torch.npu.synchronize()
        res: list[float] = []
        for _ in range(repeats):
            acc = 0.0
            for _ in range(EIGHT):
                t0 = time.perf_counter_ns()
                fn()
                if sync:
                    torch.npu.synchronize()
                acc += float(time.perf_counter_ns() - t0)
            if acc > 0.0:
                res.append(acc / float(EIGHT))
        return res
    def _fit_dma_pair(t_big_ns: list[float], t_sml_ns: list[float]) -> tuple[list[float], list[float]]:
        bps_list, lat_list = [], []
        for tb_ns, ts_ns in zip(t_big_ns, t_sml_ns):
            dt_s = (tb_ns - ts_ns) / NS_PER_S
            if dt_s <= 0.0:
                bps_list.append(-1.0)
                lat_list.append(-1.0)
                continue
            rate = float(dma_bytes - DMA_SMALL_BYTES) / dt_s
            lat = ts_ns - (float(DMA_SMALL_BYTES) / rate) * NS_PER_S
            bps_list.append(rate)
            lat_list.append(lat)
        return bps_list, lat_list
    h2d_pin_big = _single_dma_ns(lambda: d_buf.copy_(h_pin, non_blocking=True))
    h2d_pin_sml = _single_dma_ns(lambda: sp_dev.copy_(sp_pin, non_blocking=True))
    d2h_pin_big = _single_dma_ns(lambda: h_pin.copy_(d_buf, non_blocking=True))
    d2h_pin_sml = _single_dma_ns(lambda: sp_pin.copy_(sp_dev, non_blocking=True))
    h2d_pg_big = _single_dma_ns(lambda: d_buf.copy_(h_page, non_blocking=False))
    h2d_pg_sml = _single_dma_ns(lambda: sp_dev.copy_(sp_page, non_blocking=False))
    d2h_pg_big = _single_dma_ns(lambda: h_page.copy_(d_buf, non_blocking=False))
    d2h_pg_sml = _single_dma_ns(lambda: sp_page.copy_(sp_dev, non_blocking=False))
    h2d_bps, h2d_pin_lat = _fit_dma_pair(h2d_pin_big, h2d_pin_sml)
    d2h_bps, d2h_pin_lat = _fit_dma_pair(d2h_pin_big, d2h_pin_sml)
    _, h2d_pg_lat = _fit_dma_pair(h2d_pg_big, h2d_pg_sml)
    _, d2h_pg_lat = _fit_dma_pair(d2h_pg_big, d2h_pg_sml)
    p20 = {
        "host_h2d_bytes_per_s": h2d_bps, "host_d2h_bytes_per_s": d2h_bps,
        "pinned_h2d_startup_ns": h2d_pin_lat, "pinned_d2h_startup_ns": d2h_pin_lat,
        "pageable_h2d_startup_ns": h2d_pg_lat, "pageable_d2h_startup_ns": d2h_pg_lat,
        "pinned_dma_startup_ns": h2d_pin_lat, "pageable_dma_startup_ns": h2d_pg_lat,
        "eager_d2h_sync_ns": _single_dma_ns(lambda: float(ta.item()), sync=False),
    }
    del h_pin, h_page, d_buf, sp_pin, sp_page, sp_dev, ta, tb, to, aa, ab
    torch.npu.empty_cache()
    return {
        "P15_rts_stars": {"host_submit_ns": host_ns, "issue_interval_ns": q_ns, "aiv_first_issue_ns": aiv_f, "aic_first_issue_ns": aic_f},
        "P16_multistream_core_control": {"stream_event_ns": se_ns},
        "P20_host_dma_sync": p20,
    }


def run_local_npu_microbenchmarks(
    *, device_id: int, l2_bytes: int, aic_count: int, aiv_count: int,
    arch_preset: str, warmup: int = DEFAULT_WARMUP, repeats: int = DEFAULT_REPEATS,
    peer_device_id: int = -1, primitives: str = "all",
) -> dict[str, Any]:
    """Execute bounded live Ascend NPU micro-benchmarks on one (or two) idle dies."""
    import torch
    import torch_npu

    sel = _parse_selected_primitives(primitives)
    torch.npu.set_device(device_id)
    dev = torch.device(f"npu:{device_id}")
    soc_name = str(torch.npu.get_device_name(device_id))
    env_info = _detect_runtime_toolchain(soc_name)

    wa = torch.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    wb = torch.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch.bfloat16, device=dev)
    for _ in range(WARMUP_GEMM_ITERS):
        _ = torch.matmul(wa, wb)
    torch.npu.synchronize()

    s0, e0 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s0.record(); _ = torch.matmul(wa, wb); e0.record()
    torch.npu.synchronize()
    single_gemm_ms = max(MIN_SINGLE_GEMM_MS, float(s0.elapsed_time(e0)))
    blocker_reps = max(1, math.ceil(BLOCKER_TARGET_MS / single_gemm_ms))

    def _time_ms(
        fn: Any, *, inner: int = 1, blocker: bool = True,
        chunk_size: int = BLOCKER_CHUNK_REPEATS, target_ms: float = BLOCKER_TARGET_MS,
    ) -> list[float]:
        for i in range(warmup):
            fn(i)
        torch.npu.synchronize()
        out: list[float] = []
        step = max(1, chunk_size) if blocker else 1
        for start in range(0, repeats, step):
            k_rep = min(step, repeats - start)
            evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(k_rep)]
            if blocker:
                reps = max(blocker_reps, math.ceil((target_ms + (k_rep + BLOCKER_SETTLE_ITERS) * inner * DISPATCH_MARGIN_MS) / single_gemm_ms))
                for _ in range(reps):
                    _ = torch.matmul(wa, wb)
                for s_off in range(BLOCKER_SETTLE_ITERS):
                    fn((start + s_off) * inner)
            for off, (se, ee) in enumerate(evs):
                idx = start + off
                se.record()
                for k in range(inner):
                    fn((idx + BLOCKER_SETTLE_ITERS) * inner + k)
                ee.record()
            torch.npu.synchronize()
            out.extend(float(se.elapsed_time(ee)) / float(inner) for se, ee in evs if float(se.elapsed_time(ee)) > 0.0)
        return out

    raw: dict[str, Any] = {
        "device_name": soc_name, "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__), "blocker_reps": blocker_reps,
        "selected_primitives": sorted(sel) if sel is not None else "all", **env_info,
    }
    slab, rw11_ms, hbm_w = COLD_SLAB_MIB * MIB, [], []
    if sel is None or bool(sel & {"P01", "P04", "P05"}):
        p01, p04, p05, slab, rw11_ms, hbm_w = _probe_hbm_l2_domain(
            torch, dev, l2_bytes=l2_bytes, aiv_count=aiv_count, repeats=repeats, time_ms=_time_ms,
        )
        raw["P01_datacopy"], raw["P04_mte3_store"], raw["P05_l2_residency_reuse"] = p01, p04, p05
    if sel is None or bool(sel & {"P06", "P07", "P08", "P09", "P10"}):
        raw.update(_probe_compute_domain(
            torch, torch_npu, dev, aic_count=aic_count, aiv_count=aiv_count, l2_bytes=l2_bytes, arch_preset=arch_preset, time_ms=_time_ms,
        ))
    if sel is None or bool(sel & {"P15", "P16", "P20"}):
        raw.update(_probe_runtime_and_host_dma(torch, dev, warmup=warmup, repeats=repeats, time_ms=_time_ms))
    del wa, wb
    torch.npu.empty_cache()

    if shutil.which("bisheng") is not None:
        try:
            script_dir = str(Path(__file__).resolve().parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from _bench_l0_ascendc import run_ascendc_suite

            ac = run_ascendc_suite(
                device_id=device_id, arch_preset=arch_preset, aic_count=aic_count,
                aiv_count=aiv_count, l2_bytes=l2_bytes, repeats=min(repeats, DEFAULT_REPEATS),
                selected_primitives=sel,
            )
            _merge_ascendc_into_raw(raw, ac, slab=slab, rw11_ms=rw11_ms, hbm_w=hbm_w)
        except Exception as exc:
            raw["ascendc_probe_error"] = f"{type(exc).__name__}:{str(exc).splitlines()[0][:ERR_PREVIEW_CHARS]}"

    if peer_device_id >= 0 and peer_device_id != device_id and (sel is None or bool(sel & {"P18", "P19"})):
        try:
            script_dir = str(Path(__file__).resolve().parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from _bench_l0_hccl import run_hccl_two_rank

            raw.update(run_hccl_two_rank(device_id, peer_device_id, repeats, arch_preset))
        except Exception as exc:
            raw["hccl_probe_error"] = f"{type(exc).__name__}:{str(exc).splitlines()[0][:ERR_PREVIEW_CHARS]}"
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Remote Ascend NPU L0 micro-benchmark probe worker")
    for flag, default in (
        ("--device-id", 0), ("--peer-device-id", -1), ("--l2-bytes", COLD_SLAB_MIB * MIB),
        ("--aic-count", SIXTEEN * BF16_BYTES), ("--aiv-count", SIXTEEN * FP32_BYTES),
        ("--warmup", DEFAULT_WARMUP), ("--repeats", DEFAULT_REPEATS),
    ):
        ap.add_argument(flag, type=int, default=default)
    ap.add_argument("--arch", default="npu_arch_2201")
    ap.add_argument("--primitives", default="all")
    args = ap.parse_args(argv)
    raw = run_local_npu_microbenchmarks(
        device_id=args.device_id, peer_device_id=args.peer_device_id,
        l2_bytes=args.l2_bytes, aic_count=args.aic_count, aiv_count=args.aiv_count,
        arch_preset=args.arch, warmup=args.warmup, repeats=args.repeats,
        primitives=args.primitives,
    )
    print(RESULT_MARKER + json.dumps(raw, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
