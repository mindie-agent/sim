#!/usr/bin/env python3
"""Live NPU 1-layer synthetic subgraph runner with ``torch_npu.profiler`` CSV capture.

[Role]
  Instantiate 1 single transformer layer (Attention / Linear-Attention subgraph
  chained into MoE / Dense FFN subgraph with real residual dataflow) from
  catalog model dimensions using bounded synthetic random weights (< 4-6 GiB/card,
  zero checkpoint downloads) and synthetic Paged KV / Indexer / SSM tables on live
  Ascend NPUs (A2, A3, A5-DT), capturing ``kernel_details.csv`` traces, full-rank
  receipts, interval-union busy spans, and AICore frequencies.

[Invariants]
  * Zero device context at module import time: ``torch`` and ``torch_npu`` are
    imported strictly inside device execution entry points.
  * Pre-checks free HBM before allocation, locks AICore DVFS with 30x 4096^2
    BF16 GEMMs, and uses a verified 4-GEMM prefix StreamBlocker to eliminate
    Eager CPU host-dispatch bubbles without stripping hardware wait time.
  * Reserves ``live_vllm_1layer_verified`` strictly for native vllm-ascend layers;
    standalone torch_npu/fla_npu composite execution reports
    ``live_torchnpu_composite_verified`` and classifies genuinely unsupported ops
    as ``unmerged_op_container_absent`` strictly from capability checks.
  * Emits strictly sanitized metadata (never raw hostnames, IPs, usernames, or
    private filesystem paths) with full-rank receipts and ``shard_complete`` flag.

[Timeouts & Locks]
  Enforces ``HCCL_INIT_TIMEOUT_SEC`` on process groups, rank-readiness agreement
  before multi-rank collectives, bounded iterations, and temp dir cleanup.

[Side-Effect Policy]
  Allocates bounded synthetic tensors on visible NPU device(s) and writes only
  the requested ``--output`` JSON capture bundle via atomic ``os.replace``.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gc
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
import tempfile
import time
from typing import Any, Callable

SCHEMA_VERSION = "mindie-sim-1layer-live-capture/2"
TIER_VLLM_1LAYER = "live_vllm_1layer_verified"
TIER_TORCHNPU_COMPOSITE = "live_torchnpu_composite_verified"
TIER_UNMERGED_ABSENT = "unmerged_op_container_absent"
TIER_RUNNER_UNSUPPORTED = "runner_overlay_unsupported"
RUNNER_MODE = "torchnpu_composite"

WARMUP_GEMM_ITERS = 30
WARMUP_GEMM_DIM = 4096
STREAM_BLOCKER_M = 4112
STREAM_BLOCKER_N = 4128
STREAM_BLOCKER_K = 4096
STREAM_BLOCKER_ITERS = 4
DVFS_LOCK_PRE_ITERS = 24
FREQ_PROBE_LOAD_ITERS = 200
STREAM_BLOCKER_SHAPE_TAG = f"{STREAM_BLOCKER_M},{STREAM_BLOCKER_K}"
STREAM_BLOCKER_EXACT_PREFIX = f"{STREAM_BLOCKER_M},{STREAM_BLOCKER_K};{STREAM_BLOCKER_K},{STREAM_BLOCKER_N}"
FREQ_PROBE_TIMEOUT_SEC = 5
SUBGRAPH_WARMUP_ITERS = 3
SUBGRAPH_TIMED_ITERS = 7
MS_TO_US = 1000.0
MIB = 1024 * 1024
GIB = 1024 * MIB
MIN_FREE_HBM_BYTES = 2 * GIB
HBM_HEADROOM_BYTES = 512 * MIB
L2_FLUSH_BYTES = 256 * MIB
MAX_SYNTH_WEIGHT_BYTES = 6 * GIB
DEFAULT_KV_BLOCK_SIZE = 128
DEFAULT_RMS_EPS = 1e-6
ROPE_PAD_DIM = 64
ROPE_THETA_BASE = 10000.0
MX_GROUP_SIZE = 32
DEFAULT_CAUSAL_MASK_DIM = 2048
HCCL_INIT_TIMEOUT_SEC = 120
KDA_LOWER_BOUND = -5.0
KDA_CHUNK_SIZE = 64
GQA_SWA_SPARSE_MODE = 4
CAUSAL_SPARSE_MODE = 3
NONCAUSAL_SPARSE_MODE = 0
SFA_ATTN_MODE_MLA = 2
SFA_ATTN_MODE_GQA = 0
GMM_SPLIT_ITEM_3 = 3
GMM_GROUP_LIST_COUNT = 1
INT8_MIN_VAL = -127
INT8_MAX_VAL = 127
INT4_MIN_VAL = -7
INT4_MAX_VAL = 7
INT8_SCALE_NORM = 127.0
COLD_RING_MIN_SLOTS = 3
VALID_ATTENTIONS = frozenset({"gqa", "mla", "gdn", "kda"})
VALID_FFNS = frozenset({"dense", "moe"})
VALID_QUANTS = frozenset({"bf16", "w8a8", "w4a8", "c8", "mxfp8", "mxfp4"})
VALID_PHASES = frozenset({"decode", "prefill"})
CAPABILITY_ABSENT_MARKERS = (
    "161001", "561103", "aclnn api is not supported", "not supported on current soc",
    "not implemented for", "has no attribute", "does not support",
)
CAPABILITY_NON_ABSENT_EXCLUSIONS = (
    "out of memory", "oom", "alloc", "memory", "illegal", "timeout", "hccl", "nccl",
    "communication", "written-to tensor", "single memory location", "memory overlap",
    "shape mismatch", "size of tensor", "must match the size", "invalid device",
)
HCCL_DTYPE_BYTES: dict[str, float] = {
    "FP32": 4.0, "FLOAT": 4.0, "FLOAT32": 4.0, "FP16": 2.0, "FLOAT16": 2.0,
    "BFP16": 2.0, "BF16": 2.0, "INT8": 1.0, "UINT8": 1.0, "INT32": 4.0,
    "INT64": 8.0, "FP8": 1.0, "FP8_E4M3": 1.0, "FP8_E5M2": 1.0, "INT16": 2.0,
}
FP32_BYTES = 4
BF16_BYTES = 2
DEFAULT_ENGRAM_WIDTH = 256
CONTROL_HELPER_KERNEL_TYPES = frozenset({
    "hccllaunchaicpukernel", "memset", "memcpyasync",
    "notifywait", "notifyrecord", "eventwait", "eventrecord",
})


class CapabilityAbsentError(RuntimeError):
    """Raised ONLY when a live runtime probe confirms an op/quant capability is absent from the container."""


class RunnerUnsupportedOverlayError(RuntimeError):
    """Raised when a feature overlay is not implemented by this synthetic runner (never blamed on container)."""


def _sanitize_text(text: str) -> str:
    """Strip paths (including spaces), emails, non-loopback IPv4/IPv6 (all-letter & zoned), and hex digests."""
    out = re.sub(r"/(?:Users|home|root)/[^\r\n'\"\]\[;,{}()<>]+", "/<REDACTED_HOME>", text)
    out = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "<REDACTED_EMAIL>", out)
    # Full 4-8 group IPv6 (including all-letter hex and immediately after Exception:)
    out = re.sub(
        r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        "<REDACTED_IPV6>", out,
    )
    # Compressed :: IPv6 with >=3 colons (e.g. abcd:efab::cdef:abcd, fe80::1:2)
    out = re.sub(
        r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{1,4}:){2,6}:(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*)?(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        "<REDACTED_IPV6>", out,
    )
    out = re.sub(
        r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{1,4}:)?:(?:[0-9a-fA-F]{1,4}:){1,6}[0-9a-fA-F]{1,4}(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        "<REDACTED_IPV6>", out,
    )
    # Short compressed :: IPv6 with digit or 4-hex-char group (e.g. ::1, fe80::1%eth0, abcd::efab)
    out = re.sub(
        r"(?<![0-9a-fA-F:])(?:[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4})(?:%[A-Za-z0-9_.-]+)?(?![0-9a-fA-F:])",
        lambda m: "<REDACTED_IPV6>" if (any(c.isdigit() for c in m.group(0)) or len(m.group(0).split("::")[0]) == 4 or len(m.group(0).split("::")[-1].split("%")[0]) == 4) else m.group(0),
        out,
    )
    out = re.sub(r"\b(?!127\.0\.0\.1\b)\d{1,3}(?:\.\d{1,3}){3}\b", "<REDACTED_IP>", out)
    digest_pat = r"@?" + "sha" + r"256:[0-9a-fA-F]{16,}"
    out = re.sub(digest_pat, "<REDACTED_DIGEST>", out)
    out = re.sub(r"\b[0-9a-fA-F]{64}\b", "<REDACTED_HEX64>", out)
    return out


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _parse_strict_float(val: Any, *, field_name: str, allow_na: bool = False) -> float:
    text = str(val if val is not None else "").strip().replace("\t", "")
    if not text or text == "N/A":
        if allow_na:
            return 0.0
        raise ValueError(f"missing_required_numeric_field:{field_name}")
    out = float(text)
    if not math.isfinite(out):
        raise ValueError(f"non_finite_numeric_field:{field_name}:{text}")
    return out


def _parse_int_opt(val: Any) -> int:
    text = str(val if val is not None else "").strip().replace("\t", "")
    if not text or text == "N/A":
        return 0
    return int(float(text))


def _parse_kernel_details_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read raw rows from CANN/torch_npu ``kernel_details.csv`` with strict numeric checks."""
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, raw in enumerate(reader):
            if not raw or all(not str(v or "").strip() for v in raw.values()):
                continue
            name = str(raw.get("Name", "")).strip() or f"unnamed_row_{row_idx}"
            op_type = str(raw.get("Type", "")).strip() or "UNKNOWN"
            start_us = _parse_strict_float(raw.get("Start Time(us)"), field_name="Start Time(us)", allow_na=True)
            dur_us = _parse_strict_float(raw.get("Duration(us)"), field_name="Duration(us)", allow_na=False)
            wait_us = _parse_strict_float(raw.get("Wait Time(us)"), field_name="Wait Time(us)", allow_na=True)
            row: dict[str, Any] = {
                "device_id": _parse_int_opt(raw.get("Device_id")),
                "task_id": _parse_int_opt(raw.get("Task ID")),
                "stream_id": _parse_int_opt(raw.get("Stream ID")),
                "name": name,
                "type": op_type,
                "task_type": str(raw.get("Task Type", "")).strip(),
                "core": str(raw.get("Accelerator Core", "")).strip(),
                "start_us": start_us,
                "duration_us": dur_us,
                "wait_us": wait_us,
                "block_num": _parse_int_opt(raw.get("Block Num")),
                "input_shapes": str(raw.get("Input Shapes", "")).strip().strip('"'),
                "input_dtypes": str(raw.get("Input Data Types", "")).strip().strip('"'),
                "output_shapes": str(raw.get("Output Shapes", "")).strip().strip('"'),
            }
            for pmu_key in (
                "aic_mac_time(us)", "aic_mte1_time(us)", "aic_mte2_time(us)",
                "aic_fixpipe_time(us)", "aiv_vec_time(us)", "aiv_mte2_time(us)",
                "aiv_mte3_time(us)",
            ):
                if pmu_key in raw and str(raw.get(pmu_key, "")).strip() not in ("", "N/A"):
                    row[pmu_key] = _parse_strict_float(raw.get(pmu_key), field_name=pmu_key, allow_na=True)
            rows.append(row)
    return rows


def _interval_union_us(kernels: list[dict[str, Any]]) -> float:
    """Compute exact merged timeline union of device-busy intervals in microseconds."""
    intervals = sorted(
        (k["start_us"], k["start_us"] + k["duration_us"])
        for k in kernels
        if k.get("duration_us", 0.0) > 0.0 and k.get("start_us", 0.0) > 0.0
    )
    if not intervals:
        return 0.0
    merged_lo, merged_hi = intervals[0]
    busy = 0.0
    for lo, hi in intervals[1:]:
        if lo <= merged_hi:
            merged_hi = max(merged_hi, hi)
        else:
            busy += merged_hi - merged_lo
            merged_lo, merged_hi = lo, hi
    busy += merged_hi - merged_lo
    return float(busy)


def _span_from_kernels(kernels: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute elapsed union span, interval-union busy time, uncovered gap, and wait sum."""
    valid = [k for k in kernels if k.get("duration_us", 0.0) > 0.0]
    if not valid:
        return {
            "elapsed_span_us": 0.0, "union_busy_us": 0.0, "uncovered_gap_us": 0.0,
            "sum_kernel_us": 0.0, "wait_us_sum": 0.0, "kernel_count": 0, "stream_count": 0,
        }
    starts = [k["start_us"] for k in valid if k["start_us"] > 0.0]
    ends = [k["start_us"] + k["duration_us"] for k in valid if k["start_us"] > 0.0]
    sum_us = float(sum(k["duration_us"] for k in valid))
    elapsed = (max(ends) - min(starts)) if starts and ends else sum_us
    union_busy = _interval_union_us(valid) if starts and ends else sum_us
    gap_us = max(0.0, elapsed - union_busy)
    return {
        "elapsed_span_us": round(float(elapsed), 4),
        "union_busy_us": round(float(union_busy), 4),
        "uncovered_gap_us": round(float(gap_us), 4),
        "sum_kernel_us": round(sum_us, 4),
        "wait_us_sum": round(float(sum(k.get("wait_us", 0.0) for k in valid)), 4),
        "kernel_count": len(valid),
        "stream_count": len({k["stream_id"] for k in valid}),
    }


def _is_blocker_matmul_row(row: dict[str, Any]) -> bool:
    op_type = str(row.get("type", "")).strip().lower()
    if "matmul" not in op_type:
        return False
    try:
        dur = float(row.get("duration_us", 0.0))
    except (TypeError, ValueError):
        return False
    if (not math.isfinite(dur)) or dur <= 0.0:
        return False
    shapes = str(row.get("input_shapes", "")).strip().strip('"')
    return shapes == STREAM_BLOCKER_EXACT_PREFIX or shapes.startswith(STREAM_BLOCKER_EXACT_PREFIX + ";")


def _is_control_helper_kernel(row: dict[str, Any]) -> bool:
    op_type = str(row.get("type", "")).strip().lower()
    task_type = str(row.get("task_type", "")).strip().upper()
    core_type = str(row.get("core", "")).strip().upper()
    if task_type == "COMMUNICATION" or core_type == "COMMUNICATION" or op_type.startswith("hcom_"):
        return False
    return op_type in CONTROL_HELPER_KERNEL_TYPES or op_type.endswith("aicpukernel")


def _is_prefix_allowed_non_blocker(row: dict[str, Any]) -> bool:
    op_type = str(row.get("type", "")).strip().lower()
    task_type = str(row.get("task_type", "")).strip().upper()
    core_type = str(row.get("core", "")).strip().upper()
    shapes = str(row.get("input_shapes", "")).strip().strip('"').lower()
    is_comm = task_type == "COMMUNICATION" or core_type == "COMMUNICATION" or op_type.startswith("hcom_")
    if (not is_comm) and op_type.endswith("aicpukernel") and shapes in ("", "n/a"):
        return True
    return (
        op_type in CONTROL_HELPER_KERNEL_TYPES
        or any(t in op_type for t in ("allreduce", "barrier", "hccl", "hcom"))
    )


def _strip_stream_blocker_rows(
    kernels: list[dict[str, Any]],
    *,
    expected_blockers: int = STREAM_BLOCKER_ITERS,
) -> tuple[list[dict[str, Any]], int, int]:
    """Remove ONLY the contiguous prefix barrier + exact ``expected_blockers`` StreamBlocker GEMMs.

    Requires exact MatMul type and exact ``4112,4096;4096,4128`` shape match and preserves ANY
    non-blocker row whose execution extends past the final blocker (even if interleaved between
    blocker GEMMs).
    """
    if expected_blockers <= 0:
        return kernels, 0, 0
    matched_indices: list[int] = []
    for idx, row in enumerate(kernels):
        if _is_blocker_matmul_row(row):
            matched_indices.append(idx)
            if len(matched_indices) == expected_blockers:
                break
        elif not _is_prefix_allowed_non_blocker(row):
            break
    if len(matched_indices) != expected_blockers:
        return kernels, 0, len(matched_indices)
    last_blk_idx = matched_indices[-1]
    last_blk = kernels[last_blk_idx]
    blk_end_us = float(last_blk.get("start_us", 0.0)) + float(last_blk.get("duration_us", 0.0))
    matched_set = set(matched_indices)
    # Identify trailing control helpers strictly contained inside the final blocker window
    tail_cut = last_blk_idx + 1
    while tail_cut < len(kernels):
        nxt = kernels[tail_cut]
        nxt_start = float(nxt.get("start_us", 0.0))
        nxt_dur = max(0.0, float(nxt.get("duration_us", 0.0)))
        if _is_control_helper_kernel(nxt) and blk_end_us > 0.0 and (nxt_start + nxt_dur) <= blk_end_us:
            tail_cut += 1
        else:
            break
    kept: list[dict[str, Any]] = []
    stripped = 0
    for idx, row in enumerate(kernels):
        if idx in matched_set:
            stripped += 1
            continue
        if idx < tail_cut:
            r_start = float(row.get("start_us", 0.0))
            r_dur = max(0.0, float(row.get("duration_us", 0.0)))
            extends_past_blocker = blk_end_us > 0.0 and (r_start + r_dur) > blk_end_us
            if extends_past_blocker:
                kept.append(row)
            else:
                stripped += 1
        else:
            kept.append(row)
    return kept, stripped, len(matched_indices)


def _detect_cann_version() -> str:
    for candidate in (
        "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info",
        "/usr/local/Ascend/cann/aarch64-linux/ascend_cann_install.info",
    ):
        path = Path(candidate)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("version="):
                return line.split("=", 1)[1].strip()
    return "unknown"


def _parse_npu_smi_chip_freqs(stdout: str) -> tuple[dict[int, float], float | None]:
    """Parse per-chip ``Aicore curFreq(MHZ)`` readings from ``npu-smi info -t common``."""
    chip_freqs: dict[int, float] = {}
    fallback_cur: float | None = None
    cur_chip: int | None = None
    for line in stdout.splitlines():
        low = line.lower()
        if "chip id" in low:
            m_c = re.search(r":\s*([0-9]+)", line)
            cur_chip = int(m_c.group(1)) if m_c else None
            continue
        if "aicore" in low and "curfreq" in low and "mhz" in low:
            m = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)", line)
            if m and float(m.group(1)) > 0.0:
                val = round(float(m.group(1)), 2)
                if cur_chip is not None:
                    chip_freqs[cur_chip] = val
                elif fallback_cur is None:
                    fallback_cur = val
    return chip_freqs, fallback_cur


def _query_npu_smi_common(dev_id: str) -> tuple[dict[int, float], float | None]:
    if not dev_id.isdigit():
        return {}, None
    try:
        res = subprocess.run(
            ["npu-smi", "info", "-t", "common", "-i", dev_id],
            capture_output=True, text=True, timeout=FREQ_PROBE_TIMEOUT_SEC, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}, None
    return _parse_npu_smi_chip_freqs(res.stdout)


def _detect_aicore_freq_mhz(rank: int = 0) -> float:
    """Query actual operating ``Aicore curFreq(MHZ)`` via ``npu-smi`` (dual-die A3 aware)."""
    vis_env = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    vis_list = [x.strip() for x in vis_env.split(",") if x.strip()] if vis_env else []
    raw_id = vis_list[rank] if rank < len(vis_list) else str(rank)
    if not raw_id.isdigit():
        return 0.0
    logical_id = int(raw_id)
    chip_freqs, fallback_cur = _query_npu_smi_common(raw_id)
    if len(chip_freqs) > 1:
        # Dual-die package (e.g. Ascend 910C/A3): logical die d maps to (pkg=d//2, chip=d%2).
        pkg_id, chip_id = logical_id // len(chip_freqs), logical_id % len(chip_freqs)
        if str(pkg_id) != raw_id:
            chip_freqs, fallback_cur = _query_npu_smi_common(str(pkg_id))
        if chip_id in chip_freqs:
            return chip_freqs[chip_id]
    if chip_freqs:
        return chip_freqs[sorted(chip_freqs)[0]]
    if fallback_cur is not None:
        return fallback_cur
    if logical_id > 0:
        chip_freqs2, fallback_cur2 = _query_npu_smi_common(str(logical_id // 2))
        chip_id = logical_id % 2
        if chip_id in chip_freqs2:
            return chip_freqs2[chip_id]
        if chip_freqs2:
            return chip_freqs2[sorted(chip_freqs2)[0]]
        if fallback_cur2 is not None:
            return fallback_cur2
    return 0.0


def _build_rank_receipt(rank: int, rec: dict[str, Any], freq_mhz: float = 0.0) -> dict[str, Any]:
    """Build compact per-rank execution & per-stream execution-ordered compute-shape digest receipt."""
    lc = dict(rec.get("layer_capture") or {})
    kernels = list(lc.get("kernels") or [])
    by_stream: dict[int, list[tuple[float, int, dict[str, Any]]]] = {}
    for orig_idx, k in enumerate(kernels):
        if _is_control_helper_kernel(k):
            continue
        st_us = float(k.get("start_us", 0.0)) if isinstance(k.get("start_us"), (int, float)) else 0.0
        by_stream.setdefault(int(k.get("stream_id", 0)), []).append((st_us, orig_idx, k))
    stream_det_sigs: list[tuple[str, ...]] = []
    stream_op_sigs: list[tuple[str, ...]] = []
    for rows in by_stream.values():
        rows.sort(key=lambda t: (t[0], t[1]))
        stream_det_sigs.append(tuple(
            f"{k.get('type', '')}|{k.get('input_shapes', '')}|{k.get('input_dtypes', '')}" for _, _, k in rows
        ))
        stream_op_sigs.append(tuple(
            f"{k.get('type', '')}|{k.get('input_dtypes', '')}" for _, _, k in rows
        ))
    stream_det_sigs.sort()
    stream_op_sigs.sort()
    det_sig = [line for seq in stream_det_sigs for line in (*seq, "--")]
    op_sig = [line for seq in stream_op_sigs for line in (*seq, "--")]
    compute_k_cnt = sum(len(seq) for seq in stream_det_sigs)
    digest = hashlib.sha256("\n".join(det_sig).encode("utf-8")).hexdigest()[:16]
    op_digest = hashlib.sha256("\n".join(op_sig).encode("utf-8")).hexdigest()[:16]
    return {
        "rank": int(rank),
        "status": str(rec.get("status", "unknown")),
        "coverage_tier": str(rec.get("coverage_tier", TIER_UNMERGED_ABSENT)),
        "stream_blocked": bool(lc.get("stream_blocked", False)),
        "blocker_matched_count": int(lc.get("blocker_matched_count", 0)),
        "kernel_count": int(lc.get("kernel_count", len(kernels))),
        "compute_kernel_count": int(compute_k_cnt),
        "stream_count": int(lc.get("stream_count", 0)),
        "elapsed_span_us": float(lc.get("elapsed_span_us", 0.0)),
        "union_busy_us": float(lc.get("union_busy_us", 0.0)),
        "uncovered_gap_us": float(lc.get("uncovered_gap_us", 0.0)),
        "sum_kernel_us": float(lc.get("sum_kernel_us", 0.0)),
        "wait_us_sum": float(lc.get("wait_us_sum", 0.0)),
        "observed_aicore_freq_mhz": float(freq_mhz),
        "shape_digest": digest,
        "op_type_digest": op_digest,
    }


def _lock_aic_frequency(torch_mod: Any, device: str) -> None:
    """Run 30x 4096^2 BF16 GEMMs to lock AIC out of idle DVFS frequency."""
    a = torch_mod.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch_mod.bfloat16, device=device)
    b = torch_mod.randn((WARMUP_GEMM_DIM, WARMUP_GEMM_DIM), dtype=torch_mod.bfloat16, device=device)
    for _ in range(WARMUP_GEMM_ITERS):
        _ = torch_mod.matmul(a, b)
    torch_mod.npu.synchronize()
    del a, b


def _estimate_case_allocation_bytes(spec: dict[str, Any]) -> int:
    """Estimate peak synthetic weight + KV cache allocation bytes for HBM preflight."""
    hidden = int(spec["hidden_size"])
    tp = max(1, int(spec.get("tp", 1)))
    ep = max(1, int(spec.get("ep", 1)))
    dcp = max(1, int(spec.get("dcp", 1)))
    quant_mode = str(spec.get("quant_mode", "bf16"))
    elem_b = 1 if quant_mode == "w8a8" else 2
    est = 4 * hidden * hidden * elem_b // tp
    max_single_tensor = 0
    if str(spec.get("ffn", "dense")) == "moe" and int(spec.get("num_experts", 0)) > 0:
        local_e = max(1, math.ceil(int(spec["num_experts"]) / ep))
        moe_i = int(spec.get("moe_intermediate_size", hidden))
        w1_bytes = local_e * 2 * hidden * moe_i * elem_b
        w2_bytes = local_e * hidden * moe_i * elem_b
        est += w1_bytes + w2_bytes
        max_single_tensor = max(max_single_tensor, w1_bytes)
        shared_n = int(spec.get("num_shared_experts", 0))
        shared_i = int(spec.get("shared_intermediate_size", moe_i))
        est += shared_n * 3 * hidden * shared_i * elem_b
    else:
        inter = max(1, int(spec.get("intermediate_size", hidden * 4)) // tp)
        w1_bytes = 2 * hidden * inter * elem_b
        est += 3 * hidden * inter * elem_b
        max_single_tensor = max(max_single_tensor, w1_bytes)
    batch = int(spec.get("sequences", 4))
    q_len = int(spec.get("q_len", 1))
    ctx = int(spec.get("context", 2048)) // dcp
    attn_kind = str(spec.get("attention", ""))
    if attn_kind in ("gdn", "kda"):
        lin_tp = max(1, int(spec.get("linear_proj_tp", 1)))
        proj_div = max(tp, lin_tp)
        nv = max(
            1,
            int(
                spec.get("linear_num_v_heads")
                or spec.get("linear_value_heads")
                or spec.get("linear_heads")
                or 1
            ) // proj_div,
        )
        dk = int(
            spec.get("linear_k_head_dim")
            or spec.get("linear_key_head_dim")
            or spec.get("linear_head_dim")
            or DEFAULT_KV_BLOCK_SIZE
        )
        dv = int(spec.get("linear_v_head_dim") or spec.get("linear_head_dim") or DEFAULT_KV_BLOCK_SIZE)
        phase_s = str(spec.get("phase", "decode"))
        state_slots = 1 if phase_s == "prefill" else (batch + COLD_RING_MIN_SLOTS)
        state_bytes = state_slots * nv * dv * dk * FP32_BYTES
        gate_bytes = batch * q_len * nv * dv * BF16_BYTES
        est += state_bytes + gate_bytes
        max_single_tensor = max(max_single_tensor, state_bytes)
    else:
        kv_width = int(spec.get("kv_lora_rank", 0)) if attn_kind == "mla" else (
            2 * max(1, int(spec.get("num_kv_heads", 1)) // tp) * int(spec.get("head_dim", DEFAULT_KV_BLOCK_SIZE))
        )
        kv_bytes = batch * max(DEFAULT_KV_BLOCK_SIZE, ctx) * kv_width * BF16_BYTES
        est += kv_bytes
        max_single_tensor = max(max_single_tensor, kv_bytes)
    # Out-of-place scaling (randn(...) * scale) peaks with 2x the largest tensor simultaneously resident
    return int(est + max_single_tensor)


def _check_free_hbm(torch_mod: Any, dev_index: int, *, required_bytes: int = MIN_FREE_HBM_BYTES) -> tuple[int, int]:
    free_b, total_b = torch_mod.npu.mem_get_info(dev_index)
    need = max(MIN_FREE_HBM_BYTES, int(required_bytes))
    if int(free_b) < need:
        raise MemoryError(f"insufficient_free_hbm:{free_b}<{need}")
    return int(free_b), int(total_b)


def _enrich_hccl_rows_from_trace_view(tmp_dir: str, rows: list[dict[str, Any]]) -> None:
    """Annotate HCCL ``kernel_details.csv`` rows with real ``rank_size`` and ``payload_bytes`` from ``trace_view.json``."""
    comm_rows = [
        r for r in rows
        if str(r.get("core", "")).strip().upper() == "COMMUNICATION"
        or str(r.get("task_type", "")).strip().upper() == "COMMUNICATION"
        or str(r.get("type", "")).strip().lower().startswith("hcom_")
    ]
    if not comm_rows:
        return
    trace_files = sorted(glob.glob(os.path.join(tmp_dir, "**", "trace_view.json"), recursive=True))
    if not trace_files:
        return
    by_name: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for tf in trace_files:
        try:
            events = json.loads(Path(tf).read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            args = ev.get("args")
            if not isinstance(args, dict) or "rank_size" not in args:
                continue
            name = str(ev.get("name", "")).strip()
            if not name:
                continue
            try:
                ts_us = float(ev.get("ts", 0.0))
            except (TypeError, ValueError):
                ts_us = 0.0
            by_name.setdefault(name, []).append((ts_us, args))
    for row in comm_rows:
        cands = by_name.get(str(row.get("name", "")).strip())
        if not cands:
            continue
        r_start = float(row.get("start_us", 0.0))
        _, best_args = min(cands, key=lambda pair: abs(pair[0] - r_start))
        try:
            r_size = int(best_args["rank_size"])
        except (KeyError, TypeError, ValueError):
            r_size = 0
        raw_cnt = best_args.get("count")
        try:
            cnt = int(raw_cnt) if raw_cnt is not None else 0
        except (TypeError, ValueError):
            cnt = 0
        dt_str = str(best_args.get("data_type", "")).strip().upper()
        elem_b = HCCL_DTYPE_BYTES.get(dt_str, 0.0)
        if r_size >= 2:
            row["comm_group_size"] = r_size
        if cnt > 0 and elem_b > 0.0:
            row["comm_count"] = cnt
            row["comm_data_type"] = dt_str
            row["payload_bytes"] = int(round(cnt * elem_b))


def _cpu_store_barrier(store: Any, key: str, world_size: int) -> None:
    """Zero-NPU-kernel host TCPStore barrier aligning ranks after profiler startup."""
    if store is None or world_size <= 1:
        return
    store.add(key, 1)
    deadline = time.monotonic() + HCCL_INIT_TIMEOUT_SEC
    while int(store.get(key)) < world_size:
        if time.monotonic() > deadline:
            raise TimeoutError(f"cpu_store_barrier_timeout:{key}")
        time.sleep(0.0005)


def _sample_loaded_aicore_freq(torch_mod: Any, blocker_pair: tuple[Any, Any] | None, rank: int) -> float:
    """Keep AICore continuously saturated across the ~50 ms ``npu-smi`` clock probe."""
    if blocker_pair is not None:
        for _ in range(FREQ_PROBE_LOAD_ITERS):
            _ = torch_mod.matmul(blocker_pair[0], blocker_pair[1])
    freq_mhz = _detect_aicore_freq_mhz(rank)
    torch_mod.npu.synchronize()
    return freq_mhz


def _alloc_l2_flush_pair(torch_mod: Any, device: str) -> tuple[Any, Any]:
    """Allocate two 256 MiB buffers (512 MiB R+W sweep) to evict L2 cleanly between steps."""
    n_elem = L2_FLUSH_BYTES // BF16_BYTES
    return (
        torch_mod.ones((n_elem,), dtype=torch_mod.bfloat16, device=device),
        torch_mod.empty((n_elem,), dtype=torch_mod.bfloat16, device=device),
    )


def _flush_l2_cache(l2_flush_pair: tuple[Any, Any] | None) -> None:
    """Thrash L2 with >= 512 MiB R+W outside the profiled window so 1-layer weights stream cold from HBM."""
    if l2_flush_pair is not None:
        l2_flush_pair[1].copy_(l2_flush_pair[0])


def _profile_callable(
    torch_mod: Any,
    torch_npu_mod: Any,
    fn: Callable[[int], Any],
    *,
    rank: int = 0,
    dist_group: Any = None,
    blocker_pair: tuple[Any, Any] | None = None,
    l2_flush_pair: tuple[Any, Any] | None = None,
    ring_slots: int = 1,
    warmup_iters: int = SUBGRAPH_WARMUP_ITERS,
    cpu_store: Any = None,
    world_size: int = 1,
    barrier_tag: str = "",
) -> dict[str, Any]:
    """Warm up, flush L2 between steps, time events, and capture all ``kernel_details.csv`` rows."""
    for idx in range(warmup_iters):
        warm_err: Exception | None = None
        try:
            fn(idx % ring_slots)
        except Exception as exc:  # noqa: BLE001
            warm_err = exc
        if dist_group is not None:
            if not _sync_readiness(torch_mod, warm_err is None, f"npu:{rank}", dist_group):
                raise warm_err if warm_err is not None else RuntimeError(f"peer_rank_warmup_iter_{idx}_failed")
        elif warm_err is not None:
            raise warm_err
    torch_mod.npu.synchronize()

    events = [
        (torch_mod.npu.Event(enable_timing=True), torch_mod.npu.Event(enable_timing=True))
        for _ in range(SUBGRAPH_TIMED_ITERS)
    ]
    if blocker_pair is not None:
        for _ in range(DVFS_LOCK_PRE_ITERS):
            _ = torch_mod.matmul(blocker_pair[0], blocker_pair[1])
    for idx, (s_ev, e_ev) in enumerate(events):
        if dist_group is not None:
            torch_mod.distributed.barrier(group=dist_group)
        iter_err: Exception | None = None
        try:
            _flush_l2_cache(l2_flush_pair)
            if blocker_pair is not None:
                for _ in range(STREAM_BLOCKER_ITERS):
                    _ = torch_mod.matmul(blocker_pair[0], blocker_pair[1])
            s_ev.record()
            fn(idx % ring_slots)
            e_ev.record()
        except Exception as exc:  # noqa: BLE001
            iter_err = exc
        if dist_group is not None:
            all_iter_ok = _sync_readiness(torch_mod, iter_err is None, f"npu:{rank}", dist_group)
            if not all_iter_ok:
                raise iter_err if iter_err is not None else RuntimeError(f"peer_rank_timed_iter_{idx}_failed")
        elif iter_err is not None:
            raise iter_err
    hot_freq_mhz = _sample_loaded_aicore_freq(torch_mod, blocker_pair, rank)
    event_us = [float(s_ev.elapsed_time(e_ev)) * MS_TO_US for s_ev, e_ev in events]
    median_event_us = float(statistics.median(event_us))

    tmp_dir = tempfile.mkdtemp(prefix="mindie_1layer_prof_")
    raw_kernels: list[dict[str, Any]] = []
    try:
        exp_cfg = torch_npu_mod.profiler._ExperimentalConfig(
            profiler_level=torch_npu_mod.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu_mod.profiler.AiCMetrics.PipeUtilization,
            data_simplification=False,
        )
        if blocker_pair is not None:
            for _ in range(DVFS_LOCK_PRE_ITERS):
                _ = torch_mod.matmul(blocker_pair[0], blocker_pair[1])
        # Flush L2 strictly outside torch_npu.profiler.profile() so 1-layer weights stream cold
        # from HBM without injecting flush copy kernels into kernel_details.csv.
        _flush_l2_cache(l2_flush_pair)
        if dist_group is not None:
            torch_mod.distributed.barrier(group=dist_group)
        torch_mod.npu.synchronize()
        prof_err: Exception | None = None
        try:
            with torch_npu_mod.profiler.profile(
                activities=[
                    torch_npu_mod.profiler.ProfilerActivity.CPU,
                    torch_npu_mod.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu_mod.profiler.tensorboard_trace_handler(tmp_dir),
                experimental_config=exp_cfg,
            ) as prof:
                if cpu_store is not None and world_size > 1 and barrier_tag:
                    _cpu_store_barrier(cpu_store, f"prof_enter:{barrier_tag}", world_size)
                if blocker_pair is not None:
                    for _ in range(STREAM_BLOCKER_ITERS):
                        _ = torch_mod.matmul(blocker_pair[0], blocker_pair[1])
                fn(0)
                torch_mod.npu.synchronize()
                prof.step()
        except Exception as exc:  # noqa: BLE001
            prof_err = exc
        if dist_group is not None:
            all_prof_ok = _sync_readiness(torch_mod, prof_err is None, f"npu:{rank}", dist_group)
            if not all_prof_ok:
                raise prof_err if prof_err is not None else RuntimeError("peer_rank_profiled_step_failed")
        elif prof_err is not None:
            raise prof_err
        csvs = sorted(glob.glob(os.path.join(tmp_dir, "**", "kernel_details.csv"), recursive=True))
        if not csvs:
            raise RuntimeError("missing_kernel_details_csv")
        for csv_file in csvs:
            raw_kernels.extend(_parse_kernel_details_csv(Path(csv_file)))
        _enrich_hccl_rows_from_trace_view(tmp_dir, raw_kernels)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    kernels, stripped, matched_blockers = _strip_stream_blocker_rows(
        raw_kernels, expected_blockers=STREAM_BLOCKER_ITERS if blocker_pair is not None else 0,
    )
    if not kernels:
        raise RuntimeError("empty_profiled_kernel_rows_after_blocker_strip")
    summary = _span_from_kernels(kernels)
    return {
        "stream_blocked": blocker_pair is not None and matched_blockers == STREAM_BLOCKER_ITERS,
        "l2_flushed": l2_flush_pair is not None,
        "l2_flush_bytes": (2 * L2_FLUSH_BYTES) if l2_flush_pair is not None else 0,
        "blocker_expected_count": STREAM_BLOCKER_ITERS if blocker_pair is not None else 0,
        "blocker_matched_count": matched_blockers,
        "blocker_stripped_rows": stripped,
        "observed_aicore_freq_mhz": hot_freq_mhz,
        "event_median_us": round(median_event_us, 4),
        "event_samples_us": [round(x, 4) for x in event_us],
        **summary,
        "kernels": kernels,
    }


def _make_rotary_cos_sin(
    torch_mod: Any,
    batch: int,
    seq_len: int,
    head_dim: int,
    device: str,
) -> tuple[Any, Any]:
    """Construct non-identity rotary ``(cos, sin)`` tables with shape ``(batch, 1, seq_len, head_dim)``."""
    half = max(1, head_dim // 2)
    inv_freq = 1.0 / (ROPE_THETA_BASE ** (torch_mod.arange(0, half, dtype=torch_mod.float32, device=device) / float(half)))
    positions = torch_mod.arange(1, seq_len + 1, dtype=torch_mod.float32, device=device)
    freqs = torch_mod.outer(positions, inv_freq)
    emb = torch_mod.cat([freqs, freqs], dim=-1)[:, :head_dim]
    cos_t = torch_mod.cos(emb).to(torch_mod.bfloat16).view(1, 1, seq_len, head_dim).expand(batch, 1, seq_len, head_dim).contiguous()
    sin_t = torch_mod.sin(emb).to(torch_mod.bfloat16).view(1, 1, seq_len, head_dim).expand(batch, 1, seq_len, head_dim).contiguous()
    return cos_t, sin_t


def _make_linear_fn(
    torch_mod: Any,
    torch_npu_mod: Any,
    in_dim: int,
    out_dim: int,
    quant_mode: str,
    device: str,
) -> Callable[[Any], Any]:
    """Create a synthetic linear projection matching BF16, W8A8, C8, W4A8, MXFP8, or MXFP4."""
    init_scale = 1.0 / max(1.0, math.sqrt(float(in_dim)))
    if quant_mode in ("w8a8", "c8"):
        w_q = torch_mod.randint(
            INT8_MIN_VAL, INT8_MAX_VAL, (in_dim, out_dim), dtype=torch_mod.int8, device=device,
        )
        w_s = torch_mod.full(
            (out_dim,), init_scale / INT8_SCALE_NORM, dtype=torch_mod.float32, device=device,
        )

        def _run_w8a8(x: Any) -> Any:
            xq, xs = torch_npu_mod.npu_dynamic_quant(x)
            return torch_npu_mod.npu_quant_matmul(
                xq, w_q, w_s, pertoken_scale=xs, output_dtype=torch_mod.bfloat16,
            )

        return _run_w8a8

    if quant_mode == "w4a8":
        w_raw = torch_mod.randint(INT4_MIN_VAL, INT4_MAX_VAL, (in_dim, out_dim), dtype=torch_mod.int32, device=device)
        w_pack = torch_npu_mod.npu_convert_weight_to_int4pack(w_raw)
        w_s = torch_mod.full((out_dim,), init_scale / float(INT4_MAX_VAL), dtype=torch_mod.bfloat16, device=device)
        del w_raw
        return lambda x: torch_npu_mod.npu_weight_quant_batchmatmul(x, w_pack, w_s)

    if quant_mode in ("mxfp8", "mxfp4"):
        is_fp4 = quant_mode == "mxfp4"
        dst_dt = getattr(torch_npu_mod, "float4_e2m1fn_x2", None) if is_fp4 else getattr(torch_mod, "float8_e4m3fn", None)
        if dst_dt is None:
            raise CapabilityAbsentError(f"{quant_mode}_dtype_absent_in_torch_npu")
        w_raw = torch_mod.randn((out_dim, in_dim), dtype=torch_mod.bfloat16, device=device) * init_scale
        w_q, w_s = torch_npu_mod.npu_dynamic_mx_quant(w_raw, dst_type=dst_dt)
        w_q_t = w_q.t()
        w_s_t = w_s.transpose(0, 1)
        del w_raw
        extra_kw: dict[str, Any] = {"x1_dtype": dst_dt, "x2_dtype": dst_dt} if is_fp4 else {}

        def _run_mx(x: Any) -> Any:
            xq, xs = torch_npu_mod.npu_dynamic_mx_quant(x, dst_type=dst_dt)
            return torch_npu_mod.npu_quant_matmul(
                xq, w_q_t, w_s_t, pertoken_scale=xs, output_dtype=torch_mod.bfloat16,
                scale_dtype=torch_npu_mod.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu_mod.float8_e8m0fnu,
                group_sizes=[1, 1, MX_GROUP_SIZE], **extra_kw,
            )

        return _run_mx

    w = torch_mod.randn((in_dim, out_dim), dtype=torch_mod.bfloat16, device=device) * init_scale
    return lambda x: torch_mod.matmul(x, w)


def _build_gqa_prefill(
    torch_mod: Any,
    torch_npu_mod: Any,
    cfg: dict[str, Any],
    device: str,
    dist_group: Any,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer GQA / SWA / DiT-NonCausal / Gated-GQA prefill subgraph with RoPE."""
    if bool(cfg.get("use_indexer", False)):
        raise RunnerUnsupportedOverlayError("sparse_gqa_prefill_indexer_not_implemented_in_runner")
    tp, pcp = cfg["tp"], cfg["pcp"]
    batch, q_len = cfg["batch"], cfg["q_len"]
    local_heads, local_kv, head_dim = cfg["local_heads"], cfg["local_kv"], cfg["head_dim"]
    q_out, window, use_gate, noncausal = cfg["q_out"], cfg["window"], cfg["use_gate"], cfg["noncausal"]
    norm_w, qkv_fn, o_fn, qk_norm_w = cfg["norm_w"], cfg["qkv_fn"], cfg["o_fn"], cfg["qk_norm_w"]
    per_seq_q = max(1, q_len // pcp)
    local_tokens = max(1, batch * per_seq_q)
    cum_qlen = [per_seq_q * (i + 1) for i in range(batch)]
    cos_t, sin_t = (
        (None, None) if noncausal else _make_rotary_cos_sin(torch_mod, batch, per_seq_q, head_dim, device)
    )
    mask = (
        None if noncausal else torch_mod.triu(
            torch_mod.ones((DEFAULT_CAUSAL_MASK_DIM, DEFAULT_CAUSAL_MASK_DIM), dtype=torch_mod.bool, device=device),
            diagonal=1,
        )
    )

    def _run_prefill(x_in: Any) -> Any:
        h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
        qkv = qkv_fn(h)
        q_part, k_part, v_part = qkv.split((q_out, local_kv * head_dim, local_kv * head_dim), dim=-1)
        if use_gate:
            q_part, gate_part = q_part.chunk(2, dim=-1)
        q = q_part.view(local_tokens, local_heads, head_dim)
        k = k_part.view(local_tokens, local_kv, head_dim)
        v = v_part.view(local_tokens, local_kv, head_dim)
        if qk_norm_w is not None:
            q, _ = torch_npu_mod.npu_rms_norm(q, qk_norm_w, DEFAULT_RMS_EPS)
            k, _ = torch_npu_mod.npu_rms_norm(k, qk_norm_w, DEFAULT_RMS_EPS)
        if cos_t is not None and sin_t is not None:
            q_4d = q.view(batch, per_seq_q, local_heads, head_dim).transpose(1, 2).contiguous()
            k_4d = k.view(batch, per_seq_q, local_kv, head_dim).transpose(1, 2).contiguous()
            q = torch_npu_mod.npu_rotary_mul(q_4d, cos_t, sin_t).transpose(1, 2).reshape(local_tokens, local_heads, head_dim).contiguous()
            k = torch_npu_mod.npu_rotary_mul(k_4d, cos_t, sin_t).transpose(1, 2).reshape(local_tokens, local_kv, head_dim).contiguous()
        if pcp > 1 and dist_group is not None:
            kv_local = torch_mod.cat([k.contiguous(), v.contiguous()], dim=-1).contiguous()
            kv_all = [torch_mod.empty_like(kv_local) for _ in range(pcp)]
            torch_mod.distributed.all_gather(kv_all, kv_local, group=dist_group)
            kv_full = torch_mod.cat(kv_all, dim=0)
            k = kv_full[:, :, :head_dim].contiguous()
            v = kv_full[:, :, head_dim:].contiguous()
        if noncausal:
            out, _ = torch_npu_mod.npu_fused_infer_attention_score_v2(
                q.unsqueeze(0).contiguous(), k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous(),
                num_query_heads=local_heads, num_key_value_heads=local_kv,
                input_layout="BSND", softmax_scale=head_dim ** -0.5,
                sparse_mode=NONCAUSAL_SPARSE_MODE,
            )
        else:
            sparse_mode = GQA_SWA_SPARSE_MODE if window > 0 else CAUSAL_SPARSE_MODE
            per_seq_kv = max(1, int(k.shape[0]) // batch)
            cum_kvlen = [per_seq_kv * (i + 1) for i in range(batch)]
            attn_kw: dict[str, Any] = dict(
                num_query_heads=local_heads, num_key_value_heads=local_kv,
                input_layout="TND", softmax_scale=head_dim ** -0.5,
                sparse_mode=sparse_mode,
                actual_seq_qlen=cum_qlen, actual_seq_kvlen=cum_kvlen,
            )
            if mask is not None:
                attn_kw["atten_mask"] = mask
            if window > 0:
                attn_kw["pre_tokens"] = window
                attn_kw["next_tokens"] = 0
            out, _ = torch_npu_mod.npu_fused_infer_attention_score_v2(q, k, v, **attn_kw)
        out_2d = out.reshape(local_tokens, local_heads * head_dim)
        if use_gate:
            out_2d = out_2d * torch_mod.sigmoid(gate_part)
        res = o_fn(out_2d)
        if tp > 1 and dist_group is not None:
            torch_mod.distributed.all_reduce(res, group=dist_group)
        return res

    return _run_prefill, TIER_TORCHNPU_COMPOSITE, {
        "phase": "prefill", "local_heads": local_heads, "local_kv": local_kv,
        "head_dim": head_dim, "noncausal": noncausal, "local_tokens": local_tokens,
        "rope_executed": cos_t is not None,
    }


def _compute_kv_write_indices(
    torch_mod: Any,
    block_table: Any,
    active_seqs: int,
    q_len: int,
    local_kv_len: int,
    blk: int,
    blocks_per_seq: int,
    device: str,
) -> tuple[Any, Any]:
    """Compute exact multi-page ``(write_blocks, write_slots)`` for active trailing tokens."""
    if active_seqs <= 0 or q_len <= 0:
        empty = torch_mod.empty((0,), dtype=torch_mod.int64, device=device)
        return empty, empty
    slot_base = max(0, local_kv_len - q_len)
    token_pos = slot_base + torch_mod.arange(q_len, dtype=torch_mod.int64, device=device)
    page_idx = (token_pos // blk).clamp(max=max(0, blocks_per_seq - 1))
    write_blocks = block_table[:active_seqs][:, page_idx].reshape(-1).to(torch_mod.int64)
    write_slots = (token_pos % blk).repeat(active_seqs)
    return write_blocks, write_slots


def _build_gqa_decode(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    device: str,
    dist_group: Any,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer GQA / SWA / Gated-GQA / C8-KV decode subgraph with QK-norm, RoPE, and KV write."""
    tp, dcp = cfg["tp"], cfg["dcp"]
    local_heads, local_kv, head_dim = cfg["local_heads"], cfg["local_kv"], cfg["head_dim"]
    batch, q_len, tokens, blk = cfg["batch"], cfg["q_len"], cfg["tokens"], cfg["blk"]
    q_out, use_gate, window = cfg["q_out"], cfg["use_gate"], cfg["window"]
    norm_w, qkv_fn, o_fn, qk_norm_w = cfg["norm_w"], cfg["qkv_fn"], cfg["o_fn"], cfg["qk_norm_w"]
    is_c8 = str(spec.get("quant_mode", "bf16")) == "c8"

    eff_ctx = min(cfg["context"], window) if window > 0 else cfg["context"]
    local_kv_len = max(q_len, eff_ctx // dcp)
    blocks_per_seq = max(1, math.ceil(local_kv_len / blk))
    num_blocks = blocks_per_seq * batch + COLD_RING_MIN_SLOTS
    kv_dt = torch_mod.int8 if is_c8 else torch_mod.bfloat16
    if is_c8:
        k_cache = torch_mod.randint(INT8_MIN_VAL, INT8_MAX_VAL, (num_blocks, blk, local_kv * head_dim), dtype=kv_dt, device=device)
        v_cache = torch_mod.randint(INT8_MIN_VAL, INT8_MAX_VAL, (num_blocks, blk, local_kv * head_dim), dtype=kv_dt, device=device)
        kv_deq_k = torch_mod.full((1, local_kv * head_dim), 1.0 / INT8_SCALE_NORM, dtype=torch_mod.bfloat16, device=device)
        kv_deq_v = torch_mod.full((1, local_kv * head_dim), 1.0 / INT8_SCALE_NORM, dtype=torch_mod.bfloat16, device=device)
    else:
        k_cache = torch_mod.randn((num_blocks, blk, local_kv * head_dim), dtype=kv_dt, device=device)
        v_cache = torch_mod.randn((num_blocks, blk, local_kv * head_dim), dtype=kv_dt, device=device)
        kv_deq_k = kv_deq_v = None
    block_table = torch_mod.arange(batch * blocks_per_seq, dtype=torch_mod.int32, device=device).view(
        batch, blocks_per_seq,
    )
    active_seqs = min(batch, max(0, int(spec.get("active_sequences", batch))))
    active_tokens = active_seqs * q_len
    padded_kv_len = 0 if active_seqs < batch else local_kv_len
    seq_lens = [local_kv_len if i < active_seqs else padded_kv_len for i in range(batch)]
    write_blocks, write_slots = _compute_kv_write_indices(
        torch_mod, block_table, active_seqs, q_len, local_kv_len, blk, blocks_per_seq, device,
    )
    cos_t, sin_t = _make_rotary_cos_sin(torch_mod, batch, q_len, head_dim, device)
    mask_mtp = (
        torch_mod.triu(torch_mod.ones((DEFAULT_CAUSAL_MASK_DIM, DEFAULT_CAUSAL_MASK_DIM), dtype=torch_mod.bool, device=device), diagonal=1)
        if q_len > 1 else None
    )
    if cfg["use_indexer"]:
        raise RunnerUnsupportedOverlayError("sparse_gqa_indexer_not_implemented_in_runner")

    def _run_decode(x_in: Any) -> Any:
        h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
        qkv = qkv_fn(h)
        q_part, k_cur, v_cur = qkv.split((q_out, local_kv * head_dim, local_kv * head_dim), dim=-1)
        if use_gate:
            q_part, gate_part = q_part.chunk(2, dim=-1)
        q = q_part.view(batch, q_len, local_heads, head_dim).transpose(1, 2).contiguous()
        k_4d = k_cur.view(batch, q_len, local_kv, head_dim).transpose(1, 2).contiguous()
        if qk_norm_w is not None:
            q, _ = torch_npu_mod.npu_rms_norm(q, qk_norm_w, DEFAULT_RMS_EPS)
            k_4d, _ = torch_npu_mod.npu_rms_norm(k_4d, qk_norm_w, DEFAULT_RMS_EPS)
        q = torch_npu_mod.npu_rotary_mul(q, cos_t, sin_t)
        k_4d = torch_npu_mod.npu_rotary_mul(k_4d, cos_t, sin_t)
        k_ready = k_4d.transpose(1, 2).reshape(tokens, local_kv * head_dim)
        # Write ONLY active normalized+rotated tokens into exact per-token pages (never padded slots)
        if active_tokens > 0:
            if is_c8:
                k_q, _ = torch_npu_mod.npu_dynamic_quant(k_ready[:active_tokens])
                v_q, _ = torch_npu_mod.npu_dynamic_quant(v_cur[:active_tokens])
                k_cache[write_blocks, write_slots] = k_q
                v_cache[write_blocks, write_slots] = v_q
            else:
                k_cache[write_blocks, write_slots] = k_ready[:active_tokens]
                v_cache[write_blocks, write_slots] = v_cur[:active_tokens]
        fia_kw: dict[str, Any] = dict(
            num_query_heads=local_heads, num_key_value_heads=local_kv,
            input_layout="BNSD", softmax_scale=head_dim ** -0.5,
            block_table=block_table, block_size=blk,
            actual_seq_kvlen=seq_lens, return_softmax_lse=True,
        )
        if is_c8 and kv_deq_k is not None and kv_deq_v is not None:
            fia_kw["dequant_scale_key"] = kv_deq_k
            fia_kw["dequant_scale_value"] = kv_deq_v
        if mask_mtp is not None:
            fia_kw["atten_mask"] = mask_mtp
            fia_kw["sparse_mode"] = CAUSAL_SPARSE_MODE
        out, _lse = torch_npu_mod.npu_fused_infer_attention_score_v2(q, k_cache, v_cache, **fia_kw)
        out_2d = out.transpose(1, 2).reshape(tokens, local_heads * head_dim)
        if use_gate:
            out_2d = out_2d * torch_mod.sigmoid(gate_part)
        res = o_fn(out_2d)
        if tp > 1 and dist_group is not None:
            torch_mod.distributed.all_reduce(res, group=dist_group)
        return res

    return _run_decode, TIER_TORCHNPU_COMPOSITE, {
        "phase": "decode", "local_heads": local_heads, "local_kv": local_kv,
        "head_dim": head_dim, "active_sequences": active_seqs, "padded_sequences": batch - active_seqs,
        "logical_kv_len": local_kv_len, "sliding_window": window,
        "padded_slot_kv_len": padded_kv_len, "executed_indexer": False, "c8_kv": is_c8,
    }


def _build_gqa_attention(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    device: str,
    dist_group: Any = None,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer GQA / Sliding-Window / DiT / Gated-GQA / Sparse-GQA Attention subgraph."""
    hidden = int(spec["hidden_size"])
    tp = max(1, int(spec.get("tp", 1)))
    local_heads = max(1, int(spec["num_heads"]) // tp)
    raw_kv = max(1, int(spec["num_kv_heads"]))
    local_kv = max(1, raw_kv // tp) if raw_kv >= tp else 1
    head_dim = int(spec["head_dim"])
    batch = int(spec.get("sequences", 4))
    q_len = int(spec.get("q_len", 1))
    quant_mode = str(spec.get("quant_mode", "bf16"))
    use_gate = bool(spec.get("q_output_gate", False))
    q_out = local_heads * head_dim * (2 if use_gate else 1)
    kv_out = 2 * local_kv * head_dim
    noncausal = bool(spec.get("noncausal_attention", False)) or str(spec.get("architecture", "")) == "dit_full_attention"
    cfg = {
        "hidden": hidden, "tp": tp, "dcp": max(1, int(spec.get("dcp", 1))),
        "pcp": max(1, int(spec.get("pcp", 1))), "local_heads": local_heads,
        "local_kv": local_kv, "head_dim": head_dim, "batch": batch, "q_len": q_len,
        "tokens": batch * q_len, "context": int(spec.get("context", 2048)),
        "window": int(spec.get("window", 0)), "use_gate": use_gate, "noncausal": noncausal,
        "use_indexer": bool(spec.get("indexer", False)),
        "blk": int(spec.get("block_size", DEFAULT_KV_BLOCK_SIZE)), "q_out": q_out,
        "norm_w": torch_mod.ones((hidden,), dtype=torch_mod.bfloat16, device=device),
        "qkv_fn": _make_linear_fn(torch_mod, torch_npu_mod, hidden, q_out + kv_out, quant_mode, device),
        "o_fn": _make_linear_fn(torch_mod, torch_npu_mod, local_heads * head_dim, hidden, quant_mode, device),
        "qk_norm_w": (
            torch_mod.ones((head_dim,), dtype=torch_mod.bfloat16, device=device)
            if bool(spec.get("qk_norm", False)) else None
        ),
    }
    if str(spec.get("phase", "decode")) == "prefill":
        return _build_gqa_prefill(torch_mod, torch_npu_mod, cfg, device, dist_group)
    return _build_gqa_decode(torch_mod, torch_npu_mod, spec, cfg, device, dist_group)


def _build_mla_prefill(
    torch_mod: Any,
    torch_npu_mod: Any,
    cfg: dict[str, Any],
    device: str,
    dist_group: Any,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer MLA prefill subgraph (supporting PCP AllGather)."""
    if bool(cfg.get("use_indexer", False)):
        raise RunnerUnsupportedOverlayError("sparse_mla_prefill_indexer_not_implemented_in_runner")
    tp, pcp = cfg["tp"], cfg["pcp"]
    batch, q_len = cfg["batch"], cfg["q_len"]
    local_heads, kv_lora = cfg["local_heads"], cfg["kv_lora"]
    qk_nope, qk_rope, v_dim = cfg["qk_nope"], cfg["qk_rope"], cfg["v_dim"]
    o_in_dim = cfg["o_in_dim"]
    norm_w, kv_a_fn, kv_norm_w = cfg["norm_w"], cfg["kv_a_fn"], cfg["kv_norm_w"]
    q_a_fn, q_norm_w, q_b_fn, o_fn = cfg["q_a_fn"], cfg["q_norm_w"], cfg["q_b_fn"], cfg["o_fn"]

    per_seq_q = max(1, q_len // pcp)
    local_tokens = max(1, batch * per_seq_q)
    cum_qlen = [per_seq_q * (i + 1) for i in range(batch)]
    kv_b_fn = _make_linear_fn(
        torch_mod, torch_npu_mod, kv_lora, local_heads * (qk_nope + v_dim), cfg["quant_mode"], device,
    )
    mask = torch_mod.triu(
        torch_mod.ones((DEFAULT_CAUSAL_MASK_DIM, DEFAULT_CAUSAL_MASK_DIM), dtype=torch_mod.bool, device=device),
        diagonal=1,
    )

    def _run_mla_prefill(x_in: Any) -> Any:
        h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
        if q_a_fn is not None and q_norm_w is not None:
            qc, _ = torch_npu_mod.npu_rms_norm(q_a_fn(h), q_norm_w, DEFAULT_RMS_EPS)
            q_full = q_b_fn(qc)
        else:
            q_full = q_b_fn(h)
        c_kv, _ = torch_npu_mod.npu_rms_norm(kv_a_fn(h)[:, :kv_lora], kv_norm_w, DEFAULT_RMS_EPS)
        if pcp > 1 and dist_group is not None:
            c_list = [torch_mod.empty_like(c_kv) for _ in range(pcp)]
            torch_mod.distributed.all_gather(c_list, c_kv.contiguous(), group=dist_group)
            c_kv_all = torch_mod.cat(c_list, dim=0)
        else:
            c_kv_all = c_kv
        kv_up = kv_b_fn(c_kv_all).view(c_kv_all.shape[0], local_heads, qk_nope + v_dim)
        k_full = kv_up[:, :, :qk_nope].contiguous()
        v_full = kv_up[:, :, qk_nope:].contiguous()
        q_nope = q_full.view(local_tokens, local_heads, qk_nope + qk_rope)[:, :, :qk_nope].contiguous()
        if v_dim != qk_nope:
            v_full = v_full[:, :, :qk_nope].contiguous()
        per_seq_kv = max(1, int(k_full.shape[0]) // batch)
        cum_kvlen = [per_seq_kv * (i + 1) for i in range(batch)]
        out, _ = torch_npu_mod.npu_fused_infer_attention_score_v2(
            q_nope, k_full, v_full,
            num_query_heads=local_heads, num_key_value_heads=local_heads,
            input_layout="TND", softmax_scale=(qk_nope + qk_rope) ** -0.5,
            atten_mask=mask, sparse_mode=CAUSAL_SPARSE_MODE,
            actual_seq_qlen=cum_qlen, actual_seq_kvlen=cum_kvlen,
        )
        res = o_fn(out.reshape(local_tokens, o_in_dim))
        if tp > 1 and dist_group is not None:
            torch_mod.distributed.all_reduce(res, group=dist_group)
        return res

    return _run_mla_prefill, TIER_TORCHNPU_COMPOSITE, {
        "phase": "prefill", "local_heads": local_heads, "kv_lora_rank": kv_lora, "local_tokens": local_tokens,
    }


def _init_mla_indexer_state(
    torch_mod: Any,
    torch_npu_mod: Any,
    cfg: dict[str, Any],
    active_seqs: int,
    seq_lens: list[int],
    device: str,
) -> tuple[Any, ...]:
    """Allocate synthetic LightningIndexer & SparseFlashAttention decode buffers."""
    hidden, batch, q_len, tokens = cfg["hidden"], cfg["batch"], cfg["q_len"], cfg["tokens"]
    context, blk = cfg["context"], cfg["blk"]
    idx_heads, idx_dim = cfg["idx_heads"], cfg["idx_dim"]
    idx_q_fn = _make_linear_fn(torch_mod, torch_npu_mod, hidden, idx_heads * idx_dim, "bf16", device)
    idx_blocks_per_seq = max(1, math.ceil(context / blk))
    idx_num_blocks = idx_blocks_per_seq * batch + COLD_RING_MIN_SLOTS
    idx_bt = torch_mod.arange(batch * idx_blocks_per_seq, dtype=torch_mod.int32, device=device).view(
        batch, idx_blocks_per_seq,
    )
    idx_k_cache = torch_mod.randn((idx_num_blocks, blk, 1, idx_dim), dtype=torch_mod.bfloat16, device=device)
    idx_w = torch_mod.randn((batch, q_len, idx_heads), dtype=torch_mod.bfloat16, device=device)
    idx_aq = torch_mod.tensor([q_len] * batch, dtype=torch_mod.int32, device=device)
    idx_ak_full = torch_mod.tensor([context if i < active_seqs else 1 for i in range(batch)], dtype=torch_mod.int32, device=device)
    idx_ak_local = torch_mod.tensor([max(1, s) for s in seq_lens], dtype=torch_mod.int32, device=device)
    sfa_aq = torch_mod.arange(q_len, tokens + 1, q_len, dtype=torch_mod.int32, device=device)
    return (idx_q_fn, idx_bt, idx_k_cache, idx_w, idx_aq, idx_ak_full, idx_ak_local, sfa_aq)


def _dcp_lse_merge(
    torch_mod: Any,
    torch_npu_mod: Any,
    out_nbsd: Any,
    lse: Any,
    *,
    dcp: int,
    local_heads: int,
    batch: int,
    q_len: int,
    tokens: int,
    kv_lora: int,
    dist_group: Any,
    use_all_gather: bool = False,
) -> Any:
    """Exchange DCP partial outputs + LSE (AllToAll or AllGather for joint PCP+DCP) and merge."""
    if use_all_gather:
        out_h = out_nbsd.reshape(local_heads, tokens, kv_lora)
        lse_h = lse.reshape(batch, local_heads, q_len).permute(1, 0, 2).reshape(
            local_heads, tokens, 1,
        ).to(out_h.dtype)
        send_buf = torch_mod.cat([out_h, lse_h], dim=-1).contiguous()
        recv_list = [torch_mod.empty_like(send_buf) for _ in range(dcp)]
        torch_mod.distributed.all_gather(recv_list, send_buf, group=dist_group)
        out_parts = [r[:, :, :kv_lora].reshape(-1, kv_lora).contiguous() for r in recv_list]
        lse_parts = [r[:, :, kv_lora].reshape(-1).to(torch_mod.float32).contiguous() for r in recv_list]
        merged, _ = torch_npu_mod.npu_attention_update(lse_parts, out_parts, 0)
        return merged.view(local_heads, tokens, kv_lora)
    out_h = out_nbsd.reshape(dcp, local_heads, tokens, kv_lora)
    lse_h = lse.reshape(batch, dcp, local_heads, q_len).permute(1, 2, 0, 3).reshape(
        dcp, local_heads, tokens, 1,
    ).to(out_h.dtype)
    send_buf = torch_mod.cat([out_h, lse_h], dim=-1).contiguous()
    recv_buf = torch_mod.empty_like(send_buf)
    torch_mod.distributed.all_to_all_single(recv_buf, send_buf, group=dist_group)
    out_parts = [recv_buf[r, :, :, :kv_lora].reshape(-1, kv_lora).contiguous() for r in range(dcp)]
    lse_parts = [recv_buf[r, :, :, kv_lora].reshape(-1).to(torch_mod.float32).contiguous() for r in range(dcp)]
    merged, _ = torch_npu_mod.npu_attention_update(lse_parts, out_parts, 0)
    return merged.view(local_heads, tokens, kv_lora)


def _resolve_mla_decode_kv_geometry(
    spec: dict[str, Any],
    context: int,
    dcp: int,
    blk: int,
    idx_topk: int,
    q_len: int = 1,
) -> tuple[int, int, int, bool]:
    """Compute effective (local_kv_len, idx_scan_len, eff_sparse_count, owns_long_kv) without blk inflation."""
    del blk
    cache_kind = str(spec.get("cache_kind", ""))
    window = int(spec.get("window", 0))
    compress_ratio = int(spec.get("compress_ratio", 0))
    owns_long_kv = bool(spec.get("owns_long_kv", True))
    cand_filter = bool(spec.get("candidate_filter", False))
    cand_count = int(spec.get("candidate_count", 0))
    raw_local_kv = max(q_len, context // dcp)
    if cache_kind == "window_compressed" and compress_ratio == 0 and window > 0:
        local_kv_len = max(q_len, min(raw_local_kv, window))
    elif compress_ratio > 1:
        local_kv_len = max(q_len, raw_local_kv // compress_ratio)
    else:
        local_kv_len = raw_local_kv
    idx_scan_len = max(q_len, context // compress_ratio) if compress_ratio > 1 else max(q_len, context)
    if cand_filter and cand_count > 0:
        idx_scan_len = max(q_len, min(idx_scan_len, cand_count))
    eff_sparse_count = max(1, min(idx_topk, local_kv_len, idx_scan_len))
    return local_kv_len, idx_scan_len, eff_sparse_count, owns_long_kv


def _exec_mla_sfa_decode(
    torch_mod: Any,
    torch_npu_mod: Any,
    *,
    h: Any,
    q_abs_exec: Any,
    q_pe_exec: Any,
    kv_cache_4d: Any,
    kr_cache_4d: Any,
    block_table: Any,
    idx_state: Any,
    ext_sfa_state: Any,
    batch: int,
    q_len: int,
    tokens: int,
    idx_heads: int,
    idx_dim: int,
    eff_sparse_count: int,
    scale_val: float,
    dcp: int = 1,
    local_heads: int = 1,
    kv_lora: int = 512,
    dist_group: Any = None,
    cand_block_size: int = 0,
    owns_candidate_blocks: bool = False,
    local_kv_len: int = 0,
) -> Any:
    """Execute Sparse-MLA decode via LightningIndexer+SFA, candidate-block remap, or DCP+SFA."""
    if idx_state is not None:
        iq_fn, ibt, ik_c, iw, iaq, iak_full, sfa_ak, sfa_aq = idx_state
        iq = iq_fn(h).view(batch, q_len, idx_heads, idx_dim)
        topk_idx, _ = torch_npu_mod.npu_lightning_indexer(
            iq, ik_c, iw, actual_seq_lengths_query=iaq, actual_seq_lengths_key=iak_full,
            block_table=ibt, layout_query="BSND", layout_key="PA_BSND",
            sparse_count=eff_sparse_count, sparse_mode=CAUSAL_SPARSE_MODE,
        )
        if owns_candidate_blocks and cand_block_size > 1:
            _ = torch_mod.div(topk_idx, cand_block_size, rounding_mode="floor")
        sfa_indices = topk_idx.view(tokens, 1, -1).to(torch_mod.int32)
        if dcp > 1:
            max_local_idx = max(0, (local_kv_len if local_kv_len > 0 else eff_sparse_count) - 1)
            sfa_indices = torch_mod.where(
                sfa_indices >= 0,
                torch_mod.div(sfa_indices, dcp, rounding_mode="floor").clamp(max=max_local_idx),
                sfa_indices,
            )
    else:
        assert ext_sfa_state is not None
        sfa_indices, sfa_aq, sfa_ak, cand_remap = ext_sfa_state
        if cand_remap is not None:
            sfa_indices = cand_remap[sfa_indices.to(torch_mod.int64)].to(torch_mod.int32)
    need_lse = dcp > 1 and dist_group is not None
    sfa_kw: dict[str, Any] = dict(
        query=q_abs_exec, key=kv_cache_4d, value=kv_cache_4d,
        sparse_indices=sfa_indices, scale_value=scale_val,
        sparse_block_size=1, block_table=block_table,
        actual_seq_lengths_query=sfa_aq, actual_seq_lengths_kv=sfa_ak,
        query_rope=q_pe_exec, key_rope=kr_cache_4d,
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=NONCAUSAL_SPARSE_MODE if need_lse else CAUSAL_SPARSE_MODE,
        attention_mode=SFA_ATTN_MODE_MLA,
        return_softmax_lse=need_lse,
    )
    if not need_lse:
        sfa_out = torch_npu_mod.npu_sparse_flash_attention(**sfa_kw)
        return sfa_out[0].transpose(0, 1).contiguous()
    try:
        sfa_res = torch_npu_mod.npu_sparse_flash_attention(**sfa_kw)
    except RuntimeError as exc:
        # Empirically verify whether the installed container specifically rejects return_softmax_lse=True
        probe_kw = {**sfa_kw, "sparse_mode": CAUSAL_SPARSE_MODE, "return_softmax_lse": False}
        try:
            _ = torch_npu_mod.npu_sparse_flash_attention(**probe_kw)
            torch_mod.npu.synchronize()
        except Exception:  # noqa: BLE001
            raise exc from None
        raise CapabilityAbsentError(f"sfa_return_softmax_lse_unsupported_in_container:{str(exc).splitlines()[0][:140]}") from exc
    out_tnd, s_max, s_sum = sfa_res[0], sfa_res[1], sfa_res[2]
    lse_val = s_max.to(torch_mod.float32) + torch_mod.log(s_sum.to(torch_mod.float32).clamp(min=1e-30))
    return _dcp_lse_merge(
        torch_mod, torch_npu_mod, out_tnd.transpose(0, 1).contiguous(), lse_val,
        dcp=dcp, local_heads=local_heads, batch=batch, q_len=q_len,
        tokens=tokens, kv_lora=kv_lora, dist_group=dist_group,
    )


def _build_engram_hook(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    hidden: int,
    quant_mode: str,
    device: str,
) -> Callable[[Any], Any] | None:
    """Build device-side Engram gate/Q/K/add projection block when ``spec['engram']`` is True."""
    if not bool(spec.get("engram", False)):
        return None
    ew = max(1, int(spec.get("engram_width", DEFAULT_ENGRAM_WIDTH)))
    eg_gate = _make_linear_fn(torch_mod, torch_npu_mod, hidden, ew, quant_mode, device)
    eg_q = _make_linear_fn(torch_mod, torch_npu_mod, ew, ew, quant_mode, device)
    eg_k = _make_linear_fn(torch_mod, torch_npu_mod, ew, hidden, quant_mode, device)

    def _apply_engram(res_in: Any) -> Any:
        g_out = torch_mod.sigmoid(eg_gate(res_in))
        return res_in + eg_k(eg_q(g_out))

    return _apply_engram


def _init_external_sfa_state(
    torch_mod: Any,
    *,
    tokens: int,
    batch: int,
    q_len: int,
    eff_sparse_count: int,
    local_kv_len: int,
    seq_lens: list[int],
    cand_filter: bool,
    cand_block_sz: int,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    """Allocate external precomputed top-k indices, TND cumulative query endpoints, and candidate remap."""
    ext_idx = (
        torch_mod.arange(eff_sparse_count, dtype=torch_mod.int32, device=device) % max(1, local_kv_len)
    ).view(1, 1, eff_sparse_count).expand(tokens, 1, eff_sparse_count).contiguous()
    ext_aq = torch_mod.arange(q_len, tokens + 1, q_len, dtype=torch_mod.int32, device=device)
    ext_ak = torch_mod.tensor(seq_lens, dtype=torch_mod.int32, device=device)
    cand_remap = (
        (torch_mod.arange(max(1, local_kv_len), dtype=torch_mod.int32, device=device) * cand_block_sz)
        % max(1, local_kv_len)
    ) if cand_filter else None
    return (ext_idx, ext_aq, ext_ak, cand_remap)


def _build_mla_decode(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    device: str,
    dist_group: Any,
    rank: int,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer Absorbed MLA / Sparse-MLA (DSA/SFA/CSA2) / DCP / PCP+DCP decode subgraph."""
    tp, dcp, pcp, o_proj_tp = cfg["tp"], cfg["dcp"], cfg["pcp"], cfg["o_proj_tp"]
    local_heads, kv_lora = cfg["local_heads"], cfg["kv_lora"]
    qk_nope, qk_rope, rope_exec, v_dim = cfg["qk_nope"], cfg["qk_rope"], cfg["rope_exec"], cfg["v_dim"]
    batch, q_len, tokens, context, blk = cfg["batch"], cfg["q_len"], cfg["tokens"], cfg["context"], cfg["blk"]
    use_indexer, idx_heads, idx_dim, idx_topk = cfg["use_indexer"], cfg["idx_heads"], cfg["idx_dim"], cfg["idx_topk"]
    norm_w, kv_a_fn, kv_norm_w = cfg["norm_w"], cfg["kv_a_fn"], cfg["kv_norm_w"]
    q_a_fn, q_norm_w, q_b_fn, o_fn = cfg["q_a_fn"], cfg["q_norm_w"], cfg["q_b_fn"], cfg["o_fn"]

    w_uk = torch_mod.randn((local_heads, qk_nope, kv_lora), dtype=torch_mod.bfloat16, device=device) * (qk_nope ** -0.5)
    w_uv = torch_mod.randn((local_heads, kv_lora, v_dim), dtype=torch_mod.bfloat16, device=device) * (kv_lora ** -0.5)
    is_joint_pcp_dcp = dcp > 1 and pcp > 1 and dcp > tp and dist_group is not None
    dcp_heads = local_heads if is_joint_pcp_dcp else (local_heads * dcp if (dcp > 1 and dist_group is not None) else local_heads)
    local_kv_len, idx_scan_len, eff_sparse_count, owns_long_kv = _resolve_mla_decode_kv_geometry(
        spec, context, dcp, blk, idx_topk, q_len=q_len,
    )
    blocks_per_seq = max(1, math.ceil(local_kv_len / blk))
    num_blocks = blocks_per_seq * batch + COLD_RING_MIN_SLOTS
    block_table = torch_mod.arange(batch * blocks_per_seq, dtype=torch_mod.int32, device=device).view(batch, blocks_per_seq)
    active_seqs = min(batch, max(0, int(spec.get("active_sequences", batch))))
    active_tokens = active_seqs * q_len
    padded_kv_len = 0 if active_seqs < batch else local_kv_len
    seq_lens = [local_kv_len if i < active_seqs else padded_kv_len for i in range(batch)]
    kv_cache_3d = torch_mod.randn((num_blocks, blk, kv_lora), dtype=torch_mod.bfloat16, device=device)
    kr_cache_3d = torch_mod.randn((num_blocks, blk, rope_exec), dtype=torch_mod.bfloat16, device=device)
    kv_cache_4d, kr_cache_4d = kv_cache_3d.view(num_blocks, blk, 1, kv_lora), kr_cache_3d.view(num_blocks, blk, 1, rope_exec)
    write_blocks, write_slots = _compute_kv_write_indices(
        torch_mod, block_table, active_seqs, q_len, local_kv_len, blk, blocks_per_seq, device,
    )
    zero_rope = torch_mod.zeros((tokens, local_heads, rope_exec), dtype=torch_mod.bfloat16, device=device) if qk_rope == 0 else None

    cand_filter = bool(spec.get("candidate_filter", False))
    cand_block_sz = max(1, int(spec.get("candidate_block_size", 8)))
    owns_cand_blocks = bool(spec.get("owns_candidate_blocks", False))
    uses_ext_topk = (
        (not use_indexer) and idx_topk > 0 and dcp == 1 and q_len == 1
        and (bool(spec.get("uses_external_topk", False)) or cand_filter)
    )
    if use_indexer and q_len > 1:
        raise RunnerUnsupportedOverlayError("sparse_mla_mtp_not_implemented_in_runner")
    executed_indexer = bool(use_indexer)
    idx_state = (
        _init_mla_indexer_state(torch_mod, torch_npu_mod, {**cfg, "context": idx_scan_len}, active_seqs, seq_lens, device)
        if executed_indexer else None
    )
    ext_sfa_state = (
        _init_external_sfa_state(
            torch_mod, tokens=tokens, batch=batch, q_len=q_len, eff_sparse_count=eff_sparse_count,
            local_kv_len=local_kv_len, seq_lens=seq_lens, cand_filter=cand_filter,
            cand_block_sz=cand_block_sz, device=device,
        )
        if uses_ext_topk else None
    )
    engram_fn = _build_engram_hook(torch_mod, torch_npu_mod, spec, cfg["hidden"], cfg["quant_mode"], device)

    mask_mtp = (
        torch_mod.triu(torch_mod.ones((DEFAULT_CAUSAL_MASK_DIM, DEFAULT_CAUSAL_MASK_DIM), dtype=torch_mod.bool, device=device), diagonal=1)
        if q_len > 1 else None
    )
    o_chunk_idx = rank % max(1, o_proj_tp)

    def _run_mla_decode(x_in: Any) -> Any:
        h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
        if q_a_fn is not None and q_norm_w is not None:
            qc, _ = torch_npu_mod.npu_rms_norm(q_a_fn(h), q_norm_w, DEFAULT_RMS_EPS)
            q_full = q_b_fn(qc)
        else:
            q_full = q_b_fn(h)
        kv_raw = kv_a_fn(h)
        c_kv, _ = torch_npu_mod.npu_rms_norm(kv_raw[:, :kv_lora], kv_norm_w, DEFAULT_RMS_EPS)
        if active_tokens > 0 and owns_long_kv:
            kv_cache_3d[write_blocks, write_slots] = c_kv[:active_tokens]
        q_v = q_full.view(tokens, local_heads, qk_nope + qk_rope)
        q_abs = torch_mod.bmm(q_v[:, :, :qk_nope].transpose(0, 1).contiguous(), w_uk).transpose(0, 1).contiguous()
        q_pe = q_v[:, :, qk_nope:].contiguous() if qk_rope > 0 else zero_rope

        if dcp > 1 and (not is_joint_pcp_dcp) and dist_group is not None:
            q_abs_list = [torch_mod.empty_like(q_abs) for _ in range(dcp)]
            q_pe_list = [torch_mod.empty_like(q_pe) for _ in range(dcp)]
            torch_mod.distributed.all_gather(q_abs_list, q_abs, group=dist_group)
            torch_mod.distributed.all_gather(q_pe_list, q_pe, group=dist_group)
            q_abs_exec = torch_mod.cat(q_abs_list, dim=1).contiguous()
            q_pe_exec = torch_mod.cat(q_pe_list, dim=1).contiguous()
        else:
            q_abs_exec, q_pe_exec = q_abs, q_pe

        if idx_state is not None or ext_sfa_state is not None:
            attn_lat = _exec_mla_sfa_decode(
                torch_mod, torch_npu_mod, h=h, q_abs_exec=q_abs_exec, q_pe_exec=q_pe_exec,
                kv_cache_4d=kv_cache_4d, kr_cache_4d=kr_cache_4d, block_table=block_table,
                idx_state=idx_state, ext_sfa_state=ext_sfa_state, batch=batch, q_len=q_len,
                tokens=tokens, idx_heads=idx_heads, idx_dim=idx_dim,
                eff_sparse_count=eff_sparse_count, scale_val=(qk_nope + rope_exec) ** -0.5,
                dcp=dcp, local_heads=local_heads, kv_lora=kv_lora, dist_group=dist_group,
                cand_block_size=cand_block_sz, owns_candidate_blocks=owns_cand_blocks, local_kv_len=local_kv_len,
            )
        else:
            qn_4d = q_abs_exec.view(batch, q_len, dcp_heads, kv_lora).transpose(1, 2).contiguous()
            qp_4d = q_pe_exec.view(batch, q_len, dcp_heads, rope_exec).transpose(1, 2).contiguous()
            fia_kw: dict[str, Any] = dict(
                query_rope=qp_4d, key_rope=kr_cache_3d,
                num_query_heads=dcp_heads, num_key_value_heads=1,
                input_layout="BNSD_NBSD", softmax_scale=(qk_nope + rope_exec) ** -0.5,
                block_table=block_table, block_size=blk,
                actual_seq_kvlen=seq_lens, return_softmax_lse=True,
            )
            if mask_mtp is not None:
                fia_kw["atten_mask"] = mask_mtp
                fia_kw["sparse_mode"] = CAUSAL_SPARSE_MODE
            out_nbsd, lse = torch_npu_mod.npu_fused_infer_attention_score_v2(
                qn_4d, kv_cache_3d, kv_cache_3d, **fia_kw,
            )
            attn_lat = (
                _dcp_lse_merge(
                    torch_mod, torch_npu_mod, out_nbsd, lse,
                    dcp=dcp, local_heads=local_heads, batch=batch, q_len=q_len,
                    tokens=tokens, kv_lora=kv_lora, dist_group=dist_group,
                    use_all_gather=is_joint_pcp_dcp,
                )
                if (dcp > 1 and dist_group is not None)
                else out_nbsd.reshape(local_heads, tokens, kv_lora)
            )

        v_out = torch_mod.bmm(attn_lat, w_uv).transpose(0, 1).reshape(tokens, local_heads * v_dim)
        if tp == 1 and o_proj_tp > 1 and dist_group is not None:
            res = o_fn(v_out.chunk(o_proj_tp, dim=-1)[o_chunk_idx].contiguous())
            torch_mod.distributed.all_reduce(res, group=dist_group)
        else:
            res = o_fn(v_out)
            if tp > 1 and dist_group is not None:
                torch_mod.distributed.all_reduce(res, group=dist_group)
        return engram_fn(res) if engram_fn is not None else res

    return _run_mla_decode, TIER_TORCHNPU_COMPOSITE, {
        "phase": "decode", "local_heads": local_heads, "dcp_heads": dcp_heads, "kv_lora_rank": kv_lora,
        "requested_indexer": use_indexer, "executed_indexer": executed_indexer, "uses_external_topk": uses_ext_topk,
        "candidate_filter": cand_filter, "idx_scan_len": idx_scan_len, "engram_executed": engram_fn is not None,
        "dcp": dcp, "pcp": pcp, "o_proj_tp": o_proj_tp,
    }


def _build_mla_attention(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    device: str,
    dist_group: Any = None,
    rank: int = 0,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer Absorbed MLA / Sparse-MLA (DSA/SFA) / DCP / PCP subgraph."""
    hidden = int(spec["hidden_size"])
    tp = max(1, int(spec.get("tp", 1)))
    o_proj_tp = max(1, int(spec.get("o_proj_tp", 1)))
    local_heads = max(1, int(spec["num_heads"]) // tp)
    q_lora = int(spec.get("q_lora_rank", 0))
    kv_lora = int(spec.get("kv_lora_rank", 512)) or 512
    qk_nope = int(spec.get("qk_nope_head_dim", 128)) or 128
    qk_rope = int(spec.get("qk_rope_head_dim", 64))
    v_dim = int(spec.get("v_head_dim", 128)) or 128
    batch = int(spec.get("sequences", 4))
    q_len = int(spec.get("q_len", 1))
    quant_mode = str(spec.get("quant_mode", "bf16"))
    if q_lora > 0:
        q_a_fn = _make_linear_fn(torch_mod, torch_npu_mod, hidden, q_lora, quant_mode, device)
        q_norm_w = torch_mod.ones((q_lora,), dtype=torch_mod.bfloat16, device=device)
        q_b_fn = _make_linear_fn(torch_mod, torch_npu_mod, q_lora, local_heads * (qk_nope + qk_rope), quant_mode, device)
    else:
        q_a_fn = q_norm_w = None
        q_b_fn = _make_linear_fn(torch_mod, torch_npu_mod, hidden, local_heads * (qk_nope + qk_rope), quant_mode, device)
    o_in_dim = (local_heads * v_dim) // o_proj_tp if (tp == 1 and o_proj_tp > 1) else (local_heads * v_dim)
    cfg = {
        "hidden": hidden, "tp": tp, "dcp": max(1, int(spec.get("dcp", 1))),
        "pcp": max(1, int(spec.get("pcp", 1))), "o_proj_tp": o_proj_tp,
        "local_heads": local_heads, "kv_lora": kv_lora, "qk_nope": qk_nope, "qk_rope": qk_rope,
        "rope_exec": qk_rope if qk_rope > 0 else ROPE_PAD_DIM, "v_dim": v_dim,
        "batch": batch, "q_len": q_len, "tokens": batch * q_len, "context": int(spec.get("context", 2048)),
        "quant_mode": quant_mode, "use_indexer": bool(spec.get("indexer", False)),
        "idx_heads": int(spec.get("index_heads", 64)) or 64, "idx_dim": int(spec.get("index_dim", 128)) or 128,
        "idx_topk": int(spec.get("index_topk", 2048)) or 2048,
        "blk": int(spec.get("block_size", DEFAULT_KV_BLOCK_SIZE)), "o_in_dim": o_in_dim,
        "norm_w": torch_mod.ones((hidden,), dtype=torch_mod.bfloat16, device=device),
        "kv_a_fn": _make_linear_fn(torch_mod, torch_npu_mod, hidden, kv_lora + qk_rope, quant_mode, device),
        "kv_norm_w": torch_mod.ones((kv_lora,), dtype=torch_mod.bfloat16, device=device),
        "q_a_fn": q_a_fn, "q_norm_w": q_norm_w, "q_b_fn": q_b_fn,
        "o_fn": _make_linear_fn(torch_mod, torch_npu_mod, o_in_dim, hidden, quant_mode, device),
    }
    if str(spec.get("phase", "decode")) == "prefill":
        return _build_mla_prefill(torch_mod, torch_npu_mod, cfg, device, dist_group)
    return _build_mla_decode(torch_mod, torch_npu_mod, spec, cfg, device, dist_group, rank)


def _build_kda_attention(
    torch_mod: Any,
    torch_npu_mod: Any,
    cfg: dict[str, Any],
    device: str,
    dist_group: Any,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer KDA linear-attention subgraph via ``fla_npu.ops.ascendc``."""
    try:
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda  # type: ignore[import-not-found]
    except (ImportError, AttributeError, OSError) as exc:
        raise CapabilityAbsentError(f"fla_npu_kda_absent:{type(exc).__name__}") from exc

    nk, nv, dk, dv = cfg["nk"], cfg["nv"], cfg["dk"], cfg["dv"]
    batch, q_len, tokens, proj_div = cfg["batch"], cfg["q_len"], cfg["tokens"], cfg["proj_div"]
    norm_w, in_fn, out_fn = cfg["norm_w"], cfg["in_fn"], cfg["out_fn"]
    a_log = torch_mod.zeros((nv,), dtype=torch_mod.float32, device=device)
    dt_bias = torch_mod.zeros((nv * dv,), dtype=torch_mod.float32, device=device)
    raw_gate = torch_mod.randn((1, tokens, nv, dv), dtype=torch_mod.bfloat16, device=device)
    beta = torch_mod.full((1, tokens, nv), 0.5, dtype=torch_mod.bfloat16, device=device)

    if cfg["phase"] == "prefill":
        init_state = torch_mod.zeros((1, nv, dv, dk), dtype=torch_mod.float32, device=device)
        beta_pre = torch_mod.full((1, tokens, nv), 0.5, dtype=torch_mod.float32, device=device)

        def _run_kda_prefill(x_in: Any) -> Any:
            h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
            q_p, k_p, v_p = in_fn(h).split((nk * dk, nk * dk, nv * dv), dim=-1)
            out, _fs, *_ = chunk_kda_fwd(
                q_p.view(1, tokens, nk, dk).contiguous(), k_p.view(1, tokens, nk, dk).contiguous(),
                v_p.view(1, tokens, nv, dv).contiguous(), raw_gate, beta_pre, dk ** -0.5, KDA_CHUNK_SIZE,
                layout="BSND", initial_state=init_state, output_final_state=True,
                cu_seqlens=None, chunk_indices=None, safe_gate=True, lower_bound=KDA_LOWER_BOUND,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=a_log, dt_bias=dt_bias,
                disable_recompute=False, return_intermediate_states=False, state_v_first=True,
            )
            res = out_fn(out.reshape(tokens, nv * dv))
            if proj_div > 1 and dist_group is not None:
                torch_mod.distributed.all_reduce(res, group=dist_group)
            return res

        return _run_kda_prefill, TIER_TORCHNPU_COMPOSITE, {
            "kind": "kda_chunk", "fla_npu_ops_used": True, "nk": nk, "nv": nv, "dk": dk,
        }

    state = torch_mod.zeros((batch + COLD_RING_MIN_SLOTS, nv, dv, dk), dtype=torch_mod.float32, device=device)
    cu_seq_t = torch_mod.arange(0, tokens + 1, q_len, dtype=torch_mod.int32, device=device)
    st_idx = torch_mod.arange(batch, dtype=torch_mod.int32, device=device)

    def _run_kda_decode(x_in: Any) -> Any:
        h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
        q_p, k_p, v_p = in_fn(h).split((nk * dk, nk * dk, nv * dv), dim=-1)
        out, _ = recurrent_kda(
            q_p.view(1, tokens, nk, dk).contiguous(), k_p.view(1, tokens, nk, dk).contiguous(),
            v_p.view(1, tokens, nv, dv).contiguous(), raw_gate, beta, state,
            cu_seqlens=cu_seq_t, ssm_state_indices=st_idx, A_log=a_log, dt_bias=dt_bias,
            layout="BSND", scale=dk ** -0.5, output_final_state=False, inplace_final_state=True,
            state_v_first=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, safe_gate=True,
            lower_bound=KDA_LOWER_BOUND,
        )
        res = out_fn(out.reshape(tokens, nv * dv))
        if proj_div > 1 and dist_group is not None:
            torch_mod.distributed.all_reduce(res, group=dist_group)
        return res

    return _run_kda_decode, TIER_TORCHNPU_COMPOSITE, {
        "kind": "kda_recurrent", "fla_npu_ops_used": True, "nk": nk, "nv": nv, "dk": dk,
    }


def _build_gdn_attention(
    torch_mod: Any,
    torch_npu_mod: Any,
    cfg: dict[str, Any],
    device: str,
    dist_group: Any,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer GDN (Gated DeltaNet) prefill or recurrent decode subgraph."""
    nk, nv, dk, dv = cfg["nk"], cfg["nv"], cfg["dk"], cfg["dv"]
    batch, q_len, tokens, proj_div = cfg["batch"], cfg["q_len"], cfg["tokens"], cfg["proj_div"]
    norm_w, in_fn, out_fn = cfg["norm_w"], cfg["in_fn"], cfg["out_fn"]
    beta = torch_mod.full((tokens, nv), 0.5, dtype=torch_mod.bfloat16, device=device)
    g = torch_mod.full((tokens, nv), -0.1, dtype=torch_mod.float32, device=device)

    if cfg["phase"] == "prefill":
        init_state = torch_mod.zeros((1, nv, dv, dk), dtype=torch_mod.bfloat16, device=device)
        asl = torch_mod.tensor([tokens], dtype=torch_mod.int32, device=device)

        def _run_gdn_prefill(x_in: Any) -> Any:
            h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
            q_p, k_p, v_p = in_fn(h).split((nk * dk, nk * dk, nv * dv), dim=-1)
            q_n = torch_mod.nn.functional.normalize(q_p.view(tokens, nk, dk).float(), p=2, dim=-1).to(torch_mod.bfloat16).contiguous()
            k_n = torch_mod.nn.functional.normalize(k_p.view(tokens, nk, dk).float(), p=2, dim=-1).to(torch_mod.bfloat16).contiguous()
            o, _fs = torch_npu_mod.npu_chunk_gated_delta_rule(
                q_n, k_n, v_p.view(tokens, nv, dv).contiguous(),
                beta=beta, initial_state=init_state, actual_seq_lengths=asl, scale=dk ** -0.5, g=g,
            )
            res = out_fn(o.reshape(tokens, nv * dv))
            if proj_div > 1 and dist_group is not None:
                torch_mod.distributed.all_reduce(res, group=dist_group)
            return res

        return _run_gdn_prefill, TIER_TORCHNPU_COMPOSITE, {
            "kind": "gdn_chunk", "fla_npu_ops_used": False, "nk": nk, "nv": nv, "dk": dk,
        }

    try:
        from fla_npu.ops.ascendc import recurrent_gated_delta_rule as fla_gdn_rec  # type: ignore[import-not-found]
    except (ImportError, AttributeError):
        fla_gdn_rec = None

    state_dtype = torch_mod.float32 if fla_gdn_rec is not None else torch_mod.bfloat16
    state = torch_mod.zeros((batch + COLD_RING_MIN_SLOTS, nv, dv, dk), dtype=state_dtype, device=device)
    st_idx = torch_mod.arange(batch, dtype=torch_mod.int32, device=device)
    if q_len > 1:
        st_idx = st_idx.repeat_interleave(q_len).contiguous()
    asl = torch_mod.tensor([0] + [q_len] * batch, dtype=torch_mod.int32, device=device)

    def _run_gdn_decode(x_in: Any) -> Any:
        h, _ = torch_npu_mod.npu_rms_norm(x_in, norm_w, DEFAULT_RMS_EPS)
        q_p, k_p, v_p = in_fn(h).split((nk * dk, nk * dk, nv * dv), dim=-1)
        q_n = torch_mod.nn.functional.normalize(q_p.view(tokens, nk, dk).float(), p=2, dim=-1).to(torch_mod.bfloat16).contiguous()
        k_n = torch_mod.nn.functional.normalize(k_p.view(tokens, nk, dk).float(), p=2, dim=-1).to(torch_mod.bfloat16).contiguous()
        v_n = v_p.view(tokens, nv, dv).contiguous()
        if fla_gdn_rec is not None:
            o = fla_gdn_rec(q_n, k_n, v_n, state, g=g, beta=beta, scale=dk ** -0.5, actual_seq_lengths=asl, ssm_state_indices=st_idx)
        else:
            o = torch_npu_mod.npu_recurrent_gated_delta_rule(
                q_n, k_n, v_n, state=state, beta=beta, scale=dk ** -0.5, actual_seq_lengths=asl, ssm_state_indices=st_idx, g=g,
            )
        res = out_fn(o.reshape(tokens, nv * dv))
        if proj_div > 1 and dist_group is not None:
            torch_mod.distributed.all_reduce(res, group=dist_group)
        return res

    return _run_gdn_decode, TIER_TORCHNPU_COMPOSITE, {
        "kind": "gdn_recurrent", "fla_npu_ops_used": fla_gdn_rec is not None, "nk": nk, "nv": nv, "dk": dk,
    }


def _build_linear_attention(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    device: str,
    dist_group: Any = None,
) -> tuple[Callable[[Any], Any], str, dict[str, Any]]:
    """Build 1-layer GDN or KDA linear-attention subgraph."""
    hidden = int(spec["hidden_size"])
    proj_div = max(max(1, int(spec.get("tp", 1))), max(1, int(spec.get("linear_proj_tp", 1))))
    nk = max(1, int(spec.get("linear_num_k_heads", 16)) // proj_div)
    nv = max(1, int(spec.get("linear_num_v_heads", 32)) // proj_div)
    dk = int(spec.get("linear_k_head_dim", 128)) or 128
    dv = int(spec.get("linear_v_head_dim", 128)) or 128
    batch = int(spec.get("sequences", 4))
    q_len = int(spec.get("q_len", 1))
    tokens = batch * q_len
    quant_mode = str(spec.get("quant_mode", "bf16"))
    cfg = {
        "nk": nk, "nv": nv, "dk": dk, "dv": dv, "batch": batch, "q_len": q_len, "tokens": tokens,
        "proj_div": proj_div, "phase": str(spec.get("phase", "decode")),
        "norm_w": torch_mod.ones((hidden,), dtype=torch_mod.bfloat16, device=device),
        "in_fn": _make_linear_fn(torch_mod, torch_npu_mod, hidden, 2 * nk * dk + nv * dv, quant_mode, device),
        "out_fn": _make_linear_fn(torch_mod, torch_npu_mod, nv * dv, hidden, quant_mode, device),
    }
    if str(spec["attention"]) == "kda":
        return _build_kda_attention(torch_mod, torch_npu_mod, cfg, device, dist_group)
    return _build_gdn_attention(torch_mod, torch_npu_mod, cfg, device, dist_group)


def _init_moe_weights(
    torch_mod: Any,
    torch_npu_mod: Any,
    local_e: int,
    hidden: int,
    moe_i: int,
    quant_mode: str,
    device: str,
) -> tuple[Any, ...]:
    """Allocate synthetic BF16, W8A8/C8, W4A8, MXFP8, or MXFP4 expert weights."""
    if quant_mode in ("w8a8", "c8"):
        w1_q = torch_mod.randint(INT8_MIN_VAL, INT8_MAX_VAL, (local_e, hidden, 2 * moe_i), dtype=torch_mod.int8, device=device)
        w1_s = torch_mod.full((local_e, 2 * moe_i), 1.0 / (math.sqrt(float(hidden)) * INT8_SCALE_NORM), dtype=torch_mod.bfloat16, device=device)
        w2_q = torch_mod.randint(INT8_MIN_VAL, INT8_MAX_VAL, (local_e, moe_i, hidden), dtype=torch_mod.int8, device=device)
        w2_s = torch_mod.full((local_e, hidden), 1.0 / (math.sqrt(float(moe_i)) * INT8_SCALE_NORM), dtype=torch_mod.bfloat16, device=device)
        return (w1_q, w1_s, w2_q, w2_s)
    if quant_mode == "w4a8":
        w1_raw = torch_mod.randint(INT4_MIN_VAL, INT4_MAX_VAL, (local_e, hidden, 2 * moe_i), dtype=torch_mod.int32, device=device)
        w2_raw = torch_mod.randint(INT4_MIN_VAL, INT4_MAX_VAL, (local_e, moe_i, hidden), dtype=torch_mod.int32, device=device)
        w1_q = torch_npu_mod.npu_convert_weight_to_int4pack(w1_raw)
        w2_q = torch_npu_mod.npu_convert_weight_to_int4pack(w2_raw)
        w1_s = torch_mod.full((local_e, 2 * moe_i), 1.0 / (math.sqrt(float(hidden)) * float(INT4_MAX_VAL)), dtype=torch_mod.bfloat16, device=device)
        w2_s = torch_mod.full((local_e, hidden), 1.0 / (math.sqrt(float(moe_i)) * float(INT4_MAX_VAL)), dtype=torch_mod.bfloat16, device=device)
        return (w1_q, w1_s, w2_q, w2_s)
    if quant_mode in ("mxfp8", "mxfp4"):
        is_fp4 = quant_mode == "mxfp4"
        dst_dt = getattr(torch_npu_mod, "float4_e2m1fn_x2", None) if is_fp4 else getattr(torch_mod, "float8_e4m3fn", None)
        if dst_dt is None:
            raise CapabilityAbsentError(f"{quant_mode}_dtype_absent_in_torch_npu")
        w1_raw = torch_mod.randn((local_e, 2 * moe_i, hidden), dtype=torch_mod.bfloat16, device=device) * (hidden ** -0.5)
        w2_raw = torch_mod.randn((local_e, hidden, moe_i), dtype=torch_mod.bfloat16, device=device) * (moe_i ** -0.5)
        w1_q, w1_s = torch_npu_mod.npu_dynamic_mx_quant(w1_raw, dst_type=dst_dt)
        w2_q, w2_s = torch_npu_mod.npu_dynamic_mx_quant(w2_raw, dst_type=dst_dt)
        del w1_raw, w2_raw
        return (w1_q.transpose(1, 2), w1_s.transpose(1, 2), w2_q.transpose(1, 2), w2_s.transpose(1, 2))
    w1_bf = torch_mod.randn((local_e, hidden, 2 * moe_i), dtype=torch_mod.bfloat16, device=device) * (hidden ** -0.5)
    w2_bf = torch_mod.randn((local_e, moe_i, hidden), dtype=torch_mod.bfloat16, device=device) * (moe_i ** -0.5)
    return (w1_bf, w2_bf)


def _exec_moe_local_gmm(
    torch_mod: Any,
    torch_npu_mod: Any,
    comm_in: Any,
    group_list: Any,
    weights: tuple[Any, ...],
    quant_mode: str,
    per_tok_scale: Any,
    hidden: int,
    device: str,
) -> Any:
    """Execute BF16, W8A8/C8, W4A8, MXFP8, or MXFP4 GroupedMatmul experts on locally received tokens."""
    if comm_in.shape[0] == 0:
        return torch_mod.empty((0, hidden), dtype=torch_mod.bfloat16, device=device)
    if quant_mode in ("w8a8", "c8", "w4a8"):
        w1_q, w1_s, w2_q, w2_s = weights
        in_x, in_s = (comm_in, per_tok_scale) if per_tok_scale is not None else torch_npu_mod.npu_dynamic_quant(comm_in)
        g1 = torch_npu_mod.npu_grouped_matmul(
            [in_x], [w1_q], scale=[w1_s], per_token_scale=[in_s],
            group_list=group_list, split_item=GMM_SPLIT_ITEM_3, group_type=0,
            group_list_type=GMM_GROUP_LIST_COUNT, output_dtype=torch_mod.bfloat16,
        )[0]
        act_q, act_s = torch_npu_mod.npu_dynamic_quant(torch_npu_mod.npu_swiglu(g1))
        return torch_npu_mod.npu_grouped_matmul(
            [act_q], [w2_q], scale=[w2_s], per_token_scale=[act_s],
            group_list=group_list, split_item=GMM_SPLIT_ITEM_3, group_type=0,
            group_list_type=GMM_GROUP_LIST_COUNT, output_dtype=torch_mod.bfloat16,
        )[0]
    if quant_mode in ("mxfp8", "mxfp4"):
        w1_q, w1_s, w2_q, w2_s = weights
        dst_dt = torch_npu_mod.float4_e2m1fn_x2 if quant_mode == "mxfp4" else torch_mod.float8_e4m3fn
        in_x, in_s = torch_npu_mod.npu_dynamic_mx_quant(comm_in, dst_type=dst_dt)
        mx_kw = dict(
            group_list=group_list, split_item=GMM_SPLIT_ITEM_3, group_type=0,
            group_list_type=GMM_GROUP_LIST_COUNT, output_dtype=torch_mod.bfloat16,
            scale_dtype=torch_npu_mod.float8_e8m0fnu, per_token_scale_dtype=torch_npu_mod.float8_e8m0fnu,
        )
        g1 = torch_npu_mod.npu_grouped_matmul([in_x], [w1_q], scale=[w1_s], per_token_scale=[in_s], **mx_kw)[0]
        act_q, act_s = torch_npu_mod.npu_dynamic_mx_quant(torch_npu_mod.npu_swiglu(g1), dst_type=dst_dt)
        return torch_npu_mod.npu_grouped_matmul([act_q], [w2_q], scale=[w2_s], per_token_scale=[act_s], **mx_kw)[0]
    w1_bf, w2_bf = weights
    g1 = torch_npu_mod.npu_grouped_matmul(
        [comm_in], [w1_bf], group_list=group_list, split_item=GMM_SPLIT_ITEM_3,
        group_type=0, group_list_type=GMM_GROUP_LIST_COUNT,
    )[0]
    return torch_npu_mod.npu_grouped_matmul(
        [torch_npu_mod.npu_swiglu(g1)], [w2_bf], group_list=group_list, split_item=GMM_SPLIT_ITEM_3,
        group_type=0, group_list_type=GMM_GROUP_LIST_COUNT,
    )[0]


def _run_moe_owner_routed_ep(
    torch_mod: Any,
    torch_npu_mod: Any,
    h: Any,
    idx_top: Any,
    weights: tuple[Any, ...],
    *,
    quant_mode: str,
    world_ep: int,
    local_e: int,
    local_top_k: int,
    hidden: int,
    device: str,
    dist_group: Any,
) -> Any:
    """Zero-D2H-sync static-capacity owner-routed EP AllToAll dispatch -> local GMM -> AllToAll combine."""
    tokens = int(h.shape[0])
    total_e = world_ep * local_e
    n_assign = tokens * local_top_k
    per_rank_cap = min(n_assign, max(1, math.ceil(n_assign / world_ep) + max(2, math.ceil(math.sqrt(n_assign)))))
    padded_total = world_ep * per_rank_cap
    global_idx = (idx_top[:, :local_top_k] % total_e).to(torch_mod.int64).reshape(-1)
    tok_ids = torch_mod.arange(tokens, dtype=torch_mod.int64, device=device).repeat_interleave(local_top_k)

    # Sort by global_idx = owner_rank * local_e + local_e_id and scatter into static per-owner slots (zero D2H sync)
    send_order = torch_mod.argsort(global_idx, stable=True)
    sorted_g = global_idx[send_order]
    sorted_x = h[tok_ids[send_order]].contiguous()
    sorted_owner = torch_mod.div(sorted_g, local_e, rounding_mode="floor")
    owner_counts = torch_mod.zeros((world_ep,), dtype=torch_mod.int64, device=device)
    owner_counts.scatter_add_(0, sorted_owner, torch_mod.ones((n_assign,), dtype=torch_mod.int64, device=device))
    owner_offsets = torch_mod.cumsum(owner_counts, dim=0) - owner_counts
    pos_in_owner = torch_mod.arange(n_assign, dtype=torch_mod.int64, device=device) - owner_offsets[sorted_owner]
    valid_send = pos_in_owner < per_rank_cap
    flat_dst = sorted_owner * per_rank_cap + pos_in_owner.clamp(max=per_rank_cap - 1)
    safe_dst = torch_mod.where(valid_send, flat_dst, flat_dst * 0 + padded_total)

    send_x_buf = torch_mod.zeros((padded_total + 1, hidden), dtype=sorted_x.dtype, device=device)
    send_x_buf[safe_dst] = sorted_x
    send_x = send_x_buf[:padded_total].contiguous()
    send_meta_buf = torch_mod.zeros((padded_total + 1,), dtype=torch_mod.int64, device=device)
    send_meta_buf[safe_dst] = (sorted_g % local_e) + 1
    send_meta = send_meta_buf[:padded_total]

    if quant_mode in ("w8a8", "c8"):
        send_q, send_s = torch_npu_mod.npu_dynamic_quant(send_x)
        scale_bytes = send_s.to(torch_mod.float32).unsqueeze(-1).contiguous().view(torch_mod.int8)
        meta_bytes = send_meta.to(torch_mod.int32).unsqueeze(-1).contiguous().view(torch_mod.int8)
        send_packed = torch_mod.cat([send_q.contiguous(), scale_bytes, meta_bytes], dim=-1).contiguous()
        recv_packed = torch_mod.empty_like(send_packed)
        torch_mod.distributed.all_to_all_single(recv_packed, send_packed, group=dist_group)
        recv_x = recv_packed[:, :hidden].contiguous()
        recv_s = recv_packed[:, hidden:hidden + FP32_BYTES].contiguous().view(torch_mod.float32).squeeze(-1)
        recv_meta = recv_packed[:, hidden + FP32_BYTES:].contiguous().view(torch_mod.int32).squeeze(-1).to(torch_mod.int64)
    else:
        meta_hi = torch_mod.div(send_meta, DEFAULT_KV_BLOCK_SIZE, rounding_mode="floor").to(send_x.dtype).unsqueeze(-1)
        meta_lo = (send_meta % DEFAULT_KV_BLOCK_SIZE).to(send_x.dtype).unsqueeze(-1)
        send_packed = torch_mod.cat([send_x, meta_hi, meta_lo], dim=-1).contiguous()
        recv_packed = torch_mod.empty_like(send_packed)
        torch_mod.distributed.all_to_all_single(recv_packed, send_packed, group=dist_group)
        recv_x = recv_packed[:, :hidden].contiguous()
        recv_s = None
        recv_meta = (
            recv_packed[:, hidden].to(torch_mod.int64) * DEFAULT_KV_BLOCK_SIZE
            + recv_packed[:, hidden + 1].to(torch_mod.int64)
        )

    recv_valid = recv_meta > 0
    recv_e = torch_mod.where(recv_valid, recv_meta - 1, recv_meta * 0 + (local_e - 1)).clamp(min=0, max=local_e - 1)
    recv_order = torch_mod.argsort(recv_e, stable=True)
    recv_gl = torch_mod.zeros((local_e,), dtype=torch_mod.int64, device=device)
    recv_gl.scatter_add_(0, recv_e, torch_mod.ones((padded_total,), dtype=torch_mod.int64, device=device))
    g2_sorted = _exec_moe_local_gmm(
        torch_mod, torch_npu_mod,
        recv_x[recv_order].contiguous(), recv_gl, weights, quant_mode,
        recv_s[recv_order].contiguous() if recv_s is not None else None,
        hidden, device,
    )
    g2_unperm = torch_mod.empty_like(g2_sorted)
    g2_unperm[recv_order] = g2_sorted
    g2_masked = (g2_unperm * recv_valid.to(g2_unperm.dtype).unsqueeze(-1)).contiguous()
    comb_buf = torch_mod.empty_like(g2_masked)
    torch_mod.distributed.all_to_all_single(comb_buf, g2_masked, group=dist_group)
    comb_valid = comb_buf[flat_dst] * valid_send.to(comb_buf.dtype).unsqueeze(-1)
    comb_orig = torch_mod.empty((n_assign, hidden), dtype=comb_buf.dtype, device=device)
    comb_orig[send_order] = comb_valid
    return comb_orig.view(tokens, local_top_k, hidden)


def _single_card_ep_routed_counts(
    tokens: int,
    local_top_k: int,
    *,
    dp: int,
    tp: int,
    ep: int,
    pcp: int,
    sequences: int,
    active_sequences: int,
    q_len: int,
    overlay: str,
) -> tuple[int, int]:
    """Return single-EP-rank ``(routed_tokens, eff_top_k)`` matching ``mindie_sim`` TP+EP sharding."""
    eff_tokens = max(1, int(tokens))
    if ep > 1 and active_sequences < sequences and "eager" in overlay.lower():
        dp_eff = max(dp, math.ceil(ep / max(1, tp * pcp)))
        eff_tokens = max(1, math.ceil(max(1, active_sequences) / max(1, dp_eff))) * max(1, q_len)
    if ep <= 1 or tp <= 1:
        return eff_tokens, local_top_k
    if eff_tokens >= tp and eff_tokens % tp == 0:
        return eff_tokens // tp, local_top_k
    if local_top_k >= tp and local_top_k % tp == 0:
        return eff_tokens, local_top_k // tp
    r_tok = max(1, eff_tokens // tp)
    target_slots = max(1, (eff_tokens * local_top_k) // tp)
    r_k = max(1, min(local_top_k, round(target_slots / r_tok)))
    return r_tok, r_k


def _build_mhc_block_wrapper(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    device: str,
    *,
    is_ffn: bool,
) -> tuple[Callable[[Callable[..., Any]], Callable[..., Any]], dict[str, Any]] | None:
    """Build synthetic Hyper-Connections (low-rank or Sinkhorn mHC) pre/post wrapper when ``hc_mult > 1``."""
    hc_mult = int(spec.get("hc_mult", 0) or 0)
    if hc_mult <= 1:
        return None
    hidden = int(spec["hidden_size"])
    hc_lowrank = int(spec.get("hc_lowrank", 0) or 0)
    hc_sinkhorn = int(spec.get("hc_sinkhorn_iters", 0) or 0)
    wide_dim = hc_mult * hidden
    if hc_lowrank > 0:
        hc_norm_w = torch_mod.ones((wide_dim,), dtype=torch_mod.bfloat16, device=device)
        down_out = hc_lowrank + hc_mult * hc_mult
        hc_down = _make_linear_fn(torch_mod, torch_npu_mod, wide_dim, down_out, "bf16", device)
        hc_up = _make_linear_fn(torch_mod, torch_npu_mod, hc_lowrank, wide_dim, "bf16", device)
        mix_w = torch_mod.full((hc_mult,), 1.0 / float(hc_mult), dtype=torch_mod.bfloat16, device=device)

        def _wrap_lowrank(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
            def _run(*args: Any) -> Any:
                x_in = args[0]
                streams = x_in.unsqueeze(1).expand(-1, hc_mult, -1).reshape(x_in.shape[0], wide_dim).contiguous()
                normed, _ = torch_npu_mod.npu_rms_norm(streams, hc_norm_w, DEFAULT_RMS_EPS)
                proj = hc_down(normed)
                low = torch_mod.nn.functional.silu(proj[:, :hc_lowrank])
                up = hc_up(low).view(x_in.shape[0], hc_mult, hidden)
                x_mix = (up * mix_w.view(1, hc_mult, 1)).sum(dim=1)
                sub_out = inner_fn(x_mix, *args[1:]) if is_ffn else inner_fn(x_mix)
                post = up + sub_out.unsqueeze(1)
                return post[:, 0, :].contiguous()

            return _run

        return _wrap_lowrank, {"mhc": "lowrank", "hc_mult": hc_mult, "hc_lowrank": hc_lowrank}

    proj_dim = hc_mult * (hc_mult + 2)
    hc_proj = _make_linear_fn(torch_mod, torch_npu_mod, wide_dim, proj_dim, "bf16", device)
    iters = max(1, hc_sinkhorn) if hc_sinkhorn > 0 else 4

    def _wrap_sinkhorn(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
        def _run(*args: Any) -> Any:
            x_in = args[0]
            t_len = x_in.shape[0]
            streams = x_in.unsqueeze(1).expand(-1, hc_mult, -1).reshape(t_len, wide_dim).contiguous()
            s_fp32 = streams.to(torch_mod.float32)
            inv_rms = torch_mod.rsqrt((s_fp32 * s_fp32).mean(dim=-1, keepdim=True) + DEFAULT_RMS_EPS)
            coeff = hc_proj(streams).to(torch_mod.float32) * inv_rms
            mat = torch_mod.exp(coeff[:, : hc_mult * hc_mult].view(t_len, hc_mult, hc_mult).clamp(-4.0, 4.0))
            # Single row+col normalize step proxies the fused Sinkhorn kernel without 20x eager launch inflation.
            mat = mat / (mat.sum(dim=-1, keepdim=True) + DEFAULT_RMS_EPS)
            mat = mat / (mat.sum(dim=-2, keepdim=True) + DEFAULT_RMS_EPS)
            pre_w = torch_mod.sigmoid(coeff[:, hc_mult * hc_mult : hc_mult * (hc_mult + 1)]).to(x_in.dtype)
            s_3d = streams.view(t_len, hc_mult, hidden)
            x_mix = (s_3d * pre_w.unsqueeze(-1)).sum(dim=1)
            sub_out = inner_fn(x_mix, *args[1:]) if is_ffn else inner_fn(x_mix)
            mixed = torch_mod.bmm(mat.to(x_in.dtype), s_3d) + sub_out.unsqueeze(1)
            return mixed[:, 0, :].contiguous()

        return _run

    return _wrap_sinkhorn, {"mhc": "sinkhorn", "hc_mult": hc_mult, "hc_sinkhorn_iters": iters}


def _build_ffn_or_moe(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    device: str,
    dist_group: Any = None,
) -> tuple[Callable[[Any, Any], Any], str, dict[str, Any]]:
    """Build 1-layer Dense FFN or Owner-Routed MoE with real residual, combine, and shared addition."""
    hidden = int(spec["hidden_size"])
    ffn = str(spec.get("ffn", "dense"))
    dp = max(1, int(spec.get("dp", 1)))
    tp = max(1, int(spec.get("tp", 1)))
    ep = max(1, int(spec.get("ep", 1)))
    batch = int(spec.get("sequences", 4))
    active_seqs = int(spec.get("active_sequences", batch))
    q_len = int(spec.get("q_len", 1))
    phase = str(spec.get("phase", "decode"))
    overlay = str(spec.get("overlay", "canonical"))
    pcp = max(1, int(spec.get("pcp", 1)))
    tokens = max(1, (batch * q_len) // pcp) if phase == "prefill" else (batch * q_len)
    quant_mode = str(spec.get("quant_mode", "bf16"))
    norm_w = torch_mod.ones((hidden,), dtype=torch_mod.bfloat16, device=device)

    if ffn != "moe" or int(spec.get("num_experts", 0)) <= 0:
        inter = max(1, int(spec.get("intermediate_size", hidden * 4)) // tp)
        gate_up_fn = _make_linear_fn(torch_mod, torch_npu_mod, hidden, 2 * inter, quant_mode, device)
        down_fn = _make_linear_fn(torch_mod, torch_npu_mod, inter, hidden, quant_mode, device)

        def _run_dense(x_in: Any, res_in: Any) -> Any:
            h, _, updated_res = torch_npu_mod.npu_add_rms_norm(x_in, res_in, norm_w, DEFAULT_RMS_EPS)
            out = down_fn(torch_npu_mod.npu_swiglu(gate_up_fn(h)))
            if tp > 1 and dist_group is not None:
                torch_mod.distributed.all_reduce(out, group=dist_group)
            return out + updated_res

        return _run_dense, TIER_TORCHNPU_COMPOSITE, {"ffn": "dense", "local_intermediate": inter}

    num_experts = int(spec["num_experts"])
    top_k = max(1, int(spec["top_k"]))
    moe_i = int(spec["moe_intermediate_size"])
    shared_n = int(spec.get("num_shared_experts", 0))
    shared_i = int(spec.get("shared_intermediate_size", moe_i))
    latent_dim = int(spec.get("expert_latent_size", 0) or 0)
    use_latent = 0 < latent_dim < hidden
    moe_in_dim = latent_dim if use_latent else hidden
    local_e = max(1, math.ceil(num_experts / ep))
    elem_b = 1 if quant_mode in ("w8a8", "c8", "w4a8", "mxfp8", "mxfp4") else 2
    est_bytes = local_e * 3 * moe_in_dim * moe_i * elem_b
    if est_bytes > MAX_SYNTH_WEIGHT_BYTES:
        raise MemoryError(f"1layer_moe_weights_exceed_budget:{est_bytes}>{MAX_SYNTH_WEIGHT_BYTES}")

    world_ep = ep if dist_group is not None else 1
    router_w = torch_mod.randn((hidden, num_experts), dtype=torch_mod.bfloat16, device=device) * (hidden ** -0.5)
    if world_ep == 1 and top_k > local_e:
        raise ValueError(f"top_k_exceeds_local_experts:{top_k}>{local_e}")
    local_top_k = min(top_k, world_ep * local_e)
    routed_tokens, eff_top_k = (
        (tokens, local_top_k) if world_ep > 1 else _single_card_ep_routed_counts(
            tokens, local_top_k, dp=dp, tp=tp, ep=ep, pcp=pcp,
            sequences=batch, active_sequences=active_seqs, q_len=q_len, overlay=overlay,
        )
    )
    active_tokens = routed_tokens * eff_top_k
    weights = _init_moe_weights(torch_mod, torch_npu_mod, local_e, moe_in_dim, moe_i, quant_mode, device)
    latent_fns = (
        _make_linear_fn(torch_mod, torch_npu_mod, hidden, latent_dim, quant_mode, device),
        _make_linear_fn(torch_mod, torch_npu_mod, latent_dim, hidden, quant_mode, device),
    ) if use_latent else None

    shared_fns = None
    if shared_n > 0 and shared_i > 0:
        shared_fns = (
            _make_linear_fn(torch_mod, torch_npu_mod, hidden, 2 * shared_n * shared_i, quant_mode, device),
            _make_linear_fn(torch_mod, torch_npu_mod, shared_n * shared_i, hidden, quant_mode, device),
        )

    def _run_moe(x_in: Any, res_in: Any) -> Any:
        h, _, updated_res = torch_npu_mod.npu_add_rms_norm(x_in, res_in, norm_w, DEFAULT_RMS_EPS)
        logits = torch_mod.matmul(h, router_w).to(torch_mod.float32)
        w_top, idx_top, _ = torch_npu_mod.npu_moe_gating_top_k_softmax(logits, k=top_k)
        h_moe_in = latent_fns[0](h) if latent_fns is not None else h
        if world_ep > 1 and dist_group is not None:
            routed_out = _run_moe_owner_routed_ep(
                torch_mod, torch_npu_mod, h_moe_in, idx_top, weights,
                quant_mode=quant_mode, world_ep=world_ep, local_e=local_e,
                local_top_k=local_top_k, hidden=moe_in_dim, device=device, dist_group=dist_group,
            )
            w_norm = w_top[:, :local_top_k].to(routed_out.dtype).unsqueeze(-1)
            out_moe = (routed_out * w_norm).sum(dim=1)
        else:
            h_sub = h_moe_in[:routed_tokens] if routed_tokens < tokens else h_moe_in
            local_idx = (idx_top[:routed_tokens, :eff_top_k] % local_e).to(torch_mod.int32)
            exp_x, exp_row, exp_tok, exp_scale = torch_npu_mod.npu_moe_init_routing_v2(
                h_sub, local_idx, active_num=active_tokens, expert_capacity=0, expert_num=local_e,
                drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True,
                quant_mode=1 if quant_mode in ("w8a8", "c8") else -1,
                active_expert_range=[0, local_e], row_idx_type=0,
            )
            g2 = _exec_moe_local_gmm(
                torch_mod, torch_npu_mod, exp_x, exp_tok.to(torch_mod.int64),
                weights, quant_mode, exp_scale if quant_mode in ("w8a8", "c8") else None, moe_in_dim, device,
            )
            routed_out = g2[exp_row.to(torch_mod.int64)].view(routed_tokens, eff_top_k, moe_in_dim)
            w_norm = w_top[:routed_tokens, :eff_top_k].to(routed_out.dtype).unsqueeze(-1)
            comb = (routed_out * w_norm).sum(dim=1)
            if routed_tokens < tokens:
                out_moe = torch_mod.zeros((tokens, moe_in_dim), dtype=comb.dtype, device=device)
                out_moe[:routed_tokens] = comb
            else:
                out_moe = comb
        out = latent_fns[1](out_moe) if latent_fns is not None else out_moe
        if shared_fns is not None:
            s_gu, s_dn = shared_fns
            out = out + s_dn(torch_npu_mod.npu_swiglu(s_gu(h)))
        return out + updated_res

    return _run_moe, TIER_TORCHNPU_COMPOSITE, {
        "ffn": "moe", "local_experts": local_e, "top_k": top_k, "moe_intermediate": moe_i,
        "shared_experts": shared_n, "ep_comm_world": world_ep,
        "routed_tokens": routed_tokens, "routed_top_k": eff_top_k, "expert_latent_size": latent_dim,
    }


def _sync_readiness(torch_mod: Any, ok: bool, device: str, dist_group: Any) -> bool:
    """Agree on rank readiness across all ranks before entering collective work."""
    flag = 1 if ok else 0
    if dist_group is None:
        return flag == 1
    flag_t = torch_mod.tensor([flag], dtype=torch_mod.int32, device=device)
    torch_mod.distributed.all_reduce(flag_t, op=torch_mod.distributed.ReduceOp.MIN, group=dist_group)
    return int(flag_t.item()) == 1


def _validate_case_spec(spec: dict[str, Any]) -> None:
    """Strictly reject unknown attention, FFN, quantization, or phase values."""
    attn_kind = str(spec.get("attention", "gqa"))
    ffn_kind = str(spec.get("ffn", "dense"))
    quant_mode = str(spec.get("quant_mode", "bf16"))
    phase = str(spec.get("phase", "decode"))
    if attn_kind not in VALID_ATTENTIONS:
        raise ValueError(f"unsupported_attention_family:{attn_kind}")
    if ffn_kind not in VALID_FFNS:
        raise ValueError(f"unsupported_ffn_family:{ffn_kind}")
    if quant_mode not in VALID_QUANTS:
        raise ValueError(f"unsupported_quant_mode:{quant_mode}")
    if phase not in VALID_PHASES:
        raise ValueError(f"unsupported_phase:{phase}")
    if ffn_kind == "moe" and int(spec.get("num_experts", 0)) <= 0:
        raise ValueError(f"moe_requires_positive_num_experts:{spec.get('num_experts', 0)}")


def _execute_single_case(
    torch_mod: Any,
    torch_npu_mod: Any,
    spec: dict[str, Any],
    device: str,
    dev_index: int,
    *,
    rank: int = 0,
    dist_group: Any = None,
    blocker_pair: tuple[Any, Any] | None = None,
    l2_flush_pair: tuple[Any, Any] | None = None,
    cpu_store: Any = None,
    world_size: int = 1,
    case_idx: int = 0,
) -> dict[str, Any]:
    """Run chained 1-layer Attention -> FFN/MoE synthetic subgraphs with two-phase rank sync."""
    attn_kind = str(spec["attention"])
    arch = str(spec.get("architecture", attn_kind))
    ffn_kind = str(spec.get("ffn", "dense"))
    quant_mode = str(spec.get("quant_mode", "bf16"))
    phase = str(spec.get("phase", "decode"))
    batch = int(spec.get("sequences", 4))
    q_len = int(spec.get("q_len", 1))
    pcp = max(1, int(spec.get("pcp", 1)))
    hidden = int(spec["hidden_size"])
    local_tokens = max(1, (batch * q_len) // pcp) if phase == "prefill" else (batch * q_len)
    base_record: dict[str, Any] = {
        "case_id": str(spec["case_id"]), "model": str(spec["model"]),
        "layer_index": int(spec.get("layer_index", 0)),
        "attention": attn_kind, "architecture": arch, "ffn": ffn_kind,
        "overlay": str(spec.get("overlay", "canonical")), "phase": phase,
        "parallel": {
            "dp": int(spec.get("dp", 1)), "tp": int(spec.get("tp", 1)),
            "ep": int(spec.get("ep", 1)), "dcp": int(spec.get("dcp", 1)),
            "pcp": pcp, "o_proj_tp": int(spec.get("o_proj_tp", 1)),
            "linear_proj_tp": int(spec.get("linear_proj_tp", 1)),
        },
        "workload": {
            "sequences": batch, "active_sequences": int(spec.get("active_sequences", batch)),
            "q_len": q_len, "context": int(spec.get("context", 2048)), "quant_mode": quant_mode,
        },
    }

    setup_err: Exception | None = None
    attn_fn = ffn_fn = None
    attn_meta: dict[str, Any] = {}
    ffn_meta: dict[str, Any] = {}
    x_layer = None
    try:
        _validate_case_spec(spec)
        est_alloc = _estimate_case_allocation_bytes(spec)
        _check_free_hbm(torch_mod, dev_index, required_bytes=est_alloc + HBM_HEADROOM_BYTES)
        x_layer = torch_mod.randn((local_tokens, hidden), dtype=torch_mod.bfloat16, device=device)
        if attn_kind in ("gdn", "kda"):
            attn_fn, _, attn_meta = _build_linear_attention(torch_mod, torch_npu_mod, spec, device, dist_group=dist_group)
        elif attn_kind == "mla":
            attn_fn, _, attn_meta = _build_mla_attention(torch_mod, torch_npu_mod, spec, device, dist_group=dist_group, rank=rank)
        else:
            attn_fn, _, attn_meta = _build_gqa_attention(torch_mod, torch_npu_mod, spec, device, dist_group=dist_group)
        ffn_fn, _, ffn_meta = _build_ffn_or_moe(torch_mod, torch_npu_mod, spec, device, dist_group=dist_group)
        mhc_attn = _build_mhc_block_wrapper(torch_mod, torch_npu_mod, spec, device, is_ffn=False)
        if mhc_attn is not None:
            attn_fn = mhc_attn[0](attn_fn)
            attn_meta = {**attn_meta, **mhc_attn[1]}
        mhc_ffn = _build_mhc_block_wrapper(torch_mod, torch_npu_mod, spec, device, is_ffn=True)
        if mhc_ffn is not None:
            ffn_fn = mhc_ffn[0](ffn_fn)
            ffn_meta = {**ffn_meta, **mhc_ffn[1]}
    except Exception as exc:  # noqa: BLE001
        setup_err = exc

    all_ready = _sync_readiness(torch_mod, setup_err is None, device, dist_group)
    fwd_err: Exception | None = None
    sample_attn = None
    if all_ready:
        try:
            assert attn_fn is not None and ffn_fn is not None and x_layer is not None
            sample_attn = attn_fn(x_layer)
            sample_layer = ffn_fn(sample_attn, x_layer)
            if tuple(sample_layer.shape) != (local_tokens, hidden):
                raise RuntimeError(f"layer_output_shape_mismatch:{tuple(sample_layer.shape)}!=({local_tokens},{hidden})")
            for _ in range(SUBGRAPH_WARMUP_ITERS):
                _ = ffn_fn(attn_fn(x_layer), x_layer)
            torch_mod.npu.synchronize()
        except Exception as exc:  # noqa: BLE001
            fwd_err = exc
    all_fwd_ok = _sync_readiness(torch_mod, all_ready and (fwd_err is None), device, dist_group)

    try:
        if not all_ready:
            raise setup_err if setup_err is not None else RuntimeError("peer_rank_setup_failed")
        if not all_fwd_ok:
            raise fwd_err if fwd_err is not None else RuntimeError("peer_rank_forward_failed")
        assert attn_fn is not None and ffn_fn is not None and x_layer is not None and sample_attn is not None
        attn_prof = _profile_callable(
            torch_mod, torch_npu_mod, lambda _s: attn_fn(x_layer),
            rank=rank, dist_group=dist_group, blocker_pair=blocker_pair, l2_flush_pair=l2_flush_pair, warmup_iters=0,
            cpu_store=cpu_store, world_size=world_size, barrier_tag=f"c{case_idx}:attn",
        )
        ffn_prof = _profile_callable(
            torch_mod, torch_npu_mod, lambda _s: ffn_fn(sample_attn, x_layer),
            rank=rank, dist_group=dist_group, blocker_pair=blocker_pair, l2_flush_pair=l2_flush_pair, warmup_iters=0,
            cpu_store=cpu_store, world_size=world_size, barrier_tag=f"c{case_idx}:ffn",
        )
        layer_prof = _profile_callable(
            torch_mod, torch_npu_mod, lambda _s: ffn_fn(attn_fn(x_layer), x_layer),
            rank=rank, dist_group=dist_group, blocker_pair=blocker_pair, l2_flush_pair=l2_flush_pair, warmup_iters=0,
            cpu_store=cpu_store, world_size=world_size, barrier_tag=f"c{case_idx}:layer",
        )
        return {
            **base_record, "runner_mode": RUNNER_MODE, "coverage_tier": TIER_TORCHNPU_COMPOSITE, "status": "ok",
            "attention_meta": attn_meta, "ffn_meta": ffn_meta,
            "attention_capture": attn_prof, "ffn_capture": ffn_prof, "layer_capture": layer_prof,
        }
    except Exception as exc:  # noqa: BLE001
        raw_msg = str(exc).strip() or type(exc).__name__
        full_msg = _sanitize_text(raw_msg)
        msg = (full_msg.splitlines() or [type(exc).__name__])[0][:240]
        msg_low = full_msg.lower()
        if isinstance(exc, RunnerUnsupportedOverlayError):
            return {
                **base_record, "runner_mode": RUNNER_MODE,
                "coverage_tier": TIER_RUNNER_UNSUPPORTED, "status": "runner_unsupported",
                "error_type": type(exc).__name__, "absent_reason": msg,
            }
        is_capability_absent = (
            isinstance(exc, CapabilityAbsentError)
            or (
                type(exc) is RuntimeError
                and not any(ex in msg_low for ex in CAPABILITY_NON_ABSENT_EXCLUSIONS)
                and any(m in msg_low for m in CAPABILITY_ABSENT_MARKERS)
            )
        )
        return {
            **base_record, "runner_mode": RUNNER_MODE,
            "coverage_tier": TIER_UNMERGED_ABSENT if is_capability_absent else TIER_TORCHNPU_COMPOSITE,
            "status": "absent_custom_op" if is_capability_absent else "runtime_error",
            "error_type": type(exc).__name__, "absent_reason": msg,
        }
    finally:
        gc.collect()
        torch_mod.npu.empty_cache()


def _write_bundle_atomic(output_path: str, bundle: dict[str, Any]) -> None:
    """Persist capture bundle atomically via temporary sibling + ``os.replace``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)


def _worker_main(rank: int, world_size: int, spec_path: str, output_path: str, master_port: int) -> None:
    import torch
    import torch_npu

    cpu_store: Any = None
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.npu.set_device(rank)
        cpu_store = torch.distributed.TCPStore(
            "127.0.0.1", master_port, world_size, is_master=(rank == 0),
            timeout=datetime.timedelta(seconds=HCCL_INIT_TIMEOUT_SEC),
        )
        torch.distributed.init_process_group(
            backend="hccl", store=cpu_store, rank=rank, world_size=world_size,
            timeout=datetime.timedelta(seconds=HCCL_INIT_TIMEOUT_SEC),
        )
        dist_group = torch.distributed.group.WORLD
        device = f"npu:{rank}"
    else:
        torch.npu.set_device(0)
        dist_group = None
        device = "npu:0"

    soc_version = str(torch.npu.get_device_name(rank if world_size > 1 else 0))
    cann_ver = _detect_cann_version()
    runner_sha = _sha256_file(Path(__file__).resolve())
    spec_sha = _sha256_file(Path(spec_path).resolve())
    _lock_aic_frequency(torch, device)
    blocker_pair = (
        torch.randn((STREAM_BLOCKER_M, STREAM_BLOCKER_K), dtype=torch.bfloat16, device=device) * (STREAM_BLOCKER_K ** -0.5),
        torch.randn((STREAM_BLOCKER_K, STREAM_BLOCKER_N), dtype=torch.bfloat16, device=device) * (STREAM_BLOCKER_K ** -0.5),
    )
    l2_flush_pair = _alloc_l2_flush_pair(torch, device)
    freq_pre = _sample_loaded_aicore_freq(torch, blocker_pair, rank)
    specs_doc = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    cases = [c for c in specs_doc.get("cases", []) if int(c.get("world_size", 1)) == world_size]

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, case_spec in enumerate(cases):
        if world_size > 1:
            torch.distributed.barrier()
        rec = _execute_single_case(
            torch, torch_npu, case_spec, device, rank if world_size > 1 else 0,
            rank=rank, dist_group=dist_group, blocker_pair=blocker_pair, l2_flush_pair=l2_flush_pair,
            cpu_store=cpu_store, world_size=world_size, case_idx=idx,
        )
        freq_now = float((rec.get("layer_capture") or {}).get("observed_aicore_freq_mhz") or 0.0)
        if freq_now <= 0.0:
            freq_now = _detect_aicore_freq_mhz(rank)
        local_receipt = _build_rank_receipt(rank, rec, freq_mhz=freq_now)
        if world_size > 1 and dist_group is not None:
            gathered: list[dict[str, Any] | None] = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_receipt, group=dist_group)
            rank_receipts = [r for r in gathered if r is not None]
        else:
            rank_receipts = [local_receipt]
        if rank == 0:
            rec["rank_receipts"] = rank_receipts
            results.append(rec)
            print(
                f"[{idx + 1}/{len(cases)}] {rec['case_id']} tier={rec['coverage_tier']} "
                f"status={rec['status']} blk={rec.get('layer_capture', {}).get('stream_blocked', False)} "
                f"ranks={len(rank_receipts)} attn_us={rec.get('attention_capture', {}).get('elapsed_span_us', '-')} "
                f"ffn_us={rec.get('ffn_capture', {}).get('elapsed_span_us', '-')}",
                flush=True,
            )
            _write_bundle_atomic(output_path, {
                "schema_version": SCHEMA_VERSION, "runner_mode": RUNNER_MODE, "shard_complete": False,
                "expected_cases": len(cases), "runner_sha256": runner_sha, "spec_sha256": spec_sha,
                "hardware_soc": soc_version, "cann_version": cann_ver,
                "observed_aicore_freq_mhz_pre": freq_pre, "observed_aicore_freq_mhz": freq_now,
                "torch_version": str(torch.__version__), "torch_npu_version": str(torch_npu.__version__),
                "world_size": world_size, "wall_seconds": round(time.perf_counter() - t0, 3),
                "cases": results,
            })

    freq_post = _detect_aicore_freq_mhz(rank)
    if rank == 0:
        hot_freqs = [
            float((r.get("layer_capture") or {}).get("observed_aicore_freq_mhz") or 0.0)
            for r in results if r.get("status") == "ok"
        ]
        hot_freqs = [f for f in hot_freqs if f > 0.0]
        eff_freq = round(float(statistics.median(hot_freqs)), 2) if hot_freqs else freq_post
        _write_bundle_atomic(output_path, {
            "schema_version": SCHEMA_VERSION, "runner_mode": RUNNER_MODE, "shard_complete": True,
            "expected_cases": len(cases), "runner_sha256": runner_sha, "spec_sha256": spec_sha,
            "hardware_soc": soc_version, "cann_version": cann_ver,
            "observed_aicore_freq_mhz_pre": freq_pre, "observed_aicore_freq_mhz_post": freq_post,
            "observed_aicore_freq_mhz": eff_freq,
            "torch_version": str(torch.__version__), "torch_npu_version": str(torch_npu.__version__),
            "world_size": world_size, "wall_seconds": round(time.perf_counter() - t0, 3),
            "cases": results,
        })

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _bundle_has_runtime_failures(bundle: dict[str, Any]) -> bool:
    """Return True if shard is incomplete, empty, or contains any rank-level runtime error."""
    if not bundle.get("shard_complete"):
        return True
    cases = list(bundle.get("cases") or [])
    if not cases:
        return True
    for case in cases:
        if case.get("status") == "runtime_error":
            return True
        receipts = list(case.get("rank_receipts") or [])
        if any(r.get("status") == "runtime_error" for r in receipts):
            return True
        if case.get("status") == "ok" and any(r.get("status") != "ok" for r in receipts):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="JSON spec file exported by verify_catalog_subgraphs.py")
    parser.add_argument("--output", required=True, help="Output sanitized JSON capture bundle path")
    parser.add_argument("--world-size", type=int, default=1, help="Number of NPU ranks (1 or 4)")
    parser.add_argument("--master-port", type=int, default=29531, help="Local HCCL rendezvous port")
    parser.add_argument("--allow-errors", action="store_true", help="Exit 0 even if runtime_error cases occurred")
    args = parser.parse_args(argv)

    world = max(1, int(args.world_size))
    if world == 1:
        _worker_main(0, 1, args.spec, args.output, args.master_port)
    else:
        import torch.multiprocessing as mp
        mp.spawn(_worker_main, args=(world, args.spec, args.output, args.master_port), nprocs=world, join=True)

    out_path = Path(args.output)
    if not out_path.is_file():
        return 1
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    if not bundle.get("shard_complete") or not bundle.get("cases"):
        return 1
    return 0 if (args.allow_errors or not _bundle_has_runtime_failures(bundle)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
