"""Calibrated L0 synchronization and RTS launch lookup helpers for operator programs.

[Role] Read silicon-calibrated intracore flag (P12), cross-core barrier (P13),
  and RTS/STARS dispatch (P15) latencies in milliseconds from the active
  ``RecipeContext.primitive_table``.
[Invariants] Returns 0.0 when no ``primitive_table`` is bound or when a metric
  is unmeasured; never introduces hardcoded timing constants or SKU strings.
[Timeouts & Locks] Pure O(1) dictionary lookup. No locks or I/O.
[Side-Effect Policy] Read-only inspection of immutable ``PrimitiveTable`` cells.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import log2
from typing import Any

from mindie_sim.schema.common import ModelingError

NS_PER_MS = 1_000_000.0
# Scalar blockTable.GetValue + DataCopy(ND2NZ) paged command setup ratio relative
# to calibrated on-chip P02_load_data.setup_ns (27 cycles): Arch22 scalar GM lookup
# + ND2NZ issue (~210 cycles -> 8x); Arch35 unified scalar cache + ND2NZ (~81 cycles -> 3x).
ARCH22_PAGED_COPY_SETUP_MULT = 8.0
ARCH35_PAGED_COPY_SETUP_MULT = 3.0


def _p50_ms(rc: Any, primitive_id: str, metric: str) -> float:
    table = getattr(rc, "primitive_table", None)
    if table is None or not hasattr(table, "cell"):
        return 0.0
    try:
        band = table.cell(primitive_id).band(metric)
    except ModelingError:
        return 0.0
    if band.p50 is None or band.p50 <= 0.0:
        return 0.0
    return float(band.p50) / NS_PER_MS


def _p50_raw(rc: Any, primitive_id: str, metric: str) -> float | None:
    table = getattr(rc, "primitive_table", None)
    if table is None or not hasattr(table, "cell"):
        return None
    try:
        band = table.cell(primitive_id).band(metric)
    except ModelingError:
        return None
    if band.p50 is None or band.p50 <= 0.0:
        return None
    return float(band.p50)


def _cell_params(rc: Any, primitive_id: str) -> Mapping[str, Any]:
    table = getattr(rc, "primitive_table", None)
    if table is None or not hasattr(table, "cell"):
        return {}
    try:
        cell = table.cell(primitive_id)
    except ModelingError:
        return {}
    if getattr(cell, "status", None) != "calibrated":
        return {}
    return cell.parameters or {}


def p01_strided_cold_bps(rc: Any, row_bytes: int, stride_bytes: int) -> float | None:
    """Lookup calibrated P01 cold strided HBM bandwidth (bytes/s) for (row_bytes, stride_bytes)."""
    surface = _cell_params(rc, "P01_datacopy").get("strided_cold_surface")
    if not surface:
        return None
    pts = sorted(
        (int(entry["stride_bytes"]), float(entry["bps_p50"]))
        for entry in surface
        if int(entry.get("row_bytes", 0)) == row_bytes and float(entry.get("bps_p50", 0.0)) > 0.0
    )
    if not pts:
        return None
    if stride_bytes <= pts[0][0]:
        return pts[0][1]
    if stride_bytes >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        s0, b0 = pts[i]
        s1, b1 = pts[i + 1]
        if s0 <= stride_bytes <= s1:
            if s0 == stride_bytes:
                return b0
            if s1 == stride_bytes:
                return b1
            frac = (log2(stride_bytes) - log2(s0)) / max(1e-9, log2(s1) - log2(s0))
            return b0 + (b1 - b0) * frac
    return pts[-1][1]


def p01_gather_v2_cold_bps(rc: Any, row_bytes: int, *, scatter: bool = True) -> float | None:
    """Lookup calibrated P01 cold gather_v2 bandwidth (bytes/s) for a given row width."""
    surface = _cell_params(rc, "P01_datacopy").get("gather_v2_cold_surface")
    if not isinstance(surface, Mapping):
        return None
    entry = surface.get(f"{int(row_bytes)}B")
    if not isinstance(entry, Mapping):
        return None
    key = "random_scatter_rw_bps" if scatter else "sequential_rw_bps"
    val = entry.get(key)
    return float(val) if (val is not None and float(val) > 0.0) else None


def p01_hbm_scatter_row_open_bytes(rc: Any) -> int:
    """Return calibrated P01 HBM scatter row-open granule in bytes (0 if absent)."""
    val = _cell_params(rc, "P01_datacopy").get("hbm_scatter_row_open_bytes")
    return int(val) if (val is not None and int(val) > 0) else 0


def p02_load_setup_ms(rc: Any) -> float:
    """Return calibrated P02 LoadData setup latency in ms (0.0 if absent)."""
    return _p50_ms(rc, "P02_load_data", "setup_ns")


def p02_l0b_load_factor(rc: Any) -> float:
    """Return calibrated L0A/L0B bandwidth ratio (>= 1.0; ~2.0 on Arch22, ~1.0 on Arch35)."""
    l0a = _p50_raw(rc, "P02_load_data", "l1_to_l0a_bytes_per_s_per_aic")
    l0b = _p50_raw(rc, "P02_load_data", "l1_to_l0b_bytes_per_s_per_aic")
    if l0a is None or l0b is None or l0b <= 0.0:
        return 1.0
    return max(1.0, l0a / l0b)


FP32_BYTES_SYNC = 4.0


def p09_cast_to_alu_op_ratio(rc: Any) -> float:
    """Calibrated P07-ALU / P09-Cast per-element service ratio (<= 1.0; 1.0 if uncalibrated)."""
    alu_ops = (
        _p50_raw(rc, "P07_vec_alu", "vec_alu_ops_per_s_per_aiv")
        or _p50_raw(rc, "P07_vec_alu", "fp32_ops_per_s_per_aiv")
    )
    cast_bps = _p50_raw(rc, "P09_vec_reduce_cast", "cast_bytes_per_s_per_aiv")
    if alu_ops is None or cast_bps is None or cast_bps <= 0.0:
        return 1.0
    cast_elems_per_s = cast_bps / FP32_BYTES_SYNC
    return min(1.0, alu_ops / cast_elems_per_s)


def p10_sort_to_alu_op_ratio(rc: Any) -> float:
    """Calibrated P07-ALU / P10-Sort per-elem-pass service ratio (>= 1.0; 1.0 if uncalibrated)."""
    alu_ops = (
        _p50_raw(rc, "P07_vec_alu", "vec_alu_ops_per_s_per_aiv")
        or _p50_raw(rc, "P07_vec_alu", "fp32_ops_per_s_per_aiv")
    )
    sort_eps = _p50_raw(rc, "P10_sort_topk", "sort_elems_per_s_per_aiv")
    if alu_ops is None or sort_eps is None or sort_eps <= 0.0:
        return 1.0
    return max(1.0, alu_ops / sort_eps)


def p01_paged_copy_setup_ms(rc: Any, *, is_arch35: bool) -> float:
    """Calibrated per-page blockTable.GetValue + ND2NZ command setup latency (ms)."""
    base = p02_load_setup_ms(rc)
    if base <= 0.0:
        return 0.0
    mult = ARCH35_PAGED_COPY_SETUP_MULT if is_arch35 else ARCH22_PAGED_COPY_SETUP_MULT
    return base * mult


def p03_fixpipe_setup_ms(rc: Any) -> float:
    """Return calibrated P03 Fixpipe setup latency in ms (0.0 if absent)."""
    return _p50_ms(rc, "P03_fixpipe", "setup_ns")


def p12_flag_ms(rc: Any, metric: str = "flag_ns") -> float:
    """Return calibrated P12 intracore hardware-flag latency in ms (0.0 if absent)."""
    return _p50_ms(rc, "P12_intracore_flag", metric)


def p12_pipe_v_ms(rc: Any) -> float:
    """Return calibrated P12 PIPE_V barrier latency in ms (0.0 if absent)."""
    return _p50_ms(rc, "P12_intracore_flag", "pipe_v_barrier_ns") or p12_flag_ms(rc)


def p13_syncall_ms(rc: Any) -> float:
    """Return calibrated P13 mode-0 AIV SyncAll barrier latency in ms (0.0 if absent)."""
    return _p50_ms(rc, "P13_crosscore_sync", "mode0_aiv_barrier_ns")


def p13_paired_ms(rc: Any) -> float:
    """Return calibrated P13 mode-2 AIC/AIV paired cross-core sync latency in ms (0.0 if absent)."""
    return _p50_ms(rc, "P13_crosscore_sync", "mode2_paired_ns")


def p13_mode4_ms(rc: Any) -> float:
    """Return calibrated P13 mode-4 Arch35 intra-block sync latency in ms (0.0 if absent)."""
    return _p50_ms(rc, "P13_crosscore_sync", "mode4_intra_block_ns")


def p15_first_issue_ms(rc: Any, *, aiv: bool = True) -> float:
    """Return calibrated dispatch-only P15 first-issue latency in ms (0.0 if absent or kernel-inclusive)."""
    table = getattr(rc, "primitive_table", None)
    if table is None or not hasattr(table, "cell"):
        return 0.0
    try:
        params = table.cell("P15_rts_stars").parameters or {}
    except ModelingError:
        return 0.0
    if params.get("first_issue_includes_min_kernel", False):
        return 0.0
    primary = "aiv_first_issue_ns" if aiv else "aic_first_issue_ns"
    return _p50_ms(rc, "P15_rts_stars", primary)


def p15_issue_interval_ms(rc: Any, *, owner_kind: str = "aiv") -> float:
    """Return calibrated dispatch-only P15 stream issue interval in ms (0.0 if absent or kernel-inclusive)."""
    table = getattr(rc, "primitive_table", None)
    if table is None or not hasattr(table, "cell"):
        return 0.0
    try:
        params = table.cell("P15_rts_stars").parameters or {}
    except ModelingError:
        return 0.0
    if params.get("issue_interval_includes_min_kernel", False):
        return 0.0
    measured_owner = str(params.get("issue_interval_owner") or "all").strip().lower()
    if measured_owner not in {"all", owner_kind}:
        return 0.0
    return _p50_ms(rc, "P15_rts_stars", "issue_interval_ns")
