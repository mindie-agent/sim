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

from typing import Any

from mindie_sim.schema.common import ModelingError

NS_PER_MS = 1_000_000.0


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


def p15_first_issue_ms(rc: Any, *, aiv: bool = True) -> float:
    """Return calibrated P15 first-issue/submit latency in ms (0.0 if absent)."""
    primary = "aiv_first_issue_ns" if aiv else "aic_first_issue_ns"
    return _p50_ms(rc, "P15_rts_stars", primary) or _p50_ms(rc, "P15_rts_stars", "host_submit_ns")


def p15_issue_interval_ms(rc: Any) -> float:
    """Return calibrated P15 back-to-back stream issue interval in ms (0.0 if absent)."""
    return _p50_ms(rc, "P15_rts_stars", "issue_interval_ns")
