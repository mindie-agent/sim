"""Layer 1 thermal clock ranges, reference lower bounds and link ceilings.

[Role]
  Convert explicit FP8/BF16 work into reference Cube cycles, express reference
  lower bounds at sustained and short-burst clocks, build the (W-C) sensitivity
  matrix, and compute per-link byte/cycle lower bounds.

[Invariants]
  * A generic call without workload FLOPs/reference cycles or a complete
    (wall, counted) PMU pair returns ``unmodeled`` and never injects an
    A5/K3 FlashMLA case workload.
  * Reference cycles use caller/simulator peaks and are lower bounds, not
    board-certified measurements or predictions of a whole kernel.
  * Sustained clocks and short-burst clocks remain separate ranges; a boost
    observation never becomes a sustained guarantee.
  * ``T(r,f) = [C + (1-r)*(W-C)]/f_MHz`` is reproducible arithmetic; ``r`` is a
    hypothetical removed fraction, not an identified bubble.
  * Link bounds are individual ceilings and are not summed across shared ports.

[Timeouts & Locks]
  Pure computation.  No I/O, locks, sleeps, subprocesses or caches.

[Side-Effect Policy]
  No filesystem, network, device or global-state mutation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .hardware_common import (
    A5_CANN_SIM_LINK_CEILINGS_BPC,
    A5_REFERENCE_PEAKS,
    SHORT_BURST_MHZ_RANGE,
    SUSTAINED_MHZ_RANGE,
    HardwareError,
    _canonical_link_key,
    _finite,
    _float_value,
)

A5_FLASHMLA_C8_GOLDEN_WORKLOAD: dict[str, float] = {
    "fp8_flops": 206158430208.0,
    "bf16_flops": 12884901888.0,
    "target_us": 350.0,
    "wall_cycles": 566344.125,
    "counted_cycles": 476981.3125,
    "vector_cycles": 456026.906,
    "fix_cycles": 437384.344,
}
MISSING_THERMAL_INPUT_REASON = "missing_workload_or_cycle_inputs"


def cube_reference_cycles(
    flops_fp8: float,
    flops_bf16: float,
    *,
    peak_fp8_flops_per_s: float = A5_REFERENCE_PEAKS["fp8_flops_per_s"],
    peak_bf16_flops_per_s: float = A5_REFERENCE_PEAKS["bf16_flops_per_s"],
    reference_mhz: float = A5_REFERENCE_PEAKS["reference_mhz"],
) -> float:
    if not _finite(flops_fp8) or not _finite(flops_bf16) or flops_fp8 < 0 or flops_bf16 < 0:
        raise HardwareError("cube_reference_flops_invalid")
    if not _finite(peak_fp8_flops_per_s) or not _finite(peak_bf16_flops_per_s):
        raise HardwareError("cube_reference_peak_invalid")
    if peak_fp8_flops_per_s <= 0 or peak_bf16_flops_per_s <= 0 or reference_mhz <= 0:
        raise HardwareError("cube_reference_peak_invalid")
    seconds = flops_fp8 / float(peak_fp8_flops_per_s) + flops_bf16 / float(peak_bf16_flops_per_s)
    return seconds * float(reference_mhz) * 1e6


def cycles_to_us(cycles: float, mhz: float) -> float:
    if not _finite(cycles) or not _finite(mhz) or mhz <= 0:
        raise HardwareError("cycle_time_invalid")
    return float(cycles) / float(mhz)


def reference_time_us(cycles: float, mhz: float) -> float:
    return cycles_to_us(cycles, mhz)


def _resolve_clock_ranges(
    sustained_mhz_range: Sequence[float],
    short_burst_mhz_range: Sequence[float],
) -> tuple[list[float], list[float]]:
    sustained = sorted(float(value) for value in sustained_mhz_range)
    burst = sorted(float(value) for value in short_burst_mhz_range)
    if len(sustained) != 2 or len(burst) != 2 or sustained[0] <= 0 or burst[0] <= 0:
        raise HardwareError("thermal_clock_range_invalid")
    if sustained[0] > sustained[1] or burst[0] > burst[1]:
        raise HardwareError("thermal_clock_range_invalid")
    return sustained, burst


def _frequencies(
    sustained: Sequence[float],
    burst: Sequence[float],
    extra_frequencies_mhz: Sequence[float],
) -> list[float]:
    return sorted({
        *sustained,
        *burst,
        *(float(value) for value in extra_frequencies_mhz if _finite(value) and float(value) > 0),
    })


def _target_value(target_us: float | None) -> float | None:
    if target_us is None:
        return None
    target = _float_value(target_us, "thermal_target_invalid")
    if target <= 0:
        raise HardwareError("thermal_target_invalid")
    return target


def unmodeled_thermal_bounds(
    reason: str,
    *,
    sustained_mhz_range: Sequence[float] = SUSTAINED_MHZ_RANGE,
    short_burst_mhz_range: Sequence[float] = SHORT_BURST_MHZ_RANGE,
    extra_frequencies_mhz: Sequence[float] = (1280.0,),
    target_us: float | None = None,
) -> dict[str, Any]:
    sustained, burst = _resolve_clock_ranges(sustained_mhz_range, short_burst_mhz_range)
    return {
        "status": "unmodeled",
        "unmodeled_reason": str(reason),
        "cube_reference_cycles": None,
        "reference_cube_cycles_per_aic": None,
        "reference_mhz_for_cycles": None,
        "frequencies_mhz": _frequencies(sustained, burst, extra_frequencies_mhz),
        "reference_lower_bounds_us": {},
        "sustained_mhz_range": sustained,
        "sustained_reference_us": None,
        "short_burst_mhz_range": burst,
        "short_burst_reference_us": None,
        "target_us": _target_value(target_us),
        "required_mhz_for_target": None,
        "sensitivity": None,
        "clock_table": [],
        "gap_sensitivity": [],
        "wall_us": None,
        "counted_us": None,
        "gap_us": None,
        "ideal_hardware_lower_bound_us": None,
        "assumptions": [
            "no workload FLOPs/reference cycles and no complete (wall_cycles, counted_cycles) pair were supplied",
            "missing calibration inputs are not replaced with an A5/K3 FlashMLA golden workload",
        ],
    }


def sensitivity_matrix(
    wall_cycles: float,
    counted_cycles: float,
    *,
    frequencies_mhz: Sequence[float] = (1250.0, 1280.0, 1300.0),
    r_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    wall = _float_value(wall_cycles, "wall_cycles_invalid")
    counted = _float_value(counted_cycles, "counted_cycles_invalid")
    if counted > wall:
        raise HardwareError("counted_cycles_exceed_wall_cycles")
    frequencies = sorted({float(value) for value in frequencies_mhz if _finite(value) and float(value) > 0})
    fractions = [float(value) for value in r_values]
    if not frequencies or any(not _finite(value) or value < 0 or value > 1 for value in fractions):
        raise HardwareError("sensitivity_matrix_arguments_invalid")
    gap = wall - counted
    table: list[dict[str, Any]] = []
    for fraction in fractions:
        remaining = counted + (1.0 - fraction) * gap
        times = {f"{frequency:g}": remaining / frequency for frequency in frequencies}
        table.append({
            "r": fraction,
            "removed_gap_cycles": fraction * gap,
            "remaining_cycles": remaining,
            "time_us": times,
        })
    return {
        "wall_cycles": wall,
        "counted_cycles": counted,
        "gap_cycles": gap,
        "frequencies_mhz": frequencies,
        "r_values": fractions,
        "table": table,
        "formula": "T(r,f) = [C + (1-r)*(W-C)] / f_MHz",
        "assumptions": [
            "W and C are aggregate PMU cycle counts, not per-instruction service times",
            "r is a hypothetical removable fraction, not an identified bubble",
            "frequency changes are assumed not to change the counted cycles",
        ],
    }


def thermal_bounds(
    *,
    fp8_flops: float | None = None,
    bf16_flops: float | None = None,
    sustained_mhz_range: Sequence[float] = SUSTAINED_MHZ_RANGE,
    short_burst_mhz_range: Sequence[float] = SHORT_BURST_MHZ_RANGE,
    extra_frequencies_mhz: Sequence[float] = (1280.0,),
    target_us: float | None = None,
    wall_cycles: float | None = None,
    counted_cycles: float | None = None,
    vector_cycles: float | None = None,
    fix_cycles: float | None = None,
    cube_reference_cycles: float | None = None,
    fp8_ops_per_cycle: float | None = None,
    bf16_ops_per_cycle: float | None = None,
    link_traffic_bytes_per_aic: Mapping[str, int | float] | None = None,
    link_reference_mhz: float = 1280.0,
    link_ceilings_bpc: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    sustained, burst = _resolve_clock_ranges(sustained_mhz_range, short_burst_mhz_range)
    frequencies = _frequencies(sustained, burst, extra_frequencies_mhz)
    target = _target_value(target_us)
    explicit_cycles = None if cube_reference_cycles is None else _float_value(
        cube_reference_cycles, "cube_reference_cycles_invalid"
    )
    fp8 = None if fp8_flops is None else _float_value(fp8_flops, "cube_reference_flops_invalid")
    bf16 = None if bf16_flops is None else _float_value(bf16_flops, "cube_reference_flops_invalid")
    has_flops = fp8 is not None or bf16 is not None
    wall = None if wall_cycles is None else _float_value(wall_cycles, "wall_cycles_invalid")
    counted = None if counted_cycles is None else _float_value(counted_cycles, "counted_cycles_invalid")
    has_measured = wall is not None and counted is not None

    if explicit_cycles is None and not has_flops and not has_measured:
        result = unmodeled_thermal_bounds(
            MISSING_THERMAL_INPUT_REASON,
            sustained_mhz_range=sustained,
            short_burst_mhz_range=burst,
            extra_frequencies_mhz=extra_frequencies_mhz,
            target_us=target,
        )
        if link_traffic_bytes_per_aic:
            result["link_bounds"] = link_reference_bounds(
                link_traffic_bytes_per_aic,
                ceilings_bpc=link_ceilings_bpc,
                mhz=link_reference_mhz,
            )
        return result

    reference_mhz = float(A5_REFERENCE_PEAKS["reference_mhz"])
    if explicit_cycles is not None:
        cycles: float | None = explicit_cycles
        cycle_source = "explicit cube_reference_cycles"
    elif has_flops:
        fp8_ops = (
            _float_value(fp8_ops_per_cycle, "cube_reference_peak_invalid")
            if fp8_ops_per_cycle is not None
            else float(A5_REFERENCE_PEAKS["fp8_flops_per_s"]) / (reference_mhz * 1e6)
        )
        bf16_ops = (
            _float_value(bf16_ops_per_cycle, "cube_reference_peak_invalid")
            if bf16_ops_per_cycle is not None
            else float(A5_REFERENCE_PEAKS["bf16_flops_per_s"]) / (reference_mhz * 1e6)
        )
        if fp8_ops <= 0 or bf16_ops <= 0:
            raise HardwareError("cube_reference_peak_invalid")
        cycles = (fp8 or 0.0) / fp8_ops + (bf16 or 0.0) / bf16_ops
        cycle_source = (
            "caller-supplied per-cycle Cube peaks"
            if fp8_ops_per_cycle is not None or bf16_ops_per_cycle is not None
            else "generic simulator peaks at the reference clock"
        )
    else:
        cycles = None
        cycle_source = None

    lower_bounds = (
        {f"{frequency:g}": reference_time_us(cycles, frequency) for frequency in frequencies}
        if cycles is not None else {}
    )
    sensitivity = (
        sensitivity_matrix(wall, counted)
        if has_measured else None
    )
    clock_table: list[dict[str, Any]] = []
    for frequency in frequencies:
        row: dict[str, Any] = {
            "frequency_mhz": frequency,
            "mhz": frequency,
            "reference_lower_bound_us": lower_bounds.get(f"{frequency:g}"),
        }
        if has_measured:
            row.update({
                "wall_us": wall / frequency,
                "counted_us": counted / frequency,
                "gap_us": (wall - counted) / frequency,
            })
        else:
            row.update({"wall_us": None, "counted_us": None, "gap_us": None})
        clock_table.append(row)

    required = None
    if target is not None:
        required = {
            "reference_cube_mhz": cycles / target if cycles is not None else None,
            "counted_mac_mhz": counted / target if counted is not None else None,
            "vector_mhz": float(vector_cycles) / target if vector_cycles is not None else None,
            "fix_mhz": float(fix_cycles) / target if fix_cycles is not None else None,
            "wall_mhz": wall / target if wall is not None else None,
        }

    if cycles is None:
        status = "partial_measurement_only"
    elif has_measured:
        status = "ok"
    else:
        status = "partial_reference_only"

    wall_us = wall / reference_mhz if has_measured else None
    counted_us = counted / reference_mhz if has_measured else None
    result: dict[str, Any] = {
        "status": status,
        "unmodeled_reason": None,
        "cube_reference_cycles": cycles,
        "reference_cube_cycles_per_aic": cycles,
        "reference_mhz_for_cycles": reference_mhz if cycles is not None else None,
        "frequencies_mhz": frequencies,
        "reference_lower_bounds_us": lower_bounds,
        "sustained_mhz_range": sustained,
        "sustained_reference_us": (
            [reference_time_us(cycles, sustained[0]), reference_time_us(cycles, sustained[1])]
            if cycles is not None else None
        ),
        "short_burst_mhz_range": burst,
        "short_burst_reference_us": (
            [reference_time_us(cycles, burst[0]), reference_time_us(cycles, burst[1])]
            if cycles is not None else None
        ),
        "target_us": target,
        "required_mhz_for_target": required,
        "sensitivity": sensitivity,
        "clock_table": clock_table,
        "gap_sensitivity": sensitivity["table"] if sensitivity is not None else [],
        "wall_us": wall_us,
        "counted_us": counted_us,
        "gap_us": wall_us - counted_us if wall_us is not None and counted_us is not None else None,
        "ideal_hardware_lower_bound_us": min(lower_bounds.values()) if lower_bounds else None,
        "assumptions": [
            f"cube reference source: {cycle_source}",
            "sustained thermal range is supplied evidence, not inferred from the configured clock",
            "short-burst clocks are never extrapolated to sustained performance",
            "the reference lower bound excludes launch, prologue/drain and resource contention",
            "W and C are aggregate PMU cycle counts; a missing half of the pair is not invented",
        ],
    }
    if link_traffic_bytes_per_aic:
        result["link_bounds"] = link_reference_bounds(
            link_traffic_bytes_per_aic,
            ceilings_bpc=link_ceilings_bpc,
            mhz=link_reference_mhz,
        )
    return result


def link_reference_bounds(
    traffic_bytes: Mapping[str, int | float],
    *,
    ceilings_bpc: Mapping[str, int | float] | None = None,
    mhz: float = 1280.0,
) -> dict[str, Any]:
    ceilings = dict(A5_CANN_SIM_LINK_CEILINGS_BPC) if ceilings_bpc is None else {
        _canonical_link_key(str(key)): float(value) for key, value in ceilings_bpc.items()
    }
    if not _finite(mhz) or mhz <= 0:
        raise HardwareError("link_reference_mhz_invalid")
    links: list[dict[str, Any]] = []
    for key, raw_bytes in sorted(traffic_bytes.items()):
        canonical = _canonical_link_key(str(key))
        bpc = ceilings.get(canonical)
        if bpc is None:
            raise HardwareError(f"link_ceiling_missing:{canonical}")
        byte_count = _float_value(raw_bytes, f"link_traffic_invalid:{canonical}")
        links.append({
            "link": canonical,
            "bytes_per_aic": byte_count,
            "bytes_per_cycle": bpc,
            "minimum_us": byte_count / (bpc * mhz),
        })
    if not links:
        raise HardwareError("link_traffic_empty")
    return {
        "at_mhz": float(mhz),
        "links": links,
        "max_us": max(item["minimum_us"] for item in links),
        "sum_us_not_additive": True,
        "note": "Individual link ceilings; do not sum links sharing ports or contention.",
    }


__all__ = [
    "A5_FLASHMLA_C8_GOLDEN_WORKLOAD",
    "MISSING_THERMAL_INPUT_REASON",
    "cube_reference_cycles",
    "cycles_to_us",
    "reference_time_us",
    "sensitivity_matrix",
    "unmodeled_thermal_bounds",
    "thermal_bounds",
    "link_reference_bounds",
]
