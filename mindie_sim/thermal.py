"""Layer 1 thermal clock ranges, reference lower bounds and link ceilings.

[Role]
  Convert explicit FP8/BF16 work or explicit reference cycles into reference
  Cube cycles, express reference lower bounds at caller-supplied frequencies,
  build the (W-C) sensitivity matrix, and compute per-link byte/cycle lower
  bounds.

[Invariants]
  * A generic call without workload FLOPs/reference cycles or a complete
    (wall, counted) PMU pair returns ``unmodeled`` and never injects an
    A5/K3 FlashMLA case workload.
  * Reference cycles are frequency-independent aggregate per-die cycle counts.
    Every time value is indexed by an explicitly listed frequency; top-level
    W/C times exist only when the caller supplies ``measurement_mhz``.
  * Reference cycles use caller/profile per-die ops-per-cycle peaks and are
    lower bounds, not board-certified measurements or whole-kernel predictions.
  * Sustained clocks and short-burst clocks remain separate ranges; a boost
    observation never becomes a sustained guarantee.  A sustained band is a
    measurement: it is empty (``sustained_reference_us=None``) unless the caller
    supplies one, and no family constant is substituted for it.
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
    HardwareError,
    _canonical_link_key,
    _finite,
    _float_value,
    parse_mhz_range,
)

# Explicit A5 C8 FlashMLA Golden workload fixture.  The peaks are the A5
# simulator reference peaks expressed as total per-die ops/cycle at 1650 MHz;
# they are carried by the fixture so a generic thermal call never falls back to
# them.  Imported fixture data is not a new A5 board measurement.
A5_FLASHMLA_C8_GOLDEN_WORKLOAD: dict[str, Any] = {
    "fp8_flops": 206158430208.0,
    "bf16_flops": 12884901888.0,
    "fp8_ops_per_cycle": float(A5_REFERENCE_PEAKS["fp8_flops_per_s"])
    / (float(A5_REFERENCE_PEAKS["reference_mhz"]) * 1e6),
    "bf16_ops_per_cycle": float(A5_REFERENCE_PEAKS["bf16_flops_per_s"])
    / (float(A5_REFERENCE_PEAKS["reference_mhz"]) * 1e6),
    "target_us": 350.0,
    "wall_cycles": 566344.125,
    "counted_cycles": 476981.3125,
    "vector_cycles": 456026.906,
    "fix_cycles": 437384.344,
    "workload_origin": "A5_C8_FLASHMLA_GOLDEN_FIXTURE",
}
MISSING_THERMAL_INPUT_REASON = "missing_workload_or_cycle_inputs"
MISSING_CUBE_PEAK_REASON = "missing_cube_ops_per_cycle"


def cube_reference_cycles(
    flops_fp8: float,
    flops_bf16: float,
    *,
    peak_fp8_flops_per_s: float = A5_REFERENCE_PEAKS["fp8_flops_per_s"],
    peak_bf16_flops_per_s: float = A5_REFERENCE_PEAKS["bf16_flops_per_s"],
    reference_mhz: float = A5_REFERENCE_PEAKS["reference_mhz"],
) -> float:
    """Legacy A5 reference helper with explicit A5 simulator peak defaults.

    ``thermal_bounds`` does not call this helper without caller/profile peaks.
    """
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


def _clock_value(value: Any, reason: str) -> float:
    if not _finite(value) or float(value) <= 0:
        raise HardwareError(reason)
    return float(value)


def _resolve_clock_ranges(
    sustained_mhz_range: Sequence[float] | None,
    short_burst_mhz_range: Sequence[float] | None,
) -> tuple[list[float], list[float]]:
    # A sustained band is measured evidence: ``None``/empty stays unmeasured
    # (empty list), and a supplied band is validated without substitution.  The
    # short-burst band keeps its historical two-value contract.
    sustained = parse_mhz_range(sustained_mhz_range, reason="thermal_clock_range_invalid")
    burst = parse_mhz_range(short_burst_mhz_range, reason="thermal_clock_range_invalid")
    if burst is None:
        raise HardwareError("thermal_clock_range_invalid")
    return (list(sustained) if sustained is not None else []), list(burst)


def _frequencies(
    sustained: Sequence[float],
    burst: Sequence[float],
    extra_frequencies_mhz: Sequence[float],
    measurement_mhz: float | None,
) -> list[float]:
    values = {*sustained, *burst}
    try:
        values.update(_clock_value(value, "thermal_frequency_invalid") for value in extra_frequencies_mhz)
    except TypeError as exc:
        raise HardwareError("thermal_frequency_invalid") from exc
    if measurement_mhz is not None:
        values.add(_clock_value(measurement_mhz, "thermal_measurement_mhz_invalid"))
    return sorted(values)


def _target_value(target_us: float | None) -> float | None:
    if target_us is None:
        return None
    target = _float_value(target_us, "thermal_target_invalid")
    if target <= 0:
        raise HardwareError("thermal_target_invalid")
    return target


def _time_basis(measurement_mhz: float | None) -> dict[str, Any]:
    return {
        "reference_cycles": "aggregate per-die Cube cycles; frequency-independent",
        "reference_lower_bounds": "cube_reference_cycles / frequency_mhz for each listed frequency",
        "clock_table": "wall/counted/gap PMU cycles / frequency_mhz in the same row",
        "sensitivity": "uses frequencies_mhz; no separate default clock set",
        "top_level_times": (
            "wall/counted/gap PMU cycles / measurement_mhz"
            if measurement_mhz is not None
            else "null: no caller-supplied measurement_mhz; use clock_table"
        ),
        "reference_mhz_for_cycles": None,
        "frequency_source": (
            "caller-supplied sustained (empty when unmeasured) and short-burst "
            "ranges plus caller-supplied extra/measurement frequencies"
        ),
    }


def _result_shell(
    *,
    sustained: Sequence[float],
    burst: Sequence[float],
    frequencies: Sequence[float],
    target: float | None,
    measurement_mhz: float | None,
    workload_origin: str | None,
) -> dict[str, Any]:
    return {
        "status": "unmodeled",
        "unmodeled_reason": None,
        "workload_origin": None if workload_origin is None else str(workload_origin),
        "cycle_basis": None,
        "unsupported_dtypes": [],
        "cube_reference_cycles": None,
        "reference_cube_cycles_per_die": None,
        "reference_mhz_for_cycles": None,
        "frequencies_mhz": list(frequencies),
        "measurement_mhz": measurement_mhz,
        "reference_lower_bounds_us": {},
        "sustained_mhz_range": list(sustained),
        "sustained_reference_us": None,
        "short_burst_mhz_range": list(burst),
        "short_burst_reference_us": None,
        "target_us": target,
        "required_mhz_for_target": None,
        "sensitivity": None,
        "clock_table": [],
        "gap_sensitivity": [],
        "wall_cycles": None,
        "counted_cycles": None,
        "wall_us": None,
        "counted_us": None,
        "gap_us": None,
        "ideal_hardware_lower_bound_us": None,
        "ideal_hardware_lower_bound_mhz": None,
        "time_basis": _time_basis(measurement_mhz),
        "assumptions": [],
    }


def unmodeled_thermal_bounds(
    reason: str,
    *,
    sustained_mhz_range: Sequence[float] | None = None,
    short_burst_mhz_range: Sequence[float] = SHORT_BURST_MHZ_RANGE,
    extra_frequencies_mhz: Sequence[float] = (),
    target_us: float | None = None,
    measurement_mhz: float | None = None,
    workload_origin: str | None = None,
    unsupported_dtypes: Sequence[str] = (),
) -> dict[str, Any]:
    sustained, burst = _resolve_clock_ranges(sustained_mhz_range, short_burst_mhz_range)
    frequencies = _frequencies(sustained, burst, extra_frequencies_mhz, measurement_mhz)
    result = _result_shell(
        sustained=sustained,
        burst=burst,
        frequencies=frequencies,
        target=_target_value(target_us),
        measurement_mhz=measurement_mhz,
        workload_origin=workload_origin,
    )
    result["unmodeled_reason"] = str(reason)
    result["unsupported_dtypes"] = list(unsupported_dtypes)
    result["assumptions"] = [
        f"thermal inputs are insufficient: {reason}",
        "no implicit A5/K3 FlashMLA golden workload was injected",
        "missing calibration inputs are not replaced with a guessed device peak",
    ]
    return result


def sensitivity_matrix(
    wall_cycles: float,
    counted_cycles: float,
    *,
    frequencies_mhz: Sequence[float] = (1250.0, 1280.0, 1300.0),
    r_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Build the removable-gap table at the explicitly listed frequencies.

    The function's own default frequency set is a historical A5 sensitivity
    set.  ``thermal_bounds`` always forwards its report frequencies explicitly.
    """
    wall = _float_value(wall_cycles, "wall_cycles_invalid")
    counted = _float_value(counted_cycles, "counted_cycles_invalid")
    if counted > wall:
        raise HardwareError("counted_cycles_exceed_wall_cycles")
    try:
        frequencies = sorted({_clock_value(value, "sensitivity_matrix_arguments_invalid") for value in frequencies_mhz})
        fractions = [float(value) for value in r_values]
    except (TypeError, ValueError) as exc:
        raise HardwareError("sensitivity_matrix_arguments_invalid") from exc
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


def _peak_value(value: Any, reason: str) -> float | None:
    if value is None:
        return None
    peak = _float_value(value, reason)
    if peak <= 0:
        raise HardwareError(reason)
    return peak


def _flops_to_cycles(
    fp8_flops: float | None,
    bf16_flops: float | None,
    fp8_ops_per_cycle: float | None,
    bf16_ops_per_cycle: float | None,
) -> tuple[float, list[str]]:
    missing: list[str] = []
    if fp8_flops is not None and fp8_flops > 0 and fp8_ops_per_cycle is None:
        missing.append("fp8")
    if bf16_flops is not None and bf16_flops > 0 and bf16_ops_per_cycle is None:
        missing.append("bf16")
    if missing:
        return 0.0, missing
    cycles = 0.0
    if fp8_flops:
        cycles += fp8_flops / fp8_ops_per_cycle  # type: ignore[operator]
    if bf16_flops:
        cycles += bf16_flops / bf16_ops_per_cycle  # type: ignore[operator]
    return cycles, []


def thermal_bounds(
    *,
    fp8_flops: float | None = None,
    bf16_flops: float | None = None,
    sustained_mhz_range: Sequence[float] | None = None,
    short_burst_mhz_range: Sequence[float] = SHORT_BURST_MHZ_RANGE,
    extra_frequencies_mhz: Sequence[float] = (),
    target_us: float | None = None,
    wall_cycles: float | None = None,
    counted_cycles: float | None = None,
    vector_cycles: float | None = None,
    fix_cycles: float | None = None,
    cube_reference_cycles: float | None = None,
    fp8_ops_per_cycle: float | None = None,
    bf16_ops_per_cycle: float | None = None,
    measurement_mhz: float | None = None,
    workload_origin: str | None = None,
    cycle_source: str | None = None,
    link_traffic_bytes_per_aic: Mapping[str, int | float] | None = None,
    link_reference_mhz: float = 1280.0,
    link_ceilings_bpc: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    sustained, burst = _resolve_clock_ranges(sustained_mhz_range, short_burst_mhz_range)
    measurement = None if measurement_mhz is None else _clock_value(
        measurement_mhz, "thermal_measurement_mhz_invalid"
    )
    frequencies = _frequencies(sustained, burst, extra_frequencies_mhz, measurement)
    target = _target_value(target_us)

    fp8 = None if fp8_flops is None else _float_value(fp8_flops, "cube_reference_flops_invalid")
    bf16 = None if bf16_flops is None else _float_value(bf16_flops, "cube_reference_flops_invalid")
    explicit = None if cube_reference_cycles is None else _float_value(
        cube_reference_cycles, "cube_reference_cycles_invalid"
    )
    wall = None if wall_cycles is None else _float_value(wall_cycles, "wall_cycles_invalid")
    counted = None if counted_cycles is None else _float_value(counted_cycles, "counted_cycles_invalid")
    vector = None if vector_cycles is None else _float_value(vector_cycles, "vector_cycles_invalid")
    fix = None if fix_cycles is None else _float_value(fix_cycles, "fix_cycles_invalid")
    if (wall is None) != (counted is None):
        raise HardwareError("thermal_cycles_pair_incomplete")
    fp8_ops = _peak_value(fp8_ops_per_cycle, "cube_reference_peak_invalid")
    bf16_ops = _peak_value(bf16_ops_per_cycle, "cube_reference_peak_invalid")
    has_flops = fp8 is not None or bf16 is not None
    has_measured = wall is not None and counted is not None

    if explicit is None and not has_flops and not has_measured:
        return unmodeled_thermal_bounds(
            MISSING_THERMAL_INPUT_REASON,
            sustained_mhz_range=sustained,
            short_burst_mhz_range=burst,
            extra_frequencies_mhz=extra_frequencies_mhz,
            target_us=target,
            measurement_mhz=measurement,
            workload_origin=workload_origin,
        )

    cycles: float | None = None
    cycle_basis: str | None = None
    unsupported: list[str] = []
    if explicit is not None:
        cycles = explicit
        cycle_basis = "explicit_cube_reference_cycles"
        source_label = "explicit cube_reference_cycles"
    elif has_flops:
        cycles, unsupported = _flops_to_cycles(fp8, bf16, fp8_ops, bf16_ops)
        if unsupported:
            cycles = None
            cycle_basis = None
            source_label = "none (FLOPs peak unavailable)"
        else:
            cycle_basis = "flops_over_ops_per_cycle"
            source_label = cycle_source or "caller-supplied per-die Cube ops/cycle"
    else:
        source_label = "none (measurement-only)"

    result = _result_shell(
        sustained=sustained,
        burst=burst,
        frequencies=frequencies,
        target=target,
        measurement_mhz=measurement,
        workload_origin=workload_origin,
    )
    result["unsupported_dtypes"] = unsupported
    result["cycle_basis"] = cycle_basis
    result["cube_reference_cycles"] = cycles
    result["reference_cube_cycles_per_die"] = cycles

    lower_bounds: dict[str, float] = {}
    if cycles is not None:
        lower_bounds = {
            f"{frequency:g}": reference_time_us(cycles, frequency)
            for frequency in frequencies
        }
        result["reference_lower_bounds_us"] = lower_bounds
        if sustained:
            result["sustained_reference_us"] = [
                reference_time_us(cycles, frequency) for frequency in sustained
            ]
        result["short_burst_reference_us"] = [
            reference_time_us(cycles, burst[0]),
            reference_time_us(cycles, burst[1]),
        ]
        result["ideal_hardware_lower_bound_us"] = min(lower_bounds.values())
        result["ideal_hardware_lower_bound_mhz"] = max(frequencies)

    sensitivity = sensitivity_matrix(wall, counted, frequencies_mhz=frequencies) if has_measured else None
    result["sensitivity"] = sensitivity
    result["gap_sensitivity"] = sensitivity["table"] if sensitivity is not None else []
    result["wall_cycles"] = wall
    result["counted_cycles"] = counted

    clock_table: list[dict[str, Any]] = []
    for frequency in frequencies:
        row: dict[str, Any] = {
            "frequency_mhz": frequency,
            "reference_lower_bound_us": lower_bounds.get(f"{frequency:g}"),
            "wall_us": wall / frequency if has_measured else None,
            "counted_us": counted / frequency if has_measured else None,
            "gap_us": (wall - counted) / frequency if has_measured else None,
        }
        clock_table.append(row)
    result["clock_table"] = clock_table

    if measurement is not None and has_measured:
        result["wall_us"] = wall / measurement
        result["counted_us"] = counted / measurement
        result["gap_us"] = (wall - counted) / measurement

    required = None
    if target is not None:
        required = {
            "reference_cube_mhz": cycles / target if cycles is not None else None,
            "counted_mac_mhz": counted / target if counted is not None else None,
            "vector_mhz": vector / target if vector is not None else None,
            "fix_mhz": fix / target if fix is not None else None,
            "wall_mhz": wall / target if wall is not None else None,
        }
    result["required_mhz_for_target"] = required

    if cycles is not None and has_measured:
        result["status"] = "ok"
    elif cycles is not None:
        result["status"] = "partial_reference_only"
    elif has_measured:
        result["status"] = "partial_measurement_only"
        result["unmodeled_reason"] = MISSING_CUBE_PEAK_REASON if unsupported else None
    else:
        result["status"] = "unmodeled"
        result["unmodeled_reason"] = MISSING_CUBE_PEAK_REASON if unsupported else MISSING_THERMAL_INPUT_REASON

    result["assumptions"] = [
        f"cube reference source: {source_label}",
        "sustained thermal range is supplied evidence, not inferred from the configured clock",
        (
            f"sustained MHz band supplied by the caller: {sustained[0]:g}-{sustained[1]:g}"
            if sustained
            else "no sustained MHz band supplied; sustained_reference_us is null"
        ),
        "short-burst clocks are never extrapolated to sustained performance",
        "the reference lower bound excludes launch, prologue/drain and resource contention",
        "W and C are aggregate PMU cycle counts; a missing half of the pair is not invented",
        (
            f"top-level times use caller-supplied measurement_mhz={measurement:g}"
            if measurement is not None
            else "top-level wall/counted times are null; per-frequency values are in clock_table"
        ),
        "no implicit A5/K3 FlashMLA golden workload was injected",
    ]
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
    "MISSING_CUBE_PEAK_REASON",
    "cube_reference_cycles",
    "cycles_to_us",
    "reference_time_us",
    "sensitivity_matrix",
    "unmodeled_thermal_bounds",
    "thermal_bounds",
    "link_reference_bounds",
]
