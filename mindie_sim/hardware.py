"""Layer 1 hardware, on-chip buffer, and thermal-bound analyzer.

[Role]
  Parse CANN platform INI files, cannsim link ceilings, and npu-smi snapshots;
  keep platform metadata strictly separate from verified runtime capacity; audit
  L1/L0A/L0B/L0C/UB layouts, alignment, ping-pong slots and region overlap; and
  compute reference lower bounds from thermal clock ranges and PMU cycles.

[Invariants]
  * ``platform_config_hbm_mib`` and ``simulator_ub_kib`` are metadata.  They are
    never copied into ``verified_runtime_hbm_mib`` or ``allocatable_ub_kib``.
  * A runtime capacity is reported only from npu-smi evidence; missing evidence
    yields ``verified_runtime_hbm_mib=None`` and ``runtime_unverified``.
  * Buffer sizes/offsets are rounded only by the requested 32B/256B alignment;
    overlap, overflow and misalignment are diagnostics, never silently fixed.
  * Thermal reference times use caller/profile per-die peaks and caller-supplied
    frequencies; they are lower bounds, not measured sustained performance.
    Short-burst clocks never replace sustained clocks.  An unsupported dtype is
    ``unmodeled`` rather than borrowing A5 peaks.
  * The (W-C) sensitivity table is reproducible arithmetic, not a promise that
    the gap is removable.

[Timeouts & Locks]
  Pure parsing/computation.  No locks, sleeps, subprocesses, network, or caches.

[Side-Effect Policy]
  Reads only explicit paths supplied by the caller.  Never writes files, changes
  clocks, changes device state, or invokes npu-smi itself.
"""
from __future__ import annotations

import configparser
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .buffers import BufferAllocation, check_mx_scale_layout, validate_buffer_allocations
from .comm import (
    TRANSFER_CHANNELS,
    NpuTopologySnapshot,
    PcieLinkSnapshot,
    TransferChannelSpec,
    default_transfer_channels,
    evaluate_transfer_plan,
    parse_npu_smi_topo,
    parse_pcie_link_info,
    solve_collective_bound,
    solve_transfer_bound,
)
from .hardware_common import (
    A5_CANN_SIM_LINK_CEILINGS_BPC,
    A5_REFERENCE_PEAKS,
    DEFAULT_ALIGNMENT_BYTES,
    PING_PONG_ALIGNMENT_BYTES,
    SHORT_BURST_MHZ_RANGE,
    # Deprecated alias re-exported for callers that still import it here; this
    # module no longer defaults any clock range to it.
    SUSTAINED_MHZ_RANGE,
    HardwareError,
    _canonical_link_key,
    _float_value,
    _int_value,
    _kib,
    _mib,
    align_up,
    default_link_ceilings,
    parse_mhz_range,
)
from .hardware_probe import (
    NpuSmiChip,
    NpuSmiDevice,
    NpuSmiSnapshot,
    load_npu_smi_snapshot,
    parse_npu_smi_board,
    parse_npu_smi_common,
    parse_npu_smi_info,
    parse_npu_smi_snapshot,
)
from .sku_profile import (
    AMBIGUOUS_SKU_REASON,
    BUILTIN_SKU_CATALOG,
    ClockAndTopologySpec,
    CubeSpec,
    DeviceCapabilityProfile,
    MemoryHierarchySpec,
    VectorSpec,
    infer_arch_family,
    is_ambiguous_family_token,
    load_sku_profile,
    resolve_device_capability_profile,
)
from .thermal import (
    A5_FLASHMLA_C8_GOLDEN_WORKLOAD,
    cube_reference_cycles,
    cycles_to_us,
    link_reference_bounds,
    reference_time_us,
    sensitivity_matrix,
    thermal_bounds,
    unmodeled_thermal_bounds,
)


@dataclass(frozen=True)
class PlatformConfig:
    soc_version: str
    short_soc_version: str
    npu_arch: str | None
    ai_core_cnt: int
    cube_core_cnt: int
    vector_core_cnt: int
    ai_cpu_cnt: int | None
    memory_size_bytes: int
    l2_size_bytes: int
    cube_freq_mhz: float
    l0_a_bytes: int
    l0_b_bytes: int
    l0_c_bytes: int
    l1_bytes: int
    ub_bytes: int
    link_rates_bpc: Mapping[str, float] = field(default_factory=dict)
    source: str = "<text>"

    @property
    def platform_config_hbm_mib(self) -> float:
        return _mib(self.memory_size_bytes)

    @property
    def platform_config_l2_mib(self) -> float:
        return _mib(self.l2_size_bytes)

    @property
    def l0_a_kib(self) -> float:
        return _kib(self.l0_a_bytes)

    @property
    def l0_b_kib(self) -> float:
        return _kib(self.l0_b_bytes)

    @property
    def l0_c_kib(self) -> float:
        return _kib(self.l0_c_bytes)

    @property
    def l1_kib(self) -> float:
        return _kib(self.l1_bytes)

    @property
    def platform_allocatable_ub_kib(self) -> float:
        return _kib(self.ub_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "soc_version": self.soc_version,
            "short_soc_version": self.short_soc_version,
            "npu_arch": self.npu_arch,
            "ai_core_cnt": self.ai_core_cnt,
            "cube_core_cnt": self.cube_core_cnt,
            "vector_core_cnt": self.vector_core_cnt,
            "ai_cpu_cnt": self.ai_cpu_cnt,
            "platform_config_hbm_mib": self.platform_config_hbm_mib,
            "platform_config_l2_mib": self.platform_config_l2_mib,
            "cube_freq_mhz": self.cube_freq_mhz,
            "l1_kib": self.l1_kib,
            "l0_a_kib": self.l0_a_kib,
            "l0_b_kib": self.l0_b_kib,
            "l0_c_kib": self.l0_c_kib,
            "platform_allocatable_ub_kib": self.platform_allocatable_ub_kib,
            "link_rates_bpc": dict(sorted(self.link_rates_bpc.items())),
            "source": self.source,
        }


def parse_platform_ini(path: str | Path) -> PlatformConfig:
    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise HardwareError(f"platform_ini_unreadable:{type(exc).__name__}") from exc
    return parse_platform_ini_text(text, source=str(source))


def parse_platform_ini_text(text: str, *, source: str = "<text>") -> PlatformConfig:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(text.lstrip("\ufeff"))
    except configparser.Error as exc:
        raise HardwareError(f"platform_ini_invalid:{type(exc).__name__}") from exc

    def get(section: str, key: str, default: Any = None) -> Any:
        if not parser.has_section(section) or not parser.has_option(section, key):
            return default
        return parser.get(section, key).strip()

    def required_int(section: str, key: str) -> int:
        raw = get(section, key)
        if raw is None:
            raise HardwareError(f"platform_ini_missing:{section}.{key}")
        try:
            return _int_value(int(raw, 0), f"platform_ini_invalid:{section}.{key}")
        except ValueError as exc:
            raise HardwareError(f"platform_ini_invalid:{section}.{key}") from exc

    def required_float(section: str, key: str) -> float:
        raw = get(section, key)
        if raw is None:
            raise HardwareError(f"platform_ini_missing:{section}.{key}")
        try:
            return _float_value(float(raw), f"platform_ini_invalid:{section}.{key}")
        except ValueError as exc:
            raise HardwareError(f"platform_ini_invalid:{section}.{key}") from exc

    version = "version"
    soc = get(version, "soc_version", "")
    short = get(version, "short_soc_version", soc)
    if not soc:
        raise HardwareError("platform_ini_missing:version.soc_version")
    ai_cpu_raw = get("SoCInfo", "ai_cpu_cnt")
    ai_cpu = int(ai_cpu_raw, 0) if ai_cpu_raw is not None else None
    rates: dict[str, float] = {}
    for key in (
        "ddr_rate", "l2_rate", "l1_to_l0_a_rate", "l1_to_l0_b_rate",
        "l1_to_ub_rate", "l0_c_to_ub_rate", "ub_to_l2_rate", "ub_to_ddr_rate",
        "ub_to_l1_rate",
    ):
        raw = get("AICoreMemoryRates", key)
        if raw is not None:
            try:
                rates[key] = float(raw)
            except ValueError as exc:
                raise HardwareError(f"platform_ini_invalid:AICoreMemoryRates.{key}") from exc
    return PlatformConfig(
        soc_version=soc,
        short_soc_version=short,
        npu_arch=get(version, "NpuArch"),
        ai_core_cnt=required_int("SoCInfo", "ai_core_cnt"),
        cube_core_cnt=required_int("SoCInfo", "cube_core_cnt"),
        vector_core_cnt=required_int("SoCInfo", "vector_core_cnt"),
        ai_cpu_cnt=ai_cpu,
        memory_size_bytes=required_int("SoCInfo", "memory_size"),
        l2_size_bytes=required_int("SoCInfo", "l2_size"),
        cube_freq_mhz=required_float("AICoreSpec", "cube_freq"),
        l0_a_bytes=required_int("AICoreSpec", "l0_a_size"),
        l0_b_bytes=required_int("AICoreSpec", "l0_b_size"),
        l0_c_bytes=required_int("AICoreSpec", "l0_c_size"),
        l1_bytes=required_int("AICoreSpec", "l1_size"),
        ub_bytes=required_int("AICoreSpec", "ub_size"),
        link_rates_bpc=rates,
        source=source,
    )


def cross_check_capacities(
    config: PlatformConfig,
    snapshot: NpuSmiSnapshot | None = None,
    *,
    verified_runtime_hbm_mib: float | None = None,
    simulator_ub_kib: float | None = None,
    sustained_mhz_range: Sequence[float] | None = None,
) -> dict[str, Any]:
    if snapshot is not None and snapshot.devices:
        runtime_hbm = snapshot.min_hbm_mib
        runtime_max = snapshot.max_hbm_mib
    else:
        runtime_hbm = float(verified_runtime_hbm_mib) if verified_runtime_hbm_mib is not None else None
        runtime_max = runtime_hbm
    flags: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    hbm_discrepancy = None
    if runtime_hbm is not None:
        hbm_discrepancy = config.platform_config_hbm_mib - runtime_hbm
        if abs(hbm_discrepancy) > 0.5:
            flags.append("platform_config_hbm_mib_differs_from_runtime")
            diagnostics.append({
                "code": "platform_config_hbm_mib_differs_from_runtime",
                "platform_config_hbm_mib": config.platform_config_hbm_mib,
                "verified_runtime_hbm_mib": runtime_hbm,
                "difference_mib": hbm_discrepancy,
                "authority": "verified_runtime_capacity",
            })
    else:
        flags.append("runtime_hbm_unverified")
        diagnostics.append({"code": "runtime_hbm_unverified", "detail": "no npu-smi runtime capacity evidence"})
    ub_discrepancy = None
    if simulator_ub_kib is not None:
        ub_discrepancy = float(simulator_ub_kib) - config.platform_allocatable_ub_kib
        if abs(ub_discrepancy) > 0.5:
            flags.append("simulator_ub_kib_differs_from_allocatable")
            diagnostics.append({
                "code": "simulator_ub_kib_differs_from_allocatable",
                "simulator_ub_kib": float(simulator_ub_kib),
                "platform_allocatable_ub_kib": config.platform_allocatable_ub_kib,
                "difference_kib": ub_discrepancy,
            })
    if snapshot is not None and snapshot.chips:
        runtime_counts = sorted({chip.aicore_count for chip in snapshot.chips if chip.aicore_count})
        if runtime_counts and any(value != config.ai_core_cnt for value in runtime_counts):
            flags.append("runtime_aicore_count_differs_from_platform_config")
            diagnostics.append({
                "code": "runtime_aicore_count_differs_from_platform_config",
                "platform_config_ai_core_cnt": config.ai_core_cnt,
                "runtime_aicore_count": runtime_counts,
            })
        runtime_clocks = sorted({chip.aicore_freq_mhz for chip in snapshot.chips if chip.aicore_freq_mhz})
        if runtime_clocks and any(abs(value - config.cube_freq_mhz) > 0.5 for value in runtime_clocks):
            flags.append("runtime_configured_clock_differs_from_platform_config")
            diagnostics.append({
                "code": "runtime_configured_clock_differs_from_platform_config",
                "platform_config_cube_freq_mhz": config.cube_freq_mhz,
                "runtime_aicore_freq_mhz": runtime_clocks,
                "runtime_aicore_current_freq_mhz": sorted(
                    {chip.aicore_current_freq_mhz for chip in snapshot.chips if chip.aicore_current_freq_mhz}
                ),
            })
    # An empty/None band means unmeasured (no "configured clock is not
    # sustained" claim can be made without a measured band).
    sustained_band = parse_mhz_range(sustained_mhz_range, reason="hardware_sustained_mhz_range_invalid")
    if sustained_band is not None:
        low, high = sustained_band
        if high < config.cube_freq_mhz:
            flags.append("platform_clock_is_not_sustained_clock")
            diagnostics.append({
                "code": "platform_clock_is_not_sustained_clock",
                "platform_config_cube_freq_mhz": config.cube_freq_mhz,
                "sustained_mhz_range": [low, high],
            })
    return {
        "soc_version": config.soc_version,
        "short_soc_version": config.short_soc_version,
        "platform_config_hbm_mib": config.platform_config_hbm_mib,
        "verified_runtime_hbm_mib": runtime_hbm,
        "verified_runtime_hbm_mib_max": runtime_max,
        "hbm_discrepancy_mib": hbm_discrepancy,
        "platform_allocatable_ub_kib": config.platform_allocatable_ub_kib,
        "simulator_ub_kib": float(simulator_ub_kib) if simulator_ub_kib is not None else None,
        "allocatable_ub_kib": config.platform_allocatable_ub_kib,
        "ub_discrepancy_kib": ub_discrepancy,
        "runtime_verified": runtime_hbm is not None,
        "capacity_authority": "verified_runtime" if runtime_hbm is not None else "platform_config_metadata",
        "flags": flags,
        "diagnostics": diagnostics,
        "note": "Platform INI values are metadata. Runtime HBM must come from npu-smi.",
    }


def parse_cannsim_link_ceilings(path_or_text: str | Path) -> dict[str, Any]:
    candidate = Path(path_or_text) if isinstance(path_or_text, (str, Path)) else None
    text: str
    source: str
    if candidate is not None and candidate.exists() and candidate.is_file():
        text = candidate.read_text(encoding="utf-8-sig", errors="replace")
        source = str(candidate)
    else:
        text = str(path_or_text)
        source = "<text>"
    ceilings: dict[str, float] = {}
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise HardwareError("cannsim_link_json_invalid") from exc
        items = data.items() if isinstance(data, Mapping) else ()
        for key, value in items:
            ceilings[_canonical_link_key(str(key))] = _float_value(value, "cannsim_link_invalid")
    else:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            try:
                numeric = float(value)
            except ValueError:
                continue
            ceilings[_canonical_link_key(key)] = _float_value(numeric, "cannsim_link_invalid")
    if not ceilings:
        raise HardwareError("cannsim_link_constants_empty")
    return {
        "ceilings_bpc": dict(sorted(ceilings.items())),
        "source": source,
        "verified_bandwidth": False,
        "note": "Simulator link ceilings are B/cycle references, not measured bandwidth.",
    }


def _thermal_input(
    direct: Any,
    workload: Mapping[str, Any] | None,
    *keys: str,
) -> Any:
    if direct is not None:
        return direct
    if workload is not None:
        for key in keys:
            value = workload.get(key)
            if value is not None:
                return value
    return None


def _profile_cube_ops_per_cycle(
    profile: DeviceCapabilityProfile,
) -> tuple[float | None, float | None]:
    accessor = getattr(profile.cube_spec, "ops_per_cycle_per_die", None)
    if not callable(accessor):
        return None, None
    return accessor("fp8"), accessor("bf16")


def _thermal_for_profile(
    profile: DeviceCapabilityProfile,
    *,
    sustained_mhz_range: Sequence[float] | None,
    short_burst_mhz_range: Sequence[float],
    extra_frequencies_mhz: Sequence[float],
    workload: Mapping[str, Any] | None,
    fp8_flops: float | None,
    bf16_flops: float | None,
    target_us: float | None,
    wall_cycles: float | None,
    counted_cycles: float | None,
    vector_cycles: float | None,
    fix_cycles: float | None,
    cube_reference_cycles: float | None,
    fp8_ops_per_cycle: float | None,
    bf16_ops_per_cycle: float | None,
    measurement_mhz: float | None,
    workload_origin: str | None,
) -> dict[str, Any]:
    fp8 = _thermal_input(fp8_flops, workload, "fp8_flops", "flops_fp8")
    bf16 = _thermal_input(bf16_flops, workload, "bf16_flops", "flops_bf16")
    target = _thermal_input(target_us, workload, "target_us")
    wall = _thermal_input(wall_cycles, workload, "wall_cycles")
    counted = _thermal_input(counted_cycles, workload, "counted_cycles")
    vector = _thermal_input(vector_cycles, workload, "vector_cycles")
    fix = _thermal_input(fix_cycles, workload, "fix_cycles")
    explicit = _thermal_input(cube_reference_cycles, workload, "cube_reference_cycles")
    measurement = _thermal_input(measurement_mhz, workload, "measurement_mhz")
    origin = _thermal_input(workload_origin, workload, "workload_origin")
    fp8_override = _thermal_input(fp8_ops_per_cycle, workload, "fp8_ops_per_cycle")
    bf16_override = _thermal_input(bf16_ops_per_cycle, workload, "bf16_ops_per_cycle")
    if (wall is None) != (counted is None):
        raise HardwareError("thermal_cycles_pair_incomplete")

    fp8_value = None if fp8 is None else _float_value(fp8, "cube_reference_flops_invalid")
    bf16_value = None if bf16 is None else _float_value(bf16, "cube_reference_flops_invalid")
    profile_fp8, profile_bf16 = _profile_cube_ops_per_cycle(profile)
    fp8_ops = fp8_override if fp8_override is not None else profile_fp8
    bf16_ops = bf16_override if bf16_override is not None else profile_bf16
    peak_sources = {
        "fp8": (
            "caller_explicit" if fp8_override is not None
            else "sku_profile.cube_spec.ops_per_cycle_per_die" if profile_fp8 is not None
            else None
        ),
        "bf16": (
            "caller_explicit" if bf16_override is not None
            else "sku_profile.cube_spec.ops_per_cycle_per_die" if profile_bf16 is not None
            else None
        ),
    }
    unsupported: list[str] = []
    if fp8_value is not None and fp8_value > 0 and fp8_ops is None:
        unsupported.append("fp8")
    if bf16_value is not None and bf16_value > 0 and bf16_ops is None:
        unsupported.append("bf16")

    if explicit is not None:
        thermal = thermal_bounds(
            fp8_flops=fp8_value,
            bf16_flops=bf16_value,
            sustained_mhz_range=sustained_mhz_range,
            short_burst_mhz_range=short_burst_mhz_range,
            extra_frequencies_mhz=extra_frequencies_mhz,
            target_us=target,
            wall_cycles=wall,
            counted_cycles=counted,
            vector_cycles=vector,
            fix_cycles=fix,
            cube_reference_cycles=explicit,
            fp8_ops_per_cycle=fp8_ops,
            bf16_ops_per_cycle=bf16_ops,
            measurement_mhz=measurement,
            workload_origin=origin,
        )
        thermal["unsupported_dtypes"] = unsupported
        thermal["peak_sources"] = peak_sources
        return thermal

    if unsupported:
        if wall is not None and counted is not None:
            thermal = thermal_bounds(
                sustained_mhz_range=sustained_mhz_range,
                short_burst_mhz_range=short_burst_mhz_range,
                extra_frequencies_mhz=extra_frequencies_mhz,
                target_us=target,
                wall_cycles=wall,
                counted_cycles=counted,
                vector_cycles=vector,
                fix_cycles=fix,
                measurement_mhz=measurement,
                workload_origin=origin,
            )
            thermal["unsupported_dtypes"] = unsupported
            thermal["peak_sources"] = peak_sources
            return thermal
        thermal = unmodeled_thermal_bounds(
            "unsupported_dtype_or_missing_sku_cube_throughput",
            sustained_mhz_range=sustained_mhz_range,
            short_burst_mhz_range=short_burst_mhz_range,
            extra_frequencies_mhz=extra_frequencies_mhz,
            target_us=target,
            measurement_mhz=measurement,
            workload_origin=origin,
            unsupported_dtypes=unsupported,
        )
        thermal["peak_sources"] = peak_sources
        return thermal

    uses_profile = (
        (fp8_value is not None and fp8_value > 0 and fp8_override is None and profile_fp8 is not None)
        or (bf16_value is not None and bf16_value > 0 and bf16_override is None and profile_bf16 is not None)
    )
    uses_caller = (
        (fp8_value is not None and fp8_value > 0 and fp8_override is not None)
        or (bf16_value is not None and bf16_value > 0 and bf16_override is not None)
    )
    if uses_profile and uses_caller:
        cycle_source = "mixed caller/profile per-die ops/cycle"
    elif uses_profile:
        cycle_source = "sku profile cube_spec.ops_per_cycle_per_die"
    else:
        cycle_source = "caller-supplied per-die ops/cycle"
    thermal = thermal_bounds(
        fp8_flops=fp8_value,
        bf16_flops=bf16_value,
        sustained_mhz_range=sustained_mhz_range,
        short_burst_mhz_range=short_burst_mhz_range,
        extra_frequencies_mhz=extra_frequencies_mhz,
        target_us=target,
        wall_cycles=wall,
        counted_cycles=counted,
        vector_cycles=vector,
        fix_cycles=fix,
        cube_reference_cycles=explicit,
        fp8_ops_per_cycle=fp8_ops,
        bf16_ops_per_cycle=bf16_ops,
        measurement_mhz=measurement,
        workload_origin=origin,
        cycle_source=cycle_source,
    )
    thermal["peak_sources"] = peak_sources
    return thermal


def _attach_link_bounds(
    thermal: dict[str, Any],
    workload: Mapping[str, Any] | None,
    cannsim_links: Mapping[str, Any] | None,
) -> dict[str, Any]:
    traffic = workload.get("link_traffic_bytes_per_aic") if workload else None
    if not traffic:
        return thermal
    links_source = cannsim_links or default_link_ceilings()
    ceilings = links_source.get("ceilings_bpc", links_source) if isinstance(links_source, Mapping) else {}
    thermal["link_bounds"] = link_reference_bounds(
        traffic,
        ceilings_bpc=ceilings,
        mhz=float(workload.get("link_reference_mhz", 1280.0)),
    )
    return thermal


def hardware_report(
    config: PlatformConfig | None = None,
    *,
    ini_text: str | None = None,
    soc: str | None = None,
    sku_profile: str | Path | Mapping[str, Any] | None = None,
    snapshot: NpuSmiSnapshot | None = None,
    topo_snapshot: NpuTopologySnapshot | None = None,
    pcie_snapshot: PcieLinkSnapshot | None = None,
    verified_runtime_hbm_mib: float | None = None,
    simulator_ub_kib: float | None = 256.0,
    allocatable_ub_kib: float | None = None,
    # A sustained band is caller-supplied evidence; ``None`` leaves the profile
    # and the thermal report explicitly unmeasured (no family constant).
    sustained_mhz_range: Sequence[float] | None = None,
    short_burst_mhz_range: Sequence[float] = SHORT_BURST_MHZ_RANGE,
    extra_frequencies_mhz: Sequence[float] = (),
    buffer_plan: Mapping[str, Any] | None = None,
    workload: Mapping[str, Any] | None = None,
    transfer_plan: Mapping[str, Any] | None = None,
    cannsim_links: Mapping[str, Any] | None = None,
    fp8_flops: float | None = None,
    bf16_flops: float | None = None,
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
) -> dict[str, Any]:
    if config is not None and ini_text is not None:
        raise HardwareError("hardware_config_and_ini_text_conflict")
    if config is None and ini_text is not None:
        config = parse_platform_ini_text(ini_text)
    if workload is not None and not isinstance(workload, Mapping):
        raise HardwareError("hardware_workload_invalid")
    profile = resolve_device_capability_profile(
        config=config,
        soc=soc,
        sku_profile=sku_profile,
        snapshot=snapshot,
        topo_snapshot=topo_snapshot,
        pcie_snapshot=pcie_snapshot,
        verified_runtime_hbm_mib=verified_runtime_hbm_mib,
        simulator_ub_kib=simulator_ub_kib,
        allocatable_ub_kib=allocatable_ub_kib,
        sustained_mhz_range=sustained_mhz_range,
        cannsim_links=cannsim_links,
    )
    if profile.status == "ambiguous_sku":
        payload: dict[str, Any] = {
            "status": "ambiguous_sku",
            "unmodeled_reason": profile.unmodeled_reason,
            "device_capability_profile": profile.to_dict(),
            "effective_clock_mhz": profile.effective_clock_mhz(),
            "effective_clock_source": profile.clock_and_topology_spec.clock_source(),
            "soc": None,
            "capacity_cross_check": None,
            "cannsim_links": dict(cannsim_links) if cannsim_links else default_link_ceilings(),
        }
        if snapshot is not None:
            payload["npu_smi"] = snapshot.to_dict()
        if buffer_plan:
            capacities = buffer_plan.get("capacities_kib") or {
                "L1": profile.cube_spec.l1_kib,
                "L0A": profile.cube_spec.l0a_kib,
                "L0B": profile.cube_spec.l0b_kib,
                "L0C": profile.cube_spec.l0c_kib,
                "UB": profile.vector_spec.allocatable_ub_kib,
            }
            payload["buffers"] = validate_buffer_allocations(
                buffer_plan.get("allocations", ()),
                capacities,
                alignment_bytes=int(buffer_plan.get("alignment_bytes", DEFAULT_ALIGNMENT_BYTES)),
                ping_pong=buffer_plan.get("ping_pong"),
            )
        thermal = _thermal_for_profile(
            profile,
            sustained_mhz_range=sustained_mhz_range,
            short_burst_mhz_range=short_burst_mhz_range,
            extra_frequencies_mhz=extra_frequencies_mhz,
            workload=workload,
            fp8_flops=fp8_flops,
            bf16_flops=bf16_flops,
            target_us=target_us,
            wall_cycles=wall_cycles,
            counted_cycles=counted_cycles,
            vector_cycles=vector_cycles,
            fix_cycles=fix_cycles,
            cube_reference_cycles=cube_reference_cycles,
            fp8_ops_per_cycle=fp8_ops_per_cycle,
            bf16_ops_per_cycle=bf16_ops_per_cycle,
            measurement_mhz=measurement_mhz,
            workload_origin=workload_origin,
        )
        if thermal["status"] == "unmodeled":
            thermal["unmodeled_reason"] = profile.unmodeled_reason or AMBIGUOUS_SKU_REASON
            thermal["assumptions"] = [
                f"thermal inputs are insufficient: {thermal['unmodeled_reason']}",
                "explicit reference cycles or a complete W/C pair remain independently usable",
                "no implicit A5/K3 FlashMLA golden workload was injected",
            ]
        payload["thermal"] = _attach_link_bounds(thermal, workload, cannsim_links)
        return payload

    if config is None:
        config = PlatformConfig(
            soc_version=profile.clock_and_topology_spec.soc_version or (soc or "custom_sku"),
            short_soc_version=profile.clock_and_topology_spec.short_soc_version or (soc or "custom_sku"),
            npu_arch=profile.clock_and_topology_spec.npu_arch,
            ai_core_cnt=profile.cube_spec.aic_count_per_die or 0,
            cube_core_cnt=profile.cube_spec.aic_count_per_die or 0,
            vector_core_cnt=profile.vector_spec.aiv_count_per_die or 0,
            ai_cpu_cnt=None,
            memory_size_bytes=int((profile.memory_hierarchy_spec.platform_config_hbm_mib or 0.0) * 1024 * 1024),
            l2_size_bytes=int((profile.memory_hierarchy_spec.l2_capacity_mib or 0.0) * 1024 * 1024),
            cube_freq_mhz=float(profile.clock_and_topology_spec.configured_clock_mhz or 0.0),
            l0_a_bytes=int(profile.cube_spec.l0a_kib * 1024),
            l0_b_bytes=int(profile.cube_spec.l0b_kib * 1024),
            l0_c_bytes=int(profile.cube_spec.l0c_kib * 1024),
            l1_bytes=int(profile.cube_spec.l1_kib * 1024),
            ub_bytes=int(profile.vector_spec.allocatable_ub_kib * 1024),
            link_rates_bpc=profile.memory_hierarchy_spec.link_ceilings_bpc,
            source=",".join(profile.resolution_sources) or "<sku-profile>",
        )
    cross = cross_check_capacities(
        config,
        snapshot,
        verified_runtime_hbm_mib=verified_runtime_hbm_mib,
        simulator_ub_kib=simulator_ub_kib,
        sustained_mhz_range=sustained_mhz_range,
    )
    payload = {
        "status": "modeled" if cross["runtime_verified"] else "runtime_unverified",
        "unmodeled_reason": None,
        "device_capability_profile": profile.to_dict(),
        "effective_clock_mhz": profile.effective_clock_mhz(),
        "effective_clock_source": profile.clock_and_topology_spec.clock_source(),
        "soc": config.to_dict(),
        "capacity_cross_check": cross,
        "cannsim_links": dict(cannsim_links) if cannsim_links else default_link_ceilings(),
    }
    if snapshot is not None:
        payload["npu_smi"] = snapshot.to_dict()
    if topo_snapshot is not None:
        payload["topology"] = topo_snapshot.to_dict()
    if pcie_snapshot is not None:
        payload["pcie_link"] = pcie_snapshot.to_dict()
    if transfer_plan:
        payload["transfers"] = evaluate_transfer_plan(
            transfer_plan,
            profile.memory_hierarchy_spec.transfer_channels,
            dies_per_package=profile.clock_and_topology_spec.dies_per_package,
        )
    if buffer_plan:
        capacities = buffer_plan.get("capacities_kib") or {
            "L1": profile.cube_spec.l1_kib,
            "L0A": profile.cube_spec.l0a_kib,
            "L0B": profile.cube_spec.l0b_kib,
            "L0C": profile.cube_spec.l0c_kib,
            "UB": profile.vector_spec.allocatable_ub_kib,
        }
        payload["buffers"] = validate_buffer_allocations(
            buffer_plan.get("allocations", ()),
            capacities,
            alignment_bytes=int(buffer_plan.get("alignment_bytes", DEFAULT_ALIGNMENT_BYTES)),
            ping_pong=buffer_plan.get("ping_pong"),
        )
    thermal = _thermal_for_profile(
        profile,
        sustained_mhz_range=sustained_mhz_range,
        short_burst_mhz_range=short_burst_mhz_range,
        extra_frequencies_mhz=extra_frequencies_mhz,
        workload=workload,
        fp8_flops=fp8_flops,
        bf16_flops=bf16_flops,
        target_us=target_us,
        wall_cycles=wall_cycles,
        counted_cycles=counted_cycles,
        vector_cycles=vector_cycles,
        fix_cycles=fix_cycles,
        cube_reference_cycles=cube_reference_cycles,
        fp8_ops_per_cycle=fp8_ops_per_cycle,
        bf16_ops_per_cycle=bf16_ops_per_cycle,
        measurement_mhz=measurement_mhz,
        workload_origin=workload_origin,
    )
    payload["thermal"] = _attach_link_bounds(thermal, workload, cannsim_links)
    return payload


# Public aliases used by the CLI and tests.
from .theoretical_model import (  # noqa: E402
    OperatorHapEvaluation,
    compute_hap_score,
    evaluate_operator_hap,
)

parse_ini = parse_platform_ini
cross_check = cross_check_capacities
parse_cannsim = parse_cannsim_link_ceilings
__all__ = [
    "AMBIGUOUS_SKU_REASON", "BUILTIN_SKU_CATALOG", "HardwareError",
    "PlatformConfig", "CubeSpec", "VectorSpec", "MemoryHierarchySpec", "ClockAndTopologySpec",
    "DeviceCapabilityProfile", "NpuSmiDevice", "NpuSmiChip", "NpuSmiSnapshot", "BufferAllocation",
    # ``SUSTAINED_MHZ_RANGE`` is a deprecated alias for the A5 cannsim reference
    # ladder; no report defaults to it any more.
    "A5_CANN_SIM_LINK_CEILINGS_BPC", "A5_REFERENCE_PEAKS", "SUSTAINED_MHZ_RANGE",
    "SHORT_BURST_MHZ_RANGE", "DEFAULT_ALIGNMENT_BYTES", "PING_PONG_ALIGNMENT_BYTES",
    "align_up", "infer_arch_family", "is_ambiguous_family_token", "load_sku_profile",
    "resolve_device_capability_profile", "parse_platform_ini", "parse_platform_ini_text",
    "parse_ini", "parse_npu_smi_info", "parse_npu_smi_common", "parse_npu_smi_board",
    "parse_npu_smi_snapshot", "load_npu_smi_snapshot", "cross_check_capacities", "cross_check",
    "parse_cannsim_link_ceilings", "parse_cannsim", "default_link_ceilings",
    "validate_buffer_allocations", "check_mx_scale_layout", "cube_reference_cycles",
    "cycles_to_us", "reference_time_us", "sensitivity_matrix", "thermal_bounds",
    "unmodeled_thermal_bounds", "A5_FLASHMLA_C8_GOLDEN_WORKLOAD",
    "link_reference_bounds", "hardware_report", "TRANSFER_CHANNELS", "TransferChannelSpec",
    "NpuTopologySnapshot", "PcieLinkSnapshot", "parse_npu_smi_topo", "parse_pcie_link_info",
    "default_transfer_channels", "solve_transfer_bound", "solve_collective_bound", "evaluate_transfer_plan",
    "OperatorHapEvaluation", "compute_hap_score", "evaluate_operator_hap",
]
