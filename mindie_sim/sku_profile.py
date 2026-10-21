"""Multi-SKU capability schema and 3-tier resolution for A2 / A3 / A5 NPUs.

[Role]
  Represent Ascend A2 (910B*), A3 (910_93* dual-die) and A5 (950DT/PR*) device
  capabilities via four decoupled sub-specs (``cube_spec``, ``vector_spec``,
  ``memory_hierarchy_spec``, ``clock_and_topology_spec``) and resolve them across
  three tiers: (1) CANN ``.ini`` + exact SKU registry + external JSON profile,
  (2) live/snapshot ``npu-smi`` runtime probe override, and (3) explicit caller
  overrides, with safe degradation when only an ambiguous family name is passed.

[Invariants]
  * ``"A2"``, ``"A3"`` and ``"A5"`` are architecture families, never fixed SKUs.
    Passing a family name without an exact SKU, ``.ini``, ``sku_profile`` or
    disambiguating ``npu-smi`` snapshot yields ``status="ambiguous_sku"`` and
    ``unmodeled_reason="ambiguous_soc_sku_requires_exact_soc_or_ini_or_probe"``.
    An *exact* SKU the registry does not know is not that case: it yields
    ``unmodeled_reason="sku_not_registered:<soc>"`` so a catalog gap is not
    reported as a caller mistake (blocker B-5).
  * Architecture inference requires *positive* evidence: an ``910_93`` /
    ``910C`` token, a dual-die package or an A5 arch id.  ``NpuArch=2201`` is
    shared by A2 (910B4, CANN 9.1) and A3, so it never selects A3 by itself.
  * Clock precedence is single-sourced in
    :meth:`ClockAndTopologySpec.effective_clock_mhz`: explicit caller override >
    verified ``npu-smi`` runtime clock (``Aicore Freq(MHZ)``, the rated maximum)
    > ``.ini``/catalog ``configured_clock_mhz``.  ``configured_clock_mhz`` keeps
    the declared value forever; every disagreement is recorded in
    ``discrepancies`` instead of overwriting a tier.
  * ``sustained_clock_mhz_range`` is empty unless a caller supplies a measured
    band; no family-level clock constant is substituted for a measurement.
  * A vector peak is *declared* (``ops_per_cycle_per_aiv``) or undeclared.  The
    accessor returns ``None`` when undeclared; ``simd_width_bytes`` is a SIMD
    width (bytes/cycle), never an ops/cycle count.
  * ``hbm_peak_bw_gbps`` is the declared registry peak, ``sustained_hbm_bw_gbps``
    a measured sustained figure, and ``hbm_peak_bw_verified`` states whether the
    peak itself has board evidence.  None of them is silently promoted.
  * ``platform_config_hbm_mib`` and ``verified_runtime_hbm_mib`` remain distinct;
    runtime probe overrides populate ``verified_runtime_hbm_mib`` and record a
    structured entry in ``discrepancies`` whenever they diverge from static data.
  * ``physical_ub_kib`` and ``allocatable_ub_kib`` remain distinct (e.g. 256 KiB
    simulator/physical UB vs 248 KiB allocatable UB on A5, 192 KiB on A2/A3).

[Timeouts & Locks]
  Pure in-memory resolution and optional local JSON file parsing.  No locks,
  sleeps, subprocesses or network calls.

[Side-Effect Policy]
  Reads only explicit ``sku_profile`` paths supplied by the caller.  Never mutates
  device state or global registries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .comm import (
    NpuTopologySnapshot,
    PcieLinkSnapshot,
    TransferChannelSpec,
    default_transfer_channels,
)
from .hardware_common import (
    A5_CANN_SIM_LINK_CEILINGS_BPC,
    UNMEASURED_MHZ_RANGE,
    HardwareError,
    _finite,
    _float_value,
    _int_value,
    parse_mhz_range,
)

if TYPE_CHECKING:
    from .hardware import NpuSmiSnapshot, PlatformConfig

AMBIGUOUS_SKU_REASON = "ambiguous_soc_sku_requires_exact_soc_or_ini_or_probe"

_AMBIGUOUS_FAMILY_TOKENS: dict[str, str] = {
    "A2": "A2",
    "ASCEND910B": "A2",
    "910B": "A2",
    "A3": "A3",
    "ASCEND910C": "A3",
    "910C": "A3",
    "ASCEND910_93": "A3",
    "910_93": "A3",
    "A5": "A5",
    "ASCEND950": "A5",
    "ASCEND950DT": "A5",
    "ASCEND950PR": "A5",
    "950": "A5",
}

_FAMILY_COMMON_BUFFERS: dict[str, dict[str, float]] = {
    "A2": {"l1_kib": 512.0, "l0a_kib": 64.0, "l0b_kib": 64.0, "l0c_kib": 128.0, "physical_ub_kib": 192.0, "allocatable_ub_kib": 192.0},
    "A3": {"l1_kib": 512.0, "l0a_kib": 64.0, "l0b_kib": 64.0, "l0c_kib": 128.0, "physical_ub_kib": 192.0, "allocatable_ub_kib": 192.0},
    "A5": {"l1_kib": 512.0, "l0a_kib": 64.0, "l0b_kib": 64.0, "l0c_kib": 256.0, "physical_ub_kib": 256.0, "allocatable_ub_kib": 248.0},
}

_FAMILY_DEFAULT_OPS_PER_CYCLE: dict[str, dict[str, int]] = {
    "A2": {"int8": 16384, "bf16": 8192, "fp16": 8192, "fp32": 4096},
    "A3": {"int8": 16384, "bf16": 8192, "fp16": 8192, "fp32": 4096},
    "A5": {"fp8_e4m3": 16384, "fp8_e5m2": 16384, "int8": 16384, "bf16": 8192, "fp16": 8192, "fp32": 4096},
}

# Dtype spelling aliases only.  They resolve inside the canonical capability
# maps (``CubeSpec.ops_per_cycle_per_aic`` / ``VectorSpec.ops_per_cycle_per_aiv``);
# there is no alternate capability source and no cross-architecture fallback.
_DTYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "fp8": ("fp8", "fp8_e4m3", "fp8_e5m2"),
    "fp8_e4m3": ("fp8_e4m3", "fp8"),
    "fp8_e5m2": ("fp8_e5m2", "fp8"),
    "bf16": ("bf16", "bfloat16"),
    "bfloat16": ("bfloat16", "bf16"),
    "fp16": ("fp16", "float16"),
    "float16": ("float16", "fp16"),
    "fp32": ("fp32", "float32"),
    "float32": ("float32", "fp32"),
    "int8": ("int8",),
}

_A2_A3_LINK_CEILINGS_BPC: dict[str, float] = {
    "gm_l2_to_l1": 110.0,
    "l1_to_l0a": 512.0,
    "l1_to_l0b": 256.0,
    "l1_to_ub": 128.0,
    "l0c_to_ub": 256.0,
    "ub_to_l2": 64.0,
    "ub_to_ddr": 64.0,
    "ub_to_l1": 128.0,
}

# Built-in exact SKU catalog for A2 / A3 / A5 sub-models.
BUILTIN_SKU_CATALOG: dict[str, dict[str, Any]] = {
    "Ascend910B1": {
        "arch_family": "A2", "short_soc_version": "Ascend910B", "dies_per_package": 1,
        "aic_count_per_die": 24, "aiv_count_per_die": 48, "configured_clock_mhz": 1850.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1800.0,
    },
    "Ascend910B2": {
        "arch_family": "A2", "short_soc_version": "Ascend910B", "dies_per_package": 1,
        "aic_count_per_die": 24, "aiv_count_per_die": 48, "configured_clock_mhz": 1800.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1800.0,
    },
    "Ascend910B2C": {
        "arch_family": "A2", "short_soc_version": "Ascend910B", "dies_per_package": 1,
        "aic_count_per_die": 24, "aiv_count_per_die": 48, "configured_clock_mhz": 1800.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1800.0,
    },
    "Ascend910B3": {
        "arch_family": "A2", "short_soc_version": "Ascend910B", "dies_per_package": 1,
        "aic_count_per_die": 20, "aiv_count_per_die": 40, "configured_clock_mhz": 1800.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1600.0,
    },
    "Ascend910B4": {
        "arch_family": "A2", "short_soc_version": "Ascend910B", "dies_per_package": 1,
        "aic_count_per_die": 20, "aiv_count_per_die": 40, "configured_clock_mhz": 1500.0,
        "platform_config_hbm_mib": 32768.0, "l2_capacity_mib": 96.0, "hbm_peak_bw_gbps": 800.0,
        # The registry peak has no local source: only npu-smi "HBM Clock Speed
        # 1600 MHz" exists, and 1600 MT/s x 4096 bit / 8 = 819.2 GB/s rests on an
        # assumed bus width (.workstreams/a2-recon/a2-hardware-evidence.md §7.1).
        "hbm_peak_bw_verified": False,
    },
    "Ascend910B4-1": {
        "arch_family": "A2", "short_soc_version": "Ascend910B", "dies_per_package": 1,
        "aic_count_per_die": 20, "aiv_count_per_die": 40, "configured_clock_mhz": 1650.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 96.0, "hbm_peak_bw_gbps": 1200.0,
    },
    "Ascend910_9391": {
        "arch_family": "A3", "short_soc_version": "Ascend910_93", "npu_arch": "2201", "dies_per_package": 2,
        "aic_count_per_die": 24, "aiv_count_per_die": 48, "configured_clock_mhz": 1850.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1600.0,
        "intra_package_die_bw_gbps": 448.0,
    },
    "Ascend910_9392": {
        "arch_family": "A3", "short_soc_version": "Ascend910_93", "npu_arch": "2201", "dies_per_package": 2,
        "aic_count_per_die": 24, "aiv_count_per_die": 48, "configured_clock_mhz": 1800.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1600.0,
        "intra_package_die_bw_gbps": 448.0,
    },
    "Ascend910_9381": {
        "arch_family": "A3", "short_soc_version": "Ascend910_93", "npu_arch": "2201", "dies_per_package": 2,
        "aic_count_per_die": 20, "aiv_count_per_die": 40, "configured_clock_mhz": 1800.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1600.0,
        "intra_package_die_bw_gbps": 448.0,
    },
    "Ascend910_9382": {
        "arch_family": "A3", "short_soc_version": "Ascend910_93", "npu_arch": "2201", "dies_per_package": 2,
        "aic_count_per_die": 20, "aiv_count_per_die": 40, "configured_clock_mhz": 1650.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1600.0,
        "intra_package_die_bw_gbps": 448.0,
    },
    "Ascend910_9372": {
        "arch_family": "A3", "short_soc_version": "Ascend910_93", "npu_arch": "2201", "dies_per_package": 2,
        "aic_count_per_die": 20, "aiv_count_per_die": 40, "configured_clock_mhz": 1650.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 192.0, "hbm_peak_bw_gbps": 1400.0,
        "intra_package_die_bw_gbps": 448.0,
    },
    # Ascend910_9362 is the standing A3 host (mindie-qwen35hist, 2026-10-21).
    # Board evidence for the per-die core counts, both live captures:
    #   * npu-smi info -t common -> "Chip Count: 2" and "Aicore Count: 20" on
    #     each of the two chips (/Volumes/data/mindie-sim-data/a3/2026-10-20/
    #     production-loop-v3-30-20261021/live-environment.json, clock_common_*)
    #   * torch_npu device properties -> cube_core_num=20, vector_core_num=40
    #     (/Volumes/data/mindie-sim-data/a3/2026-10-20/graph-replay-20261021/
    #     capability-results.json)
    # The previous 16/32 was the "2x AIC" convention applied to an assumed
    # 16-AIC die; both sources say 20 AIC / 40 AIV per die (a3-dual-die).
    # L2 is 168 MiB, not 192: the live CANN platform config declares
    # l2_size=176160768 and torch_npu on the A3 host reports
    # L2_cache_size=176160768 (176160768 / 1024^2 = 168.0 MiB,
    # /Volumes/data/mindie-sim-data/a3/2026-10-20/graph-replay-20261021/
    # capability-results.json).  The A2 96 MiB entry stays as-is (its own
    # .ini l2_size=100663296 agrees with the same kind of capture).
    # The 1200 GB/s HBM peak stays UNCHANGED and unverified: live memory-bound
    # GEMMs imply 1.45-1.64 TB/s while the streamed weights fit in L2 (168 MiB)
    # and 0.69-1.09 TB/s once they exceed it
    # (.workstreams/residual-analysis/decomposition.md §3.7), so the campaign
    # must decide whether 1200 is an HBM peak or an L2-resident rate.
    "Ascend910_9362": {
        "arch_family": "A3", "short_soc_version": "Ascend910_93", "npu_arch": "2201", "dies_per_package": 2,
        "aic_count_per_die": 20, "aiv_count_per_die": 40, "configured_clock_mhz": 1650.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 168.0, "hbm_peak_bw_gbps": 1200.0,
        "hbm_peak_bw_verified": False,
        "intra_package_die_bw_gbps": 448.0,
    },
    "Ascend950DT_9581": {
        "arch_family": "A5", "short_soc_version": "Ascend950", "npu_arch": "3510", "dies_per_package": 1,
        "aic_count_per_die": 32, "aiv_count_per_die": 64, "configured_clock_mhz": 1650.0,
        "platform_config_hbm_mib": 147456.0, "l2_capacity_mib": 128.0, "hbm_peak_bw_gbps": 4000.0,
    },
    "Ascend950DT_9582": {
        "arch_family": "A5", "short_soc_version": "Ascend950", "npu_arch": "3510", "dies_per_package": 1,
        "aic_count_per_die": 32, "aiv_count_per_die": 64, "configured_clock_mhz": 1650.0,
        "platform_config_hbm_mib": 98304.0, "l2_capacity_mib": 128.0, "hbm_peak_bw_gbps": 4000.0,
    },
    "Ascend950PR_9579": {
        "arch_family": "A5", "short_soc_version": "Ascend950", "npu_arch": "3510", "dies_per_package": 1,
        "aic_count_per_die": 32, "aiv_count_per_die": 64, "configured_clock_mhz": 1650.0,
        "platform_config_hbm_mib": 65536.0, "l2_capacity_mib": 128.0, "hbm_peak_bw_gbps": 1600.0,
    },
}


@dataclass(frozen=True)
class CubeSpec:
    aic_count_per_die: int | None
    ops_per_cycle_per_aic: Mapping[str, int]
    l1_kib: float
    l0a_kib: float
    l0b_kib: float
    l0c_kib: float

    def ops_per_cycle_per_die(self, dtype: str) -> float | None:
        """Total per-die Cube ops/cycle for ``dtype``, or ``None`` if undeclared.

        The stored map is per-AIC; this accessor is the single unit-explicit
        normalization shared by hardware/thermal and A3.  Dtype spelling
        aliases resolve within the same map.  A missing or non-positive entry
        returns ``None`` rather than borrowing another architecture's peak.
        """
        if self.aic_count_per_die is None or int(self.aic_count_per_die) <= 0:
            return None
        key = str(dtype).strip().lower()
        for candidate in _DTYPE_ALIASES.get(key, (key,)):
            raw = self.ops_per_cycle_per_aic.get(candidate)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            if not _finite(raw) or float(raw) <= 0:
                continue
            return float(raw) * float(self.aic_count_per_die)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aic_count_per_die": self.aic_count_per_die,
            "ops_per_cycle_per_aic": dict(sorted(self.ops_per_cycle_per_aic.items())),
            "l1_kib": self.l1_kib,
            "l0a_kib": self.l0a_kib,
            "l0b_kib": self.l0b_kib,
            "l0c_kib": self.l0c_kib,
        }


@dataclass(frozen=True)
class VectorSpec:
    aiv_count_per_die: int | None
    aiv_per_aic_ratio: int
    simd_width_bytes: int
    physical_ub_kib: float
    allocatable_ub_kib: float
    #: Declared per-AIV vector ops/cycle, keyed by dtype.  Empty means the
    #: capability is undeclared: the 128 B/clk ``vec_calc_size`` in the A2
    #: ``.ini`` is a SIMD width, and no board source in this tree states an A2/A3
    #: AIV ops/cycle count (see ``.workstreams/code-audit/a2-blockers.md`` B8).
    ops_per_cycle_per_aiv: Mapping[str, int] = field(default_factory=dict)

    def ops_per_cycle_per_die(self, dtype: str) -> float | None:
        """Total per-die AIV ops/cycle for ``dtype``, or ``None`` if undeclared.

        Mirrors :meth:`CubeSpec.ops_per_cycle_per_die`: the stored map is
        per-AIV, this accessor is the single unit-explicit normalization.  A
        dtype nobody declared returns ``None``; ``simd_width_bytes`` is
        deliberately not consulted, because bytes/cycle is not ops/cycle.
        """
        if self.aiv_count_per_die is None or int(self.aiv_count_per_die) <= 0:
            return None
        key = str(dtype).strip().lower()
        for candidate in _DTYPE_ALIASES.get(key, (key,)):
            raw = self.ops_per_cycle_per_aiv.get(candidate)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            if not _finite(raw) or float(raw) <= 0:
                continue
            return float(raw) * float(self.aiv_count_per_die)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aiv_count_per_die": self.aiv_count_per_die,
            "aiv_per_aic_ratio": self.aiv_per_aic_ratio,
            "simd_width_bytes": self.simd_width_bytes,
            "physical_ub_kib": self.physical_ub_kib,
            "allocatable_ub_kib": self.allocatable_ub_kib,
            "ops_per_cycle_per_aiv": dict(sorted(self.ops_per_cycle_per_aiv.items())),
        }


@dataclass(frozen=True)
class MemoryHierarchySpec:
    platform_config_hbm_mib: float | None
    verified_runtime_hbm_mib: float | None
    hbm_peak_bw_gbps: float | None
    sustained_hbm_bw_gbps: float | None
    l2_capacity_mib: float | None
    link_ceilings_bpc: Mapping[str, float]
    intra_package_die_bw_gbps: float | None = None
    intra_node_hccs_bw_gbps: float | None = None
    inter_node_roce_bw_gbps: float | None = None
    transfer_channels: Mapping[str, TransferChannelSpec] = field(default_factory=dict)
    #: ``True``/``False`` only when an authority states it; ``None`` = undeclared.
    #: A registry peak that no measurement confirms stays ``False`` and is never
    #: promoted to ``sustained_hbm_bw_gbps``.
    hbm_peak_bw_verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_config_hbm_mib": self.platform_config_hbm_mib,
            "verified_runtime_hbm_mib": self.verified_runtime_hbm_mib,
            "hbm_peak_bw_gbps": self.hbm_peak_bw_gbps,
            "sustained_hbm_bw_gbps": self.sustained_hbm_bw_gbps,
            "hbm_peak_bw_verified": self.hbm_peak_bw_verified,
            "l2_capacity_mib": self.l2_capacity_mib,
            "link_ceilings_bpc": dict(sorted(self.link_ceilings_bpc.items())),
            "intra_package_die_bw_gbps": self.intra_package_die_bw_gbps,
            "intra_node_hccs_bw_gbps": self.intra_node_hccs_bw_gbps,
            "inter_node_roce_bw_gbps": self.inter_node_roce_bw_gbps,
            "transfer_channels": {k: v.to_dict() for k, v in self.transfer_channels.items()},
        }


@dataclass(frozen=True)
class ClockAndTopologySpec:
    arch_family: str
    soc_version: str | None
    short_soc_version: str | None
    npu_arch: str | None
    dies_per_package: int
    configured_clock_mhz: float | None
    sustained_clock_mhz_range: tuple[float, ...]
    runtime_clock_mhz: float | None = None
    #: Explicit caller override; highest precedence, never written into
    #: ``configured_clock_mhz``.
    clock_override_mhz: float | None = None
    #: DVFS observation (``Aicore curFreq``).  Reported, never a peak.
    runtime_current_clock_mhz: float | None = None
    #: Provenance of ``sustained_clock_mhz_range``; ``None`` when unmeasured.
    sustained_clock_source: str | None = None

    def clock_source(self) -> str | None:
        """Which tier ``effective_clock_mhz`` resolves to, or ``None``."""
        for value, source in (
            (self.clock_override_mhz, "caller_override"),
            (self.runtime_clock_mhz, "npu_smi_runtime"),
            (self.configured_clock_mhz, "platform_config"),
        ):
            if value is not None and _finite(value) and float(value) > 0:
                return source
        return None

    def effective_clock_mhz(self) -> float | None:
        """Clock the model must use, by documented precedence.

        1. explicit caller override (``clock_override_mhz``),
        2. verified ``npu-smi`` runtime clock (``Aicore Freq(MHZ)`` — the rated
           maximum the part reaches under load, not the DVFS ``Aicore curFreq``),
        3. the ``.ini`` / catalog declared clock (``configured_clock_mhz``).

        ``configured_clock_mhz`` is never overwritten by this resolution: all
        three values stay readable and every disagreement is a
        ``discrepancies`` entry.  Returns ``None`` when no tier is usable.
        """
        for value in (self.clock_override_mhz, self.runtime_clock_mhz, self.configured_clock_mhz):
            if value is not None and _finite(value) and float(value) > 0:
                return float(value)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch_family": self.arch_family,
            "soc_version": self.soc_version,
            "short_soc_version": self.short_soc_version,
            "npu_arch": self.npu_arch,
            "dies_per_package": self.dies_per_package,
            "configured_clock_mhz": self.configured_clock_mhz,
            "sustained_clock_mhz_range": list(self.sustained_clock_mhz_range),
            "sustained_clock_source": self.sustained_clock_source,
            "runtime_clock_mhz": self.runtime_clock_mhz,
            "runtime_current_clock_mhz": self.runtime_current_clock_mhz,
            "clock_override_mhz": self.clock_override_mhz,
            # Derived, not stored: the precedence result, so reports do not
            # re-implement the ordering.
            "effective_clock_mhz": self.effective_clock_mhz(),
            "effective_clock_source": self.clock_source(),
        }


@dataclass(frozen=True)
class DeviceCapabilityProfile:
    status: str
    unmodeled_reason: str | None
    cube_spec: CubeSpec
    vector_spec: VectorSpec
    memory_hierarchy_spec: MemoryHierarchySpec
    clock_and_topology_spec: ClockAndTopologySpec
    discrepancies: tuple[Mapping[str, Any], ...] = ()
    resolution_sources: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "unmodeled_reason": self.unmodeled_reason,
            "cube_spec": self.cube_spec.to_dict(),
            "vector_spec": self.vector_spec.to_dict(),
            "memory_hierarchy_spec": self.memory_hierarchy_spec.to_dict(),
            "clock_and_topology_spec": self.clock_and_topology_spec.to_dict(),
            "discrepancies": [dict(item) for item in self.discrepancies],
            "resolution_sources": list(self.resolution_sources),
        }

    def effective_clock_mhz(self) -> float | None:
        """:meth:`ClockAndTopologySpec.effective_clock_mhz` for the profile."""
        return self.clock_and_topology_spec.effective_clock_mhz()

    def _peak_clock(self, clock_mhz: float | None) -> float | None:
        value = self.effective_clock_mhz() if clock_mhz is None else clock_mhz
        if value is None or not _finite(value) or float(value) <= 0:
            return None
        return float(value)

    def peak_cube_ops_per_s(self, dtype: str = "bf16", *, clock_mhz: float | None = None) -> float | None:
        """Per-die Cube ops/s from the declared capability, or ``None``.

        ``clock_mhz`` defaults to :meth:`effective_clock_mhz`; the result is
        ``ops_per_cycle_per_die(dtype) x clock`` and never a registry literal.
        """
        clock = self._peak_clock(clock_mhz)
        ops = self.cube_spec.ops_per_cycle_per_die(dtype)
        if clock is None or ops is None:
            return None
        return ops * clock * 1e6

    def peak_vector_ops_per_s(self, dtype: str = "bf16", *, clock_mhz: float | None = None) -> float | None:
        """Per-die AIV ops/s from the declared capability, or ``None``.

        Returns ``None`` when ``ops_per_cycle_per_aiv`` does not declare the
        dtype: a vector peak may not be synthesized from ``simd_width_bytes``
        (bytes/cycle is not ops/cycle), so an undeclared capability stays
        unmodeled instead of borrowing 128 or 256.
        """
        clock = self._peak_clock(clock_mhz)
        ops = self.vector_spec.ops_per_cycle_per_die(dtype)
        if clock is None or ops is None:
            return None
        return ops * clock * 1e6


def infer_arch_family(soc_or_family: str | None, *, npu_arch: str | None = None, dual_die: bool = False) -> str:
    """Infer ``A2``/``A3``/``A5`` from *positive* board evidence only.

    ``NpuArch`` values are not a family discriminator on their own: CANN 9.1
    tags an A2 ``Ascend910B4`` with ``NpuArch=2201``, the same id A3 parts use
    (``.workstreams/a2-recon/a2-hardware-evidence.md`` §7.5).  A3 therefore
    requires an ``910_93``/``910C`` SKU token or a dual-die package; an A2
    ``910B`` token wins over the shared arch id.  The final ``"A5"`` is the
    legacy catch-all for a caller-supplied non-Ascend token.
    """
    token = (soc_or_family or "").strip().upper()
    if token in _AMBIGUOUS_FAMILY_TOKENS:
        return _AMBIGUOUS_FAMILY_TOKENS[token]
    if npu_arch == "3510" or "950" in token:
        return "A5"
    if "910_93" in token or "910C" in token or dual_die:
        return "A3"
    if "910B" in token or "910" in token:
        return "A2"
    return "A5"


def is_ambiguous_family_token(soc_or_family: str | None) -> bool:
    if not soc_or_family:
        return False
    return soc_or_family.strip().upper() in _AMBIGUOUS_FAMILY_TOKENS


def is_exact_sku_token(soc: str | None) -> bool:
    """Whether ``soc`` names one SKU rather than a family/model prefix.

    ``Ascend910`` is a model prefix (a family selection), while
    ``Ascend910B4``, ``Ascend910_9362`` and ``Ascend950DT_9599`` name a part:
    the registry may or may not have a row for it, but the caller did supply an
    exact identity.  This only separates the *diagnosis*; it never grants a
    capability.
    """
    if not soc:
        return False
    token = soc.strip()
    if is_ambiguous_family_token(token):
        return False
    rest = token.lower()
    if not rest.startswith("ascend"):
        return False
    rest = rest[len("ascend"):]
    digits = 0
    while digits < len(rest) and rest[digits].isdigit():
        digits += 1
    return digits >= 3 and len(rest) > digits


def load_sku_profile(path_or_mapping: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_mapping, Mapping):
        return dict(path_or_mapping)
    path = Path(path_or_mapping).expanduser()
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise HardwareError(f"sku_profile_unreadable:{type(exc).__name__}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HardwareError("sku_profile_json_invalid") from exc
    if not isinstance(data, Mapping):
        raise HardwareError("sku_profile_must_be_object")
    return dict(data)


def _snapshot_exact_soc(snapshot: NpuSmiSnapshot | None) -> str | None:
    if snapshot is None or not snapshot.devices:
        return None
    raw_name = snapshot.devices[0].name.strip()
    if not raw_name:
        return None
    prefixed = raw_name if raw_name.startswith("Ascend") else f"Ascend{raw_name}"
    if is_ambiguous_family_token(prefixed):
        return None
    return prefixed


def resolve_device_capability_profile(
    *,
    config: PlatformConfig | None = None,
    soc: str | None = None,
    sku_profile: str | Path | Mapping[str, Any] | None = None,
    snapshot: NpuSmiSnapshot | None = None,
    topo_snapshot: NpuTopologySnapshot | None = None,
    pcie_snapshot: PcieLinkSnapshot | None = None,
    verified_runtime_hbm_mib: float | None = None,
    simulator_ub_kib: float | None = None,
    allocatable_ub_kib: float | None = None,
    sustained_mhz_range: Sequence[float] | None = None,
    clock_override_mhz: float | None = None,
    cannsim_links: Mapping[str, Any] | None = None,
) -> DeviceCapabilityProfile:
    """Resolve a device profile from the strongest evidence the caller supplies.

    Tiers: custom ``sku_profile`` < CANN ``.ini`` < exact SKU catalog < npu-smi
    snapshot < explicit caller arguments.  Two clocks are kept apart:
    ``configured_clock_mhz`` is the declared ``.ini``/catalog value and
    ``runtime_clock_mhz`` the verified npu-smi rated clock; which one a model
    should use is decided only by ``ClockAndTopologySpec.effective_clock_mhz``
    (override > runtime > configured).  ``sustained_mhz_range`` is a measured
    band and stays empty unless the caller passes one.
    """
    custom = load_sku_profile(sku_profile) if sku_profile is not None else {}
    cube_override = dict(custom.get("cube_spec") or {}) if isinstance(custom.get("cube_spec"), Mapping) else {}
    vec_override = dict(custom.get("vector_spec") or {}) if isinstance(custom.get("vector_spec"), Mapping) else {}
    mem_override = dict(custom.get("memory_hierarchy_spec") or {}) if isinstance(custom.get("memory_hierarchy_spec"), Mapping) else {}
    clk_override = dict(custom.get("clock_and_topology_spec") or {}) if isinstance(custom.get("clock_and_topology_spec"), Mapping) else {}

    # The sustained band is a measurement.  ``None``/empty stays unmeasured and
    # nothing (least of all a family constant) is substituted for it.
    sustained = parse_mhz_range(sustained_mhz_range, reason="sku_invalid_sustained_mhz_range")
    sustained_range: tuple[float, ...] = sustained or UNMEASURED_MHZ_RANGE
    sustained_source = "caller_supplied" if sustained else None
    # Explicit caller override: highest clock precedence, never stored into
    # ``configured_clock_mhz``.
    override_f = _float_value(float(clock_override_mhz), "sku_invalid_clock_override") if clock_override_mhz is not None else None
    if override_f is not None and override_f <= 0:
        raise HardwareError("sku_invalid_clock_override")

    sources: list[str] = []
    if config is not None:
        sources.append(f"platform_ini:{config.source}")
    snap_soc = _snapshot_exact_soc(snapshot)
    explicit_soc = (
        clk_override.get("soc_version")
        or custom.get("soc_version")
        or (config.soc_version if config is not None else None)
        or (soc if soc and not is_ambiguous_family_token(soc) else None)
        or snap_soc
    )
    catalog_entry = dict(BUILTIN_SKU_CATALOG.get(str(explicit_soc), {})) if explicit_soc else {}
    if catalog_entry:
        sources.append(f"builtin_sku:{explicit_soc}")
    if custom:
        sources.append("custom_sku_profile")
    if snapshot is not None:
        sources.append(f"npu_smi:{snapshot.source}")
    if topo_snapshot is not None:
        sources.append(f"npu_smi_topo:{topo_snapshot.source}")
    if pcie_snapshot is not None:
        sources.append(f"pcie_link:{pcie_snapshot.source}")

    family = str(
        clk_override.get("arch_family")
        or custom.get("arch_family")
        or catalog_entry.get("arch_family")
        or infer_arch_family(
            explicit_soc or soc,
            npu_arch=(config.npu_arch if config is not None else None),
            dual_die=bool(snapshot and snapshot.dual_die),
        )
    )
    common_buf = _FAMILY_COMMON_BUFFERS.get(family, _FAMILY_COMMON_BUFFERS["A5"])

    # Check if we only have an ambiguous family token and no exact SKU / INI / probe disambiguation.
    has_exact_sku = bool(config is not None or catalog_entry or custom.get("soc_version") or clk_override.get("soc_version") or snap_soc)
    # An exact token that no tier registers is *not* an ambiguous family name:
    # the caller named one SKU, and the resolution failed because the registry
    # has no row for it.  Reporting the family-token reason there hides a
    # catalog gap behind a caller mistake (blocker B-5).
    unregistered_exact_soc = bool(
        is_exact_sku_token(str(explicit_soc)) if explicit_soc else False
    ) and not catalog_entry and config is None and snap_soc is None
    if not has_exact_sku:
        phys_ub = float(simulator_ub_kib) if simulator_ub_kib is not None else common_buf["physical_ub_kib"]
        alloc_ub = float(allocatable_ub_kib) if allocatable_ub_kib is not None else common_buf["allocatable_ub_kib"]
        links = dict(_A2_A3_LINK_CEILINGS_BPC if family in {"A2", "A3"} else A5_CANN_SIM_LINK_CEILINGS_BPC)
        unmodeled_reason = (
            f"sku_not_registered:{explicit_soc}"
            if unregistered_exact_soc else AMBIGUOUS_SKU_REASON
        )
        provenance = (
            ("unregistered_exact_soc",)
            if unregistered_exact_soc else tuple(sources or ["ambiguous_family_only"])
        )
        return DeviceCapabilityProfile(
            status="ambiguous_sku",
            unmodeled_reason=unmodeled_reason,
            cube_spec=CubeSpec(
                aic_count_per_die=None,
                ops_per_cycle_per_aic=dict(_FAMILY_DEFAULT_OPS_PER_CYCLE.get(family, _FAMILY_DEFAULT_OPS_PER_CYCLE["A5"])),
                l1_kib=common_buf["l1_kib"],
                l0a_kib=common_buf["l0a_kib"],
                l0b_kib=common_buf["l0b_kib"],
                l0c_kib=common_buf["l0c_kib"],
            ),
            vector_spec=VectorSpec(
                aiv_count_per_die=None,
                aiv_per_aic_ratio=2,
                simd_width_bytes=256 if family == "A5" else 128,
                physical_ub_kib=phys_ub,
                allocatable_ub_kib=alloc_ub,
            ),
            memory_hierarchy_spec=MemoryHierarchySpec(
                platform_config_hbm_mib=None,
                verified_runtime_hbm_mib=snapshot.min_hbm_mib if snapshot is not None else verified_runtime_hbm_mib,
                hbm_peak_bw_gbps=None,
                sustained_hbm_bw_gbps=None,
                l2_capacity_mib=None,
                link_ceilings_bpc=links,
            ),
            clock_and_topology_spec=ClockAndTopologySpec(
                arch_family=family,
                soc_version=None,
                short_soc_version=soc,
                npu_arch=None,
                dies_per_package=2 if (snapshot and snapshot.dual_die) or family == "A3" else 1,
                configured_clock_mhz=None,
                sustained_clock_mhz_range=sustained_range,
                sustained_clock_source=sustained_source,
                clock_override_mhz=override_f,
            ),
            discrepancies=(),
            resolution_sources=provenance,
        )

    discrepancies: list[dict[str, Any]] = []
    static_aic = (
        cube_override.get("aic_count_per_die")
        or custom.get("aic_count_per_die")
        or (config.cube_core_cnt if config is not None else None)
        or catalog_entry.get("aic_count_per_die")
    )
    static_aiv = (
        vec_override.get("aiv_count_per_die")
        or custom.get("aiv_count_per_die")
        or (config.vector_core_cnt if config is not None else None)
        or catalog_entry.get("aiv_count_per_die")
        or (int(static_aic) * 2 if static_aic is not None else None)
    )
    runtime_aic = None
    runtime_clock = None
    runtime_current_clock = None
    if snapshot is not None and snapshot.chips:
        counts = sorted({chip.aicore_count for chip in snapshot.chips if chip.aicore_count})
        if counts:
            runtime_aic = counts[0]
            if static_aic is not None and runtime_aic != int(static_aic):
                discrepancies.append({
                    "field": "aic_count_per_die",
                    "static_value": int(static_aic),
                    "runtime_value": runtime_aic,
                    "authority": "npu_smi_runtime",
                })
        # "Aicore Freq(MHZ)" is the rated maximum the part reaches under load;
        # "Aicore curFreq(MHZ)" is the instantaneous DVFS point and never a peak.
        runtime_clock = snapshot.rated_aicore_freq_mhz
        runtime_current_clock = snapshot.observed_aicore_freq_mhz

    resolved_aic = _int_value(int(runtime_aic if runtime_aic is not None else static_aic), "sku_invalid_aic") if (runtime_aic or static_aic) else None
    resolved_aiv = (
        _int_value(int(runtime_aic * 2 if (runtime_aic is not None and static_aic and runtime_aic != int(static_aic)) else static_aiv), "sku_invalid_aiv")
        if (runtime_aic or static_aiv)
        else None
    )

    plat_hbm = (
        mem_override.get("platform_config_hbm_mib")
        or custom.get("platform_config_hbm_mib")
        or (config.platform_config_hbm_mib if config is not None else None)
        or catalog_entry.get("platform_config_hbm_mib")
    )
    plat_hbm_f = _float_value(float(plat_hbm), "sku_invalid_hbm") if plat_hbm is not None else None
    rt_hbm = snapshot.min_hbm_mib if (snapshot is not None and snapshot.devices) else (
        float(verified_runtime_hbm_mib) if verified_runtime_hbm_mib is not None else mem_override.get("verified_runtime_hbm_mib")
    )
    rt_hbm_f = _float_value(float(rt_hbm), "sku_invalid_runtime_hbm") if rt_hbm is not None else None
    if plat_hbm_f is not None and rt_hbm_f is not None and abs(plat_hbm_f - rt_hbm_f) > 0.5:
        discrepancies.append({
            "field": "hbm_capacity_mib",
            "platform_config_hbm_mib": plat_hbm_f,
            "verified_runtime_hbm_mib": rt_hbm_f,
            "difference_mib": plat_hbm_f - rt_hbm_f,
            "authority": "verified_runtime",
        })

    cfg_clock = (
        clk_override.get("configured_clock_mhz")
        or custom.get("configured_clock_mhz")
        or (config.cube_freq_mhz if config is not None else None)
        or catalog_entry.get("configured_clock_mhz")
    )
    cfg_clock_f = _float_value(float(cfg_clock), "sku_invalid_clock") if cfg_clock is not None else None
    if cfg_clock_f is not None and runtime_clock is not None and abs(cfg_clock_f - runtime_clock) > 0.5:
        # Both numbers are real and answer different questions: the .ini value is
        # the declared configuration, npu-smi's "Aicore Freq" the rated maximum
        # reached under load.  They stay separate and the disagreement is logged.
        discrepancies.append({
            "field": "configured_clock_mhz",
            "platform_config_clock_mhz": cfg_clock_f,
            "runtime_aicore_freq_mhz": runtime_clock,
            "observed_current_freq_mhz": runtime_current_clock,
            "caller_override_clock_mhz": override_f,
            "authority": "npu_smi_runtime",
        })
    if override_f is not None and any(
        other is not None and abs(override_f - other) > 0.5 for other in (cfg_clock_f, runtime_clock)
    ):
        discrepancies.append({
            "field": "effective_clock_mhz",
            "caller_override_clock_mhz": override_f,
            "platform_config_clock_mhz": cfg_clock_f,
            "runtime_aicore_freq_mhz": runtime_clock,
            "authority": "caller_override",
        })

    base_alloc_ub = (
        allocatable_ub_kib
        if allocatable_ub_kib is not None
        else vec_override.get("allocatable_ub_kib")
        or custom.get("allocatable_ub_kib")
        or (config.platform_allocatable_ub_kib if config is not None else None)
        or common_buf["allocatable_ub_kib"]
    )
    alloc_ub_f = _float_value(float(base_alloc_ub), "sku_invalid_allocatable_ub")
    base_phys_ub = (
        simulator_ub_kib
        if simulator_ub_kib is not None
        else vec_override.get("physical_ub_kib")
        or custom.get("physical_ub_kib")
        or common_buf["physical_ub_kib"]
    )
    phys_ub_f = _float_value(float(base_phys_ub), "sku_invalid_physical_ub")
    if abs(phys_ub_f - alloc_ub_f) > 0.5:
        discrepancies.append({
            "field": "ub_kib",
            "physical_or_simulator_ub_kib": phys_ub_f,
            "allocatable_ub_kib": alloc_ub_f,
            "difference_kib": phys_ub_f - alloc_ub_f,
            "authority": "allocatable_ub_kib",
        })

    links_raw = (
        cannsim_links.get("ceilings_bpc", cannsim_links)
        if isinstance(cannsim_links, Mapping)
        else (_A2_A3_LINK_CEILINGS_BPC if family in {"A2", "A3"} else A5_CANN_SIM_LINK_CEILINGS_BPC)
    )
    links = {str(k): float(v) for k, v in (mem_override.get("link_ceilings_bpc") or links_raw).items()}
    dies = int(
        clk_override.get("dies_per_package")
        or custom.get("dies_per_package")
        or (snapshot.chip_count if snapshot is not None and snapshot.chip_count > 1 else None)
        or catalog_entry.get("dies_per_package")
        or (2 if family == "A3" else 1)
    )
    l2_mib = (
        mem_override.get("l2_capacity_mib")
        or custom.get("l2_capacity_mib")
        or (config.platform_config_l2_mib if config is not None else None)
        or catalog_entry.get("l2_capacity_mib")
    )
    ops_map = dict(_FAMILY_DEFAULT_OPS_PER_CYCLE.get(family, _FAMILY_DEFAULT_OPS_PER_CYCLE["A5"]))
    if isinstance(cube_override.get("ops_per_cycle_per_aic"), Mapping):
        ops_map.update({str(k): int(v) for k, v in cube_override["ops_per_cycle_per_aic"].items()})
    # Declared AIV capability: catalog first, caller profile last.  Undeclared
    # stays empty -> ``VectorSpec.ops_per_cycle_per_die`` returns ``None``; the
    # SIMD width is never converted into an ops/cycle count.
    vec_ops_map: dict[str, int] = {}
    for source_map in (catalog_entry.get("ops_per_cycle_per_aiv"), vec_override.get("ops_per_cycle_per_aiv")):
        if isinstance(source_map, Mapping):
            vec_ops_map.update({str(k): int(v) for k, v in source_map.items()})
    # HBM peak bandwidth evidence tier: the declared registry peak and a measured
    # sustained figure stay separate, and the peak is only "verified" when an
    # authority says so (use ``is None`` because ``False`` is a real value).
    bw_verified: bool | None = None
    for candidate in (mem_override.get("hbm_peak_bw_verified"), custom.get("hbm_peak_bw_verified"), catalog_entry.get("hbm_peak_bw_verified")):
        if candidate is not None:
            bw_verified = bool(candidate)
            break
    intra_pkg_bw = (
        float(mem_override.get("intra_package_die_bw_gbps") or catalog_entry.get("intra_package_die_bw_gbps"))
        if (mem_override.get("intra_package_die_bw_gbps") or catalog_entry.get("intra_package_die_bw_gbps"))
        else None
    )
    intra_node_bw = float(mem_override["intra_node_hccs_bw_gbps"]) if "intra_node_hccs_bw_gbps" in mem_override else None
    inter_node_bw = float(mem_override["inter_node_roce_bw_gbps"]) if "inter_node_roce_bw_gbps" in mem_override else None
    if pcie_snapshot is not None and pcie_snapshot.downgraded:
        discrepancies.append({
            "field": "pcie_link_speed",
            "cap_speed_gtps": pcie_snapshot.cap_speed_gtps,
            "sta_speed_gtps": pcie_snapshot.sta_speed_gtps,
            "cap_raw_uni_bw_gbps": round(pcie_snapshot.cap_raw_uni_bw_gbps, 3),
            "sta_raw_uni_bw_gbps": round(pcie_snapshot.sta_raw_uni_bw_gbps, 3),
            "effective_pinned_uni_bw_gbps": round(pcie_snapshot.effective_pinned_uni_bw_gbps, 3),
            "authority": "lspci_lnksta",
        })
    channels = default_transfer_channels(
        family,
        dies_per_package=dies,
        intra_package_die_bw_gbps=intra_pkg_bw,
        intra_node_hccs_bw_gbps=intra_node_bw,
        inter_node_roce_bw_gbps=inter_node_bw,
        pcie_snapshot=pcie_snapshot,
        topo_snapshot=topo_snapshot,
        overrides=mem_override.get("transfer_channels") if isinstance(mem_override.get("transfer_channels"), Mapping) else None,
    )

    return DeviceCapabilityProfile(
        status="modeled" if rt_hbm_f is not None else "runtime_unverified",
        unmodeled_reason=None,
        cube_spec=CubeSpec(
            aic_count_per_die=resolved_aic,
            ops_per_cycle_per_aic=ops_map,
            l1_kib=float(cube_override.get("l1_kib") or (config.l1_kib if config is not None else common_buf["l1_kib"])),
            l0a_kib=float(cube_override.get("l0a_kib") or (config.l0_a_kib if config is not None else common_buf["l0a_kib"])),
            l0b_kib=float(cube_override.get("l0b_kib") or (config.l0_b_kib if config is not None else common_buf["l0b_kib"])),
            l0c_kib=float(cube_override.get("l0c_kib") or (config.l0_c_kib if config is not None else common_buf["l0c_kib"])),
        ),
        vector_spec=VectorSpec(
            aiv_count_per_die=resolved_aiv,
            aiv_per_aic_ratio=int(vec_override.get("aiv_per_aic_ratio", 2)),
            simd_width_bytes=int(vec_override.get("simd_width_bytes", 256 if family == "A5" else 128)),
            physical_ub_kib=phys_ub_f,
            allocatable_ub_kib=alloc_ub_f,
            ops_per_cycle_per_aiv=vec_ops_map,
        ),
        memory_hierarchy_spec=MemoryHierarchySpec(
            platform_config_hbm_mib=plat_hbm_f,
            verified_runtime_hbm_mib=rt_hbm_f,
            hbm_peak_bw_gbps=float(mem_override["hbm_peak_bw_gbps"]) if "hbm_peak_bw_gbps" in mem_override else (
                float(catalog_entry["hbm_peak_bw_gbps"]) if "hbm_peak_bw_gbps" in catalog_entry else None
            ),
            sustained_hbm_bw_gbps=float(mem_override["sustained_hbm_bw_gbps"]) if "sustained_hbm_bw_gbps" in mem_override else None,
            l2_capacity_mib=float(l2_mib) if l2_mib is not None else None,
            link_ceilings_bpc=links,
            intra_package_die_bw_gbps=intra_pkg_bw,
            intra_node_hccs_bw_gbps=intra_node_bw or channels["d2d_intra_node"].effective_uni_bw_gbps,
            inter_node_roce_bw_gbps=inter_node_bw or channels["d2d_inter_node"].effective_uni_bw_gbps,
            transfer_channels=channels,
            hbm_peak_bw_verified=bw_verified,
        ),
        clock_and_topology_spec=ClockAndTopologySpec(
            arch_family=family,
            soc_version=str(explicit_soc) if explicit_soc else None,
            short_soc_version=str(
                clk_override.get("short_soc_version")
                or (config.short_soc_version if config is not None else None)
                or catalog_entry.get("short_soc_version")
                or explicit_soc
            ),
            npu_arch=str(
                clk_override.get("npu_arch")
                or (config.npu_arch if config is not None else None)
                or catalog_entry.get("npu_arch")
            ) if (clk_override.get("npu_arch") or (config and config.npu_arch) or catalog_entry.get("npu_arch")) else None,
            dies_per_package=dies,
            configured_clock_mhz=cfg_clock_f,
            sustained_clock_mhz_range=sustained_range,
            sustained_clock_source=sustained_source,
            runtime_clock_mhz=runtime_clock,
            runtime_current_clock_mhz=runtime_current_clock,
            clock_override_mhz=override_f,
        ),
        discrepancies=tuple(discrepancies),
        resolution_sources=tuple(sources),
    )


__all__ = [
    "AMBIGUOUS_SKU_REASON",
    "BUILTIN_SKU_CATALOG",
    "CubeSpec",
    "VectorSpec",
    "MemoryHierarchySpec",
    "ClockAndTopologySpec",
    "DeviceCapabilityProfile",
    "infer_arch_family",
    "is_ambiguous_family_token",
    "is_exact_sku_token",
    "load_sku_profile",
    "resolve_device_capability_profile",
]
