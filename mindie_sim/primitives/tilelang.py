"""Read and validate sealed TileLang-Ascend silicon calibration envelopes.

[Role] Load, SHA-256 verify, and cache immutable per-architecture TileLang API
  calibration tables covering L0 dispatch, L1 pipelined DMA, L2 SimtVF/SimdVF
  vector execution tiers, L3 Cube GEMM/blockscaled attainment, L4 flag sync,
  and verified on-chip UB ceilings.
[Invariants] Strictly read-only CPU loader; zero device imports or host writes.
  Always verifies canonical full-envelope SHA-256 before returning data;
  never synthesizes silent defaults for uncalibrated architectures.
[Timeouts & Locks] Bounded local JSON file reads with no locks or network I/O.
[Side-Effect Policy] Read local packaged JSON into an immutable LRU-cached record.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mindie_sim.schema.common import ModelingError

from .loader import _read_json, default_primitives_dir

TILELANG_CALIBRATION_SCHEMA_VERSION = "mindie-tilelang-api-calibration/1"
TILELANG_DEFAULT_ARCH_PRESET = "npu_arch_3510"
TILELANG_SEALED_FILES: Mapping[str, str] = MappingProxyType({
    "npu_arch_3510": "arch35_a5_950dt_tilelang.json",
})
TILELANG_ARCH_ALIASES: Mapping[str, str] = MappingProxyType({
    "npu_arch_3510": "npu_arch_3510",
    "arch35": "npu_arch_3510",
    "dav_3510": "npu_arch_3510",
    "dav-3510": "npu_arch_3510",
    "ascend950dt_9582": "npu_arch_3510",
    "ascend950dt_9581": "npu_arch_3510",
})
TILELANG_SEALED_ENVELOPE_SHA256: Mapping[str, str] = MappingProxyType({
    "npu_arch_3510": "b19cc4d48a11aa4ec41a715276828c31dc709d0bd94f73760a0211e2d9ade0ae",
})

TILELANG_DMA_STAGE_SINGLE = 1
TILELANG_DMA_STAGE_DOUBLE = 2
TILELANG_DMA_STAGE_TRIPLE = 3
TILELANG_CALIBRATION_CACHE_SIZE = 8
TILELANG_OPS_PER_GOP = 1_000_000_000.0
TILELANG_BYTES_PER_KIB = 1024.0
TILELANG_ACL_UB_OVERFLOW_CODE = "ACL 107000"

_REQUIRED_ATTAINMENT_KEYS = (
    "eta_copy_rw11_stages1",
    "eta_copy_rw11_stages2",
    "eta_copy_rw11_stages3",
    "eta_copy_rw21_simt",
    "eta_copy_rw21_simd",
    "eta_cube_bf16_fixpipe",
    "eta_cube_bf16_dualcopy_fp32",
    "eta_cube_bf16_atomic_fp32",
    "eta_cube_fp8_fixpipe",
    "eta_cube_fp8_dualcopy_fp32",
    "eta_cube_fp32_ieee",
    "eta_cube_hf32",
    "eta_cube_mxfp8_blockscaled",
    "eta_simt_reg_fma_vs_p11",
    "eta_simt_ub_fma_vs_p11",
    "eta_simd_sfu_exp_vs_p08",
)
_FP8_DTYPE_NAMES = frozenset({"fp8", "fp8_e4m3", "fp8_e4m3fn", "fp8_e5m2"})
_MXFP8_DTYPE_NAMES = frozenset({"mxfp8", "mxfp8_e4m3", "paired_e8m0"})
_FP32_DTYPE_NAMES = frozenset({"fp32", "float32"})


def compute_tilelang_envelope_sha256(payload: Mapping[str, Any]) -> str:
    """Return canonical SHA-256 digest of a TileLang calibration envelope."""
    body = {key: value for key, value in payload.items() if key != "envelope_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_arch_preset(arch_preset: str) -> str:
    key = str(arch_preset or "").strip().lower()
    resolved = TILELANG_ARCH_ALIASES.get(key)
    if resolved is None:
        raise ModelingError(f"tilelang_calibration_arch_unsupported:{arch_preset}")
    return resolved


def _require_mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ModelingError(f"tilelang_calibration_section_invalid:{label}")
    return raw


def _require_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(f"tilelang_calibration_field_invalid:{key}")
    return value.strip()


def _require_pos_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if type(value) is not int or value <= 0:
        raise ModelingError(f"tilelang_calibration_int_invalid:{key}")
    return value


def _require_pos_float(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"tilelang_calibration_float_invalid:{key}")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ModelingError(f"tilelang_calibration_float_invalid:{key}")
    return fval


def _metric_p50(section: Mapping[str, Any], *path: str) -> float:
    cur: Any = section
    for part in path:
        cur = _require_mapping(cur, ".".join(path)).get(part)
    band = _require_mapping(cur, ".".join(path))
    return _require_pos_float(band, "p50")


@dataclass(frozen=True, slots=True)
class TileLangCalibration:
    """Immutable silicon-calibrated TileLang-Ascend multi-level API envelope."""

    schema_version: str
    arch_preset: str
    observed_silicon: str
    tilelang_version: str
    bisheng_target: str
    envelope_sha256: str
    ascendc_l0_envelope_sha256: str
    total_ub_per_aiv: int
    simd_usable_ub_bytes: int
    simt_usable_ub_bytes: int
    simt_recommended_user_ub_bytes: int
    empirical_simt_launch_fail_dyn_ubuf_bytes: int
    aclgraph_supported: bool
    aclgraph_issue_us: float
    eager_tvm_ffi_prealloc_us: float
    eager_tvm_ffi_callee_alloc_us: float
    jit_compile_ms: float
    v_mte3_set_wait_pair_ns: float
    simt_reg_fma_gops_per_aiv: float
    simt_reg_exp_gops_per_aiv: float
    simt_ub_fma_gops_per_aiv: float
    simt_ub_exp_gops_per_aiv: float
    simt_reducer_sum_gops_per_aiv: float
    simd_reg_alu_gops_per_aiv: float
    simd_ub_stream_vadd_gops_per_aiv: float
    simd_sfu_exp_gops_per_aiv: float
    simd_vcvt_bf16_gbps_per_aiv: float
    attainment_ratios_vs_ascendc_l0: Mapping[str, float]
    ascendc_l0_reference_p50: Mapping[str, float]

    @property
    def eta_copy_rw11_stages1(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_copy_rw11_stages1"]

    @property
    def eta_copy_rw11_stages2(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_copy_rw11_stages2"]

    @property
    def eta_copy_rw11_stages3(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_copy_rw11_stages3"]

    @property
    def eta_copy_rw21_simt(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_copy_rw21_simt"]

    @property
    def eta_copy_rw21_simd(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_copy_rw21_simd"]

    @property
    def eta_cube_bf16_fixpipe(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_bf16_fixpipe"]

    @property
    def eta_cube_bf16_dualcopy_fp32(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_bf16_dualcopy_fp32"]

    @property
    def eta_cube_bf16_atomic_fp32(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_bf16_atomic_fp32"]

    @property
    def eta_cube_fp8_fixpipe(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_fp8_fixpipe"]

    @property
    def eta_cube_fp8_dualcopy_fp32(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_fp8_dualcopy_fp32"]

    @property
    def eta_cube_fp32_ieee(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_fp32_ieee"]

    @property
    def eta_cube_hf32(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_hf32"]

    @property
    def eta_cube_mxfp8_blockscaled(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_cube_mxfp8_blockscaled"]

    @property
    def eta_simt_reg_fma_vs_p11(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_simt_reg_fma_vs_p11"]

    @property
    def eta_simt_ub_fma_vs_p11(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_simt_ub_fma_vs_p11"]

    @property
    def eta_simd_sfu_exp_vs_p08(self) -> float:
        return self.attainment_ratios_vs_ascendc_l0["eta_simd_sfu_exp_vs_p08"]

    def ub_ceiling_bytes(self, *, uses_simt: bool) -> int:
        """Return verified usable UB ceiling for ``T.SimtVF`` vs ``T.SimdVF``."""
        return self.simt_usable_ub_bytes if uses_simt else self.simd_usable_ub_bytes

    def dispatch_floor_us(
        self, *, aclgraph_captured: bool = True, callee_alloc: bool = False,
    ) -> float:
        """Return per-op dispatch latency (us) for ACLGraph replay vs eager tvm_ffi."""
        if aclgraph_captured:
            return self.aclgraph_issue_us
        return self.eager_tvm_ffi_callee_alloc_us if callee_alloc else self.eager_tvm_ffi_prealloc_us

    def cube_attainment(
        self,
        dtype: str,
        *,
        epilogue: str = "fixpipe",
        blockscaled: bool = False,
        hf32: bool = False,
    ) -> float:
        """Return capped (<= 1.0) Cube attainment multiplier vs AscendC L0."""
        dt = str(dtype or "bf16").strip().lower()
        epi = str(epilogue or "fixpipe").strip().lower()
        if blockscaled or dt in _MXFP8_DTYPE_NAMES:
            raw = self.eta_cube_mxfp8_blockscaled
        elif dt in _FP8_DTYPE_NAMES:
            raw = self.eta_cube_fp8_dualcopy_fp32 if epi == "dual_copy" else self.eta_cube_fp8_fixpipe
        elif dt in _FP32_DTYPE_NAMES:
            raw = self.eta_cube_hf32 if hf32 else self.eta_cube_fp32_ieee
        elif epi == "dual_copy":
            raw = self.eta_cube_bf16_dualcopy_fp32
        elif epi == "atomic_add":
            raw = self.eta_cube_bf16_atomic_fp32
        else:
            raw = self.eta_cube_bf16_fixpipe
        return min(1.0, float(raw))

    def dma_attainment(
        self,
        stages: int = TILELANG_DMA_STAGE_DOUBLE,
        *,
        uses_simt: bool | None = None,
        rw_pattern: str = "rw11",
    ) -> float:
        """Return capped (<= 1.0) DMA copy attainment for ``T.Pipelined`` stages or RW2:1."""
        if type(stages) is not int or stages < TILELANG_DMA_STAGE_SINGLE:
            raise ModelingError(f"tilelang_dma_stages_invalid:{stages}")
        if stages <= TILELANG_DMA_STAGE_SINGLE:
            stage_eta = self.eta_copy_rw11_stages1
        elif stages == TILELANG_DMA_STAGE_DOUBLE:
            stage_eta = self.eta_copy_rw11_stages2
        else:
            stage_eta = self.eta_copy_rw11_stages3
        pattern = str(rw_pattern or "rw11").strip().lower()
        if pattern == "rw21":
            if uses_simt is None:
                raise ModelingError("tilelang_dma_rw21_requires_uses_simt")
            base_rw21 = self.eta_copy_rw21_simt if uses_simt else self.eta_copy_rw21_simd
            if stages == TILELANG_DMA_STAGE_DOUBLE:
                return min(1.0, float(base_rw21))
            scaled = float(base_rw21) * (float(stage_eta) / float(self.eta_copy_rw11_stages2))
            return min(1.0, scaled)
        if pattern != "rw11":
            raise ModelingError(f"tilelang_dma_pattern_invalid:{rw_pattern}")
        return min(1.0, float(stage_eta))

    def vector_gops_per_aiv(self, *, uses_simt: bool, op_kind: str | None = None) -> float:
        """Return calibrated per-AIV vector throughput (Gops/s/AIV) for SimtVF or SimdVF."""
        kind = str(op_kind or ("ub_fma" if uses_simt else "ub_stream")).strip().lower()
        if uses_simt:
            table = {
                "ub_fma": self.simt_ub_fma_gops_per_aiv,
                "reg_fma": self.simt_reg_fma_gops_per_aiv,
                "ub_exp": self.simt_ub_exp_gops_per_aiv,
                "sfu_exp": self.simt_ub_exp_gops_per_aiv,
                "reg_exp": self.simt_reg_exp_gops_per_aiv,
                "reduce": self.simt_reducer_sum_gops_per_aiv,
            }
        else:
            table = {
                "ub_stream": self.simd_ub_stream_vadd_gops_per_aiv,
                "ub_fma": self.simd_ub_stream_vadd_gops_per_aiv,
                "reg_alu": self.simd_reg_alu_gops_per_aiv,
                "reg_fma": self.simd_reg_alu_gops_per_aiv,
                "sfu_exp": self.simd_sfu_exp_gops_per_aiv,
                "exp": self.simd_sfu_exp_gops_per_aiv,
                "reduce": self.simd_ub_stream_vadd_gops_per_aiv,
            }
        rate = table.get(kind)
        if rate is None:
            raise ModelingError(f"tilelang_vector_mode_invalid:{kind}")
        return float(rate)

    def vector_ops_per_s_per_aiv(self, *, uses_simt: bool, op_kind: str | None = None) -> float:
        """Return calibrated per-AIV vector throughput in elementary ops/s/AIV."""
        return self.vector_gops_per_aiv(uses_simt=uses_simt, op_kind=op_kind) * TILELANG_OPS_PER_GOP

    def simt_vs_simd_ub_wall_warning(self, ub_live_bytes: int) -> str | None:
        """Return explicit ACL 107000 UB-wall warning when live bytes fall in (216 KiB, 248 KiB]."""
        live = int(ub_live_bytes)
        if self.simt_usable_ub_bytes < live <= self.simd_usable_ub_bytes:
            live_kib = float(live) / TILELANG_BYTES_PER_KIB
            simt_kib = float(self.simt_usable_ub_bytes) / TILELANG_BYTES_PER_KIB
            simd_kib = float(self.simd_usable_ub_bytes) / TILELANG_BYTES_PER_KIB
            return (
                f"UB live footprint {live} B ({live_kib:.1f} KiB) exceeds T.SimtVF "
                f"{simt_kib:.0f} KiB ({self.simt_usable_ub_bytes} B) ceiling "
                f"({TILELANG_ACL_UB_OVERFLOW_CODE}); requires T.SimdVF "
                f"(<= {simd_kib:.0f} KiB / {self.simd_usable_ub_bytes} B) or AscendC."
            )
        return None


def parse_tilelang_calibration(raw: Mapping[str, Any]) -> TileLangCalibration:
    """Validate and materialize a frozen ``TileLangCalibration`` from raw JSON mapping."""
    if not isinstance(raw, Mapping):
        raise ModelingError("tilelang_calibration_mapping_required")
    schema_version = _require_str(raw, "schema_version")
    if schema_version != TILELANG_CALIBRATION_SCHEMA_VERSION:
        raise ModelingError(f"tilelang_calibration_schema_invalid:{schema_version}")
    declared_sha = _require_str(raw, "envelope_sha256")
    computed_sha = compute_tilelang_envelope_sha256(raw)
    if declared_sha != computed_sha:
        raise ModelingError("tilelang_calibration_sha256_mismatch")

    ub = _require_mapping(raw.get("onchip_ub_ceilings_bytes"), "onchip_ub_ceilings_bytes")
    total_ub = _require_pos_int(ub, "total_ub_per_aiv")
    simd_ub = _require_pos_int(ub, "simd_usable_ub_bytes")
    simt_ub = _require_pos_int(ub, "simt_usable_ub_bytes")
    simt_rec_ub = _require_pos_int(ub, "simt_recommended_user_ub_bytes")
    emp_fail_ub = _require_pos_int(ub, "empirical_simt_launch_fail_dyn_ubuf_bytes")
    if not (simt_rec_ub <= simt_ub < simd_ub <= total_ub):
        raise ModelingError("tilelang_calibration_ub_ceilings_inconsistent")

    ratios_raw = _require_mapping(raw.get("attainment_ratios_vs_ascendc_l0"), "attainment_ratios_vs_ascendc_l0")
    ratios: dict[str, float] = {}
    for key in _REQUIRED_ATTAINMENT_KEYS:
        ratios[key] = _require_pos_float(ratios_raw, key)
    if not (ratios["eta_copy_rw11_stages1"] < ratios["eta_copy_rw11_stages2"] < ratios["eta_copy_rw11_stages3"]):
        raise ModelingError("tilelang_calibration_dma_stages_not_monotone")

    l0_ref_raw = _require_mapping(raw.get("ascendc_l0_reference_p50"), "ascendc_l0_reference_p50")
    l0_ref = {str(k): _require_pos_float(l0_ref_raw, str(k)) for k in l0_ref_raw}

    levels = _require_mapping(raw.get("levels"), "levels")
    l0 = _require_mapping(levels.get("L0_dispatch"), "levels.L0_dispatch")
    aclgraph = _require_mapping(l0.get("aclgraph_capture"), "levels.L0_dispatch.aclgraph_capture")
    acl_supported = aclgraph.get("supported")
    if not isinstance(acl_supported, bool):
        raise ModelingError("tilelang_calibration_aclgraph_supported_invalid")
    l2 = _require_mapping(levels.get("L2_vector_simt_vs_simd"), "levels.L2_vector_simt_vs_simd")
    l4 = _require_mapping(levels.get("L4_sync_primitives"), "levels.L4_sync_primitives")

    return TileLangCalibration(
        schema_version=schema_version,
        arch_preset=_require_str(raw, "arch_preset"),
        observed_silicon=_require_str(raw, "observed_silicon"),
        tilelang_version=_require_str(raw, "tilelang_version"),
        bisheng_target=_require_str(raw, "bisheng_target"),
        envelope_sha256=declared_sha,
        ascendc_l0_envelope_sha256=_require_str(raw, "ascendc_l0_envelope_sha256"),
        total_ub_per_aiv=total_ub,
        simd_usable_ub_bytes=simd_ub,
        simt_usable_ub_bytes=simt_ub,
        simt_recommended_user_ub_bytes=simt_rec_ub,
        empirical_simt_launch_fail_dyn_ubuf_bytes=emp_fail_ub,
        aclgraph_supported=acl_supported,
        aclgraph_issue_us=_metric_p50(aclgraph, "per_kernel_issue_us"),
        eager_tvm_ffi_prealloc_us=_metric_p50(l0, "eager_tvm_ffi_prealloc_us"),
        eager_tvm_ffi_callee_alloc_us=_metric_p50(l0, "eager_tvm_ffi_callee_alloc_us"),
        jit_compile_ms=_require_pos_float(l0, "jit_compile_ms"),
        v_mte3_set_wait_pair_ns=_metric_p50(l4, "v_mte3_set_wait_pair_ns", "ns_per_flag_pair"),
        simt_reg_fma_gops_per_aiv=_metric_p50(l2, "simt_reg_fma_t1024", "gops_per_s_per_aiv"),
        simt_reg_exp_gops_per_aiv=_metric_p50(l2, "simt_reg_exp_t1024", "gops_per_s_per_aiv"),
        simt_ub_fma_gops_per_aiv=_metric_p50(l2, "simt_ub_fma_t1024", "gops_per_s_per_aiv"),
        simt_ub_exp_gops_per_aiv=_metric_p50(l2, "simt_ub_exp_t512", "gops_per_s_per_aiv"),
        simt_reducer_sum_gops_per_aiv=_metric_p50(l2, "simt_reducer_sum_row4096", "gops_per_s_per_aiv"),
        simd_reg_alu_gops_per_aiv=_metric_p50(l2, "simd_alu_chain8", "gops_per_s_per_aiv"),
        simd_ub_stream_vadd_gops_per_aiv=_metric_p50(l2, "simd_ub_stream_vadd", "gops_per_s_per_aiv"),
        simd_sfu_exp_gops_per_aiv=_metric_p50(l2, "simd_sfu_vexp_chain3", "gops_per_s_per_aiv"),
        simd_vcvt_bf16_gbps_per_aiv=_metric_p50(l2, "simd_vcvt_bf16_roundtrip", "cast_gbps_per_aiv"),
        attainment_ratios_vs_ascendc_l0=MappingProxyType(ratios),
        ascendc_l0_reference_p50=MappingProxyType(l0_ref),
    )


@lru_cache(maxsize=TILELANG_CALIBRATION_CACHE_SIZE)
def _load_tilelang_calibration_cached(
    canonical_arch: str, directory_str: str, verify_pinned_sha: bool,
) -> TileLangCalibration:
    filename = TILELANG_SEALED_FILES.get(canonical_arch)
    if filename is None:
        raise ModelingError(f"tilelang_calibration_arch_unsupported:{canonical_arch}")
    path = Path(directory_str) / filename
    raw = _read_json(path, "tilelang_calibration")
    calibration = parse_tilelang_calibration(raw)
    if calibration.arch_preset != canonical_arch:
        raise ModelingError(
            f"tilelang_calibration_arch_mismatch:{calibration.arch_preset}:{canonical_arch}"
        )
    if verify_pinned_sha:
        pinned = TILELANG_SEALED_ENVELOPE_SHA256.get(canonical_arch)
        if pinned is not None and calibration.envelope_sha256 != pinned:
            raise ModelingError("tilelang_calibration_sha256_mismatch")
    return calibration


def load_tilelang_calibration(
    arch_preset: str = TILELANG_DEFAULT_ARCH_PRESET,
    *,
    primitives_dir: Path | None = None,
) -> TileLangCalibration:
    """Load, verify, and cache the sealed TileLang calibration table for ``arch_preset``."""
    canonical = _canonical_arch_preset(arch_preset)
    directory = default_primitives_dir() if primitives_dir is None else Path(primitives_dir)
    return _load_tilelang_calibration_cached(
        canonical, str(directory.resolve()), primitives_dir is None,
    )


def try_load_tilelang_calibration(
    arch_preset: str,
    *,
    primitives_dir: Path | None = None,
) -> TileLangCalibration | None:
    """Return sealed ``TileLangCalibration`` when ``arch_preset`` is supported, else ``None``."""
    try:
        canonical = _canonical_arch_preset(arch_preset)
    except ModelingError:
        return None
    directory = default_primitives_dir() if primitives_dir is None else Path(primitives_dir)
    return _load_tilelang_calibration_cached(
        canonical, str(directory.resolve()), primitives_dir is None,
    )


__all__ = [
    "TILELANG_ACL_UB_OVERFLOW_CODE",
    "TILELANG_ARCH_ALIASES",
    "TILELANG_CALIBRATION_SCHEMA_VERSION",
    "TILELANG_DEFAULT_ARCH_PRESET",
    "TILELANG_DMA_STAGE_DOUBLE",
    "TILELANG_DMA_STAGE_SINGLE",
    "TILELANG_DMA_STAGE_TRIPLE",
    "TILELANG_SEALED_ENVELOPE_SHA256",
    "TILELANG_SEALED_FILES",
    "TileLangCalibration",
    "compute_tilelang_envelope_sha256",
    "load_tilelang_calibration",
    "parse_tilelang_calibration",
    "try_load_tilelang_calibration",
]
