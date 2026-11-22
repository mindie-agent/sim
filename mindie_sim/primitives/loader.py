"""Read L0 primitive declarations and select certified or simulation rates.

[Role] Resolve SKU aliases, verify trusted envelopes and check die capabilities.
[Invariants] A trusted manifest digest covers the original full table document.
[Timeouts & Locks] Bounded local file reads with no locks or background work.
[Side-Effect Policy] Read local JSON and return immutable modeling inputs.
"""
from __future__ import annotations

import importlib.metadata
import json
import site
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import GENERIC_ARCH_ID, DatapathCapabilitySpec
from mindie_sim.schema.hardware_links import UNCALIBRATED_SKU_WARNING

from .schema import PRIMITIVE_SCHEMA_VERSION, PrimitiveCell, PrimitiveTable, compute_envelope_sha256, parse_primitive_table

_MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MS_PER_SECOND = 1000.0
NS_PER_MS = 1_000_000.0
P12_FLAG_CYCLES = 4.0
AIV_SCALAR_INIT_CYCLES = 2560.0
AIV_MIN_TILE_BYTES = 2048
AIV_FULL_DIE_BARRIER_ROUNDS = 3.0
MIX_AIC_AIV_BARRIER_ROUNDS = 2.0
MIX_AIC_AIC_BARRIER_ROUNDS = 2.0
MIX_AIC_PAIRED_SYNC_ROUNDS = 4.0
DEFAULT_CORE_CLOCK_HZ = 1_650_000_000.0
RateRole = Literal["sim", "dual"]


def default_primitives_dir() -> Path:
    checkout = Path(__file__).resolve().parents[2] / "data" / "primitives"
    if (checkout / "manifest.json").is_file():
        return checkout
    try:
        dist = importlib.metadata.distribution("mindie-sim")
    except importlib.metadata.PackageNotFoundError:
        dist = None
    if dist is not None:
        for rel in dist.files or ():
            if rel.name == "manifest.json" and "primitives" in rel.parts:
                located = Path(dist.locate_file(rel)).resolve()
                if located.is_file():
                    return located.parent
        located = Path(dist.locate_file("data/primitives/manifest.json")).resolve()
        if located.is_file():
            return located.parent
    for base in (Path(sys.prefix), Path(site.getuserbase())):
        candidate = base / "data" / "primitives"
        if (candidate / "manifest.json").is_file():
            return candidate
    return checkout


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelingError(f"primitive_json_duplicate_key:{key}")
        result[key] = value
    return result


def _read_json(path: Path, kind: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            encoded = stream.read(_MAX_DOCUMENT_BYTES + 1)
    except OSError as exc:
        raise ModelingError(f"primitive_{kind}_read_failed") from exc
    if len(encoded) > _MAX_DOCUMENT_BYTES:
        raise ModelingError(f"primitive_{kind}_too_large")
    try:
        raw = json.loads(encoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelingError(f"primitive_{kind}_json_invalid") from exc
    if not isinstance(raw, dict):
        raise ModelingError(f"primitive_{kind}_mapping_required")
    return raw


def load_manifest(primitives_dir: Path | None = None) -> dict[str, Any]:
    directory = default_primitives_dir() if primitives_dir is None else Path(primitives_dir)
    manifest = _read_json(directory / "manifest.json", "manifest")
    if manifest.get("schema_version") != PRIMITIVE_SCHEMA_VERSION or not isinstance(manifest.get("tables"), dict):
        raise ModelingError("primitive_manifest_schema_invalid")
    for entry in manifest["tables"].values():
        if not isinstance(entry, dict):
            raise ModelingError("primitive_manifest_entry_invalid")
        filename = entry.get("file")
        aliases = entry.get("compatible_skus", [])
        if (not isinstance(filename, str) or Path(filename).name != filename
                or Path(filename).suffix != ".json" or not isinstance(aliases, list)
                or any(not isinstance(alias, str) for alias in aliases)):
            raise ModelingError("primitive_manifest_entry_invalid")
    return manifest


def resolve_primitive_rate(
    table: PrimitiveTable, primitive_id: str, metric: str, *,
    mode: str | None = None, role: RateRole = "sim",
    allow_uncalibrated_declaration: bool = False,
) -> tuple[float, tuple[str, ...]]:
    cell = table.cell(primitive_id)
    if role == "dual":
        return cell.certified_upper_capacity(metric, mode=mode), ()
    if role != "sim":
        raise ModelingError(f"primitive_rate_role_invalid:{role}")
    band = cell.band(metric, mode=mode)
    warnings = _simulation_warnings(cell, mode, role, allow_uncalibrated_declaration, band_measured=band.p50 is not None)
    return band.resolved_sim_rate(allow_uncalibrated_declaration=allow_uncalibrated_declaration), warnings


def _nonnegative_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"primitive_integer_invalid:{label}")
    return value


def _declared_mode(cell: PrimitiveCell, mode: Any) -> str | None:
    return mode if isinstance(mode, str) and mode in cell.modes else None


def _optional_rate(table, primitive_id, metric, mode, role, allow, default):
    if default is not None:
        try:
            table.cell(primitive_id).band(metric, mode=mode)
        except ModelingError as exc:
            if exc.reason.startswith("primitive_metric_unknown:"):
                return default, ()
            raise
    return resolve_primitive_rate(table, primitive_id, metric, mode=mode, role=role,
                                  allow_uncalibrated_declaration=allow)


def _simulation_warnings(
    cell: PrimitiveCell, mode: str | None, role: RateRole, allow: bool, *, band_measured: bool = True,
) -> tuple[str, ...]:
    status = cell.mode_status(mode) if cell.supported else "unsupported"
    if status == "unsupported":
        raise ModelingError(f"primitive_unsupported:{cell.primitive_id}")
    if role == "sim" and (status == "unmeasured" or not band_measured):
        if not allow:
            raise ModelingError(f"primitive_unmeasured_requires_declaration:{cell.primitive_id}")
        return (UNCALIBRATED_SKU_WARNING,)
    return ()

def _check_datapath(table: PrimitiveTable, datapath: DatapathCapabilitySpec) -> None:
    silicon = table.provenance.silicon_fingerprint
    buffers = dict(datapath.buffers)
    simt = table.cell("P11_simt")
    if (
        (datapath.arch_id != GENERIC_ARCH_ID and table.arch_preset != str(datapath.arch_id))
        or silicon.aic_count != datapath.aic_count
        or silicon.aiv_count != datapath.aiv_count
        or any(getattr(silicon, f"{pool}_bytes") != buffers[pool]
               for pool in ("l1", "l0a", "l0b", "l0c", "ub", "l2") if pool in buffers)
        or (simt.status != "unmeasured" and simt.supported != (datapath.simt_ub_carveout_bytes > 0))
    ):
        raise ModelingError("primitive_table_arch_mismatch")


def _unique_entry(matches: list[Mapping[str, Any]], reason: str) -> Mapping[str, Any]:
    if len(matches) != 1:
        raise ModelingError(reason)
    return matches[0]


def load_primitives(
    sku_or_path: str | Path, *, datapath: DatapathCapabilitySpec | None = None,
    primitives_dir: Path | None = None, allow_uncalibrated_declaration: bool = False,
    require_calibrated: bool = False,
) -> PrimitiveTable:
    """Read a SKU or a table path, always verifying against the trusted manifest."""
    del allow_uncalibrated_declaration
    path = Path(sku_or_path)
    direct = isinstance(sku_or_path, Path) or path.suffix == ".json" or path.parent != Path(".") or path.is_file()
    directory = default_primitives_dir() if primitives_dir is None else Path(primitives_dir)
    entries = load_manifest(directory)["tables"]
    entry = None
    if not direct:
        sku = str(sku_or_path)
        entry = _unique_entry([
            value for key, value in entries.items()
            if sku in (key, value.get("sku"), *value.get("compatible_skus", ()))
        ], f"primitive_table_sku_unknown_or_ambiguous:{sku}")
        path = directory / entry["file"]
    raw = _read_json(path, "table")
    if direct:
        entry = _unique_entry([
            value for key, value in entries.items()
            if value["file"] == path.name or raw.get("sku") == value.get("sku", key)
        ], "primitive_table_sha256_mismatch")
    prov = raw.get("provenance") if isinstance(raw.get("provenance"), Mapping) else {}
    if (
        raw.get("envelope_sha256") != entry.get("envelope_sha256")
        or compute_envelope_sha256(raw) != entry.get("envelope_sha256")
        or raw.get("sku") != entry.get("sku", raw.get("sku"))
        or raw.get("arch_preset") != entry.get("arch_preset", raw.get("arch_preset"))
        or raw.get("cann_version") != entry.get("cann_version", raw.get("cann_version"))
        or prov.get("calibrated") != entry.get("calibrated", prov.get("calibrated"))
        or prov.get("capture_uri") != entry.get("capture_uri", prov.get("capture_uri"))
    ):
        raise ModelingError("primitive_table_sha256_mismatch")
    table = parse_primitive_table(raw)
    if datapath is not None:
        _check_datapath(table, datapath)
    if not table.provenance.calibrated:
        if require_calibrated:
            raise ModelingError("primitive_table_uncalibrated")
        object.__setattr__(table, "warnings", (UNCALIBRATED_SKU_WARNING,))
    return table


_SIM_RATE_METRICS: tuple[tuple[str, str, str, str | None], ...] = (
    ("mte2_aic", "P01_datacopy", "mte2_aic_hit_bytes_per_s_per_core", "aic_count"),
    ("mte2_aiv", "P01_datacopy", "mte2_aiv_hit_bytes_per_s_per_core", "aiv_count"),
    ("fixpipe", "P03_fixpipe", "fixpipe_in_bytes_per_s_per_aic", "aic_count"),
    ("mte3", "P04_mte3_store", "mte3_store_bytes_per_s_per_aiv", "aiv_count"),
    ("l2", "P05_l2_residency_reuse", "l2_die_bytes_per_s", None),
    ("vector", "P07_vec_alu", "vec_alu_ops_per_s_per_aiv", "aiv_count"),
    ("simt", "P11_simt", "simt_ops_per_s_per_aiv", "aiv_count"),
)
_MMAD_MODE_DTYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bf16", ("bf16", "bfloat16")),
    ("fp16", ("fp16", "float16")),
    ("fp32", ("fp32", "float32")),
    ("int8", ("int8",)),
    ("fp8_e4m3", ("fp8_e4m3", "fp8")),
    ("mxfp8", ("mxfp8",)),
    ("mxfp4", ("mxfp4",)),
)


@dataclass(frozen=True, slots=True)
class CalibratedRateOverlay:
    """Resolved per-die ``sim_rate_p50`` production pricing view from one PrimitiveTable."""

    sku: str
    cann_version: str
    envelope_sha256: str
    rates: tuple[tuple[str, float], ...] = ()
    compute_peaks: tuple[tuple[str, float], ...] = ()
    hbm_read_bytes_per_s: float | None = None
    hbm_write_bytes_per_s: float | None = None
    hbm_rw11_bytes_per_s: float | None = None
    hbm_rw21_bytes_per_s: float | None = None
    hbm_rw_alpha: float | None = None
    l2_die_bytes_per_s: float | None = None
    l2_contiguous_bytes_per_s: float | None = None
    coverage_gaps: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple:
        return (
            self.sku, self.cann_version, self.envelope_sha256,
            self.rates, self.compute_peaks,
            self.hbm_read_bytes_per_s, self.hbm_write_bytes_per_s,
            self.hbm_rw11_bytes_per_s, self.hbm_rw21_bytes_per_s,
            self.hbm_rw_alpha, self.l2_die_bytes_per_s,
            self.l2_contiguous_bytes_per_s,
        )


def _table_band_p50(table: PrimitiveTable, primitive_id: str, metric: str, *, mode: str | None = None) -> float | None:
    try:
        cell = table.cell(primitive_id)
    except ModelingError:
        return None
    if not cell.supported or (mode is not None and mode not in cell.modes) or cell.mode_status(mode) == "unsupported":
        return None
    try:
        band = cell.band(metric, mode=mode)
    except ModelingError:
        return None
    if band.p50 is None or band.p50 <= 0.0:
        return None
    return float(band.p50)


def _cap_cube_peak(datapath: DatapathCapabilitySpec, dtype: str, peak: float) -> float:
    try:
        arch_peak = datapath.cube_peak(dtype)
    except ModelingError:
        return peak
    return min(peak, arch_peak) if arch_peak > 0.0 else peak


def is_sku_compatible_with_table(
    table: Any, sku: str, *, primitives_dir: Path | None = None,
) -> bool:
    """True when ``sku`` is explicitly compatible with ``table`` per shipped manifest."""
    table_sku = str(getattr(table, "sku", "") or "")
    target_sku = str(sku or "")
    if not table_sku or not target_sku or table_sku == target_sku:
        return True
    try:
        entries = load_manifest(default_primitives_dir() if primitives_dir is None else Path(primitives_dir))["tables"]
    except ModelingError:
        return False
    entry = entries.get(table_sku) or next(
        (v for v in entries.values() if v.get("sku") == table_sku), None,
    )
    if entry is None:
        return False
    return target_sku in (table_sku, *entry.get("compatible_skus", ()))


def validate_table_sku_compatibility(
    table: Any, sku: str, *, primitives_dir: Path | None = None,
) -> None:
    """Fail closed when ``table`` is not manifest-compatible with ``sku``."""
    if not is_sku_compatible_with_table(table, sku, primitives_dir=primitives_dir):
        raise ModelingError(f"primitive_table_sku_mismatch:{getattr(table, 'sku', '')}:{sku}")


def resolve_sim_rate_overlay(
    table: Any, datapath: DatapathCapabilitySpec, *, sku: str = "",
) -> CalibratedRateOverlay:
    """Bind calibrated ``sim_rate_p50`` supply from ``table`` for one die's datapath."""
    if sku:
        validate_table_sku_compatibility(table, sku)
    if not isinstance(table, PrimitiveTable):
        from mindie_sim.optimization.contracts import normalize_capacity_per_second
        entries = getattr(table, "entries", ())
        if isinstance(entries, Mapping):
            entries = tuple(entries.values())
        rates: dict[str, float] = {}
        peaks: dict[str, float] = {}
        hbm_r = hbm_w = None
        for entry in entries:
            pid = str(getattr(entry, "primitive_id", ""))
            p50 = getattr(entry, "sim_rate_p50", None)
            unit = getattr(entry, "unit", None)
            if p50 is None or unit is None:
                continue
            val = float(normalize_capacity_per_second(float(p50), str(unit)))
            if pid in {"hbm_bus_r", "P01_hbm_read"}:
                hbm_r = min(val, datapath.hbm_read_bytes_per_s) if datapath.hbm_read_bytes_per_s > 0.0 else val
            elif pid in {"hbm_bus_w", "P01_hbm_write"}:
                hbm_w = min(val, datapath.hbm_write_bytes_per_s) if datapath.hbm_write_bytes_per_s > 0.0 else val
            elif pid.startswith("cube:"):
                dt = pid.split(":", 1)[1]
                peaks[dt] = _cap_cube_peak(datapath, dt, val)
            elif pid in {"hbm", "l2", "vector", "simt", "mte2_aic", "mte2_aiv", "fixpipe", "mte3"}:
                rates[pid] = val
            elif pid == "l2_bw":
                rates["l2"] = val
        if hbm_r is not None or hbm_w is not None:
            rates.setdefault("hbm", min(v for v in (hbm_r, hbm_w) if v is not None))
        return CalibratedRateOverlay(
            sku=str(getattr(table, "sku", "") or sku),
            cann_version="",
            envelope_sha256="",
            rates=tuple(sorted(rates.items())),
            compute_peaks=tuple(sorted(peaks.items())),
            hbm_read_bytes_per_s=hbm_r,
            hbm_write_bytes_per_s=hbm_w,
        )
    if datapath.arch_id != GENERIC_ARCH_ID and table.arch_preset != str(datapath.arch_id):
        raise ModelingError("primitive_table_arch_mismatch")
    if sku and table.sku and sku != table.sku:
        # Compatible SKUs within the same arch_preset are permitted when datapath matches.
        _check_datapath(table, datapath)
    rates_map: dict[str, float] = {}
    peaks_map: dict[str, float] = {}
    gaps: list[str] = []
    hbm_r = _table_band_p50(table, "P01_datacopy", "hbm_read_bytes_per_s")
    hbm_w = _table_band_p50(table, "P01_datacopy", "hbm_write_bytes_per_s")
    hbm_rw11 = _table_band_p50(table, "P01_datacopy", "hbm_rw11_bytes_per_s")
    hbm_rw21 = _table_band_p50(table, "P01_datacopy", "hbm_rw21_bytes_per_s")
    if hbm_r is not None and datapath.hbm_read_bytes_per_s > 0.0:
        hbm_r = min(hbm_r, datapath.hbm_read_bytes_per_s)
    if hbm_w is not None and datapath.hbm_write_bytes_per_s > 0.0:
        hbm_w = min(hbm_w, datapath.hbm_write_bytes_per_s)
    if hbm_rw11 is not None and datapath.hbm_read_bytes_per_s > 0.0:
        hbm_rw11 = min(hbm_rw11, datapath.hbm_read_bytes_per_s)
    if hbm_rw21 is not None and datapath.hbm_read_bytes_per_s > 0.0:
        hbm_rw21 = min(hbm_rw21, datapath.hbm_read_bytes_per_s)
    l2_die = _table_band_p50(table, "P05_l2_residency_reuse", "l2_die_bytes_per_s")
    l2_contiguous = None
    try:
        p05_params = table.cell("P05_l2_residency_reuse").parameters or {}
        for entry in p05_params.get("l2_stride_conflict") or ():
            if isinstance(entry, Mapping) and entry.get("kind") == "contiguous":
                val = entry.get("bps_p50")
                if isinstance(val, (int, float)) and float(val) > 0.0:
                    l2_contiguous = float(val)
                    break
    except ModelingError:
        pass
    hbm_alpha_val = None
    try:
        alpha_band = table.cell("P01_datacopy").band("hbm_rw_alpha")
        if alpha_band.p50 is not None and alpha_band.p50 >= 0.0:
            hbm_alpha_val = float(alpha_band.p50)
    except ModelingError:
        pass
    if hbm_r is None:
        gaps.append("P01_datacopy:hbm_read_bytes_per_s")
    if hbm_w is None:
        gaps.append("P01_datacopy:hbm_write_bytes_per_s")
    if hbm_r is not None or hbm_w is not None:
        rates_map["hbm"] = min(v for v in (hbm_r, hbm_w) if v is not None)
    for role, pid, metric, owner_attr in _SIM_RATE_METRICS:
        per_unit = _table_band_p50(table, pid, metric)
        if per_unit is None:
            gaps.append(f"{pid}:{metric}")
            continue
        scale = int(getattr(datapath, owner_attr)) if owner_attr else 1
        if scale > 0:
            rates_map[role] = per_unit * scale
    aic = int(datapath.aic_count)
    if aic > 0:
        for mode, dtypes in _MMAD_MODE_DTYPES:
            per_aic = _table_band_p50(table, "P06_mmad", "cube_only_rate", mode=mode)
            if per_aic is None:
                gaps.append(f"P06_mmad:{mode}")
                continue
            peak = per_aic * aic
            for dt in dtypes:
                peaks_map[dt] = _cap_cube_peak(datapath, dt, peak)
    return CalibratedRateOverlay(
        sku=table.sku,
        cann_version=table.cann_version,
        envelope_sha256=table.canonical_envelope_sha256(),
        rates=tuple(sorted(rates_map.items())),
        compute_peaks=tuple(sorted(peaks_map.items())),
        hbm_read_bytes_per_s=hbm_r,
        hbm_write_bytes_per_s=hbm_w,
        hbm_rw11_bytes_per_s=hbm_rw11,
        hbm_rw21_bytes_per_s=hbm_rw21,
        hbm_rw_alpha=hbm_alpha_val,
        l2_die_bytes_per_s=l2_die,
        l2_contiguous_bytes_per_s=l2_contiguous,
        coverage_gaps=tuple(sorted(dict.fromkeys(gaps))),
    )


@dataclass(frozen=True, slots=True)
class CalibratedDispatchEnvelope:
    """Calibrated hardware kernel-launch and cross-core block-dispatch envelope."""

    sku: str
    envelope_sha256: str
    aiv_count: int
    aic_count: int
    mode4_supported: bool
    issue_interval_ms: float
    aiv_barrier_ms: float
    aic_barrier_ms: float
    paired_sync_ms: float
    stream_event_ms: float
    clock_period_ms: float
    aiv_min_block_floor_ms: float
    aiv_full_die_floor_ms: float

    def aiv_floor_ms(self, active_aiv: int) -> float:
        """Inclusive AIV kernel completion floor scaled by active block fanout."""
        clamped = max(1, min(self.aiv_count, int(active_aiv)))
        if self.aiv_count <= 1:
            return self.aiv_min_block_floor_ms
        fanout_fraction = float(clamped - 1) / float(self.aiv_count - 1)
        return self.aiv_min_block_floor_ms + fanout_fraction * (
            self.aiv_full_die_floor_ms - self.aiv_min_block_floor_ms
        )

    def mix_aic_floor_ms(self, cube_floor_ms: float) -> float:
        """Inclusive coupled MIX_AIC kernel completion floor (AIC + AIV + FFTS+ sync)."""
        handoff_ms = (
            MIX_AIC_AIV_BARRIER_ROUNDS * self.aiv_barrier_ms
            + MIX_AIC_AIC_BARRIER_ROUNDS * self.aic_barrier_ms
            + MIX_AIC_PAIRED_SYNC_ROUNDS * self.paired_sync_ms
            + self.stream_event_ms
        )
        return float(cube_floor_ms) + self.aiv_full_die_floor_ms + handoff_ms


def resolve_dispatch_envelope(
    table: Any, datapath: DatapathCapabilitySpec,
) -> CalibratedDispatchEnvelope | None:
    """Derive a physical dispatch/block-startup envelope from calibrated P12/P13/P15/P16."""
    if not isinstance(table, PrimitiveTable) or not table.provenance.calibrated:
        return None
    issue_ns = _table_band_p50(table, "P15_rts_stars", "issue_interval_ns")
    aiv_bar_ns = _table_band_p50(table, "P13_crosscore_sync", "mode0_aiv_barrier_ns")
    paired_ns = _table_band_p50(table, "P13_crosscore_sync", "mode2_paired_ns")
    if issue_ns is None or aiv_bar_ns is None or paired_ns is None:
        return None
    aic_bar_ns = _table_band_p50(table, "P13_crosscore_sync", "mode0_aic_barrier_ns")
    event_ns = _table_band_p50(table, "P16_multistream_core_control", "stream_event_ns") or 0.0
    flag_ns = _table_band_p50(table, "P12_intracore_flag", "flag_ns")
    if flag_ns is not None and flag_ns > 0.0:
        clock_period_ms = (flag_ns / P12_FLAG_CYCLES) / NS_PER_MS
    else:
        clock_period_ms = MS_PER_SECOND / DEFAULT_CORE_CLOCK_HZ
    p13_params = table.cell("P13_crosscore_sync").parameters or {}
    issue_ms = issue_ns / NS_PER_MS
    aiv_bar_ms = aiv_bar_ns / NS_PER_MS
    aic_bar_ms = (aic_bar_ns if aic_bar_ns is not None else aiv_bar_ns) / NS_PER_MS
    paired_ms = paired_ns / NS_PER_MS
    event_ms = event_ns / NS_PER_MS
    aiv_min_ms = issue_ms + aiv_bar_ms + AIV_SCALAR_INIT_CYCLES * clock_period_ms
    aiv_full_ms = (
        aiv_min_ms + issue_ms + AIV_FULL_DIE_BARRIER_ROUNDS * aiv_bar_ms + paired_ms
    )
    return CalibratedDispatchEnvelope(
        sku=table.sku,
        envelope_sha256=table.canonical_envelope_sha256(),
        aiv_count=max(1, int(datapath.aiv_count)),
        aic_count=max(1, int(datapath.aic_count)),
        mode4_supported=bool(p13_params.get("mode4_supported", False)),
        issue_interval_ms=issue_ms,
        aiv_barrier_ms=aiv_bar_ms,
        aic_barrier_ms=aic_bar_ms,
        paired_sync_ms=paired_ms,
        stream_event_ms=event_ms,
        clock_period_ms=clock_period_ms,
        aiv_min_block_floor_ms=aiv_min_ms,
        aiv_full_die_floor_ms=aiv_full_ms,
    )


__all__ = [
    "AIV_MIN_TILE_BYTES", "CalibratedDispatchEnvelope", "CalibratedRateOverlay",
    "RateRole", "default_primitives_dir", "is_sku_compatible_with_table",
    "load_manifest", "load_primitives", "resolve_dispatch_envelope",
    "resolve_primitive_rate", "resolve_sim_rate_overlay",
    "validate_table_sku_compatibility",
]
