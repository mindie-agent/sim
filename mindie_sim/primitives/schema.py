"""Immutable L0 primitive calibration tables.

[Role] Validate measurement bands, capture provenance and sealed JSON envelopes.
[Invariants] Simulation medians and certified bounds retain their distinct meanings.
[Timeouts & Locks] No I/O or locking.
[Side-Effect Policy] Freeze independent copies of caller-supplied records.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from mindie_sim.prediction.collectives.protocol import frame_wire_bytes as _protocol_frame_wire_bytes
from mindie_sim.prediction.runtime.memory_physics import (
    DEFAULT_SECTOR_BYTES, AddressRun, AddressRunPlan, burst_turnaround_efficiency,
)
from mindie_sim.schema.common import ModelingError, align_up
from mindie_sim.schema.hardware_links import CollectiveBandSpec

if TYPE_CHECKING:
    from mindie_sim.prediction.program.stages import KernelProgram, Stage
    from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
    from mindie_sim.prediction.runtime.memory_physics import BoundAccess, HbmBusSpec, MemoryAccessPattern
    from mindie_sim.prediction.runtime.prepared_terms import BoundVectorOp
    from mindie_sim.prediction.runtime.ring_buffer import RingBuffer
    from mindie_sim.schema.device_input import StarsSpec

PRIMITIVE_SCHEMA_VERSION = "mindie-l0-primitives/1"
CellStatus = Literal["calibrated", "unmeasured", "unsupported"]
BoundKind = Literal["rate_upper", "latency_lower"]
MAX_CALIBRATED_CV_PCT = 5.0
PRIMITIVE_IDS = (
    "P01_datacopy", "P02_load_data", "P03_fixpipe", "P04_mte3_store",
    "P05_l2_residency_reuse", "P06_mmad", "P07_vec_alu", "P08_vec_sfu",
    "P09_vec_reduce_cast", "P10_sort_topk", "P11_simt", "P12_intracore_flag",
    "P13_crosscore_sync", "P14_mutex_block_mmad", "P15_rts_stars",
    "P16_multistream_core_control", "P17_aicpu_metadata_launch",
    "P18_hccl_collective_band", "P19_direct_peer_window", "P20_host_dma_sync",
)
_STATUSES = frozenset({"calibrated", "unmeasured", "unsupported"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CAPTURE = re.compile(r"capture://[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*/[0-9a-f]{64}")
_PRIVATE = re.compile(r"/(?:Users|home|root)/|@|(?<!\d)\d{1,3}(?:[._-]\d{1,3}){3}(?!\d)")
_LOWER_BOUND_METRICS = frozenset({"hbm_rw_alpha", "kappa_trans", "transpose_factor"})


def _mapping(raw: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or any(type(key) is not str for key in raw):
        raise ModelingError(reason)
    return raw


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return tuple(map(_freeze, value)) if isinstance(value, (tuple, list)) else value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return list(map(_json_value, value))
    if isinstance(value, UncertaintyBand):
        return value.to_dict()
    return value


def _record(record: Any) -> dict[str, Any]:
    return {item.name: getattr(record, item.name) for item in fields(record)}


def _require_capture(uri: str, reason: str) -> None:
    if not isinstance(uri, str) or not _CAPTURE.fullmatch(uri) or "pending" in uri:
        raise ModelingError(reason)


@dataclass(frozen=True, slots=True)
class UncertaintyBand:
    certified_upper_capacity: float
    p05: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    trimmed_mean: float | None = None
    cv_pct: float | None = None
    n_samples: int = 0
    bound_kind: BoundKind = "rate_upper"

    def __post_init__(self) -> None:
        if self.bound_kind not in ("rate_upper", "latency_lower"):
            raise ModelingError("uncertainty_band_bound_kind_invalid")
        if type(self.n_samples) is not int or self.n_samples < 0:
            raise ModelingError("uncertainty_band_n_samples_invalid")
        for name in ("certified_upper_capacity", "p05", "p50", "p95", "p99", "trimmed_mean", "cv_pct"):
            value = getattr(self, name)
            if value is None and name != "certified_upper_capacity":
                continue
            try:
                valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ModelingError(f"uncertainty_band_number_invalid:{name}")
        quantiles = [value for value in (self.p05, self.p50, self.p95, self.p99) if value is not None]
        if quantiles != sorted(quantiles):
            raise ModelingError("uncertainty_band_quantile_order_invalid")
        has_any_sample_stat = any(v is not None for v in (self.p05, self.p50, self.p95, self.p99, self.cv_pct)) or self.n_samples > 0
        if has_any_sample_stat:
            if self.n_samples < 1:
                raise ModelingError("uncertainty_band_samples_required")
            if any(v is None for v in (self.p05, self.p50, self.p95, self.p99, self.cv_pct)):
                raise ModelingError("uncertainty_band_quantiles_required")
            if self.bound_kind == "rate_upper":
                if self.p05 <= 0 or self.p50 <= 0:
                    raise ModelingError("uncertainty_band_rate_nonpositive")
                if self.certified_upper_capacity < self.p50:
                    raise ModelingError("uncertainty_band_upper_below_p50")
                if self.certified_upper_capacity < self.p99:
                    raise ModelingError("uncertainty_band_upper_below_p99")
            else:
                if self.certified_upper_capacity > self.p50:
                    raise ModelingError("uncertainty_band_lower_above_p50")
                if self.certified_upper_capacity > self.p05:
                    raise ModelingError("uncertainty_band_lower_above_p05")

    @property
    def sim_rate_p50(self) -> float:
        if self.p50 is None:
            raise ModelingError("primitive_unmeasured_requires_declaration")
        return self.p50

    def resolved_sim_rate(self, *, allow_uncalibrated_declaration: bool = False) -> float:
        if self.p50 is not None:
            return self.p50
        if not allow_uncalibrated_declaration:
            raise ModelingError("primitive_unmeasured_requires_declaration")
        return self.certified_upper_capacity

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, bound_kind: BoundKind = "rate_upper") -> UncertaintyBand:
        raw = _mapping(raw, "uncertainty_band_mapping_required")
        if "certified_upper_capacity" not in raw:
            raise ModelingError("uncertainty_band_capacity_required")
        values = {item.name: raw[item.name] for item in fields(cls) if item.name in raw}
        values.setdefault("bound_kind", bound_kind)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {name: value for name, value in _record(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class SiliconFingerprint:
    aic_count: int
    aiv_count: int
    hbm_gib: int
    l2_bytes: int
    ub_bytes: int
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int

    def __post_init__(self) -> None:
        for name, value in _record(self).items():
            if type(value) is not int or value <= 0:
                raise ModelingError(f"primitive_silicon_fingerprint_invalid:{name}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SiliconFingerprint:
        raw = _mapping(raw, "primitive_silicon_fingerprint_mapping_required")
        return cls(**{item.name: raw.get(item.name) for item in fields(cls)})

    def to_dict(self) -> dict[str, int]:
        return _record(self)


@dataclass(frozen=True, slots=True)
class PrimitiveProvenance:
    campaign_id: str
    endpoint_alias: str
    capture_uri: str
    calibrated: bool
    collected_utc: str
    silicon_fingerprint: SiliconFingerprint

    def __post_init__(self) -> None:
        for name in ("campaign_id", "endpoint_alias", "capture_uri", "collected_utc"):
            if not isinstance(value := getattr(self, name), str):
                raise ModelingError(f"primitive_provenance_invalid:{name}")
            if name in {"endpoint_alias", "capture_uri"} and _PRIVATE.search(value):
                raise ModelingError("primitive_provenance_secret_leak")
        if type(self.calibrated) is not bool or not isinstance(self.silicon_fingerprint, SiliconFingerprint):
            raise ModelingError("primitive_provenance_invalid")
        if self.calibrated:
            _require_capture(self.capture_uri, "primitive_provenance_capture_invalid")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PrimitiveProvenance:
        raw = _mapping(raw, "primitive_provenance_mapping_required")
        values = {item.name: raw.get(item.name) for item in fields(cls)}
        values["silicon_fingerprint"] = SiliconFingerprint.from_mapping(values["silicon_fingerprint"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {**_record(self), "silicon_fingerprint": self.silicon_fingerprint.to_dict()}


def _is_band(value: Any) -> bool:
    return isinstance(value, UncertaintyBand) or (
        isinstance(value, Mapping) and not {"certified_upper_capacity", "p05", "p50", "p95", "p99", "cv_pct"}.isdisjoint(value)
    )


def _band(metric: str, value: Any) -> UncertaintyBand:
    is_rate = ("_per_s" in metric or "_per_ms" in metric or metric.endswith("_rate"))
    kind = "latency_lower" if (
        metric.endswith(("_ns", "_us", "_ms", "_cycles")) or metric in _LOWER_BOUND_METRICS
    ) and not is_rate else "rate_upper"
    band = value if isinstance(value, UncertaintyBand) else UncertaintyBand.from_mapping(value, bound_kind=kind)
    if is_rate and band.bound_kind != "rate_upper":
        raise ModelingError(f"primitive_metric_rate_upper_required:{metric}")
    return band


def _mode_value(key: str, value: Any) -> Any:
    if _is_band(value):
        return _band(key, value)
    if isinstance(value, Mapping):
        return {name: _mode_value(name, item) for name, item in value.items()}
    return value


def _check_calibrated(values: Mapping[str, Any], primitive_id: str) -> int:
    measured = 0
    for name, value in values.items():
        if isinstance(value, UncertaintyBand):
            has_meas = value.p50 is not None or value.cv_pct is not None or value.n_samples > 0
            if not has_meas:
                continue
            if value.p50 is None or value.cv_pct is None or value.cv_pct > MAX_CALIBRATED_CV_PCT:
                raise ModelingError(f"primitive_metric_calibration_invalid:{primitive_id}:{name}")
            measured += 1
        elif isinstance(value, Mapping):
            measured += _check_calibrated(value, primitive_id)
    return measured


@dataclass(frozen=True, slots=True)
class PrimitiveCell:
    primitive_id: str
    status: CellStatus
    units: Mapping[str, str]
    metrics: Mapping[str, UncertaintyBand]
    modes: Mapping[str, Mapping[str, Any]]
    parameters: Mapping[str, Any]
    evidence_id: str

    def __post_init__(self) -> None:
        if self.primitive_id not in PRIMITIVE_IDS:
            raise ModelingError(f"primitive_id_unknown:{self.primitive_id}")
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise ModelingError(f"primitive_status_invalid:{self.primitive_id}")
        if not isinstance(self.evidence_id, str):
            raise ModelingError(f"primitive_evidence_invalid:{self.primitive_id}")
        units = _mapping(self.units, "primitive_units_mapping_required")
        if any(not isinstance(unit, str) or not unit for unit in units.values()):
            raise ModelingError(f"primitive_units_invalid:{self.primitive_id}")
        metrics = {name: _band(name, value) for name, value in
                   _mapping(self.metrics, "primitive_metrics_mapping_required").items()}
        calibrated_bands = 0
        if self.status == "calibrated":
            _require_capture(self.evidence_id, f"primitive_evidence_invalid:{self.primitive_id}")
            calibrated_bands += _check_calibrated(metrics, self.primitive_id)
        modes = {}
        for name, value in _mapping(self.modes, "primitive_modes_mapping_required").items():
            mode = _mode_value(name, _mapping(value, f"primitive_mode_mapping_required:{name}"))
            status = mode.get("status", self.status)
            if not isinstance(status, str) or status not in _STATUSES:
                raise ModelingError(f"primitive_mode_status_invalid:{self.primitive_id}:{name}")
            if status == "calibrated":
                _require_capture(mode.get("evidence_id", self.evidence_id),
                                 f"primitive_evidence_invalid:{self.primitive_id}:{name}")
                mode_measured = _check_calibrated(mode, self.primitive_id)
                if mode_measured < 1:
                    raise ModelingError(f"primitive_metric_calibration_invalid:{self.primitive_id}:{name}")
                calibrated_bands += mode_measured
            modes[name] = mode
        if self.status == "calibrated" and calibrated_bands < 1:
            raise ModelingError(f"primitive_metric_calibration_invalid:{self.primitive_id}:empty")
        parameters = _mapping(self.parameters, "primitive_parameters_mapping_required")
        for key, value in (("units", units), ("metrics", metrics), ("modes", modes), ("parameters", parameters)):
            object.__setattr__(self, key, _freeze(value))

    @property
    def supported(self) -> bool:
        return self.status != "unsupported"

    def mode_status(self, mode: str | None) -> CellStatus:
        if mode is not None and mode not in self.modes:
            raise ModelingError(f"primitive_mode_unknown:{self.primitive_id}:{mode}")
        return self.modes[mode].get("status", self.status) if mode is not None and self.supported else self.status

    def band(self, metric: str, *, mode: str | None = None) -> UncertaintyBand:
        if mode is not None:
            self.mode_status(mode)
            entry = self.modes.get(mode, {})
            nested = entry.get("metrics", {})
            value = entry.get(metric, nested.get(metric) if isinstance(nested, Mapping) else None)
            if isinstance(value, UncertaintyBand):
                return value
        try:
            return self.metrics[metric]
        except KeyError:
            raise ModelingError(f"primitive_metric_unknown:{self.primitive_id}:{metric}") from None

    @classmethod
    def from_mapping(cls, primitive_id: str, raw: Mapping[str, Any]) -> PrimitiveCell:
        raw = _mapping(raw, f"primitive_cell_mapping_required:{primitive_id}")
        metrics = dict(_mapping(raw.get("metrics", {}), "primitive_metrics_mapping_required"))
        parameters = dict(_mapping(raw.get("parameters", {}), "primitive_parameters_mapping_required"))
        for key, value in raw.items():
            if key not in {"status", "units", "modes", "evidence_id", "metrics", "parameters"}:
                (metrics if _is_band(value) else parameters)[key] = value
        return cls(primitive_id, raw.get("status"), raw.get("units", {}), metrics,
                   raw.get("modes", {}), parameters, raw.get("evidence_id", ""))

    def sim_rate_p50(self, metric: str, *, mode: str | None = None,
                     allow_uncalibrated_declaration: bool = False) -> float:
        return self.band(metric, mode=mode).resolved_sim_rate(
            allow_uncalibrated_declaration=allow_uncalibrated_declaration,
        )

    def certified_upper_capacity(self, metric: str, *, mode: str | None = None) -> float:
        return self.band(metric, mode=mode).certified_upper_capacity

    def to_dict(self) -> dict[str, Any]:
        return {key: _json_value(value) for key, value in _record(self).items()
                if key != "primitive_id"}


def compute_envelope_sha256(payload: Mapping[str, Any]) -> str:
    """Hash all envelope fields other than the envelope's own digest."""
    body = {key: _json_value(value) for key, value in payload.items() if key != "envelope_sha256"}
    try:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ModelingError("primitive_table_json_invalid") from exc
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class PrimitiveTable:
    schema_version: str
    arch_preset: str
    sku: str
    cann_version: str
    envelope_sha256: str
    provenance: PrimitiveProvenance
    primitives: Mapping[str, PrimitiveCell]
    warnings: tuple[str, ...] = ()
    _envelope: Mapping[str, Any] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != PRIMITIVE_SCHEMA_VERSION:
            raise ModelingError("primitive_table_schema_version_invalid")
        if not isinstance(self.envelope_sha256, str) or not _SHA256.fullmatch(self.envelope_sha256):
            raise ModelingError("primitive_table_sha256_invalid")
        for name in ("arch_preset", "sku", "cann_version"):
            if not isinstance(value := getattr(self, name), str) or not value:
                raise ModelingError(f"primitive_table_field_invalid:{name}")
        if not isinstance(self.provenance, PrimitiveProvenance):
            raise ModelingError("primitive_table_provenance_invalid")
        cells = _mapping(self.primitives, "primitive_table_cells_mapping_required")
        if set(cells) != set(PRIMITIVE_IDS):
            raise ModelingError("primitive_table_ids_invalid")
        if any(not isinstance(cell, PrimitiveCell) or cell.primitive_id != key for key, cell in cells.items()):
            raise ModelingError("primitive_table_cell_id_mismatch")
        object.__setattr__(self, "primitives", _freeze(cells))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def cell(self, primitive_id: str) -> PrimitiveCell:
        try:
            return self.primitives[primitive_id]
        except KeyError:
            raise ModelingError(f"primitive_id_unknown:{primitive_id}") from None

    def sim_rate_p50(self, primitive_id: str, metric: str, *, mode: str | None = None,
                     allow_uncalibrated_declaration: bool = False) -> float:
        return self.cell(primitive_id).sim_rate_p50(
            metric, mode=mode, allow_uncalibrated_declaration=allow_uncalibrated_declaration,
        )

    def certified_upper_capacity(self, primitive_id: str, metric: str, *, mode: str | None = None) -> float:
        return self.cell(primitive_id).certified_upper_capacity(metric, mode=mode)

    def to_envelope_dict(self) -> dict[str, Any]:
        if self._envelope is not None:
            return _json_value(self._envelope)
        return {
            "schema_version": self.schema_version, "arch_preset": self.arch_preset,
            "sku": self.sku, "cann_version": self.cann_version, "envelope_sha256": self.envelope_sha256,
            "provenance": self.provenance.to_dict(),
            "primitives": {key: cell.to_dict() for key, cell in self.primitives.items()},
        }

    def canonical_envelope_sha256(self) -> str:
        return compute_envelope_sha256(self.to_envelope_dict())


def parse_primitive_table(raw: Mapping[str, Any], *, verify_envelope: bool = True) -> PrimitiveTable:
    raw = _mapping(raw, "primitive_table_mapping_required")
    if verify_envelope and raw.get("envelope_sha256") != compute_envelope_sha256(raw):
        raise ModelingError("primitive_table_sha256_mismatch")
    cells = _mapping(raw.get("primitives"), "primitive_table_cells_mapping_required")
    table = PrimitiveTable(
        raw.get("schema_version"), raw.get("arch_preset"), raw.get("sku"),
        raw.get("cann_version"), raw.get("envelope_sha256"),
        PrimitiveProvenance.from_mapping(raw.get("provenance")),
        {key: PrimitiveCell.from_mapping(key, cell) for key, cell in cells.items()},
    )
    object.__setattr__(table, "_envelope", _freeze(raw))
    return table


_FRAME_POLICIES = ("exact_plus_header", "pad_full_frame")
ShortFramePolicy = Literal["exact_plus_header", "pad_full_frame"]


def frame_wire_bytes(
    payload_bytes: int, payload_bytes_per_frame: int, wire_bytes_per_frame: int,
    short_final_frame_policy: ShortFramePolicy = "exact_plus_header",
) -> tuple[int, int]:
    """Return exact wire bytes and frame count, sharing production protocol framing."""
    payload, size, wire = payload_bytes, payload_bytes_per_frame, wire_bytes_per_frame
    if any(type(value) is not int for value in (payload, size, wire)) or payload < 0 or not 1 <= size <= wire:
        raise ModelingError("primitive_frame_geometry_invalid")
    if short_final_frame_policy not in _FRAME_POLICIES:
        raise ModelingError("primitive_short_frame_policy_invalid")
    if payload == 0:
        return 0, 0
    if short_final_frame_policy == "exact_plus_header":
        band = CollectiveBandSpec.from_mapping({
            "payload_bytes_per_frame": size, "wire_bytes_per_frame": wire,
            "evidence_id": "frame_wire_bytes",
        })
        frames, wire_total = _protocol_frame_wire_bytes(payload, band)
        return int(round(wire_total)), int(frames)
    frames = (payload + size - 1) // size
    return frames * wire, frames


@dataclass(frozen=True, slots=True)
class _FramedCollectiveBand(CollectiveBandSpec):
    short_final_frame_policy: ShortFramePolicy = "exact_plus_header"

    def __post_init__(self) -> None:
        CollectiveBandSpec.__post_init__(self)
        if self.short_final_frame_policy not in _FRAME_POLICIES:
            raise ModelingError("primitive_short_frame_policy_invalid")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> _FramedCollectiveBand:
        base = CollectiveBandSpec.from_mapping(raw)
        return cls(**base.to_dict(), short_final_frame_policy=raw.get("short_final_frame_policy", "exact_plus_header"))

    def wire_bytes_for(self, payload_bytes: float) -> tuple[int, int]:
        if isinstance(payload_bytes, float) and payload_bytes.is_integer():
            payload_bytes = int(payload_bytes)
        if self.short_final_frame_policy == "exact_plus_header":
            if any(type(v) is not int for v in (payload_bytes, self.payload_bytes_per_frame, self.wire_bytes_per_frame)) or payload_bytes < 0:
                raise ModelingError("primitive_frame_geometry_invalid")
            frames, wire_total = _protocol_frame_wire_bytes(payload_bytes, self)
            return int(round(wire_total)), int(frames)
        return frame_wire_bytes(payload_bytes, self.payload_bytes_per_frame,
                                self.wire_bytes_per_frame, self.short_final_frame_policy)

    def to_dict(self) -> dict[str, Any]:
        return {**CollectiveBandSpec.to_dict(self), "short_final_frame_policy": self.short_final_frame_policy}


def _submit_stars_plan(engine, stars_spec, stars_plan, after, name):
    """Submit a descriptor sequence through production ``StarsDescriptorQueue``."""
    from mindie_sim.prediction.execution.step_helpers import StarsDescriptorQueue

    die, ns, desc_rows = stars_plan
    queue = StarsDescriptorQueue(stars_spec, die_id=die, namespace=ns)
    queue.ensure_engine_resources(engine)
    prev_host, done_events = tuple(after), []
    for occ, (replay_id, stream_id, host_ms, work) in enumerate(desc_rows):
        ready = engine.submit(f"{name}:host_submit:{occ}", after=prev_host, latency=host_ms)
        prev_host = (ready,)
        k_done, _ = queue.submit_kernel(
            engine, replay=replay_id, stream=stream_id, kernel_occurrence=occ,
            descriptor_ready=ready, work=work, owner=name, die=die,
        )
        done_events.append(k_done)
    return done_events[-1] if len(done_events) == 1 else engine.boundary(f"{name}:complete", after=tuple(done_events))


def _scratch_resources(rows, ctx, caps=None):
    from mindie_sim.prediction.runtime.engine import Resource

    role_by_row = {v: k for k, v in getattr(ctx, "resource_names", {}).items()} if ctx is not None else {}
    rates = getattr(ctx, "rates", {}) if ctx is not None else {}
    units_map = getattr(ctx, "resource_units", {}) if ctx is not None else {}
    caps = caps or {}
    out = []
    for row in dict.fromkeys(rows):
        if row in rates:
            role = role_by_row.get(row, row)
            units = units_map.get(row, units_map.get(role, 1))
            out.append(Resource(row, float(rates[row]), units=int(units)))
        else:
            out.append(Resource(row, float(caps.get(row, 1.0))))
    return out


def _stars_plan_span_ms(stars_spec, stars_plan, *, ctx=None) -> float:
    from mindie_sim.prediction.runtime.engine import Engine

    needed = [row for _, _, _, work in stars_plan[2] for row in work]
    scratch = Engine(_scratch_resources(needed, ctx), backend="python")
    done = _submit_stars_plan(scratch, stars_spec, stars_plan, (), "stars")
    scratch.advance(until=(done,))
    return float(done.time)


def _program_span_ms(program: KernelProgram, ctx: Any) -> float:
    """Derive exact isolated span of a nested ``KernelProgram`` via ``Engine``."""
    from mindie_sim.prediction.runtime.engine import Engine

    rows, caps, pools = [], {}, {}
    def _walk(prog):
        for st in prog.stages:
            for row in st.work:
                rows.append(row)
                if row in st.caps:
                    caps[row] = max(caps.get(row, 0.0), float(st.caps[row]))
            for pool, sz in getattr(st, "acquire", ()):
                pools[pool] = pools.get(pool, 0.0) + float(sz)
            if st.program is not None:
                _walk(st.program)
    _walk(program)
    if ctx is not None and getattr(ctx, "pool_names", None) and getattr(ctx, "pool_capacities", None):
        for role, cap in ctx.pool_capacities.items():
            pname = ctx.pool_names.get(role)
            if pname:
                pools[pname] = float(cap)
    scratch = Engine(_scratch_resources(rows, ctx, caps), backend="python", pools=pools or None)
    done = program.start(scratch, "span", scratch.submit("span:start"))
    scratch.advance(until=(done,))
    return float(done.time)


@dataclass(frozen=True, slots=True)
class PrimitiveLoweringResult:
    primitive_id: str
    stages: tuple[Stage, ...]
    program: KernelProgram
    analytical_ms: float
    bound_access: BoundAccess | None = None
    bound_vector: BoundVectorOp | None = None
    ring_buffers: tuple[RingBuffer, ...] = ()
    stars_spec: StarsSpec | None = None
    collective_band: CollectiveBandSpec | None = None
    warnings: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()
    _live_access: tuple[str, MemoryAccessPattern, AdmissionMemoryView | None, HbmBusSpec] | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _stars_plan: tuple[int, str, tuple[tuple[int, str, float, dict[str, float]], ...]] | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _required_resources: tuple[tuple[str, float, str, int], ...] = field(
        default=(), init=False, repr=False, compare=False,
    )

    def _start_lowered(self, engine, after, memory_state, pattern, name_prefix):
        """Submit the lowered DAG with live admission or STARS descriptor queue."""
        from mindie_sim.prediction.engine import Resource
        from mindie_sim.prediction.program.stages import scope_allocations

        name = name_prefix or self.primitive_id
        missing = [
            Resource(r_name, r_rate, r_unit, units=r_units)
            for r_name, r_rate, r_unit, r_units in self._required_resources
            if r_name not in engine.resources
        ]
        if missing:
            engine.extend(missing)
        if self.stars_spec is not None and self._stars_plan is not None:
            return _submit_stars_plan(engine, self.stars_spec, self._stars_plan, after, name)
        if self.bound_access is None:
            return self.program.start(engine, name, engine.boundary(f"{name}:ready", after=after))
        if self._live_access is None:
            return engine.submit_bound_access(name, self.bound_access, after=after,
                                              memory_state=memory_state, pattern=pattern)
        access_name, built_pattern, built_state, bus = self._live_access
        pattern = built_pattern if pattern is None else pattern
        memory_state = built_state if memory_state is None else memory_state
        missing_bus = [Resource(row, 1.0, "progress") for row in (bus.row_r, bus.row_w) if row not in engine.resources]
        if missing_bus:
            engine.extend(missing_bus)
        events, consumed = {}, set()
        for stage in self.stages:
            consumed.update(stage.after)
            ready = tuple(events[key] for key in stage.after) if stage.after else tuple(after)
            stage_name = f"{name}:{stage.name}"
            if stage.name == access_name:
                events[stage.name] = engine.submit_bound_access(
                    stage_name, self.bound_access, after=ready, memory_state=memory_state, pattern=pattern,
                )
            else:
                events[stage.name] = stage.submit(engine, stage_name, ready, lambda items: scope_allocations(name, items), "")
        sinks = tuple(value for key, value in events.items() if key not in consumed)
        return sinks[0] if len(sinks) == 1 else engine.boundary(f"{name}:complete", after=sinks)


class _PrimitiveAddressPlan(AddressRunPlan):
    """Use one DRAM charge per sector across overlapping physical runs."""
    __slots__ = ()

    def dram_sector_and_service(self, burst_rules=()):
        sec = self.sector_bytes
        runs = [run for run in self.runs if run.rows and run.row_bytes]
        bounds = sorted((run.base_byte // sec * sec,
                         align_up(run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes, sec))
                        for run in runs)
        if all(left[1] <= right[0] for left, right in zip(bounds, bounds[1:])):
            return super().dram_sector_and_service(burst_rules)

        def boundaries(run, ordinal):
            intervals = AddressRunPlan((run,), sector_bytes=sec).sector_intervals()
            if not intervals:
                intervals = ((addr // sec * sec, align_up(addr + run.row_bytes, sec))
                             for addr in range(run.base_byte, run.base_byte + run.rows * run.stride_bytes,
                                               run.stride_bytes))
            weight = 1.0 / burst_turnaround_efficiency(run.row_bytes, run.stride_bytes, burst_rules)
            end = None
            for lo, hi in intervals:
                if end is None or lo > end:
                    if end is not None:
                        yield end, ordinal, 0.0
                    yield lo, ordinal, weight
                end = hi
            if end is not None:
                yield end, ordinal, 0.0

        sector_bytes = service = previous = 0
        active = {}
        for address, ordinal, weight in heapq.merge(*(boundaries(run, i) for i, run in enumerate(runs)),
                                                     key=lambda event: event[0]):
            if active:
                sector_bytes += address - previous
                service += (address - previous) * max(active.values())
            if weight:
                active[ordinal] = weight
            else:
                active.pop(ordinal)
            previous = address
        scale = align_up(self.scale_bytes, sec)
        return sector_bytes + scale, service + scale


def _address_plan(values, integer, size=None):
    runs = values("runs")
    if isinstance(runs, AddressRunPlan):
        return _PrimitiveAddressPlan(runs.runs, runs.alignment_bytes, runs.scale_bytes,
                                     runs.sector_bytes, runs.expanded_queries)
    if runs is None:
        row = integer("row_bytes", size if size is not None else values("payload_bytes", values("bytes", 0)))
        runs = (AddressRun(integer("base_byte"), integer("rows", 1), row, integer("stride_bytes", row)),)
    return _PrimitiveAddressPlan(
        tuple(run if isinstance(run, AddressRun) else AddressRun(**run) for run in runs),
        integer("alignment_bytes", DEFAULT_SECTOR_BYTES, 1), integer("scale_bytes"),
        integer("sector_bytes", DEFAULT_SECTOR_BYTES, 1),
    )


__all__ = [
    "PRIMITIVE_SCHEMA_VERSION", "PRIMITIVE_IDS", "MAX_CALIBRATED_CV_PCT", "CellStatus",
    "BoundKind", "UncertaintyBand", "SiliconFingerprint", "PrimitiveProvenance",
    "PrimitiveCell", "PrimitiveTable", "compute_envelope_sha256", "parse_primitive_table",
]
