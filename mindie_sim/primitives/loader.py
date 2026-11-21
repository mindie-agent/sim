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
from pathlib import Path
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import GENERIC_ARCH_ID, DatapathCapabilitySpec
from mindie_sim.schema.hardware_links import UNCALIBRATED_SKU_WARNING

from .schema import PRIMITIVE_SCHEMA_VERSION, PrimitiveCell, PrimitiveTable, compute_envelope_sha256, parse_primitive_table

_MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
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


__all__ = ["RateRole", "default_primitives_dir", "load_manifest", "load_primitives", "resolve_primitive_rate"]
