"""Freeze predictions for a declared geometry-only query manifest.

[Role] Turns a declared query manifest into an immutable ``predictions.frozen.json`` plus a
  ``freeze-receipt.json`` that binds the predictions to the source commit, the package
  hashes, the protocol declarations, the training-dataset hash and the exact manifest.
[Invariants]
  * The manifest is geometry-only: any observed/predicted field is a hard refusal before a
    single product call, so a freeze cannot read an outcome by construction (L2).
  * Refusals are integrity failures (exit 3): a dirty tree unless explicitly allowed, a
    training condition that overlaps a declared heldout key, a query that is not declared
    heldout, a duplicate condition id, a changed input hash during the run.
  * ``evaluation_outcomes_read: false`` and ``outcomes_directory_listed: false`` are
    written by the writer and re-checked by the scorer; the freeze records the claim, the
    scorer is the only process that opens an outcome file.
  * A heldout query is accepted only when the split *declarations* declare its geometry
    heldout. The dataset may not contain the new heldout rows yet (they are measured after
    the freeze), so the declaration is the authority and the dataset is the second witness.
  * An ``operator`` entry evaluates the declared condition through the public product
    model for the same timing scope: ``analytic`` (pure ``PredictiveCostModel``),
    ``physical`` (the fitted physical-calibration artifact; applied only when the
    artifact's own policy says applied) and ``curve`` (``query_curve_artifact`` on the
    declared artifact(s)). A variant whose artifact is absent, has no row for the
    identity, or does not declare a numeric prediction is recorded with that status and
    a null prediction — a value is never invented, and a low-confidence physical number
    is kept as ``physical_prediction_ms`` diagnostics, never promoted to a prediction.
  * Every variant of every entry is recorded, including unsupported / unresolved /
    extrapolated results. A partial freeze (completed != expected) is not scoreable.
[Timeouts & Locks] One sequential bounded call per entry; the git probe is a single
  short-timeout subprocess; no device work and no fitting.
[Side-Effect Policy] Reads the manifest, the dataset and the runtime package; writes only
  --out. Never mutates a measurement dataset, a calibration store or the source tree.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import metrics
from .metrics import (
    FREEZE_SCHEMA, PREDICTIONS_SCHEMA, canonical_json, geometry_key, identity_candidates,
    protocol_core, protocol_signature, sha256_file, sha256_value,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DECLARATIONS = REPO_ROOT / ".workstreams/accuracy-protocol"
DEFAULT_DATASET = REPO_ROOT / ".workstreams/dataset"

# Any of these keys anywhere in the manifest means the caller is trying to filter, score
# or calibrate with a private outcome. The freeze refuses before opening a product API.
OUTCOME_KEYS = {
    "raw_samples_ms", "raw_device_ms_per_op", "observed_ms", "actual_ms", "duration_ms",
    "median_ms", "mean_ms", "latency_ms", "production_result", "results", "trials_data",
    "measured_ms", "actual_median_ms", "predicted_ms", "prediction_ms", "calibration",
    "observations", "raw_record", "raw_records", "heldout_results", "expected_ms",
    "ape_pct", "signed_error_pct", "error_ratio", "score", "scores",
}

VARIANTS = ("analytic", "physical", "curve", "calibrated_db", "db_and_curve")
PRIMARY_POLICIES = ("curve_first", "calibrated_first", "physical_first")

#: Entry kinds this driver can evaluate. ``curve``/``predict``/``operator_cost`` are the
#: round-1 kinds (kept so the round-1 freeze can be re-keyed); ``operator`` is the round-2
#: kind: one declared condition evaluated through the public model, the fitted physical
#: layer and the declared curve artifact(s).
ENTRY_KINDS = ("curve", "predict", "operator_cost", "operator")

#: Cost-model settings a geometry-only manifest may not supply: the driver, never the
#: manifest, decides which artifact is loaded. This keeps a query from smuggling a
#: calibration into a prediction.
FORBIDDEN_COST_SETTINGS = frozenset({
    "calibration", "curve_artifact", "communication_calibration", "cv_calibration",
    "physical_calibration",
})


class IntegrityError(RuntimeError):
    """A freeze precondition was violated; the caller must exit 3."""


# --------------------------------------------------------------------- utils ---
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify_geometry_only(value: Any, path: str = "manifest") -> None:
    """Refuse a manifest that carries an observed or already-predicted field."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in OUTCOME_KEYS:
                raise IntegrityError(f"observed_or_prediction_field_forbidden:{path}.{key}")
            verify_geometry_only(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            verify_geometry_only(item, f"{path}[{index}]")


def _git(repo: Path, *arguments: str, timeout: float = 10.0) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *arguments],
                               capture_output=True, text=True, timeout=timeout,
                               check=False)
    if completed.returncode != 0:
        raise IntegrityError(f"git_probe_failed:{arguments}:{completed.stderr.strip()[:200]}")
    return completed.stdout


def source_identity(repo: Path) -> dict[str, Any]:
    """Commit, dirty status and the hash of every file under the package."""
    commit = _git(repo, "rev-parse", "HEAD").strip()
    porcelain = [line for line in _git(repo, "status", "--porcelain").splitlines() if line.strip()]
    package = repo / "mindie_sim"
    hashes = {}
    if package.is_dir():
        for path in sorted(package.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                hashes[str(path.relative_to(repo))] = sha256_file(path)
    return {
        "commit": commit,
        "dirty": bool(porcelain),
        "dirty_entries": porcelain,
        "package_files": len(hashes),
        "package_file_hashes_sha256": sha256_value(hashes),
        "python_version": sys.version.split()[0],
    }


def protocol_identity(declarations: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {"declaration_files": {}}
    for path in sorted(declarations.glob("*.json")):
        identity["declaration_files"][path.name] = sha256_file(path)
    for name in ("a2-split-declaration.json", "a3-split-declaration.json"):
        path = declarations / name
        if not path.exists():
            raise IntegrityError(f"declaration_missing:{path}")
        identity[name.replace("-", "_").replace(".json", "_sha256")] = sha256_file(path)
    identity["combined_sha256"] = sha256_value(identity["declaration_files"])
    return identity


# ------------------------------------------------------------------- dataset ---
def load_dataset(dataset_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if dataset_path.is_dir():
        observations = dataset_path / "observations.json"
        manifest = dataset_path / "dataset-manifest.json"
    else:
        observations = dataset_path
        manifest = dataset_path.with_name("dataset-manifest.json")
    if not observations.exists():
        raise IntegrityError(f"dataset_missing:{observations}")
    document = load_json(observations)
    if document.get("schema_version") != metrics.OBSERVATION_SCHEMA:
        raise IntegrityError(f"dataset_schema_mismatch:{document.get('schema_version')}")
    side = load_json(manifest) if manifest.exists() else {}
    return document, side


def declared_heldout(dataset: Mapping[str, Any]) -> dict[str, list[str]]:
    heldout: dict[str, list[str]] = {}
    for observation in dataset.get("observations") or []:
        if observation.get("split") != "heldout":
            continue
        heldout.setdefault(observation["hardware_key"], []).append(
            observation["diagnostics"].get("geometry_identity")
            or observation["diagnostics"].get("geometry_key"))
    return heldout


def training_inventory(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the *set of calibratable conditions*, not just a row count (L5/L10)."""
    rows = []
    heldout_rows = []
    for observation in dataset.get("observations") or []:
        row = {
            "geometry_key": observation["diagnostics"].get("geometry_key"),
            "geometry_identity": observation["diagnostics"].get("geometry_identity"),
            "hardware_key": observation["hardware_key"],
            "protocol_signature": observation["protocol_signature"],
            "duration_ms": observation["duration_ms"],
            "source_class": observation["source_class"],
        }
        if observation.get("split") == "train":
            rows.append(row)
        elif observation.get("split") == "heldout":
            heldout_rows.append(row)
    rows.sort(key=lambda item: canonical_json(item))
    heldout_rows.sort(key=lambda item: canonical_json(item))
    return {
        "schema_version": "mindie-sim.training-inventory/1",
        "conditions": len(rows),
        "sha256": sha256_value(rows),
        "heldout_conditions": len(heldout_rows),
        "heldout_sha256": sha256_value(heldout_rows),
        "rows": rows,
    }


def heldout_rows(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for observation in dataset.get("observations") or []:
        if observation.get("split") != "heldout":
            continue
        rows.append({
            "geometry_key": observation["diagnostics"].get("geometry_key"),
            "geometry_identity": observation["diagnostics"].get("geometry_identity")
            or observation["diagnostics"].get("geometry_key"),
            "hardware_key": observation["hardware_key"],
        })
    return rows


def heldout_overlap(train_rows: Sequence[Mapping[str, Any]],
                    heldout_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Train conditions that share a declared heldout geometry — must be empty."""
    index = {(row["hardware_key"], row["geometry_identity"]) for row in heldout_rows}
    return [dict(row) for row in train_rows
            if (row["hardware_key"], row["geometry_identity"]) in index]


# ------------------------------------------------------------------ manifest ---
def normalize_manifest(manifest: Mapping[str, Any], *,
                       default_hardware: str | None = None) -> dict[str, Any]:
    """Validate a geometry-only query manifest and compute its identity keys."""
    verify_geometry_only(manifest)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("manifest_has_no_entries")
    condition_ids = [str(entry.get("condition_id")) for entry in entries]
    duplicates = sorted({cid for cid in condition_ids if condition_ids.count(cid) > 1})
    if duplicates:
        raise IntegrityError(f"duplicate_condition_id:{duplicates[:5]}")
    normalized = []
    manifest_machine = manifest.get("machine")
    manifest_scope = manifest.get("timing_scope") or manifest.get("declared_timing_scope")
    for entry in entries:
        query = entry.get("query") or entry.get("curve_query") or {}
        family = entry.get("family") or query.get("family")
        shape = entry.get("shape") or query.get("shape")
        hardware = entry.get("hardware_key") or query.get("hardware_key") or default_hardware
        if not family or not isinstance(shape, Mapping) or not hardware:
            raise IntegrityError(f"manifest_entry_incomplete:{entry.get('condition_id')}")
        protocol = entry.get("protocol") or query.get("protocol") or {}
        core = protocol_core(protocol)
        kind = str(entry.get("kind") or "curve")
        if kind not in ENTRY_KINDS:
            raise IntegrityError(f"unknown_entry_kind:{kind}")
        query_geometry_key = geometry_key(
            hardware, family, shape,
            recipe=entry.get("recipe") or query.get("recipe"),
            execution_kind=entry.get("execution_kind") or query.get("execution_kind"),
            worker_recipe=entry.get("worker_recipe") or query.get("worker_recipe"),
            precision=entry.get("precision") or query.get("precision"),
            layout=(entry.get("layout") or query.get("layout") or "contiguous"),
            quantization=(entry.get("quantization") or query.get("quantization") or "none"),
            features=entry.get("features") or query.get("features"))
        normalized.append({
            "condition_id": str(entry.get("condition_id")),
            "kind": kind,
            "family": family,
            "shape": {str(k): v for k, v in shape.items()},
            "hardware_key": hardware,
            # A declared point's geometry key is authoritative when it is supplied (the
            # split declarations compute it with every optional identity field left None);
            # the query key that includes recipe/precision is recorded beside it.
            "geometry_key": entry.get("geometry_key") or query_geometry_key,
            "query_geometry_key": query_geometry_key,
            "protocol_core": core,
            "protocol_key": protocol_signature(core),
            "timing_scope": (entry.get("timing_scope") or query.get("timing_scope")
                             or manifest_scope or core.get("timing_scope") or ""),
            "machine": entry.get("machine") or manifest_machine,
            "query": query or {key: entry.get(key) for key in
                               ("family", "shape", "protocol", "precision")},
            "raw_entry": entry,
        })
    return {"manifest_id": manifest.get("manifest_id") or "unnamed",
            "entries": normalized,
            "test_sets": manifest.get("test_sets") or ["test1_old_heldout"],
            "sha256": sha256_value(manifest)}


def check_manifest_against_declarations(entries: Sequence[Mapping[str, Any]],
                                       dataset: Mapping[str, Any],
                                       declarations: Path | None = None) -> dict[str, Any]:
    """Every queried geometry must be a declared heldout geometry of its SKU.

    The split *declaration* is the primary authority (the new heldout rows do not exist in
    the dataset until after the freeze); a declared-heldout row already present in the
    dataset is a second, independent witness. ``undeclared_queries`` lists entries neither
    source declares heldout.
    """
    dataset_heldout: dict[str, set[str]] = {}
    for observation in dataset.get("observations") or []:
        if observation.get("split") != "heldout":
            continue
        identities = observation["diagnostics"].get("geometry_identities") or [
            observation["diagnostics"].get("geometry_identity")]
        dataset_heldout.setdefault(observation["hardware_key"], set()).update(
            identity for identity in identities if identity)
    declared_heldout: dict[str, set[str]] = {}
    declared_geo_keys: dict[str, set[str]] = {}
    declaration_train: dict[str, set[str]] = {}
    if declarations is not None:
        from .assemble_dataset import load_split_declarations
        for declaration in load_split_declarations(Path(declarations)):
            hardware = declaration["hardware_key"]
            for point in declaration["heldout"]:
                if not point.get("shape") or not point.get("family"):
                    continue
                declared_heldout.setdefault(hardware, set()).update(
                    identity_candidates(point["family"], point["shape"]))
                if point.get("geometry_key"):
                    declared_geo_keys.setdefault(hardware, set()).add(point["geometry_key"])
            for point in declaration["train"]:
                if not point.get("shape") or not point.get("family"):
                    continue
                declaration_train.setdefault(hardware, set()).update(
                    identity_candidates(point["family"], point["shape"]))
    undeclared = []
    witnesses: list[dict[str, Any]] = []
    for entry in entries:
        candidates = identity_candidates(entry["family"], entry["shape"])
        in_dataset = any(candidate in dataset_heldout.get(entry["hardware_key"], set())
                         for candidate in candidates)
        in_declaration = (
            any(candidate in declared_heldout.get(entry["hardware_key"], set())
                for candidate in candidates)
            or entry["geometry_key"] in declared_geo_keys.get(entry["hardware_key"], set())
        )
        witnesses.append({"condition_id": entry["condition_id"],
                          "dataset_heldout": in_dataset,
                          "declaration_heldout": in_declaration})
        if not (in_dataset or in_declaration):
            undeclared.append({"condition_id": entry["condition_id"],
                               "geometry_key": entry["geometry_key"],
                               "hardware_key": entry["hardware_key"]})
    return {"declared_heldout_conditions": sum(len(v) for v in dataset_heldout.values()),
            "declaration_heldout_conditions": sum(len(v) for v in declared_heldout.values()),
            "declaration_heldout_geometry_keys": sum(len(v) for v in declared_geo_keys.values()),
            "undeclared_queries": undeclared,
            "query_witnesses": witnesses}


def declaration_overlap(declarations: Path | None) -> dict[str, Any]:
    """Train points that share a declared heldout geometry, straight from the declaration.

    Independent of the assembled dataset: a gap-fill train point that hits a heldout
    identity is caught here even before the new train rows are ingested.
    """
    if declarations is None:
        return {"checked": False, "train_identities": 0, "heldout_identities": 0,
                "overlap": []}
    from .assemble_dataset import load_split_declarations
    overlap = []
    train_total = heldout_total = 0
    for declaration in load_split_declarations(Path(declarations)):
        hardware = declaration["hardware_key"]
        train: set[str] = set()
        heldout: set[str] = set()
        for point in declaration["train"]:
            if point.get("shape") and point.get("family"):
                train.update(identity_candidates(point["family"], point["shape"]))
        for point in declaration["heldout"]:
            if point.get("shape") and point.get("family"):
                heldout.update(identity_candidates(point["family"], point["shape"]))
        train_total += len(train)
        heldout_total += len(heldout)
        for identity in sorted(train & heldout):
            overlap.append({"hardware_key": hardware, "identity": identity})
    return {"checked": True, "train_identities": train_total,
            "heldout_identities": heldout_total, "overlap": overlap}


def artifact_identity(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Hashes and decoded identity of every prediction input, written into the receipt."""
    identity: dict[str, Any] = {
        "timing_scope": str(artifacts.get("timing_scope") or ""),
        "allow_primitive_transfer": bool(artifacts.get("allow_primitive_transfer", False)),
        "physical_calibration": None,
        "curve_artifacts": [],
        "machine": None,
    }
    physical = artifacts.get("physical_calibration")
    if physical:
        identity["physical_calibration"] = {
            "path": str(physical["path"]),
            "sha256": sha256_file(physical["path"]),
            "hardware_key": str(getattr(physical["object"], "hardware_key", "") or ""),
            "schema_version": str(getattr(physical["object"], "schema_version", "") or ""),
            "families": len(dict(getattr(physical["object"], "families", {}) or {})),
        }
    for item in artifacts.get("curve_artifacts") or []:
        identity["curve_artifacts"].append({
            "path": str(item["path"]),
            "sha256": sha256_file(item["path"]),
            "curves": len((item["object"].get("curves") or [])
                          if isinstance(item["object"], Mapping) else ()),
        })
    machine = artifacts.get("machine")
    if machine:
        identity["machine"] = {"sha256": sha256_value(machine),
                               "hardware_key": machine.get("hardware_key"),
                               "arch_family": machine.get("arch_family"),
                               "source": machine.get("source")}
    return identity


def prepare_artifacts(artifacts: Mapping[str, Any], repo: Path | None) -> dict[str, Any]:
    """Load and hash the declared artifacts once, before the first entry is evaluated."""
    prepared = dict(artifacts)
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    physical_path = prepared.get("physical_calibration")
    if physical_path and isinstance(physical_path, (str, Path)):
        from mindie_sim.a3_physical_calibration import PhysicalCalibration
        prepared["physical_calibration"] = {
            "path": Path(physical_path),
            "object": PhysicalCalibration.load(str(physical_path)),
        }
    curve_paths = prepared.get("curve_artifacts") or []
    loaded_curves = []
    for path in curve_paths:
        if isinstance(path, Mapping):
            loaded_curves.append(dict(path))
            continue
        from mindie_sim.a3_operator_curve_forecast import load_curve_artifact
        loaded_curves.append({"path": Path(path),
                              "object": load_curve_artifact(str(path))})
    if loaded_curves:
        prepared["curve_artifacts"] = loaded_curves
    return prepared


# ----------------------------------------------------------------- predictors ---
def _default_predictor(entry: Mapping[str, Any], artifacts: Mapping[str, Any],
                       repo: Path | None) -> dict[str, Any]:
    """Call the public product APIs for one entry; never fits and never scores.

    ``curve`` entries go through ``a3_operator_curve_forecast.query_curve_artifact``;
    ``operator_cost`` entries build the declared machine/op and evaluate the analytic,
    optionally calibrated and optionally curve variants. ``predict`` entries invoke the
    public ``mindie_sim predict`` CLI in-process.
    """
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    kind = entry["kind"]
    raw = entry["raw_entry"]
    if kind == "curve":
        prepared_curves = artifacts.get("curve_artifacts") or []
        if prepared_curves:
            artifact = prepared_curves[0]["object"]
        else:
            artifact_path = artifacts.get("curve_artifact")
            if not artifact_path:
                raise ValueError("curve_entry_requires_curve_artifact")
            from mindie_sim.a3_operator_curve_forecast import load_curve_artifact
            artifact = load_curve_artifact(str(artifact_path))
        from mindie_sim.a3_operator_curve_forecast import query_curve_artifact
        result = query_curve_artifact(
            artifact, entry["query"],
            allow_primitive_transfer=bool(artifacts.get("allow_primitive_transfer",
                                                       False)))
        value = result.get("prediction_ms")
        return {"curve": _variant(value, result.get("status"), result.get("region"),
                                  result.get("lower_bound_ms"), result.get("upper_bound_ms"),
                                  result)}
    if kind == "operator_cost":
        from mindie_sim.a3_calibration_store import SQLiteCalibrationStore
        from mindie_sim.a3_cost import CalibrationStore, PredictiveCostModel
        from mindie_sim.a3_types import MachineSpec
        from mindie_sim.a3_workload import OpSpec
        machine = MachineSpec.from_mapping(raw["machine"])
        op_fields = dict(raw["op"])
        for key in ("requests", "semantic_shape", "local_shape", "padded_shape",
                    "launch_shape", "dependencies"):
            if key in op_fields:
                op_fields[key] = tuple(op_fields[key])
        if isinstance(op_fields.get("metadata"), dict):
            op_fields["metadata"] = tuple(op_fields["metadata"].items())
        op = OpSpec(**op_fields)
        settings = dict(raw.get("cost_settings") or {})
        calibrations = {"analytic": CalibrationStore()}
        if artifacts.get("calibration_db"):
            calibrations["calibrated_db"] = SQLiteCalibrationStore(
                artifacts["calibration_db"]).as_calibration_store(include_heldout=False)
        output = {}
        for name, calibration in calibrations.items():
            model = PredictiveCostModel(machine, calibration=calibration, **settings)
            cost = model.cost(op)
            product = cost.to_dict()
            output[name] = _variant(cost.estimated_ms, product.get("status"),
                                    product.get("region"), product.get("lower_bound_ms"),
                                    product.get("upper_bound_ms"), product)
        return output
    if kind == "predict":
        import contextlib
        import io
        from mindie_sim.__main__ import main as cli_main
        case_path = Path(artifacts["scratch"]) / f"{entry['condition_id']}.case.json"
        write_json(case_path, raw["case"])
        output = {}
        for name in ("analytic", "calibrated_db", "curve", "db_and_curve"):
            flags = []
            if name in {"calibrated_db", "db_and_curve"} and artifacts.get("calibration_db"):
                flags += ["--calibration-db", artifacts["calibration_db"]]
            if name in {"curve", "db_and_curve"} and artifacts.get("curve_artifact"):
                flags += ["--curve-artifact", artifacts["curve_artifact"]]
            if name != "analytic" and not flags:
                continue
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = cli_main(["predict", "--input", str(case_path), *flags])
            payload = json.loads(stream.getvalue())
            if code:
                raise RuntimeError(f"public_cli_failed:{code}:{canonical_json(payload)[:200]}")
            value = payload
            for part in entry["raw_entry"].get("metric_path") or ["duration_ms"]:
                value = value.get(part) if isinstance(value, Mapping) else None
            output[name] = _variant(value, payload.get("status"), payload.get("region"),
                                    payload.get("lower_bound_ms"),
                                    payload.get("upper_bound_ms"), payload)
        return output
    if kind == "operator":
        return _operator_variants(entry, artifacts, repo)
    raise ValueError(f"unknown_entry_kind:{kind}")


def _op_spec(fields: Mapping[str, Any]):
    from mindie_sim.a3_ops import OpSpec
    payload = dict(fields)
    for key in ("requests", "semantic_shape", "local_shape", "padded_shape",
                "launch_shape", "dependencies", "deps"):
        if key in payload:
            payload[key] = tuple(payload[key])
    if isinstance(payload.get("metadata"), dict):
        payload["metadata"] = tuple(payload["metadata"].items())
    return OpSpec(**payload)


def _operator_variants(entry: Mapping[str, Any], artifacts: Mapping[str, Any],
                       repo: Path | None) -> dict[str, Any]:
    """Evaluate one declared condition for analytic / physical / curve variants.

    All three variants share the entry's declared machine, op and timing scope. The
    physical layer is applied only when its own policy says ``applied``; a low-confidence
    or unmodeled artifact answer keeps the fitted number as a diagnostic and returns no
    prediction, so a diagnostic can never be scored as accuracy.
    """
    from mindie_sim.a3_cost import (PHYSICAL_CALIBRATED, CalibrationStore,
                                    PredictiveCostModel)
    from mindie_sim.a3_types import MachineSpec

    raw = entry["raw_entry"]
    machine_mapping = entry.get("machine") or artifacts.get("machine")
    if not isinstance(machine_mapping, Mapping):
        raise ValueError("operator_entry_requires_machine")
    machine = MachineSpec.from_mapping(machine_mapping)
    op_fields = raw.get("op")
    if not isinstance(op_fields, Mapping):
        raise ValueError("operator_entry_requires_op")
    op = _op_spec(op_fields)
    settings = dict(raw.get("cost_settings") or {})
    forbidden = sorted(set(settings) & FORBIDDEN_COST_SETTINGS)
    if forbidden:
        raise ValueError(f"geometry_cost_settings_must_not_embed_artifacts:{forbidden}")
    scope = str(artifacts.get("timing_scope") or entry.get("timing_scope") or "")
    settings["timing_scope"] = scope

    output: dict[str, Any] = {}
    analytic_model = PredictiveCostModel(
        machine, calibration=CalibrationStore(), **settings)
    analytic_cost = analytic_model.cost(op)
    analytic_product = analytic_cost.to_dict()
    analytic_ms = float(analytic_cost.analytic_ms)
    output["analytic"] = {
        **_variant(analytic_ms, analytic_product.get("status") or "ok",
                   analytic_product.get("region"), None, None, analytic_product),
        "analytic_ms": analytic_ms,
        "estimated_ms": float(analytic_cost.estimated_ms),
        "timing_scope": scope,
        "machine_hardware_key": machine.hardware_key,
        "variants_policy": "analytic_calibration_store_empty",
    }

    physical = artifacts.get("physical_calibration")
    if not physical:
        output["physical"] = {
            "status": "no_physical_calibration_configured",
            "prediction_ms": None, "physical_prediction_ms": None,
            "estimated_ms": float(analytic_cost.estimated_ms),
            "applied": False, "primary_eligible": False, "low_confidence": False,
            "region": None, "lower_bound_ms": None, "upper_bound_ms": None,
            "detail": "no artifact supplied to the freeze",
            "product_result_sha256": None, "product_result": None,
            "timing_scope": scope,
        }
    else:
        physical_model = PredictiveCostModel(
            machine, calibration=CalibrationStore(),
            physical_calibration=physical["object"], **settings)
        physical_cost = physical_model.cost(op)
        physical_product = physical_cost.to_dict()
        record = dict(physical_cost.physical or {})
        applied = bool(record.get("applied"))
        status = str(record.get("status") or "physical_not_requested")
        modeled = status in {"physical_calibrated", "physical_low_confidence"}
        output["physical"] = {
            "status": status,
            # The artifact's own duration is the variant value whenever the artifact
            # models this identity. ``applied`` is the product's policy decision
            # (``physical_low_confidence`` is reported and flagged, never silently
            # dropped); ``primary_eligible`` keeps the flagged number out of the
            # primary roll-up unless the policy applied it.
            "prediction_ms": (float(record.get("duration_ms") or physical_cost.estimated_ms)
                              if modeled else None),
            "estimated_ms": float(physical_cost.estimated_ms),
            "physical_prediction_ms": record.get("duration_ms"),
            "applied": applied,
            # ``applied`` is the integration policy's decision; only an applied fitted
            # row is primary-eligible. A modeled-but-not-applied row (for example a
            # copy fit whose rates are unidentified) is still numeric and flagged in the
            # all-variant roll-up, but the analytic/curve path stays primary.
            "primary_eligible": bool(applied),
            "low_confidence": status == "physical_low_confidence",
            "region": None, "lower_bound_ms": None, "upper_bound_ms": None,
            "detail": record.get("detail"),
            "reason": record.get("reason"),
            "artifact_status": record.get("artifact_status"),
            "quality": dict(record.get("parameters") or {}).get("quality"),
            "fit_level": record.get("fit_level"),
            "physical": record,
            "timing_scope": scope,
            "product_result_sha256": sha256_value(physical_product),
            "product_result": physical_product,
        }

    curve_artifacts = list(artifacts.get("curve_artifacts") or [])
    query = entry.get("query") or {}
    if not curve_artifacts:
        output["curve"] = {
            "status": "no_curve_artifact_configured",
            "prediction_ms": None, "region": None,
            "lower_bound_ms": None, "upper_bound_ms": None,
            "product_result_sha256": None, "product_result": None,
            "attempts": [], "timing_scope": scope,
        }
    else:
        from mindie_sim.a3_operator_curve_forecast import query_curve_artifact
        attempts = []
        chosen: Mapping[str, Any] | None = None
        chosen_attempt: dict[str, Any] | None = None
        for item in curve_artifacts:
            result = query_curve_artifact(
                item["object"], query,
                allow_primitive_transfer=bool(
                    artifacts.get("allow_primitive_transfer", False)))
            attempt = {
                "artifact_sha256": sha256_file(item["path"]),
                "status": result.get("status"),
                "prediction_ms": result.get("prediction_ms"),
                "region": result.get("region"),
                "reason": result.get("reason"),
                "match_scope": result.get("match_scope"),
                "matched_curve": result.get("matched_curve"),
                "rejection_diagnostics": result.get("diagnostics"),
                "product_result_sha256": sha256_value(result),
            }
            attempts.append(attempt)
            if chosen is None and result.get("prediction_ms") is not None:
                chosen, chosen_attempt = result, attempt
        if chosen is not None:
            output["curve"] = dict(_variant(
                chosen.get("prediction_ms"), chosen.get("status"),
                chosen.get("region"), chosen.get("lower_bound_ms"),
                chosen.get("upper_bound_ms"), dict(chosen)))
            output["curve"].update({
                "attempts": attempts, "timing_scope": scope,
                "artifact_sha256": chosen_attempt["artifact_sha256"],
                "production_eligible": bool(chosen.get("production_eligible")),
                "production_match": bool(chosen.get("production_match")),
                "match_scope": chosen.get("match_scope"),
                "protocol_source": "declared",
            })
        else:
            reconciled = _curve_attested_protocol_query(
                entry, curve_artifacts, attempts, artifacts)
            if reconciled is not None:
                result, attempt, contract = reconciled
                output["curve"] = dict(_variant(
                    result.get("prediction_ms"), result.get("status"),
                    result.get("region"), result.get("lower_bound_ms"),
                    result.get("upper_bound_ms"), dict(result)))
                output["curve"].update({
                    "attempts": attempts, "timing_scope": scope,
                    "artifact_sha256": attempt["artifact_sha256"],
                    "production_eligible": bool(result.get("production_eligible")),
                    "production_match": bool(result.get("production_match")),
                    "match_scope": result.get("match_scope"),
                    "protocol_source": "attested_series",
                    "protocol_contract": contract,
                })
            else:
                first = attempts[0]
                output["curve"] = {
                    "status": "unsupported",
                    "prediction_ms": None, "region": first.get("region"),
                    "lower_bound_ms": None, "upper_bound_ms": None,
                    "reason": "no_numeric_curve_in_declared_or_attested_contract",
                    "attempts": attempts,
                    "production_eligible": False,
                    "product_result_sha256": first.get("product_result_sha256"),
                    "product_result": None,
                    "timing_scope": scope,
                    "protocol_source": "none",
                }
    return output


#: Protocol fields the declaration treats as machine-checkable identity. The free-text
#: fields (``measurement_scope``/``attention_scope``/``warmup_scope``) describe the same run
#: and are reported as prose differences, exactly as the curve owner's coverage accounting
#: does.
CURVE_CONTRACT_FIELDS = ("timer", "iterations", "trials", "warmup", "preallocated",
                         "cache_policy", "timing_scope")
CURVE_PROSE_FIELDS = ("measurement_scope", "attention_scope", "warmup_scope",
                      "pre_soak_scope", "interval_scope")


def _curve_series_protocols(artifact: Mapping[str, Any], family: str,
                            hardware: str, declared_scope: str,
                            ) -> dict[str, dict[str, Any]]:
    """Attested protocols of the artifact's curves in the *declared* timing series."""
    protocols: dict[str, dict[str, Any]] = {}
    attention_family = family in {"attention_qk", "rope"}
    for curve in artifact.get("curves") or ():
        physical = curve.get("physical") or {}
        if str(physical.get("family")) != family:
            continue
        if hardware and str(physical.get("hardware_key")) != hardware:
            continue
        protocol = dict(curve.get("protocol") or {})
        scope = str(protocol.get("timing_scope") or "")
        if declared_scope and scope != declared_scope:
            if attention_family and "block_interval" in scope:
                pass
            elif "batched" in declared_scope and "batched_amortized" in scope \
                    and "per_call" not in scope:
                pass
            else:
                continue
        protocols[canonical_json(protocol)] = protocol
    return protocols


def _curve_attested_protocol_query(entry: Mapping[str, Any],
                                   curve_artifacts: Sequence[Mapping[str, Any]],
                                   attempts: list[dict[str, Any]],
                                   artifacts: Mapping[str, Any],
                                   ) -> tuple[Mapping[str, Any], dict[str, Any],
                                              dict[str, Any]] | None:
    """Retry a curve query with the artifact's own attested series protocol.

    The declaration's ``measurement_scope``/``attention_scope``/``warmup_scope`` are prose
    descriptions of the same run; the artifact's curve identity stores the attested block.
    A retry happens only when the artifact has exactly **one** attested protocol for this
    family and hardware — two protocols would be a real ambiguity and stay unsupported. The
    declared-versus-attested agreement is recorded, so a contract mismatch (for example the
    A3 rmsnorm runner attests ``preallocated: false``) is visible rather than laundered.
    """
    from mindie_sim.a3_operator_curve_forecast import query_curve_artifact
    family = str(entry["family"])
    hardware = str(entry["hardware_key"])
    declared = dict(entry.get("query", {}).get("protocol") or {})
    declared_scope = str(declared.get("timing_scope") or "")
    for item in curve_artifacts:
        protocols = _curve_series_protocols(
            item["object"], family, hardware, declared_scope)
        if len(protocols) != 1:
            continue
        attested = next(iter(protocols.values()))
        contract = {
            "protocol_source": "attested_series",
            "artifact_sha256": sha256_file(item["path"]),
            "contract_fields": {
                field: {"declared": declared.get(field), "attested": attested.get(field)}
                for field in CURVE_CONTRACT_FIELDS},
            "prose_differences": {
                field: {"declared": declared.get(field), "attested": attested.get(field)}
                for field in CURVE_PROSE_FIELDS
                if declared.get(field) != attested.get(field)},
        }
        contract["contract_mismatches"] = {
            field: value for field, value in contract["contract_fields"].items()
            if value["declared"] is not None and value["declared"] != value["attested"]}
        query = {**dict(entry.get("query") or {}), "protocol": attested}
        result = query_curve_artifact(
            item["object"], query,
            allow_primitive_transfer=bool(artifacts.get("allow_primitive_transfer", False)))
        attempt = {
            "artifact_sha256": sha256_file(item["path"]),
            "protocol_source": "attested_series",
            "status": result.get("status"),
            "prediction_ms": result.get("prediction_ms"),
            "region": result.get("region"),
            "reason": result.get("reason"),
            "match_scope": result.get("match_scope"),
            "matched_curve": result.get("matched_curve"),
            "rejection_diagnostics": result.get("diagnostics"),
            "product_result_sha256": sha256_value(result),
        }
        attempts.append(attempt)
        if result.get("prediction_ms") is not None:
            return result, attempt, contract
    return None


def _variant(value: Any, status: Any, region: Any = None, lower: Any = None,
             upper: Any = None, product: Any = None) -> dict[str, Any]:
    numeric = value is not None
    if numeric:
        value = float(value)
        if not (math.isfinite(value) and value > 0):
            numeric = False
    return {
        "status": status or ("ok" if numeric else "no_numeric_prediction"),
        "prediction_ms": float(value) if numeric else None,
        "region": region,
        "lower_bound_ms": lower,
        "upper_bound_ms": upper,
        "product_result_sha256": sha256_value(product) if product is not None else None,
        "product_result": product,
    }


# -------------------------------------------------------------------- freeze ---
def freeze(*, manifest_path: Path, out: Path, repo: Path, dataset_path: Path,
           declarations: Path, artifacts: Mapping[str, Any] | None = None,
           predictor: Callable[..., dict[str, Any]] | None = None,
           allow_dirty: bool = False, allow_undeclared: bool = False,
           strict_protocol: bool = False,
           import_predictions: Path | None = None) -> dict[str, Any]:
    artifacts = dict(artifacts or {})
    if import_predictions is not None:
        artifacts["import_predictions"] = str(import_predictions)
    source = source_identity(repo)
    if source["dirty"] and not allow_dirty:
        raise IntegrityError(
            "source_tree_dirty:" + ";".join(source["dirty_entries"][:5])
            + " (re-run with --allow-dirty to record the dirty tree explicitly)")
    protocol = protocol_identity(declarations)
    dataset, dataset_side = load_dataset(dataset_path)
    dataset_file_sha = sha256_file(dataset_path if dataset_path.is_file()
                                   else dataset_path / "observations.json")
    dataset_hash = sha256_value(dataset["observations"])
    if dataset_side.get("dataset_sha256") and dataset_side["dataset_sha256"] != dataset_hash:
        raise IntegrityError("dataset_hash_mismatch_with_manifest")

    manifest = normalize_manifest(load_json(manifest_path))
    declared = check_manifest_against_declarations(manifest["entries"], dataset,
                                                   declarations)
    if declared["undeclared_queries"] and not allow_undeclared:
        raise IntegrityError(
            f"query_not_declared_heldout:{declared['undeclared_queries'][:3]}")

    inventory = training_inventory(dataset)
    overlap = heldout_overlap(inventory["rows"], heldout_rows(dataset))
    if overlap:
        raise IntegrityError(f"training_condition_overlaps_heldout:{overlap[:3]}")
    declared_overlap = declaration_overlap(declarations)
    if declared_overlap["overlap"]:
        raise IntegrityError(
            f"declaration_train_condition_overlaps_heldout:{declared_overlap['overlap'][:3]}")

    prepared = prepare_artifacts(artifacts, repo)
    # A top-level manifest machine is the SKU declaration; it is bound into the receipt.
    manifest_machine = None
    if manifest["entries"]:
        manifest_machine = manifest["entries"][0].get("machine")
    if manifest_machine is not None:
        prepared.setdefault("machine", manifest_machine)
    inputs = artifact_identity(prepared)

    frozen_at = utc_now()
    entries = []
    if import_predictions is not None:
        entries = import_frozen_entries(manifest, load_json(import_predictions))
    else:
        predict_fn = predictor or _default_predictor
        prepared.setdefault("scratch", str(out / "scratch"))
        Path(prepared["scratch"]).mkdir(parents=True, exist_ok=True)
        for entry in manifest["entries"]:
            try:
                variants = predict_fn(entry, prepared, repo)
                status = "completed"
            except Exception as exc:  # a failed entry stays visible and blocks the freeze
                variants = {}
                status = "failed"
                entries.append(_entry_record(entry, variants, status,
                                             {"type": type(exc).__name__,
                                              "message": str(exc)[:400]}))
                continue
            entries.append(_entry_record(entry, variants, status))

    completed = sum(1 for entry in entries if entry["status"] == "completed")
    # Stable identity of the prediction *content*: the entries carry no timestamp, so a
    # re-freeze against the reassembled dataset (same manifest, same artifacts, same order)
    # must reproduce this value even though ``frozen_at`` and the dataset block change.
    content_sha256 = sha256_value(entries)
    document = {
        "schema_version": PREDICTIONS_SCHEMA,
        "frozen_at": frozen_at,
        "manifest": {"manifest_id": manifest["manifest_id"],
                     "sha256": manifest["sha256"],
                     "path": str(manifest_path), "entries": len(manifest["entries"])},
        "source": source,
        "protocol": protocol,
        "dataset": {"sha256": dataset_hash, "file_sha256": dataset_file_sha,
                    "path": str(dataset_path),
                    "training_conditions": inventory["conditions"]},
        "inputs": inputs,
        "scope": {"hardware_keys": sorted({entry["hardware_key"]
                                          for entry in manifest["entries"]}),
                  "test_sets": manifest["test_sets"],
                  "declared_heldout_conditions": declared["declared_heldout_conditions"],
                  "declaration_heldout_conditions":
                      declared["declaration_heldout_conditions"],
                  "undeclared_queries": declared["undeclared_queries"],
                  "query_witnesses": declared["query_witnesses"]},
        "declaration_train_heldout_overlap": declared_overlap["overlap"],
        "status": "completed" if completed == len(entries) else "partial",
        "expected_entries": len(entries),
        "completed_entries": completed,
        "predictions_content_sha256": content_sha256,
        "evaluation_outcomes_read": False,
        "outcomes_directory_listed": False,
        "strict_protocol": bool(strict_protocol),
        "automatic_fitting_performed": False,
        "scoring_performed": False,
        "predictions": entries,
    }
    predictions_path = out / "predictions.frozen.json"
    write_json(predictions_path, document)
    receipt = {
        "schema_version": FREEZE_SCHEMA,
        "frozen_at": frozen_at,
        "round_id": artifacts.get("round_id") or manifest["manifest_id"],
        "manifest": document["manifest"],
        "source": source,
        "protocol": protocol,
        "dataset": document["dataset"],
        "training_inventory_sha256": inventory["sha256"],
        "training_inventory_conditions": inventory["conditions"],
        "heldout_conditions": inventory["heldout_conditions"],
        "training_heldout_overlap": 0,
        "declaration_train_heldout_overlap": declared_overlap["overlap"],
        "declaration_train_identities": declared_overlap["train_identities"],
        "declaration_heldout_identities": declared_overlap["heldout_identities"],
        "inputs": inputs,
        "variant_kinds": sorted({name for entry in entries
                                 for name in (entry.get("variants") or {})}),
        "scope": document["scope"],
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "predictions_content_sha256": content_sha256,
        "evaluation_outcomes_read": False,
        "outcomes_directory_listed": False,
        "expected_entries": len(entries),
        "completed_entries": completed,
        "complete": completed == len(entries),
        "strict_protocol": bool(strict_protocol),
        "dirty_tree_recorded": bool(source["dirty"]),
        "imported_from": str(import_predictions) if import_predictions else None,
        "command": list(sys.argv),
    }
    write_json(out / "freeze-receipt.json", receipt)
    return {"document": document, "receipt": receipt,
            "predictions_path": predictions_path,
            "receipt_path": out / "freeze-receipt.json"}


def _entry_record(entry: Mapping[str, Any], variants: Mapping[str, Any], status: str,
                  error: Mapping[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "condition_id": entry["condition_id"],
        "geometry_key": entry["geometry_key"],
        "query_geometry_key": entry.get("query_geometry_key"),
        "protocol_key": entry["protocol_key"],
        "protocol_core": entry["protocol_core"],
        "timing_scope": entry.get("timing_scope"),
        "kind": entry["kind"],
        "family": entry["family"],
        "shape": entry["shape"],
        "hardware_key": entry["hardware_key"],
        "query_sha256": sha256_value(entry["query"]),
        "variants": {name: dict(value) for name, value in variants.items()},
        "status": status,
    }
    for key in ("test_set", "acceptance_bearing", "declaration_test_set",
                "declared_regime", "declared_lower_bound_ms",
                "recomputed_band", "recomputed_lower_bound_ms", "band_agrees",
                "axis", "report_only"):
        value = (entry.get("raw_entry") or {}).get(key)
        if value is not None:
            record[key] = value
    if error:
        record["error"] = dict(error)
    return record


def import_frozen_entries(manifest: Mapping[str, Any],
                          frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Re-key an already frozen prediction document (round-1 format or this schema)."""
    if "predictions" not in frozen:
        raise IntegrityError("import_document_has_no_predictions")
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    for source_entry in frozen["predictions"]:
        by_id.setdefault(str(source_entry.get("condition_id")), []).append(source_entry)
    entries = []
    for entry in manifest["entries"]:
        sources = by_id.get(entry["condition_id"]) or []
        if not sources:
            entries.append(_entry_record(entry, {}, "failed",
                                         {"type": "MissingPrediction",
                                          "message": "condition not present in the imported document"}))
            continue
        variants: dict[str, Any] = {}
        # A round-1 condition is frozen twice (a direct curve query and an operator-cost
        # evaluation). They are one condition, so their variants are merged — with the
        # direct curve applied last, because *that* is the curve prediction the primary
        # chain means (the operator-cost "curve" variant returns analytic on most entries).
        for source_entry in sorted(sources, key=lambda item: item.get("kind") == "curve"):
            if "variants" in source_entry:
                for name, value in source_entry["variants"].items():
                    variants[name] = dict(value)
                continue
            for name, value in (source_entry.get("predictions") or {}).items():
                product = value.get("product_result") or {}
                calibration = product.get("calibration") or {}
                status = (product.get("status") or calibration.get("status")
                          or product.get("calibration_status"))
                variants[name] = _variant(value.get("value"), status,
                                          product.get("region"),
                                          product.get("lower_bound_ms"),
                                          product.get("upper_bound_ms"), product)
        status = ("completed" if all(str(item.get("status") or "completed") == "completed"
                                     for item in sources) else "failed")
        record = _entry_record(entry, variants, status)
        record["kinds"] = sorted({str(item.get("kind")) for item in sources})
        entries.append(record)
    return entries


def rebind_receipt(*, receipt_path: Path, predictions_path: Path, out: Path,
                   repo: Path, dataset_path: Path, declarations: Path) -> dict[str, Any]:
    """Re-bind an existing frozen prediction file to the reassembled dataset.

    The predictor is never called: the prediction bytes, their sha256 and the timestamp-free
    content hash are carried unchanged. This is the only legitimate join between a
    pre-measurement freeze and a dataset that gained heldout rows: scoring must use the
    predictions as frozen, and the scorer requires the receipt's dataset hash to equal the
    scored dataset. The rebind records the old and new dataset hashes, re-verifies the
    declaration hashes and every frozen input artifact, and reports whether the model source
    changed since the freeze (it may have; the predictions cannot).
    """
    receipt = load_json(receipt_path)
    document = load_json(predictions_path)
    if receipt.get("schema_version") != FREEZE_SCHEMA:
        raise IntegrityError(f"receipt_schema_mismatch:{receipt.get('schema_version')}")
    if document.get("schema_version") != PREDICTIONS_SCHEMA:
        raise IntegrityError(f"predictions_schema_mismatch:{document.get('schema_version')}")
    if sha256_file(predictions_path) != receipt.get("predictions_sha256"):
        raise IntegrityError("predictions_changed_after_freeze")
    if document.get("status") != "completed":
        raise IntegrityError(f"freeze_incomplete:{document.get('status')}")
    if document.get("completed_entries") != document.get("expected_entries"):
        raise IntegrityError("freeze_entries_incomplete")
    for flag in ("evaluation_outcomes_read", "outcomes_directory_listed"):
        if document.get(flag) is not False or receipt.get(flag) is not False:
            raise IntegrityError(f"freeze_claims_{flag}")
    content_sha = sha256_value(document.get("predictions") or [])
    if content_sha != receipt.get("predictions_content_sha256"):
        raise IntegrityError("predictions_content_changed_after_freeze")
    dataset, dataset_side = load_dataset(dataset_path)
    dataset_file = dataset_path if dataset_path.is_file() else dataset_path / "observations.json"
    dataset_hash = sha256_value(dataset["observations"])
    dataset_file_sha = sha256_file(dataset_file)
    if dataset_side.get("dataset_sha256") and dataset_side["dataset_sha256"] != dataset_hash:
        raise IntegrityError("dataset_manifest_hash_mismatch")
    inventory = training_inventory(dataset)
    manifest_path = Path(receipt["manifest"]["path"])
    manifest_document = load_json(manifest_path)
    if sha256_value(manifest_document) != receipt["manifest"]["sha256"]:
        raise IntegrityError("manifest_changed_since_freeze")
    normalized = normalize_manifest(manifest_document)
    protocol = protocol_identity(declarations)
    for name, expected in (receipt.get("protocol") or {}).get("declaration_files", {}).items():
        if protocol["declaration_files"].get(name) != expected:
            raise IntegrityError(f"protocol_declaration_changed:{name}")
    declared = check_manifest_against_declarations(
        normalized["entries"], dataset, declarations)
    if declared["undeclared_queries"]:
        raise IntegrityError(f"query_not_declared_heldout:{declared['undeclared_queries'][:3]}")
    inputs = receipt.get("inputs") or {}
    artifact_checks = []
    physical = inputs.get("physical_calibration")
    if physical:
        actual = sha256_file(physical["path"])
        artifact_checks.append({"path": physical["path"], "recorded": physical["sha256"],
                                "actual": actual, "unchanged": actual == physical["sha256"]})
    for item in inputs.get("curve_artifacts") or []:
        actual = sha256_file(item["path"])
        artifact_checks.append({"path": item["path"], "recorded": item["sha256"],
                                "actual": actual, "unchanged": actual == item["sha256"]})
    if any(not check["unchanged"] for check in artifact_checks):
        raise IntegrityError("frozen_input_artifact_changed")
    source = source_identity(repo)
    new_receipt = dict(receipt)
    new_receipt.update({
        "dataset": {"sha256": dataset_hash, "file_sha256": dataset_file_sha,
                    "path": str(dataset_path),
                    "training_conditions": inventory["conditions"]},
        "training_inventory_sha256": inventory["sha256"],
        "training_inventory_conditions": inventory["conditions"],
        "heldout_conditions": inventory["heldout_conditions"],
        "scope": {**receipt.get("scope", {}),
                  "declared_heldout_conditions": declared["declared_heldout_conditions"],
                  "declaration_heldout_conditions":
                      declared["declaration_heldout_conditions"],
                  "undeclared_queries": declared["undeclared_queries"],
                  "query_witnesses": declared["query_witnesses"]},
        "rebind": {
            "rebound_at": utc_now(),
            "from_receipt": str(receipt_path),
            "from_receipt_sha256": sha256_file(receipt_path),
            "dataset_at_freeze": (receipt.get("dataset") or {}).get("sha256"),
            "dataset_at_rebind": dataset_hash,
            "predictions_recomputed": False,
            "predictions_content_sha256": content_sha,
            "frozen_input_artifacts": artifact_checks,
            "source_changed_since_freeze": (
                source["package_file_hashes_sha256"]
                != (receipt.get("source") or {}).get("package_file_hashes_sha256")),
            "current_package_file_hashes_sha256": source["package_file_hashes_sha256"],
            "current_source": {"commit": source["commit"], "dirty": source["dirty"]},
        },
        "command": list(sys.argv),
    })
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "freeze-receipt.json", new_receipt)
    return {"receipt": new_receipt, "receipt_path": out / "freeze-receipt.json"}


# ----------------------------------------------------------------------- cli ---
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--declarations", type=Path, default=DEFAULT_DECLARATIONS)
    parser.add_argument("--curve-artifact", type=Path, action="append", default=None,
                        help="declared train-only curve artifact (repeatable; each entry "
                             "queries the declared artifacts in order, and a numeric result "
                             "from any of them is recorded with its artifact hash)")
    parser.add_argument("--physical-calibration", type=Path,
                        help="fitted physical-calibration artifact for operator entries")
    parser.add_argument("--timing-scope",
                        help="the declared measurement timing scope (e.g. "
                             "device_event_warm_batched); it overrides the manifest's scope")
    parser.add_argument("--allow-primitive-transfer", action="store_true",
                        help="forward the product's primitive-transfer opt-in to the curve "
                             "query (off by default, as the production gate is)")
    parser.add_argument("--calibration-db", type=Path)
    parser.add_argument("--import-predictions", type=Path,
                        help="re-key an existing frozen prediction document instead of "
                             "calling the product APIs (validation and re-freeze path)")
    parser.add_argument("--round-id")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="record a dirty source tree explicitly instead of refusing")
    parser.add_argument("--allow-undeclared", action="store_true",
                        help="allow queries that are not declared heldout (recorded, scored "
                             "as excluded)")
    parser.add_argument("--strict-protocol", action="store_true")
    parser.add_argument("--rebind-receipt", type=Path,
                        help="re-bind an existing frozen prediction file to the reassembled "
                             "dataset without calling the model (see rebind_receipt)")
    parser.add_argument("--rebind-predictions", type=Path,
                        help="the frozen predictions file the rebound receipt points at")
    args = parser.parse_args(argv)
    if args.rebind_receipt:
        if not args.rebind_predictions:
            parser.error("--rebind-predictions is required with --rebind-receipt")
        try:
            result = rebind_receipt(
                receipt_path=args.rebind_receipt,
                predictions_path=args.rebind_predictions,
                out=args.out, repo=args.repo, dataset_path=args.dataset,
                declarations=args.declarations)
        except IntegrityError as exc:
            print(f"integrity: {exc}", file=sys.stderr)
            return 3
        receipt = result["receipt"]
        print(json.dumps({"status": "rebound",
                          "predictions_sha256": receipt["predictions_sha256"],
                          "predictions_content_sha256":
                              receipt["predictions_content_sha256"],
                          "dataset": receipt["dataset"]["sha256"],
                          "source_changed_since_freeze":
                              receipt["rebind"]["source_changed_since_freeze"],
                          "receipt": str(result["receipt_path"])}, indent=2))
        return 0
    if not args.manifest:
        parser.error("--manifest is required unless --rebind-receipt is used")

    artifacts: dict[str, Any] = {"round_id": args.round_id}
    if args.curve_artifact:
        artifacts["curve_artifact"] = str(args.curve_artifact[0])
        artifacts["curve_artifacts"] = [str(path) for path in args.curve_artifact]
    if args.physical_calibration:
        artifacts["physical_calibration"] = str(args.physical_calibration)
    if args.timing_scope:
        artifacts["timing_scope"] = str(args.timing_scope)
    if args.allow_primitive_transfer:
        artifacts["allow_primitive_transfer"] = True
    if args.calibration_db:
        artifacts["calibration_db"] = str(args.calibration_db)
    try:
        result = freeze(manifest_path=args.manifest, out=args.out, repo=args.repo,
                        dataset_path=args.dataset, declarations=args.declarations,
                        artifacts=artifacts, allow_dirty=args.allow_dirty,
                        allow_undeclared=args.allow_undeclared,
                        strict_protocol=args.strict_protocol,
                        import_predictions=args.import_predictions)
    except IntegrityError as exc:
        print(f"integrity: {exc}", file=sys.stderr)
        return 3
    receipt = result["receipt"]
    print(json.dumps({"status": "frozen" if receipt["complete"] else "partial",
                      "entries": receipt["expected_entries"],
                      "completed": receipt["completed_entries"],
                      "dirty_tree_recorded": receipt["dirty_tree_recorded"],
                      "predictions_sha256": receipt["predictions_sha256"],
                      "predictions": str(result["predictions_path"]),
                      "receipt": str(result["receipt_path"])}, indent=2))
    return 0 if receipt["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
