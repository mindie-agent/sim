"""Assemble one unified observation dataset from every measured A3/A2 artifact.

[Role] Scans the declared measurement artifacts (through the readers in
  :mod:`tools.accuracy.sources`), normalises every measured condition to
  ``mindie-sim.observations/1``, dedupes repeated runs of one condition, enforces the
  declared train/heldout split, and publishes the counts, exclusions and alias groups.
[Invariants]
  * Scored target = median of per-trial medians. The pooled-event median is carried as a
    diagnostic and its difference is reported, never substituted (metrics.md §1.1).
  * Row identity includes the physical die and the full protocol signature: the same
    condition under two protocols, or on two dies, is two rows and is never averaged.
  * Exactly identical measurements of one row (same run, same samples) are collapsed and
    counted; repeated runs are merged into ``runs`` with every raw sample retained.
  * A condition whose family-aware shape matches *any* declared heldout key is heldout —
    including a row the campaign labelled train. Those relabels are published one by one.
  * No SKU constant is hardcoded: hardware keys, arch family, work models and band edges
    all come from the artifacts and the declarations.
[Timeouts & Locks] One bounded read-only pass; no device, network or subprocess work.
[Side-Effect Policy] Reads the data roots and the split declarations; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import metrics
from .metrics import (
    OBSERVATION_SCHEMA, canonical_json, condition_key, geometry_key, median,
    pooled_median_ms, protocol_core, protocol_signature, scored_target_ms,
    shape_identity, sha256_file, sha256_value,
)
from .metrics import FAMILY_ALIASES, family_class as _family_class, identity_candidates
from .sources import (
    MEASURED_STATUSES, READERS, ROUND1_CAMPAIGNS, SOURCES, _clean_trials, _die_for,
    _ints, _is_synthetic, _load_json, _number, _split_hint, discover_round2_sources,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DATA_ROOT = Path("/Volumes/data/mindie-sim-data/a3/2026-10-20")
DEFAULT_A2_ROOT = Path("/Volumes/data/mindie-sim-data/a2/2026-10-21")
DEFAULT_A3_ROUND2_ROOT = Path("/Volumes/data/mindie-sim-data/a3/2026-10-20/accuracy-round-20261021")
DEFAULT_DECLARATIONS = REPO_ROOT / ".workstreams/accuracy-protocol"

# --------------------------------------------------------------- declaration ---
def load_split_declarations(directory: Path) -> list[dict[str, Any]]:
    """Read every ``*-split-declaration.json``; each yields heldout + train points."""
    declarations = []
    for path in sorted(directory.glob("*-split-declaration.json")):
        document = _load_json(path)
        sku = document.get("sku") or {}
        hardware = sku.get("hardware_key")
        arch = sku.get("arch_family")
        heldout: list[dict[str, Any]] = []
        train: list[dict[str, Any]] = []
        test_sets = document.get("test_sets") or {}
        fresh = document.get("new_heldout") or {}
        for point in fresh.get("points") or []:
            heldout.append({**point, "test_set": fresh.get("test_set") or "test2_fresh_heldout"})
        for family_block in document.get("families") or []:
            for point in family_block.get("heldout") or []:
                heldout.append({**point, "test_set": "test2_fresh_heldout",
                                "acceptance_bearing": family_block.get("acceptance_bearing")})
            for point in family_block.get("train") or []:
                train.append(point)
        for point in (document.get("gap_fill_train") or {}).get("points") or []:
            train.append({**point, "test_set": "gap_fill_train"})
        declarations.append({
            "path": str(path), "sha256": sha256_file(path),
            "declaration_id": document.get("declaration_id"),
            "hardware_key": hardware, "arch_family": arch,
            "heldout": heldout, "train": train,
            "test_sets": test_sets,
            "regime_rule": document.get("regime_rule") or {},
            "protocol_identity": document.get("protocol_identity") or {},
            "raw": document,
        })
    return declarations


def load_round1_heldout(evidence_dir: Path, hardware_key: str | None) -> list[dict[str, Any]]:
    """The 159 round-1 heldout conditions, from the declared geometry-only query manifests."""
    points = []
    if not evidence_dir.is_dir():
        return points
    for campaign in ROUND1_CAMPAIGNS:
        path = evidence_dir / "geometry" / f"{campaign}-public-queries.json"
        if not path.exists():
            continue
        document = _load_json(path)
        for entry in document.get("entries") or []:
            if entry.get("kind") != "curve":
                continue
            query = entry.get("query") or {}
            points.append({
                "point_id": entry.get("condition_id"), "family": query.get("family"),
                "shape": query.get("shape"), "hardware_key": query.get("hardware_key") or hardware_key,
                "test_set": "test1_old_heldout", "campaign": campaign,
                "regime": None, "geometry_key": None, "acceptance_bearing": None,
                "protocol": query.get("protocol") or {},
                "recipe": query.get("recipe"), "worker_recipe": query.get("worker_recipe"),
                "source": str(path), "reason": "round1_declared_heldout",
            })
    return points


def build_heldout_index(points: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        hardware = point.get("hardware_key")
        family = point.get("family")
        shape = point.get("shape")
        if not hardware or not family or not isinstance(shape, Mapping):
            continue
        try:
            identity = shape_identity(_family_class(family), shape, dtype=point.get("dtype"))
        except KeyError:
            continue
        index[(str(hardware), identity)].append(dict(point))
    return dict(index)


# --------------------------------------------------------------- normalise ----
def normalize_record(ir: Mapping[str, Any], source: Mapping[str, Any],
                     work_model: metrics.WorkModel, *, strict_protocol: bool = False
                     ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """IR -> (observation | exclusion). Exclusion is a record, never a silent drop."""
    exclusion = {"record_id": ir.get("record_id"), "source": source.get("id"),
                 "artifact": ir.get("artifact"), "family": ir.get("family"),
                 "shape": ir.get("shape"), "reason": None, "detail": None}
    status = str(ir.get("status") or "").lower()
    if status in {"derived_census"}:
        return None, None
    if status not in MEASURED_STATUSES:
        exclusion.update(reason="not_measured", detail=f"status={status}")
        return None, exclusion
    synthetic = _is_synthetic(ir)
    if synthetic:
        exclusion.update(reason="not_a_device_measurement", detail=synthetic)
        return None, exclusion
    if "not_a3_measured" in (ir.get("flags") or []):
        exclusion.update(reason="not_a3_measured",
                         detail="artifact declares not_a3_measured")
        return None, exclusion
    if ir.get("source_class") not in {"measured", "observed"}:
        exclusion.update(reason="not_calibratable_source_class",
                         detail=str(ir.get("source_class")))
        return None, exclusion
    if not ir.get("shape"):
        exclusion.update(reason="no_shape", detail="artifact does not attest a physical shape")
        return None, exclusion
    if not ir.get("family"):
        exclusion.update(reason="no_family", detail="artifact does not attest an op family")
        return None, exclusion
    trials = _clean_trials(ir.get("trials") or [])
    if not trials:
        exclusion.update(reason="no_usable_samples", detail="no positive finite device samples")
        return None, exclusion
    try:
        duration, per_trial = scored_target_ms(trials)
    except ValueError:
        exclusion.update(reason="no_usable_samples", detail="empty trial group")
        return None, exclusion
    pooled = pooled_median_ms(trials)
    try:
        core = protocol_core(ir.get("protocol_raw") or {},
                             defaults=ir.get("defaults") or {},
                             record_level=ir.get("record_level") or {},
                             strict=strict_protocol)
    except (TypeError, ValueError) as exc:
        exclusion.update(reason="protocol_unreadable", detail=str(exc))
        return None, exclusion
    signature = protocol_signature(core, strict=strict_protocol)
    shape = _ints(ir["shape"])
    family = ir["family"]
    hardware = ir.get("hardware_key")
    if not hardware:
        exclusion.update(reason="no_hardware_key", detail="artifact does not attest the SKU")
        return None, exclusion
    try:
        demand = work_model.demand(family, shape)
    except metrics.WorkModelError as exc:
        demand = {}
        flag = f"no_work_model:{exc}"
    else:
        flag = None
    key = condition_key(hardware, ir.get("die") or "unknown", family, shape,
                        ir.get("recipe"), signature,
                        precision=ir.get("precision"), layout=ir.get("layout"),
                        quantization=ir.get("quantization"),
                        features=ir.get("identity_extra") or {},
                        execution_kind=ir.get("execution_kind"),
                        worker_recipe=ir.get("worker_recipe"),
                        amortization=core.get("amortization"))
    flags = list(ir.get("flags") or [])
    if flag:
        flags.append(flag)
    observation = {
        "condition_key": key,
        "hardware_key": hardware,
        "die": ir.get("die") or "unknown",
        "arch_family": ir.get("arch_family"),
        "family": family,
        "recipe": ir.get("recipe"),
        "protocol_signature": signature,
        "run_id": ir.get("run_id"),
        "split": None,
        "shape": shape,
        "elements": int(shape["elements"]) if "elements" in shape else None,
        "flops": int(round(demand["flops"])) if demand.get("flops") is not None else None,
        "hbm_bytes": int(round(demand["bytes"])) if demand.get("bytes") is not None else None,
        "duration_ms": duration,
        "raw_samples_ms": [value for trial in trials for value in trial],
        "source_class": ir.get("source_class"),
        "timing_scope": core.get("timing_scope") if core.get("timing_scope") != metrics.UNKNOWN else None,
        "runs": [],
        "diagnostics": {
            "geometry_key": geometry_key(hardware, family, shape),
            "geometry_identity": None,   # filled after the split index exists
            "family_class": _family_class(family),
            "pooled_median_ms": pooled,
            "pooled_minus_trial_median_ms": (None if pooled is None else pooled - duration),
            "trial_medians_ms": per_trial,
            "trials": len(trials),
            "raw_events": len(observation_samples(trials)),
            "estimator": "median_of_per_trial_medians",
            "single_trial": len(trials) == 1,
            "protocol": core,
            "protocol_origin": metrics.protocol_origins(
                ir.get("protocol_raw") or {}, record_level=ir.get("record_level") or {},
                defaults=ir.get("defaults") or {}, strict=strict_protocol),
            "die_attested": bool(ir.get("die_attested")),
            "die_evidence": ir.get("die_evidence"),
            "precision": ir.get("precision"),
            "layout": ir.get("layout"),
            "quantization": ir.get("quantization"),
            "features": ir.get("identity_extra") or {},
            "execution_kind": ir.get("execution_kind"),
            "worker_recipe": ir.get("worker_recipe"),
            "backend": ir.get("backend"),
            "analytic_ms_artifact": ir.get("analytic_ms"),
            "source_artifact": ir.get("artifact"),
            "source_sha256": ir.get("artifact_sha256"),
            "source_id": source.get("id"),
            "record_id": ir.get("record_id"),
            "split_origin": ir.get("split_origin"),
            "split_hint": ir.get("split_hint"),
            "declared_heldout": False,
            "heldout_point_id": None,
            "heldout_test_set": None,
            "flags": flags,
            "alias_group": None,
            "alias_members": [],
        },
    }
    return observation, None


def observation_samples(trials: Sequence[Sequence[float]]) -> list[float]:
    return [value for trial in trials for value in trial]


# ----------------------------------------------------------------- assembly ---
def _source_files(source: Mapping[str, Any], data_root: Path) -> list[Path]:
    return [data_root / name for name in source.get("files") or []]


def scan_sources(data_root: Path, sources: Sequence[Mapping[str, Any]],
                 arch_default: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (records, source_census, missing)."""
    records: list[dict[str, Any]] = []
    census: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for source in sources:
        entry = {"id": source.get("id"), "role": source.get("role"),
                 "arch_family": source.get("arch_family") or arch_default,
                 "files": [], "records": 0, "status": "ok"}
        reader = READERS.get(str(source.get("reader")))
        if reader is None:
            entry["status"] = "no_reader"
            census.append(entry)
            continue
        for path in _source_files(source, data_root):
            if not path.exists():
                entry["file_status"] = "missing"
                missing.append({"source": source.get("id"), "path": str(path)})
                continue
            digest = sha256_file(path)
            entry["files"].append({"path": str(path), "sha256": digest,
                                   "bytes": path.stat().st_size})
            ingest = source.get("ingest", source.get("role") == "raw")
            if not ingest:
                try:
                    produced = reader(path, source)
                except (json.JSONDecodeError, sqlite3.Error, OSError, KeyError) as exc:
                    entry.setdefault("errors", []).append(f"{path.name}:{type(exc).__name__}:{exc}")
                    continue
                entry["records"] += len(produced)
                entry["ingested"] = False
                continue
            try:
                produced = reader(path, source)
            except (json.JSONDecodeError, sqlite3.Error, OSError, KeyError) as exc:
                entry.setdefault("errors", []).append(f"{path.name}:{type(exc).__name__}:{exc}")
                entry["status"] = "read_error"
                continue
            for record in produced:
                record["artifact"] = str(path)
                record["artifact_sha256"] = digest
                record["_source_id"] = source.get("id")
                records.append(record)
            entry["records"] += len(produced)
        census.append(entry)
    return records, census, missing


def discover_round2_dirs(roots: Sequence[tuple[str, str, Path]]) -> dict[str, Any]:
    """Enumerate the round-2 campaign roots; classify every JSON/SQLite file found."""
    discovery: dict[str, Any] = {}
    for label, arch, root in roots:
        entry: dict[str, Any] = {"path": str(root), "arch_family": arch, "exists": root.is_dir()}
        if root.is_dir():
            files = []
            for path in sorted(root.rglob("*")):
                if path.is_dir() or path.name.startswith("."):
                    continue
                if path.suffix not in {".json", ".jsonl", ".sqlite", ".csv", ".md"}:
                    continue
                classification = "unclassified"
                if path.suffix == ".json":
                    try:
                        document = json.loads(path.read_text())
                    except (json.JSONDecodeError, OSError):
                        classification = "unreadable_json"
                    else:
                        if isinstance(document, dict):
                            if isinstance(document.get("results"), list):
                                classification = "campaign_results"
                            elif isinstance(document.get("cases"), list):
                                first = document["cases"][0] if document["cases"] else {}
                                classification = ("case_trials" if isinstance(
                                    first.get("trials"), list) else "root_cases")
                            elif "observations" in document:
                                classification = "assembled_observations"
                            elif "predictions" in document:
                                classification = "frozen_predictions"
                elif path.suffix == ".sqlite":
                    classification = "sqlite_store"
                elif path.suffix == ".jsonl":
                    classification = "jsonl_mirror"
                elif path.suffix == ".md":
                    classification = "doc"
                files.append({"path": str(path), "bytes": path.stat().st_size,
                              "classification": classification})
            entry["files"] = files
            entry["counts"] = dict(Counter(item["classification"] for item in files))
        discovery[label] = entry
    return discovery


def assemble(*, data_root: Path, a2_root: Path, a3_round2_root: Path,
             declarations_dir: Path, round1_evidence: Path, strict_protocol: bool,
             keep_excluded: bool, extra_sources: Sequence[Mapping[str, Any]] = ()
             ) -> dict[str, Any]:
    declarations = load_split_declarations(declarations_dir)
    declared_points: list[dict[str, Any]] = []
    for declaration in declarations:
        for point in declaration["heldout"]:
            declared_points.append({**point, "hardware_key": declaration["hardware_key"],
                                    "declaration": declaration["declaration_id"]})
    a3_hardware = next((item["hardware_key"] for item in declarations
                        if item["arch_family"] == "A3"), None)
    round1 = load_round1_heldout(round1_evidence, a3_hardware)
    for point in round1:
        point["declaration"] = "round1:independent-generalization-20261021"
    heldout_index = build_heldout_index(list(declared_points) + list(round1))

    regime_rule = next((item["regime_rule"] for item in declarations
                        if item["regime_rule"]), {})
    work_model = metrics.WorkModel(regime_rule) if regime_rule else None
    if work_model is None:
        raise SystemExit("integrity: no split declaration carries a regime_rule work model")

    sources = [dict(source) for source in SOURCES] + [dict(item) for item in extra_sources]
    records, census, missing = scan_sources(data_root, sources, "A3")
    # The round-2 roots are written by other workstreams while this tool runs, so their
    # files are discovered and ingested with the same reader instead of being pre-listed.
    for arch, root in (("A2", a2_root), ("A3", a3_round2_root)):
        dynamic = discover_round2_sources(Path(root), arch)
        if not dynamic:
            continue
        round_records, round_census, round_missing = scan_sources(Path(root), dynamic, arch)
        records.extend(round_records)
        census.extend(round_census)
        missing.extend(round_missing)
        sources.extend(dynamic)
    sources_by_id = {source["id"]: source for source in sources}

    observations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        source = sources_by_id.get(record.get("_source_id") or "") or {}
        observation, exclusion = normalize_record(record, source, work_model,
                                                 strict_protocol=strict_protocol)
        if exclusion:
            exclusions.append(exclusion)
        if observation is None:
            continue
        identity_family = _family_class(observation["family"])
        precision = observation["diagnostics"].get("precision")
        dtype = precision.get("input_dtype") if isinstance(precision, Mapping) else None
        identities: list[str] = []
        for candidate in (identity_family, *FAMILY_ALIASES.get(identity_family, ())):
            try:
                identities.append(shape_identity(candidate, observation["shape"], dtype=dtype))
            except KeyError as exc:
                exclusions.append({"record_id": record.get("record_id"),
                                   "source": record.get("_source_id"),
                                   "family": observation["family"],
                                   "shape": observation["shape"],
                                   "reason": "shape_identity_failed", "detail": str(exc)})
                identities = []
                break
        if not identities:
            continue
        identity = identities[0]
        observation["diagnostics"]["geometry_identity"] = identity
        observation["diagnostics"]["geometry_identities"] = identities
        matches = []
        for candidate in identities:
            matches.extend(heldout_index.get((observation["hardware_key"], candidate)) or [])
        campaign_heldout = record.get("split_hint") == "heldout"
        if matches:
            primary = matches[0]
            observation["split"] = "heldout"
            observation["diagnostics"].update({
                "declared_heldout": True,
                "heldout_point_id": primary.get("point_id"),
                "heldout_test_set": primary.get("test_set"),
                "heldout_declaration": primary.get("declaration"),
                "heldout_matches": len(matches),
            })
            if campaign_heldout is False:
                observation["diagnostics"]["flags"].append("train_row_relabelled_heldout")
        elif campaign_heldout:
            observation["split"] = "heldout"
            observation["diagnostics"].update({
                "declared_heldout": False,
                "heldout_point_id": None,
                "heldout_test_set": "campaign_local_holdout",
                "flags": [*observation["diagnostics"]["flags"],
                          "campaign_holdout_not_declared"],
            })
        else:
            observation["split"] = "train"
        observations.append(observation)

    # ---- merge exactly identical conditions, retaining every run and sample ----
    merged: dict[str, dict[str, Any]] = {}
    merge_stats = {"records": len(observations), "distinct_conditions": 0,
                   "merged_rows": 0, "exact_duplicate_rows": 0,
                   "same_run_conflicting_samples": 0}
    for observation in observations:
        key = observation["condition_key"]
        samples_hash = sha256_value([round(value, 12) for value in observation["raw_samples_ms"]])
        run = {
            "run_id": observation["run_id"],
            "trials": observation["diagnostics"]["trials"],
            "trial_medians_ms": observation["diagnostics"]["trial_medians_ms"],
            "raw_events": observation["diagnostics"]["raw_events"],
            "raw_samples_sha256": samples_hash,
            "pooled_median_ms": observation["diagnostics"]["pooled_median_ms"],
            "source_artifact": observation["diagnostics"]["source_artifact"],
            "source_sha256": observation["diagnostics"]["source_sha256"],
            "source_id": observation["diagnostics"]["source_id"],
            "record_id": observation["diagnostics"]["record_id"],
        }
        existing = merged.get(key)
        if existing is None:
            observation["runs"] = [run]
            merged[key] = observation
            continue
        same_record = [item for item in existing["runs"]
                       if item["record_id"] == run["record_id"]
                       and item["run_id"] == run["run_id"]]
        if same_record and same_record[0]["raw_samples_sha256"] == run["raw_samples_sha256"]:
            merge_stats["exact_duplicate_rows"] += 1
            continue
        if same_record:
            # One record id, two different sample streams: an acquisition inconsistency
            # that must stay visible instead of being averaged away.
            merge_stats["same_run_conflicting_samples"] += 1
            existing["diagnostics"]["flags"].append("same_record_conflicting_samples")
        elif any(item["run_id"] == run["run_id"] for item in existing["runs"]):
            # Same run, different record: a correlated stream of one condition (e.g. the
            # two ranks of a collective). Retained, and labelled as correlated.
            merge_stats["correlated_stream_rows"] = merge_stats.get("correlated_stream_rows", 0) + 1
            existing["diagnostics"]["flags"].append("correlated_streams_merged")
        existing["runs"].append(run)
        existing["raw_samples_ms"].extend(observation["raw_samples_ms"])
        trials = existing["diagnostics"]["trials"] + observation["diagnostics"]["trials"]
        existing["diagnostics"]["trials"] = trials
        existing["diagnostics"]["trial_medians_ms"].extend(
            observation["diagnostics"]["trial_medians_ms"])
        existing["diagnostics"]["single_trial"] = trials == 1
        existing["duration_ms"] = median(existing["diagnostics"]["trial_medians_ms"])
        existing["diagnostics"]["pooled_median_ms"] = pooled_median_ms([existing["raw_samples_ms"]])
        existing["diagnostics"]["raw_events"] = len(existing["raw_samples_ms"])
        existing["diagnostics"]["pooled_minus_trial_median_ms"] = (
            existing["diagnostics"]["pooled_median_ms"] - existing["duration_ms"])
        existing["diagnostics"]["estimator"] = "median_of_per_trial_medians_over_merged_runs"
        merge_stats["merged_rows"] += 1
    merge_stats["distinct_conditions"] = len(merged)

    # ---- alias groups: one payload, several factorisations ----
    alias_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in merged.values():
        diagnostics = observation["diagnostics"]
        if _family_class(observation["family"]) != "copy":
            continue
        payload = diagnostics["geometry_identity"].split(":", 1)[-1]
        alias_index[metrics.alias_group_key(
            observation["hardware_key"], observation["die"],
            _family_class(observation["family"]), observation["protocol_signature"],
            payload)].append(observation)
    alias_groups = []
    for group_key, members in sorted(alias_index.items()):
        shapes = {canonical_json(member["shape"]) for member in members}
        if len(shapes) < 2:
            continue
        alias_groups.append({
            "alias_group": group_key,
            "payload_identity": members[0]["diagnostics"]["geometry_identity"],
            "members": [{"condition_key": member["condition_key"],
                         "shape": member["shape"],
                         "family": member["family"],
                         "condition_ids": [run["record_id"] for run in member["runs"]]}
                        for member in members],
        })
        for member in members:
            member["diagnostics"]["alias_group"] = group_key
            member["diagnostics"]["alias_members"] = [
                other["condition_key"] for other in members
                if other["condition_key"] != member["condition_key"]]

    # ---- split enforcement report ----
    split_counts = Counter(observation["split"] for observation in merged.values())
    relabelled = [observation for observation in merged.values()
                  if "train_row_relabelled_heldout" in observation["diagnostics"]["flags"]]
    undeclared_holdout = [observation for observation in merged.values()
                          if "campaign_holdout_not_declared" in observation["diagnostics"]["flags"]]
    train_heldout_conflicts = []
    for observation in merged.values():
        if observation["split"] != "train":
            continue
        matches = heldout_index.get((observation["hardware_key"],
                                     observation["diagnostics"]["geometry_identity"]))
        if matches:
            train_heldout_conflicts.append({
                "condition_key": observation["condition_key"],
                "heldout_points": [match.get("point_id") for match in matches],
            })

    ordered = sorted(merged.values(), key=lambda item: (
        str(item["arch_family"]), str(item["hardware_key"]), str(item["family"]),
        str(item["die"]), str(item["protocol_signature"]),
        canonical_json(item["shape"]), item["condition_key"]))
    if not keep_excluded:
        for observation in ordered:
            if observation["split"] == "excluded":
                exclusions.append({"record_id": observation["diagnostics"]["record_id"],
                                   "source": observation["diagnostics"]["source_id"],
                                   "reason": "kept_out_by_request", "detail": None})
        ordered = [item for item in ordered if item["split"] != "excluded"]

    dataset = {"schema_version": OBSERVATION_SCHEMA, "observations": ordered}
    dataset_hash = sha256_value(ordered)
    counts = {
        "observations": len(ordered),
        "by_split": dict(sorted(split_counts.items())),
        "by_hardware": dict(sorted(Counter(item["hardware_key"] for item in ordered).items())),
        "by_arch_family": dict(sorted(Counter(str(item["arch_family"]) for item in ordered).items())),
        "by_family": dict(sorted(Counter(item["family"] for item in ordered).items())),
        "by_die": dict(sorted(Counter(item["die"] for item in ordered).items())),
        "by_protocol": dict(sorted(Counter(item["protocol_signature"] for item in ordered).items())),
        "by_timing_scope": dict(sorted(Counter(str(item["timing_scope"]) for item in ordered).items())),
        "by_source_id": dict(sorted(Counter(item["diagnostics"]["source_id"]
                                           for item in ordered).items())),
        "raw_events": sum(len(item["raw_samples_ms"]) for item in ordered),
        "merged_runs": sum(len(item["runs"]) - 1 for item in ordered),
        "single_trial_conditions": sum(1 for item in ordered
                                       if item["diagnostics"]["single_trial"]),
        "alias_groups": len(alias_groups),
        "excluded_records": len(exclusions),
        "relabelled_train_to_heldout": len(relabelled),
        "campaign_holdout_not_declared": len(undeclared_holdout),
        "train_heldout_conflicts": len(train_heldout_conflicts),
        "heldout_index_keys": len(heldout_index),
    }
    estimator_deltas = [item["diagnostics"]["pooled_minus_trial_median_ms"] for item in ordered
                        if item["diagnostics"]["pooled_minus_trial_median_ms"] not in (None, 0.0)]
    counts["estimator_mismatch_conditions"] = len(estimator_deltas)
    counts["estimator_mismatch_median_ms"] = median(estimator_deltas) if estimator_deltas else None
    counts["estimator_mismatch_max_abs_ms"] = (max(abs(value) for value in estimator_deltas)
                                               if estimator_deltas else None)

    discovery = discover_round2_dirs([("a2", "A2", a2_root), ("a3_round2", "A3", a3_round2_root)])
    manifest = {
        "schema_version": "mindie-sim.dataset-manifest/1",
        "generated_at": _utc_now(),
        "dataset_schema": OBSERVATION_SCHEMA,
        "dataset_sha256": dataset_hash,
        "counts": counts,
        "merge": merge_stats,
        "sources": census,
        "missing_sources": missing,
        "exclusions": exclusions,
        "alias_groups": alias_groups,
        "split_enforcement": {
            "declared_heldout_points": len(declared_points) + len(round1),
            "round1_heldout_points": len(round1),
            "new_declared_heldout_points": len(declared_points),
            "heldout_index_keys": len(heldout_index),
            "relabelled_train_rows": [
                {"condition_key": item["condition_key"],
                 "family": item["family"], "shape": item["shape"],
                 "heldout_point": item["diagnostics"]["heldout_point_id"],
                 "record_id": item["diagnostics"]["record_id"]}
                for item in relabelled],
            "campaign_holdout_not_declared": [
                {"condition_key": item["condition_key"], "family": item["family"],
                 "shape": item["shape"], "record_id": item["diagnostics"]["record_id"]}
                for item in undeclared_holdout],
            "train_heldout_conflicts": train_heldout_conflicts,
            "policy": ("a condition whose family-aware shape matches any declared heldout key "
                       "is heldout for the lifetime of the corpus; a campaign-declared holdout "
                       "that the round never declared stays heldout and is published as "
                       "campaign_holdout_not_declared; train rows can never overlap a heldout key"),
        },
        "declarations": [
            {"path": item["path"], "sha256": item["sha256"],
             "declaration_id": item["declaration_id"],
             "hardware_key": item["hardware_key"], "arch_family": item["arch_family"],
             "heldout_points": len(item["heldout"]), "train_points": len(item["train"]),
             "regime_rule": item["regime_rule"]}
            for item in declarations],
        "data_roots": {"a3": str(data_root), "a2": str(a2_root),
                       "a3_round2": str(a3_round2_root),
                       "round1_evidence": str(round1_evidence)},
        "round2_discovery": discovery,
        "protocol_drift": protocol_drift(ordered, declarations),
        "work_model": {"constants": (regime_rule or {}).get("constants"),
                       "work_models": (regime_rule or {}).get("work_models"),
                       "band_edges_ms": list(metrics.band_edges((regime_rule or {}).get("bands") or {}))},
        "strict_protocol": bool(strict_protocol),
    }
    return {"dataset": dataset, "manifest": manifest}


def protocol_drift(observations: Sequence[Mapping[str, Any]],
                   declarations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare every observation's protocol with its SKU's declared protocol identity.

    The declarations fix the protocol for *new* measurements. Older campaigns legitimately
    differ (round 1 has one-trial runs and a warmup scope of once-per-condition), so this is
    a report, never a rejection: it is how a campaign that drifted from its own declaration
    becomes visible instead of being discovered at score time.
    """
    declared: dict[str, tuple[dict[str, Any], str]] = {}
    for declaration in declarations:
        identity = declaration.get("protocol_identity") or {}
        if not identity:
            continue
        core = protocol_core(identity)
        declared[declaration["arch_family"]] = (core, metrics.protocol_signature(core))
    report: dict[str, Any] = {}
    for arch, (want, want_signature) in sorted(declared.items()):
        rows = [item for item in observations if item["arch_family"] == arch]
        by_source: dict[str, Any] = {}
        for item in rows:
            have = item["diagnostics"].get("protocol") or {}
            applicable = metrics.CORE_PROTOCOL_FIELDS
            if not str(item["family"]).startswith(("attention", "rope", "masked")):
                # ``attention_scope`` is declared for the attention lane only; comparing a
                # plain GEMM against it would manufacture a drift that does not exist.
                applicable = tuple(name for name in applicable if name != "attention_scope")
            differences = {name: {"declared": want.get(name), "observed": have.get(name)}
                           for name in applicable if want.get(name) != have.get(name)}
            if not differences:
                continue
            bucket = by_source.setdefault(item["diagnostics"]["source_id"], {
                "conditions": 0, "fields": {}, "declared_signature": want_signature})
            bucket["conditions"] += 1
            for name, values in differences.items():
                field = bucket["fields"].setdefault(name, {"declared": values["declared"],
                                                           "observed": {}})
                key = str(values["observed"])
                field["observed"][key] = field["observed"].get(key, 0) + 1
        report[arch] = {
            "declared_signature": want_signature,
            "declared_protocol": want,
            "conditions": len(rows),
            "matching": sum(1 for item in rows
                            if item["protocol_signature"] == want_signature),
            "drifting_by_source": dict(sorted(by_source.items())),
        }
    return report


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ derived table ---
STORE_FAMILIES = {"gemm", "add", "copy", "copy_bytes"}


def build_outside_store_table(dataset: Mapping[str, Any], manifest: Mapping[str, Any],
                             store_identities: Iterable[str] | None = None) -> dict[str, Any]:
    """Train conditions that never entered the calibration store, with both estimators."""
    store_ids = None if store_identities is None else set(store_identities)
    rows = []
    cannot = []
    for observation in dataset["observations"]:
        diagnostics = observation["diagnostics"]
        if observation["split"] != "train":
            continue
        if store_ids is None:
            # No store census supplied: the generic store is a derived ingest of the
            # nine-model run only, so that source is the only one it can contain.
            in_store = observation["family"] in STORE_FAMILIES and any(
                run.get("source_id") == "nine_model_linearity" for run in observation["runs"])
        else:
            # The store is one SKU's ingest; keying it without the hardware would mark a
            # same-shaped condition on the other SKU as "already in the store".
            in_store = (observation["hardware_key"], observation["family"],
                        observation["diagnostics"]["geometry_identity"]) in store_ids
        if in_store:
            continue
        record_id = diagnostics.get("record_id")
        duration = observation["duration_ms"]
        pooled = diagnostics.get("pooled_median_ms")
        row = {
            "condition_key": observation["condition_key"],
            "record_id": record_id,
            "source_id": diagnostics.get("source_id"),
            "source_artifact": diagnostics.get("source_artifact"),
            "family": observation["family"],
            "recipe": observation["recipe"],
            "shape": canonical_json(observation["shape"]),
            "duration_ms": duration,
            "pooled_median_ms": pooled,
            "delta_ms": (None if pooled is None else pooled - duration),
            "delta_pct": (None if pooled in (None, 0) else 100.0 * (pooled - duration) / duration),
            "trials": diagnostics.get("trials"),
            "raw_events": diagnostics.get("raw_events"),
            "run_ids": sorted({run.get("run_id") for run in observation["runs"]}),
            "protocol_signature": observation["protocol_signature"],
            "timing_scope": observation["timing_scope"],
            "die": observation["die"],
            "hardware_key": observation["hardware_key"],
            "source_class": observation["source_class"],
            "split": observation["split"],
            "estimator": "median_of_per_trial_medians",
            "normalizable": True,
            "note": None,
        }
        if diagnostics.get("analytic_ms_artifact") is None:
            row["note"] = "artifact_does_not_attest_analytic_ms"
        if "trial_structure_not_recovered" in (diagnostics.get("flags") or []):
            row["normalizable"] = False
            row["note"] = "trial_structure_not_recovered;target_equals_pooled_median"
            cannot.append({"condition_key": observation["condition_key"],
                           "reason": "trial_structure_not_recovered"})
        rows.append(row)
    for exclusion in manifest.get("exclusions") or []:
        if exclusion.get("reason") in {"not_measured", "no_usable_samples"} and \
                exclusion.get("detail", "").startswith("status=pending"):
            cannot.append({"condition_key": None, "record_id": exclusion.get("record_id"),
                           "reason": exclusion.get("reason")})
    return {
        "schema_version": "mindie-sim.train-outside-store/1",
        "generated_at": _utc_now(),
        "estimator_note": ("duration_ms is the scored target (median of per-trial medians); "
                           "pooled_median_ms is the diagnostic the calibration store uses, so "
                           "delta_ms is the estimator mismatch of the old store per condition"),
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "distinct_conditions": len({row["condition_key"] for row in rows}),
            "by_source": dict(sorted(Counter(row["source_id"] for row in rows).items())),
            "by_family": dict(sorted(Counter(row["family"] for row in rows).items())),
            "not_normalizable": len(cannot),
            "not_normalizable_detail": cannot,
            "estimator_mismatch_conditions": sum(1 for row in rows
                                                 if (row["delta_ms"] or 0) not in (0, None)),
            "estimator_mismatch_median_ms": median(
                [row["delta_ms"] for row in rows if row["delta_ms"] not in (None, 0.0)]),
        },
    }


# -------------------------------------------------------------------- output ---
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (canonical_json(value) if isinstance(value, (list, dict))
                                   else value) for key, value in row.items()})


def render_report(manifest: Mapping[str, Any], derived: Mapping[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Assembled observation dataset — A2/A3 accuracy round",
        "",
        f"Generated: `{manifest['generated_at']}`  ",
        f"Schema: `{manifest['dataset_schema']}`  ",
        f"Dataset hash (content): `{manifest['dataset_sha256']}`",
        "",
        "Produced by `python -m tools.accuracy.assemble_dataset`. Nothing here is read by a",
        "model: the file is an evidence index, and the scorer joins frozen predictions to it.",
        "",
        "## 1. Counts",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| observations (distinct conditions) | {counts['observations']} |",
        f"| raw event samples retained | {counts['raw_events']} |",
        f"| repeated runs merged into existing conditions | {counts['merged_runs']} |",
        f"| exact duplicate rows collapsed | {manifest['merge']['exact_duplicate_rows']} |",
        f"| single-trial conditions | {counts['single_trial_conditions']} |",
        f"| equal-payload alias groups | {counts['alias_groups']} |",
        f"| excluded records | {counts['excluded_records']} |",
        f"| declared-heldout points loaded | {manifest['split_enforcement']['declared_heldout_points']} |",
        "",
        "### By split",
        "",
        "| split | conditions |",
        "| --- | ---: |",
    ]
    for key, value in counts["by_split"].items():
        lines.append(f"| `{key}` | {value} |")
    for title, key in (("hardware", "by_hardware"), ("architecture", "by_arch_family"),
                       ("die", "by_die"), ("timing scope", "by_timing_scope")):
        lines += ["", f"### By {title}", "", f"| {title} | conditions |", "| --- | ---: |"]
        for name, value in counts[key].items():
            lines.append(f"| `{name}` | {value} |")
    lines += ["", "### By family", "", "| family | conditions |", "| --- | ---: |"]
    for name, value in counts["by_family"].items():
        lines.append(f"| `{name}` | {value} |")
    lines += ["", "### By protocol signature", "",
              "| protocol_signature | conditions |", "| --- | ---: |"]
    for name, value in counts["by_protocol"].items():
        lines.append(f"| `{name}` | {value} |")
    lines += ["", "### By source artifact", "", "| source | conditions |", "| --- | ---: |"]
    for name, value in counts["by_source_id"].items():
        lines.append(f"| `{name}` | {value} |")

    lines += ["", "## 2. Sources scanned", "",
              "| source | role | files | records | status |", "| --- | --- | ---: | ---: | --- |"]
    for source in manifest["sources"]:
        lines.append(f"| `{source['id']}` | {source.get('role')} | {len(source.get('files') or [])} "
                     f"| {source.get('records', 0)} | {source.get('status')} |")
    if manifest["missing_sources"]:
        lines += ["", "Missing declared artifacts:", ""]
        for item in manifest["missing_sources"]:
            lines.append(f"- `{item['path']}` ({item['source']})")

    discovery = manifest["round2_discovery"]
    lines += ["", "### Round-2 campaign directories", ""]
    for label, entry in discovery.items():
        if not entry.get("exists"):
            lines.append(f"- **{label}** `{entry['path']}` — **does not exist yet**; 0 conditions ingested.")
            continue
        lines.append(f"- **{label}** `{entry['path']}` — {len(entry.get('files') or [])} files; "
                     f"classifications: {entry.get('counts') or '{}'}; 0 conditions ingested.")
    lines += ["", "The round-2 roots are discovered and classified on every run so that new",
              "campaign output is picked up without a code change; when a file is not a shape",
              "this reader knows it is counted as `unclassified`/`unreadable_json` rather than",
              "silently skipped."]

    enforcement = manifest["split_enforcement"]
    lines += ["", "## 3. Split enforcement", "",
              f"- declared heldout points: **{enforcement['declared_heldout_points']}** "
              f"({enforcement['round1_heldout_points']} round-1 test set 1, "
              f"{enforcement['new_declared_heldout_points']} newly declared)",
              f"- heldout index keys (hardware, family-aware shape): **{enforcement['heldout_index_keys']}**",
              f"- train rows relabelled to heldout because they hit a declared key: "
              f"**{counts['relabelled_train_to_heldout']}**",
              f"- campaign holdouts the round never declared (kept heldout): "
              f"**{counts['campaign_holdout_not_declared']}**",
              f"- residual train/heldout conflicts: **{counts['train_heldout_conflicts']}** "
              f"(must be 0)",
              "", "Policy: " + enforcement["policy"], ""]
    if enforcement["relabelled_train_rows"]:
        lines += ["| relabelled condition | family | shape | heldout point | record |",
                  "| --- | --- | --- | --- | --- |"]
        for item in enforcement["relabelled_train_rows"]:
            lines.append(f"| `{item['condition_key']}` | {item['family']} | "
                         f"`{canonical_json(item['shape'])}` | `{item['heldout_point']}` | "
                         f"`{item['record_id']}` |")
        lines.append("")
    if enforcement["campaign_holdout_not_declared"]:
        lines += ["Campaign-declared holdouts outside the round declarations:", "",
                  "| condition | family | shape | record |", "| --- | --- | --- | --- |"]
        for item in enforcement["campaign_holdout_not_declared"]:
            lines.append(f"| `{item['condition_key']}` | {item['family']} | "
                         f"`{canonical_json(item['shape'])}` | `{item['record_id']}` |")
        lines.append("")

    lines += ["## 4. Exclusions", ""]
    if manifest["exclusions"]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for exclusion in manifest["exclusions"]:
            grouped[str(exclusion.get("reason"))].append(exclusion)
        lines += ["| reason | records | detail |", "| --- | ---: | --- |"]
        for reason, items in sorted(grouped.items()):
            detail = sorted({str(item.get("detail")) for item in items})[:3]
            lines.append(f"| `{reason}` | {len(items)} | {'; '.join(detail)} |")
        lines += ["", "Every excluded record, individually:", "",
                  "| record | source | family | shape | reason | detail |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for exclusion in manifest["exclusions"]:
            lines.append(f"| `{exclusion.get('record_id')}` | `{exclusion.get('source')}` | "
                         f"{exclusion.get('family')} | `{canonical_json(exclusion.get('shape'))}` | "
                         f"`{exclusion.get('reason')}` | {exclusion.get('detail')} |")
    else:
        lines.append("No records were excluded.")
    lines.append("")

    lines += ["## 5. Estimator mismatch (pooled median vs median of per-trial medians)", "",
              "The old calibration store targets the pooled-event median; the scorer targets the",
              "median of per-trial medians. Both are recomputed here per condition.", "",
              "| metric | value |", "| --- | ---: |",
              f"| conditions where the two differ | {counts['estimator_mismatch_conditions']} |",
              f"| median difference (pooled − target, ms) | {counts['estimator_mismatch_median_ms']} |",
              f"| max absolute difference (ms) | {counts['estimator_mismatch_max_abs_ms']} |", ""]

    lines += ["## 6. Equal-payload alias groups", ""]
    if manifest["alias_groups"]:
        lines += ["| group | payload | members |", "| --- | --- | ---: |"]
        for group in manifest["alias_groups"]:
            lines.append(f"| `{group['alias_group']}` | `{group['payload_identity']}` | "
                         f"{len(group['members'])} |")
        lines += ["", "Member factorisations:", ""]
        for group in manifest["alias_groups"]:
            rendered = ", ".join(canonical_json(member["shape"]) for member in group["members"])
            lines.append(f"- `{group['alias_group']}`: {rendered}")
    else:
        lines.append("No equal-payload alias groups were formed.")
    lines.append("")

    summary = derived["summary"]
    lines += ["## 7. Train conditions outside the calibration store", "",
              "Derived table: `train-outside-store.json` / `.csv`.", "",
              "| metric | value |", "| --- | ---: |",
              f"| records | {summary['rows']} |",
              f"| distinct conditions | {summary['distinct_conditions']} |",
              f"| not normalizable | {summary['not_normalizable']} |",
              f"| conditions where pooled ≠ trial-median | {summary['estimator_mismatch_conditions']} |",
              "", "By source:", "", "| source | records |", "| --- | ---: |"]
    for name, value in summary["by_source"].items():
        lines.append(f"| `{name}` | {value} |")
    lines += ["", "By family:", "", "| family | records |", "| --- | ---: |"]
    for name, value in summary["by_family"].items():
        lines.append(f"| `{name}` | {value} |")
    if summary["not_normalizable_detail"]:
        lines += ["", "Records that could not be normalised:", ""]
        for item in summary["not_normalizable_detail"][:40]:
            lines.append(f"- `{item.get('condition_key') or item.get('record_id')}`: {item['reason']}")
    lines += ["", "## 8. Declarations in force", "",
              "| declaration | sha256 | hardware | arch | heldout | train |",
              "| --- | --- | --- | --- | ---: | ---: |"]
    for item in manifest["declarations"]:
        lines.append(f"| `{item['declaration_id']}` | `{item['sha256'][:16]}…` | "
                     f"`{item['hardware_key']}` | {item['arch_family']} | "
                     f"{item['heldout_points']} | {item['train_points']} |")
    work = manifest["work_model"]
    lines += ["", f"Regime bands: launch `< {work['band_edges_ms'][0]} ms`, "
                  f"large `>= {work['band_edges_ms'][1]} ms` (from the declaration, not from code).", ""]
    return "\n".join(lines) + "\n"


def store_condition_identities(data_root: Path) -> set[tuple[str, str, str]]:
    """Condition identities inside the generic calibration store (for the derived table)."""
    path = data_root / "independent-generalization-20261021/generic-training.sqlite"
    if not path.exists():
        return set()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT hardware_key, op_family, shape_json, recipe FROM sample_statistics").fetchall()
    finally:
        connection.close()
    identities: set[tuple[str, str, str]] = set()
    for hardware, family, shape_json, recipe in rows:
        try:
            shape = json.loads(shape_json or "{}")
        except json.JSONDecodeError:
            continue
        try:
            identities.add((hardware, str(family),
                            shape_identity(_family_class(family), shape)))
        except KeyError:
            continue
    return identities


# ----------------------------------------------------------------------- cli ---
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--a2-root", type=Path, default=DEFAULT_A2_ROOT)
    parser.add_argument("--a3-round2-root", type=Path, default=DEFAULT_A3_ROUND2_ROOT)
    parser.add_argument("--declarations", type=Path, default=DEFAULT_DECLARATIONS)
    parser.add_argument("--round1-evidence", type=Path,
                        default=DEFAULT_DATA_ROOT / "independent-generalization-20261021")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / ".workstreams/dataset")
    parser.add_argument("--strict-protocol", action="store_true",
                        help="include warmup_scope and per-call amortisation in the protocol signature")
    parser.add_argument("--keep-excluded", action="store_true",
                        help="keep non-calibratable rows in the dataset with split='excluded'")
    parser.add_argument("--no-derived-table", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common(parser)
    args = parser.parse_args(argv)

    if not args.data_root.is_dir():
        print(f"integrity: data root missing: {args.data_root}", file=sys.stderr)
        return 3
    assembled = assemble(
        data_root=args.data_root, a2_root=args.a2_root,
        a3_round2_root=args.a3_round2_root, declarations_dir=args.declarations,
        round1_evidence=args.round1_evidence, strict_protocol=args.strict_protocol,
        keep_excluded=args.keep_excluded)
    dataset, manifest = assembled["dataset"], assembled["manifest"]
    if manifest["counts"]["train_heldout_conflicts"]:
        print("integrity: train rows overlap declared heldout keys", file=sys.stderr)
        return 3

    out: Path = args.out
    write_json(out / "observations.json", dataset)
    write_json(out / "dataset-manifest.json", manifest)
    write_json(out / "exclusions.json", manifest["exclusions"])
    write_json(out / "alias-groups.json", manifest["alias_groups"])
    derived = {"schema_version": "mindie-sim.train-outside-store/1", "rows": [], "summary": {}}
    if not args.no_derived_table:
        derived = build_outside_store_table(dataset, manifest,
                                           store_condition_identities(args.data_root))
        write_json(out / "train-outside-store.json", derived)
        write_csv(out / "train-outside-store.csv", derived["rows"])
        manifest["outside_store_table"] = derived["summary"]
        write_json(out / "dataset-manifest.json", manifest)
    (out / "dataset-report.md").write_text(render_report(manifest, derived))
    print(json.dumps({
        "status": "assembled",
        "observations": manifest["counts"]["observations"],
        "raw_events": manifest["counts"]["raw_events"],
        "by_split": manifest["counts"]["by_split"],
        "excluded_records": manifest["counts"]["excluded_records"],
        "dataset_sha256": manifest["dataset_sha256"],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
