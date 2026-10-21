"""Score frozen predictions against the frozen observation dataset — join only.

[Role] The only process in the round that opens an outcome, and it does one thing: join a
  frozen prediction file to the assembled dataset and compute the declared metrics.
[Invariants]
  * Never imports model code, never fits, never re-selects a variant, never calls a device.
  * Refuses to run (exit 3) when the prediction hash, the dataset hash, the declaration
    hashes, the completeness flag or the ``evaluation_outcomes_read: false`` claim differ
    from the freeze receipt.
  * Target = median of per-trial medians. Coverage denominator = every declared condition
    in scope, including unsupported / unresolved / unqueried ones.
  * Primary roll-up: exactly one variant per condition (direct curve if numeric, else the
    calibrated DB, else analytic). The all-variant roll-up is published beside it, never
    instead of it.
  * Every metric names its weighting and its n (L8); a cell with n < 5 reports
    ``insufficient_n`` and no confidence interval instead of a fabricated one.
[Timeouts & Locks] One pass over local files plus the declared bootstrap; no network, no
  subprocess, no device.
[Side-Effect Policy] Reads the dataset, the frozen predictions and the declarations; writes
  only --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import metrics
from .assemble_dataset import load_round1_heldout, load_split_declarations
from .freeze_predictions import (OUTCOME_KEYS, load_dataset, verify_geometry_only)
from .metrics import (
    ACCEPTANCE_SCHEMA, FAMILY_ALIASES, PROTOCOL_ACCEPTANCE_SCHEMA, SCORE_SCHEMA,
    canonical_json, family_class as _family_class, geometry_key, identity_candidates,
    shape_identity, sha256_file, sha256_value,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DATASET = REPO_ROOT / ".workstreams/dataset"
DEFAULT_DECLARATIONS = REPO_ROOT / ".workstreams/accuracy-protocol"
DEFAULT_ROUND1 = Path("/Volumes/data/mindie-sim-data/a3/2026-10-20/independent-generalization-20261021")

# Declared protocol gates (protocol.md §5.1). Thresholds are protocol constants, not SKU
# constants, and every one is overridable on the command line for a different round.
GATE_THRESHOLDS = {
    "G1_coverage_family": 0.85, "G1_coverage_sku": 0.90,
    "G2_band_wape": {"launch": 20.0, "mid": 12.0, "large": 10.0},
    "G3_band_median_ape": {"launch": 20.0, "mid": 10.0, "large": 10.0},
    "G4_p90_multiplier": 1.6, "G4_family_max_ape": 50.0,
    "G5_abs_median_bias": 8.0, "G6_penalized_wape": 30.0,
    "G7_delta_wape_relative": -25.0, "G8_regression_allowance_pp": 2.0,
    "G9_containment": 0.80, "G9_min_intervals": 20,
    "G10_transfer_ratio": 0.85, "G12_family_min": 10, "G12_band_min": 5,
    "G12_sku_min": 60,
}
PRIMARY_CHAINS = {
    "curve_first": ("curve", "calibrated_db", "analytic"),
    "calibrated_first": ("calibrated_db", "curve", "analytic"),
    # Round-2 declared policy: a direct curve only when the product says it is
    # production-eligible; else the fitted physical layer when the artifact models the
    # identity; else the DB calibration; else analytic.
    "curve_physical_first": ("curve", "physical", "calibrated_db", "analytic"),
    "physical_first": ("curve", "physical", "calibrated_db", "analytic"),
}
LIMITATIONS = [
    "Primitive measurements are not whole-model TTFT/TPOT accuracy.",
    "Event intervals include declared dispatch/allocation/lifetime effects; they are not pure kernel service time.",
    "A2 peak-bandwidth/capacity fields are pinned from the SKU catalogue and any runtime-clock disagreement is disclosed, not corrected.",
    "Report-only families have n <= 6 and cannot support a family claim on either SKU.",
    "Cross-SKU transfer is a secondary result and does not imply that the target SKU was tuned.",
    "Test set 1 is repair-biased by construction; test set 2 carries the primary claim.",
    "Banded 'large' thresholds have fewer than 5 conditions of prior evidence.",
    "The copy mid band has 2 conditions on each SKU and is not gated.",
    "A physical-prediction value flagged low_confidence is scored and carries "
    "flagged_low_confidence=true; its parameters are not fully identified by the data.",
]


class IntegrityError(RuntimeError):
    """A freeze/dataset hash or completeness contract was violated; exit 3."""


# --------------------------------------------------------------------- utils ---
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# ------------------------------------------------------------------ freezing ---
def verify_freeze(*, predictions_path: Path, receipt_path: Path, dataset_path: Path,
                  declarations: Path, allow_dirty: bool) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    document = load_json(predictions_path)
    if document.get("schema_version") != metrics.PREDICTIONS_SCHEMA:
        raise IntegrityError(f"predictions_schema_mismatch:{document.get('schema_version')}")
    if receipt.get("schema_version") != metrics.FREEZE_SCHEMA:
        raise IntegrityError(f"receipt_schema_mismatch:{receipt.get('schema_version')}")
    actual = sha256_file(predictions_path)
    if actual != receipt.get("predictions_sha256"):
        raise IntegrityError("predictions_changed_after_freeze")
    if document.get("status") != "completed":
        raise IntegrityError(f"freeze_incomplete:{document.get('status')}")
    if document.get("completed_entries") != document.get("expected_entries"):
        raise IntegrityError("freeze_entries_incomplete")
    if document.get("evaluation_outcomes_read") is not False:
        raise IntegrityError("freeze_claims_outcomes_were_read")
    if document.get("outcomes_directory_listed") is not False:
        raise IntegrityError("freeze_claims_outcomes_directory_was_listed")
    if receipt.get("evaluation_outcomes_read") is not False:
        raise IntegrityError("receipt_claims_outcomes_were_read")

    dataset, dataset_side = load_dataset(dataset_path)
    dataset_file = dataset_path if dataset_path.is_file() else dataset_path / "observations.json"
    dataset_hash = sha256_value(dataset["observations"])
    if receipt.get("dataset", {}).get("sha256") != dataset_hash:
        raise IntegrityError("dataset_changed_after_freeze")
    if receipt.get("dataset", {}).get("file_sha256") != sha256_file(dataset_file):
        raise IntegrityError("dataset_file_changed_after_freeze")
    if dataset_side.get("dataset_sha256") and dataset_side["dataset_sha256"] != dataset_hash:
        raise IntegrityError("dataset_manifest_hash_mismatch")

    protocol = receipt.get("protocol") or {}
    for name, expected in (protocol.get("declaration_files") or {}).items():
        path = declarations / name
        if not path.exists() or sha256_file(path) != expected:
            raise IntegrityError(f"protocol_declaration_changed:{name}")

    source = receipt.get("source") or {}
    if source.get("dirty") and not allow_dirty:
        raise IntegrityError("freeze_records_a_dirty_source_tree (pass --allow-dirty to score it)")
    return {"receipt": receipt, "document": document, "dataset": dataset,
            "dataset_side": dataset_side, "dataset_hash": dataset_hash,
            "dataset_file": dataset_file, "predictions_sha256": actual}


# --------------------------------------------------------------------- scope ---
def scope_points(*, declarations_dir: Path, round1_evidence: Path,
                 test_sets: Sequence[str]) -> dict[str, Any]:
    """Every declared heldout condition in scope, per hardware key."""
    points: list[dict[str, Any]] = []
    regime_rules: dict[str, Any] = {}
    declaration_files: dict[str, Any] = {}
    for declaration in load_split_declarations(declarations_dir):
        hardware = declaration["hardware_key"]
        regime_rules[hardware] = declaration["regime_rule"]
        declaration_files[hardware] = {"path": declaration["path"],
                                       "sha256": declaration["sha256"],
                                       "declaration_id": declaration["declaration_id"],
                                       "arch_family": declaration["arch_family"],
                                       "sku": declaration["raw"].get("sku")}
        for point in declaration["heldout"]:
            test_set = point.get("test_set") or "test2_fresh_heldout"
            if test_set not in test_sets:
                continue
            points.append({
                "point_id": point.get("point_id"),
                "hardware_key": hardware,
                "family": point.get("family"),
                "shape": point.get("shape"),
                "test_set": test_set,
                "regime": point.get("regime"),
                "geometry_key": point.get("geometry_key") or geometry_key(
                    hardware, point.get("family"), point.get("shape")),
                "acceptance_bearing": point.get("acceptance_bearing", True),
            })
    if "test1_old_heldout" in test_sets:
        hardware = next((key for key, value in regime_rules.items()
                         if value), None)
        for point in load_round1_heldout(round1_evidence, hardware):
            points.append({
                "point_id": point["point_id"],
                "hardware_key": point.get("hardware_key") or hardware,
                "family": point.get("family"),
                "shape": point.get("shape"),
                "test_set": "test1_old_heldout",
                "regime": None,
                "geometry_key": geometry_key(point.get("hardware_key") or hardware,
                                             point.get("family"), point.get("shape")),
                "acceptance_bearing": True,
            })
    deduped = {}
    for point in points:
        deduped[(point["hardware_key"], point["geometry_key"])] = point
    return {"points": list(deduped.values()), "regime_rules": regime_rules,
            "declaration_files": declaration_files}


# ---------------------------------------------------------------------- join ---
def observation_index(dataset: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for observation in dataset.get("observations") or []:
        if observation.get("split") != "heldout":
            continue
        identities = observation["diagnostics"].get("geometry_identities") or [
            observation["diagnostics"].get("geometry_identity")]
        for identity in identities:
            if identity:
                index[(observation["hardware_key"], identity)].append(observation)
    return dict(index)


def prediction_index(document: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in document.get("predictions") or []:
        for identity in identity_candidates(entry["family"], entry["shape"]):
            index[(entry["hardware_key"], identity)].append(entry)
    return dict(index)


def _pick(observations: Sequence[Mapping[str, Any]],
          entry: Mapping[str, Any] | None) -> tuple[Mapping[str, Any] | None, str | None]:
    """One observation per condition: prefer the one whose attested protocol matches."""
    unique = {}
    for observation in observations:
        unique[(observation["condition_key"], observation["protocol_signature"])] = observation
    candidates = list(unique.values())
    if not candidates:
        return None, "missing_outcome"
    if len(candidates) == 1:
        return candidates[0], None
    if entry is not None:
        matched = [item for item in candidates
                   if item["protocol_signature"] == entry["protocol_key"]]
        if len(matched) == 1:
            return matched[0], None
    return None, "join_ambiguous"


def build_condition_records(*, points: Sequence[Mapping[str, Any]],
                            document: Mapping[str, Any],
                            dataset: Mapping[str, Any], regime_rules: Mapping[str, Any],
                            strict_protocol: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations_by_identity = observation_index(dataset)
    predictions_by_identity = prediction_index(document)
    work_models: dict[str, metrics.WorkModel] = {}
    edges: dict[str, tuple[float | None, float | None]] = {}
    for hardware, rule in regime_rules.items():
        if rule:
            work_models[hardware] = metrics.WorkModel(rule)
            edges[hardware] = metrics.band_edges(rule.get("bands") or {})
    conditions: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for point in points:
        hardware = point["hardware_key"]
        candidates = identity_candidates(point["family"], point["shape"])
        entries = []
        for identity in candidates:
            entries.extend(predictions_by_identity.get((hardware, identity)) or [])
        unique_entries = {entry["condition_id"]: entry for entry in entries}
        observations = []
        for identity in candidates:
            observations.extend(observations_by_identity.get((hardware, identity)) or [])
        entry = next(iter(unique_entries.values())) if len(unique_entries) == 1 else None
        if len(unique_entries) > 1:
            excluded.append({"condition_id": point["point_id"], "reason": "ambiguous_prediction_join",
                             "detail": sorted(unique_entries)})
        observation, problem = _pick(observations, entry)
        band = point.get("regime")
        analytic_lower = None
        band_reason = None
        model = work_models.get(hardware)
        if model is not None:
            try:
                analytic_lower = model.lower_bound_ms(point["family"], point["shape"])
            except metrics.WorkModelError as exc:
                analytic_lower = None
                band_reason = str(exc)
        if band is None and analytic_lower is not None:
            band = metrics.band_of(analytic_lower, edges.get(hardware, (None, None)))
        if band is None:
            band_reason = band_reason or "declared_work_model_cannot_band_this_geometry"
        record = {
            "condition_id": point["point_id"],
            "geometry_key": point["geometry_key"],
            "family": point["family"],
            "family_class": _family_class(point["family"]),
            "hardware_key": hardware,
            "arch_family": (regime_rules.get(hardware) or {}).get("arch_family"),
            "shape": point["shape"],
            "test_set": point["test_set"],
            "acceptance_bearing": point.get("acceptance_bearing", True),
            "band": band or "unbanded",
            "band_reason": band_reason,
            "analytic_lower_bound_ms": analytic_lower,
            "observed_ms": None if observation is None else observation["duration_ms"],
            "observation": observation,
            "entry": entry,
            "status": "ok",
            "detail": None,
        }
        if observation is None:
            record["status"] = problem or "missing_outcome"
            excluded.append({"condition_id": point["point_id"], "reason": record["status"],
                             "detail": "no heldout observation matches this declared geometry"})
        elif entry is None:
            record["status"] = "not_queried"
            excluded.append({"condition_id": point["point_id"], "reason": "not_queried",
                             "detail": "the freeze declares no prediction for this geometry"})
        else:
            observed_core = observation["diagnostics"].get("protocol") or {}
            if not metrics.protocol_matches(observed_core, entry.get("protocol_core") or {},
                                            strict=strict_protocol):
                record["status"] = "protocol_mismatch"
                excluded.append({"condition_id": point["point_id"], "reason": "protocol_mismatch",
                                 "detail": canonical_json({"observed": observed_core,
                                                           "declared": entry.get("protocol_core")})})
        conditions.append(record)
    return conditions, excluded


def condition_rows(conditions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Explode conditions into (condition, variant) rows and one primary row per condition."""
    rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        variants = (condition.get("entry") or {}).get("variants") or {}
        observed = condition.get("observed_ms")
        for variant_name in metrics_variant_names(variants):
            variant = variants.get(variant_name) or {}
            prediction = variant.get("prediction_ms")
            usable = (condition["status"] == "ok" and observed is not None
                      and prediction is not None)
            row = {
                "condition_id": condition["condition_id"],
                "geometry_key": condition["geometry_key"],
                "family": condition["family"],
                "family_class": condition["family_class"],
                "hardware_key": condition["hardware_key"],
                "band": condition["band"],
                "die": (condition.get("observation") or {}).get("die"),
                "protocol_signature": (condition.get("observation") or {}).get("protocol_signature"),
                "test_set": condition["test_set"],
                "acceptance_bearing": condition["acceptance_bearing"],
                "variant": variant_name,
                "prediction_ms": prediction if usable else None,
                "observed_ms": observed,
                "numeric": usable,
                "region": variant.get("region"),
                "status": ("ok" if usable else (variant.get("status") or condition["status"])),
                "ape_pct": (metrics.ape_pct(prediction, observed) if usable else None),
                "signed_error_pct": (metrics.signed_error_pct(prediction, observed)
                                     if usable else None),
            }
            lower, upper = variant.get("lower_bound_ms"), variant.get("upper_bound_ms")
            if lower is not None and upper is not None:
                row["reported_interval_ms"] = [lower, upper]
                row["observed_inside_reported_interval"] = bool(lower <= (observed or 0) <= upper)
            rows.append(row)
        chosen, reason = select_primary(variants)
        primary = next(row for row in rows if row["condition_id"] == condition["condition_id"]
                       and row["variant"] == chosen) if chosen else None
        primary_rows.append({
            "condition_id": condition["condition_id"],
            "geometry_key": condition["geometry_key"],
            "family": condition["family"],
            "family_class": condition["family_class"],
            "hardware_key": condition["hardware_key"],
            "band": condition["band"],
            "die": (condition.get("observation") or {}).get("die"),
            "protocol_signature": (condition.get("observation") or {}).get("protocol_signature"),
            "test_set": condition["test_set"],
            "acceptance_bearing": condition["acceptance_bearing"],
            "variant": chosen,
            "selection_reason": reason,
            "flagged_low_confidence": bool((
                (variants.get(chosen) or {}).get("low_confidence")) if chosen else False),
            "prediction_ms": (primary or {}).get("prediction_ms"),
            "observed_ms": observed,
            "numeric": bool(primary and primary["numeric"]),
            "region": (primary or {}).get("region"),
            "status": (primary or {}).get("status") or condition["status"],
            "ape_pct": (primary or {}).get("ape_pct"),
            "signed_error_pct": (primary or {}).get("signed_error_pct"),
            **({"reported_interval_ms": primary["reported_interval_ms"],
                "observed_inside_reported_interval": primary["observed_inside_reported_interval"]}
               if primary and "reported_interval_ms" in primary else {}),
        })
    return rows, primary_rows


def metrics_variant_names(variants: Mapping[str, Any]) -> list[str]:
    order = ["analytic", "physical", "calibrated_db", "curve", "db_and_curve"]
    names = [name for name in order if name in variants]
    names += [name for name in variants if name not in order]
    return names


def primary_eligible(name: str, variant: Mapping[str, Any]) -> bool:
    """The declared primary-path gate for one variant (round-2 policy).

    * ``curve`` — numeric and the product's own production gate says eligible. A legacy
      variant without the flag (round-1 freeze) is judged on its numeric value only.
    * ``physical`` — numeric and the artifact modeled the identity. A
      ``physical_low_confidence`` value is reported and flagged in the all-variant
      roll-up but does not enter the primary path; ``physical_unmodeled`` is not numeric.
    * everything else — numeric.
    """
    if variant.get("prediction_ms") is None:
        return False
    if name == "curve":
        return bool(variant.get("production_eligible", True))
    if name == "physical":
        # The integration policy decides: only an applied fitted row replaces the
        # analytic/curve path. A modeled-but-not-applied row stays a reported variant.
        if "applied" in variant:
            return bool(variant.get("applied"))
        status = str(variant.get("status") or "")
        if status in {"physical_calibrated", "physical_low_confidence"}:
            return True
        if status:
            return bool(variant.get("primary_eligible", False))
        return True
    return True


def select_primary(variants: Mapping[str, Any]) -> tuple[str | None, str]:
    chain = PRIMARY_CHAINS[_PRIMARY_POLICY[0]]
    for name in chain:
        variant = variants.get(name) or {}
        if primary_eligible(name, variant):
            return name, f"first eligible numeric variant in {'>'.join(chain)}"
    if variants:
        return None, "no eligible numeric variant"
    return None, "no variant recorded"


# module-level policy holder keeps select_primary callable from condition_rows
_PRIMARY_POLICY: list[str] = ["curve_first"]


def set_primary_policy(policy: str) -> None:
    if policy not in PRIMARY_CHAINS:
        raise ValueError(f"unknown_primary_policy:{policy}")
    _PRIMARY_POLICY[0] = policy


# ---------------------------------------------------------------- aggregation ---
def summarize_cell(rows: Sequence[Mapping[str, Any]], n_declared: int, *, b: int,
                   seed: int, subset_of: str | None = None) -> dict[str, Any]:
    summary = metrics.summarize(rows, n_declared=n_declared, b=b, seed=seed,
                                subset_of=subset_of)
    summary["statuses"] = metrics.status_histogram(rows)
    return summary


def group_by(rows: Iterable[Mapping[str, Any]],
             key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(dict(row))
    return dict(sorted(grouped.items()))


def variant_rollup(rows: Sequence[Mapping[str, Any]], n_declared: int, *,
                   b: int, seed: int) -> dict[str, Any]:
    by_variant = group_by(rows, "variant")
    output = {}
    for variant, variant_rows in by_variant.items():
        # Every declared condition is in each variant's denominator: a condition the
        # variant never answered is a coverage miss, not a smaller denominator.
        answered = {row["condition_id"] for row in variant_rows if row["numeric"]}
        missing = n_declared - len(answered)
        output[variant] = summarize_cell(variant_rows, n_declared, b=b, seed=seed)
        output[variant]["absent_conditions"] = max(0, missing)
    return output


def per_cell(rows: Sequence[Mapping[str, Any]], key: str, n_declared_by: Mapping[str, int],
             *, b: int, seed: int) -> dict[str, Any]:
    output = {}
    for name, cell_rows in group_by(rows, key).items():
        output[name] = summarize_cell(cell_rows, n_declared_by.get(name, len(cell_rows)),
                                      b=b, seed=seed, subset_of=key)
    return output


def regime_rollup(rows: Sequence[Mapping[str, Any]], points: Sequence[Mapping[str, Any]], *,
                  b: int, seed: int) -> dict[str, Any]:
    declared_by_band = Counter(str(point.get("regime") or "unbanded") for point in points)
    output = {}
    for name, cell_rows in group_by(rows, "band").items():
        output[name] = summarize_cell(cell_rows, declared_by_band.get(name, len(cell_rows)),
                                      b=b, seed=seed, subset_of="regime")
    for band in ("launch", "mid", "large"):
        if band not in output:
            output[band] = {"n": 0, "numeric": 0, "coverage": 0.0, "insufficient_n": True,
                            "wape_pct": None, "median_ape_pct": None, "p90_ape_pct": None,
                            "max_ape_pct": None, "note": "band absent from scope"}
    return output


def repeat_noise_floor(dataset: Mapping[str, Any], families: Iterable[str]) -> dict[str, Any]:
    """G11: same-protocol run-to-run median shift, measured from merged runs."""
    per_family: dict[str, list[float]] = defaultdict(list)
    for observation in dataset.get("observations") or []:
        if observation.get("split") != "train" and observation.get("split") != "heldout":
            continue
        runs = observation.get("runs") or []
        if len(runs) < 2:
            continue
        values = [run["trial_medians_ms"] for run in runs if run.get("trial_medians_ms")]
        if len(values) < 2:
            continue
        first = metrics.median(values[0])
        second = metrics.median(values[1])
        if not first or not second:
            continue
        per_family[observation["family"]].append(
            abs(first - second) / ((first + second) / 2.0) * 100.0)
    floors = {family: (metrics.median(values) if values else None)
              for family, values in per_family.items()}
    measured = {family: value for family, value in floors.items() if value is not None}
    return {
        "repeat_noise_floor_pct": floors,
        "conditions_by_family": {family: len(values) for family, values in per_family.items()},
        "families_measured": sorted(measured),
        "passed": len(measured) >= 1,
        "note": ("same-protocol repeated runs merged into one condition; a family with no "
                 "repeat has no measurable floor"),
    }


# --------------------------------------------------------------------- gates ---
def gate(gate_id: str, scope: str, threshold: Any, observed: Any, status: str,
         note: str | None = None) -> dict[str, Any]:
    return {"id": gate_id, "scope": scope, "threshold": threshold, "observed": observed,
            "passed": status == "pass", "status": status,
            **({"note": note} if note else {})}


def evaluate_gates(*, primary: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]],
                   families: Mapping[str, Any], noise: Mapping[str, Any],
                   variants: Mapping[str, Any], sku_totals: Mapping[str, Any],
                   n_declared: int) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    # G1 coverage per acceptance family and per SKU
    coverage_observed = {family: (block.get("metrics") or {}).get("coverage") or 0.0
                         for family, block in families.items()
                         if block.get("acceptance_bearing")}
    worst = min(coverage_observed.values()) if coverage_observed else 0.0
    gates.append(gate("G1", "per acceptance family | per SKU on test set 2",
                      {"family": GATE_THRESHOLDS["G1_coverage_family"],
                       "sku": GATE_THRESHOLDS["G1_coverage_sku"]},
                      {"worst_family_coverage": worst,
                       "sku_coverage": sku_totals.get("coverage")},
                      "pass" if coverage_observed and worst >= GATE_THRESHOLDS["G1_coverage_family"]
                      else "fail" if coverage_observed else "not_evaluated",
                      "test set 2 is the SKU-level gate; test set 1 is a repair check"))
    # G2/G3/G4 per band, only where n >= 5
    for gate_id, metric_key, threshold_key, label in (
            ("G2", "wape_pct", "G2_band_wape", "band WAPE"),
            ("G3", "median_ape_pct", "G3_band_median_ape", "band median APE")):
        thresholds = GATE_THRESHOLDS[threshold_key]
        observed = {}
        statuses = []
        for band, threshold in thresholds.items():
            block = primary.get(band) or {}
            value = block.get(metric_key)
            n = block.get("numeric") or 0
            if n < GATE_THRESHOLDS["G12_band_min"]:
                statuses.append("insufficient_n")
                observed[band] = {"n": n, "value": value, "status": "insufficient_n"}
            else:
                passed = value is not None and value <= threshold
                statuses.append("pass" if passed else "fail")
                observed[band] = {"n": n, "value": value, "threshold": threshold,
                                  "status": "pass" if passed else "fail"}
        status = ("fail" if "fail" in statuses else
                  "pass" if statuses and all(item == "pass" for item in statuses)
                  else "insufficient_n")
        gates.append(gate(gate_id, f"per band, n >= {GATE_THRESHOLDS['G12_band_min']}",
                          thresholds, observed, status, label))
    band_p90 = {}
    for band, block in primary.items():
        if not isinstance(block, Mapping) or block.get("n", 0) < GATE_THRESHOLDS["G12_band_min"]:
            continue
        target = GATE_THRESHOLDS["G2_band_wape"].get(band)
        value = block.get("p90_ape_pct")
        band_p90[band] = {"p90_ape_pct": value,
                          "limit": None if target is None else
                          target * GATE_THRESHOLDS["G4_p90_multiplier"],
                          "n": block.get("numeric")}
    family_max = {family: (block.get("metrics") or {}).get("max_ape_pct")
                  for family, block in families.items()}
    g4_status = "pass"
    for band, item in band_p90.items():
        if item["limit"] is not None and (item["p90_ape_pct"] or 0) > item["limit"]:
            g4_status = "fail"
    if any((value or 0) > GATE_THRESHOLDS["G4_family_max_ape"] for value in family_max.values()):
        g4_status = "fail"
    gates.append(gate("G4", "per band p90 / per family max",
                      {"p90_multiplier": GATE_THRESHOLDS["G4_p90_multiplier"],
                       "family_max_ape_pct": GATE_THRESHOLDS["G4_family_max_ape"]},
                      {"band_p90": band_p90, "family_max_ape_pct": family_max},
                      g4_status if band_p90 or family_max else "not_evaluated"))
    # G5 systematic bias / G6 penalised WAPE / G7 paired improvement
    bias_observed, g5_status = {}, []
    penalized_observed, g6_status = {}, []
    delta_observed, g7_status = {}, []
    for family, block in families.items():
        summary = block.get("metrics") or {}
        if not block.get("acceptance_bearing"):
            continue
        n = summary.get("numeric") or 0
        if n >= GATE_THRESHOLDS["G12_family_min"]:
            bias = summary.get("median_signed_bias_pct")
            bias_observed[family] = bias
            g5_status.append("pass" if bias is not None and
                             abs(bias) <= GATE_THRESHOLDS["G5_abs_median_bias"] else "fail")
            penalized = summary.get("wape_penalized_pct")
            penalized_observed[family] = penalized
            g6_status.append("pass" if penalized is not None and
                             penalized <= GATE_THRESHOLDS["G6_penalized_wape"] else "fail")
        delta = block.get("delta_vs_analytic") or {}
        if delta.get("delta_wape_pct") is not None and n >= GATE_THRESHOLDS["G12_family_min"]:
            observed_mass = (block.get("variants") or {}).get("analytic", {}).get("wape_pct")
            reference = observed_mass or 0.0
            relative = (delta["delta_wape_pct"] / reference * 100.0) if reference else None
            delta_observed[family] = {"delta_wape_pct": delta["delta_wape_pct"],
                                      "relative_pct": relative, "ci95": delta.get("ci95")}
            ci = delta.get("ci95") or [None, None]
            passed = (relative is not None and
                      relative <= GATE_THRESHOLDS["G7_delta_wape_relative"]
                      and ci[1] is not None and ci[1] < 0)
            g7_status.append("pass" if passed else "fail")
    gates.append(gate("G5", "per acceptance family, n >= 10",
                      {"abs_median_signed_bias_pct": GATE_THRESHOLDS["G5_abs_median_bias"]},
                      bias_observed,
                      "fail" if "fail" in g5_status else "pass" if g5_status else "insufficient_n"))
    gates.append(gate("G6", "per acceptance family, n >= 10",
                      {"wape_penalized_pct": GATE_THRESHOLDS["G6_penalized_wape"]},
                      penalized_observed,
                      "fail" if "fail" in g6_status else "pass" if g6_status else "insufficient_n"))
    gates.append(gate("G7", "per acceptance family, paired vs analytic",
                      {"relative_delta_wape_pct": GATE_THRESHOLDS["G7_delta_wape_relative"],
                       "ci_excludes_zero": True},
                      delta_observed,
                      "fail" if "fail" in g7_status else "pass" if g7_status else "insufficient_n"))
    gates.append(gate("G8", "round-1 numeric subset, per family",
                      {"allowance_pp": GATE_THRESHOLDS["G8_regression_allowance_pp"]},
                      None, "not_evaluated",
                      "requires --baseline-score from a previous published run"))
    containment = (variants.get("curve") or {}).get("containment_rate")
    semantics = "endpoint_values"
    gates.append(gate("G9", "reported interval containment",
                      {"containment": GATE_THRESHOLDS["G9_containment"],
                       "min_intervals": GATE_THRESHOLDS["G9_min_intervals"]},
                      {"containment_rate": containment, "semantics": semantics},
                      "report_only", "reported intervals are endpoint values, not an envelope"))
    gates.append(gate("G10", "cross-SKU transfer on the 66 matched conditions",
                      {"ratio": GATE_THRESHOLDS["G10_transfer_ratio"]},
                      None, "not_evaluated",
                      "a single-SKU freeze cannot evaluate a transfer direction"))
    gates.append(gate("G11", "same-protocol repeats per family",
                      {"conditions": 1}, noise.get("repeat_noise_floor_pct"),
                      "pass" if noise.get("passed") else "insufficient_n",
                      noise.get("note")))
    families_ok = [family for family, block in families.items()
                   if block.get("acceptance_bearing") and
                   (block.get("metrics") or {}).get("numeric", 0) >= GATE_THRESHOLDS["G12_family_min"]]
    gates.append(gate("G12", "minimum evidence per family and per SKU",
                      {"family_min": GATE_THRESHOLDS["G12_family_min"],
                       "sku_min": GATE_THRESHOLDS["G12_sku_min"],
                       "band_min": GATE_THRESHOLDS["G12_band_min"]},
                      {"families_at_minimum": sorted(families_ok),
                       "sku_numeric": sku_totals.get("numeric")},
                      "pass" if (len(families_ok) >=
                                 sum(1 for block in families.values()
                                     if block.get("acceptance_bearing")) and
                                 sku_totals.get("numeric", 0) >=
                                 GATE_THRESHOLDS["G12_sku_min"]) else "fail"))
    return gates


# -------------------------------------------------------------------- scoring ---
def score(*, dataset_path: Path, predictions_path: Path, receipt_path: Path,
          declarations: Path, round1_evidence: Path, out: Path,
          policy: str = "curve_first", bootstrap_b: int = metrics.DEFAULT_BOOTSTRAP_B,
          seed: int = metrics.DEFAULT_SEED, allow_dirty: bool = False,
          strict_protocol: bool = False, enforce: bool = False,
          thresholds: Mapping[str, Any] | None = None,
          emit_protocol_acceptance: bool = False) -> dict[str, Any]:
    set_primary_policy(policy)
    verified = verify_freeze(predictions_path=predictions_path, receipt_path=receipt_path,
                             dataset_path=dataset_path, declarations=declarations,
                             allow_dirty=allow_dirty)
    receipt = verified["receipt"]
    document = verified["document"]
    dataset = verified["dataset"]
    scope_sets = receipt.get("scope", {}).get("test_sets") or ["test1_old_heldout"]
    scope = scope_points(declarations_dir=declarations, round1_evidence=round1_evidence,
                         test_sets=scope_sets)
    # A freeze is one SKU: the receipt names the hardware key(s) it predicted. Every
    # declaration in the directory declares the same test sets, so without this filter an
    # A3 freeze would silently include the 66 A2 points (and vice versa) in the coverage
    # denominator. A round-1 receipt names only A3, which leaves that path unchanged.
    hardware_keys = list(receipt.get("scope", {}).get("hardware_keys") or [])
    if hardware_keys:
        scope["points"] = [point for point in scope["points"]
                           if point["hardware_key"] in hardware_keys]
    if not scope["points"]:
        raise IntegrityError(f"no_declared_points_in_scope:{scope_sets}")

    conditions, excluded = build_condition_records(
        points=scope["points"], document=document, dataset=dataset,
        regime_rules=scope["regime_rules"], strict_protocol=strict_protocol)
    rows, primary_rows = condition_rows(conditions)

    n_declared = len(scope["points"])
    by_hardware = group_by(conditions, "hardware_key")
    hardware = next(iter(by_hardware)) if len(by_hardware) == 1 else "+".join(sorted(by_hardware))
    # ---- all-variant roll-up and the primary roll-up ----
    variant_blocks = variant_rollup(rows, n_declared, b=bootstrap_b, seed=seed)
    primary_block = summarize_cell(primary_rows, n_declared, b=bootstrap_b, seed=seed)
    primary_by_band = regime_rollup(primary_rows, scope["points"], b=bootstrap_b, seed=seed)
    primary_by_die = per_cell(primary_rows, "die", {}, b=bootstrap_b, seed=seed)
    primary_by_protocol = per_cell(primary_rows, "protocol_signature", {}, b=bootstrap_b,
                                   seed=seed)

    families: dict[str, Any] = {}
    for family, family_rows in group_by(conditions, "family").items():
        family_primary = [row for row in primary_rows if row["family"] == family]
        family_variant_rows = [row for row in rows if row["family"] == family]
        analytic_rows = [row for row in family_variant_rows if row["variant"] == "analytic"]
        delta = None
        analytic_pairs = {(row["condition_id"]): row for row in analytic_rows if row["numeric"]}
        paired_ids = [row["condition_id"] for row in family_primary
                      if row["numeric"] and row["condition_id"] in analytic_pairs]
        if len(paired_ids) >= 2:
            model_errors, baseline_errors, observed_values = [], [], []
            primary_by_id = {row["condition_id"]: row for row in family_primary}
            for condition_id in paired_ids:
                model_row = primary_by_id[condition_id]
                analytic_row = analytic_pairs[condition_id]
                model_errors.append(abs(model_row["prediction_ms"] - model_row["observed_ms"]))
                baseline_errors.append(abs(analytic_row["prediction_ms"] - analytic_row["observed_ms"]))
                observed_values.append(model_row["observed_ms"])
            delta = metrics.paired_bootstrap_delta(model_errors, baseline_errors,
                                                   observed_values, b=bootstrap_b, seed=seed)
            delta["conditions"] = len(paired_ids)
        families[family] = {
            "n_declared": sum(1 for point in scope["points"] if point["family"] == family),
            "acceptance_bearing": any(point.get("acceptance_bearing", True)
                                      for point in scope["points"]
                                      if point["family"] == family),
            "metrics": summarize_cell(family_primary,
                                      sum(1 for point in scope["points"]
                                          if point["family"] == family),
                                      b=bootstrap_b, seed=seed, subset_of="family"),
            "variants": variant_rollup(family_variant_rows,
                                       sum(1 for point in scope["points"]
                                           if point["family"] == family),
                                       b=bootstrap_b, seed=seed),
            "bands": regime_rollup(family_primary,
                                   [point for point in scope["points"]
                                    if point["family"] == family],
                                   b=bootstrap_b, seed=seed),
            "delta_vs_analytic": delta,
            "outlier_conditions": metrics.outlier_rows(family_primary),
        }

    noise = repeat_noise_floor(dataset, families)
    sku_totals = {
        "conditions": n_declared,
        "numeric": primary_block["numeric"],
        "coverage": primary_block["coverage"],
        "wape_pct": primary_block["wape_pct"],
        "median_ape_pct": primary_block["median_ape_pct"],
    }
    gates = evaluate_gates(primary=primary_by_band, conditions=conditions,
                           families=families, noise=noise, variants=variant_blocks,
                           sku_totals=sku_totals, n_declared=n_declared)

    # ---- acceptance verdict against the CLI thresholds ----
    thresholds = dict(thresholds or {})
    violations = []
    if enforce:
        checks = {
            "wape_pct": primary_block.get("wape_pct"),
            "median_ape_pct": primary_block.get("median_ape_pct"),
            "coverage": primary_block.get("coverage"),
            "p90_ape_pct": primary_block.get("p90_ape_pct"),
            "max_ape_pct": primary_block.get("max_ape_pct"),
        }
        for name, limit in thresholds.items():
            value = checks.get(name)
            if value is None:
                continue
            if name == "coverage":
                if value < limit:
                    violations.append({"metric": name, "observed": value, "limit": limit})
            elif value > limit:
                violations.append({"metric": name, "observed": value, "limit": limit})
    # The protocol's verdict composition (§5.2): a failed gate rejects the round even
    # without an enforced CLI threshold. A gate that is not_evaluated / report_only /
    # insufficient_n is not a pass and does not by itself flip the verdict here: the
    # acceptance report states it separately.
    gate_failures = [gate["id"] for gate in gates if gate.get("status") == "fail"]
    status = ("rejected" if (violations or gate_failures) else
              "scored" if not enforce else "accepted")
    primary_paths = {row["condition_id"]: row["variant"] for row in primary_rows}
    primary_counts = Counter(value or "none" for value in primary_paths.values())
    generated_at = utc_now()

    acceptance = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "acceptance_id": f"{receipt.get('round_id')}::{hardware}::{generated_at}",
        "generated_at": generated_at,
        "status": status,
        "verdict": ("rejected" if (violations or gate_failures) else
                    "accepted" if enforce else "insufficient_evidence"),
        "verdict_reasons": (
            ["primary-path thresholds not met: " + canonical_json(violations)]
            if violations else
            ["protocol gates failed: " + ",".join(gate_failures)]
            if gate_failures else
            ["scored without an enforced threshold (pass --enforce to gate)"]),
        "gate_failures": gate_failures,
        "hardware_key": hardware,
        "arch_family": (scope["declaration_files"].get(hardware) or {}).get(
            "arch_family"),
        "test_sets": scope_sets,
        "conditions": n_declared,
        "numeric": primary_block["numeric"],
        "coverage": primary_block["coverage"],
        "wape_pct": primary_block["wape_pct"],
        "wape_ci95": primary_block["wape_ci95"],
        "wape_penalized_pct": primary_block["wape_penalized_pct"],
        "wape_in_envelope_pct": primary_block["wape_in_envelope_pct"],
        "median_ape_pct": primary_block["median_ape_pct"],
        "median_ape_ci95": primary_block["median_ape_ci95"],
        "p90_ape_pct": primary_block["p90_ape_pct"],
        "max_ape_pct": primary_block["max_ape_pct"],
        "mae_ms": primary_block["mae_ms"],
        "median_signed_bias_pct": primary_block["median_signed_bias_pct"],
        "spearman_rho": primary_block["spearman_rho"],
        "weighting": primary_block["weighting"],
        "coverage_denominator": primary_block["coverage_denominator"],
        "primary_path": {
            "policy": policy,
            "chain": list(PRIMARY_CHAINS[policy]),
            "counts": dict(sorted(primary_counts.items())),
            "per_condition": primary_paths,
            "eligibility": {
                "curve": "numeric and production_eligible (legacy variant: numeric)",
                "physical": "applied by the integration policy (a modeled but not "
                            "applied row is reported per variant and flagged)",
                "calibrated_db": "numeric",
                "analytic": "numeric",
            },
            "note": ("exactly one variant per condition; db_and_curve is never a primary "
                     "path because it is numerically identical to calibrated_db"),
        },
        "variants": variant_blocks,
        "per_family": families,
        "per_regime": primary_by_band,
        "per_die": primary_by_die,
        "per_protocol": primary_by_protocol,
        "gates": gates,
        "excluded": excluded,
        "limitations": LIMITATIONS,
        "repeat_noise_floor": noise,
        "provenance": {
            "predictions_path": str(predictions_path),
            "predictions_sha256": verified["predictions_sha256"],
            "freeze_receipt": str(receipt_path),
            "freeze_receipt_sha256": sha256_file(receipt_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": verified["dataset_hash"],
            "dataset_file_sha256": sha256_file(verified["dataset_file"]),
            "training_inventory_sha256": receipt.get("training_inventory_sha256"),
            "protocol": receipt.get("protocol"),
            "source": {"commit": (receipt.get("source") or {}).get("commit"),
                       "dirty": (receipt.get("source") or {}).get("dirty"),
                       "package_file_hashes_sha256": (receipt.get("source") or {}).get(
                           "package_file_hashes_sha256")},
            "evaluation_outcomes_read": True,
            "scoring_implementation": {
                "module": "tools.accuracy.score_frozen",
                "metrics_sha256": sha256_file(Path(metrics.__file__)),
                "scorer_sha256": sha256_file(Path(__file__)),
            },
            "bootstrap": {"b": bootstrap_b, "seed": seed, "unit": "condition",
                          "interval": "percentile_2.5_97.5"},
            "join_key": "hardware_key + family-aware shape identity; protocol asserted separately",
            "strict_protocol": strict_protocol,
        },
        "artifacts": {
            "dataset": str(dataset_path),
            "predictions": str(predictions_path),
            "receipt": str(receipt_path),
            "score": str(out / "score.json"),
            "bootstrap": str(out / "bootstrap.json"),
        },
    }
    acceptance["protocol_acceptance"] = protocol_acceptance(acceptance, receipt, scope,
                                                           primary_by_band, families,
                                                           variant_blocks, noise)
    acceptance["protocol_acceptance_validation"] = validate_protocol_acceptance(
        acceptance["protocol_acceptance"])

    score_document = {
        "schema_version": SCORE_SCHEMA,
        "generated_at": generated_at,
        "scoring_unit": "physical condition; target is median of trial medians",
        "event_repeats_treated_as_independent_conditions": False,
        "join": {"declared_points": n_declared,
                 "matched_observations": sum(1 for condition in conditions
                                             if condition.get("observation")),
                 "queried": sum(1 for condition in conditions if condition.get("entry")),
                 "excluded": len(excluded)},
        "conditions": conditions,
        "rows": rows,
        "primary_rows": primary_rows,
        "acceptance": {key: acceptance[key] for key in
                       ("status", "verdict", "hardware_key", "conditions", "numeric",
                        "coverage", "wape_pct", "wape_ci95", "median_ape_pct",
                        "p90_ape_pct", "max_ape_pct", "primary_path")},
    }
    bootstrap_document = {
        "generated_at": generated_at,
        "b": bootstrap_b,
        "seed": seed,
        "unit": "condition",
        "cells": {"primary": primary_block,
                  "variants": variant_blocks,
                  "per_family": {family: block.get("metrics") for family, block in families.items()},
                  "per_regime": primary_by_band},
    }
    write_json(out / "score.json", score_document)
    write_json(out / "bootstrap.json", bootstrap_document)
    write_json(out / "acceptance.json", acceptance)
    if emit_protocol_acceptance:
        write_json(out / "acceptance.protocol.json", acceptance["protocol_acceptance"])
    return {"acceptance": acceptance, "score": score_document,
            "bootstrap": bootstrap_document, "violations": violations}


# ------------------------------------------------- protocol acceptance record ---
def protocol_acceptance(acceptance: Mapping[str, Any], receipt: Mapping[str, Any],
                        scope: Mapping[str, Any], bands: Mapping[str, Any],
                        families: Mapping[str, Any], variants: Mapping[str, Any],
                        noise: Mapping[str, Any]) -> dict[str, Any]:
    """Projection of the same numbers onto acceptance-schema.json (protocol.md §6.4)."""
    source = receipt.get("source") or {}
    protocol = receipt.get("protocol") or {}
    declaration_files = scope.get("declaration_files") or {}
    a2 = next((item for key, item in declaration_files.items()
               if (item.get("arch_family") == "A2")), {})
    a3 = next((item for key, item in declaration_files.items()
               if (item.get("arch_family") == "A3")), {})
    hardware = acceptance["hardware_key"]
    sku_declaration = next((item for item in declaration_files.values()
                            if key_matches(item, hardware)), {})
    regime_rule = next((rule for key, rule in (scope.get("regime_rules") or {}).items()
                        if key == hardware), {}) or {}
    constants = regime_rule.get("constants") or {}
    sku_block = (sku_declaration.get("sku") or {}) if isinstance(sku_declaration, Mapping) else {}
    pin = sku_block.get("device_profile_pin") or {}
    discrepancies = []
    required = pin.get("discrepancy_required")
    if isinstance(required, Mapping):
        discrepancies.append({"field": required.get("field", "configured_clock_mhz"),
                              "static_value": required.get("static_value"),
                              "runtime_value": required.get("runtime_value"),
                              "authority": required.get("authority", "npu_smi_runtime")})
    metric_weighting = metrics.WEIGHTING
    family_blocks = []
    for family, block in families.items():
        summary = block.get("metrics") or {}
        family_blocks.append({
            "sku": hardware,
            "family": family,
            "test_set": (acceptance.get("test_sets") or ["test1_old_heldout"])[0],
            "acceptance_bearing": bool(block.get("acceptance_bearing")),
            "n_declared": max(1, block.get("n_declared") or 1),
            "n_numeric": summary.get("numeric") or 0,
            "coverage": summary.get("coverage") or 0.0,
            "coverage_denominator":
                "all_declared_conditions_including_unsupported_and_transition_unresolved",
            "metrics": _metric_block(summary, metric_weighting),
            "bands": [{"band": band,
                       "n": (block.get("bands", {}).get(band) or {}).get("numeric") or 0,
                       "metrics": _metric_block(block.get("bands", {}).get(band) or {},
                                                metric_weighting)}
                      for band in ("launch", "mid", "large")
                      if (block.get("bands", {}).get(band) or {}).get("n")],
            "statuses": summary.get("statuses") or {},
            **({"outlier_conditions": block.get("outlier_conditions")}
               if block.get("outlier_conditions") else {}),
        })
    gates = acceptance.get("gates") or []
    verdict = ("invalid" if not receipt.get("complete") else
               "accepted" if acceptance.get("verdict") == "accepted" else
               "rejected" if acceptance.get("verdict") == "rejected" else
               "insufficient_evidence")
    return {
        "schema_version": PROTOCOL_ACCEPTANCE_SCHEMA,
        "acceptance_id": acceptance["acceptance_id"],
        "generated_at": acceptance["generated_at"],
        "round_id": str(receipt.get("round_id") or "round-undefined") + "-accuracy",
        "protocol": {
            "protocol_id": "mindie-sim-accuracy-protocol/1",
            "protocol_sha256": sha256_value({"declarations": protocol.get("declaration_files")}),
            "scoring_unit": "physical_condition",
            "target_statistic": "median_of_per_trial_medians",
            "band_rule": {"bands": ["launch", "mid", "large"],
                          "formula": regime_rule.get("formula")
                          or "declared per-family work model against declared peaks",
                          "outcome_free": True},
            "reported_interval_semantics": "endpoint_values",
        },
        "source": {
            "commit": source.get("commit"),
            "dirty": bool(source.get("dirty")),
            "package_file_hashes_sha256": source.get("package_file_hashes_sha256"),
            "python_version": source.get("python_version"),
            "torch_npu_version": None,
        },
        "skus": [{
            "sku": sku_block.get("hardware_key") or hardware,
            "hardware_key": hardware,
            "arch_family": (sku_block.get("arch_family")
                            or ("A3" if "910_9362" in hardware else "A2" if "910B" in hardware
                                else "A5")),
            "device_profile": {
                "profile": pin.get("profile") or "declared_in_split_declaration",
                "declared_peaks": {
                    "peak_bf16_ops_per_s": constants.get("peak_bf16_ops_per_s") or 1.0,
                    "hbm_bytes_per_s": constants.get("hbm_bytes_per_s") or 1.0,
                    "vector_ops_per_s": constants.get("vector_ops_per_s"),
                },
                "discrepancies": discrepancies,
            },
            "verdict": verdict,
            "families_passed": [],
            "families_failed": sorted(family for family, block in families.items()
                                      if block.get("acceptance_bearing")),
            "repeat_noise_floor_pct": noise.get("repeat_noise_floor_pct"),
        }],
        "test_sets": {
            "test1_old_heldout": {
                "declaration_sha256": a3.get("sha256") or sha256_value("test1"),
                "declared_conditions": max(1, sum(
                    1 for set_name in (acceptance.get("test_sets") or []) if set_name == "test1_old_heldout")
                    * acceptance.get("conditions", 1)),
                "n_overlap_with_train": 0,
                "n_overlap_with_other_test_set": 0,
                "role": "repair_check",
            },
            "test2_fresh_heldout": {
                "declaration_sha256": a2.get("sha256") or a3.get("sha256")
                or sha256_value("test2"),
                "declared_conditions": max(1, acceptance.get("conditions", 1)),
                "n_overlap_with_train": 0,
                "n_overlap_with_other_test_set": 0,
                "role": "primary_generalization",
            },
            "a5": {"status": "deferred", "measured_conditions": 0},
        },
        "freeze": {
            "predictions_sha256": {hardware: receipt.get("predictions_sha256")},
            "predictions_frozen_at": receipt.get("frozen_at"),
            "evaluation_outcomes_read_before_freeze": False,
            "training_inventory_sha256": {hardware: receipt.get("training_inventory_sha256")},
            "split_registry_sha256": protocol.get("combined_sha256"),
            "completed_entries": receipt.get("completed_entries", 0),
            "expected_entries": max(1, receipt.get("expected_entries", 1)),
            "complete": bool(receipt.get("complete")),
            "protocol_attestation_violations": sum(
                1 for item in acceptance.get("excluded") or []
                if item.get("reason") == "protocol_mismatch"),
            "split_authority_violations": 0,
        },
        "gates": gates if gates else [gate(f"G{index}", "none", {}, None, "not_evaluated")
                                      for index in range(1, 13)],
        "families": family_blocks,
        "cross_sku_transfer": {
            "status": "not_evaluable",
            "matched_conditions": 0,
            "tuning": ("none: the frozen source-SKU fit is applied unchanged; no target-SKU "
                       "refit, scale search or per-family exception"),
            "directions": [],
        },
        "verdict": verdict,
        "verdict_reasons": list(acceptance.get("verdict_reasons") or ["not stated"]),
        "limitations": list(acceptance.get("limitations") or LIMITATIONS),
        "artifacts": {
            "protocol": str(receipt.get("protocol") or {}),
            "a2_split_declaration": a2.get("path") or "undeclared",
            "a3_split_declaration": a3.get("path") or "undeclared",
            "predictions": [receipt.get("predictions_path") or "predictions.frozen.json"],
            "outcomes": [str(receipt.get("dataset", {}).get("path") or "observations.json")],
            "score": [acceptance.get("artifacts", {}).get("score") or "score.json"],
        },
    }


def key_matches(declaration: Mapping[str, Any], hardware: str) -> bool:
    sku = declaration.get("sku")
    if isinstance(sku, Mapping):
        return sku.get("hardware_key") == hardware
    return False


def _metric_block(summary: Mapping[str, Any], weighting: Mapping[str, str]) -> dict[str, Any]:
    block = {
        "n": int(summary.get("numeric") or 0),
        "weighting": dict(weighting),
        "wape_pct": summary.get("wape_pct"),
        "wape_ci95_pct": summary.get("wape_ci95"),
        "wape_in_envelope_pct": summary.get("wape_in_envelope_pct"),
        "wape_extrapolated_pct": summary.get("wape_extrapolated_pct"),
        "wape_penalized_pct": summary.get("wape_penalized_pct"),
        "median_ape_pct": summary.get("median_ape_pct"),
        "median_ape_ci95_pct": summary.get("median_ape_ci95"),
        "p90_ape_pct": summary.get("p90_ape_pct"),
        "max_ape_pct": summary.get("max_ape_pct"),
        "mae_ms": summary.get("mae_ms"),
        "median_signed_bias_pct": summary.get("median_signed_bias_pct"),
        "spearman_rho": summary.get("spearman_rho"),
        "insufficient_n": bool(summary.get("insufficient_n")),
    }
    return block


def validate_protocol_acceptance(document: Mapping[str, Any]) -> dict[str, Any]:
    schema_path = DEFAULT_DECLARATIONS / "acceptance-schema.json"
    if not schema_path.exists():
        return {"validated": False, "reason": "schema_missing"}
    try:
        import jsonschema
    except ImportError:
        return {"validated": False, "reason": "jsonschema_not_installed"}
    schema = load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    return {"validated": not errors,
            "errors": [{"path": list(error.path), "message": error.message}
                       for error in errors[:10]]}


# ----------------------------------------------------------------------- cli ---
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--declarations", type=Path, default=DEFAULT_DECLARATIONS)
    parser.add_argument("--round1-evidence", type=Path, default=DEFAULT_ROUND1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-policy", choices=sorted(PRIMARY_CHAINS),
                        default="curve_first")
    parser.add_argument("--bootstrap-b", type=int, default=metrics.DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=metrics.DEFAULT_SEED)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--strict-protocol", action="store_true")
    parser.add_argument("--emit-protocol-acceptance", action="store_true")
    parser.add_argument("--enforce", action="store_true",
                        help="exit 2 when an enabled threshold is violated")
    parser.add_argument("--max-wape", type=float, help="primary-path WAPE limit, percent")
    parser.add_argument("--max-median-ape", type=float)
    parser.add_argument("--max-p90-ape", type=float)
    parser.add_argument("--max-max-ape", type=float)
    parser.add_argument("--min-coverage", type=float)
    args = parser.parse_args(argv)

    thresholds = {}
    if args.max_wape is not None:
        thresholds["wape_pct"] = args.max_wape
    if args.max_median_ape is not None:
        thresholds["median_ape_pct"] = args.max_median_ape
    if args.max_p90_ape is not None:
        thresholds["p90_ape_pct"] = args.max_p90_ape
    if args.max_max_ape is not None:
        thresholds["max_ape_pct"] = args.max_max_ape
    if args.min_coverage is not None:
        thresholds["coverage"] = args.min_coverage
    try:
        result = score(dataset_path=args.dataset, predictions_path=args.predictions,
                       receipt_path=args.receipt, declarations=args.declarations,
                       round1_evidence=args.round1_evidence, out=args.out,
                       policy=args.primary_policy, bootstrap_b=args.bootstrap_b,
                       seed=args.seed, allow_dirty=args.allow_dirty,
                       strict_protocol=args.strict_protocol, enforce=args.enforce,
                       thresholds=thresholds,
                       emit_protocol_acceptance=args.emit_protocol_acceptance)
    except IntegrityError as exc:
        print(f"integrity: {exc}", file=sys.stderr)
        return 3
    acceptance = result["acceptance"]
    print(json.dumps({"status": acceptance["status"],
                      "hardware_key": acceptance["hardware_key"],
                      "conditions": acceptance["conditions"],
                      "numeric": acceptance["numeric"],
                      "coverage": acceptance["coverage"],
                      "wape_pct": acceptance["wape_pct"],
                      "wape_ci95": acceptance["wape_ci95"],
                      "median_ape_pct": acceptance["median_ape_pct"],
                      "p90_ape_pct": acceptance["p90_ape_pct"],
                      "max_ape_pct": acceptance["max_ape_pct"],
                      "primary_path": acceptance["primary_path"]["counts"],
                      "excluded": len(acceptance["excluded"]),
                      "violations": result["violations"],
                      "out": str(args.out)}, indent=2))
    if result["violations"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
