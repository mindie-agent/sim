"""Campaign reporting, pending coverage and artifact writing.

[Role]
  Aggregate durable pending coverage, render coverage/status markdown and write
  the explicit manifest/inventory/conditions/unmeasured/config artifacts.

[Invariants]
  * Counts and examples never replace the complete conditions file.
  * Pending/unsupported entries retain reason and required backend.
  * No source SHA admission gate.
  * Writes only the explicit output directory.

[Timeouts & Locks]
  Pure local formatting/writes; no locks or devices.

[Side-Effect Policy]
  Writes only the explicit output directory when called.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .a3_operator_campaign_conditions import _canonical
from .a3_operator_campaign_sweeps import _axes_for_kind

def _pending_axes(condition: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Document concrete axes even when no executable backend exists."""
    kind = condition["shape_kind"]
    shape = dict(condition["shape"])
    axes = _axes_for_kind(kind)
    result = []
    for axis in axes:
        if axis not in shape:
            continue
        actual = max(1, int(shape[axis]))
        candidates = sorted({max(1, actual // 2), actual, actual * 2})
        result.append({"axis": axis, "actual": actual, "candidates": candidates,
                       "status": "unmeasured_no_verified_backend"})
    routing = (condition.get("features") or {}).get("routing") or {}
    if kind == "moe_grouped" and routing:
        for axis, value in (
            ("routing_tokens", routing.get("routing_tokens")),
            ("local_experts", routing.get("local_experts")),
            ("local_padded_tokens", routing.get("local_padded_tokens")),
        ):
            if value:
                actual = max(1, int(float(value)))
                result.append({"axis": axis, "actual": actual,
                               "candidates": sorted({max(1, actual // 2), actual, actual * 2}),
                               "status": "unmeasured_routing_condition"})
    return result


def _pending_summary(pending: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate durable unmeasured conditions without a 20 MB per-shape list."""
    groups: dict[str, dict[str, Any]] = {}
    for condition in pending:
        key = _canonical({
            "family": condition["family"], "recipe": condition["recipe"],
            "execution_kind": condition["execution_kind"],
            "backend": condition["backend"],
            "required_backend": condition["required_backend"],
            "reason": condition["reason"],
        })
        entry = groups.setdefault(key, {
            "family": condition["family"], "recipe": condition["recipe"],
            "execution_kind": condition["execution_kind"],
            "backend": condition["backend"], "status": condition["status"],
            "reason": condition["reason"], "required_backend": condition["required_backend"],
            "condition_count": 0, "models": set(), "shape_examples": [],
            "condition_keys": [],
        })
        entry["condition_count"] += 1
        for origin in condition["sources"].values():
            entry["models"].add(origin["model"])
        if len(entry.get("axis_examples", [])) < 8:
            entry.setdefault("axis_examples", []).extend(_pending_axes(condition))
        if len(entry["shape_examples"]) < 4:
            entry["shape_examples"].append({
                "shape": condition["shape"], "semantic_shape": condition["semantic_shape"],
                "phase": condition["phase"], "layout": condition["layout"],
                "input_dtype": condition["precision"].get("input_dtype"),
                "weight_dtype": condition["precision"].get("weight_dtype"),
            })
        if len(entry["condition_keys"]) < 4:
            entry["condition_keys"].append(condition["condition_key"])
    result = []
    for key in sorted(groups):
        entry = groups[key]
        entry["models"] = sorted(entry["models"])
        seen = set()
        unique = []
        for item in entry.get("axis_examples", []):
            item_key = _canonical(item)
            if item_key not in seen:
                seen.add(item_key)
                unique.append(item)
        entry["axis_examples"] = unique[:12]
        result.append(entry)
    return result


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def _runner_contract(config: CampaignConfig) -> dict[str, Any]:
    return {
        "worker_module": "mindie_sim.a3_operator_worker",
        "invoke": "python -m mindie_sim.a3_operator_worker --manifest manifest.json --results results.json",
        "dry_run": "python -m mindie_sim.a3_operator_worker --manifest manifest.json --results dry-run.json --dry-run",
        "device_owner": "sim_audit is the sole NPU load owner; this worker never acquires a lease, SSHes, selects a free device or mutates a service",
        "device_injection": "the coordinator injects the visible device; device defaults to npu:0",
        "protocol": dict(config.protocol),
        "correctness": "worker checks a CPU reference sample before timing where an output exists",
        "unsupported": "returns status=pending with reason/required_backend and no samples",
        "split": "split is frozen in this manifest before timing; analysis fits train only",
    }


def _coverage_markdown(manifest: Mapping[str, Any]) -> str:
    coverage = manifest["coverage"]
    lines = [
        "# Operator shape campaign coverage",
        "",
        f"Campaign: `{manifest['campaign_id']}`",
        f"Models: {len(coverage['all_models'])}; conditions: {coverage['condition_count']}; "
        f"inventory entries: {coverage['inventory_entries']}; selected requests: {coverage['selected_requests']}",
        f"Train/heldout selected: {coverage['selected_train']}/{coverage['selected_heldout']}; "
        f"extrapolation holdouts: {coverage['selected_extrapolation']}",
        f"Derived cases: {coverage['derived_cases']}/{coverage['total_cases']}",
        "",
        "## Model coverage",
        "",
        "| Model | Body layers | Emitted families | Conditions | Selected points | Selected families |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, model in manifest["models"].items():
        lines.append(
            f"| {key} | {model.get('body_layers')} | {len(model.get('families', []))} | "
            f"{model.get('condition_count', 0)} | {model.get('selected_points', 0)} | "
            f"{', '.join(model.get('selected_families', [])) or '-'} |"
        )
    lines += ["", "## Inventory status", ""]
    for status, count in coverage["inventory_status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines += ["", "## Pending/unsupported exact recipes", ""]
    if not manifest["pending_requests"]:
        lines.append("None.")
    else:
        for item in manifest["pending_requests"][:80]:
            lines.append(
                f"- `{item['family']}/{item['recipe']}` ({item['execution_kind']}): "
                f"{item['reason']} Required: {item.get('required_backend')}"
            )
        if len(manifest["pending_requests"]) > 80:
            lines.append(f"- ... {len(manifest['pending_requests']) - 80} more in manifest.pending_requests")
    failures = coverage.get("derivation_failures", [])
    lines += ["", "## Derivation failures", ""]
    if not failures:
        lines.append("None.")
    else:
        for item in failures[:40]:
            lines.append(
                f"- `{item.get('model')}` / `{item.get('topology')}` / {item.get('phase')} "
                f"{item.get('demand')}: {item.get('reason')}"
            )
        if len(failures) > 40:
            lines.append(f"- ... {len(failures) - 40} more in manifest.coverage.derivation_failures")
    lines += [
        "", "## Notes", "",
        "- Selected requests are frozen with train/heldout splits before any timing.",
        "- Pending cases are durable unmeasured coverage, not zero/failed samples.",
        "- Communication is excluded here; A5/AICPU/AIV expansion remains root/core-owned.",
        "",
    ]
    return "\n".join(lines)


def write_campaign(manifest: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": root / "manifest.json",
        "inventory": root / "inventory.json",
        "coverage": root / "coverage.md",
        "unmeasured": root / "unmeasured.json",
        "config": root / "campaign-config.json",
        "conditions": root / "conditions.json",
    }
    conditions = dict(manifest.get("conditions") or {})
    manifest_to_write = {key: value for key, value in manifest.items() if key != "conditions"}
    manifest_to_write["conditions_file"] = "conditions.json"
    manifest_to_write["conditions_count"] = len(conditions)
    paths["manifest"].write_text(json.dumps(manifest_to_write, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    compressed = base64.b64encode(gzip.compress(
        json.dumps(conditions, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        compresslevel=9,
    )).decode("ascii")
    paths["conditions"].write_text(json.dumps({
        "campaign_id": manifest["campaign_id"],
        "protocol": manifest.get("protocol"),
        "encoding": "gzip+base64",
        "condition_keys": sorted(conditions),
        "conditions_gzip_base64": compressed,
        "materialization": "load_conditions/materialize_conditions resolve this exact table; points are deterministic from each condition and manifest config",
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    paths["inventory"].write_text(json.dumps({
        "campaign_id": manifest["campaign_id"],
        "coverage": manifest["coverage"],
        "models": manifest["models"],
        "inventory": manifest["inventory"],
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    paths["coverage"].write_text(_coverage_markdown(manifest), encoding="utf-8")
    paths["unmeasured"].write_text(json.dumps({
        "campaign_id": manifest["campaign_id"],
        "pending_requests": manifest["pending_requests"],
        "pending_condition_total": len(manifest.get("pending_condition_index", [])),
        "pending_condition_file": "conditions.json",
        "pending_condition_index_head": manifest.get("pending_condition_index", [])[:64],
        "derivation_failures": manifest["coverage"].get("derivation_failures", []),
        "inventory_not_selected": [
            {key: item[key] for key in (
                "family", "recipe", "collection_status", "selected_points",
                "pending_variants", "required_backends",
            )}
            for item in manifest["inventory"]
            if item["collection_status"] in {
                "pending_backend", "executable_not_selected_budget", "unsupported",
            }
        ],
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    paths["config"].write_text(json.dumps(manifest["config"], indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}




def load_conditions(path: str | Path) -> dict[str, Any]:
    """Load the complete persisted condition table from conditions.json."""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("encoding") == "gzip+base64":
        raw = gzip.decompress(base64.b64decode(payload["conditions_gzip_base64"]))
        return json.loads(raw.decode("utf-8"))
    conditions = payload.get("conditions")
    if isinstance(conditions, dict):
        return conditions
    raise ValueError("conditions_file_invalid_or_missing_complete_table")
