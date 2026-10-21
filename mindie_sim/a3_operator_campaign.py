"""Canonical operator-condition extraction and shape-campaign manifest builder.

[Role]
  Expand representative full-model catalog iterations through the public
  ``derive_iteration_work`` lowering, retain every emitted op family/recipe and
  its model/layer origin, deduplicate identical physical measurement
  conditions, generate finite legal axis ladders/boundary probes, precommit
  train/heldout splits and an explicit collection budget, and write a reusable
  campaign manifest.  The campaign deliberately measures operators, not
  communication; A5 collection is a separate root-owned task.

[Invariants]
  * Model dimensions come from ``model_catalog`` and full model configs; this
    module never substitutes a smaller architecture to manufacture coverage.
  * A condition identity excludes model origin/source revision but includes
    family, recipe, execution kind, semantic/launch shape, dtype, layout,
    quantization/packing, feature/state and protocol.  Origins are retained.
  * Primitive standalone, actual fused kernel and reference decomposition are
    different execution kinds and are never pooled or relabelled.
  * Axis points are legal, finite and selected against an explicit budget.
    Simulator heuristic tiles are not asserted as hardware boundaries; only
    lowering padding, declared quantization blocks and declared KV blocks are
    used, and they are labelled as such.
  * Split assignment happens before any timing is available and is part of the
    manifest.  A physical shape already present in train is never also a
    holdout; duplicate train anchors win.
  * Pending/unsupported exact recipes remain concrete unmeasured cases with a
    reason and required backend; analytic work stays numeric.
  * No source SHA is an admission gate, and no module import touches a device.

[Timeouts & Locks]
  Pure in-memory lowering/arithmetic; loops are bounded by the configured
  model/topology/demand/sweep/budget selections.  No I/O locks or sleeps.

[Side-Effect Policy]
  Import and extraction write nothing.  The CLI writes only the explicit
  output directory when invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .a3_cost import PredictiveCostModel
from .a3_ops import OpSpec
from .a3_types import (
    A3InputError, MachineSpec, ParallelSpec, RuntimeSpec, align_up, dtype_bytes,
)
from .a3_workload import IterationPlan, RequestView, derive_iteration_work
from .model_catalog import CATALOG, load_model

from .a3_operator_campaign_conditions import (
    ALTERNATE_TOPOLOGIES, DEFAULT_DECODE_CASES, DEFAULT_MACHINE,
    DEFAULT_PREFILL_TOKENS, DEFAULT_PROTOCOL, LADDER_FACTORS,
    REFERENCE_TOPOLOGIES, _as_int, _as_positive, _bytes_for, _canonical,
    _build_condition, _condition_record, _execution_variants, _hash, _merge_origin,
    _origin_summary, _precision, _shape_kind, _shape_map, _source_summary, _tuple_ints,
)
from .a3_operator_campaign_machine import resolve_campaign_machine
from .a3_operator_campaign_report import (
    _coverage_markdown, _counts, _pending_summary, _runner_contract, write_campaign,
    load_conditions,
)
from .a3_operator_campaign_selection import _inventory, _select_conditions
from .a3_operator_campaign_sweeps import (
    _analytic_ms, _axes_for_kind, _axis_legal, _point_key, _point_priority,
    _request_from_point, _semantic_for_shape, assign_splits,
    condition_point_materialization, condition_points, materialize_conditions,
)

# These are actual lowering families, not a GEMM-only subset.  A family that
# is not explicitly recognised below becomes a pending/unsupported inventory
# entry rather than being silently dropped.
@dataclass(frozen=True)
class CampaignConfig:
    campaign_id: str = "a3-operator-shape-campaign-v1"
    model_keys: tuple[str, ...] = field(default_factory=lambda: tuple(CATALOG))
    model_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    prefill_tokens: tuple[int, ...] = DEFAULT_PREFILL_TOKENS
    decode_cases: tuple[dict[str, int], ...] = field(
        default_factory=lambda: tuple(dict(item) for item in DEFAULT_DECODE_CASES)
    )
    budget: int = 320
    max_working_set_bytes: int = 4 * 1024 ** 3
    max_points_per_condition: int = 8
    ladder_factors: tuple[float, ...] = LADDER_FACTORS
    protocol: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_PROTOCOL))
    machine: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_MACHINE))
    holdout_fraction: float = 0.34

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CampaignConfig":
        raw = dict(value or {})
        model_keys = raw.get("models", list(CATALOG))
        if isinstance(model_keys, Mapping):
            model_options = {str(k): dict(v) for k, v in model_keys.items()}
            model_keys = list(model_options)
        else:
            model_options = {str(k): dict(v) for k, v in dict(raw.get("model_options") or {}).items()}
            model_keys = [str(k) for k in model_keys]
        unknown = [key for key in model_keys if key not in CATALOG]
        if unknown:
            raise A3InputError(f"campaign_unknown_models:{','.join(sorted(unknown))}")
        prefill = tuple(_as_int(v, "campaign.prefill_tokens_invalid")
                        for v in raw.get("prefill_tokens", DEFAULT_PREFILL_TOKENS))
        decode_raw = raw.get("decode_cases", DEFAULT_DECODE_CASES)
        decode = []
        for item in decode_raw:
            batch = _as_int(item.get("batch"), "campaign.decode_batch_invalid")
            context = _as_int(item.get("context", 0), "campaign.decode_context_invalid", minimum=0)
            decode.append({"batch": batch, "context": context})
        factors = tuple(_as_positive(v, "campaign.ladder_factor_invalid")
                        for v in raw.get("ladder_factors", raw.get("ladder", LADDER_FACTORS)))
        protocol = {**DEFAULT_PROTOCOL, **dict(raw.get("protocol") or {})}
        for key in ("warmup", "iterations", "trials"):
            protocol[key] = _as_int(protocol.get(key), f"campaign.protocol.{key}_invalid")
        protocol["preallocated"] = bool(protocol.get("preallocated", True))
        return cls(
            campaign_id=str(raw.get("campaign_id", "a3-operator-shape-campaign-v1")),
            model_keys=tuple(model_keys),
            model_options=model_options,
            prefill_tokens=prefill,
            decode_cases=tuple(decode),
            budget=_as_int(raw.get("budget", 320), "campaign.budget_invalid"),
            max_working_set_bytes=_as_int(
                raw.get("max_working_set_bytes", 4 * 1024 ** 3),
                "campaign.max_working_set_invalid",
            ),
            max_points_per_condition=_as_int(
                raw.get("max_points_per_condition", 8), "campaign.max_points_invalid",
            ),
            ladder_factors=factors,
            protocol=protocol,
            machine=resolve_campaign_machine(raw.get("machine")),
            holdout_fraction=float(raw.get("holdout_fraction", 0.34)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CampaignConfig":
        return cls.from_mapping(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        models = {}
        for key in self.model_keys:
            options = dict(self.model_options.get(key) or {})
            topologies = options.get("topologies")
            if topologies is None:
                topologies = list(REFERENCE_TOPOLOGIES.get(key, ())) + list(
                    ALTERNATE_TOPOLOGIES.get(key, ())
                )
            models[key] = {
                **options,
                "topologies": [dict(item) for item in topologies],
                "prefill_tokens": list(options.get("prefill_tokens", self.prefill_tokens)),
                "decode_cases": [
                    dict(item) for item in options.get("decode_cases", self.decode_cases)
                ],
            }
        return {
            "campaign_id": self.campaign_id,
            "models": models,
            "prefill_tokens": list(self.prefill_tokens),
            "decode_cases": [dict(item) for item in self.decode_cases],
            "budget": self.budget,
            "max_working_set_bytes": self.max_working_set_bytes,
            "max_points_per_condition": self.max_points_per_condition,
            "ladder_factors": list(self.ladder_factors),
            "protocol": dict(self.protocol),
            "machine": dict(self.machine),
            "holdout_fraction": self.holdout_fraction,
        }

    def model(self, key: str) -> Any:
        return load_model(key)

    def topologies_for(self, key: str) -> tuple[dict[str, Any], ...]:
        options = self.model_options.get(key) or {}
        if options.get("topologies") is not None:
            return tuple(dict(item) for item in options["topologies"])
        return tuple(REFERENCE_TOPOLOGIES.get(key, ())) + tuple(
            ALTERNATE_TOPOLOGIES.get(key, ())
        )

    def prefill_for(self, key: str) -> tuple[int, ...]:
        options = self.model_options.get(key) or {}
        return tuple(_as_int(v, "campaign.model.prefill_invalid")
                     for v in options.get("prefill_tokens", self.prefill_tokens))

    def decode_for(self, key: str) -> tuple[dict[str, int], ...]:
        options = self.model_options.get(key) or {}
        raw = options.get("decode_cases", self.decode_cases)
        out = []
        for item in raw:
            out.append({
                "batch": _as_int(item.get("batch"), "campaign.model.decode_batch_invalid"),
                "context": _as_int(item.get("context", 0), "campaign.model.decode_context_invalid", minimum=0),
            })
        return tuple(out)

    def machine_spec(self) -> MachineSpec:
        return MachineSpec.from_mapping(dict(self.machine))


def _runtime_for(topology: Mapping[str, Any], config: CampaignConfig) -> RuntimeSpec:
    tp = _as_int(topology.get("tp", 1), "campaign.topology.tp_invalid")
    ep = _as_int(topology.get("ep", 1), "campaign.topology.ep_invalid")
    pp = _as_int(topology.get("pp", 1), "campaign.topology.pp_invalid")
    dp = _as_int(topology.get("dp", 1), "campaign.topology.dp_invalid")
    max_tokens = max(config.prefill_tokens + tuple(
        item["batch"] for item in config.decode_cases
    ))
    max_len = max(65536, max_tokens + max(item["context"] for item in config.decode_cases))
    return RuntimeSpec.from_mapping({
        "max_num_batched_tokens": max_tokens,
        "max_num_seqs": max(1, max(item["batch"] for item in config.decode_cases)),
        "max_model_len": max_len,
        "enable_chunked_prefill": True,
        "graph_mode": "eager",
        "parallel": {"tp": tp, "ep": ep, "pp": pp, "dp": dp},
    })


def _case_iter(config: CampaignConfig, key: str, model: Any, topology: Mapping[str, Any],
               runtime: RuntimeSpec):
    for tokens in config.prefill_for(key):
        yield {
            "phase": "prefill", "mode": "prefill", "prompt": tokens, "context": 0,
            "demand": f"prefill:{tokens}", "view": RequestView("r0", tokens, 0, True, "prefill"),
        }
    for item in config.decode_for(key):
        batch, context = item["batch"], item["context"]
        yield {
            "phase": "decode", "mode": "decode", "prompt": batch, "context": context,
            "demand": f"decode:{batch}x{context}",
            "view": RequestView("r0", batch, context, True, "decode"),
        }


def extract_conditions(config: CampaignConfig) -> dict[str, Any]:
    """Expand all configured full-model cases and deduplicate conditions."""
    machine = config.machine_spec()
    conditions: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    case_count = 0
    derived_cases = 0
    for key in config.model_keys:
        model = config.model(key)
        facts = dict(model.catalog_facts)
        models[key] = {
            "key": key, "name": model.name, "body_layers": model.layers,
            "hidden_size": model.hidden_size, "num_heads": model.num_heads,
            "attention": model.attention, "ffn": model.ffn,
            "num_experts": model.num_experts, "top_k": model.top_k,
            "quantization": model.quantization,
            "weight_dtype": model.weight_dtype,
            "activation_dtype": model.activation_dtype,
            "catalog_facts": {"profile": facts.get("profile"), "attention_types": facts.get("attention_types")},
            "topologies": [], "demand_cases": [], "families": [],
            "condition_count": 0, "selected_points": 0,
        }
        for topology in config.topologies_for(key):
            topo = dict(topology)
            parallel = ParallelSpec(
                dp=int(topo.get("dp", 1)), tp=int(topo.get("tp", 1)),
                ep=int(topo.get("ep", 1)), pp=int(topo.get("pp", 1)),
            )
            try:
                parallel.validate(model)
                machine.validate_placement(parallel)
                runtime = _runtime_for(topo, config)
            except Exception as exc:  # noqa: BLE001 - explicit invalid topology record
                failures.append({
                    "model": key, "topology": topo.get("name", ""),
                    "phase": None, "demand": None,
                    "reason": f"invalid_topology:{type(exc).__name__}:{exc}",
                })
                continue
            models[key]["topologies"].append({
                **topo, "parallel": parallel.to_dict(), "status": "valid",
            })
            for case in _case_iter(config, key, model, topo, runtime):
                case_count += 1
                plan = IterationPlan((case["view"],))
                try:
                    work = derive_iteration_work(plan, model, runtime, machine)
                except Exception as exc:  # noqa: BLE001 - durable extraction failure
                    failures.append({
                        "model": key, "topology": topo.get("name", ""),
                        "phase": case["phase"], "demand": case["demand"],
                        "reason": f"derive_iteration_work:{type(exc).__name__}:{exc}",
                    })
                    continue
                derived_cases += 1
                models[key]["demand_cases"].append({
                    "phase": case["phase"], "demand": case["demand"],
                    "topology": topo.get("name", ""), "status": "derived",
                    "valid_tokens": work.valid_tokens, "padded_tokens": work.padded_tokens,
                    "mode": work.mode, "recipe": work.recipe, "ops": len(work.ops),
                })
                for op in work.ops:
                    kind = _shape_kind(op)
                    for variant in _execution_variants(
                        op, _precision(op, model), kind, _shape_map(op, kind)
                    ):
                        condition = _build_condition(
                            op, model=model, model_key=key, topology=topo, case=case,
                            work=work, runtime=runtime, variant=variant, config=config,
                        )
                        current = conditions.get(condition["condition_key"])
                        if current is None:
                            conditions[condition["condition_key"]] = condition
                        else:
                            for origin in condition["sources"].values():
                                _merge_origin(current["sources"], origin)
        models[key]["condition_count"] = sum(
            1 for cond in conditions.values()
            if any(origin["model"] == key for origin in cond["sources"].values())
        )
        models[key]["families"] = sorted({
            cond["family"] for cond in conditions.values()
            if any(origin["model"] == key for origin in cond["sources"].values())
        })
    return {
        "conditions": conditions, "models": models, "failures": failures,
        "case_count": case_count, "derived_cases": derived_cases,
    }


def build_campaign(config: CampaignConfig) -> dict[str, Any]:
    extraction = extract_conditions(config)
    conditions = extraction["conditions"]
    machine = config.machine_spec()
    cost_model = PredictiveCostModel(machine)
    points_by_condition: dict[str, list[dict[str, Any]]] = {}
    all_points: list[dict[str, Any]] = []
    pending_conditions: list[dict[str, Any]] = []
    point_counts: dict[str, int] = {}
    rejections_by_condition: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(conditions):
        condition = conditions[key]
        materialization = condition_point_materialization(condition, config)
        point_counts[key] = len(materialization["points"])
        rejections_by_condition[key] = list(materialization["rejections"])
        if condition["status"] in {"pending_backend", "excluded_communication", "unsupported"}:
            pending_conditions.append(condition)
            continue
        points = materialization["points"]
        for point in points:
            point["condition_keys"] = [key]
            point["lineage"] = [f"{key}:{_canonical(point['shape'])}"]
            point["family"] = condition["family"]
            point["recipe"] = condition["recipe"]
            point["execution_kind"] = condition["execution_kind"]
        points_by_condition[key] = points
        all_points.extend(points)
    assigned = assign_splits(all_points)
    # Points are now globally split.  Put the decided split back on each
    # condition's point copy so the manifest stays explicit.
    by_key = {point["point_key"]: point for point in assigned}
    for key, points in points_by_condition.items():
        for index, point in enumerate(points):
            decided = by_key.get(point["point_key"])
            if decided is not None:
                points[index] = {
                    **point, "split": decided["split"],
                    "split_hint": decided["split_hint"],
                    "lineage": decided.get("lineage", point.get("lineage", [])),
                    "condition_keys": decided.get("condition_keys", point.get("condition_keys", [])),
                }
    selected, selection = _select_conditions(conditions, points_by_condition, config)
    requests = []
    actual_condition_keys = {
        str(key) for point in selected if point.get("actual")
        for key in point.get("condition_keys", ())
    }
    for point in selected:
        keys = [str(key) for key in point.get("condition_keys", ()) if key in conditions]
        primary_key = next(
            (key for key in keys if key in actual_condition_keys),
            keys[0] if keys else None,
        )
        if primary_key is None:
            continue
        condition = conditions[primary_key]
        lineage_keys = [str(key) for key in point.get("condition_keys", (primary_key,))]
        request = _request_from_point(
            point, config=config, condition=condition, machine=machine,
            cost_model=cost_model,
            source_cases=None,
        )
        request["lineage_condition_keys"] = ([primary_key] + [
            key for key in lineage_keys if key != primary_key
        ])[:9]
        request["lineage_condition_count"] = len(lineage_keys)
        request["equivalent_models"] = sorted({
            origin["model"]
            for key in lineage_keys if key in conditions
            for origin in conditions[key]["sources"].values()
        })
        requests.append(request)
    inventory = _inventory(conditions, extraction["models"], selected)
    selected_by_model: dict[str, set[str]] = {key: set() for key in config.model_keys}
    selected_families: dict[str, set[str]] = {key: set() for key in config.model_keys}
    for request in requests:
        for key in request.get("lineage_condition_keys", ()):
            lineage = conditions.get(str(key))
            if lineage is None:
                continue
            for origin in lineage["sources"].values():
                model_key = origin["model"]
                if model_key in selected_by_model:
                    selected_by_model[model_key].add(str(request["measurement_id"]))
                    selected_families[model_key].add(lineage["family"])
    for key, model in extraction["models"].items():
        model["selected_points"] = len(selected_by_model.get(key, set()))
        model["selected_families"] = sorted(selected_families.get(key, set()))
    all_families = sorted({cond["family"] for cond in conditions.values()})
    all_recipes = sorted({(cond["family"], cond["recipe"]) for cond in conditions.values()})
    pending_cases = _pending_summary(pending_conditions)
    pending_index = [
        {
            "condition_key": condition["condition_key"],
            "family": condition["family"], "recipe": condition["recipe"],
            "execution_kind": condition["execution_kind"], "status": condition["status"],
            "shape": condition["shape"], "phase": condition["phase"],
            "precision": {key: condition["precision"].get(key) for key in (
                "input_dtype", "weight_dtype", "output_dtype", "compute_dtype",
                "accumulation_dtype", "quantization", "weight_format",
                "weight_block", "layout")},
            "protocol": {key: condition["protocol"].get(key) for key in (
                "timer", "warmup", "iterations", "trials", "preallocated",
                "cache_policy")},
            "layout": condition["layout"],
            "quantization": condition["quantization"],
            "models": sorted({origin["model"] for origin in condition["sources"].values()}),
            "source_case_count": sum(
                int(_origin_summary(origin).get("count", 1))
                for origin in condition["sources"].values()
            ),
            "source_file": f"conditions.json#{condition['condition_key']}",
            "complete_record": "conditions.json (compressed complete table)",
            "reason": condition["reason"],
            "required_backend": condition["required_backend"],
        }
        for condition in sorted(
            pending_conditions,
            key=lambda c: (c["family"], c["recipe"], c["execution_kind"], _canonical(c["shape"])),
        )
    ]
    coverage = {
        "all_models": list(config.model_keys),
        "all_families": all_families,
        "all_family_recipes": [list(item) for item in all_recipes],
        "condition_count": len(conditions),
        "derived_cases": extraction["derived_cases"],
        "total_cases": extraction["case_count"],
        "derivation_failures": extraction["failures"],
        "inventory_entries": len(inventory),
        "selected_requests": len(requests),
        "selected_train": sum(1 for r in requests if r["split"] == "train"),
        "selected_heldout": sum(1 for r in requests if r["split"] == "heldout"),
        "selected_extrapolation": sum(1 for r in requests if r["extrapolation"]),
        "pending_conditions": len(pending_conditions),
        "pending_family_recipes": sorted({(p["family"], p["recipe"]) for p in pending_conditions}),
        "model_coverage": {
            key: {
                "emitted_families": model.get("families", []),
                "condition_count": model.get("condition_count", 0),
                "selected_points": model.get("selected_points", 0),
                "selected_families": model.get("selected_families", []),
            }
            for key, model in extraction["models"].items()
        },
        "inventory_status_counts": _counts(item["collection_status"] for item in inventory),
        "selection": selection,
        "budget": {
            "limit": config.budget, "selected_requests": len(requests),
            "omitted": selection.get("omitted", {}),
            "max_working_set_bytes": config.max_working_set_bytes,
            "max_points_per_condition": config.max_points_per_condition,
        },
        "split_policy": {
            "precommitted": True,
            "anchors": "actual demand and broad ladder anchors are train",
            "interpolation": "geometric midpoints x0/2, x0*2 are heldout",
            "boundaries": "T-2,T,T+2 and wider T/2,2T train; T-1,T+1 heldout",
            "extrapolation": "one outer x8 interval is heldout",
            "global_rule": "a physical point already in train is never holdout",
            "holdout_fraction_target": config.holdout_fraction,
        },
    }
    condition_records = {
        key: _condition_record(
            condition, point_count=point_counts.get(key, 0),
            rejections=rejections_by_condition.get(key, ()),
        )
        for key, condition in conditions.items()
    }
    manifest = {
        "campaign_id": config.campaign_id,
        "status": "ready",
        "config": config.to_dict(),
        "conditions": condition_records,
        "protocol": dict(config.protocol),
        "machine": machine.to_dict(),
        "hardware_key": machine.hardware_key or machine.name,
        "models": extraction["models"],
        "inventory": inventory,
        "requests": requests,
        "pending_requests": pending_cases,
        "pending_condition_index": pending_index,
        "pending_condition_index_policy": {
            "limit": None, "total": len(pending_conditions),
            "note": "complete key index; exact precision/protocol/source/state are in the compressed complete conditions.json table",
        },
        "coverage": coverage,
        "runner_contract": _runner_contract(config),
        "notes": [
            "conditions are extracted from the actual public derive_iteration_work lowering",
            "primitive, fused and reference-decomposition execution kinds are separate and never pooled",
            "no source SHA is an admission gate; provenance is attribution only",
            "A5 communication collection is root-owned and not part of this campaign",
        ],
    }
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise A3InputError("campaign_manifest_invalid:not_object")
    for key in ("campaign_id", "requests"):
        if key not in manifest:
            raise A3InputError(f"campaign_manifest_missing:{key}")
    requests = manifest.get("requests")
    if not isinstance(requests, list):
        raise A3InputError("campaign_manifest_invalid:requests")
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise A3InputError(f"campaign_manifest_invalid:requests[{index}]")
        for key in ("measurement_id", "request_key", "point", "condition"):
            if key not in request:
                raise A3InputError(f"campaign_manifest_missing:requests[{index}].{key}")
        if not isinstance(request.get("point"), Mapping) or "shape" not in request["point"]:
            raise A3InputError(f"campaign_manifest_invalid:requests[{index}].point.shape")
        if not isinstance(request.get("condition"), Mapping):
            raise A3InputError(f"campaign_manifest_invalid:requests[{index}].condition")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A3 operator shape-linearity campaign planner")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="extract conditions and write a campaign manifest")
    plan.add_argument("--config", default=None)
    plan.add_argument("--output-dir", default="examples/operator-campaign")
    plan.add_argument("--budget", type=int, default=None)
    plan.add_argument("--models", default="", help="comma-separated model key subset")
    plan.add_argument("--dump-conditions", action="store_true")
    plan.add_argument("--output", default=None, help="optional single manifest path")
    inspect = sub.add_parser("inspect", help="summarize an existing manifest")
    inspect.add_argument("--manifest", required=True)
    analyze = sub.add_parser("analyze", help="analyze raw worker results")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--results", required=True)
    analyze.add_argument("--output-dir", default="examples/operator-campaign/analysis")
    analyze.add_argument("--max-segments", type=int, default=3)
    materialize = sub.add_parser("materialize", help="materialize exact pending/selected conditions by key")
    materialize.add_argument("--config", default=None)
    materialize.add_argument("--conditions", default=None,
                             help="persisted conditions.json (skips re-extraction)")
    materialize.add_argument("--condition-key", action="append", default=[])
    materialize.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "materialize":
        config = CampaignConfig.from_file(args.config) if args.config else CampaignConfig()
        if args.conditions:
            conditions = load_conditions(args.conditions)
        else:
            conditions = extract_conditions(config)["conditions"]
        keys = [item for value in args.condition_key for item in str(value).split(",") if item]
        requests = materialize_conditions(conditions, keys, config)
        Path(args.output).expanduser().write_text(json.dumps({
            "campaign_id": config.campaign_id,
            "condition_keys": keys,
            "requests": requests,
        }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        sys.stdout.write(json.dumps({"status": "written", "condition_keys": len(keys),
                                     "requests": len(requests), "output": args.output},
                                    indent=1, sort_keys=True) + "\n")
        return 0
    if args.command == "inspect":
        manifest = load_manifest(args.manifest)
        payload = {
            "campaign_id": manifest["campaign_id"],
            "models": list(manifest["models"]),
            "requests": len(manifest["requests"]),
            "inventory": len(manifest["inventory"]),
            "pending": len(manifest["pending_requests"]),
            "coverage": manifest["coverage"],
        }
        sys.stdout.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        return 0
    if args.command == "analyze":
        from .a3_operator_curves import analyze_files
        result = analyze_files(
            args.manifest, args.results, output_dir=args.output_dir,
            max_segments=args.max_segments,
        )
        sys.stdout.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
        return 0
    raw = CampaignConfig.from_file(args.config) if args.config else CampaignConfig()
    payload = raw.to_dict()
    if args.models:
        payload["models"] = {
            key: value for key, value in payload["models"].items()
            if key in {item.strip() for item in args.models.split(",") if item.strip()}
        }
        payload["model_keys"] = list(payload["models"])
    if args.budget is not None:
        payload["budget"] = args.budget
    config = CampaignConfig.from_mapping(payload)
    manifest = build_campaign(config)
    if args.output:
        Path(args.output).expanduser().write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8",
        )
        written = {"manifest": str(args.output)}
    else:
        written = write_campaign(manifest, args.output_dir)
    if args.dump_conditions:
        # Optional full condition dump for root review; not required by the
        # worker and intentionally not part of the default repository example.
        out = Path(written["manifest"]).expanduser().parent / "conditions-full.json"
        extraction = extract_conditions(config)
        out.write_text(json.dumps({
            "conditions": list(extraction["conditions"].values()),
        }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        written["conditions"] = str(out)
    summary = {
        "status": "written", "campaign_id": manifest["campaign_id"],
        "models": len(manifest["models"]),
        "conditions": manifest["coverage"]["condition_count"],
        "selected_requests": len(manifest["requests"]),
        "train": manifest["coverage"]["selected_train"],
        "heldout": manifest["coverage"]["selected_heldout"],
        "pending": len(manifest["pending_requests"]),
        "files": written,
    }
    sys.stdout.write(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return 0


__all__ = [
    "CampaignConfig",
    "REFERENCE_TOPOLOGIES", "ALTERNATE_TOPOLOGIES", "DEFAULT_PROTOCOL",
    "extract_conditions", "condition_point_materialization", "condition_points",
    "materialize_conditions", "assign_splits",
    "build_campaign", "write_campaign", "load_manifest", "load_conditions", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
