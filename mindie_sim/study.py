"""Finite candidate studies built from the ordinary public modeling paths.

[Role] Connect operator screening, graph composition and serving constraints.
[Invariants] Costs keep their boundaries. Rankings compare declared equal work,
  never different models or cohort sizes. No observation is required to run.
  Invalid candidates remain failed records and never become zero-cost winners.
[Timeouts & Locks] Sequential evaluation; retain compact summaries, not graphs.
[Side-Effect Policy] Pure local simulation. No files, devices or external jobs.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
import json

from .prediction.api import predict
from .prediction.dispatch import cohorts
from .prediction.graph_analysis import graph_costs
from .prediction.operator_analysis import operator_costs, target_graphs
from .prediction.program import RecipeBook
from .prediction.structure import parse_case
from .schema.case import Software
from .schema.device_input import merge_device_inputs
from .schema.model import LayerPlan, ModelSpec
from .schema.parallel import ParallelSpec
from .coverage import model_coverage


def _fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _merge(base, updates):
    """Mappings merge recursively; sequences and scalars replace explicitly."""
    result = deepcopy(base)
    for key, value in updates.items():
        result[key] = (_merge(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else deepcopy(value))
    return result


def _model_identity(case):
    model = case.model.to_dict()
    model.pop("name", None)
    metadata = model.pop("metadata", {})
    if metadata.get("model_coverage") == "partial":
        model["coverage_identity"] = {key: metadata.get(key) for key in (
            "model_coverage", "unmodeled_features", "degraded_submodules",
            "unrecognized_config_fields", "unconsumed_config_fields",
            "declared_quantization", "source_model_type",
        )}
    return _fingerprint(model)


def _hf_model(model):
    return "layer_plans" not in model and any(
        key in model for key in ("model_type", "architectures", "text_config", "hf_config"))


def _hf_fields(base, updates):
    """Apply aliases using this native config's spelling, without rewriting it."""
    from .catalog.hf_config import ALIASES
    result = deepcopy(updates)
    for field, aliases in ALIASES.items():
        names = (field, *aliases)
        supplied = [name for name in names if name in updates]
        if not supplied:
            continue
        value = updates[supplied[0]]
        if any(updates[name] != value for name in supplied[1:]):
            raise ValueError("ambiguous_hf_override_aliases:" + field)
        for name in supplied:
            result.pop(name, None)
        # Native adapters may consume a family-specific spelling. Existing
        # synonymous declarations receive the same explicit update.
        targets = [name for name in names if name in base] or [field]
        for name in targets:
            result[name] = deepcopy(value)
    return result


def _hf_override(base, updates):
    """Model overrides describe the text body even when HF wraps its config."""
    base, updates = deepcopy(base), _hf_fields(base, updates)
    wrapper = next((key for key in ("hf_config", "text_config") if key in base), None)
    if wrapper is None:
        return _merge(base, updates)
    if wrapper == "hf_config":
        # Existing simulator overrides remain at their explicit outer boundary;
        # do not move a new value under an older outer value that still wins.
        normalized = {}
        for key, value in updates.items():
            canonical = next(iter(ModelSpec.canonical_fields({key: value})))
            normalized[canonical if canonical in base else key] = value
        updates = normalized
        if "quantization" in updates and "quant" not in updates:
            base.pop("quant", None)
        elif "quant" in updates and "quantization" not in updates:
            base.pop("quantization", None)
    outer = ({"name", "quant", "quantization", "metadata", "auxiliary", "layer_overrides"}
             if wrapper == "hf_config" else {"name", "vision_config", "architectures"})
    if wrapper == "hf_config":
        outer |= base.keys() - {wrapper}
    body = {key: value for key, value in updates.items() if key not in outer | {wrapper}}
    nested = updates.get(wrapper, {})
    overlap = body.keys() & nested.keys()
    if overlap:
        raise ValueError("ambiguous_hf_override_paths:" + ",".join(sorted(overlap)))
    result = _merge(base, {key: value for key, value in updates.items() if key in outer})
    result[wrapper] = _hf_override(base[wrapper], {**body, **nested})
    return result


def _case_input(raw, *, hf_model=False):
    """Canonicalize aliases before merging a partial candidate override."""
    result = deepcopy(raw)
    if "model" in result and not (hf_model or _hf_model(result["model"])):
        result["model"] = ModelSpec.canonical_fields(result["model"])
    if "machine" in result:
        machine = result["machine"]
        if "device" in machine:
            machine["device"] = merge_device_inputs(machine["device"])
        for field in ("devices", "dies"):
            if field in machine:
                machine[field] = [merge_device_inputs(device) for device in machine[field]]
    if "runtime" in result:
        runtime = result["runtime"]
        parallel = Software.parallel_mapping(runtime)
        for field in fields(ParallelSpec):
            runtime.pop(field.name, None)
        if parallel:
            runtime["parallel"] = parallel
    return result


def _override_case(raw, updates):
    hf = _hf_model(raw.get("model", {}))
    raw, updates = _case_input(raw), _case_input(updates, hf_model=hf)
    if hf and "model" in updates:
        updates["model"] = _hf_override(raw["model"], updates["model"])
    model, changes = raw.get("model", {}), updates.get("model", {})
    if model.get("layer_plans") and "layer_plans" not in changes:
        plan_fields = {field.name for field in fields(LayerPlan)} | {"layers", "mla"}
        shadowed = {key for key in changes.keys() & plan_fields if changes[key] != model.get(key)}
        if shadowed:
            raise ValueError("study_override_requires_explicit_layer_plans:" +
                             ",".join(sorted(shadowed)))
    result = _merge(raw, updates)
    if hf and "model" in updates:
        # _hf_override already resolved the complete model. A second merge
        # would resurrect a replaced representation declaration.
        result["model"] = updates["model"]
    return result


def _summarize_operators(rows, total, top):
    families, implementations, work = {}, {}, Counter()
    calibration = set()
    for row in rows:
        op, count = row["operator"], row["occurrences"]
        family_key = op["family"], op["recipe"]
        group = families.setdefault(family_key, {
            "family": op["family"], "recipe": op["recipe"],
            "occurrences": 0, "isolated_ms": 0.0,
        })
        group["occurrences"] += count
        group["isolated_ms"] += row["total_ms"]
        implementation = row["program"], op.get("source_boundary", "")
        entry = implementations.setdefault(implementation, {
            "program": implementation[0], "source_boundary": implementation[1],
            "occurrences": 0, "isolated_ms": 0.0,
        })
        entry["occurrences"] += count
        entry["isolated_ms"] += row["total_ms"]
        calibration.add(row["basis"]["calibration"])
        for field in ("flops", "vector_ops", "hbm_bytes", "link_bytes"):
            work[field] += op[field] * count
    ranked = sorted(families.values(), key=lambda row: -row["isolated_ms"])
    cumulative = 0.0
    for row in ranked:
        cumulative += row["isolated_ms"]
        row["share"] = row["isolated_ms"] / total if total else 0
        row["cumulative_share"] = cumulative / total if total else 0
    return {
        "operator_sum_ms": total,
        "operator_occurrences": sum(row["occurrences"] for row in rows),
        "physical_groups": len(rows), "work": dict(work),
        "family_count": len(ranked), "top_families": ranked[:top],
        "omitted_family_cost_ms": sum(row["isolated_ms"] for row in ranked[top:]),
        "implementations": list(implementations.values()),
        "calibration": sorted(calibration),
        "interpretation": "isolated_cost_concentration_not_graph_sensitivity_or_measured_error",
    }


def _probe(case, probe, *, level, primitives, top, include_submission):
    unknown = probe.keys() - {"id", "phase", "tokens", "context", "sequences",
                              "scope", "media_tokens_per_sequence", "apply_body_defaults"}
    if unknown:
        raise ValueError("unknown_study_probe_fields:" + ",".join(sorted(unknown)))
    if not {"phase", "tokens"} <= probe.keys():
        raise ValueError("study_probe_requires_phase_and_tokens")
    geometry = {key: probe[key] for key in
                ("phase", "tokens", "context", "sequences", "scope", "media_tokens_per_sequence", "apply_body_defaults")
                if key in probe}
    geometry.setdefault("sequences", 1)
    geometry.setdefault("context", 0)
    geometry.setdefault("scope", "model")
    graphs = target_graphs(case, **geometry)
    from .prediction.body_defaults import consume_target_declaration
    case = consume_target_declaration(case, graphs)
    book = RecipeBook.from_mapping(case.runtime.kernels)
    if level == "graph":
        report = graph_costs(
            case, graphs, recipes=book, primitives=primitives,
            external_inputs={"analysis_input": 0} if geometry["scope"] == "body" else None,
            include_submission=include_submission,
        )
        isolated = report["isolated"]
        rows, total = isolated["operators"], isolated["operator_sum_ms"]
        graph = {
            "duration_ms": report["duration_ms"], "boundary": report["boundary"],
            "replay_scope": report["replay_scope"],
            "bounds": {**report["bounds"], "precedence": {
                key: value for key, value in report["bounds"]["precedence"].items()
                if key != "operators"}},
            "resources": report["resources"],
            "execution": {key: report["execution"][key]
                          for key in ("live_tasks", "live_programs", "peak_live_tasks")},
        }
    else:
        isolated = operator_costs(case, graphs, recipes=book, primitives=primitives)
        rows, total = isolated["operators"], isolated["operator_sum_ms"]
        graph = None
    summary = _summarize_operators(rows, total, top)
    logical_work = {
        "model": _model_identity(case),
        "phase": geometry["phase"], "tokens_per_sequence": geometry["tokens"],
        "context": geometry["context"], "scope": geometry["scope"],
        "sequences_in_cohort": geometry["sequences"] * len(graphs),
        "media_tokens_per_sequence": geometry.get("media_tokens_per_sequence"),
        "routing": case.runtime.routing,
    }
    return {
        "status": "modeled", "geometry": geometry, "replicas": list(graphs),
        **model_coverage(case.model),
        "deployment_scope": "first_independent_dp_cohort",
        "physical_devices": len(graphs) * case.runtime.parallel.tp *
                            case.runtime.parallel.pp * case.runtime.parallel.pcp,
        "logical_work": logical_work, "workload_key": _fingerprint(logical_work),
        "operators": summary, "graph": graph,
        "operator_isolation": (None if graph is not None else
                               isolated["cross_dp_transport_isolation"]),
        "operator_isolation_signature": (None if graph is not None else
                                         isolated["transport_isolation_signature"]),
        "boundary": graph["boundary"] if graph else "sum_of_isolated_device_programs",
        "duration_ms": graph["duration_ms"] if graph else total,
    }


def _rankings(evaluations):
    groups = {}
    for row in evaluations:
        if row["status"] != "modeled":
            continue
        key = (row["probe"], row["workload_key"], row["boundary"],
               row["operator_isolation"], row["operator_isolation_signature"])
        groups.setdefault(key, []).append(row)
    result = []
    for (probe, workload, boundary, isolation, signature), members in groups.items():
        ordered = sorted(members, key=lambda row: row["duration_ms"])
        best = ordered[0]["duration_ms"]
        result.append({
            "probe": probe, "workload_key": workload, "boundary": boundary,
            "operator_isolation": isolation,
            "operator_isolation_signature": signature,
            "status": "ranked" if len(ordered) > 1 else "single_candidate",
            "candidates": [{
                "candidate": row["candidate"], "duration_ms": row["duration_ms"],
                "physical_devices": row["physical_devices"],
                "ratio_to_fastest": row["duration_ms"] / best if best else
                                    (1 if row["duration_ms"] == 0 else None),
            } for row in ordered],
        })
    return result


def study(candidates, probes, *, level="operators", serving=False, primitives=None,
          top_families=12, include_submission=False, max_evaluations=64):
    """Evaluate explicit candidate cases and named geometry probes sequentially.

    Candidates are {id, case, overrides?}; probes are {id, phase, tokens, ...}.
    Cases are mappings or parsed Case records. The CLI resolves file references.
    Serving uses each case's actual workload, separately from synthetic probes.
    """
    if level not in {"operators", "graph"}:
        raise ValueError("study_level_requires_operators_or_graph")
    if include_submission and level != "graph":
        raise ValueError("study_submission_requires_graph_level")
    if not candidates or not probes or top_families < 1:
        raise ValueError("study_requires_candidates_probes_and_positive_family_limit")
    if len(candidates) * (len(probes) + int(serving)) > max_evaluations:
        raise ValueError("study_evaluation_budget_exceeded")
    for items, kind in ((candidates, "candidate"), (probes, "probe")):
        if any(not isinstance(item.get("id"), str) or not item["id"] for item in items):
            raise ValueError(f"study_{kind}_requires_nonempty_id")
        if len({item["id"] for item in items}) != len(items):
            raise ValueError(f"duplicate_study_{kind}_id")
    evaluations, candidate_rows = [], []
    for candidate in candidates:
        name = candidate["id"]
        try:
            unknown = candidate.keys() - {"id", "case", "overrides"}
            if unknown:
                raise ValueError("unknown_study_candidate_fields:" + ",".join(sorted(unknown)))
            if "case" not in candidate:
                raise ValueError("study_candidate_requires_case")
            raw = candidate["case"]
            if candidate.get("overrides"):
                if not isinstance(raw, dict):
                    raise ValueError("study_overrides_require_mapping_case")
                raw = _override_case(raw, candidate["overrides"])
            case = parse_case(raw)
        except ValueError as exc:
            candidate_rows.append({"id": name, "status": "failed", "error": str(exc)})
            evaluations.extend({"candidate": name, "probe": probe["id"], "status": "failed",
                                "error": str(exc), "stage": "case"} for probe in probes)
            continue
        entry = {
            "id": name, "status": "modeled", "case_key": _fingerprint(case.to_dict()),
            **model_coverage(case.model),
            "parallel": case.runtime.parallel.to_dict(),
            "independent_cohorts": len(cohorts(case.runtime.parallel)),
        }
        for probe in probes:
            try:
                row = _probe(case, probe, level=level, primitives=primitives,
                             top=top_families, include_submission=include_submission)
            except ValueError as exc:
                row = {"status": "failed", "stage": level, "error": str(exc)}
                entry["status"] = "partial"
            evaluations.append({"candidate": name, "probe": probe["id"], **row})
        if serving:
            try:
                result = predict(case, primitives=primitives, trace_limit=0, event_limit=0)
                entry["serving"] = {
                    "status": result["status"], "boundary": result["metric_boundary"],
                    **model_coverage(case.model),
                    "metrics": result["metrics"], "memory": result["memory"],
                    "assumptions": result["assumptions"],
                    "failed_requests": [
                        {key: row[key] for key in ("request_id", "status", "reason")
                         if key in row} for row in result["requests"] if row["status"] != "served"],
                }
            except ValueError as exc:
                entry["status"] = "partial"
                entry["serving"] = {"status": "failed", "error": str(exc)}
        candidate_rows.append(entry)
    succeeded = sum(row["status"] == "modeled" for row in evaluations)
    failures = any(row["status"] != "modeled" for row in candidate_rows)
    return {
        "schema": "mindie-sim.study/1",
        "coverage_summary": {
            "complete_candidates": sum(row.get("model_coverage") == "complete" for row in candidate_rows),
            "partial_candidates": sum(row.get("model_coverage") == "partial" for row in candidate_rows),
        },
        "status": ("partial" if succeeded else "failed") if failures else "completed",
        "level": level, "evaluations": evaluations, "candidates": candidate_rows,
        "rankings": _rankings(evaluations),
        "summary": {"modeled": succeeded, "failed": len(evaluations) - succeeded,
                    "candidate_count": len(candidates), "probe_count": len(probes)},
        "interpretation": {
            "rankings": "predicted_order_within_equal_declared_work_not_measured_accuracy_or_global_optimum",
            "serving": "actual_case_requests_and_slo_separate_from_probe_device_costs",
            "validation": "independent_accuracy_report_no_measurement_admission_gate",
        },
    }
