"""Census dominant costs across unchanged model probes and explicit mappings.

[Role] Run bounded public model probes and retain compact before/after evidence.
[Invariants] Isolated sums, graph spans and measured errors are different.
  Comparisons require identical effective inputs except explicit dense/balanced
  matrix recipe axes. Routing changes are never ignored in a comparison.
  Source coverage is execution identity, not timing accuracy.
[Timeouts & Locks] One subprocess at a time, per-probe and total wall budgets.
[Side-Effect Policy] Reads cases/runtime files; writes the requested report only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time


SCHEMA = "mindie-sim.model-convergence/1"
MODELS = (
    ("qwen38", "qwen38-a5.json"),
    ("qwen35", "qwen35-a5.json"),
    ("qwen36", "qwen36-a5.json"),
    ("kimi-k3", "kimi-k3-a5.json"),
    ("glm53", "glm53-a5.json"),
    ("v41-dp4", "v41-a5-dp4.json"),
)
CONTROLS = (
    ("source-gqa-norm", "gqa-norm-control.json", 4, 17),
    ("source-grouped", "grouped-gemm-control.json", 4, 17),
    ("source-mla", "mla-decode-control.json", 4, 33),
    ("source-int8", "dynamic-int8-control.json", 3, 17),
)
BALANCED_RECIPES = ("balanced_grouped_gate_up", "balanced_grouped_down")


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(f"metadata_file_exceeds_50_mib:{path}")
    return json.loads(path.read_text())


def file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def representative_probes(prefill_tokens=256, decode_context=2048, model_prefill_tokens=None,
                          model_decode_sequences=None):
    """Keep actual model dimensions/deployments; shorten only the probe workload."""
    overrides = model_prefill_tokens or {}
    decode_sequences = model_decode_sequences or {}
    for values in (overrides, decode_sequences):
        if (set(values) - {name for name, _ in MODELS}
                or any(type(value) is not int or value < 1 for value in values.values())):
            raise ValueError("model_probe_overrides_require_known_models_and_positive_integers")
    probes = []
    for name, filename in MODELS:
        for phase, tokens, context in (
            ("prefill", overrides.get(name, prefill_tokens), 0), ("decode", 1, decode_context),
        ):
            probes.append({
                "id": f"{name}-{phase}", "case_id": name,
                "case_path": f"examples/cases/{filename}", "kind": "catalog_model",
                "geometry": {"phase": phase, "tokens": tokens, "context": context,
                             "sequences": decode_sequences.get(name, 1) if phase == "decode" else 1,
                             "scope": "model"},
            })
    for name, filename, prefill, context in CONTROLS:
        for phase, tokens, prefix in (("prefill", prefill, 0), ("decode", 1, context)):
            probes.append({
                "id": f"{name}-{phase}", "case_id": name,
                "case_path": f"examples/source-families/{filename}", "kind": "source_control",
                "geometry": {"phase": phase, "tokens": tokens, "context": prefix,
                             "sequences": 1, "scope": "model"},
            })
    return probes


def select_probes(probes, requested):
    """Keep explicit priority order when a finite campaign budget can expire."""
    if not requested:
        return probes
    selected = {}
    for name in requested:
        for probe in probes:
            if name in (probe["case_id"], probe["id"]):
                selected.setdefault(probe["id"], probe)
    return list(selected.values())


def worker_usage():
    """Report this worker's lifetime peak, including graph and isolated analysis."""
    try:
        import resource
    except ImportError:
        return {"status": "unavailable", "reason": "resource_module_not_available"}
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on Darwin and KiB on the other supported Unix hosts.
    unit_bytes = 1 if sys.platform == "darwin" else 1024
    return {
        "status": "measured", "method": "getrusage(RUSAGE_SELF).ru_maxrss",
        "raw_peak_rss": raw, "raw_unit_bytes": unit_bytes,
        "peak_rss_bytes": raw * unit_bytes,
        "peak_rss_mib": raw * unit_bytes / 1024**2,
    }


def _rank(groups, total):
    ranked = sorted(groups.values(), key=lambda row: (-row["isolated_ms"], row["family"]))
    cumulative = 0.0
    for row in ranked:
        cumulative += row["isolated_ms"]
        row["share"] = row["isolated_ms"] / total if total else 0.0
        row["cumulative_share"] = cumulative / total if total else 0.0
    return ranked


def summarize_costs(rows, total, *, top=12):
    families, implementations, signatures = {}, {}, []
    mappings, recorded_mappings, unrecorded_mappings = {}, 0, 0
    mapped_groups = 0
    semantic_work, serviced = Counter(), Counter()
    calibration, sources = set(), set()
    for row in rows:
        op, count = row["operator"], row["occurrences"]
        family = op["family"]
        key = family, op["recipe"]
        group = families.setdefault(key, {
            "family": family, "recipe": op["recipe"], "occurrences": 0, "isolated_ms": 0.0,
        })
        group["occurrences"] += count
        group["isolated_ms"] += row["total_ms"]
        metadata = op.get("metadata", {})
        boundary = op.get("source_boundary", "analytic")
        implementation = metadata.get("implementation", "")
        key = row["program"], boundary, fingerprint(implementation)
        group = implementations.setdefault(key, {
            "family": row["program"], "program": row["program"], "source_boundary": boundary,
            "implementation": implementation, "occurrences": 0, "isolated_ms": 0.0,
        })
        group["occurrences"] += count
        group["isolated_ms"] += row["total_ms"]
        for field in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "link_bytes"):
            semantic_work[field] += op.get(field, 0) * count
        for resource, amount in row["serviced"].items():
            serviced[resource] += amount * count
        calibration.add(row["basis"]["calibration"])
        sources.update(row["basis"].get("declared_parameter_sources", ()))
        mapping_keys = []
        for mapping in row["basis"].get("matrix_mappings", ()):
            detail = {key: value for key, value in mapping.items() if key != "invocations"}
            key = fingerprint(detail)
            entry = mappings.setdefault(key, {
                "key": key, "diagnostic": detail, "invocations": 0,
            })
            invocations = mapping["invocations"] * count
            entry["invocations"] += invocations
            recorded_mappings += invocations
            mapping_keys.append(key)
        mapped_groups += bool(mapping_keys)
        unrecorded_mappings += row["basis"].get("unrecorded_mapping_invocations", 0) * count
        signatures.append({
            "replica": row["replica"], "physical_id": row["physical_id"],
            "family": family, "recipe": op["recipe"], "program": row["program"],
            "source_boundary": boundary, "implementation": implementation,
            "local_shape": op["local_shape"], "semantic_shape": op["semantic_shape"],
            "dtype": op.get("dtype"), "layout": op.get("layout"),
            "matrix_groups": metadata.get("matrix_groups", 1),
            "matrix_mapping_domain": metadata.get("matrix_mapping_domain", ""),
            "occurrences": count, "per_call_ms": row["duration_ms"],
            "isolated_ms": row["total_ms"], "matrix_mapping_keys": mapping_keys,
        })
    ranked = _rank(families, total)
    return {
        "operator_sum_ms": total, "operator_occurrences": sum(r["occurrences"] for r in rows),
        "physical_groups": len(rows), "families": ranked,
        "top_signatures": sorted(signatures, key=lambda r: -r["isolated_ms"])[:top],
        "programs": _rank(implementations, total),
        "semantic_work": dict(semantic_work), "executed_service": dict(serviced),
        "calibration": sorted(calibration), "declared_parameter_sources": sorted(sources),
        "matrix_mappings": list(mappings.values()),
        "matrix_mapping_coverage": {
            "physical_groups_with_mapping": mapped_groups,
            "recorded_invocations": recorded_mappings,
            "unrecorded_invocations": unrecorded_mappings,
        },
        "remaining_live_tasks": sum(row["live_tasks"] for row in rows),
        "remaining_allocations": sum(sum(row["remaining_buffers"].values()) for row in rows),
        "interpretation": (
            "Occurrence-weighted isolated programs in one independent DP cohort. "
            "Source boundaries describe selection, not accuracy; service units follow "
            "the declared resource bindings and are not interchangeable with semantic work."
        ),
    }


def _runtime_identity():
    import mindie_sim

    root = Path(mindie_sim.__file__).resolve().parent
    files = sorted(p for p in root.rglob("*") if p.suffix in {".py", ".json"} and p.is_file())
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(file_hash(path).encode())
    return {
        "import_path": str(root), "version": mindie_sim.__version__,
        "package_content_sha256": digest.hexdigest(), "hashed_files": len(files),
        "matrix_sha256": file_hash(root / "prediction/matrix.py"),
    }


def run_probe(spec):
    """One public frontend/operator or graph execution; no measurements are read."""
    from mindie_sim import graph_costs, operator_costs, target_graphs
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.structure import parse_case

    started = time.perf_counter()
    raw = deepcopy(spec["case"])
    if spec.get("routing") is not None:
        raw.setdefault("runtime", {})["routing"] = deepcopy(spec["routing"])
    if spec.get("matmul") is not None:
        raw.setdefault("runtime", {}).setdefault("kernels", {}).setdefault(
            "programs", {})["matmul"] = spec["matmul"]
    if spec.get("balanced_matmul") is not None:
        overrides = raw.setdefault("runtime", {}).setdefault("kernels", {}).setdefault(
            "recipes", {})
        overrides.update({name: deepcopy(spec["balanced_matmul"]) for name in BALANCED_RECIPES})
    case = parse_case(raw)
    normalized = case.to_dict()
    without_matmul = deepcopy(normalized)
    programs = without_matmul["runtime"]["kernels"].get("programs", {})
    selected_matmul = programs.pop("matmul", None)
    if not programs:
        without_matmul["runtime"]["kernels"].pop("programs", None)
    without_mapping = deepcopy(without_matmul)
    overrides = without_mapping["runtime"]["kernels"].get("recipes", {})
    selected_balanced = {name: overrides.pop(name, None) for name in BALANCED_RECIPES}
    if not overrides:
        without_mapping["runtime"]["kernels"].pop("recipes", None)
    graphs = target_graphs(case, **spec["geometry"])
    recipes = RecipeBook.from_mapping(case.runtime.kernels)
    expanded = sum(len(recipes.expand(ops)) for ops in graphs.values())
    if expanded > spec["max_operators"]:
        raise ValueError(f"expanded_operator_budget_exceeded:{expanded}>{spec['max_operators']}")
    graph = None
    if spec["level"] == "graph":
        result = graph_costs(case, graphs, recipes=recipes, trace_limit=0)
        rows = result["isolated"]["operators"]
        total = result["isolated"]["operator_sum_ms"]
        graph = {
            "duration_ms": result["duration_ms"], "boundary": result["boundary"],
            "combined_lower_bound_ms": result["bounds"]["combined_ms"],
            "live_tasks": result["execution"]["live_tasks"],
            "live_programs": result["execution"]["live_programs"],
            "completed_tasks": result["execution"]["completed_tasks"],
            "peak_live_tasks": result["execution"]["peak_live_tasks"],
            "live_allocated": result["execution"]["live_allocated"],
            "matrix_mappings": result["parameters"].get("matrix_mappings", []),
            "unrecorded_mapping_invocations": result["parameters"].get(
                "unrecorded_mapping_invocations", 0),
        }
    else:
        result = operator_costs(case, graphs, recipes=recipes)
        rows, total = result["operators"], result["operator_sum_ms"]
    return {
        "status": "modeled", "runtime": _runtime_identity(),
        "effective_case_key": fingerprint(normalized),
        "case_without_matmul_key": fingerprint(without_matmul), "matmul": selected_matmul,
        "case_without_mapping_key": fingerprint(without_mapping),
        "routing": case.runtime.routing,
        "balanced_recipes": {
            name: {"declared": selected_balanced[name],
                   "resolved_program": type(recipes.recipes[name]).__name__}
            for name in BALANCED_RECIPES if name in recipes.recipes
        },
        "replicas": list(graphs), "deployment_scope": "first_independent_dp_cohort",
        "parallel": case.runtime.parallel.to_dict(),
        "boundary": "sum_of_isolated_device_programs",
        "costs": summarize_costs(rows, total, top=spec["top"]),
        "graph": graph, "wall_seconds": time.perf_counter() - started,
        "worker_usage": worker_usage(),
    }


def compare_runs(current, reference):
    """Compare equal probes; only dense and balanced matrix recipes may vary."""
    prior = {row["id"]: row for row in reference["probes"]}
    current_ids = {row["id"] for row in current["probes"]}
    comparisons = []
    for row in current["probes"]:
        before = prior.get(row["id"])
        result = {"probe": row["id"]}
        if not before or row["status"] != "modeled" or before["status"] != "modeled":
            comparisons.append({**result, "status": "unpaired",
                                "reason": "missing_or_unmodeled_probe"})
            continue
        current_key = row.get("case_without_mapping_key", row["case_without_matmul_key"])
        reference_key = before.get("case_without_mapping_key", before["case_without_matmul_key"])
        if row["input_key"] != before["input_key"] or current_key != reference_key:
            comparisons.append({**result, "status": "unpaired",
                                "reason": "effective_input_changed_outside_matrix_recipes"})
            continue
        old, new = before["costs"]["operator_sum_ms"], row["costs"]["operator_sum_ms"]
        old_families = {(r["family"], r["recipe"]): r for r in before["costs"]["families"]}
        changes = []
        for family in row["costs"]["families"]:
            previous = old_families.pop((family["family"], family["recipe"]), None)
            changes.append({
                "family": family["family"], "recipe": family["recipe"],
                "before_ms": previous["isolated_ms"] if previous else None,
                "after_ms": family["isolated_ms"],
                "difference_ms": family["isolated_ms"] - previous["isolated_ms"]
                if previous else None,
            })
        changes.extend({"family": f["family"], "recipe": f["recipe"],
                        "before_ms": f["isolated_ms"], "after_ms": None, "difference_ms": None}
                       for f in old_families.values())
        old_work, new_work = before["costs"]["semantic_work"], row["costs"]["semantic_work"]
        work_delta = {key: new_work.get(key, 0) - old_work.get(key, 0)
                      for key in old_work.keys() | new_work.keys()
                      if old_work.get(key, 0) != new_work.get(key, 0)}
        old_service = before["costs"]["executed_service"]
        new_service = row["costs"]["executed_service"]
        service_keys = old_service.keys() | new_service.keys()
        service_delta = {key: new_service.get(key, 0) - old_service.get(key, 0)
                         for key in service_keys
                         if old_service.get(key, 0) != new_service.get(key, 0)}
        comparisons.append({
            **result, "status": "compared",
            "configuration": (
                "identical" if row["effective_case_key"] == before["effective_case_key"]
                else "explicit_matrix_recipe_axes"
                if row.get("balanced_recipes") != before.get("balanced_recipes")
                else "explicit_matmul_recipe_axis"
            ),
            "recipe_axes": {
                "matmul": {"before": before.get("matmul"), "after": row.get("matmul")},
                "balanced": {"before": before.get("balanced_recipes"),
                             "after": row.get("balanced_recipes")},
            },
            "before_ms": old, "after_ms": new, "difference_ms": new - old,
            "relative_change": new / old - 1 if old else None,
            "semantic_work_unchanged": all(math.isclose(
                old_work.get(key, 0), new_work.get(key, 0), rel_tol=1e-12, abs_tol=1e-9)
                for key in old_work.keys() | new_work.keys()),
            "semantic_work_delta": work_delta,
            "executed_service_unchanged": all(math.isclose(
                old_service.get(key, 0), new_service.get(key, 0), rel_tol=1e-12, abs_tol=1e-9)
                for key in service_keys),
            "executed_service_delta": service_delta,
            "executed_service_max_relative_delta": max((
                abs(new_service.get(key, 0) - old_service.get(key, 0))
                / max(1, abs(old_service.get(key, 0))) for key in service_keys), default=0),
            "family_changes": sorted(changes, key=lambda r: -abs(r["difference_ms"] or 0)),
        })
    missing = [probe for probe in prior if probe not in current_ids]
    comparisons.extend({"probe": probe, "status": "unpaired", "reason": "missing_in_current"}
                       for probe in missing)
    return {
        "boundary": "sum_of_isolated_device_programs", "probes": comparisons,
        "coverage": {
            "reference_probes": len(prior), "current_probes": len(current_ids),
            "compared": sum(row["status"] == "compared" for row in comparisons),
            "missing_in_current": missing,
            "missing_in_reference": [row["id"] for row in current["probes"]
                                     if row["id"] not in prior],
        },
        "work_comparison_roundoff": {"relative": 1e-12, "absolute": 1e-9},
        "interpretation": "Prediction change for equal probes, not measured improvement or cross-model ranking.",
    }


def retained_gemm_impact(prediction, observation_report):
    """Project retained kernel medians onto uniquely matching GEMM occurrences."""
    for field in ("tokens_per_sequence", "sequences"):
        if prediction[field] != observation_report[field]:
            raise ValueError(f"prior_impact_geometry_mismatch:{field}")
    rows, unmatched = [], []
    def contract(op):
        return op["recipe"], tuple(op["local_shape"]), op["dtype"]
    model_contracts = Counter(contract(row["operator"]) for row in prediction["operators"]
                              if row["program"] == "TiledMatmul")
    for row in prediction["operators"]:
        if row["program"] != "TiledMatmul":
            continue
        op = row["operator"]
        candidates = [pair for pair in observation_report["comparisons"]
                      if (pair["recipe"], pair["shape"], pair["dtype"]) ==
                      (op["recipe"], op["local_shape"], op["dtype"])]
        unique = len(candidates) == 1 and model_contracts[contract(op)] == 1
        pair = candidates[0] if unique else None
        if not unique or not (
            row["occurrences"] == pair["contract_model_calls"]
            and pair["capture_calls"] == pair["contract_capture_calls"]
        ):
            unmatched.append({
                "recipe": op["recipe"], "shape": op["local_shape"],
                "predicted_ms": row["total_ms"], "candidate_count": len(candidates),
            })
            continue
        observed = pair["observed_median_us"] * row["occurrences"] / 1000
        difference = row["total_ms"] - observed
        rows.append({
            "recipe": op["recipe"], "shape": op["local_shape"], "dtype": op["dtype"],
            "occurrences": row["occurrences"], "kernel": pair["kernel"],
            "source_lines": pair["source_lines"], "match": pair["match"],
            "predicted_per_call_ms": row["duration_ms"],
            "observed_median_per_call_ms": pair["observed_median_us"] / 1000,
            "predicted_contribution_ms": row["total_ms"],
            "signed_residual_ms": difference, "absolute_residual_ms": abs(difference),
            "fraction_of_predicted_model_sum": abs(difference) / prediction["operator_sum_ms"],
        })
    ranked = sorted(rows, key=lambda row: -row["absolute_residual_ms"])
    return {
        "scope": "retained_tensor_contract_gemm_priority_not_accuracy",
        "prediction_boundary": prediction["boundary"],
        "reference_isolated_sum_ms": prediction["operator_sum_ms"],
        "capture_source": observation_report["source"],
        "capture_line_window": observation_report["line_window"],
        "ranked_pairs": ranked, "unpaired_or_ambiguous_gemm": unmatched,
        "top_four_signed_residual_ms": sum(row["signed_residual_ms"] for row in ranked[:4]),
        "top_four_absolute_residual_ms": sum(row["absolute_residual_ms"] for row in ranked[:4]),
        "interpretation": (
            "Existing analytical predictions and retained kernel medians; no fit or new "
            "prediction. Tensor-contract matching does not establish binary/dispatch identity, "
            "heldout roles, whole-model accuracy or critical-path sensitivity."
        ),
    }


def run_census(args):
    model_prefill = json.loads(args.model_prefill_tokens) if args.model_prefill_tokens else {}
    model_decode = json.loads(args.model_decode_sequences) if args.model_decode_sequences else {}
    probes = representative_probes(args.prefill_tokens, args.decode_context, model_prefill,
                                  model_decode)
    probes = select_probes(probes, args.only)
    if not probes or len(probes) > args.max_probes:
        raise ValueError("selected_probe_count_outside_budget")
    matmul = json.loads(args.matmul) if args.matmul else None
    balanced_matmul = json.loads(args.balanced_matmul) if args.balanced_matmul else None
    routing = json.loads(args.routing) if args.routing else None
    started, rows = time.monotonic(), []
    for probe in probes:
        path = args.root / probe["case_path"]
        raw = read_json(path)
        input_case = deepcopy(raw)
        if routing is not None:
            input_case.setdefault("runtime", {})["routing"] = routing
        input_key = fingerprint({"case": input_case, "geometry": probe["geometry"],
                                 "level": args.level})
        row = {**probe, "case_sha256": file_hash(path), "input_key": input_key}
        remaining = args.total_seconds - (time.monotonic() - started)
        if remaining <= 0:
            row.update(status="not_started", reason="total_wall_budget_exhausted")
        else:
            spec = {
                "case": raw, "geometry": probe["geometry"], "matmul": matmul,
                "routing": routing, "balanced_matmul": balanced_matmul,
                "level": args.level, "top": args.top, "max_operators": args.max_operators,
            }
            command = [str(args.python), "-I", str(Path(__file__).resolve()), "--worker"]
            if args.source_root:
                command += ["--source-root", str(args.source_root)]
            worker_started = time.monotonic()
            try:
                completed = subprocess.run(
                    command, input=json.dumps(spec), text=True, capture_output=True,
                    timeout=min(args.probe_seconds, remaining), check=False)
                if completed.returncode == 0:
                    row.update(json.loads(completed.stdout))
                else:
                    row.update(status="failed", exit_code=completed.returncode,
                               reason=completed.stderr[-4000:])
            except subprocess.TimeoutExpired:
                row.update(status="timeout", reason="probe_or_total_wall_budget_exhausted")
            row["worker_wall_seconds"] = time.monotonic() - worker_started
        rows.append(row)
        print(json.dumps({"id": row["id"], "status": row["status"],
                          "operator_sum_ms": row.get("costs", {}).get("operator_sum_ms"),
                          "wall_seconds": row.get("wall_seconds")}), flush=True)
    runtimes = {r["runtime"]["package_content_sha256"] for r in rows if r["status"] == "modeled"}
    result = {
        "schema": SCHEMA, "scope": "predicted_model_cost_and_frontend_coverage_not_accuracy",
        "runtime_label": args.runtime_label, "driver_sha256": file_hash(__file__),
        "level": args.level,
        "configuration": {
            "routing_override": routing, "matmul_override": matmul,
            "balanced_matmul_override": balanced_matmul,
            "prefill_tokens": args.prefill_tokens, "model_prefill_tokens": model_prefill,
            "decode_context": args.decode_context, "model_decode_sequences": model_decode,
        },
        "budgets": {"probe_seconds": args.probe_seconds, "total_seconds": args.total_seconds,
                    "max_probes": args.max_probes, "max_operators": args.max_operators},
        "status": "completed" if all(r["status"] == "modeled" for r in rows) and len(runtimes) == 1
        else "partial",
        "runtime_consistent": len(runtimes) <= 1, "probes": rows,
        "summary": dict(Counter(r["status"] for r in rows)),
        "wall_seconds": time.monotonic() - started,
        "accuracy": {"status": "unassessed", "observations_consumed": 0},
    }
    if args.reference:
        reference = read_json(args.reference)
        result["reference"] = {"path": str(args.reference), "sha256": file_hash(args.reference)}
        result["comparison"] = compare_runs(result, reference)
    if args.prior_operators:
        result["prior_gemm_impact"] = {
            "inputs": [{"path": str(path), "sha256": file_hash(path)}
                       for path in (args.prior_operators, args.prior_comparison)],
            **retained_gemm_impact(read_json(args.prior_operators),
                                  read_json(args.prior_comparison)),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--runtime-label", default="explicit-runtime")
    parser.add_argument("--prefill-tokens", type=int, default=256)
    parser.add_argument("--model-prefill-tokens",
                        help="JSON model-id to prefill-token count overrides")
    parser.add_argument("--model-decode-sequences",
                        help="JSON model-id to simultaneous decode-sequence count overrides")
    parser.add_argument("--decode-context", type=int, default=2048)
    parser.add_argument("--level", choices=("operators", "graph"), default="operators")
    parser.add_argument("--only", nargs="+", help="Model or probe IDs in execution priority order")
    parser.add_argument("--matmul", help="Explicit JSON recipe for runtime.kernels.programs.matmul")
    parser.add_argument("--balanced-matmul",
                        help="JSON recipe for both balanced grouped gate/up and down")
    parser.add_argument("--routing", help="Explicit JSON runtime.routing for every selected probe")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prior-operators", type=Path)
    parser.add_argument("--prior-comparison", type=Path)
    parser.add_argument("--probe-seconds", type=float, default=60)
    parser.add_argument("--total-seconds", type=float, default=300)
    parser.add_argument("--max-probes", type=int, default=24)
    parser.add_argument("--max-operators", type=int, default=50000)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.source_root:
            sys.path.insert(0, str(args.source_root.resolve()))
        print(json.dumps(run_probe(json.load(sys.stdin)), allow_nan=False))
        return 0
    if not args.output:
        parser.error("--output is required")
    if bool(args.prior_operators) != bool(args.prior_comparison):
        parser.error("--prior-operators and --prior-comparison must be supplied together")
    if min(args.probe_seconds, args.total_seconds, args.max_probes, args.max_operators,
           args.top, args.prefill_tokens) <= 0 or args.decode_context < 0:
        parser.error("budgets and prefill must be positive; context must be nonnegative")
    result = run_census(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
