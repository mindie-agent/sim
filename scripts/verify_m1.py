#!/usr/bin/env python3
"""M1 verification gates: hygiene, catalog conservation and the spine benchmark.

[Role]
  Execute the three reproducible M1 gates and write machine-readable JSON.  The
  catalog gate rebuilds constructor shapes from the declared reference manifest
  (meta-device parameters, not simulator helpers) and conserves logical
  parameters, KV payload bytes and KV allocated bytes per catalog model; the
  spine gate measures the full Kimi-K3.1 body step through the public efficiency
  entry point.

[Invariants]
  * No device context, weight load or network access.  Holdout directories are
    never read.
  * The spine benchmark composes and imports outside the timed interval, uses a
    fresh evaluation context per call and reports every raw sample.
  * A gate returns nonzero on failure; a missing model or skipped case is a
    failure, not a zero measurement.

[Timeouts & Locks]
  Pure process-local measurement with no locks.  Catalog files are streamed and
  bounded by the simulator's ordinary metadata limits.

[Side-Effect Policy]
  Reads the repository and writes only the requested ``--output`` JSON path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

SIM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SIM_ROOT / "state" / "m1"
CATALOG_MODELS = (
    "deepseek-v4-10t", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813",
    "deepseek-v4.1-flash", "deepseek-v2-lite", "deepseek-v3.2-prune-3l",
    "deepseek-v3", "deepseek-r1", "deepseek-v3.2", "glm-5.3",
    "glm-5.3-flash", "glm-5.1", "glm-5.2", "kimi-k3", "kimi-k3.1",
    "kimi-k3-dspark", "kimi-k3-dspark-block5", "kimi-k2", "kimi-k2-thinking",
    "kimi-k2.5", "qwen3-30b-a3b", "qwen3-32b", "qwen3-235b-a22b",
    "qwen3-next-80b-a3b", "qwen3.5-27b", "qwen3.5-35b-a3b",
    "qwen3.5-397b-a17b", "qwen3.6-27b", "qwen3.8-2.4t-a95b", "qwen3.8-27b",
    "minimax-m3", "minimax-m3.1-flash", "gemma-4-26b-a4b", "gemma-4-31b",
    "qwen3.6-35b-a3b", "qwen3.8-flash-next", "step-5-preview",
)
CAPTURE_TOKENS = 4096
BENCH_MODEL = "kimi-k3.1"
BENCH_HARDWARE = "Ascend950DT_9581"
BENCH_CONTEXT = 4096
PERCENTILE_P95 = 0.95
MIB = 1024 * 1024
STREAM_LIMIT = 50 * MIB
FORBIDDEN_MODULES = ("opflow", "op_formulas", "prediction.piecewise",
                     "prediction.subgraphs")
PRIVATE_PREFIX = "/Users" + "/maoxx241"
IMAGE_MARKERS = ("docker" + ".io/", "swr" + ".cn-", "@sha" + "256:")


def _output(path: str | None, payload: dict) -> None:
    target = Path(path) if path else DEFAULT_OUTPUT / f"{payload['gate']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _pass(gate: str, **metrics) -> dict:
    return {"gate": gate, "status": "pass", **metrics}


def _fail(gate: str, reason: str, **metrics) -> dict:
    return {"gate": gate, "status": "fail", "reason": reason, **metrics}


def _climb(start: Path, name: str) -> Path | None:
    for parent in (start, *start.parents):
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return None


def _python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        try:
            if path.stat().st_size > STREAM_LIMIT:
                continue
        except OSError:
            continue
        yield path


def hygiene(args) -> dict:
    sim = Path(args.sim or SIM_ROOT).resolve()
    profiler = Path(args.profiler).resolve() if args.profiler else _climb(
        sim.parent, "profiler")
    deleted = [
        sim / "mindie_sim" / "opflow.py",
        sim / "mindie_sim" / "op_formulas.py",
        sim / "mindie_sim" / "prediction" / "piecewise.py",
        sim / "mindie_sim" / "prediction" / "subgraphs.py",
        sim / "mindie_sim" / "opflow",
    ]
    roots = [sim, *( [profiler] if profiler else [] )]
    sidecar_imports: list[str] = []
    private_paths: list[str] = []
    image_literals: list[str] = []
    for root in roots:
        for path in _python_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            relative = str(path.relative_to(root))
            if relative.startswith(("tests/", "docs/", "data/")) or relative == "scripts/verify_m1.py":
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if not (stripped.startswith(("import ", "from "))):
                    continue
                for module in FORBIDDEN_MODULES:
                    if module in stripped:
                        sidecar_imports.append(f"{relative}:{stripped[:120]}")
            if PRIVATE_PREFIX in text:
                private_paths.append(relative)
            for marker in IMAGE_MARKERS:
                if marker in text:
                    image_literals.append(f"{relative}:{marker}")
    present = [str(path.relative_to(sim)) for path in deleted if path.exists()]
    imports_ok = True
    detail = ""
    try:
        import mindie_sim  # noqa: F401
        import mindie_sim.cli  # noqa: F401
        import mindie_sim.compose  # noqa: F401
        import mindie_sim.prediction.api  # noqa: F401
        import mindie_sim.prediction.efficiency  # noqa: F401
    except Exception as error:  # pragma: no cover - import failure detail
        imports_ok = False
        detail = f"{type(error).__name__}:{error}"
    metrics = {
        "deleted_paths_present": present,
        "sidecar_imports": sidecar_imports,
        "private_home_paths": private_paths,
        "image_literals": image_literals,
        "canonical_imports": imports_ok,
        "canonical_import_detail": detail,
    }
    if present or sidecar_imports or private_paths or image_literals or not imports_ok:
        return _fail("hygiene", "hygiene_violation", **metrics)
    return _pass("hygiene", **metrics)


DEFAULT_MANIFEST = SIM_ROOT / "tests" / "fixtures" / "catalog_reference_manifest.json"
REFERENCE_BLOCK = 128
REFERENCE_TOKENS = 4096
REFERENCE_PARALLEL_CASES = ((1, 1, 1), (8, 1, 1), (1, 2, 1), (1, 1, 2))
CACHE_ABS_TOLERANCE = 1e-6
REPORT_SHAPES = 5


def _oracle_modules():
    tests_dir = SIM_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from reference import cache_meta_oracle, catalog_meta_oracle
    return catalog_meta_oracle, cache_meta_oracle


def _body_entries(model) -> list[tuple]:
    from mindie_sim.catalog.weights import _body_terms
    return [(term.name, int(term.rows), int(term.cols), int(term.count),
             "source" if term.source_owned else "runtime")
            for term in _body_terms(model)]


def _cache_actuals(model, parallel, tokens: int) -> tuple[float, float, float]:
    from mindie_sim.catalog.cache import (
        layer_cache_decomposition, request_cache_breakdown,
    )
    payload = 0.0
    allocated = 0.0
    for plan in model.layer_plans:
        for padded in (False, True):
            parts = layer_cache_decomposition(
                model, plan, parallel, tokens, capacity_padded=padded)
            total = float(sum(float(entry["total"]) for entry in parts.values()))
            if padded:
                allocated += total
            else:
                payload += total
    breakdown = float(request_cache_breakdown(model, parallel, tokens)["total_bytes"])
    return payload, allocated, breakdown


def _reference_tokens(reference) -> list[int]:
    window = max([int(layer.window) for layer in reference.layers] + [REFERENCE_BLOCK])
    return sorted({0, 1, REFERENCE_BLOCK - 1, REFERENCE_BLOCK,
                   max(0, window - 1), window, REFERENCE_TOKENS})


def _cache_matrix(reference, cache, model) -> tuple[int, float, list[dict]]:
    from mindie_sim.schema.parallel import ParallelSpec
    tokens = _reference_tokens(reference)
    cases = 0
    largest = 0.0
    failures: list[dict] = []
    for tp, pp, cp in REFERENCE_PARALLEL_CASES:
        parallel = ParallelSpec(tp=tp, dp=1, pp=pp, cp=cp)
        selected = tokens if (tp, pp, cp) == (1, 1, 1) else (REFERENCE_TOKENS,)
        for seq in selected:
            payload, allocated, breakdown = _cache_actuals(model, parallel, seq)
            reference_payload = float(cache.cache_totals(
                reference.geometry, seq, padded=False, tp=tp, cp=cp))
            reference_allocated = float(cache.cache_totals(
                reference.geometry, seq, padded=True, tp=tp, cp=cp))
            deltas = (payload - reference_payload,
                      allocated - reference_allocated,
                      allocated - breakdown)
            cases += 1
            largest = max(largest, *(abs(value) for value in deltas))
            if any(abs(value) > CACHE_ABS_TOLERANCE for value in deltas):
                failures.append({
                    "tp": tp, "pp": pp, "cp": cp, "tokens": seq,
                    "payload_delta": deltas[0],
                    "allocated_delta": deltas[1],
                    "layer_sum_vs_breakdown_delta": deltas[2],
                })
    return cases, largest, failures


def _auxiliary_summary(model) -> dict:
    aux = getattr(model, "auxiliary", None)
    if aux is None:
        return {}
    for kind in ("mtp", "dspark"):
        spec = getattr(aux, kind, None)
        if spec is not None:
            return {"kind": str(spec.kind), "layers": int(spec.layers),
                    "standalone": bool(aux.standalone_draft)}
    return {}


def catalog_meta(args) -> dict:
    from mindie_sim.model_catalog import load_model
    from mindie_sim.schema.parallel import ParallelSpec
    catalog, cache = _oracle_modules()
    required = int(args.require_models or len(CATALOG_MODELS))
    if len(CATALOG_MODELS) != required:
        return _fail("catalog-meta",
                     f"catalog_model_set_size:{len(CATALOG_MODELS)}!={required}")
    manifest_path = Path(args.references or DEFAULT_MANIFEST).resolve()
    if not manifest_path.is_file():
        return _fail("catalog-meta", f"reference_manifest_missing:{manifest_path}")
    try:
        manifest = catalog.load_manifest(manifest_path)
        models = catalog.manifest_models(manifest)
    except (OSError, ValueError) as error:
        return _fail("catalog-meta", f"reference_manifest_invalid:{error}")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    missing = sorted(set(CATALOG_MODELS) - set(models))
    extra = sorted(set(models) - set(CATALOG_MODELS))
    if missing or extra or len(models) != required:
        return _fail(
            "catalog-meta", "reference_model_set_mismatch",
            required_models=required, manifest_models=len(models),
            missing_references=missing[:REPORT_SHAPES],
            extra_references=extra[:REPORT_SHAPES],
            references=str(manifest_path), references_sha256=manifest_digest)
    rows = []
    failures = []
    parameter_total = 0
    reference_parameter_total = 0
    payload_total = 0.0
    allocated_total = 0.0
    for key in CATALOG_MODELS:
        try:
            reference = catalog.build_reference(key, SIM_ROOT, models[key])
        except (OSError, ValueError, KeyError) as error:
            failures.append({"model": key, "reason": f"reference_invalid:{error}"})
            continue
        try:
            model = load_model(key)
        except Exception as error:  # pragma: no cover - model load detail
            failures.append({"model": key, "reason": f"load_failed:{error}"})
            continue
        actual_entries = _body_entries(model)
        comparison = catalog.compare_entries(reference, actual_entries)
        parallel = ParallelSpec(tp=1, dp=1, pp=1, cp=1)
        payload, allocated, breakdown = _cache_actuals(model, parallel, REFERENCE_TOKENS)
        reference_payload = float(cache.cache_totals(
            reference.geometry, REFERENCE_TOKENS, padded=False))
        reference_allocated = float(cache.cache_totals(
            reference.geometry, REFERENCE_TOKENS, padded=True))
        cases, largest, cache_failures = _cache_matrix(reference, cache, model)
        payload_delta = payload - reference_payload
        allocated_delta = allocated - reference_allocated
        auxiliary = _auxiliary_summary(model)
        declared_auxiliary = models[key]["auxiliary_modes"]
        auxiliary_ok = all(
            declared_auxiliary.get(name) == auxiliary.get(name)
            for name in ("kind", "layers", "standalone")) if auxiliary else not declared_auxiliary
        scope_ok = (reference.scope == models[key]["parameter_scope"]
                    and model.accounting_scope == reference.scope)
        schedule_ok = (
            [layer.window for layer in reference.layers]
            == [plan.window for plan in model.layer_plans]
            and [layer.compress_ratio for layer in reference.layers]
            == [plan.compress_ratio for plan in model.layer_plans]
            and [layer.owns_index_k for layer in reference.layers]
            == [plan.owns_index_k for plan in model.layer_plans])
        row = {
            "model": key,
            "config": reference.config_path,
            "builder": reference.builder,
            "parameter_scope": reference.scope,
            "parameters": comparison["actual_parameters"],
            "parameter_reference": comparison["reference_parameters"],
            "parameter_delta": comparison["parameter_delta"],
            "runtime_parameters": comparison["actual_runtime_parameters"],
            "runtime_reference": comparison["reference_runtime_parameters"],
            "runtime_delta": comparison["runtime_delta"],
            "shapes_compared": len(actual_entries),
            "missing_in_actual": len(comparison["missing_in_actual"]),
            "missing_in_reference": len(comparison["missing_in_reference"]),
            "shape_mismatches": len(comparison["shape_mismatches"]),
            "kv_payload_bytes": payload,
            "kv_payload_reference": reference_payload,
            "kv_payload_delta": payload_delta,
            "kv_allocated_bytes": allocated,
            "kv_allocated_reference": reference_allocated,
            "kv_allocated_delta": allocated_delta,
            "layer_sum_vs_breakdown_delta": allocated - breakdown,
            "cache_cases": cases,
            "cache_max_abs_delta": largest,
            "auxiliary_modes": auxiliary,
            "auxiliary_declared": declared_auxiliary,
            "auxiliary_ok": auxiliary_ok,
            "scope_ok": scope_ok,
            "schedule_ok": schedule_ok,
            "complete": bool(comparison["complete"] and not cache_failures
                             and auxiliary_ok and scope_ok and schedule_ok),
        }
        rows.append(row)
        parameter_total += comparison["actual_parameters"]
        reference_parameter_total += comparison["reference_parameters"]
        payload_total += payload
        allocated_total += allocated
        if not row["complete"] or any(abs(value) > 0 for value in (
                comparison["parameter_delta"], comparison["runtime_delta"],
                payload_delta, allocated_delta)):
            failures.append({
                "model": key,
                **{name: row[name] for name in (
                    "parameter_delta", "runtime_delta", "kv_payload_delta",
                    "kv_allocated_delta", "missing_in_actual",
                    "missing_in_reference", "shape_mismatches", "cache_cases",
                    "cache_max_abs_delta", "auxiliary_ok", "scope_ok",
                    "schedule_ok")},
                "shape_detail": comparison["shape_mismatches"][:REPORT_SHAPES],
                "missing_actual_detail": comparison["missing_in_actual"][:REPORT_SHAPES],
                "missing_reference_detail": comparison["missing_in_reference"][:REPORT_SHAPES],
                "cache_detail": cache_failures[:REPORT_SHAPES],
            })
    metrics = {
        "required_models": required,
        "manifest_models": len(models),
        "completed_models": len(rows),
        "failed_models": len(failures),
        "references": str(manifest_path),
        "references_sha256": manifest_digest,
        "parameter_total": parameter_total,
        "parameter_reference_total": reference_parameter_total,
        "parameter_delta_total": parameter_total - reference_parameter_total,
        "kv_payload_total": payload_total,
        "kv_allocated_total": allocated_total,
        "parallel_cases": [list(case) for case in REFERENCE_PARALLEL_CASES],
        "meta_tensor_backend": ("torch" if catalog._torch_available()
                                else "arithmetic"),
        "failures": failures[:10],
        "models": rows,
    }
    if failures or len(rows) != required:
        return _fail("catalog-meta", "catalog_conservation_mismatch", **metrics)
    return _pass("catalog-meta", **metrics)


def _affinity() -> str | None:
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None
    if not allowed:
        return None
    try:
        os.sched_setaffinity(0, {allowed[0]})
    except OSError:
        return None
    return str(allowed[0])


def _rss_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / MIB


def _count_constructions() -> dict:
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.evaluation import ExecutionBindings
    counts = {"cost_models": 0, "bindings": 0}
    original_cost = CostModel.__init__
    original_bindings = ExecutionBindings.create.__func__

    def counted_cost(self, *args, **kwargs):
        counts["cost_models"] += 1
        return original_cost(self, *args, **kwargs)

    def counted_bindings(cls, *args, **kwargs):
        counts["bindings"] += 1
        return original_bindings(cls, *args, **kwargs)

    CostModel.__init__ = counted_cost
    ExecutionBindings.create = classmethod(counted_bindings)
    return counts


def spine_benchmark(args) -> dict:
    cwd = os.getcwd()
    os.chdir(SIM_ROOT)
    if str(SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(SIM_ROOT))
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.efficiency import analyze_network_efficiency

    cpu = _affinity()
    cold_start = time.perf_counter()
    case = compose_case(
        BENCH_MODEL, BENCH_HARDWARE, tp=8, ep=8, dp=1, pp=1,
        num_devices=8, ranks_per_node=8,
        prompt_tokens=CAPTURE_TOKENS, output_tokens=2, concurrency=1,
    )
    cold_ms = (time.perf_counter() - cold_start) * 1000.0

    counters = _count_constructions()
    samples: list[float] = []
    artifacts: list[dict] = []
    try:
        for index in range(int(args.warmups) + int(args.repetitions)):
            before = dict(counters)
            start = time.perf_counter()
            report = analyze_network_efficiency(
                case, phase="decode", tokens=1, context=BENCH_CONTEXT,
                sequences=1, scope="body")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            gc.collect()
            delta = {key: counters[key] - before.get(key, 0) for key in counters}
            if index >= int(args.warmups):
                samples.append(elapsed_ms)
                artifacts.append({
                    "graph_duration_ms": float(report["graph_duration_ms"]),
                    "operators": len(report["operators"]),
                    "cost_models": delta["cost_models"],
                    "bindings": delta["bindings"],
                })
            if any(delta[key] != 1 for key in counters):
                return _fail(
                    "spine-benchmark", "construction_count_invariant",
                    cost_models=counters["cost_models"],
                    bindings=counters["bindings"], sample=index)
    finally:
        os.chdir(cwd)
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(PERCENTILE_P95 * len(ordered)) - 1))
    metrics = {
        "model": BENCH_MODEL, "hardware": BENCH_HARDWARE,
        "warmups": int(args.warmups), "repetitions": int(args.repetitions),
        "max_allowed_ms": float(args.max_ms),
        "cold_ms": cold_ms,
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[index],
        "max_ms": max(samples),
        "min_ms": min(samples),
        "rss_mib": _rss_mib(),
        "cpu": cpu,
        "cost_models_per_call": [item["cost_models"] for item in artifacts],
        "bindings_per_call": [item["bindings"] for item in artifacts],
        "samples_ms": samples,
        "graph_duration_ms": [item["graph_duration_ms"] for item in artifacts],
        "operators": [item["operators"] for item in artifacts],
    }
    if not samples:
        return _fail("spine-benchmark", "no_samples", **metrics)
    if max(samples) >= float(args.max_ms):
        return _fail("spine-benchmark", "measured_call_exceeds_limit", **metrics)
    return _pass("spine-benchmark", **metrics)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    hygiene_parser = sub.add_parser("hygiene")
    hygiene_parser.add_argument("--sim", default=str(SIM_ROOT))
    hygiene_parser.add_argument("--profiler", default=None)
    hygiene_parser.add_argument("--output", default=None)

    catalog_parser = sub.add_parser("catalog-meta")
    catalog_parser.add_argument("--references", default=str(DEFAULT_MANIFEST))
    catalog_parser.add_argument("--require-models", type=int, default=37)
    catalog_parser.add_argument("--output", default=None)

    spine_parser = sub.add_parser("spine-benchmark")
    spine_parser.add_argument("--warmups", type=int, default=5)
    spine_parser.add_argument("--repetitions", type=int, default=100)
    spine_parser.add_argument("--max-ms", type=float, default=50.0)
    spine_parser.add_argument("--output", default=None)

    args = parser.parse_args(argv)
    handlers = {"hygiene": hygiene, "catalog-meta": catalog_meta,
                "spine-benchmark": spine_benchmark}
    payload = handlers[args.command](args)
    _output(args.output, payload)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
