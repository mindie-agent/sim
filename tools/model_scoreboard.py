"""Build a model scoreboard from independent observed and predicted records.

[Role] Load bounded manifests and execute explicitly requested model probes.
[Invariants] Predictions do not read measurements. Missing predictions remain
  visible; a timeout never becomes zero cost. All graph execution uses the
  existing public convergence worker and the common simulation engine.
[Timeouts & Locks] Sequential subprocesses with explicit per-probe timeout.
[Side-Effect Policy] Reads manifests/cases; writes only requested report files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from mindie_sim.observations.scoreboard import (
    model_scoreboard, prediction_from_probe, render_markdown,
)
from mindie_sim.observations.scoreboard_accuracy import (
    annotate_kimi_report, render_kimi_markdown,
)
from mindie_sim.observations.cohort_scoreboard import (
    render_cohort_campaigns, summarize_cohorts,
)


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(f"scoreboard_metadata_exceeds_50_mib:{path}")
    return json.loads(path.read_text())


def source(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def run_prediction(spec, *, timeout):
    """The child receives only case/geometry, with no observation fields."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sim-scoreboard-") as directory:
        packet, output = Path(directory) / "probe.json", Path(directory) / "result.json"
        packet.write_text(json.dumps(spec))
        try:
            child = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(packet),
                 "-o", str(output)], timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": f"prediction_timeout:{timeout}s",
                    "wall_seconds": time.monotonic() - started}
        if child.returncode:
            return {"status": "failed", "error": child.stderr[-4000:],
                    "returncode": child.returncode, "wall_seconds": time.monotonic() - started}
        return read_json(output)


def execution_identity(spec, report):
    """Bind comparison facts to the case/geometry actually given to run_probe."""
    from mindie_sim.prediction.structure import parse_case
    import model_convergence
    from model_convergence import fingerprint

    case = parse_case(spec["case"])
    case_key = fingerprint(case.to_dict())
    if case_key != report["effective_case_key"]:
        raise ValueError("scoreboard_worker_case_differs_from_executed_case")
    geometry = {"context": 0, "sequences": 1, "scope": "model", **spec["geometry"]}
    scheduled = spec.get("driver") in {"scheduled_iteration", "scheduled_prefill"}
    prefill = spec.get("driver") == "scheduled_prefill"
    draft_structure = None
    if scheduled:
        from mindie_sim.catalog.draft import optional_draft_spec, draft_quant
        draft = optional_draft_spec(case.model, dict(case.runtime.metadata).get("draft_mode", "mtp"))
        draft_structure = {
            "kind": draft.kind, "layers": draft.layers,
            "weight_format": draft_quant(case.model, draft).weight_format,
            "noncausal_query_block": draft.noncausal_query_block,
            "markov_rank": draft.markov_rank,
        }
    quant = case.model.quant
    return {
        "effective_case_key": case_key, "input_case_sha256": fingerprint(spec["case"]),
        "geometry": geometry,
        "conditions": {
            "model": case.model.name,
            "hardware": {
                "family": case.machine.arch_family,
                "memory_bytes_per_device": sorted({die.memory_bytes for die in case.machine.dies}),
                "devices": len(case.machine.dies),
                "dies_per_package": case.machine.dies_per_package,
            },
            "parallel": case.runtime.parallel.to_dict(),
            **{key: geometry[key] for key in ("phase", "tokens", "context", "sequences")},
            **({"tokens": geometry["tokens"] + spec["proposal_tokens"]}
               if scheduled and not prefill else {}),
            "precision": {
                **{key: getattr(quant, key) for key in (
                    "weight_format", "activation_format", "expert_weight_format", "kv_dtype")},
                "model_activation": case.model.activation_dtype,
            },
            # target_graphs creates RequestViews without draft preparation or
            # proposals and excludes sampling/token_commit from model probes.
            "branch": ("target_plus_mtp_prefill" if prefill else
                       "full_speculative_step" if scheduled else "target_only"),
        },
        "frontend": "Frontend(IterationPlan)" if scheduled else "target_graphs",
        "hardware_profile": case.machine.sku or case.machine.hardware_key or case.machine.name,
        "proposal_tokens": spec.get("proposal_tokens", 0),
        "draft_preparation": prefill,
        "draft_structure": draft_structure,
        "excluded_from_graph": [] if scheduled else ["speculative_draft", "sampling", "token_commit"],
        "worker_sources": [source(Path(__file__)), source(Path(model_convergence.__file__))],
    }


def scheduled_probe(spec):
    """Execute a complete fixed iteration without request-admission inference."""
    from mindie_sim import operator_costs
    from mindie_sim.prediction.attention import AttentionPolicy
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.dispatch import cohorts
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.workload import IterationPlan, RequestView
    from model_convergence import _runtime_identity, fingerprint, summarize_costs, worker_usage

    started = time.monotonic()
    case, geometry = parse_case(spec["case"]), spec["geometry"]
    prefill = spec.get("driver") == "scheduled_prefill"
    if prefill:
        if (geometry["phase"] != "prefill" or geometry.get("scope", "model") != "model"
                or spec.get("proposal_tokens", 0) != 0
                or dict(case.runtime.metadata).get("draft_mode") != "mtp"):
            raise ValueError("scheduled_prefill_requires_mtp_preparation_without_decode_proposals")
        from mindie_sim.catalog.draft import optional_draft_spec
        draft = optional_draft_spec(case.model, "mtp")
        if spec.get("draft_layers") is not None and draft.layers != spec["draft_layers"]:
            raise ValueError("scheduled_prefill_declared_draft_layer_count_mismatch")
    elif (geometry["phase"] != "decode" or geometry.get("scope", "model") != "model"
          or geometry["tokens"] != 1):
        raise ValueError("scheduled_iteration_requires_single_token_decode_model_scope")
    participants = cohorts(case.runtime.parallel)[0]
    views = {rank: tuple(RequestView(
        f"r{rank}:{index}", geometry["tokens"], geometry["context"], True, geometry["phase"],
        draft_proposal_tokens=spec.get("proposal_tokens", 0), draft_prepare=prefill,
    ) for index in range(geometry.get("sequences", 1))) for rank in participants}
    global_views = tuple(view for group in views.values() for view in group)
    global_tokens = sum(view.q_len + view.draft_proposal_tokens for view in global_views)
    frontend = Frontend(attention_policy=AttentionPolicy(**case.runtime.attention_policy))
    works = {rank: frontend(
        IterationPlan(group), case.model, case.runtime, case.machine,
        global_tokens=global_tokens if len(participants) > 1 else None,
        global_views=global_views if len(participants) > 1 else None,
    ) for rank, group in views.items()}
    recipes = RecipeBook.from_mapping(case.runtime.kernels)
    if sum(len(recipes.expand(work.ops)) for work in works.values()) > spec["max_operators"]:
        raise ValueError("scheduled_iteration_operator_budget_exceeded")
    execution = Execution(case, CostModel(case), recipes=recipes, include_submission=False)
    if len(works) > 1:
        steps = execution.add_group(works, 0)
    else:
        rank, work = next(iter(works.items()))
        steps = {rank: execution.add(work, rank, 0)}
    execution.engine.advance(until=tuple(step.completion for step in steps.values()), first=False)
    duration = max(step.completion.time for step in steps.values())
    isolated = operator_costs(case, {rank: work.ops for rank, work in works.items()},
                              recipes=recipes)
    summary = execution.engine.summary()
    return {
        "status": "modeled", "runtime": _runtime_identity(),
        "effective_case_key": fingerprint(case.to_dict()),
        "parallel": case.runtime.parallel.to_dict(), "replicas": list(works),
        "deployment_scope": "first_independent_dp_cohort",
        "costs": summarize_costs(isolated["operators"], isolated["operator_sum_ms"], top=spec["top"]),
        "graph": {"duration_ms": duration, "boundary": "device_iteration",
                  **{key: summary[key] for key in (
                      "live_tasks", "live_programs", "completed_tasks", "peak_live_tasks")},
                  "remaining_allocations": sum(execution.engine.allocated.values())},
        "iteration": {
            "plans": {rank: work.plan.to_dict() for rank, work in works.items()},
            "replay_scopes": {rank: work.replay_scope for rank, work in works.items()},
            "families": sorted({op.family for work in works.values() for op in work.ops}),
            "phase_occurrences": {
                phase: sum(op.phase == phase for work in works.values() for op in work.ops)
                for phase in sorted({op.phase for work in works.values() for op in work.ops})},
            "submission_included": False,
        },
        "resources": worker_usage(),
        "wall_seconds": time.monotonic() - started,
    }


def load_prediction(prediction, entry, manifest, args, started):
    """One retained or freshly executed prediction, with the same case binding."""
    if args.run_predictions and prediction.get("case"):
        case_path = (manifest.parent / prediction["case"]).resolve()
        spec = {
            "case": read_json(case_path), "geometry": prediction["geometry"],
            "level": "graph", "max_operators": prediction.get("max_operators", 50000),
            "top": prediction.get("top", 12),
            **{key: prediction[key] for key in ("driver", "proposal_tokens", "draft_layers")
               if key in prediction},
        }
        remaining = args.total_timeout - (time.monotonic() - started)
        probe = (run_prediction(spec, timeout=min(args.timeout, remaining)) if remaining > 0 else
                 {"status": "failed", "reason": "scoreboard_total_prediction_budget_exhausted"})
        entry.setdefault("sources", []).append(source(case_path))
    elif prediction.get("report"):
        path = (manifest.parent / prediction["report"]).resolve()
        probe = read_json(path)
        entry.setdefault("sources", []).append(source(path))
    else:
        probe = {"status": "unavailable", "reason": "prediction_not_run"}
    adapted = prediction_from_probe(
        probe, conditions=prediction["conditions"],
        comparison_scope=prediction["comparison_scope"])
    if prediction.get("case") and probe.get("status") == "modeled":
        from model_convergence import fingerprint
        path = (manifest.parent / prediction["case"]).resolve()
        expected_key = fingerprint(read_json(path))
        if probe.get("execution_identity", {}).get("input_case_sha256") != expected_key:
            adapted["identity_issues"].append("prediction_input_case_mismatch")
    return adapted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--run-predictions", action="store_true")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--total-timeout", type=float, default=150)
    parser.add_argument("--worker")
    parser.add_argument("--kimi-family-report",
                        help="Annotate a retained Kimi descriptor report without predicting again")
    args = parser.parse_args(argv)
    if args.worker:
        from model_convergence import _runtime_identity, run_probe
        spec = read_json(args.worker)
        before = _runtime_identity()
        report = (scheduled_probe(spec)
                  if spec.get("driver") in {"scheduled_iteration", "scheduled_prefill"}
                  else run_probe(spec))
        report["execution_identity"] = execution_identity(spec, report)
        report["runtime_integrity"] = {
            "status": "unchanged" if before == report["runtime"] else "changed",
            "before": before, "after": report["runtime"],
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        return 0
    if args.kimi_family_report:
        if args.manifest or args.run_predictions:
            parser.error("--kimi-family-report is a report-only input, separate from a manifest")
        report = annotate_kimi_report(read_json(args.kimi_family_report))
        report["accuracy_refresh"] = {
            "input": source(args.kimi_family_report), "producer": source(__file__),
            "execution": "retained_predictions_not_reexecuted",
        }
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.markdown:
            Path(args.markdown).write_text(render_kimi_markdown(report))
        print(json.dumps({"schema": report["schema"], "output": args.output}))
        return 0
    if not args.manifest:
        parser.error("manifest is required")
    manifest = Path(args.manifest).resolve()
    packet = read_json(manifest)
    entries = deepcopy(packet["entries"])
    started = time.monotonic()
    for entry in entries:
        prediction = entry.pop("prediction_input", None)
        if prediction is None:
            continue
        if prediction.get("scenarios"):
            shared = {key: value for key, value in prediction.items() if key != "scenarios"}
            entry["predicted"] = {"scenarios": [
                load_prediction({**shared, **scenario}, entry, manifest, args, started)
                for scenario in prediction["scenarios"]]}
            continue
        entry["predicted"] = load_prediction(prediction, entry, manifest, args, started)
    report = model_scoreboard(entries, top=packet.get("top", 5))
    report["manifest"] = source(manifest)
    report["source_context"] = packet.get("source_context", {})
    report["cohort_campaigns"] = [
        summarize_cohorts(
            read_json(manifest.parent / item["report"]),
            label=item["label"], provenance=source(manifest.parent / item["report"]))
        for item in packet.get("cohort_inputs", ())
    ]
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        markdown = render_markdown(report)
        notes = report["source_context"].get("report_notes", [])
        if notes:
            title, body = markdown.split("\n", 1)
            markdown = title + "\n\n" + "\n\n".join(notes) + "\n" + body
        Path(args.markdown).write_text(
            markdown + render_cohort_campaigns(report["cohort_campaigns"]))
    print(json.dumps({"coverage": report["coverage"], "output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
