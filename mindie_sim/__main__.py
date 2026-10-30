"""Command line interface for the neutral modeling system.

[Role] Run prediction, inspect structure, calibrate primitives and validate.
[Invariants] Measurements enter simulation only through explicit calibration.
[Timeouts & Locks] Bounded metadata reads; streaming observation commands.
[Side-Effect Policy] Writes only explicitly requested output/index/store paths.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from . import predict
from .prediction.structure import parse_case
from .prediction.primitives import PrimitiveKey, PrimitiveStore


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024**2:
        raise ValueError("metadata_input_exceeds_50_MiB:use_streaming_observation_import")
    with path.open() as stream:
        return json.load(stream)


def emit(value, output):
    if output:
        with Path(output).open("w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def parser():
    root = argparse.ArgumentParser(prog="mindie-sim")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("predict", "inspect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("case")
        cmd.add_argument("--output", "-o")
        if name == "predict":
            cmd.add_argument("--primitives")
            cmd.add_argument("--trace-limit", type=int, default=2)
            cmd.add_argument("--event-limit", type=int, default=256)
            cmd.add_argument("--scope", choices=("serving", "body"), default="serving")
            cmd.add_argument("--phase", choices=("decode", "prefill"))
            cmd.add_argument("--tokens", type=int, help="target verification rows per sequence")
            cmd.add_argument("--context", type=int)
            cmd.add_argument("--sequences", type=int, help="sequences per participating DP replica")
    operators = sub.add_parser("operators", help="isolated target-body device program costs")
    operators.add_argument("case")
    operators.add_argument("--phase", choices=("prefill", "decode"), required=True)
    operators.add_argument("--tokens", type=int, required=True)
    operators.add_argument("--sequences", type=int, default=1)
    operators.add_argument("--scope", choices=("body", "model"), default="body")
    operators.add_argument("--context", type=int, default=0)
    operators.add_argument("--media-tokens-per-sequence", type=int)
    operators.add_argument("--draft-prepare", action="store_true",
                           help="include declared draft cache preparation after model prefill")
    operators.add_argument("--no-body-defaults", dest="apply_body_defaults", action="store_false")
    operators.add_argument("--output", "-o")
    graph = sub.add_parser("graph", help="compose a pure target graph on shared device resources")
    graph.add_argument("case")
    graph.add_argument("--phase", choices=("prefill", "decode"), required=True)
    graph.add_argument("--tokens", type=int, required=True, help="tokens per sequence")
    graph.add_argument("--sequences", type=int, default=1, help="sequences per participating DP replica")
    graph.add_argument("--scope", choices=("body", "model"), default="model")
    graph.add_argument("--context", type=int, default=0)
    graph.add_argument("--media-tokens-per-sequence", type=int)
    graph.add_argument("--draft-prepare", action="store_true",
                       help="include declared draft cache preparation after model prefill")
    graph.add_argument("--input-ready-ms", type=float, default=0)
    graph.add_argument("--include-submission", action="store_true")
    graph.add_argument("--replay-scope", choices=("none", "full", "piecewise"),
                       help="override the target frontend's selected capture scope")
    graph.add_argument("--skip-isolated", action="store_true")
    graph.add_argument("--primitives")
    graph.add_argument("--event-limit", type=int, default=0)
    graph.add_argument("--no-body-defaults", dest="apply_body_defaults", action="store_false")
    graph.add_argument("--output", "-o")
    catalog = sub.add_parser("catalog")
    catalog.add_argument("name", nargs="?")
    catalog.add_argument("--output", "-o")
    observations = sub.add_parser("observations")
    observations.add_argument("index")
    observations.add_argument("--scan", action="append", default=[])
    observations.add_argument("--attach-profiler")
    observations.add_argument("--find")
    observations.add_argument("--output", "-o")
    rows = sub.add_parser("profiler-rows")
    rows.add_argument("source")
    rows.add_argument("--after")
    rows.add_argument("--limit", type=int, default=50)
    rows.add_argument("--output", "-o")
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("store")
    calibrate.add_argument("samples", help="JSONL: key, duration_ms, run_id, repeat_index, source, split")
    calibrate.add_argument("--output", "-o")
    bind = sub.add_parser("bind-profiler")
    bind.add_argument("catalogue")
    bind.add_argument("store")
    bind.add_argument("bindings", help="JSONL with variant, key, run_id, split")
    bind.add_argument("--output", "-o")
    validate = sub.add_parser("validate")
    validate.add_argument("prediction")
    validate.add_argument("observation")
    validate.add_argument("--boundary", required=True)
    validate.add_argument("--output", "-o")
    rates = sub.add_parser("fit-rate")
    rates.add_argument("observations")
    rates.add_argument("--output", "-o")
    service = sub.add_parser("fit-service", help="fit one primitive work-response curve")
    service.add_argument("observations")
    service.add_argument("--output", "-o")
    study = sub.add_parser("study", help="compare explicit candidates across operator or graph probes")
    study.add_argument("manifest")
    study.add_argument("--primitives")
    study.add_argument("--output", "-o")
    accuracy = sub.add_parser("accuracy", help="independent matched-boundary accuracy and trend report")
    accuracy.add_argument("packet")
    accuracy.add_argument("--output", "-o")
    call = sub.add_parser("call", help="compose a complete declared callback and repetition protocol")
    call.add_argument("case")
    call.add_argument("declaration", help="builder, arguments and optional protocol mapping")
    call.add_argument("--include-submission", action="store_true")
    call.add_argument("--output", "-o")
    alignment = sub.add_parser("compare-calls", help="conditional historical event/call boundary comparison")
    alignment.add_argument("case")
    alignment.add_argument("protocol")
    alignment.add_argument("--root", default=".", help="base for protocol source paths")
    alignment.add_argument("--include-submission", action="store_true",
                           help="also evaluate a separately declared host submission scenario")
    alignment.add_argument("--submission", help="JSON host submission declaration, never a timing fit")
    alignment.add_argument("--max-predictions", type=int, default=512)
    alignment.add_argument("--max-conditions", type=int, default=4096)
    alignment.add_argument("--output", "-o")
    return root


def run(args):
    if args.command == "call":
        from .prediction.call import call_costs
        from .prediction.call_builders import build_call
        declaration = read_json(args.declaration)
        call = build_call(declaration)
        return {
            **call_costs(read_json(args.case), call.ops, recipes=call.recipes,
                         include_submission=args.include_submission,
                         **declaration.get("protocol", {})),
            "call": call.ledger, "limitations": list(call.limitations),
        }
    if args.command == "compare-calls":
        from .observations.call_alignment import compare_calls
        return compare_calls(
            read_json(args.case), read_json(args.protocol), root=args.root,
            include_submission=args.include_submission,
            submission=read_json(args.submission) if args.submission else None,
            max_predictions=args.max_predictions, max_conditions=args.max_conditions,
        )
    if args.command == "study":
        from .study import study
        packet = read_json(args.manifest)
        root = Path(args.manifest).resolve().parent
        candidates = [
            {**row, "case": read_json(root / row["case"])}
            if isinstance(row.get("case"), str) else row
            for row in packet["candidates"]]
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            return study(candidates, packet["probes"], primitives=store, **packet.get("options", {}))
    if args.command == "accuracy":
        from .observations.accuracy import accuracy_report
        return accuracy_report(read_json(args.packet))
    if args.command == "graph":
        from .prediction.graph_analysis import graph_costs
        from .prediction.operator_analysis import target_graphs
        case = parse_case(read_json(args.case))
        graphs = target_graphs(case, phase=args.phase, tokens=args.tokens, context=args.context,
                               sequences=args.sequences, scope=args.scope,
                               media_tokens_per_sequence=args.media_tokens_per_sequence,
                               draft_prepare=args.draft_prepare,
                               apply_body_defaults=args.apply_body_defaults)
        inputs = {"analysis_input": args.input_ready_ms} if args.scope == "body" else None
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            return {
                **graph_costs(case, graphs, primitives=store, external_inputs=inputs,
                              input_ready_ms=args.input_ready_ms,
                              include_submission=args.include_submission,
                              replay_scope=args.replay_scope, isolated=not args.skip_isolated,
                              trace_limit=args.event_limit),
                "graph_scope": args.scope, "tokens_per_sequence": args.tokens,
                "sequences_per_replica": args.sequences,
                "media_tokens_per_sequence": args.media_tokens_per_sequence,
                "draft_prepare": args.draft_prepare,
                "deployment_scope": "first_independent_dp_cohort",
            }
    if args.command == "operators":
        from .prediction.operator_analysis import operator_costs, target_graphs
        case = parse_case(read_json(args.case))
        ops = target_graphs(case, phase=args.phase, tokens=args.tokens, context=args.context,
                            sequences=args.sequences, scope=args.scope,
                            media_tokens_per_sequence=args.media_tokens_per_sequence,
                            draft_prepare=args.draft_prepare,
                               apply_body_defaults=args.apply_body_defaults)
        return {**operator_costs(case, ops), "graph_scope": args.scope,
                "tokens_per_sequence": args.tokens, "sequences": args.sequences,
                "sequences_per_replica": args.sequences,
                "draft_prepare": args.draft_prepare,
                "deployment_scope": "first_independent_dp_cohort"}
    if args.command == "predict":
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            return predict(read_json(args.case), primitives=store,
                           trace_limit=args.trace_limit, event_limit=args.event_limit,
                           scope=args.scope, phase=args.phase, tokens=args.tokens,
                           context=args.context, sequences=args.sequences)
    if args.command == "inspect":
        case = parse_case(read_json(args.case))
        from .prediction.capacity import Memory
        from .prediction.scheduler import Scheduler
        from .prediction.frontend import Frontend
        from .prediction.attention import AttentionPolicy
        memory = Memory(case)
        scheduler = Scheduler(case, list(case.workload.requests), memory)
        plan = scheduler.schedule(min(r.arrival_ms for r in case.workload.requests))
        work = Frontend(attention_policy=AttentionPolicy(**case.runtime.attention_policy))(
            plan, case.model, scheduler.next_runtime(plan), case.machine) if plan else None
        return {"case": case.to_dict(), "memory": memory.to_dict(),
                "operators": [op.to_dict() for op in work.ops] if work else []}
    if args.command == "catalog":
        from .model_catalog import load_model, list_models
        return load_model(args.name).to_dict() if args.name else list_models()
    if args.command == "observations":
        from .observations.store import ObservationIndex
        with ObservationIndex(args.index) as index:
            for directory in args.scan:
                index.scan(directory)
            if args.attach_profiler:
                index.attach_profiler_catalogue(args.attach_profiler)
            return index.find_sources(args.find) if args.find else index.inventory()
    if args.command == "profiler-rows":
        from .observations.store import ProfilerCatalogue, profiler_rows
        if Path(args.source).suffix == ".csv":
            from itertools import islice
            return list(islice(profiler_rows(args.source), min(1000, args.limit)))
        with ProfilerCatalogue(args.source) as catalog:
            return list(catalog.variants(limit=args.limit, after=args.after))
    if args.command == "calibrate":
        import ijson
        with PrimitiveStore(args.store, writable=True) as store, Path(args.samples).open("rb") as stream:
            for sample in ijson.items(stream, "", multiple_values=True, use_float=True):
                key = PrimitiveKey.from_mapping(sample.pop("key"))
                store.record(key, **sample)
            return store.inventory()
    if args.command == "validate":
        from .observations.validation import service_observation, compare
        return compare(read_json(args.prediction), service_observation(args.observation), boundary=args.boundary)
    if args.command == "bind-profiler":
        import ijson
        from .observations.calibration import import_profiler
        with PrimitiveStore(args.store, writable=True) as store, Path(args.bindings).open("rb") as stream:
            return import_profiler(args.catalogue, store,
                                   ijson.items(stream, "", multiple_values=True, use_float=True))
    if args.command == "fit-rate":
        from .observations.validation import fit_resource_rate
        return fit_resource_rate(read_json(args.observations)["samples"])
    if args.command == "fit-service":
        from .observations.service_fitting import fit_service_curve
        return fit_service_curve(read_json(args.observations))
    raise ValueError("unknown_command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
        emit(result, args.output)
        if args.command in {"study", "compare-calls"} and result["status"] != "completed":
            return 2
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
