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


MAX_CLI_JSON_BYTES = 50 * 1024**2


SWEEP_DEFAULT_MAX_SEQUENCES = 256


class CliUsageError(RuntimeError):
    """Local usage rejection; never a profiler runtime dependency."""


def load_cli_json(path):
    """Bounded CLI JSON ingress: size gate, UTF-8 and finite constants."""
    path = Path(path)
    if path.stat().st_size > MAX_CLI_JSON_BYTES:
        raise CliUsageError("JSON input exceeds 50 MiB budget")
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError("invalid UTF-8 JSON input") from exc

    def bad_const(token):
        raise CliUsageError(f"non-finite JSON constant: {token}")

    try:
        return json.loads(raw, parse_constant=bad_const)
    except (ValueError, RecursionError) as exc:
        raise CliUsageError(f"invalid JSON input: {exc}") from exc


def read_json(path, *, bounded=False):
    path = Path(path)
    if path.stat().st_size > MAX_CLI_JSON_BYTES:
        raise ValueError("metadata_input_exceeds_50_MiB:use_streaming_observation_import")
    if bounded:
        return load_cli_json(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


#: Default latency budget of a multi-SKU comparison (ms).
DEFAULT_COMPARE_SLA_MS = 50.0


def emit(value, output):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    data = text.encode("utf-8")
    if output:
        Path(output).write_bytes(data)
    else:
        try:
            print(text, end="")
        except UnicodeEncodeError:
            print(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False))


def _add_supply_arguments(command):
    command.add_argument(
        "--supply", help="profiler C7 report or mindie-profiler-measured-supply/1 JSON")
    command.add_argument("--supply-recipe", help="selected operator recipe to bind")
    command.add_argument("--supply-kernel", help="selected native kernel name to bind")
    command.add_argument("--supply-hardware", help="explicit hardware key for the binding")
    command.add_argument("--supply-owners", type=int,
                         help="selected implementation active-owner default")
    command.add_argument("--supply-legacy-diagnostics", action="store_true",
                         help="retain legacy service_regimes alongside C7 resource_services")


def _case_with_supply(args):
    raw = read_json(args.case)
    if getattr(args, "supply", None):
        from .observations.calibration import apply_profiler_supply
        raw = apply_profiler_supply(
            raw, args.supply, recipe=getattr(args, "supply_recipe", None),
            kernel=getattr(args, "supply_kernel", None),
            hardware=getattr(args, "supply_hardware", None),
            owners=getattr(args, "supply_owners", None),
            legacy_diagnostics=bool(getattr(args, "supply_legacy_diagnostics", False)))
    return raw


def _export_structure(args, case, *, phase, tokens, context, sequences):
    """Write the same geometry's structure export and describe the artifact.

    The structure record is the one exported by ``mindie_sim.structure``; the
    graph command only selects the geometry so both entries agree on the
    resolved operator stack.
    """
    from .structure import export_structure
    report = export_structure(
        case, phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope=args.scope, apply_body_defaults=args.apply_body_defaults)
    path = Path(args.export_structure)
    data = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"structure_export": {"path": str(path), "nodes": len(report["nodes"]),
                                 "graph_scope": report.get("graph_scope")}}


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
            _add_supply_arguments(cmd)
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
    _add_supply_arguments(operators)
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
    graph.add_argument("--export-structure",
                       help="also write the same geometry's structure export JSON here")
    _add_supply_arguments(graph)
    graph.add_argument("--output", "-o")
    # Delegated commands keep their own parser and help text.
    structure = sub.add_parser(
        "structure", help="export declared operator structure without timing",
        add_help=False)
    structure.add_argument("arguments", nargs=argparse.REMAINDER)
    cohorts = sub.add_parser(
        "compare-cohorts", help="compare declared cohort variants of one case",
        add_help=False)
    cohorts.add_argument("arguments", nargs=argparse.REMAINDER)
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
    compose = sub.add_parser("compose", help="deterministically synthesize a complete simulation Case")
    compose.add_argument("--model")
    compose.add_argument("--hardware")
    for flag in ("--tp", "--dp", "--ep", "--pp", "--pcp", "--dcp", "--concurrency"):
        compose.add_argument(flag, type=int)
    compose.add_argument("--sp", action="store_true")
    for flag in ("--num-devices", "--ranks-per-node", "--layers", "--max-num-seqs", "--max-num-batched-tokens", "--max-model-len"):
        compose.add_argument(flag, type=int)
    compose.add_argument("--quant")
    compose.add_argument("--experts", type=int, dest="experts")
    compose.add_argument("--top-k", type=int, dest="top_k")
    compose.add_argument("--prompt-tokens", type=int)
    compose.add_argument("--output-tokens", type=int)
    compose.add_argument("--prefix-tokens", type=int)
    compose.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    compose.add_argument("--graph-mode", choices=("none", "full", "piecewise"))
    compose.add_argument("--mtp-tokens", type=int)
    compose.add_argument("--mtp-accept", type=float)
    compose.add_argument("--ttft-slo-ms", type=float)
    compose.add_argument("--tpot-slo-ms", type=float)
    compose.add_argument("--va-recommended", action="store_true",
                         help="apply a documented vllm-ascend simulation subset; preserve quant without a hardware-specific reference")
    compose.add_argument("--stack", default="vllm-ascend",
                         help="registered runtime stack name or manifest path")
    compose.add_argument("--stack-manifest", default=None,
                         help="explicit JSON stack manifest path")
    compose.add_argument("--feature-track", default="merged",
                         choices=("merged", "unmerged_pr", "expected_ideal"))
    compose.add_argument("--enable-opt", action="append", default=[],
                         help="enable individual OptimizationMechanism ID (repeatable)")
    compose.add_argument("--disable-opt", action="append", default=[],
                         help="exclude individual OptimizationMechanism ID (repeatable)")
    compose.add_argument("--block-size", type=int, default=None,
                         help="explicit KV block size (auto-resolved when omitted)")
    compose.add_argument("--output", "-o")
    from .cli_options import add_runtime_arguments
    add_runtime_arguments(compose, command="compose")
    feature_tracks = sub.add_parser(
        "feature-tracks",
        help="4-level waterfall (merged -> unmerged_pr -> expected_ideal -> roofline)")
    feature_tracks.add_argument("--model", required=True)
    feature_tracks.add_argument("--hardware", default="a3")
    feature_tracks.add_argument("--stack", default="vllm-ascend")
    feature_tracks.add_argument("--stack-manifest", default=None)
    feature_tracks.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    feature_tracks.add_argument("--tokens", type=int, default=1)
    from .feature_waterfall import (
        DEFAULT_WATERFALL_CONTEXT, DEFAULT_WATERFALL_SEQUENCES,
    )
    feature_tracks.add_argument("--context", type=int,
                                default=DEFAULT_WATERFALL_CONTEXT)
    feature_tracks.add_argument("--sequences", type=int,
                                default=DEFAULT_WATERFALL_SEQUENCES)
    feature_tracks.add_argument("--scope", choices=("body", "model"), default="body")
    for flag in ("--tp", "--dp", "--ep", "--pp", "--pcp", "--dcp",
                 "--num-devices", "--layers", "--mtp-tokens"):
        feature_tracks.add_argument(flag, type=int, default=None)
    feature_tracks.add_argument("--quant", default=None)
    feature_tracks.add_argument("--jobs", default="auto",
                                help="worker budget for independent feature levels")
    feature_tracks.add_argument("--no-ablate-prs", dest="ablate_prs",
                                action="store_false")
    feature_tracks.add_argument("--output", "-o")
    sweep = sub.add_parser("sweep", help="multi-topology parallel sharding and P/D disaggregation Pareto solver")
    sweep.add_argument("case")
    sweep.add_argument("--mode", choices=("topology", "pd"), default="topology")
    sweep.add_argument("--decode-case")
    sweep.add_argument("--num-devices", type=int)
    sweep.add_argument("--sla-ms", type=float, default=30.0)
    sweep.add_argument("--ttft-sla-ms", type=float, default=2000.0)
    sweep.add_argument("--tpot-sla-ms", type=float, default=30.0)
    sweep.add_argument("--context", type=int, default=32768)
    sweep.add_argument("--prompt-tokens", type=int, default=4096)
    sweep.add_argument("--output-tokens", type=int, default=1024)
    sweep.add_argument("--output-budget", type=int, default=1024)
    sweep.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    sweep.add_argument("--scope", choices=("body", "model"), default="body")
    sweep.add_argument("--allow-pp", action="store_true")
    sweep.add_argument("--allow-pcp", action="store_true")
    sweep.add_argument("--allow-dcp", action="store_true")
    sweep.add_argument("--dcp-candidates", default=None,
                       help="comma-separated DCP degrees")
    sweep.add_argument("--num-devices-grid", default=None,
                       help="comma-separated device counts, e.g. 2,4,8")
    sweep.add_argument("--tokens-per-seq", type=int)
    sweep.add_argument("--max-sequences-per-dp", type=int,
                       default=SWEEP_DEFAULT_MAX_SEQUENCES)
    sweep.add_argument("--candidates", default=None,
                       help="JSON list of explicit candidate mappings")
    sweep.add_argument("--two-stage", action="store_true")
    sweep.add_argument("--finalist-top-k", type=int, default=None)
    sweep.add_argument("--candidate-timeout-s", type=float, default=None)
    sweep.add_argument("--output-jsonl", default=None)
    sweep.add_argument("--checkpoint-jsonl", default=None)
    sweep.add_argument("--isolate-workers", action="store_true")
    sweep.add_argument("--resume", action="store_true")
    sweep.add_argument("--progress", action="store_true")
    sweep.add_argument("--output", "-o")
    add_runtime_arguments(sweep, command="sweep")
    sweep.add_argument("--jobs", default="auto",
                       help="worker budget for independent topology candidates")
    sweep.add_argument("--hardware-derivations", default=None,
                       help="JSON file with named derivation records")
    eff = sub.add_parser("efficiency", help="operator-level MBU/MFU decomposition and roofline target evaluation")
    eff.add_argument("case")
    eff.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    eff.add_argument("--tokens", type=int, default=1)
    eff.add_argument("--context", type=int, default=4096)
    eff.add_argument("--sequences", type=int, default=1)
    eff.add_argument("--scope", choices=("body", "model"), default="body")
    for flag in ("--target-mbu", "--target-cube-mfu", "--target-vec-mfu", "--target-link-util", "--fixed-launch-us"):
        eff.add_argument(flag, type=float)
    eff.add_argument("--mode", choices=("clamp_to_target", "exact_target"), default="clamp_to_target")
    eff.add_argument("--fit-targets", action="store_true")
    eff.add_argument("--observed-durations")
    eff.add_argument("--pmu")
    eff.add_argument("--skip-graph", action="store_true")
    eff.add_argument("--output", "-o")
    sla = sub.add_parser("sla", help="SLA-constrained concurrency and hardware scaling solver")
    sla.add_argument("case")
    sla.add_argument("--sla-ms", type=float, default=30.0)
    sla.add_argument("--context", type=int, default=126000)
    sla.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    sla.add_argument("--scope", choices=("body", "model"), default="body")
    sla.add_argument("--tokens-per-seq", type=int)
    sla.add_argument("--emitted-tokens-per-seq", type=float)
    sla.add_argument("--max-sequences-per-dp", type=int, default=256)
    sla.add_argument("--output-budget", type=int, default=1024)
    sla.add_argument("--target-mbu", type=float, default=0.80)
    sla.add_argument("--target-cube-mfu", type=float, default=0.75)
    sla.add_argument("--target-vec-mfu", type=float, default=0.65)
    for flag in ("--hbm-bw-scale", "--cube-flops-scale", "--vector-ops-scale", "--link-bw-scale", "--hbm-capacity-scale"):
        sla.add_argument(flag, type=float)
    sla.add_argument("--scenarios")
    sla.add_argument("--sensitivity", choices=("baseline", "full"),
                     default="baseline",
                     help="baseline = one counterfactual (fast default); "
                          "full = all documented counterfactuals")
    sla.add_argument("--jobs", default="auto",
                     help="worker budget for the flat (sku, scenario) pool")
    sla.add_argument("--output", "-o")
    add_runtime_arguments(sla, command="sla")
    compare = sub.add_parser(
        "compare-skus",
        help="compare two or more hardware SKUs under one model/deployment config")
    compare.add_argument("--model", required=True)
    compare.add_argument("--skus", nargs="+", required=True,
                         help="SKU names (a3, a5, ...) or custom hardware JSON paths")
    for flag in ("--tp", "--dp", "--ep", "--pp", "--pcp", "--num-devices",
                 "--layers"):
        compare.add_argument(flag, type=int, default=None)
    compare.add_argument("--sla-ms", type=float,
                         default=DEFAULT_COMPARE_SLA_MS)
    compare.add_argument("--max-batch", type=int, default=None)
    compare.add_argument("--sensitivity", choices=("baseline", "full"),
                         default="baseline")
    compare.add_argument("--scope", choices=("body", "model"), default="body")
    compare.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    compare.add_argument("--context", type=int, default=None)
    compare.add_argument("--query-tokens", type=int, default=None)
    compare.add_argument("--quant", default=None,
                         help="explicit quantization override for every SKU")
    compare.add_argument("--jobs", default="auto")
    compare.add_argument("--no-advisory", dest="emit_advisory",
                         action="store_false")
    compare.add_argument("--output", "-o")
    doctor = sub.add_parser(
        "doctor", help="probe CPU/RAM/C11 readiness and print agent offload guidance")
    doctor.add_argument("--no-compile", dest="compile_accel",
                        action="store_false")
    doctor.add_argument("--output", "-o")
    guide = sub.add_parser("remote-guide", help="alias of `doctor` for agent offload")
    guide.add_argument("--no-compile", dest="compile_accel",
                       action="store_false")
    guide.add_argument("--output", "-o")
    accel = sub.add_parser("build-accel", help="compile or verify the C11 runtime kernel")
    accel.add_argument("--force", action="store_true", help="recompile even when cached")
    accel.add_argument("--check", action="store_true", help="verify cached artifact without compiling")
    accel.add_argument("--output", "-o")
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
        case = parse_case(_case_with_supply(args))
        graphs = target_graphs(case, phase=args.phase, tokens=args.tokens, context=args.context,
                               sequences=args.sequences, scope=args.scope,
                               media_tokens_per_sequence=args.media_tokens_per_sequence,
                               draft_prepare=args.draft_prepare,
                               apply_body_defaults=args.apply_body_defaults)
        inputs = {"analysis_input": args.input_ready_ms} if args.scope == "body" else None
        exported = {}
        if args.export_structure:
            exported = _export_structure(
                args, case, phase=args.phase, tokens=args.tokens,
                context=args.context, sequences=args.sequences)
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
                **exported,
            }
    if args.command == "operators":
        from .prediction.operator_analysis import operator_costs, target_graphs
        case = parse_case(_case_with_supply(args))
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
            return predict(_case_with_supply(args), primitives=store,
                           trace_limit=args.trace_limit, event_limit=args.event_limit,
                           scope=args.scope, phase=args.phase, tokens=args.tokens,
                           context=args.context, sequences=args.sequences)
    if args.command == "inspect":
        return _inspect(read_json(args.case))
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
    if args.command == "compose":
        from .compose import compose_case
        from .cli_options import compose_case_kwargs
        if not args.model:
            raise ValueError("compose_requires_model")
        return compose_case(
            model=args.model, hardware=args.hardware or "a3",
            **compose_case_kwargs(args),
        )
    if args.command == "feature-tracks":
        from .feature_waterfall import compare_feature_tracks
        return compare_feature_tracks(
            args.model, args.hardware, runtime_stack=(
                args.stack_manifest or args.stack),
            phase=args.phase, tokens=args.tokens, context=args.context,
            sequences=args.sequences, scope=args.scope,
            ablate_prs=bool(args.ablate_prs),
            tp=args.tp or 1, dp=args.dp or 1, ep=args.ep or 1,
            pp=args.pp or 1, pcp=args.pcp or 1, dcp=args.dcp or 1,
            num_devices=args.num_devices, layers=args.layers,
            quant=args.quant,
            mtp_proposal_tokens=args.mtp_tokens or 0,
        )
    if args.command == "sweep":
        from .sweep import sweep_pd_disaggregation_ratios, sweep_topologies
        from .cli_options import apply_case_mapping_overrides, case_overrides
        case_data = apply_case_mapping_overrides(
            read_json(args.case), case_overrides(args))
        if args.mode == "pd":
            decode_data = read_json(args.decode_case) if args.decode_case else case_data
            return sweep_pd_disaggregation_ratios(
                case_data, decode_data, total_dies=args.num_devices or 16,
                prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
                ttft_sla_ms=args.ttft_sla_ms, tpot_sla_ms=args.tpot_sla_ms,
            )
        from .cli_options import load_hardware_derivations
        if args.resume and not (args.output_jsonl or args.checkpoint_jsonl):
            raise CliUsageError(
                "--resume requires --checkpoint-jsonl or --output-jsonl")
        grids = ([int(item) for item in str(args.num_devices_grid).split(",")
                  if item.strip()]
                 if args.num_devices_grid else None)
        dcp_candidates = ([int(item) for item in str(args.dcp_candidates).split(",")
                            if item.strip()]
                           if args.dcp_candidates else None)
        explicit = (read_json(args.candidates) if args.candidates else None)
        derivations = (load_hardware_derivations(args.hardware_derivations)
                       if args.hardware_derivations else None)
        progress = None
        if args.progress:
            def progress(row):
                print(
                    f"{row.get('candidate_id', '')} "
                    f"{row.get('status', '')} "
                    f"{row.get('evaluation_stage', '')} "
                    f"d={row.get('num_devices', '')} "
                    f"b1={row.get('step_ms_at_b1')}",
                    file=sys.stderr, flush=True)
        return sweep_topologies(
            case_data, candidates=explicit,
            num_devices=args.num_devices, num_devices_grid=grids,
            sla_ms=args.sla_ms, context=args.context,
            tokens_per_seq=args.tokens_per_seq,
            output_budget=args.output_budget,
            max_sequences_per_dp=args.max_sequences_per_dp,
            phase=args.phase, scope=args.scope,
            allow_pp=args.allow_pp, allow_pcp=args.allow_pcp,
            allow_dcp=args.allow_dcp, dcp_candidates=dcp_candidates,
            hardware_derivations=derivations,
            two_stage=args.two_stage,
            finalist_top_k=args.finalist_top_k,
            candidate_timeout_s=args.candidate_timeout_s,
            output_jsonl=args.output_jsonl,
            checkpoint_jsonl=args.checkpoint_jsonl,
            isolate_workers=args.isolate_workers,
            resume=args.resume,
            progress_callback=progress,
        )
    if args.command == "efficiency":
        from .prediction.efficiency import analyze_network_efficiency, fit_efficiency_targets
        case_data = read_json(args.case)
        if args.fit_targets or any(
            v is not None for v in (args.target_mbu, args.target_cube_mfu, args.target_vec_mfu, args.target_link_util)
        ):
            fit_res = fit_efficiency_targets(
                case_data, phase=args.phase, tokens=args.tokens, context=args.context,
                sequences=args.sequences, scope=args.scope,
                target_mbu=0.80 if args.target_mbu is None else args.target_mbu,
                target_cube_mfu=0.75 if args.target_cube_mfu is None else args.target_cube_mfu,
                target_vec_mfu=0.65 if args.target_vec_mfu is None else args.target_vec_mfu,
                target_link_util=args.target_link_util, fixed_launch_us=args.fixed_launch_us,
                mode=args.mode,
            )
            return {**fit_res["baseline_analysis"], **fit_res, "schema": "mindie-sim.efficiency/1"}
        return analyze_network_efficiency(
            case_data, phase=args.phase, tokens=args.tokens, context=args.context,
            sequences=args.sequences, scope=args.scope,
            observed_durations_ms=read_json(args.observed_durations) if args.observed_durations else None,
            pmu_by_op=read_json(args.pmu) if args.pmu else None, run_graph=not args.skip_graph,
        )
    if args.command == "sla":
        from .prediction.efficiency import DEFAULT_SLA_SCENARIOS, solve_sla_sizing
        from .cli_options import apply_case_mapping_overrides, case_overrides
        case_data = apply_case_mapping_overrides(
            read_json(args.case), case_overrides(args))
        has_custom_scale = any(
            v is not None
            for v in (args.hbm_bw_scale, args.cube_flops_scale, args.vector_ops_scale, args.link_bw_scale, args.hbm_capacity_scale)
        )
        if args.scenarios:
            scenarios = read_json(args.scenarios)
        elif has_custom_scale:
            scenarios = [
                {
                    "id": "baseline", "label": "1x Baseline", "hbm_bw_scale": 1.0,
                    "cube_flops_scale": 1.0, "vector_ops_scale": 1.0,
                    "link_bw_scale": 1.0, "hbm_capacity_scale": 1.0,
                },
                {
                    "id": "scaled", "label": "Custom Scaled",
                    "hbm_bw_scale": 1.0 if args.hbm_bw_scale is None else args.hbm_bw_scale,
                    "cube_flops_scale": 1.0 if args.cube_flops_scale is None else args.cube_flops_scale,
                    "vector_ops_scale": 1.0 if args.vector_ops_scale is None else args.vector_ops_scale,
                    "link_bw_scale": 1.0 if args.link_bw_scale is None else args.link_bw_scale,
                    "hbm_capacity_scale": 1.0 if args.hbm_capacity_scale is None else args.hbm_capacity_scale,
                },
            ]
        else:
            # ``None`` lets the solver apply the requested sensitivity set, so
            # the out-of-the-box command is a single fast counterfactual.
            scenarios = None
        from . import runtime_advisor
        jobs = runtime_advisor.resolve_jobs(
            args.jobs, task_count=len(scenarios or DEFAULT_SLA_SCENARIOS))
        result = solve_sla_sizing(
            case_data, sla_ms=args.sla_ms, context=args.context,
            tokens_per_seq=args.tokens_per_seq, emitted_tokens_per_seq=args.emitted_tokens_per_seq,
            max_sequences_per_dp=args.max_sequences_per_dp, phase=args.phase,
            scope=args.scope, output_budget=args.output_budget, target_mbu=args.target_mbu,
            target_cube_mfu=args.target_cube_mfu, target_vec_mfu=args.target_vec_mfu,
            scenarios=scenarios, sensitivity=args.sensitivity, jobs=jobs,
        )
        result["sensitivity"] = args.sensitivity
        result["jobs"] = jobs
        return result
    if args.command == "compare-skus":
        from .sku_comparison import compare_skus
        options = {flag: getattr(args, flag.lstrip("-").replace("-", "_"), None)
                   for flag in ("tp", "dp", "ep", "pp", "pcp", "num_devices",
                                "layers")}
        if args.max_batch is None:
            from .prediction.efficiency import DEFAULT_MAX_SEQUENCES
            max_batch = DEFAULT_MAX_SEQUENCES
        else:
            max_batch = args.max_batch
        return compare_skus(
            args.model, args.skus, sla_ms=args.sla_ms, max_batch=max_batch,
            sensitivity=args.sensitivity, scope=args.scope, phase=args.phase,
            context=args.context, query_tokens_per_sequence=args.query_tokens,
            quant=args.quant, jobs=args.jobs,
            emit_advisory=bool(args.emit_advisory), **options)
    if args.command in {"doctor", "remote-guide"}:
        from .runtime_advisor import doctor_report
        report = doctor_report(compile_accel=bool(args.compile_accel))
        report["command"] = args.command
        return report
    if args.command == "build-accel":
        from .prediction.runtime.accel_build import build_accel

        return build_accel(force=bool(args.force), check_only=bool(args.check))
    raise ValueError("unknown_command")



def _inspect(raw_case):
    """Inspect the first real scheduled iteration under its effective declaration.

    Selection happens on the iteration's own views, padded bucket and replay
    scope, and the resolved declaration is handed to the same ``Frontend`` that
    execution uses, so a decode-only default is never installed over a prefill
    iteration.
    """
    from .prediction.attention import AttentionPolicy
    from .prediction.body_defaults import effective_body_declaration
    from .prediction.capacity import Memory
    from .prediction.frontend import Frontend
    from .prediction.graph import GraphPolicy
    from .prediction.implementation_selection import (
        combine_summaries, records_summary, resolve,
    )
    from .prediction.scheduler import Scheduler

    case = parse_case(raw_case)
    memory = Memory(case)
    scheduler = Scheduler(case, list(case.workload.requests), memory)
    plan = scheduler.schedule(min(r.arrival_ms for r in case.workload.requests))
    if plan is None:
        return {"case": case.to_dict(), "memory": memory.to_dict(), "operators": [],
                "operator_selection_summary": {}}
    views = tuple(plan.requests)
    phase = next((view.phase for view in views), "decode")
    effective, selection = effective_body_declaration(
        case, phase, tokens=max(1, sum(view.q_len for view in views)), scope="body")
    runtime = scheduler.next_runtime(plan)
    bucket, replay_scope = GraphPolicy().select(views, runtime)
    padded = max(sum(view.q_len for view in views), bucket or 0)
    runtime, records = resolve(runtime, views, padded_tokens=padded,
                               replay_scope=replay_scope)
    frontend = Frontend(attention_policy=AttentionPolicy(**runtime.attention_policy))
    work = frontend(plan, effective.model, runtime, effective.machine)
    summary = combine_summaries([
        selection.get("operator_selection_summary"),
        records_summary(arch_family=effective.machine.arch_family,
                        phase=phase, records=records),
    ])
    return {"case": case.to_dict(), "memory": memory.to_dict(),
            "operators": [op.to_dict() for op in work.ops],
            "operator_selection_summary": summary}


def _delegate(raw):
    """Run a registered sub-command's own parser, or return ``None``.

    The outer parser lists these commands but the delegate owns its complete
    argument surface, help text and exit codes.
    """
    if not raw:
        return None
    command, rest = raw[0], raw[1:]
    if command == "structure":
        from .structure import main as structure_main
        return structure_main(rest)
    if command == "compare-cohorts":
        from .cohort_comparison import main as cohorts_main
        return cohorts_main(rest)
    return None


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    delegated = _delegate(raw)
    if delegated is not None:
        return delegated
    args = parser().parse_args(raw)
    try:
        result = run(args)
        emit(result, args.output)
        if args.command in {"study", "compare-calls"} and result["status"] != "completed":
            return 2
        if args.command == "build-accel" and not result.get("ok"):
            return 2
    except TimeoutError:
        raise
    except (CliUsageError, TypeError, ValueError, OverflowError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
