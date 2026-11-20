"""Unified CLI route resolver and argument parser for ``mindie-sim``.

[Role] Define ``CliRoute``, ``resolve_route`` and ``build_parser`` covering all
  29 legacy commands plus the canonical multi-word subcommands (``predict
  service|step|tiers|call``, ``inspect case|structure``, ``compare
  cohorts|skus|study``, ``obs ...``, ``calibrate record|bind-profiler|fit``,
  ``doctor remote``).
[Invariants] Route resolution uses only command paths, option arity and
  required positional counts (never probes the filesystem), so legacy paths
  named ``step`` or ``record`` remain unambiguous. ``--`` is preserved intact.
[Timeouts & Locks] Pure argument-vector parsing; no I/O or locks.
[Side-Effect Policy] Zero side effects.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from .cli_options import add_runtime_arguments
from .cli_sku import add_sku_commands
from .feature_waterfall import DEFAULT_WATERFALL_CONTEXT, DEFAULT_WATERFALL_SEQUENCES
from .prediction.results import DEFAULT_TARGET_EFFICIENCY

DEFAULT_TRACE_LIMIT = 2
DEFAULT_EVENT_LIMIT = 256
DEFAULT_PROFILER_ROW_LIMIT = 50
DEFAULT_MAX_PREDICTIONS = 512
DEFAULT_MAX_CONDITIONS = 4096
DEFAULT_SLA_MS = 30.0
DEFAULT_TTFT_SLA_MS = 2000.0
DEFAULT_TPOT_SLA_MS = 30.0
DEFAULT_SWEEP_CONTEXT = 32768
DEFAULT_PROMPT_TOKENS = 4096
DEFAULT_OUTPUT_BUDGET = 1024
DEFAULT_SWEEP_MAX_SEQUENCES = 256
DEFAULT_EFF_CONTEXT = 4096
DEFAULT_SLA_CONTEXT = 126000
DEFAULT_TARGET_MBU = 0.80
DEFAULT_TARGET_CUBE_MFU = 0.75
DEFAULT_TARGET_VEC_MFU = 0.65

_BOOLEAN_FLAGS = frozenset({
    "-h", "--help", "--isolated", "--include-submission", "--draft-prepare",
    "--no-body-defaults", "--skip-isolated", "--skip-graph", "--fit-targets",
    "--supply-legacy-diagnostics", "--sp", "--va-recommended", "--no-ablate-prs",
    "--allow-pp", "--allow-pcp", "--allow-dcp", "--two-stage", "--isolate-workers",
    "--resume", "--progress", "--no-sla", "--no-advisory", "--no-compile",
    "--force", "--check", "--discover-installed", "--async-scheduling",
    "--no-async-scheduling", "--enable-dbo", "--no-enable-dbo",
    "--multistream-overlap-shared-expert", "--no-multistream-overlap-shared-expert",
})

DELEGATED_COMMANDS = frozenset({
    "structure", "compare-cohorts", "extensions", "profile",
})


@dataclass(frozen=True, slots=True)
class CliRoute:
    """Resolved CLI route mapping user argv tokens to one internal command."""

    path: tuple[str, ...]
    command: str
    legacy_profile: str | None = None


def _count_positionals(tokens: Sequence[str]) -> int:
    count = 0
    idx = 0
    total = len(tokens)
    while idx < total:
        tok = tokens[idx]
        if tok == "--":
            count += total - idx - 1
            break
        if tok.startswith("-") and tok != "-":
            if "=" not in tok and tok not in _BOOLEAN_FLAGS:
                idx += 2
                continue
            idx += 1
            continue
        count += 1
        idx += 1
    return count


def _split_double_dash(argv: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    for idx, tok in enumerate(argv):
        if tok == "--":
            return tuple(argv[:idx]), tuple(argv[idx:])
    return tuple(argv), ()


def resolve_route(argv: Sequence[str], /) -> tuple[CliRoute, tuple[str, ...]]:
    """Resolve canonical and legacy command paths without touching the filesystem."""
    head, tail = _split_double_dash(argv)
    if not head:
        return CliRoute((), ""), ()
    first = head[0]
    rest = head[1:] + tail

    if first in {"extension", "ext"}:
        return CliRoute((first,), "extensions", legacy_profile=first), rest
    if first == "extensions":
        return CliRoute(("extensions",), "extensions"), rest
    if first == "profile":
        return CliRoute(("profile",), "profile"), rest

    if first == "predict" and len(head) >= 2:
        sub = head[1]
        after = head[2:] + tail
        pos = _count_positionals(after)
        if sub == "service" and pos >= 1:
            return CliRoute(("predict", "service"), "predict-service"), after
        if sub == "step" and pos >= 1:
            return CliRoute(("predict", "step"), "predict-step"), after
        if sub == "tiers" and pos >= 1:
            return CliRoute(("predict", "tiers"), "predict-tiers"), after
        if sub == "call" and pos >= 2:
            return CliRoute(("predict", "call"), "call"), after
        return CliRoute(("predict",), "predict", legacy_profile="predict"), rest

    if first == "inspect" and len(head) >= 2:
        sub = head[1]
        after = head[2:] + tail
        pos = _count_positionals(after)
        if sub == "case" and pos >= 1:
            return CliRoute(("inspect", "case"), "inspect"), after
        if sub == "structure" and pos >= 1:
            return CliRoute(("inspect", "structure"), "structure"), after
        return CliRoute(("inspect",), "inspect", legacy_profile="inspect"), rest

    if first == "compare" and len(head) >= 2:
        sub = head[1]
        after = head[2:] + tail
        if sub == "cohorts":
            return CliRoute(("compare", "cohorts"), "compare-cohorts"), after
        if sub == "skus":
            return CliRoute(("compare", "skus"), "compare-skus"), after
        if sub == "study":
            return CliRoute(("compare", "study"), "study"), after

    if first == "obs" and len(head) >= 2:
        sub = head[1]
        after = head[2:] + tail
        if sub == "compare-calls":
            return CliRoute(("obs", "compare-calls"), "compare-calls"), after
        if sub in {"inventory", "index"}:
            return CliRoute(("obs", sub), "observations"), after
        if sub == "find" and len(head) >= 4:
            rewritten = (head[2], "--find", head[3], *head[4:], *tail)
            return CliRoute(("obs", "find"), "observations"), rewritten
        if sub == "rows":
            return CliRoute(("obs", "rows"), "profiler-rows"), after
        if sub == "validate":
            return CliRoute(("obs", "validate"), "validate"), after
        if sub == "accuracy":
            return CliRoute(("obs", "accuracy"), "accuracy"), after

    if first == "calibrate" and len(head) >= 2:
        sub = head[1]
        after = head[2:] + tail
        pos = _count_positionals(after)
        if sub == "record" and pos >= 2:
            return CliRoute(("calibrate", "record"), "calibrate"), after
        if sub == "bind-profiler":
            return CliRoute(("calibrate", "bind-profiler"), "bind-profiler"), after
        if sub == "fit" and len(head) >= 3:
            fit_kind = head[2]
            fit_after = head[3:] + tail
            if fit_kind == "rate":
                return CliRoute(("calibrate", "fit", "rate"), "fit-rate"), fit_after
            if fit_kind == "service":
                return CliRoute(("calibrate", "fit", "service"), "fit-service"), fit_after
        return CliRoute(("calibrate",), "calibrate", legacy_profile="calibrate"), rest

    if first == "doctor" and len(head) >= 2 and head[1] == "remote":
        return CliRoute(("doctor", "remote"), "remote-guide"), head[2:] + tail

    return CliRoute((first,), first), rest


def _add_supply_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--supply", help="profiler C7 report or mindie-profiler-measured-supply/1 JSON"
    )
    command.add_argument("--supply-recipe", help="selected operator recipe to bind")
    command.add_argument("--supply-kernel", help="selected native kernel name to bind")
    command.add_argument("--supply-hardware", help="explicit hardware key for the binding")
    command.add_argument("--supply-owners", type=int,
                         help="selected implementation active-owner default")
    command.add_argument("--supply-legacy-diagnostics", action="store_true",
                         help="retain legacy service_regimes alongside C7 resource_services")


def _add_sdk_step_parsers(sub) -> None:
    pred_svc = sub.add_parser("predict-service", help="evaluate serving workload via SDK")
    pred_svc.add_argument("case")
    pred_svc.add_argument("--primitives")
    pred_svc.add_argument("--event-limit", type=int, default=0)
    _add_supply_arguments(pred_svc)
    pred_svc.add_argument("--output", "-o")

    pred_step = sub.add_parser("predict-step", help="evaluate fixed step via PredictionSpine")
    pred_step.add_argument("case")
    pred_step.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    pred_step.add_argument("--tokens", type=int, default=1)
    pred_step.add_argument("--context", type=int, default=0)
    pred_step.add_argument("--sequences", type=int, default=1)
    pred_step.add_argument("--scope", choices=("body", "model"), default="body")
    pred_step.add_argument("--media-tokens-per-sequence", type=int)
    pred_step.add_argument("--draft-prepare", action="store_true")
    pred_step.add_argument("--view", choices=("step", "operators", "graph", "efficiency"),
                           default="step")
    pred_step.add_argument("--isolated", action="store_true")
    pred_step.add_argument("--include-submission", action="store_true")
    pred_step.add_argument("--event-limit", type=int, default=0)
    pred_step.add_argument("--primitives")
    _add_supply_arguments(pred_step)
    pred_step.add_argument("--output", "-o")

    pred_tiers = sub.add_parser(
        "predict-tiers",
        help="evaluate unmerge / mainline / should_achieve via PredictionSpine",
    )
    pred_tiers.add_argument("case")
    pred_tiers.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    pred_tiers.add_argument("--tokens", type=int, default=1)
    pred_tiers.add_argument("--context", type=int, default=0)
    pred_tiers.add_argument("--sequences", type=int, default=1)
    pred_tiers.add_argument("--scope", choices=("body", "model"), default="body")
    pred_tiers.add_argument("--target-eff", type=float, default=DEFAULT_TARGET_EFFICIENCY)
    pred_tiers.add_argument("--isolated", action="store_true")
    pred_tiers.add_argument("--event-limit", type=int, default=0)
    pred_tiers.add_argument("--primitives")
    _add_supply_arguments(pred_tiers)
    pred_tiers.add_argument("--output", "-o")


def build_parser(*, prog: str = "mindie-sim") -> argparse.ArgumentParser:
    """Build the complete argument parser for all legacy and canonical commands."""
    root = argparse.ArgumentParser(prog=prog)
    sub = root.add_subparsers(dest="command", required=True)
    _add_sdk_step_parsers(sub)

    for name in ("predict", "inspect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("case")
        cmd.add_argument("--output", "-o")
        if name == "predict":
            cmd.add_argument("--primitives")
            cmd.add_argument("--trace-limit", type=int, default=DEFAULT_TRACE_LIMIT)
            cmd.add_argument("--event-limit", type=int, default=DEFAULT_EVENT_LIMIT)
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

    structure = sub.add_parser(
        "structure", help="export declared operator structure without timing", add_help=False
    )
    structure.add_argument("arguments", nargs=argparse.REMAINDER)
    cohorts = sub.add_parser(
        "compare-cohorts", help="compare declared cohort variants of one case", add_help=False
    )
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
    rows.add_argument("--limit", type=int, default=DEFAULT_PROFILER_ROW_LIMIT)
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
    alignment.add_argument("--max-predictions", type=int, default=DEFAULT_MAX_PREDICTIONS)
    alignment.add_argument("--max-conditions", type=int, default=DEFAULT_MAX_CONDITIONS)
    alignment.add_argument("--output", "-o")

    compose = sub.add_parser("compose", help="deterministically synthesize a complete simulation Case")
    compose.add_argument("--model")
    compose.add_argument("--hardware")
    for flag in ("--tp", "--dp", "--ep", "--pp", "--pcp", "--dcp", "--concurrency"):
        compose.add_argument(flag, type=int)
    compose.add_argument("--sp", action="store_true")
    for flag in ("--num-devices", "--ranks-per-node", "--layers", "--max-num-seqs",
                 "--max-num-batched-tokens", "--max-model-len"):
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
                         choices=("raw_eager", "merged", "unmerged_pr", "expected_ideal"))
    compose.add_argument("--enable-opt", action="append", default=[],
                         help="enable individual OptimizationMechanism ID (repeatable)")
    compose.add_argument("--disable-opt", action="append", default=[],
                         help="exclude individual OptimizationMechanism ID (repeatable)")
    compose.add_argument("--block-size", type=int, default=None,
                         help="explicit KV block size (auto-resolved when omitted)")
    compose.add_argument("--output", "-o")
    add_runtime_arguments(compose, command="compose")

    feature_tracks = sub.add_parser(
        "feature-tracks",
        help="4-level waterfall (merged -> unmerged_pr -> expected_ideal -> roofline)",
    )
    feature_tracks.add_argument("--model", required=True)
    feature_tracks.add_argument("--hardware", default="a3")
    feature_tracks.add_argument("--stack", default="vllm-ascend")
    feature_tracks.add_argument("--stack-manifest", default=None)
    feature_tracks.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    feature_tracks.add_argument("--tokens", type=int, default=1)
    feature_tracks.add_argument("--context", type=int, default=DEFAULT_WATERFALL_CONTEXT)
    feature_tracks.add_argument("--sequences", type=int, default=DEFAULT_WATERFALL_SEQUENCES)
    feature_tracks.add_argument("--scope", choices=("body", "model"), default="body")
    for flag in ("--tp", "--dp", "--ep", "--pp", "--pcp", "--dcp",
                 "--num-devices", "--layers", "--mtp-tokens"):
        feature_tracks.add_argument(flag, type=int, default=None)
    feature_tracks.add_argument("--quant", default=None)
    feature_tracks.add_argument("--jobs", default="auto",
                                help="worker budget for independent feature levels")
    feature_tracks.add_argument("--no-ablate-prs", dest="ablate_prs", action="store_false")
    feature_tracks.add_argument("--output", "-o")

    sweep = sub.add_parser("sweep", help="multi-topology parallel sharding and P/D disaggregation Pareto solver")
    sweep.add_argument("case")
    sweep.add_argument("--mode", choices=("topology", "pd"), default="topology")
    sweep.add_argument("--decode-case")
    sweep.add_argument("--num-devices", type=int)
    sweep.add_argument("--sla-ms", type=float, default=DEFAULT_SLA_MS)
    sweep.add_argument("--ttft-sla-ms", type=float, default=DEFAULT_TTFT_SLA_MS)
    sweep.add_argument("--tpot-sla-ms", type=float, default=DEFAULT_TPOT_SLA_MS)
    sweep.add_argument("--context", type=int, default=DEFAULT_SWEEP_CONTEXT)
    sweep.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    sweep.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_BUDGET)
    sweep.add_argument("--output-budget", type=int, default=DEFAULT_OUTPUT_BUDGET)
    sweep.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    sweep.add_argument("--scope", choices=("body", "model"), default="body")
    sweep.add_argument("--allow-pp", action="store_true")
    sweep.add_argument("--allow-pcp", action="store_true")
    sweep.add_argument("--allow-dcp", action="store_true")
    sweep.add_argument("--dcp-candidates", default=None, help="comma-separated DCP degrees")
    sweep.add_argument("--num-devices-grid", default=None,
                       help="comma-separated device counts, e.g. 2,4,8")
    sweep.add_argument("--tokens-per-seq", type=int)
    sweep.add_argument("--max-sequences-per-dp", type=int, default=DEFAULT_SWEEP_MAX_SEQUENCES)
    sweep.add_argument("--candidates", default=None, help="JSON list of explicit candidate mappings")
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
    eff.add_argument("--context", type=int, default=DEFAULT_EFF_CONTEXT)
    eff.add_argument("--sequences", type=int, default=1)
    eff.add_argument("--scope", choices=("body", "model"), default="body")
    for flag in ("--target-mbu", "--target-cube-mfu", "--target-vec-mfu",
                 "--target-link-util", "--fixed-launch-us"):
        eff.add_argument(flag, type=float)
    eff.add_argument("--mode", choices=("clamp_to_target", "exact_target"), default="clamp_to_target")
    eff.add_argument("--fit-targets", action="store_true")
    eff.add_argument("--observed-durations")
    eff.add_argument("--pmu")
    eff.add_argument("--skip-graph", action="store_true")
    eff.add_argument("--output", "-o")

    sla = sub.add_parser("sla", help="SLA-constrained concurrency and hardware scaling solver")
    sla.add_argument("case")
    sla.add_argument("--sla-ms", type=float, default=DEFAULT_SLA_MS)
    sla.add_argument("--context", type=int, default=DEFAULT_SLA_CONTEXT)
    sla.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    sla.add_argument("--scope", choices=("body", "model"), default="body")
    sla.add_argument("--tokens-per-seq", type=int)
    sla.add_argument("--emitted-tokens-per-seq", type=float)
    sla.add_argument("--max-sequences-per-dp", type=int, default=DEFAULT_SWEEP_MAX_SEQUENCES)
    sla.add_argument("--output-budget", type=int, default=DEFAULT_OUTPUT_BUDGET)
    sla.add_argument("--target-mbu", type=float, default=DEFAULT_TARGET_MBU)
    sla.add_argument("--target-cube-mfu", type=float, default=DEFAULT_TARGET_CUBE_MFU)
    sla.add_argument("--target-vec-mfu", type=float, default=DEFAULT_TARGET_VEC_MFU)
    for flag in ("--hbm-bw-scale", "--cube-flops-scale", "--vector-ops-scale",
                 "--link-bw-scale", "--hbm-capacity-scale"):
        sla.add_argument(flag, type=float)
    sla.add_argument("--scenarios")
    sla.add_argument("--sensitivity", choices=("baseline", "full"), default="baseline",
                     help="baseline = one counterfactual (fast default); full = all documented counterfactuals")
    sla.add_argument("--jobs", default="auto",
                     help="worker budget for the flat (sku, scenario) pool")
    sla.add_argument("--output", "-o")
    add_runtime_arguments(sla, command="sla")

    add_sku_commands(sub)

    accel = sub.add_parser("build-accel", help="compile or verify the C11 runtime kernel")
    accel.add_argument("--force", action="store_true", help="recompile even when cached")
    accel.add_argument("--check", action="store_true", help="verify cached artifact without compiling")
    accel.add_argument("--output", "-o")
    return root


__all__ = [
    "CliRoute",
    "DELEGATED_COMMANDS",
    "build_parser",
    "resolve_route",
]
