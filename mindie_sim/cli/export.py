"""CLI subcommands and standalone entrypoint for ``report``, ``quickstart``, and ``export*``.

[Role] Register and dispatch:
  (1) Turnkey ``mindie-sim report`` & ``mindie-sim quickstart`` (delegated to
      ``mindie_sim.cli.report``),
  (2) One-shot ``mindie-sim export --model ... --hardware ... --xlsx ...
      --trace ...`` (defaulting omitted/``auto`` ``--weight-quant`` to
      chip-matched ``auto_va`` production quantization),
  (3) ``export-trace`` (1:1 Ascend ``msprof`` ``trace_view.json`` from a case JSON),
  (4) ``export-xlsx`` (OOXML ``merge_cells`` comparison & Memory/Quant/PP workbook
      from case JSON(s); ``--kernel-designs-json`` is retired in favor of ``--rewrite-plan``).
[Invariants] ``--out``/``--output``/``-o`` targets the exported trace/xlsx file
  (stored in ``dest="export_output"``) so the top-level CLI JSON emitter never
  overwrites exported artifacts; optional ``--receipt`` writes the structured
  JSON summary. ``TimeoutError`` is never swallowed by ``except OSError:``.
[Timeouts & Locks] Pure argument construction; bounded JSON ingress (``<= 50 MiB``)
  and single-pass O(Archetypes) export delegation; no locks or device contexts.
[Side-Effect Policy] Writes only the caller-requested export output path(s) (and
  optional ``--receipt`` JSON path).
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from mindie_sim.export import (
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    build_export_bundle,
    compose_export_case,
    export_comparison_xlsx,
    export_msprof_trace,
)
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.schema.common import ModelingError

from .report import (
    REPORT_BOOLEAN_FLAGS,
    REPORT_COMMANDS,
    add_report_commands,
    parse_token_shorthand,
    resolve_auto_va_policy,
    run_quickstart_command,
    run_report_command,
)

#: Commands owned by this CLI module.
EXPORT_COMMANDS = frozenset({"export", "export-trace", "export-xlsx"}) | REPORT_COMMANDS
#: Boolean flags registered by this module for route-resolution token counting.
EXPORT_BOOLEAN_FLAGS = frozenset({
    "--no-pipe-lanes",
    "--no-counters",
    "--no-overlap",
    "--include-submission",
    "--engram-offload",
    "--no-engram-offload",
}) | REPORT_BOOLEAN_FLAGS
#: Bounded metadata JSON size ceiling (50 MiB) matching main CLI policy.
MAX_EXPORT_CASE_JSON_BYTES = 50 * 1024 * 1024
#: Standard CLI error exit code.
CLI_ERROR_EXIT = 2


def _read_bounded_json_object(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{label}_file_not_found:{target}")
    if target.stat().st_size > MAX_EXPORT_CASE_JSON_BYTES:
        raise ModelingError(f"{label}_json_exceeds_50_mib_limit")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelingError(f"{label}_json_must_be_object")
    return data


def _add_common_export_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("case", help="path to MindIE-Sim case JSON file")
    cmd.add_argument(
        "--out", "--output", "-o", dest="export_output", required=True,
        help="destination file path for exported artifact",
    )
    cmd.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    cmd.add_argument("--tokens", type=int, default=1)
    cmd.add_argument("--context", type=parse_token_shorthand, default=None)
    cmd.add_argument("--sequences", type=int, default=1)
    cmd.add_argument("--scope", choices=("body", "model"), default="model")
    cmd.add_argument(
        "--observations", "--kernel-csv", dest="observations", default=None,
        help="optional msprof kernel_details.csv or observation bundle JSON for 1-to-N reconciliation",
    )
    cmd.add_argument("--include-submission", action="store_true")
    cmd.add_argument(
        "--receipt", dest="output", default=None,
        help="optional JSON path to save the structured export receipt",
    )


def _add_oneshot_export_command(subparsers) -> None:
    cmd = subparsers.add_parser(
        "export",
        help="one-shot scenario export generating human-native comparison .xlsx and/or msprof trace_view.json",
    )
    cmd.add_argument("--model", required=True, help="catalog model, alias, HF config.json, or model directory")
    cmd.add_argument("--hardware", required=True, help="hardware SKU or alias (e.g. 950dt, a5-pod, a3, a2)")
    cmd.add_argument("--topology-preset", default=None, help="optional topology preset override (e.g. a5_pod, a3_16p)")
    cmd.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    cmd.add_argument("--dp", type=int, default=1)
    cmd.add_argument("--tp", type=int, default=1)
    cmd.add_argument("--dcp", type=int, default=1)
    cmd.add_argument("--ep", type=int, default=1)
    cmd.add_argument("--pp", type=int, default=1)
    cmd.add_argument("--pcp", type=int, default=1)
    cmd.add_argument(
        "--prompt-tokens", "--context", dest="prompt_tokens",
        type=parse_token_shorthand, default=DEFAULT_SCENARIO_PROMPT_TOKENS,
    )
    cmd.add_argument("--output-tokens", type=parse_token_shorthand, default=DEFAULT_SCENARIO_OUTPUT_TOKENS)
    cmd.add_argument("--concurrency", type=int, default=1)
    cmd.add_argument("--tpot-ms", type=float, default=None)
    cmd.add_argument("--ttft-ms", type=float, default=None)
    cmd.add_argument("--spec-tokens", "--mtp", dest="spec_tokens", type=int, default=0)
    cmd.add_argument("--accept-len", type=float, default=1.0)
    cmd.add_argument("--stack", default=None, help="runtime stack (vllm-ascend, ideal-target, raw-eager, diffusion-native, or /path/to/manifest.json)")
    cmd.add_argument(
        "--compare-track",
        choices=("eager", "shipping", "proposed", "target"),
        default="shipping",
        help="baseline ('Before') stack track (default: shipping)",
    )
    cmd.add_argument(
        "--target-track",
        choices=("eager", "shipping", "proposed", "target"),
        default=None,
        help="optional candidate ('After') stack track (e.g. target)",
    )
    cmd.add_argument("--xlsx", dest="xlsx_output", default=None, help="destination path for comparison .xlsx workbook")
    cmd.add_argument("--trace", dest="trace_output", default=None, help="destination path for msprof trace_view.json")
    cmd.add_argument("--trace-side", choices=("before", "after"), default="after", help="which track to emit into --trace when comparing")
    cmd.add_argument("--weight-quant", default="auto_va", help="weight quant (default: auto_va; or checkpoint/bf16/w4a8/mxfp4)")
    cmd.add_argument("--cache-dtype", default=None)
    cmd.add_argument("--engram-offload", dest="engram_offload", action="store_true", default=None)
    cmd.add_argument("--no-engram-offload", dest="engram_offload", action="store_false")
    cmd.add_argument(
        "--deployment-mode",
        choices=("colocated", "pd_disaggregated", "pd_prefill", "pd_decode"),
        default=None,
        help="explicit deployment mode label for Summary Block 1",
    )
    cmd.add_argument("--observations", "--kernel-csv", dest="observations", default=None)
    cmd.add_argument("--include-submission", action="store_true")
    cmd.add_argument("--no-pipe-lanes", dest="include_pipe_lanes", action="store_false")
    cmd.add_argument("--no-counters", dest="include_counters", action="store_false")
    cmd.add_argument("--no-overlap", dest="include_overlap", action="store_false")
    cmd.add_argument("--receipt", dest="output", default=None, help="optional JSON path to save structured export receipt")


def add_export_commands(subparsers) -> None:
    """Register ``report``, ``quickstart``, ``export``, ``export-trace``, and ``export-xlsx``."""
    add_report_commands(subparsers)
    _add_oneshot_export_command(subparsers)

    trace_cmd = subparsers.add_parser(
        "export-trace",
        help="stream 1:1 Ascend msprof trace_view.json opening natively in MindStudio Insight",
    )
    _add_common_export_args(trace_cmd)
    trace_cmd.add_argument("--no-pipe-lanes", dest="include_pipe_lanes", action="store_false")
    trace_cmd.add_argument("--no-counters", dest="include_counters", action="store_false")
    trace_cmd.add_argument("--no-overlap", dest="include_overlap", action="store_false")

    xlsx_cmd = subparsers.add_parser(
        "export-xlsx",
        help="export human-native Before-vs-After comparison & Memory/Quant/PP .xlsx workbook",
    )
    _add_common_export_args(xlsx_cmd)
    xlsx_cmd.add_argument("--compare-case", default=None, help="optional candidate ('After') case JSON")
    xlsx_cmd.add_argument("--weight-quant", default=None)
    xlsx_cmd.add_argument("--cache-dtype", default=None)
    xlsx_cmd.add_argument("--engram-offload", dest="engram_offload", action="store_true", default=None)
    xlsx_cmd.add_argument("--no-engram-offload", dest="engram_offload", action="store_false")


def _run_oneshot_export(args: argparse.Namespace) -> dict[str, Any]:
    if not args.xlsx_output and not args.trace_output:
        raise ValueError("export_command_requires_at_least_one_of_xlsx_or_trace")
    dep_mode = getattr(args, "deployment_mode", None)
    if dep_mode == "pd_decode" and getattr(args, "ttft_ms", None) is not None:
        raise ModelingError(
            "pd_decode_ttft_unavailable: --ttft-ms is unavailable in pd_decode mode (must be omitted/None)"
        )
    policy = resolve_auto_va_policy(
        args.model,
        args.hardware,
        context_tokens=int(args.prompt_tokens),
        weight_quant=getattr(args, "weight_quant", "auto_va"),
        cache_dtype=getattr(args, "cache_dtype", None),
    )
    eff_quant = policy["compose_quant"]
    eff_cache = policy["kv_cache_dtype"]
    eff_idx = policy["indexer_kv_dtype"]
    eff_ttft = None if (dep_mode == "pd_decode" or args.ttft_ms is None) else float(args.ttft_ms)
    replay = ReplayOptions(include_submission=bool(getattr(args, "include_submission", False)))
    common_kw = dict(
        topology_preset=args.topology_preset,
        dp=int(args.dp), tp=int(args.tp), dcp=int(args.dcp),
        ep=int(args.ep), pp=int(args.pp), pcp=int(args.pcp),
        prompt_tokens=int(args.prompt_tokens), output_tokens=int(args.output_tokens),
        concurrency=int(args.concurrency),
        tpot_slo_ms=(float(args.tpot_ms) if args.tpot_ms is not None else None),
        ttft_slo_ms=eff_ttft,
        mtp_proposal_tokens=int(args.spec_tokens),
        mtp_acceptance_length=float(args.accept_len),
        quant=eff_quant,
        kv_cache_dtype=eff_cache,
        indexer_kv_dtype=eff_idx,
        deployment_mode=dep_mode,
        runtime_stack=getattr(args, "stack", None),
    )
    raw_before, auto_req = compose_export_case(
        args.model, args.hardware, feature_track=str(args.compare_track or "shipping"), **common_kw,
    )
    step_req = StepRequest(
        phase=args.phase,
        tokens=auto_req.tokens if args.phase == "decode" else int(args.prompt_tokens),
        context=auto_req.context if args.phase == "decode" else 0,
        sequences=auto_req.sequences,
        scope="model",
    )
    # Both Cases were just composed with ``eff_quant``; the ledger reads their resolved QuantSpec,
    # so a catalog preset name (``w8a8_dynamic``) is never re-passed as a storage-format override.
    before_bundle = build_export_bundle(
        raw_before, request=step_req, replay=replay,
        observations_path=args.observations,
        cache_dtype=eff_cache,
        engram_host_offload=args.engram_offload,
    )
    after_bundle = None
    if args.target_track:
        raw_after, _ = compose_export_case(
            args.model, args.hardware, feature_track=str(args.target_track), **common_kw,
        )
        after_bundle = build_export_bundle(
            raw_after, request=step_req, replay=replay,
            cache_dtype=eff_cache,
            engram_host_offload=args.engram_offload,
        )
    result: dict[str, Any] = {
        "ok": True,
        "command": "export",
        "model_name": before_bundle.model_name,
        "hardware_sku": before_bundle.hardware_sku,
        "phase": before_bundle.phase,
    }
    if args.xlsx_output:
        xlsx_sum = export_comparison_xlsx(
            before_bundle, args.xlsx_output,
            after_bundle=after_bundle,
        )
        result["xlsx"] = xlsx_sum.to_dict()
    if args.trace_output:
        trace_bundle = after_bundle if (after_bundle is not None and args.trace_side == "after") else before_bundle
        trace_sum = export_msprof_trace(
            trace_bundle, args.trace_output,
            include_pipe_lanes=bool(getattr(args, "include_pipe_lanes", True)),
            include_counters=bool(getattr(args, "include_counters", True)),
            include_overlap=bool(getattr(args, "include_overlap", True)),
        )
        result["trace"] = trace_sum.to_dict()
    return result


def run_export_command(args: argparse.Namespace) -> dict[str, Any]:
    """Execute ``report``, ``quickstart``, ``export``, ``export-trace``, or ``export-xlsx``."""
    if args.command == "report":
        return run_report_command(args)
    if args.command == "quickstart":
        return run_quickstart_command(args)
    if args.command == "export":
        return _run_oneshot_export(args)
    case_data = _read_bounded_json_object(args.case, "case")
    if args.context is None:
        raise ModelingError("export_command_requires_explicit_context_tokens")
    request = StepRequest(
        phase=args.phase,
        tokens=int(args.tokens),
        context=int(args.context),
        sequences=int(args.sequences),
        scope=args.scope,
    )
    replay = ReplayOptions(include_submission=bool(getattr(args, "include_submission", False)))
    if args.command == "export-trace":
        summary = export_msprof_trace(
            case_data,
            args.export_output,
            request=request,
            replay=replay,
            observations_path=args.observations,
            include_pipe_lanes=bool(getattr(args, "include_pipe_lanes", True)),
            include_counters=bool(getattr(args, "include_counters", True)),
            include_overlap=bool(getattr(args, "include_overlap", True)),
        )
        return {"ok": True, "command": "export-trace", **summary.to_dict()}
    if args.command == "export-xlsx":
        compare_data = (
            _read_bounded_json_object(args.compare_case, "compare_case")
            if getattr(args, "compare_case", None) else None
        )
        summary = export_comparison_xlsx(
            case_data,
            args.export_output,
            after_bundle=compare_data,
            request=request,
            replay=replay,
            observations_path=args.observations,
            weight_quant=getattr(args, "weight_quant", None),
            cache_dtype=getattr(args, "cache_dtype", None),
            engram_host_offload=getattr(args, "engram_offload", None),
        )
        return {"ok": True, "command": "export-xlsx", **summary.to_dict()}
    raise ValueError(f"not_an_export_command:{args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone CLI entrypoint (``python -m mindie_sim.cli.export``)."""
    root = argparse.ArgumentParser(prog="mindie-sim-export")
    sub = root.add_subparsers(dest="command", required=True)
    add_export_commands(sub)
    args = root.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        result = run_export_command(args)
        md_text = result.pop("_stdout_markdown", None) if isinstance(result, dict) else None
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            Path(args.output).write_bytes(payload.encode("utf-8"))
        if md_text:
            print(md_text, end="")
        elif not getattr(args, "output", None):
            print(payload, end="")
    except TimeoutError:
        raise
    except (ModelingError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return CLI_ERROR_EXIT
    return 0


__all__ = [
    "EXPORT_BOOLEAN_FLAGS",
    "EXPORT_COMMANDS",
    "add_export_commands",
    "main",
    "run_export_command",
]

if __name__ == "__main__":
    raise SystemExit(main())
