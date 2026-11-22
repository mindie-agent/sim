"""CLI subcommands and standalone entrypoint for ``export``, ``export-trace``, and ``export-xlsx``.

[Role] Register and dispatch:
  (1) One-shot zero-boilerplate ``mindie-sim export --model ... --hardware ...
      --xlsx ... --trace ...`` (building Before/After ``ExportBundle`` once and
      generating both human-native ``.xlsx`` and streaming ``trace_view.json``),
  (2) ``export-trace`` (1:1 Ascend ``msprof`` ``trace_view.json`` from a case JSON),
  (3) ``export-xlsx`` (OOXML ``merge_cells`` comparison & Memory/Quant/PP workbook
      from case JSON(s), with optional ``--kernel-designs-json`` Agent design overlay).
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
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

from .export import (
    DEFAULT_EXPORT_CONTEXT_TOKENS,
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    DEFAULT_TPOT_SLO_MS,
    DEFAULT_TTFT_SLO_MS,
    build_export_bundle,
    compose_export_case,
    export_comparison_xlsx,
    export_msprof_trace,
)
from .prediction.results import ReplayOptions, StepRequest
from .schema.common import ModelingError

#: Commands owned by this CLI module.
EXPORT_COMMANDS = frozenset({"export", "export-trace", "export-xlsx"})
#: Boolean flags registered by this module for route-resolution token counting.
EXPORT_BOOLEAN_FLAGS = frozenset({
    "--no-pipe-lanes",
    "--no-counters",
    "--no-overlap",
    "--include-submission",
    "--engram-offload",
    "--no-engram-offload",
})
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


def _load_kernel_designs(path: str | None) -> Mapping[str, Mapping[str, str]] | None:
    if not path:
        return None
    raw = _read_bounded_json_object(path, "kernel_designs")
    designs: dict[str, dict[str, str]] = {}
    for key, val in raw.items():
        if isinstance(val, Mapping):
            designs[str(key)] = {str(k): str(v) for k, v in val.items() if v is not None}
        elif isinstance(val, str):
            designs[str(key)] = {"design": val}
    return designs


def _add_common_export_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("case", help="path to MindIE-Sim case JSON file")
    cmd.add_argument(
        "--out", "--output", "-o", dest="export_output", required=True,
        help="destination file path for exported artifact",
    )
    cmd.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    cmd.add_argument("--tokens", type=int, default=1)
    cmd.add_argument("--context", type=int, default=DEFAULT_EXPORT_CONTEXT_TOKENS)
    cmd.add_argument("--sequences", type=int, default=1)
    cmd.add_argument("--scope", choices=("body", "model"), default="body")
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
    cmd.add_argument("--model", required=True, help="catalog model name (e.g. kimi-k3.1, deepseek-v4.1-flash)")
    cmd.add_argument("--hardware", required=True, help="hardware SKU or alias (e.g. 950dt-96g-pod, Ascend950DT_9582)")
    cmd.add_argument("--topology-preset", default=None, help="optional topology preset override (e.g. a5_pod, a3_16p)")
    cmd.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    cmd.add_argument("--dp", type=int, default=1)
    cmd.add_argument("--tp", type=int, default=1)
    cmd.add_argument("--dcp", type=int, default=1)
    cmd.add_argument("--ep", type=int, default=1)
    cmd.add_argument("--pp", type=int, default=1)
    cmd.add_argument("--pcp", type=int, default=1)
    cmd.add_argument("--prompt-tokens", type=int, default=DEFAULT_SCENARIO_PROMPT_TOKENS)
    cmd.add_argument("--output-tokens", type=int, default=DEFAULT_SCENARIO_OUTPUT_TOKENS)
    cmd.add_argument("--concurrency", type=int, default=1)
    cmd.add_argument("--tpot-ms", type=float, default=DEFAULT_TPOT_SLO_MS)
    cmd.add_argument("--ttft-ms", type=float, default=DEFAULT_TTFT_SLO_MS)
    cmd.add_argument("--spec-tokens", "--mtp", dest="spec_tokens", type=int, default=0)
    cmd.add_argument("--accept-len", type=float, default=1.0)
    cmd.add_argument("--compare-track", default="merged", help="baseline ('Before') feature track (default: merged)")
    cmd.add_argument("--target-track", default=None, help="optional candidate ('After') feature track (e.g. expected_ideal)")
    cmd.add_argument("--xlsx", dest="xlsx_output", default=None, help="destination path for comparison .xlsx workbook")
    cmd.add_argument("--trace", dest="trace_output", default=None, help="destination path for msprof trace_view.json")
    cmd.add_argument("--trace-side", choices=("before", "after"), default="after", help="which track to emit into --trace when comparing")
    cmd.add_argument("--weight-quant", default=None)
    cmd.add_argument("--cache-dtype", default=None)
    cmd.add_argument("--engram-offload", dest="engram_offload", action="store_true", default=None)
    cmd.add_argument("--no-engram-offload", dest="engram_offload", action="store_false")
    cmd.add_argument("--kernel-designs-json", default=None, help="optional JSON mapping kernel keys to Agent design/acceptance drafts")
    cmd.add_argument("--observations", "--kernel-csv", dest="observations", default=None)
    cmd.add_argument("--include-submission", action="store_true")
    cmd.add_argument("--no-pipe-lanes", dest="include_pipe_lanes", action="store_false")
    cmd.add_argument("--no-counters", dest="include_counters", action="store_false")
    cmd.add_argument("--no-overlap", dest="include_overlap", action="store_false")
    cmd.add_argument("--receipt", dest="output", default=None, help="optional JSON path to save structured export receipt")


def add_export_commands(subparsers) -> None:
    """Register ``export``, ``export-trace``, and ``export-xlsx`` on shared CLI subparsers."""
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
    xlsx_cmd.add_argument("--kernel-designs-json", default=None, help="optional JSON mapping kernel keys to Agent design/acceptance drafts")


def _run_oneshot_export(args: argparse.Namespace) -> dict[str, Any]:
    if not args.xlsx_output and not args.trace_output:
        raise ValueError("export_command_requires_at_least_one_of_xlsx_or_trace")
    replay = ReplayOptions(include_submission=bool(getattr(args, "include_submission", False)))
    common_kw = dict(
        topology_preset=args.topology_preset,
        dp=int(args.dp), tp=int(args.tp), dcp=int(args.dcp),
        ep=int(args.ep), pp=int(args.pp), pcp=int(args.pcp),
        prompt_tokens=int(args.prompt_tokens), output_tokens=int(args.output_tokens),
        concurrency=int(args.concurrency),
        tpot_slo_ms=float(args.tpot_ms), ttft_slo_ms=float(args.ttft_ms),
        mtp_proposal_tokens=int(args.spec_tokens),
        mtp_acceptance_length=float(args.accept_len),
    )
    raw_before, auto_req = compose_export_case(
        args.model, args.hardware, feature_track=str(args.compare_track or "merged"), **common_kw,
    )
    step_req = StepRequest(
        phase=args.phase,
        tokens=auto_req.tokens if args.phase == "decode" else int(args.prompt_tokens),
        context=auto_req.context if args.phase == "decode" else 0,
        sequences=auto_req.sequences,
        scope="model",
    )
    before_bundle = build_export_bundle(
        raw_before, request=step_req, replay=replay,
        observations_path=args.observations,
        weight_quant=args.weight_quant, cache_dtype=args.cache_dtype,
        engram_host_offload=args.engram_offload,
    )
    after_bundle = None
    if args.target_track:
        raw_after, _ = compose_export_case(
            args.model, args.hardware, feature_track=str(args.target_track), **common_kw,
        )
        after_bundle = build_export_bundle(
            raw_after, request=step_req, replay=replay,
            weight_quant=args.weight_quant, cache_dtype=args.cache_dtype,
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
        designs = _load_kernel_designs(getattr(args, "kernel_designs_json", None))
        xlsx_sum = export_comparison_xlsx(
            before_bundle, args.xlsx_output,
            compare_source=after_bundle,
            agent_kernel_designs=designs,
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
    """Execute ``export``, ``export-trace``, or ``export-xlsx`` and return a compact JSON receipt."""
    if args.command == "export":
        return _run_oneshot_export(args)
    case_data = _read_bounded_json_object(args.case, "case")
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
        designs = _load_kernel_designs(getattr(args, "kernel_designs_json", None))
        summary = export_comparison_xlsx(
            case_data,
            args.export_output,
            compare_source=compare_data,
            request=request,
            replay=replay,
            observations_path=args.observations,
            weight_quant=getattr(args, "weight_quant", None),
            cache_dtype=getattr(args, "cache_dtype", None),
            engram_host_offload=getattr(args, "engram_offload", None),
            agent_kernel_designs=designs,
        )
        return {"ok": True, "command": "export-xlsx", **summary.to_dict()}
    raise ValueError(f"not_an_export_command:{args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone CLI entrypoint (``python -m mindie_sim.cli_export``)."""
    root = argparse.ArgumentParser(prog="mindie-sim-export")
    sub = root.add_subparsers(dest="command", required=True)
    add_export_commands(sub)
    args = root.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        result = run_export_command(args)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            Path(args.output).write_bytes(payload.encode("utf-8"))
        else:
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
