"""CLI subcommands and standalone entrypoint for ``export-trace`` and ``export-xlsx``.

[Role] Register and dispatch the ``export-trace`` (1:1 ``msprof``
  ``trace_view.json``) and ``export-xlsx`` (3-sheet ``merge_cells`` comparison
  & Memory/Quant/PP workbook) commands on the shared ``mindie-sim`` parser and
  expose a standalone ``python -m mindie_sim.cli_export`` CLI entrypoint.
[Invariants] ``--out``/``--output``/``-o`` targets the exported trace/xlsx file
  (stored in ``dest="export_output"``) so the top-level CLI JSON emitter never
  overwrites exported artifacts; optional ``--receipt`` writes the structured
  JSON summary. ``TimeoutError`` is never swallowed by ``except OSError:``.
[Timeouts & Locks] Pure argument construction; bounded JSON case ingress
  (``<= 50 MiB``) and streaming export delegation; no locks or device contexts.
[Side-Effect Policy] Writes only the caller-requested export output path (and
  optional ``--receipt`` JSON path).
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from .export import (
    export_comparison_xlsx,
    export_msprof_trace,
)
from .prediction.results import ReplayOptions, StepRequest
from .schema.common import ModelingError

#: Commands owned by this CLI module.
EXPORT_COMMANDS = frozenset({"export-trace", "export-xlsx"})
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


def _read_bounded_case_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"case_file_not_found:{target}")
    if target.stat().st_size > MAX_EXPORT_CASE_JSON_BYTES:
        raise ModelingError("case_json_exceeds_50_mib_limit")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelingError("case_json_must_be_object")
    return data


def _add_common_export_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("case", help="path to MindIE-Sim case JSON file")
    cmd.add_argument(
        "--out", "--output", "-o", dest="export_output", required=True,
        help="destination file path for exported artifact",
    )
    cmd.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    cmd.add_argument("--tokens", type=int, default=1)
    cmd.add_argument("--context", type=int, default=1024)
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


def add_export_commands(subparsers) -> None:
    """Register ``export-trace`` and ``export-xlsx`` on the shared CLI subparsers."""
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
        help="export 3-sheet merge_cells comparison & Memory/Quant/PP .xlsx workbook",
    )
    _add_common_export_args(xlsx_cmd)
    xlsx_cmd.add_argument(
        "--compare-case", default=None,
        help="optional candidate ('After') case JSON for Before/After Summary delta table",
    )
    xlsx_cmd.add_argument(
        "--weight-quant", default=None,
        help="optional weight quantization preset override (e.g. BF16, W8A8, W4A8, MXFP8, MXFP4, w4a8_mxfp)",
    )
    xlsx_cmd.add_argument(
        "--cache-dtype", default=None,
        help="optional KV/Indexer/KeyPool cache dtype override (e.g. BF16, C8, FP8, fp8_e4m3, int8)",
    )
    xlsx_cmd.add_argument("--engram-offload", dest="engram_offload", action="store_true", default=None)
    xlsx_cmd.add_argument("--no-engram-offload", dest="engram_offload", action="store_false")


def run_export_command(args: argparse.Namespace) -> dict[str, Any]:
    """Execute ``export-trace`` or ``export-xlsx`` and return a compact JSON receipt."""
    case_data = _read_bounded_case_json(args.case)
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
            _read_bounded_case_json(args.compare_case)
            if getattr(args, "compare_case", None) else None
        )
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
