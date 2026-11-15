"""CLI adapters for launch inspection, SOL comparison and feature deltas.

[Role] Read explicit launch/input files, call the same public APIs, and write
  one canonical JSON object or a Markdown projection.  No second parser,
  physics path or case validator lives here.
[Invariants] Exit 0 is a complete report, 2 an invalid/strict-rejected input,
  3 a partial/needs-input result and 4 an execution failure.  stdout is one
  object; warnings/diagnostics go to stderr.  Missing inputs remain missing;
  the command never reads host environment variables as deployment defaults.
  Output writes are explicit local-file operations only.
[Timeouts & Locks] Delegates all bounded work to the APIs; no locks.
[Side-Effect Policy] Reads only named files/stdin and writes the named output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .feature_analysis import FEATURES, LaunchComparisonError, compare_launch
from .feature_reports import (
    ReportValidationError, canonical_json, feature_markdown, launch_markdown,
    sol_markdown, to_dict,
)
from .launch_bindings import parse_launch
from .launch_documents import DocumentInputError, finite_json_loads
from .launch_parser import LaunchSyntaxError
from .schema.operators import OpSpec

COMMANDS = ("inspect-launch", "sol", "feature-deltas")


def _read_text(path):
    limit = 8 * 1024 * 1024
    if path == "-":
        return sys.stdin.read(limit + 1)
    source = Path(path)
    if source.stat().st_size > limit:
        raise ValueError("input_exceeds_8_mib")
    return source.read_text(encoding="utf-8")


def _load_json(path, default=None):
    if path is None:
        return default
    return finite_json_loads(_read_text(path), str(path))


def _load_machine(value):
    if value is None:
        return None
    path = Path(value)
    if path.exists() and path.is_file():
        return _load_json(value)
    return value


def _load_model(value):
    if value is None:
        return None
    path = Path(value)
    if path.exists() and path.is_file():
        return _load_json(value)
    return value


def _ops_from_payload(payload):
    if isinstance(payload, dict):
        payload = payload.get("ops", payload.get("operators", ()))
    if not isinstance(payload, list):
        raise ValueError("ops_json_requires_list")
    ops = []
    defaults = {"family": "matmul", "phase": "decode", "stage": 0, "layer": 0,
                "requests": ("r0",), "semantic_shape": (1, 1), "local_shape": (1, 1),
                "padded_shape": (1, 1), "launch_shape": (1, 1)}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("ops_json_member_requires_mapping")
        data = {**defaults, **row}
        data["requests"] = tuple(data.get("requests") or ("r0",))
        for key in ("semantic_shape", "local_shape", "padded_shape", "launch_shape"):
            data[key] = tuple(data.get(key) or (1, 1))
        data["deps"] = tuple(data.get("deps") or ())
        data["metadata"] = tuple(sorted(dict(data.get("metadata") or {}).items()))
        data.setdefault("dtype", "bf16")
        ops.append(OpSpec(**{key: value for key, value in data.items()
                             if key in OpSpec.__dataclass_fields__}))
    return tuple(ops)


def _load_env_files(paths):
    result = {}
    for path in paths or ():
        content = {}
        for line in _read_text(path).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            content[name.strip()] = value
        result[str(path)] = content
    return result


def _base_inputs(args):
    return {
        "framework_version": args.framework_version,
        "machine": _load_machine(args.machine),
        "model": _load_model(args.model),
        "workload": _load_json(args.workload),
        "speculative": _load_json(args.speculative_acceptance),
        "base_case": _load_json(args.case),
        "strict": bool(args.strict),
        "container": args.container,
        "env_files": _load_env_files(args.env_file),
    }


def _launch(args):
    text = _read_text(args.launch)
    if args.input_format == "argv":
        text = finite_json_loads(text, args.launch)
    return parse_launch(text, format=args.input_format, source=args.launch,
                        **_base_inputs(args))


def _exit(status):
    return {"completed": 0, "partial": 3, "needs_inputs": 3, "failed": 2}.get(status, 2)


def _emit(text, output):
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _parser(prog="mindie-sim"):
    root = argparse.ArgumentParser(prog=prog)
    sub = root.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        cmd = sub.add_parser(name)
        cmd.add_argument("launch", help="shell/argv/JSON/YAML launch file, or -")
        cmd.add_argument("--input-format", choices=("shell", "argv", "json", "yaml"), default="shell")
        cmd.add_argument("--framework-version")
        cmd.add_argument("--machine", help="machine name or JSON file")
        cmd.add_argument("--model", "--model-config", dest="model", help="model name or JSON file")
        cmd.add_argument("--workload", help="workload JSON file")
        cmd.add_argument("--speculative-acceptance", help="acceptance JSON file")
        cmd.add_argument("--case", help="base Case JSON file")
        cmd.add_argument("--container", help="select one container from a structured document")
        cmd.add_argument("--env-file", action="append", default=[],
                         help="explicit container env-file contents (repeatable)")
        cmd.add_argument("--strict", action="store_true")
        cmd.add_argument("--output", "-o")
        cmd.add_argument("--format", choices=("json", "markdown"), default="json")
        if name != "inspect-launch":
            cmd.add_argument("--phase", choices=("prefill", "decode"), default="decode")
            cmd.add_argument("--scope", choices=("body", "model"), default="model")
            cmd.add_argument("--tokens", type=int, default=1)
            cmd.add_argument("--context", type=int, default=4096)
            cmd.add_argument("--sequences", type=int, default=16)
            cmd.add_argument("--dispatch", choices=("ideal_zero", "declared_physical"), default="ideal_zero")
            cmd.add_argument("--request-metrics", action="store_true")
            cmd.add_argument("--allow-partial", action="store_true")
            cmd.add_argument("--ops-json", help="explicit selected-plan OpSpec list")
            cmd.add_argument("--prefill-ops-json")
            cmd.add_argument("--request-trace", help="explicit arrival/decode trace JSON")
            cmd.add_argument("--commits", type=float)
            cmd.add_argument("--pool-budget", type=int, default=16)
        if name == "feature-deltas":
            cmd.add_argument("--features", default=",".join(FEATURES))
            cmd.add_argument("--max-candidates", type=int, default=16)
    return root


def _compare(args, report):
    ops = _ops_from_payload(_load_json(args.ops_json)) if args.ops_json else None
    prefill = _ops_from_payload(_load_json(args.prefill_ops_json)) if args.prefill_ops_json else None
    features = tuple(item.strip() for item in args.features.split(",") if item.strip()) \
        if args.command == "feature-deltas" else ()
    return compare_launch(
        report, phase=args.phase, scope=args.scope, tokens=args.tokens, context=args.context,
        sequences=args.sequences, dispatch=args.dispatch, request_metrics=args.request_metrics,
        counterfactuals=features, max_candidates=int(getattr(args, "max_candidates", 16)),
        allow_partial=args.allow_partial, plan_ops=ops, prefill_ops=prefill,
        request_trace=_load_json(getattr(args, "request_trace", None)),
        commits=getattr(args, "commits", None))


def run(args):
    report = _launch(args)
    if args.command == "inspect-launch":
        return report, "launch", _exit(report.status)
    if report.status == "failed":
        return report, "launch", 2
    if report.status == "needs_inputs" or (report.status == "partial" and not args.allow_partial):
        return report, "launch", 3
    comparison = _compare(args, report)
    return comparison, "sol", _exit(comparison.status)


def main(argv=None, prog="mindie-sim"):
    args = _parser(prog).parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        report, kind, code = run(args)
        text = (launch_markdown(report) if kind == "launch" else
                feature_markdown(report) if args.command == "feature-deltas" else
                sol_markdown(report)) if args.format == "markdown" else canonical_json(report)
        _emit(text, args.output)
        return code
    except TimeoutError:
        raise
    except (LaunchComparisonError, ReportValidationError, LaunchSyntaxError,
            DocumentInputError, ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)[:300]}), file=sys.stderr)
        return 2
    except Exception as exc:  # explicit execution failure boundary
        print(json.dumps({"status": "failed", "error": str(exc)[:300]}), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
