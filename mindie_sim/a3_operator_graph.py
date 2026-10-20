"""Isolated A3 native-operator graph timing worker (CLI and dry-run).

[Role]
  Provide the versioned manifest-driven entry point for the deliberately
  separate operator-graph lane.  ``--dry-run`` validates schema, physical
  geometry and the complete 54-condition/full-matrix or capability plan with
  the standard library only.  A real run imports the device runtime lazily and
  executes the coordinator-visible ``npu:0`` under explicit budgets.

[Invariants]
  * Dry-run never imports torch/torch_npu and never writes a results file.
  * A real run requires a non-empty run id and exactly logical device npu:0;
    it never acquires a lease, selects another card, SSHes or changes a queue.
  * Existing results evidence is not overwritten unless the caller explicitly
    passes ``--overwrite``.  Partial results are the runtime's responsibility.
  * Exit codes: 0 all requested conditions measured and measurement-complete
    with successful cleanup and persistence; 1 partial/incomplete/failed,
    cleanup-failed or persistence-failed evidence; 2 invalid input or an unsafe
    output path; 130 interrupt.
  * No existing CLI, operator worker, campaign, catalog or core module is
    modified or re-exported here.

[Timeouts & Locks]
  Parsing/validation is immediate.  Device setup/work/case/cleanup deadlines are
  enforced by the lazy runtime; the outer supervisor must allow at least 1380 s.
  No lock is taken by this CLI.

[Side-Effect Policy]
  Reads only the explicit manifest.  Writes only the explicit results path and,
  when requested, the explicit debug directory.  No network, install, commit or
  device action occurs in dry-run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

from .a3_operator_graph_plan import (
    ManifestError,
    build_plan,
    dry_run_payload,
    load_manifest_file,
)
from .a3_operator_graph_results import OutputPathError

_DEVICE_RE = re.compile(r"^npu:(\d+)$")


def _valid_run_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("--run-id must be a non-empty string")
    if len(value) > 256:
        raise ValueError("--run-id must be at most 256 characters")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError("--run-id must not contain control line separators")
    return value


def _valid_device(value: str) -> str:
    match = _DEVICE_RE.match(value)
    if not match:
        raise ValueError("--device must have the form npu:<index>")
    if int(match.group(1)) != 0:
        raise ValueError(
            "this isolated lane is authorized only for coordinator-visible npu:0")
    return "npu:0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_sim.a3_operator_graph",
        description=(
            "Isolated A3 native-operator eager/graph timing lane. Dry-run uses "
            "stdlib only; real runs require the external coordinator lease."))
    parser.add_argument("--manifest", required=True,
                        help="versioned graph-lane manifest JSON")
    parser.add_argument("--results", required=True,
                        help="explicit results JSON path (no default)")
    parser.add_argument("--run-id", required=True,
                        help="external coordinator run identity")
    parser.add_argument("--device", default="npu:0",
                        help="coordinator-visible device (this lane: npu:0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate schema/geometry/counts without importing torch")
    parser.add_argument("--debug-dir", default=None,
                        help="optional explicit directory for graph debug_dump JSON")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing results file")
    return parser


def _error_payload(kind: str, message: str, *, errors: Sequence[str] = ()) -> dict:
    return {
        "error": kind,
        "message": message,
        "errors": list(errors),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run_id = _valid_run_id(args.run_id)
        device = _valid_device(args.device)
    except ValueError as exc:
        print(json.dumps(_error_payload("invalid_run_arguments", str(exc))),
              file=sys.stderr)
        return 2

    if args.dry_run:
        try:
            payload = dry_run_payload(args.manifest)
        except ManifestError as exc:
            print(json.dumps(exc.as_dict()), file=sys.stderr)
            return 2
        payload["run_id"] = run_id
        payload["device"] = device
        payload["results_path"] = str(Path(args.results))
        payload["writes_results"] = False
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        return 0

    try:
        document, manifest_sha256, manifest_path = load_manifest_file(
            args.manifest)
        plan = build_plan(document)
    except ManifestError as exc:
        print(json.dumps(exc.as_dict()), file=sys.stderr)
        return 2
    results_path = Path(args.results)
    if results_path.exists() and not args.overwrite:
        print(json.dumps(_error_payload(
            "output_path_error",
            f"refusing to overwrite existing results evidence: {results_path}")),
            file=sys.stderr)
        return 2
    if results_path.exists() and results_path.is_dir():
        print(json.dumps(_error_payload(
            "output_path_error", f"results path is a directory: {results_path}")),
            file=sys.stderr)
        return 2
    try:
        from .a3_operator_graph_runtime import run_manifest
    except Exception as exc:
        print(json.dumps(_error_payload(
            "runtime_import_error",
            f"cannot import device runtime: {exc}")), file=sys.stderr)
        return 1
    try:
        return run_manifest(
            plan,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            device=device,
            results_path=results_path,
            debug_dir=args.debug_dir,
            allow_overwrite=bool(args.overwrite),
        )
    except OutputPathError as exc:
        print(json.dumps(_error_payload("output_path_error", str(exc))),
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps(_error_payload(
            "unexpected_worker_error",
            f"{type(exc).__name__}: {exc}")), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
