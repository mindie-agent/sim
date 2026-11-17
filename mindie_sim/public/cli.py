"""CLI diagnostics for extension discovery and local user profiles.

[Role]
  Expose one-shot load/inspect commands shared by the sim and profiler
  frontends.  These are diagnostics; ordinary Python callers do not need a
  separate discover/inspect/activate sequence.

[Invariants]
  * Output is the same context description consumed by Python callers.
  * A missing/invalid source or unknown extension ID is a non-zero result.
  * Loading does not start simulation, read device state or write the profile.

[Timeouts & Locks]
  Delegates the bounded local reads to the loader; no lock or network.

[Side-Effect Policy]
  Reads explicit sources; writes only the requested output file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .contracts import load_extensions
from .environment import ExecutionEnvironment
from .errors import ExtensionError
from .profiles import load_user_profile

PUBLIC_COMMANDS = frozenset({"extensions", "extension", "ext", "profile"})


def _emit(payload: dict[str, Any], output: str | None) -> int:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output:
        path = Path(output).expanduser()
        path.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({"output": str(path), "status": "completed"}))
    else:
        print(rendered)
    return 0


def _load_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", action="append", default=[],
                        help="workspace directory or explicit mindie_ext.py")
    parser.add_argument("--package", action="append", default=[],
                        help="installed distribution with mindie_sim.extensions entry points")
    parser.add_argument("--entry-point", action="append", default=[],
                        help="explicit module:factory entry point")
    parser.add_argument("--profile", action="append", default=[],
                        help="local mindie-sim-user-profile/1 JSON")
    parser.add_argument("--discover-installed", action="store_true",
                        help="load every installed mindie_sim.extensions entry point")
    parser.add_argument("--output", "-o", help="write JSON here; otherwise print it")


def _extensions_parser(prog: str) -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog=prog, description="Load once, then list or inspect the fixed context.",
        allow_abbrev=False)
    sub = root.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list", allow_abbrev=False)
    _load_arguments(listing)
    inspect = sub.add_parser("inspect", allow_abbrev=False)
    inspect.add_argument("id")
    _load_arguments(inspect)
    return root


def extensions_main(argv: Sequence[str] | None = None, *,
                    prog: str = "mindie-sim extensions") -> int:
    args = _extensions_parser(prog).parse_args(argv)
    try:
        context = load_extensions(
            paths=args.path, packages=args.package,
            entry_points=args.entry_point, profiles=args.profile,
            discover_installed=args.discover_installed,
        )
        payload = context.describe()
        if args.action == "inspect":
            payload = context.describe(extension_id=args.id)
            payload["inspection"] = {"id": args.id}
        return _emit(payload, args.output)
    except (ExtensionError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2


def _profile_parser(prog: str) -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog=prog, description="Validate a local user calibration profile.",
        allow_abbrev=False)
    sub = root.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", allow_abbrev=False)
    inspect.add_argument("file")
    inspect.add_argument("--id", dest="profile_id")
    inspect.add_argument("--output", "-o", help="write JSON here; otherwise print it")
    return root


def profile_main(argv: Sequence[str] | None = None, *,
                 prog: str = "mindie-sim profile") -> int:
    args = _profile_parser(prog).parse_args(argv)
    try:
        profile = load_user_profile(args.file, profile_id=args.profile_id)
        return _emit(profile.describe(), args.output)
    except (ExtensionError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2


def public_main(argv: Sequence[str] | None = None, *,
                prog: str = "mindie-sim") -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in PUBLIC_COMMANDS:
        print(json.dumps({
            "status": "failed",
            "error": "public_command_requires:" + ",".join(sorted(PUBLIC_COMMANDS)),
        }), file=sys.stderr)
        return 2
    command, rest = args[0], args[1:]
    if command == "profile":
        return profile_main(rest, prog=prog + " profile")
    return extensions_main(rest, prog=prog + " extensions")
