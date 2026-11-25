"""Single authoritative CLI entrypoint routing console script and ``-m mindie_sim``.

[Role] Resolve canonical and legacy command routes through ``cli_parser``,
  dispatch through ``cli_commands`` / ``api`` / ``PredictionSpine``, and emit
  JSON output with consistent error handling and exit codes.
[Invariants] Both ``mindie-sim`` (console script) and ``python -m mindie_sim``
  execute the exact same ``_main`` argv route. ``TimeoutError`` is never
  swallowed by ``except OSError:``.
[Timeouts & Locks] Delegates bounded execution to command handlers; no locks.
[Side-Effect Policy] Writes only stdout/stderr or explicitly requested output
  paths.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys

from .commands import CliUsageError, dispatch, emit
from .parser import build_parser, resolve_route
from mindie_sim.public.cli import PUBLIC_COMMANDS

__all__ = ["PUBLIC_COMMANDS", "_main", "main", "parser", "run"]


def parser() -> argparse.ArgumentParser:
    """Return the shared argument parser."""
    return build_parser(prog="mindie-sim")


def run(args: argparse.Namespace) -> object:
    """Dispatch a parsed ``argparse.Namespace`` and return the report object."""
    return dispatch(args)


def _delegate(command: str, rest: tuple[str, ...], *, prog: str) -> int | None:
    if command == "extensions":
        from mindie_sim.public.cli import extensions_main
        return extensions_main(rest, prog=prog + " extensions")
    if command == "profile":
        from mindie_sim.public.cli import profile_main
        return profile_main(rest, prog=prog + " profile")
    if command == "structure":
        from mindie_sim.structure import main as structure_main
        return structure_main(list(rest))
    if command == "compare-cohorts":
        from mindie_sim.cohort_accuracy import main as cohorts_main
        return cohorts_main(list(rest))
    if command == "scoreboard":
        from mindie_sim.observations.scoreboard import main as scoreboard_main
        return scoreboard_main(list(rest), prog=f"{prog} obs scoreboard")
    if command == "subgraph-parity":
        from mindie_sim.observations.subgraph_parity import main as subgraph_parity_main
        return subgraph_parity_main(list(rest), prog=f"{prog} obs subgraph-parity")
    return None


def _print_markdown(text: str) -> None:
    try:
        print(text, end="")
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"), end="")


def _main(argv: Sequence[str], /, *, prog: str = "mindie-sim") -> int:
    raw = list(argv)
    if not raw or raw[0] in {"-h", "--help"}:
        build_parser(prog=prog).parse_args(raw)
        return 0
    route, rest = resolve_route(raw)
    delegated = _delegate(route.command, rest, prog=prog)
    if delegated is not None:
        return delegated
    args = build_parser(prog=prog).parse_args([route.command, *rest])
    try:
        result = run(args)
        md_text = result.pop("_stdout_markdown", None) if isinstance(result, dict) else None
        out_path = getattr(args, "output", None)
        if md_text is not None:
            if out_path:
                emit(result, out_path)
            _print_markdown(str(md_text))
        else:
            emit(result, out_path)
        if (
            args.command in {"study", "compare-calls"}
            and isinstance(result, dict)
            and result.get("status") != "completed"
        ):
            return 2
        if (
            args.command in {"build-accel", "onboard-sku", "onboard-operator"}
            and isinstance(result, dict)
            and result.get("ok") is False
        ):
            return 2
    except TimeoutError:
        raise
    except (CliUsageError, TypeError, ValueError, OverflowError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI process entrypoint for ``mindie-sim`` and ``python -m mindie_sim``."""
    raw = list(sys.argv[1:] if argv is None else argv)
    return _main(raw, prog="mindie-sim")


if __name__ == "__main__":
    raise SystemExit(main())
