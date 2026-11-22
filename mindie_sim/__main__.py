"""Module entrypoint delegating verbatim to ``mindie_sim.cli``.

[Role] Ensure ``python -I -X utf8 -m mindie_sim`` executes the exact same
  argument route, emitter and exit codes as the ``mindie-sim`` console script.
[Invariants] Zero duplicate parser, dispatch or I/O code resides here; re-exports
  the live callables from ``mindie_sim.cli`` and ``mindie_sim.cli.commands``.
[Timeouts & Locks] Pure module re-export; no I/O or locks on import.
[Side-Effect Policy] Zero side effects on import; executes CLI only when run
  as ``__main__``.
"""
from __future__ import annotations

from .cli import main, parser, run
from .cli.commands import (
    CliUsageError,
    MAX_CLI_JSON_BYTES,
    emit,
    load_cli_json,
    read_json,
)

__all__ = [
    "CliUsageError",
    "MAX_CLI_JSON_BYTES",
    "emit",
    "load_cli_json",
    "main",
    "parser",
    "read_json",
    "run",
]

if __name__ == "__main__":
    raise SystemExit(main())
