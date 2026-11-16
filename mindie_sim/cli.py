"""Stable CLI shim: core command surface plus public SDK diagnostics.

[Role]
  Publish ``main``, ``parser`` and ``run`` from ``mindie_sim.__main__`` and
  route the additive ``extensions``/``profile`` diagnostics to the public
  facade.  Downstream harnesses keep importing one stable command module.

[Invariants]
  * Core commands still execute the exact live objects from ``__main__``; no
    duplicate parser, prediction or dispatch logic exists here.
  * Public commands only load/validate explicit extension or profile sources.
  * Importing this shim performs no I/O, hardware access or writes.

[Timeouts & Locks]
  Pure import-time bindings; no locks or device contexts.

[Side-Effect Policy]
  No side effects beyond module loading; explicit load calls read selected files.
"""
from __future__ import annotations

import sys

from .__main__ import main as _core_main, parser, run
from .public.cli import PUBLIC_COMMANDS, public_main

__all__ = ["main", "parser", "run", "PUBLIC_COMMANDS"]


_CMD_IDX = 0
_ARGV_START = 1


def main(argv=None):
    args = list(sys.argv[_ARGV_START:] if argv is None else argv)
    if args and args[_CMD_IDX] in PUBLIC_COMMANDS:
        return public_main(args, prog="mindie-sim")
    return _core_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
