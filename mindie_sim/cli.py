"""Stable CLI shim: core command surface plus public SDK diagnostics.

[Role]
  Publish ``main``, ``parser``, and ``run`` from ``mindie_sim.__main__`` and
  route the additive ``extensions``/``profile`` diagnostics to the public
  facade.  Downstream harnesses keep importing one stable command module.

[Invariants]
  * Core commands still execute the exact live objects from ``__main__``; no
    duplicate parser, prediction or dispatch logic exists here.
  * Public commands only load/validate explicit extension or profile sources.
  * Importing this shim performs no I/O, hardware access, or writes.

[Timeouts & Locks]
  Pure import-time bindings; no locks or device contexts.

[Side-Effect Policy]
  No side effects beyond module loading; explicit load calls read selected files.
"""
from __future__ import annotations

import sys

from .__main__ import main as _core_main, parser, run
from .extended_cli import EXTENDED_COMMANDS, extended_main
from .public.cli import PUBLIC_COMMANDS, public_main
from .feature_cli import COMMANDS as FEATURE_COMMANDS, main as feature_main

__all__ = ["main", "parser", "run", "PUBLIC_COMMANDS", "FEATURE_COMMANDS", "EXTENDED_COMMANDS"]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in FEATURE_COMMANDS:
        return feature_main(args, prog="mindie-sim")
    if args and args[0] in EXTENDED_COMMANDS:
        return extended_main(args, prog="mindie-sim")
    if args and args[0] in PUBLIC_COMMANDS:
        return public_main(args, prog="mindie-sim")
    return _core_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
