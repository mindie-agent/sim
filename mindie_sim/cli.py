"""Stable CLI shim re-exporting the MindIE simulation command surface.

[Role]
  Publish ``main``, ``parser``, and ``run`` from ``mindie_sim.__main__`` so
  downstream harnesses can import one stable command module.

[Invariants]
  * Re-exports the exact live objects; no duplicate parser or dispatch logic.
  * Importing this shim performs no I/O, hardware access, or writes.

[Timeouts & Locks]
  Pure import-time bindings; no locks or device contexts.

[Side-Effect Policy]
  No side effects beyond module loading.
"""
from .__main__ import main, parser, run

__all__ = ["main", "parser", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
