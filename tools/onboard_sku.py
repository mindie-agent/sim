#!/usr/bin/env python3
"""Thin CLI entrypoint for the 4-stage New-SKU Onboarding Pipeline.

[Role] Expose ``mindie_sim.cli_onboarding.sku_main`` directly as an executable
  tool (``python3 tools/onboard_sku.py <probe-soc|calibrate-l0-and-fabric|
  validate-512-400k|verify-gates|run-all> ...``) with 100% flag parity to
  ``mindie-sim onboard-sku``.
[Invariants] Zero duplicated business logic; returns exit code 0 on pass and
  exit code 2 on failed gates/conflicts; ``TimeoutError`` propagates cleanly.
[Timeouts & Locks] Delegates bounded execution to ``mindie_sim.onboarding``.
[Side-Effect Policy] Writes only explicitly requested output paths.
"""
from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "mindie_sim").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mindie_sim.cli_onboarding import sku_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(sku_main())
