#!/usr/bin/env python3
"""Thin CLI entrypoint for the 4-stage New-Operator Onboarding Pipeline.

[Role] Expose ``mindie_sim.cli_onboarding.operator_main`` directly as an
  executable tool (``python3 tools/onboard_operator.py <scaffold|
  validate-contract|generate-cliff-plan|bench-and-score|certify|run-all> ...``)
  with 100% flag parity to ``mindie-sim onboard-operator``.
[Invariants] Zero duplicated business logic; returns exit code 0 on pass and
  exit code 2 on failed contract/certification gates; ``TimeoutError``
  propagates cleanly.
[Timeouts & Locks] Delegates bounded execution to ``mindie_sim.onboarding``.
[Side-Effect Policy] Writes only explicitly requested output paths.
"""
from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "mindie_sim").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mindie_sim.cli_onboarding import operator_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(operator_main())
