#!/usr/bin/env python3
"""Thin CLI delegate to ``mindie_sim.observations.subgraph_parity``.

[Role] Preserve direct script invocation (``python tools/live_subgraph_parity_gate.py``)
  and re-exported parity helpers while delegating all parity orchestration, SKU
  selection, and exit codes to the packaged ``mindie_sim.observations.subgraph_parity``
  module (also exposed via ``mindie-sim obs subgraph-parity``).
[Invariants] Zero duplicated parity logic; zero NPU device context.
[Timeouts & Locks] Delegates directly to ``mindie_sim.observations.subgraph_parity``.
[Side-Effect Policy] Read-only against case and capture inputs; delegates output handling.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mindie_sim.observations.subgraph_parity import (  # noqa: E402
    OfflineCaptureCaseConfig,
    _resolve_op_kernels,
    build_offline_expected_nodes,
    build_predicted_nodes_from_case,
    build_predicted_nodes_from_prepared_step,
    evaluate_live_dir,
    evaluate_offline_bundle,
    main,
)

__all__ = [
    "OfflineCaptureCaseConfig",
    "_resolve_op_kernels",
    "build_offline_expected_nodes",
    "build_predicted_nodes_from_case",
    "build_predicted_nodes_from_prepared_step",
    "evaluate_live_dir",
    "evaluate_offline_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
