"""Thin compatibility CLI wrapper delegating to ``mindie_sim.cli.matrix``."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mindie_sim.cli.matrix import export_subgraph_bench_plan_cli


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate directly to packaged ``mindie_sim.cli.matrix.export_subgraph_bench_plan_cli``."""
    return export_subgraph_bench_plan_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
