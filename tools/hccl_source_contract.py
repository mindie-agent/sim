#!/usr/bin/env python3
"""Print one explicit HCCL source-contract plan for review or integration tests.

[Role] Ordinary CLI recipe over `hccl_contracts.select` and the existing
  `hccl_aicpu.contract`; no new registry, engine or selector.
[Invariants] Template, buffers, dtype and all native segments/channel weights
  are explicit. Missing or fractional declarations fail visibly.
[Timeouts & Locks] Finite metadata only; no locks or external work.
[Side-Effect Policy] Reads arguments, writes JSON to stdout; no device or files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindie_sim.prediction.hccl_aicpu import contract  # noqa: E402
from mindie_sim.prediction.hccl_contracts import select  # noqa: E402
from mindie_sim.schema.common import ModelingError  # noqa: E402


def _ints(value):
    if value is None or value == "":
        return ()
    return tuple(int(item) for item in value.split(",") if item != "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collective", required=True,
                        choices=("allreduce", "allgather", "reduce_scatter"))
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--payload-bytes", required=True, type=int)
    parser.add_argument("--template", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--buffer-mode", default="separate",
                        choices=("separate", "symmetric", "remote_output"))
    parser.add_argument("--segments", default="",
                        help="comma-separated ordered native segment bytes")
    parser.add_argument("--level1-weights", default="",
                        help="comma-separated explicit channel byte-share weights")
    args = parser.parse_args(argv)

    raw = {
        "expansion": "AICPU_TS", "template": args.template,
        "dtype_bytes": args.dtype_bytes, "buffer_mode": args.buffer_mode,
    }
    try:
        segments = _ints(args.segments)
        weights = _ints(args.level1_weights)
    except ValueError:
        parser.exit(2, "error: segments and level1 weights must be exact positive integers\n")
    if segments:
        raw["segments"] = segments
    if weights:
        raw["level1_weights"] = weights
    try:
        selection = select(raw, collective=args.collective, ranks=args.ranks,
                           payload_bytes=args.payload_bytes)
        plan = contract(selection, args.rank)
    except ModelingError as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({
        "selection": selection.to_dict(),
        "rank": plan.rank,
        "thread_count": plan.thread_count,
        "scratch_bytes": plan.scratch_bytes,
        "control_counts": dict(plan.control_counts),
        "unmodeled": list(plan.unmodeled),
        "phases": [{
            "name": phase.name,
            "after": list(phase.after),
            "work": dict(phase.work),
            "steps": [{"name": name, "work": dict(work)}
                      for name, work in phase.steps],
            "exchange": [list(row) for row in phase.exchange],
            # Match the public lower: no exchange and no signal means local.
            "native_direction": (phase.native_direction
                                 if phase.exchange or phase.signals else "local"),
            "control_counts": dict(phase.control_counts),
        } for phase in plan.phases],
    }, indent=2))


if __name__ == "__main__":
    main()
