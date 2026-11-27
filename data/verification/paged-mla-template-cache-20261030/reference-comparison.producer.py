"""Compare cached immutable MLA tiles against the pre-cache implementation.

[Role] Verify complete result/trace identity and record CPU construction cost.
[Invariants] Same input work, physical owners, event schedule and dependencies.
[Timeouts & Locks] Local bounded geometries and one local git read; no locks.
[Side-Effect Policy] Writes one comparison receipt and restores the live plan.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import replace
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.operators.paged_mla import paged_mla

REFERENCE_REVISION = "f11514e5"
REFERENCE_SOURCE = "mindie_sim/prediction/paged_mla.py"
SHAPES = ((1, 0), (1, 128), (1, 257), (2, 383), (3, 519), (5, 1024), (12, 126000))
TRACE_CONTEXT_LIMIT = 2000
TRACE_EVENT_LIMIT = 20000


def reference_plan():
    source = subprocess.check_output(
        ["git", "show", f"{REFERENCE_REVISION}:{REFERENCE_SOURCE}"], cwd=ROOT, text=True,
    )
    module = ModuleType(f"{paged_mla.__package__}._paged_mla_reference")
    # The historical module shared its package with the runtime interfaces.
    source = source.replace("from .mechanism import", "from mindie_sim.prediction.mechanism import")
    source = source.replace("from .program import", "from mindie_sim.prediction.program import")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(source, REFERENCE_SOURCE, "exec"), module.__dict__)
    return module._plan


def main():
    old_plan = reference_plan()
    new_plan = paged_mla._plan
    case = ROOT / "examples/cases/kimi-a5-paged-host-20261030/c12.json"
    raw = json.loads(case.read_text())
    rows = []
    for batch, context in SHAPES:
        graph = target_graphs(
            raw, scope="body", phase="decode", tokens=1, sequences=batch, context=context,
        )[0]
        op = next(o for o in graph if o.recipe == paged_mla.IMPLEMENTATION)
        pair = []
        for label, plan in (("old", old_plan), ("new", new_plan)):
            started = time.monotonic()
            with patch.object(paged_mla, "_plan", plan):
                result = graph_costs(
                    raw, [replace(op, deps=())], isolated=False, input_ready_ms=7,
                    trace_limit=TRACE_EVENT_LIMIT if context < TRACE_CONTEXT_LIMIT else 0,
                )
            execution = result["execution"]
            pair.append({
                "label": label, "wall_seconds": time.monotonic() - started,
                "sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
                "duration_ms": result["duration_ms"],
                "tasks": execution.get("tasks_submitted", execution),
            })
        assert pair[0]["sha256"] == pair[1]["sha256"], (batch, context, pair)
        rows.append({"batch": batch, "context": context, "exact_result_and_trace_equal": True, "pair": pair})
        print(batch, context, [(row["label"], round(row["wall_seconds"], 4)) for row in pair], flush=True)
    Path("/tmp/paged-mla-template-validation-20261030.json").write_text(json.dumps({
        "reference_revision": REFERENCE_REVISION,
        "candidate_file_sha256": hashlib.sha256(Path(paged_mla.__file__).read_bytes()).hexdigest(),
        "cases": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
