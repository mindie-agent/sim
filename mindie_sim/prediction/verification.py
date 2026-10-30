"""Independent checks on an isolated operator's resource contract.

[Role] Expose work conservation, buffer release and finite completion to authors.
[Invariants] Structural correctness is separate from measurement accuracy.
  No comparison result supplies a simulation parameter.
[Timeouts & Locks] Uses one bounded engine run per input; no external work.
[Side-Effect Policy] Returns diagnostics without changing recipes or calibration.
"""
from __future__ import annotations

import math

from .operator_analysis import operator_costs
from .program import RecipeBook
from .mechanism import PlannedRecipe


def _check_row(row, rel_tol):
    declared, serviced = row["declared_service"], row["serviced"]
    mismatches = {
        name: {"declared": declared.get(name, 0), "executed": serviced.get(name, 0)}
        for name in declared.keys() | serviced.keys()
        if not math.isclose(declared.get(name, 0), serviced.get(name, 0),
                            rel_tol=rel_tol, abs_tol=1e-12)
    }
    finite = math.isfinite(row["duration_ms"]) and row["duration_ms"] >= 0
    released = not any(row["remaining_buffers"].values()) and row["live_tasks"] == 0
    ready_obeyed = not any(row["service_before_input"].values())
    return {"finite_completion": finite, "buffers_released": released,
            "input_dependency_obeyed": ready_obeyed, "work_mismatches": mismatches}


def verify_operator(case, op, build, *, kernels=None, rel_tol=1e-9, input_ready_ms=0):
    """Check the actual public isolated-cost path, not a second recipe executor."""
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(build, kernels)})
    rows = operator_costs(case, (op,), recipes=recipes,
                          input_ready_ms=input_ready_ms)["operators"]
    checks = [_check_row(item, rel_tol) for item in rows]
    if kernels is None:
        row = rows[0]
        mismatches = checks[0]["work_mismatches"]
    else:
        def total(field):
            keys = {key for item in rows for key in item[field]}
            return {key: sum(item[field].get(key, 0) * item["occurrences"] for item in rows)
                    for key in keys}
        row = {
            "operator": op.to_dict(), "kernels": rows,
            "duration_ms": sum(item["total_ms"] for item in rows),
            "declared_service": total("declared_service"), "serviced": total("serviced"),
            "remaining_buffers": total("remaining_buffers"),
            "service_before_input": total("service_before_input"),
            "live_tasks": sum(item["live_tasks"] for item in rows),
        }
        # Opposite errors in two kernels must not cancel in the display total.
        mismatches = {item["operator"]["op_id"]: check["work_mismatches"]
                      for item, check in zip(rows, checks) if check["work_mismatches"]}
    finite = all(check["finite_completion"] for check in checks)
    released = all(check["buffers_released"] for check in checks)
    ready_obeyed = all(check["input_dependency_obeyed"] for check in checks)
    return {"status": "passed" if finite and released and ready_obeyed and not mismatches else "failed",
            "scope": "resource_contract_only", "finite_completion": finite,
            "buffers_released": released,
            "input_dependency_obeyed": ready_obeyed,
            "kernel_checks": [{"operator_id": item["operator"]["op_id"], **check}
                              for item, check in zip(rows, checks)],
            "work_mismatches": mismatches, "result": row}
