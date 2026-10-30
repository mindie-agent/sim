"""Small-workload structural execution across all maintained example cases.

[Role] Detect broken adapters, missing geometry and incomplete executions.
[Invariants] Model dimensions are preserved. This is not historical workload
  reproduction, device validation or a model-accuracy acceptance result.
[Timeouts & Locks] One finite case at a time; bounded traces disabled.
[Side-Effect Policy] Writes only the requested JSON report.
"""
import argparse
import json
from pathlib import Path
import time

from mindie_sim import Case, predict


def smoke(root):
    rows = []
    for path in sorted(Path(root).glob("*.json")):
        raw = json.loads(path.read_text())
        raw["workload"] = {"requests": [{"id": "smoke", "prompt_tokens": 8, "output_tokens": 3}]}
        started = time.perf_counter()
        try:
            case = Case.from_mapping(raw)
            restored = Case.from_mapping(json.loads(json.dumps(case.to_dict())))
            if json.dumps(case.to_dict(), sort_keys=True) != json.dumps(restored.to_dict(), sort_keys=True):
                raise ValueError("case_roundtrip_changed_semantics")
            result = predict(case, trace_limit=0, event_limit=0)
            row = {"case": path.name, "status": result["status"],
                   "live_tasks": result["execution"]["live_tasks"],
                   "live_programs": result["execution"]["live_programs"]}
            row["passed"] = row["status"] == "modeled" and row["live_tasks"] == row["live_programs"] == 0
        except (ValueError, KeyError, TypeError) as exc:
            row = {"case": path.name, "status": "failed", "error": str(exc), "passed": False}
        row["wall_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"scope": "structural_smoke_p8_o3_original_model_dimensions_no_accuracy_claim", "cases": rows,
            "passed": sum(row["passed"] for row in rows), "total": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", default="examples/cases", nargs="?")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    report = smoke(args.cases)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
