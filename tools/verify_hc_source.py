"""Reproduce declared HcPre/HcPost calls without loading observations.

[Role] Execute ordinary-frontend HC source calls and retain their work ledger.
[Invariants] Explicit peak overrides are inputs, never inferred from timings.
  Missing physical supply remains an unsupported result, not a zero cost.
[Timeouts & Locks] Finite two-call local control; no locks or device work.
[Side-Effect Policy] Read small case/package files and write one JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mindie_sim
from mindie_sim.prediction.family_hc import (
    POST_PROFILE, PRE_PROFILE, pre_phase_work, selected_plan,
)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(root):
    files = {str(path.relative_to(root)): _sha(path)
             for path in sorted((root / "mindie_sim").rglob("*"))
             if path.suffix in {".py", ".json"} and path.is_file()}
    return {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def _work(work):
    return {"matrix": dict(work.matrix), "resources": dict(work.resources)}


def run(case_path, tokens, peaks, ready_ms):
    started = time.perf_counter()
    root = Path(mindie_sim.__file__).resolve().parent.parent
    before = _fingerprint(root)
    raw = json.loads(case_path.read_text())
    if peaks:
        raw["machine"].setdefault("device", {}).setdefault("compute_peaks", {}).update(peaks)
    raw.setdefault("runtime", {}).setdefault("kernels", {}).setdefault("lowerings", {})[
        "native_fusion"] = {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["hc_pre", "hc_post"], "traffic_profile": {
                "hc_pre": PRE_PROFILE, "hc_post": POST_PROFILE}},
        }
    case = parse_case(raw)
    geometry = Geometry.from_devices(case.machine.dies)
    semantic = target_operators(case, phase="prefill", tokens=tokens, resource_fusion=False)
    physical = lower_native_fusions(semantic, case.runtime)
    rows = []
    for role in ("hc_pre", "hc_post"):
        declaration, members = next(
            (declaration, members) for declaration, members in _groups(semantic)
            if declaration["role"] == role)
        op = next(op for op in physical
                  if dict(op.metadata).get("fusion_group_key") == declaration["key"])
        plan = selected_plan(op, geometry)
        row = {"role": role, "tensor_view": declaration["tensor_view"],
               "physical_work": _work(plan.work)}
        if role == "hc_pre":
            row["phases"] = {name: _work(work) for name, work in
                             pre_phase_work(declaration["tensor_view"], geometry).items()}
        try:
            result = operator_costs(case, (op,), input_ready_ms=ready_ms)["operators"][0]
        except ModelingError as exc:
            row.update(status="unsupported", reason=str(exc), duration_ms=None)
        else:
            row.update(status="modeled", **{
                key: result[key] for key in (
                    "duration_ms", "serviced", "service_before_input", "live_tasks",
                    "remaining_buffers") if key in result})
        rows.append(row)
    after = _fingerprint(root)
    return {
        "schema": "hc-source-control/v1",
        "boundary": "single_selected_source_operator_device_program",
        "source_root": str(root),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5).strip(),
        "package_before": before, "package_after": after,
        "package_stable": before == after,
        "case": {"path": str(case_path), "sha256": _sha(case_path)},
        "inputs": {"phase": "prefill", "tokens": tokens,
                   "input_ready_ms": ready_ms, "matrix_peak_overrides": peaks},
        "device": case.machine.dies[0].to_dict(),
        "rows": rows, "wall_seconds": time.perf_counter() - started,
        "limitations": [
            "Source profile selection does not establish the loaded capture binary.",
            "Pre retains four global phases; native panel overlap and MTE1 work are absent.",
            "Post uses aggregate vector operations; instruction/flag latency is unmodeled.",
            "No observation duration or PMU counter is read by this producer.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--matrix-peaks", type=json.loads, default={})
    parser.add_argument("--input-ready-ms", type=float, default=7)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.case.resolve(), args.tokens, args.matrix_peaks, args.input_ready_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": [
        {key: row[key] for key in ("role", "status", "duration_ms")}
        for row in report["rows"]], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
