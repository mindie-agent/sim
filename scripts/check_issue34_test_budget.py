#!/usr/bin/env python3
"""Per-file Issue #34 test wall-time and new-module budget gate.

[Role] Run every new Issue #34 test file in its own pytest subprocess, measure
  the whole subprocess with a monotonic clock and fail at or above the declared
  limit; also check module header/line budgets and legacy integration deltas.
[Invariants] No retries or best-of sampling: one wall clock around the entire
  subprocess; a timeout is a failure. Header/line budgets read actual files.
[Timeouts & Locks] Each subprocess has an explicit timeout; no background work.
[Side-Effect Policy] Writes only JUnit/basetemp artifacts under the supplied
  directory and removes an internally created directory on every exit path.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LIMIT_SECONDS = 3.0
TIMEOUT_MULTIPLIER = 10.0
MIN_TIMEOUT_SECONDS = 30.0
HEADER_SECTIONS = ("[Role]", "[Invariants]", "[Timeouts & Locks]", "[Side-Effect Policy]")
HEADER_WINDOW = 30
ZERO = 0
TEST_LIMIT = 400
NEW_MODULE_LIMITS = {
    "mindie_sim/schema/runtime_features.py": 350,
    "mindie_sim/schema/compute_peak_aliases.py": 200,
    "mindie_sim/issue34_cli.py": 350,
    "mindie_sim/prediction/collective_startup.py": 220,
    "mindie_sim/prediction/collective_waves.py": 390,
    "mindie_sim/prediction/matrix_precision.py": 250,
    "mindie_sim/catalog/projection_placement.py": 390,
    "mindie_sim/prediction/local_tp_groups.py": 250,
    "mindie_sim/prediction/local_projection_tp.py": 390,
    "mindie_sim/catalog/speculative_state.py": 300,
    "mindie_sim/prediction/recover_ssm_commit.py": 300,
    "mindie_sim/catalog/draft_geometry.py": 300,
    "mindie_sim/catalog/cache_ownership.py": 390,
    "mindie_sim/prediction/prefix_memory.py": 300,
    "mindie_sim/prediction/prefix_reads.py": 300,
    "mindie_sim/prediction/batch_prefix_indexer.py": 390,
    "mindie_sim/prediction/batch_prefix_attention.py": 390,
    "mindie_sim/prediction/capacity_features.py": 350,
    "mindie_sim/catalog/expert_slots.py": 300,
    "mindie_sim/prediction/expert_slot_work.py": 350,
    "mindie_sim/hardware_derivation.py": 390,
    "mindie_sim/hardware_efficiency.py": 220,
    "mindie_sim/prediction/recurrent_template.py": 250,
    "mindie_sim/prediction/cohort_reuse.py": 390,
    "mindie_sim/prediction/graph_cache.py": 300,
    "mindie_sim/prediction/exposed_breakdown.py": 299,
    "mindie_sim/prediction/exposed_program.py": 350,
    "mindie_sim/prediction/runtime/exposed_observer.py": 390,
    "mindie_sim/prediction/runtime/exposed_intervals.py": 350,
    "tests/issue34_helpers.py": 300,
    "scripts/check_no_magic_numbers.py": 350,
    "scripts/check_issue34_test_budget.py": 200,
}
LEGACY_OVERSIZED = (
    "mindie_sim/prediction/topology.py", "mindie_sim/compose.py",
    "mindie_sim/prediction/capacity.py", "mindie_sim/prediction/frontend.py",
    "mindie_sim/prediction/efficiency.py", "mindie_sim/prediction/execution.py",
    "mindie_sim/prediction/program.py", "mindie_sim/custom_hardware.py",
    "mindie_sim/schema/hardware.py", "mindie_sim/schema/compute_arch.py",
)


def check_new_modules(root: Path) -> list[str]:
    problems: list[str] = []
    for name, limit in sorted(NEW_MODULE_LIMITS.items()):
        path = root / name
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        missing = [item for item in HEADER_SECTIONS
                   if item not in "\n".join(lines[:HEADER_WINDOW])]
        if missing:
            problems.append(f"{name}: missing header sections {missing}")
        if len(lines) > limit:
            problems.append(f"{name}: {len(lines)} lines exceeds limit {limit}")
        if name.startswith("tests/") and len(lines) >= TEST_LIMIT:
            problems.append(f"{name}: {len(lines)} lines >= {TEST_LIMIT}")
    return problems


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    return completed.stdout if completed.returncode == ZERO else ""


def legacy_deltas(root: Path, base: str | None) -> list[tuple[str, int]]:
    if not base:
        return []
    result = []
    for name in LEGACY_OVERSIZED:
        if not (root / name).is_file():
            continue
        added = ZERO
        for line in _git(root, "diff", "--unified=0", base, "--", name).splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            text = line[1:].strip()
            if text and not text.startswith(("#", "from ", "import ", ")", "}", "]")):
                added += 1
        result.append((name, added))
    return result


def run_file(root: Path, path: Path, *, limit: float, artifacts: Path) -> dict:
    basetemp = artifacts / f"tmp-{path.stem}"
    env = dict(os.environ)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{root}:{root / 'tests'}",
        "TMPDIR": str(artifacts / "tmp"),
        "MINDIE_CACHE_HOME": str(artifacts / "cache"),
    })
    (artifacts / "tmp").mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p",
               "no:cacheprovider", str(path), f"--basetemp={basetemp}"]
    start = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command, cwd=root, env=env, check=False, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            timeout=max(MIN_TIMEOUT_SECONDS, limit * TIMEOUT_MULTIPLIER))
        status, output = completed.returncode, completed.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out, status = True, -1
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
    elapsed = time.monotonic() - start
    return {"file": str(path.relative_to(root)), "elapsed_s": elapsed,
            "status": status, "timed_out": timed_out,
            "over_budget": elapsed >= limit, "output_tail": output[-2000:]}


def _run(args, root: Path, artifacts: Path) -> int:
    files = sorted(root.glob(args.pattern))
    problems = check_new_modules(root)
    problems.extend(f"{path.relative_to(root)}: {len(path.read_text().splitlines())}"
                    " lines >= 400"
                    for path in files if len(path.read_text().splitlines()) >= 400)
    results = []
    if not args.skip_tests:
        for path in files:
            result = run_file(root, path, limit=args.limit_seconds,
                              artifacts=artifacts)
            results.append(result)
            state = ("TIMEOUT" if result["timed_out"] else
                     "FAIL" if result["status"] != ZERO else
                     "OVER" if result["over_budget"] else "PASS")
            print(f"{state:7s} {result['elapsed_s']:.3f}s {result['file']}")
            if result["status"] != ZERO or result["over_budget"]:
                problems.append(f"{result['file']}: status={result['status']} "
                                f"elapsed={result['elapsed_s']:.3f}s")
                print(result["output_tail"], file=sys.stderr)
    for name, added in legacy_deltas(root, args.base):
        print(f"legacy integration delta {name}: {added} added executable lines")
    for problem in problems:
        print(f"ERROR {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"issue34 budget: {len(results)} test files under "
          f"{args.limit_seconds}s; headers/line limits ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="tests/test_issue34_*.py")
    parser.add_argument("--limit-seconds", type=float,
                        default=DEFAULT_LIMIT_SECONDS)
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--base", default=None)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root or SCRIPT_DIR.parent).resolve()
    created_temp = args.artifacts is None
    artifacts = Path(args.artifacts or tempfile.mkdtemp(
        prefix="issue34-budget-")).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        return _run(args, root, artifacts)
    finally:
        if created_temp:
            shutil.rmtree(artifacts, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
