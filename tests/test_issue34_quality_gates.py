"""Differential numeric-literal and new-module budget gate behavior."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_no_magic_numbers.py"
BUDGET = ROOT / "scripts/check_issue34_test_budget.py"
HEADER = ('"""[Role] x\n[Invariants] x\n[Timeouts & Locks] x\n'
          '[Side-Effect Policy] x"""\n')


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True)


def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "mindie_sim").mkdir(parents=True)
    git(root, "init")
    git(root, "config", "user.email", "issue34@example.invalid")
    git(root, "config", "user.name", "issue34")
    (root / "mindie_sim/a.py").write_text(
        HEADER + "OLD = 3\n\ndef f():\n    return OLD\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    return root


def run(*args):
    return subprocess.run([sys.executable, "-B", *map(str, args)],
                          cwd=ROOT, check=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


def test_new_literal_is_reported_and_named_constant_is_allowed(tmp_path):
    root = repo(tmp_path)
    base = git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "mindie_sim/a.py").write_text(
        HEADER + "OLD = 3\nNEW = 4\n\ndef f():\n    return OLD + NEW\n",
        encoding="utf-8")
    allowed = run(CHECKER, "--base", base, "--root", root)
    assert allowed.returncode == 0, allowed.stdout
    assert "new numeric literal violations: 0" in allowed.stdout
    (root / "mindie_sim/a.py").write_text(
        HEADER + "OLD = 3\n\ndef f():\n    return OLD + 1\n", encoding="utf-8")
    captured = run(CHECKER, "--base", base, "--root", root)
    assert captured.returncode != 0
    assert "new numeric literal 1" in captured.stdout


def test_new_literal_inside_an_old_function_is_detected(tmp_path):
    root = repo(tmp_path)
    base = git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "mindie_sim/a.py").write_text(
        HEADER + "OLD = 3\n\ndef f():\n    value = OLD\n    return value\n",
        encoding="utf-8")
    git(root, "add", ".")
    assert run(CHECKER, "--base", base, "--root", root).returncode == 0
    (root / "mindie_sim/a.py").write_text(
        HEADER + "OLD = 3\n\ndef f():\n    value = OLD + 2\n    return value\n",
        encoding="utf-8")
    captured = run(CHECKER, "--base", base, "--root", root)
    assert captured.returncode != 0
    assert "new numeric literal 2" in captured.stdout


def test_untracked_module_and_oversized_header_gate(tmp_path):
    root = tmp_path / "budget"
    target = root / "mindie_sim/prediction/exposed_breakdown.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n" + "\n".join("pass" for _ in range(310)),
                      encoding="utf-8")
    captured = run(BUDGET, "--root", root, "--skip-tests")
    assert captured.returncode != 0
    assert "missing header sections" in captured.stdout
    assert "exceeds limit 299" in captured.stdout


def test_valid_new_module_header_and_limit_pass(tmp_path):
    root = tmp_path / "budget-ok"
    target = root / "mindie_sim/prediction/exposed_breakdown.py"
    target.parent.mkdir(parents=True)
    target.write_text(HEADER + "VALUE = 1\n", encoding="utf-8")
    captured = run(BUDGET, "--root", root, "--skip-tests")
    assert captured.returncode == 0, captured.stdout
    assert "headers/line limits ok" in captured.stdout
