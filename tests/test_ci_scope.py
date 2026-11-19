"""[Role] Unit tests for `.github/scripts/ci_scope.py`.

[Invariants]
- Exercises temporary Git repositories only; never mutates checkout state.
[Timeouts & Locks] Bounded local git calls (`<5s` Tier 0).
[Side-Effect Policy] Writes only inside temporary directories cleaned up on teardown.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "ci_scope.py"
_SPEC = importlib.util.spec_from_file_location("ci_scope", SCRIPT_PATH)
assert _SPEC and _SPEC.loader
ci_scope = importlib.util.module_from_spec(_SPEC)
sys.modules["ci_scope"] = ci_scope
_SPEC.loader.exec_module(ci_scope)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=True,
    )
    return proc.stdout.strip()


class TestCiScope(unittest.TestCase):
    def _init_repo(self, root: Path) -> str:
        _git(root, "init")
        _git(root, "config", "user.email", "ci@example.com")
        _git(root, "config", "user.name", "CI")
        (root / "README.md").write_text("# Demo\n", encoding="utf-8")
        (root / "src.py").write_text("VALUE = 1\n", encoding="utf-8")
        _git(root, "add", ".")
        _git(root, "commit", "-m", "init")
        return _git(root, "rev-parse", "HEAD")

    def test_is_pure_doc_path_boundaries(self) -> None:
        self.assertTrue(ci_scope.is_pure_doc_path("CONTRIBUTING.md"))
        self.assertTrue(ci_scope.is_pure_doc_path("AGENTS.md"))
        self.assertTrue(ci_scope.is_pure_doc_path("docs/contributing/ascend-a5.md"))
        self.assertTrue(ci_scope.is_pure_doc_path(".github/pull_request_template.md"))
        self.assertFalse(ci_scope.is_pure_doc_path("docs/releases/sim-0.9.19.md"))
        self.assertFalse(ci_scope.is_pure_doc_path("data/observations/sources/README.md"))
        self.assertFalse(ci_scope.is_pure_doc_path("tests/fixtures/README.md"))
        self.assertFalse(ci_scope.is_pure_doc_path("docs/evidence.json"))
        self.assertFalse(ci_scope.is_pure_doc_path(".github/workflows/ci.yml"))

    def test_pure_docs_pr_sets_docs_only_true_while_push_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            base = self._init_repo(repo)
            (repo / "CONTRIBUTING.md").write_text("# Guide\n", encoding="utf-8")
            (repo / ".github").mkdir()
            (repo / ".github" / "pull_request_template.md").write_text("## Summary\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "docs")
            head = _git(repo, "rev-parse", "HEAD")

            docs_only, paths, _ = ci_scope.classify_pr_scope(
                repo_root=repo, event_name="pull_request", base_sha=base, head_sha=head
            )
            self.assertTrue(docs_only)
            self.assertEqual(sorted(paths), [".github/pull_request_template.md", "CONTRIBUTING.md"])

            push_only, _, _ = ci_scope.classify_pr_scope(
                repo_root=repo, event_name="push", base_sha=base, head_sha=head
            )
            self.assertFalse(push_only)

    def test_code_workflow_and_release_notes_fail_closed(self) -> None:
        for touched in ("src.py", ".github/workflows/ci.yml", "docs/releases/v1.md"):
            with self.subTest(touched=touched), tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                base = self._init_repo(repo)
                target = repo / touched
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("updated\n", encoding="utf-8")
                _git(repo, "add", ".")
                _git(repo, "commit", "-m", "change")
                head = _git(repo, "rev-parse", "HEAD")
                docs_only, _, _ = ci_scope.classify_pr_scope(
                    repo_root=repo, event_name="pull_request", base_sha=base, head_sha=head
                )
                self.assertFalse(docs_only)

    def test_verify_needs_payload_modes(self) -> None:
        docs_payload = json.dumps({
            "quality": {"result": "success", "outputs": {"docs_only": "true"}},
            "test": {"result": "skipped", "outputs": {}},
            "wheel": {"result": "skipped", "outputs": {}},
        })
        ok, _ = ci_scope.verify_needs_payload(docs_payload, required_jobs=["quality", "test", "wheel"])
        self.assertTrue(ok)

        full_skip_payload = json.dumps({
            "quality": {"result": "success", "outputs": {"docs_only": "false"}},
            "test": {"result": "skipped", "outputs": {}},
            "wheel": {"result": "success", "outputs": {}},
        })
        ok, _ = ci_scope.verify_needs_payload(full_skip_payload, required_jobs=["quality", "test", "wheel"])
        self.assertFalse(ok)

        failed_quality = json.dumps({
            "quality": {"result": "failure", "outputs": {"docs_only": "true"}},
            "test": {"result": "skipped", "outputs": {}},
            "wheel": {"result": "skipped", "outputs": {}},
        })
        ok, _ = ci_scope.verify_needs_payload(failed_quality, required_jobs=["quality", "test", "wheel"])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
