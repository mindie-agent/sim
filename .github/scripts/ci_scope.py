#!/usr/bin/env python3
"""[Role] Classify CI pull-request diff scope and verify upstream job results.

[Invariants]
- Only `pull_request` diffs touching exclusively pure documentation Markdown
  (`*.md` outside `docs/releases/`, `docs/architecture*.md`, `data/`, and
  `tests/`) set `docs_only=true`.
- Architecture contract manifests (`docs/architecture*.md`) are never pure docs.
- Any `push` event (`main` / `codex/**`), empty/unresolvable diff, missing
  `README.md`, or non-doc file fails closed (`docs_only=false`).
- `verify-needs` always requires `quality == "success"`; `docs_only=false`
  requires every job to be `"success"`, whereas `docs_only=true` permits heavy
  downstream jobs to be `"skipped"` (never `"failure"` or `"cancelled"`).

[Timeouts & Locks]
- Git subprocess calls use a 30s timeout and `CREATE_NO_WINDOW` on Windows.

[Side-Effect Policy]
- Read-only except appending `docs_only=<true|false>` to `$GITHUB_OUTPUT` when set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

GIT_TIMEOUT_S = 30
EXCLUDED_DOC_PREFIXES = ("docs/releases/", "data/", "tests/")
ARCHITECTURE_CONTRACT_PREFIX = "docs/architecture"


def is_pure_doc_path(rel_path: str) -> bool:
    rel = rel_path.replace("\\", "/").lstrip("./")
    if not rel or ".." in rel.split("/"):
        return False
    if not rel.endswith(".md"):
        return False
    if rel.startswith(ARCHITECTURE_CONTRACT_PREFIX) and "/" not in rel[len("docs/"):]:
        return False
    return not rel.startswith(EXCLUDED_DOC_PREFIXES)


def classify_pr_scope(
    *,
    repo_root: Path,
    event_name: str,
    base_sha: str,
    head_sha: str,
) -> tuple[bool, list[str], str]:
    if event_name != "pull_request":
        return False, [], f"event={event_name or 'unknown'} runs full CI"
    if not base_sha or not head_sha:
        return False, [], "missing pull_request base/head SHA"
    if not (repo_root / "README.md").is_file():
        return False, [], "README.md missing from repository root"
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--no-renames", "--name-only", "-z", f"{base_sha}...{head_sha}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            stdin=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_S,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except TimeoutError:
        raise
    except OSError as exc:
        return False, [], f"git diff invocation failed: {exc}"
    if proc.returncode != 0:
        return False, [], f"git diff exited {proc.returncode}: {proc.stderr.strip()}"
    paths = [entry for entry in proc.stdout.split("\0") if entry]
    if not paths:
        return False, [], "empty diff fails closed"
    non_doc = [entry for entry in paths if not is_pure_doc_path(entry)]
    if non_doc:
        return False, paths, f"non-doc paths touched: {non_doc[:5]}"
    for rel in paths:
        candidate = repo_root / rel
        if candidate.is_file():
            try:
                candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                return False, paths, f"invalid UTF-8 in {rel}: {exc}"
    return True, paths, f"pure-docs PR ({len(paths)} markdown file(s))"


def verify_needs_payload(raw_json: str, *, required_jobs: Sequence[str]) -> tuple[bool, str]:
    try:
        needs = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return False, f"invalid RESULTS JSON: {exc}"
    if not isinstance(needs, dict):
        return False, "RESULTS payload must be a JSON object"
    missing = [job for job in required_jobs if job not in needs]
    if missing:
        return False, f"missing required jobs in RESULTS: {missing}"
    quality = needs.get("quality") or {}
    if quality.get("result") != "success":
        return False, f"quality job did not succeed: {quality.get('result')!r}"
    docs_only = (quality.get("outputs") or {}).get("docs_only") == "true"
    allowed = {"skipped", "success"} if docs_only else {"success"}
    bad = {
        job: (needs.get(job) or {}).get("result")
        for job in required_jobs
        if job != "quality" and (needs.get(job) or {}).get("result") not in allowed
    }
    if bad:
        mode = "docs-only" if docs_only else "full-suite"
        return False, f"{mode} CI check failed; unexpected job results: {bad}"
    mode = "docs-only fast path" if docs_only else "full suite"
    return True, f"release-ready verified ({mode})"


def _cmd_detect(args: argparse.Namespace) -> int:
    docs_only, _, reason = classify_pr_scope(
        repo_root=Path(args.repo_root).resolve(),
        event_name=args.event_name,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
    )
    flag = "true" if docs_only else "false"
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"docs_only={flag}\n")
    print(f"[ci-scope] docs_only={flag} ({reason})")
    return 0


def _cmd_verify_needs(args: argparse.Namespace) -> int:
    raw = os.environ.get(args.env_var, "")
    ok, message = verify_needs_payload(raw, required_jobs=args.jobs)
    stream = sys.stdout if ok else sys.stderr
    print(f"[ci-scope] {message}", file=stream)
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    detect = sub.add_parser("detect", help="Classify PR diff as docs-only or full-suite")
    detect.add_argument("--repo-root", default=".")
    detect.add_argument("--event-name", default="")
    detect.add_argument("--base-sha", default="")
    detect.add_argument("--head-sha", default="")
    detect.set_defaults(func=_cmd_detect)

    verify = sub.add_parser("verify-needs", help="Verify GitHub Actions needs JSON in release-ready")
    verify.add_argument("--env-var", default="RESULTS")
    verify.add_argument("--jobs", nargs="+", required=True)
    verify.set_defaults(func=_cmd_verify_needs)

    parsed = parser.parse_args(argv)
    return parsed.func(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
