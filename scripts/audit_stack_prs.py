#!/usr/bin/env python3
"""Audit open vs merged PRs for any registered RuntimeStackProfile repository.

[Role]
  Cross-check a registered ``RuntimeStackProfile``'s ``unmerged_pr`` bindings
  against live ``gh`` PR status (or a deterministic ``--offline-fixture`` JSON
  snapshot) to detect PRs that have merged onto ``main`` (ready to promote from
  ``unmerged_pr`` to ``merged``), closed/superseded PRs, and newly opened
  ``[Performance]`` PRs not yet cataloged.

[Invariants]
  * Parameterized by ``--stack`` and ``--repo`` (defaults to ``vllm-ascend`` /
    ``vllm-project/vllm-ascend``; works identically for ``sgl-project/sglang``,
    ``vllm-project/vllm``, or custom JSON stack manifests).
  * ``--offline-fixture`` performs zero subprocess or network calls.
  * Subprocess ``gh`` invocations use explicit ``timeout=30`` seconds.

[Timeouts & Locks]
  Bounded 30s subprocess timeout per ``gh`` query; no locks.

[Side-Effect Policy]
  Read-only: prints JSON report (or writes only to explicit ``--output``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mindie_sim.feature_mechanisms import resolve_runtime_stack  # noqa: E402

GH_TIMEOUT_SECONDS = 30
GH_OPEN_LIMIT = 100
PR_REF_RE = re.compile(r"^(?:([^#]+))?#(\d+)$")
PERF_KEYWORDS = (
    "performance", "fuse", "overlap", "triton", "chunk", "chunk32",
    "recoverssm", "megamoe", "mhc", "gbsa", "sfa", "indexer", "dspark", "fa3",
)


def _parse_pr_ref(ref: str | None, default_repo: str) -> tuple[str, int] | None:
    if not ref:
        return None
    match = PR_REF_RE.match(ref.strip())
    if not match:
        return None
    repo = match.group(1) or default_repo
    return repo, int(match.group(2))


def _gh_json(args: list[str]) -> Any:
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=GH_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def load_live_snapshot(repo: str, tracked_numbers: list[int]) -> dict[str, Any]:
    open_prs = _gh_json([
        "pr", "list", "-R", repo, "--state", "open",
        "--limit", str(GH_OPEN_LIMIT),
        "--json", "number,title,state,labels,headRefName,url",
    ])
    by_number: dict[str, Any] = {str(item["number"]): item for item in open_prs}
    for num in tracked_numbers:
        if str(num) not in by_number:
            detail = _gh_json([
                "pr", "view", str(num), "-R", repo,
                "--json", "number,title,state,mergedAt,url",
            ])
            by_number[str(num)] = detail
    return {"repository": repo, "open_prs": open_prs, "tracked_prs": by_number}


def audit_snapshot(stack_name: str, repo: str, bindings: list[dict[str, Any]],
                   snapshot: dict[str, Any]) -> dict[str, Any]:
    tracked_map = snapshot.get("tracked_prs") or {}
    open_prs = snapshot.get("open_prs") or []
    cataloged_numbers: set[int] = set()
    ready_to_promote: list[dict[str, Any]] = []
    closed_unmerged: list[dict[str, Any]] = []
    still_open: list[dict[str, Any]] = []

    for binding in bindings:
        if binding.get("state") != "unmerged_pr":
            continue
        parsed = _parse_pr_ref(binding.get("ref"), repo)
        if parsed is None or parsed[0] != repo:
            continue
        num = parsed[1]
        cataloged_numbers.add(num)
        info = tracked_map.get(str(num)) or {}
        state = str(info.get("state") or "UNKNOWN").upper()
        row = {
            "mechanism_id": binding["mechanism_id"],
            "pr_number": num,
            "ref": binding["ref"],
            "state": state,
            "merged_at": info.get("mergedAt"),
            "title": info.get("title", ""),
        }
        if state == "MERGED" or info.get("mergedAt"):
            ready_to_promote.append(row)
        elif state == "CLOSED":
            closed_unmerged.append(row)
        else:
            still_open.append(row)

    uncataloged_candidates = []
    for pr in open_prs:
        num = int(pr["number"])
        if num in cataloged_numbers:
            continue
        title = str(pr.get("title") or "")
        lower_title = title.lower()
        labels = [str(label.get("name") or "") for label in (pr.get("labels") or ())]
        if any(keyword in lower_title for keyword in PERF_KEYWORDS) or "module:ops" in labels:
            uncataloged_candidates.append({
                "pr_number": num,
                "title": title,
                "labels": labels,
                "url": pr.get("url") or f"https://github.com/{repo}/pull/{num}",
            })

    return {
        "stack": stack_name,
        "repository": repo,
        "ready_to_promote_to_merged": ready_to_promote,
        "closed_without_merge": closed_unmerged,
        "active_open_prs": still_open,
        "uncataloged_performance_candidates": uncataloged_candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default="vllm-ascend",
                        help="registered stack name, 'cann93', or JSON manifest path")
    parser.add_argument("--repo", default=None,
                        help="override GitHub owner/repo (defaults to stack.repository)")
    parser.add_argument("--offline-fixture", default=None,
                        help="path to offline JSON fixture for deterministic testing")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args(argv)

    stack_arg = args.stack
    if str(stack_arg).strip().lower() in {"cann93", "vllm-ascend-cann93"}:
        from mindie_sim.feature_mechanisms import VLLM_ASCEND_MANIFEST_PATH as MANIFEST_PATH
        stack_arg = MANIFEST_PATH
    profile = resolve_runtime_stack(stack_arg)
    repo = args.repo or profile.repository
    bindings = [b.to_dict() for b in profile.bindings]
    tracked_nums = [
        parsed[1] for b in bindings
        if b["state"] == "unmerged_pr"
        and (parsed := _parse_pr_ref(b.get("ref"), repo)) is not None
        and parsed[0] == repo
    ]
    if args.offline_fixture:
        snapshot = json.loads(Path(args.offline_fixture).read_text(encoding="utf-8"))
    else:
        snapshot = load_live_snapshot(repo, tracked_nums)

    report = audit_snapshot(profile.name, repo, bindings, snapshot)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
