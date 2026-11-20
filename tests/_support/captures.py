"""Golden-capture and observation-protocol fixture helpers for tests.

[Role] Centralize repository-relative resolution of ``tests/fixtures/`` and
  SHA-256-verified observation protocol manifests so domain tests never build
  ad-hoc ``Path(__file__).parents[...]`` chains or fabricate unverified mocks.
[Invariants] Reads only tracked files under ``tests/fixtures/`` and
  ``data/observations/protocols/``; computes fresh SHA-256 digests per call.
[Timeouts & Locks] Local file reads on small JSON/CSV fixtures; no locks.
[Side-Effect Policy] Read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
PROTOCOLS_ROOT = REPO_ROOT / "data" / "observations" / "protocols"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fixture_path(*parts: str) -> Path:
    return FIXTURES_ROOT.joinpath(*parts)


def fixture_json(*parts: str) -> Any:
    return json.loads(fixture_path(*parts).read_text(encoding="utf-8"))


def observation_protocol(
    protocol_name: str,
    captures: tuple[str, ...] | list[str],
    producer: str,
    *,
    subdir: str = "observations/capture_adapters",
) -> dict[str, Any]:
    """Bind a protocol template in ``data/observations/protocols`` to golden fixtures."""
    base = fixture_path(*subdir.split("/"))
    manifest = json.loads((PROTOCOLS_ROOT / f"{protocol_name}-protocol.json").read_text(encoding="utf-8"))
    producer_path = base / producer
    manifest["producer"] = {"path": str(producer_path), "sha256": sha256_file(producer_path)}
    manifest["supporting_sources"] = []
    manifest["captures"] = [
        {"path": str(base / name), "sha256": sha256_file(base / name), "label": name}
        for name in captures
    ]
    return manifest
