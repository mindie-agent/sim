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


def activation_protocol() -> dict[str, Any]:
    return observation_protocol(
        "activation-call", ["activation_a.json", "activation_b.json"], "activation_producer.py.txt"
    )


def grouped_protocol() -> dict[str, Any]:
    return observation_protocol("grouped-call", ["grouped_probe.json"], "grouped_producer.py.txt")


def served_protocol() -> dict[str, Any]:
    return observation_protocol("grouped-served", ["served_grouped.csv"], "served_reducer.py.txt")


def norm_case() -> dict[str, Any]:
    from _support.memory_cases import case as small_case

    raw = small_case()
    raw["model"].update(hidden_size=6144, num_heads=48, num_kv_heads=8, head_dim=128)
    return raw


def captured_report(
    case: Any,
    capture: str,
    kind: str,
    *,
    phase: str = "decode",
    tokens: int = 1,
    repeat: int = 2,
    clip: bool = False,
    rank_id: str = "rank_0",
    structure: dict[str, Any] | None = None,
    host_targets: list[str] | tuple[str, ...] | None = None,
    source_row: int = 0,
    context: int = 0,
):
    """Map controlled repetitions of a retained profiler fusion fixture onto ``case``."""
    from dataclasses import replace
    from itertools import islice
    import mindie_profiler
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.structure_mapping import map_structure
    from mindie_sim.structure import export_structure

    rel = Path("tests/fixtures/fusion_contracts") / capture
    candidates = (
        Path(mindie_profiler.__file__).resolve().parents[1] / rel,
        REPO_ROOT.parent / "profiler" / rel,
        REPO_ROOT.parent.parent / "profiler" / rel,
        REPO_ROOT.parent / "profiler-clean-sync" / rel,
    )
    path = next((cand for cand in candidates if cand.is_file()), candidates[0])
    assert path.is_file(), f"retained fusion fixture unavailable: {path}"
    original = next(
        islice((row.event for row in kernel_rows(path, rank_id=rank_id) if row.kind == kind), source_row, None)
    )
    events = [
        replace(
            original,
            start_ns=index * 10_000_000,
            end_ns=index * 10_000_000 + original.duration_ns * (index + 1),
            task_id=str(index + 1),
            locator={**original.locator, "csv_line": index + 2},
        )
        for index in range(repeat)
    ]
    window = Window(1 if clip else 0, events[-1].end_ns, step=19)
    stat = path.stat()
    observation = Observation(
        rank_id,
        0,
        window,
        None,
        events,
        SourceRef(
            rank_id=rank_id,
            rank_index=0,
            kind="csv",
            path=str(path),
            basename=path.name,
            sha256=None,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            schema_version="retained_capture",
            schema_variant="kernel_details",
        ),
        QueryStats(rank_id, "controlled_repetitions"),
    )
    if structure is None:
        structure = export_structure(case, phase=phase, tokens=tokens, context=context, scope="body")
    options: dict[str, Any] = {}
    if host_targets is not None:
        assert len(host_targets) == len(events)
        scopes, calls = [], []
        for index, (event, target) in enumerate(zip(events, host_targets)):
            name = f"controlled.host.scope{index}"
            scopes.append({"name": name, "node_ids": [target]})
            calls.append({
                "device_task": {
                    "task_id": event.task_id,
                    "stream_id": event.stream_id,
                    "device_id": event.device_id,
                    "start_ns": event.start_ns,
                    "end_ns": event.end_ns,
                    "global_task_id": event.locator.get("task_global_id"),
                },
                "global_task_id_unique": True,
                "association_basis": {"pytorch": "connection_ids"},
                "association_unique": {"pytorch": True},
                "host_stack": [{"name": name, "table": "PYTORCH_API"}],
            })
        options = {
            "bindings": {
                "schema_version": "mindie-profile-bindings/1",
                "graph_id": structure["graph_id"],
                "host_scopes": scopes,
            },
            "correlations": [{
                "schema_version": "mindie-profiler-correlation/1",
                "rank_id": rank_id,
                "source_identity": {
                    "path": observation.source.path,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                },
                "correlated_calls": calls,
            }],
        }
    return map_structure([observation], structure, mapping_limit=0, **options), events
