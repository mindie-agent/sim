"""Streaming JSONL sweep checkpoints with bounded torn-tail repair.

[Role] Index, repair and append committed sweep rows without whole-file reads.
[Invariants] A checkpoint row is valid only when it is a complete JSON object
  carrying a candidate identity and an evaluation stage.  An incomplete trailing
  line is truncated on resume; no evaluated row is ever dropped silently.
[Timeouts & Locks] One sequential pass bounded by ``CHECKPOINT_INDEX_LIMIT``.
[Side-Effect Policy] Writes only the caller's explicit checkpoint stream.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from typing import Any

from mindie_sim.sweep_engine import B1_STAGE


MAX_CHECKPOINT_ROW_BYTES = 4 * 1024 * 1024
CHECKPOINT_INDEX_LIMIT = 100000
CANDIDATE_ID_KEY = "candidate_id"
STAGE_SCREEN = "screen"
STAGE_FULL = "full"
STAGES = frozenset({STAGE_SCREEN, STAGE_FULL})


@dataclass(frozen=True)
class CheckpointRef:
    """One committed row position and the evaluation stage it records."""

    offset: int
    stage: str


def _stage_of(row: Mapping[str, Any]) -> str:
    stage = row.get("search_stage")
    if stage in STAGES:
        return str(stage)
    return STAGE_SCREEN if row.get("evaluation_stage") == B1_STAGE else STAGE_FULL


def checkpoint_index(stream) -> dict[str, CheckpointRef]:
    """Index complete lines and truncate one incomplete trailing line."""
    index: dict[str, CheckpointRef] = {}
    while stream is not None:
        offset = stream.tell()
        line = stream.readline(MAX_CHECKPOINT_ROW_BYTES)
        if not line:
            break
        if not line.endswith(b"\n"):
            if len(line) == MAX_CHECKPOINT_ROW_BYTES:
                raise ValueError("checkpoint_row_too_large")
            # The writer flushes complete rows; a tail without its newline is
            # an interrupted append and is removed before new rows are written.
            stream.seek(offset)
            stream.truncate()
            break
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict)
                or not row.get(CANDIDATE_ID_KEY)
                or _stage_of(row) not in STAGES):
            raise ValueError("invalid_sweep_checkpoint_row")
        index[str(row[CANDIDATE_ID_KEY])] = CheckpointRef(
            offset, _stage_of(row))
        if len(index) > CHECKPOINT_INDEX_LIMIT:
            raise ValueError("checkpoint_index_limit_exceeded")
    return index


def read_checkpoint_row(stream, ref: CheckpointRef) -> dict[str, Any]:
    """Read exactly the indexed line, never the whole checkpoint."""
    stream.seek(ref.offset)
    return json.loads(stream.readline(MAX_CHECKPOINT_ROW_BYTES))


def write_checkpoint_row(stream, row: Mapping[str, Any]) -> CheckpointRef:
    """Append one compact row and return its committed position."""
    if stream is None:
        return CheckpointRef(0, _stage_of(dict(row)))
    payload = (json.dumps(dict(row), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False) + "\n").encode("utf-8")
    if len(payload) >= MAX_CHECKPOINT_ROW_BYTES:
        raise ValueError("checkpoint_row_too_large")
    stream.seek(0, os.SEEK_END)
    offset = stream.tell()
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    return CheckpointRef(offset, _stage_of(dict(row)))


def open_checkpoint(path, resume: bool):
    """Open a checkpoint for index/append, repairing only on resume."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a+b" if resume else "w+b")
