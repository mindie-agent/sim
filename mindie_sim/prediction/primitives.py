"""Independent kernel measurements consumed by the public predictor.

[Role] Store repeated, implementation-matched primitive observations.
[Invariants] Keys contain execution geometry, never model/request/cell names.
  Held-out observations are readable evidence but cannot price a prediction.
  A missing measurement remains a miss; a neighbouring cell is not substituted.
[Timeouts & Locks] SQLite WAL and a five-second busy timeout on writable open.
  Queries are indexed and bounded; the reader does not load the database.
[Side-Effect Policy] Only an explicitly writable store can create or edit files.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Mapping


@dataclass(frozen=True)
class PrimitiveKey:
    """The complete declared condition of one executable kernel."""

    document: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PrimitiveKey:
        required = {"hardware", "implementation", "kernel", "dtype", "shape", "scope"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"primitive_identity_missing:{','.join(sorted(missing))}")
        return cls(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.document.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.document)


@dataclass(frozen=True)
class PrimitiveTiming:
    duration_ms: float
    trials: int
    minimum_ms: float
    maximum_ms: float
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "trials": self.trials,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "sources": list(self.sources),
            "estimator": "median_of_trial_medians",
            "basis": "independent_primitive",
        }


class PrimitiveStore:
    """A small exact-condition store; context management owns its connection."""

    def __init__(self, path: str | Path, *, writable: bool = False):
        self.writable = writable
        if str(path) == ":memory:":
            self.connection = sqlite3.connect(":memory:", timeout=5.0)
            self.writable = True
        elif writable:
            self.connection = sqlite3.connect(str(path), timeout=5.0)
        else:
            uri = Path(path).expanduser().resolve().as_uri() + "?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._cache: OrderedDict[str, PrimitiveTiming | None] = OrderedDict()
        if self.writable:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS primitive_samples (
                    condition TEXT NOT NULL,
                    identity TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    repeat_index INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    source TEXT NOT NULL,
                    split TEXT NOT NULL,
                    PRIMARY KEY (condition, run_id, repeat_index, split)
                )
            """)
            self.connection.commit()
        else:
            # Validate the public store contract once, before any simulation.
            self.connection.execute("SELECT condition FROM primitive_samples LIMIT 0")

    def __enter__(self) -> PrimitiveStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def record(
        self, key: PrimitiveKey, *, duration_ms: float, run_id: str,
        repeat_index: int, source: str, split: str = "train",
    ) -> None:
        if not self.writable:
            raise ValueError("primitive_store_read_only")
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError("primitive_duration_must_be_positive")
        if not run_id or not source:
            raise ValueError("primitive_measurement_requires_run_and_source")
        if split not in {"train", "heldout"}:
            raise ValueError("primitive_split_invalid")
        values = (
            key.digest, key.document, run_id, repeat_index,
            float(duration_ms), source, split,
        )
        existing = self.connection.execute(
            "SELECT * FROM primitive_samples "
            "WHERE condition=? AND run_id=? AND repeat_index=? AND split=?",
            (key.digest, run_id, repeat_index, split),
        ).fetchone()
        if existing is not None and tuple(existing) != values:
            raise ValueError("primitive_repeat_conflicts_with_recorded_measurement")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO primitive_samples VALUES (?,?,?,?,?,?,?)", values,
            )
        self._cache.pop(key.digest, None)

    def lookup(self, key: PrimitiveKey) -> PrimitiveTiming | None:
        digest = key.digest
        if digest in self._cache:
            self._cache.move_to_end(digest)
            return self._cache[digest]
        # LIMIT applies to observations at one condition, not the whole store.
        # Excessive repeats are a data issue, not a reason to silently truncate.
        rows = self.connection.execute(
            "SELECT run_id,duration_ms,source FROM primitive_samples "
            "WHERE condition=? AND split='train' ORDER BY run_id,repeat_index LIMIT 10001",
            (digest,),
        ).fetchall()
        if len(rows) > 10000:
            raise ValueError(f"primitive_condition_repeat_limit:{digest}")
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(row["run_id"], []).append(float(row["duration_ms"]))
        medians = [statistics.median(values) for values in grouped.values()]
        result = None if not medians else PrimitiveTiming(
            duration_ms=statistics.median(medians),
            trials=len(medians), minimum_ms=min(medians), maximum_ms=max(medians),
            sources=tuple(sorted({str(row["source"]) for row in rows})),
        )
        self._cache[digest] = result
        if len(self._cache) > 1024:
            self._cache.popitem(last=False)
        return result

    def inventory(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT split,COUNT(*),COUNT(DISTINCT condition) "
            "FROM primitive_samples GROUP BY split",
        )
        return {
            split: {"repeats": repeats, "conditions": conditions}
            for split, repeats, conditions in rows
        }
