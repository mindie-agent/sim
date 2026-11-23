"""Bounded multi-core MapReduce worker governance for deployment sweeps and solvers.

[Role] Resolve effective worker concurrency from ``MINDIE_GRANTED_WORKERS`` or
  explicit ``granted_workers`` parameters (clamped to ``[1, 64]`` with
  automatic test-isolation clamping to ``1`` under ``pytest`` / ``pytest-xdist``),
  and execute deterministic, order-preserving parallel Map steps over a single
  flat ``ProcessPoolExecutor`` with guaranteed process-reaping shutdown.
[Invariants]
  * ``resolve_granted_workers()`` always returns an integer in ``[1, 64]``.
  * Whenever ``PYTEST_CURRENT_TEST`` or ``PYTEST_XDIST_WORKER`` is set and
    ``granted_workers`` was NOT explicitly passed, effective workers clamp to
    ``1`` so ``pytest-xdist`` shards never oversubscribe host cores.
  * Child workers pin ``MINDIE_GRANTED_WORKERS=1`` so nested pools never spawn
    (Principle 18: one flat pool per workflow).
  * ``map_ordered`` preserves exact input index order so downstream Reduce
    stages are bit-for-bit identical across ``workers=1`` and ``workers=N``.
[Timeouts & Locks] Uses ``fork`` only on single-threaded Linux processes and
  ``spawn`` elsewhere; unconditionally executes
  ``shutdown(wait=True, cancel_futures=True)`` on pool exit so zero zombie
  worker processes remain.
[Side-Effect Policy] Pure in-memory worker orchestration; mutates only the
  child worker process's own ``MINDIE_GRANTED_WORKERS`` environment variable.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import sys
import threading
from typing import Any, TypeVar

from mindie_sim.schema.common import ModelingError

#: Environment variable populated by ``mindie-harness`` remote/local resource governors.
GRANTED_WORKERS_ENV = "MINDIE_GRANTED_WORKERS"
#: Hard lower and upper bounds on MapReduce worker pool fan-out.
MIN_GRANTED_WORKERS = 1
MAX_GRANTED_WORKERS = 64
_PYTEST_GUARD_ENV_KEYS = ("PYTEST_CURRENT_TEST", "PYTEST_XDIST_WORKER")

_T = TypeVar("_T")
_R = TypeVar("_R")


def _clamp_workers(raw_value: int) -> int:
    return max(MIN_GRANTED_WORKERS, min(MAX_GRANTED_WORKERS, int(raw_value)))


def resolve_granted_workers(
    granted_workers: int | None = None,
    *,
    task_count: int | None = None,
) -> int:
    """Resolve effective MapReduce worker count clamped to ``[1, 64]``.

    When ``granted_workers`` is ``None`` and ``PYTEST_CURRENT_TEST`` or
    ``PYTEST_XDIST_WORKER`` is set, this automatically clamps to ``1`` to
    prevent ``pytest-xdist`` core oversubscription while still honoring
    explicit ``granted_workers=N`` calls in determinism tests.
    """
    if granted_workers is not None:
        if isinstance(granted_workers, bool) or not isinstance(granted_workers, int):
            raise ModelingError(f"invalid_granted_workers:{granted_workers!r}")
        effective = _clamp_workers(granted_workers)
    elif any(os.environ.get(key) for key in _PYTEST_GUARD_ENV_KEYS):
        effective = MIN_GRANTED_WORKERS
    else:
        raw_env = os.environ.get(GRANTED_WORKERS_ENV)
        if raw_env is None or not str(raw_env).strip():
            effective = MIN_GRANTED_WORKERS
        else:
            try:
                parsed = int(str(raw_env).strip())
            except ValueError:
                parsed = MIN_GRANTED_WORKERS
            effective = _clamp_workers(parsed)

    if task_count is not None:
        return min(effective, max(MIN_GRANTED_WORKERS, int(task_count)))
    return effective


def _select_mp_context() -> multiprocessing.context.BaseContext:
    """Choose ``fork`` on single-threaded Linux for COW sharing, ``spawn`` otherwise."""
    if sys.platform.startswith("linux") and threading.active_count() == 1:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _mapreduce_worker_init() -> None:
    """Pin child worker env to 1 worker so nested MapReduce pools never spawn."""
    os.environ[GRANTED_WORKERS_ENV] = str(MIN_GRANTED_WORKERS)


class BoundedMapPool:
    """Single flat lazy ``ProcessPoolExecutor`` with guaranteed zombie-free shutdown."""

    __slots__ = ("_effective_workers", "_has_task_bound", "_pool")

    def __init__(
        self,
        *,
        granted_workers: int | None = None,
        max_tasks: int | None = None,
    ) -> None:
        self._has_task_bound = max_tasks is not None
        self._effective_workers = resolve_granted_workers(
            granted_workers, task_count=max_tasks,
        )
        self._pool: ProcessPoolExecutor | None = None

    @property
    def effective_workers(self) -> int:
        return self._effective_workers

    def __enter__(self) -> BoundedMapPool:
        return self

    def _ensure_pool(self, batch_len: int) -> ProcessPoolExecutor:
        if self._pool is None:
            workers = (
                self._effective_workers
                if self._has_task_bound
                else min(self._effective_workers, max(MIN_GRANTED_WORKERS, int(batch_len)))
            )
            self._pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=_select_mp_context(),
                initializer=_mapreduce_worker_init,
            )
        return self._pool

    def map_ordered(self, fn: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
        """Map ``fn`` across ``items`` preserving exact input index order."""
        item_list = list(items)
        if not item_list:
            return []
        if self._effective_workers <= MIN_GRANTED_WORKERS or len(item_list) <= 1:
            return [fn(item) for item in item_list]
        pool = self._ensure_pool(len(item_list))
        futures = [pool.submit(fn, item) for item in item_list]
        return [fut.result() for fut in futures]

    def close(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def map_ordered(
    fn: Callable[[_T], _R],
    items: Sequence[_T],
    *,
    granted_workers: int | None = None,
) -> list[_R]:
    """Run one bounded, order-preserving Map pass with guaranteed pool cleanup."""
    item_list = list(items)
    if not item_list:
        return []
    with BoundedMapPool(granted_workers=granted_workers, max_tasks=len(item_list)) as pool:
        return pool.map_ordered(fn, item_list)


__all__ = [
    "GRANTED_WORKERS_ENV",
    "MAX_GRANTED_WORKERS",
    "MIN_GRANTED_WORKERS",
    "BoundedMapPool",
    "map_ordered",
    "resolve_granted_workers",
]
