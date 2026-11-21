"""Shared sparse indexer query blocks and key tile partitions.

[Role] Map batched query rows and key tiles to bounded work classes.
[Invariants] Query blocks stay within requests. Each S2 tile has one owner.
[Timeouts & Locks] Constant-size arithmetic with no locks.
[Side-Effect Policy] Pure geometry with no I/O.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ...program import KernelProgram, PipelineProgram, Program, Stage


@dataclass(frozen=True, slots=True)
class _DisjointOwnerProgram:
    """Execute the stages with physical IDs already bound by the planner."""

    placed: Program

    def start(self, engine, name, ready, *, owner=""):
        return self.placed.start(engine, name, ready, owner=owner)


def request_program(builders, waves, context):
    def wave(index):
        first, last = waves[index]
        return tuple(Stage(f"request{j}", program=builders[j](context)) for j in range(first, last))

    placed = PipelineProgram(len(waves), 1, wave) if waves else KernelProgram(())
    return _DisjointOwnerProgram(placed) if context.stream_owners and context.stream_owners.isolation == "disjoint" else placed


def sort_ops(items: int) -> int:
    """Pair compare/exchange work for a comparison-based sort."""
    return items * (max(1, items) - 1).bit_length()


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def align(n: int, granule: int) -> int:
    return ceil_div(n, granule) * granule


@dataclass(frozen=True, slots=True)
class S1Partition:
    """Query blocks over the balanced request lengths implied by aggregate S1."""

    rows: int
    batches: int
    tile_size: int

    def _request_groups(self) -> Iterator[tuple[int, int]]:
        rows, extra = divmod(self.rows, self.batches)
        if extra:
            yield rows + 1, extra
        if rows:
            yield rows, self.batches - extra

    @property
    def count(self) -> int:
        return sum(ceil_div(rows, self.tile_size) * requests for rows, requests in self._request_groups())

    def live_rows(self, block: int) -> int:
        for rows, requests in self._request_groups():
            tiles = ceil_div(rows, self.tile_size)
            if block < tiles * requests:
                return min(self.tile_size, rows - block % tiles * self.tile_size)
            block -= tiles * requests
        raise IndexError("S1 block outside query partition")

    def variants(self, l2_reuse: bool) -> Iterator[tuple[int, int, bool]]:
        if l2_reuse:
            yield self.live_rows(0), 1, False
        skip_first = l2_reuse
        for rows, requests in self._request_groups():
            full, tail = divmod(rows, self.tile_size)
            for live, count in ((self.tile_size, full * requests), (tail, bool(tail) * requests)):
                if count:
                    if count > skip_first:
                        yield live, count - skip_first, l2_reuse
                    skip_first = False


@dataclass(frozen=True, slots=True)
class S2Partition:
    """Balanced contiguous intervals of whole S2 tiles and one final tail."""

    rows: int
    tile_size: int
    splits: int

    @property
    def tiles(self) -> int:
        return ceil_div(self.rows, self.tile_size)

    @property
    def owners(self) -> int:
        return min(self.splits, self.tiles)

    def owner_rows(self, split: int) -> int:
        base, extra = divmod(self.tiles, self.owners)
        count = base + (split < extra)
        return min(count * self.tile_size, self.rows - self.owner_start(split))

    def owner_start(self, split: int) -> int:
        base, extra = divmod(self.tiles, self.owners)
        return (split * base + min(split, extra)) * self.tile_size

    def variants(self) -> Iterator[tuple[int, int, bool]]:
        """Distinct owner row counts with multiplicities, independent of S2."""
        if not self.owners:
            return
        base, extra = divmod(self.tiles, self.owners)
        if extra:
            yield (base + 1) * self.tile_size, extra, False
        if self.owners - extra > 1:
            yield base * self.tile_size, self.owners - extra - 1, False
        yield self.owner_rows(self.owners - 1), 1, True
