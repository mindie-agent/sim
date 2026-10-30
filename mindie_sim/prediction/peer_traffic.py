"""Explicit directed peer payloads for collective transport.

[Role] Parse immutable forward/transpose peer traffic declarations used by
  topology accounting and causal PairwiseGroup transport.
[Invariants] The index is physical-group-relative rank order. Uniform peers
  retain O(N) source totals and split only over off-diagonal peers. Matrix
  declarations retain only the supplied explicit cells. Transposed reads the
  reversed original cell; it never swaps stored row totals. All payloads are
  finite and nonnegative, and matrix diagonals carry no network traffic.
[Timeouts & Locks] Pure bounded arithmetic; no I/O, timers or locks.
[Side-Effect Policy] Immutable values; no filesystem, engine or global mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


_DIRECTIONS = frozenset({"forward", "transpose"})
_KINDS = frozenset({"uniform_peers", "matrix"})


def _number(value, reason):
    if isinstance(value, bool):
        raise ValueError(reason)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(reason) from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(reason)
    return result


@dataclass(frozen=True)
class PeerTraffic:
    """One physical group's declared directed payloads."""

    kind: str
    size: int
    values: tuple[float, ...]
    direction: str = "forward"

    @classmethod
    def from_mapping(cls, raw, size):
        from mindie_sim.schema.common import ModelingError

        if (not isinstance(size, int) or isinstance(size, bool) or size < 1):
            raise ModelingError(f"peer_traffic_group_size_invalid:{size}")
        if not isinstance(raw, Mapping):
            raise ModelingError("peer_traffic_must_be_mapping")
        allowed = {"kind", "source_bytes", "payload_bytes", "direction"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ModelingError(f"peer_traffic_unknown_field:{unknown[0]}")
        kind = raw.get("kind")
        if kind not in _KINDS:
            raise ModelingError(f"peer_traffic_kind_unsupported:{kind}")
        direction = raw.get("direction", "forward")
        if direction not in _DIRECTIONS:
            raise ModelingError(
                f"peer_traffic_direction_unsupported:{direction}")

        def values(items, label):
            if not isinstance(items, (list, tuple)) or len(items) != size:
                raise ModelingError(f"peer_traffic_{label}_shape_invalid:{size}")
            return tuple(_number(value, f"peer_traffic_{label}_value_invalid:{value}")
                         for value in items)

        try:
            if kind == "uniform_peers":
                if "payload_bytes" in raw:
                    raise ModelingError(
                        "peer_traffic_uniform_peers_forbids_payload_bytes")
                if "source_bytes" not in raw:
                    raise ModelingError("peer_traffic_source_bytes_required")
                source = values(raw["source_bytes"], "source_bytes")
                if size == 1 and any(source):
                    raise ModelingError(
                        "peer_traffic_singleton_requires_zero_source_bytes")
                return cls(kind, size, source, direction)

            if "source_bytes" in raw:
                raise ModelingError("peer_traffic_matrix_forbids_source_bytes")
            rows = raw.get("payload_bytes")
            if not isinstance(rows, (list, tuple)) or len(rows) != size:
                raise ModelingError(f"peer_traffic_matrix_shape_invalid:{size}")
            cells = []
            for source, row in enumerate(rows):
                if not isinstance(row, (list, tuple)) or len(row) != size:
                    raise ModelingError(
                        f"peer_traffic_matrix_shape_invalid:{size}")
                for destination, value in enumerate(row):
                    cell = _number(
                        value,
                        f"peer_traffic_payload_value_invalid:{value}")
                    if source == destination and cell:
                        raise ModelingError(
                            f"peer_traffic_diagonal_must_be_zero:{source}")
                    cells.append(cell)
            return cls(kind, size, tuple(cells), direction)
        except ModelingError:
            raise
        except ValueError as error:
            raise ModelingError(str(error)) from None

    def _index(self, index, label):
        from mindie_sim.schema.common import ModelingError

        if (not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < self.size):
            raise ModelingError(f"peer_traffic_{label}_index_invalid:{index}")
        return index

    def payload(self, source_index, destination_index):
        """Useful bytes from one physical-group-relative source to destination."""
        source = self._index(source_index, "source")
        destination = self._index(destination_index, "destination")
        if source == destination or self.size == 1:
            return 0.0
        forward = self.direction == "forward"
        if self.kind == "uniform_peers":
            owner = source if forward else destination
            return self.values[owner] / (self.size - 1)
        original_source, original_destination = (
            (source, destination) if forward else (destination, source))
        return self.values[original_source * self.size + original_destination]

    def row_total(self, source_index):
        """Total off-diagonal useful bytes sent by one forward/transpose source."""
        source = self._index(source_index, "source")
        if self.size == 1:
            return 0.0
        if self.kind == "uniform_peers":
            if self.direction == "forward":
                return self.values[source]
            return (math.fsum(value for index, value in enumerate(self.values)
                              if index != source) / (self.size - 1))
        if self.direction == "forward":
            return math.fsum(
                self.values[source * self.size + destination]
                for destination in range(self.size) if destination != source)
        return math.fsum(
            self.values[original_source * self.size + source]
            for original_source in range(self.size)
            if original_source != source)

    def transposed(self):
        """Read the reversed original cell without materializing a dense uniform."""
        return PeerTraffic(
            self.kind, self.size, self.values,
            "transpose" if self.direction == "forward" else "forward",
        )
