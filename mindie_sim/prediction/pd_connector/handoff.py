"""Physical P/D transfer and handoff contracts.

[Role] Validate transfer regions, endpoint ownership and release dependencies.
[Invariants] Source readiness, destination grant and consumer release are distinct.
[Timeouts & Locks] Validation is bounded by the declared transfer and event records.
[Side-Effect Policy] Constructs handoff records and dependency lists in memory.
"""
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError

from ..communication import MessageFlow


def _non_negative_number(value, label):
    # Out-of-scope Wave S-Global caller: pd_connector/connector.py:17,36,42,46
    # (hop_startup_us/chunk_bytes); NEVER used by PDTransfer or PDHandoff.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"{label}_invalid:{value}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ModelingError(f"{label}_invalid:{value}")
    return result


def _strict_positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelingError(f"{label}_invalid:{value}")
    return value


@dataclass(frozen=True)
class PDTransfer:
    """One directed physical-rank shard inside a request handoff."""

    source_rank: int
    destination_rank: int
    payload_bytes: int
    source_stage: int = 0
    destination_stage: int = 0
    source_layers: tuple[int, ...] = ()
    destination_layers: tuple[int, ...] = ()
    component: str = "kv"
    source_version: int = 0
    destination_version: int = 0
    source_heads: tuple[int, int] = ()
    destination_heads: tuple[int, int] = ()
    source_token_span: tuple[int, int] = ()
    destination_token_span: tuple[int, int] = ()
    producer: str = ""
    output: str = ""
    #: Physical destination domain.  ``hbm`` is a decode NPU receive buffer;
    #: ``host`` is the shared Host DRAM copy made once by the SFA RD2H
    #: connector.  A host transfer still names the decode rank whose replica
    #: owns that host buffer, but it does not consume that rank's HBM footprint.
    destination_tier: str = "hbm"
    source_storage: str = ""
    destination_storage: str = ""
    #: Exact integer half-open byte ranges ``[start, end)`` inside source/destination storage.
    source_byte_range: tuple[int, int] = ()
    destination_byte_range: tuple[int, int] = ()
    source_scope: str = ""
    destination_scope: str = ""

    def __post_init__(self):
        if (isinstance(self.source_rank, bool)
                or not isinstance(self.source_rank, int)
                or isinstance(self.destination_rank, bool)
                or not isinstance(self.destination_rank, int)):
            raise ModelingError("pd_transfer_requires_integer_ranks")
        _strict_positive_int(self.payload_bytes, "pd_transfer_payload_requires_positive_int_bytes")
        for name in ("source_stage", "destination_stage"):
            if (isinstance(getattr(self, name), bool)
                    or not isinstance(getattr(self, name), int)
                    or getattr(self, name) < 0):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("source_layers", "destination_layers"):
            value = getattr(self, name)
            if (not isinstance(value, tuple)
                    or any(isinstance(layer, bool) or not isinstance(layer, int)
                           or layer < 0 for layer in value)):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        if not isinstance(self.component, str) or not self.component:
            raise ModelingError("pd_transfer_component_invalid")
        for name in ("source_version", "destination_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("source_heads", "destination_heads",
                     "source_token_span", "destination_token_span"):
            value = getattr(self, name)
            if value and (not isinstance(value, tuple) or len(value) != 2
                          or any(isinstance(item, bool) or not isinstance(item, int)
                                 or item < 0 for item in value)
                          or value[0] > value[1]):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("producer", "output"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        if self.destination_tier not in {"hbm", "host"}:
            raise ModelingError(
                f"pd_transfer_destination_tier_invalid:{self.destination_tier}")
        for name in ("source_storage", "destination_storage",
                     "source_scope", "destination_scope"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("source_byte_range", "destination_byte_range"):
            value = getattr(self, name)
            if not value:
                continue
            if (not isinstance(value, tuple) or len(value) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           or item < 0 for item in value)
                    or value[0] >= value[1]
                    or (value[1] - value[0]) != self.payload_bytes):
                raise ModelingError(f"pd_transfer_{name}_invalid")


@dataclass(frozen=True)
class PDHandoff:
    """Actual request/rank/stage/chunk/destination-receive binding.

    ``transfers`` own the directed physical shards.  ``receive_capacity_bytes``
    and ``receive_buffer_ids`` bind each destination rank to the decode pool's
    physical receive allocation; no global endpoint is invented.  ``memory_work``
    carries the prefill/decode stage HBM demands separately from wire payload.
    A transfer with ``destination_tier == "host"`` is the single SFA RD2H copy
    into shared Host DRAM: it is real wire/source work but not that rank's HBM
    receive footprint.  C (pool budget) and F (this call's footprint) therefore
    stay separate for both HBM and host domains.
    """

    request_id: str
    transfers: tuple[PDTransfer, ...]
    memory_work: tuple[tuple[str, float], ...] = ()
    receive_capacity_bytes: tuple[tuple[int, int], ...] = ()
    receive_buffer_ids: tuple[tuple[int, str], ...] = ()
    receive_pool_capacity_bytes: tuple[tuple[int, int], ...] = ()
    receive_host_bytes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if not self.request_id:
            raise ModelingError("pd_handoff_requires_request_id")
        if not self.transfers:
            return
        if any(not isinstance(item, PDTransfer) for item in self.transfers):
            raise ModelingError("pd_handoff_transfer_record_invalid")
        totals = defaultdict(int)
        host_totals = defaultdict(int)
        for item in self.transfers:
            if item.destination_tier == "host":
                host_totals[item.destination_rank] += item.payload_bytes
            else:
                totals[item.destination_rank] += item.payload_bytes
        capacities = dict(self.receive_capacity_bytes)
        if len(capacities) != len(self.receive_capacity_bytes):
            raise ModelingError("pd_handoff_duplicate_receive_capacity")
        for rank, value in capacities.items():
            if (isinstance(rank, bool) or not isinstance(rank, int)
                    or isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                raise ModelingError(
                    f"pd_handoff_receive_capacity_invalid:{rank}:{value}")
        buffers = dict(self.receive_buffer_ids)
        if len(buffers) != len(self.receive_buffer_ids):
            raise ModelingError("pd_handoff_duplicate_receive_buffer")
        for rank, value in buffers.items():
            if (isinstance(rank, bool) or not isinstance(rank, int)
                    or not isinstance(value, str) or not value):
                raise ModelingError(
                    f"pd_handoff_receive_buffer_invalid:{rank}:{value}")
        for rank, value in capacities.items():
            if rank not in totals:
                raise ModelingError(f"pd_handoff_capacity_without_destination:{rank}")
            if value < totals[rank]:
                raise ModelingError(
                    "pd_handoff_receive_capacity_below_physical_shard:"
                    f"{rank}:{value}<{totals[rank]}")
        pool_capacities = dict(self.receive_pool_capacity_bytes)
        if len(pool_capacities) != len(self.receive_pool_capacity_bytes):
            raise ModelingError("pd_handoff_duplicate_pool_capacity")
        for rank, value in pool_capacities.items():
            if rank not in capacities:
                raise ModelingError(
                    f"pd_handoff_pool_capacity_without_destination:{rank}")
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelingError(
                    f"pd_handoff_pool_capacity_invalid:{rank}:{value}")
            # C may legitimately be smaller than this call's F: that is a
            # real backpressure fact, not a record error.  The whole-message
            # submit refuses it with zero issue in submit_handoff().
        for key, value in self.memory_work:
            if not isinstance(key, str) or not key:
                raise ModelingError("pd_handoff_memory_resource_invalid")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ModelingError(
                    f"pd_handoff_memory_work_invalid:{key}:{value}")
        host_capacity = dict(self.receive_host_bytes)
        if len(host_capacity) != len(self.receive_host_bytes):
            raise ModelingError("pd_handoff_duplicate_host_capacity")
        for rank, value in host_capacity.items():
            if (isinstance(rank, bool) or not isinstance(rank, int)
                    or isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                raise ModelingError(
                    f"pd_handoff_host_capacity_invalid:{rank}:{value}")
            if rank not in host_totals:
                raise ModelingError(
                    f"pd_handoff_host_capacity_without_host_transfer:{rank}")
            if value < host_totals[rank]:
                raise ModelingError(
                    "pd_handoff_host_capacity_below_host_shard:"
                    f"{rank}:{value}<{host_totals[rank]}")

    @classmethod
    def build(cls, request_id, transfers, *, memory_work=(),
              receive_capacity_bytes=None, receive_buffer_ids=None,
              receive_pool_capacity_bytes=None, receive_host_bytes=None):
        transfers = tuple(transfers)
        totals = defaultdict(int)
        host_totals = defaultdict(int)
        for item in transfers:
            if item.destination_tier == "host":
                host_totals[item.destination_rank] += item.payload_bytes
            else:
                totals[item.destination_rank] += item.payload_bytes
        if receive_capacity_bytes is None:
            capacity = tuple(sorted(totals.items()))
        elif isinstance(receive_capacity_bytes, Mapping):
            capacity = tuple(sorted(
                (int(rank), _strict_positive_int(value, "pd_handoff_receive_capacity"))
                for rank, value in receive_capacity_bytes.items()))
        else:
            capacity = tuple(sorted(
                (int(rank), _strict_positive_int(value, "pd_handoff_receive_capacity"))
                for rank, value in receive_capacity_bytes))
        if receive_buffer_ids is None:
            buffers = tuple((rank, f"pd:{request_id}:rank{rank}")
                            for rank in sorted(totals))
        elif isinstance(receive_buffer_ids, Mapping):
            buffers = tuple(sorted(
                (int(rank), str(value)) for rank, value in receive_buffer_ids.items()))
        else:
            buffers = tuple(sorted(
                (int(rank), str(value)) for rank, value in receive_buffer_ids))
        memory = (tuple(memory_work.items()) if isinstance(memory_work, Mapping)
                  else tuple(memory_work))
        if receive_pool_capacity_bytes is None:
            buf_map = dict(buffers)
            grouped_cap = defaultdict(int)
            for rank, value in capacity:
                buf_id = buf_map.get(rank, f"{request_id}:rank{rank}")
                grouped_cap[buf_id] += value
            pool_capacity = tuple(
                (rank, grouped_cap[buf_map.get(rank, f"{request_id}:rank{rank}")])
                for rank, _value in capacity
            )
        elif isinstance(receive_pool_capacity_bytes, Mapping):
            pool_capacity = tuple(sorted(
                (int(rank), _strict_positive_int(value, "pd_handoff_pool_capacity"))
                for rank, value in receive_pool_capacity_bytes.items()))
        else:
            pool_capacity = tuple(sorted(
                (int(rank), _strict_positive_int(value, "pd_handoff_pool_capacity"))
                for rank, value in receive_pool_capacity_bytes))
        if receive_host_bytes is None:
            host = tuple(sorted(host_totals.items()))
        elif isinstance(receive_host_bytes, Mapping):
            host = tuple(sorted(
                (int(rank), _strict_positive_int(value, "pd_handoff_host_capacity"))
                for rank, value in receive_host_bytes.items()))
        else:
            host = tuple(sorted(
                (int(rank), _strict_positive_int(value, "pd_handoff_host_capacity"))
                for rank, value in receive_host_bytes))
        return cls(str(request_id), transfers, memory, capacity, buffers,
                   pool_capacity, host)

    @property
    def ranks(self):
        return tuple(sorted({item.source_rank for item in self.transfers}
                            | {item.destination_rank for item in self.transfers}))

    def flows(self):
        return tuple(MessageFlow(item.source_rank, item.destination_rank,
                                 float(item.payload_bytes))
                     for item in self.transfers)

    def hbm_transfers(self):
        return tuple(item for item in self.transfers
                     if item.destination_tier == "hbm")

    def host_transfers(self):
        return tuple(item for item in self.transfers
                     if item.destination_tier == "host")

    def capacity_for(self, rank):
        for destination, value in self.receive_capacity_bytes:
            if destination == rank:
                return float(value)
        return None

    def resolved_receive_pools(self):
        """Return per-destination bindings and per-pool aggregate footprints."""
        capacities = dict(self.receive_capacity_bytes)
        pool_capacities = dict(self.receive_pool_capacity_bytes)
        buffers = dict(self.receive_buffer_ids)
        per_rank = {}
        by_buffer = {}
        for destination, footprint in sorted(capacities.items()):
            buffer_id = buffers.get(
                destination, f"{self.request_id}:rank{destination}")
            pool_cap = pool_capacities.get(destination, footprint)
            per_rank[destination] = (buffer_id, int(footprint), int(pool_cap))
            entry = by_buffer.get(buffer_id)
            if entry is None:
                by_buffer[buffer_id] = [int(footprint), int(pool_cap), [destination]]
            else:
                entry[0] += int(footprint)
                entry[2].append(destination)
        return per_rank, {
            buf_id: (vals[0], vals[1], tuple(vals[2]))
            for buf_id, vals in by_buffer.items()
        }


def _normalize_releases(release_events):
    """Return destination release events keyed by transport endpoint rank."""
    if release_events is None:
        return None
    if isinstance(release_events, Mapping):
        items = release_events.items()
    else:
        items = ((1, release_events),)
    result = {}
    for rank, value in items:
        result[int(rank)] = (tuple(value) if isinstance(value, (tuple, list))
                             else (value,))
    return result
