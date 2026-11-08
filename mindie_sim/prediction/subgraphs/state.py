"""Typed state ports, cross-layer transitions and ownership invariants.

[Role]
  Represent the sufficient dynamic state ``Z`` explicitly: SharedKV ownership
  and readiness, compression residuals, MoE routing counts, residual slots and
  draft acceptance/commit progress.  Transitions are pure functions over
  immutable views so composition can prove ordering instead of relying on a
  layer id convention.

[Invariants]
  * A KV consumer can only read a representation produced earlier in the same
    iteration or by a previous iteration; reading an absent owner raises
    ``state_port_not_ready``.
  * An owner stores once; every consumer pays its own read and a cross-stage
    path keeps its transfer bytes.
  * Compression advances by ``floor(s/r) - floor((s-q)/r)`` per request.
    Each request/owner pair keeps its own consumed/remainder record; chunked
    advances telescope to the single final floor and never merge request
    streams into one concatenated counter.
  * MoE routing conserves ``sum_e n_e == k * T_valid`` over non-negative
    integers.  Expected active-expert moments remain fractional and separate
    from integer realisations.
  * Acceptance changes commit/cache progress only; draft and target-verify
    work are never multiplied by acceptance.

[Timeouts & Locks]
  Pure bounded arithmetic and validation.  No waits, I/O, locks or devices.

[Side-Effect Policy]
  Returns new immutable state views; no global mutation or persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping

from mindie_sim.schema.common import ModelingError


def _count(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(reason)
    return value


@dataclass(frozen=True)
class StateTransition:
    """One declared port transition in a subgraph."""

    name: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    owner: str = "self"
    readiness: str = "producer_complete"
    scope: str = "layer"

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelingError("state_transition_name_invalid")
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "writes", tuple(self.writes))
        if not set(self.reads) | set(self.writes):
            raise ModelingError(f"state_transition_empty:{self.name}")
        if set(self.reads) & set(self.writes):
            raise ModelingError(f"state_transition_overlap:{self.name}")
        if self.scope not in {"operator", "layer", "stage", "request", "cohort"}:
            raise ModelingError(f"state_transition_scope_invalid:{self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "reads": list(self.reads),
                "writes": list(self.writes), "owner": self.owner,
                "readiness": self.readiness, "scope": self.scope}


@dataclass(frozen=True)
class CompressionState:
    """Telescoping K1 compression state for one cache owner."""

    ratio: int
    consumed: int = 0
    rows: int = 0
    remainder: int = 0

    def __post_init__(self) -> None:
        _count(self.ratio, "compression_ratio_invalid")
        if self.ratio < 1:
            raise ModelingError("compression_ratio_invalid")
        _count(self.consumed, "compression_consumed_invalid")
        _count(self.rows, "compression_rows_invalid")
        if not 0 <= self.remainder < self.ratio:
            raise ModelingError(f"compression_remainder_invalid:{self.remainder}")
        if self.consumed // self.ratio != self.rows:
            raise ModelingError(
                f"compression_rows_inconsistent:{self.rows}!={self.consumed // self.ratio}")
        if self.consumed % self.ratio != self.remainder:
            raise ModelingError(
                f"compression_remainder_inconsistent:{self.remainder}")

    @classmethod
    def start(cls, ratio: int, consumed: int = 0) -> "CompressionState":
        if ratio < 1:
            raise ModelingError("compression_ratio_invalid")
        consumed = _count(consumed, "compression_consumed_invalid")
        return cls(ratio, consumed, consumed // ratio, consumed % ratio)

    def new_rows(self, query: int) -> int:
        """Exact new complete groups for one chunk of ``query`` tokens."""
        query = _count(query, "compression_query_invalid")
        end = self.consumed + query
        return end // self.ratio - self.consumed // self.ratio

    def advance(self, query: int) -> "CompressionState":
        query = _count(query, "compression_query_invalid")
        end = self.consumed + query
        return CompressionState(self.ratio, end, end // self.ratio, end % self.ratio)

    def advance_chunks(self, chunks: Iterable[int]) -> tuple["CompressionState", int]:
        state = self
        total = 0
        for query in chunks:
            total += state.new_rows(query)
            state = state.advance(query)
        return state, total

    @staticmethod
    def telescoped_new_rows(ratio: int, start: int, query: int) -> int:
        """Closed form ``floor(s/r) - floor((s-q)/r)`` with s = start + q."""
        if ratio < 1:
            raise ModelingError("compression_ratio_invalid")
        start = _count(start, "compression_start_invalid")
        query = _count(query, "compression_query_invalid")
        end = start + query
        return end // ratio - start // ratio

    def to_dict(self) -> dict[str, Any]:
        return {"ratio": self.ratio, "consumed": self.consumed,
                "rows": self.rows, "remainder": self.remainder}


@dataclass(frozen=True)
class MoeRoutingState:
    """Integer per-expert routed rows with conservation validation."""

    counts: tuple[int, ...]
    top_k: int
    valid_tokens: int
    local_expert_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        counts = tuple(self.counts)
        if not counts:
            raise ModelingError("moe_routing_counts_empty")
        for value in counts:
            _count(value, "moe_routing_count_invalid")
        if self.top_k < 1:
            raise ModelingError("moe_routing_top_k_invalid")
        _count(self.valid_tokens, "moe_routing_valid_tokens_invalid")
        if sum(counts) != self.top_k * self.valid_tokens:
            raise ModelingError(
                f"moe_routing_conservation_failed:{sum(counts)}!="
                f"{self.top_k * self.valid_tokens}")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "local_expert_ids", tuple(self.local_expert_ids))

    @property
    def experts(self) -> int:
        return len(self.counts)

    @property
    def active_experts(self) -> int:
        return sum(1 for value in self.counts if value > 0)

    @property
    def total_rows(self) -> int:
        return sum(self.counts)

    @property
    def skew(self) -> float:
        if not self.total_rows:
            return 0.0
        peak = max(self.counts)
        mean = self.total_rows / len(self.counts)
        return peak / mean if mean else 0.0

    def waves(self, row_tile: int, cores: int) -> int:
        if row_tile < 1 or cores < 1:
            raise ModelingError("moe_wave_geometry_invalid")
        tiles = sum(math.ceil(value / row_tile) for value in self.counts if value)
        return math.ceil(tiles / cores) if tiles else 0

    def padded_rows(self, row_tile: int) -> int:
        if row_tile < 1:
            raise ModelingError("moe_row_tile_invalid")
        return sum(math.ceil(value / row_tile) * row_tile
                   for value in self.counts if value)

    def with_counts(self, counts: Iterable[int], *, valid_tokens: int | None = None) -> "MoeRoutingState":
        return MoeRoutingState(tuple(counts), self.top_k,
                               self.valid_tokens if valid_tokens is None else valid_tokens,
                               self.local_expert_ids)

    @staticmethod
    def expected_active_experts(experts: int, top_k: int, tokens: int) -> float:
        """Independent-inclusion expectation ``E[E_a]`` (fractional)."""
        if experts < 1 or top_k < 0 or tokens < 0:
            raise ModelingError("moe_expectation_geometry_invalid")
        if tokens == 0 or top_k == 0:
            return 0.0
        if top_k >= experts:
            return float(experts)
        return experts * (1.0 - (1.0 - top_k / experts) ** tokens)

    def expected_padded_rows(self, row_tile: int) -> float:
        """Expected aligned rows; explicitly fractional, not an integer shape."""
        if row_tile < 1:
            raise ModelingError("moe_row_tile_invalid")
        if not self.total_rows:
            return 0.0
        mean = self.total_rows / max(1, self.experts)
        return self.experts * math.ceil(mean / row_tile) * row_tile

    def to_dict(self) -> dict[str, Any]:
        return {"counts": list(self.counts), "top_k": self.top_k,
                "valid_tokens": self.valid_tokens,
                "active_experts": self.active_experts,
                "total_rows": self.total_rows, "skew": self.skew}


@dataclass(frozen=True)
class AcceptanceState:
    """Draft acceptance/commit progress; never a work discount."""

    accepted: int = 0
    committed: int = 0
    rejected: int = 0
    commits: int = 0

    def __post_init__(self) -> None:
        for name in ("accepted", "committed", "rejected", "commits"):
            _count(getattr(self, name), f"acceptance_{name}_invalid")

    def commit(self, accepted: int, *, verified: int = 1) -> "AcceptanceState":
        accepted = _count(accepted, "acceptance_accepted_invalid")
        verified = _count(verified, "acceptance_verified_invalid")
        if accepted > verified:
            raise ModelingError(f"acceptance_exceeds_verified:{accepted}>{verified}")
        return AcceptanceState(self.accepted + accepted,
                               self.committed + accepted,
                               self.rejected + verified - accepted,
                               self.commits + 1)

    @property
    def progress(self) -> int:
        return self.committed

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "committed": self.committed,
                "rejected": self.rejected, "commits": self.commits}


@dataclass(frozen=True)
class ResidualSlotState:
    """Bounded resident attention-residual slot stack.

    ``push`` is the raw rolling-slot operation.  The model walk uses
    ``push_block`` so a completed block becomes a source only after its last
    member layer; a layer's R1 site then sees ``sources(layer)`` completed
    blocks plus the current residual.
    """

    block_size: int
    retained: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ModelingError("residual_block_size_invalid")
        retained = tuple(self.retained)
        if any(not isinstance(item, int) or item < 0 for item in retained):
            raise ModelingError("residual_slot_id_invalid")
        object.__setattr__(self, "retained", retained)

    def push(self, layer: int) -> "ResidualSlotState":
        if not isinstance(layer, int) or layer < 0:
            raise ModelingError("residual_slot_id_invalid")
        retained = (self.retained + (layer,))[-self.block_size:]
        return ResidualSlotState(self.block_size, retained)

    def push_block(self, layer: int) -> "ResidualSlotState":
        """Append a completed block id after its last member layer."""
        if not isinstance(layer, int) or layer < 0:
            raise ModelingError("residual_slot_id_invalid")
        if (layer + 1) % self.block_size:
            return self
        block = layer // self.block_size
        retained = self.retained if block in self.retained else self.retained + (block,)
        return ResidualSlotState(self.block_size, retained)

    def sources(self, layer: int) -> int:
        """Sources visible at ``layer``: completed blocks plus the current."""
        if not isinstance(layer, int) or layer < 0:
            raise ModelingError("residual_slot_id_invalid")
        return 1 + layer // self.block_size

    @property
    def slots(self) -> int:
        return len(self.retained)

    @property
    def full(self) -> bool:
        return len(self.retained) >= self.block_size


@dataclass(frozen=True)
class SharedKvContract:
    """Static owner/consumer map for SharedKV.

    The map belongs to ``Theta``; the produced representation and readiness
    belong to ``Z``.  A consumer never regenerates the owner's K/V.
    """

    owner_layer: int
    consumers: tuple[int, ...]
    row_bytes: int
    cross_stage: bool = False
    transfer_bytes: int = 0
    store_count: int = 1

    def __post_init__(self) -> None:
        _count(self.owner_layer, "shared_kv_owner_invalid")
        consumers = tuple(self.consumers)
        if not consumers:
            raise ModelingError("shared_kv_consumers_empty")
        for layer in consumers:
            _count(layer, "shared_kv_consumer_invalid")
            if layer <= self.owner_layer:
                raise ModelingError(
                    f"shared_kv_consumer_before_owner:{layer}<={self.owner_layer}")
        object.__setattr__(self, "consumers", consumers)
        if self.row_bytes < 1:
            raise ModelingError("shared_kv_row_bytes_invalid")
        _count(self.transfer_bytes, "shared_kv_transfer_invalid")
        if self.store_count < 1:
            raise ModelingError("shared_kv_store_count_invalid")

    def store_bytes(self, rows: int = 1) -> int:
        _count(rows, "shared_kv_rows_invalid")
        return rows * self.row_bytes * self.store_count

    def read_bytes(self, rows: int) -> int:
        _count(rows, "shared_kv_rows_invalid")
        return rows * self.row_bytes

    def transfer(self) -> int:
        return self.transfer_bytes if self.cross_stage else 0

    def store(self, state: "StateView", *, rows: int = 1,
              representation: str = "kv_cache") -> tuple["StateView", int]:
        new_state = state.mark_kv(self.owner_layer, representation)
        return new_state, self.store_bytes(rows)

    def read(self, state: "StateView", layer: int, *, rows: int) -> tuple["StateView", int]:
        if layer not in self.consumers:
            raise ModelingError(f"shared_kv_unknown_consumer:{layer}")
        state.require_kv(self.owner_layer)
        return state, self.read_bytes(rows)

    def check_order(self, state: "StateView") -> None:
        state.require_kv(self.owner_layer)

    def to_dict(self) -> dict[str, Any]:
        return {"owner_layer": self.owner_layer,
                "consumers": list(self.consumers),
                "row_bytes": self.row_bytes,
                "cross_stage": self.cross_stage,
                "transfer_bytes": self.transfer(),
                "store_count": self.store_count}


@dataclass(frozen=True)
class StateView:
    """Immutable sufficient state for one iteration slice."""

    iteration: int = 0
    kv_ready: tuple[tuple[int, str], ...] = ()
    index_ready: tuple[tuple[int, str], ...] = ()
    selection_ready: tuple[tuple[int, str], ...] = ()
    cross_stage_ready: tuple[tuple[tuple[int, int], str], ...] = ()
    compression: tuple[tuple[int, CompressionState], ...] = ()
    compression_requests: tuple[tuple[tuple[int, Any], CompressionState], ...] = ()
    routing: tuple[tuple[int, MoeRoutingState], ...] = ()
    residual: ResidualSlotState = field(default_factory=lambda: ResidualSlotState(1))
    acceptance: AcceptanceState = field(default_factory=AcceptanceState)
    accepted_tokens: tuple[int, ...] = ()
    peer_bytes: tuple[tuple[int, int], ...] = ()
    candidate_blocks: tuple[tuple[int, int], ...] = ()
    env_extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        _count(self.iteration, "state_iteration_invalid")
        for name in ("kv_ready", "index_ready", "selection_ready", "cross_stage_ready",
                     "compression", "compression_requests", "routing", "accepted_tokens",
                     "peer_bytes", "candidate_blocks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name, value in (("kv_ready", self.kv_ready), ("index_ready", self.index_ready),
                            ("selection_ready", self.selection_ready),
                            ("cross_stage_ready", self.cross_stage_ready),
                            ("compression", self.compression),
                            ("compression_requests", self.compression_requests),
                            ("routing", self.routing)):
            keys = [key for key, _ in value]
            if len(keys) != len(set(keys)):
                raise ModelingError(f"state_{name}_duplicate_key")
        for key, _ in self.cross_stage_ready:
            if (not isinstance(key, tuple) or len(key) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                           for item in key)):
                raise ModelingError(f"state_cross_stage_key_invalid:{key}")
        for key, record in self.compression:
            if not isinstance(record, CompressionState):
                raise ModelingError(f"state_compression_record_invalid:{key}")
        for key, record in self.compression_requests:
            if (not isinstance(key, tuple) or len(key) != 2
                    or not isinstance(key[0], int) or key[0] < 0
                    or not isinstance(key[1], (int, str)) or key[1] == ""):
                raise ModelingError(f"state_compression_request_key_invalid:{key}")
            if not isinstance(record, CompressionState):
                raise ModelingError(f"state_compression_record_invalid:{key}")
            if isinstance(key[1], int) and key[1] < 0:
                raise ModelingError(f"state_compression_request_key_invalid:{key}")
        for key, record in self.routing:
            if not isinstance(record, MoeRoutingState):
                raise ModelingError(f"state_routing_record_invalid:{key}")

    # -- KV readiness -----------------------------------------------------
    def is_kv_ready(self, layer: int) -> bool:
        return any(key == layer for key, _ in self.kv_ready)

    def mark_kv(self, layer: int, representation: str = "kv_cache") -> "StateView":
        _count(layer, "state_kv_layer_invalid")
        values = dict(self.kv_ready)
        values[layer] = representation
        return replace(self, kv_ready=tuple(sorted(values.items())))

    def require_kv(self, layer: int) -> str:
        for key, representation in self.kv_ready:
            if key == layer:
                return representation
        raise ModelingError(f"state_port_not_ready:kv_cache:layer={layer}")

    def mark_index(self, layer: int, representation: str = "index") -> "StateView":
        _count(layer, "state_index_layer_invalid")
        values = dict(self.index_ready)
        values[layer] = representation
        return replace(self, index_ready=tuple(sorted(values.items())))

    def require_index(self, layer: int) -> str:
        for key, representation in self.index_ready:
            if key == layer:
                return representation
        raise ModelingError(f"state_port_not_ready:index_cache:layer={layer}")

    def mark_selection(self, layer: int, representation: str = "selection") -> "StateView":
        _count(layer, "state_selection_layer_invalid")
        values = dict(self.selection_ready)
        values[layer] = representation
        return replace(self, selection_ready=tuple(sorted(values.items())))

    def require_selection(self, layer: int) -> str:
        for key, representation in self.selection_ready:
            if key == layer:
                return representation
        raise ModelingError(f"state_port_not_ready:index_selection:layer={layer}")

    def mark_cross_stage(self, owner: int, stage: int,
                         representation: str = "kv_cache") -> "StateView":
        _count(owner, "state_cross_stage_owner_invalid")
        _count(stage, "state_cross_stage_stage_invalid")
        values = dict(self.cross_stage_ready)
        values[(owner, stage)] = representation
        return replace(self, cross_stage_ready=tuple(sorted(values.items())))

    def require_cross_stage(self, owner: int, stage: int) -> str:
        for key, representation in self.cross_stage_ready:
            if key == (owner, stage):
                return representation
        raise ModelingError(
            f"state_port_not_ready:cross_stage_kv:owner={owner}:stage={stage}")

    # -- compression ------------------------------------------------------
    def compression_state(self, owner: int, ratio: int) -> CompressionState:
        for key, record in self.compression:
            if key == owner:
                if record.ratio != ratio:
                    raise ModelingError(f"state_compression_ratio_mismatch:{owner}")
                return record
        return CompressionState.start(ratio)

    def compression_request_state(self, owner: int, request: int,
                                  ratio: int) -> CompressionState:
        for key, record in self.compression_requests:
            if key == (owner, request):
                if record.ratio != ratio:
                    raise ModelingError(f"state_compression_ratio_mismatch:{key}")
                return record
        return CompressionState.start(ratio)

    @staticmethod
    def _compression_requests(requests: Iterable[Any]
                              ) -> tuple[tuple[int, int, int], ...]:
        """Normalise ``(context, q_len)`` / ``RequestGeometry`` records.

        A three-tuple form ``(request_key, context, q_len)`` lets a caller keep
        stable request identity across scheduler steps; the default key is the
        position in the supplied vector.
        """
        if hasattr(requests, "requests"):
            requests = requests.requests
        elif hasattr(requests, "q_len"):
            requests = (requests,)
        normalised = []
        for index, item in enumerate(requests):
            key: Any = index
            if hasattr(item, "request_id"):
                key = item.request_id
            if hasattr(item, "context") and hasattr(item, "q_len"):
                context, q_len = int(item.context), int(item.q_len)
            else:
                parts = tuple(item)
                if len(parts) == 2:
                    context, q_len = int(parts[0]), int(parts[1])
                elif len(parts) == 3:
                    key, context, q_len = parts[0], int(parts[1]), int(parts[2])
                else:
                    raise ModelingError("state_compression_request_record_invalid")
            if (not isinstance(key, (int, str)) or key == ""
                    or (isinstance(key, int) and key < 0)
                    or context < 0 or q_len < 0):
                raise ModelingError("state_compression_request_record_invalid")
            normalised.append((key, context, q_len))
        return tuple(normalised)

    def update_compression(self, owner: int, ratio: int,
                           requests: Iterable[Any]) -> tuple["StateView", int]:
        """Advance one compression owner per request and sum completed rows.

        Every request keeps its own consumed/remainder record.  A request's
        first observation starts from its declared context; later observations
        continue that record while the context advances with the sequence.
        """
        _count(owner, "state_compression_owner_invalid")
        if ratio < 1:
            raise ModelingError("compression_ratio_invalid")
        values = dict(self.compression_requests)
        produced = 0
        for key, context, q_len in self._compression_requests(requests):
            current = values.get((owner, key))
            # A zero-context observation is a new sequence even when the
            # scheduler reuses a request id/slot; discard stale progress.
            if current is None or (context == 0 and current.consumed > 0):
                current = CompressionState.start(ratio, context)
            elif context > current.consumed or context < current.consumed:
                # The caller skipped/repositioned a chunk outside this owner;
                # resynchronise instead of keeping a stale counter.
                current = CompressionState.start(ratio, context)
            produced += current.new_rows(q_len)
            values[(owner, key)] = current.advance(q_len)
        ordered = sorted(values.items(), key=lambda item: (item[0][0], str(item[0][1])))
        return replace(self, compression_requests=tuple(ordered)), produced

    def retire_compression(self, request_key: Any) -> "StateView":
        """Drop completed-request compression records by stable identity."""
        values = tuple(item for item in self.compression_requests
                       if item[0][1] != request_key)
        return replace(self, compression_requests=values)

    def advance_compression(self, owner: int, ratio: int, query: int) -> tuple["StateView", int]:
        current = self.compression_state(owner, ratio)
        new_rows = current.new_rows(query)
        values = dict(self.compression)
        values[owner] = current.advance(query)
        return replace(self, compression=tuple(sorted(values.items()))), new_rows

    # -- routing ----------------------------------------------------------
    def routing_state(self, layer: int) -> MoeRoutingState:
        for key, record in self.routing:
            if key == layer:
                return record
        raise ModelingError(f"state_port_not_ready:expert_rows:layer={layer}")

    def with_routing(self, layer: int, state: MoeRoutingState) -> "StateView":
        values = dict(self.routing)
        values[layer] = state
        return replace(self, routing=tuple(sorted(values.items())))

    # -- residual/acceptance ---------------------------------------------
    def push_residual(self, layer: int) -> "StateView":
        return replace(self, residual=self.residual.push_block(layer))

    def commit(self, accepted: int, *, verified: int = 1) -> "StateView":
        # Keep only the current step's accepted count: acceptance progress is
        # scalar state, not an unbounded verification history.
        return replace(self, acceptance=self.acceptance.commit(accepted, verified=verified),
                       accepted_tokens=(accepted,) if accepted else ())

    # -- traffic ----------------------------------------------------------
    def add_peer_bytes(self, rank: int, amount: int) -> "StateView":
        _count(rank, "state_peer_rank_invalid")
        _count(amount, "state_peer_bytes_invalid")
        values = dict(self.peer_bytes)
        values[rank] = values.get(rank, 0) + amount
        return replace(self, peer_bytes=tuple(sorted(values.items())))

    def advance_iteration(self) -> "StateView":
        return replace(self, iteration=self.iteration + 1)

    def with_candidates(self, layer: int, blocks: int) -> "StateView":
        values = dict(self.candidate_blocks)
        values[layer] = _count(blocks, "state_candidate_blocks_invalid")
        return replace(self, candidate_blocks=tuple(sorted(values.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "kv_ready": {str(key): value for key, value in self.kv_ready},
            "index_ready": {str(key): value for key, value in self.index_ready},
            "selection_ready": {str(key): value for key, value in self.selection_ready},
            "cross_stage_ready": {f"{key[0]}:{key[1]}": value
                                  for key, value in self.cross_stage_ready},
            "compression": {str(key): value.to_dict() for key, value in self.compression},
            "compression_requests": {
                f"{key[0]}:{key[1]}": value.to_dict()
                for key, value in self.compression_requests},
            "routing": {str(key): value.to_dict() for key, value in self.routing},
            "residual": {"block_size": self.residual.block_size,
                         "retained": list(self.residual.retained)},
            "acceptance": self.acceptance.to_dict(),
            "accepted_tokens": list(self.accepted_tokens),
            "peer_bytes": dict(self.peer_bytes),
            "candidate_blocks": dict(self.candidate_blocks),
        }


def apply_transition(state: StateView, transition: StateTransition) -> StateView:
    """Apply a declared port transition; readiness is part of the contract."""
    if transition.scope == "stage" and transition.readiness == "producer_complete":
        for port_name in transition.reads:
            if port_name.startswith("kv:"):
                layer = int(port_name.split(":", 1)[1])
                state.require_kv(layer)
    if transition.name.startswith("kv_store"):
        parts = transition.name.split(":")
        return state.mark_kv(int(parts[1]) if len(parts) > 1 else 0)
    if transition.name.startswith("index_store"):
        parts = transition.name.split(":")
        return state.mark_index(int(parts[1]) if len(parts) > 1 else 0)
    if transition.name.startswith("compress"):
        parts = transition.name.split(":")
        if len(parts) == 4:
            return state.advance_compression(int(parts[1]), int(parts[2]), int(parts[3]))[0]
    if transition.name.startswith("residual_push"):
        return state.push_residual(int(transition.name.split(":")[1]))
    if transition.name.startswith("accept"):
        parts = transition.name.split(":")
        if len(parts) >= 3:
            return state.commit(int(parts[1]), verified=int(parts[2]))
    return state


def merge_states(left: StateView, right: StateView) -> StateView:
    """Merge independent request/replica views with explicit conflicts."""
    if left.iteration != right.iteration:
        raise ModelingError("state_merge_iteration_mismatch")
    kv = dict(left.kv_ready)
    for key, value in right.kv_ready:
        if key in kv and kv[key] != value:
            raise ModelingError(f"state_merge_kv_conflict:{key}")
        kv[key] = value
    index = dict(left.index_ready)
    for key, value in right.index_ready:
        if key in index and index[key] != value:
            raise ModelingError(f"state_merge_index_conflict:{key}")
        index[key] = value
    selection = dict(left.selection_ready)
    for key, value in right.selection_ready:
        if key in selection and selection[key] != value:
            raise ModelingError(f"state_merge_selection_conflict:{key}")
        selection[key] = value
    cross_stage = dict(left.cross_stage_ready)
    for key, value in right.cross_stage_ready:
        if key in cross_stage and cross_stage[key] != value:
            raise ModelingError(f"state_merge_cross_stage_conflict:{key}")
        cross_stage[key] = value
    compression = dict(left.compression)
    for key, value in right.compression:
        if key in compression and compression[key] != value:
            raise ModelingError(f"state_merge_compression_conflict:{key}")
        compression[key] = value
    compression_requests = dict(left.compression_requests)
    for key, value in right.compression_requests:
        if key in compression_requests and compression_requests[key] != value:
            raise ModelingError(f"state_merge_compression_conflict:{key}")
        compression_requests[key] = value
    routing = dict(left.routing)
    for key, value in right.routing:
        if key in routing and routing[key] != value:
            raise ModelingError(f"state_merge_routing_conflict:{key}")
        routing[key] = value
    peer = dict(left.peer_bytes)
    for key, value in right.peer_bytes:
        peer[key] = peer.get(key, 0) + value
    return StateView(
        iteration=left.iteration, kv_ready=tuple(sorted(kv.items())),
        index_ready=tuple(sorted(index.items())),
        selection_ready=tuple(sorted(selection.items())),
        cross_stage_ready=tuple(sorted(cross_stage.items())),
        compression=tuple(sorted(compression.items())),
        compression_requests=tuple(sorted(
            compression_requests.items(), key=lambda item: (item[0][0], str(item[0][1])))),
        routing=tuple(sorted(routing.items())), residual=left.residual,
        acceptance=AcceptanceState(
            left.acceptance.accepted + right.acceptance.accepted,
            left.acceptance.committed + right.acceptance.committed,
            left.acceptance.rejected + right.acceptance.rejected,
            left.acceptance.commits + right.acceptance.commits),
        accepted_tokens=(left.accepted_tokens + right.accepted_tokens)[-1:],
        peer_bytes=tuple(sorted(peer.items())),
        candidate_blocks=tuple(sorted({**dict(left.candidate_blocks),
                                       **dict(right.candidate_blocks)}.items())),
    )


def routing_from_logits(top_k: int, valid_tokens: int, counts: Iterable[int]) -> MoeRoutingState:
    return MoeRoutingState(tuple(counts), top_k, valid_tokens)
