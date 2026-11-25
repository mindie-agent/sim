"""Typed canonical subgraph vocabulary and tensor/state/event port contracts.

[Role]
  Define the finite semantic subgraph-kind vocabulary and the typed ports
  shared by model, framework and native program fragments.  ``StatePortSpec``
  carries ownership/readiness instead of an untyped owner dictionary;
  ``TensorPortSpec``/``EventPortSpec``/``PortSet`` are the common fragment
  ports.

[Invariants]
  * Port records are immutable; all mappings become tuples.
  * Port names are unique inside one ``PortSet`` so a binding has one target.
    Direction, scope and readiness are explicit; a tensor port never infers a
    dtype or layout from another value.
  * A consumer state port must declare a producer and a readiness contract;
    this module validates records, never device work.
  * Unknown kinds/variants fail loudly.

[Timeouts & Locks]
  Pure immutable record construction and bounded validation.  No waits, I/O,
  locks, device probes or weight loading.

[Side-Effect Policy]
  No filesystem/network/global mutation; only new frozen records are returned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .common import ModelingError

#: The 29 canonical LLM subgraph families under their paper / vLLM names.
#: Unknown mechanisms must remain an explicit coverage hole rather than being
#: folded into one of these entries.  Attention is split by the mechanism the
#: kernel actually executes (MHA/GQA/MQA versus latent MLA/DSA versus
#: window-compressed SWA/CSA/CSA2/HCA versus linear GDN/KDA), not by a numeric
#: family code.
SUBGRAPH_KINDS: tuple[str, ...] = (
    "vocab_parallel_embedding",
    "layernorm",
    "kv_cache",
    "kv_compressor",
    "indexer_prolog",
    "lightning_indexer",
    "mlp",
    "moe_router",
    "moe_token_dispatcher",
    "routed_experts",
    "shared_experts",
    "hyper_connection",
    "attn_residual",
    "per_layer_embedding",
    "lm_head_sampler",
    "spec_decode",
    "mm_encoder_projector",
    "mha_attention",
    "gqa_attention",
    "mqa_attention",
    "attention",
    "mla_attention",
    "dsa_attention",
    "swa_attention",
    "csa_attention",
    "csa2_attention",
    "hca_attention",
    "gdn_linear_attn",
    "kda_linear_attn",
)

SUBGRAPH_KIND_SET = frozenset(SUBGRAPH_KINDS)

#: Standard (softmax) attention kinds; the ``attention`` member is the
#: shape-free fallback used when a plan declares no head geometry.
STANDARD_ATTENTION_KINDS: frozenset[str] = frozenset({
    "mha_attention", "gqa_attention", "mqa_attention", "attention",
})

#: Latent attention kinds: uncompressed MLA and sparse (DSA) latent MLA.
LATENT_ATTENTION_KINDS: frozenset[str] = frozenset({
    "mla_attention", "dsa_attention",
})

#: Window and compressed-sparse attention kinds (SWA plus the compressed
#: branches that merge a window and a compressed-KV result through LSE).
WINDOW_SPARSE_ATTENTION_KINDS: frozenset[str] = frozenset({
    "swa_attention", "csa_attention", "csa2_attention", "hca_attention",
})

#: Compressed branches only (SWA is a window branch without compression).
COMPRESSED_ATTENTION_KINDS: frozenset[str] = frozenset({
    "csa_attention", "csa2_attention", "hca_attention",
})

#: Linear-attention kinds with a recurrent state instead of a KV cache.
LINEAR_ATTENTION_KINDS: frozenset[str] = frozenset({
    "gdn_linear_attn", "kda_linear_attn",
})

#: Kinds that may execute the standard softmax attention macro (including the
#: window-only SWA schedule, which shares that macro and its qkv_norm_rope
#: fusion).  The window/compressed macro is *not* part of this set even when
#: it emits ``swa_attention`` for a compression-free plan.
SOFTMAX_ATTENTION_KINDS: frozenset[str] = STANDARD_ATTENTION_KINDS | frozenset({
    "swa_attention",
})

#: Every kind that executes an attention (or linear-attention) core.
ALL_ATTENTION_KINDS: frozenset[str] = (
    STANDARD_ATTENTION_KINDS | LATENT_ATTENTION_KINDS
    | WINDOW_SPARSE_ATTENTION_KINDS | LINEAR_ATTENTION_KINDS
)

#: Kinds that read a KV cache; a draft/verify stage needs one of these.
KV_READING_ATTENTION_KINDS: frozenset[str] = frozenset({
    "mha_attention", "gqa_attention", "mqa_attention", "attention",
    "mla_attention", "dsa_attention",
    "swa_attention", "csa_attention", "csa2_attention", "hca_attention",
})

#: MoE kinds that form the routed/shared expert execution chain.
MOE_KINDS: frozenset[str] = frozenset({
    "moe_router", "moe_token_dispatcher", "routed_experts", "shared_experts",
})

_PORT_DIRECTIONS = frozenset({"in", "out", "inout"})
_PORT_SCOPES = frozenset({"operator", "layer", "stage", "request", "cohort"})
_PORT_READINESS = frozenset({
    "producer_complete", "same_invocation", "previous_iteration", "external",
})

#: Event ports name the causality a fragment waits for or publishes.
EVENT_PORT_KINDS = frozenset({
    "ready", "complete", "commit", "release", "grant", "record", "wait",
})

#: Fusion membership may never cross these boundaries.
_FUSION_BOUNDARIES = frozenset({"stream", "stage", "phase", "capture"})


def _freeze_value(item: Any, reason: str) -> Any:
    if item is None or isinstance(item, (bool, int, float, str)):
        return item
    if isinstance(item, Mapping):
        raise ModelingError(f"{reason}:mapping_value_invalid")
    if isinstance(item, (tuple, list)):
        return tuple(_freeze_value(sub, reason) for sub in item)
    raise ModelingError(f"{reason}:mutable_value_invalid:{type(item).__name__}")


def _tuple_of_pairs(value: Any, reason: str) -> tuple[tuple[str, Any], ...]:
    """Normalise a mapping or iterable of pairs into sorted immutable pairs."""
    if not value:
        return ()
    if isinstance(value, dict) or isinstance(value, Mapping):
        value = tuple(value.items())
    try:
        pairs = tuple((str(key), _freeze_value(item, reason)) for key, item in value)
    except ModelingError:
        raise
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    seen: set[str] = set()
    for key, _ in pairs:
        if not key or key in seen:
            raise ModelingError(f"{reason}:duplicate_or_empty:{key}")
        seen.add(key)
    return tuple(sorted(pairs))


def _typed_port_tuple(value: Any, cls: type, reason: str) -> tuple:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise ModelingError(reason)
    result = tuple(value)
    if not all(isinstance(port, cls) for port in result):
        raise ModelingError(reason)
    names = [port.name for port in result]
    if len(set(names)) != len(names):
        raise ModelingError(f"{reason}:duplicate_port")
    return result


def _port_tuple(value: Any, reason: str) -> tuple["StatePortSpec", ...]:
    return _typed_port_tuple(value, StatePortSpec, reason)


@dataclass(frozen=True)
class StatePortSpec:
    """One typed state/production port of a subgraph.

    ``owner`` is ``self`` or ``layer:<n>``/``stage:<n>``; ``readiness`` states
    which completion edge a consumer must observe.  A port describes a
    dependency and a representation, not device traffic by itself.
    """

    name: str
    direction: str
    representation: str
    scope: str = "layer"
    owner: str = "self"
    readiness: str = "producer_complete"
    unit: str = ""
    persistent: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("state_port_name_invalid")
        if self.direction not in _PORT_DIRECTIONS:
            raise ModelingError(f"state_port_direction_invalid:{self.direction}")
        if self.scope not in _PORT_SCOPES:
            raise ModelingError(f"state_port_scope_invalid:{self.scope}")
        if self.readiness not in _PORT_READINESS:
            raise ModelingError(f"state_port_readiness_invalid:{self.readiness}")
        if not isinstance(self.representation, str) or not self.representation.strip():
            raise ModelingError(f"state_port_representation_invalid:{self.name}")
        if not isinstance(self.owner, str):
            raise ModelingError(f"state_port_owner_invalid:{self.name}")
        if not isinstance(self.unit, str):
            raise ModelingError(f"state_port_unit_invalid:{self.name}")

    @property
    def is_producer(self) -> bool:
        return self.direction in {"out", "inout"}

    @property
    def is_consumer(self) -> bool:
        return self.direction in {"in", "inout"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "representation": self.representation, "scope": self.scope,
            "owner": self.owner, "readiness": self.readiness,
            "unit": self.unit, "persistent": self.persistent,
        }


@dataclass(frozen=True)
class TensorPortSpec:
    """A typed tensor input/output of a fragment or subgraph.

    ``dtype``/``layout_kind`` describe this value's own representation; they
    are never inherited from a model-level activation dtype.  ``optional``
    marks a port that may be absent, not a port whose value may be guessed.
    """

    name: str
    direction: str
    representation: str
    dtype: str = ""
    layout_kind: str = "strided"
    scope: str = "operator"
    owner: str = "self"
    readiness: str = "producer_complete"
    optional: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("tensor_port_name_invalid")
        if self.direction not in _PORT_DIRECTIONS:
            raise ModelingError(f"tensor_port_direction_invalid:{self.direction}")
        if self.scope not in _PORT_SCOPES:
            raise ModelingError(f"tensor_port_scope_invalid:{self.scope}")
        if self.readiness not in _PORT_READINESS:
            raise ModelingError(f"tensor_port_readiness_invalid:{self.name}")
        if not isinstance(self.representation, str) or not self.representation.strip():
            raise ModelingError(f"tensor_port_representation_invalid:{self.name}")
        for field in ("dtype", "layout_kind", "owner"):
            if not isinstance(getattr(self, field), str):
                raise ModelingError(f"tensor_port_{field}_invalid:{self.name}")
        if not self.layout_kind:
            raise ModelingError(f"tensor_port_layout_kind_invalid:{self.name}")
        if not isinstance(self.optional, bool):
            raise ModelingError(f"tensor_port_optional_invalid:{self.name}")

    @property
    def is_producer(self) -> bool:
        return self.direction in {"out", "inout"}

    @property
    def is_consumer(self) -> bool:
        return self.direction in {"in", "inout"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "representation": self.representation, "dtype": self.dtype,
            "layout_kind": self.layout_kind, "scope": self.scope,
            "owner": self.owner, "readiness": self.readiness,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class EventPortSpec:
    """A typed event dependency or publication.

    ``generation`` identifies the recorded event instance; a consumer waits on
    that record, not on an arbitrary same-named historical event.
    """

    name: str
    direction: str
    event: str
    scope: str = "stage"
    owner: str = "self"
    readiness: str = "producer_complete"
    generation: int | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("event_port_name_invalid")
        if self.direction not in _PORT_DIRECTIONS:
            raise ModelingError(f"event_port_direction_invalid:{self.direction}")
        if self.event not in EVENT_PORT_KINDS:
            raise ModelingError(f"event_port_kind_unknown:{self.event}")
        if self.scope not in _PORT_SCOPES:
            raise ModelingError(f"event_port_scope_invalid:{self.scope}")
        if self.readiness not in _PORT_READINESS:
            raise ModelingError(f"event_port_readiness_invalid:{self.name}")
        if not isinstance(self.owner, str):
            raise ModelingError(f"event_port_owner_invalid:{self.name}")
        if self.generation is not None and (
                not isinstance(self.generation, int)
                or isinstance(self.generation, bool)
                or self.generation < 0):
            raise ModelingError(f"event_port_generation_invalid:{self.name}")
        if not isinstance(self.optional, bool):
            raise ModelingError(f"event_port_optional_invalid:{self.name}")

    @property
    def is_producer(self) -> bool:
        return self.direction in {"out", "inout"}

    @property
    def is_consumer(self) -> bool:
        return self.direction in {"in", "inout"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "event": self.event, "scope": self.scope, "owner": self.owner,
            "readiness": self.readiness, "generation": self.generation,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class PortSet:
    """The common tensor/state/event ports of one fragment.

    Names are unique across all six groups.  Each group is normalised to a
    tuple of the declared port type; a plain string or an untyped mapping is
    rejected instead of being guessed.
    """

    tensor_in: tuple[TensorPortSpec, ...] = ()
    tensor_out: tuple[TensorPortSpec, ...] = ()
    state_in: tuple[StatePortSpec, ...] = ()
    state_out: tuple[StatePortSpec, ...] = ()
    event_in: tuple[EventPortSpec, ...] = ()
    event_out: tuple[EventPortSpec, ...] = ()

    def __post_init__(self) -> None:
        groups = (
            ("tensor_in", TensorPortSpec), ("tensor_out", TensorPortSpec),
            ("state_in", StatePortSpec), ("state_out", StatePortSpec),
            ("event_in", EventPortSpec), ("event_out", EventPortSpec),
        )
        names: dict[str, str] = {}
        for group, cls in groups:
            ports = _typed_port_tuple(getattr(self, group), cls,
                                      f"port_set_{group}_invalid")
            object.__setattr__(self, group, ports)
            for port in ports:
                if not (port.is_consumer if group.endswith("_in") else port.is_producer):
                    raise ModelingError(f"port_set_direction_mismatch:{group}:{port.name}:{port.direction}")
                if port.name in names:
                    raise ModelingError(
                        f"port_set_name_duplicate:{port.name}:"
                        f"{names[port.name]}:{group}")
                names[port.name] = group

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(port.name for group in (
            self.tensor_in, self.tensor_out, self.state_in, self.state_out,
            self.event_in, self.event_out) for port in group)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "tensor_in": [port.to_dict() for port in self.tensor_in],
            "tensor_out": [port.to_dict() for port in self.tensor_out],
            "state_in": [port.to_dict() for port in self.state_in],
            "state_out": [port.to_dict() for port in self.state_out],
            "event_in": [port.to_dict() for port in self.event_in],
            "event_out": [port.to_dict() for port in self.event_out],
        }


def validate_kind(kind: str) -> str:
    if kind not in SUBGRAPH_KIND_SET:
        raise ModelingError(f"subgraph_kind_unknown:{kind}")
    return kind
