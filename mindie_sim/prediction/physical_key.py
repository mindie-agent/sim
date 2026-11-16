"""Typed immutable physical identity for lowered operators.

[Role]
  Replace dictionary serialization as the grouping primitive.  A physical key
  is a hashable, typed projection of one operator's declared physical work and
  implementation, with schema-recognized graph references alpha-renamed to an
  occurrence-relative form so corresponding operators in different layers
  compare equal without erasing physical information.

[Invariants]
  * Only declared graph-reference leaves are normalized.  Every tensor shape,
    stride, offset, layout, dtype, quantization block/scale, byte count,
    routing distribution, cache geometry, state effect and allocation size is
    preserved verbatim.
  * Alpha-renaming is injective over the declared reference namespace: a
    self-reference becomes ``s0.l0.``, a reference to the previous layer stays
    ``s0.l-1.``, and role suffixes are never rewritten.  Equality and
    inequality of references, aliasing and ownership relationships therefore
    survive the normalization.
  * Unknown metadata is retained unchanged.  An unrecognized identity-bearing
    field can only prevent deduplication; it can never cause a false merge.
  * No regular expression is applied to a serialized JSON document; the rewrite
    visits declared values and only their occurrence prefixes.
  * ``physical_signature`` is the deterministic serialization of the key and
    ``physical_id`` its SHA-256 digest.

[Timeouts & Locks]
  Pure value transformation over one operator; no I/O.

[Side-Effect Policy]
  Reads the operator declaration and returns a new immutable value.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any


#: Fields excluded from physical identity.  They name one occurrence, not work.
OCCURRENCE_FIELDS = frozenset({
    "op_id", "deps", "stage", "layer", "requests", "stream", "notes",
})

#: Explicit graph-reference containers reported by operator lowering.
REFERENCE_KEYS = frozenset({
    "activation_input_segments",
    "activation_payload_producer",
    "activation_residual_inputs",
    "activation_residual_outputs",
    "binding_name",
    "collective_id",
    "device_kernel_group",
    "fused_origin_scopes",
    "fusion_group_key",
    "fusion_semantic_ids",
    "gdn_state_contract",
    "gdn_state_effects",
    "input_conversion_subgraphs",
    "inverse_index_producer",
    "kda_state_contract",
    "kda_state_effects",
    "memory_accesses",
    "packed_parent",
    "payload_producer",
    "receive_allocation",
    "reverse_of_collective",
    "storage",
    "synchronization_boundary",
})

#: Prefix families that describe occurrence ownership or state effects.
REFERENCE_PREFIXES = ("state_", "attn_res_", "structure_subgraph")

#: Suffix families whose value names the producer occurrence.
REFERENCE_SUFFIXES = ("_producer", "_consumer", "_parent")

_OCCURRENCE = re.compile(r"(?<![\w.])s(\d+)\.l(\d+)\.")
STAGE_GROUP = 1
LAYER_GROUP = 2
_IDENTITY_VERSION = "physical-key/2"


def is_reference_key(key: str) -> bool:
    if key in REFERENCE_KEYS:
        return True
    if key.startswith(REFERENCE_PREFIXES):
        return True
    return key.endswith(REFERENCE_SUFFIXES)


def normalize_reference(value: str, stage: int, layer: int) -> str:
    """Rename occurrence prefixes relative to one anchor, preserving offsets."""
    def _rename(match: "re.Match[str]") -> str:
        return (f"s{int(match.group(STAGE_GROUP)) - int(stage)}"
                f".l{int(match.group(LAYER_GROUP)) - int(layer)}.")
    return _OCCURRENCE.sub(_rename, value)


def normalize_graph_value(value: Any, stage: int, layer: int) -> Any:
    """Recursively normalize declared reference leaves."""
    if isinstance(value, str):
        return normalize_reference(value, stage, layer)
    if isinstance(value, Mapping):
        return {key: normalize_graph_value(item, stage, layer)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(normalize_graph_value(item, stage, layer) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(normalize_graph_value(item, stage, layer) for item in value)
    return value


class PhysicalKey(str):
    """Immutable canonical identity token (hashable string specialization)."""

    __slots__ = ()


def _normalized_metadata(metadata: Any, stage: int, layer: int) -> Any:
    items = metadata.items() if isinstance(metadata, Mapping) else tuple(metadata or ())
    return {str(key): (normalize_graph_value(value, stage, layer)
                       if is_reference_key(str(key)) else value)
            for key, value in items}


def operator_metadata_key(metadata: Any, stage: int, layer: int) -> Any:
    """Canonical hashable projection of one operator's metadata."""
    return canonical_json(_normalized_metadata(metadata, stage, layer))


#: Instance pairing ids: they name one collective occurrence, not its work.
PAIRING_ONLY_KEYS = frozenset({"collective_id", "reverse_of_collective"})


def physical_payload(op: Any) -> dict[str, Any]:
    """Occurrence-independent physical declaration with normalized references."""
    data = op.to_dict()
    payload = {key: value for key, value in data.items()
               if key not in OCCURRENCE_FIELDS and key != "metadata"}
    metadata = _normalized_metadata(
        data.get("metadata", {}), int(op.stage), int(op.layer))
    payload["metadata"] = {key: value for key, value in metadata.items()
                           if key not in PAIRING_ONLY_KEYS}
    return payload


def physical_key(op: Any) -> PhysicalKey:
    """Typed immutable physical identity of one operator occurrence."""
    return PhysicalKey(canonical_json(physical_payload(op)))


def canonical_json(key: Any) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"), default=repr,
                      ensure_ascii=False, allow_nan=False)


def physical_signature(op: Any) -> str:
    """Deterministic serialization of the physical key."""
    return str(physical_key(op))


def physical_id(op: Any) -> str:
    return hashlib.sha256(physical_signature(op).encode("utf-8")).hexdigest()
