"""Typed immutable physical identity for lowered operators.

[Role]
  Provide the grouping primitive for lowered operators.  A physical key is a
  hashable, typed tuple projection of one operator's declared physical work and
  implementation, built directly from ``OpSpec`` fields.  Schema-recognized
  graph references are alpha-renamed to an occurrence-relative form so
  corresponding operators in different layers compare equal without erasing
  physical information.  Serialization happens only when a caller explicitly
  asks for ``physical_signature`` or ``physical_id``.

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
  * Layer-indexed state references (``kda.layer[L]`` / ``gdn.layer[L]``) and the
    integer ``stage``/``layer`` leaves of ``state_accesses`` are relative to the
    operator that owns them, for the same injective reason.
  * Unknown metadata is retained unchanged.  An unrecognized identity-bearing
    field can only prevent deduplication; it can never cause a false merge.
  * No regular expression is applied to a serialized JSON document; the rewrite
    visits declared values and only their occurrence prefixes.
  * ``physical_key`` returns a tuple and performs no serialization;
    ``physical_signature`` is its deterministic serialization and
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


#: Version tag of the physical identity projection.
IDENTITY_VERSION = "physical-key/3"

#: Fields excluded from physical identity.  They name one occurrence, not work.
OCCURRENCE_FIELDS = frozenset({
    "op_id", "deps", "stage", "layer", "requests", "stream", "notes",
})

#: Metadata keys whose values name other occurrences and must be normalized.
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

#: Instance pairing ids: they name one collective occurrence, not its work.
PAIRING_ONLY_KEYS = frozenset({"collective_id", "reverse_of_collective"})

#: Metadata leaves carrying an absolute stage/layer pair to be made relative.
STATE_ACCESS_KEY = "state_accesses"
STATE_STAGE_FIELD = "stage"
STATE_LAYER_FIELD = "layer"

_STAGE_GROUP = 1
_LAYER_GROUP = 2
_LAYER_REFERENCE_GROUP = 2
_OCCURRENCE = re.compile(r"(?<![\w.])s(\d+)\.l(\d+)\.")
_LAYER_REFERENCE = re.compile(r"(?<![\w.])(kda|gdn)\.layer\[(\d+)\]")

#: Identity field order.  A stable declaration, not a per-call inspection.
IDENTITY_FIELDS = (
    "family", "phase", "semantic_shape", "local_shape", "padded_shape",
    "launch_shape", "flops", "hbm_bytes", "mte2_bytes", "mte3_bytes",
    "vector_ops", "link_bytes", "link_scope", "recipe", "coverage", "dtype",
    "layout",
)
OPTIONAL_IDENTITY_FIELDS = ("returns", "source_boundary")
ZERO = 0
ONE = 1
ASCII = "utf-8"


def is_reference_key(key: str) -> bool:
    """True when this metadata key's value names other graph occurrences."""
    if key in REFERENCE_KEYS:
        return True
    if key.startswith(REFERENCE_PREFIXES):
        return True
    return key.endswith(REFERENCE_SUFFIXES)


def normalize_reference(value: str, stage: int, layer: int) -> str:
    """Rename occurrence prefixes relative to one anchor, preserving offsets."""
    def _rename(match: "re.Match[str]") -> str:
        return (f"s{int(match.group(_STAGE_GROUP)) - int(stage)}"
                f".l{int(match.group(_LAYER_GROUP)) - int(layer)}.")

    def _layer(match: "re.Match[str]") -> str:
        family = match.group(_STAGE_GROUP)
        return (f"{family}.layer["
                f"{int(match.group(_LAYER_REFERENCE_GROUP)) - int(layer)}]")

    renamed = _OCCURRENCE.sub(_rename, value)
    return _LAYER_REFERENCE.sub(_layer, renamed)


def normalize_graph_value(value: Any, stage: int, layer: int) -> Any:
    """Recursively normalize declared reference leaves into a hashable value."""
    if isinstance(value, str):
        return normalize_reference(value, stage, layer)
    if isinstance(value, Mapping):
        return tuple(sorted(
            ((str(key), normalize_graph_value(item, stage, layer))
             for key, item in value.items()),
            key=lambda item: item[ZERO]))
    if isinstance(value, (list, tuple)):
        return tuple(normalize_graph_value(item, stage, layer) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(normalize_graph_value(item, stage, layer)
                         for item in value)
    return value


def _relative_state_accesses(value: Any, stage: int, layer: int) -> Any:
    """Normalize the integer stage/layer leaves of declared state accesses."""
    if not isinstance(value, (list, tuple)):
        return normalize_graph_value(value, stage, layer)
    entries = []
    for item in value:
        if not isinstance(item, Mapping):
            entries.append(normalize_graph_value(item, stage, layer))
            continue
        row = {}
        for key, leaf in item.items():
            if key == STATE_STAGE_FIELD and isinstance(leaf, int):
                row[key] = int(leaf) - int(stage)
            elif key == STATE_LAYER_FIELD and isinstance(leaf, int):
                row[key] = int(leaf) - int(layer)
            else:
                row[key] = normalize_graph_value(leaf, stage, layer)
        entries.append(tuple(sorted(row.items(), key=lambda pair: str(pair[ZERO]))))
    return tuple(entries)


def _normalized_metadata(metadata: Any, stage: int, layer: int) -> tuple:
    """Hashable occurrence-relative projection of one operator's metadata."""
    items = metadata.items() if isinstance(metadata, Mapping) else tuple(metadata or ())
    normalized = []
    for key, value in items:
        name = str(key)
        if name in PAIRING_ONLY_KEYS:
            continue
        if name == STATE_ACCESS_KEY:
            normalized.append((name, _relative_state_accesses(value, stage, layer)))
        elif is_reference_key(name):
            normalized.append((name, normalize_graph_value(value, stage, layer)))
        else:
            normalized.append((name, _hashable(value)))
    return tuple(sorted(normalized, key=lambda item: item[ZERO]))


def _hashable(value: Any) -> Any:
    """Non-reference leaf projection; preserves value, only makes it hashable."""
    if isinstance(value, Mapping):
        return tuple(sorted(((str(key), _hashable(item))
                             for key, item in value.items()),
                            key=lambda item: item[ZERO]))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_hashable(item) for item in value)
    return value


def operator_metadata_key(metadata: Any, stage: int, layer: int) -> tuple:
    """Canonical hashable projection of one operator's metadata."""
    return _normalized_metadata(metadata, stage, layer)


def physical_key(op: Any) -> tuple:
    """Typed immutable physical identity of one operator occurrence."""
    key = [IDENTITY_VERSION]
    for field in IDENTITY_FIELDS:
        key.append(getattr(op, field, None))
    for field in OPTIONAL_IDENTITY_FIELDS:
        if getattr(op, field, None):
            key.append((field, getattr(op, field)))
    if getattr(op, "matrix_flops", ()):
        key.append(("matrix_flops", tuple(sorted(
            (str(name), float(value))
            for name, value in op.matrix_flops))))
    key.append(operator_metadata_key(
        getattr(op, "metadata", ()), int(op.stage), int(op.layer)))
    return tuple(key)


def canonical_json(key: Any) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"), default=repr,
                      ensure_ascii=False, allow_nan=False)


def physical_signature(op: Any) -> str:
    """Deterministic serialization of the physical key."""
    return canonical_json(physical_key(op))


def physical_id(op: Any) -> str:
    """Stable digest of the serialized physical key."""
    return hashlib.sha256(physical_signature(op).encode(ASCII)).hexdigest()
