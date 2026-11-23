"""Lossless, runtime-agnostic on-disk checkpoint quantization contracts.

[Role] Define immutable L0 records describing on-disk checkpoint quantization
  metadata (``CheckpointProvenance``, ``TensorPresenceDomain``,
  ``TensorQuantRule``, ``LayerQuantException``, ``CacheQuantRule``,
  ``CheckpointQuantMap``) independently of any inference engine runtime.
[Invariants] Preserves raw checkpoint tags verbatim, maintains exact
  source-native layer and expert coordinates, enforces precedence
  (``layer_exceptions`` -> ``ignored_modules`` -> ``rules``), never guesses
  unknown float dtypes, and never embeds runtime-specific prefix rewrites or
  fallback defaults.
[Timeouts & Locks] Pure in-memory validation and deterministic SHA-256 hashing;
  no I/O or locks.
[Side-Effect Policy] Strictly immutable dataclasses; zero mutation of inputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Literal

from .common import ModelingError

CHECKPOINT_QUANT_MAP_SCHEMA = "mindie-sim.checkpoint-quant-map/1"

CheckpointFormat = Literal[
    "modelslim_description",
    "hf_quantization_config",
    "unquantized_native",
    "explicit_synthetic_preset",
]
CHECKPOINT_FORMATS = frozenset({
    "modelslim_description",
    "hf_quantization_config",
    "unquantized_native",
    "explicit_synthetic_preset",
})

ModelScope = Literal["target", "draft", "vision", "projector"]
MODEL_SCOPES = frozenset({"target", "draft", "vision", "projector"})

CacheQuantProvenance = Literal[
    "checkpoint",
    "cli_override",
    "stack_default",
    "native",
]
CACHE_QUANT_PROVENANCES = frozenset({
    "checkpoint",
    "cli_override",
    "stack_default",
    "native",
})

CheckpointOrigin = Literal[
    "verified_checkpoint_descriptor",
    "hf_config_quantization",
    "unquantized_native",
    "explicit_synthetic_preset",
    "stack_default_quant_projection",
]
CHECKPOINT_ORIGINS = frozenset({
    "verified_checkpoint_descriptor",
    "hf_config_quantization",
    "unquantized_native",
    "explicit_synthetic_preset",
    "stack_default_quant_projection",
})

_ABS_HOME_PREFIX_RE = re.compile(r"^(?:/(?:Users|home|root)/|[A-Za-z]:\\Users\\)")


def _require_nonempty_str(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


def _require_nonneg_int_tuple(
    values: Sequence[int], code: str, *, require_nonempty: bool = False
) -> tuple[int, ...]:
    items = tuple(values)
    if require_nonempty and not items:
        raise ModelingError(code)
    out: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ModelingError(f"{code}:{item!r}")
        out.append(int(item))
    return tuple(sorted(set(out)))


def _require_pos_int_tuple(values: Sequence[int], code: str) -> tuple[int, ...]:
    out: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ModelingError(f"{code}:{item!r}")
        out.append(int(item))
    return tuple(out)


def _freeze_value(value: Any, code: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(k), _freeze_value(v, code)) for k, v in value.items())
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item, code) for item in value)
    raise ModelingError(f"{code}:{type(value).__name__}")


def _pattern_matches(pattern: str, tensor_path: str) -> bool:
    """Match dot-separated tensor paths where ``*`` matches one segment."""
    pat_parts = pattern.split(".")
    path_parts = tensor_path.split(".")
    if len(pat_parts) != len(path_parts):
        return False
    for pat_tok, path_tok in zip(pat_parts, path_parts):
        if pat_tok == "*":
            continue
        if pat_tok != path_tok:
            return False
    return True


def _ignored_module_matches(entry: str, tensor_path: str) -> bool:
    if entry == tensor_path:
        return True
    if "*" in entry and _pattern_matches(entry, tensor_path):
        return True
    return tensor_path.startswith(entry + ".")


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    """Immutable source provenance for an on-disk or synthetic checkpoint map."""

    checkpoint_id: str
    source_format: CheckpointFormat
    source_revision: str = ""
    source_sha256: str = ""
    descriptor_path: str = ""
    origin: CheckpointOrigin = "verified_checkpoint_descriptor"

    def __post_init__(self) -> None:
        cid = _require_nonempty_str(self.checkpoint_id, "checkpoint_id_required")
        if _ABS_HOME_PREFIX_RE.match(cid):
            raise ModelingError(f"checkpoint_id_leaks_host_home:{cid}")
        if self.source_format not in CHECKPOINT_FORMATS:
            raise ModelingError(f"checkpoint_source_format_invalid:{self.source_format!r}")
        if self.origin not in CHECKPOINT_ORIGINS:
            raise ModelingError(f"checkpoint_origin_invalid:{self.origin!r}")
        for attr in ("source_revision", "source_sha256", "descriptor_path"):
            val = getattr(self, attr)
            if not isinstance(val, str):
                raise ModelingError(f"checkpoint_provenance_field_invalid:{attr}")
        if self.descriptor_path and _ABS_HOME_PREFIX_RE.match(self.descriptor_path):
            raise ModelingError(f"checkpoint_descriptor_path_leaks_host_home:{self.descriptor_path}")
        object.__setattr__(self, "checkpoint_id", cid)

    def to_dict(self) -> dict[str, str]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "source_format": self.source_format,
            "source_revision": self.source_revision,
            "source_sha256": self.source_sha256,
            "descriptor_path": self.descriptor_path,
            "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class TensorPresenceDomain:
    """Exact on-disk coordinates where a wildcard tensor pattern exists."""

    scope: ModelScope
    layer_indices: tuple[int, ...] = ()
    expert_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in MODEL_SCOPES:
            raise ModelingError(f"tensor_presence_scope_invalid:{self.scope!r}")
        object.__setattr__(
            self,
            "layer_indices",
            _require_nonneg_int_tuple(self.layer_indices, "tensor_presence_layers_invalid"),
        )
        object.__setattr__(
            self,
            "expert_indices",
            _require_nonneg_int_tuple(self.expert_indices, "tensor_presence_experts_invalid"),
        )

    def contains(
        self,
        *,
        scope: ModelScope,
        source_native_layer_id: int | None,
        expert_id: int | None = None,
    ) -> bool:
        if self.scope != scope:
            return False
        if self.layer_indices and (
            source_native_layer_id is None or source_native_layer_id not in self.layer_indices
        ):
            return False
        if self.expert_indices and (
            expert_id is None or expert_id not in self.expert_indices
        ):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "layer_indices": list(self.layer_indices),
            "expert_indices": list(self.expert_indices),
        }


@dataclass(frozen=True, slots=True)
class TensorQuantRule:
    """Lossless on-disk tensor quantization rule with optional presence domain."""

    pattern: str
    raw_tag: str
    presence: TensorPresenceDomain | None = None
    weight_dtype: str = ""
    activation_dtype: str = ""
    scale_dtype: str = ""
    group_size: tuple[int, ...] = ()
    symmetric: bool | None = None

    def __post_init__(self) -> None:
        pat = _require_nonempty_str(self.pattern, "tensor_quant_rule_pattern_required")
        tag = _require_nonempty_str(self.raw_tag, "tensor_quant_rule_raw_tag_required")
        if self.presence is not None and not isinstance(self.presence, TensorPresenceDomain):
            raise ModelingError("tensor_quant_rule_presence_invalid")
        for attr in ("weight_dtype", "activation_dtype", "scale_dtype"):
            if not isinstance(getattr(self, attr), str):
                raise ModelingError(f"tensor_quant_rule_field_invalid:{attr}")
        if self.symmetric is not None and not isinstance(self.symmetric, bool):
            raise ModelingError("tensor_quant_rule_symmetric_invalid")
        object.__setattr__(self, "pattern", pat)
        object.__setattr__(self, "raw_tag", tag)
        object.__setattr__(
            self,
            "group_size",
            _require_pos_int_tuple(self.group_size, "tensor_quant_rule_group_size_invalid"),
        )

    def matches(
        self,
        tensor_path: str,
        *,
        scope: ModelScope,
        source_native_layer_id: int | None,
        expert_id: int | None = None,
    ) -> bool:
        if not _pattern_matches(self.pattern, tensor_path):
            return False
        if self.presence is not None:
            return self.presence.contains(
                scope=scope,
                source_native_layer_id=source_native_layer_id,
                expert_id=expert_id,
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "raw_tag": self.raw_tag,
            "presence": self.presence.to_dict() if self.presence is not None else None,
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "scale_dtype": self.scale_dtype,
            "group_size": list(self.group_size),
            "symmetric": self.symmetric,
        }


@dataclass(frozen=True, slots=True)
class LayerQuantException:
    """Lossless per-layer / per-expert tag override from raw checkpoint metadata."""

    pattern: str
    scope: ModelScope
    layer_indices: tuple[int, ...]
    raw_tag: str
    expert_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        pat = _require_nonempty_str(self.pattern, "layer_quant_exception_pattern_required")
        tag = _require_nonempty_str(self.raw_tag, "layer_quant_exception_raw_tag_required")
        if self.scope not in MODEL_SCOPES:
            raise ModelingError(f"layer_quant_exception_scope_invalid:{self.scope!r}")
        object.__setattr__(self, "pattern", pat)
        object.__setattr__(self, "raw_tag", tag)
        object.__setattr__(
            self,
            "layer_indices",
            _require_nonneg_int_tuple(
                self.layer_indices,
                "layer_quant_exception_layers_invalid",
                require_nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "expert_indices",
            _require_nonneg_int_tuple(
                self.expert_indices, "layer_quant_exception_experts_invalid"
            ),
        )

    def matches(
        self,
        tensor_path: str,
        *,
        scope: ModelScope,
        source_native_layer_id: int | None,
        expert_id: int | None = None,
    ) -> bool:
        if self.scope != scope:
            return False
        if source_native_layer_id is None or source_native_layer_id not in self.layer_indices:
            return False
        if self.expert_indices and (expert_id is None or expert_id not in self.expert_indices):
            return False
        return _pattern_matches(self.pattern, tensor_path)

    def specificity(self) -> tuple[int, int]:
        literal_tokens = sum(1 for tok in self.pattern.split(".") if tok != "*")
        return (1 if self.expert_indices else 0, literal_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "scope": self.scope,
            "layer_indices": list(self.layer_indices),
            "raw_tag": self.raw_tag,
            "expert_indices": list(self.expert_indices),
        }


@dataclass(frozen=True, slots=True)
class CacheQuantRule:
    """Independent per-field on-disk KV and Indexer cache quantization rule."""

    kv_tag: str = ""
    index_tag: str = ""
    kv_scale_dtype: str = ""
    index_scale_dtype: str = ""
    kv_offset_dtype: str = ""
    layer_scoped_kv: tuple[tuple[int, str], ...] = ()
    layer_scoped_index: tuple[tuple[int, str], ...] = ()
    provenance: CacheQuantProvenance = "native"

    def __post_init__(self) -> None:
        if self.provenance not in CACHE_QUANT_PROVENANCES:
            raise ModelingError(f"cache_quant_provenance_invalid:{self.provenance!r}")
        for attr in ("kv_tag", "index_tag", "kv_scale_dtype", "index_scale_dtype", "kv_offset_dtype"):
            if not isinstance(getattr(self, attr), str):
                raise ModelingError(f"cache_quant_field_invalid:{attr}")
        for attr in ("layer_scoped_kv", "layer_scoped_index"):
            raw = getattr(self, attr)
            normalized: list[tuple[int, str]] = []
            seen_layers: set[int] = set()
            for entry in raw:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise ModelingError(f"cache_quant_layer_scope_invalid:{attr}")
                layer_id, tag = entry
                if isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id < 0:
                    raise ModelingError(f"cache_quant_layer_id_invalid:{attr}:{layer_id!r}")
                if layer_id in seen_layers:
                    raise ModelingError(f"cache_quant_duplicate_layer:{attr}:{layer_id}")
                seen_layers.add(layer_id)
                normalized.append((layer_id, _require_nonempty_str(tag, f"cache_quant_tag_empty:{attr}")))
            object.__setattr__(self, attr, tuple(sorted(normalized)))

    def kv_tag_for_layer(self, source_native_layer_id: int | None) -> str:
        if source_native_layer_id is not None:
            for layer_id, tag in self.layer_scoped_kv:
                if layer_id == source_native_layer_id:
                    return tag
        return self.kv_tag

    def index_tag_for_layer(self, source_native_layer_id: int | None) -> str:
        if source_native_layer_id is not None:
            for layer_id, tag in self.layer_scoped_index:
                if layer_id == source_native_layer_id:
                    return tag
        return self.index_tag

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_tag": self.kv_tag,
            "index_tag": self.index_tag,
            "kv_scale_dtype": self.kv_scale_dtype,
            "index_scale_dtype": self.index_scale_dtype,
            "kv_offset_dtype": self.kv_offset_dtype,
            "layer_scoped_kv": [list(item) for item in self.layer_scoped_kv],
            "layer_scoped_index": [list(item) for item in self.layer_scoped_index],
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class CheckpointQuantMap:
    """Lossless, runtime-agnostic representation of on-disk checkpoint quantization."""

    provenance: CheckpointProvenance
    schema_version: str = CHECKPOINT_QUANT_MAP_SCHEMA
    native_float_dtype: str = ""
    model_quant_type: str = ""
    rules: tuple[TensorQuantRule, ...] = ()
    layer_exceptions: tuple[LayerQuantException, ...] = ()
    auxiliary_buffers: tuple[str, ...] = ()
    hf_quant_config: tuple[tuple[str, Any], ...] = ()
    ignored_modules: tuple[str, ...] = ()
    cache_rule: CacheQuantRule = field(default_factory=CacheQuantRule)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, CheckpointProvenance):
            raise ModelingError("checkpoint_quant_map_provenance_invalid")
        if self.schema_version != CHECKPOINT_QUANT_MAP_SCHEMA:
            raise ModelingError(f"checkpoint_quant_map_schema_unsupported:{self.schema_version!r}")
        for attr in ("native_float_dtype", "model_quant_type"):
            if not isinstance(getattr(self, attr), str):
                raise ModelingError(f"checkpoint_quant_map_field_invalid:{attr}")
        rules = tuple(self.rules)
        if not all(isinstance(rule, TensorQuantRule) for rule in rules):
            raise ModelingError("checkpoint_quant_map_rules_invalid")
        exceptions = tuple(self.layer_exceptions)
        if not all(isinstance(exc, LayerQuantException) for exc in exceptions):
            raise ModelingError("checkpoint_quant_map_exceptions_invalid")
        aux = tuple(self.auxiliary_buffers)
        if not all(isinstance(buf, str) and buf.strip() for buf in aux):
            raise ModelingError("checkpoint_quant_map_auxiliary_buffers_invalid")
        ignored = tuple(self.ignored_modules)
        if not all(isinstance(mod, str) and mod.strip() for mod in ignored):
            raise ModelingError("checkpoint_quant_map_ignored_modules_invalid")
        if not isinstance(self.cache_rule, CacheQuantRule):
            raise ModelingError("checkpoint_quant_map_cache_rule_invalid")
        raw_hf = self.hf_quant_config
        if isinstance(raw_hf, Mapping):
            raw_hf = tuple(sorted(raw_hf.items()))
        hf_frozen = tuple(
            (str(k), _freeze_value(v, "checkpoint_hf_quant_config_invalid"))
            for k, v in raw_hf
        )
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "layer_exceptions", exceptions)
        object.__setattr__(self, "auxiliary_buffers", aux)
        object.__setattr__(self, "ignored_modules", ignored)
        object.__setattr__(self, "hf_quant_config", hf_frozen)

    @property
    def checkpoint_id(self) -> str:
        return self.provenance.checkpoint_id

    @property
    def source_format(self) -> CheckpointFormat:
        return self.provenance.source_format

    def resolve_raw_tag(
        self,
        tensor_path: str,
        *,
        scope: ModelScope = "target",
        source_native_layer_id: int | None = None,
        expert_id: int | None = None,
    ) -> str | None:
        """Exact lossless lookup: (1) layer_exceptions -> (2) ignored_modules -> (3) rules."""
        if scope not in MODEL_SCOPES:
            raise ModelingError(f"checkpoint_lookup_scope_invalid:{scope!r}")
        matched_exc = [
            exc
            for exc in self.layer_exceptions
            if exc.matches(
                tensor_path,
                scope=scope,
                source_native_layer_id=source_native_layer_id,
                expert_id=expert_id,
            )
        ]
        if matched_exc:
            matched_exc.sort(key=lambda item: item.specificity(), reverse=True)
            return matched_exc[0].raw_tag
        for ignored in self.ignored_modules:
            if _ignored_module_matches(ignored, tensor_path):
                return "FLOAT"
        for rule in self.rules:
            if rule.matches(
                tensor_path,
                scope=scope,
                source_native_layer_id=source_native_layer_id,
                expert_id=expert_id,
            ):
                return rule.raw_tag
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance.to_dict(),
            "native_float_dtype": self.native_float_dtype,
            "model_quant_type": self.model_quant_type,
            "rules": [rule.to_dict() for rule in self.rules],
            "layer_exceptions": [exc.to_dict() for exc in self.layer_exceptions],
            "auxiliary_buffers": list(self.auxiliary_buffers),
            "hf_quant_config": [list(item) for item in self.hf_quant_config],
            "ignored_modules": list(self.ignored_modules),
            "cache_rule": self.cache_rule.to_dict(),
        }

    def digest(self) -> str:
        """Deterministic SHA-256 hex digest of this checkpoint quant map."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CACHE_QUANT_PROVENANCES",
    "CHECKPOINT_FORMATS",
    "CHECKPOINT_ORIGINS",
    "CHECKPOINT_QUANT_MAP_SCHEMA",
    "MODEL_SCOPES",
    "CacheQuantProvenance",
    "CacheQuantRule",
    "CheckpointFormat",
    "CheckpointOrigin",
    "CheckpointProvenance",
    "CheckpointQuantMap",
    "LayerQuantException",
    "ModelScope",
    "TensorPresenceDomain",
    "TensorQuantRule",
]
