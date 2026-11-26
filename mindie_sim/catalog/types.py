"""Typed quantization, draft, vision and Engram specifications for catalog models.

[Role]
  Hold the reusable, table-driven semantics needed by the nine-model catalog:
  packed weight/scale byte formulas, optional draft/MTP/verification geometry,
  vision-tower/media expansion geometry and Engram lookup/tables.  Model-family
  adapters in ``catalog`` only populate these records.

[Invariants]
  * Weight-file bytes, loaded/runtime residency bytes, activation bytes and
    collective payload bytes are separate quantities.
  * FP4 packing is rows*ceil(cols/2) plus E8M0 groups rows*ceil(cols/32);
    scales never disappear into a generic compression ratio.
  * FP8 block scales use explicit block shape and scale dtype; GLM's FP32
    scales differ from V4/V41 E8M0 scales.
  * Unquantized exclusions are category-based and explicit; router, norms,
    LM head, embeddings, vision and selected draft paths stay in their native
    format.
  * A missing vision/draft/Engram declaration raises on use instead of
    silently producing zero work.
  * Source revisions are provenance only and are not admission keys.

[Timeouts & Locks]
  Pure immutable records and arithmetic.  No I/O, locks, sleeps or devices.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Any, Mapping, Sequence

from mindie_sim.catalog.draft_spec import DraftSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes

_FP32_CATEGORIES = {
    "param_fp32", "mhc", "ape", "conv_weight_f32", "a_log", "dt_bias",
    "compressor_fp32_runtime",
}
_BF16_CATEGORIES = {
    "norm", "router", "router_bias", "embedding", "vision",
    "projector", "vision_pos", "image_token", "dspark_input",
    "compressor_bf16_runtime",
    "target_alias_embedding", "target_alias_lm_head",
}
_RUNTIME_OVERRIDE = {
    "dspark_input": "bf16",
    "engram_table": "fp8_e4m3",
}
_EXPERT_CATEGORIES = {"routed_expert", "dspark_routed_expert"}
_SHARED_CATEGORIES = {"shared_expert"}
#: Explicit normalization-parameter roles.  These are model semantics, not
#: arbitrary quant categories: a checkpoint header binds a stored gamma format
#: to one of these names, and the norm builders consume only these names.
_NORM_ROLES = frozenset({
    "norm", "attn_norm", "ffn_norm", "kv_a", "q_a", "indexer_k",
    "final", "shared_head", "enorm", "hnorm",
})
#: Roles covered by the generic ``norm`` declaration.  Shared-head, Engram and
#: hyper-connection norms have separate source headers and are not silently
#: promoted by a generic layer-norm declaration.
_NORM_GENERIC_ROLES = frozenset({
    "attn_norm", "ffn_norm", "kv_a", "q_a", "indexer_k", "final",
})
_NORM_FORMATS = {
    "fp32": "fp32", "float32": "fp32", "f32": "fp32",
    "bf16": "bf16", "bfloat16": "bf16",
}
MX_SCALE_GROUP_SIZE = 32
_HIGH_PRECISION_WEIGHT_FORMATS = frozenset({
    "bf16", "bfloat16", "fp16", "float16", "fp32", "float32",
})


def norm_role(name: str) -> str:
    """Map an operator name to one fixed normalization-parameter role."""
    text = str(name or "").lower().replace("-", "_")
    if "q_norm" in text or "q_a_layernorm" in text:
        return "q_a"
    if "kv_norm" in text or "kv_a_layernorm" in text:
        return "kv_a"
    if "index_norm" in text or ("indexer" in text and "norm" in text):
        return "indexer_k"
    if "final_norm" in text or text.endswith("model.norm") or text.endswith("model.norm.weight"):
        return "final"
    if "shared_head" in text:
        return "shared_head"
    if "enorm" in text:
        return "enorm"
    if "hnorm" in text:
        return "hnorm"
    if "norm1" in text or "attn_norm" in text or "input_layernorm" in text:
        return "attn_norm"
    if "norm2" in text or "ffn_norm" in text or "post_attention_layernorm" in text:
        return "ffn_norm"
    return "norm"
_FP8_CATEGORIES = {
    "attn", "mla", "gdn", "kda", "dense_mlp", "compressor", "engram_gate", "indexer",
    "indexer_fp8", "shared_expert",
    # MTP/draft-head linears: a quantized family like any other projection, but
    # its own category so a recipe can keep the draft head at BF16 (the Qwen3.8
    # checkpoint stores ``mtp.layers.0.*`` BF16 while the body is MXFP8).
    "mtp",
}


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


_CATEGORY_OVERRIDE_FIELDS = frozenset({
    "weight_format", "activation_format", "weight_block", "expert_block",
    "scale_format", "scale_bytes", "activation_group_size",
    "activation_scale_format", "activation_scale_pack",
    "int8_weight_metadata", "output_format",
    "runtime_weight_scale_format", "runtime_weight_scale_bytes",
    "runtime_activation_scale_format", "runtime_int8_weight_metadata",
    "weight_symmetric",
})
_CATEGORY_OVERRIDE_ALLOWED = _CATEGORY_OVERRIDE_FIELDS | frozenset({
    "expert_quant_count", "expert_quant_overrides",
})


def _normalize_override_fields(raw: Any, allowed: frozenset[str]) -> tuple[tuple[str, Any], ...]:
    fields = dict(raw)
    unknown = fields.keys() - allowed
    if unknown:
        raise ModelingError("quant_category_fields_unsupported:" + ",".join(sorted(unknown)))
    if "weight_block" in fields:
        fields["weight_block"] = tuple(fields["weight_block"])
    if "expert_quant_count" in fields:
        cnt = int(fields["expert_quant_count"])
        if cnt < 0:
            raise ModelingError("quant_expert_count_invalid")
        fields["expert_quant_count"] = cnt
    if "expert_quant_overrides" in fields:
        fields["expert_quant_overrides"] = _normalize_expert_quant_overrides(
            fields["expert_quant_overrides"], int(fields.get("expert_quant_count", 0) or 0),
        )
    return tuple(sorted(fields.items()))


def _normalize_expert_quant_overrides(
    raw: Any, expert_count: int = 0,
) -> tuple[tuple[tuple[int, ...], tuple[tuple[str, Any], ...]], ...]:
    if not raw:
        return ()
    if expert_count <= 0:
        raise ModelingError("quant_expert_count_required")
    entries = [((int(k),), v) for k, v in raw.items()] if isinstance(raw, Mapping) else raw
    seen: set[int] = set()
    normalized: list[tuple[tuple[int, ...], tuple[tuple[str, Any], ...]]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            raw_ids, raw_fields = entry.get("expert_indices", ()), entry.get("fields", {})
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            raw_ids, raw_fields = entry[0], entry[1]
        else:
            raise ModelingError("quant_expert_override_entry_invalid")
        indices = tuple(sorted({int(i) for i in raw_ids}))
        if not indices or indices[0] < 0:
            raise ModelingError("quant_expert_indices_invalid")
        if indices[-1] >= expert_count:
            raise ModelingError(f"quant_expert_index_out_of_range:{indices[-1]}>={expert_count}")
        if seen.intersection(indices):
            raise ModelingError("quant_expert_indices_duplicate")
        seen.update(indices)
        normalized.append((indices, _normalize_override_fields(raw_fields, _CATEGORY_OVERRIDE_FIELDS)))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _serialize_expert_quant_overrides(
    entries: tuple[tuple[tuple[int, ...], tuple[tuple[str, Any], ...]], ...],
) -> list[dict[str, Any]]:
    return [
        {"expert_indices": list(indices), "fields": dict(fields)}
        for indices, fields in entries
    ]


def _serialize_category_fields(fields: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    out = dict(fields)
    if "expert_quant_overrides" in out:
        out["expert_quant_overrides"] = _serialize_expert_quant_overrides(out["expert_quant_overrides"])
    if "weight_block" in out:
        out["weight_block"] = list(out["weight_block"])
    return out


def _apply_override_fields(spec: "QuantSpec", fields: tuple[tuple[str, Any], ...]) -> "QuantSpec":
    changes = dict(fields)
    if "weight_format" in changes:
        changes["expert_weight_format"] = changes["weight_format"]
        if spec.exclusions:
            changes["exclusions"] = tuple(
                ex for ex in spec.exclusions
                if ex.split(".", 1)[0] not in _EXPERT_CATEGORIES
            )
    return replace(spec, **changes)


@dataclass(frozen=True)
class QuantSpec:
    """Explicit representation plan; never a single global activation dtype."""

    name: str
    weight_format: str = "bf16"
    activation_format: str = ""
    expert_weight_format: str = "bf16"
    weight_block: tuple[int, int] = (0, 0)
    expert_block: int = 32
    scale_format: str = "fp32"
    scale_bytes: float = 4.0
    kv_dtype: str = "bf16"
    index_dtype: str = "int8"
    index_scale_dtype: str = "fp16"
    exclusions: tuple[str, ...] = ()
    engram_table_mode: str = "device_bf16"
    source: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    activation_group_size: int = 0
    activation_scale_format: str = ""
    activation_scale_pack: int = 1
    category_overrides: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    native_format: str = "bf16"
    int8_weight_metadata: str = "scale_offset"
    # Execution ABI can differ from checkpoint storage: the W8 dynamic
    # QuantBatchMatmulV3 normal caller consumes one BF16 scale per output
    # channel and one FLOAT scale per activation row, even when the checkpoint
    # stores FP32 per-channel scales.  Empty strings/zero preserve the prior
    # storage-derived behavior for every source that does not declare this ABI.
    runtime_weight_scale_format: str = ""
    runtime_weight_scale_bytes: float = 0.0
    runtime_activation_scale_format: str = ""
    runtime_int8_weight_metadata: str = ""
    output_format: str = ""
    #: Checkpoint/stored gamma formats per fixed norm role.  The source header
    #: proves storage; the runtime table may declare a deployment-selected cast.
    norm_storage: tuple[tuple[str, str], ...] = ()
    norm_runtime: tuple[tuple[str, str], ...] = ()
    layer_quant_overrides: tuple[tuple[str, Any], ...] = ()
    #: Symmetric integer quantization omits checkpoint zero-point/offset tensors
    #: on disk (``source=True``); runtime residency (``source=False``) keeps any
    #: kernel-materialized offset.
    weight_symmetric: bool = True
    #: Per-expert mixed-quantization overrides for routed-expert categories.
    #: ``expert_quant_count`` records total experts in the mixed layer/category
    #: (0 infers from covered indices); each override binds disjoint sorted
    #: expert indices to field overrides over the base expert representation.
    expert_quant_count: int = 0
    expert_quant_overrides: tuple[tuple[tuple[int, ...], tuple[tuple[str, Any], ...]], ...] = ()
    #: Internal categories whose ``weight_format`` came from an explicit category override.
    explicit_weight_categories: tuple[str, ...] = ()

    @staticmethod
    def _freeze_override_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(sorted((str(k), QuantSpec._freeze_override_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(QuantSpec._freeze_override_value(item) for item in value)
        return value

    @staticmethod
    def _thaw_override_value(value: Any) -> Any:
        if isinstance(value, tuple):
            if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
                return {k: QuantSpec._thaw_override_value(v) for k, v in value}
            return tuple(QuantSpec._thaw_override_value(item) for item in value)
        return value

    def layer_quant_overrides_dict(self) -> dict[str, Any]:
        """Return mutable mapping view of per-layer quant overrides (e.g. ``native_layers``)."""
        if not self.layer_quant_overrides:
            return {}
        thawed = self._thaw_override_value(self.layer_quant_overrides)
        return dict(thawed) if isinstance(thawed, Mapping) else {}

    def __post_init__(self):
        if not self.activation_format:
            wf = str(self.weight_format or "").strip().lower()
            default_act = "int8" if wf in {"int8", "w8a8"} else (self.native_format or "bf16")
            object.__setattr__(self, "activation_format", default_act)
        if int(self.expert_quant_count) < 0:
            raise ModelingError("quant_expert_count_invalid")
        object.__setattr__(self, "expert_quant_count", int(self.expert_quant_count))
        if self.int8_weight_metadata not in {"scale_only", "scale_offset"}:
            raise ModelingError("int8_weight_metadata_requires_scale_only_or_scale_offset")
        if self.runtime_int8_weight_metadata not in {"", "scale_only", "scale_offset"}:
            raise ModelingError("runtime_int8_weight_metadata_requires_scale_only_or_scale_offset")
        if self.runtime_weight_scale_bytes < 0:
            raise ModelingError("runtime_weight_scale_bytes_requires_nonnegative")
        if self.runtime_weight_scale_format and not self.runtime_weight_scale_bytes:
            raise ModelingError("runtime_weight_scale_bytes_required_with_format")
        if self.output_format and self.output_format not in {
                "bf16", "bfloat16", "fp16", "float16", "fp32", "float32",
                "int8", "fp8", "fp8_e4m3", "fp8_e5m2"}:
            raise ModelingError("quant_output_format_unsupported:" + self.output_format)
        # JSON mappings become immutable records before they enter model IDs.
        for key in ("weight_block", "exclusions", "notes", "explicit_weight_categories"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if isinstance(self.source, Mapping):
            object.__setattr__(self, "source", tuple(sorted(self.source.items())))
        entries = (self.category_overrides.items() if isinstance(self.category_overrides, Mapping)
                   else self.category_overrides)
        normalized = [
            (str(category), _normalize_override_fields(raw, _CATEGORY_OVERRIDE_ALLOWED))
            for category, raw in entries
        ]
        object.__setattr__(self, "category_overrides", tuple(sorted(normalized)))
        object.__setattr__(
            self,
            "expert_quant_overrides",
            _normalize_expert_quant_overrides(self.expert_quant_overrides, self.expert_quant_count),
        )
        for key in ("norm_storage", "norm_runtime"):
            raw = getattr(self, key)
            records = raw.items() if isinstance(raw, Mapping) else raw
            parsed = []
            for role, fmt in records:
                role = str(role)
                if role not in _NORM_ROLES:
                    raise ModelingError("quant_norm_role_unsupported:" + role)
                canonical = _NORM_FORMATS.get(str(fmt).strip().lower())
                if canonical is None:
                    raise ModelingError("quant_norm_format_unsupported:" + str(fmt))
                parsed.append((role, canonical))
            object.__setattr__(self, key, tuple(sorted(parsed)))
        # The checkpoint mapper historically put a stored F32 norm gamma into
        # the generic ``norm`` category override.  Keep that as storage
        # provenance only: it never selects the kernel gamma dtype, and its
        # activation_format is ignored.  Actual deployment gamma remains the
        # model's native format unless a separate runtime ABI is declared.
        norm_override = dict(self.category_overrides).get("norm")
        if norm_override is not None:
            stored = _NORM_FORMATS.get(
                str(dict(norm_override).get("weight_format", "")).strip().lower())
            if stored is not None:
                if not self.norm_storage:
                    object.__setattr__(self, "norm_storage", (("norm", stored),))
                object.__setattr__(self, "category_overrides", tuple(
                    (category, fields) for category, fields in self.category_overrides
                    if category != "norm"))
        if self.layer_quant_overrides:
            frozen_layer_ov = self._freeze_override_value(self.layer_quant_overrides)
            object.__setattr__(self, "layer_quant_overrides", frozen_layer_ov if isinstance(frozen_layer_ov, tuple) else ())
        else:
            object.__setattr__(self, "layer_quant_overrides", ())

    def for_category(self, category: str) -> "QuantSpec":
        """Resolve family defaults then progressively more specific roles."""
        if not self.category_overrides:
            return self
        cat = str(category)
        try:
            return _resolve_quant_category(self, cat)
        except TypeError:
            return _resolve_quant_category.__wrapped__(self, cat)

    def for_expert(self, expert_id: int) -> "QuantSpec":
        """Resolve representation for one specific routed expert ID."""
        if not self.expert_quant_overrides:
            return self
        idx = int(expert_id)
        base = replace(self, expert_quant_overrides=(), expert_quant_count=0)
        for indices, fields in self.expert_quant_overrides:
            if idx in indices:
                return _apply_override_fields(base, fields)
        return base

    _MX_METHODS = frozenset({"mx_fp8", "mx_fp4"})
    _FOUR_BIT_WEIGHT_FORMATS = frozenset({
        "mxfp4", "mx_fp4", "mxfp4_e2m1", "fp4", "fp4_e2m1",
        "int4", "int4_weight", "w4a8",
    })

    def linear_method(self, category: str) -> str:
        """Authoritative runtime linear method for one resolved role."""
        resolved = self.for_category(category)
        wf = str(resolved.format_for(category) or "").lower().replace("-", "_")
        if wf in _HIGH_PRECISION_WEIGHT_FORMATS:
            return "dense"
        af = str(resolved.activation_format or "").lower().replace("-", "_")
        sf = str(resolved.scale_format or "").lower().replace("-", "_")
        block = resolved.weight_block
        group = int(block[1] if isinstance(block, (tuple, list)) and len(block) > 1 else 0)
        exp_group = int(resolved.expert_block or 0)
        mx_scale_fmt = sf in {"fp8_e8m0", "ue8m0"}
        act_mx_group = int(resolved.activation_group_size or 0) == MX_SCALE_GROUP_SIZE
        if mx_scale_fmt and (group == MX_SCALE_GROUP_SIZE or exp_group == MX_SCALE_GROUP_SIZE) and wf in {
                "mxfp4", "mx_fp4", "mxfp4_e2m1", "fp4", "fp4_e2m1"}:
            return "mx_fp4"
        if mx_scale_fmt and group == MX_SCALE_GROUP_SIZE and act_mx_group and wf in {
                "mxfp8", "mx_fp8", "mxfp8_e4m3", "fp8_e4m3", "fp8_e5m2"} and af in {
                "mxfp8", "mx_fp8", "mxfp8_e4m3", "fp8_e4m3", "fp8_e5m2"}:
            return "mx_fp8"
        if wf in {"int4", "w4a8", "int4_weight"}:
            return "w4a8" if af == "int8" else "w4a16"
        if wf == "w8a8":
            return "w8a8"
        if wf == "int8":
            return "w8a8" if af == "int8" else "w8a16"
        if af == "int8":
            return "w8a8"
        return "fp8" if wf in {"fp8", "fp8_e4m3", "fp8_e5m2"} else "dense"

    def output_format_for(self, category: str, default: str = "", *, prequant: str = "") -> str:
        resolved = self.for_category(category)
        if resolved.output_format:
            return resolved.output_format
        if resolved.linear_method(category) in self._MX_METHODS:
            return prequant or resolved.native_format or default
        if str(resolved.format_for(category)).lower() in self._FOUR_BIT_WEIGHT_FORMATS:
            return prequant or resolved.native_format or default
        return default

    def format_for(self, category: str, *, source: bool = False, expert_id: int | None = None) -> str:
        cat = str(category)
        resolved = self.for_category(cat)
        if resolved is not self:
            return resolved.format_for(cat, source=source, expert_id=expert_id)
        family = cat.split(".", 1)[0]
        if expert_id is not None and self.expert_quant_overrides and family in _EXPERT_CATEGORIES:
            return self.for_expert(expert_id).format_for(cat, source=source)
        if source and family == "norm" and self.norm_gamma_declared("norm"):
            return self.norm_gamma_format("norm", source=True)
        if family in _FP32_CATEGORIES:
            return "fp32"
        if family in {"conv_weight_bf16", "linear_param_bf16"}:
            return "bf16"
        if cat in self.exclusions or family in self.exclusions:
            return self.native_format
        if cat == "mla_absorb":
            return self.format_for("mla", source=source)
        if (
            family in _BF16_CATEGORIES
            and cat not in self.explicit_weight_categories
            and family not in self.explicit_weight_categories
        ):
            return self.native_format
        if family in _EXPERT_CATEGORIES:
            return self.expert_weight_format
        if family in _SHARED_CATEGORIES:
            return "bf16" if self.weight_format == "bf16" else self.weight_format
        if family in _FP8_CATEGORIES:
            return self.weight_format
        if family == "engram_table":
            return self.engram_table_format
        if family == "norm_f32":
            return "fp32"
        return self.weight_format if self.weight_format != "bf16" else "bf16"

    def source_format_for(self, category: str, *, expert_id: int | None = None) -> str:
        """Checkpoint representation before an explicit runtime conversion."""
        cat = str(category)
        resolved = self.for_category(cat)
        if resolved is not self:
            return resolved.source_format_for(cat, expert_id=expert_id)
        if expert_id is not None and self.expert_quant_overrides and cat.split(".", 1)[0] in _EXPERT_CATEGORIES:
            return self.for_expert(expert_id).source_format_for(category)
        if cat == "norm" and self.norm_gamma_declared("norm"):
            return self.norm_gamma_format("norm", source=True)
        if cat in {"compressor_fp32_runtime", "compressor_bf16_runtime"}:
            return "bf16"
        if cat == "engram_table":
            return "fp8_e4m3" if self.weight_format != "bf16" else "bf16"
        if cat == "dspark_input":
            return self.weight_format if self.weight_format != "bf16" else "bf16"
        return self.format_for(cat, source=True)

    def norm_gamma_declared(self, role: str = "norm") -> bool:
        role = str(role)
        storage, runtime = dict(self.norm_storage), dict(self.norm_runtime)
        if role in storage or role in runtime:
            return True
        return role in _NORM_GENERIC_ROLES and ("norm" in storage or "norm" in runtime)

    def norm_gamma_format(self, role: str = "norm", *, source: bool = False) -> str:
        role = str(role)
        table = dict(self.norm_storage if source else self.norm_runtime)
        if role in table:
            return table[role]
        if role in _NORM_GENERIC_ROLES and "norm" in table:
            return table["norm"]
        return self.native_format or "bf16"

    def norm_gamma_bytes(self, width: int, role: str = "norm", *, source: bool = False) -> float:
        return float(int(width)) * float(dtype_bytes(self.norm_gamma_format(role, source=source)))

    def execution_weight_scale_format(self) -> str:
        return self.runtime_weight_scale_format or self.scale_format

    def execution_weight_scale_bytes(self) -> float:
        return self.runtime_weight_scale_bytes or self.scale_bytes

    def execution_activation_scale_format(self, dtype: str = "") -> str:
        if self.runtime_activation_scale_format:
            return self.runtime_activation_scale_format
        act = str(dtype or self.activation_format or "").lower()
        if act == "int8":
            if str(self.activation_format or "").lower() == "int8" and self.activation_scale_format:
                return self.activation_scale_format
            return "fp32"
        if self.activation_scale_format:
            return self.activation_scale_format
        if str(self.weight_format or "").lower() in {
                "int8", "w8a8", "int4", "w4a8"}:
            return "fp32"
        return self.scale_format

    def execution_int8_weight_metadata(self) -> str:
        return self.runtime_int8_weight_metadata or self.int8_weight_metadata

    def expert_subset_bytes(
        self,
        rows: int,
        cols: int,
        category: str,
        expert_ids: Sequence[int],
        *,
        source: bool = False,
        include_scales: bool = True,
    ) -> float:
        """Exact byte sum across concrete routed ``expert_ids`` (e.g. one EP rank)."""
        resolved = self.for_category(category)
        if resolved is not self:
            return resolved.expert_subset_bytes(
                rows, cols, category, expert_ids, source=source, include_scales=include_scales,
            )
        method = "matrix_bytes" if include_scales else "matrix_payload_bytes"
        if not self.expert_quant_overrides or str(category).split(".", 1)[0] not in _EXPERT_CATEGORIES:
            return float(len(expert_ids)) * getattr(self, method)(rows, cols, category, source=source)
        return float(sum(
            getattr(self, method)(rows, cols, category, source=source, expert_id=int(eid))
            for eid in expert_ids
        ))

    def _expert_weighted_bytes(
        self,
        rows: int,
        cols: int,
        category: str,
        *,
        source: bool,
        include_scales: bool,
        expert_id: int | None,
    ) -> float:
        method = "matrix_bytes" if include_scales else "matrix_payload_bytes"
        if expert_id is not None:
            return getattr(self.for_expert(expert_id), method)(rows, cols, category, source=source)
        base = replace(self, expert_quant_overrides=(), expert_quant_count=0)
        covered = 0
        weighted_total = 0.0
        for indices, fields in self.expert_quant_overrides:
            count = len(indices)
            covered += count
            expert_spec = _apply_override_fields(base, fields)
            weighted_total += count * getattr(expert_spec, method)(rows, cols, category, source=source)
        total_experts = max(int(self.expert_quant_count or 0), covered)
        if total_experts > covered:
            weighted_total += (total_experts - covered) * getattr(base, method)(rows, cols, category, source=source)
        return weighted_total / float(total_experts)

    def matrix_payload_bytes(
        self, rows: int, cols: int, category: str, *, source: bool = False,
        expert_id: int | None = None,
    ) -> float:
        """Packed payload bytes only (scales/zero-points excluded).

        ``matrix_bytes`` stays the resident weight quantity.  This split lets
        the memory plan report quant metadata as its own component without
        changing any weight byte or charging the scales twice.
        """
        resolved = self.for_category(category)
        if resolved is not self:
            return resolved.matrix_payload_bytes(rows, cols, category, source=source, expert_id=expert_id)
        rows, cols = int(rows), int(cols)
        if rows <= 0 or cols <= 0:
            raise ModelingError(f"catalog_weight_shape_invalid:{category}:{rows}x{cols}")
        if self.expert_quant_overrides and str(category).split(".", 1)[0] in _EXPERT_CATEGORIES:
            return self._expert_weighted_bytes(
                rows, cols, category, source=source, include_scales=False, expert_id=expert_id,
            )
        fmt = self.source_format_for(category) if source else self.format_for(category)
        if str(category) == "engram_table":
            mode = "host_fp8_e8m0" if source else self.engram_table_mode
            return float(rows * cols * (2 if mode == "device_bf16" else 1))
        if fmt in {"bf16", "bfloat16", "fp16", "float16"}:
            return float(rows * cols * 2)
        if fmt in {"fp32", "float32"}:
            return float(rows * cols * 4)
        if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2", "int8", "w8a8"}:
            return float(rows * cols)
        if fmt in {"mxfp4", "fp4", "fp4_e2m1", "int4", "w4a8", "w4a16"}:
            return float(rows * _ceil_div(cols, 2))
        raise ModelingError(f"catalog_weight_format_unsupported:{fmt}")

    def matrix_bytes(
        self, rows: int, cols: int, category: str, *, source: bool = False,
        expert_id: int | None = None,
    ) -> float:
        resolved = self.for_category(category)
        if resolved is not self:
            return resolved.matrix_bytes(rows, cols, category, source=source, expert_id=expert_id)
        rows, cols = int(rows), int(cols)
        if rows <= 0 or cols <= 0:
            raise ModelingError(f"catalog_weight_shape_invalid:{category}:{rows}x{cols}")
        if self.expert_quant_overrides and str(category).split(".", 1)[0] in _EXPERT_CATEGORIES:
            return self._expert_weighted_bytes(
                rows, cols, category, source=source, include_scales=True, expert_id=expert_id,
            )
        fmt = self.source_format_for(category) if source else self.format_for(category)
        scale_bytes = self.scale_bytes if source else self.execution_weight_scale_bytes()
        metadata_columns = (1 if self.int8_weight_metadata == "scale_only" else 2) if source else (
            1 if self.execution_int8_weight_metadata() == "scale_only" else 2)
        if str(category) == "engram_table":
            # Engram tables have explicit host/device storage modes.  Scales
            # are one per row per32 columns, not a 2-D block grid.
            mode = "host_fp8_e8m0" if source else self.engram_table_mode
            if mode == "device_bf16":
                return float(rows * cols * 2)
            if mode == "device_int8_fp32":
                return float(rows * cols + rows * _ceil_div(cols, 32) * 4)
            if mode in {"host_fp8_e8m0", "source_fp8_e8m0"}:
                return float(rows * cols + rows * _ceil_div(cols, 32))
            raise ModelingError(f"catalog_engram_table_mode_unsupported:{mode}")
        if fmt in {"bf16", "bfloat16", "fp16", "float16"}:
            return float(rows * cols * 2)
        if fmt in {"fp32", "float32"}:
            return float(rows * cols * 4)
        if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            data = float(rows * cols)
            block_rows, block_cols = self.weight_block
            if block_rows <= 0 or block_cols <= 0:
                raise ModelingError("catalog_fp8_block_missing")
            scale_rows = _ceil_div(rows, block_rows)
            scale_cols = _ceil_div(cols, block_cols)
            return data + scale_rows * scale_cols * float(scale_bytes)
        if fmt in {"mxfp4", "fp4", "fp4_e2m1"}:
            group = max(1, int(self.expert_block))
            data = float(rows * _ceil_div(cols, 2))
            scales = float(rows * _ceil_div(cols, group))
            return data + scales
        if fmt in {"int4", "w4a8", "w4a16"}:
            # W4A8_DYNAMIC packs two int4 values per byte and keeps one fp32
            # scale per output row (per-channel, ``expert_block=1``).  Groupwise
            # W4A16 (``wna16``, BF16/FP16 activations with ``group > 1``)
            # stores scales only on disk when symmetric (``source=True``) and
            # allocates both ``weight_scale`` and ``weight_offset`` at runtime.
            group = max(1, int(self.expert_block))
            data = float(rows * _ceil_div(cols, 2))
            scale_count = rows if group <= 1 else rows * _ceil_div(cols, group)
            act_fmt = str(self.activation_format or "").strip().lower()
            is_w4a16_groupwise = (
                fmt in {"int4", "w4a16"}
                and act_fmt in {"bf16", "bfloat16", "fp16", "float16"}
                and group > 1
            )
            scale_tensors = (
                (1 if self.weight_symmetric else 2) if source else 2
            ) if is_w4a16_groupwise else 1
            return data + float(scale_tensors * scale_count) * float(scale_bytes)
        if str(category) == "shared_expert_payload":
            # Payload only; the shared expert's scale is charged once.
            return float(rows * cols)
        if fmt in {"int8", "w8a8"}:
            act_fmt = str(self.activation_format or "").strip().lower()
            if fmt == "int8" and act_fmt in {"bf16", "bfloat16", "fp16", "float16"}:
                group = (
                    int(self.weight_block[1])
                    if (len(self.weight_block) > 1 and int(self.weight_block[1]) > 1)
                    else max(1, int(self.expert_block))
                )
                scale_count = rows if group <= 1 else rows * _ceil_div(cols, group)
                scale_tensors = (1 if self.weight_symmetric else 2) if source else metadata_columns
                return float(rows * cols) + float(scale_tensors * scale_count) * float(scale_bytes)
            # Native INT8 callers may provide one scale or a scale/offset pair.
            # This declaration also owns residency; an absent offset has no bytes.
            return float(rows * cols + rows * float(scale_bytes) * metadata_columns)
        raise ModelingError(f"catalog_weight_format_unsupported:{fmt}")

    @property
    def engram_table_format(self) -> str:
        if self.engram_table_mode == "device_bf16":
            return "bf16"
        return "int8" if self.engram_table_mode == "device_int8_fp32" else "fp8_e4m3"

    def logical_parameters(self, rows: int, cols: int) -> int:
        return int(rows) * int(cols)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "weight_format": self.weight_format, "native_format": self.native_format,
            "int8_weight_metadata": self.int8_weight_metadata, "activation_format": self.activation_format,
            "expert_weight_format": self.expert_weight_format, "weight_block": list(self.weight_block),
            "expert_block": self.expert_block, "scale_format": self.scale_format, "scale_bytes": self.scale_bytes,
            "kv_dtype": self.kv_dtype, "index_dtype": self.index_dtype, "index_scale_dtype": self.index_scale_dtype,
            "exclusions": list(self.exclusions), "engram_table_mode": self.engram_table_mode,
            "activation_group_size": self.activation_group_size, "activation_scale_format": self.activation_scale_format,
            "activation_scale_pack": self.activation_scale_pack, "runtime_weight_scale_format": self.runtime_weight_scale_format,
            "runtime_weight_scale_bytes": self.runtime_weight_scale_bytes,
            "runtime_activation_scale_format": self.runtime_activation_scale_format,
            "runtime_int8_weight_metadata": self.runtime_int8_weight_metadata, "output_format": self.output_format,
            "norm_storage": dict(self.norm_storage), "norm_runtime": dict(self.norm_runtime),
            "category_overrides": {cat: _serialize_category_fields(fields) for cat, fields in self.category_overrides},
            **({"layer_quant_overrides": self.layer_quant_overrides_dict()} if self.layer_quant_overrides else {}),
            **({"weight_symmetric": self.weight_symmetric} if not self.weight_symmetric else {}),
            **({"expert_quant_count": self.expert_quant_count} if self.expert_quant_count else {}),
            **({"expert_quant_overrides": _serialize_expert_quant_overrides(self.expert_quant_overrides)}
               if self.expert_quant_overrides else {}),
            "source": dict(self.source), "notes": list(self.notes),
        }


@dataclass(frozen=True)
class VisionSpec:
    kind: str
    layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    qkv_width: int
    patch_size: int
    in_channels: int = 3
    temporal_patch: int = 1
    merge_area: int = 1
    downsample_ratio: int = 1
    out_hidden_size: int = 0
    aligner_hidden_size: int = 0
    pos_embed_tokens: int = 0
    deepstack: bool = False
    deepstack_visual_indexes: tuple[int, ...] = ()
    gated_mlp: bool = False
    activation: str = "gelu"
    norm_type: str = "layernorm"
    norm_eps: float = 1e-6
    bias: bool = True
    image_token_id: int = 0
    video_token_id: int = 0
    source: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        indexes = tuple(int(idx) for idx in self.deepstack_visual_indexes)
        object.__setattr__(self, "deepstack_visual_indexes", indexes)
        if indexes and not self.deepstack:
            object.__setattr__(self, "deepstack", True)
        if isinstance(self.source, Mapping):
            object.__setattr__(self, "source", tuple(sorted((str(k), str(v)) for k, v in self.source.items())))

    @property
    def patch_input_width(self) -> int:
        return int(self.in_channels * self.patch_size ** 2 * self.temporal_patch)

    @property
    def aligner_input_width(self) -> int:
        return int(self.hidden_size * self.merge_area)

    @property
    def query_width(self) -> int:
        return self.qkv_width // 3

    @property
    def head_dim(self) -> int:
        return 0 if self.num_heads <= 0 or self.qkv_width <= 0 else self.query_width // self.num_heads

    def merged_tokens(self, patch_tokens: int) -> int:
        return max(1, int(math.ceil(int(patch_tokens) / max(1, self.merge_area))))

    def patch_tokens_for_media(self, media_tokens: int) -> int:
        return max(1, int(media_tokens) * max(1, self.merge_area))

    def validate(self) -> None:
        for name, value in (
            ("layers", self.layers), ("hidden_size", self.hidden_size), ("intermediate_size", self.intermediate_size),
            ("num_heads", self.num_heads), ("qkv_width", self.qkv_width), ("patch_size", self.patch_size),
            ("merge_area", self.merge_area), ("out_hidden_size", self.out_hidden_size),
        ):
            if int(value) <= 0:
                raise ModelingError(f"catalog_vision_field_invalid:{name}")
        if self.qkv_width % 3 != 0 or self.query_width % self.num_heads != 0:
            raise ModelingError(f"catalog_vision_qkv_not_divisible:{self.qkv_width}%3/{self.query_width}%{self.num_heads}")
        for idx in self.deepstack_visual_indexes:
            if idx < 0 or idx >= int(self.layers):
                raise ModelingError(f"catalog_vision_deepstack_index_invalid:{idx}:{self.layers}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "layers": self.layers, "hidden_size": self.hidden_size, "intermediate_size": self.intermediate_size, "num_heads": self.num_heads, "head_dim": self.head_dim,
            "query_width": self.query_width, "qkv_width": self.qkv_width, "qkv_packed_rows": self.qkv_width, "patch_size": self.patch_size, "in_channels": self.in_channels, "temporal_patch": self.temporal_patch,
            "merge_area": self.merge_area, "downsample_ratio": self.downsample_ratio, "out_hidden_size": self.out_hidden_size, "aligner_hidden_size": self.aligner_hidden_size, "aligner_input_width": self.aligner_input_width, "patch_input_width": self.patch_input_width,
            "pos_embed_tokens": self.pos_embed_tokens, "deepstack": self.deepstack, "deepstack_visual_indexes": list(self.deepstack_visual_indexes), "gated_mlp": self.gated_mlp, "activation": self.activation, "norm_type": self.norm_type, "norm_eps": self.norm_eps, "bias": self.bias, "image_token_id": self.image_token_id, "video_token_id": self.video_token_id, "source": dict(self.source),
        }


@lru_cache(None)
def _resolve_quant_category(spec: QuantSpec, cat: str) -> QuantSpec:
    names = [".".join(cat.split(".")[:i + 1]) for i in range(len(cat.split(".")))]
    if cat == "mla_absorb":
        names.insert(0, "mla")
    overrides = dict(spec.category_overrides)
    changes = {}
    for name in names:
        changes.update(dict(overrides.get(name, ())))
    if not changes:
        return spec
    if "weight_format" in changes and changes["weight_format"]:
        changes["expert_weight_format"] = changes["weight_format"]
        bf16_matched = {name for name in names if name.split(".", 1)[0] in _BF16_CATEGORIES}
        if bf16_matched:
            changes["explicit_weight_categories"] = tuple(
                sorted(set(spec.explicit_weight_categories) | bf16_matched)
            )
    return replace(spec, **changes, category_overrides=())


@dataclass(frozen=True)
class EngramTableSpec:
    layer_id: int
    table_id: int
    rows: int
    width: int
    source_dtype: str = "fp8_e4m3"
    scale_dtype: str = "fp8_e8m0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id, "table_id": self.table_id, "rows": self.rows,
            "width": self.width, "source_dtype": self.source_dtype, "scale_dtype": self.scale_dtype,
        }


@dataclass(frozen=True)
class EngramSpec:
    tables: tuple[EngramTableSpec, ...]
    max_ngram: int
    heads: int
    width: int
    pad_token_id: int = 0
    no_dedup: bool = True
    storage_mode: str = "device_bf16"
    host_response_bytes: int = 512
    id_bytes: int = 8
    source: tuple[tuple[str, str], ...] = ()

    @property
    def columns_per_table(self) -> int:
        return max(0, (self.max_ngram - 1) * self.heads)

    @property
    def lookups_per_token_per_table(self) -> int:
        return self.columns_per_table

    @property
    def lookups_per_token(self) -> int:
        return self.columns_per_table * len(self.tables)

    def validate(self) -> None:
        if not self.tables:
            raise ModelingError("catalog_engram_tables_missing")
        if self.max_ngram < 2 or self.heads <= 0 or self.width <= 0:
            raise ModelingError("catalog_engram_geometry_invalid")
        for table in self.tables:
            if table.rows <= 0 or table.width != self.width:
                raise ModelingError(f"catalog_engram_table_invalid:{table.layer_id}")

    @property
    def row_bytes(self) -> float:
        width = int(self.width)
        if self.storage_mode == "device_bf16":
            return float(width * 2)
        if self.storage_mode == "device_int8_fp32":
            return float(width + _ceil_div(width, 32) * 4)
        if self.storage_mode == "host_fp8_e8m0":
            return float(width + _ceil_div(width, 32))
        raise ModelingError(f"catalog_engram_storage_mode_unsupported:{self.storage_mode}")

    def table_storage_bytes(self, *, source: bool = False, node_rows: int | None = None) -> float:
        total = 0.0
        for table in self.tables:
            rows = int(math.ceil(table.rows / node_rows)) if node_rows is not None and node_rows > 1 else table.rows
            if source or self.storage_mode not in {"device_bf16", "device_int8_fp32"}:
                total += rows * (table.width + _ceil_div(table.width, 32))
            elif self.storage_mode == "device_bf16":
                total += rows * table.width * 2
            else:
                total += rows * (table.width + _ceil_div(table.width, 32) * 4)
        return float(total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": [table.to_dict() for table in self.tables],
            "max_ngram": self.max_ngram, "heads": self.heads, "width": self.width,
            "columns_per_table": self.columns_per_table, "lookups_per_token_per_table": self.lookups_per_token_per_table,
            "lookups_per_token": self.lookups_per_token, "pad_token_id": self.pad_token_id, "no_dedup": self.no_dedup,
            "storage_mode": self.storage_mode, "row_bytes": self.row_bytes,
            "host_response_bytes": self.host_response_bytes, "id_bytes": self.id_bytes, "source": dict(self.source),
        }


@dataclass(frozen=True)
class AuxiliarySpec:
    """Model-level optional branches kept outside the body layer schedule."""

    mtp: DraftSpec | None = None
    dspark: DraftSpec | None = None
    external_draft_key: str = ""
    standalone_draft: bool = False
    include_vision_weights: bool = True
    notes: tuple[str, ...] = ()

    def enabled_draft(self, mode: str) -> DraftSpec | None:
        if mode in {"dspark", "verify", "draft_verify"}:
            return self.dspark or self.mtp
        return (self.mtp or self.dspark) if mode in {"mtp", "draft"} else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mtp": self.mtp.to_dict() if self.mtp else None,
            "dspark": self.dspark.to_dict() if self.dspark else None,
            "external_draft_key": self.external_draft_key, "standalone_draft": self.standalone_draft,
            "include_vision_weights": self.include_vision_weights, "notes": list(self.notes),
        }


def merge_source(base: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
    items = {str(k): str(v) for k, v in dict(base).items() if v is not None}
    if extra:
        items.update({str(k): str(v) for k, v in dict(extra).items() if v is not None})
    return tuple(sorted(items.items()))


__all__ = ["QuantSpec", "VisionSpec", "DraftSpec", "EngramTableSpec", "EngramSpec", "AuxiliarySpec", "merge_source"]
