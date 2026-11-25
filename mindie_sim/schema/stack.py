"""Runtime-agnostic plugin adapter, cache-state ABI, and live scheduling contracts.

[Role] Define immutable L0 contracts at the boundary between Core ``mindie_sim``
  and pluggable inference-runtime adapters: ``WorkloadDomain``, ``WorkUnitKind``,
  typed per-layer physical ``CachePlaneABI`` / ``CacheStateABI``,
  ``RuntimeQuantPlan``, live scheduling contracts (``CaptureBucketRule``,
  ``PiecewiseSplitPoint``, ``StarsSubmitPolicy``, ``StreamOverlapPolicy``,
  ``PostAcceptCommitSpec``, ``PdKvTransferSpec``, ``SchedulingContract``),
  ``StackLoweringPlan``, ``LaunchConfigCodec``, and ``RuntimeStackAdapter``.
[Invariants] Zero inference-engine implementation names or imports reside here.
  ``RuntimeQuantPlan`` cache fields are strictly typed ``tuple[CacheStateABI, ...]``
  with unique ``producer_key`` ownership. ``plan_iteration`` preserves one-step
  live scheduling semantics returning ``IterationPlan | WaitForEvent | SchedulerDone``.
  ``PostAcceptCommitSpec`` requires strictly advancing published state versions.
[Timeouts & Locks] Pure immutable metadata validation and deterministic hashing;
  no I/O or locks.
[Side-Effect Policy] Zero global state or input mutation.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, asdict, dataclass, field, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .checkpoint import (
    CHECKPOINT_ORIGINS,
    CacheQuantRule,
    CheckpointOrigin,
    CheckpointQuantMap,
    TensorQuantRule,
)
from .common import ModelingError, dtype_bytes
from .datapath import SourceRef
from .hardware import Hardware
from .model import LayerPlan, ModelSpec
from .operators import (
    KERNEL_READINESS_VALUES,
    OWNER_KINDS,
    KernelReadiness,
    OwnerKind,
    TensorPrecisionABI,
)
from .parallel import ParallelSpec, local_kv_heads
from .speculative import SpeculativeSpec
from .workload import IterationPlan

STACK_ADAPTER_PROTOCOL_VERSION = "mindie-sim.stack-adapter/1"

WorkloadDomain = Literal["llm", "vlm", "diffusion", "custom"]
WORKLOAD_DOMAINS = frozenset({"llm", "vlm", "diffusion", "custom"})

WorkUnitKind = Literal[
    "prefill_decode_step",
    "denoise_timestep",
    "vae_decode",
    "text_encode",
    "vision_encode",
    "custom_step",
]
WORK_UNIT_KINDS = frozenset({
    "prefill_decode_step",
    "denoise_timestep",
    "vae_decode",
    "text_encode",
    "vision_encode",
    "custom_step",
})

StackTrack = Literal["eager", "shipping", "proposed", "target"]
STACK_TRACKS = frozenset({"eager", "shipping", "proposed", "target"})

ManifestBindingState = Literal["merged", "unmerged_pr"]
MANIFEST_BINDING_STATES = frozenset({"merged", "unmerged_pr"})

StackOptimizationMode = Literal["simulation", "modeling"]
STACK_OPTIMIZATION_MODES = frozenset({"simulation", "modeling"})

SYNTHESIS_READINESS_CLASSES = frozenset({"synthesized_target", "tiler_extension"})

RewriteTrack = Literal["config", "framework", "kernel_new", "kernel_tune"]
REWRITE_TRACKS = frozenset({"config", "framework", "kernel_new", "kernel_tune"})

FrameworkRewriteKind = Literal[
    "merge_sibling_projections",
    "bind_shipping_op",
    "multistream_overlap",
    "eliminate_graph_break",
]
FRAMEWORK_REWRITE_KINDS = frozenset({
    "merge_sibling_projections",
    "bind_shipping_op",
    "multistream_overlap",
    "eliminate_graph_break",
})

CacheScope = Literal["target", "draft"]
CACHE_SCOPES = frozenset({"target", "draft"})

CacheComponentRole = Literal[
    "kv_k",
    "kv_v",
    "kv_combined",
    "mla_latent",
    "mla_rotary",
    "window_kv",
    "compressed_kv",
    "index_k",
    "index_scale",
    "index_tail",
    "compressor_state",
    "conv_history",
    "recurrent_state",
    "recurrent_checkpoint",
    "replay_record",
    "transient_state",
    "engram_table",
    "ple_conv_state",
]
CACHE_COMPONENT_ROLES = frozenset({
    "kv_k", "kv_v", "kv_combined", "mla_latent", "mla_rotary",
    "window_kv", "compressed_kv", "index_k", "index_scale", "index_tail",
    "compressor_state", "conv_history", "recurrent_state",
    "recurrent_checkpoint", "replay_record", "transient_state",
    "engram_table", "ple_conv_state",
})

CachePlaneKind = Literal["payload", "scale", "offset", "suffix", "padding", "page_table"]
CACHE_PLANE_KINDS = frozenset({"payload", "scale", "offset", "suffix", "padding", "page_table"})

AllocationFrequency = Literal[
    "checkpoint", "layer", "page", "token_group", "token", "request_slot",
]
ALLOCATION_FREQUENCIES = frozenset({
    "checkpoint", "layer", "page", "token_group", "token", "request_slot",
})

WriteTrigger = Literal[
    "load", "page_assign", "page_reassign", "append", "verify", "accepted_commit",
]
WRITE_TRIGGERS = frozenset({
    "load", "page_assign", "page_reassign", "append", "verify", "accepted_commit",
})

MemoryTier = Literal["hbm", "dram", "ssd"]
MEMORY_TIERS = frozenset({"hbm", "dram", "ssd"})

TokenOwnershipAxis = Literal["replicated", "cp_striped", "cp_contiguous", "dcp_head", "none"]
TOKEN_OWNERSHIP_AXES = frozenset({"replicated", "cp_striped", "cp_contiguous", "dcp_head", "none"})

CaptureMode = Literal["none", "piecewise", "full", "full_decode_only", "full_and_piecewise"]
CAPTURE_MODES = frozenset({"none", "piecewise", "full", "full_decode_only", "full_and_piecewise"})

CaptureUnit = Literal["tokens", "verify_slots", "sequences"]
CAPTURE_UNITS = frozenset({"tokens", "verify_slots", "sequences"})

CaptureFallback = Literal["eager", "piecewise", "reject"]
CAPTURE_FALLBACKS = frozenset({"eager", "piecewise", "reject"})

PdTransferMode = Literal["blocking", "layerwise_overlap"]
PD_TRANSFER_MODES = frozenset({"blocking", "layerwise_overlap"})

PdTransportKind = Literal["direct_d2d", "pooled_dram", "pooled_ssd"]
PD_TRANSPORT_KINDS = frozenset({"direct_d2d", "pooled_dram", "pooled_ssd"})

PreemptionPolicy = Literal["recompute", "swap", "reject"]
PREEMPTION_POLICIES = frozenset({"recompute", "swap", "reject"})


def _req_str(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


def _nonneg_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _pos_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _nonneg_float(value: Any, code: str) -> float:
    if isinstance(value, bool):
        raise ModelingError(f"{code}:{value!r}")
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelingError(f"{code}:{value!r}") from exc
    if not math.isfinite(num) or num < 0.0:
        raise ModelingError(f"{code}:{value!r}")
    return num


def _int_tuple(values: Sequence[int], code: str, *, minimum: int = 0) -> tuple[int, ...]:
    out: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ModelingError(f"{code}:{item!r}")
        out.append(int(item))
    return tuple(out)


def _freeze_pairs(raw: Any, code: str) -> tuple[tuple[str, Any], ...]:
    items = tuple(raw.items()) if isinstance(raw, Mapping) else tuple(raw)
    out: list[tuple[str, Any]] = []
    for entry in items:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ModelingError(code)
        out.append((str(entry[0]), _freeze_val(entry[1], code)))
    return tuple(out)


def _freeze_val(value: Any, code: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze_val(v, code)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_val(item, code) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_val(item, code) for item in sorted(value, key=repr))
    raise ModelingError(f"{code}:{type(value).__name__}")


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachePlaneABI:
    """Physical layout and lifecycle of one cache or state storage plane."""

    plane_id: str
    plane_kind: CachePlaneKind
    logical_axes: tuple[str, ...]
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    dtype: str
    packed_elements: int = 1
    layout: str = "ND"
    byte_strides: tuple[int, ...] = ()
    group_axes: tuple[int, ...] = ()
    group_extents: tuple[int, ...] = ()
    scale_dtype: str = ""
    offset_dtype: str = ""
    suffix_shape: tuple[int, ...] = ()
    suffix_dtype: str = ""
    offset_in_page: int = 0
    alignment: int = 1
    page_pitch: int = 0
    allocation_frequency: AllocationFrequency = "page"
    write_trigger: WriteTrigger = "append"
    bytes_per_unit: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane_id", _req_str(self.plane_id, "cache_plane_id_required"))
        if self.plane_kind not in CACHE_PLANE_KINDS:
            raise ModelingError(f"cache_plane_kind_invalid:{self.plane_kind!r}")
        if self.allocation_frequency not in ALLOCATION_FREQUENCIES:
            raise ModelingError(f"cache_plane_frequency_invalid:{self.allocation_frequency!r}")
        if self.write_trigger not in WRITE_TRIGGERS:
            raise ModelingError(f"cache_plane_trigger_invalid:{self.write_trigger!r}")
        axes = tuple(str(ax) for ax in self.logical_axes)
        lshape = _int_tuple(self.logical_shape, "cache_plane_logical_shape_invalid", minimum=0)
        pshape = _int_tuple(self.physical_shape, "cache_plane_physical_shape_invalid", minimum=0)
        if axes and lshape and len(axes) != len(lshape):
            raise ModelingError(f"cache_plane_axes_shape_mismatch:{self.plane_id}")
        gaxes = _int_tuple(self.group_axes, "cache_plane_group_axes_invalid", minimum=0)
        gext = _int_tuple(self.group_extents, "cache_plane_group_extents_invalid", minimum=1)
        if len(gaxes) != len(gext):
            raise ModelingError(f"cache_plane_group_mismatch:{self.plane_id}")
        object.__setattr__(self, "logical_axes", axes)
        object.__setattr__(self, "logical_shape", lshape)
        object.__setattr__(self, "physical_shape", pshape)
        object.__setattr__(self, "dtype", _req_str(self.dtype, "cache_plane_dtype_required").lower())
        object.__setattr__(self, "packed_elements", _pos_int(self.packed_elements, "cache_plane_packed_invalid"))
        object.__setattr__(self, "layout", _req_str(self.layout, "cache_plane_layout_required"))
        object.__setattr__(self, "byte_strides", _int_tuple(self.byte_strides, "cache_plane_strides_invalid"))
        object.__setattr__(self, "group_axes", gaxes)
        object.__setattr__(self, "group_extents", gext)
        object.__setattr__(self, "suffix_shape", _int_tuple(self.suffix_shape, "cache_plane_suffix_shape_invalid"))
        object.__setattr__(self, "offset_in_page", _nonneg_int(self.offset_in_page, "cache_plane_offset_invalid"))
        object.__setattr__(self, "alignment", _pos_int(self.alignment, "cache_plane_alignment_invalid"))
        object.__setattr__(self, "page_pitch", _nonneg_int(self.page_pitch, "cache_plane_pitch_invalid"))
        object.__setattr__(self, "bytes_per_unit", _nonneg_int(self.bytes_per_unit, "cache_plane_bytes_invalid"))

        if self.packed_elements != 1 and dtype_bytes(self.dtype) != 1:
            raise ModelingError("cache_packed_elements_requires_byte_container")
        if self.byte_strides and len(self.byte_strides) != len(pshape):
            raise ModelingError(f"cache_plane_stride_rank_mismatch:{self.plane_id}")
        if self.byte_strides and self.bytes_per_unit and self.bytes_per_unit < self._backing_span():
            raise ModelingError(f"cache_plane_allocation_too_small:{self.plane_id}")

    def _backing_span(self) -> int:
        shape = self.physical_shape
        if not shape or 0 in shape:
            return 0
        numerator, denominator = dtype_bytes(self.dtype).as_integer_ratio()
        if self.byte_strides:
            # The innermost packed axis addresses byte containers.
            containers = (shape[-1] + self.packed_elements - 1) // self.packed_elements
            extents = (*shape[:-1], containers)
            last = sum((n - 1) * stride for n, stride in zip(extents, self.byte_strides))
            return last + (numerator + denominator - 1) // denominator
        denominator *= self.packed_elements
        return (math.prod(shape) * numerator + denominator - 1) // denominator

    def exact_bytes(self) -> int:
        """Backing span of one allocation unit, including declared strided gaps."""
        return self.bytes_per_unit or self._backing_span()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane_id": self.plane_id,
            "plane_kind": self.plane_kind,
            "logical_axes": list(self.logical_axes),
            "logical_shape": list(self.logical_shape),
            "physical_shape": list(self.physical_shape),
            "dtype": self.dtype,
            "packed_elements": self.packed_elements,
            "layout": self.layout,
            "byte_strides": list(self.byte_strides),
            "group_axes": list(self.group_axes),
            "group_extents": list(self.group_extents),
            "scale_dtype": self.scale_dtype,
            "offset_dtype": self.offset_dtype,
            "suffix_shape": list(self.suffix_shape),
            "suffix_dtype": self.suffix_dtype,
            "offset_in_page": self.offset_in_page,
            "alignment": self.alignment,
            "page_pitch": self.page_pitch,
            "allocation_frequency": self.allocation_frequency,
            "write_trigger": self.write_trigger,
            "bytes_per_unit": self.exact_bytes(),
        }


@dataclass(frozen=True, slots=True)
class CachePlacementRule:
    """Per-component physical placement, head range, token ownership, and replication."""

    pp_stage: int = 0
    rank_ids: tuple[int, ...] = ()
    die_ids: tuple[int, ...] = ()
    memory_tier: MemoryTier = "hbm"
    head_range: tuple[int, int] | None = None
    token_ownership_axis: TokenOwnershipAxis = "replicated"
    replica_group: tuple[int, ...] = ()
    replication_factor: int = 1
    state_shard_axis: str = ""

    def __post_init__(self) -> None:
        if self.memory_tier not in MEMORY_TIERS:
            raise ModelingError(f"cache_placement_tier_invalid:{self.memory_tier!r}")
        if self.token_ownership_axis not in TOKEN_OWNERSHIP_AXES:
            raise ModelingError(f"cache_placement_token_axis_invalid:{self.token_ownership_axis!r}")
        object.__setattr__(self, "pp_stage", _nonneg_int(self.pp_stage, "cache_placement_pp_stage_invalid"))
        object.__setattr__(self, "rank_ids", _int_tuple(self.rank_ids, "cache_placement_ranks_invalid"))
        object.__setattr__(self, "die_ids", _int_tuple(self.die_ids, "cache_placement_dies_invalid"))
        object.__setattr__(self, "replica_group", _int_tuple(self.replica_group, "cache_placement_replica_invalid"))
        object.__setattr__(
            self, "replication_factor", _pos_int(self.replication_factor, "cache_placement_replication_invalid")
        )
        if self.head_range is not None:
            hr = _int_tuple(self.head_range, "cache_placement_head_range_invalid")
            if len(hr) != 2 or hr[1] < hr[0]:
                raise ModelingError(f"cache_placement_head_range_bounds_invalid:{hr!r}")
            object.__setattr__(self, "head_range", (hr[0], hr[1]))
        if not isinstance(self.state_shard_axis, str):
            raise ModelingError("cache_placement_shard_axis_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pp_stage": self.pp_stage,
            "rank_ids": list(self.rank_ids),
            "die_ids": list(self.die_ids),
            "memory_tier": self.memory_tier,
            "head_range": list(self.head_range) if self.head_range is not None else None,
            "token_ownership_axis": self.token_ownership_axis,
            "replica_group": list(self.replica_group),
            "replication_factor": self.replication_factor,
            "state_shard_axis": self.state_shard_axis,
        }


@dataclass(frozen=True, slots=True)
class CacheStateABI:
    """Per-native-layer physical storage ABI for KV, Indexer, and recurrent state."""

    scope: CacheScope
    native_layer: int
    component_role: CacheComponentRole
    producer_key: str
    planes: tuple[CachePlaneABI, ...]
    consumer_keys: tuple[str, ...] = ()
    phase: str = "decode"
    deployment_role: str = "colocated"
    checkpoint_digest: str = ""
    recipe_abi_id: str = ""
    source_revision: str = ""
    kv_block_size: int = 0
    kernel_block_size: int = 0
    cp_interleave_size: int = 0
    window: int | None = None
    compression_ratio: int = 1
    pooling_ratio: int = 1
    tail_slots: int = 0
    draft_context_cap: int | None = None
    history_slots: int = 0
    rollback_slots: int = 0
    checkpoint_slots: int = 0
    record_shape: tuple[int, ...] = ()
    record_dtype: str = ""
    page_group: str = ""
    overlay_region: str = ""
    alias_of: str = ""
    placement: CachePlacementRule = field(default_factory=CachePlacementRule)

    def __post_init__(self) -> None:
        if self.scope not in CACHE_SCOPES:
            raise ModelingError(f"cache_state_scope_invalid:{self.scope!r}")
        if self.component_role not in CACHE_COMPONENT_ROLES:
            raise ModelingError(f"cache_state_role_invalid:{self.component_role!r}")
        object.__setattr__(self, "native_layer", _nonneg_int(self.native_layer, "cache_state_native_layer_invalid"))
        object.__setattr__(self, "producer_key", _req_str(self.producer_key, "cache_state_producer_key_required"))
        planes = tuple(self.planes)
        if not planes or not all(isinstance(plane, CachePlaneABI) for plane in planes):
            raise ModelingError(f"cache_state_planes_invalid:{self.producer_key}")
        if len({plane.plane_id for plane in planes}) != len(planes):
            raise ModelingError(f"cache_state_duplicate_plane_id:{self.producer_key}")
        object.__setattr__(self, "planes", planes)
        consumers = tuple(str(k).strip() for k in self.consumer_keys if str(k).strip())
        object.__setattr__(self, "consumer_keys", consumers)
        object.__setattr__(self, "kv_block_size", _nonneg_int(self.kv_block_size, "cache_state_kv_block_invalid"))
        object.__setattr__(
            self, "kernel_block_size", _nonneg_int(self.kernel_block_size, "cache_state_kernel_block_invalid")
        )
        object.__setattr__(
            self, "cp_interleave_size", _nonneg_int(self.cp_interleave_size, "cache_state_cp_interleave_invalid")
        )
        if self.window is not None:
            object.__setattr__(self, "window", _pos_int(self.window, "cache_state_window_invalid"))
        object.__setattr__(
            self, "compression_ratio", _pos_int(self.compression_ratio, "cache_state_compression_invalid")
        )
        object.__setattr__(self, "pooling_ratio", _pos_int(self.pooling_ratio, "cache_state_pooling_invalid"))
        object.__setattr__(self, "tail_slots", _nonneg_int(self.tail_slots, "cache_state_tail_slots_invalid"))
        if self.draft_context_cap is not None:
            object.__setattr__(
                self, "draft_context_cap", _pos_int(self.draft_context_cap, "cache_state_draft_cap_invalid")
            )
        object.__setattr__(self, "history_slots", _nonneg_int(self.history_slots, "cache_state_history_invalid"))
        object.__setattr__(self, "rollback_slots", _nonneg_int(self.rollback_slots, "cache_state_rollback_invalid"))
        object.__setattr__(
            self, "checkpoint_slots", _nonneg_int(self.checkpoint_slots, "cache_state_checkpoint_slots_invalid")
        )
        object.__setattr__(self, "record_shape", _int_tuple(self.record_shape, "cache_state_record_shape_invalid"))
        if not isinstance(self.placement, CachePlacementRule):
            raise ModelingError("cache_state_placement_invalid")

    def total_plane_bytes(self, *, include_padding: bool = True) -> int:
        """Exact integer bytes summed across declared planes."""
        return sum(
            plane.exact_bytes()
            for plane in self.planes
            if include_padding or plane.plane_kind != "padding"
        )

    def compressed_append_units(self, context_tokens: int, query_tokens: int) -> int:
        """Exact integer compressed/pooled append count: floor((c+q)/r) - floor(c/r)."""
        ctx = _nonneg_int(context_tokens, "cache_state_context_invalid")
        qry = _nonneg_int(query_tokens, "cache_state_query_invalid")
        factor = max(self.compression_ratio, self.pooling_ratio)
        return ((ctx + qry) // factor) - (ctx // factor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "native_layer": self.native_layer,
            "component_role": self.component_role,
            "producer_key": self.producer_key,
            "consumer_keys": list(self.consumer_keys),
            "phase": self.phase,
            "deployment_role": self.deployment_role,
            "checkpoint_digest": self.checkpoint_digest,
            "recipe_abi_id": self.recipe_abi_id,
            "source_revision": self.source_revision,
            "planes": [plane.to_dict() for plane in self.planes],
            "kv_block_size": self.kv_block_size,
            "kernel_block_size": self.kernel_block_size,
            "cp_interleave_size": self.cp_interleave_size,
            "window": self.window,
            "compression_ratio": self.compression_ratio,
            "pooling_ratio": self.pooling_ratio,
            "tail_slots": self.tail_slots,
            "draft_context_cap": self.draft_context_cap,
            "history_slots": self.history_slots,
            "rollback_slots": self.rollback_slots,
            "checkpoint_slots": self.checkpoint_slots,
            "record_shape": list(self.record_shape),
            "record_dtype": self.record_dtype,
            "page_group": self.page_group,
            "overlay_region": self.overlay_region,
            "alias_of": self.alias_of,
            "placement": self.placement.to_dict(),
        }

    def digest(self) -> str:
        return _sha256_payload(self.to_dict())


@dataclass(frozen=True, slots=True)
class PreparedWeightSpec:
    """Runtime-synthesized or post-load repacked weight copy (0 checkpoint bytes, real HBM)."""

    allocation_id: str
    source_tensors: tuple[str, ...]
    transform: str
    scope: str
    layer_indices: tuple[int, ...]
    local_shape: tuple[int, ...]
    dtype: str
    layout: str
    extra_metadata_bytes: int = 0
    retains_source_tensors: bool = True
    alias_of: str = ""
    lifetime: Literal["persistent", "load_peak_only"] = "persistent"

    def __post_init__(self) -> None:
        object.__setattr__(self, "allocation_id", _req_str(self.allocation_id, "prepared_weight_id_required"))
        object.__setattr__(self, "transform", _req_str(self.transform, "prepared_weight_transform_required"))
        object.__setattr__(self, "scope", _req_str(self.scope, "prepared_weight_scope_required"))
        object.__setattr__(self, "dtype", _req_str(self.dtype, "prepared_weight_dtype_required").lower())
        object.__setattr__(self, "layout", _req_str(self.layout, "prepared_weight_layout_required"))
        object.__setattr__(
            self, "source_tensors", tuple(_req_str(t, "prepared_weight_source_invalid") for t in self.source_tensors)
        )
        object.__setattr__(self, "layer_indices", _int_tuple(self.layer_indices, "prepared_weight_layers_invalid"))
        object.__setattr__(self, "local_shape", _int_tuple(self.local_shape, "prepared_weight_shape_invalid"))
        object.__setattr__(
            self, "extra_metadata_bytes", _nonneg_int(self.extra_metadata_bytes, "prepared_weight_meta_bytes_invalid")
        )
        if self.lifetime not in {"persistent", "load_peak_only"}:
            raise ModelingError(f"prepared_weight_lifetime_invalid:{self.lifetime!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "source_tensors": list(self.source_tensors),
            "transform": self.transform,
            "scope": self.scope,
            "layer_indices": list(self.layer_indices),
            "local_shape": list(self.local_shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "extra_metadata_bytes": self.extra_metadata_bytes,
            "retains_source_tensors": self.retains_source_tensors,
            "alias_of": self.alias_of,
            "lifetime": self.lifetime,
        }


@dataclass(frozen=True, slots=True)
class RuntimeQuantPlan:
    """Stack-instantiated per-port execution ABI, typed cache ABIs, and prepared weights."""

    checkpoint_digest: str
    stack_id: str
    port_abis: tuple[tuple[tuple[str, int, str], TensorPrecisionABI], ...] = ()
    packed_groups: tuple[tuple[str, tuple[str, ...], str], ...] = ()
    kv_cache_abi: tuple[CacheStateABI, ...] = ()
    index_cache_abi: tuple[CacheStateABI, ...] = ()
    recurrent_state_abi: tuple[CacheStateABI, ...] = ()
    prepared_weights: tuple[PreparedWeightSpec, ...] = ()
    dispositions: tuple[tuple[str, str, str], ...] = ()
    provenance: CheckpointOrigin = "verified_checkpoint_descriptor"
    expected_plan_digest: InitVar[str] = ""
    plan_digest: str = field(init=False)

    def __post_init__(self, expected_plan_digest: str) -> None:
        object.__setattr__(self, "stack_id", _req_str(self.stack_id, "runtime_quant_plan_stack_id_required"))
        if not isinstance(self.checkpoint_digest, str):
            raise ModelingError("runtime_quant_plan_checkpoint_digest_invalid")
        if self.provenance not in CHECKPOINT_ORIGINS:
            raise ModelingError(f"runtime_quant_plan_provenance_invalid:{self.provenance!r}")
        ports: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
        for entry in self.port_abis:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ModelingError("runtime_quant_plan_port_entry_invalid")
            coord, abi = entry
            if (
                not isinstance(coord, (tuple, list))
                or len(coord) != 3
                or not isinstance(abi, TensorPrecisionABI)
            ):
                raise ModelingError("runtime_quant_plan_port_abi_invalid")
            ports.append(((str(coord[0]), _nonneg_int(coord[1], "port_layer_invalid"), str(coord[2])), abi))
        object.__setattr__(self, "port_abis", tuple(ports))
        for field_name in ("kv_cache_abi", "index_cache_abi", "recurrent_state_abi"):
            items = tuple(getattr(self, field_name))
            if not all(isinstance(item, CacheStateABI) for item in items):
                raise ModelingError(f"runtime_quant_plan_{field_name}_must_be_typed_cache_state_abi")
            object.__setattr__(self, field_name, items)
        seen_owners: set[str] = set()
        for abi in (*self.kv_cache_abi, *self.index_cache_abi, *self.recurrent_state_abi):
            if abi.producer_key in seen_owners:
                raise ModelingError(f"runtime_quant_plan_duplicate_cache_owner:{abi.producer_key}")
            seen_owners.add(abi.producer_key)
        prepared = tuple(self.prepared_weights)
        if not all(isinstance(item, PreparedWeightSpec) for item in prepared):
            raise ModelingError("runtime_quant_plan_prepared_weights_invalid")
        object.__setattr__(self, "prepared_weights", prepared)
        object.__setattr__(
            self,
            "packed_groups",
            tuple((str(name), tuple(str(r) for r in roles), str(sig)) for name, roles, sig in self.packed_groups),
        )
        object.__setattr__(
            self,
            "dispositions",
            tuple((str(a), str(b), str(c)) for a, b, c in self.dispositions),
        )
        digest = self._compute_digest()
        if expected_plan_digest and expected_plan_digest != digest:
            raise ModelingError("runtime_quant_plan_digest_mismatch")
        object.__setattr__(self, "plan_digest", digest)

    def _compute_digest(self) -> str:
        return _sha256_payload({
            "checkpoint_digest": self.checkpoint_digest,
            "stack_id": self.stack_id,
            "port_abis": [[list(coord), abi.abi_signature] for coord, abi in self.port_abis],
            "packed_groups": [[n, list(r), s] for n, r, s in self.packed_groups],
            "kv_cache_abi": [item.digest() for item in self.kv_cache_abi],
            "index_cache_abi": [item.digest() for item in self.index_cache_abi],
            "recurrent_state_abi": [item.digest() for item in self.recurrent_state_abi],
            "prepared_weights": [item.to_dict() for item in self.prepared_weights],
            "dispositions": self.dispositions,
            "provenance": self.provenance,
        })

    def all_cache_abis(self) -> tuple[CacheStateABI, ...]:
        return (*self.kv_cache_abi, *self.index_cache_abi, *self.recurrent_state_abi)

    def cache_abi_for(self, producer_key: str) -> CacheStateABI | None:
        for abi in self.all_cache_abis():
            if abi.producer_key == producer_key:
                return abi
        return None


@dataclass(frozen=True, slots=True)
class CaptureBucketRule:
    """Non-lossy graph-capture eligibility, bucket sizing, and fallback rule."""

    mode: CaptureMode
    units: CaptureUnit = "tokens"
    full_keys: tuple[int, ...] = ()
    piecewise_keys: tuple[int, ...] = ()
    verify_stride: int = 1
    max_slots: int | None = None
    allow_mixed: bool = False
    allow_ragged: bool = False
    fallback_mode: CaptureFallback = "eager"
    enforce_eager_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in CAPTURE_MODES:
            raise ModelingError(f"capture_bucket_mode_invalid:{self.mode!r}")
        if self.units not in CAPTURE_UNITS:
            raise ModelingError(f"capture_bucket_units_invalid:{self.units!r}")
        if self.fallback_mode not in CAPTURE_FALLBACKS:
            raise ModelingError(f"capture_bucket_fallback_invalid:{self.fallback_mode!r}")
        object.__setattr__(
            self, "full_keys", tuple(sorted(set(_int_tuple(self.full_keys, "capture_full_keys_invalid", minimum=1))))
        )
        object.__setattr__(
            self,
            "piecewise_keys",
            tuple(sorted(set(_int_tuple(self.piecewise_keys, "capture_piecewise_keys_invalid", minimum=1)))),
        )
        object.__setattr__(self, "verify_stride", _pos_int(self.verify_stride, "capture_verify_stride_invalid"))
        if self.max_slots is not None:
            object.__setattr__(self, "max_slots", _pos_int(self.max_slots, "capture_max_slots_invalid"))
        object.__setattr__(
            self, "enforce_eager_scopes", tuple(str(s).strip() for s in self.enforce_eager_scopes if str(s).strip())
        )

    def select_bucket(
        self,
        *,
        phase: str,
        active_tokens: int,
        is_ragged: bool = False,
        is_mixed: bool = False,
        scope: str = "target",
    ) -> tuple[int | None, Literal["full", "piecewise", "none"]]:
        """Resolve exact capture bucket and scope without collapsing FULL_AND_PIECEWISE."""
        if self.mode == "none" or scope in self.enforce_eager_scopes:
            return None, "none"
        tokens = _pos_int(active_tokens, "capture_active_tokens_invalid")
        if self.max_slots is not None and tokens > self.max_slots:
            return self._fallback(tokens, is_mixed=is_mixed)
        full_eligible = self.mode in {"full", "full_decode_only", "full_and_piecewise"}
        if self.mode in {"full_decode_only", "full_and_piecewise"} and phase != "decode":
            full_eligible = False
        if (is_ragged and not self.allow_ragged) or (is_mixed and not self.allow_mixed):
            full_eligible = False
        if full_eligible:
            bucket = next(
                (k for k in self.full_keys if k >= tokens and k % self.verify_stride == 0),
                None,
            )
            if bucket is not None:
                return bucket, "full"
        if self.mode in {"piecewise", "full_and_piecewise"}:
            if not is_mixed or self.allow_mixed:
                pw = next((k for k in self.piecewise_keys if k >= tokens), None)
                if pw is not None:
                    return pw, "piecewise"
        return self._fallback(tokens, is_mixed=is_mixed)

    def _fallback(
        self, tokens: int, *, is_mixed: bool
    ) -> tuple[int | None, Literal["full", "piecewise", "none"]]:
        if self.fallback_mode == "piecewise" and (not is_mixed or self.allow_mixed):
            pw = next((k for k in self.piecewise_keys if k >= tokens), None)
            if pw is not None:
                return pw, "piecewise"
        if self.fallback_mode == "reject":
            raise ModelingError(f"capture_bucket_unavailable:mode={self.mode}:tokens={tokens}")
        return None, "none"


@dataclass(frozen=True, slots=True)
class PiecewiseSplitPoint:
    """Stable occurrence-bound split point dividing captured segments and eager islands."""

    call_or_occurrence_id: str
    scope: str = "target"
    stage: int = 0
    segment_index: int = 0
    reason: str = ""
    selected_key: int | None = None
    physical_rows: int = 0
    active_mask_count: int = 0
    host_launch_id: str = ""
    eager_island: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "call_or_occurrence_id", _req_str(self.call_or_occurrence_id, "piecewise_split_id_required")
        )
        object.__setattr__(self, "stage", _nonneg_int(self.stage, "piecewise_split_stage_invalid"))
        object.__setattr__(self, "segment_index", _nonneg_int(self.segment_index, "piecewise_split_segment_invalid"))
        object.__setattr__(self, "physical_rows", _nonneg_int(self.physical_rows, "piecewise_split_rows_invalid"))
        object.__setattr__(
            self, "active_mask_count", _nonneg_int(self.active_mask_count, "piecewise_split_mask_invalid")
        )
        if self.selected_key is not None:
            object.__setattr__(self, "selected_key", _pos_int(self.selected_key, "piecewise_split_key_invalid"))


@dataclass(frozen=True, slots=True)
class StarsSubmitPolicy:
    """Per-occurrence STARS descriptor queue and host/replay submission contract."""

    controller_scope: Literal["die", "chip", "node"] = "die"
    die_id: int = 0
    stream_id: str = "main"
    owner_kind: OwnerKind = "aic"
    fifo_lanes: int = 1
    queue_depth: int = 1
    issue_cadence_ns: float = 0.0
    first_issue_cost_ns: float = 0.0
    predecessor_policy: Literal["stream_fifo", "explicit_deps", "barrier"] = "stream_fifo"
    credit_release_on: Literal["kernel_complete", "stage_complete", "replay_complete"] = "kernel_complete"
    replay_mode: Literal["eager", "captured_segment", "full_graph"] = "eager"

    def __post_init__(self) -> None:
        if self.controller_scope not in {"die", "chip", "node"}:
            raise ModelingError(f"stars_submit_controller_scope_invalid:{self.controller_scope!r}")
        if self.owner_kind not in OWNER_KINDS:
            raise ModelingError(f"stars_submit_owner_kind_invalid:{self.owner_kind!r}")
        if self.predecessor_policy not in {"stream_fifo", "explicit_deps", "barrier"}:
            raise ModelingError(f"stars_submit_predecessor_policy_invalid:{self.predecessor_policy!r}")
        if self.credit_release_on not in {"kernel_complete", "stage_complete", "replay_complete"}:
            raise ModelingError(f"stars_submit_credit_release_invalid:{self.credit_release_on!r}")
        if self.replay_mode not in {"eager", "captured_segment", "full_graph"}:
            raise ModelingError(f"stars_submit_replay_mode_invalid:{self.replay_mode!r}")
        object.__setattr__(self, "die_id", _nonneg_int(self.die_id, "stars_submit_die_id_invalid"))
        object.__setattr__(self, "stream_id", _req_str(self.stream_id, "stars_submit_stream_id_required"))
        object.__setattr__(self, "fifo_lanes", _pos_int(self.fifo_lanes, "stars_submit_fifo_lanes_invalid"))
        object.__setattr__(self, "queue_depth", _pos_int(self.queue_depth, "stars_submit_queue_depth_invalid"))
        object.__setattr__(
            self, "issue_cadence_ns", _nonneg_float(self.issue_cadence_ns, "stars_submit_cadence_invalid")
        )
        object.__setattr__(
            self, "first_issue_cost_ns", _nonneg_float(self.first_issue_cost_ns, "stars_submit_first_issue_invalid")
        )


@dataclass(frozen=True, slots=True)
class StreamLaneBudget:
    """Per-die stream lane core budget and isolation mode."""

    stream_id: str
    die_id: int = 0
    aic_budget: int | None = None
    aiv_budget: int | None = None
    isolation: Literal["disjoint", "shared_cap"] = "disjoint"
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _req_str(self.stream_id, "stream_lane_id_required"))
        object.__setattr__(self, "die_id", _nonneg_int(self.die_id, "stream_lane_die_invalid"))
        if self.aic_budget is not None:
            object.__setattr__(self, "aic_budget", _nonneg_int(self.aic_budget, "stream_lane_aic_invalid"))
        if self.aiv_budget is not None:
            object.__setattr__(self, "aiv_budget", _nonneg_int(self.aiv_budget, "stream_lane_aiv_invalid"))
        if self.isolation not in {"disjoint", "shared_cap"}:
            raise ModelingError(f"stream_lane_isolation_invalid:{self.isolation!r}")
        object.__setattr__(self, "priority", _nonneg_int(self.priority, "stream_lane_priority_invalid"))


@dataclass(frozen=True, slots=True)
class StreamSyncEdge:
    """Cross-stream event record/wait synchronization edge."""

    event_id: str
    record_stream: str
    record_after: str
    wait_stream: str
    wait_before: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _req_str(self.event_id, "stream_sync_event_id_required"))
        object.__setattr__(self, "record_stream", _req_str(self.record_stream, "stream_sync_record_stream_required"))
        object.__setattr__(self, "record_after", _req_str(self.record_after, "stream_sync_record_after_required"))
        object.__setattr__(self, "wait_stream", _req_str(self.wait_stream, "stream_sync_wait_stream_required"))
        waits = tuple(_req_str(w, "stream_sync_wait_before_invalid") for w in self.wait_before)
        if not waits:
            raise ModelingError(f"stream_sync_wait_before_required:{self.event_id}")
        object.__setattr__(self, "wait_before", waits)


@dataclass(frozen=True, slots=True)
class StreamOverlapPolicy:
    """Per-die stream lanes, sync edges, joins, and service terminal."""

    lanes: tuple[StreamLaneBudget, ...] = ()
    sync_edges: tuple[StreamSyncEdge, ...] = ()
    joins: tuple[tuple[str, tuple[str, ...]], ...] = ()
    service_terminal: str = ""
    enabled_overlaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lanes = tuple(self.lanes)
        if not all(isinstance(lane, StreamLaneBudget) for lane in lanes):
            raise ModelingError("stream_overlap_lanes_invalid")
        edges = tuple(self.sync_edges)
        if not all(isinstance(edge, StreamSyncEdge) for edge in edges):
            raise ModelingError("stream_overlap_edges_invalid")
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "sync_edges", edges)
        object.__setattr__(
            self, "joins", tuple((str(name), tuple(str(dep) for dep in deps)) for name, deps in self.joins)
        )
        object.__setattr__(self, "enabled_overlaps", tuple(str(x) for x in self.enabled_overlaps))


@dataclass(frozen=True, slots=True)
class PostAcceptCommitSpec:
    """RecoverSSM / recurrent post-accept state commit causal contract."""

    recurrent_abi_key: str
    request_id: str
    slot_id: int
    accepted_prefix: int
    base_version: int
    published_version: int
    verify_record_ready_event: str
    acceptance_ready_event: str
    base_version_ready_event: str
    publish_event: str
    dependent_consumer_events: tuple[str, ...] = ()
    crosses_checkpoint_boundary: bool = False
    owner_count: int = 1
    batched_kernel_launches: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "recurrent_abi_key", _req_str(self.recurrent_abi_key, "post_accept_abi_key_required")
        )
        object.__setattr__(self, "request_id", _req_str(self.request_id, "post_accept_request_id_required"))
        object.__setattr__(self, "slot_id", _nonneg_int(self.slot_id, "post_accept_slot_invalid"))
        object.__setattr__(
            self, "accepted_prefix", _nonneg_int(self.accepted_prefix, "post_accept_prefix_invalid")
        )
        base_ver = _nonneg_int(self.base_version, "post_accept_base_version_invalid")
        pub_ver = _nonneg_int(self.published_version, "post_accept_published_version_invalid")
        if pub_ver <= base_ver:
            raise ModelingError(
                f"post_accept_version_must_advance:base={base_ver}:published={pub_ver}"
            )
        object.__setattr__(self, "base_version", base_ver)
        object.__setattr__(self, "published_version", pub_ver)
        for attr in (
            "verify_record_ready_event",
            "acceptance_ready_event",
            "base_version_ready_event",
            "publish_event",
        ):
            object.__setattr__(self, attr, _req_str(getattr(self, attr), f"post_accept_{attr}_required"))
        object.__setattr__(
            self,
            "dependent_consumer_events",
            tuple(_req_str(ev, "post_accept_consumer_event_invalid") for ev in self.dependent_consumer_events),
        )
        object.__setattr__(self, "owner_count", _pos_int(self.owner_count, "post_accept_owner_count_invalid"))
        object.__setattr__(
            self,
            "batched_kernel_launches",
            _pos_int(self.batched_kernel_launches, "post_accept_batched_launches_invalid"),
        )


@dataclass(frozen=True, slots=True)
class PdLayerTransferEdge:
    """One per-layer source-ready -> destination-ready KV/state transfer edge."""

    native_layer: int
    source_ready_event: str
    dest_ready_event: str
    source_abi_key: str
    dest_abi_key: str
    byte_range: tuple[int, int] = (0, 0)
    requires_repack: bool = False
    repack_recipe_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "native_layer", _nonneg_int(self.native_layer, "pd_edge_layer_invalid"))
        object.__setattr__(
            self, "source_ready_event", _req_str(self.source_ready_event, "pd_edge_source_event_required")
        )
        object.__setattr__(
            self, "dest_ready_event", _req_str(self.dest_ready_event, "pd_edge_dest_event_required")
        )
        object.__setattr__(self, "source_abi_key", _req_str(self.source_abi_key, "pd_edge_source_abi_required"))
        object.__setattr__(self, "dest_abi_key", _req_str(self.dest_abi_key, "pd_edge_dest_abi_required"))
        br = _int_tuple(self.byte_range, "pd_edge_byte_range_invalid")
        if len(br) != 2 or br[1] < br[0]:
            raise ModelingError(f"pd_edge_byte_range_bounds_invalid:{br!r}")
        object.__setattr__(self, "byte_range", (br[0], br[1]))
        if self.requires_repack and not self.repack_recipe_id.strip():
            raise ModelingError(f"pd_edge_repack_recipe_required:layer={self.native_layer}")


@dataclass(frozen=True, slots=True)
class PdKvTransferSpec:
    """Explicit disaggregated KV/state handoff schedule (blocking or layerwise overlap)."""

    mode: PdTransferMode
    transport: PdTransportKind = "direct_d2d"
    source_pool_id: str = "prefill"
    dest_pool_id: str = "decode"
    layer_edges: tuple[PdLayerTransferEdge, ...] = ()
    route: str = ""
    receive_credits: int = 1
    conversion_required: bool = False
    conversion_recipe_id: str = ""
    last_reader_event: str = ""
    pooled_tier_policy: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in PD_TRANSFER_MODES:
            raise ModelingError(f"pd_transfer_mode_invalid:{self.mode!r}")
        if self.transport not in PD_TRANSPORT_KINDS:
            raise ModelingError(f"pd_transfer_transport_invalid:{self.transport!r}")
        object.__setattr__(self, "source_pool_id", _req_str(self.source_pool_id, "pd_transfer_source_pool_required"))
        object.__setattr__(self, "dest_pool_id", _req_str(self.dest_pool_id, "pd_transfer_dest_pool_required"))
        edges = tuple(self.layer_edges)
        if not all(isinstance(edge, PdLayerTransferEdge) for edge in edges):
            raise ModelingError("pd_transfer_layer_edges_invalid")
        object.__setattr__(self, "layer_edges", edges)
        object.__setattr__(
            self, "receive_credits", _pos_int(self.receive_credits, "pd_transfer_receive_credits_invalid")
        )
        if self.conversion_required and not self.conversion_recipe_id.strip():
            raise ModelingError("pd_transfer_conversion_recipe_required")
        object.__setattr__(
            self, "pooled_tier_policy", _freeze_pairs(self.pooled_tier_policy, "pd_transfer_pooled_policy_invalid")
        )


@dataclass(frozen=True, slots=True)
class SchedulingContract:
    """Bundled capture, split-point, STARS, stream, RecoverSSM, and PD transfer policy."""

    capture_bucket_rule: CaptureBucketRule | None = None
    piecewise_split_points: tuple[PiecewiseSplitPoint, ...] = ()
    stars_submit_policy: tuple[StarsSubmitPolicy, ...] = ()
    stream_overlap_policy: StreamOverlapPolicy | None = None
    post_accept_commits: tuple[PostAcceptCommitSpec, ...] = ()
    pd_kv_transfer: PdKvTransferSpec | None = None
    chunked_prefill_enabled: bool = True
    preemption_policy: PreemptionPolicy = "recompute"
    async_scheduling: bool = False
    speculative_proposal_count: int = 0
    draft_invocation_count: int = 0

    def __post_init__(self) -> None:
        if self.capture_bucket_rule is not None and not isinstance(self.capture_bucket_rule, CaptureBucketRule):
            raise ModelingError("scheduling_contract_capture_rule_invalid")
        splits = tuple(self.piecewise_split_points)
        if not all(isinstance(sp, PiecewiseSplitPoint) for sp in splits):
            raise ModelingError("scheduling_contract_split_points_invalid")
        stars = tuple(self.stars_submit_policy)
        if not all(isinstance(st, StarsSubmitPolicy) for st in stars):
            raise ModelingError("scheduling_contract_stars_policy_invalid")
        if self.stream_overlap_policy is not None and not isinstance(
            self.stream_overlap_policy, StreamOverlapPolicy
        ):
            raise ModelingError("scheduling_contract_stream_policy_invalid")
        commits = tuple(self.post_accept_commits)
        if not all(isinstance(cm, PostAcceptCommitSpec) for cm in commits):
            raise ModelingError("scheduling_contract_post_accept_commits_invalid")
        if self.pd_kv_transfer is not None and not isinstance(self.pd_kv_transfer, PdKvTransferSpec):
            raise ModelingError("scheduling_contract_pd_transfer_invalid")
        if self.preemption_policy not in PREEMPTION_POLICIES:
            raise ModelingError(f"scheduling_contract_preemption_invalid:{self.preemption_policy!r}")
        object.__setattr__(self, "piecewise_split_points", splits)
        object.__setattr__(self, "stars_submit_policy", stars)
        object.__setattr__(self, "post_accept_commits", commits)
        object.__setattr__(
            self,
            "speculative_proposal_count",
            _nonneg_int(self.speculative_proposal_count, "scheduling_contract_proposal_count_invalid"),
        )
        object.__setattr__(
            self,
            "draft_invocation_count",
            _nonneg_int(self.draft_invocation_count, "scheduling_contract_draft_invocations_invalid"),
        )


@dataclass(frozen=True, slots=True)
class IterationSchedulerState:
    """Live scheduler and memory snapshot consumed by one-step ``plan_iteration``."""

    now_ms: float
    ready_request_ids: tuple[str, ...] = ()
    request_priorities: tuple[tuple[str, int], ...] = ()
    request_progress: tuple[tuple[str, int, int], ...] = ()
    in_flight_completions: tuple[tuple[str, float], ...] = ()
    grant_epochs: tuple[tuple[str, int], ...] = ()
    prefix_hits: tuple[tuple[str, int], ...] = ()
    pool_ready_bytes: int = 0
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 1
    max_num_input_tokens: int | None = None
    chunked_prefill: bool = True
    preemption_policy: PreemptionPolicy = "recompute"
    async_scheduling: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "now_ms", _nonneg_float(self.now_ms, "scheduler_state_now_invalid"))
        object.__setattr__(self, "pool_ready_bytes", _nonneg_int(self.pool_ready_bytes, "scheduler_state_pool_invalid"))
        object.__setattr__(self, "max_num_seqs", _pos_int(self.max_num_seqs, "scheduler_state_max_seqs_invalid"))
        object.__setattr__(
            self, "max_num_batched_tokens", _pos_int(self.max_num_batched_tokens, "scheduler_state_max_tokens_invalid")
        )
        if self.max_num_input_tokens is not None:
            object.__setattr__(
                self, "max_num_input_tokens", _pos_int(self.max_num_input_tokens, "scheduler_state_max_input_invalid")
            )
        if self.preemption_policy not in PREEMPTION_POLICIES:
            raise ModelingError(f"scheduler_state_preemption_invalid:{self.preemption_policy!r}")


@dataclass(frozen=True, slots=True)
class WaitForEvent:
    """Live scheduler response indicating no step can launch until an event or arrival."""

    reason: Literal["arrival", "completion", "grant_release", "pd_transfer", "pool_restore"]
    until_ms: float | None = None
    event_id: str = ""

    def __post_init__(self) -> None:
        if self.reason not in {"arrival", "completion", "grant_release", "pd_transfer", "pool_restore"}:
            raise ModelingError(f"wait_for_event_reason_invalid:{self.reason!r}")
        if self.until_ms is not None:
            object.__setattr__(self, "until_ms", _nonneg_float(self.until_ms, "wait_for_event_until_invalid"))


@dataclass(frozen=True, slots=True)
class SchedulerDone:
    """Live scheduler terminal response once every eligible request has completed."""

    completed_requests: int = 0
    reason: str = "all_requests_completed"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "completed_requests", _nonneg_int(self.completed_requests, "scheduler_done_count_invalid")
        )


IterationDecision: TypeAlias = IterationPlan | WaitForEvent | SchedulerDone


@dataclass(frozen=True, slots=True)
class StackCapabilitySpec:
    """Declared domain, work-unit, hardware, checkpoint, and recipe capabilities of one plugin."""

    stack_id: str
    manifest_revision: str
    supported_domains: tuple[WorkloadDomain, ...]
    supported_work_units: tuple[WorkUnitKind, ...]
    supported_hardware_backends: tuple[str, ...]
    supported_phases: tuple[str, ...]
    supported_checkpoint_formats: tuple[str, ...]
    supported_tracks: tuple[StackTrack, ...]
    supported_recipe_ids: frozenset[str]
    has_launch_codec: bool
    protocol_version: str = STACK_ADAPTER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack_id", _req_str(self.stack_id, "stack_capability_id_required"))
        object.__setattr__(
            self, "manifest_revision", _req_str(self.manifest_revision, "stack_capability_revision_required")
        )
        domains = tuple(self.supported_domains)
        if not domains or any(d not in WORKLOAD_DOMAINS for d in domains):
            raise ModelingError(f"stack_capability_domains_invalid:{domains!r}")
        work_units = tuple(self.supported_work_units)
        if not work_units or any(w not in WORK_UNIT_KINDS for w in work_units):
            raise ModelingError(f"stack_capability_work_units_invalid:{work_units!r}")
        tracks = tuple(self.supported_tracks)
        if not tracks or any(t not in STACK_TRACKS for t in tracks):
            raise ModelingError(f"stack_capability_tracks_invalid:{tracks!r}")
        object.__setattr__(self, "supported_domains", domains)
        object.__setattr__(self, "supported_work_units", work_units)
        object.__setattr__(self, "supported_hardware_backends", tuple(str(x).lower() for x in self.supported_hardware_backends))
        object.__setattr__(self, "supported_phases", tuple(str(x) for x in self.supported_phases))
        object.__setattr__(self, "supported_checkpoint_formats", tuple(str(x) for x in self.supported_checkpoint_formats))
        object.__setattr__(self, "supported_tracks", tracks)
        object.__setattr__(self, "supported_recipe_ids", frozenset(str(x) for x in self.supported_recipe_ids))

    def supports_domain(self, domain: WorkloadDomain) -> bool:
        return domain in self.supported_domains

    def supports_work_unit(self, work_unit: WorkUnitKind) -> bool:
        return work_unit in self.supported_work_units


@dataclass(frozen=True, slots=True)
class ResolvedStackOptions:
    """Separates user-requested options from stack-effective canonical mechanisms with reasons."""

    stack_id: str
    track: StackTrack
    requested_mechanisms: tuple[tuple[str, Any], ...] = ()
    effective_mechanisms: tuple[tuple[str, Any], ...] = ()
    dispositions: tuple[tuple[str, str, str], ...] = ()
    capture_rule: CaptureBucketRule | None = None
    graph_capture_mode: CaptureMode = "none"
    graph_capture_keys: tuple[int, ...] = ()
    piecewise_capture_keys: tuple[int, ...] = ()
    splitting_ops: tuple[str, ...] = ()
    speculative: SpeculativeSpec | None = None
    stream_policy: StreamOverlapPolicy | None = None
    collective_policy: tuple[tuple[str, Any], ...] = ()
    expected_options_digest: InitVar[str] = ""
    options_digest: str = field(init=False)

    def __post_init__(self, expected_options_digest: str) -> None:
        object.__setattr__(self, "stack_id", _req_str(self.stack_id, "resolved_options_stack_id_required"))
        if self.track not in STACK_TRACKS:
            raise ModelingError(f"resolved_options_track_invalid:{self.track!r}")
        if self.graph_capture_mode not in CAPTURE_MODES:
            raise ModelingError(f"resolved_options_capture_mode_invalid:{self.graph_capture_mode!r}")
        object.__setattr__(
            self, "requested_mechanisms", _freeze_pairs(self.requested_mechanisms, "resolved_options_req_invalid")
        )
        object.__setattr__(
            self, "effective_mechanisms", _freeze_pairs(self.effective_mechanisms, "resolved_options_eff_invalid")
        )
        object.__setattr__(
            self, "collective_policy", _freeze_pairs(self.collective_policy, "resolved_options_coll_invalid")
        )
        object.__setattr__(
            self,
            "dispositions",
            tuple((str(k), str(status), str(reason)) for k, status, reason in self.dispositions),
        )
        object.__setattr__(
            self, "graph_capture_keys", _int_tuple(self.graph_capture_keys, "resolved_options_full_keys_invalid", minimum=1)
        )
        object.__setattr__(
            self,
            "piecewise_capture_keys",
            _int_tuple(self.piecewise_capture_keys, "resolved_options_pw_keys_invalid", minimum=1),
        )
        object.__setattr__(self, "splitting_ops", tuple(str(op) for op in self.splitting_ops))
        digest = _sha256_payload({
            "stack_id": self.stack_id,
            "track": self.track,
            "effective_mechanisms": self.effective_mechanisms,
            "graph_capture_mode": self.graph_capture_mode,
            "graph_capture_keys": self.graph_capture_keys,
            "piecewise_capture_keys": self.piecewise_capture_keys,
            "splitting_ops": self.splitting_ops,
            "collective_policy": self.collective_policy,
            "capture_rule": _digest_value(self.capture_rule),
            "speculative": _digest_value(self.speculative),
            "stream_policy": _digest_value(self.stream_policy),
        })
        if expected_options_digest and expected_options_digest != digest:
            raise ModelingError("resolved_options_digest_mismatch")
        object.__setattr__(self, "options_digest", digest)


def _digest_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _digest_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _digest_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (frozenset, set)):
        return sorted((_digest_value(v) for v in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_digest_value(v) for v in value]
    return value


def _eval_pred_node(node: ast.AST, capabilities: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in capabilities:
            raise KeyError(node.id)
        return capabilities[node.id]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return tuple(_eval_pred_node(elt, capabilities) for elt in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_eval_pred_node(node.operand, capabilities))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for val_node in node.values:
                if not bool(_eval_pred_node(val_node, capabilities)):
                    return False
            return True
        for val_node in node.values:
            if bool(_eval_pred_node(val_node, capabilities)):
                return True
        return False
    if isinstance(node, ast.Compare):
        left = _eval_pred_node(node.left, capabilities)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _eval_pred_node(comp, capabilities)
            ok = (
                left == right if isinstance(op, ast.Eq) else
                left != right if isinstance(op, ast.NotEq) else
                left < right if isinstance(op, ast.Lt) else
                left <= right if isinstance(op, ast.LtE) else
                left > right if isinstance(op, ast.Gt) else
                left >= right if isinstance(op, ast.GtE) else
                left in right if isinstance(op, ast.In) else
                left not in right if isinstance(op, ast.NotIn) else False
            )
            if not ok:
                return False
            left = right
        return True
    raise ModelingError("manifest_binding_predicate_unsupported_syntax")


def _eval_capability_predicate(predicate: str, capabilities: Mapping[str, Any]) -> bool:
    try:
        expr = ast.parse(predicate, mode="eval").body
        return bool(_eval_pred_node(expr, capabilities))
    except (SyntaxError, KeyError, TypeError):
        return False


@dataclass(frozen=True, slots=True)
class ManifestBinding:
    """One verified stack manifest entry binding a mechanism to an executable lowering pass."""

    mechanism_id: str
    state: ManifestBindingState
    ref: str
    lowering_pass_id: str = ""
    provenance: SourceRef = field(default_factory=lambda: SourceRef("", "", "", ""))
    archs: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    options: tuple[tuple[str, Any], ...] = ()
    capability_predicates: tuple[str, ...] = ()
    phase_scope: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanism_id", _req_str(self.mechanism_id, "manifest_binding_mechanism_id_required"))
        if self.state not in MANIFEST_BINDING_STATES:
            raise ModelingError(f"manifest_binding_state_invalid:{self.state!r}")
        object.__setattr__(self, "ref", _req_str(self.ref, "manifest_binding_ref_required"))
        if not isinstance(self.lowering_pass_id, str):
            raise ModelingError("manifest_binding_lowering_pass_id_invalid")
        object.__setattr__(self, "lowering_pass_id", self.lowering_pass_id.strip())
        if not isinstance(self.provenance, SourceRef):
            raise ModelingError("manifest_binding_provenance_must_be_source_ref")
        object.__setattr__(self, "archs", tuple(str(a).strip().lower() for a in self.archs if str(a).strip()))
        object.__setattr__(self, "models", tuple(str(m).strip() for m in self.models if str(m).strip()))
        object.__setattr__(self, "options", _freeze_pairs(self.options, "manifest_binding_options_invalid"))
        object.__setattr__(
            self,
            "capability_predicates",
            tuple(_req_str(p, "manifest_binding_predicate_invalid") for p in self.capability_predicates),
        )
        object.__setattr__(
            self,
            "phase_scope",
            tuple(str(p).strip().lower() for p in self.phase_scope if str(p).strip()),
        )
        if not isinstance(self.notes, str):
            raise ModelingError("manifest_binding_notes_invalid")

    @property
    def phases(self) -> tuple[str, ...]:
        """Alias for ``phase_scope`` used by manifest-binding consumers."""
        return self.phase_scope

    def matches(
        self,
        canonical_model: str,
        arch: str,
        phase: str | None = None,
        *,
        capabilities: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return whether this binding applies to ``(canonical_model, arch, phase, capabilities)``."""
        if self.models and canonical_model not in self.models:
            return False
        if self.archs and arch.strip().lower() not in self.archs:
            return False
        if phase and self.phase_scope and phase.strip().lower() not in self.phase_scope:
            return False
        if self.capability_predicates:
            if capabilities is None:
                return False
            if not all(_eval_capability_predicate(pred, capabilities) for pred in self.capability_predicates):
                return False
        return True

    def matches_arch_and_phase(self, arch: str, phase: str) -> bool:
        """Return whether this binding applies to the requested architecture and phase."""
        arch_ok = not self.archs or arch.strip().lower() in self.archs
        phase_ok = not self.phase_scope or phase.strip().lower() in self.phase_scope
        return arch_ok and phase_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "state": self.state,
            "ref": self.ref,
            "lowering_pass_id": self.lowering_pass_id,
            "provenance": {
                "repository": self.provenance.repository,
                "revision": self.provenance.revision,
                "relative_path": self.provenance.relative_path,
                "source_sha256": self.provenance.source_sha256,
            },
            "archs": list(self.archs),
            "models": list(self.models),
            "options": [list(pair) for pair in self.options],
            "capability_predicates": list(self.capability_predicates),
            "phase_scope": list(self.phase_scope),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class StackTrackSpec:
    """Resolved 4-track stack capability contract coupling active manifest bindings to readiness."""

    stack_id: str
    track: StackTrack
    optimization_mode: StackOptimizationMode
    active_bindings: tuple[ManifestBinding, ...] = ()
    allowed_readiness: frozenset[KernelReadiness] = frozenset({"shipping"})
    allows_target_synthesis: bool = False
    spec_digest: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack_id", _req_str(self.stack_id, "stack_track_spec_stack_id_required"))
        if self.track not in STACK_TRACKS:
            raise ModelingError(f"stack_track_spec_track_invalid:{self.track!r}")
        if self.optimization_mode not in STACK_OPTIMIZATION_MODES:
            raise ModelingError(f"stack_track_spec_optimization_mode_invalid:{self.optimization_mode!r}")
        bindings = tuple(self.active_bindings)
        if not all(isinstance(item, ManifestBinding) for item in bindings):
            raise ModelingError("stack_track_spec_active_bindings_invalid")
        if len({(item.mechanism_id, item.state) for item in bindings}) != len(bindings):
            raise ModelingError("stack_track_spec_duplicate_mechanism_id")
        readiness = frozenset(self.allowed_readiness)
        if not readiness or any(item not in KERNEL_READINESS_VALUES for item in readiness):
            raise ModelingError(f"stack_track_spec_readiness_invalid:{sorted(readiness)!r}")
        if not isinstance(self.allows_target_synthesis, bool):
            raise ModelingError("stack_track_spec_allows_target_synthesis_invalid")
        if self.track != "target":
            if self.allows_target_synthesis:
                raise ModelingError(f"stack_track_spec_target_synthesis_requires_target_track:{self.track}")
            if readiness & SYNTHESIS_READINESS_CLASSES:
                raise ModelingError(f"stack_track_spec_non_target_forbids_synthesis_readiness:{sorted(readiness)}")
        if self.allows_target_synthesis and self.optimization_mode != "modeling":
            raise ModelingError("stack_track_spec_target_synthesis_requires_modeling_mode")
        if self.optimization_mode == "simulation" and (readiness & SYNTHESIS_READINESS_CLASSES):
            raise ModelingError(f"stack_track_spec_simulation_forbids_synthesis_readiness:{sorted(readiness)}")
        if self.track == "shipping" and any(item.state != "merged" for item in bindings):
            raise ModelingError("stack_track_spec_shipping_requires_merged_bindings_only")
        if self.track == "eager" and bindings:
            raise ModelingError("stack_track_spec_eager_forbids_active_bindings")
        object.__setattr__(self, "active_bindings", bindings)
        object.__setattr__(self, "allowed_readiness", readiness)
        digest = _sha256_payload({
            "stack_id": self.stack_id,
            "track": self.track,
            "optimization_mode": self.optimization_mode,
            "active_bindings": [item.to_dict() for item in bindings],
            "allowed_readiness": sorted(readiness),
            "allows_target_synthesis": self.allows_target_synthesis,
        })
        object.__setattr__(self, "spec_digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_id": self.stack_id,
            "track": self.track,
            "optimization_mode": self.optimization_mode,
            "active_bindings": [item.to_dict() for item in self.active_bindings],
            "allowed_readiness": sorted(self.allowed_readiness),
            "allows_target_synthesis": self.allows_target_synthesis,
            "spec_digest": self.spec_digest,
        }


@dataclass(frozen=True, slots=True)
class StackLoweringPlan:
    """Frozen effective lowering and scheduling directives injected into Core PredictionSpine."""

    stack_id: str
    track: StackTrack
    recipe_bindings: tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...] = ()
    lowering_passes: tuple[str, ...] = ()
    allowed_rewrite_classes: frozenset[KernelReadiness] = frozenset({"shipping"})
    allowed_rewrite_tracks: tuple[RewriteTrack, ...] = ("config",)
    allowed_framework_rewrites: tuple[FrameworkRewriteKind, ...] = ()
    scheduling: SchedulingContract = field(default_factory=SchedulingContract)
    tensor_layouts: tuple[tuple[str, str, str], ...] = ()
    binding_origins: tuple[tuple[str, str], ...] = ()
    plan_digest: str = ""
    track_spec: StackTrackSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack_id", _req_str(self.stack_id, "lowering_plan_stack_id_required"))
        if self.track not in STACK_TRACKS:
            raise ModelingError(f"lowering_plan_track_invalid:{self.track!r}")
        readiness = frozenset(self.allowed_rewrite_classes)
        if not readiness or any(item not in KERNEL_READINESS_VALUES for item in readiness):
            raise ModelingError(f"lowering_plan_rewrite_classes_invalid:{sorted(readiness)!r}")
        tracks = tuple(self.allowed_rewrite_tracks)
        if any(item not in REWRITE_TRACKS for item in tracks):
            raise ModelingError(f"lowering_plan_rewrite_tracks_invalid:{tracks!r}")
        fw_rewrites = tuple(self.allowed_framework_rewrites)
        if any(item not in FRAMEWORK_REWRITE_KINDS for item in fw_rewrites):
            raise ModelingError(f"lowering_plan_framework_rewrites_invalid:{fw_rewrites!r}")
        if "kernel_new" in tracks and "synthesized_target" not in readiness:
            raise ModelingError("lowering_plan_kernel_new_requires_synthesized_target_readiness")
        if "kernel_tune" in tracks and "tiler_extension" not in readiness:
            raise ModelingError("lowering_plan_kernel_tune_requires_tiler_extension_readiness")
        if not isinstance(self.scheduling, SchedulingContract):
            raise ModelingError("lowering_plan_scheduling_invalid")
        if self.track_spec is not None:
            if not isinstance(self.track_spec, StackTrackSpec):
                raise ModelingError("lowering_plan_track_spec_invalid")
            if self.track_spec.stack_id != self.stack_id:
                raise ModelingError("lowering_plan_track_spec_stack_id_mismatch")
            if self.track_spec.track != self.track:
                raise ModelingError("lowering_plan_track_spec_track_mismatch")
            if not readiness.issubset(self.track_spec.allowed_readiness):
                raise ModelingError("lowering_plan_readiness_exceeds_track_spec")
            if not self.track_spec.allows_target_synthesis and (set(tracks) & {"kernel_new", "kernel_tune"}):
                raise ModelingError("lowering_plan_synthesis_track_forbidden_by_track_spec")
        bindings = tuple(
            (str(sel), str(rec_id), _freeze_pairs(opts, "lowering_plan_recipe_opts_invalid"))
            for sel, rec_id, opts in self.recipe_bindings
        )
        object.__setattr__(self, "recipe_bindings", bindings)
        object.__setattr__(self, "lowering_passes", tuple(str(p) for p in self.lowering_passes))
        object.__setattr__(self, "allowed_rewrite_classes", readiness)
        object.__setattr__(self, "allowed_rewrite_tracks", tracks)
        object.__setattr__(self, "allowed_framework_rewrites", fw_rewrites)
        object.__setattr__(
            self, "tensor_layouts", tuple((str(a), str(b), str(c)) for a, b, c in self.tensor_layouts)
        )
        object.__setattr__(
            self, "binding_origins", tuple((str(k), str(v)) for k, v in self.binding_origins)
        )
        payload: dict[str, Any] = {
            "stack_id": self.stack_id,
            "track": self.track,
            "recipe_bindings": [[s, r, list(o)] for s, r, o in bindings],
            "lowering_passes": list(self.lowering_passes),
            "allowed_rewrite_classes": sorted(readiness),
            "allowed_rewrite_tracks": list(tracks),
            "allowed_framework_rewrites": list(fw_rewrites),
            "scheduling": _digest_value(self.scheduling),
            "tensor_layouts": [list(x) for x in self.tensor_layouts],
            "binding_origins": [list(x) for x in self.binding_origins],
        }
        if self.track_spec is not None:
            payload["track_spec_digest"] = self.track_spec.spec_digest
        object.__setattr__(self, "plan_digest", _sha256_payload(payload))


@dataclass(frozen=True, slots=True)
class LaunchConfigIngress:
    """Schema-level launch declaration decoded statically from script, YAML, or CLI."""

    stack_id: str
    source_label: str
    model_ref: str
    checkpoint_hint: str
    parallel: ParallelSpec
    deployment_mode_hint: str | None = None
    runtime_limits: tuple[tuple[str, Any], ...] = ()
    raw_stack_flags: tuple[tuple[str, Any], ...] = ()
    field_origins: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack_id", _req_str(self.stack_id, "launch_ingress_stack_id_required"))
        object.__setattr__(self, "source_label", _req_str(self.source_label, "launch_ingress_source_required"))
        if not isinstance(self.parallel, ParallelSpec):
            raise ModelingError("launch_ingress_parallel_invalid")
        object.__setattr__(
            self, "runtime_limits", _freeze_pairs(self.runtime_limits, "launch_ingress_limits_invalid")
        )
        object.__setattr__(
            self, "raw_stack_flags", _freeze_pairs(self.raw_stack_flags, "launch_ingress_flags_invalid")
        )
        object.__setattr__(
            self, "field_origins", tuple((str(k), str(v)) for k, v in self.field_origins)
        )


@runtime_checkable
class LaunchConfigCodec(Protocol):
    """Optional script/YAML/CLI ingress and egress codec; never imports export/ or compose/."""

    def parse_launch_config(self, source: Path | str) -> LaunchConfigIngress: ...

    def audit_launch_config(
        self,
        ingress: LaunchConfigIngress,
        before_facts: Mapping[str, Any],
        tuned_facts: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def render_launch_config(self, tuned_facts: Mapping[str, Any]) -> str: ...


@runtime_checkable
class RuntimeStackAdapter(Protocol):
    """Authoritative plugin boundary for any inference runtime stack."""

    def capabilities(self) -> StackCapabilitySpec: ...

    def default_checkpoint_for(
        self, canonical_model: str, hardware: Hardware
    ) -> str | None: ...

    def resolve_options(
        self,
        model: ModelSpec,
        hardware: Hardware,
        parallel: ParallelSpec,
        *,
        role: str,
        track: StackTrack,
        requested_options: Mapping[str, Any],
        field_origins: Mapping[str, str],
    ) -> ResolvedStackOptions: ...

    def resolve_quant_instantiation(
        self,
        model: ModelSpec,
        checkpoint: CheckpointQuantMap,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
        *,
        role: str,
        phase: str,
    ) -> RuntimeQuantPlan: ...

    def select_lowering_plan(
        self,
        model: ModelSpec,
        quant_plan: RuntimeQuantPlan,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
    ) -> StackLoweringPlan: ...

    def plan_iteration(
        self,
        model: ModelSpec,
        state: IterationSchedulerState,
        options: ResolvedStackOptions,
        *,
        phase: str,
    ) -> IterationDecision: ...

    @property
    def launch_codec(self) -> LaunchConfigCodec | None: ...


_DEFAULT_KV_BLOCK_SIZE = 128
_MX_BLOCK_ELEMENTS = 32
_MX_KV_SCALE_VALUES_PER_GROUP = 2
_MX_KV_NZ_DIM_FRAG = 32
_MX_K_SCALE_NZ_TOKEN_FRAG = 16
_MX_V_SCALE_NZ_DIM_FRAG = 16
_FP8_BLOCK_ELEMENTS = 128
_INT4_GROUP_ELEMENTS = 128
_MLA_SCALE_GROUP_SIZE = 128
_FP32_SCALE_BYTES = 4
_E8M0_SCALE_BYTES = 1
_DEFAULT_CONV_KERNEL_SIZE = 4
_DEFAULT_LINEAR_HEAD_DIM = 128
_MIN_CONV_KERNEL_SIZE = 2
_LINEAR_ATTENTION_KINDS = frozenset({
    "kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet",
})


def resolve_checkpoint_tensor_abi(
    raw_tag: str,
    native_dtype: str,
    rule: TensorQuantRule | None = None,
    *,
    role: str = "",
) -> TensorPrecisionABI:
    """Resolve a checkpoint's physical declarations and role-dependent quantizer."""
    upper = (raw_tag or "").strip().upper().replace("-", "_")
    abi = _tensor_abi_for_tag(upper, native_dtype, role)
    if rule is None:
        return abi
    weight = rule.weight_dtype or abi.weight_dtype
    activation = rule.activation_dtype or abi.input_dtype
    scale = rule.scale_dtype or abi.weight_scale_dtype
    block = rule.group_size or abi.weight_block
    return replace(
        abi, weight_dtype=weight, input_dtype=activation,
        compute_dtype=(activation if abi.compute_dtype == abi.input_dtype else abi.compute_dtype),
        weight_block=block, weight_scale_dtype=scale,
        weight_scale_bytes=int(dtype_bytes(scale)) if scale and scale != "none" else 0,
        act_group_size=block[-1] if block and abi.act_group_size else abi.act_group_size,
        act_scale_dtype=scale if abi.act_scale_dtype else "",
        weight_offset_dtype=weight if rule.symmetric is False else abi.weight_offset_dtype,
    )


def _tensor_abi_for_tag(upper: str, native_dtype: str, role: str) -> TensorPrecisionABI:
    dense = {"": native_dtype, "NONE": native_dtype, "FLOAT": native_dtype, "BF16": "bf16", "BFLOAT16": "bf16",
             "FP16": "fp16", "FLOAT16": "fp16", "FP32": "fp32", "FLOAT32": "fp32"}
    if upper in dense:
        dtype = dense[upper]
        return TensorPrecisionABI("dense", dtype, dtype, dtype, dtype)
    if upper in {"W4A4", "W4A4_DYNAMIC"}:
        return TensorPrecisionABI(
            "w4a4_laos", "int4", "int4", native_dtype, "int4", accum_dtype="int32",
            weight_scale_dtype="fp32", weight_scale_bytes=_FP32_SCALE_BYTES,
            weight_offset_dtype="fp32", weight_layout="INT4PACK", act_scale_dtype="fp32",
            activation_quant_mode="dynamic_per_token", explicit_cast_on_exit=native_dtype,
        )
    if upper in {"W8A8_MXFP8", "W8A8_MXFP", "W8A8_MXFP8_MXFP8", "MXFP8", "MXFP8_DYNAMIC"}:
        blk = (1, _MX_BLOCK_ELEMENTS)
        return TensorPrecisionABI(
            linear_method="mx_fp8", activation_quant_mode="dynamic_mx32",
            input_dtype="fp8_e4m3",
            weight_dtype="fp8_e4m3",
            output_dtype=native_dtype,
            compute_dtype="fp8_e4m3",
            accum_dtype="fp32",
            weight_block=blk,
            weight_scale_dtype="e8m0",
            weight_scale_bytes=_E8M0_SCALE_BYTES,
            act_group_size=blk[-1],
            act_scale_dtype="e8m0",
            act_scale_pack="2",
        )
    if upper in {
        "W4A8_MXFP", "W4A8_MXFP4", "W4A8_MXFP4_DYNAMIC", "W4A8_MXFP4_MXFP8",
        "W4A4_MXFP", "W4A4_MXFP4", "W4A4_MXFP4_MXFP4",
        "W4A4F4_DAOS", "MXFP4", "MXFP4_W4A4", "MXFP4_ROUTED_ONLY", "MXFP4_ALL",
        "MXFP8_MXFP4", "FP4", "QUARK_MXFP4", "QUANT_FOUR_OVER_SIX",
    }:
        blk = (1, _MX_BLOCK_ELEMENTS)
        if "A8" in upper or upper in {
            "MXFP4", "MXFP4_ROUTED_ONLY", "MXFP4_ALL", "MXFP8_MXFP4", "QUARK_MXFP4",
        }:
            act = "fp8_e4m3"
        else:
            act = "fp4_e2m1"
        return TensorPrecisionABI(
            linear_method="mx_fp4", activation_quant_mode="dynamic_mx32",
            input_dtype=act,
            weight_dtype="fp4_e2m1",
            output_dtype=native_dtype,
            compute_dtype="fp4_e2m1",
            accum_dtype="fp32",
            weight_block=blk,
            weight_scale_dtype="e8m0",
            weight_scale_bytes=_E8M0_SCALE_BYTES,
            act_group_size=blk[-1],
            act_scale_dtype="e8m0",
            act_scale_pack="2",
        )
    if upper in {"W8A8_FP8", "FP8", "FP8_DYNAMIC", "FP8_E4M3", "FP8_E5M2", "FP8_BLOCK", "QUARK_FP8"}:
        dtype = "fp8_e5m2" if upper == "FP8_E5M2" else "fp8_e4m3"
        return TensorPrecisionABI(
            "fp8", dtype, dtype, native_dtype, dtype,
            weight_block=(_FP8_BLOCK_ELEMENTS, _FP8_BLOCK_ELEMENTS),
            weight_scale_dtype="fp32", weight_scale_bytes=_FP32_SCALE_BYTES,
            act_group_size=_FP8_BLOCK_ELEMENTS, act_scale_dtype="fp32",
            activation_quant_mode="dynamic_per_token",
        )
    if upper in {"W8A8", "W8A8_INT8", "W8A8_DYNAMIC", "W8A8_MIX", "W8A8_PDMIX",
                 "INT8", "INT8_DYNAMIC", "INT8_SYM", "QUARK_INT8"}:
        dynamic = upper in {"W8A8_DYNAMIC", "W8A8_INT8", "INT8_DYNAMIC", "INT8_SYM"}
        mode = "dynamic_per_token" if dynamic else "static_per_tensor"
        if upper in {"W8A8_MIX", "W8A8_PDMIX"}:
            mode = {"pd_decode": "static_per_tensor", "pd_prefill": "dynamic_per_token"}.get(role, "pd_mixed")
        return TensorPrecisionABI(
            linear_method="w8a8", activation_quant_mode=mode,
            input_dtype="int8",
            weight_dtype="int8",
            output_dtype=native_dtype,
            compute_dtype="int8",
            accum_dtype="int32",
            weight_scale_dtype="fp32",
            weight_scale_bytes=_FP32_SCALE_BYTES,
            act_scale_dtype="fp32",
        )
    if upper in {"W4A8", "W4A8_DYNAMIC", "W4A8_COMPRESSED"}:
        return TensorPrecisionABI(
            linear_method="w4a8", activation_quant_mode="dynamic_per_token",
            input_dtype="int8",
            weight_dtype="int4",
            output_dtype=native_dtype,
            compute_dtype="int8",
            accum_dtype="int32",
            weight_scale_dtype="fp32",
            weight_scale_bytes=_FP32_SCALE_BYTES,
            act_scale_dtype="fp32",
        )
    if upper in {"W4A16", "INT4", "AWQ", "GPTQ", "W4A16_AWQ", "W4A16_GPTQ", "W4A16_COMPRESSED",
                 "W8A16_AWQ", "W8A16_GPTQ"}:
        is_int8 = upper.startswith("W8A16")
        return TensorPrecisionABI(
            linear_method="w8a16" if is_int8 else "w4a16",
            input_dtype=native_dtype,
            weight_dtype="int8" if is_int8 else "int4",
            output_dtype=native_dtype,
            compute_dtype=native_dtype,
            accum_dtype="fp32",
            weight_block=(1, _INT4_GROUP_ELEMENTS),
            weight_scale_dtype=native_dtype,
            weight_scale_bytes=int(dtype_bytes(native_dtype)),
        )
    raise ModelingError(f"unknown_checkpoint_quant_tag:{upper}")


_FLOAT_CACHE_DTYPES = {"BF16": "bf16", "BFLOAT16": "bf16", "FP16": "fp16", "FLOAT16": "fp16"}
_STATIC_INT8_CACHE = {"C8", "INT8", "INT8_SYM", "FAQUANT", "FAKQUANT"}
_FP8_CACHE = {"FP8", "FP8_DYNAMIC", "FP8_E4M3", "FP8_E5M2"}
_MX_CACHE = {"MXFP8", "MXFP8_DYNAMIC"}
_MX_GQA_PACKET = "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"


def _cache_dtype(tag: str, native_dtype: str) -> str:
    if tag in {"", "FLOAT"}:
        return native_dtype
    if tag in _FLOAT_CACHE_DTYPES:
        return _FLOAT_CACHE_DTYPES[tag]
    if tag in _STATIC_INT8_CACHE or tag == "INT8_DYNAMIC":
        return "int8"
    if tag in _FP8_CACHE or tag in _MX_CACHE:
        return "fp8_e5m2" if tag == "FP8_E5M2" else "fp8_e4m3"
    raise ModelingError(f"unknown_checkpoint_cache_tag:{tag}")


def _plane(name: str, kind: CachePlaneKind, axes: tuple[str, ...], shape: tuple[int, ...],
           dtype: str, frequency: AllocationFrequency, *, group_axes: tuple[int, ...] = (),
           group_extents: tuple[int, ...] = ()) -> CachePlaneABI:
    return CachePlaneABI(name, kind, axes, shape, shape, dtype, allocation_frequency=frequency,
                         write_trigger="load" if frequency in {"checkpoint", "layer"} else "append",
                         group_axes=group_axes, group_extents=group_extents)


def _mxfp8_page(heads: int, block: int, kd: int, vd: int) -> tuple[CachePlaneABI, ...]:
    group = _MX_BLOCK_ELEMENTS
    pair = _MX_KV_SCALE_VALUES_PER_GROUP
    for value, alignment, axis in ((block, group * pair, "block"), (kd, group * pair, "k_dim"),
                                   (vd, _MX_KV_NZ_DIM_FRAG, "v_dim")):
        if value <= 0 or value % alignment:
            raise ModelingError(f"mxfp8_kv_{axis}_not_aligned:{value}%{alignment}")
    frag = _MX_KV_NZ_DIM_FRAG
    kt = _MX_K_SCALE_NZ_TOKEN_FRAG
    vc = _MX_V_SCALE_NZ_DIM_FRAG
    planes = (
        CachePlaneABI("k_payload", "payload", ("head", "token", "dim"), (heads, block, kd),
                      (heads, kd // frag, block, frag), "fp8_e4m3", layout="PA_NZ", alignment=frag),
        CachePlaneABI("k_scale", "scale", ("head", "token", "group"), (heads, block, kd // group),
                      (heads, block // kt, kd // (group * pair), kt, pair), "e8m0", layout="PA_NZ",
                      group_axes=(2,), group_extents=(group,), scale_dtype="e8m0", alignment=kt),
        CachePlaneABI("v_payload", "payload", ("head", "token", "dim"), (heads, block, vd),
                      (heads, vd // frag, block, frag), "fp8_e4m3", layout="PA_NZ", alignment=frag),
        CachePlaneABI("v_scale", "scale", ("head", "dim", "block_group"), (heads, vd, block // group),
                      (heads, vd // vc, block // (group * pair), vc, pair), "e8m0", layout="PA_NZ",
                      group_axes=(2,), group_extents=(group,), scale_dtype="e8m0", alignment=vc,
                      write_trigger="page_reassign"),
    )
    offset = 0
    pitch = sum(p.exact_bytes() for p in planes)
    packet = []
    for plane in planes:
        packet.append(replace(plane, offset_in_page=offset, page_pitch=pitch))
        offset += plane.exact_bytes()
    return tuple(packet)


def _resolve_gqa_kv_planes(plan: LayerPlan, tag: str, rule: CacheQuantRule, native_dtype: str,
                           heads: int, tp: int, block: int) -> tuple[CachePlaneABI, ...]:
    kd, vd = plan.head_dim, plan.v_head_dim or plan.head_dim
    if tag == _MX_GQA_PACKET:
        return _mxfp8_page(heads, block, kd, vd)
    dtype = _cache_dtype(tag, native_dtype)
    sides = (("k", kd), ("v", vd))
    if not tag or tag == "FLOAT" or tag in _FLOAT_CACHE_DTYPES:
        if kd == vd:
            return (_plane("kv_payload", "payload", ("kv", "head", "token", "dim"),
                           (2, heads, block, kd), dtype, "page"),)
    planes = [_plane(f"{side}_payload", "payload", ("head", "token", "dim"),
                     (heads, block, width), dtype, "page") for side, width in sides]
    if tag in _STATIC_INT8_CACHE:
        scale = rule.kv_scale_dtype or "fp32"
        offset = rule.kv_offset_dtype or "int8"
        for kind, dt in (("scale", scale), ("offset", offset)):
            if dt != "none":
                planes.extend(_plane(f"{side}_{kind}", kind, ("head", "dim"), (heads, width),
                                     dt, "checkpoint") for side, width in sides)
        if tag == "FAQUANT":
            q_heads = max(1, plan.num_heads // tp)
            planes.append(_plane("q_scale", "scale", ("head", "dim"), (q_heads, kd), scale, "layer"))
    elif tag in _FP8_CACHE:
        planes.extend(_plane(f"{side}_scale", "scale", ("tensor",), (1,), rule.kv_scale_dtype or "fp32",
                             "checkpoint") for side, _ in sides)
    elif tag in _MX_CACHE or tag == "INT8_DYNAMIC":
        for side, width in sides:
            group = _MX_BLOCK_ELEMENTS if tag in _MX_CACHE else width
            groups = (width + group - 1) // group
            scale = rule.kv_scale_dtype or ("e8m0" if tag in _MX_CACHE else "fp32")
            planes.append(_plane(f"{side}_scale", "scale", ("head", "group"), (heads, groups),
                                 scale, "token", group_axes=(1,), group_extents=(group,)))
    return tuple(planes)


def _resolve_mla_cache_planes(plan: LayerPlan, tag: str, rule: CacheQuantRule,
                              native_dtype: str) -> tuple[CachePlaneABI, ...]:
    width = plan.cache_row_elements
    rope = plan.qk_rope_head_dim
    latent = width - rope
    if latent < 0:
        raise ModelingError(f"cache_suffix_exceeds_row_width:{rope}>{width}")
    dtype = _cache_dtype(tag, native_dtype)
    planes = [_plane("mla_latent", "payload", ("dim",), (latent,), dtype, "token")]
    if tag in _STATIC_INT8_CACHE:
        planes.append(_plane("mla_latent_scale", "scale", ("kv_head",), (1,),
                             rule.kv_scale_dtype or "fp32", "checkpoint"))
        if rule.kv_offset_dtype and rule.kv_offset_dtype != "none":
            planes.append(_plane("mla_latent_offset", "offset", ("kv_head",), (1,),
                                 rule.kv_offset_dtype, "checkpoint"))
    elif tag in _MX_CACHE or tag in _FP8_CACHE or tag == "INT8_DYNAMIC":
        group = _MX_BLOCK_ELEMENTS if tag in _MX_CACHE else _MLA_SCALE_GROUP_SIZE
        groups = (latent + group - 1) // group
        scale = rule.kv_scale_dtype or ("e8m0" if tag in _MX_CACHE else "fp32")
        planes.append(_plane("mla_latent_scale", "scale", ("group",), (groups,), scale, "token",
                             group_axes=(0,), group_extents=(group,)))
    if rope:
        rope_dtype = dtype if tag in _FLOAT_CACHE_DTYPES else native_dtype
        planes.append(_plane("mla_rotary", "suffix", ("dim",), (rope,), rope_dtype, "token"))
    return tuple(planes)


def _resolve_index_cache_planes(tag: str, native_dtype: str, width: int,
                                scale_dtype: str) -> tuple[CachePlaneABI, ...]:
    dtype = _cache_dtype(tag, native_dtype)
    payload = _plane("index_k", "payload", ("dim",), (width,), dtype, "token")
    if not tag or tag == "FLOAT" or tag in _FLOAT_CACHE_DTYPES:
        return (payload,)
    scale = scale_dtype or ("e8m0" if tag in _MX_CACHE else "fp32")
    groups = (width + _MX_BLOCK_ELEMENTS - 1) // _MX_BLOCK_ELEMENTS if tag in _MX_CACHE else 1
    return payload, _plane("index_k_scale", "scale", ("group",), (groups,), scale, "token")


def _resolve_recurrent_cache_abi(
    plan: LayerPlan,
    parallel: ParallelSpec,
    *,
    native_id: int,
    role: str,
    phase: str,
    ckpt_digest: str,
    scope: CacheScope,
    kda_parallel_axis: str,
    stage: int,
) -> CacheStateABI:
    """Resolve sharded recurrent state + convolution history planes for GDN/KDA layers."""
    attn = str(plan.attention or "").lower()
    use_pcp = attn == "kda" and parallel.pcp > 1 and kda_parallel_axis == "tp_x_pcp"
    shard_div = parallel.tp * (parallel.pcp if use_pcp else 1)
    shard_axis = "tp_x_pcp" if use_pcp else "tp"
    v_heads = max(1, int(plan.linear_value_heads or plan.linear_heads or 1))
    local_heads = max(1, v_heads // shard_div)
    v_dim = max(1, int(plan.linear_head_dim or _DEFAULT_LINEAR_HEAD_DIM))
    k_dim = max(1, int(plan.linear_key_head_dim or v_dim))
    rec_dtype = str(plan.recurrent_state_dtype or "fp32").lower()
    rec_bytes = int(math.ceil(local_heads * k_dim * v_dim * dtype_bytes(rec_dtype)))

    raw_conv_k = int(plan.conv_kernel) if plan.conv_kernel is not None else _DEFAULT_CONV_KERNEL_SIZE
    has_conv = raw_conv_k >= _MIN_CONV_KERNEL_SIZE and not (
        plan.conv_state_elements is not None and int(plan.conv_state_elements) == 0 and raw_conv_k <= 1
    )
    history_slots = max(0, raw_conv_k - 1) if has_conv else 0
    rec_plane = CachePlaneABI(
        plane_id="recurrent_state",
        plane_kind="payload",
        logical_axes=("head", "k_dim", "v_dim"),
        logical_shape=(local_heads, k_dim, v_dim),
        physical_shape=(local_heads, k_dim, v_dim),
        dtype=rec_dtype,
        layout="ND",
        allocation_frequency="request_slot",
        write_trigger="accepted_commit",
        bytes_per_unit=rec_bytes,
    )
    if has_conv:
        conv_w = int(plan.conv_width or (plan.conv_state_elements // max(1, history_slots)) or 0)
        local_conv_w = max(1, conv_w // shard_div) if conv_w > 0 else max(1, local_heads * v_dim)
        conv_dtype = str(plan.conv_state_dtype or "bf16").lower()
        conv_bytes = int(math.ceil(local_conv_w * history_slots * dtype_bytes(conv_dtype)))
        planes: tuple[CachePlaneABI, ...] = (
            rec_plane,
            CachePlaneABI(
                plane_id="conv_history",
                plane_kind="payload",
                logical_axes=("channel", "history_slot"),
                logical_shape=(local_conv_w, history_slots),
                physical_shape=(local_conv_w, history_slots),
                dtype=conv_dtype,
                layout="ND",
                offset_in_page=rec_bytes,
                allocation_frequency="request_slot",
                write_trigger="append",
                bytes_per_unit=conv_bytes,
            ),
        )
    else:
        planes = (rec_plane,)

    return CacheStateABI(
        scope=scope,
        native_layer=native_id,
        component_role="recurrent_state",
        producer_key=f"{scope}.l{native_id}.recurrent_state",
        planes=planes,
        phase=phase,
        deployment_role=role,
        checkpoint_digest=ckpt_digest,
        recipe_abi_id=f"{attn}_recurrent_state",
        history_slots=history_slots,
        placement=CachePlacementRule(
            pp_stage=stage, token_ownership_axis="none",
            replication_factor=1,
            state_shard_axis=shard_axis,
        ),
    )


def _cache_scope(plan: LayerPlan) -> CacheScope:
    if plan.scope in {"target", "body"}:
        return "target"
    if plan.scope in {"draft", "mtp", "dspark", "standalone_draft"}:
        return "draft"
    raise ModelingError(f"cache_state_scope_invalid:{plan.scope}")


def _cache_placement(plan: LayerPlan, parallel: ParallelSpec, stage: int) -> CachePlacementRule:
    gqa = plan.attention in {"gqa", "projection_only"}
    heads = plan.num_kv_heads or plan.num_heads or 1
    replication = max(1, parallel.tp // heads) if gqa else 1
    gather = parallel.dcp_tp_gather
    if gqa:
        if replication == 1:
            gather = 1
        elif replication % gather:
            raise ModelingError(f"cp_exceeds_gqa_decode_domain:tp={parallel.tp},kv_heads={heads}")
    striped = parallel.dcp_pcp_factor * gather * parallel.dcp_dp_width > 1
    return CachePlacementRule(
        pp_stage=stage, token_ownership_axis="cp_striped" if striped else "replicated",
        replication_factor=replication, head_range=(0, local_kv_heads(heads, parallel.tp)) if gqa else None,
        state_shard_axis="tp" if gqa else "",
    )


def resolve_checkpoint_cache_abis(
    layers: tuple[LayerPlan, ...],
    checkpoint: CheckpointQuantMap,
    parallel: ParallelSpec,
    native_dtype: str,
    *,
    role: str,
    phase: str,
    kv_block_size: int = _DEFAULT_KV_BLOCK_SIZE,
    cp_interleave_size: int | None = None,
    kda_parallel_axis: Literal["tp", "tp_x_pcp"] = "tp_x_pcp",
    stage: int = 0,
) -> tuple[tuple[CacheStateABI, ...], tuple[CacheStateABI, ...], tuple[CacheStateABI, ...]]:
    """Instantiate storage at the native producer and attach same-scope consumers."""
    block = _pos_int(kv_block_size, "cache_state_kv_block_invalid")
    interleave = block if cp_interleave_size is None else _pos_int(cp_interleave_size, "cache_state_cp_interleave_invalid")
    if kda_parallel_axis not in {"tp", "tp_x_pcp"}:
        raise ModelingError(f"kda_parallel_axis_invalid:{kda_parallel_axis}")
    digest = checkpoint.digest()
    rule = checkpoint.cache_rule
    native_ids = {(_cache_scope(p), p.index): p.native_layer_id if p.native_layer_id >= 0 else p.index
                  for p in layers}

    def owner_id(scope: CacheScope, owner: int, native: int) -> int:
        return native_ids.get((scope, owner), owner) if owner >= 0 else native

    consumers: dict[tuple[CacheScope, int, str], list[str]] = {}
    for plan in layers:
        scope = _cache_scope(plan)
        native = native_ids[scope, plan.index]
        for component, owner in (("kv", plan.kv_owner_layer_id), ("index_k", plan.index_k_owner_layer_id)):
            if component == "index_k" and not plan.indexer:
                continue
            producer = owner_id(scope, owner, native)
            if producer != native:
                consumers.setdefault((scope, producer, component), []).append(f"{scope}.l{native}.{component}_consumer")

    kv_abis = []
    idx_abis = []
    rec_abis = []
    for plan in layers:
        scope = _cache_scope(plan)
        native = native_ids[scope, plan.index]
        if plan.attention in _LINEAR_ATTENTION_KINDS:
            rec_abis.append(_resolve_recurrent_cache_abi(
                plan, parallel, native_id=native, role=role, phase=phase, ckpt_digest=digest,
                scope=scope, kda_parallel_axis=kda_parallel_axis, stage=stage,
            ))
            continue
        if plan.cache_kind == "none":
            continue
        placement = _cache_placement(plan, parallel, stage)
        common = dict(scope=scope, native_layer=native, phase=phase, deployment_role=role,
                      checkpoint_digest=digest, kv_block_size=block, kernel_block_size=block,
                      cp_interleave_size=interleave, placement=placement)
        ratio = max(1, plan.compress_ratio)
        owns_kv = owner_id(scope, plan.kv_owner_layer_id, native) == native
        windowed_mla = plan.attention == "mla" and plan.cache_kind == "window_compressed"
        if owns_kv or windowed_mla:
            tag = rule.kv_tag_for_layer(native).upper()
            if plan.attention == "mla":
                planes = _resolve_mla_cache_planes(plan, tag, rule, native_dtype)
                components = ["window_kv"] if windowed_mla else ["mla_latent"]
                if windowed_mla and owns_kv and plan.owns_long_kv:
                    components.append("compressed_kv")
            else:
                heads = local_kv_heads(plan.num_kv_heads or plan.num_heads or 1, parallel.tp)
                planes = _resolve_gqa_kv_planes(plan, tag, rule, native_dtype, heads, parallel.tp, block)
                components = ["kv_combined"]
            for component in components:
                shared = () if component == "window_kv" else tuple(consumers.get((scope, native, "kv"), ()))
                kv_abis.append(CacheStateABI(
                    **common, component_role=component, producer_key=f"{scope}.l{native}.{component}",
                    planes=planes, window=None if component == "compressed_kv" else plan.window or None,
                    compression_ratio=1 if component == "window_kv" else ratio, consumer_keys=shared,
                ))
        if (plan.indexer or plan.owns_index_k) and owner_id(scope, plan.index_k_owner_layer_id, native) == native:
            pool = max(1, plan.index_kpool)
            idx_abis.append(CacheStateABI(
                **common, component_role="index_k", producer_key=f"{scope}.l{native}.index_k",
                planes=_resolve_index_cache_planes(rule.index_tag_for_layer(native).upper(), native_dtype,
                                                   plan.index_dim, rule.index_scale_dtype),
                compression_ratio=ratio, pooling_ratio=pool, tail_slots=2 * pool if pool > 1 else 0,
                consumer_keys=tuple(consumers.get((scope, native, "index_k"), ())),
            ))
    return tuple(kv_abis), tuple(idx_abis), tuple(rec_abis)


__all__ = [
    "resolve_checkpoint_cache_abis",
    "resolve_checkpoint_tensor_abi",
    "ALLOCATION_FREQUENCIES",
    "CACHE_COMPONENT_ROLES",
    "CACHE_PLANE_KINDS",
    "CACHE_SCOPES",
    "CAPTURE_FALLBACKS",
    "CAPTURE_MODES",
    "CAPTURE_UNITS",
    "FRAMEWORK_REWRITE_KINDS",
    "MANIFEST_BINDING_STATES",
    "MEMORY_TIERS",
    "PD_TRANSFER_MODES",
    "PD_TRANSPORT_KINDS",
    "PREEMPTION_POLICIES",
    "REWRITE_TRACKS",
    "STACK_ADAPTER_PROTOCOL_VERSION",
    "STACK_OPTIMIZATION_MODES",
    "STACK_TRACKS",
    "SYNTHESIS_READINESS_CLASSES",
    "TOKEN_OWNERSHIP_AXES",
    "WORKLOAD_DOMAINS",
    "WORK_UNIT_KINDS",
    "WRITE_TRIGGERS",
    "AllocationFrequency",
    "CacheComponentRole",
    "CachePlacementRule",
    "CachePlaneABI",
    "CachePlaneKind",
    "CacheScope",
    "CacheStateABI",
    "CaptureBucketRule",
    "CaptureFallback",
    "CaptureMode",
    "CaptureUnit",
    "FrameworkRewriteKind",
    "IterationDecision",
    "IterationSchedulerState",
    "LaunchConfigCodec",
    "LaunchConfigIngress",
    "ManifestBinding",
    "ManifestBindingState",
    "MemoryTier",
    "PdKvTransferSpec",
    "PdLayerTransferEdge",
    "PdTransferMode",
    "PdTransportKind",
    "PiecewiseSplitPoint",
    "PostAcceptCommitSpec",
    "PreemptionPolicy",
    "PreparedWeightSpec",
    "ResolvedStackOptions",
    "RewriteTrack",
    "RuntimeQuantPlan",
    "RuntimeStackAdapter",
    "SchedulerDone",
    "SchedulingContract",
    "StackCapabilitySpec",
    "StackLoweringPlan",
    "StackOptimizationMode",
    "StackTrack",
    "StackTrackSpec",
    "StarsSubmitPolicy",
    "StreamLaneBudget",
    "StreamOverlapPolicy",
    "StreamSyncEdge",
    "TokenOwnershipAxis",
    "WaitForEvent",
    "WorkUnitKind",
    "WorkloadDomain",
    "WriteTrigger",
]
