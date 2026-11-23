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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .checkpoint import CheckpointOrigin, CheckpointQuantMap, CHECKPOINT_ORIGINS
from .common import ModelingError, dtype_bytes
from .hardware import Hardware
from .model import ModelSpec
from .operators import (
    KERNEL_READINESS_VALUES,
    OWNER_KINDS,
    KernelReadiness,
    OwnerKind,
    TensorPrecisionABI,
)
from .parallel import ParallelSpec
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

    def exact_bytes(self) -> int:
        """Exact integer bytes occupied by one allocation unit of this plane."""
        if self.bytes_per_unit > 0:
            return self.bytes_per_unit
        if not self.physical_shape:
            return 0
        elements = math.prod(self.physical_shape)
        raw = elements * dtype_bytes(self.dtype) / float(self.packed_elements)
        return int(math.ceil(raw))

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
    plan_digest: str = ""

    def __post_init__(self) -> None:
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
        if not self.plan_digest:
            object.__setattr__(self, "plan_digest", self._compute_digest())

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
    options_digest: str = ""

    def __post_init__(self) -> None:
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
        if not self.options_digest:
            digest = _sha256_payload({
                "stack_id": self.stack_id,
                "track": self.track,
                "effective_mechanisms": list(self.effective_mechanisms),
                "graph_capture_mode": self.graph_capture_mode,
                "graph_capture_keys": list(self.graph_capture_keys),
                "piecewise_capture_keys": list(self.piecewise_capture_keys),
                "splitting_ops": list(self.splitting_ops),
                "collective_policy": list(self.collective_policy),
            })
            object.__setattr__(self, "options_digest", digest)


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
        if not self.plan_digest:
            digest = _sha256_payload({
                "stack_id": self.stack_id,
                "track": self.track,
                "recipe_bindings": [[s, r, list(o)] for s, r, o in bindings],
                "lowering_passes": list(self.lowering_passes),
                "allowed_rewrite_classes": sorted(readiness),
                "allowed_rewrite_tracks": list(tracks),
                "allowed_framework_rewrites": list(fw_rewrites),
                "tensor_layouts": [list(x) for x in self.tensor_layouts],
            })
            object.__setattr__(self, "plan_digest", digest)


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


__all__ = [
    "ALLOCATION_FREQUENCIES",
    "CACHE_COMPONENT_ROLES",
    "CACHE_PLANE_KINDS",
    "CACHE_SCOPES",
    "CAPTURE_FALLBACKS",
    "CAPTURE_MODES",
    "CAPTURE_UNITS",
    "FRAMEWORK_REWRITE_KINDS",
    "MEMORY_TIERS",
    "PD_TRANSFER_MODES",
    "PD_TRANSPORT_KINDS",
    "PREEMPTION_POLICIES",
    "REWRITE_TRACKS",
    "STACK_ADAPTER_PROTOCOL_VERSION",
    "STACK_TRACKS",
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
    "StackTrack",
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
