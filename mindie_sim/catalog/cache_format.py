"""Declared packed cache rows and typed per-layer CacheStateABI builders.

[Role] Resolve per-native-layer, per-role, and per-phase cache plane layouts
  shared by producers, readers, and capacity accounting, including NoPE/RoPE
  MLA (520/528/656 B), asymmetric GQA K/V, dense C8/FAQuant, and
  ``QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`` packet pages.
[Invariants] A group scale is stored once per row group. Trailing alignment
  padding is tracked separately from numerical scale bytes. Per-layer NoPE vs
  RoPE geometry never collapses across heterogeneous layers.
[Timeouts & Locks] Constant/layer-bounded pure arithmetic, no I/O or locks.
[Side-Effect Policy] Returns immutable format and CacheStateABI metadata only.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mindie_sim.catalog.cache_pools import (
    LINEAR_ATTENTION_KINDS,
    _ceil_div,
    exact_element_bytes,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.stack import CachePlacementRule, CachePlaneABI, CacheStateABI


#: NoPE C8 block-quantized MLA cache: one FP8 row split into 128-element
#: scale groups.  A 512-element latent therefore stores ``512 B`` of payload
#: plus four FP16 scales (``8 B``) = ``520 B`` per row, or with FP32 scales
#: (``16 B``) = ``528 B`` per row.
NOPE_BLOCK_QUANT_TILE = 128
NOPE_BLOCK_PAYLOAD_DTYPE = "fp8_e4m3"
NOPE_BLOCK_SCALE_DTYPE = "fp16"
C8_MLA_SCALE_GROUP = 128
C8_MLA_SCALE_DTYPE = "fp32"
C8_MLA_ROTARY_DTYPE = "bf16"
MXFP_KV_GROUP_SIZE = 32
MXFP_KV_SCALE_ALIGN = 64
MXFP_KV_V_DIM_ALIGN = 32
MXFP_KV_PAYLOAD_DTYPE = "fp8_e4m3"
MXFP_KV_SCALE_DTYPE = "fp8_e8m0"
MXFP_KV_TAG = "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"
FAQUANT_KV_TAG = "FAQUANT"
C8_KV_TAG = "C8"
DEFAULT_CACHE_PAGE_BLOCK = 128
NOPE_FP8_PAYLOAD_DTYPES = frozenset({"fp8", "fp8_e4m3"})
ALLOWED_CACHE_REPR_FIELDS = frozenset({
    "dtype", "group_size", "scale_dtype", "suffix_elements",
    "suffix_dtype", "packed_elements", "alignment",
})


def block_quantized_mla_format(
    tile: int = NOPE_BLOCK_QUANT_TILE,
    *,
    scale_dtype: str = NOPE_BLOCK_SCALE_DTYPE,
    payload_dtype: str = NOPE_BLOCK_PAYLOAD_DTYPE,
    suffix_elements: int = 0,
    suffix_dtype: str = C8_MLA_ROTARY_DTYPE,
) -> "CacheFormat":
    """Declared block-quantized MLA row format shared by capacity, store and attention."""
    return CacheFormat(
        payload_dtype, tile, scale_dtype,
        suffix_elements=suffix_elements, suffix_dtype=suffix_dtype,
    )


def block_quantized_row_bytes(
    width: int,
    tile: int = NOPE_BLOCK_QUANT_TILE,
    *,
    scale_dtype: str = NOPE_BLOCK_SCALE_DTYPE,
    payload_dtype: str = NOPE_BLOCK_PAYLOAD_DTYPE,
    suffix_elements: int = 0,
) -> int:
    return block_quantized_mla_format(
        tile, scale_dtype=scale_dtype, payload_dtype=payload_dtype,
        suffix_elements=suffix_elements,
    ).row_bytes(width)


def payload_bytes(dtype: str) -> float:
    # Packed cache storage does not imply a native FP4 matrix compute peak.
    return 0.5 if dtype == "fp4_e2m1" else dtype_bytes(dtype)


def _align_up_int(value: int, alignment: int) -> int:
    if alignment <= 1 or value <= 0:
        return max(0, int(value))
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def mxfp8_kv_packet_section_bytes(
    num_kv_heads: int,
    block_size: int,
    k_dim: int,
    v_dim: int | None = None,
    *,
    page_alignment: int = 1,
) -> tuple[int, int, int, int, int]:
    """Exact byte sizes for ``QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`` packet order.

    Returns ``(k_bytes, k_scale_bytes, v_bytes, v_scale_bytes, padding_bytes)``
    in physical order ``K | K-scale | V | V-scale | padding``. For 1 head,
    128 tokens, and ``Dk=Dv=128``, the four sections occupy ``33,792 B`` before
    trailing page alignment.
    """
    heads = int(num_kv_heads)
    block = int(block_size)
    kd = int(k_dim)
    vd = kd if v_dim is None else int(v_dim)
    align = int(page_alignment)
    if heads <= 0 or block <= 0 or kd <= 0 or vd <= 0 or align <= 0:
        raise ModelingError("mxfp8_kv_packet_geometry_invalid")
    if kd % MXFP_KV_SCALE_ALIGN != 0:
        raise ModelingError(f"mxfp8_kv_k_dim_not_aligned:{kd}%{MXFP_KV_SCALE_ALIGN}")
    if block % MXFP_KV_SCALE_ALIGN != 0:
        raise ModelingError(f"mxfp8_kv_block_not_aligned:{block}%{MXFP_KV_SCALE_ALIGN}")
    if vd % MXFP_KV_V_DIM_ALIGN != 0:
        raise ModelingError(f"mxfp8_kv_v_dim_not_aligned:{vd}%{MXFP_KV_V_DIM_ALIGN}")
    k_bytes = heads * block * kd
    k_scale = heads * block * (kd // MXFP_KV_GROUP_SIZE)
    v_bytes = heads * block * vd
    v_scale = heads * block * (vd // MXFP_KV_GROUP_SIZE)
    unpadded = k_bytes + k_scale + v_bytes + v_scale
    padding = _align_up_int(unpadded, align) - unpadded
    return k_bytes, k_scale, v_bytes, v_scale, padding


def mxfp8_kv_packet_planes(
    num_kv_heads: int,
    block_size: int,
    k_dim: int,
    v_dim: int | None = None,
    *,
    page_alignment: int = 1,
    plane_prefix: str = "",
) -> tuple[CachePlaneABI, ...]:
    """Physical ``CachePlaneABI`` tuple for ``QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL``."""
    vd = int(k_dim) if v_dim is None else int(v_dim)
    k_b, ks_b, v_b, vs_b, pad_b = mxfp8_kv_packet_section_bytes(
        num_kv_heads, block_size, k_dim, vd, page_alignment=page_alignment,
    )
    pitch = k_b + ks_b + v_b + vs_b + pad_b
    pfx = f"{plane_prefix}." if plane_prefix else ""
    planes: list[CachePlaneABI] = [
        CachePlaneABI(
            plane_id=f"{pfx}k_payload",
            plane_kind="payload",
            logical_axes=("kv_heads", "block_tokens", "head_dim"),
            logical_shape=(int(num_kv_heads), int(block_size), int(k_dim)),
            physical_shape=(int(num_kv_heads), int(block_size), int(k_dim)),
            dtype=MXFP_KV_PAYLOAD_DTYPE,
            layout="PA_NZ",
            group_axes=(2,),
            group_extents=(MXFP_KV_GROUP_SIZE,),
            scale_dtype=MXFP_KV_SCALE_DTYPE,
            offset_in_page=0,
            alignment=1,
            page_pitch=pitch,
            allocation_frequency="page",
            write_trigger="append",
            bytes_per_unit=k_b,
        ),
        CachePlaneABI(
            plane_id=f"{pfx}k_scale",
            plane_kind="scale",
            logical_axes=("kv_heads", "block_tokens", "k_groups"),
            logical_shape=(int(num_kv_heads), int(block_size), int(k_dim) // MXFP_KV_GROUP_SIZE),
            physical_shape=(int(num_kv_heads), int(block_size), int(k_dim) // MXFP_KV_GROUP_SIZE),
            dtype=MXFP_KV_SCALE_DTYPE,
            layout="PA_NZ_SCALE",
            group_axes=(2,),
            group_extents=(MXFP_KV_GROUP_SIZE,),
            offset_in_page=k_b,
            alignment=1,
            page_pitch=pitch,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=ks_b,
        ),
        CachePlaneABI(
            plane_id=f"{pfx}v_payload",
            plane_kind="payload",
            logical_axes=("kv_heads", "block_tokens", "v_head_dim"),
            logical_shape=(int(num_kv_heads), int(block_size), vd),
            physical_shape=(int(num_kv_heads), int(block_size), vd),
            dtype=MXFP_KV_PAYLOAD_DTYPE,
            layout="PA_NZ",
            group_axes=(1,),
            group_extents=(MXFP_KV_GROUP_SIZE,),
            scale_dtype=MXFP_KV_SCALE_DTYPE,
            offset_in_page=k_b + ks_b,
            alignment=1,
            page_pitch=pitch,
            allocation_frequency="page",
            write_trigger="append",
            bytes_per_unit=v_b,
        ),
        CachePlaneABI(
            plane_id=f"{pfx}v_scale",
            plane_kind="scale",
            logical_axes=("kv_heads", "v_head_dim", "block_groups"),
            logical_shape=(int(num_kv_heads), vd, int(block_size) // MXFP_KV_GROUP_SIZE),
            physical_shape=(int(num_kv_heads), vd, int(block_size) // MXFP_KV_GROUP_SIZE),
            dtype=MXFP_KV_SCALE_DTYPE,
            layout="PA_NZ_SCALE",
            group_axes=(1,),
            group_extents=(MXFP_KV_GROUP_SIZE,),
            offset_in_page=k_b + ks_b + v_b,
            alignment=1,
            page_pitch=pitch,
            allocation_frequency="page",
            write_trigger="page_reassign",
            bytes_per_unit=vs_b,
        ),
    ]
    if pad_b > 0:
        planes.append(
            CachePlaneABI(
                plane_id=f"{pfx}trailing_padding",
                plane_kind="padding",
                logical_axes=("padding_bytes",),
                logical_shape=(pad_b,),
                physical_shape=(pad_b,),
                dtype="uint8",
                layout="ND",
                offset_in_page=k_b + ks_b + v_b + vs_b,
                alignment=int(page_alignment),
                page_pitch=pitch,
                allocation_frequency="page",
                write_trigger="page_assign",
                bytes_per_unit=pad_b,
            )
        )
    return tuple(planes)


def dense_c8_kv_planes(
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    *,
    scale_dtype: str = "fp32",
    offset_dtype: str = "int8",
    faquant: bool = False,
    num_q_heads: int = 0,
    plane_prefix: str = "",
) -> tuple[CachePlaneABI, ...]:
    """Explicit payload, static checkpoint scale, and INT8 offset planes for C8/FAQuant."""
    heads = int(num_kv_heads)
    block = int(block_size)
    dim = int(head_dim)
    if heads <= 0 or block <= 0 or dim <= 0:
        raise ModelingError("dense_c8_kv_geometry_invalid")
    pfx = f"{plane_prefix}." if plane_prefix else ""
    kv_plane_bytes = heads * block * dim
    scale_elems = heads * dim
    scale_bytes = exact_element_bytes(scale_elems, scale_dtype)
    planes: list[CachePlaneABI] = [
        CachePlaneABI(
            plane_id=f"{pfx}k_payload",
            plane_kind="payload",
            logical_axes=("block_tokens", "kv_heads", "head_dim"),
            logical_shape=(block, heads, dim),
            physical_shape=(block, heads, dim),
            dtype="int8",
            layout="ND",
            scale_dtype=scale_dtype,
            offset_dtype=offset_dtype,
            page_pitch=kv_plane_bytes,
            allocation_frequency="page",
            write_trigger="append",
            bytes_per_unit=kv_plane_bytes,
        ),
        CachePlaneABI(
            plane_id=f"{pfx}v_payload",
            plane_kind="payload",
            logical_axes=("block_tokens", "kv_heads", "head_dim"),
            logical_shape=(block, heads, dim),
            physical_shape=(block, heads, dim),
            dtype="int8",
            layout="ND",
            scale_dtype=scale_dtype,
            offset_dtype=offset_dtype,
            page_pitch=kv_plane_bytes,
            allocation_frequency="page",
            write_trigger="append",
            bytes_per_unit=kv_plane_bytes,
        ),
        CachePlaneABI(
            plane_id=f"{pfx}k_scale",
            plane_kind="scale",
            logical_axes=("kv_heads", "head_dim"),
            logical_shape=(heads, dim),
            physical_shape=(heads, dim),
            dtype=scale_dtype,
            layout="ND",
            allocation_frequency="checkpoint",
            write_trigger="load",
            bytes_per_unit=scale_bytes,
        ),
        CachePlaneABI(
            plane_id=f"{pfx}v_scale",
            plane_kind="scale",
            logical_axes=("kv_heads", "head_dim"),
            logical_shape=(heads, dim),
            physical_shape=(heads, dim),
            dtype=scale_dtype,
            layout="ND",
            allocation_frequency="checkpoint",
            write_trigger="load",
            bytes_per_unit=scale_bytes,
        ),
    ]
    if offset_dtype and offset_dtype != "none":
        off_bytes = exact_element_bytes(scale_elems, offset_dtype)
        for side in ("k", "v"):
            planes.append(
                CachePlaneABI(
                    plane_id=f"{pfx}{side}_offset",
                    plane_kind="offset",
                    logical_axes=("kv_heads", "head_dim"),
                    logical_shape=(heads, dim),
                    physical_shape=(heads, dim),
                    dtype=offset_dtype,
                    layout="ND",
                    allocation_frequency="checkpoint",
                    write_trigger="load",
                    bytes_per_unit=off_bytes,
                )
            )
    if faquant:
        q_heads = max(heads, int(num_q_heads or heads))
        q_scale_bytes = exact_element_bytes(q_heads * dim, scale_dtype)
        planes.append(
            CachePlaneABI(
                plane_id=f"{pfx}q_scale",
                plane_kind="scale",
                logical_axes=("q_heads", "head_dim"),
                logical_shape=(q_heads, dim),
                physical_shape=(q_heads, dim),
                dtype=scale_dtype,
                layout="ND",
                allocation_frequency="layer",
                write_trigger="load",
                bytes_per_unit=q_scale_bytes,
            )
        )
    return tuple(planes)


@dataclass(frozen=True)
class CacheFormat:
    dtype: str
    group_size: int = 0
    scale_dtype: str = "none"
    suffix_elements: int = 0
    suffix_dtype: str = "bf16"
    # Number of logical elements represented by one payload storage unit.
    # uint8 packed rows use 2; plain int8/FP8/FP16 rows use 1.
    packed_elements: int = 1
    alignment: int = 1

    def __post_init__(self):
        if (not isinstance(self.group_size, int) or isinstance(self.group_size, bool)
                or self.group_size < 0
                or bool(self.group_size) != (self.scale_dtype != "none")):
            raise ModelingError("cache_representation_scale_group_mismatch")
        if (not isinstance(self.suffix_elements, int)
                or isinstance(self.suffix_elements, bool) or self.suffix_elements < 0):
            raise ModelingError("cache_suffix_elements_invalid")
        if (not isinstance(self.packed_elements, int)
                or isinstance(self.packed_elements, bool) or self.packed_elements < 1):
            raise ModelingError("cache_packed_elements_invalid")
        if (not isinstance(self.alignment, int)
                or isinstance(self.alignment, bool) or self.alignment < 1):
            raise ModelingError("cache_alignment_invalid")
        if self.packed_elements != 1 and payload_bytes(self.dtype) != 1.0:
            # A sub-byte dtype already carries its own packing width; a second
            # element-to-container factor would divide the row payload twice.
            raise ModelingError("cache_packed_elements_requires_byte_container")

    def row_parts(self, width: int) -> tuple[int, int, int]:
        """Exact integer ``(payload_bytes, scale_bytes, padding_bytes)`` per row."""
        quantized = int(width) - int(self.suffix_elements)
        if quantized < 0:
            raise ModelingError("cache_suffix_exceeds_row_width")
        payload = (
            _ceil_div(quantized, self.packed_elements)
            if self.packed_elements > 1
            else exact_element_bytes(quantized, self.dtype)
        )
        suffix = (
            exact_element_bytes(self.suffix_elements, self.suffix_dtype)
            if self.suffix_elements else 0
        )
        scales = (
            exact_element_bytes(_ceil_div(quantized, self.group_size), self.scale_dtype)
            if self.group_size else 0
        )
        unpadded = payload + suffix + scales
        padding = _align_up_int(unpadded, self.alignment) - unpadded
        return int(payload + suffix), int(scales), int(padding)

    def row_bytes(self, width: int) -> int:
        payload, scales, padding = self.row_parts(int(width))
        return payload + scales + padding

    def stored_bytes(self, width: int, rows: int = 1) -> int:
        """Exact bytes of ``rows`` complete rows, including suffix, scales, and padding."""
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise ModelingError("cache_rows_must_be_nonnegative_integer")
        return rows * self.row_bytes(width)

    def transaction_count(self, width: int, rows: int = 1, granule: int = 1) -> int:
        """Physical transactions at a declared path granule, not logical bytes."""
        if not isinstance(granule, int) or isinstance(granule, bool) or granule < 1:
            raise ModelingError("cache_transaction_granule_must_be_positive_integer")
        total = self.stored_bytes(width, rows)
        return (total + granule - 1) // granule

    def transaction_bytes(self, width: int, rows: int = 1, granule: int = 1) -> int:
        """Wire/line bytes after granule rounding of one row sweep."""
        return self.transaction_count(width, rows, granule) * granule


def _nope_mla_layers(model: Any) -> bool:
    """True when every MLA layer declares a zero-width rotary head."""
    layers = [plan for plan in getattr(model, "layer_plans", ())
              if getattr(plan, "attention", "") == "mla"]
    return bool(layers) and all(
        int(getattr(plan, "qk_rope_head_dim", 0) or 0) == 0 for plan in layers)


def _lookup_region_decl(
    metadata: Mapping[str, Any],
    region: str,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> Mapping[str, Any] | None:
    keys = (f"{region}:{phase}", region) if phase else (region,)
    if plan is not None:
        per_layer = metadata.get("layer_cache_representations")
        if isinstance(per_layer, Mapping):
            layer_map = per_layer.get(str(plan.index))
            if layer_map is None:
                layer_map = per_layer.get(plan.index)
            if isinstance(layer_map, Mapping):
                for key in keys:
                    if key in layer_map and layer_map[key] is not None:
                        return layer_map[key]
    global_regions = metadata.get("cache_representations")
    if isinstance(global_regions, Mapping):
        for key in keys:
            if key in global_regions and global_regions[key] is not None:
                return global_regions[key]
    return None


def _parse_format_dict(raw: Mapping[str, Any], region: str) -> CacheFormat:
    if set(raw) - ALLOWED_CACHE_REPR_FIELDS:
        raise ModelingError("cache_representation_fields_unknown:" + region)
    dtype = raw["dtype"]
    payload_bytes(dtype)
    group = raw.get("group_size", 0)
    scale = raw.get("scale_dtype", "none")
    if not isinstance(group, int) or isinstance(group, bool) or group < 0 or (bool(group) != (scale != "none")):
        raise ModelingError("cache_representation_scale_group_mismatch:" + region)
    if group:
        dtype_bytes(scale)
    suffix = raw.get("suffix_elements", 0)
    suffix_dtype = raw.get("suffix_dtype", "bf16")
    if not isinstance(suffix, int) or isinstance(suffix, bool) or suffix < 0:
        raise ModelingError("cache_suffix_elements_invalid:" + region)
    dtype_bytes(suffix_dtype)
    packed = raw.get("packed_elements", 1)
    if not isinstance(packed, int) or isinstance(packed, bool) or packed < 1:
        raise ModelingError("cache_representation_packed_elements_invalid:" + region)
    alignment = raw.get("alignment", 1)
    if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment < 1:
        raise ModelingError("cache_representation_alignment_invalid:" + region)
    return CacheFormat(dtype, group, scale, suffix, suffix_dtype, packed, alignment)


def _declared_kv_tag(model: Any) -> str:
    metadata = dict(getattr(model, "metadata", ()) or ())
    quant = getattr(model, "quant", None)
    for raw in (
        metadata.get("kv_cache_tag"),
        metadata.get("kv_cache_type"),
        getattr(quant, "kv_dtype", "") if quant is not None else "",
        getattr(model, "kv_dtype", ""),
    ):
        text = str(raw or "").strip()
        if text:
            return text
    return ""


def cache_format(
    model: Any,
    region: str,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> CacheFormat:
    """Resolve one physical cache format per native layer owner, role, and phase."""
    metadata = dict(getattr(model, "metadata", ()) or ())
    raw = _lookup_region_decl(metadata, region, plan=plan, phase=phase)
    if raw is None and region in {"gqa_kv_key", "gqa_kv_value"}:
        raw = _lookup_region_decl(metadata, "gqa_kv", plan=plan, phase=phase)
    if raw is not None:
        return _parse_format_dict(raw, region)
    quant = getattr(model, "quant", None)
    raw_tag = _declared_kv_tag(model)
    upper_tag = raw_tag.upper().replace("-", "_")
    if upper_tag == MXFP_KV_TAG and region in {"gqa_kv", "gqa_kv_key", "gqa_kv_value"}:
        return CacheFormat(MXFP_KV_PAYLOAD_DTYPE, MXFP_KV_GROUP_SIZE, MXFP_KV_SCALE_DTYPE)
    if upper_tag in {C8_KV_TAG, FAQUANT_KV_TAG} and region in {"gqa_kv", "gqa_kv_key", "gqa_kv_value"}:
        return CacheFormat("int8")
    resolved = str(
        (getattr(quant, "kv_dtype", "") if quant is not None else "")
        or getattr(model, "kv_dtype", "bf16")
        or "bf16"
    ).strip().lower()
    if resolved.upper().replace("-", "_") == MXFP_KV_TAG:
        return CacheFormat(MXFP_KV_PAYLOAD_DTYPE, MXFP_KV_GROUP_SIZE, MXFP_KV_SCALE_DTYPE)
    scale_override = str(
        metadata.get("kv_scale_dtype")
        or (getattr(quant, "kv_scale_dtype", "") if quant is not None else "")
        or ""
    ).strip().lower()
    if region == "mla_kv":
        if plan is not None and getattr(plan, "attention", "") == "mla":
            rope = int(getattr(plan, "qk_rope_head_dim", 0) or 0)
            latent = int(getattr(plan, "kv_lora_rank", 0) or 0)
            is_nope = rope == 0
            if resolved in NOPE_FP8_PAYLOAD_DTYPES:
                scale = scale_override or NOPE_BLOCK_SCALE_DTYPE
                if is_nope:
                    return block_quantized_mla_format(scale_dtype=scale)
                if latent > 0 and (
                    int(getattr(plan, "cache_row_elements", 0) or 0) in {0, latent + rope}
                ):
                    return block_quantized_mla_format(
                        NOPE_BLOCK_QUANT_TILE, scale_dtype=scale,
                        payload_dtype=NOPE_BLOCK_PAYLOAD_DTYPE,
                        suffix_elements=rope, suffix_dtype=C8_MLA_ROTARY_DTYPE,
                    )
            if resolved == "int8" and latent > 0 and (
                int(getattr(plan, "cache_row_elements", 0) or 0) in {0, latent + rope}
            ):
                scale = scale_override or C8_MLA_SCALE_DTYPE
                return block_quantized_mla_format(
                    C8_MLA_SCALE_GROUP, scale_dtype=scale, payload_dtype="int8",
                    suffix_elements=rope, suffix_dtype=C8_MLA_ROTARY_DTYPE,
                )
        elif resolved in NOPE_FP8_PAYLOAD_DTYPES and _nope_mla_layers(model):
            scale = scale_override or NOPE_BLOCK_SCALE_DTYPE
            return block_quantized_mla_format(scale_dtype=scale)
    return CacheFormat(resolved)


def gqa_kv_formats(
    model: Any,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> tuple[CacheFormat, CacheFormat, bool]:
    """Resolve shared or asymmetric K/V ``CacheFormat`` pair for one GQA owner."""
    metadata = dict(getattr(model, "metadata", ()) or ())
    key_raw = _lookup_region_decl(metadata, "gqa_kv_key", plan=plan, phase=phase)
    val_raw = _lookup_region_decl(metadata, "gqa_kv_value", plan=plan, phase=phase)
    if key_raw is not None or val_raw is not None:
        return (
            cache_format(model, "gqa_kv_key", plan=plan, phase=phase),
            cache_format(model, "gqa_kv_value", plan=plan, phase=phase),
            False,
        )
    upper_tag = _declared_kv_tag(model).upper().replace("-", "_")
    if upper_tag == MXFP_KV_TAG:
        fmt = CacheFormat(MXFP_KV_PAYLOAD_DTYPE, MXFP_KV_GROUP_SIZE, MXFP_KV_SCALE_DTYPE)
        return fmt, fmt, False
    shared = cache_format(model, "gqa_kv", plan=plan, phase=phase)
    return shared, shared, True


def _row_planes_from_format(
    fmt: CacheFormat,
    width: int,
    *,
    prefix: str,
    block_size: int,
) -> tuple[CachePlaneABI, ...]:
    """Decompose one ``CacheFormat`` row into separate payload, suffix, scale, and padding planes."""
    quantized = int(width) - int(fmt.suffix_elements)
    if quantized < 0:
        raise ModelingError("cache_suffix_exceeds_row_width")
    payload_only = (
        _ceil_div(quantized, fmt.packed_elements)
        if fmt.packed_elements > 1
        else exact_element_bytes(quantized, fmt.dtype)
    )
    suffix_b = (
        exact_element_bytes(fmt.suffix_elements, fmt.suffix_dtype)
        if fmt.suffix_elements else 0
    )
    scale_b = (
        exact_element_bytes(_ceil_div(quantized, fmt.group_size), fmt.scale_dtype)
        if fmt.group_size else 0
    )
    unpadded = payload_only + suffix_b + scale_b
    pad_b = _align_up_int(unpadded, fmt.alignment) - unpadded
    pitch = unpadded + pad_b
    planes: list[CachePlaneABI] = [
        CachePlaneABI(
            plane_id=f"{prefix}.payload",
            plane_kind="payload",
            logical_axes=("tokens", "elements"),
            logical_shape=(1, quantized),
            physical_shape=(1, quantized),
            dtype=fmt.dtype,
            packed_elements=fmt.packed_elements,
            layout="packed_row" if (fmt.group_size or fmt.packed_elements > 1) else "ND",
            group_axes=(1,) if fmt.group_size else (),
            group_extents=(fmt.group_size,) if fmt.group_size else (),
            scale_dtype=fmt.scale_dtype if fmt.group_size else "",
            suffix_shape=(fmt.suffix_elements,) if fmt.suffix_elements else (),
            suffix_dtype=fmt.suffix_dtype if fmt.suffix_elements else "",
            offset_in_page=0,
            alignment=fmt.alignment,
            page_pitch=pitch,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=payload_only,
        )
    ]
    cursor = payload_only
    if suffix_b > 0:
        planes.append(
            CachePlaneABI(
                plane_id=f"{prefix}.suffix",
                plane_kind="suffix",
                logical_axes=("tokens", "suffix_elements"),
                logical_shape=(1, fmt.suffix_elements),
                physical_shape=(1, fmt.suffix_elements),
                dtype=fmt.suffix_dtype,
                layout="ND",
                offset_in_page=cursor,
                alignment=1,
                page_pitch=pitch,
                allocation_frequency="token",
                write_trigger="append",
                bytes_per_unit=suffix_b,
            )
        )
        cursor += suffix_b
    if scale_b > 0:
        groups = _ceil_div(quantized, fmt.group_size)
        planes.append(
            CachePlaneABI(
                plane_id=f"{prefix}.scale",
                plane_kind="scale",
                logical_axes=("tokens", "scale_groups"),
                logical_shape=(1, groups),
                physical_shape=(1, groups),
                dtype=fmt.scale_dtype,
                layout="ND",
                group_axes=(1,),
                group_extents=(fmt.group_size,),
                offset_in_page=cursor,
                alignment=1,
                page_pitch=pitch,
                allocation_frequency="token",
                write_trigger="append",
                bytes_per_unit=scale_b,
            )
        )
        cursor += scale_b
    if pad_b > 0:
        planes.append(
            CachePlaneABI(
                plane_id=f"{prefix}.padding",
                plane_kind="padding",
                logical_axes=("padding_bytes",),
                logical_shape=(pad_b,),
                physical_shape=(pad_b,),
                dtype="uint8",
                layout="ND",
                offset_in_page=cursor,
                alignment=fmt.alignment,
                page_pitch=pitch,
                allocation_frequency="page" if block_size > 1 else "token",
                write_trigger="page_assign",
                bytes_per_unit=pad_b,
            )
        )
    return tuple(planes)


def _index_consumer_keys(model: Any, owner_layer: int, scope: str) -> tuple[str, ...]:
    consumers: list[str] = []
    for candidate in getattr(model, "layer_plans", ()):
        if int(getattr(candidate, "index_k_owner_layer_id", -1)) == int(owner_layer):
            consumers.append(f"{scope}:layer{candidate.index}:indexer")
    return tuple(consumers)


def _linear_cache_abis(
    plan: Any,
    parallel: Any,
    *,
    scope: str,
    native_layer: int,
    phase: str,
    deployment_role: str,
    runtime: Any,
    stage: int,
    checkpoint_digest: str,
) -> list[CacheStateABI]:
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    kda_axis = str(dict(getattr(runtime, "metadata", ()) or ()).get("kda_parallel_axis", "")).lower()
    shards = tp * (1 if (plan.attention != "kda" or kda_axis == "tp") else pcp)
    rec_elems = _ceil_div(int(plan.recurrent_state_elements), shards)
    rec_bytes = exact_element_bytes(rec_elems, str(plan.recurrent_state_dtype))
    conv_elems = _ceil_div(int(plan.conv_state_elements), shards)
    conv_bytes = exact_element_bytes(conv_elems, str(plan.conv_state_dtype))
    rec_place = CachePlacementRule(
        pp_stage=int(stage),
        memory_tier="hbm",
        token_ownership_axis="none",
        state_shard_axis="tp_x_pcp" if (plan.attention == "kda" and shards > tp) else "tp",
    )
    out: list[CacheStateABI] = []
    if rec_bytes > 0:
        out.append(
            CacheStateABI(
                scope=scope,
                native_layer=native_layer,
                component_role="recurrent_state",
                producer_key=f"{scope}:layer{native_layer}:recurrent_state",
                planes=(CachePlaneABI(
                    plane_id=f"l{native_layer}.recurrent_state",
                    plane_kind="payload",
                    logical_axes=("elements",),
                    logical_shape=(rec_elems,),
                    physical_shape=(rec_elems,),
                    dtype=str(plan.recurrent_state_dtype),
                    allocation_frequency="request_slot",
                    write_trigger="accepted_commit",
                    bytes_per_unit=rec_bytes,
                ),),
                phase=phase,
                deployment_role=deployment_role,
                checkpoint_digest=checkpoint_digest,
                placement=rec_place,
            )
        )
    if conv_bytes > 0:
        kernel = max(1, int(getattr(plan, "conv_kernel", 1) or 1))
        out.append(
            CacheStateABI(
                scope=scope,
                native_layer=native_layer,
                component_role="conv_history",
                producer_key=f"{scope}:layer{native_layer}:conv_history",
                planes=(CachePlaneABI(
                    plane_id=f"l{native_layer}.conv_history",
                    plane_kind="payload",
                    logical_axes=("elements",),
                    logical_shape=(conv_elems,),
                    physical_shape=(conv_elems,),
                    dtype=str(plan.conv_state_dtype),
                    allocation_frequency="request_slot",
                    write_trigger="append",
                    bytes_per_unit=conv_bytes,
                ),),
                phase=phase,
                deployment_role=deployment_role,
                checkpoint_digest=checkpoint_digest,
                history_slots=max(1, kernel - 1),
                placement=rec_place,
            )
        )
    return out


def _gqa_cache_abis(
    model: Any,
    plan: Any,
    *,
    scope: str,
    native_layer: int,
    phase: str,
    deployment_role: str,
    tp: int,
    block: int,
    interleave: int,
    placement: CachePlacementRule,
    checkpoint_digest: str,
) -> list[CacheStateABI]:
    from mindie_sim.schema.parallel import local_kv_heads

    if 0 <= plan.kv_owner_layer_id < plan.index:
        return []
    local_heads = local_kv_heads(plan.num_kv_heads, tp)
    upper_tag = _declared_kv_tag(model).upper().replace("-", "_")
    if upper_tag == MXFP_KV_TAG:
        planes = mxfp8_kv_packet_planes(
            local_heads, block, plan.head_dim, plan.head_dim,
            plane_prefix=f"l{native_layer}.gqa",
        )
        return [
            CacheStateABI(
                scope=scope,
                native_layer=native_layer,
                component_role="kv_combined",
                producer_key=f"{scope}:layer{native_layer}:gqa_kv",
                planes=planes,
                phase=phase,
                deployment_role=deployment_role,
                checkpoint_digest=checkpoint_digest,
                kv_block_size=block,
                kernel_block_size=block,
                cp_interleave_size=interleave,
                window=plan.window or None,
                placement=placement,
            )
        ]
    if upper_tag in {C8_KV_TAG, FAQUANT_KV_TAG}:
        q_heads = max(local_heads, int(getattr(plan, "num_q_heads", 0) or local_heads) // max(1, tp))
        planes = dense_c8_kv_planes(
            local_heads, block, plan.head_dim,
            faquant=(upper_tag == FAQUANT_KV_TAG),
            num_q_heads=q_heads,
            plane_prefix=f"l{native_layer}.gqa",
        )
        return [
            CacheStateABI(
                scope=scope,
                native_layer=native_layer,
                component_role="kv_combined",
                producer_key=f"{scope}:layer{native_layer}:gqa_kv",
                planes=planes,
                phase=phase,
                deployment_role=deployment_role,
                checkpoint_digest=checkpoint_digest,
                kv_block_size=block,
                kernel_block_size=block,
                cp_interleave_size=interleave,
                window=plan.window or None,
                placement=placement,
            )
        ]
    key_fmt, val_fmt, _shared = gqa_kv_formats(model, plan=plan, phase=phase)
    side_width = local_heads * plan.head_dim
    out: list[CacheStateABI] = []
    for role, suffix, fmt in (("kv_k", "k", key_fmt), ("kv_v", "v", val_fmt)):
        out.append(
            CacheStateABI(
                scope=scope,
                native_layer=native_layer,
                component_role=role,
                producer_key=f"{scope}:layer{native_layer}:{role}",
                planes=_row_planes_from_format(
                    fmt, side_width, prefix=f"l{native_layer}.{suffix}", block_size=block,
                ),
                phase=phase,
                deployment_role=deployment_role,
                checkpoint_digest=checkpoint_digest,
                kv_block_size=block,
                kernel_block_size=block,
                cp_interleave_size=interleave,
                window=plan.window or None,
                placement=placement,
            )
        )
    return out


def _mla_cache_abis(
    model: Any,
    plan: Any,
    *,
    scope: str,
    native_layer: int,
    phase: str,
    deployment_role: str,
    block: int,
    interleave: int,
    placement: CachePlacementRule,
    checkpoint_digest: str,
) -> list[CacheStateABI]:
    region = "mla_kv" if plan.cache_kind == "dense" else "window"
    role = "mla_latent" if plan.cache_kind == "dense" else "window_kv"
    fmt = cache_format(model, region, plan=plan, phase=phase)
    out = [
        CacheStateABI(
            scope=scope,
            native_layer=native_layer,
            component_role=role,
            producer_key=f"{scope}:layer{native_layer}:{region}",
            planes=_row_planes_from_format(
                fmt, int(plan.cache_row_elements), prefix=f"l{native_layer}.{region}",
                block_size=block,
            ),
            phase=phase,
            deployment_role=deployment_role,
            checkpoint_digest=checkpoint_digest,
            kv_block_size=block,
            kernel_block_size=block,
            cp_interleave_size=interleave,
            window=plan.window or None,
            compression_ratio=max(1, int(plan.compress_ratio or 1)),
            placement=placement,
        )
    ]
    if plan.owns_long_kv and plan.cache_kind != "dense":
        comp_fmt = cache_format(model, "compressed_kv", plan=plan, phase=phase)
        out.append(
            CacheStateABI(
                scope=scope,
                native_layer=native_layer,
                component_role="compressed_kv",
                producer_key=f"{scope}:layer{native_layer}:compressed_kv",
                planes=_row_planes_from_format(
                    comp_fmt, int(plan.cache_row_elements),
                    prefix=f"l{native_layer}.compressed_kv", block_size=block,
                ),
                phase=phase,
                deployment_role=deployment_role,
                checkpoint_digest=checkpoint_digest,
                kv_block_size=block,
                kernel_block_size=block,
                cp_interleave_size=interleave,
                compression_ratio=max(1, int(plan.compress_ratio or 1)),
                placement=placement,
            )
        )
    return out


def _index_cache_abi(
    model: Any,
    plan: Any,
    *,
    scope: str,
    native_layer: int,
    phase: str,
    deployment_role: str,
    block: int,
    interleave: int,
    placement: CachePlacementRule,
    checkpoint_digest: str,
) -> CacheStateABI:
    idx_fmt = _resolve_index_cache_format(model, plan, phase=phase)
    consumers = _index_consumer_keys(model, plan.index, scope)
    return CacheStateABI(
        scope=scope,
        native_layer=native_layer,
        component_role="index_k",
        producer_key=f"{scope}:layer{native_layer}:index_k",
        consumer_keys=consumers,
        planes=_row_planes_from_format(
            idx_fmt, int(plan.index_dim), prefix=f"l{native_layer}.index_k",
            block_size=block,
        ),
        phase=phase,
        deployment_role=deployment_role,
        checkpoint_digest=checkpoint_digest,
        kv_block_size=block,
        kernel_block_size=block,
        cp_interleave_size=interleave,
        compression_ratio=max(1, int(plan.compress_ratio or 1)),
        pooling_ratio=max(1, int(plan.index_kpool or 1)),
        tail_slots=(2 * int(plan.index_kpool) if int(plan.index_kpool or 0) > 1 else 0),
        placement=placement,
    )


def resolve_layer_cache_abis(
    model: Any,
    plan: Any,
    parallel: Any,
    *,
    scope: str = "target",
    phase: str = "decode",
    deployment_role: str = "colocated",
    runtime: Any = None,
    stage: int = 0,
    checkpoint_digest: str = "",
) -> tuple[CacheStateABI, ...]:
    """Build typed L0 ``CacheStateABI`` entries for one native layer owner."""
    from mindie_sim.catalog.context_parallel import cp_interleave, layer_dcp_degree

    native_layer = int(getattr(plan, "native_layer_id", -1))
    if native_layer < 0:
        native_layer = int(plan.index)
    block = int(getattr(runtime, "kv_block_size", 0) or DEFAULT_CACHE_PAGE_BLOCK)
    interleave = int(cp_interleave(runtime)) if runtime is not None else block
    cp_deg = layer_dcp_degree(plan, parallel)
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    placement = CachePlacementRule(
        pp_stage=int(stage),
        memory_tier="hbm",
        token_ownership_axis="cp_striped" if cp_deg > 1 else "replicated",
        replication_factor=max(1, tp // max(1, int(getattr(plan, "num_kv_heads", 1) or 1)))
        if plan.attention in {"gqa", "projection_only"} else 1,
    )
    out: list[CacheStateABI] = []
    if plan.attention in LINEAR_ATTENTION_KINDS:
        out.extend(_linear_cache_abis(
            plan, parallel, scope=scope, native_layer=native_layer, phase=phase,
            deployment_role=deployment_role, runtime=runtime, stage=stage,
            checkpoint_digest=checkpoint_digest,
        ))
    elif plan.attention in {"gqa", "projection_only"}:
        out.extend(_gqa_cache_abis(
            model, plan, scope=scope, native_layer=native_layer, phase=phase,
            deployment_role=deployment_role, tp=tp, block=block,
            interleave=interleave, placement=placement,
            checkpoint_digest=checkpoint_digest,
        ))
    elif plan.attention == "mla":
        out.extend(_mla_cache_abis(
            model, plan, scope=scope, native_layer=native_layer, phase=phase,
            deployment_role=deployment_role, block=block, interleave=interleave,
            placement=placement, checkpoint_digest=checkpoint_digest,
        ))
    if getattr(plan, "owns_index_k", False):
        out.append(_index_cache_abi(
            model, plan, scope=scope, native_layer=native_layer, phase=phase,
            deployment_role=deployment_role, block=block, interleave=interleave,
            placement=placement, checkpoint_digest=checkpoint_digest,
        ))
    return tuple(out)


def _resolve_index_cache_format(model: Any, plan: Any, *, phase: str | None = None) -> CacheFormat:
    metadata = dict(getattr(model, "metadata", ()) or ())
    raw = _lookup_region_decl(metadata, "index_k", plan=plan, phase=phase)
    if raw is not None:
        return _parse_format_dict(raw, "index_k")
    quant = getattr(model, "quant", None)
    idx_dtype = str((getattr(quant, "index_dtype", "") if quant is not None else "") or "bf16").strip().lower()
    idx_scale = str((getattr(quant, "index_scale_dtype", "") if quant is not None else "") or "none").strip().lower()
    if idx_scale and idx_scale != "none" and int(plan.index_dim) > 0:
        return CacheFormat(idx_dtype, int(plan.index_dim), idx_scale)
    return CacheFormat(idx_dtype)


def with_packed_mla_cache(model: Any, rope: int):
    """Install the selected source's mixed row ABI in capacity accounting."""
    from dataclasses import replace
    metadata = dict(model.metadata)
    regions = dict(metadata.get("cache_representations", {}))
    declaration = {"dtype": "int8", "group_size": C8_MLA_SCALE_GROUP, "scale_dtype": C8_MLA_SCALE_DTYPE,
                   "suffix_elements": rope, "suffix_dtype": C8_MLA_ROTARY_DTYPE}
    if "mla_kv" in regions and regions["mla_kv"] != declaration:
        raise ModelingError("packed_mla_cache_conflicts_with_declared_representation")
    regions["mla_kv"] = declaration
    return replace(model, metadata=tuple({**metadata, "cache_representations": regions}.items()))
