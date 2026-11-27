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
from dataclasses import dataclass, replace
from typing import Any

from mindie_sim.catalog.cache_pools import (
    LINEAR_ATTENTION_KINDS,
    _ceil_div,
    exact_element_bytes,
)
from mindie_sim.schema.checkpoint import (
    CacheQuantRule,
    CheckpointProvenance,
    CheckpointQuantMap,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.stack import (
    CachePlaneABI,
    CacheStateABI,
    resolve_checkpoint_cache_abis,
)


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


def _layer_quant(model: Any, plan: Any = None) -> tuple[Any, bool]:
    """Resolve effective per-layer ``QuantSpec``, reporting whether a layer override matched."""
    base = getattr(model, "quant", None)
    if base is None or plan is None or not getattr(base, "layer_quant_overrides", ()):
        return base, False
    from mindie_sim.catalog.quant import apply_layer_quant_override

    resolved = apply_layer_quant_override(base, plan)
    return resolved, (resolved is not base)


def _declared_kv_tag(model: Any, *, plan: Any = None) -> str:
    metadata = dict(getattr(model, "metadata", ()) or ())
    base_quant = getattr(model, "quant", None)
    quant, has_override = _layer_quant(model, plan)
    if has_override:
        base_kv = str(getattr(base_quant, "kv_dtype", "") if base_quant is not None else "").strip()
        layer_kv = str(getattr(quant, "kv_dtype", "") if quant is not None else "").strip()
        if layer_kv and layer_kv != base_kv:
            return layer_kv
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
    quant, _ = _layer_quant(model, plan)
    raw_tag = _declared_kv_tag(model, plan=plan)
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
    upper_tag = _declared_kv_tag(model, plan=plan).upper().replace("-", "_")
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
                allocation_frequency="token",
                write_trigger="append",
                bytes_per_unit=pad_b,
            )
        )
    return tuple(planes)


_SCHEMA_CACHE_TAGS = frozenset({
    "FLOAT", "BF16", "BFLOAT16", "FP16", "FLOAT16",
    "C8", "INT8", "INT8_SYM", "INT8_DYNAMIC", "FAQUANT", "FAKQUANT",
    "FP8", "FP8_DYNAMIC", "FP8_E4M3", "FP8_E5M2",
    "MXFP8", "MXFP8_DYNAMIC", MXFP_KV_TAG,
})


def _materialize_plane_bytes(planes: tuple[CachePlaneABI, ...]) -> tuple[CachePlaneABI, ...]:
    return tuple(
        plane if plane.bytes_per_unit > 0 else replace(plane, bytes_per_unit=plane.exact_bytes())
        for plane in planes
    )


def _mqsmla_packed_planes(
    model: Any,
    plan: Any,
    region: str,
    *,
    phase: str,
    block_size: int,
) -> tuple[CachePlaneABI, ...] | None:
    """Emit canonical MQSMLA packed planes matching ``mqsmla_layout.packed_row_parts``."""
    del block_size
    from mindie_sim.catalog.mqsmla_layout import (
        HEAD_WIDTH,
        NATIVE_GROUP,
        ROPE_WIDTH,
        packed_row_parts,
    )

    parts = packed_row_parts(model, region, int(plan.cache_row_elements), plan=plan, phase=phase)
    if parts is None:
        return None
    data_b, scale_b, pad_b = parts
    fmt = cache_format(model, region, plan=plan, phase=phase)
    nope = HEAD_WIDTH - ROPE_WIDTH
    rope_b = data_b - nope
    scale_groups = nope // NATIVE_GROUP
    pitch = data_b + scale_b + pad_b
    planes: list[CachePlaneABI] = [
        CachePlaneABI(
            plane_id="mla_latent",
            plane_kind="payload",
            logical_axes=("tokens", "dim"),
            logical_shape=(1, nope),
            physical_shape=(1, nope),
            dtype=fmt.dtype,
            layout="packed_row",
            group_axes=(1,),
            group_extents=(NATIVE_GROUP,),
            scale_dtype=fmt.scale_dtype,
            suffix_shape=(ROPE_WIDTH,),
            suffix_dtype=C8_MLA_ROTARY_DTYPE,
            offset_in_page=0,
            alignment=1,
            page_pitch=pitch,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=nope,
        ),
        CachePlaneABI(
            plane_id="mla_rotary",
            plane_kind="suffix",
            logical_axes=("tokens", "dim"),
            logical_shape=(1, ROPE_WIDTH),
            physical_shape=(1, ROPE_WIDTH),
            dtype=C8_MLA_ROTARY_DTYPE,
            layout="ND",
            offset_in_page=nope,
            alignment=1,
            page_pitch=pitch,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=rope_b,
        ),
        CachePlaneABI(
            plane_id="mla_latent_scale",
            plane_kind="scale",
            logical_axes=("tokens", "scale_groups"),
            logical_shape=(1, scale_groups),
            physical_shape=(1, scale_groups),
            dtype=fmt.scale_dtype,
            layout="ND",
            group_axes=(1,),
            group_extents=(NATIVE_GROUP,),
            offset_in_page=data_b,
            alignment=1,
            page_pitch=pitch,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=scale_b,
        ),
    ]
    if pad_b > 0:
        planes.append(
            CachePlaneABI(
                plane_id="mla_padding",
                plane_kind="padding",
                logical_axes=("padding_bytes",),
                logical_shape=(pad_b,),
                physical_shape=(pad_b,),
                dtype="uint8",
                layout="ND",
                offset_in_page=data_b + scale_b,
                alignment=1,
                page_pitch=pitch,
                allocation_frequency="token",
                write_trigger="append",
                bytes_per_unit=pad_b,
            )
        )
    return tuple(planes)


def _layer_checkpoint_for_cache(
    model: Any,
    plan: Any,
    *,
    phase: str,
) -> tuple[CheckpointQuantMap, str, bool]:
    metadata = dict(getattr(model, "metadata", ()) or ())
    quant, _ = _layer_quant(model, plan)
    orig_native = str((getattr(quant, "native_format", "") if quant is not None else "") or "bf16").strip().lower()
    native_dtype = orig_native
    raw_kv = _declared_kv_tag(model, plan=plan).upper().replace("-", "_")
    kv_scale = str(
        metadata.get("kv_scale_dtype")
        or (getattr(quant, "kv_scale_dtype", "") if quant is not None else "")
        or ""
    ).strip().lower()
    if raw_kv in {"", "AUTO", "NONE"}:
        kv_tag = "FLOAT"
    elif raw_kv in {"FP32", "FLOAT32"}:
        kv_tag = "FLOAT"
        native_dtype = "fp32"
    elif plan.attention == "mla" and raw_kv in {"INT8", "C8"}:
        kv_tag = "INT8_DYNAMIC"
        kv_scale = kv_scale or C8_MLA_SCALE_DTYPE
    elif plan.attention == "mla" and raw_kv in {"FP8", "FP8_E4M3", "FP8_E5M2", "FP8_DYNAMIC"}:
        kv_tag = raw_kv
        kv_scale = kv_scale or NOPE_BLOCK_SCALE_DTYPE
    elif raw_kv in _SCHEMA_CACHE_TAGS:
        kv_tag = raw_kv
    else:
        kv_tag = "FLOAT"

    idx_fmt = _resolve_index_cache_format(model, plan, phase=phase)
    idx_raw = idx_fmt.dtype.upper().replace("-", "_")
    custom_index = (
        _lookup_region_decl(metadata, "index_k", plan=plan, phase=phase) is not None
        or idx_raw not in _SCHEMA_CACHE_TAGS
        or (idx_raw in {"FP32", "FLOAT32"} and native_dtype != "fp32")
    )
    if custom_index:
        idx_tag = "FLOAT"
    elif idx_raw in {"", "AUTO", "NONE"}:
        idx_tag = orig_native.upper() if orig_native in {"bf16", "fp16"} else "FLOAT"
    else:
        idx_tag = idx_raw
    idx_scale = "" if idx_fmt.scale_dtype == "none" else idx_fmt.scale_dtype
    ckpt = CheckpointQuantMap(
        provenance=CheckpointProvenance(
            checkpoint_id=str(getattr(model, "name", "catalog") or "catalog"),
            origin="explicit_synthetic_preset",
            source_format="explicit_synthetic_preset",
        ),
        native_float_dtype=native_dtype,
        cache_rule=CacheQuantRule(
            kv_tag=kv_tag,
            index_tag=idx_tag,
            kv_scale_dtype=kv_scale,
            index_scale_dtype=idx_scale,
            provenance="native",
        ),
    )
    return ckpt, native_dtype, custom_index


def _split_recurrent_abi(
    abi: CacheStateABI,
    plan: Any,
    *,
    checkpoint_digest: str,
    original_attention: str,
) -> list[CacheStateABI]:
    out: list[CacheStateABI] = []
    rec_elems = int(getattr(plan, "recurrent_state_elements", 0) or 0)
    conv_elems = int(getattr(plan, "conv_state_elements", 0) or 0)
    for plane in _materialize_plane_bytes(abi.planes):
        if plane.bytes_per_unit <= 0:
            continue
        role = "conv_history" if plane.plane_id == "conv_history" else "recurrent_state"
        if role == "conv_history" and conv_elems <= 0:
            continue
        if role == "recurrent_state" and rec_elems <= 0:
            continue
        out.append(
            replace(
                abi,
                component_role=role,
                producer_key=f"{abi.scope}.l{abi.native_layer}.{role}",
                planes=(plane,),
                checkpoint_digest=checkpoint_digest or abi.checkpoint_digest,
                recipe_abi_id=f"{original_attention}_recurrent_state",
                history_slots=abi.history_slots if role == "conv_history" else 0,
            )
        )
    return out


def _overlay_explicit_kv_planes(
    model: Any,
    plan: Any,
    abi: CacheStateABI,
    *,
    parallel: Any,
    phase: str,
    block: int,
    metadata: Mapping[str, Any],
) -> list[CacheStateABI]:
    if plan.attention == "mla":
        if abi.component_role == "window_kv":
            abi = replace(abi, compression_ratio=1)
        region = (
            "mla_kv" if abi.component_role == "mla_latent"
            else ("window" if abi.component_role == "window_kv" else "compressed_kv")
        )
        mqsmla_planes = _mqsmla_packed_planes(model, plan, region, phase=phase, block_size=block)
        if mqsmla_planes is not None:
            return [replace(abi, planes=mqsmla_planes)]
        if _lookup_region_decl(metadata, region, plan=plan, phase=phase) is not None:
            fmt = cache_format(model, region, plan=plan, phase=phase)
            return [replace(abi, planes=_row_planes_from_format(
                fmt, int(plan.cache_row_elements), prefix=f"l{abi.native_layer}.{region}", block_size=block,
            ))]
        return [abi]
    upper_tag = _declared_kv_tag(model, plan=plan).upper().replace("-", "_")
    has_gqa_override = (
        upper_tag not in {MXFP_KV_TAG, C8_KV_TAG, FAQUANT_KV_TAG}
        and any(
            _lookup_region_decl(metadata, reg, plan=plan, phase=phase) is not None
            for reg in ("gqa_kv_key", "gqa_kv_value", "gqa_kv")
        )
    )
    if not has_gqa_override:
        return [abi]
    from mindie_sim.schema.parallel import local_kv_heads

    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    local_heads = local_kv_heads(plan.num_kv_heads or plan.num_heads or 1, tp)
    key_fmt, val_fmt, is_shared = gqa_kv_formats(model, plan=plan, phase=phase)
    kd = int(plan.head_dim)
    vd = int(getattr(plan, "v_head_dim", 0) or kd)
    if is_shared and kd == vd:
        return [
            replace(
                abi,
                component_role="kv_combined",
                producer_key=f"{abi.scope}.l{abi.native_layer}.kv_combined",
                planes=_row_planes_from_format(
                    key_fmt, 2 * local_heads * kd, prefix=f"l{abi.native_layer}.kv", block_size=block,
                ),
            )
        ]
    return [
        replace(
            abi,
            component_role=role,
            producer_key=f"{abi.scope}.l{abi.native_layer}.{role}",
            planes=_row_planes_from_format(fmt, local_heads * width, prefix=f"l{abi.native_layer}.{suffix}", block_size=block),
        )
        for role, suffix, fmt, width in (("kv_k", "k", key_fmt, kd), ("kv_v", "v", val_fmt, vd))
    ]


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
    """Build typed L0 ``CacheStateABI`` entries for one native layer owner.

    Delegates core cache-ABI construction to the sole native L0 builder
    :func:`mindie_sim.schema.stack.resolve_checkpoint_cache_abis`.
    """
    from mindie_sim.catalog.context_parallel import cp_interleave

    native_layer = int(getattr(plan, "native_layer_id", -1))
    if native_layer < 0:
        native_layer = int(plan.index)
    block = int(getattr(runtime, "kv_block_size", 0) or DEFAULT_CACHE_PAGE_BLOCK)
    interleave = int(cp_interleave(runtime)) if runtime is not None else block
    kda_axis_raw = str(dict(getattr(runtime, "metadata", ()) or ()).get("kda_parallel_axis", "")).lower()
    kda_axis = "tp" if kda_axis_raw == "tp" else "tp_x_pcp"
    target_scope = "draft" if scope in {"draft", "mtp", "dspark", "standalone_draft"} else "target"
    scoped_plan = plan if getattr(plan, "scope", "target") == target_scope else replace(plan, scope=target_scope)
    orig_attention = str(scoped_plan.attention or "").lower()
    delegated_plan = (
        replace(scoped_plan, attention="gdn")
        if orig_attention in LINEAR_ATTENTION_KINDS and orig_attention not in {"kda", "gdn"}
        else scoped_plan
    )

    owner_ids = {int(plan.index), native_layer}
    relevant: list[Any] = [delegated_plan]
    for cand in getattr(model, "layer_plans", ()):
        cand_native = int(getattr(cand, "native_layer_id", -1))
        if cand_native < 0:
            cand_native = int(cand.index)
        if cand_native == native_layer or int(cand.index) == int(plan.index):
            continue
        if (
            int(getattr(cand, "kv_owner_layer_id", -1)) in owner_ids
            or int(getattr(cand, "index_k_owner_layer_id", -1)) in owner_ids
        ):
            cand_scope = "draft" if getattr(cand, "scope", "target") in {"draft", "mtp", "dspark", "standalone_draft"} else "target"
            if cand_scope == target_scope:
                relevant.append(cand)

    ckpt, native_dtype, custom_index = _layer_checkpoint_for_cache(model, scoped_plan, phase=phase)
    kv_abis, idx_abis, rec_abis = resolve_checkpoint_cache_abis(
        tuple(relevant),
        ckpt,
        parallel,
        native_dtype,
        role=deployment_role,
        phase=phase,
        kv_block_size=block,
        cp_interleave_size=interleave,
        kda_parallel_axis=kda_axis,
        stage=int(stage),
    )

    metadata = dict(getattr(model, "metadata", ()) or ())
    out: list[CacheStateABI] = []
    for abi in rec_abis:
        if abi.native_layer == native_layer and abi.scope == target_scope:
            out.extend(_split_recurrent_abi(
                abi, scoped_plan, checkpoint_digest=checkpoint_digest, original_attention=orig_attention,
            ))
    for abi in kv_abis:
        if abi.native_layer != native_layer or abi.scope != target_scope:
            continue
        stamped = replace(
            abi,
            planes=_materialize_plane_bytes(abi.planes),
            checkpoint_digest=checkpoint_digest or abi.checkpoint_digest,
        )
        out.extend(_overlay_explicit_kv_planes(
            model, scoped_plan, stamped, parallel=parallel, phase=phase, block=block, metadata=metadata,
        ))
    for abi in idx_abis:
        if abi.native_layer != native_layer or abi.scope != target_scope:
            continue
        planes = (
            _row_planes_from_format(
                _resolve_index_cache_format(model, scoped_plan, phase=phase),
                int(scoped_plan.index_dim),
                prefix=f"l{native_layer}.index_k",
                block_size=block,
            )
            if custom_index
            else _materialize_plane_bytes(abi.planes)
        )
        out.append(replace(abi, planes=planes, checkpoint_digest=checkpoint_digest or abi.checkpoint_digest))
    return tuple(out)


def _resolve_index_cache_format(model: Any, plan: Any, *, phase: str | None = None) -> CacheFormat:
    metadata = dict(getattr(model, "metadata", ()) or ())
    raw = _lookup_region_decl(metadata, "index_k", plan=plan, phase=phase)
    if raw is not None:
        return _parse_format_dict(raw, "index_k")
    quant, _ = _layer_quant(model, plan)
    idx_dtype = str((getattr(quant, "index_dtype", "") if quant is not None else "") or "bf16").strip().lower()
    idx_scale = str((getattr(quant, "index_scale_dtype", "") if quant is not None else "") or "none").strip().lower()
    if idx_scale and idx_scale != "none" and int(plan.index_dim) > 0:
        return CacheFormat(idx_dtype, int(plan.index_dim), idx_scale)
    return CacheFormat(idx_dtype)


_DRAFT_CACHE_SCOPES = frozenset({"draft", "mtp", "dspark", "standalone_draft"})
_UNPACKED_DRAFT_MLA_DTYPES = frozenset({"bf16", "fp8", "fp8_e4m3"})


def _packed_mla_payload_dtype(raw_kv: str) -> str:
    return (
        NOPE_BLOCK_PAYLOAD_DTYPE
        if raw_kv in NOPE_FP8_PAYLOAD_DTYPES | {"fp8_e4m3fn"}
        else "int8"
    )


def with_packed_mla_cache(model: Any, rope: int):
    """Install the selected source's mixed row ABI in capacity accounting."""
    from dataclasses import replace
    metadata = dict(model.metadata)
    regions = dict(metadata.get("cache_representations", {}))
    quant = getattr(model, "quant", None)
    raw_kv = str(
        (getattr(quant, "kv_dtype", "") if quant is not None else "")
        or getattr(model, "kv_dtype", "")
        or ""
    ).strip().lower()
    global_packed_dtype = _packed_mla_payload_dtype(raw_kv)
    declaration = {"dtype": global_packed_dtype, "group_size": C8_MLA_SCALE_GROUP, "scale_dtype": C8_MLA_SCALE_DTYPE,
                   "suffix_elements": rope, "suffix_dtype": C8_MLA_ROTARY_DTYPE}
    if "mla_kv" in regions and regions["mla_kv"] != declaration:
        raise ModelingError("packed_mla_cache_conflicts_with_declared_representation")
    regions["mla_kv"] = declaration
    raw_layer_regions = metadata.get("layer_cache_representations")
    layer_regions: dict[str, dict[str, Any]] = (
        {str(k): dict(v) for k, v in raw_layer_regions.items() if isinstance(v, Mapping)}
        if isinstance(raw_layer_regions, Mapping) else {}
    )
    updated_layers = False
    for plan in getattr(model, "layer_plans", ()):
        if getattr(plan, "attention", "") != "mla":
            continue
        layer_q, _ = _layer_quant(model, plan)
        layer_kv = str(
            (getattr(layer_q, "kv_dtype", "") if layer_q is not None else "")
            or getattr(model, "kv_dtype", "")
            or ""
        ).strip().lower()
        layer_rope = int(getattr(plan, "qk_rope_head_dim", 0) or rope)
        if getattr(plan, "scope", "body") in _DRAFT_CACHE_SCOPES and layer_kv in _UNPACKED_DRAFT_MLA_DTYPES:
            layer_decl = (
                {"dtype": "bf16"}
                if layer_kv == "bf16"
                else {"dtype": NOPE_BLOCK_PAYLOAD_DTYPE, "group_size": NOPE_BLOCK_QUANT_TILE,
                      "scale_dtype": NOPE_BLOCK_SCALE_DTYPE, "suffix_elements": layer_rope,
                      "suffix_dtype": C8_MLA_ROTARY_DTYPE}
            )
        else:
            layer_packed_dtype = _packed_mla_payload_dtype(layer_kv)
            if layer_packed_dtype == global_packed_dtype and layer_rope == rope:
                continue
            layer_decl = {**declaration, "dtype": layer_packed_dtype, "suffix_elements": layer_rope}
        lkey = str(plan.index)
        entry = dict(layer_regions.get(lkey, {}))
        if "mla_kv" in entry and entry["mla_kv"] != layer_decl:
            raise ModelingError("packed_mla_cache_conflicts_with_declared_representation")
        entry["mla_kv"] = layer_decl
        layer_regions[lkey] = entry
        updated_layers = True
    next_meta = {**metadata, "cache_representations": regions}
    if updated_layers:
        next_meta["layer_cache_representations"] = layer_regions
    return replace(model, metadata=tuple(next_meta.items()))
