"""Declared packed cache rows shared by producers, readers and capacity.

[Role] Resolve separate original-window and compressed-cache representations.
[Invariants] A group scale is stored once per row group. Unspecified regions
  retain QuantSpec KV precision; no model name or timing selects a format.
[Timeouts & Locks] Constant arithmetic, no I/O or locks.
[Side-Effect Policy] Returns immutable format metadata only.
"""
from dataclasses import dataclass

from mindie_sim.catalog.cache_pools import _ceil_div, exact_element_bytes
from mindie_sim.schema.common import ModelingError, dtype_bytes


#: NoPE C8 block-quantized MLA cache: one FP8 row split into 128-element
#: scale groups.  A 512-element latent therefore stores ``512 B`` of payload
#: plus four FP16 scales (``8 B``) = ``520 B`` per row.
NOPE_BLOCK_QUANT_TILE = 128
NOPE_BLOCK_PAYLOAD_DTYPE = "fp8_e4m3"
NOPE_BLOCK_SCALE_DTYPE = "fp16"


def block_quantized_mla_format(tile=NOPE_BLOCK_QUANT_TILE):
    """Declared C8 NoPE row format shared by capacity, store and attention."""
    return CacheFormat(NOPE_BLOCK_PAYLOAD_DTYPE, tile, NOPE_BLOCK_SCALE_DTYPE)


def block_quantized_row_bytes(width, tile=NOPE_BLOCK_QUANT_TILE):
    return block_quantized_mla_format(tile).row_bytes(width)


def payload_bytes(dtype):
    # Packed cache storage does not imply a native FP4 matrix compute peak.
    return 0.5 if dtype == "fp4_e2m1" else dtype_bytes(dtype)


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
        if self.packed_elements != 1 and payload_bytes(self.dtype) != 1.0:
            # A sub-byte dtype already carries its own packing width; a second
            # element-to-container factor would divide the row payload twice.
            raise ModelingError("cache_packed_elements_requires_byte_container")

    def row_bytes(self, width):
        quantized = int(width) - int(self.suffix_elements)
        if quantized < 0:
            raise ModelingError("cache_suffix_exceeds_row_width")
        # Pure integer ceilings for packed payload, MX group exponents, suffix,
        # and explicit scale planes (zero float precision loss at 2**53 + 1).
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
        return payload + suffix + scales

    def stored_bytes(self, width, rows=1):
        """Exact bytes of ``rows`` complete rows, including suffix and scales."""
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise ModelingError("cache_rows_must_be_nonnegative_integer")
        return rows * self.row_bytes(width)

    def transaction_count(self, width, rows=1, granule=1):
        """Physical transactions at a declared path granule, not logical bytes."""
        if not isinstance(granule, int) or isinstance(granule, bool) or granule < 1:
            raise ModelingError("cache_transaction_granule_must_be_positive_integer")
        total = self.stored_bytes(width, rows)
        return (total + granule - 1) // granule

    def transaction_bytes(self, width, rows=1, granule=1):
        """Wire/line bytes after granule rounding of one row sweep."""
        return self.transaction_count(width, rows, granule) * granule


def _nope_mla_layers(model):
    """True when every MLA layer declares a zero-width rotary head."""
    layers = [plan for plan in getattr(model, "layer_plans", ())
              if getattr(plan, "attention", "") == "mla"]
    return bool(layers) and all(
        int(getattr(plan, "qk_rope_head_dim", 0) or 0) == 0 for plan in layers)


def cache_format(model, region):
    """Metadata declarations are explicit physical contracts, not calibration."""
    raw = dict(getattr(model, "metadata", ())).get("cache_representations", {}).get(region)
    if raw is None:
        quant = getattr(model, "quant", None)
        resolved = (getattr(quant, "kv_dtype", "")
                    or getattr(model, "kv_dtype", "bf16"))
        # A NoPE MLA layer keeps no rotary tail, so an FP8 KV checkpoint has
        # exactly the declared 128-element block row: 512 B payload + four
        # FP16 scales.  This is a format resolution, not a precision change.
        if (region == "mla_kv" and str(resolved) in {"fp8", "fp8_e4m3"}
                and _nope_mla_layers(model)):
            return block_quantized_mla_format()
        return CacheFormat(resolved)
    if set(raw) - {"dtype", "group_size", "scale_dtype", "suffix_elements",
                   "suffix_dtype", "packed_elements"}:
        raise ModelingError("cache_representation_fields_unknown:" + region)
    dtype = raw["dtype"]
    payload_bytes(dtype)
    group = raw.get("group_size", 0)
    scale = raw.get("scale_dtype", "none")
    if not isinstance(group, int) or group < 0 or (bool(group) != (scale != "none")):
        raise ModelingError("cache_representation_scale_group_mismatch:" + region)
    if group:
        dtype_bytes(scale)
    suffix = raw.get("suffix_elements", 0)
    suffix_dtype = raw.get("suffix_dtype", "bf16")
    if not isinstance(suffix, int) or suffix < 0:
        raise ModelingError("cache_suffix_elements_invalid:" + region)
    dtype_bytes(suffix_dtype)
    packed = raw.get("packed_elements", 1)
    if not isinstance(packed, int) or packed < 1:
        raise ModelingError("cache_representation_packed_elements_invalid:" + region)
    return CacheFormat(dtype, group, scale, suffix, suffix_dtype, packed)


def with_packed_mla_cache(model, rope):
    """Install the selected source's mixed row ABI in capacity accounting."""
    from dataclasses import replace
    metadata = dict(model.metadata)
    regions = dict(metadata.get("cache_representations", {}))
    declaration = {"dtype": "int8", "group_size": 128, "scale_dtype": "fp32",
                   "suffix_elements": rope, "suffix_dtype": "bf16"}
    if "mla_kv" in regions and regions["mla_kv"] != declaration:
        raise ModelingError("packed_mla_cache_conflicts_with_declared_representation")
    regions["mla_kv"] = declaration
    return replace(model, metadata=tuple({**metadata, "cache_representations": regions}.items()))
