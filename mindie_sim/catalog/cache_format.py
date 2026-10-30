"""Declared packed cache rows shared by producers, readers and capacity.

[Role] Resolve separate original-window and compressed-cache representations.
[Invariants] A group scale is stored once per row group. Unspecified regions
  retain QuantSpec KV precision; no model name or timing selects a format.
[Timeouts & Locks] Constant arithmetic, no I/O or locks.
[Side-Effect Policy] Returns immutable format metadata only.
"""
from dataclasses import dataclass
from math import ceil

from mindie_sim.schema.common import ModelingError, dtype_bytes


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

    def row_bytes(self, width):
        quantized = width - self.suffix_elements
        if quantized < 0:
            raise ModelingError("cache_suffix_exceeds_row_width")
        return (ceil(quantized * payload_bytes(self.dtype))
                + self.suffix_elements * dtype_bytes(self.suffix_dtype) + (
            ceil(quantized / self.group_size) * dtype_bytes(self.scale_dtype)
            if self.group_size else 0))


def cache_format(model, region):
    """Metadata declarations are explicit physical contracts, not calibration."""
    raw = dict(getattr(model, "metadata", ())).get("cache_representations", {}).get(region)
    if raw is None:
        quant = getattr(model, "quant", None)
        return CacheFormat(getattr(quant, "kv_dtype", "") or getattr(model, "kv_dtype", "bf16"))
    if set(raw) - {"dtype", "group_size", "scale_dtype", "suffix_elements", "suffix_dtype"}:
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
    return CacheFormat(dtype, group, scale, suffix, suffix_dtype)


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
