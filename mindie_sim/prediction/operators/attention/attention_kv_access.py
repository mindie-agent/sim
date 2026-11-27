"""Typed KV cache representation and access for attention families.

[Role] Resolve one shared row declaration into the source K/V value, each
  side's stored payload/scale/suffix, the conversion result and the compute
  operand.  Producers and independent attention recipes consume the same
  result.
[Invariants] A ``gqa_kv`` row is parsed over its whole declared width; suffix
  and group scale are not re-created per side.  Independent K/V declarations
  keep their own dtype/format.  Payload bytes, on-chip operands and issued
  read/scale/page traffic are distinct.  A missing view preserves a known
  dtype conversion, and conversion never mutates the stored value.
[Timeouts & Locks] Constant arithmetic over one declared view; no I/O or locks.
[Side-Effect Policy] Pure records; no Engine, device or observation access.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, sqrt
from typing import Mapping

from mindie_sim.catalog.cache_format import CacheFormat
from mindie_sim.schema.common import ModelingError, dtype_bytes


LSE_ELEMENT_BYTES = 4


_PAGE_TABLE_DTYPE = "int32"
_PAGE_TABLE_BYTES = 4
_DTYPE_CANONICAL = {
    "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
    "fp8": "fp8_e4m3", "fp4": "fp4_e2m1",
    "mxfp8": "fp8_e4m3", "qk_mxfp8_dynamic_v_mxfp8_per_channel": "fp8_e4m3",
}


def _canonical(dtype: str) -> str:
    key = dtype.strip().lower()
    return _DTYPE_CANONICAL.get(key, key)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"attention_kv_{name}_must_be_positive_integer")
    return value


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"attention_kv_{name}_must_be_nonnegative_integer")
    return value


def _dtype(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(f"attention_kv_{name}_must_be_dtype")
    normalized = _canonical(value)
    dtype_bytes(normalized)
    return normalized


def _format_from_declaration(raw: Mapping, dtype: str) -> CacheFormat:
    declaration = dict(raw)
    declaration["dtype"] = _canonical(str(declaration.get("dtype", dtype)))
    unknown = set(declaration) - {
        "dtype", "group_size", "scale_dtype", "suffix_elements",
        "suffix_dtype", "packed_elements",
    }
    if unknown:
        raise ModelingError(
            "attention_kv_representation_fields_unknown:" + ",".join(sorted(unknown)))
    fmt = CacheFormat(**declaration)
    if _canonical(fmt.dtype) != dtype:
        raise ModelingError("attention_kv_representation_dtype_disagrees_with_view")
    return fmt


def _legacy_format(view: Mapping, dtype: str) -> CacheFormat:
    group = view.get("kv_group_size", 0)
    declaration = {
        "dtype": dtype,
        "group_size": group,
        # A dtype without a group is a page/row-scalar declaration, kept in
        # ``page_scale_bytes``; it is not a CacheFormat group scale.
        "scale_dtype": view.get("kv_scale_dtype", "none") if group else "none",
        "suffix_elements": view.get("kv_suffix_elements", 0),
        "suffix_dtype": view.get("kv_suffix_dtype", "bf16"),
        "packed_elements": view.get("kv_packed_elements", 1),
    }
    return _format_from_declaration(declaration, dtype)


def _regions(view: Mapping):
    regions = view.get("cache_representations")
    if regions is not None and not isinstance(regions, Mapping):
        raise ModelingError("attention_kv_cache_representations_must_be_mapping")
    if not regions:
        return None, None, None
    shared = regions.get("gqa_kv")
    key = regions.get("gqa_kv_key", shared)
    value = regions.get("gqa_kv_value", shared)
    if key is None and value is None:
        raise ModelingError("attention_kv_cache_representations_require_gqa_kv")
    if key is None:
        key = value
    if value is None:
        value = key
    return shared, key, value


def _side(fmt: CacheFormat, width: int, query_dtype: str) -> dict:
    payload_fmt = replace(fmt, group_size=0, scale_dtype="none")
    payload = int(payload_fmt.row_bytes(width))
    quantized = width - fmt.suffix_elements
    if quantized < 0:
        raise ModelingError("attention_kv_suffix_exceeds_row_width")
    scale = (int(ceil(quantized / fmt.group_size)
                 * dtype_bytes(fmt.scale_dtype)) if fmt.group_size else 0)
    needs = (fmt.dtype != query_dtype
             or (fmt.suffix_elements and fmt.suffix_dtype != query_dtype))
    operand = (width * dtype_bytes(query_dtype) if needs else payload)
    return {"payload": payload, "scale": scale, "needs": needs,
            "operand": int(operand), "quantized": quantized}


def _shared_scale_split(
    fmt: CacheFormat,
    *,
    total_quantized: int,
    head_quantized: int,
    key_quantized: int,
    kv_heads: int,
) -> tuple[int, int]:
    """Derive combined ``gqa_kv`` row scale bytes once in whole addressable units."""
    if not fmt.group_size or total_quantized <= 0:
        return 0, 0
    scale_unit = int(dtype_bytes(fmt.scale_dtype))
    total_scale = int(ceil(total_quantized / fmt.group_size) * scale_unit)
    if total_scale % (kv_heads * scale_unit) == 0:
        per_head_scale = total_scale // kv_heads
    else:
        per_head_scale = (
            int(ceil(head_quantized / fmt.group_size) * scale_unit)
            if head_quantized > 0 else 0
        )
    raw_key_scale = (
        int(ceil(key_quantized / fmt.group_size) * scale_unit)
        if key_quantized > 0 else 0
    )
    key_scale = min(per_head_scale, raw_key_scale)
    value_scale = per_head_scale - key_scale
    return key_scale, value_scale


def _shared_suffix_side(shared: Mapping, *, row_width: int, kv_heads: int,
                        head_dim: int, v_head_dim: int | None = None,
                        query_dtype: str = "bf16"):
    """Split one ``gqa_kv`` row with a suffix once, intersecting suffix per plane."""
    fmt = CacheFormat(**shared)
    if fmt.suffix_elements <= 0:
        return None
    if kv_heads != 1:
        # The combined row tail cannot be assigned to an individual head
        # without a declared head layout.  Keep this a concrete gap.
        raise ModelingError(
            "attention_kv_shared_row_suffix_requires_single_kv_head")
    v_dim = head_dim if v_head_dim is None else v_head_dim
    total_quantized = row_width - fmt.suffix_elements
    if total_quantized < 0:
        raise ModelingError("attention_kv_suffix_exceeds_row_width")
    key_width = kv_heads * head_dim
    value_width = kv_heads * v_dim
    key_quantized = min(key_width, total_quantized)
    key_suffix = key_width - key_quantized
    value_quantized = max(0, total_quantized - key_width)
    value_suffix = value_width - value_quantized
    payload_fmt = replace(fmt, group_size=0, scale_dtype="none",
                          suffix_elements=0)
    suffix_bytes = dtype_bytes(fmt.suffix_dtype)
    key_payload = int(payload_fmt.row_bytes(key_quantized) + key_suffix * suffix_bytes)
    value_payload = int(payload_fmt.row_bytes(value_quantized) + value_suffix * suffix_bytes)
    key_scale, value_scale = _shared_scale_split(
        fmt, total_quantized=total_quantized,
        head_quantized=total_quantized,
        key_quantized=key_quantized, kv_heads=kv_heads,
    )
    key_needs = (
        (key_quantized > 0 and fmt.dtype != query_dtype)
        or (key_suffix > 0 and fmt.suffix_dtype != query_dtype)
    )
    value_needs = (
        (value_quantized > 0 and fmt.dtype != query_dtype)
        or (value_suffix > 0 and fmt.suffix_dtype != query_dtype)
    )
    key_operand = (key_width * dtype_bytes(query_dtype)
                   if key_needs else key_payload)
    value_operand = (value_width * dtype_bytes(query_dtype)
                     if value_needs else value_payload)
    return {
        "key": {"payload": key_payload, "scale": key_scale,
                "needs": key_needs, "operand": int(key_operand),
                "quantized": key_quantized, "suffix": key_suffix},
        "value": {"payload": value_payload, "scale": value_scale,
                  "needs": value_needs, "operand": int(value_operand),
                  "quantized": value_quantized, "suffix": value_suffix},
        "format": fmt,
    }


@dataclass(frozen=True)
class KVCacheAccess:
    """One owner's declared K/V storage, conversion and page access."""

    declared: bool = True
    paged: bool = False
    page_size: int = 0
    page_meta_bytes: int = 0
    key_format: CacheFormat | None = None
    value_format: CacheFormat | None = None
    key_payload_row: int = 0
    value_payload_row: int = 0
    key_operand_row: int = 0
    value_operand_row: int = 0
    key_scale_row: int = 0
    value_scale_row: int = 0
    page_scale_bytes: int = 0
    scale_scope: str = "none"
    dequant_ops_per_element: int = 0
    key_needs_conversion: bool = False
    value_needs_conversion: bool = False
    query_dtype: str = ""
    head_dim: int = 0
    v_head_dim: int = 0
    key_suffix: int = 0
    value_suffix: int = 0
    key_quantized_width: int = 0
    value_quantized_width: int = 0
    row_width: int = 0
    shared_row: bool = False
    source_residency: str = "unresolved"

    @property
    def needs_conversion(self) -> bool:
        return self.key_needs_conversion or self.value_needs_conversion

    def pages(self, start: int, end: int) -> int:
        if not self.paged or end <= start:
            return 0
        return (end - 1) // self.page_size - start // self.page_size + 1

    def payload_bytes(self, rows: int) -> int:
        return rows * (self.key_payload_row + self.value_payload_row)

    def operand_bytes(self, rows: int) -> int:
        return rows * (self.key_operand_row + self.value_operand_row)

    def row_scale_bytes(self, rows: int) -> int:
        return rows * (self.key_scale_row + self.value_scale_row)

    def key_conversion_elements(self, rows: int) -> int:
        if not self.key_needs_conversion:
            return 0
        return rows * (self.key_quantized_width
                       + (self.key_suffix
                          if self.key_format is not None
                          and self.key_format.suffix_dtype != self.query_dtype
                          else 0))

    def value_conversion_elements(self, rows: int) -> int:
        if not self.value_needs_conversion:
            return 0
        return rows * (self.value_quantized_width
                       + (self.value_suffix
                          if self.value_format is not None
                          and self.value_format.suffix_dtype != self.query_dtype
                          else 0))

    def conversion_elements(self, rows: int) -> int:
        return (self.key_conversion_elements(rows)
                + self.value_conversion_elements(rows))

    def key_conversion_ops(self, rows: int) -> float:
        return (self.key_conversion_elements(rows)
                * self.dequant_ops_per_element)

    def value_conversion_ops(self, rows: int) -> float:
        return (self.value_conversion_elements(rows)
                * self.dequant_ops_per_element)

    def conversion_ops(self, rows: int) -> float:
        return (self.key_conversion_ops(rows)
                + self.value_conversion_ops(rows))

    def report(self) -> dict:
        def fmt(value):
            return asdict(value) if value is not None else None
        return {
            "storage": ("undeclared" if not self.declared else
                        "paged" if self.paged else "contiguous"),
            "page_size": self.page_size,
            "page_metadata_bytes_per_page": self.page_meta_bytes,
            "page_scale_bytes_per_page": self.page_scale_bytes,
            "scale_scope": self.scale_scope,
            "row_width": self.row_width,
            "shared_row": self.shared_row,
            "source_residency": self.source_residency,
            "key_format": fmt(self.key_format),
            "value_format": fmt(self.value_format),
            "key_payload_row_bytes": self.key_payload_row,
            "value_payload_row_bytes": self.value_payload_row,
            "key_operand_row_bytes": self.key_operand_row,
            "value_operand_row_bytes": self.value_operand_row,
            "key_scale_row_bytes": self.key_scale_row,
            "value_scale_row_bytes": self.value_scale_row,
            "key_quantized_width": self.key_quantized_width,
            "value_quantized_width": self.value_quantized_width,
            "key_needs_conversion": self.key_needs_conversion,
            "value_needs_conversion": self.value_needs_conversion,
            "dequant_ops_per_element": self.dequant_ops_per_element,
            "payload_scope": "valid_causal_rows_no_padded_page_tail_read",
            "page_metadata_scope": "per_owner_task_no_l2_sharing_model",
            "conversion_scope": "on_chip_operand_materialization_no_hbm_writeback",
        }


def _dequant_ops(view: Mapping, needs_conversion: bool) -> int:
    declared = view.get("kv_dequant")
    if declared is not None and not isinstance(declared, bool):
        raise ModelingError("attention_kv_dequant_must_be_bool")
    if not needs_conversion:
        if declared is True:
            raise ModelingError(
                "attention_kv_dequant_requested_without_dtype_conversion")
        return 0
    if declared is False:
        raise ModelingError(
            "attention_kv_quantized_cache_conversion_cannot_be_disabled")
    return _positive_int(view.get("kv_dequant_ops_per_element", 2),
                         "dequant_ops_per_element")


def _page_scale(view: Mapping, key: CacheFormat, value: CacheFormat) -> int:
    direct = view.get("kv_scale_bytes_per_page")
    if direct is not None:
        if key.group_size or value.group_size:
            raise ModelingError(
                "attention_kv_scale_page_and_row_representations_ambiguous")
        return _nonnegative_int(direct, "scale_bytes_per_page")
    if key.group_size or value.group_size:
        return 0
    scale_dtype = view.get("kv_scale_dtype")
    if scale_dtype in {None, "", "none"}:
        return 0
    return int(2 * dtype_bytes(_dtype(scale_dtype, "scale_dtype")))


def kv_access(
    op, *, query_dtype: str, kv_dtype: str, head_dim: int,
    v_head_dim: int | None = None,
) -> KVCacheAccess:
    """Resolve one attention operator's declared K/V storage access."""
    raw = dict(getattr(op, "metadata", ())).get("attention_tensor_view")
    if raw is None:
        v_dim = head_dim if v_head_dim is None else _positive_int(v_head_dim, "v_head_dim")
        key_fmt = CacheFormat(kv_dtype)
        value_fmt = CacheFormat(kv_dtype)
        key = _side(key_fmt, head_dim, query_dtype)
        value = _side(value_fmt, v_dim, query_dtype)
        needs = key["needs"] or value["needs"]
        return KVCacheAccess(
            declared=False, key_format=key_fmt, value_format=value_fmt,
            key_payload_row=key["payload"], value_payload_row=value["payload"],
            key_operand_row=key["operand"], value_operand_row=value["operand"],
            key_scale_row=int(key["scale"]), value_scale_row=int(value["scale"]),
            dequant_ops_per_element=_dequant_ops({}, needs),
            key_needs_conversion=key["needs"],
            value_needs_conversion=value["needs"],
            query_dtype=query_dtype, head_dim=head_dim, v_head_dim=v_dim,
            key_suffix=key_fmt.suffix_elements,
            value_suffix=value_fmt.suffix_elements,
            key_quantized_width=key["quantized"],
            value_quantized_width=value["quantized"],
            row_width=head_dim + v_dim,
            source_residency="unresolved",
        )
    if not isinstance(raw, Mapping):
        raise ModelingError("attention_kv_tensor_view_must_be_mapping")
    v_dim = _positive_int(
        v_head_dim if v_head_dim is not None else raw.get("v_head_dim", head_dim),
        "v_head_dim",
    )
    storage = raw.get("kv_storage")
    if storage not in {"paged", "contiguous"}:
        raise ModelingError("attention_kv_storage_requires_paged_or_contiguous")
    kv_heads = _positive_int(raw.get("local_kv_heads", 1), "local_kv_heads")
    row_width = _positive_int(
        raw.get("cache_row_width", kv_heads * (head_dim + v_dim)), "cache_row_width")
    key_dtype = _dtype(raw.get("key_dtype", kv_dtype), "key_dtype")
    value_dtype = _dtype(raw.get("value_dtype", kv_dtype), "value_dtype")
    shared_raw, key_raw, value_raw = _regions(raw)
    if shared_raw is None and key_raw is None and value_raw is None:
        key_fmt = _legacy_format(raw, key_dtype)
        value_fmt = _legacy_format(raw, value_dtype)
        key = _side(key_fmt, head_dim, query_dtype)
        value = _side(value_fmt, v_dim, query_dtype)
        shared = False
    elif shared_raw is not None and key_raw is shared_raw and value_raw is shared_raw:
        shared_fmt = _format_from_declaration(shared_raw, key_dtype)
        if value_dtype != key_dtype:
            raise ModelingError(
                "attention_kv_shared_row_requires_one_storage_dtype")
        split = _shared_suffix_side(
            shared_raw, row_width=row_width, kv_heads=kv_heads,
            head_dim=head_dim, v_head_dim=v_dim, query_dtype=query_dtype)
        if split is not None:
            key, value = split["key"], split["value"]
            key_fmt = value_fmt = split["format"]
        else:
            key_fmt = value_fmt = shared_fmt
            key = _side(key_fmt, head_dim, query_dtype)
            value = _side(value_fmt, v_dim, query_dtype)
            if shared_fmt.group_size:
                k_scale, v_scale = _shared_scale_split(
                    shared_fmt, total_quantized=row_width,
                    head_quantized=head_dim + v_dim,
                    key_quantized=head_dim, kv_heads=kv_heads,
                )
                key = {**key, "scale": k_scale}
                value = {**value, "scale": v_scale}
        shared = True
    else:
        if key_raw is None or value_raw is None:
            raise ModelingError("attention_kv_independent_formats_require_both_sides")
        key_fmt = _format_from_declaration(key_raw, key_dtype)
        value_fmt = _format_from_declaration(value_raw, value_dtype)
        key = _side(key_fmt, head_dim, query_dtype)
        value = _side(value_fmt, v_dim, query_dtype)
        shared = False
    if not raw.get("cache_representations") and (
            key_dtype != kv_dtype or value_dtype != kv_dtype):
        raise ModelingError("attention_kv_view_dtype_disagrees_with_gqa_decode")
    needs = key["needs"] or value["needs"]
    dequant = _dequant_ops(raw, needs)
    page_scale = _page_scale(raw, key_fmt, value_fmt) if storage == "paged" else 0
    if page_scale and (key["scale"] or value["scale"]):
        raise ModelingError(
            "attention_kv_scale_page_and_row_representations_ambiguous")
    page_size = 0
    page_meta = 0
    if storage == "paged":
        page_size = _positive_int(raw.get("kv_block_size"), "block_size")
        meta = raw.get("kv_page_table_bytes_per_page")
        if meta is None:
            meta = int(dtype_bytes(raw.get("kv_page_table_dtype",
                                           _PAGE_TABLE_DTYPE)))
        page_meta = _nonnegative_int(meta, "page_table_bytes_per_page")
    if page_scale:
        scale_scope = "page"
    elif key["scale"] or value["scale"]:
        scale_scope = "row"
    else:
        scale_scope = "none"
    return KVCacheAccess(
        declared=True, paged=storage == "paged",
        page_size=page_size, page_meta_bytes=page_meta,
        key_format=key_fmt, value_format=value_fmt,
        key_payload_row=int(key["payload"]),
        value_payload_row=int(value["payload"]),
        key_operand_row=int(key["operand"]),
        value_operand_row=int(value["operand"]),
        key_scale_row=int(key["scale"]), value_scale_row=int(value["scale"]),
        page_scale_bytes=page_scale, scale_scope=scale_scope,
        dequant_ops_per_element=dequant,
        key_needs_conversion=key["needs"],
        value_needs_conversion=value["needs"],
        query_dtype=query_dtype, head_dim=head_dim, v_head_dim=v_dim,
        key_suffix=key.get("suffix", key_fmt.suffix_elements),
        value_suffix=value.get("suffix", value_fmt.suffix_elements),
        key_quantized_width=key["quantized"],
        value_quantized_width=value["quantized"],
        row_width=row_width, shared_row=shared,
        source_residency=str(raw.get("cache_source_residency", "unresolved")),
    )


__all__ = ["KVCacheAccess", "kv_access"]


PAGED_FIA_PLAN_SCHEMA = "mindie-sim/paged-fia-native-plan/1"
PAGED_FIA_ENTRY = "torch_npu.npu_fused_infer_attention_score"


def noncausal_window_pairs_and_span(
    q: int, context: int, kv: int, window: int,
) -> tuple[int, int, int]:
    """O(1) symmetric band ``(W-1, W-1)`` pairs and union ``[lo, hi)`` for noncausal SWA.

    Mirrors ``vllm_ascend/attention/attention_v1.py:679-686`` where each query at
    position ``p`` attends to ``[max(0, p - (W - 1)), min(kv, p + W))``.
    """
    if q <= 0 or kv <= 0:
        return 0, 0, 0
    if window <= 0:
        return q * kv, 0, kv
    radius = window - 1
    p0 = max(0, int(context))
    p1 = p0 + q - 1
    u_lo = max(0, p0 - radius)
    u_hi = min(kv, p1 + radius + 1)
    if u_hi <= u_lo:
        return 0, 0, 0
    # sum_{p=p0}^{p1} min(kv, p + radius + 1) - sum_{p=p0}^{p1} max(0, p - radius)
    x0 = p0 + radius + 1
    x1 = p1 + radius + 1
    if x0 <= kv:
        n_lin = min(x1, kv) - x0 + 1
        sum_hi = n_lin * (2 * x0 + n_lin - 1) // 2 + (q - n_lin) * kv
    else:
        sum_hi = q * kv
    y0 = p0 - radius
    y1 = p1 - radius
    if y1 <= 0:
        sum_lo = 0
    else:
        s_start = max(0, y0)
        count = y1 - s_start + 1
        sum_lo = count * (s_start + y1) // 2
    return max(0, sum_hi - sum_lo), u_lo, u_hi


def _paged_fia_request_span(
    q: int, kv: int, *, mask: str, window: int, page_size: int,
) -> tuple[int, int, int]:
    """Return ``(visible_pairs, active_kv_rows, active_pages)`` for one request."""
    if q <= 0 or kv <= 0:
        return 0, 0, 0
    past = max(0, kv - q)
    if mask != "causal":
        pairs, lo, hi = noncausal_window_pairs_and_span(q, past, kv, window)
        active = max(0, hi - lo)
        pages = ((hi - 1) // page_size - lo // page_size + 1) if active > 0 else 0
        return pairs, active, pages
    if window <= 0:
        if kv >= q:
            pairs = q * past + q * (q + 1) // 2
        else:
            pairs = sum(min(kv, past + i + 1) for i in range(q))
        lo, hi = 0, kv
    else:
        lo = max(0, past + 1 - window)
        hi = kv
        if kv >= q:
            first_vis = past + 1
            if first_vis >= window:
                pairs = q * window
            elif kv <= window:
                pairs = q * past + q * (q + 1) // 2
            else:
                ramp = window - past - 1
                pairs = ramp * past + ramp * (ramp + 1) // 2 + (q - ramp) * window
        else:
            pairs = sum(
                max(0, min(kv, past + i + 1) - max(0, past + i + 1 - window))
                for i in range(q)
            )
    active = max(0, hi - lo)
    pages = ((hi - 1) // page_size - lo // page_size + 1) if active > 0 else 0
    return pairs, active, pages


def paged_fia_plan(*, requests, query_lengths, kv_lengths, page_size,
                   query_heads, kv_heads, head_dim, query_dtype, output_dtype,
                   key_format, value_format, phase="decode", mask="causal",
                   return_lse=False, output_prepare=None,
                   v_head_dim=None, window=0, shared_format=False):
    """Ordinary paged FIA typed ports and executable-plan geometry.

    ``kv_lengths`` are the actual per-request backing KV extents (past context
    plus current query tokens), not a page-pool capacity.  When ``window > 0``,
    active pairs, KV read bytes, and page reads reflect the sliding-window span
    while backing cache/block-table ports retain full ``kv_lengths``.  K/V are
    static reads: no state/KV write is published.  An LSE port exists only when
    the caller requests it; the ordinary no-LSE path materializes zero bytes.
    """
    requests = tuple((int(q), int(kv)) for q, kv in requests)
    query_lengths = tuple(int(value) for value in query_lengths)
    kv_lengths = tuple(int(value) for value in kv_lengths)
    if not requests or len(requests) != len(query_lengths) or len(requests) != len(kv_lengths):
        raise ModelingError("paged_fia_plan_requests_mismatch")
    v_dim = head_dim if v_head_dim is None else int(v_head_dim)
    win = int(window or 0)
    if page_size < 1 or query_heads < 1 or kv_heads < 1 or head_dim < 1 or v_dim < 1:
        raise ModelingError("paged_fia_plan_nonpositive_geometry")
    if win < 0:
        raise ModelingError("paged_fia_plan_negative_window")
    if query_heads % kv_heads:
        raise ModelingError("paged_fia_plan_query_heads_not_divisible")
    query_dtype = _canonical(str(query_dtype))
    output_dtype = _canonical(str(output_dtype))
    page_counts = tuple(-(-kv // page_size) for kv in kv_lengths)
    total_pages = sum(page_counts)
    max_pages = max(page_counts)
    q_total = sum(query_lengths)

    spans = tuple(
        _paged_fia_request_span(q, kv, mask=mask, window=win, page_size=page_size)
        for q, kv in zip(query_lengths, kv_lengths)
    )
    pairs = query_heads * sum(item[0] for item in spans)
    active_kv_lengths = tuple(item[1] for item in spans)
    active_page_counts = tuple(item[2] for item in spans)
    active_pages = sum(active_page_counts)
    qk_mac = pairs * head_dim
    pv_mac = pairs * v_dim
    q_bytes = q_total * query_heads * head_dim * dtype_bytes(query_dtype)
    out_bytes = q_total * query_heads * v_dim * dtype_bytes(output_dtype)
    kv_row_b = (
        int(key_format.row_bytes(kv_heads * (head_dim + v_dim)))
        if shared_format
        else (int(key_format.row_bytes(kv_heads * head_dim))
              + int(value_format.row_bytes(kv_heads * v_dim)))
    )
    kv_bytes = sum(akv * kv_row_b for akv in active_kv_lengths)
    block_table_bytes = active_pages * _PAGE_TABLE_BYTES
    q_shape = (q_total, query_heads, head_dim)
    out_shape = (q_total, query_heads, v_dim)
    key_cache_shape = (total_pages, page_size, kv_heads, head_dim)
    value_cache_shape = (total_pages, page_size, kv_heads, v_dim)
    block_shape = (len(requests), max_pages)
    ports = {
        "query": {"role": "query", "direction": "in", "shape": q_shape,
                  "dtype": query_dtype, "layout": "TND", "offset": 0,
                  "required": True},
        "key_cache": {"role": "key", "direction": "in",
                      "shape": key_cache_shape, "dtype": key_format.dtype,
                      "layout": "PA_BBND", "required": True},
        "value_cache": {"role": "value", "direction": "in",
                        "shape": value_cache_shape, "dtype": value_format.dtype,
                        "layout": "PA_BBND", "required": True},
        "block_table": {"role": "block_table", "direction": "in",
                        "shape": block_shape, "dtype": "int32",
                        "layout": "row_major", "required": True},
        "sequence_lengths": {"role": "kv_lengths", "direction": "in",
                             "shape": (len(requests),), "dtype": "int32",
                             "layout": "row_major", "required": True},
        "scale": {"role": "softmax_scale", "direction": "in", "shape": (),
                  "dtype": "fp32", "value": 1.0 / sqrt(head_dim),
                  "required": True},
        "mask": {"role": "mask", "direction": "in", "kind": mask,
                 **({"window": win} if win > 0 else {}),
                 "required": True},
        "output": {"role": "output", "direction": "out", "shape": out_shape,
                   "dtype": output_dtype, "layout": "TND", "offset": 0,
                   "required": True},
        "lse": ({"role": "lse", "direction": "out",
                 "shape": (q_total, query_heads), "dtype": "fp32",
                 "layout": "row_major", "required": True}
                if return_lse else None),
    }
    work = {
        "query_tokens": q_total,
        "kv_lengths": kv_lengths,
        "active_kv_lengths": active_kv_lengths,
        "head_token_pairs": pairs,
        "qk_mac": qk_mac,
        "pv_mac": pv_mac,
        "q_bytes": q_bytes,
        "out_bytes": out_bytes,
        "kv_valid_bytes": kv_bytes,
        "page_counts": active_page_counts,
        "pages": active_pages,
        "page_table_bytes": block_table_bytes,
        "key_page_reads": active_pages,
        "value_page_reads": active_pages,
        "lse_bytes": 0 if not return_lse else q_total * query_heads * LSE_ELEMENT_BYTES,
        "state_writes": 0,
    }
    plan = {
        "schema": PAGED_FIA_PLAN_SCHEMA,
        "kind": "paged_fia",
        "phase": phase,
        "geometry": {
            "batch": len(requests),
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "v_head_dim": v_dim,
            "window": win,
            "group": query_heads // kv_heads,
            "hq_equals_hkv": query_heads == kv_heads,
            "page_size": page_size,
            "max_pages": max_pages,
            "page_counts": page_counts,
        },
        "layouts": {"query": "TND", "key": "PA_BBND", "value": "PA_BBND",
                    "output": "TND"},
        "ports": ports,
        "work": work,
        "abi": {
            "namespace": "torch",
            "entry": PAGED_FIA_ENTRY,
            "aclnn_symbol": None,
            "source_state": "unresolved",
            "binary_identity": "unresolved",
            "basis": "ordinary_paged_fia_no_d256_ifa_no_mla_no_broadcast",
        },
        "execution": {
            "program": "StagedGQADecode",
            "builder": "StagedGQADecode.build",
            "recipe": "staged_gqa_decode",
            "phase": phase,
            "owner": "family_attention_ifa",
        },
        "state": {"read_only_kv": True, "writes": (),
                  "lse_materialized": bool(return_lse)},
        "output_prepare": output_prepare,
        "limits": {"d256_specialized_ifa": False, "mla": False, "sfa": False,
                   "multiquery_broadcast": False},
    }
    return plan


@dataclass(frozen=True)
class FixpipeHandoffPolicy:
    """Architecture-specific L0C egress policy for staged GQA scores and PV partials."""

    enabled: bool = False
    via_gm: bool = False
    mem_tier: str | None = None

    def handoff_resources(self, nbytes: int) -> dict[str, int]:
        if not self.enabled or nbytes <= 0:
            return {}
        res = {"fixpipe": nbytes}
        if self.via_gm and self.mem_tier:
            res[self.mem_tier] = nbytes
        return res

    def read_resources(self, nbytes: int, vread: str) -> dict[str, int]:
        if not (self.enabled and self.via_gm) or nbytes <= 0:
            return {}
        res = {vread: nbytes}
        if self.mem_tier:
            res[self.mem_tier] = nbytes
        return res


def gqa_fixpipe_policy(source, workspace_bytes: int = 0) -> FixpipeHandoffPolicy:
    """Resolve L0C->UB (Arch35) vs L0C->L2/HBM + AIV read (Arch22) from Geometry or RecipeContext."""
    if source is None:
        return FixpipeHandoffPolicy()
    names = getattr(source, "resource_names", None)
    if names is not None:
        has_fixpipe = "fixpipe" in names
        has_l2 = "l2" in names
    else:
        units = getattr(source, "resource_units", ())
        rates = getattr(source, "rates", ())
        has_fixpipe = "fixpipe" in units
        has_l2 = "l2" in rates or "l2" in units
    if not has_fixpipe:
        return FixpipeHandoffPolicy()
    datapath = getattr(source, "datapath", None)
    if bool(getattr(datapath, "has_l0c_to_ub", False)):
        return FixpipeHandoffPolicy(enabled=True, via_gm=False, mem_tier=None)
    mech = getattr(source, "physical_mechanisms", None)
    hierarchy = getattr(mech, "memory_hierarchy", None)
    buffers = getattr(source, "buffers", None)
    if buffers is None:
        buffers = getattr(source, "pool_capacities", {})
    l2_cap = int(getattr(hierarchy, "effective_l2_bytes", buffers.get("l2", 0)) or 0)
    if l2_cap > 0 and workspace_bytes <= l2_cap:
        return FixpipeHandoffPolicy(enabled=True, via_gm=True, mem_tier="l2" if has_l2 else None)
    return FixpipeHandoffPolicy(enabled=True, via_gm=True, mem_tier="hbm")


def remap_stream_pool_units(units: tuple[int, ...], pool: tuple[int, ...]) -> tuple[int, ...]:
    """Map logical unit indices into a non-empty stream core pool without duplicates."""
    return tuple(dict.fromkeys(pool[u % len(pool)] for u in units)) if pool and units else units


def stream_pool_for_role(owners, role: str, *, buffer_pool: bool = False) -> tuple[int, ...]:
    """Return the AIC or AIV core pool from ``owners`` for one role or SRAM buffer pool."""
    from mindie_sim.prediction.streams import AIC_BUFFER_POOLS, AIC_ROLES, AIV_BUFFER_POOLS, AIV_ROLES
    aic_set = AIC_BUFFER_POOLS if buffer_pool else AIC_ROLES
    aiv_set = AIV_BUFFER_POOLS if buffer_pool else AIV_ROLES
    if role in aic_set:
        return owners.aic if owners.isolation == "disjoint" else owners.pool_aic
    if role in aiv_set:
        return owners.aiv if owners.isolation == "disjoint" else owners.pool_aiv
    return ()
