"""GDN Slice and ZerosLike launches from vllm-ascend v0.30.0 gdn.py b21508b4.

The packed qkvz row is the producer. Its last dimension is
``2 * key_heads * d_k + 2 * value_heads * d_v``. The Z operand used by
``norm(core_out, z)`` is the trailing ``value_heads * d_v`` elements of that
row, not a recurrent state. When ``d_v`` divides the packed width the same
bytes are the rank-3 view ``[tokens, packed / d_v, d_v]`` sliced down to
``[tokens, value_heads, d_v]``. For the Qwen TP=2 plan that is
``8192 / 128 = 64`` source heads and ``3072 / 128 = 24`` kept heads, with
the copy starting at element offset ``5120``. The nickname ``z`` is only the
source variable; the extent, offset and rank come from this layout.

``_rearrange_decode_qkv`` (lines 199–219) and the BA ``contiguous`` calls
(lines 392–402) copy a last-dimension split only when the leading dimension
is greater than one. A single row is already contiguous, because a size-1
dimension does not break contiguity, so ``contiguous`` and ``reshape`` do
not allocate. ``torch.zeros`` always materializes.

Which rows a launch sees:

* beta, alpha and the Z subview use every token of the forward
* q, k and v are the decode rearrange only
* the qkv contiguous copy is each side whose own row count is greater than
  one (a mixed step with one decode token copies the prefill rows only)
* the zero fill uses every forward token, including a single row

Pure prefill does not call the decode rearrange, so q, k and v stay absent.
Chunked-prefill A3 does not emit this group.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class DecodeView:
    """One captured copy or fill. Element counts are per token."""

    name: str
    role: str
    elements: int
    read_elements: int
    write_elements: int
    source_kernel: str
    placement: str
    row_source: str
    transfer: str
    execution: str
    rank: int
    source_extent: int
    offset_elements: int
    head_dim: int = 0
    source_heads: int = 0
    kept_heads: int = 0


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"gdn_decode_view_extent_invalid:{name}")
    return value


def _slice(name: str, kept: int, *, placement: str, row_source: str,
           source_extent: int, offset: int, rank: int = 2,
           head_dim: int = 0, source_heads: int = 0,
           kept_heads: int = 0) -> DecodeView:
    return DecodeView(
        name, "view", kept, kept, kept, "Slice", placement, row_source,
        "mte", "copy", rank, source_extent, offset, head_dim, source_heads,
        kept_heads)


def decode_views(key_heads: int, value_heads: int, d_k: int, d_v: int) -> tuple[DecodeView, ...]:
    """TP-local copies and the core zero fill, in capture order within each band."""
    keys = _positive("key_heads", key_heads)
    values = _positive("value_heads", value_heads)
    dk = _positive("d_k", d_k)
    dv = _positive("d_v", d_v)
    query = keys * dk
    value = values * dv
    qkv = 2 * query + value
    packed = qkv + value
    if packed % dv != 0 or value % dv != 0:
        raise ModelingError(f"gdn_decode_z_view_not_aligned:{packed}:{dv}")
    source_heads = packed // dv
    kept_heads = value // dv
    return (
        _slice("beta_slice", values, placement="early", row_source="forward_multirow",
               source_extent=values * 2, offset=0),
        _slice("alpha_slice", values, placement="early", row_source="forward_multirow",
               source_extent=values * 2, offset=values),
        DecodeView("core_zeros", "fill", value, 0, value, "ZerosLike", "early",
                   "forward_fill", "fill", "zero_fill", 3, value, 0, dv,
                   kept_heads, kept_heads),
        _slice("qkv_slice", qkv, placement="early", row_source="conv_qkv",
               source_extent=packed, offset=0),
        _slice("q_slice", query, placement="late", row_source="decode_multirow",
               source_extent=qkv, offset=0),
        _slice("k_slice", query, placement="late", row_source="decode_multirow",
               source_extent=qkv, offset=query),
        _slice("v_slice", value, placement="late", row_source="decode_multirow",
               source_extent=qkv, offset=2 * query),
        _slice("z_slice", value, placement="gate", row_source="forward_multirow",
               source_extent=packed, offset=qkv, rank=3, head_dim=dv,
               source_heads=source_heads, kept_heads=kept_heads),
    )
