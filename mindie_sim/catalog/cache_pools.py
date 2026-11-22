"""Separate integer pool/page pitches for packed MLA, recurrent-state and draft pools.

[Role]
  Resolve the declared GQA/GBSA cache layout selector, the target packed MLA
  page pitch, the recurrent/convolution state page pitch and the dense-draft
  page pitch, and expose the independent coexistence report consumed by the
  per-request cache breakdown.

[Invariants]
  * All page and row pitches use exact integer ceiling arithmetic (zero float
    division truncation or banker's rounding).
  * Target MLA, recurrent-state, and draft pools keep strictly independent
    allocation pitches; their three pitches are never summed as a single
    pool's allocation pitch.
  * ``draft_sharded`` is true only when a draft exists, declares
    ``dcp_sharded=True``, runs under a DCP-active plan with ``dcp > 1``, and
    its attention geometry actually shards KV across DCP ranks.

[Timeouts & Locks]
  Pure integer arithmetic bounded by layer count; no I/O, locks or device
  context.

[Side-Effect Policy]
  Returns immutable integers and plain dictionaries; never mutates model,
  runtime or graph state.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes

#: Declared packed C8 implementation row: one FP32 scale per 128 payload bytes
#: (512 INT8 latent bytes + four FP32 scales + zero/64 BF16 rotary lanes).
C8_SCALE_GROUP = 128
FP32_BYTES = 4
BF16_BYTES = 2
INT8_PACKED_KV_DTYPES = frozenset({"int8"})
MX_GROUP_SIZE = 32
MXFP8_DTYPES = frozenset({"mxfp8", "mx_fp8", "mxfp8_e4m3"})
MXFP4_DTYPES = frozenset({"mxfp4", "mx_fp4", "mxfp4_e2m1"})
DCP_ACTIVE_PLANS = frozenset({"dcp", "pcp_dcp"})
FRACTION_DENOM_LIMIT = 1024

#: Exact rational byte widths ``(numerator, denominator)`` for non-MX dtypes.
_EXACT_DTYPE_RATIO: dict[str, tuple[int, int]] = {
    "fp32": (4, 1),
    "float32": (4, 1),
    "int32": (4, 1),
    "int64": (8, 1),
    "bf16": (2, 1),
    "bfloat16": (2, 1),
    "fp16": (2, 1),
    "float16": (2, 1),
    "fp8": (1, 1),
    "fp8_e4m3": (1, 1),
    "fp8_e5m2": (1, 1),
    "fp8_e8m0": (1, 1),
    "e8m0": (1, 1),
    "int8": (1, 1),
    "uint8": (1, 1),
    "w8a8": (1, 1),
    "bool": (1, 1),
    "int4": (1, 2),
    "w4a8": (1, 2),
    "w4a16": (1, 2),
    "fp4": (1, 2),
    "fp4_e2m1": (1, 2),
    "fp64": (8, 1),
    "float64": (8, 1),
}

LINEAR_ATTENTION_KINDS = frozenset({
    "kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet",
})


def _ceil_div(numerator: int, denominator: int) -> int:
    """Exact integer ceiling division ``ceil(numerator / denominator)``."""
    num = int(numerator)
    den = int(denominator)
    if den <= 0:
        raise ModelingError(f"cache_pool_divisor_invalid:{den}")
    if num <= 0:
        return 0
    return (num + den - 1) // den


def exact_element_bytes(elements: int, dtype: str) -> int:
    """Exact integer allocation bytes for ``elements`` values of ``dtype``.

    Avoids float division/truncation and banker's ``round()`` for integer,
    4-bit sub-byte, and MX group-32 scaled dtypes alike.
    """
    count = int(elements)
    if count <= 0:
        return 0
    key = str(dtype or "bf16").strip().lower()
    if key in MXFP8_DTYPES:
        return count + _ceil_div(count, MX_GROUP_SIZE)
    if key in MXFP4_DTYPES:
        return _ceil_div(count, 2) + _ceil_div(count, MX_GROUP_SIZE)
    ratio = _EXACT_DTYPE_RATIO.get(key)
    if ratio is not None:
        return _ceil_div(count * ratio[0], ratio[1])
    frac = Fraction(float(dtype_bytes(key))).limit_denominator(FRACTION_DENOM_LIMIT)
    return _ceil_div(count * frac.numerator, frac.denominator)


def gbsa_cache_layout(runtime: Any) -> str:
    """Declared GQA/GBSA cache layout (``int8_fa3`` selects the C8 rows)."""
    return str(dict(getattr(runtime, "metadata", ()) or ()).get(
        "gbsa_cache_layout", "") or "")


def _packed_mla_row_parts(model: Any, plan: Any) -> tuple[int, int] | None:
    """Return ``(payload_bytes, scale_bytes)`` for implicit packed INT8 MLA rows."""
    width = int(plan.cache_row_elements)
    regions = dict(getattr(model, "metadata", ()) or ()).get(
        "cache_representations", {})
    quant = getattr(model, "quant", None)
    kv_dtype = str(getattr(quant, "kv_dtype", "") or "")
    latent = int(getattr(plan, "kv_lora_rank", 0) or 0)
    rope = int(getattr(plan, "qk_rope_head_dim", 0) or 0)
    if ("mla_kv" not in dict(regions) and kv_dtype in INT8_PACKED_KV_DTYPES
            and latent > 0 and latent + rope == width):
        scales = _ceil_div(latent, C8_SCALE_GROUP) * FP32_BYTES
        return int(latent + rope * BF16_BYTES), int(scales)
    return None


def _packed_mla_row_bytes(model: Any, plan: Any) -> int:
    """Physical MLA row width from the declared cache representation.

    An explicit ``cache_representations['mla_kv']`` mapping is authoritative.
    A declared INT8 KV row is the packed C8 record ``latent + FP32 group
    scales + BF16 rotary`` (512 + 16 + 0 = 528 B for NoPE C8); every other
    row keeps its declared ``CacheFormat`` price as an exact integer.
    """
    from mindie_sim.catalog.cache import packed_region_row_bytes

    parts = _packed_mla_row_parts(model, plan)
    if parts is not None:
        return parts[0] + parts[1]
    return int(packed_region_row_bytes(model, "mla_kv", int(plan.cache_row_elements)))


def target_mla_page_bytes(model: Any, block_size: int, *,
                          packed_head_bytes: int | None = None) -> int:
    """One target MLA page; a dense draft never changes its pitch."""
    block = int(block_size)
    if block <= 0:
        raise ModelingError(f"cache_pool_block_size_invalid:{block}")
    if packed_head_bytes is not None:
        return block * int(packed_head_bytes)
    plans = [plan for plan in getattr(model, "layer_plans", ())
             if plan.attention == "mla" and plan.cache_row_elements]
    if not plans:
        return 0
    row = max(_packed_mla_row_bytes(model, plan) for plan in plans)
    return block * int(row)


def _state_page_bytes(model: Any, parallel: Any, block_size: int,
                      runtime: Any = None) -> int:
    """Per-rank integer recurrent+convolution state pitch for one request slot.

    Each sharded state tensor uses exact integer ceiling division per rank
    (``ceil(elements / shards)`` elements converted via ``exact_element_bytes``)
    so indivisible tails are never silently truncated.
    """
    del block_size
    plans = [plan for plan in getattr(model, "layer_plans", ())
             if plan.attention in LINEAR_ATTENTION_KINDS]
    if not plans:
        return 0
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    axis = str(dict(getattr(runtime, "metadata", ()) or ()).get(
        "kda_parallel_axis", "")).strip().lower()
    # KDA under PCP shards heads (and state) across tp * pcp by default.
    kda_pcp = 1 if axis == "tp" else pcp
    total_bytes = 0
    for plan in plans:
        shards = tp * (kda_pcp if plan.attention == "kda" else 1)
        local_rec = _ceil_div(int(plan.recurrent_state_elements), shards)
        local_conv = _ceil_div(int(plan.conv_state_elements), shards)
        total_bytes += exact_element_bytes(local_rec, str(plan.recurrent_state_dtype))
        total_bytes += exact_element_bytes(local_conv, str(plan.conv_state_dtype))
    return total_bytes


def _draft_row_bytes_int(model: Any, draft: Any, *, tp: int) -> int:
    """Exact integer KV row width for one draft token on one TP rank."""
    from mindie_sim.catalog.cache import local_kv_heads
    from mindie_sim.catalog.draft import draft_quant

    if str(getattr(draft, "attention_kind", "")) == "mla":
        width = int(
            getattr(draft, "kv_cache_elements", 0)
            or (int(getattr(draft, "kv_lora_rank", 0))
                + int(getattr(draft, "qk_rope_head_dim", 0)))
        )
    else:
        width = 2 * int(local_kv_heads(int(draft.num_kv_heads), tp)) * int(draft.head_dim)
    kv_dtype = str(draft_quant(model, draft).kv_dtype)
    return exact_element_bytes(width, kv_dtype)


def _resolve_draft_spec(model: Any, draft_kind: str | None) -> Any | None:
    if not draft_kind or getattr(model, "auxiliary", None) is None:
        return None
    from mindie_sim.catalog.draft import optional_draft_spec
    return optional_draft_spec(model, draft_kind)


def _draft_page_bytes(model: Any, parallel: Any, block_size: int,
                      draft_kind: str | None) -> int:
    block = int(block_size)
    if block <= 0:
        raise ModelingError(f"cache_pool_block_size_invalid:{block}")
    draft = _resolve_draft_spec(model, draft_kind)
    if draft is None:
        return 0
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    return block * _draft_row_bytes_int(model, draft, tp=tp)


def _draft_is_dcp_sharded(
    model: Any, parallel: Any, draft_kind: str | None, *, dcp: int, draft_page: int,
) -> bool:
    """True iff the draft actually shards its KV across DCP ranks.

    Checks the draft's explicit ``dcp_sharded`` contract, an active DCP plan,
    and whether the draft's attention KV-head domain can legally shard across
    ``dcp`` ranks; never sets ``True`` from ``dcp > 1`` alone.
    """
    dcp_deg = int(dcp)
    if draft_page <= 0 or dcp_deg <= 1:
        return False
    draft = _resolve_draft_spec(model, draft_kind)
    if draft is None or not bool(getattr(draft, "dcp_sharded", False)):
        return False
    cp_plan = str(getattr(parallel, "cp_plan", "dcp") or "dcp")
    if cp_plan not in DCP_ACTIVE_PLANS:
        return False
    kind = str(getattr(draft, "attention_kind", "") or "")
    if kind == "mla":
        return True
    if kind == "gqa":
        tp = max(1, int(getattr(parallel, "tp", 1) or 1))
        pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
        if pcp > 1 and dcp_deg != pcp:
            return False
        kv_heads = max(1, int(getattr(draft, "num_kv_heads", 1) or 1))
        if pcp <= 1:
            domain = tp // kv_heads
        elif tp >= kv_heads:
            domain = (tp // kv_heads) * pcp
        else:
            domain = pcp if kv_heads % tp == 0 else 0
        return bool(domain >= dcp_deg and domain % dcp_deg == 0)
    return False


def separate_pool_page_pitches(
    model: Any, parallel: Any, *, block_size: int,
    draft_kind: str | None = None, dcp: int = 1,
    target_packed_head_bytes: int | None = None, runtime: Any = None,
) -> dict[str, Any]:
    """Keep target MLA, recurrent-state and dense-draft pools on independent pitches.

    The three pools have distinct allocation granularities (token pages vs
    per-request state slots vs draft token pages) and are never summed into a
    single allocation pitch.
    """
    target = target_mla_page_bytes(
        model, block_size, packed_head_bytes=target_packed_head_bytes)
    state_page = _state_page_bytes(model, parallel, block_size, runtime)
    draft_page = _draft_page_bytes(model, parallel, block_size, draft_kind)
    draft_sharded = _draft_is_dcp_sharded(
        model, parallel, draft_kind, dcp=dcp, draft_page=draft_page)
    return {
        "target_mla_page_bytes": target,
        "state_page_bytes": state_page,
        "draft_page_bytes": draft_page,
        "draft_sharded": draft_sharded,
        "target_pool_page_bytes": target,
        "draft_pool_page_bytes": draft_page,
        "state_pool_page_bytes": state_page,
        "independent_pool_pitches": {
            "target_mla": target,
            "recurrent_state": state_page,
            "draft": draft_page,
        },
        "target_page_unchanged_by_draft": True,
    }


def request_pool_page_pitches(
    model: Any, parallel: Any, *, block_size: int,
    draft_kind: str | None = None, dcp: int = 1,
    runtime: Any = None,
) -> dict[str, Any] | None:
    """Coexistence report for a target MLA pool, linear state and draft.

    The report exists only when a packed target MLA page and a linear-attention
    state page are both allocated for the same request; an optional dense
    draft adds its own independent pitch.
    """
    plans = tuple(getattr(model, "layer_plans", ()))
    has_target = any(plan.attention == "mla" and plan.cache_row_elements
                     for plan in plans)
    has_state = any(plan.attention in LINEAR_ATTENTION_KINDS for plan in plans)
    if not (has_target and has_state):
        return None
    return separate_pool_page_pitches(
        model, parallel, block_size=block_size, draft_kind=draft_kind, dcp=dcp,
        runtime=runtime)


__all__ = [
    "LINEAR_ATTENTION_KINDS", "exact_element_bytes", "gbsa_cache_layout",
    "target_mla_page_bytes", "separate_pool_page_pitches",
    "request_pool_page_pitches",
]
