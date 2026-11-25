"""Canonical integer pool pitches and G3 structural hybrid block-pool algebra.

[Role]
  Own the single validated ``PoolAllocatorContract`` and ``HybridPoolReceipt``
  algebra across all four mutually-exclusive structural regimes:
    1. ``dual_pitch_paired_kpool_mamba`` (NoPE/MLA + KPool indexer + KDA runs,
       with ``patch_mamba_config`` manager block normalization),
    2. ``shared_slab_private_aux`` (QSA main/raw/compressed + GDN + private PLE
       with dual alignment ``(A, U)`` and ``r0/r1/r2/r4/r5`` slab layout),
    3. ``aliased_window_compressed`` (4 physical backings aliasing SWA, C2
       compressor rings, long/index/folded KV, and optional draft SWA),
    4. ``uniform_independent`` (dense GQA/MLA/DSA/paired-SFA/pure SWA).
  Also expose exact integer dtype byte arithmetic, canonical packed MLA row
  delegation, state/draft pool page pitches, and independent coexistence reports.

[Invariants]
  * All page, stride, alignment, and slab calculations use exact Python integer
    arithmetic (zero float truncation or banker's rounding).
  * Zero AG014 model-name checks; regime selection inspects structural owner
    roles only and raises ``ModelingError("ambiguous_or_incomplete_pool_contract")``
    on conflicting or incomplete roles.
  * KPool and aliased window-compressed regimes hard-reject ``DCP > 1``; shared
    slab rejects ``DCP > 1`` or ``PCP > 1`` and enforces strict PP slot counts;
    aliased window-compressed hard-rejects ``INT8`` KV.
  * Target MLA, recurrent-state, and draft pools keep strictly independent
    allocation pitches in ``separate_pool_page_pitches``.

[Timeouts & Locks]
  Pure integer arithmetic bounded by layer and stage count; no I/O, locks, or
  device contexts.

[Side-Effect Policy]
  Returns frozen dataclasses, integers, and plain dictionaries; never mutates
  caller models, runtimes, or graphs.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec

#: Declared packed C8 implementation row: one FP32 scale per 128 payload bytes
#: (512 INT8 latent bytes + four FP32 scales + zero/64 BF16 rotary lanes).
C8_SCALE_GROUP = 128
FP32_BYTES = 4
BF16_BYTES = 2
INT8_PACKED_KV_DTYPES = frozenset({"int8", "c8"})
FP8_PACKED_KV_DTYPES = frozenset({"fp8", "fp8_e4m3"})
FP8_MLA_DEFAULT_SCALE_DTYPE = "fp16"
MX_GROUP_SIZE = 32
MXFP8_DTYPES = frozenset({"mxfp8", "mx_fp8", "mxfp8_e4m3"})
MXFP4_DTYPES = frozenset({"mxfp4", "mx_fp4", "mxfp4_e2m1"})
DCP_ACTIVE_PLANS = frozenset({"dcp", "pcp_dcp"})
FRACTION_DENOM_LIMIT = 1024

#: Structural hybrid pool regime identifiers (mutually exclusive).
REGIME_DUAL_PITCH_KPOOL_MAMBA = "dual_pitch_paired_kpool_mamba"
REGIME_SHARED_SLAB_PRIVATE_AUX = "shared_slab_private_aux"
REGIME_ALIASED_WINDOW_COMPRESSED = "aliased_window_compressed"
REGIME_UNIFORM_INDEPENDENT = "uniform_independent"
VALID_POOL_REGIMES = frozenset({
    REGIME_DUAL_PITCH_KPOOL_MAMBA,
    REGIME_SHARED_SLAB_PRIVATE_AUX,
    REGIME_ALIASED_WINDOW_COMPRESSED,
    REGIME_UNIFORM_INDEPENDENT,
})

#: Upstream manager block & slab alignment quanta (grounded in vllm-ascend).
DEFAULT_MANAGER_BLOCK_ALIGN = 128
KPOOL_BLOCK_QUANTUM = 16
SLAB_BASE_ALIGNMENT = 16
QSA_SUBPAGE_KERNEL_BLOCK = 128
TAIL_BUFFER_SIDES = 2
RAW_RING_MIN_FACTOR = 2
KV_PAIR_SIDES = 2
DS_V41_FOLDED_GROUP_TOKENS = 8
DS_V41_FOLDED_ROW_BYTES = 544
DEFAULT_PLE_CONV_KERNEL = 4

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


def _align_up(value: int, alignment: int) -> int:
    """Exact integer round-up of ``value`` to a multiple of ``alignment``."""
    align = int(alignment)
    if align <= 0:
        raise ModelingError(f"cache_pool_alignment_invalid:{align}")
    val = int(value)
    if val <= 0:
        return 0
    return ((val + align - 1) // align) * align


def _lcm_all(values: Iterable[int]) -> int:
    """Exact integer least common multiple across positive values."""
    acc = 1
    for raw in values:
        item = int(raw)
        if item > 0:
            acc = math.lcm(acc, item)
    return acc


def exact_element_bytes(elements: int, dtype: str) -> int:
    """Exact integer allocation bytes for ``elements`` values of ``dtype``.

    Fails closed with ``ModelingError('cache_dtype_missing')`` when ``dtype``
    is empty, and avoids float truncation for 4-bit and MX group-32 formats.
    """
    count = int(elements)
    if count <= 0:
        return 0
    if not dtype or not str(dtype).strip():
        raise ModelingError("cache_dtype_missing")
    key = str(dtype).strip().lower()
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


def _packed_mla_row_parts(
    model: Any, plan: Any, *, phase: str | None = None,
) -> tuple[int, int] | None:
    """Delegate packed MLA row split ``(payload, scales)`` to canonical ``cache_format``.

    Returns ``None`` when the format carries no scales or carries explicit
    trailing alignment padding so ``cache.cache_component_parts`` preserves
    separate padding accounting.
    """
    from mindie_sim.catalog.cache_format import cache_format

    width = int(getattr(plan, "cache_row_elements", 0) or 0)
    if width <= 0:
        return None
    fmt = cache_format(model, "mla_kv", plan=plan, phase=phase)
    payload, scales, padding = fmt.row_parts(width)
    if scales <= 0 or padding > 0:
        return None
    return int(payload), int(scales)


def _packed_mla_row_bytes(
    model: Any, plan: Any, *, phase: str | None = None,
) -> int:
    """Physical MLA row width from the declared per-layer cache representation."""
    from mindie_sim.catalog.cache import packed_region_row_bytes

    parts = _packed_mla_row_parts(model, plan, phase=phase)
    if parts is not None:
        return parts[0] + parts[1]
    return int(packed_region_row_bytes(
        model, "mla_kv", int(plan.cache_row_elements), plan=plan, phase=phase,
    ))


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
    """Per-rank integer single-slot recurrent+conv pitch via canonical ``speculative_state_plan``."""
    from mindie_sim.catalog.speculative_state import speculative_state_plan

    del block_size
    max_page = 0
    for plan in getattr(model, "layer_plans", ()):
        if getattr(plan, "attention", "") not in LINEAR_ATTENTION_KINDS:
            continue
        sp = speculative_state_plan(
            model, plan, parallel,
            query_tokens=1,
            recover_ssm=True,
            capacity_padded=True,
            state_contract=getattr(plan, "state_contract", None),
            runtime=runtime,
        )
        page = int(sp.recurrent_bytes + sp.conv_bytes)
        if page > max_page:
            max_page = page
    return max_page


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
    if str(draft_kind).strip().lower() in {"none", "ngram", "ngram_gpu"}:
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
    """True iff the draft actually shards its KV across DCP ranks."""
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
    """Keep target MLA, recurrent-state and dense-draft pools on independent pitches."""
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
    """Coexistence report for a target MLA pool, linear state and draft."""
    plans = tuple(getattr(model, "layer_plans", ()))
    has_target = any(plan.attention == "mla" and plan.cache_row_elements
                     for plan in plans)
    has_linear = any(plan.attention in LINEAR_ATTENTION_KINDS
                     for plan in plans)
    if not (has_target and has_linear):
        return None
    return separate_pool_page_pitches(
        model, parallel, block_size=block_size, draft_kind=draft_kind,
        dcp=dcp, runtime=runtime,
    )


# ---------------------------------------------------------------------------
# G3 Unified Structural Hybrid Block-Pool Algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoolAllocatorContract:
    """Validated, mutually-exclusive structural hybrid pool contract."""

    pool_regime: str
    raw_block_size: int
    normalized_block_size: int
    main_page_bytes: int
    small_page_bytes: int
    slab_stride_u: int
    slab_align_a: int
    private_bytes_per_block: int
    local_slot_counts: tuple[int, ...]
    small_slot_counts: tuple[int, ...] = ()
    mamba_group_counts: tuple[int, ...] = ()
    mamba_blocks_per_group: int = 0
    draft_page_bytes_by_stage: tuple[int, ...] = ()
    cohosted_mtp_layers: int = 0
    alias_backing_pitches_by_stage: tuple[tuple[int, ...], ...] = ()
    swa_only_page_bytes_by_stage: tuple[int, ...] = ()
    swa_group_counts_by_stage: tuple[int, ...] = ()
    swa_max_window: int = 0
    has_compressor_rings_by_stage: tuple[bool, ...] = ()
    has_draft_swa_by_stage: tuple[bool, ...] = ()
    draft_swa_cap: int = 0
    raw_ring_page_bytes: int = 0
    compressed_index_page_bytes: int = 0
    raw_ring_blocks_per_req: int = 0
    ple_group_counts_by_stage: tuple[int, ...] = ()
    private_ple_page_bytes_by_stage: tuple[int, ...] = ()
    kpool_ratio: int = 1
    subsumed_layers: tuple[int, ...] = ()
    subsumed_components: tuple[str, ...] = ()
    subsumed_draft_kv: bool = False

    def validate_roles(self) -> bool:
        """Verify structural contract integrity invariants."""
        if self.pool_regime not in VALID_POOL_REGIMES:
            raise ModelingError(f"pool_contract_regime_invalid:{self.pool_regime!r}")
        if self.raw_block_size <= 0 or self.normalized_block_size <= 0:
            raise ModelingError("pool_contract_block_size_invalid")
        if self.pool_regime != REGIME_UNIFORM_INDEPENDENT and not self.local_slot_counts:
            raise ModelingError(f"pool_contract_missing_slots:{self.pool_regime}")
        return True


@dataclass(frozen=True, slots=True)
class HybridPoolReceipt:
    """Exact integer per-stage physical allocation receipt for one hybrid pool."""

    contract: PoolAllocatorContract
    blocks_per_request: int
    total_pool_bytes_by_stage: tuple[int, ...]
    subsumed_logical_bytes_by_stage: tuple[int, ...]
    stride_reserve_by_stage: tuple[int, ...]
    outside_replay_bytes_by_stage: tuple[int, ...]
    slab_bytes_by_stage: tuple[int, ...] = ()
    private_bytes_by_stage: tuple[int, ...] = ()
    swa_only_bytes_by_stage: tuple[int, ...] = ()
    pool_unit_bytes_by_stage: tuple[int, ...] = ()
    qsa_blocks_per_request: int = 0
    raw_blocks_per_request: int = 0
    mamba_blocks_per_request: int = 0
    ple_blocks_per_request: int = 0
    per_request_ple_page_bytes: int = 0


def _detect_pool_regime(model: Any) -> str:
    """Classify a model into one mutually-exclusive structural pool regime."""
    from mindie_sim.catalog.weights import is_ple_engram

    plans = tuple(getattr(model, "layer_plans", ()))
    meta = dict(getattr(model, "metadata", ()) or ())
    if meta.get("cache_allocator") == "paired_sfa":
        return REGIME_UNIFORM_INDEPENDENT
    has_mla_kpool = any(
        p.attention == "mla" and int(p.index_kpool or 1) > 1 and bool(p.owns_index_k)
        for p in plans
    )
    has_gqa_kpool = any(
        p.attention == "gqa" and bool(p.indexer) and int(p.index_kpool or 1) > 1
        for p in plans
    )
    has_linear = any(p.attention in LINEAR_ATTENTION_KINDS for p in plans)
    has_window_compressed = any(
        str(getattr(p, "cache_kind", "")) == "window_compressed"
        and str(getattr(p, "mla_variant", "")) == "latent_mqa"
        for p in plans
    )
    active_count = sum(bool(flag) for flag in (has_mla_kpool, has_gqa_kpool, has_window_compressed))
    if active_count > 1:
        raise ModelingError("ambiguous_or_incomplete_pool_contract:conflicting_hybrid_roles")
    if has_mla_kpool:
        return REGIME_DUAL_PITCH_KPOOL_MAMBA
    if has_gqa_kpool and has_linear:
        if not is_ple_engram(getattr(model, "engram", None)):
            raise ModelingError("ambiguous_or_incomplete_pool_contract:shared_slab_missing_ple")
        return REGIME_SHARED_SLAB_PRIVATE_AUX
    if has_window_compressed:
        return REGIME_ALIASED_WINDOW_COMPRESSED
    return REGIME_UNIFORM_INDEPENDENT


def _mamba_allowance_blocks(runtime: Any, proposals: int) -> int:
    """Blocks reserved per recurrent state group (``none`` vs ``align`` mode)."""
    from mindie_sim.catalog.speculative_state import runtime_recover_ssm

    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    mode = str(rt_meta.get("mamba_cache_mode") or "none").strip().lower()
    if mode not in {"none", "align"}:
        raise ModelingError(f"hybrid_pool_mamba_cache_mode_invalid:{mode}")
    recover = runtime_recover_ssm(runtime)
    n_b = 0 if recover else max(0, int(proposals))
    extra = int(rt_meta.get("num_prefill_checkpoint_blocks", 0) or 0)
    if extra < 0:
        raise ModelingError(f"hybrid_pool_negative_prefill_blocks:{extra}")
    return (2 + n_b + extra) if mode == "align" else (1 + n_b)


def _consecutive_linear_run_groups(plans: Sequence[Any]) -> list[list[int]]:
    """Group linear-attention layers by maximum consecutive run length (exact upstream rule)."""
    linear_ids = sorted(int(p.index) for p in plans if p.attention in LINEAR_ATTENTION_KINDS)
    if not linear_ids:
        return []
    linear_set = set(linear_ids)
    max_run = 0
    run = 0
    prev_idx: int | None = None
    for idx in sorted(int(p.index) for p in plans):
        is_consec = prev_idx is not None and idx == prev_idx + 1
        if idx in linear_set:
            run = (run + 1) if is_consec else 1
            if run > max_run:
                max_run = run
        else:
            run = 0
        prev_idx = idx
    if max_run <= 0:
        return []
    return [linear_ids[offset::max_run] for offset in range(max_run)]


def _resolve_kpool_mamba_contract(
    model: Any,
    parallel: ParallelSpec,
    *,
    raw_block: int,
    stage_map: tuple[int, ...],
    proposals: int,
    draft_kind: str,
    runtime: Any,
    phase: str,
) -> PoolAllocatorContract:
    """Resolve Regime 1: ``dual_pitch_paired_kpool_mamba`` (patch_mamba_config + dual pitch)."""
    from mindie_sim.catalog.cache_index import index_row_parts
    from mindie_sim.catalog.paired_sfa import independent_draft
    from mindie_sim.catalog.speculative_state import runtime_recover_ssm, speculative_state_plan

    dcp = max(1, int(getattr(parallel, "cp", 1) or 1))
    if dcp > 1:
        raise ModelingError("kpool_indexer_requires_dcp_1")
    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    plans = tuple(model.layer_plans)
    mla_plans = [p for p in plans if p.attention == "mla"]
    linear_plans = [p for p in plans if p.attention in LINEAR_ATTENTION_KINDS]
    kpool_ratios = {max(1, int(p.index_kpool or 1)) for p in mla_plans if p.owns_index_k}
    if not mla_plans or len(kpool_ratios) != 1:
        raise ModelingError("ambiguous_or_incomplete_pool_contract:kpool_ratio_mismatch")
    kpool = next(iter(kpool_ratios))
    if raw_block % kpool != 0:
        raise ModelingError(f"kpool_block_size_not_divisible:{raw_block}%{kpool}")
    if runtime_recover_ssm(runtime) and (pp > 1 or dcp > 1):
        raise ModelingError("recover_ssm_requires_pp_1_dcp_1")

    p_kda = 0
    for lp in linear_plans:
        sp = speculative_state_plan(
            model, lp, parallel,
            query_tokens=1 + max(0, proposals),
            recover_ssm=True,
            capacity_padded=True,
            state_contract=getattr(lp, "state_contract", None),
            runtime=runtime,
        )
        p_kda = max(p_kda, int(sp.recurrent_bytes + sp.conv_bytes))

    mla_rows = [_packed_mla_row_bytes(model, lp, phase=phase) for lp in mla_plans]
    r_main_min = min(mla_rows)
    r_main_max = max(mla_rows)
    if r_main_min <= 0:
        raise ModelingError("kpool_mla_row_bytes_invalid")
    a_mgr = math.lcm(DEFAULT_MANAGER_BLOCK_ALIGN, KPOOL_BLOCK_QUANTUM * kpool)
    b_eff = max(raw_block, _align_up(_ceil_div(p_kda, r_main_min), a_mgr)) if p_kda > 0 else raw_block
    p_m = max(b_eff * r_main_max, p_kda)

    b_storage = b_eff // kpool
    p_idx_max = 0
    p_tail_max = 0
    scale_align_terms: list[int] = []
    has_idx_scale = False
    for lp in mla_plans:
        if not lp.owns_index_k:
            continue
        p_b, s_b, pad_b = index_row_parts(model, lp.index_dim, plan=lp, phase=phase)
        r_idx = int(p_b + s_b + pad_b)
        p_idx_max = max(p_idx_max, b_storage * r_idx)
        r_tail = TAIL_BUFFER_SIDES * int(lp.index_dim) * FP32_BYTES
        p_tail_max = max(p_tail_max, (kpool + max(0, proposals)) * r_tail)
        scale_align_terms.append(r_tail)
        if s_b > 0:
            has_idx_scale = True
            scale_align_terms.append(b_storage * int(s_b))
    p_s_raw = max(p_idx_max, p_tail_max)
    p_s = _align_up(p_s_raw, _lcm_all(scale_align_terms)) if has_idx_scale else p_s_raw

    global_mamba_groups = _consecutive_linear_run_groups(plans)
    draft = _resolve_draft_spec(model, draft_kind)
    cohosted_mtp = 0
    foreign_draft_page = 0
    subsumed_draft = False
    if draft is not None:
        if (
            str(getattr(draft, "kind", "")) == "mtp"
            and str(getattr(draft, "attention_kind", "")) == "mla"
            and not independent_draft(draft)
        ):
            cohosted_mtp = max(1, int(draft.layers))
            subsumed_draft = True
        else:
            tp = max(1, int(getattr(parallel, "tp", 1) or 1))
            foreign_draft_page = b_eff * _draft_row_bytes_int(model, draft, tp=tp) * max(1, int(draft.layers))
            subsumed_draft = True

    s_m_by_stage: list[int] = []
    s_s_by_stage: list[int] = []
    g_mamba_by_stage: list[int] = []
    draft_pages_by_stage: list[int] = []
    for st in range(pp):
        st_mla = [lp for lp in mla_plans if stage_map[lp.index] == st]
        st_mamba_groups = [
            [idx for idx in grp if stage_map[idx] == st]
            for grp in global_mamba_groups
        ]
        st_mtp = cohosted_mtp if st == pp - 1 else 0
        s_m = max([len(st_mla) + st_mtp, *(len(g) for g in st_mamba_groups)], default=0)
        s_s = len(st_mla) + st_mtp
        s_m_by_stage.append(s_m)
        s_s_by_stage.append(s_s)
        g_mamba_by_stage.append(sum(1 for g in st_mamba_groups if g))
        draft_pages_by_stage.append(foreign_draft_page if st == pp - 1 else 0)

    contract = PoolAllocatorContract(
        pool_regime=REGIME_DUAL_PITCH_KPOOL_MAMBA,
        raw_block_size=raw_block,
        normalized_block_size=b_eff,
        main_page_bytes=p_m,
        small_page_bytes=p_s,
        slab_stride_u=0,
        slab_align_a=a_mgr,
        private_bytes_per_block=0,
        local_slot_counts=tuple(s_m_by_stage),
        small_slot_counts=tuple(s_s_by_stage),
        mamba_group_counts=tuple(g_mamba_by_stage),
        mamba_blocks_per_group=_mamba_allowance_blocks(runtime, proposals),
        draft_page_bytes_by_stage=tuple(draft_pages_by_stage),
        cohosted_mtp_layers=cohosted_mtp,
        kpool_ratio=kpool,
        subsumed_layers=tuple(int(p.index) for p in plans),
        subsumed_components=(
            "mla_kv", "index_k", "index_pool_tail", "recurrent_state", "conv_state",
        ),
        subsumed_draft_kv=subsumed_draft,
    )
    contract.validate_roles()
    return contract


def _resolve_shared_slab_contract(
    model: Any,
    parallel: ParallelSpec,
    *,
    raw_block: int,
    stage_map: tuple[int, ...],
    proposals: int,
    runtime: Any,
    phase: str,
) -> PoolAllocatorContract:
    """Resolve Regime 2: ``shared_slab_private_aux`` (QSA slab + GDN + private PLE)."""
    from mindie_sim.catalog.cache import _gqa_layer_row_bytes, local_kv_heads
    from mindie_sim.catalog.cache_index import index_row_bytes
    from mindie_sim.catalog.speculative_state import speculative_state_plan

    dcp = max(1, int(getattr(parallel, "cp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    if dcp > 1 or pcp > 1:
        raise ModelingError("shared_slab_requires_dcp_1_pcp_1")
    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    plans = tuple(model.layer_plans)
    qsa_plans = [p for p in plans if p.attention == "gqa" and p.indexer]
    gdn_plans = [p for p in plans if p.attention in LINEAR_ATTENTION_KINDS]
    engram = getattr(model, "engram", None)
    if not qsa_plans or not gdn_plans or engram is None:
        raise ModelingError("ambiguous_or_incomplete_pool_contract:shared_slab_incomplete")

    kpool = max(1, int(qsa_plans[0].index_kpool or 1))
    if any(max(1, int(p.index_kpool or 1)) != kpool for p in qsa_plans):
        raise ModelingError("ambiguous_or_incomplete_pool_contract:qsa_kpool_mismatch")

    int8_fa3 = gbsa_cache_layout(runtime) == "int8_fa3"
    act_dtype = str(getattr(model, "activation_dtype", "") or "").strip()
    if not act_dtype:
        raise ModelingError("cache_dtype_missing")
    act_bytes = exact_element_bytes(1, act_dtype)
    kv_dtype = str(
        getattr(getattr(model, "quant", None), "kv_dtype", "")
        or getattr(model, "kv_dtype", "")
        or act_dtype
    ).strip()
    idx_dtype = str(
        getattr(getattr(model, "quant", None), "index_dtype", "")
        or act_dtype
    ).strip()
    kv_elem_bytes = 1 if int8_fa3 else exact_element_bytes(1, kv_dtype)
    idx_elem_bytes = 1 if int8_fa3 else exact_element_bytes(1, idx_dtype)

    max_ssm_bytes = 0
    conv_bytes = 0
    dtype_sizes: set[int] = {act_bytes, kv_elem_bytes, idx_elem_bytes}
    for lp in gdn_plans:
        # Upstream qwen4_exp/cache_config.py:237 uses ONE single physical SSM matrix (q=1)
        # in state_stride, with conv history widened by configured proposals (zero replay_bytes!).
        sp = speculative_state_plan(
            model, lp, parallel,
            query_tokens=1 + max(0, proposals),
            recover_ssm=True,
            capacity_padded=True,
            state_contract=getattr(lp, "state_contract", None),
            runtime=runtime,
        )
        max_ssm_bytes = max(max_ssm_bytes, int(sp.recurrent_bytes))
        conv_bytes = max(conv_bytes, int(sp.conv_bytes))
        dtype_sizes.add(exact_element_bytes(1, str(lp.recurrent_state_dtype)))
        dtype_sizes.add(exact_element_bytes(1, str(lp.conv_state_dtype)))

    kv_token_bytes_list: list[int] = []
    subpage_terms: list[int] = []
    for lp in qsa_plans:
        loc_kv = int(local_kv_heads(int(lp.num_kv_heads), tp))
        row_b = int(_gqa_layer_row_bytes(model, lp, tp, int8_fa3=int8_fa3, phase=phase))
        kv_token_bytes_list.append(row_b)
        d_k = int(lp.head_dim)
        d_v = int(getattr(lp, "v_head_dim", 0) or d_k)
        subpage_terms.append(KV_PAIR_SIDES * QSA_SUBPAGE_KERNEL_BLOCK * loc_kv * d_k * kv_elem_bytes)
        subpage_terms.append(KV_PAIR_SIDES * QSA_SUBPAGE_KERNEL_BLOCK * loc_kv * d_v * kv_elem_bytes)

    min_kv_row = min(kv_token_bytes_list)
    block_quantum = math.lcm(QSA_SUBPAGE_KERNEL_BLOCK, kpool)
    min_block = _align_up(_ceil_div(max_ssm_bytes, min_kv_row), block_quantum)
    b_eff = max(min_block, _align_up(raw_block, block_quantum))
    main_pages = [b_eff * rb for rb in kv_token_bytes_list]
    subpage_u = _lcm_all(subpage_terms)
    align_a = _lcm_all([SLAB_BASE_ALIGNMENT, *dtype_sizes])
    state_stride = _align_up(max(conv_bytes + max_ssm_bytes, *main_pages), subpage_u)

    raw_ring_cap = max(RAW_RING_MIN_FACTOR * kpool, _align_up(kpool + max(0, proposals), kpool))
    raw_max = max(raw_ring_cap * int(lp.index_dim) * act_bytes for lp in qsa_plans)
    comp_pages: list[int] = []
    for lp in qsa_plans:
        idx_row = (
            int(lp.index_dim) + FP32_BYTES
            if int8_fa3 else int(index_row_bytes(model, lp.index_dim, plan=lp, phase=phase))
        )
        comp_pages.append((b_eff // kpool) * idx_row)
    comp_max = max(comp_pages)

    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    conv_state_len = (conv_k - 1) * int(engram.max_ngram) + max(0, proposals)
    hc_mult = max((int(p.hc_mult) for p in plans), default=1) or 1
    ple_page_per_table = int(conv_state_len * hc_mult * int(model.hidden_size) * act_bytes)

    slot_counts: list[int] = []
    gdn_groups_by_stage: list[int] = []
    ple_groups_by_stage: list[int] = []
    priv_ple_by_stage: list[int] = []
    for st in range(pp):
        st_qsa = [lp for lp in qsa_plans if stage_map[lp.index] == st]
        st_gdn = [lp for lp in gdn_plans if stage_map[lp.index] == st]
        st_ple = [t for t in engram.tables if 0 <= t.layer_id < len(stage_map) and stage_map[t.layer_id] == st]
        s_local = len(st_qsa)
        if st_gdn and s_local <= 0:
            raise ModelingError(f"shared_slab_pp_stage_missing_qsa_slots:stage={st}:gdn={len(st_gdn)}")
        slot_counts.append(s_local)
        gdn_groups_by_stage.append(_ceil_div(len(st_gdn), s_local) if st_gdn else 0)
        ple_groups_by_stage.append(
            _ceil_div(len(st_ple), s_local) if (st_ple and s_local > 0) else (1 if st_ple else 0)
        )
        priv_ple_by_stage.append(len(st_ple) * ple_page_per_table)

    contract = PoolAllocatorContract(
        pool_regime=REGIME_SHARED_SLAB_PRIVATE_AUX,
        raw_block_size=raw_block,
        normalized_block_size=b_eff,
        main_page_bytes=state_stride,
        small_page_bytes=comp_max,
        slab_stride_u=subpage_u,
        slab_align_a=align_a,
        private_bytes_per_block=sum(priv_ple_by_stage),
        local_slot_counts=tuple(slot_counts),
        mamba_group_counts=tuple(gdn_groups_by_stage),
        mamba_blocks_per_group=_mamba_allowance_blocks(runtime, proposals),
        raw_ring_page_bytes=raw_max,
        compressed_index_page_bytes=comp_max,
        raw_ring_blocks_per_req=1,
        ple_group_counts_by_stage=tuple(ple_groups_by_stage),
        private_ple_page_bytes_by_stage=tuple(priv_ple_by_stage),
        kpool_ratio=kpool,
        subsumed_layers=tuple(int(p.index) for p in plans),
        subsumed_components=(
            "gqa_kv", "index_k", "index_pool_tail", "recurrent_state", "conv_state", "ple_conv_state",
        ),
        subsumed_draft_kv=False,
    )
    contract.validate_roles()
    return contract


def _resolve_aliased_window_contract(
    model: Any,
    parallel: ParallelSpec,
    *,
    raw_block: int,
    stage_map: tuple[int, ...],
    proposals: int,
    draft_kind: str,
    runtime: Any,
    phase: str,
) -> PoolAllocatorContract:
    """Resolve Regime 3: ``aliased_window_compressed`` (4 physical backings + A3 INT8 ban)."""
    from mindie_sim.catalog.cache import packed_region_row_bytes
    from mindie_sim.catalog.cache_index import index_row_bytes
    from mindie_sim.catalog.draft_spec import effective_draft_context_cap

    dcp = max(1, int(getattr(parallel, "cp", 1) or 1))
    if dcp > 1:
        raise ModelingError("aliased_window_compressed_requires_dcp_1")
    kv_dt = str(
        getattr(getattr(model, "quant", None), "kv_dtype", "")
        or getattr(model, "kv_dtype", "")
    ).strip().lower()
    if kv_dt in INT8_PACKED_KV_DTYPES:
        raise ModelingError("aliased_window_compressed_rejects_int8_kv")

    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    plans = tuple(model.layer_plans)
    draft = _resolve_draft_spec(model, draft_kind)
    draft_active = draft is not None
    draft_layers = int(draft.layers) if draft_active else 0
    draft_swa_page = (
        raw_block * _draft_row_bytes_int(model, draft, tp=tp) if draft_active else 0
    )
    draft_cap = effective_draft_context_cap(draft, runtime) if draft_active else 0
    max_window = max((int(p.window or 0) for p in plans), default=0)

    pitches_by_stage: list[tuple[int, ...]] = []
    swa_only_by_stage: list[int] = []
    swa_groups_by_stage: list[int] = []
    has_rings_by_stage: list[bool] = []
    has_draft_by_stage: list[bool] = []
    slot_counts: list[int] = []
    is_fp8_packed = kv_dt in FP8_PACKED_KV_DTYPES

    for st in range(pp):
        st_plans = [p for p in plans if stage_map[p.index] == st]
        full_owners = sorted((p for p in st_plans if p.owns_long_kv), key=lambda p: int(p.index))
        swa_plans = sorted((p for p in st_plans if int(p.window or 0) > 0), key=lambda p: int(p.index))
        comp_rings = [p for p in full_owners if int(p.compressor_state_rows or 0) > 0]
        st_draft_count = draft_layers if st == pp - 1 else 0
        if not full_owners:
            swa_total = sum(
                raw_block * int(packed_region_row_bytes(
                    model, "window", p.cache_row_elements, plan=p, phase=phase,
                ))
                for p in swa_plans
            ) + st_draft_count * draft_swa_page
            pitches_by_stage.append(())
            swa_only_by_stage.append(swa_total)
            swa_groups_by_stage.append(len(swa_plans))
            has_rings_by_stage.append(False)
            has_draft_by_stage.append(st_draft_count > 0)
            slot_counts.append(len(swa_plans) + st_draft_count)
            continue

        num_backings = len(full_owners)
        stage_pitches: list[int] = []
        for slot_idx, owner in enumerate(full_owners):
            ratio = max(1, int(owner.compress_ratio or 1))
            long_rows = max(1, _ceil_div(raw_block, ratio))
            long_row_bytes = int(packed_region_row_bytes(
                model, "compressed_kv", owner.cache_row_elements, plan=owner, phase=phase,
            ))
            long_bytes = long_rows * long_row_bytes
            idx_bytes = 0
            folded_bytes = 0
            if owner.owns_index_k:
                idx_ratio = max(1, int(owner.compress_ratio or 1), int(owner.index_kpool or 1))
                idx_rows = max(1, _ceil_div(raw_block, idx_ratio))
                idx_bytes = idx_rows * int(index_row_bytes(model, owner.index_dim, plan=owner, phase=phase))
                is_packed_folded = (
                    is_fp8_packed
                    or long_row_bytes < int(owner.cache_row_elements) * BF16_BYTES
                )
                if is_packed_folded and int(owner.compress_ratio or 1) <= 1:
                    folded_bytes = _ceil_div(raw_block, DS_V41_FOLDED_GROUP_TOKENS) * DS_V41_FOLDED_ROW_BYTES
            alias_pages: list[int] = [
                raw_block * int(packed_region_row_bytes(
                    model, "window", sp.cache_row_elements, plan=sp, phase=phase,
                ))
                for sp in swa_plans[slot_idx::num_backings]
            ]
            if slot_idx < len(comp_rings):
                ring = comp_rings[slot_idx]
                ring_bytes = (
                    int(ring.compressor_state_rows)
                    * KV_PAIR_SIDES
                    * max(1, int(ring.compressor_width or 1))
                    * int(ring.head_dim)
                    * FP32_BYTES
                )
                alias_pages.append(ring_bytes)
            if slot_idx < st_draft_count and draft_swa_page > 0:
                alias_pages.append(draft_swa_page)
            stage_pitches.append(max(long_bytes + idx_bytes + folded_bytes, *alias_pages))

        pitches_by_stage.append(tuple(stage_pitches))
        swa_only_by_stage.append(0)
        swa_groups_by_stage.append(_ceil_div(len(swa_plans), num_backings) if swa_plans else 0)
        has_rings_by_stage.append(bool(comp_rings))
        has_draft_by_stage.append(st_draft_count > 0)
        slot_counts.append(num_backings)

    all_pitches = [p for st_p in pitches_by_stage for p in st_p]
    contract = PoolAllocatorContract(
        pool_regime=REGIME_ALIASED_WINDOW_COMPRESSED,
        raw_block_size=raw_block,
        normalized_block_size=raw_block,
        main_page_bytes=max(all_pitches, default=0),
        small_page_bytes=0,
        slab_stride_u=0,
        slab_align_a=SLAB_BASE_ALIGNMENT,
        private_bytes_per_block=0,
        local_slot_counts=tuple(slot_counts),
        alias_backing_pitches_by_stage=tuple(pitches_by_stage),
        swa_only_page_bytes_by_stage=tuple(swa_only_by_stage),
        swa_group_counts_by_stage=tuple(swa_groups_by_stage),
        swa_max_window=max_window,
        has_compressor_rings_by_stage=tuple(has_rings_by_stage),
        has_draft_swa_by_stage=tuple(has_draft_by_stage),
        draft_swa_cap=draft_cap,
        subsumed_layers=tuple(int(p.index) for p in plans),
        subsumed_components=(
            "window", "compressed_kv", "index_k", "compressor_state", "index_compressor_state",
        ),
        subsumed_draft_kv=draft_active,
    )
    contract.validate_roles()
    return contract


def resolve_pool_allocator_contract(
    model: Any,
    parallel: ParallelSpec,
    *,
    block_size: int | None = None,
    runtime: Any = None,
    draft_kind: str | None = None,
    query_tokens: int | None = None,
    phase: str = "decode",
) -> PoolAllocatorContract:
    """Resolve one validated, mutually-exclusive ``PoolAllocatorContract`` for ``model``."""
    from mindie_sim.catalog.cache import DEFAULT_CACHE_BLOCK_SIZE, layer_stage_map
    from mindie_sim.catalog.cache_ownership import _active_draft_kind
    from mindie_sim.catalog.speculative_state import resolve_speculative_proposals

    raw_block = int(
        block_size if block_size is not None else (
            getattr(runtime, "kv_block_size", 0) or DEFAULT_CACHE_BLOCK_SIZE
        )
    )
    if raw_block <= 0:
        raise ModelingError(f"cache_pool_block_size_invalid:{raw_block}")
    regime = _detect_pool_regime(model)
    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    if regime == REGIME_UNIFORM_INDEPENDENT:
        return PoolAllocatorContract(
            pool_regime=REGIME_UNIFORM_INDEPENDENT,
            raw_block_size=raw_block,
            normalized_block_size=raw_block,
            main_page_bytes=0,
            small_page_bytes=0,
            slab_stride_u=0,
            slab_align_a=1,
            private_bytes_per_block=0,
            local_slot_counts=(0,) * pp,
        )
    stage_map = layer_stage_map(model, parallel, runtime)
    if str(phase or "decode").strip().lower() == "prefill":
        proposals = 0
    elif query_tokens is not None:
        proposals = max(0, int(query_tokens) - 1)
    else:
        proposals = resolve_speculative_proposals(runtime, phase=phase)
    eff_draft = (
        str(draft_kind).strip().lower()
        if draft_kind is not None
        else _active_draft_kind(model, runtime)
    )
    if regime == REGIME_DUAL_PITCH_KPOOL_MAMBA:
        return _resolve_kpool_mamba_contract(
            model, parallel, raw_block=raw_block, stage_map=stage_map,
            proposals=proposals, draft_kind=eff_draft, runtime=runtime, phase=phase,
        )
    if regime == REGIME_SHARED_SLAB_PRIVATE_AUX:
        return _resolve_shared_slab_contract(
            model, parallel, raw_block=raw_block, stage_map=stage_map,
            proposals=proposals, runtime=runtime, phase=phase,
        )
    return _resolve_aliased_window_contract(
        model, parallel, raw_block=raw_block, stage_map=stage_map,
        proposals=proposals, draft_kind=eff_draft, runtime=runtime, phase=phase,
    )


def _subsumed_and_outside_by_stage(
    model: Any,
    parallel: ParallelSpec,
    contract: PoolAllocatorContract,
    *,
    sequence_tokens: int,
    num_requests: int,
    runtime: Any,
    query_tokens: int | None,
    draft_kind: str,
    phase: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute per-stage subsumed logical component bytes and outside replay bytes."""
    from mindie_sim.catalog.cache import layer_cache_components, layer_stage_map
    from mindie_sim.catalog.cache_ownership import draft_cache_components

    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    subsumed = [0] * pp
    outside_replay = [0] * pp
    if sequence_tokens <= 0 or num_requests <= 0:
        return tuple(subsumed), tuple(outside_replay)
    stage_map = layer_stage_map(model, parallel, runtime)
    sub_layers = set(contract.subsumed_layers)
    sub_comps = set(contract.subsumed_components)
    for lp in model.layer_plans:
        st = stage_map[lp.index] if lp.index < len(stage_map) else pp - 1
        comps = layer_cache_components(
            model, lp, parallel, sequence_tokens,
            capacity_padded=True,
            interleave=contract.normalized_block_size,
            block_size=contract.normalized_block_size,
            runtime=runtime,
            query_tokens=query_tokens,
            phase=phase,
        )
        for name, val in comps.items():
            ival = int(math.ceil(float(val)))
            if name in {"replay_records", "state_transient"}:
                outside_replay[st] += ival * num_requests
            elif int(lp.index) in sub_layers and name in sub_comps:
                subsumed[st] += ival * num_requests
    if contract.subsumed_draft_kv and draft_kind:
        d_comps = draft_cache_components(
            model, parallel, draft_kind=draft_kind, sequence_tokens=sequence_tokens,
            runtime=runtime, capacity_padded=True, query_tokens=query_tokens, phase=phase,
        )
        subsumed[-1] += int(math.ceil(float(d_comps.get("draft_kv", 0.0)))) * num_requests
    return tuple(subsumed), tuple(outside_replay)


def _eval_kpool_mamba_receipt(
    contract: PoolAllocatorContract,
    parallel: ParallelSpec,
    *,
    seq: int,
    reqs: int,
    subsumed: tuple[int, ...],
    outside_replay: tuple[int, ...],
) -> HybridPoolReceipt:
    pp = len(contract.local_slot_counts)
    dcp = max(1, int(getattr(parallel, "cp", 1) or 1))
    n_full = _ceil_div(seq, contract.normalized_block_size * dcp)
    n_tail = 1
    g_mamba_max = max(contract.mamba_group_counts, default=0)
    n_mamba = g_mamba_max * contract.mamba_blocks_per_group
    b_req = n_full + n_tail + n_mamba
    total_blocks = reqs * b_req
    pool_units = tuple(
        contract.local_slot_counts[st] * contract.main_page_bytes
        + contract.small_slot_counts[st] * contract.small_page_bytes
        + contract.draft_page_bytes_by_stage[st]
        for st in range(pp)
    )
    totals = tuple(total_blocks * unit for unit in pool_units)
    reserves = tuple(max(0, totals[st] - subsumed[st]) for st in range(pp))
    return HybridPoolReceipt(
        contract=contract,
        blocks_per_request=b_req,
        total_pool_bytes_by_stage=totals,
        subsumed_logical_bytes_by_stage=subsumed,
        stride_reserve_by_stage=reserves,
        outside_replay_bytes_by_stage=outside_replay,
        pool_unit_bytes_by_stage=pool_units,
        mamba_blocks_per_request=n_mamba,
    )


def _eval_shared_slab_receipt(
    contract: PoolAllocatorContract,
    *,
    seq: int,
    reqs: int,
    subsumed: tuple[int, ...],
    outside_replay: tuple[int, ...],
) -> HybridPoolReceipt:
    pp = len(contract.local_slot_counts)
    n_qsa = _ceil_div(seq, contract.normalized_block_size)
    n_raw = contract.raw_ring_blocks_per_req
    n_gdn = max(contract.mamba_group_counts, default=0) * contract.mamba_blocks_per_group
    n_ple = max(contract.ple_group_counts_by_stage, default=0) * contract.mamba_blocks_per_group
    b_req = n_qsa + n_raw + n_gdn + n_ple
    n_tot = reqs * b_req
    r0 = n_tot * contract.main_page_bytes
    r4_off = _align_up(r0, contract.slab_align_a)
    r5_off = _align_up(r4_off + n_tot * contract.raw_ring_page_bytes, contract.slab_align_a)
    slab_n = _align_up(r5_off + n_tot * contract.compressed_index_page_bytes, contract.slab_align_a)
    slab_1 = _align_up(
        _align_up(
            _align_up(contract.main_page_bytes, contract.slab_align_a)
            + contract.raw_ring_page_bytes,
            contract.slab_align_a,
        ) + contract.compressed_index_page_bytes,
        contract.slab_align_a,
    )
    slab_by_stage = tuple(contract.local_slot_counts[st] * slab_n for st in range(pp))
    priv_by_stage = tuple(n_tot * contract.private_ple_page_bytes_by_stage[st] for st in range(pp))
    totals = tuple(slab_by_stage[st] + priv_by_stage[st] for st in range(pp))
    pool_units = tuple(
        contract.local_slot_counts[st] * slab_1 + contract.private_ple_page_bytes_by_stage[st]
        for st in range(pp)
    )
    reserves = tuple(max(0, totals[st] - subsumed[st]) for st in range(pp))
    return HybridPoolReceipt(
        contract=contract,
        blocks_per_request=b_req,
        total_pool_bytes_by_stage=totals,
        subsumed_logical_bytes_by_stage=subsumed,
        stride_reserve_by_stage=reserves,
        outside_replay_bytes_by_stage=outside_replay,
        slab_bytes_by_stage=slab_by_stage,
        private_bytes_by_stage=priv_by_stage,
        pool_unit_bytes_by_stage=pool_units,
        qsa_blocks_per_request=n_qsa,
        raw_blocks_per_request=n_raw,
        mamba_blocks_per_request=n_gdn,
        ple_blocks_per_request=n_ple,
        per_request_ple_page_bytes=contract.private_bytes_per_block,
    )


def _eval_aliased_window_receipt(
    contract: PoolAllocatorContract,
    *,
    seq: int,
    reqs: int,
    subsumed: tuple[int, ...],
    outside_replay: tuple[int, ...],
) -> HybridPoolReceipt:
    pp = len(contract.local_slot_counts)
    block = contract.normalized_block_size
    n_long = _ceil_div(seq, block)
    win = contract.swa_max_window
    swa_eff_rows = min(seq, win) + (block if win > 0 and seq > win else 0) if win > 0 else seq
    n_swa_per_group = _ceil_div(swa_eff_rows, block)
    totals_list: list[int] = []
    swa_only_list: list[int] = []
    units_list: list[int] = []
    max_b_req = 0
    for st in range(pp):
        pitches = contract.alias_backing_pitches_by_stage[st]
        if not pitches:
            swa_bytes = reqs * n_swa_per_group * contract.swa_only_page_bytes_by_stage[st]
            totals_list.append(swa_bytes)
            swa_only_list.append(swa_bytes)
            units_list.append(contract.swa_only_page_bytes_by_stage[st])
            max_b_req = max(max_b_req, n_swa_per_group)
            continue
        n_state = 1 if contract.has_compressor_rings_by_stage[st] else 0
        n_swa_total = contract.swa_group_counts_by_stage[st] * n_swa_per_group
        n_draft = 0
        if contract.has_draft_swa_by_stage[st]:
            d_cap = contract.draft_swa_cap if contract.draft_swa_cap > 0 else win
            d_rows = min(seq, d_cap) + (block if d_cap > 0 and seq > d_cap else 0) if d_cap > 0 else seq
            n_draft = _ceil_div(d_rows, block)
        st_b_req = n_long + n_state + n_swa_total + n_draft
        max_b_req = max(max_b_req, st_b_req)
        unit = sum(pitches)
        totals_list.append(reqs * st_b_req * unit)
        swa_only_list.append(0)
        units_list.append(unit)
    totals_t = tuple(totals_list)
    reserves_t = tuple(max(0, totals_t[st] - subsumed[st]) for st in range(pp))
    return HybridPoolReceipt(
        contract=contract,
        blocks_per_request=max_b_req,
        total_pool_bytes_by_stage=totals_t,
        subsumed_logical_bytes_by_stage=subsumed,
        stride_reserve_by_stage=reserves_t,
        outside_replay_bytes_by_stage=outside_replay,
        swa_only_bytes_by_stage=tuple(swa_only_list),
        pool_unit_bytes_by_stage=tuple(units_list),
    )


def evaluate_hybrid_pool_receipt(
    model: Any,
    parallel: ParallelSpec,
    sequence_tokens: int,
    *,
    num_requests: int = 1,
    block_size: int | None = None,
    runtime: Any = None,
    draft_kind: str | None = None,
    query_tokens: int | None = None,
    phase: str = "decode",
) -> HybridPoolReceipt:
    """Evaluate exact integer per-stage physical bytes for a hybrid block pool."""
    from mindie_sim.catalog.cache_ownership import _active_draft_kind

    seq = max(0, int(sequence_tokens))
    reqs = max(0, int(num_requests))
    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    eff_draft = (
        str(draft_kind).strip().lower()
        if draft_kind is not None
        else _active_draft_kind(model, runtime)
    )
    contract = resolve_pool_allocator_contract(
        model, parallel, block_size=block_size, runtime=runtime,
        draft_kind=eff_draft, query_tokens=query_tokens, phase=phase,
    )
    if contract.pool_regime == REGIME_UNIFORM_INDEPENDENT or seq <= 0 or reqs <= 0:
        zeros = (0,) * pp
        return HybridPoolReceipt(
            contract=contract,
            blocks_per_request=0,
            total_pool_bytes_by_stage=zeros,
            subsumed_logical_bytes_by_stage=zeros,
            stride_reserve_by_stage=zeros,
            outside_replay_bytes_by_stage=zeros,
            slab_bytes_by_stage=zeros,
            private_bytes_by_stage=zeros,
            swa_only_bytes_by_stage=zeros,
            pool_unit_bytes_by_stage=zeros,
        )

    subsumed, outside_replay = _subsumed_and_outside_by_stage(
        model, parallel, contract,
        sequence_tokens=seq, num_requests=reqs, runtime=runtime,
        query_tokens=query_tokens, draft_kind=eff_draft, phase=phase,
    )
    if contract.pool_regime == REGIME_DUAL_PITCH_KPOOL_MAMBA:
        return _eval_kpool_mamba_receipt(
            contract, parallel, seq=seq, reqs=reqs,
            subsumed=subsumed, outside_replay=outside_replay,
        )
    if contract.pool_regime == REGIME_SHARED_SLAB_PRIVATE_AUX:
        return _eval_shared_slab_receipt(
            contract, seq=seq, reqs=reqs,
            subsumed=subsumed, outside_replay=outside_replay,
        )
    return _eval_aliased_window_receipt(
        contract, seq=seq, reqs=reqs,
        subsumed=subsumed, outside_replay=outside_replay,
    )


__all__ = [
    "BF16_BYTES",
    "C8_SCALE_GROUP",
    "DCP_ACTIVE_PLANS",
    "DEFAULT_MANAGER_BLOCK_ALIGN",
    "DS_V41_FOLDED_GROUP_TOKENS",
    "DS_V41_FOLDED_ROW_BYTES",
    "FP32_BYTES",
    "FP8_MLA_DEFAULT_SCALE_DTYPE",
    "FP8_PACKED_KV_DTYPES",
    "HybridPoolReceipt",
    "INT8_PACKED_KV_DTYPES",
    "KPOOL_BLOCK_QUANTUM",
    "LINEAR_ATTENTION_KINDS",
    "MXFP4_DTYPES",
    "MXFP8_DTYPES",
    "MX_GROUP_SIZE",
    "PoolAllocatorContract",
    "QSA_SUBPAGE_KERNEL_BLOCK",
    "REGIME_ALIASED_WINDOW_COMPRESSED",
    "REGIME_DUAL_PITCH_KPOOL_MAMBA",
    "REGIME_SHARED_SLAB_PRIVATE_AUX",
    "REGIME_UNIFORM_INDEPENDENT",
    "SLAB_BASE_ALIGNMENT",
    "VALID_POOL_REGIMES",
    "_ceil_div",
    "_draft_is_dcp_sharded",
    "_draft_page_bytes",
    "_draft_row_bytes_int",
    "_packed_mla_row_bytes",
    "_packed_mla_row_parts",
    "_state_page_bytes",
    "evaluate_hybrid_pool_receipt",
    "exact_element_bytes",
    "gbsa_cache_layout",
    "request_pool_page_pitches",
    "resolve_pool_allocator_contract",
    "separate_pool_page_pitches",
    "target_mla_page_bytes",
]
