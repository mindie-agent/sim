"""Cohesive 1:1 vLLM-Ascend DCP, PCP, and PCP+DCP computation-graph rewrites.

[Role]
  Single owner for Context Parallelism graph lowering across all KV-reading
  attention families (MLA, SFA/DSA, CSA/CSA2/SWA/HCA, GQA): universal Q gather,
  MTP dual-stream History/Current split, SFA index remap and compact KV gather,
  fast BF16 INT32 vs generic fused LSE AllToAll + PCP AllGather combine,
  V4/V4.1 PCP C2 ring-tail gather, PCP o_proj weight sharding, and paired
  packed-C8/FP8 QSFA execution.

[Invariants]
  * Useful attention FLOPs summed across participating ranks equal 100% of CP=1
    across every valid ``(dp, tp, pcp, dcp)`` factorization.
  * Every emitted collective carries an explicit non-empty ``dtype`` so typed
    output resolution never fails on compressed or index gathers.
  * Fast ``sfa_dcp_exchange`` (4-byte INT32-packed LSE) is selected strictly
    under power-of-two BF16 single-group conditions; all other cases use
    generic ``dcp_a2a_fused`` (8-byte BF16/FP16 or 4-byte FP32 LSE).
  * MTP ``q_len > 1`` dual-stream split launches CURRENT FIA on
    ``cp_decode_comm_stream`` with local heads only, overlapping History's
    pack + AllToAll [+ PCP AllGather] before a single-pass LSE combine.

[Timeouts & Locks]
  Pure O(requests) graph construction; zero I/O, locks, or device calls.

[Side-Effect Policy]
  Appends immutable ``OpSpec`` records to caller-owned lists only.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any, Sequence

from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.context_parallel import (
    CP_STRATEGY,
    layer_dcp_degree,
    layer_dcp_factors,
    owned_positions,
    pcp_rank_chunk_ranges,
)
from mindie_sim.catalog.ops import comm_op, vector_op
from mindie_sim.catalog.paired_sfa import SOURCE as PAIRED_SFA_SOURCE
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operators.attention.attention_pipeline import (
    TiledFlashAndPagedAttentionRecipe,
)
from mindie_sim.prediction.physical_recipe_base import (
    IO_HBM_CATEGORY,
    Phase,
    io_weight,
    tiled_phased_program,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

FP32_BYTES = 4
INT32_BYTES = 4
INT8_BYTES = 1
FAST_LSE_WIRE_BYTES = 4
GENERIC_LSE_WIRE_BYTES = 8
PACK_LSE_OPS = 4
MERGE_OPS = 3
REMAP_VECTOR_OPS = 6
UNPERMUTE_VECTOR_OPS = 2
RING_WRITE_VECTOR_OPS = 2
QUANT_VECTOR_OPS = 4
DEQUANT_VECTOR_OPS = 2
SOFTMAX_VECTOR_OPS_PER_PAIR = 5
V41_STATE_RING_ROWS = 32
QUANTIZED_ALLREDUCE_MIN_NUMEL = 16_777_216
QUANTIZED_ALLREDUCE_MIN_CHANNELS = 64
QUANTIZED_ALLREDUCE_MAX_WORLD = 8
MAX_HEADS = 128
SFA_MAX_GENERIC_QUERY_GROUP_SIZE = 64
SFA_PADDED_QUERY_GROUP_SIZE = 128
SINGLE_SFA_CALL = 1
PAIRED_SFA_CALLS = 2
_GROUP_META_KEYS = (
    "group_axis", "group_ranks", "local_tp_width",
    "local_tp_stride", "local_tp_stage", "local_tp_axis",
)
PACKED_C8_GROUP_SIZE = 128
PACKED_C8_LATENT = 512
PACKED_C8_ROPE = frozenset({0, 64})
FP8_FORMATS = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2"})
PACKED_CACHE_DTYPES = frozenset({"int8"})
PAIRED_FP8_QSFA_IMPLEMENTATION = "paired_fp8_qsfa_analytic"
DCP_LSE_MERGE_RECIPE = "dcp_lse_merge"
DCP_LSE_MERGE_EAGER_RECIPE = "dcp_lse_merge_eager"
MS_PER_SECOND = 1000.0
ATTENTION_UPDATE_SOURCE = "ops-transformer/attention/attention_update/op_kernel/decode_update.h"
ATTENTION_UPDATE_MAX_EFFECTIVE_SP = 16
ATTENTION_UPDATE_PIPELINE_SLOTS = 2
ATTENTION_UPDATE_TILE_CAP = 64
ATTENTION_UPDATE_LSE_EXP_OPS = 4
PACKED_KERNEL = "npu_kv_quant_sparse_flash_attention_vllm"
CP_DECODE_COMM_STREAM = "cp_decode_comm_stream"
CP_CHUNKED_PREFILL_STREAM = "cp_chunkedprefill_comm_stream"
CP_WEIGHT_STREAM = "cp_weight_stream"
CAUSAL_SPARSE_MODE = 3
QUANTIZED_ALLGATHER_DTYPES = frozenset({
    "int8", "fp8", "fp8_e4m3", "fp8_e5m2", "w8a8", "w4a8",
})


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _runtime_flag(runtime: Any, *keys: str) -> Any:
    if runtime is None:
        return None
    meta = dict(getattr(runtime, "metadata", ()) or ())
    extra = meta.get("additional_config")
    extra_map = dict(extra) if isinstance(extra, dict) else {}
    for source in (meta, extra_map):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def dcp_q_replicate_enabled(
    runtime: Any, *, pcp_factor: int, dp_width: int, sparse: bool,
) -> bool:
    """Opt-in Q-projection replication on dense MLA with single-axis TP DCP."""
    if pcp_factor != 1 or dp_width != 1 or sparse:
        return False
    return bool(_runtime_flag(runtime, "dcp_q_replicate", "enable_dcp_q_replicate"))


def pcp_o_proj_weight_sharding_enabled(runtime: Any) -> bool:
    """True when ``enable_pcp_o_proj_weight_sharding`` is enabled."""
    return bool(_runtime_flag(runtime, "enable_pcp_o_proj_weight_sharding"))


def pcp_quantized_allgather_enabled(runtime: Any) -> bool:
    """True when PR #22230 quantized W8A8 PCP AllGather is enabled."""
    return bool(_runtime_flag(
        runtime, "pcp_w8a8_quantized_allgather", "enable_pcp_w8a8_allgather",
    ))


def pcp_activation_gather_scale(runtime: Any, act_dtype: str) -> float:
    """Wire-byte scale factor (0.5 when quantized INT8/FP8 PCP AllGather is active)."""
    if not pcp_quantized_allgather_enabled(runtime):
        return 1.0
    return 0.5 if dtype_bytes(act_dtype) >= 2 else 1.0


def is_fast_sfa_dcp_exchange(
    *, pcp_factor: int, dp_width: int, tp_gather: int, local_heads: int,
    v_dim: int, act_dtype: str, defer_combine: bool,
) -> bool:
    """Exact ``sfa_dcp_exchange.py`` INT32-packed 4-byte LSE eligibility."""
    return (
        pcp_factor == 1
        and dp_width == 1
        and not defer_combine
        and str(act_dtype).lower() in {"bf16", "bfloat16"}
        and tp_gather > 1
        and _is_power_of_two(tp_gather)
        and _is_power_of_two(local_heads)
        and v_dim > 0
        and v_dim % 2 == 0
        and _is_power_of_two(v_dim // 2)
    )


def should_split_mtp_dcp(
    plan: Any, views: Sequence[Any], *, layer_dcp: int, sparse: bool,
) -> bool:
    """True for causal full-context dense MLA/GQA decode with ``q_len > 1`` and ``context > 0``."""
    if (layer_dcp <= 1 or sparse or bool(getattr(plan, "noncausal_attention", False))
            or int(getattr(plan, "window", 0) or 0) > 0):
        return False
    return any(
        getattr(view, "phase", "") in {"decode", "draft"}
        and int(getattr(view, "q_len", 0)) > 1
        and int(getattr(view, "context", 0)) > 0
        for view in views
    )


def _parse_bool_flag(value: Any) -> bool:
    """Parse explicit boolean flags without treating string ``'false'`` as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes", "on"}:
            return True
        if norm in {"false", "0", "no", "off", ""}:
            return False
    raise ModelingError(f"invalid_boolean_config_value:{value!r}")


def _extract_quantized_allreduce_dict(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    extra = metadata.get("additional_config")
    if isinstance(extra, Mapping) and isinstance(extra.get("quantized_all_reduce_config"), Mapping):
        return dict(extra["quantized_all_reduce_config"])
    raw = metadata.get("quantized_all_reduce_config")
    if isinstance(raw, Mapping):
        return dict(raw)
    return None


def validate_quantized_allreduce_runtime(
    metadata: Mapping[str, Any], *, arch: str, tp: int,
) -> None:
    """Validate explicit ``additional_config.quantized_all_reduce_config`` at parse time."""
    cfg = _extract_quantized_allreduce_dict(metadata)
    if cfg is None:
        return
    enabled = _parse_bool_flag(cfg.get("enable", False))
    if not enabled:
        return
    min_numel = int(cfg.get("min_numel", QUANTIZED_ALLREDUCE_MIN_NUMEL))
    if min_numel < 1:
        raise ModelingError(
            "additional_config.quantized_all_reduce_config.min_numel must be >= 1"
        )
    arch_norm = str(arch or "").strip().lower()
    if arch_norm != "a2":
        raise ModelingError(
            f"additional_config.quantized_all_reduce_config_unsupported_arch:{arch_norm}"
        )
    if int(tp) > QUANTIZED_ALLREDUCE_MAX_WORLD:
        raise ModelingError(
            f"additional_config.quantized_all_reduce_config_unsupported_tp:{int(tp)}"
        )


def _quantized_allreduce_config(runtime: Any) -> tuple[bool, int]:
    """Parse ``quantized_all_reduce_config`` (takes strict precedence) or opt-in flag."""
    if runtime is None:
        return False, QUANTIZED_ALLREDUCE_MIN_NUMEL
    meta = dict(getattr(runtime, "metadata", ()) or ())
    cfg = _extract_quantized_allreduce_dict(meta)
    if cfg is not None:
        if not _parse_bool_flag(cfg.get("enable", False)):
            return False, QUANTIZED_ALLREDUCE_MIN_NUMEL
        min_numel = int(cfg.get("min_numel", QUANTIZED_ALLREDUCE_MIN_NUMEL))
        if min_numel < 1:
            return False, min_numel
        return True, min_numel
    opt_in = _runtime_flag(runtime, "quantized_allreduce")
    if isinstance(opt_in, Mapping):
        if not _parse_bool_flag(opt_in.get("enable", False)):
            return False, QUANTIZED_ALLREDUCE_MIN_NUMEL
        min_numel = int(opt_in.get("min_numel", QUANTIZED_ALLREDUCE_MIN_NUMEL))
        return (min_numel >= 1), min_numel
    if opt_in is not None and _parse_bool_flag(opt_in):
        return True, QUANTIZED_ALLREDUCE_MIN_NUMEL
    return False, QUANTIZED_ALLREDUCE_MIN_NUMEL


def _allreduce_group_size(op: Any, op_meta: dict[str, Any], parallel: Any, bf16_bytes: float) -> int:
    """Resolve communicator participant count for an AllReduce op."""
    if op_meta.get("group_ranks"):
        return len(op_meta["group_ranks"])
    if op_meta.get("local_tp_width"):
        return int(op_meta["local_tp_width"])
    axis = str(op_meta.get("group_axis") or "tp").strip().lower()
    if axis == "tp" and getattr(parallel, "tp", None) is not None:
        return int(parallel.tp or 1)
    if axis == "ep" and getattr(parallel, "ep", None) is not None:
        return int(parallel.ep or 1)
    if axis == "pcp" and getattr(parallel, "pcp", None) is not None:
        return int(parallel.pcp or 1)
    if axis == "dp" and getattr(parallel, "dp", None) is not None:
        return int(parallel.dp or 1)
    if bf16_bytes > 0.0 and float(getattr(op, "link_bytes", 0.0) or 0.0) > 0.0:
        ratio = float(op.link_bytes) / (2.0 * bf16_bytes)
        if 0.0 < ratio < 1.0:
            return max(2, int(round(1.0 / (1.0 - ratio))))
    return int(getattr(parallel, "tp", 1) or 1)


def quantized_allreduce_eligible(
    runtime: Any, *, arch: str, tp: int, act_dtype: str, rows: int, hidden: int,
) -> bool:
    """Exact ``quantized_all_reduce.py`` eligibility: opt-in, A2, BF16, 2<=world<=8, >=min_numel."""
    if str(arch).strip().lower() != "a2":
        return False
    if not (1 < int(tp) <= QUANTIZED_ALLREDUCE_MAX_WORLD):
        return False
    if str(act_dtype).strip().lower() not in {"bf16", "bfloat16"}:
        return False
    if int(hidden) < QUANTIZED_ALLREDUCE_MIN_CHANNELS or int(rows) <= 0:
        return False
    enabled, min_numel = _quantized_allreduce_config(runtime)
    if not enabled or min_numel < 1:
        return False
    return int(rows) * int(hidden) >= min_numel


def rewrite_quantized_allreduce(ops: Sequence[Any], runtime: Any, source_dtype: str) -> list[Any]:
    """Rewrite eligible A2 BF16 AllReduces to ``quantized_all_reduce.py`` INT8 chain."""
    enabled, min_numel = _quantized_allreduce_config(runtime)
    if not enabled or min_numel < 1:
        return list(ops)
    from dataclasses import replace as _replace
    import math
    from mindie_sim.schema.operators import OpSpec

    meta = dict(getattr(runtime, "metadata", ()) or ())
    arch = str(meta.get("arch_family", "") or "").strip().lower()
    if arch != "a2":
        return list(ops)
    parallel = getattr(runtime, "parallel", None)
    out: list[Any] = []
    for op in ops:
        if op.family != "allreduce":
            out.append(op)
            continue
        raw_shape = tuple(int(d) for d in getattr(op, "local_shape", ()))
        if len(raw_shape) < 2 or any(d <= 0 for d in raw_shape):
            out.append(op)
            continue
        hidden = raw_shape[-1]
        rows = math.prod(raw_shape[:-1])
        numel = rows * hidden
        bf16_bytes = float(numel * 2)
        op_meta = dict(getattr(op, "metadata", ()) or ())
        group_axis = str(op_meta.get("group_axis") or "tp")
        group_size = _allreduce_group_size(op, op_meta, parallel, bf16_bytes)
        dtype_in = str(op.dtype or source_dtype or "bf16").strip().lower()
        if not quantized_allreduce_eligible(
            runtime, arch=arch, tp=group_size, act_dtype=dtype_in, rows=rows, hidden=hidden,
        ):
            out.append(op)
            continue
        ring_factor = 2.0 * float(group_size - 1) / float(group_size)
        amax_id = f"{op.op_id}.quant_ar_amax"
        sync_id = f"{op.op_id}.quant_ar_amax_sync"
        quant_id = f"{op.op_id}.quant_ar_quant"
        int8_id = f"{op.op_id}.quant_ar_int8_reduce"
        int8_bytes = float(numel)
        scale_bytes = float(hidden * FAST_LSE_WIRE_BYTES)
        base_kw = dict(
            phase=op.phase, stage=op.stage, layer=op.layer, requests=op.requests,
            stream=op.stream,
        )
        row_shape = (rows, hidden)
        vec_shape = (hidden,)
        out.append(OpSpec(
            op_id=amax_id, family="vector", recipe="quant_allreduce_amax",
            deps=op.deps, semantic_shape=row_shape, local_shape=row_shape,
            padded_shape=row_shape, launch_shape=row_shape, dtype="fp32",
            vector_ops=float(2 * numel), mte2_bytes=bf16_bytes, mte3_bytes=scale_bytes,
            hbm_bytes=bf16_bytes + scale_bytes,
            metadata=(("quantized_allreduce", True),), **base_kw,
        ))
        sync_group_meta = tuple(
            (key, op_meta[key]) for key in _GROUP_META_KEYS if key in op_meta
        )
        if not any(key == "group_axis" for key, _ in sync_group_meta):
            sync_group_meta = (("group_axis", group_axis), *sync_group_meta)
        out.append(OpSpec(
            op_id=sync_id, family="allreduce", recipe="quant_allreduce_amax_sync",
            deps=(amax_id,), semantic_shape=vec_shape, local_shape=vec_shape,
            padded_shape=vec_shape, launch_shape=vec_shape, dtype="fp32",
            link_bytes=scale_bytes * ring_factor,
            metadata=(*sync_group_meta, ("reduce_op", "max"),
                      ("quantized_allreduce", True)), **base_kw,
        ))
        out.append(OpSpec(
            op_id=quant_id, family="quantize", recipe="quant_allreduce_quantize",
            deps=(sync_id,), semantic_shape=row_shape, local_shape=row_shape,
            padded_shape=row_shape, launch_shape=row_shape, dtype="int8",
            vector_ops=float(2 * numel), mte2_bytes=bf16_bytes + scale_bytes,
            mte3_bytes=int8_bytes, hbm_bytes=bf16_bytes + scale_bytes + int8_bytes,
            metadata=(("quantized_allreduce", True),), **base_kw,
        ))
        out.append(_replace(
            op, op_id=int8_id, deps=(quant_id,), dtype="int8",
            semantic_shape=row_shape, local_shape=row_shape,
            padded_shape=row_shape, launch_shape=row_shape,
            link_bytes=float(op.link_bytes) * 0.5,
            metadata=(*op.metadata, ("quantized_allreduce", True),
                      ("quantized_allreduce_orig_bytes", float(op.link_bytes))),
        ))
        out.append(OpSpec(
            op_id=op.op_id, family="vector", recipe="quant_allreduce_dequantize",
            deps=(int8_id,), semantic_shape=op.semantic_shape, local_shape=raw_shape,
            padded_shape=raw_shape, launch_shape=raw_shape, dtype="bf16",
            vector_ops=float(2 * numel), mte2_bytes=int8_bytes + scale_bytes,
            mte3_bytes=bf16_bytes, hbm_bytes=int8_bytes + scale_bytes + bf16_bytes,
            metadata=(("quantized_allreduce", True),
                      ("source", "quantized_all_reduce.py:npu_anti_quant")), **base_kw,
        ))
    return out


def resolve_cp_rank_candidates(
    parallel: Any, *, eff_cp: int, pcp_factor: int, tp_gather: int, dp_width: int,
    local_dp_pos: int = 0, physical_rank: int | None = None,
    explicit_dp_cohort: bool = False, explicit_physical_rank: bool = False,
) -> tuple[int, ...]:
    """Candidate DCP rank indices owned by this lowered graph (exact rank when pinned)."""
    cp = max(1, int(eff_cp))
    if cp <= 1 or pcp_factor * tp_gather * dp_width != cp:
        return tuple(range(cp))
    p_tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    p_pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    if physical_rank is not None:
        within_stage = int(physical_rank) % (p_tp * p_pcp)
        tp_local = (within_stage % p_tp) % max(1, tp_gather)
        pcp_local = (within_stage // p_tp) % max(1, pcp_factor)
    else:
        tp_local = pcp_local = 0
    dp_vals = ((int(local_dp_pos) % dp_width,)
               if dp_width > 1 and (explicit_dp_cohort or explicit_physical_rank)
               else tuple(range(dp_width)))
    pcp_vals = ((pcp_local,)
                if pcp_factor > 1 and explicit_physical_rank and physical_rank is not None
                else tuple(range(pcp_factor)))
    tp_vals = ((tp_local,)
               if tp_gather > 1 and explicit_physical_rank and physical_rank is not None
               else tuple(range(tp_gather)))
    if dp_width > 1:
        return tuple(
            (dp_i * pcp_factor + pcp_i) * tp_gather + tp_i
            for dp_i in dp_vals for pcp_i in pcp_vals for tp_i in tp_vals
        )
    return tuple(
        tp_i * pcp_factor + pcp_i
        for tp_i in tp_vals for pcp_i in pcp_vals
    )


def pcp_uniform_peer_rows(
    views: Sequence[Any], pcp: int, *, local_rows: int = 0,
) -> tuple[int, int]:
    """Return ``(common_peer_rows, global_rows)`` matching ``pcp_manager.py:282``."""
    degree = max(1, int(pcp))
    prefill = [v for v in views if getattr(v, "phase", "") == "prefill"]
    if degree <= 1 or not prefill:
        total = sum(int(getattr(v, "q_len", 0) or 0) for v in views)
        return max(0, int(local_rows)), max(total, int(local_rows))
    global_pf = sum(
        int(getattr(v, "global_q_len", 0) or getattr(v, "q_len", 0) or 0)
        for v in prefill
    )
    decode_rows = sum(
        int(getattr(v, "q_len", 0) or 0)
        for v in views if getattr(v, "phase", "") != "prefill"
    )
    per_rank = [
        decode_rows + sum(
            sum(hi - lo for lo, hi in pcp_rank_chunk_ranges(
                int(getattr(v, "global_q_len", 0) or getattr(v, "q_len", 0) or 0),
                degree, r))
            for v in prefill
        )
        for r in range(degree)
    ]
    common = max(max(per_rank, default=0), int(local_rows))
    return common, global_pf + decode_rows


def active_layer_dcp_factors(
    plan: Any, parallel: Any, phase: str, *, sharded_indexer: bool = False,
) -> tuple[int, int, int]:
    """Effective ``(pcp_factor, layer_tp_gather, dp_width)`` for one layer and phase."""
    if str(phase) == "prefill" and not sharded_indexer:
        return (1, 1, 1)
    return layer_dcp_factors(plan, parallel)


def active_layer_dcp(
    plan: Any, parallel: Any, phase: str, *, sharded_indexer: bool = False,
) -> int:
    """Effective DCP degree for one layer (prefill active only under sharded indexer)."""
    if str(phase) == "prefill" and not sharded_indexer:
        return 1
    return layer_dcp_degree(plan, parallel)


def resolve_dcp_cohort_decode(
    views: Sequence[Any], *, dp_width: int,
    cohort_views_by_member: Sequence[Sequence[Any]] | None,
    local_dp_pos: int,
) -> tuple[tuple[Any, ...], tuple[int, ...], int]:
    """Return ``(cohort_decode_views, peer_decode_rows, local_dp_pos)`` for cross-DP DCP."""
    width = max(1, int(dp_width))
    if width <= 1:
        local_dec = tuple(v for v in views if getattr(v, "phase", "") != "prefill")
        return local_dec, (sum(int(v.q_len) for v in local_dec),), 0
    if cohort_views_by_member is not None and len(cohort_views_by_member) == width:
        per_member = tuple(
            tuple(v for v in grp if getattr(v, "phase", "") != "prefill")
            for grp in cohort_views_by_member
        )
        pos = int(local_dp_pos) if 0 <= int(local_dp_pos) < width else 0
    else:
        local_dec = tuple(v for v in views if getattr(v, "phase", "") != "prefill")
        per_member = tuple(local_dec for _ in range(width))
        pos = 0
    union = tuple(v for grp in per_member for v in grp)
    peer_rows = tuple(sum(int(v.q_len) for v in grp) for grp in per_member)
    return union, peer_rows, pos


def emit_dcp_query_gather(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, deps: tuple[str, ...],
    rows: int, local_heads: int, head_width: int, act_dtype: str,
    tp_gather: int, dp_width: int, pack_inputs: bool = False,
    q_replicate: bool = False, stream: str = "",
    peer_rows: Sequence[int] | None = None,
    local_dp_pos: int = 0, pcp_factor: int = 1,
) -> tuple[str, int, int]:
    """Emit universal Q gather across ``dcp_tp`` (heads) and ``dcp_dp`` (tokens)."""
    last = deps[0]
    norm_peers = (tuple(max(0, int(r)) for r in peer_rows)
                  if peer_rows is not None else (max(0, int(rows)),) * max(1, int(dp_width)))
    total_rows = sum(norm_peers)
    max_peer_rows = max(norm_peers, default=0)
    if q_replicate:
        return last, local_heads * tp_gather, total_rows
    if tp_gather <= 1 and dp_width <= 1:
        return last, local_heads, max(0, int(rows))
    elem_bytes = dtype_bytes(act_dtype)
    eff_heads = local_heads
    eff_rows = max(0, int(rows))
    cur_deps = deps
    stream_meta = (("stream", stream),) if stream else ()
    if tp_gather > 1:
        if rows > 0:
            if pack_inputs:
                size = float(rows * local_heads * head_width) * elem_bytes
                pack_op = OpSpec(
                    op_id=f"{prefix}.dcp_query_pack", deps=cur_deps,
                    stage=stage, layer=layer, requests=requests, phase=mode,
                    family="elementwise", recipe="dcp_query_pack",
                    semantic_shape=(rows, local_heads, head_width),
                    local_shape=(rows, local_heads, head_width),
                    padded_shape=(rows, local_heads, head_width),
                    launch_shape=(rows, local_heads, head_width),
                    hbm_bytes=2.0 * size, mte2_bytes=size, mte3_bytes=size,
                    dtype=act_dtype, stream=stream, metadata=stream_meta,
                )
                ops.append(pack_op)
                cur_deps = (pack_op.op_id,)
            wire = float(rows * local_heads * head_width) * elem_bytes * float(tp_gather - 1)
            eff_heads = local_heads * tp_gather
            ag = comm_op(
                op_id=f"{prefix}.dcp_query_allgather", deps=cur_deps,
                stage=stage, layer=layer, requests=requests, family="allgather",
                phase=mode, recipe="dcp_query_allgather", link_bytes=wire,
                shape=(rows, eff_heads, head_width),
                metadata=(("group_axis", "dcp_tp"), ("gathered_heads", eff_heads),
                          ("dcp_tp_gather", tp_gather),
                          ("dcp_pcp_factor", int(pcp_factor)),
                          ("dcp_dp_width", int(dp_width))) + stream_meta,
            )
            ag = replace(ag, dtype=act_dtype, stream=stream)
            ops.append(ag)
            last = ag.op_id
            cur_deps = (last,)
        else:
            eff_heads = local_heads * tp_gather
    if dp_width > 1:
        wire = float(max_peer_rows * eff_heads * head_width) * elem_bytes * float(dp_width - 1)
        eff_rows = total_rows
        padded_rows = max_peer_rows * dp_width
        dp_ag = comm_op(
            op_id=f"{prefix}.dcp_dp_query_allgather", deps=cur_deps,
            stage=stage, layer=layer, requests=requests, family="allgather",
            phase=mode, recipe="dcp_dp_query_allgather", link_bytes=wire,
            shape=(eff_rows, eff_heads, head_width),
            metadata=(("group_axis", "dcp_dp"), ("gathered_tokens", eff_rows),
                      ("dcp_dp_width", int(dp_width)),
                      ("dcp_tp_gather", int(tp_gather)),
                      ("dcp_pcp_factor", int(pcp_factor)),
                      ("dcp_peer_rows", norm_peers),
                      ("dcp_padded_rows_per_peer", int(max_peer_rows)),
                      ("dcp_local_dp_pos", int(local_dp_pos))) + stream_meta,
        )
        dp_ag = replace(
            dp_ag, dtype=act_dtype, stream=stream,
            padded_shape=(padded_rows, eff_heads, head_width),
            launch_shape=(padded_rows, eff_heads, head_width),
        )
        ops.append(dp_ag)
        last = dp_ag.op_id
    return last, eff_heads, eff_rows


def emit_sfa_index_remap(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, topk_dep: str,
    rows: int, topk: int, layer_dcp: int, interleave: int,
) -> str:
    """Triton ``sparse_index_remap`` translating global Top-K indices to local DCP slots."""
    elems = max(1, int(rows)) * max(1, int(topk))
    nbytes = float(elems * INT32_BYTES)
    remap = vector_op(
        op_id=f"{prefix}.sparse_index_remap", deps=(topk_dep,),
        stage=stage, layer=layer, requests=requests, elements=elems,
        ops_per_element=REMAP_VECTOR_OPS, dtype="int32", family="elementwise",
        phase=mode, recipe="sparse_index_remap", shape=(rows, topk),
        read_bytes_override=nbytes, write_bytes_override=nbytes,
        metadata=(("dcp_degree", int(layer_dcp)), ("cp_interleave", int(interleave)),
                  ("kernel", "remap_sparse_indices_triton")),
    )
    ops.append(remap)
    return remap.op_id


def emit_sfa_prefill_kv_gather(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, store_dep: str,
    prefill_context_rows: int, kv_row_bytes: float, layer_dcp: int,
    kv_dtype: str, recipe: str = "sfa_dcp_prefill_kv_gather",
) -> str:
    """Async compact/history KV block AllGather for dense chunked MLA or replicated SFA prefill."""
    if prefill_context_rows <= 0 or layer_dcp <= 1:
        return store_dep
    wire = float(prefill_context_rows) * float(kv_row_bytes) * float(layer_dcp - 1) / float(layer_dcp)
    gather = comm_op(
        op_id=f"{prefix}.{recipe}", deps=(store_dep,),
        stage=stage, layer=layer, requests=requests, family="allgather",
        phase=mode, recipe=recipe, link_bytes=wire,
        shape=(prefill_context_rows, int(round(kv_row_bytes))),
        metadata=(("group_axis", "cp"), ("dcp_degree", int(layer_dcp)),
                  ("stream", CP_DECODE_COMM_STREAM),
                  ("sfa_prefill_compact_kv", recipe == "sfa_dcp_prefill_kv_gather")),
    )
    gather = replace(gather, dtype=str(kv_dtype or "bf16"), stream=CP_DECODE_COMM_STREAM)
    ops.append(gather)
    return gather.op_id


def emit_mtp_current_attention(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, deps: tuple[str, ...],
    views: Sequence[Any], local_heads: int, qk_dim: int, v_dim: int,
    kv_row_bytes: float, act_dtype: str,
    stream: str = CP_DECODE_COMM_STREAM, allow_q1: bool = False, window: int = 0,
) -> str:
    """CURRENT FIA on ``stream`` (local heads, disjoint causal current-token pairs)."""
    min_q = 1 if allow_q1 else 2
    q_tokens = sum(int(v.q_len) for v in views if int(v.q_len) >= min_q and int(v.context) > 0)
    tri_pairs = 0.0
    for v in views:
        qv = int(v.q_len)
        if qv < min_q or int(v.context) <= 0:
            continue
        ranges = tuple(getattr(v, "query_ranges", ()) or ())
        if ranges and allow_q1:
            for lo_q, hi_q in ranges:
                for pos in range(int(lo_q), int(hi_q)):
                    attend = pos + 1
                    tri_pairs += float(min(attend, window) if window > 0 else attend)
        elif window > 0 and window < qv:
            tri_pairs += float(window * (window + 1) // 2 + (qv - window) * window)
        else:
            tri_pairs += float(qv * (qv + 1) / 2.0)
    elem_bytes = dtype_bytes(act_dtype)
    q_read = float(q_tokens * local_heads * qk_dim) * elem_bytes
    kv_read = float(q_tokens) * float(kv_row_bytes)
    out_write = float(q_tokens * local_heads) * (float(v_dim) * elem_bytes + float(FP32_BYTES))
    stream_meta = (("stream", stream),) if stream else ()
    current = OpSpec(
        op_id=f"{prefix}.dcp_mtp_current_attention", deps=deps,
        stage=stage, layer=layer, requests=requests, phase=mode,
        family="attention", recipe="dcp_mtp_current_attention",
        semantic_shape=(q_tokens, q_tokens, local_heads, qk_dim, v_dim),
        local_shape=(q_tokens, q_tokens, local_heads, qk_dim, v_dim),
        padded_shape=(q_tokens, q_tokens, local_heads, qk_dim, v_dim),
        launch_shape=(q_tokens, q_tokens, local_heads, qk_dim, v_dim),
        flops=float(2.0 * tri_pairs * local_heads * (qk_dim + v_dim)),
        vector_ops=float(SOFTMAX_VECTOR_OPS_PER_PAIR * tri_pairs * local_heads),
        hbm_bytes=q_read + kv_read + out_write,
        mte2_bytes=q_read + kv_read, mte3_bytes=out_write,
        dtype=act_dtype, stream=stream,
        metadata=stream_meta + (("sparse_mode", CAUSAL_SPARSE_MODE),
                                ("dcp_mtp_split", "current"),
                                ("local_heads", int(local_heads)),
                                ("logical_pairs", float(tri_pairs))),
    )
    ops.append(current)
    return current.op_id


def emit_dcp_output_exchange_combine(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, deps: tuple[str, ...],
    rows: int, local_heads: int, v_dim: int, act_dtype: str,
    pcp_factor: int, tp_gather: int, dp_width: int,
    local_current_dep: str | None = None,
    lse_in_bytes: int = FP32_BYTES,
    meta_extra: tuple[tuple[str, Any], ...] = (),
    peer_rows: Sequence[int] | None = None,
    local_dp_pos: int = 0,
    stream: str = "",
    merge_rows: int | None = None,
) -> str:
    """Emit pack -> AllToAll (tp*dp) -> AllGather (pcp) -> fused LSE combine."""
    dcp_total = max(1, int(pcp_factor) * int(tp_gather) * int(dp_width))
    if dcp_total <= 1:
        return deps[0]
    a2a_degree = max(1, int(tp_gather) * int(dp_width))
    norm_peers = (tuple(max(0, int(r)) for r in peer_rows)
                  if peer_rows is not None
                  else (max(0, int(rows)),) * max(1, int(dp_width)))
    local_rows = (norm_peers[local_dp_pos]
                  if 0 <= int(local_dp_pos) < len(norm_peers) else max(0, int(rows)))
    total_rows = sum(norm_peers)
    ragged_dp = int(dp_width) > 1 and len(set(norm_peers)) > 1
    elem_bytes = dtype_bytes(act_dtype)
    defer_combine = local_current_dep is not None
    fast = is_fast_sfa_dcp_exchange(
        pcp_factor=pcp_factor, dp_width=dp_width, tp_gather=tp_gather,
        local_heads=local_heads, v_dim=v_dim, act_dtype=act_dtype,
        defer_combine=defer_combine,
    )
    lse_wire = FAST_LSE_WIRE_BYTES if (fast or int(elem_bytes) == FP32_BYTES) else GENERIC_LSE_WIRE_BYTES
    row_packed_bytes = float(v_dim) * elem_bytes + float(lse_wire)
    packed_rows = max(1, total_rows)
    packed_heads = max(1, int(local_heads)) * max(1, int(tp_gather))
    pack_read = float(packed_rows * packed_heads) * (float(v_dim) * elem_bytes + float(lse_in_bytes))
    pack_write = float(packed_rows * packed_heads) * row_packed_bytes
    stream_meta = (("stream", stream),) if stream else ()
    pack_op = OpSpec(
        op_id=f"{prefix}.dcp_output_pack", deps=deps,
        stage=stage, layer=layer, requests=requests, phase=mode,
        family="elementwise", recipe="dcp_output_pack",
        semantic_shape=(packed_rows, packed_heads, v_dim),
        local_shape=(packed_rows, packed_heads, v_dim),
        padded_shape=(packed_rows, packed_heads, v_dim),
        launch_shape=(packed_rows, packed_heads, v_dim),
        vector_ops=float(packed_rows * packed_heads * PACK_LSE_OPS),
        hbm_bytes=pack_read + pack_write, mte2_bytes=pack_read, mte3_bytes=pack_write,
        dtype=act_dtype, stream=stream,
        metadata=(("dcp_exchange_path", "sfa_dcp_exchange_int32" if fast else "dcp_a2a_fused"),
                  ("lse_wire_bytes", int(lse_wire))) + stream_meta,
    )
    ops.append(pack_op)
    cur_dep = pack_op.op_id
    if a2a_degree > 1:
        if ragged_dp:
            matrix = [
                [
                    0.0 if src == dst
                    else float(norm_peers[dst // max(1, int(tp_gather))] * local_heads) * row_packed_bytes
                    for dst in range(a2a_degree)
                ]
                for src in range(a2a_degree)
            ]
            src_idx = int(local_dp_pos) * max(1, int(tp_gather))
            a2a_wire = float(sum(matrix[src_idx]))
            a2a_family = "alltoallv"
            peer_meta: tuple[tuple[str, Any], ...] = (
                ("peer_traffic", {"kind": "matrix", "payload_bytes": matrix}),
            )
        else:
            a2a_wire = float(local_rows * local_heads) * row_packed_bytes * float(a2a_degree - 1)
            a2a_family = "alltoall"
            peer_meta = ()
        a2a_op = comm_op(
            op_id=f"{prefix}.dcp_output_alltoall", deps=(cur_dep,),
            stage=stage, layer=layer, requests=requests, family=a2a_family,
            phase=mode, recipe="dcp_output_alltoall", link_bytes=a2a_wire,
            shape=(max(0, local_rows), local_heads, v_dim),
            metadata=(("group_axis", "dcp_a2a"),
                      ("source", "sfa_dcp_exchange" if fast else "dcp_a2a_fused"),
                      ("lse_wire_bytes", int(lse_wire)),
                      ("dcp_a2a_degree", int(a2a_degree)),
                      ("dcp_tp_gather", int(tp_gather)),
                      ("dcp_dp_width", int(dp_width)),
                      ("dcp_pcp_factor", int(pcp_factor)),
                      ("dcp_peer_rows", norm_peers)) + peer_meta + stream_meta,
        )
        a2a_op = replace(a2a_op, dtype=act_dtype, stream=stream)
        ops.append(a2a_op)
        cur_dep = a2a_op.op_id
    if pcp_factor > 1:
        pcp_wire = float(local_rows * local_heads * a2a_degree) * row_packed_bytes * float(pcp_factor - 1)
        pcp_ag = comm_op(
            op_id=f"{prefix}.dcp_pcp_allgather", deps=(cur_dep,),
            stage=stage, layer=layer, requests=requests, family="allgather",
            phase=mode, recipe="dcp_pcp_allgather", link_bytes=pcp_wire,
            shape=(pcp_factor, a2a_degree, max(0, local_rows), local_heads, v_dim),
            metadata=(("group_axis", "dcp_pcp"), ("pcp_factor", int(pcp_factor)),
                      ("dcp_pcp_factor", int(pcp_factor)),
                      ("dcp_tp_gather", int(tp_gather)),
                      ("dcp_dp_width", int(dp_width)),
                      ("lse_wire_bytes", int(lse_wire))) + stream_meta,
        )
        pcp_ag = replace(pcp_ag, dtype=act_dtype, stream=stream)
        ops.append(pcp_ag)
        cur_dep = pcp_ag.op_id
    out_rows = local_rows if merge_rows is None else max(0, int(merge_rows))
    if out_rows <= 0:
        return cur_dep
    merge_deps = (cur_dep, local_current_dep) if local_current_dep is not None else (cur_dep,)
    partial_count = dcp_total + (1 if local_current_dep is not None else 0)
    elements = float(out_rows * local_heads * v_dim)
    merge_read = float(dcp_total * out_rows * local_heads) * row_packed_bytes
    if local_current_dep is not None:
        merge_read += float(out_rows * local_heads) * (float(v_dim) * elem_bytes + float(FP32_BYTES))
    merge_write = elements * elem_bytes
    merge_op = OpSpec(
        op_id=f"{prefix}.dcp_lse_merge", deps=merge_deps,
        stage=stage, layer=layer, requests=requests, phase=mode,
        family="attention_merge", recipe="dcp_lse_merge",
        semantic_shape=(out_rows, local_heads, v_dim),
        local_shape=(out_rows, local_heads, v_dim),
        padded_shape=(out_rows, local_heads, v_dim),
        launch_shape=(out_rows, local_heads, v_dim),
        vector_ops=float(MERGE_OPS) * elements * float(partial_count),
        hbm_bytes=merge_read + merge_write, mte2_bytes=merge_read, mte3_bytes=merge_write,
        dtype=act_dtype,
        metadata=(
            ("cp", int(dcp_total)),
            ("cp_strategy", CP_STRATEGY),
            ("dcp_exchange_path", "sfa_dcp_exchange_int32" if fast else "dcp_a2a_fused"),
            ("dcp_factors", (int(pcp_factor), int(tp_gather), int(dp_width))),
            ("lse_wire_bytes", int(lse_wire)),
            ("partial_count", int(partial_count)),
            ("partial_reduction", "logsumexp"),
            ("merge_dtype", "fp32"),
            ("defer_combine", bool(defer_combine)),
        ) + tuple(meta_extra),
    )
    ops.append(merge_op)
    return merge_op.op_id


def emit_v41_pcp_c2_ring_gather(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, deps: tuple[str, ...],
    views: Sequence[Any], pcp: int, compressor_width: int, head_dim: int,
    min_send_rows: int = 0,
) -> str:
    """``dsa_v41_cp.py`` V41PCPC2Metadata per-peer deduplicated segment-end + 32-row tail gather."""
    if pcp <= 1 or not any(getattr(v, "phase", "") == "prefill" for v in views):
        return deps[0]
    peer_counts = [0] * pcp
    for v in views:
        gq = int(getattr(v, "global_q_len", 0) or v.q_len)
        if gq <= 0:
            continue
        if getattr(v, "phase", "") == "prefill":
            tail_start = max(0, gq - V41_STATE_RING_ROWS)
            for r in range(pcp):
                ranges = pcp_rank_chunk_ranges(gq, pcp, r)
                unique_rows = {
                    idx for a, b_end in ranges if b_end > a
                    for idx in (*range(max(a, tail_start), b_end), b_end - 1)
                }
                peer_counts[r] += len(unique_rows)
        else:
            # Replicated decode rows participate on every PCP peer (dsa_v41_cp.py:160-166).
            decode_tail = min(gq, V41_STATE_RING_ROWS)
            for r in range(pcp):
                peer_counts[r] += decode_tail
    send_rows = max(1, max(peer_counts, default=0), max(0, int(min_send_rows)))
    row_elems = 2 * max(1, int(compressor_width)) * max(1, int(head_dim))
    wire = float(send_rows * row_elems * FP32_BYTES) * float(pcp - 1)
    ag = comm_op(
        op_id=f"{prefix}.pcp_c2_ring_tail_gather", deps=deps,
        stage=stage, layer=layer, requests=requests, family="allgather",
        phase=mode, recipe="pcp_c2_ring_tail_gather", link_bytes=wire,
        shape=(send_rows * pcp, row_elems),
        metadata=(("group_axis", "pcp"), ("pcp_c2_ring_rows", V41_STATE_RING_ROWS),
                  ("pcp_c2_peer_send_counts", tuple(peer_counts)),
                  ("pcp_c2_send_rows", int(send_rows)),
                  ("source", "dsa_v41_cp.V41PCPC2Metadata")),
    )
    ag = replace(ag, dtype="fp32")
    ops.append(ag)
    return ag.op_id


def emit_v41_pcp_c2_ring_write(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, deps: tuple[str, ...],
    views: Sequence[Any], compressor_width: int, head_dim: int,
    num_ring_rows: int = 0,
) -> str:
    """``dsa_v41_cp.py:698`` post-pool persistent ring update over global request tails."""
    genuine_writes = sum(
        min(max(0, int(getattr(v, "global_q_len", 0) or v.q_len)), V41_STATE_RING_ROWS)
        for v in views
    )
    ring_rows = max(0, int(num_ring_rows)) or genuine_writes
    if ring_rows <= 0:
        return deps[-1]
    row_elems = 2 * max(1, int(compressor_width)) * max(1, int(head_dim))
    ring_write = vector_op(
        op_id=f"{prefix}.pcp_c2_ring_write", deps=deps,
        stage=stage, layer=layer, requests=requests,
        elements=ring_rows * row_elems, ops_per_element=RING_WRITE_VECTOR_OPS,
        dtype="fp32", family="elementwise", phase=mode,
        recipe="pcp_c2_ring_write", shape=(ring_rows, row_elems),
        metadata=(("pcp_c2_ring_write_rows", int(ring_rows)),
                  ("pcp_c2_genuine_ring_writes", int(genuine_writes)),
                  ("source", "dsa_v41_cp.write_pcp_ring_rows")),
    )
    ops.append(ring_write)
    return ring_write.op_id


def emit_pcp_restore_unpermute(
    ops: list[OpSpec], *, prefix: str, name: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, dep: str,
    global_rows: int, row_bytes: float, dtype: str, stream: str = "",
    meta_extra: tuple[tuple[str, Any], ...] = (),
) -> str:
    """``index_select(hidden_restore_idx)`` after PCP DualChunkSwap AllGather."""
    if global_rows <= 0:
        return dep
    eb = max(1.0, dtype_bytes(dtype))
    elems = max(1, int(round(float(global_rows) * float(row_bytes) / eb)))
    nbytes = float(global_rows) * float(row_bytes)
    stream_meta = (("stream", stream),) if stream else ()
    restore = vector_op(
        op_id=f"{prefix}.{name}", deps=(dep,), stage=stage, layer=layer,
        requests=requests, elements=elems, ops_per_element=UNPERMUTE_VECTOR_OPS,
        dtype=str(dtype or "bf16"), family="elementwise", phase=mode,
        recipe=name, shape=(global_rows, max(1, int(round(row_bytes / eb)))),
        read_bytes_override=nbytes, write_bytes_override=nbytes,
        metadata=(("pcp_restore_idx", True),) + tuple(meta_extra) + stream_meta,
    )
    ops.append(replace(restore, stream=stream))
    return restore.op_id


def emit_pcp_hidden_gather_restore(
    ops: list[OpSpec], *, prefix: str, gather_name: str, restore_name: str,
    stage: int, layer: int, requests: tuple[str, ...], mode: str, dep: str,
    local_rows: int, global_rows: int, hidden: int, pcp: int, act_dtype: str,
    runtime: Any = None, source: str = "", stream: str = "",
    restore: bool = True, per_peer_rows: int | None = None,
    restore_meta: tuple[tuple[str, Any], ...] = (),
) -> str:
    """PCP hidden-state AllGather + optional restore, with explicit INT8 quant/dequant if enabled."""
    if pcp <= 1 or global_rows <= 0:
        return dep
    per_peer = (max(1, int(per_peer_rows)) if per_peer_rows is not None
                else max(int(local_rows), (int(global_rows) + int(pcp) - 1) // int(pcp)))
    elem_bytes = dtype_bytes(act_dtype)
    quantized = (pcp_quantized_allgather_enabled(runtime) and elem_bytes >= 2)
    cur = dep
    src_meta = (("source", source),) if source else ()
    stream_meta = (("stream", stream),) if stream else ()
    if quantized:
        in_b = float(per_peer * hidden) * elem_bytes
        out_b = float(per_peer * hidden * INT8_BYTES + per_peer * FP32_BYTES)
        q_op = vector_op(
            op_id=f"{prefix}.{gather_name}_quant", deps=(cur,),
            stage=stage, layer=layer, requests=requests,
            elements=per_peer * hidden, ops_per_element=QUANT_VECTOR_OPS,
            dtype="int8", family="quantize", phase=mode,
            recipe=f"{gather_name}_quant", shape=(per_peer, hidden),
            read_bytes_override=in_b, write_bytes_override=out_b,
            metadata=(("input_format", act_dtype), ("output_format", "int8"),
                      ("scale_dtype", "fp32"), ("scale_bytes", float(per_peer * FP32_BYTES)))
                     + stream_meta,
        )
        ops.append(replace(q_op, stream=stream))
        cur = q_op.op_id
        wire = out_b * float(pcp - 1)
        wire_dtype = "int8"
        ag_meta = (("group_axis", "pcp"), ("pcp_quantized_wire", True),
                   ("scale_bytes_per_peer", float(per_peer * FP32_BYTES))) + src_meta + stream_meta
    else:
        wire = float(per_peer * hidden) * elem_bytes * float(pcp - 1)
        wire_dtype = str(act_dtype or "bf16")
        ag_meta = (("group_axis", "pcp"),) + src_meta + stream_meta
    ag = comm_op(
        op_id=f"{prefix}.{gather_name}", deps=(cur,), stage=stage, layer=layer,
        requests=requests, family="allgather", phase=mode, recipe=gather_name,
        link_bytes=wire, shape=(per_peer * pcp, hidden), metadata=ag_meta,
    )
    ag = replace(ag, dtype=wire_dtype, stream=stream)
    ops.append(ag)
    cur = ag.op_id
    if quantized:
        g_in = float(per_peer * pcp * hidden * INT8_BYTES + per_peer * pcp * FP32_BYTES)
        g_out = float(per_peer * pcp * hidden) * elem_bytes
        dq_op = vector_op(
            op_id=f"{prefix}.{gather_name}_dequant", deps=(cur,),
            stage=stage, layer=layer, requests=requests,
            elements=per_peer * pcp * hidden, ops_per_element=DEQUANT_VECTOR_OPS,
            dtype=str(act_dtype or "bf16"), family="vector", phase=mode,
            recipe=f"{gather_name}_dequant", shape=(per_peer * pcp, hidden),
            read_bytes_override=g_in, write_bytes_override=g_out,
            metadata=(("input_format", "int8"), ("output_format", act_dtype),
                      ("scale_dtype", "fp32")) + stream_meta,
        )
        ops.append(replace(dq_op, stream=stream))
        cur = dq_op.op_id
    if restore:
        cur = emit_pcp_restore_unpermute(
            ops, prefix=prefix, name=restore_name, stage=stage, layer=layer,
            requests=requests, mode=mode, dep=cur, global_rows=global_rows,
            row_bytes=float(hidden) * elem_bytes, dtype=act_dtype, stream=stream,
            meta_extra=restore_meta,
        )
    return cur


def emit_pcp_o_proj_weight_allgather(
    ops: list[OpSpec], *, prefix: str, stage: int, layer: int,
    requests: tuple[str, ...], mode: str, deps: tuple[str, ...],
    weight_bytes: float, pcp: int, weight_dtype: str,
) -> str:
    """``weight_switch/o_proj.py`` overlapped PCP weight AllGather in prefill."""
    wire = float(weight_bytes) * float(pcp - 1) / float(pcp)
    ag = comm_op(
        op_id=f"{prefix}.pcp_o_proj_weight_allgather", deps=deps,
        stage=stage, layer=layer, requests=requests, family="allgather",
        phase=mode, recipe="pcp_o_proj_weight_allgather", link_bytes=wire,
        shape=(max(1, int(round(weight_bytes))),),
        metadata=(("group_axis", "pcp"), ("stream", CP_WEIGHT_STREAM),
                  ("source", "weight_switch/o_proj.py")),
    )
    ag = replace(ag, dtype=str(weight_dtype or "bf16"), stream=CP_WEIGHT_STREAM)
    ops.append(ag)
    return ag.op_id


def paired_sfa_calls(heads: int) -> tuple[int, ...]:
    """Declared sequential native calls of one gathered-head C8 SFA decode."""
    if heads < PAIRED_SFA_CALLS or heads % PAIRED_SFA_CALLS:
        raise ModelingError("paired_sfa_requires_even_gathered_heads")
    return (heads // PAIRED_SFA_CALLS,) * PAIRED_SFA_CALLS


def packed_sfa_geometry(
    meta: dict[str, Any], heads: int, plan: Any, views: Sequence[Any], cp: int,
    *, cache_dtype: str = "", cache_fmt: Any = None, runtime: Any = None,
) -> dict[str, Any] | None:
    """Packed-C8 geometry of the declared paired calls, or ``None``."""
    view = dict(meta.get("attention_tensor_view") or {})
    dtype = str(view.get("cache_dtype") or cache_dtype or "")
    if dtype not in PACKED_CACHE_DTYPES:
        return None
    if cache_fmt is not None and not (
            cache_fmt.dtype == "int8"
            and cache_fmt.packed_elements == 1
            and cache_fmt.group_size == PACKED_C8_GROUP_SIZE
            and cache_fmt.scale_dtype == "fp32"
            and cache_fmt.suffix_elements == int(plan.qk_rope_head_dim)
            and cache_fmt.suffix_dtype == "bf16"):
        return None
    if (heads <= 0 or heads > MAX_HEADS or plan.cache_row_elements <= 0
            or int(plan.kv_lora_rank) != PACKED_C8_LATENT
            or int(plan.qk_rope_head_dim) not in PACKED_C8_ROPE):
        return None
    degree = max(1, int(cp))
    sharded_topk = (max(1, -(-int(plan.index_topk) // degree))
                    if int(plan.index_topk) > 0 else 0)
    sequences = tuple((int(v.q_len), max(0, -(-int(v.context) // degree))) for v in views)
    extents = tuple(
        max(0, -(-min(int(v.prefix_tokens or 0), int(v.context)) // degree))
        for v in views)
    shared = getattr(runtime, "prefix_cache_mode", "private") == "shared" and any(extents)
    keys = tuple(str(v.prefix_cache_key or "") for v in views)
    # Upstream ``device_op.py:660-667`` executes packed INT8 QSFA in one single
    # kernel call over all gathered query heads (e.g. 96 heads on Kimi-K3.1).
    return {
        "sequences": sequences, "topk": sharded_topk, "compression_ratio": 1,
        "window": 0, "cp": 1, "heads": heads,
        "latent": int(plan.kv_lora_rank), "rope": int(plan.qk_rope_head_dim),
        "query_dtype": "bf16", "cache_dtype": "int8",
        **({"prefix_cache_mode": "shared", "prefix_tokens": extents,
            "prefix_cache_keys": keys} if shared else {"prefix_cache_mode": "private"}),
        "paired_calls": SINGLE_SFA_CALL, "dcp_peer_exchange": int(cp),
    }


def paired_sfa_attention_batch(
    b: Any, outputs: list[str], views: Sequence[Any], *, plan: Any, heads: int,
    local_heads: int, cp: int, rows: int, runtime: Any = None,
    pcp_factor: int = 1, tp_gather: int = 1, dp_width: int = 1,
    peer_rows: Sequence[int] | None = None, local_dp_pos: int = 0,
) -> list[str]:
    """Batch paired-SFA decode ops and emit fused DCP output exchange + LSE combine."""
    ids = set(outputs)
    members = [op for op in b.ops if op.op_id in ids]
    if not members:
        return outputs
    meta = dict(members[0].metadata)
    resolved = cache_format(b.model, "mla_kv")
    fp8 = resolved.dtype in FP8_FORMATS and plan.indexer
    packed = (packed_sfa_geometry(meta, heads, plan, views, cp,
                                  cache_dtype=resolved.dtype, cache_fmt=resolved,
                                  runtime=runtime)
              if plan.indexer else None)
    native_packed = fp8 or packed is not None
    if native_packed and (b.model.activation_dtype != "bf16"
                          or not 1 <= heads <= MAX_HEADS):
        raise ModelingError("fp8_qsfa_requires_bf16_queries_and_native_head_domain")
    meta.update({
        "program_kind": "paired_latent_attention",
        "attention_fidelity": "source_call_resource_envelope",
        "gathered_q_heads": heads, "local_projection_heads": local_heads,
        "source": {**PAIRED_SFA_SOURCE, "path": "vllm_ascend/device/device_op.py"},
        "native_kernel": PACKED_KERNEL if native_packed else "mla_attention",
        "native_layout": {"query": "TND", "kv": "PA_BSND"},
        "unmodeled_mechanisms": ("compiled_tiling", "internal_pipeline_overlap",
                                 "host_runtime_calibration"),
        "request_sequences": tuple((view.q_len, view.context) for view in views),
    })
    meta.pop("indexed_attention_geometry", None)
    if packed is not None:
        meta.update({"program_kind": "indexed_attention",
                     "indexed_attention_geometry": packed,
                     "paired_sfa": {"calls": packed["paired_calls"],
                                    "call_heads": packed["heads"],
                                    "peer_exchange": packed["dcp_peer_exchange"]}})
    shape = (sum(view.q_len for view in views), heads, plan.kv_lora_rank)
    out_bytes = dtype_bytes(b.model.activation_dtype)
    phys_rows = max(shape[0], sum(peer_rows) if peer_rows is not None else rows * max(1, dp_width))
    # Upstream ``device_op.py:674-684``: non-quantized BF16 SFA accepts 1..64 or 128
    # query heads, zero-padding intermediate head counts (e.g. Kimi 96 -> 128) for the
    # kernel call and slicing output/LSE back to ``heads``.
    pad_bf16_heads = (
        not native_packed and bool(plan.indexer)
        and SFA_MAX_GENERIC_QUERY_GROUP_SIZE < heads < SFA_PADDED_QUERY_GROUP_SIZE
    )
    launch_heads = SFA_PADDED_QUERY_GROUP_SIZE if pad_bf16_heads else heads
    physical_shape = (phys_rows, launch_heads, plan.kv_lora_rank)
    head_pad_scale = float(launch_heads) / float(heads) if pad_bf16_heads else 1.0
    padding_output = max(0, phys_rows - shape[0]) * heads * plan.kv_lora_rank * out_bytes
    statistics = phys_rows * heads * 2 * FP32_BYTES if native_packed or cp > 1 else 0
    extra_output = statistics + padding_output
    if packed is not None:
        packed["statistics_bytes"] = statistics
        packed["padding_output_bytes"] = padding_output
    meta.update({
        "qsfa_output_bytes": phys_rows * heads * (plan.kv_lora_rank * out_bytes + 2 * FP32_BYTES),
        "qsfa_statistics_bytes": statistics,
        "qsfa_output_dtype": b.model.activation_dtype,
    })
    if pad_bf16_heads:
        meta["sfa_padded_query_heads"] = SFA_PADDED_QUERY_GROUP_SIZE
    batch = replace(
        members[0], op_id=b.prefix + ".latent_attention_batch",
        recipe=PAIRED_FP8_QSFA_IMPLEMENTATION if native_packed else "paired_mla_attention_analytic",
        source_boundary="kernel" if native_packed else "",
        semantic_shape=shape, local_shape=shape, padded_shape=physical_shape,
        launch_shape=physical_shape,
        deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
        flops=sum(op.flops for op in members) * head_pad_scale,
        vector_ops=sum(op.vector_ops for op in members) * head_pad_scale,
        hbm_bytes=sum(op.hbm_bytes for op in members) + extra_output,
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members) + extra_output,
        metadata=tuple(meta.items()),
    )
    b.ops[:] = [op for op in b.ops if op.op_id not in ids]
    b.ops.append(batch)
    if cp <= 1:
        return [batch.op_id]
    eff_tp = tp_gather if (pcp_factor * tp_gather * dp_width == cp) else cp
    eff_pcp = pcp_factor if (pcp_factor * tp_gather * dp_width == cp) else 1
    eff_dp = dp_width if (pcp_factor * tp_gather * dp_width == cp) else 1
    merged = emit_dcp_output_exchange_combine(
        b.ops, prefix=b.prefix, stage=b.common["stage"], layer=plan.index,
        requests=b.common["requests"], mode=b.common["phase"],
        deps=(batch.op_id,), rows=rows, local_heads=local_heads,
        v_dim=plan.kv_lora_rank, act_dtype=b.model.activation_dtype,
        pcp_factor=eff_pcp, tp_gather=eff_tp, dp_width=eff_dp,
        lse_in_bytes=2 * FP32_BYTES,
        peer_rows=peer_rows, local_dp_pos=local_dp_pos,
    )
    return [merged]


def _build_paired_qsfa(op: OpSpec, geometry: Any) -> OperatorPlan:
    meta = dict(op.metadata)
    if meta.get("native_kernel") != PACKED_KERNEL:
        raise ModelingError("paired_fp8_qsfa_call_contract_required")
    if (meta.get("program_kind") == "indexed_attention"
            and meta.get("indexed_attention_geometry")):
        from mindie_sim.prediction.operators.paged_mla.indexed_quant_attention import (
            SCHEDULE_ARCH22,
            SCHEDULE_ARCH35,
            build as build_packed,
        )
        arch_id = getattr(getattr(geometry, "compute_arch", None), "arch_id", None)
        if arch_id is None and getattr(geometry, "datapath", None) is not None:
            arch_id = getattr(getattr(geometry.datapath, "compute_arch", None), "arch_id", None)
        arch_str = str(arch_id or "").strip()
        if arch_str in {"npu_arch_3510", "DAV_3510"}:
            sched = SCHEDULE_ARCH35
        elif arch_str.startswith(("npu_arch_22", "DAV_22")):
            sched = SCHEDULE_ARCH22
        else:
            raise ModelingError(f"packed_int8_indexed_attention_arch_unknown:{arch_id!r}")
        return build_packed(op, geometry, scratch_policy="offchip", compact=True,
                            operand_modes=True, schedule=sched)
    work = Work(matrix={op.dtype: op.flops}, resources={
        "hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
        "mte3": op.mte3_bytes, "vector": op.vector_ops})
    return OperatorPlan(work, TiledFlashAndPagedAttentionRecipe().lower,
                        {"fidelity": "class_a_tiled_attention",
                         "source": PAIRED_SFA_SOURCE})


def _bind_dcp_merge_dram(context: Any) -> Any:
    """Bind sustainable AIV DRAM service rate on the shared HBM bus."""
    dram_rate = context.service_rate("hbm_rw21_dram") or context.service_rate("mte2_aiv_dram")
    if dram_rate is None or "hbm" not in context.resource_names:
        return context
    ceiling = min(dram_rate, context.rates[context.resource_names["hbm"]] * MS_PER_SECOND)
    return replace(context, service_modes={**context.service_modes, "hbm": ("hbm", ceiling)})


def expand_dcp_lse_eager_kernels(
    op: OpSpec, *, l2_bytes: int = 0, include_stack: bool = False, local_ids: bool = False,
) -> tuple[OpSpec, ...]:
    """Expand a ``dcp_lse_merge`` call into the 9-kernel eager LSE combine chain."""
    meta = dict(op.metadata)
    cp = max(2, int(meta.get("partial_count") or meta.get("cp") or 2))
    shape = tuple(int(d) for d in (op.local_shape or op.semantic_shape or (1, 1, 1)))
    if len(shape) >= 3:
        rows, heads, dim = max(1, shape[0]), max(1, shape[1]), max(1, shape[2])
    elif len(shape) == 2:
        rows, heads, dim = 1, max(1, shape[0]), max(1, shape[1])
    else:
        rows, heads, dim = 1, 1, max(1, shape[0])
    out_bytes = dtype_bytes(op.dtype or "bf16")
    n_lse = cp * rows * heads
    n_out = cp * rows * heads * dim
    lse_b = n_lse * FP32_BYTES
    red_lse_b = rows * heads * FP32_BYTES
    ol_in_b = n_out * out_bytes
    ol_fp32_b = n_out * FP32_BYTES
    res_fp32_b = rows * heads * dim * FP32_BYTES
    res_out_b = rows * heads * dim * out_bytes

    def _hbm_bytes(rd: int, wr: int, l2_in: int, *, writeback_l2: bool = False) -> float:
        fits_l2 = l2_bytes > 0 and (rd + wr) <= l2_bytes
        hit = min(rd, l2_in) if fits_l2 else 0
        # Ephemeral chain intermediates whose inputs and outputs fit in L2 stay
        # in write-back L2 until consumed by the next kernel (zero HBM spill).
        hbm_wr = 0 if (writeback_l2 and fits_l2 and hit >= rd) else wr
        return float(max(0, rd - hit) + hbm_wr)

    def _cid(suffix: str) -> str:
        return suffix if local_ids else f"{op.op_id}.{suffix}"

    base_deps = () if local_ids else tuple(op.deps)
    specs: list[tuple[str, tuple[str, ...], str, int, int, int, int, int, str]] = []
    if include_stack:
        specs.append(("lse_stack", base_deps, "fp32", n_lse, n_lse, lse_b, lse_b, 0, ""))
        amax_deps, amax_l2 = (_cid("lse_stack"),), lse_b
    else:
        amax_deps, amax_l2 = base_deps, 0
    specs.extend((
        ("amax", amax_deps, "fp32", n_lse, n_lse, lse_b, red_lse_b, amax_l2, ""),
        ("sub", (_cid("amax"),), "fp32", n_lse, n_lse, lse_b + red_lse_b, lse_b, lse_b + red_lse_b, ""),
        ("exp", (_cid("sub"),), "fp32", n_lse, ATTENTION_UPDATE_LSE_EXP_OPS * n_lse, lse_b, lse_b, lse_b, ""),
        ("wsum", (_cid("exp"),), "fp32", n_lse, n_lse, lse_b, red_lse_b, lse_b, ""),
        ("wdiv", (_cid("wsum"),), "fp32", n_lse, n_lse, lse_b + red_lse_b, lse_b, lse_b + red_lse_b, ""),
        ("cast32", (_cid("wdiv"),), "fp32", n_out, n_out, ol_in_b, ol_fp32_b, 0, ""),
    ))
    if include_stack:
        specs.append(("o_stack", (_cid("cast32"),), "fp32", n_out, n_out, ol_fp32_b, ol_fp32_b, ol_fp32_b, ""))
        mul_deps = (_cid("o_stack"),)
    else:
        mul_deps = (_cid("cast32"),)
    specs.extend((
        ("mul", mul_deps, "fp32", n_out, n_out, ol_fp32_b + lse_b, ol_fp32_b, ol_fp32_b + lse_b, "broadcast_elementwise"),
        ("osum", (_cid("mul"),), "fp32", n_out, n_out, ol_fp32_b, res_fp32_b, ol_fp32_b, "outer_axis_reduction"),
        ("cast16", (_cid("osum"),), op.dtype or "bf16", rows * heads * dim, rows * heads * dim, res_fp32_b, res_out_b, res_fp32_b, ""),
    ))
    child_meta = (("execution_class", "aiv_only"), ("cp", cp), ("partial_reduction", "logsumexp"))
    return tuple(
        OpSpec(
            _cid(suffix), "vector", op.phase, op.stage, op.layer, op.requests,
            (numel,), (numel,), (numel,), (numel,),
            deps=deps, dtype=dt, stream=op.stream,
            vector_ops=float(vops), mte2_bytes=float(rd), mte3_bytes=float(wr),
            hbm_bytes=_hbm_bytes(rd, wr, l2_in, writeback_l2=bool(vsub)),
            metadata=(
                ("program_execution", {"execution_class": "aiv_only"}),
                *child_meta,
                *((("vector_submode", vsub),) if vsub else ()),
            ),
        )
        for suffix, deps, dt, numel, vops, rd, wr, l2_in, vsub in specs
    )


def _expand_dcp_lse_merge_kernels(op: OpSpec) -> tuple[OpSpec, ...]:
    meta = dict(op.metadata)
    return expand_dcp_lse_eager_kernels(
        op,
        l2_bytes=int(meta.get("l2_bytes", 0) or 0),
        include_stack=bool(meta.get("include_stack", False)),
        local_ids=True,
    )


def _build_dcp_lse_merge(op: OpSpec, geometry: Any) -> OperatorPlan:
    """Lower fused CANN ``AttentionUpdate`` (``decode_update.h``) LSE combine across DCP=2..16."""
    meta = dict(op.metadata)
    partial_count = max(2, int(meta.get("partial_count") or meta.get("cp") or 2))
    shape = tuple(int(d) for d in (op.local_shape or op.semantic_shape or (1, 1, 1)))
    total_rows = max(1, shape[0] * shape[1]) if len(shape) >= 2 else max(1, shape[0])
    aiv_count = max(1, int((getattr(geometry, "resource_units", {}) or {}).get("vector", 1)))
    active_aiv = max(1, min(aiv_count, total_rows))
    sp_step = ATTENTION_UPDATE_MAX_EFFECTIVE_SP - 1
    sp_loops = (
        1
        if partial_count <= ATTENTION_UPDATE_MAX_EFFECTIVE_SP
        else 1 + math.ceil((partial_count - ATTENTION_UPDATE_MAX_EFFECTIVE_SP) / sp_step)
    )
    work = Work(resources={
        "hbm": float(op.hbm_bytes),
        "mte2": float(op.mte2_bytes),
        "mte3": float(op.mte3_bytes),
        "vector": float(op.vector_ops),
    })
    occupancy = float(active_aiv) / float(aiv_count)

    def _lower(context: Any) -> Any:
        read_share = io_weight(context, "hbm_read")
        write_share = io_weight(context, "hbm_write")
        phases = (
            Phase("copy_in_sp", {IO_HBM_CATEGORY: read_share, "mte2": 1.0}),
            Phase("lse_combine", {"vector": 1.0}, after=("copy_in_sp",)),
            Phase("copy_out", {IO_HBM_CATEGORY: write_share, "mte3": 1.0}, after=("lse_combine",)),
        )
        tile_count = max(1, min(ATTENTION_UPDATE_TILE_CAP, sp_loops * math.ceil(total_rows / active_aiv)))
        if occupancy < 1.0:
            scaled_rates = dict(context.rates)
            for role in ("vector", "mte2", "mte2_aiv", "mte3"):
                rname = context.resource_names.get(role)
                if rname is not None and rname in scaled_rates:
                    scaled_rates[rname] = scaled_rates[rname] * occupancy
            context = replace(context, rates=scaled_rates)
        return tiled_phased_program(
            context, phases, slots=ATTENTION_UPDATE_PIPELINE_SLOTS, cap=tile_count,
        )

    return OperatorPlan(
        work,
        _lower,
        {
            "execution_class": "aiv_only",
            "launched_blocks": active_aiv,
            "partial_count": partial_count,
            "sp_loops": sp_loops,
            "source": ATTENTION_UPDATE_SOURCE,
        },
        context_binder=_bind_dcp_merge_dram,
        traffic_rewrite_capable=True,
    )


def recipes() -> dict[str, PlannedRecipe]:
    return {
        PAIRED_FP8_QSFA_IMPLEMENTATION: PlannedRecipe(_build_paired_qsfa),
        DCP_LSE_MERGE_RECIPE: PlannedRecipe(
            _build_dcp_lse_merge,
            traffic_rewrite_capable=True,
        ),
        DCP_LSE_MERGE_EAGER_RECIPE: PlannedRecipe(
            _build_dcp_lse_merge,
            kernels=_expand_dcp_lse_merge_kernels,
            traffic_rewrite_capable=True,
        ),
    }


__all__ = [
    "ATTENTION_UPDATE_MAX_EFFECTIVE_SP",
    "CP_CHUNKED_PREFILL_STREAM",
    "CP_DECODE_COMM_STREAM",
    "CP_WEIGHT_STREAM",
    "DCP_LSE_MERGE_EAGER_RECIPE",
    "DCP_LSE_MERGE_RECIPE",
    "FAST_LSE_WIRE_BYTES",
    "GENERIC_LSE_WIRE_BYTES",
    "PACK_LSE_OPS",
    "PAIRED_FP8_QSFA_IMPLEMENTATION",
    "active_layer_dcp",
    "active_layer_dcp_factors",
    "dcp_q_replicate_enabled",
    "emit_dcp_output_exchange_combine",
    "emit_dcp_query_gather",
    "emit_mtp_current_attention",
    "emit_pcp_hidden_gather_restore",
    "emit_pcp_o_proj_weight_allgather",
    "emit_pcp_restore_unpermute",
    "emit_sfa_index_remap",
    "emit_sfa_prefill_kv_gather",
    "emit_v41_pcp_c2_ring_gather",
    "emit_v41_pcp_c2_ring_write",
    "expand_dcp_lse_eager_kernels",
    "is_fast_sfa_dcp_exchange",
    "owned_positions",
    "packed_sfa_geometry",
    "paired_sfa_attention_batch",
    "paired_sfa_calls",
    "pcp_activation_gather_scale",
    "pcp_o_proj_weight_sharding_enabled",
    "pcp_quantized_allgather_enabled",
    "pcp_uniform_peer_rows",
    "quantized_allreduce_eligible",
    "recipes",
    "resolve_cp_rank_candidates",
    "resolve_dcp_cohort_decode",
    "rewrite_quantized_allreduce",
    "should_split_mtp_dcp",
]
