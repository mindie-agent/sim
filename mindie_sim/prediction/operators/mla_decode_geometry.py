"""Triton MLA source dispatch and representation geometry.

[Role] Normalize the source caller into geometry and work counters.
[Invariants] Explicit splits and supported latent/rope shapes determine one immutable plan.
  No durations, hardware-rate fits or runtime source reads enter normalization.
[Timeouts & Locks] Bounded metadata arithmetic; no waits or locks.
[Side-Effect Policy] Immutable declarations; no device or external I/O.
"""
from __future__ import annotations

from typing import Any, Mapping

IMPLEMENTATION = "triton_decode_grouped_mla"

# Source identity is recorded here so a plan can be audited without reading
# campaign metadata.  The coordinator compares these files independently.
SOURCE_FILES = (
    {
        "path": "/Users/maoxx241/code/vllm-v0.25.1/vllm/v1/attention/backends/mla/triton_mla.py",
        "sha256": "5e9aa13eca7ce718ce0648f11454bb0e0e98e9f5da40ee22bd8d0b05f4ba1d03",
        "lines": "35-47 split auto-selection (not used); 166-171 logit cap rejected; "
                 "199-237 wrapper q-cat/zero-init/cache alias; 241-256 grouped MLA call",
    },
    {
        "path": "/Users/maoxx241/code/vllm-v0.25.1/vllm/v1/attention/ops/triton_decode_attention.py",
        "sha256": "b9823c65eba22d76abf64dad7d7d16a950f67471c86d6cb5b3e983aaf14381ac",
        "lines": "59-65 page stride; 279-467 grouped stage 1; 415-432 latent PV alias; "
                 "486-572 MLA block dispatch; 575-681 stage 2 merge; 719-817 grouped dispatch",
    },
)

# Selected non-HIP block dispatch from _decode_grouped_att_m_fwd lines 486-519.
_BLOCK_H = 16
_BLOCK_N = 32          # non-HIP BLOCK = 32 (line 506)
_STAGE1_NUM_WARPS = 4  # line 566
_STAGE2_NUM_WARPS = 4  # line 678
_STAGE2_NUM_STAGES = 2 # line 679

# Lazy windows.  One resident CTA per physical stream.  Stage-1 key tiles are
# strictly serial inside a CTA: qk(i+1) must not run before pv(i), because both
# consume the same matrix unit and the source loop body is ordered.  The
# stage-2 merge prefetches one small partial while the previous online merge
# runs; its reduction chain remains serial.
_CTA_STREAM_SLOTS = 1
_TILE_PIPELINE_SLOTS = 1
_MERGE_PIPELINE_SLOTS = 2

# Explicitly modelled BLOCK_DMODEL/BLOCK_DPE/BLOCK_DV special cases (lines
# 491-504).  The generic fallback is deliberately not selected here: for a
# non-power-of-two Lv it can overlap the latent and PE tiles and would need a
# separate address review.
_SPECIAL_HEAD_DIMS = {
    (512, 64): (512, 64, 512),
    (256, 32): (256, 32, 256),
}

_DTYPE_ALIASES = {"bf16": "bf16", "bfloat16": "bf16"}

# B_Seqlen / Req_to_tokens entries are not declared by the kernel signature;
# the 4-byte integer width is an explicit modelling assumption, not a
# measurement.
_INDEX_BYTES = 4
_PROGRAM_BYTES = 2  # BF16 q/k cache and BF16 output/lse
_ACCUM_BYTES = 4    # FP32 partial output / LSE scratch

_BOUNDARY = (
    "decode_attention_fwd -> decode_attention_fwd_grouped -> "
    "_decode_grouped_att_m_fwd + _decode_softmax_reducev_fwd.  The wrapper's "
    "optional q tuple concat and o/lse zero-init in TritonMLAImpl.forward_mqa "
    "are preceding materialisations outside this operator's launch boundary."
)

_KERNEL_PHASES = ("stage1", "stage2")


def _required(params: Mapping[str, Any], key: str):
    if key not in params:
        raise ValueError(f"mla_decode_parameter_required:{key}")
    return params[key]


def _integer(value, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"mla_decode_integer_required:{name}")
    if minimum is not None and value < minimum:
        raise ValueError(f"mla_decode_integer_out_of_range:{name}:{value}")
    return value


def _dtype(value, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"mla_decode_dtype_required:{name}")
    try:
        return _DTYPE_ALIASES[value.lower()]
    except KeyError:
        raise ValueError(f"mla_decode_unsupported_dtype:{name}:{value}") from None


def _cdiv(numerator: int, denominator: int) -> int:
    if denominator < 1:
        raise ValueError("mla_decode_cdiv_requires_positive_denominator")
    return -(-numerator // denominator)


def _validate_contexts(raw) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("mla_decode_contexts_must_be_sequence")
    if not raw:
        raise ValueError("mla_decode_contexts_must_be_non_empty")
    return tuple(_integer(value, "contexts[]", minimum=1) for value in raw)


def _select(params: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate one explicit source dispatch.

    Every branch flag is retained in the returned mapping.  Unsupported
    neighbouring paths raise instead of falling back to another kernel.
    """
    implementation = _required(params, "implementation")
    if implementation != IMPLEMENTATION:
        raise ValueError(
            f"mla_decode_unsupported_implementation:{implementation}:"
            f"selected:{IMPLEMENTATION}"
        )

    is_mla = _required(params, "is_mla")
    if is_mla is not True:
        raise ValueError(
            "mla_decode_unsupported_dispatch:is_mla:"
            "selected_IS_MLA_true_only"
        )
    is_hip = _required(params, "is_hip")
    if is_hip is not False:
        raise ValueError(
            "mla_decode_unsupported_dispatch:is_hip:"
            "selected_non_hip_blocks_only"
        )

    logit_cap = params.get("logit_cap", 0.0)
    if isinstance(logit_cap, bool) or not isinstance(logit_cap, (int, float)):
        raise ValueError("mla_decode_logit_cap_must_be_numeric")
    if float(logit_cap) != 0.0:
        # TritonMLAImpl rejects a truthy logits_soft_cap at lines 166-171.
        raise ValueError(
            "mla_decode_unsupported_dispatch:logit_cap:"
            "triton_mla_rejects_soft_cap"
        )

    dtype = _dtype(_required(params, "dtype"), "dtype")
    cache_dtype = _dtype(_required(params, "cache_dtype"), "cache_dtype")
    if dtype != "bf16" or cache_dtype != "bf16":
        raise ValueError(
            "mla_decode_unsupported_dispatch:cache_dtype:"
            "selected_bf16_query_and_bf16_cache_only"
        )

    heads = _integer(_required(params, "heads"), "heads", minimum=1)
    key_heads = _integer(_required(params, "key_heads"), "key_heads", minimum=1)
    latent_dim = _integer(_required(params, "latent_dim"), "latent_dim", minimum=1)
    rope_dim = _integer(_required(params, "rope_dim"), "rope_dim", minimum=1)
    page_size = _integer(_required(params, "page_size"), "page_size", minimum=1)
    splits = _integer(
        _required(params, "num_kv_splits"), "num_kv_splits", minimum=1
    )
    contexts = _validate_contexts(_required(params, "contexts"))

    if key_heads != 1:
        raise ValueError(
            "mla_decode_unsupported_dispatch:key_heads:"
            "selected_latent_mla_single_kv_head"
        )
    if heads < 2:
        # _decode_grouped_att_m_fwd is selected only when kv_group_num > 1
        # (decode_attention_fwd lines 780-817).
        raise ValueError(
            "mla_decode_unsupported_dispatch:heads:"
            "grouped_path_requires_kv_group_num_gt_1"
        )
    # TritonMLABackend.supports_block_size requires block_size % 16 == 0.
    if page_size % 16:
        raise ValueError(
            f"mla_decode_unsupported_page_size:{page_size}:"
            "triton_mla_supported_kernel_block_sizes_are_multiples_of_16"
        )
    try:
        block_dmodel, block_dpe, block_dv = _SPECIAL_HEAD_DIMS[
            (latent_dim, rope_dim)
        ]
    except KeyError:
        raise ValueError(
            "mla_decode_unsupported_head_dim:"
            f"latent={latent_dim}:rope={rope_dim}:"
            "selected_special_block_dispatch_(512,64)_or_(256,32)_only"
        ) from None

    # Block 2 of the source grid can carry masked head lanes; the MMA still
    # executes BLOCK_H rows.  This is recorded, not normalised away.
    kv_group_num = heads // key_heads
    valid_block_h = _BLOCK_H if kv_group_num > _BLOCK_H else kv_group_num
    head_blocks = _cdiv(heads, valid_block_h)
    padded_heads = head_blocks * _BLOCK_H
    num_warps = _STAGE1_NUM_WARPS
    # Non-HIP stage 1 uses num_stages=2 unless BLOCK_DMODEL >= 1024.
    num_stages = 1 if block_dmodel >= 1024 else 2

    dispatch = {
        "implementation": IMPLEMENTATION,
        "is_mla": True,
        "is_hip": False,
        "logit_cap": 0.0,
        "dtype": dtype,
        "cache_dtype": cache_dtype,
        "key_heads": key_heads,
        "page_size": page_size,
        "num_kv_splits": splits,
        "split_selection": "explicit",
        "grouped": True,
        "kv_group_num": kv_group_num,
        "path": "grouped",
        "stage1_kernel": "_fwd_grouped_kernel_stage1",
        "stage2_kernel": "_fwd_kernel_stage2",
        "kernel_launches": ("stage1_grouped_decode", "stage2_split_merge"),
        "source_kernel_launches": 2,
        "stage1_program": "per_split_head_cta",
        "stage2_program": "per_head_merge_cta",
        "tile_order": "qk_softmax_pv",
        "block_h": _BLOCK_H,
        "block_n": _BLOCK_N,
        "block_dmodel": block_dmodel,
        "block_dpe": block_dpe,
        "block_dv": block_dv,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "stage2_block_dv": block_dv,
        "stage2_num_warps": _STAGE2_NUM_WARPS,
        "stage2_num_stages": _STAGE2_NUM_STAGES,
        "qk_parts": "latent_plus_pe",
        "pv_source": "latent_k_alias",
        "v_buffer_reads": 0,
        "softmax": "online_lse",
        "empty_splits": "stage1_launches_stage2_skips",
        "split_reduction_owner": "batch_head",
    }

    split_chunks, nonempty_counts, block_tiles = (), (), ()
    for length in contexts:
        chunk, nonempty, tiles = _split_summary(length, splits, _BLOCK_N)
        split_chunks += (chunk,)
        nonempty_counts += (nonempty,)
        block_tiles += (tiles,)

    plan = {
        "batch": len(contexts),
        "heads": heads,
        "key_heads": key_heads,
        "latent_dim": latent_dim,
        "rope_dim": rope_dim,
        "qk_dim": latent_dim + rope_dim,
        "page_size": page_size,
        "num_kv_splits": splits,
        "block_h": _BLOCK_H,
        "block_n": _BLOCK_N,
        "block_dmodel": block_dmodel,
        "block_dpe": block_dpe,
        "block_dv": block_dv,
        "head_blocks": head_blocks,
        "valid_block_h": valid_block_h,
        "padded_heads": padded_heads,
        "stage1_num_warps": num_warps,
        "stage1_num_stages": num_stages,
        "stage2_num_warps": _STAGE2_NUM_WARPS,
        "stage2_num_stages": _STAGE2_NUM_STAGES,
        "stage1_grid": (len(contexts), head_blocks, splits),
        "stage2_grid": (len(contexts), heads),
        # Per-request split/page audit data.  O(batch), never per token.
        "split_chunks": split_chunks,
        "nonempty_splits": nonempty_counts,
        "block_tiles": block_tiles,
        "pages_per_request": tuple(_cdiv(length, page_size) for length in contexts),
        "page_index_entries": head_blocks * sum(contexts),
        "stage1_programs": len(contexts) * head_blocks * splits,
        "stage2_programs": len(contexts) * heads,
    }
    return {"dispatch": dispatch, "plan": plan, "seq_lens": contexts}


def _split_summary(length: int, splits: int, block_n: int) -> tuple[int, int, int]:
    """Return (chunk, nonempty_splits, block_n_tiles) for one request.

    Mirrors ``kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)`` and the
    tile loop ``range(start, end, BLOCK_N)``.  Each non-empty split restarts
    the tile grid, so tails are not merged across splits.
    """
    chunk = _cdiv(length, splits)
    nonempty = _cdiv(length, chunk)
    remainder = length - (nonempty - 1) * chunk
    tiles = (nonempty - 1) * _cdiv(chunk, block_n) + _cdiv(remainder, block_n)
    return chunk, nonempty, tiles

