"""CANN 9.3 Arch35 ``GenericBlockSparseAttention`` and C8 ``KvQuantSFA`` recipes.

[Role]
  Model ``cann_ops_transformer.generic_block_sparse_attention`` (parameterized
  by ``block_shape=(1, B_k)`` for ``[1, 4]`` QSA, ``[1, 8]`` Step-5 CSA, and
  ``[1, 64]`` / ``[1, 128]`` MiniMax-M3 MSA across ``quant_mode in {0, 1, 5, 6}``),
  ``#22467`` A5 C8 ``KvQuantSparseFlashAttentionV2`` with precomputed FP8 scale
  addresses, and ``#21588`` NoPE ``sparse_flash_mla``.

[Invariants]
  * ``OperatorPlan.work`` equals the exact sum of issued per-owner ``Stage``
    demands across ``hbm``, ``l2``, ``mte2_aic``, ``mte2_aiv``, ``mte1``,
    ``cube``, ``vector``, ``fixpipe``, and ``mte3``.
  * Shared-prefix KV/scale block reuse in L2 is computed from declared
    ``geometry.buffers`` capacity; query waves are placed across physical cores.
  * Zero model-name branches or ``ResourceEnvelope`` fallbacks participate.

[Timeouts & Locks]
  O(compute_units) plan construction; bounded finite stages, no I/O or locks.

[Side-Effect Policy]
  Pure immutable ``OpSpec`` and ``OperatorPlan`` construction.
"""
from __future__ import annotations

import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ..mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from ..ownership import read_role, unit_group
from ..program import KernelProgram, ParallelProgram, Stage

GBSA_RECIPE = "generic_block_sparse_attention_arch35"
KV_QUANT_SFA_C8_RECIPE = "kv_quant_sfa_c8_arch35"
SPARSE_FLASH_MLA_NOPE_RECIPE = "sparse_flash_mla_nope_arch35"

QUANT_MODE_BF16 = 0
QUANT_MODE_INT8_C8 = 1
QUANT_MODE_FP8_C8 = 5
QUANT_MODE_MXFP8 = 6
SUPPORTED_QUANT_MODES = frozenset({
    QUANT_MODE_BF16, QUANT_MODE_INT8_C8, QUANT_MODE_FP8_C8, QUANT_MODE_MXFP8,
})
DEFAULT_GBSA_BLOCK_SIZE = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_SELECTED_TOKENS = 128
DEFAULT_SFA_LATENT_RANK = 512
DEFAULT_SFA_TOPK_TOKENS = 2048

GBSA_SOURCE = {
    "repository": "cann_ops_transformer",
    "revision": "1.0.0",
    "relative_path": (
        "cann_ops_transformer/ops/attention/generic_block_sparse_attention/"
        "generic_block_sparse_attention.py"
    ),
    "source_sha256": "2f8a14651d6c0b2a0ea1dbae7bfae3be6b078e32e79f8640e1d019b800a5568c",
}
VLLM_GBSA_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/ops/gbsa.py",
    "source_sha256": "b64283d085fa7cd3149c17ea3622424b078796651bf4ebef46f64e60ef7ba78b",
}
KV_QUANT_SFA_SOURCE = {
    "repository": "cann-toolkit",
    "revision": "9.3.0-20261112_192141454",
    "relative_path": "include/aclnnop/aclnn_kv_quant_sparse_flash_attention_v2.h",
    "source_sha256": "88b59a2b2a19cf487b0c13038b2ffc8a410760ae381424797447b00c08838f3a",
}
PR22467_SFA_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "1b33178393",
    "relative_path": "vllm_ascend/attention/sfa_v1.py",
    "source_sha256": "058aeb723a4ef3cdb0e07a5fc0180ec53f8227f62d98710144ecffb9cc66678f",
}
SPARSE_FLASH_MLA_SOURCE = {
    "repository": "cann_ops_transformer",
    "revision": "1.0.0",
    "relative_path": "cann_ops_transformer/ops/attention/sparse_flash_mla/sparse_flash_mla.py",
    "source_sha256": "f79685270ab99ecdab03341ade8c5fe11a0bc47c3ad43d937488a6a67e0b5d80",
}

BF16_BYTES = 2
INT32_BYTES = 4
FP32_BYTES = 4
MX_GROUP_SIZE = 32
C8_SCALE_GROUP = 128
ONLINE_SOFTMAX_OPS = 5
SSMAX_EXTRA_OPS = 2
DEQUANT_OPS_PER_ELEM = 2
SCALE_GATHER_OPS_PER_BLOCK = 12
PRECOMPUTED_SCALE_OPS_PER_BLOCK = 2
L2_EFFECTIVE_SHARE = 0.75

SFA_C8_SETUP_MS = 0.0175
SFA_NOPE_SETUP_MS = 0.014
GBSA_SETUP_MS = 0.0078
GBSA_GQA_GROUP_ROWS = 16
SFA_GATHER_MS_PER_TOKEN = 0.0000182
GBSA_SEL_MS_PER_TOKEN = 0.0000033
GBSA_DESC_SQRT_MS = 0.00064
SFA_SCATTER_DRAM_EFF = 0.56
GBSA_PEAK_DRAM_EFF = 0.68
GBSA_SCATTER_PIVOT_BYTES = 9216.0
GBSA_FLASH_TASK_NUMERATOR = 10
GBSA_FLASH_CORE_FACTOR = 3
GBSA_FLASH_COMBINE_BASE_MS = 0.0082
GBSA_FLASH_COMBINE_PER_QUERY_MS = 0.00105

BSA_BUFFERS = {
    "ub": 128 * 1024, "l1": 192 * 1024, "l0a": 16 * 1024,
    "l0b": 32 * 1024, "l0c": 32 * 1024,
}


def _check_buffers(geometry: Geometry) -> dict[str, int]:
    available = dict(geometry.buffers or {})
    for pool, required in BSA_BUFFERS.items():
        cap = int(available.get(pool, 0))
        if cap > 0 and required > cap:
            raise ModelingError(f"block_sparse_attn_buffer_capacity_exceeded:{pool}")
    return {p: sz for p, sz in BSA_BUFFERS.items() if available.get(p, 0) > 0}


def _effective_l2(geometry: Geometry) -> float:
    mechanisms = getattr(geometry, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    if hierarchy is not None:
        return float(hierarchy.effective_l2_bytes)
    return L2_EFFECTIVE_SHARE * float((geometry.buffers or {}).get("l2", 0))


def _kv_bytes_per_token(quant_mode: int, head_dim: int, v_dim: int) -> tuple[float, str]:
    if quant_mode == QUANT_MODE_BF16:
        return float((head_dim + v_dim) * BF16_BYTES), "bf16"
    if quant_mode == QUANT_MODE_INT8_C8:
        scales = 2.0 * FP32_BYTES
        return float(head_dim + v_dim) + scales, "bf16"
    if quant_mode == QUANT_MODE_FP8_C8:
        groups = math.ceil(head_dim / C8_SCALE_GROUP) + math.ceil(v_dim / C8_SCALE_GROUP)
        return float(head_dim + v_dim + groups * BF16_BYTES), "bf16"
    if quant_mode == QUANT_MODE_MXFP8:
        scales = math.ceil(head_dim / MX_GROUP_SIZE) + math.ceil(v_dim / MX_GROUP_SIZE)
        return float(head_dim + v_dim + scales), "fp8_e4m3"
    raise ModelingError(f"gbsa_unsupported_quant_mode:{quant_mode}")


def make_gbsa_op(
    *,
    op_id: str,
    deps: tuple[str, ...] = (),
    stage: int = 0,
    layer: int = 0,
    requests: tuple[str, ...] = (),
    phase: str = "decode",
    query_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int | None = None,
    selected_tokens: int,
    context_tokens: int | None = None,
    block_size: int = DEFAULT_GBSA_BLOCK_SIZE,
    quant_mode: int = QUANT_MODE_BF16,
    ssmax: bool = False,
    metadata_reuse: bool = False,
    batch_size: int | None = None,
    shared_prefix: bool = False,
) -> OpSpec:
    """Construct a ``generic_block_sparse_attention_arch35`` ``OpSpec``."""
    if int(quant_mode) not in SUPPORTED_QUANT_MODES:
        raise ModelingError(f"gbsa_unsupported_quant_mode:{quant_mode}")
    q_tok = max(1, int(query_tokens))
    q_heads = max(1, int(num_heads))
    kv_heads = max(1, int(num_kv_heads))
    dk = max(1, int(head_dim))
    dv = max(1, int(v_head_dim or dk))
    sel = max(1, int(selected_tokens))
    ctx_tok = max(sel, int(context_tokens)) if context_tokens is not None else None
    blk = max(1, int(block_size))
    batch = max(1, int(batch_size)) if batch_size is not None else max(1, len(requests) or q_tok)
    _, comp_dt = _kv_bytes_per_token(int(quant_mode), dk, dv)
    flops = float(q_tok * q_heads * (2 * sel * dk + 2 * sel * dv))
    shape = (q_tok, q_heads, dv)
    gbsa_geom: dict[str, Any] = {
        "query_tokens": q_tok, "num_heads": q_heads, "num_kv_heads": kv_heads,
        "head_dim": dk, "v_head_dim": dv, "selected_tokens": sel,
        "block_shape": (1, blk), "quant_mode": int(quant_mode),
        "ssmax": bool(ssmax), "metadata_reuse": bool(metadata_reuse),
        "batch_size": batch, "shared_prefix": bool(shared_prefix),
    }
    if ctx_tok is not None:
        gbsa_geom["context_tokens"] = ctx_tok
    meta = {
        "native_kind": "aclnnGenericBlockSparseAttention",
        "gbsa_geometry": gbsa_geom,
        "native_source": GBSA_SOURCE,
        "caller_source": VLLM_GBSA_SOURCE,
    }
    return OpSpec(
        op_id=op_id, deps=deps, family="attention", recipe=GBSA_RECIPE,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", flops=flops, matrix_flops=((comp_dt, flops),),
        source_boundary="kernel", metadata=tuple(sorted(meta.items())),
    )


def make_kv_quant_sfa_c8_op(
    *,
    op_id: str,
    deps: tuple[str, ...] = (),
    stage: int = 0,
    layer: int = 0,
    requests: tuple[str, ...] = (),
    phase: str = "decode",
    query_tokens: int,
    num_heads: int,
    kv_lora_rank: int = DEFAULT_SFA_LATENT_RANK,
    rope_dim: int = 0,
    selected_tokens: int = DEFAULT_SFA_TOPK_TOKENS,
    precompute_scale_addr: bool = True,
    nope_bf16: bool = False,
    batch_size: int | None = None,
    shared_prefix: bool = False,
) -> OpSpec:
    """Construct a C8 ``KvQuantSparseFlashAttentionV2`` (#22467) or NoPE SFA (#21588) op."""
    q_tok = max(1, int(query_tokens))
    q_heads = max(1, int(num_heads))
    latent = max(1, int(kv_lora_rank))
    rope = max(0, int(rope_dim))
    sel = max(1, int(selected_tokens))
    batch = max(1, int(batch_size)) if batch_size is not None else max(1, len(requests) or q_tok)
    qk_dim = latent + rope
    flops = float(q_tok * q_heads * (2 * sel * qk_dim + 2 * sel * latent))
    recipe = SPARSE_FLASH_MLA_NOPE_RECIPE if nope_bf16 else KV_QUANT_SFA_C8_RECIPE
    src = SPARSE_FLASH_MLA_SOURCE if nope_bf16 else KV_QUANT_SFA_SOURCE
    shape = (q_tok, q_heads, latent)
    meta = {
        "native_kind": (
            "aclnnSparseFlashMla" if nope_bf16 else "aclnnKvQuantSparseFlashAttentionV2"
        ),
        "sfa_geometry": {
            "query_tokens": q_tok, "num_heads": q_heads, "kv_lora_rank": latent,
            "rope_dim": rope, "selected_tokens": sel,
            "precompute_scale_addr": bool(precompute_scale_addr),
            "nope_bf16": bool(nope_bf16),
            "batch_size": batch, "shared_prefix": bool(shared_prefix),
        },
        "native_source": src,
        "caller_source": PR22467_SFA_SOURCE,
    }
    return OpSpec(
        op_id=op_id, deps=deps, family="mla", recipe=recipe,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", flops=flops, matrix_flops=(("bf16", flops),),
        source_boundary="kernel", metadata=tuple(sorted(meta.items())),
    )


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Lower ``GBSA``, C8 ``KvQuantSFA``, or NoPE ``SparseFlashMla`` with conserved work."""
    buf_sizes = _check_buffers(geometry)
    meta = dict(op.metadata)
    is_sfa = "sfa_geometry" in meta
    if is_sfa:
        sfa = dict(meta["sfa_geometry"])
        q_tok = int(sfa["query_tokens"])
        q_heads = int(sfa["num_heads"])
        kv_heads = 1
        dk = int(sfa["kv_lora_rank"]) + int(sfa["rope_dim"])
        dv = int(sfa["kv_lora_rank"])
        sel = int(sfa["selected_tokens"])
        blk = C8_SCALE_GROUP
        nope_bf16 = bool(sfa.get("nope_bf16", False))
        precomp = bool(sfa.get("precompute_scale_addr", True))
        batch = max(1, int(sfa.get("batch_size", q_tok)))
        shared = bool(sfa.get("shared_prefix", False))
        ctx_tok = None
        kv_tok_bytes = float(dk * BF16_BYTES) if nope_bf16 else float(
            int(sfa["kv_lora_rank"]) + int(sfa["rope_dim"]) * BF16_BYTES
            + math.ceil(int(sfa["kv_lora_rank"]) / C8_SCALE_GROUP) * FP32_BYTES
        )
        comp_dt = "bf16"
        blocks = max(1, math.ceil(sel / blk))
        addr_ops = PRECOMPUTED_SCALE_OPS_PER_BLOCK if precomp else SCALE_GATHER_OPS_PER_BLOCK
        extra_aiv_ops = float(q_tok * blocks * addr_ops + (0 if nope_bf16 else q_tok * sel * dk * DEQUANT_OPS_PER_ELEM))
        idx_bytes = float(q_tok * sel * INT32_BYTES + (q_tok * blocks * INT32_BYTES if precomp else q_tok * sel * INT32_BYTES))
        ssmax_flag = False
        src = meta.get("native_source") or KV_QUANT_SFA_SOURCE
    else:
        gbsa = dict(meta.get("gbsa_geometry") or {})
        q_tok = int(gbsa.get("query_tokens", op.local_shape[0] if op.local_shape else 1))
        q_heads = int(gbsa.get("num_heads", op.local_shape[1] if len(op.local_shape) > 1 else 1))
        kv_heads = max(1, int(gbsa.get("num_kv_heads", 1)))
        dk = int(gbsa.get("head_dim", op.local_shape[-1] if op.local_shape else DEFAULT_HEAD_DIM))
        dv = int(gbsa.get("v_head_dim", dk))
        sel = int(gbsa.get("selected_tokens", DEFAULT_SELECTED_TOKENS))
        block_shape = tuple(gbsa.get("block_shape") or (1, DEFAULT_GBSA_BLOCK_SIZE))
        blk = max(1, int(block_shape[-1]))
        q_mode = int(gbsa.get("quant_mode", QUANT_MODE_BF16))
        ssmax_flag = bool(gbsa.get("ssmax", False))
        meta_reuse = bool(gbsa.get("metadata_reuse", False))
        batch = max(1, int(gbsa.get("batch_size", q_tok)))
        shared = bool(gbsa.get("shared_prefix", False))
        raw_ctx = gbsa.get("context_tokens")
        ctx_tok = max(sel, int(raw_ctx)) if raw_ctx is not None else None
        nope_bf16 = False
        kv_tok_bytes, comp_dt = _kv_bytes_per_token(q_mode, dk, dv)
        blocks = max(1, math.ceil(sel / blk))
        idx_bytes = 0.0 if meta_reuse else float(q_tok * blocks * INT32_BYTES)
        dequant_modes = {QUANT_MODE_INT8_C8, QUANT_MODE_FP8_C8}
        extra_aiv_ops = float(q_tok * sel * dk * DEQUANT_OPS_PER_ELEM) if q_mode in dequant_modes else 0.0
        src = meta.get("native_source") or GBSA_SOURCE

    q_read = float(q_tok * q_heads * dk * BF16_BYTES)
    kv_read = float(q_tok * kv_heads * sel * kv_tok_bytes)
    aic_read = q_read + kv_read
    aiv_read = idx_bytes
    out_write = float(q_tok * q_heads * dv * BF16_BYTES)

    has_l2 = bool((geometry.rates or {}).get("l2", 0.0) or (geometry.resource_units or {}).get("l2", 0))
    l2_cap = _effective_l2(geometry) if has_l2 else 0.0
    if shared:
        unique_kv = float(kv_heads * sel * kv_tok_bytes)
    elif is_sfa:
        unique_kv = float(min(batch, q_tok) * kv_heads * sel * kv_tok_bytes)
    elif ctx_tok is not None and sel >= ctx_tok:
        unique_kv = float(batch * kv_heads * ctx_tok * kv_tok_bytes)
    else:
        unique_kv = float(min(batch, q_tok) * kv_heads * sel * kv_tok_bytes)
    repeated_kv = max(0.0, kv_read - unique_kv)
    l2_hit = repeated_kv * min(1.0, l2_cap / max(1.0, unique_kv)) if (l2_cap > 0.0 and repeated_kv > 0.0) else 0.0
    aic_gm_read = max(0.0, aic_read - l2_hit)

    flops = float(q_tok * q_heads * (2 * sel * dk + 2 * sel * dv))
    l0_bytes = q_read + kv_read + float(q_tok * q_heads * sel * BF16_BYTES)
    fix_bytes = float(q_tok * q_heads * sel * FP32_BYTES)
    softmax_ops = ONLINE_SOFTMAX_OPS + (SSMAX_EXTRA_OPS if ssmax_flag else 0)
    vec_ops = float(q_tok * q_heads * sel * softmax_ops) + extra_aiv_ops

    cores = max(1, int(geometry.compute_units))
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 2 * cores)))
    aic_pipe = read_role(geometry.resource_units or {}, aiv=None)
    aiv_pipe = read_role(geometry.resource_units or {}, aiv=0)

    plan_res: dict[str, float] = {
        "hbm": aic_gm_read + aiv_read + out_write,
        "mte1": l0_bytes, "fixpipe": fix_bytes,
        "vector": vec_ops, "mte3": out_write,
    }
    if aic_pipe == aiv_pipe:
        plan_res[aic_pipe] = aic_read + aiv_read
    else:
        plan_res[aic_pipe] = aic_read
        if aiv_read > 0.0:
            plan_res[aiv_pipe] = aiv_read
    if l2_hit > 0.0:
        plan_res["l2"] = l2_hit
    total_work = Work(matrix={comp_dt: flops}, resources={k: v for k, v in plan_res.items() if v > 0.0})

    head_groups = (
        1
        if is_sfa
        else kv_heads * max(1, math.ceil(q_heads / (kv_heads * GBSA_GQA_GROUP_ROWS)))
    )
    total_tasks = q_tok * head_groups
    active_cores = min(cores, total_tasks)
    waves = max(1, math.ceil(total_tasks / cores))

    def program(context):
        c_aic_pipe = read_role(context.resource_names, aiv=None)
        c_aiv_pipe = read_role(context.resource_names, aiv=0)
        share = 1.0 / float(active_cores)
        is_a5 = "mte2_aic_dram" in context.service_modes
        setup = (
            (SFA_NOPE_SETUP_MS if nope_bf16 else SFA_C8_SETUP_MS) if is_sfa else GBSA_SETUP_MS
        ) if is_a5 else 0.0
        gather_lat = 0.0
        caps: dict[str, float] = {}
        if is_a5:
            if is_sfa and not nope_bf16:
                gather_lat = float(waves) * float(sel) * SFA_GATHER_MS_PER_TOKEN
                dram_eff = SFA_SCATTER_DRAM_EFF
            else:
                seq_q = float(q_tok) / float(max(1, batch))
                density = (
                    min(1.0, float(sel) / float(max(1, ctx_tok)))
                    if ctx_tok is not None
                    else 0.0
                )
                uses_flash_split = (
                    ctx_tok is not None
                    and GBSA_FLASH_TASK_NUMERATOR * (q_tok * kv_heads) < GBSA_FLASH_CORE_FACTOR * cores
                )
                combine_lat = (
                    GBSA_FLASH_COMBINE_BASE_MS + GBSA_FLASH_COMBINE_PER_QUERY_MS * max(0.0, seq_q - 1.0)
                    if uses_flash_split
                    else 0.0
                )
                gather_lat = combine_lat + float(waves) * (
                    float(sel) * GBSA_SEL_MS_PER_TOKEN + math.sqrt(float(blocks)) * GBSA_DESC_SQRT_MS
                )
                block_bytes = float(blk) * kv_tok_bytes
                scatter_eff = GBSA_PEAK_DRAM_EFF * (
                    block_bytes / (block_bytes + GBSA_SCATTER_PIVOT_BYTES)
                )
                dram_eff = scatter_eff + (GBSA_PEAK_DRAM_EFF - scatter_eff) * density
            if "hbm" in context.resource_names:
                hbm_phys = context.resource_names["hbm"]
                hbm_rate = float(context.rates.get(hbm_phys, 0.0))
                if hbm_rate > 0.0:
                    caps[hbm_phys] = (dram_eff * hbm_rate) * share

        def per_core(core: int):
            if core >= active_cores:
                return KernelProgram(())
            aic_owned = context.owned(core, cores, ("cube", "mte1", "fixpipe", c_aic_pipe))
            aiv_owned = context.owned(core, cores, ("vector", c_aiv_pipe, "mte3"))
            allocs = tuple(
                context.allocate(
                    f"bsa_{core}_{pool}", pool, size,
                    unit_group(core, cores, context.pool_units.get(pool, cores)),
                )
                for pool, size in buf_sizes.items() if pool in context.pool_names
            )
            s_setup = Stage("bsa_setup", latency=setup, acquire=allocs, keep_empty=setup <= 0.0)
            aic_res: dict[str, float] = {
                "hbm": aic_gm_read * share, c_aic_pipe: aic_read * share,
                "mte1": l0_bytes * share, "fixpipe": fix_bytes * share,
            }
            if l2_hit > 0.0:
                aic_res["l2"] = l2_hit * share
            s_qk_pv = Stage(
                "sparse_qk_pv",
                context.bind(Work(matrix={comp_dt: flops * share}, resources=aic_res)),
                after=("bsa_setup",), latency=gather_lat, placement=aic_owned, caps=caps,
            )
            aiv_res: dict[str, float] = {
                "vector": vec_ops * share,
                "hbm": (aiv_read + out_write) * share,
                "mte3": out_write * share,
            }
            if aiv_read > 0.0:
                aiv_res[c_aiv_pipe] = aiv_read * share
            s_soft = Stage(
                "online_softmax_store",
                context.bind(Work(resources=aiv_res)),
                after=("sparse_qk_pv",), free=allocs, placement=aiv_owned,
            )
            return KernelProgram((s_setup, s_qk_pv, s_soft))

        return ParallelProgram(tuple(per_core(c) for c in range(cores)))

    return OperatorPlan(total_work, program, {
        "execution_class": "mixed_aic_aiv",
        "l2_shared_kv_bytes": l2_hit,
        "query_waves": waves,
        "source_program": src,
    })


def recipes() -> dict[str, PlannedRecipe]:
    planner = PlannedRecipe(build)
    return {
        GBSA_RECIPE: planner,
        KV_QUANT_SFA_C8_RECIPE: planner,
        SPARSE_FLASH_MLA_NOPE_RECIPE: planner,
    }
