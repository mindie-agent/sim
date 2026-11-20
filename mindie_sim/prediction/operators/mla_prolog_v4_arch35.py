"""CANN 9.3 Arch35 ``MlaPrologV3WeightNz`` / ``MlaPrologV4WeightNz`` recipe.

[Role]
  Model ``cann_ops_transformer.mla_prolog`` and ``aclnnMlaPrologV4WeightNz``
  (``weight_quant_mode in {0, 2, 3}`` BF16 / INT8 / MXFP8, ``kv_cache_quant_mode
  in {0, 1}`` BF16 or C8 packed 656 B / 520 B rows, ``do_rope in {True, False}``),
  fusing ``q_a_proj``, ``q_a_layernorm`` + MX quant, ``q_b_proj``,
  ``kv_a_proj_with_mqa``, ``kv_a_layernorm``, optional RoPE, ``w_uk`` absorption,
  and paged KV scatter without intermediate HBM round-trips.

[Invariants]
  * ``OperatorPlan.work`` equals the exact sum of issued per-owner ``Stage``
    demands across ``hbm``, ``l2``, ``mte2_aic``, ``mte2_aiv``, ``mte1``,
    ``cube``, ``vector``, ``fixpipe``, and ``mte3``.
  * ``do_rope=False`` (NoPE MLA) executes zero rotary vector arithmetic.
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

MLA_PROLOG_V4_RECIPE = "mla_prolog_v4_arch35"

MLA_PROLOG_V4_HEADER_SOURCE = {
    "repository": "cann-toolkit",
    "revision": "9.3.0-20261112_192141454",
    "relative_path": "include/aclnnop/aclnn_mla_prolog_v4_weight_nz.h",
    "source_sha256": "170ae3ebc125ddf56916ed83c37d84ce6bc3b92a0889bfc44a06cb6e30b22729",
}
CANN_MLA_PROLOG_SOURCE = {
    "repository": "cann_ops_transformer",
    "revision": "1.0.0",
    "relative_path": "cann_ops_transformer/ops/attention/mla_prolog_v3/mla_prolog.py",
    "source_sha256": "b8b259cc11e6acc174bbcdb60afaa32511e94a6a12e71520ba39e1faa8a43d9d",
}
KIMI_MLA_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/ops/kimi_mla.py",
    "source_sha256": "a0af4446b37d8b4b3d2ca7ddf7e21604c56300b6bd32cc70644605bbd9891f75",
}

BF16_BYTES = 2
FP32_BYTES = 4
SLOT_INDEX_BYTES = 8
MX_GROUP_SIZE = 32
C8_BLOCK_SIZE = 128
RMSNORM_OPS_PER_ELEM = 5
MX_QUANT_OPS_PER_ELEM = 4
C8_QUANT_OPS_PER_ELEM = 4
ROPE_OPS_PER_ELEM = 6
L2_EFFECTIVE_SHARE = 0.75

WEIGHT_QUANT_BF16 = 0
WEIGHT_QUANT_INT8 = 2
WEIGHT_QUANT_MXFP8 = 3
SUPPORTED_WEIGHT_MODES = frozenset({WEIGHT_QUANT_BF16, WEIGHT_QUANT_INT8, WEIGHT_QUANT_MXFP8})
DEFAULT_PROLOG_HIDDEN = 7168
DEFAULT_PROLOG_HEADS = 16
DEFAULT_Q_LORA_RANK = 1536
DEFAULT_KV_LORA_RANK = 512
DEFAULT_QK_NOPE_DIM = 128
DEFAULT_QK_ROPE_DIM = 64

PROLOG_SETUP_MS = 0.011
PROLOG_V3_SUBTILE_CUBE_EFF = 0.202
PROLOG_V4_SUBTILE_CUBE_EFF = 0.412
PROLOG_WIDE_Q_DIM_PIVOT = 3072.0
PROLOG_WIDE_Q_BASE_SCALE = 0.85
PROLOG_WIDE_Q_SLOPE_SCALE = 0.15
PROLOG_WIDE_Q_MAX_SCALE = 1.45
PROLOG_V3_EPILOG_MS_PER_TOK = 0.000232
PROLOG_V4_EPILOG_MS_PER_TOK = 0.000097
PROLOG_COLD_WEIGHT_OVERLAP = 0.35

PROLOG_BUFFERS = {
    "ub": 128 * 1024, "l1": 256 * 1024, "l0a": 32 * 1024,
    "l0b": 32 * 1024, "l0c": 64 * 1024,
}


def _check_buffers(geometry: Geometry) -> dict[str, int]:
    available = dict(geometry.buffers or {})
    for pool, required in PROLOG_BUFFERS.items():
        cap = int(available.get(pool, 0))
        if cap > 0 and required > cap:
            raise ModelingError(f"mla_prolog_v4_buffer_capacity_exceeded:{pool}")
    return {p: sz for p, sz in PROLOG_BUFFERS.items() if available.get(p, 0) > 0}


def _effective_l2(geometry: Geometry) -> float:
    mechanisms = getattr(geometry, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    if hierarchy is not None:
        return float(hierarchy.effective_l2_bytes)
    return L2_EFFECTIVE_SHARE * float((geometry.buffers or {}).get("l2", 0))


def _mx_weight_bytes(out_dim: int, in_dim: int) -> float:
    return float(out_dim * (in_dim + math.ceil(in_dim / MX_GROUP_SIZE)))


def make_op(
    *,
    op_id: str,
    deps: tuple[str, ...] = (),
    stage: int = 0,
    layer: int = 0,
    requests: tuple[str, ...] = (),
    phase: str = "decode",
    tokens: int,
    hidden_size: int,
    num_heads: int,
    q_lora_rank: int = DEFAULT_Q_LORA_RANK,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_nope_head_dim: int = DEFAULT_QK_NOPE_DIM,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_DIM,
    weight_quant_mode: int = WEIGHT_QUANT_MXFP8,
    kv_cache_quant_mode: int = 1,
    do_rope: bool | None = None,
) -> OpSpec:
    """Construct an ``mla_prolog_v4_arch35`` composite ``OpSpec``."""
    tok = max(1, int(tokens))
    hidden = max(1, int(hidden_size))
    heads = max(1, int(num_heads))
    q_rank = max(1, int(q_lora_rank))
    kv_rank = max(1, int(kv_lora_rank))
    nope = max(1, int(qk_nope_head_dim))
    rope = max(0, int(qk_rope_head_dim))
    rope_active = (rope > 0) if do_rope is None else (bool(do_rope) and rope > 0)
    w_mode = int(weight_quant_mode)
    kv_mode = int(kv_cache_quant_mode)
    if w_mode not in SUPPORTED_WEIGHT_MODES or kv_mode not in {0, 1}:
        raise ModelingError(f"mla_prolog_v4_unsupported_modes:{w_mode}:{kv_mode}")

    proj_flops = float(
        2 * tok * hidden * q_rank
        + 2 * tok * q_rank * (heads * (nope + rope))
        + 2 * tok * hidden * (kv_rank + rope)
    )
    absorb_flops = float(2 * tok * heads * nope * kv_rank)
    comp_dt = (
        "fp8_e4m3" if w_mode == WEIGHT_QUANT_MXFP8
        else ("int8" if w_mode == WEIGHT_QUANT_INT8 else "bf16")
    )
    matrix_pairs = (
        (("bf16", proj_flops + absorb_flops),)
        if comp_dt == "bf16"
        else ((comp_dt, proj_flops), ("bf16", absorb_flops))
    )
    shape = (tok, heads, kv_rank + rope)
    native_name = (
        "aclnnMlaPrologV4WeightNz"
        if w_mode == WEIGHT_QUANT_MXFP8 else "aclnnMlaPrologV3WeightNz"
    )
    meta = {
        "native_kind": native_name,
        "prolog_v4_geometry": {
            "tokens": tok, "hidden_size": hidden, "num_heads": heads,
            "q_lora_rank": q_rank, "kv_lora_rank": kv_rank,
            "qk_nope_head_dim": nope, "qk_rope_head_dim": rope,
            "weight_quant_mode": w_mode, "kv_cache_quant_mode": kv_mode,
            "do_rope": rope_active,
        },
        "native_source": MLA_PROLOG_V4_HEADER_SOURCE,
        "wrapper_source": CANN_MLA_PROLOG_SOURCE,
        "caller_source": KIMI_MLA_SOURCE,
    }
    return OpSpec(
        op_id=op_id, deps=deps, family="native_fusion", recipe=MLA_PROLOG_V4_RECIPE,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", flops=proj_flops + absorb_flops,
        matrix_flops=matrix_pairs, source_boundary="kernel",
        metadata=tuple(sorted(meta.items())),
    )


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build a work-conserving AIC+AIV program for ``MlaPrologV3/V4WeightNz``."""
    buf_sizes = _check_buffers(geometry)
    meta = dict(op.metadata)
    pg = dict(meta.get("prolog_v4_geometry") or {})
    tok = max(1, int(pg.get("tokens", op.local_shape[0] if op.local_shape else 1)))
    hidden = max(1, int(pg.get("hidden_size", DEFAULT_PROLOG_HIDDEN)))
    heads = max(1, int(pg.get("num_heads", op.local_shape[1] if len(op.local_shape) > 1 else DEFAULT_PROLOG_HEADS)))
    q_rank = max(1, int(pg.get("q_lora_rank", DEFAULT_Q_LORA_RANK)))
    kv_rank = max(1, int(pg.get("kv_lora_rank", DEFAULT_KV_LORA_RANK)))
    nope = max(1, int(pg.get("qk_nope_head_dim", DEFAULT_QK_NOPE_DIM)))
    rope = max(0, int(pg.get("qk_rope_head_dim", DEFAULT_QK_ROPE_DIM)))
    w_mode = int(pg.get("weight_quant_mode", WEIGHT_QUANT_MXFP8))
    kv_mode = int(pg.get("kv_cache_quant_mode", 1))
    do_rope = bool(pg.get("do_rope", rope > 0)) and rope > 0

    comp_dt = (
        "fp8_e4m3" if w_mode == WEIGHT_QUANT_MXFP8
        else ("int8" if w_mode == WEIGHT_QUANT_INT8 else "bf16")
    )
    if w_mode == WEIGHT_QUANT_MXFP8:
        w_bytes = (
            _mx_weight_bytes(q_rank, hidden)
            + _mx_weight_bytes(heads * (nope + rope), q_rank)
            + _mx_weight_bytes(kv_rank + rope, hidden)
        )
        in_x_bytes = float(tok * (hidden + math.ceil(hidden / MX_GROUP_SIZE)))
    elif w_mode == WEIGHT_QUANT_INT8:
        w_bytes = float(
            q_rank * hidden + heads * (nope + rope) * q_rank + (kv_rank + rope) * hidden
            + (q_rank + heads * (nope + rope) + kv_rank + rope) * FP32_BYTES
        )
        in_x_bytes = float(tok * (hidden + FP32_BYTES))
    else:
        w_bytes = float(
            (q_rank * hidden + heads * (nope + rope) * q_rank + (kv_rank + rope) * hidden)
            * BF16_BYTES
        )
        in_x_bytes = float(tok * hidden * BF16_BYTES)

    w_uk_bytes = float(heads * nope * kv_rank * BF16_BYTES)
    aic_read = in_x_bytes + w_bytes + w_uk_bytes

    has_l2 = bool((geometry.rates or {}).get("l2", 0.0) or (geometry.resource_units or {}).get("l2", 0))
    l2_cap = _effective_l2(geometry) if has_l2 else 0.0
    l2_hit = min(in_x_bytes, l2_cap) if l2_cap > 0.0 else 0.0
    aic_gm_read = max(0.0, aic_read - l2_hit)

    norm_param_bytes = float((q_rank + kv_rank) * BF16_BYTES)
    rope_table_bytes = float(2 * tok * rope * FP32_BYTES) if do_rope else 0.0
    slot_bytes = float(tok * SLOT_INDEX_BYTES)
    aiv_read = norm_param_bytes + rope_table_bytes + slot_bytes

    if kv_mode == 1:
        scale_groups = math.ceil(kv_rank / C8_BLOCK_SIZE)
        scale_bytes = scale_groups * (FP32_BYTES if rope > 0 else BF16_BYTES)
        kv_row_bytes = float(kv_rank + rope * BF16_BYTES + scale_bytes)
    else:
        kv_row_bytes = float((kv_rank + rope) * BF16_BYTES)
    q_out_bytes = float(tok * heads * (kv_rank + rope) * BF16_BYTES)
    qc_out_bytes = float(tok * q_rank * BF16_BYTES)
    out_write = q_out_bytes + qc_out_bytes + float(tok) * kv_row_bytes

    proj_flops = float(
        2 * tok * hidden * q_rank
        + 2 * tok * q_rank * (heads * (nope + rope))
        + 2 * tok * hidden * (kv_rank + rope)
    )
    absorb_flops = float(2 * tok * heads * nope * kv_rank)
    l0_proj = in_x_bytes + w_bytes
    l0_abs = w_uk_bytes
    fix_proj = float(tok * (q_rank + heads * (nope + rope) + kv_rank + rope) * BF16_BYTES)
    fix_abs = float(tok * heads * kv_rank * BF16_BYTES)

    quant_modes = {WEIGHT_QUANT_INT8, WEIGHT_QUANT_MXFP8}
    vec_ops = float(
        tok * (q_rank + kv_rank) * RMSNORM_OPS_PER_ELEM
        + (tok * q_rank * MX_QUANT_OPS_PER_ELEM if w_mode in quant_modes else 0)
        + (tok * kv_rank * C8_QUANT_OPS_PER_ELEM if kv_mode == 1 else 0)
        + (tok * (heads + 1) * rope * ROPE_OPS_PER_ELEM if do_rope else 0)
    )

    cores = max(1, int(geometry.compute_units))
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 2 * cores)))
    aic_pipe = read_role(geometry.resource_units or {}, aiv=None)
    aiv_pipe = read_role(geometry.resource_units or {}, aiv=0)

    plan_res: dict[str, float] = {
        "hbm": aic_gm_read + aiv_read + out_write,
        "mte1": l0_proj + l0_abs,
        "fixpipe": fix_proj + fix_abs,
        "vector": vec_ops,
        "mte3": out_write,
    }
    if aic_pipe == aiv_pipe:
        plan_res[aic_pipe] = aic_read + aiv_read
    else:
        plan_res[aic_pipe] = aic_read
        plan_res[aiv_pipe] = aiv_read
    if l2_hit > 0.0:
        plan_res["l2"] = l2_hit

    matrix_map: dict[str, float] = {}
    matrix_map[comp_dt] = matrix_map.get(comp_dt, 0.0) + proj_flops
    matrix_map["bf16"] = matrix_map.get("bf16", 0.0) + absorb_flops
    total_work = Work(matrix=matrix_map, resources={k: v for k, v in plan_res.items() if v > 0.0})

    def program(context):
        c_aic_pipe = read_role(context.resource_names, aiv=None)
        c_aiv_pipe = read_role(context.resource_names, aiv=0)
        share = 1.0 / float(cores)
        abs_owners = min(cores, heads)
        abs_share = 1.0 / float(abs_owners)
        is_a5 = "mte2_aic_dram" in context.service_modes
        setup = PROLOG_SETUP_MS if is_a5 else 0.0
        proj_caps: dict[str, float] = {}
        epilog_lat = 0.0
        cold_w_lat = 0.0
        if is_a5:
            is_v4 = w_mode in quant_modes
            q_width = float(heads * (nope + rope))
            wide_scale = min(
                PROLOG_WIDE_Q_MAX_SCALE,
                max(1.0, PROLOG_WIDE_Q_BASE_SCALE + PROLOG_WIDE_Q_SLOPE_SCALE * (q_width / PROLOG_WIDE_Q_DIM_PIVOT)),
            )
            cube_eff = (PROLOG_V4_SUBTILE_CUBE_EFF if is_v4 else PROLOG_V3_SUBTILE_CUBE_EFF) * wide_scale
            if "cube" in context.resource_names:
                cube_phys = context.resource_names["cube"]
                bf16_rate = float(context.rates.get(cube_phys, 0.0))
                if bf16_rate > 0.0:
                    proj_caps[cube_phys] = (cube_eff * bf16_rate) * share
            epilog_lat = float(tok) * (
                PROLOG_V4_EPILOG_MS_PER_TOK if is_v4 else PROLOG_V3_EPILOG_MS_PER_TOK
            )
            dram_entry = context.service_modes.get("mte2_aic_dram")
            dram_rate = (
                float(dram_entry[1]) if isinstance(dram_entry, tuple) else float(dram_entry or 0.0)
            )
            if dram_rate > 0.0:
                cold_w_lat = PROLOG_COLD_WEIGHT_OVERLAP * (w_bytes + w_uk_bytes) / dram_rate

        proj_gm = max(0.0, aic_gm_read - w_uk_bytes)
        abs_gm = aic_gm_read - proj_gm

        def per_core(core: int):
            aic_owned = context.owned(core, cores, ("cube", "mte1", "fixpipe", c_aic_pipe))
            aiv_owned = context.owned(core, cores, ("vector", c_aiv_pipe, "mte3"))
            allocs = tuple(
                context.allocate(
                    f"mlapo4_{core}_{pool}", pool, size,
                    unit_group(core, cores, context.pool_units.get(pool, cores)),
                )
                for pool, size in buf_sizes.items() if pool in context.pool_names
            )
            s_setup = Stage(
                "prolog_setup", latency=setup + cold_w_lat,
                acquire=allocs, keep_empty=(setup + cold_w_lat) <= 0.0,
            )
            proj_res: dict[str, float] = {
                "hbm": proj_gm * share, c_aic_pipe: l0_proj * share,
                "mte1": l0_proj * share, "fixpipe": fix_proj * share,
            }
            if l2_hit > 0.0:
                proj_res["l2"] = l2_hit * share
            s_proj = Stage(
                "prolog_projections",
                context.bind(Work(matrix={comp_dt: proj_flops * share}, resources=proj_res)),
                after=("prolog_setup",), placement=aic_owned, caps=proj_caps,
            )
            stages_list = [s_setup, s_proj]
            epilog_pred = ("prolog_projections",)
            if core < abs_owners:
                s_abs = Stage(
                    "prolog_wuk_absorb",
                    context.bind(Work(
                        matrix={"bf16": absorb_flops * abs_share},
                        resources={
                            "hbm": abs_gm * abs_share, c_aic_pipe: l0_abs * abs_share,
                            "mte1": l0_abs * abs_share, "fixpipe": fix_abs * abs_share,
                        },
                    )),
                    after=("prolog_projections",), placement=aic_owned,
                )
                stages_list.append(s_abs)
                epilog_pred = ("prolog_wuk_absorb",)
            aiv_res: dict[str, float] = {
                c_aiv_pipe: aiv_read * share, "vector": vec_ops * share,
                "hbm": (aiv_read + out_write) * share, "mte3": out_write * share,
            }
            s_epilog = Stage(
                "norm_rope_kv_scatter",
                context.bind(Work(resources=aiv_res)),
                after=epilog_pred, latency=epilog_lat, free=allocs, placement=aiv_owned,
            )
            stages_list.append(s_epilog)
            return KernelProgram(tuple(stages_list))

        return ParallelProgram(tuple(per_core(c) for c in range(cores)))

    return OperatorPlan(total_work, program, {
        "execution_class": "mixed_aic_aiv",
        "weight_quant_mode": w_mode,
        "kv_cache_quant_mode": kv_mode,
        "do_rope": do_rope,
        "kv_row_bytes": kv_row_bytes,
        "source_program": MLA_PROLOG_V4_HEADER_SOURCE,
    })


def recipes() -> dict[str, PlannedRecipe]:
    return {MLA_PROLOG_V4_RECIPE: PlannedRecipe(build)}
