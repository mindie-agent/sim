"""Arch35 grouped matmul, MX activation and wave MegaMoE.

[Role] Bind the paired FixPipe-to-UB producers, AIV epilogues and wave order.
[Invariants] MX scales are E8M0 per 32 elements. AIC-to-AIV transfers consume
  the direct route and never charge a GM accumulator write or read.
[Timeouts & Locks] Finite wave and owner work; no I/O or locks.
[Side-Effect Policy] Build immutable plans without querying device state.
Upstream-ABI: aclnnMegaMoe
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    ACTIVATION_OPS, FRACTAL_M_DIM, FRACTAL_N_DIM, GEMM_MODES, INT32_HANDOFF_BYTES_PER_ELEM,
    MX_DTYPES, MX_SCALE_GROUP_SIZE, PER_TOKEN_SCALE_BYTES, QUANT_OPS_PER_ELEMENT, ROUTED_MODES,
    SHIPPING_MEGAMOE_ACTIVATIONS, SHIPPING_SWIGLU_QUANT_ACTIVATIONS, SPARSE_GROUP_LIST_TYPE,
    StageDemand, activation_stages, capability_gap, contract_fields,
    blaze_tasks, ceil_div, element_bytes, fractal_issues, fractal_k0, make_plan, matrix_schedule,
    matrix_stages, tensor_bytes, validate_contract, vector_stages,
)

from .lowering import DualBatchMoEPipeline, mx_scale_bytes, routing_stages, validate_transport_work

TARGET_BACKEND: Literal["ascend"] = "ascend"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_infershape.cpp",
        source_sha256="78530fcdc4c9e441127781c3ad0d4941ece3fca8947445d7b708195e048962cd",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling_check.h",
        source_sha256="2da68d2404d401691b92c8f6af3a1f250c6e40eb10deaa11376b8920cd077237",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_kernel_arch35.h",
        source_sha256="ec37a6ab8f9568ff25985fdb71e37e2d21551acd140430a9931dac45b94e868d",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_mxquant.h",
        source_sha256="7c3dee59db38f4f3d89b8ce721634e457827838842e19cfe5d6634d0c174c60f",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h",
        source_sha256="799f7caf064cb129177529d70210bc8cbc94e2d314bc08cb6b13924b2f13c54e",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/common/cgmct/epilogue/cgmct_block_epilogue_swiglu_mx_quant.h",
        source_sha256="b98ff86940e80d42a6e33f087f71fa39324909f5f3e772e08a9a89ff56e846ee",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp",
        source_sha256="a6aec88582ccaa2de389ba5f483a2b56b7897f8fe92ad220960b08513ec01a35",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h",
        source_sha256="ebb1b7326d655c1d281eed8b835f44bf46b86fbb73d2a7e77642966f49f80297",
    ),
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        relative_path="quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp",
        source_sha256="84ac2b05ccfb7366de7706e4ae4c92c7bbec7b544cc4b12de75537538e7d2bc3",
    ),
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        relative_path="quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant.h",
        source_sha256="5626ba254394928a42c70e31907cb7269c99b9ea7759ba57e7ef0a0f6b07ee2f",
    ),
    SourceRef(
        repository="cann/ops-tensor",
        revision="c7798ac7b1b53a73a4a69eac2ca0b405393e09b0",
        relative_path="include/blaze/gemm/block/block_scheduler_grouped_matmul.h",
        source_sha256="3ee3e474c99195ef1ee86b18d59a3083096ab591e24c118e4790bb57664310b8",
    ),
)

ARCH_ID = ArchId("npu_arch_3510")
GMM2_LAG_MIN_TOKEN_NUM = 4096
GMM2_LAG_SPECIALIZATIONS = frozenset({
    "MegaMoeA16W16Wave", "MegaMoeA8W8Wave", "MegaMoeA8W4Wave",
})
MEGA_MOE_WAVE_TILE = 256
MEGA_MOE_PREFETCH_SUBTILE_M = 128
GMM1_WAVE_N_TILES_PER_CORE = 4
MEGA_MOE_META_INFO_ELEMENTS = 8
MEGA_MOE_META_INFO_BYTES = MEGA_MOE_META_INFO_ELEMENTS * INT32_HANDOFF_BYTES_PER_ELEM
FP4_DTYPES = frozenset({"fp4", "fp4_e2m1", "mxfp4"})
FP8_E4M3_COMPAT_DTYPES = frozenset({"fp8", "fp8_e4m3", "mxfp8", "hif8"})
A8W4_NZ_LAYOUTS = frozenset({"FRACTAL_NZ_C0_32", "FRACTAL_NZ", "NZ"})
ARCH35_MIN_EP_WORLD_SIZE = 2
ARCH35_MAX_EP_WORLD_SIZE = 1024
V2_GATED_ABI = "aclnnGroupedMatmulSwigluQuantV2"
V2_PERTOKEN_DTYPE_TUPLES = frozenset({
    ("int8", "int8", "int8"),
    ("fp8_e4m3", "fp8_e4m3", "fp8_e4m3"),
    ("fp8_e4m3", "fp8_e4m3", "int8"),
    ("fp8_e5m2", "fp8_e5m2", "fp8_e5m2"),
    ("fp8_e5m2", "fp8_e5m2", "int8"),
    ("hif8", "hif8", "hif8"),
})
_GATED_ACTIVATION_DIVISOR = 2
SHARED_DTYPE_ALIASES = {
    "fp8": "fp8_e4m3", "mxfp8": "fp8_e4m3", "fp4": "fp4_e2m1", "mxfp4": "fp4_e2m1",
}
SHARED_GEMM_PRECISIONS = frozenset({
    ("bf16", "bf16"), ("fp8_e4m3", "fp8_e4m3"), ("fp8_e5m2", "fp8_e5m2"),
    ("fp8_e4m3", "fp4_e2m1"), ("fp4_e2m1", "fp4_e2m1"),
})
AIV_INPUT_DTYPES = frozenset({"int32", "fp16", "bf16"})
MX_AIVS_PER_AIC = 2
MX_ABI = "aclnnGroupedMatmulSwigluMxQuant"
FLOAT_EPILOGUE_DTYPES = frozenset({"bf16", "fp16", "fp32"})
DEQUANT_GROUP_CUT_MIN_BINS = 32
DEQUANT_SPARSE_CUT_MIN_BINS = 64
DEQUANT_GROUP_CUT_MAX_AVG_ROWS = 16
DEQUANT_FULL_LOAD_MAX_WIDTH = 5120
DEQUANT_UB_RESERVE_BYTES = 1024
UB_BLOCK_BYTES = 32
_NOT_FULL_Y_SPLIT_BASE = 1
_NOT_FULL_INT32_X_SPLIT = 8
_NOT_FULL_FP16_X_SPLIT = 4
_NOT_FULL_WEIGHT_SCALE_SPLIT = 8
_NOT_FULL_QUANT_SCALE_SPLIT = 4
_NOT_FULL_QUANT_OFFSET_SPLIT = 4
_NOT_FULL_BIAS_4B_SPLIT = 8
_NOT_FULL_BIAS_2B_SPLIT = 4
_NOT_FULL_DOUBLE_BUFFER = 2
_NOT_FULL_TMP_BUFFER_SPLIT = 8
_BIAS_4B_THRESHOLD = 4
_HALF_DIVISOR = 2
_QUANT_MODE_DYNAMIC = 1


def _nz_tail_alignment(contract):
    return fractal_k0(contract.matrix_weight_dtype) if contract.padded_weight else 1


def _is_arch35_v2_pertoken(contract, fields) -> bool:
    if (contract.abi != V2_GATED_ABI or contract.activation_divisor != _GATED_ACTIVATION_DIVISOR
            or fields.get("quant_mode") != 0 or fields.get("dequant_mode") != 0):
        return False
    a = "fp8_e4m3" if contract.dtype_a == "fp8" else contract.dtype_a
    b = "fp8_e4m3" if contract.dtype_b == "fp8" else contract.dtype_b
    out = "fp8_e4m3" if contract.output_dtype == "fp8" else contract.output_dtype
    return (a, b, out) in V2_PERTOKEN_DTYPE_TUPLES


def _specialization(op, ctx, contract):
    a, b = contract.dtype_a, contract.dtype_b
    if a == b == contract.output_dtype == "bf16":
        return "MegaMoeA16W16Wave"
    if a in FP4_DTYPES and b in FP4_DTYPES:
        return "MegaMoeA4W4Wave"
    if a in FP8_E4M3_COMPAT_DTYPES and b in FP4_DTYPES:
        if contract.layout_b not in A8W4_NZ_LAYOUTS:
            raise capability_gap(op, ctx, f"arch35_mega_moe_a8w4_requires_nz_layout:{contract.layout_b}")
        return "MegaMoeA8W4Wave"
    if a == "fp8_e5m2" and b == "fp8_e5m2":
        if contract.padded_weight:
            raise capability_gap(op, ctx, "arch35_mega_moe_fp8_e5m2_requires_nd")
        return "MegaMoeA8W8Wave"
    if a in FP8_E4M3_COMPAT_DTYPES and b in FP8_E4M3_COMPAT_DTYPES:
        return "MegaMoeA8W8Wave"
    raise capability_gap(op, ctx, f"arch35_mega_moe_operand_pair:{a}:{b}")


def _validate(op, ctx, contract):
    mode = contract.mode
    if mode != "dequant_swiglu_quant" and ctx.effective_aic_cores < 1:
        raise capability_gap(op, ctx, "aic_owners")
    if mode != "dequant_swiglu_quant" and mode != "gmm_swiglu_quant" and not ctx.datapath.has_path("l0c", "gm"):
        raise capability_gap(op, ctx, "l0c_to_gm_fixpipe")
    if mode not in GEMM_MODES and ctx.effective_aiv_cores < 1:
        raise capability_gap(op, ctx, "aiv_owners")
    requested = contract.abi
    if requested == "aclnnMoeGroupedMatmulSwigluQuant":
        raise capability_gap(op, ctx, "arch35_xllm_int32_gm_abi")
    if mode in ROUTED_MODES | {"gmm_swiglu_quant"} and not ctx.datapath.has_l0c_to_ub:
        raise capability_gap(op, ctx, "l0c_to_ub_fixpipe")
    fields = contract_fields(op)
    allow_abi_ext = bool(fields.get("abi_variant_extension", False))
    if mode in ROUTED_MODES:
        if not (ARCH35_MIN_EP_WORLD_SIZE <= contract.ep <= ARCH35_MAX_EP_WORLD_SIZE):
            raise capability_gap(op, ctx, f"arch35_mega_moe_ep_world_size:{contract.ep}")
        if not allow_abi_ext and contract.activation not in SHIPPING_MEGAMOE_ACTIVATIONS:
            raise capability_gap(op, ctx, f"arch35_mega_moe_activation:{contract.activation}")
        _specialization(op, ctx, contract)
        if contract.shared_intermediate:
            compute, weight = contract.shared_compute_dtype, contract.shared_weight_dtype
            dtypes = tuple(SHARED_DTYPE_ALIASES.get(dtype, dtype) for dtype in (compute, weight))
            if dtypes not in SHARED_GEMM_PRECISIONS:
                raise ModelingError(f"grouped_moe_unsupported_shared_expert_precision:{compute}:{weight}")
    if mode in ROUTED_MODES | {"gmm_swiglu_quant"}:
        ctx.datapath.route("l0c", "ub")
    if mode == "gmm_swiglu_quant":
        if not allow_abi_ext and contract.activation not in SHIPPING_SWIGLU_QUANT_ACTIVATIONS:
            raise capability_gap(op, ctx, f"arch35_swiglu_quant_activation:{contract.activation}")
        is_mx_pair = (
            (contract.dtype_a in MX_DTYPES or contract.act_mx_scaled)
            and (contract.dtype_b in MX_DTYPES or contract.weight_mx_scaled)
            and contract.activation_divisor == _GATED_ACTIVATION_DIVISOR
        )
        if not (is_mx_pair or _is_arch35_v2_pertoken(contract, fields)):
            raise capability_gap(op, ctx, "arch35_paired_mx_operands")
    if mode == "dequant_swiglu_quant":
        if not allow_abi_ext and contract.activation not in SHIPPING_SWIGLU_QUANT_ACTIVATIONS:
            raise capability_gap(op, ctx, f"arch35_dequant_swiglu_activation:{contract.activation}")
        if contract.activation_divisor != _GATED_ACTIVATION_DIVISOR:
            raise capability_gap(op, ctx, f"arch35_dequant_swiglu_gated_activation:{contract.activation}")
        if contract.handoff_dtype not in AIV_INPUT_DTYPES:
            raise capability_gap(op, ctx, f"arch35_dequant_input:{contract.handoff_dtype}")


def _is_a8w4_routed(contract) -> bool:
    return (
        contract.mode in ROUTED_MODES
        and contract.dtype_a in FP8_E4M3_COMPAT_DTYPES
        and contract.dtype_b in FP4_DTYPES
    )


def _uses_gm_handoff(contract) -> bool:
    return contract.mode in ROUTED_MODES and (contract.topk_weights_type == 1 or _is_a8w4_routed(contract))


def fit_ub(contract, ctx):
    """Reserve the largest AIV share of a handoff and leave room for DBO credits."""
    capacity = ctx.datapath.usable_bytes("ub")
    if contract.enable_dbo:
        capacity -= 2 * min(capacity // 2, contract.top_k * INT32_HANDOFF_BYTES_PER_ELEM)
    # Ascend AIV upcasts BF16/FP16 handoffs to FP32 (4B/elem) inside UB for
    # gated activations and quantization scale math.
    working_elem = max(INT32_HANDOFF_BYTES_PER_ELEM, element_bytes(1, contract.handoff_dtype))
    full_n_tiling = contract.mode in ROUTED_MODES
    gm_handoff = _uses_gm_handoff(contract)
    if full_n_tiling:
        columns = min(contract.n, contract.base_n)
        col_align = contract.activation_divisor * (MX_SCALE_GROUP_SIZE if contract.uses_mx_scales else 1)
        if columns < contract.n:
            columns = columns // col_align * col_align
        row_bytes = columns * working_elem + (
            MEGA_MOE_META_INFO_BYTES if contract.topk_weights_type == 1 else 0
        )
        consumers = 1
    else:
        accum_bytes = contract.activation_divisor * working_elem
        columns = min(contract.intermediate, contract.base_n)
        # Reduce M before N so independently stored MX scales remain group aligned.
        if columns < contract.intermediate and contract.uses_mx_scales:
            columns = columns // MX_SCALE_GROUP_SIZE * MX_SCALE_GROUP_SIZE
        row_bytes = columns * accum_bytes
        consumers = MX_AIVS_PER_AIC
    if not columns:
        raise ModelingError("grouped_moe_arch35_ub_tile_width_invalid")
    rows = min(contract.base_m, max(contract.group_rows, default=1) or 1)
    if gm_handoff:
        # When AIC publishes GMM1 output to GM (topk_weights_type=1 or A8W4),
        # AIV reads back in bounded subtiles without shrinking the AIC cube grid.
        max_aiv_rows = capacity // row_bytes
        if max_aiv_rows < 1:
            raise ModelingError("grouped_moe_arch35_ub_tile_does_not_fit")
        sub_cap = MEGA_MOE_PREFETCH_SUBTILE_M if contract.topk_weights_type == 1 else rows
        return replace(contract, base_m=rows, base_n=columns), min(rows, sub_cap, max_aiv_rows) * row_bytes
    limit = consumers * (capacity // row_bytes)
    if rows > limit:
        rows = limit // FRACTAL_M_DIM * FRACTAL_M_DIM
    if not rows:
        raise ModelingError("grouped_moe_arch35_ub_tile_does_not_fit")
    return replace(contract, base_m=rows, base_n=columns), ceil_div(rows, consumers) * row_bytes


def _gated_stages(
    contract, ctx, name, after, metadata, *, start_core=0, owner_ready=None,
    pertoken_v2: bool = False, weight_scale_dtype: str = "fp32",
):
    """Lower paired MMADs with a UB credit and an AIV acknowledgement per tile."""
    tiled, slot_bytes = fit_ub(contract, ctx)
    full_n_tiling = contract.mode in ROUTED_MODES
    is_a8w4 = _is_a8w4_routed(contract)
    divisor = contract.activation_divisor
    schedule_n = contract.n if full_n_tiling else contract.intermediate
    n_alignment = (divisor if full_n_tiling else 1) * (MX_SCALE_GROUP_SIZE if contract.uses_mx_scales else 1)
    schedule = matrix_schedule(tiled, ctx, contract.group_rows, schedule_n,
                               contract.k, start_core=start_core, n_tail_alignment=n_alignment)
    gmm, activation = name + "gmm1", name + "swiglu_quant"
    gmm_ends, stores, stages = [], [], []
    pairs = dict(ctx.datapath.aic_aiv_pairs)
    previous = {} if owner_ready is None else owner_ready
    for group in schedule.groups:
        split_group = group if tiled.tail_split else replace(group, tail_tiles=0)
        for task, rows, columns, column in blaze_tasks(
            split_group, schedule_n, tiled.base_m, tiled.base_n,
            tiled.m_tail_cnt, tiled.n_tail_cnt, n_alignment,
        ):
            owner = (group.start_core + task) % ctx.effective_aic_cores
            paired = pairs.get(owner, ())
            if contract.mode == "gmm_swiglu_quant":
                if len(paired) != MX_AIVS_PER_AIC or max(paired) >= ctx.effective_aiv_cores:
                    raise ModelingError("grouped_moe_arch35_mx_requires_paired_aivs")
                consumers = paired
            elif is_a8w4:
                # A8W4 uses AIV0 for FP4->FP8 weight prologue and AIV1 for activation epilogue.
                fallback = owner * ctx.effective_aiv_cores // ctx.effective_aic_cores
                consumers = (paired[1] if len(paired) >= MX_AIVS_PER_AIC else (paired[0] if paired else fallback),)
            else:
                consumers = (paired[0] if paired else owner * ctx.effective_aiv_cores // ctx.effective_aic_cores,)
            key = f"e{group.expert}:t{task}"
            deps = (*after, *previous.get(owner, ()))
            body, fixpipe, done = _paired_tile(
                contract, gmm, activation, key, deps, rows, columns, column,
                owner, consumers, slot_bytes, metadata,
                full_n_tiling=full_n_tiling, pertoken_v2=pertoken_v2, weight_scale_dtype=weight_scale_dtype,
            )
            stages.extend(body)
            gmm_ends.append(fixpipe)
            stores.extend(done)
            previous[owner] = done
    flops_scale = 1 if full_n_tiling else divisor
    stages.append(StageDemand(gmm, tuple(gmm_ends) or after, metadata=(*metadata,
        ("useful_flops", flops_scale * schedule.useful_flops),
        ("issued_fractals", flops_scale * schedule.issued_fractals),
        ("max_waves", schedule.max_waves), ("wave_tail_imbalance", schedule.wave_tail_imbalance),
    )))
    if pertoken_v2 and contract.m:
        # CANN kernel_gmm_swiglu_pertoken_quant.h:449 runs SyncAll across all
        # tiles' FP32 SwiGLU workspace writes before pass-2 per-token quant.
        sync_name = activation + ":sync_all"
        stages.append(StageDemand(sync_name, (gmm, *stores), metadata=metadata))
        pass2_name = activation + ":pertoken_quant"
        ws_row_bytes = contract.intermediate * INT32_HANDOFF_BYTES_PER_ELEM
        out_row_bytes = element_bytes(contract.intermediate, contract.quant_dtype) + PER_TOKEN_SCALE_BYTES
        stages.extend(vector_stages(
            pass2_name, (sync_name,), ctx, contract.group_rows,
            read=ws_row_bytes,
            compute=contract.intermediate * QUANT_OPS_PER_ELEMENT,
            write=out_row_bytes,
            metadata=metadata,
        ))
        stages.append(StageDemand(activation, (pass2_name,), metadata=metadata))
        return stages, schedule
    if not contract.uses_mx_scales and contract.quant_dtype not in FLOAT_EPILOGUE_DTYPES and contract.m:
        scale_bytes = contract.m * PER_TOKEN_SCALE_BYTES
        scales = StageDemand(activation + ":scales", tuple(stores), Work(resources={
            "hbm": scale_bytes, "l2": scale_bytes, "mte3": scale_bytes,
        }), metadata=metadata)
        stages.append(scales)
        stores = [scales.name]
    stages.append(StageDemand(activation, (gmm, *stores), metadata=metadata))
    return stages, schedule


def _paired_tile(contract, gmm, activation, key, after, rows, columns, column,
                 owner, consumers, slot_bytes, metadata, *,
                 full_n_tiling: bool = False, pertoken_v2: bool = False, weight_scale_dtype: str = "fp32"):
    divisor = contract.activation_divisor
    gm_handoff = full_n_tiling and _uses_gm_handoff(contract)
    prefetch_weights = full_n_tiling and contract.topk_weights_type == 1
    full_cols = columns if full_n_tiling else columns * divisor
    out_cols = columns // divisor if full_n_tiling else columns
    out_col = column // divisor if full_n_tiling else column
    input_scaled = contract.dtype_a == "int8" and (contract.mode in ROUTED_MODES or pertoken_v2)
    weight_k, weight_n = contract.k, out_cols
    if contract.padded_weight:
        k0 = fractal_k0(contract.dtype_b)
        k_align, n_align = (k0, FRACTAL_N_DIM) if contract.transpose_weight else (FRACTAL_M_DIM, k0)
        weight_k = ceil_div(weight_k, k_align) * k_align
        weight_n = ceil_div(weight_n, n_align) * n_align
    a_bytes = tensor_bytes(
        rows, contract.k, contract.dtype_a, scaled=input_scaled,
        mx_scaled=contract.act_mx_scaled, mx_padded=contract.act_mx_padded,
    )
    b_bytes = tensor_bytes(
        weight_n, weight_k, contract.dtype_b,
        mx_scaled=contract.weight_mx_scaled, mx_padded=contract.weight_mx_padded,
    )
    operand_bytes = (a_bytes + divisor * b_bytes) if full_n_tiling else divisor * (a_bytes + b_bytes)
    if full_n_tiling:
        _, issued_flops = fractal_issues(rows, full_cols, contract.k, contract.compute_dtype)
    else:
        _, issued = fractal_issues(rows, out_cols, contract.k, contract.compute_dtype)
        issued_flops = divisor * issued
    fix_row_bytes = element_bytes(full_cols, contract.handoff_dtype)
    fix_bytes = rows * fix_row_bytes
    quantized = contract.quant_dtype not in FLOAT_EPILOGUE_DTYPES
    if pertoken_v2:
        # Pass 1 dequantizes with channel scales + SwiGLU and writes FP32 GM workspace.
        ops_per_row = out_cols * ACTIVATION_OPS[contract.activation]
        output_per_row = out_cols * INT32_HANDOFF_BYTES_PER_ELEM
    else:
        extra_ops = (QUANT_OPS_PER_ELEMENT if quantized else 0) + (1 if prefetch_weights else 0)
        ops_per_row = out_cols * (ACTIVATION_OPS[contract.activation] + extra_ops)
        output_per_row = element_bytes(out_cols, contract.quant_dtype)
        if contract.uses_mx_scales:
            end_column = out_col + out_cols
            # Interior tiles write valid groups; only the last tile writes row padding.
            output_per_row += mx_scale_bytes(end_column, padded=end_column == contract.intermediate)
            output_per_row -= mx_scale_bytes(out_col, padded=False)
    base = f"{gmm}:{key}"
    read, cube, flag = (f"{base}:{phase}" for phase in ("read", "cube", "fixpipe"))
    stages = [
        StageDemand(read, after, Work(resources={
            "hbm": operand_bytes, "l2": operand_bytes, "mte2_aic": operand_bytes,
        }), "aic", owner, metadata),
        StageDemand(cube, (read,), Work({contract.compute_dtype: issued_flops},
                                        {"mte1": operand_bytes}), "aic", owner, metadata),
    ]
    credits, done, epilogue = [], [], []
    rows_per_aiv = ceil_div(rows, len(consumers))
    tile_ch_scale_bytes = element_bytes(full_cols, weight_scale_dtype) if pertoken_v2 else 0
    for subblock, aiv in enumerate(consumers):
        prefix = f"{activation}:{key}:aiv{aiv}"
        credit, vector, store, sync = (f"{prefix}:{phase}" for phase in ("credit", "vector", "store", "sync"))
        active = max(0, min(rows_per_aiv, rows - subblock * rows_per_aiv))
        slot = (f"{gmm}:aic{owner}:handoff:aiv{aiv}", slot_bytes)
        tag = (*metadata, ("aiv_consumer", aiv), ("subblock", subblock), ("rows", active))
        if active:
            if gm_handoff:
                gm_read = f"{prefix}:read"
                rd_bytes = active * (fix_row_bytes + (MEGA_MOE_META_INFO_BYTES if prefetch_weights else 0))
                epilogue.extend((
                    StageDemand(credit, (flag,), kind="aiv", owner=aiv, metadata=tag, ub_acquire=slot),
                    StageDemand(gm_read, (credit,), Work(resources={
                        "hbm": rd_bytes, "l2": rd_bytes, "mte2_aiv": rd_bytes,
                    }), "aiv", aiv, tag),
                ))
                vec_deps = (gm_read,)
            else:
                stages.append(StageDemand(credit, after, kind="aiv", owner=aiv, metadata=tag, ub_acquire=slot))
                credits.append(credit)
                vec_deps = (flag,)
                if pertoken_v2 and tile_ch_scale_bytes:
                    scale_stage = f"{prefix}:weight_scale"
                    epilogue.append(StageDemand(scale_stage, (flag,), Work(resources={
                        "hbm": tile_ch_scale_bytes, "l2": tile_ch_scale_bytes, "mte2_aiv": tile_ch_scale_bytes,
                    }), "aiv", aiv, tag))
                    vec_deps = (scale_stage,)
            output = active * output_per_row
            epilogue.extend((
                StageDemand(vector, vec_deps, Work(resources={"vector": active * ops_per_row}), "aiv", aiv, tag),
                StageDemand(store, (vector,), Work(resources={
                    "hbm": output, "l2": output, "mte3": output,
                }), "aiv", aiv, tag),
            ))
        # The idle subblock still acknowledges both gate/up buffers to the AIC.
        epilogue.append(StageDemand(sync, (store if active else flag,), kind="aiv", owner=aiv,
                                    metadata=tag, ub_release=slot if active else None))
        done.append(sync)
    if gm_handoff:
        stages.extend((
            StageDemand(flag, (cube,), Work(resources={
                "hbm": fix_bytes, "l2": fix_bytes, "fixpipe": fix_bytes,
            }), "aic", owner, metadata),
            *epilogue,
        ))
    else:
        # Join the independent UB grants before publishing one accumulator tile.
        admitted = f"{base}:admitted"
        stages.extend((
            StageDemand(admitted, tuple(credits), metadata=metadata),
            StageDemand(flag, (cube, admitted), Work(resources={"fixpipe": fix_bytes}), "aic", owner, metadata),
            *epilogue,
        ))
    return stages, flag, tuple(done)


def _wave_histograms(contract, aic_cores):
    """Cut expert rows at the common GMM1 M-group budget for a wave."""
    m_group = 1 if contract.wave_rows else MEGA_MOE_WAVE_TILE
    groups_per_wave = contract.wave_rows or max(
        ceil_div(aic_cores * GMM1_WAVE_N_TILES_PER_CORE, ceil_div(contract.n, MEGA_MOE_WAVE_TILE)),
        ceil_div(aic_cores, ceil_div(contract.k, MEGA_MOE_WAVE_TILE)),
    )
    batch = [0] * len(contract.group_rows)
    used = 0
    for expert, rows in enumerate(contract.group_rows):
        while rows:
            count = min(rows, (groups_per_wave - used) * m_group)
            batch[expert] += count
            rows -= count
            used += ceil_div(count, m_group)
            if used == groups_per_wave:
                yield tuple(batch)
                batch = [0] * len(batch)
                used = 0
    if used:
        yield tuple(batch)


def _routed_stages(ctx, specialization, contract, name, after, metadata, transport_builder):
    dispatch, combine = name + "dispatch", name + "combine"
    cohort = tuple((key, value) for key, value in metadata if key in {"cohort_id", "slot"})
    wave_count = sum(1 for _ in _wave_histograms(contract, ctx.effective_aic_cores))
    shared = contract.shared_intermediate and contract.tokens
    if not wave_count:
        stages = routing_stages(dispatch, after, contract, metadata, transport_builder)
        deps = (dispatch,)
        if shared:
            deps += (_shared_stages(stages, ctx, contract, name, after, cohort),)
        return stages + routing_stages(combine, deps, contract, metadata, transport_builder)
    stages, dispatch_ends = [], []
    last_dispatch = after
    batch_dispatch = batch_combine = 0
    if wave_count > 1 and not transport_builder:
        # A batch total has no expert placement. Preserve its complete wire
        # transfer at the batch boundary until a caller supplies a routed cut.
        batch_dispatch, batch_combine = contract.transport_bytes
        if batch_dispatch:
            stages.extend(routing_stages(dispatch, after, contract, metadata, None))
    amounts = contract.transport_bytes if wave_count == 1 else (
        (None, None) if transport_builder else (0, 0)
    )
    owner_ready = {}
    gmm1_core = gmm2_core = 0
    gmm1_ends, activation_ends, waves = [], [], []
    gmm2_ends, wave_combines = [], []
    lag = specialization in GMM2_LAG_SPECIALIZATIONS and contract.tokens >= GMM2_LAG_MIN_TOKEN_NUM

    def emit_gmm2(wave_idx, extra_deps=()):
        nonlocal gmm2_core
        w_prefix, w_active, w_route_md = waves[wave_idx]
        deps = (activation_ends[wave_idx], *gmm2_ends[-1:], *extra_deps)
        schedule = matrix_schedule(
            contract, ctx, w_active.group_rows, contract.k, contract.intermediate, start_core=gmm2_core,
            n_tail_alignment=_nz_tail_alignment(contract),
        )
        gmm2_core = schedule.groups[-1].end_block % ctx.effective_aic_cores
        gmm2_ends.append(w_prefix + "gmm2")
        stages.extend(matrix_stages(
            gmm2_ends[-1], deps, contract, schedule, contract.intermediate,
            contract.output_dtype, input_dtype=contract.quant_dtype,
            input_scaled=contract.quant_dtype == "int8", metadata=(*cohort, ("wave", wave_idx)),
        ))
        if wave_count > 1 and not batch_combine:
            wave_combines.append(w_prefix + "combine")
            stages.extend(routing_stages(wave_combines[-1], (gmm2_ends[-1],), w_active,
                                         w_route_md, transport_builder,
                                         publish_output=False, read_counts=False))

    offset = 0
    for wave, rows in enumerate(_wave_histograms(contract, ctx.effective_aic_cores)):
        prefix = name if wave_count == 1 else f"{name}wave{wave}_"
        end = offset + sum(rows)
        tokens = contract.tokens * end // contract.m - contract.tokens * offset // contract.m
        active = replace(contract, group_rows=rows, m=end - offset, tokens=tokens, transport_bytes=amounts)
        route_metadata = (*metadata, ("wave", wave), ("wave_row_start", offset), ("wave_row_end", end),
                          ("wave_expert_rows", rows))
        offset = end
        ready = prefix + "dispatch"
        if batch_dispatch:
            stages.append(StageDemand(ready, (dispatch,), metadata=route_metadata))
        else:
            stages.extend(routing_stages(ready, last_dispatch, active, route_metadata, transport_builder,
                                         read_counts=wave == 0))
            last_dispatch = (ready,)
            dispatch_ends.append(ready)
        wave_metadata = (*cohort, ("wave", wave))
        gmm1_after = (ready, gmm2_ends[-1]) if (lag and gmm2_ends) else (ready,)
        body, schedule = _gated_stages(
            active, ctx, prefix, gmm1_after, wave_metadata, start_core=gmm1_core, owner_ready=owner_ready,
        )
        gmm1_core = schedule.groups[-1].end_block % ctx.effective_aic_cores
        stages.extend(body)
        gmm1_ends.append(prefix + "gmm1")
        activation_ends.append(prefix + "swiglu_quant")
        waves.append((prefix, active, route_metadata))
        if lag and wave > 0:
            # CANN wave loop: GMM1(w) -> GMM2(w-1) -> GMM1(w+1).
            emit_gmm2(wave - 1, (gmm1_ends[wave],))
    if lag:
        emit_gmm2(wave_count - 1)
    else:
        for wave in range(wave_count):
            emit_gmm2(wave)
    if wave_count == 1:
        combine_after = (name + "gmm2",)
    else:
        if transport_builder:
            validate_transport_work(stages, contract.transport_bytes)
        if not batch_dispatch:
            stages.append(StageDemand(dispatch, tuple(dispatch_ends), metadata=metadata))
        for phase, deps in (("gmm1", gmm1_ends), ("swiglu_quant", activation_ends), ("gmm2", gmm2_ends)):
            stages.append(StageDemand(name + phase, tuple(deps), metadata=metadata))
        combine_after = (name + "gmm2", *wave_combines)
        # Wave combines have already read their routed rows. Batch combine
        # owns those reads when only the aggregate transfer is available.
        contract = replace(contract, m=contract.m if batch_combine else 0,
                           transport_bytes=(0, batch_combine))
        transport_builder = None
    if shared:
        combine_after += (_shared_stages(stages, ctx, contract, name, after, cohort),)
    stages.extend(routing_stages(combine, combine_after, contract, metadata, transport_builder))
    return stages


def _shared_stages(stages, ctx, contract, name, after, metadata):
    """One local shared expert with independent weights and the same token view."""
    compute = contract.shared_compute_dtype
    input_dtype = contract.dtype_a if compute == contract.compute_dtype else ("int8" if compute == "w4a8" else compute)
    handoff = "int32" if compute in {"int8", "w4a8"} else (
        "fp32" if compute in MX_DTYPES else contract.output_dtype
    )
    shared = replace(
        contract, group_rows=(contract.tokens,), m=contract.tokens,
        n=contract.activation_divisor * contract.shared_intermediate,
        compute_dtype=compute, dtype_a=input_dtype, dtype_b=contract.shared_weight_dtype,
        quant_dtype=input_dtype, handoff_dtype=handoff,
        act_mx_scaled=input_dtype in MX_DTYPES,
        weight_mx_scaled=contract.shared_weight_dtype in MX_DTYPES,
        act_mx_padded=True,
        weight_mx_padded=True,
        topk_weights_type=0,
    )
    prefix = name + "shared_"
    body, _ = _gated_stages(shared, ctx, prefix, after, metadata)
    stages.extend(body)
    schedule = matrix_schedule(shared, ctx, shared.group_rows, shared.k, shared.intermediate,
                               n_tail_alignment=_nz_tail_alignment(shared))
    end = prefix + "gmm2"
    stages.extend(matrix_stages(
        end, (prefix + "swiglu_quant",), shared, schedule, shared.intermediate,
        shared.output_dtype, input_dtype=shared.quant_dtype,
        input_scaled=shared.quant_dtype == "int8", metadata=metadata,
    ))
    return end


def _not_full_dequant_workspace_per_row(
    ctx, width: int, is_int32: bool, has_bias: bool, bias_elem_bytes: int, fields: dict,
) -> int:
    quant_mode = fields.get("quant_mode", _QUANT_MODE_DYNAMIC)
    ub_available = ctx.datapath.usable_bytes("ub") - DEQUANT_UB_RESERVE_BYTES - UB_BLOCK_BYTES
    ub_split = (
        _NOT_FULL_Y_SPLIT_BASE
        + (_NOT_FULL_INT32_X_SPLIT if is_int32 else _NOT_FULL_FP16_X_SPLIT)
        + (_NOT_FULL_WEIGHT_SCALE_SPLIT if is_int32 else 0)
        + _NOT_FULL_QUANT_SCALE_SPLIT
    )
    if fields.get("has_quant_offset"):
        ub_split += _NOT_FULL_QUANT_OFFSET_SPLIT
    if has_bias:
        ub_split += _NOT_FULL_BIAS_4B_SPLIT if bias_elem_bytes >= _BIAS_4B_THRESHOLD else _NOT_FULL_BIAS_2B_SPLIT
    ub_split = ub_split * _NOT_FULL_DOUBLE_BUFFER + _NOT_FULL_TMP_BUFFER_SPLIT
    ub_factor = (ub_available // ub_split) // UB_BLOCK_BYTES * UB_BLOCK_BYTES
    if ub_factor < 1:
        raise ModelingError("grouped_moe_dequant_not_full_does_not_fit")
    loops = ceil_div(width, ub_factor)
    if quant_mode != _QUANT_MODE_DYNAMIC or loops <= 1:
        return 0
    balanced = ceil_div(ceil_div(width, loops), UB_BLOCK_BYTES) * UB_BLOCK_BYTES
    if 0 < balanced <= ub_factor and ceil_div(width, balanced) == loops:
        ub_factor = balanced
    return (loops - 1) * ub_factor * INT32_HANDOFF_BYTES_PER_ELEM


def _dequant_group_slices(count: int, owners: int, *, special: bool, not_full: bool, group: int):
    if special:
        yield group % owners, 0, count
        return
    if not_full:
        # Wide-row kernel distributes rows cyclically across all active AIVs.
        base, rem = divmod(count, owners)
        start = 0
        for owner in range(min(owners, count)):
            size = base + (1 if owner < rem else 0)
            yield owner, start, size
            start += size
        return
    step = ceil_div(count, owners)
    for part, start in enumerate(range(0, count, step)):
        yield part, start, min(step, count - start)


def _dequant_stages(contract, ctx, fields):
    name = "dequant_swiglu_quant"
    if not contract.m:
        return [StageDemand(name)]
    rows, width = contract.group_rows, contract.intermediate
    allocated = contract.allocated_rows
    is_int32 = contract.handoff_dtype == "int32"
    channels = element_bytes(contract.n, "fp32") if is_int32 else 0
    has_bias = is_int32 and bool(fields.get("has_bias", False))
    bias_dtype = str(fields.get("bias_dtype", "fp32")).lower()
    has_quant_scale = bool(fields.get("has_quant_scale", False))
    quant_scale_dtype = str(fields.get("quant_scale_dtype", "fp32")).lower()
    has_quant_offset = bool(fields.get("has_quant_offset", False))
    quant_offset_dtype = str(fields.get("quant_offset_dtype", "fp32")).lower()
    quant_mode = fields.get("quant_mode", _QUANT_MODE_DYNAMIC)
    dynamic_quant = quant_mode == _QUANT_MODE_DYNAMIC
    bias_bytes = element_bytes(contract.n, bias_dtype) if has_bias else 0
    bias_elem_bytes = element_bytes(1, bias_dtype) if has_bias else 0
    quant_scale_bytes = element_bytes(width, quant_scale_dtype) if has_quant_scale else 0
    quant_offset_bytes = element_bytes(width, quant_offset_dtype) if has_quant_offset else 0
    group_param_bytes = channels + bias_bytes + quant_scale_bytes + quant_offset_bytes
    max_owners = min(ctx.effective_aiv_cores, allocated)
    sparse = fields.get("group_list_type") == SPARSE_GROUP_LIST_TYPE
    min_groups = DEQUANT_SPARSE_CUT_MIN_BINS if sparse else DEQUANT_GROUP_CUT_MIN_BINS
    special = (width <= DEQUANT_FULL_LOAD_MAX_WIDTH and len(rows) >= min_groups
               and allocated // len(rows) <= DEQUANT_GROUP_CUT_MAX_AVG_ROWS
               and (sparse or not (has_bias or has_quant_scale or has_quant_offset)))
    owners = max_owners
    not_full = width > DEQUANT_FULL_LOAD_MAX_WIDTH
    ws_per_row = 0
    if not_full:
        ws_per_row = _not_full_dequant_workspace_per_row(ctx, width, is_int32, has_bias, bias_elem_bytes, fields)
    elif not special:
        # Full-load CANN tiling uses two row slots for x, activation scales
        # and y, plus the FP32 vector scratch, weight-scale/quant-scale planes,
        # and optional bias plane.
        aligned = ceil_div(element_bytes(width, "fp32"), UB_BLOCK_BYTES) * UB_BLOCK_BYTES
        y_aligned = ceil_div(width, UB_BLOCK_BYTES) * UB_BLOCK_BYTES
        row_bytes = (
            _NOT_FULL_DOUBLE_BUFFER * (_GATED_ACTIVATION_DIVISOR * aligned + UB_BLOCK_BYTES + y_aligned)
            + UB_BLOCK_BYTES + aligned
        )
        if has_bias:
            row_bytes += aligned if bias_elem_bytes >= _BIAS_4B_THRESHOLD else (aligned // _HALF_DIVISOR)
        remaining = ctx.datapath.usable_bytes("ub") - DEQUANT_UB_RESERVE_BYTES - _GATED_ACTIVATION_DIVISOR * aligned - aligned
        tile_rows = remaining // row_bytes
        if tile_rows < 1:
            raise ModelingError("grouped_moe_dequant_full_load_does_not_fit")
        owners = min(owners, ceil_div(allocated, tile_rows))
    stages, ends, offset = [], {}, 0
    row_scale = PER_TOKEN_SCALE_BYTES if (is_int32 and fields.get("has_activation_scale", True)) else 0
    row_x_bytes = element_bytes(contract.n, contract.handoff_dtype)
    for group, count in enumerate(rows):
        if not count:
            continue
        for owner, start, size in _dequant_group_slices(
            count, owners, special=special, not_full=not_full, group=group,
        ):
            base = f"{name}:group{group}:aiv{owner}"
            after = (ends[owner],) if owner in ends else ()
            tag = (("group", group), ("row_start", offset + start), ("rows", size),
                   ("weight_scale_bytes", channels), ("special_core_cut", special))
            param_reloads = size if not_full else 1
            inputs = size * (row_x_bytes + row_scale) + param_reloads * group_param_bytes
            output = tensor_bytes(size, width, contract.quant_dtype, scaled=dynamic_quant)
            bias_ops = size * contract.n if has_bias else 0
            smooth_ops = size * width if has_quant_scale else 0
            offset_ops = size * width if has_quant_offset else 0
            if ws_per_row > 0:
                ws_bytes = size * ws_per_row
                pass1_ops = size * width * ACTIVATION_OPS[contract.activation] + bias_ops
                pass2_ops = size * width * QUANT_OPS_PER_ELEMENT + smooth_ops + offset_ops
                read, vec1, ws_st, ws_rd, vec2, store = (
                    f"{base}:{phase}" for phase in ("read", "vector_p1", "ws_store", "ws_read", "vector_p2", "store")
                )
                stages.extend((
                    StageDemand(read, after, Work(resources={"hbm": inputs, "l2": inputs, "mte2_aiv": inputs}),
                                "aiv", owner, tag),
                    StageDemand(vec1, (read,), Work(resources={"vector": pass1_ops}), "aiv", owner, tag),
                    StageDemand(ws_st, (vec1,), Work(resources={"hbm": ws_bytes, "l2": ws_bytes, "mte3": ws_bytes}),
                                "aiv", owner, tag),
                    StageDemand(ws_rd, (ws_st,), Work(resources={"hbm": ws_bytes, "l2": ws_bytes, "mte2_aiv": ws_bytes}),
                                "aiv", owner, tag),
                    StageDemand(vec2, (ws_rd,), Work(resources={"vector": pass2_ops}), "aiv", owner, tag),
                    StageDemand(store, (vec2,), Work(resources={"hbm": output, "l2": output, "mte3": output}),
                                "aiv", owner, tag),
                ))
            else:
                read, vector, store = (f"{base}:{phase}" for phase in ("read", "vector", "store"))
                ops = (
                    size * width * (ACTIVATION_OPS[contract.activation] + QUANT_OPS_PER_ELEMENT)
                    + bias_ops + smooth_ops + offset_ops
                )
                stages.extend((
                    StageDemand(read, after, Work(resources={"hbm": inputs, "l2": inputs, "mte2_aiv": inputs}),
                                "aiv", owner, tag),
                    StageDemand(vector, (read,), Work(resources={"vector": ops}), "aiv", owner, tag),
                    StageDemand(store, (vector,), Work(resources={"hbm": output, "l2": output, "mte3": output}),
                                "aiv", owner, tag),
                ))
            ends[owner] = store
        offset += count
    return [*stages, StageDemand(name, tuple(ends.values()))]


@dataclass(frozen=True, slots=True)
class _Arch35Strategy:
    arch_id: ArchId = ARCH_ID
    abi: str = "aclnnMegaMoe"
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        try:
            _validate(op, ctx, validate_contract(op, ctx))
        except (ValueError, TypeError):
            return False
        return True

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        if ctx.compute_arch.arch_id != self.arch_id:
            raise capability_gap(op, ctx, "arch35_compute_identity")
        contract = validate_contract(op, ctx)
        _validate(op, ctx, contract)
        mode = contract.mode
        metadata = {"mode": mode, "enable_dbo": contract.enable_dbo}
        if mode in GEMM_MODES:
            schedule = matrix_schedule(contract, ctx, contract.group_rows, contract.n, contract.k,
                                       n_tail_alignment=_nz_tail_alignment(contract))
            stages = matrix_stages("gmm", (), contract, schedule, contract.k, contract.output_dtype)
            metadata.update(useful_flops=schedule.useful_flops, issued_fractals=schedule.issued_fractals,
                            max_waves=schedule.max_waves, wave_tail_imbalance=schedule.wave_tail_imbalance)
            implementation = "aclnnGroupedMatmul"
        elif mode == "dequant_swiglu_quant":
            fields = contract_fields(op)
            use_dequant = contract.handoff_dtype == "int32" or contract.intermediate > DEQUANT_FULL_LOAD_MAX_WIDTH
            stages = (
                _dequant_stages(contract, ctx, fields) if use_dequant
                else activation_stages(
                    "dequant_swiglu_quant", (), ctx, contract, contract.group_rows,
                    has_activation_scale=fields.get("has_activation_scale", True),
                )
            )
            implementation = "aclnnDequantSwigluQuant"
        elif mode == "gmm_swiglu_quant":
            fields = contract_fields(op)
            is_pertoken_v2 = _is_arch35_v2_pertoken(contract, fields)
            handoff = "int32" if (is_pertoken_v2 and contract.dtype_a == "int8") else "fp32"
            contract = replace(
                contract,
                handoff_dtype=handoff,
                quant_dtype=contract.output_dtype if is_pertoken_v2 else contract.quant_dtype,
            )
            wt_scale_dtype = str(fields.get("weight_scale_dtype", "fp32")).lower()
            stages, schedule = _gated_stages(
                contract, ctx, "", (), (),
                pertoken_v2=is_pertoken_v2, weight_scale_dtype=wt_scale_dtype,
            )
            metadata.update(
                useful_flops=_GATED_ACTIVATION_DIVISOR * schedule.useful_flops,
                issued_fractals=_GATED_ACTIVATION_DIVISOR * schedule.issued_fractals,
                gm_handoff_eliminated_bytes=(
                    _GATED_ACTIVATION_DIVISOR * contract.m * contract.n * INT32_HANDOFF_BYTES_PER_ELEM
                ),
            )
            implementation = contract.abi or MX_ABI
        else:
            specialization = _specialization(op, ctx, contract)
            emit = partial(_routed_stages, ctx, specialization)
            transport_builder = contract_fields(op).get("cohort_transport_builder")
            if contract.dbo_split is not None:
                stages = DualBatchMoEPipeline(contract.dbo_split).stages(
                    contract, ctx, emit, transport_builder=transport_builder
                )
            else:
                stages = emit(contract, "", (), (), transport_builder) if (
                    contract.m or contract.tokens or any(contract.transport_bytes) or transport_builder
                ) else []
            metadata["wave_specialization"] = specialization
            metadata["topk_weights_type"] = contract.topk_weights_type
            if _uses_gm_handoff(contract):
                metadata["gm_handoff_bytes"] = (
                    _GATED_ACTIVATION_DIVISOR * contract.m * element_bytes(contract.n, contract.handoff_dtype)
                )
                metadata["gm_handoff_eliminated_bytes"] = 0
            else:
                metadata["gm_handoff_eliminated_bytes"] = (
                    _GATED_ACTIVATION_DIVISOR * contract.m * contract.n * INT32_HANDOFF_BYTES_PER_ELEM
                )
            implementation = self.abi
        return make_plan(implementation, ctx, stages, self.sources, metadata=metadata)


STRATEGY: ArchStrategy = _Arch35Strategy()
