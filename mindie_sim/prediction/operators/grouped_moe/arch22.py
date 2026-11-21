"""Arch22 grouped matmul, dequantization and MegaMoE.

[Role] Bind Catlass/Blaze AIC schedules and AIV epilogues via GM.
[Invariants] Gated INT32 output crosses the kernel boundary through GM.
  The Cube and vector pipelines have independent physical owner placement.
[Timeouts & Locks] Bounded per-owner planning with no I/O or locks.
[Side-Effect Policy] Build plans and descriptors from declared operands only.
Upstream-ABI: aclnnGroupedMatmul
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    FUSED_MC2_SHARED_MODE, GEMM_MODES, INT32_HANDOFF_BYTES_PER_ELEM, MX_DTYPES,
    ROUTED_MODES, activation_stages, capability_gap,
    PER_TOKEN_SCALE_BYTES, StageDemand, contract_fields, element_bytes, make_plan, matrix_schedule,
    matrix_stages, validate_contract,
)

from .lowering import DualBatchMoEPipeline, routing_stages

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/catlass",
        revision="dd03bc0998cbcd67c982e7ab13df6ee15780c6f3",
        relative_path="include/catlass/catlass.hpp",
        source_sha256="b19458b6430c303df94c1d2f8c945b3f3b20ff0edb168c945ab0777082cff144",
    ),
    SourceRef(
        repository="cann/catlass",
        revision="dd03bc0998cbcd67c982e7ab13df6ee15780c6f3",
        relative_path="include/catlass/gemm/tile/tile_mmad.hpp",
        source_sha256="b5a0b24ecd9f71866cd8e3fe63b52fc203aa741e0ebf2306a9457d7103f78231",
    ),
    SourceRef(
        repository="cann/ops-tensor",
        revision="c7798ac7b1b53a73a4a69eac2ca0b405393e09b0",
        relative_path="include/blaze/gemm/block/block_scheduler_grouped_matmul.h",
        source_sha256="3ee3e474c99195ef1ee86b18d59a3083096ab591e24c118e4790bb57664310b8",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="mc2/mega_moe/op_kernel/arch22/mega_moe.cpp",
        source_sha256="be5dea396c9f60d83c4de585406a2f1b1ba54b1076b2aa07fabba36b2c7abae1",
    ),
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        relative_path="quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_base.cpp",
        source_sha256="db7838ecfb7990418bf92bd555d336e10137d4d5ac519aa0877d924d65e393f7",
    ),
    SourceRef(
        repository="xllm-ops",
        revision="32717633770e523782460c52db79af7f195ce575",
        relative_path="xllm_ops/moe_grouped_matmul_swiglu_quant/op_host/moe_grouped_matmul_swiglu_quant_tiling.cpp",
        source_sha256="21c3f11c96a4ab5ab193f644a10d8fe6153aacefec95de48ae8424b1ad00ca44",
    ),
)

ARCH_ID = ArchId("npu_arch_2201")
MATRIX_DTYPES = frozenset({"fp32", "bf16", "fp16", "int8", "w4a8", "int4"})
ROUTED_DTYPES = frozenset({"bf16", "fp16", "int8", "w4a8"})
AIV_INPUT_DTYPES = frozenset({"int32", "fp16", "bf16", "fp32"})
XLLM_GATED_ABI = "aclnnMoeGroupedMatmulSwigluQuant"


def _excludes_packed_weight(fields, ctx):
    if fields.get("a2_exclude_int32_weight") is True:
        return True
    abi = fields.get("transport_abi")
    if abi in {"a2_hccl_client", "a3_sio_hccs"}:
        return abi == "a2_hccl_client"
    tier = str(fields.get("soc_tier", "")).lower()
    if tier in {"a2", "a3"}:
        return tier == "a2"
    variant = str(fields.get("soc_variant", "")).lower()
    if variant.startswith(("ascend910b", "ascend910_93")):
        return variant.startswith("ascend910b")
    memory = getattr(ctx.geometry.physical_mechanisms, "memory_hierarchy", None)
    return memory is not None and memory.provenance.get("arch") == "A2"


def _validate(op, ctx, contract):
    if contract.mode != "dequant_swiglu_quant":
        if not ctx.datapath.has_path("l0c", "gm") or ctx.effective_aic_cores < 1:
            raise capability_gap(op, ctx, "l0c_to_gm_fixpipe_and_aic_owners")
    if contract.mode not in GEMM_MODES and ctx.effective_aiv_cores < 1:
        raise capability_gap(op, ctx, "aiv_owners")
    if contract.mode != "dequant_swiglu_quant" and contract.compute_dtype not in MATRIX_DTYPES:
        raise capability_gap(op, ctx, f"arch22_compute_dtype:{contract.compute_dtype}")
    if {contract.dtype_a, contract.dtype_b, contract.quant_dtype} & MX_DTYPES:
        raise capability_gap(op, ctx, "arch22_mx_operands")
    requested = contract.abi
    if requested == "aclnnGroupedMatmulSwigluMxQuant":
        raise capability_gap(op, ctx, "arch22_mx_epilogue")
    if requested == XLLM_GATED_ABI and (
        contract.dtype_a, contract.layout_a, contract.dtype_b, contract.layout_b
    ) != ("int8", "ND", "int8", "NZ"):
        raise capability_gap(op, ctx, "xllm_requires_int8_nd_by_int8_nz")
    if contract.mode in ROUTED_MODES:
        if contract.dtype_b == "int32" and _excludes_packed_weight(contract_fields(op), ctx):
            raise capability_gap(op, ctx, "arch22_mega_moe_int32_weight")
        if contract.dtype_b == "int32" and contract.dtype_a != "int8":
            raise capability_gap(op, ctx, "arch22_mega_moe_packed_weight_requires_int8")
        if contract.compute_dtype not in ROUTED_DTYPES:
            raise capability_gap(op, ctx, f"arch22_mega_moe_dtype:{contract.compute_dtype}")
        if contract.enable_fused_mc2 == FUSED_MC2_SHARED_MODE:
            raise capability_gap(op, ctx, "arch22_shared_mc2")
    if contract.mode != "dequant_swiglu_quant" and contract.matrix_weight_dtype not in MATRIX_DTYPES:
        raise capability_gap(op, ctx, f"arch22_weight_dtype:{contract.dtype_b}")
    if contract.mode == "gmm_swiglu_quant" and (
        contract.compute_dtype not in {"int8", "w4a8"} or contract.activation_divisor != 2
    ):
        raise capability_gap(op, ctx, "arch22_gated_int32_accumulator")
    if contract.mode in ROUTED_MODES | {"gmm_swiglu_quant"}:
        fields = contract_fields(op)
        if fields.get("mask_routing") or fields.get("mask_blocked_output"):
            raise capability_gap(op, ctx, "arch22_mask_routing")
        if fields.get("l0c_to_ub") or fields.get("handoff_path") == "l0c->ub" or ctx.datapath.has_l0c_to_ub:
            raise capability_gap(op, ctx, "arch22_requires_gm_handoff")
    if contract.mode == "dequant_swiglu_quant" and contract.handoff_dtype not in AIV_INPUT_DTYPES:
        raise capability_gap(op, ctx, f"arch22_dequant_input:{contract.handoff_dtype}")


def _gated_stages(contract, ctx, name, after, metadata):
    schedule = matrix_schedule(contract, ctx, contract.group_rows, contract.n, contract.k)
    stages = matrix_stages(
        name + "gmm1", after, contract, schedule, contract.k, contract.handoff_dtype,
        input_scaled=contract.mode in ROUTED_MODES and contract.dtype_a == "int8", metadata=metadata,
    )
    ready = (name + "gmm1",)
    if contract.abi == XLLM_GATED_ABI and contract.m:
        # The xLLM ABI has one FP32 weight-scale vector per active group.
        scales = sum(bool(rows) for rows in contract.group_rows) * contract.n * PER_TOKEN_SCALE_BYTES
        scale_read = name + "weight_scales"
        stages.append(StageDemand(scale_read, ready, Work(resources={
            "hbm": scales, "l2": scales, "mte2_aiv": scales,
        }), metadata=metadata))
        ready = (scale_read,)
    stages.extend(activation_stages(
        name + "swiglu_quant", ready, ctx, contract, contract.group_rows, metadata
    ))
    return stages, schedule


def _routed_stages(ctx, contract, name, after, metadata, transport_builder):
    dispatch, combine = name + "dispatch", name + "combine"
    stages = routing_stages(dispatch, after, contract, metadata, transport_builder)
    if not contract.m:
        return stages + routing_stages(combine, (dispatch,), contract, metadata, transport_builder)
    cohort = tuple((key, value) for key, value in metadata if key in {"cohort_id", "slot"})
    gated, _ = _gated_stages(contract, ctx, name, (dispatch,), cohort)
    stages.extend(gated)
    schedule2 = matrix_schedule(contract, ctx, contract.group_rows, contract.k, contract.intermediate)
    stages.extend(matrix_stages(
        name + "gmm2", (name + "swiglu_quant",), contract, schedule2, contract.intermediate,
        contract.output_dtype, input_dtype=contract.quant_dtype,
        input_scaled=contract.quant_dtype == "int8", metadata=cohort,
    ))
    stages.extend(routing_stages(combine, (name + "gmm2",), contract, metadata, transport_builder))
    return stages


@dataclass(frozen=True, slots=True)
class _Arch22Strategy:
    arch_id: ArchId = ARCH_ID
    abi: str = "aclnnGroupedMatmul"
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
            raise capability_gap(op, ctx, "arch22_compute_identity")
        contract = validate_contract(op, ctx)
        _validate(op, ctx, contract)
        mode = contract.mode
        metadata = {"mode": mode, "enable_dbo": contract.enable_dbo}
        if mode in GEMM_MODES:
            schedule = matrix_schedule(contract, ctx, contract.group_rows, contract.n, contract.k)
            stages = matrix_stages("gmm", (), contract, schedule, contract.k, contract.output_dtype)
            metadata.update(useful_flops=schedule.useful_flops, issued_fractals=schedule.issued_fractals,
                            max_waves=schedule.max_waves, wave_tail_imbalance=schedule.wave_tail_imbalance)
            implementation = self.abi
        elif mode == "gmm_swiglu_quant":
            contract = replace(contract, handoff_dtype="int32")
            stages, schedule = _gated_stages(contract, ctx, "", (), ())
            metadata.update(useful_flops=schedule.useful_flops, issued_fractals=schedule.issued_fractals,
                            gm_handoff_bytes=2 * contract.m * contract.n * INT32_HANDOFF_BYTES_PER_ELEM)
            implementation = contract.abi or "aclnnGroupedMatmulSwigluQuant"
        elif mode == "dequant_swiglu_quant":
            stages = activation_stages("dequant_swiglu_quant", (), ctx, contract, contract.group_rows)
            implementation = "aclnnDequantSwigluQuant"
        else:
            transport_builder = contract_fields(op).get("cohort_transport_builder")
            emit = partial(_routed_stages, ctx)
            if contract.dbo_split is not None:
                stages = DualBatchMoEPipeline(contract.dbo_split).stages(
                    contract, ctx, emit, transport_builder=transport_builder
                )
            elif contract.m or contract.tokens or any(contract.transport_bytes) or transport_builder:
                stages = emit(contract, "", (), (), transport_builder)
            else:
                stages = []
            implementation = "aclnnMegaMoe"
            metadata["gm_handoff_bytes"] = 2 * contract.m * element_bytes(contract.n, contract.handoff_dtype)
        return make_plan(implementation, ctx, stages, self.sources, metadata=metadata)


STRATEGY: ArchStrategy = _Arch22Strategy()
