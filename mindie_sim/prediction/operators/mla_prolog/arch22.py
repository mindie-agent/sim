"""ACLNN MLA prolog split-N strategy for the 2201 architecture.

Upstream-ABI: aclnnMlaPrologV3WeightNz

[Role] Select the ACLNN V3 split-N phase graph and its GM workspaces.
[Invariants] BF16 and INT8 projections use host step-batch and head ownership.
[Timeouts & Locks] Bounded plan construction with no I/O or locks.
[Side-Effect Policy] Returns immutable programs and source references.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    ACLNN_V3_ABIS, GROUP_COMPUTE_MIN_HEADS, WEIGHT_QUANT_BF16, WEIGHT_QUANT_INT8, WEIGHT_QUANT_MXFP8,
    build_plan, contract_for, unsupported,
)

ARCH_ID = ArchId("npu_arch_2201")
UPSTREAM_ABI = "aclnnMlaPrologV3WeightNz"
BF16_ND_OUTPUT_BLOCK_BYTES = 32
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_host/mla_prolog_tiling.cpp",
        source_sha256="1770937bd4a58222c5606ce03bec3c574e23bd563ea341c03a1710f153962909",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_host/mla_prolog_tiling.h",
        source_sha256="ca141526d57ede6e0d87b3c2d3a1c5e126f228cac3a4dd71a55068f072688f3c",
    ),
    SourceRef(
        repository="Ascend/op-plugin",
        revision="48189acf95882e23fae56ea96ceeb89358cb42da",
        relative_path="op_plugin/ops/opapi/MlaPrologV3KernelNpuOpApi.cpp",
        source_sha256="44f2a01f2acd64c6ce8ea1262b8e68cb11d74fd58ea3e8cd457085d3e0715c14",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/attention/mla_v1.py",
        source_sha256="f7f672e001651a748d2e88ef0f36797ef78db63321ac657f1c7a944c0f561c22",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h",
        source_sha256="07750f2b7ab0650f4ea886f5c77416a3d8860929900e3d1b280ca3102ff33a14",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_kernel/arch22/mla_prolog_comm_arch22.h",
        source_sha256="e977456ed248a643161d6e7f740909f6d02ec10c819cdb3111b7f2ad0d88d86b",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp",
        source_sha256="5e34239dabdf9a159d41b542401fc2bcf7c3551b6e6695c5d761dff5638e6d57",
    ),
)


@dataclass(frozen=True, slots=True)
class _SplitNStrategy:
    arch_id: ArchId = ARCH_ID
    abi: str = UPSTREAM_ABI
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        c = contract_for(op, ctx)
        return c.abi in ACLNN_V3_ABIS and c.weight_quant_mode != WEIGHT_QUANT_MXFP8

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        c = contract_for(op, ctx)
        if c.abi not in ACLNN_V3_ABIS or c.weight_quant_mode == WEIGHT_QUANT_MXFP8:
            raise unsupported(op, ctx, f"unsupported_abi_or_weight_mode:{c.abi}:{c.weight_quant_mode}")
        # The other host dequant modes and the partial-quant group mode are outside this ABI binding.
        cq_scale = c.weight_quant_mode == WEIGHT_QUANT_INT8 and c.num_heads >= GROUP_COMPUTE_MIN_HEADS
        # The V3 split-N host tiler rounds ND output columns to BLOCK_SIZE / sizeof(T).
        alignment = BF16_ND_OUTPUT_BLOCK_BYTES if c.weight_quant_mode == WEIGHT_QUANT_BF16 else None
        return build_plan(c, ctx, sources=self.sources, mode="split_n", cq_scale=cq_scale,
                          fixpipe_alignment=alignment)


STRATEGY: ArchStrategy = _SplitNStrategy()
