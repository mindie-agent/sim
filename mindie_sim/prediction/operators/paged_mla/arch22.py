"""Arch22 FIA paged MLA with GM intermediates and source clock skew.

[Role] Lower the FIA ABI on npu_arch_2201.
[Invariants] Task records and score/probability slots rotate on S2 tiles.
  MM2 slots rotate on owner intervals and stay fixed through each S2 tail.
[Timeouts & Locks] Bounded per-owner pipeline window. No locks.
[Side-Effect Policy] Plan from immutable geometry and pinned source references.
Upstream-ABI: aclnnFusedInferAttentionScore
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ..contracts import ArchStrategy, OperatorPlanningContext
from ._lowering import FIA_WORKSPACE_DEPTH, PipelineLayout, make_plan
from .common import FIA_ABI, FP32_BYTES, capability_gap, contract_for

ARCH_ID = "npu_arch_2201"
TASK_RECORD_COUNT = 3
WORKSPACE_DEPTH = FIA_WORKSPACE_DEPTH
# The source assigns <=16 live M rows to the first AIV subblock.
HEAD_TILE = 16
CLOCKS = (
    ("q_load", "kv_load", "qk_operands", "qk", "score_store"),
    ("score_read", "softmax", "probability", "pv_load", "pv_operands", "pv",
     "atomic_rescale", "atomic_write", "partial"),
    ("output_read", "output_update", "split_write"),
)
WORKSPACE_REGIONS = (
    ("mm1", WORKSPACE_DEPTH, "score_store", "score_read"),
    ("vec1", WORKSPACE_DEPTH, "probability", "pv_load"),
    ("vec2", WORKSPACE_DEPTH, "output_update", "split_write"),
)
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/fused_infer_attention_score/op_kernel/arch22/fia_kernel_nonquant_mla.h",
        source_sha256="47c22ec83a6e771df0185add46c91c4404ce87ad7feee1f9666a129190231e92",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/fused_infer_attention_score/op_kernel/arch22/fia_block_cube_nonquant_mla.h",
        source_sha256="8108fea02edc14e0549f3bcafa228829b1ea6502b0f373d2ee13dd2c5f678b50",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/fused_infer_attention_score/op_kernel/arch22/fia_block_vec_nonquant_mla.h",
        source_sha256="6c89b6c8bc41c81220418f391c17cad6a504e4b0eabe1040907ee08a9e3c5913",
    ),
)


def arch22_fia_schedule(num_tiles: int) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    """Return clock indices modulo each ring and the active stages in issue order."""
    if type(num_tiles) is not int or num_tiles < 0:
        raise ModelingError(f"paged_mla_invalid_num_tiles:{num_tiles!r}")
    if not num_tiles:
        return ()
    stage_groups = (("qk",), ("softmax", "pv"), ("output_update",))
    return tuple(
        (clock % TASK_RECORD_COUNT, clock % WORKSPACE_DEPTH,
         tuple(stage for delay, group in enumerate(stage_groups)
               if 0 <= clock - delay < num_tiles for stage in group))
        for clock in range(num_tiles + TASK_RECORD_COUNT - 1)
    )


@dataclass(frozen=True, slots=True)
class _FiaStrategy:
    arch_id = ARCH_ID
    abi = FIA_ABI
    sources = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        return ctx.compute_arch.arch_id == self.arch_id and contract_for(op, ctx).abi == self.abi

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /):
        c = contract_for(op, ctx)
        if ctx.compute_arch.arch_id != self.arch_id or c.abi != self.abi:
            raise capability_gap("paged_mla_unsupported_abi", c.abi, ctx.compute_arch.arch_id)
        elem = int(dtype_bytes(op.dtype or "bf16"))
        heads = min(HEAD_TILE, c.physical_heads)
        keys = min(c.s2_base_size, max((kv for q, kv in c.sequences if q), default=0))
        workspace_regions = (
            ("mm1", WORKSPACE_DEPTH, heads * keys * FP32_BYTES),
            ("vec1", WORKSPACE_DEPTH, heads * keys * elem),
            ("mm2", WORKSPACE_DEPTH, heads * c.dv * FP32_BYTES),
            ("vec2", WORKSPACE_DEPTH, heads * c.dv * elem),
        )
        layout = PipelineLayout(CLOCKS, WORKSPACE_REGIONS, HEAD_TILE, c.s2_base_size, gm_workspace=True)
        return make_plan(
            op, ctx, c, layout, self.sources,
            task_record_count=TASK_RECORD_COUNT, workspace_depth=WORKSPACE_DEPTH,
            workspace_regions=workspace_regions,
        )


STRATEGY: ArchStrategy = _FiaStrategy()
