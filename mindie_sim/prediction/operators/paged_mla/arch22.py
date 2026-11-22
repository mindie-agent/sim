"""Arch22 FIA paged MLA with GM intermediates and source clock skew.

[Role] Lower the FIA ABI on npu_arch_2201.
[Invariants] Task records and score/probability slots rotate on S2 tiles.
  MM2 slots rotate on owner intervals and stay fixed through each S2 tail.
[Timeouts & Locks] Bounded per-owner pipeline window. No locks.
[Side-Effect Policy] Plan from immutable geometry and pinned source references.
Upstream-ABI: aclnnFusedInferAttentionScore
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ..contracts import ArchStrategy, OperatorPlanningContext
from ..sharedkv.split_core import resolve_fia_mla_tiling
from ._lowering import FIA_WORKSPACE_DEPTH, PipelineLayout, make_plan
from .common import FP32_BYTES, capability_gap, contract_for

ARCH_ID = "npu_arch_2201"
UPSTREAM_ABI = "aclnnFusedInferAttentionScore"
TASK_RECORD_COUNT = 3
WORKSPACE_DEPTH = FIA_WORKSPACE_DEPTH
# The source assigns <=16 live M rows to the first AIV subblock.
HEAD_TILE = 16
NATIVE_S2_MIN = 512
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
    abi = UPSTREAM_ABI
    sources = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        return ctx.compute_arch.arch_id == self.arch_id and contract_for(op, ctx).abi == self.abi

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /):
        c = contract_for(op, ctx)
        if ctx.compute_arch.arch_id != self.arch_id or c.abi != self.abi:
            raise capability_gap("paged_mla_unsupported_abi", c.abi, ctx.compute_arch.arch_id)
        elem = int(dtype_bytes(op.dtype or "bf16"))
        l2_cap = int(ctx.geometry.buffers.get("l2", dict(ctx.datapath.buffers).get("l2", 0)))
        native_mla = (
            c.splits == 1
            and c.physical_heads > HEAD_TILE
            and c.s2_base_size >= NATIVE_S2_MIN
        )
        head_tile = HEAD_TILE
        s2_tile = c.s2_base_size
        used_cores = 1
        if native_mla:
            tiling = resolve_fia_mla_tiling(
                sequences=c.sequences, physical_heads=c.physical_heads,
                aic_count=ctx.effective_aic_cores, is_arch35=False,
            )
            head_tile = min(tiling.m_tile, c.physical_heads)
            s2_tile = tiling.s2_tile
            used_cores = tiling.used_cores
            if tiling.fd_splits_per_row > 1:
                c = replace(c, splits=tiling.fd_splits_per_row)
        keys = min(s2_tile, max((kv for q, kv in c.sequences if q), default=0))
        heads = min(head_tile, c.physical_heads) if keys or not ctx.requests else 0
        ws_per_core = WORKSPACE_DEPTH * (
            heads * keys * (FP32_BYTES + elem) + heads * c.dv * (FP32_BYTES + elem)
        )
        latent_in_flight = WORKSPACE_DEPTH * keys * c.dqk * elem
        ws_l2_resident = native_mla and l2_cap > 0 and (ws_per_core * max(1, used_cores)) <= l2_cap
        latent_l2_hit = (
            native_mla and bool(c.latent_alias) and l2_cap > 0
            and ((ws_per_core + latent_in_flight) * max(1, used_cores)) <= l2_cap
        )
        workspace_regions = (
            ("mm1", WORKSPACE_DEPTH, heads * keys * FP32_BYTES),
            ("vec1", WORKSPACE_DEPTH, heads * keys * elem),
            ("mm2", WORKSPACE_DEPTH, heads * c.dv * FP32_BYTES),
            ("vec2", WORKSPACE_DEPTH, heads * c.dv * elem),
        )
        layout = PipelineLayout(
            CLOCKS, WORKSPACE_REGIONS, head_tile, s2_tile,
            gm_workspace=True, ws_l2_resident=ws_l2_resident, latent_l2_hit=latent_l2_hit,
        )
        return make_plan(
            op, ctx, c, layout, self.sources,
            task_record_count=TASK_RECORD_COUNT, workspace_depth=WORKSPACE_DEPTH,
            workspace_regions=workspace_regions,
        )


STRATEGY: ArchStrategy = _FiaStrategy()
