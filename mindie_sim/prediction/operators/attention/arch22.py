"""Arch22 fused inference attention score.

Upstream-ABI: aclnnFusedInferAttentionScore

[Role] Lower nonquantized GQA with the Arch22 GM workspace crossover.
[Invariants] Select by explicit ISA and ABI. Keep analytical panel candidates
  separate from shipping templates.
[Timeouts & Locks] Finite panel and workspace planning without locks.
[Side-Effect Policy] Read-only capability and source descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan
from ..contracts import OperatorPlanningContext
from .common import ARCH22, FIA_ABI, AttentionContract, dense_fia, unmodeled

OPS_TRANSFORMER_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/fused_infer_attention_score/op_kernel/arch22/fia_block_cube_nonquant_gqa.h",
        "29eb00df04cfebd0d2aac0c3a49f87546c79d04a856284559d235b353146fcf4",
    ),
)


@dataclass(frozen=True, slots=True)
class _Strategy:
    arch_id: ArchId = ArchId(ARCH22)
    abi: str = FIA_ABI
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        abi = ctx.requested_abi or dict(op.metadata).get("native_kind", FIA_ABI)
        return ctx.compute_arch.arch_id == self.arch_id and abi == self.abi

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        contract: AttentionContract = dict(op.metadata)["attention_contract"]
        if not self.accepts(op, ctx) or contract.abi != self.abi:
            raise unmodeled("attention_arch22_abi", f"Arch22 attention requires {self.abi}")
        for src, dst in (("l0c", "gm"), ("gm", "ub"), ("ub", "gm"), ("gm", "l1")):
            ctx.datapath.route(src, dst)
        return dense_fia(op, contract, ctx, self.sources, gm_workspace=True)


STRATEGY = _Strategy()
