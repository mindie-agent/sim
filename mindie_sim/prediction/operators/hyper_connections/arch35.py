"""RegBase strategy for hyper connections on Ascend architecture 3510.

[Role] Select RegBase HC, TextGatedResidual, AttnRes, and Engram gate.
[Invariants] Strategy uses declared ISA, RegBase UB capacity, and caller ABI.
[Timeouts & Locks] Pure bounded planning. No locks.
[Side-Effect Policy] No device context or external I/O.
Upstream-ABI: aclnnTextGatedResidual
"""
from __future__ import annotations

from mindie_sim.schema.datapath import SourceRef
from ..contracts import ArchStrategy
from .common import ARCH35, HyperConnectionsStrategy

REGBASE_UB_BYTES = 248 * 1024
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="mhc/text_gated_residual/op_host/text_gated_residual_tiling.cpp",
        source_sha256="1c89c53b186a3d5cd8c6bc766b8cbe6cb25286bb7b8f1a14f44dc28863e193b5",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="mhc/text_gated_residual/op_kernel/text_gated_residual.cpp",
        source_sha256="5d31d00e857b0b35be1b4165c5006aef8bfc00ba4562f7cf8eb6047d7ecae82a",
    ),
    SourceRef(
        repository="cann/cann-recipes-infer",
        revision="3c9266727c2a25bd17d503ad37d189a134ff364b",
        relative_path="ops/ascendc/src/hc_pre_sinkhorn/op_kernel/hc_pre_sinkhorn_regbase_base.h",
        source_sha256="bc252a6a996370d852e3a9c4c9d3564cc990ec799b6256e3205e880df6deacc6",
    ),
)
STRATEGY: ArchStrategy = HyperConnectionsStrategy(
    arch_id=ARCH35, abi="aclnnTextGatedResidual", sources=SOURCES, ub_bytes=REGBASE_UB_BYTES,
    datapath_name="regbase", exclusive_abis=frozenset({"regbase", "RegBase", "fusedSmall"}),
)
