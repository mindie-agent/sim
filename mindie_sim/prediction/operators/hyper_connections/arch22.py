"""MemBase strategy for hyper connections on Ascend architecture 2201.

[Role] Select MemBase HC, TextGatedResidual, AttnRes, and Engram gate.
[Invariants] Strategy uses the declared ISA and MemBase UB capacity.
[Timeouts & Locks] Pure bounded planning. No locks.
[Side-Effect Policy] No device context or external I/O.
Upstream-ABI: hc_pre_sinkhorn
"""
from __future__ import annotations

from mindie_sim.schema.datapath import SourceRef
from ..contracts import ArchStrategy
from .common import ARCH22, HyperConnectionsStrategy

MEMBASE_UB_BYTES = 192 * 1024
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/cann-recipes-infer",
        revision="3c9266727c2a25bd17d503ad37d189a134ff364b",
        relative_path="ops/ascendc/src/hc_pre_sinkhorn/op_kernel/hc_pre_sinkhorn_base.h",
        source_sha256="90687ddfda8abf22dca0cdddb524444e9dce14f7e1ba4905eda6d8f50f0b858b",
    ),
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
)
STRATEGY: ArchStrategy = HyperConnectionsStrategy(
    arch_id=ARCH22, abi="hc_pre_sinkhorn", sources=SOURCES, ub_bytes=MEMBASE_UB_BYTES,
    datapath_name="membase", exclusive_abis=frozenset({"membase", "MemBase"}),
)
