"""Arch35 MC2 direct peer, MTE window, and nonblocking CCU strategies.

[Role] Bind declared Arch35 transport and topology to the shared MC2 ledger.
[Invariants] Peer credits bound live windows and injection. Window publication
  follows payload completion; CCU completion gates its consuming core.
[Timeouts & Locks] Finite group and cohort planning; no I/O or locks.
[Side-Effect Policy] Produces analytical declarations only.

Upstream-ABI: aclnnMoeDistributeDispatchV2
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    COLLECTIVES, COMBINE, MMRS, Mc2Contract, PeerRoute, Transport, build_transport_stages, capability_gap,
    contract_fields, make_plan, topology_hops, validate_contract,
)

_OPS_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
_SKILLS_REVISION = "750045df251d031bf708de9dc1614c2202eed6dd"
_XLLM_REVISION = "c77a313bfdd59c0f6e312558808c0de7d2a8d019"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/all_gather_matmul_v2/op_kernel/arch35/all_gather_matmul_fp16_bf16.h",
        source_sha256="fe27d0c153368f57396c14bcab681e7a12c8a8b5f732bc45be4c393eb74d82a7",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h",
        source_sha256="121ad9870776e27c1986e612b047817267e0de8976236df7eac748ef1643eff3",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h",
        source_sha256="3194ed49e10aab74976053840a5e771051be946d0945664fd5a3ca34a32bdb6b",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h",
        source_sha256="4e2a3f571df2e0e3718c8e61145638483be15445383ed77c3873381874ffb5a9",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_ccu.h",
        source_sha256="e34bbda26c14fc7b9b8767568541f72d7e67a89dc8dd0d4b9a324add0aba417a",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_ccu.h",
        source_sha256="2b4d5fde8c865e54017949ae7618b7c28b40a92ef725253e847fc04113a6cf81",
    ),
    SourceRef(
        repository="cannbot-skills",
        revision=_SKILLS_REVISION,
        relative_path="ops/ascendc-mc2-best-practice/references/requirement-analysis/comm-path-decision.md",
        source_sha256="05a9e0df5d30d32096454a7fa0cdcab23596254ecdf1639f441657ca9a2f9b25",
    ),
    SourceRef(
        repository="xllm",
        revision=_XLLM_REVISION,
        relative_path="xllm/core/kernels/npu/npu_moe_distribute_dispatch_v2.cpp",
        source_sha256="f4eb13448064c7987efb2bb72ffa2cda3e0b89ef0986f293c8ff98978096e05d",
    ),
    SourceRef(
        repository="xllm",
        revision=_XLLM_REVISION,
        relative_path="xllm/core/kernels/npu/npu_moe_distribute_combine_v2.cpp",
        source_sha256="5ffa8553993325dac3d9ff04ccfb999a25f5b6987b71964ce301b71ed679f418",
    ),
)


SHMEM = "a5_shmem"
MTE_WINDOW = "a5_mte"
CCU = "a5_ccu"
URMA_ALIGN_BYTES = 512
MESH_RANK_LIMIT = 8
ARCH_TRANSPORTS = frozenset({SHMEM, MTE_WINDOW, CCU})


def _transport(contract: Mc2Contract, op: OpSpec, ctx: OperatorPlanningContext) -> Transport:
    fields = contract_fields(op)
    group = dict(fields.get("group", ()))
    raw_binding = fields.get("transport_binding", {})
    binding = dict(raw_binding) if not isinstance(raw_binding, str) else {"caller_abi": raw_binding}
    caller = str(binding.get("caller_abi", fields.get("caller_abi", ""))).lower()
    abi = contract.transport_abi
    if abi == "auto":
        if caller in {"shmem", "aclshmem", SHMEM}:
            abi = SHMEM
        elif caller in {"ccu", "hcomm", CCU} or contract.abi in COLLECTIVES:
            abi = CCU
        else:
            abi = MTE_WINDOW
    if abi not in ARCH_TRANSPORTS or op.recipe == "arch22_v2_caller_local":
        raise capability_gap(op, ctx, f"transport_unsupported:{abi}")
    algorithm = contract.topology_algorithm or ("direct_peer" if abi == SHMEM else "fullmesh")
    rank = contract.rank
    peers = tuple(peer for peer in contract.ranks if peer != rank)
    hops = topology_hops(fields, group)
    offset = binding.get("window_offset_bytes", fields.get("window_offset_bytes", 0))
    if type(offset) is not int or offset < 0:
        raise capability_gap(op, ctx, "invalid_window_offset")
    if abi == SHMEM:
        if caller not in {"shmem", "aclshmem", SHMEM}:
            raise capability_gap(op, ctx, "shmem_requires_shmem_caller")
        if algorithm not in {"fullmesh", "direct_peer"} or contract.world_size > MESH_RANK_LIMIT:
            raise capability_gap(op, ctx, "shmem_requires_direct_mesh")
        edges = set(ctx.datapath.ub_peer_edges)
        edges.update(tuple(edge) for edge in group.get("peer_edges", fields.get("peer_edges", ())))
        direct = {"UB", "UBMEM", "URMA", "DIRECT_PEER"}
        for peer in peers:
            path = hops.get((rank, peer))
            if path is None:
                valid = (rank, peer) in edges
            else:
                valid = len(path) == 1 and path[0].upper() in direct
            if not valid:
                raise capability_gap(op, ctx, "shmem_peer_capability_required")
        if offset % URMA_ALIGN_BYTES:
            raise capability_gap(op, ctx, "shmem_requires_512_byte_aligned_window")
    if abi != SHMEM and caller in {"shmem", "aclshmem", SHMEM}:
        raise capability_gap(op, ctx, "shmem_caller_transport_mismatch")
    if abi == MTE_WINDOW and contract.abi in COLLECTIVES:
        raise capability_gap(op, ctx, "mte_window_requires_moe_dispatch_or_combine")
    if abi == MTE_WINDOW and peers and not (
        ctx.datapath.has_path("gm", "ub") and ctx.datapath.has_path("ub", "gm")
    ):
        raise capability_gap(op, ctx, "mte_window_requires_gm_ub_paths")
    if algorithm in {"ring", "two_tier"} and any((rank, peer) not in hops for peer in peers):
        raise capability_gap(op, ctx, "algorithm_requires_group_hops")
    native_chunking = abi == CCU and contract.abi in COLLECTIVES
    if contract.internal_chunks == 2 and not all(
        binding.get(key, fields.get(key, native_chunking)) is True
        for key in ("independent_buffers", "independent_handles")
    ):
        raise capability_gap(op, ctx, "two_chunks_require_independent_buffers_and_handles")

    route_kind = "UB_WINDOW" if abi == MTE_WINDOW else "URMA"
    routes = tuple(PeerRoute(rank, peer, hops.get((rank, peer), (route_kind,))) for peer in peers)
    credits = binding.get("peer_credits", fields.get("peer_credits", max(1, len(peers))))
    if type(credits) is not int or credits < 1:
        raise capability_gap(op, ctx, "invalid_peer_credit_limit")
    depth = contract.internal_chunks if contract.abi in COLLECTIVES else (2 if (
        contract.independent_buffers and contract.independent_handles
    ) else 1)
    return Transport(
        abi, algorithm, routes, wire_alignment=URMA_ALIGN_BYTES if abi == SHMEM else 1,
        window_alignment=URMA_ALIGN_BYTES if abi == SHMEM else 1, window_offset=offset,
        credits=1 if algorithm == "ring" else min(credits, max(1, len(peers))),
        slot_depth=depth, peer_limit=len(peers),
        ccu=abi == CCU, aiv_staging=abi == MTE_WINDOW,
        direct_remote_gm=abi == MTE_WINDOW and contract.abi == COMBINE,
    )


@dataclass(frozen=True, slots=True)
class _Strategy:
    arch_id: ArchId = ArchId("npu_arch_3510")
    abi: str = "aclnnMoeDistributeDispatchV2"
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        try:
            _transport(validate_contract(op, ctx), op, ctx)
        except ModelingError:
            return False
        return True

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        if ctx.compute_arch.arch_id != self.arch_id:
            raise capability_gap(op, ctx, "unsupported_compute_architecture")
        contract = validate_contract(op, ctx)
        transport = _transport(contract, op, ctx)
        ub_output = (contract.abi == MMRS and contract.world_size > 1
                     and ctx.datapath.has_l0c_to_ub)
        phases = build_transport_stages(contract, transport, ub_output=ub_output)
        return make_plan(contract, transport, phases, self.sources, arch_id=self.arch_id)


STRATEGY: ArchStrategy = _Strategy()
