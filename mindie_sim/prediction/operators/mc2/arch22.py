"""Arch22 MC2 HCCL-client and explicitly placed SIO/HCCS strategies.

[Role] Bind the Arch22 MC2 caller to a transport and declared group routes.
[Invariants] HCCL server binding and topology are independent. SIO, HCCS
  switches and cross-domain paths follow placement or explicit hop records.
[Timeouts & Locks] Finite group and cohort planning; no I/O or locks.
[Side-Effect Policy] Produces analytical declarations only.

Upstream-ABI: aclnnAllGatherMatmulV2
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ... import mc2_local_caller
from ...mechanism import OperatorPlan, Work
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    Mc2Contract, PeerRoute, Phase, Transport, build_transport_stages, capability_gap,
    contract_fields, make_plan, native_protocol_peer_transport, topology_hops,
    validate_contract, validate_native_dispatch_quant,
)
from .lowering import SIO_HCCS, declared_incoming_routes

_OPS_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
_SKILLS_REVISION = "750045df251d031bf708de9dc1614c2202eed6dd"
_XLLM_REVISION = "c77a313bfdd59c0f6e312558808c0de7d2a8d019"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/all_gather_matmul_v2/op_host/op_tiling/arch22/all_gather_matmul_aiv_mode_tiling.cpp",
        source_sha256="ecbb57b4c34aced7f2172bfdc48c153166f45aadbc7616745dd6db5eab0bebdd",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch22/matmul_reduce_scatter_v2_a2a3.cpp",
        source_sha256="c5eb062a27b5f1460c90f68f2b95df21fad805b2bf2d1416c15557d7068bce0d",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_a2.cpp",
        source_sha256="37716ea752cfe9792948004633b9ec16d8b68952c0bf12e5d634a429a97bacde",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_a3.cpp",
        source_sha256="b22b033d36406958bdf87de1489d207377836b0dfbfbc17148743952ddd35094",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_a2.cpp",
        source_sha256="b8d19cb6fe102e606f56cfe976a6c7b12a3c070eeb2043c1788d5859c8a9dde3",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_a3.cpp",
        source_sha256="b7eb67de6390df7688ce9f24b4ecbc08172d66fa5e2b2a0f664d2c4f15169466",
    ),
    SourceRef(
        repository="cannbot-skills",
        revision=_SKILLS_REVISION,
        relative_path="ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/all_gather_matmul/x_all_gather_matmul/op_host/op_tiling/arch22/x_all_gather_matmul_tiling_a2a3.cpp",
        source_sha256="48544c803eb2be74234c99075bb32cf952faefcae9a8848d213dbe3d0cc3aa2f",
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


HCCL_CLIENT = "a2_hccl_client"
ARCH_TRANSPORTS = frozenset({HCCL_CLIENT, SIO_HCCS})
SERVER_MODES = frozenset({"server", "aicpu", "aiv", "hccl_server_type_aicpu", "hccl_server_type_aiv"})


def _placement(raw, ranks, op, ctx, field):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        if len(raw) != len(ranks):
            raise capability_gap(op, ctx, f"{field}_must_cover_group")
        raw = dict(zip(ranks, raw))
    if set(raw) != set(ranks) or any(type(value) not in {int, str} for value in raw.values()):
        raise capability_gap(op, ctx, f"{field}_must_cover_group")
    return raw


def _sio_hccs_hops(contract, fields, group, explicit, op, ctx):
    ranks, rank = contract.ranks, contract.rank
    packages = _placement(fields.get("package_ids", group.get("package_ids")), ranks, op, ctx, "package_ids")
    nodes = _placement(fields.get("node_ids", group.get("node_ids")), ranks, op, ctx, "node_ids")
    if packages:
        by_package = {}
        for member, package in packages.items():
            key = nodes.get(member), package
            by_package[key] = by_package.get(key, 0) + 1
        if any(count > 2 for count in by_package.values()):
            raise capability_gap(op, ctx, "sio_requires_two_die_packages")
    cross_domain = group.get("cross_domain_hops", fields.get("cross_domain_hops", ()))
    cross_domain = (cross_domain,) if isinstance(cross_domain, str) else tuple(cross_domain)
    routes = []
    for peer in ranks:
        if peer == rank:
            continue
        path = explicit.get((rank, peer))
        if path is None:
            if nodes and nodes[peer] != nodes[rank]:
                path = cross_domain
            elif packages:
                path = ("SIO",) if packages[peer] == packages[rank] else ("HCCS_SW",)
            elif (peer, rank) in explicit:
                path = tuple(reversed(explicit[peer, rank]))
        if not path:
            raise capability_gap(op, ctx, "a3_peer_hops_required")
        routes.append(PeerRoute(rank, peer, path))
    return tuple(routes)


def _transport(contract: Mc2Contract, op: OpSpec, ctx: OperatorPlanningContext) -> Transport:
    fields = contract_fields(op)
    group = dict(fields.get("group", ()))
    raw_binding = fields.get("transport_binding", {})
    binding = dict(raw_binding) if not isinstance(raw_binding, str) else {"caller_abi": raw_binding}
    caller = str(binding.get("caller_abi", fields.get("caller_abi", ""))).lower()
    mode = str(binding.get("server_mode", fields.get("server_mode", ""))).lower()
    hops = topology_hops(fields, group)
    abi = contract.transport_abi
    if abi == "auto":
        placed = any(key in fields or key in group for key in ("package_ids", "node_ids"))
        sio_hccs = any(hop.upper() in {"SIO", "HCCS_SW"} for route in hops.values() for hop in route)
        if caller in {"hccl_client", HCCL_CLIENT}:
            abi = HCCL_CLIENT
        else:
            abi = SIO_HCCS if placed or sio_hccs or caller == SIO_HCCS else HCCL_CLIENT
    if abi not in ARCH_TRANSPORTS:
        raise capability_gap(op, ctx, f"transport_unsupported:{abi}")
    algorithm = contract.topology_algorithm or ("fullmesh" if abi == HCCL_CLIENT else "direct_peer")
    if contract.transport_abi == "auto" and (contract.world_size == 1 or "mc2_v2_local" in fields):
        return Transport(abi, algorithm, (), slot_depth=contract.internal_chunks)
    peers = tuple(peer for peer in contract.ranks if peer != contract.rank)
    if abi == HCCL_CLIENT:
        if caller not in {"hccl_client", HCCL_CLIENT} or mode not in SERVER_MODES:
            raise capability_gap(op, ctx, "hccl_client_requires_server_binding")
        if algorithm == "direct_peer":
            raise capability_gap(op, ctx, "hccl_client_algorithm_unsupported")
        if algorithm != "fullmesh":
            allowed = binding.get("algorithms", ())
            if isinstance(allowed, str):
                allowed = (allowed,)
            binding_algorithm = binding.get("topology_algorithm", binding.get("algorithm"))
            if (algorithm != binding_algorithm and algorithm not in allowed) or any(
                (contract.rank, peer) not in hops for peer in peers
            ):
                raise capability_gap(op, ctx, "hccl_algorithm_requires_binding_and_group_hops")
        routes = tuple(PeerRoute(contract.rank, peer, hops.get((contract.rank, peer), ("HCCL_FULLMESH",)))
                       for peer in peers)
    else:
        if algorithm in {"ring", "two_tier"} and any((contract.rank, peer) not in hops for peer in peers):
            raise capability_gap(op, ctx, "a3_algorithm_requires_group_hops")
        routes = _sio_hccs_hops(contract, fields, group, hops, op, ctx)
    if contract.internal_chunks == 2 and not (
        (contract.independent_buffers or binding.get("independent_buffers") is True)
        and (contract.independent_handles or binding.get("independent_handles") is True)
    ):
        raise capability_gap(op, ctx, "two_chunks_require_independent_buffers_and_handles")
    credits = 1 if algorithm == "ring" else max(1, len(peers))
    return Transport(abi, algorithm, routes, credits=credits,
                     slot_depth=contract.internal_chunks, peer_limit=len(peers),
                     incoming_routes=declared_incoming_routes(contract.rank, peers, hops))


@dataclass(frozen=True, slots=True)
class _Strategy:
    arch_id: ArchId = ArchId("npu_arch_2201")
    abi: str = "aclnnAllGatherMatmulV2"
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
        fields = contract_fields(op)
        local = fields.get("mc2_v2_local") if op.recipe == "arch22_v2_caller_local" else None
        if local is None:
            phases = build_transport_stages(contract, transport)
        else:
            group = dict(fields.get("device_kernel_group", ()))
            name = "caller_publish" if group.get("exit") else "caller_pack"
            phases = (Phase(name, work=Work(resources=dict(local["work"])), owner="aiv"),)
        return make_plan(contract, transport, phases, self.sources, arch_id=self.arch_id)


STRATEGY: ArchStrategy = _Strategy()

NATIVE_PROTOCOL_KIND = "arch22_v2_caller"


def validate_native_protocol(raw):
    if raw.get("dispatch_quant") != "per_token_int8":
        raise ModelingError("arch22_v2_caller_requires_per_token_int8_dispatch")
    if "global_bs" in raw and (type(raw["global_bs"]) is not int or raw["global_bs"] < 0):
        raise ModelingError("arch22_v2_global_bs_requires_nonnegative_integer")
    validate_native_dispatch_quant(raw)
    if any(raw.get(key) is not None for key in (
        "chunk_bytes", "hop_startup_us", "receive_capacity_bytes_per_peer",
    )):
        raise ModelingError("arch22_v2_caller_does_not_accept_chunked_direct_peer_options")
    native_protocol_peer_transport(raw)


def native_protocol_recipes(selected):
    return mc2_local_caller.recipes(selected)


def lower_native_protocol(
    ops, *, selected, runtime, prefix, tokens, top_k, width, experts, ep,
    source_partitions, received_assignments, dtype, source_routing=None,
    valid_tokens=None, source_rows_by_replica=None, destination_rows_by_rank=None,
):
    return mc2_local_caller.lower(
        ops, selected=selected, runtime=runtime, prefix=prefix, tokens=tokens,
        top_k=top_k, width=width, experts=experts, ep=ep,
        source_partitions=source_partitions, received_assignments=received_assignments,
        dtype=dtype, source_routing=source_routing)
