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
from mindie_sim.prediction.operators.mc2 import mc2_local_caller
from ...mechanism import OperatorPlan, Work
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    COLLECTIVES, COMBINE, DISPATCH, FLOAT_ROWS, MMAR, Mc2Contract, PeerRoute, Phase, Transport,
    build_transport_stages, capability_gap, contract_fields, geometry_has_l2, make_plan,
    native_protocol_peer_transport, topology_hops, validate_contract, validate_native_dispatch_quant,
)
from .lowering import A2_MTE, SIO_HCCS, declared_incoming_routes

TARGET_BACKEND = "ascend"
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
        relative_path="mc2/matmul_all_reduce/op_kernel/arch22/matmul_all_reduce_base.h",
        source_sha256="e12221d971267ec5302d7fa6db853df366f92daaa715dc5956981e5af7de0a6b",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch22/moe_distribute_dispatch_v2_tiling_arch22.cpp",
        source_sha256="d3322d5b3a664c7f0e2cd3bab9718a464b8f15be09426d3fa5c0ad001334aa91",
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
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/mega_moe/op_kernel/arch22/mega_moe.cpp",
        source_sha256="0a8d7824e28e93aef07c4c2b67a50afdc29989bd55820e397d3580d88baecd4c",
    ),
    SourceRef(
        repository="cann/ascendc-mc2-best-practice",
        revision=_SKILLS_REVISION,
        relative_path="ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/all_gather_matmul/x_all_gather_matmul/op_host/op_tiling/arch22/x_all_gather_matmul_tiling_a2a3.cpp",
        source_sha256="48544c803eb2be74234c99075bb32cf952faefcae9a8848d213dbe3d0cc3aa2f",
    ),
    SourceRef(
        repository="cann/ascendc-mc2-best-practice",
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
ARCH_TRANSPORTS = frozenset({HCCL_CLIENT, SIO_HCCS, A2_MTE})
SERVER_MODES = frozenset({"server", "aicpu", "aiv", "hccl_server_type_aicpu", "hccl_server_type_aiv"})
_ARCH22_COLLECTIVE_WORLD_SIZES = frozenset({1, 2, 4, 8})
_ARCH22_COLLECTIVE_DTYPES = FLOAT_ROWS | {"int8", "int4"}


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
    elif abi == A2_MTE:
        if contract.abi in COLLECTIVES:
            raise capability_gap(op, ctx, "mte_window_requires_moe_dispatch_or_combine")
        routes = tuple(PeerRoute(contract.rank, peer, hops.get((contract.rank, peer), ("HCCS_MTE_WINDOW",)))
                       for peer in peers)
    else:
        if algorithm in {"ring", "two_tier"} and any((contract.rank, peer) not in hops for peer in peers):
            raise capability_gap(op, ctx, "a3_algorithm_requires_group_hops")
        routes = _sio_hccs_hops(contract, fields, group, hops, op, ctx)
    if contract.internal_chunks >= 2 and not (
        (contract.independent_buffers or binding.get("independent_buffers") is True)
        and (contract.independent_handles or binding.get("independent_handles") is True)
    ):
        raise capability_gap(op, ctx, "two_chunks_require_independent_buffers_and_handles")
    credits = 1 if algorithm == "ring" else max(1, len(peers))
    return Transport(
        abi, algorithm, routes, credits=credits,
        slot_depth=min(contract.internal_chunks, 2), peer_limit=len(peers),
        aiv_staging=True,
        direct_remote_gm=abi == A2_MTE and contract.abi == COMBINE,
        gm_bounce_hbm=contract.abi in COLLECTIVES,
        incoming_routes=declared_incoming_routes(contract.rank, peers, hops),
    )


def _validate_arch22_contract(contract: Mc2Contract, op: OpSpec, ctx: OperatorPlanningContext) -> None:
    if contract.abi in COLLECTIVES:
        if contract.world_size not in _ARCH22_COLLECTIVE_WORLD_SIZES:
            raise capability_gap(op, ctx, f"arch22_collective_world_size_unsupported:{contract.world_size}")
        if contract.dtype not in _ARCH22_COLLECTIVE_DTYPES:
            raise capability_gap(op, ctx, f"arch22_collective_dtype_unsupported:{contract.dtype}")
    if contract.abi == DISPATCH and contract.row_layout is not None:
        layout = contract.row_layout
        unquantized_float = (
            layout.quant_mode == "none"
            and layout.input_dtype in FLOAT_ROWS
            and layout.payload_dtype == layout.input_dtype
            and layout.input_scale_bytes == 0
        )
        per_token_int8 = (
            layout.quant_mode == "per_token"
            and layout.input_dtype in FLOAT_ROWS
            and layout.payload_dtype == "int8"
        )
        if not (unquantized_float or per_token_int8):
            raise capability_gap(
                op, ctx, f"arch22_dispatch_quant_unsupported:{layout.quant_mode}:{layout.payload_dtype}",
            )


@dataclass(frozen=True, slots=True)
class _Strategy:
    arch_id: ArchId = ArchId("npu_arch_2201")
    abi: str = "aclnnAllGatherMatmulV2"
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        try:
            contract = validate_contract(op, ctx)
            _validate_arch22_contract(contract, op, ctx)
            _transport(contract, op, ctx)
        except ModelingError:
            return False
        return True

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        if ctx.compute_arch.arch_id != self.arch_id:
            raise capability_gap(op, ctx, "unsupported_compute_architecture")
        contract = validate_contract(op, ctx)
        _validate_arch22_contract(contract, op, ctx)
        transport = _transport(contract, op, ctx)
        fields = contract_fields(op)
        local = fields.get("mc2_v2_local") if op.recipe == "arch22_v2_caller_local" else None
        if local is None:
            phases = build_transport_stages(contract, transport, has_l2=geometry_has_l2(ctx.geometry))
        else:
            group = dict(fields.get("device_kernel_group", ()))
            name = "caller_publish" if group.get("exit") else "caller_pack"
            phases = (Phase(name, work=Work(resources=dict(local["work"])), owner="aiv"),)
        return make_plan(contract, transport, phases, self.sources, arch_id=self.arch_id)


STRATEGY: ArchStrategy = _Strategy()

NATIVE_PROTOCOL_KIND = "arch22_v2_caller"
DISPATCH_FFN_COMBINE_ABI = "aclnnDispatchFFNCombine"
DISPATCH_FFN_COMBINE_W4A8_ABI = "aclnnDispatchFFNCombineW4A8"
DISPATCH_FFN_COMBINE_ENTRY = "torch_npu.npu_moe_distribute_dispatch_ffn_combine"
MEGAMOE_ABI = "aclnnMegaMoe"
MEGAMOE_ENTRY = "cann_ops_transformer.ops.mega_moe"
_FUSED_MC2_MODE_DISPATCH = 1
_FUSED_MC2_MODE_MEGAMOE = 2
_INT4_WEIGHT_FORMATS = frozenset({"int4", "w4a8"})
_DISPATCH_FFN_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h",
    "source_sha256": "b22b033d36406958bdf87de1489d207377836b0dfbfbc17148743952ddd35094",
}


def fused_mc2_native_binding(weight_dtype: str = "int8", *, fused_level: int = _FUSED_MC2_MODE_DISPATCH) -> dict:
    """Resolve the exact Arch22 (A3) fused MoE MC2 public identity and ACLNN symbol."""
    level = int(fused_level or _FUSED_MC2_MODE_DISPATCH)
    if level > _FUSED_MC2_MODE_MEGAMOE:
        raise ModelingError("arch22_fused_mc2_mode3_requires_arch35")
    if level == _FUSED_MC2_MODE_MEGAMOE:
        return {
            "public_name": MEGAMOE_ABI,
            "api": {
                "namespace": "aclnn",
                "entry": MEGAMOE_ENTRY,
                "aclnn_symbol": MEGAMOE_ABI,
            },
            "source": _DISPATCH_FFN_SOURCE,
            "execution_core": "mix",
        }
    w_fmt = str(weight_dtype or "int8").strip().lower()
    is_w4 = w_fmt in _INT4_WEIGHT_FORMATS
    public_name = "dispatch_ffn_combine_w4_a8" if is_w4 else "dispatch_ffn_combine"
    symbol = DISPATCH_FFN_COMBINE_W4A8_ABI if is_w4 else DISPATCH_FFN_COMBINE_ABI
    return {
        "public_name": public_name,
        "api": {
            "namespace": "aclnn",
            "entry": DISPATCH_FFN_COMBINE_ENTRY,
            "aclnn_symbol": symbol,
        },
        "source": _DISPATCH_FFN_SOURCE,
        "execution_core": "mix",
    }


def common_native_protocol(chosen, fused_level, moe_comm):
    """Describe the Arch22 (A3) protocol selected by common MoE communication selection."""
    return {
        "kind": NATIVE_PROTOCOL_KIND,
        "arch_family": "A3",
        "dispatch_quant": "per_token_int8",
        "max_inflight_per_rank": None,
        "selection": "moe_comm_config",
        "selected_method": chosen,
        "fused_mc2_level": int(fused_level or 0),
        "moe_comm": moe_comm,
    }


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


def _rebind_fused_mc2_op(
    ops, fused_level: int, weight_dtype: str, *, experts: int = 0, ep: int = 1,
):
    """Project Arch22 (A3) fused MoE native binding onto the lowered fused MC2 op."""
    from dataclasses import replace
    from . import megamoe as mc2_megamoe

    binding = fused_mc2_native_binding(weight_dtype, fused_level=fused_level)
    public_name = binding["public_name"]
    api = binding["api"]
    is_w4 = str(weight_dtype or "int8").strip().lower() in _INT4_WEIGHT_FORMATS
    local_experts = max(1, int(experts) // max(1, int(ep))) if int(experts or 0) > 0 else 1
    rebuilt = []
    for item in ops:
        if item.recipe != "megamoe_arch35":
            rebuilt.append(item)
            continue
        meta = dict(item.metadata)
        meta["native_kind"] = public_name
        meta["aclnn_symbol"] = api["aclnn_symbol"]
        meta["native_source"] = binding["source"]
        meta["fused_mc2_level"] = int(fused_level)
        meta["native_call"] = {
            "api": dict(api),
            "source": binding["source"],
            "kernels": ({"name": public_name, "role": "fused_moe_mc2", "evidence": "source_declared"},),
            "execution_core": binding["execution_core"],
        }
        geo = dict(meta.get("megamoe_geometry") or {})
        raw_group_rows = meta.get("group_rows")
        has_explicit_groups = (
            isinstance(raw_group_rows, (list, tuple)) and len(raw_group_rows) > 0
        )
        routed_float = max(0.0, float(geo.get("routed_rows", 0.0) or 0.0))
        if has_explicit_groups:
            group_rows = tuple(max(0, int(round(float(x)))) for x in raw_group_rows)
            routed_int = sum(group_rows)
        elif routed_float > 0.0:
            routed_int = max(1, int(round(routed_float)))
            q_rows, rem_rows = divmod(routed_int, local_experts)
            group_rows = tuple(q_rows + (1 if idx < rem_rows else 0) for idx in range(local_experts))
            if abs(routed_float - routed_int) > 0.0:
                meta["expected_row_scale"] = routed_float / routed_int
        else:
            routed_int = 0
            group_rows = (0,) * local_experts
        geo["routed_rows"] = routed_int
        geo["dispatch_link_bytes"] = max(0.0, float(geo.get("dispatch_link_bytes", 0.0) or 0.0))
        geo["combine_link_bytes"] = max(0.0, float(geo.get("combine_link_bytes", 0.0) or 0.0))
        act = str(geo.get("activation") or "swiglu").lower()
        inter_raw = int(geo.get("moe_intermediate", 0) or 0)
        if inter_raw > 0 and act in mc2_megamoe._ELEMENTWISE_ACTIVATIONS:
            geo["moe_intermediate"] = mc2_megamoe._gmm2_in_dim(inter_raw, act)
        meta["group_rows"] = group_rows
        meta["input_format"] = "int8"
        meta["dtype_a"] = "int8"
        meta["layout_b"] = "FRACTAL_NZ"
        op_flops = item.flops
        matrix_flops = item.matrix_flops
        op_vector_ops = float(item.vector_ops)
        g1_msd = bool(geo.get("gmm1_msd_w4a8"))
        g2_msd = bool(geo.get("gmm2_msd_w4a8"))
        newly_enabled_w4 = is_w4 and not (g1_msd or g2_msd)
        if newly_enabled_w4:
            logical = float(geo.get("logical_flops") or item.flops)
            g1_msd = g2_msd = True
            geo["gmm1_msd_w4a8"] = True
            geo["gmm2_msd_w4a8"] = True
            geo["logical_flops"] = logical
            meta["logical_flops"] = logical
            op_flops = logical * mc2_megamoe.MSD_INT4_PASSES
            matrix_flops = (("int4", op_flops),)
        hidden_dim = int(geo.get("hidden", 0) or 0)
        gmm2_in_dim = (
            mc2_megamoe._gmm2_in_dim(inter_raw, act) if inter_raw > 0 else 0
        )
        if g1_msd or g2_msd:
            split_ops = (
                (routed_float * hidden_dim * mc2_megamoe.MSD_INT4_PASSES if g1_msd else 0.0)
                + (routed_float * gmm2_in_dim * mc2_megamoe.MSD_INT4_PASSES if g2_msd else 0.0)
            )
            merge_added = (
                (
                    routed_float * (mc2_megamoe.MSD_INT4_PASSES * inter_raw) * mc2_megamoe.MSD_MERGE_OPS_PER_ELEM
                    + routed_float * hidden_dim * mc2_megamoe.MSD_MERGE_OPS_PER_ELEM
                )
                if newly_enabled_w4 else 0.0
            )
            op_vector_ops += split_ops + merge_added
        meta["gmm1_msd_w4a8"] = g1_msd
        meta["gmm2_msd_w4a8"] = g2_msd
        if g1_msd and g2_msd:
            geo["compute_dtype"] = "w4a8"
            meta["dtype_b"] = "int4"
            meta["compute_format"] = "w4a8"
            meta["msd_w4a8"] = True
        else:
            geo["compute_dtype"] = "int8"
            meta["dtype_b"] = "int8"
            meta["compute_format"] = "int8"
        meta["megamoe_geometry"] = geo
        rebuilt.append(replace(
            item, recipe="mega_moe", flops=op_flops, matrix_flops=matrix_flops,
            vector_ops=op_vector_ops, metadata=tuple(meta.items()),
        ))
    return rebuilt


def lower_native_protocol(
    ops, *, selected, runtime, prefix, tokens, top_k, width, experts, ep,
    source_partitions, received_assignments, dtype, source_routing=None,
    valid_tokens=None, source_rows_by_replica=None, destination_rows_by_rank=None,
):
    if selected.get("selection") == "moe_comm_config":
        by_id = {op.op_id: op for op in ops}
        gate = by_id.get(prefix + ".expert_up")
        gmeta = dict(gate.metadata) if gate is not None else {}
        if selected.get("selected_method") == "fused_mc2" and received_assignments:
            parallel_cfg = getattr(runtime, "parallel", None)
            has_redundant = int(getattr(parallel_cfg, "redundant_experts_per_rank", 0) or 0) > 0
            if not has_redundant and all(
                prefix + sfx in by_id for sfx in (".pack", ".dispatch", ".combine", ".unpack")
            ):
                from . import megamoe as mc2_megamoe

                fused_lvl = min(
                    _FUSED_MC2_MODE_MEGAMOE,
                    max(_FUSED_MC2_MODE_DISPATCH, int(selected.get("fused_mc2_level") or _FUSED_MC2_MODE_DISPATCH)),
                )
                pack, dispatch, combine, unpack = (
                    by_id[prefix + sfx] for sfx in (".pack", ".dispatch", ".combine", ".unpack")
                )
                fused = mc2_megamoe.try_lower_fused_frame(
                    ops, by_id, {**selected, "fused_mc2_level": fused_lvl},
                    runtime, prefix, tokens, width, top_k, received_assignments,
                    ep, dtype, pack, dispatch, combine, unpack, by_id.get(prefix + ".pack.quantize"),
                    allow_int8_arch22=True,
                )
                if fused is not None:
                    w_fmt = str(gmeta.get("weight_format") or gmeta.get("activation_format") or "int8")
                    return _rebind_fused_mc2_op(fused, fused_lvl, w_fmt, experts=experts, ep=ep)
        if gmeta.get("activation_format") != "int8":
            return ops
        if selected.get("max_inflight_per_rank") is None:
            selected = {**selected, "max_inflight_per_rank": max(1, int(ep) - 1)}
    return mc2_local_caller.lower(
        ops, selected=selected, runtime=runtime, prefix=prefix, tokens=tokens,
        top_k=top_k, width=width, experts=experts, ep=ep,
        source_partitions=source_partitions, received_assignments=received_assignments,
        dtype=dtype, source_routing=source_routing)
