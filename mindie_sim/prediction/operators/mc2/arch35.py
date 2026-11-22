"""Arch35 MC2 direct peer, MTE window, and nonblocking CCU strategies.

[Role] Bind declared Arch35 transport and topology to the shared MC2 ledger.
[Invariants] Peer credits bound live windows and injection. Window publication
  follows payload completion; CCU completion gates its consuming core.
[Timeouts & Locks] Finite group and cohort planning; no I/O or locks.
[Side-Effect Policy] Produces analytical declarations only.

Upstream-ABI: aclnnMoeDistributeDispatchV2
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from . import megamoe as mc2_megamoe
from mindie_sim.prediction.operators.mc2.mc2_phase import (
    RECIPE, build as build_fullmesh, contract, declared_floor_work,
)
from ...mechanism import OperatorPlan, PlannedRecipe
from mindie_sim.prediction.operators.mc2.moe_comm_selection import FUSED_MC2
from ..contracts import ArchStrategy, OperatorPlanningContext
from mindie_sim.prediction.operators.mc2.mc2_fullmesh import SOURCE, combine_layout, dispatch_layout
from mindie_sim.prediction.operators.mc2.mc2_mx_quant import mx32_work
from .common import (
    COLLECTIVES, COMBINE, FLOAT_ROWS, MMAR, MMRS, Mc2Contract, PeerRoute, Transport, build_transport_stages, capability_gap,
    contract_fields, make_plan, native_protocol_peer_transport, topology_hops,
    validate_contract, validate_native_dispatch_quant,
)
from .lowering import A5_ROCE, CCU, SHMEM, declared_incoming_routes

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
        relative_path="mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_based_all_reduce.h",
        source_sha256="2af6e2f75f0fdf99778730665c5ad0d217b8f9910358fa324c25171500879c6e",
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


MTE_WINDOW = "a5_mte"
URMA_ALIGN_BYTES = 512
MESH_RANK_LIMIT = 8
ARCH_TRANSPORTS = frozenset({SHMEM, MTE_WINDOW, CCU, A5_ROCE})

NATIVE_PROTOCOL_KIND = "arch35_fullmesh"
NETWORK = "arch35_fullmesh_peer_channel"
NO_ASSIGNMENTS = 0
MX_DISPATCH_GROUP = 32
FUSED_MC2_ROUTED_MODE = 2
FUSED_MC2_SHARED_MODE = 3


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
    if contract.internal_chunks >= 2 and not all(
        binding.get(key, fields.get(key, native_chunking)) is True
        for key in ("independent_buffers", "independent_handles")
    ):
        raise capability_gap(op, ctx, "two_chunks_require_independent_buffers_and_handles")

    route_kind = "ROCE" if abi == A5_ROCE else ("UB_WINDOW" if abi == MTE_WINDOW else "URMA")
    routes = tuple(PeerRoute(rank, peer, hops.get((rank, peer), (route_kind,))) for peer in peers)
    credits = binding.get("peer_credits", fields.get("peer_credits", max(1, len(peers))))
    if type(credits) is not int or credits < 1:
        raise capability_gap(op, ctx, "invalid_peer_credit_limit")
    depth = min(contract.internal_chunks, 2) if contract.abi in COLLECTIVES else (2 if (
        contract.independent_buffers and contract.independent_handles
    ) else 1)
    return Transport(
        abi, algorithm, routes, wire_alignment=URMA_ALIGN_BYTES if abi == SHMEM else 1,
        window_alignment=URMA_ALIGN_BYTES if abi == SHMEM else 1, window_offset=offset,
        credits=1 if algorithm == "ring" else min(credits, max(1, len(peers))),
        slot_depth=depth, peer_limit=len(peers),
        ccu=abi == CCU, aiv_staging=abi == MTE_WINDOW,
        direct_remote_gm=abi == MTE_WINDOW and contract.abi == COMBINE,
        gm_bounce_hbm=False,
        incoming_routes=declared_incoming_routes(rank, peers, hops),
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
        scatter = contract.abi in {MMRS, MMAR} and contract.world_size > 1
        temp_gm = scatter and contract.dtype in FLOAT_ROWS
        ub_output = contract.abi == MMRS and scatter and not temp_gm and ctx.datapath.has_l0c_to_ub
        phases = build_transport_stages(contract, transport, ub_output=ub_output, temp_gm=temp_gm)
        return make_plan(contract, transport, phases, self.sources, arch_id=self.arch_id)


STRATEGY: ArchStrategy = _Strategy()


def validate_native_protocol(raw):
    if "global_bs" in raw:
        raise ModelingError("arch22_v2_global_bs_requires_nonnegative_integer")
    validate_native_dispatch_quant(raw)
    native_protocol_peer_transport(raw)


def native_protocol_recipes(selected):
    # Common MoE communication selection can use the planner without an
    # explicit kernel declaration. Only explicit declarations bind a window.
    result = {RECIPE: PlannedRecipe(build_fullmesh), **mc2_megamoe.recipes()}
    if selected is not None:
        result[NETWORK] = native_protocol_peer_transport(selected)
    return result


def common_native_protocol(chosen, fused_level, moe_comm):
    return {"kind": NATIVE_PROTOCOL_KIND, "max_inflight_per_rank": None,
            "chunk_bytes": None, "hop_startup_us": 0.0,
            "receive_capacity_bytes_per_peer": None,
            "selection": "moe_comm_config", "selected_method": chosen,
            "fused_mc2_level": fused_level, "moe_comm": moe_comm}


def _absorbed_quant_mode(routing):
    """Dispatch payload mode declared by one absorbed init-routing pack."""
    payload = str(routing.get("payload_format") or "")
    scale_format = str(routing.get("scale_format") or "")
    if payload in {"fp8_e4m3", "fp8_e5m2"} and scale_format in {"fp8_e8m0", "e8m0"}:
        return "mx"
    if payload == "int8":
        return "per_token_int8"
    raise ModelingError("mc2_init_routing_absorption_requires_mx_or_per_token_int8_payload")


def _received_quantizer(pack_quant, gate, dispatch_id, rows, width):
    """Keep receiver conversion when native dispatch publishes float rows."""
    from mindie_sim.catalog.ops import vector_op

    meta = dict(pack_quant.metadata)
    source_rows = float(meta.get("source_rows") or NO_ASSIGNMENTS)
    if source_rows <= NO_ASSIGNMENTS:
        raise ModelingError("mc2_pack_quantizer_requires_source_rows")
    scale = float(rows) / source_rows
    scale_shape = tuple(meta["activation_scale_shape"])
    _, *scale_tail = scale_shape
    meta.update(
        source_rows=rows,
        activation_scale_shape=(math.ceil(rows), *scale_tail),
        activation_scale_bytes=float(meta["activation_scale_bytes"]) * scale,
        quantization_scope="moe_received_expert_input",
    )
    quantizer = vector_op(
        op_id=dispatch_id + ".received_quantize", deps=(dispatch_id,),
        stage=pack_quant.stage, layer=pack_quant.layer,
        requests=pack_quant.requests, phase=pack_quant.phase,
        family=pack_quant.family, recipe=pack_quant.recipe,
        dtype=pack_quant.dtype,
        elements=math.ceil(rows) * width, ops_per_element=NO_ASSIGNMENTS,
        vector_ops_override=pack_quant.vector_ops * scale,
        read_bytes_override=pack_quant.mte2_bytes * scale,
        write_bytes_override=pack_quant.mte3_bytes * scale,
        shape=(math.ceil(rows), width), returns=pack_quant.returns,
        metadata=tuple(meta.items()),
    )
    gate_meta = dict(gate.metadata)
    if gate_meta.get("activation_format") != meta["output_format"]:
        raise ModelingError("mc2_received_quantizer_format_mismatch")
    gate_meta.update(
        activation_producer="explicit_routing_quantization",
        activation_payload_producer=quantizer.op_id,
        activation_scale_producer=quantizer.op_id,
    )
    return quantizer, replace(
        gate, deps=tuple(quantizer.op_id if dep == dispatch_id else dep
                         for dep in gate.deps),
        metadata=tuple(gate_meta.items()))


def lower_native_protocol(
    ops, *, selected, runtime, prefix, tokens, top_k, width, experts, ep,
    source_partitions, received_assignments, dtype, source_routing=None,
    valid_tokens=None, source_rows_by_replica=None, destination_rows_by_rank=None,
):
    if (source_partitions < 1 or ep < 2 or dtype not in {"bf16", "fp16"}
            or getattr(runtime, "moe_recipe", "alltoall") != "alltoall"):
        raise ModelingError("mc2_fullmesh_requires_unpartitioned_float_rows_and_ep_alltoall")
    if runtime.kernels.get("moe_transport"):
        raise ModelingError("mc2_fullmesh_cannot_duplicate_separate_quantized_transport")
    by_id = {op.op_id: op for op in ops}
    pack, dispatch, combine, unpack = (by_id[prefix + suffix]
                                      for suffix in (".pack", ".dispatch", ".combine", ".unpack"))
    pack_quant = by_id.get(prefix + ".pack.quantize")
    if pack_quant is not None and (
            pack_quant.plan_required
            or pack_quant.deps != (pack.op_id,)
            or dict(pack_quant.metadata).get("quantization_scope")
            != "moe_dispatch_expert_input"
            or any(pack_quant.op_id in item.deps and item.op_id != dispatch.op_id
                   for item in ops)):
        raise ModelingError("mc2_cannot_absorb_external_pack_quantizer")
    source_metadata = source_routing.to_dict() if source_routing is not None else None
    # TPxDP without an explicit slice still publishes one source rank's rows.
    if source_routing is not None:
        valid_source_tokens = source_routing.valid_local_tokens
    elif valid_tokens is not None:
        # The caller's actual logical rows, not a physical padding count.
        valid_source_tokens = valid_tokens
    else:
        valid_source_tokens = tokens
    if source_partitions > 1:
        tokens = math.ceil(tokens / source_partitions)
        if source_routing is None:
            valid_source_tokens = (valid_tokens / source_partitions
                                   if valid_tokens is not None else tokens)
    send = valid_source_tokens * top_k
    quant, absorbed = _dispatch_quant_mode(selected, pack, pack_quant, by_id, prefix, dispatch, width)
    parallel_cfg = getattr(runtime, "parallel", None)
    has_redundant_slots = int(getattr(parallel_cfg, "redundant_experts_per_rank", 0) or 0) > 0
    if (selected.get("selected_method") == FUSED_MC2
            and received_assignments
            and not has_redundant_slots):
        megamoe = _try_lower_megamoe(
            ops, by_id, selected, runtime, prefix, tokens, width, top_k, received_assignments,
            ep, dtype, pack, dispatch, combine, unpack, pack_quant,
        )
        if megamoe is not None:
            return megamoe
    received_quant = None
    if pack_quant is not None and quant == "none" and received_assignments:
        gate = by_id.get(prefix + ".expert_up")
        if gate is None or gate.deps != (dispatch.op_id,):
            raise ModelingError("mc2_received_quantizer_requires_direct_expert")
        received_quant, by_id[gate.op_id] = _received_quantizer(
            pack_quant, gate, dispatch.op_id, received_assignments, width)
    payload_dtype = {"mx": "fp8_e4m3", "per_token_int8": "int8"}.get(quant, dtype)
    quant_mode = "per_token" if quant == "per_token_int8" else quant
    layouts = (dispatch_layout(width, input_dtype=dtype, payload_dtype=payload_dtype,
                               quant_mode=quant_mode),
               combine_layout(width, dtype=dtype))
    if quant in {"mx", "per_token_int8"}:
        _bind_quantized_expert_input(
            by_id, prefix, dispatch, received_assignments, width, dtype,
            quant, payload_dtype, layouts,
        )
    row_format = {"forward_bytes": layouts[0].packet_bytes,
                  "reverse_bytes": layouts[1].packet_bytes}
    forward_raw, backward_raw, payload_basis, source_key = _routing_contracts(
        source_routing, source_rows_by_replica, destination_rows_by_rank,
        ep, width, dtype, tokens, valid_source_tokens, top_k, experts,
        quant, quant_mode, payload_dtype, layouts, send, received_assignments,
    )
    dispatch_op, combine_op = _native_dispatch_combine(
        pack, dispatch, combine, unpack, prefix, dtype, width, ep, selected,
        quant, payload_dtype, layouts, received_assignments, absorbed,
        forward_raw, backward_raw, source_routing, source_key, source_metadata,
        row_format, source_partitions, payload_basis,
    )
    return _replace_protocol_ops(
        ops, by_id, pack, dispatch, combine, unpack, pack_quant,
        dispatch_op, combine_op, received_quant,
    )


def _dispatch_quant_mode(selected, pack, pack_quant, by_id, prefix, dispatch, width):
    quant = selected.get("dispatch_quant", "none")
    # ``aclnnMoeInitRoutingV3`` may own the dispatch-input quantization
    # (``quant_mode=1``) instead of a separate ``pack.quantize`` node.  The
    # sender then publishes quantized rows and the receiver scales come from
    # that pack, so no received quantizer is synthesized here.
    absorbed = dict(pack.metadata).get("moe_init_routing")
    if absorbed is not None:
        if pack_quant is not None:
            raise ModelingError("mc2_init_routing_absorption_conflicts_with_pack_quantizer")
        absorbed_quant = _absorbed_quant_mode(absorbed)
        if quant not in {"none", absorbed_quant}:
            raise ModelingError("mc2_init_routing_absorption_conflicts_with_dispatch_quant")
        quant = absorbed_quant
    if (selected.get("selection") == "moe_comm_config"
            and "dispatch_quant" not in selected
            and pack_quant is not None
            and absorbed is None):
        pmeta = dict(pack_quant.metadata)
        gate_cand = by_id.get(prefix + ".expert_up")
        gmeta = dict(gate_cand.metadata) if gate_cand is not None else {}
        if (gate_cand is not None and gate_cand.deps == (dispatch.op_id,)
                and gate_cand.local_shape[-1] == width):
            if (pmeta.get("output_format") in {"fp8_e4m3", "fp8_e5m2"}
                    and pmeta.get("activation_group_size") == MX_DISPATCH_GROUP
                    and pmeta.get("activation_scale_format") in {"fp8_e8m0", "e8m0"}
                    and gmeta.get("activation_format") == pmeta.get("output_format")
                    and gmeta.get("activation_group_size") == MX_DISPATCH_GROUP
                    and gmeta.get("activation_scale_format") == "fp8_e8m0"):
                quant = "mx"
            elif (pmeta.get("output_format") == "int8"
                  and pmeta.get("activation_group_size") == width
                  and pmeta.get("activation_scale_format") == "fp32"
                  and gmeta.get("activation_format") == "int8"):
                quant = "per_token_int8"
    return quant, absorbed


def _allow_megamoe_abi_extension(runtime) -> bool:
    """Return True when the active feature track or runtime explicitly permits ABI extensions."""
    rmeta = dict(getattr(runtime, "metadata", ()) or ())
    if bool(rmeta.get("allow_megamoe_abi_extension", False)) or bool(
        rmeta.get("allow_extended_activation_abi", False)
    ):
        return True
    report = rmeta.get("feature_track_report")
    track = (
        (report.get("feature_track") if isinstance(report, Mapping) else None)
        or rmeta.get("feature_track")
        or ""
    )
    return str(track).strip().lower() == "expected_ideal"


def _try_lower_megamoe(
    ops, by_id, selected, runtime, prefix, tokens, width, top_k, received_assignments,
    ep, dtype, pack, dispatch, combine, unpack, pack_quant,
):
    e_up = by_id.get(prefix + ".expert_up")
    e_act = by_id.get(prefix + ".expert_act")
    e_down = by_id.get(prefix + ".expert_down")
    up_meta = dict(e_up.metadata) if e_up is not None else {}
    e_comp = str(up_meta.get("compute_format") or "").lower()
    e_wfmt = str(up_meta.get("weight_format") or "").lower()
    mx_act = (
        up_meta.get("activation_group_size") == MX_DISPATCH_GROUP
        and up_meta.get("activation_scale_format") in {"fp8_e8m0", "e8m0"}
    )
    mxfp_eligible = mx_act and (
        e_comp in {"fp8_e4m3", "mxfp4", "fp4_e2m1"}
        or e_wfmt in {"mxfp8", "mxfp4", "fp8_e4m3", "fp4_e2m1"}
    )
    if not (e_up is not None and e_act is not None and e_down is not None
            and e_up.deps == (dispatch.op_id,)
            and e_act.deps == (e_up.op_id,)
            and e_down.deps == (e_act.op_id,)
            and combine.deps == (e_down.op_id,)
            and mxfp_eligible):
        return None
    act_meta = dict(e_act.metadata)
    act_kind = str(act_meta.get("activation_kind") or e_act.family or "swiglu").lower()
    allow_ext = _allow_megamoe_abi_extension(runtime)
    allowed_acts = mc2_megamoe.ALL_ACTIVATIONS if allow_ext else mc2_megamoe.SUPPORTED_ACTIVATIONS
    if act_kind not in allowed_acts:
        return None

    fused_lvl = int(selected.get("fused_mc2_level") or FUSED_MC2_ROUTED_MODE)
    fused_lvl = FUSED_MC2_SHARED_MODE if fused_lvl >= FUSED_MC2_SHARED_MODE else FUSED_MC2_ROUTED_MODE
    moe_inter = int(e_down.local_shape[-1])
    down_meta = dict(e_down.metadata)
    w13_b = float(up_meta.get("packed_weight_bytes") or max(0.0, e_up.mte2_bytes - received_assignments * width))
    w2_b = float(down_meta.get("packed_weight_bytes") or max(0.0, e_down.mte2_bytes - received_assignments * moe_inter))
    sh_ops = [item for item in ops if item.op_id.startswith(prefix + ".shared.")]
    join_op = by_id.get(prefix + ".join")
    sh_up = by_id.get(prefix + ".shared.gate_up")
    sh_down = by_id.get(prefix + ".shared.down")
    fuse_shared = (
        fused_lvl == FUSED_MC2_SHARED_MODE
        and bool(sh_ops) and sh_up is not None
        and sh_down is not None and join_op is not None
        and len(join_op.deps) == 2 and unpack.op_id in join_op.deps
        and any(dep.startswith(prefix + ".shared.") for dep in join_op.deps)
    )
    sh_inter = int(sh_up.local_shape[1] // 2) if fuse_shared else 0
    sh_w13 = float(dict(sh_up.metadata).get("packed_weight_bytes") or 0.0) if fuse_shared else 0.0
    sh_w2 = float(dict(sh_down.metadata).get("packed_weight_bytes") or 0.0) if fuse_shared else 0.0
    removed_ids = {
        pack.op_id, dispatch.op_id, e_up.op_id, e_act.op_id,
        e_down.op_id, combine.op_id, unpack.op_id,
    }
    if pack_quant is not None:
        removed_ids.add(pack_quant.op_id)
    target_id = unpack.op_id
    mega_deps = list(pack.deps)
    for dep in unpack.deps:
        if dep not in removed_ids and dep not in mega_deps:
            mega_deps.append(dep)
    if fuse_shared:
        sh_ids = {item.op_id for item in sh_ops}
        removed_ids.update(sh_ids)
        removed_ids.add(join_op.op_id)
        target_id = join_op.op_id
        for item in sh_ops:
            for dep in item.deps:
                if dep not in removed_ids and dep not in mega_deps:
                    mega_deps.append(dep)
    if any(
        dep in removed_ids and dep != target_id
        for item in ops if item.op_id not in removed_ids
        for dep in item.deps
    ):
        return None

    mega_op = mc2_megamoe.make_op(
        op_id=target_id, deps=tuple(mega_deps),
        stage=unpack.stage, layer=unpack.layer,
        requests=unpack.requests, phase=unpack.phase,
        tokens=int(tokens), hidden=int(width),
        moe_inter=moe_inter, top_k=int(top_k),
        routed_rows=float(received_assignments),
        w13_bytes=w13_b, w2_bytes=w2_b,
        enable_fused_mc2=fused_lvl if fuse_shared else FUSED_MC2_ROUTED_MODE,
        ep=int(ep), activation=act_kind,
        compute_dtype="fp8_e4m3", output_dtype=dtype,
        dispatch_link_bytes=float(dispatch.link_bytes),
        combine_link_bytes=float(combine.link_bytes),
        shared_inter=sh_inter, shared_w13_bytes=sh_w13,
        shared_w2_bytes=sh_w2,
        allow_abi_extension=allow_ext,
        activation_params=act_meta.get("activation_params"),
        extra_metadata=(
            ("device_kernel_group", {
                "id": prefix + ".megamoe",
                "entry": True,
                "exit": True,
                "kernel": mc2_megamoe.NATIVE_KIND,
            }),
        ),
    )
    rebuilt = []
    inserted = False
    absorbed_groups = [
        dict(group)
        for item in ops if item.op_id in removed_ids
        for group in dict(item.metadata).get("structure_subgraphs", ())
        if isinstance(group, dict) and str(group.get("role", "")).startswith("grouped_gate_up")
    ]
    for item in ops:
        if item.op_id == target_id:
            rebuilt.append(mega_op)
            inserted = True
        elif item.op_id in removed_ids:
            continue
        else:
            rebuilt.append(item)
    if inserted:
        if absorbed_groups:
            from mindie_sim.prediction.structure_groups import declare_group
            for group in absorbed_groups:
                impl = dict(group.get("implementation") or {})
                impl["native_kind"] = mc2_megamoe.NATIVE_KIND
                declare_group(
                    rebuilt, key=group["key"], role=group["role"],
                    family=group.get("family", "moe"),
                    members=(mega_op.op_id,), entries=(mega_op.op_id,),
                    anchor=mega_op.op_id, block="ffn", scope_role="body",
                    view=group.get("tensor_view", {}),
                    implementation=impl, source=group.get("source"),
                    coverage=group.get("coverage"),
                    fusion_status="fused_existing",
                    fusion_reason="megamoe_arch35_fused_mc2_kernel",
                    native_fusion_eligible=False,
                )
        return rebuilt


def _bind_quantized_expert_input(
    by_id, prefix, dispatch, received_assignments, width, dtype,
    quant, payload_dtype, layouts,
):
    if quant == "mx":
        mx32_work(width, dtype, payload_dtype)
    gate = by_id.get(prefix + ".expert_up")
    if gate is None:
        if received_assignments:
            raise ModelingError("mc2_quantized_requires_direct_expert_input_consumer")
    else:
        meta = dict(gate.metadata)
        # This dispatch owns exactly this consumer's input representation.
        # A transport selection cannot suppress an unrelated conversion.
        direct = (gate.deps == (dispatch.op_id,) and gate.local_shape[-1] == width
                  and meta.get("activation_format") == payload_dtype)
        if quant == "mx":
            matched = (meta.get("activation_group_size") == MX_DISPATCH_GROUP
                       and meta.get("activation_scale_format") == "fp8_e8m0"
                       and tuple(meta.get("activation_scale_shape", ()))[1:] == (width // (2 * MX_DISPATCH_GROUP), 2))
            requirement = "matching_e4m3_e8m0_group32"
        else:
            matched = (meta.get("activation_group_size") == width
                       and meta.get("activation_scale_format") == "fp32"
                       and tuple(meta.get("activation_scale_shape", ())) == (math.ceil(received_assignments),))
            requirement = "matching_int8_fp32_per_row"
        if (not direct or not matched
                or meta.get("activation_scale_bytes") != received_assignments * layouts[0].scale_bytes):
            raise ModelingError(f"mc2_{quant}_expert_input_requires_{requirement}_producer")
        # The fused dispatch owns the absorbed init-routing quantization,
        # so it publishes both the quantized payload and its scales.
        meta.update(activation_producer="explicit_transport_quantization",
                    activation_payload_producer=dispatch.op_id,
                    activation_scale_producer=dispatch.op_id,
                    moe_transport_kind=("arch35_fullmesh_mx32" if quant == "mx"
                                        else "arch35_fullmesh_per_token_int8"))
        by_id[gate.op_id] = replace(gate, metadata=tuple(meta.items()))
    down = by_id.get(prefix + ".expert_down")
    if down is not None and dict(down.metadata).get("output_format") != dtype:
        raise ModelingError("mc2_quantized_combine_requires_declared_float_expert_return")


def _routing_contracts(
    source_routing, source_rows_by_replica, destination_rows_by_rank,
    ep, width, dtype, tokens, valid_source_tokens, top_k, experts,
    quant, quant_mode, payload_dtype, layouts, send, received_assignments,
):
    # An exact source force-EPLB ledger is a whole routed-row matrix.  It is
    # never synthesized from a statistical destination expectation.
    peer_matrix = None
    if source_routing is not None:
        cells = getattr(source_routing, "peer_assignments", None)
        if cells is not None:
            peer_matrix = tuple(tuple(int(value) for value in row) for row in cells)
    route_rank = (int(getattr(source_routing, "representative_rank", 0))
                  if source_routing is not None else 0)
    if peer_matrix is not None:
        payload_basis = source_routing.basis
    elif source_routing is not None:
        payload_basis = str(getattr(source_routing, "basis",
                                    "declared_peer_expectation"))
    else:
        payload_basis = "uniform_peer_payload"
    source_key = ("source_force_eplb" if source_routing is not None and
                  source_routing.basis == "source_force_eplb_masked_integer_counts"
                  else "source_token_mask")
    # A complete TP1 DP cohort exposes one declared logical source row count
    # per EP rank.  Use that declared policy to build the SAME expected
    # assignment matrix for dispatch and its transpose for combine.  For a
    # wider TP group the vector is DP-scoped, not EP-scoped, and is ignored.
    global_source_rows = None
    if (peer_matrix is None and source_routing is None
            and source_rows_by_replica is not None
            and len(source_rows_by_replica) == ep):
        global_source_rows = tuple(
            max(0, int(value)) for value in source_rows_by_replica)

    def raw_contract(kind, incoming, outgoing):
        raw = {"kind": kind, "width": width, "dtype": dtype,
               "source_tokens": tokens if kind == "dispatch" else valid_source_tokens,
               "top_k": top_k, "source_assignments": incoming,
               "received_assignments": outgoing, "experts": experts, "ep": ep,
               "payload_basis": payload_basis}
        if kind == "dispatch" and quant != "none":
            raw.update(dispatch_quant=quant_mode, payload_dtype=payload_dtype)
        layout = layouts[0] if kind == "dispatch" else layouts[1]
        if peer_matrix is not None:
            # One representation: the shared source->destination forward
            # matrix. route_reverse=True below is the combine transpose.
            raw.update(route_matrix=peer_matrix, route_rank=route_rank)
        elif (source_routing is not None
              and hasattr(source_routing, "payload_matrix")):
            # Declared statistical destination policy: expected directed bytes,
            # never integer frame counts presented as observed peer traffic.
            expected = source_routing.payload_matrix(
                layout.packet_bytes, reverse=kind == "combine")
            raw["route_matrix_bytes"] = tuple(
                tuple(float(value) for value in row) for row in expected)
        elif global_source_rows is not None:
            # A source-blind two-way expectation conditional on the SAME
            # source and destination marginals used by F3:
            #   E[N_ij | s,d] = s_i * d_j / sum(s).
            # The matrix includes the local diagonal; native transport still
            # omits the local wire.  K3's all-one destination marginal keeps
            # the signed 31 off-diagonal assignments; DS's 12 occupied and 4
            # empty F3 ranks give .5 per occupied edge instead of .75*16.
            source_values = tuple(float(rows * top_k)
                                  for rows in global_source_rows)
            total_source = sum(source_values)
            if (destination_rows_by_rank is not None
                    and len(destination_rows_by_rank) == ep):
                destination_values = tuple(
                    float(max(0, int(value)))
                    for value in destination_rows_by_rank)
            else:
                destination_values = (
                    (total_source / float(ep),) * ep
                    if total_source > 0 else (0.0,) * ep)
            matrix = []
            for source in range(ep):
                row = []
                for destination in range(ep):
                    # Dispatch is [original source, expert destination].
                    # Combine is its transpose: [expert source, original
                    # destination], re-framed at the combine packet width.
                    s_value = (source_values[destination]
                               if kind == "combine" else source_values[source])
                    d_value = (destination_values[source]
                               if kind == "combine"
                               else destination_values[destination])
                    expected = (s_value * d_value / total_source
                                if total_source > 0 else 0.0)
                    row.append(expected * float(layout.packet_bytes))
                matrix.append(tuple(row))
            raw["route_matrix_bytes"] = tuple(matrix)
            raw["moe_expected_pairing"] = (
                "conditional_source_destination_marginals"
                if destination_rows_by_rank is not None
                else "conditional_uniform_destination")
        else:
            # Uniform peer policy: one source's useful payload spread across
            # its declared peers.  The expectation is positive unless the
            # source row itself is a known zero.
            raw["uniform_payload_bytes"] = float(
                incoming * layout.packet_bytes)
        raw["route_reverse"] = kind == "combine"
        if incoming <= 0 and source_routing is None:
            raw["payload_basis"] = "actual_zero"
        return raw

    forward_raw = raw_contract("dispatch", send, received_assignments)
    backward_raw = raw_contract("combine", received_assignments, send)
    return forward_raw, backward_raw, payload_basis, source_key


def _native_dispatch_combine(
    pack, dispatch, combine, unpack, prefix, dtype, width, ep, selected,
    quant, payload_dtype, layouts, received_assignments, absorbed,
    forward_raw, backward_raw, source_routing, source_key, source_metadata,
    row_format, source_partitions, payload_basis,
):
    phase_names = {
        "dispatch": tuple(p.name for p in contract(forward_raw, 2).phases),
        "combine": tuple(p.name for p in contract(backward_raw, 2).phases),
    }
    dispatch_returns = dispatch.returns
    received_output = {}
    if quant != "none":
        # The native kernel completes after receiver materialization and flag
        # retirement. These are receiver rows, even for an idle sender;
        # original.local_shape still describes the source payload.
        rows = math.ceil(received_assignments)
        received_output = {
            "output_format": payload_dtype,
            "activation_payload_shape": (rows, width),
            "activation_group_size": MX_DISPATCH_GROUP if quant == "mx" else width,
            "activation_scale_format": "fp8_e8m0" if quant == "mx" else "fp32",
            "activation_scale_shape": (rows, width // (2 * MX_DISPATCH_GROUP), 2) if quant == "mx" else (rows,),
            "activation_scale_bytes": received_assignments * layouts[0].scale_bytes,
            "activation_expected_rows": received_assignments,
        }
        if absorbed is None:
            dispatch_returns = tuple(dict.fromkeys((*dispatch_returns, "activation_scales")))

    def native_metadata(kind, raw, kernel, group):
        implementation = (
            {"mx": "arch35_fullmesh_mx32_bf16_e4m3",
             "per_token_int8": "arch35_fullmesh_per_token_int8"}.get(
                 quant, "arch35_fullmesh_unquantized_rows")
            if kind == "dispatch" else "arch35_fullmesh_unquantized_rows")
        metadata = {
            **(received_output if kind == "dispatch" else {}),
            **({source_key: source_metadata} if source_routing is not None else {}),
            "mc2_phase": raw,
            "mc2_phases": phase_names[kind],
            "group_axis": "ep", "steps": max(0, ep - 1),
            "device_kernel_group": {"id": group, "entry": True, "exit": True,
                                    "kernel": kernel},
            "implementation": implementation,
            "source": SOURCE, "native_kind": kernel,
            "invocation_class": "vector",
            **({"moe_comm_selection": dict(selected.get("moe_comm") or ())}
               if selected.get("selection") == "moe_comm_config" else {}),
            "peer_traffic_row_format": row_format,
            "source_assignment_partitions": source_partitions,
            "traffic_model": payload_basis,
            "transport_options": {
                "chunk_bytes": selected.get("chunk_bytes"),
                "hop_startup_us": selected.get("hop_startup_us", 0.0),
                "receive_capacity_bytes_per_peer": selected.get(
                    "receive_capacity_bytes_per_peer"),
                # The fused native phase scheduler owns its peer loop.  These
                # are declared for the direct-peer transport recipe and are
                # recorded here instead of being silently dropped.
                "application": "direct_peer_transport_recipe",
                "native_fused_phase_scheduler": (
                    "separate_owner; not_rescheduled_by_mc2_graph"),
            },
            "coverage": "source_protocol_with_declared_row_grid_and_peer_window",
        }
        if kind == "combine":
            metadata["transpose_of_source_assignments"] = True
        if kind == "dispatch" and absorbed is not None:
            # ``MoeDistributeDispatchV3`` owns the permutation *and* the
            # ``aclnnMoeInitRoutingV3`` dynamic quantization (``quant_mode=1``)
            # in one native call: no separate pack quantizer remains.
            metadata["moe_init_routing"] = dict(absorbed)
            metadata["fused_native_kinds"] = ("MoeDistributeDispatchV3",
                                              absorbed["native_kind"])
        return tuple(metadata.items())

    dispatch_op = declared_floor_work(forward_raw).apply(replace(
        dispatch, op_id=dispatch.op_id, deps=pack.deps,
        family="mc2_protocol", recipe=RECIPE, source_boundary="kernel",
        dtype=dtype, returns=dispatch_returns,
        metadata=native_metadata(
            "dispatch", forward_raw, "MoeDistributeDispatchV3",
            prefix + ".mc2_dispatch")))
    # The source reduction retains the actual weight producer dependency while
    # replacing generic payload reads, arithmetic and final output publication.
    final_deps = (dispatch.op_id,) + tuple(
        d for d in unpack.deps if d not in {pack.op_id, combine.op_id})
    combine_deps = tuple(dict.fromkeys(combine.deps + final_deps))
    combine_op = declared_floor_work(backward_raw).apply(replace(
        combine, op_id=unpack.op_id, deps=combine_deps,
        family="mc2_protocol", recipe=RECIPE, source_boundary="kernel",
        dtype=dtype,
        metadata=native_metadata(
            "combine", backward_raw, "MoeDistributeCombineV3",
            prefix + ".mc2_combine")))
    return dispatch_op, combine_op


def _replace_protocol_ops(
    ops, by_id, pack, dispatch, combine, unpack, pack_quant,
    dispatch_op, combine_op, received_quant,
):
    # One native call replaces its semantic phase operators; the original
    # pack/dispatch/combine/unpack aliases remain connected through the entry
    # and exit ids, never through duplicated scheduling.
    redirect = {pack.op_id: dispatch.op_id, combine.op_id: unpack.op_id}
    result = []
    for op in ops:
        if op.op_id == pack.op_id:
            result.append(dispatch_op)
            if received_quant is not None:
                result.append(received_quant)
            continue
        if op.op_id == unpack.op_id:
            result.append(combine_op)
            continue
        if (op.op_id in {dispatch.op_id, combine.op_id}
                or (pack_quant is not None and op.op_id == pack_quant.op_id)):
            continue
        current = by_id[op.op_id]
        deps = tuple(redirect.get(dependency, dependency)
                     for dependency in current.deps)
        if deps != current.deps:
            current = replace(current, deps=deps)
        result.append(current)
    return result
