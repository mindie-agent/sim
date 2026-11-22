"""CANN 9.3 Arch35 ``MegaMoE`` (``aclnnMegaMoe``) composite physical recipe.

[Role]
  Model ``cann_ops_transformer.mega_moe`` across ``enable_fused_mc2 in {0, 2, 3}``,
  ``MXFP8`` / ``MXFP4`` expert weights, and ``swiglu`` / ``swigluoai`` /
  ``situglu`` (``PR #22333``) gated activations, fusing EP fullmesh MX32
  Dispatch, ``GMM1``, in-UB activation + group-32 E8M0 quantization, ``GMM2``,
  optional ``enable_fused_mc2=3`` Shared Expert, and BF16 Combine into one
  kernel boundary; only the quantized GMM2 input workspace is stored (MTE3)
  and reloaded (AIC MTE2), split across L2/HBM by residency.

[Invariants]
  * ``OperatorPlan.work`` equals the exact sum of issued per-owner ``Stage``
    demands across ``hbm``, ``l2``, ``mte2_aic``, ``mte2_aiv``, ``mte1``,
    ``cube``, ``vector``, ``fixpipe``, ``mte3``, and ``link``.
  * Buffer footprints (``ub``, ``l1``, ``l0a``, ``l0b``, ``l0c``) are validated
    dynamically against ``geometry.buffers``.
  * Zero model-name branches or ``ResourceEnvelope`` fallbacks participate.

[Timeouts & Locks]
  O(compute_units) plan construction; bounded finite stages, no I/O or locks.

[Side-Effect Policy]
  Pure immutable ``OpSpec`` and ``OperatorPlan`` construction.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from ...ownership import read_role, unit_group
from ...program import KernelProgram, ParallelProgram, Stage
from ..grouped_moe.lowering import element_bytes, mx_scale_bytes

RECIPE = "megamoe_arch35"
NATIVE_KIND = "aclnnMegaMoe"

#: Stock shipping CANN ``dispatch_ffn_combine`` / ``aclnnMegaMoe`` activations.
#: Custom gated variants (``situ``, ``situglu``, ``swigluoai``, ``geglu``,
#: ``gelu_tanh``) require an explicit ABI variant extension rather than silent
#: admission on shipping ``merged`` / ``unmerged_pr`` baselines.
SHIPPING_SUPPORTED_ACTIVATIONS = frozenset({"swiglu", "silu", "gelu"})
EXTENDED_ABI_ACTIVATIONS = frozenset({
    "swigluoai", "situglu", "situ", "gelu_tanh", "geglu",
})
SUPPORTED_ACTIVATIONS = SHIPPING_SUPPORTED_ACTIVATIONS
ALL_ACTIVATIONS = SHIPPING_SUPPORTED_ACTIVATIONS | EXTENDED_ABI_ACTIVATIONS
FUSED_MC2_ROUTED_MODE = 2
FUSED_MC2_SHARED_MODE = 3
SUPPORTED_FUSED_MC2_MODES = frozenset({0, FUSED_MC2_ROUTED_MODE, FUSED_MC2_SHARED_MODE})

CANN_MEGAMOE_SOURCE = {
    "repository": "cann_ops_transformer",
    "revision": "1.0.0",
    "relative_path": "cann_ops_transformer/ops/mc2/mega_moe/mega_moe.py",
    "source_sha256": "a273a05f54da16dc657a507d1cc8022cc7b2b99cbbfbca235a7f3ca916bee34d",
}
VLLM_MEGAMOE_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/ops/fused_moe/moe_mlp.py",
    "source_sha256": "7882f9d0847e0303c78f2b216cfc50da9d3d062d6bd6f0cd22c94d1c7e78af0e",
}
PR22333_SITUGLU_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "a45c664d76",
    "relative_path": "vllm_ascend/ops/fused_moe/moe_mlp.py",
    "source_sha256": "7882f9d0847e0303c78f2b216cfc50da9d3d062d6bd6f0cd22c94d1c7e78af0e",
}

BF16_BYTES = 2
FP32_BYTES = 4
INT32_BYTES = 4
MX_GROUP_SIZE = 32
MX_QUANT_OPS_PER_ELEM = 4
SWIGLU_OPS_PER_ELEM = 5
OAI_SWIGLU_OPS_PER_ELEM = 8
SITUGLU_OPS_PER_ELEM = 7
COMBINE_REDUCE_OPS_PER_ELEM = 2
L2_EFFECTIVE_SHARE = 0.75

MEGAMOE_BUFFERS = {
    "ub": 128 * 1024,
    "l1": 256 * 1024,
    "l0a": 32 * 1024,
    "l0b": 32 * 1024,
    "l0c": 64 * 1024,
}


def _activation_ops(activation: str) -> int:
    act = str(activation or "swiglu").lower()
    if act == "swigluoai":
        return OAI_SWIGLU_OPS_PER_ELEM + MX_QUANT_OPS_PER_ELEM
    if act in {"situglu", "situ"}:
        return SITUGLU_OPS_PER_ELEM + MX_QUANT_OPS_PER_ELEM
    return SWIGLU_OPS_PER_ELEM + MX_QUANT_OPS_PER_ELEM


def _effective_l2(geometry: Geometry) -> float:
    view = getattr(geometry, "admission_memory_view", None)
    if view is not None:
        die = int(getattr(geometry, "rank", 0) or 0)
        cap_map = getattr(view, "_l2_capacity", None)
        if isinstance(cap_map, Mapping) and cap_map:
            cap = cap_map.get(die, next(iter(cap_map.values())))
            return float(cap)
    mechanisms = getattr(geometry, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    if hierarchy is not None:
        return float(hierarchy.effective_l2_bytes)
    return L2_EFFECTIVE_SHARE * float((geometry.buffers or {}).get("l2", 0))


_UNQUANTIZED_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})


def _mx_row_bytes(elements: int, payload_dtype: str) -> float:
    """Packed operand row: payload elements plus E8M0 MX scales when quantized."""
    if elements <= 0:
        return 0.0
    dt = str(payload_dtype or "fp8_e4m3").lower()
    payload = element_bytes(elements, dt)
    scales = 0 if dt in _UNQUANTIZED_DTYPES else mx_scale_bytes(elements)
    return float(payload + scales)


def _merge_matrix_flops(
    routed_dt: str, routed_flops: float, shared_dt: str, shared_flops: float,
) -> dict[str, float]:
    """Combine routed and optional shared Cube FLOPs under their own operand dtypes."""
    matrix: dict[str, float] = {}
    if routed_flops > 0.0:
        matrix[routed_dt] = routed_flops
    if shared_flops > 0.0:
        matrix[shared_dt] = matrix.get(shared_dt, 0.0) + shared_flops
    return matrix


def _check_buffers(geometry: Geometry) -> dict[str, int]:
    available = dict(geometry.buffers or {})
    for pool, required in MEGAMOE_BUFFERS.items():
        cap = int(available.get(pool, 0))
        if cap > 0 and required > cap:
            raise ModelingError(f"megamoe_arch35_buffer_capacity_exceeded:{pool}")
    return {p: sz for p, sz in MEGAMOE_BUFFERS.items() if available.get(p, 0) > 0}


def make_op(
    *,
    op_id: str,
    deps: tuple[str, ...] = (),
    stage: int = 0,
    layer: int = 0,
    requests: tuple[str, ...] = (),
    phase: str = "decode",
    tokens: int,
    hidden: int,
    moe_inter: int,
    top_k: int,
    routed_rows: float,
    w13_bytes: float,
    w2_bytes: float,
    enable_fused_mc2: int = 2,
    ep: int = 1,
    activation: str = "swiglu",
    compute_dtype: str = "fp8_e4m3",
    output_dtype: str = "bf16",
    dispatch_link_bytes: float = 0.0,
    combine_link_bytes: float = 0.0,
    group_axis: str = "ep",
    collective: str = "alltoall",
    dispatch_peer_traffic: Any = None,
    combine_peer_traffic: Any = None,
    source_stage: int | None = None,
    peer_volume_basis: str | None = None,
    shared_inter: int = 0,
    shared_w13_bytes: float = 0.0,
    shared_w2_bytes: float = 0.0,
    shared_compute_dtype: str | None = None,
    allow_abi_extension: bool = False,
    activation_params: Mapping[str, Any] | None = None,
    extra_metadata: tuple[tuple[str, Any], ...] = (),
) -> OpSpec:
    """Construct a ``megamoe_arch35`` composite ``OpSpec``."""
    act = str(activation or "swiglu").strip().lower()
    allowed_acts = ALL_ACTIVATIONS if allow_abi_extension else SUPPORTED_ACTIVATIONS
    if act not in allowed_acts:
        raise ModelingError(f"megamoe_arch35_unsupported_activation:{act}")
    is_abi_ext = act not in SUPPORTED_ACTIVATIONS
    mc2_mode = int(enable_fused_mc2)
    if mc2_mode not in SUPPORTED_FUSED_MC2_MODES:
        raise ModelingError(f"megamoe_arch35_unsupported_fused_mc2:{mc2_mode}")
    tok = max(0, int(tokens))
    width = max(1, int(hidden))
    inter = max(1, int(moe_inter))
    k_sel = max(1, int(top_k))
    r_rows = max(0.0, float(routed_rows))
    shape = (tok, width)
    routed_flops = 2.0 * r_rows * width * (2 * inter) + 2.0 * r_rows * inter * width
    sh_inter = int(shared_inter) if mc2_mode == FUSED_MC2_SHARED_MODE else 0
    sh_comp_dt = str(shared_compute_dtype or compute_dtype) if sh_inter > 0 else compute_dtype
    shared_flops = (
        2.0 * tok * width * (2 * sh_inter) + 2.0 * tok * sh_inter * width
        if sh_inter > 0 else 0.0
    )
    total_flops = routed_flops + shared_flops
    matrix_flops_map = _merge_matrix_flops(compute_dtype, routed_flops, sh_comp_dt, shared_flops)
    out_elem = float(dtype_bytes(output_dtype))
    in_aiv_bytes = float(tok * (width * out_elem + k_sel * (FP32_BYTES + INT32_BYTES)))
    out_aiv_bytes = float(tok * width * out_elem)
    sh_w13_b = float(shared_w13_bytes) if sh_inter > 0 else 0.0
    sh_w2_b = float(shared_w2_bytes) if sh_inter > 0 else 0.0
    mte2_b = in_aiv_bytes + float(w13_bytes) + float(w2_bytes) + sh_w13_b + sh_w2_b
    mte3_b = out_aiv_bytes
    hbm_b = mte2_b + mte3_b
    link_b = (
        float(dispatch_link_bytes) + float(combine_link_bytes)
        if (mc2_mode >= FUSED_MC2_ROUTED_MODE and int(ep) > 1) else 0.0
    )
    vec_ops = (
        float(tok * width * MX_QUANT_OPS_PER_ELEM)
        + (r_rows * inter + (tok * sh_inter if sh_inter > 0 else 0.0)) * _activation_ops(act)
        + float(tok * width * (k_sel * COMBINE_REDUCE_OPS_PER_ELEM + (1 if sh_inter > 0 else 0)))
    )
    ep_axis = str(group_axis or "ep")
    ep_collective = str(collective or "alltoall")
    has_ep_comm = mc2_mode >= FUSED_MC2_ROUTED_MODE and int(ep) > 1
    megamoe_geo: dict[str, Any] = {
        "tokens": tok,
        "hidden": width,
        "moe_intermediate": inter,
        "top_k": k_sel,
        "routed_rows": r_rows,
        "w13_bytes": float(w13_bytes),
        "w2_bytes": float(w2_bytes),
        "enable_fused_mc2": mc2_mode,
        "ep": max(1, int(ep)),
        "activation": act,
        "abi_variant_extension": bool(is_abi_ext),
        "compute_dtype": compute_dtype,
        "output_dtype": output_dtype,
        "dispatch_link_bytes": float(dispatch_link_bytes),
        "combine_link_bytes": float(combine_link_bytes),
        "shared_intermediate": sh_inter,
        "shared_w13_bytes": float(shared_w13_bytes) if mc2_mode == FUSED_MC2_SHARED_MODE else 0.0,
        "shared_w2_bytes": float(shared_w2_bytes) if mc2_mode == FUSED_MC2_SHARED_MODE else 0.0,
        **({"shared_compute_dtype": sh_comp_dt} if sh_inter > 0 and sh_comp_dt != compute_dtype else {}),
    }
    if has_ep_comm:
        megamoe_geo["group_axis"] = ep_axis
        megamoe_geo["collective"] = ep_collective
        if dispatch_peer_traffic is not None:
            megamoe_geo["dispatch_peer_traffic"] = dispatch_peer_traffic
        if combine_peer_traffic is not None:
            megamoe_geo["combine_peer_traffic"] = combine_peer_traffic
    spec_meta = {
        "native_kind": NATIVE_KIND,
        "input_format": output_dtype,
        "output_format": output_dtype,
        "compute_format": compute_dtype,
        "activation_compute_dtype": "fp32",
        "ub_working_bytes_per_elem": FP32_BYTES,
        "abi_variant_extension": bool(is_abi_ext),
        **({"activation_params": dict(activation_params)} if activation_params else {}),
        **(
            {
                "group_axis": ep_axis,
                "comm_axis": ep_axis,
                "comm_group_size": max(1, int(ep)),
                "collective": ep_collective,
                **({"source_stage": int(source_stage)} if source_stage is not None else {}),
                **({"peer_volume_basis": str(peer_volume_basis)} if peer_volume_basis is not None else {}),
                **({"dispatch_peer_traffic": dispatch_peer_traffic} if dispatch_peer_traffic is not None else {}),
                **({"combine_peer_traffic": combine_peer_traffic} if combine_peer_traffic is not None else {}),
            }
            if has_ep_comm else {}
        ),
        "megamoe_geometry": megamoe_geo,
        "native_source": CANN_MEGAMOE_SOURCE,
        "caller_source": (
            PR22333_SITUGLU_SOURCE if act in {"situglu", "situ"} else VLLM_MEGAMOE_SOURCE
        ),
        **dict(extra_metadata),
    }
    return OpSpec(
        op_id=op_id, deps=deps, family="grouped_gemm", recipe=RECIPE,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype=output_dtype, flops=total_flops,
        matrix_flops=tuple(matrix_flops_map.items()),
        vector_ops=vec_ops, hbm_bytes=hbm_b, mte2_bytes=mte2_b, mte3_bytes=mte3_b,
        link_bytes=link_b,
        source_boundary="kernel",
        metadata=tuple(sorted(spec_meta.items())),
    )


def _gmm_stage(
    context, name, after, placement, share, matrix_flops,
    aic_pipe, gm_read_bytes, total_read_bytes, operand_bytes, fixpipe_bytes, l2_hit,
):
    resources = {
        "hbm": gm_read_bytes * share,
        aic_pipe: total_read_bytes * share,
        "mte1": operand_bytes * share,
        "fixpipe": fixpipe_bytes * share,
    }
    if l2_hit > 0.0:
        resources["l2"] = l2_hit * share
    return Stage(
        name,
        context.bind(Work(
            matrix={dt: flops * share for dt, flops in matrix_flops.items() if flops > 0.0},
            resources=resources,
        )),
        after=(after,), placement=placement,
    )


def _resolve_phase_links(
    context, disp_link: float, comb_link: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve routed Topology (or die-local) dispatch/combine wire demands."""
    phase_links = getattr(context, "phase_link_work", None)
    if isinstance(phase_links, Mapping) and (
        phase_links.get("dispatch") or phase_links.get("combine")
    ):
        disp = {str(k): float(v) for k, v in dict(phase_links.get("dispatch") or {}).items() if float(v) > 0.0}
        comb = {str(k): float(v) for k, v in dict(phase_links.get("combine") or {}).items() if float(v) > 0.0}
        return disp, comb
    total_link = disp_link + comb_link
    topo_links = {
        str(k): float(v)
        for k, v in dict(getattr(context, "work", {}) or {}).items()
        if float(v) > 0.0 and "link" in str(k).split(":")
    }
    if topo_links:
        disp_frac = (disp_link / total_link) if total_link > 0.0 else 0.5
        disp = {k: v * disp_frac for k, v in topo_links.items() if v * disp_frac > 0.0}
        comb = {k: v - disp.get(k, 0.0) for k, v in topo_links.items() if v - disp.get(k, 0.0) > 0.0}
        return disp, comb
    if total_link > 0.0 and "link" in context.resource_names:
        disp = context.bind(Work(resources={"link": disp_link})) if disp_link > 0.0 else {}
        comb = context.bind(Work(resources={"link": comb_link})) if comb_link > 0.0 else {}
        return disp, comb
    return {}, {}


def _core_portion(total: float, core: int, cores: int) -> float:
    """Exact per-core slice closing floating-point rounding on the final core."""
    if cores <= 1:
        return total
    unit = total / float(cores)
    if core + 1 == cores:
        return max(0.0, total - unit * float(cores - 1))
    return unit


def _build_megamoe_program(
    context, *, cores: int, aiv_units: int, buf_sizes: dict[str, int],
    in_aiv_bytes: float, out_aiv_bytes: float, pack_vec_ops: float,
    act_vec_ops: float, comb_vec_ops: float, disp_link: float, comb_link: float,
    gmm1_matrix: Mapping[str, float], gmm2_matrix: Mapping[str, float],
    gmm1_gm_read: float, gmm2_gm_read: float,
    gmm1_read_total: float, gmm2_read_total: float,
    l0_gmm1: float, l0_gmm2: float, fix_gmm1: float,
    fix_gmm2: float, gmm1_l2_hit: float, act2_hot: float = 0.0, act2_cold: float = 0.0,
):
    c_aic_pipe = read_role(context.resource_names, aiv=None)
    c_aiv_pipe = read_role(context.resource_names, aiv=0)
    share = 1.0 / float(cores)
    act2 = act2_hot + act2_cold
    disp_wire, comb_wire = _resolve_phase_links(context, disp_link, comb_link)

    def per_core(core: int):
        aiv_idx = min(aiv_units - 1, 2 * core if aiv_units >= 2 * cores else core)
        aic_owned = context.owned(core, cores, ("cube", "mte1", "fixpipe", c_aic_pipe))
        aiv_owned = context.owned(aiv_idx, aiv_units, ("vector", c_aiv_pipe, "mte3"))
        allocs = tuple(
            context.allocate(
                f"megamoe_{core}_{pool}", pool, size,
                unit_group(core, cores, context.pool_units.get(pool, cores)),
            )
            for pool, size in buf_sizes.items() if pool in context.pool_names
        )
        disp_res: dict[str, float] = {
            "hbm": in_aiv_bytes * share,
            c_aiv_pipe: in_aiv_bytes * share,
            "vector": pack_vec_ops * share,
        }
        disp_work = context.bind(Work(resources=disp_res))
        for res_name, wire_val in disp_wire.items():
            portion = _core_portion(wire_val, core, cores)
            if portion > 0.0:
                disp_work[res_name] = disp_work.get(res_name, 0.0) + portion
        s_disp = Stage("dispatch_mx", disp_work, acquire=allocs, placement=aiv_owned)
        s_gmm1 = _gmm_stage(
            context, "gmm1_gate_up", "dispatch_mx", aic_owned, share, gmm1_matrix,
            c_aic_pipe, gmm1_gm_read, gmm1_read_total, l0_gmm1, fix_gmm1, gmm1_l2_hit,
        )
        act_res = {"vector": act_vec_ops * share, "mte3": act2 * share}
        for tier, amount in (("hbm", act2_cold), ("l2", act2_hot)):
            if amount > 0.0:
                act_res[tier] = amount * share
        s_act = Stage(
            "act_mx_quant", context.bind(Work(resources=act_res)),
            after=("gmm1_gate_up",), placement=aiv_owned,
        )
        s_gmm2 = _gmm_stage(
            context, "gmm2_down", "act_mx_quant", aic_owned, share, gmm2_matrix,
            c_aic_pipe, gmm2_gm_read + act2_cold, gmm2_read_total + act2, l0_gmm2, fix_gmm2, act2_hot,
        )
        comb_res: dict[str, float] = {
            "vector": comb_vec_ops * share,
            "hbm": out_aiv_bytes * share,
            "mte3": out_aiv_bytes * share,
        }
        comb_work = context.bind(Work(resources=comb_res))
        for res_name, wire_val in comb_wire.items():
            portion = _core_portion(wire_val, core, cores)
            if portion > 0.0:
                comb_work[res_name] = comb_work.get(res_name, 0.0) + portion
        s_comb = Stage(
            "combine_store",
            comb_work,
            after=("gmm2_down",), free=allocs, placement=aiv_owned,
        )
        return KernelProgram((s_disp, s_gmm1, s_act, s_gmm2, s_comb))

    return ParallelProgram(tuple(per_core(c) for c in range(cores)))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build a work-conserving per-core AIC+AIV(+Link) program for ``MegaMoE``."""
    buf_sizes = _check_buffers(geometry)
    meta = dict(op.metadata)
    spec = dict(meta.get("megamoe_geometry") or {})
    tok = int(spec.get("tokens", op.local_shape[0] if op.local_shape else 1))
    width = int(spec.get("hidden", op.local_shape[-1] if op.local_shape else 1))
    inter = int(spec.get("moe_intermediate", width))
    k_sel = int(spec.get("top_k", 1))
    r_rows = float(spec.get("routed_rows", tok * k_sel))
    mc2_mode = int(spec.get("enable_fused_mc2", 2))
    ep = max(1, int(spec.get("ep", 1)))
    act = str(spec.get("activation", "swiglu")).lower()
    comp_dt = str(spec.get("compute_dtype", meta.get("compute_format") or "fp8_e4m3"))
    out_dt = str(spec.get("output_dtype", op.dtype or "bf16"))
    out_elem = float(dtype_bytes(out_dt))

    allow_ext = bool(spec.get("abi_variant_extension") or meta.get("abi_variant_extension"))
    allowed_acts = ALL_ACTIVATIONS if allow_ext else SUPPORTED_ACTIVATIONS
    if act not in allowed_acts:
        raise ModelingError(f"megamoe_arch35_unsupported_activation:{act}")
    if mc2_mode not in SUPPORTED_FUSED_MC2_MODES:
        raise ModelingError(f"megamoe_arch35_unsupported_fused_mc2:{mc2_mode}")

    w13 = float(spec.get("w13_bytes", 2.0 * inter * width))
    w2 = float(spec.get("w2_bytes", width * inter))
    sh_inter = int(spec.get("shared_intermediate", 0)) if mc2_mode == FUSED_MC2_SHARED_MODE else 0
    sh_comp_dt = str(spec.get("shared_compute_dtype") or comp_dt) if sh_inter > 0 else comp_dt
    sh_w13 = float(spec.get("shared_w13_bytes", 0.0)) if sh_inter > 0 else 0.0
    sh_w2 = float(spec.get("shared_w2_bytes", 0.0)) if sh_inter > 0 else 0.0
    disp_link = (
        float(spec.get("dispatch_link_bytes", 0.0))
        if (mc2_mode >= FUSED_MC2_ROUTED_MODE and ep > 1) else 0.0
    )
    comb_link = (
        float(spec.get("combine_link_bytes", 0.0))
        if (mc2_mode >= FUSED_MC2_ROUTED_MODE and ep > 1) else 0.0
    )

    # Input hidden + topk metadata read once on AIV; final output written once on AIV.
    in_aiv_bytes = float(tok * (width * out_elem + k_sel * (FP32_BYTES + INT32_BYTES)))
    out_aiv_bytes = float(tok * width * out_elem)

    # Routed + optional shared weight/operand reads on AIC; MX rows follow each projection's dtype.
    mx_row_bytes = _mx_row_bytes(width, comp_dt)
    inter_mx_bytes = _mx_row_bytes(inter, comp_dt)
    sh_row_bytes = _mx_row_bytes(width, sh_comp_dt) if sh_inter > 0 else 0.0
    aic_act_read = r_rows * mx_row_bytes + (tok * sh_row_bytes if sh_inter > 0 else 0.0)
    aic_weight_read = w13 + w2 + sh_w13 + sh_w2
    aic_read_total = aic_act_read + aic_weight_read

    has_l2 = bool((geometry.rates or {}).get("l2", 0.0) or (geometry.resource_units or {}).get("l2", 0))
    l2_cap = _effective_l2(geometry) if has_l2 else 0.0
    # Dispatched MX activations land in L2/UB on-chip and are consumed by GMM1 only.
    gmm1_read_total = aic_act_read + w13 + sh_w13
    gmm2_read_total = w2 + sh_w2
    l2_hit = min(aic_act_read, l2_cap) if l2_cap > 0.0 else 0.0
    gmm1_gm_read = max(0.0, gmm1_read_total - l2_hit)
    gmm2_gm_read = gmm2_read_total
    aic_gm_read = gmm1_gm_read + gmm2_gm_read

    r_gmm1_flops = 2.0 * r_rows * width * (2 * inter)
    s_gmm1_flops = 2.0 * tok * width * (2 * sh_inter) if sh_inter > 0 else 0.0
    r_gmm2_flops = 2.0 * r_rows * inter * width
    s_gmm2_flops = 2.0 * tok * sh_inter * width if sh_inter > 0 else 0.0
    gmm1_matrix = _merge_matrix_flops(comp_dt, r_gmm1_flops, sh_comp_dt, s_gmm1_flops)
    gmm2_matrix = _merge_matrix_flops(comp_dt, r_gmm2_flops, sh_comp_dt, s_gmm2_flops)
    l0_gmm1 = aic_act_read + w13 + sh_w13
    l0_gmm2 = r_rows * inter_mx_bytes + w2 + sh_w2 + (
        tok * _mx_row_bytes(sh_inter, sh_comp_dt) if sh_inter > 0 else 0.0
    )
    # GMM1 epilogue writes the quantized activation (data + E8M0 scales) to the GMM2 input
    # workspace (MTE3); GMM2 reads it back on the AIC GM path.  L2 residency only splits tiers.
    act2 = max(0.0, l0_gmm2 - w2 - sh_w2)
    act2_hot = min(act2, l2_cap) if l2_cap > 0.0 else 0.0
    act2_cold = act2 - act2_hot
    fix_gmm1 = 2.0 * r_rows * inter * BF16_BYTES + (
        2.0 * tok * sh_inter * BF16_BYTES if sh_inter > 0 else 0.0
    )
    fix_gmm2 = r_rows * width * BF16_BYTES + (
        tok * width * BF16_BYTES if sh_inter > 0 else 0.0
    )
    act_vec_ops = (r_rows * inter + (tok * sh_inter if sh_inter > 0 else 0.0)) * _activation_ops(act)
    pack_vec_ops = float(tok * width * MX_QUANT_OPS_PER_ELEM)
    comb_vec_ops = float(tok * width * (k_sel * COMBINE_REDUCE_OPS_PER_ELEM + (1 if sh_inter > 0 else 0)))

    cores = max(1, int(geometry.compute_units))
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 2 * cores)))
    aic_pipe = read_role(geometry.resource_units or {}, aiv=None)
    aiv_pipe = read_role(geometry.resource_units or {}, aiv=0)

    plan_res: dict[str, float] = {
        "hbm": in_aiv_bytes + aic_gm_read + out_aiv_bytes + 2.0 * act2_cold,
        "mte1": l0_gmm1 + l0_gmm2,
        "fixpipe": fix_gmm1 + fix_gmm2,
        "vector": pack_vec_ops + act_vec_ops + comb_vec_ops,
        "mte3": out_aiv_bytes + act2,
    }
    if aic_pipe == aiv_pipe:
        plan_res[aic_pipe] = in_aiv_bytes + aic_read_total + act2
    else:
        plan_res[aiv_pipe] = in_aiv_bytes
        plan_res[aic_pipe] = aic_read_total + act2
    if l2_hit + act2_hot > 0.0:
        plan_res["l2"] = l2_hit + 2.0 * act2_hot
    if disp_link + comb_link > 0.0:
        plan_res["link"] = disp_link + comb_link
    matrix_map = _merge_matrix_flops(
        comp_dt, r_gmm1_flops + r_gmm2_flops, sh_comp_dt, s_gmm1_flops + s_gmm2_flops,
    )
    total_work = Work(matrix=matrix_map, resources={k: v for k, v in plan_res.items() if v > 0.0})

    def program(context):
        return _build_megamoe_program(
            context, cores=cores, aiv_units=aiv_units, buf_sizes=buf_sizes,
            in_aiv_bytes=in_aiv_bytes, out_aiv_bytes=out_aiv_bytes,
            pack_vec_ops=pack_vec_ops, act_vec_ops=act_vec_ops, comb_vec_ops=comb_vec_ops,
            disp_link=disp_link, comb_link=comb_link,
            gmm1_matrix=gmm1_matrix, gmm2_matrix=gmm2_matrix,
            gmm1_gm_read=gmm1_gm_read, gmm2_gm_read=gmm2_gm_read,
            gmm1_read_total=gmm1_read_total, gmm2_read_total=gmm2_read_total,
            l0_gmm1=l0_gmm1, l0_gmm2=l0_gmm2,
            fix_gmm1=fix_gmm1, fix_gmm2=fix_gmm2, gmm1_l2_hit=l2_hit,
            act2_hot=act2_hot, act2_cold=act2_cold,
        )

    return OperatorPlan(total_work, program, {
        "execution_class": "mixed_aic_aiv_mc2",
        "enable_fused_mc2": mc2_mode,
        "activation": act,
        "activation_compute_dtype": "fp32",
        "ub_working_bytes_per_elem": FP32_BYTES,
        "abi_variant_extension": bool(act not in SUPPORTED_ACTIVATIONS),
        "shared_expert_fused": bool(sh_inter > 0),
        "dispatch_link_bytes": disp_link,
        "combine_link_bytes": comb_link,
        "l2_onchip_activation_bytes": l2_hit,
        "gmm2_activation_workspace_bytes": act2,
        "source_program": CANN_MEGAMOE_SOURCE,
    })


def recipes() -> dict[str, PlannedRecipe]:
    return {RECIPE: PlannedRecipe(build)}
