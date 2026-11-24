"""Causal acceptance-tail placement, MoE ABI legality, and barrier rewiring.

[Role] Own pure structural DAG placement and Probed-ABI MoE route legality
  helpers for target-state optimization and export:
  - Validate MoE ``(enable_fused_mc2, MSO)`` legality strictly via
    ``mindie_sim.backends.abi_registry.ProbedAbiRegistry.verify()`` (zero fake
    shipping defaults, zero ``"Arch20"``/``"swiglu"`` fallbacks);
  - Compute the exact causal ``target_tail`` closure on the full occurrence DAG
    before barrier filtering so real post-accept ``recover_ssm_commit`` Engine
    occurrences always execute strictly after ``accept`` and before Draft;
  - Transitively resolve chains of removed zero-duration barriers so external
    data dependencies never break.
[Invariants] Every acceptance-dependent target op belongs to ``target_tail``.
  Zero shadow schedulers, zero unvalidated activation whitelists, and zero
  silent architecture or activation fallbacks.
[Timeouts & Locks] Linear O(|V| + |E|) DAG walks and cached sealed ABI lookup;
  no I/O, locks or devices.
[Side-Effect Policy] Pure functional transformations over immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.rewrite.algebra import (
    _norm_abi_dtype,
    _port,
    _quant_contract_for_port,
)
from mindie_sim.schema.abi import (
    RANK_PLACEMENTS,
    AbiQueryKey,
    AbiSupportVerdict,
    OperandPortSig,
    RankPlacement,
    SocArch,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.stack import ManifestBinding, StackTrack, StackTrackSpec

BARRIER_EPS_US = 1e-3
#: Hardware ISA / arch-family identifiers.
ARCH35_TOKENS = frozenset({"A5", "ARCH35", "DAV_3510", "NPU_ARCH_3510"})
ARCH22_TOKENS = frozenset({"A3", "ARCH22"})
ARCH20_TOKENS = frozenset({"A2", "ARCH20"})
_ARCH_TAG_TO_SOC: Mapping[str, SocArch] = {
    "Arch35": "A5",
    "Arch22": "A3",
    "Arch20": "A2",
}
VALID_FUSED_MC2_LEVELS = frozenset({0, 1, 2, 3})
FUSED_MC2_CSRC_LEVEL = 1
FUSED_MC2_ROUTED_LEVEL = 2
FUSED_MC2_SHARED_LEVEL = 3
_MEGAMOE_CONTEXT_SHAPE = (64,)
_A3_SINGLE_NODE_MAX_EP = 16
#: Families that represent acceptance anchors and post-accept commits.
ACCEPTANCE_ANCHOR_FAMILIES = frozenset({"acceptance", "sampling"})
TAIL_COMMIT_FAMILIES = frozenset({"token_commit", "recover_ssm_commit"})
#: Persistent whole-die fused MoE kernel markers.
PERSISTENT_MOE_FAMILIES = frozenset({"grouped_moe", "mc2", "fused_moe_mc2"})
_MOE_ACTIVATION_ALIASES = {
    "silu": "swiglu",
    "swiglu": "swiglu",
    "situ": "situglu",
    "situglu": "situglu",
    "geglu": "gelu",
    "gelu": "gelu",
    "relu_squared": "relu2",
    "relu2": "relu2",
    "swigluoai": "swigluoai",
    "swiglustep": "swiglustep",
    "clipped_swiglu": "clipped_swiglu",
}
_NON_ACT_AND_MUL_ACTIVATIONS = frozenset({"relu2"})


@dataclass(frozen=True, slots=True)
class MoERouteDecision:
    """Resolved legal MoE route validated against sealed ``ProbedAbiRegistry``."""

    enable_fused_mc2: int
    multistream_overlap_shared_expert: bool
    route_id: str
    abi_id: str
    needs_new_kernel: bool
    is_shipping_abi: bool
    reason: str
    legal: bool = True
    makespan_us: float = 0.0
    stream_core_budget_summary: str = ""
    abi_verdict: AbiSupportVerdict | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "legal": bool(self.legal),
            "enable_fused_mc2": int(self.enable_fused_mc2),
            "multistream_overlap_shared_expert": bool(self.multistream_overlap_shared_expert),
            "route_id": self.route_id,
            "abi_id": self.abi_id,
            "needs_new_kernel": bool(self.needs_new_kernel),
            "is_shipping_abi": bool(self.is_shipping_abi),
            "reason": self.reason,
            "makespan_us": round(float(self.makespan_us), 4),
            "stream_core_budget": self.stream_core_budget_summary,
        }
        if self.abi_verdict is not None:
            payload["abi_status"] = self.abi_verdict.status
            payload["matched_entry_id"] = self.abi_verdict.matched_entry_id
        return payload


def resolve_arch_family_tag(case: Case) -> str:
    """Return canonical architecture family tag (``Arch35``, ``Arch22``, ``Arch20``)."""
    die = case.machine.dies[0] if case.machine.dies else None
    raw_arch = str(getattr(case.machine, "arch_family", "") or "").strip().upper()
    if raw_arch in ARCH35_TOKENS:
        return "Arch35"
    if raw_arch in ARCH22_TOKENS:
        return "Arch22"
    if raw_arch in ARCH20_TOKENS:
        return "Arch20"
    if die is not None:
        comp_arch = getattr(die, "compute_arch", None)
        isa = str(getattr(comp_arch, "isa_family", "") or "").strip().upper()
        if isa in ARCH35_TOKENS:
            return "Arch35"
        dpath = datapath_for_die(die)
        if bool(getattr(dpath, "has_l0c_to_ub", False)):
            return "Arch35"
        if int(getattr(case.machine, "dies_per_package", 1) or 1) >= 2:
            return "Arch22"
    if not raw_arch:
        raise ModelingError("arch_family_unspecified")
    return raw_arch


def is_a5_hardware(case: Case) -> bool:
    """True strictly when ``case`` targets Ascend A5 (``Arch35`` / ``DAV_3510``)."""
    return resolve_arch_family_tag(case) == "Arch35"


def is_latent_moe_model(case: Case) -> bool:
    """True when the model uses latent MoE or declares non-null ``routed_expert_hidden_size``."""
    model_meta = dict(getattr(case.model, "metadata", ()) or ())
    if model_meta.get("routed_expert_hidden_size") is not None:
        return True
    raw_cfg = model_meta.get("hf_text_config") or model_meta.get("model_config")
    if isinstance(raw_cfg, Mapping) and raw_cfg.get("routed_expert_hidden_size") is not None:
        return True
    for lp in getattr(case.model, "layer_plans", ()) or ():
        if int(getattr(lp, "expert_latent_size", 0) or 0) > 0:
            return True
        if getattr(lp, "routed_expert_hidden_size", None) is not None:
            return True
        lp_meta = dict(getattr(lp, "metadata", ()) or ())
        if lp_meta.get("routed_expert_hidden_size") is not None:
            return True
    return False


def is_persistent_whole_die_node(node: Any) -> bool:
    """True when ``node`` is a persistent whole-die fused MoE kernel locking AIC+AIV."""
    if bool(getattr(node, "persistent_whole_die", False)):
        return True
    fam = str(getattr(node, "family", "") or "").strip().lower()
    rec = str(getattr(node, "recipe", "") or "").strip().lower()
    kern = str(getattr(node, "ascend_kernel", "") or "").strip().lower()
    if fam in PERSISTENT_MOE_FAMILIES or "megamoe" in rec or "megamoe" in kern:
        return True
    if "dispatch_ffn_combine" in rec or "dispatch_ffn_combine" in kern:
        return True
    spec = getattr(node, "spec", None)
    if spec is not None:
        meta = dict(getattr(spec, "metadata", ()) or ())
        if bool(meta.get("persistent_whole_die")) or bool(meta.get("whole_die_residency")):
            return True
    return False


def is_shared_expert_op_id(op_id: str) -> bool:
    low = str(op_id or "").lower()
    return ".shared." in low or low.startswith("shared.") or "shared_expert" in low


def _reject_moe(
    lvl: int,
    mso: bool,
    reason: str,
    *,
    abi_verdict: AbiSupportVerdict | None = None,
) -> MoERouteDecision:
    return MoERouteDecision(
        enable_fused_mc2=lvl,
        multistream_overlap_shared_expert=mso,
        route_id="illegal",
        abi_id="",
        needs_new_kernel=False,
        is_shipping_abi=False,
        reason=reason,
        legal=False,
        abi_verdict=abi_verdict,
    )


def _model_has_shared_experts(case: Case, nodes: Sequence[Any]) -> bool:
    for lp in getattr(case.model, "layer_plans", ()) or ():
        if int(getattr(lp, "num_shared_experts", 0) or 0) > 0:
            return True
        if int(getattr(lp, "shared_intermediate_size", 0) or 0) > 0:
            return True
    return any(is_shared_expert_op_id(str(getattr(op, "op_id", "") or "")) for op in nodes)


def _resolve_moe_layer_plan(case: Case) -> Any | None:
    for lp in getattr(case.model, "layer_plans", ()) or ():
        if int(getattr(lp, "num_experts", 0) or 0) > 0 or str(getattr(lp, "ffn", "")).lower() == "moe":
            return lp
    return None


def _resolve_moe_activation_strict(
    case: Case,
    nodes: Sequence[Any],
    explicit_activation: str,
) -> tuple[str, bool]:
    """Resolve canonical MoE activation and ``is_act_and_mul`` without silent defaults."""
    raw = str(explicit_activation or "").strip().lower()
    if not raw:
        lp = _resolve_moe_layer_plan(case)
        if lp is not None:
            raw = str(
                getattr(lp, "ffn_activation", "")
                or dict(getattr(lp, "metadata", ()) or ()).get("activation")
                or ""
            ).strip().lower()
    if not raw:
        model_meta = dict(getattr(case.model, "metadata", ()) or ())
        raw = str(model_meta.get("moe_activation") or model_meta.get("hidden_act") or "").strip().lower()
    if not raw:
        for node in nodes:
            spec = getattr(node, "spec", None)
            meta = dict(getattr(spec, "metadata", ()) or ()) if spec is not None else {}
            cand = str(meta.get("activation") or meta.get("activation_kind") or "").strip().lower()
            if cand:
                raw = cand
                break
    if not raw:
        return "", False
    canon = _MOE_ACTIVATION_ALIASES.get(raw, raw)
    is_act_and_mul = canon not in _NON_ACT_AND_MUL_ACTIVATIONS
    return canon, is_act_and_mul


def _resolve_moe_rank_placement(case: Case, soc_arch: SocArch, ep: int) -> RankPlacement:
    if ep <= 1:
        return "local"
    rt_meta = dict(getattr(case.runtime, "metadata", ()) or ())
    explicit = str(rt_meta.get("rank_placement") or "").strip().lower()
    if explicit in RANK_PLACEMENTS and explicit != "local":
        return explicit  # type: ignore[return-value]
    link_names = tuple(str(getattr(link, "name", "") or "").lower() for link in getattr(case.machine, "links", ()))
    has_inter_node = any("inter_node" in name or "roce" in name for name in link_names)
    has_superpod = any("superpod" in name or "supernode" in name or "urma" in name for name in link_names)
    if soc_arch == "A5":
        if has_inter_node and has_superpod:
            return "layered_ub_roce"
        if has_superpod:
            return "urma_cross_supernode"
        if has_inter_node:
            return "roce_flat_cross_node"
        return "ub_ccu_intramachine"
    if soc_arch == "A3":
        if has_inter_node and not has_superpod:
            return "roce_flat_cross_node"
        return "hccs_supernode_a3" if (has_superpod or ep > _A3_SINGLE_NODE_MAX_EP) else "hccs_single_node"
    if has_inter_node:
        return "a2_switched_multinode"
    return "hccs_single_node"


def _resolve_moe_weight_contract(case: Case) -> tuple[str, str, tuple[int, ...], str]:
    quant = getattr(case.model, "quant", None)
    raw_w = str(
        getattr(quant, "expert_weight_format", "")
        or getattr(quant, "weight_format", "")
        or getattr(case.model, "weight_dtype", "")
        or getattr(case.model, "compute_dtype", "")
    ).strip().lower()
    if not raw_w:
        raise ModelingError("moe_weight_dtype_unspecified")
    w_dt = _norm_abi_dtype(raw_w)
    scale_fmt = str(getattr(quant, "scale_format", "") or "").strip().lower() if quant is not None else ""
    exp_block = int(getattr(quant, "expert_block", 0) or 0) if quant is not None else 0
    group_size = exp_block if exp_block > 1 else None
    wq, wg, ws_dt = _quant_contract_for_port(w_dt, scale_fmt, group_size, is_weight=True)
    return w_dt, wq, wg, ws_dt


def _node_weight_contract(node: Any) -> tuple[str, str, tuple[int, ...], str] | None:
    p_abi = getattr(node, "precision_abi", None)
    meta = dict(getattr(node, "metadata", ()) or ())
    raw_w = str(
        (getattr(p_abi, "weight_dtype", "") if p_abi is not None else "")
        or meta.get("weight_format")
        or meta.get("weight_dtype")
        or getattr(node, "dtype", "")
        or ""
    ).strip().lower()
    if not raw_w:
        return None
    w_dt = _norm_abi_dtype(raw_w)
    scale_fmt = str(
        (getattr(p_abi, "weight_scale_dtype", "") if p_abi is not None else "")
        or meta.get("weight_scale_format")
        or meta.get("quant_scale_dtype")
        or ""
    ).strip().lower()
    wb = getattr(p_abi, "weight_block", None) if p_abi is not None else None
    wb_gs = (
        max((int(x) for x in wb if isinstance(x, int) and not isinstance(x, bool) and x > 0), default=0)
        if isinstance(wb, tuple)
        else (int(wb) if isinstance(wb, int) and not isinstance(wb, bool) and wb > 0 else 0)
    )
    raw_gs = wb_gs if wb_gs > 0 else meta.get("weight_group_size", meta.get("quant_group_size"))
    gs = int(raw_gs) if isinstance(raw_gs, int) and not isinstance(raw_gs, bool) and raw_gs > 1 else None
    wq, wg, ws_dt = _quant_contract_for_port(w_dt, scale_fmt, gs, is_weight=True)
    return w_dt, wq, wg, ws_dt


def _resolve_moe_weight_contracts(
    case: Case,
    nodes: Sequence[Any] = (),
) -> tuple[tuple[str, str, tuple[int, ...], str], tuple[str, str, tuple[int, ...], str]]:
    base = _resolve_moe_weight_contract(case)
    gmm_nodes = [
        n for n in nodes
        if str(getattr(n, "family", "") or "").strip().lower() in {"grouped_gemm", "grouped_matmul", "grouped_moe"}
    ]
    if len(gmm_nodes) >= 2:
        c1 = _node_weight_contract(gmm_nodes[0]) or base
        c2 = _node_weight_contract(gmm_nodes[1]) or base
        return c1, c2
    if len(gmm_nodes) == 1:
        c1 = _node_weight_contract(gmm_nodes[0]) or base
        return c1, c1
    return base, base


def _build_moe_abi_query(
    case: Case,
    *,
    soc_arch: SocArch,
    sku_id: str,
    abi_id: str,
    ep: int,
    placement: RankPlacement,
    activation_kind: str,
    is_act_and_mul: bool,
    shared_experts_per_rank: int,
    pr_track: StackTrack,
    w_dt: str,
    wq: str,
    wg: tuple[int, ...],
    ws_dt: str,
    w2_contract: tuple[str, str, tuple[int, ...], str] | None = None,
) -> AbiQueryKey:
    w2_dt, wq2, wg2, ws2_dt = w2_contract if w2_contract is not None else (w_dt, wq, wg, ws_dt)
    lp = _resolve_moe_layer_plan(case)
    h_dim = int(getattr(case.model, "hidden_size", 0) or 0)
    inter_dim = int(getattr(lp, "moe_intermediate_size", 0) or 0) if lp is not None else 0
    top_k = int(getattr(lp, "top_k", 0) or 0) if lp is not None else 0
    num_experts = int(getattr(lp, "num_experts", 0) or 0) if lp is not None else 0
    if abi_id == "aclnnMoeDistributeDispatchV2":
        operands: tuple[OperandPortSig, ...] = (
            _port("x", "bf16", (1, h_dim)),
            _port("expert_ids", "int32", (1, max(1, top_k)), quant_mode="none"),
        )
        outputs: tuple[OperandPortSig, ...] = (_port("expand_x", "bf16", (1, h_dim)),)
        motif = "moe_decode_dispatch"
        act_field = "none"
    elif abi_id == "dispatch_ffn_combine":
        base_ops: list[OperandPortSig] = [
            _port("x", "bf16", (1, h_dim)),
            _port("weight1", w_dt, (h_dim, inter_dim), layout="FRACTAL_NZ", quant_mode=wq, group_shape=wg, scale_dtype=ws_dt),
            _port("weight2", w2_dt, (inter_dim, h_dim), layout="FRACTAL_NZ", quant_mode=wq2, group_shape=wg2, scale_dtype=ws2_dt),
            _port("expert_idx", "int32", (1, max(1, top_k)), quant_mode="none"),
            _port("scale1", "fp32", (1,)),
            _port("scale2", "fp32", (1,)),
        ]
        if w_dt == "int4" or w2_dt == "int4":
            base_ops.extend((_port("bias1", "fp32", (1,)), _port("bias2", "fp32", (1,))))
        base_ops.append(_port("probs", "fp32", (1, max(1, top_k))))
        operands = tuple(base_ops)
        outputs = (_port("out", "bf16", (1, h_dim)), _port("expert_token_nums", "int64", (1,)))
        motif = "moe_mc2_frame"
        act_field = activation_kind
    else:
        operands = (
            _port("context", "int8", _MEGAMOE_CONTEXT_SHAPE, quant_mode="none"),
            _port("x", "bf16", (1, h_dim)),
            _port("topk_ids", "int32", (1, max(1, top_k)), quant_mode="none"),
            _port("topk_weights", "fp32", (1, max(1, top_k))),
            _port("weight1", w_dt, (h_dim, inter_dim), layout="FRACTAL_NZ", quant_mode=wq, group_shape=wg, scale_dtype=ws_dt),
            _port("weight2", w2_dt, (inter_dim, h_dim), layout="FRACTAL_NZ", quant_mode=wq2, group_shape=wg2, scale_dtype=ws2_dt),
        )
        outputs = (_port("y", "bf16", (1, h_dim)), _port("expert_token_nums", "int64", (1,)))
        motif = "moe_mc2_frame"
        act_field = activation_kind

    return AbiQueryKey(
        soc_arch=soc_arch,
        sku_id=sku_id,
        abi_id=abi_id,
        graph_motif=motif,
        model_scope="target",
        draft_quant_mode="none",
        pr_track=pr_track,
        operands=operands,
        outputs=outputs,
        rank_placement=placement,
        world_size=ep,
        comm_group_id="ep" if ep > 1 else "local",
        comm_domain_width=ep,
        comm_dtype="bf16" if ep > 1 else "none",
        reduction_kind="permute",
        activation_kind=act_field,
        activation_params=(),
        has_bias=(w_dt == "int4" and abi_id == "dispatch_ffn_combine"),
        has_active_mask=True,
        dispatch_quant_mode=0,
        shared_experts_per_rank=shared_experts_per_rank,
        top_k=top_k,
        moe_expert_num=num_experts,
        qk_rope_head_dim=0,
        mla_variant="none",
        sfa_dcp_active=False,
        pcp_size=max(1, int(case.runtime.parallel.pcp)),
        kv_cache_dtype="none",
        has_kpool_indexer=False,
        dcp_size=max(1, int(case.runtime.parallel.dcp)),
        is_act_and_mul=is_act_and_mul,
        enabled_stack_passes=frozenset(),
    )


def moe_route_legality(
    case: Case,
    *,
    enable_fused_mc2: int = 0,
    multistream_overlap_shared_expert: bool = False,
    nodes: Sequence[Any] = (),
    activation: str = "",
    allow_unbound_formula: bool = False,
    registry: ProbedAbiRegistry | None = None,
    track_spec: StackTrackSpec | None = None,
    manifest_bindings: tuple[ManifestBinding, ...] = (),
) -> MoERouteDecision:
    """Validate MoE ``(enable_fused_mc2, MSO)`` legality via canonical ``ProbedAbiRegistry``."""
    lvl = int(enable_fused_mc2)
    mso = bool(multistream_overlap_shared_expert)
    if lvl not in VALID_FUSED_MC2_LEVELS:
        return _reject_moe(lvl, mso, f"invalid_fused_mc2_level:{lvl}")
    if lvl > 0 and mso:
        return _reject_moe(lvl, mso, "fused_mc2_conflicts_with_multistream_overlap_shared_expert")

    arch_tag = resolve_arch_family_tag(case)
    soc_arch = _ARCH_TAG_TO_SOC.get(arch_tag)
    if soc_arch is None:
        return _reject_moe(lvl, mso, f"unsupported_soc_arch:{arch_tag}")
    sku_id = str(getattr(case.machine, "sku", "") or getattr(case.machine, "name", "") or "").strip()
    if not sku_id:
        raise ModelingError("moe_route_sku_unspecified")

    ep = max(1, int(case.runtime.parallel.ep))
    reg = registry or default_abi_registry()
    pr_track: StackTrack = track_spec.track if track_spec is not None else "shipping"
    eff_bindings = manifest_bindings or (track_spec.active_bindings if track_spec is not None else ())
    placement = _resolve_moe_rank_placement(case, soc_arch, ep)
    (w_dt, wq, wg, ws_dt), w2_contract = _resolve_moe_weight_contracts(case, nodes)

    if lvl == 0:
        if mso and not _model_has_shared_experts(case, nodes):
            return _reject_moe(lvl, mso, "mso_requires_shared_experts")
        rid = "discrete_moe_mso" if mso else "discrete_moe_serial"
        verdict: AbiSupportVerdict | None = None
        is_shipping = False
        if ep > 1:
            query = _build_moe_abi_query(
                case, soc_arch=soc_arch, sku_id=sku_id, abi_id="aclnnMoeDistributeDispatchV2",
                ep=ep, placement=placement, activation_kind="none", is_act_and_mul=True,
                shared_experts_per_rank=0, pr_track=pr_track, w_dt=w_dt, wq=wq, wg=wg, ws_dt=ws_dt,
            )
            verdict = reg.verify(query, track_spec=track_spec, manifest_bindings=eff_bindings)
            is_shipping = verdict.status == "admitted_shipping"
        return MoERouteDecision(
            enable_fused_mc2=0,
            multistream_overlap_shared_expert=mso,
            route_id=rid,
            abi_id="discrete_route",
            needs_new_kernel=False,
            is_shipping_abi=is_shipping,
            reason=rid,
            legal=True,
            abi_verdict=verdict,
        )

    if ep <= 1:
        return _reject_moe(lvl, mso, "fused_mc2_requires_ep_gt_1")
    act_kind, is_act_and_mul = _resolve_moe_activation_strict(case, nodes, activation)
    if not act_kind:
        return _reject_moe(lvl, mso, "moe_activation_unspecified")
    if lvl == FUSED_MC2_SHARED_LEVEL:
        if is_latent_moe_model(case):
            return _reject_moe(lvl, mso, "fused_mc2_level3_forbids_latent_moe")
        if not _model_has_shared_experts(case, nodes):
            return _reject_moe(lvl, mso, "fused_mc2_level3_requires_shared_experts")

    abi_id = "dispatch_ffn_combine" if lvl == FUSED_MC2_CSRC_LEVEL else "aclnnMegaMoe"
    lp = _resolve_moe_layer_plan(case)
    shared_per_rank = (
        max(1, int(getattr(lp, "num_shared_experts", 0) or 0))
        if lvl == FUSED_MC2_SHARED_LEVEL else 0
    )
    query = _build_moe_abi_query(
        case, soc_arch=soc_arch, sku_id=sku_id, abi_id=abi_id,
        ep=ep, placement=placement, activation_kind=act_kind, is_act_and_mul=is_act_and_mul,
        shared_experts_per_rank=shared_per_rank, pr_track=pr_track,
        w_dt=w_dt, wq=wq, wg=wg, ws_dt=ws_dt, w2_contract=w2_contract,
    )
    verdict = reg.verify(query, track_spec=track_spec, manifest_bindings=eff_bindings)
    if verdict.status in {"admitted_shipping", "admitted_proposed_pr"}:
        return MoERouteDecision(
            enable_fused_mc2=lvl,
            multistream_overlap_shared_expert=False,
            route_id=f"fused_mc2_level{lvl}",
            abi_id=abi_id,
            needs_new_kernel=False,
            is_shipping_abi=(verdict.status == "admitted_shipping"),
            reason=f"{abi_id}:{verdict.matched_entry_id}",
            legal=True,
            abi_verdict=verdict,
        )
    allows_custom = allow_unbound_formula and (track_spec is None or track_spec.allows_target_synthesis)
    if (
        allows_custom
        and verdict.available_on_device
        and verdict.rejection_reason.startswith("rejected_dtype:")
    ):
        return MoERouteDecision(
            enable_fused_mc2=lvl,
            multistream_overlap_shared_expert=False,
            route_id=f"fused_mc2_level{lvl}",
            abi_id=abi_id,
            needs_new_kernel=True,
            is_shipping_abi=False,
            reason=f"{abi_id}_custom_synthesis:{verdict.rejection_reason}",
            legal=True,
            abi_verdict=verdict,
        )
    return _reject_moe(lvl, False, verdict.rejection_reason, abi_verdict=verdict)


def _record_meta(rec: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = rec.get("spec")
    if spec is not None and getattr(spec, "metadata", None):
        return dict(spec.metadata)
    raw = rec.get("metadata")
    return dict(raw) if isinstance(raw, Mapping) else {}


def is_draft_record(rec: Mapping[str, Any]) -> bool:
    """True when a timeline/occurrence record belongs to speculative draft scope."""
    oid = str(rec.get("op_id") or "")
    if oid.startswith("draft"):
        return True
    meta = _record_meta(rec)
    for key in ("layer_scope", "producer_scope", "model_scope", "structure_model_scope"):
        if str(meta.get(key) or "").strip().lower() == "draft":
            return True
    if bool(int(meta.get("is_draft_model", 0) or 0)):
        return True
    return any(str(rec.get(k) or "").strip().lower() == "draft" for k in ("layer_scope", "producer_scope", "model_scope"))


def is_acceptance_commit_record(rec: Mapping[str, Any]) -> bool:
    """True when ``rec`` is an acceptance-bound per-layer state commit (e.g. recover_ssm_commit)."""
    if is_draft_record(rec):
        return False
    meta = _record_meta(rec)
    if str(meta.get("acceptance_boundary") or "").strip().lower() == "sampling_acceptance":
        return True
    if str(meta.get("recover_ssm_stage") or "").strip().lower() == "commit":
        return True
    return str(rec.get("family") or "").strip().lower() == "recover_ssm_commit"


def compute_acceptance_tail_closure(
    records: Sequence[Mapping[str, Any]],
) -> frozenset[OpOccurrence]:
    """Compute exact causal ``target_tail`` closure on the full DAG BEFORE layer folding."""
    if not records:
        return frozenset()
    occ_by_op_key: dict[tuple[int, str], OpOccurrence] = {}
    is_draft_occ: dict[OpOccurrence, bool] = {}
    preds_by_occ: dict[OpOccurrence, list[OpOccurrence]] = {}
    occ_order: list[OpOccurrence] = []

    for rec in records:
        occ: OpOccurrence = rec["occurrence"]
        oid = str(rec.get("op_id") or occ.op_id)
        occ_by_op_key[(occ.replica, oid)] = occ
        is_draft_occ[occ] = is_draft_record(rec)
        occ_order.append(occ)

    closure: set[OpOccurrence] = set()
    for rec in records:
        occ = rec["occurrence"]
        if is_draft_occ.get(occ, False):
            preds_by_occ[occ] = []
            continue
        oid = str(rec.get("op_id") or occ.op_id)
        fam = str(rec.get("family") or "").strip().lower()
        if (
            oid in {"accept", "token_commit"}
            or fam in ACCEPTANCE_ANCHOR_FAMILIES
            or fam in TAIL_COMMIT_FAMILIES
            or is_acceptance_commit_record(rec)
        ):
            closure.add(occ)
        preds: list[OpOccurrence] = list(rec.get("predecessors") or ())
        spec = rec.get("spec")
        if spec is not None:
            for dep in getattr(spec, "deps", ()) or ():
                dep_occ = occ_by_op_key.get((occ.replica, str(dep)))
                if dep_occ is not None:
                    preds.append(dep_occ)
        preds_by_occ[occ] = preds

    changed = True
    while changed:
        changed = False
        for occ in occ_order:
            if occ in closure or is_draft_occ.get(occ, False):
                continue
            if any(pred in closure for pred in preds_by_occ.get(occ, ())):
                closure.add(occ)
                changed = True
    return frozenset(closure)


def resolve_barrier_predecessors(
    records: Sequence[Mapping[str, Any]],
    *,
    barrier_eps_us: float = BARRIER_EPS_US,
    protected_occurrences: frozenset[OpOccurrence] = frozenset(),
) -> list[dict[str, Any]]:
    """Filter zero-duration barriers while transitively preserving ALL upstream predecessors."""
    del protected_occurrences
    removed_preds: dict[OpOccurrence, tuple[OpOccurrence, ...]] = {}
    kept_raw: list[dict[str, Any]] = []
    for rec in records:
        occ: OpOccurrence = rec["occurrence"]
        oid = str(rec.get("op_id") or occ.op_id)
        dur = max(0.0, float(rec["end_us"]) - float(rec["start_us"]))
        fam = str(rec.get("family") or "").strip().lower()
        if fam == "barrier" and dur <= barrier_eps_us and oid not in {"accept", "token_commit"}:
            removed_preds[occ] = tuple(rec.get("predecessors") or ())
            continue
        kept_raw.append(dict(rec))

    if not removed_preds:
        return kept_raw

    resolved_cache: dict[OpOccurrence, tuple[OpOccurrence, ...]] = {}

    def _expand(occ: OpOccurrence, visiting: set[OpOccurrence]) -> tuple[OpOccurrence, ...]:
        if occ not in removed_preds:
            return (occ,)
        if occ in resolved_cache:
            return resolved_cache[occ]
        if occ in visiting:
            return ()
        visiting.add(occ)
        out: list[OpOccurrence] = []
        for pred in removed_preds[occ]:
            out.extend(_expand(pred, visiting))
        visiting.discard(occ)
        dedup = tuple(dict.fromkeys(out))
        resolved_cache[occ] = dedup
        return dedup

    for rec in kept_raw:
        rewired: list[OpOccurrence] = []
        for pred in rec.get("predecessors") or ():
            rewired.extend(_expand(pred, set()))
        occ = rec["occurrence"]
        rec["predecessors"] = tuple(dict.fromkeys(p for p in rewired if p != occ))
    return kept_raw
