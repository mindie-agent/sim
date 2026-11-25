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
from dataclasses import dataclass, replace
from typing import Any

from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.catalog.quant import projection_quant
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.projections import effective_linear_method
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
_MEGAMOE_DISPATCH_MODE_PASSTHROUGH = 0
_MEGAMOE_DISPATCH_MODE_INT8 = 2
_MEGAMOE_DISPATCH_MODE_MXFP = 4
_MEGAMOE_MAX_EXPERTS_A5 = 2048
_MEGAMOE_MAX_EXPERT_PER_RANK_A5 = 1024
_MEGAMOE_MAX_EXPERTS_A23 = 1024
_MEGAMOE_MAX_EXPERT_PER_RANK_A23 = 128
_MEGAMOE_GATED_WIDTH_FACTOR = 2
_MEGAMOE_MX_BLOCK_SIZE = 32
_MEGAMOE_MX_SCALE_ALIGN = 64
_MEGAMOE_MX_SCALE_TRAILING = 2
_EP_EXHAUSTIVE_PAIR_LIMIT = 16
_EP_REPS_PER_PACKAGE = 2
_MEGAMOE_MXFP_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e5m2", "fp4_e2m1"})
_MEGAMOE_FP8_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e5m2"})
_MEGAMOE_INT_DTYPES = frozenset({"int8", "int4"})
_MEGAMOE_FLOAT_DTYPES = frozenset({"bf16", "fp16"})
_MEGAMOE_MX_SCALE_FORMATS = frozenset({"fp8_e8m0", "ue8m0"})
_MEGAMOE_A5_VALID_ACT_WEIGHT_PAIRS = frozenset({
    ("bf16", "bf16"),
    ("fp8_e4m3fn", "fp8_e4m3fn"),
    ("fp8_e5m2", "fp8_e5m2"),
    ("fp8_e4m3fn", "fp4_e2m1"),
    ("fp4_e2m1", "fp4_e2m1"),
})
_LEVEL3_SHARED_ACTIVATIONS = frozenset({"swiglu", "situglu", "swigluoai"})
_LEVEL3_MXFP_METHODS = frozenset({"mx_fp4", "mx_fp8"})
_SUPERPOD_TOKENS = ("superpod", "supernode", "urma", "ub_optical", "optical_hccs")
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
    "geglu": "geglu",
    "gelu": "gelu",
    "relu_squared": "relu2",
    "relu2": "relu2",
    "swigluoai": "swigluoai",
    "swiglustep": "swiglustep",
    "clipped_swiglu": "clipped_swiglu",
}
_NON_ACT_AND_MUL_ACTIVATIONS = frozenset({"relu2", "gelu"})


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
    facts = dict(getattr(case.model, "catalog_facts", ()) or ())
    if facts.get("routed_expert_hidden_size") is not None:
        return True
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


def _is_moe_layer_plan(lp: Any) -> bool:
    return int(getattr(lp, "num_experts", 0) or 0) > 0 or str(getattr(lp, "ffn", "")).lower() == "moe"


def _moe_layer_plans(case: Case) -> tuple[Any, ...]:
    return tuple(lp for lp in (getattr(case.model, "layer_plans", ()) or ()) if _is_moe_layer_plan(lp))


def _resolve_moe_layer_plan(case: Case) -> Any | None:
    plans = _moe_layer_plans(case)
    return plans[0] if plans else None


def _resolve_moe_activation_strict(
    case: Case,
    nodes: Sequence[Any],
    explicit_activation: str,
    lp: Any | None = None,
) -> tuple[str, bool]:
    """Resolve canonical MoE activation and ``is_act_and_mul`` without silent defaults."""
    raw = str(explicit_activation or "").strip().lower()
    if not raw:
        eff_lp = lp if lp is not None else _resolve_moe_layer_plan(case)
        if eff_lp is not None:
            raw = str(
                getattr(eff_lp, "ffn_activation", "")
                or dict(getattr(eff_lp, "metadata", ()) or ()).get("activation")
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


def _resolve_moe_activation_params(lp: Any | None, act_kind: str) -> tuple[tuple[str, float], ...]:
    """Extract canonical typed activation parameters for CANN ``aclnnMegaMoe``."""
    if act_kind == "situglu":
        beta = float(getattr(lp, "activation_situ_beta", 1.0) if lp is not None else 1.0)
        params: list[tuple[str, float]] = [("beta", beta)]
        lin_beta = getattr(lp, "activation_situ_linear_beta", None) if lp is not None else None
        if lin_beta is not None:
            params.append(("linear_beta", float(lin_beta)))
        return tuple(params)
    if act_kind == "swigluoai":
        alpha = float(getattr(lp, "swiglu_alpha", 1.0) if lp is not None else 1.0)
        beta = float(getattr(lp, "swiglu_beta", 0.0) if lp is not None else 0.0)
        clamp = float(getattr(lp, "swiglu_limit", 0.0) or 0.0) if lp is not None else 0.0
        params = [("alpha", alpha), ("beta", beta)]
        if clamp > 0.0:
            params.append(("clamp", clamp))
        return tuple(params)
    if act_kind in {"swiglu", "swiglustep"}:
        clamp = float(getattr(lp, "swiglu_limit", 0.0) or 0.0) if lp is not None else 0.0
        return (("clamp", clamp),) if clamp > 0.0 else ()
    return ()


def _moe_pp_stages(case: Case) -> tuple[int, ...]:
    """Return pipeline stages that actually execute at least one MoE layer."""
    pp = max(1, int(case.runtime.parallel.pp))
    plans = tuple(getattr(case.model, "layer_plans", ()) or ())
    if pp <= 1 or not plans:
        return tuple(range(pp))
    raw_part = tuple(int(v) for v in (getattr(case.runtime, "pp_layer_partition", ()) or ()))
    n_layers = int(getattr(case.model, "layers", 0) or len(plans))
    counts = raw_part if len(raw_part) == pp else tuple(
        n_layers // pp + int(s < n_layers % pp) for s in range(pp)
    )
    stage_by_layer = tuple(stage for stage, count in enumerate(counts) for _ in range(count))
    stages = sorted({
        stage_by_layer[idx]
        for idx, lp in enumerate(plans)
        if idx < len(stage_by_layer) and _is_moe_layer_plan(lp)
    })
    return tuple(stages) if stages else tuple(range(pp))


def _ep_group_representatives(
    grp: tuple[int, ...], die_by_rank: Mapping[int, Any],
) -> tuple[int, ...]:
    if len(grp) <= _EP_EXHAUSTIVE_PAIR_LIMIT:
        return grp
    reps: list[int] = []
    counts: dict[tuple[Any, Any], int] = {}
    for rank in grp:
        die = die_by_rank.get(rank)
        key = (getattr(die, "node", 0), getattr(die, "package", 0)) if die is not None else (0, 0)
        seen = counts.get(key, 0)
        if seen < _EP_REPS_PER_PACKAGE:
            reps.append(rank)
            counts[key] = seen + 1
    return tuple(reps)


def _link_protocol_tokens(link_name: str, link: Any, channel_map: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    tokens = [link_name.lower()]
    if link is not None:
        raw_proto = str(getattr(link, "protocol", "") or "").strip().lower()
        if raw_proto:
            tokens.append(raw_proto)
        ch_name = str(getattr(link, "channel", "") or link_name.split(":")[0]).strip().lower()
    else:
        ch_name = link_name.split(":")[0].strip().lower()
    ch_decl = channel_map.get(ch_name)
    if ch_decl:
        for fld in ("protocol", "transport_mode", "switch_id", "rail_id"):
            val = str(ch_decl.get(fld) or "").strip().lower()
            if val:
                tokens.append(val)
    return tuple(tokens)


def _resolve_ep_group_links(case: Case) -> tuple[bool, bool, int]:
    """Resolve ``(has_superpod, has_roce_inter_node, max_nodes)`` across active MoE EP groups."""
    channel_map = {
        str(k).strip().lower(): dict(v)
        for k, v in (getattr(case.machine, "transfer_channels", ()) or ())
        if isinstance(v, Mapping) or (isinstance(v, tuple) and all(isinstance(it, tuple) and len(it) == _MEGAMOE_GATED_WIDTH_FACTOR for it in v))
    }
    dies = getattr(case.machine, "dies", ()) or ()
    links = getattr(case.machine, "links", ()) or ()
    if not dies:
        has_superpod = False
        has_roce = False
        for lk in links:
            lname = str(getattr(lk, "name", "") or "").lower()
            toks = _link_protocol_tokens(lname, lk, channel_map)
            is_sp = any(st in tok for tok in toks for st in _SUPERPOD_TOKENS)
            has_superpod = has_superpod or is_sp
            if any("roce" in tok for tok in toks) or ("inter_node" in lname and not is_sp):
                has_roce = True
        return has_superpod, has_roce, 1
    parallel = case.runtime.parallel
    topo = Topology(case.machine, parallel)
    ep_groups = tuple(dict.fromkeys(
        tuple(grp)
        for rep in range(max(1, int(parallel.dp)))
        for stage in _moe_pp_stages(case)
        for grp in topo.groups("ep", rep, stage)
    ))
    die_by_rank = {d.rank: d for d in dies}
    link_by_name = {link.name: link for link in links}
    has_superpod = False
    has_roce = False
    max_nodes = 1
    for grp in ep_groups:
        nodes = {die_by_rank[r].node for r in grp if r in die_by_rank}
        max_nodes = max(max_nodes, len(nodes))
        reps = _ep_group_representatives(grp, die_by_rank)
        for idx, src in enumerate(reps):
            for dst in reps[idx + 1:]:
                for link_name, _, _ in topo.route(src, dst):
                    lk = link_by_name.get(link_name)
                    toks = _link_protocol_tokens(link_name, lk, channel_map)
                    is_sp = any(st in tok for tok in toks for st in _SUPERPOD_TOKENS)
                    if is_sp:
                        has_superpod = True
                    scope = str(getattr(lk, "scope", "") or "").strip().lower() if lk is not None else ""
                    if any("roce" in tok for tok in toks) or (("inter_node" in link_name.lower() or scope == "cross_node") and not is_sp):
                        has_roce = True
    if max_nodes > 1 and not has_superpod:
        has_roce = True
    return has_superpod, has_roce, max_nodes


def _resolve_moe_rank_placement(case: Case, soc_arch: SocArch, ep: int) -> RankPlacement:
    if ep <= 1:
        return "local"
    rt_meta = dict(getattr(case.runtime, "metadata", ()) or ())
    explicit = str(rt_meta.get("rank_placement") or "").strip().lower()
    if explicit in RANK_PLACEMENTS and explicit != "local":
        return explicit  # type: ignore[return-value]
    has_superpod, has_roce, _ = _resolve_ep_group_links(case)
    if soc_arch == "A5":
        if has_roce and has_superpod:
            return "layered_ub_roce"
        if has_superpod:
            return "urma_cross_supernode"
        if has_roce:
            return "roce_flat_cross_node"
        return "ub_ccu_intramachine"
    if soc_arch == "A3":
        if has_roce and not has_superpod:
            return "roce_flat_cross_node"
        return "hccs_supernode_a3" if (has_superpod or ep > _A3_SINGLE_NODE_MAX_EP) else "hccs_single_node"
    if has_roce or has_superpod:
        return "a2_switched_multinode"
    return "hccs_single_node"


def _resolve_moe_weight_contract(case: Case, lp: Any | None = None) -> tuple[str, str, tuple[int, ...], str]:
    base_quant = getattr(case.model, "quant", None)
    eff_lp = lp if lp is not None else _resolve_moe_layer_plan(case)
    quant = (
        projection_quant(base_quant, eff_lp).for_category("routed_expert")
        if (base_quant is not None and eff_lp is not None)
        else (base_quant.for_category("routed_expert") if base_quant is not None else None)
    )
    raw_w = str(
        getattr(quant, "expert_weight_format", "")
        or getattr(quant, "weight_format", "")
        or getattr(case.model, "weight_dtype", "")
        or getattr(case.model, "compute_dtype", "")
    ).strip().lower()
    if not raw_w:
        raise ModelingError("moe_weight_dtype_unspecified")
    w_dt = _norm_abi_dtype(raw_w)
    scale_fmt = str(
        (quant.execution_weight_scale_format() or quant.scale_format or "") if quant is not None else ""
    ).strip().lower()
    wb = tuple(getattr(quant, "weight_block", ()) or ()) if quant is not None else ()
    wb_k = int(wb[1]) if len(wb) > 1 and int(wb[1]) > 0 else (int(wb[0]) if len(wb) == 1 and int(wb[0]) > 0 else 0)
    exp_block = int(getattr(quant, "expert_block", 0) or 0) if quant is not None else 0
    act_group = int(getattr(quant, "activation_group_size", 0) or 0) if quant is not None else 0
    if w_dt in _MEGAMOE_FP8_DTYPES:
        eff_group = wb_k if wb_k > 1 else (exp_block if exp_block > 1 else None)
        if eff_group == _MEGAMOE_MX_BLOCK_SIZE and act_group > 1 and act_group != _MEGAMOE_MX_BLOCK_SIZE:
            eff_group = act_group
    else:
        eff_group = exp_block if exp_block > 1 else (wb_k if wb_k > 1 else None)
        if w_dt == "fp4_e2m1" and eff_group == _MEGAMOE_MX_BLOCK_SIZE and act_group > 1 and act_group != _MEGAMOE_MX_BLOCK_SIZE:
            eff_group = act_group
    wq, wg, ws_dt = _quant_contract_for_port(w_dt, scale_fmt, eff_group, is_weight=True)
    return w_dt, wq, wg, ws_dt


def _resolve_moe_routed_act_dtype(case: Case, nodes: Sequence[Any] = (), lp: Any | None = None) -> str:
    """Resolve routed-expert internal activation dtype (distinguishing A8W4-FP vs A4W4-FP)."""
    for node in nodes:
        fam = str(getattr(node, "family", "") or "").strip().lower()
        if fam not in {"grouped_gemm", "grouped_matmul", "grouped_moe"}:
            continue
        p_abi = getattr(node, "precision_abi", None)
        raw_in = str(getattr(p_abi, "input_dtype", "") or "").strip().lower() if p_abi is not None else ""
        if raw_in:
            return _norm_abi_dtype(raw_in)
    base_quant = getattr(case.model, "quant", None)
    if base_quant is not None:
        eff_lp = lp if lp is not None else _resolve_moe_layer_plan(case)
        lq = projection_quant(base_quant, eff_lp) if eff_lp is not None else base_quant
        _, raw_act = compute_dtype_for(lq, "routed_expert")
        if raw_act:
            return _norm_abi_dtype(str(raw_act))
    raw_act = str(
        getattr(case.model, "activation_dtype", "")
        or getattr(case.model, "compute_dtype", "")
        or "bf16"
    ).strip().lower()
    return _norm_abi_dtype(raw_act)


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


_MATMUL_SHAPE_RANK = 3


def _classify_moe_gmm_role(node: Any, hidden_size: int, inter_size: int) -> str:
    """Classify a grouped-MoE GEMM node into ``'w1'`` (gate/up) or ``'w2'`` (down) by role/shape."""
    meta = dict(getattr(node, "metadata", ()) or ())
    tokens = " ".join((
        str(meta.get("role") or ""),
        str(meta.get("projection") or ""),
        str(meta.get("quant_category") or ""),
        str(getattr(node, "op_id", "") or ""),
    )).lower()
    if any(tok in tokens for tok in ("down", "w2", "fc2", "proj_down")):
        return "w2"
    if any(tok in tokens for tok in ("gate", "up", "w13", "w1", "fc1", "proj_up")):
        return "w1"
    shape = getattr(node, "local_shape", ()) or ()
    if len(shape) == _MATMUL_SHAPE_RANK and hidden_size > 0:
        _, n_dim, k_dim = (int(shape[0]), int(shape[1]), int(shape[2]))
        if n_dim == hidden_size and (inter_size <= 0 or k_dim != hidden_size):
            return "w2"
        if k_dim == hidden_size and (inter_size <= 0 or n_dim != hidden_size):
            return "w1"
    return ""


def _node_declared_weight_layout(node: Any) -> str:
    p_abi = getattr(node, "precision_abi", None)
    meta = dict(getattr(node, "metadata", ()) or ())
    raw = str(
        (getattr(p_abi, "weight_layout", "") if p_abi is not None else "")
        or meta.get("weight_layout")
        or meta.get("layout")
        or ""
    ).strip()
    return raw


def _resolve_moe_node_weight_layouts(
    case: Case, nodes: Sequence[Any], lp: Any | None = None,
) -> tuple[str, str]:
    """Return explicitly declared ``(w1_layout, w2_layout)`` from bound grouped-GEMM nodes if present."""
    gmm_nodes = [
        n for n in nodes
        if str(getattr(n, "family", "") or "").strip().lower() in {"grouped_gemm", "grouped_matmul", "grouped_moe"}
    ]
    if not gmm_nodes:
        return "", ""
    eff_lp = lp if lp is not None else _resolve_moe_layer_plan(case)
    hidden_size = int(getattr(case.model, "hidden_size", 0) or 0)
    inter_size = int(getattr(eff_lp, "moe_intermediate_size", 0) or 0) if eff_lp is not None else 0
    w1_lay = ""
    w2_lay = ""
    for n in gmm_nodes:
        lay = _node_declared_weight_layout(n)
        if not lay:
            continue
        role = _classify_moe_gmm_role(n, hidden_size, inter_size)
        if role == "w1" and not w1_lay:
            w1_lay = lay
        elif role == "w2" and not w2_lay:
            w2_lay = lay
    return w1_lay, w2_lay


def _resolve_moe_weight_contracts(
    case: Case,
    nodes: Sequence[Any] = (),
    lp: Any | None = None,
) -> tuple[tuple[str, str, tuple[int, ...], str], tuple[str, str, tuple[int, ...], str]]:
    base = _resolve_moe_weight_contract(case, lp=lp)
    gmm_nodes = [
        n for n in nodes
        if str(getattr(n, "family", "") or "").strip().lower() in {"grouped_gemm", "grouped_matmul", "grouped_moe"}
    ]
    if not gmm_nodes:
        return base, base
    eff_lp = lp if lp is not None else _resolve_moe_layer_plan(case)
    hidden_size = int(getattr(case.model, "hidden_size", 0) or 0)
    inter_size = int(getattr(eff_lp, "moe_intermediate_size", 0) or 0) if eff_lp is not None else 0
    w1_contracts: list[tuple[str, str, tuple[int, ...], str]] = []
    w2_contracts: list[tuple[str, str, tuple[int, ...], str]] = []
    unknown_nodes: list[Any] = []
    for n in gmm_nodes:
        role = _classify_moe_gmm_role(n, hidden_size, inter_size)
        contract = _node_weight_contract(n) or base
        if role == "w1":
            w1_contracts.append(contract)
        elif role == "w2":
            w2_contracts.append(contract)
        else:
            unknown_nodes.append(n)
    if unknown_nodes:
        # Order-invariant DAG disambiguation: nodes feeding another grouped GEMM are w1, sinks are w2.
        gmm_ids = {str(getattr(n, "op_id", "") or "") for n in gmm_nodes}
        fed_ids = {
            dep for n in gmm_nodes
            for dep in (getattr(n, "deps", ()) or ())
            if dep in gmm_ids
        }
        for n in unknown_nodes:
            oid = str(getattr(n, "op_id", "") or "")
            contract = _node_weight_contract(n) or base
            if oid and oid in fed_ids:
                w1_contracts.append(contract)
            else:
                w2_contracts.append(contract)
    # Canonical order-invariant representative (sort distinct contracts so list permutation cannot flip verdict).
    c1_set = sorted(set(w1_contracts))
    c2_set = sorted(set(w2_contracts))
    if c1_set and c2_set:
        # If multiple up/gate projections disagree, prefer the non-base/stricter contract deterministically.
        c1 = next((c for c in c1_set if c != base), c1_set[0])
        c2 = next((c for c in c2_set if c != base), c2_set[0])
        return c1, c2
    only = c1_set or c2_set
    rep = next((c for c in only if c != base), only[0]) if only else base
    return rep, rep


def _megamoe_weight_layouts(
    soc_arch: SocArch,
    w1_dt: str,
    w2_dt: str,
    routed_act_dt: str,
    declared_w1_layout: str = "",
    declared_w2_layout: str = "",
) -> tuple[str, str]:
    """Return coupled ``(weight1_layout, weight2_layout)`` honoring declared node layouts."""
    if soc_arch == "A5":
        if w1_dt == "fp4_e2m1":
            default_w1 = "FRACTAL_NZ" if routed_act_dt == "fp4_e2m1" else "FRACTAL_NZ_C0_32"
        elif w1_dt in {"fp8_e5m2", "bf16"}:
            default_w1 = "ND"
        else:
            default_w1 = "FRACTAL_NZ"
        if w2_dt == "fp4_e2m1":
            default_w2 = "FRACTAL_NZ_C0_32"
        elif w2_dt in {"fp8_e5m2", "bf16"}:
            default_w2 = "ND"
        else:
            default_w2 = "FRACTAL_NZ"
        return declared_w1_layout or default_w1, declared_w2_layout or default_w2
    default_w1 = "ND" if w1_dt in _MEGAMOE_FLOAT_DTYPES else "FRACTAL_NZ"
    default_w2 = "ND" if w2_dt in _MEGAMOE_FLOAT_DTYPES else "FRACTAL_NZ"
    return declared_w1_layout or default_w1, declared_w2_layout or default_w2


def _megamoe_physical_local_experts(num_experts: int, ep: int, redundant: int) -> int:
    return (num_experts // ep) + redundant


def _check_megamoe_expert_placement(case: Case, soc_arch: SocArch, ep: int) -> str | None:
    """Enforce CANN & vLLM-Ascend uniform EP expert divisibility, redundant slots, and ETP splits."""
    plans = _moe_layer_plans(case)
    if not plans:
        return "moe_layer_plan_missing"
    max_experts = _MEGAMOE_MAX_EXPERTS_A5 if soc_arch == "A5" else _MEGAMOE_MAX_EXPERTS_A23
    max_per_rank = _MEGAMOE_MAX_EXPERT_PER_RANK_A5 if soc_arch == "A5" else _MEGAMOE_MAX_EXPERT_PER_RANK_A23
    redundant = max(0, int(getattr(case.runtime.parallel, "redundant_experts_per_rank", 0) or 0))
    etp = max(1, int(getattr(case.runtime.parallel, "etp", 1) or 1))
    for lp in plans:
        num_experts = int(getattr(lp, "num_experts", 0) or 0)
        if num_experts < ep or (num_experts % ep) != 0:
            return f"megamoe_experts_indivisible:E={num_experts},ep={ep}"
        per_rank_phys = _megamoe_physical_local_experts(num_experts, ep, redundant)
        total_phys = per_rank_phys * ep
        if total_phys > max_experts or per_rank_phys < 1 or per_rank_phys > max_per_rank:
            return (
                f"megamoe_experts_out_of_range:E={num_experts},ep={ep},"
                f"redundant={redundant},per_rank_phys={per_rank_phys},total_phys={total_phys}"
            )
        moe_inter = int(getattr(lp, "moe_intermediate_size", 0) or 0)
        if etp > 1 and moe_inter > 0 and (moe_inter % etp) != 0:
            return f"megamoe_inter_indivisible_by_etp:inter={moe_inter},etp={etp}"
    return None


def _resolve_mxfp_category_mode(lq: Any, category: str) -> tuple[str, str, str]:
    """Return ``(method, act_dt, weight_dt)`` where ``method`` is ``mx_fp4``/``mx_fp8`` only for true MXFP."""
    method = effective_linear_method(lq, category)
    q_cat = lq.for_category(category)
    _, raw_act = compute_dtype_for(lq, category)
    act_dt = _norm_abi_dtype(str(raw_act or "bf16"))
    raw_w = str(
        (getattr(q_cat, "expert_weight_format", "") if category == "routed_expert" else "")
        or getattr(q_cat, "weight_format", "")
        or "bf16"
    ).strip().lower()
    w_dt = _norm_abi_dtype(raw_w)
    if method not in _LEVEL3_MXFP_METHODS and category == "routed_expert" and w_dt == "fp4_e2m1":
        sf = str(q_cat.execution_weight_scale_format() or q_cat.scale_format or "").strip().lower()
        if (sf in _MEGAMOE_MX_SCALE_FORMATS
                and int(getattr(q_cat, "expert_block", 0) or 0) == _MEGAMOE_MX_BLOCK_SIZE
                and int(getattr(q_cat, "activation_group_size", 0) or 0) == _MEGAMOE_MX_BLOCK_SIZE):
            method = "mx_fp4"
    return method, act_dt, w_dt


def _check_fused_mc2_level3_shared_eligibility(
    case: Case,
    soc_arch: SocArch,
    placement: RankPlacement,
    act_kind: str,
    nodes: Sequence[Any],
    lp: Any | None = None,
) -> str | None:
    """Enforce upstream vLLM-Ascend ``shared_experts.py`` + CANN level-3 shared-MLP fusion guards."""
    if soc_arch != "A5":
        return f"fused_mc2_level3_requires_a5:{soc_arch}"
    if is_latent_moe_model(case):
        return "fused_mc2_level3_forbids_latent_moe"
    if not _model_has_shared_experts(case, nodes):
        return "fused_mc2_level3_requires_shared_experts"
    if act_kind not in _LEVEL3_SHARED_ACTIVATIONS:
        return f"fused_mc2_level3_unsupported_activation:{act_kind}"
    rt_meta = dict(getattr(case.runtime, "metadata", ()) or ())
    exec_policy = rt_meta.get("execution_policy")
    enable_shared_expert_dp = bool(rt_meta.get("enable_shared_expert_dp", False)) or (
        isinstance(exec_policy, Mapping) and bool(exec_policy.get("enable_shared_expert_dp", False))
    )
    tp = max(1, int(getattr(case.runtime.parallel, "tp", 1) or 1))
    shared_tp_cfg = max(1, int(getattr(case.runtime.parallel, "shared_expert_tp", 1) or 1))
    if (not enable_shared_expert_dp and tp > 1) or shared_tp_cfg > 1:
        return (
            f"fused_mc2_level3_forbids_tp_sharded_shared_experts:"
            f"tp={tp},shared_expert_tp={shared_tp_cfg},shared_expert_dp={enable_shared_expert_dp}"
        )
    etp = max(1, int(getattr(case.runtime.parallel, "etp", 1) or 1))
    moe_plans = (lp,) if lp is not None else _moe_layer_plans(case)
    model_quant = getattr(case.model, "quant", None)
    for cur_lp in moe_plans:
        n_shared = int(getattr(cur_lp, "num_shared_experts", 0) or 0)
        sh_inter = int(getattr(cur_lp, "shared_intermediate_size", 0) or 0)
        if n_shared <= 0 and sh_inter <= 0:
            continue
        if bool(getattr(cur_lp, "shared_expert_gate", False)):
            return "fused_mc2_level3_forbids_shared_expert_gate"
        sh_limit = float(getattr(cur_lp, "swiglu_shared_limit", 0.0) or 0.0)
        r_limit = float(getattr(cur_lp, "swiglu_limit", 0.0) or 0.0)
        if sh_limit != r_limit:
            return f"fused_mc2_level3_shared_activation_mismatch:routed={r_limit},shared={sh_limit}"
        moe_inter = int(getattr(cur_lp, "moe_intermediate_size", 0) or 0)
        if moe_inter > 0:
            local_routed_inter = moe_inter // etp
            effective_shared_inter = sh_inter if sh_inter > 0 else (n_shared * moe_inter)
            if effective_shared_inter != local_routed_inter:
                return (
                    f"fused_mc2_level3_shared_intermediate_mismatch:"
                    f"shared={effective_shared_inter},routed_local={local_routed_inter}"
                )
        if model_quant is None:
            return "fused_mc2_level3_requires_mxfp_quant"
        lq = projection_quant(model_quant, cur_lp)
        r_method, r_act, r_w = _resolve_mxfp_category_mode(lq, "routed_expert")
        s_method, s_act, s_w = _resolve_mxfp_category_mode(lq, "shared_expert")
        if r_method not in _LEVEL3_MXFP_METHODS:
            return f"fused_mc2_level3_requires_mxfp_routed:{r_method}"
        if s_method not in _LEVEL3_MXFP_METHODS:
            return f"fused_mc2_level3_requires_mxfp_shared:{s_method}"
        if placement != "ub_ccu_intramachine" and (r_act, r_w) != (s_act, s_w):
            return (
                f"fused_mc2_level3_mixed_quant_requires_mte:"
                f"routed={r_act}/{r_w},shared={s_act}/{s_w},placement={placement}"
            )
    return None


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
    routed_act_dt: str = "bf16",
    lp: Any | None = None,
    declared_w1_layout: str = "",
    declared_w2_layout: str = "",
) -> AbiQueryKey:
    w2_dt, wq2, wg2, ws2_dt = w2_contract if w2_contract is not None else (w_dt, wq, wg, ws_dt)
    eff_lp = lp if lp is not None else _resolve_moe_layer_plan(case)
    latent_h = int(getattr(eff_lp, "expert_latent_size", 0) or 0) if eff_lp is not None else 0
    h_dim = latent_h if latent_h > 0 else int(getattr(case.model, "hidden_size", 0) or 0)
    etp = max(1, int(getattr(case.runtime.parallel, "etp", 1) or 1))
    raw_inter = int(getattr(eff_lp, "moe_intermediate_size", 0) or 0) if eff_lp is not None else 0
    inter_dim = raw_inter // etp if raw_inter > 0 else 0
    top_k = int(getattr(eff_lp, "top_k", 0) or 0) if eff_lp is not None else 0
    num_experts = int(getattr(eff_lp, "num_experts", 0) or 0) if eff_lp is not None else 0
    redundant = max(0, int(getattr(case.runtime.parallel, "redundant_experts_per_rank", 0) or 0))
    local_experts = max(1, (num_experts // max(1, ep)) + redundant) if num_experts > 0 else 1
    phys_total_experts = local_experts * max(1, ep) if (num_experts > 0 and (num_experts % max(1, ep)) == 0) else num_experts
    tp = max(1, int(getattr(case.runtime.parallel, "tp", 1) or 1))
    has_active_mask = True
    dispatch_quant_mode = _MEGAMOE_DISPATCH_MODE_PASSTHROUGH
    act_params: tuple[tuple[str, float], ...] = ()

    if abi_id == "aclnnMoeDistributeDispatchV2":
        dispatch_cap = max(1, ep * min(local_experts, max(1, top_k)))
        operands: tuple[OperandPortSig, ...] = (
            _port("x", "bf16", (1, h_dim)),
            _port("expert_ids", "int32", (1, max(1, top_k)), quant_mode="none"),
        )
        outputs: tuple[OperandPortSig, ...] = (
            _port("expand_x", "bf16", (dispatch_cap, h_dim)),
            _port("dynamic_scales", "fp32", (dispatch_cap,)),
            _port("assist_info_for_combine", "int32", (max(1, top_k),), quant_mode="none"),
            _port("expert_token_nums", "int64", (local_experts,), quant_mode="none"),
            _port("ep_recv_counts", "int32", (max(1, ep),), quant_mode="none"),
            _port("tp_recv_counts", "int32", (tp,), quant_mode="none"),
            _port("expand_scales", "fp32", (dispatch_cap,)),
        )
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
        gmm1_out_dim = inter_dim if activation_kind == "gelu" else _MEGAMOE_GATED_WIDTH_FACTOR * inter_dim
        w1_layout, w2_layout = _megamoe_weight_layouts(
            soc_arch, w_dt, w2_dt, routed_act_dt,
            declared_w1_layout=declared_w1_layout, declared_w2_layout=declared_w2_layout,
        )
        w1_scale_shape: tuple[int, ...] = ()
        w2_scale_shape: tuple[int, ...] = ()
        if soc_arch == "A5":
            has_active_mask = False
            dispatch_quant_mode = (
                _MEGAMOE_DISPATCH_MODE_MXFP
                if (w_dt in _MEGAMOE_MXFP_DTYPES or w2_dt in _MEGAMOE_MXFP_DTYPES)
                else _MEGAMOE_DISPATCH_MODE_PASSTHROUGH
            )
            w1_shape: tuple[int, ...] = (local_experts, gmm1_out_dim, h_dim)
            w2_shape: tuple[int, ...] = (local_experts, h_dim, inter_dim)
            if w_dt in _MEGAMOE_MXFP_DTYPES:
                w1_scale_shape = (
                    local_experts, gmm1_out_dim,
                    (h_dim + _MEGAMOE_MX_SCALE_ALIGN - 1) // _MEGAMOE_MX_SCALE_ALIGN,
                    _MEGAMOE_MX_SCALE_TRAILING,
                )
            if w2_dt in _MEGAMOE_MXFP_DTYPES:
                w2_scale_shape = (
                    local_experts, h_dim,
                    (inter_dim + _MEGAMOE_MX_SCALE_ALIGN - 1) // _MEGAMOE_MX_SCALE_ALIGN,
                    _MEGAMOE_MX_SCALE_TRAILING,
                )
            act_params = _resolve_moe_activation_params(eff_lp, activation_kind)
        else:
            has_active_mask = True
            dispatch_quant_mode = (
                _MEGAMOE_DISPATCH_MODE_INT8
                if (w_dt in _MEGAMOE_INT_DTYPES or w2_dt in _MEGAMOE_INT_DTYPES)
                else _MEGAMOE_DISPATCH_MODE_PASSTHROUGH
            )
            w1_shape = (h_dim, gmm1_out_dim)
            w2_shape = (inter_dim, h_dim)
        w1_port = replace(
            _port("weight1", w_dt, w1_shape, layout=w1_layout, quant_mode=wq, group_shape=wg, scale_dtype=ws_dt),
            scale_shape=w1_scale_shape,
        )
        w2_port = replace(
            _port("weight2", w2_dt, w2_shape, layout=w2_layout, quant_mode=wq2, group_shape=wg2, scale_dtype=ws2_dt),
            scale_shape=w2_scale_shape,
        )
        operands = (
            _port("context", "int32", _MEGAMOE_CONTEXT_SHAPE, quant_mode="none"),
            _port("x", "bf16", (1, h_dim)),
            _port("topk_ids", "int32", (1, max(1, top_k)), quant_mode="none"),
            _port("topk_weights", "fp32", (1, max(1, top_k))),
            w1_port,
            w2_port,
        )
        outputs = (
            _port("y", "bf16", (1, h_dim)),
            _port("expert_token_nums", "int32", (local_experts,), quant_mode="none"),
        )
        motif = "moe_mc2_frame"
        act_field = activation_kind

    has_bias = (
        w_dt == "int4"
        and (abi_id == "dispatch_ffn_combine" or (abi_id == "aclnnMegaMoe" and soc_arch != "A5"))
    )
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
        activation_params=act_params,
        has_bias=has_bias,
        has_active_mask=has_active_mask,
        dispatch_quant_mode=dispatch_quant_mode,
        shared_experts_per_rank=shared_experts_per_rank,
        top_k=top_k,
        moe_expert_num=phys_total_experts,
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


def _evaluate_fused_mc2_layer(
    case: Case,
    lp: Any | None,
    *,
    lvl: int,
    soc_arch: SocArch,
    sku_id: str,
    abi_id: str,
    ep: int,
    placement: RankPlacement,
    nodes: Sequence[Any],
    activation: str,
    allow_unbound_formula: bool,
    reg: ProbedAbiRegistry,
    pr_track: StackTrack,
    track_spec: StackTrackSpec | None,
    eff_bindings: tuple[ManifestBinding, ...],
) -> MoERouteDecision:
    act_kind, is_act_and_mul = _resolve_moe_activation_strict(case, nodes, activation, lp=lp)
    if not act_kind:
        return _reject_moe(lvl, False, "moe_activation_unspecified")
    (w_dt, wq, wg, ws_dt), w2_contract = _resolve_moe_weight_contracts(case, nodes, lp=lp)
    w2_dt = w2_contract[0]
    routed_act_dt = _resolve_moe_routed_act_dtype(case, nodes, lp=lp)
    if lvl == FUSED_MC2_SHARED_LEVEL:
        lvl3_err = _check_fused_mc2_level3_shared_eligibility(case, soc_arch, placement, act_kind, nodes, lp=lp)
        if lvl3_err is not None:
            return _reject_moe(lvl, False, lvl3_err)
    if abi_id == "aclnnMegaMoe" and soc_arch == "A5":
        if ((routed_act_dt, w_dt) not in _MEGAMOE_A5_VALID_ACT_WEIGHT_PAIRS
                or (routed_act_dt, w2_dt) not in _MEGAMOE_A5_VALID_ACT_WEIGHT_PAIRS):
            return _reject_moe(
                lvl, False,
                f"rejected_dtype:megamoe_a5_act_weight_mismatch:act={routed_act_dt},w1={w_dt},w2={w2_dt}",
            )
    shared_per_rank = (
        max(1, int(getattr(lp, "num_shared_experts", 0) or 0))
        if lvl == FUSED_MC2_SHARED_LEVEL else 0
    )
    decl_w1_lay, decl_w2_lay = _resolve_moe_node_weight_layouts(case, nodes, lp=lp)
    query = _build_moe_abi_query(
        case, soc_arch=soc_arch, sku_id=sku_id, abi_id=abi_id,
        ep=ep, placement=placement, activation_kind=act_kind, is_act_and_mul=is_act_and_mul,
        shared_experts_per_rank=shared_per_rank, pr_track=pr_track,
        w_dt=w_dt, wq=wq, wg=wg, ws_dt=ws_dt, w2_contract=w2_contract,
        routed_act_dt=routed_act_dt, lp=lp,
        declared_w1_layout=decl_w1_lay, declared_w2_layout=decl_w2_lay,
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
    moe_plans = _moe_layer_plans(case)

    if lvl == 0:
        if mso and not _model_has_shared_experts(case, nodes):
            return _reject_moe(lvl, mso, "mso_requires_shared_experts")
        rid = "discrete_moe_mso" if mso else "discrete_moe_serial"
        verdict: AbiSupportVerdict | None = None
        is_shipping = False
        if ep > 1:
            check_plans = moe_plans or (None,)
            is_shipping = True
            for lp in check_plans:
                (w_dt, wq, wg, ws_dt), _ = _resolve_moe_weight_contracts(case, nodes, lp=lp)
                query = _build_moe_abi_query(
                    case, soc_arch=soc_arch, sku_id=sku_id, abi_id="aclnnMoeDistributeDispatchV2",
                    ep=ep, placement=placement, activation_kind="none", is_act_and_mul=True,
                    shared_experts_per_rank=0, pr_track=pr_track, w_dt=w_dt, wq=wq, wg=wg, ws_dt=ws_dt,
                    lp=lp,
                )
                verdict = reg.verify(query, track_spec=track_spec, manifest_bindings=eff_bindings)
                if verdict.status != "admitted_shipping":
                    is_shipping = False
                    break
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
    abi_id = "dispatch_ffn_combine" if lvl == FUSED_MC2_CSRC_LEVEL else "aclnnMegaMoe"
    if abi_id == "aclnnMegaMoe":
        placement_err = _check_megamoe_expert_placement(case, soc_arch, ep)
        if placement_err is not None:
            return _reject_moe(lvl, mso, placement_err)

    check_plans = moe_plans or (None,)
    best_decision: MoERouteDecision | None = None
    seen_keys: set[tuple[Any, ...]] = set()
    for lp in check_plans:
        lp_key = (
            None if lp is None else (
                getattr(lp, "num_experts", 0), getattr(lp, "top_k", 0),
                getattr(lp, "moe_intermediate_size", 0), getattr(lp, "expert_latent_size", 0),
                getattr(lp, "num_shared_experts", 0), getattr(lp, "shared_intermediate_size", 0),
                getattr(lp, "shared_expert_gate", False), getattr(lp, "ffn_activation", ""),
                getattr(lp, "swiglu_limit", 0.0), getattr(lp, "swiglu_shared_limit", 0.0),
                getattr(lp, "swiglu_alpha", 1.0), getattr(lp, "swiglu_beta", 0.0),
                getattr(lp, "activation_situ_beta", 1.0), getattr(lp, "activation_situ_linear_beta", None),
                getattr(lp, "index", 0) if getattr(getattr(case.model, "quant", None), "layer_quant_overrides", ()) else 0,
            )
        )
        if lp_key in seen_keys:
            continue
        seen_keys.add(lp_key)
        decision = _evaluate_fused_mc2_layer(
            case, lp, lvl=lvl, soc_arch=soc_arch, sku_id=sku_id, abi_id=abi_id,
            ep=ep, placement=placement, nodes=nodes, activation=activation,
            allow_unbound_formula=allow_unbound_formula, reg=reg, pr_track=pr_track,
            track_spec=track_spec, eff_bindings=eff_bindings,
        )
        if not decision.legal:
            return decision
        if best_decision is None or (best_decision.is_shipping_abi and not decision.is_shipping_abi):
            best_decision = decision
    assert best_decision is not None
    return best_decision


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
