"""Causal acceptance-tail placement, barrier rewiring & upstream MoE route legality.

[Role] Own structural DAG placement and MoE route legality for target-state
  optimization and export:
  - Compute the exact causal ``target_tail`` closure on the full occurrence DAG
    before barrier filtering or layer folding so ``recover_ssm_commit`` always
    executes strictly after ``accept`` and before Draft;
  - Transitively resolve chains of removed zero-duration barriers so external
    data dependencies never break;
  - Restore folded per-layer acceptance commits inside ``target_tail`` with
    ``repeat_count = 1`` so commit bytes and trace spans match unfolded runs;
  - Enforce upstream ``vllm-ascend`` MoE mutex (``enable_fused_mc2``
    perpendicular to ``multistream_overlap_shared_expert``), A5-only non-latent
    level-3 guards, and stock vs extended-ABI activation rules.
[Invariants] Every acceptance-dependent target op belongs to ``target_tail``.
  Fused MC2 and MSO Stream 1 are never enabled simultaneously. Zero model-name
  branches or cross-generation ``AIV/CCU`` labels.
[Timeouts & Locks] Linear O(|V| + |E|) DAG walks; no I/O, locks or devices.
[Side-Effect Policy] Pure functional transformations over immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any

from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.streams import MAIN_STREAM
from mindie_sim.schema.case import Case
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec

BARRIER_EPS_US = 1e-3
#: Legal ``enable_fused_mc2`` level constants (matching upstream ``vllm-ascend``).
FUSED_MC2_ROUTED_LEVEL = 2
FUSED_MC2_SHARED_LEVEL = 3
VALID_FUSED_MC2_LEVELS = frozenset({0, 1, FUSED_MC2_ROUTED_LEVEL, FUSED_MC2_SHARED_LEVEL})
#: Stock shipping CANN activations supported by stock MegaMoE / dispatch_ffn_combine.
SHIPPING_MOE_ACTIVATIONS = frozenset({"", "swiglu", "silu", "gelu"})
#: Extended non-stock activations requiring ABI extension in Modeling mode.
EXTENDED_MOE_ACTIVATIONS = frozenset({"situglu", "situ", "swigluoai", "geglu", "gelu_tanh"})
#: Hardware ISA / arch-family identifiers.
ARCH35_TOKENS = frozenset({"A5", "ARCH35", "DAV_3510", "NPU_ARCH_3510"})
ARCH22_TOKENS = frozenset({"A3", "ARCH22"})
ARCH20_TOKENS = frozenset({"A2", "ARCH20"})
#: Families that represent acceptance anchors and post-accept commits.
ACCEPTANCE_ANCHOR_FAMILIES = frozenset({"acceptance", "sampling"})
TAIL_COMMIT_FAMILIES = frozenset({"token_commit", "recover_ssm_commit"})
#: Persistent whole-die fused MoE kernel markers.
PERSISTENT_MOE_FAMILIES = frozenset({"grouped_moe", "mc2", "fused_moe_mc2"})
_LAYER_TOKEN_RE = re.compile(r"(\.l)(\d+)(\.)")


@dataclass(frozen=True, slots=True)
class MoERouteDecision:
    """Resolved legal MoE route selected by Die Engine comparison."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    return raw_arch or "Arch20"


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


def _model_moe_activation(case: Case, ops: Sequence[Any] = ()) -> str:
    for op in ops:
        spec = getattr(op, "spec", op)
        if spec is None:
            continue
        meta = dict(getattr(spec, "metadata", ()) or ())
        for key in ("activation", "ffn_activation", "activation_kind"):
            val = str(meta.get(key) or "").strip().lower()
            if val:
                return val
        fam = str(getattr(spec, "family", "") or "").strip().lower()
        if fam in EXTENDED_MOE_ACTIVATIONS:
            return fam
    for lp in getattr(case.model, "layer_plans", ()) or ():
        if str(getattr(lp, "ffn", "") or "").lower() == "moe":
            act = str(getattr(lp, "ffn_activation", "") or "").strip().lower()
            if act:
                return act
    return str(getattr(case.model, "ffn_activation", "") or "swiglu").strip().lower()


def _model_has_shared_experts(case: Case, ops: Sequence[Any] = ()) -> bool:
    for lp in getattr(case.model, "layer_plans", ()) or ():
        if int(getattr(lp, "num_shared_experts", 0) or 0) > 0:
            return True
    return any(is_shared_expert_op_id(str(getattr(op, "op_id", "") or "")) for op in ops)


def _reject_moe(lvl: int, mso: bool, reason: str) -> MoERouteDecision:
    return MoERouteDecision(
        enable_fused_mc2=lvl, multistream_overlap_shared_expert=mso,
        route_id="illegal", abi_id="", needs_new_kernel=False,
        is_shipping_abi=False, reason=reason, legal=False,
    )


def moe_route_legality(
    case: Case,
    *,
    enable_fused_mc2: int = 0,
    multistream_overlap_shared_expert: bool = False,
    nodes: Sequence[Any] = (),
    activation: str = "",
    allow_unbound_formula: bool = False,
) -> MoERouteDecision:
    """Validate MoE ``(enable_fused_mc2, MSO)`` legality against upstream ``vllm-ascend`` rules."""
    lvl = int(enable_fused_mc2)
    mso = bool(multistream_overlap_shared_expert)
    if lvl not in VALID_FUSED_MC2_LEVELS:
        return _reject_moe(lvl, mso, f"invalid_fused_mc2_level:{lvl}")
    # Upstream ascend_config.py:936-941 hard mutex: enable_fused_mc2 perpendicular to MSO.
    if lvl > 0 and mso:
        return _reject_moe(lvl, mso, "fused_mc2_conflicts_with_multistream_overlap_shared_expert")
    if lvl == 0:
        if mso and not _model_has_shared_experts(case, nodes):
            return _reject_moe(lvl, mso, "mso_requires_shared_experts")
        rid = "discrete_moe_mso" if mso else "discrete_moe_serial"
        return MoERouteDecision(
            enable_fused_mc2=0, multistream_overlap_shared_expert=mso,
            route_id=rid, abi_id="discrete_mc2", needs_new_kernel=False,
            is_shipping_abi=True, reason=rid, legal=True,
        )

    arch = resolve_arch_family_tag(case)
    act = (activation or _model_moe_activation(case, nodes)).strip().lower()
    is_shipping_act = act in SHIPPING_MOE_ACTIVATIONS
    if max(1, int(case.runtime.parallel.ep)) <= 1:
        return _reject_moe(lvl, mso, "fused_mc2_requires_ep_gt_1")

    if lvl == 1:
        if arch != "Arch22":
            return _reject_moe(lvl, mso, f"dispatch_ffn_combine_requires_arch22:{arch}")
        if not is_shipping_act and not allow_unbound_formula:
            return _reject_moe(lvl, mso, f"dispatch_ffn_combine_unsupported_activation:{act}")
        return MoERouteDecision(
            enable_fused_mc2=1, multistream_overlap_shared_expert=False,
            route_id="fused_mc2_level1", abi_id="dispatch_ffn_combine",
            needs_new_kernel=(not is_shipping_act), is_shipping_abi=is_shipping_act,
            reason="dispatch_ffn_combine_shipping" if is_shipping_act else f"dispatch_ffn_combine_extended_abi:{act}",
            legal=True,
        )

    if lvl == FUSED_MC2_ROUTED_LEVEL:
        if arch not in {"Arch35", "Arch22"}:
            return _reject_moe(lvl, mso, f"megamoe_unsupported_on_arch:{arch}")
        if not is_shipping_act and not allow_unbound_formula:
            return _reject_moe(lvl, mso, f"megamoe_activation_requires_abi_extension:{act}")
        return MoERouteDecision(
            enable_fused_mc2=FUSED_MC2_ROUTED_LEVEL, multistream_overlap_shared_expert=False,
            route_id="fused_mc2_level2", abi_id="aclnnMegaMoe",
            needs_new_kernel=(not is_shipping_act), is_shipping_abi=is_shipping_act,
            reason="megamoe_routed_shipping" if is_shipping_act else f"megamoe_extended_abi:{act}",
            legal=True,
        )

    # lvl == FUSED_MC2_SHARED_LEVEL (3): A5-only non-latent MoE guard (ascend_config.py:1118-1145).
    if arch != "Arch35":
        return _reject_moe(lvl, mso, f"fused_mc2_level3_requires_arch35:{arch}")
    if is_latent_moe_model(case):
        return _reject_moe(lvl, mso, "fused_mc2_level3_forbids_latent_moe")
    if not _model_has_shared_experts(case, nodes):
        return _reject_moe(lvl, mso, "fused_mc2_level3_requires_shared_experts")
    if not is_shipping_act and not allow_unbound_formula:
        return _reject_moe(lvl, mso, f"megamoe_level3_activation_requires_abi_extension:{act}")
    return MoERouteDecision(
        enable_fused_mc2=FUSED_MC2_SHARED_LEVEL, multistream_overlap_shared_expert=False,
        route_id="fused_mc2_level3", abi_id="aclnnMegaMoe",
        needs_new_kernel=(not is_shipping_act), is_shipping_abi=is_shipping_act,
        reason="megamoe_shared_fusion_shipping" if is_shipping_act else f"megamoe_level3_extended_abi:{act}",
        legal=True,
    )


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


def _replace_layer_in_op_id(op_id: str, orig_layer: int) -> str:
    if _LAYER_TOKEN_RE.search(op_id):
        return _LAYER_TOKEN_RE.sub(rf"\g<1>{orig_layer}\g<3>", op_id, count=1)
    return f"{op_id}.l{orig_layer}"


def restore_folded_tail_commits(
    records: Sequence[Mapping[str, Any]],
    *,
    tail_closure: frozenset[OpOccurrence],
    layer_indices_by_rep_layer: Mapping[int, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], frozenset[OpOccurrence]]:
    """Expand folded representative acceptance commits into distinct per-layer tail commits."""
    if not layer_indices_by_rep_layer or not tail_closure:
        return [dict(r) for r in records], tail_closure

    expanded: list[dict[str, Any]] = []
    new_closure = set(tail_closure)
    restored_by_orig_occ: dict[OpOccurrence, tuple[OpOccurrence, ...]] = {}
    clone_occs: set[OpOccurrence] = set()

    for rec in records:
        occ: OpOccurrence = rec["occurrence"]
        if occ not in tail_closure or not is_acceptance_commit_record(rec):
            expanded.append(dict(rec))
            continue
        spec: OpSpec | None = rec.get("spec")
        rep_layer = spec.layer if (spec is not None and spec.layer is not None and spec.layer >= 0) else rec.get("row_layer")
        if rep_layer is None:
            expanded.append(dict(rec))
            continue
        full_layers = layer_indices_by_rep_layer.get(int(rep_layer))
        if not full_layers or len(full_layers) <= 1:
            expanded.append(dict(rec))
            continue
        clones: list[OpOccurrence] = []
        base_start = float(rec["start_us"])
        dur = max(0.0, float(rec["end_us"]) - base_start)
        prev_clone_occ: OpOccurrence | None = None
        for idx, orig_lyr in enumerate(full_layers):
            new_oid = _replace_layer_in_op_id(str(rec["op_id"]), int(orig_lyr))
            new_occ = replace(occ, op_id=new_oid)
            new_spec = spec
            if spec is not None:
                meta = dict(spec.metadata) if spec.metadata else {}
                meta["state_owner"] = f"layer{orig_lyr}"
                meta["original_layer"] = int(orig_lyr)
                new_spec = replace(
                    spec,
                    op_id=new_oid,
                    layer=int(orig_lyr),
                    metadata=tuple(sorted(meta.items(), key=lambda kv: kv[0])),
                )
            preds = list(rec.get("predecessors") or ())
            if prev_clone_occ is not None:
                preds.append(prev_clone_occ)
            clone_rec = dict(rec)
            clone_rec["occurrence"] = new_occ
            clone_rec["op_id"] = new_oid
            clone_rec["spec"] = new_spec
            clone_rec["row_layer"] = int(orig_lyr)
            clone_rec["start_us"] = base_start + float(idx) * dur
            clone_rec["end_us"] = base_start + float(idx + 1) * dur
            clone_rec["predecessors"] = tuple(dict.fromkeys(p for p in preds if p != new_occ))
            expanded.append(clone_rec)
            clones.append(new_occ)
            clone_occs.add(new_occ)
            new_closure.add(new_occ)
            prev_clone_occ = new_occ
        restored_by_orig_occ[occ] = tuple(clones)

    if not restored_by_orig_occ:
        return expanded, frozenset(new_closure)

    for rec in expanded:
        occ = rec["occurrence"]
        if occ in clone_occs:
            continue
        preds = rec.get("predecessors") or ()
        if not any(p in restored_by_orig_occ for p in preds):
            continue
        rewired: list[OpOccurrence] = []
        for pred in preds:
            if pred in restored_by_orig_occ:
                rewired.append(restored_by_orig_occ[pred][-1])
            else:
                rewired.append(pred)
        rec["predecessors"] = tuple(dict.fromkeys(p for p in rewired if p != occ))
    return expanded, frozenset(new_closure)


def enforce_moe_mutex_on_nodes(
    nodes: Sequence[Any],
    case: Case,
    *,
    allow_unbound_formula: bool = False,
) -> tuple[list[Any], MoERouteDecision | None]:
    """Enforce ``enable_fused_mc2 perpendicular to MSO`` and arch/activation legality."""
    has_fused_moe = any(is_persistent_whole_die_node(n) for n in nodes)
    if not has_fused_moe:
        has_mso_stream = any(
            int(n.stream_index) == 1 and is_shared_expert_op_id(n.op_id) for n in nodes
        )
        if not any("expert" in n.op_id.lower() or str(n.family).lower() in {"grouped_gemm", "dispatch", "combine"} for n in nodes):
            return list(nodes), None
        return list(nodes), MoERouteDecision(
            enable_fused_mc2=0,
            multistream_overlap_shared_expert=has_mso_stream,
            route_id="discrete_moe_mso" if has_mso_stream else "discrete_moe_serial",
            abi_id="discrete_mc2",
            needs_new_kernel=False,
            is_shipping_abi=True,
            reason="discrete_moe_mso" if has_mso_stream else "discrete_moe_serial",
        )

    arch = resolve_arch_family_tag(case)
    act = _model_moe_activation(case, nodes)
    is_shipping_act = act in SHIPPING_MOE_ACTIVATIONS
    out: list[Any] = []
    chosen_abi = "aclnnMegaMoe" if arch == "Arch35" else "dispatch_ffn_combine"
    needs_new_any = False

    for node in nodes:
        if is_persistent_whole_die_node(node):
            kern = str(node.ascend_kernel or "")
            if arch == "Arch35" and "dispatch_ffn_combine" in kern.lower():
                kern = "aclnnMegaMoe"
            chosen_abi = kern or chosen_abi
            upd: dict[str, Any] = {"persistent_whole_die": True, "stream": MAIN_STREAM, "stream_index": 0}
            if not is_shipping_act:
                needs_new_any = True
                if allow_unbound_formula:
                    upd["needs_new_kernel"] = True
                    upd["is_shipping_abi"] = False
                    upd["binding_state"] = "unbound_formula_projection"
            if kern != node.ascend_kernel:
                upd["ascend_kernel"] = kern
            out.append(replace(node, **{k: v for k, v in upd.items() if hasattr(node, k)}))
        elif int(node.stream_index) == 1 and is_shared_expert_op_id(node.op_id):
            out.append(replace(node, stream=MAIN_STREAM, stream_index=0))
        else:
            out.append(node)

    lvl = FUSED_MC2_ROUTED_LEVEL if (arch == "Arch35" or "megamoe" in chosen_abi.lower()) else 1
    return out, MoERouteDecision(
        enable_fused_mc2=lvl,
        multistream_overlap_shared_expert=False,
        route_id=f"fused_mc2_level{lvl}",
        abi_id=chosen_abi,
        needs_new_kernel=needs_new_any,
        is_shipping_abi=not needs_new_any,
        reason="fused_mc2_mutex_enforced",
    )


def build_discrete_mso_candidate_nodes(nodes: Sequence[Any]) -> list[Any] | None:
    """Assign independent separate Shared Expert ops to MSO Stream 1 (discrete MoE only)."""
    if any(is_persistent_whole_die_node(n) for n in nodes):
        return None
    shared_indices = [
        i for i, n in enumerate(nodes)
        if is_shared_expert_op_id(n.op_id)
        and str(n.family).lower() not in {"allreduce", "reduce_scatter", "allgather"}
    ]
    if not shared_indices:
        return None
    routed_occs = {
        n.occurrence for n in nodes
        if not is_shared_expert_op_id(n.op_id)
        and (
            str(n.family).lower() in {"dispatch", "combine", "grouped_gemm", "alltoall", "alltoallv"}
            or any(tok in n.op_id.lower() for tok in (".dispatch", ".combine", ".expert_", ".router"))
        )
    }
    if not routed_occs:
        return None
    out = list(nodes)
    moved = 0
    for idx in shared_indices:
        node = out[idx]
        if any(p in routed_occs for p in node.predecessors):
            continue
        if int(node.stream_index) != 1:
            out[idx] = replace(node, stream="aux", stream_index=1)
        moved += 1
    return out if moved > 0 else None


__all__ = [
    "ACCEPTANCE_ANCHOR_FAMILIES",
    "EXTENDED_MOE_ACTIVATIONS",
    "FUSED_MC2_ROUTED_LEVEL",
    "FUSED_MC2_SHARED_LEVEL",
    "PERSISTENT_MOE_FAMILIES",
    "SHIPPING_MOE_ACTIVATIONS",
    "TAIL_COMMIT_FAMILIES",
    "MoERouteDecision",
    "build_discrete_mso_candidate_nodes",
    "compute_acceptance_tail_closure",
    "enforce_moe_mutex_on_nodes",
    "is_a5_hardware",
    "is_acceptance_commit_record",
    "is_draft_record",
    "is_latent_moe_model",
    "is_persistent_whole_die_node",
    "is_shared_expert_op_id",
    "moe_route_legality",
    "resolve_arch_family_tag",
    "resolve_barrier_predecessors",
    "restore_folded_tail_commits",
]
