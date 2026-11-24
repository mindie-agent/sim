"""Causal acceptance-tail placement and barrier rewiring for target-state DAGs.

[Role] Own pure structural DAG placement helpers for target-state optimization
  and export:
  - Compute the exact causal ``target_tail`` closure on the full occurrence DAG
    before barrier filtering so real post-accept ``recover_ssm_commit`` Engine
    occurrences always execute strictly after ``accept`` and before Draft;
  - Transitively resolve chains of removed zero-duration barriers so external
    data dependencies never break.
[Invariants] Every acceptance-dependent target op belongs to ``target_tail``.
  Zero shadow schedulers, zero unvalidated activation whitelists, and zero
  silent architecture fallbacks.
[Timeouts & Locks] Linear O(|V| + |E|) DAG walks; no I/O, locks or devices.
[Side-Effect Policy] Pure functional transformations over immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die

BARRIER_EPS_US = 1e-3
#: Hardware ISA / arch-family identifiers.
ARCH35_TOKENS = frozenset({"A5", "ARCH35", "DAV_3510", "NPU_ARCH_3510"})
ARCH22_TOKENS = frozenset({"A3", "ARCH22"})
ARCH20_TOKENS = frozenset({"A2", "ARCH20"})
#: Families that represent acceptance anchors and post-accept commits.
ACCEPTANCE_ANCHOR_FAMILIES = frozenset({"acceptance", "sampling"})
TAIL_COMMIT_FAMILIES = frozenset({"token_commit", "recover_ssm_commit"})
#: Persistent whole-die fused MoE kernel markers.
PERSISTENT_MOE_FAMILIES = frozenset({"grouped_moe", "mc2", "fused_moe_mc2"})


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


def moe_route_legality(
    case: Case,
    *,
    enable_fused_mc2: int = 0,
    multistream_overlap_shared_expert: bool = False,
    nodes: Sequence[Any] = (),
    activation: str = "",
    allow_unbound_formula: bool = False,
) -> MoERouteDecision:
    """Validate MoE ``(enable_fused_mc2, MSO)`` legality via canonical ProbedAbiRegistry."""
    del case, enable_fused_mc2, multistream_overlap_shared_expert, nodes, activation, allow_unbound_formula
    raise NotImplementedError(
        "LEGACY_PURGED: delegate legality to mindie_sim.backends.abi_registry.ProbedAbiRegistry"
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
