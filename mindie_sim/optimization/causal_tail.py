"""Causal acceptance-tail placement, barrier rewiring & upstream MoE route legality.

[Role] Own structural DAG placement and MoE route legality for target-state
  optimization and export:
  - Compute the exact causal ``target_tail`` closure on the full occurrence DAG
    before barrier filtering so real post-accept ``recover_ssm_commit`` Engine
    occurrences always execute strictly after ``accept`` and before Draft;
  - Transitively resolve chains of removed zero-duration barriers so external
    data dependencies never break;
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

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.prediction.collectives import Topology
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.runtime.exposed_observer import (
    ServiceGrantObserver,
    attribute_engine_exposure,
)
from mindie_sim.prediction.streams import (
    MAIN_STREAM,
    StreamCoreBudget,
    StreamLane,
    StreamOwners,
    StreamPlan,
    bind_stream_owners,
)
from mindie_sim.primitives.loader import resolve_sim_rate_overlay
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.hardware import Device

BARRIER_EPS_US = 1e-3
US_PER_MS = 1000.0
MS_PER_SECOND = 1000.0
NS_PER_US = 1000.0
BYTES_PER_GB = 1_000_000_000.0
PERCENT_SCALE = 100.0
TIME_EPS_US = 1e-6
EXPOSED_EPS_US = 1e-3
#: Admission capacity for concurrent non-persistent streams on one physical die.
DIE_EXEC_CONCURRENCY_SLOTS = 16
#: Minimum relative makespan improvement (0.5%) to adopt a disjoint core split.
DISJOINT_SPLIT_MIN_GAIN_RATIO = 0.005
#: Candidate main-lane AIC split fractions (numerator, denominator) evaluated for disjoint budgets.
DISJOINT_SPLIT_RATIOS: tuple[tuple[int, int], ...] = (
    (3, 4),
    (5, 8),
    (1, 2),
)
DEFAULT_CUBE_TILE_M = 128
DEFAULT_CUBE_TILE_N = 128
MATMUL_3D_RANK = 3
DEFAULT_VEC_TILE_ELEMS = 4096
HBM_RW_ASYMMETRY_RATIO = 1.5
ZERO_AIC_COMM_FAMILIES = frozenset({"dispatch", "combine", "alltoall", "alltoallv", "mc2_transport", "mc2_protocol"})
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


@dataclass(frozen=True, slots=True)
class DieScheduleResult:
    """Result of executing a target graph on the real Die + Topology Engine."""

    nodes: tuple[Any, ...]
    archetype_spans_us: Mapping[str, float]
    step_start_us: float
    step_end_us: float
    step_span_us: float
    stream_core_budget: Mapping[tuple[int, int, int], Mapping[str, Any]]
    stream_core_budget_summary: str
    max_same_stream_overlap_us: float


def _canonical_stream_lane_name(stream: str, stream_index: int) -> str:
    idx = int(stream_index)
    if idx == 0:
        return MAIN_STREAM
    raw = str(stream or "").strip()
    if raw and raw != MAIN_STREAM and raw.isascii() and not any(ch.isspace() for ch in raw):
        return raw
    return f"stream_{idx}"


def format_stream_core_budget_receipt(owners_by_lane: Mapping[str, StreamOwners]) -> str:
    """Format resolved integer core-budget receipt (e.g. ``disjoint (main: 24 AIC / 48 AIV, aux: 8 AIC / 16 AIV)``)."""
    if not owners_by_lane or not any(o.isolation == "disjoint" for o in owners_by_lane.values()):
        return ""
    parts = [f"{lid}: {int(o.aic_limit)} AIC / {int(o.aiv_limit)} AIV" for lid, o in owners_by_lane.items()]
    return f"disjoint ({', '.join(parts)})"


def _effective_die_supply(
    die: Device, links: Sequence[Any], channels: Sequence[Any], primitive_table: Any = None,
) -> tuple[DatapathCapabilitySpec, dict[str, float], dict[str, float], dict[str, tuple[str, float]], float]:
    ub_edges = ub_peer_edges_for_rank(die.rank, links, channels)
    datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
    rates = dict(die.rates)
    peaks = dict(die.compute_peaks)
    service_modes = dict(die.service_modes)
    if primitive_table is not None:
        overlay = resolve_sim_rate_overlay(primitive_table, datapath, sku=str(getattr(die, "sku", "") or ""))
        for role, val in overlay.rates:
            if val > 0.0:
                rates[role] = min(val, rates[role]) if rates.get(role, 0.0) > 0.0 else val
        for dt, pk in overlay.compute_peaks:
            if pk > 0.0:
                peaks[dt] = pk
        if overlay.hbm_read_bytes_per_s is not None:
            service_modes["mte2_aiv_dram"] = ("mte2_aiv" if "mte2_aiv" in rates else "mte2", overlay.hbm_read_bytes_per_s)
        if overlay.hbm_write_bytes_per_s is not None:
            service_modes["hbm_write_dram"] = ("hbm", overlay.hbm_write_bytes_per_s)
        if overlay.hbm_rw11_bytes_per_s is not None:
            service_modes["hbm_rw11_dram"] = ("hbm", overlay.hbm_rw11_bytes_per_s)
        if overlay.hbm_rw21_bytes_per_s is not None:
            service_modes["hbm_rw21_dram"] = ("hbm", overlay.hbm_rw21_bytes_per_s)
    link_bps = float(rates.get("link") or 0.0)
    if links:
        def _lbps(lnk: Any) -> float:
            b_ms = float(getattr(lnk, "capacity_bytes_per_ms", 0.0) or 0.0)
            return b_ms * MS_PER_SECOND if b_ms > 0.0 else float(getattr(lnk, "capacity_gbps", 0.0) or 0.0) * BYTES_PER_GB

        best_bps = max((_lbps(lnk) for lnk in links), default=0.0)
        if best_bps > 0.0:
            link_bps = best_bps
    return datapath, rates, peaks, service_modes, link_bps


def _cube_peak_for_dtype(peaks: Mapping[str, float], dtype: str) -> float:
    tok = str(dtype or "bf16").strip().lower()
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"}
    for key in (tok, aliases.get(tok, tok), "bf16", "fp16"):
        val = float(peaks.get(key, 0.0) or 0.0)
        if val > 0.0:
            return val
    return max((float(v) for v in peaks.values() if float(v) > 0.0), default=0.0)


def _node_cube_peak_ms(node: Any, peaks: Mapping[str, float]) -> float:
    spec = getattr(node, "spec", None)
    entries = getattr(spec, "matrix_flops", ()) if spec is not None else ()
    if entries:
        total = sum(
            MS_PER_SECOND * float(fl) / pk
            for dt, fl in entries
            if float(fl) > 0.0 and (pk := _cube_peak_for_dtype(peaks, str(dt))) > 0.0
        )
        if total > 0.0:
            return total
    flops = float(getattr(node, "flops", 0.0) or 0.0)
    if flops <= 0.0:
        return 0.0
    pk = _cube_peak_for_dtype(peaks, str(getattr(node, "dtype", "bf16") or "bf16"))
    return (MS_PER_SECOND * flops / pk) if pk > 0.0 else 0.0


def _effective_hbm_rate_bps(
    hbm_read: float, hbm_write: float, hbm_peak: float, service_modes: Mapping[str, tuple[str, float]],
) -> float:
    def _mode_rate(name: str) -> float | None:
        entry = service_modes.get(name)
        return float(entry[1]) if (entry is not None and float(entry[1]) > 0.0) else None

    if hbm_read <= 0.0 or hbm_write > HBM_RW_ASYMMETRY_RATIO * hbm_read:
        rate = _mode_rate("hbm_write_dram") or _mode_rate("mte2_aiv_dram")
    elif hbm_write <= 0.0:
        rate = _mode_rate("mte2_aiv_dram")
    elif hbm_read > HBM_RW_ASYMMETRY_RATIO * hbm_write:
        rate = _mode_rate("hbm_rw21_dram") or _mode_rate("mte2_aiv_dram")
    else:
        rate = _mode_rate("hbm_rw11_dram") or _mode_rate("hbm_rw21_dram") or _mode_rate("mte2_aiv_dram")
    if rate is None or rate <= 0.0:
        return hbm_peak
    return min(rate, hbm_peak) if hbm_peak > 0.0 else rate


def _estimate_core_tiles(node: Any, *, kind: str, full_cores: int) -> int:
    spec = getattr(node, "spec", None)
    meta = dict(getattr(spec, "metadata", ()) or ()) if spec is not None else {}
    explicit = meta.get("matrix_tiles" if kind == "aic" else "vector_tiles") or meta.get("tiles")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    shape = tuple(int(d) for d in (getattr(node, "shape", ()) or ()) if int(d) > 0)
    if kind == "aic":
        if len(shape) >= MATMUL_3D_RANK:
            m_t = max(1, math.ceil(shape[-MATMUL_3D_RANK] / DEFAULT_CUBE_TILE_M))
            n_t = max(1, math.ceil(shape[-2] / DEFAULT_CUBE_TILE_N))
            batch = math.prod(shape[:-MATMUL_3D_RANK]) if len(shape) > MATMUL_3D_RANK else 1
            return max(1, int(batch * m_t * n_t))
        if len(shape) == 2:
            return max(1, math.ceil(shape[0] / DEFAULT_CUBE_TILE_M) * math.ceil(shape[1] / DEFAULT_CUBE_TILE_N))
        return max(1, int(full_cores))
    elems = math.prod(shape) if shape else 0
    return max(1, math.ceil(elems / DEFAULT_VEC_TILE_ELEMS)) if elems > 0 else max(1, int(full_cores))


def _wave_scale_factor(tiles: int, full_cores: int, eff_cores: int) -> float:
    if eff_cores <= 0 or full_cores <= 0 or eff_cores >= full_cores:
        return 1.0
    return float(max(1, math.ceil(tiles / eff_cores))) / float(max(1, math.ceil(tiles / full_cores)))


def _node_physical_demand_and_placement(
    node: Any, *, die_tag: str, lane_id: str, datapath: DatapathCapabilitySpec,
    rates: Mapping[str, float], peaks: Mapping[str, float],
    service_modes: Mapping[str, tuple[str, float]], link_bps: float,
    owners: StreamOwners | None, has_topology_plan: bool, launch_floor_us: float,
) -> tuple[dict[str, float], dict[str, tuple[int, ...]], float, bool]:
    aic_full, aiv_full = max(1, int(datapath.aic_count)), max(1, int(datapath.aiv_count))
    is_disjoint = owners is not None and owners.isolation == "disjoint"
    aic_eff = int(owners.aic_limit) if is_disjoint else aic_full
    aiv_eff = int(owners.aiv_limit) if is_disjoint else aiv_full
    sfx = f":{lane_id}" if is_disjoint else ""

    cube_ms = _node_cube_peak_ms(node, peaks)
    vec_ops = max(0.0, float(getattr(node, "vector_ops", 0.0) or 0.0))
    hbm_r = max(0.0, float(getattr(node, "hbm_read_bytes", 0.0) or 0.0))
    hbm_w = max(0.0, float(getattr(node, "hbm_write_bytes", 0.0) or 0.0))
    hbm_tot = max(0.0, float(getattr(node, "hbm_total_bytes", 0.0) or 0.0))
    mte2_b = max(0.0, float(getattr(node, "mte2_bytes", 0.0) or 0.0))
    mte3_b = max(0.0, float(getattr(node, "mte3_bytes", 0.0) or 0.0))
    fixpipe_b = max(0.0, float(getattr(node, "fixpipe_bytes", 0.0) or 0.0))
    link_b = max(0.0, float(getattr(node, "link_bytes", 0.0) or 0.0))
    if str(getattr(node, "family", "") or "").strip().lower() in ZERO_AIC_COMM_FAMILIES:
        cube_ms = 0.0
    if cube_ms > 0.0 and aic_eff <= 0:
        raise ModelingError(f"stream_zero_aic_budget:{node.op_id}")
    if vec_ops > 0.0 and aiv_eff <= 0:
        raise ModelingError(f"stream_zero_aiv_budget:{node.op_id}")

    demand: dict[str, float] = {}
    solo_ms: dict[str, float] = {}
    aic_res, aiv_res = [], []
    pure_cube = cube_ms > 0.0 and vec_ops <= 0.0

    if cube_ms > 0.0:
        rname = f"matrix@{die_tag}{sfx}"
        demand[rname], solo_ms[rname] = cube_ms, cube_ms
        aic_res.append(rname)
    vec_rate_s = float(rates.get("vector") or 0.0)
    if vec_ops > 0.0 and vec_rate_s > 0.0:
        rname = f"vector@{die_tag}{sfx}"
        demand[rname], solo_ms[rname] = vec_ops, (vec_ops / vec_rate_s) * MS_PER_SECOND
        aiv_res.append(rname)
    hbm_peak = float(rates.get("hbm") or max(datapath.hbm_read_bytes_per_s, datapath.hbm_write_bytes_per_s) or 0.0)
    if hbm_tot > 0.0 and hbm_peak > 0.0:
        rname = f"memory@{die_tag}"
        eff_hbm = _effective_hbm_rate_bps(hbm_r, hbm_w, hbm_peak, service_modes)
        scaled = hbm_tot * (hbm_peak / eff_hbm) if eff_hbm > 0.0 else hbm_tot
        demand[rname], solo_ms[rname] = scaled, (scaled / hbm_peak) * MS_PER_SECOND
    if mte2_b > 0.0:
        use_aic_m2 = pure_cube or (is_disjoint and aiv_eff <= 0 and aic_eff > 0)
        m2_key = "mte2_aic" if (use_aic_m2 and "mte2_aic" in rates) else ("mte2_aiv" if (not use_aic_m2 and "mte2_aiv" in rates) else "mte2")
        m2_r = float(rates.get(m2_key) or hbm_peak or 0.0)
        if m2_r > 0.0:
            m2_sfx = sfx if (is_disjoint and (aic_eff > 0 if use_aic_m2 else aiv_eff > 0)) else ""
            rname = f"{m2_key}@{die_tag}{m2_sfx}"
            demand[rname], solo_ms[rname] = mte2_b, (mte2_b / m2_r) * MS_PER_SECOND
            (aic_res if use_aic_m2 else aiv_res).append(rname)
    if mte3_b > 0.0:
        m3_r = float(rates.get("mte3") or hbm_peak or 0.0)
        if m3_r > 0.0:
            use_aic_m3 = pure_cube or (is_disjoint and aiv_eff <= 0 and aic_eff > 0)
            m3_key = "mte3_aic" if (is_disjoint and use_aic_m3 and aic_eff > 0) else "mte3"
            m3_sfx = sfx if (is_disjoint and (aic_eff > 0 if use_aic_m3 else aiv_eff > 0)) else ""
            rname = f"{m3_key}@{die_tag}{m3_sfx}"
            demand[rname], solo_ms[rname] = mte3_b, (mte3_b / m3_r) * MS_PER_SECOND
            (aic_res if use_aic_m3 else aiv_res).append(rname)
    if fixpipe_b > 0.0:
        fp_r = float(rates.get("fixpipe") or hbm_peak or 0.0)
        if fp_r > 0.0:
            fp_sfx = sfx if (is_disjoint and aic_eff > 0) else ""
            rname = f"fixpipe@{die_tag}{fp_sfx}"
            demand[rname], solo_ms[rname] = fixpipe_b, (fixpipe_b / fp_r) * MS_PER_SECOND
            aic_res.append(rname)
    if link_b > 0.0 and not has_topology_plan and link_bps > 0.0:
        rname = f"link@{die_tag}"
        demand[rname], solo_ms[rname] = link_b, (link_b / link_bps) * MS_PER_SECOND

    base_dur_ms = max(0.0, float(getattr(node, "dur_us", 0.0) or 0.0)) / US_PER_MS
    l_star_ms = max(0.0, float(getattr(node, "l_star_lean_us", 0.0) or 0.0)) / US_PER_MS
    floor_ms = max(0.0, float(launch_floor_us)) / US_PER_MS
    is_overlapped = bool(getattr(node, "is_overlapped", False))

    if is_disjoint:
        for res_list, kind, full_n, eff_n, core_pref in (
            (aic_res, "aic", aic_full, aic_eff, "matrix@"),
            (aiv_res, "aiv", aiv_full, aiv_eff, "vector@"),
        ):
            if eff_n > 0 and res_list:
                wave = _wave_scale_factor(_estimate_core_tiles(node, kind=kind, full_cores=full_n), full_n, eff_n)
                frac = float(eff_n) / float(full_n)
                for rname in res_list:
                    if rname.startswith(core_pref):
                        demand[rname] *= wave * frac
                        solo_ms[rname] *= wave
                    elif sfx and rname.endswith(sfx):
                        solo_ms[rname] /= max(frac, TIME_EPS_US)

    max_pipe_ms = max(solo_ms.values(), default=0.0)
    eff_floor_ms = min(floor_ms, base_dur_ms) if base_dur_ms > 0.0 else 0.0
    if has_topology_plan:
        latency_ms = 0.0
    elif is_disjoint or is_overlapped:
        if max_pipe_ms > 0.0 or l_star_ms > 0.0:
            latency_ms = max(eff_floor_ms, l_star_ms, max_pipe_ms)
        else:
            latency_ms = base_dur_ms
    else:
        latency_ms = max(l_star_ms, base_dur_ms, max_pipe_ms)
    return demand, {}, latency_ms, bool(cube_ms > 0.0 or vec_ops > 0.0 or is_persistent_whole_die_node(node))


def _die_for_occurrence(case: Case, occ: OpOccurrence) -> Device:
    dies = case.machine.dies
    if not dies:
        raise ModelingError("target_schedule_requires_at_least_one_die")
    if occ.rank is not None:
        for die in dies:
            if int(die.rank) == int(occ.rank):
                return die
    width = max(1, int(case.runtime.parallel.tp) * int(case.runtime.parallel.pcp))
    idx = int(occ.stage) * width
    return dies[idx] if 0 <= idx < len(dies) else dies[0]


def _build_die_resources_and_pools(
    die_coords: Sequence[tuple[int, int, int]],
    supply_by_coord: Mapping[tuple[int, int, int], tuple[DatapathCapabilitySpec, dict[str, float], dict[str, float], dict[str, tuple[str, float]], float]],
    stream_indices_by_coord: Mapping[tuple[int, int, int], set[int]],
    topology: Topology | None,
    collective_ranks: set[int],
    *,
    owners_by_coord: Mapping[tuple[int, int, int], Mapping[str, StreamOwners]] | None = None,
) -> tuple[list[Resource], dict[str, float], dict[str, int]]:
    resources: list[Resource] = []
    pools: dict[str, float] = {}
    pool_units: dict[str, int] = {}
    seen_res: set[str] = set()

    def _add(res: Resource) -> None:
        if res.name not in seen_res:
            seen_res.add(res.name)
            resources.append(res)

    for coord in die_coords:
        die_tag = f"r{coord[0]}:s{coord[1]}:d{coord[2]}"
        datapath, rates, _peaks, _smodes, link_bps = supply_by_coord[coord]
        aic_full, aiv_full = max(1, int(datapath.aic_count)), max(1, int(datapath.aiv_count))
        for s_idx in sorted(stream_indices_by_coord[coord] | {0, 1, 2}):
            _add(Resource(f"stream_{s_idx}@{die_tag}", 1.0, "ms", "fifo", 1))
        _add(Resource(f"matrix@{die_tag}", 1.0, "peak_ms", "fair", 1))
        vec_r = float(rates.get("vector") or 0.0)
        if vec_r > 0.0:
            _add(Resource(f"vector@{die_tag}", vec_r / MS_PER_SECOND, "ops", "fair", 1))
        hbm_r = float(rates.get("hbm") or max(datapath.hbm_read_bytes_per_s, datapath.hbm_write_bytes_per_s) or 0.0)
        if hbm_r > 0.0:
            _add(Resource(f"memory@{die_tag}", hbm_r / MS_PER_SECOND, "bytes", "fair", 1))
        m_rates: dict[str, float] = {}
        for mkey in ("mte2_aic", "mte2_aiv", "mte2", "mte3", "fixpipe"):
            mrate = float(rates.get(mkey) or (hbm_r if mkey in {"mte2", "mte3", "fixpipe"} else 0.0))
            if mrate > 0.0:
                m_rates[mkey] = mrate
                _add(Resource(f"{mkey}@{die_tag}", mrate / MS_PER_SECOND, "bytes", "fair", 1))
        if link_bps > 0.0:
            _add(Resource(f"link@{die_tag}", link_bps / MS_PER_SECOND, "bytes", "fair", 1))
        for lid, own in (owners_by_coord or {}).get(coord, {}).items():
            if own.isolation != "disjoint":
                continue
            aic_f = float(own.aic_limit) / float(aic_full)
            aiv_f = float(own.aiv_limit) / float(aiv_full)
            if aic_f > 0.0:
                _add(Resource(f"matrix@{die_tag}:{lid}", aic_f, "peak_ms", "fair", 1))
                for mk in ("mte2_aic", "fixpipe", "mte2"):
                    if mk in m_rates:
                        _add(Resource(f"{mk}@{die_tag}:{lid}", (m_rates[mk] * aic_f) / MS_PER_SECOND, "bytes", "fair", 1))
                if "mte3" in m_rates:
                    _add(Resource(f"mte3_aic@{die_tag}:{lid}", (m_rates["mte3"] * aic_f) / MS_PER_SECOND, "bytes", "fair", 1))
            if aiv_f > 0.0:
                if vec_r > 0.0:
                    _add(Resource(f"vector@{die_tag}:{lid}", (vec_r * aiv_f) / MS_PER_SECOND, "ops", "fair", 1))
                for mk in ("mte2_aiv", "mte2", "mte3"):
                    if mk in m_rates:
                        _add(Resource(f"{mk}@{die_tag}:{lid}", (m_rates[mk] * aiv_f) / MS_PER_SECOND, "bytes", "fair", 1))
        pools[f"die_exec@{die_tag}"] = float(DIE_EXEC_CONCURRENCY_SLOTS)
        pool_units[f"die_exec@{die_tag}"] = 1

    if topology is not None and collective_ranks:
        for res in topology.resources_for_ranks(tuple(sorted(collective_ranks))):
            _add(res)
    return resources, pools, pool_units


def _max_same_stream_overlap(
    nodes: Sequence[Any], case: Case, intervals_us: Sequence[tuple[float, float]],
) -> float:
    by_ds: dict[tuple[int, int, int, int], list[tuple[float, float]]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        s_us, e_us = intervals_us[idx]
        if e_us > s_us + TIME_EPS_US:
            occ = node.occurrence
            die = _die_for_occurrence(case, occ)
            by_ds[(int(occ.replica), int(occ.stage), int(die.rank), int(node.stream_index))].append((s_us, e_us))
    best = 0.0
    for ivs in by_ds.values():
        ivs.sort()
        for (_, pe), (cs, _) in zip(ivs, ivs[1:]):
            if pe > cs:
                best = max(best, pe - cs)
    return best


def _bind_stream_plan_by_coord(
    stream_plan: StreamPlan | None,
    die_coords: Sequence[tuple[int, int, int]],
    supply_by_coord: Mapping[tuple[int, int, int], Any],
) -> tuple[dict[tuple[int, int, int], dict[str, StreamOwners]], dict[tuple[int, int, int], dict[str, Any]], str]:
    owners_by_coord: dict[tuple[int, int, int], dict[str, StreamOwners]] = {}
    receipt_by_coord: dict[tuple[int, int, int], dict[str, Any]] = {}
    summary = ""
    if stream_plan is None:
        return owners_by_coord, receipt_by_coord, summary
    for coord in die_coords:
        bound = bind_stream_owners(stream_plan, supply_by_coord[coord][0])
        owners_by_coord[coord] = bound
        receipt_by_coord[coord] = {
            lid: {
                "stream_id": lid, "isolation": own.isolation,
                "aic_limit": int(own.aic_limit), "aiv_limit": int(own.aiv_limit),
                "aic_owners": list(own.aic), "aiv_owners": list(own.aiv),
            }
            for lid, own in bound.items()
        }
        if not summary:
            summary = format_stream_core_budget_receipt(bound)
    return owners_by_coord, receipt_by_coord, summary


def _materialize_scheduled_result(
    nodes: Sequence[Any], case: Case, arch_order: Sequence[str], intervals_us: Sequence[tuple[float, float]],
    exposed_list: Sequence[float], masked_list: Sequence[str], step_span_us: float,
    receipt_by_coord: Mapping[tuple[int, int, int], dict[str, Any]], budget_summary: str,
) -> DieScheduleResult:
    arch_spans: dict[str, float] = {aid: 0.0 for aid in arch_order}
    out_nodes: list[Any] = []
    for idx, node in enumerate(nodes):
        s_us, e_us = intervals_us[idx]
        dur_us = max(0.0, e_us - s_us)
        exp_us = min(dur_us, max(0.0, exposed_list[idx]))
        fully_hidden = dur_us > EXPOSED_EPS_US and exp_us <= EXPOSED_EPS_US
        eff_exp = 0.0 if fully_hidden else exp_us
        masker = masked_list[idx] if (dur_us - eff_exp > EXPOSED_EPS_US and masked_list[idx]) else ""
        l_star = float(node.l_star_lean_us)
        roof_eff = min(1.0, l_star / dur_us) if (dur_us > 0.0 and l_star > 0.0) else None
        upd: dict[str, Any] = dict(
            start_us=s_us, end_us=e_us, dur_us=dur_us, exposed_dur_us=eff_exp,
            is_overlapped=fully_hidden, masked_by=masker,
            gap_us=0.0 if fully_hidden else max(0.0, dur_us - l_star),
            gap_ratio=None if fully_hidden else ((dur_us / l_star) if l_star > 0.0 else None),
            roofline_efficiency=roof_eff,
            opt_attainment_pct=round(roof_eff * PERCENT_SCALE, 1) if roof_eff is not None else node.opt_attainment_pct,
            bound_display=f"Overlapped by {masker} (0 exposed)" if (fully_hidden and masker) else node.bound_display,
        )
        if hasattr(node, "stream_core_budget") and budget_summary:
            upd["stream_core_budget"] = budget_summary
        out_nodes.append(replace(node, **upd))
        arch_spans[str(node.archetype_id)] = arch_spans.get(str(node.archetype_id), 0.0) + eff_exp
    return DieScheduleResult(
        nodes=tuple(out_nodes), archetype_spans_us=arch_spans,
        step_start_us=0.0, step_end_us=step_span_us, step_span_us=step_span_us,
        stream_core_budget=dict(receipt_by_coord), stream_core_budget_summary=budget_summary,
        max_same_stream_overlap_us=_max_same_stream_overlap(nodes, case, intervals_us),
    )


def _submit_scheduled_node(
    engine: Engine,
    observer: ServiceGrantObserver,
    idx: int,
    node: Any,
    *,
    case: Case,
    supply_by_coord: Mapping[tuple[int, int, int], Any],
    owners_by_coord: Mapping[tuple[int, int, int], Mapping[str, StreamOwners]],
    col_plans: Mapping[OpOccurrence, tuple[Any, float]],
    occ_to_idx: Mapping[OpOccurrence, int],
    end_by_occ: dict[OpOccurrence, Any],
    queue_tail: dict[tuple[tuple[int, int, int], int], tuple[Any, int | None]],
    arch_barrier: Any,
    arch_barrier_idxs: Sequence[int],
    launch_floor_us: float,
) -> tuple[Any, Any, tuple[int, ...]]:
    occ = node.occurrence
    die = _die_for_occurrence(case, occ)
    coord = (int(occ.replica), int(occ.stage), int(die.rank))
    die_tag = f"r{coord[0]}:s{coord[1]}:d{coord[2]}"
    s_idx = int(node.stream_index)
    lane_id = _canonical_stream_lane_name(str(node.stream), s_idx)
    owners = owners_by_coord.get(coord, {}).get(lane_id)

    deps_ev: list[Any] = [arch_barrier]
    c_preds: list[int] = list(arch_barrier_idxs)
    for pred in node.predecessors:
        if pred in end_by_occ:
            deps_ev.append(end_by_occ[pred])
            if pred in occ_to_idx:
                c_preds.append(occ_to_idx[pred])
    qkey = (coord, s_idx)
    if qkey in queue_tail:
        q_ev, q_idx = queue_tail[qkey]
        deps_ev.append(q_ev)
        if q_idx is not None:
            c_preds.append(q_idx)

    ready_ev = engine.submit(f"r:{idx}", after=tuple({id(e): e for e in deps_ev}.values()))
    col_entry = col_plans.get(occ)
    datapath, rates, peaks, service_modes, link_bps = supply_by_coord[coord]
    demand, placement, latency_ms, uses_cores = _node_physical_demand_and_placement(
        node, die_tag=die_tag, lane_id=lane_id, datapath=datapath, rates=rates, peaks=peaks,
        service_modes=service_modes, link_bps=link_bps, owners=owners,
        has_topology_plan=(col_entry is not None), launch_floor_us=launch_floor_us,
    )
    is_persist = is_persistent_whole_die_node(node)
    if is_persist and owners is not None and owners.isolation == "disjoint":
        if int(owners.aic_limit) < int(datapath.aic_count) or int(owners.aiv_limit) < int(datapath.aiv_count):
            raise ModelingError(f"persistent_kernel_requires_full_die:{node.op_id}")
    alloc = Allocation(f"exec:{idx}", f"die_exec@{die_tag}", float(DIE_EXEC_CONCURRENCY_SLOTS) if is_persist else 1.0)
    acq, rel = ((alloc,), (alloc,)) if uses_cores else ((), ())

    observer.current_node_idx = idx
    if col_entry is not None:
        plan, min_us = col_entry
        admit_ev = engine.submit(f"a:{idx}", after=(ready_ev,), acquire=acq) if acq else ready_ev
        done_deps: list[Any] = [plan.start(engine, f"c:{idx}:{node.op_id}", admit_ev)]
        min_floor_us = max(float(min_us), float(getattr(node, "l_star_lean_us", 0.0) or 0.0))
        if min_floor_us > 0.0:
            done_deps.append(engine.submit(f"f:{idx}", after=(admit_ev,), latency=min_floor_us / US_PER_MS))
        if demand:
            done_deps.append(engine.submit(f"d:{idx}", work=demand, after=(admit_ev,), placement=placement or None))
        start_ev = admit_ev
        end_ev = engine.submit(f"e:{idx}", after=tuple(done_deps), free=rel)
    elif demand or latency_ms > 0.0 or acq:
        end_ev = engine.submit(
            f"op:{idx}", work=demand or None, after=(ready_ev,),
            latency=latency_ms, placement=placement or None, acquire=acq, free=rel,
        )
        start_ev = end_ev
    else:
        start_ev = end_ev = ready_ev
    observer.current_node_idx = None

    end_by_occ[occ] = end_ev
    queue_tail[qkey] = (end_ev, idx)
    return start_ev, end_ev, tuple(dict.fromkeys(c_preds))


def schedule_target_graph_on_engine(
    nodes: Sequence[Any],
    case: Case,
    *,
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]] | None = None,
    topology: Topology | None = None,
    primitive_table: Any = None,
    stream_plan: StreamPlan | None = None,
    launch_floor_us: float = 0.0,
) -> DieScheduleResult:
    """Schedule ``nodes`` on one real Die + Topology ``Engine`` with ``ExposedObserver`` attribution."""
    if not nodes:
        return DieScheduleResult((), {}, 0.0, 0.0, 0.0, {}, "", 0.0)

    col_plans = dict(collective_plans or {})
    by_arch: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    arch_order: list[str] = []
    die_by_coord: dict[tuple[int, int, int], Device] = {}
    stream_indices_by_coord: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for idx, node in enumerate(nodes):
        aid = str(node.archetype_id)
        if aid not in by_arch:
            arch_order.append(aid)
        by_arch[aid].append((idx, node))
        die = _die_for_occurrence(case, node.occurrence)
        coord = (int(node.occurrence.replica), int(node.occurrence.stage), int(die.rank))
        die_by_coord[coord] = die
        stream_indices_by_coord[coord].add(int(node.stream_index))

    die_coords = sorted(die_by_coord)
    supply_by_coord = {
        c: _effective_die_supply(die_by_coord[c], case.machine.links, case.machine.transfer_channels, primitive_table)
        for c in die_coords
    }
    owners_by_coord, receipt_by_coord, budget_summary = _bind_stream_plan_by_coord(
        stream_plan, die_coords, supply_by_coord,
    )
    col_ranks = {int(r) for plan, _ in col_plans.values() for r in getattr(plan, "ranks", ())}
    resources, pools, pool_units = _build_die_resources_and_pools(
        die_coords, supply_by_coord, stream_indices_by_coord, topology, col_ranks,
        owners_by_coord=owners_by_coord,
    )
    engine = Engine(resources, pools=pools, pool_units=pool_units, backend="python")
    observer = ServiceGrantObserver()
    engine.attach_wave_recorder(observer)
    root_ev = engine.submit("step_root")

    occ_to_idx = {node.occurrence: idx for idx, node in enumerate(nodes)}
    end_by_occ: dict[OpOccurrence, Any] = {}
    queue_tail: dict[tuple[tuple[int, int, int], int], tuple[Any, int | None]] = {}
    arch_barrier, arch_barrier_idxs = root_ev, []
    start_events: dict[int, Any] = {}
    end_events: dict[int, Any] = {}
    causal_preds: dict[int, tuple[int, ...]] = {}

    unaggregated = all(max(1, int(getattr(n, "repeat_count", 1) or 1)) <= 1 for n in nodes)
    if unaggregated:
        ordered_all = sorted(enumerate(nodes), key=lambda item: (item[1].ordinal, item[1].start_us))
        for idx, node in ordered_all:
            s_ev, e_ev, preds = _submit_scheduled_node(
                engine, observer, idx, node, case=case, supply_by_coord=supply_by_coord,
                owners_by_coord=owners_by_coord, col_plans=col_plans, occ_to_idx=occ_to_idx,
                end_by_occ=end_by_occ, queue_tail=queue_tail, arch_barrier=root_ev,
                arch_barrier_idxs=(), launch_floor_us=launch_floor_us,
            )
            start_events[idx], end_events[idx], causal_preds[idx] = s_ev, e_ev, preds
    else:
        for arch_idx, aid in enumerate(arch_order):
            ordered = sorted(by_arch[aid], key=lambda item: (item[1].ordinal, item[1].start_us))
            arch_done_evs: list[Any] = []
            arch_done_idxs: list[int] = []
            for idx, node in ordered:
                s_ev, e_ev, preds = _submit_scheduled_node(
                    engine, observer, idx, node, case=case, supply_by_coord=supply_by_coord,
                    owners_by_coord=owners_by_coord, col_plans=col_plans, occ_to_idx=occ_to_idx,
                    end_by_occ=end_by_occ, queue_tail=queue_tail, arch_barrier=arch_barrier,
                    arch_barrier_idxs=arch_barrier_idxs, launch_floor_us=launch_floor_us,
                )
                start_events[idx], end_events[idx], causal_preds[idx] = s_ev, e_ev, preds
                arch_done_evs.append(e_ev)
                arch_done_idxs.append(idx)
            if arch_done_evs:
                arch_barrier = engine.submit(f"arch:{arch_idx}", after=tuple(arch_done_evs))
                arch_barrier_idxs = arch_done_idxs

    step_done = engine.submit("step_done", after=tuple(end_events.values()))
    engine.advance(until=(step_done,))
    engine.detach_wave_recorder()

    step_start_us = float(root_ev.time or 0.0) * US_PER_MS
    step_span_us = max(0.0, float(step_done.time or 0.0) * US_PER_MS - step_start_us)
    intervals_us = [
        (
            s_u := max(0.0, float(
                observer.admit_ms_by_node.get(i)
                if observer.admit_ms_by_node.get(i) is not None
                else (start_events[i].started if start_events[i].started is not None else (start_events[i].time or 0.0))
            ) * US_PER_MS - step_start_us),
            max(s_u, float(end_events[i].time or 0.0) * US_PER_MS - step_start_us),
        )
        for i in range(len(nodes))
    ]
    exposed_list, masked_list = attribute_engine_exposure(
        nodes, intervals_us, observer, causal_preds, step_span_us,
        step_start_ns=int(round(step_start_us * NS_PER_US)),
    )
    return _materialize_scheduled_result(
        nodes, case, arch_order, intervals_us, exposed_list, masked_list,
        step_span_us, receipt_by_coord, budget_summary,
    )


def weighted_step_span_us(nodes: Sequence[Any]) -> float:
    """Return repeat-weighted exposed step duration in microseconds."""
    return sum(
        max(0.0, float(getattr(n, "exposed_dur_us", 0.0))) * max(1, int(getattr(n, "repeat_count", 1) or 1))
        for n in nodes
    )


def evaluate_disjoint_core_budgets(
    nodes: Sequence[Any],
    case: Case,
    *,
    shared_result: DieScheduleResult,
    collective_plans: Mapping[OpOccurrence, tuple[Any, float]] | None = None,
    topology: Topology | None = None,
    primitive_table: Any = None,
    launch_floor_us: float = 0.0,
) -> DieScheduleResult:
    """Test explicit ``StreamCoreBudget(isolation='disjoint')`` splits; adopt only if makespan drops."""
    stream_lanes: dict[str, tuple[bool, bool]] = {}
    for node in nodes:
        if float(node.dur_us) <= TIME_EPS_US:
            continue
        lname = _canonical_stream_lane_name(str(node.stream), int(node.stream_index))
        has_c = float(node.flops) > 0.0 and str(node.family).lower() not in ZERO_AIC_COMM_FAMILIES
        has_v = float(node.vector_ops) > 0.0
        prev_c, prev_v = stream_lanes.get(lname, (False, False))
        stream_lanes[lname] = (prev_c or has_c, prev_v or has_v)

    aux_lanes = [lid for lid, (hc, hv) in stream_lanes.items() if lid != MAIN_STREAM and (hc or hv)]
    if MAIN_STREAM not in stream_lanes or len(aux_lanes) != 1 or any(is_persistent_whole_die_node(n) for n in nodes):
        return shared_result
    if sum(float(n.dur_us) for n in shared_result.nodes) <= shared_result.step_span_us + EXPOSED_EPS_US:
        return shared_result

    die = case.machine.dies[0]
    datapath = datapath_for_die(die)
    aic_tot, aiv_tot = int(datapath.aic_count), int(datapath.aiv_count)
    if aic_tot < 2 or aiv_tot < 2:
        return shared_result

    aux_id = aux_lanes[0]
    aux_needs_aic, aux_needs_aiv = stream_lanes[aux_id]
    main_needs_aic, main_needs_aiv = stream_lanes[MAIN_STREAM]
    best = shared_result
    best_weighted = weighted_step_span_us(shared_result.nodes)
    threshold = best_weighted * (1.0 - DISJOINT_SPLIT_MIN_GAIN_RATIO)

    for num, den in DISJOINT_SPLIT_RATIOS:
        main_aic, aux_aic = ((aic_tot, 0) if not aux_needs_aic else ((aic_tot * num) // den, aic_tot - (aic_tot * num) // den))
        if aux_needs_aic and main_needs_aic and (main_aic <= 0 or aux_aic <= 0):
            continue
        main_aiv, aux_aiv = ((aiv_tot, 0) if not aux_needs_aiv else ((aiv_tot * num) // den, aiv_tot - (aiv_tot * num) // den))
        if aux_needs_aiv and main_needs_aiv and (main_aiv <= 0 or aux_aiv <= 0):
            continue
        extra_lanes = tuple(
            StreamLane(lid, StreamCoreBudget(aic=0, aiv=0, isolation="disjoint"))
            for lid in stream_lanes if lid not in {MAIN_STREAM, aux_id}
        )
        plan = StreamPlan(
            rank=int(die.rank),
            lanes=(
                StreamLane(MAIN_STREAM, StreamCoreBudget(aic=main_aic, aiv=main_aiv, isolation="disjoint")),
                StreamLane(aux_id, StreamCoreBudget(aic=aux_aic, aiv=aux_aiv, isolation="disjoint")),
                *extra_lanes,
            ),
        )
        try:
            cand = schedule_target_graph_on_engine(
                nodes, case, collective_plans=collective_plans, topology=topology,
                primitive_table=primitive_table, stream_plan=plan, launch_floor_us=launch_floor_us,
            )
        except ModelingError:
            continue
        cand_weighted = weighted_step_span_us(cand.nodes)
        if cand_weighted < best_weighted and cand_weighted <= threshold:
            best = cand
            best_weighted = cand_weighted
    return best


__all__ = [
    "ACCEPTANCE_ANCHOR_FAMILIES",
    "DIE_EXEC_CONCURRENCY_SLOTS",
    "EXTENDED_MOE_ACTIVATIONS",
    "FUSED_MC2_ROUTED_LEVEL",
    "FUSED_MC2_SHARED_LEVEL",
    "PERSISTENT_MOE_FAMILIES",
    "SHIPPING_MOE_ACTIVATIONS",
    "TAIL_COMMIT_FAMILIES",
    "DieScheduleResult",
    "MoERouteDecision",
    "ServiceGrantObserver",
    "attribute_engine_exposure",
    "build_discrete_mso_candidate_nodes",
    "compute_acceptance_tail_closure",
    "enforce_moe_mutex_on_nodes",
    "evaluate_disjoint_core_budgets",
    "format_stream_core_budget_receipt",
    "is_a5_hardware",
    "is_acceptance_commit_record",
    "is_draft_record",
    "is_latent_moe_model",
    "is_persistent_whole_die_node",
    "is_shared_expert_op_id",
    "moe_route_legality",
    "resolve_arch_family_tag",
    "resolve_barrier_predecessors",
    "schedule_target_graph_on_engine",
    "weighted_step_span_us",
]
