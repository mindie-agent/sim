"""Receipt-driven D3 taxonomy, Waterfall accounting, Sheet 4 cards & 3-track roadmap.

[Role] Own single-source comparison accounting and actionable task projection over
  aligned ``ExportBundle`` comparisons:
  (1) Strict 3-category operator taxonomy enforcing the unified inclusive
      ``>= 0.050 ms`` materiality gate (and ``> 0 us/(tok*die)``) across BOTH
      genuine ``N->1`` new fused kernels (``[需新开发融合算子]``) and shipping
      ``1->1`` tuning groups (``[现成算子深度调优]``), routing sub-threshold
      micro-fusions cleanly to ``Residual Ops`` (``[已近物理下界 / 无需改动]``,
      zero cards and zero framework roadmap badges);
  (2) Exact signed Waterfall Ledger accounting conserving ``Step_ms`` and
      ``us/(tok*die)`` across Sheet 2 columns ``AB..AF`` and Sheet 4;
  (3) Deterministic canonical ABI ``card_id`` generation (zero ``_2`` serial
      suffixes), max-gain submodule attribution, and ``<= 12`` high-signal cards;
  (4) Pure receipt-driven 3-track roadmap (``[1] 配置与并行`` /
      ``[2] 框架图与多流改造`` / ``[3] 新算子开发与 Tiling 调优``) with strict
      allowlist-only public view projection and inclusive ``>= 0.050 ms`` gate.
[Invariants] Zero string-sniffing prose dispatch tables, zero ``>=75%/78%/85%``
  target phrases, zero guessed ``.py`` paths, zero ACLGraph catch-all defaults,
  zero ``1->1`` native ops in Track ``[2]``, zero sub-``0.050 ms`` promotions,
  and zero raw internal keys or ``str(dict)`` leaks in public diffs/receipts.
[Timeouts & Locks] Pure O(Aligned Rows) in-memory passes; no I/O or locks.
[Side-Effect Policy] Read-only over immutable export data records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

from .models import (
    BINDING_STATE_MEASURED, BINDING_STATE_UNBOUND_FORMULA,
    DEFAULT_TOP_OPTIMIZE_KERNELS, KIB_BYTES, MIN_DENOM_EPS,
    NS_PER_MS, NS_PER_US, OVERLAP_EXPOSED_EPS_US,
    PENDING_KIND_AT_FLOOR, PENDING_KIND_TO_IMPLEMENT, PENDING_KIND_TO_OPTIMIZE,
    TRACK_CONFIG_ENV, TRACK_FRAMEWORK, TRACK_KERNEL, US_PER_MS,
    ActionRoadmapRow, AlignedComparisonRow, ExportBundle, NewKernelBacklogItem,
    OperatorExportRow, PendingKernelItem, bundle_step_int_ns, op_exposed_int_ns,
)
from .public_view import (
    PUBLIC_PIPE_NAMES,
    THREE_TRACK_LABELS,
    diff_public_scenario_configs,
    is_same_public_config,
    project_public_effective_config,
    sanitize_public_text,
    sanitize_source_ref,
)

CONSERVATION_EPS_MS = 1e-6
TUNE_MIN_SAVED_MS = 0.05
TUNE_THRESHOLD_TOL_MS = 1e-12
MAX_DISPLAY_CARDS = 12
MAX_ROADMAP_FW_ITEMS = 6
MAX_ROADMAP_KERNEL_ITEMS = 8
CARD_SLUG_MAX_LEN = 44
CARD_DIGEST_HEX_LEN = 6
SUB_MS_DISPLAY_THRESHOLD = 0.001
#: Unified inclusive materiality threshold (>= 0.050 ms) for new kernels, tuning, and framework promotion.
NEW_KERNEL_MIN_SAVED_MS = TUNE_MIN_SAVED_MS
MIN_FUSION_MEMBERS = 2
MAX_SOURCE_REF_SUMMARY_LEN = 160
MAX_ROLES_IN_CHAIN = 3
STRUCTURAL_EDIT_OPCODES = frozenset({
    "FuseRegion", "CollectiveTransform", "HoistOrDeduplicate", "StreamSchedule", "StreamCoreBudget",
})

#: Canonical 3-stage + residual waterfall stage identifiers shared across Sheet 1 & Sheet 4.
WF_STAGE_CONFIG = "config"
WF_STAGE_FRAMEWORK = "framework"
WF_STAGE_KERNEL = "kernel"
WF_STAGE_RESIDUAL = "residual"
_FRAMEWORK_REASONS = frozenset({
    "stream_masking_change",
    "shipping_fusion_enablement",
    "subgraph_reassociation",
    "shipping_route_switch",
    "framework_graph_edit",
})

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class GroupAccounting:
    """Per-aligned-row signed Waterfall accounting and canonical card group assignment."""

    row_index: int
    bucket: Literal["config", "new", "tune", "residual"]
    card_id: str
    category: str
    before_ms: float
    after_ms: float
    saved_ms: float
    before_upt: float
    after_upt: float
    saved_upt: float
    after_pre_edit_ms: float | None
    config_share_ms: float
    config_share_upt: float
    kernel_share_ms: float
    kernel_share_upt: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class WaterfallSummary:
    """End-to-end signed Waterfall Ledger reconciling Before Step -> After Step."""

    before_step_ms: float
    after_step_ms: float
    before_upt: float
    after_upt: float
    config_saved_ms: float
    config_saved_upt: float
    new_kernel_saved_ms: float
    new_kernel_saved_upt: float
    tune_kernel_saved_ms: float
    tune_kernel_saved_upt: float
    residual_saved_ms: float
    residual_saved_upt: float
    reconstructed_after_ms: float
    reconciliation_error_ms: float
    reconciliation_error_upt: float
    sheet2_before_total_ms: float
    sheet2_after_total_ms: float
    before_endpoint_error_ms: float
    after_endpoint_error_ms: float
    before_step_ns: int = 0
    after_step_ns: int = 0
    idle_before_ns: int = 0
    idle_after_ns: int = 0
    reconciliation_error_ns: int = 0

    def to_brief_dict(self) -> dict[str, float]:
        return {
            "config_saved_ms": float(self.config_saved_ms), "new_kernel_saved_ms": float(self.new_kernel_saved_ms),
            "tune_kernel_saved_ms": float(self.tune_kernel_saved_ms), "residual_saved_ms": float(self.residual_saved_ms),
            "reconciliation_error_ms": float(self.reconciliation_error_ms),
            "config_saved_us_per_tok_die": float(self.config_saved_upt),
            "new_kernel_saved_us_per_tok_die": float(self.new_kernel_saved_upt),
            "tune_kernel_saved_us_per_tok_die": float(self.tune_kernel_saved_upt),
            "residual_saved_us_per_tok_die": float(self.residual_saved_upt),
            "reconciliation_error_us_per_tok_die": float(self.reconciliation_error_upt),
        }


@dataclass(frozen=True, slots=True)
class ComparisonPlan:
    """Single coherent comparison projection consumed by XLSX, JSON brief, and SDK."""

    aligned_rows: tuple[AlignedComparisonRow, ...]
    group_accounting: tuple[GroupAccounting, ...]
    waterfall: WaterfallSummary
    pending_items: tuple[PendingKernelItem, ...]
    roadmap: tuple[ActionRoadmapRow, ...]
    codesign_note: str
    stage_waterfall: tuple[dict[str, Any], ...] = ()


def format_signed_gain_text(saved_ms: float) -> str:
    """Format a gain without rendering small positive values as ``0.000 ms``."""
    return f"{saved_ms * US_PER_MS:.2f} us" if 0.0 < abs(saved_ms) < SUB_MS_DISPLAY_THRESHOLD else f"{saved_ms:.3f} ms"


def _arch_tag_from_bundle(bundle: ExportBundle) -> str:
    """Read structured architecture family from ExportBundle metadata or canonical HW normalizer."""
    from mindie_sim.catalog.npu_checkpoints import normalize_hw_family

    eff = bundle.effective_config if isinstance(bundle.effective_config, Mapping) else {}
    raw = str(eff.get("arch_family") or "").strip().lower()
    if not raw and isinstance(bundle.hardware_die_summary, Mapping):
        raw = str(bundle.hardware_die_summary.get("arch_family") or "").strip().lower()
    if not raw and bundle.hardware_sku:
        raw = str(normalize_hw_family(bundle.hardware_sku) or "").strip().lower()
    if not raw:
        raise ModelingError(f"missing_structured_arch_family_on_bundle:{bundle.hardware_sku!r}")
    return _SLUG_RE.sub("_", raw).strip("_")


def _short_arch_name(title: str) -> str:
    return re.sub(r"\s*\(×\d+[^)]*\)\s*$", "", title.split("—")[0].strip())


def _public_effective_config(bundle: ExportBundle) -> dict[str, Any]:
    return project_public_effective_config(bundle)


def _is_same_deployment_config(before: ExportBundle, after: ExportBundle) -> bool:
    return (
        is_same_public_config(before, after)
        and abs(float(before.n_tok_per_die or 1.0) - float(after.n_tok_per_die or 1.0)) <= MIN_DENOM_EPS
    )


def _extract_deployment_diffs(before: ExportBundle, after: ExportBundle) -> list[tuple[str, str, str]]:
    """Return factual ``(field_label, before_value, after_value)`` tuples strictly from allowlisted fields."""
    return diff_public_scenario_configs(before, after)


def build_parallelism_codesign_note(before_bundle: ExportBundle, after_bundle: ExportBundle) -> str:
    """Render concise factual diff of resolved deployment winner fields (empty if unchanged)."""
    return " | ".join(f"{f}: {bv} -> {av}" for f, bv, av in _extract_deployment_diffs(before_bundle, after_bundle))


def _op_exposed_ns(op: OperatorExportRow, repeat_count: int) -> int:
    return op_exposed_int_ns(op, repeat_count)


def _bundle_step_and_idle_ns(bundle: ExportBundle) -> tuple[int, int]:
    step_ns = bundle_step_int_ns(bundle)
    op_sum_ns = sum(_op_exposed_ns(op, op.repeat_count) for op in bundle.operators)
    if step_ns < op_sum_ns:
        step_ns = op_sum_ns
    return step_ns, max(0, step_ns - op_sum_ns)


def _row_exposed_metrics_ns(
    row: AlignedComparisonRow, before_ntok: float, after_ntok: float,
) -> tuple[int, int, int, float, float, float, float, float, float]:
    b_rep, a_rep = row.effective_before_repeat, row.effective_after_repeat
    b_ns = sum(_op_exposed_ns(b, b_rep) for b in row.before_ops)
    a_ns = sum(_op_exposed_ns(a, a_rep) for a in row.after_ops)
    saved_ns = b_ns - a_ns
    b_ms = float(b_ns) / float(NS_PER_MS)
    a_ms = float(a_ns) / float(NS_PER_MS)
    saved_ms = float(saved_ns) / float(NS_PER_MS)
    b_upt = float(b_ns) / (NS_PER_US * before_ntok)
    a_upt = float(a_ns) / (NS_PER_US * after_ntok)
    return b_ns, a_ns, saved_ns, b_ms, a_ms, saved_ms, b_upt, a_upt, b_upt - a_upt


def _row_exposed_metrics(row: AlignedComparisonRow, before_ntok: float, after_ntok: float) -> tuple[float, float, float, float, float, float]:
    _b_ns, _a_ns, _s_ns, b_ms, a_ms, saved_ms, b_upt, a_upt, saved_upt = _row_exposed_metrics_ns(
        row, before_ntok, after_ntok,
    )
    return b_ms, a_ms, saved_ms, b_upt, a_upt, saved_upt


def _extract_clean_equations(raw_eqs: Any) -> list[str]:
    if not isinstance(raw_eqs, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw_eqs:
        if not isinstance(item, str):
            continue
        cleaned = sanitize_public_text(item)
        if cleaned:
            out.append(cleaned)
    return out


def _safe_nonneg_int(val: Any) -> int:
    return int(val) if isinstance(val, int) and not isinstance(val, bool) and val > 0 else 0


def _safe_nonneg_float(val: Any) -> float:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        fval = float(val)
        if math.isfinite(fval) and fval > 0.0:
            return fval
    return 0.0


def _format_formula_eval_receipt(formula_eval: Any) -> str:
    if formula_eval is None:
        return ""
    get = formula_eval.get if isinstance(formula_eval, Mapping) else lambda k, d=None: getattr(formula_eval, k, d)
    eqs = _extract_clean_equations(get("composed_equations"))
    ub_p = _safe_nonneg_int(get("ub_peak_bytes"))
    ub_b = _safe_nonneg_int(get("ub_budget_bytes"))
    sram_ok = bool(get("sram_verified", False)) if isinstance(get("sram_verified", False), bool) else False
    proj_us = _safe_nonneg_float(get("projected_us"))
    raw_pipe = get("dominant_pipe")
    pipe_cand = sanitize_public_text(raw_pipe) if isinstance(raw_pipe, str) else ""
    pipe = pipe_cand if pipe_cand.lower() in PUBLIC_PIPE_NAMES else ""
    parts = [p for p in (
        ("Formula: " + " ; ".join(eqs)) if eqs else "",
        f"UB={ub_p / KIB_BYTES:.1f}/{ub_b / KIB_BYTES:.1f}KiB ({'PASS' if sram_ok else 'FAIL'})" if ub_b > 0 else "",
        (f"Projected={proj_us:.2f}us" + (f", pipe={pipe}" if pipe else "")) if proj_us > 0.0 else "",
    ) if p]
    return " | ".join(parts)


def _after_pre_edit_exposed_ms(row: AlignedComparisonRow) -> float | None:
    if not row.after_ops:
        return None
    a_rep, vals = row.effective_after_repeat, []
    for a_op in row.after_ops:
        fe = a_op.formula_eval
        raw = fe.get("after_pre_edit_exposed_us") if isinstance(fe, Mapping) else getattr(fe, "after_pre_edit_exposed_us", None)
        if raw is None:
            return None
        vals.append(float(raw) * a_rep / US_PER_MS)
    return sum(vals)


def _canonical_abi_key(category_tag: str, a_op: OperatorExportRow, arch_tag: str) -> tuple[str, str]:
    kspec = a_op.kernel_synthesis_spec
    kernel_name = (str(getattr(kspec, "kernel_name", "") or "").strip() if kspec is not None and category_tag == "new" else "") or (
        a_op.ascend_kernel or a_op.implementation or a_op.canonical_op_key or a_op.op_id
    )
    abi_ext = str(getattr(kspec, "abi_signature_id", "") or "").strip() if kspec is not None else ""
    base_norm = _SLUG_RE.sub("_", kernel_name.lower()).strip("_") or "kernel"
    ext_norm = _SLUG_RE.sub("_", abi_ext.lower()).strip("_") if abi_ext else ""
    return f"{category_tag}::{arch_tag}::{base_norm}" + (f"::{ext_norm}" if ext_norm else ""), base_norm[:CARD_SLUG_MAX_LEN].rstrip("_") or "kernel_card"


def _assign_deterministic_card_ids(groups: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    stem_counts: dict[str, int] = {}
    for entry in groups:
        stem_counts[str(entry["stem"])] = stem_counts.get(str(entry["stem"]), 0) + 1
    result: dict[str, str] = {}
    for entry in groups:
        key, stem = str(entry["kernel_key"]), str(entry["stem"])
        if stem_counts[stem] <= 1:
            result[key] = stem
        else:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:CARD_DIGEST_HEX_LEN]
            result[key] = f"{stem[:max(1, CARD_SLUG_MAX_LEN - CARD_DIGEST_HEX_LEN - 1)].rstrip('_')}_{digest}"
    return result


def _applied_fuse_outputs(after_bundle: ExportBundle) -> set[str]:
    out: set[str] = set()
    for rec in (after_bundle.applied_edit_receipts or ()):
        if not isinstance(rec, Mapping):
            continue
        if str(rec.get("opcode") or "").strip() != "FuseRegion":
            continue
        if str(rec.get("status", "applied")).lower() != "applied":
            continue
        if rec.get("verified") is False:
            continue
        for key in ("output_occurrences", "members"):
            for item in (rec.get(key) or ()):
                if item:
                    out.add(str(item))
    return out


def _is_verified_new_fusion(
    row: AlignedComparisonRow,
    *,
    is_comparison: bool,
    saved_ms: float,
    fuse_outputs: set[str],
    rewrite_verified: bool = False,
) -> bool:
    """Require real verified FuseRegion receipt or SRAM-verified KernelSynthesisSpec."""
    del saved_ms
    if len(row.after_ops) != 1:
        return False
    a_op = row.after_ops[0]
    if not a_op.needs_new_kernel and a_op.is_shipping_abi:
        return False
    kspec = a_op.kernel_synthesis_spec
    if kspec is not None and bool(getattr(kspec, "sram_verified", False)):
        members = getattr(kspec, "constituent_op_ids", ()) or ()
        if len(members) >= MIN_FUSION_MEMBERS:
            return True
    if not rewrite_verified or not fuse_outputs:
        return False
    if a_op.op_id in fuse_outputs:
        return True
    if is_comparison and len(row.before_ops) >= MIN_FUSION_MEMBERS:
        if any(b.op_id in fuse_outputs for b in row.before_ops):
            return True
    if len(a_op.absorbed_source_ops) >= MIN_FUSION_MEMBERS:
        if any(src in fuse_outputs for src in a_op.absorbed_source_ops):
            return True
    return False


def _is_masking_only_change(row: AlignedComparisonRow) -> bool:
    if len(row.before_ops) != 1 or len(row.after_ops) != 1:
        return False
    b_op, a_op = row.before_ops[0], row.after_ops[0]
    if b_op.is_overlapped == a_op.is_overlapped and abs(b_op.exposed_dur_us - a_op.exposed_dur_us) <= OVERLAP_EXPOSED_EPS_US:
        return False
    return abs(b_op.dur_us - a_op.dur_us) <= OVERLAP_EXPOSED_EPS_US and (
        a_op.is_overlapped or b_op.is_overlapped or a_op.exposed_dur_us < a_op.dur_us - OVERLAP_EXPOSED_EPS_US
    )


def _has_tuning_evidence(b_op: OperatorExportRow, a_op: OperatorExportRow, *, same_config: bool) -> bool:
    return (
        a_op.formula_eval is not None or a_op.measured_sum_us is not None
        or a_op.binding_state in {BINDING_STATE_UNBOUND_FORMULA, BINDING_STATE_MEASURED}
        or a_op.opt_pillar in {"pillar2_roofline", "pillar2_tiler_extension", "pillar1_fusion"}
        or (same_config and (b_op.dur_us - a_op.dur_us) > OVERLAP_EXPOSED_EPS_US)
    )


def _prelim_classify_row(
    row: AlignedComparisonRow, *, is_comparison: bool, same_config: bool,
    saved_ms: float, saved_upt: float, after_pre_edit_ms: float | None, fuse_outputs: set[str],
    rewrite_verified: bool = False,
) -> tuple[str, OperatorExportRow | None, str]:
    if _is_verified_new_fusion(
        row, is_comparison=is_comparison, saved_ms=saved_ms,
        fuse_outputs=fuse_outputs, rewrite_verified=rewrite_verified,
    ):
        return "new", row.after_ops[0], "verified_new_fusion"
    if not is_comparison:
        if len(row.after_ops) == 1:
            a_op = row.after_ops[0]
            gap_ms = 0.0 if a_op.is_overlapped else max(0.0, (a_op.exposed_dur_us - a_op.l_star_lean_us) * row.effective_after_repeat / US_PER_MS)
            if gap_ms >= TUNE_MIN_SAVED_MS - TUNE_THRESHOLD_TOL_MS and not a_op.needs_new_kernel and a_op.is_shipping_abi:
                return "tune", a_op, "standalone_lstar_gap"
        return "residual", (row.after_ops[0] if row.after_ops else None), "at_floor_or_below_threshold"

    if not row.after_ops or not row.before_ops:
        return "config", None, "eliminated_or_added_op"
    if len(row.before_ops) >= MIN_FUSION_MEMBERS and len(row.after_ops) == 1:
        return ("residual", row.after_ops[0], "zero_gain_unreceipted_fusion") if row.after_ops[0].needs_new_kernel else ("config", row.after_ops[0], "shipping_fusion_enablement")
    if len(row.before_ops) != 1 or len(row.after_ops) != 1:
        return "config", None, "subgraph_reassociation"
    b_op, a_op = row.before_ops[0], row.after_ops[0]
    if _is_masking_only_change(row):
        return "config", a_op, "stream_masking_change"
    if a_op.is_overlapped and b_op.is_overlapped:
        return "residual", a_op, "fully_masked_both_sides"
    if abs(saved_ms) <= CONSERVATION_EPS_MS and abs(saved_upt) <= CONSERVATION_EPS_MS:
        return "residual", a_op, "zero_delta"
    if a_op.is_shipping_abi and not a_op.needs_new_kernel and _has_tuning_evidence(b_op, a_op, same_config=same_config):
        if not same_config and after_pre_edit_ms is None and saved_ms <= 0.0:
            return "config", a_op, "batch_or_parallel_delta"
        return "tune", a_op, "shipping_kernel_tuning_candidate"
    if not same_config:
        return "config", a_op, "config_or_batch_scaling"
    if b_op.ascend_kernel != a_op.ascend_kernel and a_op.is_shipping_abi:
        return "config", a_op, "shipping_route_switch"
    return "residual", a_op, "at_floor_or_unattributed"


def _add_row_to_group(
    groups: dict[str, dict[str, Any]], *, key: str, stem: str, category: str,
    row: AlignedComparisonRow, row_idx: int, a_op: OperatorExportRow, after_bundle: ExportBundle,
    b_ms: float, a_ms: float, saved_ms: float, b_upt: float, a_upt: float, saved_upt: float,
    kernel_share_ms: float, kernel_share_upt: float, has_split: bool, is_comparison: bool,
) -> None:
    a_rep, b_rep = row.effective_after_repeat, row.effective_before_repeat
    rep = a_rep or b_rep or max(1, int(row.repeat_count))
    lstar_ms = sum(0.0 if a.is_overlapped else (a.l_star_lean_us * a_rep / US_PER_MS) for a in row.after_ops)
    b_lstar_ms = sum(0.0 if b.is_overlapped else (b.l_star_lean_us * b_rep / US_PER_MS) for b in row.before_ops)
    gap_ms = max(0.0, a_ms - lstar_ms)
    gain_ms = saved_ms if is_comparison else gap_ms
    gain_upt = saved_upt if is_comparison else (gap_ms * US_PER_MS / max(float(after_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS))
    kspec, fe = a_op.kernel_synthesis_spec, a_op.formula_eval
    raw_eqs = (
        getattr(kspec, "symbolic_equations", ())
        if kspec is not None
        else (fe.get("composed_equations") if isinstance(fe, Mapping) else getattr(fe, "composed_equations", ()))
        if fe is not None else ()
    )
    math_formula = " ; ".join(_extract_clean_equations(raw_eqs))
    sram_sum = (
        sanitize_public_text(kspec.format_formula_block())
        if kspec is not None else _format_formula_eval_receipt(fe)
    )
    occ_tie = (row.archetype_id, a_op.op_id)
    bef_bounds = " / ".join(dict.fromkeys(b.bound_display or b.dominant_bound for b in row.before_ops if b.bound_display or b.dominant_bound)) or (a_op.bound_display or a_op.dominant_bound)

    entry = groups.get(key)
    if entry is None:
        raw_target = (
            str(getattr(kspec, "kernel_name", "") or "").strip()
            if kspec is not None and category == PENDING_KIND_TO_IMPLEMENT else ""
        ) or a_op.ascend_kernel or a_op.implementation
        entry = {
            "kernel_key": key, "stem": stem, "category": category,
            "target_kernel": sanitize_public_text(raw_target),
            "backend": sanitize_public_text(a_op.kernel_backend),
            "status": sanitize_public_text(a_op.kernel_status),
            "submodule": sanitize_public_text(a_op.submodule),
            "logical_role": sanitize_public_text(a_op.logical_role or a_op.op_id),
            "roles": [], "replaces": [], "sources": [], "archetypes": [], "calls": 0,
            "before_ms": 0.0, "after_ms": 0.0, "lstar_ms": 0.0, "before_lstar_ms": 0.0, "gap_ms": 0.0,
            "gain_ms": 0.0, "before_upt": 0.0, "after_upt": 0.0, "gain_upt": 0.0,
            "attr_gain_ms": 0.0, "attr_gain_upt": 0.0, "has_split": False,
            "before_bound": sanitize_public_text(bef_bounds),
            "after_bound": sanitize_public_text(a_op.bound_display or a_op.dominant_bound),
            "contract": sanitize_public_text(a_op.tensor_contract),
            "single_lstar_us": a_op.l_star_lean_us, "single_sim_us": a_op.dur_us,
            "die_summary": sanitize_public_text(after_bundle.hardware_die_summary),
            "core_budget": sanitize_public_text(a_op.stream_core_budget),
            "opt_pillar": a_op.opt_pillar or ("pillar1_fusion" if category == PENDING_KIND_TO_IMPLEMENT else "pillar2_roofline"),
            "synthesis_spec": kspec, "math_formula": math_formula, "sram_residency_summary": sram_sum,
            "formula_summary": _format_formula_eval_receipt(fe), "binding_state": a_op.binding_state,
            "source_row_indices": [], "best_member_rank": (gain_ms, gain_upt), "best_member_occ": occ_tie,
        }
        groups[key] = entry

    if (gain_ms, gain_upt) > entry["best_member_rank"] or ((gain_ms, gain_upt) == entry["best_member_rank"] and occ_tie < entry["best_member_occ"]):
        entry.update({
            "best_member_rank": (gain_ms, gain_upt), "best_member_occ": occ_tie,
            "submodule": sanitize_public_text(a_op.submodule),
            "logical_role": sanitize_public_text(a_op.logical_role or a_op.op_id),
            "before_bound": sanitize_public_text(bef_bounds),
            "after_bound": sanitize_public_text(a_op.bound_display or a_op.dominant_bound),
            "contract": sanitize_public_text(a_op.tensor_contract),
            "single_lstar_us": a_op.l_star_lean_us, "single_sim_us": a_op.dur_us,
        })
        if a_op.stream_core_budget:
            entry["core_budget"] = sanitize_public_text(a_op.stream_core_budget)

    entry["source_row_indices"].append(int(row_idx))
    if a_op.binding_state == BINDING_STATE_UNBOUND_FORMULA:
        entry["binding_state"] = BINDING_STATE_UNBOUND_FORMULA
    for b_op in row.before_ops:
        entry["replaces"].append(sanitize_public_text(b_op.ascend_kernel))
        if b_op.source_ref:
            clean_ref = sanitize_source_ref(b_op.source_ref)
            if clean_ref:
                entry["sources"].append(clean_ref)
    if a_op.source_ref:
        clean_ref = sanitize_source_ref(a_op.source_ref)
        if clean_ref:
            entry["sources"].append(clean_ref)
    role_lbl = sanitize_public_text(a_op.logical_role or a_op.op_id)
    if role_lbl and role_lbl not in entry["roles"]:
        entry["roles"].append(role_lbl)
    entry["archetypes"].append(f"{_short_arch_name(row.archetype_title)} (×{rep})")
    entry["calls"] += rep
    for name, delta in (
        ("before_ms", b_ms), ("after_ms", a_ms), ("lstar_ms", lstar_ms), ("before_lstar_ms", b_lstar_ms),
        ("gap_ms", gap_ms), ("gain_ms", gain_ms), ("before_upt", b_upt), ("after_upt", a_upt),
        ("gain_upt", gain_upt), ("attr_gain_ms", kernel_share_ms), ("attr_gain_upt", kernel_share_upt),
    ):
        entry[name] += delta
    if has_split:
        entry["has_split"] = True
    if kspec is not None and entry["synthesis_spec"] is None:
        entry["synthesis_spec"], entry["math_formula"], entry["sram_residency_summary"] = kspec, math_formula, sram_sum
    elif fe is not None and not entry["formula_summary"]:
        entry["formula_summary"] = _format_formula_eval_receipt(fe)
        if not entry["sram_residency_summary"]:
            entry["sram_residency_summary"] = entry["formula_summary"]


def _materialize_card(prio: int, entry: Mapping[str, Any], card_id: str, fw_by_card: Mapping[str, str]) -> PendingKernelItem:
    is_new = entry["category"] == PENDING_KIND_TO_IMPLEMENT
    replaces_txt = " + ".join(dict.fromkeys(entry["replaces"])) if is_new else f"{entry['submodule']} -> {', '.join(sorted(entry['roles'])[:MAX_ROLES_IN_CHAIN])}"
    sources_txt = "; ".join(dict.fromkeys(s for s in entry["sources"] if s))[:MAX_SOURCE_REF_SUMMARY_LEN]
    arch_txt = "; ".join(dict.fromkeys(sorted(entry["archetypes"])))
    bef_ms, aft_ms, ls_ms = float(entry["before_ms"]), float(entry["after_ms"]), float(entry["lstar_ms"])
    b_ls = float(entry["before_lstar_ms"]) or ls_ms
    bef_att = min(100.0, 100.0 * b_ls / bef_ms) if bef_ms > 0.0 and b_ls > 0.0 else 0.0
    aft_att = min(100.0, 100.0 * ls_ms / aft_ms) if aft_ms > 0.0 and ls_ms > 0.0 else (100.0 if aft_ms <= 0.0 else 0.0)
    gain_ms, gain_upt = float(entry["gain_ms"]), float(entry["gain_upt"])
    is_prereq = bool(is_new and gain_ms <= CONSERVATION_EPS_MS)

    budget_part = f" | Core Budget: {entry['core_budget']}" if entry.get("core_budget") else ""
    microarch = (
        f"Before: {entry['before_bound']} (L*/Before={bef_att:.1f}%) -> After: {entry['after_bound']} (L*/After={aft_att:.1f}%) | "
        f"Single L*={entry['single_lstar_us']:.2f}us vs After={entry['single_sim_us']:.2f}us | {entry['die_summary']}{budget_part}"
    )
    kernel_receipt = " | ".join(p for p in (
        str(entry.get("sram_residency_summary") or entry.get("formula_summary") or ""),
        f"Exposed: {bef_ms:.3f}ms -> {aft_ms:.3f}ms (Δ={format_signed_gain_text(gain_ms)}, L*={ls_ms:.3f}ms, Bound={entry['after_bound']})",
        f"Cores: {entry['core_budget']}" if entry.get("core_budget") else "",
    ) if p)
    status_txt = str(entry["status"] or "")
    if is_prereq:
        status_txt = f"{status_txt} [实现前置条件 · {'flat' if abs(gain_ms) <= CONSERVATION_EPS_MS else 'regressive'}]".strip()
    fw_receipt = fw_by_card.get(card_id, "")
    if not fw_receipt and is_new:
        kspec = entry.get("synthesis_spec")
        n_mem = len(getattr(kspec, "constituent_op_ids", ()) or ()) or len(dict.fromkeys(entry["replaces"]))
        fw_receipt = f"FuseRegion ({n_mem} ops -> {entry['target_kernel']})"
    accept_receipt = (
        f"Receipt: Before={bef_ms:.3f}ms, After={aft_ms:.3f}ms, Δ={format_signed_gain_text(gain_ms)}, "
        f"Δ摊薄={gain_upt:.2f} us/(tok·die), L*={ls_ms:.3f}ms, Observed L*/After={aft_att:.1f}%"
    )
    return PendingKernelItem(
        priority=prio, kernel_key=str(entry["kernel_key"]), target_kernel=f"{entry['target_kernel']} [{entry['backend']}]",
        suggested_backend=str(entry["backend"]), status=status_txt, replaces_chain=replaces_txt,
        source_refs=sources_txt, archetype_summary=arch_txt, total_calls_per_step=int(entry["calls"]),
        expected_gain_ms=gain_ms, tensor_contract=str(entry["contract"]), microarch_target=microarch,
        default_design_slot=kernel_receipt, default_acceptance_slot=accept_receipt, category=str(entry["category"]),
        logical_role=", ".join(sorted(entry["roles"])[:2]) or str(entry["logical_role"]), submodule=str(entry["submodule"]),
        before_bound=str(entry["before_bound"]), after_bound=str(entry["after_bound"]),
        before_model_ms=bef_ms, after_model_ms=aft_ms, lstar_model_ms=ls_ms, gap_vs_lstar_ms=float(entry["gap_ms"]),
        card_id=card_id, config_howto="", kernel_action=kernel_receipt, framework_action=fw_receipt,
        before_attainment_pct=round(bef_att, 1), target_attainment_pct=round(aft_att, 1),
        opt_pillar=str(entry["opt_pillar"]), before_us_per_tok_die=float(entry["before_upt"]),
        after_us_per_tok_die=float(entry["after_upt"]), saved_us_per_tok_die=gain_upt,
        synthesis_spec=entry["synthesis_spec"], math_formula=str(entry["math_formula"] or ""),
        sram_residency_summary=str(entry["sram_residency_summary"] or ""),
        binding_state=str(entry.get("binding_state") or ""),
        source_row_indices=tuple(sorted(int(i) for i in entry.get("source_row_indices", ()))),
        core_budget=str(entry.get("core_budget") or ""),
        attributable_gain_ms=(float(entry["attr_gain_ms"]) if entry.get("has_split") else None),
        attributable_gain_us_per_tok_die=(float(entry["attr_gain_upt"]) if entry.get("has_split") else None),
        is_prerequisite=is_prereq, formula_summary=str(entry.get("formula_summary") or ""),
    )


def _passes_materiality_gate(
    saved_ms: float, saved_upt: float, *, min_saved_ms: float = NEW_KERNEL_MIN_SAVED_MS,
) -> bool:
    """Enforce the inclusive ``>= 0.050 ms`` and ``> 0 us/(tok*die)`` materiality gate."""
    return (
        float(saved_ms) >= float(min_saved_ms) - TUNE_THRESHOLD_TOL_MS
        and float(saved_upt) > 0.0
    )


def _build_framework_track_rows(
    aligned_rows: Sequence[AlignedComparisonRow], accounting: Sequence[GroupAccounting],
    after_bundle: ExportBundle, pending_items: Sequence[PendingKernelItem],
) -> tuple[list[ActionRoadmapRow], dict[str, str]]:
    fw_rows, fw_by_card = [], {}
    card_by_row = {idx: item for item in pending_items for idx in item.source_row_indices}
    seen_regions: set[tuple[str, tuple[int, ...]]] = set()

    for rec in after_bundle.applied_edit_receipts or ():
        if not isinstance(rec, Mapping) or str(rec.get("status", "applied")).lower() != "applied":
            continue
        opcode = str(rec.get("opcode") or "").strip()
        if opcode not in STRUCTURAL_EDIT_OPCODES:
            continue
        edit_id = sanitize_public_text(rec.get("edit_id") or opcode) or opcode
        in_ops = {str(x) for x in (rec.get("input_occurrences") or rec.get("affected_ops") or ()) if x}
        out_ops = {str(x) for x in (rec.get("output_occurrences") or ()) if x}
        matched = [
            i for i, row in enumerate(aligned_rows)
            if (in_ops and ({op.op_id for op in row.before_ops} & in_ops)) or (out_ops and ({op.op_id for op in row.after_ops} & out_ops))
        ]
        if not matched:
            continue
        if opcode in {"FuseRegion", "CollectiveTransform"} and len(matched) == 1:
            only_row = aligned_rows[matched[0]]
            if len(only_row.before_ops) == 1 and len(only_row.after_ops) == 1:
                continue
        if opcode == "FuseRegion":
            if not any(len(aligned_rows[i].before_ops) >= MIN_FUSION_MEMBERS and len(aligned_rows[i].after_ops) == 1 for i in matched):
                continue
            if all(accounting[i].bucket == "residual" for i in matched):
                continue
        rkey = (opcode, tuple(sorted(matched)))
        if rkey in seen_regions:
            continue
        seen_regions.add(rkey)
        saved_ms, saved_upt = sum(accounting[i].saved_ms for i in matched), sum(accounting[i].saved_upt for i in matched)
        if not _passes_materiality_gate(saved_ms, saved_upt):
            continue
        b_names = [sanitize_public_text(op.ascend_kernel) for i in matched for op in aligned_rows[i].before_ops]
        a_names = [sanitize_public_text(op.ascend_kernel) for i in matched for op in aligned_rows[i].after_ops]
        chain = f"{' + '.join(dict.fromkeys(b_names))} -> {' + '.join(dict.fromkeys(a_names)) or 'eliminated'}"
        linked = [card_by_row[i] for i in matched if i in card_by_row]
        detail = f"[{opcode}] {chain} (Δ={format_signed_gain_text(saved_ms)}, Δ摊薄={saved_upt:.2f} us/(tok·die))"
        for card in linked:
            fw_by_card.setdefault(card.card_id, detail)
        fw_rows.append(ActionRoadmapRow(
            track=TRACK_FRAMEWORK, track_label=THREE_TRACK_LABELS[TRACK_FRAMEWORK],
            title=f"{opcode}: {' + '.join(dict.fromkeys(a_names)) or edit_id}", expected_gain_ms=saved_ms,
            action_detail=detail, cross_ref=f"Sheet 4 Card #{linked[0].priority} ({linked[0].card_id})" if linked else f"Edit {edit_id}",
        ))

    if not fw_rows:
        for item in pending_items:
            if item.category == PENDING_KIND_TO_IMPLEMENT and _passes_materiality_gate(item.expected_gain_ms, item.saved_us_per_tok_die):
                detail = f"[FuseRegion] {item.replaces_chain} -> {item.target_kernel} (Δ={format_signed_gain_text(item.expected_gain_ms)})"
                fw_by_card.setdefault(item.card_id, detail)
                fw_rows.append(ActionRoadmapRow(
                    track=TRACK_FRAMEWORK, track_label=THREE_TRACK_LABELS[TRACK_FRAMEWORK],
                    title=f"FuseRegion: {item.target_kernel.split('[', 1)[0].strip()} ({item.submodule})",
                    expected_gain_ms=item.expected_gain_ms, action_detail=detail,
                    cross_ref=f"Sheet 4 Card #{item.priority} ({item.card_id})",
                ))
    fw_rows.sort(key=lambda r: (-(r.expected_gain_ms or 0.0), r.title))
    return fw_rows[:MAX_ROADMAP_FW_ITEMS], fw_by_card


def _extract_audit_issues(script_audit: Any) -> list[tuple[str, str, str, str]]:
    if script_audit is None:
        return []
    if hasattr(script_audit, "issues"):
        return [
            (
                sanitize_public_text(getattr(i, "title", "")),
                sanitize_public_text(getattr(i, "current_value", "")),
                sanitize_public_text(getattr(i, "recommended_value", "")),
                sanitize_public_text(getattr(i, "impact_note", "")),
            )
            for i in (getattr(script_audit, "issues") or ())
        ]
    if isinstance(script_audit, Mapping):
        return [
            (
                sanitize_public_text(r.get("title") or ""),
                sanitize_public_text(r.get("current") or ""),
                sanitize_public_text(r.get("recommended") or ""),
                sanitize_public_text(r.get("impact") or ""),
            )
            for r in (script_audit.get("issues_found") or ()) if isinstance(r, Mapping)
        ]
    return []


def _build_track_a_rows(before_bundle: ExportBundle, after_bundle: ExportBundle, *, is_comparison: bool, script_audit: Any) -> list[ActionRoadmapRow]:
    if script_audit is not None:
        return [
            ActionRoadmapRow(
                track=TRACK_CONFIG_ENV, track_label=THREE_TRACK_LABELS[TRACK_CONFIG_ENV],
                title=title, expected_gain_ms=None, action_detail=f"{cur_val} -> {rec_val} | {impact}",
                cross_ref="Script Audit / optimized_serve.sh", before_value=cur_val, after_value=rec_val,
            )
            for title, cur_val, rec_val, impact in _extract_audit_issues(script_audit)
        ]
    if is_comparison:
        return [
            ActionRoadmapRow(
                track=TRACK_CONFIG_ENV, track_label=THREE_TRACK_LABELS[TRACK_CONFIG_ENV],
                title=field_lbl, expected_gain_ms=None, action_detail=f"{field_lbl} · {bv} -> {av}",
                cross_ref="Sheet 1 Block 1", before_value=bv, after_value=av,
            )
            for field_lbl, bv, av in _extract_deployment_diffs(before_bundle, after_bundle)
        ]
    return []


def _qualify_action_groups(
    groups: Mapping[str, dict[str, Any]],
    *,
    is_comparison: bool,
    check_gap_when_single: bool,
    min_saved_ms: float = TUNE_MIN_SAVED_MS,
) -> tuple[list[dict[str, Any]], set[str]]:
    qualified: list[dict[str, Any]] = []
    qualified_keys: set[str] = set()
    threshold = float(min_saved_ms) - TUNE_THRESHOLD_TOL_MS
    for key, entry in groups.items():
        if is_comparison:
            net_ok = float(entry["gain_ms"]) >= threshold and float(entry["gain_upt"]) > 0.0
            split_ok = (not entry["has_split"]) or (
                float(entry["attr_gain_ms"]) >= threshold
                and float(entry["attr_gain_upt"]) > 0.0
            )
            if net_ok and split_ok:
                qualified.append(entry)
                qualified_keys.add(key)
        elif (not check_gap_when_single) or float(entry["gap_ms"]) >= threshold:
            qualified.append(entry)
            qualified_keys.add(key)
    return qualified, qualified_keys


def classify_waterfall_stage(
    acc: GroupAccounting,
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
) -> tuple[str, str | None]:
    """Classify one ``GroupAccounting`` row into canonical waterfall stage(s) via public config."""
    same_cfg = is_same_public_config(before_bundle, after_bundle)
    if acc.bucket == "config":
        if same_cfg or acc.reason in _FRAMEWORK_REASONS:
            return WF_STAGE_FRAMEWORK, None
        return WF_STAGE_CONFIG, None
    has_cfg_split = (
        not same_cfg
        and acc.after_pre_edit_ms is not None
        and abs(float(acc.config_share_ms)) > CONSERVATION_EPS_MS
    )
    if acc.bucket in {"new", "tune"}:
        return (WF_STAGE_CONFIG, WF_STAGE_KERNEL) if has_cfg_split else (WF_STAGE_KERNEL, None)
    if has_cfg_split:
        return WF_STAGE_CONFIG, WF_STAGE_RESIDUAL
    return WF_STAGE_RESIDUAL, None


def _classified_stage_ns(
    accounting: Sequence[GroupAccounting],
    waterfall: WaterfallSummary,
    before_bundle: ExportBundle,
    after: ExportBundle,
) -> tuple[int, int]:
    """Compute exact int-ns Sheet 4 ``[1] Config`` and ``[2] Framework`` deltas."""
    cfg_ns = 0
    fwk_ns = 0
    for acc in accounting:
        st_primary, st_secondary = classify_waterfall_stage(acc, before_bundle, after)
        if st_secondary is not None and acc.after_pre_edit_ms is not None:
            pre_ns = int(round(float(acc.config_share_ms) * float(NS_PER_MS)))
            if st_primary == WF_STAGE_CONFIG:
                cfg_ns += pre_ns
            elif st_primary == WF_STAGE_FRAMEWORK:
                fwk_ns += pre_ns
        else:
            saved_ns = int(round(float(acc.saved_ms) * float(NS_PER_MS)))
            if st_primary == WF_STAGE_CONFIG:
                cfg_ns += saved_ns
            elif st_primary == WF_STAGE_FRAMEWORK:
                fwk_ns += saved_ns
    same_pub_cfg = is_same_public_config(before_bundle, after)
    if not same_pub_cfg and (waterfall.idle_before_ns > 0 or waterfall.idle_after_ns > 0):
        cfg_ns += int(waterfall.idle_before_ns) - int(waterfall.idle_after_ns)
    return cfg_ns, fwk_ns


def _project_canonical_stage_waterfall(
    before_bundle: ExportBundle,
    after: ExportBundle,
    *,
    b_step_ns: int,
    a_step_ns: int,
    same_config: bool,
    accounting: Sequence[GroupAccounting] = (),
    waterfall: WaterfallSummary | None = None,
) -> tuple[dict[str, Any], ...]:
    """Project exact int-ns 4-stage waterfall aligned identically with Sheet 4 ``D3:D9``."""
    cfg_ns, fwk_ns = (
        _classified_stage_ns(accounting, waterfall, before_bundle, after)
        if waterfall is not None
        else (0, 0)
    )
    s1_ns = b_step_ns - cfg_ns
    s2_ns = s1_ns - fwk_ns
    receipt = getattr(after, "engine_step_receipt", None)
    if receipt is not None and getattr(receipt, "stages", None):
        stages = tuple(receipt.stages)
        spans_by_idx = (b_step_ns, s1_ns, s2_ns, a_step_ns)
        out: list[dict[str, Any]] = []
        prev_ns = b_step_ns
        for idx, st in enumerate(stages):
            st_ns = spans_by_idx[idx] if idx < len(spans_by_idx) else a_step_ns
            delta_ns = 0 if idx == 0 else int(prev_ns - st_ns)
            prev_ns = st_ns
            out.append({
                "stage_id": st.stage_id,
                "stack_track": st.stack_track,
                "status": st.status,
                "span_ns": st_ns,
                "delta_ns": delta_ns,
                "span_ms": float(st_ns) / float(NS_PER_MS),
                "delta_ms": float(delta_ns) / float(NS_PER_MS),
                "l_star_ns": int(st.l_star_ns),
                "l_star_valid": bool(st.l_star_valid),
                "idle_or_wait_ns": int(st.idle_or_wait_ns),
                "config_bridge_ns": cfg_ns if idx == 1 else 0,
            })
        return tuple(out)

    a_track = str(after.feature_track or "shipping").strip().lower()
    b_track = str(before_bundle.feature_track or "shipping").strip().lower()
    spans = (
        ("before_merged", b_track, b_step_ns, 0),
        ("stage1_ops_config", a_track if s1_ns != b_step_ns else b_track, s1_ns, b_step_ns - s1_ns),
        ("stage2_unmerged_pr", a_track if s2_ns != s1_ns else b_track, s2_ns, s1_ns - s2_ns),
        ("stage3_target_after", a_track, a_step_ns, s2_ns - a_step_ns),
    )
    return tuple(
        {
            "stage_id": sid,
            "stack_track": trk,
            "status": "committed",
            "span_ns": int(sp),
            "delta_ns": int(dl),
            "span_ms": float(sp) / float(NS_PER_MS),
            "delta_ms": float(dl) / float(NS_PER_MS),
        }
        for sid, trk, sp, dl in spans
    )


def _assemble_waterfall_and_accounting(
    prelim_rows: Sequence[Mapping[str, Any]],
    *,
    before_bundle: ExportBundle,
    after: ExportBundle,
    same_config: bool,
    b_ntok: float,
    a_ntok: float,
    qualified_new_keys: set[str],
    qualified_tune_keys: set[str],
    card_id_by_key: Mapping[str, str],
    displayed_keys: set[str],
) -> tuple[list[GroupAccounting], WaterfallSummary, int, int]:
    accounting: list[GroupAccounting] = []
    totals_ns = {k: 0 for k in ("cfg", "new", "tune", "res", "s2_b", "s2_a")}
    totals_upt = {k: 0.0 for k in ("cfg", "new", "tune", "res")}
    for rec in prelim_rows:
        bucket: Literal["config", "new", "tune", "residual"] = rec["cand_bucket"]
        gkey = rec["group_key"]
        if bucket == "new" and gkey not in qualified_new_keys:
            bucket, reason, cid, cat = "residual", "below_materiality_threshold", "", PENDING_KIND_AT_FLOOR
            if rec["pre_edit_ms"] is not None:
                cfg_ns, cfg_ms, cfg_upt = rec["cfg_ns"], rec["cfg_ms"], rec["cfg_upt"]
                kern_ns, kern_ms, kern_upt = rec["kern_ns"], rec["kern_ms"], rec["kern_upt"]
            else:
                cfg_ns, cfg_ms, cfg_upt, kern_ns, kern_ms, kern_upt = 0, 0.0, 0.0, rec["saved_ns"], rec["saved_ms"], rec["saved_upt"]
        elif bucket == "tune" and gkey not in qualified_tune_keys:
            bucket, reason, cid, cat = "residual", "below_tuning_threshold", "", PENDING_KIND_AT_FLOOR
            if rec["pre_edit_ms"] is not None:
                cfg_ns, cfg_ms, cfg_upt = rec["cfg_ns"], rec["cfg_ms"], rec["cfg_upt"]
                kern_ns, kern_ms, kern_upt = rec["kern_ns"], rec["kern_ms"], rec["kern_upt"]
            else:
                cfg_ns, cfg_ms, cfg_upt, kern_ns, kern_ms, kern_upt = 0, 0.0, 0.0, rec["saved_ns"], rec["saved_ms"], rec["saved_upt"]
        else:
            cfg_ns, kern_ns = rec["cfg_ns"], rec["kern_ns"]
            cfg_ms, cfg_upt, kern_ms, kern_upt, reason = rec["cfg_ms"], rec["cfg_upt"], rec["kern_ms"], rec["kern_upt"], rec["reason"]
            cid = card_id_by_key.get(gkey, "") if gkey in displayed_keys else ""
            cat = PENDING_KIND_TO_IMPLEMENT if bucket == "new" else (PENDING_KIND_TO_OPTIMIZE if bucket == "tune" else PENDING_KIND_AT_FLOOR)
        accounting.append(GroupAccounting(
            row_index=rec["row_index"], bucket=bucket, card_id=cid, category=cat, before_ms=rec["b_ms"], after_ms=rec["a_ms"],
            saved_ms=rec["saved_ms"], before_upt=rec["b_upt"], after_upt=rec["a_upt"], saved_upt=rec["saved_upt"],
            after_pre_edit_ms=rec["pre_edit_ms"] if (bucket in {"new", "tune"} or cfg_ns != 0) else None,
            config_share_ms=cfg_ms, config_share_upt=cfg_upt, kernel_share_ms=kern_ms, kernel_share_upt=kern_upt, reason=reason,
        ))
        totals_ns["s2_b"] += int(rec["b_ns"])
        totals_ns["s2_a"] += int(rec["a_ns"])
        totals_ns["cfg"] += int(cfg_ns)
        totals_upt["cfg"] += cfg_upt
        if bucket in {"new", "tune", "residual"}:
            pfx = "res" if bucket == "residual" else bucket
            totals_ns[pfx] += int(kern_ns)
            totals_upt[pfx] += kern_upt

    b_step_ns, b_idle_ns = _bundle_step_and_idle_ns(before_bundle)
    a_step_ns, a_idle_ns = _bundle_step_and_idle_ns(after)
    idle_b_ns = max(0, b_step_ns - totals_ns["s2_b"]) if b_step_ns >= totals_ns["s2_b"] else b_idle_ns
    idle_a_ns = max(0, a_step_ns - totals_ns["s2_a"]) if a_step_ns >= totals_ns["s2_a"] else a_idle_ns
    totals_ns["s2_b"] += idle_b_ns
    totals_ns["s2_a"] += idle_a_ns
    idle_saved_ns = idle_b_ns - idle_a_ns
    idle_saved_upt = (float(idle_b_ns) / (NS_PER_US * b_ntok)) - (float(idle_a_ns) / (NS_PER_US * a_ntok))
    if not same_config:
        totals_ns["cfg"] += idle_saved_ns
        totals_upt["cfg"] += idle_saved_upt
    else:
        totals_ns["res"] += idle_saved_ns
        totals_upt["res"] += idle_saved_upt

    tot_saved_ns = totals_ns["cfg"] + totals_ns["new"] + totals_ns["tune"] + totals_ns["res"]
    recon_after_ns = b_step_ns - tot_saved_ns
    recon_err_ns = recon_after_ns - a_step_ns
    bef_step_ms, aft_step_ms = float(b_step_ns) / float(NS_PER_MS), float(a_step_ns) / float(NS_PER_MS)
    bef_step_upt, aft_step_upt = float(b_step_ns) / (NS_PER_US * b_ntok), float(a_step_ns) / (NS_PER_US * a_ntok)
    tot_saved_upt = totals_upt["cfg"] + totals_upt["new"] + totals_upt["tune"] + totals_upt["res"]
    waterfall = WaterfallSummary(
        before_step_ms=bef_step_ms, after_step_ms=aft_step_ms, before_upt=bef_step_upt, after_upt=aft_step_upt,
        config_saved_ms=float(totals_ns["cfg"]) / float(NS_PER_MS), config_saved_upt=totals_upt["cfg"],
        new_kernel_saved_ms=float(totals_ns["new"]) / float(NS_PER_MS), new_kernel_saved_upt=totals_upt["new"],
        tune_kernel_saved_ms=float(totals_ns["tune"]) / float(NS_PER_MS), tune_kernel_saved_upt=totals_upt["tune"],
        residual_saved_ms=float(totals_ns["res"]) / float(NS_PER_MS), residual_saved_upt=totals_upt["res"],
        reconstructed_after_ms=float(recon_after_ns) / float(NS_PER_MS),
        reconciliation_error_ms=float(recon_err_ns) / float(NS_PER_MS),
        reconciliation_error_upt=tot_saved_upt - (bef_step_upt - aft_step_upt),
        sheet2_before_total_ms=float(totals_ns["s2_b"]) / float(NS_PER_MS),
        sheet2_after_total_ms=float(totals_ns["s2_a"]) / float(NS_PER_MS),
        before_endpoint_error_ms=float(totals_ns["s2_b"] - b_step_ns) / float(NS_PER_MS),
        after_endpoint_error_ms=float(totals_ns["s2_a"] - a_step_ns) / float(NS_PER_MS),
        before_step_ns=b_step_ns, after_step_ns=a_step_ns,
        idle_before_ns=idle_b_ns, idle_after_ns=idle_a_ns,
        reconciliation_error_ns=recon_err_ns,
    )
    return accounting, waterfall, b_step_ns, a_step_ns


def build_comparison_plan(
    before_bundle: ExportBundle, after_bundle: ExportBundle | None, aligned_rows: Sequence[AlignedComparisonRow],
    *, top_k_optimize: int = MAX_DISPLAY_CARDS, include_to_optimize: bool = True, script_audit: Any = None,
) -> ComparisonPlan:
    """Build single source-of-truth Waterfall, classified cards, and 3-track roadmap."""
    is_comparison = after_bundle is not None
    after = after_bundle or before_bundle
    b_ntok, a_ntok = max(float(before_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS), max(float(after.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    same_config = (not is_comparison) or _is_same_deployment_config(before_bundle, after)
    arch_tag, fuse_outputs = _arch_tag_from_bundle(after), _applied_fuse_outputs(after)
    rewrite_verified = bool(getattr(after, "rewrite_verification_passed", False))

    prelim_rows: list[dict[str, Any]] = []
    new_groups: dict[str, dict[str, Any]] = {}
    tune_groups: dict[str, dict[str, Any]] = {}
    for r_idx, row in enumerate(aligned_rows):
        b_ns, a_ns, saved_ns, b_ms, a_ms, saved_ms, b_upt, a_upt, saved_upt = _row_exposed_metrics_ns(row, b_ntok, a_ntok)
        pre_edit_ms = _after_pre_edit_exposed_ms(row) if (is_comparison and not same_config) else None
        cand_bucket, a_op, reason = _prelim_classify_row(
            row, is_comparison=is_comparison, same_config=same_config,
            saved_ms=saved_ms, saved_upt=saved_upt, after_pre_edit_ms=pre_edit_ms,
            fuse_outputs=fuse_outputs, rewrite_verified=rewrite_verified,
        )
        if pre_edit_ms is not None and cand_bucket in {"new", "tune"}:
            pre_edit_ns = max(0, int(round(pre_edit_ms * NS_PER_MS)))
            cfg_ns, kern_ns = b_ns - pre_edit_ns, saved_ns - (b_ns - pre_edit_ns)
            cfg_ms, cfg_upt = float(cfg_ns) / float(NS_PER_MS), b_upt - (float(pre_edit_ns) / (NS_PER_US * a_ntok))
            kern_ms, kern_upt, has_split = float(kern_ns) / float(NS_PER_MS), saved_upt - cfg_upt, True
        elif cand_bucket == "config":
            cfg_ns, kern_ns, cfg_ms, cfg_upt, kern_ms, kern_upt, has_split = saved_ns, 0, saved_ms, saved_upt, 0.0, 0.0, False
        else:
            cfg_ns, kern_ns, cfg_ms, cfg_upt, kern_ms, kern_upt, has_split = 0, saved_ns, 0.0, 0.0, saved_ms, saved_upt, False

        group_key = ""
        if cand_bucket == "new" and a_op is not None:
            group_key, stem = _canonical_abi_key("new", a_op, arch_tag)
            _add_row_to_group(
                new_groups, key=group_key, stem=stem, category=PENDING_KIND_TO_IMPLEMENT, row=row, row_idx=r_idx,
                a_op=a_op, after_bundle=after, b_ms=b_ms, a_ms=a_ms, saved_ms=saved_ms, b_upt=b_upt, a_upt=a_upt,
                saved_upt=saved_upt, kernel_share_ms=kern_ms, kernel_share_upt=kern_upt, has_split=has_split, is_comparison=is_comparison,
            )
        elif cand_bucket == "tune" and include_to_optimize and a_op is not None:
            group_key, stem = _canonical_abi_key("tune", a_op, arch_tag)
            _add_row_to_group(
                tune_groups, key=group_key, stem=stem, category=PENDING_KIND_TO_OPTIMIZE, row=row, row_idx=r_idx,
                a_op=a_op, after_bundle=after, b_ms=b_ms, a_ms=a_ms, saved_ms=saved_ms, b_upt=b_upt, a_upt=a_upt,
                saved_upt=saved_upt, kernel_share_ms=kern_ms, kernel_share_upt=kern_upt, has_split=has_split, is_comparison=is_comparison,
            )
        elif cand_bucket == "tune" and not include_to_optimize:
            cand_bucket, reason = "residual", "tuning_excluded"
        prelim_rows.append({
            "row_index": r_idx, "cand_bucket": cand_bucket, "group_key": group_key,
            "b_ns": b_ns, "a_ns": a_ns, "saved_ns": saved_ns, "cfg_ns": cfg_ns, "kern_ns": kern_ns,
            "b_ms": b_ms, "a_ms": a_ms, "saved_ms": saved_ms, "b_upt": b_upt, "a_upt": a_upt,
            "saved_upt": saved_upt, "pre_edit_ms": pre_edit_ms,
            "cfg_ms": cfg_ms, "cfg_upt": cfg_upt, "kern_ms": kern_ms, "kern_upt": kern_upt, "reason": reason,
        })

    qualified_new, qualified_new_keys = _qualify_action_groups(
        new_groups, is_comparison=is_comparison, check_gap_when_single=False, min_saved_ms=NEW_KERNEL_MIN_SAVED_MS,
    )
    qualified_tune, qualified_tune_keys = _qualify_action_groups(
        tune_groups, is_comparison=is_comparison, check_gap_when_single=True, min_saved_ms=TUNE_MIN_SAVED_MS,
    )
    card_id_by_key = _assign_deterministic_card_ids([*qualified_new, *qualified_tune])
    ordered_new = sorted(qualified_new, key=lambda e: (-float(e["gain_upt"]), -float(e["gain_ms"]), card_id_by_key[e["kernel_key"]]))
    ordered_tune = sorted(
        qualified_tune,
        key=(lambda e: (-float(e["gain_upt"]), -float(e["gain_ms"]), card_id_by_key[e["kernel_key"]])) if is_comparison else (lambda e: (-float(e["gap_ms"]), -int(e["calls"]), card_id_by_key[e["kernel_key"]])),
    )
    cap = max(0, min(MAX_DISPLAY_CARDS, int(top_k_optimize))) if not is_comparison else MAX_DISPLAY_CARDS
    displayed_entries = [*ordered_new, *ordered_tune][:cap]
    displayed_keys = {e["kernel_key"] for e in displayed_entries}
    accounting, waterfall, b_step_ns, a_step_ns = _assemble_waterfall_and_accounting(
        prelim_rows, before_bundle=before_bundle, after=after, same_config=same_config,
        b_ntok=b_ntok, a_ntok=a_ntok, qualified_new_keys=qualified_new_keys,
        qualified_tune_keys=qualified_tune_keys, card_id_by_key=card_id_by_key, displayed_keys=displayed_keys,
    )

    prelim_cards = tuple(_materialize_card(p, e, card_id_by_key[e["kernel_key"]], {}) for p, e in enumerate(displayed_entries, 1))
    fw_rows, fw_by_card = _build_framework_track_rows(aligned_rows, accounting, after, prelim_cards)
    final_cards = tuple(_materialize_card(p, e, card_id_by_key[e["kernel_key"]], fw_by_card) for p, e in enumerate(displayed_entries, 1))
    roadmap = _build_track_a_rows(before_bundle, after, is_comparison=is_comparison, script_audit=script_audit) + fw_rows + [
        ActionRoadmapRow(
            track=TRACK_KERNEL, track_label=f"{THREE_TRACK_LABELS[TRACK_KERNEL]} · {item.category}",
            title=f"{item.target_kernel} ({item.before_model_ms:.3f} -> {item.after_model_ms:.3f} ms)",
            expected_gain_ms=item.expected_gain_ms, action_detail=item.kernel_action,
            cross_ref=f"Sheet 4 Card #{item.priority} ({item.card_id})",
        )
        for item in final_cards[:MAX_ROADMAP_KERNEL_ITEMS]
    ]
    anchored_stage_wf = (
        _project_canonical_stage_waterfall(
            before_bundle, after, b_step_ns=b_step_ns, a_step_ns=a_step_ns, same_config=same_config,
            accounting=accounting, waterfall=waterfall,
        )
        if is_comparison
        else tuple(dict(item) for item in (before_bundle.stage_waterfall or ()) if isinstance(item, Mapping))
    )
    return ComparisonPlan(
        aligned_rows=tuple(aligned_rows), group_accounting=tuple(accounting), waterfall=waterfall,
        pending_items=final_cards, roadmap=tuple(roadmap),
        codesign_note=build_parallelism_codesign_note(before_bundle, after) if is_comparison else "",
        stage_waterfall=anchored_stage_wf,
    )


def build_pending_kernel_details(
    aligned_rows: Sequence[AlignedComparisonRow], after_bundle: ExportBundle, *,
    before_bundle: ExportBundle | None = None, top_k_optimize: int = DEFAULT_TOP_OPTIMIZE_KERNELS, include_to_optimize: bool = True,
) -> tuple[PendingKernelItem, ...]:
    """Build high-signal Sheet 4 cards via canonical D3 classification."""
    return build_comparison_plan(
        before_bundle if before_bundle is not None else after_bundle,
        after_bundle if before_bundle is not None else None,
        aligned_rows, top_k_optimize=top_k_optimize, include_to_optimize=include_to_optimize,
    ).pending_items


def build_new_kernel_backlog(
    aligned_rows: Sequence[AlignedComparisonRow], after_bundle: ExportBundle, *, before_bundle: ExportBundle | None = None,
) -> tuple[NewKernelBacklogItem, ...]:
    """Aggregate ``[需新开发融合算子]`` items into Sheet 4 backlog."""
    return build_pending_kernel_details(aligned_rows, after_bundle, before_bundle=before_bundle, include_to_optimize=False)


def build_actionable_roadmap(
    before_bundle: ExportBundle, after_bundle: ExportBundle, pending_items: Sequence[PendingKernelItem] = (),
    *, aligned_rows: Sequence[AlignedComparisonRow] | None = None, script_audit: Any = None,
) -> tuple[ActionRoadmapRow, ...]:
    """Build receipt-driven 3-track roadmap (script-free diff or explicit audit)."""
    if aligned_rows is not None:
        return build_comparison_plan(
            before_bundle, after_bundle if after_bundle is not before_bundle else None, aligned_rows, script_audit=script_audit,
        ).roadmap
    roadmap = _build_track_a_rows(before_bundle, after_bundle, is_comparison=(after_bundle is not before_bundle), script_audit=script_audit)
    for item in pending_items:
        if item.category == PENDING_KIND_TO_IMPLEMENT and _passes_materiality_gate(item.expected_gain_ms, item.saved_us_per_tok_die):
            roadmap.append(ActionRoadmapRow(
                track=TRACK_FRAMEWORK, track_label=THREE_TRACK_LABELS[TRACK_FRAMEWORK],
                title=f"FuseRegion: {item.target_kernel.split('[', 1)[0].strip()} ({item.submodule})",
                expected_gain_ms=item.expected_gain_ms,
                action_detail=item.framework_action or f"[FuseRegion] {item.replaces_chain} -> {item.target_kernel}",
                cross_ref=f"Sheet 4 Card #{item.priority} ({item.card_id})",
            ))
    for item in pending_items[:MAX_ROADMAP_KERNEL_ITEMS]:
        roadmap.append(ActionRoadmapRow(
            track=TRACK_KERNEL, track_label=f"{THREE_TRACK_LABELS[TRACK_KERNEL]} · {item.category}",
            title=f"{item.target_kernel} ({item.before_model_ms:.3f} -> {item.after_model_ms:.3f} ms)",
            expected_gain_ms=item.expected_gain_ms, action_detail=item.kernel_action,
            cross_ref=f"Sheet 4 Card #{item.priority} ({item.card_id})",
        ))
    return tuple(roadmap)


__all__ = [
    "CONSERVATION_EPS_MS", "MAX_DISPLAY_CARDS", "NEW_KERNEL_MIN_SAVED_MS", "TUNE_MIN_SAVED_MS",
    "WF_STAGE_CONFIG", "WF_STAGE_FRAMEWORK", "WF_STAGE_KERNEL", "WF_STAGE_RESIDUAL",
    "ComparisonPlan", "GroupAccounting", "WaterfallSummary",
    "build_actionable_roadmap", "build_comparison_plan", "build_new_kernel_backlog",
    "build_parallelism_codesign_note", "build_pending_kernel_details",
    "classify_waterfall_stage", "format_signed_gain_text",
]
