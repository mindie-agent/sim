"""Compact (< 8 KB) ``agent_brief.json`` builder, budget compactor & Markdown digest.

[Role] Project ``ExportBundle`` comparisons and ``ComparisonPlan`` receipts into
  a deterministic ``< 8 KB`` allowlist-only JSON brief (atomic same-directory
  persistence) and crisp ~25-line Markdown executive digest sharing the exact
  same Waterfall, cards, and 3-track roadmap rendered in the OOXML workbook.
[Invariants] Strictly enforces ``len(utf8_bytes) < 8192`` including trailing LF,
  routes BOTH bundle and mapping overloads plus Markdown digest through
  ``public_view.project_agent_brief_mapping`` (zero unknown keys, zero raw
  internal track enums, zero ``str(dict)`` leaks), preserves full-precision KPIs
  and Waterfall reconciliation fields, omits ``script_audit`` and shell
  env-switch nags in script-free mode, and never swallows ``TimeoutError`` under
  ``except OSError:``.
[Timeouts & Locks] Pure O(Archetypes) in-memory projection; atomic ``os.replace``
  only inside ``export_agent_brief``.
[Side-Effect Policy] Writes only the caller-requested brief JSON path.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from mindie_sim.schema.common import ModelingError

from .action_plan import (
    CONSERVATION_EPS_MS,
    ComparisonPlan,
    build_comparison_plan,
)
from .models import (
    AGENT_BRIEF_SCHEMA,
    BINDING_STATE_BOUND,
    GIB_BYTES,
    KIB_BYTES,
    MACRO_L1_HIERARCHY,
    MODE_MODELING,
    PENDING_KIND_TO_IMPLEMENT,
    TRACK_CONFIG_ENV,
    TRACK_FRAMEWORK,
    TRACK_KERNEL,
    US_PER_MS,
    AlignedComparisonRow,
    ExportBundle,
    MacroModuleRow,
    format_binding_badge,
)
from .public_view import (
    THREE_TRACK_LABELS,
    format_track_label,
    project_agent_brief_mapping,
    project_artifacts,
    project_codesign_opportunities,
    project_pd_pool,
    project_recommended_checkpoint,
    project_script_audit,
    project_serving_rules,
    sanitize_public_text,
    sanitize_source_ref,
)

#: Maximum number of compact kernel tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_KERNEL_TASKS = 5
#: Maximum number of framework/config tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_FRAMEWORK_TASKS = 3
#: Maximum number of synthesis specs, co-design opportunities & audit triggers embedded.
BRIEF_MAX_TRIGGERS = 4
BRIEF_MAX_SPECS = 4
BRIEF_MAX_CODESIGN_OPPS = 4
BRIEF_MAX_CAPTURE_SIZES = 8
#: Maximum character lengths for compact string fields inside ``agent_brief.json``.
BRIEF_TEXT_LIMIT = 72
BRIEF_TITLE_LIMIT = 44
BRIEF_KERNEL_LIMIT = 40
BRIEF_SUMMARY_LIMIT = 60
BRIEF_MATH_LIMIT = 88
BRIEF_CMD_LIMIT = 92
BRIEF_CODESIGN_LIMIT = 140
BRIEF_MAX_BLOCKED_OPS = 4
BRIEF_MAX_REFS = 2
BRIEF_DIGEST_MAX_CFG = 2
BRIEF_DIGEST_MAX_FW = 2
BRIEF_DIGEST_MAX_KERN = 3
#: Byte budget for ``agent_brief.json`` (< 8 KB) and pretty-print threshold.
AGENT_BRIEF_MAX_BYTES = 8192
AGENT_BRIEF_PRETTY_LIMIT_BYTES = 6400


def _trim_text(text: Any, limit: int = BRIEF_TEXT_LIMIT) -> str:
    clean = sanitize_public_text(text)
    return clean if len(clean) <= limit else (clean[: max(0, limit - 3)] + "...")


def _bundle_track_label(bundle: ExportBundle) -> str:
    raw = bundle.feature_track or (
        bundle.effective_config.get("feature_track")
        if isinstance(bundle.effective_config, Mapping) else ""
    )
    if not raw and bundle.target_envelope_applied:
        raw = "target"
    return format_track_label(raw or "shipping")


def _bundle_kpi_dict(bundle: ExportBundle) -> dict[str, Any]:
    step_ms = bundle.step_span_us / US_PER_MS
    is_decode = bundle.phase == "decode"
    acc = max(1.0, bundle.avg_accepted_tokens)
    handoff_ms = float(getattr(bundle, "handoff_tpot_ms", 0.0) or 0.0) if is_decode else 0.0
    tpot_ms = ((step_ms / acc) + handoff_ms) if is_decode else step_ms
    sla = bundle.tpot_slo_ms if is_decode else bundle.ttft_slo_ms
    l_star_ms = bundle.step_l_star_lean_us / US_PER_MS
    pool = project_pd_pool(bundle.pd_pool_summary) or {}
    cluster_tps = pool.get("e2e_total_tps")
    if cluster_tps is None:
        cluster_tps = bundle.throughput_tokens_per_s
    return {
        "analysis_mode": bundle.analysis_mode,
        "track_label": _bundle_track_label(bundle),
        "binding_state": bundle.binding_state,
        "has_unbound_formula_ops": bool(bundle.has_unbound_formula_ops),
        "step_ms": float(step_ms),
        "handoff_tpot_ms": float(handoff_ms),
        "effective_tpot_ms": float(tpot_ms),
        "sla_target_ms": (float(sla) if sla is not None else None),
        "sla_verdict": ("UNCONSTRAINED" if sla is None else ("PASS" if tpot_ms <= sla else "FAIL")),
        "sla_regime": bundle.sla_regime,
        "sla_slack_ms": (float(sla - tpot_ms) if sla is not None else None),
        "n_tok_per_die": float(bundle.n_tok_per_die),
        "us_per_tok_die": float(bundle.us_per_tok_die),
        "single_die_tps": float(bundle.throughput_per_die_tokens_per_s),
        "cluster_tps": float(cluster_tps),
        "l_star_ms": float(l_star_ms),
        "l_star_attainment_pct": round((100.0 * l_star_ms / step_ms) if step_ms > 0.0 else 0.0, 1),
        "cube_util_pct": round(bundle.network_cube_mfu, 1),
        "vector_util_pct": round(bundle.network_vec_mfu, 1),
        "hbm_mbu_pct": round(bundle.network_mbu, 1),
        "hccl_link_util_pct": round(bundle.network_link_util, 1),
    }


def _brief_utf8_len(payload: Mapping[str, Any]) -> int:
    return len((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def _truncate_utf8_bytes(text: str, max_bytes: int) -> str:
    """Truncate ``text`` strictly below or equal to ``max_bytes`` UTF-8 bytes."""
    raw = str(text)
    if max_bytes <= 0:
        return ""
    encoded = raw.encode("utf-8")
    if len(encoded) <= max_bytes:
        return raw
    ellipsis = "..."
    ell_b = len(ellipsis)
    if max_bytes <= ell_b:
        return encoded[:max_bytes].decode("utf-8", errors="ignore")
    body = encoded[: max_bytes - ell_b].decode("utf-8", errors="ignore")
    return body + ellipsis


def _shrink_value_for_budget(val: Any, excess_bytes: int) -> Any:
    """Shrink one oversized value by at least ``excess_bytes`` UTF-8 bytes."""
    if isinstance(val, list):
        return val[: len(val) // 2] if len(val) > 1 else []
    if isinstance(val, Mapping):
        if len(val) > 1:
            items = list(val.items())
            return dict(items[: len(items) // 2])
        if len(val) == 1:
            k_only, v_only = next(iter(val.items()))
            shrunk = _shrink_value_for_budget(v_only, excess_bytes)
            return {} if shrunk == v_only else {k_only: shrunk}
        return {}
    if isinstance(val, str):
        cur_bytes = len(val.encode("utf-8"))
        target_bytes = max(0, min(cur_bytes // 2, cur_bytes - max(1, excess_bytes)))
        return _truncate_utf8_bytes(val, target_bytes)
    return None


def _trim_keys_to_budget(trimmed: dict[str, Any], protected: frozenset[str]) -> None:
    while _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        excess = _brief_utf8_len(trimmed) - AGENT_BRIEF_MAX_BYTES + 1
        candidates = [
            (len(json.dumps(val, ensure_ascii=False, separators=(",", ":")).encode("utf-8")), key)
            for key, val in trimmed.items()
            if key not in protected and val not in (None, "", [], {})
        ]
        if not candidates:
            break
        _, largest_key = max(candidates)
        cur_val = trimmed[largest_key]
        shrunk = _shrink_value_for_budget(cur_val, excess)
        if shrunk is None or shrunk == cur_val:
            trimmed.pop(largest_key, None)
        else:
            trimmed[largest_key] = shrunk


def _compact_agent_brief_to_budget(brief: dict[str, Any]) -> dict[str, Any]:
    """Guarantee ``agent_brief.json`` stays strictly below ``AGENT_BRIEF_MAX_BYTES`` (8 KB)."""
    if _brief_utf8_len(brief) < AGENT_BRIEF_MAX_BYTES:
        return brief
    trimmed = dict(brief)
    for key in ("codesign_opportunities", "kernel_synthesis_specs", "semantic_audit_triggers"):
        val = trimmed.get(key)
        if isinstance(val, list) and len(val) > 2 and _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
            trimmed[key] = val[:2]
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES and isinstance(trimmed.get("va_serving_rules"), Mapping):
        va_copy = dict(trimmed["va_serving_rules"])
        va_copy.pop("hw_isa_guidance", None)
        if isinstance(va_copy.get("capture_sizes"), list):
            va_copy["capture_sizes"] = va_copy["capture_sizes"][:BRIEF_MAX_CAPTURE_SIZES]
        trimmed["va_serving_rules"] = va_copy
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        trimmed["module_hierarchy"] = [
            {k: v for k, v in grp.items() if k != "children"}
            for grp in trimmed.get("module_hierarchy", ())
        ]
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        for key in ("codesign_opportunities", "kernel_synthesis_specs", "semantic_audit_triggers"):
            val = trimmed.get(key)
            if isinstance(val, list) and len(val) > 1:
                trimmed[key] = val[:1]
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES and isinstance(trimmed.get("scenario_and_parallelism"), Mapping):
        sc = dict(trimmed["scenario_and_parallelism"])
        if sc.get("module_parallel_domains"):
            sc["module_parallel_domains"] = _trim_text(str(sc["module_parallel_domains"]), BRIEF_TEXT_LIMIT)
        sc.pop("recommended_checkpoint", None)
        trimmed["scenario_and_parallelism"] = sc
    for drop_key in (
        "codesign_opportunities", "va_serving_rules", "kernel_synthesis_specs",
        "semantic_audit_triggers", "macro_wall_trigger", "generated_files",
    ):
        if _brief_utf8_len(trimmed) < AGENT_BRIEF_MAX_BYTES:
            break
        if drop_key in trimmed:
            trimmed[drop_key] = (
                [] if isinstance(trimmed[drop_key], list)
                else ({} if isinstance(trimmed[drop_key], Mapping) else None)
            )
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES and isinstance(trimmed.get("action_plan_3_tracks"), Mapping):
        plan = dict(trimmed["action_plan_3_tracks"])
        plan.pop("track_labels", None)
        for list_key in ("kernel_tasks", "framework_tasks", "config_and_env"):
            items = plan.get(list_key)
            if isinstance(items, list) and len(items) > 2:
                plan[list_key] = items[:2]
        plan["parallelism_codesign_note"] = _trim_text(str(plan.get("parallelism_codesign_note") or ""), BRIEF_TEXT_LIMIT)
        trimmed["action_plan_3_tracks"] = plan
    _trim_keys_to_budget(trimmed, frozenset({"schema", "summary", "kpis_before_vs_after"}))
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        _trim_keys_to_budget(trimmed, frozenset({"schema", "summary"}))
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        _trim_keys_to_budget(trimmed, frozenset({"schema"}))
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        _trim_keys_to_budget(trimmed, frozenset())
    return trimmed


def _factual_wall_summary(bundle: ExportBundle) -> dict[str, Any] | None:
    wall = bundle.macro_wall_trigger
    if wall is None:
        return None
    return {
        "wall_kind": str(wall.wall_kind),
        "tpot_ms": round(float(wall.tpot_ms), 3),
        "tpot_slo_ms": round(float(wall.tpot_slo_ms), 3),
        "sla_slack_ms": round(float(wall.sla_slack_ms), 3),
        "hbm_headroom_gib": round(float(wall.hbm_headroom_gib), 2),
        "hbm_utilization_pct": round(float(wall.hbm_utilization_pct), 1),
    }


def _factual_serving_facts(bundle: ExportBundle) -> dict[str, Any] | None:
    return project_serving_rules(bundle.va_serving_rules_summary)


def _build_brief_l1_groups(macro_rows: Sequence[MacroModuleRow]) -> list[dict[str, Any]]:
    l1_groups: list[dict[str, Any]] = []
    by_l1: dict[str, list[MacroModuleRow]] = defaultdict(list)
    for row in macro_rows:
        by_l1[row.l1_group].append(row)
    tot_aft = max(sum(r.after_ms for r in macro_rows), CONSERVATION_EPS_MS)
    for l1_name, _ in MACRO_L1_HIERARCHY:
        children = by_l1.get(l1_name)
        if not children:
            continue
        l1_bef = sum(c.before_ms for c in children)
        l1_aft = sum(c.after_ms for c in children)
        l1_groups.append({
            "l1": l1_name.split(" (", 1)[0],
            "before_ms": round(l1_bef, 3),
            "after_ms": round(l1_aft, 3),
            "share_pct": round(100.0 * l1_aft / tot_aft, 1),
            "children": [
                {
                    "l2": c.module_name.split(" (", 1)[0],
                    "exposed_ms": round(c.after_ms, 3),
                    "before_ms": round(c.before_ms, 3),
                    "saved_us_per_tok_die": round(c.saved_us_per_tok_die, 2),
                }
                for c in children
            ],
        })
    return l1_groups


def _build_brief_action_items(
    roadmap: Sequence[Any],
    pending: Sequence[Any],
    *,
    is_script_audit: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg_items: list[dict[str, Any]] = []
    for r in roadmap:
        if r.track != TRACK_CONFIG_ENV:
            continue
        item = {
            "title": _trim_text(r.title, BRIEF_TITLE_LIMIT),
            "before": _trim_text(r.before_value, BRIEF_TITLE_LIMIT),
            "after": _trim_text(r.after_value, BRIEF_CMD_LIMIT if is_script_audit else BRIEF_TITLE_LIMIT),
        }
        if is_script_audit:
            item["cmd"] = _trim_text(r.action_detail, BRIEF_CMD_LIMIT)
        cfg_items.append(item)
    fw_items = [
        {
            "title": _trim_text(r.title, BRIEF_TITLE_LIMIT),
            "saved_ms": round(float(r.expected_gain_ms or 0.0), 3),
            "work": _trim_text(r.action_detail, BRIEF_TEXT_LIMIT),
        }
        for r in roadmap if r.track == TRACK_FRAMEWORK
    ][:BRIEF_MAX_FRAMEWORK_TASKS]
    kern_items = [
        {
            "card_id": sanitize_public_text(item.card_id),
            "prio": f"P{item.priority}",
            "cat": "new" if item.category == PENDING_KIND_TO_IMPLEMENT else "tune",
            "kernel": _trim_text(item.target_kernel, BRIEF_KERNEL_LIMIT),
            "backend": sanitize_public_text(item.suggested_backend),
            "submodule": _trim_text(item.submodule, BRIEF_KERNEL_LIMIT),
            "saved_ms": round(item.expected_gain_ms, 3),
            "saved_us_per_tok_die": round(item.saved_us_per_tok_die, 2),
            "before_ms": round(item.before_model_ms, 3),
            "after_ms": round(item.after_model_ms, 3),
            "binding": item.binding_state,
            "summary": _trim_text(item.kernel_action, BRIEF_SUMMARY_LIMIT),
        }
        for item in pending[:BRIEF_MAX_KERNEL_TASKS]
    ]
    return cfg_items[:BRIEF_MAX_FRAMEWORK_TASKS], fw_items, kern_items


def _build_brief_scenario_block(after: ExportBundle, par_short: str, par_detail: str) -> dict[str, Any]:
    mem = after.memory_ledger
    eff_total_dies = (
        int(after.deployment_total_dies)
        if int(getattr(after, "deployment_total_dies", 0) or 0) > 0
        else mem.total_dies
    )
    eff_cluster_conc = (
        int(after.cluster_concurrency)
        if int(getattr(after, "cluster_concurrency", 0) or 0) > 0
        else (mem.concurrency * mem.dp)
    )
    scen_block: dict[str, Any] = {
        "model": sanitize_public_text(after.model_name),
        "hardware_sku": sanitize_public_text(after.hardware_sku),
        "topology_preset": sanitize_public_text(after.topology_preset),
        "phase": sanitize_public_text(after.phase),
        "deployment_mode": sanitize_public_text(after.deployment_mode),
        "track_label": _bundle_track_label(after),
        "total_dies": eff_total_dies,
        "dp": mem.dp, "tp": mem.tp, "ep": mem.ep, "pp": mem.pp,
        "dcp": mem.dcp, "pcp": mem.pcp, "local_tp": mem.local_tp,
        "sequence_parallel": after.sequence_parallel,
        "parallelism_co_domain": sanitize_public_text(par_short),
        "module_parallel_domains": sanitize_public_text(par_detail),
        "prompt_tokens": after.prompt_tokens, "output_tokens": after.output_tokens,
        "tpot_slo_ms": after.tpot_slo_ms, "ttft_slo_ms": after.ttft_slo_ms,
        "concurrency_per_dp": mem.concurrency, "cluster_concurrency": eff_cluster_conc,
        "max_hbm_concurrency_per_dp": mem.max_hbm_concurrency_per_dp,
        "spec_method": sanitize_public_text(after.spec_method),
        "spec_proposals": after.spec_proposals,
        "avg_accepted_tokens": after.avg_accepted_tokens,
        "weight_quant": sanitize_public_text(mem.weight_quant_display or mem.weight_quant),
        "cache_dtype": sanitize_public_text(mem.cache_dtype),
        "native_bf16_weight_gib_per_die": round(mem.native_bf16_weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
        "quantized_weight_gib_per_die": round(mem.weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
        "weight_compression_ratio": round(mem.weight_compression_ratio, 2),
        "fits_hbm": mem.fits_hbm,
        "recommended_checkpoint": project_recommended_checkpoint(after.recommended_checkpoint),
    }
    pd_pool = project_pd_pool(after.pd_pool_summary)
    if pd_pool is not None:
        scen_block["pd_pool"] = pd_pool
    return scen_block


def build_agent_brief(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle | None = None,
    *,
    generated_files: Mapping[str, str] | None = None,
    script_audit: Mapping[str, Any] | None = None,
    comparison_plan: ComparisonPlan | None = None,
    comparison_request: Any = None,
) -> dict[str, Any]:
    """Build compact (< 8 KB) allowlist-only deterministic JSON brief for Agent workflows."""
    from .comparison import (
        align_before_after_operators,
        assert_comparison_polarity,
        build_default_comparison_request,
        build_macro_module_breakdown,
        format_parallelism_codomain_summary,
    )

    if after_bundle is not None:
        eff_req = (
            comparison_request
            if comparison_request is not None
            else build_default_comparison_request(before_bundle, after_bundle)
        )
        assert_comparison_polarity(before_bundle, after_bundle, eff_req)
    after = after_bundle or before_bundle
    macro_rows = build_macro_module_breakdown(before_bundle, after_bundle)
    if comparison_plan is None:
        aligned = (
            align_before_after_operators(before_bundle, after)
            if after_bundle is not None
            else tuple(
                AlignedComparisonRow(
                    op.archetype_id, op.archetype_title or op.layer_label, op.repeat_count,
                    op.submodule, (op,), (op,), "1->1", "",
                )
                for op in before_bundle.operators
            )
        )
        plan = build_comparison_plan(
            before_bundle, after_bundle, aligned,
            script_audit=script_audit, comparison_request=comparison_request,
        )
    else:
        plan = comparison_plan

    pending, roadmap, codesign = plan.pending_items, plan.roadmap, plan.codesign_note
    par_short, par_detail = format_parallelism_codomain_summary(after)
    l1_groups = _build_brief_l1_groups(macro_rows)
    cfg_items, fw_items, kern_items = _build_brief_action_items(
        roadmap, pending, is_script_audit=(script_audit is not None),
    )
    b_kpi, a_kpi = _bundle_kpi_dict(before_bundle), _bundle_kpi_dict(after)
    speedup = round(b_kpi["step_ms"] / a_kpi["step_ms"], 2) if a_kpi["step_ms"] > 0.0 else 1.0
    synth_brief = [
        {
            "kernel": sanitize_public_text(sp.kernel_name),
            "tier": sanitize_public_text(sp.abi_tier),
            "sram_ok": bool(sp.sram_verified),
            "ub_kb": f"{sp.ub_peak_bytes / KIB_BYTES:.1f}/{sp.ub_budget_bytes / KIB_BYTES:.1f}",
            "math": _trim_text(
                " ; ".join(
                    sanitize_public_text(eq)
                    for eq in (sp.symbolic_equations or ())
                    if isinstance(eq, str)
                ),
                BRIEF_MATH_LIMIT,
            ),
        }
        for sp in list(after.kernel_synthesis_specs)[:BRIEF_MAX_SPECS]
    ]
    trig_brief = [
        {
            "id": sanitize_public_text(tr.trigger_id),
            "kind": sanitize_public_text(tr.kind),
            "ops": [sanitize_public_text(op) for op in list(tr.blocked_ops[:BRIEF_MAX_BLOCKED_OPS]) if isinstance(op, str)],
            "gain_us_tok_die": round(tr.potential_gain_us_per_tok_die, 2),
            "reason": _trim_text(tr.blocking_reason, BRIEF_TEXT_LIMIT),
            "refs": [sanitize_source_ref(ref) for ref in list(tr.source_refs[:BRIEF_MAX_REFS]) if isinstance(ref, str)],
        }
        for tr in sorted(after.semantic_audit_triggers, key=lambda t: -t.potential_gain_us_per_tok_die)[:BRIEF_MAX_TRIGGERS]
    ]
    codesign_opps = project_codesign_opportunities(
        getattr(after, "co_design_opportunities", ()),
        max_items=BRIEF_MAX_CODESIGN_OPPS,
    )
    brief: dict[str, Any] = {
        "schema": AGENT_BRIEF_SCHEMA,
        "scenario_and_parallelism": _build_brief_scenario_block(after, par_short, par_detail),
        "kpis_before_vs_after": {
            "before": b_kpi, "after": a_kpi,
            "saved_step_ms": float(b_kpi["step_ms"] - a_kpi["step_ms"]),
            "saved_step_ns": int(plan.waterfall.before_step_ns - plan.waterfall.after_step_ns),
            "reconciliation_error_ns": int(plan.waterfall.reconciliation_error_ns),
            "saved_us_per_tok_die": float(b_kpi["us_per_tok_die"] - a_kpi["us_per_tok_die"]),
            "speedup": speedup,
            "sheet4_cards_total_gain_ms": float(sum(x.expected_gain_ms for x in pending)),
            "sheet4_cards_total_gain_us_per_tok_die": float(sum(x.saved_us_per_tok_die for x in pending)),
            "waterfall": plan.waterfall.to_brief_dict(),
        },
        "module_hierarchy": l1_groups,
        "action_plan_3_tracks": {
            "track_labels": dict(THREE_TRACK_LABELS),
            "config_and_env": cfg_items, "framework_tasks": fw_items,
            "kernel_tasks": kern_items,
            "parallelism_codesign_note": _trim_text(codesign, BRIEF_CODESIGN_LIMIT),
        },
        "codesign_opportunities": codesign_opps,
        "kernel_synthesis_specs": synth_brief,
        "semantic_audit_triggers": trig_brief,
        "macro_wall_trigger": _factual_wall_summary(after),
        "va_serving_rules": _factual_serving_facts(after),
        "generated_files": project_artifacts(generated_files),
    }
    if script_audit is not None:
        projected_sa = project_script_audit(
            script_audit,
            max_top_issues=BRIEF_MAX_FRAMEWORK_TASKS,
            title_limit=BRIEF_KERNEL_LIMIT,
        )
        if projected_sa is not None:
            brief["script_audit"] = projected_sa
    return _compact_agent_brief_to_budget(project_agent_brief_mapping(brief))


def export_agent_brief(
    first: Mapping[str, Any] | ExportBundle,
    second: str | Path | ExportBundle | None,
    third: str | Path | None = None,
    *,
    generated_files: Mapping[str, str] | None = None,
    script_audit: Mapping[str, Any] | None = None,
    comparison_plan: ComparisonPlan | None = None,
    comparison_request: Any = None,
) -> Path | dict[str, Any]:
    """Atomically persist compact (< 8 KB) allowlist-projected ``agent_brief.json``."""
    return_dict = False
    if isinstance(first, ExportBundle):
        after_b = second if isinstance(second, ExportBundle) else None
        if third is None:
            raise ValueError("export_agent_brief_requires_output_path")
        brief = build_agent_brief(
            first, after_b,
            generated_files=generated_files,
            script_audit=script_audit,
            comparison_plan=comparison_plan,
            comparison_request=comparison_request,
        )
        target = Path(third)
        return_dict = True
    else:
        if not isinstance(second, (str, Path)):
            raise ValueError("export_agent_brief_requires_output_path")
        brief = _compact_agent_brief_to_budget(project_agent_brief_mapping(first))
        target = Path(second)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(brief, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded_len = len(payload.encode("utf-8"))
    if encoded_len <= AGENT_BRIEF_PRETTY_LIMIT_BYTES:
        pretty = json.dumps(brief, ensure_ascii=False, indent=1) + "\n"
        if len(pretty.encode("utf-8")) < AGENT_BRIEF_MAX_BYTES:
            payload = pretty
            encoded_len = len(payload.encode("utf-8"))
    if encoded_len >= AGENT_BRIEF_MAX_BYTES:
        raise ModelingError(f"agent_brief_exceeds_max_bytes:{encoded_len}>={AGENT_BRIEF_MAX_BYTES}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(tmp_path, target)
    except TimeoutError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return brief if return_dict else target


def _format_infeasible_digest_lines(
    sc: Mapping[str, Any], infeasible: Mapping[str, Any],
) -> list[str]:
    lines = [
        f"## MindIE-Sim Executive Digest: {sc.get('model') or 'Model'} @ {sc.get('hardware_sku') or 'NPU'} [INFEASIBLE]",
        (
            f"- **Outcome**: `{infeasible.get('status') or 'infeasible'}` "
            f"(Deployment={infeasible.get('deployment_mode') or sc.get('deployment_mode') or '—'})"
        ),
        f"- **Limiting Constraint**: {infeasible.get('limiting_constraint') or 'no_feasible_configuration'}",
        (
            f"- **Capacity & Concurrency**: Feasible B/DP={infeasible.get('feasible_b_per_dp', 0)}, "
            f"Physical B_vram_max={infeasible.get('physical_b_vram_max', 0)}"
            + (
                f", Requested={infeasible['requested_concurrency']}"
                if infeasible.get("requested_concurrency") is not None else ""
            )
        ),
    ]
    return lines


def format_executive_digest_markdown(brief: Mapping[str, Any]) -> str:
    """Format a crisp ~25-line allowlist-sanitized executive Markdown digest for stdout or chat."""
    clean_brief = project_agent_brief_mapping(brief)
    sc = clean_brief.get("scenario_and_parallelism") or {}
    kp = clean_brief.get("kpis_before_vs_after") or {}
    infeasible = clean_brief.get("infeasible_outcome")
    if isinstance(infeasible, Mapping) and not kp:
        lines = _format_infeasible_digest_lines(sc, infeasible)
        artifacts = project_artifacts(clean_brief.get("generated_files"), basenames_only=True)
        if artifacts:
            lines.append("- **Artifacts**: " + ", ".join(f"{k}=`{v}`" for k, v in artifacts.items()))
        return "\n".join(lines) + "\n"

    bef = kp.get("before") or {}
    aft = kp.get("after") or {}
    plan = clean_brief.get("action_plan_3_tracks") or {}
    is_modeling = str(aft.get("analysis_mode") or "") == MODE_MODELING or bool(aft.get("has_unbound_formula_ops"))
    bef_track = format_track_label(bef.get("track_label") or "shipping")
    aft_track = format_track_label(aft.get("track_label") or ("target" if is_modeling else "shipping"))
    aft_col = f"After [{aft_track} · 卡SLA目标建模终态]" if is_modeling else f"After [{aft_track}]"
    bind_badge = format_binding_badge(str(aft.get("binding_state") or BINDING_STATE_BOUND))
    sla_target = aft.get("sla_target_ms")
    sla_cell = (
        f"SLA <= {sla_target} ms ({aft.get('sla_regime')})"
        if sla_target is not None
        else f"SLA: 不限 ({aft.get('sla_regime') or 'UNCONSTRAINED'})"
    )
    t1_lbl = THREE_TRACK_LABELS[TRACK_CONFIG_ENV]
    t2_lbl = THREE_TRACK_LABELS[TRACK_FRAMEWORK]
    t3_lbl = THREE_TRACK_LABELS[TRACK_KERNEL]
    lines = [
        f"## MindIE-Sim Executive Digest: {sc.get('model')} @ {sc.get('hardware_sku')} {bind_badge}",
        (
            f"- **Deploy**: {sc.get('total_dies')} Dies ({sc.get('parallelism_co_domain') or ''}), "
            f"{sc.get('deployment_mode')}, Prompt={sc.get('prompt_tokens')}, "
            f"Concurrency={sc.get('concurrency_per_dp')}/DP (Max HBM={sc.get('max_hbm_concurrency_per_dp')})"
        ),
        (
            f"- **Weights/Die**: {sc.get('quantized_weight_gib_per_die')} GiB "
            f"(原生 BF16: {sc.get('native_bf16_weight_gib_per_die')} GiB, 压缩 {sc.get('weight_compression_ratio')}x, "
            f"Quant={sc.get('weight_quant')}, KV={sc.get('cache_dtype')})"
        ),
        "",
        f"| KPI | Before [{bef_track}] | {aft_col} | Delta / Verdict |",
        "| --- | ---: | ---: | --- |",
        f"| Step Latency (ms) | {bef.get('step_ms')} | **{aft.get('step_ms')}** | -{kp.get('saved_step_ms')} ms (**{kp.get('speedup')}x**) |",
        f"| Effective TPOT (ms) | {bef.get('effective_tpot_ms')} ({bef.get('sla_verdict')}) | **{aft.get('effective_tpot_ms')} ({aft.get('sla_verdict')})** | {sla_cell} |",
        f"| Amortized Cost (us/tok·die) | {bef.get('us_per_tok_die')} | **{aft.get('us_per_tok_die')}** | -{kp.get('saved_us_per_tok_die')} us/tok·die |",
        f"| Single-Die / Cluster TPS | {bef.get('single_die_tps')} / {bef.get('cluster_tps')} | **{aft.get('single_die_tps')} / {aft.get('cluster_tps')}** | tok/s |",
        f"| Physical Limit Attainment (L*/Step) | {bef.get('l_star_attainment_pct')}% | **{aft.get('l_star_attainment_pct')}%** | L*={aft.get('l_star_ms')} ms |",
        "",
        f"### 3-Track Action Plan ({t1_lbl} / {t2_lbl} / {t3_lbl})",
    ]
    if isinstance(infeasible, Mapping):
        lines.append(f"- **Infeasible Constraint**: {infeasible.get('limiting_constraint') or infeasible.get('status')}")
    wall = clean_brief.get("macro_wall_trigger")
    if isinstance(wall, Mapping) and wall.get("wall_kind") not in {None, "", "BALANCED"}:
        lines.append(
            f"- **Macro Wall (`{wall.get('wall_kind')}`)**: "
            f"HBM headroom={wall.get('hbm_headroom_gib')} GiB, SLA slack={wall.get('sla_slack_ms')} ms"
        )
    for cfg in (plan.get("config_and_env") or ())[:BRIEF_DIGEST_MAX_CFG]:
        detail = cfg.get("cmd") or f"{cfg.get('before', '')} -> {cfg.get('after', '')}"
        lines.append(f"- **{t1_lbl}** {cfg.get('title')}: `{detail}`")
    for fw in (plan.get("framework_tasks") or ())[:BRIEF_DIGEST_MAX_FW]:
        lines.append(f"- **{t2_lbl}** {fw.get('title')} (-{fw.get('saved_ms')} ms): {fw.get('work')}")
    for kt in (plan.get("kernel_tasks") or ())[:BRIEF_DIGEST_MAX_KERN]:
        lines.append(
            f"- **{t3_lbl} (`{kt.get('card_id')}`)** {kt.get('kernel')} "
            f"({kt.get('before_ms')}->{kt.get('after_ms')} ms, -{kt.get('saved_us_per_tok_die')} us/tok·die): {kt.get('summary')}"
        )
    for opp in (clean_brief.get("codesign_opportunities") or ())[:BRIEF_DIGEST_MAX_FW]:
        lines.append(f"- **Co-Design Opp (`{opp.get('kind')}`)**: {opp.get('title')}")
    if plan.get("parallelism_codesign_note"):
        lines.append(f"- **Co-Design**: {plan['parallelism_codesign_note']}")
    artifacts = project_artifacts(clean_brief.get("generated_files"), basenames_only=True)
    if artifacts:
        files_str = ", ".join(f"{k}=`{v}`" for k, v in artifacts.items())
        lines.append(f"- **Artifacts**: {files_str}")
    return "\n".join(lines) + "\n"


__all__ = [
    "AGENT_BRIEF_MAX_BYTES",
    "AGENT_BRIEF_PRETTY_LIMIT_BYTES",
    "_compact_agent_brief_to_budget",
    "build_agent_brief",
    "export_agent_brief",
    "format_executive_digest_markdown",
]
