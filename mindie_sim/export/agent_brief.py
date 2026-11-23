"""Compact (< 8 KB) ``agent_brief.json`` builder, budget compactor & Markdown digest.

[Role] Project ``ExportBundle`` comparisons and ``ComparisonPlan`` receipts into
  a deterministic ``< 8 KB`` JSON brief (atomic same-directory persistence) and
  crisp ~25-line Markdown executive digest sharing the exact same Waterfall,
  cards, and 3-track roadmap rendered in the OOXML workbook.
[Invariants] Strictly enforces ``len(utf8_bytes) < 8192`` including trailing LF,
  preserves full-precision KPIs and Waterfall reconciliation fields, omits
  ``script_audit`` and shell env-switch nags in script-free mode, and never
  swallows ``TimeoutError`` under ``except OSError:``.
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
    US_PER_MS,
    AlignedComparisonRow,
    ExportBundle,
    MacroModuleRow,
    format_binding_badge,
)

#: Maximum number of compact kernel tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_KERNEL_TASKS = 5
#: Maximum number of framework/config tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_FRAMEWORK_TASKS = 3
#: Maximum number of synthesis specs & audit triggers embedded in ``agent_brief.json``.
BRIEF_MAX_TRIGGERS = 4
BRIEF_MAX_SPECS = 4
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


def _trim_text(text: str, limit: int = BRIEF_TEXT_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else (clean[: max(0, limit - 3)] + "...")


def _bundle_kpi_dict(bundle: ExportBundle) -> dict[str, Any]:
    step_ms = bundle.step_span_us / US_PER_MS
    is_decode = bundle.phase == "decode"
    acc = max(1.0, bundle.avg_accepted_tokens)
    handoff_ms = float(getattr(bundle, "handoff_tpot_ms", 0.0) or 0.0) if is_decode else 0.0
    tpot_ms = ((step_ms / acc) + handoff_ms) if is_decode else step_ms
    sla = bundle.tpot_slo_ms if is_decode else bundle.ttft_slo_ms
    l_star_ms = bundle.step_l_star_lean_us / US_PER_MS
    pool = bundle.pd_pool_summary if isinstance(bundle.pd_pool_summary, Mapping) else {}
    cluster_tps = pool.get("e2e_total_tps")
    if cluster_tps is None:
        cluster_tps = bundle.throughput_tokens_per_s
    return {
        "analysis_mode": bundle.analysis_mode,
        "binding_state": bundle.binding_state,
        "has_unbound_formula_ops": bool(bundle.has_unbound_formula_ops),
        "step_ms": float(step_ms),
        "handoff_tpot_ms": float(handoff_ms),
        "effective_tpot_ms": float(tpot_ms),
        "sla_target_ms": float(sla),
        "sla_verdict": "PASS" if tpot_ms <= sla else "FAIL",
        "sla_regime": bundle.sla_regime,
        "sla_slack_ms": float(sla - tpot_ms),
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
    for key in ("kernel_synthesis_specs", "semantic_audit_triggers"):
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
        for key in ("kernel_synthesis_specs", "semantic_audit_triggers"):
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
        "va_serving_rules", "kernel_synthesis_specs", "semantic_audit_triggers",
        "macro_wall_trigger", "generated_files",
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
    raw = bundle.va_serving_rules_summary
    if not isinstance(raw, Mapping):
        return None
    return {
        k: v for k, v in raw.items()
        if k != "hw_isa_guidance"
    }


def build_agent_brief(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle | None = None,
    *,
    generated_files: Mapping[str, str] | None = None,
    script_audit: Mapping[str, Any] | None = None,
    comparison_plan: ComparisonPlan | None = None,
) -> dict[str, Any]:
    """Build compact (< 8 KB) deterministic JSON brief for Agent workflows."""
    from .comparison import (
        align_before_after_operators,
        build_macro_module_breakdown,
        format_parallelism_codomain_summary,
    )

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
            before_bundle, after_bundle, aligned, script_audit=script_audit,
        )
    else:
        plan = comparison_plan

    pending = plan.pending_items
    roadmap = plan.roadmap
    codesign = plan.codesign_note
    par_short, par_detail = format_parallelism_codomain_summary(after)
    mem = after.memory_ledger

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

    is_script_audit = script_audit is not None
    cfg_items: list[dict[str, Any]] = []
    for r in roadmap:
        if r.track != TRACK_CONFIG_ENV:
            continue
        if is_script_audit:
            cfg_items.append({
                "title": _trim_text(r.title, BRIEF_TITLE_LIMIT),
                "before": _trim_text(r.before_value, BRIEF_TITLE_LIMIT),
                "after": _trim_text(r.after_value, BRIEF_CMD_LIMIT),
                "cmd": _trim_text(r.action_detail, BRIEF_CMD_LIMIT),
            })
        else:
            cfg_items.append({
                "title": _trim_text(r.title, BRIEF_TITLE_LIMIT),
                "before": _trim_text(r.before_value, BRIEF_TITLE_LIMIT),
                "after": _trim_text(r.after_value, BRIEF_TITLE_LIMIT),
            })
    cfg_items = cfg_items[:BRIEF_MAX_FRAMEWORK_TASKS]

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
            "card_id": item.card_id,
            "prio": f"P{item.priority}",
            "cat": "new" if item.category == PENDING_KIND_TO_IMPLEMENT else "tune",
            "kernel": _trim_text(item.target_kernel, BRIEF_KERNEL_LIMIT),
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
    b_kpi = _bundle_kpi_dict(before_bundle)
    a_kpi = _bundle_kpi_dict(after)
    speedup = round(b_kpi["step_ms"] / a_kpi["step_ms"], 2) if a_kpi["step_ms"] > 0.0 else 1.0

    synth_brief = [
        {
            "kernel": sp.kernel_name,
            "tier": sp.abi_tier,
            "sram_ok": sp.sram_verified,
            "ub_kb": f"{sp.ub_peak_bytes / KIB_BYTES:.1f}/{sp.ub_budget_bytes / KIB_BYTES:.1f}",
            "math": _trim_text(" ; ".join(sp.symbolic_equations), BRIEF_MATH_LIMIT),
        }
        for sp in list(after.kernel_synthesis_specs)[:BRIEF_MAX_SPECS]
    ]
    trig_brief = [
        {
            "id": tr.trigger_id,
            "kind": tr.kind,
            "ops": list(tr.blocked_ops[:BRIEF_MAX_BLOCKED_OPS]),
            "gain_us_tok_die": round(tr.potential_gain_us_per_tok_die, 2),
            "reason": _trim_text(tr.blocking_reason, BRIEF_TEXT_LIMIT),
            "refs": list(tr.source_refs[:BRIEF_MAX_REFS]),
        }
        for tr in sorted(after.semantic_audit_triggers, key=lambda t: -t.potential_gain_us_per_tok_die)[:BRIEF_MAX_TRIGGERS]
    ]
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
        "model": after.model_name,
        "hardware_sku": after.hardware_sku,
        "topology_preset": after.topology_preset,
        "phase": after.phase,
        "deployment_mode": after.deployment_mode,
        "total_dies": eff_total_dies,
        "dp": mem.dp, "tp": mem.tp, "ep": mem.ep, "pp": mem.pp,
        "dcp": mem.dcp, "pcp": mem.pcp, "local_tp": mem.local_tp,
        "sequence_parallel": after.sequence_parallel,
        "parallelism_co_domain": par_short,
        "module_parallel_domains": par_detail,
        "prompt_tokens": after.prompt_tokens,
        "output_tokens": after.output_tokens,
        "concurrency_per_dp": mem.concurrency,
        "cluster_concurrency": eff_cluster_conc,
        "max_hbm_concurrency_per_dp": mem.max_hbm_concurrency_per_dp,
        "spec_method": after.spec_method,
        "spec_proposals": after.spec_proposals,
        "avg_accepted_tokens": after.avg_accepted_tokens,
        "weight_quant": mem.weight_quant_display or mem.weight_quant,
        "cache_dtype": mem.cache_dtype,
        "native_bf16_weight_gib_per_die": round(mem.native_bf16_weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
        "quantized_weight_gib_per_die": round(mem.weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
        "weight_compression_ratio": round(mem.weight_compression_ratio, 2),
        "fits_hbm": mem.fits_hbm,
        "recommended_checkpoint": after.recommended_checkpoint,
    }
    if after.pd_pool_summary:
        scen_block["pd_pool"] = dict(after.pd_pool_summary)
    if after.calibration_basis:
        scen_block["calibration_basis"] = {
            "mode": after.calibration_basis.get("mode"),
            "calibrated": bool(after.calibration_basis.get("calibrated")),
            "pt_sku": after.calibration_basis.get("primitive_table_sku"),
            "cann": after.calibration_basis.get("cann_version"),
        }
    brief: dict[str, Any] = {
        "schema": AGENT_BRIEF_SCHEMA,
        "scenario_and_parallelism": scen_block,
        "kpis_before_vs_after": {
            "before": b_kpi,
            "after": a_kpi,
            "saved_step_ms": float(b_kpi["step_ms"] - a_kpi["step_ms"]),
            "saved_us_per_tok_die": float(b_kpi["us_per_tok_die"] - a_kpi["us_per_tok_die"]),
            "speedup": speedup,
            "sheet4_cards_total_gain_ms": float(sum(x.expected_gain_ms for x in pending)),
            "sheet4_cards_total_gain_us_per_tok_die": float(sum(x.saved_us_per_tok_die for x in pending)),
            "waterfall": plan.waterfall.to_brief_dict(),
        },
        "module_hierarchy": l1_groups,
        "action_plan_3_tracks": {
            "config_and_env": cfg_items,
            "framework_tasks": fw_items,
            "kernel_tasks": kern_items,
            "parallelism_codesign_note": _trim_text(codesign, BRIEF_CODESIGN_LIMIT),
        },
        "kernel_synthesis_specs": synth_brief,
        "semantic_audit_triggers": trig_brief,
        "macro_wall_trigger": _factual_wall_summary(after),
        "va_serving_rules": _factual_serving_facts(after),
        "generated_files": dict(generated_files or {}),
    }
    if script_audit is not None:
        raw_issues = script_audit.get("issues_found") or ()
        brief["script_audit"] = {
            "source_script": script_audit.get("source_script", ""),
            "issues_count": len(raw_issues),
            "top_issues": [
                {"id": i.get("issue_id", ""), "title": _trim_text(str(i.get("title", "")), BRIEF_KERNEL_LIMIT)}
                for i in list(raw_issues)[:BRIEF_MAX_FRAMEWORK_TASKS]
            ],
            "two_step_gain": script_audit.get("two_step_gain", {}),
            "optimized_script_path": script_audit.get("optimized_script_path", ""),
        }
    return _compact_agent_brief_to_budget(brief)


def export_agent_brief(
    first: Mapping[str, Any] | ExportBundle,
    second: str | Path | ExportBundle | None,
    third: str | Path | None = None,
    *,
    generated_files: Mapping[str, str] | None = None,
    script_audit: Mapping[str, Any] | None = None,
    comparison_plan: ComparisonPlan | None = None,
) -> Path | dict[str, Any]:
    """Atomically persist compact (< 8 KB) ``agent_brief.json``."""
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
        )
        target = Path(third)
        return_dict = True
    else:
        if not isinstance(second, (str, Path)):
            raise ValueError("export_agent_brief_requires_output_path")
        brief = _compact_agent_brief_to_budget(dict(first))
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


def format_executive_digest_markdown(brief: Mapping[str, Any]) -> str:
    """Format a crisp ~25-line executive Markdown digest for stdout or chat."""
    sc = brief.get("scenario_and_parallelism") or {}
    kp = brief.get("kpis_before_vs_after") or {}
    bef = kp.get("before") or {}
    aft = kp.get("after") or {}
    plan = brief.get("action_plan_3_tracks") or {}
    is_modeling = str(aft.get("analysis_mode") or "") == MODE_MODELING or bool(aft.get("has_unbound_formula_ops"))
    aft_col = "After [卡SLA目标建模终态]" if is_modeling else "After [仿真对比]"
    bind_badge = format_binding_badge(str(aft.get("binding_state") or BINDING_STATE_BOUND))
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
        f"| KPI | Before [当前栈仿真基线] | {aft_col} | Delta / Verdict |",
        "| --- | ---: | ---: | --- |",
        f"| Step Latency (ms) | {bef.get('step_ms')} | **{aft.get('step_ms')}** | -{kp.get('saved_step_ms')} ms (**{kp.get('speedup')}x**) |",
        f"| Effective TPOT (ms) | {bef.get('effective_tpot_ms')} ({bef.get('sla_verdict')}) | **{aft.get('effective_tpot_ms')} ({aft.get('sla_verdict')})** | SLA <= {aft.get('sla_target_ms')} ms ({aft.get('sla_regime')}) |",
        f"| Amortized Cost (us/tok·die) | {bef.get('us_per_tok_die')} | **{aft.get('us_per_tok_die')}** | -{kp.get('saved_us_per_tok_die')} us/tok·die |",
        f"| Single-Die / Cluster TPS | {bef.get('single_die_tps')} / {bef.get('cluster_tps')} | **{aft.get('single_die_tps')} / {aft.get('cluster_tps')}** | tok/s |",
        f"| Physical Limit Attainment (L*/Step) | {bef.get('l_star_attainment_pct')}% | **{aft.get('l_star_attainment_pct')}%** | L*={aft.get('l_star_ms')} ms |",
        "",
        "### 3-Track Action Plan (`Config / Diff` / `Framework` / `Kernel`)",
    ]
    wall = brief.get("macro_wall_trigger")
    if isinstance(wall, Mapping) and wall.get("wall_kind") not in {None, "", "BALANCED"}:
        lines.append(
            f"- **Macro Wall (`{wall.get('wall_kind')}`)**: "
            f"HBM headroom={wall.get('hbm_headroom_gib')} GiB, SLA slack={wall.get('sla_slack_ms')} ms"
        )
    for cfg in (plan.get("config_and_env") or ())[:BRIEF_DIGEST_MAX_CFG]:
        detail = cfg.get("cmd") or f"{cfg.get('before', '')} -> {cfg.get('after', '')}"
        lines.append(f"- **[Config]** {cfg.get('title')}: `{detail}`")
    for fw in (plan.get("framework_tasks") or ())[:BRIEF_DIGEST_MAX_FW]:
        lines.append(f"- **[Framework]** {fw.get('title')} (-{fw.get('saved_ms')} ms): {fw.get('work')}")
    for kt in (plan.get("kernel_tasks") or ())[:BRIEF_DIGEST_MAX_KERN]:
        lines.append(
            f"- **[Kernel `{kt.get('card_id')}`]** {kt.get('kernel')} "
            f"({kt.get('before_ms')}->{kt.get('after_ms')} ms, -{kt.get('saved_us_per_tok_die')} us/tok·die): {kt.get('summary')}"
        )
    if plan.get("parallelism_codesign_note"):
        lines.append(f"- **Co-Design**: {plan['parallelism_codesign_note']}")
    if brief.get("generated_files"):
        files_str = ", ".join(f"{k}=`{v}`" for k, v in brief["generated_files"].items())
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
