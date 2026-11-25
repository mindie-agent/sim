"""Human-native Ascend optimization OOXML ``.xlsx`` workbook orchestrator.

[Role] Expose the single public ``export_comparison_xlsx`` entrypoint, bundle
  resolution, ``XlsxExportSummary`` receipt, and 4-sheet (plus conditional 5th
  calibration sheet) workbook orchestration delegating each worksheet domain to
  its dedicated cohesive module (``xlsx_summary``, ``xlsx_breakdown``,
  ``xlsx_memory``, ``xlsx_pending``).
[Invariants] Zero third-party OOXML deps, zero free-text prose overrides (retired
  in favor of verified ``--rewrite-plan`` receipts), single ``ComparisonPlan``
  shared identically across Sheet 1, Sheet 2, Sheet 4, and optional
  ``agent_brief.json``, and atomic same-directory ``os.replace`` persistence.
[Timeouts & Locks] Bounded O(Archetypes) in-memory XML serialization; no locks
  or device contexts.
[Side-Effect Policy] Writes only the caller-requested ``.xlsx`` (and optional
  ``agent_brief.json``) output path.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mindie_sim.prediction.results import CaseInput, ReplayOptions, StepEvaluation, StepRequest
from mindie_sim.schema.common import ModelingError

from .action_plan import ComparisonPlan, build_comparison_plan
from .agent_brief import build_agent_brief, export_agent_brief
from .bundle import build_export_bundle, compose_export_case
from .comparison import (
    align_before_after_operators,
    assert_comparison_polarity,
    build_default_comparison_request,
    build_macro_module_breakdown,
)
from .models import (
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    PENDING_KIND_TO_IMPLEMENT,
    AlignedComparisonRow,
    ComparisonRequest,
    ExportBundle,
)
from .ooxml import NS_MAIN as NS_MAIN, Workbook
from .xlsx_breakdown import (
    CMP_COLS,
    MEAS_COLS,
    STD_COLS,
    populate_comparison_breakdown_sheet,
    populate_measured_calibration_sheet,
    populate_standalone_breakdown_sheet,
)
from .xlsx_memory import MEMORY_COLS, populate_memory_sheet
from .xlsx_pending import PENDING_COLS, populate_pending_kernel_details_sheet
from .xlsx_summary import (
    SHEET_BREAKDOWN,
    SHEET_MEASURED,
    SHEET_MEMORY,
    SHEET_PENDING_KERNELS,
    SHEET_SUMMARY,
    SUMMARY_COLS,
    populate_summary_sheet,
)

#: Schema tag for XLSX export receipts.
XLSX_EXPORT_RECEIPT_SCHEMA = "mindie-sim.xlsx-export-receipt/3"


@dataclass(frozen=True, slots=True)
class XlsxExportSummary:
    """Compact receipt returned after writing an ``.xlsx`` workbook."""

    schema: str
    output_path: str
    model_name: str
    hardware_sku: str
    phase: str
    graph_id: str
    sheets: tuple[str, ...]
    merge_cell_counts: dict[str, int]
    operator_rows: int
    weight_rows: int
    has_comparison: bool
    has_measured: bool
    bytes_written: int
    new_kernel_backlog_count: int = 0
    to_optimize_kernel_count: int = 0
    agent_brief_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {
            "schema": self.schema,
            "output_path": self.output_path,
            "model_name": self.model_name,
            "hardware_sku": self.hardware_sku,
            "phase": self.phase,
            "graph_id": self.graph_id,
            "sheets": list(self.sheets),
            "merge_cell_counts": dict(self.merge_cell_counts),
            "operator_rows": self.operator_rows,
            "weight_rows": self.weight_rows,
            "has_comparison": self.has_comparison,
            "has_measured": self.has_measured,
            "bytes_written": self.bytes_written,
            "new_kernel_backlog_count": self.new_kernel_backlog_count,
            "to_optimize_kernel_count": self.to_optimize_kernel_count,
        }
        if self.agent_brief_path:
            out["agent_brief_path"] = self.agent_brief_path
        return out


def _resolve_xlsx_bundles(
    before_bundle: ExportBundle | CaseInput | None,
    after_bundle: ExportBundle | CaseInput | None,
    comparison_request: ComparisonRequest | None = None,
    *,
    request: StepRequest | None, step_eval: StepEvaluation | None, replay: ReplayOptions,
    observations_path: str | Path | None, weight_quant: str | None,
    cache_dtype: str | None, engram_host_offload: bool | None,
    include_vision_tower: bool | None, apply_preset_to_ignored: bool,
    model: str | None, hardware: str | None, topology_preset: str | None,
    dp: int, tp: int, dcp: int, ep: int, pp: int, pcp: int,
    prompt_tokens: int, output_tokens: int, concurrency: int,
    mtp_proposal_tokens: int, mtp_acceptance_length: float,
    before_track: str | None, after_track: str | None,
) -> tuple[ExportBundle, ExportBundle | None]:
    """Resolve explicit Before/After bundles and enforce 3-arg comparison polarity lock."""
    build_kw: dict[str, Any] = {
        "replay": replay,
        "weight_quant": weight_quant,
        "cache_dtype": cache_dtype,
        "engram_host_offload": engram_host_offload,
        "include_vision_tower": include_vision_tower,
        "apply_preset_to_ignored": apply_preset_to_ignored,
    }
    if before_bundle is None:
        if not model or not hardware:
            raise ModelingError("export_comparison_xlsx_requires_before_bundle_or_model_hardware")
        b_track = str(before_track or "shipping").strip().lower()
        raw_bef, req_bef = compose_export_case(
            model, hardware, topology_preset=topology_preset,
            dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens, concurrency=concurrency,
            mtp_proposal_tokens=mtp_proposal_tokens, mtp_acceptance_length=mtp_acceptance_length,
            feature_track=b_track,
            quant=weight_quant, kv_cache_dtype=cache_dtype,
            engram_cpu_offload=engram_host_offload,
        )
        eff_req = request if request is not None else req_bef
        composed_build_kw = {**build_kw, "weight_quant": None}
        resolved_before = build_export_bundle(
            raw_bef, request=eff_req, step_eval=step_eval,
            observations_path=observations_path, **composed_build_kw,
        )
        resolved_after: ExportBundle | None = None
        if after_bundle is not None:
            resolved_after = (
                after_bundle if isinstance(after_bundle, ExportBundle)
                else build_export_bundle(after_bundle, request=eff_req, **build_kw)
            )
        elif after_track is not None:
            raw_aft, req_aft = compose_export_case(
                model, hardware, topology_preset=topology_preset,
                dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
                prompt_tokens=prompt_tokens, output_tokens=output_tokens, concurrency=concurrency,
                mtp_proposal_tokens=mtp_proposal_tokens, mtp_acceptance_length=mtp_acceptance_length,
                feature_track=str(after_track).strip().lower(),
                quant=weight_quant, kv_cache_dtype=cache_dtype,
                engram_cpu_offload=engram_host_offload,
            )
            resolved_after = build_export_bundle(
                raw_aft, request=(request if request is not None else req_aft), **composed_build_kw,
            )
    else:
        resolved_before = (
            before_bundle if isinstance(before_bundle, ExportBundle)
            else build_export_bundle(
                before_bundle, request=request, step_eval=step_eval,
                observations_path=observations_path, **build_kw,
            )
        )
        resolved_after = None
        if after_bundle is not None:
            resolved_after = (
                after_bundle if isinstance(after_bundle, ExportBundle)
                else build_export_bundle(after_bundle, request=request, **build_kw)
            )

    if resolved_after is not None:
        eff_cmp_req = (
            comparison_request
            if comparison_request is not None
            else build_default_comparison_request(resolved_before, resolved_after)
        )
        assert_comparison_polarity(resolved_before, resolved_after, eff_cmp_req)
    return resolved_before, resolved_after


def _populate_workbook_sheets(
    wb: Workbook,
    bundle: ExportBundle,
    compare_bundle: ExportBundle | None,
    macro_rows: Any,
    plan: ComparisonPlan,
    *,
    is_script_audit: bool,
) -> tuple[list[str], dict[str, int]]:
    ws_summary = wb.create_sheet(SHEET_SUMMARY)
    ws_breakdown = wb.create_sheet(SHEET_BREAKDOWN)
    ws_memory = wb.create_sheet(SHEET_MEMORY)
    populate_summary_sheet(
        ws_summary, bundle, compare_bundle=compare_bundle,
        macro_rows=macro_rows, roadmap=plan.roadmap,
        codesign_note=plan.codesign_note, is_script_audit=is_script_audit,
        stage_waterfall=plan.stage_waterfall,
    )
    ws_summary.validate_max_column(SUMMARY_COLS)
    if compare_bundle is not None:
        s2_receipt = populate_comparison_breakdown_sheet(
            ws_breakdown, bundle, compare_bundle, plan.aligned_rows, plan.group_accounting,
        )
        ws_breakdown.validate_max_column(CMP_COLS)
    else:
        s2_receipt = populate_standalone_breakdown_sheet(ws_breakdown, bundle)
        ws_breakdown.validate_max_column(CMP_COLS)
    populate_memory_sheet(ws_memory, compare_bundle or bundle)
    ws_memory.validate_max_column(MEMORY_COLS)
    sheet_names = [SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY]
    merge_counts = {
        SHEET_SUMMARY: len(ws_summary.merged_ranges),
        SHEET_BREAKDOWN: len(ws_breakdown.merged_ranges),
        SHEET_MEMORY: len(ws_memory.merged_ranges),
    }
    target_bundle = compare_bundle or bundle
    ws_pending = wb.create_sheet(SHEET_PENDING_KERNELS)
    populate_pending_kernel_details_sheet(
        ws_pending, plan.pending_items, target_bundle,
        is_comparison=(compare_bundle is not None),
        waterfall=(plan.waterfall if compare_bundle is not None else None),
        breakdown_receipt=s2_receipt,
        before_bundle=bundle,
        group_accounting=plan.group_accounting,
        co_design_opportunities=target_bundle.co_design_opportunities,
    )
    ws_pending.validate_max_column(PENDING_COLS)
    sheet_names.append(SHEET_PENDING_KERNELS)
    merge_counts[SHEET_PENDING_KERNELS] = len(ws_pending.merged_ranges)
    if bundle.reconciliation is not None:
        ws_meas = wb.create_sheet(SHEET_MEASURED)
        populate_measured_calibration_sheet(ws_meas, bundle)
        ws_meas.validate_max_column(MEAS_COLS)
        sheet_names.append(SHEET_MEASURED)
        merge_counts[SHEET_MEASURED] = len(ws_meas.merged_ranges)
    return sheet_names, merge_counts


def export_comparison_xlsx(
    before_bundle: ExportBundle | CaseInput | None = None,
    output_path: str | Path = "",
    *,
    after_bundle: ExportBundle | CaseInput | None = None,
    comparison_request: Any = None,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
    include_vision_tower: bool | None = None,
    apply_preset_to_ignored: bool = False,
    agent_brief_path: str | Path | None = None,
    script_audit: Any = None,
    comparison_plan: ComparisonPlan | None = None,
    model: str | None = None,
    hardware: str | None = None,
    topology_preset: str | None = None,
    dp: int = 1,
    tp: int = 1,
    dcp: int = 1,
    ep: int = 1,
    pp: int = 1,
    pcp: int = 1,
    prompt_tokens: int = DEFAULT_SCENARIO_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_SCENARIO_OUTPUT_TOKENS,
    concurrency: int = 1,
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    before_track: str | None = None,
    after_track: str | None = None,
) -> XlsxExportSummary:
    """Build and atomically save the receipt-driven Ascend 4-sheet comparison ``.xlsx`` workbook."""
    if not str(output_path):
        raise ValueError("export_comparison_xlsx_requires_output_path")

    bundle, compare_bundle = _resolve_xlsx_bundles(
        before_bundle, after_bundle, comparison_request,
        request=request, step_eval=step_eval, replay=replay,
        observations_path=observations_path,
        weight_quant=weight_quant, cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload,
        include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored,
        model=model, hardware=hardware, topology_preset=topology_preset,
        dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        concurrency=concurrency, mtp_proposal_tokens=mtp_proposal_tokens,
        mtp_acceptance_length=mtp_acceptance_length,
        before_track=before_track, after_track=after_track,
    )

    macro_rows = build_macro_module_breakdown(bundle, compare_bundle)
    if comparison_plan is not None:
        plan = comparison_plan
    else:
        if compare_bundle is not None:
            aligned_rows = align_before_after_operators(bundle, compare_bundle)
        else:
            aligned_rows = tuple(
                AlignedComparisonRow(
                    op.archetype_id, op.archetype_title or op.layer_label, op.repeat_count,
                    op.submodule, (op,), (op,), "1->1", "",
                )
                for op in bundle.operators
            )
        plan = build_comparison_plan(
            bundle, compare_bundle, aligned_rows,
            script_audit=script_audit, comparison_request=comparison_request,
        )

    pending_items = plan.pending_items
    wb = Workbook()
    sheet_names, merge_counts = _populate_workbook_sheets(
        wb, bundle, compare_bundle, macro_rows, plan,
        is_script_audit=(script_audit is not None),
    )

    out = Path(output_path)
    size = wb.save(out)
    brief_path_str = ""
    if agent_brief_path is not None:
        audit_dict = (
            script_audit.to_dict() if hasattr(script_audit, "to_dict")
            else (dict(script_audit) if isinstance(script_audit, Mapping) else None)
        )
        brief = build_agent_brief(
            bundle, compare_bundle,
            generated_files={
                "comparison_xlsx": out.name,
                "agent_brief_json": Path(agent_brief_path).name,
            },
            script_audit=audit_dict,
            comparison_plan=plan,
            comparison_request=comparison_request,
        )
        export_agent_brief(brief, agent_brief_path)
        brief_path_str = str(Path(agent_brief_path))

    new_cnt = sum(1 for item in pending_items if item.category == PENDING_KIND_TO_IMPLEMENT)
    opt_cnt = len(pending_items) - new_cnt
    return XlsxExportSummary(
        schema=XLSX_EXPORT_RECEIPT_SCHEMA,
        output_path=str(out),
        model_name=bundle.model_name,
        hardware_sku=bundle.hardware_sku,
        phase=bundle.phase,
        graph_id=bundle.graph_id,
        sheets=tuple(sheet_names),
        merge_cell_counts=merge_counts,
        operator_rows=len(bundle.operators),
        weight_rows=len((compare_bundle or bundle).memory_ledger.weight_rows),
        has_comparison=compare_bundle is not None,
        has_measured=bundle.reconciliation is not None,
        bytes_written=size,
        new_kernel_backlog_count=new_cnt,
        to_optimize_kernel_count=opt_cnt,
        agent_brief_path=brief_path_str,
    )


__all__ = [
    "ComparisonRequest",
    "NS_MAIN",
    "SHEET_BREAKDOWN",
    "SHEET_MEASURED",
    "SHEET_MEMORY",
    "SHEET_PENDING_KERNELS",
    "SHEET_SUMMARY",
    "XLSX_EXPORT_RECEIPT_SCHEMA",
    "XlsxExportSummary",
    "assert_comparison_polarity",
    "export_comparison_xlsx",
]
