"""Sheet 4 (``Pending_Kernel_Details``) live Waterfall Ledger & Kernel Cards writer.

[Role] Render Sheet 4 cleanly decoupled into:
  (1) Live-formula ``端到端收益瀑布对账表 (Waterfall Ledger)`` at rows 2..10
      reconciling ``Before Step -> Config/Parallel/Eliminated + New Fused
      Kernel + Existing Kernel Tuning + Residual Ops == Reconstructed After
      Step == Summary!$C$14`` via bounded ``SUMIF`` ranges over Sheet 2
      accounting columns ``AB..AF`` (0 reconciliation error, 0 Excel errors);
  (2) High-signal actionable kernel index (``<= 12`` genuine cards:
      ``[需新开发融合算子]`` & ``[现成算子深度调优]`` only — zero glue cards,
      zero pseudo-cards, zero ``Unchanged Operators`` fudge row);
  (3) Structured physical-receipt cards (zero Config & Env nag row, zero
      guessed layout/prose overrides, zero ``>=78%`` target strings).
[Invariants] Waterfall is always emitted on comparison workbooks, including
  identical bundles and zero-card reports; all formulas use bounded ranges and
  guarded denominators so zero ``#DIV/0!``/``#REF!``/``#VALUE!`` errors occur.
[Timeouts & Locks] Pure in-memory OOXML cell assembly; no I/O or locks.
[Side-Effect Policy] Mutates only the caller-provided ``Worksheet``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .action_plan import WaterfallSummary, format_signed_gain_text
from .models import (
    PENDING_KIND_TO_IMPLEMENT,
    ExportBundle,
    PendingKernelItem,
    format_binding_badge,
)
from .ooxml import (
    BACKLOG_ROW_HEIGHT_PT,
    CARD_ACCEPT_ROW_HEIGHT_PT,
    CARD_DESIGN_ROW_HEIGHT_PT,
    HEADER_ROW_HEIGHT_PT,
    STYLE_ALERT_HIGH,
    STYLE_ALERT_OK,
    STYLE_CARD_HDR,
    STYLE_FLOAT2,
    STYLE_FLOAT3,
    STYLE_KPI_HIGHLIGHT,
    STYLE_MERGED,
    STYLE_MUTED_TEXT,
    STYLE_NEW_KERNEL,
    STYLE_OPT_BADGE,
    STYLE_OVERLAP,
    STYLE_SECTION,
    STYLE_TEXT,
    STYLE_TEXT_WRAP,
    STYLE_TITLE,
    STYLE_TOTAL_FLOAT3,
    STYLE_TOTAL_NUM,
    STYLE_TOTAL_TEXT,
    WRAP_ROW_HEIGHT_PT,
    Worksheet,
    column_letter,
)
from .xlsx_breakdown import (
    BreakdownSheetReceipt,
    CMP_COL_AFT_EXP_MS,
    CMP_COL_BEF_EXP_MS,
    CMP_COL_CARD_ID,
    CMP_COL_CFG_SHARE_MS,
    CMP_COL_CFG_SHARE_UPT,
    CMP_COL_SAVED_MS,
    CMP_COL_SAVED_UPT,
    CMP_COL_WF_BUCKET,
    CMP_DATA_START_ROW,
    STD_COL_EXP_MS,
    STD_COL_LSTAR_MS,
)
from .xlsx_summary import (
    SHEET_BREAKDOWN,
    SUM_AFT_NTOK_REF,
    SUM_AFT_STEP_REF,
    SUM_AFT_UPT_REF,
    SUM_BEF_NTOK_REF,
    SUM_BEF_STEP_REF,
    SUM_BEF_UPT_REF,
    apply_col_widths,
    write_banner,
    write_headers,
    write_row,
)

#: Sheet 4 structural layout constants (14 columns A..N).
PENDING_COLS = 14
WF_HDR_ROW = 2
WF_ROW_BEFORE = 3
WF_ROW_CONFIG = 4
WF_ROW_NEW = 5
WF_ROW_TUNE = 6
WF_ROW_RESIDUAL = 7
WF_ROW_RECON = 8
WF_ROW_AFTER = 9
WF_ROW_ERROR = 10
WF_NOTE_START_COL = 6

CMP_INDEX_BANNER_ROW = 12
CMP_INDEX_HEADER_ROW = 13
CMP_INDEX_DATA_START_ROW = 14
STD_INDEX_BANNER_ROW = 2
STD_INDEX_HEADER_ROW = 3
STD_INDEX_DATA_START_ROW = 4

PENDING_COL_GAIN = 7
PENDING_COL_GAIN_UPT = 11
PENDING_COL_BINDING = 12
PENDING_COL_BEF_MS = 13
PENDING_COL_AFT_MS = 14
PENDING_LABEL_MERGE_END = 6
PENDING_CARD_SPACER_GAP = 3
PENDING_CARD_STRIDE = 9
PENDING_CARD_LBL_END = 2
PENDING_CARD_LEFT_START = 3
PENDING_CARD_LEFT_END = 5
PENDING_CARD_MID_COL = 6
PENDING_CARD_RIGHT_START = 7
PENDING_CARD_NOTE_START = 8
PENDING_CARD_ROW_STACK = 1
PENDING_CARD_ROW_CHAIN = 2
PENDING_CARD_ROW_FREQ = 3
PENDING_CARD_ROW_SHAPE = 4
PENDING_CARD_ROW_BOUND = 5
PENDING_CARD_ROW_KERNEL = 6
PENDING_CARD_ROW_FRAMEWORK = 7
PENDING_CARD_ROW_RECEIPT = 8

WF_RECON_TOL_MS = 1e-4

PENDING_COL_WIDTHS = (
    12.0, 26.0, 28.0, 26.0, 28.0, 24.0, 15.0, 22.0, 24.0, 22.0, 16.0, 26.0, 14.0, 14.0,
)


def _sheet2_ref(col_idx: int, r_start: int, r_end: int) -> str:
    let = column_letter(col_idx)
    return f"'{SHEET_BREAKDOWN}'!${let}${r_start}:${let}${r_end}"


def _write_waterfall_ledger(
    ws: Worksheet,
    wf: WaterfallSummary,
    receipt: BreakdownSheetReceipt,
) -> None:
    """Write live-formula Waterfall Ledger at rows 2..10 matching Blueprint §3."""
    write_headers(ws, WF_HDR_ROW, (
        "端到端收益瀑布对账表 (Waterfall Ledger)",
        "净省 Δ (ms)",
        "单步变化 (ms)",
        "累计单步耗时 (ms)",
        "摊薄净省 Δ (us/tok·die)",
        "会计口径与 Sheet 2 端点校验 (Accounting Basis & Endpoint Audit)",
    ))
    ws.merge_range(WF_HDR_ROW, WF_NOTE_START_COL, WF_HDR_ROW, PENDING_COLS)

    r_start = CMP_DATA_START_ROW
    r_end = receipt.last_detail_row
    has_rows = r_end >= r_start
    ab_rng = _sheet2_ref(CMP_COL_WF_BUCKET, r_start, r_end) if has_rows else ""
    q_rng = _sheet2_ref(CMP_COL_SAVED_MS, r_start, r_end) if has_rows else ""
    z_rng = _sheet2_ref(CMP_COL_SAVED_UPT, r_start, r_end) if has_rows else ""
    ae_rng = _sheet2_ref(CMP_COL_CFG_SHARE_MS, r_start, r_end) if has_rows else ""
    af_rng = _sheet2_ref(CMP_COL_CFG_SHARE_UPT, r_start, r_end) if has_rows else ""

    def _bucket_ms_f(tag: str) -> str:
        if not has_rows:
            return "=0"
        return f'=SUMIF({ab_rng},"{tag}",{q_rng})-SUMIF({ab_rng},"{tag}",{ae_rng})'

    def _bucket_upt_f(tag: str) -> str:
        if not has_rows:
            return "=0"
        return f'=SUMIF({ab_rng},"{tag}",{z_rng})-SUMIF({ab_rng},"{tag}",{af_rng})'

    gt = receipt.grand_total_row
    j_let = column_letter(CMP_COL_BEF_EXP_MS)
    p_let = column_letter(CMP_COL_AFT_EXP_MS)
    bef_ep_f = f"='{SHEET_BREAKDOWN}'!{j_let}{gt}-{SUM_BEF_STEP_REF}" if gt >= r_start else "=0"
    aft_ep_f = f"='{SHEET_BREAKDOWN}'!{p_let}{gt}-{SUM_AFT_STEP_REF}" if gt >= r_start else "=0"

    run_cfg = wf.before_step_ms - wf.config_saved_ms
    run_new = run_cfg - wf.new_kernel_saved_ms
    run_tune = run_new - wf.tune_kernel_saved_ms
    run_res = run_tune - wf.residual_saved_ms
    tot_saved_ms = wf.config_saved_ms + wf.new_kernel_saved_ms + wf.tune_kernel_saved_ms + wf.residual_saved_ms
    tot_saved_upt = wf.config_saved_upt + wf.new_kernel_saved_upt + wf.tune_kernel_saved_upt + wf.residual_saved_upt

    # Row 3: Before Step
    ws.set_row_height(WF_ROW_BEFORE, WRAP_ROW_HEIGHT_PT)
    write_row(ws, WF_ROW_BEFORE, (
        ("Before Step", None, STYLE_MERGED),
        ("", None, STYLE_TEXT),
        ("", None, STYLE_TEXT),
        (wf.before_step_ms, f"={SUM_BEF_STEP_REF}", STYLE_FLOAT3),
        ("", None, STYLE_TEXT),
        (wf.before_endpoint_error_ms, bef_ep_f, STYLE_ALERT_OK if abs(wf.before_endpoint_error_ms) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH),
        ("Before Endpoint Δ = Sheet2 Grand Total J - Summary!$B$14 (ms)", None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(WF_ROW_BEFORE, WF_NOTE_START_COL + 1, WF_ROW_BEFORE, PENDING_COLS)

    # Rows 4..7: 4 signed waterfall buckets
    bucket_rows = (
        (
            WF_ROW_CONFIG, "Config / Parallel / Eliminated",
            wf.config_saved_ms, f"=SUM({ae_rng})" if has_rows else "=0",
            run_cfg, wf.config_saved_upt, f"=SUM({af_rng})" if has_rows else "=0",
            "配置/并行/批量缩放、掩盖变化、现成融合使能与增删算子净变化 (Sheet 2 AE/AF)",
        ),
        (
            WF_ROW_NEW, "New Fused Kernel",
            wf.new_kernel_saved_ms, _bucket_ms_f("new"),
            run_new, wf.new_kernel_saved_upt, _bucket_upt_f("new"),
            "需新开发 N->1 融合算子净收益 ([需新开发融合算子]，Sheet 2 AB=\"new\")",
        ),
        (
            WF_ROW_TUNE, "Existing Kernel Tuning",
            wf.tune_kernel_saved_ms, _bucket_ms_f("tune"),
            run_tune, wf.tune_kernel_saved_upt, _bucket_upt_f("tune"),
            "现成 1->1 算子深度调优净收益 ([现成算子深度调优]，聚合 >= 0.050 ms 且 Δ摊薄 > 0)",
        ),
        (
            WF_ROW_RESIDUAL, "Residual Ops",
            wf.residual_saved_ms, _bucket_ms_f("residual"),
            run_res, wf.residual_saved_upt, _bucket_upt_f("residual"),
            "已近物理下界 / 低于阈值 / 胶水残差算子 ([已近物理下界 / 无需改动]，0 卡片)",
        ),
    )
    for r, lbl, b_val, b_f, run_val, e_val, e_f, note in bucket_rows:
        ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
        write_row(ws, r, (
            (lbl, None, STYLE_TEXT),
            (b_val, b_f, STYLE_FLOAT3),
            (-b_val, f"=-B{r}", STYLE_FLOAT3),
            (run_val, f"=D{r - 1}+C{r}", STYLE_FLOAT3),
            (e_val, e_f, STYLE_FLOAT2),
            (note, None, STYLE_TEXT_WRAP),
        ))
        ws.merge_range(r, WF_NOTE_START_COL, r, PENDING_COLS)

    # Row 8: Reconstructed After Step
    ws.set_row_height(WF_ROW_RECON, WRAP_ROW_HEIGHT_PT)
    write_row(ws, WF_ROW_RECON, (
        ("Reconstructed After Step", None, STYLE_TOTAL_TEXT),
        (tot_saved_ms, f"=SUM(B{WF_ROW_CONFIG}:B{WF_ROW_RESIDUAL})", STYLE_TOTAL_FLOAT3),
        (-tot_saved_ms, f"=SUM(C{WF_ROW_CONFIG}:C{WF_ROW_RESIDUAL})", STYLE_TOTAL_FLOAT3),
        (wf.reconstructed_after_ms, f"=D{WF_ROW_BEFORE}+C{WF_ROW_RECON}", STYLE_TOTAL_FLOAT3),
        (tot_saved_upt, f"=SUM(E{WF_ROW_CONFIG}:E{WF_ROW_RESIDUAL})", STYLE_TOTAL_NUM),
        ("Before Step + Σ(Config + New + Tuning + Residual) == Reconstructed After Step", None, STYLE_TOTAL_TEXT),
    ))
    ws.merge_range(WF_ROW_RECON, WF_NOTE_START_COL, WF_ROW_RECON, PENDING_COLS)

    # Row 9: After Step receipt
    ws.set_row_height(WF_ROW_AFTER, WRAP_ROW_HEIGHT_PT)
    write_row(ws, WF_ROW_AFTER, (
        ("After Step receipt", None, STYLE_MERGED),
        ("", None, STYLE_TEXT),
        ("", None, STYLE_TEXT),
        (wf.after_step_ms, f"={SUM_AFT_STEP_REF}", STYLE_FLOAT3),
        (wf.before_upt - wf.after_upt, f"={SUM_BEF_UPT_REF}-{SUM_AFT_UPT_REF}", STYLE_FLOAT2),
        (wf.after_endpoint_error_ms, aft_ep_f, STYLE_ALERT_OK if abs(wf.after_endpoint_error_ms) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH),
        ("After Endpoint Δ = Sheet2 Grand Total P - Summary!$C$14 (ms)", None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(WF_ROW_AFTER, WF_NOTE_START_COL + 1, WF_ROW_AFTER, PENDING_COLS)

    # Row 10: Reconciliation error
    ws.set_row_height(WF_ROW_ERROR, WRAP_ROW_HEIGHT_PT)
    err_ms_st = STYLE_ALERT_OK if abs(wf.reconciliation_error_ms) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH
    err_upt_st = STYLE_ALERT_OK if abs(wf.reconciliation_error_upt) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH
    write_row(ws, WF_ROW_ERROR, (
        ("Reconciliation error", None, STYLE_TOTAL_TEXT),
        ("", None, STYLE_TOTAL_TEXT),
        ("", None, STYLE_TOTAL_TEXT),
        (wf.reconciliation_error_ms, f"=D{WF_ROW_RECON}-D{WF_ROW_AFTER}", err_ms_st),
        (wf.reconciliation_error_upt, f"=E{WF_ROW_RECON}-E{WF_ROW_AFTER}", err_upt_st),
        ("Reconstructed After - Summary After Step receipt (目标严格 = 0.000 ms)", None, STYLE_TOTAL_TEXT),
    ))
    ws.merge_range(WF_ROW_ERROR, WF_NOTE_START_COL, WF_ROW_ERROR, PENDING_COLS)


def _write_pending_kernel_card(
    ws: Worksheet,
    card_start: int,
    idx_row: int,
    item: PendingKernelItem,
    target_bundle: ExportBundle,
) -> None:
    """Render one structured 8-row physical-receipt kernel card (zero Config/Env row)."""
    is_new = item.category == PENDING_KIND_TO_IMPLEMENT
    cat_style = STYLE_NEW_KERNEL if is_new else STYLE_OPT_BADGE
    bind_badge = format_binding_badge(item.binding_state)
    gain_hdr = format_signed_gain_text(item.expected_gain_ms)

    ws.set_row_height(card_start, HEADER_ROW_HEIGHT_PT)
    write_banner(
        ws, card_start, PENDING_COLS,
        (
            f"Card #{item.priority} [{item.card_id}] | {item.category} {item.target_kernel} — {item.logical_role} "
            f"| Δ={gain_hdr} / Δ摊薄={item.saved_us_per_tok_die:.2f} us/(tok·die) {bind_badge}"
        ),
        STYLE_CARD_HDR,
    )

    r_st = card_start + PENDING_CARD_ROW_STACK
    ws.set_row_height(r_st, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_st, 1, "分类 & 绑定状态", style=STYLE_MERGED)
    ws.merge_range(r_st, 1, r_st, PENDING_CARD_LBL_END)
    ws.set_cell(r_st, PENDING_CARD_LEFT_START, f"{item.category} | {item.status} | {bind_badge}", style=cat_style)
    ws.merge_range(r_st, PENDING_CARD_LEFT_START, r_st, PENDING_CARD_LEFT_END)
    ws.set_cell(r_st, PENDING_CARD_MID_COL, "技术栈 (Backend)", style=STYLE_MERGED)
    ws.set_cell(r_st, PENDING_CARD_RIGHT_START, f"{item.suggested_backend} — {item.target_kernel}", style=STYLE_TEXT_WRAP)
    ws.merge_range(r_st, PENDING_CARD_RIGHT_START, r_st, PENDING_COLS)

    r_ch = card_start + PENDING_CARD_ROW_CHAIN
    ws.set_row_height(r_ch, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_ch, 1, "替代算子链 / 主归属模块", style=STYLE_MERGED)
    ws.merge_range(r_ch, 1, r_ch, PENDING_CARD_LBL_END)
    ws.set_cell(r_ch, PENDING_CARD_LEFT_START, f"[{item.submodule}] {item.replaces_chain}", style=STYLE_TEXT_WRAP)
    ws.merge_range(r_ch, PENDING_CARD_LEFT_START, r_ch, PENDING_CARD_LEFT_END)
    ws.set_cell(r_ch, PENDING_CARD_MID_COL, "源码出处 (Source Ref)", style=STYLE_MERGED)
    ws.set_cell(r_ch, PENDING_CARD_RIGHT_START, item.source_refs or "", style=STYLE_MUTED_TEXT)
    ws.merge_range(r_ch, PENDING_CARD_RIGHT_START, r_ch, PENDING_COLS)

    r_fr = card_start + PENDING_CARD_ROW_FREQ
    ws.set_row_height(r_fr, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_fr, 1, "全网频次 (Archetypes)", style=STYLE_MERGED)
    ws.merge_range(r_fr, 1, r_fr, PENDING_CARD_LBL_END)
    ws.set_cell(r_fr, PENDING_CARD_LEFT_START, f"{item.archetype_summary} ({item.total_calls_per_step} calls/step)", style=STYLE_TEXT_WRAP)
    ws.merge_range(r_fr, PENDING_CARD_LEFT_START, r_fr, PENDING_CARD_LEFT_END)
    ws.set_cell(r_fr, PENDING_CARD_MID_COL, "预期全网收益 Δ (ms)", style=STYLE_MERGED)
    ws.set_cell(r_fr, PENDING_CARD_RIGHT_START, item.expected_gain_ms, formula=f"=G{idx_row}", style=STYLE_KPI_HIGHLIGHT)
    attr_part = (
        f" | Kernel-Attributable Δ={format_signed_gain_text(item.attributable_gain_ms)}"
        if item.attributable_gain_ms is not None else ""
    )
    ws.set_cell(
        r_fr, PENDING_CARD_NOTE_START,
        (
            f"Before={item.before_model_ms:.3f}ms -> After={item.after_model_ms:.3f}ms{attr_part} | "
            f"Δ摊薄={item.saved_us_per_tok_die:.2f} us/(tok·die) | L*={item.lstar_model_ms:.3f}ms "
            f"(Observed L*/Step {item.before_attainment_pct:.1f}% -> {item.target_attainment_pct:.1f}%)"
        ),
        style=STYLE_TEXT_WRAP,
    )
    ws.merge_range(r_fr, PENDING_CARD_NOTE_START, r_fr, PENDING_COLS)

    budget_txt = f" | Core Budget: {item.core_budget}" if item.core_budget else ""
    contract_txt = f"{item.tensor_contract} | {target_bundle.hardware_die_summary}{budget_txt}"
    formula_txt = item.sram_residency_summary or item.formula_summary or item.kernel_action
    fw_txt = item.framework_action or "—"
    full_width_rows = (
        (PENDING_CARD_ROW_SHAPE, WRAP_ROW_HEIGHT_PT, "张量契约与硬件几何 (Contract & Geometry)", contract_txt, STYLE_TEXT_WRAP),
        (PENDING_CARD_ROW_BOUND, WRAP_ROW_HEIGHT_PT, "瓶颈与物理下界收据 (Bound & L* Receipt)", item.microarch_target, STYLE_TEXT_WRAP),
        (PENDING_CARD_ROW_KERNEL, CARD_DESIGN_ROW_HEIGHT_PT, "符号公式、SRAM 驻留与物理定价收据 (Formula & SRAM)", formula_txt, STYLE_TEXT_WRAP),
        (PENDING_CARD_ROW_FRAMEWORK, CARD_ACCEPT_ROW_HEIGHT_PT, "已应用结构图改写收据 (Applied Graph Rewrite)", fw_txt, STYLE_OPT_BADGE if item.framework_action else STYLE_MUTED_TEXT),
        (PENDING_CARD_ROW_RECEIPT, CARD_ACCEPT_ROW_HEIGHT_PT, "数值对账与状态收据 (Numeric Receipt)", item.default_acceptance_slot, STYLE_TEXT_WRAP),
    )
    for offset, height, label, text, st in full_width_rows:
        r = card_start + offset
        ws.set_row_height(r, height)
        ws.set_cell(r, 1, label, style=STYLE_MERGED)
        ws.merge_range(r, 1, r, PENDING_CARD_LBL_END)
        ws.set_cell(r, PENDING_CARD_LEFT_START, text, style=st)
        ws.merge_range(r, PENDING_CARD_LEFT_START, r, PENDING_COLS)


def _sheet2_spans_sum_formula(col_let: str, spans: Sequence[tuple[int, int]]) -> str:
    if not spans:
        return "=0"
    ordered = sorted((int(s), int(e)) for s, e in spans)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for s, e in ordered[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    tokens = [
        f"'{SHEET_BREAKDOWN}'!{col_let}{s}:{col_let}{e}" if s != e else f"'{SHEET_BREAKDOWN}'!{col_let}{s}"
        for s, e in merged
    ]
    if len(tokens) == 1 and ":" not in tokens[0]:
        return f"={tokens[0]}"
    return f"=SUM({','.join(tokens)})"


def populate_pending_kernel_details_sheet(
    ws: Worksheet,
    pending_items: Sequence[PendingKernelItem],
    target_bundle: ExportBundle,
    *,
    is_comparison: bool = True,
    waterfall: WaterfallSummary | None = None,
    breakdown_receipt: BreakdownSheetReceipt | None = None,
) -> None:
    """Populate Sheet 4 with live Waterfall Ledger + high-signal Kernel Cards."""
    apply_col_widths(ws, PENDING_COL_WIDTHS)
    write_banner(
        ws, 1, PENDING_COLS,
        f"端到端收益瀑布对账 & 算子攻关明细 (Waterfall & Kernel Details) | {target_bundle.model_name} @ {target_bundle.hardware_sku} ({target_bundle.hardware_die_summary})",
        STYLE_TITLE,
    )

    receipt = breakdown_receipt or BreakdownSheetReceipt({}, 0, 0)
    if is_comparison and waterfall is not None:
        _write_waterfall_ledger(ws, waterfall, receipt)
        idx_banner_row = CMP_INDEX_BANNER_ROW
        idx_hdr_row = CMP_INDEX_HEADER_ROW
        idx_data_start = CMP_INDEX_DATA_START_ROW
        ws.freeze_panes(WF_HDR_ROW, 1)
    else:
        idx_banner_row = STD_INDEX_BANNER_ROW
        idx_hdr_row = STD_INDEX_HEADER_ROW
        idx_data_start = STD_INDEX_DATA_START_ROW
        ws.freeze_panes(STD_INDEX_HEADER_ROW, 2)

    write_banner(
        ws, idx_banner_row, PENDING_COLS,
        (
            "2. 高价值算子攻关索引表 ([需新开发融合算子] & 聚合 >= 0.050 ms 的 [现成算子深度调优] — 最多 12 项)"
            if is_comparison else
            "1. 高价值算子攻关索引表 (Standalone Top L* Gap & New Fused Kernels)"
        ),
        STYLE_SECTION,
    )
    headers = (
        "优先级", "分类 (3-Category Taxonomy)", "目标算子 & 技术栈",
        "状态 (Status)", "主归属模块 & 替代链", "全网频次 (Archetypes & Calls)",
        "预期全网收益 Δ (ms)", "Before -> After (L* ms)", "Before -> After Bound",
        "详情卡片定位 (card_id)", "摊薄净省 Δ (us/tok·die)", "绑定状态 (Binding State)",
        "Before 暴露 (ms)" if is_comparison else "当前暴露 (ms)",
        "After 暴露 (ms)" if is_comparison else "L* 下界 (ms)",
    )
    write_headers(ws, idx_hdr_row, headers)

    if not pending_items:
        ws.set_cell(
            idx_data_start, 1,
            "无待新开发融合算子或达到阈值 (>= 0.050 ms 且 Δ摊薄 > 0) 的现成算子调优项；端到端对账见上方 Waterfall 表。",
            style=STYLE_OVERLAP,
        )
        ws.merge_range(idx_data_start, 1, idx_data_start, PENDING_COLS)
        return

    r_end = idx_data_start + len(pending_items) - 1
    r_tot = r_end + 1
    cards_banner_row = r_tot + 2
    cards_start_row = cards_banner_row + 1
    n_new = sum(1 for x in pending_items if x.category == PENDING_KIND_TO_IMPLEMENT)
    n_tune = len(pending_items) - n_new

    has_s2_rows = receipt.last_detail_row >= CMP_DATA_START_ROW
    ac_rng = _sheet2_ref(CMP_COL_CARD_ID, CMP_DATA_START_ROW, receipt.last_detail_row) if (is_comparison and has_s2_rows) else ""
    j_rng = _sheet2_ref(CMP_COL_BEF_EXP_MS, CMP_DATA_START_ROW, receipt.last_detail_row) if (is_comparison and has_s2_rows) else ""
    p_rng = _sheet2_ref(CMP_COL_AFT_EXP_MS, CMP_DATA_START_ROW, receipt.last_detail_row) if (is_comparison and has_s2_rows) else ""
    spans_map: Mapping[int, tuple[int, int]] = receipt.row_spans

    for idx, item in enumerate(pending_items):
        r = idx_data_start + idx
        card_row = cards_start_row + idx * PENDING_CARD_STRIDE
        is_new = item.category == PENDING_KIND_TO_IMPLEMENT
        ws.set_row_height(r, BACKLOG_ROW_HEIGHT_PT)
        m_val = item.before_model_ms if is_comparison else item.after_model_ms
        n_val = item.after_model_ms if is_comparison else item.lstar_model_ms
        if is_comparison:
            safe_id = item.card_id.replace('"', '""')
            m_formula = f'=SUMIF({ac_rng},"{safe_id}",{j_rng})' if has_s2_rows else "=0"
            n_formula = f'=SUMIF({ac_rng},"{safe_id}",{p_rng})' if has_s2_rows else "=0"
            gain_ms_f = f"=M{r}-N{r}"
            gain_upt_f = (
                f"=IF({SUM_BEF_NTOK_REF}>0,M{r}*1000/{SUM_BEF_NTOK_REF},0)"
                f"-IF({SUM_AFT_NTOK_REF}>0,N{r}*1000/{SUM_AFT_NTOK_REF},0)"
            )
        else:
            item_spans = [spans_map[i] for i in item.source_row_indices if i in spans_map]
            m_formula = _sheet2_spans_sum_formula(column_letter(STD_COL_EXP_MS), item_spans)
            n_formula = _sheet2_spans_sum_formula(column_letter(STD_COL_LSTAR_MS), item_spans)
            gain_ms_f = f"=MAX(0,M{r}-N{r})"
            gain_upt_f = f"=IF({SUM_AFT_NTOK_REF}>0,G{r}*1000/{SUM_AFT_NTOK_REF},0)"
        write_row(ws, r, (
            (f"P{item.priority}", None, STYLE_MERGED),
            (item.category, None, STYLE_NEW_KERNEL if is_new else STYLE_OPT_BADGE),
            (item.target_kernel, None, STYLE_TEXT_WRAP),
            (item.status, None, STYLE_NEW_KERNEL if is_new else STYLE_TEXT_WRAP),
            (f"[{item.submodule}] {item.replaces_chain}", None, STYLE_TEXT_WRAP),
            (f"{item.archetype_summary} ({item.total_calls_per_step} calls/step)", None, STYLE_TEXT_WRAP),
            (item.expected_gain_ms, gain_ms_f, STYLE_FLOAT3),
            (f"{item.before_model_ms:.3f} -> {item.after_model_ms:.3f} (L* {item.lstar_model_ms:.3f})", None, STYLE_TEXT_WRAP),
            (f"{item.before_bound} -> {item.after_bound}", None, STYLE_TEXT_WRAP),
            (f"Card #{item.priority} [{item.card_id}] (Row {card_row})", None, STYLE_MERGED),
            (item.saved_us_per_tok_die, gain_upt_f, STYLE_FLOAT2),
            (format_binding_badge(item.binding_state), None, STYLE_TEXT_WRAP),
            (m_val, m_formula, STYLE_FLOAT3),
            (n_val, n_formula, STYLE_FLOAT3),
        ))

    ws.set_cell(
        r_tot, 1,
        f"DISPLAYED CARDS SUBTOTAL ({n_new} 需新开发融合 + {n_tune} 现成算子调优 = {len(pending_items)} 项)",
        style=STYLE_TOTAL_TEXT,
    )
    ws.merge_range(r_tot, 1, r_tot, PENDING_LABEL_MERGE_END)
    ws.set_cell(
        r_tot, PENDING_COL_GAIN,
        sum(x.expected_gain_ms for x in pending_items),
        formula=f"=SUM(G{idx_data_start}:G{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )
    ws.set_cell(
        r_tot, PENDING_COL_GAIN + 1,
        "已展示高价值算子卡片小计（全网 100% 端到端对账见上方 Waterfall 表）",
        style=STYLE_TOTAL_TEXT,
    )
    ws.merge_range(r_tot, PENDING_COL_GAIN + 1, r_tot, PENDING_COL_GAIN_UPT - 1)
    ws.set_cell(
        r_tot, PENDING_COL_GAIN_UPT,
        sum(x.saved_us_per_tok_die for x in pending_items),
        formula=f"=SUM(K{idx_data_start}:K{r_end})",
        style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(r_tot, PENDING_COL_BINDING, format_binding_badge(target_bundle.binding_state), style=STYLE_TOTAL_TEXT)
    ws.set_cell(
        r_tot, PENDING_COL_BEF_MS,
        sum(x.before_model_ms if is_comparison else x.after_model_ms for x in pending_items),
        formula=f"=SUM(M{idx_data_start}:M{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )
    ws.set_cell(
        r_tot, PENDING_COL_AFT_MS,
        sum(x.after_model_ms if is_comparison else x.lstar_model_ms for x in pending_items),
        formula=f"=SUM(N{idx_data_start}:N{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )

    write_banner(
        ws, cards_banner_row, PENDING_COLS,
        "3. 逐算子物理收据卡片 (Structured Kernel Receipt Cards — Formula / SRAM / Applied Graph Rewrite)",
        STYLE_SECTION,
    )
    for idx, item in enumerate(pending_items):
        idx_row = idx_data_start + idx
        card_row = cards_start_row + idx * PENDING_CARD_STRIDE
        _write_pending_kernel_card(ws, card_row, idx_row, item, target_bundle)


__all__ = [
    "CMP_INDEX_DATA_START_ROW",
    "PENDING_CARD_STRIDE",
    "WF_ROW_AFTER",
    "WF_ROW_BEFORE",
    "WF_ROW_CONFIG",
    "WF_ROW_ERROR",
    "WF_ROW_NEW",
    "WF_ROW_RECON",
    "WF_ROW_RESIDUAL",
    "WF_ROW_TUNE",
    "populate_pending_kernel_details_sheet",
]
