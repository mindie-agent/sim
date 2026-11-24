"""Sheet 4 (``Pending_Kernel_Details``) live Waterfall, Kernel & Co-Design Cards writer.

[Role] Render Sheet 4 strictly within visible columns ``A..N`` (``PENDING_COLS = 14``):
  (1) Live-formula ``端到端收益瀑布对账表 (Waterfall Ledger)`` at rows 2..10
      separating ``[1] 配置与并行`` / ``[2] 框架图与多流改造`` /
      ``[3] 新算子开发与 Tiling 调优`` / ``[4] 残差与物理下界算子`` and
      compiling directly over coalesced disjoint visible Sheet 2 ``A:AA`` row
      spans (zero ``AB..AF`` columns, zero hidden sheets, 0 reconciliation error);
  (2) Automatic visible ``A:N`` Group Attribution Table fallback below the cards
      ONLY when a coalesced span formula would exceed Excel's 255-argument or
      8,192-character limit on a pathologically fragmented workbook;
  (3) High-signal actionable kernel index and structured physical-receipt cards
      plus typed ``CoDesignOpportunity`` cards (``MegaFusionBound``,
      ``SpaceTimeTradeoffSpec``, ``CrossHwOffloadSpec``) defaulting primary
      kernel backend recommendations to ``TileLang / AscendC / Catlass``.
[Invariants] Waterfall is always emitted on comparison workbooks, including
  identical bundles and zero-card reports; all formulas use bounded visible
  ranges and guarded denominators so zero Excel formula errors occur; Sheet 4
  populated columns never exceed ``14`` (``N``); zero legacy DSL strings.
[Timeouts & Locks] Pure in-memory OOXML cell assembly; no I/O or locks.
[Side-Effect Policy] Mutates only the caller-provided ``Worksheet``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from .action_plan import GroupAccounting, WaterfallSummary, format_signed_gain_text
from .comparison import (
    WF_STAGE_CONFIG,
    WF_STAGE_FRAMEWORK,
    WF_STAGE_KERNEL,
    WF_STAGE_RESIDUAL,
    classify_waterfall_stage,
)
from .models import (
    BACKEND_ASCENDC,
    BACKEND_CATLASS,
    BACKEND_TILELANG,
    GIB_BYTES,
    KIB_BYTES,
    MIB_BYTES,
    PENDING_KIND_TO_IMPLEMENT,
    ExportBundle,
    PendingKernelItem,
    format_binding_badge,
)
from .ooxml import (
    BACKLOG_ROW_HEIGHT_PT,
    CARD_ACCEPT_ROW_HEIGHT_PT,
    CARD_DESIGN_ROW_HEIGHT_PT,
    EXCEL_MAX_FORMULA_ARGS,
    EXCEL_MAX_FORMULA_CHARS,
    HEADER_ROW_HEIGHT_PT,
    STYLE_ALERT_HIGH,
    STYLE_ALERT_OK,
    STYLE_CARD_HDR,
    STYLE_FLOAT2,
    STYLE_FLOAT3,
    STYLE_INT,
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
    CMP_COL_AFT_UPT,
    CMP_COL_BEF_EXP_MS,
    CMP_COL_BEF_UPT,
    CMP_COL_SAVED_MS,
    CMP_COL_SAVED_UPT,
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

#: Sheet 4 structural layout constants (strictly 14 columns A..N).
PENDING_COLS = 14
WF_HDR_ROW = 2
WF_ROW_BEFORE = 3
WF_ROW_CONFIG = 4
WF_ROW_FRAMEWORK = 5
WF_ROW_KERNEL = 6
WF_ROW_RESIDUAL = 7
WF_ROW_RECON = 8
WF_ROW_AFTER = 9
WF_ROW_ERROR = 10
#: Backward-compatible aliases preserving exact row indices 5 & 6.
WF_ROW_NEW = WF_ROW_FRAMEWORK
WF_ROW_TUNE = WF_ROW_KERNEL
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
PENDING_CARD_STRIDE = 9
CODESIGN_CARD_STRIDE = 5
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
CODESIGN_ROW_META = 1
CODESIGN_ROW_PRIMARY = 2
CODESIGN_ROW_BUDGET = 3
ATTR_TABLE_GAP_ROWS = 2
CARDS_BANNER_GAP_ROWS = 2
MAX_CODESIGN_OPS_PREVIEW = 4
CMP_NEXT_SECTION_NUM = 3
STD_NEXT_SECTION_NUM = 2

WF_RECON_TOL_MS = 1e-4
PRIMARY_KERNEL_BACKENDS = f"{BACKEND_TILELANG} / {BACKEND_ASCENDC} / {BACKEND_CATLASS}"
_LEGACY_BACKEND_RE = re.compile(r"legacy[-_]?dsl", re.IGNORECASE)
_STAGES_ORDERED = (WF_STAGE_CONFIG, WF_STAGE_FRAMEWORK, WF_STAGE_KERNEL, WF_STAGE_RESIDUAL)
_STAGE_META: dict[str, tuple[int, str, str]] = {
    WF_STAGE_CONFIG: (
        WF_ROW_CONFIG,
        "[1] 配置与并行 (Config & Parallelism)",
        "并行切分、批量缩放、量化与投机策略配置净变化 (直连 Sheet 2 可见列 A:AA)",
    ),
    WF_STAGE_FRAMEWORK: (
        WF_ROW_FRAMEWORK,
        "[2] 框架图与多流改造 (Framework Graph & Multi-Stream)",
        "框架图改写、现成融合使能、多流掩盖与图内增删算子净变化 (直连 Sheet 2 可见列 A:AA)",
    ),
    WF_STAGE_KERNEL: (
        WF_ROW_KERNEL,
        "[3] 新算子开发与 Tiling 调优 (New Kernel Dev & Tiling Tuning)",
        "需新开发融合算子与现成算子深度 Tiling 调优净收益 (扣除前置配置/框架分摊后)",
    ),
    WF_STAGE_RESIDUAL: (
        WF_ROW_RESIDUAL,
        "[4] 残差与物理下界算子 (Residual Ops)",
        "已近物理下界 / 低于展示阈值 / 胶水残差算子 (0 卡片)",
    ),
}
_SOURCE_TIER_ZH: dict[str, str] = {
    "host_ddr": "Host DDR H2D",
    "host_dram": "Host DDR H2D",
    "pinned_ddr": "Host DDR H2D",
    "ssd": "NVMe SSD",
    "remote_hbm": "Remote Die HBM",
}
_CODESIGN_KIND_ZH: dict[str, str] = {
    "mega_fusion": "超级融合算子 (MegaFusion)",
    "space_time_tradeoff": "时空权衡 (以存代算 / 以算代存)",
    "cross_hw_offload_multistream": "跨层卸载与多流掩盖 (Cross-HW Offload)",
    "tiler_tune": "Tiling 调度优化 (Tiler Tune)",
}

PENDING_COL_WIDTHS = (
    12.0, 26.0, 28.0, 26.0, 28.0, 24.0, 15.0, 22.0, 24.0, 22.0, 16.0, 26.0, 14.0, 14.0,
)


@dataclass(frozen=True, slots=True)
class _AttrRowRecord:
    row_index: int
    r1: int
    r2: int
    primary_stage: str
    pre_stage: str
    card_id: str
    before_ms: float
    after_ms: float
    saved_ms: float
    pre_edit_ms: float | None
    pre_share_ms: float
    post_share_ms: float
    saved_upt: float
    pre_share_upt: float
    post_share_upt: float


@dataclass(frozen=True, slots=True)
class _StagePlan:
    stage_ms: dict[str, float]
    stage_upt: dict[str, float]
    direct_wf_ms: dict[str, str]
    direct_wf_upt: dict[str, str]
    direct_card_m: dict[str, str]
    direct_card_n: dict[str, str]
    attr_rows: tuple[_AttrRowRecord, ...]
    needs_fallback_table: bool


def _sanitize_backend_text(text: str) -> str:
    cleaned = _LEGACY_BACKEND_RE.sub(BACKEND_TILELANG, str(text or ""))
    return cleaned


def _coalesce_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce adjacent/overlapping Sheet 2 detail spans without crossing Subtotals."""
    if not spans:
        return []
    ordered = sorted((int(s), int(e)) for s, e in spans)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for s, e in ordered[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _build_sum_expr(col_let: str, spans: Sequence[tuple[int, int]]) -> tuple[str, int]:
    """Return ``(excel_expr_without_leading_eq, arg_count)`` over coalesced Sheet 2 spans."""
    merged = _coalesce_spans(spans)
    if not merged:
        return "", 0
    tokens = [
        f"'{SHEET_BREAKDOWN}'!{col_let}{s}:{col_let}{e}" if s != e else f"'{SHEET_BREAKDOWN}'!{col_let}{s}"
        for s, e in merged
    ]
    if len(tokens) == 1 and ":" not in tokens[0]:
        return tokens[0], 1
    return f"SUM({','.join(tokens)})", len(tokens)


def _sheet2_spans_sum_formula(col_let: str, spans: Sequence[tuple[int, int]]) -> str:
    expr, _ = _build_sum_expr(col_let, spans)
    return f"={expr}" if expr else "=0"


def _build_stage_formula(
    *,
    unsplit_spans: Sequence[tuple[int, int]],
    pre_spans: Sequence[tuple[int, int]],
    pre_c_sum: float,
    post_spans: Sequence[tuple[int, int]],
    post_c_sum: float,
    unsplit_col: str,
    before_col: str,
    after_col: str,
    is_upt: bool,
) -> tuple[str, bool]:
    """Compile one Waterfall bucket formula directly over visible Sheet 2 ``A:AA`` spans."""
    terms: list[str] = []
    max_args = 0
    u_expr, u_args = _build_sum_expr(unsplit_col, unsplit_spans)
    max_args = max(max_args, u_args)
    if u_expr:
        terms.append(u_expr)
    if pre_spans:
        b_expr, b_args = _build_sum_expr(before_col, pre_spans)
        max_args = max(max_args, b_args)
        c_term = (
            f"IF({SUM_AFT_NTOK_REF}>0,{pre_c_sum!r}*1000/{SUM_AFT_NTOK_REF},0)"
            if is_upt else f"{pre_c_sum!r}"
        )
        terms.append(f"({b_expr}-{c_term})")
    if post_spans:
        a_expr, a_args = _build_sum_expr(after_col, post_spans)
        max_args = max(max_args, a_args)
        c_term = (
            f"IF({SUM_AFT_NTOK_REF}>0,{post_c_sum!r}*1000/{SUM_AFT_NTOK_REF},0)"
            if is_upt else f"{post_c_sum!r}"
        )
        terms.append(f"({c_term}-{a_expr})")
    formula = f"={'+'.join(terms)}" if terms else "=0"
    exceeds = max_args > EXCEL_MAX_FORMULA_ARGS or len(formula) > EXCEL_MAX_FORMULA_CHARS
    return formula, exceeds


def _compile_stage_plan(
    wf: WaterfallSummary,
    receipt: BreakdownSheetReceipt,
    pending_items: Sequence[PendingKernelItem],
    group_accounting: Sequence[GroupAccounting],
    before_bundle: ExportBundle | None,
    target_bundle: ExportBundle,
) -> _StagePlan:
    """Build exact 4-stage attribution, direct span formulas, and fallback-limit audit."""
    stage_ms = {st: 0.0 for st in _STAGES_ORDERED}
    stage_upt = {st: 0.0 for st in _STAGES_ORDERED}
    unsplit_spans: dict[str, list[tuple[int, int]]] = {st: [] for st in _STAGES_ORDERED}
    pre_spans: dict[str, list[tuple[int, int]]] = {st: [] for st in _STAGES_ORDERED}
    post_spans: dict[str, list[tuple[int, int]]] = {st: [] for st in _STAGES_ORDERED}
    pre_c: dict[str, float] = {st: 0.0 for st in _STAGES_ORDERED}
    post_c: dict[str, float] = {st: 0.0 for st in _STAGES_ORDERED}
    attr_rows: list[_AttrRowRecord] = []
    bef_ref = before_bundle or target_bundle

    if group_accounting and receipt.row_spans:
        for acc in group_accounting:
            span = receipt.row_spans.get(acc.row_index)
            if span is None:
                continue
            r1, r2 = span
            primary_st, pre_st = classify_waterfall_stage(acc, bef_ref, target_bundle)
            if pre_st is None:
                unsplit_spans[primary_st].append(span)
                stage_ms[primary_st] += acc.saved_ms
                stage_upt[primary_st] += acc.saved_upt
                is_pre_bucket = primary_st in {WF_STAGE_CONFIG, WF_STAGE_FRAMEWORK}
                pre_ms = acc.saved_ms if is_pre_bucket else 0.0
                post_ms = 0.0 if is_pre_bucket else acc.saved_ms
                pre_u = acc.saved_upt if is_pre_bucket else 0.0
                post_u = 0.0 if is_pre_bucket else acc.saved_upt
            else:
                c_val = float(acc.after_pre_edit_ms or 0.0)
                pre_spans[pre_st].append(span)
                pre_c[pre_st] += c_val
                stage_ms[pre_st] += acc.config_share_ms
                stage_upt[pre_st] += acc.config_share_upt
                post_spans[primary_st].append(span)
                post_c[primary_st] += c_val
                stage_ms[primary_st] += acc.kernel_share_ms
                stage_upt[primary_st] += acc.kernel_share_upt
                pre_ms, post_ms = acc.config_share_ms, acc.kernel_share_ms
                pre_u, post_u = acc.config_share_upt, acc.kernel_share_upt
            attr_rows.append(_AttrRowRecord(
                row_index=acc.row_index, r1=r1, r2=r2, primary_stage=primary_st,
                pre_stage=pre_st or "", card_id=acc.card_id,
                before_ms=acc.before_ms, after_ms=acc.after_ms, saved_ms=acc.saved_ms,
                pre_edit_ms=acc.after_pre_edit_ms, pre_share_ms=pre_ms, post_share_ms=post_ms,
                saved_upt=acc.saved_upt, pre_share_upt=pre_u, post_share_upt=post_u,
            ))
    else:
        stage_ms[WF_STAGE_CONFIG] = wf.config_saved_ms
        stage_ms[WF_STAGE_KERNEL] = wf.new_kernel_saved_ms + wf.tune_kernel_saved_ms
        stage_ms[WF_STAGE_RESIDUAL] = wf.residual_saved_ms
        stage_upt[WF_STAGE_CONFIG] = wf.config_saved_upt
        stage_upt[WF_STAGE_KERNEL] = wf.new_kernel_saved_upt + wf.tune_kernel_saved_upt
        stage_upt[WF_STAGE_RESIDUAL] = wf.residual_saved_upt

    j_let = column_letter(CMP_COL_BEF_EXP_MS)
    p_let = column_letter(CMP_COL_AFT_EXP_MS)
    q_let = column_letter(CMP_COL_SAVED_MS)
    x_let = column_letter(CMP_COL_BEF_UPT)
    y_let = column_letter(CMP_COL_AFT_UPT)
    z_let = column_letter(CMP_COL_SAVED_UPT)

    direct_wf_ms: dict[str, str] = {}
    direct_wf_upt: dict[str, str] = {}
    needs_fallback = False
    for st in _STAGES_ORDERED:
        if attr_rows:
            f_ms, ex_ms = _build_stage_formula(
                unsplit_spans=unsplit_spans[st], pre_spans=pre_spans[st], pre_c_sum=pre_c[st],
                post_spans=post_spans[st], post_c_sum=post_c[st],
                unsplit_col=q_let, before_col=j_let, after_col=p_let, is_upt=False,
            )
            f_upt, ex_upt = _build_stage_formula(
                unsplit_spans=unsplit_spans[st], pre_spans=pre_spans[st], pre_c_sum=pre_c[st],
                post_spans=post_spans[st], post_c_sum=post_c[st],
                unsplit_col=z_let, before_col=x_let, after_col=y_let, is_upt=True,
            )
            needs_fallback = needs_fallback or ex_ms or ex_upt
        else:
            f_ms, f_upt = f"={stage_ms[st]!r}", f"={stage_upt[st]!r}"
        direct_wf_ms[st] = f_ms
        direct_wf_upt[st] = f_upt

    direct_card_m: dict[str, str] = {}
    direct_card_n: dict[str, str] = {}
    for item in pending_items:
        item_spans = [receipt.row_spans[i] for i in item.source_row_indices if i in receipt.row_spans]
        m_expr, m_args = _build_sum_expr(j_let, item_spans)
        n_expr, n_args = _build_sum_expr(p_let, item_spans)
        m_f = f"={m_expr}" if m_expr else "=0"
        n_f = f"={n_expr}" if n_expr else "=0"
        if (
            max(m_args, n_args) > EXCEL_MAX_FORMULA_ARGS
            or max(len(m_f), len(n_f)) > EXCEL_MAX_FORMULA_CHARS
        ):
            needs_fallback = True
        direct_card_m[item.card_id] = m_f
        direct_card_n[item.card_id] = n_f

    return _StagePlan(
        stage_ms=stage_ms, stage_upt=stage_upt,
        direct_wf_ms=direct_wf_ms, direct_wf_upt=direct_wf_upt,
        direct_card_m=direct_card_m, direct_card_n=direct_card_n,
        attr_rows=tuple(attr_rows),
        needs_fallback_table=(needs_fallback and bool(attr_rows)),
    )


def _fallback_waterfall_formulas(
    t1: int, t2: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build compact SUMIF Waterfall formulas over the visible Sheet 4 ``A:N`` attribution table."""
    c_rng = f"$C${t1}:$C${t2}"
    d_rng = f"$D${t1}:$D${t2}"
    j_rng = f"$J${t1}:$J${t2}"
    k_rng = f"$K${t1}:$K${t2}"
    m_rng = f"$M${t1}:$M${t2}"
    n_rng = f"$N${t1}:$N${t2}"
    wf_ms = {
        WF_STAGE_CONFIG: f'=SUMIF({c_rng},"config",{j_rng})+SUMIF({d_rng},"config",{j_rng})',
        WF_STAGE_FRAMEWORK: f'=SUMIF({c_rng},"framework",{j_rng})+SUMIF({d_rng},"framework",{j_rng})',
        WF_STAGE_KERNEL: f'=SUMIF({c_rng},"kernel",{k_rng})',
        WF_STAGE_RESIDUAL: f'=SUMIF({c_rng},"residual",{k_rng})',
    }
    wf_upt = {
        WF_STAGE_CONFIG: f'=SUMIF({c_rng},"config",{m_rng})+SUMIF({d_rng},"config",{m_rng})',
        WF_STAGE_FRAMEWORK: f'=SUMIF({c_rng},"framework",{m_rng})+SUMIF({d_rng},"framework",{m_rng})',
        WF_STAGE_KERNEL: f'=SUMIF({c_rng},"kernel",{n_rng})',
        WF_STAGE_RESIDUAL: f'=SUMIF({c_rng},"residual",{n_rng})',
    }
    return wf_ms, wf_upt


def _write_waterfall_ledger(
    ws: Worksheet,
    wf: WaterfallSummary,
    receipt: BreakdownSheetReceipt,
    plan: _StagePlan,
    *,
    wf_ms_formulas: Mapping[str, str],
    wf_upt_formulas: Mapping[str, str],
) -> None:
    """Write live-formula 3-stage + residual Waterfall Ledger at rows 2..10."""
    write_headers(ws, WF_HDR_ROW, (
        "端到端收益瀑布对账表 (Waterfall Ledger)",
        "净省 Δ (ms)",
        "单步变化 (ms)",
        "累计单步耗时 (ms)",
        "摊薄净省 Δ (us/tok·die)",
        "会计口径与 Sheet 2 端点校验 (Accounting Basis & Endpoint Audit)",
    ))
    ws.merge_range(WF_HDR_ROW, WF_NOTE_START_COL, WF_HDR_ROW, PENDING_COLS)

    gt = receipt.grand_total_row
    j_let = column_letter(CMP_COL_BEF_EXP_MS)
    p_let = column_letter(CMP_COL_AFT_EXP_MS)
    bef_ep_f = f"='{SHEET_BREAKDOWN}'!{j_let}{gt}-{SUM_BEF_STEP_REF}" if gt >= CMP_DATA_START_ROW else "=0"
    aft_ep_f = f"='{SHEET_BREAKDOWN}'!{p_let}{gt}-{SUM_AFT_STEP_REF}" if gt >= CMP_DATA_START_ROW else "=0"

    run_cursor = wf.before_step_ms
    tot_saved_ms = sum(plan.stage_ms[st] for st in _STAGES_ORDERED)
    tot_saved_upt = sum(plan.stage_upt[st] for st in _STAGES_ORDERED)
    recon_after_ms = wf.before_step_ms - tot_saved_ms
    recon_err_ms = recon_after_ms - wf.after_step_ms
    recon_err_upt = tot_saved_upt - (wf.before_upt - wf.after_upt)

    ws.set_row_height(WF_ROW_BEFORE, WRAP_ROW_HEIGHT_PT)
    write_row(ws, WF_ROW_BEFORE, (
        ("Before Step", None, STYLE_MERGED),
        ("", None, STYLE_TEXT),
        ("", None, STYLE_TEXT),
        (wf.before_step_ms, f"={SUM_BEF_STEP_REF}", STYLE_FLOAT3),
        ("", None, STYLE_TEXT),
        (wf.before_endpoint_error_ms, bef_ep_f, STYLE_ALERT_OK if abs(wf.before_endpoint_error_ms) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH),
        ("Before 端点校验 = Sheet 2 全网暴露合计(J) - Summary 单步耗时 (ms)", None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(WF_ROW_BEFORE, WF_NOTE_START_COL + 1, WF_ROW_BEFORE, PENDING_COLS)

    for st in _STAGES_ORDERED:
        r, lbl, note = _STAGE_META[st]
        b_val = plan.stage_ms[st]
        e_val = plan.stage_upt[st]
        run_cursor -= b_val
        ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
        write_row(ws, r, (
            (lbl, None, STYLE_TEXT),
            (b_val, wf_ms_formulas[st], STYLE_FLOAT3),
            (-b_val, f"=-B{r}", STYLE_FLOAT3),
            (run_cursor, f"=D{r - 1}+C{r}", STYLE_FLOAT3),
            (e_val, wf_upt_formulas[st], STYLE_FLOAT2),
            (note, None, STYLE_TEXT_WRAP),
        ))
        ws.merge_range(r, WF_NOTE_START_COL, r, PENDING_COLS)

    ws.set_row_height(WF_ROW_RECON, WRAP_ROW_HEIGHT_PT)
    write_row(ws, WF_ROW_RECON, (
        ("Reconstructed After Step", None, STYLE_TOTAL_TEXT),
        (tot_saved_ms, f"=SUM(B{WF_ROW_CONFIG}:B{WF_ROW_RESIDUAL})", STYLE_TOTAL_FLOAT3),
        (-tot_saved_ms, f"=SUM(C{WF_ROW_CONFIG}:C{WF_ROW_RESIDUAL})", STYLE_TOTAL_FLOAT3),
        (recon_after_ms, f"=D{WF_ROW_BEFORE}+C{WF_ROW_RECON}", STYLE_TOTAL_FLOAT3),
        (tot_saved_upt, f"=SUM(E{WF_ROW_CONFIG}:E{WF_ROW_RESIDUAL})", STYLE_TOTAL_NUM),
        ("Before Step + Σ([1]配置 + [2]框架图 + [3]算子 + [4]残差) == Reconstructed After Step", None, STYLE_TOTAL_TEXT),
    ))
    ws.merge_range(WF_ROW_RECON, WF_NOTE_START_COL, WF_ROW_RECON, PENDING_COLS)

    ws.set_row_height(WF_ROW_AFTER, WRAP_ROW_HEIGHT_PT)
    write_row(ws, WF_ROW_AFTER, (
        ("After Step receipt", None, STYLE_MERGED),
        ("", None, STYLE_TEXT),
        ("", None, STYLE_TEXT),
        (wf.after_step_ms, f"={SUM_AFT_STEP_REF}", STYLE_FLOAT3),
        (wf.before_upt - wf.after_upt, f"={SUM_BEF_UPT_REF}-{SUM_AFT_UPT_REF}", STYLE_FLOAT2),
        (wf.after_endpoint_error_ms, aft_ep_f, STYLE_ALERT_OK if abs(wf.after_endpoint_error_ms) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH),
        ("After 端点校验 = Sheet 2 全网暴露合计(P) - Summary 单步耗时 (ms)", None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(WF_ROW_AFTER, WF_NOTE_START_COL + 1, WF_ROW_AFTER, PENDING_COLS)

    ws.set_row_height(WF_ROW_ERROR, WRAP_ROW_HEIGHT_PT)
    err_ms_st = STYLE_ALERT_OK if abs(recon_err_ms) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH
    err_upt_st = STYLE_ALERT_OK if abs(recon_err_upt) <= WF_RECON_TOL_MS else STYLE_ALERT_HIGH
    write_row(ws, WF_ROW_ERROR, (
        ("Reconciliation error", None, STYLE_TOTAL_TEXT),
        ("", None, STYLE_TOTAL_TEXT),
        ("", None, STYLE_TOTAL_TEXT),
        (recon_err_ms, f"=D{WF_ROW_RECON}-D{WF_ROW_AFTER}", err_ms_st),
        (recon_err_upt, f"=E{WF_ROW_RECON}-E{WF_ROW_AFTER}", err_upt_st),
        ("Reconstructed After - Summary After Step receipt (目标严格 = 0.000 ms)", None, STYLE_TOTAL_TEXT),
    ))
    ws.merge_range(WF_ROW_ERROR, WF_NOTE_START_COL, WF_ROW_ERROR, PENDING_COLS)


def _format_megafusion_tilelang_and_ub(mf: Any) -> tuple[str, str]:
    """Return ``(dual_tier_note, ub_warning_note)`` for an A5 ``MegaFusionBound``."""
    simt_ms = getattr(mf, "tilelang_simt_gain_ms", None)
    simd_ms = getattr(mf, "tilelang_simd_or_ascendc_gain_ms", None)
    dual_parts: list[str] = []
    if simt_ms is not None:
        dual_parts.append(f"TileLang (SimtVF quick-win)={float(simt_ms):+.3f} ms")
    if simd_ms is not None:
        dual_parts.append(f"TileLang (SimdVF) / AscendC={float(simd_ms):+.3f} ms")
    dual_txt = f"A5 双档预估: {' vs '.join(dual_parts)}" if dual_parts else ""
    ub_warn = _sanitize_backend_text(str(getattr(mf, "ub_wall_warning", "") or "").strip())
    ub_txt = f"[UB-Wall 预警] {ub_warn}" if ub_warn else ""
    return dual_txt, ub_txt


def _matching_megafusion_notes(item: PendingKernelItem, target_bundle: ExportBundle) -> str:
    """Surface matching A5 TileLang dual-tier / UB-wall guidance on a Pending Kernel Card."""
    notes: list[str] = []
    synth = getattr(item, "synthesis_spec", None)
    synth_ub = _sanitize_backend_text(str(getattr(synth, "ub_wall_warning", "") or "").strip())
    if synth_ub:
        notes.append(f"[UB-Wall 预警] {synth_ub}")
    item_tokens = {
        str(item.kernel_key or "").strip().lower(),
        str(item.target_kernel or "").strip().lower(),
    }
    for opp in getattr(target_bundle, "co_design_opportunities", ()) or ():
        mf = getattr(opp, "mega_fusion", None)
        if mf is None:
            continue
        opp_text = f"{getattr(opp, 'opportunity_id', '')} {getattr(opp, 'title', '')} {' '.join(mf.region_op_ids)}".lower()
        if any(tok and tok in opp_text for tok in item_tokens):
            dual_txt, ub_txt = _format_megafusion_tilelang_and_ub(mf)
            if dual_txt and dual_txt not in notes:
                notes.append(dual_txt)
            if ub_txt and ub_txt not in notes:
                notes.append(ub_txt)
    return " | ".join(notes)


def _write_pending_kernel_card(
    ws: Worksheet,
    card_start: int,
    idx_row: int,
    item: PendingKernelItem,
    target_bundle: ExportBundle,
) -> None:
    """Render one structured 8-row physical-receipt kernel card (strictly inside A:N)."""
    is_new = item.category == PENDING_KIND_TO_IMPLEMENT
    cat_style = STYLE_NEW_KERNEL if is_new else STYLE_OPT_BADGE
    bind_badge = format_binding_badge(item.binding_state)
    gain_hdr = format_signed_gain_text(item.expected_gain_ms)
    clean_target = _sanitize_backend_text(item.target_kernel)
    clean_backend = _sanitize_backend_text(item.suggested_backend) or PRIMARY_KERNEL_BACKENDS

    ws.set_row_height(card_start, HEADER_ROW_HEIGHT_PT)
    write_banner(
        ws, card_start, PENDING_COLS,
        (
            f"Card #{item.priority} [{item.card_id}] | {item.category} {clean_target} — {item.logical_role} "
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
    ws.set_cell(r_st, PENDING_CARD_MID_COL, "推荐技术栈 (Backend)", style=STYLE_MERGED)
    ws.set_cell(r_st, PENDING_CARD_RIGHT_START, f"{clean_backend} — {clean_target}", style=STYLE_TEXT_WRAP)
    ws.merge_range(r_st, PENDING_CARD_RIGHT_START, r_st, PENDING_COLS)

    r_ch = card_start + PENDING_CARD_ROW_CHAIN
    ws.set_row_height(r_ch, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_ch, 1, "替代算子链 / 主归属模块", style=STYLE_MERGED)
    ws.merge_range(r_ch, 1, r_ch, PENDING_CARD_LBL_END)
    ws.set_cell(r_ch, PENDING_CARD_LEFT_START, f"[{item.submodule}] {_sanitize_backend_text(item.replaces_chain)}", style=STYLE_TEXT_WRAP)
    ws.merge_range(r_ch, PENDING_CARD_LEFT_START, r_ch, PENDING_CARD_LEFT_END)
    ws.set_cell(r_ch, PENDING_CARD_MID_COL, "源码出处 (Source Ref)", style=STYLE_MERGED)
    ws.set_cell(r_ch, PENDING_CARD_RIGHT_START, _sanitize_backend_text(item.source_refs or ""), style=STYLE_MUTED_TEXT)
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
    formula_txt = _sanitize_backend_text(item.sram_residency_summary or item.formula_summary or item.kernel_action)
    mf_extra = _matching_megafusion_notes(item, target_bundle)
    if mf_extra:
        formula_txt = f"{formula_txt} | {mf_extra}" if formula_txt else mf_extra
    fw_txt = _sanitize_backend_text(item.framework_action or "—")
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


def _format_bytes_human(num_bytes: int) -> str:
    abs_b = abs(int(num_bytes))
    sign = "-" if int(num_bytes) < 0 else ("+" if int(num_bytes) > 0 else "")
    if abs_b >= GIB_BYTES:
        return f"{sign}{abs_b / float(GIB_BYTES):.3f} GiB"
    if abs_b >= MIB_BYTES:
        return f"{sign}{abs_b / float(MIB_BYTES):.2f} MiB"
    return f"{sign}{abs_b / float(KIB_BYTES):.1f} KiB"


def _write_codesign_opportunity_cards(
    ws: Worksheet, start_row: int, opportunities: Sequence[Any], *, section_num: int,
) -> int:
    """Render typed ``CoDesignOpportunity`` cards strictly inside columns ``A:N``."""
    if not opportunities:
        return start_row
    write_banner(
        ws, start_row, PENDING_COLS,
        f"{section_num}. 软硬协同优化机会卡片 (Co-Design Opportunities — MegaFusion / 以存代算 / 跨层卸载)",
        STYLE_SECTION,
    )
    cursor = start_row + 1
    for idx, opp in enumerate(opportunities, start=1):
        opp_id = str(getattr(opp, "opportunity_id", f"codesign_{idx}"))
        kind = str(getattr(opp, "kind", ""))
        kind_zh = _CODESIGN_KIND_ZH.get(kind, kind)
        title = _sanitize_backend_text(str(getattr(opp, "title", opp_id)))
        track = str(getattr(opp, "track", ""))
        readiness = str(getattr(opp, "readiness", ""))
        ev_refs = "; ".join(
            _sanitize_backend_text(str(ref)) for ref in (getattr(opp, "evidence_refs", ()) or ()) if ref
        )
        mf = getattr(opp, "mega_fusion", None)
        st = getattr(opp, "space_time", None)
        xh = getattr(opp, "cross_hw", None)

        backend_txt = PRIMARY_KERNEL_BACKENDS if (mf is not None or kind in {"mega_fusion", "tiler_tune"}) else "框架调度 / 显存管理"
        ws.set_row_height(cursor, HEADER_ROW_HEIGHT_PT)
        write_banner(
            ws, cursor, PENDING_COLS,
            f"Co-Design #{idx} [{opp_id}] | {kind_zh} | {title} | 推荐技术栈: {backend_txt}",
            STYLE_CARD_HDR,
        )

        r_meta = cursor + CODESIGN_ROW_META
        ws.set_row_height(r_meta, WRAP_ROW_HEIGHT_PT)
        ws.set_cell(r_meta, 1, "轨道 & 就绪状态", style=STYLE_MERGED)
        ws.merge_range(r_meta, 1, r_meta, PENDING_CARD_LBL_END)
        ws.set_cell(r_meta, PENDING_CARD_LEFT_START, f"Track={track} | Readiness={readiness} | Backend={backend_txt}", style=STYLE_OPT_BADGE)
        ws.merge_range(r_meta, PENDING_CARD_LEFT_START, r_meta, PENDING_CARD_LEFT_END)
        ws.set_cell(r_meta, PENDING_CARD_MID_COL, "证据引用 (Evidence)", style=STYLE_MERGED)
        ws.set_cell(r_meta, PENDING_CARD_RIGHT_START, ev_refs or "—", style=STYLE_MUTED_TEXT)
        ws.merge_range(r_meta, PENDING_CARD_RIGHT_START, r_meta, PENDING_COLS)

        if mf is not None:
            dual_txt, ub_txt = _format_megafusion_tilelang_and_ub(mf)
            p_lbl = "理论上限 Δ_max vs 硬件预演预期 Δ_expected"
            p_txt = (
                f"融合区间 ({len(mf.region_op_ids)} ops: {', '.join(mf.region_op_ids[:MAX_CODESIGN_OPS_PREVIEW])}) | "
                f"Before={mf.exposed_before_ms:.3f} ms | 边界下界 L*={mf.boundary_lower_bound_ms:.3f} ms | "
                f"硬件预演={mf.tiled_engine_ms:.3f} ms | 理论上限 Δ_max={mf.gain_ceiling_ms:+.3f} ms vs "
                f"硬件预演预期 Δ_expected={mf.gain_expected_ms:+.3f} ms"
            )
            if dual_txt:
                p_txt = f"{p_txt} | {dual_txt}"
            b_lbl = "GM 访存消除 & 片上 UB 约束"
            b_txt = (
                f"消除 GM 访存={_format_bytes_human(mf.eliminated_gm_bytes).lstrip('+')} | "
                f"片上 UB 驻留={mf.ub_live_bytes / float(KIB_BYTES):.1f} KiB / "
                f"{mf.ub_capacity_bytes / float(KIB_BYTES):.1f} KiB | "
                f"推荐实现技术栈: {PRIMARY_KERNEL_BACKENDS}"
            )
            if ub_txt:
                b_txt = f"{b_txt} | {ub_txt}"
            b_style = STYLE_ALERT_HIGH if ub_txt else STYLE_TEXT_WRAP
        elif st is not None:
            dir_zh = "以存代算 (Space-for-Time)" if st.direction == "space_for_time" else "以算代存 (Time-for-Space)"
            feas_zh = "可行 (满足 HBM & SLA 预算)" if st.feasible_under_hbm_and_sla else "受限 (超出当前 HBM/SLA 余量)"
            p_lbl = "时空权衡机制与 HBM 余量"
            p_txt = (
                f"{dir_zh} | 机制={st.mechanism} | HBM 增量={_format_bytes_human(st.delta_hbm_bytes)} | "
                f"可用 HBM 余量={_format_bytes_human(st.free_hbm_headroom_bytes).lstrip('+')} | 判定: {feas_zh}"
            )
            b_lbl = "单步延迟 / 并发 / 吞吐影响"
            b_txt = (
                f"单步耗时变化 Δ={st.delta_step_ms:+.3f} ms | "
                f"单 DP 并发变化 Δ={st.delta_concurrency_per_dp:+d} | "
                f"单卡吞吐变化 Δ={st.delta_tps_per_die:+.2f} tok/s/die"
            )
            b_style = STYLE_ALERT_OK if st.feasible_under_hbm_and_sla else STYLE_ALERT_HIGH
        elif xh is not None:
            tier_zh = _SOURCE_TIER_ZH.get(str(xh.source_tier).lower(), str(xh.source_tier))
            p_lbl = "跨层传输 vs NPU 掩盖窗口"
            p_txt = (
                f"目标组件={xh.target_component} | 源介质={tier_zh} ({xh.transfer_engine}) | "
                f"传输耗时={xh.transfer_ms:.3f} ms vs NPU 计算掩盖窗口={xh.compute_hide_window_ms:.3f} ms | "
                f"净暴露延迟={xh.net_exposed_ms:.3f} ms"
            )
            b_lbl = "释放 HBM 与并发收益"
            b_txt = (
                f"释放 HBM={_format_bytes_human(xh.freed_hbm_bytes).lstrip('+')}/die | "
                f"单 DP 并发变化 Δ={xh.delta_concurrency_per_dp:+d} | "
                f"单卡吞吐变化 Δ={xh.delta_tps_per_die:+.2f} tok/s/die"
            )
            b_style = STYLE_OVERLAP if xh.net_exposed_ms <= WF_RECON_TOL_MS else STYLE_TEXT_WRAP
        else:
            p_lbl, p_txt = "优化说明 (Summary)", title
            b_lbl, b_txt, b_style = "推荐技术栈 (Backend)", backend_txt, STYLE_TEXT_WRAP

        for offset, lbl, txt, st_cell in (
            (CODESIGN_ROW_PRIMARY, p_lbl, p_txt, STYLE_TEXT_WRAP),
            (CODESIGN_ROW_BUDGET, b_lbl, b_txt, b_style),
        ):
            r = cursor + offset
            ws.set_row_height(r, CARD_ACCEPT_ROW_HEIGHT_PT)
            ws.set_cell(r, 1, lbl, style=STYLE_MERGED)
            ws.merge_range(r, 1, r, PENDING_CARD_LBL_END)
            ws.set_cell(r, PENDING_CARD_LEFT_START, txt, style=st_cell)
            ws.merge_range(r, PENDING_CARD_LEFT_START, r, PENDING_COLS)
        cursor += CODESIGN_CARD_STRIDE
    return cursor


def _write_visible_attribution_table(
    ws: Worksheet, start_row: int, attr_rows: Sequence[_AttrRowRecord], *, section_num: int,
) -> tuple[int, int]:
    """Emit visible ``A:N`` Group Attribution Table ONLY when direct span limits are exceeded."""
    write_banner(
        ws, start_row, PENDING_COLS,
        f"{section_num}. 分组归因明细表 (Visible Group Attribution Table — 大规模碎片化回退对账，严格位于 A:N)",
        STYLE_SECTION,
    )
    hdr_row = start_row + 1
    write_headers(ws, hdr_row, (
        "Group Idx", "Sheet2 Span", "Primary Stage", "Pre-Edit Stage", "Card ID",
        "Before (ms)", "After (ms)", "Saved (ms)", "Pre-Edit C (ms)",
        "Pre Share (ms)", "Post Share (ms)", "Saved (us/tok·die)",
        "Pre Share (us/tok·die)", "Post Share (us/tok·die)",
    ))
    j_let = column_letter(CMP_COL_BEF_EXP_MS)
    p_let = column_letter(CMP_COL_AFT_EXP_MS)
    x_let = column_letter(CMP_COL_BEF_UPT)
    z_let = column_letter(CMP_COL_SAVED_UPT)

    t1 = hdr_row + 1
    for offset, rec in enumerate(attr_rows):
        t = t1 + offset
        b_expr = _sheet2_spans_sum_formula(j_let, ((rec.r1, rec.r2),))
        a_expr = _sheet2_spans_sum_formula(p_let, ((rec.r1, rec.r2),))
        z_expr = _sheet2_spans_sum_formula(z_let, ((rec.r1, rec.r2),))
        x_expr, _ = _build_sum_expr(x_let, ((rec.r1, rec.r2),))
        if rec.pre_stage:
            pre_ms_f = f"=F{t}-I{t}"
            pre_u_f = f"={x_expr}-IF({SUM_AFT_NTOK_REF}>0,I{t}*1000/{SUM_AFT_NTOK_REF},0)"
        elif rec.primary_stage in {WF_STAGE_CONFIG, WF_STAGE_FRAMEWORK}:
            pre_ms_f = f"=H{t}"
            pre_u_f = f"=L{t}"
        else:
            pre_ms_f = "=0"
            pre_u_f = "=0"
        write_row(ws, t, (
            (rec.row_index, None, STYLE_INT),
            (f"{rec.r1}:{rec.r2}", None, STYLE_TEXT),
            (rec.primary_stage, None, STYLE_TEXT),
            (rec.pre_stage, None, STYLE_TEXT),
            (rec.card_id, None, STYLE_MUTED_TEXT),
            (rec.before_ms, b_expr, STYLE_FLOAT3),
            (rec.after_ms, a_expr, STYLE_FLOAT3),
            (rec.saved_ms, f"=F{t}-G{t}", STYLE_FLOAT3),
            ("" if rec.pre_edit_ms is None else rec.pre_edit_ms, None, STYLE_TEXT if rec.pre_edit_ms is None else STYLE_FLOAT3),
            (rec.pre_share_ms, pre_ms_f, STYLE_FLOAT3),
            (rec.post_share_ms, f"=H{t}-J{t}", STYLE_FLOAT3),
            (rec.saved_upt, z_expr, STYLE_FLOAT2),
            (rec.pre_share_upt, pre_u_f, STYLE_FLOAT2),
            (rec.post_share_upt, f"=L{t}-M{t}", STYLE_FLOAT2),
        ))
    return t1, t1 + len(attr_rows) - 1


def _write_pending_index_totals(
    ws: Worksheet,
    pending_items: Sequence[PendingKernelItem],
    target_bundle: ExportBundle,
    *,
    idx_data_start: int,
    r_end: int,
    r_tot: int,
    is_comparison: bool,
) -> None:
    n_new = sum(1 for x in pending_items if x.category == PENDING_KIND_TO_IMPLEMENT)
    n_tune = len(pending_items) - n_new
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


def _write_pending_index_and_cards(
    ws: Worksheet,
    pending_items: Sequence[PendingKernelItem],
    target_bundle: ExportBundle,
    receipt: BreakdownSheetReceipt,
    stage_plan: _StagePlan | None,
    *,
    idx_data_start: int,
    r_end: int,
    r_tot: int,
    cards_banner_row: int,
    cards_start_row: int,
    fallback_t1: int,
    fallback_t2: int,
    is_comparison: bool,
) -> None:
    """Write Sheet 4 backlog index rows, totals row, and structured kernel receipt cards."""
    spans_map: Mapping[int, tuple[int, int]] = receipt.row_spans
    for idx, item in enumerate(pending_items):
        r = idx_data_start + idx
        card_row = cards_start_row + idx * PENDING_CARD_STRIDE
        is_new = item.category == PENDING_KIND_TO_IMPLEMENT
        ws.set_row_height(r, BACKLOG_ROW_HEIGHT_PT)
        m_val = item.before_model_ms if is_comparison else item.after_model_ms
        n_val = item.after_model_ms if is_comparison else item.lstar_model_ms
        if is_comparison:
            if stage_plan is not None and stage_plan.needs_fallback_table:
                safe_id = item.card_id.replace('"', '""')
                m_formula = f'=SUMIF($E${fallback_t1}:$E${fallback_t2},"{safe_id}",$F${fallback_t1}:$F${fallback_t2})'
                n_formula = f'=SUMIF($E${fallback_t1}:$E${fallback_t2},"{safe_id}",$G${fallback_t1}:$G${fallback_t2})'
            elif stage_plan is not None and item.card_id in stage_plan.direct_card_m:
                m_formula = stage_plan.direct_card_m[item.card_id]
                n_formula = stage_plan.direct_card_n[item.card_id]
            else:
                item_spans = [spans_map[i] for i in item.source_row_indices if i in spans_map]
                m_formula = _sheet2_spans_sum_formula(column_letter(CMP_COL_BEF_EXP_MS), item_spans)
                n_formula = _sheet2_spans_sum_formula(column_letter(CMP_COL_AFT_EXP_MS), item_spans)
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
            (_sanitize_backend_text(item.target_kernel), None, STYLE_TEXT_WRAP),
            (item.status, None, STYLE_NEW_KERNEL if is_new else STYLE_TEXT_WRAP),
            (f"[{item.submodule}] {_sanitize_backend_text(item.replaces_chain)}", None, STYLE_TEXT_WRAP),
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

    _write_pending_index_totals(
        ws, pending_items, target_bundle,
        idx_data_start=idx_data_start, r_end=r_end, r_tot=r_tot,
        is_comparison=is_comparison,
    )

    write_banner(
        ws, cards_banner_row, PENDING_COLS,
        f"3. 逐算子物理收据卡片 (Structured Kernel Receipt Cards — 推荐技术栈: {PRIMARY_KERNEL_BACKENDS})",
        STYLE_SECTION,
    )
    for idx, item in enumerate(pending_items):
        idx_row = idx_data_start + idx
        card_row = cards_start_row + idx * PENDING_CARD_STRIDE
        _write_pending_kernel_card(ws, card_row, idx_row, item, target_bundle)


def populate_pending_kernel_details_sheet(
    ws: Worksheet,
    pending_items: Sequence[PendingKernelItem],
    target_bundle: ExportBundle,
    *,
    is_comparison: bool = True,
    waterfall: WaterfallSummary | None = None,
    breakdown_receipt: BreakdownSheetReceipt | None = None,
    before_bundle: ExportBundle | None = None,
    group_accounting: Sequence[GroupAccounting] = (),
    co_design_opportunities: Sequence[Any] = (),
) -> None:
    """Populate Sheet 4 with live Waterfall, Kernel Cards, Co-Design Cards, and A:N fallback."""
    apply_col_widths(ws, PENDING_COL_WIDTHS)
    write_banner(
        ws, 1, PENDING_COLS,
        f"端到端收益瀑布对账 & 算子攻关明细 (Waterfall & Kernel Details) | {target_bundle.model_name} @ {target_bundle.hardware_sku} ({target_bundle.hardware_die_summary})",
        STYLE_TITLE,
    )

    receipt = breakdown_receipt or BreakdownSheetReceipt({}, 0, 0)
    opps = tuple(co_design_opportunities) if co_design_opportunities else tuple(
        getattr(target_bundle, "co_design_opportunities", ()) or ()
    )
    has_wf = is_comparison and waterfall is not None
    idx_banner_row = CMP_INDEX_BANNER_ROW if has_wf else STD_INDEX_BANNER_ROW
    idx_hdr_row = CMP_INDEX_HEADER_ROW if has_wf else STD_INDEX_HEADER_ROW
    idx_data_start = CMP_INDEX_DATA_START_ROW if has_wf else STD_INDEX_DATA_START_ROW
    ws.freeze_panes(WF_HDR_ROW if has_wf else STD_INDEX_HEADER_ROW, 1 if has_wf else 2)

    stage_plan = (
        _compile_stage_plan(
            waterfall, receipt, pending_items, group_accounting, before_bundle, target_bundle,
        )
        if has_wf and waterfall is not None else None
    )

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
        "优先级", "分类 (3-Category Taxonomy)", "目标算子 & 推荐技术栈",
        "状态 (Status)", "主归属模块 & 替代链", "全网频次 (Archetypes & Calls)",
        "预期全网收益 Δ (ms)", "Before -> After (L* ms)", "Before -> After Bound",
        "详情卡片定位 (card_id)", "摊薄净省 Δ (us/tok·die)", "绑定状态 (Binding State)",
        "Before 暴露 (ms)" if is_comparison else "当前暴露 (ms)",
        "After 暴露 (ms)" if is_comparison else "L* 下界 (ms)",
    )
    write_headers(ws, idx_hdr_row, headers)

    next_section_num = CMP_NEXT_SECTION_NUM if is_comparison else STD_NEXT_SECTION_NUM
    if pending_items:
        r_end = idx_data_start + len(pending_items) - 1
        r_tot = r_end + 1
        cards_banner_row = r_tot + CARDS_BANNER_GAP_ROWS
        cards_start_row = cards_banner_row + 1
        after_cards_row = cards_start_row + len(pending_items) * PENDING_CARD_STRIDE
        next_section_num += 1
    else:
        r_end = idx_data_start
        r_tot = idx_data_start
        cards_banner_row = 0
        cards_start_row = 0
        after_cards_row = idx_data_start + CARDS_BANNER_GAP_ROWS

    codesign_end_row = after_cards_row
    if opps:
        codesign_end_row = _write_codesign_opportunity_cards(
            ws, after_cards_row, opps, section_num=next_section_num,
        )
        next_section_num += 1

    fallback_t1 = fallback_t2 = 0
    if stage_plan is not None and stage_plan.needs_fallback_table:
        fallback_t1, fallback_t2 = _write_visible_attribution_table(
            ws, codesign_end_row + ATTR_TABLE_GAP_ROWS, stage_plan.attr_rows,
            section_num=next_section_num,
        )

    if has_wf and waterfall is not None and stage_plan is not None:
        if stage_plan.needs_fallback_table:
            wf_ms_f, wf_upt_f = _fallback_waterfall_formulas(fallback_t1, fallback_t2)
        else:
            wf_ms_f, wf_upt_f = stage_plan.direct_wf_ms, stage_plan.direct_wf_upt
        _write_waterfall_ledger(
            ws, waterfall, receipt, stage_plan,
            wf_ms_formulas=wf_ms_f, wf_upt_formulas=wf_upt_f,
        )

    if not pending_items:
        ws.set_cell(
            idx_data_start, 1,
            "无待新开发融合算子或达到阈值 (>= 0.050 ms 且 Δ摊薄 > 0) 的现成算子调优项；端到端对账见上方 Waterfall 表。",
            style=STYLE_OVERLAP,
        )
        ws.merge_range(idx_data_start, 1, idx_data_start, PENDING_COLS)
        ws.validate_max_column(PENDING_COLS)
        return

    _write_pending_index_and_cards(
        ws, pending_items, target_bundle, receipt, stage_plan,
        idx_data_start=idx_data_start, r_end=r_end, r_tot=r_tot,
        cards_banner_row=cards_banner_row, cards_start_row=cards_start_row,
        fallback_t1=fallback_t1, fallback_t2=fallback_t2,
        is_comparison=is_comparison,
    )
    ws.validate_max_column(PENDING_COLS)


__all__ = [
    "CMP_INDEX_DATA_START_ROW",
    "CODESIGN_CARD_STRIDE",
    "PENDING_CARD_STRIDE",
    "PENDING_COLS",
    "PRIMARY_KERNEL_BACKENDS",
    "WF_ROW_AFTER",
    "WF_ROW_BEFORE",
    "WF_ROW_CONFIG",
    "WF_ROW_ERROR",
    "WF_ROW_FRAMEWORK",
    "WF_ROW_KERNEL",
    "WF_ROW_NEW",
    "WF_ROW_RECON",
    "WF_ROW_RESIDUAL",
    "WF_ROW_TUNE",
    "populate_pending_kernel_details_sheet",
]
