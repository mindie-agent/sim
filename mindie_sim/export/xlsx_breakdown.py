"""Sheet 2 (``Layer_Operator_Breakdown``) & conditional ``Measured_msprof_Calibration`` writers.

[Role] Render topological Before-vs-After operator breakdown strictly inside
  visible columns ``A..AA`` (``CMP_COLS = 27``, zero ``AB..AF`` accounting
  columns), standalone breakdown (``A..P``), and conditional 1-to-N
  ``Measured_msprof_Calibration`` worksheet (``A..N``).
[Invariants] Every independent ``N->1`` fusion renders as ONE contiguous merged
  After cell with zero ``[↑ 已并入`` split rows; Sheet 2 populated column index
  never exceeds ``27`` (``AA``); zero internal proof-engine or raw internal track tokens;
  zero free-text prose overrides; zero guessed ``.py`` source paths.
[Timeouts & Locks] Single-pass O(Aligned Rows) OOXML cell population; no I/O.
[Side-Effect Policy] Mutates only caller-provided ``Worksheet`` instances.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .action_plan import GroupAccounting
from .models import (
    MIN_DENOM_EPS,
    OVERLAP_EXPOSED_EPS_US,
    US_PER_MS,
    AlignedComparisonRow,
    ExportBundle,
    OperatorExportRow,
    format_binding_badge,
)
from .public_view import format_track_label
from .ooxml import (
    STYLE_FLOAT2,
    STYLE_FLOAT3,
    STYLE_INT,
    STYLE_MERGED,
    STYLE_MUTED_TEXT,
    STYLE_NEW_KERNEL,
    STYLE_OPT_BADGE,
    STYLE_OVERLAP,
    STYLE_PARTIAL_OVERLAP,
    STYLE_PCT,
    STYLE_RATIO,
    STYLE_SUBTOTAL_NUM,
    STYLE_SUBTOTAL_RATIO,
    STYLE_SUBTOTAL_TEXT,
    STYLE_TEXT,
    STYLE_TEXT_WRAP,
    STYLE_TITLE,
    STYLE_TOTAL_FLOAT3,
    STYLE_TOTAL_NUM,
    STYLE_TOTAL_PCT,
    STYLE_TOTAL_TEXT,
    WRAP_ROW_HEIGHT_PT,
    Worksheet,
    column_letter,
)
from .xlsx_summary import (
    PCT_DIVISOR,
    SUM_AFT_NTOK_REF,
    SUM_BEF_NTOK_REF,
    apply_col_widths,
    extend_span,
    write_banner,
    write_headers,
    write_row,
)

#: Sheet 2 Before-vs-After mode column indices (strictly 27 visible columns A..AA).
CMP_COLS = 27
CMP_DATA_START_ROW = 3
CMP_FREEZE_ROWS = 2
CMP_FREEZE_COLS = 5
CMP_COL_ARCH = 1
CMP_COL_REP = 2
CMP_COL_SUB = 3
CMP_COL_ROLE = 4
CMP_COL_STREAM_MASK = 5
CMP_COL_BEF_EXP_MS = 10
CMP_COL_BEF_BOUND = 11
CMP_COL_AFT_KERN = 12
CMP_COL_STATUS = 13
CMP_COL_AFT_RAW_US = 14
CMP_COL_AFT_EXP_US = 15
CMP_COL_AFT_EXP_MS = 16
CMP_COL_SAVED_MS = 17
CMP_COL_SPEEDUP = 18
CMP_COL_AFT_BOUND = 19
CMP_COL_LSTAR_MS = 20
CMP_COL_GAP_MS = 21
CMP_COL_ACTION = 22
CMP_COL_SOURCE_REF = 23
CMP_COL_BEF_UPT = 24
CMP_COL_AFT_UPT = 25
CMP_COL_SAVED_UPT = 26
CMP_COL_BINDING = 27

CMP_AFTER_MERGE_COLS = (
    CMP_COL_AFT_KERN, CMP_COL_STATUS, CMP_COL_AFT_RAW_US, CMP_COL_AFT_EXP_US,
    CMP_COL_AFT_EXP_MS, CMP_COL_SAVED_MS, CMP_COL_SPEEDUP, CMP_COL_AFT_BOUND,
    CMP_COL_LSTAR_MS, CMP_COL_GAP_MS, CMP_COL_ACTION,
    CMP_COL_AFT_UPT, CMP_COL_SAVED_UPT, CMP_COL_BINDING,
)
CMP_CLUSTER_MERGE_COLS = (
    CMP_COL_SAVED_MS, CMP_COL_SPEEDUP, CMP_COL_GAP_MS, CMP_COL_ACTION, CMP_COL_SAVED_UPT,
)
CMP_SUBTOTAL_LABEL_END_COL = 9

#: Sheet 2 Standalone mode column indices (16 columns A..P).
STD_COLS = 16
STD_DATA_START_ROW = 3
STD_COL_ARCH = 1
STD_COL_REP = 2
STD_COL_SUB = 3
STD_COL_ROLE = 4
STD_COL_EXP_MS = 11
STD_COL_LSTAR_US = 12
STD_COL_LSTAR_MS = 13
STD_COL_GAP_MS = 14
STD_COL_BOUND = 15
STD_COL_SOURCE_REF = 16
STD_SUBTOTAL_LABEL_END_COL = 10

#: Conditional Measured_msprof_Calibration sheet layout constants (14 columns A..N).
MEAS_COLS = 14
MEAS_DATA_START_ROW = 3
MEAS_SIM_MERGE_COLS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14)
MEAS_COL_SIM_US = 9
MEAS_COL_KERN_NAME = 10
MEAS_COL_KERN_US = 11
MEAS_COL_SUM_US = 12
MEAS_COL_APE = 13
MEAS_COL_BOUND = 14
MEAS_TOTAL_SUM_COLS = (8, 9, 11, 12)

CMP_COL_WIDTHS = (
    30.0, 9.0, 22.0, 28.0, 34.0, 32.0, 26.0, 13.0, 13.0, 14.0, 20.0,
    26.0, 28.0, 13.0, 13.0, 14.0, 13.0, 10.0, 20.0, 13.0, 12.0, 38.0, 30.0,
    15.0, 15.0, 15.0, 26.0,
)
STD_COL_WIDTHS = (
    30.0, 9.0, 22.0, 28.0, 34.0, 32.0, 26.0, 28.0, 13.0, 13.0,
    14.0, 13.0, 13.0, 12.0, 20.0, 30.0,
)
MEAS_COL_WIDTHS = (
    28.0, 22.0, 24.0, 24.0, 24.0, 28.0, 12.0, 14.0, 14.0, 28.0, 14.0, 14.0, 12.0, 20.0,
)


@dataclass(frozen=True, slots=True)
class BreakdownSheetReceipt:
    """Coordinate receipt returned by Sheet 2 writers for Sheet 4 live formulas."""

    row_spans: dict[int, tuple[int, int]]
    last_detail_row: int
    grand_total_row: int


def _stream_masking_badge(op: OperatorExportRow) -> tuple[str, int]:
    """Build up-front Stream & Masking label + badge style for Sheet 2 Col E."""
    if op.dur_us <= 0.0:
        return "主计算流 [0 us]", STYLE_TEXT
    if op.is_overlapped:
        masker = op.masked_by or "主计算流"
        stream_tag = "副流并发" if op.stream_index == 1 else ("通信流并发" if op.stream_index == 2 else "并行分支")
        return f"{stream_tag} [被 {masker} 100% 掩盖 -> 暴露 0 us，不计入总耗时]", STYLE_OVERLAP
    masked_us = max(0.0, op.dur_us - op.exposed_dur_us)
    if masked_us > OVERLAP_EXPOSED_EPS_US:
        pct = 100.0 * masked_us / max(op.dur_us, OVERLAP_EXPOSED_EPS_US)
        masker = op.masked_by or "主计算流"
        stream_tag = "副流并发" if op.stream_index == 1 else ("通信流并发" if op.stream_index == 2 else "并行分支")
        return f"{stream_tag} [被 {masker} 掩盖 {pct:.0f}% -> 仅暴露 {op.exposed_dur_us:.2f} us]", STYLE_PARTIAL_OVERLAP
    if op.stream_index == 2 or op.task_type == "HCCL":
        return "通信流 [暴露 100%]", STYLE_TEXT
    if op.stream_index == 1:
        return "副流 [暴露 100%]", STYLE_TEXT
    return "主计算流 [暴露 100%]", STYLE_TEXT


def _sanitize_backend_text(text: str) -> str:
    return str(text or "")


def _kernel_with_backend(op: OperatorExportRow) -> str:
    kern = _sanitize_backend_text(op.ascend_kernel or op.implementation)
    backend = _sanitize_backend_text(op.kernel_backend)
    return f"{kern} [{backend}]" if backend else kern


def _write_cmp_before_sides(
    ws: Worksheet, r_start: int, row_count: int, item: AlignedComparisonRow,
    rep: int, b_ntok: float, *, b_rep: int | None = None,
) -> tuple[float, float]:
    """Populate Before columns (B, D..K, W, X) across ``row_count`` rows."""
    eff_b_rep = int(b_rep) if b_rep is not None else rep
    same_rep = eff_b_rep == rep
    before_ms_sum = 0.0
    before_upt_sum = 0.0
    a_primary = item.after_ops[0] if item.after_ops else None
    for offset in range(row_count):
        r = r_start + offset
        ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
        ws.set_cell(r, CMP_COL_REP, rep, style=STYLE_INT)
        ms_formula = f"=I{r}*B{r}/1000" if same_rep else f"=I{r}*{eff_b_rep}/1000"
        upt_formula = f"=IF({SUM_BEF_NTOK_REF}>0,J{r}*1000/{SUM_BEF_NTOK_REF},0)"
        if offset < len(item.before_ops):
            b_op = item.before_ops[offset]
            b_exp_us = 0.0 if b_op.is_overlapped else b_op.exposed_dur_us
            b_ms = b_exp_us * eff_b_rep / US_PER_MS
            b_upt = b_ms * US_PER_MS / b_ntok
            before_ms_sum += b_ms
            before_upt_sum += b_upt
            mask_ref = a_primary if (a_primary is not None and len(item.before_ops) == 1 and len(item.after_ops) == 1) else b_op
            stream_txt, stream_style = _stream_masking_badge(mask_ref)
            write_row(ws, r, (
                (b_op.logical_role or b_op.op_id, None, STYLE_TEXT_WRAP),
                (stream_txt, None, stream_style),
                (b_op.tensor_contract, None, STYLE_TEXT_WRAP),
                (_kernel_with_backend(b_op), None, STYLE_TEXT_WRAP),
                (b_op.dur_us, None, STYLE_FLOAT3),
                (b_exp_us, None, STYLE_FLOAT3),
                (b_ms, ms_formula, STYLE_FLOAT3),
                (b_op.bound_display, None, STYLE_OVERLAP if b_op.is_overlapped else STYLE_TEXT),
            ), start_col=CMP_COL_ROLE)
            ws.set_cell(r, CMP_COL_SOURCE_REF, b_op.source_ref or "", style=STYLE_MUTED_TEXT)
            ws.set_cell(r, CMP_COL_BEF_UPT, b_upt, formula=upt_formula, style=STYLE_FLOAT2)
        else:
            ref_a = item.after_ops[offset] if offset < len(item.after_ops) else a_primary
            assert ref_a is not None
            stream_txt, stream_style = _stream_masking_badge(ref_a)
            placeholder = "— [优化后新增, 0.000 ms]" if item.relation == "added" else "(Cluster Pad)"
            write_row(ws, r, (
                (ref_a.logical_role or ref_a.op_id, None, STYLE_TEXT_WRAP),
                (stream_txt, None, stream_style),
                (ref_a.tensor_contract, None, STYLE_TEXT_WRAP),
                (placeholder, None, STYLE_TEXT),
                (0.0, None, STYLE_FLOAT3),
                (0.0, None, STYLE_FLOAT3),
                (0.0, ms_formula, STYLE_FLOAT3),
                ("-", None, STYLE_TEXT),
            ), start_col=CMP_COL_ROLE)
            ws.set_cell(r, CMP_COL_SOURCE_REF, ref_a.source_ref or "", style=STYLE_MUTED_TEXT)
            ws.set_cell(r, CMP_COL_BEF_UPT, 0.0, formula=upt_formula, style=STYLE_FLOAT2)
    return before_ms_sum, before_upt_sum


def _write_cmp_after_cluster(
    ws: Worksheet, r_start: int, r_end: int, item: AlignedComparisonRow,
    rep: int, before_ms_sum: float, before_upt_sum: float, a_ntok: float,
    *, a_rep: int | None = None,
) -> tuple[float, float, float]:
    """Populate After columns for M->N (len(after_ops) >= 2) re-association clusters."""
    eff_a_rep = int(a_rep) if a_rep is not None else rep
    same_rep = eff_a_rep == rep
    aft_ms_sum = lstar_ms_sum = aft_upt_sum = 0.0
    for offset in range(r_end - r_start + 1):
        r = r_start + offset
        if offset < len(item.after_ops):
            a_op = item.after_ops[offset]
            a_exp_us = 0.0 if a_op.is_overlapped else a_op.exposed_dur_us
            a_ms = a_exp_us * eff_a_rep / US_PER_MS
            l_ms = 0.0 if a_op.is_overlapped else (a_op.l_star_lean_us * eff_a_rep / US_PER_MS)
            a_upt = a_ms * US_PER_MS / a_ntok
            aft_ms_sum += a_ms
            lstar_ms_sum += l_ms
            aft_upt_sum += a_upt
            st_style = (
                STYLE_NEW_KERNEL if a_op.needs_new_kernel
                else (STYLE_OPT_BADGE if a_op.opt_pillar in {"pillar2_roofline", "pillar3_comm"} else STYLE_TEXT_WRAP)
            )
            ws.set_cell(r, CMP_COL_AFT_KERN, _kernel_with_backend(a_op), style=STYLE_TEXT_WRAP)
            ws.set_cell(r, CMP_COL_STATUS, a_op.kernel_status, style=st_style)
            ws.set_cell(r, CMP_COL_AFT_RAW_US, a_op.dur_us, style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_EXP_US, a_exp_us, style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_EXP_MS, a_ms, formula=f"=O{r}*B{r}/1000" if same_rep else f"=O{r}*{eff_a_rep}/1000", style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_BOUND, a_op.bound_display, style=STYLE_OVERLAP if a_op.is_overlapped else STYLE_TEXT)
            ws.set_cell(r, CMP_COL_LSTAR_MS, l_ms, style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_UPT, a_upt, formula=f"=IF({SUM_AFT_NTOK_REF}>0,P{r}*1000/{SUM_AFT_NTOK_REF},0)", style=STYLE_FLOAT2)
            ws.set_cell(r, CMP_COL_BINDING, format_binding_badge(a_op.binding_state), style=STYLE_TEXT_WRAP)
        else:
            ws.set_cell(r, CMP_COL_AFT_KERN, "(Cluster Pad)", style=STYLE_TEXT)
            ws.set_cell(r, CMP_COL_STATUS, "-", style=STYLE_TEXT)
            ws.set_cell(r, CMP_COL_AFT_RAW_US, 0.0, style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_EXP_US, 0.0, style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_EXP_MS, 0.0, formula="=0", style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_BOUND, "-", style=STYLE_TEXT)
            ws.set_cell(r, CMP_COL_LSTAR_MS, 0.0, style=STYLE_FLOAT3)
            ws.set_cell(r, CMP_COL_AFT_UPT, 0.0, formula="=0", style=STYLE_FLOAT2)
            ws.set_cell(r, CMP_COL_BINDING, "-", style=STYLE_TEXT)
    spd = (before_ms_sum / aft_ms_sum) if aft_ms_sum > 0.0 else 0.0
    ws.set_cell(r_start, CMP_COL_SAVED_MS, before_ms_sum - aft_ms_sum, formula=f"=SUM(J{r_start}:J{r_end})-SUM(P{r_start}:P{r_end})", style=STYLE_FLOAT3)
    ws.set_cell(r_start, CMP_COL_SPEEDUP, spd, formula=f"=IF(SUM(P{r_start}:P{r_end})>0,SUM(J{r_start}:J{r_end})/SUM(P{r_start}:P{r_end}),0)", style=STYLE_RATIO)
    ws.set_cell(r_start, CMP_COL_GAP_MS, max(0.0, aft_ms_sum - lstar_ms_sum), formula=f"=MAX(0,SUM(P{r_start}:P{r_end})-SUM(T{r_start}:T{r_end}))", style=STYLE_FLOAT3)
    ws.set_cell(r_start, CMP_COL_ACTION, _sanitize_backend_text(item.action_note), style=STYLE_TEXT_WRAP)
    ws.set_cell(r_start, CMP_COL_SAVED_UPT, before_upt_sum - aft_upt_sum, formula=f"=SUM(X{r_start}:X{r_end})-SUM(Y{r_start}:Y{r_end})", style=STYLE_FLOAT2)
    if r_end > r_start:
        for col in CMP_CLUSTER_MERGE_COLS:
            ws.merge_range(r_start, col, r_end, col)
    return aft_ms_sum, lstar_ms_sum, aft_upt_sum


def _write_comparison_group(
    ws: Worksheet,
    r_start: int,
    r_end: int,
    item: AlignedComparisonRow,
    acc: GroupAccounting | None,
    b_ntok: float,
    a_ntok: float,
) -> tuple[float, float, float, float, float]:
    b_rep = item.effective_before_repeat
    a_rep = item.effective_after_repeat
    rep = a_rep or b_rep or max(1, int(item.repeat_count))
    row_count = r_end - r_start + 1
    before_ms_sum, before_upt_sum = _write_cmp_before_sides(
        ws, r_start, row_count, item, rep, b_ntok, b_rep=b_rep,
    )

    if len(item.after_ops) >= 2:
        aft_ms, lstar_ms, aft_upt = _write_cmp_after_cluster(
            ws, r_start, r_end, item, rep, before_ms_sum, before_upt_sum, a_ntok, a_rep=a_rep,
        )
        return before_ms_sum, aft_ms, lstar_ms, before_upt_sum, aft_upt

    a_primary = item.after_ops[0] if item.after_ops else None
    if a_primary is not None:
        a_exp_us = 0.0 if a_primary.is_overlapped else a_primary.exposed_dur_us
        aft_ms = a_exp_us * a_rep / US_PER_MS
        lstar_ms = 0.0 if a_primary.is_overlapped else (a_primary.l_star_lean_us * a_rep / US_PER_MS)
        aft_upt = aft_ms * US_PER_MS / a_ntok
        saved_ms = before_ms_sum - aft_ms
        speedup = (before_ms_sum / aft_ms) if aft_ms > 0.0 else 0.0
        gap_ms = max(0.0, aft_ms - lstar_ms)
        status_txt = acc.category if (acc is not None and acc.bucket in {"new", "tune"}) else (
            a_primary.kernel_status if a_primary.kernel_status else (acc.category if acc is not None else "")
        )
        status_style = (
            STYLE_NEW_KERNEL if (acc is not None and acc.bucket == "new")
            else (STYLE_OPT_BADGE if (acc is not None and acc.bucket == "tune") else STYLE_TEXT_WRAP)
        )
        aft_ms_formula = f"=O{r_start}*B{r_start}/1000" if a_rep == rep else f"=O{r_start}*{a_rep}/1000"
        write_row(ws, r_start, (
            (_kernel_with_backend(a_primary), None, STYLE_TEXT_WRAP),
            (status_txt, None, status_style),
            (a_primary.dur_us, None, STYLE_FLOAT3),
            (a_exp_us, None, STYLE_FLOAT3),
            (aft_ms, aft_ms_formula, STYLE_FLOAT3),
            (saved_ms, f"=SUM(J{r_start}:J{r_end})-P{r_start}", STYLE_FLOAT3),
            (speedup, f"=IF(P{r_start}>0,SUM(J{r_start}:J{r_end})/P{r_start},0)", STYLE_RATIO),
            (a_primary.bound_display, None, STYLE_OVERLAP if a_primary.is_overlapped else STYLE_TEXT),
            (lstar_ms, None, STYLE_FLOAT3),
            (gap_ms, f"=MAX(0,P{r_start}-T{r_start})", STYLE_FLOAT3),
            (_sanitize_backend_text(item.action_note), None, STYLE_TEXT_WRAP),
        ), start_col=CMP_COL_AFT_KERN)
        ws.set_cell(r_start, CMP_COL_AFT_UPT, aft_upt, formula=f"=IF({SUM_AFT_NTOK_REF}>0,P{r_start}*1000/{SUM_AFT_NTOK_REF},0)", style=STYLE_FLOAT2)
        ws.set_cell(r_start, CMP_COL_SAVED_UPT, before_upt_sum - aft_upt, formula=f"=SUM(X{r_start}:X{r_end})-Y{r_start}", style=STYLE_FLOAT2)
        ws.set_cell(r_start, CMP_COL_BINDING, format_binding_badge(a_primary.binding_state), style=STYLE_TEXT_WRAP)
    else:
        aft_ms = lstar_ms = aft_upt = 0.0
        write_row(ws, r_start, (
            ("— [优化后移除, 0.000 ms]", None, STYLE_OVERLAP), ("已移除", None, STYLE_OVERLAP),
            (0.0, None, STYLE_FLOAT3), (0.0, None, STYLE_FLOAT3),
            (0.0, "=0", STYLE_FLOAT3), (before_ms_sum, f"=SUM(J{r_start}:J{r_end})-P{r_start}", STYLE_FLOAT3),
            (0.0, f"=IF(P{r_start}>0,SUM(J{r_start}:J{r_end})/P{r_start},0)", STYLE_RATIO),
            ("已移除", None, STYLE_OVERLAP), (0.0, None, STYLE_FLOAT3),
            (0.0, f"=MAX(0,P{r_start}-T{r_start})", STYLE_FLOAT3), (_sanitize_backend_text(item.action_note), None, STYLE_TEXT_WRAP),
        ), start_col=CMP_COL_AFT_KERN)
        ws.set_cell(r_start, CMP_COL_AFT_UPT, 0.0, formula="=0", style=STYLE_FLOAT2)
        ws.set_cell(r_start, CMP_COL_SAVED_UPT, before_upt_sum, formula=f"=SUM(X{r_start}:X{r_end})-Y{r_start}", style=STYLE_FLOAT2)
        ws.set_cell(r_start, CMP_COL_BINDING, "-", style=STYLE_TEXT)

    if r_end > r_start:
        for col in CMP_AFTER_MERGE_COLS:
            ws.merge_range(r_start, col, r_end, col)
    return before_ms_sum, aft_ms, lstar_ms, before_upt_sum, aft_upt


def _write_comparison_subtotal_row(
    ws: Worksheet, r_sub: int, arch_title: str, r_start: int, r_end: int,
    bef_ms: float, aft_ms: float, lstar_ms: float, bef_upt: float, aft_upt: float,
) -> None:
    ws.set_cell(r_sub, 1, f"Subtotal — {arch_title}", style=STYLE_SUBTOTAL_TEXT)
    ws.merge_range(r_sub, 1, r_sub, CMP_SUBTOTAL_LABEL_END_COL)
    ws.set_cell(r_sub, CMP_COL_BEF_EXP_MS, bef_ms, formula=f"=SUM(J{r_start}:J{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_BEF_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
    ws.merge_range(r_sub, CMP_COL_BEF_BOUND, r_sub, CMP_COL_AFT_EXP_US)
    ws.set_cell(r_sub, CMP_COL_AFT_EXP_MS, aft_ms, formula=f"=SUM(P{r_start}:P{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_SAVED_MS, bef_ms - aft_ms, formula=f"=J{r_sub}-P{r_sub}", style=STYLE_SUBTOTAL_NUM)
    spd = (bef_ms / aft_ms) if aft_ms > 0.0 else 0.0
    ws.set_cell(r_sub, CMP_COL_SPEEDUP, spd, formula=f"=IF(P{r_sub}>0,J{r_sub}/P{r_sub},0)", style=STYLE_SUBTOTAL_RATIO)
    ws.set_cell(r_sub, CMP_COL_AFT_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
    ws.set_cell(r_sub, CMP_COL_LSTAR_MS, lstar_ms, formula=f"=SUM(T{r_start}:T{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_GAP_MS, max(0.0, aft_ms - lstar_ms), formula=f"=MAX(0,P{r_sub}-T{r_sub})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_ACTION, "", style=STYLE_SUBTOTAL_TEXT)
    ws.merge_range(r_sub, CMP_COL_ACTION, r_sub, CMP_COL_SOURCE_REF)
    ws.set_cell(r_sub, CMP_COL_BEF_UPT, bef_upt, formula=f"=SUM(X{r_start}:X{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_AFT_UPT, aft_upt, formula=f"=SUM(Y{r_start}:Y{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_SAVED_UPT, bef_upt - aft_upt, formula=f"=X{r_sub}-Y{r_sub}", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_BINDING, "", style=STYLE_SUBTOTAL_TEXT)


def _write_comparison_grand_total(
    ws: Worksheet, tot_row: int, subtotal_rows: Sequence[int],
    tot_bef_ms: float, tot_aft_ms: float, tot_lstar_ms: float,
    tot_bef_upt: float, tot_aft_upt: float,
) -> None:
    ws.set_cell(tot_row, 1, "WHOLE-MODEL TOTAL (实际暴露耗时合计 = Sheet 1 Step Latency)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, 1, tot_row, CMP_SUBTOTAL_LABEL_END_COL)
    sum_j = "=" + "+".join(f"J{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_p = "=" + "+".join(f"P{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_t = "=" + "+".join(f"T{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_x = "=" + "+".join(f"X{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_y = "=" + "+".join(f"Y{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    ws.set_cell(tot_row, CMP_COL_BEF_EXP_MS, tot_bef_ms, formula=sum_j, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_BEF_BOUND, "", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, CMP_COL_BEF_BOUND, tot_row, CMP_COL_AFT_EXP_US)
    ws.set_cell(tot_row, CMP_COL_AFT_EXP_MS, tot_aft_ms, formula=sum_p, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_SAVED_MS, tot_bef_ms - tot_aft_ms, formula=f"=J{tot_row}-P{tot_row}", style=STYLE_TOTAL_FLOAT3)
    spd = (tot_bef_ms / tot_aft_ms) if tot_aft_ms > 0.0 else 0.0
    ws.set_cell(tot_row, CMP_COL_SPEEDUP, spd, formula=f"=IF(P{tot_row}>0,J{tot_row}/P{tot_row},0)", style=STYLE_SUBTOTAL_RATIO)
    ws.set_cell(tot_row, CMP_COL_AFT_BOUND, "", style=STYLE_TOTAL_TEXT)
    ws.set_cell(tot_row, CMP_COL_LSTAR_MS, tot_lstar_ms, formula=sum_t, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_GAP_MS, max(0.0, tot_aft_ms - tot_lstar_ms), formula=f"=MAX(0,P{tot_row}-T{tot_row})", style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_ACTION, "100% 双口径守恒合计（Step_ms 与 us/tok·die 零重复计数）", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, CMP_COL_ACTION, tot_row, CMP_COL_SOURCE_REF)
    ws.set_cell(tot_row, CMP_COL_BEF_UPT, tot_bef_upt, formula=sum_x, style=STYLE_TOTAL_NUM)
    ws.set_cell(tot_row, CMP_COL_AFT_UPT, tot_aft_upt, formula=sum_y, style=STYLE_TOTAL_NUM)
    ws.set_cell(tot_row, CMP_COL_SAVED_UPT, tot_bef_upt - tot_aft_upt, formula=f"=X{tot_row}-Y{tot_row}", style=STYLE_TOTAL_NUM)
    ws.set_cell(tot_row, CMP_COL_BINDING, "= 10^6 / Single-Die TPS", style=STYLE_TOTAL_TEXT)


def populate_comparison_breakdown_sheet(
    ws: Worksheet,
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    aligned_rows: Sequence[AlignedComparisonRow],
    group_accounting: Sequence[GroupAccounting] = (),
) -> BreakdownSheetReceipt:
    """Unplugged socket: accumulate exact int ns across Sheet 2 (strictly 27 cols A:AA)."""
    b_track_lbl = format_track_label(before_bundle.feature_track) if before_bundle.feature_track else "基线配置"
    a_track_lbl = format_track_label(after_bundle.feature_track) if after_bundle.feature_track else "优化方案"
    del ws, aligned_rows, group_accounting, b_track_lbl, a_track_lbl
    raise NotImplementedError(
        "LEGACY_PURGED: float-ms Sheet 2 breakdown accumulation (tot_bef_ms += arch_bef_ms) deleted; "
        "accumulate exact int ns from EngineStepReceipt and divide by 1_000_000 only at cell display."
    )


def populate_standalone_breakdown_sheet(
    ws: Worksheet, bundle: ExportBundle,
) -> BreakdownSheetReceipt:
    """Unplugged socket: standalone 16-column Sheet 2 deleted (must emit 27-col A:AA)."""
    del ws, bundle
    raise NotImplementedError(
        "LEGACY_PURGED: 16-column standalone Sheet 2 (STD_COLS=16) with float-ms accumulation deleted "
        "(Astra P1); emit strict canonical 27-column A:AA Sheet 2 with exact int ns accumulation."
    )


def populate_measured_calibration_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    """Conditional sheet emitted ONLY when ``observations_path`` is supplied."""
    apply_col_widths(ws, MEAS_COL_WIDTHS)
    ws.freeze_panes(2, 2)
    rec = bundle.reconciliation
    src_lbl = rec.source_label if rec is not None else "msprof"
    write_banner(
        ws, 1, MEAS_COLS,
        f"Measured msprof 1-to-N Kernel Calibration ({src_lbl}) | {bundle.model_name} @ {bundle.hardware_sku}",
        STYLE_TITLE,
    )
    headers = (
        "Archetype", "Submodule", "Sim Op ID", "Logical Role", "Ascend Kernel",
        "Rank-Local Shapes & Dtype", "L2 Hit %", "L* Floor (us)", "Sim Pred (us)",
        "Measured Kernel (msprof)", "Kernel Measured (us)", "Measured Sum (us)",
        "APE %", "Bottleneck",
    )
    write_headers(ws, 2, headers)

    row_cursor = MEAS_DATA_START_ROW
    for op in bundle.operators:
        k_count = max(1, len(op.measured_kernels))
        r_start = row_cursor
        r_end = row_cursor + k_count - 1
        write_row(ws, r_start, (
            (op.archetype_title or op.layer_label, None, STYLE_TEXT_WRAP),
            (op.submodule, None, STYLE_TEXT_WRAP), (op.op_id, None, STYLE_TEXT),
            (op.logical_role, None, STYLE_TEXT_WRAP), (op.ascend_kernel, None, STYLE_TEXT_WRAP),
            (op.tensor_contract, None, STYLE_TEXT_WRAP), (op.l2_hit_pct / PCT_DIVISOR, None, STYLE_PCT),
            (op.l_star_lean_us, None, STYLE_FLOAT3), (op.dur_us, None, STYLE_FLOAT3),
        ))
        if op.measured_kernels:
            for offset, k_row in enumerate(op.measured_kernels):
                ws.set_cell(r_start + offset, MEAS_COL_KERN_NAME, k_row.name, style=STYLE_TEXT)
                ws.set_cell(r_start + offset, MEAS_COL_KERN_US, k_row.duration_us, style=STYLE_FLOAT3)
            ws.set_cell(r_start, MEAS_COL_SUM_US, op.measured_sum_us, formula=f"=SUM(K{r_start}:K{r_end})", style=STYLE_FLOAT3)
            ws.set_cell(r_start, MEAS_COL_APE, (op.ape_pct / PCT_DIVISOR) if op.ape_pct is not None else 0.0, formula=f"=IF(L{r_start}>0,ABS(I{r_start}-L{r_start})/L{r_start},0)", style=STYLE_PCT)
        else:
            write_row(ws, r_start, (
                ("(Unmatched)", None, STYLE_TEXT), (None, None, STYLE_FLOAT3),
                (None, None, STYLE_FLOAT3), (None, None, STYLE_PCT),
            ), start_col=MEAS_COL_KERN_NAME)
        ws.set_cell(r_start, MEAS_COL_BOUND, op.bound_display, style=STYLE_TEXT)
        if k_count > 1:
            for col in MEAS_SIM_MERGE_COLS:
                ws.merge_range(r_start, col, r_end, col)
        row_cursor = r_end + 1

    data_end = row_cursor - 1
    tot_row = row_cursor
    ws.set_cell(tot_row, 1, "TOTAL", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, 1, tot_row, MEAS_COL_SIM_US - 2)
    total_vals = (
        sum(op.l_star_lean_us for op in bundle.operators),
        sum(op.dur_us for op in bundle.operators),
        sum(k.duration_us for op in bundle.operators for k in op.measured_kernels),
        sum(op.measured_sum_us or 0.0 for op in bundle.operators),
    )
    for col, val in zip(MEAS_TOTAL_SUM_COLS, total_vals):
        c_let = column_letter(col)
        formula = f"=SUM({c_let}{MEAS_DATA_START_ROW}:{c_let}{data_end})" if data_end >= MEAS_DATA_START_ROW else "=0"
        ws.set_cell(tot_row, col, val, formula=formula, style=STYLE_TOTAL_FLOAT3)
    meas_tot = sum(op.measured_sum_us or 0.0 for op in bundle.operators)
    sim_tot = sum(op.dur_us for op in bundle.operators if op.measured_sum_us is not None)
    agg_ape = (abs(sim_tot - meas_tot) / meas_tot) if meas_tot > 0.0 else 0.0
    ws.set_cell(tot_row, MEAS_COL_APE, agg_ape, formula=f"=IF(L{tot_row}>0,ABS(I{tot_row}-L{tot_row})/L{tot_row},0)", style=STYLE_TOTAL_PCT)
    ws.validate_max_column(MEAS_COLS)


__all__ = [
    "CMP_COLS",
    "CMP_COL_AFT_EXP_MS",
    "CMP_COL_AFT_UPT",
    "CMP_COL_BEF_EXP_MS",
    "CMP_COL_BEF_UPT",
    "CMP_COL_SAVED_MS",
    "CMP_COL_SAVED_UPT",
    "CMP_DATA_START_ROW",
    "MEAS_COLS",
    "STD_COLS",
    "STD_COL_EXP_MS",
    "STD_COL_LSTAR_MS",
    "BreakdownSheetReceipt",
    "populate_comparison_breakdown_sheet",
    "populate_measured_calibration_sheet",
    "populate_standalone_breakdown_sheet",
]
