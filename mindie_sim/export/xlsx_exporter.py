"""Human-native Ascend optimization OOXML ``.xlsx`` workbook renderer.

[Role] Assemble executive-ready 4-tab (plus conditional 5th calibration tab)
  ECMA-376 ``.xlsx`` workbooks on top of ``ooxml.Workbook``:
  1. ``Summary`` — 4 Clear Executive Blocks:
     (1) Deployment, Parallelism, Concurrency & Side-by-Side Quantized vs Native
         BF16 Weight Footprint,
     (2) Before vs After Simulation KPIs with live Excel formulas for Step
         Latency, Effective TPOT + SLA Verdict, Single-Die / Cluster TPS,
         Lean 4 ``L*`` Attainment %, and Hardware Utilization (AIC MFU / AIV
         Util / HBM MBU / HCCL Link BW Util — zero ``"4D"`` jargon),
     (3) Two-Level L1/L2 Module Breakdown with explicit ``流内原始 / 被掩盖 /
         实际暴露`` columns achieving 100% Step Latency conservation with zero
         negative mystery rows,
     (4) 3-Track Actionable Implementation Roadmap (``Config & Env``,
         ``Framework``, ``Kernel``) + ``QuaRot``-first & Parallelism Co-Design;
  2. ``Layer_Operator_Breakdown`` — Strict topological order, contiguous
     ``Submodule`` merges, up-front Col E ``Stream & Masking`` badges, paired
     raw/exposed durations, 3-track actions, and trailing far-right muted
     ``Source Ref``;
  3. ``Memory_Quant_Calculator`` — Live formula-driven ``Quant Rule`` +
     ``Quant Format`` (honoring mixed ``W4A8`` INT8 linear + INT4 expert and
     ``MXFP4``/``QuaRot`` toggles), side-by-side ``原生 BF16`` vs ``量化后``,
     ``Include Vision Tower (ViT)`` toggle, and ``SUMIFS`` HBM vs Host DRAM
     subtotals;
  4. ``Pending_Kernel_Details`` — 100%-conserved Top Index + Structured 3-Track
     Per-Kernel Detail Cards (``Config & Env`` / ``Kernel`` / ``Framework`` /
     ``Acceptance``) wired to ``--kernel-designs-json`` by ``card_id``;
  5. Optional ``Measured_msprof_Calibration`` when ``observations_path`` is set.
[Invariants] Zero third-party OOXML deps, zero heuristic fudge factors, zero
  banned Python recipe class names, zero ``[待 Agent 补充]`` blanks, zero
  ``"4D"`` jargon, strict ECMA-376 element order, and non-overlapping merges.
[Timeouts & Locks] Bounded O(Archetypes) in-memory XML serialization; no locks
  or device contexts.
[Side-Effect Policy] Writes only the caller-requested ``.xlsx`` (and optional
  ``agent_brief.json``) output path via atomic same-directory ``os.replace``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindie_sim.prediction.results import CaseInput, ReplayOptions, StepEvaluation, StepRequest

from .bundle import build_export_bundle, compose_export_case
from .comparison import (
    align_before_after_operators,
    build_actionable_roadmap,
    build_agent_brief,
    build_macro_module_breakdown,
    build_pending_kernel_details,
    export_agent_brief,
    format_parallelism_codomain_summary,
)
from .models import (
    BF16_BYTES,
    BINDING_STATE_BOUND,
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    FP32_BYTES,
    GIB_BYTES,
    KV_QUANT_GROUP_SIZE,
    MACRO_L1_HIERARCHY,
    MIB_BYTES,
    MIN_DENOM_EPS,
    MODE_MODELING,
    OVERLAP_EXPOSED_EPS_US,
    PENDING_KIND_TO_IMPLEMENT,
    QUANT_RULE_CHECKPOINT_IGNORE,
    QUANT_RULE_FIXED_BF16,
    QUANT_RULE_FIXED_FP32,
    QUANT_RULE_PRESET_LINEAR_FP8,
    SUPPORTED_CACHE_DTYPES,
    SUPPORTED_WEIGHT_QUANTS,
    TRACK_CONFIG_ENV,
    TRACK_FRAMEWORK,
    US_PER_MS,
    WEIGHT_QUANT_FORMATS,
    ActionRoadmapRow,
    AlignedComparisonRow,
    ExportBundle,
    MacroModuleRow,
    MemoryCalculatorLedger,
    OperatorExportRow,
    PendingKernelItem,
    format_binding_badge,
)
from .ooxml import (
    BACKLOG_ROW_HEIGHT_PT,
    CARD_ACCEPT_ROW_HEIGHT_PT,
    CARD_DESIGN_ROW_HEIGHT_PT,
    HEADER_ROW_HEIGHT_PT,
    NS_MAIN as NS_MAIN,
    STYLE_ALERT_HIGH,
    STYLE_ALERT_MID,
    STYLE_ALERT_OK,
    STYLE_CARD_HDR,
    STYLE_FLOAT2,
    STYLE_FLOAT3,
    STYLE_HEADER,
    STYLE_INPUT_NUM,
    STYLE_INPUT_TEXT,
    STYLE_INPUT_TEXT_WRAP,
    STYLE_INT,
    STYLE_KPI_HIGHLIGHT,
    STYLE_L1_NUM,
    STYLE_L1_PCT,
    STYLE_L1_TEXT,
    STYLE_MASKED_NUM,
    STYLE_MERGED,
    STYLE_MUTED_TEXT,
    STYLE_NEW_KERNEL,
    STYLE_OPT_BADGE,
    STYLE_OVERLAP,
    STYLE_PARTIAL_OVERLAP,
    STYLE_PCT,
    STYLE_RATIO,
    STYLE_SECTION,
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
    Workbook,
    Worksheet,
    column_letter,
)

#: Schema tag for XLSX export receipts.
XLSX_EXPORT_RECEIPT_SCHEMA = "mindie-sim.xlsx-export-receipt/3"
#: Gap-ratio visual severity thresholds.
GAP_RATIO_SEVERE = 3.0
GAP_RATIO_MODERATE = 1.5
#: Cache bytes/elem approximations matching closed-form Excel formulas.
C8_CACHE_FACTOR = 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE)
#: Percentage divisor and MiB-per-GiB divisor for live Excel formulas.
PCT_DIVISOR = 100.0
MIB_PER_GIB = 1024
#: Threshold (ms) to highlight masked credit in soft green.
MASKED_HIGHLIGHT_EPS_MS = 0.0005

#: Canonical sheet names of the human-native Ascend optimization workbook.
SHEET_SUMMARY = "Summary"
SHEET_BREAKDOWN = "Layer_Operator_Breakdown"
SHEET_MEMORY = "Memory_Quant_Calculator"
SHEET_PENDING_KERNELS = "Pending_Kernel_Details"
#: Backward-compatible alias for Sheet 4.
SHEET_NEW_KERNELS = SHEET_PENDING_KERNELS
SHEET_MEASURED = "Measured_msprof_Calibration"

#: Sheet 1 (Summary) 4-block structural layout constants (13 columns A..M).
SUMMARY_COLS = 13
SUMMARY_FREEZE_ROWS = 1
SUM_B1_BANNER_ROW = 2
SUM_B1_HDR_ROW = 3
SUM_ROW_MODEL = 4
SUM_ROW_HW = 5
SUM_ROW_MODE = 6
SUM_ROW_LEN = 7
SUM_ROW_SLA = 8
SUM_ROW_SPEC = 9
SUM_ROW_BUDGET = 10
SUM_LEFT_DETAIL_START = 3
SUM_LEFT_DETAIL_END = 4
SUM_RIGHT_LBL_START = 5
SUM_RIGHT_LBL_END = 6
SUM_RIGHT_VAL_COL = 7
SUM_RIGHT_DETAIL_START = 8
SUM_B2_BANNER_ROW = 12
SUM_B2_HDR_ROW = 13
SUM_KPI_STEP_ROW = 14
SUM_KPI_TPOT_ROW = 15
SUM_KPI_SLA_ROW = 16
SUM_KPI_DIE_TPS_ROW = 17
SUM_KPI_CLUSTER_TPS_ROW = 18
SUM_KPI_LSTAR_ROW = 19
SUM_KPI_ATTAIN_ROW = 20
SUM_KPI_UTIL_START_ROW = 21
SUM_KPI_AMORTIZED_ROW = 25
SUM_KPI_NTOK_ROW = 26
SUM_KPI_WALL_ROW = 27
SUM_KPI_NOTE_START_COL = 8
SUM_B3_BANNER_ROW = 29
SUM_B3_HDR_ROW = 30
SUM_B3_DATA_START_ROW = 31
SUM_BEF_NTOK_REF = f"{SHEET_SUMMARY}!$B${SUM_KPI_NTOK_ROW}"
SUM_AFT_NTOK_REF = f"{SHEET_SUMMARY}!$C${SUM_KPI_NTOK_ROW}"
SUM_B4_TRACK_END = 2
SUM_B4_TITLE_START = 3
SUM_B4_TITLE_END = 4
SUM_B4_GAIN_COL = 5
SUM_B4_DETAIL_START = 6
SUM_B4_DETAIL_END = 11
SUM_B4_REF_START = 12

#: Sheet 2 Before-vs-After mode column indices (27 columns A..AA).
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

#: Sheet 3 (Memory_Quant_Calculator) structural layout constants (24 columns A..X).
MEMORY_COLS = 24
MEMORY_TABLE_END_COL = 16
MEMORY_WEIGHT_MERGE_END_COL = 10
MEMORY_WEIGHT_SUM_START_COL = 11
MEMORY_CACHE_BASIS_START_COL = 7
MEMORY_CACHE_MERGE_END_COL = 4
MEMORY_CACHE_REQ_COL = 5
MEMORY_CACHE_BATCH_COL = 6
MEMORY_STAGE_HEADROOM_COL = 12
MEMORY_STAGE_STATUS_COL = 13
MEMORY_STAGE_RAW_ENGRAM_COL = 14
MEMORY_STAGE_AGG_START_COL = 3
MEMORY_CONTROL_COL_LABEL = 19
MEMORY_CONTROL_COL_VALUE = 20
MEMORY_LOOKUP_START_COL = 21
MEMORY_LOOKUP_HEADER_ROW = 3
MEMORY_LOOKUP_FIRST_ROW = 4
MEMORY_LOOKUP_LAST_ROW = 10
MEMORY_TABLE_HEADER_ROW = 3
MEMORY_FREEZE_ROWS = 3
MEMORY_QUANT_LOOKUP_ORDER = ("BF16", "FP32", "FP8", "W8A8", "W4A8", "MXFP8", "MXFP4")
#: Control Panel absolute cell references in column T (20).
MEM_CTL_QUANT_REF = "$T$3"
MEM_CTL_CACHE_REF = "$T$4"
MEM_CTL_OFFLOAD_REF = "$T$5"
MEM_CTL_OVERRIDE_IGN_REF = "$T$6"
MEM_CTL_VIT_REF = "$T$7"
MEM_CTL_CONC_REF = "$T$18"
MEM_CTL_LOCAL_TOK_REF = "$T$21"
MEM_CTL_USABLE_HBM_REF = "$T$24"

#: Sheet 4 (Pending_Kernel_Details) structural layout constants (14 columns A..N).
PENDING_COLS = 14
PENDING_HEADER_ROW = 3
PENDING_DATA_START_ROW = 4
PENDING_COL_GAIN = 7
PENDING_COL_GAIN_UPT = 11
PENDING_COL_BINDING = 12
PENDING_COL_BEF_MS = 13
PENDING_COL_AFT_MS = 14
PENDING_LABEL_MERGE_END = 6
PENDING_CARD_SPACER_GAP = 4
PENDING_CARD_STRIDE = 11
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
PENDING_CARD_ROW_CONFIG = 6
PENDING_CARD_ROW_KERNEL = 7
PENDING_CARD_ROW_FRAMEWORK = 8
PENDING_CARD_ROW_ACCEPT = 9

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

#: Preset column widths across sheets.
SUMMARY_COL_WIDTHS = (
    34.0, 14.0, 14.0, 14.0, 14.0, 13.0, 14.0, 14.0, 14.0, 13.0, 14.0, 13.0, 34.0,
)
CMP_COL_WIDTHS = (
    30.0, 9.0, 22.0, 28.0, 34.0, 32.0, 26.0, 13.0, 13.0, 14.0, 20.0,
    26.0, 28.0, 13.0, 13.0, 14.0, 13.0, 10.0, 20.0, 13.0, 12.0, 38.0, 30.0,
    15.0, 15.0, 15.0, 26.0,
)
STD_COL_WIDTHS = (
    30.0, 9.0, 22.0, 28.0, 34.0, 32.0, 26.0, 28.0, 13.0, 13.0,
    14.0, 13.0, 13.0, 12.0, 20.0, 30.0,
)
MEMORY_COL_WIDTHS = (
    16.0, 24.0, 11.0, 11.0, 16.0, 11.0, 11.0, 11.0, 19.0, 13.0, 14.0, 13.0,
    12.0, 15.0, 16.0, 28.0, 3.5, 3.5, 28.0, 26.0, 11.0, 10.0, 10.0, 11.0,
)
PENDING_COL_WIDTHS = (9.0, 24.0, 30.0, 26.0, 28.0, 24.0, 15.0, 22.0, 24.0, 22.0, 16.0, 26.0, 14.0, 14.0)
MEAS_COL_WIDTHS = (
    28.0, 22.0, 24.0, 24.0, 24.0, 28.0, 12.0, 14.0, 14.0, 28.0, 14.0, 14.0, 12.0, 20.0,
)


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


def _apply_col_widths(ws: Worksheet, widths: Sequence[float]) -> None:
    for idx, width in enumerate(widths, start=1):
        ws.set_col_width(idx, width)


def _write_banner(ws: Worksheet, row: int, col_end: int, text: str, style: int) -> None:
    ws.set_cell(row, 1, text, style=style)
    ws.merge_range(row, 1, row, col_end)


def _gap_ratio_style(ratio: float | None) -> int:
    if ratio is None:
        return STYLE_RATIO
    if ratio >= GAP_RATIO_SEVERE:
        return STYLE_ALERT_HIGH
    if ratio >= GAP_RATIO_MODERATE:
        return STYLE_ALERT_MID
    return STYLE_ALERT_OK


def _write_row(
    ws: Worksheet,
    row: int,
    entries: Sequence[tuple[Any, str | None, int]],
    *,
    start_col: int = 1,
) -> None:
    for offset, (val, formula, style) in enumerate(entries):
        ws.set_cell(row, start_col + offset, val, formula=formula, style=style)


def _write_headers(
    ws: Worksheet, row: int, labels: Sequence[str], *, start_col: int = 1,
) -> None:
    ws.set_row_height(row, HEADER_ROW_HEIGHT_PT)
    for offset, label in enumerate(labels):
        ws.set_cell(row, start_col + offset, label, style=STYLE_HEADER)


def _extend_span(spans: list[tuple[Any, int, int]], key: Any, r1: int, r2: int) -> None:
    if spans and spans[-1][0] == key:
        spans[-1] = (key, spans[-1][1], r2)
    else:
        spans.append((key, r1, r2))


def _stream_masking_badge(op: OperatorExportRow) -> tuple[str, int]:
    """Build up-front intuitive Stream & Masking label + badge style for Sheet 2 Col E."""
    if op.dur_us <= 0.0:
        return "主计算流 [0 us]", STYLE_TEXT
    if op.is_overlapped:
        masker = op.masked_by or ("MoE Routed GMM / MegaMoE" if op.stream_index == 1 else "主计算流")
        stream_tag = "副流并发 (MSO)" if op.stream_index == 1 else ("通信流并发" if op.stream_index == 2 else "并行分支")
        return f"{stream_tag} [被 {masker} 100% 掩盖 -> 暴露 0 us，不计入总耗时]", STYLE_OVERLAP
    masked_us = max(0.0, op.dur_us - op.exposed_dur_us)
    if masked_us > OVERLAP_EXPOSED_EPS_US:
        pct = 100.0 * masked_us / max(op.dur_us, OVERLAP_EXPOSED_EPS_US)
        masker = op.masked_by or "主计算流"
        stream_tag = "副流并发 (MSO)" if op.stream_index == 1 else ("通信流并发" if op.stream_index == 2 else "并行分支")
        return f"{stream_tag} [被 {masker} 掩盖 {pct:.0f}% -> 仅暴露 {op.exposed_dur_us:.2f} us]", STYLE_PARTIAL_OVERLAP
    if op.stream_index == 2 or op.task_type == "HCCL":
        return "通信流 [暴露 100%]", STYLE_TEXT
    if op.stream_index == 1:
        return "副流 (MSO) [暴露 100%]", STYLE_TEXT
    return "主计算流 [暴露 100%]", STYLE_TEXT


def _kernel_with_backend(op: OperatorExportRow) -> str:
    kern = op.ascend_kernel or op.implementation
    return f"{kern} [{op.kernel_backend}]" if op.kernel_backend else kern


def _lookup_agent_design_bundle(
    designs: Mapping[str, Any] | None,
    *,
    card_id: str = "",
    kernel_key: str = "",
    target_kernel: str = "",
    logical_role: str = "",
) -> dict[str, str | None]:
    """Resolve structured 4-slot overrides (config/kernel/framework/acceptance) by card_id or key."""
    empty: dict[str, str | None] = {
        "config_howto": None, "kernel_design": None, "framework_work": None, "acceptance": None,
    }
    if not designs:
        return empty
    entry: Any = None
    for exact in (card_id, kernel_key, target_kernel):
        if exact and exact in designs:
            entry = designs[exact]
            break
    if entry is None:
        haystacks = tuple(h.lower() for h in (card_id, kernel_key, target_kernel, logical_role) if h)
        for k, val in designs.items():
            kl = str(k).lower().strip()
            if kl and any(kl == h or kl in h or (len(h) >= 6 and h in kl) for h in haystacks):
                entry = val
                break
    if isinstance(entry, str):
        clean = entry.strip()
        return {**empty, "kernel_design": clean or None}
    if not isinstance(entry, Mapping):
        return empty
    return {
        "config_howto": str(entry.get("config_howto") or entry.get("config") or "").strip() or None,
        "kernel_design": str(entry.get("kernel_design") or entry.get("design") or entry.get("proposal") or "").strip() or None,
        "framework_work": str(entry.get("framework_work") or entry.get("framework") or "").strip() or None,
        "acceptance": str(entry.get("acceptance") or entry.get("criteria") or "").strip() or None,
    }


def _populate_summary_deployment_block(
    ws: Worksheet, bundle: ExportBundle, compare_bundle: ExportBundle | None,
) -> None:
    """Block 1: Deployment, Parallelism, Concurrency & Side-by-Side Weight Footprint."""
    ref = compare_bundle or bundle
    mem = ref.memory_ledger
    _write_banner(ws, SUM_B1_BANNER_ROW, SUMMARY_COLS, "1. 部署方案、并行策略与并发总览 (Deployment, Parallelism & Concurrency)", STYLE_SECTION)
    ws.set_row_height(SUM_B1_HDR_ROW, HEADER_ROW_HEIGHT_PT)
    ws.set_cell(SUM_B1_HDR_ROW, 1, "部署与业务配置 (Deployment & Spec)", style=STYLE_HEADER)
    ws.set_cell(SUM_B1_HDR_ROW, 2, "配置值 (Value)", style=STYLE_HEADER)
    ws.set_cell(SUM_B1_HDR_ROW, SUM_LEFT_DETAIL_START, "规格明细 (Detail)", style=STYLE_HEADER)
    ws.merge_range(SUM_B1_HDR_ROW, SUM_LEFT_DETAIL_START, SUM_B1_HDR_ROW, SUM_LEFT_DETAIL_END)
    ws.set_cell(SUM_B1_HDR_ROW, SUM_RIGHT_LBL_START, "并行策略与并发/显存 (Parallelism & Concurrency)", style=STYLE_HEADER)
    ws.merge_range(SUM_B1_HDR_ROW, SUM_RIGHT_LBL_START, SUM_B1_HDR_ROW, SUM_RIGHT_LBL_END)
    ws.set_cell(SUM_B1_HDR_ROW, SUM_RIGHT_VAL_COL, "数值 (Value)", style=STYLE_HEADER)
    ws.set_cell(SUM_B1_HDR_ROW, SUM_RIGHT_DETAIL_START, "切分 / 量化显存明细 (Detail)", style=STYLE_HEADER)
    ws.merge_range(SUM_B1_HDR_ROW, SUM_RIGHT_DETAIL_START, SUM_B1_HDR_ROW, SUMMARY_COLS)

    layers_txt = (
        f"Target {ref.total_target_layers}L + Draft {ref.total_draft_layers}L"
        if ref.total_draft_layers > 0 else f"Target {ref.total_target_layers}L"
    )
    hbm_gb = int(round(mem.die_hbm_capacity_bytes / float(GIB_BYTES)))
    topo_txt = f"{ref.hardware_sku} {hbm_gb}G / {ref.topology_preset}" if ref.topology_preset else f"{ref.hardware_sku} {hbm_gb}G"
    sp_txt = "ON" if ref.sequence_parallel else "OFF"
    par_short, par_detail = format_parallelism_codomain_summary(ref)
    if compare_bundle is not None:
        b_mem = bundle.memory_ledger
        if (b_mem.dp, b_mem.tp, b_mem.ep, b_mem.dcp, b_mem.concurrency) != (
            mem.dp, mem.tp, mem.ep, mem.dcp, mem.concurrency
        ):
            par_detail = (
                f"{par_detail} | Iso-SLA: Before(DP{b_mem.dp}·TP{b_mem.tp}·EP{b_mem.ep}, B={b_mem.concurrency}) "
                f"-> After(DP{mem.dp}·TP{mem.tp}·EP{mem.ep}, B={mem.concurrency})"
            )
    mode_tag = "建模 Modeling" if ref.analysis_mode == MODE_MODELING else "仿真 Simulation"
    bind_badge = format_binding_badge(ref.binding_state)
    track_txt = (
        f"Before({bundle.feature_track or 'base'}) -> After({compare_bundle.feature_track or 'cand'})"
        if compare_bundle is not None else (bundle.feature_track or "standalone")
    )
    is_decode = ref.phase == "decode"
    sla_target = ref.tpot_slo_ms if is_decode else ref.ttft_slo_ms
    sla_detail = f"TPOT <= {ref.tpot_slo_ms:.1f} ms (TTFT <= {ref.ttft_slo_ms:.0f} ms)" if is_decode else f"TTFT <= {ref.ttft_slo_ms:.0f} ms"
    spec_detail = (
        f"Method={ref.spec_method}, Proposals={ref.spec_proposals}, Avg Accepted={ref.avg_accepted_tokens:.2f} tok/step"
        if is_decode and ref.spec_proposals > 0
        else (f"Standard Decode (Avg Accepted={ref.avg_accepted_tokens:.1f} tok/step)" if is_decode else f"Prefill Tokens/Seq={int(ref.avg_accepted_tokens)}")
    )
    step_budget = sla_target * ref.avg_accepted_tokens if is_decode else sla_target
    budget_formula = f"=B{SUM_ROW_SLA}*B{SUM_ROW_SPEC}" if is_decode else f"=B{SUM_ROW_SLA}"
    budget_detail = f"= TPOT SLA (B{SUM_ROW_SLA}) × Avg Accepted (B{SUM_ROW_SPEC})" if is_decode else f"= TTFT SLA (B{SUM_ROW_SLA})"
    eff_cluster_conc = (
        int(ref.cluster_concurrency)
        if int(getattr(ref, "cluster_concurrency", 0) or 0) > 0
        else (mem.concurrency * mem.dp)
    )
    cluster_conc_formula: str | None = (
        f"=G{SUM_ROW_MODE}*{mem.dp}" if eff_cluster_conc == mem.concurrency * mem.dp else None
    )
    pd_pool = ref.pd_pool_summary if isinstance(ref.pd_pool_summary, Mapping) else None
    dies_detail = f"DP={mem.dp} × TP={mem.tp} × PP={mem.pp} (PCP={mem.pcp})"
    if pd_pool:
        p_inst = int(pd_pool.get("prefill_instances", 0) or 0)
        d_inst = int(pd_pool.get("decode_instances", 0) or 0)
        used_d = int(pd_pool.get("used_dies", ref.deployment_total_dies or mem.total_dies) or mem.total_dies)
        dies_detail = f"{dies_detail} | PD Cluster={used_d} dies ({p_inst}P + {d_inst}D)"
    weight_gib = mem.weight_footprint_per_die_bytes / float(GIB_BYTES)
    native_gib = mem.native_bf16_weight_footprint_per_die_bytes / float(GIB_BYTES)
    offload_note = (
        f" | Host DRAM Offload={mem.max_stage_host_offload_bytes / float(GIB_BYTES):.2f} GiB (不计入 NPU HBM)"
        if mem.max_stage_host_offload_bytes > 0.0 else ""
    )
    vit_note = (
        f" | ViT={mem.vision_weight_bytes_per_die / float(GIB_BYTES):.2f} GiB ({'已计入' if mem.include_vision_tower else '纯文本默认排除'})"
        if mem.vision_weight_bytes_per_die > 0.0 else ""
    )
    weight_detail = (
        f"GiB/Die ({mem.weight_quant_display or mem.weight_quant}) "
        f"[原生 BF16: {native_gib:.2f} GiB/die, 压缩 {mem.weight_compression_ratio:.2f}x | 优先选 QuaRot 量化权重]"
        f"{offload_note}{vit_note}"
    )
    rows: tuple[tuple[int, str, Any, str | None, int, str, str, Any, str | None, int, str], ...] = (
        (
            SUM_ROW_MODEL, "模型与总层数 (Model & Total Layers)", ref.model_name, None, STYLE_MERGED, layers_txt,
            "总卡数 (Total Dies)", mem.total_dies, None, STYLE_INT, dies_detail,
        ),
        (
            SUM_ROW_HW, "硬件 SKU & 超节点拓扑 (Hardware / Topo)", ref.hardware_sku, None, STYLE_TEXT, f"{topo_txt} | {ref.hardware_die_summary}",
            "并行切分策略 (Full Parallelism)", par_short, None, STYLE_MERGED, par_detail,
        ),
        (
            SUM_ROW_MODE, "部署形态 (Deployment Mode)", ref.deployment_mode or "PD 分离 [非池化]", None, STYLE_MERGED,
            f"Phase={ref.phase.upper()} | Capture={ref.graph_capture_mode} | {mode_tag} {bind_badge} | {track_txt}",
            "单 DP 活跃并发 (Active Concurrency / DP)", mem.concurrency, None, STYLE_INPUT_NUM, f"Active requests per DP replica (DP={mem.dp})",
        ),
        (
            SUM_ROW_LEN, "业务输入/输出长度 (Prompt / Output)", f"{ref.prompt_tokens} / {ref.output_tokens}", None, STYLE_TEXT, f"Context={ref.context_tokens} tok, Output={ref.output_tokens} tok, SeqLen={mem.seq_len}",
            "集群总并发 (Cluster Total Concurrency)", eff_cluster_conc, cluster_conc_formula, STYLE_KPI_HIGHLIGHT, f"= Active Concurrency/DP (G{SUM_ROW_MODE}) × DP ({mem.dp})",
        ),
        (
            SUM_ROW_SLA, "SLA 约束 (SLA Target ms)", sla_target, None, STYLE_FLOAT2, sla_detail,
            "单卡 HBM 权重占用 (NPU Weight / Die GiB)", weight_gib, None, STYLE_FLOAT2, weight_detail,
        ),
        (
            SUM_ROW_SPEC, "投机解码策略 (Avg Accepted tok/step)", ref.avg_accepted_tokens, None, STYLE_FLOAT2, spec_detail,
            "单卡最大可承载并发 (Max HBM Concurrency / DP)", mem.max_hbm_concurrency_per_dp, None, STYLE_INT,
            f"Usable HBM={mem.usable_hbm_bytes / float(GIB_BYTES):.1f} GiB | Fit={'PASS' if mem.fits_hbm else 'OOM'} | Cache={mem.cache_dtype}",
        ),
        (
            SUM_ROW_BUDGET, "单步耗时预算 (Step Budget ms)", step_budget, budget_formula, STYLE_FLOAT2, budget_detail,
            "EP / DCP / Local-TP / SP 扩展维度", f"EP={mem.ep} / DCP={mem.dcp}", None, STYLE_TEXT,
            f"PP={mem.pp}, PCP={mem.pcp}, Local-TP={mem.local_tp}, Shared-TP={mem.shared_tp}, SP={sp_txt}",
        ),
    )
    for r, l_lbl, l_val, l_f, l_st, l_det, r_lbl, r_val, r_f, r_st, r_det in rows:
        ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
        ws.set_cell(r, 1, l_lbl, style=STYLE_TEXT)
        ws.set_cell(r, 2, l_val, formula=l_f, style=l_st)
        ws.set_cell(r, SUM_LEFT_DETAIL_START, l_det, style=STYLE_TEXT_WRAP)
        ws.merge_range(r, SUM_LEFT_DETAIL_START, r, SUM_LEFT_DETAIL_END)
        ws.set_cell(r, SUM_RIGHT_LBL_START, r_lbl, style=STYLE_TEXT)
        ws.merge_range(r, SUM_RIGHT_LBL_START, r, SUM_RIGHT_LBL_END)
        ws.set_cell(r, SUM_RIGHT_VAL_COL, r_val, formula=r_f, style=r_st)
        ws.set_cell(r, SUM_RIGHT_DETAIL_START, r_det, style=STYLE_TEXT_WRAP)
        ws.merge_range(r, SUM_RIGHT_DETAIL_START, r, SUMMARY_COLS)


def _pd_cluster_tps_cells(bundle: ExportBundle, col: str) -> tuple[float, str | None, str] | None:
    """Return PD ``(cluster_tps, cluster_formula, die_tps_formula)`` or ``None`` when not PD.

    Cluster TPS is the solver's winning ``Output × min(prefill∧KV rps, decode rps)``; the live
    formula is only emitted when it reproduces that cached value. Single-Die TPS stays the decode
    instance's own ``N_tok_per_die × 1000 / Step_ms`` (Sheets 2/4 amortization source).
    """
    pool = bundle.pd_pool_summary if isinstance(bundle.pd_pool_summary, Mapping) else None
    if not pool or pool.get("e2e_total_tps") is None:
        return None
    e2e = float(pool["e2e_total_tps"])
    r_step, r_dtps = SUM_KPI_STEP_ROW, SUM_KPI_DIE_TPS_ROW
    die_f = f"=IF({col}{r_step}>0,{col}{SUM_KPI_NTOK_ROW}*1000/{col}{r_step},0)"
    if pool.get("prefill_kv_rps") is None:
        return e2e, None, die_f
    cap = float(pool["prefill_kv_rps"])
    output = max(1, int(bundle.output_tokens))
    if output > 1:
        d_inst = max(1, int(pool.get("decode_instances", 1) or 1))
        dies = max(1, int(bundle.memory_ledger.total_dies))
        decode_rps = d_inst * dies * float(bundle.throughput_per_die_tokens_per_s) / (output - 1)
        seed = output * min(cap, decode_rps)
        formula = f"={output}*MIN({cap!r},{d_inst}*{dies}*{col}{r_dtps}/{output - 1})"
    else:
        seed, formula = cap, f"={cap!r}"
    parity = abs(seed - e2e) <= 1e-6 * max(1.0, abs(e2e))
    return e2e, (formula if parity else None), die_f


def _populate_summary_kpi_block(
    ws: Worksheet, bundle: ExportBundle, compare_bundle: ExportBundle | None,
) -> None:
    """Block 2: Before vs After KPIs & Hardware Utilization (zero '4D' jargon)."""
    after = compare_bundle or bundle
    is_cmp = compare_bundle is not None
    is_decode = bundle.phase == "decode"
    is_modeling = after.analysis_mode == MODE_MODELING or after.has_unbound_formula_ops
    title = (
        "2. 优化前后端到端性能对比 (Before [当前栈仿真基线] vs After [卡SLA目标建模终态] — 双口径守恒)"
        if (is_cmp and is_modeling) else (
            "2. 优化前后端到端仿真结果对比 (Before vs After Simulation KPIs — Highlighting TPS & Effective TPOT)"
            if is_cmp else "2. 端到端仿真核心结果总览 (End-to-End Simulation KPIs — Highlighting TPS & Effective TPOT)"
        )
    )
    _write_banner(ws, SUM_B2_BANNER_ROW, SUMMARY_COLS, title, STYLE_SECTION)
    headers = (
        "端到端关键指标 (KPI Metric)",
        "Before [当前栈仿真基线]" if is_cmp else "当前仿真 (Sim Pred)",
        ("After [卡SLA目标建模终态]" if is_modeling else "After [仿真对比]") if is_cmp else "基线对照 (Current)",
        "收益变化 (Delta)", "变化幅度 (Delta %)", "加速比 (Speedup)", "单位 (Unit)",
        "SLA 判定 / 物理含义说明 (Verdict & Physical Context)",
    )
    _write_headers(ws, SUM_B2_HDR_ROW, headers)
    ws.merge_range(SUM_B2_HDR_ROW, SUM_KPI_NOTE_START_COL, SUM_B2_HDR_ROW, SUMMARY_COLS)

    b_step = bundle.step_span_us / US_PER_MS
    a_step = after.step_span_us / US_PER_MS
    b_acc = max(1.0, bundle.avg_accepted_tokens)
    acc = max(1.0, after.avg_accepted_tokens)
    b_handoff = float(getattr(bundle, "handoff_tpot_ms", 0.0) or 0.0) if is_decode else 0.0
    a_handoff = float(getattr(after, "handoff_tpot_ms", 0.0) or 0.0) if is_decode else 0.0
    b_tpot = ((b_step / b_acc) + b_handoff) if is_decode else b_step
    a_tpot = ((a_step / acc) + a_handoff) if is_decode else a_step
    sla = after.tpot_slo_ms if is_decode else after.ttft_slo_ms
    b_pass = b_tpot <= sla
    a_pass = a_tpot <= sla
    b_ctps = bundle.throughput_tokens_per_s
    a_ctps = after.throughput_tokens_per_s
    b_dtps = bundle.throughput_per_die_tokens_per_s
    a_dtps = after.throughput_per_die_tokens_per_s
    b_lstar = bundle.step_l_star_lean_us / US_PER_MS
    a_lstar = after.step_l_star_lean_us / US_PER_MS
    b_att = (b_lstar / b_step) if b_step > 0.0 else 0.0
    a_att = (a_lstar / a_step) if a_step > 0.0 else 0.0
    a_gap_ratio = (a_step / a_lstar) if a_lstar > 0.0 else 0.0
    b_cluster_conc = (
        int(bundle.cluster_concurrency)
        if int(getattr(bundle, "cluster_concurrency", 0) or 0) > 0
        else max(1, bundle.memory_ledger.concurrency * bundle.memory_ledger.dp)
    )
    a_cluster_conc = (
        int(after.cluster_concurrency)
        if int(getattr(after, "cluster_concurrency", 0) or 0) > 0
        else max(1, after.memory_ledger.concurrency * after.memory_ledger.dp)
    )
    same_deploy = (
        bundle.memory_ledger.total_dies == after.memory_ledger.total_dies
        and b_cluster_conc == a_cluster_conc
        and abs(b_acc - acc) <= MIN_DENOM_EPS
    )
    b_dies = max(1, bundle.memory_ledger.total_dies)

    r_step, r_tpot, r_sla = SUM_KPI_STEP_ROW, SUM_KPI_TPOT_ROW, SUM_KPI_SLA_ROW
    r_dtps, r_ctps, r_ls, r_att = SUM_KPI_DIE_TPS_ROW, SUM_KPI_CLUSTER_TPS_ROW, SUM_KPI_LSTAR_ROW, SUM_KPI_ATTAIN_ROW
    b_hf_suffix = f"+{b_handoff!r}" if b_handoff > 0.0 else ""
    a_hf_suffix = f"+{a_handoff!r}" if a_handoff > 0.0 else ""
    tpot_b_f = (
        f"=IF($B${SUM_ROW_SPEC}>0,B{r_step}/$B${SUM_ROW_SPEC},B{r_step}){b_hf_suffix}"
        if (is_decode and same_deploy) else (f"=B{r_step}/{b_acc!r}{b_hf_suffix}" if is_decode else f"=B{r_step}")
    )
    tpot_a_f = (
        f"=IF($B${SUM_ROW_SPEC}>0,C{r_step}/$B${SUM_ROW_SPEC},C{r_step}){a_hf_suffix}"
        if is_decode else f"=C{r_step}"
    )
    dtps_b_f = f"=IF($G${SUM_ROW_MODEL}>0,B{r_ctps}/$G${SUM_ROW_MODEL},0)" if same_deploy else f"=B{r_ctps}/{b_dies}"
    ctps_b_f: str | None = (
        f"=IF(B{r_step}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}*1000/B{r_step},0)"
        if same_deploy else f"=IF(B{r_step}>0,{b_cluster_conc}*{b_acc!r}*1000/B{r_step},0)"
    )
    ctps_a_f: str | None = f"=IF(C{r_step}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}*1000/C{r_step},0)"
    dtps_a_f = f"=IF($G${SUM_ROW_MODEL}>0,C{r_ctps}/$G${SUM_ROW_MODEL},0)"
    ctps_note = f"= Cluster_Concurrency ($G${SUM_ROW_LEN}) × Avg_Accepted ($B${SUM_ROW_SPEC}) × 1000 / Step_ms"
    b_pd, a_pd = _pd_cluster_tps_cells(bundle, "B"), _pd_cluster_tps_cells(after, "C")
    if b_pd is not None:
        b_ctps, ctps_b_f, dtps_b_f = b_pd
    if a_pd is not None:
        a_ctps, ctps_a_f, dtps_a_f = a_pd
    if b_pd is not None or a_pd is not None:
        ctps_note = "PD 集群端到端 = Output × min(Prefill∧KV rps, Decode rps)（求解器胜出配比）；单卡 TPS 仍为 Decode 实例口径"
    kpi_specs: list[tuple[int, str, Any, str | None, int, Any, str | None, int, Any, str | None, int, Any, str | None, int, Any, str | None, int, str, str]] = [
        (
            r_step, "单步总耗时 (Step Latency)",
            b_step, None, STYLE_FLOAT3, a_step, None, STYLE_FLOAT3,
            a_step - b_step, f"=C{r_step}-B{r_step}", STYLE_FLOAT3,
            (a_step - b_step) / b_step if b_step > 0.0 else 0.0, f"=IF(B{r_step}>0,(C{r_step}-B{r_step})/B{r_step},0)", STYLE_PCT,
            b_step / a_step if a_step > 0.0 else 0.0, f"=IF(C{r_step}>0,B{r_step}/C{r_step},0)", STYLE_RATIO,
            "ms", f"全网端到端单步耗时 (单步预算 <= {sla * acc if is_decode else sla:.1f} ms，与 Block 3 实际暴露总和 100% 守恒)",
        ),
        (
            r_tpot, "等效 TPOT (Effective TPOT)" if is_decode else "首字延迟 (Prefill TTFT)",
            b_tpot, tpot_b_f, STYLE_KPI_HIGHLIGHT, a_tpot, tpot_a_f, STYLE_KPI_HIGHLIGHT,
            a_tpot - b_tpot, f"=C{r_tpot}-B{r_tpot}", STYLE_FLOAT3,
            (a_tpot - b_tpot) / b_tpot if b_tpot > 0.0 else 0.0, f"=IF(B{r_tpot}>0,(C{r_tpot}-B{r_tpot})/B{r_tpot},0)", STYLE_PCT,
            b_tpot / a_tpot if a_tpot > 0.0 else 0.0, f"=IF(C{r_tpot}>0,B{r_tpot}/C{r_tpot},0)", STYLE_RATIO,
            "ms/tok" if is_decode else "ms", f"= Step_ms / Avg_Accepted ($B${SUM_ROW_SPEC}) | SLA 目标 <= {sla:.1f} ms",
        ),
        (
            r_sla, "SLA 达标判定 (SLA Verdict)",
            "PASS" if b_pass else "FAIL", f'=IF(B{r_tpot}<=$B${SUM_ROW_SLA},"PASS","FAIL")', STYLE_ALERT_OK if b_pass else STYLE_ALERT_HIGH,
            "PASS" if a_pass else "FAIL", f'=IF(C{r_tpot}<=$B${SUM_ROW_SLA},"PASS","FAIL")', STYLE_ALERT_OK if a_pass else STYLE_ALERT_HIGH,
            sla - a_tpot, f"=$B${SUM_ROW_SLA}-C{r_tpot}", STYLE_FLOAT3,
            (sla - a_tpot) / sla if sla > 0.0 else 0.0, f"=IF($B${SUM_ROW_SLA}>0,($B${SUM_ROW_SLA}-C{r_tpot})/$B${SUM_ROW_SLA},0)", STYLE_PCT,
            "", None, STYLE_TEXT,
            "ms margin", f"Effective TPOT <= SLA ($B${SUM_ROW_SLA}) 为 PASS；Col D/E 为优化后余量",
        ),
        (
            r_dtps, "单卡吞吐 (Single-Die TPS)",
            b_dtps, dtps_b_f, STYLE_KPI_HIGHLIGHT,
            a_dtps, dtps_a_f, STYLE_KPI_HIGHLIGHT,
            a_dtps - b_dtps, f"=C{r_dtps}-B{r_dtps}", STYLE_FLOAT2,
            (a_dtps - b_dtps) / b_dtps if b_dtps > 0.0 else 0.0, f"=IF(B{r_dtps}>0,(C{r_dtps}-B{r_dtps})/B{r_dtps},0)", STYLE_PCT,
            a_dtps / b_dtps if b_dtps > 0.0 else 0.0, f"=IF(B{r_dtps}>0,C{r_dtps}/B{r_dtps},0)", STYLE_RATIO,
            "tok/s/die", f"= Cluster_TPS / Total_Dies ($G${SUM_ROW_MODEL}) — 越高越好",
        ),
        (
            r_ctps, "集群总吞吐 (Cluster Total TPS)",
            b_ctps, ctps_b_f, STYLE_KPI_HIGHLIGHT,
            a_ctps, ctps_a_f, STYLE_KPI_HIGHLIGHT,
            a_ctps - b_ctps, f"=C{r_ctps}-B{r_ctps}", STYLE_FLOAT2,
            (a_ctps - b_ctps) / b_ctps if b_ctps > 0.0 else 0.0, f"=IF(B{r_ctps}>0,(C{r_ctps}-B{r_ctps})/B{r_ctps},0)", STYLE_PCT,
            a_ctps / b_ctps if b_ctps > 0.0 else 0.0, f"=IF(B{r_ctps}>0,C{r_ctps}/B{r_ctps},0)", STYLE_RATIO,
            "tok/s", ctps_note,
        ),
        (
            r_ls, "Lean 4 物理极限下界 L* (Physical Lower Bound)",
            b_lstar, None, STYLE_FLOAT3, a_lstar, None, STYLE_FLOAT3,
            a_lstar - b_lstar, f"=C{r_ls}-B{r_ls}", STYLE_FLOAT3,
            (a_lstar - b_lstar) / b_lstar if b_lstar > 0.0 else 0.0, f"=IF(B{r_ls}>0,(C{r_ls}-B{r_ls})/B{r_ls},0)", STYLE_PCT,
            b_lstar / a_lstar if a_lstar > 0.0 else 0.0, f"=IF(C{r_ls}>0,B{r_ls}/C{r_ls},0)", STYLE_RATIO,
            "ms", f"Cert={after.step_certificate_valid} | Tight: {','.join(after.step_tight_resources) or 'bound'}",
        ),
        (
            r_att, "物理极限达成率 (Physical Limit Attainment L* / Step)",
            b_att, f"=IF(B{r_step}>0,B{r_ls}/B{r_step},0)", STYLE_PCT,
            a_att, f"=IF(C{r_step}>0,C{r_ls}/C{r_step},0)", STYLE_PCT,
            a_att - b_att, f"=C{r_att}-B{r_att}", STYLE_PCT,
            "", None, STYLE_TEXT,
            a_gap_ratio, f"=IF(C{r_ls}>0,C{r_step}/C{r_ls},0)", _gap_ratio_style(a_gap_ratio),
            "%", "Attainment = L* / Step_ms；Col F 为优化后与物理下界倍差 (Step / L*)",
        ),
    ]
    for r, lbl, bv, bf, bst, av, af, ast, dv, df, dst, pv, pf, pst, sv, sf, sst, unit, note in kpi_specs:
        _write_row(ws, r, (
            (lbl, None, STYLE_TEXT), (bv, bf, bst), (av, af, ast),
            (dv, df, dst), (pv, pf, pst), (sv, sf, sst), (unit, None, STYLE_TEXT),
            (note, None, STYLE_TEXT_WRAP),
        ))
        ws.merge_range(r, SUM_KPI_NOTE_START_COL, r, SUMMARY_COLS)
    _write_summary_util_and_wall_rows(ws, bundle, after, is_decode=is_decode)


def _write_summary_util_and_wall_rows(
    ws: Worksheet, bundle: ExportBundle, after: ExportBundle, *, is_decode: bool,
) -> None:
    """Append Hardware Utilization, Amortized us/(tok*die), and Macro Wall rows to Sheet 1 Block 2."""
    cube_note = (
        "Decode 小 Batch 受限于 HBM 访存带宽 (MBU 主导)，Cube 算力受限于算术强度"
        if is_decode else "Prefill Compute-Bound 主导，反映 AIC 矩阵核有效算力利用"
    )
    hw_utils = (
        ("Cube 计算利用率 (AIC MFU)", bundle.network_cube_mfu / PCT_DIVISOR, after.network_cube_mfu / PCT_DIVISOR, cube_note),
        ("Vector 计算利用率 (AIV Util)", bundle.network_vec_mfu / PCT_DIVISOR, after.network_vec_mfu / PCT_DIVISOR, "AIV 向量核执行 Norm / Softmax / Gating / 量化与小包通信展开"),
        ("HBM 访存带宽利用率 (MBU)", bundle.network_mbu / PCT_DIVISOR, after.network_mbu / PCT_DIVISOR, "Decode 核心瓶颈指标：权重 + KV Cache HBM 有效带宽利用率"),
        ("HCCL 通信带宽利用率 (Link BW Util)", bundle.network_link_util / PCT_DIVISOR, after.network_link_util / PCT_DIVISOR, "跨卡 TP/EP/DCP 通信带宽利用率（小包受启动开销制约，切 AIV/CCU 可显著降延迟）"),
    )
    for idx, (lbl, bv, av, note) in enumerate(hw_utils):
        r = SUM_KPI_UTIL_START_ROW + idx
        _write_row(ws, r, (
            (lbl, None, STYLE_TEXT), (bv, None, STYLE_PCT), (av, None, STYLE_PCT),
            (av - bv, f"=C{r}-B{r}", STYLE_PCT),
            ((av - bv) / bv if bv > 0.0 else 0.0, f"=IF(B{r}>0,(C{r}-B{r})/B{r},0)", STYLE_PCT),
            (av / bv if bv > 0.0 else 0.0, f"=IF(B{r}>0,C{r}/B{r},0)", STYLE_RATIO),
            ("%", None, STYLE_TEXT), (note, None, STYLE_TEXT_WRAP),
        ))
        ws.merge_range(r, SUM_KPI_NOTE_START_COL, r, SUMMARY_COLS)

    r_upt = SUM_KPI_AMORTIZED_ROW
    r_dtps = SUM_KPI_DIE_TPS_ROW
    b_upt = bundle.us_per_tok_die
    a_upt = after.us_per_tok_die
    _write_row(ws, r_upt, (
        ("单卡单Token摊薄耗时 (us / tok·die)", None, STYLE_TEXT),
        (b_upt, f"=IF(B{r_dtps}>0,1000000/B{r_dtps},0)", STYLE_KPI_HIGHLIGHT),
        (a_upt, f"=IF(C{r_dtps}>0,1000000/C{r_dtps},0)", STYLE_KPI_HIGHLIGHT),
        (a_upt - b_upt, f"=C{r_upt}-B{r_upt}", STYLE_FLOAT2),
        ((a_upt - b_upt) / b_upt if b_upt > 0.0 else 0.0, f"=IF(B{r_upt}>0,(C{r_upt}-B{r_upt})/B{r_upt},0)", STYLE_PCT),
        (b_upt / a_upt if a_upt > 0.0 else 0.0, f"=IF(C{r_upt}>0,B{r_upt}/C{r_upt},0)", STYLE_RATIO),
        ("us/(tok·die)", None, STYLE_TEXT),
        ("= 10^6 / Single-Die TPS = Exposed_us / N_tok_per_die（与 Sheet 2 / Sheet 4 摊薄列线性守恒）", None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(r_upt, SUM_KPI_NOTE_START_COL, r_upt, SUMMARY_COLS)

    r_ntok = SUM_KPI_NTOK_ROW
    b_ntok = float(bundle.n_tok_per_die)
    a_ntok = float(after.n_tok_per_die)
    a_mem = after.memory_ledger
    b_mem = bundle.memory_ledger
    a_expected_ntok = float(max(1, a_mem.concurrency * a_mem.dp)) * max(1.0, float(after.avg_accepted_tokens)) / float(max(1, a_mem.total_dies))
    b_expected_ntok = float(max(1, b_mem.concurrency * b_mem.dp)) * max(1.0, float(bundle.avg_accepted_tokens)) / float(max(1, b_mem.total_dies))
    same_deploy = (
        b_mem.total_dies == a_mem.total_dies
        and (b_mem.concurrency * b_mem.dp) == (a_mem.concurrency * a_mem.dp)
        and abs(float(bundle.avg_accepted_tokens) - float(after.avg_accepted_tokens)) <= MIN_DENOM_EPS
    )
    a_ntok_matches = abs(a_ntok - a_expected_ntok) <= MIN_DENOM_EPS * max(1.0, abs(a_expected_ntok))
    b_ntok_matches = same_deploy and abs(b_ntok - b_expected_ntok) <= MIN_DENOM_EPS * max(1.0, abs(b_expected_ntok))
    ntok_deploy_f = f"=IF($G${SUM_ROW_MODEL}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}/$G${SUM_ROW_MODEL},0)"
    _write_row(ws, r_ntok, (
        ("单卡单步有效 Token 数 (N_tok_per_die)", None, STYLE_TEXT),
        (b_ntok, ntok_deploy_f if b_ntok_matches else None, STYLE_FLOAT3),
        (a_ntok, ntok_deploy_f if a_ntok_matches else None, STYLE_FLOAT3),
        (a_ntok - b_ntok, f"=C{r_ntok}-B{r_ntok}", STYLE_FLOAT3),
        ((a_ntok - b_ntok) / b_ntok if b_ntok > 0.0 else 0.0, f"=IF(B{r_ntok}>0,(C{r_ntok}-B{r_ntok})/B{r_ntok},0)", STYLE_PCT),
        (a_ntok / b_ntok if b_ntok > 0.0 else 0.0, f"=IF(B{r_ntok}>0,C{r_ntok}/B{r_ntok},0)", STYLE_RATIO),
        ("tok/die", None, STYLE_TEXT),
        ("= Cluster_Concurrency × Avg_Accepted / Total_Dies（供 Sheet 2 / Sheet 4 全精度摊薄公式直接引用）", None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(r_ntok, SUM_KPI_NOTE_START_COL, r_ntok, SUMMARY_COLS)

    wall = after.macro_wall_trigger
    wall_kind = wall.wall_kind if wall is not None else "BALANCED"
    wall_act = (
        "; ".join(wall.recommended_actions[:2])
        if (wall is not None and wall.recommended_actions)
        else "当前显存与计算余量均衡，按 Sheet 4 摊薄收益排序推进算子与配置优化"
    )
    r_wall = SUM_KPI_WALL_ROW
    ws.set_row_height(r_wall, WRAP_ROW_HEIGHT_PT)
    _write_row(ws, r_wall, (
        ("物理墙诊断 & 绑定声明 (Macro Wall & Binding)", None, STYLE_TEXT),
        (format_binding_badge(bundle.binding_state), None, STYLE_OVERLAP if bundle.binding_state == BINDING_STATE_BOUND else STYLE_OPT_BADGE),
        (format_binding_badge(after.binding_state), None, STYLE_OPT_BADGE if after.has_unbound_formula_ops else STYLE_OVERLAP),
        (wall_kind, None, STYLE_ALERT_MID if wall_kind != "BALANCED" else STYLE_ALERT_OK),
        (f"Before N_tok/die={bundle.n_tok_per_die:.2f}", None, STYLE_TEXT),
        (f"After N_tok/die={after.n_tok_per_die:.2f}", None, STYLE_TEXT),
        ("Wall / Mode", None, STYLE_TEXT),
        (wall_act, None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(r_wall, SUM_KPI_NOTE_START_COL, r_wall, SUMMARY_COLS)


def _populate_summary_macro_modules_block(
    ws: Worksheet,
    bundle: ExportBundle,
    compare_bundle: ExportBundle | None,
    macro_rows: Sequence[MacroModuleRow],
) -> int:
    """Block 3: Two-Level L1/L2 Hierarchy + Explicit Masking Columns (100% conserved)."""
    after = compare_bundle or bundle
    _write_banner(
        ws, SUM_B3_BANNER_ROW, SUMMARY_COLS,
        "3. 全网模块两级拆解与并行掩盖对账 (L1/L2 Hierarchy — 实际暴露合计 = 100% 单步耗时，被掩盖列不计入总和)",
        STYLE_SECTION,
    )
    headers = (
        "模块层级 (L1 / L2 Module)", "层数 (Layers)",
        "Before 流内原始 (ms)", "Before 被掩盖 (ms) [不计入总和]",
        "Before 实际暴露 (ms)", "Before 暴露占比 (%)",
        "After 流内原始 (ms)", "After 被掩盖 (ms) [不计入总和]",
        "After 实际暴露 (ms)", "After 暴露占比 (%)",
        "暴露净省 Δ (ms)", "物理极限 L* (ms)", "主导瓶颈 & 掩盖说明",
    )
    _write_headers(ws, SUM_B3_HDR_ROW, headers)

    by_l1: dict[str, list[MacroModuleRow]] = defaultdict(list)
    for mod in macro_rows:
        by_l1[mod.l1_group].append(mod)
    ordered_l1 = [l1 for l1, _ in MACRO_L1_HIERARCHY if l1 in by_l1]
    for extra in by_l1:
        if extra not in ordered_l1:
            ordered_l1.append(extra)

    tot_bef = sum(m.before_ms for m in macro_rows)
    tot_aft = sum(m.after_ms for m in macro_rows)
    tot_raw_bef = sum(m.raw_before_ms for m in macro_rows)
    tot_raw_aft = sum(m.raw_after_ms for m in macro_rows)
    tot_mask_bef = sum(m.masked_before_ms for m in macro_rows)
    tot_mask_aft = sum(m.masked_after_ms for m in macro_rows)
    tot_lstar = sum(m.lstar_ms for m in macro_rows)

    total_data_rows = sum(1 + len(by_l1[l1]) for l1 in ordered_l1)
    tot_row = SUM_B3_DATA_START_ROW + total_data_rows
    row_cursor = SUM_B3_DATA_START_ROW
    l1_row_indices: list[int] = []

    for l1_name in ordered_l1:
        children = by_l1[l1_name]
        r_l1 = row_cursor
        l1_row_indices.append(r_l1)
        c_start = r_l1 + 1
        c_end = r_l1 + len(children)
        l1_raw_b = sum(c.raw_before_ms for c in children)
        l1_msk_b = sum(c.masked_before_ms for c in children)
        l1_exp_b = sum(c.before_ms for c in children)
        l1_raw_a = sum(c.raw_after_ms for c in children)
        l1_msk_a = sum(c.masked_after_ms for c in children)
        l1_exp_a = sum(c.after_ms for c in children)
        l1_ls = sum(c.lstar_ms for c in children)
        ws.set_row_height(r_l1, WRAP_ROW_HEIGHT_PT)
        _write_row(ws, r_l1, (
            (l1_name, None, STYLE_L1_TEXT),
            (f"{len(children)} 子模块", None, STYLE_L1_TEXT),
            (l1_raw_b, f"=SUM(C{c_start}:C{c_end})", STYLE_L1_NUM),
            (l1_msk_b, f"=SUM(D{c_start}:D{c_end})", STYLE_L1_NUM),
            (l1_exp_b, f"=SUM(E{c_start}:E{c_end})", STYLE_L1_NUM),
            (l1_exp_b / tot_bef if tot_bef > 0.0 else 0.0, f"=IF($E${tot_row}>0,E{r_l1}/$E${tot_row},0)", STYLE_L1_PCT),
            (l1_raw_a, f"=SUM(G{c_start}:G{c_end})", STYLE_L1_NUM),
            (l1_msk_a, f"=SUM(H{c_start}:H{c_end})", STYLE_L1_NUM),
            (l1_exp_a, f"=SUM(I{c_start}:I{c_end})", STYLE_L1_NUM),
            (l1_exp_a / tot_aft if tot_aft > 0.0 else 0.0, f"=IF($I${tot_row}>0,I{r_l1}/$I${tot_row},0)", STYLE_L1_PCT),
            (l1_exp_b - l1_exp_a, f"=E{r_l1}-I{r_l1}", STYLE_L1_NUM),
            (l1_ls, f"=SUM(L{c_start}:L{c_end})", STYLE_L1_NUM),
            (f"L1 汇总（暴露占比 Before {100.0 * l1_exp_b / tot_bef if tot_bef > 0 else 0:.1f}% -> After {100.0 * l1_exp_a / tot_aft if tot_aft > 0 else 0:.1f}%）", None, STYLE_L1_TEXT),
        ))
        row_cursor = c_start
        for idx, mod in enumerate(children):
            r = row_cursor
            ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
            prefix = "    └─ " if idx == len(children) - 1 else "    ├─ "
            is_full_mask = mod.after_ms <= OVERLAP_EXPOSED_EPS_US and mod.raw_after_ms > 0.0
            is_part_mask = (not is_full_mask) and (abs(mod.masked_after_ms) > MASKED_HIGHLIGHT_EPS_MS or abs(mod.masked_before_ms) > MASKED_HIGHLIGHT_EPS_MS)
            note_style = STYLE_OVERLAP if is_full_mask else (STYLE_PARTIAL_OVERLAP if is_part_mask else STYLE_TEXT_WRAP)
            _write_row(ws, r, (
                (f"{prefix}{mod.module_name}", None, STYLE_TEXT_WRAP),
                (mod.layer_calls_label, None, STYLE_MERGED),
                (mod.raw_before_ms, None, STYLE_FLOAT3),
                (mod.masked_before_ms, None, STYLE_MASKED_NUM if mod.masked_before_ms < -MASKED_HIGHLIGHT_EPS_MS else STYLE_FLOAT3),
                (mod.before_ms, f"=C{r}+D{r}", STYLE_FLOAT3),
                (mod.before_share_pct / PCT_DIVISOR, f"=IF($E${tot_row}>0,E{r}/$E${tot_row},0)", STYLE_PCT),
                (mod.raw_after_ms, None, STYLE_FLOAT3),
                (mod.masked_after_ms, None, STYLE_MASKED_NUM if mod.masked_after_ms < -MASKED_HIGHLIGHT_EPS_MS else STYLE_FLOAT3),
                (mod.after_ms, f"=G{r}+H{r}", STYLE_FLOAT3),
                (mod.after_share_pct / PCT_DIVISOR, f"=IF($I${tot_row}>0,I{r}/$I${tot_row},0)", STYLE_PCT),
                (mod.saved_ms, f"=E{r}-I{r}", STYLE_FLOAT3),
                (mod.lstar_ms, None, STYLE_FLOAT3),
                (mod.masking_note or mod.dominant_bound, None, note_style),
            ))
            row_cursor += 1

    def _sum_l1(col_let: str) -> str:
        return "=" + "+".join(f"{col_let}{idx}" for idx in l1_row_indices) if l1_row_indices else "=0"

    layer_tot_lbl = f"{after.total_target_layers}L+{after.total_draft_layers}D" if after.total_draft_layers > 0 else f"{after.total_target_layers}L"
    _write_row(ws, tot_row, (
        ("全网端到端合计 (Grand Total = SUM of L1 实际暴露 = Block 2 Step Latency)", None, STYLE_TOTAL_TEXT),
        (layer_tot_lbl, None, STYLE_TOTAL_TEXT),
        (tot_raw_bef, _sum_l1("C"), STYLE_TOTAL_FLOAT3),
        (tot_mask_bef, _sum_l1("D"), STYLE_TOTAL_FLOAT3),
        (tot_bef, _sum_l1("E"), STYLE_TOTAL_FLOAT3),
        (1.0 if tot_bef > 0.0 else 0.0, _sum_l1("F"), STYLE_TOTAL_PCT),
        (tot_raw_aft, _sum_l1("G"), STYLE_TOTAL_FLOAT3),
        (tot_mask_aft, _sum_l1("H"), STYLE_TOTAL_FLOAT3),
        (tot_aft, _sum_l1("I"), STYLE_TOTAL_FLOAT3),
        (1.0 if tot_aft > 0.0 else 0.0, _sum_l1("J"), STYLE_TOTAL_PCT),
        (tot_bef - tot_aft, f"=E{tot_row}-I{tot_row}", STYLE_TOTAL_FLOAT3),
        (tot_lstar, _sum_l1("L"), STYLE_TOTAL_FLOAT3),
        ("100% 端到端守恒（无负号抵消黑盒行；流内原始 + 被掩盖 = 实际暴露）", None, STYLE_TOTAL_TEXT),
    ))
    return tot_row + 1


def _populate_summary_action_roadmap_block(
    ws: Worksheet, start_row: int, roadmap: Sequence[ActionRoadmapRow],
) -> int:
    """Block 4: Concrete 3-Track Actionable Roadmap (Config / Framework / Kernel)."""
    if not roadmap:
        return start_row
    r_banner = start_row + 1
    _write_banner(
        ws, r_banner, SUMMARY_COLS,
        "4. 优化落地指引清单（A: 已有特效与通信展开配置 / B: 推理框架改造 / C: 算子新写与深度调优 — 可直接执行）",
        STYLE_SECTION,
    )
    r_hdr = r_banner + 1
    ws.set_row_height(r_hdr, HEADER_ROW_HEIGHT_PT)
    ws.set_cell(r_hdr, 1, "落地赛道 (Track)", style=STYLE_HEADER)
    ws.merge_range(r_hdr, 1, r_hdr, SUM_B4_TRACK_END)
    ws.set_cell(r_hdr, SUM_B4_TITLE_START, "优化项 (Item)", style=STYLE_HEADER)
    ws.merge_range(r_hdr, SUM_B4_TITLE_START, r_hdr, SUM_B4_TITLE_END)
    ws.set_cell(r_hdr, SUM_B4_GAIN_COL, "预期收益 Δ (ms)", style=STYLE_HEADER)
    ws.set_cell(r_hdr, SUM_B4_DETAIL_START, "可直接落地的配置命令 / 框架改造 / 算子方案 (Actionable How-To)", style=STYLE_HEADER)
    ws.merge_range(r_hdr, SUM_B4_DETAIL_START, r_hdr, SUM_B4_DETAIL_END)
    ws.set_cell(r_hdr, SUM_B4_REF_START, "明细位置 (Ref)", style=STYLE_HEADER)
    ws.merge_range(r_hdr, SUM_B4_REF_START, r_hdr, SUMMARY_COLS)

    r_cur = r_hdr + 1
    for item in roadmap:
        ws.set_row_height(r_cur, BACKLOG_ROW_HEIGHT_PT)
        tr_style = (
            STYLE_OVERLAP if item.track == TRACK_CONFIG_ENV
            else (STYLE_OPT_BADGE if item.track == TRACK_FRAMEWORK else STYLE_NEW_KERNEL)
        )
        ws.set_cell(r_cur, 1, item.track_label, style=tr_style)
        ws.merge_range(r_cur, 1, r_cur, SUM_B4_TRACK_END)
        ws.set_cell(r_cur, SUM_B4_TITLE_START, item.title, style=STYLE_TEXT_WRAP)
        ws.merge_range(r_cur, SUM_B4_TITLE_START, r_cur, SUM_B4_TITLE_END)
        ws.set_cell(r_cur, SUM_B4_GAIN_COL, item.expected_gain_ms, style=STYLE_FLOAT3)
        ws.set_cell(r_cur, SUM_B4_DETAIL_START, item.action_detail, style=STYLE_TEXT_WRAP)
        ws.merge_range(r_cur, SUM_B4_DETAIL_START, r_cur, SUM_B4_DETAIL_END)
        ws.set_cell(r_cur, SUM_B4_REF_START, item.cross_ref, style=STYLE_MERGED)
        ws.merge_range(r_cur, SUM_B4_REF_START, r_cur, SUMMARY_COLS)
        r_cur += 1
    return r_cur


def _populate_summary_sheet(
    ws: Worksheet,
    bundle: ExportBundle,
    *,
    compare_bundle: ExportBundle | None = None,
    macro_rows: Sequence[MacroModuleRow] = (),
    pending_items: Sequence[PendingKernelItem] = (),
) -> None:
    _apply_col_widths(ws, SUMMARY_COL_WIDTHS)
    ws.freeze_panes(SUMMARY_FREEZE_ROWS, 0)
    track_note = (
        f" | Before({bundle.feature_track or 'base'}) -> After({compare_bundle.feature_track or 'cand'})"
        if compare_bundle is not None else (f" | Track={bundle.feature_track}" if bundle.feature_track else "")
    )
    _write_banner(
        ws, 1, SUMMARY_COLS,
        f"MindIE-Sim Ascend Performance & Optimization Summary | {bundle.model_name} @ {bundle.hardware_sku}{track_note}",
        STYLE_TITLE,
    )
    _populate_summary_deployment_block(ws, bundle, compare_bundle)
    _populate_summary_kpi_block(ws, bundle, compare_bundle)
    next_row = _populate_summary_macro_modules_block(ws, bundle, compare_bundle, macro_rows)
    roadmap = build_actionable_roadmap(bundle, compare_bundle or bundle, pending_items)
    next_row = _populate_summary_action_roadmap_block(ws, next_row, roadmap)
    if bundle.reconciliation is not None:
        rec = bundle.reconciliation
        r_b = next_row + 1
        _write_banner(ws, r_b, SUMMARY_COLS, f"5. Measured msprof Calibration Summary ({rec.source_label} — See {SHEET_MEASURED})", STYLE_SECTION)
        delta = rec.matched_sim_sum_us - rec.matched_measured_sum_us
        ape = (abs(delta) / rec.matched_measured_sum_us) if rec.matched_measured_sum_us > 0.0 else 0.0
        _write_row(ws, r_b + 1, (
            ("Matched Groups Sum (us)", None, STYLE_TEXT),
            (f"{rec.matched_kernels}/{rec.total_kernels} kernels", None, STYLE_MERGED),
            (rec.matched_sim_sum_us, None, STYLE_FLOAT2),
            ("Sim Pred (us)", None, STYLE_TEXT),
            (rec.matched_measured_sum_us, None, STYLE_FLOAT2),
            ("Measured (us)", None, STYLE_TEXT),
            (delta, f"=C{r_b + 1}-E{r_b + 1}", STYLE_FLOAT2),
            (ape, f"=IF(E{r_b + 1}>0,ABS(C{r_b + 1}-E{r_b + 1})/E{r_b + 1},0)", STYLE_PCT),
            (f"Full 1-to-N details on {SHEET_MEASURED}", None, STYLE_TEXT),
        ))
        ws.merge_range(r_b + 1, 9, r_b + 1, SUMMARY_COLS)


def _populate_comparison_breakdown_sheet(
    ws: Worksheet,
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    aligned_rows: Sequence[AlignedComparisonRow],
    agent_designs: Mapping[str, Any] | None,
) -> dict[int, tuple[int, int]]:
    """Populate Sheet 2 with strict topological order, up-front Stream&Masking, trailing Source Ref."""
    _apply_col_widths(ws, CMP_COL_WIDTHS)
    ws.freeze_panes(CMP_FREEZE_ROWS, CMP_FREEZE_COLS)
    _write_banner(
        ws, 1, CMP_COLS,
        (
            f"Before ({before_bundle.feature_track or 'baseline'}) vs After ({after_bundle.feature_track or 'candidate'}) "
            f"Topological Operator Breakdown | {before_bundle.model_name} @ {before_bundle.hardware_sku} ({before_bundle.phase.upper()})"
        ),
        STYLE_TITLE,
    )
    headers = (
        "Archetype (Scope & Layer Range)", "Repeats (×N)", "Submodule",
        "Logical Role (带作用域标注)", "并发流与掩盖关系 (Stream & Masking)",
        "Rank-Local Shapes & Dtype",
        "Before Ascend Kernel [Backend]", "Before 流内单次 (us)", "Before 单次暴露 (us)",
        "Before 全网暴露 (ms)", "Before Bound",
        "After Ascend Kernel [Backend]", "Kernel Status (算子状态/优化手段)",
        "After 流内单次 (us)", "After 单次暴露 (us)", "After 全网暴露 (ms)",
        "暴露净省 Δ (ms)", "Speedup", "After Bound",
        "Lean L* Model (ms)", "Gap vs L* (ms)",
        "3-Track 落地指引 (配置 / 算子 / 框架)", "源码出处 / Source Ref",
        "Before 摊薄耗时 (us/tok·die)", "After 摊薄耗时 (us/tok·die)",
        "摊薄净省 Δ (us/tok·die)", "绑定状态 (Binding State)",
    )
    _write_headers(ws, 2, headers)

    b_ntok = max(float(before_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    a_ntok = max(float(after_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    by_arch: list[tuple[str, str, int, list[tuple[int, AlignedComparisonRow]]]] = []
    for row_idx, item in enumerate(aligned_rows):
        if not by_arch or by_arch[-1][0] != item.archetype_id:
            by_arch.append((item.archetype_id, item.archetype_title, item.repeat_count, []))
        by_arch[-1][3].append((row_idx, item))

    row_cursor = CMP_DATA_START_ROW
    row_spans: dict[int, tuple[int, int]] = {}
    subtotal_rows: list[int] = []
    tot_bef_ms = tot_aft_ms = tot_lstar_ms = 0.0
    tot_bef_upt = tot_aft_upt = 0.0
    for _, arch_title, _, arch_items in by_arch:
        arch_start = row_cursor
        sub_spans: list[tuple[str, int, int]] = []
        arch_bef_ms = arch_aft_ms = arch_lstar_ms = 0.0
        arch_bef_upt = arch_aft_upt = 0.0
        for row_idx, item in arch_items:
            row_count = max(1, len(item.before_ops), len(item.after_ops))
            r_start = row_cursor
            r_end = row_cursor + row_count - 1
            b_ms, a_ms, l_ms, b_u, a_u = _write_comparison_group(
                ws, r_start, r_end, item, agent_designs, b_ntok, a_ntok,
            )
            row_spans[row_idx] = (r_start, r_end)
            arch_bef_ms += b_ms
            arch_aft_ms += a_ms
            arch_lstar_ms += l_ms
            arch_bef_upt += b_u
            arch_aft_upt += a_u
            _extend_span(sub_spans, item.submodule, r_start, r_end)
            row_cursor = r_end + 1
        arch_end = row_cursor - 1
        ws.set_cell(arch_start, CMP_COL_ARCH, arch_title, style=STYLE_MERGED)
        ws.merge_range(arch_start, CMP_COL_ARCH, arch_end, CMP_COL_ARCH)
        for sub_label, s1, s2 in sub_spans:
            ws.set_cell(s1, CMP_COL_SUB, sub_label, style=STYLE_MERGED)
            ws.merge_range(s1, CMP_COL_SUB, s2, CMP_COL_SUB)
        r_sub = row_cursor
        _write_comparison_subtotal_row(
            ws, r_sub, arch_title, arch_start, arch_end,
            arch_bef_ms, arch_aft_ms, arch_lstar_ms, arch_bef_upt, arch_aft_upt,
        )
        subtotal_rows.append(r_sub)
        tot_bef_ms += arch_bef_ms
        tot_aft_ms += arch_aft_ms
        tot_lstar_ms += arch_lstar_ms
        tot_bef_upt += arch_bef_upt
        tot_aft_upt += arch_aft_upt
        row_cursor += 1
    _write_comparison_grand_total(
        ws, row_cursor, subtotal_rows, tot_bef_ms, tot_aft_ms, tot_lstar_ms, tot_bef_upt, tot_aft_upt,
    )
    return row_spans


def _write_cmp_before_sides(
    ws: Worksheet, r_start: int, row_count: int, item: AlignedComparisonRow,
    rep: int, b_ntok: float, *, b_rep: int | None = None,
) -> tuple[float, float]:
    """Populate Before columns (B, D..K, W, X) across ``row_count`` rows."""
    eff_b_rep = int(b_rep) if b_rep is not None else rep
    same_rep = (eff_b_rep == rep)
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
            _write_row(ws, r, (
                (b_op.logical_role or b_op.op_id, None, STYLE_TEXT_WRAP),
                (stream_txt, None, stream_style),
                (b_op.tensor_contract, None, STYLE_TEXT_WRAP),
                (_kernel_with_backend(b_op), None, STYLE_TEXT_WRAP),
                (b_op.dur_us, None, STYLE_FLOAT3),
                (b_exp_us, None, STYLE_FLOAT3),
                (b_ms, ms_formula, STYLE_FLOAT3),
                (b_op.bound_display, None, STYLE_OVERLAP if b_op.is_overlapped else STYLE_TEXT),
            ), start_col=CMP_COL_ROLE)
            ws.set_cell(r, CMP_COL_SOURCE_REF, b_op.source_ref or "-", style=STYLE_MUTED_TEXT)
            ws.set_cell(r, CMP_COL_BEF_UPT, b_upt, formula=upt_formula, style=STYLE_FLOAT2)
        else:
            ref_a = item.after_ops[offset] if offset < len(item.after_ops) else a_primary
            assert ref_a is not None
            stream_txt, stream_style = _stream_masking_badge(ref_a)
            placeholder = "— [随新策略引入, 0.000 ms]" if item.relation == "added" else "(Cluster Pad)"
            _write_row(ws, r, (
                (ref_a.logical_role or ref_a.op_id, None, STYLE_TEXT_WRAP),
                (stream_txt, None, stream_style),
                (ref_a.tensor_contract, None, STYLE_TEXT_WRAP),
                (placeholder, None, STYLE_TEXT),
                (0.0, None, STYLE_FLOAT3),
                (0.0, None, STYLE_FLOAT3),
                (0.0, ms_formula, STYLE_FLOAT3),
                ("-", None, STYLE_TEXT),
            ), start_col=CMP_COL_ROLE)
            ws.set_cell(r, CMP_COL_SOURCE_REF, ref_a.source_ref or "-", style=STYLE_MUTED_TEXT)
            ws.set_cell(r, CMP_COL_BEF_UPT, 0.0, formula=upt_formula, style=STYLE_FLOAT2)
    return before_ms_sum, before_upt_sum


def _write_cmp_after_cluster(
    ws: Worksheet, r_start: int, r_end: int, item: AlignedComparisonRow,
    rep: int, before_ms_sum: float, before_upt_sum: float, a_ntok: float,
    *, a_rep: int | None = None,
) -> tuple[float, float, float]:
    """Populate After columns for M->N (len(after_ops) >= 2) re-association clusters."""
    eff_a_rep = int(a_rep) if a_rep is not None else rep
    same_rep = (eff_a_rep == rep)
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
    ws.set_cell(r_start, CMP_COL_ACTION, item.action_note, style=STYLE_TEXT_WRAP)
    ws.set_cell(r_start, CMP_COL_SAVED_UPT, before_upt_sum - aft_upt_sum, formula=f"=SUM(X{r_start}:X{r_end})-SUM(Y{r_start}:Y{r_end})", style=STYLE_FLOAT2)
    if r_end > r_start:
        for col in CMP_CLUSTER_MERGE_COLS:
            ws.merge_range(r_start, col, r_end, col)
    return aft_ms_sum, lstar_ms_sum, aft_upt_sum


def _write_comparison_group(
    ws: Worksheet, r_start: int, r_end: int, item: AlignedComparisonRow,
    agent_designs: Mapping[str, Any] | None, b_ntok: float, a_ntok: float,
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
        ov = _lookup_agent_design_bundle(
            agent_designs,
            kernel_key=a_primary.ascend_kernel,
            target_kernel=_kernel_with_backend(a_primary),
            logical_role=a_primary.logical_role,
        )
        action_txt = (
            f"[Agent Design] {ov['kernel_design']}" if ov["kernel_design"]
            else (f"{item.action_note} -> 详见 {SHEET_PENDING_KERNELS}" if a_primary.needs_new_kernel else item.action_note)
        )
        status_style = (
            STYLE_NEW_KERNEL if a_primary.needs_new_kernel
            else (STYLE_OPT_BADGE if a_primary.opt_pillar in {"pillar2_roofline", "pillar3_comm"} else STYLE_TEXT_WRAP)
        )
        aft_ms_formula = f"=O{r_start}*B{r_start}/1000" if a_rep == rep else f"=O{r_start}*{a_rep}/1000"
        _write_row(ws, r_start, (
            (_kernel_with_backend(a_primary), None, STYLE_TEXT_WRAP),
            (a_primary.kernel_status, None, status_style),
            (a_primary.dur_us, None, STYLE_FLOAT3),
            (a_exp_us, None, STYLE_FLOAT3),
            (aft_ms, aft_ms_formula, STYLE_FLOAT3),
            (saved_ms, f"=SUM(J{r_start}:J{r_end})-P{r_start}", STYLE_FLOAT3),
            (speedup, f"=IF(P{r_start}>0,SUM(J{r_start}:J{r_end})/P{r_start},0)", STYLE_RATIO),
            (a_primary.bound_display, None, STYLE_OVERLAP if a_primary.is_overlapped else STYLE_TEXT),
            (lstar_ms, None, STYLE_FLOAT3),
            (gap_ms, f"=MAX(0,P{r_start}-T{r_start})", STYLE_FLOAT3),
            (action_txt, None, STYLE_TEXT_WRAP),
        ), start_col=CMP_COL_AFT_KERN)
        ws.set_cell(r_start, CMP_COL_AFT_UPT, aft_upt, formula=f"=IF({SUM_AFT_NTOK_REF}>0,P{r_start}*1000/{SUM_AFT_NTOK_REF},0)", style=STYLE_FLOAT2)
        ws.set_cell(r_start, CMP_COL_SAVED_UPT, before_upt_sum - aft_upt, formula=f"=SUM(X{r_start}:X{r_end})-Y{r_start}", style=STYLE_FLOAT2)
        ws.set_cell(r_start, CMP_COL_BINDING, format_binding_badge(a_primary.binding_state), style=STYLE_TEXT_WRAP)
    else:
        aft_ms = lstar_ms = aft_upt = 0.0
        elim_lbl = (
            f"[↑ 已并入 {item.non_adjacent_anchor_kernel}, 0.000 ms]"
            if item.relation == "fused_into_anchor" else "— [随策略切换/融合消除, 0.000 ms]"
        )
        elim_status = "已并入融合锚点 (0 重复计数)" if item.relation == "fused_into_anchor" else "已融合/消除"
        _write_row(ws, r_start, (
            (elim_lbl, None, STYLE_OVERLAP), (elim_status, None, STYLE_OVERLAP),
            (0.0, None, STYLE_FLOAT3), (0.0, None, STYLE_FLOAT3),
            (0.0, "=0", STYLE_FLOAT3), (before_ms_sum, f"=SUM(J{r_start}:J{r_end})-P{r_start}", STYLE_FLOAT3),
            (0.0, f"=IF(P{r_start}>0,SUM(J{r_start}:J{r_end})/P{r_start},0)", STYLE_RATIO),
            ("已消除", None, STYLE_OVERLAP), (0.0, None, STYLE_FLOAT3),
            (0.0, f"=MAX(0,P{r_start}-T{r_start})", STYLE_FLOAT3), (item.action_note, None, STYLE_TEXT_WRAP),
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


def _populate_standalone_breakdown_sheet(
    ws: Worksheet, bundle: ExportBundle,
) -> dict[int, tuple[int, int]]:
    _apply_col_widths(ws, STD_COL_WIDTHS)
    ws.freeze_panes(CMP_FREEZE_ROWS, CMP_FREEZE_COLS)
    _write_banner(
        ws, 1, STD_COLS,
        f"Layer-Archetype -> Submodule -> Ascend Kernel Breakdown | {bundle.model_name} @ {bundle.hardware_sku} ({bundle.phase.upper()})",
        STYLE_TITLE,
    )
    headers = (
        "Archetype (Scope & Layer Range)", "Repeats (×N)", "Submodule",
        "Logical Role (带作用域标注)", "并发流与掩盖关系 (Stream & Masking)",
        "Rank-Local Shapes & Dtype", "Ascend Kernel [Backend]",
        "Kernel Status (算子状态/开发需求)",
        "流内单次 (us)", "单次暴露 (us)", "全网暴露 (ms)",
        "Lean L* Single (us)", "Lean L* Model (ms)", "Gap vs L* (ms)",
        "Bottleneck Bound", "源码出处 / Source Ref",
    )
    _write_headers(ws, 2, headers)

    by_arch: list[tuple[str, str, list[tuple[int, OperatorExportRow]]]] = []
    for op_idx, op in enumerate(bundle.operators):
        aid = op.archetype_id or op.layer_label
        if not by_arch or by_arch[-1][0] != aid:
            by_arch.append((aid, op.archetype_title or op.layer_label, []))
        by_arch[-1][2].append((op_idx, op))

    row_cursor = STD_DATA_START_ROW
    row_spans: dict[int, tuple[int, int]] = {}
    subtotal_rows: list[int] = []
    tot_ms = tot_lstar_ms = 0.0
    for _, arch_title, ops in by_arch:
        arch_start = row_cursor
        sub_spans: list[tuple[str, int, int]] = []
        arch_ms = arch_lstar_ms = 0.0
        for op_idx, op in ops:
            r = row_cursor
            row_spans[op_idx] = (r, r)
            ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
            rep = max(1, int(op.repeat_count))
            stream_txt, stream_style = _stream_masking_badge(op)
            exp_us = 0.0 if op.is_overlapped else op.exposed_dur_us
            exp_ms = exp_us * rep / US_PER_MS
            l_ms = 0.0 if op.is_overlapped else (op.l_star_lean_us * rep / US_PER_MS)
            arch_ms += exp_ms
            arch_lstar_ms += l_ms
            ws.set_cell(r, STD_COL_REP, rep, style=STYLE_INT)
            _write_row(ws, r, (
                (op.logical_role or op.op_id, None, STYLE_TEXT_WRAP),
                (stream_txt, None, stream_style),
                (op.tensor_contract, None, STYLE_TEXT_WRAP),
                (_kernel_with_backend(op), None, STYLE_TEXT_WRAP),
                (op.kernel_status, None, STYLE_NEW_KERNEL if op.needs_new_kernel else STYLE_TEXT_WRAP),
                (op.dur_us, None, STYLE_FLOAT3),
                (exp_us, None, STYLE_FLOAT3),
                (exp_ms, f"=J{r}*B{r}/1000", STYLE_FLOAT3),
                (op.l_star_lean_us, None, STYLE_FLOAT3),
                (l_ms, "=0" if op.is_overlapped else f"=L{r}*B{r}/1000", STYLE_FLOAT3),
                (max(0.0, exp_ms - l_ms), f"=MAX(0,K{r}-M{r})", STYLE_FLOAT3),
                (op.bound_display, None, STYLE_OVERLAP if op.is_overlapped else STYLE_TEXT),
                (op.source_ref or "-", None, STYLE_MUTED_TEXT),
            ), start_col=STD_COL_ROLE)
            _extend_span(sub_spans, op.submodule, r, r)
            row_cursor += 1
        arch_end = row_cursor - 1
        ws.set_cell(arch_start, STD_COL_ARCH, arch_title, style=STYLE_MERGED)
        ws.merge_range(arch_start, STD_COL_ARCH, arch_end, STD_COL_ARCH)
        for sub_label, s1, s2 in sub_spans:
            ws.set_cell(s1, STD_COL_SUB, sub_label, style=STYLE_MERGED)
            ws.merge_range(s1, STD_COL_SUB, s2, STD_COL_SUB)
        r_sub = row_cursor
        ws.set_cell(r_sub, 1, f"Subtotal — {arch_title}", style=STYLE_SUBTOTAL_TEXT)
        ws.merge_range(r_sub, 1, r_sub, STD_SUBTOTAL_LABEL_END_COL)
        ws.set_cell(r_sub, STD_COL_EXP_MS, arch_ms, formula=f"=SUM(K{arch_start}:K{arch_end})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_LSTAR_US, "", style=STYLE_SUBTOTAL_TEXT)
        ws.set_cell(r_sub, STD_COL_LSTAR_MS, arch_lstar_ms, formula=f"=SUM(M{arch_start}:M{arch_end})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_GAP_MS, max(0.0, arch_ms - arch_lstar_ms), formula=f"=MAX(0,K{r_sub}-M{r_sub})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
        ws.merge_range(r_sub, STD_COL_BOUND, r_sub, STD_COL_SOURCE_REF)
        subtotal_rows.append(r_sub)
        tot_ms += arch_ms
        tot_lstar_ms += arch_lstar_ms
        row_cursor += 1

    ws.set_cell(row_cursor, 1, "WHOLE-MODEL TOTAL (实际暴露耗时合计)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(row_cursor, 1, row_cursor, STD_SUBTOTAL_LABEL_END_COL)
    sum_k = "=" + "+".join(f"K{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_m = "=" + "+".join(f"M{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    ws.set_cell(row_cursor, STD_COL_EXP_MS, tot_ms, formula=sum_k, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_LSTAR_US, "", style=STYLE_TOTAL_TEXT)
    ws.set_cell(row_cursor, STD_COL_LSTAR_MS, tot_lstar_ms, formula=sum_m, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_GAP_MS, max(0.0, tot_ms - tot_lstar_ms), formula=f"=MAX(0,K{row_cursor}-M{row_cursor})", style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_BOUND, f"{len(bundle.operators)} archetype ops", style=STYLE_TOTAL_TEXT)
    ws.merge_range(row_cursor, STD_COL_BOUND, row_cursor, STD_COL_SOURCE_REF)
    return row_spans


def _populate_memory_control_panel(ws: Worksheet, mem: MemoryCalculatorLedger) -> None:
    c_lbl, c_val = MEMORY_CONTROL_COL_LABEL, MEMORY_CONTROL_COL_VALUE
    ws.set_cell(2, c_lbl, "Parallel & Quant Control Panel", style=STYLE_SECTION)
    ws.merge_range(2, c_lbl, 2, c_val)
    controls: tuple[tuple[str, Any, str | None, int], ...] = (
        ("Weight Quant Preset", mem.weight_quant, None, STYLE_INPUT_TEXT),
        ("KV/Indexer/KeyPool Dtype", mem.cache_dtype, None, STYLE_INPUT_TEXT),
        ("Engram / PLE Host Offload", "ON" if mem.engram_host_offload else "OFF", None, STYLE_INPUT_TEXT),
        ("Apply Preset To Ignored Attn/Linear", "ON" if mem.apply_preset_to_ignored else "OFF", None, STYLE_INPUT_TEXT),
        ("Include Vision Tower (ViT)", "ON" if mem.include_vision_tower else "OFF", None, STYLE_INPUT_TEXT),
        ("DP", mem.dp, None, STYLE_INPUT_NUM),
        ("TP", mem.tp, None, STYLE_INPUT_NUM),
        ("EP", mem.ep, None, STYLE_INPUT_NUM),
        ("PP", mem.pp, None, STYLE_INPUT_NUM),
        ("DCP", mem.dcp, None, STYLE_INPUT_NUM),
        ("PCP", mem.pcp, None, STYLE_INPUT_NUM),
        ("Local-TP (o/q_proj)", mem.local_tp, None, STYLE_INPUT_NUM),
        ("Shared-TP", mem.shared_tp, None, STYLE_INPUT_NUM),
        ("Total Cluster Dies", mem.total_dies, "=T8*T9*T11*T13", STYLE_INT),
        ("Sequence Length (tok)", mem.seq_len, None, STYLE_INPUT_NUM),
        ("Concurrency (per DP)", mem.concurrency, None, STYLE_INPUT_NUM),
        ("Cache Block Size", mem.block_size, None, STYLE_INPUT_NUM),
        ("Spec / MTP Tokens", mem.spec_tokens, None, STYLE_INPUT_NUM),
        ("Local Cache Tokens / Rank", mem.local_cache_tokens, "=ROUNDUP(T17/(T19*T12),0)*T19", STYLE_INT),
        ("Die HBM Capacity (GiB)", mem.die_hbm_capacity_bytes / float(GIB_BYTES), None, STYLE_FLOAT2),
        ("GPU Memory Utilization", mem.gpu_memory_utilization, None, STYLE_PCT),
        ("Usable Die HBM Budget (GiB)", mem.usable_hbm_bytes / float(GIB_BYTES), "=T22*T23", STYLE_TOTAL_NUM),
        ("Quant Policy (QuaRot First)", mem.weight_quant_display or mem.weight_quant, None, STYLE_TEXT_WRAP),
    )
    for offset, (label, val, formula, style) in enumerate(controls):
        r = MEMORY_TABLE_HEADER_ROW + offset
        ws.set_cell(r, c_lbl, label, style=STYLE_TEXT)
        ws.set_cell(r, c_val, val, formula=formula, style=style)

    ws.add_data_validation("T3", formula1='"' + ",".join(SUPPORTED_WEIGHT_QUANTS) + '"')
    ws.add_data_validation("T4", formula1='"' + ",".join(SUPPORTED_CACHE_DTYPES) + '"')
    ws.add_data_validation("T5", formula1='"ON,OFF"')
    ws.add_data_validation("T6", formula1='"OFF,ON"')
    ws.add_data_validation("T7", formula1='"OFF,ON"')

    ws.set_cell(2, MEMORY_LOOKUP_START_COL, "Quant Format Storage Rules", style=STYLE_SECTION)
    ws.merge_range(2, MEMORY_LOOKUP_START_COL, 2, MEMORY_COLS)
    _write_headers(ws, MEMORY_LOOKUP_HEADER_ROW, ("Format", "B/Elem", "MX Group", "Aux B/Row"), start_col=MEMORY_LOOKUP_START_COL)
    for idx, fmt in enumerate(MEMORY_QUANT_LOOKUP_ORDER):
        r = MEMORY_LOOKUP_FIRST_ROW + idx
        spec = WEIGHT_QUANT_FORMATS[fmt]
        _write_row(ws, r, (
            (fmt, None, STYLE_TEXT), (float(spec["bytes"]), None, STYLE_FLOAT2),
            (int(spec["group"]), None, STYLE_INT), (int(spec["aux"]), None, STYLE_INT),
        ), start_col=MEMORY_LOOKUP_START_COL)


def _quant_format_formula(quant_rule: str) -> str | None:
    """Return live Excel formula linking Col J (Quant Format) to Control Panel $T$3 & $T$6."""
    if quant_rule == QUANT_RULE_FIXED_BF16:
        return None
    if quant_rule == QUANT_RULE_FIXED_FP32:
        return None
    if quant_rule == QUANT_RULE_PRESET_LINEAR_FP8:
        return f'=IF({MEM_CTL_QUANT_REF}="W4A8","W8A8",IF({MEM_CTL_QUANT_REF}="MXFP4","MXFP8",{MEM_CTL_QUANT_REF}))'
    if quant_rule == QUANT_RULE_CHECKPOINT_IGNORE:
        return f'=IF({MEM_CTL_OVERRIDE_IGN_REF}="ON",IF({MEM_CTL_QUANT_REF}="W4A8","W8A8",IF({MEM_CTL_QUANT_REF}="MXFP4","MXFP8",{MEM_CTL_QUANT_REF})),"BF16")'
    return f"={MEM_CTL_QUANT_REF}"


def _populate_memory_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    mem = bundle.memory_ledger
    _apply_col_widths(ws, MEMORY_COL_WIDTHS)
    ws.freeze_panes(MEMORY_FREEZE_ROWS, 2)
    _write_banner(
        ws, 1, MEMORY_COLS,
        f"Closed-Form Memory / Quant / Cache / Engram-PLE / PP HBM Calculator | {mem.model_name} @ {mem.hardware_sku} | {mem.weight_quant_display or mem.weight_quant}",
        STYLE_TITLE,
    )
    _populate_memory_control_panel(ws, mem)
    ws.set_cell(2, 1, "1. Per-Die Weight + Quant Scale Closed-Form Table (Side-by-Side 原生 BF16 vs 量化后; Mixed W4A8/MXFP4 & QuaRot Toggles)", style=STYLE_SECTION)
    ws.merge_range(2, 1, 2, MEMORY_TABLE_END_COL)
    rep_col_hdr = f"Peak Stage Rep (S{mem.peak_weight_stage})" if mem.pp > 1 else "Stage Rep"
    w_headers = (
        "Group", "Tensor Name", "Global Rows", "Global Cols", "Shard Rule",
        "Local Rows", "Local Cols", rep_col_hdr, "Quant Rule", "Quant Format",
        "Payload (B)", "MX Scale (B)", "Channel Aux (B)",
        "原生 BF16 单卡 (GiB)", "当前量化后单卡 (GiB, 含 Scale)", "Residency & Quant Policy",
    )
    _write_headers(ws, MEMORY_TABLE_HEADER_ROW, w_headers)

    lookup_ref = f"$U${MEMORY_LOOKUP_FIRST_ROW}:$X${MEMORY_LOOKUP_LAST_ROW}"
    w_start = MEMORY_TABLE_HEADER_ROW + 1
    group_spans: list[tuple[str, int, int]] = []
    for idx, row in enumerate(mem.weight_rows):
        r = w_start + idx
        _extend_span(group_spans, row.group, r, r)
        if row.is_engram_or_ple:
            res_text = "Host DRAM (Offload)" if mem.engram_host_offload else "NPU HBM (Engram/PLE)"
            res_formula: str | None = f'=IF({MEM_CTL_OFFLOAD_REF}="ON","Host DRAM (Offload)","NPU HBM (Engram/PLE)")'
        elif row.is_vision:
            res_text = "NPU HBM (Vision ViT)" if mem.include_vision_tower else "Excluded Vision (Text-Only)"
            res_formula = f'=IF({MEM_CTL_VIT_REF}="ON","NPU HBM (Vision ViT)","Excluded Vision (Text-Only)")'
        else:
            res_text = f"NPU HBM ({row.quant_note})" if row.quant_note else "NPU HBM"
            res_formula = None
        q_formula = _quant_format_formula(row.quant_rule)
        native_gib = float(row.native_bf16_bytes) / float(GIB_BYTES)
        _write_row(ws, r, (
            (row.name, None, STYLE_TEXT), (row.global_rows, None, STYLE_INT),
            (row.global_cols, None, STYLE_INT), (row.shard_rule, None, STYLE_TEXT),
            (row.local_rows, None, STYLE_INT), (row.local_cols, None, STYLE_INT),
            (row.peak_stage_repeats, None, STYLE_INT),
            (row.quant_rule, None, STYLE_TEXT),
            (row.quant_format, q_formula, STYLE_INPUT_TEXT if q_formula else STYLE_TEXT),
            (row.payload_bytes, f"=F{r}*ROUNDUP(G{r}*VLOOKUP(J{r},{lookup_ref},2,FALSE),0)*H{r}", STYLE_INT),
            (row.scale_bytes, f"=IF(VLOOKUP(J{r},{lookup_ref},3,FALSE)>0,F{r}*ROUNDUP(G{r}/VLOOKUP(J{r},{lookup_ref},3,FALSE),0)*H{r},0)", STYLE_INT),
            (row.aux_bytes, f"=F{r}*VLOOKUP(J{r},{lookup_ref},4,FALSE)*H{r}", STYLE_INT),
            (native_gib, f"=F{r}*G{r}*{BF16_BYTES}*H{r}/{GIB_BYTES}", STYLE_FLOAT3),
            (float(row.per_die_bytes) / float(GIB_BYTES), f"=(K{r}+L{r}+M{r})/{GIB_BYTES}", STYLE_FLOAT3),
            (res_text, res_formula, STYLE_OVERLAP if row.is_engram_or_ple or row.is_vision else STYLE_TEXT),
        ), start_col=2)
    for grp, r1, r2 in group_spans:
        ws.set_cell(r1, 1, grp, style=STYLE_MERGED)
        ws.merge_range(r1, 1, r2, 1)

    w_end = w_start + len(mem.weight_rows) - 1
    w_tot_hbm = w_end + 1
    w_tot_host = w_end + 2
    w_tot_vit = w_end + 3

    def _sumifs_row(
        r_idx: int, label: str, crit: str, pred_fn: Any, note_txt: str,
        t_st: int, n_st: int,
    ) -> None:
        ws.set_cell(r_idx, 1, label, style=t_st)
        ws.merge_range(r_idx, 1, r_idx, MEMORY_WEIGHT_MERGE_END_COL)
        matched = [x for x in mem.weight_rows if pred_fn(x)]
        vals = (
            sum(x.payload_bytes for x in matched),
            sum(x.scale_bytes for x in matched),
            sum(x.aux_bytes for x in matched),
            sum(x.native_bf16_bytes for x in matched) / float(GIB_BYTES),
            sum(x.per_die_bytes for x in matched) / float(GIB_BYTES),
        )
        for offset, tot_v in enumerate(vals):
            col = MEMORY_WEIGHT_SUM_START_COL + offset
            c_let = column_letter(col)
            f_str = f'=SUMIFS({c_let}{w_start}:{c_let}{w_end},$P${w_start}:$P${w_end},"{crit}")' if w_end >= w_start else "=0"
            ws.set_cell(r_idx, col, tot_v, formula=f_str, style=n_st)
        ws.set_cell(r_idx, MEMORY_TABLE_END_COL, note_txt, style=t_st)

    _sumifs_row(
        w_tot_hbm, "NPU HBM Weights Total (GiB, 仅计入 NPU HBM 驻留张量)", "NPU HBM*",
        lambda x: (not (x.is_engram_or_ple and mem.engram_host_offload)) and (not (x.is_vision and not mem.include_vision_tower)),
        f"压缩比 {mem.weight_compression_ratio:.2f}x (vs 原生 BF16)", STYLE_TOTAL_TEXT, STYLE_TOTAL_NUM,
    )
    _sumifs_row(
        w_tot_host, "Host DRAM Offload Total (GiB, Engram/PLE 卸载至主机内存)", "Host DRAM*",
        lambda x: x.is_engram_or_ple and mem.engram_host_offload,
        "不计入 NPU HBM 预算", STYLE_SUBTOTAL_TEXT, STYLE_SUBTOTAL_NUM,
    )
    _sumifs_row(
        w_tot_vit, "Excluded Vision Tower Total (GiB, 纯文本默认不加载 ViT)", "Excluded*",
        lambda x: x.is_vision and not mem.include_vision_tower,
        "切 T7=ON 可计入多模态显存", STYLE_SUBTOTAL_TEXT, STYLE_SUBTOTAL_NUM,
    )

    cache_end = _populate_cache_block(ws, w_tot_vit + 2, mem)
    _populate_stage_budget_block(ws, cache_end + 2, mem, cache_tot_row=cache_end)


def _populate_cache_block(ws: Worksheet, start_row: int, mem: MemoryCalculatorLedger) -> int:
    ws.set_cell(start_row, 1, "2. KV / Indexer / KeyPool / Recurrent State Cache Breakdown", style=STYLE_SECTION)
    ws.merge_range(start_row, 1, start_row, MEMORY_TABLE_END_COL)
    hdr_row = start_row + 1
    headers = (
        "Component", "Cache Kind", "Max Stage Layers", "Bytes/Tok/Layer",
        "Per-Request (MiB)", "Batch Total (GiB)", "Scaling Basis",
    )
    _write_headers(ws, hdr_row, headers)
    ws.merge_range(hdr_row, MEMORY_CACHE_BASIS_START_COL, hdr_row, MEMORY_TABLE_END_COL)

    c_start = hdr_row + 1
    # Token-scaled rates are stored at the ledger dtype; the live T4 selector rescales them by the
    # canonical storage-width ratio (same widths as ``memory_ledger._cache_dtype_override_ratio``).
    base_width = float(BF16_BYTES) if mem.cache_dtype == "BF16" else C8_CACHE_FACTOR
    live_width = f'IF({MEM_CTL_CACHE_REF}="BF16",{BF16_BYTES},{C8_CACHE_FACTOR!r})'
    for offset, row in enumerate(mem.cache_rows):
        r = c_start + offset
        if row.scales_with_tokens and row.max_stage_layers > 0 and row.bytes_per_token_per_layer > 0.0:
            rate_formula: str | None = f"={row.bytes_per_token_per_layer!r}*{live_width}/{base_width!r}"
            req_formula: str | None = f"=C{r}*D{r}*{MEM_CTL_LOCAL_TOK_REF}/{MIB_BYTES}"
            basis = f"Scales with Local Cache Tokens (BF16/C8/FP8={mem.cache_dtype}, ratio={C8_CACHE_FACTOR:.4f}x for C8)"
        else:
            rate_formula = None
            req_formula = None
            basis = "Fixed per-request state buffer (non-token-scaled)"
        _write_row(ws, r, (
            (row.component, None, STYLE_TEXT), (row.cache_kind, None, STYLE_TEXT),
            (row.max_stage_layers, None, STYLE_INT), (row.bytes_per_token_per_layer, rate_formula, STYLE_FLOAT2),
            (row.per_request_bytes / float(MIB_BYTES), req_formula, STYLE_FLOAT3),
            (row.batch_total_bytes / float(GIB_BYTES), f"=E{r}*{MEM_CTL_CONC_REF}/{MIB_PER_GIB}", STYLE_FLOAT3),
            (basis, None, STYLE_TEXT),
        ))
        ws.merge_range(r, MEMORY_CACHE_BASIS_START_COL, r, MEMORY_TABLE_END_COL)

    c_end = c_start + len(mem.cache_rows) - 1
    c_tot = c_end + 1
    ws.set_cell(c_tot, 1, "Cache Total (Max Stage)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(c_tot, 1, c_tot, MEMORY_CACHE_MERGE_END_COL)
    ws.set_cell(
        c_tot, MEMORY_CACHE_REQ_COL, sum(x.per_request_bytes for x in mem.cache_rows) / float(MIB_BYTES),
        formula=f"=SUM(E{c_start}:E{c_end})" if c_end >= c_start else "=0", style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BATCH_COL, sum(x.batch_total_bytes for x in mem.cache_rows) / float(GIB_BYTES),
        formula=f"=SUM(F{c_start}:F{c_end})" if c_end >= c_start else "=0", style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BASIS_START_COL,
        f"Local tokens={mem.local_cache_tokens}, Concurrency={mem.concurrency}", style=STYLE_TOTAL_TEXT,
    )
    ws.merge_range(c_tot, MEMORY_CACHE_BASIS_START_COL, c_tot, MEMORY_TABLE_END_COL)
    return c_tot


def _populate_stage_budget_block(
    ws: Worksheet, start_row: int, mem: MemoryCalculatorLedger, *, cache_tot_row: int = 0,
) -> int:
    ws.set_cell(start_row, 1, "3. Per-PP-Stage HBM vs Host Offload Budget (DP/TP/EP/PP/DCP/PCP/Local-TP)", style=STYLE_SECTION)
    ws.merge_range(start_row, 1, start_row, MEMORY_TABLE_END_COL)
    hdr_row = start_row + 1
    headers = (
        "PP Stage", "Layers", "Weights Payload (GiB)", "Quant Scale/Aux (GiB)",
        "Engram/PLE HBM (GiB)", "Engram/PLE Host (GiB)", "Cache (GiB)",
        "Act & Workspace (GiB)", "Runtime Reserve (GiB)", "Total HBM (GiB)",
        "Usable Budget (GiB)", "Headroom (GiB)", "Fit Status", "Raw Engram/PLE (GiB)",
    )
    _write_headers(ws, hdr_row, headers)

    sum_per_req = sum(x.per_request_bytes for x in mem.cache_rows)
    s_start = hdr_row + 1
    for offset, st in enumerate(mem.stage_budgets):
        r = s_start + offset
        raw_engram_gib = (st.engram_ple_hbm_bytes + st.engram_ple_host_bytes) / float(GIB_BYTES)
        st_per_req = (
            float(st.per_request_cache_bytes)
            if float(st.per_request_cache_bytes) > 0.0
            else (float(st.cache_bytes) / float(max(1, mem.concurrency)))
        )
        if cache_tot_row > 0 and sum_per_req > 0.0:
            # Each component carries its own stage share, so hybrid PP stages roll up exactly.
            first = cache_tot_row - len(mem.cache_rows)
            terms: list[str] = []
            for i, crow in enumerate(mem.cache_rows, first):
                if crow.per_request_bytes <= 0.0:
                    continue
                stage_vals = crow.per_stage_bytes
                share = (
                    float(stage_vals[st.stage]) / float(crow.per_request_bytes)
                    if st.stage < len(stage_vals) else 1.0
                )
                if abs(share - 1.0) <= MIN_DENOM_EPS:
                    terms.append(f"E{i}")
                elif share > 0.0:
                    terms.append(f"E{i}*{share!r}")
            cache_f = f"=({'+'.join(terms) or '0'})*{MEM_CTL_CONC_REF}/{MIB_PER_GIB}"
        else:
            cache_f = f"={st_per_req / float(GIB_BYTES)!r}*{MEM_CTL_CONC_REF}"
        _write_row(ws, r, (
            (f"Stage {st.stage}", None, STYLE_MERGED), (st.owned_layers, None, STYLE_INT),
            (st.weight_payload_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.weight_scale_aux_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.engram_ple_hbm_bytes / float(GIB_BYTES), f'=IF({MEM_CTL_OFFLOAD_REF}="ON",0,N{r})', STYLE_FLOAT3),
            (st.engram_ple_host_bytes / float(GIB_BYTES), f'=IF({MEM_CTL_OFFLOAD_REF}="ON",N{r},0)', STYLE_FLOAT3),
            (st.cache_bytes / float(GIB_BYTES), cache_f, STYLE_FLOAT3),
            (st.workspace_peak_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.reserve_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.total_hbm_bytes / float(GIB_BYTES), f"=SUM(C{r}:E{r},G{r}:I{r})", STYLE_FLOAT3),
            (st.usable_hbm_budget_bytes / float(GIB_BYTES), f"={MEM_CTL_USABLE_HBM_REF}", STYLE_FLOAT3),
            (st.headroom_bytes / float(GIB_BYTES), f"=K{r}-J{r}", STYLE_FLOAT3),
            ("PASS" if st.fits_hbm else "OOM", f'=IF(J{r}<=K{r},"PASS","OOM")', STYLE_ALERT_OK if st.fits_hbm else STYLE_ALERT_HIGH),
            (raw_engram_gib, None, STYLE_FLOAT3),
        ))

    s_end = s_start + len(mem.stage_budgets) - 1
    max_row = s_end + 1
    ws.set_cell(max_row, 1, "Max Stage (Bottleneck)", style=STYLE_TOTAL_TEXT)
    ws.set_cell(
        max_row, 2, sum(s.owned_layers for s in mem.stage_budgets),
        formula=f"=SUM(B{s_start}:B{s_end})" if s_end >= s_start else "=0", style=STYLE_TOTAL_NUM,
    )
    agg_series = (
        [s.weight_payload_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.weight_scale_aux_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.engram_ple_hbm_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.engram_ple_host_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.cache_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.workspace_peak_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.reserve_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.total_hbm_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.usable_hbm_budget_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.headroom_bytes / GIB_BYTES for s in mem.stage_budgets],
    )
    for offset, vals in enumerate(agg_series):
        col = MEMORY_STAGE_AGG_START_COL + offset
        c_let = column_letter(col)
        is_headroom = col == MEMORY_STAGE_HEADROOM_COL
        agg = "MIN" if is_headroom else "MAX"
        val = (min(vals) if is_headroom else max(vals)) if vals else 0.0
        ws.set_cell(max_row, col, val, formula=f"={agg}({c_let}{s_start}:{c_let}{s_end})" if s_end >= s_start else "=0", style=STYLE_TOTAL_NUM)
    ws.set_cell(max_row, MEMORY_STAGE_STATUS_COL, "PASS" if mem.fits_hbm else "OOM", formula=f'=IF(J{max_row}<=K{max_row},"PASS","OOM")', style=STYLE_ALERT_OK if mem.fits_hbm else STYLE_ALERT_HIGH)
    raw_engram_vals = [(s.engram_ple_hbm_bytes + s.engram_ple_host_bytes) / GIB_BYTES for s in mem.stage_budgets]
    ws.set_cell(max_row, MEMORY_STAGE_RAW_ENGRAM_COL, max(raw_engram_vals, default=0.0), formula=f"=MAX(N{s_start}:N{s_end})" if s_end >= s_start else "=0", style=STYLE_TOTAL_NUM)
    return max_row


def _write_pending_kernel_card(
    ws: Worksheet,
    card_start: int,
    idx_row: int,
    item: PendingKernelItem,
    target_bundle: ExportBundle,
    agent_designs: Mapping[str, Any] | None,
) -> None:
    """Render one structured 10-row 3-track per-kernel detail card on Sheet 4."""
    is_new = item.category == PENDING_KIND_TO_IMPLEMENT
    cat_style = STYLE_NEW_KERNEL if is_new else STYLE_OPT_BADGE
    ov = _lookup_agent_design_bundle(
        agent_designs,
        card_id=item.card_id,
        kernel_key=item.kernel_key,
        target_kernel=item.target_kernel,
        logical_role=item.logical_role,
    )

    bind_badge = format_binding_badge(item.binding_state)
    ws.set_row_height(card_start, HEADER_ROW_HEIGHT_PT)
    _write_banner(
        ws, card_start, PENDING_COLS,
        (
            f"Card #{item.priority} [{item.card_id}] | [{item.category}] {item.target_kernel} — {item.logical_role} "
            f"| Δ={item.expected_gain_ms:.3f} ms / Δ摊薄={item.saved_us_per_tok_die:.2f} us/(tok·die) {bind_badge}"
        ),
        STYLE_CARD_HDR,
    )

    r_st = card_start + PENDING_CARD_ROW_STACK
    ws.set_row_height(r_st, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_st, 1, "分类 & 优化状态", style=STYLE_MERGED)
    ws.merge_range(r_st, 1, r_st, PENDING_CARD_LBL_END)
    ws.set_cell(r_st, PENDING_CARD_LEFT_START, f"{item.category} | {item.status} | {bind_badge}", style=cat_style)
    ws.merge_range(r_st, PENDING_CARD_LEFT_START, r_st, PENDING_CARD_LEFT_END)
    ws.set_cell(r_st, PENDING_CARD_MID_COL, "建议技术栈 (Backend)", style=STYLE_MERGED)
    ws.set_cell(r_st, PENDING_CARD_RIGHT_START, f"{item.suggested_backend} — {item.target_kernel}", style=STYLE_TEXT_WRAP)
    ws.merge_range(r_st, PENDING_CARD_RIGHT_START, r_st, PENDING_COLS)

    r_ch = card_start + PENDING_CARD_ROW_CHAIN
    ws.set_row_height(r_ch, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_ch, 1, "替代的原算子链 / 模块", style=STYLE_MERGED)
    ws.merge_range(r_ch, 1, r_ch, PENDING_CARD_LBL_END)
    ws.set_cell(r_ch, PENDING_CARD_LEFT_START, item.replaces_chain, style=STYLE_TEXT_WRAP)
    ws.merge_range(r_ch, PENDING_CARD_LEFT_START, r_ch, PENDING_CARD_LEFT_END)
    ws.set_cell(r_ch, PENDING_CARD_MID_COL, "源码出处 (Source Ref)", style=STYLE_MERGED)
    ws.set_cell(r_ch, PENDING_CARD_RIGHT_START, item.source_refs or "-", style=STYLE_MUTED_TEXT)
    ws.merge_range(r_ch, PENDING_CARD_RIGHT_START, r_ch, PENDING_COLS)

    r_fr = card_start + PENDING_CARD_ROW_FREQ
    ws.set_row_height(r_fr, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_fr, 1, "全网频次 (Archetypes)", style=STYLE_MERGED)
    ws.merge_range(r_fr, 1, r_fr, PENDING_CARD_LBL_END)
    ws.set_cell(r_fr, PENDING_CARD_LEFT_START, f"{item.archetype_summary} ({item.total_calls_per_step} calls/step)", style=STYLE_TEXT_WRAP)
    ws.merge_range(r_fr, PENDING_CARD_LEFT_START, r_fr, PENDING_CARD_LEFT_END)
    ws.set_cell(r_fr, PENDING_CARD_MID_COL, "预期全网收益 Δ (ms)", style=STYLE_MERGED)
    ws.set_cell(r_fr, PENDING_CARD_RIGHT_START, item.expected_gain_ms, formula=f"=G{idx_row}", style=STYLE_KPI_HIGHLIGHT)
    ws.set_cell(
        r_fr, PENDING_CARD_NOTE_START,
        (
            f"Before={item.before_model_ms:.3f}ms -> After={item.after_model_ms:.3f}ms | "
            f"Δ摊薄={item.saved_us_per_tok_die:.2f} us/(tok·die) | L*={item.lstar_model_ms:.3f}ms "
            f"(达成率 {item.before_attainment_pct:.1f}% -> {item.target_attainment_pct:.1f}%)"
        ),
        style=STYLE_TEXT_WRAP,
    )
    ws.merge_range(r_fr, PENDING_CARD_NOTE_START, r_fr, PENDING_COLS)

    kernel_design_txt = ov["kernel_design"] or item.kernel_action or item.default_design_slot
    if not ov["kernel_design"] and (item.math_formula or item.sram_residency_summary):
        extra_spec = item.sram_residency_summary or f"Formula: {item.math_formula}"
        if extra_spec not in kernel_design_txt:
            kernel_design_txt = f"{kernel_design_txt} | {extra_spec}"
    full_width_rows = (
        (PENDING_CARD_ROW_SHAPE, WRAP_ROW_HEIGHT_PT, "Rank-Local 形状/精度/步长契约", f"{item.tensor_contract} | Layout/Stride: Rank-Local Contiguous ND/NZ | {target_bundle.hardware_die_summary}", STYLE_TEXT_WRAP),
        (PENDING_CARD_ROW_BOUND, WRAP_ROW_HEIGHT_PT, "瓶颈迁移 & L* 目标", item.microarch_target, STYLE_TEXT_WRAP),
        (PENDING_CARD_ROW_CONFIG, CARD_ACCEPT_ROW_HEIGHT_PT, "① 已有特效开关与通信展开配置 (Config & Env)", ov["config_howto"] or item.config_howto, STYLE_OVERLAP),
        (PENDING_CARD_ROW_KERNEL, CARD_DESIGN_ROW_HEIGHT_PT, "② 算子新写 / 调优方案 (Kernel & Tiling)", kernel_design_txt, STYLE_INPUT_TEXT_WRAP),
        (PENDING_CARD_ROW_FRAMEWORK, CARD_ACCEPT_ROW_HEIGHT_PT, "③ 推理框架改造要点 (vllm-ascend)", ov["framework_work"] or item.framework_action, STYLE_OPT_BADGE),
        (PENDING_CARD_ROW_ACCEPT, CARD_ACCEPT_ROW_HEIGHT_PT, "风险与验收标准 (Risks & Acceptance)", ov["acceptance"] or item.default_acceptance_slot, STYLE_TEXT_WRAP),
    )
    for offset, height, label, text, st in full_width_rows:
        r = card_start + offset
        ws.set_row_height(r, height)
        ws.set_cell(r, 1, label, style=STYLE_MERGED)
        ws.merge_range(r, 1, r, PENDING_CARD_LBL_END)
        ws.set_cell(r, PENDING_CARD_LEFT_START, text, style=st)
        ws.merge_range(r, PENDING_CARD_LEFT_START, r, PENDING_COLS)


def _sheet2_sum_formula(col_let: str, spans: Sequence[tuple[int, int]]) -> str | None:
    """Build a compact live Excel ``SUM`` formula referencing Sheet 2 cell ranges."""
    if not spans:
        return None
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


def _compute_uncovered_exposure(
    aligned_rows: Sequence[AlignedComparisonRow],
    uncovered_indices: Sequence[int],
    *,
    is_comparison: bool,
) -> tuple[float, float, int]:
    bef_ms = aft_ms = 0.0
    total_calls = 0
    for idx in uncovered_indices:
        row = aligned_rows[idx]
        b_rep = row.effective_before_repeat
        a_rep = row.effective_after_repeat
        total_calls += max(a_rep, b_rep, 1)
        if is_comparison:
            bef_ms += sum(0.0 if b.is_overlapped else (b.exposed_dur_us * b_rep / US_PER_MS) for b in row.before_ops)
            aft_ms += sum(0.0 if a.is_overlapped else (a.exposed_dur_us * a_rep / US_PER_MS) for a in row.after_ops)
        else:
            bef_ms += sum(0.0 if a.is_overlapped else (a.exposed_dur_us * a_rep / US_PER_MS) for a in row.after_ops)
            aft_ms += sum(0.0 if a.is_overlapped else (a.l_star_lean_us * a_rep / US_PER_MS) for a in row.after_ops)
    return bef_ms, aft_ms, total_calls


def _populate_pending_kernel_details_sheet(
    ws: Worksheet,
    pending_items: Sequence[PendingKernelItem],
    target_bundle: ExportBundle,
    agent_designs: Mapping[str, Any] | None,
    *,
    is_comparison: bool = True,
    sheet2_row_spans: Mapping[int, tuple[int, int]] | None = None,
    aligned_rows: Sequence[AlignedComparisonRow] = (),
) -> None:
    """Populate Sheet 4 with 100%-conserved Summary Index + 3-Track Detail Cards."""
    _apply_col_widths(ws, PENDING_COL_WIDTHS)
    ws.freeze_panes(PENDING_HEADER_ROW, 2)
    _write_banner(
        ws, 1, PENDING_COLS,
        f"待新开发 & 重点待优化算子专题明细 (Pending Kernel Details) | {target_bundle.model_name} @ {target_bundle.hardware_sku} ({target_bundle.hardware_die_summary})",
        STYLE_TITLE,
    )
    _write_banner(
        ws, 2, PENDING_COLS,
        "1. 总览索引表 (Top Summary Index — 收益合计与 Sheet 1 / Sheet 2 100% 守恒；支持 --kernel-designs-json 按 card_id 精细覆盖)",
        STYLE_SECTION,
    )
    headers = (
        "优先级", "分类 (待新开发 vs 重点待优化)", "目标算子 & 建议技术栈",
        "开发 / 优化状态", "替代原算子链 / 所在模块", "全网频次 (Archetypes & Calls)",
        "预期全网收益 Δ (ms)", "Before -> After (L* ms)", "Before -> Target Bound", "详情卡片定位 (card_id)",
        "摊薄净省 Δ (us/tok·die)", "绑定状态 (Binding State)",
        "Before 暴露 (ms)" if is_comparison else "当前暴露 (ms)",
        "After 暴露 (ms)" if is_comparison else "L* 下界 (ms)",
    )
    _write_headers(ws, PENDING_HEADER_ROW, headers)

    spans_map = sheet2_row_spans or {}
    covered_indices = {idx for item in pending_items for idx in item.source_row_indices}
    uncovered_indices = (
        [idx for idx in range(len(aligned_rows)) if idx not in covered_indices and idx in spans_map]
        if (is_comparison and aligned_rows) else []
    )
    if not pending_items and not uncovered_indices:
        ws.set_cell(
            PENDING_DATA_START_ROW, 1,
            "No pending new or gap-dominated kernels detected — all kernels are near L* physical lower bound.",
            style=STYLE_OVERLAP,
        )
        ws.merge_range(PENDING_DATA_START_ROW, 1, PENDING_DATA_START_ROW, PENDING_COLS)
        return

    has_unchanged_row = bool(uncovered_indices)
    r_end = PENDING_DATA_START_ROW + len(pending_items) + (1 if has_unchanged_row else 0) - 1
    r_tot = r_end + 1
    cards_banner_row = r_tot + 2
    cards_start_row = r_tot + PENDING_CARD_SPACER_GAP - 1
    n_new = sum(1 for x in pending_items if x.category == PENDING_KIND_TO_IMPLEMENT)
    n_opt = len(pending_items) - n_new
    col_m_src = column_letter(CMP_COL_BEF_EXP_MS if is_comparison else STD_COL_EXP_MS)
    col_n_src = column_letter(CMP_COL_AFT_EXP_MS if is_comparison else STD_COL_LSTAR_MS)

    for idx, item in enumerate(pending_items):
        r = PENDING_DATA_START_ROW + idx
        card_row = cards_start_row + idx * PENDING_CARD_STRIDE
        is_new = item.category == PENDING_KIND_TO_IMPLEMENT
        ws.set_row_height(r, BACKLOG_ROW_HEIGHT_PT)
        m_val = item.before_model_ms if is_comparison else item.after_model_ms
        n_val = item.after_model_ms if is_comparison else item.lstar_model_ms
        item_spans = [spans_map[i] for i in item.source_row_indices if i in spans_map]
        m_formula = _sheet2_sum_formula(col_m_src, item_spans)
        n_formula = _sheet2_sum_formula(col_n_src, item_spans)
        gain_ms_f = f"=M{r}-N{r}" if is_comparison else f"=MAX(0,M{r}-N{r})"
        gain_upt_f = (
            f"=IF({SUM_BEF_NTOK_REF}>0,M{r}*1000/{SUM_BEF_NTOK_REF},0)-IF({SUM_AFT_NTOK_REF}>0,N{r}*1000/{SUM_AFT_NTOK_REF},0)"
            if is_comparison else f"=IF({SUM_AFT_NTOK_REF}>0,G{r}*1000/{SUM_AFT_NTOK_REF},0)"
        )
        _write_row(ws, r, (
            (f"P{item.priority}", None, STYLE_MERGED),
            (item.category, None, STYLE_NEW_KERNEL if is_new else STYLE_OPT_BADGE),
            (item.target_kernel, None, STYLE_TEXT_WRAP),
            (item.status, None, STYLE_NEW_KERNEL if is_new else STYLE_TEXT_WRAP),
            (item.replaces_chain, None, STYLE_TEXT_WRAP),
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

    unch_bef = unch_aft = 0.0
    if has_unchanged_row:
        r_unch = PENDING_DATA_START_ROW + len(pending_items)
        ws.set_row_height(r_unch, BACKLOG_ROW_HEIGHT_PT)
        unch_bef, unch_aft, unch_calls = _compute_uncovered_exposure(
            aligned_rows, uncovered_indices, is_comparison=is_comparison,
        )
        unch_spans = [spans_map[i] for i in uncovered_indices if i in spans_map]
        _write_row(ws, r_unch, (
            ("—", None, STYLE_MERGED),
            ("无变化对账 (Unchanged)", None, STYLE_OVERLAP),
            ("Unchanged Operators Exposure Reconciliation", None, STYLE_TEXT_WRAP),
            ("Before == After (0 变化，补齐全网暴露守恒)", None, STYLE_TEXT_WRAP),
            (f"{len(uncovered_indices)} unchanged aligned rows", None, STYLE_TEXT_WRAP),
            (f"Reconciliation ({unch_calls} calls/step)", None, STYLE_TEXT_WRAP),
            (unch_bef - unch_aft, f"=M{r_unch}-N{r_unch}", STYLE_FLOAT3),
            (f"{unch_bef:.3f} -> {unch_aft:.3f}", None, STYLE_TEXT_WRAP),
            ("Unchanged", None, STYLE_TEXT_WRAP),
            ("—", None, STYLE_MERGED),
            (
                0.0,
                f"=IF({SUM_BEF_NTOK_REF}>0,M{r_unch}*1000/{SUM_BEF_NTOK_REF},0)-IF({SUM_AFT_NTOK_REF}>0,N{r_unch}*1000/{SUM_AFT_NTOK_REF},0)",
                STYLE_FLOAT2,
            ),
            (format_binding_badge(target_bundle.binding_state), None, STYLE_TEXT_WRAP),
            (unch_bef, _sheet2_sum_formula(col_m_src, unch_spans), STYLE_FLOAT3),
            (unch_aft, _sheet2_sum_formula(col_n_src, unch_spans), STYLE_FLOAT3),
        ))

    ws.set_cell(r_tot, 1, f"TOTAL ({n_new} 待新开发/扩展 + {n_opt} 重点待优化 = {len(pending_items)} 项)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(r_tot, 1, r_tot, PENDING_LABEL_MERGE_END)
    ws.set_cell(
        r_tot, PENDING_COL_GAIN,
        sum(x.expected_gain_ms for x in pending_items) + (unch_bef - unch_aft),
        formula=f"=SUM(G{PENDING_DATA_START_ROW}:G{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )
    ws.set_cell(r_tot, PENDING_COL_GAIN + 1, "已预填 3-Track 落地指导（Config/Kernel/Framework）；全网暴露与 Sheet 1/2 100% 守恒", style=STYLE_TOTAL_TEXT)
    ws.merge_range(r_tot, PENDING_COL_GAIN + 1, r_tot, PENDING_COL_GAIN_UPT - 1)
    ws.set_cell(
        r_tot, PENDING_COL_GAIN_UPT,
        sum(x.saved_us_per_tok_die for x in pending_items),
        formula=f"=SUM(K{PENDING_DATA_START_ROW}:K{r_end})",
        style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(r_tot, PENDING_COL_BINDING, format_binding_badge(target_bundle.binding_state), style=STYLE_TOTAL_TEXT)
    ws.set_cell(
        r_tot, PENDING_COL_BEF_MS,
        sum(x.before_model_ms if is_comparison else x.after_model_ms for x in pending_items) + unch_bef,
        formula=f"=SUM(M{PENDING_DATA_START_ROW}:M{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )
    ws.set_cell(
        r_tot, PENDING_COL_AFT_MS,
        sum(x.after_model_ms if is_comparison else x.lstar_model_ms for x in pending_items) + unch_aft,
        formula=f"=SUM(N{PENDING_DATA_START_ROW}:N{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )

    if pending_items:
        _write_banner(
            ws, cards_banner_row, PENDING_COLS,
            "2. 逐算子结构化 3-Track 详情卡片 (Config & Env / Kernel & Tiling / vllm-ascend Framework / Acceptance)",
            STYLE_SECTION,
        )
        for idx, item in enumerate(pending_items):
            idx_row = PENDING_DATA_START_ROW + idx
            card_row = cards_start_row + idx * PENDING_CARD_STRIDE
            _write_pending_kernel_card(ws, card_row, idx_row, item, target_bundle, agent_designs)


def _populate_measured_calibration_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    """Conditional sheet emitted ONLY when ``observations_path`` is supplied."""
    _apply_col_widths(ws, MEAS_COL_WIDTHS)
    ws.freeze_panes(2, 2)
    rec = bundle.reconciliation
    src_lbl = rec.source_label if rec is not None else "msprof"
    _write_banner(
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
    _write_headers(ws, 2, headers)

    row_cursor = MEAS_DATA_START_ROW
    for op in bundle.operators:
        k_count = max(1, len(op.measured_kernels))
        r_start = row_cursor
        r_end = row_cursor + k_count - 1
        _write_row(ws, r_start, (
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
            _write_row(ws, r_start, (
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


def _resolve_xlsx_bundles(
    source: ExportBundle | CaseInput | None,
    compare_source: ExportBundle | CaseInput | None,
    *,
    request: StepRequest | None, step_eval: StepEvaluation | None, replay: ReplayOptions,
    observations_path: str | Path | None, weight_quant: str | None,
    cache_dtype: str | None, engram_host_offload: bool | None,
    include_vision_tower: bool | None, apply_preset_to_ignored: bool,
    model: str | None, hardware: str | None, topology_preset: str | None,
    dp: int, tp: int, dcp: int, ep: int, pp: int, pcp: int,
    prompt_tokens: int, output_tokens: int, concurrency: int,
    mtp_proposal_tokens: int, mtp_acceptance_length: float,
    compare_track: str | None, target_track: str | None,
) -> tuple[ExportBundle, ExportBundle | None]:
    # Freshly composed Cases already carry the resolved QuantSpec for ``weight_quant`` (which may
    # be a catalog preset like ``auto_va``); only caller-supplied Cases keep the storage override.
    compose_before = source is None
    compose_after = compose_before and compare_source is None and bool(target_track)
    if source is None:
        if not model or not hardware:
            raise ValueError("export_comparison_xlsx_requires_source_or_model_and_hardware")
        base_track = compare_track or "merged"
        raw_bef, auto_req = compose_export_case(
            model, hardware, topology_preset=topology_preset,
            dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            concurrency=concurrency, mtp_proposal_tokens=mtp_proposal_tokens,
            mtp_acceptance_length=mtp_acceptance_length, feature_track=base_track,
            quant=weight_quant, kv_cache_dtype=cache_dtype,
            engram_cpu_offload=engram_host_offload,
        )
        eff_req = request or auto_req
        source = raw_bef
        if compare_source is None and target_track:
            raw_aft, _ = compose_export_case(
                model, hardware, topology_preset=topology_preset,
                dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
                prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                concurrency=concurrency, mtp_proposal_tokens=mtp_proposal_tokens,
                mtp_acceptance_length=mtp_acceptance_length, feature_track=target_track,
                quant=weight_quant, kv_cache_dtype=cache_dtype,
                engram_cpu_offload=engram_host_offload,
            )
            compare_source = raw_aft
    else:
        eff_req = request

    bundle = (
        source if isinstance(source, ExportBundle)
        else build_export_bundle(
            source, request=eff_req, step_eval=step_eval, replay=replay,
            observations_path=observations_path,
            weight_quant=None if compose_before else weight_quant,
            cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
            include_vision_tower=include_vision_tower,
            apply_preset_to_ignored=apply_preset_to_ignored,
        )
    )
    compare_bundle: ExportBundle | None = None
    if compare_source is not None:
        compare_bundle = (
            compare_source if isinstance(compare_source, ExportBundle)
            else build_export_bundle(
                compare_source,
                request=eff_req or StepRequest(
                    phase=bundle.phase,  # type: ignore[arg-type]
                    tokens=bundle.tokens_per_seq, context=bundle.context_tokens,
                    sequences=bundle.sequences, scope=bundle.scope,  # type: ignore[arg-type]
                ),
                replay=replay, weight_quant=None if compose_after else weight_quant,
                cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
                include_vision_tower=include_vision_tower,
                apply_preset_to_ignored=apply_preset_to_ignored,
            )
        )
    return bundle, compare_bundle


def export_comparison_xlsx(
    source: ExportBundle | CaseInput | None = None,
    output_path: str | Path = "",
    *,
    compare_source: ExportBundle | CaseInput | None = None,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
    include_vision_tower: bool | None = None,
    apply_preset_to_ignored: bool = False,
    agent_kernel_designs: Mapping[str, Any] | None = None,
    agent_brief_path: str | Path | None = None,
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
    compare_track: str | None = None,
    target_track: str | None = None,
) -> XlsxExportSummary:
    """Build and atomically save the human-native Ascend comparison ``.xlsx`` workbook."""
    if not str(output_path):
        raise ValueError("export_comparison_xlsx_requires_output_path")
    bundle, compare_bundle = _resolve_xlsx_bundles(
        source, compare_source,
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
        compare_track=compare_track, target_track=target_track,
    )

    macro_rows = build_macro_module_breakdown(bundle, compare_bundle)
    if compare_bundle is not None:
        aligned_rows = align_before_after_operators(bundle, compare_bundle)
        pending_items = build_pending_kernel_details(aligned_rows, compare_bundle, before_bundle=bundle)
    else:
        aligned_rows = tuple(
            AlignedComparisonRow(
                op.archetype_id, op.archetype_title or op.layer_label, op.repeat_count,
                op.submodule, (op,), (op,), "1->1", "",
            )
            for op in bundle.operators
        )
        pending_items = build_pending_kernel_details(aligned_rows, bundle)

    wb = Workbook()
    ws_summary = wb.create_sheet(SHEET_SUMMARY)
    ws_breakdown = wb.create_sheet(SHEET_BREAKDOWN)
    ws_memory = wb.create_sheet(SHEET_MEMORY)

    _populate_summary_sheet(
        ws_summary, bundle, compare_bundle=compare_bundle,
        macro_rows=macro_rows, pending_items=pending_items,
    )
    if compare_bundle is not None:
        sheet2_spans = _populate_comparison_breakdown_sheet(
            ws_breakdown, bundle, compare_bundle, aligned_rows, agent_kernel_designs,
        )
    else:
        sheet2_spans = _populate_standalone_breakdown_sheet(ws_breakdown, bundle)
    _populate_memory_sheet(ws_memory, compare_bundle or bundle)

    sheet_names = [SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY]
    merge_counts = {
        SHEET_SUMMARY: len(ws_summary.merged_ranges),
        SHEET_BREAKDOWN: len(ws_breakdown.merged_ranges),
        SHEET_MEMORY: len(ws_memory.merged_ranges),
    }

    if compare_bundle is not None or pending_items:
        ws_pending = wb.create_sheet(SHEET_PENDING_KERNELS)
        _populate_pending_kernel_details_sheet(
            ws_pending, pending_items, compare_bundle or bundle,
            agent_designs=agent_kernel_designs, is_comparison=(compare_bundle is not None),
            sheet2_row_spans=sheet2_spans, aligned_rows=aligned_rows,
        )
        sheet_names.append(SHEET_PENDING_KERNELS)
        merge_counts[SHEET_PENDING_KERNELS] = len(ws_pending.merged_ranges)

    if bundle.reconciliation is not None:
        ws_meas = wb.create_sheet(SHEET_MEASURED)
        _populate_measured_calibration_sheet(ws_meas, bundle)
        sheet_names.append(SHEET_MEASURED)
        merge_counts[SHEET_MEASURED] = len(ws_meas.merged_ranges)

    out = Path(output_path)
    size = wb.save(out)
    brief_path_str = ""
    if agent_brief_path is not None:
        brief = build_agent_brief(
            bundle, compare_bundle,
            generated_files={"comparison_xlsx": str(out.resolve()), "agent_brief_json": str(Path(agent_brief_path).resolve())},
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
