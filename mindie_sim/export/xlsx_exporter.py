"""Human-native Ascend optimization OOXML ``.xlsx`` workbook renderer.

[Role] Assemble executive-ready 4-tab (plus conditional 5th calibration tab)
  ECMA-376 ``.xlsx`` workbooks on top of ``ooxml.Workbook``:
  1. ``Summary`` — 3 Clear Executive Blocks:
     (1) Deployment, Parallelism & Concurrency (Model/Layers, SKU/Topo, Total
         Dies, Deployment Mode, DP/TP/EP/PP/DCP/PCP/Local-TP/SP, Prompt/Output,
         SLA, Speculative Strategy & Step Budget, Active/Cluster/Max-HBM
         Concurrency & Weight Footprint / Die),
     (2) Before vs After Simulation KPIs with live Excel formulas for Step
         Latency, Effective TPOT + SLA Verdict, Single-Die TPS, Cluster Total
         TPS, Lean 4 ``L*`` & Physical Limit Attainment %, and 4D Utilization,
     (3) Macro-Module Breakdown & Share % (Before vs After exposed ms, %,
         Saved Δ ms, ``L*``, Dominant Bound + Cross-Stream Overlap reconciliation);
  2. ``Layer_Operator_Breakdown`` — Archetype ×N side-by-side Before-vs-After
     N->1 ``merge_cells``, MSO Stream-1 overlap formulas, Source Refs & Bounds;
  3. ``Memory_Quant_Calculator`` — closed-form per-die Weight/Quant/Cache/PP HBM
     calculator honoring per-row ``QuantSpec.exclusions`` and Engram/PLE offload;
  4. ``Pending_Kernel_Details`` — Top Summary Index Table + Structured Per-Kernel
     Detail Cards covering BOTH ``待新开发 / 扩展`` AND ``重点待优化`` kernels
     with Agent Design Proposal & Tiling/Pipeline slots (``--kernel-designs-json``);
  5. Optional ``Measured_msprof_Calibration`` when ``observations_path`` is set.
[Invariants] Zero third-party OOXML deps, zero heuristic fudge factors, zero
  banned Python recipe class names, strict ECMA-376 worksheet element order,
  and non-overlapping validated ``merge_cells``.
[Timeouts & Locks] Bounded O(Archetypes) in-memory XML serialization; no locks
  or device contexts.
[Side-Effect Policy] Writes only the caller-requested ``.xlsx`` output path via
  atomic same-directory ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindie_sim.prediction.results import CaseInput, ReplayOptions, StepEvaluation, StepRequest

from .bundle import build_export_bundle, compose_export_case
from .comparison import (
    align_before_after_operators,
    build_macro_module_breakdown,
    build_pending_kernel_details,
)
from .models import (
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    FP32_BYTES,
    GIB_BYTES,
    KV_QUANT_GROUP_SIZE,
    MIB_BYTES,
    PENDING_KIND_TO_IMPLEMENT,
    SUPPORTED_CACHE_DTYPES,
    SUPPORTED_WEIGHT_QUANTS,
    US_PER_MS,
    WEIGHT_QUANT_FORMATS,
    AlignedComparisonRow,
    ExportBundle,
    MacroModuleRow,
    MemoryCalculatorLedger,
    OperatorExportRow,
    PendingKernelItem,
)
from .ooxml import (
    BACKLOG_ROW_HEIGHT_PT,
    CARD_ACCEPT_ROW_HEIGHT_PT,
    CARD_DESIGN_ROW_HEIGHT_PT,
    HEADER_ROW_HEIGHT_PT,
    NS_MAIN,
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
    STYLE_MERGED,
    STYLE_NEW_KERNEL,
    STYLE_OPT_BADGE,
    STYLE_OVERLAP,
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
    cell_ref,
    column_letter,
)

#: Schema tag for XLSX export receipts.
XLSX_EXPORT_RECEIPT_SCHEMA = "mindie-sim.xlsx-export-receipt/2"
#: Gap-ratio visual severity thresholds.
GAP_RATIO_SEVERE = 3.0
GAP_RATIO_MODERATE = 1.5
#: Cache bytes/elem approximations matching closed-form Excel formulas.
C8_CACHE_FACTOR = 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE)
#: Percentage divisor and MiB-per-GiB divisor for live Excel formulas.
PCT_DIVISOR = 100.0
MIB_PER_GIB = 1024
#: Overlap reconciliation threshold (ms) for emitting cross-stream overlap credit.
OVERLAP_RECONCILE_EPS_MS = 0.005

#: Canonical sheet names of the human-native Ascend optimization workbook.
SHEET_SUMMARY = "Summary"
SHEET_BREAKDOWN = "Layer_Operator_Breakdown"
SHEET_MEMORY = "Memory_Quant_Calculator"
SHEET_PENDING_KERNELS = "Pending_Kernel_Details"
#: Backward-compatible alias for Sheet 4.
SHEET_NEW_KERNELS = SHEET_PENDING_KERNELS
SHEET_MEASURED = "Measured_msprof_Calibration"

#: Sheet 1 (Summary) 3-block structural layout constants (9 columns A..I).
SUMMARY_COLS = 9
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
SUM_KPI_4D_START_ROW = 21
SUM_KPI_NOTE_START_COL = 8
SUM_B3_BANNER_ROW = 26
SUM_B3_HDR_ROW = 27
SUM_B3_DATA_START_ROW = 28

#: Sheet 2 Before-vs-After mode column indices (21 columns A..U).
CMP_COLS = 21
CMP_DATA_START_ROW = 3
CMP_FREEZE_ROWS = 2
CMP_FREEZE_COLS = 4
CMP_COL_ARCH = 1
CMP_COL_REP = 2
CMP_COL_SUB = 3
CMP_COL_ROLE = 4
CMP_COL_BEF_MS = 9
CMP_COL_BEF_BOUND = 10
CMP_COL_AFT_KERN = 11
CMP_COL_STATUS = 12
CMP_COL_AFT_US = 13
CMP_COL_STREAM = 14
CMP_COL_AFT_MS = 15
CMP_COL_SAVED_MS = 16
CMP_COL_SPEEDUP = 17
CMP_COL_AFT_BOUND = 18
CMP_COL_LSTAR_MS = 19
CMP_COL_GAP_MS = 20
CMP_COL_ACTION = 21
CMP_AFTER_MERGE_COLS = (
    CMP_COL_AFT_KERN, CMP_COL_STATUS, CMP_COL_AFT_US, CMP_COL_STREAM,
    CMP_COL_AFT_MS, CMP_COL_SAVED_MS, CMP_COL_SPEEDUP, CMP_COL_AFT_BOUND,
    CMP_COL_LSTAR_MS, CMP_COL_GAP_MS, CMP_COL_ACTION,
)
CMP_SUBTOTAL_LABEL_END_COL = 8

#: Sheet 2 Standalone mode column indices (15 columns A..O).
STD_COLS = 15
STD_DATA_START_ROW = 3
STD_COL_ARCH = 1
STD_COL_REP = 2
STD_COL_SUB = 3
STD_COL_ROLE = 4
STD_COL_MS = 11
STD_COL_LSTAR_US = 12
STD_COL_LSTAR_MS = 13
STD_COL_GAP_MS = 14
STD_COL_BOUND = 15
STD_SUBTOTAL_LABEL_END_COL = 10

#: Sheet 3 (Memory_Quant_Calculator) structural layout constants (22 columns A..V).
MEMORY_COLS = 22
MEMORY_TABLE_END_COL = 14
MEMORY_WEIGHT_MERGE_END_COL = 9
MEMORY_WEIGHT_SUM_START_COL = 10
MEMORY_CACHE_BASIS_START_COL = 7
MEMORY_CACHE_MERGE_END_COL = 4
MEMORY_CACHE_REQ_COL = 5
MEMORY_CACHE_BATCH_COL = 6
MEMORY_STAGE_HEADROOM_COL = 12
MEMORY_STAGE_STATUS_COL = 13
MEMORY_STAGE_RAW_ENGRAM_COL = 14
MEMORY_STAGE_AGG_START_COL = 3
MEMORY_CONTROL_COL_LABEL = 17
MEMORY_CONTROL_COL_VALUE = 18
MEMORY_LOOKUP_START_COL = 19
MEMORY_LOOKUP_HEADER_ROW = 3
MEMORY_LOOKUP_FIRST_ROW = 4
MEMORY_LOOKUP_LAST_ROW = 10
MEMORY_TABLE_HEADER_ROW = 3
MEMORY_FREEZE_ROWS = 3
MEMORY_QUANT_LOOKUP_ORDER = ("BF16", "FP32", "FP8", "W8A8", "W4A8", "MXFP8", "MXFP4")

#: Sheet 4 (Pending_Kernel_Details) structural layout constants (10 columns A..J).
PENDING_COLS = 10
PENDING_HEADER_ROW = 3
PENDING_DATA_START_ROW = 4
PENDING_COL_GAIN = 7
PENDING_LABEL_MERGE_END = 6
PENDING_CARD_SPACER_GAP = 4
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
PENDING_CARD_ROW_DESIGN = 6
PENDING_CARD_ROW_ACCEPT = 7

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
SUMMARY_COL_WIDTHS = (34.0, 18.0, 16.0, 15.0, 16.0, 15.0, 16.0, 16.0, 32.0)
CMP_COL_WIDTHS = (
    32.0, 10.0, 24.0, 28.0, 34.0, 28.0, 30.0, 13.0, 14.0, 22.0,
    28.0, 28.0, 13.0, 24.0, 15.0, 13.0, 11.0, 22.0, 14.0, 13.0, 40.0,
)
STD_COL_WIDTHS = (
    32.0, 10.0, 24.0, 28.0, 34.0, 28.0, 28.0, 30.0, 24.0, 13.0,
    15.0, 13.0, 14.0, 13.0, 22.0,
)
MEMORY_COL_WIDTHS = (
    16.0, 24.0, 12.0, 12.0, 18.0, 12.0, 12.0, 12.0, 13.0, 15.0, 15.0,
    13.0, 14.0, 28.0, 4.0, 4.0, 26.0, 30.0, 12.0, 12.0, 11.0, 12.0,
)
PENDING_COL_WIDTHS = (10.0, 26.0, 32.0, 26.0, 30.0, 24.0, 16.0, 22.0, 26.0, 20.0)
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

    def to_dict(self) -> dict[str, Any]:
        return {
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


def _stream_overlap_label(op: OperatorExportRow) -> str:
    if op.is_overlapped:
        return "Stream 1 (Overlapped, 0 exposed)"
    if op.stream_index == 1:
        return "Stream 1 (MSO Aux)"
    if op.stream_index == 2 or op.task_type == "HCCL":
        return "Stream 2 (HCCL / MC2)"
    return "Stream 0 (Compute)"


def _kernel_with_backend(op: OperatorExportRow) -> str:
    kern = op.ascend_kernel or op.implementation
    return f"{kern} [{op.kernel_backend}]" if op.kernel_backend else kern


def _lookup_agent_design(
    designs: Mapping[str, Mapping[str, str]] | None,
    kernel_key: str,
    target_kernel: str = "",
    logical_role: str = "",
) -> tuple[str | None, str | None]:
    if not designs:
        return None, None
    entry = designs.get(kernel_key)
    if entry is None:
        haystacks = (kernel_key.lower(), target_kernel.lower(), logical_role.lower())
        for k, val in designs.items():
            kl = str(k).lower().strip()
            if kl and any(kl == h or kl in h for h in haystacks if h):
                entry = val
                break
    if not isinstance(entry, Mapping):
        return None, None
    design = str(entry.get("design") or entry.get("proposal") or "").strip() or None
    accept = str(entry.get("acceptance") or entry.get("criteria") or "").strip() or None
    return design, accept


def _populate_summary_deployment_block(
    ws: Worksheet, bundle: ExportBundle, compare_bundle: ExportBundle | None,
) -> None:
    """Block 1: Deployment, Parallelism & Concurrency structured parameter table."""
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
    ws.set_cell(SUM_B1_HDR_ROW, SUM_RIGHT_DETAIL_START, "切分 / 显存明细 (Detail)", style=STYLE_HEADER)
    ws.merge_range(SUM_B1_HDR_ROW, SUM_RIGHT_DETAIL_START, SUM_B1_HDR_ROW, SUMMARY_COLS)

    layers_txt = (
        f"Target {ref.total_target_layers}L + Draft {ref.total_draft_layers}L"
        if ref.total_draft_layers > 0 else f"Target {ref.total_target_layers}L"
    )
    hbm_gb = int(round(mem.die_hbm_capacity_bytes / float(GIB_BYTES)))
    topo_txt = f"{ref.hardware_sku} {hbm_gb}G / {ref.topology_preset}" if ref.topology_preset else f"{ref.hardware_sku} {hbm_gb}G"
    sp_txt = "ON" if ref.sequence_parallel else "OFF"
    par_detail = f"DP={mem.dp}, TP={mem.tp}, EP={mem.ep}, PP={mem.pp}, DCP={mem.dcp}, PCP={mem.pcp}, Local-TP={mem.local_tp}, SP={sp_txt}"
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
    cluster_conc = mem.concurrency * mem.dp
    weight_gib = mem.weight_footprint_per_die_bytes / float(GIB_BYTES)
    offload_note = (
        f" | PLE/Engram Host Offload={mem.max_stage_host_offload_bytes / float(GIB_BYTES):.2f} GiB (Excluded from HBM)"
        if mem.max_stage_host_offload_bytes > 0.0 else ""
    )
    rows: tuple[tuple[int, str, Any, str | None, int, str, str, Any, str | None, int, str], ...] = (
        (
            SUM_ROW_MODEL, "模型与总层数 (Model & Total Layers)", ref.model_name, None, STYLE_MERGED, layers_txt,
            "总卡数 (Total Dies)", mem.total_dies, None, STYLE_INT, f"DP={mem.dp} × TP={mem.tp} × PP={mem.pp} (PCP={mem.pcp})",
        ),
        (
            SUM_ROW_HW, "硬件 SKU & 超节点拓扑 (Hardware / Topo)", ref.hardware_sku, None, STYLE_TEXT, f"{topo_txt} | {ref.hardware_die_summary}",
            "并行切分策略 (Full Parallelism)", f"DP{mem.dp}·TP{mem.tp}·EP{mem.ep}", None, STYLE_MERGED, par_detail,
        ),
        (
            SUM_ROW_MODE, "部署形态 (Deployment Mode)", ref.deployment_mode or "PD 分离 [非池化]", None, STYLE_MERGED, f"Phase={ref.phase.upper()} | Capture={ref.graph_capture_mode} | {track_txt}",
            "单 DP 活跃并发 (Active Concurrency / DP)", mem.concurrency, None, STYLE_INPUT_NUM, f"Active requests per DP replica (DP={mem.dp})",
        ),
        (
            SUM_ROW_LEN, "业务输入/输出长度 (Prompt / Output)", f"{ref.prompt_tokens} / {ref.output_tokens}", None, STYLE_TEXT, f"Context={ref.context_tokens} tok, Output={ref.output_tokens} tok, SeqLen={mem.seq_len}",
            "集群总并发 (Cluster Total Concurrency)", cluster_conc, f"=G{SUM_ROW_MODE}*{mem.dp}", STYLE_KPI_HIGHLIGHT, f"= Active Concurrency/DP (G{SUM_ROW_MODE}) × DP ({mem.dp})",
        ),
        (
            SUM_ROW_SLA, "SLA 约束 (SLA Target ms)", sla_target, None, STYLE_FLOAT2, sla_detail,
            "单卡权重占用 (Weight Footprint / Die GiB)", weight_gib, None, STYLE_FLOAT2, f"GiB / Die ({mem.weight_quant_display or mem.weight_quant}){offload_note}",
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


def _populate_summary_kpi_block(
    ws: Worksheet, bundle: ExportBundle, compare_bundle: ExportBundle | None,
) -> None:
    """Block 2: Before vs After Simulation KPIs highlighting TPS & Effective TPOT."""
    after = compare_bundle or bundle
    is_cmp = compare_bundle is not None
    is_decode = bundle.phase == "decode"
    title = (
        "2. 优化前后端到端仿真结果对比 (Before vs After Simulation KPIs — Highlighting TPS & Effective TPOT)"
        if is_cmp else "2. 端到端仿真核心结果总览 (End-to-End Simulation KPIs — Highlighting TPS & Effective TPOT)"
    )
    _write_banner(ws, SUM_B2_BANNER_ROW, SUMMARY_COLS, title, STYLE_SECTION)
    headers = (
        "端到端关键指标 (KPI Metric)",
        "优化前 (Before)" if is_cmp else "当前仿真 (Sim Pred)",
        "优化后 (After)" if is_cmp else "基线对照 (Current)",
        "收益变化 (Delta)", "变化幅度 (Delta %)", "加速比 (Speedup)", "单位 (Unit)",
        "SLA 判定 / 公式说明 (Verdict & Formula Basis)",
    )
    _write_headers(ws, SUM_B2_HDR_ROW, headers)
    ws.merge_range(SUM_B2_HDR_ROW, SUM_KPI_NOTE_START_COL, SUM_B2_HDR_ROW, SUMMARY_COLS)

    b_step = bundle.step_span_us / US_PER_MS
    a_step = after.step_span_us / US_PER_MS
    acc = max(1.0, after.avg_accepted_tokens)
    b_tpot = (b_step / acc) if is_decode else b_step
    a_tpot = (a_step / acc) if is_decode else a_step
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

    r_step, r_tpot, r_sla = SUM_KPI_STEP_ROW, SUM_KPI_TPOT_ROW, SUM_KPI_SLA_ROW
    r_dtps, r_ctps, r_ls, r_att = SUM_KPI_DIE_TPS_ROW, SUM_KPI_CLUSTER_TPS_ROW, SUM_KPI_LSTAR_ROW, SUM_KPI_ATTAIN_ROW
    tpot_b_f = f"=IF($B${SUM_ROW_SPEC}>0,B{r_step}/$B${SUM_ROW_SPEC},B{r_step})" if is_decode else f"=B{r_step}"
    tpot_a_f = f"=IF($B${SUM_ROW_SPEC}>0,C{r_step}/$B${SUM_ROW_SPEC},C{r_step})" if is_decode else f"=C{r_step}"
    kpi_specs: list[tuple[int, str, Any, str | None, int, Any, str | None, int, Any, str | None, int, Any, str | None, int, Any, str | None, int, str, str]] = [
        (
            r_step, "单步总耗时 (Step Latency)",
            b_step, None, STYLE_FLOAT3, a_step, None, STYLE_FLOAT3,
            a_step - b_step, f"=C{r_step}-B{r_step}", STYLE_FLOAT3,
            (a_step - b_step) / b_step if b_step > 0.0 else 0.0, f"=IF(B{r_step}>0,(C{r_step}-B{r_step})/B{r_step},0)", STYLE_PCT,
            b_step / a_step if a_step > 0.0 else 0.0, f"=IF(C{r_step}>0,B{r_step}/C{r_step},0)", STYLE_RATIO,
            "ms", f"runtime.Engine Full-Model Step (Budget <= {sla * acc if is_decode else sla:.1f} ms)",
        ),
        (
            r_tpot, "等效 TPOT (Effective TPOT)" if is_decode else "首字延迟 (Prefill TTFT)",
            b_tpot, tpot_b_f, STYLE_KPI_HIGHLIGHT, a_tpot, tpot_a_f, STYLE_KPI_HIGHLIGHT,
            a_tpot - b_tpot, f"=C{r_tpot}-B{r_tpot}", STYLE_FLOAT3,
            (a_tpot - b_tpot) / b_tpot if b_tpot > 0.0 else 0.0, f"=IF(B{r_tpot}>0,(C{r_tpot}-B{r_tpot})/B{r_tpot},0)", STYLE_PCT,
            b_tpot / a_tpot if a_tpot > 0.0 else 0.0, f"=IF(C{r_tpot}>0,B{r_tpot}/C{r_tpot},0)", STYLE_RATIO,
            "ms/tok" if is_decode else "ms", f"= Step_ms / Avg_Accepted ($B${SUM_ROW_SPEC}) | Target <= {sla:.1f} ms",
        ),
        (
            r_sla, "SLA 达标判定 (SLA Verdict)",
            "PASS" if b_pass else "FAIL", f'=IF(B{r_tpot}<=$B${SUM_ROW_SLA},"PASS","FAIL")', STYLE_ALERT_OK if b_pass else STYLE_ALERT_HIGH,
            "PASS" if a_pass else "FAIL", f'=IF(C{r_tpot}<=$B${SUM_ROW_SLA},"PASS","FAIL")', STYLE_ALERT_OK if a_pass else STYLE_ALERT_HIGH,
            sla - a_tpot, f"=$B${SUM_ROW_SLA}-C{r_tpot}", STYLE_FLOAT3,
            (sla - a_tpot) / sla if sla > 0.0 else 0.0, f"=IF($B${SUM_ROW_SLA}>0,($B${SUM_ROW_SLA}-C{r_tpot})/$B${SUM_ROW_SLA},0)", STYLE_PCT,
            "", None, STYLE_TEXT,
            "ms margin", f"PASS if Effective TPOT <= SLA ($B${SUM_ROW_SLA}); Col D/E = After Headroom",
        ),
        (
            r_dtps, "单卡吞吐 (Single-Die TPS)",
            b_dtps, f"=IF($G${SUM_ROW_MODEL}>0,B{r_ctps}/$G${SUM_ROW_MODEL},0)", STYLE_KPI_HIGHLIGHT,
            a_dtps, f"=IF($G${SUM_ROW_MODEL}>0,C{r_ctps}/$G${SUM_ROW_MODEL},0)", STYLE_KPI_HIGHLIGHT,
            a_dtps - b_dtps, f"=C{r_dtps}-B{r_dtps}", STYLE_FLOAT2,
            (a_dtps - b_dtps) / b_dtps if b_dtps > 0.0 else 0.0, f"=IF(B{r_dtps}>0,(C{r_dtps}-B{r_dtps})/B{r_dtps},0)", STYLE_PCT,
            a_dtps / b_dtps if b_dtps > 0.0 else 0.0, f"=IF(B{r_dtps}>0,C{r_dtps}/B{r_dtps},0)", STYLE_RATIO,
            "tok/s/die", f"= Cluster_TPS / Total_Dies ($G${SUM_ROW_MODEL}) — Higher is better",
        ),
        (
            r_ctps, "集群总吞吐 (Cluster Total TPS)",
            b_ctps, f"=IF(B{r_step}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}*1000/B{r_step},0)", STYLE_KPI_HIGHLIGHT,
            a_ctps, f"=IF(C{r_step}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}*1000/C{r_step},0)", STYLE_KPI_HIGHLIGHT,
            a_ctps - b_ctps, f"=C{r_ctps}-B{r_ctps}", STYLE_FLOAT2,
            (a_ctps - b_ctps) / b_ctps if b_ctps > 0.0 else 0.0, f"=IF(B{r_ctps}>0,(C{r_ctps}-B{r_ctps})/B{r_ctps},0)", STYLE_PCT,
            a_ctps / b_ctps if b_ctps > 0.0 else 0.0, f"=IF(B{r_ctps}>0,C{r_ctps}/B{r_ctps},0)", STYLE_RATIO,
            "tok/s", f"= Cluster_Concurrency ($G${SUM_ROW_LEN}) × Avg_Accepted ($B${SUM_ROW_SPEC}) × 1000 / Step_ms",
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
            "%", "Attainment = L* / Step_ms; Col F = After Gap Ratio (Step / L*)",
        ),
    ]
    util_4d = (
        ("4D 利用率 — Cube MFU %", bundle.network_cube_mfu / PCT_DIVISOR, after.network_cube_mfu / PCT_DIVISOR, "Cube AIC Effective Utilization"),
        ("4D 利用率 — Vector MFU %", bundle.network_vec_mfu / PCT_DIVISOR, after.network_vec_mfu / PCT_DIVISOR, "Vector AIV Effective Utilization"),
        ("4D 利用率 — HBM MBU %", bundle.network_mbu / PCT_DIVISOR, after.network_mbu / PCT_DIVISOR, "HBM Memory Bandwidth Utilization"),
        ("4D 利用率 — HCCL Link Util %", bundle.network_link_util / PCT_DIVISOR, after.network_link_util / PCT_DIVISOR, "HCCL / Interconnect Link Utilization"),
    )
    for idx, (lbl, bv, av, note) in enumerate(util_4d):
        r = SUM_KPI_4D_START_ROW + idx
        kpi_specs.append((
            r, lbl, bv, None, STYLE_PCT, av, None, STYLE_PCT,
            av - bv, f"=C{r}-B{r}", STYLE_PCT,
            (av - bv) / bv if bv > 0.0 else 0.0, f"=IF(B{r}>0,(C{r}-B{r})/B{r},0)", STYLE_PCT,
            av / bv if bv > 0.0 else 0.0, f"=IF(B{r}>0,C{r}/B{r},0)", STYLE_RATIO,
            "%", note,
        ))
    for r, lbl, bv, bf, bst, av, af, ast, dv, df, dst, pv, pf, pst, sv, sf, sst, unit, note in kpi_specs:
        _write_row(ws, r, (
            (lbl, None, STYLE_TEXT), (bv, bf, bst), (av, af, ast),
            (dv, df, dst), (pv, pf, pst), (sv, sf, sst), (unit, None, STYLE_TEXT),
            (note, None, STYLE_TEXT_WRAP),
        ))
        ws.merge_range(r, SUM_KPI_NOTE_START_COL, r, SUMMARY_COLS)


def _populate_summary_macro_modules_block(
    ws: Worksheet,
    bundle: ExportBundle,
    compare_bundle: ExportBundle | None,
    macro_rows: Sequence[MacroModuleRow],
) -> int:
    """Block 3: Macro-Module Breakdown & Share % (Before vs After) with exact Step reconciliation."""
    after = compare_bundle or bundle
    _write_banner(ws, SUM_B3_BANNER_ROW, SUMMARY_COLS, "3. 全网各大模块耗时与占比对比 (Macro-Module Breakdown & Share % — Before vs After)", STYLE_SECTION)
    headers = (
        "模块名 (Macro-Module)", "调用层数 (Layers)",
        "优化前耗时 (ms)", "优化前占比 (%)",
        "优化后耗时 (ms)", "优化后占比 (%)",
        "净省耗时 Δ (ms)", "物理极限 L* (ms)", "主导 Bound (Dominant Bound)",
    )
    _write_headers(ws, SUM_B3_HDR_ROW, headers)

    d_start = SUM_B3_DATA_START_ROW
    d_end = d_start + len(macro_rows) - 1
    tot_row = max(d_start, d_end + 1)
    tot_bef = sum(m.before_ms for m in macro_rows)
    tot_aft = sum(m.after_ms for m in macro_rows)
    tot_lstar = sum(m.lstar_ms for m in macro_rows)

    for offset, mod in enumerate(macro_rows):
        r = d_start + offset
        ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
        is_mso_hidden = "MSO Stream 1 Overlapped" in mod.dominant_bound
        _write_row(ws, r, (
            (mod.module_name, None, STYLE_TEXT_WRAP),
            (mod.layer_calls_label, None, STYLE_MERGED),
            (mod.before_ms, None, STYLE_FLOAT3),
            (mod.before_share_pct / PCT_DIVISOR, f"=IF($C${tot_row}>0,C{r}/$C${tot_row},0)", STYLE_PCT),
            (mod.after_ms, None, STYLE_FLOAT3),
            (mod.after_share_pct / PCT_DIVISOR, f"=IF($E${tot_row}>0,E{r}/$E${tot_row},0)", STYLE_PCT),
            (mod.saved_ms, f"=C{r}-E{r}", STYLE_FLOAT3),
            (mod.lstar_ms, None, STYLE_FLOAT3),
            (mod.dominant_bound, None, STYLE_OVERLAP if is_mso_hidden else STYLE_TEXT_WRAP),
        ))

    sum_c = f"=SUM(C{d_start}:C{d_end})" if d_end >= d_start else "=0"
    sum_d = f"=SUM(D{d_start}:D{d_end})" if d_end >= d_start else "=0"
    sum_e = f"=SUM(E{d_start}:E{d_end})" if d_end >= d_start else "=0"
    sum_f = f"=SUM(F{d_start}:F{d_end})" if d_end >= d_start else "=0"
    sum_g = f"=SUM(G{d_start}:G{d_end})" if d_end >= d_start else "=0"
    sum_h = f"=SUM(H{d_start}:H{d_end})" if d_end >= d_start else "=0"
    _write_row(ws, tot_row, (
        ("模块暴露耗时合计 (Module Exposed Sum)", None, STYLE_TOTAL_TEXT),
        (f"{after.total_target_layers}L+{after.total_draft_layers}D" if after.total_draft_layers > 0 else f"{after.total_target_layers}L", None, STYLE_TOTAL_TEXT),
        (tot_bef, sum_c, STYLE_TOTAL_FLOAT3),
        (1.0 if tot_bef > 0.0 else 0.0, sum_d, STYLE_TOTAL_PCT),
        (tot_aft, sum_e, STYLE_TOTAL_FLOAT3),
        (1.0 if tot_aft > 0.0 else 0.0, sum_f, STYLE_TOTAL_PCT),
        (tot_bef - tot_aft, sum_g, STYLE_TOTAL_FLOAT3),
        (tot_lstar, sum_h, STYLE_TOTAL_FLOAT3),
        ("Exposed Sum across all architectural modules (100% Share)", None, STYLE_TOTAL_TEXT),
    ))

    b_step = bundle.step_span_us / US_PER_MS
    a_step = after.step_span_us / US_PER_MS
    b_credit = b_step - tot_bef
    a_credit = a_step - tot_aft
    next_row = tot_row + 1
    if abs(b_credit) > OVERLAP_RECONCILE_EPS_MS or abs(a_credit) > OVERLAP_RECONCILE_EPS_MS:
        r_ov = next_row
        _write_row(ws, r_ov, (
            ("跨流并发重叠抵消 (Cross-Stream Overlap Credit)", None, STYLE_SUBTOTAL_TEXT),
            ("Multi-Stream", None, STYLE_SUBTOTAL_TEXT),
            (b_credit, f"=B{SUM_KPI_STEP_ROW}-C{tot_row}", STYLE_SUBTOTAL_NUM),
            (b_credit / tot_bef if tot_bef > 0.0 else 0.0, f"=IF($C${tot_row}>0,C{r_ov}/$C${tot_row},0)", STYLE_PCT),
            (a_credit, f"=C{SUM_KPI_STEP_ROW}-E{tot_row}", STYLE_SUBTOTAL_NUM),
            (a_credit / tot_aft if tot_aft > 0.0 else 0.0, f"=IF($E${tot_row}>0,E{r_ov}/$E${tot_row},0)", STYLE_PCT),
            (b_credit - a_credit, f"=C{r_ov}-E{r_ov}", STYLE_SUBTOTAL_NUM),
            (0.0, None, STYLE_SUBTOTAL_NUM),
            ("Concurrent Stream Overlap (e.g. DCP Comm // Indexer)", None, STYLE_OVERLAP),
        ))
        r_net = r_ov + 1
        _write_row(ws, r_net, (
            ("端到端单步总耗时 (Net Step Latency = Block 2)", None, STYLE_TOTAL_TEXT),
            ("Full Step", None, STYLE_TOTAL_TEXT),
            (b_step, f"=C{tot_row}+C{r_ov}", STYLE_TOTAL_FLOAT3),
            (1.0, None, STYLE_TOTAL_PCT),
            (a_step, f"=E{tot_row}+E{r_ov}", STYLE_TOTAL_FLOAT3),
            (1.0, None, STYLE_TOTAL_PCT),
            (b_step - a_step, f"=C{r_net}-E{r_net}", STYLE_TOTAL_FLOAT3),
            (after.step_l_star_lean_us / US_PER_MS, f"=C{SUM_KPI_LSTAR_ROW}", STYLE_TOTAL_FLOAT3),
            ("Exact conservation with Block 2 Step Latency", None, STYLE_TOTAL_TEXT),
        ))
        next_row = r_net + 1
    return next_row


def _populate_summary_sheet(
    ws: Worksheet,
    bundle: ExportBundle,
    *,
    compare_bundle: ExportBundle | None = None,
    macro_rows: Sequence[MacroModuleRow] = (),
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
    if bundle.reconciliation is not None:
        rec = bundle.reconciliation
        r_b = next_row + 1
        _write_banner(ws, r_b, SUMMARY_COLS, f"4. Measured msprof Calibration Summary ({rec.source_label} — See {SHEET_MEASURED})", STYLE_SECTION)
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


def _populate_comparison_breakdown_sheet(
    ws: Worksheet,
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    aligned_rows: Sequence[AlignedComparisonRow],
    agent_designs: Mapping[str, Mapping[str, str]] | None,
) -> None:
    """Populate Sheet 2 with side-by-side Before vs After N->1 ``merge_cells`` & live formulas."""
    _apply_col_widths(ws, CMP_COL_WIDTHS)
    ws.freeze_panes(CMP_FREEZE_ROWS, CMP_FREEZE_COLS)
    _write_banner(
        ws, 1, CMP_COLS,
        (
            f"Before ({before_bundle.feature_track or 'baseline'}) vs After ({after_bundle.feature_track or 'candidate'}) "
            f"Layer-Archetype Operator Breakdown | {before_bundle.model_name} @ {before_bundle.hardware_sku} ({before_bundle.phase.upper()})"
        ),
        STYLE_TITLE,
    )
    headers = (
        "Archetype (Scope & Layer Range)", "Repeats (×N)", "Submodule",
        "Logical Role", "Rank-Local Shapes & Dtype",
        "Before Ascend Kernel [Backend]", "Before Source Ref",
        "Before Single (us)", "Before Model (ms)", "Before Bound",
        "After Ascend Kernel [Backend]", "Kernel Status (算子状态/开发需求)",
        "After Single (us)", "After Stream / Overlap", "After Model Exposed (ms)",
        "Saved Δ (ms)", "Speedup", "After Bound",
        "Lean L* Model (ms)", "Gap vs L* (ms)",
        "Optimization Action & New-Kernel Design Slot",
    )
    _write_headers(ws, 2, headers)

    by_arch: list[tuple[str, str, int, list[AlignedComparisonRow]]] = []
    for item in aligned_rows:
        if not by_arch or by_arch[-1][0] != item.archetype_id:
            by_arch.append((item.archetype_id, item.archetype_title, item.repeat_count, []))
        by_arch[-1][3].append(item)

    row_cursor = CMP_DATA_START_ROW
    subtotal_rows: list[int] = []
    tot_bef_ms = tot_aft_ms = tot_lstar_ms = 0.0
    for _, arch_title, _, arch_items in by_arch:
        arch_start = row_cursor
        sub_spans: list[tuple[str, int, int]] = []
        arch_bef_ms = arch_aft_ms = arch_lstar_ms = 0.0
        for item in arch_items:
            row_count = max(1, len(item.before_ops), len(item.after_ops))
            r_start = row_cursor
            r_end = row_cursor + row_count - 1
            b_ms, a_ms, l_ms = _write_comparison_group(ws, r_start, r_end, item, agent_designs)
            arch_bef_ms += b_ms
            arch_aft_ms += a_ms
            arch_lstar_ms += l_ms
            _extend_span(sub_spans, item.submodule, r_start, r_end)
            row_cursor = r_end + 1
        arch_end = row_cursor - 1
        ws.set_cell(arch_start, CMP_COL_ARCH, arch_title, style=STYLE_MERGED)
        ws.merge_range(arch_start, CMP_COL_ARCH, arch_end, CMP_COL_ARCH)
        for sub_label, s1, s2 in sub_spans:
            ws.set_cell(s1, CMP_COL_SUB, sub_label, style=STYLE_MERGED)
            ws.merge_range(s1, CMP_COL_SUB, s2, CMP_COL_SUB)
        r_sub = row_cursor
        _write_comparison_subtotal_row(ws, r_sub, arch_title, arch_start, arch_end, arch_bef_ms, arch_aft_ms, arch_lstar_ms)
        subtotal_rows.append(r_sub)
        tot_bef_ms += arch_bef_ms
        tot_aft_ms += arch_aft_ms
        tot_lstar_ms += arch_lstar_ms
        row_cursor += 1
    _write_comparison_grand_total(ws, row_cursor, subtotal_rows, tot_bef_ms, tot_aft_ms, tot_lstar_ms)


def _write_comparison_group(
    ws: Worksheet, r_start: int, r_end: int, item: AlignedComparisonRow,
    agent_designs: Mapping[str, Mapping[str, str]] | None,
) -> tuple[float, float, float]:
    rep = max(1, int(item.repeat_count))
    before_ms_sum = 0.0
    if item.before_ops:
        for offset, b_op in enumerate(item.before_ops):
            r = r_start + offset
            ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
            b_ms = b_op.dur_us * rep / US_PER_MS
            before_ms_sum += b_ms
            ws.set_cell(r, CMP_COL_REP, rep, style=STYLE_INT)
            _write_row(ws, r, (
                (b_op.logical_role or b_op.op_id, None, STYLE_TEXT_WRAP),
                (b_op.tensor_contract, None, STYLE_TEXT_WRAP),
                (_kernel_with_backend(b_op), None, STYLE_TEXT_WRAP),
                (b_op.source_ref, None, STYLE_TEXT_WRAP),
                (b_op.dur_us, None, STYLE_FLOAT3),
                (b_ms, f"=H{r}*B{r}/1000", STYLE_FLOAT3),
                (b_op.bound_display, None, STYLE_TEXT),
            ), start_col=CMP_COL_ROLE)
    else:
        a0 = item.after_ops[0]
        ws.set_row_height(r_start, WRAP_ROW_HEIGHT_PT)
        ws.set_cell(r_start, CMP_COL_REP, rep, style=STYLE_INT)
        _write_row(ws, r_start, (
            (a0.logical_role or a0.op_id, None, STYLE_TEXT_WRAP),
            (a0.tensor_contract, None, STYLE_TEXT_WRAP),
            ("(Not in Before)", None, STYLE_TEXT),
            ("-", None, STYLE_TEXT),
            (0.0, None, STYLE_FLOAT3),
            (0.0, f"=H{r_start}*B{r_start}/1000", STYLE_FLOAT3),
            ("-", None, STYLE_TEXT),
        ), start_col=CMP_COL_ROLE)

    if item.after_ops:
        a_op = item.after_ops[0]
        stream_lbl = _stream_overlap_label(a_op)
        aft_ms = 0.0 if a_op.is_overlapped else (a_op.dur_us * rep / US_PER_MS)
        lstar_ms = 0.0 if a_op.is_overlapped else (a_op.l_star_lean_us * rep / US_PER_MS)
        saved_ms = before_ms_sum - aft_ms
        speedup = (before_ms_sum / aft_ms) if aft_ms > 0.0 else 0.0
        gap_ms = max(0.0, aft_ms - lstar_ms)
        design_override, _ = _lookup_agent_design(agent_designs, a_op.ascend_kernel, _kernel_with_backend(a_op), a_op.logical_role)
        action_txt = (
            f"[Agent Design] {design_override}" if design_override
            else (f"{item.action_note} -> See {SHEET_PENDING_KERNELS}" if a_op.needs_new_kernel else item.action_note)
        )
        _write_row(ws, r_start, (
            (_kernel_with_backend(a_op), None, STYLE_TEXT_WRAP),
            (a_op.kernel_status, None, STYLE_NEW_KERNEL if a_op.needs_new_kernel else STYLE_TEXT_WRAP),
            (a_op.dur_us, None, STYLE_FLOAT3),
            (stream_lbl, None, STYLE_OVERLAP if a_op.is_overlapped else STYLE_TEXT_WRAP),
            (aft_ms, f'=IF(ISNUMBER(SEARCH("Overlapped",N{r_start})),0,M{r_start}*B{r_start}/1000)', STYLE_FLOAT3),
            (saved_ms, f"=SUM(I{r_start}:I{r_end})-O{r_start}", STYLE_FLOAT3),
            (speedup, f"=IF(O{r_start}>0,SUM(I{r_start}:I{r_end})/O{r_start},0)", STYLE_RATIO),
            (a_op.bound_display, None, STYLE_OVERLAP if a_op.is_overlapped else STYLE_TEXT),
            (lstar_ms, None, STYLE_FLOAT3),
            (gap_ms, f"=MAX(0,O{r_start}-S{r_start})", STYLE_FLOAT3),
            (action_txt, None, STYLE_INPUT_TEXT_WRAP if (a_op.needs_new_kernel and not design_override) else STYLE_TEXT_WRAP),
        ), start_col=CMP_COL_AFT_KERN)
    else:
        aft_ms = lstar_ms = 0.0
        _write_row(ws, r_start, (
            ("(Eliminated in After)", None, STYLE_OVERLAP), ("Eliminated", None, STYLE_OVERLAP),
            (0.0, None, STYLE_FLOAT3), ("Eliminated (0 exposed)", None, STYLE_OVERLAP),
            (0.0, "=0", STYLE_FLOAT3), (before_ms_sum, f"=SUM(I{r_start}:I{r_end})-O{r_start}", STYLE_FLOAT3),
            (0.0, f"=IF(O{r_start}>0,SUM(I{r_start}:I{r_end})/O{r_start},0)", STYLE_RATIO),
            ("Eliminated", None, STYLE_OVERLAP), (0.0, None, STYLE_FLOAT3),
            (0.0, f"=MAX(0,O{r_start}-S{r_start})", STYLE_FLOAT3), (item.action_note, None, STYLE_TEXT_WRAP),
        ), start_col=CMP_COL_AFT_KERN)

    if r_end > r_start:
        for col in CMP_AFTER_MERGE_COLS:
            ws.merge_range(r_start, col, r_end, col)
    return before_ms_sum, aft_ms, lstar_ms


def _write_comparison_subtotal_row(
    ws: Worksheet, r_sub: int, arch_title: str, r_start: int, r_end: int,
    bef_ms: float, aft_ms: float, lstar_ms: float,
) -> None:
    ws.set_cell(r_sub, 1, f"Subtotal — {arch_title}", style=STYLE_SUBTOTAL_TEXT)
    ws.merge_range(r_sub, 1, r_sub, CMP_SUBTOTAL_LABEL_END_COL)
    ws.set_cell(r_sub, CMP_COL_BEF_MS, bef_ms, formula=f"=SUM(I{r_start}:I{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_BEF_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
    ws.merge_range(r_sub, CMP_COL_BEF_BOUND, r_sub, CMP_COL_STREAM)
    ws.set_cell(r_sub, CMP_COL_AFT_MS, aft_ms, formula=f"=SUM(O{r_start}:O{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_SAVED_MS, bef_ms - aft_ms, formula=f"=I{r_sub}-O{r_sub}", style=STYLE_SUBTOTAL_NUM)
    spd = (bef_ms / aft_ms) if aft_ms > 0.0 else 0.0
    ws.set_cell(r_sub, CMP_COL_SPEEDUP, spd, formula=f"=IF(O{r_sub}>0,I{r_sub}/O{r_sub},0)", style=STYLE_SUBTOTAL_RATIO)
    ws.set_cell(r_sub, CMP_COL_AFT_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
    ws.set_cell(r_sub, CMP_COL_LSTAR_MS, lstar_ms, formula=f"=SUM(S{r_start}:S{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_GAP_MS, max(0.0, aft_ms - lstar_ms), formula=f"=MAX(0,O{r_sub}-S{r_sub})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_ACTION, "", style=STYLE_SUBTOTAL_TEXT)


def _write_comparison_grand_total(
    ws: Worksheet, tot_row: int, subtotal_rows: Sequence[int],
    tot_bef_ms: float, tot_aft_ms: float, tot_lstar_ms: float,
) -> None:
    ws.set_cell(tot_row, 1, "WHOLE-MODEL TOTAL (Exposed Sum across All Archetypes)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, 1, tot_row, CMP_SUBTOTAL_LABEL_END_COL)
    sum_i = "=" + "+".join(f"I{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_o = "=" + "+".join(f"O{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_s = "=" + "+".join(f"S{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    ws.set_cell(tot_row, CMP_COL_BEF_MS, tot_bef_ms, formula=sum_i, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_BEF_BOUND, "", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, CMP_COL_BEF_BOUND, tot_row, CMP_COL_STREAM)
    ws.set_cell(tot_row, CMP_COL_AFT_MS, tot_aft_ms, formula=sum_o, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_SAVED_MS, tot_bef_ms - tot_aft_ms, formula=f"=I{tot_row}-O{tot_row}", style=STYLE_TOTAL_FLOAT3)
    spd = (tot_bef_ms / tot_aft_ms) if tot_aft_ms > 0.0 else 0.0
    ws.set_cell(tot_row, CMP_COL_SPEEDUP, spd, formula=f"=IF(O{tot_row}>0,I{tot_row}/O{tot_row},0)", style=STYLE_SUBTOTAL_RATIO)
    ws.set_cell(tot_row, CMP_COL_AFT_BOUND, "", style=STYLE_TOTAL_TEXT)
    ws.set_cell(tot_row, CMP_COL_LSTAR_MS, tot_lstar_ms, formula=sum_s, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_GAP_MS, max(0.0, tot_aft_ms - tot_lstar_ms), formula=f"=MAX(0,O{tot_row}-S{tot_row})", style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_ACTION, "Live formula totals (zero double-counting)", style=STYLE_TOTAL_TEXT)


def _populate_standalone_breakdown_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    _apply_col_widths(ws, STD_COL_WIDTHS)
    ws.freeze_panes(CMP_FREEZE_ROWS, CMP_FREEZE_COLS)
    _write_banner(
        ws, 1, STD_COLS,
        f"Layer-Archetype -> Submodule -> Ascend Kernel Breakdown | {bundle.model_name} @ {bundle.hardware_sku} ({bundle.phase.upper()})",
        STYLE_TITLE,
    )
    headers = (
        "Archetype (Scope & Layer Range)", "Repeats (×N)", "Submodule",
        "Logical Role", "Rank-Local Shapes & Dtype", "Ascend Kernel [Backend]",
        "Kernel Status (算子状态/开发需求)", "Source Ref", "Stream / Overlap",
        "Single Pred (us)", "Model Exposed (ms)", "Lean L* Single (us)",
        "Lean L* Model (ms)", "Gap vs L* (ms)", "Bottleneck Bound",
    )
    _write_headers(ws, 2, headers)

    by_arch: list[tuple[str, str, list[OperatorExportRow]]] = []
    for op in bundle.operators:
        aid = op.archetype_id or op.layer_label
        if not by_arch or by_arch[-1][0] != aid:
            by_arch.append((aid, op.archetype_title or op.layer_label, []))
        by_arch[-1][2].append(op)

    row_cursor = STD_DATA_START_ROW
    subtotal_rows: list[int] = []
    tot_ms = tot_lstar_ms = 0.0
    for _, arch_title, ops in by_arch:
        arch_start = row_cursor
        sub_spans: list[tuple[str, int, int]] = []
        arch_ms = arch_lstar_ms = 0.0
        for op in ops:
            r = row_cursor
            ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
            rep = max(1, int(op.repeat_count))
            stream_lbl = _stream_overlap_label(op)
            exp_ms = 0.0 if op.is_overlapped else (op.dur_us * rep / US_PER_MS)
            l_ms = op.l_star_lean_us * rep / US_PER_MS
            arch_ms += exp_ms
            arch_lstar_ms += l_ms
            ws.set_cell(r, STD_COL_REP, rep, style=STYLE_INT)
            _write_row(ws, r, (
                (op.logical_role or op.op_id, None, STYLE_TEXT_WRAP),
                (op.tensor_contract, None, STYLE_TEXT_WRAP),
                (_kernel_with_backend(op), None, STYLE_TEXT_WRAP),
                (op.kernel_status, None, STYLE_NEW_KERNEL if op.needs_new_kernel else STYLE_TEXT_WRAP),
                (op.source_ref, None, STYLE_TEXT_WRAP),
                (stream_lbl, None, STYLE_OVERLAP if op.is_overlapped else STYLE_TEXT_WRAP),
                (op.dur_us, None, STYLE_FLOAT3),
                (exp_ms, f'=IF(ISNUMBER(SEARCH("Overlapped",I{r})),0,J{r}*B{r}/1000)', STYLE_FLOAT3),
                (op.l_star_lean_us, None, STYLE_FLOAT3),
                (l_ms, f"=L{r}*B{r}/1000", STYLE_FLOAT3),
                (max(0.0, exp_ms - l_ms), f"=MAX(0,K{r}-M{r})", STYLE_FLOAT3),
                (op.bound_display, None, STYLE_OVERLAP if op.is_overlapped else STYLE_TEXT),
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
        ws.set_cell(r_sub, STD_COL_MS, arch_ms, formula=f"=SUM(K{arch_start}:K{arch_end})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_LSTAR_US, "", style=STYLE_SUBTOTAL_TEXT)
        ws.set_cell(r_sub, STD_COL_LSTAR_MS, arch_lstar_ms, formula=f"=SUM(M{arch_start}:M{arch_end})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_GAP_MS, max(0.0, arch_ms - arch_lstar_ms), formula=f"=MAX(0,K{r_sub}-M{r_sub})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
        subtotal_rows.append(r_sub)
        tot_ms += arch_ms
        tot_lstar_ms += arch_lstar_ms
        row_cursor += 1

    ws.set_cell(row_cursor, 1, "WHOLE-MODEL TOTAL (Exposed Sum)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(row_cursor, 1, row_cursor, STD_SUBTOTAL_LABEL_END_COL)
    sum_k = "=" + "+".join(f"K{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_m = "=" + "+".join(f"M{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    ws.set_cell(row_cursor, STD_COL_MS, tot_ms, formula=sum_k, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_LSTAR_US, "", style=STYLE_TOTAL_TEXT)
    ws.set_cell(row_cursor, STD_COL_LSTAR_MS, tot_lstar_ms, formula=sum_m, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_GAP_MS, max(0.0, tot_ms - tot_lstar_ms), formula=f"=MAX(0,K{row_cursor}-M{row_cursor})", style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_BOUND, f"{len(bundle.operators)} archetype ops", style=STYLE_TOTAL_TEXT)


def _populate_memory_control_panel(ws: Worksheet, mem: MemoryCalculatorLedger) -> None:
    c_lbl, c_val = MEMORY_CONTROL_COL_LABEL, MEMORY_CONTROL_COL_VALUE
    ws.set_cell(2, c_lbl, "Parallel & Quant Control Panel", style=STYLE_SECTION)
    ws.merge_range(2, c_lbl, 2, c_val)
    controls: tuple[tuple[str, Any, str | None, int], ...] = (
        ("Weight Quant Preset", mem.weight_quant, None, STYLE_INPUT_TEXT),
        ("KV/Indexer/KeyPool Dtype", mem.cache_dtype, None, STYLE_INPUT_TEXT),
        ("Engram / PLE Host Offload", "ON" if mem.engram_host_offload else "OFF", None, STYLE_INPUT_TEXT),
        ("DP", mem.dp, None, STYLE_INPUT_NUM),
        ("TP", mem.tp, None, STYLE_INPUT_NUM),
        ("EP", mem.ep, None, STYLE_INPUT_NUM),
        ("PP", mem.pp, None, STYLE_INPUT_NUM),
        ("DCP", mem.dcp, None, STYLE_INPUT_NUM),
        ("PCP", mem.pcp, None, STYLE_INPUT_NUM),
        ("Local-TP (o/q_proj)", mem.local_tp, None, STYLE_INPUT_NUM),
        ("Shared-TP", mem.shared_tp, None, STYLE_INPUT_NUM),
        ("Total Cluster Dies", mem.total_dies, "=R6*R7*R9*R11", STYLE_INT),
        ("Sequence Length (tok)", mem.seq_len, None, STYLE_INPUT_NUM),
        ("Concurrency (per DP)", mem.concurrency, None, STYLE_INPUT_NUM),
        ("Cache Block Size", mem.block_size, None, STYLE_INPUT_NUM),
        ("Spec / MTP Tokens", mem.spec_tokens, None, STYLE_INPUT_NUM),
        ("Local Cache Tokens / Rank", mem.local_cache_tokens, "=ROUNDUP(R15/(R17*R10),0)*R17", STYLE_INT),
        ("Die HBM Capacity (GiB)", mem.die_hbm_capacity_bytes / float(GIB_BYTES), None, STYLE_FLOAT2),
        ("GPU Memory Utilization", mem.gpu_memory_utilization, None, STYLE_PCT),
        ("Usable Die HBM Budget (GiB)", mem.usable_hbm_bytes / float(GIB_BYTES), "=R20*R21", STYLE_TOTAL_NUM),
        ("Quant Exclusion Policy", mem.weight_quant_display or mem.weight_quant, None, STYLE_TEXT_WRAP),
    )
    for offset, (label, val, formula, style) in enumerate(controls):
        r = MEMORY_TABLE_HEADER_ROW + offset
        ws.set_cell(r, c_lbl, label, style=STYLE_TEXT)
        ws.set_cell(r, c_val, val, formula=formula, style=style)

    ws.add_data_validation("R3", formula1='"' + ",".join(SUPPORTED_WEIGHT_QUANTS) + '"')
    ws.add_data_validation("R4", formula1='"' + ",".join(SUPPORTED_CACHE_DTYPES) + '"')
    ws.add_data_validation("R5", formula1='"ON,OFF"')

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
    ws.set_cell(2, 1, "1. Per-Die Weight + Quant Scale Closed-Form Table (Post-TP Local Shapes, Respecting quant.exclusions)", style=STYLE_SECTION)
    ws.merge_range(2, 1, 2, MEMORY_TABLE_END_COL)
    w_headers = (
        "Group", "Tensor Name", "Global Rows", "Global Cols", "Shard Rule",
        "Local Rows", "Local Cols", "Max Stage Rep", "Quant Format",
        "Payload (B)", "MX Scale (B)", "Channel Aux (B)", "Per-Die (GiB)", "Residency & Quant Policy",
    )
    _write_headers(ws, MEMORY_TABLE_HEADER_ROW, w_headers)

    lookup_ref = f"$S${MEMORY_LOOKUP_FIRST_ROW}:$V${MEMORY_LOOKUP_LAST_ROW}"
    w_start = MEMORY_TABLE_HEADER_ROW + 1
    group_spans: list[tuple[str, int, int]] = []
    for idx, row in enumerate(mem.weight_rows):
        r = w_start + idx
        _extend_span(group_spans, row.group, r, r)
        if row.is_engram_or_ple:
            res_text = "Host DRAM" if mem.engram_host_offload else "NPU HBM"
            res_formula: str | None = '=IF($R$5="ON","Host DRAM","NPU HBM")'
        else:
            res_text = f"NPU HBM ({row.quant_note})" if row.quant_note else "NPU HBM"
            res_formula = None
        _write_row(ws, r, (
            (row.name, None, STYLE_TEXT), (row.global_rows, None, STYLE_INT),
            (row.global_cols, None, STYLE_INT), (row.shard_rule, None, STYLE_TEXT),
            (row.local_rows, None, STYLE_INT), (row.local_cols, None, STYLE_INT),
            (row.max_stage_repeats, None, STYLE_INT), (row.quant_format, None, STYLE_INPUT_TEXT),
            (row.payload_bytes, f"=F{r}*ROUNDUP(G{r}*VLOOKUP(I{r},{lookup_ref},2,FALSE),0)*H{r}", STYLE_INT),
            (row.scale_bytes, f"=IF(VLOOKUP(I{r},{lookup_ref},3,FALSE)>0,F{r}*ROUNDUP(G{r}/VLOOKUP(I{r},{lookup_ref},3,FALSE),0)*H{r},0)", STYLE_INT),
            (row.aux_bytes, f"=F{r}*VLOOKUP(I{r},{lookup_ref},4,FALSE)*H{r}", STYLE_INT),
            (float(row.per_die_bytes) / float(GIB_BYTES), f"=(J{r}+K{r}+L{r})/{GIB_BYTES}", STYLE_FLOAT3),
            (res_text, res_formula, STYLE_TEXT),
        ), start_col=2)
    for grp, r1, r2 in group_spans:
        ws.set_cell(r1, 1, grp, style=STYLE_MERGED)
        ws.merge_range(r1, 1, r2, 1)

    w_end = w_start + len(mem.weight_rows) - 1
    if w_end >= w_start:
        ws.add_data_validation(f"I{w_start}:I{w_end}", formula1='"' + ",".join(MEMORY_QUANT_LOOKUP_ORDER) + '"')
    w_tot = w_end + 1
    ws.set_cell(w_tot, 1, "Weights Total (Max Stage)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(w_tot, 1, w_tot, MEMORY_WEIGHT_MERGE_END_COL)
    weight_totals = (
        sum(x.payload_bytes for x in mem.weight_rows),
        sum(x.scale_bytes for x in mem.weight_rows),
        sum(x.aux_bytes for x in mem.weight_rows),
        sum(x.per_die_bytes for x in mem.weight_rows) / float(GIB_BYTES),
    )
    for offset, tot_v in enumerate(weight_totals):
        col = MEMORY_WEIGHT_SUM_START_COL + offset
        c_let = column_letter(col)
        formula = f"=SUM({c_let}{w_start}:{c_let}{w_end})" if w_end >= w_start else "=0"
        ws.set_cell(w_tot, col, tot_v, formula=formula, style=STYLE_TOTAL_NUM)
    ws.set_cell(w_tot, MEMORY_TABLE_END_COL, f"{len(mem.weight_rows)} tensors", style=STYLE_TOTAL_TEXT)

    cache_end = _populate_cache_block(ws, w_tot + 2, mem)
    _populate_stage_budget_block(ws, cache_end + 2, mem)


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
    for offset, row in enumerate(mem.cache_rows):
        r = c_start + offset
        if row.scales_with_tokens:
            req_formula = f"=C{r}*D{r}*$R$19/{MIB_BYTES}"
            basis = f"Scales with Local Cache Tokens (BF16/C8/FP8={mem.cache_dtype}, ratio={C8_CACHE_FACTOR:.4f}x for C8)"
        else:
            req_formula = None
            basis = "Fixed per-request recurrent/conv/compressor state"
        _write_row(ws, r, (
            (row.component, None, STYLE_TEXT), (row.cache_kind, None, STYLE_TEXT),
            (row.max_stage_layers, None, STYLE_INT), (row.bytes_per_token_per_layer, None, STYLE_FLOAT2),
            (row.per_request_bytes / float(MIB_BYTES), req_formula, STYLE_FLOAT3),
            (row.batch_total_bytes / float(GIB_BYTES), f"=E{r}*$R$16/{MIB_PER_GIB}", STYLE_FLOAT3),
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


def _populate_stage_budget_block(ws: Worksheet, start_row: int, mem: MemoryCalculatorLedger) -> int:
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

    s_start = hdr_row + 1
    for offset, st in enumerate(mem.stage_budgets):
        r = s_start + offset
        raw_engram_gib = (st.engram_ple_hbm_bytes + st.engram_ple_host_bytes) / float(GIB_BYTES)
        _write_row(ws, r, (
            (f"Stage {st.stage}", None, STYLE_MERGED), (st.owned_layers, None, STYLE_INT),
            (st.weight_payload_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.weight_scale_aux_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.engram_ple_hbm_bytes / float(GIB_BYTES), f'=IF($R$5="ON",0,N{r})', STYLE_FLOAT3),
            (st.engram_ple_host_bytes / float(GIB_BYTES), f'=IF($R$5="ON",N{r},0)', STYLE_FLOAT3),
            (st.cache_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.workspace_peak_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.reserve_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.total_hbm_bytes / float(GIB_BYTES), f"=SUM(C{r}:E{r},G{r}:I{r})", STYLE_FLOAT3),
            (st.usable_hbm_budget_bytes / float(GIB_BYTES), "=$R$22", STYLE_FLOAT3),
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
    agent_designs: Mapping[str, Mapping[str, str]] | None,
) -> None:
    """Render one structured 8-row per-kernel detail card on Sheet 4."""
    is_new = item.category == PENDING_KIND_TO_IMPLEMENT
    cat_style = STYLE_NEW_KERNEL if is_new else STYLE_OPT_BADGE
    design_ov, accept_ov = _lookup_agent_design(agent_designs, item.kernel_key, item.target_kernel, item.logical_role)

    ws.set_row_height(card_start, HEADER_ROW_HEIGHT_PT)
    _write_banner(
        ws, card_start, PENDING_COLS,
        f"Card #{item.priority} | [{item.category}] {item.target_kernel} — {item.logical_role} | 预期收益/空间: {item.expected_gain_ms:.3f} ms ({item.total_calls_per_step} calls/step)",
        STYLE_CARD_HDR,
    )

    r_st = card_start + PENDING_CARD_ROW_STACK
    ws.set_row_height(r_st, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_st, 1, "分类 & 开发/优化状态", style=STYLE_MERGED)
    ws.merge_range(r_st, 1, r_st, PENDING_CARD_LBL_END)
    ws.set_cell(r_st, PENDING_CARD_LEFT_START, f"{item.category} | {item.status}", style=cat_style)
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
    ws.set_cell(r_ch, PENDING_CARD_RIGHT_START, item.source_refs or "-", style=STYLE_TEXT_WRAP)
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
        f"Before={item.before_model_ms:.3f}ms -> After={item.after_model_ms:.3f}ms | L*={item.lstar_model_ms:.3f}ms (Gap={item.gap_vs_lstar_ms:.3f}ms)",
        style=STYLE_TEXT_WRAP,
    )
    ws.merge_range(r_fr, PENDING_CARD_NOTE_START, r_fr, PENDING_COLS)

    r_sh = card_start + PENDING_CARD_ROW_SHAPE
    ws.set_row_height(r_sh, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_sh, 1, "Rank-Local 形状/精度/步长契约", style=STYLE_MERGED)
    ws.merge_range(r_sh, 1, r_sh, PENDING_CARD_LBL_END)
    ws.set_cell(
        r_sh, PENDING_CARD_LEFT_START,
        f"{item.tensor_contract} | Layout/Stride: Rank-Local Contiguous ND/NZ | {target_bundle.hardware_die_summary}",
        style=STYLE_TEXT_WRAP,
    )
    ws.merge_range(r_sh, PENDING_CARD_LEFT_START, r_sh, PENDING_COLS)

    r_bd = card_start + PENDING_CARD_ROW_BOUND
    ws.set_row_height(r_bd, WRAP_ROW_HEIGHT_PT)
    ws.set_cell(r_bd, 1, "瓶颈迁移 & L* 目标", style=STYLE_MERGED)
    ws.merge_range(r_bd, 1, r_bd, PENDING_CARD_LBL_END)
    ws.set_cell(r_bd, PENDING_CARD_LEFT_START, item.microarch_target, style=STYLE_TEXT_WRAP)
    ws.merge_range(r_bd, PENDING_CARD_LEFT_START, r_bd, PENDING_COLS)

    r_ds = card_start + PENDING_CARD_ROW_DESIGN
    ws.set_row_height(r_ds, CARD_DESIGN_ROW_HEIGHT_PT)
    ws.set_cell(r_ds, 1, "Agent 设计方案 / Tiling & 流水排布草稿", style=STYLE_MERGED)
    ws.merge_range(r_ds, 1, r_ds, PENDING_CARD_LBL_END)
    ws.set_cell(r_ds, PENDING_CARD_LEFT_START, design_ov or item.default_design_slot, style=STYLE_INPUT_TEXT_WRAP)
    ws.merge_range(r_ds, PENDING_CARD_LEFT_START, r_ds, PENDING_COLS)

    r_ac = card_start + PENDING_CARD_ROW_ACCEPT
    ws.set_row_height(r_ac, CARD_ACCEPT_ROW_HEIGHT_PT)
    ws.set_cell(r_ac, 1, "风险与验收标准 (Risks & Acceptance)", style=STYLE_MERGED)
    ws.merge_range(r_ac, 1, r_ac, PENDING_CARD_LBL_END)
    ws.set_cell(r_ac, PENDING_CARD_LEFT_START, accept_ov or item.default_acceptance_slot, style=STYLE_INPUT_TEXT_WRAP)
    ws.merge_range(r_ac, PENDING_CARD_LEFT_START, r_ac, PENDING_COLS)


def _populate_pending_kernel_details_sheet(
    ws: Worksheet,
    pending_items: Sequence[PendingKernelItem],
    target_bundle: ExportBundle,
    agent_designs: Mapping[str, Mapping[str, str]] | None,
) -> None:
    """Populate Sheet 4 with Top Summary Index Table + Structured Per-Kernel Detail Cards."""
    _apply_col_widths(ws, PENDING_COL_WIDTHS)
    ws.freeze_panes(PENDING_HEADER_ROW, 2)
    _write_banner(
        ws, 1, PENDING_COLS,
        f"待新开发 & 重点待优化算子专题明细 (Pending Kernel Details) | {target_bundle.model_name} @ {target_bundle.hardware_sku} ({target_bundle.hardware_die_summary})",
        STYLE_TITLE,
    )
    _write_banner(
        ws, 2, PENDING_COLS,
        "1. 总览索引表 (Top Summary Index — 待新开发/扩展 To-Implement + 重点待优化 Top Gap To-Optimize; Agent Design Slots Wired to --kernel-designs-json)",
        STYLE_SECTION,
    )
    headers = (
        "优先级", "分类 (待新开发 vs 重点待优化)", "目标算子 & 建议技术栈",
        "开发 / 优化状态", "替代原算子链 / 所在模块", "全网频次 (Archetypes & Calls)",
        "预期全网收益 Δ (ms)", "After -> L* (ms)", "Before -> Target Bound", "详情卡片定位",
    )
    _write_headers(ws, PENDING_HEADER_ROW, headers)

    if not pending_items:
        ws.set_cell(
            PENDING_DATA_START_ROW, 1,
            "No pending new or gap-dominated kernels detected — all kernels are near L* physical lower bound.",
            style=STYLE_OVERLAP,
        )
        ws.merge_range(PENDING_DATA_START_ROW, 1, PENDING_DATA_START_ROW, PENDING_COLS)
        return

    r_end = PENDING_DATA_START_ROW + len(pending_items) - 1
    r_tot = r_end + 1
    cards_banner_row = r_tot + 2
    cards_start_row = r_tot + PENDING_CARD_SPACER_GAP - 1
    n_new = sum(1 for x in pending_items if x.category == PENDING_KIND_TO_IMPLEMENT)
    n_opt = len(pending_items) - n_new

    for idx, item in enumerate(pending_items):
        r = PENDING_DATA_START_ROW + idx
        card_row = cards_start_row + idx * PENDING_CARD_STRIDE
        is_new = item.category == PENDING_KIND_TO_IMPLEMENT
        ws.set_row_height(r, BACKLOG_ROW_HEIGHT_PT)
        _write_row(ws, r, (
            (f"P{item.priority}", None, STYLE_MERGED),
            (item.category, None, STYLE_NEW_KERNEL if is_new else STYLE_OPT_BADGE),
            (item.target_kernel, None, STYLE_TEXT_WRAP),
            (item.status, None, STYLE_NEW_KERNEL if is_new else STYLE_TEXT_WRAP),
            (item.replaces_chain, None, STYLE_TEXT_WRAP),
            (f"{item.archetype_summary} ({item.total_calls_per_step} calls/step)", None, STYLE_TEXT_WRAP),
            (item.expected_gain_ms, None, STYLE_FLOAT3),
            (f"{item.after_model_ms:.3f} -> {item.lstar_model_ms:.3f} (Gap {item.gap_vs_lstar_ms:.3f})", None, STYLE_TEXT_WRAP),
            (f"{item.before_bound} -> {item.after_bound}", None, STYLE_TEXT_WRAP),
            (f"Card #{item.priority} (Row {card_row})", None, STYLE_MERGED),
        ))

    ws.set_cell(r_tot, 1, f"TOTAL ({n_new} 待新开发/扩展 + {n_opt} 重点待优化 = {len(pending_items)} 项)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(r_tot, 1, r_tot, PENDING_LABEL_MERGE_END)
    ws.set_cell(
        r_tot, PENDING_COL_GAIN,
        sum(x.expected_gain_ms for x in pending_items),
        formula=f"=SUM(G{PENDING_DATA_START_ROW}:G{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )
    ws.set_cell(r_tot, PENDING_COL_GAIN + 1, "Pass agent_kernel_designs or --kernel-designs-json to populate Agent Design Slots below", style=STYLE_TOTAL_TEXT)
    ws.merge_range(r_tot, PENDING_COL_GAIN + 1, r_tot, PENDING_COLS)

    _write_banner(
        ws, cards_banner_row, PENDING_COLS,
        "2. 逐算子结构化详情卡片 (Structured Per-Kernel Detail Cards — Shape/Dtype/Stride Contract, L* Target & Agent Design Proposal Slot)",
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
    model: str | None, hardware: str | None, topology_preset: str | None,
    dp: int, tp: int, dcp: int, ep: int, pp: int, pcp: int,
    prompt_tokens: int, output_tokens: int, concurrency: int,
    mtp_proposal_tokens: int, mtp_acceptance_length: float,
    compare_track: str | None, target_track: str | None,
) -> tuple[ExportBundle, ExportBundle | None]:
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
            )
            compare_source = raw_aft
    else:
        eff_req = request

    bundle = (
        source if isinstance(source, ExportBundle)
        else build_export_bundle(
            source, request=eff_req, step_eval=step_eval, replay=replay,
            observations_path=observations_path, weight_quant=weight_quant,
            cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
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
                replay=replay, weight_quant=weight_quant,
                cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
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
    agent_kernel_designs: Mapping[str, Mapping[str, str]] | None = None,
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
        pending_items = build_pending_kernel_details(aligned_rows, compare_bundle)
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

    _populate_summary_sheet(ws_summary, bundle, compare_bundle=compare_bundle, macro_rows=macro_rows)
    if compare_bundle is not None:
        _populate_comparison_breakdown_sheet(ws_breakdown, bundle, compare_bundle, aligned_rows, agent_kernel_designs)
    else:
        _populate_standalone_breakdown_sheet(ws_breakdown, bundle)
    _populate_memory_sheet(ws_memory, bundle)

    sheet_names = [SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY]
    merge_counts = {
        SHEET_SUMMARY: len(ws_summary.merged_ranges),
        SHEET_BREAKDOWN: len(ws_breakdown.merged_ranges),
        SHEET_MEMORY: len(ws_memory.merged_ranges),
    }

    if compare_bundle is not None or pending_items:
        ws_pending = wb.create_sheet(SHEET_PENDING_KERNELS)
        _populate_pending_kernel_details_sheet(ws_pending, pending_items, compare_bundle or bundle, agent_designs=agent_kernel_designs)
        sheet_names.append(SHEET_PENDING_KERNELS)
        merge_counts[SHEET_PENDING_KERNELS] = len(ws_pending.merged_ranges)

    if bundle.reconciliation is not None:
        ws_meas = wb.create_sheet(SHEET_MEASURED)
        _populate_measured_calibration_sheet(ws_meas, bundle)
        sheet_names.append(SHEET_MEASURED)
        merge_counts[SHEET_MEASURED] = len(ws_meas.merged_ranges)

    bytes_written = wb.save(output_path)
    n_new = sum(1 for x in pending_items if x.category == PENDING_KIND_TO_IMPLEMENT)
    return XlsxExportSummary(
        schema=XLSX_EXPORT_RECEIPT_SCHEMA,
        output_path=str(output_path),
        model_name=bundle.model_name,
        hardware_sku=bundle.hardware_sku,
        phase=bundle.phase,
        graph_id=bundle.graph_id,
        sheets=tuple(sheet_names),
        merge_cell_counts=merge_counts,
        operator_rows=len(aligned_rows) if compare_bundle is not None else len(bundle.operators),
        weight_rows=len(bundle.memory_ledger.weight_rows),
        has_comparison=compare_bundle is not None,
        has_measured=bundle.reconciliation is not None,
        bytes_written=bytes_written,
        new_kernel_backlog_count=n_new,
        to_optimize_kernel_count=len(pending_items) - n_new,
    )


__all__ = [
    "NS_MAIN",
    "SHEET_BREAKDOWN",
    "SHEET_MEASURED",
    "SHEET_MEMORY",
    "SHEET_NEW_KERNELS",
    "SHEET_PENDING_KERNELS",
    "SHEET_SUMMARY",
    "Workbook",
    "Worksheet",
    "XLSX_EXPORT_RECEIPT_SCHEMA",
    "XlsxExportSummary",
    "cell_ref",
    "column_letter",
    "export_comparison_xlsx",
]
