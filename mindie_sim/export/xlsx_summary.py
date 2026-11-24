"""Sheet 1 (``Summary``) writer and shared OOXML layout helpers for ``mindie_sim.export``.

[Role] Render the 4-block executive ``Summary`` worksheet (Deployment & Weight
  Footprint, Before-vs-After Simulation/Modeling KPIs, Two-Level L1/L2 Module
  Breakdown with explicit masking columns, and receipt-driven 3-Track Action
  Roadmap) plus shared sheet names, cross-sheet cell coordinates, and compact
  row/banner helper writers used by ``xlsx_breakdown``, ``xlsx_memory``, and
  ``xlsx_pending``.
[Invariants] Zero ``"4D"`` jargon, zero hybrid ``AIV``/``CCU`` strings, zero
  canned QuaRot/env-switch prose in script-free mode, strict ECMA-376 merge
  safety, and exact cross-sheet coordinate constants matching ``Summary!$B$14``,
  ``$C$14``, ``$B$25``, ``$C$25``, ``$B$26``, ``$C$26``.
[Timeouts & Locks] Pure in-memory OOXML cell assembly; no I/O or locks.
[Side-Effect Policy] Mutates only the caller-provided ``Worksheet``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .comparison import format_parallelism_codomain_summary
from .models import (
    BINDING_STATE_BOUND, GIB_BYTES, MACRO_L1_HIERARCHY, MIN_DENOM_EPS, MODE_MODELING,
    OVERLAP_EXPOSED_EPS_US, TRACK_CONFIG_ENV, TRACK_FRAMEWORK, US_PER_MS,
    ActionRoadmapRow, ExportBundle, MacroModuleRow, format_binding_badge,
)
from .public_view import format_track_label
from .ooxml import (
    BACKLOG_ROW_HEIGHT_PT, HEADER_ROW_HEIGHT_PT, STYLE_ALERT_HIGH, STYLE_ALERT_MID,
    STYLE_ALERT_OK, STYLE_FLOAT2, STYLE_FLOAT3, STYLE_HEADER, STYLE_INPUT_NUM, STYLE_INT,
    STYLE_KPI_HIGHLIGHT, STYLE_L1_NUM, STYLE_L1_PCT, STYLE_L1_TEXT, STYLE_MASKED_NUM,
    STYLE_MERGED, STYLE_NEW_KERNEL, STYLE_OPT_BADGE, STYLE_OVERLAP, STYLE_PARTIAL_OVERLAP,
    STYLE_PCT, STYLE_RATIO, STYLE_SECTION, STYLE_TEXT, STYLE_TEXT_WRAP, STYLE_TITLE,
    STYLE_TOTAL_FLOAT3, STYLE_TOTAL_PCT, STYLE_TOTAL_TEXT, WRAP_ROW_HEIGHT_PT, Worksheet,
)

SHEET_SUMMARY = "Summary"
SHEET_BREAKDOWN = "Layer_Operator_Breakdown"
SHEET_MEMORY = "Memory_Quant_Calculator"
SHEET_PENDING_KERNELS = "Pending_Kernel_Details"
SHEET_MEASURED = "Measured_msprof_Calibration"

PCT_DIVISOR = 100.0
MIB_PER_GIB = 1024
GAP_RATIO_SEVERE = 3.0
GAP_RATIO_MODERATE = 1.5
MASKED_HIGHLIGHT_EPS_MS = 0.0005

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
SUM_BEF_STEP_REF = f"{SHEET_SUMMARY}!$B${SUM_KPI_STEP_ROW}"
SUM_AFT_STEP_REF = f"{SHEET_SUMMARY}!$C${SUM_KPI_STEP_ROW}"
SUM_BEF_UPT_REF = f"{SHEET_SUMMARY}!$B${SUM_KPI_AMORTIZED_ROW}"
SUM_AFT_UPT_REF = f"{SHEET_SUMMARY}!$C${SUM_KPI_AMORTIZED_ROW}"
SUM_B4_TRACK_END = 2
SUM_B4_TITLE_START = 3
SUM_B4_TITLE_END = 4
SUM_B4_GAIN_COL = 5
SUM_B4_DETAIL_START = 6
SUM_B4_DETAIL_END = 11
SUM_B4_REF_START = 12
SUM_MEAS_NOTE_COL = 9
SUMMARY_COL_WIDTHS = (34.0, 14.0, 14.0, 14.0, 14.0, 13.0, 14.0, 14.0, 14.0, 13.0, 14.0, 13.0, 34.0)


def apply_col_widths(ws: Worksheet, widths: Sequence[float]) -> None:
    for idx, width in enumerate(widths, start=1):
        ws.set_col_width(idx, width)


def write_banner(ws: Worksheet, row: int, col_end: int, text: str, style: int) -> None:
    ws.set_cell(row, 1, text, style=style)
    ws.merge_range(row, 1, row, col_end)


def gap_ratio_style(ratio: float | None) -> int:
    if ratio is None:
        return STYLE_RATIO
    if ratio >= GAP_RATIO_SEVERE:
        return STYLE_ALERT_HIGH
    return STYLE_ALERT_MID if ratio >= GAP_RATIO_MODERATE else STYLE_ALERT_OK


def write_row(ws: Worksheet, row: int, entries: Sequence[tuple[Any, str | None, int]], *, start_col: int = 1) -> None:
    for offset, (val, formula, style) in enumerate(entries):
        ws.set_cell(row, start_col + offset, val, formula=formula, style=style)


def write_headers(ws: Worksheet, row: int, labels: Sequence[str], *, start_col: int = 1) -> None:
    ws.set_row_height(row, HEADER_ROW_HEIGHT_PT)
    for offset, label in enumerate(labels):
        ws.set_cell(row, start_col + offset, label, style=STYLE_HEADER)


def extend_span(spans: list[tuple[Any, int, int]], key: Any, r1: int, r2: int) -> None:
    if spans and spans[-1][0] == key:
        spans[-1] = (key, spans[-1][1], r2)
    else:
        spans.append((key, r1, r2))


_UNCONSTRAINED_SLA_TEXT = "不限 / N/A"
_RESOURCE_BOTTLENECK_ZH: dict[str, str] = {
    "hbm_read": "HBM 访存带宽受限",
    "hbm_write": "HBM 访存带宽受限",
    "hbm": "HBM 访存带宽受限",
    "memory": "HBM 访存带宽受限",
    "cube": "Cube 矩阵算力受限",
    "aic": "Cube 矩阵算力受限",
    "vector": "Vector 向量算力受限",
    "aiv": "Vector 向量算力受限",
    "link": "HCCL 跨卡通信链路带宽受限",
    "hccl": "HCCL 跨卡通信链路带宽受限",
    "critical_path": "关键路径串行依赖受限",
    "precedence": "关键路径串行依赖受限",
    "dag": "关键路径串行依赖受限",
}


def _format_physical_bottleneck_note(tight_resources: Sequence[str]) -> str:
    """Format physical lower-bound tight resources in plain Chinese (zero Cert/coord strings)."""
    labels: list[str] = []
    for raw in tight_resources:
        head = str(raw or "").split("@", 1)[0].strip().lower()
        if not head:
            continue
        mapped = _RESOURCE_BOTTLENECK_ZH.get(head)
        if mapped is None:
            for key, zh in _RESOURCE_BOTTLENECK_ZH.items():
                if key in head:
                    mapped = zh
                    break
        if mapped and mapped not in labels:
            labels.append(mapped)
    if not labels:
        return "主导物理瓶颈: 关键路径串行依赖受限"
    if len(labels) == 1:
        return f"主导物理瓶颈: {labels[0]}"
    return f"主导物理瓶颈 (并列): {' / '.join(labels)}"


def _populate_summary_deployment_block(
    ws: Worksheet, bundle: ExportBundle, compare_bundle: ExportBundle | None, *, codesign_note: str = "",
) -> None:
    ref = compare_bundle or bundle
    mem = ref.memory_ledger
    write_banner(ws, SUM_B1_BANNER_ROW, SUMMARY_COLS, "1. 部署方案、并行策略与并发总览 (Deployment, Parallelism & Concurrency)", STYLE_SECTION)
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

    layers_txt = f"Target {ref.total_target_layers}L + Draft {ref.total_draft_layers}L" if ref.total_draft_layers > 0 else f"Target {ref.total_target_layers}L"
    hbm_gb = int(round(mem.die_hbm_capacity_bytes / float(GIB_BYTES)))
    topo_txt = f"{ref.hardware_sku} {hbm_gb}G / {ref.topology_preset}" if ref.topology_preset else f"{ref.hardware_sku} {hbm_gb}G"
    par_short, par_detail = format_parallelism_codomain_summary(ref)
    if compare_bundle is not None and codesign_note:
        par_detail = f"{par_detail} | Diff: {codesign_note}"
    mode_tag = "建模 Modeling" if ref.analysis_mode == MODE_MODELING else "仿真 Simulation"
    if compare_bundle is not None:
        track_txt = (
            f"{format_track_label(bundle.feature_track) if bundle.feature_track else '基线配置'} -> "
            f"{format_track_label(compare_bundle.feature_track) if compare_bundle.feature_track else '优化方案'}"
        )
    else:
        track_txt = format_track_label(bundle.feature_track) if bundle.feature_track else "基线配置"
    is_decode = ref.phase == "decode"
    raw_sla_target = ref.tpot_slo_ms if is_decode else ref.ttft_slo_ms
    sla_constrained = raw_sla_target is not None
    sla_val: Any = float(raw_sla_target) if sla_constrained else _UNCONSTRAINED_SLA_TEXT
    sla_style = STYLE_FLOAT2 if sla_constrained else STYLE_TEXT
    ttft_label = f"TTFT <= {ref.ttft_slo_ms:.0f} ms" if ref.ttft_slo_ms is not None else f"TTFT: {_UNCONSTRAINED_SLA_TEXT}"
    if is_decode:
        tpot_label = f"TPOT <= {ref.tpot_slo_ms:.1f} ms" if ref.tpot_slo_ms is not None else f"TPOT: {_UNCONSTRAINED_SLA_TEXT}"
        sla_detail = f"{tpot_label} ({ttft_label})"
    else:
        sla_detail = ttft_label
    spec_detail = (
        f"Method={ref.spec_method}, Proposals={ref.spec_proposals}, Avg Accepted={ref.avg_accepted_tokens:.2f} tok/step"
        if is_decode and ref.spec_proposals > 0
        else (f"Standard Decode (Avg Accepted={ref.avg_accepted_tokens:.1f} tok/step)" if is_decode else f"Prefill Tokens/Seq={int(ref.avg_accepted_tokens)}")
    )
    if sla_constrained:
        step_budget: Any = float(raw_sla_target) * ref.avg_accepted_tokens if is_decode else float(raw_sla_target)
        budget_formula: str | None = f"=B{SUM_ROW_SLA}*B{SUM_ROW_SPEC}" if is_decode else f"=B{SUM_ROW_SLA}"
        budget_style = STYLE_FLOAT2
        budget_detail = "= TPOT SLA × 平均每步接受 Token 数" if is_decode else "= TTFT SLA 约束上限"
    else:
        step_budget = _UNCONSTRAINED_SLA_TEXT
        budget_formula = None
        budget_style = STYLE_TEXT
        budget_detail = "未设置 SLA 延迟上限 (不限 / N/A)"
    eff_cc = int(ref.cluster_concurrency) if int(getattr(ref, "cluster_concurrency", 0) or 0) > 0 else (mem.concurrency * mem.dp)
    cc_formula = f"=G{SUM_ROW_MODE}*{mem.dp}" if eff_cc == mem.concurrency * mem.dp else None
    pd_pool = ref.pd_pool_summary if isinstance(ref.pd_pool_summary, Mapping) else None
    dies_detail = f"DP={mem.dp} × TP={mem.tp} × PP={mem.pp} (PCP={mem.pcp})"
    if pd_pool:
        used_d = int(pd_pool.get("used_dies", ref.deployment_total_dies or mem.total_dies) or mem.total_dies)
        dies_detail = f"{dies_detail} | PD Cluster={used_d} dies ({int(pd_pool.get('prefill_instances', 0) or 0)}P + {int(pd_pool.get('decode_instances', 0) or 0)}D)"
    weight_gib = mem.weight_footprint_per_die_bytes / float(GIB_BYTES)
    native_gib = mem.native_bf16_weight_footprint_per_die_bytes / float(GIB_BYTES)
    offload_note = f" | Host DRAM Offload={mem.max_stage_host_offload_bytes / float(GIB_BYTES):.2f} GiB (不计入 NPU HBM)" if mem.max_stage_host_offload_bytes > 0.0 else ""
    vit_note = f" | ViT={mem.vision_weight_bytes_per_die / float(GIB_BYTES):.2f} GiB ({'已计入' if mem.include_vision_tower else '纯文本默认排除'})" if mem.vision_weight_bytes_per_die > 0.0 else ""
    weight_detail = f"GiB/Die ({mem.weight_quant_display or mem.weight_quant}) [原生 BF16: {native_gib:.2f} GiB/die, 压缩 {mem.weight_compression_ratio:.2f}x]{offload_note}{vit_note}"
    rows = (
        (SUM_ROW_MODEL, "模型与总层数 (Model & Total Layers)", ref.model_name, None, STYLE_MERGED, layers_txt, "总卡数 (Total Dies)", mem.total_dies, None, STYLE_INT, dies_detail),
        (SUM_ROW_HW, "硬件 SKU & 拓扑 (Hardware / Topology)", ref.hardware_sku, None, STYLE_TEXT, f"{topo_txt} | {ref.hardware_die_summary}", "并行切分策略 (Full Parallelism)", par_short, None, STYLE_MERGED, par_detail),
        (SUM_ROW_MODE, "部署形态 (Deployment Mode)", ref.deployment_mode or "Unspecified", None, STYLE_MERGED, f"Phase={ref.phase.upper()} | Capture={ref.graph_capture_mode} | {mode_tag} {format_binding_badge(ref.binding_state)} | {track_txt}", "单 DP 活跃并发 (Active Concurrency / DP)", mem.concurrency, None, STYLE_INPUT_NUM, f"Active requests per DP replica (DP={mem.dp})"),
        (SUM_ROW_LEN, "业务输入/输出长度 (Prompt / Output)", f"{ref.prompt_tokens} / {ref.output_tokens}", None, STYLE_TEXT, f"Context={ref.context_tokens} tok, Output={ref.output_tokens} tok, SeqLen={mem.seq_len}", "集群总并发 (Cluster Total Concurrency)", eff_cc, cc_formula, STYLE_KPI_HIGHLIGHT, f"= 单 DP 活跃并发 × DP ({mem.dp})"),
        (SUM_ROW_SLA, "SLA 约束 (SLA Target ms)", sla_val, None, sla_style, sla_detail, "单卡 HBM 权重占用 (NPU Weight / Die GiB)", weight_gib, None, STYLE_FLOAT2, weight_detail),
        (SUM_ROW_SPEC, "投机解码策略 (Avg Accepted tok/step)", ref.avg_accepted_tokens, None, STYLE_FLOAT2, spec_detail, "单卡最大可承载并发 (Max HBM Concurrency / DP)", mem.max_hbm_concurrency_per_dp, None, STYLE_INT, f"Usable HBM={mem.usable_hbm_bytes / float(GIB_BYTES):.1f} GiB | Fit={'PASS' if mem.fits_hbm else 'OOM'} | Cache={mem.cache_dtype}"),
        (SUM_ROW_BUDGET, "单步耗时预算 (Step Budget ms)", step_budget, budget_formula, budget_style, budget_detail, "EP / DCP / Local-TP / SP 扩展维度", f"EP={mem.ep} / DCP={mem.dcp}", None, STYLE_TEXT, f"PP={mem.pp}, PCP={mem.pcp}, Local-TP={mem.local_tp}, Shared-TP={mem.shared_tp}, SP={'ON' if ref.sequence_parallel else 'OFF'}"),
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
    pool = bundle.pd_pool_summary if isinstance(bundle.pd_pool_summary, Mapping) else None
    if not pool or pool.get("e2e_total_tps") is None:
        return None
    e2e = float(pool["e2e_total_tps"])
    die_f = f"=IF({col}{SUM_KPI_STEP_ROW}>0,{col}{SUM_KPI_NTOK_ROW}*1000/{col}{SUM_KPI_STEP_ROW},0)"
    if pool.get("prefill_kv_rps") is None:
        return e2e, None, die_f
    cap, output = float(pool["prefill_kv_rps"]), max(1, int(bundle.output_tokens))
    if output > 1:
        d_inst, dies = max(1, int(pool.get("decode_instances", 1) or 1)), max(1, int(bundle.memory_ledger.total_dies))
        seed = output * min(cap, d_inst * dies * float(bundle.throughput_per_die_tokens_per_s) / (output - 1))
        formula = f"={output}*MIN({cap!r},{d_inst}*{dies}*{col}{SUM_KPI_DIE_TPS_ROW}/{output - 1})"
    else:
        seed, formula = cap, f"={cap!r}"
    return e2e, (formula if abs(seed - e2e) <= 1e-6 * max(1.0, abs(e2e)) else None), die_f


_STAGE_WF_SHORT_LABELS: dict[str, str] = {
    "before": "Before",
    "config_only": "[1]配置",
    "framework_rewrite": "[2]框架",
    "after": "[3]算子/After",
}


def _format_stage_waterfall_note(stage_waterfall: Sequence[Mapping[str, Any]]) -> str:
    """Format anchored 4-stage waterfall spans in ms without altering row layout."""
    parts: list[str] = []
    for item in stage_waterfall:
        if not isinstance(item, Mapping):
            continue
        stage = str(item.get("stage") or "").strip()
        if stage not in _STAGE_WF_SHORT_LABELS:
            continue
        span_ms = float(item.get("step_span_us", 0.0) or 0.0) / US_PER_MS
        parts.append(f"{_STAGE_WF_SHORT_LABELS[stage]} {span_ms:.3f}")
    return f"4-Stage: {' -> '.join(parts)} ms" if len(parts) >= 2 else ""


def _populate_summary_kpi_block(
    ws: Worksheet,
    bundle: ExportBundle,
    compare_bundle: ExportBundle | None,
    *,
    stage_waterfall: Sequence[Mapping[str, Any]] = (),
) -> None:
    after = compare_bundle or bundle
    is_cmp, is_decode = compare_bundle is not None, bundle.phase == "decode"
    is_modeling = after.analysis_mode == MODE_MODELING or after.has_unbound_formula_ops
    title = (
        "2. 优化前后端到端性能对比 (Before [当前栈仿真基线] vs After [卡SLA目标建模终态] — 双口径守恒)" if (is_cmp and is_modeling)
        else ("2. 优化前后端到端仿真结果对比 (Before vs After Simulation KPIs — Highlighting TPS & Effective TPOT)" if is_cmp
              else "2. 端到端仿真核心结果总览 (End-to-End Simulation KPIs — Highlighting TPS & Effective TPOT)")
    )
    write_banner(ws, SUM_B2_BANNER_ROW, SUMMARY_COLS, title, STYLE_SECTION)
    write_headers(ws, SUM_B2_HDR_ROW, (
        "端到端关键指标 (KPI Metric)",
        "Before [当前栈仿真基线]" if is_cmp else "当前仿真 (Sim Pred)",
        ("After [卡SLA目标建模终态]" if is_modeling else "After [仿真对比]") if is_cmp else "基线对照 (Current)",
        "收益变化 (Delta)", "变化幅度 (Delta %)", "加速比 (Speedup)", "单位 (Unit)",
        "SLA 判定 / 物理含义说明 (Verdict & Physical Context)",
    ))
    ws.merge_range(SUM_B2_HDR_ROW, SUM_KPI_NOTE_START_COL, SUM_B2_HDR_ROW, SUMMARY_COLS)

    b_step, a_step = bundle.step_span_us / US_PER_MS, after.step_span_us / US_PER_MS
    b_acc, acc = max(1.0, bundle.avg_accepted_tokens), max(1.0, after.avg_accepted_tokens)
    b_hf = float(getattr(bundle, "handoff_tpot_ms", 0.0) or 0.0) if is_decode else 0.0
    a_hf = float(getattr(after, "handoff_tpot_ms", 0.0) or 0.0) if is_decode else 0.0
    b_tpot, a_tpot = ((b_step / b_acc) + b_hf) if is_decode else b_step, ((a_step / acc) + a_hf) if is_decode else a_step
    b_raw_sla = bundle.tpot_slo_ms if is_decode else bundle.ttft_slo_ms
    a_raw_sla = after.tpot_slo_ms if is_decode else after.ttft_slo_ms
    b_sla = float(b_raw_sla) if b_raw_sla is not None else None
    a_sla = float(a_raw_sla) if a_raw_sla is not None else None
    b_ctps, a_ctps = bundle.throughput_tokens_per_s, after.throughput_tokens_per_s
    b_dtps, a_dtps = bundle.throughput_per_die_tokens_per_s, after.throughput_per_die_tokens_per_s
    b_lstar, a_lstar = bundle.step_l_star_lean_us / US_PER_MS, after.step_l_star_lean_us / US_PER_MS
    b_att, a_att = (b_lstar / b_step) if b_step > 0.0 else 0.0, (a_lstar / a_step) if a_step > 0.0 else 0.0
    a_gap_ratio = (a_step / a_lstar) if a_lstar > 0.0 else 0.0
    b_cc = int(bundle.cluster_concurrency) if int(getattr(bundle, "cluster_concurrency", 0) or 0) > 0 else max(1, bundle.memory_ledger.concurrency * bundle.memory_ledger.dp)
    a_cc = int(after.cluster_concurrency) if int(getattr(after, "cluster_concurrency", 0) or 0) > 0 else max(1, after.memory_ledger.concurrency * after.memory_ledger.dp)
    same_deploy = bundle.memory_ledger.total_dies == after.memory_ledger.total_dies and b_cc == a_cc and abs(b_acc - acc) <= MIN_DENOM_EPS
    b_dies = max(1, bundle.memory_ledger.total_dies)

    r_step, r_tpot, r_sla = SUM_KPI_STEP_ROW, SUM_KPI_TPOT_ROW, SUM_KPI_SLA_ROW
    r_dtps, r_ctps, r_ls, r_att = SUM_KPI_DIE_TPS_ROW, SUM_KPI_CLUSTER_TPS_ROW, SUM_KPI_LSTAR_ROW, SUM_KPI_ATTAIN_ROW
    b_hfs, a_hfs = (f"+{b_hf!r}" if b_hf > 0.0 else ""), (f"+{a_hf!r}" if a_hf > 0.0 else "")
    tpot_b_f = f"=IF($B${SUM_ROW_SPEC}>0,B{r_step}/$B${SUM_ROW_SPEC},B{r_step}){b_hfs}" if (is_decode and same_deploy) else (f"=B{r_step}/{b_acc!r}{b_hfs}" if is_decode else f"=B{r_step}")
    tpot_a_f = f"=IF($B${SUM_ROW_SPEC}>0,C{r_step}/$B${SUM_ROW_SPEC},C{r_step}){a_hfs}" if is_decode else f"=C{r_step}"
    dtps_b_f = f"=IF($G${SUM_ROW_MODEL}>0,B{r_ctps}/$G${SUM_ROW_MODEL},0)" if same_deploy else f"=B{r_ctps}/{b_dies}"
    ctps_b_f: str | None = f"=IF(B{r_step}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}*1000/B{r_step},0)" if same_deploy else f"=IF(B{r_step}>0,{b_cc}*{b_acc!r}*1000/B{r_step},0)"
    ctps_a_f: str | None = f"=IF(C{r_step}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}*1000/C{r_step},0)"
    dtps_a_f = f"=IF($G${SUM_ROW_MODEL}>0,C{r_ctps}/$G${SUM_ROW_MODEL},0)"
    ctps_note = "= 集群总并发 × 平均每步接受 Token 数 × 1000 / 单步耗时"
    b_pd, a_pd = _pd_cluster_tps_cells(bundle, "B"), _pd_cluster_tps_cells(after, "C")
    if b_pd is not None:
        b_ctps, ctps_b_f, dtps_b_f = b_pd
    if a_pd is not None:
        a_ctps, ctps_a_f, dtps_a_f = a_pd
    if b_pd is not None or a_pd is not None:
        ctps_note = "PD 集群端到端 = Output × min(Prefill∧KV rps, Decode rps)（求解器胜出配比）；单卡 TPS 仍为 Decode 实例口径"

    if b_sla is None:
        b_verdict, b_verdict_f, b_verdict_st = _UNCONSTRAINED_SLA_TEXT, None, STYLE_TEXT
    else:
        b_ok = b_tpot <= b_sla
        b_verdict = "PASS" if b_ok else "FAIL"
        b_verdict_st = STYLE_ALERT_OK if b_ok else STYLE_ALERT_HIGH
        if a_sla is not None and abs(b_sla - a_sla) <= MIN_DENOM_EPS:
            b_verdict_f = f'=IF(B{r_tpot}<=$B${SUM_ROW_SLA},"PASS","FAIL")'
        else:
            b_verdict_f = f'=IF(B{r_tpot}<={b_sla!r},"PASS","FAIL")'

    if a_sla is None:
        a_verdict, a_verdict_f, a_verdict_st = _UNCONSTRAINED_SLA_TEXT, None, STYLE_TEXT
        sla_margin_v: Any = ""
        sla_margin_f: str | None = None
        sla_margin_st = STYLE_TEXT
        sla_margin_pct_v: Any = ""
        sla_margin_pct_f: str | None = None
        sla_margin_pct_st = STYLE_TEXT
        step_note = "全网端到端单步耗时 (未设 SLA 单步上限，与 Block 3 实际暴露总和 100% 守恒)"
        tpot_note = f"{'= 单步耗时 / 平均每步接受 Token 数' if is_decode else '首字延迟 (Prefill TTFT)'} | SLA 目标: {_UNCONSTRAINED_SLA_TEXT}"
        sla_note = "未设置 SLA 延迟约束 (各列独立按自身契约判定)"
    else:
        a_ok = a_tpot <= a_sla
        a_verdict = "PASS" if a_ok else "FAIL"
        a_verdict_st = STYLE_ALERT_OK if a_ok else STYLE_ALERT_HIGH
        a_verdict_f = f'=IF(C{r_tpot}<=$B${SUM_ROW_SLA},"PASS","FAIL")'
        sla_margin_v = a_sla - a_tpot
        sla_margin_f = f"=$B${SUM_ROW_SLA}-C{r_tpot}"
        sla_margin_st = STYLE_FLOAT3
        sla_margin_pct_v = (a_sla - a_tpot) / a_sla if a_sla > 0.0 else 0.0
        sla_margin_pct_f = f"=IF($B${SUM_ROW_SLA}>0,($B${SUM_ROW_SLA}-C{r_tpot})/$B${SUM_ROW_SLA},0)"
        sla_margin_pct_st = STYLE_PCT
        step_note = f"全网端到端单步耗时 (单步预算 <= {a_sla * acc if is_decode else a_sla:.1f} ms，与 Block 3 实际暴露总和 100% 守恒)"
        tpot_note = f"{'= 单步耗时 / 平均每步接受 Token 数' if is_decode else '首字延迟 (Prefill TTFT)'} | SLA 目标 <= {a_sla:.1f} ms"
        sla_note = "等效延迟 <= 各自 SLA 目标为 PASS；左侧差异列为优化后余量"

    wf_note = _format_stage_waterfall_note(stage_waterfall) if is_cmp else ""
    if wf_note:
        step_note = f"{step_note} | {wf_note}"

    bottleneck_note = _format_physical_bottleneck_note(after.step_tight_resources)
    kpi_specs = (
        (r_step, "单步总耗时 (Step Latency)", b_step, None, STYLE_FLOAT3, a_step, None, STYLE_FLOAT3, a_step - b_step, f"=C{r_step}-B{r_step}", STYLE_FLOAT3, (a_step - b_step) / b_step if b_step > 0.0 else 0.0, f"=IF(B{r_step}>0,(C{r_step}-B{r_step})/B{r_step},0)", STYLE_PCT, b_step / a_step if a_step > 0.0 else 0.0, f"=IF(C{r_step}>0,B{r_step}/C{r_step},0)", STYLE_RATIO, "ms", step_note),
        (r_tpot, "等效 TPOT (Effective TPOT)" if is_decode else "首字延迟 (Prefill TTFT)", b_tpot, tpot_b_f, STYLE_KPI_HIGHLIGHT, a_tpot, tpot_a_f, STYLE_KPI_HIGHLIGHT, a_tpot - b_tpot, f"=C{r_tpot}-B{r_tpot}", STYLE_FLOAT3, (a_tpot - b_tpot) / b_tpot if b_tpot > 0.0 else 0.0, f"=IF(B{r_tpot}>0,(C{r_tpot}-B{r_tpot})/B{r_tpot},0)", STYLE_PCT, b_tpot / a_tpot if a_tpot > 0.0 else 0.0, f"=IF(C{r_tpot}>0,B{r_tpot}/C{r_tpot},0)", STYLE_RATIO, "ms/tok" if is_decode else "ms", tpot_note),
        (r_sla, "SLA 达标判定 (SLA Verdict)", b_verdict, b_verdict_f, b_verdict_st, a_verdict, a_verdict_f, a_verdict_st, sla_margin_v, sla_margin_f, sla_margin_st, sla_margin_pct_v, sla_margin_pct_f, sla_margin_pct_st, "", None, STYLE_TEXT, "ms margin", sla_note),
        (r_dtps, "单卡吞吐 (Single-Die TPS)", b_dtps, dtps_b_f, STYLE_KPI_HIGHLIGHT, a_dtps, dtps_a_f, STYLE_KPI_HIGHLIGHT, a_dtps - b_dtps, f"=C{r_dtps}-B{r_dtps}", STYLE_FLOAT2, (a_dtps - b_dtps) / b_dtps if b_dtps > 0.0 else 0.0, f"=IF(B{r_dtps}>0,(C{r_dtps}-B{r_dtps})/B{r_dtps},0)", STYLE_PCT, a_dtps / b_dtps if b_dtps > 0.0 else 0.0, f"=IF(B{r_dtps}>0,C{r_dtps}/B{r_dtps},0)", STYLE_RATIO, "tok/s/die", "= 集群总吞吐 / 总卡数 — 越高越好"),
        (r_ctps, "集群总吞吐 (Cluster Total TPS)", b_ctps, ctps_b_f, STYLE_KPI_HIGHLIGHT, a_ctps, ctps_a_f, STYLE_KPI_HIGHLIGHT, a_ctps - b_ctps, f"=C{r_ctps}-B{r_ctps}", STYLE_FLOAT2, (a_ctps - b_ctps) / b_ctps if b_ctps > 0.0 else 0.0, f"=IF(B{r_ctps}>0,(C{r_ctps}-B{r_ctps})/B{r_ctps},0)", STYLE_PCT, a_ctps / b_ctps if b_ctps > 0.0 else 0.0, f"=IF(B{r_ctps}>0,C{r_ctps}/B{r_ctps},0)", STYLE_RATIO, "tok/s", ctps_note),
        (r_ls, "Roofline 理论物理下界 L* (Theoretical Lower Bound)", b_lstar, None, STYLE_FLOAT3, a_lstar, None, STYLE_FLOAT3, a_lstar - b_lstar, f"=C{r_ls}-B{r_ls}", STYLE_FLOAT3, (a_lstar - b_lstar) / b_lstar if b_lstar > 0.0 else 0.0, f"=IF(B{r_ls}>0,(C{r_ls}-B{r_ls})/B{r_ls},0)", STYLE_PCT, b_lstar / a_lstar if a_lstar > 0.0 else 0.0, f"=IF(C{r_ls}>0,B{r_ls}/C{r_ls},0)", STYLE_RATIO, "ms", bottleneck_note),
        (r_att, "物理极限达成率 (Physical Limit Attainment L* / Step)", b_att, f"=IF(B{r_step}>0,B{r_ls}/B{r_step},0)", STYLE_PCT, a_att, f"=IF(C{r_step}>0,C{r_ls}/C{r_step},0)", STYLE_PCT, a_att - b_att, f"=C{r_att}-B{r_att}", STYLE_PCT, "", None, STYLE_TEXT, a_gap_ratio, f"=IF(C{r_ls}>0,C{r_step}/C{r_ls},0)", gap_ratio_style(a_gap_ratio), "%", "达成率 = 理论物理下界 L* / 单步耗时；加速比列为优化后与物理下界倍差 (Step / L*)"),
    )
    for r, lbl, bv, bf, bst, av, af, ast, dv, df, dst, pv, pf, pst, sv, sf, sst, unit, note in kpi_specs:
        write_row(ws, r, ((lbl, None, STYLE_TEXT), (bv, bf, bst), (av, af, ast), (dv, df, dst), (pv, pf, pst), (sv, sf, sst), (unit, None, STYLE_TEXT), (note, None, STYLE_TEXT_WRAP)))
        ws.merge_range(r, SUM_KPI_NOTE_START_COL, r, SUMMARY_COLS)
    _write_summary_util_and_wall_rows(ws, bundle, after, is_decode=is_decode)


def _write_summary_util_and_wall_rows(ws: Worksheet, bundle: ExportBundle, after: ExportBundle, *, is_decode: bool) -> None:
    cube_note = "Decode 阶段 AIC 矩阵算力有效利用率" if is_decode else "Prefill 阶段 AIC 矩阵算力有效利用率"
    hw_utils = (
        ("Cube 计算利用率 (AIC MFU)", bundle.network_cube_mfu / PCT_DIVISOR, after.network_cube_mfu / PCT_DIVISOR, cube_note),
        ("Vector 计算利用率 (AIV Util)", bundle.network_vec_mfu / PCT_DIVISOR, after.network_vec_mfu / PCT_DIVISOR, "AIV 向量核计算利用率 (Norm / Softmax / Gating / 量化等)"),
        ("HBM 访存带宽利用率 (MBU)", bundle.network_mbu / PCT_DIVISOR, after.network_mbu / PCT_DIVISOR, "权重与 KV Cache 访存带宽有效利用率"),
        ("HCCL 通信带宽利用率 (Link BW Util)", bundle.network_link_util / PCT_DIVISOR, after.network_link_util / PCT_DIVISOR, "跨卡 TP/EP/DCP 集合通信链路带宽利用率"),
    )
    for idx, (lbl, bv, av, note) in enumerate(hw_utils):
        r = SUM_KPI_UTIL_START_ROW + idx
        write_row(ws, r, (
            (lbl, None, STYLE_TEXT), (bv, None, STYLE_PCT), (av, None, STYLE_PCT),
            (av - bv, f"=C{r}-B{r}", STYLE_PCT), ((av - bv) / bv if bv > 0.0 else 0.0, f"=IF(B{r}>0,(C{r}-B{r})/B{r},0)", STYLE_PCT),
            (av / bv if bv > 0.0 else 0.0, f"=IF(B{r}>0,C{r}/B{r},0)", STYLE_RATIO), ("%", None, STYLE_TEXT), (note, None, STYLE_TEXT_WRAP),
        ))
        ws.merge_range(r, SUM_KPI_NOTE_START_COL, r, SUMMARY_COLS)

    r_upt, r_dtps = SUM_KPI_AMORTIZED_ROW, SUM_KPI_DIE_TPS_ROW
    b_upt, a_upt = bundle.us_per_tok_die, after.us_per_tok_die
    write_row(ws, r_upt, (
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
    b_ntok, a_ntok = float(bundle.n_tok_per_die), float(after.n_tok_per_die)
    b_mem, a_mem = bundle.memory_ledger, after.memory_ledger
    a_exp = float(max(1, a_mem.concurrency * a_mem.dp)) * max(1.0, float(after.avg_accepted_tokens)) / float(max(1, a_mem.total_dies))
    b_exp = float(max(1, b_mem.concurrency * b_mem.dp)) * max(1.0, float(bundle.avg_accepted_tokens)) / float(max(1, b_mem.total_dies))
    same_d = b_mem.total_dies == a_mem.total_dies and (b_mem.concurrency * b_mem.dp) == (a_mem.concurrency * a_mem.dp) and abs(float(bundle.avg_accepted_tokens) - float(after.avg_accepted_tokens)) <= MIN_DENOM_EPS
    ntok_f = f"=IF($G${SUM_ROW_MODEL}>0,$G${SUM_ROW_LEN}*$B${SUM_ROW_SPEC}/$G${SUM_ROW_MODEL},0)"
    write_row(ws, r_ntok, (
        ("单卡单步有效 Token 数 (N_tok_per_die)", None, STYLE_TEXT),
        (b_ntok, ntok_f if (same_d and abs(b_ntok - b_exp) <= MIN_DENOM_EPS * max(1.0, abs(b_exp))) else None, STYLE_FLOAT3),
        (a_ntok, ntok_f if abs(a_ntok - a_exp) <= MIN_DENOM_EPS * max(1.0, abs(a_exp)) else None, STYLE_FLOAT3),
        (a_ntok - b_ntok, f"=C{r_ntok}-B{r_ntok}", STYLE_FLOAT3),
        ((a_ntok - b_ntok) / b_ntok if b_ntok > 0.0 else 0.0, f"=IF(B{r_ntok}>0,(C{r_ntok}-B{r_ntok})/B{r_ntok},0)", STYLE_PCT),
        (a_ntok / b_ntok if b_ntok > 0.0 else 0.0, f"=IF(B{r_ntok}>0,C{r_ntok}/B{r_ntok},0)", STYLE_RATIO),
        ("tok/die", None, STYLE_TEXT),
        ("= Cluster_Concurrency × Avg_Accepted / Total_Dies（供 Sheet 2 / Sheet 4 全精度摊薄公式直接引用）", None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(r_ntok, SUM_KPI_NOTE_START_COL, r_ntok, SUMMARY_COLS)

    wall = after.macro_wall_trigger
    wall_kind = wall.wall_kind if wall is not None else "BALANCED"
    min_hr = min((st.headroom_bytes for st in a_mem.stage_budgets), default=0.0) / float(GIB_BYTES)
    has_sla_limit = (after.tpot_slo_ms is not None) if is_decode else (after.ttft_slo_ms is not None)
    if wall is not None and has_sla_limit:
        wall_fact = (
            f"HBM Headroom={wall.hbm_headroom_gib:.2f} GiB ({wall.hbm_utilization_pct:.1f}% used) | "
            f"SLA Slack={wall.sla_slack_ms:+.2f} ms (Latency={wall.tpot_ms:.2f} vs SLA={wall.tpot_slo_ms:.1f} ms)"
        )
    elif wall is not None:
        wall_fact = (
            f"HBM Headroom={wall.hbm_headroom_gib:.2f} GiB ({wall.hbm_utilization_pct:.1f}% used) | "
            f"SLA={_UNCONSTRAINED_SLA_TEXT}"
        )
    else:
        wall_fact = (
            f"HBM Headroom={min_hr:.2f} GiB | SLA Slack={after.sla_slack_ms:+.2f} ms"
            if has_sla_limit else f"HBM Headroom={min_hr:.2f} GiB | SLA={_UNCONSTRAINED_SLA_TEXT}"
        )
    r_wall = SUM_KPI_WALL_ROW
    ws.set_row_height(r_wall, WRAP_ROW_HEIGHT_PT)
    write_row(ws, r_wall, (
        ("物理墙诊断 & 绑定声明 (Macro Wall & Binding)", None, STYLE_TEXT),
        (format_binding_badge(bundle.binding_state), None, STYLE_OVERLAP if bundle.binding_state == BINDING_STATE_BOUND else STYLE_OPT_BADGE),
        (format_binding_badge(after.binding_state), None, STYLE_OPT_BADGE if after.has_unbound_formula_ops else STYLE_OVERLAP),
        (wall_kind, None, STYLE_ALERT_MID if wall_kind != "BALANCED" else STYLE_ALERT_OK),
        (f"Before N_tok/die={bundle.n_tok_per_die:.2f}", None, STYLE_TEXT),
        (f"After N_tok/die={after.n_tok_per_die:.2f}", None, STYLE_TEXT),
        ("Wall / Mode", None, STYLE_TEXT), (wall_fact, None, STYLE_TEXT_WRAP),
    ))
    ws.merge_range(r_wall, SUM_KPI_NOTE_START_COL, r_wall, SUMMARY_COLS)


def _populate_summary_macro_modules_block(
    ws: Worksheet, bundle: ExportBundle, compare_bundle: ExportBundle | None, macro_rows: Sequence[MacroModuleRow],
) -> int:
    after = compare_bundle or bundle
    write_banner(ws, SUM_B3_BANNER_ROW, SUMMARY_COLS, "3. 全网模块两级拆解与并行掩盖对账 (L1/L2 Hierarchy — 实际暴露合计 = 100% 单步耗时，被掩盖列不计入总和)", STYLE_SECTION)
    write_headers(ws, SUM_B3_HDR_ROW, (
        "模块层级 (L1 / L2 Module)", "层数 (Layers)", "Before 流内原始 (ms)", "Before 被掩盖 (ms) [不计入总和]",
        "Before 实际暴露 (ms)", "Before 暴露占比 (%)", "After 流内原始 (ms)", "After 被掩盖 (ms) [不计入总和]",
        "After 实际暴露 (ms)", "After 暴露占比 (%)", "暴露净省 Δ (ms)", "物理极限 L* (ms)", "主导瓶颈 & 掩盖说明",
    ))
    by_l1: dict[str, list[MacroModuleRow]] = defaultdict(list)
    for mod in macro_rows:
        by_l1[mod.l1_group].append(mod)
    ordered_l1 = list(dict.fromkeys([*(l1 for l1, _ in MACRO_L1_HIERARCHY if l1 in by_l1), *by_l1.keys()]))
    tot_bef, tot_aft = sum(m.before_ms for m in macro_rows), sum(m.after_ms for m in macro_rows)
    tot_row = SUM_B3_DATA_START_ROW + sum(1 + len(by_l1[l1]) for l1 in ordered_l1)
    row_cursor, l1_rows = SUM_B3_DATA_START_ROW, []

    for l1_name in ordered_l1:
        children = by_l1[l1_name]
        r_l1, c_start, c_end = row_cursor, row_cursor + 1, row_cursor + len(children)
        l1_rows.append(r_l1)
        l1_exp_b, l1_exp_a = sum(c.before_ms for c in children), sum(c.after_ms for c in children)
        ws.set_row_height(r_l1, WRAP_ROW_HEIGHT_PT)
        write_row(ws, r_l1, (
            (l1_name, None, STYLE_L1_TEXT), (f"{len(children)} 子模块", None, STYLE_L1_TEXT),
            (sum(c.raw_before_ms for c in children), f"=SUM(C{c_start}:C{c_end})", STYLE_L1_NUM),
            (sum(c.masked_before_ms for c in children), f"=SUM(D{c_start}:D{c_end})", STYLE_L1_NUM),
            (l1_exp_b, f"=SUM(E{c_start}:E{c_end})", STYLE_L1_NUM),
            (l1_exp_b / tot_bef if tot_bef > 0.0 else 0.0, f"=IF($E${tot_row}>0,E{r_l1}/$E${tot_row},0)", STYLE_L1_PCT),
            (sum(c.raw_after_ms for c in children), f"=SUM(G{c_start}:G{c_end})", STYLE_L1_NUM),
            (sum(c.masked_after_ms for c in children), f"=SUM(H{c_start}:H{c_end})", STYLE_L1_NUM),
            (l1_exp_a, f"=SUM(I{c_start}:I{c_end})", STYLE_L1_NUM),
            (l1_exp_a / tot_aft if tot_aft > 0.0 else 0.0, f"=IF($I${tot_row}>0,I{r_l1}/$I${tot_row},0)", STYLE_L1_PCT),
            (l1_exp_b - l1_exp_a, f"=E{r_l1}-I{r_l1}", STYLE_L1_NUM),
            (sum(c.lstar_ms for c in children), f"=SUM(L{c_start}:L{c_end})", STYLE_L1_NUM),
            (f"L1 汇总（暴露占比 Before {100.0 * l1_exp_b / tot_bef if tot_bef > 0 else 0:.1f}% -> After {100.0 * l1_exp_a / tot_aft if tot_aft > 0 else 0:.1f}%）", None, STYLE_L1_TEXT),
        ))
        row_cursor = c_start
        for idx, mod in enumerate(children):
            r = row_cursor
            ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
            prefix = "    └─ " if idx == len(children) - 1 else "    ├─ "
            is_full = mod.after_ms <= OVERLAP_EXPOSED_EPS_US and mod.raw_after_ms > 0.0
            is_part = (not is_full) and (abs(mod.masked_after_ms) > MASKED_HIGHLIGHT_EPS_MS or abs(mod.masked_before_ms) > MASKED_HIGHLIGHT_EPS_MS)
            note_style = STYLE_OVERLAP if is_full else (STYLE_PARTIAL_OVERLAP if is_part else STYLE_TEXT_WRAP)
            write_row(ws, r, (
                (f"{prefix}{mod.module_name}", None, STYLE_TEXT_WRAP), (mod.layer_calls_label, None, STYLE_MERGED),
                (mod.raw_before_ms, None, STYLE_FLOAT3),
                (mod.masked_before_ms, None, STYLE_MASKED_NUM if mod.masked_before_ms < -MASKED_HIGHLIGHT_EPS_MS else STYLE_FLOAT3),
                (mod.before_ms, f"=C{r}+D{r}", STYLE_FLOAT3),
                (mod.before_share_pct / PCT_DIVISOR, f"=IF($E${tot_row}>0,E{r}/$E${tot_row},0)", STYLE_PCT),
                (mod.raw_after_ms, None, STYLE_FLOAT3),
                (mod.masked_after_ms, None, STYLE_MASKED_NUM if mod.masked_after_ms < -MASKED_HIGHLIGHT_EPS_MS else STYLE_FLOAT3),
                (mod.after_ms, f"=G{r}+H{r}", STYLE_FLOAT3),
                (mod.after_share_pct / PCT_DIVISOR, f"=IF($I${tot_row}>0,I{r}/$I${tot_row},0)", STYLE_PCT),
                (mod.saved_ms, f"=E{r}-I{r}", STYLE_FLOAT3), (mod.lstar_ms, None, STYLE_FLOAT3),
                (mod.masking_note or mod.dominant_bound, None, note_style),
            ))
            row_cursor += 1

    sum_l1 = lambda col: "=" + "+".join(f"{col}{idx}" for idx in l1_rows) if l1_rows else "=0"
    layer_lbl = f"{after.total_target_layers}L+{after.total_draft_layers}D" if after.total_draft_layers > 0 else f"{after.total_target_layers}L"
    write_row(ws, tot_row, (
        ("全网端到端合计 (Grand Total = SUM of L1 实际暴露 = Block 2 Step Latency)", None, STYLE_TOTAL_TEXT),
        (layer_lbl, None, STYLE_TOTAL_TEXT),
        (sum(m.raw_before_ms for m in macro_rows), sum_l1("C"), STYLE_TOTAL_FLOAT3),
        (sum(m.masked_before_ms for m in macro_rows), sum_l1("D"), STYLE_TOTAL_FLOAT3),
        (tot_bef, sum_l1("E"), STYLE_TOTAL_FLOAT3), (1.0 if tot_bef > 0.0 else 0.0, sum_l1("F"), STYLE_TOTAL_PCT),
        (sum(m.raw_after_ms for m in macro_rows), sum_l1("G"), STYLE_TOTAL_FLOAT3),
        (sum(m.masked_after_ms for m in macro_rows), sum_l1("H"), STYLE_TOTAL_FLOAT3),
        (tot_aft, sum_l1("I"), STYLE_TOTAL_FLOAT3), (1.0 if tot_aft > 0.0 else 0.0, sum_l1("J"), STYLE_TOTAL_PCT),
        (tot_bef - tot_aft, f"=E{tot_row}-I{tot_row}", STYLE_TOTAL_FLOAT3),
        (sum(m.lstar_ms for m in macro_rows), sum_l1("L"), STYLE_TOTAL_FLOAT3),
        ("100% 端到端守恒（流内原始 + 被掩盖 = 实际暴露）", None, STYLE_TOTAL_TEXT),
    ))
    return tot_row + 1


def _populate_summary_action_roadmap_block(
    ws: Worksheet, start_row: int, roadmap: Sequence[ActionRoadmapRow], *, is_script_audit: bool = False,
) -> int:
    if not roadmap:
        return start_row
    r_banner = start_row + 1
    banner_txt = (
        "4. 优化落地指引清单（A: 脚本整改项 / B: 框架图改写 / C: 算子攻关 — 基于真实 Diff 收据）"
        if is_script_audit else
        "4. 优化落地指引清单（A: 并行与配置变更 / B: 框架图改写 / C: 算子攻关 — 基于真实 Diff 收据）"
    )
    write_banner(ws, r_banner, SUMMARY_COLS, banner_txt, STYLE_SECTION)
    r_hdr = r_banner + 1
    ws.set_row_height(r_hdr, HEADER_ROW_HEIGHT_PT)
    for c1, c2, lbl in (
        (1, SUM_B4_TRACK_END, "落地赛道 (Track)"), (SUM_B4_TITLE_START, SUM_B4_TITLE_END, "优化项 (Item)"),
        (SUM_B4_GAIN_COL, SUM_B4_GAIN_COL, "预期收益 Δ (ms)"),
        (SUM_B4_DETAIL_START, SUM_B4_DETAIL_END, "真实收据明细 (Verified Diff / Rewrite / Kernel Receipt)"),
        (SUM_B4_REF_START, SUMMARY_COLS, "明细位置 (Ref)"),
    ):
        ws.set_cell(r_hdr, c1, lbl, style=STYLE_HEADER)
        if c2 > c1:
            ws.merge_range(r_hdr, c1, r_hdr, c2)

    r_cur = r_hdr + 1
    for item in roadmap:
        ws.set_row_height(r_cur, BACKLOG_ROW_HEIGHT_PT)
        tr_style = STYLE_OVERLAP if item.track == TRACK_CONFIG_ENV else (STYLE_OPT_BADGE if item.track == TRACK_FRAMEWORK else STYLE_NEW_KERNEL)
        ws.set_cell(r_cur, 1, item.track_label, style=tr_style)
        ws.merge_range(r_cur, 1, r_cur, SUM_B4_TRACK_END)
        ws.set_cell(r_cur, SUM_B4_TITLE_START, item.title, style=STYLE_TEXT_WRAP)
        ws.merge_range(r_cur, SUM_B4_TITLE_START, r_cur, SUM_B4_TITLE_END)
        ws.set_cell(r_cur, SUM_B4_GAIN_COL, "" if item.expected_gain_ms is None else float(item.expected_gain_ms), style=STYLE_TEXT if item.expected_gain_ms is None else STYLE_FLOAT3)
        ws.set_cell(r_cur, SUM_B4_DETAIL_START, item.action_detail, style=STYLE_TEXT_WRAP)
        ws.merge_range(r_cur, SUM_B4_DETAIL_START, r_cur, SUM_B4_DETAIL_END)
        ws.set_cell(r_cur, SUM_B4_REF_START, item.cross_ref, style=STYLE_MERGED)
        ws.merge_range(r_cur, SUM_B4_REF_START, r_cur, SUMMARY_COLS)
        r_cur += 1
    return r_cur


def populate_summary_sheet(
    ws: Worksheet, bundle: ExportBundle, *, compare_bundle: ExportBundle | None = None,
    macro_rows: Sequence[MacroModuleRow] = (), roadmap: Sequence[ActionRoadmapRow] = (),
    codesign_note: str = "", is_script_audit: bool = False,
    stage_waterfall: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Populate Sheet 1 (``Summary``)."""
    apply_col_widths(ws, SUMMARY_COL_WIDTHS)
    ws.freeze_panes(SUMMARY_FREEZE_ROWS, 0)
    if compare_bundle is not None:
        track_note = (
            f" | {format_track_label(bundle.feature_track) if bundle.feature_track else '基线配置'} -> "
            f"{format_track_label(compare_bundle.feature_track) if compare_bundle.feature_track else '优化方案'}"
        )
    elif bundle.feature_track:
        track_note = f" | {format_track_label(bundle.feature_track)}"
    else:
        track_note = ""
    write_banner(ws, 1, SUMMARY_COLS, f"MindIE-Sim Ascend Performance & Optimization Summary | {bundle.model_name} @ {bundle.hardware_sku}{track_note}", STYLE_TITLE)
    _populate_summary_deployment_block(ws, bundle, compare_bundle, codesign_note=codesign_note)
    _populate_summary_kpi_block(ws, bundle, compare_bundle, stage_waterfall=stage_waterfall)
    next_row = _populate_summary_macro_modules_block(ws, bundle, compare_bundle, macro_rows)
    next_row = _populate_summary_action_roadmap_block(ws, next_row, roadmap, is_script_audit=is_script_audit)
    if bundle.reconciliation is not None:
        rec = bundle.reconciliation
        r_b = next_row + 1
        write_banner(ws, r_b, SUMMARY_COLS, f"5. Measured msprof Calibration Summary ({rec.source_label} — See {SHEET_MEASURED})", STYLE_SECTION)
        delta = rec.matched_sim_sum_us - rec.matched_measured_sum_us
        ape = (abs(delta) / rec.matched_measured_sum_us) if rec.matched_measured_sum_us > 0.0 else 0.0
        write_row(ws, r_b + 1, (
            ("Matched Groups Sum (us)", None, STYLE_TEXT), (f"{rec.matched_kernels}/{rec.total_kernels} kernels", None, STYLE_MERGED),
            (rec.matched_sim_sum_us, None, STYLE_FLOAT2), ("Sim Pred (us)", None, STYLE_TEXT),
            (rec.matched_measured_sum_us, None, STYLE_FLOAT2), ("Measured (us)", None, STYLE_TEXT),
            (delta, f"=C{r_b + 1}-E{r_b + 1}", STYLE_FLOAT2),
            (ape, f"=IF(E{r_b + 1}>0,ABS(C{r_b + 1}-E{r_b + 1})/E{r_b + 1},0)", STYLE_PCT),
            (f"Full 1-to-N details on {SHEET_MEASURED}", None, STYLE_TEXT),
        ))
        ws.merge_range(r_b + 1, SUM_MEAS_NOTE_COL, r_b + 1, SUMMARY_COLS)


__all__ = [
    "MIB_PER_GIB", "PCT_DIVISOR", "SHEET_BREAKDOWN", "SHEET_MEASURED", "SHEET_MEMORY",
    "SHEET_PENDING_KERNELS", "SHEET_SUMMARY", "SUMMARY_COLS",
    "SUM_AFT_NTOK_REF", "SUM_AFT_STEP_REF", "SUM_AFT_UPT_REF", "SUM_BEF_NTOK_REF",
    "SUM_BEF_STEP_REF", "SUM_BEF_UPT_REF", "SUM_KPI_AMORTIZED_ROW", "SUM_KPI_NTOK_ROW",
    "SUM_KPI_STEP_ROW", "apply_col_widths", "extend_span", "gap_ratio_style",
    "populate_summary_sheet", "write_banner", "write_headers", "write_row",
]
