"""Sheet 3 (``Memory_Quant_Calculator``) pure ``DieResidentReceipt`` worksheet writer.

[Role] Render the 24-column Sheet 3 per-die Weight/Quant/Cache/Engram-PLE/PP HBM
  calculator directly from ``MemoryCalculatorLedger`` (projected with 0 B
  bijection from ``DieResidentReceipt`` and ``HybridPoolReceipt``).
[Invariants] Strictly hardware-scoped via canonical ``normalize_hw_family`` (zero
  SKU substring sniffing). Every cache-dtype switch formula uses exact per-dtype
  bytes projected by canonical ``Memory`` (zero scalar precision-ratio scaling).
  Fixed per-die static C8 channel scale rows never scale with concurrency
  (``$T$18``). Zero ``PresetLinearFP8`` or ``"FP8"`` formula branches on A2/A3;
  zero ``Uncalibrated`` labels.
[Timeouts & Locks] Single-pass O(Weights + Cache + Stages) OOXML assembly.
[Side-Effect Policy] Mutates only the caller-provided ``Worksheet``.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from mindie_sim.catalog.npu_checkpoints import normalize_hw_family

from .memory_ledger import (
    KV_DTYPE_SENSITIVE_COMPONENTS,
    _PREPARED_WEIGHT_GROUP,
    _weight_format_storage_spec,
)
from .models import (
    BF16_BYTES,
    GIB_BYTES,
    MIB_BYTES,
    MIN_DENOM_EPS,
    QUANT_RULE_CHECKPOINT_IGNORE,
    QUANT_RULE_FIXED_BF16,
    QUANT_RULE_FIXED_FP32,
    QUANT_RULE_PRESET_LINEAR_FP8,
    QUANT_RULE_PRESET_LINEAR_W8A8,
    SUPPORTED_WEIGHT_QUANTS,
    WEIGHT_QUANT_FORMATS,
    ExportBundle,
    MemoryCalculatorLedger,
)
from .ooxml import (
    STYLE_ALERT_HIGH,
    STYLE_ALERT_OK,
    STYLE_FLOAT2,
    STYLE_FLOAT3,
    STYLE_INPUT_NUM,
    STYLE_INPUT_TEXT,
    STYLE_INT,
    STYLE_MERGED,
    STYLE_OVERLAP,
    STYLE_PCT,
    STYLE_SECTION,
    STYLE_SUBTOTAL_NUM,
    STYLE_SUBTOTAL_TEXT,
    STYLE_TEXT,
    STYLE_TEXT_WRAP,
    STYLE_TITLE,
    STYLE_TOTAL_NUM,
    STYLE_TOTAL_TEXT,
    Worksheet,
    column_letter,
)
from .xlsx_summary import (
    MIB_PER_GIB,
    apply_col_widths,
    extend_span,
    write_banner,
    write_headers,
    write_row,
)

#: Default Sheet 3 quant mode preserving every row's evaluated format and scale bytes.
QUANT_MODE_AS_EVALUATED = "As evaluated"
#: Weight table group label for routed MoE experts eligible for DDR offload scaling.
_ROUTED_EXPERT_WEIGHT_GROUP = "MoE Routed Experts"
#: Minimum byte delta separating DDR expert offload from float rounding noise.
_EXPERT_OFFLOAD_BYTES_EPS = 1.0
#: Percentage display multiplier for partial DDR offload residency annotations.
_OFFLOAD_PCT_SCALE = 100

#: Sheet 3 structural layout constants (24 columns A..X).
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
MEMORY_TABLE_HEADER_ROW = 3
MEMORY_FREEZE_ROWS = 3

#: Hardware-scoped weight format sets (derived from canonical ``normalize_hw_family``).
A5_WEIGHT_QUANTS = ("BF16", "FP8", "MXFP8", "MXFP4")
A5_LOOKUP_ORDER = ("BF16", "FP32", "FP8", "MXFP8", "MXFP4")
A23_WEIGHT_QUANTS = ("BF16", "W8A8", "W4A8")
A23_LOOKUP_ORDER = ("BF16", "FP32", "W8A8", "W4A8")
GENERIC_LOOKUP_ORDER = ("BF16", "FP32", "FP8", "W8A8", "W4A8", "MXFP8", "MXFP4")
_FIXED_PER_DIE_COMPONENTS = frozenset({"c8_static_channel_scale"})

#: Control Panel absolute cell references in column T (20).
MEM_CTL_QUANT_REF = "$T$3"
MEM_CTL_CACHE_REF = "$T$4"
MEM_CTL_OFFLOAD_REF = "$T$5"
MEM_CTL_OVERRIDE_IGN_REF = "$T$6"
MEM_CTL_VIT_REF = "$T$7"
MEM_CTL_CONC_REF = "$T$18"
MEM_CTL_LOCAL_TOK_REF = "$T$21"
MEM_CTL_GPU_UTIL_REF = "$T$23"
MEM_CTL_USABLE_HBM_REF = "$T$24"

MEMORY_COL_WIDTHS = (
    16.0, 24.0, 11.0, 11.0, 16.0, 11.0, 11.0, 11.0, 19.0, 13.0, 14.0, 13.0,
    12.0, 15.0, 16.0, 28.0, 3.5, 3.5, 28.0, 26.0, 11.0, 10.0, 10.0, 11.0,
)


def _device_format_scope(
    mem: MemoryCalculatorLedger,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return ``(arch_kind, weight_dropdown, cache_dropdown, lookup_order)`` via canonical HW family."""
    hw_family = normalize_hw_family(str(mem.hardware_sku or ""))
    if hw_family == "a5":
        arch = "a5"
        w_opts = list(A5_WEIGHT_QUANTS)
        lookup = list(A5_LOOKUP_ORDER)
    elif hw_family in {"a2", "a3"}:
        arch = "a23"
        w_opts = list(A23_WEIGHT_QUANTS)
        lookup = list(A23_LOOKUP_ORDER)
    else:
        arch = "generic"
        w_opts = list(SUPPORTED_WEIGHT_QUANTS)
        lookup = list(GENERIC_LOOKUP_ORDER)

    # Preserve VLOOKUP completeness for every active format already present on the ledger.
    active_fmts = [mem.weight_quant, *(row.quant_format for row in mem.weight_rows)]
    for fmt in active_fmts:
        up = str(fmt or "").strip().upper()
        if up in WEIGHT_QUANT_FORMATS and up not in lookup:
            lookup.append(up)
        if up in WEIGHT_QUANT_FORMATS and up != "FP32" and up not in w_opts:
            w_opts.append(up)

    # Derive legal cache dtypes strictly from the exact Memory-resolved dtype matrix on cache_rows.
    observed_cache_dtypes: list[str] = []
    for row in mem.cache_rows:
        for dt, _val in row.per_request_bytes_by_dtype:
            up = str(dt or "").strip().upper()
            if up and up not in observed_cache_dtypes:
                observed_cache_dtypes.append(up)
    active_cd = str(mem.cache_dtype or "BF16").strip().upper()
    if active_cd and active_cd not in observed_cache_dtypes:
        observed_cache_dtypes.append(active_cd)
    ordered_cd = (["BF16"] if "BF16" in observed_cache_dtypes else []) + [
        dt for dt in observed_cache_dtypes if dt != "BF16"
    ]
    return arch, tuple(w_opts), tuple(ordered_cd or [active_cd]), tuple(lookup)


def _populate_memory_control_panel(
    ws: Worksheet,
    mem: MemoryCalculatorLedger,
    w_opts: Sequence[str],
    c_opts: Sequence[str],
    lookup_order: Sequence[str],
) -> int:
    """Populate interactive control panel (S:T) and hardware-scoped lookup table (U:X)."""
    c_lbl, c_val = MEMORY_CONTROL_COL_LABEL, MEMORY_CONTROL_COL_VALUE
    ws.set_cell(2, c_lbl, "Control Panel (Editable What-If)", style=STYLE_SECTION)
    ws.merge_range(2, c_lbl, 2, c_val)
    controls = (
        ("Weight Quant Preset", QUANT_MODE_AS_EVALUATED, None, STYLE_INPUT_TEXT),
        ("KV Cache Dtype", mem.cache_dtype, None, STYLE_INPUT_TEXT),
        ("Engram / PLE Host Offload", "ON" if mem.engram_host_offload else "OFF", None, STYLE_INPUT_TEXT),
        ("Override Checkpoint Ignore", "ON" if mem.apply_preset_to_ignored else "OFF", None, STYLE_INPUT_TEXT),
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
        ("Quant Policy", mem.weight_quant_display or mem.weight_quant, None, STYLE_TEXT_WRAP),
    )
    for offset, (label, val, formula, style) in enumerate(controls):
        r = MEMORY_TABLE_HEADER_ROW + offset
        ws.set_cell(r, c_lbl, label, style=STYLE_TEXT)
        ws.set_cell(r, c_val, val, formula=formula, style=style)

    w_dropdown = (QUANT_MODE_AS_EVALUATED, *(opt for opt in w_opts if opt != QUANT_MODE_AS_EVALUATED))
    ws.add_data_validation("T3", formula1='"' + ",".join(w_dropdown) + '"')
    ws.add_data_validation("T4", formula1='"' + ",".join(c_opts) + '"')
    ws.add_data_validation("T5", formula1='"ON,OFF"')
    ws.add_data_validation("T6", formula1='"OFF,ON"')
    ws.add_data_validation("T7", formula1='"OFF,ON"')

    ws.set_cell(2, MEMORY_LOOKUP_START_COL, "Quant Format Storage Rules", style=STYLE_SECTION)
    ws.merge_range(2, MEMORY_LOOKUP_START_COL, 2, MEMORY_COLS)
    write_headers(
        ws, MEMORY_LOOKUP_HEADER_ROW,
        ("Format", "B/Elem", "MX Group", "Aux B/Row"),
        start_col=MEMORY_LOOKUP_START_COL,
    )
    for idx, fmt in enumerate(lookup_order):
        r = MEMORY_LOOKUP_FIRST_ROW + idx
        spec = _weight_format_storage_spec(fmt)
        write_row(ws, r, (
            (fmt, None, STYLE_TEXT),
            (float(spec["bytes"]), None, STYLE_FLOAT2),
            (int(spec["group"]), None, STYLE_INT),
            (int(spec["aux"]), None, STYLE_INT),
        ), start_col=MEMORY_LOOKUP_START_COL)
    return MEMORY_LOOKUP_FIRST_ROW + len(lookup_order) - 1


def _quant_format_formula(quant_rule: str, eval_fmt: str, arch: str) -> str | None:
    """Return hardware-scoped live Excel formula honoring ``As evaluated`` default mode."""
    if quant_rule in {QUANT_RULE_FIXED_BF16, QUANT_RULE_FIXED_FP32}:
        return None
    if arch == "a5":
        linear_expr = f'IF({MEM_CTL_QUANT_REF}="MXFP4","MXFP8",{MEM_CTL_QUANT_REF})'
    elif arch == "a23":
        linear_expr = f'IF({MEM_CTL_QUANT_REF}="W4A8","W8A8",{MEM_CTL_QUANT_REF})'
    else:
        linear_expr = (
            f'IF({MEM_CTL_QUANT_REF}="W4A8","W8A8",'
            f'IF({MEM_CTL_QUANT_REF}="MXFP4","MXFP8",{MEM_CTL_QUANT_REF}))'
        )
    if quant_rule in {QUANT_RULE_PRESET_LINEAR_FP8, QUANT_RULE_PRESET_LINEAR_W8A8}:
        what_if_expr = linear_expr
    elif quant_rule == QUANT_RULE_CHECKPOINT_IGNORE:
        what_if_expr = f'IF({MEM_CTL_OVERRIDE_IGN_REF}="ON",{linear_expr},"BF16")'
    else:
        what_if_expr = MEM_CTL_QUANT_REF
    return f'=IF({MEM_CTL_QUANT_REF}="{QUANT_MODE_AS_EVALUATED}","{eval_fmt}",{what_if_expr})'


def _exact_dtype_switch_formula(
    values_by_dtype: Sequence[tuple[str, float]],
    c_opts: Sequence[str],
    *,
    divisor: float = 1.0,
) -> str | None:
    """Build exact ``IF($T$4=..., ...)`` formula strictly from Memory-projected per-dtype values."""
    val_map = {str(k).strip().upper(): float(v) / float(divisor) for k, v in values_by_dtype}
    scoped = [opt for opt in c_opts if opt in val_map]
    if not scoped:
        return None
    ordered = (["BF16"] if "BF16" in scoped else []) + [o for o in scoped if o != "BF16"]
    if len(ordered) == 1:
        return f"={val_map[ordered[0]]!r}"
    # If all legal options share the exact same value, no branch needed.
    first_val = val_map[ordered[0]]
    if all(abs(val_map[o] - first_val) <= MIN_DENOM_EPS for o in ordered[1:]):
        return None
    expr = f"{val_map[ordered[-1]]!r}"
    for opt in reversed(ordered[:-1]):
        expr = f'IF({MEM_CTL_CACHE_REF}="{opt}",{val_map[opt]!r},{expr})'
    return f"={expr}"


def _populate_cache_block(
    ws: Worksheet, start_row: int, mem: MemoryCalculatorLedger, c_opts: Sequence[str],
) -> int:
    ws.set_cell(start_row, 1, "2. KV / Indexer / KeyPool / Recurrent State Cache Breakdown", style=STYLE_SECTION)
    ws.merge_range(start_row, 1, start_row, MEMORY_TABLE_END_COL)
    hdr_row = start_row + 1
    headers = (
        "Component", "Cache Kind", "Max Stage Layers", "Bytes/Tok/Layer",
        "Per-Request (MiB)", "Batch Total (GiB)", "Scaling Basis",
    )
    write_headers(ws, hdr_row, headers)
    ws.merge_range(hdr_row, MEMORY_CACHE_BASIS_START_COL, hdr_row, MEMORY_TABLE_END_COL)

    c_start = hdr_row + 1
    opts_txt = "/".join(c_opts)
    for offset, row in enumerate(mem.cache_rows):
        r = c_start + offset
        is_fixed_die = row.component in _FIXED_PER_DIE_COMPONENTS
        kv_scaled = row.component in KV_DTYPE_SENSITIVE_COMPONENTS
        if is_fixed_die:
            rate_formula: str | None = None
            req_formula: str | None = None
            batch_formula: str | None = _exact_dtype_switch_formula(
                row.per_request_bytes_by_dtype, c_opts, divisor=float(GIB_BYTES),
            )
            basis = f"Fixed per-die checkpoint scale buffer (independent of concurrency; {opts_txt})"
        elif row.scales_with_tokens and row.max_stage_layers > 0 and (
            row.bytes_per_token_per_layer > 0.0
            or any(v > 0.0 for _, v in row.bytes_per_token_by_dtype)
        ):
            rate_formula = (
                _exact_dtype_switch_formula(row.bytes_per_token_by_dtype, c_opts)
                if kv_scaled else None
            )
            req_formula = f"=C{r}*D{r}*{MEM_CTL_LOCAL_TOK_REF}/{MIB_BYTES}"
            batch_formula = f"=E{r}*{MEM_CTL_CONC_REF}/{MIB_PER_GIB}"
            basis = (
                f"Scales with Local Cache Tokens & KV Dtype ({opts_txt}, active={mem.cache_dtype})"
                if kv_scaled
                else f"Scales with Local Cache Tokens (independent dtype: {row.cache_kind})"
            )
        else:
            rate_formula = None
            req_formula = _exact_dtype_switch_formula(
                row.per_request_bytes_by_dtype, c_opts, divisor=float(MIB_BYTES),
            )
            batch_formula = f"=E{r}*{MEM_CTL_CONC_REF}/{MIB_PER_GIB}"
            basis = (
                f"Exact Memory-projected hybrid pool / reserve ({opts_txt}, active={mem.cache_dtype})"
                if req_formula else "Fixed per-request state buffer (non-token-scaled)"
            )
        write_row(ws, r, (
            (row.component, None, STYLE_TEXT),
            (row.cache_kind, None, STYLE_TEXT),
            (row.max_stage_layers, None, STYLE_INT),
            (row.bytes_per_token_per_layer, rate_formula, STYLE_FLOAT2),
            (row.per_request_bytes / float(MIB_BYTES), req_formula, STYLE_FLOAT3),
            (row.batch_total_bytes / float(GIB_BYTES), batch_formula, STYLE_FLOAT3),
            (basis, None, STYLE_TEXT),
        ))
        ws.merge_range(r, MEMORY_CACHE_BASIS_START_COL, r, MEMORY_TABLE_END_COL)

    c_end = c_start + len(mem.cache_rows) - 1
    c_tot = c_end + 1
    ws.set_cell(c_tot, 1, "Cache Total (Max Stage)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(c_tot, 1, c_tot, MEMORY_CACHE_MERGE_END_COL)
    ws.set_cell(
        c_tot, MEMORY_CACHE_REQ_COL,
        sum(x.per_request_bytes for x in mem.cache_rows) / float(MIB_BYTES),
        formula=f"=SUM(E{c_start}:E{c_end})" if c_end >= c_start else "=0",
        style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BATCH_COL,
        sum(x.batch_total_bytes for x in mem.cache_rows) / float(GIB_BYTES),
        formula=f"=SUM(F{c_start}:F{c_end})" if c_end >= c_start else "=0",
        style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BASIS_START_COL,
        f"Local tokens={mem.local_cache_tokens}, Concurrency={mem.concurrency}",
        style=STYLE_TOTAL_TEXT,
    )
    ws.merge_range(c_tot, MEMORY_CACHE_BASIS_START_COL, c_tot, MEMORY_TABLE_END_COL)
    return c_tot


def _weight_row_stage_bytes(wrow: Any, stage_idx: int) -> tuple[float, float]:
    """Compute gross ``(payload_bytes, scale_aux_bytes)`` for one weight row on ``stage_idx``."""
    st_rep = int(wrow.stage_repeats[stage_idx]) if stage_idx < len(wrow.stage_repeats) else 0
    if st_rep <= 0:
        return 0.0, 0.0
    if wrow.peak_stage_repeats > 0:
        scale = float(st_rep) / float(wrow.peak_stage_repeats)
        return float(wrow.payload_bytes) * scale, float(wrow.scale_bytes + wrow.aux_bytes) * scale
    spec = _weight_format_storage_spec(wrow.quant_format)
    bpe = float(spec["bytes"])
    grp = int(spec["group"])
    aux = int(spec["aux"])
    row_p = float(wrow.local_rows * math.ceil(wrow.local_cols * bpe) * st_rep)
    row_s = float(
        (wrow.local_rows * math.ceil(wrow.local_cols / grp) * st_rep if grp > 0 else 0)
        + wrow.local_rows * aux * st_rep
    )
    return row_p, row_s


def _stage_expert_hbm_ratios(
    mem: MemoryCalculatorLedger, stage_idx: int,
) -> tuple[float, float, float]:
    """Return ``(payload_hbm_ratio, scale_hbm_ratio, scale_deduct_in_payload)`` for routed experts."""
    st_budget = next((s for s in mem.stage_budgets if int(s.stage) == int(stage_idx)), None)
    if st_budget is None:
        return 1.0, 1.0, 0.0
    gross_non_exp_p = gross_non_exp_s = gross_exp_p = gross_exp_s = 0.0
    for wrow in mem.weight_rows:
        if wrow.is_engram_or_ple or (wrow.is_vision and not mem.include_vision_tower):
            continue
        row_p, row_s = _weight_row_stage_bytes(wrow, stage_idx)
        if row_p <= 0.0 and row_s <= 0.0:
            continue
        if wrow.group == _ROUTED_EXPERT_WEIGHT_GROUP:
            gross_exp_p += row_p
            gross_exp_s += row_s
        else:
            gross_non_exp_p += row_p
            gross_non_exp_s += row_s
    if gross_exp_p <= MIN_DENOM_EPS and gross_exp_s <= MIN_DENOM_EPS:
        return 1.0, 1.0, 0.0
    offload_p = (gross_non_exp_p + gross_exp_p) - float(st_budget.weight_payload_bytes)
    offload_s = (gross_non_exp_s + gross_exp_s) - float(st_budget.weight_scale_aux_bytes)
    if offload_p <= _EXPERT_OFFLOAD_BYTES_EPS and offload_s <= _EXPERT_OFFLOAD_BYTES_EPS:
        return 1.0, 1.0, 0.0
    ratio_p, scale_deduct = 1.0, 0.0
    if gross_exp_p > MIN_DENOM_EPS and offload_p > _EXPERT_OFFLOAD_BYTES_EPS:
        if offload_p <= gross_exp_p:
            ratio_p = max(0.0, min(1.0, (gross_exp_p - offload_p) / gross_exp_p))
        else:
            ratio_p = 0.0
            scale_deduct = (
                max(0.0, min(1.0, (offload_p - gross_exp_p) / gross_exp_s))
                if gross_exp_s > MIN_DENOM_EPS else 0.0
            )
    ratio_s = (
        max(0.0, min(1.0, (gross_exp_s - offload_s) / gross_exp_s))
        if (gross_exp_s > MIN_DENOM_EPS and offload_s > _EXPERT_OFFLOAD_BYTES_EPS)
        else (ratio_p if gross_exp_s <= MIN_DENOM_EPS else 1.0)
    )
    return ratio_p, ratio_s, scale_deduct


def _join_signed_terms(terms: Sequence[str]) -> str:
    if not terms:
        return "0"
    return "".join(term if (idx == 0 or term.startswith("-")) else f"+{term}" for idx, term in enumerate(terms))


def _stage_weight_formulas(
    mem: MemoryCalculatorLedger, stage_idx: int, weight_start_row: int, lookup_ref: str,
) -> tuple[str | None, str | None]:
    """Build live Excel formulas for one PP stage's HBM weight payload & scale+aux (GiB)."""
    if weight_start_row <= 0 or not mem.weight_rows:
        return None, None
    exp_ratio_p, exp_ratio_s, exp_scale_deduct = _stage_expert_hbm_ratios(mem, stage_idx)
    has_exp_offload = (
        exp_ratio_p < 1.0 - MIN_DENOM_EPS
        or exp_ratio_s < 1.0 - MIN_DENOM_EPS
        or exp_scale_deduct > MIN_DENOM_EPS
    )
    payload_terms: list[str] = []
    scale_terms: list[str] = []
    zero_peak_p_terms: list[str] = []
    zero_peak_s_terms: list[str] = []
    known_peak_p = 0.0
    known_peak_s = 0.0
    run_start: int | None = None
    run_end: int | None = None

    def _flush_run() -> None:
        nonlocal run_start, run_end
        if run_start is None or run_end is None:
            return
        if run_start == run_end:
            payload_terms.append(f"K{run_start}")
            scale_terms.append(f"(L{run_start}+M{run_start})")
        else:
            payload_terms.append(f"SUM(K{run_start}:K{run_end})")
            scale_terms.append(f"(SUM(L{run_start}:L{run_end})+SUM(M{run_start}:M{run_end}))")
        run_start = run_end = None

    for idx, wrow in enumerate(mem.weight_rows):
        if wrow.is_engram_or_ple:
            _flush_run()
            continue
        st_rep = int(wrow.stage_repeats[stage_idx]) if stage_idx < len(wrow.stage_repeats) else 0
        if st_rep <= 0:
            _flush_run()
            continue
        wr = weight_start_row + idx
        is_offloaded_exp = has_exp_offload and wrow.group == _ROUTED_EXPERT_WEIGHT_GROUP
        p_scale = exp_ratio_p if is_offloaded_exp else 1.0
        s_scale = exp_ratio_s if is_offloaded_exp else 1.0
        if wrow.peak_stage_repeats > 0:
            rep_ratio = float(st_rep) / float(wrow.peak_stage_repeats)
            if not (wrow.is_vision and not mem.include_vision_tower):
                row_p = float(wrow.payload_bytes) * rep_ratio
                row_s = float(wrow.scale_bytes + wrow.aux_bytes) * rep_ratio
                known_peak_p += row_p * p_scale - (row_s * exp_scale_deduct if is_offloaded_exp else 0.0)
                known_peak_s += row_s * s_scale
        if not wrow.is_vision and not is_offloaded_exp and wrow.peak_stage_repeats == st_rep:
            if run_start is None:
                run_start = run_end = wr
            elif run_end == wr - 1:
                run_end = wr
            else:
                _flush_run()
                run_start = run_end = wr
            continue
        _flush_run()
        if wrow.peak_stage_repeats > 0:
            rep_ratio = float(st_rep) / float(wrow.peak_stage_repeats)
            eff_p = rep_ratio * p_scale
            eff_s = rep_ratio * s_scale
            p_base = f"K{wr}"
            s_base = f"(L{wr}+M{wr})"
            p_term = "" if eff_p <= MIN_DENOM_EPS else (p_base if abs(eff_p - 1.0) <= MIN_DENOM_EPS else f"{p_base}*{eff_p!r}")
            s_term = "" if eff_s <= MIN_DENOM_EPS else (s_base if abs(eff_s - 1.0) <= MIN_DENOM_EPS else f"{s_base}*{eff_s!r}")
            deduct_base = s_base if abs(rep_ratio - 1.0) <= MIN_DENOM_EPS else f"{s_base}*{rep_ratio!r}"
            target_p, target_s = payload_terms, scale_terms
        else:
            p_raw = f"F{wr}*ROUNDUP(G{wr}*VLOOKUP(J{wr},{lookup_ref},2,FALSE),0)*{st_rep}"
            s_raw = (
                f"(IF(VLOOKUP(J{wr},{lookup_ref},3,FALSE)>0,"
                f"F{wr}*ROUNDUP(G{wr}/VLOOKUP(J{wr},{lookup_ref},3,FALSE),0)*{st_rep},0)"
                f"+F{wr}*VLOOKUP(J{wr},{lookup_ref},4,FALSE)*{st_rep})"
            )
            p_term = "" if p_scale <= MIN_DENOM_EPS else (p_raw if abs(p_scale - 1.0) <= MIN_DENOM_EPS else f"({p_raw})*{p_scale!r}")
            s_term = "" if s_scale <= MIN_DENOM_EPS else (s_raw if abs(s_scale - 1.0) <= MIN_DENOM_EPS else f"({s_raw})*{s_scale!r}")
            deduct_base = s_raw
            target_p, target_s = zero_peak_p_terms, zero_peak_s_terms
        if wrow.is_vision:
            p_term = f'IF({MEM_CTL_VIT_REF}="ON",{p_term},0)' if p_term else ""
            s_term = f'IF({MEM_CTL_VIT_REF}="ON",{s_term},0)' if s_term else ""
        if p_term:
            target_p.append(p_term)
        if is_offloaded_exp and exp_scale_deduct > MIN_DENOM_EPS:
            d_term = deduct_base if abs(exp_scale_deduct - 1.0) <= MIN_DENOM_EPS else f"({deduct_base})*{exp_scale_deduct!r}"
            target_p.append(f"-{d_term}")
        if s_term:
            target_s.append(s_term)
    _flush_run()
    if zero_peak_p_terms or zero_peak_s_terms:
        st_budget = next((s for s in mem.stage_budgets if int(s.stage) == int(stage_idx)), None)
        if zero_peak_p_terms:
            v_p = _join_signed_terms(zero_peak_p_terms)
            if st_budget is not None:
                rem_p = int(round(max(0.0, float(st_budget.weight_payload_bytes) - known_peak_p)))
                payload_terms.append(f'IF({MEM_CTL_QUANT_REF}="{QUANT_MODE_AS_EVALUATED}",{rem_p},{v_p})')
            else:
                payload_terms.append(v_p)
        if zero_peak_s_terms:
            v_s = _join_signed_terms(zero_peak_s_terms)
            if st_budget is not None:
                rem_s = int(round(max(0.0, float(st_budget.weight_scale_aux_bytes) - known_peak_s)))
                scale_terms.append(f'IF({MEM_CTL_QUANT_REF}="{QUANT_MODE_AS_EVALUATED}",{rem_s},{v_s})')
            else:
                scale_terms.append(v_s)
    payload_f = f"=({_join_signed_terms(payload_terms)})/{GIB_BYTES}"
    scale_f = f"=({_join_signed_terms(scale_terms)})/{GIB_BYTES}"
    return payload_f, scale_f


def _stage_cache_formula(
    mem: MemoryCalculatorLedger, stage_idx: int, cache_tot_row: int,
) -> str:
    """Build live per-PP-stage cache formula referencing Sheet 3 cache rows (honors $T$4/$T$18/$T$21)."""
    if mem.pp == 1 and cache_tot_row > 0:
        return f"=F{cache_tot_row}"
    if cache_tot_row > len(mem.cache_rows) and mem.cache_rows:
        c_start = cache_tot_row - len(mem.cache_rows)
        terms: list[str] = []
        for idx, crow in enumerate(mem.cache_rows):
            r = c_start + idx
            is_fixed = crow.component in _FIXED_PER_DIE_COMPONENTS
            st_val = float(crow.per_stage_bytes[stage_idx]) if stage_idx < len(crow.per_stage_bytes) else 0.0
            if st_val <= MIN_DENOM_EPS:
                continue
            max_val = float(crow.batch_total_bytes) if is_fixed else float(crow.per_request_bytes)
            if max_val > MIN_DENOM_EPS:
                share = st_val / max_val
                if abs(share - 1.0) <= MIN_DENOM_EPS:
                    terms.append(f"F{r}")
                else:
                    terms.append(f"F{r}*{share!r}")
            elif is_fixed:
                terms.append(f"{st_val / float(GIB_BYTES)!r}")
            elif crow.scales_with_tokens and mem.local_cache_tokens > 0:
                terms.append(
                    f"({st_val / float(GIB_BYTES)!r}*{MEM_CTL_CONC_REF}"
                    f"*{MEM_CTL_LOCAL_TOK_REF}/{mem.local_cache_tokens})"
                )
            else:
                terms.append(f"({st_val / float(GIB_BYTES)!r}*{MEM_CTL_CONC_REF})")
        if terms:
            return f"={'+'.join(terms)}"
        return "=0"
    per_req_bytes = 0.0
    fixed_die_bytes = 0.0
    for crow in mem.cache_rows:
        st_val = float(crow.per_stage_bytes[stage_idx]) if stage_idx < len(crow.per_stage_bytes) else 0.0
        if crow.component in _FIXED_PER_DIE_COMPONENTS:
            fixed_die_bytes += st_val
        else:
            per_req_bytes += st_val
    req_gib = per_req_bytes / float(GIB_BYTES)
    fixed_gib = fixed_die_bytes / float(GIB_BYTES)
    if fixed_gib > 0.0:
        return f"={req_gib!r}*{MEM_CTL_CONC_REF}+{fixed_gib!r}"
    return f"={req_gib!r}*{MEM_CTL_CONC_REF}"


def _populate_stage_budget_block(
    ws: Worksheet, start_row: int, mem: MemoryCalculatorLedger, *,
    cache_tot_row: int = 0, weight_start_row: int = 0, lookup_ref: str = "",
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
    write_headers(ws, hdr_row, headers)

    s_start = hdr_row + 1
    for offset, st in enumerate(mem.stage_budgets):
        r = s_start + offset
        raw_engram_gib = (st.engram_ple_hbm_bytes + st.engram_ple_host_bytes) / float(GIB_BYTES)
        cache_f = _stage_cache_formula(mem, st.stage, cache_tot_row)
        w_payload_f, w_scale_f = _stage_weight_formulas(mem, st.stage, weight_start_row, lookup_ref)
        t24_bytes = float(mem.die_hbm_capacity_bytes) * float(mem.gpu_memory_utilization)
        if abs(float(st.usable_hbm_budget_bytes) - t24_bytes) <= _EXPERT_OFFLOAD_BYTES_EPS:
            stage_hbm_f: str | None = f"={MEM_CTL_USABLE_HBM_REF}"
        elif float(mem.gpu_memory_utilization) > MIN_DENOM_EPS:
            st_cap_gib = float(st.usable_hbm_budget_bytes) / (float(GIB_BYTES) * float(mem.gpu_memory_utilization))
            stage_hbm_f = f"={st_cap_gib!r}*{MEM_CTL_GPU_UTIL_REF}"
        else:
            stage_hbm_f = None
        write_row(ws, r, (
            (f"Stage {st.stage}", None, STYLE_MERGED),
            (st.owned_layers, None, STYLE_INT),
            (st.weight_payload_bytes / float(GIB_BYTES), w_payload_f, STYLE_FLOAT3),
            (st.weight_scale_aux_bytes / float(GIB_BYTES), w_scale_f, STYLE_FLOAT3),
            (st.engram_ple_hbm_bytes / float(GIB_BYTES), f'=IF({MEM_CTL_OFFLOAD_REF}="ON",0,N{r})', STYLE_FLOAT3),
            (st.engram_ple_host_bytes / float(GIB_BYTES), f'=IF({MEM_CTL_OFFLOAD_REF}="ON",N{r},0)', STYLE_FLOAT3),
            (st.cache_bytes / float(GIB_BYTES), cache_f, STYLE_FLOAT3),
            (st.workspace_peak_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.reserve_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.total_hbm_bytes / float(GIB_BYTES), f"=SUM(C{r}:E{r},G{r}:I{r})", STYLE_FLOAT3),
            (st.usable_hbm_budget_bytes / float(GIB_BYTES), stage_hbm_f, STYLE_FLOAT3),
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
        ws.set_cell(
            max_row, col, val,
            formula=f"={agg}({c_let}{s_start}:{c_let}{s_end})" if s_end >= s_start else "=0",
            style=STYLE_TOTAL_NUM,
        )
    ws.set_cell(
        max_row, MEMORY_STAGE_STATUS_COL,
        "PASS" if mem.fits_hbm else "OOM",
        formula=f'=IF(L{max_row}>=0,"PASS","OOM")',
        style=STYLE_ALERT_OK if mem.fits_hbm else STYLE_ALERT_HIGH,
    )
    raw_engram_vals = [(s.engram_ple_hbm_bytes + s.engram_ple_host_bytes) / GIB_BYTES for s in mem.stage_budgets]
    ws.set_cell(
        max_row, MEMORY_STAGE_RAW_ENGRAM_COL,
        max(raw_engram_vals, default=0.0),
        formula=f"=MAX(N{s_start}:N{s_end})" if s_end >= s_start else "=0",
        style=STYLE_TOTAL_NUM,
    )
    return max_row


def _populate_weight_block(
    ws: Worksheet,
    mem: MemoryCalculatorLedger,
    arch: str,
    w_opts: Sequence[str],
    lookup_ref: str,
) -> tuple[int, int]:
    """Render Section 1 per-die Weight + Quant Scale table; return ``(w_start, w_tot_vit)``.

    [Cohesive-Pipeline] Linear worksheet table population keeping Excel row/column formula references coupled.
    """
    ws.set_cell(
        2, 1,
        f"1. Per-Die Weight + Quant Scale Closed-Form Table (Side-by-Side 原生 BF16 vs 量化后; Supported: {'/'.join(w_opts)})",
        style=STYLE_SECTION,
    )
    ws.merge_range(2, 1, 2, MEMORY_TABLE_END_COL)
    rep_col_hdr = f"Peak Stage Rep (S{mem.peak_weight_stage})" if mem.pp > 1 else "Stage Rep"
    w_headers = (
        "Group", "Tensor Name", "Global Rows", "Global Cols", "Shard Rule",
        "Local Rows", "Local Cols", rep_col_hdr, "Quant Rule", "Quant Format",
        "Payload (B)", "MX Scale (B)", "Ch Aux (B)", "Native BF16 (GiB)",
        "Quant Per-Die (GiB)", "Residency & Quant Note",
    )
    write_headers(ws, MEMORY_TABLE_HEADER_ROW, w_headers)

    w_start = MEMORY_TABLE_HEADER_ROW + 1
    peak_exp_p, _, _ = _stage_expert_hbm_ratios(mem, mem.peak_weight_stage)
    full_exp_offload = peak_exp_p <= MIN_DENOM_EPS
    partial_exp_offload = MIN_DENOM_EPS < peak_exp_p < 1.0 - MIN_DENOM_EPS
    exp_offload_frac = max(0.0, min(1.0, 1.0 - peak_exp_p))
    exp_row_indices: list[int] = []
    group_spans: list[list[Any]] = []
    for idx, row in enumerate(mem.weight_rows):
        r = w_start + idx
        extend_span(group_spans, row.group, r, r)
        is_exp_row = row.group == _ROUTED_EXPERT_WEIGHT_GROUP
        if is_exp_row:
            exp_row_indices.append(r)
        if row.is_engram_or_ple:
            res_text = "Host DRAM (Offload)" if mem.engram_host_offload else "NPU HBM (Engram/PLE)"
            res_formula: str | None = f'=IF({MEM_CTL_OFFLOAD_REF}="ON","Host DRAM (Offload)","NPU HBM (Engram/PLE)")'
        elif row.is_vision:
            res_text = "NPU HBM (Vision ViT)" if mem.include_vision_tower else "Excluded Vision (Text-Only)"
            res_formula = f'=IF({MEM_CTL_VIT_REF}="ON","NPU HBM (Vision ViT)","Excluded Vision (Text-Only)")'
        elif is_exp_row and full_exp_offload:
            res_text = "Host DRAM (Expert DDR Offload)"
            res_formula = None
        elif is_exp_row and partial_exp_offload:
            off_pct = int(round(exp_offload_frac * _OFFLOAD_PCT_SCALE))
            res_text = f"NPU HBM ({_OFFLOAD_PCT_SCALE - off_pct}% HBM / {off_pct}% DDR Offload)"
            res_formula = None
        else:
            res_text = f"NPU HBM ({row.quant_note})" if row.quant_note else "NPU HBM"
            res_formula = None
        is_prepared = row.group == _PREPARED_WEIGHT_GROUP
        q_formula = None if is_prepared else _quant_format_formula(row.quant_rule, row.quant_format, arch)
        if is_prepared:
            payload_formula: str | None = None
            scale_formula: str | None = None
            aux_formula: str | None = None
        else:
            vlookup_p = f"F{r}*ROUNDUP(G{r}*VLOOKUP(J{r},{lookup_ref},2,FALSE),0)*H{r}"
            vlookup_s = (
                f"IF(VLOOKUP(J{r},{lookup_ref},3,FALSE)>0,"
                f"F{r}*ROUNDUP(G{r}/VLOOKUP(J{r},{lookup_ref},3,FALSE),0)*H{r},0)"
            )
            vlookup_a = f"F{r}*VLOOKUP(J{r},{lookup_ref},4,FALSE)*H{r}"
            payload_formula = f'=IF({MEM_CTL_QUANT_REF}="{QUANT_MODE_AS_EVALUATED}",{int(row.payload_bytes)},{vlookup_p})'
            scale_formula = f'=IF({MEM_CTL_QUANT_REF}="{QUANT_MODE_AS_EVALUATED}",{int(row.scale_bytes)},{vlookup_s})'
            aux_formula = f'=IF({MEM_CTL_QUANT_REF}="{QUANT_MODE_AS_EVALUATED}",{int(row.aux_bytes)},{vlookup_a})'
        native_gib = float(row.native_bf16_bytes) / float(GIB_BYTES)
        display_rule = (
            QUANT_RULE_PRESET_LINEAR_W8A8
            if (arch == "a23" and row.quant_rule == QUANT_RULE_PRESET_LINEAR_FP8)
            else row.quant_rule
        )
        write_row(ws, r, (
            (row.name, None, STYLE_TEXT),
            (row.global_rows, None, STYLE_INT),
            (row.global_cols, None, STYLE_INT),
            (row.shard_rule, None, STYLE_TEXT),
            (row.local_rows, None, STYLE_INT),
            (row.local_cols, None, STYLE_INT),
            (row.peak_stage_repeats, None, STYLE_INT),
            (display_rule, None, STYLE_TEXT),
            (row.quant_format, q_formula, STYLE_INPUT_TEXT if q_formula else STYLE_TEXT),
            (row.payload_bytes, payload_formula, STYLE_INT),
            (row.scale_bytes, scale_formula, STYLE_INT),
            (row.aux_bytes, aux_formula, STYLE_INT),
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

    def _exp_rows_sum_expr(col_let: str) -> str:
        if not exp_row_indices:
            return ""
        ranges: list[tuple[int, int]] = []
        s_r = e_r = exp_row_indices[0]
        for rr in exp_row_indices[1:]:
            if rr == e_r + 1:
                e_r = rr
            else:
                ranges.append((s_r, e_r))
                s_r = e_r = rr
        ranges.append((s_r, e_r))
        parts = [f"{col_let}{a}" if a == b else f"SUM({col_let}{a}:{col_let}{b})" for a, b in ranges]
        return "+".join(parts)

    def _sumifs_row(
        r_idx: int, label: str, crit: str, pred_fn: Any, note_txt: str,
        t_st: int, n_st: int, *, exp_delta_sign: float = 0.0,
    ) -> None:
        ws.set_cell(r_idx, 1, label, style=t_st)
        ws.merge_range(r_idx, 1, r_idx, MEMORY_WEIGHT_MERGE_END_COL)
        matched = [x for x in mem.weight_rows if pred_fn(x)]
        exp_matched = [x for x in mem.weight_rows if x.group == _ROUTED_EXPERT_WEIGHT_GROUP]
        adj_factor = exp_delta_sign * exp_offload_frac if (partial_exp_offload and exp_delta_sign != 0.0) else 0.0
        vals = (
            sum(x.payload_bytes for x in matched) + adj_factor * sum(x.payload_bytes for x in exp_matched),
            sum(x.scale_bytes for x in matched) + adj_factor * sum(x.scale_bytes for x in exp_matched),
            sum(x.aux_bytes for x in matched) + adj_factor * sum(x.aux_bytes for x in exp_matched),
            (sum(x.native_bf16_bytes for x in matched) + adj_factor * sum(x.native_bf16_bytes for x in exp_matched)) / float(GIB_BYTES),
            (sum(x.per_die_bytes for x in matched) + adj_factor * sum(x.per_die_bytes for x in exp_matched)) / float(GIB_BYTES),
        )
        for offset, tot_v in enumerate(vals):
            col = MEMORY_WEIGHT_SUM_START_COL + offset
            c_let = column_letter(col)
            if w_end >= w_start:
                base_f = f'SUMIFS({c_let}{w_start}:{c_let}{w_end},$P${w_start}:$P${w_end},"{crit}")'
                exp_e = _exp_rows_sum_expr(c_let) if abs(adj_factor) > MIN_DENOM_EPS else ""
                if exp_e:
                    op_sign = "+" if adj_factor > 0.0 else "-"
                    f_str = f"={base_f}{op_sign}({exp_e})*{abs(adj_factor)!r}"
                else:
                    f_str = f"={base_f}"
            else:
                f_str = "=0"
            ws.set_cell(r_idx, col, tot_v, formula=f_str, style=n_st)
        ws.set_cell(r_idx, MEMORY_TABLE_END_COL, note_txt, style=t_st)

    _sumifs_row(
        w_tot_hbm, "NPU HBM Weights Total (GiB, 仅计入 NPU HBM 驻留张量)", "NPU HBM*",
        lambda x: (
            not (x.is_engram_or_ple and mem.engram_host_offload)
            and not (x.is_vision and not mem.include_vision_tower)
            and not (full_exp_offload and x.group == _ROUTED_EXPERT_WEIGHT_GROUP)
        ),
        f"压缩比 {mem.weight_compression_ratio:.2f}x (vs 原生 BF16)", STYLE_TOTAL_TEXT, STYLE_TOTAL_NUM,
        exp_delta_sign=-1.0,
    )
    _sumifs_row(
        w_tot_host, "Host DRAM Offload Total (GiB, Engram/PLE 卸载至主机内存)", "Host DRAM*",
        lambda x: (x.is_engram_or_ple and mem.engram_host_offload) or (full_exp_offload and x.group == _ROUTED_EXPERT_WEIGHT_GROUP),
        "不计入 NPU HBM 预算", STYLE_SUBTOTAL_TEXT, STYLE_SUBTOTAL_NUM,
        exp_delta_sign=1.0,
    )
    _sumifs_row(
        w_tot_vit, "Excluded Vision Tower Total (GiB, 纯文本默认不加载 ViT)", "Excluded*",
        lambda x: x.is_vision and not mem.include_vision_tower,
        "切换 T7=ON 即刻计入 NPU HBM", STYLE_SUBTOTAL_TEXT, STYLE_SUBTOTAL_NUM,
    )
    return w_start, w_tot_vit


def populate_memory_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    """Populate Sheet 3 (``Memory_Quant_Calculator``) directly from ``DieResidentReceipt`` ledger."""
    mem = bundle.memory_ledger
    arch, w_opts, c_opts, lookup_order = _device_format_scope(mem)
    apply_col_widths(ws, MEMORY_COL_WIDTHS)
    ws.freeze_panes(MEMORY_FREEZE_ROWS, 2)
    write_banner(
        ws, 1, MEMORY_COLS,
        f"Closed-Form Memory / Quant / Cache / Engram-PLE / PP HBM Calculator | {mem.model_name} @ {mem.hardware_sku} | {mem.weight_quant_display or mem.weight_quant}",
        STYLE_TITLE,
    )
    lookup_last = _populate_memory_control_panel(ws, mem, w_opts, c_opts, lookup_order)
    lookup_ref = f"$U${MEMORY_LOOKUP_FIRST_ROW}:$X${lookup_last}"
    w_start, w_tot_vit = _populate_weight_block(ws, mem, arch, w_opts, lookup_ref)
    c_tot = _populate_cache_block(ws, w_tot_vit + 2, mem, c_opts)
    max_stage_row = _populate_stage_budget_block(
        ws, c_tot + 2, mem,
        cache_tot_row=c_tot, weight_start_row=w_start, lookup_ref=lookup_ref,
    )
    summary_row = max_stage_row + 2
    ws.set_cell(summary_row, 1, "4. Bottleneck Concurrency & Receipt Bijection Summary", style=STYLE_SECTION)
    ws.merge_range(summary_row, 1, summary_row, MEMORY_TABLE_END_COL)
    write_row(ws, summary_row + 1, (
        ("Max HBM Concurrency / DP", None, STYLE_SUBTOTAL_TEXT),
        (mem.max_hbm_concurrency_per_dp, None, STYLE_INT),
        ("Per-Request Cache Max Stage (MiB)", None, STYLE_SUBTOTAL_TEXT),
        (mem.per_request_cache_bytes_max_stage / float(MIB_BYTES), None, STYLE_FLOAT3),
        ("Max Stage HBM (GiB)", None, STYLE_SUBTOTAL_TEXT),
        (mem.max_stage_hbm_bytes / float(GIB_BYTES), f"=J{max_stage_row}", STYLE_FLOAT3),
        ("Min Headroom (GiB)", None, STYLE_SUBTOTAL_TEXT),
        (
            min((s.headroom_bytes for s in mem.stage_budgets), default=0.0) / float(GIB_BYTES),
            f"=L{max_stage_row}",
            STYLE_FLOAT3,
        ),
    ))


__all__ = ["MEMORY_COLS", "QUANT_MODE_AS_EVALUATED", "populate_memory_sheet"]
