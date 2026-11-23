"""Comprehensive verification of 4-sheet A:AA workbook geometry, formulas, memory, and depth.

[Role] Verify zero-``AB..AF`` Sheet 2 (``max_column == 27``), zero hidden sheets,
  strict OOXML primitive cell guards, direct coalesced Sheet 2 ``A:AA`` span
  formulas on Sheet 4 (plus automatic visible ``A:N`` Group Attribution Table
  fallback on fragmented >255-span workbooks), intermediate stage ``C``
  conservation before card materiality filtering, Sheet 3 ``As evaluated``
  default mode + MXFP8/shared-DP/A2-A3 ISA parity, Sheet 1 Roofline &
  unconstrained SLA rendering, Sheet 2 ``AttnResFwd`` depth-weighted slicing +
  ``RecoverSSMStateCommit`` ``B=69`` grouping, typed ``CoDesignOpportunity``
  cards, and post-synthesis ``per_layer_timings`` trace honesty.
[Invariants] Pure ``tier0`` stdlib XML inspection and live Excel formula
  evaluation under ``tmp_path``; zero third-party OOXML libraries, zero
  subprocesses, zero host-state pollution.
[Timeouts & Locks] In-memory execution only; no locks or network I/O.
[Side-Effect Policy] Writes temporary workbooks and traces exclusively inside
  pytest ``tmp_path``.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.export import (
    BINDING_STATE_BOUND,
    BINDING_STATE_UNBOUND_FORMULA,
    MODE_MODELING,
    SHEET_BREAKDOWN,
    SHEET_MEMORY,
    SHEET_PENDING_KERNELS,
    SHEET_SUMMARY,
    align_before_after_operators,
    build_export_bundle,
    compose_export_case,
    compute_memory_quant_ledger,
    export_comparison_xlsx,
    export_msprof_trace,
)
from mindie_sim.export.action_plan import build_comparison_plan
from mindie_sim.export.bundle import _collapse_micro_split_ops
from mindie_sim.export.comparison import enrich_comparison_plan_stage_receipts
from mindie_sim.export.models import LayerArchetypeSpec
from mindie_sim.export.ooxml import (
    EXCEL_MAX_FORMULA_ARGS,
    EXCEL_MAX_FORMULA_CHARS,
    NS_MAIN,
    Worksheet,
    _parse_cell_ref,
)
from mindie_sim.export.xlsx_breakdown import CMP_COLS
from mindie_sim.export.xlsx_memory import MEMORY_COLS, QUANT_MODE_AS_EVALUATED
from mindie_sim.export.xlsx_pending import PENDING_COLS
from mindie_sim.export.xlsx_summary import SUMMARY_COLS
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.results import OpOccurrence, StepRequest
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.scenario import (
    CoDesignOpportunity,
    CrossHwOffloadSpec,
    MegaFusionBound,
    SpaceTimeTradeoffSpec,
)
from mindie_sim.schema.stack import PreparedWeightSpec

pytestmark = pytest.mark.tier0

NS_PREFIX = f"{{{NS_MAIN}}}"
_DIM_END_RE = re.compile(r":([A-Z]+)(\d+)$")
_CELL_REF_SCAN_RE = re.compile(r"(?:'([^']+)'!|([A-Za-z_][A-Za-z0-9_]*)!)?\$?([A-Z]+)\$?(\d+)(?::\$?([A-Z]+)\$?(\d+))?")


def _inspect_xlsx(
    path: Path,
) -> tuple[list[tuple[str, dict[str, str]]], dict[str, ET.Element], list[str]]:
    with zipfile.ZipFile(path, "r") as zf:
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = [
            "".join(t.text or "" for t in si.iter(f"{NS_PREFIX}t"))
            for si in sst_root.findall(f"{NS_PREFIX}si")
        ]
        sheet_meta: list[tuple[str, dict[str, str]]] = []
        roots: dict[str, ET.Element] = {}
        for idx, sh in enumerate(wb_root.findall(f".//{NS_PREFIX}sheet"), start=1):
            name = sh.attrib["name"]
            sheet_meta.append((name, dict(sh.attrib)))
            roots[name] = ET.fromstring(zf.read(f"xl/worksheets/sheet{idx}.xml"))
        return sheet_meta, roots, shared


def _sheet_cells(root: ET.Element, shared: list[str]) -> dict[str, tuple[str | None, str | None]]:
    out: dict[str, tuple[str | None, str | None]] = {}
    for cell in root.iter(f"{NS_PREFIX}c"):
        ref = cell.attrib["r"]
        ctype = cell.attrib.get("t")
        f_el = cell.find(f"{NS_PREFIX}f")
        v_el = cell.find(f"{NS_PREFIX}v")
        val = None if v_el is None else (shared[int(v_el.text or "0")] if ctype == "s" else v_el.text)
        out[ref] = (val, f_el.text if f_el is not None else None)
    return out


def _sheet_max_col_from_xml(root: ET.Element) -> int:
    max_c = 0
    dim = root.find(f"{NS_PREFIX}dimension")
    if dim is not None:
        m = _DIM_END_RE.search(dim.attrib.get("ref", ""))
        if m is not None:
            _, c = _parse_cell_ref(f"{m.group(1)}{m.group(2)}")
            max_c = max(max_c, c)
    for col in root.iter(f"{NS_PREFIX}col"):
        max_c = max(max_c, int(col.attrib.get("max", "0")))
    for cell in root.iter(f"{NS_PREFIX}c"):
        _, c = _parse_cell_ref(cell.attrib["r"])
        max_c = max(max_c, c)
    for mc in root.iter(f"{NS_PREFIX}mergeCell"):
        parts = mc.attrib.get("ref", "").split(":")
        for p in parts:
            if p:
                _, c = _parse_cell_ref(p)
                max_c = max(max_c, c)
    return max_c


def _split_excel_args(body: str) -> list[str]:
    args: list[str] = []
    depth = 0
    in_str = False
    start = 0
    for idx, ch in enumerate(body):
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                args.append(body[start:idx].strip())
                start = idx + 1
    args.append(body[start:].strip())
    return args


def _eval_workbook_cell(
    sheet: str,
    ref: str,
    all_sheets: dict[str, dict[str, tuple[str | None, str | None]]],
    *,
    overrides: dict[tuple[str, str], object] | None = None,
    memo: dict[tuple[str, str], object] | None = None,
) -> object:
    """Minimal pure-stdlib Excel formula evaluator for Sheet 1/2/3/4 reconciliation tests."""
    cache = memo if memo is not None else {}
    norm_ref = ref.replace("$", "").upper()
    key = (sheet, norm_ref)
    if overrides and key in overrides:
        return overrides[key]
    if key in cache:
        return cache[key]
    raw_val, formula = all_sheets.get(sheet, {}).get(norm_ref, (None, None))
    if formula is None:
        val: object = _parse_scalar(raw_val)
        cache[key] = val
        return val
    res = _eval_expr(formula.strip(), sheet, all_sheets, overrides=overrides, memo=cache)
    cache[key] = res
    return res


def _parse_scalar(raw: str | None) -> object:
    if raw is None or raw == "":
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return raw


def _range_coords(r1: str, r2: str) -> list[str]:
    row1, col1 = _parse_cell_ref(r1.replace("$", ""))
    row2, col2 = _parse_cell_ref(r2.replace("$", ""))
    from mindie_sim.export.ooxml import column_letter
    return [
        f"{column_letter(c)}{r}"
        for r in range(row1, row2 + 1)
        for c in range(col1, col2 + 1)
    ]


def _match_criteria(val: object, crit: object) -> bool:
    c_str = str(crit)
    v_str = "" if val is None else str(val)
    if "*" in c_str:
        pat = "^" + re.escape(c_str).replace(r"\*", ".*") + "$"
        return re.match(pat, v_str) is not None
    return v_str == c_str


def _eval_call(
    fn: str,
    args_raw: list[str],
    sheet: str,
    all_sheets: dict[str, dict[str, tuple[str | None, str | None]]],
    *,
    overrides: dict[tuple[str, str], object] | None,
    memo: dict[tuple[str, str], object],
) -> object:
    u_fn = fn.upper()
    if u_fn == "IF":
        cond = _eval_expr(args_raw[0], sheet, all_sheets, overrides=overrides, memo=memo)
        return _eval_expr(args_raw[1 if cond else 2], sheet, all_sheets, overrides=overrides, memo=memo)
    if u_fn == "SUM":
        total = 0.0
        for arg in args_raw:
            m = _CELL_REF_SCAN_RE.fullmatch(arg.strip())
            if m and m.group(5):
                tgt_s = m.group(1) or m.group(2) or sheet
                for cell in _range_coords(f"{m.group(3)}{m.group(4)}", f"{m.group(5)}{m.group(6)}"):
                    v = _eval_workbook_cell(tgt_s, cell, all_sheets, overrides=overrides, memo=memo)
                    if isinstance(v, (int, float)):
                        total += float(v)
            else:
                v = _eval_expr(arg, sheet, all_sheets, overrides=overrides, memo=memo)
                if isinstance(v, (int, float)):
                    total += float(v)
        return total
    if u_fn == "SUMIF":
        m_c = _CELL_REF_SCAN_RE.fullmatch(args_raw[0].strip())
        m_s = _CELL_REF_SCAN_RE.fullmatch(args_raw[2].strip())
        assert m_c is not None and m_s is not None
        cs = m_c.group(1) or m_c.group(2) or sheet
        ss = m_s.group(1) or m_s.group(2) or sheet
        crit = _eval_expr(args_raw[1], sheet, all_sheets, overrides=overrides, memo=memo)
        c_cells = _range_coords(f"{m_c.group(3)}{m_c.group(4)}", f"{m_c.group(5)}{m_c.group(6)}")
        s_cells = _range_coords(f"{m_s.group(3)}{m_s.group(4)}", f"{m_s.group(5)}{m_s.group(6)}")
        tot = 0.0
        for cc, sc in zip(c_cells, s_cells):
            if _match_criteria(_eval_workbook_cell(cs, cc, all_sheets, overrides=overrides, memo=memo), crit):
                sv = _eval_workbook_cell(ss, sc, all_sheets, overrides=overrides, memo=memo)
                if isinstance(sv, (int, float)):
                    tot += float(sv)
        return tot
    if u_fn == "SUMIFS":
        m_s = _CELL_REF_SCAN_RE.fullmatch(args_raw[0].strip())
        m_c = _CELL_REF_SCAN_RE.fullmatch(args_raw[1].strip())
        assert m_s is not None and m_c is not None
        ss = m_s.group(1) or m_s.group(2) or sheet
        cs = m_c.group(1) or m_c.group(2) or sheet
        crit = _eval_expr(args_raw[2], sheet, all_sheets, overrides=overrides, memo=memo)
        s_cells = _range_coords(f"{m_s.group(3)}{m_s.group(4)}", f"{m_s.group(5)}{m_s.group(6)}")
        c_cells = _range_coords(f"{m_c.group(3)}{m_c.group(4)}", f"{m_c.group(5)}{m_c.group(6)}")
        tot = 0.0
        for sc, cc in zip(s_cells, c_cells):
            if _match_criteria(_eval_workbook_cell(cs, cc, all_sheets, overrides=overrides, memo=memo), crit):
                sv = _eval_workbook_cell(ss, sc, all_sheets, overrides=overrides, memo=memo)
                if isinstance(sv, (int, float)):
                    tot += float(sv)
        return tot
    if u_fn == "VLOOKUP":
        needle = str(_eval_expr(args_raw[0], sheet, all_sheets, overrides=overrides, memo=memo))
        m_t = _CELL_REF_SCAN_RE.fullmatch(args_raw[1].strip())
        assert m_t is not None
        ts = m_t.group(1) or m_t.group(2) or sheet
        col_offset = int(float(str(_eval_expr(args_raw[2], sheet, all_sheets, overrides=overrides, memo=memo))))
        r1, c1 = _parse_cell_ref(f"{m_t.group(3)}{m_t.group(4)}")
        r2, _ = _parse_cell_ref(f"{m_t.group(5)}{m_t.group(6)}")
        from mindie_sim.export.ooxml import column_letter
        k_col = column_letter(c1)
        v_col = column_letter(c1 + col_offset - 1)
        for rr in range(r1, r2 + 1):
            if str(_eval_workbook_cell(ts, f"{k_col}{rr}", all_sheets, overrides=overrides, memo=memo)) == needle:
                return _eval_workbook_cell(ts, f"{v_col}{rr}", all_sheets, overrides=overrides, memo=memo)
        raise KeyError(f"vlookup_na:{needle}")
    if u_fn == "ROUNDUP":
        num = float(str(_eval_expr(args_raw[0], sheet, all_sheets, overrides=overrides, memo=memo)))
        return float(math.ceil(num - 1e-12))
    if u_fn == "MAX":
        return max(float(str(_eval_expr(a, sheet, all_sheets, overrides=overrides, memo=memo))) for a in args_raw)
    raise ValueError(f"unsupported_excel_fn:{fn}")


def _eval_expr(
    expr: str,
    sheet: str,
    all_sheets: dict[str, dict[str, tuple[str | None, str | None]]],
    *,
    overrides: dict[tuple[str, str], object] | None,
    memo: dict[tuple[str, str], object],
) -> object:
    text = expr.strip()
    if text.startswith("="):
        text = text[1:].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"' and '"' not in text[1:-1]:
        return text[1:-1]
    # Replace outermost Excel function calls first.
    while True:
        m = re.search(r"\b(SUMIFS|SUMIF|VLOOKUP|ROUNDUP|SUM|MAX|IF)\(", text)
        if m is None:
            break
        start = m.start()
        p_open = m.end() - 1
        depth = 0
        in_str = False
        p_close = -1
        for i in range(p_open, len(text)):
            ch = text[i]
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        p_close = i
                        break
        assert p_close > p_open
        res = _eval_call(
            m.group(1), _split_excel_args(text[p_open + 1:p_close]),
            sheet, all_sheets, overrides=overrides, memo=memo,
        )
        rep = f'"{res}"' if isinstance(res, str) else repr(float(res))
        text = text[:start] + rep + text[p_close + 1:]
    # Resolve remaining single cell references outside string literals.
    out_parts: list[str] = []
    pos = 0
    in_str = False
    while pos < len(text):
        if text[pos] == '"':
            in_str = not in_str
            out_parts.append('"')
            pos += 1
            continue
        if not in_str:
            m = _CELL_REF_SCAN_RE.match(text, pos)
            if m and not m.group(5):
                tgt_s = m.group(1) or m.group(2) or sheet
                val = _eval_workbook_cell(
                    tgt_s, f"{m.group(3)}{m.group(4)}", all_sheets, overrides=overrides, memo=memo,
                )
                out_parts.append(f'"{val}"' if isinstance(val, str) else repr(float(val)))
                pos = m.end()
                continue
        out_parts.append(text[pos])
        pos += 1
    py_expr = "".join(out_parts)
    py_expr = re.sub(r"(?<![<>!=])=(?!=)", "==", py_expr)
    return eval(py_expr, {"__builtins__": {}}, {})


def test_ooxml_geometry_no_hidden_sheets_no_ab_af_and_primitive_cell_guard(tmp_path):
    ws = Worksheet("GuardCheck")
    with pytest.raises(TypeError, match="non_primitive_xlsx_cell_value"):
        ws.set_cell(1, 1, {"bad": "dict"})
    with pytest.raises(TypeError, match="non_primitive_xlsx_cell_value"):
        ws.set_cell(1, 1, ["bad", "list"])
    with pytest.raises(ValueError, match="non_finite_xlsx_cell_value"):
        ws.set_cell(1, 1, float("nan"))
    ws.set_cell(1, 28, "overflow")
    with pytest.raises(ValueError, match="worksheet_column_limit_exceeded"):
        ws.validate_max_column(CMP_COLS)

    raw_b, req = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=2, tp=2, ep=4, dcp=2, prompt_tokens=1024, output_tokens=64,
        concurrency=4, mtp_proposal_tokens=3, mtp_acceptance_length=3.0, feature_track="merged",
    )
    raw_a, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=2, tp=2, ep=4, dcp=2, prompt_tokens=1024, output_tokens=64,
        concurrency=4, mtp_proposal_tokens=3, mtp_acceptance_length=3.0, feature_track="expected_ideal",
    )
    b_bef = build_export_bundle(raw_b, request=req)
    b_aft = build_export_bundle(raw_a, request=req)
    out = tmp_path / "pure_4sheet.xlsx"
    export_comparison_xlsx(b_bef, out, compare_source=b_aft)

    sheet_meta, roots, shared = _inspect_xlsx(out)
    assert [name for name, _ in sheet_meta] == [
        SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY, SHEET_PENDING_KERNELS,
    ]
    for _, attrs in sheet_meta:
        assert "state" not in attrs, f"Hidden sheet forbidden: {attrs}"

    assert _sheet_max_col_from_xml(roots[SHEET_SUMMARY]) <= SUMMARY_COLS
    assert _sheet_max_col_from_xml(roots[SHEET_BREAKDOWN]) == CMP_COLS
    assert _sheet_max_col_from_xml(roots[SHEET_MEMORY]) <= MEMORY_COLS
    assert _sheet_max_col_from_xml(roots[SHEET_PENDING_KERNELS]) <= PENDING_COLS

    s2_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    assert s2_cells["AA2"][0] == "绑定状态 (Binding State)"
    assert not any(ref.startswith(("AB", "AC", "AD", "AE", "AF")) for ref in s2_cells)

    all_text = "\n".join(shared)
    for forbidden in (
        "Lean 4", "Lean L*", "@r0.p0.d0", "Cert=True", "($B$", "($G$", "(B8)", "(G6)",
        "merged", "expected_ideal", "unmerged_pr", "raw_eager", "cannbot", "{", "}",
    ):
        assert forbidden not in all_text, f"Forbidden token {forbidden!r} leaked into workbook"


def test_sheet4_coalesced_span_reconciliation_stage_c_and_visible_fallback(tmp_path):
    raw_base, req_base = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=1, prompt_tokens=512, output_tokens=32, concurrency=2,
    )
    tmpl = build_export_bundle(raw_base, request=req_base, apply_target_envelope=False)
    proto = tmpl.operators[0]

    def _op(oid: str, kern: str, dur: float, atom: str, *, aid: str = "target_arch_0",
            title: str = "Target Layers (×1)", exp: float | None = None,
            is_ov: bool = False, fe: dict | None = None, needs_new: bool = False,
            absorbed: tuple[str, ...] = (), binding: str = BINDING_STATE_BOUND):
        e = 0.0 if is_ov else (dur if exp is None else exp)
        return replace(
            proto, occurrence=OpOccurrence(0, 0, 0, oid), op_id=oid, archetype_id=aid,
            archetype_title=title, repeat_count=1, ascend_kernel=kern, logical_role=oid,
            dur_us=dur, exposed_dur_us=e, is_overlapped=is_ov, l_star_lean_us=dur * 0.5,
            gap_us=max(0.0, dur * 0.5), semantic_atoms=(atom,), formula_eval=fe,
            needs_new_kernel=needs_new, absorbed_source_ops=absorbed, binding_state=binding,
            predecessors=(),
        )

    # Case A: Split pre-edit C preserved BEFORE >=0.050 ms filtering (including a residual row!).
    b_ops = (
        _op("l0.cfg_op", "CfgOp", 2000.0, "a.cfg"),
        _op("l0.fwk_op", "FwkOp", 1500.0, "a.fwk"),
        _op("l0.kern_op", "KernOp", 3000.0, "a.kern"),
        _op("l0.res_split", "ResOp", 1200.0, "a.res"),
    )
    a_ops = (
        _op("l0.cfg_op", "CfgOp", 1400.0, "a.cfg"),  # pure config delta = 0.600 ms
        _op("l0.fwk_op", "FwkOp", 1500.0, "a.fwk", exp=0.0, is_ov=True),  # framework masking = 1.500 ms
        _op("l0.kern_op", "KernOp", 1800.0, "a.kern", fe={"after_pre_edit_exposed_us": 2600.0},
            binding=BINDING_STATE_UNBOUND_FORMULA),  # B-C=0.400ms (config), C-A=0.800ms (kernel)
        _op("l0.res_split", "ResOp", 990.0, "a.res", fe={"after_pre_edit_exposed_us": 1000.0},
            binding=BINDING_STATE_UNBOUND_FORMULA),  # B-C=0.200ms (config!), C-A=0.010ms (<0.050ms -> residual!)
    )
    b_step = sum(o.exposed_dur_us for o in b_ops)
    a_step = sum(o.exposed_dur_us for o in a_ops)
    arch0 = (LayerArchetypeSpec("target_arch_0", "target", "Target Layers (×1)", "L0", 1, (0,), "L0", 0),)
    b_bun = replace(
        tmpl, operators=b_ops, archetypes=arch0, step_span_us=b_step,
        cluster_concurrency=2, n_tok_per_die=1.0, us_per_tok_die=b_step / 1.0,
        throughput_per_die_tokens_per_s=1e6 / b_step,
        throughput_tokens_per_s=2e6 / b_step,
        memory_ledger=replace(tmpl.memory_ledger, concurrency=2),
    )
    a_bun = replace(
        tmpl, operators=a_ops, archetypes=arch0, step_span_us=a_step,
        cluster_concurrency=4, n_tok_per_die=2.0, us_per_tok_die=a_step / 2.0,
        throughput_per_die_tokens_per_s=2e6 / a_step,
        throughput_tokens_per_s=4e6 / a_step,
        memory_ledger=replace(tmpl.memory_ledger, concurrency=4),
        analysis_mode=MODE_MODELING, has_unbound_formula_ops=True,
    )
    aligned = align_before_after_operators(b_bun, a_bun)
    plan = enrich_comparison_plan_stage_receipts(
        build_comparison_plan(b_bun, a_bun, aligned), b_bun, a_bun, aligned,
    )
    res_acc = next(a for a in plan.group_accounting if a.bucket == "residual")
    assert res_acc.after_pre_edit_ms == pytest.approx(1.0, abs=1e-9)
    assert res_acc.config_share_ms == pytest.approx(0.2, abs=1e-9)
    assert res_acc.kernel_share_ms == pytest.approx(0.01, abs=1e-9)

    out_direct = tmp_path / "direct_spans.xlsx"
    export_comparison_xlsx(b_bun, out_direct, compare_source=a_bun)
    _, roots, shared = _inspect_xlsx(out_direct)
    all_sheets = {name: _sheet_cells(root, shared) for name, root in roots.items()}
    s4 = all_sheets[SHEET_PENDING_KERNELS]
    assert "[1] 配置与并行" in (s4["A4"][0] or "")
    assert "[2] 框架图与多流改造" in (s4["A5"][0] or "")
    assert "[3] 新算子开发与 Tiling 调优" in (s4["A6"][0] or "")
    assert "[4] 残差与物理下界算子" in (s4["A7"][0] or "")
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "F3", all_sheets) == pytest.approx(0.0, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "F9", all_sheets) == pytest.approx(0.0, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "D10", all_sheets) == pytest.approx(0.0, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "E10", all_sheets) == pytest.approx(0.0, abs=1e-4)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "B4", all_sheets) == pytest.approx(1.2, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "B5", all_sheets) == pytest.approx(1.5, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "B6", all_sheets) == pytest.approx(0.8, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "B7", all_sheets) == pytest.approx(0.01, abs=1e-6)

    # Case B: Pathologically fragmented >255 disjoint spans triggers visible A:N fallback table!
    frag_b, frag_a = [], []
    for i in range(260):
        frag_b.append(_op(f"l0.cfg_{i}", f"Cfg{i}", 20.0, f"c.{i}"))
        frag_a.append(_op(f"l0.cfg_{i}", f"Cfg{i}", 15.0, f"c.{i}"))
        frag_b.append(_op(f"l0.fwk_{i}", f"Fwk{i}", 10.0, f"f.{i}"))
        frag_a.append(_op(f"l0.fwk_{i}", f"Fwk{i}", 10.0, f"f.{i}", exp=0.0, is_ov=True))
    fb_step = sum(o.exposed_dur_us for o in frag_b)
    fa_step = sum(o.exposed_dur_us for o in frag_a)
    fb_bun = replace(b_bun, operators=tuple(frag_b), step_span_us=fb_step, us_per_tok_die=fb_step / 1.0,
                     throughput_per_die_tokens_per_s=1e6 / fb_step, throughput_tokens_per_s=2e6 / fb_step)
    fa_bun = replace(a_bun, operators=tuple(frag_a), step_span_us=fa_step, us_per_tok_die=fa_step / 2.0,
                     throughput_per_die_tokens_per_s=2e6 / fa_step, throughput_tokens_per_s=4e6 / fa_step)
    out_frag = tmp_path / "fragmented_fallback.xlsx"
    export_comparison_xlsx(fb_bun, out_frag, compare_source=fa_bun)
    f_meta, f_roots, f_shared = _inspect_xlsx(out_frag)
    for _, attrs in f_meta:
        assert "state" not in attrs
    assert _sheet_max_col_from_xml(f_roots[SHEET_BREAKDOWN]) == CMP_COLS
    assert _sheet_max_col_from_xml(f_roots[SHEET_PENDING_KERNELS]) == PENDING_COLS
    f_sheets = {name: _sheet_cells(root, f_shared) for name, root in f_roots.items()}
    f_s4 = f_sheets[SHEET_PENDING_KERNELS]
    assert any(v and "Visible Group Attribution Table" in v for v, _ in f_s4.values())
    for _, formula in f_s4.values():
        if formula:
            assert len(formula) <= EXCEL_MAX_FORMULA_CHARS
            if "SUM(" in formula:
                assert formula.count(",") + 1 <= EXCEL_MAX_FORMULA_ARGS
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "F3", f_sheets) == pytest.approx(0.0, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "F9", f_sheets) == pytest.approx(0.0, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "D10", f_sheets) == pytest.approx(0.0, abs=1e-6)
    assert _eval_workbook_cell(SHEET_PENDING_KERNELS, "E10", f_sheets) == pytest.approx(0.0, abs=1e-4)


def test_sheet3_as_evaluated_preservation_mxfp8_scales_shared_dp_and_a23_isa_guard(tmp_path):
    # 1. Kimi-K3.1: Sheet 3 defaults to "As evaluated", keeping all BF16 rows BF16 & matching Memory receipt!
    raw_k, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=8, ep=32, dp=4, dcp=8,
    )
    case_k = parse_case(raw_k)
    led_k = compute_memory_quant_ledger(case_k, seq_len=4096, concurrency=4)
    assert led_k.stage_budgets[0].weight_payload_bytes + led_k.stage_budgets[0].weight_scale_aux_bytes == pytest.approx(
        led_k.sim_memory_weight_bytes_by_stage[0], rel=1e-9,
    )
    out_k = tmp_path / "kimi_mem.xlsx"
    export_comparison_xlsx(raw_k, out_k, request=StepRequest(phase="decode", tokens=1, context=512, sequences=2, scope="body"))
    _, roots, shared = _inspect_xlsx(out_k)
    all_sheets = {name: _sheet_cells(root, shared) for name, root in roots.items()}
    s3 = all_sheets[SHEET_MEMORY]
    assert s3["T3"][0] == QUANT_MODE_AS_EVALUATED

    # Recalculate EVERY weight row in "As evaluated" mode: zero BF16 rows flip to MXFP4!
    for idx, wrow in enumerate(led_k.weight_rows):
        r = 4 + idx
        assert _eval_workbook_cell(SHEET_MEMORY, f"J{r}", all_sheets) == wrow.quant_format
        assert _eval_workbook_cell(SHEET_MEMORY, f"K{r}", all_sheets) == pytest.approx(wrow.payload_bytes, abs=1e-6)
        assert _eval_workbook_cell(SHEET_MEMORY, f"L{r}", all_sheets) == pytest.approx(wrow.scale_bytes, abs=1e-6)
        assert _eval_workbook_cell(SHEET_MEMORY, f"M{r}", all_sheets) == pytest.approx(wrow.aux_bytes, abs=1e-6)
    # Interactive what-if switch: changing $T$3 -> "BF16" recalculates routed experts to BF16!
    routed_idx = next(i for i, wr in enumerate(led_k.weight_rows) if wr.category == "routed_expert")
    assert _eval_workbook_cell(
        SHEET_MEMORY, f"J{4 + routed_idx}", all_sheets,
        overrides={(SHEET_MEMORY, "T3"): "BF16"},
    ) == "BF16"

    # 2. MXFP8 E8M0 scale preservation + shared_expert_dp replication parity + PreparedWeightSpec parity.
    raw_v4 = compose_case("deepseek-v3.2", "a5", dp=1, tp=8, ep=8, layers=4, prompt_tokens=512, output_tokens=32)
    case_v4 = parse_case(raw_v4)
    mxfp8_q = QuantSpec(
        "mxfp8_test", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="fp8_e4m3", weight_block=(1, 32), scale_format="fp8_e8m0",
        scale_bytes=1.0, activation_group_size=32,
    )
    case_mx = replace(case_v4, model=replace(case_v4.model, quant=mxfp8_q))
    led_mx = compute_memory_quant_ledger(case_mx, seq_len=512, concurrency=2)
    q_row = next(r for r in led_mx.weight_rows if r.name == "q_a")
    assert q_row.quant_format == "MXFP8" and q_row.scale_bytes > 0
    assert led_mx.stage_budgets[0].weight_payload_bytes + led_mx.stage_budgets[0].weight_scale_aux_bytes == pytest.approx(
        led_mx.sim_memory_weight_bytes_by_stage[0], rel=1e-9,
    )

    meta_dp = dict(case_v4.runtime.metadata)
    meta_dp["enable_shared_expert_dp"] = True
    case_sdp = replace(case_v4, runtime=replace(case_v4.runtime, metadata=tuple(meta_dp.items())))
    led_tp = compute_memory_quant_ledger(case_v4, seq_len=512, concurrency=2)
    led_sdp = compute_memory_quant_ledger(case_sdp, seq_len=512, concurrency=2)
    sh_tp = sum(r.per_die_bytes for r in led_tp.weight_rows if r.category == "shared_expert")
    sh_sdp = sum(r.per_die_bytes for r in led_sdp.weight_rows if r.category == "shared_expert")
    assert sh_sdp == sh_tp * 8
    assert led_sdp.stage_budgets[0].weight_payload_bytes + led_sdp.stage_budgets[0].weight_scale_aux_bytes == pytest.approx(
        led_sdp.sim_memory_weight_bytes_by_stage[0], rel=1e-9,
    )

    prep = PreparedWeightSpec(
        allocation_id="test.prepared.runtime", source_tensors=("w",), transform="pack_bf16",
        scope="target", layer_indices=(0, 1), local_shape=(1024, 512), dtype="bf16", layout="NZ",
    )
    led_prep = compute_memory_quant_ledger(case_v4, seq_len=512, concurrency=2, prepared_weights=(prep,))
    assert any(r.name == "test.prepared.runtime" for r in led_prep.weight_rows)
    assert led_prep.stage_budgets[0].weight_payload_bytes + led_prep.stage_budgets[0].weight_scale_aux_bytes == pytest.approx(
        led_prep.sim_memory_weight_bytes_by_stage[0], rel=1e-9,
    )

    # 3. Strict A2/A3 ISA guard + distinct C8 vs FP8 + independent index_k/draft_kv!
    raw_a3 = compose_case("qwen3-32b", "a3", dp=1, tp=2, layers=2, prompt_tokens=512, output_tokens=32)
    case_a3 = parse_case(raw_a3)
    for bad_wq in ("MXFP8", "MXFP4", "FP8"):
        with pytest.raises(ModelingError, match="hw_isa_incompatible"):
            compute_memory_quant_ledger(case_a3, weight_quant=bad_wq)
    with pytest.raises(ModelingError, match="hw_isa_incompatible"):
        compute_memory_quant_ledger(case_a3, cache_dtype="FP8")

    led_c8 = compute_memory_quant_ledger(case_v4, seq_len=1024, concurrency=2, cache_dtype="C8")
    led_fp8 = compute_memory_quant_ledger(case_v4, seq_len=1024, concurrency=2, cache_dtype="FP8")
    kv_c8 = next(r.per_request_bytes for r in led_c8.cache_rows if r.component == "mla_kv")
    kv_fp8 = next(r.per_request_bytes for r in led_fp8.cache_rows if r.component == "mla_kv")
    assert kv_c8 > kv_fp8
    idx_c8 = [r.per_request_bytes for r in led_c8.cache_rows if r.component == "index_k"]
    idx_fp8 = [r.per_request_bytes for r in led_fp8.cache_rows if r.component == "index_k"]
    if idx_c8 and idx_fp8:
        assert idx_c8[0] == pytest.approx(idx_fp8[0], rel=1e-12)


def test_sheet1_roofline_bottleneck_unconstrained_ttft_and_per_column_sla(tmp_path):
    raw_p, _ = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=1, prompt_tokens=512, output_tokens=32, concurrency=1,
    )
    req_p = StepRequest(phase="prefill", tokens=512, context=0, sequences=1, scope="body")
    bun_p = build_export_bundle(raw_p, request=req_p, apply_target_envelope=False)
    bun_uncon = replace(bun_p, ttft_slo_ms=None, step_tight_resources=("hbm_read@r0.p0.d0",))
    out_p = tmp_path / "prefill_unconstrained.xlsx"
    export_comparison_xlsx(bun_uncon, out_p)
    _, roots_p, shared_p = _inspect_xlsx(out_p)
    s1_p = _sheet_cells(roots_p[SHEET_SUMMARY], shared_p)
    assert s1_p["A19"][0] == "Roofline 理论物理下界 L* (Theoretical Lower Bound)"
    assert s1_p["H19"][0] == "主导物理瓶颈: HBM 访存带宽受限"
    assert s1_p["B8"][0] == "不限 / N/A" and s1_p["B10"][0] == "不限 / N/A"
    assert s1_p["B16"][0] == "不限 / N/A" and s1_p["C16"][0] == "不限 / N/A"

    # Tied resources + precedence-dominated + independent Before vs After SLA verdicts!
    req_d = StepRequest(phase="decode", tokens=1, context=512, sequences=2, scope="body")
    bun_d = build_export_bundle(raw_p, request=req_d, apply_target_envelope=False)
    bun_bef = replace(bun_d, step_span_us=30000.0, tpot_slo_ms=20.0, avg_accepted_tokens=1.0)
    bun_aft = replace(
        bun_d, step_span_us=30000.0, tpot_slo_ms=50.0, avg_accepted_tokens=1.0,
        step_tight_resources=("cube@r0.p0.d0", "hbm_read@r0.p0.d0"),
    )
    out_sla = tmp_path / "per_col_sla.xlsx"
    export_comparison_xlsx(bun_bef, out_sla, compare_source=bun_aft)
    _, roots_s, shared_s = _inspect_xlsx(out_sla)
    s1_s = _sheet_cells(roots_s[SHEET_SUMMARY], shared_s)
    assert s1_s["B16"][0] == "FAIL" and s1_s["C16"][0] == "PASS"
    assert "并列" in (s1_s["H19"][0] or "") and "Cube" in (s1_s["H19"][0] or "") and "HBM" in (s1_s["H19"][0] or "")

    bun_prec = replace(bun_d, step_tight_resources=())
    out_prec = tmp_path / "precedence_bound.xlsx"
    export_comparison_xlsx(bun_prec, out_prec)
    _, roots_pr, shared_pr = _inspect_xlsx(out_prec)
    assert "关键路径" in (_sheet_cells(roots_pr[SHEET_SUMMARY], shared_pr)["H19"][0] or "")


def test_sheet2_attnres_depth_slices_recoverssm_b69_codesign_cards_and_after_trace(tmp_path):
    raw_base, req_base = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=1, prompt_tokens=512, output_tokens=32, concurrency=2,
    )
    tmpl = build_export_bundle(raw_base, request=req_base, apply_target_envelope=False)
    proto = tmpl.operators[0]

    # 1. AttnResFwd depth-weighted slice reaggregation across different Before/After partitions + final=93!
    n_layers = 93
    block_sz = 8
    per_layer_b = {}
    per_layer_a = {}
    for lyr in range(n_layers):
        v_blk = (lyr + block_sz - 1) // block_sz
        d_b = 25.0 + 2.8 * v_blk
        d_a = 22.0 + 2.5 * v_blk
        per_layer_b[("target", lyr)] = {
            "span_us": d_b, "ops": {"self_attention_res": (0.0, d_b)},
            "ops_exposed": {"self_attention_res": d_b}, "ops_lstar": {"self_attention_res": d_b * 0.6},
            "residual_block_size": block_sz,
        }
        per_layer_a[("target", lyr)] = {
            "span_us": d_a, "ops": {"self_attention_res": (0.0, d_a)},
            "ops_exposed": {"self_attention_res": d_a}, "ops_lstar": {"self_attention_res": d_a * 0.6},
            "residual_block_size": block_sz,
        }

    # 2. RecoverSSMStateCommit: 69 tail rows collapse into 1 row with numeric repeat_count == 69!
    raw_rec_rows = [
        replace(
            proto, occurrence=OpOccurrence(0, 0, 0, f"s0.l{i}.recover_ssm_commit"),
            op_id=f"s0.l{i}.recover_ssm_commit", archetype_id="target_tail",
            archetype_title="Target Tail (×1)", repeat_count=1, dur_us=4.0, exposed_dur_us=4.0,
            l_star_lean_us=2.0, logical_role="Speculative SSM Commit", ascend_kernel="recover_ssm_state_commit",
            semantic_atoms=("recover_ssm",), predecessors=(),
        )
        for i in range(69)
    ]
    rec_dems = [(0, 2000, 1000, 500, 0) for _ in range(69)]
    col_rows, col_dems = _collapse_micro_split_ops(raw_rec_rows, rec_dems)
    assert len(col_rows) == 1 and col_rows[0].repeat_count == 69
    assert col_rows[0].dur_us == pytest.approx(4.0, rel=1e-9)
    assert col_rows[0].exposed_dur_us == pytest.approx(4.0, rel=1e-9)
    assert col_rows[0].ascend_kernel == "RecoverSSMStateCommit"
    assert "(×69" not in col_rows[0].logical_role
    assert col_dems[0] == (0, 2000, 1000, 500, 0)  # per-call, never multiplied by 69 again!

    mean_a0 = sum(per_layer_a[("target", i)]["ops"]["self_attention_res"][1] for i in range(33)) / 33.0
    mean_a1 = sum(per_layer_a[("target", i)]["ops"]["self_attention_res"][1] for i in range(33, 93)) / 60.0
    ar_b = replace(
        proto, occurrence=OpOccurrence(0, 0, 0, "l0.self_attention_res"), op_id="l0.self_attention_res",
        archetype_id="target_arch_0", archetype_title="Target L0..L92 (×93)", repeat_count=93,
        dur_us=40.0, exposed_dur_us=40.0, family="attn_res_prefix", semantic_atoms=("attn_res",),
        tensor_contract="[T=96, H=7168]", predecessors=(),
    )
    ar_a0 = replace(
        ar_b, archetype_id="target_arch_0", archetype_title="Resident L0..L32 (×33)", repeat_count=33,
        dur_us=mean_a0, exposed_dur_us=mean_a0,
    )
    ar_a1 = replace(
        ar_b, occurrence=OpOccurrence(0, 0, 0, "l33.self_attention_res"), op_id="l33.self_attention_res",
        archetype_id="target_arch_1", archetype_title="Reload L33..L92 (×60)", repeat_count=60,
        dur_us=mean_a1, exposed_dur_us=mean_a1,
    )
    final_ar = replace(
        proto, occurrence=OpOccurrence(0, 0, 0, "final.attn_res_fwd"), op_id="final.attn_res_fwd",
        archetype_id="target_tail", archetype_title="Target Tail (×1)", repeat_count=1,
        dur_us=56.2, exposed_dur_us=56.2, family="attn_res_prefix", semantic_atoms=("final_attn_res",),
        tensor_contract="[T=96, H=7168] | final=93，valid_blocks=12，Repeats=1", predecessors=(),
    )

    arch_b = (
        LayerArchetypeSpec("target_arch_0", "target", "Target L0..L92 (×93)", "L0..L92", 93, tuple(range(93)), "L0..L92", 0),
        LayerArchetypeSpec("target_tail", "target", "Target Tail (×1)", "Tail", 1, (), "Tail", None),
    )
    arch_a = (
        LayerArchetypeSpec("target_arch_0", "target", "Resident L0..L32 (×33)", "L0..L32", 33, tuple(range(33)), "L0..L32", 0),
        LayerArchetypeSpec("target_arch_1", "target", "Reload L33..L92 (×60)", "L33..L92", 60, tuple(range(33, 93)), "L33..L92", 33),
        LayerArchetypeSpec("target_tail", "target", "Target Tail (×1)", "Tail", 1, (), "Tail", None),
    )
    opps = (
        CoDesignOpportunity(
            opportunity_id="mf_kda_block", kind="mega_fusion", track="kernel_new", readiness="synthesized_target",
            title="Fused KDA Gate + RMSNorm MegaFusion", evidence_refs=("kda.py",),
            mega_fusion=MegaFusionBound(
                region_op_ids=("l0.kda_gate", "l0.kda_norm"), exposed_before_ms=2.4,
                boundary_lower_bound_ms=1.1, tiled_engine_ms=1.5,
                gain_ceiling_ms=1.3, gain_expected_ms=0.9,
                eliminated_gm_bytes=8 * 1024 * 1024, ub_live_bytes=128 * 1024, ub_capacity_bytes=256 * 1024,
            ),
        ),
        CoDesignOpportunity(
            opportunity_id="st_recoverssm_ring", kind="space_time_tradeoff", track="framework", readiness="proposed_pr",
            title="RecoverSSM Snapshot Ring vs Recompute",
            space_time=SpaceTimeTradeoffSpec(
                direction="space_for_time", mechanism="recover_ssm_snapshot_ring",
                delta_hbm_bytes=512 * 1024 * 1024, free_hbm_headroom_bytes=4 * 1024**3,
                delta_step_ms=-0.276, delta_concurrency_per_dp=0, delta_tps_per_die=1.8,
                feasible_under_hbm_and_sla=True,
            ),
        ),
        CoDesignOpportunity(
            opportunity_id="xh_engram_ddr", kind="cross_hw_offload_multistream", track="framework", readiness="proposed_pr",
            title="Engram Host DDR H2D over Layers 0..k-1",
            cross_hw=CrossHwOffloadSpec(
                target_component="engram_table", source_tier="host_ddr", transfer_engine="h2d",
                compute_hide_window_ms=1.8, transfer_ms=1.2, net_exposed_ms=0.0,
                freed_hbm_bytes=12 * 1024**3, delta_concurrency_per_dp=8, delta_tps_per_die=6.4,
            ),
        ),
    )
    sum_b_us = sum(per_layer_b[("target", i)]["ops"]["self_attention_res"][1] for i in range(93)) + 56.2 + 69 * 4.0
    sum_a_us = sum(per_layer_a[("target", i)]["ops"]["self_attention_res"][1] for i in range(93)) + 56.2 + 69 * 4.0
    bun_b = replace(
        tmpl, operators=(ar_b, final_ar, col_rows[0]), archetypes=arch_b, total_target_layers=93,
        per_layer_timings=per_layer_b, step_span_us=sum_b_us,
    )
    bun_a = replace(
        tmpl, operators=(ar_a0, ar_a1, final_ar, col_rows[0]), archetypes=arch_a, total_target_layers=93,
        per_layer_timings=per_layer_a, step_span_us=sum_a_us, co_design_opportunities=opps,
    )
    aligned = align_before_after_operators(bun_b, bun_a)
    ar_rows = [r for r in aligned if r.before_ops and "self_attention_res" in r.before_ops[0].op_id]
    assert len(ar_rows) == 2
    assert sum(r.before_ops[0].dur_us * r.effective_before_repeat for r in ar_rows) == pytest.approx(
        sum(per_layer_b[("target", i)]["ops"]["self_attention_res"][1] for i in range(93)), rel=1e-9,
    )
    assert sum(r.after_ops[0].dur_us * r.effective_after_repeat for r in ar_rows) == pytest.approx(
        sum(per_layer_a[("target", i)]["ops"]["self_attention_res"][1] for i in range(93)), rel=1e-9,
    )
    assert "L0 -> L32，随层深 valid_blocks=0..4" in ar_rows[0].before_ops[0].tensor_contract
    assert "L33 -> L92，随层深 valid_blocks=5..12" in ar_rows[1].before_ops[0].tensor_contract

    rec_aligned = next(r for r in aligned if r.before_ops and "recover_ssm" in r.before_ops[0].op_id)
    assert rec_aligned.effective_before_repeat == 69 and rec_aligned.effective_after_repeat == 69

    out_ar = tmp_path / "attnres_recoverssm_codesign.xlsx"
    export_comparison_xlsx(bun_b, out_ar, compare_source=bun_a)
    _, roots_ar, shared_ar = _inspect_xlsx(out_ar)
    s2 = _sheet_cells(roots_ar[SHEET_BREAKDOWN], shared_ar)
    rec_rows_s2 = [int(k[1:]) for k, (v, _) in s2.items() if k.startswith("G") and v and "RecoverSSMStateCommit" in v]
    assert len(rec_rows_s2) == 1
    rr = rec_rows_s2[0]
    assert int(s2[f"B{rr}"][0] or "0") == 69
    assert float(s2[f"I{rr}"][0] or "0") == pytest.approx(4.0, abs=1e-6)
    assert float(s2[f"J{rr}"][0] or "0") == pytest.approx(69 * 4.0 / 1000.0, abs=1e-6)
    assert float(s2[f"P{rr}"][0] or "0") == pytest.approx(69 * 4.0 / 1000.0, abs=1e-6)
    assert any(v and "final=93，valid_blocks=12，Repeats=1" in v for v, _ in s2.values())

    s4_txt = "\n".join(str(v or "") for v, _ in _sheet_cells(roots_ar[SHEET_PENDING_KERNELS], shared_ar).values())
    assert "理论上限 Δ_max" in s4_txt and "硬件预演预期 Δ_expected" in s4_txt
    assert "以存代算 (Space-for-Time)" in s4_txt and "Host DDR H2D" in s4_txt
    assert "TileLang / AscendC / Catlass" in s4_txt and "cannbot" not in s4_txt.lower()

    # 3. Post-synthesis per_layer_timings reflects After timings (never overwritten by Before timings).
    raw_bef, req_s = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=1, prompt_tokens=512, output_tokens=32, concurrency=2, feature_track="merged",
    )
    raw_aft, _ = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=1, prompt_tokens=512, output_tokens=32, concurrency=2, feature_track="expected_ideal",
    )
    b_pre = build_export_bundle(raw_bef, request=req_s, apply_target_envelope=False)
    b_post = build_export_bundle(raw_aft, request=req_s, apply_target_envelope=True)
    assert all(("target", idx) in b_post.per_layer_timings for idx in range(b_post.total_target_layers))
    tr_post = tmp_path / "after_trace.json"
    export_msprof_trace(b_post, tr_post)
    raw_tr = json.loads(tr_post.read_text(encoding="utf-8"))
    events = raw_tr["traceEvents"] if isinstance(raw_tr, dict) else raw_tr
    root_ev = next(e for e in events if e.get("cat") == "Framework" and "ModelRunner.execute_model" in str(e.get("name", "")))
    if b_post.step_span_us < b_pre.step_span_us - 10.0:
        assert float(root_ev["dur"]) < b_pre.step_span_us
