"""Orthogonal tests for human-native ``msprof`` trace and OOXML ``.xlsx`` exporters.

[Role] Verify 1:1 full-model streaming ``msprof`` ``trace_view.json`` (``PID 0``
  Host Framework/MSTX hierarchy, physically grounded AIC/AIV/MTE2/MTE3/Fixpipe
  durations & ratios with zero ``PIPE_*_FRACTION`` hacks, causal flows, bounded
  memory), human-native OOXML ``.xlsx`` workbooks (Archetype ×N grouping,
  side-by-side Before-vs-After N->1 ``merge_cells``, live formulas, MSO Stream-1
  overlap, ``New_Kernel_Design_Backlog`` Agent slots, conditional
  ``Measured_msprof_Calibration``, zero banned Python class names, zero
  ``Other`` submodules), ``QuantSpec.exclusions`` in ``Memory_Quant_Calculator``,
  and one-shot ``export`` / ``export-xlsx`` / ``export-trace`` CLI flows.
[Invariants] Every timing and memory assertion checks exact physical/schema
  identities against ``runtime.Engine`` and ``check_optimality_certificate``;
  zero ``0.80x`` heuristic fudge factors or third-party OOXML dependencies.
[Timeouts & Locks] Pure in-memory ``tier0`` tests under pytest ``tmp_path``;
  no subprocesses, network access, sleeps, or locks.
[Side-Effect Policy] Writes temporary test artifacts exclusively under
  ``tmp_path``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import tracemalloc
import xml.etree.ElementTree as ET
import zipfile

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim import cli as cli_mod
from mindie_sim import cli_export
from mindie_sim.compose import compose_case as compose_raw_case
from mindie_sim.export import (
    AGENT_BRIEF_SCHEMA,
    BANNED_IMPLEMENTATION_NAMES,
    MSPROF_PROCESS_SPEC,
    PENDING_KIND_TO_IMPLEMENT,
    PENDING_KIND_TO_OPTIMIZE,
    PID_FRAMEWORK,
    SCRIPT_AUDIT_SCHEMA,
    SHEET_BREAKDOWN,
    SHEET_MEASURED,
    SHEET_MEMORY,
    SHEET_NEW_KERNELS,
    SHEET_PENDING_KERNELS,
    SHEET_SUMMARY,
    TID_FW_ENGINE,
    TID_FW_HOST_LAUNCH,
    TID_FW_LAYER_MSTX,
    TID_FW_SUBMODULE_MSTX,
    Workbook,
    align_before_after_operators,
    audit_launch_script,
    build_agent_brief,
    build_export_bundle,
    build_macro_module_breakdown,
    build_pending_kernel_details,
    compose_export_case,
    compute_memory_quant_ledger,
    evaluate_weight_storage_bytes,
    export_agent_brief,
    export_comparison_xlsx,
    export_msprof_trace,
    format_executive_digest_markdown,
    load_kernel_details_csv,
    parse_launch_script,
    render_optimized_serve_sh,
    resolve_hardware_alias,
)
from mindie_sim.export import trace_exporter as trace_mod
from mindie_sim.export.xlsx_exporter import NS_MAIN
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.structure import parse_case

pytestmark = pytest.mark.tier0

NS_PREFIX = f"{{{NS_MAIN}}}"
STREAM_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024
MAX_SERIALIZE_SECONDS = 0.6
MAX_ARCHETYPE_OPS_IN_RAM = 250
MIN_EAGER_NORM_GATE_OPS = 11


def _read_xlsx_sheets(path: Path) -> tuple[list[str], dict[str, ET.Element], list[str]]:
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        assert {
            "[Content_Types].xml", "_rels/.rels", "xl/workbook.xml",
            "xl/_rels/workbook.xml.rels", "xl/styles.xml", "xl/sharedStrings.xml",
        } <= names
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = ["".join(si.itertext()) for si in sst_root.findall(f"{NS_PREFIX}si")]
        sheet_names = [elem.attrib["name"] for elem in wb_root.findall(f".//{NS_PREFIX}sheet")]
        roots = {
            title: ET.fromstring(zf.read(f"xl/worksheets/sheet{idx}.xml"))
            for idx, title in enumerate(sheet_names, start=1)
        }
    return sheet_names, roots, shared


def _sheet_cells(
    root: ET.Element, shared: list[str],
) -> dict[str, tuple[str | None, str | None]]:
    out: dict[str, tuple[str | None, str | None]] = {}
    for cell in root.findall(f".//{NS_PREFIX}c"):
        ref = cell.attrib["r"]
        f_el = cell.find(f"{NS_PREFIX}f")
        v_el = cell.find(f"{NS_PREFIX}v")
        formula = f_el.text if f_el is not None else None
        raw_v = v_el.text if v_el is not None else None
        val = shared[int(raw_v)] if (cell.attrib.get("t") == "s" and raw_v is not None) else raw_v
        out[ref] = (val, formula)
    return out


def _sheet_merges(root: ET.Element) -> list[str]:
    return [el.attrib["ref"] for el in root.findall(f".//{NS_PREFIX}mergeCell")]


def _compact_case(
    *, tp: int = 2, ep: int = 2, pp: int = 1, layers: int = 2, sku_key: str = "a5",
) -> dict:
    return compose_raw_case(
        "qwen3-30b-a3b",
        REPRESENTATIVE_SKUS[sku_key],
        tp=tp, dp=1, ep=ep, pp=pp, layers=layers,
        prompt_tokens=256, output_tokens=16, concurrency=2,
    )


def test_msprof_trace_schema_metadata_physical_pipes_and_host_hierarchy(tmp_path):
    for banned_attr in (
        "PIPE_LEAD_FRACTION", "PIPE_CORE_START_FRACTION",
        "PIPE_CORE_END_FRACTION", "PIPE_TAIL_START_FRACTION",
    ):
        assert not hasattr(trace_mod, banned_attr)

    case = _compact_case(tp=2, ep=2, pp=1, layers=2)
    req = StepRequest(phase="decode", tokens=1, context=512, sequences=2, scope="body")
    bundle = build_export_bundle(case, request=req)
    assert bundle.step_certificate_valid is True
    assert 0.0 <= bundle.step_l_star_lean_us <= bundle.step_span_us + 1.0

    trace_path = tmp_path / "trace_view.json"
    receipt = export_msprof_trace(bundle, trace_path)
    raw = trace_path.read_bytes()
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    for banned in BANNED_IMPLEMENTATION_NAMES:
        assert banned not in text
    events = json.loads(text)
    assert isinstance(events, list) and len(events) == receipt.total_events

    meta_names = {e["name"] for e in events if e.get("ph") == "M"}
    assert {"process_name", "process_labels", "process_sort_index", "thread_name", "thread_sort_index"} <= meta_names
    proc_names = {e["args"]["name"] for e in events if e.get("ph") == "M" and e["name"] == "process_name"}
    for required_proc in MSPROF_PROCESS_SPEC:
        assert any(name.startswith(required_proc) for name in proc_names)

    fw_events = [e for e in events if e.get("pid") == PID_FRAMEWORK and e.get("ph") == "X"]
    fw_tids = {e["tid"] for e in fw_events}
    assert {TID_FW_ENGINE, TID_FW_LAYER_MSTX, TID_FW_SUBMODULE_MSTX, TID_FW_HOST_LAUNCH} <= fw_tids

    kernel_events = [e for e in events if e.get("ph") == "X" and e.get("cat") in {"Kernel", "Communication"}]
    assert len(kernel_events) >= len(bundle.operators)
    sample_args = kernel_events[0]["args"]
    for key in (
        "Ascend Kernel", "Kernel Backend", "Kernel Status", "Logical Role", "Shapes & Dtype",
        "aic_mac_ratio", "aiv_vec_ratio", "mte2_ratio", "mte3_ratio", "fixpipe_ratio",
        "L* Physical Lower Bound (us)", "Sim Pred (us)", "Gap vs L* (us)", "Gap Radar Rank",
    ):
        assert key in sample_args
    for ratio_k in ("aic_mac_ratio", "aiv_vec_ratio", "mte2_ratio", "mte3_ratio", "fixpipe_ratio"):
        assert 0.0 <= float(sample_args[ratio_k]) <= 1.0
    assert all(e["ts"] >= 0 and e["dur"] >= 0 and math.isfinite(e["ts"]) and math.isfinite(e["dur"]) for e in kernel_events)

    flow_s = {e["id"]: e for e in events if e.get("ph") == "s"}
    flow_f = {e["id"]: e for e in events if e.get("ph") == "f"}
    assert len(flow_s) == len(flow_f) == receipt.flow_pairs and len(flow_s) > 0
    assert set(flow_s) == set(flow_f)
    for fid, s_ev in flow_s.items():
        f_ev = flow_f[fid]
        assert f_ev.get("bp") == "e"
        assert f_ev["ts"] >= s_ev["ts"] - 1e-6

    counter_names = {e["name"] for e in events if e.get("ph") == "C"}
    assert {"HBM Bandwidth (GB/s)", "Cube Utilization (%)", "Vector Utilization (%)", "Active HBM Memory (GiB)"} <= counter_names


def test_trace_streaming_memory_and_speed_budget_on_full_multilayer_model(tmp_path):
    raw_case, req = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=1, ep=1, dcp=1, prompt_tokens=2048, output_tokens=64,
        concurrency=1, mtp_proposal_tokens=5, feature_track="expected_ideal",
    )
    bundle = build_export_bundle(raw_case, request=req)
    assert len(bundle.operators) < MAX_ARCHETYPE_OPS_IN_RAM
    assert bundle.total_target_layers == 93 and bundle.total_draft_layers == 5
    arch_titles = [a.title for a in bundle.archetypes]
    assert any("Target Archetype A" in t for t in arch_titles)
    assert any("Target Archetype B" in t and "×68" in t for t in arch_titles)
    assert any("Target Archetype C" in t and "×24" in t for t in arch_titles)
    assert any(t.startswith("Draft [DSpark]") and "×5" in t for t in arch_titles)
    recover_ops = [o for o in bundle.operators if "recover_ssm" in o.op_id]
    assert recover_ops and all(o.archetype_id.startswith("target_arch_") for o in recover_ops)
    assert not any(o.op_id.endswith(("recurrent_scan0", "recurrent_scan1")) for o in bundle.operators)

    out_trace = tmp_path / "full_trace_view.json"
    out_xlsx = tmp_path / "full_standalone.xlsx"
    t0 = time.perf_counter()
    t_rec = export_msprof_trace(bundle, out_trace)
    x_rec = export_comparison_xlsx(bundle, out_xlsx)
    elapsed = time.perf_counter() - t0
    assert elapsed < MAX_SERIALIZE_SECONDS

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        export_msprof_trace(bundle, tmp_path / "mem_trace_view.json")
        export_comparison_xlsx(bundle, tmp_path / "mem_standalone.xlsx")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < STREAM_MEMORY_LIMIT_BYTES
    assert t_rec.replayed_layers >= 98
    assert t_rec.total_events > 8000
    assert x_rec.bytes_written > 0
    assert out_trace.read_bytes()[:2] == b"[\n" and out_trace.read_bytes()[-3:] == b"\n]\n"
    events = json.loads(out_trace.read_text(encoding="utf-8"))
    mstx_layers = [e["name"] for e in events if e.get("pid") == PID_FRAMEWORK and e.get("tid") == TID_FW_LAYER_MSTX and e.get("ph") == "X"]
    assert any("Layer 0" in n for n in mstx_layers)
    assert any("Layer 92" in n for n in mstx_layers)
    assert any("Draft [DSpark] Block Layer L4" in n for n in mstx_layers)


def test_xlsx_before_after_merge_cells_formulas_and_new_kernel_backlog(tmp_path):
    raw_bef, req = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=2, tp=2, ep=4, dcp=2, prompt_tokens=2048, output_tokens=64,
        concurrency=4, tpot_slo_ms=30.0, mtp_proposal_tokens=5,
        mtp_acceptance_length=3.5, feature_track="merged",
    )
    raw_aft, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=2, tp=2, ep=4, dcp=2, prompt_tokens=2048, output_tokens=64,
        concurrency=4, tpot_slo_ms=30.0, mtp_proposal_tokens=5,
        mtp_acceptance_length=3.5, feature_track="expected_ideal",
    )
    b_bef = build_export_bundle(raw_bef, request=req)
    b_aft = build_export_bundle(raw_aft, request=req)

    # Real expected_ideal natively fuses target_arch_2 MLA Prolog (7 ops -> 1 aclnnMlaPrologV3).
    mla_bef = [op for op in b_bef.operators if op.archetype_id == "target_arch_2" and "MLA Prolog" in op.submodule]
    mla_aft = [op for op in b_aft.operators if op.archetype_id == "target_arch_2" and "MLA Prolog" in op.submodule]
    assert len(mla_bef) >= 7 and any(op.canonical_op_key == "mla_prolog_v3" for op in mla_aft)

    designs = {
        "aclnnMlaPrologV3": {
            "config_howto": "export VLLM_ASCEND_ENABLE_MLAPO=1",
            "kernel_design": "Single-pass AIV vector tile (UB 248KiB double-buffered) fusing MLA prolog",
            "framework_work": "Register CustomOp in vllm_ascend/attention/mla_v1.py",
            "acceptance": "BF16 abs_err <= 1e-3, L* efficiency >= 70%",
        },
        "BlockAttnResFusedPrefix": {
            "config_howto": "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
            "kernel_design": "Single-pass AIV vector tile (UB 248KiB double-buffered) fusing BlockAttnRes prefix",
            "framework_work": "Register CustomOp in vllm_ascend/ops/attn_res.py",
            "acceptance": "BF16 abs_err <= 1e-3, L* efficiency >= 70%",
        },
        "K3 SituAndMul": {
            "config_howto": "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
            "kernel_design": "Single-pass AIV vector tile (UB 248KiB double-buffered) fusing SiTU + Mul",
            "framework_work": "Register CustomOp in vllm_ascend/ops/activation.py",
            "acceptance": "BF16 abs_err <= 1e-3, L* efficiency >= 70%",
        },
        "hcom_allgather": {
            "design": "Switch small collective payloads to superpod one-shot transport",
            "acceptance": "Exposed collective attainment >= 75% Link L*",
        },
    }
    xlsx_path = tmp_path / "comparison.xlsx"
    brief_path = tmp_path / "comparison_agent_brief.json"
    summary = export_comparison_xlsx(
        b_bef, xlsx_path, compare_source=b_aft,
        agent_kernel_designs=designs, agent_brief_path=brief_path,
    )
    assert SHEET_PENDING_KERNELS == "Pending_Kernel_Details" and SHEET_NEW_KERNELS == SHEET_PENDING_KERNELS
    assert summary.sheets == (SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY, SHEET_PENDING_KERNELS)
    assert SHEET_MEASURED not in summary.sheets
    assert summary.has_comparison is True and summary.has_measured is False
    assert summary.new_kernel_backlog_count >= 1 and summary.to_optimize_kernel_count >= 1
    assert brief_path.is_file() and brief_path.stat().st_size < 8192

    sheet_names, roots, shared = _read_xlsx_sheets(xlsx_path)
    assert sheet_names == [SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY, SHEET_PENDING_KERNELS]
    for banned in BANNED_IMPLEMENTATION_NAMES:
        assert not any(banned in s for s in shared), f"banned name leaked: {banned}"
    assert "Other" not in shared
    assert not any("4D" in s for s in shared)
    assert not any("待 Agent 补充" in s for s in shared)

    expected_order = ["dimension", "sheetViews", "sheetFormatPr", "cols", "sheetData", "mergeCells", "dataValidations", "pageMargins"]
    for _, root in roots.items():
        tags = [child.tag.removeprefix(NS_PREFIX) for child in root]
        assert tags == [t for t in expected_order if t in tags]

    # Sheet 1 — Verify all 4 Executive Blocks (Deployment, KPIs w/o 4D jargon, L1/L2 Masking, 3-Track Roadmap).
    s1 = _sheet_cells(roots[SHEET_SUMMARY], shared)
    assert s1["C4"][0] == "Target 93L + Draft 5L" and int(s1["G4"][0]) == 4
    assert "DP=2, TP=2, EP=4" in (s1["H5"][0] or "") and "DCP=2" in (s1["H5"][0] or "") and "SP=" in (s1["H5"][0] or "")
    assert int(s1["G6"][0]) == 4 and s1["G7"][1] == "G6*2" and int(s1["G7"][0]) == 8
    assert float(s1["B8"][0]) == pytest.approx(30.0) and float(s1["B9"][0]) == pytest.approx(3.5)
    assert s1["B10"][1] == "B8*B9" and float(s1["B10"][0]) == pytest.approx(105.0)
    assert float(s1["G8"][0]) > 0.0 and "原生 BF16" in (s1["H8"][0] or "") and int(s1["G9"][0]) >= 0
    assert s1["B15"][1] == "IF($B$9>0,B14/$B$9,B14)" and s1["C15"][1] == "IF($B$9>0,C14/$B$9,C14)"
    assert float(s1["C15"][0]) == pytest.approx(float(s1["C14"][0]) / 3.5, rel=1e-6)
    assert s1["B16"][1] == 'IF(B15<=$B$8,"PASS","FAIL")' and s1["C16"][0] in {"PASS", "FAIL"}
    assert s1["B17"][1] == "IF($G$4>0,B18/$G$4,0)" and s1["C17"][1] == "IF($G$4>0,C18/$G$4,0)"
    assert s1["B18"][1] == "IF(B14>0,$G$7*$B$9*1000/B14,0)" and s1["C18"][1] == "IF(C14>0,$G$7*$B$9*1000/C14,0)"
    assert float(s1["C18"][0]) == pytest.approx(8 * 3.5 * 1000.0 / float(s1["C14"][0]), rel=1e-6)
    assert float(s1["C17"][0]) == pytest.approx(float(s1["C18"][0]) / 4.0, rel=1e-6)
    assert s1["A21"][0] == "Cube 计算利用率 (AIC MFU)" and s1["A23"][0] == "HBM 访存带宽利用率 (MBU)"
    assert any(v and "1. Attention" in v for v, _ in s1.values())
    assert any(v and "KDA / GDN Linear Attention" in v for v, _ in s1.values())
    assert any(v and "MoE Routed Experts (GMM1 / Act / GMM2)" in v for v, _ in s1.values())
    assert any(v and "100% 掩盖" in v for v, _ in s1.values())
    assert not any(v and "Cross-Stream Overlap Credit" in v for v, _ in s1.values())

    # Sheet 2 — Strict topological order, up-front Stream&Masking (Col E), trailing Source Ref (Col W).
    brk_merges = _sheet_merges(roots[SHEET_BREAKDOWN])
    brk_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    assert brk_cells["E2"][0] == "并发流与掩盖关系 (Stream & Masking)"
    assert brk_cells["W2"][0] == "源码出处 / Source Ref"
    assert any(ref.startswith("A") and ":A" in ref for ref in brk_merges)
    assert any(ref.startswith("C") and ":C" in ref for ref in brk_merges)
    assert any(ref.startswith("L") and ":L" in ref for ref in brk_merges)
    assert any(ref.startswith("P") and ":P" in ref for ref in brk_merges)
    assert any(f and f.startswith("I") and "/1000" in f for _, f in brk_cells.values())
    assert any(f and f.startswith("SUM(J") and "-P" in f for _, f in brk_cells.values())
    assert not any(v == "Measured Kernel (msprof)" for v, _ in brk_cells.values())

    # Verify topological ordering inside target_arch_1: [Pre-Attn] attn_res precedes [Pre-MoE/FFN] attn_res.
    aligned = align_before_after_operators(b_bef, b_aft)
    arch1_roles = [(r.after_ops or r.before_ops)[0].logical_role for r in aligned if r.archetype_id == "target_arch_1"]
    pre_attn_idx = next(i for i, lbl in enumerate(arch1_roles) if "[Pre-Attn]" in lbl)
    pre_moe_idx = next(i for i, lbl in enumerate(arch1_roles) if "[Pre-MoE/FFN]" in lbl)
    assert pre_attn_idx == 0 and pre_moe_idx > pre_attn_idx + 2

    # Verify 100% 4-view conservation: Step Δ == Sheet 1 Δ == Sheet 2 Δ == Sheet 4 Cards Δ.
    macros = build_macro_module_breakdown(b_bef, b_aft)
    cards = build_pending_kernel_details(aligned, b_aft, before_bundle=b_bef)
    step_delta = (b_bef.step_span_us - b_aft.step_span_us) / 1000.0
    s1_delta = sum(m.before_ms - m.after_ms for m in macros)
    s2_delta = sum(
        (sum(b.exposed_dur_us for b in r.before_ops) - sum(a.exposed_dur_us for a in r.after_ops)) * r.repeat_count / 1000.0
        for r in aligned
    )
    s4_delta = sum(c.expected_gain_ms for c in cards)
    assert s1_delta == pytest.approx(step_delta, abs=1e-4)
    assert s2_delta == pytest.approx(step_delta, abs=1e-4)
    assert s4_delta == pytest.approx(step_delta, abs=1e-4)

    # Sheet 4 — Pending_Kernel_Details: Index + 3-Track Cards + structured overrides.
    s4 = _sheet_cells(roots[SHEET_PENDING_KERNELS], shared)
    assert any(v == PENDING_KIND_TO_IMPLEMENT for v, _ in s4.values())
    assert any(v == PENDING_KIND_TO_OPTIMIZE for v, _ in s4.values())
    assert any(v and v.startswith("Card #1 [") for v, _ in s4.values())
    assert any(v and "① 已有特效开关与通信展开配置" in v for v, _ in s4.values())
    assert any(v and "③ 推理框架改造要点" in v for v, _ in s4.values())
    assert any(v and "Single-pass AIV vector tile" in v for v, _ in s4.values())
    assert any(v and "BF16 abs_err <= 1e-3" in v for v, _ in s4.values())
    assert any(v and "Register CustomOp in vllm_ascend" in v for v, _ in s4.values())
    assert any(f and f.startswith("SUM(G") for _, f in s4.values())
    assert any(f == "G4" for _, f in s4.values())


def test_xlsx_conditional_measured_calibration_sheet_and_one_to_many_merges(tmp_path):
    case = _compact_case(tp=2, ep=2, pp=1, layers=1)
    req = StepRequest(phase="decode", tokens=1, context=256, sequences=1, scope="body")
    base_bundle = build_export_bundle(case, request=req)
    target_op = next(op for op in base_bundle.operators if op.dur_us > 0.0)

    csv_path = tmp_path / "kernel_details.csv"
    k_dur = round(target_op.dur_us / 3.0, 4)
    csv_lines = [
        "Device_id,Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us),Input Shapes,Input Data Types",
        f'0,101,47,aclnnStageA_Kernel,MatMul,AI_CORE,1792763502739232.637\t,{k_dur},"""4,512""",DT_BF16',
        f'0,102,47,aclnnStageB_Kernel,Add,AI_VECTOR_CORE,1792763502739250.100\t,{k_dur},"""4,512""",DT_BF16',
        f'0,103,47,aclnnStageC_Kernel,Quant,AI_VECTOR_CORE,1792763502739270.200\t,{k_dur},"""4,512""",DT_BF16',
        '0,104,47,aclnnUnmatched_Fill,Fill,AI_VECTOR_CORE,1792763502739300.000\t,2.500,"""1""",INT32',
    ]
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    assert len(load_kernel_details_csv(csv_path)) == 4

    obs_json = tmp_path / "obs_bundle.json"
    obs_json.write_text(json.dumps({
        "kernel_csv": str(csv_path),
        "mappings": [{
            "sim_op_id": target_op.op_id,
            "kernels": ["aclnnStageA_Kernel", "aclnnStageB_Kernel", "aclnnStageC_Kernel"],
        }],
    }), encoding="utf-8")

    rec_bundle = build_export_bundle(case, request=req, observations_path=obs_json)
    assert rec_bundle.reconciliation is not None
    assert rec_bundle.reconciliation.matched_kernels == 3 and len(rec_bundle.reconciliation.unmatched_kernels) == 1

    xlsx_path = tmp_path / "reconciled.xlsx"
    xlsx_rec = export_comparison_xlsx(rec_bundle, xlsx_path)
    assert xlsx_rec.has_measured is True and SHEET_MEASURED in xlsx_rec.sheets

    _, roots, shared = _read_xlsx_sheets(xlsx_path)
    meas_merges = _sheet_merges(roots[SHEET_MEASURED])
    meas_cells = _sheet_cells(roots[SHEET_MEASURED], shared)
    target_ref = next(ref for ref, (val, _) in meas_cells.items() if ref.startswith("C") and val == target_op.op_id)
    r_start = int(target_ref[1:])
    r_end = r_start + 2
    for col_let in ("C", "I", "L", "M", "N"):
        assert f"{col_let}{r_start}:{col_let}{r_end}" in meas_merges
    assert meas_cells[f"J{r_start}"][0] == "aclnnStageA_Kernel"
    assert meas_cells[f"J{r_end}"][0] == "aclnnStageC_Kernel"
    assert meas_cells[f"L{r_start}"][1] == f"SUM(K{r_start}:K{r_end})"
    assert meas_cells[f"M{r_start}"][1] == f"IF(L{r_start}>0,ABS(I{r_start}-L{r_start})/L{r_start},0)"


def test_submodule_and_kernel_hygiene_across_hybrid_architectures(tmp_path):
    catalog_matrix = (
        ("kimi-k3.1", 2),
        ("deepseek-v4.1-flash", 2),
        ("glm-5.3-flash", 2),
        ("qwen3.8-flash-next", 2),
        ("qwen3-32b", 1),
    )
    for model_name, ep in catalog_matrix:
        raw, req = compose_export_case(
            model_name, REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
            dp=1, tp=2, ep=ep, dcp=1, prompt_tokens=1024, output_tokens=32, concurrency=1,
        )
        bundle = build_export_bundle(raw, request=req)
        assert bundle.operators
        assert not [o.op_id for o in bundle.operators if o.submodule.lower() == "other"]
        assert not [o.ascend_kernel for o in bundle.operators if o.ascend_kernel in BANNED_IMPLEMENTATION_NAMES]

    raw_eager_case, raw_req = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=1, ep=1, dcp=1, prompt_tokens=1024, output_tokens=32, concurrency=1, feature_track="raw_eager",
    )
    b_raw = build_export_bundle(raw_eager_case, request=raw_req)
    ln_ops = [o for o in b_raw.operators if o.family == "layernorm"]
    rms_ops = [o for o in b_raw.operators if o.family == "rmsnorm"]
    ng_decomposed = [
        o for o in b_raw.operators
        if o.canonical_op_key.startswith(("linear_output_norm_", "linear_output_gate_"))
    ]
    assert ln_ops and all(o.ascend_kernel == "aclnnLayerNorm_LayerNormV3" for o in ln_ops)
    assert rms_ops and all("Cast->Pows->ReduceMean" in o.ascend_kernel for o in rms_ops)
    assert len(ng_decomposed) >= MIN_EAGER_NORM_GATE_OPS
    assert {"Cast", "Pows", "ReduceMean", "Adds", "Rsqrt", "Mul", "Sigmoid"} <= {
        o.ascend_kernel for o in ng_decomposed
    }

    # Verify PLE / Engram Host Offload visibility across Sheet 1/2/3 & H2D trace on hybrid GDN+PLE model.
    raw_ple, ple_req = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=2, dcp=1, prompt_tokens=1024, output_tokens=32,
        concurrency=2, engram_cpu_offload=True, feature_track="expected_ideal",
    )
    b_ple = build_export_bundle(raw_ple, request=ple_req)
    assert b_ple.memory_ledger.engram_host_offload is True and b_ple.memory_ledger.max_stage_host_offload_bytes > 0.0
    assert any(o.submodule == "Engram / PLE Memory" and "Host->Device UVA H2D" in o.ascend_kernel for o in b_ple.operators)
    ple_xlsx = tmp_path / "ple_host_offload.xlsx"
    ple_trace = tmp_path / "ple_trace.json"
    export_comparison_xlsx(b_ple, ple_xlsx)
    export_msprof_trace(b_ple, ple_trace)
    _, ple_roots, ple_shared = _read_xlsx_sheets(ple_xlsx)
    assert any("Engram / PLE Memory" in s for s in ple_shared) and any("Host DRAM Offload=" in s for s in ple_shared)
    assert "aclrtMemcpyAsync (PLE/Engram H2D" in ple_trace.read_text(encoding="utf-8")


def test_memory_quant_calculator_exclusions_offload_and_pp_budget(tmp_path):
    assert evaluate_weight_storage_bytes(6, 65, "MXFP4") == (198, 18, 0, 216)
    assert evaluate_weight_storage_bytes(16, 64, "W8A8") == (1024, 0, 128, 1152)
    assert evaluate_weight_storage_bytes(16, 64, "W4A8") == (512, 0, 256, 768)
    assert evaluate_weight_storage_bytes(16, 64, "MXFP8") == (1024, 32, 0, 1056)
    assert evaluate_weight_storage_bytes(16, 64, "BF16") == (2048, 0, 0, 2048)

    raw_k, _ = compose_export_case("kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=8, ep=32, dp=4, dcp=8)
    led_routed = compute_memory_quant_ledger(parse_case(raw_k), seq_len=4096, concurrency=4)
    assert led_routed.weight_quant == "MXFP4"
    assert led_routed.weight_quant_display == "MXFP4 (Routed+Latent+AttnRes, Attn/Shared BF16)"
    assert led_routed.include_vision_tower is False
    assert led_routed.native_bf16_weight_footprint_per_die_bytes > led_routed.weight_footprint_per_die_bytes
    by_name = {r.name: r for r in led_routed.weight_rows}
    assert by_name["q_a"].quant_format == "BF16" and by_name["q_a"].is_quant_excluded is True
    assert by_name["o_proj"].quant_format == "BF16" and by_name["dense_down"].quant_format == "BF16"
    assert by_name["latent_down"].quant_format == "MXFP4" and by_name["latent_up"].quant_format == "MXFP4"
    assert by_name["attn_res_proj"].quant_format == "MXFP4"
    assert any(r.category == "routed_expert" and r.quant_format == "MXFP4" for r in led_routed.weight_rows)
    assert all(r.quant_format == "BF16" for r in led_routed.weight_rows if r.category in {"shared_expert", "router"})

    # Verify mixed W4A8_DYNAMIC on legacy INT8/INT4 chip prices ordinary linears as W8A8 and routed experts as W4A8.
    raw_w4a8, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=8, ep=32, dp=4, dcp=8, quant="w4a8", kv_cache_dtype="int8",
    )
    led_w4a8 = compute_memory_quant_ledger(parse_case(raw_w4a8), seq_len=4096, concurrency=4)
    assert led_w4a8.weight_quant == "W4A8" and led_w4a8.cache_dtype == "C8"
    assert {r.quant_format for r in led_w4a8.weight_rows if r.group == "Attention / MLA" and not r.is_quant_excluded} == {"W8A8"}
    assert {r.quant_format for r in led_w4a8.weight_rows if r.category == "routed_expert"} == {"W4A8"}

    raw_case = json.loads(Path("examples/cases/deepseek-v4.1-flash.json").read_text(encoding="utf-8"))
    case = parse_case(raw_case)
    led_bf16_hbm = compute_memory_quant_ledger(
        case, seq_len=4096, concurrency=4, weight_quant="BF16", cache_dtype="BF16", engram_host_offload=False,
    )
    led_mxfp4_offload = compute_memory_quant_ledger(
        case, seq_len=4096, concurrency=4, weight_quant="MXFP4", cache_dtype="C8", engram_host_offload=True,
    )
    engram_hbm_total = sum(s.engram_ple_hbm_bytes for s in led_bf16_hbm.stage_budgets)
    engram_host_total = sum(s.engram_ple_host_bytes for s in led_mxfp4_offload.stage_budgets)
    assert engram_hbm_total > 0.0 and engram_host_total == pytest.approx(engram_hbm_total, rel=1e-9)
    assert all(s.engram_ple_hbm_bytes == 0.0 for s in led_mxfp4_offload.stage_budgets)

    xlsx_path = tmp_path / "memory_calc.xlsx"
    export_comparison_xlsx(
        raw_k, xlsx_path,
        request=StepRequest(phase="decode", tokens=1, context=512, sequences=2, scope="body"),
    )
    _, roots, shared = _read_xlsx_sheets(xlsx_path)
    mem_cells = _sheet_cells(roots[SHEET_MEMORY], shared)
    assert mem_cells["T16"][1] == "T8*T9*T11*T13"
    assert mem_cells["T21"][1] == "ROUNDUP(T17/(T19*T12),0)*T19"
    assert mem_cells["T24"][1] == "T22*T23"
    assert mem_cells["T25"][0] == "MXFP4 (Routed+Latent+AttnRes, Attn/Shared BF16)"
    assert mem_cells["T6"][0] == "OFF" and mem_cells["T7"][0] == "OFF"
    assert any(f and "SUMIFS(" in f and '"NPU HBM*"' in f for _, f in mem_cells.values())
    assert any(f and "SUMIFS(" in f and '"Host DRAM*"' in f for _, f in mem_cells.values())
    assert any(f and "VLOOKUP(" in f and "ROUNDUP(" in f for _, f in mem_cells.values())


def test_cli_export_commands_oneshot_receipt_isolation_and_error_guards(tmp_path, monkeypatch):
    sku, preset = resolve_hardware_alias("a5-144g-pod")
    assert sku == REPRESENTATIVE_SKUS["a5"] and preset == "a5_pod"

    designs_path = tmp_path / "designs.json"
    designs_path.write_text(json.dumps({"K3 SituAndMul": {"design": "CLI AIV fused kernel"}}), encoding="utf-8")
    oneshot_xlsx = tmp_path / "oneshot.xlsx"
    oneshot_trace = tmp_path / "oneshot_trace.json"
    oneshot_receipt = tmp_path / "oneshot_receipt.json"
    rc_one = cli_mod._main([
        "export", "--model", "qwen3-32b", "--hardware", "a5-pod",
        "--dp", "1", "--tp", "2", "--ep", "1", "--prompt-tokens", "512", "--output-tokens", "32",
        "--compare-track", "merged", "--target-track", "expected_ideal",
        "--xlsx", str(oneshot_xlsx), "--trace", str(oneshot_trace),
        "--kernel-designs-json", str(designs_path), "--receipt", str(oneshot_receipt),
    ])
    assert rc_one == 0 and zipfile.is_zipfile(oneshot_xlsx) and oneshot_trace.is_file()
    rec_one = json.loads(oneshot_receipt.read_text(encoding="utf-8"))
    assert rec_one["ok"] is True and rec_one["command"] == "export"
    assert rec_one["xlsx"]["has_comparison"] is True and rec_one["trace"]["total_events"] > 10

    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_compact_case(tp=2, ep=2, pp=1, layers=1)), encoding="utf-8")
    trace_out = tmp_path / "cli_trace_view.json"
    xlsx_out = tmp_path / "cli_report.xlsx"
    assert cli_mod._main(["export-trace", str(case_path), "-o", str(trace_out), "--context", "256"]) == 0
    assert cli_mod._main([
        "export-xlsx", str(case_path), "--output", str(xlsx_out), "--context", "256",
        "--weight-quant", "w4a8_mxfp", "--cache-dtype", "fp8_e4m3", "--engram-offload",
    ]) == 0
    assert cli_export.main(["export-xlsx", str(case_path), "-o", str(tmp_path / "standalone.xlsx"), "--context", "128"]) == 0

    wb = Workbook()
    ws = wb.create_sheet("OverlapCheck")
    ws.merge_cells("A1:B3")
    with pytest.raises(ValueError, match="overlapping_merge_range"):
        ws.merge_cells("B2:C4")

    def _boom(*_args, **_kwargs):
        raise TimeoutError("export_timeout_check")

    monkeypatch.setattr(cli_export, "export_msprof_trace", _boom)
    with pytest.raises(TimeoutError, match="export_timeout_check"):
        cli_export.main(["export-trace", str(case_path), "-o", str(tmp_path / "never.json")])


def test_agent_brief_digest_and_static_launch_script_audit(tmp_path):
    raw_bef, req = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=2, tp=2, ep=4, prompt_tokens=4096, output_tokens=64, concurrency=8,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.4, tpot_slo_ms=15.0,
        kv_cache_dtype="fp8_e4m3", engram_cpu_offload=True, feature_track="unmerged_pr",
    )
    raw_aft, _ = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=2, tp=2, ep=4, prompt_tokens=4096, output_tokens=64, concurrency=8,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.4, tpot_slo_ms=15.0,
        kv_cache_dtype="fp8_e4m3", engram_cpu_offload=True, feature_track="expected_ideal",
    )
    b_bef = build_export_bundle(raw_bef, request=req)
    b_aft = build_export_bundle(raw_aft, request=req)
    assert b_aft.step_span_us < b_bef.step_span_us
    assert not any("Qwen4Exp" in op.ascend_kernel for op in b_aft.operators)

    script_file = tmp_path / "serve.sh"
    script_file.write_text(
        "# sample user script\n"
        "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3\n"
        "export HCCL_OP_EXPANSION_MODE=AI_CPU\n"
        "vllm serve /weights/Qwen3.8-Flash-Next-bf16 \\\n"
        "  --tensor-parallel-size 4 --data-parallel-size 1 \\\n"
        "  --max-model-len 65536 --max-num-seqs 16 --enforce-eager\n",
        encoding="utf-8",
    )
    parsed = parse_launch_script(script_file)
    assert parsed.inferred_catalog_model == "qwen3.8-flash-next"
    assert parsed.inferred_quant == "bf16" and parsed.tp == 4 and parsed.enforce_eager is True
    assert parsed.inferred_num_devices == 4

    opt_sh_path = tmp_path / "optimized_serve.sh"
    opt_text = render_optimized_serve_sh(parsed, after_bundle=b_aft, output_path=opt_sh_path)
    assert opt_sh_path.is_file() and b"\r\n" not in opt_sh_path.read_bytes()
    assert 'HCCL_OP_EXPANSION_MODE="AIV"' in opt_text and "FULL_DECODE_ONLY" in opt_text
    assert "--quantization ascend" in opt_text

    audit_rep = audit_launch_script(
        parsed, before_bundle=b_bef, after_bundle=b_aft, optimized_script_path=str(opt_sh_path),
    )
    assert audit_rep.schema == SCRIPT_AUDIT_SCHEMA and len(audit_rep.issues) >= 4
    brief = build_agent_brief(b_bef, b_aft, script_audit=audit_rep.to_dict())
    assert brief["schema"] == AGENT_BRIEF_SCHEMA
    cb = brief["scenario_and_parallelism"].get("calibration_basis")
    assert isinstance(cb, dict) and isinstance(cb.get("calibrated"), bool)
    assert cb.get("mode") in {
        "calibrated_primitive_table",
        "calibrated_primitive_table_rescaled",
        "datasheet_declaration",
    }
    assert brief["kpis_before_vs_after"]["saved_step_ms"] == pytest.approx(
        brief["kpis_before_vs_after"]["sheet4_cards_total_gain_ms"], abs=0.01,
    )
    brief_out = export_agent_brief(brief, tmp_path / "brief.json")
    assert brief_out.stat().st_size < 8192
    digest = format_executive_digest_markdown(brief)
    assert "MindIE-Sim Executive Digest" in digest and "3-Track Action Plan" in digest


def test_bipartite_alignment_dual_conservation_zero_saving_guard_draft_kv_and_ag013(tmp_path):
    from dataclasses import replace
    import importlib.util
    from mindie_sim.export.comparison import format_parallelism_codomain_summary
    from mindie_sim.export.models import (
        BINDING_BADGE_BOUND,
        BINDING_BADGE_UNBOUND,
        BINDING_STATE_BOUND,
        BINDING_STATE_UNBOUND_FORMULA,
        KERNEL_STATUS_AT_FLOOR,
        MODE_MODELING,
        OperatorExportRow,
    )
    from mindie_sim.prediction.capacity import Memory

    # 1. Bug 4a: Memory ledger cache total matches Memory(case) including Draft KV Cache.
    raw_k, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=4, tp=8, ep=32, dcp=8, prompt_tokens=4096, output_tokens=64,
        concurrency=4, mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
    )
    parsed_k = parse_case(raw_k)
    led_k = compute_memory_quant_ledger(parsed_k, seq_len=4096, concurrency=4)
    assert any("draft" in row.component.lower() for row in led_k.cache_rows)
    mem_k = Memory(parsed_k, workspace_tokens=4 * 6)
    max_stage_mem_cache = max(mem_k.dp_cohort_cache_bytes(4096, padded=True)) * 4
    assert max(st.cache_bytes for st in led_k.stage_budgets) == pytest.approx(max_stage_mem_cache, rel=1e-6)

    base_raw = _compact_case(tp=2, ep=2, pp=1, layers=1)
    base_req = StepRequest(phase="decode", tokens=1, context=512, sequences=2, scope="model")
    tmpl = build_export_bundle(base_raw, request=base_req)
    proto_op = tmpl.operators[0]

    # 2. Synthetic Two-Level Bipartite Alignment covering 1->1, non-adjacent N->1 (fused_into_anchor),
    #    1->0 eliminated, 0->1 added, M->N cluster, zero-saving badge guard, and dual-metric conservation.
    def _mk_op(
        op_id: str, kernel: str, exp_us: float, atoms: tuple[str, ...], *,
        status: str = "现成可用", needs_new: bool = False,
        absorbed: tuple[str, ...] = (), comm: str = "", binding: str = BINDING_STATE_BOUND,
    ) -> OperatorExportRow:
        return replace(
            proto_op,
            archetype_id="arch_0", archetype_title="Layer Archetype 0 (×2)", repeat_count=2,
            layer_label="L0", submodule="Attention", op_id=op_id, canonical_op_key=op_id,
            logical_role=op_id, family="vector", recipe="vec", implementation="vec",
            ascend_kernel=kernel, kernel_backend="CANN", kernel_status=status,
            needs_new_kernel=needs_new, task_type="AI_VECTOR_CORE", stream_index=0,
            tensor_contract="[4, 1024] bf16", start_us=0.0, dur_us=exp_us, end_us=exp_us,
            exposed_dur_us=exp_us, l_star_lean_us=exp_us * 0.5, gap_us=exp_us * 0.5,
            is_overlapped=False, masked_by="",
            semantic_atoms=atoms, absorbed_source_ops=absorbed, comm_domain_tag=comm,
            binding_state=binding,
        )

    b_ops = (
        _mk_op("l0.part_a", "OpA", 10.0, ("atom.a",)),
        _mk_op("l0.mid_keep", "OpMid", 12.0, ("atom.mid",)),
        _mk_op("l0.part_b", "OpB", 8.0, ("atom.b",)),
        _mk_op("l0.old_comm", "AllReduce", 6.0, (), comm="TP=8"),
        _mk_op("l0.mn_1", "ClusterB1", 9.0, ("atom.c1", "atom.c2")),
        _mk_op("l0.mn_2", "ClusterB2", 5.0, ("atom.c2", "atom.c3")),
    )
    a_ops = (
        _mk_op("l0.fused_ab", "FusedAB", 11.0, ("atom.a", "atom.b"), status="[需新开发]", needs_new=True,
               absorbed=("l0.part_a", "l0.part_b"), binding=BINDING_STATE_UNBOUND_FORMULA),
        _mk_op("l0.mid_keep", "OpMid", 12.0, ("atom.mid",), status="[需新开发]", needs_new=True),
        _mk_op("l0.new_comm", "AllGather", 4.0, (), comm="DCP=8"),
        _mk_op("l0.mn_a1", "ClusterA1", 6.0, ("atom.c1", "atom.c2")),
        _mk_op("l0.mn_a2", "ClusterA2", 5.0, ("atom.c2", "atom.c3")),
    )
    b_step_us = sum(o.exposed_dur_us * o.repeat_count for o in b_ops)
    a_step_us = sum(o.exposed_dur_us * o.repeat_count for o in a_ops)
    b_bun = replace(
        tmpl, operators=b_ops, step_span_us=b_step_us,
        n_tok_per_die=2.0, us_per_tok_die=b_step_us / 2.0,
        throughput_per_die_tokens_per_s=1e6 / (b_step_us / 2.0),
        binding_state=BINDING_STATE_BOUND, has_unbound_formula_ops=False,
    )
    a_bun = replace(
        tmpl, operators=a_ops, step_span_us=a_step_us,
        n_tok_per_die=2.0, us_per_tok_die=a_step_us / 2.0,
        throughput_per_die_tokens_per_s=1e6 / (a_step_us / 2.0),
        analysis_mode=MODE_MODELING, binding_state=BINDING_STATE_UNBOUND_FORMULA,
        has_unbound_formula_ops=True,
    )
    # Co-domain parallelism banner formatting.
    par_short, _ = format_parallelism_codomain_summary(replace(a_bun, memory_ledger=led_k, sequence_parallel=True))
    assert par_short == "DP4 · TP8 (共域 DCP8) · EP32 · SP(开启)"

    aligned = align_before_after_operators(b_bun, a_bun)
    rels = [r.relation for r in aligned]
    assert {"N->1", "fused_into_anchor", "1->1", "eliminated", "added", "2->2"} <= set(rels)
    # Zero-saving status guard strips [需新开发] on l0.mid_keep (12us -> 12us).
    mid_row = next(r for r in aligned if r.relation == "1->1")
    assert mid_row.after_ops[0].needs_new_kernel is False
    assert mid_row.after_ops[0].kernel_status == KERNEL_STATUS_AT_FLOOR

    # Dual-metric conservation across Sheet 2 and Sheet 4.
    s2_b_upt = sum(sum(b.exposed_dur_us for b in r.before_ops) * r.repeat_count / 2.0 for r in aligned)
    s2_a_upt = sum(sum(a.exposed_dur_us for a in r.after_ops) * r.repeat_count / 2.0 for r in aligned)
    assert s2_b_upt == pytest.approx(b_bun.us_per_tok_die, abs=1e-6)
    assert s2_a_upt == pytest.approx(a_bun.us_per_tok_die, abs=1e-6)
    cards = build_pending_kernel_details(aligned, a_bun, before_bundle=b_bun)
    assert sum(c.saved_us_per_tok_die for c in cards) == pytest.approx(s2_b_upt - s2_a_upt, abs=1e-4)

    out_xlsx = tmp_path / "bipartite_dual.xlsx"
    export_comparison_xlsx(b_bun, out_xlsx, compare_source=a_bun)
    _, roots, shared = _read_xlsx_sheets(out_xlsx)
    assert any(BINDING_BADGE_UNBOUND in s for s in shared)
    assert any(BINDING_BADGE_BOUND in s for s in shared)
    s2_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    assert s2_cells["X2"][0] == "Before 摊薄耗时 (us/tok·die)"
    assert s2_cells["AA2"][0] == "绑定状态 (Binding State)"

    # 2b. #446 mixed-precision matrix_flops L* pricing (fp8_e4m3 / mxfp4 flops vs bf16 output dtype).
    from mindie_sim.export.bundle import _die_peaks_for_stage, _op_lean_resource_demands_ns
    peaks_a5 = _die_peaks_for_stage(parsed_k, 0)
    dem_bf16, rate_bf16 = _op_lean_resource_demands_ns(
        flops=1e12, vector_ops=0.0, hbm_read=0.0, hbm_write=0.0,
        link_bytes=0.0, dtype="bf16", peaks=peaks_a5,
    )
    dem_fp8, rate_fp8 = _op_lean_resource_demands_ns(
        flops=1e12, vector_ops=0.0, hbm_read=0.0, hbm_write=0.0,
        link_bytes=0.0, dtype="bf16", peaks=peaks_a5,
        matrix_flops=(("fp8_e4m3", 1e12),),
    )
    assert rate_fp8 > rate_bf16
    assert dem_fp8[0] < dem_bf16[0]

    # 3. Hardened AG013 anti-hardcoding gate catches model strings, static suffix tables & TARGET_ETA.
    import ast as _ast
    import sys as _sys
    from pathlib import PurePosixPath as _PurePosixPath
    gate_path = Path(__file__).resolve().parents[2] / "scripts" / "check_arch_governance.py"
    scripts_dir = str(gate_path.parent)
    added_path = scripts_dir not in _sys.path
    if added_path:
        _sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location("_ag_gate_test", gate_path)
        assert spec is not None and spec.loader is not None
        ag_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ag_mod)
    finally:
        if added_path and scripts_dir in _sys.path:
            _sys.path.remove(scripts_dir)

    sandbox = tmp_path / "ag_repo"
    target_dir = sandbox / "mindie_sim" / "export"
    target_dir.mkdir(parents=True)
    bad_file = target_dir / "bad_rules.py"
    bad_file.write_text(
        '_PURE_GLUE_EXACT_SUFFIXES = ("a",)\n'
        'TARGET_ETA_HBM = 0.85\n'
        'def pick(x):\n'
        '    return "kimi_k3" if x else "ok"\n',
        encoding="utf-8",
    )
    rel_bad = _PurePosixPath("mindie_sim/export/bad_rules.py")
    viols = ag_mod._check_ag013_anti_hardcoding(rel_bad, _ast.parse(bad_file.read_text(encoding="utf-8")))
    assert any(v.rule == "AG013" and "suffix" in v.message.lower() for v in viols)
    assert any(v.rule == "AG013" and "target_eta" in v.message.lower() for v in viols)
    assert any(v.rule == "AG013" and "kimi" in v.message.lower() for v in viols)
    bad_file.write_text('def clean(x):\n    return "generic_op" if x else "ok"\n', encoding="utf-8")
    assert ag_mod._check_ag013_anti_hardcoding(rel_bad, _ast.parse(bad_file.read_text(encoding="utf-8"))) == []

    # Generalized AG013 catches arbitrary module-level suffix/op-name matching constant tables.
    custom_tbl_src = (
        '_CUSTOM_OP_SUFFIXES = ("rms_norm", "silu_mul")\n'
        '_EXTRA_OP_NAMES = {"q_proj", "k_proj"}\n'
    )
    gen_viols = ag_mod._check_ag013_anti_hardcoding(rel_bad, _ast.parse(custom_tbl_src))
    assert len(gen_viols) >= 2 and all(v.rule == "AG013" for v in gen_viols)


def test_exporter_rebound_primitives_horizon_pp_conservation_formulas_and_brief_budget(tmp_path, monkeypatch):
    """Verify rebound primitive forwarding, Sheet 3 horizon/fixed state, PP1->PP2 conservation, live formulas & brief budget."""
    from dataclasses import replace

    from mindie_sim.export import bundle as bundle_mod
    from mindie_sim.export.comparison import AGENT_BRIEF_MAX_BYTES, _compact_agent_brief_to_budget
    from mindie_sim.export.kernel_catalog import extract_semantic_atoms
    from mindie_sim.export.memory_ledger import _compute_max_hbm_concurrency
    from mindie_sim.export.models import LayerArchetypeSpec

    # 1. Rebound primitive_table flows into solve_target_graph during target modeling.
    captured_pts: list[object] = []
    orig_solve = bundle_mod.solve_target_graph

    def _spy_solve(*args, **kwargs):
        captured_pts.append(kwargs.get("primitive_table"))
        return orig_solve(*args, **kwargs)

    monkeypatch.setattr(bundle_mod, "solve_target_graph", _spy_solve)
    raw_c, req = compose_export_case(
        "qwen3-32b", "a5", tp=2, prompt_tokens=512, output_tokens=128,
        concurrency=4, feature_track="merged",
    )
    raw_c["model"]["layers"] = 2
    raw_c["model"]["layer_plans"] = raw_c["model"]["layer_plans"][:2]
    # Perturb declared HBM bandwidth by 0.5x on the parsed Case to verify rebound primitive_table flows into solve_target_graph.
    parsed_base = parse_case(raw_c)
    perturbed_dies = tuple(
        replace(
            die,
            rates=tuple(
                (k, float(v) * 0.5 if k in {"hbm", "hbm_read", "hbm_write"} else v)
                for k, v in die.rates
            ),
        )
        for die in parsed_base.machine.dies
    )
    perturbed_case = replace(parsed_base, machine=replace(parsed_base.machine, dies=perturbed_dies))
    b_base_pt = build_export_bundle(parsed_base, request=req, apply_target_envelope=True)
    b_target = build_export_bundle(perturbed_case, request=req, apply_target_envelope=True)
    assert len(captured_pts) >= 2 and captured_pts[0] is not None and captured_pts[1] is not None
    assert captured_pts[1] is not captured_pts[0]
    assert b_target.step_span_us > b_base_pt.step_span_us
    # Decode horizon includes declared output_tokens (context + output_tokens = 640).
    assert b_target.memory_ledger.seq_len == 512 + 128

    # 2. Fixed per-request state buffers stay non-zero in Sheet 3 Excel formulas even with 0 target layers.
    fixed_row = replace(
        b_target.memory_ledger.cache_rows[0],
        component="draft_state",
        max_stage_layers=0,
        bytes_per_token_per_layer=0.0,
        per_request_bytes=4 * 1024 * 1024,
        batch_total_bytes=16 * 1024 * 1024,
        scales_with_tokens=False,
    )
    led_fixed = replace(b_target.memory_ledger, cache_rows=(*b_target.memory_ledger.cache_rows, fixed_row))
    bun_fixed = replace(b_target, memory_ledger=led_fixed)
    fixed_xlsx = tmp_path / "fixed_state_and_formulas.xlsx"
    b_bef = build_export_bundle(raw_c, request=req, apply_target_envelope=False)
    b_bef_frac = replace(b_bef, n_tok_per_die=0.0546875, us_per_tok_die=b_bef.step_span_us / 0.0546875)
    b_aft_frac = replace(bun_fixed, n_tok_per_die=0.0546875, us_per_tok_die=bun_fixed.step_span_us / 0.0546875)
    export_comparison_xlsx(b_bef_frac, fixed_xlsx, compare_source=b_aft_frac)
    _, roots, shared = _read_xlsx_sheets(fixed_xlsx)

    # Sheet 3 fixed state row keeps numeric MiB (never zeroed by layer_count=0) and scales by concurrency.
    s3_cells = _sheet_cells(roots[SHEET_MEMORY], shared)
    draft_rows = [int(k[1:]) for k, (v, _) in s3_cells.items() if k.startswith("A") and v == "draft_state"]
    assert draft_rows
    dr = draft_rows[0]
    assert s3_cells[f"E{dr}"][1] is None and float(s3_cells[f"E{dr}"][0]) == pytest.approx(4.0, abs=1e-4)
    assert s3_cells[f"F{dr}"][1] == f"E{dr}*$T$18/1024"

    # Sheet 1 stores exact full-precision N_tok_per_die on Row 26; Sheet 2 & Sheet 4 reference Summary!$B$26/$C$26.
    s1_cells = _sheet_cells(roots[SHEET_SUMMARY], shared)
    assert float(s1_cells["B26"][0]) == pytest.approx(0.0546875, abs=1e-12)
    assert float(s1_cells["C26"][0]) == pytest.approx(0.0546875, abs=1e-12)
    s2_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    s2_formulas = [f for _, f in s2_cells.values() if f]
    assert any("Summary!$B$26" in f for f in s2_formulas)
    assert any("Summary!$C$26" in f for f in s2_formulas)
    assert not any("0.054688" in f for f in s2_formulas)
    s4_cells = _sheet_cells(roots[SHEET_PENDING_KERNELS], shared)
    s4_formulas = [f for _, f in s4_cells.values() if f]
    assert any(f.startswith("M") and "-N" in f for f in s4_formulas)
    assert any("Summary!$B$26" in f and "Summary!$C$26" in f for f in s4_formulas)

    # 3. Identical Before/After reports zero pending cards & 0.0 delta (never standalone gap_us).
    aligned_same = align_before_after_operators(b_bef, b_bef)
    assert build_pending_kernel_details(aligned_same, b_bef, before_bundle=b_bef) == ()
    # Zero-gain N->1 attn_res_prefix emits no new-kernel card.
    proto = b_bef.operators[0]
    ar_b1 = replace(proto, op_id="l0.ar_pre", family="attn_res_prefix", exposed_dur_us=5.0, dur_us=5.0,
                    l_star_lean_us=1.0, gap_us=4.0, repeat_count=1, semantic_atoms=("attn_res_prefix",))
    ar_b2 = replace(proto, op_id="l0.ar_post", family="attn_res_prefix", exposed_dur_us=5.0, dur_us=5.0,
                    l_star_lean_us=1.0, gap_us=4.0, repeat_count=1, semantic_atoms=("attn_res_prefix",))
    ar_a = replace(proto, op_id="l0.ar_fused", family="attn_res_prefix", exposed_dur_us=10.0, dur_us=10.0,
                   l_star_lean_us=2.0, gap_us=8.0, repeat_count=1, needs_new_kernel=True,
                   absorbed_source_ops=("l0.ar_pre", "l0.ar_post"), semantic_atoms=("attn_res_prefix",))
    bun_ar_b = replace(b_bef, operators=(ar_b1, ar_b2), step_span_us=10.0)
    bun_ar_a = replace(b_bef, operators=(ar_a,), step_span_us=10.0)
    assert build_pending_kernel_details(
        align_before_after_operators(bun_ar_b, bun_ar_a), bun_ar_a, before_bundle=bun_ar_b,
    ) == ()

    # 4. PP1 -> PP2 physical layer weight conservation (4 layers stay 4L, never 4+2=6L).
    op_pp1 = replace(proto, archetype_id="target_arch_0", repeat_count=4, exposed_dur_us=10.0, dur_us=10.0,
                     op_id="target.l0.q_proj", canonical_op_key="q_proj")
    op_pp2_s0 = replace(proto, archetype_id="target_arch_0", repeat_count=2, exposed_dur_us=8.0, dur_us=8.0,
                        op_id="target.l0.q_proj", canonical_op_key="q_proj")
    op_pp2_s1 = replace(proto, archetype_id="target_arch_1", repeat_count=2, exposed_dur_us=8.0, dur_us=8.0,
                        op_id="target.l2.q_proj", canonical_op_key="q_proj")
    arch_pp1 = (LayerArchetypeSpec("target_arch_0", "target", "PP1 (×4)", "Attn", 4, (0, 1, 2, 3), "L0..L3", 0),)
    arch_pp2 = (
        LayerArchetypeSpec("target_arch_0", "target", "PP2 S0 (×2)", "Attn", 2, (0, 1), "L0..L1", 0),
        LayerArchetypeSpec("target_arch_1", "target", "PP2 S1 (×2)", "Attn", 2, (2, 3), "L2..L3", 1),
    )
    bun_pp1 = replace(b_bef, archetypes=arch_pp1, operators=(op_pp1,), step_span_us=40.0, total_target_layers=4)
    bun_pp2 = replace(b_bef, archetypes=arch_pp2, operators=(op_pp2_s0, op_pp2_s1), step_span_us=32.0, total_target_layers=4)
    aligned_pp = align_before_after_operators(bun_pp1, bun_pp2)
    assert sum(r.effective_before_repeat for r in aligned_pp) == 4
    assert sum(r.effective_after_repeat for r in aligned_pp) == 4
    pp_s2_bef = sum(sum(o.exposed_dur_us for o in r.before_ops) * r.effective_before_repeat for r in aligned_pp)
    pp_s2_aft = sum(sum(o.exposed_dur_us for o in r.after_ops) * r.effective_after_repeat for r in aligned_pp)
    assert pp_s2_bef == pytest.approx(40.0, abs=1e-6) and pp_s2_aft == pytest.approx(32.0, abs=1e-6)
    pp_macros = build_macro_module_breakdown(bun_pp1, bun_pp2)
    assert sum(m.before_ms for m in pp_macros) == pytest.approx(0.04, abs=1e-6)
    assert sum(m.after_ms for m in pp_macros) == pytest.approx(0.032, abs=1e-6)
    assert all(m.layer_count <= 4 for m in pp_macros)
    pp_cards = build_pending_kernel_details(aligned_pp, bun_pp2, before_bundle=bun_pp1)
    assert sum(c.expected_gain_ms for c in pp_cards) == pytest.approx(0.008, abs=1e-6)

    # 5. Distinct semantic atoms (q_rotary vs kv_rotary) and error propagation in _extract_structure_registry_view.
    assert extract_semantic_atoms("l0.q_rotary", "rope", "rope") == ("q_rotary",)
    assert extract_semantic_atoms("l0.kv_rotary", "rope", "rope") == ("kv_rotary",)
    import mindie_sim.structure as struct_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected_structure_failure")

    monkeypatch.setattr(struct_mod, "export_op_structure", _boom)
    with pytest.raises(RuntimeError, match="injected_structure_failure"):
        bundle_mod._extract_structure_registry_view(parse_case(raw_c), {0: ()}, req)

    # 6. Agent brief budget compactor enforces strictly < 8192 bytes including trailing newline.
    huge_brief = {
        "schema": AGENT_BRIEF_SCHEMA,
        "summary": {"model": "test"},
        "va_serving_rules": {f"rule_{i}": "x" * 200 for i in range(30)},
        "top_pending_kernels": [{"card_id": f"K{i}", "detail": "y" * 200} for i in range(25)],
        "rewrite_candidates": [{"id": f"C{i}", "desc": "z" * 150} for i in range(20)],
        "macro_module_breakdown": [{"module": f"M{i}", "notes": "w" * 150} for i in range(20)],
        "hardware_microarch_summary": {"info": "v" * 2000},
    }
    compacted = _compact_agent_brief_to_budget(huge_brief)
    brief_out = tmp_path / "compacted_brief.json"
    export_agent_brief(compacted, brief_out)
    assert brief_out.stat().st_size < AGENT_BRIEF_MAX_BYTES
    # Binary-search _compute_max_hbm_concurrency stays exact when workspace scales with batch.
    parsed_c = parse_case(raw_c)
    assert _compute_max_hbm_concurrency(
        parsed_c, b_target.memory_ledger.stage_budgets,
        [st.cache_bytes / max(1, b_target.memory_ledger.concurrency) for st in b_target.memory_ledger.stage_budgets],
        b_target.memory_ledger.concurrency, phase="decode", resolved_seq=b_target.memory_ledger.seq_len,
    ) == b_target.memory_ledger.max_hbm_concurrency_per_dp


def test_export_binding_conservation_pp_reconciliation_and_brief_precision(tmp_path):
    from dataclasses import replace
    from mindie_sim.api import evaluate_step
    from mindie_sim.compose import compose_case
    from mindie_sim.export import (
        BINDING_STATE_BOUND,
        BINDING_STATE_MEASURED,
        BINDING_STATE_UNBOUND_FORMULA,
        audit_launch_script,
        build_agent_brief,
        export_agent_brief,
        parse_launch_script,
    )
    from mindie_sim.export import bundle as bundle_mod
    from mindie_sim.export.comparison import AGENT_BRIEF_MAX_BYTES, _compact_agent_brief_to_budget
    from mindie_sim.schema.common import ModelingError

    # 1. Single-source hardware alias resolution matches compose.inputs (a5-pod -> 9581 144GiB).
    assert resolve_hardware_alias("a5-pod") == ("Ascend950DT_9581", "a5_pod")
    assert resolve_hardware_alias("a3-8p") == ("Ascend910_9392", "a3_8p")

    # 2. _row_to_target_node preserves binding_state (never dropping unbound tags).
    raw_base = compose_case("qwen3-32b", "a3", tp=2, layers=2, prompt_tokens=256, output_tokens=32)
    req_base = StepRequest(phase="decode", tokens=1, context=256, sequences=1, scope="model")
    bun_base = build_export_bundle(raw_base, request=req_base, apply_target_envelope=False)
    unbound_row = replace(bun_base.operators[0], binding_state=BINDING_STATE_UNBOUND_FORMULA)
    target_node = bundle_mod._row_to_target_node(unbound_row, None)
    assert target_node.binding_state == BINDING_STATE_UNBOUND_FORMULA

    # 3. Unmatched observations (matched_sim_ops == 0) never stamp BINDING_STATE_MEASURED.
    csv_empty = tmp_path / "empty_kernels.csv"
    csv_empty.write_text(
        "Device_id,Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us),Input Shapes,Input Data Types\n"
        '0,1,1,CompletelyUnrelatedKernel,Other,AI_VECTOR_CORE,100.0,1.0,"""1""",DT_BF16\n',
        encoding="utf-8",
    )
    bun_unmatched = build_export_bundle(raw_base, request=req_base, observations_path=csv_empty, apply_target_envelope=False)
    assert bun_unmatched.reconciliation is not None and bun_unmatched.reconciliation.matched_sim_ops == 0
    assert bun_unmatched.binding_state == BINDING_STATE_BOUND

    # 4. Tuple sequences normalize via cohort_sequence_counts (cluster_concurrency = sum, HBM = per-replica).
    raw_dp2 = compose_case("qwen3-32b", "a3", dp=2, tp=2, layers=2, prompt_tokens=256, output_tokens=32)
    req_tup = StepRequest(phase="decode", tokens=1, context=256, sequences=(8, 8), scope="body")
    bun_tup = build_export_bundle(raw_dp2, request=req_tup, apply_target_envelope=False)
    assert bun_tup.sequences == 8 and bun_tup.cluster_concurrency == 16
    assert bun_tup.memory_ledger.concurrency == 8
    tup_brief = build_agent_brief(bun_tup)
    assert tup_brief["scenario_and_parallelism"]["cluster_concurrency"] == 16

    # 5. step_eval preserves primitive_table=False & rejects rewrite_plan with apply_target_envelope=False.
    step_eval = evaluate_step(raw_base, req_base, primitive_table=False)
    bun_se = build_export_bundle(raw_base, request=req_base, step_eval=step_eval, primitive_table=False)
    assert bun_se.step_span_us > 0.0
    with pytest.raises(ModelingError, match="rewrite_plan_conflicts_with_apply_target_envelope_false"):
        build_export_bundle(raw_base, request=req_base, step_eval=step_eval, rewrite_plan={"edits": []}, apply_target_envelope=False)

    # 6. PP2 peak-stage weight conservation (untied embed on S0 + lm_head on S1) & kv_block_size lookup.
    raw_pp2 = compose_case("qwen3-32b", "a3", tp=2, pp=2, layers=4, prompt_tokens=256, output_tokens=32)
    raw_pp2["runtime"]["kv_block_size"] = 64
    raw_pp2["runtime"]["pp_layer_partition"] = [3, 1]
    parsed_pp2 = parse_case(raw_pp2)
    rep_pp2 = bundle_mod._build_archetype_representative_case(
        parsed_pp2, {0: parsed_pp2.model.layer_plans[0], 1: parsed_pp2.model.layer_plans[3]},
        allow_draft_dedup=True,
    )
    assert rep_pp2.runtime.pp_layer_partition == (1, 1)
    led_pp2 = compute_memory_quant_ledger(parsed_pp2, phase="decode", seq_len=512, concurrency=2)
    assert led_pp2.block_size == 64
    hbm_rows = [
        r for r in led_pp2.weight_rows
        if not (r.is_engram_or_ple and led_pp2.engram_host_offload)
        and not (r.is_vision and not led_pp2.include_vision_tower)
    ]
    assert sum(r.per_die_bytes for r in hbm_rows) == pytest.approx(led_pp2.weight_footprint_per_die_bytes, rel=1e-9)

    # 7. Sheet 4 links M/N live to Sheet 2, reconciles unchanged exposure, and keeps bound Pillar-1 fusions bound.
    op0, op1 = bun_base.operators[0], bun_base.operators[1]
    b_op0 = replace(op0, exposed_dur_us=10000.0, dur_us=10000.0, repeat_count=1, is_overlapped=False)
    b_op1 = replace(op1, exposed_dur_us=4000.0, dur_us=4000.0, repeat_count=1, is_overlapped=False)
    a_op0 = replace(b_op0)  # Unchanged 10 ms -> 10 ms
    a_op1 = replace(
        b_op1, exposed_dur_us=2000.0, dur_us=2000.0, opt_pillar="pillar1_fusion",
        needs_new_kernel=False, binding_state=BINDING_STATE_BOUND,
    )
    bun_b = replace(bun_base, operators=(b_op0, b_op1), step_span_us=14000.0, handoff_tpot_ms=4.0, tpot_slo_ms=16.0)
    bun_a = replace(bun_base, operators=(a_op0, a_op1), step_span_us=12000.0, handoff_tpot_ms=4.0, tpot_slo_ms=16.0)
    aligned_cmp = align_before_after_operators(bun_b, bun_a)
    cards_cmp = build_pending_kernel_details(aligned_cmp, bun_a, before_bundle=bun_b)
    assert len(cards_cmp) == 1
    assert cards_cmp[0].category == PENDING_KIND_TO_OPTIMIZE
    assert cards_cmp[0].binding_state == BINDING_STATE_BOUND

    xlsx_live = tmp_path / "live_sheet4_and_s3.xlsx"
    export_comparison_xlsx(bun_b, xlsx_live, compare_source=bun_a)
    _, roots, shared = _read_xlsx_sheets(xlsx_live)
    s1_cells = _sheet_cells(roots[SHEET_SUMMARY], shared)
    assert any(f and "+4.0" in f for _, f in s1_cells.values())
    s3_cells = _sheet_cells(roots[SHEET_MEMORY], shared)
    assert any(f and "$T$18" in f and "E" in f for _, f in s3_cells.values())
    s4_cells = _sheet_cells(roots[SHEET_PENDING_KERNELS], shared)
    s4_m_formulas = [f for ref, (_, f) in s4_cells.items() if ref.startswith("M") and f]
    s4_n_formulas = [f for ref, (_, f) in s4_cells.items() if ref.startswith("N") and f]
    assert any(SHEET_BREAKDOWN in f for f in s4_m_formulas)
    assert any(SHEET_BREAKDOWN in f for f in s4_n_formulas)
    assert any(v == "Unchanged Operators Exposure Reconciliation" for v, _ in s4_cells.values())

    # 8. Handoff TPOT flows into agent_brief SLA verdict + full-precision KPIs + CJK summary truncation < 8192 B.
    brief_hf = build_agent_brief(bun_b, bun_a)
    assert brief_hf["kpis_before_vs_after"]["before"]["effective_tpot_ms"] == pytest.approx(18.0, abs=1e-9)
    assert brief_hf["kpis_before_vs_after"]["before"]["sla_verdict"] == "FAIL"
    assert brief_hf["kpis_before_vs_after"]["after"]["effective_tpot_ms"] == pytest.approx(16.0, abs=1e-9)
    assert brief_hf["kpis_before_vs_after"]["after"]["sla_verdict"] == "PASS"
    bun_prec = replace(
        bun_base, n_tok_per_die=0.0546875, step_span_us=1000.0,
        us_per_tok_die=1000.0 / 0.0546875, throughput_per_die_tokens_per_s=54.6875,
    )
    prec_kpi = build_agent_brief(bun_prec)["kpis_before_vs_after"]["after"]
    assert prec_kpi["n_tok_per_die"] == pytest.approx(0.0546875, abs=1e-12)
    assert (prec_kpi["n_tok_per_die"] * 1000.0 / prec_kpi["step_ms"]) == pytest.approx(prec_kpi["single_die_tps"], rel=1e-12)
    cjk_brief = {"schema": AGENT_BRIEF_SCHEMA, "summary": "界" * 3000}
    cjk_out = tmp_path / "cjk_brief.json"
    export_agent_brief(_compact_agent_brief_to_budget(cjk_brief), cjk_out)
    assert cjk_out.stat().st_size < AGENT_BRIEF_MAX_BYTES

    # 9. script_audit respects spec_k == 0 without model-name speculative override.
    sh_path = tmp_path / "nospec.sh"
    sh_path.write_text("vllm serve deepseek-v4-pro-0813 --tensor-parallel-size 2 --data-parallel-size 1\n", encoding="utf-8")
    parsed_sh = parse_launch_script(sh_path)
    bun_nospec = replace(bun_base, model_name="deepseek-v4-pro-0813", spec_proposals=0)
    audit_rep = audit_launch_script(parsed_sh, before_bundle=bun_nospec, after_bundle=bun_nospec)
    assert not any(issue.issue_id == "draft_tp_policy_mismatch" for issue in audit_rep.issues)


def test_export_streams_spec_batch_quant_preset_cache_t4_pd_cluster_and_overlap_lstar(tmp_path, monkeypatch):
    from dataclasses import replace
    from mindie_sim.cli import _main as cli_main
    from mindie_sim.compose import compose_case
    from mindie_sim.export import (
        SHEET_BREAKDOWN,
        SHEET_MEMORY,
        SHEET_SUMMARY,
        build_agent_brief,
        build_export_bundle,
        compute_memory_quant_ledger,
        export_comparison_xlsx,
    )
    from mindie_sim.export import bundle as bundle_mod
    from mindie_sim.export.models import OperatorExportRow
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.workload import RequestView

    # 1. Ordinary prefill (spec_tokens=0) never infers proposals from prompt length;
    #    1-token decode with configured MTP resolves avg_accepted == 1.0 and spec_method == "None".
    raw_plain = compose_case("qwen3-32b", "a3", tp=2, layers=2, prompt_tokens=256, output_tokens=32)
    bun_pre = build_export_bundle(
        raw_plain, request=StepRequest(phase="prefill", tokens=256, context=0, sequences=1, scope="body"),
        apply_target_envelope=False,
    )
    assert bun_pre.spec_method == "None" and bun_pre.spec_proposals == 0
    raw_mtp = compose_case(
        "deepseek-v3.2", "a5", tp=8, ep=8, quant="mxfp4", prompt_tokens=256, output_tokens=32,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.5,
    )
    bun_dec1 = build_export_bundle(
        raw_mtp, request=StepRequest(phase="decode", tokens=1, context=256, sequences=1, scope="body"),
        apply_target_envelope=False,
    )
    assert bun_dec1.spec_method == "None" and bun_dec1.spec_proposals == 0
    assert bun_dec1.avg_accepted_tokens == pytest.approx(1.0, abs=1e-12)

    # 2. Explicit StepRequest.requests active count governs bundle concurrency & KV ledger.
    views = tuple(
        RequestView(request_id=f"r{i}", q_len=1, context=128, samples=True, phase="decode", padding=(i >= 4))
        for i in range(6)
    )
    req_views = StepRequest(phase="decode", tokens=1, context=128, sequences=1, scope="body", requests=views)
    bun_views = build_export_bundle(raw_mtp, request=req_views, apply_target_envelope=False)
    assert bun_views.sequences == 4 and bun_views.cluster_concurrency == 4
    assert bun_views.memory_ledger.concurrency == 4

    # 3. Target synthesis sees original recurrent_scan0 DAG identities before post-pricing display collapse.
    seen_target_ids: list[str] = []
    orig_solve = bundle_mod.solve_target_graph

    def _spy_solve(nodes, *a, **kw):
        seen_target_ids.extend(n.op_id for n in nodes)
        return orig_solve(nodes, *a, **kw)

    raw_gdn = compose_case("kimi-k3.1", "a5", tp=1, ep=1, layers=2, prompt_tokens=128, output_tokens=16)
    with monkeypatch.context() as mp:
        mp.setattr(bundle_mod, "solve_target_graph", _spy_solve)
        bun_gdn = build_export_bundle(
            raw_gdn, request=StepRequest(phase="decode", tokens=1, context=128, sequences=1, scope="body"),
            apply_target_envelope=True,
        )
    assert any(oid.endswith("recurrent_scan0") for oid in seen_target_ids)
    assert not any(o.op_id.endswith(("recurrent_scan0", "recurrent_scan1")) for o in bun_gdn.operators)

    # 4. Stream plans forward into spine.rewrite.apply on the direct bundle path.
    from mindie_sim.prediction.rewrite import RewritePipeline
    seen_streams: list[tuple[object, ...]] = []
    orig_apply = RewritePipeline.apply

    def _spy_apply(self, base, *, streams=()):
        seen_streams.append(tuple(streams))
        return orig_apply(self, base, streams=streams)

    raw_base = compose_case("qwen3-32b", "a3", tp=2, layers=1, prompt_tokens=64, output_tokens=8)
    with monkeypatch.context() as mp:
        mp.setattr(RewritePipeline, "apply", _spy_apply)
        build_export_bundle(
            raw_base, request=StepRequest(phase="decode", tokens=1, context=64, sequences=1, scope="body"),
            apply_target_envelope=False, streams=(),
        )
    assert len(seen_streams) >= 1

    # 5. Catalog quant preset strings (w8a8_dynamic / auto_va) work on CLI one-shot export and SDK xlsx entry.
    out_xlsx = tmp_path / "auto_va.xlsx"
    export_comparison_xlsx(
        None, out_xlsx, model="qwen3-32b", hardware="a3", tp=2,
        prompt_tokens=128, output_tokens=16, concurrency=1, weight_quant="auto_va",
    )
    assert out_xlsx.is_file()
    cli_xlsx = tmp_path / "cli_w8a8.xlsx"
    rc = cli_main([
        "export", "--model", "qwen3-32b", "--hardware", "a3", "--tp", "2", "--concurrency", "1",
        "--prompt-tokens", "128", "--output-tokens", "16", "--weight-quant", "w8a8_dynamic",
        "--xlsx", str(cli_xlsx),
    ])
    assert rc == 0 and cli_xlsx.is_file()

    # 6. DCP cohort cache bytes scale with cache-dtype override (C8 < BF16, never dropped to 0).
    raw_dcp = compose_case(
        "deepseek-v3.2", "a3", dp=2, tp=2, dcp=4, ep=4, layers=4, prompt_tokens=512, output_tokens=32,
    )
    parsed_dcp = parse_case(raw_dcp)
    led_bf16 = compute_memory_quant_ledger(parsed_dcp, phase="decode", seq_len=512, concurrency=2, cache_dtype="BF16")
    led_c8 = compute_memory_quant_ledger(parsed_dcp, phase="decode", seq_len=512, concurrency=2, cache_dtype="C8")
    cohort_bf16 = [r for r in led_bf16.cache_rows if r.component == "dcp_dp_cohort_kv"]
    cohort_c8 = [r for r in led_c8.cache_rows if r.component == "dcp_dp_cohort_kv"]
    assert cohort_bf16 and cohort_c8
    assert 0.0 < cohort_c8[0].per_request_bytes < cohort_bf16[0].per_request_bytes

    # 7. Sheet 3 T4 live cache-dtype formula, Sheet 1 PD cluster_tps + brief parity, Sheet 2 overlapped L* == '=0'.
    hidden_op = replace(bun_dec1.operators[0], exposed_dur_us=0.0, l_star_lean_us=50.0, is_overlapped=True)
    visible_op = replace(bun_dec1.operators[1], exposed_dur_us=200.0, dur_us=200.0, l_star_lean_us=120.0, is_overlapped=False)
    pd_pool = {
        "pd_ratio": "1P:2D", "pd_transfer_mode": "layerwise_overlap_d2d",
        "prefill_instances": 1, "decode_instances": 2,
        "prefill_used_dies": 2, "decode_used_dies": 4, "used_dies": 6,
        "handoff_tpot_ms": 0.5, "prefill_kv_rps": 10.0, "e2e_total_tps": 320.0,
    }
    bun_pd = replace(
        bun_dec1, operators=(hidden_op, visible_op), step_span_us=200.0,
        throughput_per_die_tokens_per_s=100.0, pd_pool_summary=pd_pool, output_tokens=32,
    )
    assert build_agent_brief(bun_pd)["kpis_before_vs_after"]["after"]["cluster_tps"] == pytest.approx(320.0, rel=1e-9)
    pd_xlsx = tmp_path / "pd_sheet_checks.xlsx"
    export_comparison_xlsx(bun_pd, pd_xlsx)
    _, roots, shared = _read_xlsx_sheets(pd_xlsx)
    s1_cells = _sheet_cells(roots[SHEET_SUMMARY], shared)
    assert float(s1_cells["B18"][0]) == pytest.approx(320.0, rel=1e-9)
    assert s1_cells["B18"][1] is not None and "MIN(" in s1_cells["B18"][1]
    s2_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    assert any(ref.startswith("M") and f in {"0", "=0"} and float(v or 0.0) == 0.0 for ref, (v, f) in s2_cells.items())
    s3_cells = _sheet_cells(roots[SHEET_MEMORY], shared)
    assert any(ref.startswith("D") and f and 'IF($T$4="BF16"' in f for ref, (_, f) in s3_cells.items())




