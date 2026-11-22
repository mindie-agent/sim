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

from dataclasses import replace
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
    BANNED_IMPLEMENTATION_NAMES,
    GIB_BYTES,
    MSPROF_PROCESS_SPEC,
    PID_FRAMEWORK,
    SHEET_BREAKDOWN,
    SHEET_MEASURED,
    SHEET_MEMORY,
    SHEET_NEW_KERNELS,
    SHEET_SUMMARY,
    TID_FW_ENGINE,
    TID_FW_HOST_LAUNCH,
    TID_FW_LAYER_MSTX,
    TID_FW_SUBMODULE_MSTX,
    Workbook,
    build_export_bundle,
    compose_export_case,
    compute_memory_quant_ledger,
    evaluate_weight_storage_bytes,
    export_comparison_xlsx,
    export_msprof_trace,
    load_kernel_details_csv,
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
        dp=1, tp=2, ep=2, dcp=1, prompt_tokens=2048, output_tokens=64,
        concurrency=2, mtp_proposal_tokens=5, feature_track="merged",
    )
    raw_aft, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=2, dcp=1, prompt_tokens=2048, output_tokens=64,
        concurrency=2, mtp_proposal_tokens=5, feature_track="expected_ideal",
    )
    b_bef = build_export_bundle(raw_bef, request=req)
    b_aft = build_export_bundle(raw_aft, request=req)

    # Inject a realistic N->1 MLA Prolog fusion in After to assert vertical K..U merge_cells.
    mla_ops = [op for op in b_aft.operators if op.archetype_id == "target_arch_2" and "MLA Prolog" in op.submodule]
    assert len(mla_ops) >= 3
    fused_mla = replace(
        mla_ops[0],
        op_id="s0.l2.mla_prolog_v3",
        canonical_op_key="mla_prolog_v3",
        logical_role="Fused MLA Prolog (Q/KV-A + RMSNorm + Q-B + W_UK + KV Cache)",
        ascend_kernel="aclnnMlaPrologV3",
        implementation="aclnnMlaPrologV3",
        absorbed_source_ops=tuple(op.op_id for op in mla_ops),
        dur_us=sum(op.dur_us for op in mla_ops) * 0.6,
        exposed_dur_us=sum(op.dur_us for op in mla_ops) * 0.6,
    )
    mla_ids = {op.op_id for op in mla_ops}
    kept = []
    inserted = False
    for op in b_aft.operators:
        if op.op_id in mla_ids:
            if not inserted:
                kept.append(fused_mla)
                inserted = True
            continue
        kept.append(op)
    b_aft_fused = replace(b_aft, operators=tuple(kept))

    designs = {
        "K3 SituAndMul": {
            "design": "Single-pass AIV vector tile (UB 248KiB double-buffered) fusing SiTU + Mul",
            "acceptance": "BF16 abs_err <= 1e-3, L* efficiency >= 70%",
        },
    }
    xlsx_path = tmp_path / "comparison.xlsx"
    summary = export_comparison_xlsx(
        b_bef, xlsx_path, compare_source=b_aft_fused, agent_kernel_designs=designs,
    )
    assert summary.sheets == (SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY, SHEET_NEW_KERNELS)
    assert SHEET_MEASURED not in summary.sheets
    assert summary.has_comparison is True and summary.has_measured is False
    assert summary.new_kernel_backlog_count >= 1

    sheet_names, roots, shared = _read_xlsx_sheets(xlsx_path)
    assert sheet_names == [SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY, SHEET_NEW_KERNELS]
    for banned in BANNED_IMPLEMENTATION_NAMES:
        assert not any(banned in s for s in shared), f"banned name leaked: {banned}"
    assert "Other" not in shared

    expected_order = ["dimension", "sheetViews", "sheetFormatPr", "cols", "sheetData", "mergeCells", "dataValidations", "pageMargins"]
    for title, root in roots.items():
        tags = [child.tag.removeprefix(NS_PREFIX) for child in root]
        assert tags == [t for t in expected_order if t in tags]

    brk_merges = _sheet_merges(roots[SHEET_BREAKDOWN])
    brk_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    assert any(ref.startswith("A") and ":A" in ref for ref in brk_merges)
    assert any(ref.startswith("C") and ":C" in ref for ref in brk_merges)
    assert any(ref.startswith("K") and ":K" in ref for ref in brk_merges), "Expected N->1 vertical merge in After Kernel column K"
    assert any(ref.startswith("O") and ":O" in ref for ref in brk_merges), "Expected N->1 vertical merge in After Model ms column O"
    assert any(f and "ISNUMBER(SEARCH(\"Overlapped\"" in f for _, f in brk_cells.values())
    assert any(f and f.startswith("H") and "/1000" in f for _, f in brk_cells.values())
    assert any(f and f.startswith("SUM(I") and "-O" in f for _, f in brk_cells.values())
    assert any(v == "Before Source Ref" for v, _ in brk_cells.values())
    assert any(v == "Before Bound" for v, _ in brk_cells.values())
    assert any(v == "After Bound" for v, _ in brk_cells.values())
    assert not any(v == "Measured Kernel (msprof)" for v, _ in brk_cells.values())

    bk_cells = _sheet_cells(roots[SHEET_NEW_KERNELS], shared)
    assert any(v and "Single-pass AIV vector tile" in v for v, _ in bk_cells.values())
    assert any(v and "BF16 abs_err <= 1e-3" in v for v, _ in bk_cells.values())
    assert any(f and f.startswith("SUM(G") for _, f in bk_cells.values())


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


def test_submodule_and_kernel_hygiene_across_hybrid_architectures():
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
    ng_ops = [o for o in b_raw.operators if o.canonical_op_key == "linear_output_norm_gate"]
    assert ln_ops and all(o.ascend_kernel == "aclnnLayerNorm_LayerNormV3" for o in ln_ops)
    assert ng_ops and all("Cast->Pows->ReduceMean" in o.ascend_kernel for o in ng_ops)


def test_memory_quant_calculator_exclusions_offload_and_pp_budget(tmp_path):
    assert evaluate_weight_storage_bytes(6, 65, "MXFP4") == (198, 18, 0, 216)
    assert evaluate_weight_storage_bytes(16, 64, "W8A8") == (1024, 0, 128, 1152)
    assert evaluate_weight_storage_bytes(16, 64, "W4A8") == (512, 0, 256, 768)
    assert evaluate_weight_storage_bytes(16, 64, "MXFP8") == (1024, 32, 0, 1056)
    assert evaluate_weight_storage_bytes(16, 64, "BF16") == (2048, 0, 0, 2048)

    raw_k, _ = compose_export_case("kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=8, ep=32, dp=4, dcp=8)
    led_routed = compute_memory_quant_ledger(parse_case(raw_k), seq_len=4096, concurrency=4)
    assert led_routed.weight_quant == "MXFP4"
    assert led_routed.weight_quant_display == "MXFP4 (Routed-Only, Attn/Shared BF16)"
    by_name = {r.name: r for r in led_routed.weight_rows}
    assert by_name["q_a"].quant_format == "BF16" and by_name["q_a"].is_quant_excluded is True
    assert by_name["o_proj"].quant_format == "BF16" and by_name["dense_down"].quant_format == "BF16"
    assert any(r.category == "routed_expert" and r.quant_format == "MXFP4" for r in led_routed.weight_rows)
    assert all(r.quant_format == "BF16" for r in led_routed.weight_rows if r.category in {"shared_expert", "router", "latent_moe"})

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
    assert mem_cells["R14"][1] == "R6*R7*R9*R11"
    assert mem_cells["R19"][1] == "ROUNDUP(R15/(R17*R10),0)*R17"
    assert mem_cells["R22"][1] == "R20*R21"
    assert mem_cells["R23"][0] == "MXFP4 (Routed-Only, Attn/Shared BF16)"
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
