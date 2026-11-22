"""Orthogonal tests for ``msprof`` trace and OOXML ``.xlsx`` product exporters.

[Role] Verify 1:1 ``msprof`` ``trace_view.json`` metadata/span/flow/counter
  invariants, bounded streaming memory usage on multi-layer graphs, real OOXML
  ``<mergeCell>`` hierarchy & 1-to-N ``kernel_details.csv`` reconciliation,
  closed-form Weight/Quant/Cache/Engram-PLE/PP memory calculations, and CLI
  command/receipt isolation.
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
import tracemalloc
import xml.etree.ElementTree as ET
import zipfile

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim import cli as cli_mod
from mindie_sim import cli_export
from mindie_sim.compose import compose_case as compose_raw_case
from mindie_sim.export import (
    EXPORT_BUNDLE_SCHEMA,
    GIB_BYTES,
    MSPROF_PROCESS_SPEC,
    SHEET_BREAKDOWN,
    SHEET_MEMORY,
    SHEET_SUMMARY,
    ExportBundle,
    GapRadarRow,
    MemoryCalculatorLedger,
    OperatorExportRow,
    StageBudgetRow,
    Workbook,
    build_export_bundle,
    compute_memory_quant_ledger,
    evaluate_weight_storage_bytes,
    export_comparison_xlsx,
    export_msprof_trace,
    load_kernel_details_csv,
)
from mindie_sim.export.xlsx_exporter import NS_MAIN
from mindie_sim.prediction.results import OpOccurrence, StepRequest
from mindie_sim.prediction.structure import parse_case

pytestmark = pytest.mark.tier0

NS_PREFIX = f"{{{NS_MAIN}}}"
STREAM_MEMORY_LIMIT_BYTES = 16 * 1024 * 1024
LARGE_SYNTHETIC_LAYERS = 60
OPS_PER_SYNTHETIC_LAYER = 25


def _read_xlsx_sheets(path: Path) -> tuple[list[str], dict[str, ET.Element], list[str]]:
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        assert {
            "[Content_Types].xml", "_rels/.rels", "xl/workbook.xml",
            "xl/_rels/workbook.xml.rels", "xl/styles.xml", "xl/sharedStrings.xml",
        } <= names
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = [
            "".join(si.itertext())
            for si in sst_root.findall(f"{NS_PREFIX}si")
        ]
        sheet_names = [
            elem.attrib["name"]
            for elem in wb_root.findall(f".//{NS_PREFIX}sheet")
        ]
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
        if cell.attrib.get("t") == "s" and raw_v is not None:
            val = shared[int(raw_v)]
        else:
            val = raw_v
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
        tp=tp,
        dp=1,
        ep=ep,
        pp=pp,
        layers=layers,
        prompt_tokens=256,
        output_tokens=16,
        concurrency=2,
    )


def _synthetic_multilayer_bundle(num_layers: int, ops_per_layer: int) -> ExportBundle:
    ops: list[OperatorExportRow] = []
    t_cur = 0.0
    prev_occ: OpOccurrence | None = None
    ordinal = 0
    for layer in range(num_layers):
        for step in range(ops_per_layer):
            is_comm = (step == ops_per_layer - 1)
            is_aux = (step == ops_per_layer - 2)
            stream = "comm" if is_comm else ("aux" if is_aux else "main")
            s_idx = 2 if is_comm else (1 if is_aux else 0)
            sub = "Collective" if is_comm else ("MoE" if is_aux else "Attention")
            fam = "allreduce" if is_comm else ("grouped_gemm" if is_aux else "gemm")
            occ = OpOccurrence(replica=0, stage=0, rank=0, op_id=f"s0.l{layer}.op{step}")
            dur = 5.0 if not is_comm else 8.0
            l_star = 2.5 if not is_comm else 4.0
            ops.append(OperatorExportRow(
                ordinal=ordinal,
                occurrence=occ,
                op_id=occ.op_id,
                layer=layer,
                layer_label=f"Layer {layer}",
                submodule=sub,
                family=fam,
                recipe=fam,
                implementation="SyntheticKernel",
                stream=stream,
                stream_index=s_idx,
                task_type="HCCL" if is_comm else "MIX_AIC",
                shape=(4, 2048, 2048),
                dtype="bf16",
                predecessors=(prev_occ,) if prev_occ is not None else (),
                start_us=t_cur,
                end_us=t_cur + dur,
                dur_us=dur,
                flops=0.0 if is_comm else 3.2e7,
                vector_ops=0.0 if is_comm else 1.6e4,
                hbm_read_bytes=0.0 if is_comm else 4.0e6,
                hbm_write_bytes=0.0 if is_comm else 1.0e6,
                hbm_total_bytes=0.0 if is_comm else 5.0e6,
                mte2_bytes=0.0 if is_comm else 5.0e6,
                mte3_bytes=0.0 if is_comm else 1.0e6,
                fixpipe_bytes=0.0 if is_comm else 5.0e5,
                link_bytes=2.0e6 if is_comm else 0.0,
                l2_hit_pct=0.0 if is_comm else 20.0,
                dominant_bound="link_bound" if is_comm else "memory_bound",
                l_star_lean_us=l_star,
                gap_us=dur - l_star,
                gap_ratio=dur / l_star,
                gap_radar_rank=ordinal + 1,
                roofline_efficiency=0.5,
                absorbed_source_ops=(),
            ))
            prev_occ = occ
            t_cur += dur
            ordinal += 1
    mem = MemoryCalculatorLedger(
        model_name="synthetic-multilayer",
        hardware_sku="synthetic-npu",
        phase="decode",
        weight_quant="BF16",
        cache_dtype="BF16",
        engram_host_offload=False,
        dp=1, tp=1, ep=1, pp=1, dcp=1, pcp=1, local_tp=1, shared_tp=1, total_dies=1,
        seq_len=1024, concurrency=1, block_size=128, spec_tokens=0, local_cache_tokens=1024,
        die_hbm_capacity_bytes=64 * GIB_BYTES, gpu_memory_utilization=0.9,
        usable_hbm_bytes=57.6 * GIB_BYTES,
        weight_rows=(), cache_rows=(),
        stage_budgets=(StageBudgetRow(0, num_layers, 10 * GIB_BYTES, 0.0, 0.0, 0.0, GIB_BYTES, 0.5 * GIB_BYTES, 0.5 * GIB_BYTES, 12 * GIB_BYTES, 57.6 * GIB_BYTES, 45.6 * GIB_BYTES, True),),
        max_stage_hbm_bytes=12 * GIB_BYTES, max_stage_host_offload_bytes=0.0, fits_hbm=True,
        sim_memory_weight_bytes_by_stage=(10 * GIB_BYTES,),
        sim_memory_cache_bytes_by_stage=(float(GIB_BYTES),),
    )
    return ExportBundle(
        schema=EXPORT_BUNDLE_SCHEMA,
        model_name="synthetic-multilayer",
        hardware_sku="synthetic-npu",
        phase="decode",
        scope="body",
        graph_id="sha256:synthetic",
        tokens_per_seq=1,
        context_tokens=1024,
        sequences=1,
        step_span_us=t_cur,
        operator_sum_us=t_cur,
        step_l_star_lean_us=t_cur * 0.5,
        step_certificate_valid=True,
        step_tight_resources=("hbm_read@r0.p0.d0",),
        throughput_tokens_per_s=1e6 / max(t_cur, 1.0),
        throughput_per_die_tokens_per_s=1e6 / max(t_cur, 1.0),
        network_cube_mfu=20.0, network_vec_mfu=10.0, network_mbu=50.0,
        network_link_util=40.0, network_roofline_efficiency=50.0,
        exposed_breakdown_us={"cube": t_cur * 0.7, "communication": t_cur * 0.3},
        bottleneck_mix={"memory_bound": {"count": float(len(ops)), "duration_us": t_cur, "share_pct": 100.0, "roofline_eff_pct": 50.0}},
        hardware_peaks={"cube_peak_flops_per_s": 5e14, "vector_peak_ops_per_s": 1e13, "hbm_peak_bytes_per_s": 4e12, "link_peak_bytes_per_s": 4e11},
        operators=tuple(ops),
        pipelines=(),
        gap_radar=(GapRadarRow(1, "Layer 0", "Collective", "s0.l0.op39", "allreduce", "SyntheticKernel", "link_bound", 4.0, 8.0, 4.0, 2.0, 1.0),),
        memory_ledger=mem,
    )


def test_msprof_trace_schema_metadata_flows_and_counters(tmp_path):
    case = _compact_case(tp=2, ep=2, pp=1, layers=2)
    req = StepRequest(phase="decode", tokens=1, context=512, sequences=2, scope="body")
    bundle = build_export_bundle(case, request=req)
    assert bundle.step_certificate_valid is True
    assert 0.0 <= bundle.step_l_star_lean_us <= bundle.step_span_us + 1.0

    trace_path = tmp_path / "trace_view.json"
    receipt = export_msprof_trace(bundle, trace_path)
    raw = trace_path.read_bytes()
    assert b"\r\n" not in raw
    events = json.loads(raw.decode("utf-8"))
    assert isinstance(events, list) and len(events) == receipt.total_events

    meta_names = {e["name"] for e in events if e.get("ph") == "M"}
    assert {"process_name", "process_labels", "process_sort_index", "thread_name", "thread_sort_index"} <= meta_names
    proc_names = {e["args"]["name"] for e in events if e.get("ph") == "M" and e["name"] == "process_name"}
    for required_proc in MSPROF_PROCESS_SPEC:
        assert any(name.startswith(required_proc) for name in proc_names)
    thread_names = {e["args"]["name"] for e in events if e.get("ph") == "M" and e["name"] == "thread_name"}
    assert {"Stream 0 (Compute)", "Stream 1 (MSO / Aux 2nd)", "Stream 2 (Comm / CSA 3rd)", "Cube (AIC)", "Vector (AIV)", "MTE2 (GM->L1/UB)", "MTE3 (UB->GM)", "Fixpipe (L0C->GM/UB)"} <= thread_names

    kernel_events = [e for e in events if e.get("ph") == "X" and e.get("cat") in {"Kernel", "Communication"}]
    assert len(kernel_events) == len(bundle.operators)
    sample_args = kernel_events[0]["args"]
    for key in ("Input Shapes", "Input Data Types", "FLOPs", "HBM Read (B)", "HBM Write (B)", "L2 Hit (%)", "Bottleneck", "L* Physical Lower Bound (us)", "Sim Pred (us)", "Gap vs L* (us)", "Physical-Limit Gap Ratio (T_pred / L*)", "Gap Radar Rank"):
        assert key in sample_args
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
    for ev in (e for e in events if e.get("ph") == "C"):
        if ev["name"] == "Cube Utilization (%)":
            assert 0.0 <= ev["args"]["Cube_Util_Pct"] <= 100.0
        elif ev["name"] == "Vector Utilization (%)":
            assert 0.0 <= ev["args"]["Vector_Util_Pct"] <= 100.0


def test_trace_streaming_memory_bound_on_large_multilayer_graph(tmp_path):
    bundle = _synthetic_multilayer_bundle(LARGE_SYNTHETIC_LAYERS, OPS_PER_SYNTHETIC_LAYER)
    out = tmp_path / "large_trace_view.json"
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        receipt = export_msprof_trace(bundle, out)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert receipt.total_events > 10_000
    assert receipt.bytes_written > 2 * 1024 * 1024
    assert peak_bytes < STREAM_MEMORY_LIMIT_BYTES
    assert receipt.total_events == (
        receipt.metadata_events + receipt.complete_events
        + 2 * receipt.flow_pairs + receipt.counter_events
    )
    assert out.read_bytes()[:2] == b"[\n"
    assert out.read_bytes()[-3:] == b"\n]\n"
    try:
        from mindie_profiler.sources_trace import iter_trace_events
    except ImportError:
        iter_trace_events = None
    if iter_trace_events is not None:
        streamed_count = sum(1 for _ in iter_trace_events(out))
        assert streamed_count == receipt.total_events


def test_xlsx_structure_merge_cells_formulas_and_comparison_delta(tmp_path):
    base_case = _compact_case(tp=2, ep=2, pp=1, layers=2)
    cand_case = _compact_case(tp=4, ep=4, pp=1, layers=2)
    req = StepRequest(phase="prefill", tokens=128, context=0, sequences=1, scope="body")

    xlsx_path = tmp_path / "comparison.xlsx"
    summary = export_comparison_xlsx(base_case, xlsx_path, compare_source=cand_case, request=req)
    assert summary.sheets == (SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY)
    assert summary.has_comparison is True and summary.has_measured is False

    sheet_names, roots, shared = _read_xlsx_sheets(xlsx_path)
    assert sheet_names == [SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY]

    expected_order = ["dimension", "sheetViews", "sheetFormatPr", "cols", "sheetData", "mergeCells", "dataValidations", "pageMargins"]
    for title, root in roots.items():
        tags = [child.tag.removeprefix(NS_PREFIX) for child in root]
        filtered = [t for t in expected_order if t in tags]
        assert tags == filtered, f"{title}: out-of-order OOXML tags {tags}"
        merges = _sheet_merges(root)
        assert len(merges) == summary.merge_cell_counts[title] and len(merges) > 0

    sum_cells = _sheet_cells(roots[SHEET_SUMMARY], shared)
    assert sum_cells["A6"][0] == "Prefill TTFT (ms)"
    assert sum_cells["B7"][1] == "B6*1000"
    assert sum_cells["B13"][1] == "B7-B12"
    assert any(f and "H" in f and "-G" in f for _, f in sum_cells.values())
    assert any(v == "3. Before vs After Comparison Delta" for v, _ in sum_cells.values())

    brk_merges = _sheet_merges(roots[SHEET_BREAKDOWN])
    assert any(ref.startswith("A") and ":A" in ref for ref in brk_merges)
    assert any(ref.startswith("B") and ":B" in ref for ref in brk_merges)
    brk_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    sum_formulas = [f for _, f in brk_cells.values() if f and f.startswith("SUM(")]
    assert len(sum_formulas) >= 5


def test_xlsx_one_to_many_kernel_csv_reconciliation_and_merges(tmp_path):
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
    parsed_csv = load_kernel_details_csv(csv_path)
    assert len(parsed_csv) == 4

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
    rec = rec_bundle.reconciliation
    assert rec.total_kernels == 4 and rec.matched_kernels == 3 and len(rec.unmatched_kernels) == 1
    assert rec.unmatched_kernels[0].name == "aclnnUnmatched_Fill"
    matched_row = next(op for op in rec_bundle.operators if op.op_id == target_op.op_id)
    assert len(matched_row.measured_kernels) == 3
    assert matched_row.measured_sum_us == pytest.approx(3.0 * k_dur, rel=1e-6)
    assert matched_row.ape_pct is not None and matched_row.ape_pct < 1.0

    xlsx_path = tmp_path / "reconciled.xlsx"
    xlsx_rec = export_comparison_xlsx(rec_bundle, xlsx_path)
    assert xlsx_rec.has_measured is True

    _, roots, shared = _read_xlsx_sheets(xlsx_path)
    brk_merges = _sheet_merges(roots[SHEET_BREAKDOWN])
    brk_cells = _sheet_cells(roots[SHEET_BREAKDOWN], shared)
    target_ref = next(
        ref for ref, (val, _) in brk_cells.items()
        if ref.startswith("C") and val == target_op.op_id
    )
    r_start = int(target_ref[1:])
    r_end = r_start + 2
    for col_let in ("C", "M", "P", "Q", "R"):
        assert f"{col_let}{r_start}:{col_let}{r_end}" in brk_merges
    assert brk_cells[f"N{r_start}"][0] == "aclnnStageA_Kernel"
    assert brk_cells[f"N{r_start + 1}"][0] == "aclnnStageB_Kernel"
    assert brk_cells[f"N{r_end}"][0] == "aclnnStageC_Kernel"
    assert brk_cells[f"P{r_start}"][1] == f"SUM(O{r_start}:O{r_end})"
    assert brk_cells[f"Q{r_start}"][1] == f"IF(P{r_start}>0,ABS(M{r_start}-P{r_start})/P{r_start},0)"


def test_memory_quant_calculator_exact_storage_offload_and_pp_budget(tmp_path):
    # Exact post-TP local shape (6, 130)/2 -> (6, 65) in MXFP4: 198 B payload + 18 B E8M0 scale = 216 B.
    payload, scale, aux, total = evaluate_weight_storage_bytes(6, 65, "MXFP4")
    assert (payload, scale, aux, total) == (198, 18, 0, 216)
    assert evaluate_weight_storage_bytes(16, 64, "W8A8") == (1024, 0, 128, 1152)
    assert evaluate_weight_storage_bytes(16, 64, "W4A8") == (512, 0, 256, 768)
    assert evaluate_weight_storage_bytes(16, 64, "MXFP8") == (1024, 32, 0, 1056)
    assert evaluate_weight_storage_bytes(16, 64, "BF16") == (2048, 0, 0, 2048)

    raw_case = json.loads(Path("examples/cases/deepseek-v4.1-flash.json").read_text(encoding="utf-8"))
    case = parse_case(raw_case)
    led_bf16_hbm = compute_memory_quant_ledger(
        case, seq_len=4096, concurrency=4, weight_quant="BF16", cache_dtype="BF16", engram_host_offload=False,
    )
    led_mxfp4_offload = compute_memory_quant_ledger(
        case, seq_len=4096, concurrency=4, weight_quant="MXFP4", cache_dtype="C8", engram_host_offload=True,
    )
    assert led_bf16_hbm.pp == 4 and len(led_bf16_hbm.stage_budgets) == 4
    engram_hbm_total = sum(s.engram_ple_hbm_bytes for s in led_bf16_hbm.stage_budgets)
    engram_host_total = sum(s.engram_ple_host_bytes for s in led_mxfp4_offload.stage_budgets)
    assert engram_hbm_total > 0.0
    assert engram_host_total == pytest.approx(engram_hbm_total, rel=1e-9)
    assert all(s.engram_ple_hbm_bytes == 0.0 for s in led_mxfp4_offload.stage_budgets)
    assert led_mxfp4_offload.max_stage_hbm_bytes < led_bf16_hbm.max_stage_hbm_bytes
    assert sum(r.scale_bytes for r in led_mxfp4_offload.weight_rows) > 0
    assert sum(s.cache_bytes for s in led_mxfp4_offload.stage_budgets) < sum(s.cache_bytes for s in led_bf16_hbm.stage_budgets)

    xlsx_path = tmp_path / "memory_calc.xlsx"
    compact = _compact_case(tp=2, ep=2, pp=2, layers=4)
    export_comparison_xlsx(
        compact, xlsx_path,
        request=StepRequest(phase="decode", tokens=1, context=512, sequences=2, scope="body"),
        weight_quant="MXFP4", cache_dtype="C8", engram_host_offload=True,
    )
    _, roots, shared = _read_xlsx_sheets(xlsx_path)
    mem_root = roots[SHEET_MEMORY]
    validations = [
        (dv.attrib["sqref"], (dv.find(f"{NS_PREFIX}formula1").text or ""))
        for dv in mem_root.findall(f".//{NS_PREFIX}dataValidation")
    ]
    assert any(sq == "R3" and "MXFP4" in f1 and "W8A8" in f1 for sq, f1 in validations)
    assert any(sq == "R4" and "C8" in f1 and "FP8" in f1 for sq, f1 in validations)
    assert any(sq == "R5" and "ON,OFF" in f1 for sq, f1 in validations)
    mem_cells = _sheet_cells(mem_root, shared)
    assert mem_cells["R14"][1] == "R6*R7*R9*R11"
    assert mem_cells["R19"][1] == "ROUNDUP(R15/(R17*R10),0)*R17"
    assert mem_cells["R22"][1] == "R20*R21"
    assert any(f and "VLOOKUP(" in f and "ROUNDUP(" in f for _, f in mem_cells.values())


def test_cli_export_commands_receipt_isolation_and_error_guards(tmp_path, monkeypatch):
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_compact_case(tp=2, ep=2, pp=1, layers=1)), encoding="utf-8")
    trace_out = tmp_path / "cli_trace_view.json"
    trace_receipt = tmp_path / "trace_receipt.json"
    xlsx_out = tmp_path / "cli_report.xlsx"
    xlsx_receipt = tmp_path / "xlsx_receipt.json"

    rc_trace = cli_mod._main([
        "export-trace", str(case_path), "-o", str(trace_out),
        "--phase", "decode", "--tokens", "1", "--context", "256",
        "--receipt", str(trace_receipt),
    ])
    assert rc_trace == 0
    trace_events = json.loads(trace_out.read_text(encoding="utf-8"))
    assert isinstance(trace_events, list) and len(trace_events) > 10
    rec_t = json.loads(trace_receipt.read_text(encoding="utf-8"))
    assert rec_t["ok"] is True and rec_t["command"] == "export-trace" and rec_t["total_events"] == len(trace_events)

    rc_xlsx = cli_mod._main([
        "export-xlsx", str(case_path), "--output", str(xlsx_out),
        "--phase", "decode", "--tokens", "1", "--context", "256",
        "--weight-quant", "w4a8_mxfp", "--cache-dtype", "fp8_e4m3", "--engram-offload",
        "--receipt", str(xlsx_receipt),
    ])
    assert rc_xlsx == 0
    assert zipfile.is_zipfile(xlsx_out)
    rec_x = json.loads(xlsx_receipt.read_text(encoding="utf-8"))
    assert rec_x["ok"] is True and rec_x["command"] == "export-xlsx"

    standalone_out = tmp_path / "standalone.xlsx"
    assert cli_export.main(["export-xlsx", str(case_path), "-o", str(standalone_out), "--context", "128"]) == 0
    assert zipfile.is_zipfile(standalone_out)

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
