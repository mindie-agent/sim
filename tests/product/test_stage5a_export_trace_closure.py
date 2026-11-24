"""Product tests closing Stage 5A export, trace, Sheet 3 C8/ISA, and kernel-identity defects.

[Role] Verify all seven Stage 5A export/trace closures:
  (1) non-contiguous W8A8_MIX cohort slicing preserves every physical layer with
      0.000 ms reconciliation error in Sheet 2, Sheet 4, and agent_brief.json;
  (2) non-speculative exports default to ``scope="model"`` including ``lm_head``
      logits in Sheet 2 and trace while preserving explicit ``scope="body"`` opt-out;
  (3) ``trace_view.json`` sanitizes ``Feature Track`` via ``public_view`` (zero
      raw internal track enum leaks);
  (4) speculative traces emit every MTP draft block execution and every repeated
      ``RecoverSSMStateCommit`` span conserving ``|trace_root - step_span| < 10 us``;
  (5) accurate shipping vs synthesized kernel contracts for A3 ``aclnnMegaMoe``,
      Qwen Triton ``qwen4_exp_ascend_hc_gate_mix``, and GMM+SwiGLU fusions;
  (6) Sheet 3 A2/A3 ISA hygiene (zero ``PresetLinearFP8`` / ``FP8`` formulas) and
      exact ``<= 0.01%`` ``T4=C8`` cache parity against ``ResidentMemoryReceipt``;
  (7) Sheet 1 ``H14`` 4-stage summary parity with Sheet 4 ``D3:D9``, A5 TileLang
      SimtVF/SimdVF & UB-wall card visibility, and CLI ``report`` MTP/SLO forwarding.
[Invariants] Pure ``tier0`` in-memory & ``tmp_path`` OOXML/JSON inspection; zero
  raw SKU string literals, zero subprocesses, and zero host-state pollution.
[Timeouts & Locks] Bounded in-memory execution only; no locks or network I/O.
[Side-Effect Policy] Writes temporary artifacts exclusively under pytest ``tmp_path``.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.cli.parser import build_parser
from mindie_sim.cli.report import _report_column, resolve_auto_va_policy
from mindie_sim.export import (
    BACKEND_TRITON,
    BINDING_STATE_UNBOUND_FORMULA,
    SHEET_BREAKDOWN,
    SHEET_MEMORY,
    SHEET_PENDING_KERNELS,
    SHEET_SUMMARY,
    align_before_after_operators,
    build_agent_brief,
    build_export_bundle,
    compose_export_case,
    export_comparison_xlsx,
    export_msprof_trace,
)
from mindie_sim.export.action_plan import build_comparison_plan
from mindie_sim.export.comparison import enrich_comparison_plan_stage_receipts
from mindie_sim.export.models import QUANT_RULE_PRESET_LINEAR_W8A8, SUB_MOE_ROUTED
from mindie_sim.export.ooxml import NS_MAIN
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.scenario import CoDesignOpportunity, MegaFusionBound

pytestmark = pytest.mark.tier0

NS_PREFIX = f"{{{NS_MAIN}}}"
RAW_TRACK_ENUM_RE = re.compile(r"\b(expected_ideal|unmerged_pr|raw_eager|merged)\b")
STAGE_SUMMARY_RE = re.compile(
    r"Before\s+([0-9.]+)\s*->\s*\[1\]配置\s+([0-9.]+)"
    r"(?:\s*->\s*\[2\]框架\s+([0-9.]+))?"
    r"\s*->\s*\[3\]算子/After\s+([0-9.]+)"
)
IF_TWO_BRANCH_RE = re.compile(r'^=?IF\(\$T\$4="([^"]+)",([^,]+),([^)]+)\)$')

FAST_PROMPT_TOKENS = 512
FAST_OUTPUT_TOKENS = 64
COMPACT_SCOPE_LAYERS = 4
PROMPT_TOKENS = 4096
OUTPUT_TOKENS = 512
QWEN32_DP = 2
QWEN32_TP = 4
QWEN32_CONC = 4
QWEN32_W8A8_COHORT_LAYERS = 55
QWEN38_DP = 8
QWEN38_TP = 2
QWEN38_EP = 16
QWEN38_CONC = 4
MTP_THREE_TOKENS = 3
MTP_THREE_ACCEPT = 2.4
QWEN38_RECOVER_SSM_REPEATS = 36
DEEPSEEK_TP = 8
DEEPSEEK_EP = 8
DEEPSEEK_CONC = 8
MIN_MEGAMOE_FLOPS = 1.0e8
RECON_TOL_MS = 1e-6
STAGE_DISPLAY_TOL_MS = 0.002
TRACE_ROOT_PARITY_TOL_US = 10.0
C8_PARITY_REL_TOL = 1e-4
GIB_DIVISOR = float(1024 ** 3)
MIB_DIVISOR = float(1024 ** 2)
SIMT_GAIN_MS = 1.15
SIMD_GAIN_MS = 1.85
MEGA_BEFORE_MS = 3.2
MEGA_BOUND_MS = 1.1
MEGA_TILED_MS = 1.35
MEGA_GM_BYTES = 12 * 1024 * 1024
MEGA_UB_LIVE_BYTES = 232 * 1024
MEGA_UB_CAP_BYTES = 256 * 1024
LONG_PROMPT_TOKENS = 131072
LONG_OUTPUT_TOKENS = 1024
TPOT_SLO_TIGHT_MS = 10.0
TPOT_SLO_RELAXED_MS = 30.0
MTP_FIVE_TOKENS = 5
MTP_FIVE_ACCEPT = 3.5


def _read_workbook_sheets(path: Path) -> tuple[dict[str, dict[str, tuple[str | None, str | None]]], dict[str, ET.Element]]:
    with zipfile.ZipFile(path, "r") as zf:
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = [
            "".join(t.text or "" for t in si.iter(f"{NS_PREFIX}t"))
            for si in sst_root.findall(f"{NS_PREFIX}si")
        ]
        sheets: dict[str, dict[str, tuple[str | None, str | None]]] = {}
        roots: dict[str, ET.Element] = {}
        for idx, sh in enumerate(wb_root.findall(f".//{NS_PREFIX}sheet"), start=1):
            name = sh.attrib["name"]
            root = ET.fromstring(zf.read(f"xl/worksheets/sheet{idx}.xml"))
            roots[name] = root
            cells: dict[str, tuple[str | None, str | None]] = {}
            for cell in root.iter(f"{NS_PREFIX}c"):
                ref = cell.attrib["r"]
                ctype = cell.attrib.get("t")
                f_el = cell.find(f"{NS_PREFIX}f")
                v_el = cell.find(f"{NS_PREFIX}v")
                val = None if v_el is None else (shared[int(v_el.text or "0")] if ctype == "s" else v_el.text)
                cells[ref] = (val, f_el.text if f_el is not None else None)
            sheets[name] = cells
        return sheets, roots


def _eval_sheet3_cache_total_for_dtype(
    s3_cells: dict[str, tuple[str | None, str | None]],
    target_dtype: str,
) -> float:
    """Evaluate Sheet 3 cache batch total bytes when ``$T$4`` is switched to ``target_dtype``."""
    conc = float(s3_cells["T18"][0] or "0")
    local_tok = float(s3_cells["T21"][0] or "0")
    hdr_row = next(
        int(ref[1:]) for ref, (val, _) in s3_cells.items()
        if ref.startswith("A") and val == "Component"
    )
    tot_row = next(
        int(ref[1:]) for ref, (val, _) in s3_cells.items()
        if ref.startswith("A") and val == "Cache Total (Max Stage)"
    )
    total_bytes = 0.0
    for r in range(hdr_row + 1, tot_row):
        layers = float(s3_cells[f"C{r}"][0] or "0")
        d_val, d_f = s3_cells[f"D{r}"]
        if d_f is not None:
            match = IF_TWO_BRANCH_RE.match(d_f.strip())
            assert match is not None, f"Unexpected Sheet 3 cache rate formula at D{r}: {d_f}"
            cond_dt, true_expr, false_expr = match.groups()
            rate = float(true_expr) if target_dtype == cond_dt else float(false_expr)
            req_bytes = layers * rate * local_tok
        else:
            e_val, e_f = s3_cells[f"E{r}"]
            if e_f is not None and f"*$T$21/" in e_f:
                rate = float(d_val or "0")
                req_bytes = layers * rate * local_tok
            elif e_f is not None and IF_TWO_BRANCH_RE.match(e_f.strip()):
                m_e = IF_TWO_BRANCH_RE.match(e_f.strip())
                assert m_e is not None
                cond_dt, true_expr, false_expr = m_e.groups()
                req_mib = float(true_expr) if target_dtype == cond_dt else float(false_expr)
                req_bytes = req_mib * MIB_DIVISOR
            else:
                req_bytes = float(e_val or "0") * MIB_DIVISOR
        total_bytes += req_bytes * conc
    return total_bytes


def _receipt_total_cache_bytes(receipt: object) -> float:
    die0 = receipt.die_receipt(0, 0, 0)  # type: ignore[attr-defined]
    return float(
        die0.kv_cache_allocated_bytes
        + die0.index_cache_allocated_bytes
        + die0.recurrent_state_bytes
        + die0.draft_cache_bytes
    )


def test_non_contiguous_w8a8_mix_cohort_preserves_all_layers_with_zero_reconciliation_error(
    tmp_path: Path,
) -> None:
    pol = resolve_auto_va_policy("qwen3-32b", REPRESENTATIVE_SKUS["a3"], context_tokens=FAST_PROMPT_TOKENS)
    raw_bef, _ = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a3"],
        dp=QWEN32_DP, tp=QWEN32_TP, ep=1,
        prompt_tokens=FAST_PROMPT_TOKENS, output_tokens=FAST_OUTPUT_TOKENS,
        concurrency=QWEN32_CONC, feature_track="merged", quant=pol["compose_quant"],
    )
    raw_aft, _ = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a3"],
        dp=QWEN32_DP, tp=QWEN32_TP, ep=1,
        prompt_tokens=FAST_PROMPT_TOKENS, output_tokens=FAST_OUTPUT_TOKENS,
        concurrency=QWEN32_CONC, feature_track="expected_ideal", quant=pol["compose_quant"],
    )
    # Exercise scope="body" explicitly so L63 emits standalone ffn.residual inside the 55-layer W8A8 cohort.
    body_req = StepRequest(
        phase="decode", tokens=1, context=FAST_PROMPT_TOKENS, sequences=QWEN32_CONC, scope="body",
    )
    b_bef = build_export_bundle(raw_bef, request=body_req, apply_target_envelope=False)
    b_aft = build_export_bundle(raw_aft, request=body_req, apply_target_envelope=True)

    arch0 = next(a for a in b_bef.archetypes if a.archetype_id == "target_arch_0")
    assert len(arch0.layer_indices) == QWEN32_W8A8_COHORT_LAYERS

    aligned = align_before_after_operators(b_bef, b_aft)
    plan = enrich_comparison_plan_stage_receipts(
        build_comparison_plan(b_bef, b_aft, aligned), b_bef, b_aft, aligned,
    )
    assert abs(plan.waterfall.reconciliation_error_ms) < RECON_TOL_MS
    assert abs(plan.waterfall.before_endpoint_error_ms) < RECON_TOL_MS
    assert abs(plan.waterfall.after_endpoint_error_ms) < RECON_TOL_MS
    arch0_rows = [r for r in aligned if r.archetype_id == "target_arch_0"]
    assert any(r.effective_before_repeat == QWEN32_W8A8_COHORT_LAYERS for r in arch0_rows)

    out_xlsx = tmp_path / "qwen32_w8a8_mix.xlsx"
    brief_path = tmp_path / "qwen32_w8a8_mix_brief.json"
    export_comparison_xlsx(
        b_bef, out_xlsx, compare_source=b_aft, agent_brief_path=brief_path,
    )
    sheets, _ = _read_workbook_sheets(out_xlsx)
    s2 = sheets[SHEET_BREAKDOWN]
    s4 = sheets[SHEET_PENDING_KERNELS]
    assert int(float(s2["B3"][0] or "0")) == QWEN32_W8A8_COHORT_LAYERS
    assert abs(float(s4["D10"][0] or "0")) < RECON_TOL_MS
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    assert abs(float(brief["kpis_before_vs_after"]["waterfall"]["reconciliation_error_ms"])) < RECON_TOL_MS


def test_non_speculative_export_defaults_to_model_scope_including_lm_head(tmp_path: Path) -> None:
    raw, _ = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a3"],
        dp=1, tp=QWEN32_TP, ep=1,
        prompt_tokens=FAST_PROMPT_TOKENS, output_tokens=FAST_OUTPUT_TOKENS,
        concurrency=QWEN32_CONC, feature_track="merged", quant="w8a8_dynamic",
    )
    raw = {
        **raw,
        "model": {
            **raw["model"],
            "layers": COMPACT_SCOPE_LAYERS,
            "layer_plans": raw["model"]["layer_plans"][:COMPACT_SCOPE_LAYERS],
        },
    }
    # Passing request=None must default to scope="model" even when mtp_proposal_tokens == 0.
    bun_default = build_export_bundle(raw, apply_target_envelope=False)
    assert any("logits" in op.op_id.lower() for op in bun_default.operators)

    out_xlsx = tmp_path / "non_spec_default_model.xlsx"
    out_trace = tmp_path / "non_spec_default_model_trace.json"
    export_comparison_xlsx(bun_default, out_xlsx)
    export_msprof_trace(bun_default, out_trace)

    sheets, _ = _read_workbook_sheets(out_xlsx)
    s2_text = " ".join(str(v or "") for v, _ in sheets[SHEET_BREAKDOWN].values())
    assert "LM Head" in s2_text or "logits" in s2_text.lower()
    trace_events = json.loads(out_trace.read_text(encoding="utf-8"))
    assert any("logits" in str(ev.get("name", "")).lower() for ev in trace_events)

    # Explicit scope="body" opt-out remains respected.
    bun_body = build_export_bundle(
        raw,
        request=StepRequest(phase="decode", tokens=1, context=FAST_PROMPT_TOKENS, sequences=QWEN32_CONC, scope="body"),
        apply_target_envelope=False,
    )
    assert not any("logits" in op.op_id.lower() for op in bun_body.operators)


def test_trace_view_sanitizes_feature_track_and_conserves_mtp_and_recoverssm_spans(
    tmp_path: Path,
) -> None:
    raw_bef, req_q = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=QWEN38_DP, tp=QWEN38_TP, ep=QWEN38_EP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        concurrency=QWEN38_CONC,
        mtp_proposal_tokens=MTP_THREE_TOKENS, mtp_acceptance_length=MTP_THREE_ACCEPT,
        feature_track="merged", quant="mxfp8",
    )
    raw_aft, _ = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=QWEN38_DP, tp=QWEN38_TP, ep=QWEN38_EP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        concurrency=QWEN38_CONC,
        mtp_proposal_tokens=MTP_THREE_TOKENS, mtp_acceptance_length=MTP_THREE_ACCEPT,
        feature_track="expected_ideal", quant="mxfp8",
    )
    b_bef = build_export_bundle(raw_bef, request=req_q, apply_target_envelope=False)
    b_aft = build_export_bundle(raw_aft, request=req_q, apply_target_envelope=True)

    # 1. Shipping Qwen hc_gate_mix resolves to Triton-Ascend qwen4_exp_ascend_hc_gate_mix.
    gate_ops = [op for op in b_bef.operators if op.op_id.endswith("hc_gate_mix")]
    assert gate_ops
    for op in gate_ops:
        assert op.ascend_kernel == "qwen4_exp_ascend_hc_gate_mix"
        assert op.kernel_backend == BACKEND_TRITON
        assert op.needs_new_kernel is False

    # 2. BF16 draft GMM1 + SwiGLU fusion never emits K1_shipping aclnnSwiGluQuant.
    draft_gmm = [op for op in b_aft.operators if op.archetype_id == "draft_arch_0" and op.op_id.endswith("expert_up")]
    assert draft_gmm
    assert draft_gmm[0].ascend_kernel != "aclnnSwiGluQuant"
    assert draft_gmm[0].needs_new_kernel is True
    assert draft_gmm[0].binding_state == BINDING_STATE_UNBOUND_FORMULA
    for spec in b_aft.kernel_synthesis_specs:
        src_joined = " ".join(str(s) for s in (getattr(spec, "constituent_op_ids", ()) or ()))
        if "expert_up" in src_joined:
            assert getattr(spec, "kernel_name", "") != "aclnnSwiGluQuant"

    # 3. Projected trace sanitizes Feature Track AND conserves MTP x3 + RecoverSSM x36 spans.
    trace_path = tmp_path / "qwen38_mtp3_trace.json"
    export_msprof_trace(b_aft, trace_path)
    raw_text = trace_path.read_text(encoding="utf-8")
    assert RAW_TRACK_ENUM_RE.search(raw_text) is None

    events = json.loads(raw_text)
    root = next(
        ev for ev in events
        if ev.get("cat") == "Framework" and "ModelRunner.execute_model" in str(ev.get("name", ""))
    )
    assert root["args"]["Feature Track"] == "Target projection"
    assert abs(float(root["dur"]) - b_aft.step_span_us) < TRACE_ROOT_PARITY_TOL_US

    draft_layers = [
        ev for ev in events
        if ev.get("cat") == "MSTX_Layer" and ev.get("args", {}).get("Archetype") == "draft_arch_0"
    ]
    assert len(draft_layers) == MTP_THREE_TOKENS

    recover_events = [
        ev for ev in events
        if ev.get("cat") == "Kernel" and "RecoverSSMStateCommit" in str(ev.get("name", ""))
    ]
    assert len(recover_events) == QWEN38_RECOVER_SSM_REPEATS


def test_a3_deepseek_megamoe_contract_and_sheet3_isa_plus_exact_c8_parity(tmp_path: Path) -> None:
    raw_ds_bef, req_ds = compose_export_case(
        "deepseek-v4.1-flash", REPRESENTATIVE_SKUS["a3"],
        dp=1, tp=DEEPSEEK_TP, ep=DEEPSEEK_EP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        concurrency=DEEPSEEK_CONC, feature_track="merged", quant="w4a8_dynamic", kv_cache_dtype="bf16",
    )
    raw_ds_aft, _ = compose_export_case(
        "deepseek-v4.1-flash", REPRESENTATIVE_SKUS["a3"],
        dp=1, tp=DEEPSEEK_TP, ep=DEEPSEEK_EP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        concurrency=DEEPSEEK_CONC, feature_track="expected_ideal", quant="w4a8_dynamic", kv_cache_dtype="bf16",
    )
    b_ds_bef = build_export_bundle(raw_ds_bef, request=req_ds, apply_target_envelope=False)
    b_ds_aft = build_export_bundle(raw_ds_aft, request=req_ds, apply_target_envelope=True)

    # 1. A3 fused MegaMoE never inherits FP32 mHC shape/dtype or "MoE Aux (dispatch.payload_network)".
    mega_ops = [op for op in b_ds_aft.operators if "fused_moe" in op.op_id or op.ascend_kernel == "aclnnMegaMoe"]
    assert mega_ops
    for op in mega_ops:
        assert op.ascend_kernel == "aclnnMegaMoe"
        assert op.submodule == SUB_MOE_ROUTED
        assert "MoE Aux" not in op.logical_role
        assert op.dtype.lower() != "fp32"
        assert "[1,20480]" not in op.tensor_contract
        assert op.flops >= MIN_MEGAMOE_FLOPS

    # 2. Export comparison workbook: verify Sheet 1 H14 == Sheet 4 D3:D9 AND Sheet 3 A3 ISA + exact C8!
    out_xlsx = tmp_path / "deepseek_a3_closure.xlsx"
    export_comparison_xlsx(b_ds_bef, out_xlsx, compare_source=b_ds_aft)
    sheets, roots = _read_workbook_sheets(out_xlsx)

    # Sheet 1 H14 vs Sheet 4 D3/D4/D5/D9 alignment.
    s1 = sheets[SHEET_SUMMARY]
    s4 = sheets[SHEET_PENDING_KERNELS]
    h14 = str(s1["H14"][0] or "")
    m_wf = STAGE_SUMMARY_RE.search(h14)
    assert m_wf is not None, f"Missing 4-Stage summary in Summary!H14: {h14!r}"
    assert float(m_wf.group(1)) == pytest.approx(float(s4["D3"][0] or "0"), abs=STAGE_DISPLAY_TOL_MS)
    assert float(m_wf.group(2)) == pytest.approx(float(s4["D4"][0] or "0"), abs=STAGE_DISPLAY_TOL_MS)
    if m_wf.group(3) is not None:
        assert float(m_wf.group(3)) == pytest.approx(float(s4["D5"][0] or "0"), abs=STAGE_DISPLAY_TOL_MS)
    assert float(m_wf.group(4)) == pytest.approx(float(s4["D9"][0] or "0"), abs=STAGE_DISPLAY_TOL_MS)
    assert abs(float(s4["D10"][0] or "0")) < RECON_TOL_MS

    # Sheet 3 A3 ISA guard: zero PresetLinearFP8 and zero "FP8" formula branches.
    s3 = sheets[SHEET_MEMORY]
    for ref, (val, formula) in s3.items():
        assert "PresetLinearFP8" not in str(val or ""), f"Leaked PresetLinearFP8 at {ref}"
        if ref.startswith("D") and formula:
            assert '"FP8"' not in formula, f"Leaked FP8 formula branch on A3 at {ref}: {formula}"
    assert any(
        ref.startswith("I") and val == QUANT_RULE_PRESET_LINEAR_W8A8
        for ref, (val, _) in s3.items()
    )
    dv_xml = ET.tostring(roots[SHEET_MEMORY], encoding="unicode")
    assert '"BF16,C8"' in dv_xml

    # Exact Sheet 3 T4="C8" parity vs simulator ResidentMemoryReceipt on DeepSeek-V4.1-Flash A3 & Kimi-K3.1 A3.
    ds_c8_calc_bytes = _eval_sheet3_cache_total_for_dtype(s3, "C8")
    raw_ds_c8, _ = compose_export_case(
        "deepseek-v4.1-flash", REPRESENTATIVE_SKUS["a3"],
        dp=1, tp=DEEPSEEK_TP, ep=DEEPSEEK_EP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        concurrency=DEEPSEEK_CONC, feature_track="expected_ideal", quant="w4a8_dynamic", kv_cache_dtype="int8",
    )
    ds_seq = b_ds_aft.memory_ledger.seq_len
    ds_receipt = Memory(parse_case(raw_ds_c8)).resident_memory_receipt(
        phase="decode", sequence_tokens=ds_seq, batch_per_dp=DEEPSEEK_CONC,
    )
    ds_sim_cache_bytes = _receipt_total_cache_bytes(ds_receipt)
    assert abs(ds_c8_calc_bytes - ds_sim_cache_bytes) / max(ds_sim_cache_bytes, 1.0) <= C8_PARITY_REL_TOL

    # Kimi-K3.1 A3 MLA C8 exact parity (656 B/tok vs 1152 B/tok BF16 + paired_sfa reserve).
    raw_km_c8, req_km = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a3"],
        dp=1, tp=DEEPSEEK_TP, ep=DEEPSEEK_EP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        concurrency=DEEPSEEK_CONC, feature_track="merged", quant="w4a8_dynamic", kv_cache_dtype="int8",
    )
    raw_km_bf16, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a3"],
        dp=1, tp=DEEPSEEK_TP, ep=DEEPSEEK_EP,
        prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
        concurrency=DEEPSEEK_CONC, feature_track="merged", quant="w4a8_dynamic", kv_cache_dtype="bf16",
    )
    raw_km_bf16 = {
        **raw_km_bf16,
        "runtime": {**raw_km_bf16["runtime"], "kv_block_size": raw_km_c8["runtime"]["kv_block_size"]},
    }
    b_km_bf16 = build_export_bundle(raw_km_bf16, request=req_km, apply_target_envelope=False)
    km_xlsx = tmp_path / "kimi_a3_c8.xlsx"
    export_comparison_xlsx(b_km_bf16, km_xlsx)
    km_sheets, _ = _read_workbook_sheets(km_xlsx)
    km_s3 = km_sheets[SHEET_MEMORY]
    km_c8_calc_bytes = _eval_sheet3_cache_total_for_dtype(km_s3, "C8")
    km_receipt = Memory(parse_case(raw_km_c8)).resident_memory_receipt(
        phase="decode", sequence_tokens=b_km_bf16.memory_ledger.seq_len, batch_per_dp=DEEPSEEK_CONC,
    )
    km_sim_cache_bytes = _receipt_total_cache_bytes(km_receipt)
    assert abs(km_c8_calc_bytes - km_sim_cache_bytes) / max(km_sim_cache_bytes, 1.0) <= C8_PARITY_REL_TOL


def test_sheet4_surfaces_a5_tilelang_dual_tier_and_ub_wall_warning(tmp_path: Path) -> None:
    raw, req = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        dp=1, tp=2, ep=1, prompt_tokens=OUTPUT_TOKENS, output_tokens=QWEN32_CONC, concurrency=2,
    )
    base = build_export_bundle(raw, request=req, apply_target_envelope=False)
    ub_msg = "UB live 232.0 KiB enters (216 KiB, 248 KiB] ACL 107000 spill zone"
    opp = CoDesignOpportunity(
        opportunity_id="mf_a5_dual_tier_test",
        kind="mega_fusion",
        track="kernel_new",
        readiness="synthesized_target",
        title="A5 Dual-Tier TileLang MegaFusion",
        evidence_refs=("kda.py",),
        mega_fusion=MegaFusionBound(
            region_op_ids=("s0.l0.qkv", "s0.l0.rope"),
            exposed_before_ms=MEGA_BEFORE_MS,
            boundary_lower_bound_ms=MEGA_BOUND_MS,
            tiled_engine_ms=MEGA_TILED_MS,
            gain_ceiling_ms=MEGA_BEFORE_MS - MEGA_BOUND_MS,
            gain_expected_ms=MEGA_BEFORE_MS - MEGA_TILED_MS,
            eliminated_gm_bytes=MEGA_GM_BYTES,
            ub_live_bytes=MEGA_UB_LIVE_BYTES,
            ub_capacity_bytes=MEGA_UB_CAP_BYTES,
            tilelang_simt_gain_ms=SIMT_GAIN_MS,
            tilelang_simd_or_ascendc_gain_ms=SIMD_GAIN_MS,
            ub_wall_warning=ub_msg,
        ),
    )
    aft = replace(base, co_design_opportunities=(opp,))
    out_xlsx = tmp_path / "a5_tilelang_ub_card.xlsx"
    export_comparison_xlsx(base, out_xlsx, compare_source=aft)
    sheets, _ = _read_workbook_sheets(out_xlsx)
    s4_text = "\n".join(str(v or "") for v, _ in sheets[SHEET_PENDING_KERNELS].values())
    assert "TileLang (SimtVF quick-win)" in s4_text
    assert "TileLang (SimdVF) / AscendC" in s4_text
    assert "(216 KiB, 248 KiB]" in s4_text

    brief = build_agent_brief(base, aft)
    mf_brief = brief["codesign_opportunities"][0]["mega_fusion"]
    assert mf_brief["tilelang_simt_gain_ms"] == pytest.approx(SIMT_GAIN_MS)
    assert mf_brief["tilelang_simd_or_ascendc_gain_ms"] == pytest.approx(SIMD_GAIN_MS)
    assert "(216 KiB, 248 KiB]" in mf_brief["ub_wall_warning"]


def test_cli_report_forwards_mtp_aliases_and_pd_decode_slo_flags() -> None:
    parser = build_parser()
    for slo_val in (TPOT_SLO_TIGHT_MS, TPOT_SLO_RELAXED_MS):
        parsed_args = parser.parse_args([
            "report",
            "--model", "qwen3-32b",
            "--hardware", "a3",
            "--prompt-tokens", str(LONG_PROMPT_TOKENS),
            "--output-tokens", str(LONG_OUTPUT_TOKENS),
            "--deployment-mode", "pd_decode",
            "--tpot-slo-ms", str(slo_val),
            "--mtp-tokens", str(MTP_FIVE_TOKENS),
            "--mtp-accept", str(MTP_FIVE_ACCEPT),
        ])
        assert parsed_args.prompt_tokens == LONG_PROMPT_TOKENS
        assert parsed_args.output_tokens == LONG_OUTPUT_TOKENS
        assert parsed_args.deployment_mode == "pd_decode"
        assert parsed_args.tpot_ms == pytest.approx(slo_val)
        assert parsed_args.spec_tokens == MTP_FIVE_TOKENS
        assert parsed_args.accept_len == pytest.approx(MTP_FIVE_ACCEPT)

    common_kw = {
        "topology_preset": None,
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "tpot_slo_ms": TPOT_SLO_RELAXED_MS,
        "ttft_slo_ms": None,
        "mtp_proposal_tokens": MTP_FIVE_TOKENS,
        "mtp_acceptance_length": MTP_FIVE_ACCEPT,
        "engram_cpu_offload": None,
        "deployment_mode": "pd_decode",
        "runtime_stack": None,
    }
    par = {"dp": 1, "tp": DEEPSEEK_TP, "ep": DEEPSEEK_EP, "pp": 1, "dcp": 1, "pcp": 1}
    for track in ("merged", "expected_ideal"):
        raw_col, step_col, _ = _report_column(
            None,
            model="deepseek-v4.1-flash", sku=REPRESENTATIVE_SKUS["a3"], phase="decode", track=track,
            quant="w4a8_dynamic", kv="bf16", idx_kv=None, par=par, conc=QWEN32_CONC,
            common_kw=common_kw, spec_tokens=MTP_FIVE_TOKENS, accept_len=MTP_FIVE_ACCEPT,
        )
        rt_dict = dict(raw_col.get("runtime") or {})
        rt_meta = dict(rt_dict.get("metadata") or {})
        assert int(rt_meta.get("draft_proposal_tokens", 0)) == MTP_FIVE_TOKENS
        assert float(rt_dict.get("mtp_acceptance_length", 0.0)) == pytest.approx(MTP_FIVE_ACCEPT)
        assert step_col.tokens == 1 + MTP_FIVE_TOKENS
        assert step_col.scope == "model"
