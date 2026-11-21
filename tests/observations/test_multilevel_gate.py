"""Tests for multilevel gate, lineage, report assembly and tail diagnosis.

[Role] Verify capture boundaries, signed Engine evidence and heldout verdicts.
[Invariants] Tier-0 execution with live Engine events and independent evidence.
[Timeouts & Locks] Deterministic in-process execution.
[Side-Effect Policy] Test manifests and reports stay inside pytest tmp_path.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import pytest

from mindie_sim.observations.multilevel_gate import (
    CaptureBoundary, CohortKey, EngineInterval, EngineRunReceipt, FrozenFitManifest,
    FrozenFitParameter, GateCase, MS_TO_NS, Manifest, Residual, bind_engine_interval,
    bind_engine_rung_interval, evaluate, gate_decision, parse_gate_case, run_l1,
    run_l2, run_l3, sanitize_capture_uri, sanitize_source_ref, stream_jsonl,
    stream_manifest, summarize_cohorts, validate_lineage,
)
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_SCRIPT = REPO_ROOT / "scripts" / "diagnose_tail_cluster.py"
PRODUCER_SCRIPT = REPO_ROOT / "data" / "verification" / "a5-cann93-multilevel-accuracy-20261120.producer.py"
CLOCK_MERGED_BUSY_NS = 300
OUTLIER_APE_PCT = 27
TRACE_LIMIT = 32
TAIL_TRACE_APE = 0.18
INDEPENDENT_L2_CAPTURE_US = 180.0
MODEL_RUN_CAPTURE = {
    "model_id": "qwen3.8-27b", "rung": "merged", "status": "ok",
    "num_layers": 4, "orig_layers": 4, "dp": 1, "tp": 2, "ep": False,
    "fx_compilation_summary": {"pre_split_full_graph_summary": {"op_counts": {"vllm.unquantized_gemm": 4}}, "num_compiled_fx_subgraphs": 1},
    "resolved_compilation_and_ascend_config": {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [8]},
    "step_and_layer_timings": {"decode_B8_S512_step_p50_ms": 0.92, "decode_B8_S512_per_layer_us": 230.0},
}

def _load_script(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def _sha(tag: str) -> str:
    return (tag * 64)[:64]

def _cohort(level: str = "L1", name: str = "gemm") -> CohortKey:
    return CohortKey(level, name, "device_x", "prefill", "eager", "medium", "tp1")  # type: ignore[arg-type]

def _boundary(
    tag: str, *, kind: str = "device_kernel", parents: tuple[str, ...] = (),
    events: tuple[str, ...] = (), acq: str | None = None, run: str | None = None,
    unit: str = "ns", uri: str = "", raw_sha: str | None = None, session_id: str = "",
) -> CaptureBoundary:
    return CaptureBoundary(
        raw_sha256=raw_sha or _sha(tag), acquisition_uid=acq or f"acq-{tag}", run_uid=run or f"run-{tag}",
        parent_digests=parents, event_ids=events or (f"ev-{tag}",), device_ranks=(0,),
        clock_domain="device_monotonic", start_event="k_start", end_event="k_end",
        boundary_kind=kind, unit=unit, capture_uri=uri or f"capture://suite/{tag}", session_id=session_id,
    )

def _case(
    cid: str, *, level: str = "L1", split: str = "heldout", tag: str = "a",
    kind: str | None = None, measured_ns: int | None = 1000, parents: tuple[str, ...] = (),
    events: tuple[str, ...] = (), acq: str | None = None, run: str | None = None,
    raw_sha: str | None = None, session_id: str = "", uri: str = "",
    debug_cross_level: bool = False, is_counterfactual: bool = False,
    has_matched_capture: bool = True, block_capture_present: bool = True,
    arm_a: int | None = None, arm_b: int | None = None, cohort_name: str = "gemm",
) -> GateCase:
    default_kind = {"L1": "device_kernel", "L2": "fx_region", "L3": "model_step"}[level]
    return GateCase(
        case_id=cid, level=level, split=split,  # type: ignore[arg-type]
        boundary=_boundary(
            tag, kind=kind or default_kind, parents=parents, events=events, acq=acq,
            run=run, raw_sha=raw_sha, session_id=session_id, uri=uri,
        ),
        input_digest=f"in-{cid}", stack_digest="stack-1", implementation="canonical_impl",
        cohort=_cohort(level, cohort_name), measured_ns=measured_ns,
        debug_cross_level=debug_cross_level, is_counterfactual=is_counterfactual,
        has_matched_capture=has_matched_capture, block_capture_present=block_capture_present,
        rung_arm_a_measured_ns=arm_a, rung_arm_b_measured_ns=arm_b,
        regime_dims=(("dtype_layout", "bf16_nd"), ("shape", "m1024"), ("warmth", "warm")),
    )

def _run_engine_for_ns(start_name: str, end_name: str, duration_ns: int) -> tuple[Engine, Any, Any]:
    eng = Engine([Resource("device", MS_TO_NS, "ns", "fifo")], trace_limit=TRACE_LIMIT)
    s_ev = eng.submit(start_name, {"device": 100.0})
    e_ev = eng.submit(end_name, {"device": float(duration_ns)}, after=(s_ev,))
    eng.advance(until=(e_ev,))
    return eng, s_ev, e_ev

def _live_interval(case: GateCase, predicted_ns: int, **kw: Any) -> EngineInterval:
    eng, s_ev, e_ev = _run_engine_for_ns(case.boundary.start_event, case.boundary.end_event, predicted_ns)
    return bind_engine_interval(case, eng, s_ev, e_ev, **kw)

class _LiveEngineRunner:
    def __init__(self, overrides: dict[str, EngineInterval] | None = None, scale: float = 1.05) -> None:
        self._overrides, self._scale = overrides or {}, scale

    def _run(self, case: GateCase) -> EngineInterval:
        if case.case_id in self._overrides:
            return self._overrides[case.case_id]
        if case.boundary.boundary_kind == "paired_rung_delta":
            assert case.rung_arm_a_measured_ns is not None and case.rung_arm_b_measured_ns is not None
            pa = max(1, int(round(case.rung_arm_a_measured_ns * self._scale)))
            pb = max(1, int(round(case.rung_arm_b_measured_ns * self._scale)))
            return bind_engine_rung_interval(
                case,
                arm_a=_run_engine_for_ns(case.boundary.start_event, case.boundary.end_event, pa),
                arm_b=_run_engine_for_ns(case.boundary.start_event, case.boundary.end_event, pb),
            )
        return _live_interval(case, max(1, int(round((case.measured_ns or 1000) * self._scale))))

    run_l1 = run_l2 = run_l3 = _run

def test_lineage_case_normalization_event_scoping_and_fit_closure(tmp_path: Path) -> None:
    fits = FrozenFitManifest("f1", "1", ())
    c_train = _case("c_train", split="train", tag="a", acq="acq-train", run="run-train")
    c_held_upper = _case("c_held_up", split="heldout", tag="b", acq="acq-held", run="run-held", raw_sha=_sha("a").upper())
    with pytest.raises(ModelingError, match="lineage_split_leakage"):
        validate_lineage(Manifest("m_upper", "1", (c_train, c_held_upper)), fits)
    c_held = _case("c_held", split="heldout", tag="2", parents=("mid_digest",))
    with pytest.raises(ModelingError, match="lineage_split_leakage"):
        validate_lineage(Manifest("m1", "1", (c_train, c_held), lineage_edges=((_sha("a").upper(), "mid_digest"),)), fits)
    with pytest.raises(ModelingError, match="duplicate_case_id"):
        validate_lineage(Manifest("m2", "1", (c_train, _case("c_train", split="train", tag="3"))), fits)
    c_tr_ev = _case("tr_ev", split="train", tag="1", acq="acq-1", run="run-1", session_id="s1", events=("ev-0", "ev-1"))
    c_ho_ev = _case("ho_ev", split="heldout", tag="2", acq="acq-2", run="run-2", session_id="s2", events=("ev-0", "ev-1"))
    idx_ev = validate_lineage(Manifest("m_ev", "1", (c_tr_ev, c_ho_ev)), fits)
    assert idx_ev.root_for("tr_ev") != idx_ev.root_for("ho_ev")
    with pytest.raises(ModelingError, match="lineage_fit_heldout_leakage"):
        validate_lineage(
            Manifest("m_fit_up", "1", (c_tr_ev, c_ho_ev)),
            FrozenFitManifest("f_up", "1", (FrozenFitParameter("p_bw", (_sha("2").upper(),), "capture://fits/bw"),)),
        )
    c_ho_chain = _case("h_chain", split="heldout", tag="3", parents=("mid_b",))
    leaking_fit = FrozenFitManifest(
        "f_chain", "1", (FrozenFitParameter("p_lat", ("fit_root_0",), "capture://fits/lat"),),
        lineage_edges=(("fit_root_0", "mid_a"), ("mid_a", "mid_b")),
    )
    with pytest.raises(ModelingError, match="lineage_fit_heldout_leakage"):
        validate_lineage(Manifest("m_chain", "1", (c_tr_ev, c_ho_chain)), leaking_fit)
    clean_fit = FrozenFitManifest("f_ok", "1", (FrozenFitParameter("p_bw", (_sha("1").upper(),), "capture://fits/bw"),))
    idx_fit = validate_lineage(Manifest("m_ok", "1", (c_tr_ev, c_ho_ev)), clean_fit)
    assert idx_fit.root_for("tr_ev") in idx_fit.fit_ancestor_components["p_bw"]
    cap_dir = tmp_path / "suite"
    cap_dir.mkdir()
    cap_doc = {"device_ranks": [0], "clock_domain": "device_monotonic", "start_event": "k_start", "end_event": "k_end", "event_ids": ["ev-ok"]}
    cap_path = cap_dir / "cap_ok.json"
    cap_path.write_text(json.dumps(cap_doc), encoding="utf-8")
    good_case = _case("cap_ok", tag="4", raw_sha=hashlib.sha256(cap_path.read_bytes()).hexdigest(), events=("ev-ok",), uri="capture://suite/cap_ok.json")
    validate_lineage(Manifest("m_disk", "1", (good_case,)), fits, capture_root=tmp_path, require_capture_files=True)
    with pytest.raises(ModelingError, match="capture_file_missing"):
        validate_lineage(Manifest("m_noloc", "1", (good_case,)), fits, require_capture_files=True)
    with pytest.raises(ModelingError, match="capture_digest_mismatch"):
        validate_lineage(Manifest("m_b1", "1", (_case("cap_bad_sha", tag="4", raw_sha=_sha("f"), events=("ev-ok",), uri="capture://suite/cap_ok.json"),)), fits, capture_root=tmp_path)
    for fname, bad_payload, err in (
        ("bad_rank.json", {**cap_doc, "device_ranks": [1]}, "capture_rank_mismatch"),
        ("bad_clk.json", {**cap_doc, "clock_domain": "host_wall"}, "capture_clock_domain_mismatch"),
        ("bad_clock_top.json", {**cap_doc, "clock_operator": "elapsed_span"}, "capture_clock_operator_mismatch"),
        ("bad_clock_boundary.json", {**cap_doc, "boundary": {"clock_operator": "elapsed_span"}}, "capture_clock_operator_mismatch"),
        ("bad_ev.json", {**cap_doc, "event_ids": ["other_ev"]}, "capture_event_mismatch"),
        ("bad_se.json", {**cap_doc, "start_event": "wrong_start"}, "capture_event_mismatch"),
        ("unrel.json", {"summary": "unrelated_without_events"}, "capture_event_mismatch"),
    ):
        fp = cap_dir / fname
        fp.write_text(json.dumps(bad_payload), encoding="utf-8")
        c_bad = _case(fname, tag="5", raw_sha=hashlib.sha256(fp.read_bytes()).hexdigest(), events=("ev-ok",), uri=f"capture://suite/{fname}")
        c_bad = replace(c_bad, boundary=replace(c_bad.boundary, clock_operator="busy_union"))
        with pytest.raises(ModelingError, match=err):
            validate_lineage(Manifest(f"m_{fname}", "1", (c_bad,)), fits, capture_root=tmp_path, require_capture_files=True)

def test_cross_level_root_overlap_rejected_unless_debug_only() -> None:
    c_l1 = _case("l1_case", level="L1", split="heldout", tag="a", acq="shared_root")
    c_l2 = _case("l2_case", level="L2", split="heldout", tag="b", parents=("shared_root",))
    fits = FrozenFitManifest("f1", "1", ())
    with pytest.raises(ModelingError, match="lineage_cross_level_root_overlap"):
        validate_lineage(Manifest("m_strict", "1", (c_l1, c_l2)), fits)
    c_l2_dbg = _case("l2_dbg", level="L2", split="heldout", tag="b", parents=("shared_root",), debug_cross_level=True)
    manifest = Manifest("m_dbg", "1", (c_l1, c_l2_dbg))
    idx = validate_lineage(manifest, fits)
    assert "l2_dbg" in idx.debug_only_cases
    residuals = list(evaluate(manifest, fits, _LiveEngineRunner(scale=1.04)))
    by_id = {r.case_id: r for r in residuals}
    assert by_id["l1_case"].status == "ok" and by_id["l2_dbg"].status == "debug_only"
    summaries = summarize_cohorts(residuals)
    assert summaries[c_l2_dbg.cohort].status == "insufficient_evidence"
    assert ("debug_cross_level_not_blind_acceptance", 1) in summaries[c_l2_dbg.cohort].excluded_reasons

def test_engine_receipt_binding_and_mandatory_three_level_coverage() -> None:
    c_l1 = _case("l1_off", level="L1", tag="1", measured_ns=1000)
    with pytest.raises(ModelingError, match="engine_receipt_unbound"):
        run_l1(c_l1, _LiveEngineRunner({"l1_off": EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "some-string")}))
    with pytest.raises(ModelingError, match="engine_interval_case_id_mismatch"):
        run_l1(c_l1, _LiveEngineRunner({"l1_off": EngineInterval("DIFFERENT_CASE", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "some-string")}))
    forged_rcpt = EngineRunReceipt("capture://r/1", "l1_off", "L1", "device_kernel", "u1", "k_start", "k_end", 0, 1, 0, 1050, 1050, "0" * 64)
    with pytest.raises(ModelingError, match="engine_receipt_signature_invalid"):
        run_l1(c_l1, _LiveEngineRunner({"l1_off": EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "capture://r/1", run_receipt=forged_rcpt)}))
    eng_bad, s_bad, e_bad = _run_engine_for_ns("wrong_start", "k_end", 1050)
    with pytest.raises(ModelingError, match="engine_event_name_mismatch"):
        bind_engine_interval(c_l1, eng_bad, s_bad, e_bad)
    eng_unset = Engine([Resource("device", 1_000_000.0, "ns", "fifo")])
    u_start = eng_unset.submit("k_start", {"device": 100.0})
    u_end = eng_unset.submit("k_end", {"device": 1050.0}, after=(u_start,))
    with pytest.raises(ModelingError, match="engine_event_handles_unsettled"):
        bind_engine_interval(c_l1, eng_unset, u_start, u_end)
    for bad_iv, err in (
        (EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "capture://r/1", external_offset_ns=4200), "non_engine_or_offset_rejected"),
        (EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "capture://r/1", engine_sourced=False), "non_engine_or_offset_rejected"),
        (EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, ""), "engine_receipt_missing"),
    ):
        with pytest.raises(ModelingError, match=err):
            run_l1(c_l1, _LiveEngineRunner({"l1_off": bad_iv}))
    c_l2 = _case("l2_clone", level="L2", tag="2", measured_ns=5000)
    with pytest.raises(ModelingError, match="l2_cloned_or_divided_from_l3_forbidden"):
        run_l2(c_l2, _LiveEngineRunner({"l2_clone": _live_interval(c_l2, 5000, derived_from_step_division=True)}))
    with pytest.raises(ModelingError, match="l2_cloned_or_divided_from_l3_forbidden"):
        run_l2(c_l2, _LiveEngineRunner({"l2_clone": _live_interval(c_l2, 5000, source_level="L3")}))
    c_l3 = _case("l3_step", level="L3", tag="3", measured_ns=50_000)
    shared_eng, shared_s, shared_e = _run_engine_for_ns("k_start", "k_end", 5000)
    iv_l2 = bind_engine_interval(c_l2, shared_eng, shared_s, shared_e)
    iv_l3 = bind_engine_interval(c_l3, shared_eng, shared_s, shared_e)
    with pytest.raises(ModelingError, match="cross_level_engine_receipt_clone"):
        list(evaluate(Manifest("m_rcpt", "1", (c_l2, c_l3)), FrozenFitManifest("f", "1", ()), _LiveEngineRunner({"l2_clone": iv_l2, "l3_step": iv_l3})))
    other_eng, _, _ = _run_engine_for_ns("other_start", "other_end", 5000)
    twin_eng, _, _ = _run_engine_for_ns("k_start", "k_end", 5000)
    untraced_eng = Engine([Resource("device", 1_000_000.0, "ns", "fifo")])
    us = untraced_eng.submit("k_start", {"device": 100.0})
    ue = untraced_eng.submit("k_end", {"device": 1000.0}, after=(us,))
    untraced_eng.advance(until=(ue,))
    with pytest.raises(ModelingError, match="engine_event_not_owned_by_engine"):
        bind_engine_interval(c_l1, other_eng, shared_s, shared_e)
    with pytest.raises(ModelingError, match="engine_event_not_owned_by_engine:.*cross_engine_handle"):
        bind_engine_interval(c_l1, twin_eng, shared_s, shared_e)
    with pytest.raises(ModelingError, match="engine_event_not_owned_by_engine:.*trace_required"):
        bind_engine_interval(c_l1, untraced_eng, us, ue)
    fits, runner = FrozenFitManifest("f", "1", ()), _LiveEngineRunner(scale=1.03)
    dec_l1 = gate_decision(summarize_cohorts(list(evaluate(Manifest("m_l1", "1", (c_l1,)), fits, runner))))
    assert dec_l1["status"] == "insufficient_evidence" and dec_l1["missing_levels"] == ["L2", "L3"]
    dec_l12 = gate_decision(summarize_cohorts(list(evaluate(Manifest("m_l12", "1", (c_l1, c_l2)), fits, runner))))
    assert dec_l12["status"] == "insufficient_evidence" and dec_l12["missing_levels"] == ["L3"]
    dec_all = gate_decision(summarize_cohorts(list(evaluate(Manifest("m_all", "1", (c_l1, c_l2, c_l3)), fits, runner))))
    assert dec_all["status"] == "pass" and dec_all["missing_levels"] == [] and dec_all["covered_levels"] == ["L1", "L2", "L3"]
    dup_root_res = [
        Residual("a1", "L1", _cohort("L1", "f1"), "same_root", 1000, 1020, "capture://r/1", 0.02, 0.02),
        Residual("a2", "L2", _cohort("L2", "f2"), "same_root", 2000, 2040, "capture://r/2", 0.02, 0.02),
        Residual("a3", "L3", _cohort("L3", "f3"), "other_root", 5000, 5100, "capture://r/3", 0.02, 0.02),
    ]
    dec_overlap = gate_decision(summarize_cohorts(dup_root_res))
    assert dec_overlap["status"] == "insufficient_evidence" and dec_overlap["overlapping_level_roots"] == ["same_root"]

def test_l2_absent_capture_near_zero_rung_and_strict_threshold() -> None:
    fits = FrozenFitManifest("f", "1", ())
    c_missing = _case("l2_no_blk", level="L2", tag="4", measured_ns=4000, block_capture_present=False)
    assert run_l2(c_missing, _LiveEngineRunner()).status == "missing_evidence"
    res_miss = list(evaluate(Manifest("m_miss", "1", (c_missing,)), fits, _LiveEngineRunner()))
    assert res_miss[0].status == "missing_evidence" and math.isnan(res_miss[0].signed_error)
    assert summarize_cohorts(res_miss)[c_missing.cohort].status == "insufficient_evidence"
    near = _case("rung_near", level="L2", tag="5", kind="paired_rung_delta", measured_ns=200, arm_a=100_000, arm_b=100_200)
    wide = _case("rung_wide", level="L2", tag="6", kind="paired_rung_delta", measured_ns=40_000, arm_a=100_000, arm_b=140_000)
    overrides = {
        "rung_near": bind_engine_rung_interval(near, arm_a=_run_engine_for_ns("k_start", "k_end", 101_000), arm_b=_run_engine_for_ns("k_start", "k_end", 101_350)),
        "rung_wide": bind_engine_rung_interval(wide, arm_a=_run_engine_for_ns("k_start", "k_end", 102_000), arm_b=_run_engine_for_ns("k_start", "k_end", 144_000)),
    }
    rung_res = {r.case_id: r for r in evaluate(Manifest("m_rung", "1", (near, wide)), fits, _LiveEngineRunner(overrides))}
    assert rung_res["rung_near"].status == "near_zero_delta" and rung_res["rung_near"].signed_delta_ns == 150
    assert rung_res["rung_near"].arm_a_signed_error == pytest.approx(0.01)
    assert rung_res["rung_wide"].status == "ok" and rung_res["rung_wide"].absolute_percentage_error == pytest.approx(0.05)
    assert summarize_cohorts([]) == {} and gate_decision({})["status"] == "insufficient_evidence"
    c_zero = _case("zero_dur", level="L1", tag="7", measured_ns=0)
    c_cf = _case("cf_only", level="L3", tag="8", measured_ns=9000, is_counterfactual=True, has_matched_capture=False)
    inv_by_id = {r.case_id: r for r in evaluate(Manifest("m_inv", "1", (c_zero, c_cf)), fits, _LiveEngineRunner())}
    assert inv_by_id["zero_dur"].status == "invalid_evidence" and inv_by_id["cf_only"].status == "counterfactual_excluded"
    c_exact = _case("exact_10", level="L1", tag="1", measured_ns=1000, cohort_name="boundary_fam")
    c_pass = _case("below_10", level="L1", tag="2", measured_ns=1000, cohort_name="pass_fam")
    thr_res = list(evaluate(
        Manifest("m_thr", "1", (c_exact, c_pass)), fits,
        _LiveEngineRunner({"exact_10": _live_interval(c_exact, 1100), "below_10": _live_interval(c_pass, 1099)}),
    ))
    thr_sum = summarize_cohorts(thr_res, threshold_pct=10.0)
    assert thr_sum[c_exact.cohort].status == "fail" and thr_sum[c_pass.cohort].status == "pass"
    assert gate_decision(thr_sum)["status"] == "fail"
    ten_res = [
        Residual(f"p{i}", "L1", _cohort("L1", "p90_fam"), f"root-{i}", 1000, 1000 + err, f"capture://r/{i}", err / 1000.0, err / 1000.0)
        for i, err in enumerate((10, 20, 30, 40, 50, 60, 70, 80, 90, 250))
    ]
    p90_sum = summarize_cohorts(ten_res)[_cohort("L1", "p90_fam")]
    assert p90_sum.p90_ape_pct == pytest.approx(9.0) and p90_sum.tail_case_ids == ("p9",)

def _raw_case(cid: str, split: str, tag: str, pad_bytes: int = 0) -> dict[str, Any]:
    return {
        "case_id": cid, "level": "L1", "split": split, "input_digest": f"in-{cid}",
        "stack_digest": "st1", "implementation": "impl1", "measured_duration": 12.5,
        "boundary": {
            "raw_sha256": _sha(tag), "acquisition_uid": f"acq-{tag}", "run_uid": f"run-{tag}",
            "parent_digests": [], "event_ids": ["e1"], "device_ranks": [0, 1],
            "clock_domain": "device_ns", "start_event": "s", "end_event": "e",
            "boundary_kind": "device_kernel", "unit": "us", "capture_uri": f"capture://campaign/{cid}",
            "source_ref": {"repository": "mindie-agent/sim", "revision": "a" * 40,
                           "relative_path": "mindie_sim/observations/multilevel_gate.py", "source_sha256": _sha("d")},
        },
        "cohort": {"cohort_name": "gemm", "sku": "device_x", "phase": "prefill", "mode": "eager", "shape_regime": "medium", "parallel_topology": "tp2"},
        "payload": {"blob": "x" * pad_bytes} if pad_bytes else {},
    }

def test_pointwise_outlier_blocks_release() -> None:
    cases = tuple(_case(f"point_{tag}", tag=tag) for tag in ("a", "b", "c"))
    outlier = cases[-1]
    prediction_ns = 1000 + OUTLIER_APE_PCT * (1000 // 100)
    runner = _LiveEngineRunner({outlier.case_id: _live_interval(outlier, prediction_ns)}, scale=1)
    rows = list(evaluate(Manifest("points", "1", cases), FrozenFitManifest("f", "1", ()), runner))
    assert [row.absolute_percentage_error * 100 for row in rows] == [0, 0, OUTLIER_APE_PCT]
    summaries = summarize_cohorts(rows)
    assert summaries[outlier.cohort].status == "fail"
    assert summaries[outlier.cohort].failed_metrics == ("max_ape_pct",)
    relaxed = summarize_cohorts(rows, strict_pointwise=False)
    assert relaxed[outlier.cohort].status == "pass" and relaxed[outlier.cohort].max_ape_pct == OUTLIER_APE_PCT
    verdict = gate_decision(relaxed, required_levels=("L1",))
    assert verdict["status"] == "fail" and verdict["failed_cohorts"] == [outlier.cohort.as_id()]
    assert verdict["pointwise_threshold_exceeded"] == [outlier.cohort.as_id()]
    assert gate_decision(summaries, plan_only=True, calibrated_count=0)["status"] == "fail"
    planned = tuple(case.case_id for case in cases)
    omitted = gate_decision(summarize_cohorts(rows[:-1]), required_levels=("L1",),
                            expected_case_ids=planned, evaluated_case_ids=planned)
    assert omitted["status"] == "insufficient_evidence" and omitted["missing_planned_cases"] == [outlier.case_id]

def test_missing_planned_case_blocks_release() -> None:
    case = _case("evaluated")
    rows = list(evaluate(Manifest("grid", "1", (case,)), FrozenFitManifest("f", "1", ()), _LiveEngineRunner()))
    summaries = summarize_cohorts(rows)
    planned = ("evaluated", "missing_cell")
    assert gate_decision(summaries, required_levels=("L1",), expected_case_ids=planned[:1],
                         evaluated_case_ids=[case.case_id])["status"] == "pass"
    missing = replace(rows[0], case_id="missing_cell", status="missing_evidence")
    verdict = gate_decision(summaries, required_levels=("L1",), expected_case_ids=planned, evaluated_case_ids=[*rows, missing])
    assert verdict["status"] == "insufficient_evidence" and verdict["missing_planned_cases"] == ["missing_cell"]
    assert summarize_cohorts([*rows, missing])[case.cohort].status == "insufficient_evidence"

def test_busy_union_clock_matches_capture_and_merges_clipped_trace() -> None:
    case = _case("clock")
    eng = Engine([Resource(name, MS_TO_NS, "ns", "fifo") for name in ("device", "aux", "host@r0:s0")], trace_limit=TRACE_LIMIT)
    start = eng.submit("k_start", {"device": 100})
    eng.submit("left_crossing", {"aux": 2 * 100})
    eng.submit("overlap", {"device": 2 * 100}, release=(100 + 100 / 2) / MS_TO_NS)
    eng.submit("host_work", {"host@r0:s0": 100}, release=(1000 / 2) / MS_TO_NS)
    stop_ns = 1000 + 100
    end = eng.submit("k_end", release=stop_ns / MS_TO_NS)
    late = eng.submit("right_crossing", {"aux": 2 * 100}, release=(stop_ns - 100 / 2) / MS_TO_NS)
    busy_case = replace(case, boundary=replace(case.boundary, clock_operator="busy_union"), measured_ns=CLOCK_MERGED_BUSY_NS)
    eng.advance(until=(end,))
    with pytest.raises(ModelingError, match="busy_union_trace_incomplete:clock"):
        bind_engine_interval(busy_case, eng, start, end)
    eng.advance(until=(late,))
    elapsed = bind_engine_interval(busy_case, eng, start, end, clock_operator="elapsed_span")
    with pytest.raises(ModelingError, match="clock_operator_mismatch:clock:busy_union:elapsed_span"):
        run_l1(busy_case, _LiveEngineRunner({"clock": elapsed}))
    interval = bind_engine_interval(busy_case, eng, start, end)
    runner = _LiveEngineRunner({"clock": interval})
    assert run_l1(busy_case, runner).predicted_ns == CLOCK_MERGED_BUSY_NS
    row, = evaluate(Manifest("clock", "1", (busy_case,)), FrozenFitManifest("f", "1", ()), runner)
    assert (row.clock_operator, row.elapsed_span_ns, row.busy_union_ns) == ("busy_union", 1000, CLOCK_MERGED_BUSY_NS)
    assert row.absolute_percentage_error == 0
    equal = _live_interval(case, 1000)
    assert equal.run_receipt is not None
    forged = replace(equal, clock_operator="busy_union", run_receipt=replace(equal.run_receipt, clock_operator="busy_union"))
    with pytest.raises(ModelingError, match="engine_receipt_signature_invalid:clock"):
        run_l1(busy_case, _LiveEngineRunner({"clock": forged}))

@pytest.mark.parametrize("trace_limit", (0, 1, TRACE_LIMIT))
def test_busy_union_idle_markers_require_complete_trace(trace_limit: int) -> None:
    case = _case("idle_clock")
    case = replace(case, boundary=replace(case.boundary, clock_operator="busy_union"))
    eng = Engine([], trace_limit=trace_limit)
    start = eng.submit(case.boundary.start_event)
    end = eng.submit(case.boundary.end_event, release=1000 / MS_TO_NS)
    eng.advance(until=(end,))
    if trace_limit < 2:
        with pytest.raises(ModelingError, match="busy_union_trace_incomplete:idle_clock"):
            bind_engine_interval(case, eng, start, end)
        return
    interval = bind_engine_interval(case, eng, start, end)
    assert (interval.busy_union_ns, interval.predicted_ns, interval.elapsed_span_ns) == (0, 0, 1000)
    with pytest.raises(ModelingError, match="interval_span_or_duration_invalid:idle_clock"):
        run_l1(case, _LiveEngineRunner({case.case_id: interval}))

def test_caller_predictions_and_uncalibrated_plans_cannot_pass() -> None:
    raw = _raw_case("caller", "heldout", "c")
    for key in ("predicted_ns", "predicted_ms", "predicted_us", "predicted_duration", "predicted_s"):
        for invalid in ({**raw, key: 1000}, {**raw, "payload": {key: None}}):
            with pytest.raises(ModelingError, match="caller_predicted_duration_forbidden:caller"):
                parse_gate_case(invalid)
    case = parse_gate_case(raw)
    with pytest.raises(ModelingError, match="clock_operator_invalid:caller"):
        parse_gate_case({**raw, "boundary": {**raw["boundary"], "clock_operator": []}})
    rows = evaluate(Manifest("caller", "1", (case,)), FrozenFitManifest("f", "1", ()), _LiveEngineRunner())
    summaries = summarize_cohorts(rows)
    for count, plan_only in ((0, False), (1, True)):
        verdict = gate_decision(summaries, required_levels=("L1",), calibrated_count=count, plan_only=plan_only)
        assert verdict["status"] == "insufficient_evidence"
        assert (verdict["calibrated_count"], verdict["plan_only"]) == (count, plan_only)

def test_two_pass_streaming_past_64k_header_and_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mindie_sim.observations.multilevel_gate as mg
    import mindie_sim.observations.multilevel_stream as ms
    big_doc = {
        "cases": [_raw_case(f"tr_{i}", "train", f"{i:x}"[-1] or "1", pad_bytes=9000) for i in range(10)]
        + [_raw_case("ho_leak", "heldout", "f")],
        "manifest_id": "trailing_meta_manifest",
        "strict_three_level_blind": True,
        "lineage_edges": [["acq-1", "acq-f"]],
    }
    raw_text = json.dumps(big_doc)
    assert len(raw_text) > 85_000 and raw_text.index('"lineage_edges"') > 70_000
    json_path = tmp_path / "trailing.json"
    json_path.write_text(raw_text, encoding="utf-8")

    monkeypatch.setattr(mg, "ijson", None)
    monkeypatch.setattr(json, "load", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("whole-file json.load forbidden")))
    m_streamed = stream_manifest(json_path, max_whole_read_bytes=1)
    assert m_streamed.manifest_id == "trailing_meta_manifest" and len(m_streamed.cases) == 11
    assert not isinstance(m_streamed.cases, (list, tuple))
    assert m_streamed.lineage_edges == (("acq-1", "acq-f"),)
    with pytest.raises(ModelingError, match="lineage_split_leakage"):
        validate_lineage(m_streamed, FrozenFitManifest("f", "1", ()))
    jsonl_path = tmp_path / "manifest.jsonl"
    jsonl_path.write_text(
        json.dumps({"record_type": "manifest_header", "manifest_id": "stream_m", "source_uri": "capture://campaign/m"}) + "\n"
        + json.dumps({"record_type": "lineage_edge", "parent": "p_root", "child": "acq-c"}) + "\n"
        + json.dumps(_raw_case("op_stream_1", "heldout", "c")) + "\n",
        encoding="utf-8",
    )
    assert len(list(stream_jsonl(jsonl_path))) == 3
    m_jsonl = stream_manifest(jsonl_path)
    assert m_jsonl.cases[0].measured_ns == 12_500 and m_jsonl.lineage_edges == (("p_root", "acq-c"),)
    ev_rows = list(evaluate(jsonl_path, FrozenFitManifest("f", "1", ()), _LiveEngineRunner(scale=1.02)))
    assert len(ev_rows) == 1 and ev_rows[0].status == "ok"
    base_case = _raw_case("bad", "heldout", "c")
    with pytest.raises(ModelingError, match="raw_sha256_invalid"):
        parse_gate_case({**base_case, "boundary": {**base_case["boundary"], "raw_sha256": "not_a_sha"}})
    with pytest.raises(ModelingError, match="measured_duration_must_be_positive"):
        parse_gate_case({**base_case, "measured_duration": 0})
    with pytest.raises(ModelingError, match="device_rank_invalid"):
        parse_gate_case({**base_case, "boundary": {**base_case["boundary"], "device_ranks": [0, 0]}})
    for bad_uri, err in (
        ("capture://bad/" + "/".join(("", "Users", "alice", "trace.json")), "private_host_path_rejected"),
        ("capture://bad/" + "/".join(("", "tmp", "run", "trace.json")), "private_host_path_rejected"),
        ("capture://10.20.30.40/trace", "private_ip_rejected"),
    ):
        with pytest.raises(ModelingError, match=err):
            sanitize_capture_uri(bad_uri)
    with pytest.raises(ModelingError, match="source_ref_relative_path_unsafe"):
        sanitize_source_ref({"repository": "r", "revision": "1", "relative_path": "../secret.py", "source_sha256": _sha("e")})
    assert ms.normalize_lineage_token("SHA256:" + _sha("a").upper()) == _sha("a")

def test_tail_diagnoser_moved_tail_unassessed_stages_and_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diag = _load_script("diagnose_tail_cluster", DIAG_SCRIPT)
    coh = _cohort("L1", "sparse_attn").to_dict()
    rdims = {"dtype_layout": "bf16_nd", "shape": "m64", "warmth": "cold", "stream": "s0"}
    row_a = {"case_id": "op_a", "level": "L1", "acquisition_root": "acq-1", "boundary_kind": "device_kernel",
             "measured_ns": 1000, "predicted_ns": 1000, "engine_receipt": "capture://receipt/L1/op_a",
             "signed_error": 0.0, "absolute_percentage_error": 0.0, "status": "ok", "cohort": coh, "regime_dims": rdims}
    row_b = {**row_a, "case_id": "op_b", "acquisition_root": "acq-2", "predicted_ns": 1000 + round(1000 * TAIL_TRACE_APE),
             "engine_receipt": "capture://receipt/L1/op_b", "signed_error": TAIL_TRACE_APE, "absolute_percentage_error": TAIL_TRACE_APE}
    ref_moved = [
        {"case_id": "op_a", "acquisition_root": "acq-rep-1", "boundary_kind": "device_kernel", "absolute_percentage_error": 0.18},
        {"case_id": "op_b", "acquisition_root": "acq-rep-2", "boundary_kind": "device_kernel", "absolute_percentage_error": 0.0},
    ]
    moved_rep = diag.diagnose_tail_clusters([row_a, row_b], reference_run=ref_moved, threshold_pct=10.0)
    assert moved_rep["status"] == "tail_warning"
    moved_cluster = moved_rep["clusters"][0]
    assert moved_cluster["blocking"] is False
    assert "persistent_tail_across_independent_repeat" not in moved_cluster["escalation_reasons"]
    stages_missing = {s["stage_number"]: s["status"] for s in moved_cluster["stages"]}
    assert stages_missing[1] == "issue_found"
    assert stages_missing[2] == "unassessed" and stages_missing[3] == "unassessed" and stages_missing[4] == "unassessed"
    assert stages_missing[5] == "action_required"
    ref_wrong_bnd = [{"case_id": "op_b", "acquisition_root": "acq-rep-2", "boundary_kind": "model_step", "absolute_percentage_error": 0.15}]
    assert "persistent_tail_across_independent_repeat" not in diag.diagnose_tail_clusters([row_a, row_b], reference_run=ref_wrong_bnd, threshold_pct=10.0)["clusters"][0]["escalation_reasons"]
    same_acq_rep = diag.diagnose_tail_clusters([row_a, row_b], reference_run=[row_a, row_b], threshold_pct=10.0)
    assert same_acq_rep["clusters"][0]["blocking"] is False
    assert "persistent_tail_across_independent_repeat" not in same_acq_rep["clusters"][0]["escalation_reasons"]

    lin_b = {"case_id": "op_b", "acquisition_root": "acq-2", "boundary_kind": "device_kernel", "capture_uri": "capture://s/b", "warmup_contaminated": True}
    rcpt_b = {
        "case_id": "op_b", "acquisition_root": "acq-2", "boundary_kind": "device_kernel", "engine_receipt": "capture://receipt/L1/op_b",
        "caller_node": "attn_call", "expected_abi": "abi_v2", "observed_abi": "abi_v1", "python_ns": 1180, "c_ns": 1120,
        "first_parity_diff_epoch": 4, "saturated_physical_row": "hbm_read", "counterfactual_ablation": "disable_l2_knee:-40ns",
        "exposed_wait_ns": 65, "silicon_mechanisms": ["STARS", "subtile_wave"],
    }
    ref_persist = [{"case_id": "op_b", "acquisition_root": "acq-rep-9", "boundary_kind": "device_kernel", "absolute_percentage_error": 0.15}]
    pers_rep = diag.diagnose_tail_clusters([row_a, row_b], lineage=[lin_b], engine_receipts=[rcpt_b], reference_run=ref_persist, threshold_pct=10.0)
    assert pers_rep["status"] == "blocking_escalation"
    assert "persistent_tail_across_independent_repeat" in pers_rep["clusters"][0]["escalation_reasons"]
    st = {s["stage_number"]: s for s in pers_rep["clusters"][0]["stages"]}
    assert all(st[n]["status"] == "issue_found" for n in (1, 2, 3, 4)) and st[5]["status"] == "action_required"

    res_f, lin_f, rcpt_f, ref_f, out_f = tmp_path / "r.json", tmp_path / "l.json", tmp_path / "e.json", tmp_path / "ref.json", tmp_path / "out.json"
    res_f.write_text(json.dumps({"residuals": [row_a, row_b]}), encoding="utf-8")
    lin_f.write_text(json.dumps([lin_b]), encoding="utf-8")
    rcpt_f.write_text(json.dumps([rcpt_b]), encoding="utf-8")
    ref_f.write_text(json.dumps([ref_persist[0]]), encoding="utf-8")
    monkeypatch.setattr(json, "load", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("whole-file json.load forbidden")))
    rc = diag.main(["--residuals", str(res_f), "--lineage", str(lin_f), "--engine-receipts", str(rcpt_f),
                    "--reference-run", str(ref_f), "--threshold-pct", "10", "--output", str(out_f)])
    assert rc == diag.EXIT_ESCALATED and out_f.is_file()

    bound = diag.ParameterBound("launch_overhead_ns", current_value=10.0, theta_min=0.0, theta_max=20.0, parameter_source="capture://train/launch")
    sens_conflict = {"launch_overhead_ns": [
        diag.CohortSensitivity("cohort_a", measured_ns=1000.0, predicted_ns=700.0, sensitivity_ns_per_unit=100.0),
        diag.CohortSensitivity("cohort_b", measured_ns=1000.0, predicted_ns=1300.0, sensitivity_ns_per_unit=100.0),
    ]}
    conflicts = diag.detect_parameter_conflicts([bound], sens_conflict, threshold_pct=10.0)
    assert len(conflicts) == 1 and conflicts[0].conflict_reason == "empty_cross_regime_intersection"
    sens_ok = {"launch_overhead_ns": [
        diag.CohortSensitivity("cohort_a", measured_ns=1000.0, predicted_ns=950.0, sensitivity_ns_per_unit=100.0),
        diag.CohortSensitivity("cohort_b", measured_ns=1000.0, predicted_ns=1020.0, sensitivity_ns_per_unit=100.0),
    ]}
    assert diag.detect_parameter_conflicts([bound], sens_ok, threshold_pct=10.0) == []

def test_producer_end_to_end_report_assembly_boundary_and_precision(tmp_path: Path) -> None:
    prod = _load_script("ver_producer_mod", PRODUCER_SCRIPT)
    assert prod._stats([])["status"] == "insufficient_evidence"
    assert prod._stats([0.09996])["status"] == "pass"
    assert prod._stats([0.0999999])["status"] == "pass"
    assert prod._stats([0.10])["status"] == "fail"
    sku = "Ascend950DT_9582"
    (tmp_path / "acquisition_split_manifest.json").write_text(json.dumps({
        "acquisitions": [{"acquisition_uid": uid, "split": split} for split, uids in (
            ("train", ("acq-l1-tr", "acq-m-tr", "acq-l2-tr")),
            ("heldout", ("acq-l1-ho", "acq-hotspot-run", "acq-d2-one", "acq-m-ho", "acq-l2-indep")),
        ) for uid in uids]
    }), encoding="utf-8")
    (tmp_path / "cann93_single_op_measurements.json").write_text(json.dumps({
        "single_op_measurements": {
            sku: [
                {"status": "ok", "acquisition_uid": "acq-l1-tr", "domain": "domain2_bf16_gemm", "op_name": "g_tr", "shape": [512, 1024, 1024], "metrics": {"rts_stream_saturated_us": 50.0}},
                {"status": "ok", "acquisition_uid": "acq-l1-ho", "domain": "domain2_bf16_gemm", "op_name": "g_ho", "shape": [1024, 1024, 1024], "metrics": {"rts_stream_saturated_us": 95.0}},
            ],
            "Ascend950PR_958b": [],
        },
        "collective_measurements": [],
    }), encoding="utf-8")
    (tmp_path / "cann93_seq512_to_400k_measurements.json").write_text(json.dumps({
        "acquisition_uid": "acq-d2-one",
        "records": [
            {"status": "ok", "domain": "shard0_fia_decode", "op_name": f"fia_s{seq}",
             "shape": {"B": 1, "S": 1, "Hq": 8, "Hkv": 2, "D": 128, "seq_kv": seq, "kv_dtype": "bf16", "swa_window": 0},
             "metrics": {"cold_p50_us": 25.0 + seq * 0.005}}
            for seq in (512, 2048)
        ],
    }), encoding="utf-8")
    (tmp_path / "task1_kimi_k31_silicon_hotspots.json").write_text(json.dumps({
        "acquisition_uid": "acq-hotspot-run",
        "records": [
            {"status": "ok", "group": "Roofline_Cube", "op_name": f"bf16_m{i}", "shape": {"M": 512, "N": 1024, "K": 1024}, "metrics": {"cold_p50_us": 50.0}}
            for i in range(3)
        ],
    }), encoding="utf-8")
    (tmp_path / "cann93_model_subgraphs_and_perf.json").write_text(json.dumps({
        "models": {"qwen3.8-27b": {"runs": {
            "r_ho": {**MODEL_RUN_CAPTURE, "sku": sku, "run_id": "r_ho", "acquisition_uid": "acq-m-ho", "l2_acquisition_uid": "acq-l2-indep",
                     "independent_l2_capture_uri": "capture://indep-l2/layer0", "independent_l2_measured_us": INDEPENDENT_L2_CAPTURE_US},
            "r_no_l2": {**MODEL_RUN_CAPTURE, "sku": sku, "run_id": "r_no_l2", "acquisition_uid": "acq-m-ho", "l2_acquisition_uid": "acq-l2-tr"},
        }}}
    }), encoding="utf-8")
    report = prod.build_report(tmp_path, workers=1)
    exec_keys = sorted(k for k in report["source_sha256"] if k.startswith("mindie_sim/prediction/execution/"))
    assert len(exec_keys) >= 8 and "mindie_sim/prediction/execution.py" not in report["source_sha256"]
    assert report["disjoint_split_verification"]["acquisition_root_overlap"] == 0
    assert len([r for r in report["level1_operators"]["train"]["sample_records"] if "kimi_" in r["id"] or "fia_s" in r["id"]]) == 0
    d2_held = [r for r in report["level1_operators"]["heldout"]["sample_records"] if "fia_s" in r["id"]]
    assert len(d2_held) == 2 and len({r["acquisition_root"] for r in d2_held}) == 1
    l2_ho = report["level2_subgraphs_and_blocks"]["heldout"]["records"]
    assert len(l2_ho) == 1 and l2_ho[0]["run_id"] == "r_ho" and l2_ho[0]["status"] == "ok" and l2_ho[0]["l2_layers"] == 1
    assert l2_ho[0]["measured_block_us"] == 180.0 and l2_ho[0]["acquisition_root"] == "acq-l2-indep"
    l2_tr = [r for r in report["level2_subgraphs_and_blocks"]["train"]["records"] if r["run_id"] == "r_no_l2"]
    assert len(l2_tr) == 1 and l2_tr[0]["status"] == "missing_evidence" and l2_tr[0]["ape_pct"] is None
    bad_manifest = tmp_path / "bad_split.json"
    bad_manifest.write_text(json.dumps({"acquisitions": [
        {"acquisition_uid": "acq-x", "raw_sha256": _sha("a"), "split": "train"},
        {"acquisition_uid": "acq-y", "raw_sha256": _sha("a").upper(), "split": "heldout"},
    ]}), encoding="utf-8")
    with pytest.raises(ModelingError, match="producer_manifest_split_conflict"):
        prod.build_report(tmp_path, split_manifest=bad_manifest, workers=1)
