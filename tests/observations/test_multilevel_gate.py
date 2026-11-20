"""Unit tests for independent multilevel accuracy gate and tail-cluster diagnosis.

[Role]
  Verify transitive acquisition-lineage closure, split/fit/cross-level isolation,
  Engine receipt enforcement, near-zero rung deltas, strict sub-10% cohort
  summaries, streaming JSON/JSONL manifest validation, 5-stage tail-cluster
  diagnosis, and cross-regime Jacobian conflict detection.
[Invariants]
  * Pure tier-0 in-memory and tmp_path execution; zero subprocesses, zero
    network calls, zero device contexts, zero environment mutations.
  * Every failure path raises structured ModelingError or emits explicit
    insufficient_evidence / missing_evidence / invalid_evidence statuses.
  * Exact 10.0% error fails the strict < 10.0% gate; empty cohorts never
    report 0.0% error.
[Timeouts & Locks]
  Deterministic in-process execution (< 2s); zero locks or background workers.
[Side-Effect Policy]
  Writes temporary test manifests/reports only inside pytest tmp_path.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import pytest

from mindie_sim.observations.multilevel_gate import (
    CaptureBoundary,
    CohortKey,
    EngineInterval,
    FrozenFitManifest,
    FrozenFitParameter,
    GateCase,
    Manifest,
    Residual,
    evaluate,
    gate_decision,
    run_l1,
    run_l2,
    run_l3,
    sanitize_capture_uri,
    sanitize_source_ref,
    stream_jsonl,
    stream_manifest,
    summarize_cohorts,
    validate_lineage,
)
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_SCRIPT = REPO_ROOT / "scripts" / "diagnose_tail_cluster.py"
PRODUCER_SCRIPT = REPO_ROOT / "data" / "verification" / "a5-cann93-multilevel-accuracy-20261120.producer.py"


def _load_diag_module():
    if "diagnose_tail_cluster" in sys.modules:
        return sys.modules["diagnose_tail_cluster"]
    spec = importlib.util.spec_from_file_location("diagnose_tail_cluster", DIAG_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(tag: str) -> str:
    return (tag * 64)[:64]


def _cohort(level: str = "L1", name: str = "gemm") -> CohortKey:
    return CohortKey(
        level=level, cohort_name=name, sku="device_x", phase="prefill",  # type: ignore[arg-type]
        mode="eager", shape_regime="medium", parallel_topology="tp1",
    )


def _boundary(
    tag: str, *, kind: str = "device_kernel", parents: tuple[str, ...] = (),
    events: tuple[str, ...] = (), acq: str | None = None, run: str | None = None,
    unit: str = "ns", uri: str = "",
) -> CaptureBoundary:
    return CaptureBoundary(
        raw_sha256=_sha(tag), acquisition_uid=acq or f"acq-{tag}", run_uid=run or f"run-{tag}",
        parent_digests=parents, event_ids=events or (f"ev-{tag}",), device_ranks=(0,),
        clock_domain="device_monotonic", start_event="k_start", end_event="k_end",
        boundary_kind=kind, unit=unit, capture_uri=uri or f"capture://suite/{tag}",
    )


def _case(
    cid: str, *, level: str = "L1", split: str = "heldout", tag: str = "a",
    kind: str | None = None, measured_ns: int | None = 1000, parents: tuple[str, ...] = (),
    events: tuple[str, ...] = (), acq: str | None = None, run: str | None = None,
    debug_cross_level: bool = False, is_counterfactual: bool = False,
    has_matched_capture: bool = True, block_capture_present: bool = True,
    arm_a: int | None = None, arm_b: int | None = None, cohort_name: str = "gemm",
) -> GateCase:
    default_kind = {"L1": "device_kernel", "L2": "fx_region", "L3": "model_step"}[level]
    return GateCase(
        case_id=cid, level=level, split=split,  # type: ignore[arg-type]
        boundary=_boundary(tag, kind=kind or default_kind, parents=parents, events=events, acq=acq, run=run),
        input_digest=f"in-{cid}", stack_digest="stack-1", implementation="canonical_impl",
        cohort=_cohort(level, cohort_name), measured_ns=measured_ns,
        debug_cross_level=debug_cross_level, is_counterfactual=is_counterfactual,
        has_matched_capture=has_matched_capture, block_capture_present=block_capture_present,
        rung_arm_a_measured_ns=arm_a, rung_arm_b_measured_ns=arm_b,
        regime_dims=(("dtype_layout", "bf16_nd"), ("shape", "m1024"), ("warmth", "warm")),
    )


class _StubRunner:
    def __init__(self, overrides: dict[str, EngineInterval] | None = None, scale: float = 1.05) -> None:
        self._overrides = overrides or {}
        self._scale = scale

    def _default(self, case: GateCase) -> EngineInterval:
        if case.case_id in self._overrides:
            return self._overrides[case.case_id]
        if case.boundary.boundary_kind == "paired_rung_delta":
            assert case.rung_arm_a_measured_ns is not None and case.rung_arm_b_measured_ns is not None
            pa = int(round(case.rung_arm_a_measured_ns * self._scale))
            pb = int(round(case.rung_arm_b_measured_ns * self._scale))
            return EngineInterval(
                case_id=case.case_id, level="L2", boundary_kind="paired_rung_delta",
                start_event=case.boundary.start_event, end_event=case.boundary.end_event,
                start_ns=0, end_ns=pb - pa, predicted_ns=pb - pa,
                engine_receipt=f"capture://receipt/{case.level}/{case.case_id}",
                arm_a_predicted_ns=pa, arm_b_predicted_ns=pb,
            )
        pred = max(1, int(round((case.measured_ns or 1000) * self._scale)))
        return EngineInterval(
            case_id=case.case_id, level=case.level, boundary_kind=case.boundary.boundary_kind,
            start_event=case.boundary.start_event, end_event=case.boundary.end_event,
            start_ns=100, end_ns=100 + pred, predicted_ns=pred,
            engine_receipt=f"capture://receipt/{case.level}/{case.case_id}",
        )

    def run_l1(self, case: GateCase) -> EngineInterval:
        return self._default(case)

    def run_l2(self, case: GateCase) -> EngineInterval:
        return self._default(case)

    def run_l3(self, case: GateCase) -> EngineInterval:
        return self._default(case)


def test_transitive_lineage_closure_rejects_split_leakage() -> None:
    c_train = _case("c_train", split="train", tag="1")
    c_held = _case("c_held", split="heldout", tag="2", parents=("intermediate_digest",))
    manifest = Manifest(
        manifest_id="m1", schema_version="1", cases=(c_train, c_held),
        lineage_edges=((_sha("1"), "intermediate_digest"),),
    )
    fits = FrozenFitManifest("f1", "1", ())
    with pytest.raises(ModelingError, match="lineage_split_leakage"):
        validate_lineage(manifest, fits)

    c_dup = _case("c_train", split="train", tag="3")
    with pytest.raises(ModelingError, match="duplicate_case_id"):
        validate_lineage(Manifest("m2", "1", (c_train, c_dup)), fits)


def test_fit_ancestor_heldout_intersection_rejected() -> None:
    c_train = _case("t1", split="train", tag="1")
    c_held = _case("h1", split="heldout", tag="2", parents=("raw_parent_2",))
    manifest = Manifest("m1", "1", (c_train, c_held))
    leaking_fits = FrozenFitManifest(
        fit_id="fit_leak", schema_version="1",
        parameters=(FrozenFitParameter("p_bw", ("fit_ancestor_x",), "capture://fits/bw"),),
        lineage_edges=(("fit_ancestor_x", "raw_parent_2"),),
    )
    with pytest.raises(ModelingError, match="lineage_fit_heldout_leakage"):
        validate_lineage(manifest, leaking_fits)

    clean_fits = FrozenFitManifest(
        fit_id="fit_clean", schema_version="1",
        parameters=(FrozenFitParameter("p_bw", (_sha("1"),), "capture://fits/bw"),),
    )
    idx = validate_lineage(manifest, clean_fits)
    assert idx.root_for("t1") in idx.fit_ancestor_components["p_bw"]
    assert not (idx.fit_ancestor_components["p_bw"] & idx.heldout_components)


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
    residuals = list(evaluate(manifest, fits, _StubRunner(scale=1.04)))
    by_id = {r.case_id: r for r in residuals}
    assert by_id["l1_case"].status == "ok"
    assert by_id["l2_dbg"].status == "debug_only"
    summaries = summarize_cohorts(residuals)
    assert summaries[c_l2_dbg.cohort].status == "insufficient_evidence"
    assert ("debug_cross_level_not_blind_acceptance", 1) in summaries[c_l2_dbg.cohort].excluded_reasons


def test_engine_receipt_rejects_external_offset_and_l3_clone() -> None:
    c_l1 = _case("l1_off", level="L1", tag="1", measured_ns=1000)
    bad_offset = EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "capture://r/1", external_offset_ns=4200)
    with pytest.raises(ModelingError, match="non_engine_or_offset_rejected"):
        run_l1(c_l1, _StubRunner({"l1_off": bad_offset}))

    non_engine = EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "capture://r/1", engine_sourced=False)
    with pytest.raises(ModelingError, match="non_engine_or_offset_rejected"):
        run_l1(c_l1, _StubRunner({"l1_off": non_engine}))

    empty_rcpt = EngineInterval("l1_off", "L1", "device_kernel", "k_start", "k_end", 0, 1050, 1050, "")
    with pytest.raises(ModelingError, match="engine_receipt_missing"):
        run_l1(c_l1, _StubRunner({"l1_off": empty_rcpt}))

    c_l2 = _case("l2_clone", level="L2", tag="2", measured_ns=5000)
    div_interval = EngineInterval("l2_clone", "L2", "fx_region", "k_start", "k_end", 0, 5000, 5000, "capture://r/2", derived_from_step_division=True)
    with pytest.raises(ModelingError, match="l2_cloned_or_divided_from_l3_forbidden"):
        run_l2(c_l2, _StubRunner({"l2_clone": div_interval}))

    l3_src = EngineInterval("l2_clone", "L2", "fx_region", "k_start", "k_end", 0, 5000, 5000, "capture://r/2", source_level="L3")
    with pytest.raises(ModelingError, match="l2_cloned_or_divided_from_l3_forbidden"):
        run_l2(c_l2, _StubRunner({"l2_clone": l3_src}))

    c_l3 = _case("l3_step", level="L3", tag="3", measured_ns=50000)
    dup_l2 = EngineInterval("l2_clone", "L2", "fx_region", "k_start", "k_end", 0, 5000, 5000, "capture://r/same")
    dup_l3 = EngineInterval("l3_step", "L3", "model_step", "k_start", "k_end", 0, 50000, 50000, "capture://r/same")
    with pytest.raises(ModelingError, match="cross_level_engine_receipt_clone"):
        list(evaluate(Manifest("m_rcpt", "1", (c_l2, c_l3)), FrozenFitManifest("f", "1", ()), _StubRunner({"l2_clone": dup_l2, "l3_step": dup_l3})))


def test_l2_absent_block_capture_emits_missing_evidence() -> None:
    c_missing = _case("l2_no_blk", level="L2", tag="4", measured_ns=4000, block_capture_present=False)
    interval = run_l2(c_missing, _StubRunner())
    assert interval.status == "missing_evidence"
    assert interval.missing_reason == "block_capture_absent"

    residuals = list(evaluate(Manifest("m_miss", "1", (c_missing,)), FrozenFitManifest("f", "1", ()), _StubRunner()))
    assert len(residuals) == 1
    assert residuals[0].status == "missing_evidence"
    assert math.isnan(residuals[0].signed_error)
    summary = summarize_cohorts(residuals)[c_missing.cohort]
    assert summary.status == "insufficient_evidence"
    assert summary.missing == 1
    assert summary.mape_pct is None


def test_near_zero_rung_delta_reports_signed_ns_and_arm_errors() -> None:
    near = _case("rung_near", level="L2", tag="5", kind="paired_rung_delta", measured_ns=200, arm_a=100_000, arm_b=100_200)
    wide = _case("rung_wide", level="L2", tag="6", kind="paired_rung_delta", measured_ns=40_000, arm_a=100_000, arm_b=140_000)
    overrides = {
        "rung_near": EngineInterval("rung_near", "L2", "paired_rung_delta", "k_start", "k_end", 0, 350, 350, "capture://r/near", arm_a_predicted_ns=101_000, arm_b_predicted_ns=101_350),
        "rung_wide": EngineInterval("rung_wide", "L2", "paired_rung_delta", "k_start", "k_end", 0, 42_000, 42_000, "capture://r/wide", arm_a_predicted_ns=102_000, arm_b_predicted_ns=144_000),
    }
    residuals = {r.case_id: r for r in evaluate(Manifest("m_rung", "1", (near, wide)), FrozenFitManifest("f", "1", ()), _StubRunner(overrides))}
    r_near, r_wide = residuals["rung_near"], residuals["rung_wide"]
    assert r_near.status == "near_zero_delta"
    assert math.isnan(r_near.absolute_percentage_error)
    assert r_near.signed_delta_ns == 150
    assert r_near.arm_a_signed_error == pytest.approx(0.01)
    assert r_near.arm_b_signed_error == pytest.approx((101_350 - 100_200) / 100_200)
    assert r_wide.status == "ok"
    assert r_wide.absolute_percentage_error == pytest.approx(0.05)


def test_empty_cohort_zero_duration_and_counterfactual_handling() -> None:
    assert summarize_cohorts([]) == {}
    assert gate_decision({})["status"] == "insufficient_evidence"

    c_zero = _case("zero_dur", level="L1", tag="7", measured_ns=0)
    c_cf = _case("cf_only", level="L3", tag="8", measured_ns=9000, is_counterfactual=True, has_matched_capture=False)
    residuals = list(evaluate(Manifest("m_inv", "1", (c_zero, c_cf)), FrozenFitManifest("f", "1", ()), _StubRunner()))
    by_id = {r.case_id: r for r in residuals}
    assert by_id["zero_dur"].status == "invalid_evidence"
    assert by_id["cf_only"].status == "counterfactual_excluded"
    summaries = summarize_cohorts(residuals)
    assert summaries[c_zero.cohort].status == "insufficient_evidence"
    assert gate_decision(summaries)["status"] == "insufficient_evidence"


def test_strict_threshold_boundary_fails_at_exact_ten_percent() -> None:
    c_exact = _case("exact_10", level="L1", tag="1", measured_ns=1000, cohort_name="boundary_fam")
    c_pass = _case("below_10", level="L1", tag="2", measured_ns=1000, cohort_name="pass_fam")
    overrides = {
        "exact_10": EngineInterval("exact_10", "L1", "device_kernel", "k_start", "k_end", 0, 1100, 1100, "capture://r/10"),
        "below_10": EngineInterval("below_10", "L1", "device_kernel", "k_start", "k_end", 0, 1099, 1099, "capture://r/99"),
    }
    res = list(evaluate(Manifest("m_thr", "1", (c_exact, c_pass)), FrozenFitManifest("f", "1", ()), _StubRunner(overrides)))
    summaries = summarize_cohorts(res, threshold_pct=10.0)
    exact_sum = summaries[c_exact.cohort]
    pass_sum = summaries[c_pass.cohort]
    assert exact_sum.mape_pct == pytest.approx(10.0)
    assert exact_sum.status == "fail"
    assert not exact_sum.passed
    assert "mape_pct" in exact_sum.failed_metrics and "medape_pct" in exact_sum.failed_metrics
    assert pass_sum.status == "pass" and pass_sum.passed
    assert gate_decision(summaries)["status"] == "fail"

    # Nearest-rank P90 across 10 independent points picks index 8 (9th sorted value)
    ten_res = [
        Residual(f"p{i}", "L1", _cohort("L1", "p90_fam"), f"root-{i}", 1000, 1000 + err, f"capture://r/{i}", err / 1000.0, err / 1000.0)
        for i, err in enumerate((10, 20, 30, 40, 50, 60, 70, 80, 90, 250))
    ]
    p90_sum = summarize_cohorts(ten_res)[_cohort("L1", "p90_fam")]
    assert p90_sum.p90_ape_pct == pytest.approx(9.0)
    assert p90_sum.max_ape_pct == pytest.approx(25.0)
    assert p90_sum.tail_exceeded and p90_sum.tail_case_ids == ("p9",)
    assert p90_sum.n_roots == 10 and p90_sum.n_rows == 10


def test_stream_manifest_json_jsonl_and_provenance_sanitization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mindie_sim.observations.multilevel_gate as mg

    base_case = {
        "case_id": "op_stream_1", "level": "L1", "split": "heldout",
        "input_digest": "in1", "stack_digest": "st1", "implementation": "impl1",
        "measured_duration": 12.5,
        "boundary": {
            "raw_sha256": _sha("c"), "acquisition_uid": "acq-c", "run_uid": "run-c",
            "parent_digests": [], "event_ids": ["e1"], "device_ranks": [0, 1],
            "clock_domain": "device_ns", "start_event": "s", "end_event": "e",
            "boundary_kind": "device_kernel", "unit": "us",
            "capture_uri": "capture://campaign/op_stream_1",
            "source_ref": {
                "repository": "mindie-agent/sim", "revision": "a" * 40,
                "relative_path": "mindie_sim/prediction/execution.py", "source_sha256": _sha("d"),
            },
        },
        "cohort": {
            "cohort_name": "gemm", "sku": "device_x", "phase": "prefill",
            "mode": "eager", "shape_regime": "medium", "parallel_topology": "tp2",
        },
    }
    jsonl_path = tmp_path / "manifest.jsonl"
    jsonl_path.write_text(
        json.dumps({"record_type": "manifest_header", "manifest_id": "stream_m", "source_uri": "capture://campaign/m"}) + "\n"
        + json.dumps({"record_type": "lineage_edge", "parent": "p_root", "child": "acq-c"}) + "\n"
        + json.dumps(base_case) + "\n",
        encoding="utf-8",
    )
    assert len(list(stream_jsonl(jsonl_path))) == 3
    m_jsonl = stream_manifest(jsonl_path)
    assert m_jsonl.manifest_id == "stream_m"
    assert m_jsonl.cases[0].measured_ns == 12_500
    assert m_jsonl.lineage_edges == (("p_root", "acq-c"),)

    json_path = tmp_path / "manifest.json"
    json_path.write_text(json.dumps({"manifest_id": "big_m", "cases": [base_case]}), encoding="utf-8")
    monkeypatch.setattr(mg, "ijson", None)
    m_streamed = stream_manifest(json_path, max_whole_read_bytes=1)
    assert m_streamed.manifest_id == "big_m" and len(m_streamed.cases) == 1

    bad_sha_case = {**base_case, "boundary": {**base_case["boundary"], "raw_sha256": "not_a_sha"}}
    with pytest.raises(ModelingError, match="raw_sha256_invalid"):
        mg.parse_gate_case(bad_sha_case)
    bad_dur_case = {**base_case, "measured_duration": 0}
    with pytest.raises(ModelingError, match="measured_duration_must_be_positive"):
        mg.parse_gate_case(bad_dur_case)
    bad_rank_case = {**base_case, "boundary": {**base_case["boundary"], "device_ranks": [0, 0]}}
    with pytest.raises(ModelingError, match="device_rank_invalid"):
        mg.parse_gate_case(bad_rank_case)

    private_unix = "/".join(("", "Users", "alice", "trace.json"))
    with pytest.raises(ModelingError, match="private_host_path_rejected"):
        sanitize_capture_uri(f"capture://bad/{private_unix}")
    with pytest.raises(ModelingError, match="private_ip_rejected"):
        sanitize_capture_uri("capture://10.20.30.40/trace")
    with pytest.raises(ModelingError, match="source_ref_relative_path_unsafe"):
        sanitize_source_ref({"repository": "r", "revision": "1", "relative_path": "../secret.py", "source_sha256": _sha("e")})


def test_tail_cluster_five_stage_ladder_and_persistence_escalation(tmp_path: Path) -> None:
    diag = _load_diag_module()
    res_row = {
        "case_id": "tail_op_1", "level": "L1", "acquisition_root": "acq-tail-1",
        "boundary_kind": "device_kernel", "measured_ns": 1000, "predicted_ns": 1180,
        "engine_receipt": "capture://receipt/L1/tail_op_1",
        "signed_error": 0.18, "absolute_percentage_error": 0.18, "status": "ok",
        "cohort": _cohort("L1", "sparse_attn").to_dict(),
        "regime_dims": {"dtype_layout": "bf16_nd", "shape": "m64", "warmth": "cold", "stream": "s0"},
    }
    lin_row = {
        "case_id": "tail_op_1", "acquisition_root": "acq-tail-1", "boundary_kind": "device_kernel",
        "capture_uri": "capture://suite/tail_op_1", "warmup_contaminated": True,
    }
    rcpt_row = {
        "case_id": "tail_op_1", "acquisition_root": "acq-tail-1", "boundary_kind": "device_kernel",
        "engine_receipt": "capture://receipt/L1/tail_op_1",
        "caller_node": "attn_call", "expected_abi": "abi_v2", "observed_abi": "abi_v1",
        "python_ns": 1180, "c_ns": 1120, "first_parity_diff_epoch": 4,
        "saturated_physical_row": "hbm_read", "counterfactual_ablation": "disable_l2_knee:-40ns",
        "exposed_wait_ns": 65, "silicon_mechanisms": ["STARS", "subtile_wave"],
    }
    ref_row = {
        "case_id": "tail_op_1", "acquisition_root": "acq-tail-1", "boundary_kind": "device_kernel",
        "absolute_percentage_error": 0.15,
    }
    report = diag.diagnose_tail_clusters(
        [res_row], lineage=[lin_row], engine_receipts=[rcpt_row], reference_run=[ref_row], threshold_pct=10.0,
    )
    assert report["status"] == "blocking_escalation"
    assert report["cluster_count"] == 1
    cluster = report["clusters"][0]
    assert cluster["blocking"] is True
    assert "persistent_tail_across_independent_repeat" in cluster["escalation_reasons"]
    assert cluster["error_direction"] == "overpredict"
    assert cluster["first_abi_diff"] == "caller=attn_call:expected=abi_v2:observed=abi_v1"
    assert cluster["saturated_physical_row"] == "hbm_read"
    assert cluster["exposed_wait_ns"] == 65
    stages = {s["stage_number"]: s for s in cluster["stages"]}
    assert stages[1]["status"] == "issue_found" and "warmup_contaminated:tail_op_1" in stages[1]["findings"]
    assert stages[2]["status"] == "issue_found" and any("abi_mismatch" in f for f in stages[2]["findings"])
    assert stages[3]["status"] == "issue_found" and any("python_c_parity_mismatch" in f for f in stages[3]["findings"])
    assert stages[4]["status"] == "issue_found" and any("STARS" in f for f in stages[4]["findings"])
    assert stages[5]["status"] == "action_required" and len(stages[5]["findings"]) == 6

    res_f, lin_f, rcpt_f, out_f = tmp_path / "r.jsonl", tmp_path / "l.json", tmp_path / "e.json", tmp_path / "out.json"
    res_f.write_text(json.dumps(res_row) + "\n", encoding="utf-8")
    lin_f.write_text(json.dumps([lin_row]), encoding="utf-8")
    rcpt_f.write_text(json.dumps([rcpt_row]), encoding="utf-8")
    rc = diag.main(["--residuals", str(res_f), "--lineage", str(lin_f), "--engine-receipts", str(rcpt_f), "--threshold-pct", "10", "--output", str(out_f)])
    assert rc == diag.EXIT_ESCALATED and out_f.is_file()


def test_cross_regime_jacobian_conflict_empty_intersection() -> None:
    diag = _load_diag_module()
    bound = diag.ParameterBound("launch_overhead_ns", current_value=10.0, theta_min=0.0, theta_max=20.0, parameter_source="capture://train/launch")
    # Cohort A needs delta_theta in (2.0, 4.0); Cohort B needs delta_theta in (-4.0, -2.0) -> empty intersection!
    sens_conflict = {
        "launch_overhead_ns": [
            diag.CohortSensitivity("cohort_a", measured_ns=1000.0, predicted_ns=700.0, sensitivity_ns_per_unit=100.0),
            diag.CohortSensitivity("cohort_b", measured_ns=1000.0, predicted_ns=1300.0, sensitivity_ns_per_unit=100.0),
        ]
    }
    conflicts = diag.detect_parameter_conflicts([bound], sens_conflict, threshold_pct=10.0)
    assert len(conflicts) == 1
    assert set(conflicts[0].conflicting_cohorts) == {"cohort_a", "cohort_b"}
    assert conflicts[0].parameter_source == "capture://train/launch"
    assert conflicts[0].conflict_reason == "empty_cross_regime_intersection"
    assert conflicts[0].read_only_notice == "diagnostic_only_profile_unmodified"

    # Strict open boundary touch: A needs (0.0, 2.0), B needs (-2.0, 0.0) -> empty open intersection
    sens_touch = {
        "launch_overhead_ns": [
            diag.CohortSensitivity("cohort_a", measured_ns=1000.0, predicted_ns=900.0, sensitivity_ns_per_unit=100.0),
            diag.CohortSensitivity("cohort_b", measured_ns=1000.0, predicted_ns=1100.0, sensitivity_ns_per_unit=100.0),
        ]
    }
    assert len(diag.detect_parameter_conflicts([bound], sens_touch, threshold_pct=10.0)) == 1

    # Overlapping feasible intervals inside bounds -> zero conflicts
    sens_ok = {
        "launch_overhead_ns": [
            diag.CohortSensitivity("cohort_a", measured_ns=1000.0, predicted_ns=950.0, sensitivity_ns_per_unit=100.0),
            diag.CohortSensitivity("cohort_b", measured_ns=1000.0, predicted_ns=1020.0, sensitivity_ns_per_unit=100.0),
        ]
    }
    assert diag.detect_parameter_conflicts([bound], sens_ok, threshold_pct=10.0) == []

    # Negative Jacobian & out-of-bounds intersection -> conflict escalates even when overall residuals pass
    tight_bound = diag.ParameterBound("link_bw_gbps", current_value=50.0, theta_min=49.0, theta_max=51.0, parameter_source="capture://train/bw")
    sens_neg = {
        "link_bw_gbps": [
            diag.CohortSensitivity("cohort_small", measured_ns=1000.0, predicted_ns=1400.0, sensitivity_ns_per_unit=-100.0),
        ]
    }
    oob = diag.detect_parameter_conflicts([tight_bound], sens_neg, threshold_pct=10.0)
    assert len(oob) == 1 and oob[0].conflict_reason == "infeasible_within_training_bounds"

    pass_row = {
        "case_id": "ok_1", "level": "L1", "acquisition_root": "acq-ok-1",
        "boundary_kind": "device_kernel", "measured_ns": 1000, "predicted_ns": 1020,
        "engine_receipt": "capture://receipt/L1/ok_1", "signed_error": 0.02,
        "absolute_percentage_error": 0.02, "status": "ok",
        "cohort": _cohort("L1", "cohort_a").to_dict(),
    }
    escalated = diag.diagnose_tail_clusters([pass_row], parameter_bounds=[bound], sensitivities_by_param=sens_conflict, threshold_pct=10.0)
    assert escalated["status"] == "blocking_escalation"
    assert escalated["conflict_count"] == 1


def test_l3_runner_contract_validation() -> None:
    c_l3_ok = _case("l3_ok", level="L3", tag="9", measured_ns=20_000)
    interval = run_l3(c_l3_ok, _StubRunner(scale=1.02))
    assert interval.level == "L3" and interval.predicted_ns == 20_400

    c_l3_bad_boundary = _case("l3_bad", level="L3", tag="a", kind="device_kernel", measured_ns=20_000)
    with pytest.raises(ModelingError, match="run_l3_contract_invalid"):
        run_l3(c_l3_bad_boundary, _StubRunner())


def test_verification_producer_removes_offsets_clones_and_empty_zero_error() -> None:
    source = PRODUCER_SCRIPT.read_text(encoding="utf-8")
    assert "\r\n" not in source
    assert "UB_MESH_PAIR_BW_GBPS" not in source
    assert "+ 4.2" not in source
    assert 'raw["runtime"]["submission"]' not in source
    assert "max(1, pcp) * max(1, dcp)" not in source
    assert "def _eval_collective(" in source
    assert "missing_evidence" in source
    tree = ast.parse(source)
    stats_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_stats")
    assert stats_fn is not None

    spec = importlib.util.spec_from_file_location("ver_producer_mod", PRODUCER_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    empty = mod._stats([])
    assert empty["status"] == "insufficient_evidence"
    assert empty["count"] == 0 and empty["mape_pct"] is None
    assert mod._stats([0.10])["status"] == "fail"
    assert mod._stats([0.099])["status"] == "pass"
