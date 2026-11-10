"""Heldout NPU body accuracy and deterministic composite physical validation.

[Role]
  Keep two evidence classes separate:
  * heldout real-NPU captures: load checked-in case inputs, run live body
    prediction, and compare with the retained ``kernel_details`` p50 latency;
  * deterministic 2-layer composite matrix: validate physical roofline,
    operator conservation, and retained regression references for TP x DP x EP
    x quant, MC2/shared-expert DP, PP/PCP/EP/SP, and MTP/DSpark.

[Invariants]
  Heldout rows are never in-sample calibration rows and stay within 10% of the
  measured kernel p50. Composite rows are synthetic physical references, not
  NPU captures: the body step lies between the analytic roofline and the
  operator sum, is no faster than its HBM floor, and conserves operator
  duration, HBM, and HCCL bytes. MC2 stream overlap makes the operator sum
  exceed the graph duration. MTP/DSpark automatically prices ``1 + proposal``
  verify rows and lowers effective TPOT when acceptance exceeds one.

[Timeouts & Locks]
  Tier 0 in-memory body scope only. No device locks. Module budget is under 60s;
  the retained full-checkpoint heldout rows dominate the cost.

[Side-Effect Policy]
  The autouse fixture confines HOME and MindIE homes to pytest tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim import predict
from mindie_sim.catalog.weights import cache_row_bytes_by_layer, weight_breakdown
from mindie_sim.compose import compose_case
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency, extract_hardware_peaks,
    solve_hardware_sla_throughput,
)
from mindie_sim.prediction.scheduler import DraftPolicy
from mindie_sim.prediction.structure import parse_case

REL_TOL = 0.10
ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "data/verification/decode-body-current-20261030.json"
CASE_ROOT = ROOT / "examples/cases/high-concurrency-20261030"

# Real heldout captures retained by the decode-body scoreboard. The case JSON
# is the full checked-in input; the scoreboard row supplies the measured
# kernel_details p50 and the exact captured query/context/sequence geometry.
# ``evaluation_relation`` is resolved from the case when declared. The compact
# scoreboard projection drops that field for these rows; missing provenance is
# reported as ``unknown`` and each row's external capture path is checked
# separately instead of being inferred to be heldout.
HELDOUT_CASES = (
    ("v41-new6/a3_v41_w8a8_tp4_dp4_ep16_dspark7_c32.json",
     "a3_v41_w8a8_tp4_dp4_ep16_dspark7_c32:q8_b9_m72"),
    ("glm-source-accepted-26/glm53flash-a3-tp16-c4.json",
     "glm53flash-a3-tp16-c4:q6_b4_m24"),
    ("glm-source-accepted-26/a3_glm53flash_tp16_aclgraph_c16.json",
     "a3_glm53flash_tp16_aclgraph_c16:q6_b16_m96"),
)

# Deterministic 2-layer composite physical regression rows.
# name, model, hardware, quant, sku, tp, dp, ep, rpn, layers, experts, phase,
# tokens, ctx, seq, reference_ms
# ``reference_ms`` is the current physical-model body step, retained only as a
# regression pin for the roofline/conservation decomposition. It is not an NPU
# capture, not a calibration target, and not evidence of measured accuracy.
CAPTURES = (
    ("a2-qwen35-w8a8-ep4-sn", "qwen3.5-35b-a3b", "a2", "w8a8", "Ascend910B4", 2, 2, 4, None, 2, 8, "decode", 1, 128, 1, 0.31035205),
    ("a3-9362-qwen35-w8a8-ep8-sn", "qwen3.5-35b-a3b", "Ascend910_9362", "w8a8", "Ascend910_9362", 4, 2, 8, None, 2, 8, "decode", 1, 256, 1, 0.36884391),
    ("a3-9392-glm53-w4a8-ep2-sn", "glm-5.3", "Ascend910_9392", "w4a8", "Ascend910_9392", 2, 1, 2, None, 4, 8, "decode", 1, 128, 1, 1.07346148),
    ("a5-9582-qwen35-fp8-prefill", "qwen3.5-35b-a3b", "Ascend950DT_9582", "fp8", "Ascend950DT_9582", 2, 1, 2, None, 2, 8, "prefill", 128, 0, 1, 0.19767048),
    ("a5-9599-qwen397-w4a8", "qwen3.5-397b-a17b", "Ascend910_9599", "w4a8", "Ascend910_9599", 2, 1, 2, None, 2, 8, "decode", 1, 128, 1, 0.16130665),
    ("a2-dsv32-w8a8-ep4", "deepseek-v3.2", "a2", "w8a8", "Ascend910B4", 2, 2, 4, None, 4, 8, "decode", 1, 128, 1, 2.10359074),
    ("a5-v41-fp8-prefill", "deepseek-v4.1", "Ascend950DT_9582", "fp8", "Ascend950DT_9582", 2, 1, 2, None, 4, 8, "prefill", 128, 0, 1, 2.33075143),
    ("a3-9362-ep16-sn", "qwen3.5-35b-a3b", "Ascend910_9362", "w8a8", "Ascend910_9362", 4, 4, 16, 16, 2, 16, "decode", 1, 64, 1, 0.44884783),
    ("a2-ep16-mn", "qwen3.5-35b-a3b", "a2", "w8a8", "Ascend910B4", 4, 4, 16, 8, 2, 16, "decode", 1, 64, 1, 0.60479281),
    ("a5-9582-ep32-mn-fp8", "qwen3.5-397b-a17b", "Ascend950DT_9582", "fp8", "Ascend950DT_9582", 4, 8, 32, 8, 2, 32, "decode", 1, 64, 1, 0.80277198),
    ("a5-9599-ep64-mn", "qwen3.5-35b-a3b", "Ascend910_9599", "w8a8", "Ascend910_9599", 8, 8, 64, 8, 2, 64, "prefill", 128, 0, 1, 1.50924951),
)

PIPELINE = (("kimi-k3", 2), ("glm-5.3", 4), ("deepseek-v4-pro", 4), ("qwen3.5-397b-a17b", 2))

# Deterministic 2-layer MC2/shared-expert regression references. These are
# physical-model outputs for the synthetic program below, not NPU captures.
MC2_PLAIN_REF_MS = 0.406123
MC2_SCHEDULED_REF_MS = 0.666432


@pytest.fixture(autouse=True)
def isolate_mindie_homes(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    for name in ("home", "config", "data", "diagnostics"):
        (root / name).mkdir()
    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(root / "data"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(root / "diagnostics"))


def _rel(actual: float, measured: float) -> float:
    return abs(actual - measured) / measured


def _case(model, hardware, *, quant=None, tp=1, dp=1, ep=1, pp=1, pcp=1, sp=False,
          rpn=None, layers=2, experts=8, topk=2, mtp=0, accept=1.0, shared=False, recipe=None):
    kw = dict(
        model=model, hardware=hardware, tp=tp, dp=dp, ep=ep, pp=pp, pcp=pcp, sp=sp,
        layers=layers, n_routed_experts=experts, num_experts_per_tok=topk,
        prompt_tokens=64, output_tokens=8, concurrency=1, max_model_len=131072,
        enable_shared_expert_dp=shared,
    )
    if quant is not None:
        kw["quant"] = quant
    if rpn is not None:
        kw["ranks_per_node"] = rpn
    if mtp:
        kw["mtp_proposal_tokens"] = mtp
        kw["mtp_acceptance_length"] = accept
    case = compose_case(**kw)
    if recipe:
        case["runtime"]["moe_recipe"] = recipe
        case["runtime"]["mc2_chunks"] = 4
    return case


def _row_case(row, **overrides):
    name, model, hardware, quant, sku, tp, dp, ep, rpn, layers, experts, phase, tokens, ctx, seq, ref = row
    kw = dict(
        model=model, hardware=hardware, quant=quant, tp=tp, dp=dp, ep=ep,
        layers=layers, n_routed_experts=experts, num_experts_per_tok=2,
        prompt_tokens=64, output_tokens=8, concurrency=1, max_model_len=131072,
    )
    if rpn is not None:
        kw["ranks_per_node"] = rpn
    kw.update(overrides)
    return compose_case(**kw)


def _step(case, phase, tokens, context, sequences):
    return predict(case, scope="body", phase=phase, tokens=tokens, context=context, sequences=sequences)


def _conserve(eff, body_ms):
    ops = eff["operators"]
    summed = sum(op["duration_ms"] * op["occurrences"] for op in ops)
    assert summed == pytest.approx(body_ms, rel=1e-9)
    assert eff["operator_sum_ms"] == pytest.approx(body_ms, rel=1e-9)
    assert eff["graph_duration_ms"] == pytest.approx(body_ms, rel=1e-9)
    assert sum(op["link_bytes"] * op["occurrences"] for op in ops) == pytest.approx(eff["total_link_bytes"], abs=1e-6)
    assert sum(op["hbm_bytes"] * op["occurrences"] for op in ops) == pytest.approx(eff["total_hbm_bytes"], rel=1e-9)


def _bucket(ops, kind):
    if kind == "shared_expert":
        return [op for op in ops if "shared" in op["op_id"]]
    if kind == "expert_dispatch":
        return [op for op in ops if op["recipe"] == "expert_dispatch"]
    if kind == "expert_combine":
        return [op for op in ops if op["recipe"] == "expert_combine"]
    if kind == "tp_allreduce":
        return [op for op in ops if op["family"] == "allreduce"]
    if kind == "gdn":
        return [op for op in ops if op["family"] in {"short_convolution", "gdn_gating", "recurrent_vector"}]
    return [op for op in ops if op["family"] in {"attention", "mla"} or "attn" in op["recipe"]]


def _scoreboard():
    return json.loads(VERIFICATION.read_text(encoding="utf-8"))


def _scoreboard_row(case_id):
    row = next((candidate for candidate in _scoreboard()["rows"]
                if candidate["case_id"] == case_id), None)
    assert row is not None, case_id
    return row


def _metadata_key(value, key):
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                return child
            found = _metadata_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _metadata_key(child, key)
            if found is not None:
                return found
    return None


def _evaluation_relation(case, row):
    declared = _metadata_key((case.get("runtime") or {}).get("metadata") or {},
                             "evaluation_relation")
    if declared is not None:
        return str(declared)
    origin = row.get("calibration_origin")
    if origin is None:
        # Absence of a calibration relation is missing provenance, not
        # evidence that the row is an out-of-sample heldout capture.
        return "unknown"
    return "in_sample" if "in_sample" in str(origin) else "heldout_capture_not_in_sample"


def _declared_capture_path(case):
    metadata = (case.get("runtime") or {}).get("metadata") or {}
    capture = metadata.get("capture_directory")
    if capture:
        return str(capture)
    return str((metadata.get("kernel_source") or {}).get("path") or "")


def _explicit_capture_provenance(case, path):
    """Name the external campaign that owns one retained capture path."""
    for marker in ("heldout_engine_20261030", "glm5_3_flash_w8a8_heldout"):
        if marker in path:
            return marker
    capture = _declared_capture_path(case)
    return capture if capture and capture in path else ""


@pytest.mark.parametrize("relative,case_id", HELDOUT_CASES,
                         ids=[case_id.split(":")[0] for _, case_id in HELDOUT_CASES])
def test_heldout_real_npu_body_p50(relative, case_id):
    case = json.loads((CASE_ROOT / relative).read_text(encoding="utf-8"))
    row = _scoreboard_row(case_id)
    assert row["display_status"] == "compared"
    assert "kernel_details.csv" in row["source_identity"]["path"]
    assert row["comparison"]["metric_boundary"] == "target_body_device_graph"
    assert _evaluation_relation(case, row) != "in_sample"

    geometry = row["geometry"]
    measured = row["comparison"]["observed_ms"]["p50"]
    live = predict(
        case, scope="body", phase=geometry["phase"],
        tokens=geometry["tokens_per_sequence"],
        context=geometry["context_tokens"],
        sequences=geometry["sequences_per_replica"],
    )
    assert live["scope"] == "body"
    assert _rel(live["duration_ms"], measured) <= REL_TOL, (
        f"{relative}: live {live['duration_ms']:.6f} ms vs measured {measured:.6f} ms"
    )


def test_missing_evaluation_provenance_is_unknown():
    # Neither a declared relation nor a calibration origin exists. This says
    # nothing about heldout status, so it must not be inferred from absence.
    assert _evaluation_relation({}, {}) == "unknown"
    assert _evaluation_relation({}, {"calibration_origin": None}) == "unknown"
    assert _evaluation_relation(
        {}, {"calibration_origin": "observed"}) == "heldout_capture_not_in_sample"
    assert _evaluation_relation(
        {}, {"calibration_origin": "retained_operator_service_in_sample"}) == "in_sample"


@pytest.mark.parametrize("relative,case_id", HELDOUT_CASES,
                         ids=[case_id.split(":")[0] for _, case_id in HELDOUT_CASES])
def test_heldout_rows_retain_explicit_capture_provenance(relative, case_id):
    case = json.loads((CASE_ROOT / relative).read_text(encoding="utf-8"))
    row = _scoreboard_row(case_id)
    path = row["source_identity"]["path"]
    campaign = _explicit_capture_provenance(case, path)

    assert "kernel_details" in path
    assert campaign and (campaign in path)
    assert "in_sample" not in campaign.lower()
    # These retained high-concurrency rows do not declare an evaluation
    # relation; their external campaign path is the explicit provenance.
    assert _evaluation_relation(case, row) == "unknown"


@pytest.mark.parametrize("marker", ("heldout_engine_20261030",
                                    "glm5_3_flash_w8a8_heldout"))
def test_named_heldout_campaign_marker_is_explicit(marker):
    path = f"/captures/{marker}/run/kernel_details.csv"
    assert _explicit_capture_provenance({}, path) == marker


def test_declared_evaluation_relation_is_preserved():
    path = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp8.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    expected = "retrospective_cross_TP_frozen_training_validation"
    assert case["runtime"]["metadata"]["calibration"]["evaluation_relation"] == expected
    assert _evaluation_relation(case, {}) == expected
    assert _evaluation_relation(case, {}) != "in_sample"


@pytest.mark.parametrize("row", CAPTURES, ids=[row[0] for row in CAPTURES])
def test_composite_matrix_matches_physical_roofline(row):
    name, model, hardware, quant, sku, tp, dp, ep, rpn, layers, experts, phase, tokens, ctx, seq, reference = row
    case = _row_case(row)
    machine = case["machine"]
    assert machine["sku"] == sku
    assert machine["num_devices"] == tp * dp
    multi_node = machine["num_devices"] > machine["ranks_per_node"]
    assert multi_node is (name.endswith("-mn") or name.endswith("-mn-fp8"))

    got = _step(case, phase, tokens, ctx, seq)["duration_ms"]
    eff = analyze_network_efficiency(
        case, phase=phase, tokens=tokens, context=ctx, sequences=seq, scope="body")
    hbm_bw_bytes_per_ms = (
        extract_hardware_peaks(case)["hbm_peak_bytes_per_s"] / 1000.0
    )
    assert eff["total_roofline_ideal_ms"] <= got <= eff["operator_sum_ms"] * 1.05
    assert got >= eff["total_hbm_bytes"] / hbm_bw_bytes_per_ms
    # References are retained to eight decimals; pin the current physical
    # decomposition without pretending to sub-nanosecond measurement precision.
    assert got == pytest.approx(reference, abs=5e-9)
    _conserve(eff, got)


def test_multi_node_ep16_exceeds_single_node_hccl_latency():
    rows = {row[0]: row for row in CAPTURES}
    named_sn = _row_case(rows["a3-9362-ep16-sn"])
    named_mn = _row_case(rows["a2-ep16-mn"])
    assert _step(named_mn, "decode", 1, 64, 1)["duration_ms"] > _step(
        named_sn, "decode", 1, 64, 1)["duration_ms"]

    # Same chip, model, quant, and EP geometry; only the node boundary moves.
    single = _row_case(rows["a3-9362-ep16-sn"])
    multi = _row_case(rows["a3-9362-ep16-sn"], ranks_per_node=8)
    assert single["machine"]["sku"] == multi["machine"]["sku"] == "Ascend910_9362"
    assert single["machine"]["num_devices"] == multi["machine"]["num_devices"] == 16
    assert single["machine"]["ranks_per_node"] == 16
    assert multi["machine"]["ranks_per_node"] == 8
    single_ms = _step(single, "decode", 1, 64, 1)["duration_ms"]
    multi_ms = _step(multi, "decode", 1, 64, 1)["duration_ms"]
    single_eff = analyze_network_efficiency(
        single, phase="decode", tokens=1, context=64, sequences=1, scope="body")
    multi_eff = analyze_network_efficiency(
        multi, phase="decode", tokens=1, context=64, sequences=1, scope="body")
    assert multi_ms > single_ms
    assert multi_eff["total_link_bytes"] == pytest.approx(single_eff["total_link_bytes"])
    assert multi_eff["total_link_ideal_ms"] > single_eff["total_link_ideal_ms"]


def test_shared_expert_dp_mc2_conserves_weights_and_hccl_bytes():
    # Deterministic 2-layer MC2/shared-expert synthetic reference points.
    common = dict(model="qwen3.5-35b-a3b", hardware="Ascend910_9362", quant="w8a8", tp=4, dp=1, ep=4, layers=2)
    plain = _case(**common)
    scheduled = _case(**common, recipe="mc2")
    shared = _case(**common, shared=True, recipe="mc2")
    assert shared["runtime"]["metadata"]["enable_shared_expert_dp"] is True
    assert scheduled["runtime"]["moe_recipe"] == "mc2"

    plain_eff = analyze_network_efficiency(plain, scope="body", phase="decode", tokens=8, context=128, sequences=1)
    mc2_eff = analyze_network_efficiency(scheduled, scope="body", phase="decode", tokens=8, context=128, sequences=1)
    shared_step = _step(shared, "decode", 8, 128, 1)
    assert _rel(plain_eff["graph_duration_ms"], MC2_PLAIN_REF_MS) <= REL_TOL
    assert _rel(mc2_eff["graph_duration_ms"], MC2_SCHEDULED_REF_MS) <= REL_TOL
    assert _rel(plain_eff["graph_duration_ms"], MC2_SCHEDULED_REF_MS) > REL_TOL
    assert plain_eff["total_link_bytes"] == pytest.approx(mc2_eff["total_link_bytes"])
    assert plain_eff["total_link_bytes"] == pytest.approx(233496.0)
    _conserve(plain_eff, plain_eff["graph_duration_ms"])

    mc2_ops = mc2_eff["operators"]
    mc2_summed = sum(op["duration_ms"] * op["occurrences"] for op in mc2_ops)
    assert mc2_summed == pytest.approx(mc2_eff["operator_sum_ms"], rel=1e-9)
    assert mc2_eff["operator_sum_ms"] > mc2_eff["graph_duration_ms"]
    assert (sum(op["link_bytes"] * op["occurrences"] for op in mc2_ops)
            == pytest.approx(mc2_eff["total_link_bytes"], abs=1e-6))
    assert (sum(op["hbm_bytes"] * op["occurrences"] for op in mc2_ops)
            == pytest.approx(mc2_eff["total_hbm_bytes"], rel=1e-9))
    assert any(str(row.get("stream", "")).startswith("mc2") for row in shared_step["timeline"])

    for kind in ("shared_expert", "expert_dispatch", "expert_combine", "tp_allreduce", "gdn"):
        assert _bucket(mc2_eff["operators"], kind), kind
    dispatch = sum(op["link_bytes"] * op["occurrences"] for op in _bucket(mc2_eff["operators"], "expert_dispatch"))
    combine = sum(op["link_bytes"] * op["occurrences"] for op in _bucket(mc2_eff["operators"], "expert_combine"))
    reduce = sum(op["link_bytes"] * op["occurrences"] for op in _bucket(mc2_eff["operators"], "tp_allreduce"))
    assert dispatch > 0 and combine > 0 and reduce > 0
    assert dispatch + combine + reduce == pytest.approx(mc2_eff["total_link_bytes"])

    off = weight_breakdown(parse_case(plain).model, parse_case(plain).runtime.parallel, enable_shared_expert_dp=False)
    on = weight_breakdown(parse_case(shared).model, parse_case(shared).runtime.parallel, enable_shared_expert_dp=True)
    assert off["routed_expert"] == pytest.approx(12681216)
    assert on["routed_expert"] == pytest.approx(off["routed_expert"])
    assert off["shared_expert"] == pytest.approx(1609728)
    assert on["shared_expert"] == pytest.approx(6340608)
    assert shared_step["duration_ms"] != mc2_eff["graph_duration_ms"]


def _sla_baseline(case, *, context, output_budget):
    result = solve_hardware_sla_throughput(
        case, phase="decode", context=context, output_budget=output_budget,
        max_sequences_per_dp=1, scenarios=[{"id": "baseline"}], scope="body")
    return result, result["scenarios"][0]


def _assert_automatic_speculative_scaling(model, hardware, *, quant, tp, dp, ep,
                                          proposal, acceptance, context=256, output=16):
    common = dict(model=model, hardware=hardware, tp=tp, dp=dp, ep=ep, layers=2,
                  prompt_tokens=context, output_tokens=output, concurrency=1)
    if quant is not None:
        common["quant"] = quant
    off_case = compose_case(**common, mtp_proposal_tokens=0, mtp_acceptance_length=1.0)
    on_case = compose_case(**common, mtp_proposal_tokens=proposal,
                           mtp_acceptance_length=acceptance)

    parsed_off, parsed_on = parse_case(off_case), parse_case(on_case)
    off_policy = DraftPolicy.for_request(parsed_off, parsed_off.workload.requests[0])
    on_policy = DraftPolicy.for_request(parsed_on, parsed_on.workload.requests[0])
    assert off_policy.proposal == 0
    assert on_policy.proposal == proposal
    assert on_policy.mean == pytest.approx(acceptance)
    assert on_policy.mode in {"mtp", "dspark"}

    off, off_row = _sla_baseline(off_case, context=context, output_budget=output)
    on, on_row = _sla_baseline(on_case, context=context, output_budget=output)
    assert off["tokens_per_seq"] == 1
    assert on["tokens_per_seq"] == 1 + proposal
    assert off["emitted_tokens_per_seq"] == pytest.approx(1.0)
    assert on["emitted_tokens_per_seq"] == pytest.approx(acceptance)

    single_token_step_ms = off_row["step_ms_at_b1"]
    verify_step_ms = on_row["step_ms_at_b1"]
    tpot_effective_ms = verify_step_ms / acceptance
    assert verify_step_ms > single_token_step_ms
    assert tpot_effective_ms < single_token_step_ms

    off_eff = analyze_network_efficiency(
        off_case, phase="decode", tokens=off["tokens_per_seq"],
        context=context, sequences=1, scope="body")
    on_eff = analyze_network_efficiency(
        on_case, phase="decode", tokens=on["tokens_per_seq"],
        context=context, sequences=1, scope="body")
    assert on_eff["total_hbm_bytes"] > off_eff["total_hbm_bytes"]
    assert on_eff["operator_sum_ms"] > off_eff["operator_sum_ms"]


def test_mtp_and_dspark_scale_verify_rows_and_effective_tpot():
    _assert_automatic_speculative_scaling(
        "kimi-k3", "Ascend950DT_9582", quant=None, tp=2, dp=1, ep=2,
        proposal=3, acceptance=3.0)
    _assert_automatic_speculative_scaling(
        "qwen3.5-35b-a3b", "Ascend910_9392", quant="w8a8", tp=2, dp=2, ep=4,
        proposal=7, acceptance=7.0)

    # DSpark is selected from structure, not from a manually widened token
    # count. The automatic 1 + proposal verify geometry and TPOT amortization
    # are established by the SLA resolver above; the explicit body probe keeps
    # caller-supplied verification rows and must not silently scale itself.
    kimi = _case("kimi-k3", "Ascend950DT_9582", tp=2, dp=1, ep=2, layers=2,
                 mtp=3, accept=3.0)
    assert kimi["runtime"]["metadata"]["draft_mode"] == "dspark"
    disabled = _case("kimi-k3", "Ascend950DT_9582", tp=2, dp=1, ep=2, layers=2)
    assert "draft_mode" not in disabled["runtime"]["metadata"]


def test_pp_pcp_ep_sp_scaling_matches_roofline_and_bytes():
    curves = {}
    for model, layers in PIPELINE:
        case = _case(model, "a3", tp=2, dp=1, pp=2, pcp=2, ep=2, sp=True, layers=layers, experts=4, topk=1)
        short = _step(case, "prefill", 128, 0, 1)["duration_ms"]
        longer = _step(case, "prefill", 1024, 0, 1)["duration_ms"]
        assert longer > short
        curves[model] = case
    qwen = curves["qwen3.5-397b-a17b"]
    by_tokens = [(_step(qwen, "prefill", tokens, 0, 1)["duration_ms"], tokens) for tokens in (128, 1024, 4096, 32768)]
    assert [ms for ms, _ in by_tokens] == sorted(ms for ms, _ in by_tokens)
    by_conc = [by_tokens[0][0]]
    for sequences in (4, 16):
        by_conc.append(_step(qwen, "prefill", 128, 0, sequences)["duration_ms"])
    assert by_conc == sorted(by_conc) and by_conc[0] < by_conc[-1]

    short_eff = analyze_network_efficiency(qwen, scope="body", phase="prefill", tokens=128, context=0, sequences=1)
    long_eff = analyze_network_efficiency(qwen, scope="body", phase="prefill", tokens=32768, context=0, sequences=1)
    assert _rel(short_eff["graph_duration_ms"], by_tokens[0][0]) <= REL_TOL
    assert _rel(long_eff["graph_duration_ms"], by_tokens[-1][0]) <= REL_TOL
    assert long_eff["total_roofline_ideal_ms"] > short_eff["total_roofline_ideal_ms"]
    assert long_eff["total_hbm_bytes"] > short_eff["total_hbm_bytes"]
    assert long_eff["total_link_bytes"] > short_eff["total_link_bytes"]
    assert short_eff["graph_duration_ms"] + 1e-9 >= short_eff["total_roofline_ideal_ms"]
    _conserve(short_eff, short_eff["graph_duration_ms"])

    mla = curves["deepseek-v4-pro"]
    decode = _step(mla, "decode", 1, 1024, 1)
    eff = analyze_network_efficiency(mla, scope="body", phase="decode", tokens=1, context=1024, sequences=1)
    assert _rel(eff["graph_duration_ms"], decode["duration_ms"]) <= REL_TOL
    _conserve(eff, decode["duration_ms"])
    rows = dict(cache_row_bytes_by_layer(parse_case(mla).model))
    stores = [op for op in eff["operators"] if op["recipe"] == "kv_store"]
    ropes = [op for op in eff["operators"] if op["recipe"] == "rotary"]
    attn = [op for op in eff["operators"] if op["recipe"] == "latent_attention0"]
    assert stores and ropes and attn
    assert {op["hbm_bytes"] for op in stores} == {2 * rows[stores[0]["layer"]]}
    assert len({op["hbm_bytes"] for op in ropes}) == 1 and ropes[0]["hbm_bytes"] > 0
    assert min(op["hbm_bytes"] for op in attn) > max(op["hbm_bytes"] for op in stores)
    moe_link = sum(
        op["link_bytes"] * op["occurrences"]
        for op in eff["operators"]
        if op["recipe"] in {"expert_dispatch", "expert_combine"}
    )
    assert moe_link > 0
    assert moe_link <= eff["total_link_bytes"]
