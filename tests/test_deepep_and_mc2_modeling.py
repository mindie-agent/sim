"""DeepEP, EPLB and shared-expert overlap behavior.

[Role] Verify MXFP8 33/32 B transport, two-stage deduplicated DeepEP volumes,
  EPLB scaling, causal DAG lowering and shared-expert stream overlap.
[Invariants] MXFP8 carries 32 E4M3 bytes plus one E8M0 block-32 scale byte per
  32 elements; combine is BF16. DeepEP deduplicates one remote delivery per
  token and node. Overlap changes scheduling only, never total FLOPs or HBM.
[Timeouts & Locks] Pure in-memory cases and bounded graph construction.
[Side-Effect Policy] No device work, file writes, subprocesses or locks.
"""
from dataclasses import asdict

import pytest

from mindie_sim import analyze_network_efficiency, graph_costs, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction import deepep, moe_transport
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError, dtype_bytes, dtype_peak_scale
from test_prediction_system import case


MXFP8_SELECTOR = {
    "kind": "separate_quantized_payload",
    "dispatch_dtype": "mxfp8",
    "scale_group": 32,
    "layer_scopes": ["body"],
}


def mxfp8_case(*, dp=2, ep=2, tokens=4, eplb=1.0):
    raw = case(dp=dp, tp=1, ep=ep, prompt=tokens, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=2, hidden_size=32,
        moe_intermediate_size=32, activation_dtype="bf16",
        quant=asdict(quant_preset("w8a8_mxfp8")),
    )
    raw["machine"]["device"]["compute_peaks"].update(
        {"fp8": 2e9, "fp8_e4m3": 2e9, "mxfp8": 2e9})
    raw["runtime"]["max_num_batched_tokens"] = 16
    raw["runtime"]["kernels"] = {
        "moe_transport": {**MXFP8_SELECTOR, "eplb_peak_to_avg": eplb}}
    return raw


def mxfp8_graphs(*, dp=2, ep=2, tokens=4, eplb=1.0):
    raw = mxfp8_case(dp=dp, ep=ep, tokens=tokens, eplb=eplb)
    graphs = target_graphs(raw, phase="prefill", tokens=tokens, scope="model")
    return raw, graphs


def _one(ops, recipe, *, scope=None, layer=None):
    rows = [op for op in ops if op.recipe == recipe
            and (scope is None or dict(op.metadata).get("quantization_scope") == scope)
            and (layer is None or op.layer == layer)]
    assert len(rows) == 1, [(op.op_id, op.recipe) for op in rows]
    return rows[0]


def _moE_case(*, dp=2, ep=2, layers=2, shared=0, quant="w8a8_int8"):
    raw = case(dp=dp, tp=1, ep=ep, prompt=4, output=1)
    raw["model"].update(
        layers=layers, ffn="moe", num_experts=4, top_k=2, hidden_size=32,
        moe_intermediate_size=32, num_shared_experts=shared,
        activation_dtype="bf16", quant=asdict(quant_preset(quant)),
    )
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e9
    raw["runtime"]["max_num_batched_tokens"] = 16
    return raw


def test_mxfp8_declares_33_over_32_bytes_and_two_x_peak():
    assert dtype_bytes("mxfp8") == pytest.approx(33.0 / 32.0) == 1.03125
    assert dtype_bytes("mx_fp8") == pytest.approx(1.03125)
    assert dtype_bytes("mxfp8_e4m3") == pytest.approx(1.03125)
    assert dtype_peak_scale("mxfp8") == 2.0
    assert dtype_peak_scale("mxfp8_e4m3") == 2.0


def test_mxfp8_transport_encodes_payload_and_block_scales_per_row():
    raw, graphs = mxfp8_graphs(tokens=4, ep=2)
    ops = graphs[0]
    width, top_k, tokens, ep = 32, 2, 4, 2
    rows = tokens * top_k
    quantize = _one(ops, "dynamic_quantize",
                    scope="moe_transport_expert_input")
    dispatch = _one(ops, "expert_quantized_dispatch")
    combine = _one(ops, "expert_combine")
    dispatch_meta = dict(dispatch.metadata)
    combine_meta = dict(combine.metadata)

    assert quantize.mte3_bytes == rows * (width + width // 32)
    assert dispatch.link_bytes == pytest.approx(
        rows * (width + width // 32) * (ep - 1) / ep)
    assert dispatch_meta["row_bytes"] == width + width // 32
    assert dispatch_meta["payload_dtype"] == "fp8_e4m3"
    assert dispatch_meta["scale_dtype"] == "fp8_e8m0"
    assert dispatch_meta["scale_group"] == 32
    assert combine.link_bytes == pytest.approx(
        rows * width * dtype_bytes("bf16") * (ep - 1) / ep)
    assert combine_meta["combine_dtype"] == "bf16"
    assert combine_meta["row_bytes"] == width * dtype_bytes("bf16")

    parsed = parse_case(raw)
    layer = parsed.model.layer_plans[0]
    fmt = moe_transport._transport_format(parsed, layer, width)
    assert fmt["dispatch_dtype"] == "mxfp8"
    assert fmt["payload_bytes"] == width
    assert fmt["scale_bytes"] == width // 32
    assert fmt["row_bytes"] == width + width // 32
    assert fmt["element_bytes"] == pytest.approx(33.0 / 32.0)
    assert fmt["combine_dtype"] == "bf16"
    report = graph_costs(raw, graphs)
    assert report["duration_ms"] > 0
    assert report["execution"]["live_tasks"] == 0


def test_mxfp8_eplb_ratio_scales_both_exchange_directions():
    _, base_graphs = mxfp8_graphs(eplb=1.0)
    raw, scaled_graphs = mxfp8_graphs(eplb=1.25)
    base_dispatch = _one(base_graphs[0], "expert_quantized_dispatch")
    scaled_dispatch = _one(scaled_graphs[0], "expert_quantized_dispatch")
    base_combine = _one(base_graphs[0], "expert_combine")
    scaled_combine = _one(scaled_graphs[0], "expert_combine")
    assert scaled_dispatch.link_bytes == pytest.approx(
        base_dispatch.link_bytes * 1.25)
    assert scaled_combine.link_bytes == pytest.approx(
        base_combine.link_bytes * 1.25)
    assert dict(scaled_dispatch.metadata)["eplb_peak_to_avg"] == 1.25
    parsed = parse_case(raw)
    fmt = moe_transport._transport_format(
        parsed, parsed.model.layer_plans[0], parsed.model.hidden_size)
    assert fmt["eplb_peak_to_avg"] == 1.25


def test_mxfp8_rejects_non_fp8_expert_weights_and_invalid_ratios():
    raw = mxfp8_case()
    raw["model"]["quant"] = asdict(quant_preset("w8a8_int8"))
    with pytest.raises(ModelingError, match="requires_fp8_expert_weights"):
        target_graphs(raw, phase="prefill", tokens=4, scope="model")
    raw = mxfp8_case(eplb=0.5)
    with pytest.raises(ModelingError, match="eplb_peak_to_avg_must_be_at_least_one"):
        target_graphs(raw, phase="prefill", tokens=4, scope="model")


def test_deepep_volumes_deduplicate_inter_node_and_zero_on_single_node():
    tokens, hidden, top_k, dtype = 100.0, 4096, 8, 2.0
    assert deepep.deepep_group_volume(
        tokens, hidden, top_k, ep=8, ranks_per_node=8,
        dtype_bytes_val=dtype) == 0.0
    intra = deepep.deepep_intra_volume(
        tokens, hidden, top_k, ep=8, ranks_per_node=8,
        dtype_bytes_val=dtype)
    assert intra == pytest.approx(tokens * hidden * dtype * top_k * 7 / 8)

    group = deepep.deepep_group_volume(
        tokens, hidden, top_k, ep=64, ranks_per_node=8,
        dtype_bytes_val=dtype)
    standard = deepep.standard_alltoall_volume(
        tokens, hidden, top_k, ep=64, dtype_bytes_val=dtype)
    assert group == pytest.approx(tokens * hidden * dtype * 1.0 * 7)
    # One delivery per token and remote node is less than flat top-k fan-out.
    assert group < standard
    assert deepep.deepep_group_volume(
        tokens, hidden, top_k, ep=64, ranks_per_node=8,
        dtype_bytes_val=dtype, eplb_peak_to_avg=1.3
    ) == pytest.approx(group * 1.3)


def multi_node_deepep_case():
    raw = case(dp=64, tp=1, ep=64, prompt=8, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=64, top_k=8,
        hidden_size=4096, num_heads=64, head_dim=64,
        moe_intermediate_size=128,
    )
    raw["machine"]["ranks_per_node"] = 8
    raw["machine"]["links"] = [
        {"name": "intra", "capacity_bytes_per_ms": 1000,
         "members": list(range(8)), "channel": "intra_package",
         "latency_us": 1.0},
        {"name": "inter", "capacity_bytes_per_ms": 100,
         "members": list(range(64)), "channel": "rdma",
         "latency_us": 10.0},
    ]
    return raw


def test_deepep_two_stage_analysis_reduces_time_all_to_all():
    raw = multi_node_deepep_case()
    report = deepep.analyze_deepep_collectives(
        raw, phase="prefill", tokens=8, sequences=1)
    assert report["ep"] == 64
    assert report["ranks_per_node"] == 8
    assert report["nodes"] == 8
    assert report["moe_layers"] == 1
    assert len(report["per_layer"]) == 1
    row = report["per_layer"][0]
    assert row["deepep_dispatch_group_bytes"] < row["standard_dispatch_bytes"]
    assert row["deepep_dispatch_intra_bytes"] > 0
    assert report["standard_total_comm_ms"] > 0
    assert report["deepep_total_comm_ms"] < report["standard_total_comm_ms"]
    assert report["comm_saving_ratio"] > 0
    # Overlap removes communication time without changing physical work.
    half = deepep.analyze_deepep_collectives(
        raw, phase="prefill", tokens=8, sequences=1,
        deepep_overlap_ratio=0.5)
    assert half["deepep_total_comm_ms"] == pytest.approx(
        half["deepep_unoverlapped_comm_ms"] * 0.5)
    assert half["comm_saving_ratio"] > report["comm_saving_ratio"]


def test_deepep_dag_lowering_scales_payloads_and_lowers_step_time():
    raw = _moE_case()
    raw["machine"]["links"][0]["capacity_bytes_per_ms"] = 10000
    rates = {}
    for overlap in (0.0, 0.3, 0.6):
        value = {**raw, "runtime": {**raw["runtime"], "metadata": {
            "enable_deepep": True, "deepep_overlap_ratio": overlap}}}
        report = analyze_network_efficiency(
            value, phase="prefill", tokens=4, scope="model")
        rates[overlap] = report["graph_duration_ms"]
    assert rates[0.0] > rates[0.3] > rates[0.6]

    baseline = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    lowered_raw = {**raw, "runtime": {**raw["runtime"], "metadata": {
        "enable_deepep": True, "deepep_overlap_ratio": 0.3}}}
    lowered = target_graphs(lowered_raw, phase="prefill", tokens=4, scope="model")
    base_dispatch = _one(baseline[0], "expert_dispatch", layer=0)
    deepep_dispatch = _one(lowered[0], "expert_dispatch", layer=0)
    deepep_dispatch_over = _one(lowered[0], "expert_dispatch_overlap", layer=0)
    base_combine = _one(baseline[0], "expert_combine", layer=0)
    deepep_combine = _one(lowered[0], "expert_combine", layer=0)
    deepep_combine_over = _one(lowered[0], "expert_combine_overlap", layer=0)
    assert deepep_dispatch.link_bytes == pytest.approx(
        base_dispatch.link_bytes * 0.7)
    assert deepep_dispatch_over.link_bytes == pytest.approx(
        base_dispatch.link_bytes * 0.3)
    assert deepep_dispatch.link_bytes + deepep_dispatch_over.link_bytes == pytest.approx(
        base_dispatch.link_bytes)
    assert deepep_combine.link_bytes == pytest.approx(
        base_combine.link_bytes * 0.7)
    assert deepep_combine_over.link_bytes == pytest.approx(
        base_combine.link_bytes * 0.3)
    assert deepep_combine.link_bytes + deepep_combine_over.link_bytes == pytest.approx(
        base_combine.link_bytes)
    assert deepep_dispatch_over.stream == "deepep_comm_stream"
    assert deepep_combine_over.stream == "deepep_comm_stream"
    assert dict(deepep_dispatch.metadata)["deepep_mode"] is True
    assert dict(deepep_dispatch.metadata)["deepep_overlap_ratio"] == 0.3
    assert dict(deepep_combine.metadata)["deepep_mode"] is True


def test_deepep_multi_node_dag_uses_the_two_stage_volume_ratio():
    raw = _moE_case(dp=8, ep=8, layers=1)
    raw["model"].update(num_experts=8)
    raw["runtime"]["metadata"] = {
        "enable_deepep": True, "deepep_overlap_ratio": 0.25,
        "deepep_ranks_per_node": 4,
    }
    lowered = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    dispatch = _one(lowered[0], "expert_dispatch")
    dispatch_over = _one(lowered[0], "expert_dispatch_overlap")
    group = deepep.deepep_group_volume(
        4, 32, 2, ep=8, ranks_per_node=4, dtype_bytes_val=2.0)
    intra = deepep.deepep_intra_volume(
        4, 32, 2, ep=8, ranks_per_node=4, dtype_bytes_val=2.0)
    standard = deepep.standard_alltoall_volume(
        4, 32, 2, ep=8, dtype_bytes_val=2.0)
    expected_factor = (group + intra) / standard
    assert dict(dispatch.metadata)["deepep_dedup_scale"] == pytest.approx(
        expected_factor)
    baseline_raw = _moE_case(dp=8, ep=8, layers=1)
    baseline_raw["model"].update(num_experts=8)
    baseline = target_graphs(
        baseline_raw, phase="prefill", tokens=4, scope="model")
    base_dispatch = _one(baseline[0], "expert_dispatch")
    assert dispatch.link_bytes == pytest.approx(
        base_dispatch.link_bytes * expected_factor * 0.75)
    assert dispatch.link_bytes + dispatch_over.link_bytes == pytest.approx(
        base_dispatch.link_bytes * expected_factor)


def test_eplb_without_deepep_scales_flat_all_to_all_dispatch_and_combine():
    base = target_graphs(_moE_case(layers=1), phase="prefill", tokens=4,
                         scope="model")
    raw_scaled = _moE_case(layers=1)
    raw_scaled["runtime"]["kernels"] = {"moe_transport": {"eplb_peak_to_avg": 1.3}}
    scaled = target_graphs(raw_scaled, phase="prefill", tokens=4, scope="model")
    base_dispatch = _one(base[0], "expert_dispatch")
    scaled_dispatch = _one(scaled[0], "expert_dispatch")
    base_combine = _one(base[0], "expert_combine")
    scaled_combine = _one(scaled[0], "expert_combine")
    assert scaled_dispatch.link_bytes == pytest.approx(
        base_dispatch.link_bytes * 1.3)
    assert scaled_combine.link_bytes == pytest.approx(
        base_combine.link_bytes * 1.3)
    assert dict(scaled_dispatch.metadata)["eplb_peak_to_avg"] == 1.3


def test_shared_expert_overlap_runs_on_shared_stream_and_conserves_work():
    raw = _moE_case(layers=2, shared=1)
    report = deepep.analyze_shared_expert_overlap(
        raw, phase="prefill", tokens=4, sequences=1, scope="model")
    assert report["overlapped_step_ms"] <= report["baseline_step_ms"]
    assert report["saved_ms"] >= 0
    assert report["speedup_ratio"] >= 1.0
    assert report["flops_conserved"] is True
    assert report["hbm_bytes_conserved"] is True
    assert report["baseline_total_flops"] == report["overlapped_total_flops"]
    assert report["baseline_total_hbm_bytes"] == report["overlapped_total_hbm_bytes"]

    stream_case = _moE_case(layers=1, shared=1)
    enabled = {**stream_case, "runtime": {**stream_case["runtime"], "metadata": {
        "overlap_shared_expert": True}}}
    graphs = target_graphs(enabled, phase="prefill", tokens=4, scope="model")
    shared = [op for op in graphs[0] if op.stream == "shared_stream"]
    assert shared
    assert all(dict(op.metadata).get("ffn_category") in {None, "shared_expert"}
               for op in shared)
    join = _one(graphs[0], "expert_add")
    assert "s0.l0.unpack" in join.deps
    assert any(dependency.startswith("s0.l0.shared") for dependency in join.deps)
    baseline = target_graphs(stream_case, phase="prefill", tokens=4, scope="model")
    assert not any(op.stream == "shared_stream" for op in baseline[0])


def test_deepep_overlap_preserves_causal_order_under_slow_compute_and_int8_scale_validation():
    raw = _moE_case(layers=1)
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e6
    raw["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.4}
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    report = graph_costs(raw, graphs)
    by_id = {row["op_id"]: row for row in report["timeline"] if row.get("replica", 0) == 0}
    assert by_id["s0.l0.expert_up_overlap.quantize"]["ready_ms"] >= by_id["s0.l0.dispatch_overlap"]["complete_ms"]
    assert by_id["s0.l0.expert_up_overlap"]["ready_ms"] >= by_id["s0.l0.dispatch_overlap"]["complete_ms"]
    assert by_id["s0.l0.expert_act"]["ready_ms"] >= by_id["s0.l0.expert_up_overlap"]["complete_ms"]
    assert by_id["s0.l0.combine_overlap"]["ready_ms"] >= by_id["s0.l0.expert_down"]["complete_ms"]
    assert by_id["s0.l0.combine"]["ready_ms"] >= by_id["s0.l0.expert_down_overlap"]["complete_ms"]

    slow_link = _moE_case(layers=1)
    slow_link["machine"]["links"][0]["capacity_bytes_per_ms"] = 1.0
    slow_link["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.4}
    slow_graphs = target_graphs(slow_link, phase="prefill", tokens=4, scope="model")
    slow_report = graph_costs(slow_link, slow_graphs)
    slow_by_id = {row["op_id"]: row for row in slow_report["timeline"] if row.get("replica", 0) == 0}
    assert slow_by_id["s0.l0.expert_up_overlap"]["ready_ms"] >= slow_by_id["s0.l0.dispatch_overlap"]["complete_ms"]

    bad_int8 = _moE_case(layers=1)
    bad_int8["runtime"]["kernels"] = {
        "moe_transport": {
            "kind": "separate_quantized_payload",
            "dispatch_dtype": "int8",
            "scale_dtype": "e8m0",
        }
    }
    with pytest.raises(ModelingError, match="scale_dtype='e8m0';expected='fp32'"):
        target_graphs(bad_int8, phase="prefill", tokens=4, scope="model")

    no_ov_case = _moE_case(layers=1)
    no_ov_case["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.0}
    ov_case = _moE_case(layers=1)
    ov_case["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.4}
    eff_no_ov = analyze_network_efficiency(no_ov_case, phase="prefill", tokens=4, scope="model")
    eff_ov = analyze_network_efficiency(ov_case, phase="prefill", tokens=4, scope="model")
    assert eff_ov["total_flops"] == pytest.approx(eff_no_ov["total_flops"])
    assert eff_ov["total_hbm_bytes"] == pytest.approx(eff_no_ov["total_hbm_bytes"])

    mn = multi_node_deepep_case()
    mn["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.25}
    ov_graphs = target_graphs(mn, phase="prefill", tokens=8, scope="model")
    disp_meta = dict(_one(ov_graphs[0], "expert_dispatch").metadata)
    assert disp_meta["peer_traffic"]["kind"] == "two_tier"

    from mindie_sim import predict
    unequal_mn = multi_node_deepep_case()
    unequal_mn["model"]["layers"] = 1
    unequal_mn["workload"]["requests"][0]["prompt_tokens"] = 8
    unequal_mn["workload"]["requests"][0]["output_tokens"] = 1
    unequal_mn["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.25}
    unequal_report = predict(unequal_mn)
    assert unequal_report["metrics"]["e2e_ms"]["mean"] > 0

    mxfp8_bad = mxfp8_case()
    mxfp8_bad["runtime"]["kernels"] = {
        "moe_transport": {
            "kind": "separate_quantized_payload",
            "dispatch_dtype": "fp8",
            "scale_dtype": "fp32",
        }
    }
    with pytest.raises(
        ModelingError,
        match="moe_transport_gate_requires_matching_quantized_payload|quantized_moe_dispatch_requires_matching_expert_input_domain",
    ):
        target_graphs(mxfp8_bad, phase="prefill", tokens=4, scope="model")

