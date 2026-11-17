"""DeepEP, EPLB and shared-expert overlap behavior.

[Role] Verify MXFP8 33/32 B transport, two-stage deduplicated DeepEP volumes,
  EPLB scaling, causal DAG lowering and shared-expert stream overlap.
[Invariants] MXFP8 carries 32 E4M3 bytes plus one E8M0 block-32 scale byte per
  32 elements; combine is BF16. DeepEP deduplicates one remote delivery per
  token and node. Overlap changes scheduling only, never total FLOPs or HBM.
[Timeouts & Locks] Pure in-memory cases and bounded graph construction.
[Side-Effect Policy] No device work, file writes, subprocesses or locks.
"""
from dataclasses import asdict, replace

import pytest

from mindie_sim import analyze_network_efficiency, graph_costs, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction import deepep, moe_transport
from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction.body_defaults import TargetGraphSet
from mindie_sim.prediction.implementations import produced_values
from mindie_sim.prediction.transforms import activation_quantization
from mindie_sim.schema.operators import OpSpec
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


def _moE_case(*, dp=2, ep=2, layers=2, shared=0, quant="w8a8_int8",
              prompt=4, top_k=2):
    raw = case(dp=dp, tp=1, ep=ep, prompt=prompt, output=1)
    raw["model"].update(
        layers=layers, ffn="moe", num_experts=4, top_k=top_k, hidden_size=32,
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



def test_deepep_dag_lowering_scales_payloads_and_lowers_step_time():
    raw = _moE_case(prompt=8, top_k=1)
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
    # The transport split is the integer row boundary (3/1 for 4 rows at 0.3),
    # not the overlap scalar (0.7/0.3) applied to a byte total.
    assert deepep_dispatch.link_bytes == pytest.approx(
        base_dispatch.link_bytes * 3 / 4)
    assert deepep_dispatch_over.link_bytes == pytest.approx(
        base_dispatch.link_bytes * 1 / 4)
    assert dict(deepep_dispatch.metadata)["deepep_row_cut"] == (3, 1)
    assert dict(deepep_dispatch.metadata)["deepep_chunk_rows"] == 3
    assert deepep_dispatch.link_bytes + deepep_dispatch_over.link_bytes == pytest.approx(
        base_dispatch.link_bytes)
    # The inverse combine reads the same integer dispatch chunks (3/1), so the
    # return edge follows its forward declaration instead of the ratio scalar.
    assert deepep_combine.link_bytes == pytest.approx(
        base_combine.link_bytes * 3 / 4)
    assert deepep_combine_over.link_bytes == pytest.approx(
        base_combine.link_bytes * 1 / 4)
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


def _expert_up_ops(raw, *, tokens, overlap):
    raw = {**raw, "runtime": {**raw["runtime"],
                              "metadata": {"enable_deepep": True,
                                           "deepep_overlap_ratio": overlap}}}
    graphs = target_graphs(raw, phase="prefill", tokens=tokens, scope="model")
    return raw, graphs, {op.op_id: op for op in graphs[0]}


def _assert_internal_deps(ops):
    ids = set(ops)
    for op in ops.values():
        for dep in op.deps:
            if dep in {"analysis_input", "late_tail"}:
                continue
            assert dep in ids, (op.op_id, dep)


def test_deepep_overlap_splits_actual_activation_intervals_and_tail_readiness():
    raw, graphs, ops = _expert_up_ops(
        _moE_case(layers=1, prompt=8, top_k=1), tokens=8, overlap=0.5)
    first = ops["s0.l0.expert_up.quantize"]
    second = ops["s0.l0.expert_up_overlap.quantize"]
    first_view = ops["s0.l0.expert_up.rows0_4"]
    second_view = ops["s0.l0.expert_up.rows4_8"]

    # The two conversions read the actual source views, not a guessed producer.
    assert first.deps == (first_view.op_id,)
    assert second.deps == (second_view.op_id,)
    assert first_view.deps == ("s0.l0.dispatch",)
    assert second_view.deps == ("s0.l0.dispatch", "s0.l0.dispatch_overlap")
    assert dict(first_view.metadata)["row_start"] == 0
    assert dict(first_view.metadata)["row_stop"] == 4
    assert dict(second_view.metadata)["row_start"] == 4
    assert dict(second_view.metadata)["row_stop"] == 8
    assert dict(second_view.metadata)["byte_offset"] == 4 * 32 * dtype_bytes(second_view.dtype)
    assert dict(first_view.metadata)["full_rows"] == 8
    assert dict(first_view.metadata)["full_active_rows"] == 8
    # The view aliases the actual BF16 receive value of the dispatch, not the
    # first dtype-bearing ancestor (pack) and not the gate's consumer dtype.
    assert dict(first_view.metadata)["source_producer"] == "s0.l0.dispatch"
    assert dict(first_view.metadata)["source_output"] == "output"
    assert dict(first_view.metadata)["source_alias"] == "s0.l0.dispatch"
    assert dict(first_view.metadata)["source_storage_shape"] == (8, 32)
    assert dict(first_view.metadata)["named_input"] == "s0.l0.dispatch.output"
    assert dict(first_view.metadata)["deepep_row_cut"] == (4, 4)
    assert dict(second_view.metadata)["row_start"] == 4
    _assert_internal_deps(ops)
    # Four rows each, BF16 source; the exact conversion work is asserted by
    # the direct T11 control below.  The full graph path must not infer its
    # source from the gate's consumer dtype.
    assert ops["s0.l0.expert_up"].flops == pytest.approx(16384)
    assert ops["s0.l0.expert_up_overlap"].flops == pytest.approx(16384)
    assert ops["s0.l0.expert_up"].mte3_bytes == pytest.approx(512)
    assert ops["s0.l0.expert_up_overlap"].mte3_bytes == pytest.approx(512)
    assert first.mte3_bytes == pytest.approx(144)
    assert second.mte3_bytes == pytest.approx(144)
    first_value = produced_values(first_view)[0]
    second_value = produced_values(second_view)[0]
    assert first_value["logical_shape"] == [4, 32]
    assert second_value["logical_shape"] == [4, 32]
    # Both views keep the full 512 B receive allocation and the same alias.
    assert first_value["storage_shape"] == [8, 32]
    assert second_value["storage_shape"] == [8, 32]
    assert first_value["storage_bytes"] == 512
    assert second_value["storage_bytes"] == 512
    assert first_value["alias_of"] == second_value["alias_of"] == "s0.l0.dispatch"
    assert first_value["storage"] == second_value["storage"] == "moe.rx:s0.l0.dispatch"
    assert first_value["state_version"] is None and second_value["state_version"] is None
    assert first_value["binding"] == second_value["binding"] == "static"
    assert first_value["binding_name"] == second_value["binding_name"] == "s0.l0.dispatch.rx"
    assert first_value["materialized"] is False and second_value["materialized"] is False
    assert first_value["write_bytes"] == second_value["write_bytes"] == 0
    assert first_value["layout"]["offset"] == 0
    assert second_value["layout"]["offset"] == 4 * 32
    assert dict(first_view.metadata)["byte_offset"] == 0
    assert dict(second_view.metadata)["byte_offset"] == 256
    act = ops["s0.l0.expert_act"]
    assert dict(act.metadata)["activation_input_segments"] == (
        ("s0.l0.expert_up", 0, 4, 4, 64),
        ("s0.l0.expert_up_overlap", 4, 8, 4, 64))
    assert dict(act.metadata)["activation_input_bytes"] == 1024

    _raw04, _graphs04, ops04 = _expert_up_ops(
        _moE_case(layers=1, prompt=8, top_k=1), tokens=8, overlap=0.4)
    assert ops04["s0.l0.expert_up"].local_shape == (5, 64, 32)
    assert ops04["s0.l0.expert_up_overlap"].local_shape == (3, 64, 32)
    assert ops04["s0.l0.expert_up"].flops == pytest.approx(20480)
    assert ops04["s0.l0.expert_up_overlap"].flops == pytest.approx(12288)
    assert ops04["s0.l0.expert_up"].mte3_bytes == pytest.approx(640)
    assert ops04["s0.l0.expert_up_overlap"].mte3_bytes == pytest.approx(384)
    assert dict(ops04["s0.l0.expert_up.rows0_5"].metadata)["row_stop"] == 5
    assert dict(ops04["s0.l0.expert_up.rows5_8"].metadata)["row_start"] == 5
    assert dict(ops04["s0.l0.expert_up.rows5_8"].metadata)["byte_offset"] == 320
    assert dict(ops04["s0.l0.expert_up.rows0_5"].metadata)["deepep_row_cut"] == (5, 3)
    assert dict(ops04["s0.l0.expert_act"].metadata)["activation_input_segments"] == (
        ("s0.l0.expert_up", 0, 5, 5, 64),
        ("s0.l0.expert_up_overlap", 5, 8, 3, 64))
    _assert_internal_deps(ops04)

    # Tail arrival is a real dependency: only the second view/quantize may wait.
    delayed = TargetGraphSet(
        {replica: tuple(replace(op, deps=(*op.deps, "late_tail"))
                        if replica == 0 and op.op_id == "s0.l0.dispatch_overlap"
                        else op for op in graphs[replica])
         for replica in graphs},
        kernels=graphs[0].effective_kernels,
        selection=graphs[0].default_selection)
    report = graph_costs(
        raw, delayed, external_inputs={"analysis_input": 0, "late_tail": 50})
    by_id = {row["op_id"]: row for row in report["timeline"]
             if row.get("replica", 0) == 0}
    assert by_id["s0.l0.expert_up.quantize"]["complete_ms"] < 50
    assert by_id["s0.l0.expert_up_overlap.quantize"]["ready_ms"] >= 50
    assert report["execution"]["live_tasks"] == 0

    # Delay the front source instead: both segments and the complete activation
    # must wait for it, while the tail ordering remains unchanged.
    front = TargetGraphSet(
        {replica: tuple(replace(op, deps=(*op.deps, "late_front"))
                        if replica == 0 and op.op_id == "s0.l0.dispatch"
                        else op for op in graphs[replica])
         for replica in graphs},
        kernels=graphs[0].effective_kernels,
        selection=graphs[0].default_selection)
    front_report = graph_costs(
        raw, front, external_inputs={"analysis_input": 0, "late_front": 30})
    front_by_id = {row["op_id"]: row for row in front_report["timeline"]
                   if row.get("replica", 0) == 0}
    assert front_by_id["s0.l0.expert_up"]["ready_ms"] >= 30
    assert front_by_id["s0.l0.expert_up_overlap"]["ready_ms"] >= 30
    assert front_by_id["s0.l0.expert_act"]["ready_ms"] >= 30
    assert front_report["execution"]["live_tasks"] == 0


def test_deepep_bf16_uses_real_input_views_without_a_quantize_node():
    raw, graphs, ops = _expert_up_ops(
        _moE_case(layers=1, prompt=8, top_k=1, quant="bf16"),
        tokens=8, overlap=0.5)
    assert not any("expert_up" in key and key.endswith(".quantize")
                   for key in ops)
    assert ops["s0.l0.expert_up"].deps == ("s0.l0.expert_up.rows0_4",)
    assert ops["s0.l0.expert_up_overlap"].deps == ("s0.l0.expert_up.rows4_8",)
    assert ops["s0.l0.expert_up.rows0_4"].dtype == "bf16"
    assert ops["s0.l0.expert_up.rows4_8"].dtype == "bf16"
    _assert_internal_deps(ops)
    report = graph_costs(raw, graphs)
    assert report["execution"]["live_tasks"] == 0

    raw4, graphs4, ops4 = _expert_up_ops(
        _moE_case(layers=1, prompt=4, top_k=1, quant="bf16"),
        tokens=4, overlap=0.5)
    assert not any("expert_up" in key and key.endswith(".quantize")
                   for key in ops4)
    assert ops4["s0.l0.expert_up.rows0_2"].local_shape == (2, 32)
    assert ops4["s0.l0.expert_up.rows2_4"].local_shape == (2, 32)
    assert produced_values(ops4["s0.l0.expert_up.rows0_2"])[0][
        "logical_shape"] == [2, 32]
    assert graph_costs(raw4, graphs4)["execution"]["live_tasks"] == 0


def test_deepep_bf16_to_int8_conversion_reads_segment_views_only():
    dispatch = vector_op(
        op_id="dispatch", deps=(), stage=0, layer=0, requests=("r",),
        elements=0, ops_per_element=0, vector_ops_override=0, dtype="bf16",
        family="dispatch", phase="prefill", recipe="row_source",
        shape=(8, 32), read_bytes_override=0, write_bytes_override=0,
        metadata=(("output_format", "bf16"), ("materialized", True)))
    tail = replace(dispatch, op_id="dispatch_overlap")
    gate = OpSpec(
        op_id="gate", family="grouped_gemm", phase="prefill", stage=0,
        layer=0, requests=("r",), semantic_shape=(8, 64, 32),
        local_shape=(8, 64, 32), padded_shape=(8, 64, 32),
        launch_shape=(8, 64, 32), deps=("dispatch",),
        recipe="grouped_gate_up", dtype="int8",
        returns=("activation_scales",),
        metadata=(("program_kind", "matmul"), ("activation_format", "int8"),
                  ("input_format", "bf16"), ("expected_rows", 8),
                  ("graph_active_query_rows", 8),
                  ("activation_scale_bytes", 32),
                  ("activation_scale_shape", (8,)),
                  ("activation_scale_format", "fp32"),
                  ("activation_group_size", 32)))
    parts, _last = deepep.split_deepep_up_overlap(
        gate, {"deepep": True, "overlap": 0.5},
        dispatch_overlap_id="dispatch_overlap", active_rows=8)
    converted = activation_quantization((dispatch, tail, *parts), "bf16")
    by_id = {op.op_id: op for op in converted}
    first = by_id["gate.quantize"]
    second = by_id["gate_overlap.quantize"]
    assert first.deps == ("gate.rows0_4",)
    assert second.deps == ("gate.rows4_8",)
    assert first.mte2_bytes == pytest.approx(256)
    assert second.mte2_bytes == pytest.approx(256)
    assert first.mte3_bytes == pytest.approx(144)
    assert second.mte3_bytes == pytest.approx(144)
    assert produced_values(by_id["gate.rows0_4"])[0]["logical_shape"] == [4, 32]
    assert produced_values(by_id["gate.rows4_8"])[0]["logical_shape"] == [4, 32]
    assert produced_values(by_id["gate.rows4_8"])[0]["layout"]["offset"] == 4 * 32
    assert dict(by_id["gate.rows4_8"].metadata)["byte_offset"] == 4 * 32 * 2


def test_deepep_bf16_to_int8_five_three_integer_control():
    dispatch = vector_op(
        op_id="dispatch", deps=(), stage=0, layer=0, requests=("r",),
        elements=0, ops_per_element=0, vector_ops_override=0, dtype="bf16",
        family="dispatch", phase="prefill", recipe="row_source",
        shape=(8, 32), read_bytes_override=0, write_bytes_override=0,
        metadata=(("output_format", "bf16"), ("materialized", True)))
    tail = replace(dispatch, op_id="dispatch_overlap")
    gate = OpSpec(
        op_id="gate", family="grouped_gemm", phase="prefill", stage=0,
        layer=0, requests=("r",), semantic_shape=(8, 64, 32),
        local_shape=(8, 64, 32), padded_shape=(8, 64, 32),
        launch_shape=(8, 64, 32), deps=("dispatch",),
        recipe="grouped_gate_up", dtype="int8",
        returns=("activation_scales",),
        metadata=(("program_kind", "matmul"), ("activation_format", "int8"),
                  ("input_format", "bf16"), ("expected_rows", 8),
                  ("graph_active_query_rows", 8),
                  ("activation_scale_bytes", 32),
                  ("activation_scale_shape", (8,)),
                  ("activation_scale_format", "fp32"),
                  ("activation_group_size", 32)))
    parts, _last = deepep.split_deepep_up_overlap(
        gate, {"deepep": True, "overlap": 0.4},
        dispatch_overlap_id="dispatch_overlap", active_rows=8)
    converted = activation_quantization((dispatch, tail, *parts), "bf16")
    by_id = {op.op_id: op for op in converted}
    first = by_id["gate.quantize"]
    second = by_id["gate_overlap.quantize"]
    assert first.mte2_bytes == pytest.approx(320)
    assert second.mte2_bytes == pytest.approx(192)
    assert first.mte3_bytes == pytest.approx(180)
    assert second.mte3_bytes == pytest.approx(108)
    assert dict(first.metadata)["activation_scale_bytes"] == pytest.approx(20)
    assert dict(second.metadata)["activation_scale_bytes"] == pytest.approx(12)
    assert produced_values(by_id["gate.rows5_8"])[0]["layout"]["offset"] == 5 * 32
    assert dict(by_id["gate.rows5_8"].metadata)["byte_offset"] == 5 * 32 * 2


def test_deepep_normal_four_query_topk_two_keeps_eight_assignment_rows():
    raw, graphs, ops = _expert_up_ops(
        _moE_case(layers=1, prompt=4, top_k=2), tokens=4, overlap=0.5)
    assert ops["s0.l0.pack"].local_shape == (4, 2)
    assert ops["s0.l0.expert_up"].local_shape == (4, 64, 32)
    assert ops["s0.l0.expert_up_overlap"].local_shape == (4, 64, 32)
    assert ops["s0.l0.expert_up"].flops == pytest.approx(16384)
    assert ops["s0.l0.expert_up_overlap"].flops == pytest.approx(16384)
    segments = dict(ops["s0.l0.expert_act"].metadata)["activation_input_segments"]
    assert segments == (
        ("s0.l0.expert_up", 0, 4, 4, 64),
        ("s0.l0.expert_up_overlap", 4, 8, 4, 64))
    _assert_internal_deps(ops)
    assert graph_costs(raw, graphs)["execution"]["live_tasks"] == 0


def test_deepep_single_assignment_row_is_not_split():
    raw, graphs, ops = _expert_up_ops(
        _moE_case(layers=1, prompt=1, top_k=1), tokens=1, overlap=0.5)
    assert "s0.l0.expert_up_overlap" not in ops
    assert "s0.l0.expert_up.quantize" in ops
    assert graph_costs(raw, graphs)["execution"]["live_tasks"] == 0


def test_deepep_transported_quantized_input_reuses_the_original_producer():
    raw, graphs, ops = _expert_up_ops(
        _moE_case(layers=1, prompt=8, top_k=1), tokens=8, overlap=0.5)
    raw["runtime"]["kernels"] = {"moe_transport": {
        "kind": "separate_quantized_payload", "dispatch_dtype": "int8",
        "scale_dtype": "fp32"}}
    graphs = target_graphs(raw, phase="prefill", tokens=8, scope="model")
    ops = {op.op_id: op for op in graphs[0]}
    assert not any("expert_up" in key and key.endswith(".quantize")
                   for key in ops)
    gate = ops["s0.l0.expert_up"]
    assert dict(gate.metadata)["activation_producer"] == \
        "explicit_transport_quantization"
    assert "s0.l0.moe_payload_dispatch" in gate.deps
    assert "s0.l0.moe_scale_dispatch" in gate.deps
    assert "s0.l0.moe_payload_dispatch_overlap" in gate.deps
    _assert_internal_deps(ops)
    report = graph_costs(raw, graphs)
    assert report["execution"]["live_tasks"] == 0


def test_deepep_overlap_preserves_other_causal_edges_and_rejects_bad_scale():
    raw = _moE_case(layers=1)
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e6
    raw["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.4}
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    report = graph_costs(raw, graphs)
    by_id = {row["op_id"]: row for row in report["timeline"] if row.get("replica", 0) == 0}
    assert by_id["s0.l0.expert_act"]["ready_ms"] >= \
        by_id["s0.l0.expert_up"]["complete_ms"]
    assert by_id["s0.l0.expert_act"]["ready_ms"] >= \
        by_id["s0.l0.expert_up_overlap"]["complete_ms"]
    assert by_id["s0.l0.expert_up_overlap"]["ready_ms"] >= \
        by_id["s0.l0.dispatch_overlap"]["complete_ms"]
    assert by_id["s0.l0.combine_overlap"]["ready_ms"] >= \
        by_id["s0.l0.expert_down"]["complete_ms"]
    assert by_id["s0.l0.combine"]["ready_ms"] >= \
        by_id["s0.l0.expert_down_overlap"]["complete_ms"]

    slow_link = _moE_case(layers=1)
    slow_link["machine"]["links"][0]["capacity_bytes_per_ms"] = 1.0
    slow_link["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.4}
    slow_graphs = target_graphs(slow_link, phase="prefill", tokens=4, scope="model")
    slow_report = graph_costs(slow_link, slow_graphs)
    slow_by_id = {row["op_id"]: row for row in slow_report["timeline"]
                  if row.get("replica", 0) == 0}
    assert slow_by_id["s0.l0.expert_up_overlap"]["ready_ms"] >= \
        slow_by_id["s0.l0.dispatch_overlap"]["complete_ms"]

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


def _acceptance_graph(overlap, *, prompt=8, top_k=1):
    raw = _moE_case(layers=1, prompt=prompt, top_k=top_k)
    raw["runtime"]["metadata"] = {"enable_deepep": True,
                                  "deepep_overlap_ratio": overlap}
    graphs = target_graphs(raw, phase="prefill", tokens=prompt, scope="model")
    return raw, graphs, {op.op_id: op for op in graphs[0]}


def test_deepep_received_value_views_share_the_full_receive_allocation():
    controls = (
        (0.5, (4, 32), (4, 32), 0, 256, 128.0, 128.0, 256.0, 256.0),
        (0.4, (5, 32), (3, 32), 0, 320, 160.0, 96.0, 320.0, 192.0),
    )
    for (overlap, first_shape, second_shape, first_byte, second_byte,
         first_link, second_link, first_payload, second_payload) in controls:
        raw, graphs, ops = _acceptance_graph(overlap)
        main = ops["s0.l0.dispatch"]
        tail = ops["s0.l0.dispatch_overlap"]
        assert main.link_bytes == pytest.approx(first_link)
        assert tail.link_bytes == pytest.approx(second_link)
        assert dict(main.metadata)["deepep_payload_edge_bytes"] == 512
        assert dict(main.metadata)["deepep_chunk_payload_bytes"] == first_payload
        assert dict(tail.metadata)["deepep_chunk_payload_bytes"] == second_payload
        assert dict(main.metadata)["deepep_chunk_rows"] == first_shape[0]
        assert dict(tail.metadata)["deepep_chunk_rows"] == second_shape[0]
        first_view = ops[f"s0.l0.expert_up.rows0_{first_shape[0]}"]
        second_view = ops[f"s0.l0.expert_up.rows{first_shape[0]}_8"]
        first_value = produced_values(first_view)[0]
        second_value = produced_values(second_view)[0]
        assert tuple(first_value["logical_shape"]) == first_shape
        assert tuple(second_value["logical_shape"]) == second_shape
        # The receive allocation is one 512 B BF16 value; both row views keep
        # the complete storage, the same alias/binding/version and add no write.
        for value, byte_offset in ((first_value, first_byte),
                                   (second_value, second_byte)):
            assert tuple(value["storage_shape"]) == (8, 32)
            assert value["storage_bytes"] == 512
            assert value["alias_of"] == "s0.l0.dispatch"
            assert value["storage"] == "moe.rx:s0.l0.dispatch"
            assert value["binding"] == "static"
            assert value["binding_name"] == "s0.l0.dispatch.rx"
            assert value["state_version"] is None
            assert value["materialized"] is False
            assert value["write_bytes"] == 0
            assert value["layout"]["offset"] * dtype_bytes("bf16") == byte_offset
            assert value["layout"]["storage_shape"] == (8, 32)
        assert dict(first_view.metadata)["byte_offset"] == first_byte
        assert dict(second_view.metadata)["byte_offset"] == second_byte
        gate_first = ops["s0.l0.expert_up"]
        gate_second = ops["s0.l0.expert_up_overlap"]
        assert gate_first.flops == 2 * first_shape[0] * 64 * 32
        assert gate_second.flops == 2 * second_shape[0] * 64 * 32
        assert gate_first.mte3_bytes == first_shape[0] * 64 * 2
        assert gate_second.mte3_bytes == second_shape[0] * 64 * 2
        act = ops["s0.l0.expert_act"]
        assert act.deps == ("s0.l0.expert_up", "s0.l0.expert_up_overlap")
        assert dict(act.metadata)["activation_input_bytes"] == 1024
        report = graph_costs(raw, graphs)
        assert report["execution"]["live_tasks"] == 0
        assert not any(report["execution"]["live_allocated"].values())


def test_deepep_full_graph_conversion_uses_each_segment_row_axis():
    """Public conversion port over the full target_graphs path.

    ``activation_quantization`` must consume the row view's own logical rows;
    the provider row axis is not the query axis.  This is the joint T11
    control: F-MOE supplies the value, T11 must not overwrite it with
    ``graph_active_query_rows``.
    """
    for overlap, reads, writes, scales, shapes in (
            (0.5, (256, 256), (144, 144), (16, 16), ((4, 32), (4, 32))),
            (0.4, (320, 192), (180, 108), (20, 12), ((5, 32), (3, 32)))):
        raw, graphs, ops = _acceptance_graph(overlap)
        first = ops["s0.l0.expert_up.quantize"]
        second = ops["s0.l0.expert_up_overlap.quantize"]
        assert first.mte2_bytes == pytest.approx(reads[0])
        assert second.mte2_bytes == pytest.approx(reads[1])
        assert first.mte3_bytes == pytest.approx(writes[0])
        assert second.mte3_bytes == pytest.approx(writes[1])
        first_meta, second_meta = dict(first.metadata), dict(second.metadata)
        assert tuple(first_meta["activation_scale_shape"]) == (shapes[0][0],)
        assert tuple(second_meta["activation_scale_shape"]) == (shapes[1][0],)
        assert first_meta["activation_scale_bytes"] == pytest.approx(scales[0])
        assert second_meta["activation_scale_bytes"] == pytest.approx(scales[1])
        assert first.mte2_bytes + second.mte2_bytes == pytest.approx(512)
        assert first.mte3_bytes + second.mte3_bytes == pytest.approx(288)


def test_deepep_transport_chunks_gate_first_segment_only_after_its_rows():
    raw, graphs, ops = _acceptance_graph(0.4, prompt=8, top_k=1)
    delayed = TargetGraphSet(
        {replica: tuple(replace(op, deps=(*op.deps, "late_tail"))
                        if replica == 0 and op.op_id == "s0.l0.dispatch_overlap"
                        else op for op in graphs[replica])
         for replica in graphs},
        kernels=graphs[0].effective_kernels,
        selection=graphs[0].default_selection)
    report = graph_costs(raw, delayed,
                         external_inputs={"analysis_input": 0, "late_tail": 50})
    by_id = {row["op_id"]: row for row in report["timeline"]
             if row.get("replica", 0) == 0}
    assert by_id["s0.l0.expert_up.quantize"]["complete_ms"] < 50
    assert by_id["s0.l0.expert_up"]["complete_ms"] < 50
    assert by_id["s0.l0.expert_up_overlap.quantize"]["ready_ms"] >= 50
    assert by_id["s0.l0.expert_up_overlap"]["ready_ms"] >= 50
    assert by_id["s0.l0.expert_act"]["ready_ms"] >= \
        by_id["s0.l0.expert_up_overlap"]["complete_ms"]
    assert report["execution"]["live_tasks"] == 0


def test_deepep_receive_and_gate_outputs_expose_real_last_use_accesses():
    from mindie_sim.prediction.execution_state import bind_op_accesses

    _raw, _graphs, ops = _acceptance_graph(0.4, prompt=8, top_k=1)
    first_view, second_view = (ops["s0.l0.expert_up.rows0_5"],
                              ops["s0.l0.expert_up.rows5_8"])
    first_reader = ops["s0.l0.expert_up.quantize"]
    second_reader = ops["s0.l0.expert_up_overlap.quantize"]
    first_gate, second_gate = (ops["s0.l0.expert_up"],
                              ops["s0.l0.expert_up_overlap"])
    act, down = ops["s0.l0.expert_act"], ops["s0.l0.expert_down"]

    def one(op):
        accesses = bind_op_accesses(None, op)
        assert len(accesses) == 1
        return accesses[0].access

    receive_first = one(first_view)
    receive_second = one(second_view)
    assert receive_first.direction == receive_second.direction == "alias"
    assert receive_first.bytes == receive_second.bytes == 0
    assert receive_first.span == ((0, 5),)
    assert receive_second.span == ((5, 8),)
    assert receive_first.storage == receive_second.storage == "moe.rx:s0.l0.dispatch"
    assert receive_first.last_use == "s0.l0.expert_up.quantize"
    assert receive_second.last_use == "s0.l0.expert_up_overlap.quantize"

    write_first = one(first_gate)
    write_second = one(second_gate)
    assert write_first.direction == write_second.direction == "write"
    assert write_first.bytes == 5 * 64 * 2
    assert write_second.bytes == 3 * 64 * 2
    assert write_first.span == ((0, 5),)
    assert write_second.span == ((0, 3),)
    assert write_first.last_use == write_second.last_use == "s0.l0.expert_act"

    activation_reads = bind_op_accesses(None, act)
    assert len(activation_reads) == 2
    assert {item.access.producer for item in activation_reads} == {
        "s0.l0.expert_up", "s0.l0.expert_up_overlap"}
    assert sum(item.access.bytes for item in activation_reads) == 1024
    assert {item.access.last_use for item in activation_reads} == {
        "s0.l0.expert_down"}

    users = {}
    for op in ops.values():
        for dependency in op.deps:
            users.setdefault(dependency, set()).add(op.op_id)
    assert users[first_reader.op_id] == {"s0.l0.expert_up"}
    assert users[second_reader.op_id] == {"s0.l0.expert_up_overlap"}
    assert users[first_gate.op_id] == {"s0.l0.expert_act"}
    assert users[second_gate.op_id] == {"s0.l0.expert_act"}
    assert users[act.op_id] <= {"s0.l0.expert_down",
                                "s0.l0.expert_down.quantize"}
    assert "s0.l0.expert_down" in users[act.op_id] or \
        "s0.l0.expert_down.quantize" in users[act.op_id]
    assert first_view.op_id not in users or users[first_view.op_id] == {
        first_reader.op_id}


def test_deepep_explicit_bank_segments_keep_slots_and_conserve_rows():
    assert deepep.expert_bank_segments((2, 2), 3) == ((2, 1), (0, 1))
    assert deepep.expert_bank_segments((1, 0, 3), 2) == ((1, 0, 1), (0, 0, 2))
    assert deepep.expert_bank_segments((0, 4), 0) == ((0, 0), (0, 4))
    main, tail = deepep.expert_bank_segments((2, 2), 3)
    assert len(main) == len(tail) == 2
    assert sum(main) + sum(tail) == 4

    realization = {
        "basis": "explicit_token_routing_integer",
        "scope": "body", "layer": 0, "phase": "prefill", "tokens": 4,
        "expert_rows": (2, 2, 2, 2), "global_expert_rows": (2, 2, 2, 2),
        "expert_ids": (0, 1, 2, 3),
        "local_expert_rows": (2, 2), "local_weight_expert_ids": (0, 1),
        "representative_expert_ids": (0, 1), "representative_rank": 0,
        "expert_placement": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "rank_expert_rows": ((2, 2), (2, 2)),
        "peer_assignments": ((0, 2), (2, 0)),
        "source_owners": (0, 0, 1, 1), "valid_local_tokens": 2,
        "rows": 4, "active": 2,
    }
    tail_op = vector_op(
        op_id="tail", deps=(), stage=0, layer=0, requests=("r",),
        elements=8, ops_per_element=1, dtype="bf16", family="dispatch",
        phase="prefill", recipe="row_source", shape=(8, 32),
        metadata=(("output_format", "bf16"), ("materialized", True)))
    tail = replace(tail_op, op_id="dispatch_overlap")
    gate = OpSpec(
        op_id="expert_up", family="grouped_gemm", phase="prefill", stage=0,
        layer=0, requests=("r",), semantic_shape=(4, 64, 32),
        local_shape=(4, 64, 32), padded_shape=(4, 64, 32),
        launch_shape=(4, 64, 32), deps=("tail",), recipe="grouped_gate_up",
        dtype="bf16", metadata=(
            ("program_kind", "matmul"), ("activation_format", "bf16"),
            ("input_format", "bf16"), ("output_format", "bf16"),
            ("weight_format", "bf16"), ("expected_rows", 4),
            ("activation_scale_bytes", 0), ("routing_basis",
             "explicit_token_routing_integer"),
            ("routing_realization", realization),
            ("packed_weight_bytes", 4 * 2 * 32 * 32 * 2)))
    parts, last = deepep.split_deepep_up_overlap(
        gate, {"deepep": True, "overlap": 0.5},
        dispatch_overlap_id="dispatch_overlap", active_rows=4,
        row_cut=(3, 1))
    by_id = {op.op_id: op for op in parts}
    first_gate = by_id["expert_up"]
    second_gate = by_id["expert_up_overlap"]
    first_real = dict(first_gate.metadata)["routing_realization"]
    second_real = dict(second_gate.metadata)["routing_realization"]
    assert first_real["local_expert_rows"] == (2, 1)
    assert second_real["local_expert_rows"] == (0, 1)
    assert sum(first_real["local_expert_rows"]) == 3
    assert sum(second_real["local_expert_rows"]) == 1
    assert dict(first_gate.metadata)["matrix_group_rows"] == (2, 1)
    assert dict(second_gate.metadata)["matrix_group_rows"] == (0, 1)
    from mindie_sim.prediction.family_grouped_gemm import _explicit_bank
    counts, group_ids, physical_ids, global_rows, global_slice = _explicit_bank(
        first_real, 3, 64, 32)
    assert counts == (2, 1)
    assert group_ids == (0, 1)
    assert physical_ids == (0, 1)
    assert global_rows == (2, 2, 2, 2)
    assert global_slice == (2, 2)


def _service_route_case(output_tokens=4):
    raw = _moE_case(dp=1, ep=1, layers=1, prompt=1, top_k=1)
    raw["runtime"]["metadata"] = {}
    raw["runtime"]["graph_mode"] = "capture"
    raw["runtime"]["graph_capture_mode"] = "full"
    raw["runtime"]["graph_capture_keys"] = [4]
    routes = (
        ("decode", 1, [1, 0, 0, 0], [[0]]),
        ("decode", 2, [0, 1, 0, 0], [[1]]),
        ("decode", 3, [1, 0, 0, 0], [[0]]),
    )
    raw["runtime"]["routing"] = {"realizations": [
        {"scope": "body", "layer": 0, "phase": phase, "tokens": 1,
         "expert_rows": rows, "topk_ids": ids,
         "expert_placement": [[0, 0], [0, 1], [0, 2], [0, 3]],
         "source_owners": [0], "query_position": position}
        for phase, position, rows, ids in routes]}
    raw["workload"]["requests"][0]["output_tokens"] = output_tokens
    return raw, routes


def test_deepep_service_rebinds_two_routes_on_template_hits(monkeypatch):
    from mindie_sim import predict

    raw, routes = _service_route_case(output_tokens=4)
    expected = [(position, tuple(rows))
                for _phase, position, rows, _ids in routes]
    selected = []
    from mindie_sim.prediction import frontend as frontend_module

    original = frontend_module.select_realization

    def recorded(runtime, layer, **kwargs):
        realization = original(runtime, layer, **kwargs)
        if kwargs.get("phase") == "decode":
            selected.append((
                int(kwargs.get("query_position")),
                tuple(realization.expert_rows) if realization is not None else None,
                int(kwargs.get("tokens")), int(kwargs.get("valid_tokens"))))
        return realization

    monkeypatch.setattr(frontend_module, "select_realization", recorded)
    report = predict(raw, trace_limit=16)

    # The physical graph is the captured bucket 4; each decode iteration must
    # carry its own logical route and template hit, not the first route.
    assert selected == [
        (position, rows, 4, 1) for position, rows in expected]
    decode_traces = [item for item in report["traces"]
                     if item["graph_bucket"] is not None]
    assert len(decode_traces) == 3
    assert {item["graph_bucket"] for item in decode_traces} == {4}
    assert {item["padded_tokens"] for item in decode_traces} == {4}
    assert {item["operators"] for item in decode_traces} == {27}

    service_totals = {"flops": 0.0}
    model_durations = []
    for position, rows in expected:
        graphs = target_graphs(raw, phase="decode", tokens=1, context=position,
                               scope="model")
        ops = {op.op_id: op for op in graphs[0]}
        gate = ops["s0.l0.expert_up"]
        realization = dict(gate.metadata)["routing_realization"]
        assert tuple(realization["local_expert_rows"]) == tuple(rows)
        assert dict(gate.metadata)["graph_active_query_rows"] == 1
        assert dict(gate.metadata)["graph_physical_query_rows"] == 4
        assert gate.local_shape[0] == 1
        costs = graph_costs(raw, graphs)
        service_totals["flops"] += costs["execution"]["work"]["flops"]
        model_durations.append(costs["duration_ms"])
    prefill = graph_costs(
        raw, target_graphs(raw, phase="prefill", tokens=1, scope="model"))
    service_totals["flops"] += prefill["execution"]["work"]["flops"]
    assert report["execution"]["work"]["flops"] == pytest.approx(
        service_totals["flops"])

    offsets = [trace["duration_ms"] - duration
               for trace, duration in zip(decode_traces, model_durations)]
    assert max(offsets) - min(offsets) == pytest.approx(0.0, abs=1e-12)
    # The second route's graph duration differs from the first by exactly the
    # same amount in the service trace and in the model entry.
    assert (decode_traces[1]["duration_ms"] - decode_traces[0]["duration_ms"]) \
        == pytest.approx(model_durations[1] - model_durations[0], abs=1e-12)
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
