"""Selected MoE quantized transport: bytes, rows, causality and public execution."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.peer_traffic import PeerTraffic
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


SELECTOR = {
    "kind": "separate_quantized_payload", "payload_dtype": "int8",
    "scale_dtype": "fp32", "scale_granularity": "row", "histogram_dtype": "int64",
}


def transport_case(*, dp=2, tp=1, ep=2, tokens=4, selected=True):
    raw = case(dp=dp, tp=tp, ep=ep, prompt=tokens, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=2, hidden_size=32,
        moe_intermediate_size=32, activation_dtype="bf16",
        quant=asdict(quant_preset("w8a8_int8")),
    )
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e9
    if selected:
        raw["runtime"]["kernels"] = {"moe_transport": dict(SELECTOR)}
    return raw


def _one(ops, recipe, *, scope=None):
    rows = [op for op in ops if op.recipe == recipe
            and (scope is None or dict(op.metadata).get("quantization_scope") == scope)]
    assert len(rows) == 1, [(op.op_id, op.recipe) for op in rows]
    return rows[0]


def test_public_selector_preserves_default_and_executes_exact_protocol_bytes():
    default_raw = transport_case(selected=False)
    default = target_graphs(default_raw, phase="prefill", tokens=4, scope="model")
    raw = transport_case(selected=True)
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    assert not any(op.family in {"histogram", "allgather"} for op in default[0])
    ops = graphs[0]
    pack = _one(ops, "route_pack")
    histogram = _one(ops, "moe_expert_count_histogram")
    gather = _one(ops, "ring_allgather")
    quantize = _one(ops, "dynamic_quantize", scope="moe_transport_expert_input")
    scale = _one(ops, "expert_scale_dispatch")
    payload = _one(ops, "expert_quantized_dispatch")
    gate = _one(ops, "grouped_gate_up")
    down = _one(ops, "grouped_down")
    combine = _one(ops, "expert_combine")

    # Four local tokens * top-k 2 = eight source rows; E=4.
    assert (histogram.mte2_bytes, histogram.mte3_bytes, histogram.vector_ops) == (32, 32, 8)
    assert dict(histogram.metadata)["histogram_dtype"] == "int64"
    assert gather.link_bytes == 32 == (2 - 1) * 4 * 8
    assert quantize.deps == (pack.op_id,)
    assert (quantize.mte2_bytes, quantize.mte3_bytes) == (512, 288)
    assert quantize.vector_ops == 4 * 8 * 32 + 2 * 8
    assert scale.link_bytes == 16 == 8 * 4 * (2 - 1) / 2
    assert payload.link_bytes == 128 == 8 * 32 * 1 * (2 - 1) / 2
    assert dict(scale.metadata)["payload_dtype"] == "fp32"
    assert dict(payload.metadata)["payload_dtype"] == "int8"
    assert dict(payload.metadata)["row_bytes"] == 32
    assert combine.link_bytes == 256 == 8 * 32 * 2 * (2 - 1) / 2
    assert dict(gate.metadata)["activation_format"] == "int8"
    assert dict(down.metadata)["output_format"] == "bf16"

    # Exactly one expert-input quantizer; the ordinary down input still has one.
    quantizers = [op for op in ops if op.family == "quantize"
                  and dict(op.metadata).get("quantization_scope")
                  == "moe_transport_expert_input"]
    assert len(quantizers) == 1
    down_quantize = next(op for op in ops if op.op_id == down.op_id + ".quantize")
    assert down.deps == (down_quantize.op_id,)
    assert down_quantize.deps == ("s0.l0.expert_act",)
    assert set(gate.deps) == {quantize.op_id, scale.op_id, payload.op_id}
    assert histogram.deps == ("s0.l0.topk",)
    assert gather.deps == (histogram.op_id,)
    assert gather.op_id in pack.deps
    assert scale.deps == (quantize.op_id, gather.op_id)
    assert payload.deps == (quantize.op_id, scale.op_id, gather.op_id)
    assert combine.deps == (down.op_id,)

    default_cost = graph_costs(raw, default)
    selected_cost = graph_costs(raw, graphs)
    assert default_cost["execution"]["live_tasks"] == 0
    assert selected_cost["execution"]["live_tasks"] == 0
    assert selected_cost["duration_ms"] > 0
    result = predict(raw, event_limit=100)
    default_result = predict(default_raw, event_limit=100)
    assert result["status"] == default_result["status"] == "modeled"
    assert result["requests"][0]["generated_tokens"] == 1
    assert default_result["requests"][0]["generated_tokens"] == 1
    assert result["execution"]["live_tasks"] == default_result["execution"]["live_tasks"] == 0


def test_body_transport_selection_preserves_separate_bf16_mtp_protocol():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "tests/fixtures/verification/"
                     "v4flash-communication-case.json").read_text())
    raw["runtime"]["kernels"]["moe_transport"] = dict(SELECTOR)
    parsed = parse_case(raw)
    view = RequestView("one", 4, 0, True, "prefill", draft_prepare=True)
    frontend = Frontend()
    # An unscoped selection must not quietly become BF16 in the MTP layer.
    with pytest.raises(ModelingError, match="requires_int8_activations"):
        frontend(IterationPlan((view,)), parsed.model, parsed.runtime, parsed.machine)
    raw["runtime"]["kernels"]["moe_transport"]["layer_scopes"] = ["body"]
    parsed = parse_case(raw)
    work = frontend(IterationPlan((view,)), parsed.model, parsed.runtime, parsed.machine)
    target_dispatch = [op for op in work.ops if op.recipe == "expert_quantized_dispatch"]
    draft_dispatch = [op for op in work.ops
                      if op.recipe == "expert_dispatch" and op.phase == "draft_prefill"]
    assert len(target_dispatch) == 43 and len(draft_dispatch) == 1
    assert all(op.phase == "prefill" for op in target_dispatch)
    assert all(dict(op.metadata)["payload_dtype"] == "int8" for op in target_dispatch)
    assert draft_dispatch[0].link_bytes == 4 * 6 * 4096 * 2 / 8 * 7 / 8
    assert len([op for op in work.ops if op.recipe == "expert_combine"]) == 44


@pytest.mark.parametrize("mutation,message", [
    (lambda raw: raw["model"]["quant"].update(activation_group_size=16),
     "requires_row_scale_width"),
    (lambda raw: raw["model"].update(quant=asdict(quant_preset("w8a8_mxfp8"))),
     "requires_int8_activations"),
])
def test_source_quant_profile_incompatibility_is_visible(mutation, message):
    raw = transport_case()
    mutation(raw)
    with pytest.raises(ModelingError, match=message):
        target_graphs(raw, phase="prefill", tokens=4, scope="model")


def test_unsupported_selector_is_an_error_not_a_silent_default():
    raw = transport_case()
    raw["runtime"]["kernels"]["moe_transport"]["payload_dtype"] = "fp16"
    with pytest.raises(ModelingError, match="moe_transport_unsupported:payload_dtype"):
        target_graphs(raw, phase="prefill", tokens=4, scope="model")
    raw = transport_case()
    raw["runtime"]["kernels"]["moe_transport"]["extra"] = True
    with pytest.raises(ModelingError, match="moe_transport_unknown_fields"):
        target_graphs(raw, phase="prefill", tokens=4, scope="model")
    raw = transport_case()
    raw["runtime"]["moe_recipe"] = "mc2"
    with pytest.raises(ModelingError, match="moe_transport_requires_alltoall_recipe"):
        target_graphs(raw, phase="prefill", tokens=4, scope="model")


def test_tp_source_partition_has_one_quantizer_and_is_not_double_counted():
    graphs = target_graphs(transport_case(dp=1, tp=2, ep=2), phase="prefill",
                           tokens=4, scope="model")
    ops = graphs[0]
    histogram = _one(ops, "moe_expert_count_histogram")
    quantize = _one(ops, "dynamic_quantize", scope="moe_transport_expert_input")
    payload = _one(ops, "expert_quantized_dispatch")
    scale = _one(ops, "expert_scale_dispatch")
    assert histogram.mte2_bytes == 16
    assert histogram.vector_ops == 4
    assert histogram.mte3_bytes == 32
    assert dict(quantize.metadata)["source_rows"] == 4.0
    assert quantize.mte2_bytes == 4 * 32 * 2
    assert quantize.mte3_bytes == 4 * 32 + 4 * 4
    assert payload.link_bytes == 64
    assert scale.link_bytes == 8
    assert len([op for op in ops if dict(op.metadata).get("quantization_scope")
                == "moe_transport_expert_input"]) == 1


@pytest.mark.parametrize("dp,tp,ep", [(2, 1, 2), (1, 2, 2), (1, 8, 8)])
def test_balanced_global_counts_never_become_invented_source_histograms(dp, tp, ep):
    raw = transport_case(dp=dp, tp=tp, ep=ep, tokens=8)
    raw["model"].update(num_experts=8, num_heads=8, num_kv_heads=8, head_dim=4)
    raw["runtime"]["routing"] = {"policy": "balanced"}
    graphs = target_graphs(raw, phase="prefill", tokens=8, scope="model")
    histogram = _one(graphs[0], "moe_expert_count_histogram")
    meta = dict(histogram.metadata)
    assert meta["counts"] is None
    assert sum(meta["global_expert_counts"]) == 8 * dp * 2
    assert histogram.mte3_bytes == 8 * 8
    assert meta["source_assignments"] == 8 * 2 / min(tp, ep)
    gate = _one(graphs[0], "balanced_grouped_gate_up")
    assert gate.local_shape[0] == 8 * dp * 2 / ep
    report = graph_costs(raw, graphs)
    assert report["execution"]["live_tasks"] == 0


def _uneven_graphs():
    raw = transport_case(dp=3, tp=1, ep=3, tokens=4)
    raw["model"]["num_experts"] = 6
    # Different local token volumes and one completely idle cohort member.
    raw["workload"]["requests"] = [
        {"id": "a", "prompt_tokens": 4, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 2, "output_tokens": 1},
    ]
    raw["runtime"]["kernels"] = {
        "moe_transport": dict(SELECTOR),
        "families": {
            "allgather": {"kind": "ring_collective"},
            "alltoall": {"kind": "pairwise_collective"},
        },
    }
    case_spec = parse_case(raw)
    sizes = (4, 2, 0)
    views = {rank: (RequestView(f"r{rank}", size, 0, False, "prefill"),)
             for rank, size in enumerate(sizes)}
    graphs = {}
    for rank, group in views.items():
        ops, _ = Frontend().body(
            case_spec.model, case_spec.runtime, group, ready="analysis_input",
            tokens=sizes[rank], mode="prefill", global_tokens=sum(sizes),
        )
        graphs[rank] = lower_implementations(
            ops, case_spec.runtime, case_spec.model.activation_dtype,
            resource_fusion=False, passes=None)
    return case_spec, bind_cohort_traffic(case_spec, graphs), sizes


def test_complete_unequal_cohort_reverses_rows_at_bf16_width_with_idle_rank():
    case_spec, graphs, sizes = _uneven_graphs()
    for rank, size in enumerate(sizes):
        ops = graphs[rank]
        histogram = _one(ops, "moe_expert_count_histogram")
        gather = _one(ops, "ring_allgather")
        quantize = _one(ops, "dynamic_quantize", scope="moe_transport_expert_input")
        payload = _one(ops, "expert_quantized_dispatch")
        combine = _one(ops, "expert_combine")
        assert histogram.mte3_bytes == 48
        assert gather.link_bytes == 96  # (EP-1) * E * 8, no TP division
        assert dict(quantize.metadata)["source_rows"] == float(size * 2)
        assert payload.link_bytes == pytest.approx(size * 2 * 32 * 2 / 3)
        assert combine.link_bytes == pytest.approx((256 / 3, 512 / 3, 256)[rank])
        assert dict(combine.metadata)["peer_traffic_row_format"] == {
            "elements": 32, "forward_dtype": "int8", "reverse_dtype": "bf16"}
        gate = _one(ops, "grouped_gate_up")
        # Uniform destinations receive four rows even when their source is
        # idle. Six total tokens * top-k2 / EP3 = four expert rows per rank.
        assert gate.local_shape[0] == 4
        assert gate.flops == 2 * 4 * 64 * 32
        assert dict(gate.metadata)["activation_payload_producer"] == payload.op_id
        assert not any(op.op_id == gate.op_id + ".quantize" for op in ops)
    report = graph_costs(case_spec, graphs, external_inputs={"analysis_input": 0},
                         trace_limit=200)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["duration_ms"] > 0
    assert any(row["program"] == "PairwiseCollective" for row in report["timeline"])
    assert any(row["program"] == "RingCollective" for row in report["timeline"])
    assert all(not row["service_before_input"] for row in report["isolated"]["operators"])


def test_reverse_explicit_peer_matrix_reencodes_rows_without_changing_forward():
    case_spec, graphs, _ = _uneven_graphs()
    # Directed useful INT8 bytes, width32. Source0 sends nothing but receives
    # three rows; the reverse must send those same rows as BF16.
    forward = ((0, 0, 0), (32, 0, 64), (64, 96, 0))
    descriptor = {"kind": "matrix", "payload_bytes": forward}
    raw = {}
    for rank, ops in graphs.items():
        send = _one(ops, "expert_quantized_dispatch")
        back = _one(ops, "expert_combine")
        send_meta = {**dict(send.metadata), "peer_traffic": descriptor}
        back_meta = dict(back.metadata)
        back_meta.pop("peer_traffic")
        raw[rank] = (
            replace(send, link_bytes=sum(forward[rank]), metadata=tuple(send_meta.items())),
            replace(back, metadata=tuple(back_meta.items())),
        )
    bound = bind_cohort_traffic(case_spec, raw)
    assert bind_cohort_traffic(case_spec, bound) == bound
    for rank, ops in bound.items():
        send, back = ops
        forward_plan = PeerTraffic.from_mapping(dict(send.metadata)["peer_traffic"], 3)
        reverse_plan = PeerTraffic.from_mapping(dict(back.metadata)["peer_traffic"], 3)
        for peer in range(3):
            assert forward_plan.payload(rank, peer) == forward[rank][peer]
            assert reverse_plan.payload(rank, peer) / 2 == forward[peer][rank]
        assert back.link_bytes == 2 * sum(forward[peer][rank] for peer in range(3))
    assert bound[0][1].link_bytes == 192
    assert sum(ops[0].link_bytes for ops in bound.values()) * 2 == sum(
        ops[1].link_bytes for ops in bound.values())


def test_asymmetric_source_readiness_reaches_each_transport_stage():
    case_spec, graphs, _ = _uneven_graphs()
    graphs = {
        rank: tuple(replace(op, deps=tuple(
            f"input{rank}" if dep == "analysis_input" else dep for dep in op.deps))
                    for op in ops)
        for rank, ops in graphs.items()
    }
    report = graph_costs(case_spec, graphs,
                         external_inputs={"input0": 0, "input1": 5, "input2": 0})
    indexed = {row["id"]: row for row in report["timeline"]}
    for rank in range(3):
        prefix = f"r{rank}:s0.l0."
        for after, before in (
            ("moe_histogram_gather", "moe_histogram"),
            ("pack", "moe_histogram_gather"),
            ("moe_transport_quantize", "pack"),
            ("moe_scale_dispatch", "moe_transport_quantize"),
            ("moe_payload_dispatch", "moe_scale_dispatch"),
            ("expert_up", "moe_payload_dispatch"),
            ("expert_combine", "expert_down"),
        ):
            assert indexed[prefix + after]["ready_ms"] >= indexed[prefix + before]["complete_ms"]
        assert indexed[prefix + "moe_histogram"]["ready_ms"] >= (5 if rank == 1 else 0)
    assert indexed["r0:s0.l0.moe_histogram"]["complete_ms"] < 5
    assert indexed["r2:s0.l0.expert_up"]["complete_ms"] > 5
    assert report["execution"]["live_tasks"] == 0
