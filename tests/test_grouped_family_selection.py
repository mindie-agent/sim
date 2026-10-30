"""Integer routing scenarios through ordinary analytical and source MoE paths."""
import pytest

from mindie_sim.prediction import predict
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case


def grouped_case(*, source=False):
    raw = case(prompt=4, output=3)
    raw["model"].update(layers=1, ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=32)
    raw["runtime"]["routing"] = {"realizations": [
        {"scope": "body", "layer": 0, "phase": "prefill", "tokens": 4,
         "expert_rows": [3, 0, 3, 2]},
        {"scope": "body", "layer": 0, "phase": "decode", "tokens": 1,
         "expert_rows": [1, 0, 1, 0]},
    ]}
    if source:
        raw["runtime"]["kernels"] = {
            "lowerings": {"grouped_gemm": "ascendc_grouped_nd_bf16"}}
    return raw


def grouped_rows(raw, *, phase="prefill", tokens=4):
    ops = [op for op in target_operators(raw, phase=phase, tokens=tokens)
           if op.family == "grouped_gemm"]
    return operator_costs(raw, ops)["operators"]


def test_integer_routing_preserves_empty_experts_and_changes_source_schedule_not_arithmetic():
    analytic, source = grouped_case(), grouped_case(source=True)
    a_rows, s_rows = grouped_rows(analytic), grouped_rows(source)
    assert len(a_rows) == len(s_rows) == 2
    for a, s in zip(a_rows, s_rows):
        am, sm = a["operator"]["metadata"], s["operator"]["metadata"]
        assert am["routing_basis"] == "declared_integer_counts"
        assert "expected_active_experts" not in am
        assert tuple(sm["expert_rows"]) == (3, 0, 3, 2)
        assert tuple(sm["group_ids"]) == (0, 1, 2, 3)
        assert tuple(sm["raw_group_list_active"]) == ((0, 3), (2, 3), (3, 2))
        assert sm["raw_group_list_padding_entries"] == 1
        assert a["operator"]["flops"] == s["operator"]["flops"]
        assert a["operator"]["hbm_bytes"] == s["operator"]["hbm_bytes"]
        assert a["serviced"] == pytest.approx(s["serviced"])
        assert a["duration_ms"] != pytest.approx(s["duration_ms"])
    gate = a_rows[0]["operator"]
    assert gate["flops"] == 2 * 8 * 64 * 32
    assert gate["hbm_bytes"] == (8 * 32 + 3 * 32 * 64 + 8 * 64) * 2
    first, second = predict(analytic), predict(source)
    assert first["status"] == second["status"] == "modeled"
    assert first["execution"]["work"] == pytest.approx(second["execution"]["work"])
    assert first["requests"][0]["generated_tokens"] == second["requests"][0]["generated_tokens"] == 3
    assert first["metrics"]["makespan_ms"] != pytest.approx(second["metrics"]["makespan_ms"])


def test_probability_histogram_never_becomes_a_source_routing_realization():
    raw = grouped_case(source=True)
    del raw["runtime"]["routing"]
    raw["model"]["expert_routing_histogram"] = [3, 0, 3, 2]
    parsed = parse_case(raw)
    assert parsed.model.expert_routing == (3 / 8, 0, 3 / 8, 2 / 8)
    with pytest.raises(ModelingError, match="explicit_integer_routing_realization"):
        grouped_rows(raw)
    del raw["runtime"]["kernels"]
    rows = grouped_rows(raw)
    for row in rows:
        metadata = row["operator"]["metadata"]
        assert metadata["routing_basis"] == "probability_moments"
        assert "routing_realization" not in metadata
        assert 2 < metadata["expected_active_experts"] < 3


def test_equal_total_and_active_experts_preserve_different_source_tile_loads():
    raw = grouped_case(source=True)
    scenario = raw["runtime"]["routing"]["realizations"][0]
    scenario.update(tokens=256, expert_rows=[128, 0, 128, 256])
    aligned = grouped_rows(raw, tokens=256)
    scenario["expert_rows"] = [192, 0, 160, 160]
    tailed = grouped_rows(raw, tokens=256)
    for first, second in zip(aligned, tailed):
        a, b = first["operator"], second["operator"]
        assert a["flops"] == b["flops"]
        # Both route 512 rows to three experts. The second scenario has six
        # M tiles instead of four, so it loads two extra expert weight tiles.
        _, n, k = a["local_shape"]
        assert b["hbm_bytes"] - a["hbm_bytes"] == 2 * n * k * 2
    del raw["runtime"]["kernels"]
    second = grouped_rows(raw, tokens=256)
    scenario["expert_rows"] = [128, 0, 128, 256]
    first = grouped_rows(raw, tokens=256)
    assert [row["operator"]["hbm_bytes"] for row in first] == [
        row["operator"]["hbm_bytes"] for row in second]


def test_routing_realization_round_trip_reuses_counts_for_each_declared_decode_scenario():
    raw = grouped_case(source=True)
    parsed = parse_case(raw)
    restored = parse_case(parsed.to_dict())
    assert restored.runtime.routing == raw["runtime"]["routing"]
    decoded = grouped_rows(restored, phase="decode", tokens=1)
    for row in decoded:
        assert tuple(row["operator"]["metadata"]["expert_rows"]) == (1, 0, 1, 0)
        assert row["operator"]["local_shape"][0] == 2
    a, b = predict(parsed), predict(restored)
    assert a["execution"]["work"] == b["execution"]["work"]
    assert a["metrics"] == b["metrics"]
    assert a["requests"][0]["iterations"] == 3


@pytest.mark.parametrize("counts", [[5, 0, 3, 0], [3, 0, 2, 2], [3.5, 0, 2.5, 2]])
def test_realization_enforces_topk_assignment_feasibility_without_rounding(counts):
    raw = grouped_case()
    raw["runtime"]["routing"]["realizations"][0]["expert_rows"] = counts
    with pytest.raises(ModelingError, match="conserve_integer_topk"):
        grouped_rows(raw)


def test_chunked_prefill_cannot_reuse_the_full_prompt_histogram():
    raw = grouped_case(source=True)
    raw["runtime"]["max_num_batched_tokens"] = 3
    with pytest.raises(ModelingError, match="one_matching_scenario:body:0:prefill:3"):
        predict(raw)


def test_integer_routing_rejects_graph_padding_instead_of_routing_padding_rows():
    raw = grouped_case()
    raw["runtime"]["graph_buckets"] = [4]
    with pytest.raises(ModelingError, match="unpadded_local_tokens"):
        predict(raw)
    del raw["runtime"]["routing"]
    assert predict(raw)["status"] == "modeled"


def test_topk_totals_do_not_prove_feasibility_under_group_restricted_routing():
    from dataclasses import replace

    raw = grouped_case(source=True)
    raw["runtime"]["routing"]["realizations"][0]["expert_rows"] = [4, 0, 4, 0]
    parsed = parse_case(raw)
    layer = replace(parsed.model.layer_plans[0], router_groups=2, router_group_topk=1)
    restricted = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)))
    # With two groups of two experts and top-k=2, selecting one group requires
    # equal counts within that group. Row totals alone cannot establish this.
    with pytest.raises(ModelingError, match="unrestricted_expert_groups"):
        grouped_rows(restricted)
    layer = replace(layer, router_group_topk=2)
    all_groups = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)))
    assert grouped_rows(all_groups)[0]["operator"]["flops"] == 2 * 8 * 64 * 32


@pytest.mark.parametrize("mode", ["expert_parallel", "mc2"])
def test_unmodeled_per_rank_or_per_chunk_histograms_are_not_invented(mode):
    raw = grouped_case(source=True)
    if mode == "expert_parallel":
        raw["runtime"].update(tp=2, ep=2)
        raw["machine"]["num_devices"] = 2
        raw["machine"]["links"][0]["members"] = [0, 1]
    else:
        raw["runtime"]["moe_recipe"] = "mc2"
    with pytest.raises(ModelingError, match="ep1_pcp1_unchunked_alltoall"):
        predict(raw)


def test_realized_routing_respects_tensor_shards_and_does_not_leak_between_layers():
    raw = grouped_case(source=True)
    raw["runtime"]["tp"] = 2
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    rows = grouped_rows(raw)
    assert rows[0]["operator"]["local_shape"] == [8, 32, 32]
    assert rows[1]["operator"]["local_shape"] == [8, 32, 16]
    # A second layer has its own statistical routing unless explicitly declared.
    del raw["runtime"]["kernels"]
    raw["model"]["layers"] = 2
    ops = target_operators(raw, phase="prefill", tokens=4)
    groups = [op for op in ops if op.family == "grouped_gemm"]
    assert all(dict(op.metadata)["routing_basis"] == "declared_integer_counts"
               for op in groups if op.layer == 0)
    assert all(dict(op.metadata)["routing_basis"] == "probability_moments"
               for op in groups if op.layer == 1)


def test_source_binding_does_not_reinterpret_quantized_expert_operands():
    from dataclasses import replace
    from mindie_sim.catalog.quant import quant_preset

    parsed = parse_case(grouped_case(source=True))
    parsed = replace(parsed, model=replace(parsed.model, quant=quant_preset("w8a8_int8")))
    with pytest.raises(ModelingError, match="unquantized_bf16_nd"):
        target_operators(parsed, phase="prefill", tokens=4)
