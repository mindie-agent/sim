"""Independent invariants for the standalone Kimi call-scope helpers."""
from collections.abc import Mapping
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from tools.kimi_call_scopes import (
    SITU_DOWN, SITU_INPUT, SITU_QUANT, kda_stage_ledger, situ_reference,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case


QUANT = QuantSpec(
    "descriptor_conditioned_mxfp8", weight_format="fp8_e4m3",
    activation_format="fp8_e4m3", expert_weight_format="mxfp4",
    weight_block=(1, 32), expert_block=32, scale_format="fp8_e8m0",
    scale_bytes=1, activation_group_size=32,
    activation_scale_format="fp8_e8m0", activation_scale_pack=2,
)
K, GATE_N, WIDTH, EXPERTS, GROUP = 3584, 6144, 3072, 28, 32


def situ_case(*, beta=4.0, linear_beta=25.0):
    raw = case()
    raw["model"].update(layers=1, intermediate_size=WIDTH)
    raw["model"]["layer_plans"] = [{
        "index": 0, "ffn": "dense", "intermediate_size": WIDTH,
        "ffn_activation": "situ", "activation_situ_beta": beta,
        "activation_situ_linear_beta": linear_beta,
    }]
    return parse_case(raw)


def situ_descriptors(m):
    return {
        "input_shapes": [[m, K], [m, K // GROUP // 2, 2], [EXPERTS, GATE_N, K // 2],
                         [EXPERTS, GATE_N, K // GROUP // 2, 2], [EXPERTS]],
        "output_shapes": [[m, WIDTH], [m, WIDTH // GROUP // 2, 2]],
        "input_dtypes": ["DT_FLOAT8_E4M3FN", "DT_FLOAT8_E8M0", "DT_FLOAT4_E2M1",
                         "DT_FLOAT8_E8M0", "INT64"],
        "output_dtypes": ["DT_FLOAT8_E4M3FN", "DT_FLOAT8_E8M0"],
        "input_formats": ["ND", "ND", "FRACTAL_NZ_C0_16", "ND", "ND"],
        "output_formats": ["ND", "ND"],
    }


def maybe_pairs(tokens, chunk):
    return (tokens // chunk) * chunk * chunk + (tokens % chunk) ** 2


@pytest.mark.parametrize("m,expected_flops", [(2875, 126_615_552_000), (3286, 144_716_070_912)])
def test_situ_work_and_traffic_match_an_independent_integer_oracle(m, expected_flops):
    result = situ_reference(situ_case(), situ_descriptors(m), QUANT)
    gate, activation, quantize = result["ops"]
    ledger = result["ledger"]
    groups = m * (WIDTH // GROUP)
    flops = 2 * m * GATE_N * K
    input_payload, input_scales = m * K, m * (K // GROUP)
    weight_payload = EXPERTS * GATE_N * (K // 2)
    weight_scales = EXPERTS * GATE_N * (K // GROUP)
    gate_bf16, situ_bf16 = m * GATE_N * 2, m * WIDTH * 2
    output_payload, output_scales = m * WIDTH, m * (WIDTH // GROUP)
    situ_ops = 12 * m * WIDTH
    quant_ops = 4 * m * WIDTH - groups + 3 * groups
    assert flops == expected_flops
    assert gate.flops == flops
    assert gate.mte2_bytes == input_payload + input_scales + weight_payload + weight_scales
    assert gate.mte3_bytes == gate_bf16
    assert activation.vector_ops == situ_ops
    assert (activation.mte2_bytes, activation.mte3_bytes) == (gate_bf16, situ_bf16)
    assert quantize.vector_ops == quant_ops
    assert (quantize.mte2_bytes, quantize.mte3_bytes) == (
        situ_bf16, output_payload + output_scales)
    assert ledger["traffic"]["weight_payload_bytes"] == weight_payload
    assert ledger["traffic"]["weight_scale_bytes"] == weight_scales
    assert ledger["traffic"]["total_op_hbm_bytes"] == gate.hbm_bytes + activation.hbm_bytes + quantize.hbm_bytes
    assert ledger["workspaces"]["intermediate_bf16_read_write_bytes"] == 2 * gate_bf16 + 2 * situ_bf16
    assert ledger["work"]["situ_scalar_ops"] == situ_ops
    assert ledger["work"]["output_quant_vector_ops"] == quant_ops
    if m == 2875:
        assert [weight_payload, weight_scales, situ_ops, quant_ops] == [
            308_281_344, 19_267_584, 105_984_000, 35_880_000]


def test_situ_graph_has_three_ops_no_down_and_no_input_quantizer():
    result = situ_reference(situ_case(), situ_descriptors(2875), QUANT)
    gate, activation, quantize = result["ops"]
    assert gate.recipe == "matrix_envelope"
    assert type(result["recipes"].select(gate)).__name__ == "ResourceEnvelope"
    assert [op.op_id for op in result["ops"]] == ["situ.gate_up", "situ.activation", SITU_QUANT]
    assert SITU_DOWN not in [op.op_id for op in result["ops"]]
    assert not any(op.family == "down_projection" for op in result["ops"])
    assert gate.deps == (SITU_INPUT,)
    assert activation.deps == (gate.op_id,)
    assert quantize.deps == (activation.op_id,)
    assert [op.op_id for op in result["ops"] if op.op_id.endswith(".quantize")] == [SITU_QUANT]
    graph = graph_costs(situ_case(), result["ops"], recipes=result["recipes"],
                        external_inputs={SITU_INPUT: 0}, isolated=False)
    assert [row["op_id"] for row in graph["timeline"]] == ["situ.gate_up", "situ.activation", SITU_QUANT]
    assert graph["timeline"][1]["predecessors"] == [graph["timeline"][0]["id"]]
    assert graph["timeline"][2]["predecessors"] == [graph["timeline"][1]["id"]]
    assert graph["duration_ms"] > 0
    assert graph["execution"]["live_tasks"] == 0
    assert graph["execution"]["live_programs"] == 0
    assert not any(graph["execution"]["live_allocated"].values())


def test_situ_beta_changes_metadata_not_the_declared_selection():
    first = situ_reference(situ_case(beta=4.0), situ_descriptors(3286), QUANT)
    changed = situ_reference(situ_case(beta=5.0), situ_descriptors(3286), QUANT)
    assert [op.op_id for op in first["ops"]] == [op.op_id for op in changed["ops"]]
    assert [op.recipe for op in first["ops"]] == [op.recipe for op in changed["ops"]]
    assert [(op.flops, op.vector_ops) for op in first["ops"]] == [
        (op.flops, op.vector_ops) for op in changed["ops"]]
    assert dict(first["ops"][1].metadata)["activation_beta"] == 4.0
    assert dict(changed["ops"][1].metadata)["activation_beta"] == 5.0
    assert first["ledger"]["activation"]["beta"] == 4.0
    assert changed["ledger"]["activation"]["beta"] == 5.0
    assert first["ledger"]["matrix_recipe"] == changed["ledger"]["matrix_recipe"]


def test_situ_rejects_geometry_that_does_not_match_the_descriptor():
    bad = situ_descriptors(2875)
    bad["output_shapes"][0][1] = 4096
    with pytest.raises(ModelingError, match="output_must_be_half_gate_up_width"):
        situ_reference(situ_case(), bad, QUANT)
    bad = situ_descriptors(2875)
    bad["input_shapes"][1] = [2875, 56]
    with pytest.raises(ModelingError, match="input_scale_geometry_mismatch"):
        situ_reference(situ_case(), bad, QUANT)
    with pytest.raises(ModelingError, match="case_plan_ffn_activation_situ"):
        raw = situ_case().to_dict()
        raw["model"]["layer_plans"][0]["ffn_activation"] = "silu"
        situ_reference(parse_case(raw), situ_descriptors(2875), QUANT)


@pytest.mark.parametrize("changes,reason", [
    ({"exclusions": ("routed_expert",)}, "quant_must_declare_fp4_weights"),
    ({"activation_scale_format": "fp32"}, "input_scale_declaration_mismatch"),
    ({"activation_scale_pack": 3}, "input_scale_declaration_mismatch"),
])
def test_situ_cannot_execute_a_different_operand_representation_than_its_ledger(changes, reason):
    with pytest.raises(ModelingError, match=reason):
        situ_reference(situ_case(), situ_descriptors(2875), replace(QUANT, **changes))


def kda_descriptors(tokens=8192, chunk=64):
    chunks = math.ceil(tokens / chunk)
    return {
        "input_shapes": [[1, tokens, 12, 128]] * 4 + [
            [1, 12, tokens], [12], [1536], [1, 12, 128, 128], [2], [2 * chunks]],
        "input_dtypes": ["DT_BF16"] * 4 + ["FLOAT"] * 4 + ["INT64", "INT64"],
        "output_shapes": [[tokens * 12 * 128], [1, 12, 128, 128], [1, 12, tokens, 128],
                          [tokens * 12 * chunk], [tokens * 12 * chunk]] + [
                          [1, 12, tokens, 128]] * 5 + [
                          [1, chunks, 12, 128, 128], [1, 12, tokens, 128], [1, 12, tokens, 128]],
        "output_dtypes": ["DT_BF16", "FLOAT", "FLOAT"] + ["DT_BF16"] * 10,
        "input_formats": ["NCHW"] * 4 + ["ND"] * 4 + ["ND"] * 2,
        "output_formats": ["NCHW", "ND", "ND"] + ["NCHW"] + ["ND"] * 9,
    }


def test_kda_source_stage_products_and_storage_are_independent_oracles():
    ledger = kda_stage_ledger(kda_descriptors())
    tokens, heads, dk, dv, chunk, pairs = 8192, 12, 128, 128, 64, 524288
    stages = {stage["name"]: stage for stage in ledger["stage_ledger"]}
    source_prepare = 4 * pairs * heads * dk
    source_finalize = 2 * tokens * heads * dk * dv + 2 * pairs * heads * dv
    assert ledger["geometry"]["pairs"] == maybe_pairs(tokens, chunk) == pairs
    assert ledger["source_condition"]["split_stages"] is True
    assert ledger["source_condition"]["launches"] == 4
    assert ledger["source_condition"]["stage_scalar_present"] is False
    assert stages["GATE_PREPARE"]["recognized_product_flops"] == source_prepare == 3_221_225_472
    assert stages["POST_WU"]["recognized_product_flops"] == 0
    assert stages["FWD_H"]["recognized_product_flops"] == 0
    assert stages["FINALIZE"]["recognized_product_flops"] == source_finalize == 4_831_838_208
    assert ledger["recognized_product_work"] == {
        "source_prepare_flops": 3_221_225_472,
        "source_finalize_flops": 4_831_838_208,
        "source_finalize_qg_h_flops": 3_221_225_472,
        "source_finalize_aqk_v_flops": 1_610_612_736,
    }
    storage_bytes = sum(
        math.prod(tensor["shape"]) *
        (2 if tensor["dtype"] == "bf16" else 4)
        for tensor in ledger["storage"]["output_contract"]["tensors"])
    assert len(ledger["storage"]["output_contract"]["tensors"]) == 13
    assert ledger["storage"]["output_contract"]["unique_output_footprint_bytes"] == storage_bytes == 327_942_144
    assert ledger["storage"]["workspace"]["known_workspace_bytes"] == 2 * tokens * heads * chunk * 4 == 50_331_648
    assert ledger["storage"]["traffic"]["total_bytes"] is None
    assert ledger["storage"]["traffic"]["status"] == "unresolved_source_stage_traffic"


def test_kda_corrected_phase_work_retains_prior_shift_without_claiming_native_timing():
    ledger = kda_stage_ledger(kda_descriptors())
    shift = ledger["semantic_phase_mismatch"]
    assert shift["flops_shifted_from_prepare_to_output"] == 0
    assert shift["generic_chunk_products_flops"] == shift["source_prepare_flops"]
    assert shift["generic_recurrent_output_flops"] == shift["source_finalize_flops"]
    assert shift["prior_to_revision"]["flops_previously_shifted_from_prepare_to_output"] == 1_610_612_736
    assert shift["combined_flops_equal"] is True
    assert shift["combined_flops"] == shift["source_prepare_flops"] + shift["source_finalize_flops"]
    assert [stage["name"] for stage in ledger["stage_ledger"]] == [
        "GATE_PREPARE", "POST_WU", "FWD_H", "FINALIZE"]
    assert all(stage["mapping_status"] == "semantic_projection_only_not_source_phase_timing"
               for stage in ledger["stage_ledger"])
    assert all(stage["evidence"] for stage in ledger["stage_ledger"])

    def numbers(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from numbers(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from numbers(item)

    def keys(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from keys(item)

    assert not {"duration_ms", "mean_ms", "p50_ms", "activity_sum_ms", "native_launch_count"} & set(keys(ledger))
    assert 1104 not in set(numbers(ledger))


def test_kda_source_branch_conditions_select_dense_full_or_split_stages():
    split = kda_stage_ledger(kda_descriptors())
    assert split["source_condition"]["dense_fastpath"]["selected"] is False
    assert split["source_condition"]["launches"] == 4
    dense = kda_descriptors()
    dense["input_shapes"] = dense["input_shapes"][:8] + dense["input_shapes"][9:]
    dense["input_dtypes"] = dense["input_dtypes"][:8] + dense["input_dtypes"][9:]
    dense["input_formats"] = dense["input_formats"][:8] + dense["input_formats"][9:]
    ledger = kda_stage_ledger(dense)
    assert ledger["source_condition"]["dense_fastpath"]["selected"] is True
    assert ledger["source_condition"]["split_stages"] is False
    assert ledger["source_condition"]["launches"] == 1
    assert [stage["name"] for stage in ledger["stage_ledger"]] == ["FULL"]
    ledger = kda_stage_ledger(kda_descriptors(tokens=64), chunk_size=64)
    assert ledger["geometry"]["total_chunks"] == 1
    assert ledger["source_condition"]["split_stages"] is False
    assert ledger["source_condition"]["launches"] == 1
    assert ledger["storage"]["output_contract"]["tensors"][0]["bytes"] == 64 * 12 * 128 * 2
    wider = kda_stage_ledger(kda_descriptors(tokens=256, chunk=128), chunk_size=128)
    assert wider["geometry"]["total_chunks"] == 2
    assert wider["source_condition"]["launches"] == 4
    assert wider["geometry"]["pairs"] == 2 * 128 ** 2
    with pytest.raises(ModelingError, match="source_chunk_size_must_be_64_or_128"):
        kda_stage_ledger(kda_descriptors(chunk=8192), chunk_size=8192)


def test_kda_sources_match_the_pinned_audit_when_it_is_present():
    audit_path = Path("/tmp/kimi-native-call-scope-audit-20261029.json")
    if not audit_path.exists():
        pytest.skip("local pinned audit file is absent")
    audit = json.loads(audit_path.read_text())
    hashes = {row["path"]: row["sha256"] for row in audit["sources"]}
    for source in kda_stage_ledger(kda_descriptors())["sources"]:
        assert hashes[source["path"]] == source["sha256"]
