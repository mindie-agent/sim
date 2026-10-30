"""Representation counterexamples use independent integer tensor arithmetic."""
from dataclasses import replace

from mindie_sim import Case
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import EngramSpec, EngramTableSpec, QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes, term_payload_bytes
from mindie_sim.catalog.weights import stage_weight_bytes, stage_weight_decomposition
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.parallel import ParallelSpec
from test_prediction_system import case


def test_odd_width_packing_and_scale_tails_are_recomputed_after_tensor_sharding():
    quant = QuantSpec("declared-fp4", expert_weight_format="mxfp4", expert_block=32)
    term = WeightTerm("expert", 3, 65, "routed_expert", "expert_down")
    whole = ParallelSpec()
    shard = ParallelSpec(tp=2)
    assert term_payload_bytes(term, quant, whole) == 3 * 33
    assert term_bytes(term, quant, whole) == 3 * 33 + 3 * 3
    # Each rank owns ceil(65 / 2)=33 columns: packed bytes and scales both tail.
    assert term_payload_bytes(term, quant, shard) == 3 * 17
    assert term_bytes(term, quant, shard) == 3 * 17 + 3 * 2
    assert term_bytes(term, quant, shard) * 2 > term_bytes(term, quant, whole)
    grouped_int4 = replace(quant, expert_weight_format="int4", scale_bytes=4)
    assert term_bytes(term, grouped_int4, whole) == 3 * 33 + 3 * 3 * 4
    channel_int4 = replace(grouped_int4, expert_block=1)
    assert term_bytes(term, channel_int4, whole) == 3 * 33 + 3 * 4


def test_fp8_block_scale_grid_drives_gemm_traffic_without_a_compression_multiplier():
    quant = QuantSpec("declared-fp8", weight_format="fp8_e4m3",
                      weight_block=(2, 32), scale_bytes=4)
    op = gemm_op(op_id="m", deps=(), stage=0, layer=0, requests=(), m=2, n=3, k=65,
                 quant=quant, category="attn", phase="prefill")
    assert dict(op.metadata)["packed_weight_bytes"] == 3 * 65 + 2 * 3 * 4
    assert quant.matrix_payload_bytes(3, 65, "attn") == 3 * 65
    assert op.flops == 2 * 2 * 3 * 65


def test_engram_source_and_runtime_payloads_decompose_actual_stage_capacity():
    parsed = Case.from_mapping(case())
    model = replace(
        parsed.model, layer_plans=tuple(replace(layer, hc_mult=1) for layer in parsed.model.layer_plans),
        engram=EngramSpec(tables=(EngramTableSpec(0, 0, 3, 65),), max_ngram=2, heads=1, width=65))
    for mode, payload, metadata in (
        ("device_bf16", 3 * 65 * 2, 0),
        ("device_int8_fp32", 3 * 65, 3 * 3 * 4),
        ("host_fp8_e8m0", 3 * 65, 3 * 3),
    ):
        quant = QuantSpec("declared-table", engram_table_mode=mode)
        selected = replace(model, quant=quant)
        included = stage_weight_decomposition(selected, parsed.runtime.parallel)[0]
        excluded = stage_weight_decomposition(
            selected, parsed.runtime.parallel, include_engram_tables=False)[0]
        assert included["payload_bytes"] - excluded["payload_bytes"] == payload
        assert included["quant_metadata_bytes"] - excluded["quant_metadata_bytes"] == metadata
        assert included["total_bytes"] == included["payload_bytes"] + included["quant_metadata_bytes"]
        assert included["total_bytes"] == stage_weight_bytes(selected, parsed.runtime.parallel)[0]
        assert Memory(replace(parsed, model=selected)).weights[0] == included["total_bytes"]
        # Source table is packed FP8/E8M0 even when device residency is BF16.
        assert quant.matrix_payload_bytes(3, 65, "engram_table", source=True) == 3 * 65
        assert quant.matrix_bytes(3, 65, "engram_table", source=True) == 3 * 65 + 3 * 3
