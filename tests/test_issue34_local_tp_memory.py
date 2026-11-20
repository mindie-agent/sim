"""Shared placement shapes, packed scales and weight-ledger identity."""
from __future__ import annotations

import pytest

from mindie_sim.catalog.projection_placement import (
    projection_placement, projection_slices,
)
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from _support.builders import dense_layer, dense_model, linear_model, quant


def parallel(**overrides):
    values = dict(o_proj_tp=4, linear_proj_tp=2, lm_head_tp=2)
    values.update(overrides)
    return ParallelSpec(**values)


def test_row_and_column_placement_shard_the_correct_weight_dimension():
    p = parallel()
    row = WeightTerm("o_proj", 128, 256, "attn.o", "in_tp", quant=quant(),
                     placement_role="attention_output")
    col = WeightTerm("kda_q", 256, 128, "kda.q", "out_tp", quant=quant(),
                     placement_role="linear_input")
    head = WeightTerm("lm_head", 128, 256, "lm_head", "out_tp", quant=quant(),
                      placement_role="logits")
    assert term_bytes(row, quant(), p) == 128 * (256 // 4) * 2
    assert term_bytes(col, quant(), p) == (256 // 2) * 128 * 2
    assert term_bytes(head, quant(), p) == (128 // 2) * 256 * 2


def test_grouped_output_pair_shards_whole_groups_or_fails():
    p = parallel()
    wo_a = WeightTerm("wo_a", 8 * 16, 32, "mla.o_a", "group_out", groups=8,
                      quant=quant(), placement_role="attention_output")
    wo_b = WeightTerm("wo_b", 32, 8 * 16, "mla.o_b", "group_in", groups=8,
                      quant=quant(), placement_role="attention_output")
    assert term_bytes(wo_a, quant(), p) == 32 * 32 * 2
    assert term_bytes(wo_b, quant(), p) == 32 * 32 * 2
    bad = WeightTerm("wo_a", 6 * 16, 32, "mla.o_a", "group_out", groups=6,
                     quant=quant(), placement_role="attention_output")
    with pytest.raises(ModelingError, match="local_tp_group_geometry_indivisible"):
        term_bytes(bad, quant(), p)


def test_projection_placement_matches_term_shapes_and_packed_slices():
    model = dense_model(layers=1)
    placement = projection_placement(
        model, model.layer_plans[0], parallel(), role="attention_output",
        node_ranks=4)
    assert placement.physical_rows == 32 and placement.physical_cols == 16
    assert placement.width == 4
    assert placement.weight_terms
    packed = quant(weight_format="mxfp4", activation_format="fp8_e4m3",
                   weight_block=(32, 32))
    kda = linear_model("kda", layers=1)
    column = projection_placement(
        kda, kda.layer_plans[0], parallel(linear_proj_tp=2),
        role="linear_input", node_ranks=4)
    slices = projection_slices(
        [term for term in column.weight_terms if term.name in {"kda_q", "kda_k"}],
        column, packed)
    assert len(slices) == 2
    assert [item.semantic_role for item in slices] == ["linear_input", "linear_input"]
    assert slices[0].byte_offset == 0
    assert slices[1].physical_extent == 32


def test_packed_scale_bytes_follow_the_placed_shape():
    p = parallel(linear_proj_tp=2)
    packed = quant(weight_format="mxfp4", activation_format="fp8_e4m3",
                   weight_block=(32, 32))
    term = WeightTerm("kda_q", 256, 128, "kda.q", "out_tp", quant=packed,
                      placement_role="linear_input")
    assert term_bytes(term, packed, p) == pytest.approx(
        packed.matrix_bytes(128, 128, "kda.q"))
    assert term_bytes(term, packed, p) != term_bytes(
        WeightTerm("kda_q", 256, 128, "kda.q", "out_tp", quant=packed,
                              placement_role="linear_input"), packed,
        ParallelSpec())


def test_resident_expert_count_is_not_divided_by_ep_again():
    p = ParallelSpec(ep=4, expert_tp=1)
    term = WeightTerm("expert_gate", 64, 32, "routed_expert",
                      "expert_gate_up", count=8, quant=quant(),
                      resident_count=3)
    assert term_bytes(term, quant(), p) == 64 * 32 * 3 * 2
    expected_unsharded = 64 * 32 * 3 * 2
    assert term_bytes(term, quant(), p) == expected_unsharded


def test_tied_embedding_keeps_one_owner_and_draft_terms_are_tagged():
    from mindie_sim.catalog.weights import body_head_terms
    from mindie_sim.schema.model import ModelSpec
    tied = ModelSpec(name="tied", layers=1, hidden_size=32, num_heads=4,
                     head_dim=8, num_kv_heads=2, vocab_size=128,
                     intermediate_size=64, tie_word_embeddings=True,
                     layer_plans=(dense_layer(0),))
    terms = {term.name: term for term in body_head_terms(tied)}
    assert "embedding" in terms and "lm_head" not in terms
    assert terms["embedding"].placement_role == ""
