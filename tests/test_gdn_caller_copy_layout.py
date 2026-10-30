"""Source z reshape addresses, padded payload and one physical materialization."""
import pytest

from mindie_sim.prediction.frontend import Frontend
from test_gdn_decode_family import execute_one, source_case


@pytest.mark.parametrize("keys,heads,tokens", ((2, 6, 16), (2, 6, 160), (2, 6, 512),
                                             (4, 12, 16), (4, 12, 128), (4, 12, 512)))
def test_captured_projection_geometry_materializes_only_selected_z_payload(keys, heads, tokens):
    parsed, views, ops = source_case(lengths=(4,) * (tokens // 4), keys=keys, heads=heads)
    copy = next(op for op in ops if op.recipe == "z_norm_materialize")
    meta = dict(copy.metadata)
    width, value = (2 * keys + heads) * 128, heads * 128
    assert meta["source_layout"]["shape"] == (tokens, value)
    assert meta["source_layout"]["strides"] == (width + value, 1)
    assert meta["source_layout"]["offset"] == width
    assert meta["target_layout"]["strides"] == (value, 1)
    assert copy.mte2_bytes == copy.mte3_bytes == tokens * value * 2
    assert copy.hbm_bytes == tokens * value * 4
    assert copy.vector_ops == 0
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    assert copy.op_id in norm.deps
    assert dict(norm.metadata)["gate_layout"]["shape"] == (tokens * heads, 128)
    assert dict(norm.metadata)["gate_layout"]["strides"] == (128, 1)
    # The source assignment remains separate even when a small device copy
    # is absent from kernel_details.csv's exported kernel population.
    assert sum(op.recipe == "core_output_copy" for op in ops) == 1
    engine, _ = execute_one(parsed, views, copy)
    assert engine.serviced["memory@r0:s0"] == pytest.approx(copy.hbm_bytes)
    assert engine.serviced["mte3@r0:s0"] == pytest.approx(copy.mte3_bytes)


@pytest.mark.parametrize("lengths,heads", (((1,), 12), ((4,), 1)))
def test_singleton_compatible_z_reshape_is_a_view(lengths, heads):
    _, _, ops = source_case(lengths=lengths, keys=1, heads=heads)
    assert not any(op.recipe == "z_norm_materialize" for op in ops)
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    projection = next(op for op in ops if op.recipe == "linear_qkvz")
    assert projection.op_id in norm.deps


def test_source_z_reshape_uses_padded_projection_rows_without_padding_recurrence():
    parsed, views, _ = source_case(lengths=(1,), keys=4, heads=12)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode", tokens=16)
    copy = next(op for op in ops if op.recipe == "z_norm_materialize")
    assert copy.local_shape == (16, 1536)
    assert copy.hbm_bytes == 16 * 1536 * 4
    recurrence = next(op for op in ops if op.family == "recurrent_vector")
    assert dict(recurrence.metadata)["tokens"] == 1
    core_copy = next(op for op in ops if op.recipe == "core_output_copy")
    assert core_copy.hbm_bytes == 1536 * 4
