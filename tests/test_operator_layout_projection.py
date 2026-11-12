"""Central source-role physical rows for R31 Qwen static dense SP."""
from __future__ import annotations

from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry


def _geometry():
    return WorkloadGeometry(
        (RequestGeometry(1, 4095, phase="decode"),),
        phase="decode", local_batch=1, dense_sequence_parallel=True,
        tp_width=2, graph_dispatch="full_decode_only",
        capture_sizes=(1, 2, 4, 8, 16, 32))


def test_static_dense_sp_selects_source_role_rows_without_multiplication():
    geometry = _geometry()
    lanes = geometry.token_lanes()
    assert lanes["valid_tokens"] == 1
    assert lanes["gathered_tokens"] == 2
    assert lanes["mlp_tokens"] == 2
    assert lanes["shard_tokens"] == 1
    assert lanes["static_decode"] is True

    def lane(kind, role, name, family="gemm"):
        return geometry.operator_layout(kind, role, name, family).effective_tokens

    assert lane("F0", "gate_up", "gate_up") == 2
    assert lane("F0", "down", "down") == 2
    assert lane("A0", "projection", "qkv_projection") == 2
    assert lane("A3", "qkvz", "qkvz_projection") == 2
    assert lane("A3", "", "core_zeros", "copy") == 2
    assert lane("A3", "", "gated_delta", "recurrent") == 1
    assert lane("A3", "", "conv", "recurrent") == 1
    assert lane("N0", "norm", "rmsnorm", "elementwise") == 1
    assert lane("N0", "residual", "fused:add_rmsnorm", "fused") == 1
    assert lane("A0", "attention", "attention", "attention") == 1
    assert lane("O0", "head", "lm_head") == 1
    assert lane("A3", "", "tp_all_gather", "collective") == 2


def test_non_sp_geometry_is_unchanged():
    geometry = WorkloadGeometry(
        (RequestGeometry(1, 4095, phase="decode"),), phase="decode",
        local_batch=1)
    projected = geometry.operator_layout("F0", "gate_up", "gate_up")
    assert projected is geometry
    assert projected.effective_tokens == 1


def test_operand_view_can_keep_shard_below_logical_and_capture_bucket():
    geometry = WorkloadGeometry(
        (RequestGeometry(3, 0, phase="decode"),),
        phase="decode", local_batch=1, graph_bucket=4,
        dense_sequence_parallel=True, tp_width=2,
        graph_dispatch="full_decode_only", capture_sizes=(4,))
    lanes = geometry.token_lanes()
    assert lanes["valid_tokens"] == 3
    assert lanes["gathered_tokens"] == 4
    assert lanes["shard_tokens"] == 2
    projected = geometry.operator_layout("N0", "norm", "fused:add_rmsnorm")
    assert projected.operand_tokens == 2
    assert projected.effective_tokens == 2
    assert projected.tokens == 3
    assert projected.graph_bucket == 4
    assert projected.bindings()["T"] == 2
    assert projected.bindings()["T_VALID"] == 3
    assert projected.bindings()["BUCKET"] == 4
    assert "operand_tokens" in projected.bindings()
    from mindie_sim.prediction.piecewise.native_add_rms_norm import _rows
    assert _rows(None, projected, "shard", 2) == 2
