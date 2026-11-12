"""Dense FlashComm sequence-parallel collective protocol.

The admission predicate is the v0.30 source contract. Assertions are operator
identity, group membership, activation-stream bytes, and the rank-pad message.
They do not assert a measured collective duration.
"""
from __future__ import annotations

import os

import pytest

from mindie_sim.prediction.piecewise.families import _collective_rounds
from mindie_sim.prediction.piecewise.flashcomm_dense import dense_flashcomm_enabled
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.schema.common import dtype_bytes
from tests.test_piecewise_subgraph_model_pipeline import (
    FEATURES, dense_plan, make_model, pipeline,
)


def _flash(**overrides):
    additional = overrides.pop("additional_config", {"enable_flashcomm1": True})
    meta = {"additional_config": additional}
    meta.update(overrides.pop("metadata", {}))
    values = dict(FEATURES, tp=2, dp=4, pp=1, metadata=meta)
    values.update(overrides)
    return values


def _qwen(**overrides):
    meta = {"source_model_type": "qwen3_5_text"}
    meta.update(overrides.pop("metadata", {}))
    return make_model(metadata=meta, **overrides)


def _names(pipe, kind, role=None, layer=0):
    kwargs = {}
    if layer is not None:
        kwargs["layer"] = layer
    if role is not None:
        kwargs["role"] = role
    return [op.name for op in pipe.instantiated.find(kind, **kwargs).operators]


def _op(pipe, kind, name, role=None, layer=0):
    kwargs = {}
    if layer is not None:
        kwargs["layer"] = layer
    if role is not None:
        kwargs["role"] = role
    return pipe.instantiated.find(kind, **kwargs).operator(name)


def test_process_env_does_not_admit_dense_sp(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
    assert os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == "1"
    model = _qwen()
    features = pipeline(model, feature_overrides={"tp": 2, "dp": 4}).features
    assert not dense_flashcomm_enabled(model, features)


def test_switch_strings_and_captured_env_only():
    model = _qwen()
    on = pipeline(model, feature_overrides=_flash()).features
    assert dense_flashcomm_enabled(model, on)
    for switch in ("1", "true", "TRUE"):
        features = pipeline(model, feature_overrides=_flash(
            additional_config={"enable_flashcomm1": switch})).features
        assert dense_flashcomm_enabled(model, features)
    for switch in ("false", "yes", "0", ""):
        features = pipeline(model, feature_overrides=_flash(
            additional_config={"enable_flashcomm1": switch})).features
        assert not dense_flashcomm_enabled(model, features)
    captured = pipeline(model, feature_overrides=dict(
        FEATURES, tp=2, metadata={"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})).features
    assert dense_flashcomm_enabled(model, captured)
    absent = pipeline(model, feature_overrides={"tp": 2}).features
    assert not dense_flashcomm_enabled(model, absent)


@pytest.mark.parametrize("model_type", ("qwen3_5", "qwen3_5_moe_text", "glm4", ""))
def test_other_model_types_keep_the_allreduce(model_type):
    meta = {"source_model_type": model_type} if model_type else {}
    model = make_model(metadata=meta)
    pipe = pipeline(model, feature_overrides=_flash())
    assert "tp_allreduce" in _names(pipe, "F0", "dense_ffn")
    assert "tp_all_gather" not in _names(pipe, "F0", "dense_ffn")


def test_tp1_and_pp_above_one_keep_the_allreduce():
    model = _qwen()
    tp1 = pipeline(model, feature_overrides=_flash(tp=1))
    assert "tp_allreduce" not in _names(tp1, "F0", "dense_ffn")
    assert "tp_all_gather" not in _names(tp1, "F0", "dense_ffn")
    pp = pipeline(model, feature_overrides=_flash(pp=2))
    assert "tp_allreduce" in _names(pp, "A0", "attention")
    assert "tp_reduce_scatter" not in _names(pp, "A0", "attention")


def test_negative_allreduce_theta_is_unchanged():
    quant = dict(name="w8a8", weight_format="int8", activation_format="int8",
                 weight_block=(128, 128), scale_bytes=4)
    plain = make_model(quant=quant)
    admitted = _qwen(quant=quant)
    off = pipeline(plain, feature_overrides={"tp": 2, "dp": 4})
    rejected = pipeline(admitted, feature_overrides={
        "tp": 2, "dp": 4,
        "metadata": {"additional_config": {"enable_flashcomm1": False}},
    })
    left = off.instantiated.find("F0", layer=0, role="dense_ffn").operator("tp_allreduce")
    right = rejected.instantiated.find("F0", layer=0, role="dense_ffn").operator("tp_allreduce")
    assert left.theta.quant == right.theta.quant
    assert left.theta.extras == right.theta.extras
    assert left.theta.find("members") == 2
    assert "tp_allreduce" in _names(off, "A0", "attention")


def test_admitted_blocks_replace_allreduce_once():
    quant = dict(name="w8a8", weight_format="int8", activation_format="int8",
                 weight_block=(128, 128), scale_bytes=4)
    gdn = dict(index=1, attention="gdn", ffn="dense", num_heads=4, num_kv_heads=2,
               head_dim=16, intermediate_size=128, linear_heads=4, linear_head_dim=8,
               linear_key_heads=4, linear_key_head_dim=8, linear_value_heads=4,
               conv_kernel=4)
    model = _qwen(quant=quant, plans=[dense_plan(0), gdn])
    pipe = pipeline(model, feature_overrides=_flash())
    wire = model.hidden_size * dtype_bytes(model.activation_dtype)
    assert wire == 64 * 2
    for kind, role in (("A0", "attention"), ("F0", "dense_ffn")):
        names = _names(pipe, kind, role)
        assert names.count("tp_allreduce") == 0
        assert names.count("tp_all_gather") == 1
        assert names.count("tp_reduce_scatter") == 1
        assert names.index("tp_all_gather") < names.index("tp_reduce_scatter")
        gather = _op(pipe, kind, "tp_all_gather", role)
        scatter = _op(pipe, kind, "tp_reduce_scatter", role)
        assert gather.theta.find("wire_bytes") == wire
        assert scatter.theta.find("wire_bytes") == wire
        assert gather.theta.find("members") == 2
        assert scatter.theta.find("members") == 2
        assert gather.theta.find("members") != pipe.features.dp
        assert gather.theta.find("collective") == "all_gather"
        assert gather.theta.find("sp_rows") == "shard"
        assert scatter.theta.find("collective") == "reduce_scatter"
        assert scatter.theta.find("sp_rows") == "padded"
        assert gather.theta.find("payload_dtype") == "bf16"
    a0 = _names(pipe, "A0", "attention")
    assert a0.index("tp_all_gather") < a0.index("qkv_projection")
    assert a0.index("o_projection") < a0.index("tp_reduce_scatter")
    f0 = _names(pipe, "F0", "dense_ffn")
    assert f0.index("tp_all_gather") < f0.index("gate_up") < f0.index("down")
    assert f0.index("down") < f0.index("tp_reduce_scatter")
    a3 = _names(pipe, "A3", "attention", layer=1)
    assert a3.count("tp_allreduce") == 0
    assert a3.index("tp_all_gather") < a3.index("qkvz_projection")
    assert a3.index("o_projection") < a3.index("tp_reduce_scatter")
    # The quant stream stays on the negative allreduce. This path moves BF16.
    negative = pipeline(make_model(quant=quant), feature_overrides={"tp": 2})
    ar = negative.instantiated.find("F0", layer=0, role="dense_ffn").operator("tp_allreduce")
    assert ar.theta.find("wire_bytes") == model.hidden_size * 1
    assert ar.theta.find("wire_bytes") != wire


def test_exit_gather_is_one_hidden_collective_and_logits_stay():
    model = _qwen()
    pipe = pipeline(model, feature_overrides=_flash())
    names = _names(pipe, "O0", layer=None)
    assert names.index("final_norm") < names.index("tp_all_gather") < names.index("lm_head")
    assert names.index("lm_head") < names.index("logits_gather")
    assert names.count("tp_all_gather") == 1
    assert "tp_reduce_scatter" not in names
    assert "tp_allreduce" not in names
    hidden = _op(pipe, "O0", "tp_all_gather", layer=None)
    logits = _op(pipe, "O0", "logits_gather", layer=None)
    assert hidden.theta.find("sp_rows") == "shard"
    assert hidden.theta.find("wire_bytes") == model.hidden_size * 2
    assert logits.theta.find("collective") == "all_gather"
    assert logits.theta.find("sp_rows") is None
    off = pipeline(model, feature_overrides={"tp": 2, "dp": 4})
    assert "tp_all_gather" not in _names(off, "O0", layer=None)
    assert "logits_gather" in _names(off, "O0", layer=None)


def test_mla_and_shared_expert_are_not_the_dense_contract():
    plan = dict(index=0, attention="mla", ffn="dense", num_heads=8, num_kv_heads=1,
                head_dim=16, q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8,
                qk_rope_head_dim=8, v_head_dim=16, mla_variant="latent_mla",
                intermediate_size=128)
    mla = _qwen(plans=[plan], hidden_size=64, num_heads=8, num_kv_heads=1, head_dim=16)
    mla_pipe = pipeline(mla, feature_overrides=_flash())
    assert "tp_allreduce" in _names(mla_pipe, "A1", "attention")
    assert "tp_all_gather" not in _names(mla_pipe, "A1", "attention")
    moe_plan = dict(index=0, attention="gqa", ffn="moe", num_heads=4, num_kv_heads=2,
                    head_dim=16, intermediate_size=128, num_experts=4, top_k=2,
                    moe_intermediate_size=32, num_shared_experts=1,
                    shared_intermediate_size=64, shared_expert_gate=True)
    moe = _qwen(plans=[moe_plan], num_experts=4, top_k=2, moe_intermediate_size=32,
                num_shared_experts=1)
    moe_pipe = pipeline(moe, feature_overrides=_flash())
    shared = _names(moe_pipe, "F4", "shared_ffn")
    assert "tp_allreduce" in shared
    assert "tp_reduce_scatter" not in shared


def test_message_pads_only_when_tokens_are_not_a_rank_multiple():
    model = _qwen()
    pipe = pipeline(model, feature_overrides=_flash())
    compiled = pipe._subgraph(pipe.instantiated.find("F0", layer=0, role="dense_ffn"))
    wire = 64 * 2
    exact = WorkloadGeometry.homogeneous(2, 1, 1)
    short = WorkloadGeometry.homogeneous(1, 1, 1)
    gather_exact = compiled.operator("tp_all_gather").evaluate(exact)
    scatter_exact = compiled.operator("tp_reduce_scatter").evaluate(exact)
    gather_short = compiled.operator("tp_all_gather").evaluate(short)
    scatter_short = compiled.operator("tp_reduce_scatter").evaluate(short)
    assert gather_exact.get("message_bytes") == wire  # ceil(2/2) * wire
    assert scatter_exact.get("message_bytes") == 2 * wire
    assert gather_short.get("message_bytes") == wire  # ceil(1/2) * wire
    assert scatter_short.get("message_bytes") == 2 * wire
    assert gather_exact.get("ranks") == 2
    assert dict(compiled.operator("tp_all_gather").program.annotations)["collective"] == "all_gather"
    assert dict(compiled.operator("tp_reduce_scatter").program.annotations)["collective"] == "reduce_scatter"
    assert dict(compiled.operator("tp_all_gather").program.annotations)["sp_rows"] == "shard"
    theta = pipe.instantiated.find("F0", layer=0, role="dense_ffn").operator("tp_all_gather").theta
    assert _collective_rounds("tp_all_gather", theta) == (1, 0)
    assert _collective_rounds("tp_reduce_scatter", theta) == (1, 0)
    wide = pipeline(model, feature_overrides=_flash(tp=4))
    wide_theta = wide.instantiated.find("F0", layer=0, role="dense_ffn").operator("tp_all_gather").theta
    assert wide_theta.find("members") == 4
    assert _collective_rounds("tp_all_gather", wide_theta) == (3, 0)
    assert _collective_rounds("tp_reduce_scatter", wide_theta) == (3, 0)
    assert _collective_rounds("tp_allreduce", wide_theta) == (6, 0)


def _instances(pipe):
    found = list(pipe.instantiated.globals)
    for layer in pipe.instantiated.layers:
        found.extend(layer.subgraphs)
    return found


def _op_names(pipe):
    return [op.name for inst in _instances(pipe) for op in inst.operators]


def test_embedding_allreduce_is_once_and_independent_of_flashcomm():
    model = _qwen()
    on = pipeline(model, feature_overrides=_flash())
    off = pipeline(model, feature_overrides={"tp": 2, "dp": 4})
    for pipe in (on, off):
        names = _op_names(pipe)
        assert names.count("embedding_allreduce") == 1
        e0 = _names(pipe, "E0", layer=None)
        assert e0.index("embedding") < e0.index("embedding_allreduce") < e0.index("input_copy")
        ar = _op(pipe, "E0", "embedding_allreduce", layer=None)
        assert ar.theta.find("collective") == "allreduce"
        assert ar.theta.find("row_binding") == "real_q"
        assert ar.theta.find("sp_rows") is None
        assert ar.theta.find("members") == 2
        assert ar.theta.find("group_axis") == "tp"
        assert ar.theta.find("wire_bytes") == model.hidden_size * 2
        assert ar.theta.find("payload_dtype") == "bf16"
    assert "embedding_allreduce" not in _names(on, "A0", "attention")
    assert "embedding_allreduce" not in _names(on, "F0", "dense_ffn")
    assert "embedding_allreduce" not in _names(on, "O0", layer=None)
    # Flash-off keeps the layer allreduce. The embedding edge is additional.
    assert "tp_allreduce" in _names(off, "F0", "dense_ffn")
    assert "tp_all_gather" not in _names(off, "F0", "dense_ffn")
    assert "tp_allreduce" not in _names(on, "F0", "dense_ffn")


def test_tp1_and_replicated_and_missing_group_omit_the_new_edges():
    model = _qwen()
    tp1 = pipeline(model, feature_overrides=_flash(tp=1))
    assert "embedding_allreduce" not in _op_names(tp1)
    assert "logits_gather" not in _op_names(tp1)
    replicated = pipeline(model, feature_overrides=_flash(
        metadata={"embedding_parallel": "replicated"}))
    assert "embedding_allreduce" not in _op_names(replicated)
    assert "logits_gather" in _names(replicated, "O0", layer=None)
    missing = pipeline(model, feature_overrides=_flash(
        metadata={"tp_group_ranks": (0,)}))
    assert "embedding_allreduce" not in _op_names(missing)
    assert "logits_gather" not in _op_names(missing)
    fine = pipeline(model, feature_overrides=_flash(
        metadata={"embedding_tensor_parallel_size": 2}))
    assert "embedding_allreduce" not in _op_names(fine)
    skipped = pipeline(model, feature_overrides=_flash(
        metadata={"skip_gather": True}))
    assert "logits_gather" not in _names(skipped, "O0", layer=None)
    assert "embedding_allreduce" in _names(skipped, "E0", layer=None)
    sampler = pipeline(model, feature_overrides=_flash(
        metadata={"logits_collective": "gather"}))
    logits = _op(sampler, "O0", "logits_gather", layer=None)
    assert logits.theta.find("collective") == "gather"
    assert _names(sampler, "O0", layer=None).count("logits_gather") == 1
    assert "tp_allreduce" not in _names(sampler, "O0", layer=None)


def test_logits_message_matches_lm_head_selected_bf16_rows():
    model = _qwen(vocab_size=192)
    pipe = pipeline(model, feature_overrides=_flash())
    compiled = pipe._subgraph(pipe.instantiated.find("O0", layer=None))
    hidden = model.hidden_size * 2
    local = 96  # pad64(192)=192, TP2

    def pair(geometry):
        head = compiled.operator("lm_head").evaluate(geometry)
        gather = compiled.operator("logits_gather").evaluate(geometry)
        embed = pipe._subgraph(pipe.instantiated.find("E0", layer=None)).operator(
            "embedding_allreduce").evaluate(geometry)
        return head, gather, embed

    one = WorkloadGeometry.homogeneous(1, 1, 2, phase="decode")
    two = WorkloadGeometry.homogeneous(2, 1, 2, phase="decode")
    pure = WorkloadGeometry(
        (RequestGeometry(8, 0, phase="prefill"),), phase="prefill", graph_bucket=32)
    mixed = WorkloadGeometry(
        (RequestGeometry(1, 4, phase="prefill"),
         RequestGeometry(8, 0, phase="prefill")),
        phase="prefill", graph_bucket=64, physical_tokens=64)
    ranged = WorkloadGeometry(
        (RequestGeometry(8, 0, phase="prefill", query_ranges=((7, 8),)),),
        phase="prefill")
    for geometry, rows, real_q in (
        (one, 1, 1),
        (two, 2, 2),
        (pure, 1, 8),
        (mixed, 2, 9),
        (ranged, 1, 8),
    ):
        head, gather, embed = pair(geometry)
        assert dict(head.materialization)["logits"] == rows * local * 2
        assert gather.get("message_bytes") == rows * local * 2
        assert gather.get("message_bytes") != geometry.effective_tokens * local * 4
        assert gather.get("padded_wire_bytes") == rows * local * 2
        assert gather.get("useful_bytes") == rows * 192 * 2
        assert gather.get("peer_padded_bytes") == rows * local * 2 * 2
        assert embed.get("message_bytes") == real_q * hidden
        assert embed.get("useful_bytes") == embed.get("padded_wire_bytes")
        assert embed.get("peer_payload_bytes") == real_q * hidden * 2


def _compiled(pipe, kind, layer=0, role=None):
    kwargs = {"layer": layer}
    if role is not None:
        kwargs["role"] = role
    return pipe._subgraph(pipe.instantiated.find(kind, **kwargs))


def _bytes(cost, name):
    return dict(cost.materialization)[name]


def test_final_norm_uses_the_exit_gather_shard_once():
    model = _qwen()
    pipe = pipeline(model, feature_overrides=_flash())
    o0 = _compiled(pipe, "O0", layer=None)
    n0 = _compiled(pipe, "N0", layer=0, role="attn_norm")
    row = model.hidden_size * 2
    declared = _op(pipe, "O0", "final_norm", layer=None)
    norm = o0.operator("final_norm")
    gather = o0.operator("tp_all_gather")
    assert declared.theta.find("sequence_rows") == "shard"
    assert declared.theta.find("sequence_ranks") == 2
    assert _op(pipe, "O0", "tp_all_gather", layer=None).theta.find("sp_rows") == "shard"

    def pair(geometry):
        left = norm.evaluate(geometry)
        right = gather.evaluate(geometry)
        return left, right

    one = WorkloadGeometry.homogeneous(1, 1, 2, phase="decode")
    two = WorkloadGeometry.homogeneous(2, 1, 2, phase="decode")
    three = WorkloadGeometry.homogeneous(1, 3, 3, phase="decode")
    padded = WorkloadGeometry(
        (RequestGeometry(1, 4, phase="decode"),),
        phase="decode", graph_bucket=64, physical_tokens=64)
    for geometry, rows in ((one, 1), (two, 1), (three, 2), (padded, 32)):
        left, right = pair(geometry)
        assert left.get("norm_rows") == rows
        assert _bytes(left, "hidden_out") == rows * row
        assert right.get("message_bytes") == rows * row
        assert _bytes(left, "hidden_out") != geometry.effective_tokens * row or rows == geometry.effective_tokens
    # Pad-1 stays one row. The rank pad is not an extra row, and it is not
    # the reduce-scatter full-tensor pad (ceil(3/2)*2 == 4).
    assert _bytes(pair(one)[0], "hidden_out") == row
    assert _bytes(pair(three)[0], "hidden_out") == 2 * row
    assert _bytes(pair(three)[0], "hidden_out") != 4 * row
    # A layer norm on the same path is not the exit distribution.
    layer_norm = n0.operator("rmsnorm").evaluate(three)
    assert _bytes(layer_norm, "rmsnorm_output") == 3 * row
    off = pipeline(model, feature_overrides={"tp": 2, "dp": 4})
    off_norm = _compiled(off, "O0", layer=None).operator("final_norm")
    assert _op(off, "O0", "final_norm", layer=None).theta.find("sequence_rows") is None
    assert _bytes(off_norm.evaluate(two), "hidden_out") == 2 * row
    world = pipeline(model, feature_overrides=_flash(tp=1))
    world_norm = _compiled(world, "O0", layer=None).operator("final_norm")
    assert _op(world, "O0", "final_norm", layer=None).theta.find("sequence_rows") is None
    assert _bytes(world_norm.evaluate(two), "hidden_out") == 2 * row


def test_selected_scope_does_not_shard_producing_rows():
    model = _qwen()
    pipe = pipeline(model, feature_overrides=_flash(final_norm_scope="selected"))
    declared = _op(pipe, "O0", "final_norm", layer=None)
    norm = _compiled(pipe, "O0", layer=None).operator("final_norm")
    row = model.hidden_size * 2
    assert declared.theta.find("sequence_rows") == "selected"
    assert declared.theta.find("sequence_ranks") is None
    ranged = WorkloadGeometry(
        (RequestGeometry(8, 0, phase="prefill", query_ranges=((2, 5),)),),
        phase="prefill")
    prefill = WorkloadGeometry(
        (RequestGeometry(8, 0, phase="prefill"),), phase="prefill")
    decode = WorkloadGeometry.homogeneous(2, 2, 4, phase="decode")
    ranged_cost = norm.evaluate(ranged)
    prefill_cost = norm.evaluate(prefill)
    decode_cost = norm.evaluate(decode)
    assert ranged_cost.get("norm_rows") == 3
    assert _bytes(ranged_cost, "hidden_out") == 3 * row
    assert _bytes(ranged_cost, "hidden_out") != 2 * row
    assert prefill_cost.get("norm_rows") == 1
    assert _bytes(prefill_cost, "hidden_out") == row
    assert decode_cost.get("norm_rows") == 4
    assert _bytes(decode_cost, "hidden_out") == 4 * row
    off = pipeline(model, feature_overrides={"tp": 2, "final_norm_scope": "selected"})
    off_norm = _compiled(off, "O0", layer=None).operator("final_norm")
    assert _bytes(off_norm.evaluate(ranged), "hidden_out") == 3 * row


def test_embedding_is_a_live_row_lookup_of_the_local_table():
    model = _qwen()
    pipe = pipeline(model, feature_overrides=_flash())
    embedding = _op(pipe, "E0", "embedding", layer=None)
    assert dict(embedding.theta.matrix) == {"table_rows": 64, "table_cols": 64}
    assert embedding.theta.find("K") is None
    assert embedding.theta.find("N") is None
    assert embedding.theta.find("lookup") == 1
    assert embedding.theta.find("masked") == 1
    assert embedding.theta.find("embedding_mode") == "standard"
    assert "masked_fill" not in _op_names(pipe)
    compiled = _compiled(pipe, "E0", layer=None).operator("embedding")
    two = WorkloadGeometry.homogeneous(2, 1, 2, phase="decode")
    cost = compiled.evaluate(two)
    row = 64 * 2
    assert _bytes(cost, "hidden") == 2 * row
    assert cost.get("lookup_rows") == 2
    assert cost.get("gather_bytes") == 2 * row
    assert cost.get("gather_bytes") != 64 * row
    assert cost.get("masked_fill") == 1
    assert cost.residency_bytes == 64 * row
    assert cost.pipe_work("cube") == 0
    assert cost.pipe_work("vector") == 2 * 2 * 64
    assert cost.pipe_work("mte2") == 2 * row + 2 * 8
    padded = WorkloadGeometry(
        (RequestGeometry(1, 4, phase="decode"),),
        phase="decode", graph_bucket=64, physical_tokens=64)
    padded_cost = compiled.evaluate(padded)
    assert padded_cost.get("lookup_rows") == 1
    assert _bytes(padded_cost, "hidden") == row
    assert padded_cost.get("gather_bytes") == row
    reduce = _compiled(pipe, "E0", layer=None).operator("embedding_allreduce").evaluate(padded)
    assert reduce.get("message_bytes") == row
    replicated = pipeline(model, feature_overrides=_flash(
        metadata={"embedding_parallel": "replicated"}))
    replicated_op = _op(replicated, "E0", "embedding", layer=None)
    assert dict(replicated_op.theta.matrix) == {"table_rows": 128, "table_cols": 64}
    assert replicated_op.theta.find("masked") == 0
    assert "embedding_allreduce" not in _op_names(replicated)
    full = _compiled(replicated, "E0", layer=None).operator("embedding").evaluate(two)
    assert full.residency_bytes == 128 * row
    assert full.get("masked_fill") == 0
    assert full.pipe_work("vector") == 2 * 64
    assert _bytes(full, "hidden") == 2 * row
    fine = pipeline(model, feature_overrides=_flash(
        metadata={"embedding_tensor_parallel_size": 4}))
    fine_op = _op(fine, "E0", "embedding", layer=None)
    assert dict(fine_op.theta.matrix)["table_rows"] == 32
    assert fine_op.theta.find("embedding_mode") == "fine_grained"
    assert fine_op.theta.find("shard_world") == 4
    assert "embedding_allreduce" not in _op_names(fine)
    missing = pipeline(model, feature_overrides=_flash(
        metadata={"tp_group_ranks": (0,)}))
    missing_op = _op(missing, "E0", "embedding", layer=None)
    assert dict(missing_op.theta.matrix)["table_rows"] == 64
    assert missing_op.theta.find("masked") == 1
    assert "embedding_allreduce" not in _op_names(missing)
    tp1 = pipeline(model, feature_overrides=_flash(tp=1))
    tp1_op = _op(tp1, "E0", "embedding", layer=None)
    assert dict(tp1_op.theta.matrix) == {"table_rows": 128, "table_cols": 64}
    assert tp1_op.theta.find("masked") == 0
    assert tp1_op.theta.find("K") is None
    pcp = pipeline(model, feature_overrides=_flash(
        pcp=2, metadata={"pcp_embedding_lmhead_weight_sharding": True}))
    pcp_op = _op(pcp, "E0", "embedding", layer=None)
    assert dict(pcp_op.theta.matrix)["table_rows"] == 32
    assert pcp_op.theta.find("embedding_mode") == "pcp"
    assert pcp_op.theta.find("shard_world") == 4
    assert _op_names(pcp).count("embedding_allreduce") == 1
