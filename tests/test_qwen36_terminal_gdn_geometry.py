"""Qwen3.6 GDN a/b and terminal projection physical geometry.

[Role]
  Pin the source-defined Qwen3.5/3.6 GDN fused a/b width (two value-head
  projections) and the vocab-parallel lm_head width/gather, including padded
  vocab and TP1.

[Invariants]
  * GDN a/b is one fused projection of 2*local value heads.  It does NOT add
    key heads: key heads belong to q/k, not beta/alpha.
  * lm_head is one vocab-parallel GEMM of ceil(pad64(vocab)/TP) outputs per
    rank; TP1 is full padded vocab and emits no TP gather.  TP>1 gathers the
    sharded logits once before sampling.
  * Active GMM/qkvz/o projection geometry is unchanged by these terminal
    widths.

[Side-Effect Policy] Pure catalog/subgraph construction; no I/O.
"""
from dataclasses import replace

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig


SUPPLY = PipeSupply((("cube", 1e9), ("vector", 1e9), ("hbm", 1e9),
                     ("mte1", 1e9), ("mte2", 1e9), ("mte3", 1e9),
                     ("fixpipe", 1e9), ("link", 1e9)))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)


def pipeline(model, *, tp=2, dp=4):
    features = FeatureConfig.from_model(
        model, tp=tp, dp=dp, ep=1, pp=1, token_tile=16, kv_tile=16,
        cores=2, moe_row_tile=4, split_target=1000, max_splits=4)
    return assemble_model(model, supply=SUPPLY, policy=POLICY, features=features)


def matrix(op):
    return dict(op.theta.matrix)


def test_qwen36_gdn_ba_is_two_value_head_projections_not_value_plus_key():
    model = load_model("qwen3.6-27b")
    pipe = pipeline(model)
    a3 = pipe.instantiated.layers[0].find("A3")
    assert matrix(a3.operator("ba_projection")) == {"K": 5120, "N": 48}
    assert matrix(a3.operator("qkvz_projection")) == {"K": 5120, "N": 8192}
    assert matrix(a3.operator("o_projection")) == {"K": 3072, "N": 5120}
    extras = dict(a3.operator("ba_projection").theta.extras)
    assert dict(a3.operator("qkvz_projection").theta.heads) == {}
    assert dict(a3.operator("qkvz_projection").theta.extras)["key_heads"] == 8
    assert dict(a3.operator("qkvz_projection").theta.extras)["value_heads"] == 24
    assert extras.get("local_value_heads") in (None, 24)


def test_qwen36_lm_head_is_vocab_parallel_and_gathers_once():
    model = load_model("qwen3.6-27b")
    pipe = pipeline(model)
    o0 = pipe.instantiated.find("O0")
    assert matrix(o0.operator("lm_head")) == {"K": 5120, "N": 124160}
    gather = o0.operator("logits_gather")
    assert gather.template.family == "collective"
    assert dict(gather.theta.extras)["collective"] == "all_gather"
    # One selected row of the local BF16 shard, not the fp32 sampler width.
    quant = dict(gather.theta.quant)
    assert quant.get("wire_bytes") == 124160 * 2
    assert gather.theta.find("result_dtype") == "bf16"
    assert gather.theta.find("row_binding") == "lm_head_selected"
    assert "embedding_allreduce" in {op.name for op in pipe.instantiated.find("E0").operators}


def _collective_counts(pipe):
    counts = {}
    instances = list(pipe.instantiated.globals)
    for layer in pipe.instantiated.layers:
        instances.extend(layer.subgraphs)
    for inst in instances:
        if inst.kind == "D0":
            continue
        for op in inst.operators:
            if op.template.family != "collective":
                continue
            counts[op.name] = counts.get(op.name, 0) + 1
    return counts


def _enrolled():
    model = load_model("qwen3.6-27b")
    meta = dict(model.metadata)
    meta["source_model_type"] = "qwen3_5_text"
    model = replace(model, metadata=tuple(meta.items()))
    return model


def _enrolled_pipe(*, flash):
    model = _enrolled()
    additional = {"enable_flashcomm1": "1" if flash else "false"}
    features = FeatureConfig.from_model(
        model, tp=2, dp=4, ep=1, pp=1, token_tile=16, kv_tile=16,
        cores=2, moe_row_tile=4, split_target=1000, max_splits=4,
        metadata={
            "source_model_type": "qwen3_5_text",
            "additional_config": additional,
        })
    return assemble_model(model, supply=SUPPLY, policy=POLICY, features=features)


def test_enrolled_qwen_step_has_259_edges_and_selected_logits():
    pipe = _enrolled_pipe(flash=True)
    counts = _collective_counts(pipe)
    assert counts == {
        "embedding_allreduce": 1,
        "tp_all_gather": 129,
        "tp_reduce_scatter": 128,
        "logits_gather": 1,
    }
    assert sum(counts.values()) == 259
    off = _enrolled_pipe(flash=False)
    off_counts = _collective_counts(off)
    assert off_counts["embedding_allreduce"] == 1
    assert off_counts["logits_gather"] == 1
    assert off_counts.get("tp_all_gather", 0) == 0
    assert off_counts.get("tp_reduce_scatter", 0) == 0
    assert off_counts["tp_allreduce"] == 128
    o0 = pipe._subgraph(pipe.instantiated.find("O0"))
    e0 = pipe._subgraph(pipe.instantiated.find("E0"))
    two = WorkloadGeometry.homogeneous(2, 1, 4, phase="decode")
    mixed = WorkloadGeometry(
        (RequestGeometry(1, 4, phase="prefill"),
         RequestGeometry(4096, 0, phase="prefill")),
        phase="prefill", graph_bucket=8192, physical_tokens=8192)
    pure = WorkloadGeometry(
        (RequestGeometry(4096, 0, phase="prefill"),),
        phase="prefill", graph_bucket=8192)
    q1 = WorkloadGeometry(
        (RequestGeometry(1, 4, phase="decode"),),
        phase="decode", graph_bucket=64, physical_tokens=64)
    for geometry, rows in ((two, 2), (mixed, 2), (pure, 1), (q1, 1)):
        head = o0.operator("lm_head").evaluate(geometry)
        gather = o0.operator("logits_gather").evaluate(geometry)
        embed = e0.operator("embedding_allreduce").evaluate(geometry)
        assert dict(head.materialization)["logits"] == rows * 124160 * 2
        assert gather.get("message_bytes") == rows * 124160 * 2
        assert gather.get("useful_bytes") == rows * 248320 * 2
        assert gather.get("peer_padded_bytes") == rows * 124160 * 2 * 2
        assert embed.get("message_bytes") == geometry.tokens * 5120 * 2
        assert embed.get("peer_payload_bytes") == geometry.tokens * 5120 * 2 * 2


def test_lm_head_tp1_is_full_and_has_no_tp_gather():
    model = load_model("qwen3.6-27b")
    pipe = pipeline(model, tp=1, dp=1)
    o0 = pipe.instantiated.find("O0")
    assert matrix(o0.operator("lm_head")) == {"K": 5120, "N": 248320}
    assert "logits_gather" not in {op.name for op in o0.operators}
    assert "embedding_allreduce" not in {op.name for op in pipe.instantiated.find("E0").operators}


@pytest.mark.parametrize("tied", (False, True))
def test_nondiv_vocab_pads_to_64_before_tp_split(tied):
    model = replace(load_model("qwen3.6-27b"), vocab_size=130,
                    tie_word_embeddings=tied)
    pipe = pipeline(model, tp=2, dp=1)
    o0 = pipe.instantiated.find("O0")
    # pad64(130)=192; 192/2=96 local outputs, 192 gathered.
    assert matrix(o0.operator("lm_head")) == {"K": 5120, "N": 96}
    gather = o0.operator("logits_gather")
    assert dict(gather.theta.quant).get("wire_bytes") == 96 * 2
    compiled = pipe._subgraph(o0)
    one = WorkloadGeometry.homogeneous(1, 1, 2)
    cost = compiled.operator("logits_gather").evaluate(one)
    # HCCL keeps the padded shard. The post-slice vocab is the useful result.
    assert cost.get("message_bytes") == 96 * 2
    assert cost.get("padded_wire_bytes") == 96 * 2
    assert cost.get("useful_bytes") == 130 * 2
    assert cost.get("peer_padded_bytes") == 96 * 2 * 2
    embed = pipe.instantiated.find("E0").operator("embedding")
    assert matrix(embed) == {"table_rows": 96, "table_cols": 5120}
    assert embed.theta.find("K") is None
    assert embed.theta.find("N") is None
    assert embed.theta.find("masked") == 1


def test_enrolled_final_norm_matches_exit_gather_and_embedding_is_local():
    pipe = _enrolled_pipe(flash=True)
    assert sum(_collective_counts(pipe).values()) == 259
    o0 = pipe._subgraph(pipe.instantiated.find("O0"))
    e0 = pipe._subgraph(pipe.instantiated.find("E0"))
    declared = pipe.instantiated.find("E0").operator("embedding")
    assert matrix(declared) == {"table_rows": 124160, "table_cols": 5120}
    assert declared.theta.find("K") is None
    assert declared.theta.find("N") is None
    assert declared.theta.find("lookup") == 1
    assert declared.theta.find("masked") == 1
    assert declared.theta.find("weight_total") == 124160 * 5120 * 2
    assert "masked_fill" not in {op.name for op in pipe.instantiated.find("E0").operators}
    row = 5120 * 2
    norm = o0.operator("final_norm")
    gather = o0.operator("tp_all_gather")
    embed = e0.operator("embedding")
    reduce = e0.operator("embedding_allreduce")
    assert pipe.instantiated.find("O0").operator("final_norm").theta.find("sequence_rows") == "shard"
    assert pipe.instantiated.find("O0").operator("final_norm").theta.find("sequence_ranks") == 2

    def check(geometry, shard_rows, live_rows):
        norm_cost = norm.evaluate(geometry)
        gather_cost = gather.evaluate(geometry)
        embed_cost = embed.evaluate(geometry)
        reduce_cost = reduce.evaluate(geometry)
        assert norm_cost.get("norm_rows") == shard_rows
        assert dict(norm_cost.materialization)["hidden_out"] == shard_rows * row
        assert gather_cost.get("message_bytes") == shard_rows * row
        assert embed_cost.get("lookup_rows") == live_rows
        assert dict(embed_cost.materialization)["hidden"] == live_rows * row
        assert embed_cost.get("gather_bytes") == live_rows * row
        assert embed_cost.get("gather_bytes") != 124160 * row
        assert embed_cost.residency_bytes == 124160 * row
        assert embed_cost.pipe_work("cube") == 0
        assert embed_cost.get("masked_fill") == 1
        assert reduce_cost.get("message_bytes") == live_rows * row

    two = WorkloadGeometry.homogeneous(2, 1, 4, phase="decode")
    one = WorkloadGeometry.homogeneous(1, 1, 4, phase="decode")
    mixed = WorkloadGeometry(
        (RequestGeometry(1, 4, phase="prefill"),
         RequestGeometry(4096, 0, phase="prefill")),
        phase="prefill")
    pure = WorkloadGeometry(
        (RequestGeometry(4096, 0, phase="prefill"),), phase="prefill")
    padded = WorkloadGeometry(
        (RequestGeometry(1, 4, phase="decode"),),
        phase="decode", graph_bucket=64, physical_tokens=64)
    check(two, 1, 2)
    check(one, 1, 1)
    check(mixed, 2049, 4097)
    check(pure, 2048, 4096)
    check(padded, 32, 1)
    assert dict(norm.evaluate(two).materialization)["hidden_out"] == 10240
    assert dict(norm.evaluate(mixed).materialization)["hidden_out"] == 20981760
    assert dict(norm.evaluate(pure).materialization)["hidden_out"] == 20971520
    assert dict(embed.evaluate(padded).materialization)["hidden"] == 10240
    off = _enrolled_pipe(flash=False)
    off_norm = off._subgraph(off.instantiated.find("O0")).operator("final_norm")
    assert off.instantiated.find("O0").operator("final_norm").theta.find("sequence_rows") is None
    assert dict(off_norm.evaluate(two).materialization)["hidden_out"] == 20480
    assert sum(_collective_counts(off).values()) == 130
