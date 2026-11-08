"""Operator -> subgraph -> model -> framework invariants.
[Role] Exercise the operator, subgraph, model and framework layers together.
[Invariants] Assertions are independent arithmetic/state identities, never
  type checks or fixed prose.
[Timeouts & Locks] Pure CPU tests with finite loops; no device or network.
[Side-Effect Policy] Read-only catalog fixtures; no filesystem writes.
The tests use independent arithmetic identities from the structural contracts:
fusion removes only declared internal materialisation, serial chains sum,
concurrent branches share one capacity engine, state edges order producers and
consumers, graph padding does not shorten active attention/MoE rows, serving
metrics come from committed events and MTP verify work is never discounted.
"""
from __future__ import annotations
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import pytest
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise.composition import ( contention_envelope, evaluate_serial_split, lower_fusion, )
from mindie_sim.prediction.piecewise.compiler import ( ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply, )
from mindie_sim.prediction.piecewise.geometry import ( RequestGeometry, WorkloadGeometry, )
from mindie_sim.prediction.piecewise.model_pipeline import (FrameworkConfig, _draft_cost,
    assemble_model, gemma4_model_spec, geometry_key, predict_framework, )
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.prediction.subgraphs.state import ( CompressionState, MoeRoutingState, SharedKvContract, StateView, )
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
SUPPLY = PipeSupply(( ("cube", 1_000_000.0), ("vector", 1_000_000.0), ("hbm", 1_000_000.0),
    ("mte1", 2_000_000.0), ("mte2", 2_000_000.0), ("mte3", 2_000_000.0), ("fixpipe", 2_000_000.0), ("link", 1_000_000.0), ))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4, split_target=1000, max_splits=4)
FEATURES = dict(token_tile=16, kv_tile=16, cores=2, moe_row_tile=4, split_target=1000, max_splits=4)
FIXTURES = Path(__file__).parent / "fixtures"
def dense_plan(index=0, **changes):
    plan = dict(index=index, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128, qk_norm=True)
    plan.update(changes)
    return plan
def make_model(*, name="toy", plans=None, **overrides):
    base = dict(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128)
    base.update(overrides)
    if plans is None:
        plans = [dense_plan()]
    raw = dict(base, name=name, layers=len(plans), layer_plans=plans)
    return ModelSpec.from_mapping(raw)
def features(model, **overrides):
    values = dict(FEATURES)
    values.update(overrides)
    return FeatureConfig.from_model(model, **values)
def pipeline(model, *, feature_overrides=None, policy=POLICY, supply=SUPPLY):
    return assemble_model(model, supply=supply, policy=policy, features=features(model, **(feature_overrides or {})))
def dense_model(*, layers=1, quant=None):
    return make_model(plans=[dense_plan(index) for index in range(layers)], quant=quant)

@pytest.mark.parametrize(("kind", "role", "fusion_name", "kept", "positive"), [ ("N0", "attn_norm", "add_rmsnorm_quant",
     {"residual", "rstd", "scale", "norm_output"}, {"residual", "rstd", "scale", "norm_output"}),
    ("A0", "attention", "qkv_norm_rope", {"query", "key", "value", "residual"}, {"query", "key", "value"}),
    ("F0", "dense_ffn", "dequant_swiglu_quant", {"down_input", "activation_scale", "residual"}, {"down_input", "activation_scale"}), ])
def test_native_fusion_removes_internal_traffic_and_keeps_consumers( kind, role, fusion_name, kept, positive):
    quant = dict(name="w8a8", weight_format="int8", activation_format="int8", weight_block=(128, 128), scale_bytes=4)
    model = dense_model(quant=quant)
    pipe = pipeline(model)
    instance = pipe.instantiated.find(kind, layer=0, role=role)
    compiled = pipe._subgraph(instance)
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    evaluation = compiled.evaluate(geometry)
    contract = next(item for item in instance.template.fusions if item.name == fusion_name)
    result = lower_fusion(instance, evaluation.operators, fusion_name, supply=SUPPLY)
    assert result.startup_count == 1 < len(result.semantic)
    assert result.removed_bytes > 0
    assert kept <= set(dict(result.kept))
    assert all(dict(result.kept)[name] > 0 for name in positive)
    assert result.fused.family == "fused"
    assert result.fused.fidelity == "fused_physical_kernel"
    assert result.fused.duration_ms <= result.isolated_ms
    assert dict(result.fused.derived)["physical_kernels"] == 1
    assert sum(value for _, value in result.fused.work) < sum( value for cost in result.semantic for _, value in cost.work)
    changed = tuple( replace(operator, template=replace(operator.template, stream="aux"))
        if operator.name == contract.members[-1] else operator for operator in compiled.instance.operators)
    with pytest.raises(ModelingError):
        lower_fusion(replace(compiled.instance, operators=changed), evaluation.operators, fusion_name, supply=SUPPLY)
def test_mla_prolog_fusion_keeps_kv_and_query_outputs():
    model = make_model(plans=[dict( index=0, attention="mla", ffn="dense", num_heads=4, num_kv_heads=1,
        head_dim=16, intermediate_size=128, q_lora_rank=16, kv_lora_rank=16,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16, mla_variant="latent_mla")])
    pipe = pipeline(model)
    instance = pipe.instantiated.find("A1", layer=0, role="attention")
    compiled = pipe._subgraph(instance)
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    evaluation = compiled.evaluate(geometry)
    result = lower_fusion(instance, evaluation.operators, "mla_prolog", supply=SUPPLY)
    assert {"query", "key", "value", "kv_cache"} <= set(dict(result.kept))
    assert result.removed_bytes > 0
    assert result.fused.duration_ms < result.isolated_ms

def test_serial_sum_identity_and_late_input_gap():
    model = dense_model()
    pipe = pipeline(model)
    compiled = pipe._subgraph(pipe.instantiated.find("F0", layer=0, role="dense_ffn"))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    result = evaluate_serial_split(compiled, geometry, fusion=())
    assert result.total_ms == sum(cost.duration_ms for cost in result.costs)
    assert result.gap_ms == 0
    late = evaluate_serial_split(compiled, geometry, fusion=(), readiness={"gate_up": 3})
    assert late.gap_ms == 3
    assert late.total_ms == result.total_ms + 3
    early = evaluate_serial_split(compiled, geometry, fusion=(), readiness={"gate_up": 0})
    assert early.total_ms == result.total_ms
    fused = evaluate_serial_split(compiled, geometry, fusion=("dequant_swiglu_quant",))
    assert fused.total_ms == sum(cost.duration_ms for cost in fused.costs)
    assert "dequant_swiglu_quant" in fused.fused

def test_fork_join_hbm_contention_and_shared_branch_envelope():
    shared = contention_envelope( {"A": [[("hbm", 1000)]], "B": [[("hbm", 1000)]]}, PipeSupply((("hbm", 1000),)))
    assert shared.makespan_ms == 2
    assert shared.isolated_max_ms == 1
    assert shared.rejected_isolated_max
    assert dict(shared.resource_work)["hbm"] == 2000
    assert shared.resource_bounds == (("hbm", Fraction(2)),)
    overlap = contention_envelope( {"A": [[("hbm", 1000)]], "B": [[("cube", 1000)]]}, PipeSupply((("hbm", 1000), ("cube", 1000))))
    assert overlap.makespan_ms == 1
    assert not overlap.rejected_isolated_max
    assert set(overlap.disjoint_pipes) == {"hbm", "cube"}
    plans = [dict(index=0, attention="gqa", ffn="moe", num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
                  num_experts=4, top_k=2, moe_intermediate_size=32, num_shared_experts=1)]
    model = make_model(plans=plans, num_experts=4, top_k=2, moe_intermediate_size=32, num_shared_experts=1)
    pipe = pipeline(model, feature_overrides={ "multistream_shared_expert": True, "moe_recipe": "alltoall"})
    prediction = pipe.forward(WorkloadGeometry.homogeneous(2, 1, 16))
    layer = prediction.layers[0]
    assert layer.contention is not None
    envelope = layer.contention
    assert envelope.rejected_isolated_max
    assert envelope.makespan_ms > envelope.isolated_max_ms
    assert {"hbm", "cube"} <= set(envelope.shared_pipes)
    assert {"shared", "routed"} == set(dict(envelope.branches))
    assert {item.kind for item in layer.subgraphs} >= {"F2", "F3", "F4"}
    for attn, flag in (("gdn", "multistream_kda_proj"), ("kda", "multistream_kda_proj")):
        m_rec = make_model(plans=[dict(index=0, attention=attn, ffn="dense", num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=64, linear_heads=4, linear_head_dim=16)])
        p_rec = pipeline(m_rec, feature_overrides={flag: True}).forward(WorkloadGeometry.homogeneous(2, 1, 16)).layers[0]
        assert p_rec.contention is not None and {"main", "aux"} == set(dict(p_rec.contention.branches))
    m_idx = make_model(plans=[dict(index=0, attention="mla", ffn="dense", num_heads=4, num_kv_heads=1, head_dim=16, intermediate_size=64, indexer=True, index_heads=2, index_dim=16, index_topk=8)], mla=dict(q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, index_topk=8))
    p_idx = pipeline(m_idx, feature_overrides={"multistream_attention_indexer": True}).forward(WorkloadGeometry.homogeneous(2, 1, 16)).layers[0]
    assert p_idx.contention is not None and {"attention", "indexer"} == set(dict(p_idx.contention.branches))

def test_shared_kv_store_once_consumer_reads_and_r1_slots_evolve():
    plans = [ dense_plan(0, owns_long_kv=True), dense_plan(1, kv_owner_layer_id=0), dense_plan(2, kv_owner_layer_id=0), ]
    model = make_model(plans=plans)
    pipe = pipeline(model)
    contracts = pipe.shared_kv_contracts()
    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.owner_layer == 0 and contract.consumers == (1, 2)
    assert contract.store_bytes(8) == 8 * contract.row_bytes
    assert contract.read_bytes(64) == 64 * contract.row_bytes
    owner = pipe._subgraph(pipe.instantiated.find("K0", layer=0, role="kv"))
    first = pipe._subgraph(pipe.instantiated.find("K0", layer=1, role="kv"))
    second = pipe._subgraph(pipe.instantiated.find("K0", layer=2, role="kv"))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    state = owner.evaluate(geometry).state_out
    with pytest.raises(ModelingError):
        first.evaluate(geometry, state.__class__())
    read_one = next(cost for cost in first.evaluate(geometry, state).operators if cost.name == "kv_read")
    read_two = next(cost for cost in second.evaluate(geometry, state).operators if cost.name == "kv_read")
    store = next(cost for cost in owner.evaluate(geometry).operators if cost.name == "kv_store")
    assert read_one.get("kv_bytes") == read_two.get("kv_bytes") == 0
    assert store.get("kv_bytes") > 0
    consumer_a0 = pipe.instantiated.find("A0", layer=1, role="attention")
    assert "qkv_projection" not in {operator.name for operator in consumer_a0.operators}
    ratio = 4
    state = CompressionState.start(ratio)
    produced = []
    for query in (3, 5, 2, 7):
        produced.append(state.new_rows(query))
        state = state.advance(query)
    assert produced == [0, 2, 0, 2]
    assert sum(produced) == 0 + 2 + 0 + 2 == 17 // ratio
    assert CompressionState.telescoped_new_rows(ratio, 0, 17) == 17 // ratio
    r1_plans = [dense_plan(index, residual_block_size=2) for index in range(4)]
    r1 = pipeline(make_model(plans=r1_plans))
    prediction = r1.forward(WorkloadGeometry.homogeneous(1, 1, 8))
    retained = [layer.state_out.residual.retained for layer in prediction.layers]
    assert retained == [(), (0,), (0,), (0, 1)]
    assert all(layer.state_out.residual.slots <= 2 for layer in prediction.layers)
    r1_scores = [next(row for row in layer.rows if row.cost.name == "score") for layer in prediction.layers]
    assert [dict(row.cost.materialization).get("score_output") for row in r1_scores] == [4, 4, 8, 8]

def gemma_config(**changes):
    config = dict( model_type="gemma4", num_hidden_layers=4, hidden_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=128, sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        num_kv_shared_layers=2, attention_k_eq_v=True, hidden_act="gelu_pytorch_tanh", per_layer_embedding=True, ple_width=32)
    config.update(changes)
    return config

def test_gemma4_standard_subgraph_assembly():
    model = gemma4_model_spec(gemma_config())
    assert model.attention == "gqa" and model.layers == 4
    pipe = pipeline(model)
    slots = [[(item.kind, item.variant) for item in layer.subgraphs] for layer in pipe.instantiated.layers]
    assert ("E1", "ple") in slots[0] and ("E1", "ple") in slots[-1]
    assert ("F0", "gelu") in slots[0]
    assert ("A0", "gqa") in slots[0] and ("A0", "shared_consumer") in slots[2]
    assert ("K0", "read") in slots[2] and ("K0", "read") in slots[3]
    local = pipe.instantiated.find("A0", layer=0, role="attention")
    global_attention = pipe.instantiated.find("A0", layer=1, role="attention")
    local_theta = local.operator("attention").theta
    global_theta = global_attention.operator("attention").theta
    assert local_theta.find("window") == 8
    assert global_theta.find("window") == 0
    assert local.operator("qkv_projection").theta.find("k_eq_v") == 0
    assert global_attention.operator("qkv_projection").theta.find("k_eq_v") == 1
    local_geometry = RequestGeometry(1, 31)
    assert local_geometry.pairs(window=8) == 8
    assert local_geometry.pairs(window=0) == 32
    assert local_geometry.pairs(window=8) < local_geometry.pairs(window=0)
    with_kv = global_attention.operator("qkv_projection").theta
    plain_model = gemma4_model_spec(gemma_config(attention_k_eq_v=False))
    plain = pipeline(plain_model)
    without_kv = plain.instantiated.find("A0", layer=1, role="attention").operator( "qkv_projection").theta
    assert with_kv.require("N") < without_kv.require("N")
    assert pipeline(model).instantiated.find("N0", layer=0, role="attn_norm").operator( "rmsnorm").theta.find("norm_weight_offset") == 1.0
    prediction = pipe.forward(WorkloadGeometry.homogeneous(2, 4, 32))
    assert prediction.total_ms > 0
    assert any(row.kind == "E1" for row in prediction.rows)

def compact(model, limit=8):
    groups = {}
    for plan in model.layer_plans:
        key = (plan.attention, plan.ffn, bool(plan.window), plan.mla_variant, bool(plan.indexer), bool(plan.compress_ratio),
               bool(plan.residual_block_size), bool(plan.engram), plan.owns_long_kv)
        groups.setdefault(key, plan)
    selected = set()
    for plan in groups.values():
        selected.add(plan.index)
        if 0 <= plan.kv_owner_layer_id < plan.index:
            selected.add(plan.kv_owner_layer_id)
    indices = sorted(selected)[:limit]
    keep = set(indices)
    for plan in model.layer_plans:
        if plan.index in keep and 0 <= plan.kv_owner_layer_id < plan.index:
            keep.add(plan.kv_owner_layer_id)
    indices = sorted(keep)
    remap = {old: new for new, old in enumerate(indices)}
    plans = []
    for new, old in enumerate(indices):
        plan = model.layer_plans[old]
        owner = remap.get(plan.kv_owner_layer_id, new)
        plans.append(replace(plan, index=new, kv_owner_layer_id=owner, owns_long_kv=owner == new))
    return replace(model, layers=len(plans), layer_plans=tuple(plans), body_layers=len(plans))

def fixture_model(name):
    raw = json.loads((FIXTURES / "hf-standard-families" / name).read_text())
    return ModelSpec.from_mapping(raw)

def llama_model():
    return make_model(name="llama", plans=[dense_plan(index) for index in range(2)])
FAMILIES = [ ("deepseek-v3", lambda: compact(load_model("deepseek-v3")), {"A1", "F1", "F2", "F3"}),
    ("deepseek-v3.2", lambda: compact(load_model("deepseek-v3.2")), {"A1", "I0", "I1", "F3"}),
    ("deepseek-v4f", lambda: compact(load_model("deepseek-v4-flash-0731")), {"A2", "K1", "I0", "I1", "R0"}),
    ("deepseek-v4.1", lambda: compact(load_model("deepseek-v4.1-flash")), {"A2", "K1", "I0", "I1"}),
    ("qwen3", lambda: compact(load_model("qwen3-32b")), {"A0", "F0"}), ("qwen3.5", lambda: compact(load_model("qwen3.5-27b")), {"A3", "A0", "F0"}),
    ("qwen3.6", lambda: compact(load_model("qwen3.6-27b")), {"A3", "A0", "F0"}),
    ("qwen3.8", lambda: compact(load_model("qwen3.8-27b")), {"A3", "A0", "F0"}),
    ("glm-4.5", lambda: compact(fixture_model("glm45.json")), {"A0", "F0", "F1", "F3"}),
    ("glm-5.1", lambda: compact(load_model("glm-5.1")), {"A1", "I0", "I1"}), ("glm-5.3", lambda: compact(load_model("glm-5.3")), {"A1", "I0", "I1"}),
    ("kimi-k2.5", lambda: compact(load_model("kimi-k2.5")), {"A1", "F1", "F3"}),
    ("kimi-k3", lambda: compact(load_model("kimi-k3")), {"A4", "A1", "R1", "F3"}), ("llama", llama_model, {"A0", "F0"}),
    ("gemma4", lambda: gemma4_model_spec(gemma_config()), {"A0", "E1", "F0"}), ]

@pytest.mark.parametrize(("family", "factory", "expected"), FAMILIES, ids=[item[0] for item in FAMILIES])
def test_all_named_model_families_assemble_and_predict(family, factory, expected):
    model = factory()
    pipe = pipeline(model)
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    prediction = pipe.forward(geometry)
    kinds = {item.kind for layer in pipe.instantiated.layers for item in layer.subgraphs}
    assert expected <= kinds
    assert len(prediction.layers) == model.layers
    assert prediction.total_ms > 0
    assert prediction.rows
    assert all(layer.duration_ms >= 0 for layer in prediction.layers)

def test_layer_template_aggregation_keeps_per_layer_report():
    model = dense_model(layers=6)
    pipe = pipeline(model)
    prediction = pipe.forward(WorkloadGeometry.homogeneous(2, 4, 32))
    assert len(prediction.layers) == 6
    assert prediction.evaluation_count == 1
    assert prediction.template_groups == 1
    assert prediction.layers[0].reused is False
    assert all(layer.reused for layer in prediction.layers[1:])
    assert all(layer.group == 0 for layer in prediction.layers)
    assert all(len(layer.rows) == len(prediction.layers[0].rows) for layer in prediction.layers)
    assert {row.layer for row in prediction.rows} >= {0, 5}
    report = prediction.to_dict()
    assert len(report["layers"]) == 6
    assert report["evaluation_count"] == 1
    assert any(key.startswith("family:") for key, _ in prediction.bottlenecks())

def test_aclgraph_bucket_padding_preserves_active_attention_rows():
    model = dense_model()
    pipe = pipeline(model)
    active = WorkloadGeometry.homogeneous(2, 3, 24)
    bucket = WorkloadGeometry(active.requests, graph_bucket=16)
    padded = pipe.forward(active, padded_geometry=bucket, fusion=False)
    exact = pipe.forward(active, fusion=False)
    assert padded.total_ms > exact.total_ms
    qkv_padded = next(row for row in padded.rows if row.cost.name == "qkv_projection" and row.layer == 0)
    qkv_exact = next(row for row in exact.rows if row.cost.name == "qkv_projection" and row.layer == 0)
    assert qkv_padded.geometry == "padded" and qkv_padded.padded_tokens == 16
    assert qkv_exact.geometry == "active" and qkv_exact.padded_tokens == 6
    attention_padded = next(row for row in padded.rows if row.cost.family == "attention")
    attention_exact = next(row for row in exact.rows if row.cost.family == "attention")
    assert attention_padded.geometry == "active"
    assert attention_padded.active_tokens == attention_exact.active_tokens == 6
    assert attention_padded.cost.get("causal_pairs") == attention_exact.cost.get( "causal_pairs")
    empty = contention_envelope({}, SUPPLY)
    assert empty.makespan_ms == 0

def test_continuous_batching_ragged_requests_and_serving_metrics():
    model = dense_model(layers=2)
    config = FrameworkConfig(supply=SUPPLY, graph_buckets=(4, 8, 16), max_batch=2)
    requests = [ dict(id="a", arrival_ms=0, prompt_tokens=5, output_tokens=3), dict(id="b", arrival_ms=0, prompt_tokens=3, output_tokens=4),
        dict(id="c", arrival_ms=1, prompt_tokens=2, output_tokens=2), ]
    result = predict_framework(model, requests, config)
    assert len(result.requests) == 3
    assert all(item.ttft_ms > 0 and item.tpot_ms > 0 for item in result.requests)
    assert all(item.completion_ms >= item.ttft_ms for item in result.requests)
    late = next(item for item in result.requests if item.request_id == "c")
    assert late.ttft_ms > next(item for item in result.requests if item.request_id == "a").ttft_ms
    assert any(step["kind"] == "prefill" for step in result.steps)
    assert any(step["kind"] == "decode" for step in result.steps)
    assert all(step["batch"] <= 2 for step in result.steps)
    assert result.padding["padded_tokens"] >= result.padding["active_tokens"]
    assert result.padding["operator_padded_rows"] > 0
    assert result.padding["attention_active_rows"] > 0
    assert result.padding["padding_waste_ms"] > 0
    decode_bodies = [step["body_ms"] for step in result.steps if step["kind"] == "decode"]
    assert result.body_duration_ms == pytest.approx( sum(decode_bodies) / len(decode_bodies))
    assert result.bottlenecks and result.bottlenecks[0][1] > 0
    left = predict_framework(model, [ dict(id="x", prompt_tokens=2, output_tokens=1), dict(id="y", prompt_tokens=2, output_tokens=1)], config)
    right = predict_framework(model, [ dict(id="x", prompt_tokens=1, output_tokens=1), dict(id="y", prompt_tokens=3, output_tokens=1)], config)
    assert left.steps[0]["active_tokens"] == right.steps[0]["active_tokens"]
    assert left.steps[0]["body_ms"] != right.steps[0]["body_ms"]

def test_dp_and_ep_cross_rank_sync_waits():
    dp = pipeline(make_model(plans=[dense_plan(index) for index in range(2)], layers=2), feature_overrides={"dp": 2})
    assert dp.features.dp == 2
    dp_result = predict_framework(dp, [ dict(id="a", prompt_tokens=5, output_tokens=2),
        dict(id="b", prompt_tokens=3, output_tokens=2), dict(id="c", prompt_tokens=2, output_tokens=2),
        dict(id="d", prompt_tokens=1, output_tokens=2), ], FrameworkConfig(supply=SUPPLY, max_batch=4))
    assert dp_result.sync["dp_ranks"] == 2
    assert dp_result.sync["dp_skew_ms"] > 0
    moe = make_model( name="moe", plans=[dict(index=0, attention="gqa", ffn="moe", num_heads=4,
                    num_kv_heads=2, head_dim=16, intermediate_size=128, num_experts=4, top_k=2, moe_intermediate_size=32)],
        num_experts=4, top_k=2, moe_intermediate_size=32)
    ep = pipeline(moe, feature_overrides={"ep": 2})
    assert ep.features.ep == 2
    ep_result = predict_framework(ep, [ dict(id="a", prompt_tokens=4, output_tokens=1)],
        FrameworkConfig(supply=SUPPLY, max_batch=1, routing=((0, (8, 0, 0, 0)),)))
    assert ep_result.sync["ep_ranks"] == 2
    assert ep_result.sync["ep_skew_ms"] > 0
    permuted = predict_framework(ep, [dict(id="a", prompt_tokens=4, output_tokens=1)],
        FrameworkConfig(supply=SUPPLY, max_batch=1, routing=((0, (0, 0, 8, 0)),)))
    assert permuted.steps[0]["body_ms"] == pytest.approx(ep_result.steps[0]["body_ms"])
    assert ep_result.sync["sync_wait_ms"] == ( ep_result.sync["dp_skew_ms"] + ep_result.sync["ep_skew_ms"])
    balanced = predict_framework(ep, [ dict(id="a", prompt_tokens=4, output_tokens=1)],
        FrameworkConfig(supply=SUPPLY, max_batch=1, routing=((0, (4, 0, 4, 0)),)))
    assert balanced.sync["ep_skew_ms"] < ep_result.sync["ep_skew_ms"]

def mtp_model():
    raw = dict( name="mtp", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128, layer_plans=[dense_plan(0)],
        auxiliary=dict(mtp=dict(kind="mtp", layers=1, hidden_size=64, intermediate_size=128, num_heads=4,
                                num_kv_heads=2, head_dim=16, q_width=64, kv_width=32)))
    return ModelSpec.from_mapping(raw)

def test_mtp_draft_and_target_verify_are_not_acceptance_discounted():
    model = mtp_model()
    pipe = pipeline(model)
    assert pipe.draft_subgraph is not None
    request = [dict(id="a", prompt_tokens=4, output_tokens=4)]
    low = predict_framework(pipe, request, FrameworkConfig( supply=SUPPLY, max_batch=1, mtp_draft_len=2, mtp_accept=0))
    high = predict_framework(pipe, request, FrameworkConfig( supply=SUPPLY, max_batch=1, mtp_draft_len=2, mtp_accept=2))
    assert low.mtp and high.mtp
    assert low.mtp["draft_ms"] > 0 and low.mtp["commit_ms"] > 0
    assert low.mtp["target_ms"] > 0
    assert low.steps[1]["body_ms"] == pytest.approx(high.steps[1]["body_ms"])
    assert low.mtp["total_ms"] == pytest.approx( low.mtp["draft_ms"] + low.mtp["target_ms"] + low.mtp["commit_ms"])
    assert high.mtp["acceptance_factor"] > low.mtp["acceptance_factor"]
    assert high.requests[0].itl_ms <= low.requests[0].itl_ms
    one = predict_framework(pipe, request, FrameworkConfig(supply=SUPPLY, max_batch=1, mtp_draft_len=1))
    two = predict_framework(pipe, request, FrameworkConfig(supply=SUPPLY, max_batch=1, mtp_draft_len=2))
    assert two.mtp["draft_ms"] > one.mtp["draft_ms"] and two.mtp["accepted"] <= 2 * 2
    assert high.mtp["acceptance_factor"] <= 1.0
    with pytest.raises(ModelingError):
        predict_framework(dense_model(), request, FrameworkConfig( supply=SUPPLY, mtp_draft_len=1))

def test_mtp_rollback_commits_only_accepted_tokens():
    pipe = pipeline(replace(a2_pipe().model, auxiliary=mtp_model().auxiliary))
    calls, original = [], pipe.forward
    def capture(geometry, *args, **kwargs):
        prediction = original(geometry, *args, **kwargs)
        calls.append(prediction.state_out)
        return prediction
    pipe.forward = capture
    predict_framework(pipe, [dict(id="r", prompt_tokens=2, output_tokens=5)],
                      FrameworkConfig(supply=SUPPLY, mtp_draft_len=2, mtp_accept=0))
    record = dict(calls[1].compression_requests)[(0, "r")]
    assert (record.consumed, record.rows) == (3, 0)
    assert calls[1].acceptance.committed == 0
    assert len(calls[1].accepted_tokens) <= 1

def test_r0_projection_matches_hyperconnection_block_width():
    model = make_model(plans=[dense_plan(0, hyper_connections=True, hc_mult=4, hc_sinkhorn_iters=2)])
    r0 = pipeline(model).instantiated.find("R0", layer=0, role="pre")
    assert dict(r0.operator("hc_projection").theta.matrix) == { "K": 64 * 4, "N": (2 + 4) * 4}

def mla_model(tp=1, o_groups=4):
    plan = dict(index=0, attention="mla", ffn="dense", num_heads=8, num_kv_heads=1, head_dim=576, q_lora_rank=128, kv_lora_rank=512,
                qk_nope_head_dim=512, qk_rope_head_dim=64, v_head_dim=128,
                mla_variant="latent_mla", o_groups=o_groups, o_lora_rank=64, intermediate_size=128)
    model = make_model(plans=[plan], hidden_size=1024, num_heads=8, num_kv_heads=1, head_dim=576)
    return model, pipeline(model, feature_overrides={"tp": tp})

def test_mla_prolog_binds_query_kv_norms_store_and_grouped_output():
    model, pipe = mla_model(tp=2)
    a1 = pipe.instantiated.find("A1", layer=0, role="attention")
    names = {operator.name for operator in a1.operators}
    assert {"q_down", "q_norm", "q_up", "kv_down", "kv_norm", "rotary",
            "kv_store", "mla_attention", "grouped_output_a", "grouped_output_b", "tp_allreduce"} <= names
    assert "v_expand" not in names
    contract = a1.template.fusions[0]
    assert set(contract.members) <= names and "kv_store" in contract.members
    compiled = pipe._subgraph(a1)
    result = lower_fusion(a1, compiled.evaluate( WorkloadGeometry.homogeneous(1, 1, 8)).operators, "mla_prolog", supply=SUPPLY)
    assert dict(result.kept)["kv_cache"] > 0
    assert dict(result.fused.materialization)["query"] > 0
    assert dict(a1.operator("q_norm").theta.matrix) == {"elements": 128}
    assert dict(a1.operator("kv_store").theta.matrix)["row_bytes"] == (512 + 64) * 2

def test_shared_consumer_owns_query_projection_and_qk_norm_runs_once():
    plans = [dense_plan(0, owns_long_kv=True), dense_plan(1, kv_owner_layer_id=0, q_output_gate=True)]
    model = make_model(plans=plans)
    pipe = pipeline(model, feature_overrides={"tp": 2})
    n0 = pipe.instantiated.find("N0", layer=1, role="attn_norm")
    assert "qk_norm" not in {operator.name for operator in n0.operators}
    owner = pipe.instantiated.find("A0", layer=0, role="attention")
    consumer = pipe.instantiated.find("A0", layer=1, role="attention")
    assert "qkv_projection" in {operator.name for operator in owner.operators}
    cnames = {operator.name for operator in consumer.operators}
    assert {"q_projection", "qk_norm", "rotary", "output_gate"} <= cnames
    assert "qkv_projection" not in cnames
    assert dict(consumer.operator("q_projection").theta.matrix)["N"] == 2 * 2 * 16

def test_local_kv_rows_and_raw_tp_allreduce_payload():
    model = make_model(plans=[dense_plan(0, num_heads=64, num_kv_heads=2, head_dim=64)], hidden_size=512, num_heads=64, num_kv_heads=2, head_dim=64)
    pipe = pipeline(model, feature_overrides={"tp": 8})
    a0 = pipe.instantiated.find("A0", layer=0, role="attention")
    assert a0.operator("attention").theta.find("kv_bytes") == 1 * (64 + 64) * 2
    assert a0.operator("tp_allreduce").theta.find("wire_bytes") == 512 * 2

def test_recurrent_tp_shards_heads_and_conv_consumes_full_qkv():
    plan = dict(index=0, attention="gdn", ffn="dense", num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
                linear_heads=4, linear_head_dim=8, linear_key_heads=4, linear_key_head_dim=8, linear_value_heads=4, conv_kernel=4)
    model = make_model(plans=[plan])
    for tp in (1, 2):
        pipe = pipeline(model, feature_overrides={"tp": tp})
        a3 = pipe.instantiated.find("A3", layer=0, role="attention")
        key_heads, value_heads = 4 // tp, 4 // tp
        assert dict(a3.operator("conv").theta.matrix)["row_bytes"] == ( 2 * key_heads * 8 + value_heads * 8) * 2
        assert dict(a3.operator("qkvz_projection").theta.matrix)["N"] == ( key_heads * 8 * 2 + value_heads * 8 * 2)

def test_fusion_preserves_external_stores_stages_and_member_readiness():
    quant = dict(name="w8a8", weight_format="int8", activation_format="int8", weight_block=(128, 128), scale_bytes=4)
    pipe = pipeline(dense_model(quant=quant))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    f0 = pipe._subgraph(pipe.instantiated.find("F0", layer=0, role="dense_ffn"))
    fused = lower_fusion(f0.instance, f0.evaluate(geometry).operators, "dequant_swiglu_quant", supply=SUPPLY)
    assert fused.fused.pipe_work("mte3") == 128 * 8
    assert dict(fused.kept)["down_input"] == 128 * 8
    assert len(fused.fused.stage_work) > 2
    first = dict(fused.fused.stage_work[0])
    assert "mte2" in first and fused.fused.duration_ms > max( fused.fused.stage_times)
    baseline = evaluate_serial_split(f0, geometry, fusion=("dequant_swiglu_quant",))
    late = evaluate_serial_split(f0, geometry, fusion=("dequant_swiglu_quant",), readiness={"gate_up": 3})
    assert late.gap_ms == 3 and late.total_ms == baseline.total_ms + 3
    n0 = pipe._subgraph(pipe.instantiated.find("N0", layer=0, role="attn_norm"))
    n0_fused = lower_fusion(n0.instance, n0.evaluate(geometry).operators, "add_rmsnorm_quant", supply=SUPPLY)
    assert n0_fused.fused.pipe_work("mte3") >= 2 * 64 * 8

def test_engine_coalesces_duplicate_pipes_and_completes_empty_branch():
    duplicate = contention_envelope( {"A": [[("hbm", 1000), ("hbm", 1000)]]}, PipeSupply((("hbm", 1000),)))
    assert duplicate.makespan_ms == 2
    assert dict(duplicate.resource_work)["hbm"] == 2000
    assert duplicate.lower_bound_ms == 2
    empty = contention_envelope({"A": []}, PipeSupply((("hbm", 1000),)))
    assert empty.makespan_ms == 0 and dict(empty.branches)["A"] == 0

def moe_model(gated=True):
    plan = dict(index=0, attention="gqa", ffn="moe", num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
                num_experts=4, top_k=2, moe_intermediate_size=32, num_shared_experts=1, shared_intermediate_size=64, shared_expert_gate=gated)
    return make_model(plans=[plan], num_experts=4, top_k=2, moe_intermediate_size=32, num_shared_experts=1)

def test_f2_combines_after_experts_and_f4_gates_join_after_both():
    model = moe_model()
    pipe = pipeline(model, feature_overrides={ "multistream_shared_expert": True, "tp": 2})
    common = pipe.instantiated.find("F4", layer=0, role="shared_ffn")
    assert dict(common.operator("shared_gate_up").theta.matrix)["N"] == 2 * 32
    assert {"shared_gate", "shared_gate_apply", "tp_allreduce"} <= { operator.name for operator in common.operators}
    join = pipe.instantiated.find("F4", layer=0, role="join")
    assert [operator.name for operator in join.operators] == ["join"]
    prediction = pipe.forward(WorkloadGeometry.homogeneous(2, 1, 16))
    rows = prediction.layers[0].rows
    names = [row.cost.name for row in rows]
    assert names.index("combine") > names.index("expert_down")
    assert names.index("join") > names.index("combine")
    assert prediction.layers[0].contention.branches

def test_ungated_f0_and_o0_selected_rows():
    model = make_model(plans=[dense_plan(0, ffn_activation="ungated")])
    f0 = pipeline(model).instantiated.find("F0", layer=0, role="dense_ffn")
    assert dict(f0.operator("gate_up").theta.matrix)["N"] == 128
    assert f0.template.fusions == ()
    for scope, norm_rows in (("all", 32), ("selected", 4)):
        pipe = pipeline(model, feature_overrides={"final_norm_scope": scope})
        rows = pipe.forward(WorkloadGeometry.homogeneous(4, 8, 8)).rows
        norm = next(row for row in rows if row.cost.name == "final_norm")
        head = next(row for row in rows if row.cost.name == "lm_head")
        assert norm.padded_tokens == norm_rows and head.padded_tokens == 4

def test_state_digest_is_hashable_and_layer_cache_is_bounded():
    pipe = pipeline(dense_model())
    state = StateView(compression=((0, CompressionState.start(4, 3)),))
    digest = pipe._state_digest(state, pipe.instantiated.layers[0])
    assert hash(digest) == hash(pipe._state_digest(state, pipe.instantiated.layers[0]))
    full = WorkloadGeometry((RequestGeometry(4, 8, phase="prefill", query_ranges=((0, 4),)),), phase="prefill")
    part = WorkloadGeometry((RequestGeometry(4, 8, phase="prefill", query_ranges=((0, 1),)),), phase="prefill")
    assert geometry_key(full) != geometry_key(part)
    for context in range(10):
        pipe.forward(WorkloadGeometry.homogeneous(1, 1, 8 + context))
        assert len(pipe._cache) <= len(pipe.instantiated.layers)

def test_compression_advances_each_request_independently():
    state = StateView()
    state, produced = state.update_compression( 0, 4, (RequestGeometry(1, 3), RequestGeometry(1, 3)))
    assert produced == 2
    records = dict(state.compression_requests)
    assert records[(0, 0)].consumed == records[(0, 1)].consumed == 4
    state, produced = state.update_compression( 0, 4, (RequestGeometry(1, 4), RequestGeometry(1, 4)))
    assert produced == 0
    assert dict(state.compression_requests)[(0, 0)].consumed == 5

def test_aclgraph_pads_a3_gemm_but_keeps_recurrent_active():
    plan = dict(index=0, attention="gdn", ffn="dense", num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
                linear_heads=4, linear_head_dim=8, linear_key_heads=4, linear_key_head_dim=8, linear_value_heads=4, conv_kernel=4)
    pipe = pipeline(make_model(plans=[plan]))
    active = WorkloadGeometry.homogeneous(1, 1, 16)
    padded = WorkloadGeometry(active.requests, graph_bucket=64)
    rows = pipe.forward(active, padded_geometry=padded, fusion=False).rows
    qkv = next(row for row in rows if row.cost.name == "qkvz_projection")
    delta = next(row for row in rows if row.cost.name == "gated_delta")
    assert qkv.geometry == "padded" and qkv.padded_tokens == 64
    assert delta.geometry == "active" and delta.padded_tokens == 1

def test_dp_completion_is_true_rank_maximum():
    model = make_model(plans=[dense_plan(0), dense_plan(1)])
    pipe = pipeline(model, feature_overrides={"dp": 2})
    prompts = (5, 3, 2, 1)
    result = predict_framework(pipe, [ dict(id=str(index), prompt_tokens=prompt, output_tokens=1)
        for index, prompt in enumerate(prompts)], FrameworkConfig(supply=SUPPLY, max_batch=4))
    rank_totals = []
    for rank in range(2):
        subset = tuple(RequestGeometry(prompts[index], 0, phase="prefill") for index in range(len(prompts)) if index % 2 == rank)
        rank_totals.append(pipe.forward( WorkloadGeometry(subset, phase="prefill")).total_ms)
    assert _frac_close(result.steps[0]["body_ms"], float(max(rank_totals)))
    assert result.steps[0]["body_ms"] < sum(float(value) for value in rank_totals)

def a2_pipe():
    plan = dict(index=0, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
                head_dim=16, intermediate_size=128, cache_kind="window_compressed", window=8, compress_ratio=4, compressor_kind="weighted_pool",
                indexer=True, index_heads=2, index_dim=8, index_topk=4, index_kpool=1, owns_long_kv=True)
    return pipeline(make_model(plans=[plan]))

def test_a0_fusion_keeps_v_slice_and_recovers_supply():
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    instance = pipe.instantiated.find("A0", layer=0, role="attention")
    compiled = pipe._subgraph(instance)
    costs = compiled.evaluate(geometry).operators
    fused = lower_fusion(instance, costs, "qkv_norm_rope", supply=SUPPLY)
    kept = dict(fused.kept)
    v_bytes = 2 * 16 * 2 * 8
    assert kept["value"] == v_bytes
    assert kept["query"] == 4 * 16 * 2 * 8 and kept["key"] == v_bytes
    assert dict(fused.fused.materialization)["value"] == v_bytes
    assert fused.fused.pipe_work("mte3") >= v_bytes
    for pipe, value in fused.fused.times:
        service = sum(dict(stage).get(pipe, 0) for stage in fused.fused.stage_work)
        assert float(value) == pytest.approx(service / SUPPLY.effective(pipe))
    gated = pipeline(make_model(plans=[dense_plan(0, q_output_gate=True)]))
    gated_instance = gated.instantiated.find("A0", layer=0, role="attention")
    gated_fused = lower_fusion(gated_instance, gated._subgraph(gated_instance).evaluate(geometry).operators,
                               "qkv_norm_rope", supply=SUPPLY)
    assert dict(gated_fused.kept)["gate"] == 8 * 4 * 16 * 2
    assert gated_fused.fused.pipe_work("mte3") == 3072 == sum(
        value for _, value in gated_fused.fused.materialization)
    recovered = lower_fusion(instance, costs, "qkv_norm_rope")
    assert recovered.fused.duration_ms == fused.fused.duration_ms
    bad = tuple(replace(cost, times=()) if cost.name == "qkv_projection" else cost for cost in costs)
    with pytest.raises(ModelingError):
        lower_fusion(instance, bad, "qkv_norm_rope")

def test_a2_producers_cache_ownership_and_sharded_reductions():
    pipe = a2_pipe()
    prolog = pipe.instantiated.find("A2", layer=0, role="attention_prolog")
    core = pipe.instantiated.find("A2", layer=0, role="attention")
    assert {"q_direct", "kv_down", "kv_norm", "rotary", "kv_store"} <= { operator.name for operator in prolog.operators}
    rows = pipe.forward(WorkloadGeometry.homogeneous(2, 1, 8), fusion=False).rows
    names = [row.cost.name for row in rows]
    assert names.index("index_cache_store") < names.index("window_attention")
    assert names.index("compress") < names.index("window_attention")
    assert len([row for row in rows if row.cost.name == "kv_store"]) == 1
    with pytest.raises(ModelingError):
        pipe._subgraph(core).evaluate(WorkloadGeometry.homogeneous(1, 1, 8), StateView())
    indexed = StateView().mark_kv(0).mark_index(0)
    with pytest.raises(ModelingError):
        pipe._subgraph(core).evaluate(WorkloadGeometry.homogeneous(1, 4, 0), indexed)
    a1 = pipeline(mla_model(tp=1, o_groups=4)[0])
    assert not any(item.kind == "K0" for item in a1.instantiated.layers[0].subgraphs)
    grouped = a1._subgraph(a1.instantiated.find("A1", layer=0)).operator("grouped_output_a").evaluate( WorkloadGeometry.homogeneous(1, 1, 1))
    assert grouped.get("flops") == 2 * 4 * 256 * 64 and grouped.get("output_bytes") == 4 * 64 * 2
    gdn = dict(index=0, attention="gdn", ffn="dense", num_heads=4, num_kv_heads=2,
               head_dim=16, intermediate_size=128, linear_heads=4, linear_head_dim=8, linear_key_heads=4, linear_key_head_dim=8, linear_value_heads=4,
               conv_kernel=4)
    recurrent = pipeline(make_model(plans=[gdn]), feature_overrides={"tp": 2})
    assert not any(item.kind == "K0" for item in recurrent.instantiated.layers[0].subgraphs)
    assert "tp_allreduce" in {operator.name for operator in recurrent.instantiated.find( "A3", layer=0).operators}
    experts = pipeline(moe_model(), feature_overrides={"tp": 4, "ep": 2, "etp": 2})
    f3 = experts.instantiated.find("F3", layer=0)
    assert "etp_allreduce" in {operator.name for operator in f3.operators}
    geometry = WorkloadGeometry.homogeneous(4, 1, 4)
    grouped = experts._subgraph(f3).operator("expert_gate_up").evaluate(geometry)
    activation = experts._subgraph(f3).operator("expert_activation").evaluate(geometry)
    assert activation.get("elements") == grouped.get("moe_rows") * 16
    plain = pipeline(moe_model())
    canonical_f3 = plain.instantiated.find("F3", layer=0)
    canonical_rows = plain._subgraph(canonical_f3).operator("expert_gate_up").evaluate(geometry)
    canonical = plain._subgraph(canonical_f3).operator("expert_activation").evaluate(geometry)
    assert canonical.get("elements") == canonical_rows.get("moe_rows") * 32 == 256
    dispatch = experts._subgraph(experts.instantiated.find( "F2", layer=0, role="transport")).operator("dispatch").evaluate(
        WorkloadGeometry.homogeneous(4, 1, 4))
    assert dispatch.get("ranks") == 2 and dispatch.get("peer_bytes") == 4 * 2 * 64 * 2

def test_gemma_native_defaults_and_ple_structure():
    native = gemma_config(num_hidden_layers=12, num_kv_shared_layers=0, attention_k_eq_v=False, num_global_key_value_heads=1)
    native.pop("layer_types")
    model = gemma4_model_spec(native)
    assert [plan.index for plan in model.layer_plans if not plan.window] == [5, 11]
    nullable = gemma4_model_spec(gemma_config(num_hidden_layers=12, num_kv_shared_layers=0,
        attention_k_eq_v=False, layer_types=None))
    assert [plan.index for plan in nullable.layer_plans if not plan.window] == [5, 11]
    assert all(plan.head_dim == 512 for plan in model.layer_plans if not plan.window)
    assert all(plan.num_kv_heads == 2 for plan in model.layer_plans)
    pipe = pipeline(gemma4_model_spec(gemma_config()))
    e0 = pipe.instantiated.find("E0")
    assert {"per_layer_model_projection", "per_layer_projection_norm", "ple_combine"} <= { operator.name for operator in e0.operators}
    e1 = pipe.instantiated.find("E1", layer=0)
    assert {"gate_activation", "projection", "post_projection_norm", "residual_add"} <= {operator.name for operator in e1.operators}
    assert e1.template.coverage == "structural"

def test_r1_sources_prefill_outputs_and_mtp_state():
    plans = [dense_plan(index, residual_block_size=2) for index in range(8)]
    r1 = pipeline(make_model(plans=plans))
    prediction = r1.forward(WorkloadGeometry.homogeneous(1, 1, 8))
    assert [r1.instantiated.find("R1", layer=index).operator("score").theta.find("slots") for index in range(8)] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert prediction.layers[-1].state_out.residual.retained == (0, 1, 2, 3)
    second = r1.forward(WorkloadGeometry.homogeneous(1, 1, 9), prediction.state_out)
    assert second.layers[0].state_in.residual.retained == ()
    assert "norm" in {operator.name for operator in r1.instantiated.find("R1", layer=6).operators}
    out = pipeline(dense_model(), feature_overrides={"final_norm_scope": "selected"})
    chunks = WorkloadGeometry((RequestGeometry(4, 0, phase="prefill"),), phase="prefill")
    first = out.forward(chunks, emit_mask=(False,), fusion=False).rows
    second = out.forward(chunks, emit_mask=(True,), fusion=False).rows
    assert not any(row.cost.name in {"lm_head", "sampling"} for row in first)
    assert next(row for row in second if row.cost.name == "lm_head").padded_tokens == 1
    assert next(row for row in second if row.cost.name == "hidden_select").padded_tokens == 1
    verify = WorkloadGeometry((RequestGeometry(3, 2, phase="verify"),), phase="verify")
    checked = out.forward(verify, emit_mask=(True,), fusion=False, acceptance=(2, 2))
    assert next(row for row in checked.rows if row.cost.name == "lm_head").padded_tokens == 3
    assert checked.state_out.acceptance.accepted == 2
    draft = mtp_model()
    one = pipeline(draft)
    low, _ = _draft_cost(one, WorkloadGeometry.homogeneous(1, 1, 8), StateView())
    long_context, _ = _draft_cost(one, WorkloadGeometry.homogeneous(1, 1, 4104), StateView())
    assert long_context > low
    deep = replace(draft, auxiliary=replace( draft.auxiliary, mtp=replace(draft.auxiliary.mtp, layers=12)))
    twelve, _ = _draft_cost(pipeline(deep), WorkloadGeometry.homogeneous(1, 1, 8), StateView())
    assert twelve > 5 * low
    high = predict_framework(one, [dict(id="r", prompt_tokens=4, output_tokens=4)], FrameworkConfig(supply=SUPPLY, max_batch=1,
                                             mtp_draft_len=2, mtp_accept=2))
    assert high.mtp and high.mtp["accepted"] == 2
    assert high.mtp["acceptance_factor"] <= 2

def test_dp_local_completion_ep_join_and_request_state():
    dp = pipeline(make_model(plans=[dense_plan(0), dense_plan(1)]), feature_overrides={"dp": 2})
    result = predict_framework(dp, [ dict(id="long", prompt_tokens=64, output_tokens=1), dict(id="short", prompt_tokens=1, output_tokens=1)],
        FrameworkConfig(supply=SUPPLY, max_batch=2))
    long, short = result.requests
    local_short = dp.forward(WorkloadGeometry( (RequestGeometry(1, 0, phase="prefill"),), phase="prefill")).total_ms
    local_long = dp.forward(WorkloadGeometry( (RequestGeometry(64, 0, phase="prefill"),), phase="prefill")).total_ms
    assert short.completion_ms < long.completion_ms
    assert short.completion_ms == pytest.approx(float(local_short), rel=1e-6)
    assert long.completion_ms == pytest.approx(float(local_long), rel=1e-6)
    two = predict_framework(dp, [dict(id="long", prompt_tokens=64, output_tokens=1),
        dict(id="short", prompt_tokens=1, output_tokens=2)], FrameworkConfig(supply=SUPPLY, max_batch=2))
    local_decode = dp.forward(WorkloadGeometry((RequestGeometry(1, 1, phase="decode"),), phase="decode"))
    assert two.requests[1].tpot_ms == pytest.approx(float(local_decode.total_ms))
    moe_plan = dict(index=0, attention="gqa", ffn="moe", num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128, num_experts=4, top_k=2,
                    moe_intermediate_size=32)
    ep = pipeline(make_model(plans=[dict(moe_plan, index=index) for index in range(2)], layers=2, num_experts=4, top_k=2, moe_intermediate_size=32),
                  feature_overrides={"tp": 2, "ep": 2})
    counts = ((0, (8, 0, 0, 0)), (1, (0, 0, 8, 0)))
    predicted = [ep.forward(WorkloadGeometry( (RequestGeometry(4, 0, phase="prefill"),), phase="prefill",
        routing=((layer, values[2 * rank:2 * rank + 2]) for layer, values in counts))) for rank in range(2)]
    expected = sum(max(item.layers[index].duration_ms for item in predicted)
                   for index in range(2)) + sum(item.duration_ms for item in predicted[0].globals)
    actual = predict_framework(ep, [dict(id="r", prompt_tokens=4, output_tokens=1)], FrameworkConfig(supply=SUPPLY, routing=counts))
    assert _frac_close(actual.steps[0]["body_ms"], float(expected))
    assert actual.sync["ep_skew_ms"] > 0
    uniform = predict_framework(ep, [dict(id="r", prompt_tokens=4, output_tokens=1)], FrameworkConfig(supply=SUPPLY))
    explicit = predict_framework(ep, [dict(id="r", prompt_tokens=4, output_tokens=1)], FrameworkConfig(supply=SUPPLY, routing=(
                                     (0, (2, 2, 2, 2)), (1, (2, 2, 2, 2)))))
    assert uniform.steps[0]["body_ms"] == pytest.approx(explicit.steps[0]["body_ms"])
    state = StateView()
    state, produced = state.update_compression(0, 4, (("old", 3, 1),))
    assert produced == 1 and dict(state.compression_requests)[(0, "old")].consumed == 4
    state, produced = state.update_compression(0, 4, (("old", 4, 1),))
    assert produced == 0 and dict(state.compression_requests)[(0, "old")].consumed == 5
    state, produced = state.update_compression(0, 4, (("old", 0, 1),))
    assert produced == 0 and dict(state.compression_requests)[(0, "old")].consumed == 1
    assert state.retire_compression("old").compression_requests == ()
    pipe = a2_pipe()
    seen = []
    original = pipe.forward
    def capture(geometry, *args, **kwargs):
        prediction = original(geometry, *args, **kwargs)
        seen.append(dict(prediction.state_out.compression_requests))
        return prediction
    pipe.forward = capture
    predict_framework(pipe, [dict(id="old", prompt_tokens=4, output_tokens=1), dict(id="new", arrival_ms=100, prompt_tokens=1, output_tokens=1)],
                      FrameworkConfig(supply=SUPPLY, max_batch=1))
    assert any((0, "old") in item for item in seen)
    assert any((0, "new") in item for item in seen)
    assert all((0, "old") not in item for item in seen[1:])

def _frac_close(left, right): return abs(left - right) <= 1e-9 * max(1.0, abs(right))

def test_prefix_cache_and_mixed_batch_preserve_request_phases():
    pipe = pipeline(dense_model(layers=2))
    prefixed = predict_framework(pipe, [dict(id="a", prompt_tokens=8, output_tokens=2, prefix_tokens=6)], FrameworkConfig(supply=SUPPLY, max_batch=1))
    assert prefixed.steps[0]["active_tokens"] == 2
    mixed = predict_framework(pipe, [dict(id="a", prompt_tokens=2, output_tokens=6), dict(id="b", arrival_ms=1, prompt_tokens=2, output_tokens=1)], FrameworkConfig(supply=SUPPLY, max_batch=2))
    assert any(step["kind"] == "mixed" for step in mixed.steps) and any(set(step["request_ids"]) == {"a", "b"} for step in mixed.steps)

def test_gemma4_per_layer_attention_and_ple_width():
    config = gemma_config(num_hidden_layers=6, global_head_dim=32, num_global_key_value_heads=1,
                          layer_types=["sliding_attention", "full_attention"] * 3)
    model = gemma4_model_spec(config)
    assert [(plan.head_dim, plan.num_kv_heads) for plan in model.layer_plans] == [ (16, 2), (32, 1)] * 3
    assert [plan.kv_owner_layer_id for plan in model.layer_plans] == [0, 1, 2, 3, 2, 3]
    pipe = pipeline(model)
    assert pipe.instantiated.find("A0", layer=0, role="attention").operator( "qkv_projection").theta.find("k_eq_v") == 0
    assert pipe.instantiated.find("A0", layer=1, role="attention").operator( "qkv_projection").theta.find("k_eq_v") == 1
    ple = pipe.instantiated.find("E1", layer=0, role="ple")
    assert dict(ple.operator("projection").theta.matrix) == {"K": 32, "N": 64}

def test_pp_transfer_only_at_stage_cut_and_invalid_tp_rejected():
    plans = [dense_plan(index) for index in range(4)]
    pipe = pipeline(make_model(plans=plans, layers=4), feature_overrides={"pp": 2})
    rows = pipe.forward(WorkloadGeometry.homogeneous(2, 1, 16), fusion=False).rows
    transfers = [row for row in rows if row.cost.name == "stage_transfer"]
    assert len(transfers) == 1 and transfers[0].layer == 1
    assert transfers[0].cost.family == "p2p"
    assert all("stage_transfer" not in {operator.name for operator in instance.operators}
               for layer in pipe.instantiated.layers for instance in layer.subgraphs if instance.kind == "K0")
    plans3 = [dense_plan(0, owns_long_kv=True), dense_plan(1, kv_owner_layer_id=0),
              dense_plan(2, kv_owner_layer_id=0)]
    multicast = pipeline(make_model(plans=plans3, layers=3), feature_overrides={"pp": 3})
    transfer = next(instance.operator("stage_transfer") for layer in multicast.instantiated.layers
                    for instance in layer.subgraphs if instance.kind == "K0"
                    and "stage_transfer" in {operator.name for operator in instance.operators})
    assert transfer.theta.find("destination_stages") == (1, 2)
    predicted = multicast.forward(WorkloadGeometry.homogeneous(1, 1, 31), fusion=False)
    assert sum(row.cost.pipe_work("link") for row in predicted.rows if row.kind == "K0") == 2 * 128
    invalid = make_model(plans=[dense_plan(0, num_heads=10, num_kv_heads=1)], num_heads=10, num_kv_heads=1, hidden_size=160)
    with pytest.raises(ModelingError):
        pipeline(invalid, feature_overrides={"tp": 3})
