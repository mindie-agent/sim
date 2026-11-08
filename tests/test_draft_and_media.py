"""Draft preparation, integer acceptance and media execution dependencies.

[Role] Verify draft model weight terms, speculative acceptance, and media dependencies.
[Invariants] Draft/media dependencies preserve causal ordering and exact byte counts.
[Timeouts & Locks] Tier 0 (<5s), no external locks.
[Side-Effect Policy] Pure read-only in-memory unit tests.
"""
from dataclasses import replace
import json
from pathlib import Path

from mindie_sim import Case, predict
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.catalog.draft_weights import draft_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.scheduler import DraftPolicy, Progress
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def draft_case():
    parsed = Case.from_mapping(case(prompt=4, output=6))
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
                      num_heads=4, num_kv_heads=2, head_dim=8, q_width=32, kv_width=16,
                      vocab_size=128, target_shared_embedding=True, target_shared_lm_head=True)
    return replace(parsed, model=replace(parsed.model, auxiliary=AuxiliarySpec(mtp=draft)),
                   runtime=replace(parsed.runtime, metadata=(
                       ("draft_mode", "mtp"), ("draft_proposal_tokens", 2), ("draft_accept_tokens", 1))))


def test_draft_prefill_populates_cache_after_target_first_token_then_verifies_once():
    parsed = draft_case()
    plain = replace(parsed, runtime=replace(parsed.runtime, metadata=()))
    baseline = predict(plain)
    result = predict(parsed, event_limit=2000)
    assert result["requests"][0]["generated_tokens"] == 6
    assert result["requests"][0]["speculative"]["proposed_tokens"] > 0
    assert result["requests"][0]["ttft_ms"] == baseline["requests"][0]["ttft_ms"]
    commits = [event for event in result["events"] if event["task"].endswith("token_commit:complete")]
    prepared = [event for event in result["events"] if "draft_prefill." in event["task"]
                and not event["task"].endswith(":submit")]
    assert prepared and min(event["start_ms"] for event in prepared) >= commits[0]["end_ms"]
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    view = RequestView("r", 1, 4, True, "decode", draft_proposal_tokens=2)
    work = Frontend()(IterationPlan((view,)), parsed.model, parsed.runtime, parsed.machine)
    target = [op for op in work.ops if op.recipe == "qkv_projection" and not op.op_id.startswith("draft")]
    assert len(target) == 2 and all(op.local_shape[0] == 3 for op in target)
    assert len([op for op in work.ops if op.recipe == "draft_features"]) == 2


def test_prefix_marginals_match_categorical_counts_and_samples_are_reproducible():
    spec = SpeculativeSpec(3, (.8, .4, .1))
    assert spec.expected_committed_tokens() == 2.3
    conditional = SpeculativeSpec(3, (.8, .5, .25), acceptance_mode="conditional")
    assert spec.acceptance_distribution() == conditional.acceptance_distribution()
    parsed = draft_case()
    raw = {"draft_len": 2, "accepted_distribution": [.5, 0, .5]}
    runtime = replace(parsed.runtime, metadata=(("draft_mode", "mtp"), ("speculative", raw)))
    parsed = replace(parsed, runtime=runtime)
    request = parsed.workload.requests[0]
    policy = DraftPolicy.for_request(parsed, request)
    a, b = Progress(request), Progress(request)
    sequence = []
    for step in range(100):
        a.decode_steps = b.decode_steps = step
        value = policy.committed(a, 2)
        assert value == policy.committed(b, 2)
        sequence.append(value)
    assert set(sequence) == {1, 3}
    assert DsparkSpec(2, (.5, .2), confidence=(.1, .8),
                      confidence_threshold=.5).step_spec().draft_len == 0


def test_encoded_media_keeps_sequence_positions_and_skips_only_the_tower():
    path = Path(__file__).parents[1] / "examples/cases/qwen3.8-27b-media.json"
    raw = json.loads(path.read_text())
    raw["workload"] = {"requests": [{"id": "media", "text_tokens": 4, "media_tokens": 4, "output_tokens": 1}]}
    parsed = Case.from_mapping(raw)
    fresh = RequestView("media", 8, 0, True, "prefill")
    ready = replace(fresh, media_encoded=True)
    a = Frontend()(IterationPlan((fresh,)), parsed.model, parsed.runtime, parsed.machine)
    b = Frontend()(IterationPlan((ready,)), parsed.model, parsed.runtime, parsed.machine)
    assert a.valid_tokens == b.valid_tokens == 8
    assert any(op.op_id.startswith("media0:") for op in a.ops)
    assert not any(op.op_id.startswith("media0:") for op in b.ops)
    assert sum(op.flops for op in a.ops) > sum(op.flops for op in b.ops)
    assert parsed.workload.requests[0].prompt_tokens == 8
    patches = parsed.model.vision.patch_tokens_for_media(4)
    qk = next(op for op in a.ops if op.op_id == "media0:vision.l0.qk")
    pv = next(op for op in a.ops if op.op_id == "media0:vision.l0.pv")
    assert qk.flops == pv.flops == 2 * patches**2 * parsed.model.vision.query_width
    assert pv.hbm_bytes == pv.mte2_bytes + pv.mte3_bytes


def test_draft_quantization_and_projection_shards_match_resident_weights():
    from mindie_sim.catalog.quant import quant_preset
    parsed = draft_case()
    quant = quant_preset("w8a8_dynamic")
    draft = replace(parsed.model.auxiliary.mtp, weight_category="mtp")
    model = replace(parsed.model, quant=quant, auxiliary=AuxiliarySpec(mtp=draft))
    views = (RequestView("r", 1, 4, True, "decode", draft_proposal_tokens=2),)
    work = Frontend()(IterationPlan(views), model, parsed.runtime, parsed.machine)
    draft_matrices = [op for op in work.ops if op.op_id.startswith("draft") and op.flops]
    assert all(op.dtype == "bf16" for op in draft_matrices)
    body = [op for op in work.ops if not op.op_id.startswith("draft") and op.recipe == "qkv_projection"]
    assert all(op.dtype == "int8" for op in body)
    term = next(term for term in draft_terms(draft, model) if term.name == "draft_q")
    assert term_bytes(term, quant, parsed.runtime.parallel) == draft.q_width * draft.hidden_size * 2


def test_one_token_response_does_not_prepare_an_unused_draft():
    parsed = draft_case()
    workload = replace(parsed.workload, requests=(replace(parsed.workload.requests[0], output_tokens=1),))
    parsed = replace(parsed, workload=workload)
    result = predict(parsed, event_limit=2000)
    assert not any("draft_prefill." in event["task"] for event in result["events"])
    assert result["metrics"]["makespan_ms"] == result["requests"][0]["ttft_ms"]


def test_kimi_k3_dspark_kv_cache_footprint_and_block_draft_subgraph():
    from mindie_sim.catalog.cache import cache_bytes_per_token_estimate, request_cache_breakdown
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, instantiate_model
    from mindie_sim.schema.parallel import ParallelSpec

    path = Path(__file__).parents[1] / "examples/cases/kimi-k3.json"
    parsed = Case.from_mapping(json.loads(path.read_text()))
    model = parsed.model
    assert model.auxiliary is not None and model.auxiliary.dspark is not None

    # 16 MLA layers: 576 elements * 2B = 1152 B/token/layer -> 18,432 B/token
    # 5 DSpark GQA layers (16 kv_heads, head_dim=64):
    # - TP=8 -> 2 local kv_heads -> 5 * 2 * 2 * 64 * 2B = 2,560 B/token (+13.89%)
    # - TP=1 -> 16 local kv_heads -> 5 * 2 * 16 * 64 * 2B = 20,480 B/token (+111.11%)
    p_tp8 = ParallelSpec(tp=8, dp=4, ep=32, pp=1)
    base_tp8 = request_cache_breakdown(model, p_tp8, 4096)
    dspark_tp8 = request_cache_breakdown(model, p_tp8, 4096, draft_kind="dspark")
    assert "draft_kv" not in base_tp8["by_component"]
    assert dspark_tp8["by_component"]["draft_kv"] == 4096 * 2560.0
    assert dspark_tp8["total_bytes"] - base_tp8["total_bytes"] == 4096 * 2560.0

    p_tp1 = ParallelSpec(tp=1, dp=32, ep=32, pp=1)
    base_tp1 = request_cache_breakdown(model, p_tp1, 4096)
    dspark_tp1 = request_cache_breakdown(model, p_tp1, 4096, draft_kind="dspark")
    assert dspark_tp1["by_component"]["draft_kv"] == 4096 * 20480.0
    assert base_tp1["by_component"]["mla_kv"] == 4096 * 24 * 1152.0
    bpt_base, _ = cache_bytes_per_token_estimate(model, p_tp1, 4096)
    bpt_dspark, _ = cache_bytes_per_token_estimate(model, p_tp1, 4096, draft_kind="dspark")
    assert bpt_dspark - bpt_base == 20480.0

    # Memory capacity accounting with DSpark enabled vs disabled
    rt_base = replace(parsed.runtime, parallel=p_tp8, pp_layer_partition=(), metadata=())
    rt_dspark = replace(parsed.runtime, parallel=p_tp8, pp_layer_partition=(),
                        dspark=DsparkSpec(7, (0.9,) * 7), metadata=(("draft_mode", "dspark"),))
    mem_base = Memory(replace(parsed, runtime=rt_base))
    mem_dspark = Memory(replace(parsed, runtime=rt_dspark))
    assert mem_dspark.weights[0] > mem_base.weights[0]
    assert mem_dspark.workspace[0] > mem_base.workspace[0]
    assert mem_dspark.cache_bytes(4096)[0] - mem_base.cache_bytes(4096)[0] == 4096 * 2560.0

    # Subgraph D0 instantiation for Kimi-K3 DSpark
    inst = instantiate_model(model, FeatureConfig.from_model(model, tp=8, ep=32, dp=4))
    d0 = next(g for g in inst.globals if g.kind == "D0")
    assert d0.template.variant == "dspark"
    proj = d0.operator("draft_projection")
    assert proj.theta.find("K") == 5 * model.hidden_size
    heads = d0.operator("draft_heads")
    assert heads.theta.find("noncausal_block") == 1
    assert heads.theta.find("markov_rank") == 256
    assert heads.theta.find("draft_len") == 7


def _draft_kv_delta(model, tp: int, tokens: int = 4096) -> float:
    from mindie_sim.catalog.cache import cache_bytes_per_token_estimate
    from mindie_sim.schema.parallel import ParallelSpec

    parallel = ParallelSpec(tp=tp, dp=1, ep=1, pp=1)
    base, _ = cache_bytes_per_token_estimate(model, parallel, tokens)
    drafted, _ = cache_bytes_per_token_estimate(
        model, parallel, tokens, draft_kind="dspark")
    return drafted - base


def _assert_mla_d0(model, *, hidden: int) -> None:
    from mindie_sim.catalog.draft_weights import draft_terms
    from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, instantiate_model

    draft = model.auxiliary.dspark
    assert draft is not None
    assert draft.attention_kind == "mla"
    assert draft.mla_variant == "latent_mla"
    assert draft.layers == 5
    assert draft.kv_cache_elements == 576
    assert draft.head_dim == 192
    assert draft.q_lora_rank == 1536
    assert draft.kv_lora_rank == 512
    assert draft.qk_nope_head_dim == 128
    assert draft.qk_rope_head_dim == 64
    assert draft.v_head_dim == 128
    assert draft.q_width == 64 * 192
    assert draft.kv_width == 576
    assert draft.num_kv_heads == 1
    assert draft.target_layer_ids == (2, 23, 47, 71, 89)
    wkv = next(term for term in draft_terms(
        draft, model, standalone_heads_only=False) if term.name == "draft_wkv")
    out = next(term for term in draft_terms(
        draft, model, standalone_heads_only=False) if term.name == "draft_o")
    assert wkv.rows == 512 + 64
    assert out.cols == 64 * 128
    inst = instantiate_model(model, FeatureConfig.from_model(model, tp=1, ep=1, dp=1))
    d0 = next(graph for graph in inst.globals if graph.kind == "D0")
    for name in (
        "draft_layer0_attn_norm", "draft_layer0_q_a", "draft_layer0_q_norm",
        "draft_layer0_q_b", "draft_layer0_kv_a", "draft_layer0_kv_norm",
        "draft_layer0_kv_store", "draft_layer0_attention",
        "draft_layer0_o_projection", "draft_layer0_attn_residual",
        "draft_layer0_ffn_norm", "draft_layer0_gate_up", "draft_layer0_activation",
        "draft_layer0_down", "draft_layer0_ffn_residual",
    ):
        assert d0.operator(name).name == name
    assert "draft_layer0_qkv" not in d0.template.operator_names
    assert d0.operator("draft_layer0_q_a").theta.find("K") == hidden
    assert d0.operator("draft_layer0_q_a").theta.find("N") == 1536
    assert d0.operator("draft_layer0_q_b").theta.find("K") == 1536
    assert d0.operator("draft_layer0_q_b").theta.find("N") == 64 * 192
    assert d0.operator("draft_layer0_kv_a").theta.find("K") == hidden
    assert d0.operator("draft_layer0_kv_a").theta.find("N") == 576
    assert d0.operator("draft_layer0_o_projection").theta.find("K") == 64 * 128
    assert d0.operator("draft_layer0_o_projection").theta.find("N") == hidden
    assert d0.operator("draft_layer4_q_a").theta.find("N") == 1536


def test_kimi_k3_dspark_block5_mla_catalog_and_draft_subgraph():
    from mindie_sim.model_catalog import load_model

    model = load_model("kimi-k3-dspark-block5")
    meta = dict(model.metadata)
    assert meta["model_coverage"] == "complete"
    assert meta["unrecognized_config_fields"] == {}
    assert model.attention == "mla"
    assert model.layers == 5
    assert all(plan.attention == "mla" and plan.kv_cache_elements == 576
               and plan.noncausal_attention is False for plan in model.layer_plans)
    assert model.auxiliary is not None and model.auxiliary.standalone_draft
    assert model.auxiliary.dspark.block_size == 5
    for tp in (1, 2, 4, 8):
        assert _draft_kv_delta(model, tp) == 5760.0
    _assert_mla_d0(model, hidden=model.hidden_size)


def test_kimi_k3_dspark_block_size_five_selects_mla_draft():
    from mindie_sim.schema.model import ModelSpec

    raw = json.loads((Path(__file__).parents[1] / "mindie_sim/model_configs/kimi-k3.json").read_text())
    raw["text_config"]["dspark_block_size"] = 5
    model = ModelSpec.from_mapping(raw)
    assert model.auxiliary is not None and model.auxiliary.dspark is not None
    assert model.auxiliary.dspark.kind == "dspark"
    assert model.auxiliary.dspark.block_size == 5
    for tp in (1, 2, 4, 8):
        assert _draft_kv_delta(model, tp) == 5760.0
    _assert_mla_d0(model, hidden=model.hidden_size)
