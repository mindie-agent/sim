"""Recommended cases preserve source domains, quantization and normalization."""
from copy import deepcopy

import pytest

from mindie_sim import predict, target_graphs
from mindie_sim.compose import apply_va_recommended, compose_case, va_recommended_config
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.implementation_selection import graph_selections
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView


@pytest.mark.parametrize("hardware,source", [
    ("a3", "vllm_ascend_recurrent_kda_arch22"),
    ("a5", "vllm_ascend_recurrent_kda_arch35"),
])
def test_kda_recommendation_selects_native_only_inside_decode_domain(hardware, source):
    raw = compose_case("kimi-k3", hardware=hardware, tp=8, ep=8, layers=4,
                       va_recommended=True)
    for phase, tokens, expected in (
        ("decode", 1, source), ("decode", 8, source),
        ("decode", 9, "analytical"), ("prefill", 4, "analytical"),
    ):
        graph = target_graphs(raw, phase=phase, tokens=tokens, context=4)[0]
        decision, = [row for row in graph_selections(graph) if row["selector"] == "kda"]
        assert decision["implementation"] == expected
        native = [op for op in graph if "recurrent_kda" in op.recipe]
        assert bool(native) == (expected == source)
        assert any(op.family in {"recurrent_scan", "kda_recurrent"} for op in graph)

    case = parse_case(raw)
    views = (RequestView("decode", 1, 4, True, "decode"),
             RequestView("prefill", 4, 0, True, "prefill"))
    work = Frontend()(IterationPlan(views), case.model, case.runtime, case.machine)
    decision, = [row for row in work.implementation_selections if row["selector"] == "kda"]
    assert decision["coverage"] == "analytical_semantic_work"
    assert not any("recurrent_kda" in op.recipe for op in work.ops)
    assert {request for op in work.ops for request in op.requests} == {"decode", "prefill"}


def test_recommended_kda_completes_chunked_mixed_serving():
    raw = compose_case(
        "kimi-k3", hardware="a3", tp=8, ep=8, layers=4,
        prompt_tokens=4, output_tokens=3, concurrency=2,
        max_num_batched_tokens=5, va_recommended=True,
    )
    result = predict(raw, event_limit=0, trace_limit=0)
    assert [request["generated_tokens"] for request in result["requests"]] == [3, 3]
    decisions = result["execution"]["step_summary"]["implementation_selections"]
    assert {row["implementation"] for row in decisions if row["selector"] == "kda"} == {
        "analytical", "vllm_ascend_recurrent_kda_arch22",
    }
    assert result["execution"]["live_tasks"] == 0


def test_recommended_native_kda_propagates_source_domain_failures():
    raw = compose_case("kimi-k3", hardware="a5", tp=8, ep=8, layers=4,
                       va_recommended=True)
    plan = next(plan for plan in raw["model"]["layer_plans"] if plan["attention"] == "kda")
    plan["linear_head_dim"] = plan["linear_key_head_dim"] = 64
    with pytest.raises(ModelingError, match="recurrent_kda_source_head_geometry_or_precision_unsupported"):
        target_graphs(raw, phase="decode", tokens=1, context=4)


@pytest.mark.parametrize("hardware", ("a3", "a5"))
def test_gdn_recommendation_preserves_heterogeneous_query_work(hardware):
    raw = compose_case("qwen3.5-35b-a3b", hardware=hardware, tp=8, ep=8,
                       layers=4, va_recommended=True)
    case = parse_case(raw)
    for lengths, native in (((1, 1), True), ((2, 3), True), ((1, 2), False),
                            ((9, 9), hardware == "a3")):
        views = tuple(RequestView(str(i), q, 16, True, "decode")
                      for i, q in enumerate(lengths))
        work = Frontend()(IterationPlan(views), case.model, case.runtime, case.machine)
        choice, = [row for row in work.implementation_selections
                   if row["selector"] == "gdn_decode"]
        assert choice["implementation"] == ("vllm_ascend_gdn_decode" if native else "analytical")
        assert choice["geometry"]["min_query_tokens"] == min(lengths)
        assert choice["geometry"]["max_query_tokens"] == max(lengths)
        assert {request for op in work.ops for request in op.requests} == {"0", "1"}


def test_recommended_gdn_completes_requests_with_different_mtp_tail_lengths():
    raw = compose_case(
        "qwen3.5-35b-a3b", hardware="a3", tp=8, ep=8, layers=4,
        prompt_tokens=4, output_tokens=4, concurrency=2,
        mtp_proposal_tokens=1, mtp_acceptance_length=1, va_recommended=True,
    )
    raw["workload"]["requests"][0]["output_tokens"] = 2
    result = predict(raw, event_limit=0, trace_limit=0)
    assert [request["generated_tokens"] for request in result["requests"]] == [2, 4]
    choices = result["execution"]["step_summary"]["implementation_selections"]
    assert {choice["implementation"] for choice in choices
            if choice["selector"] == "gdn_decode"} == {"analytical", "vllm_ascend_gdn_decode"}
    assert result["execution"]["live_tasks"] == 0


def test_va_recommendation_preserves_gemma_normalization_in_gdn_only_subset():
    raw = compose_case("qwen3-next-80b-a3b", hardware="a3", tp=4, ep=4, layers=3)
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "residual_norm", "norm_quant"]
    original = deepcopy(raw)
    enriched = apply_va_recommended(raw)
    graph = target_graphs(enriched, phase="decode", tokens=1, context=128)[0]
    norms = [op for op in graph if op.family in {"rmsnorm", "add_rmsnorm", "add_rmsnorm_quant"}]
    assert norms
    assert any(dict(op.metadata).get("norm_weight_offset") == 1.0 for op in norms)
    assert not any(dict(op.metadata).get("native_norm_kind") for op in norms)
    assert raw == original


@pytest.mark.parametrize("alias,canonical,quant_name,expert_weight_format", [
    ("deepseek-v4-flash", "deepseek-v4-flash-0731", "w8a8", "int8"),
    ("deepseek-v4-pro", "deepseek-v4-pro-0813", "w4a8", "int4"),
])
def test_deepseek_v4_alias_and_checkpoint_share_recommendations(alias, canonical, quant_name, expert_weight_format):
    config = va_recommended_config(alias, "a3")
    assert config == va_recommended_config(canonical, "a3")
    assert config["quant"] == quant_name
    assert config["mtp_proposal_tokens"] == (1 if "flash" in canonical else 0)
    raw = compose_case(alias, hardware="a3", tp=8, layers=4, va_recommended=True)
    quant = raw["model"]["quant"]
    assert quant["weight_format"] == "int8"
    assert quant["expert_weight_format"] == expert_weight_format
    assert quant["activation_format"] == "int8"


def test_documented_recommendations_replace_catalog_fp8_but_preserve_explicit_quantization():
    options = dict(hardware="a5", tp=8, layers=4)
    raw_model = compose_case("qwen3.5-397b-a17b", **options)["model"]
    raw_model["quant"] = compose_case("deepseek-v3", **options)["model"]["quant"]
    baseline = compose_case(raw_model, **options)
    assert baseline["model"]["quant"]["weight_block"] == [128, 128]
    recommended = compose_case(raw_model, **options, va_recommended=True)
    quant = recommended["model"]["quant"]
    assert quant["weight_block"] == [1, 32]
    assert quant["scale_format"] == "fp8_e8m0"
    assert quant["scale_bytes"] == 1
    for explicit in ("bf16", baseline["model"]["quant"]):
        expected = compose_case(raw_model, **options, quant=explicit)["model"]["quant"]
        actual = compose_case(raw_model, **options, quant=explicit, va_recommended=True)
        assert actual["model"]["quant"] == expected
        assert actual["runtime"]["metadata"]["va_profile"]["quantization_action"] == "explicit_quant_preserved"


@pytest.mark.parametrize("model", ["deepseek-v3", "deepseek-v4-pro", "glm-5.3", "kimi-k3"])
def test_unconfirmed_a5_profile_keeps_catalog_quantization(model):
    options = dict(hardware="a5", tp=8, layers=4)
    baseline = compose_case(model, **options)
    recommended = compose_case(model, **options, va_recommended=True)
    assert recommended["model"]["quant"] == baseline["model"]["quant"]
    profile = recommended["runtime"]["metadata"]["va_profile"]
    assert profile["quant"] is None
    assert profile["quantization_action"] == "no_profile_override"
    assert profile["quantization_basis"] == "preserve_caller_no_documented_hardware_quantization"
    assert profile["coverage"] == "supported_simulation_subset"


@pytest.mark.parametrize("model", ["qwen3.5-397b-a17b", "glm-5.3-flash"])
def test_a5_documented_mxfp8_alternative_retains_provenance(model):
    raw = compose_case(model, hardware="a5", tp=8, layers=4, va_recommended=True)
    quant = raw["model"]["quant"]
    assert quant["weight_block"] == [1, 32]
    assert quant["activation_format"] == "fp8_e4m3"
    assert quant["scale_format"] == "fp8_e8m0"
    profile = raw["runtime"]["metadata"]["va_profile"]
    assert profile["quantization_action"] == "documented_alternative_applied"
    assert profile["source"] == va_recommended_config(model, "a5")["source"]
    assert profile["source"]["repository"] == "vllm-project/vllm-ascend"
    assert raw["runtime"]["metadata"].get("draft_proposal_tokens", 0) == 0
