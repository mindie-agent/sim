"""Composable HF frontend lowering, deployment aliases and wrapper overrides.

[Role]
  Exercise the public HF config frontend: unknown-field visibility, explicit
  partial-attention holes, shared-KV ownership across TP/PP, quantization and
  precision proxies, deployment/checkpoint aliases, explicit wrapper overrides,
  retained GLM/Kimi source variants, Qwen3.8 hybrid alignment and situational
  activation declarations.

[Invariants]
  * An unknown field never disables known work and never disappears silently.
  * A missing attention core stays an explicit hole: structure export remains
    inspectable while every timing entry refuses to fabricate a duration.
  * Only the declared mechanism changes: hardware, ordinary matrix mapping,
    accepted-token state ABI and unrelated fused schedules are conserved.
  * Aliases and wrappers change representation selection, never the physical
    graph or the coverage claim.

[Timeouts & Locks]
  Bounded in-memory HF configs, small graphs and fixture reads; no device,
  network, subprocess or locks.

[Side-Effect Policy]
  Read-only repository fixtures and examples; all execution is in memory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.hf_configs import case, config, run_case
from mindie_sim import (
    compose_case, graph_costs, operator_costs, predict, study, target_graphs, target_operators,
)
from mindie_sim.catalog.cache import layer_cache_components, request_cache_breakdown
from mindie_sim.catalog.hf_value_formats import _role_value_formats, model_value_formats
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.weights import cache_row_bytes_by_layer, engram_table_terms
from mindie_sim.catalog.weights import layer_weight_terms, stage_weight_bytes
from mindie_sim.model_catalog import ARCHITECTURE_PRESET_CATALOG, load_model
from mindie_sim.prediction.efficiency import analyze_network_efficiency, solve_sla_sizing
from mindie_sim.prediction.operators.vector.gated_activation import gated_activation
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.structure import export_structure

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
COMPOSABLE = ROOT / "tests" / "fixtures" / "composable-hf"
RECEIPT = ROOT / "tests" / "fixtures" / "verification" / "glm-supported-variants.json"
HIGH_CONCURRENCY = ROOT / "examples" / "cases" / "high-concurrency-20261030"
KIMI_SOURCE = HIGH_CONCURRENCY / "kimi-source"
SKUS = (REPRESENTATIVE_SKUS["a3"], REPRESENTATIVE_SKUS["a5"])
SHARED_KV_LAYERS = [0, 0]
GLM_VARIANTS = (("glm53flash-a5-tp8-c3", "attention-only"), ("glm53flash-a5-tp8-c3", "attention-plus-down"))
KIMI_SOURCES = (("a5_kimi_tp8_e16_bf16_c1_p4k", 93, 69, "vllm_ascend_recurrent_kda_arch35"),
                ("a3_kimi4_tp8_w4a8_c1", 4, 3, "vllm_ascend_recurrent_kda_arch22"))

def _wrapped() -> dict:
    raw = case()
    raw["machine"]["device"]["compute_peaks"]["int8"] = 2e9
    raw["model"] = { "hf_config": {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 32,
                      "num_attention_heads": 4, "num_key_value_heads": 2,
                      "intermediate_size": 64, "vocab_size": 128},
        "quantization": "w8a8", "layers": 1,
        "metadata": {"model_coverage": "partial", "unmodeled_features": ["custom_capture_layout"]}}
    return raw

def _deployment_case(model: str, quantization: str, sku: str) -> dict:
    return { "machine": {"backend": "ascend", "sku": sku, "num_devices": 8},
        "model": {"preset": model, "quantization": quantization, "layers": 1, "vocab_size": 128, "auxiliary": None},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {"tp": 8, "max_num_batched_tokens": 16,
                    "kernels": {"programs": {"matmul": {"kind": "steady_matmul"}}}}}

def _kimi_case(name: str, layers: int | None = None) -> dict:
    raw = json.loads((HIGH_CONCURRENCY / "kimi-variants" / f"{name}.json").read_text())
    if layers is not None:
        raw["model"]["layers"] = layers
    return raw

def test_unknown_config_fields_do_not_disable_the_known_graph_and_stay_visible() -> None:
    original = config()
    custom = {**original, "model_type": "step2_structural_test", "custom_cross_layer": {"mode": "not_understood"}}
    parsed = parse_case(run_case(custom))
    baseline = graph_costs(run_case(original), target_graphs(run_case(original), phase="prefill", tokens=3))
    result = graph_costs(run_case(custom), target_graphs(run_case(custom), phase="prefill", tokens=3))
    assert result["duration_ms"] == baseline["duration_ms"]
    assert result["model_coverage"] == "partial"
    assert "unrecognized_config_field:custom_cross_layer" in result["unmodeled_features"]
    reported = dict(parsed.model.metadata)["unrecognized_config_fields"]
    assert reported["custom_cross_layer"] == custom["custom_cross_layer"]
    serving = predict(run_case(custom))
    isolated = operator_costs(run_case(custom), target_operators(run_case(custom), phase="prefill", tokens=3))
    assert serving["status"] == "modeled"
    assert isolated["model_coverage"] == serving["model_coverage"] == "partial"
    assert ModelSpec.from_mapping(parsed.model.to_dict()) == parsed.model
    assert "model_coverage" not in custom

def test_partial_attention_keeps_projections_and_refuses_fabricated_timing() -> None:
    raw = run_case({**config(), "layer_types": ["custom_kernel", "full_attention"]})
    model = parse_case(raw).model
    assert [p.attention for p in model.layer_plans] == ["projection_only", "gqa"]
    ops = target_operators(raw, phase="prefill", tokens=3)
    first = [op for op in ops if op.layer == 0]
    graphs = target_graphs(raw, phase="prefill", tokens=3)
    assert any(op.family == "barrier" and dict(op.metadata).get("unmodeled_component") == "custom_kernel"
               for graph in graphs.values() for op in graph)
    assert not any(op.family == "attention" for op in first)
    assert next(op for op in first if op.op_id.endswith("known_qkv")).flops == 2 * 3 * (4 + 2 * 2) * 8 * 32
    assert any(op.family == "swiglu" for op in first)
    structure = export_structure(raw, phase="prefill", tokens=3)
    assert structure["nodes"]
    hints = structure["unmodeled_structure_onboarding_hints"]
    assert hints and all(h["schema"] == "mindie-unmodeled-structure-hint/1" for h in hints)
    for entry in (lambda: graph_costs(raw, target_graphs(raw, phase="prefill", tokens=3)),
                  lambda: operator_costs(raw, target_operators(raw, phase="prefill", tokens=3)),
                  lambda: predict(raw)):
        with pytest.raises(UnmodeledCapabilityError) as raised:
            entry()
        assert raised.value.result_available is False
        assert raised.value.onboarding_hint["ssot_owner_repo"] == "mindie-sim"

@pytest.mark.parametrize("entry", (
    "predict", "operator_costs", "analyze_network_efficiency", "solve_sla_sizing",
))
def test_timing_entries_refuse_an_unmodeled_attention_core(entry):
    raw = compose_case("qwen3-32b", hardware=SKUS[1], layers=2)
    raw["model"]["layer_plans"][0].update(
        attention="unmodeled", architecture="synthetic_unmodeled_attention")
    calls = {
        "predict": lambda: predict(raw),
        "operator_costs": lambda: operator_costs(raw, target_operators(raw, phase="prefill", tokens=3)),
        "analyze_network_efficiency": lambda: analyze_network_efficiency(raw, phase="prefill", tokens=3),
        "solve_sla_sizing": lambda: solve_sla_sizing(raw, sla_ms=100),
    }
    with pytest.raises(UnmodeledCapabilityError) as raised:
        calls[entry]()
    error = raised.value
    assert error.kind == "unmodeled_capability"
    assert error.result_available is False
    assert error.entry == entry
    hints = error.unmodeled_structure_onboarding_hints
    assert hints, error.code
    for hint in hints:
        assert hint["schema"] == "mindie-unmodeled-structure-hint/1"
        assert hint["ssot_owner_repo"] == "mindie-sim"
        assert hint["onboarding_targets"]
        assert hint["observed_abi_summary"]["charging_policy"] == "no_timing_fabricated_for_unmodeled_core"
    assert error.onboarding_hint["schema"] == "mindie-unmodeled-structure-hint/1"
    assert error.onboarding_hint["ssot_owner_repo"] == "mindie-sim"

@pytest.mark.parametrize("tp", (1, 4))
@pytest.mark.parametrize("pp", (1, 2))
@pytest.mark.parametrize("phase,query,context", (("prefill", 3, 0), ("decode", 1, 16)))
def test_shared_kv_tp_pp_preserves_payload_weights_and_real_ready(
    tp: int, pp: int, phase: str, query: int, context: int,
) -> None:
    raw = run_case({**config(), "kv_source_layer_ids": SHARED_KV_LAYERS, "qk_norm": True}, tp=tp, pp=pp, prompt=3)
    parsed = parse_case(raw)
    ops = target_graphs(raw, phase=phase, tokens=query, context=context)[0]
    heads, kv_heads = 4 // tp, max(1, 2 // tp)
    assert [op.flops for op in ops if op.family == "qkv_projection"] == [
        2 * query * 32 * (heads + 2 * kv_heads) * 8, 2 * query * 32 * heads * 8]
    weights = [{term.name: term for term in layer_weight_terms(layer, parsed.model)}
               for layer in parsed.model.layer_plans]
    assert weights[0]["k_proj"].logical_parameters == 2 * 8 * 32
    assert not {"k_proj", "v_proj", "k_norm"} & weights[1].keys()
    stores = [op for op in ops if op.recipe == "kv_store"]
    assert len(stores) == 1
    assert stores[0].mte3_bytes == query * 2 * kv_heads * 8 * 2
    cache = [layer_cache_components(parsed.model, layer, parsed.runtime.parallel,
                                    context + query, capacity_padded=False)
             for layer in parsed.model.layer_plans]
    assert sum(cache[0].values()) == (context + query) * 2 * kv_heads * 8 * 2
    assert cache[1] == {}
    consumer = next(op for op in ops if op.recipe == "shared_cache_dependency")
    transfers = [op for op in ops if op.recipe == "shared_kv_transfer"]
    if pp == 1:
        assert not transfers and stores[0].op_id in consumer.deps
    else:
        assert transfers[0].deps == (stores[0].op_id,) and transfers[0].op_id in consumer.deps
        flows = [flow for flow, _ in Topology(parsed.machine, parsed.runtime.parallel).messages(
            transfers[0], 0, transfers[0].link_bytes)]
        assert len(flows) == tp
        assert sum(flow.payload_bytes for flow in flows) == tp * transfers[0].link_bytes
    report = graph_costs(parsed, target_graphs(raw, phase=phase, tokens=query, context=context),
                         input_ready_ms=7, isolated=False)
    timeline = {row["op_id"]: row for row in report["timeline"]}
    assert all(row["ready_ms"] >= 7 for row in timeline.values())
    assert timeline[consumer.op_id]["ready_ms"] >= timeline[stores[0].op_id]["complete_ms"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

def test_precision_is_independent_from_kv_and_unknown_quant_has_explicit_proxy() -> None:
    native = ModelSpec.from_mapping({**config(), "torch_dtype": "float32", "kv_dtype": "bf16"})
    assert native.activation_dtype == native.compute_dtype == "fp32"
    assert native.quant.matrix_bytes(32, 128, "embedding") == 4 * 32 * 128
    assert native.quant.kv_dtype == "bf16"
    assert predict(run_case(native.to_dict()))["status"] == "modeled"
    for extra in ({"weight_dtype": "fp32"}, {"quantization_config": {"quant_method": "fp8", "fmt": "e5m2",
                                           "activation_scheme": "static",
                                           "weight_block_size": [128, 128]}},
                  {"weight_dtype": "fp8"}):
        model = ModelSpec.from_mapping({**config(), **extra})
        assert dict(model.metadata)["model_coverage"] == "partial"
        assert any("proxy" in value for value in dict(model.metadata)["degraded_submodules"])

def test_declared_fp8_blocks_apply_to_experts_and_survive_roundtrip() -> None:
    raw = {**config(), "n_routed_experts": 4, "num_experts_per_tok": 2,
           "moe_intermediate_size": 64, "quantization_config": {
               "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
               "weight_block_size": [128, 128]}}
    model = ModelSpec.from_mapping(raw)
    assert model.quant.matrix_bytes(130, 129, "routed_expert") == 130 * 129 + 2 * 2 * 4
    assert model.quant.format_for("router") == "bf16"
    assert ModelSpec.from_mapping(model.to_dict()).quant == model.quant

def test_global_sliding_window_is_work_and_attention_bias_is_a_visible_gap() -> None:
    full = run_case(config())
    windowed = run_case({**config(), "model_type": "mistral", "sliding_window": 4})

    def attention_flops(raw: dict) -> float:
        return sum(op.flops for op in target_operators(raw, phase="decode", tokens=1, context=32)
                   if op.family == "attention")
    assert attention_flops(windowed) / attention_flops(full) == pytest.approx(4 / 33)
    biased = parse_case(run_case({**config(), "attention_bias": True})).model
    assert "custom_submodule:attention_bias" in dict(biased.metadata)["unmodeled_features"]

@pytest.mark.parametrize("declared", ({"1": 0}, [0, 0]))
def test_shared_owner_alias_applies_to_standard_and_generic_families(declared) -> None:
    for family in ("llama", "qwen3"):
        raw = run_case({**config(), "model_type": family, "kv_consumer_to_owner": declared})
        model = parse_case(raw).model
        assert [plan.kv_owner_layer_id for plan in model.layer_plans] == [0, 0]
        assert len([op for op in target_operators(raw, phase="prefill", tokens=3) if op.recipe == "kv_store"]) == 1

@pytest.mark.parametrize("family", ("deepseek_future", "glm_future", "qwen_next", "kimi_future"))
def test_unverified_family_variant_is_executable_but_never_claimed_complete(family: str) -> None:
    raw = run_case({**config(), "model_type": family, "hidden_act": "gelu"})
    model = parse_case(raw).model
    assert "ffn_activation:gelu:silu_proxy" in dict(model.metadata)["unmodeled_features"]
    assert predict(raw)["model_coverage"] == "partial"

def test_declared_mla_without_indexer_keeps_direct_query() -> None:
    raw = run_case({**config(), "model_type": "glm4_moe", "kv_lora_rank": 16,
                    "q_lora_rank": None, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
                    "v_head_dim": 8})
    model = parse_case(raw).model
    assert all(plan.attention == "mla" and not plan.indexer and plan.q_lora_rank == 0 for plan in model.layer_plans)
    assert predict(raw)["status"] == "modeled"

@pytest.mark.parametrize("name", ("minimax-m1-80k", "minimax-text-01", "step3", "baichuan2-7b-base"))
def test_official_external_configs_load_without_weights_and_roundtrip(name: str) -> None:
    raw = json.loads((COMPOSABLE / f"{name}.json").read_text())
    model = ModelSpec.from_mapping(raw)
    assert model.layers == raw.get("num_hidden_layers", raw.get("text_config", {}).get("num_hidden_layers"))
    assert model.parameter_count() > 0
    assert ModelSpec.from_mapping(model.to_dict()) == model
    path = str(COMPOSABLE / f"{name}.json")
    assert load_model(path) == replace(model, name=path)
    if name.startswith("minimax"):
        linear = model.layer_plans[0]
        assert linear.attention == "projection_only"
        assert (linear.num_kv_heads, linear.num_heads) == (64, 64)
        assert linear.recurrent_state_elements == 64 * 128 * 128
    if name == "step3":
        assert model.layer_plans[0].attention == "unmodeled"
        assert model.layer_plans[3].ffn == "dense" and model.layer_plans[4].ffn == "moe"

@pytest.mark.parametrize("name", ("minimax-m1-80k", "minimax-text-01", "step3"))
def test_external_custom_models_run_known_backbone_with_explicit_holes(name: str) -> None:
    raw_config = json.loads((COMPOSABLE / f"{name}.json").read_text())
    text = raw_config.get("text_config", raw_config)
    text.update(num_hidden_layers=2, hidden_size=32, intermediate_size=64, vocab_size=128,
                num_attention_heads=4, head_dim=8, num_key_value_heads=2,
                moe_intermediate_size=16)
    if "attn_type_list" in text:
        text.update(attn_type_list=[0, 1], num_local_experts=4, num_experts_per_tok=2)
    else:
        text.update(moe_layers_enum="1", moe_num_experts=4, moe_top_k=2, share_expert_dim=16, share_q_dim=16)
    raw = run_case(raw_config)
    model = parse_case(raw).model
    assert [plan for plan in model.layer_plans if plan.attention in {"unmodeled", "projection_only"}]
    structure = export_structure(raw, phase="prefill", tokens=3)
    nodes = structure["nodes"]
    assert any(node.get("family") == "barrier" for node in nodes)
    assert any(node.get("family") in {"swiglu", "down_projection"} for node in nodes)
    hints = structure["unmodeled_structure_onboarding_hints"]
    assert hints and all(hint["schema"] == "mindie-unmodeled-structure-hint/1" for hint in hints)
    with pytest.raises(UnmodeledCapabilityError):
        predict(raw)

@pytest.mark.parametrize("key", tuple(ARCHITECTURE_PRESET_CATALOG))
def test_architecture_presets_reach_real_frontend_and_engine(key: str) -> None:
    model = load_model(key)
    one = replace(model, layers=1, body_layers=1, layer_plans=model.layer_plans[:1], auxiliary=None)
    raw = run_case(one.to_dict(), prompt=1, output=1)
    raw["machine"]["device"]["memory_bytes"] = 10**15
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul"}}}
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=1), isolated=False)
    assert graph["duration_ms"] > 0 and graph["execution"]["live_tasks"] == 0
    assert graph["model_coverage"] == "partial"
    assert predict(raw)["status"] == "modeled"

@pytest.mark.parametrize("short,canonical", ( ("deepseek-v4-flash", "deepseek-v4-flash-0731"),
    ("deepseek-v4-pro", "deepseek-v4-pro-0813"),
    ("deepseek-v4.1", "deepseek-v4.1-flash"), ("qwen3.8", "qwen3.8-27b")))
def test_model_short_labels_preserve_geometry_and_quantization(short: str, canonical: str) -> None:
    model = load_model(canonical)
    assert load_model(short) == model == load_model(short.upper())

def test_quant_aliases_preserve_real_packed_bytes_and_checkpoint_exclusions() -> None:
    w4 = quant_preset("w4a8")
    assert w4 == quant_preset("w4a8_dynamic")
    assert w4.matrix_bytes(3, 5, "routed_expert") == 3 * 3 + 3 * 4
    assert w4.matrix_bytes(3, 5, "attn") == 3 * 5 + 3 * 4 * 2
    assert quant_preset("w8a8") == quant_preset("w8a8_dynamic")
    assert quant_preset("bf16").matrix_bytes(3, 5, "routed_expert") == 3 * 5 * 2
    fp8 = quant_preset(" FP8 ")
    assert fp8 == quant_preset("w8a8_mxfp8")
    assert fp8.matrix_bytes(3, 33, "dense_mlp") == 3 * 33 + 3 * 2
    assert fp8.matrix_bytes(3, 33, "gdn") == 3 * 33 * 2
    assert dict(fp8.source)["checkpoint"] == "Eco-Tech/Qwen3.8-27B-w8a8-mxfp8"

def test_quant_alias_and_canonical_execute_identical_public_programs() -> None:
    raw = _deployment_case("qwen3.8", "fp8", SKUS[1])
    reference = deepcopy(raw)
    reference["model"]["quantization"] = "w8a8_mxfp8"
    assert target_graphs(raw, phase="prefill", tokens=1) == target_graphs( reference, phase="prefill", tokens=1)
    actual, expected = predict(raw), predict(reference)
    assert actual["requests"][0]["status"] == "served"
    assert actual["execution"]["serviced"] == expected["execution"]["serviced"]
    assert actual["metrics"] == expected["metrics"]

@pytest.mark.parametrize("model", ("deepseek-v4-flash", "qwen3.8", "glm-5.3"))
@pytest.mark.parametrize("quantization", ("w8a8", "w4a8"))
def test_supported_representations_reach_public_serving(model: str, quantization: str) -> None:
    raw = _deployment_case(model, quantization, SKUS[0])
    parsed = parse_case(raw)
    result = predict(raw)
    assert result["status"] == "modeled"
    assert result["requests"][0]["status"] == "served"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["execution"]["work"]["flops"] > 0
    coverage = dict(parsed.model.metadata).get("model_coverage", "complete")
    assert result["model_coverage"] == coverage

def test_fp8_alias_does_not_grant_an_undeclared_matrix_capability() -> None:
    raw = _deployment_case("qwen3.8", "fp8", SKUS[0])
    assert parse_case(raw).model.quant == quant_preset("w8a8_mxfp8")
    with pytest.raises(ModelingError, match="compute_precision_rate_required:fp8_e4m3"):
        predict(raw)
    raw["machine"]["sku"] = SKUS[1]
    supported = predict(raw)
    assert supported["requests"][0]["status"] == "served"
    assert supported["execution"]["live_tasks"] == 0

@pytest.mark.parametrize("case_id,name", GLM_VARIANTS)
def test_retained_compatible_variants_change_only_supported_programs(case_id: str, name: str) -> None:
    receipt = json.loads(RECEIPT.read_text())
    row = next(row for row in receipt["cases"] if row["case_id"] == case_id)
    variant = next(v for v in row["variants"] if v["name"] == name)
    original = json.loads((ROOT / row["baseline"]["path"]).read_text())
    raw = json.loads((ROOT / variant["path"]).read_text())
    assert raw["machine"] == original["machine"]
    assert raw["runtime"].get("routing") == original["runtime"].get("routing")
    assert raw["model"]["quant"] == original["model"]["quant"]
    raw["model"]["layers"] = original["model"]["layers"] = 4
    geometry = row["geometry"]
    kwargs = dict(scope="body", phase="decode", tokens=geometry["tokens_per_sequence"],
                  sequences=geometry["sequences_per_replica"],
                  context=geometry["context_tokens"])
    baseline, graphs = target_graphs(original, **kwargs), target_graphs(raw, **kwargs)
    assert graphs.effective_kernels["programs"]["matmul"] == baseline.effective_kernels["programs"]["matmul"]
    for prior, after in zip((op for op in baseline[0] if op.family == "recurrent_scan"),
                            (op for op in graphs[0] if op.family == "recurrent_scan"), strict=True):
        assert prior == after
    result = graph_costs(raw, graphs, input_ready_ms=7, external_inputs={"analysis_input": 7}, isolated=False)
    assert result["model_coverage"] == "partial"
    assert result["prediction_scope"] == "known_components_with_declared_degradations"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    attention = [row for row in result["timeline"] if row["family"] == "attention"]
    assert len(attention) == 1
    if name == "attention-plus-down":
        index = next(i for i, op in enumerate(graphs[0]) if op.op_id.endswith(".expert_down"))
        down = operator_costs(raw, graphs[0][index:index + 1])["operators"][0]
        plan = down["operator"]["metadata"]["program_execution"]
        assert plan["source_program"]["branch"].startswith("MX NZ FP8")
        assert sum(plan["source_mapping"]["group_rows"]) == geometry["query_rows"] * 8
        assert down["serviced"]["fixpipe@r0:s0"] == pytest.approx( 2 * geometry["query_rows"] * 8 * 4096)
        assert not any("vector" in key or "mte3" in key for key in down["serviced"])

def test_hf_wrapper_preset_and_coverage_reach_the_physical_graph() -> None:
    raw = _wrapped()
    model = parse_case(raw).model
    assert model.quant.weight_format == model.quant.activation_format == "int8"
    assert model.activation_dtype == model.compute_dtype == "bf16"
    assert model.layers == 1
    operators = target_operators(raw, phase="prefill", tokens=4)
    assert any(dict(op.metadata).get("weight_format") == "int8" for op in operators)
    assert any(op.family == "quantize" for op in operators)
    report = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4))
    assert report["unmodeled_features"] == ["custom_capture_layout"]
    assert report["execution"]["live_tasks"] == 0

def test_study_hf_representation_overrides_stay_outside_the_native_config() -> None:
    raw = _wrapped()
    report = study([{"id": "int8", "case": raw},
                    {"id": "bf16", "case": raw, "overrides": {"model": {"quantization": "bf16"}}}],
                   [{"id": "p4", "phase": "prefill", "tokens": 4}], level="graph")
    assert report["status"] == "completed"
    one, two = report["evaluations"]
    assert one["operators"]["work"]["hbm_bytes"] < two["operators"]["work"]["hbm_bytes"]
    assert all(row["model_coverage"] == "partial" for row in report["evaluations"])

def test_existing_outer_hf_dimensions_do_not_overwrite_study_updates() -> None:
    from mindie_sim.study import _override_case
    raw = _kimi_case("a5_kimi_tp8_e16_bf16_c1_p4k", layers=2)
    changed = _override_case(raw, {"model": {"num_hidden_layers": 3, "quantization": "w8a8"}})
    parsed = parse_case(changed)
    assert parsed.model.layers == 3
    assert parsed.model.quant.expert_weight_format == "int8"
    assert raw["model"]["layers"] == 2
    assert raw["model"]["quant"]["expert_weight_format"] == "bf16"
    gates = [op for op in target_operators(changed, phase="decode", tokens=1, context=128, scope="body")
             if op.recipe == "balanced_grouped_gate_up"]
    assert len(gates) == 2
    assert all(dict(op.metadata)["weight_format"] == "int8" for op in gates)

@pytest.mark.parametrize("name,layers,kda,implementation", KIMI_SOURCES)
def test_complete_source_inputs_preserve_geometry_and_replace_native_boundaries(
    name: str, layers: int, kda: int, implementation: str,
) -> None:
    base = json.loads((HIGH_CONCURRENCY / "kimi-variants" / f"{name}.json").read_text())
    source = json.loads((KIMI_SOURCE / f"{name}_source.json").read_text())
    assert source["model"] == base["model"] and source["machine"] == base["machine"]
    assert source["runtime"]["routing"] == base["runtime"]["routing"]
    assert len(parse_case(source).model.layer_plans) == layers
    batch = source["runtime"]["metadata"]["client"]["concurrency"]
    graph = target_graphs(source, phase="decode", tokens=1, context=4096, sequences=batch, scope="body")[0]
    selected = [op for op in graph if op.recipe == implementation]
    assert len(selected) == kda
    for op in selected:
        params = dict(op.metadata)["kda_recurrent_params"]
        assert params["lengths"] == (1,) * batch
        assert (params["gate_dtype"], params["beta_dtype"], params["state_dtype"]) == ( "bf16", "fp32", "fp32")
        assert not params["has_accepted_tokens"]
    if name.startswith("a5_"):
        assert sum(op.recipe == "mla_query_pad_nope" for op in graph) == 24
        assert not any(op.recipe in {"expert_dispatch", "expert_combine"} for op in graph)
    else:
        assert any(op.recipe == "expert_dispatch" for op in graph)

def test_hybrid_ple_indexer_hyper_connections_and_cache_pool_are_conserved() -> None:
    model = load_model("qwen3.8-flash-next")
    table = model.engram.tables[0]
    assert (table.layer_id, table.rows, table.width) == (1, 320_001_536, 160)
    terms = engram_table_terms(model)
    assert len(terms) == 1 and terms[0].shard == "out_tp"
    bf16_total = table.rows * table.width * 2
    assert 102.40e9 <= bf16_total <= 102.41e9
    tp8 = stage_weight_bytes(model, ParallelSpec(tp=8, ep=8))[0]
    offload = stage_weight_bytes(model, ParallelSpec(tp=8, ep=8), include_engram_tables=False)[0]
    assert pytest.approx(tp8 - offload, rel=1e-4) == bf16_total / 8
    qsa = next(plan for plan in model.layer_plans if plan.indexer)
    names = {term.name for term in layer_weight_terms(qsa, model)}
    assert {"index_qk_proj", "indexer_q_norm", "indexer_k_norm"} <= names
    assert names.isdisjoint({"indexer_weights", "indexer_pool_gate", "indexer_pool_ape"})
    assert {"hc_attn_norm", "hc_attn_down_inject", "hc_attn_up"} <= names
    breakdown = request_cache_breakdown(model, ParallelSpec(tp=8, ep=8), 4096)
    pool = breakdown["qwen4_exp_pool"]
    assert (pool["slot_count"], pool["gdn_state_groups"], pool["raw_blocks"]) == (12, 3, 1)
    assert pool["private_ple_bytes"] > 0 and breakdown["by_component"]["ple_conv_state"] > 0
    row_map = dict(cache_row_bytes_by_layer(load_model("gemma-4-26b-a4b")))
    keqv = next(plan for plan in load_model("gemma-4-26b-a4b").layer_plans if plan.k_eq_v)
    assert row_map[keqv.index] == 2 * keqv.num_kv_heads * keqv.head_dim * 2

def test_hybrid_body_executes_without_sinkhorn_or_fp32_stream_casts() -> None:
    raw = _deployment_case("qwen3.8-flash-next", "w8a8", SKUS[1])
    raw["model"]["layers"] = 4
    graphs = target_graphs(raw, phase="decode", tokens=1, context=512, scope="body")
    recipes = {op.recipe for op in graphs[0]}
    assert {"hc_lowrank_down_inject", "hc_lowrank_up", "hc_gate_mix", "hc_combine"} <= recipes
    assert recipes.isdisjoint({"hc_sinkhorn", "hc_values_fp32"})
    result = graph_costs(raw, graphs, external_inputs={"analysis_input": 0})
    assert result["duration_ms"] > 0

def test_situ_declarations_replace_swiglu_and_bound_mx_quant_operands() -> None:
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    raw["model"]["layers"] = 2
    case = parse_case(raw)
    activations = [op for op in target_operators(case, phase="prefill", tokens=8) if op.family == "situ"]
    assert {op.recipe for op in activations} == {"situ", "grouped_situ"}
    assert any(".shared." in op.op_id for op in activations)
    assert not any(op.family == "swiglu" for op in target_operators(case, phase="prefill", tokens=8))
    for op in activations:
        meta = dict(op.metadata)
        assert (meta["activation_beta"], meta["activation_linear_beta"]) == (4, 25)
        assert meta["activation_compute_dtype"] == "fp32"
        elements = op.mte3_bytes / 2
        assert op.vector_ops == pytest.approx(12 * elements)
        assert op.mte2_bytes == pytest.approx(4 * elements)
    graph = export_structure(case, phase="prefill", tokens=8, scope="body")
    assert not any(group["role"] in {"situ_quant", "grouped_gate_up_situ_quant", "grouped_gate_up_activation_quant"}
                   for group in graph["subgraphs"])

    quant = replace(case.model.quant, weight_format="fp8_e4m3",
                    expert_weight_format="fp8_e4m3", activation_format="fp8_e4m3",
                    weight_block=(32, 32), activation_group_size=32,
                    activation_scale_format="fp8_e8m0", activation_scale_pack=2, exclusions=())
    selected = export_structure(replace(case, model=replace(case.model, quant=quant)),
                                phase="prefill", tokens=8, scope="body")
    nodes = {node["node_id"]: node for node in selected["nodes"]}
    groups = [group for group in selected["subgraphs"] if group["role"] == "situ_quant"]
    assert groups
    for group in groups:
        members = [nodes[node_id]["operator"] for node_id in group["node_ids"]]
        assert {op["family"] for op in members} == {"situ", "quantize"}
        assert group["contract"][:2] == ["situ_mxquant_active", "SituMxQuant"]
    routed = [group for group in selected["subgraphs"] if group["role"] == "grouped_gate_up_situ_quant"]
    assert routed and all(group["contract"] is None for group in routed)
    assert all(group["coverage"]["kind"] == "native_source_domain_mismatch" for group in routed)

def test_situ_parameter_expression_adds_clipping_work_without_extra_gm_tensors() -> None:
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    raw["model"]["layers"] = 2
    plan = parse_case(raw).model.layer_plans[0]
    common = dict(op_id="activation", deps=("up",), rows=3.5, width=128, dtype="bf16",
                  stage=0, requests=(), phase="prefill", recipe="swiglu")
    situ = gated_activation(plan=plan, **common)
    unclipped = gated_activation(plan=replace(plan, activation_situ_linear_beta=None), **common)
    silu = gated_activation(plan=replace(plan, ffn_activation="silu"), **common)
    assert situ.vector_ops - unclipped.vector_ops == 3 * 3.5 * 128
    assert silu.vector_ops == 4 * 3.5 * 128
    assert situ.mte2_bytes == unclipped.mte2_bytes == silu.mte2_bytes == 4 * 3.5 * 128
    assert situ.mte3_bytes == unclipped.mte3_bytes == silu.mte3_bytes == 2 * 3.5 * 128

def test_declared_value_formats_cover_role_overrides_and_a_heterogeneous_router() -> None:
    model = ModelSpec.from_mapping(dict(
        name="value-formats", layers=5, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=64, vocab_size=128,
        activation_dtype="fp16", weight_dtype="bf16", kv_dtype="fp8", compute_dtype="fp32",
        metadata=[("rope_parameters", "invalid"), ("rope_theta", 10000)],
        layer_plans=[
            dict(index=0, attention="mla", ffn="moe", q_lora_rank=16, o_groups=2,
                 mla_output_gate=True, num_shared_experts=1, expert_latent_size=32,
                 indexer=True, index_kpool=2, router_dtype="fp32"),
            dict(index=1, attention="mla", ffn="moe", q_lora_rank=0, o_groups=0,
                 mla_output_gate=False, num_shared_experts=0, expert_latent_size=0,
                 indexer=True, index_kpool=1, router_output_dtype="int8", scope="mtp"),
            dict(index=2, attention="gqa", ffn="dense"),
            dict(index=3, attention="gqa", ffn="moe"),
            dict(index=4, attention="gqa", ffn="none")]))
    formats = model_value_formats(model)
    roles = formats["matrix_roles"]
    assert (formats["activation_stream"], formats["compute"], formats["kv_cache"]) == (
        "fp16", "fp32", "fp8_e4m3")
    assert {"mla.q_a", "mla.q_b", "mla.o_a", "mla.o_b", "mla.g", "shared_expert",
            "latent_moe", "indexer.pool_gate", "indexer.pool_projection", "mtp"} <= set(roles)
    assert roles["mla.q_a"]["compute"] == roles["dense_mlp"]["compute"] == "fp32"
    assert roles["router"]["heterogeneous"] is True
    assert [roles["router"]["by_layer"][layer]["output"] for layer in (0, 1)] == ["fp32", "int8"]
    assert formats["rotary"]["rope_parameters"] == {"rope_theta": 10000}
    assert formats["rotary"]["rotary_dim_by_attention"] == {"gqa": (16,), "mla": (0,)}
    overridden = _role_value_formats(quant_preset("bf16"), "dense_mlp", native="bf16",
                                     compute_override="fp32", output_override="int8")
    assert (overridden["compute"], overridden["output"]) == ("fp32", "int8")

def test_local_hf_config_path_without_max_position_composes_an_explicit_zero(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 1,
                                "hidden_size": 32, "num_attention_heads": 4,
                                "num_key_value_heads": 2, "intermediate_size": 64,
                                "vocab_size": 128}))
    raw = compose_case(path, hardware={"sku": "a3"}, tp=1, dp=1, prompt_tokens=8, output_tokens=1)
    assert parse_case(raw).model.layers == 1
    assert raw["runtime"]["max_model_len"] == 32768
    named = compose_case(str(path), hardware={"sku": "a3"}, tp=1, dp=1, prompt_tokens=8,
                         output_tokens=1)
    assert parse_case(named).model.layers == 1
