"""Phase-domain decisions preserve source boundaries and complete target work."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.implementation_selection import graph_selections
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case
from test_family_selection import declared_case
from test_sparse_mla_metadata import raw_case as sparse_case


def policy(binding, domain=None, *, otherwise="error"):
    return {"cases": [{"domain": domain or {"phases": ["decode"]}, "binding": binding}],
            "otherwise": otherwise}


def test_serving_and_both_probe_scopes_share_padding_aware_selection():
    raw = declared_case()
    raw["workload"]["requests"][0]["output_tokens"] = 3
    raw["runtime"].update(graph_mode="auto", graph_buckets=[4])
    raw["runtime"]["kernels"] = {"lowerings": {"rmsnorm": policy(
        "vllm_cuda_rms_norm", {"phases": ["decode"], "valid_tokens": {"max": 1},
                                "padded_tokens": {"min": 4, "max": 4}},
        otherwise="analytical")}}
    original = deepcopy(raw)
    result = predict(raw, event_limit=0, trace_limit=0)
    decisions = result["execution"]["step_summary"]["implementation_selections"]
    assert {(d["case"], d["steps"]) for d in decisions} == {("otherwise", 1), (0, 2)}
    assert result["requests"][0]["generated_tokens"] == 3
    assert not result["execution"]["live_tasks"]
    for scope in ("body", "model"):
        graph = target_graphs(raw, phase="decode", tokens=1, context=4, scope=scope)[0]
        assert any(op.recipe.startswith("cuda_norm.") for op in graph)
        choice, = graph_selections(graph)
        assert choice["case"] == 0
        assert choice["geometry"]["valid_tokens"] == 1
        assert choice["geometry"]["padded_tokens"] == 4
        prefill = target_graphs(raw, phase="prefill", tokens=4, scope=scope)[0]
        assert not any(op.recipe.startswith("cuda_norm.") for op in prefill)
        assert graph_selections(prefill)[0]["coverage"] == "analytical_semantic_work"
    assert raw == original


def test_mixed_batch_requires_its_whole_phase_domain_and_unmatched_is_visible():
    raw = case()
    raw["runtime"]["kernels"] = {"lowerings": {"rmsnorm": policy("vllm_cuda_rms_norm")}}
    parsed = parse_case(raw)
    views = (RequestView("decode", 1, 8, True, "decode"),
             RequestView("prefill", 4, 0, True, "prefill"))
    with pytest.raises(ModelingError, match="implementation_domain_unmatched:rmsnorm"):
        Frontend()(IterationPlan(views), parsed.model, parsed.runtime, parsed.machine)
    raw["runtime"]["kernels"]["lowerings"]["rmsnorm"]["cases"].append({
        "domain": {"phases": ["decode", "prefill"]}, "binding": "analytical"})
    parsed = parse_case(raw)
    work = Frontend()(IterationPlan(views), parsed.model, parsed.runtime, parsed.machine)
    choice, = work.implementation_selections
    assert choice["case"] == 1 and choice["geometry"]["valid_tokens"] == 5
    assert not any(op.recipe.startswith("cuda_norm.") for op in work.ops)
    assert set(work.ops[-1].requests) == {"decode", "prefill"}


def test_overlapping_geometry_does_not_silently_give_first_case_priority():
    raw = case()
    binding = policy("vllm_cuda_rms_norm")
    binding["cases"].append({"domain": {"valid_tokens": {"min": 1}}, "binding": "analytical"})
    raw["runtime"]["kernels"] = {"lowerings": {"rmsnorm": binding}}
    with pytest.raises(ModelingError, match="implementation_domains_overlap:rmsnorm"):
        target_graphs(raw, phase="decode", tokens=1)


def test_sparse_decode_metadata_is_explicitly_excluded_from_prefill():
    raw = sparse_case()
    bindings = raw["runtime"]["kernels"]["lowerings"]
    bindings["sparse_mla_metadata"] = policy(bindings["sparse_mla_metadata"], otherwise="analytical")
    for scope in ("body", "model"):
        prefill = target_graphs(raw, phase="prefill", tokens=8, scope=scope)[0]
        decode = target_graphs(raw, phase="decode", tokens=4, context=128, scope=scope)[0]
        assert not any(op.family == "attention_metadata" for op in prefill)
        metadata, = [op for op in decode if op.family == "attention_metadata"]
        assert dict(metadata.metadata)["metadata_call"]["rows"] == 4
        assert graph_selections(prefill)[0]["implementation"] == "analytical"
    # A selected native domain error remains a failure despite an explicit
    # analytical alternative outside that domain.
    raw["model"]["layer_overrides"]["0"]["qk_rope_head_dim"] = 64
    with pytest.raises(ModelingError, match="sparse_mla_metadata_requires_nope_bf16_paged_decode"):
        target_graphs(raw, phase="decode", tokens=4, context=128)


def test_prefill_kda_chooses_its_chunk_source_without_decode_source_in_mixed_batch():
    from test_kda_decode_family import caller
    from mindie_sim.prediction.family_kda_recurrent_arch22 import IMPLEMENTATION as DECODE
    from mindie_sim.prediction.family_kda_state import IMPLEMENTATION as PREFILL

    parsed, _ = caller((1,))
    plan = replace(parsed.model.layer_plans[0], linear_key_heads=2)
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=(plan,)))
    binding = policy(DECODE, otherwise=PREFILL)
    parsed = replace(parsed, runtime=replace(parsed.runtime,
        kernels={"lowerings": {"kda": binding}}))
    views = (RequestView("d", 1, 32, True, "decode"),
             RequestView("p", 16, 0, True, "prefill"))
    work = Frontend()(IterationPlan(views), parsed.model, parsed.runtime, parsed.machine)
    choice, = work.implementation_selections
    assert choice["implementation"] == PREFILL
    assert not any(op.recipe == DECODE for op in work.ops)
    assert any(op.source_boundary for op in work.ops)
    decode = target_graphs(parsed, phase="decode", tokens=1, scope="body")[0]
    assert any(op.recipe == DECODE for op in decode)
    native = tuple(op for op in decode if op.recipe == DECODE)
    result = graph_costs(parsed, native, isolated=False,
                         external_inputs={dep: 0 for op in native for dep in op.deps})
    assert result["duration_ms"] > 0
    assert result["execution"]["live_tasks"] == 0


def test_hc_prefill_geometry_selects_source_only_inside_its_declared_row_domain():
    from test_hc_arch35_source import raw_case

    parsed = parse_case(raw_case())
    parsed = replace(parsed, model=replace(parsed.model, layers=1, body_layers=1,
                                           layer_plans=parsed.model.layer_plans[:1]))
    binding = parsed.runtime.kernels["lowerings"]["native_fusion"]
    kernels = {**parsed.runtime.kernels, "lowerings": {"native_fusion": policy(
        binding, {"padded_tokens": {"min": 1, "max": 256}}, otherwise="analytical")}}
    parsed = replace(parsed, runtime=replace(parsed.runtime, kernels=kernels,
        graph_mode="none", pp_layer_partition=(1,),
        parallel=replace(parsed.runtime.parallel, dp=1, ep=1)))
    small = target_graphs(parsed, phase="prefill", tokens=256, scope="body")[0]
    large = target_graphs(parsed, phase="prefill", tokens=257, scope="body")[0]
    assert any(dict(op.metadata).get("fusion_role") == "hc_pre" for op in small)
    assert not any(dict(op.metadata).get("fusion_role") == "hc_pre" for op in large)
    assert graph_selections(small)[0]["implementation"] == "declared_native_fusion"
    assert graph_selections(large)[0]["implementation"] == "analytical"
    projection = next(op for op in large if op.recipe == "hc_projection")
    assert projection.local_shape[0] == 257 and projection.flops > 0


def test_optional_prepared_weights_stay_resident_when_mixed_calls_select_analytical():
    from mindie_sim.catalog.weights import stage_weight_bytes
    from mindie_sim.prediction.gdn_decode_programs import CONV
    from test_causal_conv_update_family import case as kimi_case

    original = parse_case(kimi_case(1))
    model = replace(original.model, layers=4, body_layers=4,
                    layer_plans=original.model.layer_plans[:4])
    original = replace(original, model=model,
                       runtime=replace(original.runtime, pp_layer_partition=(4,)))
    resident = stage_weight_bytes(original.model, original.runtime.parallel)
    kernels = deepcopy(original.runtime.kernels)
    for selector in ("mla_head_padding", "causal_conv_update", "kda", "paged_mla_attention"):
        kernels["lowerings"][selector] = policy(kernels["lowerings"][selector], {
            "phases": ["decode"], "max_query_tokens": {"max": 1}}, otherwise="analytical")
    parsed = parse_case(replace(original, runtime=replace(original.runtime, kernels=kernels)))
    assert stage_weight_bytes(parsed.model, parsed.runtime.parallel) == resident
    views = (RequestView("d", 1, 4104, False, "decode", draft_advanced_tokens=1),
             RequestView("p", 8, 0, False, "prefill"))
    mixed = Frontend()(IterationPlan(views), parsed.model, parsed.runtime, parsed.machine)
    assert all(row["implementation"] == "analytical" for row in mixed.implementation_selections)
    assert any(op.recipe == "short_convolution" for op in mixed.ops)
    assert not any(op.recipe in {CONV, "mla_query_pad_nope", "mla_query_pad_rope"} for op in mixed.ops)
    decode = Frontend()(IterationPlan(views[:1]), parsed.model, parsed.runtime, parsed.machine)
    assert any(op.recipe == CONV for op in decode.ops)
    assert any(op.recipe == "mla_query_pad_nope" for op in decode.ops)
    assert stage_weight_bytes(parsed.model, parsed.runtime.parallel) == resident
