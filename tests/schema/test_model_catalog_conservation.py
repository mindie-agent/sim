"""Independent 39-model catalog conservation, fleet geometry and scaling laws.

[Role]
  Rebuild every catalog model's parameter shapes, KV-cache bytes, layer
  declarations and aliases from the raw JSON configs through ``tests/_oracles``;
  then check fleet geometry, balanced routing, ``customize_model_spec`` scaling
  and HF quantization role exclusions.

[Invariants]
  * Every manifest entry is checked end to end: a missing reference, shape,
    alias or config hash is a failure, never a skip.
  * Parameter totals, per-name shapes and payload/allocated cache bytes are
    compared separately across TP/PP/CP and block/window tokens.
  * Geometry is data-driven: no model-name latency constant, no measurement and
    no per-model branch in the engine or ``PhysicalRecipe``.
  * Balanced routing conserves expert rows for every integer batch/query; dense
    siblings never manufacture experts to satisfy a routing sweep.

[Timeouts & Locks]
  Bounded in-memory catalog loads and small FFN graphs; no device, network or locks.

[Side-Effect Policy]
  Read-only catalog/config access plus one ``tmp_path`` config copy; never
  mutates repository files, environment variables or host state.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields as dataclass_fields, replace
from functools import lru_cache
import json
import math
from pathlib import Path

import pytest

from _oracles.cache_meta_oracle import cache_totals
from _oracles.catalog_meta_oracle import build_reference, compare_entries, load_manifest, manifest_models
from _support.axes import REPRESENTATIVE_SKUS
from _support.hf_configs import case as serving_case
from mindie_sim import compose_case, graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.hf_config import resolve_quant, resolve_role_exclusions
from mindie_sim.coverage import model_coverage
from mindie_sim.model_catalog import ARCHITECTURE_PRESET_ALIASES, ARCHITECTURE_PRESET_CATALOG, CATALOG
from mindie_sim.model_catalog import MODEL_ALIASES, body_parameter_count, body_weight_bytes
from mindie_sim.model_catalog import build_model_spec, customize_model_spec, load_model, model_value_formats
from mindie_sim.prediction.feedforward import alltoall, dense
from mindie_sim.prediction.frontend.features import FeatureConfig
from mindie_sim.prediction.operators.vector.gated_activation import gated_activation
from mindie_sim.prediction.sharding.routing import select_realization
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "fixtures" / "catalog_reference_manifest.json"
BLOCK, CAPTURE_TOKENS = 128, 4096
PARALLEL_CASES = ((1, 1, 1), (2, 1, 1), (8, 1, 1), (1, 2, 1), (1, 1, 2))
SERVER_SKUS = ("Ascend910_9362", "Ascend950DT_9582")
A5_SKU = REPRESENTATIVE_SKUS["a5"]
MODELS = (
    "deepseek-v4-10t", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813", "deepseek-v4.1-flash",
    "deepseek-v2-lite", "deepseek-v3.2-prune-3l", "deepseek-v3", "deepseek-r1",
    "deepseek-v3.2", "glm-5.3", "glm-5.3-flash", "glm-5.1", "glm-5.2", "kimi-k3",
    "kimi-k3.1", "kimi-k3-dspark", "kimi-k3-dspark-block5", "kimi-k2", "kimi-k2-thinking",
    "kimi-k2.5", "kimi-k2.6", "qwen3-30b-a3b", "qwen3-32b", "qwen3-235b-a22b",
    "qwen3-next-80b-a3b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b",
    "qwen3.5-397b-a17b", "qwen3.6-27b", "qwen3.8-2.4t-a95b", "qwen3.8-27b",
    "minimax-m3", "minimax-m3.1-flash", "gemma-4-26b-a4b", "gemma-4-31b",
    "qwen3.6-35b-a3b", "qwen3.8-flash-next", "step-5-preview")
#: Representative core entries and their documented checkpoint/prompt aliases.
ALIASED_ENTRIES = {
    "deepseek-v3": ("DeepSeek-V3", "DeepSeek-V3-0324", "dsv3"),
    "kimi-k2.5": ("Kimi-K2.5", "kimi_k2.5", "kimi_k25", "kimi-k25"),
    "qwen3.5-397b-a17b": ("Qwen3.5-397B-A17B", "qwen3_5_397b_a17b", "qwen3.5-397b")}
#: ``(layers, hidden, experts, top_k, dense, gdn, gqa)`` declared by each config.
CORE_GEOMETRY = {
    "deepseek-v2-lite": (27, 2048, 64, 6, 1, 0, 0), "deepseek-v3": (61, 7168, 256, 8, 3, 0, 0),
    "glm-5.1": (78, 6144, 256, 8, 3, 0, 0), "kimi-k2": (61, 7168, 384, 8, 1, 0, 0),
    "qwen3-30b-a3b": (48, 2048, 128, 8, 0, 0, 48), "qwen3-32b": (64, 5120, 0, 0, 64, 0, 64),
    "qwen3-235b-a22b": (94, 4096, 128, 8, 0, 0, 94),
    "qwen3-next-80b-a3b": (48, 2048, 512, 10, 0, 36, 12),
    "qwen3.5-27b": (64, 5120, 0, 0, 64, 48, 16),
    "qwen3.5-397b-a17b": (60, 4096, 512, 10, 0, 45, 15)}
BALANCED_MODELS = ("qwen3.5-35b-a3b", "qwen3.6-27b", "qwen3.8-27b", "glm-5.3",
                   "deepseek-v4-flash-0731", "deepseek-v4-pro-0813",
                   "deepseek-v4.1-flash", "kimi-k3", "minimax-m3")

@lru_cache(maxsize=1)
def _manifest() -> dict:
    return load_manifest(MANIFEST)

@lru_cache(maxsize=None)
def _entry(key: str) -> dict:
    return manifest_models(_manifest())[key]

@lru_cache(maxsize=None)
def _reference(key: str):
    return build_reference(key, ROOT, _entry(key))

@lru_cache(maxsize=None)
def _model(key: str):
    return load_model(key)

def _actual_entries(model) -> list[tuple]:
    from mindie_sim.catalog.weights import _body_terms
    return [(term.name, int(term.rows), int(term.cols), int(term.count),
             "source" if term.source_owned else "runtime") for term in _body_terms(model)]

def _cache_bytes(model, parallel, tokens: int, padded: bool) -> float:
    from mindie_sim.catalog.cache import layer_cache_decomposition
    return float(sum(float(sum(float(part["total"]) for part in layer_cache_decomposition(
        model, plan, parallel, tokens, capacity_padded=padded).values()))
        for plan in model.layer_plans))

def _declared_auxiliary(model) -> dict:
    aux = getattr(model, "auxiliary", None)
    for kind in ("mtp", "dspark"):
        spec = getattr(aux, kind, None) if aux is not None else None
        if spec is not None:
            return {"kind": str(spec.kind), "layers": int(spec.layers),
                    "standalone": bool(aux.standalone_draft)}
    return {}

def _representatives(model):
    names = tuple(f.name for f in dataclass_fields(model.layer_plans[0])
                  if any(t in f.name for t in ("expert", "router", "shared", "ffn", "situ"))
                  or f.name in {"moe_intermediate_size", "intermediate_size", "scope"})
    groups: dict[tuple, list] = {}
    for layer in model.layer_plans:
        groups.setdefault(tuple(getattr(layer, name) for name in names), []).append(layer)
    return tuple((members[0], tuple(members)) for members in groups.values())

@lru_cache(maxsize=None)
def _balanced_case(name: str):
    routed = any(plan.ffn == "moe" for plan in _model(name).layer_plans)
    return parse_case({
        "model": {"preset": name, "quantization": "w8a8"},
        "machine": {"backend": "ascend", "sku": SERVER_SKUS[0], "num_devices": 8},
        "runtime": {"tp": 8, "ep": 8 if routed else 1, "routing": {"policy": "balanced"},
                    "kernels": {"programs": {"matmul": {"kind": "steady_matmul"}}}},
        "workload": {"requests": [{"id": "component", "prompt_tokens": 1, "output_tokens": 1}]}})

def _ffn_ops(case, layer, tokens: int) -> tuple:
    routing = (select_realization(case.runtime, layer, phase="decode", tokens=tokens,
                                  valid_tokens=tokens) if layer.ffn == "moe" else None)
    ops, _ = (alltoall if layer.ffn == "moe" else dense)(
        prefix="component", plan=layer, deps="input", stage=0, requests=("component",),
        tokens=tokens, model=case.model, parallel=case.runtime.parallel, mode="decode",
        runtime=case.runtime, routing_realization=routing)
    return lower_implementations(ops, case.runtime, case.model.activation_dtype,
                                 local_tokens=tokens, resource_fusion=False)

def _compose(key: str, layers: int) -> dict:
    return compose_case(key, hardware=A5_SKU, layers=layers)

def test_manifest_catalog_and_alias_inventory_is_conserved() -> None:
    models = manifest_models(_manifest())
    assert len(models) == len(MODELS) == len(CATALOG) == 39 and set(models) == set(MODELS)
    for key, entry in models.items():
        assert entry["parameter_scope"] == "text_backbone"
        assert {"config", "config_sha256", "builder", "cache_layout", "auxiliary_modes",
                "sharing_relations", "quantization"} <= set(entry), key
        path = ROOT / entry["config"]
        assert path.is_file(), (key, entry["config"])
        assert _reference(key).config_sha256 == entry["config_sha256"]
    for alias, key in {**MODEL_ALIASES, **ARCHITECTURE_PRESET_ALIASES}.items():
        assert key in CATALOG or key in ARCHITECTURE_PRESET_CATALOG, alias
        assert load_model(alias) == load_model(key), alias
    for key, aliases in ALIASED_ENTRIES.items():
        spec = _model(key)
        assert body_parameter_count(spec) > 0 and body_weight_bytes(spec) > 0
        assert load_model(key) == spec
        for alias in aliases:
            assert MODEL_ALIASES[alias] == key and load_model(alias) == spec

@pytest.mark.parametrize("key", MODELS)
def test_manifest_parameter_shapes_and_totals_conserve(key: str) -> None:
    comparison = compare_entries(_reference(key), _actual_entries(_model(key)))
    assert comparison["complete"], {
        "parameter_delta": comparison["parameter_delta"],
        "runtime_delta": comparison["runtime_delta"],
        "missing_in_actual": comparison["missing_in_actual"][:5],
        "missing_in_reference": comparison["missing_in_reference"][:5],
        "shape_mismatches": comparison["shape_mismatches"][:5]}
    assert comparison["parameter_delta"] == comparison["runtime_delta"] == 0
    assert comparison["actual_parameters"] == comparison["reference_parameters"] > 0

@pytest.mark.parametrize("overrides,missing,reason", [
    ({}, "config_sha256", "reference_missing_sha256"),
    ({"config_sha256": "0" * 64}, None, "reference_sha256_mismatch"),
    ({}, "cache_layout", "reference_declaration_missing"),
    ({"parameter_scope": "full_checkpoint"}, None, "reference_scope_unsupported"),
    ({"config": "mindie_sim/nope.json"}, None, "reference_config_missing"),
])
def test_reference_gate_rejects_missing_or_incomplete_declarations(overrides, missing, reason):
    key = "deepseek-v3"
    entry = _entry(key) | overrides
    if missing is not None:
        del entry[missing]
    with pytest.raises(ValueError, match=reason):
        build_reference(key, ROOT, entry)

def test_named_shape_comparison_detects_missing_and_extra_shapes():
    reference = _reference("qwen3-32b")
    actual = _actual_entries(_model("qwen3-32b"))
    assert compare_entries(reference, actual)["complete"]
    broken = compare_entries(reference, actual[1:])
    assert not broken["complete"]
    assert broken["missing_in_actual"] and broken["parameter_delta"] != 0
    extra = compare_entries(reference, actual + [("bogus", 1, 1, 1, "source")])
    assert not extra["complete"]
    assert extra["missing_in_reference"] == [["bogus", 1, 1, 1, "source"]]

@pytest.mark.parametrize("key", MODELS)
def test_kv_payload_and_allocated_bytes_conserve(key: str) -> None:
    from mindie_sim.catalog.cache import request_cache_breakdown
    from mindie_sim.schema.parallel import ParallelSpec
    model, reference = _model(key), _reference(key)
    window = max([layer.window for layer in reference.layers] + [BLOCK])
    tokens = sorted({0, 1, BLOCK - 1, BLOCK, max(0, window - 1), window, CAPTURE_TOKENS})
    for tp, pp, cp in PARALLEL_CASES:
        parallel = ParallelSpec(tp=tp, dp=1, pp=pp, cp=cp)
        for seq in tokens:
            actual = {flag: _cache_bytes(model, parallel, seq, padded=flag) for flag in (False, True)}
            assert float(request_cache_breakdown(model, parallel, seq)["total_bytes"]) == (
                pytest.approx(actual[True], abs=1e-6))
            for padded, measured in actual.items():
                assert measured - cache_totals(
                    reference.geometry, seq, padded=padded, tp=tp, cp=cp
                ) == pytest.approx(0.0, abs=1e-6), (key, tp, pp, cp, seq, padded)

@pytest.mark.parametrize("key", MODELS)
def test_auxiliary_quantization_and_layer_declarations_conserve(key: str) -> None:
    model, entry, reference = _model(key), _entry(key), _reference(key)
    assert model.accounting_scope == entry["parameter_scope"] == "text_backbone"
    declared, actual = entry["auxiliary_modes"], _declared_auxiliary(model)
    if actual:
        assert (declared["kind"], int(declared["layers"]), bool(declared["standalone"])) == (
            actual["kind"], actual["layers"], actual["standalone"])
    else:
        assert declared == {}
    quant = entry["quantization"]
    assert (quant["kv_dtype"], quant["index_dtype"], quant["index_scale_dtype"]) == (
        model.quant.kv_dtype, model.quant.index_dtype, model.quant.index_scale_dtype or "none")
    for attribute in ("attention", "ffn", "window", "compress_ratio", "owns_index_k"):
        assert [getattr(layer, attribute) for layer in reference.layers] == [
            getattr(plan, attribute) for plan in model.layer_plans], (key, attribute)
    sharing = entry["sharing_relations"]
    assert sharing["kv_owner_layers"] == [
        plan.index for plan in model.layer_plans if plan.kv_owner_layer_id == plan.index]
    assert sharing["k_eq_v_layers"] == [
        plan.index for plan in model.layer_plans if plan.k_eq_v]

@pytest.mark.parametrize("key", MODELS)
def test_catalog_composition_keeps_declared_attention_and_quantization(key: str) -> None:
    case = _compose(key, 2)
    assert case["model"]["layer_plans"], key
    assert [plan["attention"] for plan in case["model"]["layer_plans"]
            if plan["attention"] in {"unmodeled", "projection_only"}] == []
    gaps = tuple(dict(case["model"].get("metadata") or ()).get("unmodeled_features") or ())
    assert not any("compressed-tensors" in gap and "proxy" in gap for gap in gaps), gaps
    quant = case["model"].get("quant") or {}
    assert quant.get("weight_format") != "int4" or quant.get("activation_format") not in {"int8", ""}
    kinds = {target_graphs(case, scope="body", phase="decode", tokens=1, context=128,
                           sequences=1).effective_kernels["programs"]["matmul"]["kind"]}
    assert kinds == {"steady_matmul"}, f"{key} still selected {kinds}"

def test_partial_catalog_metadata_survives_normalization_and_real_costs() -> None:
    raw = serving_case()
    gaps = ["int8_sparse_kv_source_unmodeled", "native_dispatch_protocol_unbound"]
    raw["model"] = {
        "catalog_key": "glm-5.3-flash", "name": "declared-capture-model",
        "metadata": {"model_coverage": "partial", "unmodeled_features": gaps,
                     "degraded_submodules": ["sparse_attention_analytic"],
                     "source": {"identity": "capture-launch", "dimensions": [24, 4096]}}}
    original = deepcopy(raw)
    parsed = parse_case(raw)
    assert parsed.model.layers == 45
    metadata = dict(parsed.model.metadata)
    assert metadata["source"] == original["model"]["metadata"]["source"]
    assert metadata["unmodeled_features"] == gaps
    norm = next(op for op in target_operators(parsed, phase="decode", tokens=1, context=2048)
                if op.family == "rmsnorm")
    costs = operator_costs(parsed, (norm,))
    assert costs["model_coverage"] == "partial"
    assert costs["unmodeled_features"] == gaps
    assert costs["degraded_submodules"] == ["sparse_attention_analytic"]
    assert costs["operator_sum_ms"] > 0
    assert raw == original

@pytest.mark.parametrize("name", BALANCED_MODELS)
def test_balanced_routing_conserves_expert_rows_and_expert_banks(name: str) -> None:
    case = _balanced_case(name)
    groups = _representatives(case.model)
    assert sum(len(members) for _, members in groups) == case.model.layers
    routed = [(layer, members) for layer, members in groups if layer.ffn == "moe"]
    for layer, _members in routed:
        latent = layer.expert_latent_size or case.model.hidden_size
        width = math.ceil(layer.moe_intermediate_size / case.runtime.parallel.etp)
        for batch in range(1, 129):
            for query in (1, 4, 5, 8):
                tokens = batch * query
                route = select_realization(case.runtime, layer, phase="decode",
                                           tokens=tokens, valid_tokens=tokens)
                counts = Counter(route.expert_rows)
                ranks = tuple(sum(route.expert_rows[rank::case.runtime.parallel.ep])
                              for rank in range(case.runtime.parallel.ep))
                assert sum(route.expert_rows) == tokens * layer.top_k
                assert len(route.expert_rows) == layer.num_experts
                assert max(counts) - min(counts) <= 1
                if tokens * layer.top_k < layer.num_experts:
                    assert counts[0] == layer.num_experts - tokens * layer.top_k
                assert route.rows == ranks[0] == max(ranks)
                assert route.active == sum(count > 0 for count in route.local_expert_rows)
        for tokens in (1, 5, 128 * 5):
            route = select_realization(case.runtime, layer, phase="decode",
                                       tokens=tokens, valid_tokens=tokens)
            matrices = [op for op in _ffn_ops(case, layer, tokens)
                        if op.family == "grouped_gemm"]
            assert [op.local_shape for op in matrices] == [
                (route.rows, 2 * width, latent), (route.rows, latent, width)]
            for op in matrices:
                rows, n, k = op.local_shape
                assert op.flops == 2 * rows * n * k
                assert dict(op.metadata)["matrix_group_rows"] == route.local_expert_rows
                assert dict(op.metadata)["packed_weight_bytes"] == pytest.approx(
                    route.active * case.model.quant.matrix_bytes(n, k, "routed_expert"))
        report = graph_costs(case, {0: tuple(_ffn_ops(case, layer, 5))},
                             external_inputs={"input": 0})
        assert report["duration_ms"] > 0
        assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    if not routed:
        ops = _ffn_ops(case, groups[0][0], 1)
        assert any(op.flops > 0 for op in ops)
        assert not any(op.family == "grouped_gemm" for op in ops)

def test_fleet_layer_plan_geometries_match_declared_configs() -> None:
    expected = {"minimax-m3": (60, 128, 4, 3), "minimax-m3.1-flash": (60, 128, 4, 3),
                "gemma-4-26b-a4b": (30, 128, 8, 0), "gemma-4-31b": (60, 0, 0, 60),
                "qwen3.6-35b-a3b": (40, 256, 8, 0), "qwen3.8-flash-next": (48, 512, 10, 0),
                "step-5-preview": (92, 352, 8, 4)}
    for key, (layers, experts, top_k, dense) in expected.items():
        model = load_model(key)
        assert (model.layers, model.num_experts, model.top_k) == (layers, experts, top_k), key
        assert sum(plan.ffn == "dense" for plan in model.layer_plans) == dense, key
        assert model.layers == model.body_layers == len(model.layer_plans)
    gemma26 = load_model("gemma-4-26b-a4b")
    assert [p.index for p in gemma26.layer_plans if not p.window] == [5, 11, 17, 23, 29]
    assert all((p.ffn, p.num_experts, p.top_k, p.moe_intermediate_size,
                p.num_shared_experts, p.shared_intermediate_size, p.ffn_activation)
               == ("moe", 128, 8, 704, 1, 2112, "gelu") for p in gemma26.layer_plans)
    for key in ("minimax-m3", "minimax-m3.1-flash"):
        plans = load_model(key).layer_plans
        assert all((p.num_heads, p.num_kv_heads, p.head_dim) == (64, 4, 128) for p in plans)
        assert [p.ffn for p in plans[:3]] == ["dense"] * 3
        assert all((p.num_experts, p.top_k, p.moe_intermediate_size,
                    p.num_shared_experts, p.shared_intermediate_size, p.router_mode)
                   == (128, 4, 3072, 1, 3072, "sigmoid") for p in plans[3:])
        active = [p.index for p in plans if p.indexer]
        assert active == (list(range(3, 60)) if key == "minimax-m3" else list(range(60)))
        assert all((p.indexer_type, p.index_heads, p.index_dim, p.index_topk,
                    p.candidate_block_size, p.candidate_topk_blocks)
                   == ("full", 4, 128, 2176, 128, 17) for p in plans if p.indexer)
    qwen38 = load_model("qwen3.8-flash-next")
    assert (qwen38.hidden_size, qwen38.num_experts, qwen38.top_k) == (2560, 512, 10)
    assert len([p for p in qwen38.layer_plans if p.attention == "gdn"]) == 36
    assert all(p.hyper_connections and p.hc_mult == 4 and p.hc_lowrank == 320
               for p in qwen38.layer_plans)
    indexers = [p for p in qwen38.layer_plans if p.indexer]
    assert [p.index for p in indexers] == list(range(3, 48, 4))
    assert all((p.indexer_type, p.index_heads, p.index_dim, p.index_topk,
                p.index_kpool, p.index_kpool_compress)
               == ("full", 4, 128, 2048, 4, True) for p in indexers)
    table = qwen38.engram.tables[0]
    assert (table.layer_id, table.rows, table.width) == (1, 320_001_536, 160)
    assert qwen38.layer_plans[1].engram and not qwen38.layer_plans[2].engram
    step = load_model("step-5-preview")
    assert len([p for p in step.layer_plans if p.window == 512]) == 69
    assert [p.index for p in step.layer_plans if p.ffn == "dense"] == [0, 1, 2, 91]
    assert all((p.num_experts, p.top_k, p.moe_intermediate_size, p.num_shared_experts,
                p.shared_intermediate_size) == (352, 8, 1536, 1, 1536)
               for p in step.layer_plans if p.ffn == "moe")
    assert all((p.num_heads, p.num_kv_heads, p.head_dim, p.q_output_gate) == (64, 4, 192, True)
               for p in step.layer_plans)

def test_core_catalog_geometry_and_routed_declarations_are_conserved() -> None:
    for key, (layers, hidden, experts, top_k, dense, gdn, gqa) in CORE_GEOMETRY.items():
        model = load_model(key)
        assert (model.layers, model.hidden_size, model.num_experts, model.top_k) == (
            layers, hidden, experts, top_k), key
        plans = model.layer_plans
        assert sum(p.ffn == "dense" for p in plans) == dense, key
        assert sum(p.attention == "gdn" for p in plans) == gdn, key
        assert sum(p.attention == "gqa" for p in plans) == gqa, key
        assert len(plans) == layers and all(p.index == index for index, p in enumerate(plans))
    prune = load_model("deepseek-v3.2-prune-3l")
    assert [p.ffn for p in prune.layer_plans] == ["dense", "moe", "moe"]
    assert (prune.layer_plans[1].router_groups, prune.layer_plans[1].router_group_topk) == (8, 4)
    assert all(p.indexer and p.indexer_type == "full" for p in load_model("glm-5.1").layer_plans)
    assert any(p.indexer_type == "shared" for p in load_model("glm-5.2").layer_plans)
    assert "vision_encoder_and_projector:language_backbone_only" in model_coverage(
        load_model("kimi-k2.5"))["unmodeled_features"]
    assert all(p.qk_norm and p.ffn == "moe" for p in load_model("qwen3-30b-a3b").layer_plans)
    assert load_model("qwen3-32b").ffn == "dense"
    assert load_model("qwen3.5-27b").intermediate_size == 17408
    gemma = {"model_type": "gemma4", "num_hidden_layers": 6, "hidden_size": 2816,
             "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
             "vocab_size": 262144, "moe_intermediate_size": 704,
             "shared_expert_intermediate_size": 2112, "n_routed_experts": 128,
             "num_experts_per_tok": 8, "n_shared_experts": 1,
             "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}
    assert all(p.ffn == "moe" and p.num_experts == 128
               for p in build_model_spec("gemma4-legacy", gemma).layer_plans)
    step = {"model_type": "step5", "num_hidden_layers": 4, "hidden_size": 4096,
            "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 151936, "moe_intermediate_size": 1280,
            "shared_expert_intermediate_size": 1280, "n_routed_experts": 256,
            "num_experts_per_tok": 8, "n_shared_experts": 1,
            "layer_types": ["full_attention"] * 4}
    assert all(p.ffn == "moe" and p.top_k == 8
               for p in build_model_spec("step5-legacy", step).layer_plans)

def test_declared_rope_context_and_native_dtype_reach_frontend_metadata(tmp_path) -> None:
    deepseek = load_model("deepseek-v4.1-flash")
    rope = dict(deepseek.metadata)["rope_parameters"]
    assert (rope["rope_theta"], rope["rope_type"], rope["factor"]) == (10000, "yarn", 16)
    features = FeatureConfig.from_model(deepseek)
    assert dict(features.metadata)["rope_theta"] == 10000
    assert dict(features.metadata)["rope_parameters"]["factor"] == 16
    values = model_value_formats(load_model("qwen3.6-27b"))
    assert values["rotary"]["rope_parameters"]["partial_rotary_factor"] == 0.25
    assert values["rotary"]["rope_parameters"]["mrope_section"] == [11, 11, 10]
    assert values["rotary"]["rotary_dim_by_attention"]["gqa"] == (64,)
    assert values["rotary"]["max_position_embeddings"] == 262144
    glm = load_model("glm-5.3")
    assert (glm.mla.index_topk, glm.mla.index_head_dim, glm.mla.index_n_heads) == (2048, 128, 32)
    raw = json.loads((ROOT / "mindie_sim/model_configs/qwen3.5-27b.json").read_text())
    raw["torch_dtype"] = "float16"
    native = build_model_spec("external-fp16", raw)
    assert (native.activation_dtype, native.weight_dtype, native.compute_dtype) == ("fp16",) * 3
    assert not any("native_activation_dtype" in gap
                   for gap in dict(native.metadata)["unmodeled_features"])
    local = FeatureConfig.from_model(native)
    assert (local.activation_dtype, local.weight_dtype, local.kv_dtype) == ("fp16",) * 3
    (tmp_path / "qwen3.5-27b.json").write_text(json.dumps(raw), encoding="utf-8")
    rooted = load_model("qwen3.5-27b", config_root=tmp_path)
    assert (rooted.activation_dtype, rooted.weight_dtype, rooted.kv_dtype) == ("fp16",) * 3

def test_gated_activation_spellings_use_tanh_approximated_geglu_work() -> None:
    plan = load_model("gemma-4-31b").layer_plans[0]
    for activation in ("gelu", "geglu", "gelu_tanh", "gelu_pytorch_tanh"):
        op = gated_activation(
            plan=replace(plan, ffn_activation=activation), op_id="probe", deps=("input",),
            rows=2, width=4, dtype="bf16", stage=0, requests=(), phase="decode",
            recipe="swiglu")
        assert op.family == op.recipe == "geglu" and op.vector_ops == 8 * 2 * 4
        assert op.mte2_bytes == 2 * 8 * 2 and op.mte3_bytes == 8 * 2

def test_shared_value_projection_keeps_two_cache_planes_and_qkv_widths() -> None:
    from mindie_sim.catalog.weights import layer_weight_terms
    model = load_model("gemma-4-26b-a4b")
    global_plan, local_plan = model.layer_plans[5], model.layer_plans[0]
    assert global_plan.k_eq_v and not local_plan.k_eq_v
    terms = {term.name: term for term in layer_weight_terms(global_plan, model)}
    assert terms["q_proj"].rows == 16 * 512
    assert terms["k_proj"].rows == 2 * 512 and "v_proj" not in terms
    local_terms = {term.name: term for term in layer_weight_terms(local_plan, model)}
    assert local_terms["k_proj"].rows == local_terms["v_proj"].rows == 8 * 256
    assert global_plan.cache_row_elements == 2 * 2 * 512
    operators = target_operators(_compose("gemma-4-26b-a4b", 6), phase="decode", tokens=1)
    widths = {layer: {op.local_shape[1] for op in operators
                      if op.layer == layer and op.recipe == "qkv_projection"}
              for layer in (0, 5)}
    assert widths == {0: {8192}, 5: {9216}}

def test_customize_model_spec_conserves_parameters_layers_and_router_groups() -> None:
    from mindie_sim.schema.common import ModelingError
    for name in ("qwen3-30b-a3b", "deepseek-v2-lite", "deepseek-v4-flash-0731"):
        for experts in (16, 48):
            for top_k in (2, 8):
                spec = customize_model_spec(
                    name, layers=8, n_routed_experts=experts, num_experts_per_tok=top_k)
                assert spec.layers == len(spec.layer_plans) == 8
                assert (spec.num_experts, spec.top_k) == (experts, top_k)
                moe_plans = [p for p in spec.layer_plans if p.ffn == "moe"]
                assert moe_plans and all(p.num_experts == experts and p.top_k == top_k
                                         for p in moe_plans)
                narrow = customize_model_spec(
                    name, layers=8, n_routed_experts=experts, num_experts_per_tok=2)
                assert body_parameter_count(spec) == body_parameter_count(narrow)
    cycled = customize_model_spec(
        "deepseek-v3.2-prune-3l", layers=5, n_routed_experts=4, num_experts_per_tok=2)
    assert [p.index for p in cycled.layer_plans] == [0, 1, 2, 3, 4]
    assert [p.ffn for p in cycled.layer_plans] == ["dense", "moe", "moe", "moe", "moe"]
    assert all(p.router_groups <= 4 for p in cycled.layer_plans if p.ffn == "moe")
    with pytest.raises(ModelingError):
        customize_model_spec("qwen3-30b-a3b", layers=0)
    with pytest.raises(ModelingError):
        customize_model_spec("qwen3-30b-a3b", n_routed_experts=8, num_experts_per_tok=16)
    for name in ("qwen3-next-80b-a3b", "qwen3.5-397b-a17b"):
        hybrid = customize_model_spec(name, layers=4)
        assert all(plan.router_mode == "softmax" and plan.router_norm_topk
                   for plan in hybrid.layer_plans)
        raw = serving_case(prompt=4, output=1)
        raw["model"] = hybrid.to_dict()
        normalized = [op for op in target_operators(
            raw, phase="decode", tokens=1, context=4) if op.recipe == "route_normalize"]
        assert len(normalized) == 4
        assert all(op.vector_ops == 3 * hybrid.top_k and op.deps for op in normalized)
        raw["model"]["layer_plans"][0]["router_norm_topk"] = False
        control = target_operators(raw, phase="decode", tokens=1, context=4)
        assert sum(op.recipe == "route_normalize" for op in control) == 3

def test_resolve_role_exclusions_maps_standard_patterns_and_keeps_proxy_gaps() -> None:
    roles, unresolved = resolve_role_exclusions([
        "lm_head", "*lm_head*", "model.embed_tokens*", "*norm*", "*.mlp.gate",
        "*.mlp.shared_expert_gate", "*self_attn*", "*shared_experts*", "*mlp.experts*",
        "model.layers.0.unknown_custom_proj"])
    assert roles == ("lm_head", "embedding", "norm", "router", "attention",
                     "shared_expert", "routed_expert")
    assert unresolved == ("model.layers.0.unknown_custom_proj",)
    clean = {"torch_dtype": "bfloat16", "quantization_config": {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["lm_head", "model.embed_tokens*", "*norm*",
                                   "model.layers.*.mlp.gate", "*.mlp.shared_expert_gate"]}}
    quant, gaps = resolve_quant(clean)
    assert gaps == [] and quant.name == "hf_fp8_block"
    assert quant.weight_format == "fp8_e4m3"
    assert set(quant.exclusions) >= {"lm_head", "embedding", "norm", "router"}
    proxied, proxy_gaps = resolve_quant({**clean, "quantization_config": {
        **clean["quantization_config"], "modules_to_not_convert": ["lm_head", "*self_attn*"]}})
    assert "quant_module_exclusions_unresolved" in proxy_gaps
    assert proxied.weight_format == "bf16"

def test_engram_quant_override_moves_table_storage_mode() -> None:
    from mindie_sim.catalog.weight_terms import term_bytes
    from mindie_sim.catalog.weights import engram_table_terms
    case = parse_case(compose_case(
        "deepseek-v4.1-flash", hardware=A5_SKU, tp=8, ep=8, num_devices=8,
        ranks_per_node=8, engram_cpu_offload=False, graph_capture_mode="none",
        quant={"name": "w8a8", "weight_format": "int8", "activation_format": "int8",
               "expert_weight_format": "int8", "kv_dtype": "bf16",
               "engram_table_mode": "device_int8_fp32"}))
    assert case.model.engram.storage_mode == "device_int8_fp32"
    ledger = sum(term_bytes(term, case.model.quant, case.runtime.parallel,
                            case.machine.ranks_per_node)
                 for term in engram_table_terms(case.model))
    assert ledger == pytest.approx(case.model.engram.table_storage_bytes(
        node_rows=case.machine.ranks_per_node))  # per-node sharded table

def test_draft_indexer_and_prepared_router_are_resident() -> None:
    from mindie_sim.catalog.draft_weights import draft_terms
    from mindie_sim.catalog.weight_terms import term_bytes
    from mindie_sim.catalog.weights import stage_weight_terms

    def audit(model: str, **overrides):
        options = {"hardware": A5_SKU, "graph_capture_mode": "none",
                   "prompt_tokens": 128, "output_tokens": 2}
        options.update(overrides)
        return parse_case(compose_case(model, **options))

    glm = audit("glm-5.3-flash", tp=8, ep=8, layers=4, mtp_proposal_tokens=1)
    indexer = [term for term in draft_terms(glm.model.auxiliary.mtp, glm.model)
               if term.name.startswith("indexer_")]
    assert indexer
    assert sum(term_bytes(term, glm.model.quant, glm.runtime.parallel,
                          glm.machine.ranks_per_node) for term in indexer) == pytest.approx(14_945_280)
    pro = audit("deepseek-v4-pro", tp=8, ep=8, layers=4, mtp_proposal_tokens=2, raw_eager=True)
    prepared = [term for term in draft_terms(
        pro.model.auxiliary.mtp or pro.model.auxiliary.dspark, pro.model)
        if term.name == "draft_router_precast"]
    assert len(prepared) == 3
    assert sum(term_bytes(term, pro.model.quant, pro.runtime.parallel,
                          pro.machine.ranks_per_node) for term in prepared) == pytest.approx(33_030_144)
    staged = [term for group in stage_weight_terms(
        pro.model, pro.runtime.parallel, include_draft=True, draft_kind="dspark")
        for term in group]
    assert any(term.name == "draft_router_precast" for term in staged)

def test_pp_partition_is_shared_by_capacity_and_catalog_paths() -> None:
    from mindie_sim.catalog.cache import request_cache_breakdown
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.capacity.features import batch_cache_bytes
    case = parse_case(compose_case(
        "deepseek-v3", hardware=A5_SKU, graph_capture_mode="none", prompt_tokens=128,
        output_tokens=2, tp=8, ep=8, layers=5, pp=2, pp_layer_partition=(1, 4), quant="w8a8"))
    expected = list(Memory(case).cache_bytes(128))
    report = request_cache_breakdown(case.model, case.runtime.parallel, 128, runtime=case.runtime)
    assert report["stage_bytes"] == pytest.approx(expected)
    footprint = batch_cache_bytes(case, sequences=1, context=127, tokens_per_seq=1,
                                  phase="decode", padded=True)
    assert list(footprint.total_bytes) == pytest.approx(expected)
