"""Independent 37-model catalog conservation against constructor-shape oracles.

[Role]
  Consume the declared reference manifest, rebuild every parameter shape and KV
  cache byte total from the raw JSON config through ``tests/_oracles`` and
  compare the result with the simulator's declared terms.  The simulator is
  imported only here for the actual side of the comparison; the oracle modules
  never import it.

[Invariants]
  * Every manifest key is checked; a missing reference or shape is a failure,
    never a skip.
  * Parameter totals, per-name shapes, row/column/count, ownership and
    payload/allocated cache bytes are compared separately.
  * Token counts cover 0, 1, block-1, block, window-1 and window; TP, PP and CP
    variants exercise KV replication and block-interleaved sharding.

[Timeouts & Locks]
  Pure CPU work with per-process memoized model loads; no timeouts or locks.

[Side-Effect Policy]
  Read-only; no filesystem writes, device context or network.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path

import pytest

from _oracles.catalog_meta_oracle import (
    Entry, build_reference, compare_entries, load_manifest, logical_numel,
    make_meta_parameters, manifest_models,
)
from _oracles.cache_meta_oracle import cache_totals

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "catalog_reference_manifest.json"
BLOCK = 128
CAPTURE_TOKENS = 4096
MODELS = (
    "deepseek-v4-10t", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813",
    "deepseek-v4.1-flash", "deepseek-v2-lite", "deepseek-v3.2-prune-3l",
    "deepseek-v3", "deepseek-r1", "deepseek-v3.2", "glm-5.3",
    "glm-5.3-flash", "glm-5.1", "glm-5.2", "kimi-k3", "kimi-k3.1",
    "kimi-k3-dspark", "kimi-k3-dspark-block5", "kimi-k2", "kimi-k2-thinking",
    "kimi-k2.5", "qwen3-30b-a3b", "qwen3-32b", "qwen3-235b-a22b",
    "qwen3-next-80b-a3b", "qwen3.5-27b", "qwen3.5-35b-a3b",
    "qwen3.5-397b-a17b", "qwen3.6-27b", "qwen3.8-2.4t-a95b", "qwen3.8-27b",
    "minimax-m3", "minimax-m3.1-flash", "gemma-4-26b-a4b", "gemma-4-31b",
    "qwen3.6-35b-a3b", "qwen3.8-flash-next", "step-5-preview",
)
PARALLEL_CASES = ((1, 1, 1), (2, 1, 1), (8, 1, 1), (1, 2, 1), (1, 1, 2))


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
    from mindie_sim.model_catalog import load_model
    return load_model(key)


def _actual_entries(model) -> list[tuple]:
    from mindie_sim.catalog.weights import _body_terms
    return [(term.name, int(term.rows), int(term.cols), int(term.count),
             "source" if term.source_owned else "runtime")
            for term in _body_terms(model)]


def _cache_bytes(model, parallel, tokens: int, padded: bool) -> float:
    from mindie_sim.catalog.cache import layer_cache_decomposition
    total = 0.0
    for plan in model.layer_plans:
        parts = layer_cache_decomposition(
            model, plan, parallel, tokens, capacity_padded=padded)
        total += float(sum(float(part["total"]) for part in parts.values()))
    return total


def _sim_auxiliary(model) -> dict:
    aux = getattr(model, "auxiliary", None)
    if aux is None:
        return {}
    for kind in ("mtp", "dspark"):
        spec = getattr(aux, kind, None)
        if spec is not None:
            return {"kind": str(spec.kind), "layers": int(spec.layers),
                    "standalone": bool(aux.standalone_draft)}
    return {}


def test_manifest_covers_exact_catalog_set():
    models = manifest_models(_manifest())
    assert len(models) == len(MODELS) == 37
    assert set(models) == set(MODELS)
    for key, entry in models.items():
        assert entry["parameter_scope"] == "text_backbone"
        for name in ("config", "config_sha256", "builder", "cache_layout",
                     "auxiliary_modes", "sharing_relations", "quantization"):
            assert name in entry, (key, name)
        path = ROOT / entry["config"]
        assert path.is_file(), (key, entry["config"])
        assert _reference(key).config_sha256 == entry["config_sha256"]


def test_manifest_spans_both_catalog_config_directories():
    paths = [entry["config"] for entry in manifest_models(_manifest()).values()]
    assert sum(path.startswith("mindie_sim/model_configs/") for path in paths) == 30
    assert sum(path.startswith("mindie_sim/catalog/configs/") for path in paths) == 7


@pytest.mark.parametrize("key", MODELS)
def test_parameters_and_named_shapes_conserve(key):
    model = _model(key)
    reference = _reference(key)
    comparison = compare_entries(reference, _actual_entries(model))
    assert comparison["complete"], {
        "parameter_delta": comparison["parameter_delta"],
        "runtime_delta": comparison["runtime_delta"],
        "missing_in_actual": comparison["missing_in_actual"][:5],
        "missing_in_reference": comparison["missing_in_reference"][:5],
        "shape_mismatches": comparison["shape_mismatches"][:5],
    }
    assert comparison["parameter_delta"] == 0
    assert comparison["runtime_delta"] == 0
    assert comparison["reference_parameters"] > 0
    assert comparison["actual_parameters"] == comparison["reference_parameters"]


def test_meta_parameters_are_real_tensors_and_deduplicate_aliases():
    reference = _reference("deepseek-v4.1-flash")

    class _Shape:
        def __init__(self, numel):
            self._numel = numel

        def numel(self):
            return self._numel

    shared = _Shape(12)
    assert logical_numel({"a": shared, "b": shared}, [("a", 2), ("b", 1)]) == 24
    if importlib.util.find_spec("torch") is None:
        return
    import torch
    parameters, names = reference.meta_parameters()
    assert parameters
    for name, _count, _owner in names[:64]:
        assert parameters[name].device.type == "meta"
        assert parameters[name].requires_grad is False
    source = [(name, count) for name, count, owner in names if owner == "source"]
    assert logical_numel(parameters, source) == reference.source_total()
    with torch.device("meta"):
        parameter = torch.nn.Parameter(torch.empty((3, 4)), requires_grad=False)
    assert logical_numel({"a": parameter, "b": parameter}, [("a", 2), ("b", 1)]) == 24
    aliased = make_meta_parameters({"owner": (5, 7)}, {"alias": "owner"})
    assert aliased["alias"] is aliased["owner"]
    assert logical_numel(aliased, [("owner", 3), ("alias", 3)]) == 105


@pytest.mark.parametrize("key", MODELS)
def test_kv_conserves_across_token_and_parallel_matrix(key):
    from mindie_sim.catalog.cache import request_cache_breakdown
    from mindie_sim.schema.parallel import ParallelSpec
    model = _model(key)
    reference = _reference(key)
    window = max([layer.window for layer in reference.layers] + [BLOCK])
    tokens = sorted({0, 1, BLOCK - 1, BLOCK, max(0, window - 1), window, CAPTURE_TOKENS})
    for tp, pp, cp in PARALLEL_CASES:
        parallel = ParallelSpec(tp=tp, dp=1, pp=pp, cp=cp)
        for seq in tokens:
            payload = _cache_bytes(model, parallel, seq, padded=False)
            allocated = _cache_bytes(model, parallel, seq, padded=True)
            breakdown = request_cache_breakdown(model, parallel, seq)
            assert float(breakdown["total_bytes"]) == pytest.approx(allocated, abs=1e-6)
            reference_payload = cache_totals(
                reference.geometry, seq, padded=False, tp=tp, cp=cp)
            reference_allocated = cache_totals(
                reference.geometry, seq, padded=True, tp=tp, cp=cp)
            assert payload - reference_payload == pytest.approx(0.0, abs=1e-6), (
                key, tp, pp, cp, seq, "payload")
            assert allocated - reference_allocated == pytest.approx(0.0, abs=1e-6), (
                key, tp, pp, cp, seq, "allocated")


@pytest.mark.parametrize("key", MODELS)
def test_auxiliary_quantization_and_sharing_declarations(key):
    model = _model(key)
    entry = _entry(key)
    assert model.accounting_scope == entry["parameter_scope"] == "text_backbone"
    declared_aux = entry["auxiliary_modes"]
    actual_aux = _sim_auxiliary(model)
    if actual_aux:
        assert declared_aux["kind"] == actual_aux["kind"]
        assert int(declared_aux["layers"]) == actual_aux["layers"]
        assert bool(declared_aux["standalone"]) is actual_aux["standalone"]
    else:
        assert declared_aux == {}
    quant = entry["quantization"]
    assert quant["kv_dtype"] == model.quant.kv_dtype
    assert quant["index_dtype"] == model.quant.index_dtype
    assert quant["index_scale_dtype"] == (model.quant.index_scale_dtype or "none")
    reference = _reference(key)
    assert [layer.attention for layer in reference.layers] == [
        plan.attention for plan in model.layer_plans]
    assert [layer.ffn for layer in reference.layers] == [
        plan.ffn for plan in model.layer_plans]
    assert [layer.window for layer in reference.layers] == [
        plan.window for plan in model.layer_plans]
    assert [layer.compress_ratio for layer in reference.layers] == [
        plan.compress_ratio for plan in model.layer_plans]
    assert [layer.owns_index_k for layer in reference.layers] == [
        plan.owns_index_k for plan in model.layer_plans]
    sharing = entry["sharing_relations"]
    assert sharing["tied_word_embeddings"] is bool(model.tie_word_embeddings)
    assert sharing["kv_owner_layers"] == [
        plan.index for plan in model.layer_plans if plan.kv_owner_layer_id == plan.index]
    assert sharing["shared_index_layers"] == [
        plan.index for plan in model.layer_plans if plan.indexer_type == "shared"]
    assert sharing["k_eq_v_layers"] == [
        plan.index for plan in model.layer_plans if plan.k_eq_v]


def test_reference_gate_rejects_missing_or_incomplete_declarations():
    entry = dict(_entry("deepseek-v3"))
    without_sha = {key: value for key, value in entry.items() if key != "config_sha256"}
    with pytest.raises(ValueError, match="reference_missing_sha256"):
        build_reference("deepseek-v3", ROOT, without_sha)
    with pytest.raises(ValueError, match="reference_sha256_mismatch"):
        build_reference("deepseek-v3", ROOT, {**entry, "config_sha256": "0" * 64})
    without_layout = {key: value for key, value in entry.items() if key != "cache_layout"}
    with pytest.raises(ValueError, match="reference_declaration_missing"):
        build_reference("deepseek-v3", ROOT, without_layout)
    with pytest.raises(ValueError, match="reference_scope_unsupported"):
        build_reference("deepseek-v3", ROOT, {**entry, "parameter_scope": "full_checkpoint"})
    with pytest.raises(ValueError, match="reference_config_missing"):
        build_reference("deepseek-v3", ROOT, {**entry, "config": "mindie_sim/nope.json"})


def test_named_shape_comparison_detects_a_missing_shape():
    reference = _reference("qwen3-32b")
    actual = _actual_entries(_model("qwen3-32b"))
    assert compare_entries(reference, actual)["complete"]
    broken = compare_entries(reference, actual[1:])
    assert not broken["complete"]
    assert broken["missing_in_actual"]
    assert broken["parameter_delta"] != 0
    extra = compare_entries(reference, actual + [("bogus", 1, 1, 1, "source")])
    assert not extra["complete"]
    assert extra["missing_in_reference"] == [["bogus", 1, 1, 1, "source"]]
    assert isinstance(reference.entries()[0], Entry)
