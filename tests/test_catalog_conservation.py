"""37-model catalog conservation without skipped reference coverage."""
import json
from pathlib import Path

from mindie_sim.catalog.cache import (
    layer_cache_decomposition, request_cache_breakdown,
)
from mindie_sim.catalog.ownership import body_parameter_count
from mindie_sim.catalog.weights import _body_terms
from mindie_sim.model_catalog import load_model
from mindie_sim.schema.parallel import ParallelSpec

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "catalog_reference_manifest.json"
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
TOKENS = 4096
PARALLEL = ParallelSpec(tp=1, dp=1, pp=1, ep=1)


def test_manifest_covers_exact_catalog_set():
    assert MANIFEST.is_file()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["models"]) == set(MODELS)


def test_parameters_and_kv_conserve_for_every_model():
    failures = []
    for key in MODELS:
        model = load_model(key)
        declared = int(body_parameter_count(model))
        reference = int(sum(int(term.rows) * int(term.cols) * int(term.count)
                            for term in _body_terms(model) if term.source_owned))
        breakdown = request_cache_breakdown(model, PARALLEL, TOKENS)
        payload = 0.0
        allocated = 0.0
        for plan in model.layer_plans:
            for padded in (False, True):
                parts = layer_cache_decomposition(
                    model, plan, PARALLEL, TOKENS, capacity_padded=padded)
                total = sum(float(entry["total"]) for entry in parts.values())
                if padded:
                    allocated += total
                else:
                    payload += total
        deltas = {
            "parameter": declared - reference,
            "kv_payload": float(breakdown["total_bytes"]) - payload,
            "kv_allocated": float(breakdown["total_bytes"]) - max(allocated, payload),
        }
        if any(abs(value) > 0 for value in deltas.values()):
            failures.append({"model": key, **deltas})
    assert not failures, failures


def test_runtime_only_copies_do_not_inflate_logical_parameters():
    for key in ("deepseek-v4-10t", "kimi-k3.1"):
        model = load_model(key)
        logical = sum(int(term.logical_parameters) for term in _body_terms(model)
                      if term.source_owned)
        runtime_only = [term for term in _body_terms(model)
                        if not term.source_owned and term.runtime_owned]
        for term in runtime_only:
            assert term.logical_parameters > 0
        assert logical > 0
