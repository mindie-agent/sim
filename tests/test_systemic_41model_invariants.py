"""Systemic 41-model invariants: no silent fallback, bounded identity, real gate.

[Role] One gate over every built-in catalog model plus the timing refusal
  contract for genuinely unmodeled attention structures.
[Invariants]
  * ``compose_case`` reaches a declared layer plan for all 41 models on A5 with
    no ``unmodeled``/``projection_only`` attention core, no
    ``compressed-tensors`` BF16 proxy and no default ``TiledMatmul``.
  * The unique ``physical_key`` count is bounded by the layer archetypes: a
    homogeneous schedule keeps an identical count across 2/6/12 layers and no
    model grows past three times its two-layer identity set at six layers.
  * An unmodeled attention core is refused by every timing entry with a
    ``mindie-unmodeled-structure-hint/1`` payload while ``export_structure``
    still succeeds and its ``graph_id`` remains self-verifying *including* the
    hints.
[Timeouts & Locks] Bounded catalog loads and one tiny graph per model; no
  device, network or filesystem writes.
[Side-Effect Policy] Read-only; pure records and one in-memory digest.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from mindie_sim import compose_case, operator_costs, predict, target_graphs, target_operators
from mindie_sim.model_catalog import list_models
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency, solve_sla_sizing,
)
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from mindie_sim.prediction.physical_key import physical_key
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.structure import export_structure

MODELS = tuple(entry["key"] for entry in list_models())
HARDWARE = "a5"
LAYER_COUNTS = (2, 6, 12)
#: A two-layer prefix can miss a model's later archetypes (dense->MoE switch,
#: indexer owners). Three times remains the ceiling of a layer-linear identity
#: set, which the relative key never reaches.
LINEAR_CEILING = 3
#: Scaffolding counts that do not scale with layer count.
SCAFFOLD_SLACK = 4
GRAPH_SCOPE = dict(scope="body", phase="decode", tokens=1, context=128,
                   sequences=1)


def _case(key: str, layers: int) -> dict:
    return compose_case(key, hardware=HARDWARE, layers=layers)


def _ops(case: dict) -> tuple:
    graphs = target_graphs(case, **GRAPH_SCOPE)
    return tuple(op for items in graphs.values() for op in items)


def _archetypes(case: dict) -> set[tuple]:
    return {
        (plan["attention"], plan["ffn"], plan["indexer"], plan["window"],
         plan["compress_ratio"])
        for plan in case["model"]["layer_plans"]
    }


def _declared_gaps(case: dict) -> tuple[str, ...]:
    meta = dict((case["model"].get("metadata") or ()))
    return tuple(meta.get("unmodeled_features") or ())


def test_default_matrix_program_is_steady_and_tiled_is_explicit_only():
    programs = RecipeBook().programs
    assert isinstance(programs["matmul"], SteadyMatmul)
    assert type(programs["matmul"]).__name__ == "SteadyMatmul"


@pytest.mark.parametrize("key", MODELS, ids=list(MODELS))
def test_catalog_model_composes_without_silent_attention_or_quant_fallback(key):
    case = _case(key, 2)
    plans = case["model"]["layer_plans"]
    assert plans, key
    holes = [(plan["index"], plan["attention"]) for plan in plans
             if plan["attention"] in {"unmodeled", "projection_only"}]
    assert holes == [], f"{key} kept a silent attention core hole: {holes}"
    gaps = _declared_gaps(case)
    assert not any("compressed-tensors" in gap and "proxy" in gap for gap in gaps), gaps
    assert not any(gap.startswith("quantization_geometry_unresolved:compressed-tensors")
                   for gap in gaps), gaps
    quant = case["model"].get("quant") or {}
    # Packed INT4 never becomes an INT8 compute identity without an INT8
    # activation declaration, and the W4A16 byte identity is 0.5625 B/elem.
    assert not (quant.get("weight_format") == "int4"
                and quant.get("activation_format") == "int8"), key
    if quant.get("weight_format") == "int4":
        assert quant.get("activation_format") not in {"int8", ""}, key
        assert quant.get("weight_block") == [1, 32], key
    graphs = target_graphs(case, **GRAPH_SCOPE)
    kinds = {graphs.effective_kernels["programs"]["matmul"]["kind"]}
    assert kinds == {"steady_matmul"}, f"{key} still selected {kinds}"


@pytest.mark.parametrize("key", MODELS, ids=list(MODELS))
def test_physical_key_count_is_bounded_by_layer_archetypes_not_layer_count(key):
    cases = {layers: _case(key, layers) for layers in LAYER_COUNTS}
    counts = {layers: len({physical_key(op) for op in _ops(case)})
              for layers, case in cases.items()}
    assert counts[2] > 0 and counts[6] > 0 and counts[12] > 0
    assert counts[6] <= LINEAR_CEILING * counts[2], (key, counts)
    if len(_archetypes(cases[6])) == 1:
        # One layer archetype: the relative identity set must not grow at all.
        assert counts[2] == counts[6] == counts[12], (key, counts)
    else:
        assert counts[12] <= counts[6] + counts[2] + SCAFFOLD_SLACK, (key, counts)


def _synthetic_unmodeled_case() -> dict:
    raw = _case("qwen3-32b", 2)
    plan = raw["model"]["layer_plans"][0]
    plan["attention"] = "unmodeled"
    plan["architecture"] = "synthetic_unmodeled_attention"
    return raw


@pytest.mark.parametrize("entry", [
    "predict", "operator_costs", "analyze_network_efficiency", "solve_sla_sizing",
])
def test_timing_entries_refuse_an_unmodeled_attention_core(entry):
    raw = _synthetic_unmodeled_case()
    calls = {
        "predict": lambda: predict(raw),
        "operator_costs": lambda: operator_costs(
            raw, target_operators(raw, phase="prefill", tokens=3)),
        "analyze_network_efficiency": lambda: analyze_network_efficiency(
            raw, phase="prefill", tokens=3),
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
        assert hint["observed_abi_summary"]["charging_policy"] == (
            "no_timing_fabricated_for_unmodeled_core")
    assert error.onboarding_hint["schema"] == "mindie-unmodeled-structure-hint/1"
    assert error.onboarding_hint["ssot_owner_repo"] == "mindie-sim"


def test_structure_export_keeps_hints_inside_the_self_verifying_digest():
    raw = _synthetic_unmodeled_case()
    report = export_structure(raw, phase="prefill", tokens=3)
    hints = report["unmodeled_structure_onboarding_hints"]
    assert hints and all(hint["schema"] == "mindie-unmodeled-structure-hint/1"
                         for hint in hints)
    assert all(hint["ssot_owner_repo"] == "mindie-sim" for hint in hints)
    payload = {key: value for key, value in report.items()
               if key not in ("graph_id", "source")}
    digest = sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    assert report["graph_id"] == "sha256:" + digest
    # Dropping the hint payload must break the digest: it is hashed content.
    without = {key: value for key, value in payload.items()
               if key != "unmodeled_structure_onboarding_hints"}
    other = sha256(json.dumps(
        without, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    assert other != digest
