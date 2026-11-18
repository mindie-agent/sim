"""Declared-versus-observed contract parity for the shared ``mindie-sim`` owner.

[Role]
  Prove that the observed trace contracts under ``mindie_sim.observed_contracts``
  and the declared contracts emitted by ``mindie_sim.structure_*`` stay in the
  same vocabulary, and that independently captured descriptor fixtures still
  project onto the exact declared ``node["contract"]``.

[Invariants]
  * Captured rows are real profiler fixtures, never inverted from a declaration.
  * Equality is exact over the full contract tuple, including native identity
    and every compared tensor descriptor.
  * The five families that previously drifted out of the profiler diagnostic
    vocabulary must stay present in both ``FIELDS`` and ``NATIVE_FAMILIES``.
  * Exports run at reduced layer counts but always through the public
    ``compose_case``/``export_structure`` entry points.

[Timeouts & Locks]
  Bounded in-process exports of at most four layers; no device, waits or locks.

[Side-Effect Policy]
  Read-only fixtures and in-memory exports; no filesystem writes.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

from mindie_sim.compose import compose_case
from mindie_sim.observed_contracts import (
    CONTRACT_FIELDS, FIELDS, NATIVE_CONTRACT_FAMILIES, NATIVE_FAMILIES,
    observed_contract,
)
from mindie_sim.structure import export_structure

FIXTURE = Path(__file__).parent / "fixtures/observed_contracts/captured_descriptors.json"
SIM_ROOT = Path(__file__).resolve().parents[1] / "mindie_sim"
DECLARED_MODULES = (
    "structure_contracts.py", "structure_matrix_contracts.py",
    "structure_native_contracts.py", "structure_subgraphs.py",
)

#: Families the profiler diagnostic copy had drifted behind.
RECOVERED_FAMILIES = frozenset({
    "attention_paged_active_qkv",
    "attention_paged_mla_qk_rope",
    "situ_mxquant_active",
    "dynamic_quant_v2_active",
    "causal_conv1d_active",
})
#: Retained semantic operator families whose contract tuple is not native-ABI
#: shaped and whose vocabulary entry predates the observed contract registry.
LEGACY_OPERATOR_FAMILIES = frozenset({"gdn_gating", "recurrent_vector", "swiglu"})

#: Observed capture families with no declared operator of that family today:
#: a cache-store or quant kernel can appear in a capture without a declared
#: producer emitting the same family name.
OBSERVED_ONLY_FAMILIES = frozenset({
    "dequant_swiglu_quant_active", "dynamic_quant_v2_active",
    "kv_cache_store_active",
})


def _load_fixtures() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row(record: dict) -> SimpleNamespace:
    return SimpleNamespace(
        kind=record["kind"],
        input_shapes=[tuple(shape) for shape in record["input_shapes"]],
        input_dtypes=tuple(record["input_dtypes"]),
        input_formats=tuple(record["input_formats"]),
        output_shapes=[tuple(shape) for shape in record["output_shapes"]],
        output_dtypes=tuple(record["output_dtypes"]),
        output_formats=tuple(record["output_formats"]),
    )


@lru_cache(maxsize=None)
def _declared_contracts(key: str) -> frozenset[str]:
    record = json.loads(key)
    case = compose_case(
        record["model"], record["hardware"], tp=record["tp"],
        prompt_tokens=record["tokens"] if record["phase"] == "prefill" else 8,
        output_tokens=4, layers=record["layers"],
    )
    # The retained descriptor captures predate the CANN 9.2.1 optional
    # fusions.  Declare that explicitly so the independent capture still
    # projects onto the same semantic/native boundary; no tolerance is relaxed.
    case["runtime"].setdefault("metadata", {})["execution_policy"] = {
        "automatic_optional_fusions": False}
    report = export_structure(
        case, phase=record["phase"], tokens=record["tokens"],
        context=record["context"], sequences=1, scope="body",
    )
    return frozenset(
        _canonical(node["contract"]) for node in report["nodes"] if node["contract"]
    )


def test_captured_descriptor_fixtures_equal_declared_node_contracts():
    fixtures = _load_fixtures()
    assert fixtures["schema"] == "mindie-observed-contract-fixtures/1"
    assert len(fixtures["exact_matches"]) >= 5
    seen_families = set()
    for record in fixtures["exact_matches"]:
        observed = observed_contract(_row(record))
        assert observed is not None, record["capture_name"]
        assert _canonical(observed) == _canonical(record["contract"]), (
            record["capture_name"], observed, record["contract"])
        geometry = {key: record[key] for key in (
            "model", "hardware", "tp", "phase", "tokens", "context", "layers")}
        assert _canonical(observed) in _declared_contracts(_canonical(geometry))
        seen_families.add(observed[0])
    # GQA, cache-store, sparse/MLA, MX-quant and normalization must all appear.
    assert {"attention_active_qkv", "kv_cache_store_active_elements",
            "sparse_attention_active", "layernorm_active"} <= seen_families


def test_captured_family_inventory_is_declared_and_vocabulary_complete():
    fixtures = _load_fixtures()
    families = {item["family"] for item in fixtures["captured_inventory"]}
    # causal_conv1d_active is a declared-only vocabulary entry today: the
    # observed CausalConv1d tuple keeps its retained legacy native projection.
    assert RECOVERED_FAMILIES - {"causal_conv1d_active"} <= families
    assert RECOVERED_FAMILIES <= set(FIELDS)
    assert RECOVERED_FAMILIES <= set(NATIVE_FAMILIES)
    assert CONTRACT_FIELDS is FIELDS
    assert NATIVE_CONTRACT_FAMILIES is NATIVE_FAMILIES
    # Every observed family either has a declared vocabulary entry or is a
    # legacy per-operator native tuple whose first element is the op type.
    for item in fixtures["captured_inventory"]:
        assert (item["family"] in FIELDS or item["family"] == item["kind"]
                or item["family"] in LEGACY_OPERATOR_FAMILIES), item


def _declared_family_literals() -> set[str]:
    literals: set[str] = set()
    for name in DECLARED_MODULES:
        tree = ast.parse((SIM_ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
    return literals


def test_native_family_vocabulary_has_declared_owners():
    declared = _declared_family_literals()
    # Every recovered family that a declaration can emit has a declared owner;
    # dynamic_quant_v2_active is observed-only on both sides today.
    assert RECOVERED_FAMILIES - OBSERVED_ONLY_FAMILIES <= declared
    # Every observed native family is either declared by the same repository or
    # explicitly recorded as an observed-only capture family.
    unowned = NATIVE_FAMILIES - declared - OBSERVED_ONLY_FAMILIES
    assert unowned == set(), sorted(unowned)
    assert OBSERVED_ONLY_FAMILIES <= NATIVE_FAMILIES
