"""Ordinary sparse-attention selection, finite owners and real Engine service."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_operators
from mindie_sim.prediction.indexed_attention import IMPLEMENTATION, _phases, query_positions
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_contracts import predicted_contract
from test_prediction_system import case


def source_case():
    raw = case(tp=8)
    raw["model"] = {"catalog_key": "GLM-5.3-W4A8", "layers": 1}
    device = raw["machine"]["device"]
    device["compute_units"] = 20
    device["compute_peaks"] = {"bf16": 2.7e14, "int8": 5.4e14, "fp32": 1.35e14}
    device["rates"] = {
        "hbm": 1.2e12, "vector": 2e12, "mte2_aic": 1e12, "mte2_aiv": 2e12,
        "mte1": 4e12, "mte3": 4e12, "fixpipe": 4e12,
    }
    device["resource_units"] = {
        "hbm": 1, "vector": 40, "mte2_aic": 20, "mte2_aiv": 40,
        "mte1": 20, "mte3": 40, "fixpipe": 20,
    }
    device["service_modes"] = {
        "mte1_l0a": {"resource": "mte1", "rate": 8e12},
        "mte1_l0b": {"resource": "mte1", "rate": 4e12},
    }
    raw["runtime"]["kernels"] = {"lowerings": {
        "indexed_attention": {"implementation": IMPLEMENTATION,
                              "options": {"scratch_policy": "offchip"}},
    }}
    return raw


@pytest.mark.parametrize("phase,query,context", [
    ("decode", 1, 4096), ("decode", 4, 4096), ("prefill", 21, 0),
])
def test_native_binding_preserves_tensor_contract_and_real_input_dependencies(
        phase, query, context):
    raw = source_case()
    baseline = deepcopy(raw)
    baseline["runtime"].pop("kernels")
    before = target_operators(baseline, phase=phase, tokens=query, context=context)
    after = target_operators(raw, phase=phase, tokens=query, context=context)
    source = []
    for old, new in zip(before, after, strict=True):
        if new.recipe == IMPLEMENTATION:
            assert old.deps == new.deps
            assert new.source_boundary == "kernel"
            assert predicted_contract(new.to_dict()) == predicted_contract(old.to_dict())
            source.append(new)
        else:
            assert old == new
    assert len(source) == 1
    op = source[0]
    report = graph_costs(
        raw, (op,), input_ready_ms=7,
        external_inputs={dep: 11 for dep in op.deps}, trace_limit=10000)
    assert report["duration_ms"] > 0
    assert all(row["start_ms"] >= 11 for row in report["events"] if row["demand"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())
    foreign = replace(op, recipe="other_source")
    assert predicted_contract(foreign.to_dict()) is None
    changed = {**dict(op.metadata), "indexed_attention_geometry": {
        **dict(op.metadata)["indexed_attention_geometry"], "heads": 4}}
    assert predicted_contract(replace(op, metadata=tuple(changed.items())).to_dict()) is None


def test_contiguous_queries_conserve_work_without_hiding_causal_owner_imbalance():
    positions = [list(query_positions(owner, 21, 20)) for owner in range(20)]
    assert [p for owned in positions for p in owned] == list(range(21))
    assert positions[0] == [0, 1] and positions[10] == [20]
    loads = [sum(min(p + 1, 2048) for p in owned) for owned in positions]
    assert sum(loads) == 21 * 22 // 2
    assert max(loads) == 39
    # Round-robin's 22-row maximum would redistribute query work away from its
    # source owner and understate the tail despite conserving aggregate bytes.
    assert max(loads) > max(sum(p + 1 for p in range(c, 21, 20)) for c in range(20))


def test_operand_directions_share_service_and_expose_slow_b_rate():
    legacy = dict(_phases(512, 8, 512, 64, first=True, last=True,
                          scratch_policy="offchip"))
    directional = dict(_phases(512, 8, 512, 64, first=True, last=True,
                               scratch_policy="offchip", operand_modes=True))
    for name in ("qk_load", "pv_load"):
        work = directional[name].resources
        assert work["mte1_l0a"] + work["mte1_l0b"] == legacy[name].resources["mte1"]
    raw = source_case()
    op = next(op for op in target_operators(raw, phase="decode", tokens=1, context=4096)
              if op.recipe == IMPLEMENTATION)
    external = {dep: 0 for dep in op.deps}
    fast = graph_costs(raw, (op,), external_inputs=external)["duration_ms"]
    raw["machine"]["device"]["service_modes"]["mte1_l0b"]["rate"] = 1e8
    slow = graph_costs(raw, (op,), external_inputs=external)["duration_ms"]
    assert slow > 100 * fast


def test_source_domain_miss_is_not_replaced_with_aggregate_envelope():
    raw = source_case()
    with pytest.raises(ModelingError, match="single_contiguous_sequence"):
        target_operators(raw, phase="prefill", tokens=8, sequences=2)
    raw["machine"]["device"].pop("service_modes")
    op = next(op for op in target_operators(raw, phase="decode", tokens=1, context=4096)
              if op.recipe == IMPLEMENTATION)
    with pytest.raises(ModelingError, match="physical_resource_rate_missing:mte1_l0"):
        graph_costs(parse_case(raw), (op,), external_inputs={dep: 0 for dep in op.deps})
