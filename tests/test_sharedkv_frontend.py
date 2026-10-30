"""Source SCFA selection preserves the other attention branches and readiness."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_contracts import predicted_contract


def selected_case():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "tests/fixtures/verification/"
                     "v4flash-sharedkv-case.json").read_text())
    raw["runtime"].pop("routing", None)
    raw["runtime"]["kernels"].setdefault("lowerings", {})["sharedkv_attention"] = {
        "implementation": "vllm_ascend_sharedkv_scfa",
        "options": {
            "owner_schedule": "metadata_cost_balanced", "selected_rows": "valid_prefix",
            "sinks_dtype": "fp32", "metadata_s2_split": False,
            "return_softmax_lse": False, "scratch_policy": "offchip",
        },
    }
    return raw


@pytest.mark.parametrize("query,context", [(1, 0), (4, 0), (17, 511)])
def test_source_binding_executes_only_indexed_scfa_and_waits_for_producers(query, context):
    raw = selected_case()
    baseline = deepcopy(raw)
    del baseline["runtime"]["kernels"]["lowerings"]["sharedkv_attention"]
    before = target_operators(baseline, phase="prefill", tokens=query, context=context)
    selected = parse_case(raw)
    after = target_operators(selected, phase="prefill", tokens=query, context=context)
    source = []
    for old, new in zip(before, after, strict=True):
        if new.recipe == "vllm_ascend_sharedkv_scfa":
            assert old.family == new.family == "attention"
            assert old.deps == new.deps
            assert dict(new.metadata)["sharedkv_attention_geometry"]["branch"] == "SCFA"
            assert predicted_contract(new.to_dict()) == predicted_contract(old.to_dict())
            assert predicted_contract(new.to_dict())[1] == "SparseAttnSharedkv"
            assert new.source_boundary == "kernel"
            source.append(new)
        else:
            assert old == new
    assert len(source) == 21
    assert sum(op.family == "attention" for op in after) == 43

    # Synthetic declared copy resources establish causality, not A3 accuracy.
    dies = []
    for die in selected.machine.dies:
        rates = {**dict(die.rates), **dict.fromkeys(
            ("mte2_aic", "mte2_aiv", "mte1", "mte3", "fixpipe"), 1e12)}
        units = {**dict(die.resource_units), "mte2_aic": die.compute_units,
                 "mte1": die.compute_units, "fixpipe": die.compute_units,
                 "vector": 2 * die.compute_units, "mte2_aiv": 2 * die.compute_units,
                 "mte3": 2 * die.compute_units}
        dies.append(replace(die, rates=tuple(rates.items()), resource_units=tuple(units.items())))
    selected = replace(selected, machine=replace(selected.machine, dies=tuple(dies)))
    op = source[0]
    result = graph_costs(selected, (op,), input_ready_ms=7,
                         external_inputs={dep: 11 for dep in op.deps}, trace_limit=10000)
    assert all(row["start_ms"] >= 11 for row in result["events"] if row["demand"])
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert not any(result["execution"]["live_allocated"].values())
    assert result["isolated"]["operators"][0]["operator"]["source_boundary"] == "kernel"
    # An unrelated selected kernel cannot inherit the SCFA ABI from its shape.
    foreign = replace(op, recipe="other_source_kernel")
    assert predicted_contract(foreign.to_dict()) is None
    changed = dict(op.metadata)
    changed["sharedkv_attention_geometry"] = {
        **changed["sharedkv_attention_geometry"], "width": 256}
    assert predicted_contract(replace(op, metadata=tuple(changed.items())).to_dict()) is None


def test_decode_stays_separate_and_multisequence_source_miss_is_visible():
    raw = selected_case()
    decode = target_operators(raw, phase="decode", tokens=1, context=4096)
    assert not any(op.recipe == "vllm_ascend_sharedkv_scfa" for op in decode)
    with pytest.raises(ModelingError, match="single_contiguous_prefill"):
        target_operators(raw, phase="prefill", tokens=8, sequences=2)
