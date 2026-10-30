"""Source state address selection without acceptance-scaled physical work."""
from copy import deepcopy
from itertools import accumulate

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.operators import kda_recurrent as arch35
from mindie_sim.prediction.operators import kda_recurrent_arch22 as arch22
from mindie_sim.prediction.operators.kda_state_selection import resolve
from mindie_sim.schema.common import ModelingError
from test_kda_decode_family import caller
from test_kda_recurrent_source import execute


def exact(lengths=(2, 4), accepted=(1, 4), *, packed=False):
    rows = [[17, 4, -1, -1], [31, 11, 27, 2]]
    return {"kind": "exact", "prefix": list(accumulate(lengths, initial=0)),
            "index_layout": "packed" if packed else "speculative",
            "index_row_stride": 0 if packed else 4,
            "state_indices": [17, 4, 31, 11, 27, 2] if packed else rows,
            "accepted_tokens": accepted, "state_capacity": 32}


def symbolic(stride=6):
    return {"kind": "symbolic_valid_selection", "index_layout": "speculative",
            "index_row_stride": stride,
            "validity": {name: True for name in
                         ("prefix", "state_indices", "accepted_tokens", "independent_sequences")}}


@pytest.mark.parametrize("packed", (False, True))
def test_initial_state_uses_previous_acceptance_but_checkpoints_cover_all_current_queries(packed):
    before = resolve(exact(packed=packed), (2, 4))
    after = resolve(exact(accepted=(2, 1), packed=packed), (2, 4))
    assert before["initial_slots"] == (17, 2)
    assert after["initial_slots"] == (4, 31)
    assert before["checkpoint_slots"] == after["checkpoint_slots"] == ((17, 4), (31, 11, 27, 2))
    # Padded metadata cells are not queried, even if they contain sentinels.
    assert -1 not in {slot for row in before["checkpoint_slots"] for slot in row}


@pytest.mark.parametrize("module", (arch22, arch35))
@pytest.mark.parametrize("units,heads", ((1, 3), (4, 7), (40, 4)))
def test_accepted_state_keeps_resident_payload_queues_and_actual_aiv_work(module, units, heads):
    kwargs = dict(heads=heads, key_heads=1, units=units, planner=module.build,
                  constructor=module.make_op, rates={"mte3": 1e7, "vector": 3e7})
    baseline = execute((2, 4), **kwargs)
    variants = [exact(), exact(accepted=(2, 1)), symbolic(4)]
    for declaration in variants:
        op, plan, engine, duration = execute((2, 4), state_selection=declaration, **kwargs)
        assert dict(op.metadata)["kda_recurrent_params"]["has_accepted_tokens"] is True
        assert plan.work == baseline[1].work
        assert engine.serviced == pytest.approx(baseline[2].serviced)
        assert duration == pytest.approx(baseline[3], rel=1e-12)
        assert engine.peak_allocated == baseline[2].peak_allocated
        assert engine.serviced["mte3"] == pytest.approx(6 * heads * (128*128*4 + 128*2), rel=1e-12)
        blocks = min(units, 2*heads)
        loads = 2*blocks + 2*min(heads, blocks) if module is arch22 else 4*heads
        schedule = plan.metadata["kda_recurrent_schedule"]
        assert schedule["active_aiv"] == blocks
        assert schedule["accepted_metadata_getvalue_bytes"] == 4*loads
        assert schedule["scalar_metadata_getvalue_bytes"] == (
            baseline[1].metadata["kda_recurrent_schedule"]["scalar_metadata_getvalue_bytes"] + 4*loads)


def test_ordinary_source_binding_waits_for_state_metadata_and_retains_one_launch():
    declaration = exact()
    declaration["dependencies"] = ["metadata_ready", "state_ready"]
    case, ops = caller((2, 4), options={"state_selection": declaration})
    native = next(op for op in ops if op.recipe == arch35.IMPLEMENTATION)
    assert native.deps[-2:] == ("metadata_ready", "state_ready")
    assert dict(native.metadata)["state_selection"]["initial_slots"] == (17, 2)
    group, = dict(native.metadata)["structure_subgraphs"]
    assert group["tensor_view"]["has_accepted_tokens"] is True
    assert group["implementation"]["conditions"]["state_slot_mapping"] == "exact"
    result = graph_costs(case, ops, input_ready_ms=7, isolated=False,
                         external_inputs={"input": 7, "metadata_ready": 11, "state_ready": 13},
                         trace_limit=3000)
    launches = [row for row in result["timeline"] if row["recipe"] == arch35.IMPLEMENTATION]
    assert len(launches) == 1 and launches[0]["ready_ms"] >= 13
    execution = result["execution"]
    assert execution["live_tasks"] == execution["live_programs"] == 0
    assert not any(execution["live_allocated"].values())
    native_events = [row for row in result["events"] if ":state_load" in row["task"]]
    assert native_events and all(row["start_ms"] >= 13 for row in native_events)


def test_exact_duplicate_slot_is_ordered_within_sequence_but_cross_sequence_alias_is_not():
    declaration = exact()
    declaration["state_indices"][0] = [17, 17, -1, -1]
    assert resolve(declaration, (2, 4))["checkpoint_slots"][0] == (17, 17)
    declaration["state_indices"][1][0] = 17
    with pytest.raises(ModelingError, match="cross_sequence_state_alias"):
        resolve(declaration, (2, 4))


def test_invalid_source_metadata_cannot_silently_price_a_skipped_sequence():
    for field, value, reason in (
        ("accepted_tokens", (0, 4), "accepted_values"),
        ("accepted_tokens", (1, 5), "accepted_values"),
        ("prefix", [0, 1, 6], "prefix"),
        ("state_capacity", 31, "slot_outside"),
    ):
        declaration = exact();declaration[field] = value
        with pytest.raises(ModelingError, match=reason):
            arch35.make_op({"lengths": (2, 4), "heads": 4, "key_heads": 4,
                            "state_selection": declaration})
    with pytest.raises(ModelingError, match="state_slot_branch_not_bound"):
        arch35.make_op({"lengths": (2, 4), "heads": 4, "key_heads": 4,
                        "has_accepted_tokens": True})


def test_old_accepted_count_cannot_exceed_current_q_even_when_index_table_has_spare_columns():
    declaration = {"kind": "exact", "prefix": [0, 1], "index_layout": "speculative",
                   "index_row_stride": 8, "state_indices": [list(range(8))],
                   "accepted_tokens": [7], "state_capacity": 8}
    # Both retained ValidateStateSlots implementations compare acceptance
    # against current seqLen, not table width. Such a block returns before
    # recurrence; its otherwise addressable old slot 6 cannot be priced as
    # a normal invocation of this source branch.
    for module in (arch22, arch35):
        with pytest.raises(ModelingError, match="accepted_values_outside_sequence"):
            module.make_op({"lengths": (1,), "heads": 1, "key_heads": 1,
                           "state_selection": declaration})
    declaration["accepted_tokens"] = [1]
    assert resolve(declaration, (1,))["initial_slots"] == (0,)


def test_symbolic_valid_selection_never_invents_observed_slots_or_acceptance_values():
    declaration = symbolic()
    contract = resolve(declaration, (6, 6, 6, 6))
    assert contract["prefix"] == (0, 6, 12, 18, 24)
    assert not ({"initial_slots", "checkpoint_slots", "accepted_tokens", "state_capacity"} & contract.keys())
    for name in declaration["validity"]:
        omitted = deepcopy(declaration);del omitted["validity"][name]
        with pytest.raises(ModelingError, match="explicit_validity"):
            resolve(omitted, (6,)*4)


def test_b4_q6_tensor_geometry_executes_all_checkpoint_writes_under_declared_validity():
    # The retained GLM ABI proves these extents, not the metadata values.
    kwargs = dict(heads=4, key_heads=4, units=40, ub=196608,
                  planner=arch22.build, constructor=arch22.make_op)
    before = execute((6,)*4, **kwargs)
    op, plan, engine, duration = execute((6,)*4, state_selection=symbolic(), **kwargs)
    assert op.local_shape == (24, 4, 128, 128)
    assert plan.work == before[1].work
    assert duration == pytest.approx(before[3], rel=1e-12)
    # Every current token writes all four per-head FP32 state matrices and
    # BF16 outputs; the accepted tensor's B=4 extent does not change this.
    writes = 24 * 4 * (128*128*4 + 128*2)
    assert engine.serviced["mte3"] == pytest.approx(writes, rel=1e-12)
    assert plan.metadata["kda_recurrent_schedule"]["active_aiv"] == 16
    assert not ({"initial_slots", "accepted_tokens"} & dict(op.metadata)["state_selection"].keys())
