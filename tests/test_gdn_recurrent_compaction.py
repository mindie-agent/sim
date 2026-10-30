"""Serial head compaction against detailed real-Engine schedules."""
import pytest

from mindie_sim.prediction.gdn_recurrent import head_ownership, head_runs
from test_gdn_recurrent_source import run


@pytest.mark.parametrize("q", (1, 2, 3, 4, 5, 8))
@pytest.mark.parametrize("dk,dv", ((16, 16), (129, 97), (256, 129)))
def test_uniform_one_owner_repetition_is_time_homogeneous(q, dk, dv):
    detailed = run((q, q), heads=3, dk=dk, dv=dv, units=1, compact=False)
    compact = run((q, q), heads=3, dk=dk, dv=dv, units=1, compact=True)
    assert compact[2] == pytest.approx(detailed[2], rel=1e-9, abs=1e-12)
    assert compact[1].serviced == pytest.approx(detailed[1].serviced, rel=1e-11)
    assert compact[1].completed_count < detailed[1].completed_count / 4
    assert compact[1].peak_unit_allocated == detailed[1].peak_unit_allocated


@pytest.mark.parametrize("lengths,heads,units", (
    ((1, 2, 3, 4, 5, 8), 3, 4), ((4, 4, 3, 3, 4), 7, 4),
    ((8, 8, 1, 1, 8), 9, 20), ((3, 3, 3), 7, 4),
))
def test_runs_keep_ragged_lengths_and_unequal_owner_assignments(lengths, heads, units):
    ownership = head_ownership(lengths, heads, min(units, len(lengths) * heads))
    original = [q for q in lengths for _ in range(heads)]
    offset = 0
    for segments in ownership:
        count = sum(heads_in_segment for _, heads_in_segment in segments)
        cycles = tuple(head_runs(lengths, segments))
        expanded = [q for q, repetitions in cycles for _ in range(repetitions)]
        assert expanded == original[offset:offset + count]
        assert all(a[0] != b[0] for a, b in zip(cycles, cycles[1:]))
        offset += count
    assert offset == len(lengths) * heads


@pytest.mark.parametrize("limited,contender", (
    ("hbm", {"hbm": 10000}),
    ("mte2_aiv", {"hbm": 10000, "mte2_aiv": 10000}),
    ("mte3", {"hbm": 10000, "mte3": 10000}),
    ("vector", {"vector": 10000}),
))
@pytest.mark.parametrize("lengths,heads,dk,dv,units", (
    ((4, 4, 3, 3, 4), 3, 129, 97, 4),
    ((1, 1, 2, 2, 5), 7, 128, 65, 4),
    ((8, 8, 8), 3, 32, 257, 4),
))
def test_compaction_preserves_service_with_copy_and_compute_competitors(limited, contender,
                                                                      lengths, heads, dk, dv, units):
    kwargs = dict(heads=heads, dk=dk, dv=dv, units=units,
                  rates={limited: 1e6}, contender=contender)
    detailed = run(lengths, compact=False, **kwargs)
    compact = run(lengths, compact=True, **kwargs)
    assert compact[1].serviced == pytest.approx(detailed[1].serviced, rel=1e-10)
    for plan, engine, duration in (detailed, compact):
        for role, work in plan.work.resources.items():
            assert engine.serviced[role] == pytest.approx(work + contender.get(role, 0), rel=1e-10)
        assert duration >= plan.work.resources[limited] / 1e6 - 1e-10
        for event in engine.trace:
            if event["task"] == "external":
                continue
            for role in ("mte2_aiv", "mte3", "vector"):
                if event["demand"].get(role):
                    assert len(event["placement"][role]) == 1
                    assert event["placement"][role][0] < units


def test_prepared_output_exposes_approximation_without_replacing_source_contract():
    from dataclasses import replace
    from mindie_sim.prediction.gdn_recurrent import IMPLEMENTATION
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.prediction.frontend import Frontend
    from test_gdn_decode_family import source_case

    parsed, views, _ = source_case(lengths=(4,) * 8, keys=1, heads=6, units=4)
    binding = {"implementation": "vllm_ascend_gdn_decode",
               "options": {"recurrent_implementation": "arch35_fp32"}}
    parsed = replace(parsed, runtime=replace(parsed.runtime,
                     kernels={"lowerings": {"gdn_decode": binding}}))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode")
    original = next(op for op in ops if op.recipe == IMPLEMENTATION)
    original = replace(original, metadata=tuple({**dict(original.metadata),
                       "program_execution": {"source_abi": "normalized_lengths"}}.items()))
    result = operator_costs(parsed, (original,))["operators"][0]["operator"]
    for key, value in original.metadata:
        if key != "program_execution":
            assert result["metadata"][key] == value
    assert result["metadata"]["program_execution"]["source_abi"] == "normalized_lengths"
    assert dict(original.metadata)["program_execution"] == {"source_abi": "normalized_lengths"}
    assert result["source_boundary"] == original.source_boundary
    schedule = result["metadata"]["program_execution"]["recurrent_head_schedule"]
    assert schedule["logical_heads"] == 48 and schedule["resource_cycles"] == 4
    assert schedule["owner_heads"] == (12,) * 4
    assert schedule["contention_approximation"] == "serial_cycle_fluid_rephasing"
