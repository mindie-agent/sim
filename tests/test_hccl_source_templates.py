"""Explicit AR/RS/AG source contracts on the ordinary HCCL lower path."""
from __future__ import annotations

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.hccl_aicpu import contract
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_graph import lower
from mindie_sim.prediction.hccl_source_shapes import rs_chunks, slice_sizes
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from test_prediction_system import case


def source(collective, template, ranks, payload, **options):
    return select({"expansion": "AICPU_TS", "template": template,
                   "dtype_bytes": 2, "buffer_mode": "separate", **options},
                  collective=collective, ranks=ranks, payload_bytes=payload)


def work(plan, role):
    return sum(dict(phase.work).get(role, 0) for phase in plan.phases)


def exchanges(plan):
    return [phase for phase in plan.phases if phase.exchange]


def test_ar_mesh_chunk_unseen_tail_keeps_per_phase_direction_and_peer_ledger():
    s = source("allreduce", "mesh_chunk_twoshot", 4, 2006)
    plans = [contract(s, rank) for rank in range(4)]
    plan = plans[0]
    assert slice_sizes(2006, 2, 4) == (502, 502, 502, 500)
    assert work(plan, "hccl_copy") == 2 * 502
    assert work(plan, "hccl_reduce") == 3 * 502
    rs = [phase for phase in exchanges(plan) if phase.name.startswith("rs_step")]
    gather = next(phase for phase in exchanges(plan) if phase.name == "gather")
    assert [phase.native_direction for phase in rs] == ["send", "send", "send"]
    assert gather.native_direction == "read"
    assert gather.exchange[0] == (0, 502, 502, 502)
    assert tuple(gather.exchange[source][0] for source in range(4)) == (0, 502, 502, 500)
    remote = {
        peer: [phase.exchange[0][peer] for phase in rs]
        for peer in range(4) if peer
    }
    assert remote == {1: [166, 168, 168], 2: [168, 168, 166],
                      3: [168, 166, 166]}
    # Every RS step's work is the physical row read plus the incoming column
    # read/write; the matrix itself stays a send matrix.
    for phase in rs:
        assert dict(phase.work)["hbm"] == sum(phase.exchange[0]) + 2 * sum(
            row[0] for row in phase.exchange)


def test_ar_mesh_chunk_zero_tail_phases_remain_integer():
    s = source("allreduce", "mesh_chunk_twoshot", 4, 2)
    plan = contract(s, 3)
    assert slice_sizes(2, 2, 4) == (2, 0, 0, 0)
    assert [dict(phase.work).get("hccl_copy") for phase in plan.phases
            if dict(phase.work).get("hccl_copy")] == []
    gather = next(phase for phase in plan.phases if phase.name == "gather")
    assert not any(gather.exchange[3])


def test_rs_mesh_chunk_ordered_57_plus_7_and_4k_tail():
    s = source("reduce_scatter", "rs_mesh_chunk", 8, 536870912,
               segments=(59768832, 7340032))
    plan = contract(s, 0)
    copies = [dict(phase.work)["hccl_copy"] for phase in plan.phases
              if dict(phase.work).get("hccl_copy")]
    assert copies == [59768832, 59768832, 7340032, 7340032]
    assert work(plan, "hccl_reduce") == 7 * 67108864
    assert rs_chunks(59768832, 2, 8) == (8536064,) * 6 + (8552448,)
    assert rs_chunks(10485760, 2, 8) == (1495040,) * 6 + (1515520,)
    step = next(phase for phase in exchanges(plan) if phase.name == "seg0_step0")
    assert step.native_direction == "send"
    assert all(sum(row) > 0 for row in step.exchange)
    # Every RS phase has one full-shard row/column ledger per rank.
    assert all(sum(row) == 59768832 for row in step.exchange)
    assert all(sum(row[source] for row in step.exchange) == 59768832
               for source in range(8))
    # Two ordered segments, each with its own PreCopy/PostCopy; no reordering.
    names = [phase.name for phase in plan.phases if phase.name.endswith("precopy")
             or phase.name.endswith("postcopy")]
    assert names == ["seg0_precopy", "seg0_postcopy",
                     "seg1_precopy", "seg1_postcopy"]
    assert sum(sum(row) for row in step.exchange) == 8 * 59768832


def test_ag_detour_ordered_50_plus_14_and_explicit_channel_bytes():
    s = source("allgather", "z_axis_detour_allgather", 8, 67108864,
               segments=(52428800, 14680064), level1_weights=(6, 2))
    plan = contract(s, 0)
    copies = [dict(phase.work)["hccl_copy"] for phase in plan.phases
              if dict(phase.work).get("hccl_copy")]
    assert copies == [52428800, 52428800, 14680064, 14680064]
    channels = exchanges(plan)
    assert [phase.name for phase in channels] == [
        "seg0_ch0", "seg0_ch1", "seg0_ch2",
        "seg1_ch0", "seg1_ch1", "seg1_ch2"]
    assert {phase.native_direction for phase in channels} == {"read"}
    assert [phase.exchange[0][1] for phase in channels] == [
        24465408, 20971520, 6991872, 6848512, 5873664, 1957888]
    assert all(sum(row[1] for row in phase.exchange) ==
               sum(phase.exchange[0]) for phase in channels)
    assert [sum(phase.exchange[0][peer] for phase in channels[:3])
            for peer in range(1, 8)] == [52428800] * 7
    assert [sum(phase.exchange[0][peer] for phase in channels[3:])
            for peer in range(1, 8)] == [14680064] * 7
    assert work(plan, "hccl_reduce") == 0


def test_explicit_selection_rejects_fractional_missing_and_ignored_options():
    with pytest.raises(ModelingError):
        source("allgather", "z_axis_detour_allgather", 8, 67108864,
               segments=(8192.75,), level1_weights=(6, 2))
    with pytest.raises(ModelingError):
        source("allgather", "z_axis_detour_allgather", 8, 67108864,
               segments=(8192,), level1_weights=(6.9, 2.2))
    with pytest.raises(ModelingError):
        source("reduce_scatter", "rs_mesh_chunk", 8, 536870912)
    with pytest.raises(ModelingError):
        source("allgather", "z_axis_detour_allgather", 8, 67108864,
               segments=(8192,))
    with pytest.raises(ModelingError):
        source("allreduce", "mesh_chunk_twoshot", 4, 2006, segments=(8192,))
    with pytest.raises(ModelingError):
        source("allreduce", "mesh_chunk_twoshot", 4, 2006,
               level1_weights=(6, 2))
    with pytest.raises(ModelingError):
        contract(source("reduce_scatter", "z_axis_detour_reduce_scatter", 4,
                        268435456, segments=(67108864, 67108864),
                        level1_weights=(6, 2)), 0)
    with pytest.raises(ModelingError):
        select({"template": "mesh_oneshot", "dtype_bytes": 2},
               collective="allreduce", ranks=4, payload_bytes=1024)
    with pytest.raises(ModelingError):
        select({"expansion": "AICPU_TS", "template": "mesh_oneshot",
                "dtype_bytes": 2, "unused_rate": 1},
               collective="allreduce", ranks=4, payload_bytes=1024)


def test_selection_content_identity_includes_only_declared_parameters():
    old = source("allreduce", "mesh_oneshot", 4, 1024)
    assert "segments" not in old.to_dict()
    assert "level1_weights" not in old.to_dict()
    first = source("allgather", "z_axis_detour_allgather", 8, 67108864,
                   segments=(52428800, 14680064), level1_weights=(6, 2))
    second = source("allgather", "z_axis_detour_allgather", 8, 67108864,
                    segments=(52428802, 14680062), level1_weights=(6, 2))
    assert first.to_dict()["segments"] == (52428800, 14680064)
    assert first.to_dict()["level1_weights"] == (6, 2)
    assert first.to_dict() != second.to_dict()


def _run_public(ranks, collective, template, payload, factor, **options):
    raw = _raw_case(ranks, collective, template, **options)
    parsed = parse_case(raw)
    op = OpSpec("collective", collective, "decode", 0, 0, (),
                (1,), (1,), (1,), (1,), link_bytes=payload * factor,
                metadata=(("collective", collective), ("group_axis", "tp")))
    selected = lower((op,), parsed.runtime, "bf16")
    report = graph_costs(raw, selected, isolated=False, input_ready_ms=0,
                         trace_limit=0)
    return selected, report


def test_legacy_public_local_metadata_is_local_not_send():
    selected, report = _run_public(4, "allreduce", "mesh_oneshot", 1024, 1.5)
    assert report["execution"]["live_tasks"] == 0
    directions = {
        dict(item.metadata)["hccl_phase"]["phase"]:
        dict(item.metadata)["hccl_phase"]["native_direction"]
        for item in selected if "hccl_phase" in dict(item.metadata)
    }
    assert directions["allwrite"] == "send"
    for name in ("pre_sync", "input_to_scratch", "post_sync",
                 "local_reduce", "scratch_to_output"):
        assert directions[name] == "local"


def _raw_case(ranks, collective, template, **options):
    raw = case(tp=ranks)
    raw["machine"]["device"]["rates"].update(
        hbm=4e12, hccl_copy=1.5e12, hccl_reduce=1e12)
    raw["model"]["num_heads"] = ranks
    raw["model"]["num_kv_heads"] = max(1, ranks // 2)
    raw["runtime"]["kernels"] = {"hccl": {
        collective: {"expansion": "AICPU_TS", "template": template, **options}}}
    return raw


def test_public_hccl_graph_lower_consumes_new_templates_without_live_state():
    raw = _raw_case(4, "allreduce", "mesh_chunk_twoshot")
    parsed = parse_case(raw)
    op = OpSpec("collective", "allreduce", "decode", 0, 0, (),
                (1,), (1,), (1,), (1,), link_bytes=2006 * 1.5,
                metadata=(("collective", "allreduce"), ("group_axis", "tp")))
    selected = lower((op,), parsed.runtime, "bf16")
    report = graph_costs(raw, selected, isolated=False, input_ready_ms=0,
                         trace_limit=0)
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
    phases = [dict(item.metadata)["hccl_phase"] for item in selected
              if "hccl_phase" in dict(item.metadata)]
    assert any(phase["phase"] == "gather" and
               phase["native_direction"] == "read" for phase in phases)
    assert any(phase["phase"].startswith("rs_step") and
               phase["native_direction"] == "send" for phase in phases)
    network = [item for item in selected if item.family == "alltoallv"]
    assert network
    # Physical network traffic is the sum of the plan's send matrices once.
    assert sum(sum(sum(row) for row in dict(item.metadata)["peer_traffic"]["payload_bytes"])
               for item in network) == 6 * 2006


@pytest.mark.parametrize("ranks,collective,template,payload,factor,options", [
    (8, "allgather", "z_axis_detour_allgather", 67108864, 7,
     {"segments": (52428800, 14680064), "level1_weights": (6, 2)}),
    (8, "reduce_scatter", "rs_mesh_chunk", 536870912, 7.0 / 8.0,
     {"segments": (59768832, 7340032)}),
])
def test_public_lower_accepts_segmented_ag_and_rs(ranks, collective, template,
                                                 payload, factor, options):
    selected, report = _run_public(ranks, collective, template, payload,
                                   factor, **options)
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
    network = [item for item in selected if item.family == "alltoallv"]
    assert network
    phase_dirs = [
        dict(item.metadata)["hccl_phase"]["native_direction"]
        for item in selected
        if "hccl_phase" in dict(item.metadata)
        and dict(item.metadata)["hccl_phase"]["phase"].startswith("seg")
        and dict(item.metadata)["hccl_phase"]["native_direction"] in {"read", "send"}]
    if collective == "allgather":
        assert len(network) == 6
        assert phase_dirs[:6] == ["read"] * 6
    else:
        assert len(network) == 14
        assert phase_dirs == ["send"] * 14
