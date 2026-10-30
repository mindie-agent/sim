"""Independent source layout and phase-ledger controls for MC2 fullmesh."""
import math

import pytest

from mindie_sim.prediction.operators.mc2_fullmesh import (
    combine_layout, combine_plan, dispatch_layout, dispatch_plan)


def align(value, block):
    return ((value + block - 1) // block) * block


@pytest.mark.parametrize("width", [1, 31, 32, 127, 128, 255, 256, 448, 480, 481, 1536, 7168])
def test_per_row_packet_source_layout_is_independent_of_assignment_count(width):
    declarations = [
        ("none", "bf16", "bf16", 0, align(width * 2, 32), 0),
        ("none", "fp8_e4m3", "fp8_e4m3", 64, align(width, 32), 64),
        ("per_token", "bf16", "int8", 0, align(width, 32), 4),
        ("mx", "bf16", "fp8_e4m3", 0, align(width, 256), align((width + 31) // 32, 2)),
        ("per_group", "fp16", "fp8_e5m2", 0, align(width, 128), ((width + 127) // 128) * 4),
    ]
    for mode, incoming, outgoing, input_scales, scale_offset, scales in declarations:
        row = dispatch_layout(width, input_dtype=incoming, payload_dtype=outgoing,
                              quant_mode=mode, input_scale_bytes=input_scales)
        triple = align(scale_offset + scales, 32)
        expected_blocks = (triple + 32 + 479) // 480
        assert row.triple_offset_bytes == triple
        assert row.scale_bytes == scales
        assert row.packet_bytes == expected_blocks * 512
        assert row.data_region_bytes + row.flag_bytes == row.packet_bytes
        payload_bytes = width * (2 if outgoing == "bf16" else 1)
        assert row.materialized_bytes == payload_bytes + scales + 12


def test_fractional_expected_rows_scale_whole_row_packets_without_fractional_rounding():
    row = dispatch_layout(1536, payload_dtype="fp8_e4m3", quant_mode="mx")
    plan = dispatch_plan(row, source_tokens=1, top_k=8, source_assignments=.1,
                         received_assignments=1.75, experts=384, ep=32, aiv_count=64)
    phases = {p.name: p for p in plan.phases}
    transfer = dict(phases["payload_exchange"].work)
    assert transfer["mte3"] == row.packet_bytes * .1
    assert transfer["hbm"] == row.packet_bytes * 1.75
    assert transfer["mte3"] != math.ceil((row.triple_offset_bytes + 32) * .1 / 480) * 512
    assert plan.count_basis == "expected_assignments"
    quant = dict(phases["send_read_quantize"].work)
    # Six whole 256-value blocks, with per-row register constants and 48 scales.
    assert quant["vector"] == pytest.approx(.1 * (1024 + 6 * 1784 + 48 * 13))


@pytest.mark.parametrize("aiv_count,experts,ep,expected_count_cores", [
    (64, 384, 32, 8), (40, 288, 16, 8), (4, 16, 4, 1), (2, 2, 2, 1),
])
def test_count_and_payload_partitions_parallel_then_joint_receive(aiv_count, experts, ep, expected_count_cores):
    row = dispatch_layout(4096, payload_dtype="int8", quant_mode="per_token")
    plan = dispatch_plan(row, source_tokens=2, top_k=8, source_assignments=16,
                         received_assignments=12, experts=experts, ep=ep, aiv_count=aiv_count)
    phases = {p.name: p for p in plan.phases}
    count, payload = phases["count_compute"], phases["send_read_quantize"]
    assert len(count.owner_domain) == expected_count_cores
    assert not set(count.owner_domain) & set(payload.owner_domain)
    assert set(count.owner_domain) | set(payload.owner_domain) == set(range(aiv_count))
    assert count.after == payload.after == ("routing_metadata",)
    assert phases["count_status_exchange"].after == ("count_compute",)
    assert phases["payload_exchange"].after == ("frame_payload",)
    assert phases["receive_flags"].after == ("payload_exchange", "receive_metadata")
    assert phases["receive_counts"].after == ("count_status_exchange",)
    assert dict(phases["count_status_exchange"].details)["records_per_source"] == experts
    assert dict(phases["count_status_exchange"].work) == {"mte3": experts * 32, "hbm": experts * 32}
    steps = {name: dict(values) for name, values in payload.steps}
    assert steps["quantize"]["vector"] == 16 * (7 * 4096 - 1)
    assert steps["assignment_prefix_scan"]["vector"] == sum(5 * i - 1 for i in range(8, 16))
    assert dict(payload.work)["vector"] == sum(v.get("vector", 0) for v in steps.values())


def test_local_receive_copy_strips_flags_but_keeps_scales_and_route_triple():
    row = dispatch_layout(481, payload_dtype="fp8_e4m3", quant_mode="per_group")
    plan = dispatch_plan(row, source_tokens=4, top_k=8, source_assignments=32,
                         received_assignments=7, experts=288, ep=16, aiv_count=40)
    phases = {p.name: p for p in plan.phases}
    reads = dict(phases["receive_local_copy"].work)
    assert reads["mte2_aiv"] == 7 * row.blocks * 480
    assert reads["mte3"] == 7 * (481 + 4 * 4 + 12)
    assert reads["hbm"] == reads["mte2_aiv"] + reads["mte3"]
    assert dict(phases["receive_flags"].work)["mte2_aiv"] == 7 * row.blocks * 32
    assert dict(phases["clear_data_flags"].work)["mte3"] == 7 * row.blocks * 32


def test_return_uses_distinct_row_frame_and_retains_weighted_reduction_boundary():
    row = combine_layout(240)
    dispatch_row = dispatch_layout(240)
    # The dispatch triple needs a second frame; the unquantized return does not.
    assert row.packet_bytes == 512 and dispatch_row.packet_bytes == 1024
    plan = combine_plan(row, source_assignments=3, received_assignments=8,
                        experts=256, ep=16, aiv_count=40)
    phases = {p.name: p for p in plan.phases}
    assert dict(phases["return_read_frame"].work)["mte2_aiv"] == 3 * (480 + 12)
    assert dict(phases["payload_exchange"].work) == {"mte3": 3 * 512, "hbm": 8 * 512}
    assert dict(phases["receive_slots"].work)["mte2_aiv"] == 8 * 480
    assert "weighted sum arithmetic" in plan.retained_generic_work
    assert "final output write" in plan.retained_generic_work


def test_optional_expert_weight_shares_reserved_tail_but_has_separate_receive_dma():
    plain = dispatch_layout(1536, payload_dtype="fp8_e4m3", quant_mode="mx")
    weighted = dispatch_layout(1536, payload_dtype="fp8_e4m3", quant_mode="mx",
                               has_expert_scales=True)
    assert weighted.packet_bytes == plain.packet_bytes
    assert weighted.materialized_bytes == plain.materialized_bytes + 4
    plan = dispatch_plan(weighted, source_tokens=1, top_k=8, source_assignments=8,
                         received_assignments=4, experts=384, ep=32, aiv_count=64)
    local = next(p for p in plan.phases if p.name == "receive_local_copy")
    assert dict(local.work)["mte2_aiv"] == 4 * (weighted.data_region_bytes + 4)
    assert dict(local.work)["mte3"] == 4 * weighted.materialized_bytes
