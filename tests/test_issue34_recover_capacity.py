"""Runtime verify-window state/record capacity across seven linear families."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.speculative_state import (
    MODE_CHECKPOINT, MODE_REPLAY, MODE_SINGLE, record_head_bytes,
    resolve_speculative_query_tokens, speculative_state_plan,
)
from mindie_sim.catalog.cache import request_cache_breakdown
from mindie_sim.schema.case import Software
from mindie_sim.schema.parallel import ParallelSpec
from issue34_helpers import linear_model, view


FAMILIES = ("kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet")
PARALLEL = ParallelSpec(tp=1)
S = 4 * 16 * 16 * 4
R = 4 * record_head_bytes(16, 16, "bf16")


def plan(kind="kda", q=1, recover=False):
    model = linear_model(kind, layers=1)
    return speculative_state_plan(
        model, model.layer_plans[0], PARALLEL, query_tokens=q,
        recover_ssm=recover, capacity_padded=False)


def test_q1_is_identical_under_both_recover_switches_for_all_families():
    for family in FAMILIES:
        off = plan(family, 1, False)
        on = plan(family, 1, True)
        assert off == on
        assert off.mode == MODE_SINGLE and off.state_slots == 1
        assert off.recurrent_bytes == S
        assert off.replay_bytes == 0
        assert off.conv_bytes == pytest.approx(4 * (256 / 4) * 2)


def test_q6_checkpoint_versus_replay_equations():
    off = plan("kda", 6, False)
    on = plan("kda", 6, True)
    assert off.mode == MODE_CHECKPOINT and off.state_slots == 6
    assert off.recurrent_bytes == 6 * S and off.replay_bytes == 0
    assert on.mode == MODE_REPLAY and on.state_slots == 1
    assert on.recurrent_bytes == S
    assert on.replay_bytes == pytest.approx(6 * R)
    # Convolution history is a compact rollback buffer, not q full snapshots.
    assert off.conv_bytes == on.conv_bytes == pytest.approx((4 + 5) * 64 * 2)
    assert off.conv_bytes < 6 * plan("kda", 1).conv_bytes


def test_state_contract_transient_and_explicit_conv_snapshots_are_reported():
    model = linear_model("gdn", layers=1)
    value = speculative_state_plan(
        model, model.layer_plans[0], PARALLEL, query_tokens=4,
        recover_ssm=True, capacity_padded=False,
        state_contract={"conv_snapshot_slots": 4, "transient_bytes": 128,
                        "record_bytes_per_token": 256})
    assert value.replay_bytes == pytest.approx(4 * 256)
    assert value.transient_bytes == 128
    assert value.conv_bytes == pytest.approx(4 * 4 * 64 * 2)


def test_query_width_resolution_uses_view_then_capacity_then_declaration():
    software = Software(metadata=(("draft_proposal_tokens", 5),),
                        mtp_acceptance_length=3.5)
    assert resolve_speculative_query_tokens(software, view=view(q_len=6)) == 6
    assert resolve_speculative_query_tokens(software, tokens_per_seq=4) == 4
    assert resolve_speculative_query_tokens(software) == 6
    assert resolve_speculative_query_tokens(software, phase="prefill") == 1
    assert resolve_speculative_query_tokens(
        Software(metadata=(("draft_mode", "dspark"),
                            ("draft_proposal_tokens", 7),))) == 7


def test_capacity_reserves_the_max_verify_window_not_the_acceptance_mean():
    model = linear_model("kda", layers=2)
    parallel = ParallelSpec(tp=1)
    base = request_cache_breakdown(model, parallel, 512)
    runtime = Software(parallel=parallel, recover_ssm=False,
                       mtp_acceptance_length=3.5,
                       metadata=(("draft_proposal_tokens", 5),))
    active = request_cache_breakdown(model, parallel, 512, runtime=runtime,
                                     query_tokens=6)
    conv_delta = (4 + 5) * 64 * 2 - 4 * 64 * 2
    assert active["total_bytes"] == pytest.approx(
        base["total_bytes"] + 2 * ((6 - 1) * S + conv_delta))
    replay = request_cache_breakdown(
        model, parallel, 512, runtime=replace(runtime, recover_ssm=True),
        query_tokens=6)
    assert replay["total_bytes"] == pytest.approx(
        base["total_bytes"] + 2 * (conv_delta + 6 * R))
    assert replay["by_component"]["replay_records"] == pytest.approx(2 * 6 * R)
