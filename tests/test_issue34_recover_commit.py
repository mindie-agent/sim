"""Acceptance-bound RecoverSSM commit publication and integer advancement."""
from __future__ import annotations

import pytest

from mindie_sim.prediction.recover_ssm_commit import (
    bind_declared_body_commits, bind_recover_ssm_commits,
)
from mindie_sim.prediction.recover_ssm_work import commit_tokens_for_view
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from issue34_helpers import view


def intent(owner="layer0", version="v0", requests=("r0", "r1")):
    return OpSpec(
        "verify", "recurrent_scan", "decode", 0, 0, requests,
        (2,), (2,), (2,), (2,), vector_ops=100.0, dtype="fp32",
        deps=("x",),
        metadata=(
            ("recover_ssm_intent", True), ("state_owner", owner),
            ("input_state_version", version), ("heads", 2), ("dk", 4),
            ("dv", 8), ("state_dtype", "fp32"),
            ("record_kg_dtype", "bf16"), ("record_bytes_per_token", 48.0),
        ))


def test_integer_commits_sum_and_publish_after_acceptance():
    ops = (intent(),)
    views = (view("r0", 6, advanced=3), view("r1", 6, advanced=4))
    bound = bind_recover_ssm_commits(
        ops, views=views, acceptance_events={"r0": "accept0", "r1": "accept1"},
        state_bindings={"layer0": "base0"}, runtime=None)
    commit = bound[0]
    assert commit.deps == ("x", "accept0", "accept1", "base0")
    metadata = dict(commit.metadata)
    assert metadata["recover_ssm_stage"] == "commit"
    assert metadata["committed_tokens"] == (("r0", 3), ("r1", 4))
    assert metadata["state_slots"] == 1
    assert metadata["input_state_version"] == "v0"
    assert metadata["published_state_version"] == "v0"
    assert metadata["acceptance_boundary"] == "sampling_acceptance"
    state = 2 * 4 * 8 * 4
    assert commit.vector_ops == pytest.approx(4 * 7 * 2 * 4 * 8)
    assert commit.mte2_bytes == pytest.approx(2 * state + 7 * 48.0)
    assert commit.mte3_bytes == pytest.approx(2 * state)
    assert commit.hbm_bytes == pytest.approx(commit.mte2_bytes + commit.mte3_bytes)


def test_missing_acceptance_duplicate_owner_and_ambiguous_mean_fail():
    views = (view("r0", 6, advanced=3),)
    with pytest.raises(ModelingError, match="requires_acceptance_event"):
        bind_recover_ssm_commits((intent(requests=("r0",)),), views=views,
                                 acceptance_events={}, state_bindings={},
                                 runtime=None)
    with pytest.raises(ModelingError, match="duplicate_commit"):
        bind_recover_ssm_commits((intent(), intent()), views=views,
                                 acceptance_events={"r0": "a"},
                                 state_bindings={}, runtime=None)
    ambiguous = view("r0", 6)
    with pytest.raises(ModelingError, match="resolved_integer_advancement"):
        commit_tokens_for_view(ambiguous, Software(mtp_acceptance_length=3.5))
    assert commit_tokens_for_view(ambiguous, Software()) == 6
    assert commit_tokens_for_view(view("r0", 1), Software(
        mtp_acceptance_length=3.5)) == 1


def test_pending_commit_is_rebound_without_duplication():
    pending = OpSpec(
        "commit", "elementwise", "decode", 0, 0, ("r0",), (1,), (1,), (1,), (1,),
        vector_ops=4 * 3 * 2 * 4 * 8, hbm_bytes=100.0, mte2_bytes=60.0,
        mte3_bytes=40.0, deps=("verify",),
        metadata=(("recover_ssm_pending_acceptance", True),
                  ("state_owner", "layer0"), ("input_state_version", "v1"),
                  ("heads", 2), ("dk", 4), ("dv", 8),
                  ("record_bytes_per_token", 48.0)))
    bound = bind_recover_ssm_commits(
        (pending,), views=(view("r0", 6, advanced=3),),
        acceptance_events={"r0": "accept"}, state_bindings={}, runtime=None)
    assert len(bound) == 1
    assert bound[0].deps == ("verify", "accept")
    assert dict(bound[0].metadata)["published_state_version"] == "v1"


def test_body_boundary_requires_and_records_declared_advancement():
    ops = (intent(requests=("r0",)),)
    marked = bind_declared_body_commits(
        ops, views=(view("r0", 6, advanced=2),))
    assert dict(marked[0].metadata)["acceptance_boundary"] == "declared_body_input"
    with pytest.raises(ModelingError, match="resolved_integer_advancement"):
        bind_declared_body_commits(ops, views=(view("r0", 6),))
