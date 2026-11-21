"""v0.9.19 Track B deep-audit PD handoff & occurrence-DAG regressions.

[Role]
  Pin disaggregated prefill-decode (PD) KV-handoff head/token pairing and
  public baseline occurrence-DAG round-trip invariants extracted from Track B.

[Invariants]
  Each test asserts exact transfer payload bytes, source/destination rank
  head-and-token spans, and exact critical-path duration preservation across
  JSON serialization.

[Timeouts & Locks]
  Bounded single-layer in-memory graphs; no device context, network, sleep or
  filesystem I/O.

[Side-Effect Policy]
  Pure immutable cases; writes only pytest's temporary state.
"""
from __future__ import annotations

import json

import pytest

from mindie_sim import predict_pd
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency,
)
from test_prediction_deployment import pd_case
from test_prediction_system import case as plain_case
from test_prediction_system import linear_work


def _pd_model_case(tokens, num_kv_heads):
    raw = pd_case([{"id": "r", "prompt_tokens": tokens, "output_tokens": 3}])
    raw["model"].update(
        layers=1, hidden_size=64, num_heads=4, num_kv_heads=num_kv_heads,
        head_dim=16, intermediate_size=64, vocab_size=128)
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0, "semantics": "per_pair"},
        "chunk_bytes": 64}
    return raw


def _bind_pd_pool(raw, pool, ranks, tp, cp):
    machine = raw["pools"][pool]["machine"]
    machine["num_devices"] = len(ranks)
    machine["devices"] = [{"rank": rank} for rank in ranks]
    machine["links"][0]["members"] = list(ranks)
    raw["pools"][pool]["runtime"].update(
        tp=tp, pp=1, cp=cp, dcp=cp, cp_plan="dcp" if cp > 1 else "none")


def test_pd_handoff_pairs_replicated_heads_with_every_token_shard():
    raw = _pd_model_case(4, 2)
    _bind_pd_pool(raw, "prefill", (0, 1, 2, 3), tp=4, cp=2)
    _bind_pd_pool(raw, "decode", (4,), tp=1, cp=1)
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]

    # TP4 > 2 KV heads replicates each head; the DCP2 axis must pair each
    # replica with a different token shard so the union covers every
    # (token, head) pair exactly once.
    assert len(transfers) == 4
    assert sum(row["payload_bytes"] for row in transfers) == pytest.approx(512.0)
    assert all(row["payload_bytes"] == pytest.approx(128.0)
               for row in transfers)
    assert {(row["source_rank"], tuple(row["source_token_span"]))
            for row in transfers} == {
        (0, (0, 2)), (1, (0, 2)), (2, (2, 4)), (3, (2, 4))}
    assert {(row["source_rank"], tuple(row["source_heads"]))
            for row in transfers} == {
        (0, (0, 1)), (1, (1, 2)), (2, (0, 1)), (3, (1, 2))}


def test_pd_handoff_weights_destination_shards_by_exact_token_rows():
    raw = _pd_model_case(5, 1)
    _bind_pd_pool(raw, "prefill", (0,), tp=1, cp=1)
    _bind_pd_pool(raw, "decode", (1, 2), tp=2, cp=2)
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]

    assert {(row["destination_rank"], tuple(row["destination_token_span"]),
             row["payload_bytes"]) for row in transfers} == {
        (1, (0, 2), 128.0), (2, (2, 5), 192.0)}
    assert sum(row["payload_bytes"] for row in transfers) == pytest.approx(320.0)


def test_public_baseline_round_trips_the_occurrence_dag():
    raw = plain_case(prompt=4, output=2)
    raw["model"]["layers"] = 1
    baseline = analyze_network_efficiency(
        raw, phase="decode", tokens=1, context=32, sequences=1, scope="model")
    assert "occurrence_dag" in baseline
    public = json.loads(json.dumps(baseline))
    assert len(public["critical_path_nodes"]) == 15
    assert public["critical_path_roofline_ms"] == pytest.approx(0.033888)
    assert public["network_roofline_efficiency"] <= 1.0
