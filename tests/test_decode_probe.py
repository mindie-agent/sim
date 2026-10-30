"""Decode progress, DP residency and graph padding are independent axes."""
from copy import deepcopy

import pytest

from mindie_sim import predict
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.decode_probe import decode_windows, probe_capacity
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case


def test_sampled_verification_windows_follow_output_progress_without_acceptance_scaling():
    windows = decode_windows(126000, 3000, query_tokens=4)
    assert [row["context"] for row in windows] == [126000, 127498, 128996]
    assert [row["kv_length"] for row in windows] == [126004, 127502, 129000]
    assert all(row["tokens"] == 4 for row in windows)
    assert [row["context"] for row in decode_windows(126000, 3000)] == [126000, 127499, 128999]
    observed = decode_windows(131201, 64, query_tokens=6, output_offsets=[0, 11, 37, 58])
    assert [row["kv_length"] for row in observed] == [131207, 131218, 131244, 131265]


def test_ep_idle_ranks_keep_weights_without_inventing_kv_or_global_batch():
    raw = case(dp=4, ep=4, prompt=126000)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"]["max_model_len"] = 129000
    one = probe_capacity(raw, context=126000, tokens=1, sequences=[1, 0, 0, 0])
    full = probe_capacity(raw, context=126000, tokens=1, sequences=1)
    assert one["sequences_in_cohort"] == 1 and full["sequences_in_cohort"] == 4
    assert [row["cache_bytes"] for row in one["ranks"]] == [one["ranks"][0]["cache_bytes"], 0, 0, 0]
    assert all(row["weights_bytes"] > 0 for row in one["ranks"])
    assert sum(row["cache_bytes"] for row in full["ranks"]) == 4 * one["ranks"][0]["cache_bytes"]
    independent = probe_capacity(case(dp=4), context=64, tokens=1, sequences=2)
    assert independent["sequences_in_cohort"] == 2
    assert independent["unprobed_replicas"] == [1, 2, 3]


def test_padded_decode_workspace_uses_actual_rows_and_cache_uses_valid_sequences():
    raw = case(prompt=126000)
    raw["runtime"].update(max_model_len=129000, max_num_batched_tokens=8192,
                          graph_mode="auto", graph_capture_mode="full", graph_capture_keys=[16])
    first = probe_capacity(raw, context=126000, tokens=1, sequences=3)
    before = deepcopy(raw)
    raw["workload"]["requests"][0]["prompt_tokens"] = 1
    second = probe_capacity(raw, context=126000, tokens=1, sequences=3)
    assert first == second
    assert first["padded_query_tokens_by_replica"] == {0: 16}
    memory = Memory(parse_case(before), workspace_tokens=16)
    assert first["ranks"][0]["workspace_bytes"] == memory.workspace[0]
    assert first["ranks"][0]["cache_bytes"] == 3 * memory.cache_bytes(126001)[0]
    next_page = probe_capacity(raw, context=126128, tokens=1, sequences=3)
    assert next_page["ranks"][0]["cache_bytes"] > first["ranks"][0]["cache_bytes"]


def test_long_context_capacity_failure_does_not_gate_analytical_body_execution():
    raw = case()
    raw["runtime"].update(max_model_len=129000, max_num_seqs=64)
    raw["machine"]["device"]["memory_bytes"] = 1000000
    baseline = predict(raw, scope="body", context=126000, sequences=2)
    assert baseline["status"] == "modeled" and baseline["body_forward_ms"] > 0
    assert baseline["capacity"]["status"] == "outside_declared_capacity"
    assert any(row["constraint"] == "resident_memory" for row in baseline["capacity"]["violations"])
    raw["machine"]["device"]["memory_bytes"] = 1024 ** 3
    larger = predict(raw, scope="body", context=126000, sequences=2)
    assert larger["capacity"]["status"] == "within_declared_capacity"
    assert larger["body_forward_ms"] == baseline["body_forward_ms"]


def test_capacity_retains_pipeline_stage_and_replica_limits_without_truncating_windows():
    raw = case(pp=2, dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"].update(max_model_len=128000, max_num_seqs=2, max_num_batched_tokens=8)
    result = probe_capacity(raw, context=128996, tokens=4, sequences=[3, 0])
    assert result["kv_length"] == 129000
    assert len(result["ranks"]) == 4
    constraints = {row["constraint"] for row in result["violations"]}
    assert {"max_model_len", "max_num_seqs", "max_num_batched_tokens"} <= constraints
    for row in result["ranks"]:
        assert row["resident_bytes"] == pytest.approx(
            row["weights_bytes"] + row["workspace_bytes"] + row["reserved_bytes"] + row["cache_bytes"])
        assert row["headroom_bytes"] == row["cache_capacity_bytes"] - row["cache_bytes"]


def test_valid_requests_capture_bucket_and_sp_padding_are_distinct_public_geometry():
    from mindie_sim.prediction.sequence_parallel import DECLARATION
    raw = case(tp=16, ep=16)
    raw['model'].update(hidden_size=128, num_heads=16, num_kv_heads=2,
                        intermediate_size=256, ffn='moe', num_experts=16,
                        top_k=2, moe_intermediate_size=64,
                        layer_plans=[{'index':i, 'attention':'gqa', 'ffn':'moe',
                                      'num_heads':16, 'num_kv_heads':2, 'head_dim':8,
                                      'num_experts':16, 'top_k':2, 'moe_intermediate_size':64,
                                      'residual_block_size':2} for i in range(2)])
    raw['runtime'].update(graph_mode='auto', graph_capture_mode='full',
                           graph_capture_keys=[8], routing={'policy':'balanced'},
                           kernels={'sequence_parallel':dict(DECLARATION,padding='right_to_tp')})
    report = predict(raw, scope='body', tokens=1, sequences=6, context=32)
    geometry = report['geometry']
    assert geometry['valid_query_tokens_by_replica'] == {0:6}
    assert geometry['graph_query_tokens_by_replica'] == {0:8}
    assert geometry['padded_query_tokens_by_replica'] == {0:16}
    assert report['replay_scope_by_replica'] == {0:'full'}
    capacity = report['capacity']
    assert capacity['graph_query_tokens_by_replica'] == {0:8}
    assert capacity['ranks'][0]['cache_bytes'] == 6 * capacity['cache_bytes_per_sequence_stage'][0]
