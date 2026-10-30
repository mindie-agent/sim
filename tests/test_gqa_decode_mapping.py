"""KV occupancy mapping conserves causal work and retains real split merges."""
from mindie_sim.prediction.attention_decode import StagedGQADecode
from test_attention_decode import make_op, run, expected_traffic


def test_query_parallelism_consumes_core_budget_before_kv_partitioning():
    recipe = StagedGQADecode(target_cores=20)
    for batch, expected_splits in ((1, 20), (2, 10), (4, 5), (20, 1), (32, 1)):
        op = make_op([(4, 126976)] * batch, heads=6, kv_heads=1, dim=256)
        spec = recipe.spec(op)
        assert spec.kv_splits == expected_splits
        assert sum(len(output.contributions) for output in spec.outputs) == max(20, batch)
        assert recipe.work(op).matrix == StagedGQADecode().work(op).matrix


def test_short_and_ragged_contexts_never_create_empty_split_partials():
    requests = [(1, 0), (4, 512), (0, 126000)]
    op = make_op(requests, heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=128, query_tile=64, target_cores=20)
    spec = recipe.spec(op)
    assert [len(out.contributions) for out in spec.outputs] == [1, 1, 5, 5]
    assert all(part.tile_count > 0 for out in spec.outputs for part in out.contributions)
    expected = expected_traffic(requests, heads=4, kv_heads=2, dim=8,
                                key_tile=128, query_tile=64, kv_splits=5)
    actual = recipe.work(op)
    assert actual.resources["hbm"] == expected["hbm"]
    assert sum(actual.matrix.values()) == expected["flops"]
    assert actual.resources["vector"] == expected["vector"]
    assert expected["partial_read"] == expected["partial_write"] > 0


def test_more_partitions_use_actual_owners_and_charge_merge_before_completion():
    op = make_op([(1, 4096)], heads=1, kv_heads=1, dim=8)
    fixed = StagedGQADecode(key_tile=128)
    mapped = StagedGQADecode(key_tile=128, target_cores=4)
    slow = run(op, fixed, units=4, ready=7)
    fast = run(op, mapped, units=4, ready=7)
    assert fast[1].time < slow[1].time
    assert mapped.work(op).resources["hbm"] > fixed.work(op).resources["hbm"]
    for result in (slow, fast):
        engine = result[0]
        assert engine.now >= 7
        assert all(event["start_ms"] >= 7 for event in engine.trace if event.get("demand"))
        assert engine.summary()["live_tasks"] == 0
