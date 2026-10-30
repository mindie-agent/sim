"""Source-host partitions and causal engine controls, without timing fits."""
import json
from collections import defaultdict
from pathlib import Path

import pytest

from mindie_sim.prediction.attention_decode import StagedGQADecode
from mindie_sim.prediction.fai_decode_mapping import partitions
from test_attention_decode import make_op, run


def test_retained_cpp_host_oracle_and_nonempty_complete_kv_coverage():
    fixture = json.loads((Path(__file__).parent / "fixtures/fai_decode_mapping.json").read_text())
    for case in fixture["cases"]:
        requests = tuple((q, length - q) for q, length in case["requests"])
        actual = partitions(requests, case["query_heads"], case["kv_heads"],
                            case["cores"], key_tile=128, query_tile=128)
        observed = sorted((owner, batch, head, start // 4, (end + 3) // 4)
                          for (batch, head), ranges in actual.items()
                          for owner, start, end in ranges)
        assert observed == sorted(tuple(row) for row in case["partitions"])
        for (batch, _), ranges in actual.items():
            cursor = 0
            for owner, start, end in ranges:
                assert 0 <= owner < case["cores"]
                assert start == cursor < end
                cursor = end
            assert cursor == (case["requests"][batch][1] + 127) // 128


def test_ragged_queries_share_owners_without_splitting_the_short_query_twenty_times():
    op = make_op([(4, 1020), (4, 131068)], heads=6, kv_heads=1, dim=8)
    recipe = StagedGQADecode(target_cores=20, kv_mapping="fai_flash_decode")
    outputs = recipe.spec(op).outputs
    assert len(outputs[0].contributions) == 1
    assert len(outputs[1].contributions) == 20
    assert outputs[0].contributions[0].owner == outputs[1].contributions[0].owner == 0
    assert recipe.work(op).matrix == StagedGQADecode().work(op).matrix


def test_source_owner_tail_and_global_merge_gate_execute_on_the_common_engine():
    # Three query tasks share a four-core grid, crossing query boundaries.
    op = make_op([(2, 4094)] * 3, heads=4, kv_heads=1, dim=8)
    recipe = StagedGQADecode(target_cores=4, kv_mapping="fai_flash_decode")
    engine, end, before, _, _ = run(op, recipe, units=4, ready=3)
    assert not before and end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(recipe.work(op).resources["hbm"])
    assert engine.summary()["live_tasks"] == 0
    assert all(v == pytest.approx(0) for v in engine.allocated.values())
    trace = list(engine.trace)
    gate = next(event for event in trace if event["task"].endswith(":all_producers"))
    merges = [event for event in trace if "merge_load" in event["task"]]
    assert merges and all(event["start_ms"] >= gate["end_ms"] for event in merges)


def test_cross_query_budget_avoids_uniform_half_query_tail():
    requests = [(4, 126976)] * 12
    op = make_op(requests, heads=6, kv_heads=1, dim=8)
    source = StagedGQADecode(target_cores=20, kv_mapping="fai_flash_decode").spec(op)
    uniform = StagedGQADecode(target_cores=20).spec(op)
    loads = defaultdict(int)
    for output in source.outputs:
        for part in output.contributions:
            loads[part.owner] += part.tile_count
    other = defaultdict(int)
    for index, part in enumerate(p for out in uniform.outputs for p in out.contributions):
        other[index % 20] += part.tile_count
    assert max(loads.values()) < max(other.values())
    assert sum(loads.values()) == sum(other.values())


@pytest.mark.parametrize('lengths,cores,heads', [
    ([(16,6143),(4,1024),(2,1024),(4,1024)],24,2),
    ([(1,6143),(1,4097),(1,1024)],16,4),
    ([(2,4097),(2,131073),(2,1024)],20,2),
])
def test_final_request_completion_does_not_reproduce_invalid_cpp_terminal_owner(lengths, cores, heads):
    # The pinned host emits one final startB=batch/endB=batch-1 record for
    # these inputs. Preserve physical work, ending after the final real block.
    requests = [(q, k-q) for q, k in lengths]
    result = partitions(requests, heads, 1, cores, key_tile=128, query_tile=128)
    assert len(result) == len(requests)
    for (batch, _), parts in result.items():
        assert 0 <= batch < len(requests)
        assert parts[0][1] == 0
        assert parts[-1][2] == (lengths[batch][1]+127)//128
        assert all(parts[i][2] == parts[i+1][1] for i in range(len(parts)-1))
        assert all(0 <= owner < cores and begin < end for owner, begin, end in parts)


def test_ragged_query_work_uses_selected_owners_even_with_extra_physical_cores():
    op = make_op([(1,1024),(3,1023),(8,12281)], heads=6, kv_heads=2, dim=8)
    recipe = StagedGQADecode(target_cores=8, kv_mapping='fai_flash_decode')
    engine, end, before, _, _ = run(op, recipe, units=12, ready=5)
    assert end.done and not before
    assert engine.serviced['memory@r0:s0'] == pytest.approx(recipe.work(op).resources['hbm'])
    owners = {unit for event in engine.trace for resource, units in event.get('placement', {}).items()
              if resource == 'mte1@r0:s0' for unit in units}
    assert max(owners) < 8
    assert engine.summary()['live_tasks'] == 0


@pytest.mark.parametrize('batch,kv_heads', [(8,2),(9,2),(16,1),(17,1),(48,1),(64,2)])
def test_source_dispatch_switches_to_regular_query_tasks_at_occupancy_boundary(batch, kv_heads):
    from mindie_sim.prediction.fai_decode_mapping import host_partitions, uses_flash_decode
    requests = [(4,126000)] * batch
    mapping = host_partitions(requests, kv_heads*6, kv_heads, 20,
                              key_tile=128, query_tile=64)
    if uses_flash_decode(requests, kv_heads,20):
        assert mapping == partitions(requests,kv_heads*6,kv_heads,20,key_tile=128,query_tile=64)
    else:
        # Independent source task ordinal: one whole query/head task per
        # iteration, striding by launch grid rather than repartitioning KV.
        for ordinal, ((b,h), spans) in enumerate(mapping.items()):
            assert ordinal == b*kv_heads+h
            assert spans == ((ordinal%20,0,(126004+127)//128),)
    assert sum(end-start for spans in mapping.values() for _,start,end in spans) == batch*kv_heads*((126004+127)//128)


def test_regular_host_branch_executes_without_split_workspace_or_merge():
    op=make_op([(4,1020)]*4,heads=4,kv_heads=1,dim=8)
    recipe=StagedGQADecode(target_cores=4,kv_mapping='fai_host')
    spec=recipe.spec(op)
    assert all(len(out.contributions)==1 for out in spec.outputs)
    engine,end,before,_,_=run(op,recipe,units=4,ready=2)
    assert end.done and not before
    assert not any('merge_load' in event['task'] for event in engine.trace)
    assert engine.serviced['memory@r0:s0']==pytest.approx(recipe.work(op).resources['hbm'])
    assert engine.summary()['live_tasks']==0
