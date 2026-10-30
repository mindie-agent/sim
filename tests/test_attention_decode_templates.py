"""Finite GQA tile reuse preserves causal tails and physical contention.

[Role] Compare actual Engine execution with and without immutable tile reuse.
[Invariants] Every task, service value, buffer lifetime and callback remains
  identical across owner mappings, pipeline slots and ragged causal tails.
[Timeouts & Locks] Small in-memory finite programs; no sleeps or subprocesses.
[Side-Effect Policy] Test-local cache disabling; no device or external state.
"""
import pytest

from mindie_sim.prediction import attention_decode
from mindie_sim.prediction.attention_decode import StagedGQADecode
from test_attention_decode import make_op, run


@pytest.mark.parametrize('slots,splits', ((1, 1), (2, 1), (2, 3)))
def test_tile_templates_preserve_complete_schedule_and_service(monkeypatch, slots, splits):
    op = make_op(((1, 127), (4, 513), (8, 1111)), heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=64, query_tile=7, slots=slots, kv_splits=splits)
    external = {'work': {'memory@r0:s0': 10000, 'vector@r0:s0': 10000},
                'placement': {'vector@r0:s0': (0, 1)}}
    cached = run(op, recipe, ready=0.0003, external=external)
    monkeypatch.setattr(attention_decode, 'lru_cache', lambda **_: lambda function: function)
    uncached = run(op, recipe, ready=0.0003, external=external)
    assert cached[0].summary() == uncached[0].summary()
    assert cached[0].trace == uncached[0].trace
    assert cached[1].time == uncached[1].time
    assert cached[2] == uncached[2]
    assert cached[0].summary()['live_tasks'] == cached[0].summary()['live_programs'] == 0
    assert not any(cached[0].allocated.values())
