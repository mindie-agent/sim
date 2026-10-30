"""Original host ownership, padded storage and public-engine causal controls."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.paged_mla import build, IMPLEMENTATION
from mindie_sim.prediction.paged_mla_dispatch import Dispatch
from mindie_sim.prediction.paged_mla_host_v2 import source_dispatch
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/paged_mla_host_v2.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['cases'])
def test_source_split_core_v2_matches_original_cpp_and_covers_only_active_kv(case):
    contexts = tuple(case['contexts'])
    actual = source_dispatch(contexts, case['heads'], case['cores'], case['capacity'])
    assert actual['native_host'] == case['source_output']
    partition = Dispatch.parse(actual, contexts, case['heads'])
    for sequence, context in enumerate(contexts):
        spans = sorted((start, count) for rows in partition.owners
                       for seq, start, count in rows if seq == sequence)
        cursor = 0
        for start, count in spans:
            assert start == cursor
            cursor += count
        assert cursor == (context+128)//128
    assert len(partition.owners) <= case['cores']
    assert {seq for rows in partition.owners for seq, _, _ in rows} == set(range(len(contexts)))


def case(batch):
    return json.loads((ROOT / f'examples/cases/kimi-a5-paged-host-20261030/c{batch}.json').read_text())


def op(raw, batch, context):
    graph = target_graphs(raw, scope='body', phase='decode', tokens=1,
                          sequences=batch, context=context)[0]
    return next(o for o in graph if o.recipe == IMPLEMENTATION), graph


@pytest.mark.parametrize('batch,capacity', [(1,1),(2,2),(4,4),(6,8),(8,8),(12,16)])
def test_frontend_keeps_native_capacity_separate_from_active_work(batch, capacity):
    raw = case(batch)
    selected, graph = op(raw, batch, 4096)
    metadata = dict(selected.metadata)
    assert selected.local_shape == (batch,16,1,512)
    assert selected.padded_shape == selected.launch_shape == (capacity,16,1,512)
    assert metadata['attention_tensor_view']['query_shape'] == (capacity,16,1,512)
    assert metadata['attention_tensor_view']['output_shape'] == (16,capacity,1,512)
    assert metadata['paged_mla_geometry']['contexts'] == (4096,)*batch
    assert metadata['paged_mla_geometry']['inactive_sequence_kv_length'] == 0
    # Existing caller head padding owns the complete capacity; no second pack.
    assert not any(o.recipe.startswith('paged_mla_query_pad') for o in graph)
    geometry = Geometry.from_devices(parse_case(raw).machine.dies)
    plan = build(selected, geometry)
    active = deepcopy(raw)
    active['runtime']['kernels']['lowerings']['paged_mla_attention']['options']['native_query_capacity'] = batch
    active_op, _ = op(active, batch, 4096)
    assert build(active_op, geometry).work == plan.work


@pytest.mark.parametrize('batch', [1,6,12])
def test_dynamic_source_policy_equals_explicit_source_metadata_on_public_engine(batch):
    raw = case(batch)
    selected, _ = op(raw, batch, 256)
    metadata = deepcopy(dict(selected.metadata))
    geometry = metadata['paged_mla_geometry']
    geometry['dispatch'] = source_dispatch(geometry['contexts'], geometry['heads'], 32,
                                           geometry['native_query_capacity'])
    explicit = replace(selected, metadata=tuple(metadata.items()))
    dynamic_result = operator_costs(raw, [selected], input_ready_ms=7)['operators'][0]
    explicit_result = operator_costs(raw, [explicit], input_ready_ms=7)['operators'][0]
    assert dynamic_result['duration_ms'] == explicit_result['duration_ms']
    assert dynamic_result['operator']['hbm_bytes'] == explicit_result['operator']['hbm_bytes']
    assert not dynamic_result['service_before_input']
    assert dynamic_result['live_tasks'] == 0
    assert all(abs(v) < 1e-7 for v in dynamic_result['remaining_buffers'].values())
    # A hardware supply change still changes elapsed time; host scores are not rates.
    slower = deepcopy(raw)
    machine = parse_case(raw).machine.to_dict()
    for die in machine['devices']:
        die['rates'] = {k:v/2 for k,v in die['rates'].items()}
        die['compute_peaks'] = {k:v/2 for k,v in die['compute_peaks'].items()}
    slower['machine'] = machine
    assert operator_costs(slower, [selected])['operators'][0]['duration_ms'] > dynamic_result['duration_ms']
