"""Streamed complete identity retains every selected candidate across replicas."""
from copy import deepcopy

import pytest

from mindie_sim.structure import export_structure
from mindie_sim.structure_selection import select_structure_targets
from test_prediction_system import case


@pytest.mark.parametrize('sequences', ([1, 3], [2, 2]))
def test_full_identity_is_preserved_without_retaining_unselected_replicas(sequences):
    raw = case(dp=2, ep=2)
    raw['model'].update(ffn='moe', num_experts=4, top_k=2, moe_intermediate_size=32)
    args = dict(phase='decode', tokens=1, context=8, scope='body', sequences=sequences)
    full = export_structure(raw, **args)
    all_targets = {row['node_id']: row for row in full['nodes']}
    all_targets.update((row['subgraph_id'], row) for row in full['subgraphs'])
    wanted = {key for key in all_targets if key.startswith('rank:0/')}
    wanted.add('rank:1/not-a-node')
    reduced = select_structure_targets(raw, wanted, **args)
    assert reduced['graph_id'] == full['graph_id']
    assert reduced['execution'] == full['execution']
    assert reduced['targets'] == {key: value for key, value in all_targets.items() if key in wanted}
    assert reduced['record_counts'] == {key: len(full[key]) for key in reduced['record_counts']}
    # Work on an unselected DP participant still changes full identity. It is
    # never hidden by choosing only rank0 candidates for a local observation.
    altered = select_structure_targets(raw, wanted, **{**args, 'sequences': [sequences[0], 5]})
    assert altered['graph_id'] != reduced['graph_id']


def test_source_native_fusion_and_calls_preserve_full_hash_and_boundaries():
    from test_mla_family_selection import mla_case
    raw = mla_case()
    raw['runtime']['kernels']['lowerings'] = {'mla_decode': {
        'implementation': 'vllm_triton_mla_decode_tuple', 'options': {'num_kv_splits': 2}}}
    original = deepcopy(raw)
    args = dict(phase='decode', tokens=1, context=33, scope='body', sequences=2)
    full = export_structure(raw, **args)
    wanted = {row['node_id'] for row in full['nodes']}
    wanted.update(row['subgraph_id'] for row in full['subgraphs'])
    selected = select_structure_targets(raw, wanted, **args)
    assert selected['graph_id'] == full['graph_id']
    assert selected['record_counts']['source_calls'] == len(full['source_calls']) > 0
    assert selected['record_counts']['resource_operators'] == len(full['resource_operators'])
    assert raw == original
