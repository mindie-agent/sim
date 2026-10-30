"""Projected matching records preserve full-cohort identity and fusion contracts."""
from copy import deepcopy

from mindie_sim.structure import export_structure
from mindie_sim.structure_index import export_structure_index, index_digest
from test_prediction_system import case


def test_index_preserves_peer_bound_work_and_detects_projected_record_changes():
    raw = case(dp=2, ep=2)
    raw['model'].update(ffn='moe', num_experts=4, top_k=2, moe_intermediate_size=32)
    args = dict(phase='decode', tokens=1, context=8, scope='body', sequences=[1, 3])
    full = export_structure(raw, **args)
    index = export_structure_index(raw, **args)
    assert index['graph_id'] == full['graph_id']
    assert index['execution'] == full['execution']
    for key in index['record_counts']:
        assert index[key] == full[key]
    assert index_digest(index) == index['index_id']
    changed = deepcopy(index)
    changed['nodes'][0]['physical_work']['hbm_bytes'] += 1
    assert index_digest(changed) != index['index_id']
    other_peer = export_structure_index(raw, **{**args, 'sequences': [1, 5]})
    assert other_peer['graph_id'] != index['graph_id']


def test_source_plan_is_omitted_but_native_contract_and_physical_identity_remain():
    from test_mla_family_selection import mla_case
    raw = mla_case()
    raw['runtime']['kernels']['lowerings'] = {'mla_decode': {
        'implementation': 'vllm_triton_mla_decode_tuple', 'options': {'num_kv_splits': 2}}}
    original = deepcopy(raw)
    args = dict(phase='decode', tokens=1, context=33, scope='body', sequences=2)
    full = export_structure(raw, **args)
    index = export_structure_index(raw, **args)
    assert index['graph_id'] == full['graph_id']
    assert index['source_calls'] and index['record_counts']['source_calls'] == len(full['source_calls'])
    for expected, actual in zip(full['nodes'], index['nodes'], strict=True):
        for field in ('contract', 'contract_scope', 'physical_signature', 'physical_work',
                      'deps', 'lineage', 'implementation', 'structure'):
            assert actual[field] == expected[field]
    assert raw == original


def test_mc2_execution_tree_is_removed_after_full_peer_identity_is_computed():
    from test_mc2_mx_dispatch import case as mc2_case
    raw = mc2_case()
    args = dict(phase='decode', tokens=1, context=128, scope='body', sequences=[1, 3])
    full = export_structure(raw, **args)
    index = export_structure_index(raw, **args)
    assert index['graph_id'] == full['graph_id']
    plans = [node for node in full['nodes'] if 'program_execution' in node['operator']['metadata']]
    assert plans
    indexed = {node['node_id']: node for node in index['nodes']}
    for node in plans:
        projected = indexed[node['node_id']]
        expected = deepcopy(node)
        del expected['operator']['metadata']['program_execution']
        assert projected == expected
    assert index['execution']['replicas'] == [0, 1]
