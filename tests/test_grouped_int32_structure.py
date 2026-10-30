"""Source INT32 caller retains real fused owners through public structure export."""
import json
from pathlib import Path

from mindie_sim.structure import export_structure


def test_full_qwen35_source_complete_structure_keeps_fused_activation_owners():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / 'examples/cases/high-concurrency-20261030/qwen35-hccl-conditional/qwen35-35b-a5-tp4-c128-hccl-source.json').read_text())
    kernels = raw['runtime']['kernels']
    kernels['native_moe_protocol'] = {'kind': 'arch35_fullmesh', 'max_inflight_per_rank': 4,
                                      'dispatch_quant': 'per_token_int8'}
    kernels['moe_token_partition'] = {'kind': 'vllm_ascend_mc2_tp_slice',
                                     'gather_output_materialization': 'per_rank_copy'}
    report = export_structure(raw, scope='body', phase='decode', tokens=4,
                              sequences=128, context=4096, replicas=(0,))
    nodes = report['nodes']
    fused = [node for node in nodes
             if node['operator']['recipe'] == 'grouped_int32_dequant_swiglu']
    assert len(fused) == 40
    for node in fused:
        origins = node['operator']['metadata']['fused_origin_scopes']
        assert len(origins) == 2
        assert {origin['stage'] for origin in origins} == {node['stage']}
        assert {origin['layer'] for origin in origins} == {node['layer']}
        assert {origin['block'] for origin in origins} == {node['block']} == {'ffn'}
        containing = [group for group in report['subgraphs']
                      if node['node_id'] in group['node_ids']]
        assert containing
        for group in containing:
            assert {'replica': 0, 'stage': node['stage'], 'layer': node['layer'],
                    'block': 'ffn'} in group['owner_scopes']
