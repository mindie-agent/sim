"""Captured MLA rows and caller-only MC2 allocations coexist across 78 layers."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT/'examples/cases/high-concurrency-20261030/glm78-capture-rows'


def summarize(raw, rows):
    ops=target_graphs(raw,phase='decode',tokens=6,sequences=rows//6,
                       context=4096,scope='body',apply_body_defaults=False)[0]
    prologs=[dict(o.metadata)['fusion_tensor_view'] for o in ops
             if dict(o.metadata).get('fusion_role')=='mla_prolog']
    ledgers=[dict(o.metadata)['routing_realization'] for o in ops
             if o.recipe=='balanced_grouped_gate_up']
    return ops,prologs,ledgers


@pytest.mark.parametrize('name,rows',[
    ('a3_glm53_w4a8_tp16_tpot30_c2',12),
    ('a3_glm53_w4a8_tp16_tpot50_c2',12),
    ('a3_glm53_w4a8_tp16_tpot88_c8',48),
])
def test_full_model_capture_contract_preserves_original_and_allocated_rows(name,rows):
    raw=json.loads((CASES/(name+'.json')).read_text())
    ops,prologs,ledgers=summarize(raw,rows)
    assert len(prologs)==78
    assert Counter(p['query_norm_output'] for p in prologs)=={False:57,True:21}
    assert all(p['tokens']==rows for p in prologs)
    assert all(p['token_scale_shape']==(rows,1) for p in prologs)
    assert len(ledgers)==75
    assert all(sum(d['global_expert_rows'])==rows*8 for d in ledgers)
    assert all(o.local_shape[0]==rows for o in ops if o.family=='router')
    assert all(o.local_shape[0]==rows for o in ops if '.shared.gate_up' in o.op_id)
    assert sum(o.family=='allgather' for o in ops)==75
    allocated=(rows+15)//16*16
    assert all(o.local_shape[0]==allocated for o in ops if o.family=='allgather')
    concats=[o for o in ops if dict(o.metadata).get('native_kind')=='ConcatD']
    assert len(concats)==(150 if rows==12 else 0)
    if rows==12:
        assert Counter(tuple(dict(o.metadata)['input_shapes']) for o in concats)=={
            ((12,6144),(4,6144)):75,((12,256),(4,256)):75}
        assert sum(o.recipe=='mc2_tp_output_view' for o in ops)==75
        assert all(not any(row) for o in ops if o.family=='alltoallv'
                   and not dict(o.metadata).get('source_token_mask',{}).get('basis','').startswith('source_')
                   and o.op_id.endswith('.dispatch')
                   for row in dict(o.metadata)['peer_traffic']['payload_bytes'][12:])


def test_true_full_capture_bucket16_is_not_trimmed_to_twelve_queries():
    raw=json.loads((CASES/'a3_glm53_w4a8_tp16_tpot30_c2.json').read_text())
    raw['runtime']['graph_capture_keys']=[16]
    ops,prologs,ledgers=summarize(raw,12)
    assert len(prologs)==78 and all(p['tokens']==16 for p in prologs)
    assert all(sum(d['global_expert_rows'])==16*8 for d in ledgers)
    assert not any(o.recipe in {'tensor_concat','mc2_tp_output_view'} for o in ops)
    assert all(o.local_shape[0]==16 for o in ops if o.family=='allgather')
