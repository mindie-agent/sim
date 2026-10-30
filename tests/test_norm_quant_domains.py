"""Fusion policy preserves consumer domains, native boundaries and tensor work."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path


from mindie_sim import target_graphs
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.transforms import activation_quantization, norm_quant

ROOT=Path(__file__).resolve().parents[1]
SELECTORS=({'recipe':'dense_gate_up'},)


def glm_case():
    raw=json.loads((ROOT/'examples/cases/high-concurrency-20261030/glm78-capture-rows/a3_glm53_w4a8_tp16_tpot30_c2.json').read_text())
    raw['runtime']['lowering_passes']=['activation_quantization','norm_quant','residual_norm']
    return raw


def graph(raw):
    return target_graphs(raw,scope='body',phase='decode',tokens=6,sequences=2,
                         context=4096,apply_body_defaults=False)[0]


def test_complete_78_layer_body_only_fuses_three_dense_ffn_inputs():
    original=glm_case();generic=graph(original)
    raw=deepcopy(original)
    raw['runtime']['kernels']['fusion_domains']={'norm_quant':{'consumers':list(SELECTORS)}}
    selected=graph(raw)
    before=Counter(o.family for o in generic)
    after=Counter(o.family for o in selected)
    assert before['add_rmsnorm_quant']==60
    assert after['add_rmsnorm_quant']==3
    assert after['add_rmsnorm']==152
    assert after['quantize']-before['quantize']==57
    fused=[o for o in selected if o.family=='add_rmsnorm_quant']
    for op in fused:
        # The selector is a consumer recipe, not the current model/layer ID.
        consumers=[o for o in selected if op.op_id in o.deps and o.recipe=='dense_gate_up']
        assert len(consumers)==1
        assert dict(consumers[0].metadata)['structure_block']=='ffn'
        assert dict(consumers[0].metadata)['structure_role']=='body'
        assert dict(op.metadata)['fusion_consumer_domain']['consumers']==SELECTORS
    ordinary_before={o.op_id for o in generic if o.family=='quantize' and '.q_a.quantize' in o.op_id}
    ordinary_after={o.op_id for o in selected if o.family=='quantize' and '.q_a.quantize' in o.op_id}
    assert len(ordinary_after-ordinary_before)==57
    assert len(ordinary_before)==21 and len(ordinary_after)==78
    assert sum(dict(o.metadata).get('fusion_role')=='mla_prolog' for o in selected)==78
    # Splitting calls restores the BF16 intermediate store and read, not new
    # arithmetic, a multiplier, a new cache sweep or changed matrix weights.
    assert sum(o.flops for o in generic)==sum(o.flops for o in selected)
    assert sum(o.vector_ops for o in generic)==sum(o.vector_ops for o in selected)
    assert sum(o.hbm_bytes for o in selected)-sum(o.hbm_bytes for o in generic)==57*2*12*6144*2
    assert graph(original)==generic  # no mutation; no-config behavior remains


def shared_conversion():
    norm=vector_op(op_id='n',deps=('input',),stage=0,layer=0,requests=(),elements=32,
        ops_per_element=5,dtype='bf16',family='rmsnorm',phase='decode',recipe='rmsnorm',shape=(1,32))
    quant=QuantSpec(name='control',weight_format='int8',activation_format='int8')
    def matrix(name,recipe,block):
        return gemm_op(op_id=name,deps=('n',),stage=0,layer=0,requests=(),m=1,n=64,k=32,
            quant=quant,category='dense_mlp',phase='decode',recipe=recipe,
            metadata=(('structure_block',block),('structure_role','body')))
    return activation_quantization((norm,matrix('a','dense_gate_up','ffn'),
                                     matrix('b','q_projection','attention')),'bf16')


def test_one_allowed_consumer_cannot_fuse_a_shared_conversion_for_other_domains():
    ops=shared_conversion()
    assert sum(o.family=='quantize' for o in ops)==1
    selected=norm_quant(ops,'bf16',consumers=SELECTORS)
    assert sum(o.family=='rmsnorm' for o in selected)==1
    assert sum(o.family=='quantize' for o in selected)==1
    assert not any(o.family=='rmsnorm_quant' for o in selected)
    both=norm_quant(ops,'bf16',consumers=(*SELECTORS,{'recipe':'q_projection'}))
    assert sum(o.family=='rmsnorm_quant' for o in both)==1
    assert {o.deps for o in both if o.recipe in {'dense_gate_up','q_projection'}}=={('a.quantize',)}
    assert norm_quant(ops,'bf16')==norm_quant(ops,'bf16',consumers=None)


def test_selector_structure_axes_are_conjunctive_and_empty_domain_disables_fusion():
    ops=shared_conversion()
    both=[replace(o,recipe='dense_gate_up') if o.op_id=='b' else o for o in ops]
    assert any(o.family=='rmsnorm_quant' for o in norm_quant(both,'bf16',consumers=SELECTORS))
    assert not any(o.family=='rmsnorm_quant' for o in norm_quant(both,'bf16',consumers=(
        {'recipe':'dense_gate_up','structure_block':'ffn'},)))
    assert not any(o.family=='rmsnorm_quant' for o in norm_quant(both,'bf16',consumers=()))
