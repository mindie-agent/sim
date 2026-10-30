"""Rate views retain actual multi-pool resources without per-op copies."""
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case
from test_graph_analysis import op


def test_contexts_share_readonly_rates_and_see_new_pd_namespace():
    shared=Engine(())
    parsed=parse_case(case())
    prefill=Execution(parsed,CostModel(parsed),engine=shared,namespace='prefill:')
    first=prefill._context(op('first',flops=1e6),0,'none')
    second=prefill._context(op('second',flops=2e6),0,'none')
    assert first.rates is second.rates
    before=dict(first.rates)
    decode=Execution(parsed,CostModel(parsed),engine=shared,namespace='decode:')
    other=decode._context(op('decode',flops=1e6),0,'none')
    assert first.rates['decode:matrix@r0:s0']==other.rates['decode:matrix@r0:s0']
    assert all(first.rates[key]==value for key,value in before.items())
    assert len(first.rates)>len(before)
