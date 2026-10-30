"""Independent timing/slot oracles for the source's separate instruction FIFOs."""
from dataclasses import replace
import random

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mxfp4_pipeline import Panel, Slots, SourcePipeline, pipeline_cycle
from mindie_sim.prediction.program import Stage


def panel(read=3, vector=4, publish=2, operand=0, cube=0, scale=False):
    return Panel(None, None, ((0, 0, 0, read, vector, publish), (0, 0, 0, 0, 0, 0)),
                 (scale, False), ((operand, cube),))


def actual(panels, slots=Slots(), ready=0):
    resources = ('a', 'r0', 'v0', 'p0', 'r1', 'v1', 'p1', 'operand', 'cube')
    engine = Engine([Resource(name, 1) for name in resources], trace_limit=10000)
    def convert(p):
        def stage(name, work):
            return Stage(name, {name: work} if work else {})
        return Panel(stage('a', p.activation) if p.activation is not None else None,
                     stage('a', p.activation_scale) if p.activation_scale is not None else None,
                     tuple(tuple(stage(f'{role}{sub}', work) for role, work in
                         zip(('r','v','p','r','v','p'), stages)) for sub, stages in enumerate(p.vectors)),
                     p.scales, tuple((stage('operand', a),stage('cube', b)) for a,b in p.micros))
    program = SourcePipeline(len(panels), lambda i: convert(panels[i]), slots)
    end = program.start(engine, 'source', engine.submit('input', release=ready))
    if ready:
        engine.advance(horizon=ready - 1)
        assert not any(engine.serviced.values())
    engine.advance(until=(end,))
    assert engine.summary()['live_tasks'] == engine.summary()['live_programs'] == 0
    return end.time-ready, engine


@pytest.mark.parametrize('count,expected', ((1,9),(2,13),(3,17)))
def test_independent_three_pipe_hand_oracle(count, expected):
    p = panel()
    # read: 0-3,3-6,6-9; V:3-7,7-11,11-15; publish:7-9,11-13,15-17.
    assert pipeline_cycle((p,),count).span_ms == expected
    assert actual([p]*count)[0] == expected
    if count > 1:
        assert expected < 9*count


def test_l1_recycles_at_operand_completion_not_cube_completion():
    p = panel(3,4,2,1,10)
    span, engine = actual([p]*3,Slots(l1=1))
    times = {x['task']: (x['start_ms'],x['end_ms']) for x in engine.trace}
    assert times['source:panel0:operand0'] == (9,10)
    assert times['source:panel0:cube0'] == (10,20)
    assert times['source:panel1:publish0'] == (11,13)
    assert times['source:panel1:publish0'][1] < times['source:panel0:cube0'][1]
    assert span == pipeline_cycle((p,),3,slots=Slots(l1=1)).span_ms


@pytest.mark.parametrize('width',(1,2,4))
def test_input_and_expanded_slots_enforce_earliest_legal_reuse(width):
    p = panel(1,2,10,1,20)
    _, engine = actual([p]*8,Slots(input=width,widened=width,l1=2))
    times = {x['task']: (x['start_ms'],x['end_ms']) for x in engine.trace}
    for i in range(width,8):
        assert times[f'source:panel{i}:packed_read0'][0] >= times[f'source:panel{i-width}:widen0'][1]
        assert times[f'source:panel{i}:widen0'][0] >= times[f'source:panel{i-width}:publish0'][1]
    assert actual([p]*8,Slots(input=width,widened=width))[0] == pipeline_cycle(
        (p,),8,slots=Slots(input=width,widened=width)).span_ms


def test_scale_read_shares_input_ring_and_has_separate_output_ring():
    p = panel(1,4,2)
    scale = replace(p, vectors=((5,6,7,1,4,2),p.vectors[1]),scales=(True,False))
    _, engine=actual([scale,p,p,p,scale,p,p],Slots(input=4))
    times={x['task']:(x['start_ms'],x['end_ms']) for x in engine.trace}
    # scale + first weight consumed slots0/1; weight of panel3 reuses slot0.
    assert times['source:panel3:packed_read0'][0] >= times['source:panel0:scale_transpose0'][1]
    assert times['source:panel4:scale_read0'][0] >= times['source:panel0:widen0'][1]
    assert pipeline_cycle((scale,p,p,p),1,tail=(scale,p,p)).span_ms == engine.now


def test_cycle_jump_preserves_unused_anchors_and_nonuniform_tail():
    p=panel()
    tail=replace(p,activation=100,activation_scale=17,
                 vectors=((8,4,3,2,1,5),p.vectors[1]),scales=(True,False))
    for n in (1,3,5,31,100):
        # Two independent initial A transfers cost 117; producer tail extends
        # the hand-derived four-unit cadence by ten beyond its final publish.
        assert pipeline_cycle((p,),n,tail=(tail,)).span_ms == max(117,4*n+15)
        if n <= 31:
            assert pipeline_cycle((p,),n,tail=(tail,)).span_ms == actual([p]*n+[tail])[0]
    large=pipeline_cycle((p,),100000)
    assert large.span_ms == 400005
    assert large.stepped_panels < 100
    assert large.jumped_panels > 99000


def test_random_finite_states_against_independent_resource_engine():
    randomizer=random.Random(7281)
    for _ in range(30):
        slots=Slots(*(randomizer.choice((1,2,4)) for _ in range(5)))
        pattern=tuple(panel(*(randomizer.randint(0,7) for _ in range(5)))
                      for _ in range(randomizer.choice((3,5))))
        repeats=randomizer.randint(1,5)
        tail=(panel(2,3,1,4,5),)
        expected=actual(list(pattern)*repeats+list(tail),slots,ready=7)[0]
        assert pipeline_cycle(pattern,repeats,tail=tail,slots=slots).span_ms == expected
