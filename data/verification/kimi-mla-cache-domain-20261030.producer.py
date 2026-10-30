"""Execute source MLA cache declarations through ordinary graph costs.

[Role] Reproduce a core, complete layer, or complete four-layer Body control.
[Invariants] Native durations are never inputs. Retention and service remain
  explicit source conditions; context/active batch are caller declarations.
[Timeouts & Locks] One synchronous bounded graph, no device or network work.
[Side-Effect Policy] Writes only the requested JSON evidence file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.paged_mla_arch22 import IMPLEMENTATION, build
from mindie_sim.prediction.structure import parse_case


def source_case(tp, retained):
    source = ROOT/'examples/cases/high-concurrency-20261030/kimi-source/a3_kimi4_tp8_w4a8_c1_source.json'
    raw = json.loads(source.read_text())
    raw['machine']['num_devices'] = tp
    raw['runtime'].update(tp=tp,ep=tp,dp=1,pp=1,max_model_len=131072)
    raw['runtime'].pop('metadata',None)
    # Retain the independently audited full KDA QKV/BFG caller groups in body
    # controls; the high-concurrency template predates this explicit selector.
    raw['runtime']['kernels']['lowerings']['kda_input_projection'] = 'vllm_ascend_kda_mixed_projection'
    raw['runtime']['kernels']['lowerings']['paged_mla_attention'] = {
        'implementation':IMPLEMENTATION,'options':{'retained_kv_rows':retained}}
    # Independent platform declaration, not native-op calibration. Modes share
    # one AIC copy pipe and one shared L2 port; capacity is per physical die.
    raw['machine'].update(
        rates={'l2':110*20*1.65e9}, buffers={'l2':168*1024*1024},
        resource_units={'l2':1}, buffer_units={'l2':1}, service_modes={
            'mte2_aic_cached':{'resource':'mte2_aic','rate':110*20*1.65e9},
            'l2_read':{'resource':'l2','rate':110*20*1.65e9},
            'l2_write':{'resource':'l2','rate':86*20*1.65e9}})
    return source, raw


def execute(tp, batch, context, retained=512, scope='core'):
    started=time.monotonic()
    source,raw=source_case(tp,retained)
    graph=target_graphs(raw,scope='body',phase='decode',tokens=1,
                        sequences=batch,context=context)[0]
    attention=next(op for op in graph if op.recipe==IMPLEMENTATION)
    selected = ((attention,) if scope=='core' else
                tuple(op for op in graph if op.layer==3) if scope=='layer' else graph)
    names={op.op_id for op in selected}
    external={dep:0 for op in selected for dep in op.deps if dep not in names}
    report=graph_costs(raw,selected,external_inputs=external,isolated=False,input_ready_ms=7)
    plan=build(attention,Geometry.from_devices(parse_case(raw).machine.dies))
    return {
        'status':'conditional_source_control_not_accuracy_acceptance',
        'tp':tp,'dp':1,'pp':1,'ep':tp,'batch':batch,'context':context,
        'scope':scope,'operator_count':len(selected),'input_ready_ms':7,
        'source_case':str(source.relative_to(ROOT)),
        'source_case_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'machine':raw['machine'],'runtime':raw['runtime'],
        'graph_duration_ms':report['duration_ms'],
        'attention_duration_ms':next(row['elapsed_ms'] for row in report['timeline']
                                     if row['recipe']==IMPLEMENTATION),
        'source_core_work':{'resources':dict(plan.work.resources),
                            'matrix_flops':dict(plan.work.matrix)},
        'cache_policy':plan.metadata['cache_policy'],
        'timeline':report['timeline'],'execution':report['execution'],
        'wall_seconds':time.monotonic()-started,
        'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tp',type=int,required=True,choices=(8,16))
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--context',type=int,required=True)
    parser.add_argument('--retained-kv-rows',type=int,default=512)
    parser.add_argument('--scope',choices=('core','layer','body'),default='core')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=execute(args.tp,args.batch,args.context,args.retained_kv_rows,args.scope)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','tp','batch','context','scope',
          'graph_duration_ms','attention_duration_ms','wall_seconds','maxrss_bytes')}))
