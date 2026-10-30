"""Compare cached immutable MLA tiles against the pre-cache implementation.

[Role] Verify complete result/trace identity and record CPU construction cost.
[Invariants] Same input work, physical owners, event schedule and dependencies.
[Timeouts & Locks] Local bounded geometries; no network or locks.
[Side-Effect Policy] Reads a Git source and writes one local comparison receipt.
"""
import hashlib, importlib.util, json, pathlib, subprocess, sys, time
from dataclasses import replace
root=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
from mindie_sim import target_graphs, graph_costs, operator_costs
from mindie_sim.prediction import paged_mla
name='mindie_sim.prediction._paged_mla_reference'
oldfile=pathlib.Path('/tmp/paged_mla_reference_f11514e5.py')
oldfile.write_bytes(subprocess.check_output(['git','show','f11514e5:mindie_sim/prediction/paged_mla.py'],cwd=root))
spec=importlib.util.spec_from_file_location(name,oldfile); old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
new_plan=paged_mla._plan
raw=json.loads((root/'examples/cases/kimi-a5-paged-host-20261030/c12.json').read_text())
rows=[]
for batch,context in [(1,0),(1,128),(1,257),(2,383),(3,519),(5,1024),(12,126000)]:
 ops=target_graphs(raw,scope='body',phase='decode',tokens=1,sequences=batch,context=context)[0]
 op=next(o for o in ops if o.recipe==paged_mla.IMPLEMENTATION)
 pair=[]
 for label,plan in [('old',old._plan),('new',new_plan)]:
  paged_mla._plan=plan;started=time.monotonic()
  result=graph_costs(raw,[replace(op,deps=())],isolated=False,input_ready_ms=7,trace_limit=20000 if context<2000 else 0)
  pair.append({'label':label,'wall_seconds':time.monotonic()-started,'sha256':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest(),'duration_ms':result['duration_ms'],'tasks':result['execution']['tasks_submitted'] if 'tasks_submitted' in result['execution'] else result['execution']})
 assert pair[0]['sha256']==pair[1]['sha256'], (batch,context,pair)
 rows.append({'batch':batch,'context':context,'exact_result_and_trace_equal':True,'pair':pair})
 print(batch,context,[(r['label'],round(r['wall_seconds'],4)) for r in pair],flush=True)
paged_mla._plan=new_plan
pathlib.Path('/tmp/paged-mla-template-validation-20261030.json').write_text(json.dumps({'reference_revision':'f11514e5','candidate_file_sha256':hashlib.sha256((root/'mindie_sim/prediction/paged_mla.py').read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n')
