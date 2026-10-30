"""Read-only, same-contract body primitive residual decomposition.

[Role] Compare native operators with independently generated source declarations.
[Invariants] No fits or body floors. ABI match scope and call-count differences
  remain explicit; whole-capture native family accounting is a separate ledger.
[Timeouts & Locks] Two bounded streaming CSV captures, no network or device work.
[Side-Effect Policy] Writes only the selected evidence output.
"""
import argparse,bisect,collections,hashlib,json,runpy,statistics,sys,time,resource
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
parser=argparse.ArgumentParser();parser.add_argument('--controls-dir',type=Path,default=Path('/tmp'));parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parameters',type=Path);args=parser.parse_args()
from mindie_sim import target_graphs,Case
from mindie_sim.structure import export_structure
from mindie_sim.structure_contracts import predicted_contract,comparison_scope
from mindie_profiler.normalized import kernel_rows
from mindie_profiler.tensor_contracts import observed_contract
from mindie_profiler.decode_segmentation import is_aicpu_controller
producer=runpy.run_path(str(root/'data/verification/kimi-mla-cache-domain-20261030.producer.py'))
census=json.loads((root/'data/verification/kimi-mla-cache-domain-20261030.json').read_text())['native_census']
def freeze(x):
 return tuple(freeze(v) for v in x) if isinstance(x,list) else x
start=time.monotonic();result={'schema':'kimi-body-primitive-residual/1','runtime_commit':'896a10e9','boundary':'native_kernel_elapsed_per_four_layer_body_not_additive_critical_path','fit':'none','limitations':['ABI equality does not establish compiled implementation or scalar/index values.','All HCOM envelopes and Aiv children are retained as unpaired diagnostics, never summed into body attribution.','Native per-call context remains bounded from client usage, not inferred from cache capacity.'],'cases':[]}
for batch,context in ((1,126000),(24,4096)):
 c=next(c for c in census['cases'] if c['tp']==8 and c['client_concurrency']==batch and c['context_evidence']['lower_bound']==context)
 e=json.loads(Path(c['extraction']['path']).read_text()); forwards=e['forwards'];starts=[f['start_ns'] for f in forwards]
 replay=json.loads((args.controls_dir/f'kimi-body-cache-tp8-b{batch}-k{context}.json').read_text());source,raw=producer['source_case'](8,512)
 structure=export_structure(Case.from_mapping(raw),scope='body',phase='decode',tokens=1,sequences=batch,context=context)
 subcontracts={s['node_ids'][0].split('/',1)[1]:(freeze(s['contract']),s['contract_scope']) for s in structure['subgraphs'] if s['role']=='kda_recurrent' and len(s['node_ids'])==1}
 graph=target_graphs(raw,scope='body',phase='decode',tokens=1,sequences=batch,context=context)[0]; byop={o.op_id:o for o in graph};sim={};unpaired_sim=collections.defaultdict(list)
 for t in replay['timeline']:
  o=byop[t['op_id']];contract=predicted_contract(o.to_dict());scope=comparison_scope(o.to_dict())
  if contract is None and o.op_id in subcontracts:contract,scope=subcontracts[o.op_id]
  if contract is None:
   unpaired_sim[o.recipe].append(t['elapsed_ms']*1000);continue
  s=sim.setdefault(contract,{'contract':contract,'simulated_sum_us':0,'simulated_calls':0,'operators':[]})
  s['simulated_sum_us']+=t['elapsed_ms']*1000;s['simulated_calls']+=1
  s['operators'].append({'op_id':o.op_id,'recipe':o.recipe,'shape':o.local_shape,'dtype':o.dtype,'layer':o.layer,'elapsed_us':t['elapsed_ms']*1000,'match_scope':scope})
 groups={};excluded=collections.Counter()
 for row in kernel_rows(c['source']['path']):
  i=bisect.bisect_right(starts,row.start_ns)-1
  if i<0 or row.start_ns>=forwards[i]['end_ns'] or row.event.device_id!=forwards[i]['device_id']:continue
  if is_aicpu_controller(row.event):excluded['aicpu_controller']+=1;continue
  contract=observed_contract(row);key=contract if contract is not None else ('unmapped',row.kind,row.signature[1:])
  n=groups.setdefault(key,{'contract':contract,'types':collections.Counter(),'signatures':{},'samples_us':[],'per_forward_sum_us':collections.defaultdict(float),'per_forward_calls':collections.Counter()})
  n['types'][row.kind]+=1;n['samples_us'].append(row.duration_ns/1000);n['per_forward_sum_us'][i]+=row.duration_ns/1000;n['per_forward_calls'][i]+=1
  sig=n['signatures'].setdefault(row.signature[1:],{'name':row.name,'kind':row.kind,'core':row.core,'state':row.state,'blocks':row.blocks,'input_shapes':row.input_shapes,'output_shapes':row.output_shapes,'input_dtypes':row.input_dtypes,'output_dtypes':row.output_dtypes,'input_formats':row.input_formats,'output_formats':row.output_formats,'first_csv_line':row.line,'last_csv_line':row.line,'count':0,'counters_us':collections.defaultdict(list)})
  sig['count']+=1;sig['last_csv_line']=row.line
  for k,v in row.counters_us.items():sig['counters_us'][k].append(v)
 matched=[];unpaired=[]
 for key,n in groups.items():
  sums=[n['per_forward_sum_us'].get(i,0) for i in range(len(forwards))];counts=[n['per_forward_calls'].get(i,0) for i in range(len(forwards))]
  report={k:n[k] for k in ('contract','types')};report.update(native_calls=len(n['samples_us']),native_kernel_median_us=statistics.median(n['samples_us']),native_sum_per_body_median_us=statistics.median(sums),native_sum_per_body_mean_us=statistics.mean(sums),native_calls_per_body_min=min(counts),native_calls_per_body_max=max(counts),signatures=[])
  for v in n['signatures'].values():
   v['counters_median_us']={k:statistics.median(a) for k,a in v.pop('counters_us').items()};report['signatures'].append(v)
  if key in sim:
   report.update(sim[key]);report['native_minus_simulated_sum_us']=report['native_sum_per_body_median_us']-report['simulated_sum_us'];report['calls_match_every_forward']=min(counts)==max(counts)==report['simulated_calls'];matched.append(report)
  else:unpaired.append(report)
 matched.sort(key=lambda x:x['native_minus_simulated_sum_us'],reverse=True);unpaired.sort(key=lambda x:x['native_sum_per_body_median_us'],reverse=True)
 result['cases'].append({'case_id':c['case_id'],'source':c['source'],'native_extraction':c['extraction'],'context_evidence':c['context_evidence'],'batch':batch,'simulated_body_ms':replay['graph_duration_ms'],'native_body_ms':e['body_forward_span_ms'],'native_gap_ms':e['body_internal_gap_ms'],'native_forwards':len(forwards),'excluded':excluded,'matched':matched,'unmatched_native':unpaired,'unmatched_simulated':[{'recipe':k,'calls':len(v),'sum_us':sum(v)} for k,v in unpaired_sim.items()]})
 print(c['case_id'],'matched',len(matched),'unmatched',len(unpaired))
 for m in matched[:12]:print('MATCH',list(m['types']),round(m['native_minus_simulated_sum_us'],3),'native',round(m['native_sum_per_body_median_us'],3),'sim',round(m['simulated_sum_us'],3),'count',m['native_calls_per_body_min'],m['native_calls_per_body_max'],m['simulated_calls'])
 for m in unpaired[:8]:print('UNMATCH',list(m['types']),round(m['native_sum_per_body_median_us'],3))
if args.parameters:
 from mindie_sim.prediction.service_regimes import lookup,axis_equal
 parameters=json.loads(args.parameters.read_text());rules=parameters['service_regimes']
 audit=[]
 for rule in rules:
  matching=[o.op_id for o in graph if all(axis_equal(lookup({**o.to_dict(),'hardware':raw['machine']['sku']},k),v) for k,v in rule['match'].items())]
  audit.append({'id':rule['id'],'match':rule['match'],'matching_Kimi_operators':matching})
 result['existing_short_context_parameters']={'path':str(args.parameters),'sha256':hashlib.sha256(args.parameters.read_bytes()).hexdigest(),'training_context':4096,'validation_context':16384,'selector_identity_audit':audit,'conclusion':'No fixed Qwen A3 rule matches Kimi; GDN vs KDA, widths, matrix NK and expert formats differ. No parameter copied.'}
result.update(wall_seconds=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
args.output.write_text(json.dumps(result,indent=2)+'\n')
