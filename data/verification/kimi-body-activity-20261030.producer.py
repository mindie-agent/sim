"""Account for every native family within retained four-layer body boundaries.

[Role] Compare observed activity and overlap with declared simulation roles.
[Invariants] Family sums are not critical paths; unmatched events stay visible.
  AI_CPU control intervals are separate from HCOM envelopes and never added.
[Timeouts & Locks] Two bounded streaming captures; no subprocess/network/device.
[Side-Effect Policy] Write one caller-selected evidence JSON, never parameters.
"""
import argparse,bisect,collections,json,statistics,sys,time,resource,runpy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mindie_profiler.normalized import kernel_rows
from mindie_profiler.decode_segmentation import is_aicpu_controller
from mindie_sim import target_graphs

def union(intervals):
 end=None;total=0
 for a,b in sorted(intervals):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total

def stats(values):
 return {'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values)}

def mapping(kind,op):
 recipe=op.recipe
 direct={'hcom_allGather_':('ring_allgather',),'hcom_reduceScatter_':('ring_reduce_scatter',),
 'FusedInferAttentionScore':('ascend_paged_mla_decode_arch22',),
 'RecurrentKda':('vllm_ascend_recurrent_kda_arch22',),
 'MoeDistributeDispatchV2':('expert_dispatch','route_pack'),
 'MoeDistributeCombineV2':('expert_combine','route_unpack'),
 'DequantSituQuant':('grouped_situ','situ'),
 'AttnResFwd':('attention_residual_mix','attention_residual_softmax','attention_residual_output_cast','residual_values_fp32'),
 'DynamicQuant':('dynamic_quantize',),'CausalConv1d':('short_convolution',),
 'TransposeBatchMatMul':('v_expand',),'MoeGatingTopK':('router_topk',),
 'MatMulV3':('router',),'ForeachAddList':('expert_add','residual_add')}
 if kind=='DequantSituQuant':
  return op.layer>0 and (recipe in ('grouped_situ','situ') or recipe=='dynamic_quantize' and any(dep.endswith(('expert_act','shared.situ')) for dep in op.deps))
 if kind=='MoeDistributeDispatchV2' and recipe=='dynamic_quantize' and any(dep.endswith('.dispatch') for dep in op.deps):return True
 if kind in direct:return recipe in direct[kind]
 if kind=='GroupedMatmul':return op.family=='grouped_gemm'
 if kind=='QuantBatchMatmulV3':return op.family in ('gemm','qkv_projection','gate_up_projection','down_projection') and dict(op.metadata).get('activation_format')=='int8'
 if kind=='MatMulV2':return op.family in ('gemm','qkv_projection','gate_up_projection','down_projection') and op.dtype=='bf16'
 if kind=='PadV3':return 'pad' in recipe
 return False

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--controls-dir',type=Path,default=Path('/tmp'));parser.add_argument('--output',type=Path,default=Path('/tmp/kimi-body-activity-20261030.json'));args=parser.parse_args()
 census=json.loads((ROOT/'data/verification/kimi-mla-cache-domain-20261030.json').read_text())['native_census'];producer=runpy.run_path(str(ROOT/'data/verification/kimi-mla-cache-domain-20261030.producer.py'))
 result={'schema':'kimi-body-family-activity/1','boundary':'all native event types within same-device body; union clips event ends at retained body end','status':'diagnosis_only_no_parameters','limitations':['Activity is elapsed event coverage, not independently identified pipe busy time.','Family candidate simulation roles indicate caller ownership, not same ABI or additive time equivalence; candidates can overlap across family rows.','No HCOM AIV child rows are inferred when absent from the capture.','Unknown group_list and active-mask values stay unknown.'],'cases':[]};started=time.monotonic()
 for batch,context in((1,126000),(24,4096)):
  c=next(c for c in census['cases'] if c['tp']==8 and c['client_concurrency']==batch and c['context_evidence']['lower_bound']==context);ex=json.loads(Path(c['extraction']['path']).read_text());fs=ex['forwards'];starts=[f['start_ns'] for f in fs];N=len(fs)
  active=[[] for _ in fs];controllers=[[] for _ in fs];hcom=[[] for _ in fs];hcom_aiv=[[] for _ in fs];families={};controltypes=collections.Counter();control_samples=[]
  for row in kernel_rows(c['source']['path']):
   i=bisect.bisect_right(starts,row.start_ns)-1
   if i<0 or row.start_ns>=fs[i]['end_ns'] or row.event.device_id!=fs[i]['device_id']:continue
   interval=(row.start_ns,min(row.end_ns,fs[i]['end_ns']))
   if is_aicpu_controller(row.event):
    controllers[i].append(interval);controltypes[row.kind]+=1
    if len(control_samples)<6:control_samples.append({'line':row.line,'type':row.kind,'core':row.core,'task_type':row.event.task_type,'duration_us':row.duration_ns/1000})
    continue
   active[i].append(interval)
   f=families.setdefault(row.kind,{'intervals':[[] for _ in fs],'cores':collections.Counter(),'task_types':collections.Counter(),'count':0,'first_csv_line':row.line,'last_csv_line':row.line})
   f['intervals'][i].append(interval);f['cores'][row.core]+=1;f['task_types'][row.event.task_type]+=1;f['count']+=1;f['last_csv_line']=row.line
   if row.kind.startswith('hcom_'):hcom[i].append(interval)
   if ('allgather' in row.name.lower() or 'reduce_scatter' in row.name.lower()) and 'aiv' in row.name.lower():hcom_aiv[i].append(interval)
  activity=[union(v)/1000 for v in active];spans=[(f['end_ns']-f['start_ns'])/1000 for f in fs];gaps=[s-a for s,a in zip(spans,activity)];control_union=[union(v)/1000 for v in controllers];hcom_union=[union(v)/1000 for v in hcom];hc_overlap=[(union(a)+union(b)-union(a+b))/1000 for a,b in zip(hcom,controllers)]
  replay=json.loads((args.controls_dir/f'kimi-body-cache-tp8-b{batch}-k{context}.json').read_text());_,raw=producer['source_case'](8,512);graph=target_graphs(raw,scope='body',phase='decode',tokens=1,sequences=batch,context=context)[0];ops={o.op_id:o for o in graph};sim_by_id={t['op_id']:t for t in replay['timeline']}
  rows=[]
  for kind,f in families.items():
   native=[union(v)/1000 for v in f['intervals']];summed=[sum(b-a for a,b in v)/1000 for v in f['intervals']];candidate=[o for o in graph if mapping(kind,o)];ts=[sim_by_id[o.op_id] for o in candidate]
   entry={'native_type':kind,'native_calls':f['count'],'cores':f['cores'],'task_types':f['task_types'],'first_csv_line':f['first_csv_line'],'last_csv_line':f['last_csv_line'],'sum_elapsed_us':stats(summed),'union_activity_us':stats(native),'within_family_overlap_us':stats([s-u for s,u in zip(summed,native)]),'mean_share_of_body_activity_pct':100*statistics.mean(native)/statistics.mean(activity),'simulation_role_candidates':{'recipes':sorted({o.recipe for o in candidate}),'calls':len(candidate),'sum_elapsed_us':sum(t['elapsed_ms']*1000 for t in ts),'union_elapsed_us':union([(t['ready_ms']*1000,t['complete_ms']*1000) for t in ts]),'pairing':'caller_role_only_not_native_ABI_or_valid_routing_equivalence' if candidate else 'unmatched_no_declared_role'}}
   rows.append(entry)
  rows.sort(key=lambda x:x['union_activity_us']['mean'],reverse=True);cumulative=[[] for _ in fs]
  for row in rows:
   for i,ints in enumerate(families[row['native_type']]['intervals']):cumulative[i].extend(ints)
   row['cumulative_family_union_us']=stats([union(v)/1000 for v in cumulative]);row['cumulative_mean_activity_coverage_pct']=100*row['cumulative_family_union_us']['mean']/statistics.mean(activity)
  simroles=[]
  for recipe in sorted({o.recipe for o in graph}):
   same=[o for o in graph if o.recipe==recipe];ts=[sim_by_id[o.op_id] for o in same];simroles.append({'recipe':recipe,'calls':len(same),'sum_elapsed_us':sum(t['elapsed_ms']*1000 for t in ts),'union_elapsed_us':union([(t['ready_ms']*1000,t['complete_ms']*1000) for t in ts])})
  simroles.sort(key=lambda x:x['union_elapsed_us'],reverse=True)
  routed=[]
  for o in graph:
   if o.family!='grouped_gemm':continue
   m=dict(o.metadata);rr=m.get('routing_realization',{});routed.append({'op_id':o.op_id,'shape':o.local_shape,'weight_bank_experts':m.get('matrix_weight_bank_groups'),'active_expert_matrices':m.get('matrix_groups'),'routing_basis':m.get('routing_basis'),'valid_received_assignment_rows':sum(rr.get('local_expert_rows',[])),'global_declared_tokens':rr.get('global_tokens'),'row_classes':rr.get('row_classes'),'capture_group_list_values':'unobserved'})
  result['cases'].append({'case_id':c['case_id'],'source':c['source'],'extraction':c['extraction'],'native_forwards':N,'body_span_us':stats(spans),'body_activity_union_us':stats(activity),'body_gap_us':stats(gaps),'extraction_activity_agreement_max_us':max(abs(a-f['body_activity_ms']*1000) for a,f in zip(activity,fs)),'sim_body_us':replay['graph_duration_ms']*1000,'native_minus_sim_body_median_us':statistics.median(spans)-replay['graph_duration_ms']*1000,'all_native_families':rows,'sim_roles':simroles,'routed_sim_geometry':routed,'hcom_and_controller_overlap':{'hcom_envelope_union_us':stats(hcom_union),'controller_union_us':stats(control_union),'intersection_union_us':stats(hc_overlap),'controller_types':controltypes,'controller_examples':control_samples,'separately_exported_hcom_aiv_rows':sum(map(len,hcom_aiv)),'inclusion':'HCOM envelope contributes once to global union. AI_CPU controllers are diagnostic only; no independently exported transport AIV child rows in these selected bodies.'}})
  print(c['case_id'],stats(activity),stats(gaps),'HCOM',stats(hcom_union),'controllers',stats(control_union),'overlap',stats(hc_overlap))
  for row in rows[:10]:print(row['native_type'],round(row['union_activity_us']['mean'],3),round(row['mean_share_of_body_activity_pct'],2),'sim',round(row['simulation_role_candidates']['union_elapsed_us'],3),round(row['cumulative_mean_activity_coverage_pct'],2))
 result.update(wall_seconds=time.monotonic()-started,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
