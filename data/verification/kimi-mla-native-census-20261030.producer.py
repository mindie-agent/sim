"""Read bounded native MLA rows; timing remains validation, not runtime input."""
import csv,hashlib,json,re,statistics,sys,time,resource
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mindie_sim.prediction.paged_mla_arch22 import tile_work
from mindie_sim.prediction.paged_mla_arch22_dispatch import Dispatch,source_dispatch
R=ROOT
C=R/'data/verification/kimi-mixed-projection-source-control-20261030.json'
fields=['Type','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num']
counters=['Duration(us)','aicore_time(us)','aic_mac_time(us)','aic_scalar_time(us)','aic_mte1_time(us)','aic_mte2_time(us)','aic_fixpipe_time(us)','aiv_time(us)','aiv_vec_time(us)','aiv_scalar_time(us)','aiv_mte2_time(us)','aiv_mte3_time(us)']
def pin(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size}
def stats(a):
 valid=[float(x) for x in a if x!='N/A'];return {'count':len(valid),'unavailable_count':len(a)-len(valid),**({'min':min(valid),'median':statistics.median(valid),'mean':statistics.fmean(valid),'max':max(valid)} if valid else {})}
def work(batch,heads,context):
 d=Dispatch.parse(source_dispatch((context,)*batch,heads,20),(context,)*batch,heads,20)
 loads=[];hbms=[];flops=[];tiles=[]
 for jobs in d.owners:
  load=hbm=flop=tilecount=0
  for seq,start,count,splits in jobs:
   for t in range(count):
    k=min(512,context+1-(start+t)*512)
    for w in tile_work(k,heads,first=t==0,last=t+1==count,splits=splits,page_size=128).values():
     load+=w.resources.get('mte2_aic',0);hbm+=w.resources.get('hbm',0);flop+=w.matrix.get('bf16',0)
    tilecount+=1
  loads.append(load);hbms.append(hbm);flops.append(flop);tiles.append(tilecount)
 return {'context':context,'active_aic_owners':len(d.owners),'max_owner_tiles':max(tiles),'sum_mte2_aic_bytes':sum(loads),'max_owner_mte2_aic_bytes':max(loads),'sum_fa_hbm_bytes':sum(hbms),'max_owner_fa_hbm_bytes':max(hbms),'sum_bf16_flops':sum(flops)}
started=time.monotonic();result=[]
for c in json.loads(C.read_text())['census']:
 if c['mixed_projection_contract']!='matched':continue
 p=Path(c['source']['path']);summary=p.with_name('benchmark_summary.json');s=json.loads(summary.read_text());requests=s['requests'];prompt=sorted({r['usage']['prompt_tokens'] for r in requests});completion=sorted({r['requested_max_tokens'] for r in requests});concurrency=s['summary']['concurrency']
 groups=defaultdict(list);matched=[]
 with p.open(newline='') as f:
  for line,r in enumerate(csv.DictReader(f),2):
   if r['Type']!='FusedInferAttentionScore':continue
   key=tuple(r[k] for k in fields);groups[key].append((line,{k:r[k] for k in counters}))
 abi=[]
 for key,rows in groups.items():
  raw=dict(zip(fields,key));shapes=[[int(v) for v in shape.split(',')] if shape else [] for shape in raw['Input Shapes'].strip('"').split(';')];batch,heads=shapes[0][:2]
  ws=[work(batch,heads,k) for k in (prompt[0],prompt[0]+completion[-1])]
  abi.append({'contract':raw,'physical_batch':batch,'physical_heads':heads,'valid_sequence_count_equal_physical_batch':batch==concurrency,'first_csv_line':rows[0][0],'last_csv_line':rows[-1][0],'native_counters_us':{k:stats([r[k] for _,r in rows]) for k in counters},'source_work_context_bounds':ws})
 ex=json.loads(Path(c['extraction']['path']).read_text());layer=next((l for l in ex.get('per_layer_ms',[]) if l['layer_id']==3),None)
 result.append({'case_id':c['case_id'],'tp':c['tp'],'source':pin(p),'benchmark_summary':pin(summary),'extraction':c['extraction'],'client_concurrency':concurrency,'context_evidence':{'prompt_tokens':prompt,'max_output_budget':completion,'lower_bound':prompt[0],'upper_bound':prompt[0]+completion[-1],'per_call_values_observed':False,'basis':'same_source_client_usage_and_requested_output_budget; cache_shape_not_context'},'layer3_span_ms':layer['stats'] if layer else None,'native_abi_groups':abi})
 print(c['tp'],concurrency,prompt[0],[(a['physical_batch'],a['native_counters_us']['Duration(us)']['median']) for a in abi],flush=True)
out={'schema':'mindie-native-operator-residual/1','operator':'FusedInferAttentionScore','source_implementation':'ascend_paged_mla_decode_arch22','scope':'A3 46 source captures only; full native FIA population, same physical tensor domain; source binary identity remains unestablished','producer':pin(Path(__file__)),'census':pin(C),'cases':result,'fit_status':'not_fitted','timing_scope':'native_operator_only; body/layer timings are not parameters','limitations':['Native per-call KV values and binary tiling key are not observed.','Padded physical query rows are not proof of equal valid work; C6/C12 must be excluded from equal-length source response fitting.','Hardware pipe counters overlap and include waiting; they are not additive service-time targets.','Source work includes all GM traffic as HBM; no cache discount is inferred.'],'elapsed_seconds':time.monotonic()-started,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
