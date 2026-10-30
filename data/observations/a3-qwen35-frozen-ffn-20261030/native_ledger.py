"""Read one existing Qwen3.5 q4 capture against retained complete Body windows.

[Role] Retain typed native activity and compare existing scheduled recipes.
[Invariants] No fitting, no prediction, no new segmentation or timing correction.
[Timeouts & Locks] Entire source hash plus streamed scan bounded to 90 seconds.
[Side-Effect Policy] Writes a local compact diagnostic only; no device execution.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import sha256
import argparse, csv, gzip, json, resource, signal, sys, time
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch',type=int,required=True)
parser.add_argument('--comparison',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
batch=args.batch
start=time.monotonic()
def timeout(*_): raise TimeoutError('qwen35_existing_source_scan_exceeded_90s')
signal.signal(signal.SIGALRM,timeout); signal.alarm(90)
report_path=args.comparison
report=json.loads(report_path.read_text())
report_identity={'path':str(report_path),'sha256':sha256(report_path.read_bytes()).hexdigest(),
                 'size_bytes':report_path.stat().st_size}
caseid=report['case_id']
extract_identity=report['extraction_identity']
extract_path=Path(extract_identity['path']); extract=json.loads(extract_path.read_text())
assert sha256(extract_path.read_bytes()).hexdigest()==extract_identity['sha256']
source=extract['source_identity']; source_path=Path(source['path'])
digest=sha256()
with source_path.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
assert digest.hexdigest()==source['sha256']==report['source_identity']['sha256']
forwards=[f for f in extract['forwards'] if f['geometry']['tokens_per_sequence']==4 and f['geometry']['sequences_per_replica']==batch]
assert len(forwards)==len(extract['forwards'])
starts=[f['start_ns'] for f in forwards]
layers=[[s for s in f['stages'] if s['layer_id'] is not None] for f in forwards]
assert all(len(s) in (0,40) for s in layers)
layer_starts=[[s['start_ns'] for s in stages] for stages in layers]
groups={}; counts=Counter(); forward_counts=[Counter() for _ in forwards]
hcom_seen=set(); latest_forward=-1
abi_keys=['Type','Accelerator Core','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num']
opener=gzip.open if source_path.suffix=='.gz' else open
with opener(source_path,'rt',newline='') as f:
 reader=csv.DictReader(f)
 for line,row in enumerate(reader,2):
  counts['rows_scanned']+=1
  begin=int(Decimal(row['Start Time(us)'].strip())*1000)
  duration=int(Decimal(row['Duration(us)'].strip())*1000); end=begin+duration
  index=bisect_right(starts,begin)-1
  if index<0 or begin>=forwards[index]['end_ns'] or end>forwards[index]['end_ns']:
   continue
  if row['Device_id']!=forwards[index]['device_id']: continue
  if row['Accelerator Core']=='AI_CPU' and row['Type'] in {'allreduceAicpuKernel','allgatherAicpuKernel','reducescatterAicpuKernel','alltoallAicpuKernel'}:
   counts['scheduler_wrappers_excluded']+=1; continue
  if index!=latest_forward:
   assert index>latest_forward, 'CSV crosses backwards between Body windows'
   latest_forward=index; hcom_seen.clear()
  if row['Type'].startswith('hcom_'):
   identity=(row['Device_id'],row['Type'],begin,end)
   if identity in hcom_seen:
    counts['duplicate_hcom_intervals_excluded']+=1; continue
   hcom_seen.add(identity)
  layer_index=bisect_right(layer_starts[index],begin)-1
  stage=layers[index][layer_index] if layer_starts[index] else None
  key=tuple(row[k] for k in abi_keys)
  group=groups.setdefault(key,{'abi':{k:row[k] for k in abi_keys},'count':0,'sum_ns':0,'min_ns':duration,'max_ns':duration,'locators':[]})
  group['count']+=1;group['sum_ns']+=duration;group['min_ns']=min(group['min_ns'],duration);group['max_ns']=max(group['max_ns'],duration)
  if len(group['locators'])<2:group['locators'].append({'csv_line':line,'forward_id':forwards[index]['forward_id'],'layer':stage['layer_id'] if stage else None,'start_ns':begin,'end_ns':end,'Name':row['Name'],'Task ID':row['Task ID'],'Stream ID':row['Stream ID'],'Device_id':row['Device_id']})
  forward_counts[index][row['Type']]+=1;counts['selected_events']+=1
for row in groups.values():
 row['mean_us']=row['sum_ns']/row['count']/1000;row['activity_per_body_ms']=row['sum_ns']/len(forwards)/1e6;row['calls_per_body']=row['count']/len(forwards)
result={'scope':'existing_complete_q4_native_Body_diagnostic_no_fit_no_prediction','case_id':caseid,'source_identity':source,'source_hash_verified':True,'extraction_identity':extract_identity,'comparison_report_identity':report_identity,'comparison_runtime':report['runtime_identity'],'geometry':extract['geometry'],'forwards':len(forwards),'body_observed_ms':report['comparison'],'native_activity':extract['body_activity_ms'],'native_internal_gap':extract['body_internal_gap_ms'],'scan_counts':dict(counts),'native_counts_per_body':{kind:{'min':min(c[kind] for c in forward_counts),'max':max(c[kind] for c in forward_counts),'total':sum(c[kind] for c in forward_counts)} for kind in sorted(set().union(*forward_counts))},'native_groups':sorted(groups.values(),key=lambda r:-r['activity_per_body_ms']),'scheduled_recipe_groups':report['prediction']['scheduled_operator_summary'],'limitations':['native activity sums and scheduled intervals are not additive Body decompositions','shape-less helper geometry remains source-conditional','all-rank arrival and compiled binary identity not observed','only three retained forwards have layer windows; native grouping covers all complete Bodies without inventing unavailable layer occurrence bindings','existing ee3 prediction retained; no current runtime prediction performed'],'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
path=args.output;path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(path),'sources':1,'forwards':len(forwards),'elapsed_seconds':result['elapsed_seconds'],'peak_rss_bytes':result['peak_rss_bytes'],'groups':len(groups),'counts':dict(counts)}))
signal.alarm(0)
