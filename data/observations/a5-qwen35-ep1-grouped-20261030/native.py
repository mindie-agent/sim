"""Stream old A5 EP1 grouped kernels inside retained complete q4 Body windows.

[Role] Retain native ABI and primitive durations for eight independent sources.
[Invariants] Training labels are explicit; no Body duration enters aggregation.
[Timeouts & Locks] Per-source90s, bounded ABI groups and streamed compressed CSV.
[Side-Effect Policy] Read original captures; write one compact local ledger.
"""
from bisect import bisect_right
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import csv
import gzip
import json
from pathlib import Path
import resource
import signal
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORT = ROOT.parent / 'profiler/examples/structure/results/decode.json'
FIELDS = ('Name', 'Type', 'Accelerator Core', 'Input Shapes', 'Input Data Types',
          'Input Formats', 'Output Shapes', 'Output Data Types', 'Output Formats',
          'Block Num', 'Mix Block Num')
TRAIN = {'a5-qwen35-tp4-longctx-p65536-c1-hold', 'a5-qwen35-tp4-longctx-p65536-c4-hold'}


def digest(path):
    result = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def scan(card):
    started = time.monotonic()
    def deadline(*_): raise TimeoutError('native_grouped_source_scan_exceeded_90s')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(90)
    report_path = Path(card['evidence']['path'])
    report = json.loads(report_path.read_text())
    assert digest(report_path) == card['evidence']['sha256']
    extraction = report['extraction_identity']; path = Path(extraction['path'])
    assert digest(path) == extraction['sha256']
    source = json.loads(path.read_text()); raw_path = Path(source['source_identity']['path'])
    assert digest(raw_path) == source['source_identity']['sha256'] == card['source']['sha256']
    windows = source['forwards']; starts = [f['start_ns'] for f in windows]
    assert all(f['geometry']['tokens_per_sequence']==4 for f in windows)
    counts = [Counter() for _ in windows]; groups = {}; scanned = selected = 0
    opener = gzip.open if raw_path.suffix=='.gz' else open
    with opener(raw_path, 'rt', newline='') as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            scanned += 1
            if row['Type']!='GroupedMatmul': continue
            begin = int(Decimal(row['Start Time(us)'])*1000)
            duration = int(Decimal(row['Duration(us)'])*1000)
            index = bisect_right(starts, begin)-1
            if (index<0 or begin+duration>windows[index]['end_ns']
                    or row['Device_id']!=windows[index]['device_id']): continue
            key = tuple(row[k] for k in FIELDS)
            group = groups.setdefault(key, {'abi':dict(zip(FIELDS,key)), 'count':0,
                'sum_ns':0, 'min_ns':duration, 'max_ns':duration,
                'first_csv_line':line, 'last_csv_line':line, 'locators':[],
                'forward_geometry_counts':Counter()})
            group['count'] += 1; group['sum_ns'] += duration
            group['min_ns'] = min(group['min_ns'], duration)
            group['max_ns'] = max(group['max_ns'], duration); group['last_csv_line'] = line
            geometry = windows[index]['geometry']
            geometry_key = (geometry['query_rows'], geometry['tokens_per_sequence'],
                            geometry['sequences_per_replica'])
            group['forward_geometry_counts'][geometry_key] += 1
            if len(group['locators'])<2:
                group['locators'].append({'csv_line':line,'forward_id':windows[index]['forward_id'],
                    'start_ns':begin,'duration_ns':duration,'device':row['Device_id'],
                    'stream':row['Stream ID'],'task':row['Task ID']})
            counts[index][key] += 1; selected += 1
            assert len(groups)<32, 'unexpected grouped ABI domain expansion'
    rows = []
    for key, group in groups.items():
        activation = [int(x) for x in group['abi']['Input Shapes'].strip('"').split(';')[0].split(',')]
        output = [int(x) for x in group['abi']['Output Shapes'].strip('"').split(',')]
        group.update(mean_us=group['sum_ns']/group['count']/1000,
                     routed_rows=activation[0], n=output[1], k=activation[1],
                     calls_per_forward={'min':min(c[key] for c in counts),
                         'max':max(c[key] for c in counts),'mean':group['count']/len(windows)})
        group['forward_geometry_counts'] = [
            {'query_rows':key[0],'tokens_per_sequence':key[1],'sequences':key[2],'kernel_calls':value}
            for key,value in group['forward_geometry_counts'].items()]
        rows.append(group)
    role = 'parameter_assignment' if card['case_id'] in TRAIN else (
        'fixed_126k_validation' if 'p126976' in card['case_id'] else '64k_nonassignment_control')
    result = {'case_id':card['case_id'], 'assignment_role':role, 'original_role':'unassigned',
        'source_identity':source['source_identity'], 'source_hash_verified':True,
        'extraction_identity':extraction, 'comparison_identity':card['evidence'],
        'geometry':card['geometry'], 'forwards':len(windows), 'groups':rows,
        'scan':{'rows':scanned,'selected_grouped_kernels':selected,
            'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    signal.alarm(0)
    return result


if __name__=='__main__':
    cards = [r for r in json.loads(REPORT.read_text())['cases']
             if r['case_id'].startswith('a5-qwen35-tp4-longctx-')]
    assert len(cards)==8
    results=[]
    for card in cards:
        result=scan(card); results.append(result)
        (HERE/'native.json').write_text(json.dumps({'schema':'a5-ep1-grouped-native/1',
            'boundary':'inclusive native kernel entirely inside retained q4 complete Body windows',
            'split':'Retrospective source split; no Body total used; raw roles remain unassigned.',
            'limitations':['group_list values and native binary identity unobserved',
                'means pool correlated layer calls; not independent hardware trials'],
            'cases':results},indent=2)+'\n')
        print(card['case_id'],[(r['routed_rows'],r['n'],r['k'],round(r['mean_us'],4))
                               for r in result['groups']],flush=True)
