"""Bounded native FAI census for a separately declared response experiment.

[Role] Preserve complete native-call ABI and PMU means over long-context runs.
[Invariants] No Body or serving time enters this table. Original roles remain
  unassigned; <=96K runs and 126K runs are separate analysis populations.
  Context is the experiment's prompt declaration, not an observed seq_lens
  value. Captured pages, aliasing and source tiling remain unidentified.
[Timeouts & Locks] Sequential streaming; at most 2048 durations per ABI group.
[Side-Effect Policy] Reads external CSVs; writes one compact census beside this file.
"""
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


BASE = Path('/Volumes/data/mindie-sim-data/a3/heldout_engine_20261030')
FOLDERS = ('qwen35_35b_tp4_128k_20261030', 'qwen36_27b_tp2_128k_20261030',
           'qwen38_27b_tp4_128k_20261030')
ABI = ('Type', 'Input Shapes', 'Input Data Types', 'Input Formats',
       'Output Shapes', 'Output Data Types', 'Output Formats',
       'Block Num', 'Mix Block Num', 'Accelerator Core')
PMU = ('aicore_time(us)', 'aic_mac_time(us)', 'aic_mte1_time(us)',
       'aic_mte2_time(us)', 'aiv_time(us)', 'aiv_vec_time(us)')


def shapes(value):
    return tuple(tuple(map(int, part.split(','))) if part else ()
                 for part in value.strip('"').split(';'))


def collect():
    records = []
    for folder in FOLDERS:
        for path in sorted((BASE / folder / 'prof').glob('*/kernel_details.csv')):
            tag = path.parent.name
            short = re.fullmatch(r'decode_c(\d+)_p(\d+)k', tag)
            long = re.fullmatch(r'decode_128k_in126k_out3k_c(\d+)', tag)
            if not short and not long:
                continue
            batch = int((short or long)[1])
            declaration = BASE / folder / 'raw' / f'cellset_{tag}.json'
            declaration_bytes = declaration.read_bytes()
            design = json.loads(declaration_bytes)['design']
            context = design['prompt_tokens']
            groups = {}
            with path.open() as source:
                for line, row in enumerate(csv.DictReader(source), 2):
                    if row['Type'] != 'FusedInferAttentionScore':
                        continue
                    ss = shapes(row['Input Shapes'])
                    if (len(ss) != 7 or len(ss[0]) != 3 or len(ss[4]) != 1
                            or ss[0][0] != 4 * batch or ss[4] != (batch,)):
                        continue
                    identity = tuple(row[field] for field in ABI)
                    group = groups.setdefault(identity, dict(
                        count=0, total_us=0., samples=[], pmu=defaultdict(float),
                        pmu_counts=defaultdict(int), first_line=line))
                    duration = float(row['Duration(us)'])
                    group['count'] += 1
                    group['total_us'] += duration
                    if len(group['samples']) < 2048:
                        group['samples'].append(duration)
                    for key in PMU:
                        try:
                            value = float(row.get(key, ''))
                        except ValueError:
                            continue
                        group['pmu'][key] += value
                        group['pmu_counts'][key] += 1
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1 << 20), b''):
                    digest.update(block)
            for identity, group in groups.items():
                record = dict(source=str(path), sha256=digest.hexdigest(),
                              first_line=group['first_line'], native_abi=dict(zip(ABI, identity)),
                              declared_context=context, active_batch=batch, query_tokens=4,
                              experiment_declaration=dict(path=str(declaration),
                                  sha256=hashlib.sha256(declaration_bytes).hexdigest(), design=design),
                              original_split='unassigned',
                              analysis_role=('independent_126k_validation' if long
                                             else 'candidate_operator_calibration'),
                              event_count=group['count'], mean_us=group['total_us']/group['count'],
                              median_us=statistics.median(group['samples']),
                              retained_samples=len(group['samples']),
                              median_basis='first_at_most_2048_events_per_native_ABI',
                              pmu_mean_us={k: v/group['pmu_counts'][k]
                                           for k, v in group['pmu'].items()})
                records.append(record)
            print(folder, tag, len(groups), flush=True)
    return records


if __name__ == '__main__':
    output = dict(boundary='native device kernel call; q4 only',
                  context_uncertainty='declared prompt; generated suffix and seq_lens values not captured',
                  records=collect())
    Path(__file__).with_name('native-census.json').write_text(json.dumps(output, indent=2)+'\n')
