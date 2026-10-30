"""Audit the conditional A3 FAI read bound against retained native evidence.

[Role] Preserve source assumptions, native ABI, PMU conversion and page capacity.
[Invariants] No observed duration sets bandwidth, cache hits or branch selection.
  PMU conversion frequency is distinct from an independent DVFS measurement.
[Timeouts & Locks] Stream one capture; at most 32 ABI groups and 4096 rows/group.
  Four local Git reads have independent ten-second subprocess limits.
[Side-Effect Policy] Write only the requested report; no runtime or parameters.
"""
import argparse
import subprocess
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture-root', type=Path, default=Path('/Volumes/data/mindie-sim-data/a3/heldout_engine_20261030/qwen36_27b_tp2_128k_20261030'))
parser.add_argument('--source-repository', type=Path, default=Path('/Users/maoxx241/code/refs/ops-transformer'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
BASE = args.capture_root
REVISION = 'a3925fd49cd256ad478d2faf983bf2961b8ae2c8'
SOURCE_FILES = (
    'op_host/fused_infer_attention_score_tiling.cpp',
    'op_host/flash_attention_infer_tiling.h',
    'op_kernel/arch22/flash_attention_regular.h',
    'op_kernel/arch22/fia_block_cube_nonquant_gqa.h',
)
source_identities = []
for relative in SOURCE_FILES:
    path = 'attention/fused_infer_attention_score/' + relative
    data = subprocess.check_output(['git', '-C', str(args.source_repository),
                                    'show', REVISION + ':' + path], timeout=10)
    source_identities.append({'path': path, 'sha256': hashlib.sha256(data).hexdigest(),
                              'size_bytes': len(data)})
CSV = BASE / 'prof/decode_128k_in126k_out3k_c12/kernel_details.csv'
FIELDS = ('Input Shapes', 'Input Data Types', 'Input Formats', 'Output Shapes',
          'Output Data Types', 'Output Formats', 'Block Num', 'Mix Block Num',
          'Accelerator Core', 'Name')
groups = defaultdict(list)
with CSV.open(newline='') as source:
    for line, row in enumerate(csv.DictReader(source), 2):
        if row.get('Type') != 'FusedInferAttentionScore':
            continue
        key = tuple(row.get(field, '') for field in FIELDS)
        if key not in groups and len(groups) >= 32:
            raise RuntimeError('bounded ABI group cap exceeded')
        if len(groups[key]) >= 4096:
            raise RuntimeError('bounded sample cap exceeded')
        groups[key].append((line, row))
summaries = []
for key, items in groups.items():
    durations = [float(row['Duration(us)']) for _, row in items]
    clocks = {}
    for label, cycles, duration, blocks in (
        ('aic', 'aic_total_cycles', 'aicore_time(us)', 'Block Num'),
        ('aiv', 'aiv_total_cycles', 'aiv_time(us)', 'Mix Block Num'),
    ):
        values = [float(row[cycles]) / float(row[duration]) / int(row[blocks])
                  for _, row in items if row[duration] not in ('N/A', '')
                  and float(row[duration]) > 0 and int(row[blocks]) > 0]
        clocks[label] = {'samples': len(values),
                         'min': min(values) if values else None,
                         'max': max(values) if values else None}
    summaries.append({'abi': dict(zip(FIELDS, key)), 'calls': len(items),
                      'first_csv_line': items[0][0], 'first_row': items[0][1],
                      'duration_us': {'mean': statistics.mean(durations),
                                      'median': statistics.median(durations),
                                      'min': min(durations), 'max': max(durations)},
                      'pmu_clock_conversion_MHz': clocks})
digest = hashlib.sha256()
with CSV.open('rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(chunk)
raw_path = BASE / 'raw/cellset_decode_128k_in126k_out3k_c12.json'
raw = json.loads(raw_path.read_text())
first = summaries[0]['first_row']
kv_bytes = (126000 + 4) * 2 * 256 * 2
q_bytes = 4 * 6 * 256 * 2
owner_bytes = 2 * (kv_bytes + q_bytes)
result = {
    'schema': 'a3-qwen36-fai-cold-read-audit/1',
    'scope': 'Read-only conditional consistency audit. No rate, cache hit or source dispatch is fitted.',
    'source': {'csv_path': str(CSV), 'sha256': digest.hexdigest(),
               'size_bytes': CSV.stat().st_size, 'raw_path': str(raw_path),
               'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               'log_path': str(BASE / 'logs/vllm.log')},
    'raw_design': raw['design'], 'raw_prefix_metrics': raw['cells'][0]['prefix'],
    'native_groups': summaries,
    'clock_interpretation': 'Cycles / mean per-core time / launched core count shows the report conversion uses 1650 MHz. It is not an independent measurement of real DVFS.',
    'conditional_cold_work': {
        'B': 12, 'q': 4, 'query_heads': 12, 'kv_heads': 2, 'head_dim': 256,
        'context': 126000, 'query_head_tasks': 24, 'owners': 20,
        'tasks_per_owner_max': 2, 'kv_bytes_per_task': kv_bytes,
        'query_bytes_per_task': q_bytes, 'max_owner_mte2_bytes': owner_bytes,
        'nominal_owner_bytes_per_s': 32 * 1650e6,
        'max_owner_lower_ms': owner_bytes / (32 * 1650e6) * 1000,
        'total_read_bytes': 24 * (kv_bytes + q_bytes),
        'aggregate_read_bound_ms': 24 * (kv_bytes + q_bytes) / (32 * 20 * 1650e6) * 1000,
        'apparent_read_B_per_cycle_from_first_mean_mte2': 24 * (kv_bytes + q_bytes) / (float(first['aic_mte2_time(us)']) * 1e-6 * 20 * 1650e6),
        'nominal_rate_status': 'platform AICoreMemoryRates ddr_read_rate mapping, not measured per-core pipe ceiling',
    },
    'conditional_page_capacity': {
        'allocated_cache_pages': 8806, 'page_rows': 128, 'token_slots': 8806 * 128,
        'minimum_page_references_if_all_12_contexts_126K': 12 * ((126000 + 127) // 128),
        'page_reference_excess': 12 * ((126000 + 127) // 128) - 8806,
        'condition': 'All 12 sequences active with actual KV length >=126000 in this call; captured K/V tensors are complete page pools. Actual per-call seq_lens and blocktable values are not retained, so this is a consistency constraint only.',
        'not_identified': ['page addresses', 'which references alias', 'shared-prefix residency', 'L2 hit fraction', 'measured cold-path bandwidth'],
    },
    'source_conditions': {
        'repository': str(args.source_repository),
        'revision': REVISION, 'file_identities': source_identities,
        'fai_route': 'op_host/fused_infer_attention_score_tiling.cpp:1689-1735 requires scalar TND, supported sparse mode, innerPrecise0 for GQA; PA D256/3D keys supported at 1650-1659. Scalar attrs and capture binary identity are not retained.',
        'flash_selector': 'same file 1520-1573: tasks 24 >0.8*cores 20=16, so flashDecodeFlag false; q=4 also rules out the high-batch q=1 specialized decoder.',
        'regular_tasks': 'op_host/flash_attention_infer_tiling.h:373-399 and op_kernel/arch22/flash_attention_regular.h:306-365 yield 24 request/KV-head tasks and stride 20 ownership.',
        'cache_hint_difference': 'No explicit L2 disable in flash_attention_regular.h or its attn_infra headers. Newer fia_block_cube_nonquant_gqa.h:482/497 explicitly sets CACHE_MODE_DISABLE. This does not establish native cache hits.',
        'launch_occupancy_consistency': {
            'uniform_regular_expected_mean_over_max': 24 / (2 * 20),
            'first_aicore_mean_over_duration': float(first['aicore_time(us)']) / float(first['Duration(us)']),
            'interpretation': 'Approximately 0.6 is compatible with regular 24-task/20-owner imbalance, but is not captured dispatch identity.',
        },
    },
    'conclusion': 'No demonstrated clock mismatch or proven cache hit. The span contradicts the conjunction of declared independent cold per-owner reads at 32 B/cycle and selected regular ownership for this actual call. Source regular ownership is internally consistent; cold bytes, effective read path and exact binary branch remain conditional.',
}
output = args.output
output.write_text(json.dumps(result, indent=2) + '\n')
print(output)
for row in summaries:
    print(row['calls'], row['duration_us'], row['pmu_clock_conversion_MHz'])
