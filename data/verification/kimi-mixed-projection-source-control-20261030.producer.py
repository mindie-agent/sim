"""Bounded local evidence for the explicit KDA mixed projection caller.

[Role] Check retained source shapes and execute two first-layer controls.
[Invariants] Native durations never set simulation parameters. Source prefixes
  are streamed; physical captured rows remain distinct from active requests.
[Timeouts & Locks] Finite 52-source census, two small one-layer Engine controls.
[Side-Effect Policy] Reads local evidence and writes the selected JSON report.
"""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case

PENDING = Path('/Volumes/data/mindie-sim-data/acceptance-archives-20261030/pending/long-context-decode')
SOURCE_FIELDS = ('Type', 'Input Shapes', 'Input Data Types', 'Input Formats',
                 'Output Shapes', 'Output Data Types', 'Block Num', 'Mix Block Num',
                 'Task ID', 'Stream ID', 'Device_id')


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(),
            'size_bytes': path.stat().st_size}


def shapes(row):
    return [[int(x) for x in part.split(',') if x]
            for part in row['Input Shapes'].strip('"').split(';')]


def census():
    records = []
    for filename, key, tp, selected in (
        ('kimi-tp8-longctx-prepared-entries.json', 'cases', 8, True),
        ('kimi-tp16-128k-prepared-entries.json', 'cases', 16, True),
        ('kimi-a5-tp8-128k-prepared-entries.json', 'entries', 8, False),
    ):
        prepared = json.loads((PENDING / filename).read_text())
        for entry in prepared[key]:
            extracted = json.loads(Path(entry['extraction_path']).read_text())
            forward = next(iter(extracted['forwards']), None)
            if forward:
                body = next(stage for stage in forward['stages'] if stage['kind'] == 'target_body')
                first = body['evidence']['first_locator']['csv_line']
                scope = 'first complete source KDA block, not all-forward ABI census'
            else:
                first = 2
                scope = 'bounded first native KDA block; prepared extraction has no complete forward'
            rows = []
            with Path(entry['source']['path']).open() as stream:
                for line, row in enumerate(csv.DictReader(stream), 2):
                    if line < first:
                        continue
                    rows.append({'csv_line': line, **{key: row[key] for key in SOURCE_FIELDS}})
                    if row['Type'] == 'RecurrentKda':
                        break
                    if line >= first + 600:
                        raise RuntimeError(f'first KDA source block exceeds bound: {entry["id"]}')
            heads = 96 // tp
            bfg = [row for row in rows if row['Type'] == 'MatMulV2'
                   and shapes(row)[1:] == [[heads * 257, 7168]]]
            qkv = [row for row in rows if row['Type'] == 'QuantBatchMatmulV3'
                   and shapes(row)[1:3] == [[heads * 12, 448, 16, 32], [heads * 384]]]
            if selected:
                assert len(bfg) == len(qkv) == 1, entry['id']
                physical = shapes(bfg[0])[0][0]
                assert shapes(bfg[0])[0] == shapes(qkv[0])[0] == [physical, 7168]
                assert bfg[0]['Input Data Types'] == 'DT_BF16;DT_BF16'
                assert qkv[0]['Input Data Types'] == 'INT8;INT8;DT_BF16;FLOAT'
                if forward:
                    assert physical == forward['geometry']['query_rows'], entry['id']
            else:
                physical = forward['geometry']['query_rows'] if forward else None
                assert not bfg and not qkv, entry['id']
            records.append({
                'case_id': entry['case_id'], 'tp': tp, 'source': entry['source'],
                'extraction': entry['extraction_identity'],
                'extraction_status': extracted['status'],
                'geometry': forward['geometry'] if forward else None,
                'physical_projection_rows': physical,
                'scope': scope,
                'source_csv_lines': [rows[0]['csv_line'], rows[-1]['csv_line']],
                'mixed_projection_contract': 'matched' if selected else 'excluded_BF16_deployment',
                'projection_rows': bfg + qkv if selected else [row for row in rows
                    if row['Type'] in {'MatMulV2', 'MatMulV3', 'QuantBatchMatmulV3'}],
            })
    return records


def layer_control(tp, batch):
    path = ROOT / f'examples/cases/kimi-mixed-projection-20261030/a3-kimi4-tp{tp}.json'
    raw = json.loads(path.read_text())
    output = {'tp': tp, 'batch_slots': batch, 'case': identity(path), 'variants': {}}
    for variant in ('separate_semantic_projections', 'source_mixed_projection'):
        value = deepcopy(raw)
        if variant == 'separate_semantic_projections':
            value['runtime']['kernels']['lowerings'].pop('kda_input_projection')
        parsed = parse_case(value)
        full = target_graphs(parsed, phase='decode', tokens=1, sequences=batch,
                             context=126000, scope='body', resource_fusion=False)[0]
        stop = next(index for index, op in enumerate(full) if op.layer > 0)
        ops = full[:stop]
        result = graph_costs(parsed, ops, isolated=False, trace_limit=0,
                             external_inputs={'analysis_input': 0})
        projection = [op for op in ops if op.recipe.startswith(('linear_q', 'linear_k',
                      'linear_v', 'linear_b', 'linear_decay', 'linear_gate'))]
        rows = {row['op_id']: row for row in result['timeline']}
        terms = layer_weight_terms(parsed.model.layer_plans[0], parsed.model)
        output['variants'][variant] = {
            'first_layer_span_ms': result['duration_ms'], 'operator_count': len(ops),
            'execution': result['execution'],
            'resident_layer_weights_bytes': sum(term_bytes(term, parsed.model.quant,
                parsed.runtime.parallel) for term in terms),
            'derived_composed_F_bytes': sum(term_bytes(term, parsed.model.quant,
                parsed.runtime.parallel) for term in terms if term.name == 'kda_composed_f'),
            'projection_ops': [{
                'recipe': op.recipe, 'shape': op.local_shape, 'dtype': op.dtype,
                'flops': op.flops, 'mte2_bytes': op.mte2_bytes, 'mte3_bytes': op.mte3_bytes,
                'deps': op.deps, 'stream': op.stream,
                'ready_ms': rows[op.op_id]['ready_ms'],
                'complete_ms': rows[op.op_id]['complete_ms'],
            } for op in projection],
        }
    return output


def main():
    started = time.monotonic()
    records = census()
    controls = [layer_control(8, 2), layer_control(16, 1)]
    report = {
        'schema': 'mindie-kda-mixed-projection-source-control/1',
        'scope': 'A3 source-compatible mixed projection; local structural and Engine evidence',
        'timing_acceptance': 'not_established; no captured duration is fitted or scored',
        'source_binary_identity': 'unverified; retained source revision and ABI are explicit',
        'producer': identity(__file__),
        'runtime_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                   text=True, timeout=3).strip(),
        'runtime_modules': [identity(ROOT / path) for path in (
            'mindie_sim/prediction/family_kda_projection.py', 'mindie_sim/prediction/recurrent.py',
            'mindie_sim/prediction/transforms.py', 'mindie_sim/catalog/weights.py')],
        'census_summary': dict(Counter(row['mixed_projection_contract'] for row in records)),
        'census': records, 'first_layer_controls': controls,
        'elapsed_seconds': time.monotonic() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
        'kimi-mixed-projection-source-control-20261030.json')
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': identity(target), 'census': report['census_summary'],
                      'elapsed_seconds': report['elapsed_seconds'],
                      'peak_rss_bytes': report['peak_rss_bytes']}))


if __name__ == '__main__':
    main()
